"""Shared view-layer helpers for the Evaluate/Draft/About tabs.

Since the split into focused modules, this file re-exports their public
surface so tab modules (and tests) keep one import point:

- ``app.views.uploads``  — upload size gate + extraction caching
- ``app/views/records``  — record-source widget (Upload/Fetch/VA.gov/Local)
- ``app/views/usage``    — usage watchdog, credit rates, usage summary

plus the genuinely shared bits that live here: correlation ids, the LLM
handle, the shutdown gate, progress widgets, and audit metadata.
"""
from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import Any

import streamlit as st

from .. import preflight
from .. import telemetry
from ..error_report import (  # noqa: F401 - re-exported for tab modules and tests
    ensure_request_id,
    format_error_for_user,
    reference_suffix,
    report_failure,
)
from ..logging_config import get_logger, get_request_id, new_request_id, set_request_id
from ..llm import LLMClient, LLMError
from ..run_log import run_log_event
from ..pipeline_guard import check_pipeline_cancelled
from ..shutdown import is_shutting_down
from .records import (  # noqa: F401
    is_local_run,
    records_uploader,
    remember_source_records,
    render_record_search,
)
from .uploads import (  # noqa: F401
    check_upload_limits,
    extract_uploads,
    render_record_volume_warning,
)
from .usage import (  # noqa: F401
    effective_credit_rates,
    load_usage_history,
    record_watchdog_run,
    render_usage_summary,
    save_usage_history,
)

logger = get_logger("app.views.shared")

REQUEST_ID_KEY = "va_lse_request_id"

# Feature: Condition-Specific Templates
FEATURE_ID = "02f0935a-ee5e-4083-88a2-10e11753ccc9"  # condition-specific-templates


# ------------------------------------------------------------- correlation ids
def get_or_create_request_id() -> str:
    """Return the active run's correlation id, minting one if needed."""
    rid_raw: Any = st.session_state.get(REQUEST_ID_KEY, "")
    rid: str = str(rid_raw) if isinstance(rid_raw, str) and rid_raw else ""
    if rid:
        # Ensure ContextVar mirrors session state (Streamlit reruns may reset context).
        set_request_id(rid)
        return rid
    rid = new_request_id()
    st.session_state[REQUEST_ID_KEY] = rid
    set_request_id(rid)
    return rid


def new_run_request_id() -> str:
    """Mint a fresh correlation id for a new Evaluate/Draft run."""
    rid = new_request_id()
    st.session_state[REQUEST_ID_KEY] = rid
    set_request_id(rid)
    return rid


# Fallback locations tried (in order) when no file logging is configured, so
# ``req_…`` references stay correlatable even without VA_LSE_LOG_DIR set.
_RUN_LOG_FALLBACK_DIRS = ("logs", "outputs")


def log_unhandled_render_error(exc: Exception, rid: str, phase: str = "app") -> None:
    """Best-effort persistence for an error outside the run flow.

    Mirrors the run-log contract: a ``rejected`` event keyed by the same
    reference the UI shows, plus a short traceback, so no failure is ever
    "unexpected" in the logs — even when it happens outside
    ``_run_draft_flow``/``_run_evaluation_flow`` (e.g. results rendering).
    """
    run_log_event(
        "app", "rejected", request_id=rid, phase=phase,
        error=f"{type(exc).__name__}: {exc}", error_class=type(exc).__name__,
        traceback=traceback.format_exc(limit=8),
    )
    # File-logging fallback: if no VA_LSE_LOG_DIR is configured the structured
    # logger goes to stdout only; duplicate the digest to a stable file so the
    # reference can always be resolved on this machine.
    try:
        import os
        from pathlib import Path

        if not os.getenv("VA_LSE_LOG_DIR", "").strip() and not os.getenv("VA_LSE_RUN_LOG_DIR", "").strip():
            for cand in _RUN_LOG_FALLBACK_DIRS:
                try:
                    p = Path(cand)
                    p.mkdir(parents=True, exist_ok=True)
                    with open(p / "unhandled_errors.log", "a", encoding="utf-8") as fh:
                        fh.write(
                            f"{datetime.now(timezone.utc).isoformat()} {rid} "
                            f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}\n"
                        )
                    break
                except Exception:  # noqa: BLE001 - try next candidate
                    continue
    except Exception:  # noqa: BLE001 - best-effort only
        pass


# ---------------------------------------------------------------- LLM handle
def session_settings() -> Any:
    """The settings a run will actually use, on-screen API key included.

    One definition, because two callers must agree on it: :func:`get_llm` builds the
    client from these, and the endpoint preflight judges whether that client can
    work. A preflight that checked the *saved* key while the run used the typed one
    would clear a configuration that then fails.
    """
    try:
        settings = st.session_state.settings
    except AttributeError:
        # Session state exists but settings not initialized yet (edge reruns).
        from ..config import load_settings

        settings = st.session_state.settings = load_settings()
    settings.api_key = st.session_state.get("api_key_input", settings.api_key).strip()
    return settings


def get_llm() -> LLMClient | None:
    """Build the session's LLMClient, or show the reason and return None."""
    rid = st.session_state.get(REQUEST_ID_KEY, "") or get_request_id() or "-"
    settings = session_settings()
    if not settings.configured:
        logger.warning(
            "LLM not configured — missing API key",
            extra={"request_id": rid, "phase": "llm_config", "status": "error"},
        )
        st.error("Enter your LLM API key in the sidebar before running.")
        return None
    try:
        return LLMClient(settings)
    except LLMError as exc:
        logger.error(
            "LLM client init failed: %s",
            exc,
            exc_info=exc,
            extra={
                "request_id": rid,
                "phase": "llm_config",
                "status": "error",
                "error_class": type(exc).__name__,
            },
        )
        st.error(report_failure(str(exc), phase="llm_config", exc=exc))
        return None


def check_shutdown_gate(action: str) -> bool:
    """Return True when a new run may start; otherwise show why not.

    ``action`` is "evaluation" or "draft" — used only in the user message.
    """
    if is_shutting_down():
        st.error(
            f"The app is shutting down to finish a deployment or restart. "
            f"No new {action} runs can start right now — please try again in a moment."
        )
        return False
    return True


# ------------------------------------------------------- endpoint preflight
def endpoint_waiver_key(action: str, signature: str) -> str:
    """Session key for "I know this check is wrong for my endpoint".

    Keyed by the configuration's signature, not just the tab: a waiver is a
    statement about one endpoint + key + model set, so changing any of them has to
    ask again. A per-tab key would silently carry a waiver onto a configuration the
    user never saw a complaint about.
    """
    return f"endpoint_preflight_waiver_{action}_{signature}"


def _endpoint_block_key(action: str) -> str:
    return f"endpoint_preflight_block_{action}"


def _endpoint_age_key(action: str) -> str:
    """Session key for how old a *reused* block's verdict was when it was stored."""
    return _endpoint_block_key(action) + "_age"


def _clear_endpoint_block(action: str) -> None:
    st.session_state.pop(_endpoint_block_key(action), None)
    st.session_state.pop(_endpoint_block_key(action) + "_sig", None)
    st.session_state.pop(_endpoint_age_key(action), None)


def _age_label(seconds: float) -> str:
    """A short human age — ``40 seconds`` / ``3 minutes`` — for the reuse notes."""
    if seconds < 90:
        value, unit = max(1, round(seconds)), "second"
    else:
        value, unit = max(1, round(seconds / 60)), "minute"
    return f"{value} {unit}{'' if value == 1 else 's'}"


def check_endpoint_gate(action: str, *, log_action: str, request_id: str = "") -> bool:
    """Return True when the endpoint can serve the configured models.

    Two cheap requests per attempt — a model listing and one short chat call per
    configured model — and only when a run is actually being attempted, so ordinary
    reruns cost nothing. A verdict this gate or **Test connection** just produced for
    this same configuration is reused instead (see
    :func:`app.preflight.reusable_verdict`), so a healthy configuration is not probed
    twice in a row. A block is stored so
    :func:`render_endpoint_preflight_notice` can keep it on screen with the waiver
    beside it — a user must be able to overrule a check that is wrong about their
    endpoint, because refusing to start a working run is worse than the failure the
    check prevents. A reused check is stated where the run starts — with the age of
    the check behind it — so a skipped probe is never silent.
    """
    settings = session_settings()
    signature = preflight.signature(settings)
    # **Test connection** and this gate's previous attempt run the same check on the
    # same configuration; their verdict is evidence until it goes stale, and reusing it
    # keeps a healthy configuration from being probed once per run.
    verdict = preflight.reusable_verdict(st.session_state, settings)
    reused = verdict is not None
    age = preflight.verdict_age(st.session_state, settings) if reused else None
    if verdict is None:
        verdict = preflight.check_endpoint(settings)
        # Keep it for the next attempt on this configuration. Only a probe refreshes
        # this: a reused verdict ages from when it was actually gathered, so a session
        # that keeps running cannot postpone the next real check forever.
        preflight.remember_verdict(st.session_state, settings, verdict)
    logger.info(
        "endpoint preflight action=%s kind=%s status=%s missing=%s reused=%s age_s=%s",
        action,
        verdict.kind,
        verdict.status,
        list(verdict.missing),
        reused,
        None if age is None else round(age, 1),
        extra={"request_id": request_id or "-", "phase": "endpoint_preflight", "status": verdict.kind},
    )
    if not verdict.blocks:
        if age is not None:
            st.caption(
                f"Endpoint preflight: reused the check from {_age_label(age)} ago — "
                "no request was sent."
            )
        _clear_endpoint_block(action)
        return True

    waived = bool(st.session_state.get(endpoint_waiver_key(action, signature), False))
    if waived:
        logger.warning(
            "endpoint preflight overruled by the user action=%s status=%s missing=%s",
            action,
            verdict.status,
            list(verdict.missing),
            extra={"request_id": request_id or "-", "phase": "endpoint_preflight", "status": "waived"},
        )
        _clear_endpoint_block(action)
        return True

    st.session_state[_endpoint_block_key(action)] = verdict
    st.session_state[_endpoint_block_key(action) + "_sig"] = signature
    if age is not None:
        st.session_state[_endpoint_age_key(action)] = age
    else:
        st.session_state.pop(_endpoint_age_key(action), None)
    run_log_event(
        log_action,
        "rejected",
        request_id=request_id,
        error=verdict.headline,
        reason="endpoint_preflight",
    )
    # Put the notice above the button the user just pressed. This raises in real
    # Streamlit; the caller returns on False either way, so nothing depends on it.
    st.rerun()
    return False


def render_endpoint_preflight_notice(action: str) -> None:
    """Show a stored endpoint block above the run button, waiver included.

    Rendered from session state rather than probed here: this runs on every rerun of
    the tab, and a network request per rerun is not a price a preflight may charge.
    """
    verdict = preflight.verdict_from_session(st.session_state.get(_endpoint_block_key(action)))
    if verdict is None or not verdict.blocks:
        return
    signature = preflight.signature(session_settings())
    if st.session_state.get(_endpoint_block_key(action) + "_sig") != signature:
        # The endpoint, key, or models changed since the check, so this block is about
        # a configuration that is no longer in force and must not be shown.
        _clear_endpoint_block(action)
        return
    st.error(f"⛔ Run not started — {verdict.headline}\n\n{verdict.fix}")
    age = st.session_state.get(_endpoint_age_key(action))
    if isinstance(age, (int, float)):
        st.caption(
            f"Reused the check from {_age_label(float(age))} ago — no request was sent "
            "when the run was attempted."
        )
    with st.expander("The check can be wrong — start the run anyway"):
        st.caption(
            "Some OpenAI-compatible servers answer `/models` differently than they serve "
            "completions, and a provider's catalog can lag what it actually serves. Tick this "
            "to skip the check for this endpoint, key and model set; changing any of them asks "
            "you again."
        )
        st.checkbox("Ignore the endpoint check and run anyway", key=endpoint_waiver_key(action, signature))


# ------------------------------------------------------------- progress + audit
def progress_widgets(llm: LLMClient | None = None, *, request_id: str | None = None) -> tuple[Any, Any]:
    """Progress bar whose caption appends a live estimated-usage line."""
    bar = st.progress(0.0, text="Starting…")
    rid = request_id or get_request_id() or "-"

    def update(frac: float, msg: str) -> None:
        check_pipeline_cancelled()
        logger.debug(
            "progress %.0f%% — %s",
            frac * 100,
            msg,
            extra={"request_id": rid, "phase": "progress", "status": "ok"},
        )
        text = msg
        if llm is not None:
            text += llm.usage.live_line()
        bar.progress(min(max(frac, 0.0), 1.0), text=text)

    return bar, update


def audit_record_meta(slot: str, records: list[Any]) -> tuple[list[str], int, int]:
    """Return (record_sources, file_count, page_count) for audit events."""
    try:
        store_any: Any = st.session_state.get(f"source_records_{slot}", {})
        if isinstance(store_any, dict) and store_any:
            sources = [str(k) for k in store_any.keys() if str(k).strip()]
        else:
            sources = []
    except Exception:  # noqa: BLE001
        sources = []
    # Fallback label when store is empty but records exist (e.g. direct upload in tests).
    if not sources and records:
        sources = ["Upload"]
    files = len(records)
    try:
        pages = sum(len(getattr(d, "pages", [])) for d in records)
    except Exception:  # noqa: BLE001
        pages = 0
    return sources, files, pages


def audit_condition_for_slot(slot: str) -> str:
    """Best-effort claimed-condition label for the audit (no PII)."""
    try:
        selected_any: Any = st.session_state.get(f"selected_conditions_{slot}", [])
        if isinstance(selected_any, list) and selected_any:
            names: list[str] = []
            for item in selected_any:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    names.append(str(item[1]).strip())
                elif isinstance(item, str):
                    names.append(item.strip())
            names = [n for n in names if n]
            if names:
                return ", ".join(names)[:120]
    except Exception:  # noqa: BLE001
        pass
    return ""


def render_condition_selector_for_slot(slot: str) -> None:
    """Fire the selector impression once and render the condition selector."""
    from ..condition_selector import render_condition_selector

    render_condition_selector(slot, FEATURE_ID)
