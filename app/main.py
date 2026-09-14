"""VA Lay Statement Evaluator — Streamlit application entry point.

Run with:
    streamlit run app/main.py
"""
from __future__ import annotations

import logging
import time

import streamlit as st

from . import config
from .condition_selector import render_condition_selector
from .config import DEFAULT_BASE_URL, load_settings
from .documents import (
    DRAFT_INTERNAL_MAX_CHARS,
    EVALUATE_INTERNAL_MAX_CHARS,
    ExtractionError,
    MAX_OBSERVATIONS_CHARS,
    MAX_STATEMENT_CHARS,
    extract_document,
    extract_uploaded_documents,
    records_from_local_path,
)
from .draft import grounding_markdown, run_draft
from .evaluate import DIMENSION_LABELS, run_evaluation
from .fetch_client import FetchClient, FetchSandboxError
from .llm import LLMClient, LLMError
from .logging_config import (
    configure_logging,
    get_request_id,
    get_logger,
    new_request_id,
    set_request_id,
)
from . import telemetry
from . import va_gov_client
from . import watchdog
from .prompt_sanitize import validate_api_key, validate_model_name

logger = get_logger("app.main")
_REQUEST_ID_KEY = "va_lse_request_id"

# Feature: Condition-Specific Templates
FEATURE_ID = "02f0935a-ee5e-4083-88a2-10e11753ccc9"  # condition-specific-templates

st.set_page_config(
    page_title="VA Lay Statement Evaluator",
    page_icon="🎖️",
    layout="wide",
)

CLAIM_TYPES = [
    "Service connection (new claim)",
    "Increased rating (worsening condition)",
    "PTSD stressor corroboration",
    "TDIU / individual unemployability",
    "Continuity of symptoms since service",
]

RELATIONSHIPS = [
    "Spouse",
    "Family member",
    "Friend",
    "Coworker / supervisor",
    "Fellow service member",
    "Other",
]


def _check_streamlit_config_hardening() -> None:
    """Warn once per session if .streamlit/config.toml hardening is not active.

    Streamlit itself does not expose a clean API to confirm config source, so
    this checks the filesystem so a deployment missing the config file surfaces
    visibly instead of silently running with defaults.
    """
    if st.session_state.get("_streamlit_hardening_checked"):
        return
    st.session_state["_streamlit_hardening_checked"] = True
    try:
        from pathlib import Path as _P
        cfg = (_P(__file__).resolve().parent.parent / ".streamlit" / "config.toml")
        text = cfg.read_text(encoding="utf-8") if cfg.exists() else ""
        missing: list[str] = []
        if "enableXsrfProtection" not in text:
            missing.append("server.enableXsrfProtection")
        if 'toolbarMode' not in text:
            missing.append("client.toolbarMode")
        if "maxUploadSize" not in text:
            missing.append("server.maxUploadSize")
        if missing:
            msg = (
                "⚠️ Streamlit security hardening is not fully active (missing from .streamlit/config.toml: "
                + ", ".join(missing)
                + "). The app still works, but XSRF protection, toolbar hardening, and upload caps "
                "depend on that file — see README → Production hardening."
            )
            logger.warning("streamlit hardening incomplete: %s", ", ".join(missing), extra={"request_id": get_request_id() or "-", "phase": "app", "status": "warning"})
            st.warning(msg)
    except Exception:  # noqa: BLE001 - hardening check must never break the app
        pass


def _check_upload_limits(files) -> tuple[list, list[str]]:
    """Split uploaded files into accepted vs rejected by VA_LSE_MAX_UPLOAD_BYTES.

    Returns (accepted_files, rejection_messages). Accepted files also pass a
    total-batch cap (VA_LSE_MAX_TOTAL_UPLOAD_BYTES) — the largest files are
    dropped first until the batch fits, with one message per dropped file.
    """
    if not files:
        return [], []
    # Each Streamlit UploadedFile exposes .name and .size (bytes). Fall back to
    # len(getvalue()) for test fakes that only expose getvalue().
    def _size(f) -> int:
        try:
            return int(getattr(f, "size", None) or len(f.getvalue()))
        except Exception:  # noqa: BLE001
            return 0
    per_file_limit = config.MAX_UPLOAD_BYTES
    total_limit = config.MAX_TOTAL_UPLOAD_BYTES
    rejected_msgs: list[str] = []
    # Per-file check
    accepted: list = []
    for f in files:
        sz = _size(f)
        if sz > per_file_limit:
            rejected_msgs.append(
                f"✖️ {f.name}: {sz // 1_048_576} MB exceeds the per-file limit "
                f"({per_file_limit // 1_048_576} MB). Reduce or split this file."
            )
        else:
            accepted.append(f)
    # Batch total check — drop excess largest-first so the user's first files tend to survive.
    total = sum(_size(f) for f in accepted)
    if total > total_limit and accepted:
        accepted.sort(key=_size)  # smallest first; we keep small ones
        kept: list = []
        running = 0
        for f in accepted:
            if running + _size(f) <= total_limit:
                kept.append(f)
                running += _size(f)
            else:
                rejected_msgs.append(
                    f"✖️ {f.name}: batch total would exceed {total_limit // 1_048_576} MB — file skipped. "
                    "Remove some files or raise VA_LSE_MAX_TOTAL_UPLOAD_BYTES."
                )
        accepted = kept
    return accepted, rejected_msgs


# ------------------------------------------------------------------- settings
def _sidebar_settings() -> None:
    if "settings" not in st.session_state:
        st.session_state.settings = load_settings()
    settings = st.session_state.settings

    with st.sidebar:
        st.title("⚙️ LLM Settings")
        st.session_state.api_key_input = st.text_input(
            "API key",
            value=settings.api_key,
            type="password",
            help="Stored only in this browser session and used for LLM calls.",
        )
        st.session_state.base_url_input = st.text_input(
            "Base URL (OpenAI-compatible)", value=settings.base_url or DEFAULT_BASE_URL
        )
        col1, col2 = st.columns(2)
        st.session_state.model_main_input = col1.text_input(
            "Main model", value=settings.model_main, help="Analysis, scoring, drafting"
        )
        st.session_state.model_fast_input = col2.text_input(
            "Fast model",
            value=settings.model_fast,
            help="Bulk record digests — use a cheap model here to preserve "
            "QwenCloud Lite quota.",
        )
        st.divider()
        st.subheader("Fetch Sandbox")
        st.session_state.fetch_api_key_input = st.text_input(
            "Fetch API key",
            value=settings.fetch_api_key,
            type="password",
            help="Optional if the sandbox runs in relaxed auth mode.",
        )
        st.session_state.fetch_base_url_input = st.text_input(
            "Fetch base URL", value=settings.fetch_base_url
        )
        st.session_state.fetch_records_path_input = st.text_input(
            "Fetch records path",
            value=settings.fetch_records_path,
            help="GET path for records. Use {patient_id} where the selected ID belongs.",
        )
        if st.button("Apply settings"):
            api_key_val = st.session_state.api_key_input.strip()
            fetch_key_val = st.session_state.fetch_api_key_input.strip()
            model_main_val = st.session_state.model_main_input.strip()
            model_fast_val = st.session_state.model_fast_input.strip()
            errors: list[str] = []
            for label, val, validator in (
                ("API key", api_key_val, validate_api_key),
                ("Fetch API key", fetch_key_val, validate_api_key),
                ("Main model", model_main_val, validate_model_name),
                ("Fast model", model_fast_val, validate_model_name),
            ):
                msg = validator(val)
                if msg:
                    errors.append(f"{label}: {msg}")
            if errors:
                for msg in errors:
                    st.error(msg)
            else:
                settings.api_key = api_key_val
                settings.base_url = st.session_state.base_url_input.strip() or DEFAULT_BASE_URL
                settings.model_main = model_main_val
                settings.model_fast = model_fast_val
                settings.fetch_api_key = fetch_key_val
                settings.fetch_base_url = st.session_state.fetch_base_url_input.strip()
                settings.fetch_records_path = st.session_state.fetch_records_path_input.strip()
                st.rerun()

        _compat_model_warning(settings)

        st.divider()
        _credit_calibration_widget()
        st.divider()
        st.caption(
            "⚠️ Uploaded documents are sent to the configured LLM endpoint for analysis. "
            "Review privacy before uploading sensitive records."
        )
        st.caption(
            "This tool is an aid for drafting and reviewing lay statements. It is not "
            "legal, medical, or claims advice."
        )


def _get_or_create_request_id() -> str:
    """Return the active run's correlation id, minting one if needed."""
    rid = st.session_state.get(_REQUEST_ID_KEY, "")
    if rid:
        # Ensure ContextVar mirrors session state (Streamlit reruns may reset context).
        set_request_id(rid)
        return rid
    rid = new_request_id()
    st.session_state[_REQUEST_ID_KEY] = rid
    set_request_id(rid)
    return rid


def _new_run_request_id() -> str:
    """Mint a fresh correlation id for a new Evaluate/Draft run."""
    rid = new_request_id()
    st.session_state[_REQUEST_ID_KEY] = rid
    set_request_id(rid)
    return rid


def _get_llm() -> LLMClient | None:
    rid = st.session_state.get(_REQUEST_ID_KEY, "") or get_request_id() or "-"
    settings = st.session_state.settings
    settings.api_key = st.session_state.get("api_key_input", settings.api_key).strip()
    if not settings.configured:
        logger.warning("LLM not configured — missing API key", extra={"request_id": rid, "phase": "llm_config", "status": "error"})
        st.error("Enter your LLM API key in the sidebar before running.")
        return None
    try:
        return LLMClient(settings)
    except LLMError as exc:
        logger.error(
            "LLM client init failed: %s", exc, exc_info=exc,
            extra={"request_id": rid, "phase": "llm_config", "status": "error", "error_class": type(exc).__name__},
        )
        st.error(str(exc))
        return None


# ------------------------------------------------------------- shared uploads
def _format_error_for_user(exc: Exception, request_id: str) -> str:
    """User-facing error string that carries the correlation id without PII."""
    rid_suffix = f" (reference: {request_id})" if request_id and request_id != "-" else ""
    return f"{exc}{rid_suffix}"


def _extract_uploads(files, slot: str) -> list:
    """Extract text from uploaded files; cache results per file identity.

    Files that fail extraction (e.g. image-only PDFs) are reported as
    per-file warnings plus a loaded-vs-skipped summary, recomputed fresh each
    run so unreadable uploads never silently disappear and never linger once
    the bad file is removed or replaced.
    """
    documents = []
    to_extract = []
    for uploaded in files:
        cache_key = f"{slot}:{uploaded.name}:{uploaded.size}"
        if cache_key in st.session_state:
            documents.append(st.session_state[cache_key])
        else:
            to_extract.append(uploaded)

    new_docs, skipped = extract_uploaded_documents(to_extract)
    for doc in new_docs:
        # Cache each successful extraction by its (slot, name, size) identity.
        for uploaded in to_extract:
            if uploaded.name == doc.filename:
                st.session_state[f"{slot}:{uploaded.name}:{uploaded.size}"] = doc
                break
        documents.append(doc)
    # The uploader re-delivers files on every rerun, so warnings are recomputed
    # fresh each run: they persist while a bad file is still uploaded and clear
    # as soon as it is removed or replaced.
    _render_skip_summary(files, documents, skipped)
    for message in skipped:
        st.warning(message)
    return documents


def _render_skip_summary(files, documents: list, skipped: list[str]) -> None:
    """Show a loaded-vs-skipped summary under an uploader when any file failed."""
    if not files or not skipped:
        return
    total = len(files)
    loaded = len(documents)
    st.caption(
        f"Loaded {loaded} of {total} file(s) — {len(skipped)} skipped "
        "(listed below)."
    )


def _is_local_run() -> bool:
    """True when the app is served on the same machine as the browser.

    Used to gate reading records directly from the local filesystem: that is
    only safe for a local Streamlit run, never for a public deployment (where
    it would let anyone read server files). Fall back to an explicit opt-in
    env var for unusual local setups.
    """
    import os
    from urllib.parse import urlparse

    if os.getenv("VA_LSE_ALLOW_LOCAL_PATHS", "").strip() == "1":
        return True
    try:
        context = st.context
        host = (context.headers.get("Host") or "").split(":")[0].lower()
        if host in ("localhost", "127.0.0.1", "::1"):
            return True
        url_host = urlparse(context.url or "").hostname or ""
        return url_host in ("localhost", "127.0.0.1", "::1")
    except Exception:  # noqa: BLE001 - context may be unavailable in tests
        return False


def _track_selector_impression(slot: str) -> None:
    """Fire one impression event per Streamlit session per workflow slot."""
    flag_key = f"va_gov_selector_impression_{slot}"
    if st.session_state.get(flag_key):
        return
    st.session_state[flag_key] = True
    telemetry.track_impression(
        va_gov_client.FEATURE_ID,
        entry_point=slot,
        selector_rendered=True,
        workflow=slot,
    )


def _remember_source_records(slot: str, label: str, docs: list) -> None:
    """Track the most recent successful load per source for this slot/session.

    Used to merge VA.gov records with whatever other sources the user has
    already loaded in this workflow slot this session (FR4/FR6), without
    changing the existing single-select source radio into a multi-select.
    """
    if not docs:
        return
    store = st.session_state.setdefault(f"source_records_{slot}", {})
    store[label] = docs


def _records_uploader(slot: str) -> list:
    _track_selector_impression(slot)
    sources = ["Upload files", "Fetch Sandbox", "VA.gov"]
    if _is_local_run():
        sources.append("Local folder / file")
    source = st.radio(
        "Medical record source",
        sources,
        horizontal=True,
        key=f"records_source_{slot}",
    )
    if source == "Fetch Sandbox":
        return _fetch_records(slot)
    if source == "VA.gov":
        return _va_gov_records(slot)
    if source == "Local folder / file":
        return _local_records(slot)

    files = st.file_uploader(
        "Upload medical records (PDF, TXT, MD, DOCX — multiple allowed)",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        key=f"files_{slot}",
    )
    # Enforce upload size caps before extraction so the tight 50 MB default
    # surfaces as a clear message rather than a downstream failure.
    if files:
        accepted, rejections = _check_upload_limits(files)
        for msg in rejections:
            st.warning(msg)
        files = accepted if rejections else files
    documents = _extract_uploads(files, slot)
    total_pages = sum(len(d.pages) for d in documents)
    if documents and total_pages > config.MAX_RECORD_PAGES:
        st.error(
            f"Record set is {total_pages:,} pages, which exceeds the configured limit of "
            f"{config.MAX_RECORD_PAGES:,} pages. Remove some files or raise "
            "VA_LSE_MAX_RECORD_PAGES."
        )
        return []
    _remember_source_records(slot, "Upload", documents)
    return documents


def _local_records(slot: str) -> list:
    """Load record files straight from a path on the local machine."""
    st.caption(
        "Reads supported record files (.pdf/.txt/.md/.docx) directly from this "
        "machine's filesystem. Only available when the app runs locally."
    )
    path = st.text_input(
        "Folder or file path",
        key=f"local_path_{slot}",
        placeholder="e.g. ~/Desktop/ClaimRecords or ~/Desktop/records.pdf",
    )
    import_key = f"local_records_{slot}"
    skipped_key = f"local_records_skipped_{slot}"
    if st.button("Load records from path", key=f"local_load_{slot}"):
        st.session_state.pop(import_key, None)
        st.session_state.pop(skipped_key, None)
        try:
            records, skipped = records_from_local_path(path)
        except ExtractionError as exc:
            st.warning(str(exc))
        else:
            st.session_state[import_key] = records
            if skipped:
                st.session_state[skipped_key] = skipped
    for message in st.session_state.get(skipped_key, []):
        st.warning(message)
    records = st.session_state.get(import_key, [])
    _remember_source_records(slot, "Local folder / file", records)
    return records


def _fetch_records(slot: str) -> list:
    settings = st.session_state.settings
    patient_id = st.text_input(
        "Patient or record ID",
        key=f"fetch_patient_id_{slot}",
        help="Used for the {patient_id} placeholder in the Fetch records path.",
    )
    st.caption(
        f"GET {settings.fetch_base_url.rstrip('/')}{settings.fetch_records_path}"
    )
    import_key = f"fetch_records_{slot}"
    if st.button("Import medical records from Fetch Sandbox", key=f"fetch_import_{slot}"):
        st.session_state.pop(import_key, None)
        try:
            records = FetchClient(settings).fetch_documents(patient_id)
        except FetchSandboxError as exc:
            st.warning(str(exc))
        else:
            st.session_state[import_key] = records
    records = st.session_state.get(import_key, [])
    _remember_source_records(slot, "Fetch Sandbox", records)
    return records


def _va_gov_records(slot: str) -> list:
    """VA.gov source: per-session secure login + consent, then automatic fetch.

    Failure to authenticate or fetch never blocks the other record sources —
    switching the radio back to Upload/Fetch Sandbox/Local always works
    regardless of VA.gov state.
    """
    st.caption(
        "⚠️ Fetched documents are sent to the configured LLM endpoint for analysis. "
        "Review privacy before fetching sensitive records — the same warning shown for "
        "every other record source applies equally here."
    )
    st.caption(
        "Signs in to VA.gov for this session only, then automatically fetches your "
        "available records. Credentials are never stored to disk or `.env`; records "
        "stay local after fetch."
    )
    consent = st.checkbox(
        "I consent to VA.gov fetching my medical records for this session only. "
        "Records are not stored beyond this session and my credentials are never saved.",
        key=f"va_gov_consent_{slot}",
    )
    authed = bool(st.session_state.get(f"va_gov_authed_{slot}"))
    with st.expander("🔒 VA.gov secure login", expanded=not authed):
        username = st.text_input("VA.gov username", key=f"va_gov_user_{slot}")
        password = st.text_input(
            "VA.gov password", type="password", key=f"va_gov_pass_{slot}"
        )
        login_clicked = st.button(
            "Log in and fetch VA.gov records",
            key=f"va_gov_login_{slot}",
        )

    if login_clicked:
        telemetry.track_interaction(
            va_gov_client.FEATURE_ID,
            {
                "source_selected": "va_gov",
                "login_modal_opened": True,
                "consent_given": consent,
            },
        )
        if not consent:
            st.warning("Consent is required before VA.gov records can be fetched.")
        else:
            try:
                session = va_gov_client.authenticate_va_gov(username, password)
            except va_gov_client.VaGovError as exc:
                st.error(str(exc))
            else:
                st.session_state[f"va_gov_session_{slot}"] = session
                st.session_state[f"va_gov_authed_{slot}"] = True
                st.session_state[f"va_gov_fetch_{slot}"] = va_gov_client.fetch_va_records(session)
                st.session_state.pop(f"va_gov_accept_partial_{slot}", None)

    result = st.session_state.get(f"va_gov_fetch_{slot}")
    if result is None:
        return []

    if result.error_message:
        st.error(
            f"VA.gov fetch problem: {result.error_message} "
            f"(retrieved {result.retrieved} of {result.expected} expected record(s))."
        )
        col_retry, col_continue = st.columns(2)
        if col_retry.button("Retry VA.gov fetch", key=f"va_gov_retry_{slot}"):
            session = st.session_state.get(f"va_gov_session_{slot}")
            if session is not None:
                st.session_state[f"va_gov_fetch_{slot}"] = va_gov_client.fetch_va_records(session)
                st.rerun()
        if col_continue.button(
            "Continue with available VA.gov records", key=f"va_gov_continue_{slot}"
        ):
            st.session_state[f"va_gov_accept_partial_{slot}"] = True

    if not result.documents:
        return []
    if result.error_message and not st.session_state.get(f"va_gov_accept_partial_{slot}"):
        return []

    _remember_source_records(slot, "VA.gov", result.documents)
    other_sources = dict(st.session_state.get(f"source_records_{slot}", {}))
    merged = va_gov_client.merge_records(other_sources)

    st.subheader("Merged records summary")
    st.dataframe(
        [
            {"Source": row.source, "File": row.filename, "Pages": row.pages}
            for row in merged.summary
        ],
        use_container_width=True,
        hide_index=True,
    )
    confirmed = st.checkbox(
        "I confirm this merged record set is correct and want to proceed.",
        key=f"va_gov_confirm_{slot}",
    )
    telemetry.track_interaction(
        va_gov_client.FEATURE_ID,
        {
            "records_fetched": result.retrieved,
            "records_expected": result.expected,
            "sources_merged": merged.sources_merged,
            "confirmed": confirmed,
        },
    )
    if not confirmed:
        return []
    return merged.documents


def _load_usage_history() -> watchdog.UsageHistory:
    """Load cached usage history; never fails (empty on first run)."""
    try:
        return watchdog.load_history()
    except Exception:  # noqa: BLE001 - keep the app usable on any I/O error
        return watchdog.UsageHistory()


def _save_usage_history(history: watchdog.UsageHistory) -> None:
    """Persist usage history; swallow I/O errors so the app never breaks on them."""
    try:
        watchdog.save_history(history)
    except Exception:  # noqa: BLE001
        pass


def _record_watchdog_run(usage) -> None:
    """Append a finished run's token totals to the persisted history.

    Per-role token totals (main vs fast) are carried along so the watchdog can
    fit separate credit rates for the two models.
    """
    total = usage.totals()
    if not total.calls:
        return
    history = _load_usage_history()
    watchdog.record_run(
        history,
        prompt_tokens=total.prompt_tokens,
        completion_tokens=total.completion_tokens,
        calls=total.calls,
        by_role=usage.per_role_tokens(),
    )
    _save_usage_history(history)


def _effective_credit_rates() -> tuple[dict[str, float | None], str]:
    """Resolve credits-per-1M rates for the estimator.

    Prefers explicitly-configured env/per-model rates; otherwise falls back to the
    effective blended rate learned by the watchdog (if it has enough data).
    Returns (rates_by_model, source_label).
    """
    rates = {
        config.DEFAULT_MODEL_MAIN: config.CREDITS_PER_1M_MAIN,
        config.DEFAULT_MODEL_FAST: config.CREDITS_PER_1M_FAST,
    }
    if all(value is not None for value in rates.values()):
        return rates, "configured in .env"
    fit = watchdog.fit_effective_rate(_load_usage_history())
    if fit.any_rate():  # pragma: no branch - guarded
        label = "estimated by the usage watchdog"
        # Fill only the rates missing from .env; never overwrite an explicit
        # value the user configured for one model.
        if rates[config.DEFAULT_MODEL_MAIN] is None and fit.main_rate is not None:
            rates[config.DEFAULT_MODEL_MAIN] = fit.main_rate
        if rates[config.DEFAULT_MODEL_FAST] is None and fit.fast_rate is not None:
            rates[config.DEFAULT_MODEL_FAST] = fit.fast_rate
        return rates, label
    return rates, "entry"


def _credit_calibration_widget() -> None:
    """Sidebar: record console readings and surface the learned rate."""
    with st.expander("🎚️ Usage watchdog (credit rate)"):
        history = _load_usage_history()
        fit = watchdog.fit_effective_rate(history)

        last_credits = st.session_state.get("watchdog_last_credits", "")
        credits = st.text_input(
            "Total credits used (from QwenCloud console)",
            value=last_credits,
            key="watchdog_credits_input",
        )
        captured = False
        if st.button("Record this reading"):
            try:
                parsed = float(credits)
                if parsed < 0:
                    raise ValueError
                watchdog.record_calibration(history, credits=parsed)
                _save_usage_history(history)
                st.session_state["watchdog_last_credits"] = credits
                captured = True
            except (TypeError, ValueError):
                st.warning("Enter a non-negative number for credits used.")

        n_runs = len(history.runs)
        if captured:
            st.success(
                f"Reading saved ({n_runs} run(s) recorded). Repeat after more runs to refine."
            )

        st.caption(
            f"Runs tracked: {n_runs} · calibrations: {len(history.calibrations)}"
        )
        if fit.any_rate():
            separate = fit.main_rate != fit.fast_rate
            if separate:
                st.markdown(
                    f"**Learned rates:** main ≈{fit.main_rate:,.0f} · fast "
                    f"≈{fit.fast_rate:,.0f} credits/1M tokens. {fit.multiline_note}"
                )
            else:
                st.markdown(
                    f"**Learned effective rate:** ≈{fit.blended_rate:,.0f} credits / 1M "
                    f"tokens {fit.multiline_note}"
                )
            enabled = all(
                config.CREDITS_PER_1M_MAIN is None
                and config.CREDITS_PER_1M_FAST is None
            )
            if enabled:
                rate_desc = (
                    f"main **{fit.main_rate:,.0f}** / fast **{fit.fast_rate:,.0f} credits/1M**"
                    if separate
                    else f"**{fit.blended_rate:,.0f} credits/1M**"
                )
                st.markdown(
                    f"The estimator will now use {rate_desc} as a fallback for the "
                    "credit estimate until you set explicit rates in `.env`."
                )
        else:
            st.markdown(
                "Add the total credits your plan reports each time after a run. Once you've "
                "recorded at least two readings separated by new runs, the app fits your "
                "effective credits-per-1M rate and starts estimating credit burn."
            )
            st.caption(
                "Tip: post each eval/draft run's totals (shown here) and your console's "
                "cumulative credits to converge in a few runs."
            )


def _compat_model_warning(settings) -> None:
    """Warn if the configured models are not listed at GET {base_url}/models.

    Advisory only: network/permission failures are silently ignored and the
    warning is cached per-session so the endpoint is not hit on every rerun.
    """
    sig = f"{settings.base_url}|{settings.model_main}|{settings.model_fast}"
    if st.session_state.get("_compat_checked_sig") == sig:
        for msg in st.session_state.get("_compat_warnings", []):
            st.warning(msg)
        return
    try:
        from .llm import check_model_availability

        available = check_model_availability(settings.base_url, settings.api_key)
    except Exception:  # noqa: BLE001 - never break the UI on a compat check
        available = None
    warnings: list[str] = []
    if available is not None:
        for label, model in (("Main model", settings.model_main), ("Fast model", settings.model_fast)):
            if model and model not in available:
                warnings.append(
                    f"⚠️ {label} `{model}` not found at `{settings.base_url.rstrip('/')}/models`. "
                    "The provider may have deprecated it — check `COMPATIBILITY.md` and `MIGRATION.md`."
                )
    st.session_state["_compat_checked_sig"] = sig
    st.session_state["_compat_warnings"] = warnings
    for msg in warnings:
        logger.warning("model availability warning: %s", msg, extra={"request_id": get_request_id() or "-", "phase": "compat", "status": "warning"})
        st.warning(msg)


def _progress_widgets(llm: LLMClient | None = None, *, request_id: str | None = None):
    """Progress bar whose caption appends a live estimated-usage line."""
    bar = st.progress(0.0, text="Starting…")
    rid = request_id or get_request_id() or "-"

    def update(frac: float, msg: str) -> None:
        logger.debug("progress %.0f%% — %s", frac * 100, msg, extra={"request_id": rid, "phase": "progress", "status": "ok"})
        text = msg
        if llm is not None:
            text += llm.usage.live_line()
        bar.progress(min(max(frac, 0.0), 1.0), text=text)

    return bar, update


def _render_usage_summary(usage) -> None:
    """Render an estimated per-phase usage breakdown after a successful run."""
    if usage is None:
        return
    total = usage.totals()
    if not total.calls:
        return

    with st.expander("⚙️ Estimated API usage (tokens / calls)", expanded=False):
        rows = []
        for phase, stats in usage.per_phase().items():
            models = ", ".join(f"{m}×{c}" for m, c in stats.models.items())
            rows.append(
                {
                    "Phase": phase,
                    "Calls": stats.calls,
                    "Est. input tokens": stats.prompt_tokens,
                    "Est. output tokens": stats.completion_tokens,
                    "Models (calls)": models,
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(
            f"**Total:** {total.calls} call(s) · "
            f"{total.prompt_tokens:,} input / {total.completion_tokens:,} output tokens "
            f"({total.total_tokens:,} total). Token counts are estimates based on prompt "
            "length and model output; they use the provider's reported usage when available."
        )

        rates, rate_source = _effective_credit_rates()
        credits = usage.credit_estimate(rates)
        if credits is not None:
            pct = credits / config.CREDIT_QUOTA * 100 if config.CREDIT_QUOTA else 0.0
            st.caption(
                f"**Estimated credit burn:** {credits:,.0f} / {config.CREDIT_QUOTA:,.0f} "
                f"({pct:.1f}% of the weekly quota, {rate_source})"
            )
        else:
            st.caption(
                "Set `VA_LSE_CREDITS_PER_1M_MAIN` / `VA_LSE_CREDITS_PER_1M_FAST` in your "
                ".env — or add console readings in the sidebar's **Usage watchdog** "
                "panel so the app can estimate your rate — to see credit burn here."
            )



# --------------------------------------------------------------- evaluate tab
def evaluate_tab() -> None:
    st.subheader("Step 1 — Provide the lay statement")
    mode = st.radio(
        "Statement source", ["Upload file", "Paste text"], key="eval_mode", horizontal=True
    )
    statement_text = ""
    if mode == "Paste text":
        statement_text = st.text_area(
            "Paste the full lay/witness statement", height=260, key="eval_paste"
        )
    else:
        files = st.file_uploader(
            "Upload the statement (PDF, TXT, MD, DOCX)",
            type=["pdf", "txt", "md", "docx"],
            key="eval_statement_file",
        )
        if files is not None:
            accepted, rejections = _check_upload_limits([files])
            for msg in rejections:
                st.warning(msg)
            if rejections:
                docs = []
            else:
                docs = _extract_uploads([files], "eval_statement")
            if docs:
                statement_text = docs[0].full_text

    if statement_text:
        n = len(statement_text)
        st.caption(
            f"Statement length: {n:,} / {MAX_STATEMENT_CHARS:,} characters "
            f"(recommended limit; hard prompt limit {EVALUATE_INTERNAL_MAX_CHARS:,})."
        )
        if n > MAX_STATEMENT_CHARS:
            over = n - MAX_STATEMENT_CHARS
            will_truncate = max(0, n - EVALUATE_INTERNAL_MAX_CHARS)
            if will_truncate:
                st.warning(
                    f"⚠️ Statement is {n:,} characters — {over:,} over the {MAX_STATEMENT_CHARS:,} "
                    f"recommended limit. {will_truncate:,} characters beyond the "
                    f"{EVALUATE_INTERNAL_MAX_CHARS:,} internal prompt limit will be "
                    f"truncated and not analyzed. Claims at the end (e.g., family impact, "
                    f"caregiver necessity) may be missed. Consider splitting the statement "
                    f"into smaller parts or shortening it."
                )
            else:
                st.warning(
                    f"⚠️ Statement is {n:,} characters — {over:,} over the {MAX_STATEMENT_CHARS:,} "
                    f"recommended limit. It will still be analyzed in full (internal limit "
                    f"{EVALUATE_INTERNAL_MAX_CHARS:,}), but very long statements may reduce "
                    f"model accuracy. Consider shortening for best results."
                )
            st.checkbox(
                f"I understand the statement is {over:,} characters over the limit and "
                "want to proceed anyway (any truncated portion will be noted in the report).",
                key="eval_confirm_oversize",
            )
        elif n > int(MAX_STATEMENT_CHARS * 0.85):
            st.caption(
                f"ℹ️ Approaching the {MAX_STATEMENT_CHARS:,} character recommended limit "
                f"({MAX_STATEMENT_CHARS - n:,} remaining before a confirmation is required)."
            )

    st.subheader("Step 2 — Provide the medical records")
    records = _records_uploader("eval")
    if records:
        total_pages = sum(len(d.pages) for d in records)
        st.success(
            f"Loaded {len(records)} record file(s), {total_pages:,} page(s): "
            + ", ".join(d.filename for d in records)
        )
        if total_pages > 200:
            st.info(
                "Large record set: chunks are digested in parallel with duplicate pages "
                "skipped, but expect a longer run for a meticulous review."
            )

    render_condition_selector("eval", FEATURE_ID)

    run = st.button("🔍 Run exhaustive evaluation", type="primary", key="eval_run")
    if run:
        if not statement_text.strip():
            st.error("Provide the lay statement first (upload or paste).")
            return
        if len(statement_text) > MAX_STATEMENT_CHARS and not st.session_state.get(
            "eval_confirm_oversize"
        ):
            over = len(statement_text) - MAX_STATEMENT_CHARS
            will_truncate = max(0, len(statement_text) - EVALUATE_INTERNAL_MAX_CHARS)
            msg = (
                f"Statement is {len(statement_text):,} characters — {over:,} over the "
                f"{MAX_STATEMENT_CHARS:,} limit. Check the confirmation box above to proceed, "
                "or split/shorten the statement."
            )
            if will_truncate:
                msg += f" {will_truncate:,} characters would be truncated and not analyzed."
            st.error(msg)
            return
        if not records:
            st.error("Upload at least one medical record file.")
            return
        rid = _new_run_request_id()
        llm = _get_llm()
        if llm is None:
            return

        logger.info(
            "evaluate run start statement_chars=%d pages=%d",
            len(statement_text.strip()), sum(len(d.pages) for d in records),
            extra={"request_id": rid, "phase": "evaluate", "status": "start"},
        )
        bar, update = _progress_widgets(llm, request_id=rid)
        t0 = time.perf_counter()
        try:
            result = run_evaluation(llm, statement_text.strip(), records, progress=update)
        except Exception as exc:  # noqa: BLE001
            bar.empty()
            duration_ms = int((time.perf_counter() - t0) * 1000)
            logger.error(
                "evaluate run error duration_ms=%d error=%s",
                duration_ms, f"{type(exc).__name__}: {exc}",
                exc_info=exc,
                extra={"request_id": rid, "phase": "evaluate", "status": "error", "duration_ms": duration_ms, "error_class": type(exc).__name__},
            )
            st.error(f"Evaluation failed: {_format_error_for_user(exc, rid)}")
            return
        duration_ms = int((time.perf_counter() - t0) * 1000)
        total = llm.usage.totals()
        logger.info(
            "evaluate run done duration_ms=%d calls=%d tokens_in=%d tokens_out=%d",
            duration_ms, total.calls, total.prompt_tokens, total.completion_tokens,
            extra={"request_id": rid, "phase": "evaluate", "status": "ok", "duration_ms": duration_ms, "calls": total.calls, "prompt_tokens": total.prompt_tokens, "completion_tokens": total.completion_tokens},
        )
        bar.empty()
        st.session_state.eval_result = result
        st.session_state.eval_usage = llm.usage
        st.session_state.eval_request_id = rid
        _record_watchdog_run(llm.usage)

    result = st.session_state.get("eval_result")
    if result is None:
        return

    _render_usage_summary(st.session_state.get("eval_usage"))

    if getattr(result, "truncation_warning", ""):
        st.warning(
            f"⚠️ {result.truncation_warning} (input was {result.input_chars:,} chars; "
            f"{result.truncated_chars:,} truncated). Review the report header for details."
        )

    st.divider()
    st.subheader("📋 Evaluation Results")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall rating", result.overall_rating)
    col2.metric("Claims verified", len(result.verifications))
    col3.metric("Contradictions", result.contradiction_count)
    _applicable = [t for t in result.topic_rows if t.get("applicable")]
    _covered = [t for t in _applicable if t.get("coverage") == "covered"]
    col4.metric("Topics covered", f"{len(_covered)}/{len(_applicable)}" if _applicable else "—")

    with st.expander("Executive summary", expanded=True):
        st.write(result.executive_summary)

    with st.expander("Claim-by-claim verification table", expanded=True):
        rows = []
        claim_text = {c["id"]: c.get("text", "") for c in result.claims}
        for v in result.verifications:
            rows.append(
                {
                    "Claim": claim_text.get(v.get("id"), ""),
                    "Verdict": v.get("verdict", ""),
                    "Record reference": v.get("record_reference", ""),
                    "Note": v.get("note", ""),
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("Rubric scores", expanded=True):
        score_rows = [
            {
                "Dimension": DIMENSION_LABELS.get(k, k),
                "Score": result.scores.get(k, 0),
                "Rationale": result.rationales.get(k, ""),
            }
            for k in DIMENSION_LABELS
        ]
        st.dataframe(score_rows, use_container_width=True, hide_index=True)
        st.bar_chart(
            {DIMENSION_LABELS[k]: result.scores.get(k, 0) for k in DIMENSION_LABELS},
            horizontal=True,
        )

    if result.topic_rows:
        with st.expander(
            "🧭 Topic coverage — what the statement does and does not address", expanded=True
        ):
            if result.topic_focus:
                st.write(f"**Claim focus:** {result.topic_focus}")
            topic_table = [
                {
                    "Topic": t.get("topic", ""),
                    "Applicable": "Yes" if t.get("applicable") else "No",
                    "Coverage": t.get("coverage", ""),
                    "Evidence in statement": t.get("evidence", ""),
                    "How to strengthen": t.get("gap_note", ""),
                }
                for t in result.topic_rows
            ]
            st.dataframe(topic_table, use_container_width=True, hide_index=True)
            if result.topic_critical_gaps:
                st.warning(
                    "**Critical gaps — the highest-impact topics this statement still misses:**"
                )
                for gap in result.topic_critical_gaps:
                    st.write(f"- {gap}")
            if result.topic_notes:
                st.caption(result.topic_notes)

    with st.expander("Improvements & record facts to add", expanded=True):
        for imp in result.improvements:
            st.markdown(f"**{imp.get('priority', '?')}. {imp.get('problem', '')}**")
            st.write(imp.get("suggestion", ""))
            if imp.get("example_rewrite"):
                st.caption(f"Example: “{imp.get('example_rewrite')}”")
        if result.omitted_record_facts:
            st.markdown("**Facts from the records you could add (verify first):**")
            for fact in result.omitted_record_facts:
                st.write(f"- {fact.get('fact', '')} _(source: {fact.get('source', '')})_")

    if result.revised_statement or result.revision_changes:
        with st.expander("📝 Suggested improvements — proposed rewrite", expanded=True):
            if result.revision_notes:
                st.info(result.revision_notes)
            if result.revision_changes:
                change_rows = [
                    {
                        "Category": c.get("category", ""),
                        "Original": c.get("original", "") or "(addition)",
                        "Suggested": c.get("revised", ""),
                        "Why": c.get("reason", ""),
                    }
                    for c in result.revision_changes
                ]
                st.dataframe(change_rows, use_container_width=True, hide_index=True)
            if result.added_facts_to_verify:
                st.markdown(
                    "**Record-sourced facts added — the witness must confirm each before signing:**"
                )
                for fact in result.added_facts_to_verify:
                    st.write(f"- {fact}")
            st.markdown("#### Revised statement")
            st.caption(
                "Contradictions have been corrected to match the medical records. Resolve every "
                "[Confirm: ...] placeholder with the witness before signing."
            )
            revised = st.text_area(
                "Revised statement (editable)",
                value=result.revised_statement,
                height=420,
                key="eval_revised_statement",
            )
            col_a, col_b = st.columns(2)
            col_a.download_button(
                "⬇️ Download revised statement (.txt)",
                data=revised.encode("utf-8"),
                file_name="lay_statement_revised.txt",
                mime="text/plain",
            )
            col_b.download_button(
                "⬇️ Download revised statement (.md)",
                data=revised.encode("utf-8"),
                file_name="lay_statement_revised.md",
                mime="text/markdown",
            )

    with st.expander("Full markdown report"):
        st.markdown(result.report_markdown)
    st.download_button(
        "⬇️ Download evaluation report (.md)",
        data=result.report_markdown.encode("utf-8"),
        file_name="lay_statement_evaluation.md",
        mime="text/markdown",
    )


# ----------------------------------------------------------------- draft tab
def draft_tab() -> None:
    st.subheader("Step 1 — Upload the veteran's medical records")
    records = _records_uploader("draft")
    if records:
        total_pages = sum(len(d.pages) for d in records)
        st.success(
            f"Loaded {len(records)} record file(s), {total_pages:,} page(s): "
            + ", ".join(d.filename for d in records)
        )
        if total_pages > 200:
            st.info(
                "Large record set: chunks are digested in parallel with duplicate pages "
                "skipped, but expect a longer run for a meticulous review."
            )

    render_condition_selector("draft", FEATURE_ID)

    st.subheader("Step 2 — Claim details")
    col1, col2 = st.columns(2)
    veteran_name = col1.text_input("Veteran's name", key="draft_vet_name")
    condition = col2.text_input(
        "Condition the statement supports (e.g., PTSD, lumbar strain, tinnitus)",
        key="draft_condition",
    )
    col3, col4 = st.columns(2)
    claim_type = col3.selectbox("Claim type", CLAIM_TYPES, key="draft_claim_type")
    relationship = col4.selectbox("Witness relationship", RELATIONSHIPS, key="draft_rel")

    st.subheader("Step 3 — Witness details")
    col5, col6, col7 = st.columns(3)
    witness_name = col5.text_input("Witness full name", key="draft_witness_name")
    known_since = col6.text_input("Known the veteran since / for", key="draft_known")
    contact_frequency = col7.text_input(
        "How often they see each other (opportunity to observe)", key="draft_freq"
    )
    witnessed_event = st.radio(
        "Did the witness personally see the in-service event happen?",
        ["No", "Yes", "Not applicable"],
        horizontal=True,
        key="draft_witnessed",
    )

    st.subheader("Step 4 — What has the witness observed?")
    observations = st.text_area(
        "Describe everything the witness has personally seen, heard, or experienced "
        "regarding the veteran's condition: symptoms, incidents, changes over time, "
        "impact on work, family and social life. Bullet points are fine — the app will "
        "turn them into a polished, factually grounded statement.",
        height=220,
        key="draft_observations",
    )
    if observations:
        n = len(observations)
        st.caption(
            f"Observations length: {n:,} / {MAX_OBSERVATIONS_CHARS:,} characters "
            f"(recommended limit; hard prompt limit {DRAFT_INTERNAL_MAX_CHARS:,})."
        )
        if n > MAX_OBSERVATIONS_CHARS:
            over = n - MAX_OBSERVATIONS_CHARS
            will_truncate = max(0, n - DRAFT_INTERNAL_MAX_CHARS)
            if will_truncate:
                st.warning(
                    f"⚠️ Observations are {n:,} characters — {over:,} over the "
                    f"{MAX_OBSERVATIONS_CHARS:,} recommended limit. {will_truncate:,} "
                    f"characters beyond the {DRAFT_INTERNAL_MAX_CHARS:,} internal limit "
                    f"will be truncated and not grounded. Details at the end may be missed. "
                    f"Consider shortening or splitting."
                )
            else:
                st.warning(
                    f"⚠️ Observations are {n:,} characters — {over:,} over the "
                    f"{MAX_OBSERVATIONS_CHARS:,} recommended limit. They will still be "
                    f"grounded in full (internal limit {DRAFT_INTERNAL_MAX_CHARS:,}), but "
                    f"very long inputs may reduce accuracy."
                )
            st.checkbox(
                f"I understand the observations are {over:,} characters over the limit and "
                "want to proceed anyway (any truncated portion will be noted in the results).",
                key="draft_confirm_oversize",
            )
        elif n > int(MAX_OBSERVATIONS_CHARS * 0.85):
            st.caption(
                f"ℹ️ Approaching the {MAX_OBSERVATIONS_CHARS:,} recommended limit "
                f"({MAX_OBSERVATIONS_CHARS - n:,} remaining before confirmation is required)."
            )

    run = st.button("✍️ Draft the statement", type="primary", key="draft_run")
    if run:
        if not records:
            st.error("Upload at least one medical record file first.")
            return
        if not observations.strip() or not condition.strip():
            st.error("Enter the condition and the witness's observations.")
            return
        if len(observations) > MAX_OBSERVATIONS_CHARS and not st.session_state.get(
            "draft_confirm_oversize"
        ):
            over = len(observations) - MAX_OBSERVATIONS_CHARS
            will_truncate = max(0, len(observations) - DRAFT_INTERNAL_MAX_CHARS)
            msg = (
                f"Observations are {len(observations):,} characters — {over:,} over the "
                f"{MAX_OBSERVATIONS_CHARS:,} limit. Check the confirmation box above to proceed, "
                "or shorten/split the observations."
            )
            if will_truncate:
                msg += f" {will_truncate:,} characters would be truncated and not grounded."
            st.error(msg)
            return
        rid = _new_run_request_id()
        llm = _get_llm()
        if llm is None:
            return

        logger.info(
            "draft run start observations_chars=%d pages=%d condition=%s",
            len(observations.strip()), sum(len(d.pages) for d in records),
            condition.strip()[:60] if condition.strip() else "-",
            extra={"request_id": rid, "phase": "draft", "status": "start"},
        )
        witness = {
            "name": witness_name.strip(),
            "relationship": relationship,
            "known_since": known_since.strip(),
            "contact_frequency": contact_frequency.strip(),
            "veteran_name": veteran_name.strip(),
            "witnessed_event": witnessed_event,
        }
        bar, update = _progress_widgets(llm, request_id=rid)
        t0 = time.perf_counter()
        try:
            result = run_draft(
                llm, records, witness, observations.strip(), condition.strip(),
                claim_type, progress=update,
            )
        except Exception as exc:  # noqa: BLE001
            bar.empty()
            duration_ms = int((time.perf_counter() - t0) * 1000)
            logger.error(
                "draft run error duration_ms=%d error=%s",
                duration_ms, f"{type(exc).__name__}: {exc}",
                exc_info=exc,
                extra={"request_id": rid, "phase": "draft", "status": "error", "duration_ms": duration_ms, "error_class": type(exc).__name__},
            )
            st.error(f"Drafting failed: {_format_error_for_user(exc, rid)}")
            return
        duration_ms = int((time.perf_counter() - t0) * 1000)
        total = llm.usage.totals()
        logger.info(
            "draft run done duration_ms=%d calls=%d tokens_in=%d tokens_out=%d",
            duration_ms, total.calls, total.prompt_tokens, total.completion_tokens,
            extra={"request_id": rid, "phase": "draft", "status": "ok", "duration_ms": duration_ms, "calls": total.calls, "prompt_tokens": total.prompt_tokens, "completion_tokens": total.completion_tokens},
        )
        bar.empty()
        st.session_state.draft_result = result
        st.session_state.draft_usage = llm.usage
        st.session_state.draft_request_id = rid
        _record_watchdog_run(llm.usage)

    result = st.session_state.get("draft_result")
    if result is None:
        return

    _render_usage_summary(st.session_state.get("draft_usage"))

    if getattr(result, "truncation_warning", ""):
        st.warning(
            f"⚠️ {result.truncation_warning} (input was {result.input_chars:,} chars; "
            f"{result.truncated_chars:,} truncated). Review the grounding section for details."
        )

    st.divider()
    st.subheader("📋 Draft Results")

    with st.expander("Grounding analysis — how the draft ties to the records", expanded=True):
        st.markdown(grounding_markdown(result))

    if result.review_issues:
        with st.expander("Self-review findings (fixed in the final version)"):
            for issue in result.review_issues:
                st.write(f"- {issue}")

    st.markdown("### Final statement (editable)")
    st.caption(
        "Review every bracketed [Confirm: ...] placeholder and resolve it before signing. "
        "Submit on VA Form 21-10210 (one form per witness)."
    )
    edited = st.text_area(
        "Statement", value=result.output_statement, height=460, key="draft_edited"
    )
    col_a, col_b = st.columns(2)
    col_a.download_button(
        "⬇️ Download statement (.txt)",
        data=edited.encode("utf-8"),
        file_name="lay_statement_draft.txt",
        mime="text/plain",
    )
    col_b.download_button(
        "⬇️ Download statement (.md)",
        data=edited.encode("utf-8"),
        file_name="lay_statement_draft.md",
        mime="text/markdown",
    )

    if result.digest:
        with st.expander("Medical record digest used for grounding"):
            st.caption(
                f"{len(result.digest.facts):,} facts extracted from "
                f"{result.digest.pages_reviewed:,} pages "
                f"({result.digest.chunks_reviewed} chunks, "
                f"{result.digest.duplicates_skipped} duplicate page(s) skipped)"
            )
            st.write(result.digest.summary)
            st.code(result.digest.timeline_text()[:20000], language=None)


# ----------------------------------------------------------------- about tab
def about_tab() -> None:
    from .config import load_knowledge

    st.subheader("What this tool does")
    st.markdown(
        """
**Pathway 1 — Evaluate:** Upload an already-written lay/witness statement plus the veteran's
medical records. The app conducts an exhaustive review of the records, extracts every factual
claim in the statement, verifies each claim against the records (supported / contradicted /
partially supported / not found), scores the statement on an 8-dimension rubric drawn from VA
lay-evidence law, and audits it against the topic checklist (hazards and dangers, caregiver
necessity, personal care, medication and financial management, household safety, errands and
driving, before/after progression, observable behaviors, family impact, medication side
effects). It then suggests how to improve it: a prioritized improvement plan plus a proposed
rewrite with corrections grounded in the records and confirmation placeholders.

**Pathway 2 — Draft:** Upload the veteran's medical records and answer questions about what the
witness has personally observed. The app grounds the statement in the records, checks the
observations against the topic checklist and asks follow-up questions for applicable topics the
witness has not yet covered, flags anything that conflicts or cannot be verified, and drafts a
first-person statement in VA Form 21-10210 style that stays strictly within lay-competence
boundaries.

Both pathways review every page of every uploaded document — records are processed in chunks
so very long files are handled exhaustively. Large record sets (hundreds to thousands of pages,
up to a configurable cap of ~5,000 pages) are supported: chunks are digested in parallel,
duplicate pages are skipped automatically, and verification always searches the full digest for
evidence relevant to each claim rather than reading only the first pages.
"""
    )
    st.subheader("Legal foundation")
    with st.expander("Legal framework distilled into this tool"):
        st.markdown(load_knowledge("legal_framework.md"))
    with st.expander("Evaluation rubric"):
        st.markdown(load_knowledge("evaluation_rubric.md"))
    with st.expander("Drafting guide"):
        st.markdown(load_knowledge("drafting_guide.md"))
    with st.expander("Topic checklist"):
        st.markdown(load_knowledge("topic_checklist.md"))
    st.info(
        "This tool is an educational and drafting aid. It is not legal, medical, or claims "
        "advice, and no output should be submitted without the witness personally verifying "
        "every fact. For accredited help: www.va.gov/ogc/apps/accreditation"
    )


# --------------------------------------------------------------------- layout
def main() -> None:
    configure_logging()
    _check_streamlit_config_hardening()
    # Ensure every browser session has a baseline correlation id (also used
    # for pre-run validation / upload errors so those logs are correlatable).
    try:
        _get_or_create_request_id()
    except Exception:  # noqa: BLE001
        pass
    logger.info("app start", extra={"request_id": get_request_id() or "-", "phase": "app", "status": "start"})
    telemetry.init_telemetry()
    _sidebar_settings()
    st.title("🎖️ VA Lay Statement Evaluator")
    st.caption(
        "Exhaustive medical-record review to verify existing lay statements — and to draft "
        "factually correct new ones. Grounded in 38 U.S.C. § 1154(a), § 5107(b) and the "
        "Jandreau/Buchanan/Caluza line of cases."
    )

    tab_eval, tab_draft, tab_about = st.tabs(
        ["🔍 Evaluate a statement", "✍️ Draft a statement", "📖 About / Guide"]
    )
    try:
        with tab_eval:
            evaluate_tab()
        with tab_draft:
            draft_tab()
        with tab_about:
            about_tab()
    except Exception as exc:  # noqa: BLE001 - root error boundary
        rid = get_request_id() or st.session_state.get(_REQUEST_ID_KEY, "-") or "-"
        logger.error(
            "unhandled app error: %s", f"{type(exc).__name__}: {exc}",
            exc_info=exc,
            extra={"request_id": rid, "phase": "app", "status": "error", "error_class": type(exc).__name__},
        )
        telemetry.track_app_error(exc)
        st.error(f"Something went wrong while rendering the app: {_format_error_for_user(exc, rid)}")


if __name__ == "__main__":
    main()
