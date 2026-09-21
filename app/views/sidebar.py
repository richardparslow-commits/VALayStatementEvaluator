"""Sidebar view: LLM settings, Fetch Sandbox settings, usage-watchdog widget.

Extracted from ``app/main.py`` so the entry point stays a thin router. The
settings object itself lives in ``st.session_state.settings`` and is mutated
in place exactly as before — only the rendering moved here.
"""
from __future__ import annotations

import time
from copy import copy
from dataclasses import is_dataclass, replace
from typing import Any

import streamlit as st

from .. import config
from .. import watchdog
from ..config import DEFAULT_BASE_URL, load_settings
from ..error_report import report_failure
from ..llm import check_model_availability
from ..preflight import (
    BLOCKED,
    OK,
    RETIRED_PROVIDER_KINDS,
    Verdict,
    check_endpoint,
    remember_verdict,
    retired_endpoint_reason,
)
from ..prompt_sanitize import validate_api_key, validate_model_name
from .usage import load_usage_history, save_usage_history


def render_sidebar_settings() -> None:
    """Render the LLM/Fetch settings sidebar, mutating session settings in place."""
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
            "Base URL (OpenAI-compatible)",
            value=settings.base_url or DEFAULT_BASE_URL,
            help=(
                "Perplexity's Agent API is the default — it speaks the OpenAI Responses "
                "schema at {base_url}/responses and routes to first-party models "
                "(OpenAI, Anthropic, Google, xAI) under one Perplexity key. Any "
                "Chat-Completions endpoint also works and is detected automatically. "
                "One Perplexity key covers both this endpoint and the Research tab's "
                "grounded lookups."
            ),
        )
        _secrets_source_note(settings)
        col1, col2 = st.columns(2)
        st.session_state.model_main_input = col1.text_input(
            "Main model", value=settings.model_main, help="Analysis, scoring, drafting"
        )
        st.session_state.model_fast_input = col2.text_input(
            "Fast model",
            value=settings.model_fast,
            help="Bulk record digests — one call per record chunk, so use the "
            "cheapest model that extracts accurately to keep a large run cheap.",
        )
        # Both boxes are half-width, so a long id is clipped at the edge — and the
        # character the clip cuts through can read as part of the id, which is
        # exactly how "perplexity/glm-5.3-flash" comes to look like it ends in a
        # stray period. Print the applied ids where they cannot be clipped.
        st.caption(_model_ids_note(settings))
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
                # A rejected Apply is a failure the user may need to describe later
                # ("it says my API key is invalid"), so each message carries an id
                # that resolves to a log line naming which field was rejected.
                for msg in errors:
                    st.error(
                        report_failure(
                            msg, phase="settings_validation", severity="warning"
                        )
                    )
            else:
                settings.api_key = api_key_val
                settings.base_url = st.session_state.base_url_input.strip() or DEFAULT_BASE_URL
                settings.model_main = model_main_val
                settings.model_fast = model_fast_val
                settings.fetch_api_key = fetch_key_val
                settings.fetch_base_url = st.session_state.fetch_base_url_input.strip()
                settings.fetch_records_path = st.session_state.fetch_records_path_input.strip()
                st.rerun()

        if st.button(
            "Test connection",
            help=(
                "Runs the same preflight a run does, against the values above: the model "
                "listing and one real call per configured model."
            ),
        ):
            _test_connection_report(settings)

        _pending_settings_warning(settings)
        _compat_model_warning(settings)
        _retired_endpoint_warning(settings)

        st.divider()
        _credit_calibration_widget()
        st.divider()
        _job_queue_panel()
        st.divider()
        _audit_backup_panel()
        _llm_failover_panel()
        st.divider()
        st.caption(
            "⚠️ Uploaded documents are sent to the configured LLM endpoint for analysis. "
            "Review privacy before uploading sensitive records."
        )
        st.caption(
            "This tool is an aid for drafting and reviewing lay statements. It is not "
            "legal, medical, or claims advice."
        )


def _job_queue_panel() -> None:
    """Read-only view of how runs execute and whether the queue is healthy.

    Deliberately does no I/O on render: this runs on every sidebar paint, and the
    Upstash tier's ``depth()`` is two HTTP requests, so probing it here would add
    network latency to every widget interaction. The backlog is fetched only when
    asked for, and cached with the time it was taken.
    """
    from ..blob_store import get_blob_store
    from ..job_queue import get_job_backend

    with st.expander("🛠️ Job queue — how runs execute", expanded=False):
        if not config.JOB_QUEUE_ENABLED:
            st.caption(
                "Runs execute inside this Streamlit process. For large record sets that "
                "means the pod serving your browser also carries the digest (a 2,000-page "
                "bundle peaks near 1.8 GB) and a pod restart loses the run."
            )
            st.caption(
                "Set `VA_LSE_JOB_QUEUE=1` and configure Redis or the shared cache to hand "
                "runs to a worker pool — see DEPLOYMENT.md → Pattern C."
            )
            return
        try:
            backend = get_job_backend()
        except Exception as exc:  # noqa: BLE001 - this panel must never break the app
            st.error(
                report_failure(
                    f"Job queue unavailable: {type(exc).__name__}: {exc}",
                    phase="job_queue_panel",
                    exc=exc,
                    once=True,  # repainted on every rerun; log the first only
                )
            )
            return

        distributed = backend.is_distributed
        st.caption(
            f"Enabled. Runs are submitted to a **{backend.name}** queue for a worker to "
            "execute."
        )
        if not distributed:
            st.warning(
                f"The active backend (**{backend.name}**) only works inside one process, so "
                "a separate worker can never claim these jobs. Set `VA_LSE_REDIS_URL` or "
                "`VA_LSE_SHARED_CACHE_URL`/`_TOKEN`."
            )

        try:
            blob = get_blob_store()
            blob_label = f"{blob.name} (shared)" if blob.is_shared else blob.name
        except Exception:  # noqa: BLE001
            blob_label = "unavailable"
        st.dataframe(
            [
                {"Setting": "Backend", "Value": backend.name},
                {"Setting": "Distributed", "Value": "yes" if distributed else "no"},
                {"Setting": "Job documents", "Value": blob_label},
                {
                    "Setting": "Inline payload limit",
                    "Value": f"{config.JOB_QUEUE_INLINE_MAX_BYTES / 1024:.0f} KB",
                },
            ],
            width="stretch",
            hide_index=True,
        )

        # Backlog is the one value that needs the queue, so it is on demand.
        if st.button("Check backlog", key="job_queue_probe"):
            try:
                with st.spinner("Asking the queue…"):
                    st.session_state["job_queue_depth"] = backend.depth()
                    st.session_state["job_queue_reachable"] = backend.ping()
                    st.session_state["job_queue_probe_at"] = time.time()
            except Exception as exc:  # noqa: BLE001
                st.session_state["job_queue_depth"] = None
                st.session_state["job_queue_reachable"] = False
                st.session_state["job_queue_probe_at"] = time.time()
                st.error(
                    report_failure(
                        f"Queue probe failed: {type(exc).__name__}: {exc}",
                        phase="job_queue_probe",
                        exc=exc,
                    )
                )
        depth = st.session_state.get("job_queue_depth")
        if st.session_state.get("job_queue_probe_at") and depth is not None:
            # A probe that threw already reported its own cause above; reporting a
            # backlog we never managed to read would only add a vaguer error.
            reachable = st.session_state.get("job_queue_reachable")
            age = time.time() - float(st.session_state["job_queue_probe_at"])
            if reachable is False:
                st.error("Queue unreachable — check Redis/Upstash and the worker pods.")
            elif depth:
                st.info(f"{depth} job(s) waiting for a worker ({age:.0f}s ago).")
            else:
                st.caption(f"No jobs waiting ({age:.0f}s ago).")

        st.caption(
            "Full status (backend, backlog, reachability) is on `GET /health` → "
            "`job_queue`. Workers expose the same on their own health port."
        )


def _audit_backup_panel() -> None:
    """Read-only view of the compliance stream: is it backed up and is it surviving?

    Unlike the job-queue panel this does **not** defer its values behind a button,
    because nothing here touches the network: ``audit_backup_health`` reads the
    state file the backup process writes and stats the log files, and
    ``disk_status`` is one ``statvfs``. Those are the questions an operator has to
    be able to answer without shelling into a pod — ``last_success`` staleness is
    the whole difference between "we are backing up" and "we stopped three weeks
    ago and nobody noticed".
    """
    from ..audit_backup import audit_backup_health, disk_status

    with st.expander("🗄️ Audit log backup — retention and off-pod copies", expanded=False):
        try:
            payload = audit_backup_health()
            disk = disk_status()
        except Exception as exc:  # noqa: BLE001 - the panel must never break the app
            st.error(
                report_failure(
                    f"Audit backup status unavailable: {type(exc).__name__}: {exc}",
                    phase="audit_backup_panel",
                    exc=exc,
                    once=True,  # repainted on every rerun; log the first only
                )
            )
            return

        status = str(payload.get("status") or "unknown")
        if not payload.get("configured"):
            st.warning(
                "No backup destination is configured, so the audit log exists only on "
                "this volume — a pod restart or volume loss takes the record with it."
            )
            st.caption(
                "Set `VA_LSE_AUDIT_BACKUP_DESTINATION` (filesystem, s3, gcs, or azure) and "
                "run `scripts/backup_audit_logs.py`. See DEPLOYMENT.md → Audit log "
                "retention and backup."
            )
        elif status == "error":
            st.error(f"Last backup pass failed: {payload.get('reason') or payload.get('last_error')}")
        elif status == "never_ran":
            st.warning(
                "Configured, but no successful pass has been recorded against this "
                "volume — check that the CronJob or sidecar is actually running."
            )
        elif status == "stale":
            st.warning(f"Backups have stopped: {payload.get('reason')}")
        else:
            age = payload.get("age_seconds")
            st.success(
                "Last pass succeeded"
                + (f" {age / 3600:.1f}h ago." if isinstance(age, int) else ".")
            )

        # The single most misleading configuration: a "backup" that shares the
        # volume it is supposed to survive.
        if payload.get("configured") and payload.get("off_pod") is False:
            st.warning(
                "The destination is on this pod's own volume, so it does **not** survive "
                "the failure a backup exists for. Point it at object storage or a "
                "separate mount."
            )

        pending = payload.get("pending_bytes")
        st.dataframe(
            [
                {"Setting": "Status", "Value": status},
                {
                    "Setting": "Destination",
                    "Value": f"{payload.get('destination')}"
                    + (" (off-pod)" if payload.get("off_pod") else ""),
                },
                {
                    "Setting": "Last success",
                    "Value": payload.get("last_success_utc") or "never",
                },
                {
                    "Setting": "Not yet shipped",
                    "Value": (
                        f"{pending / 1024:.0f} KB would be lost with this pod"
                        if isinstance(pending, int)
                        else "unknown"
                    ),
                },
                {
                    "Setting": "Uploaded",
                    "Value": f"{payload.get('uploaded_objects', 0)} object(s), "
                    f"{payload.get('uploaded_bytes', 0) / 1024:.0f} KB",
                },
                {
                    "Setting": "Retention",
                    "Value": f"{payload.get('local_retention_days')}d local / "
                    f"{payload.get('cloud_retention_days')}d cloud",
                },
                {
                    "Setting": "Log volume free",
                    "Value": (
                        f"{disk['free_bytes'] / 1024 ** 3:.1f} GB "
                        f"({disk['used_percent']}% used)"
                        if disk.get("checked")
                        else "unknown"
                    ),
                },
            ],
            width="stretch",
            hide_index=True,
        )

        if disk.get("checked") and disk.get("below_floor"):
            st.error(
                "Free space is below `VA_LSE_DISK_MIN_FREE_BYTES`. Rotation is by count "
                "and can delete a file that is younger than the retention window — back "
                "up or prune before records are lost."
            )

        st.caption(
            "Full status is on `GET /health` → `audit`, `audit_backup`, `disk`, and as "
            "Prometheus metrics on `GET /metrics`. Verify or restore the backup with "
            "`scripts/restore_audit_logs.py`."
        )


def _primary_breaker_state() -> str:
    """The primary endpoint's circuit-breaker state, or "unavailable".

    Guarded because this panel also renders on the Streamlit-Cloud pattern, where
    the breaker module may not be importable in the same process as the render.
    """
    try:
        from ..circuit_breaker import get_llm_breaker

        return str(get_llm_breaker().state)
    except Exception:  # noqa: BLE001 - display only, never raise
        return "unavailable"


def _primary_breaker_recovery_seconds() -> float:
    """How often a failing primary gets another chance, in seconds."""
    try:
        from ..circuit_breaker import get_llm_breaker

        return float(get_llm_breaker().recovery_timeout)
    except Exception:  # noqa: BLE001 - display only, never raise
        return 60.0


def _llm_failover_panel() -> None:
    """Show which LLM endpoint is serving this session, and what happens next.

    Network-free, so it is not deferred behind a button: ``failover_status`` reads
    the environment and the process-wide breaker, and ``circuit_breaker_state`` is
    the same object the call path consults.

    The countdown is the point of this panel. When the primary is unhealthy but the
    grace period has not elapsed, calls fail fast *by design* (failover is not
    rushed for a blip) — and a user staring at an error has no way to tell a
    two-minute provider hiccup from a broken key. Naming the remaining wait, and
    the one setting that removes it, is what makes that window legible without
    changing the behaviour.
    """
    from ..llm import failover_status

    with st.expander("🔀 LLM endpoint failover", expanded=False):
        try:
            status = failover_status()
        except Exception as exc:  # noqa: BLE001 - the panel must never break the app
            st.error(
                report_failure(
                    f"Failover status unavailable: {type(exc).__name__}: {exc}",
                    phase="llm_failover_panel",
                    exc=exc,
                    once=True,  # repainted on every rerun; log the first only
                )
            )
            return

        configured = bool(status.get("configured"))
        active = bool(status.get("active"))
        threshold = status.get("after_seconds")
        unhealthy = status.get("primary_unhealthy_seconds")

        if not configured:
            st.caption(
                "Running on a **single endpoint**, so an extended outage means waiting "
                "for it to recover (or pointing the app at another one). Set "
                "`OPENAI_BASE_URL_FALLBACK` to arm a backup endpoint — see "
                "DEPLOYMENT.md → LLM endpoint failover."
            )
        elif active:
            st.warning(
                "Calls are being served by the **backup endpoint** because the primary "
                "has been failing. Output may differ slightly from a normal run, and "
                "the audit record for these runs records both endpoints "
                "(`llm_endpoints`). Traffic returns to the primary automatically."
            )
        elif isinstance(unhealthy, (int, float)) and unhealthy > 0:
            remaining = max(0.0, float(threshold or 0) - float(unhealthy))
            recovery = _primary_breaker_recovery_seconds()
            # Precise on purpose: the breaker still lets a retry through every
            # recovery window, so calls are not *uniformly* failing here — and a
            # successful retry ends this state. Saying "everything is down" would
            # send someone hunting for a problem that does not exist.
            st.warning(
                f"The primary endpoint has been failing for {unhealthy:.0f}s, so most "
                f"calls are failing fast. It is re-tried every {recovery:.0f}s, and one "
                f"successful retry puts everything back on it. Failover to the backup "
                f"engages after {float(threshold or 0):.0f}s of continuous failure — in "
                f"about {remaining:.0f}s."
            )
            st.caption(
                "The wait is deliberate: it stops a short provider blip from moving the "
                "work onto another provider. To fail over immediately instead, set "
                "`LLM_ENDPOINT_FALLBACK_TIMEOUT_SECONDS=0`."
            )
        else:
            st.success("Primary endpoint healthy; the backup is configured and unused.")

        rows = [
            {"Setting": "Backup configured", "Value": "yes" if configured else "no"},
            {"Setting": "Serving from backup", "Value": "yes" if active else "no"},
            {"Setting": "Primary breaker", "Value": _primary_breaker_state()},
            {
                "Setting": "Failover after",
                "Value": (
                    f"{float(threshold):.0f}s of continuous failure"
                    if threshold is not None
                    else "unknown"
                ),
            },
            {
                "Setting": "Primary unhealthy for",
                "Value": (
                    f"{float(unhealthy):.0f}s"
                    if isinstance(unhealthy, (int, float))
                    else "healthy"
                ),
            },
        ]
        st.dataframe(rows, width="stretch", hide_index=True)
        st.caption(
            "The backup is configured through the environment, not this sidebar: a "
            "worker builds its own client from its environment, so a web-only setting "
            "would silently not apply to queued runs. Live state is on `GET /health` "
            "→ `llm_failover` and as `va_lse_llm_failover_active` on `GET /metrics`."
        )


def _field_value(key: str, fallback: str = "") -> str:
    """Read a sidebar widget value as a stripped string (fallback when empty)."""
    raw: Any = st.session_state.get(key, "")
    value = raw.strip() if isinstance(raw, str) else ""
    return value or fallback


def _model_ids_note(settings: Any) -> str:
    """The model ids a run will use, in full — the boxes above clip long ones.

    The two model fields sit side by side, so a long id is cut off at the box
    edge. The character the cut lands on can be read as part of the id (a value
    that appears to end in "."), which sends the user hunting a typo that is not
    in the value at all — so the ids in use are stated here, where no width limit
    applies.
    """
    return (
        f"Ids in use: `{settings.model_main}` (main) · `{settings.model_fast}` "
        "— these boxes clip long ids, so check them here."
    )


def _secrets_source_note(settings: Any) -> None:
    """Caption which fields came from Streamlit secrets (hosted deployments).

    Cloud deployments ship no ``.env``, so the fields arrive pre-filled from
    ``.streamlit/secrets.toml``.  Without this note the user cannot tell a
    secret-backed value from a stale one left over from a previous run, which is
    exactly the confusion that sends a valid key to the wrong base URL.
    """
    names = sorted(getattr(settings, "from_secrets", None) or ())
    if not names:
        return
    labels = ", ".join(config.SECRET_FIELD_LABELS.get(name, name) for name in names)
    st.caption(
        f"🔐 From Streamlit secrets: {labels}. Edit the field(s) here and click "
        "**Apply settings** to override for this session."
    )


def _pending_settings_warning(settings: Any) -> None:
    """Warn when on-screen sidebar values are not the ones a run will use.

    The API key is read live on every run, but base URL and model names only
    take effect when **Apply settings** is pressed. That asymmetry is invisible
    and bites hardest on hosted deployments (Streamlit Community Cloud has no
    ``.env``): the app then starts on the default Token Plan base URL, so a key
    pasted for a different endpoint is sent to the wrong host and fails with a
    rejected-key error that looks nothing like "you forgot to apply settings".
    """
    pending: list[str] = []
    for label, key, applied in (
        ("base URL", "base_url_input", settings.base_url),
        ("main model", "model_main_input", settings.model_main),
        ("fast model", "model_fast_input", settings.model_fast),
    ):
        value = _field_value(key)
        if value and value != str(applied or "").strip():
            pending.append(label)
    if not pending:
        return
    listed = ", ".join(pending)
    st.warning(
        f"You edited the {listed}, but it has not been applied — runs will still use the "
        f"saved {listed}. Click **Apply settings** to use the new value(s). "
        "(The API key is used immediately; the base URL and model names are not.)"
    )


def _test_connection_report(settings: Any) -> None:
    """Run the same preflight a run does, on the on-screen settings, and show its verdict.

    A listing is not a promise — Perplexity's Router API published its ids and then refused
    every completion — so checking ``GET /models`` alone let this button call a dead
    configuration healthy. Going through :func:`app.preflight.check_endpoint` means the whole
    preflight runs here too: the same two probes and the same policy that decides whether a
    run starts, so the button cannot promise something the run buttons will refuse. It is
    still the screen where a bad key or endpoint is fixed in seconds, which is why the
    verdict is rendered whole — headline and fix. The verdict is also remembered for
    the run gate, which would otherwise repeat the same two requests seconds later
    (see :data:`app.preflight.VERDICT_REUSE_SECONDS`).
    """
    base_url = _field_value("base_url_input", settings.base_url)
    api_key = _field_value("api_key_input", settings.api_key)
    model_main = _field_value("model_main_input", settings.model_main)
    model_fast = _field_value("model_fast_input", settings.model_fast)

    if not api_key:
        st.error("Enter an API key first, then test the connection.")
        return

    probe_settings = _with_on_screen_fields(
        settings,
        base_url=base_url,
        api_key=api_key,
        model_main=model_main,
        model_fast=model_fast,
    )
    with st.spinner("Checking the endpoint — the model listing and a real call…"):
        verdict = check_endpoint(probe_settings)

    # Hand the run gate this verdict: it checks the same configuration seconds later and
    # should not pay for the same two requests while the evidence is fresh.
    remember_verdict(st.session_state, probe_settings, verdict)

    message = verdict.headline if not verdict.fix else f"{verdict.headline}\n\n{verdict.fix}"
    if verdict.kind == OK:
        st.success(f"✅ {message}")
    elif verdict.kind == BLOCKED:
        st.error(f"⛔ {message}")
    else:
        # UNVERIFIED: the run is allowed, but the check could not tell — say so.
        st.warning(f"⚠️ {message}")
    _render_preflight_evidence(verdict, base_url)


def _with_on_screen_fields(settings: Any, **fields: str) -> Any:
    """*settings* with the sidebar's field values applied, whatever object it is.

    ``Settings`` is a mutable dataclass, so it is rebuilt with
    :func:`dataclasses.replace`; the views' tests hand in a ``SimpleNamespace``, so the
    generic path copies it instead. The preflight reads the base URL, the key and the
    two model ids, and nothing else.
    """
    if is_dataclass(settings) and not isinstance(settings, type):
        return replace(settings, **fields)
    shadow = copy(settings)
    for name, value in fields.items():
        setattr(shadow, name, value)
    return shadow


def _render_preflight_evidence(verdict: Verdict, base_url: str) -> None:
    """What the check compared, under the verdict — evidence, not a second opinion."""
    parts = [f"endpoint `{base_url.rstrip('/')}`" if base_url else "no base URL set"]
    if verdict.listed:
        parts.append(f"{verdict.listed:,} models listed")
    if verdict.checked:
        parts.append("checked " + ", ".join(f"`{m}`" for m in verdict.checked))
    retired_kind = getattr(verdict, "retired_kind", "")
    if retired_kind:
        # The verdict carries the retired finding even when a green one — a healthy
        # probe on a retired provider is exactly the trap — so the evidence names it.
        label = RETIRED_PROVIDER_KINDS.get(retired_kind, retired_kind)
        parts.append(f"⚠️ retired provider: {label}")
    st.caption("Preflight: " + " · ".join(parts) + ".")


def _compat_model_warning(settings: Any) -> None:
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
        available = check_model_availability(settings.base_url, settings.api_key)
    except Exception:  # noqa: BLE001 - never break the UI on a compat check
        available = None
    warnings: list[str] = []
    if available is not None:
        for label, model in (
            ("Main model", settings.model_main),
            ("Fast model", settings.model_fast),
        ):
            if model and model not in available:
                # Reported (and given its reference) at the moment the warning is
                # computed, then cached as shown text: the cached replay below must
                # quote the same id, and must not log a second time.
                warnings.append(
                    report_failure(
                        f"⚠️ {label} `{model}` not found at `{settings.base_url.rstrip('/')}/models`. "
                        "The provider may have deprecated it — check `COMPATIBILITY.md` and `MIGRATION.md`.",
                        phase="compat_model_availability",
                        severity="warning",
                        once=True,
                    )
                )
    st.session_state["_compat_checked_sig"] = sig
    st.session_state["_compat_warnings"] = warnings
    for msg in warnings:
        st.warning(msg)


def _retired_endpoint_warning(settings: Any) -> None:
    """Banner when the configured endpoint or models belong to a retired provider.

    Vercel AI Gateway and Perplexity's Router each worked here once and were retired,
    and both fail runs in ways a pre-run check can name without any request: the
    gateway's free tier rate-limits per model (a measured 329-chunk run lost 315
    chunks to `429` in under a minute) and the Router refuses completions without
    private-preview entitlement. ``retired_endpoint_reason`` is pure string matching,
    so the banner renders on every paint with no network call and no session state —
    visible before the user reaches for a run button, which is the point.
    """
    reason = retired_endpoint_reason(settings)
    if reason:
        st.warning(reason)


def _credit_calibration_widget() -> None:
    """Sidebar: record console readings and surface the learned rate."""
    with st.expander("🎚️ Usage watchdog (credit rate)"):
        history = load_usage_history()
        fit = watchdog.fit_effective_rate(history)
        st.caption(
            "Record the cumulative figure your provider console shows — the unit is "
            "yours (credits, spend), and the app fits a rate in it."
        )

        last_credits = st.session_state.get("watchdog_last_credits", "")
        credits = st.text_input(
            "Total credits used (from your provider console)",
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
                save_usage_history(history)
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
            enabled = config.CREDITS_PER_1M_MAIN is None and config.CREDITS_PER_1M_FAST is None
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
