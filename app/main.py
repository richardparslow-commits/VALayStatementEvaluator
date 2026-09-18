"""VA Lay Statement Evaluator — Streamlit application entry point.

Run with:
    streamlit run app/main.py

This module is a thin router: it configures logging/audit, renders the app
chrome (title, tabs, sidebar), delegates each tab to the view layer in
``app/views/``, and guards everything with a root error boundary. Business
logic lives in ``app/evaluate.py`` / ``app/draft.py``; tab rendering and
run orchestration live in ``app/views/*``.
"""
from __future__ import annotations

import traceback

import streamlit as st

from . import audit as audit_log
from . import telemetry
from . import tracing
from .logging_config import (
    configure_logging,
    get_logger,
    get_request_id,
)
from .run_log import run_log_event
from .views.about_view import render_about_tab
from .views.draft_view import render_draft_tab
from .views.evaluate_view import render_evaluate_tab
from .views.shared import (
    REQUEST_ID_KEY,
    format_error_for_user,
    get_or_create_request_id,
    report_failure,
)
from .views.sidebar import render_sidebar_settings

logger = get_logger("app.main")

st.set_page_config(
    page_title="VA Lay Statement Evaluator",
    page_icon="🎖️",
    layout="wide",
)


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

        cfg = _P(__file__).resolve().parent.parent / ".streamlit" / "config.toml"
        text = cfg.read_text(encoding="utf-8") if cfg.exists() else ""
        missing: list[str] = []
        if "enableXsrfProtection" not in text:
            missing.append("server.enableXsrfProtection")
        if "toolbarMode" not in text:
            missing.append("client.toolbarMode")
        if "maxUploadSize" not in text:
            missing.append("server.maxUploadSize")
        if missing:
            st.warning(
                report_failure(
                    "⚠️ Streamlit security hardening is not fully active (missing from "
                    ".streamlit/config.toml: "
                    + ", ".join(missing)
                    + "). The app still works, but XSRF protection, toolbar hardening, and "
                    "upload caps depend on that file — see README → Production hardening.",
                    phase="security_hardening",
                    severity="warning",
                    once=True,  # main() paints on every rerun
                )
            )
    except Exception:  # noqa: BLE001 - hardening check must never break the app
        pass


# --------------------------------------------------------------------- layout
def main() -> None:
    configure_logging()
    try:
        audit_log.configure_audit_logging()
    except Exception:  # noqa: BLE001 - audit is best-effort
        pass
    # No-op unless VA_LSE_TRACING=1 (and the OTel packages are installed): the
    # provider is process-global, so one call here covers every session.
    tracing.setup_tracing(role="web")
    _check_streamlit_config_hardening()
    # Ensure every browser session has a baseline correlation id (also used
    # for pre-run validation / upload errors so those logs are correlatable).
    try:
        get_or_create_request_id()
    except Exception:  # noqa: BLE001
        pass
    # Stamp this browser session for va_lse_session_count. Streamlit re-executes
    # this function on every widget interaction, which is what makes a per-run stamp
    # an accurate "active recently" signal — the session id is the same one the
    # audit stream already carries, so there is no second identity to keep in sync.
    try:
        from .metrics import touch_session
        from .audit import get_audit_session_id

        touch_session(get_audit_session_id())
    except Exception:  # noqa: BLE001 - instrumentation is best-effort by design
        pass
    logger.info(
        "app start",
        extra={"request_id": get_request_id() or "-", "phase": "app", "status": "start"},
    )
    telemetry.init_telemetry()
    render_sidebar_settings()
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
            render_evaluate_tab()
        with tab_draft:
            render_draft_tab()
        with tab_about:
            render_about_tab()
    except Exception as exc:  # noqa: BLE001 - root error boundary
        rid = get_request_id() or st.session_state.get(REQUEST_ID_KEY, "-") or "-"
        logger.error(
            "unhandled app error: %s",
            f"{type(exc).__name__}: {exc}",
            exc_info=exc,
            extra={
                "request_id": rid,
                "phase": "app",
                "status": "error",
                "error_class": type(exc).__name__,
            },
        )
        run_log_event(
            "app", "error", request_id=rid,
            error=f"{type(exc).__name__}: {exc}", error_class=type(exc).__name__,
            traceback=traceback.format_exc(limit=8),
        )
        telemetry.track_app_error(exc)
        st.error(
            f"Something went wrong while rendering the app: {format_error_for_user(exc, rid)}"
        )


if __name__ == "__main__":
    main()
