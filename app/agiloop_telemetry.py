"""Agiloop Inspect telemetry client for the VA Lay Statement Evaluator.

This is the Python/Streamlit equivalent of the TypeScript `lib/agiloop-telemetry.ts`
client SDK described by the `agiloop-instrumentation` skill: it is the module
every feature call site imports to fire impression/interaction/error/goal
events. It contains NO reference to `AGILOOP_INSPECT_API_KEY` or any other
Inspect secret — building the outbound payload here and physically attaching
the API key happen in two different modules, mirroring the browser-SDK /
same-origin-proxy split the skill describes for split-origin apps. The actual
network call (and the only place the API key is read) lives in
`app/telemetry_proxy.py`; this module only ever calls
`telemetry_proxy.forward_event(payload)`.

Feature-id neutrality: this module is SHARED telemetry infrastructure. It must
never hardcode a feature id. Every public function accepts `feature_id` as a
parameter and forwards it dynamically to the event payload/query string.
"""
from __future__ import annotations

import logging
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any

import streamlit as st

from . import telemetry_proxy

logger = logging.getLogger(__name__)

_ANON_ID_KEY = "agiloop_anonymous_id"
_SESSION_ID_KEY = "agiloop_session_id"
_INITIALIZED_KEY = "agiloop_telemetry_initialized"
_ENV_LOGGED_KEY = "agiloop_telemetry_env_logged"


def _telemetry_enabled() -> bool:
    return telemetry_proxy.is_configured()


def _log_env_mode_once() -> None:
    """Emit the required `[ENV] <integration>: real|mock (...)` startup line once."""
    if st.session_state.get(_ENV_LOGGED_KEY):
        return
    st.session_state[_ENV_LOGGED_KEY] = True
    missing = telemetry_proxy.missing_env_vars()
    if missing:
        logger.info("[ENV] agiloop-telemetry: mock (missing: %s)", ", ".join(missing))
    else:
        logger.info("[ENV] agiloop-telemetry: real")


def init_telemetry() -> None:
    """Initialize telemetry for the current Streamlit session.

    Call once at the app root (see `app/main.py::main`). Idempotent across
    Streamlit reruns — safe to call on every script run since Streamlit
    re-executes the whole script on every interaction.
    """
    _log_env_mode_once()
    if st.session_state.get(_INITIALIZED_KEY):
        return
    st.session_state[_ANON_ID_KEY] = st.session_state.get(_ANON_ID_KEY, str(uuid.uuid4()))
    st.session_state[_SESSION_ID_KEY] = st.session_state.get(_SESSION_ID_KEY, str(uuid.uuid4()))
    st.session_state[_INITIALIZED_KEY] = True
    _send_event("app.launch", feature_id=None, metadata=None)


def _user_id() -> str:
    return str(st.session_state.get(_ANON_ID_KEY, "unknown"))


def _session_id() -> str:
    return str(st.session_state.get(_SESSION_ID_KEY, "unknown"))


def track_impression(feature_id: str, entry_point: str | None = None, **attributes: Any) -> None:
    """Track a feature impression — call when the feature UI becomes visible."""
    metadata: dict[str, Any] = dict(attributes)
    if entry_point:
        metadata["entryPoint"] = entry_point
    _send_event("feature.impression", feature_id=feature_id, metadata=metadata or None)


def track_interaction(feature_id: str, **attributes: Any) -> None:
    """Track a feature interaction — call on key user actions (not every click)."""
    _send_event("feature.interaction", feature_id=feature_id, metadata=attributes or None)


_MAX_ERROR_MESSAGE_LEN = 200


def _sanitize_error_message(error: BaseException) -> str:
    """Bound the outbound error message to a short, non-content-bearing summary.

    Some internal exceptions (e.g. LLM chunk-digest failures) can embed
    fragments of the underlying medical-record text or other user-supplied
    content in their `str()` representation. Telemetry must never carry PII
    or record content off the server (see Telemetry Leakage Rules), so we
    only forward the exception type plus a short, truncated message — enough
    for triage, not enough to leak substantive record content.
    """
    message = str(error).replace("\n", " ").replace("\r", " ").strip()
    truncated = message[:_MAX_ERROR_MESSAGE_LEN]
    if len(message) > _MAX_ERROR_MESSAGE_LEN:
        truncated += "…[truncated]"
    return f"{type(error).__name__}: {truncated}" if truncated else type(error).__name__


_MAX_ERROR_STACK_LEN = 2000


def _sanitize_error_stack(error: BaseException) -> str:
    """Bound the outbound stack trace similarly to the error message.

    `traceback.format_exception` includes file/line/function context and the
    literal source line, but never runtime variable values — so it cannot leak
    the *content* of a statement/record that was being processed when the
    error occurred. It is still truncated defensively so a single event
    cannot balloon in size.
    """
    try:
        frames = traceback.format_exception(type(error), error, error.__traceback__)
    except Exception:  # noqa: BLE001 - formatting the stack must never raise
        return ""
    joined = "".join(frames).replace("\r", "")
    if len(joined) > _MAX_ERROR_STACK_LEN:
        return joined[:_MAX_ERROR_STACK_LEN] + "…[truncated]"
    return joined


def track_feature_error(feature_id: str, error: BaseException, **attributes: Any) -> None:
    """Track a feature-level error caught at a feature boundary.

    ``**attributes`` accepts additional non-PII context (e.g. ``stage``) that
    callers want attached alongside the sanitized message/stack — merged in
    without changing any existing 2-argument call site's behavior.
    """
    _send_event(
        "feature.error",
        feature_id=feature_id,
        metadata={
            "errorMessage": _sanitize_error_message(error),
            "errorStack": _sanitize_error_stack(error),
            **attributes,
        },
    )


def track_app_error(error: BaseException) -> None:
    """Track an app-level (root error boundary) error. No feature id attached."""
    _send_event(
        "app.error", feature_id=None, metadata={"errorMessage": _sanitize_error_message(error)}
    )


def track_goal(feature_id: str, goal_description: str, **attributes: Any) -> None:
    """Track a custom, measurable goal reached within a feature."""
    metadata = {"goalDescription": goal_description, **attributes}
    _send_event("goal.reached", feature_id=feature_id, metadata=metadata)


def _send_event(
    event_type: str, *, feature_id: str | None, metadata: dict[str, Any] | None
) -> None:
    """Build the event payload and hand it to the telemetry proxy.

    Never raises — telemetry must never break the app (Rule 7). The proxy
    itself no-ops (mock mode) whenever `AGILOOP_INSPECT_API_KEY` or
    `AGILOOP_PROJECT_ID` is unset; this module never reads those vars or the
    API key directly (see module docstring).
    """
    if not _telemetry_enabled():
        logger.debug("telemetry mock mode — dropping event %s", event_type)
        return
    try:
        payload: dict[str, Any] = {
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "userId": _user_id(),
            "sessionId": _session_id(),
        }
        if feature_id:
            payload["featureId"] = feature_id
        if metadata:
            payload["metadata"] = metadata
        telemetry_proxy.forward_event(payload)
    except Exception:  # noqa: BLE001 - telemetry must never propagate to the user
        logger.debug("telemetry send failed for event %s", event_type, exc_info=True)
