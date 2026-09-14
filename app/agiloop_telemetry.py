"""Agiloop Inspect telemetry client for the VA Lay Statement Evaluator.

This is the Python/Streamlit equivalent of the TypeScript `lib/agiloop-telemetry.ts`
client SDK described by the `agiloop-instrumentation` skill. Streamlit apps run
entirely server-side (there is no separate browser bundle), so sending events
directly from this module already satisfies the "same-origin proxy route"
requirement in the skill: the rendered page never sees or transmits
`AGILOOP_INSPECT_API_KEY` — only this server-side module reads it from the
environment and attaches it to outbound requests.

Feature-id neutrality: this module is SHARED telemetry infrastructure. It must
never hardcode a feature id. Every public function accepts `feature_id` as a
parameter and forwards it dynamically to the event payload/query string.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)

_DEFAULT_INSPECT_URL = "https://inspect.api.agiloop.app"

_ANON_ID_KEY = "agiloop_anonymous_id"
_SESSION_ID_KEY = "agiloop_session_id"
_INITIALIZED_KEY = "agiloop_telemetry_initialized"
_ENV_LOGGED_KEY = "agiloop_telemetry_env_logged"


def _inspect_api_key() -> str:
    return os.getenv("AGILOOP_INSPECT_API_KEY", "").strip()


def _inspect_url() -> str:
    return (os.getenv("AGILOOP_INSPECT_URL", "").strip() or _DEFAULT_INSPECT_URL)


def _project_id() -> str:
    return os.getenv("AGILOOP_PROJECT_ID", "").strip()


def _missing_vars() -> list[str]:
    missing = []
    if not _inspect_api_key():
        missing.append("AGILOOP_INSPECT_API_KEY")
    if not _project_id():
        missing.append("AGILOOP_PROJECT_ID")
    return missing


def _telemetry_enabled() -> bool:
    return not _missing_vars()


def _log_env_mode_once() -> None:
    """Emit the required `[ENV] <integration>: real|mock (...)` startup line once."""
    if st.session_state.get(_ENV_LOGGED_KEY):
        return
    st.session_state[_ENV_LOGGED_KEY] = True
    missing = _missing_vars()
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


def track_impression(feature_id: str, entry_point: str | None = None) -> None:
    """Track a feature impression — call when the feature UI becomes visible."""
    metadata = {"entryPoint": entry_point} if entry_point else None
    _send_event("feature.impression", feature_id=feature_id, metadata=metadata)


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


def track_feature_error(feature_id: str, error: BaseException) -> None:
    """Track a feature-level error caught at a feature boundary."""
    _send_event(
        "feature.error",
        feature_id=feature_id,
        metadata={"errorMessage": _sanitize_error_message(error)},
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
    """Fire-and-forget telemetry send.

    Never raises — telemetry must never break the app (Rule 7). No-ops
    (mock mode) whenever `AGILOOP_INSPECT_API_KEY` or `AGILOOP_PROJECT_ID`
    is unset; this is gated solely on env-var presence, never on any
    NODE_ENV/build-flag equivalent.
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

        url = f"{_inspect_url().rstrip('/')}/{_project_id()}/event"
        if feature_id:
            url = f"{url}?featureId={feature_id}"

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-API-Key": _inspect_api_key(),
            },
            method="POST",
        )
        urllib.request.urlopen(request, timeout=3)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        # Telemetry failures must never propagate to the user.
        logger.debug("telemetry send failed for event %s", event_type, exc_info=True)
