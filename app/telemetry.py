"""Agiloop Inspect telemetry — feature-id-neutral instrumentation helper.

This module is shared infrastructure: it never hardcodes a feature id. Every
call site supplies its own ``featureId`` (the ``uuid`` from
``.implement/work-breakdown.json`` for whichever feature is calling it).

Because this is a single-origin Streamlit app (no browser JS bundle), the
Inspect API key never leaves the Python process — there is no client/server
split to bridge, so the "server-only secret" rule is satisfied by construction
as long as the key is only read here, from server-side environment variables,
and never rendered into the page.

Real vs. mock: telemetry is best-effort in both cases. When
``AGILOOP_INSPECT_API_KEY``/``AGILOOP_PROJECT_ID`` are unset, events are
logged at debug level and dropped instead of sent — this is a mock/no-op sink,
not a hard requirement for the app to run. Sending is fire-and-forget on a
background thread so a slow/broken telemetry endpoint can never block the UI.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from http.client import HTTPConnection, HTTPSConnection
from typing import Any
from urllib.parse import urlencode, urlparse

logger = logging.getLogger("agiloop.telemetry")

DEFAULT_INSPECT_URL = "https://inspect.api.agiloop.app"
REQUEST_TIMEOUT_SECONDS = 5.0

_session_id: str | None = None
_user_id: str | None = None
_initialized = False
_lock = threading.Lock()


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def _inspect_configured() -> tuple[bool, str]:
    """Check whether telemetry can reach a real Inspect endpoint.

    Returns ``(configured, reason)``. Mock mode (configured=False) activates
    solely on missing env vars — never on NODE_ENV/build-mode equivalents.
    """
    api_key = _env("AGILOOP_INSPECT_API_KEY")
    project_id = _env("AGILOOP_PROJECT_ID")
    missing = [
        name
        for name, value in (
            ("AGILOOP_INSPECT_API_KEY", api_key),
            ("AGILOOP_PROJECT_ID", project_id),
        )
        if not value
    ]
    if missing:
        return False, f"mock (missing: {', '.join(missing)})"
    return True, "real"


def init_telemetry(user_identifier: str | None = None) -> None:
    """Initialize telemetry for the current process/session. Idempotent.

    Call once at app startup (or once per Streamlit session). Safe to call
    repeatedly — later calls are no-ops once a session is active.
    """
    global _session_id, _user_id, _initialized
    with _lock:
        if _initialized:
            return
        _user_id = (user_identifier or "").strip() or f"anon-{uuid.uuid4()}"
        _session_id = str(uuid.uuid4())
        _initialized = True

    configured, reason = _inspect_configured()
    logger.info("[ENV] agiloop-inspect: %s", reason)
    _dispatch({"type": "app.launch"})


def end_session() -> None:
    """End the current telemetry session."""
    global _session_id, _user_id, _initialized
    with _lock:
        if not _initialized:
            return
        _initialized = False
        _session_id = None
        _user_id = None


def track_impression(feature_id: str, entry_point: str | None = None, **metadata: Any) -> None:
    """Track a feature impression — call when the feature UI becomes visible."""
    payload_metadata: dict[str, Any] = dict(metadata)
    if entry_point is not None:
        payload_metadata["entryPoint"] = entry_point
    _dispatch({"type": "feature.impression", "featureId": feature_id, "metadata": payload_metadata})


def track_interaction(feature_id: str, metadata: dict[str, Any] | None = None) -> None:
    """Track a feature interaction — call on user actions that indicate real use."""
    _dispatch({"type": "feature.interaction", "featureId": feature_id, "metadata": metadata or {}})


def track_feature_error(feature_id: str, error: BaseException, **metadata: Any) -> None:
    """Track a feature-scoped error. Never raises."""
    payload_metadata: dict[str, Any] = {"errorMessage": str(error), **metadata}
    _dispatch({"type": "feature.error", "featureId": feature_id, "metadata": payload_metadata})


def track_app_error(error: BaseException) -> None:
    """Track an app-level (root error boundary) error. Never raises."""
    _dispatch({"type": "app.error", "metadata": {"errorMessage": str(error)}})


def track_goal(goal_description: str, feature_id: str | None = None) -> None:
    """Track a custom goal completion."""
    event: dict[str, Any] = {"type": "goal.reached", "metadata": {"goalDescription": goal_description}}
    if feature_id:
        event["featureId"] = feature_id
    _dispatch(event)


# ------------------------------------------------------------------ internal
def _dispatch(event: dict[str, Any]) -> None:
    """Fire-and-forget event send. Never raises, never blocks the caller."""
    if not _initialized:
        init_telemetry()

    payload = {
        **event,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "userId": _user_id,
        "sessionId": _session_id,
    }
    thread = threading.Thread(target=_send, args=(payload,), daemon=True)
    thread.start()


def _send(payload: dict[str, Any]) -> None:
    configured, _reason = _inspect_configured()
    if not configured:
        logger.debug("telemetry (mock, dropped): %s", payload.get("type"))
        return
    try:
        api_key = _env("AGILOOP_INSPECT_API_KEY")
        project_id = _env("AGILOOP_PROJECT_ID")
        base_url = _env("AGILOOP_INSPECT_URL") or DEFAULT_INSPECT_URL
        feature_id = payload.get("featureId")
        query = f"?{urlencode({'featureId': feature_id})}" if feature_id else ""
        parsed = urlparse(base_url)
        if not parsed.hostname:
            return
        connection_cls = HTTPSConnection if parsed.scheme == "https" else HTTPConnection
        connection = connection_cls(parsed.hostname, parsed.port, timeout=REQUEST_TIMEOUT_SECONDS)
        try:
            connection.request(
                "POST",
                f"{parsed.path or ''}/{project_id}/event{query}",
                body=json.dumps(payload),
                headers={"Content-Type": "application/json", "X-API-Key": api_key},
            )
            connection.getresponse().read()
        finally:
            connection.close()
    except Exception:  # noqa: BLE001 - telemetry must never break the app
        logger.debug("telemetry send failed", exc_info=True)
