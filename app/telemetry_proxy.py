"""Server-side Agiloop Inspect telemetry proxy.

Per the ``agiloop-instrumentation`` skill, telemetry infrastructure is split
into:

- a client/app SDK (``app/agiloop_telemetry.py``) that has NO knowledge of
  the Inspect API key, and
- this module, the *only* place ``AGILOOP_INSPECT_API_KEY`` is read from the
  environment and attached to an outbound request.

This is a single-origin Streamlit app — there is no separate browser bundle,
so there is no literal HTTP ``/api/telemetry`` route to stand up. The
equivalent same-origin boundary here is a Python-level one:
``app/agiloop_telemetry.py`` (the module every feature call site imports)
never touches the network or the API key directly; it builds the event
payload and hands it to :func:`forward_event`, which is the only function in
the codebase that opens a socket to Inspect and attaches ``X-API-Key``.

Feature-id neutrality: this module is shared infrastructure. It never
hardcodes a feature id — :func:`forward_event` forwards whatever
``featureId`` (if any) is already present on the payload it is given.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger("app.telemetry_proxy")

_DEFAULT_INSPECT_URL = "https://inspect.api.agiloop.app"


def _inspect_api_key() -> str:
    return os.getenv("AGILOOP_INSPECT_API_KEY", "").strip()


def _inspect_url() -> str:
    return os.getenv("AGILOOP_INSPECT_URL", "").strip() or _DEFAULT_INSPECT_URL


def _project_id() -> str:
    return os.getenv("AGILOOP_PROJECT_ID", "").strip()


def missing_env_vars() -> list[str]:
    """Return the required env vars that are currently unset (empty when configured)."""
    missing = []
    if not _inspect_api_key():
        missing.append("AGILOOP_INSPECT_API_KEY")
    if not _project_id():
        missing.append("AGILOOP_PROJECT_ID")
    return missing


def is_configured() -> bool:
    """True when telemetry can reach a real Inspect endpoint (all required env vars set).

    This is the sole gate between real and mock mode — it is never gated on
    ``NODE_ENV``/build-mode equivalents, only on env-var presence, per the
    Env-Optional Runtime contract.
    """
    return not missing_env_vars()


def forward_event(payload: dict[str, Any]) -> None:
    """Forward an already-built telemetry event payload to Inspect.

    The ``X-API-Key`` header is attached here, server-side, and the key
    itself never leaves this function. No-ops (mock mode) when
    :func:`is_configured` is False. Never raises — a broken/slow telemetry
    endpoint must never propagate to the user (Rule 7 of the
    agiloop-instrumentation skill).
    """
    if not is_configured():
        logger.debug("telemetry proxy: mock mode — dropping event %s", payload.get("type"))
        return
    try:
        feature_id = payload.get("featureId")
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
        logger.debug("telemetry proxy: send failed for %s", payload.get("type"), exc_info=True)
