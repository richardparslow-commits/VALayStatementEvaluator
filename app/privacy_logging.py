"""Remove source-bearing fields before standard Python logging delivers a record.

Pilot-only; custom sinks that bypass Logger.callHandlers and direct stdout still
need the actual-host canary review. This is suppression, not a PHI classifier.
"""
from __future__ import annotations

import logging
import math
import threading
from datetime import datetime, timezone

_lock = threading.Lock()
_original_call_handlers = logging.Logger.callHandlers


def _sanitize(record: logging.LogRecord) -> None:
    from .pilot import safe_metadata
    metadata = safe_metadata(record.__dict__)
    audit = record.__dict__.get("audit_payload")
    created = record.created
    if type(created) not in (int, float) or not math.isfinite(created):
        created = 0.0
    clean = logging.LogRecord("pilot", logging.INFO, "", 0,
                              "Application event", (), None)
    # Preserve severity and time, never a caller-defined level/worker name.
    level = record.levelno if type(record.levelno) is int else logging.INFO
    clean.levelno = level
    clean.levelname = {10: "DEBUG", 20: "INFO", 30: "WARNING", 40: "ERROR",
                       50: "CRITICAL"}.get(level, "EVENT")
    clean.created = created
    clean.threadName = "worker"
    clean.processName = "process"
    clean.__dict__.update(metadata)
    if isinstance(audit, dict):
        clean.__dict__["audit_payload"] = {
            "timestamp": datetime.fromtimestamp(created, timezone.utc).isoformat(),
            **safe_metadata(audit),
        }
    record.__dict__.clear()
    record.__dict__.update(clean.__dict__)


def _call_handlers(logger: logging.Logger, record: logging.LogRecord) -> None:
    from . import pilot
    if pilot.enabled():
        try:
            _sanitize(record)
        except Exception:  # noqa: BLE001 - malformed metadata must never restore raw fields
            clean = logging.LogRecord("pilot", logging.ERROR, "", 0,
                                      "Application event", (), None)
            clean.threadName, clean.processName = "worker", "process"
            record.__dict__.clear()
            record.__dict__.update(clean.__dict__)
    _original_call_handlers(logger, record)


def install() -> None:
    """Install once, before handlers fan out; later standard handlers are covered."""
    with _lock:
        if logging.Logger.callHandlers is not _call_handlers:
            setattr(logging.Logger, "callHandlers", _call_handlers)
