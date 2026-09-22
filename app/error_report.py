"""Attributable reporting for anything a failure shows the user.

A failure the user can see is only actionable if it can also be found in the
logs, so every user-facing failure path routes through here. The helpers
guarantee a correlation id exists, record the failure (with a traceback when an
exception is available) and return the message with that id appended.

Two view paths used to render ``str(exc)`` with neither a reference nor a log
line - the VA.gov authentication failure and the settings validation errors - so
a user who hit them had nothing to quote and there was nothing to search. That
is the shape of an unattributable failure: the message existed only on screen.

That shape was more widespread than those two. The extractor's per-file skip
messages ("it said my file was skipped") were built as plain strings and shown as
warnings, so the only record that a bundled PDF had failed to parse was the
sentence on screen, and a rejected settings Apply or an over-cap upload left the
same kind of nothing behind. All of them render through here.

Deliberately free of Streamlit and of app-config imports: ``app.views.shared``
imports the tab modules, so the tab modules cannot import back from it, and this
has to be usable from all of them.
"""
from __future__ import annotations

import threading
from typing import Any, Literal

from .logging_config import get_logger, get_request_id, new_request_id, set_request_id

logger = get_logger("app.error_report")

Severity = Literal["error", "warning"]

# Phases already reported in this process, for callers that opt into ``once``.
# Bounded because these live for the process lifetime.
#
# Guarded by a lock because ``report_failure(once=True)`` is called while
# *rendering*, and Streamlit runs each session's script on its own thread — two
# browser tabs repainting the same broken sidebar are two threads inside
# ``_mark_once``. The lock makes the check-then-add atomic, so "first time" means
# first time rather than first-to-win; it is also what keeps this correct on a
# free-threaded interpreter.
_ONCE_SEEN: set[tuple[str, str]] = set()
_ONCE_LOCK = threading.Lock()
_ONCE_LIMIT = 512


def reference_suffix(request_id: str) -> str:
    """The user-facing reference suffix for a correlation id (``""`` when absent)."""
    return f" (reference: {request_id})" if request_id and request_id != "-" else ""


def format_error_for_user(exc: Exception, request_id: str) -> str:
    """User-facing error string that carries the correlation id without PII."""
    return f"{exc}{reference_suffix(request_id)}"


def ensure_request_id() -> str:
    """The active correlation id, minting and activating one when absent.

    For paths that already record the failure themselves (the run log's
    ``rejected`` events) and only need the id to agree on both ends.
    """
    rid = get_request_id() or new_request_id()
    set_request_id(rid)
    return rid


def report_failure(
    message: str,
    *,
    phase: str,
    exc: BaseException | None = None,
    severity: Severity = "error",
    request_id: str | None = None,
    once: bool = False,
) -> str:
    """Record a failure, and return the message the user should read.

    ``message`` is shown verbatim apart from the appended reference, so each
    caller keeps wording that fits its own context. ``phase`` labels where the
    failure happened for the structured log, and ``exc`` supplies the traceback
    when one exists.

    When no correlation id is active one is minted, so the returned reference
    still resolves to the log line this call just wrote. That is the point: a
    failure reported here is findable afterwards rather than being a sentence
    that existed only on screen. Callers that already hold an id for the run
    (``job_runner``, the run pipeline) pass it in so both ends agree.

    ``once`` is for failures raised while *rendering* rather than in response to
    an action. Sidebar panels repaint on every Streamlit rerun, so a condition
    that persists (an unreachable queue, a broken failover config) would write
    the same error and traceback once per interaction and bury the log. With
    ``once`` the first occurrence in the process is logged and later renders
    reuse its id - which still resolves, because it is the same failure.
    """
    rid = request_id or ensure_request_id()
    extra: dict[str, Any] = {
        "request_id": rid,
        "phase": phase,
        "status": severity,
        "error_class": type(exc).__name__ if exc is not None else None,
    }
    if once and not _mark_once(phase, message):
        return f"{message}{reference_suffix(rid)}"
    if severity == "warning":
        logger.warning("%s", message, exc_info=exc, extra=extra)
    else:
        logger.error("%s", message, exc_info=exc, extra=extra)
    return f"{message}{reference_suffix(rid)}"


def _mark_once(phase: str, message: str) -> bool:
    """True the first time this (phase, message) is seen in the process."""
    key = (phase, message)
    with _ONCE_LOCK:
        if key in _ONCE_SEEN:
            return False
        if len(_ONCE_SEEN) >= _ONCE_LIMIT:
            _ONCE_SEEN.clear()
        _ONCE_SEEN.add(key)
        return True
