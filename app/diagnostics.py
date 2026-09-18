"""Resolve a ``req_…`` reference to the lines behind it, from inside the app.

The About tab has always told the user to run ``grep req_… logs/runs.jsonl``.
That advice assumes a shell on the pod, which is precisely what the person
looking at a failed run does not have. This module answers the same question from
inside the app, where the user's own session is the authorization. Nothing is
published over HTTP: the health sidecar binds ``0.0.0.0`` with no authentication,
and a route there returning log content would hand it to anything that can reach
its port.

Two sources, because one run can span two processes:

* **This process's records.** A bounded in-process buffer (``CaptureHandler``,
  installed by ``configure_logging``) keeps the most recent records with the
  ``request_id`` each was emitted under. This is the only way to answer when
  ``VA_LSE_LOG_DIR`` is unset and the app is stdout-only, which is the default.
* **The shared run log.** ``logs/runs.jsonl`` is written by the web process *and*
  by workers, so its events are what connect a failed queued run to a reference
  when the failure happened somewhere else. That is also how the panel can say
  "this ran in a worker" instead of leaving an empty answer.

Returned text is meant to be pasted into a support conversation, so secret-shaped
strings are redacted (``redact_secrets``). Upload filenames are deliberately kept:
"which file failed" is usually the entire question, and a filename is a string the
user themself typed. That is a thing ``SECURITY.md`` is now explicit about rather
than silent on.

The cost of an answer is bounded but not free, and the bounds are honest ones. The
buffer holds ``CAPTURE_LIMIT`` records clipped to ``MAX_FIELD_CHARS`` each, so it
cannot grow past a few megabytes of process memory. The run-log half is a *scan of
the most recent* ``EVENT_SCAN`` events rather than an index — ``read_recent_events``
loads the newest file whole, which is at most ``RUN_LOG_MAX_BYTES`` (10 MB by
default) — and the panel reports how many events it examined, so an absent event is
never presented as "this did not happen".
"""
from __future__ import annotations

import logging
import re
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .logging_config import get_request_id

# Every one of these is a bound on process-lifetime memory or on how much a single
# answer can contain. An answer is meant to be read, not paged through.
CAPTURE_LIMIT = 400  # log records retained in memory
MAX_LINES = 100  # log lines returned for one reference
MAX_EVENTS = 50  # run-log events returned for one reference
EVENT_SCAN = 2000  # run-log events examined per lookup
MAX_FIELD_CHARS = 4000  # per-message and per-traceback cap

_TRUNCATION = " … [truncated]"

# Long enough that a typo cannot silently match nothing, narrow enough that a
# path, a glob, or a filename can never be mistaken for an id.
_REFERENCE_RE = re.compile(r"^req_[0-9a-f]{6,32}$")
_REFERENCE_IN_TEXT_RE = re.compile(r"req_[0-9a-fA-F]{6,32}")

_CAPTURE_MARKER = "_va_lse_capture_handler"

# Applied in order. Deliberately narrow: these are the shapes that turn a log line
# into an incident (SECURITY.md: "treat any log line containing sk-sp as an
# incident"), not a general-purpose scrubber that would gut the answer.
# Order matters, and so does the lookahead in the first rule.
#
# ``Authorization: Bearer <token>`` is the trap: a naive ``key: value`` rule treats
# the scheme word as the value, redacts the word "Bearer", and leaves the token
# sitting in plain sight right after it. So the scheme is excluded here and handled
# by the rule below, which knows the value follows the scheme rather than being it.
#
# The order also makes the result stable under re-application: ``\1=[redacted]``
# re-matches to itself, and neither of the later rules can re-match their own output.
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?i)\b(api[_-]?key|access[_-]?key|secret|password|passwd|token|authorization)"
            r"\b\s*[=:]\s*\"?(?!bearer\b|basic\b)[^\s\",;}]{6,}"
        ),
        r"\1=[redacted]",
    ),
    (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"), r"\1 [redacted]"),
    (re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{8,}"), "[redacted-key]"),
)

_TRACEBACK_FORMATTER = logging.Formatter()


def redact_secrets(text: str) -> str:
    """Replace secret-shaped strings so a line can be shared safely."""
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def _clip(text: str) -> str:
    """Cap a field at ``MAX_FIELD_CHARS`` so one bad record cannot fill memory."""
    if len(text) <= MAX_FIELD_CHARS:
        return text
    return text[:MAX_FIELD_CHARS] + _TRUNCATION


def _entry(record: logging.LogRecord) -> dict[str, Any]:
    """Snapshot one record into the plain dict the buffer stores."""
    request_id = getattr(record, "request_id", None) or get_request_id() or "-"
    traceback_text: str | None = None
    if record.exc_info:
        traceback_text = _clip("".join(traceback.format_exception(*record.exc_info)))
    return {
        "request_id": str(request_id),
        "timestamp": datetime.fromtimestamp(
            record.created, tz=timezone.utc
        ).isoformat(timespec="milliseconds"),
        "level": record.levelname,
        "logger": record.name,
        "phase": getattr(record, "phase", None),
        "status": getattr(record, "status", None),
        "error_class": getattr(record, "error_class", None),
        "message": _clip(record.getMessage()),
        "traceback": traceback_text,
    }


class CaptureHandler(logging.Handler):
    """Keep a bounded, in-process copy of the records this app emits.

    Attached to the ``app`` logger, so it sees everything the app logs regardless
    of whether file logging is configured. Records are stored as dicts rather than
    formatted strings: the reference is an attribute on the record, and filtering
    on a field is the whole job here.
    """

    def __init__(self, *, limit: int = CAPTURE_LIMIT) -> None:
        super().__init__(level=logging.NOTSET)
        self._records: deque[dict[str, Any]] = deque(maxlen=limit)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._records.append(_entry(record))
        except Exception:  # noqa: BLE001 - a diagnostic buffer must never break logging
            pass

    def snapshot(self) -> list[dict[str, Any]]:
        """A point-in-time copy, oldest first."""
        return list(self._records)

    def clear(self) -> None:
        self._records.clear()


def install_capture(
    logger: logging.Logger | None = None, *, limit: int = CAPTURE_LIMIT
) -> CaptureHandler:
    """Attach the capture handler to ``logger`` (idempotent, ``app`` by default).

    Deliberately **not** marked as a managed handler: ``configure_logging`` removes
    and rebuilds the streaming/file handlers, and the buffer exists to survive that
    — a reference looked up just after a reconfiguration should still resolve.
    """
    target = logger if logger is not None else logging.getLogger("app")
    existing = getattr(target, _CAPTURE_MARKER, None)
    if isinstance(existing, CaptureHandler) and existing in target.handlers:
        return existing
    handler = CaptureHandler(limit=limit)
    target.addHandler(handler)
    setattr(target, _CAPTURE_MARKER, handler)
    return handler


def capture_handler(logger: logging.Logger | None = None) -> CaptureHandler | None:
    """The installed capture handler, or ``None`` when capture is not active."""
    target = logger if logger is not None else logging.getLogger("app")
    existing = getattr(target, _CAPTURE_MARKER, None)
    if isinstance(existing, CaptureHandler) and existing in target.handlers:
        return existing
    return None


def extract_reference(text: str) -> str:
    """The first ``req_…`` id in ``text``, lowercased.

    People paste the whole failure, not the id, so the panel accepts
    ``"Drafting failed: … (reference: req_4f8a2b1c9d0e)"``.
    """
    match = _REFERENCE_IN_TEXT_RE.search(text or "")
    return match.group(0).lower() if match else ""


def is_reference(text: str) -> bool:
    """True for a bare, well-formed reference id."""
    return bool(_REFERENCE_RE.match((text or "").strip().lower()))


def _render_line(entry: dict[str, Any]) -> str:
    """One log line as the user would see it in the log file."""
    fields = " ".join(
        f"{key}={entry[key]}"
        for key in ("phase", "status", "error_class")
        if entry.get(key)
    )
    head = f"{entry['timestamp']}  {entry['level']:<7}  {entry['logger']}"
    if fields:
        head = f"{head}  [{fields}]"
    line = f"{head}  {entry['message']}"
    traceback_text = entry.get("traceback")
    if traceback_text:
        indented = "\n".join(f"    {part}" for part in str(traceback_text).splitlines())
        line = f"{line}\n{indented}"
    return redact_secrets(line)


def _redact_event(event: dict[str, Any]) -> dict[str, Any]:
    """Redact the string fields of a run-log event, leaving structure intact."""
    return {
        key: redact_secrets(value) if isinstance(value, str) else value
        for key, value in event.items()
    }


@dataclass(frozen=True)
class ReferenceDetail:
    """Everything the app can say about one reference id."""

    reference: str = ""
    valid: bool = False
    problem: str = ""
    lines: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    buffered: int = 0  # records the buffer held when the lookup ran
    scanned: int = 0  # run-log events examined
    truncated: bool = False  # an answer bound clipped the result
    note: str = ""  # why a source came back empty, when it did


def lookup(
    query: str, *, limit: int = MAX_LINES, events_scan: int = EVENT_SCAN
) -> ReferenceDetail:
    """Resolve ``query`` (an id or a whole error message) to its recorded lines.

    Never raises and never reads anything but this process's buffer and the run
    log: an unknown reference returns an empty, valid detail with a note.
    """
    reference = extract_reference(query) or (query or "").strip().lower()
    if not is_reference(reference):
        return ReferenceDetail(
            reference=(query or "").strip(),
            problem=(
                "That is not a reference id. It looks like `req_` followed by hex "
                "characters, for example `req_4f8a2b1c9d0e` — pasting the whole error "
                "message works too."
            ),
        )

    # Imported here, not at module scope: app/run_log.py calls get_logger() while it
    # is still executing, which runs configure_logging(), which imports this module
    # — a top-level import of run_log would then fail on a half-built module.
    from .run_log import read_recent_events

    handler = capture_handler()
    buffered = handler.snapshot() if handler is not None else []
    matched = [entry for entry in buffered if entry["request_id"] == reference]
    truncated = len(matched) > limit
    lines = [_render_line(entry) for entry in matched[-limit:]] if limit > 0 else []

    events = read_recent_events(limit=events_scan)
    matched_events = [e for e in events if e.get("request_id") == reference]
    truncated = truncated or len(matched_events) > MAX_EVENTS
    rendered_events = [_redact_event(e) for e in matched_events[-MAX_EVENTS:]]

    note = ""
    if not lines:
        if handler is None:
            note = (
                "In-process log capture is not active in this process, so only the "
                "shared run log could be searched."
            )
        elif matched_events:
            note = (
                "No log lines in **this** process, but the run log has events for this "
                "reference — the run executed elsewhere (a worker process), so its "
                "lines are on that process, or in the app log when `VA_LSE_LOG_DIR` "
                "points at a shared volume."
            )
        else:
            note = (
                "Nothing recorded for this reference in this process, and no run-log "
                "event matched it. The buffer holds only the most recent "
                f"{CAPTURE_LIMIT} records and starts empty after a restart; check the "
                "reference characters against the error message."
            )

    return ReferenceDetail(
        reference=reference,
        valid=True,
        lines=lines,
        events=rendered_events,
        buffered=len(buffered),
        scanned=len(events),
        truncated=truncated,
        note=note,
    )
