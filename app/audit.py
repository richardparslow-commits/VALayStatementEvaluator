"""Audit logging for Evaluate and Draft workflows.

Separate from diagnostic ``app`` logging (``app.log``) — this module emits a
dedicated JSON-lines stream to ``audit.log`` so audit records can be queried
and retained under a different policy from debug logs.

Each audit entry is a single JSON line with:

* ``timestamp``      — ISO-8601 UTC
* ``action``         — ``evaluate`` | ``draft``
* ``status``         — ``start`` | ``ok`` | ``error``
* ``request_id``     — correlation id for the run (``req_…``)
* ``user_session_id``— stable per-browser-session id (``sess_…``)
* ``condition``      — claimed condition classification (truncated, never PII text)
* ``record_sources`` — list of source labels (e.g. ``["Upload", "VA.gov"]``)
* ``record_files`` / ``record_pages`` — counts only, never file content
* ``duration_ms``    — for ``ok``/``error``
* ``outcome``        — small classification dict (e.g. ``overall_rating``), never statement text
* ``error_class`` / ``error_message`` — for ``error`` only, user-facing, truncated

Privacy contract — audit entries **never** contain: statement / observations /
record text / veteran or witness names / file content. Only counts,
classifications and source labels are recorded.

One field sits outside that guarantee and is called out deliberately:
``error_message`` is arbitrary ``str(exc)`` from an upstream library, and a
library can put anything in an exception message. It is scrubbed of
PII-shaped tokens and whitespace-collapsed, and it can be omitted entirely with
``VA_LSE_AUDIT_ERROR_MESSAGES=0`` — which is what a deployment shipping audit
logs to third-party storage should do (see DEPLOYMENT.md → Audit log backup).
``error_class`` is always recorded and is always safe.

Configuration via env (see ``.env.example``):

* ``VA_LSE_AUDIT_LOG_DIR``       — directory for ``audit.log`` (default ``logs``,
  or ``VA_LSE_LOG_DIR`` when set; empty → stdout-only fallback)
* ``VA_LSE_AUDIT_LOG_FILE``      — file name (default ``audit.log``)
* ``VA_LSE_AUDIT_LOG_MAX_BYTES`` — rotate size (default 10 MiB)
* ``VA_LSE_AUDIT_LOG_BACKUPS``   — rotated files kept (default 10)

``configure_audit_logging`` is idempotent; ``audit_event`` is best-effort and
never raises. ``get_audit_session_id`` mints a per-browser-session id in
``st.session_state`` when Streamlit is available, otherwise falls back to a
process-global id.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import re
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from . import config as _config

_AUDIT_LOGGER_NAME = "audit"
_CONFIGURED = False
_CONFIGURED_KEY = "_va_lse_audit_configured"

# Fallback in-memory session id when Streamlit is unavailable (tests / CLI).
_process_session_id: str | None = None

# ---------------------------------------------------------------------------
# Write-failure visibility.
#
# ``logging`` swallows handler-level exceptions (``Handler.handleError`` prints
# to stderr at most, and only when ``raiseExceptions`` is on), and every call
# here is wrapped in a blanket ``except``. So when the log volume filled, the app
# kept serving and *silently stopped auditing* — the worst possible outcome for a
# compliance stream. Counting handler failures turns that into something
# ``/health`` can report.
# ---------------------------------------------------------------------------
_write_lock = threading.Lock()
_write_failures = 0
_last_write_error = ""

# PII-shaped tokens that can turn up inside an arbitrary exception message.
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_LONG_DIGITS_RE = re.compile(r"\b\d{7,}\b")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_NEWLINE_RE = re.compile(r"\s+")


def _note_write_failure(exc: BaseException) -> None:
    global _write_failures, _last_write_error  # noqa: PLW0603
    with _write_lock:
        _write_failures += 1
        _last_write_error = f"{type(exc).__name__}: {exc}"[:300]


def _scrub_error_message(text: str) -> str:
    """Collapse whitespace and redact PII-shaped tokens from free-text errors."""
    scrubbed = _NEWLINE_RE.sub(" ", text or "").strip()
    for pattern in (_SSN_RE, _EMAIL_RE, _LONG_DIGITS_RE):
        scrubbed = pattern.sub("[redacted]", scrubbed)
    return _safe_truncate(scrubbed, 300)

# ``research`` covers the Perplexity Agent API panel (app/perplexity_agent.py): a
# web-grounded lookup against the legal framework, which shares this stream's
# metadata-only contract (preset, source count, citation count — never the
# question text or an answer).
Action = Literal["evaluate", "draft", "research"]
Status = Literal["start", "ok", "error"]


# ------------------------------------------------------------------ helpers


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name, "")
    if raw is None:
        return default
    raw = raw.strip()
    return raw if raw else default


def _resolve_audit_dir(explicit: str | Path | None) -> str:
    if explicit is not None:
        return str(explicit).strip()
    # Explicit audit dir wins; else fall back to diagnostic log dir; else "logs".
    audit_env = os.getenv("VA_LSE_AUDIT_LOG_DIR", "").strip()
    if audit_env:
        return audit_env
    diag_env = os.getenv("VA_LSE_LOG_DIR", "").strip()
    if diag_env:
        return diag_env
    return "logs"


def _safe_truncate(text: str, limit: int = 120) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


# -------------------------------------------------------------- configuration


def configure_audit_logging(
    *,
    log_dir: str | Path | None = None,
    log_file: str | None = None,
    max_bytes: int | None = None,
    backups: int | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the dedicated ``audit`` logger (idempotent).

    The logger writes JSON lines to ``{log_dir}/{log_file}`` with rotation and
    does **not** propagate to the ``app`` parent — it is a separate stream for
    retention/compliance tooling. A stdout handler is always present so
    ``docker logs`` / platform drains see the stream even when file creation
    fails.
    """
    global _CONFIGURED  # noqa: PLW0603

    if _CONFIGURED and not force:
        return logging.getLogger(_AUDIT_LOGGER_NAME)

    # Resolve settings — explicit args win, otherwise env / config defaults.
    raw_dir = _resolve_audit_dir(log_dir)
    raw_file = log_file if log_file is not None else _env_str("VA_LSE_AUDIT_LOG_FILE", "audit.log")
    raw_file = raw_file.strip() or "audit.log"

    if max_bytes is None:
        try:
            raw_max_str = os.getenv("VA_LSE_AUDIT_LOG_MAX_BYTES", "").strip()
            raw_max = int(raw_max_str) if raw_max_str else int(_config.AUDIT_LOG_MAX_BYTES)
        except (ValueError, AttributeError):
            raw_max = 10 * 1024 * 1024
    else:
        raw_max = max_bytes

    if backups is None:
        try:
            raw_backups_str = os.getenv("VA_LSE_AUDIT_LOG_BACKUPS", "").strip()
            raw_backups = int(raw_backups_str) if raw_backups_str else int(_config.AUDIT_LOG_BACKUPS)
        except (ValueError, AttributeError):
            raw_backups = 10
    else:
        raw_backups = backups

    # JSON formatter — single line per audit entry, no diagnostic extras.
    class _FailureCountingHandler(logging.Handler):
        """Mixin that records handler-level failures (full disk, read-only mount)."""

        def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802 - logging API
            exc = sys.exc_info()[1]
            _note_write_failure(exc if exc is not None else RuntimeError("unknown logging failure"))
            super().handleError(record)

    class _CountingRotatingFileHandler(_FailureCountingHandler, logging.handlers.RotatingFileHandler):
        pass

    class _CountingStreamHandler(_FailureCountingHandler, logging.StreamHandler):
        pass

    class _AuditJsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:  # noqa: A003
            # Prefer a pre-serialized ``audit_payload`` when set; else fall back
            # to standard message handling. This keeps unit tests simple.
            payload_obj = getattr(record, "audit_payload", None)
            if isinstance(payload_obj, dict):
                try:
                    return json.dumps(payload_obj, ensure_ascii=False)
                except Exception:  # noqa: BLE001
                    return json.dumps({"message": record.getMessage()}, ensure_ascii=False)
            return json.dumps({"message": record.getMessage()}, ensure_ascii=False)

    formatter = _AuditJsonFormatter()

    handlers: list[logging.Handler] = []

    # Stdout handler — always present (separate from ``app`` stdout so a log
    # aggregator can filter on logger name ``audit``).
    stream_handler = _CountingStreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    # Mark so we can cleanly replace on reconfigure.
    setattr(stream_handler, "_va_lse_audit_managed", True)
    handlers.append(stream_handler)

    # Rotating file handler — best-effort.
    if raw_dir:
        try:
            log_path = Path(raw_dir).expanduser().resolve()
            log_path.mkdir(parents=True, exist_ok=True)
            file_path = log_path / raw_file
            file_handler = _CountingRotatingFileHandler(
                str(file_path), maxBytes=raw_max, backupCount=raw_backups, encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            setattr(file_handler, "_va_lse_audit_managed", True)
            handlers.append(file_handler)
        except Exception as exc:  # noqa: BLE001 - audit setup must not crash the app
            logging.getLogger("audit.setup").warning(
                "Could not set up audit file logging at %s/%s: %s — continuing with stdout only.",
                raw_dir,
                raw_file,
                exc,
            )

    audit_logger = logging.getLogger(_AUDIT_LOGGER_NAME)
    # Remove previously-installed managed handlers on reconfigure.
    for h in list(audit_logger.handlers):
        if getattr(h, "_va_lse_audit_managed", False):
            audit_logger.removeHandler(h)
            try:
                h.close()
            except Exception:  # noqa: BLE001
                pass
    for h in handlers:
        audit_logger.addHandler(h)
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False

    _CONFIGURED = True
    setattr(sys.modules[__name__], _CONFIGURED_KEY, True)
    return audit_logger


def get_audit_logger() -> logging.Logger:
    """Return the configured audit logger, configuring it once if needed."""
    if not _CONFIGURED:
        configure_audit_logging()
    return logging.getLogger(_AUDIT_LOGGER_NAME)


# -------------------------------------------------------------- session id


def get_audit_session_id() -> str:
    """Return a stable per-browser-session id (``sess_…``).

    When Streamlit is available the id is stored in ``st.session_state`` so it
    survives reruns within the same browser session. Otherwise a process-global
    id is used (suitable for tests / CLI).
    """
    global _process_session_id  # noqa: PLW0603
    try:
        import streamlit as st

        key = "_va_lse_audit_session_id"
        existing = st.session_state.get(key, "")
        if isinstance(existing, str) and existing.startswith("sess_"):
            return existing
        new_id = f"sess_{uuid.uuid4().hex[:12]}"
        try:
            st.session_state[key] = new_id
        except Exception:  # noqa: BLE001
            pass
        return new_id
    except Exception:  # noqa: BLE001 - streamlit unavailable in tests
        pass
    if _process_session_id is None:
        _process_session_id = f"sess_{uuid.uuid4().hex[:12]}"
    return _process_session_id


def audit_health() -> dict[str, Any]:
    """Health payload for ``GET /health`` — is the audit stream actually writing?

    ``write_failures`` is the number of failed handler emits in this process. A
    non-zero value means audit records were lost; the usual cause is a full or
    read-only log volume, and it is reported rather than raised because the app
    is designed to keep serving when audit logging breaks.
    """
    with _write_lock:
        failures = _write_failures
        last_error = _last_write_error
    payload: dict[str, Any] = {
        "configured": _CONFIGURED,
        "dir": os.getenv("VA_LSE_AUDIT_LOG_DIR", "").strip()
        or os.getenv("VA_LSE_LOG_DIR", "").strip()
        or "logs",
        "file": _env_str("VA_LSE_AUDIT_LOG_FILE", "audit.log"),
        "max_bytes": _config.AUDIT_LOG_MAX_BYTES,
        "backups": _config.AUDIT_LOG_BACKUPS,
        "retention_days": _config.AUDIT_RETENTION_DAYS,
        "error_messages_enabled": _config.AUDIT_ERROR_MESSAGES,
        "write_failures": failures,
        "last_write_error": last_error or None,
    }
    if not _CONFIGURED:
        payload["status"] = "not_configured"
    elif failures:
        payload["status"] = "degraded"
        payload["reason"] = (
            f"{failures} audit write(s) failed; records were lost. "
            "Check free space and permissions on the audit log directory."
        )
    else:
        payload["status"] = "ok"
    return payload


def _reset_for_tests() -> None:
    """Reset process-global session id and write counters (tests only)."""
    global _process_session_id, _CONFIGURED, _write_failures, _last_write_error  # noqa: PLW0603

    _process_session_id = None
    _CONFIGURED = False
    with _write_lock:
        _write_failures = 0
        _last_write_error = ""
    # Remove handlers so tmp-path tests do not leak.
    audit_logger = logging.getLogger(_AUDIT_LOGGER_NAME)
    for h in list(audit_logger.handlers):
        if getattr(h, "_va_lse_audit_managed", False):
            audit_logger.removeHandler(h)
            try:
                h.close()
            except Exception:  # noqa: BLE001
                pass


# -------------------------------------------------------------- public API


def audit_event(
    action: Action,
    status: Status,
    *,
    request_id: str,
    user_session_id: str | None = None,
    condition: str | None = None,
    record_sources: list[str] | None = None,
    record_files: int | None = None,
    record_pages: int | None = None,
    duration_ms: int | None = None,
    outcome: dict[str, Any] | None = None,
    error_class: str | None = None,
    error_message: str | None = None,
    llm_endpoints: list[str] | None = None,
) -> None:
    """Emit one audit log entry (best-effort, never raises).

    Only metadata and classifications are recorded — no statement text,
    observations, or record content. ``condition`` is truncated to 120 chars;
    ``error_message`` to 300 chars.

    ``llm_endpoints`` names the endpoints that served the run (``primary`` when a
    single endpoint was used). It is here because a failover silently changes
    which model wrote the document: this is a legal work product, and "the backup
    provider generated this one" belongs in the audit record next to the rest of
    the run's metadata, not only in a log line.
    """
    try:
        audit_logger = get_audit_logger()
        # Resolve session id lazily when caller does not supply one.
        sess_id = (user_session_id or "").strip() or get_audit_session_id()
        # Normalize condition — never log names/PII, only the claim type.
        cond = _safe_truncate(condition or "", 120) if condition else ""
        # Normalize sources — only labels, not filenames/paths that might leak.
        sources: list[str] = []
        if record_sources:
            for s in record_sources:
                label = _safe_truncate(str(s), 40)
                if label:
                    sources.append(label)

        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "status": status,
            "request_id": request_id or "-",
            "user_session_id": sess_id,
        }
        if cond:
            payload["condition"] = cond
        if sources:
            payload["record_sources"] = sources
        if record_files is not None:
            payload["record_files"] = int(record_files)
        if record_pages is not None:
            payload["record_pages"] = int(record_pages)
        if duration_ms is not None:
            payload["duration_ms"] = int(duration_ms)
        if llm_endpoints:
            # Bounded and truncated like every other field here: these come from
            # the caller, and an audit record must not grow without limit.
            safe_endpoints = [
                _safe_truncate(str(name), 32) for name in llm_endpoints[:4] if str(name).strip()
            ]
            if safe_endpoints:
                payload["llm_endpoints"] = safe_endpoints
        if outcome is not None and isinstance(outcome, dict) and outcome:
            # Shallow-copy and ensure JSON-serializable primitives only.
            safe_outcome: dict[str, Any] = {}
            for k, v in outcome.items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    safe_outcome[str(k)] = v
                elif isinstance(v, (list, dict)):
                    try:
                        json.dumps(v)
                        safe_outcome[str(k)] = v
                    except Exception:  # noqa: BLE001
                        safe_outcome[str(k)] = repr(v)
                else:
                    safe_outcome[str(k)] = repr(v)
            payload["outcome"] = safe_outcome
        if error_class:
            payload["error_class"] = _safe_truncate(error_class, 80)
        # Free-text upstream error text is the one field outside the module's
        # "counts and classifications only" contract; see the module docstring.
        if error_message and _config.AUDIT_ERROR_MESSAGES:
            payload["error_message"] = _scrub_error_message(error_message)

        # Emit as structured JSON — the formatter serializes ``audit_payload``.
        audit_logger.info("", extra={"audit_payload": payload})
    except Exception:  # noqa: BLE001 - audit must never break the app
        pass


def audit_evaluate_start(
    *,
    request_id: str,
    condition: str | None = None,
    record_sources: list[str] | None = None,
    record_files: int | None = None,
    record_pages: int | None = None,
    user_session_id: str | None = None,
) -> None:
    audit_event(
        "evaluate",
        "start",
        request_id=request_id,
        user_session_id=user_session_id,
        condition=condition,
        record_sources=record_sources,
        record_files=record_files,
        record_pages=record_pages,
    )


def audit_evaluate_ok(
    *,
    request_id: str,
    duration_ms: int,
    condition: str | None = None,
    record_sources: list[str] | None = None,
    record_files: int | None = None,
    record_pages: int | None = None,
    outcome: dict[str, Any] | None = None,
    user_session_id: str | None = None,
    llm_endpoints: list[str] | None = None,
) -> None:
    audit_event(
        "evaluate",
        "ok",
        request_id=request_id,
        user_session_id=user_session_id,
        condition=condition,
        record_sources=record_sources,
        record_files=record_files,
        record_pages=record_pages,
        duration_ms=duration_ms,
        outcome=outcome,
        llm_endpoints=llm_endpoints,
    )


def audit_evaluate_error(
    *,
    request_id: str,
    duration_ms: int,
    error: BaseException,
    condition: str | None = None,
    record_sources: list[str] | None = None,
    record_files: int | None = None,
    record_pages: int | None = None,
    user_session_id: str | None = None,
) -> None:
    audit_event(
        "evaluate",
        "error",
        request_id=request_id,
        user_session_id=user_session_id,
        condition=condition,
        record_sources=record_sources,
        record_files=record_files,
        record_pages=record_pages,
        duration_ms=duration_ms,
        error_class=type(error).__name__,
        error_message=str(error),
    )


def audit_draft_start(
    *,
    request_id: str,
    condition: str | None = None,
    record_sources: list[str] | None = None,
    record_files: int | None = None,
    record_pages: int | None = None,
    user_session_id: str | None = None,
) -> None:
    audit_event(
        "draft",
        "start",
        request_id=request_id,
        user_session_id=user_session_id,
        condition=condition,
        record_sources=record_sources,
        record_files=record_files,
        record_pages=record_pages,
    )


def audit_draft_ok(
    *,
    request_id: str,
    duration_ms: int,
    condition: str | None = None,
    record_sources: list[str] | None = None,
    record_files: int | None = None,
    record_pages: int | None = None,
    outcome: dict[str, Any] | None = None,
    user_session_id: str | None = None,
    llm_endpoints: list[str] | None = None,
) -> None:
    audit_event(
        "draft",
        "ok",
        request_id=request_id,
        user_session_id=user_session_id,
        condition=condition,
        record_sources=record_sources,
        record_files=record_files,
        record_pages=record_pages,
        duration_ms=duration_ms,
        outcome=outcome,
        llm_endpoints=llm_endpoints,
    )


def audit_draft_error(
    *,
    request_id: str,
    duration_ms: int,
    error: BaseException,
    condition: str | None = None,
    record_sources: list[str] | None = None,
    record_files: int | None = None,
    record_pages: int | None = None,
    user_session_id: str | None = None,
) -> None:
    audit_event(
        "draft",
        "error",
        request_id=request_id,
        user_session_id=user_session_id,
        condition=condition,
        record_sources=record_sources,
        record_files=record_files,
        record_pages=record_pages,
        duration_ms=duration_ms,
        error_class=type(error).__name__,
        error_message=str(error),
    )
