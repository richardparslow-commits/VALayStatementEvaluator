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
import sys
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

Action = Literal["evaluate", "draft"]
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
    stream_handler = logging.StreamHandler(sys.stdout)
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
            file_handler = logging.handlers.RotatingFileHandler(
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


def _reset_for_tests() -> None:
    """Reset process-global session id (tests only)."""
    global _process_session_id, _CONFIGURED  # noqa: PLW0603

    _process_session_id = None
    _CONFIGURED = False
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
) -> None:
    """Emit one audit log entry (best-effort, never raises).

    Only metadata and classifications are recorded — no statement text,
    observations, or record content. ``condition`` is truncated to 120 chars;
    ``error_message`` to 300 chars.
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
        if error_message:
            payload["error_message"] = _safe_truncate(error_message, 300)

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
