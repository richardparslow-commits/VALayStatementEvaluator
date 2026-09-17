"""Structured application logging with correlation (request) IDs.

Goals per spec:
 - Python ``logging`` (not ``print``) so aggregation tools can consume it.
 - JSON line format (configurable) plus plain fallback.
 - Each *run* gets a unique ``request_id`` stored in ``st.session_state`` and
   available via a :class:`contextvars.ContextVar` so parallel workers carry it.
 - Fields logged for observability: ``request_id``, ``phase``, ``duration_ms``,
   ``status``, ``model``, ``tokens``, ``retry``/``attempt``, error messages with
   stack traces.
 - PII discipline: medical record text, statement text, observations, and LLM
   prompt bodies are **never** logged — only counts/sizes/classifications.

Configuration via env (see ``.env.example``):
 - ``VA_LSE_LOG_LEVEL``    – ``DEBUG``/``INFO``/``WARNING``…  (default ``INFO``)
 - ``VA_LSE_LOG_JSON``     – ``1`` emits JSON lines (default off → plain)
 - ``VA_LSE_LOG_DIR``      – directory to write ``app.log`` (default stdout-only)
 - ``VA_LSE_LOG_FILE``     – file name inside that dir (default ``app.log``)
 - ``VA_LSE_LOG_MAX_BYTES``– rotate size   (default 10 MiB)
 - ``VA_LSE_LOG_BACKUPS``  – rotated files kept (default 5)

``configure_logging()`` is idempotent and safe to call from tests, CLI entry
points, or Streamlit ``main()``. Existing loggers (e.g. ``app.va_gov_client``)
benefit without code changes because we attach handlers to a shared ``app``
parent logger plus keep the root handler for third-party compatibility.
"""

from __future__ import annotations

import contextvars
import json
import logging
import logging.handlers
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------- request id

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "va_lse_request_id", default=""
)


def new_request_id() -> str:
    """Return a fresh short correlation id (e.g. ``req_a1b2c3d4e5f6``)."""
    return f"req_{uuid.uuid4().hex[:12]}"


def set_request_id(request_id: str) -> contextvars.Token[str]:
    return _request_id_var.set(request_id or "")


def get_request_id() -> str:
    try:
        return _request_id_var.get() or ""
    except LookupError:
        return ""


def clear_request_id(token: contextvars.Token[str] | None = None) -> None:
    if token is not None:
        try:
            _request_id_var.reset(token)
            return
        except ValueError:
            pass
    try:
        _request_id_var.set("")
    except Exception:  # noqa: BLE001
        pass


def decorate_logger_with_request_id(
    logger: logging.Logger,
) -> logging.Logger:
    """Attach a filter that injects the current ``request_id`` into each record.

    Safe to call repeatedly; filter is added once per logger instance.
    """
    marker = "_va_lse_request_id_filter"

    if getattr(logger, marker, False):
        return logger

    class _RequestIdFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
            # Only fill if the caller didn't supply an explicit request_id.
            if not getattr(record, "request_id", None):
                try:
                    setattr(record, "request_id", _request_id_var.get() or "-")
                except LookupError:
                    setattr(record, "request_id", "-")
            return True

    logger.addFilter(_RequestIdFilter())
    setattr(logger, marker, True)
    return logger


# ---------------------------------------------------------------- formatters


_DROP_ATTRS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
    }
)


class JsonFormatter(logging.Formatter):
    """Render each record as a single JSON line.

    Extra attributes supplied via ``logger.info(msg, extra={...})`` appear as
    top-level fields so aggregation queries can filter on e.g. ``phase``,
    ``status``, ``duration_ms``. Never serialize message args lazily — we
    format here and emit a ``message`` field.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        # Ensure request_id is present (fallback to "-")
        request_id = getattr(record, "request_id", None) or getattr(
            record, "requestId", None
        )
        if not request_id or request_id == "-":
            try:
                request_id = _request_id_var.get() or "-"
            except LookupError:
                request_id = "-"

        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id,
            "message": record.getMessage(),
        }
        # Promote conventional observability fields when present.
        for key in ("phase", "status", "duration_ms", "model", "attempt", "retries",
                    "tokens_in", "tokens_out", "prompt_tokens", "completion_tokens",
                    "total_tokens", "calls", "chunks", "pages", "facts", "error_class",
                    "error_kind", "retryable", "status_code", "upstream_request_id",
                    "workflow", "requestId"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val

        # Include any other *extra* attributes the caller attached that are not
        # part of the standard LogRecord namespace.
        for key, val in record.__dict__.items():
            if key in _DROP_ATTRS or key in payload or key.startswith("_"):
                continue
            # Only JSON-serializable primitives / simple structures are emitted.
            # Anything else is coerced to its repr to avoid dropping the line.
            try:
                json.dumps(val)
                payload[key] = val
            except Exception:  # noqa: BLE001
                payload[key] = repr(val)

        if record.exc_info and record.exc_info[0] is not None:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False)


class PlainFormatter(logging.Formatter):
    """Human-readable fallback when ``VA_LSE_LOG_JSON`` is off."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        request_id = getattr(record, "request_id", None)
        if not request_id:
            try:
                request_id = _request_id_var.get() or "-"
            except LookupError:
                request_id = "-"
            setattr(record, "request_id", request_id)
        # Attach request_id to the message prefix; still emit extra fields if provided.
        base = super().format(record)
        extras: list[str] = []
        for key in ("phase", "status", "duration_ms", "model", "attempt"):
            val = getattr(record, key, None)
            if val is not None:
                extras.append(f"{key}={val}")
        if extras:
            base = f"{base}  [{', '.join(extras)}]"
        return base


# -------------------------------------------------------------- configuration

_CONFIGURED = False
_CONFIGURED_KEY = "_va_lse_logging_configured"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _level_from_env() -> int:
    raw = os.getenv("VA_LSE_LOG_LEVEL", "INFO").strip().upper() or "INFO"
    return getattr(logging, raw, logging.INFO)


def configure_logging(
    *,
    level: int | None = None,
    json_output: bool | None = None,
    log_dir: str | Path | None = None,
    log_file: str | None = None,
    max_bytes: int | None = None,
    backups: int | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure structured app logging exactly once (idempotent).

    Subsequent calls without ``force=True`` are no-ops and return the shared
    ``app`` logger so instrumentation can call this from multiple entry points
    without duplicating handlers. In tests call with ``force=True`` to reapply.

    Logs never contain PII — instrumentation is responsible for emitting only
    sizes/classifications (see ``llm.py``/``medical_review.py`` docstrings).
    """
    global _CONFIGURED  # noqa: PLW0603

    if _CONFIGURED and not force:
        return logging.getLogger("app")

    lvl = level if level is not None else _level_from_env()
    use_json = json_output if json_output is not None else _env_bool("VA_LSE_LOG_JSON", False)

    raw_dir = os.getenv("VA_LSE_LOG_DIR", "").strip() if log_dir is None else str(log_dir).strip()
    raw_file = os.getenv("VA_LSE_LOG_FILE", "app.log").strip() if log_file is None else (log_file or "app.log").strip()
    raw_max = max_bytes
    if raw_max is None:
        try:
            raw_max = int(os.getenv("VA_LSE_LOG_MAX_BYTES", str(10 * 1024 * 1024)).strip() or 10 * 1024 * 1024)
        except ValueError:
            raw_max = 10 * 1024 * 1024
    raw_backups = backups
    if raw_backups is None:
        try:
            raw_backups = int(os.getenv("VA_LSE_LOG_BACKUPS", "5").strip() or 5)
        except ValueError:
            raw_backups = 5

    formatter: logging.Formatter
    if use_json:
        formatter = JsonFormatter()
    else:
        formatter = PlainFormatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] [%(request_id)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )

    # Stream handler (always present — ensures visibility even when VA_LSE_LOG_DIR unset,
    # and keeps platform log drains / ``docker logs`` / AGILOOP streams useful).
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(lvl)

    handlers: list[logging.Handler] = [stream_handler]

    file_handler: logging.Handler | None = None
    if raw_dir:
        try:
            log_path = Path(raw_dir).expanduser().resolve()
            log_path.mkdir(parents=True, exist_ok=True)
            file_path = log_path / raw_file
            file_handler = logging.handlers.RotatingFileHandler(
                str(file_path), maxBytes=raw_max, backupCount=raw_backups, encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(lvl)
            handlers.append(file_handler)
        except Exception as exc:  # noqa: BLE001 - logging setup must not crash the app
            # Fall back to stdout-only and note why file logging is disabled.
            logging.getLogger("app.logging_config").warning(
                "Could not set up file logging at %s/%s: %s — continuing with stdout only.",
                raw_dir,
                raw_file,
                exc,
            )

    # Configure the shared ``app`` parent logger; children like ``app.llm``,
    # ``app.medical_review`` etc. propagate to it and inherit handlers + level.
    app_logger = logging.getLogger("app")
    # Remove previously-installed VA_LSE handlers to keep reconfiguration clean.
    for h in list(app_logger.handlers):
        if getattr(h, "_va_lse_managed", False):
            app_logger.removeHandler(h)
            try:
                h.close()
            except Exception:  # noqa: BLE001
                pass
    for h in handlers:
        setattr(h, "_va_lse_managed", True)
        app_logger.addHandler(h)
    app_logger.setLevel(lvl)
    app_logger.propagate = False  # avoid double-emitting via root
    decorate_logger_with_request_id(app_logger)

    # Keep the root logger minimally configured so libraries that log to root
    # still surface (at WARNING) without flooding stdout.
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.WARNING)

    # Also decorate commonly-used child loggers so direct ``logging.getLogger("app.xxx")``
    # usage inherits the correlation id automatically.
    for child_name in (
        "app.llm",
        "app.medical_review",
        "app.evaluate",
        "app.draft",
        "app.main",
        "app.fetch_client",
        "app.va_gov_client",
        "app.telemetry",
        "app.agiloop_telemetry",
        "app.health",
    ):
        decorate_logger_with_request_id(logging.getLogger(child_name))

    _CONFIGURED = True
    # Marker so tests can detect that setup ran even if handlers were replaced.
    setattr(sys.modules[__name__], _CONFIGURED_KEY, True)
    # Emit one line confirming the logging mode (without PII) so operators can
    # verify the config is active; level-aware so DEBUG test suites stay quiet.
    app_logger.debug(
        "logging configured (level=%s json=%s dir=%s file=%s)",
        logging.getLevelName(lvl),
        use_json,
        raw_dir or "(stdout only)",
        raw_file if raw_dir else "-",
    )
    return app_logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger that automatically carries ``request_id``.

    Ensures :func:`configure_logging` has run at least once so early imports
    that log before ``main()`` still have handlers.
    """
    if not _CONFIGURED:
        configure_logging()
    logger = logging.getLogger(name)
    decorate_logger_with_request_id(logger)
    return logger


# ----------------------------------------------------------------- timing

def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _record_phase_metric(phase: str, outcome: str, duration_ms: int) -> None:
    """Feed a completed phase to the metrics registry.

    Imported lazily and guarded because this is called from the logging layer,
    which sits underneath everything else: a failure to record a metric must never
    be able to break logging, and logging must never be able to break a run.
    """
    try:
        from .metrics import observe_phase

        observe_phase(phase, outcome, duration_ms)
    except Exception:  # noqa: BLE001 - instrumentation is best-effort by design
        pass


class PhaseTimer:
    """Context manager that logs phase start/done with ``duration_ms``.

    Example
    -------
    >>> with PhaseTimer(logger, \"records:digest\", request_id=rid, chunks=42):
    ...     do_work()

    On success logs at INFO with ``status=ok``; on exception logs at ERROR
    with ``status=error`` and re-raises. Never logs PII.
    """

    def __init__(
        self,
        logger: logging.Logger,
        phase: str,
        *,
        request_id: str | None = None,
        level: int = logging.INFO,
        **fields: Any,
    ) -> None:
        self.logger = logger
        self.phase = phase
        self.request_id = request_id or get_request_id() or "-"
        self.level = level
        self.fields = fields
        self._t0: float = 0.0

    def __enter__(self) -> PhaseTimer:
        self._t0 = _now_ms()
        self.logger.log(
            self.level,
            "phase start: %s",
            self.phase,
            extra={"request_id": self.request_id, "phase": self.phase, "status": "start", **self.fields},
        )
        return self

    def __exit__(  # noqa: ANN001
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> Literal[False]:
        duration_ms = int(_now_ms() - self._t0) if self._t0 else 0
        if exc_type is None:
            _record_phase_metric(self.phase, "ok", duration_ms)
            self.logger.log(
                self.level,
                "phase done: %s (%d ms)",
                self.phase,
                duration_ms,
                extra={
                    "request_id": self.request_id,
                    "phase": self.phase,
                    "status": "ok",
                    "duration_ms": duration_ms,
                    **self.fields,
                },
            )
            return False
        # Error path — include stack trace but never body/PII
        _record_phase_metric(self.phase, "error", duration_ms)
        from types import TracebackType as _TracebackType

        _tb: _TracebackType | None = exc_tb if isinstance(exc_tb, _TracebackType) else None
        _exc_info: tuple[type[BaseException], BaseException, _TracebackType | None] | None = None
        if exc_type is not None and exc_val is not None:
            _exc_info = (exc_type, exc_val, _tb)
        self.logger.error(
            "phase error: %s (%d ms): %s",
            self.phase,
            duration_ms,
            exc_val,
            exc_info=_exc_info,
            extra={
                "request_id": self.request_id,
                "phase": self.phase,
                "status": "error",
                "duration_ms": duration_ms,
                **self.fields,
            },
        )
        return False
