"""Persistent structured run log for Evaluate/Draft lifecycle events.

Every run — including ones that die in pre-run validation, before the audit
log's ``start`` event — appends one JSON line to ``{log_dir}/runs.jsonl``
(default ``logs/runs.jsonl``; override with ``VA_LSE_RUN_LOG_DIR``, disable
with ``VA_LSE_RUN_LOG_DISABLED=1``).

Why this exists: the ``req_…`` reference shown to users must always be
correlatable. Before this module, a run that failed *before*
``audit_draft_start``/``audit_evaluate_start`` (input validation, LLM-client
setup, shutdown gate, memory gate) was only visible if stdout logging was
captured — which the launchd preview lost, leaving the reference untraceable.

Design:
- JSON lines, one event per line, containing only metadata — no statement,
  observations, or record text (same PII discipline as app/audit.py).
- stdlib only; append+flush per event; never raises (best-effort like audit).
- The UI-facing error text and the run log are written through the same
  helper so a user-shown reference always has a matching log line.
- Size-bounded (``VA_LSE_RUN_LOG_MAX_BYTES`` x ``VA_LSE_RUN_LOG_BACKUPS``). Until
  this was added this file was the *only* unbounded writer in the app — the audit
  log has always rotated — so it, not ``audit.log``, was what could fill the log
  volume on a long-lived pod. Rotation happens under the same lock as the append,
  so it cannot lose a line or interleave with a concurrent writer.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config
from .logging_config import get_logger

logger = get_logger("app.run_log")

_LOCK = threading.Lock()
_FILENAME = "runs.jsonl"


def _rotated_names(path: Path) -> list[Path]:
    """Rotated siblings, newest first (``runs.jsonl.1`` … ``runs.jsonl.N``)."""
    return [path.with_name(f"{path.name}.{i}") for i in range(1, config.RUN_LOG_BACKUPS + 1)]


def _rotate_if_needed(path: Path) -> None:
    """Shift ``runs.jsonl`` to ``.1`` once it reaches the size limit.

    Called with ``_LOCK`` held. Rotation failures are swallowed by the caller's
    best-effort contract — a run log that stops rotating is bad, but a run log
    that breaks a run is worse.
    """
    try:
        if not path.exists() or path.stat().st_size < config.RUN_LOG_MAX_BYTES:
            return
        backups = max(1, config.RUN_LOG_BACKUPS)
        path.with_name(f"{path.name}.{backups}").unlink(missing_ok=True)
        for index in range(backups - 1, 0, -1):
            source = path.with_name(f"{path.name}.{index}")
            if source.exists():
                source.replace(path.with_name(f"{path.name}.{index + 1}"))
        path.replace(path.with_name(f"{path.name}.1"))
    except OSError as exc:
        logger.debug("could not rotate the run log: %s", exc)


def _resolve_log_path() -> Path:
    """Resolve the run-log file path from env; logs/runs.jsonl by default."""
    if os.getenv("VA_LSE_RUN_LOG_DISABLED", "").strip() == "1":
        return Path("/dev/null")  # writes become cheap no-ops
    raw_dir = os.getenv("VA_LSE_RUN_LOG_DIR", "").strip()
    if not raw_dir:
        # Mirror the audit log's dir resolution: explicit run-log dir wins,
        # then the diagnostic log dir, then ./logs.
        diag_dir = os.getenv("VA_LSE_LOG_DIR", "").strip()
        raw_dir = diag_dir or "logs"
    path = Path(raw_dir).expanduser().resolve() / _FILENAME
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001 - best-effort
        path = Path("/dev/null")
    return path


def run_log_event(
    action: str,
    status: str,
    *,
    request_id: str,
    error: str | None = None,
    error_class: str | None = None,
    **extra: Any,
) -> None:
    """Append one lifecycle event for a run. Never raises.

    ``extra`` values must be small JSON-safe metadata (ints, short strings,
    bools) — no user text. Long strings are truncated defensively.
    """
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "status": status,
        "request_id": request_id or "-",
    }
    if error:
        payload["error"] = str(error)[:300]
    if error_class:
        payload["error_class"] = str(error_class)[:80]
    for key, value in extra.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            payload[str(key)[:40]] = value if isinstance(value, (int, float, bool)) else str(value)[:120]
        else:
            payload[str(key)[:40]] = repr(value)[:120]
    try:
        line = json.dumps(payload, ensure_ascii=False)
        with _LOCK:
            path = _resolve_log_path()
            _rotate_if_needed(path)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:  # noqa: BLE001 - run log must never break a run
        pass


def read_recent_events(limit: int = 200) -> list[dict[str, Any]]:
    """Best-effort read of the last ``limit`` events (for the ops UI/tests).

    Reads newest file first and stops as soon as ``limit`` events are parsed, so
    the common case touches one file rather than the whole retained history.
    """
    try:
        path = _resolve_log_path()
        if not path.exists():
            return []
        newest_first: list[dict[str, Any]] = []
        for candidate in [path, *_rotated_names(path)]:
            try:
                if not candidate.exists():
                    continue
                for line in reversed(candidate.read_text(encoding="utf-8").splitlines()):
                    if not line.strip():
                        continue
                    try:
                        newest_first.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                    if len(newest_first) >= limit:
                        break
            except OSError:
                continue
            if len(newest_first) >= limit:
                break
        newest_first.reverse()
        return newest_first[-limit:]
    except Exception:  # noqa: BLE001
        return []
