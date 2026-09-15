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
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .logging_config import get_logger

logger = get_logger("app.run_log")

_LOCK = threading.Lock()
_FILENAME = "runs.jsonl"


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
            with open(_resolve_log_path(), "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:  # noqa: BLE001 - run log must never break a run
        pass


def read_recent_events(limit: int = 200) -> list[dict[str, Any]]:
    """Best-effort read of the last ``limit`` events (for the ops UI/tests)."""
    try:
        path = _resolve_log_path()
        if not path.exists() or path.name == "runs.jsonl" and not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        out: list[dict[str, Any]] = []
        for line in lines[-limit:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
    except Exception:  # noqa: BLE001
        return []
