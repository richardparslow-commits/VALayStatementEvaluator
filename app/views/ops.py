"""Ops view helpers: in-browser tail of the persistent run log.

Renders the most recent ``logs/runs.jsonl`` events in the About tab so an
operator (or the user) can confirm a ``req_…`` reference without shelling
into the machine. Read-only, best-effort: any I/O problem collapses to a
quiet "run log unavailable" note.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import streamlit as st

from ..run_log import _resolve_log_path

_RUN_LOG_LIMIT = 30

_STATUS_EMOJI = {
    "ok": "✅",
    "start": "▶️",
    "error": "❌",
    "timeout": "⏱️",
    "rejected": "🚫",
    "empty": "⚠️",
    "interrupted": "⏹️",
}


def _format_event_time(iso: str) -> str:
    """Render a run-log UTC timestamp as a compact local-ish string."""
    try:
        dt = datetime.fromisoformat(iso)
        return dt.astimezone().strftime("%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return str(iso)[:19]


def _event_row(event: dict[str, Any]) -> dict[str, str]:
    """Project one run-log event onto the displayed columns (no user text)."""
    status = str(event.get("status", "?"))
    error = str(event.get("error", "") or "")
    if len(error) > 90:
        error = error[:87] + "…"
    return {
        "Time": _format_event_time(str(event.get("timestamp", ""))),
        "Action": str(event.get("action", "?")),
        "Status": f"{_STATUS_EMOJI.get(status, '•')} {status}",
        "Request ID": str(event.get("request_id", "-")),
        "Duration (ms)": str(event.get("duration_ms", "")),
        "Detail": error,
    }


def render_run_log_tail(limit: int = _RUN_LOG_LIMIT) -> None:
    """Show the latest run-log events (newest last, like the file itself)."""
    path = _resolve_log_path()
    events: list[dict[str, Any]] = []
    available = True
    try:
        if path.exists() and path.name != "runs.jsonl" or (path.exists() and path.name == "runs.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:  # noqa: BLE001 - viewer must never break the tab
        available = False

    if not available:
        st.caption("Run log is unavailable (disabled or not writable on this host).")
        return
    if not events:
        st.caption(
            "No runs recorded yet — events appear here after the first Evaluate or Draft run."
        )
        return

    error_count = sum(1 for e in events if str(e.get("status")) in {"error", "timeout"})
    st.caption(
        f"Last {len(events)} run-log event(s) from `{path}` — newest last. "
        f"{error_count} failure(s) in this window. "
        "Full detail: `grep <request-id> logs/runs.jsonl`."
    )
    st.dataframe(
        [_event_row(e) for e in events],
        width="stretch",
        hide_index=True,
    )


__all__ = ["render_run_log_tail"]
