"""Ops view helpers: in-browser answers about what a run did.

Two renderers, for two moments. ``render_run_log_tail`` shows the most recent
``logs/runs.jsonl`` events in the About tab, so an operator (or the user) can see
recent activity without shelling into the machine. ``render_failure_detail``
resolves one ``req_…`` reference to its lines and is placed directly under the
error that quoted it, so the reason is one click from the failure instead of a
second navigation. Read-only and best-effort: any I/O problem collapses to a
quiet note rather than an exception.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import streamlit as st

from ..diagnostics import ReferenceDetail, is_reference, lookup
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


# One lookup per reference per session: see _cached_detail.
_FAILURE_CACHE_KEY = "diagnostics_failure_detail_cache"
_FAILURE_CACHE_LIMIT = 12


def _cached_detail(reference: str) -> ReferenceDetail:
    """Look a reference up once, then remember the answer for the session.

    An expander's body runs on every rerun even while it stays collapsed, and a
    lookup reads the run log — so without this, a failure left on screen would
    re-read the log on every unrelated widget interaction anywhere in the app. The
    record of a finished run does not change, and the cache is bounded.
    """
    cache = st.session_state.get(_FAILURE_CACHE_KEY)
    if not isinstance(cache, dict):
        cache = {}
    cached = cache.get(reference)
    if isinstance(cached, ReferenceDetail):
        return cached
    detail = lookup(reference)
    cache[reference] = detail
    while len(cache) > _FAILURE_CACHE_LIMIT:
        cache.pop(next(iter(cache)))
    st.session_state[_FAILURE_CACHE_KEY] = cache
    return detail


def render_failure_detail(reference: str, *, label: str = "What happened?") -> None:
    """Resolve a failure that was just shown to the user, in place.

    The moment a user reads "Drafting failed … (reference: req_…)" is the moment
    they want the reason; sending them to another tab to paste an id asks for a
    step they should not have to take. A run executed by a worker is covered too,
    because the lookup reads the shared run log as well as this process's buffer.
    """
    if not is_reference(reference):
        return
    detail = _cached_detail(reference)
    with st.expander(label, expanded=False):
        if detail.lines:
            st.code("\n".join(detail.lines), language=None)
        if detail.events:
            st.dataframe(
                [_event_row(e) for e in detail.events],
                width="stretch",
                hide_index=True,
            )
        if detail.note:
            st.caption(detail.note)
        st.caption(
            f"Reference `{detail.reference}` — secrets are redacted above, and "
            "**About → 🔎 Look up a reference** searches the shared run log for any "
            "other reference."
        )


__all__ = ["render_failure_detail", "render_run_log_tail"]
