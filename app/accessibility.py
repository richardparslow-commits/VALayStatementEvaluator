"""Literal pilot messages and reading help; no provider or admission authority."""
from __future__ import annotations

from html import escape
from typing import Any


def status_html(method: str, value: Any) -> str:
    """Only fixed markup is active; every caller-controlled character is text."""
    labels = {"error": "Error", "warning": "Warning", "info": "Information", "success": "Update"}
    label = labels[method]
    role, live = ("alert", "assertive") if method in ("error", "warning") else ("status", "polite")
    return (f'<div role="{role}" aria-live="{live}" aria-atomic="true" '
            'style="max-width:100%;min-width:0;border:1px solid currentColor;border-radius:0.4rem;'
            'padding:0.75rem;font:inherit;white-space:pre-wrap;overflow-wrap:anywhere">'
            f'<strong>{label}: </strong>{escape(str(value), quote=True)}</div>')


def table_html(rows: list[dict[str, Any]]) -> str:
    """Native table semantics, exact cell text and no export toolbar/assets."""
    columns = list(dict.fromkeys(key for row in rows for key in row))
    cell_style = 'style="border:1px solid currentColor;padding:0.5rem;vertical-align:top;white-space:pre-wrap;overflow-wrap:anywhere"'
    headers = "".join(f'<th scope="col" {cell_style}>{escape(str(key), quote=True)}</th>' for key in columns)
    body = "".join('<tr>' + "".join(f'<td {cell_style}>{escape(str(row.get(key, "")), quote=True)}</td>'
                                  for key in columns) + '</tr>' for row in rows)
    return ('<div role="region" aria-label="Scrollable data table" tabindex="0" '
            'style="max-width:100%;overflow-x:auto">'
            '<table style="width:100%;border-collapse:collapse;font:inherit">'
            '<caption>Read-only data table</caption><thead><tr>' + headers + '</tr></thead><tbody>'
            + body + '</tbody></table></div>')


READING_GUIDE = (
    ("NOT FOUND", "The model did not find matching text in the supplied readable records. "
     "This does not show that the event did not happen. Check missing or unreadable pages and the witness account."),
    ("CONTRADICTED", "The model flagged an apparent conflict. Read the exact passages, dates and who said each thing. "
     "A label alone does not establish which account is correct."),
    ("SUPPORTED / PARTIALLY SUPPORTED", "The model found material it considers relevant. Read the original passage "
     "to check the meaning, uncertainty and attribution. A matching quote does not prove the draft is accurate."),
    ("Incomplete review", "Missing pages, unfinished checks or unavailable sources limit the result. "
     "Do not treat a partial review as a complete review."),
    ("[Confirm: ...] and factual review items", "Resolve placeholders and changed or unsupported facts with the witness "
     "and original sources. Recheck after any edit. A checkbox records your review of this exact text; it does not certify truth."),
    ("Copying and clearing", "Pilot file downloads are disabled. Copy reviewed text only to an operator-approved destination. "
     "Clear case releases this session's working data and registered uploads; it cannot erase provider copies, originals or text copied elsewhere."),
)


def render_reading_guide(container: Any, *, expanded: bool = False) -> None:
    from . import pilot
    if not pilot.enabled():
        return
    with container.expander("How to read these results", expanded=expanded):
        for label, explanation in READING_GUIDE:
            # Fixed labels have native headings; explanations are literal text.
            container.subheader(label)
            pilot.display(explanation, container=container, method="write")
