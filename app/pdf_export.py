"""PDF export for the final lay/witness statement (VA Form 21-10210 style).

Feature: Final Statement PDF Export
(`.implement/FEATURE-BRIEF.md`, feature id 0d76d70b-8dd6-4561-a874-f768d5929222)

Given the final statement/rewrite produced by either the Evaluate or Draft
pathway, `generate_statement_pdf` renders a plain, VA Form 21-10210-style PDF
in memory: a short header (claimed condition + witness role) followed by the
statement body, with a prominent verification disclaimer prepended whenever
unconfirmed `[Confirm ...]` placeholders remain in the text.

This module is feature-specific business logic, not shared telemetry
infrastructure, so the `FEATURE_ID` constant below is intentional and scoped
to this feature's own `error` telemetry call site (fired when PDF generation
raises). Shared telemetry infrastructure (`app/agiloop_telemetry.py`,
`app/telemetry_proxy.py`) never hardcodes a feature id — see those modules.
"""
from __future__ import annotations

import logging
import re
from io import BytesIO

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from .agiloop_telemetry import track_feature_error

logger = logging.getLogger("app.pdf_export")

# Feature: Final Statement PDF Export
FEATURE_ID = "0d76d70b-8dd6-4561-a874-f768d5929222"  # final-statement-pdf-export

# Matches the literal "[Confirm]" marker as well as this app's actual
# placeholder convention, "[Confirm: ...]" (see app/evaluate.py, app/draft.py
# rewrite prompts) — both start with the same "[Confirm" token.
_PLACEHOLDER_PATTERN = re.compile(r"\[Confirm\b", re.IGNORECASE)

DISCLAIMER_TEXT = (
    "VERIFICATION REQUIRED: This statement still contains one or more "
    "[Confirm] placeholders marking facts drawn from medical records that "
    "have not yet been personally confirmed by the witness. Review every "
    "bracketed item and resolve it before this statement is signed or "
    "submitted to VA on Form 21-10210."
)


def detect_unconfirmed_placeholders(statement_text: str) -> bool:
    """True when `statement_text` still contains an unconfirmed `[Confirm ...]` marker."""
    return bool(_PLACEHOLDER_PATTERN.search(statement_text or ""))


def _escape(text: str) -> str:
    """Escape characters significant to ReportLab's mini-XML paragraph markup."""
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _split_paragraphs(text: str) -> list[str]:
    """Split statement text on blank lines; fall back to one paragraph."""
    parts = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    return parts or [text.strip()]


def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "header": ParagraphStyle(
            "StatementHeader", parent=base["Heading2"], alignment=TA_LEFT, spaceAfter=6
        ),
        "meta": ParagraphStyle("StatementMeta", parent=base["Normal"], spaceAfter=14, leading=14),
        "disclaimer": ParagraphStyle(
            "Disclaimer",
            parent=base["Normal"],
            textColor="#7a0000",
            borderColor="#7a0000",
            borderWidth=1,
            borderPadding=8,
            spaceAfter=16,
            leading=14,
        ),
        "body": ParagraphStyle("StatementBody", parent=base["Normal"], leading=16, spaceAfter=10),
    }


def generate_statement_pdf(
    statement_text: str,
    condition: str,
    witness_role: str,
    has_unconfirmed_placeholders: bool | None = None,
) -> bytes:
    """Render a final lay/witness statement as a PDF for VA Form 21-10210.

    Args:
        statement_text: the full final statement/rewrite body.
        condition: claimed disability condition, shown in the header.
        witness_role: witness's relationship/role, shown in the header.
        has_unconfirmed_placeholders: explicit override. When `None` (the
            default) this is derived automatically by scanning
            `statement_text` for unresolved `[Confirm ...]` markers — see
            `detect_unconfirmed_placeholders`.

    Returns:
        Raw PDF bytes, ready to stream via `st.download_button`.

    Raises:
        ValueError: if `statement_text` is empty or blank.
    """
    try:
        if not statement_text or not statement_text.strip():
            raise ValueError("statement_text must be non-empty to generate a PDF")

        show_disclaimer = (
            has_unconfirmed_placeholders
            if has_unconfirmed_placeholders is not None
            else detect_unconfirmed_placeholders(statement_text)
        )

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=LETTER,
            topMargin=0.9 * inch,
            bottomMargin=0.9 * inch,
            leftMargin=1 * inch,
            rightMargin=1 * inch,
            title="VA Statement",
        )
        styles = _build_styles()

        story: list[object] = [
            Paragraph("Statement in Support of Claim (VA Form 21-10210)", styles["header"]),
            Paragraph(
                f"<b>Claimed condition:</b> {_escape(condition) or '—'}<br/>"
                f"<b>Witness role:</b> {_escape(witness_role) or '—'}",
                styles["meta"],
            ),
        ]

        if show_disclaimer:
            story.append(Paragraph(_escape(DISCLAIMER_TEXT), styles["disclaimer"]))
            story.append(Spacer(1, 6))

        for paragraph_text in _split_paragraphs(statement_text):
            story.append(Paragraph(_escape(paragraph_text), styles["body"]))

        doc.build(story)
        return buffer.getvalue()
    except Exception as exc:  # noqa: BLE001 - boundary error handling; re-raised below
        track_feature_error(FEATURE_ID, exc)
        logger.error("PDF generation failed: %s", exc, exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Timeline PDF export
# ---------------------------------------------------------------------------

TIMELINE_FEATURE_ID = "timeline-pdf-export"  # medical record timeline PDF

# Type colors for timeline events (matching the UI colors)
TYPE_COLORS = {
    "diagnosis": "#3b82f6",  # Blue
    "symptom": "#ef4444",  # Red
    "treatment": "#10b981",  # Green
    "medication": "#8b5cf6",  # Purple
    "test_result": "#f59e0b",  # Amber
    "hospitalization": "#ec4899",  # Pink
    "provider_visit": "#06b6d4",  # Cyan
    "in_service_event": "#f97316",  # Orange
    "functional_limitation": "#6366f1",  # Indigo
    "observable_behavior": "#14b8a6",  # Teal
    "administrative": "#6b7280",  # Gray
    "other": "#9ca3af",  # Light gray
}


def _build_timeline_styles() -> dict[str, ParagraphStyle]:
    """Build styles for timeline PDF."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TimelineTitle", parent=base["Heading1"], alignment=TA_LEFT, spaceAfter=12
        ),
        "subtitle": ParagraphStyle(
            "TimelineSubtitle", parent=base["Normal"], spaceAfter=16, leading=14
        ),
        "year_header": ParagraphStyle(
            "YearHeader", parent=base["Heading2"], spaceBefore=12, spaceAfter=6
        ),
        "event_date": ParagraphStyle(
            "EventDate", parent=base["Normal"], fontName="Helvetica-Bold", spaceAfter=2
        ),
        "event_type": ParagraphStyle(
            "EventType", parent=base["Normal"], fontName="Helvetica-BoldOblique", spaceAfter=4
        ),
        "event_desc": ParagraphStyle(
            "EventDesc", parent=base["Normal"], leading=14, spaceAfter=4
        ),
        "event_quote": ParagraphStyle(
            "EventQuote",
            parent=base["Normal"],
            leftIndent=20,
            textColor="#666666",
            leading=12,
            spaceAfter=4,
        ),
        "event_source": ParagraphStyle(
            "EventSource", parent=base["Normal"], textColor="#888888", fontName="Helvetica-Oblique"
        ),
        "gap_header": ParagraphStyle(
            "GapHeader",
            parent=base["Normal"],
            textColor="#dc2626",
            fontName="Helvetica-Bold",
            spaceBefore=8,
            spaceAfter=4,
        ),
    }


def generate_timeline_pdf(events: list[Any]) -> bytes:
    """Generate a PDF of the medical record timeline.

    Args:
        events: List of TimelineEvent objects to render.

    Returns:
        Raw PDF bytes ready for download.
    """
    try:
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib import colors
    except ImportError:
        raise RuntimeError("reportlab is required for PDF export")

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        title="Medical Record Timeline",
    )
    styles = _build_timeline_styles()

    story: list[object] = []

    # Title
    story.append(Paragraph("Medical Record Timeline", styles["title"]))
    story.append(Paragraph(
        f"Generated from medical record review • {len(events)} events",
        styles["subtitle"],
    ))
    story.append(Spacer(1, 12))

    # Group events by year
    current_year = None
    for event in events:
        year = event.date_sortable[:4] if event.date_sortable != "9999-99-99" else "Unknown"

        if year != current_year:
            current_year = year
            story.append(Paragraph(f"## {year}", styles["year_header"]))

        # Event date
        date_display = event.date if event.date and event.date != "unknown" else "Date unknown"
        story.append(Paragraph(date_display, styles["event_date"]))

        # Event type
        type_label = event.type.replace("_", " ").title()
        story.append(Paragraph(type_label, styles["event_type"]))

        # Description
        story.append(Paragraph(event.full_description, styles["event_desc"]))

        # Quote if present
        if event.quote:
            story.append(Paragraph(f"\"{event.quote}\"", styles["event_quote"]))

        # Source
        story.append(Paragraph(f"Source: {event.source}", styles["event_source"]))

        story.append(Spacer(1, 8))

    doc.build(story)
    return buffer.getvalue()

