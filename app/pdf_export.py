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
