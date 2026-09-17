"""Fact Citation Exporter — CSV / Markdown / PDF export of medical-digest facts.

Feature: Fact Citation Exporter
(`.implement/FEATURE-BRIEF.md`, feature id 051bb638-ac1c-40cf-95f5-164779b4382c)

Reuses the in-memory `MedicalDigest` produced by `app.medical_review` during an
Evaluate run (`EvaluationResult.digest`, see `app/evaluate.py`). Per
`.implement/technical-spec.md` there are no schema changes to `MedicalFact` /
`MedicalDigest` — this module is pure filtering + formatting logic layered on
top of the existing dataclasses, triggered from the "Export Facts" button in
`app/views/evaluate_view.py`.

This module is feature-specific business logic (not shared telemetry
infrastructure), so the `FEATURE_ID` constant below is intentional and scoped
to this feature's own `error` telemetry call site. Shared telemetry
infrastructure (`app/agiloop_telemetry.py`, `app/telemetry_proxy.py`) never
hardcodes a feature id — see those modules' docstrings.
"""
from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import dataclass
from typing import Iterable, Literal
import unicodedata

from fpdf import FPDF

from .agiloop_telemetry import track_feature_error
from .logging_config import get_request_id
from .medical_review import MedicalDigest, MedicalFact

logger = logging.getLogger("app.exporter")

# Feature: Fact Citation Exporter
FEATURE_ID = "051bb638-ac1c-40cf-95f5-164779b4382c"  # fact-citation-exporter

ExportFormat = Literal["csv", "md", "pdf"]

FACT_COLUMNS = ("Date", "Type", "Fact Description", "Source Quote", "Page(s)", "Document Name")

# ------------------------------------------------------------ CSV injection guard
# Facts originate from uploaded medical-record content (untrusted input flowing
# through the digest LLM), so a formula-guard is required before any value is
# written to a spreadsheet-openable format — mirrors the fix already applied to
# `app.documents.export_citation_index` for the same class of export.
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _csv_formula_guard(value: str) -> str:
    """Neutralize CSV/spreadsheet formula injection (OWASP CSV Injection).

    If a cell's value begins with ``=``, ``+``, ``-``, ``@`` (or a leading
    tab/CR), spreadsheet applications such as Excel or Google Sheets may
    interpret it as a formula when the exported file is opened. Prefixing
    with a single quote forces the cell to be treated as literal text while
    leaving the visible content unchanged.
    """
    text = value or ""
    if text.startswith(_CSV_FORMULA_PREFIXES):
        return "'" + text
    return text


# --------------------------------------------------------------- source parsing
_SOURCE_PATTERNS = (
    # "[filename — page 3]" / "filename — page 3" / "filename - pages 3-4"
    # and the block-addressed form for .txt/.md/.docx ("filename — block 2").
    re.compile(
        r"^\[?(?P<doc>.+?)\s*[—-]\s*(?:pages?|blocks?)\s*(?P<pages>[\d,\-\s]+)\]?$",
        re.IGNORECASE,
    ),
    # "filename p.3" / "filename b.2" (app.documents.DocumentPage.label), plus
    # the chunk-span form the digest now emits: "clinic.pdf p.3-p.9".
    re.compile(r"^(?P<doc>.+?)\s+[pb]\.(?P<pages>[\d,\-]+)$", re.IGNORECASE),
)


def parse_source(source: str) -> tuple[str, str]:
    """Best-effort split of a `MedicalFact.source` string into (document, page(s)).

    ``MedicalFact`` now carries ``document``/``page`` fields set from the chunk's
    page span (see ``app.medical_review``), so a fact whose citation could be
    resolved needs no parsing; callers should prefer those fields when present.
    This remains for the unresolved case, where ``source`` is free text: usually a
    page citation such as ``"records.pdf — page 3"``, sometimes a coarser
    fallback like ``"chunk 2/9"`` when the model could not resolve a page. It
    parses the common citation shapes and otherwise reports the raw string as the
    document name with an empty page — it never guesses/invents a page.
    """
    text = (source or "").strip()
    if not text:
        return "", ""
    for pattern in _SOURCE_PATTERNS:
        match = pattern.match(text)
        if match:
            doc = match.group("doc").strip(" [")
            pages = re.sub(r"\s+", "", match.group("pages")).replace(",", ", ")
            return doc, pages
    return text, ""


# ------------------------------------------------------------------- filtering
def filter_facts(
    facts: Iterable[MedicalFact],
    *,
    only_rubric_cited: bool = False,
    only_positive_outcomes: bool = False,
    rubric_cited_sources: set[str] | None = None,
    positive_outcome_sources: set[str] | None = None,
) -> list[MedicalFact]:
    """Apply the two "Export Facts" filter checkboxes to a fact list.

    ``rubric_cited_sources`` / ``positive_outcome_sources`` are sets of
    ``MedicalFact.source`` values the caller has already cross-referenced
    against the evaluation's rubric-verification results (see
    ``app/views/evaluate_view.py``, which derives them from
    ``EvaluationResult.verifications``). When a filter flag is on but its
    matching set is ``None``/empty, the result is an empty list (no
    unfiltered fallback) — the caller must supply the set whenever the
    checkbox is checked.
    """
    result = list(facts)
    if only_rubric_cited:
        cited = rubric_cited_sources or set()
        result = [f for f in result if f.source in cited]
    if only_positive_outcomes:
        positive = positive_outcome_sources or set()
        result = [f for f in result if f.source in positive]
    return result


# -------------------------------------------------------------------- summary
@dataclass
class ExportSummary:
    total_facts: int
    date_range: str
    conditions: str


_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


def _summarize(facts: list[MedicalFact], digest: MedicalDigest) -> ExportSummary:
    years: list[int] = []
    for fact in facts:
        years.extend(int(y) for y in _YEAR_RE.findall(fact.date))
    if years:
        date_range = str(min(years)) if min(years) == max(years) else f"{min(years)}\u2013{max(years)}"
    else:
        date_range = "Unknown"
    conditions = ", ".join(digest.conditions) if digest.conditions else "None recorded"
    return ExportSummary(total_facts=len(facts), date_range=date_range, conditions=conditions)


# --------------------------------------------------------------------- CSV
def export_facts_csv(facts: list[MedicalFact], digest: MedicalDigest) -> bytes:
    """Render `facts` as CSV bytes: header row, summary row, then one row per fact."""
    summary = _summarize(facts, digest)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(FACT_COLUMNS)
    writer.writerow(
        [
            "Summary",
            "",
            f"Total facts: {summary.total_facts} | Date range: {summary.date_range} | "
            f"Conditions mentioned: {summary.conditions}",
            "",
            "",
            "",
        ]
    )
    for fact in facts:
        doc_name, pages = parse_source(fact.source)
        writer.writerow(
            [
                _csv_formula_guard(fact.date),
                _csv_formula_guard(fact.type),
                _csv_formula_guard(fact.description),
                _csv_formula_guard(fact.quote),
                _csv_formula_guard(pages),
                _csv_formula_guard(doc_name),
            ]
        )
    return buffer.getvalue().encode("utf-8")


# ---------------------------------------------------------------- markdown
def _md_escape(value: str) -> str:
    return (value or "").replace("|", "\\|").replace("\n", " ").strip()


def export_facts_markdown(facts: list[MedicalFact], digest: MedicalDigest) -> bytes:
    """Render `facts` as a markdown document: summary section + fact table."""
    summary = _summarize(facts, digest)
    lines = [
        "# Medical Digest Fact Citations",
        "",
        "## Summary",
        "",
        f"- **Total facts:** {summary.total_facts}",
        f"- **Date range:** {summary.date_range}",
        f"- **Conditions mentioned:** {summary.conditions}",
        "",
        "## Facts",
        "",
        "| " + " | ".join(FACT_COLUMNS) + " |",
        "|" + "|".join(["---"] * len(FACT_COLUMNS)) + "|",
    ]
    for fact in facts:
        doc_name, pages = parse_source(fact.source)
        cells = [fact.date, fact.type, fact.description, fact.quote, pages, doc_name]
        lines.append("| " + " | ".join(_md_escape(c) for c in cells) + " |")
    return ("\n".join(lines) + "\n").encode("utf-8")


# --------------------------------------------------------------------- PDF
# Landscape Letter gives ~259mm of usable width (vs. ~176mm portrait) so all
# six columns — including the free-text Fact Description and Source Quote
# columns — stay readable without shrinking the font below print-legible
# size; still a standard, e-filing-safe Letter page per the acceptance
# criteria ("fits standard letter size").
_PDF_COL_WIDTHS = (18, 22, 60, 60, 16, 40)  # mm; sums to 216mm, well within 259mm usable width

# fpdf2's built-in "core" fonts (Helvetica, etc.) only support latin-1. Medical
# record text is untrusted, arbitrary input that frequently contains smart
# quotes/dashes/ellipses (word-processor exports) or other non-latin-1
# characters; rather than raising `FPDFUnicodeEncodingException` mid-export,
# normalize common punctuation to an ASCII equivalent and replace anything
# still unencodable rather than failing the whole export.
_PDF_UNICODE_REPLACEMENTS = {
    "–": "-", "—": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"', "…": "...",
}


def _pdf_safe_text(value: str) -> str:
    text = value or ""
    for char, replacement in _PDF_UNICODE_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    try:
        text.encode("latin-1")
        return text
    except UnicodeEncodeError:
        normalized = unicodedata.normalize("NFKD", text)
        return normalized.encode("latin-1", errors="replace").decode("latin-1")


def export_facts_pdf(facts: list[MedicalFact], digest: MedicalDigest) -> bytes:
    """Render `facts` as a landscape-Letter PDF table with a summary header."""
    summary = _summarize(facts, digest)
    pdf = FPDF(orientation="L", format="Letter")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Medical Digest Fact Citations", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0,
        5,
        _pdf_safe_text(
            f"Total facts: {summary.total_facts}  |  Date range: {summary.date_range}  |  "
            f"Conditions mentioned: {summary.conditions}"
        ),
    )
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 8)
    with pdf.table(col_widths=_PDF_COL_WIDTHS, text_align="LEFT") as table:
        header_row = table.row()
        for column in FACT_COLUMNS:
            header_row.cell(column)
        for fact in facts:
            doc_name, pages = parse_source(fact.source)
            row = table.row()
            for value in (fact.date, fact.type, fact.description, fact.quote, pages, doc_name):
                row.cell(_pdf_safe_text(value or ""))
    return bytes(pdf.output())


# --------------------------------------------------------------- entry point
_EXPORTERS = {
    "csv": export_facts_csv,
    "md": export_facts_markdown,
    "pdf": export_facts_pdf,
}


def export_facts(
    digest: MedicalDigest,
    fmt: ExportFormat,
    *,
    only_rubric_cited: bool = False,
    only_positive_outcomes: bool = False,
    rubric_cited_sources: set[str] | None = None,
    positive_outcome_sources: set[str] | None = None,
) -> bytes:
    """Filter `digest.facts` per the two checkboxes and render them as `fmt` bytes.

    Raises:
        ValueError: for an unsupported `fmt`.
    """
    normalized = (fmt or "").strip().lower()
    exporter = _EXPORTERS.get(normalized)
    if exporter is None:
        raise ValueError(f"Unsupported export format: {fmt!r} (expected 'csv', 'md', or 'pdf')")

    request_id = get_request_id() or "-"
    facts = filter_facts(
        digest.facts,
        only_rubric_cited=only_rubric_cited,
        only_positive_outcomes=only_positive_outcomes,
        rubric_cited_sources=rubric_cited_sources,
        positive_outcome_sources=positive_outcome_sources,
    )
    try:
        data = exporter(facts, digest)
    except Exception as exc:  # noqa: BLE001 - boundary error handling; re-raised below
        track_feature_error(FEATURE_ID, exc, format=normalized)
        logger.error(
            "fact export failed format=%s error=%s",
            normalized,
            exc,
            exc_info=True,
            extra={
                "request_id": request_id,
                "phase": "export_facts",
                "status": "error",
                "format": normalized,
            },
        )
        raise
    logger.info(
        "fact export ok format=%s fact_count=%d filtered=%s",
        normalized,
        len(facts),
        only_rubric_cited or only_positive_outcomes,
        extra={
            "request_id": request_id,
            "phase": "export_facts",
            "status": "ok",
            "format": normalized,
            "fact_count": len(facts),
            "filtered": only_rubric_cited or only_positive_outcomes,
        },
    )
    return data


__all__ = [
    "FEATURE_ID",
    "FACT_COLUMNS",
    "ExportSummary",
    "export_facts",
    "export_facts_csv",
    "export_facts_markdown",
    "export_facts_pdf",
    "filter_facts",
    "parse_source",
]
