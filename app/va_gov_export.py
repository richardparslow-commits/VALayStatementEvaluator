"""Recognise a VA.gov medical-records export among uploaded files.

The realistic way to get a *real* record set into the app is to download it from
VA.gov (see ``scripts/va_records_download.py``) and upload the PDF. Detecting that
here lets the upload be labelled as the VA.gov source, so the merged records
summary says where it came from instead of showing an anonymous file.

The decision is made on the document's own text, not its filename: anyone can
rename a file, and a clinician's PDF renamed to ``VA_medical_records.pdf`` is not a
VA.gov export. A filename hint only widens the "is this about records at all?"
test, and a VA.gov marker must be present in the text for a positive match.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

# Phrases only a VA.gov-produced document carries.
VA_GOV_MARKERS = (
    r"\bva\.gov\b",
    r"download your medical records",
    r"my healthevet",
    r"department of veterans affairs",
    r"\bva medical center\b",
)

# A records context. Required as well, so a VA.gov page about something else (a
# benefits letter, an appointment notice) is not labelled as a record set.
RECORD_MARKERS = (
    r"medical record",
    r"health record",
    r"medical histor",
    r"encounter",
    r"provider",
    r"facility",
    r"medication",
    r"immunization",
    r"lab\b",
    r"diagnos",
)

# Filenames VA.gov and My HealtheVet produce for this report.
FILENAME_HINTS = (
    r"medical.?record",
    r"va.?record",
    r"myhealthevet",
    r"blue.?button",
)

# Only the first pages are inspected: the report's header carries the markers, and
# a multi-thousand-page record set should not be scanned in full on every rerun.
SCAN_PAGES = 3
SCAN_CHARS = 8_000


def _scan_text(document: Any) -> str:
    """First ``SCAN_PAGES`` pages of ``document``, capped at ``SCAN_CHARS``."""
    pages = getattr(document, "pages", None) or []
    parts: list[str] = []
    for page in list(pages)[:SCAN_PAGES]:
        parts.append(str(getattr(page, "text", "") or ""))
    return "\n".join(parts)[:SCAN_CHARS].lower()


def _matches_any(markers: Iterable[str], text: str) -> bool:
    return any(re.search(marker, text) for marker in markers)


def is_va_gov_records_export(document: Any) -> bool:
    """True when ``document`` looks like a medical-records export from VA.gov.

    Conservative by design: a VA.gov marker in the text *and* a records context
    (in the text or the filename). Anything else stays an ordinary upload, because
    mislabelling a private provider's records as VA.gov data would be worse than
    making the user pick the source themselves.
    """
    filename = str(getattr(document, "filename", "") or "").lower()
    text = _scan_text(document)
    if not text:
        return False
    if not _matches_any(VA_GOV_MARKERS, text):
        return False
    return _matches_any(RECORD_MARKERS, text) or _matches_any(FILENAME_HINTS, filename)


def split_va_gov_exports(documents: Iterable[Any]) -> tuple[list[Any], list[str]]:
    """Return ``(va_gov_documents, filenames)`` for labeling by the uploader."""
    exports: list[Any] = []
    names: list[str] = []
    for document in documents:
        if is_va_gov_records_export(document):
            exports.append(document)
            names.append(str(getattr(document, "filename", "") or "?"))
    return exports, names


# VA.gov's "Download your medical records" report is organised by category, and
# those category headings are how a VSO or attorney reads a claim file ("what does
# the Problem list say"). Recognizing them lets each extracted fact carry the
# section it came from, so the statement work can say which part of the record it
# is drawing on. Matched on a whole normalized line, so a patient note that merely
# mentions "medications" mid-sentence is not mistaken for the heading.
SECTION_HEADINGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Patient information", ("patient information", "patient demographics")),
    ("Problem list", ("problem list", "active problems", "problems")),
    ("Allergies", ("allergies", "allergy list", "allergies and adverse reactions")),
    (
        "Medications",
        ("medications", "active medications", "medication list", "outpatient medications"),
    ),
    ("Immunizations", ("immunizations", "immunization history", "immunizations given")),
    ("Vitals", ("vitals", "vital signs", "measurements and vitals")),
    (
        "Lab and test results",
        ("lab and test results", "laboratory results", "lab results", "test results"),
    ),
    ("Imaging", ("imaging", "diagnostic imaging", "radiology", "radiology results")),
    (
        "Care summaries and notes",
        ("care summaries and notes", "progress notes", "clinical notes", "care summaries"),
    ),
    ("Appointments", ("appointments", "appointment history", "visits")),
    (
        "Admissions and discharges",
        ("admissions and discharges", "admission and discharge", "hospitalizations"),
    ),
    ("Providers", ("providers", "provider list", "treating providers")),
    (
        "Military service information",
        ("military service information", "service history", "military service"),
    ),
)

# How far into a page to look for a heading. VA.gov puts the category at the top
# of the first page of that category; scanning a whole page would let a quoted
# sentence elsewhere on it be read as a heading.
_SECTION_SCAN_LINES = 40
_SECTION_MAX_HEADING_CHARS = 60


def _normalized_heading(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", line.strip().lower())).strip()


def _heading_on_page(text: str, current: str) -> str:
    """Section a page belongs to, carrying ``current`` forward when unchanged."""
    for line in text.splitlines()[:_SECTION_SCAN_LINES]:
        normalized = _normalized_heading(line)
        if not normalized or len(normalized) > _SECTION_MAX_HEADING_CHARS:
            continue
        for section, headings in SECTION_HEADINGS:
            if normalized in headings:
                return section
    return current


def section_map(document: Any) -> dict[int, str]:
    """Map page number -> VA.gov record section for a section-structured export.

    Every page is scanned (a category can begin mid-document) and the section is
    carried forward until the next heading. Pages before the first recognized
    heading — and every page of a document that has none — map to nothing: an
    empty section is honest, a guessed one is not.
    """
    mapping: dict[int, str] = {}
    current = ""
    for page in getattr(document, "pages", None) or []:
        current = _heading_on_page(str(getattr(page, "text", "") or ""), current)
        if current:
            mapping[int(getattr(page, "page", 0) or 0)] = current
    return mapping


__all__ = [
    "VA_GOV_MARKERS",
    "RECORD_MARKERS",
    "FILENAME_HINTS",
    "SECTION_HEADINGS",
    "is_va_gov_records_export",
    "split_va_gov_exports",
    "section_map",
]
