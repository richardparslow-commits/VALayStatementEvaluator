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


__all__ = [
    "VA_GOV_MARKERS",
    "RECORD_MARKERS",
    "FILENAME_HINTS",
    "is_va_gov_records_export",
    "split_va_gov_exports",
]
