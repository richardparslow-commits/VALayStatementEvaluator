"""Tests for recognising a VA.gov medical-records export among uploads.

The point of the detection is that a real record set arrives as a downloaded PDF
uploaded by the user; labelling it as a VA.gov source is only correct if the file
really came from VA.gov, so the negative cases here matter as much as the positive.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app.documents import DocumentPage, ExtractedDocument  # noqa: E402
from app.va_gov_export import is_va_gov_records_export, split_va_gov_exports  # noqa: E402

VA_GOV_EXPORT_TEXT = (
    "Download your medical records\n"
    "va.gov | My HealtheVet\n"
    "Report date: September 15, 2026\n"
    "Facility: VA Medical Center - Austin\n"
    "Provider: Smith, John MD\n"
    "Medications, immunizations, and lab results follow.\n"
    "This report was downloaded from VA.gov.\n"
)

PRIVATE_RECORDS_TEXT = (
    "Valley Regional Clinic\n"
    "Patient medical records — office visit\n"
    "Provider: Dr. Jane Doe\n"
    "Medications and lab results follow.\n"
)


def _doc(filename: str, *page_texts: str) -> ExtractedDocument:
    return ExtractedDocument(
        filename=filename,
        pages=[
            DocumentPage(filename=filename, page=i + 1, text=text)
            for i, text in enumerate(page_texts)
        ],
    )


class TestIsVaGovRecordsExport(unittest.TestCase):
    def test_recognises_a_va_gov_export(self) -> None:
        self.assertTrue(is_va_gov_records_export(_doc("records.pdf", VA_GOV_EXPORT_TEXT)))

    def test_recognises_the_script_default_filename_with_va_gov_header(self) -> None:
        doc = _doc("VA_medical_records.pdf", "Download your medical records\nva.gov\n")
        self.assertTrue(is_va_gov_records_export(doc))

    def test_private_provider_records_are_not_va_gov(self) -> None:
        self.assertFalse(
            is_va_gov_records_export(_doc("records.pdf", PRIVATE_RECORDS_TEXT))
        )

    def test_va_gov_page_without_a_records_context_is_not_an_export(self) -> None:
        # A benefits letter or appointment notice downloaded from VA.gov is not a
        # record set, and labelling it as one would misrepresent the input.
        doc = _doc("letter.pdf", "va.gov\nDepartment of Veterans Affairs\nYour benefits")
        self.assertFalse(is_va_gov_records_export(doc))

    def test_renaming_a_private_pdf_is_not_enough(self) -> None:
        doc = _doc("VA_medical_records.pdf", PRIVATE_RECORDS_TEXT)
        self.assertFalse(is_va_gov_records_export(doc))

    def test_filename_hint_counts_as_the_records_context(self) -> None:
        doc = _doc("my_medical_records.pdf", "Download your medical records\nva.gov\n")
        self.assertTrue(is_va_gov_records_export(doc))

    def test_empty_or_textless_documents_are_not_exports(self) -> None:
        self.assertFalse(is_va_gov_records_export(_doc("empty.pdf")))
        self.assertFalse(is_va_gov_records_export(_doc("blank.pdf", "   ")))
        self.assertFalse(is_va_gov_records_export(object()))

    def test_marker_beyond_the_scan_window_is_ignored(self) -> None:
        """Only the first pages are scanned, so a long record set is not walked."""
        pages = ["Page " + str(i) + " of routine records.\n" for i in range(1, 6)]
        pages[4] = VA_GOV_EXPORT_TEXT  # page 5, outside SCAN_PAGES
        self.assertFalse(is_va_gov_records_export(_doc("records.pdf", *pages)))

    def test_marker_inside_the_scan_window_is_found(self) -> None:
        pages = ["Page 1: intake.\n", VA_GOV_EXPORT_TEXT, "Page 3.\n"]
        self.assertTrue(is_va_gov_records_export(_doc("records.pdf", *pages)))


class TestSplitVaGovExports(unittest.TestCase):
    def test_splits_and_names_only_the_exports(self) -> None:
        exports, names = split_va_gov_exports(
            [
                _doc("private.pdf", PRIVATE_RECORDS_TEXT),
                _doc("va_records.pdf", VA_GOV_EXPORT_TEXT),
                _doc("notes.txt", "Notes about the appointment."),
            ]
        )
        self.assertEqual(names, ["va_records.pdf"])
        self.assertEqual([d.filename for d in exports], ["va_records.pdf"])

    def test_no_exports_is_an_empty_result(self) -> None:
        exports, names = split_va_gov_exports([_doc("private.pdf", PRIVATE_RECORDS_TEXT)])
        self.assertEqual(exports, [])
        self.assertEqual(names, [])


if __name__ == "__main__":
    unittest.main()
