"""Tests for `app.pdf_export.generate_statement_pdf`.

Covers the disclaimer / no-disclaimer branches (F2.S1/F2.S2 acceptance
criteria) using real PDF text extraction (pypdf) rather than asserting on
ReportLab internals, so the tests validate what a reviewer would actually see
when opening the exported PDF.
"""
from __future__ import annotations

import sys
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pypdf import PdfReader  # noqa: E402

from app.pdf_export import (  # noqa: E402
    DISCLAIMER_TEXT,
    detect_unconfirmed_placeholders,
    generate_statement_pdf,
)

STATEMENT_WITH_PLACEHOLDER = (
    "I am the spouse of the veteran. I have observed his back pain worsen "
    "since 2020. [Confirm: exact date of the initial injury] He has trouble "
    "standing for long periods."
)

STATEMENT_WITHOUT_PLACEHOLDER = (
    "I am the spouse of the veteran. I have observed his back pain worsen "
    "since 2020. He has trouble standing for long periods and requires help "
    "with household chores."
)


def _extract_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


class TestGenerateStatementPdf(unittest.TestCase):
    def test_returns_valid_pdf_bytes(self):
        pdf_bytes = generate_statement_pdf(
            STATEMENT_WITHOUT_PLACEHOLDER, "Lumbar strain", "Spouse"
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_header_contains_condition_and_witness_role(self):
        pdf_bytes = generate_statement_pdf(
            STATEMENT_WITHOUT_PLACEHOLDER, "Lumbar strain", "Spouse"
        )
        text = _extract_text(pdf_bytes)
        self.assertIn("Lumbar strain", text)
        self.assertIn("Spouse", text)
        self.assertIn("VA Form 21-10210", text)

    def test_includes_disclaimer_when_placeholder_present(self):
        pdf_bytes = generate_statement_pdf(
            STATEMENT_WITH_PLACEHOLDER, "PTSD", "Fellow service member"
        )
        text = _extract_text(pdf_bytes)
        self.assertIn("VERIFICATION REQUIRED", text)
        self.assertIn("[Confirm", text)
        self.assertIn(STATEMENT_WITH_PLACEHOLDER.split(".")[0], text)

    def test_no_disclaimer_when_no_placeholder_remains(self):
        pdf_bytes = generate_statement_pdf(
            STATEMENT_WITHOUT_PLACEHOLDER, "PTSD", "Fellow service member"
        )
        text = _extract_text(pdf_bytes)
        self.assertNotIn("VERIFICATION REQUIRED", text)
        self.assertNotIn("[Confirm", text)
        # Only header + statement text — no extra disclaimer content leaks in.
        for word in DISCLAIMER_TEXT.split()[:3]:
            self.assertNotIn(word, text)

    def test_explicit_has_unconfirmed_placeholders_overrides_detection(self):
        # Statement has no literal placeholder, but caller explicitly forces
        # the disclaimer on (e.g. an upstream flag not captured by regex).
        pdf_bytes = generate_statement_pdf(
            STATEMENT_WITHOUT_PLACEHOLDER,
            "PTSD",
            "Fellow service member",
            has_unconfirmed_placeholders=True,
        )
        text = _extract_text(pdf_bytes)
        self.assertIn("VERIFICATION REQUIRED", text)

    def test_empty_statement_raises_value_error(self):
        with self.assertRaises(ValueError):
            generate_statement_pdf("   ", "PTSD", "Spouse")

    def test_missing_condition_and_role_render_placeholder_dash(self):
        pdf_bytes = generate_statement_pdf(STATEMENT_WITHOUT_PLACEHOLDER, "", "")
        text = _extract_text(pdf_bytes)
        self.assertIn("—", text)

    def test_generation_error_is_tracked_and_reraised(self):
        with patch(
            "app.pdf_export.SimpleDocTemplate.build",
            side_effect=RuntimeError("boom"),
        ), patch("app.pdf_export.track_feature_error") as mock_track:
            with self.assertRaises(RuntimeError):
                generate_statement_pdf(STATEMENT_WITHOUT_PLACEHOLDER, "PTSD", "Spouse")
            mock_track.assert_called_once()
            args = mock_track.call_args[0]
            self.assertEqual(args[0], "0d76d70b-8dd6-4561-a874-f768d5929222")
            self.assertIsInstance(args[1], RuntimeError)


class TestDetectUnconfirmedPlaceholders(unittest.TestCase):
    def test_detects_literal_confirm_marker(self):
        self.assertTrue(detect_unconfirmed_placeholders("some text [Confirm] more text"))

    def test_detects_app_convention_confirm_colon_marker(self):
        self.assertTrue(
            detect_unconfirmed_placeholders("text [Confirm: exact date] more text")
        )

    def test_false_when_no_marker_present(self):
        self.assertFalse(detect_unconfirmed_placeholders("nothing to confirm here"))

    def test_false_for_empty_string(self):
        self.assertFalse(detect_unconfirmed_placeholders(""))

    def test_case_insensitive_match(self):
        self.assertTrue(detect_unconfirmed_placeholders("please [confirm: date] this"))


class TestPlaceholderDisclaimerIntegration(unittest.TestCase):
    """F2.S2 — placeholder check drives the conditional disclaimer end-to-end."""

    def test_explicit_false_suppresses_disclaimer_even_if_marker_present(self):
        # An explicit has_unconfirmed_placeholders=False always wins over
        # auto-detection (e.g. an upstream caller has already resolved the
        # placeholders in its own copy but the raw text still shows one).
        pdf_bytes = generate_statement_pdf(
            STATEMENT_WITH_PLACEHOLDER,
            "PTSD",
            "Spouse",
            has_unconfirmed_placeholders=False,
        )
        text = _extract_text(pdf_bytes)
        self.assertNotIn("VERIFICATION REQUIRED", text)

    def test_pdf_with_disclaimer_still_contains_full_statement_body(self):
        pdf_bytes = generate_statement_pdf(
            STATEMENT_WITH_PLACEHOLDER, "PTSD", "Spouse"
        )
        text = _extract_text(pdf_bytes)
        self.assertIn("trouble", text)
        self.assertIn("standing", text)

    def test_pdf_without_disclaimer_contains_only_header_and_body(self):
        pdf_bytes = generate_statement_pdf(
            STATEMENT_WITHOUT_PLACEHOLDER, "PTSD", "Spouse"
        )
        text = _extract_text(pdf_bytes)
        self.assertIn("VA Form 21-10210", text)
        self.assertIn("PTSD", text)
        self.assertIn("Spouse", text)
        self.assertIn("household chores", text)
        self.assertNotIn("VERIFICATION", text)


if __name__ == "__main__":
    unittest.main()
