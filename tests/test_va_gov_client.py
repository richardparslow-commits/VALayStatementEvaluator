"""Offline unit tests for the VA.gov client (mock-mode auth/fetch/merge)."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app import va_gov_client  # noqa: E402
from app.va_gov_client import (  # noqa: E402
    VaGovError,
    VaGovFetchResult,
    VaGovSession,
    authenticate_va_gov,
    fetch_va_records,
    merge_records,
)


class TestAuthenticateVaGov(unittest.TestCase):
    def test_mock_mode_returns_session_token_for_valid_credentials(self):
        with patch.dict("os.environ", {"VA_GOV_API_BASE_URL": ""}, clear=False):
            session = authenticate_va_gov("veteran1", "hunter2")
        self.assertIsInstance(session, VaGovSession)
        self.assertTrue(session.token)
        self.assertEqual(session.patient_id, "veteran1")

    def test_rejects_blank_credentials(self):
        with self.assertRaises(VaGovError) as ctx:
            authenticate_va_gov("", "")
        self.assertEqual(ctx.exception.error_class, "invalid_credentials")

    def test_never_persists_credentials_on_session_object(self):
        session = authenticate_va_gov("veteran2", "s3cret!")
        # The session dataclass must not carry the raw password anywhere.
        self.assertNotIn("s3cret", vars(session).values())


class TestFetchVaRecords(unittest.TestCase):
    def test_mock_mode_returns_extracted_documents(self):
        session = VaGovSession(token="mock-token", patient_id="veteran1")
        with patch.dict("os.environ", {"VA_GOV_API_BASE_URL": ""}, clear=False):
            result = fetch_va_records(session)
        self.assertIsInstance(result, VaGovFetchResult)
        self.assertFalse(result.partial)
        self.assertGreater(len(result.documents), 0)
        self.assertEqual(result.retrieved, result.expected)

    def test_real_mode_partial_response_returns_partial_results_with_metadata(self):
        session = VaGovSession(token="real-token", patient_id="veteran3")
        payload = {
            "records": [{"filename": "note1", "text": "Visit note text."}],
            "expected": 3,
            "partial": True,
            "error": "Connection dropped after 1 of 3 records.",
        }
        with patch.dict("os.environ", {"VA_GOV_API_BASE_URL": "https://api.va.gov"}, clear=False):
            with patch.object(va_gov_client, "_request_records", return_value=payload):
                result = fetch_va_records(session)
        self.assertTrue(result.partial)
        self.assertEqual(result.retrieved, 1)
        self.assertEqual(result.expected, 3)
        self.assertIn("Connection dropped", result.error_message)

    def test_real_mode_connection_error_exhausts_retries_and_returns_partial(self):
        session = VaGovSession(token="real-token", patient_id="veteran4")
        with patch.dict("os.environ", {"VA_GOV_API_BASE_URL": "https://api.va.gov"}, clear=False):
            with patch.object(va_gov_client, "_request_records", side_effect=OSError("boom")):
                with patch("time.sleep"):
                    result = fetch_va_records(session)
        self.assertTrue(result.partial)
        self.assertEqual(result.documents, [])
        self.assertIsNotNone(result.error_message)


class TestMergeRecords(unittest.TestCase):
    def test_merges_and_labels_multiple_sources(self):
        session = VaGovSession(token="mock-token", patient_id="veteran5")
        with patch.dict("os.environ", {"VA_GOV_API_BASE_URL": ""}, clear=False):
            va_result = fetch_va_records(session)
        from app.documents import document_from_text

        upload_doc = document_from_text("upload_record.txt", "Uploaded record text.")
        merged = merge_records({"Upload": [upload_doc], "VA.gov": va_result.documents})

        self.assertEqual(merged.sources_merged, 2)
        self.assertEqual(len(merged.documents), 1 + len(va_result.documents))
        sources_in_summary = {row.source for row in merged.summary}
        self.assertEqual(sources_in_summary, {"Upload", "VA.gov"})

    def test_deduplicates_identical_documents_across_sources(self):
        from app.documents import document_from_text

        doc_a = document_from_text("shared.txt", "Same content across sources.")
        doc_b = document_from_text("shared.txt", "Same content across sources.")
        merged = merge_records({"VA.gov": [doc_a], "Fetch Sandbox": [doc_b]})
        self.assertEqual(len(merged.documents), 1)
        self.assertEqual(merged.sources_merged, 2)

    def test_ignores_empty_sources(self):
        merged = merge_records({"Upload": [], "VA.gov": []})
        self.assertEqual(merged.sources_merged, 0)
        self.assertEqual(merged.documents, [])


if __name__ == "__main__":
    unittest.main()
