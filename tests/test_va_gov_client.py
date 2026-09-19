"""Offline unit tests for the VA.gov client (mock-mode auth/fetch/merge)."""
import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app import va_gov_client  # noqa: E402
from app.documents import BLOCK, DocumentPage, ExtractedDocument, document_from_text  # noqa: E402
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
        self.assertEqual(len(merged.summary), 1)
        self.assertEqual(merged.summary[0].source, "VA.gov, Fetch Sandbox")
        self.assertEqual(merged.summary[0].filename, merged.documents[0].filename)

    def test_same_name_and_size_do_not_discard_distinct_records(self):
        first = document_from_text("records.txt", "Patient A: asthma.")
        second = document_from_text("records.txt", "Patient B: injury.")
        self.assertEqual(first.char_count, second.char_count)
        originals = deepcopy([first, second])
        merged = merge_records({"Upload": [first], "VA.gov": [second]})
        self.assertEqual([doc.full_text for doc in merged.documents], [first.full_text, second.full_text])
        self.assertEqual([row.source for row in merged.summary], ["Upload", "VA.gov"])
        self.assertEqual([row.filename for row in merged.summary], [doc.filename for doc in merged.documents])
        self.assertEqual(len({p.label for d in merged.documents for p in d.pages}), 2)
        self.assertEqual([first, second], originals)

    def test_same_source_can_contain_distinct_same_named_records(self):
        docs = [document_from_text("records.txt", f"Patient {letter}: asthma.") for letter in "ABC"]
        merged = merge_records({"Upload": docs})
        self.assertEqual([doc.full_text for doc in merged.documents], [doc.full_text for doc in docs])
        self.assertEqual(len({doc.filename for doc in merged.documents}), 3)
        self.assertEqual(len(merged.summary), 3)

    def test_alias_does_not_collide_with_an_existing_filename(self):
        docs = [
            document_from_text("records.txt", "Patient A: asthma."),
            document_from_text("records.txt", "Patient B: injury."),
            document_from_text("Upload/records.txt", "Patient C: asthma."),
        ]
        merged = merge_records({"Upload": docs})
        self.assertEqual(len({doc.filename for doc in merged.documents}), 3)
        self.assertEqual(merged.documents[2].filename, "Upload/records.txt")
        self.assertEqual(merged.documents[1].pages[0].filename, merged.documents[1].filename)

    def test_duplicate_after_renaming_retains_all_sources_once(self):
        first = document_from_text("records.txt", "Patient A: asthma.")
        second = document_from_text("records.txt", "Patient B: injury.")
        merged = merge_records({
            "Upload": [first], "VA.gov": [second, deepcopy(second)], "Fetch Sandbox": [deepcopy(second)],
        })
        self.assertEqual(len(merged.documents), 2)
        self.assertEqual(len(merged.summary), 2)
        self.assertEqual(merged.summary[1].source, "VA.gov, Fetch Sandbox")
        self.assertEqual(merged.summary[1].filename, merged.documents[1].filename)
        self.assertEqual(merged.sources_merged, 3)

    def test_document_identity_includes_page_boundaries_order_and_metadata(self):
        original = ExtractedDocument("record.pdf", [
            DocumentPage("record.pdf", 1, "a" * 50),
            DocumentPage("record.pdf", 2, "b" * 50),
        ], total_pages=3, unreadable_pages=[3])
        for change in ("boundaries", "order", "page_number", "page_kind", "page_filename",
                       "total_pages", "unreadable_pages", "pagination"):
            with self.subTest(change=change):
                modified = deepcopy(original)
                if change == "boundaries":
                    modified.pages[0].text = "a" * 49
                    modified.pages[1].text = "a" + "b" * 50
                elif change == "order":
                    modified.pages.reverse()
                elif change == "page_number":
                    modified.pages[0].page = 7
                elif change == "page_kind":
                    modified.pages[0].kind = BLOCK
                elif change == "page_filename":
                    modified.pages[0].filename = "other.pdf"
                elif change == "total_pages":
                    modified.total_pages = 4
                elif change == "unreadable_pages":
                    modified.unreadable_pages = [4]
                else:
                    modified.pagination = BLOCK
                merged = merge_records({"Upload": [original], "VA.gov": [modified]})
                self.assertEqual(len(merged.documents), 2)
                self.assertEqual(merged.documents[1].total_pages, modified.total_pages)
                self.assertEqual(merged.documents[1].unreadable_pages, modified.unreadable_pages)
                self.assertEqual(merged.documents[1].pagination, modified.pagination)

    def test_different_filenames_keep_their_provenance_even_with_identical_text(self):
        docs = [document_from_text(name, "Same clinical evidence.") for name in ("a.txt", "b.txt")]
        merged = merge_records({"Upload": [docs[0]], "VA.gov": [docs[1]]})
        self.assertEqual(merged.documents, docs)
        self.assertEqual(len(merged.summary), 2)

    def test_merged_record_citations_point_to_each_distinct_document(self):
        from app.medical_review import MedicalFact, verify_citations

        docs = [document_from_text("records.txt", text) for text in (
            "Patient A reports chronic asthma symptoms during exercise",
            "Patient B reports chronic injury symptoms during exercise",
        )]
        merged = merge_records({"Upload": [docs[0]], "VA.gov": [docs[1]]})
        self.assertEqual(len(merged.documents), 2)
        facts = [MedicalFact(
            "2025", "diagnosis", doc.full_text, doc.pages[0].label,
            quote=doc.full_text, document=doc.filename, page=doc.pages[0].page,
        ) for doc in merged.documents]
        checked = verify_citations(facts, merged.documents)
        self.assertEqual(checked["checked"], 2)
        self.assertEqual(checked["missing"], 0)

    def test_ignores_empty_sources(self):
        merged = merge_records({"Upload": [], "VA.gov": []})
        self.assertEqual(merged.sources_merged, 0)
        self.assertEqual(merged.documents, [])


if __name__ == "__main__":
    unittest.main()
