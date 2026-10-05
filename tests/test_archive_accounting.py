"""Synthetic functional regression coverage for complete archive member accounting."""
from __future__ import annotations

import hashlib
import io
import zipfile
import unittest
from unittest.mock import patch

from tests import hermetic  # noqa: F401
from tests.ingestion_fixtures import docx_parts, package
from tests.test_extractors import _pdf_bytes
from app import config, documents, pilot
from app.documents import ExtractionError, InProcessExtractor
from app.views import uploads

VISIBLE = b"Synthetic initial report: discomfort reported."
CORRECTION = b"Synthetic correction: symptoms were denied."


class Uploaded:
    def __init__(self, data: bytes) -> None:
        self.name = "records.zip"
        self.size = len(data)
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


class ArchiveAccountingTests(unittest.TestCase):
    def test_supported_hidden_corrections_preserve_text_hash_and_source_address(self):
        for name in (".correction.txt", "notes/.correction.txt",
                     "__MACOSX/correction.txt", "notes/__MACOSX/.correction.MD"):
            with self.subTest(name=name):
                docs, skipped = InProcessExtractor().extract("records.zip", package({
                    "record.txt": VISIBLE, name: CORRECTION,
                }))
                self.assertEqual(skipped, [])
                self.assertEqual([doc.filename for doc in docs], ["records/record.txt", "records/" + name])
                self.assertEqual(docs[1].full_text, CORRECTION.decode())
                self.assertEqual(docs[1].source_sha256, hashlib.sha256(CORRECTION).hexdigest())
                self.assertIn("records/" + name, docs[1].page_labelled_text())

    def test_hidden_records_use_normal_format_validation_and_parsers(self):
        bodies = {
            ".note.txt": CORRECTION,
            ".note.md": CORRECTION,
            "__MACOSX/.note.pdf": _pdf_bytes([CORRECTION.decode()]),
            "__MACOSX/.note.docx": package(docx_parts(CORRECTION.decode())),
        }
        docs, skipped = InProcessExtractor().extract("records.zip", package(bodies))
        self.assertEqual(skipped, [])
        self.assertEqual([doc.filename for doc in docs], ["records/" + name for name in bodies])
        for doc in docs:
            self.assertIn(CORRECTION.decode(), doc.full_text)

    def test_archive_containing_only_a_hidden_record_is_readable(self):
        docs, skipped = InProcessExtractor().extract("records.zip", package({".correction.txt": CORRECTION}))
        self.assertEqual(skipped, [])
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].full_text, CORRECTION.decode())

    def test_every_unreadable_or_unsupported_hidden_member_has_a_named_refusal(self):
        rejected = {
            ".DS_Store": b"Unsupported synthetic bookkeeping",
            "__MACOSX/.DS_Store": b"Unsupported synthetic bookkeeping",
            ".nested.zip": package({"inner.txt": CORRECTION}),
            "__MACOSX/later.zip": package({"inner.txt": CORRECTION}),
            ".damaged.pdf": b"Synthetic text spoofing a PDF suffix",
            "__MACOSX/._notes.txt": b"\x00\x05\x16\x07Synthetic binary sidecar",
        }
        docs, skipped = InProcessExtractor().extract("records.zip", package({"record.txt": VISIBLE, **rejected}))
        self.assertEqual([doc.filename for doc in docs], ["records/record.txt"])
        self.assertEqual(len(skipped), len(rejected))
        for name in rejected:
            self.assertTrue(any(name in message for message in skipped), (name, skipped))

    def test_unsupported_only_hidden_archive_reports_its_member(self):
        docs, skipped = InProcessExtractor().extract("records.zip", package({".DS_Store": b"Unsupported metadata"}))
        self.assertEqual(docs, [])
        self.assertEqual(len(skipped), 1)
        self.assertIn(".DS_Store", skipped[0])

    def test_hidden_member_remains_subject_to_per_member_limit(self):
        with patch.object(config, "ZIP_MAX_MEMBER_BYTES", 20):
            docs, skipped = InProcessExtractor().extract("records.zip", package({
                "record.txt": b"Short note.", ".oversized.txt": b"A" * 21,
            }))
        self.assertEqual([doc.filename for doc in docs], ["records/record.txt"])
        self.assertEqual(len(skipped), 1)
        self.assertIn(".oversized.txt", skipped[0])

    def test_hidden_member_remains_subject_to_compression_ratio_limit(self):
        data = package({"record.txt": b"Short note.", "__MACOSX/.bomb.txt": b"A" * 10000}, zipfile.ZIP_DEFLATED)
        with patch.object(config, "ZIP_MAX_COMPRESSION_RATIO", 2):
            docs, skipped = InProcessExtractor().extract("records.zip", data)
        self.assertEqual([doc.filename for doc in docs], ["records/record.txt"])
        self.assertEqual(len(skipped), 1)
        self.assertIn(".bomb.txt", skipped[0])
        self.assertIn("compression ratio", skipped[0])

    def test_hidden_member_counts_toward_total_before_any_expansion(self):
        data = package({"record.txt": b"A" * 10, ".correction.txt": b"B" * 11})
        with patch.object(config, "ZIP_MAX_TOTAL_UNCOMPRESSED_BYTES", 20), \
                patch.object(zipfile.ZipFile, "open", side_effect=AssertionError("Unexpected archive read")) as opened, \
                self.assertRaises(ExtractionError):
            InProcessExtractor().extract("records.zip", data)
        opened.assert_not_called()

    def test_hidden_actual_expansion_refusal_identifies_its_member(self):
        data = package({"record.txt": b"Short note.", ".oversized.txt": b"A"})
        original_open = zipfile.ZipFile.open

        def open_member(archive, member, *args, **kwargs):
            if member.filename == ".oversized.txt":
                return io.BytesIO(b"A" * 21)
            return original_open(archive, member, *args, **kwargs)

        with patch.object(config, "ZIP_MAX_MEMBER_BYTES", 20), \
                patch.object(zipfile.ZipFile, "open", autospec=True, side_effect=open_member):
            docs, skipped = InProcessExtractor().extract("records.zip", data)
        self.assertEqual([doc.filename for doc in docs], ["records/record.txt"])
        self.assertEqual(len(skipped), 1)
        self.assertIn(".oversized.txt", skipped[0])
        self.assertIn("actual expansion", skipped[0])

    def test_pilot_refuses_incomplete_archive_on_first_read_and_cached_rerun(self):
        uploaded = Uploaded(package({"record.txt": VISIBLE, ".DS_Store": b"Unsupported metadata"}))
        state = {}
        with patch.object(documents, "_ACTIVE_EXTRACTOR", InProcessExtractor()), \
                patch.object(uploads.st, "session_state", state), \
                patch.object(pilot, "enabled", return_value=True), \
                patch.object(pilot, "display"), patch.object(uploads, "report_failure"), \
                patch("app.upload_admission.claim_documents"):
            for _ in range(2):
                self.assertEqual(uploads.extract_uploads([uploaded], "eval"), [])
        self.assertEqual(len(state), 1)
        self.assertEqual(len(next(iter(state.values()))["skipped"]), 1)

    def test_legacy_archive_cache_is_recomputed_and_pruned(self):
        uploaded = Uploaded(package({"record.txt": VISIBLE, ".correction.txt": CORRECTION}))
        legacy_key = f"eval:{uploaded.name}:{uploaded.size}:{hashlib.sha256(uploaded.getvalue()).hexdigest()}"
        legacy_doc = documents.extract_document("records/record.txt", VISIBLE)
        state = {legacy_key: {"documents": [legacy_doc], "skipped": []}}
        with patch.object(documents, "_ACTIVE_EXTRACTOR", InProcessExtractor()), \
                patch.object(uploads.st, "session_state", state), \
                patch.object(pilot, "enabled", return_value=True), \
                patch.object(pilot, "display"), patch.object(uploads, "report_failure"), \
                patch("app.upload_admission.claim_documents") as claimed:
            result = uploads.extract_uploads([uploaded], "eval")
        self.assertEqual([doc.full_text for doc in result], [VISIBLE.decode(), CORRECTION.decode()])
        self.assertNotIn(legacy_key, state)
        self.assertEqual(len(state), 1)
        claimed.assert_called_once()


if __name__ == "__main__":
    unittest.main()
