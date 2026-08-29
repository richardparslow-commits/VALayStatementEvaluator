"""Offline unit tests: extraction, chunking, JSON parsing. No network needed.

Run from project root: .venv/bin/python -m unittest discover -s tests -v
"""
import io
import sys
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.documents import (  # noqa: E402
    ExtractionError,
    chunk_page_labelled_text,
    extract_document,
    paragraph_index,
    records_from_local_path,
)
from app.llm import _parse_json  # noqa: E402
from app import config  # noqa: E402
from app.medical_review import (  # noqa: E402
    MedicalDigest,
    MedicalFact,
    _dedupe_facts,
    _merge_facts,
    review_medical_records,
)


class TestExtraction(unittest.TestCase):
    def test_txt_extraction(self):
        doc = extract_document("note.txt", b"Hello world. This is a note.")
        self.assertEqual(doc.filename, "note.txt")
        self.assertEqual(len(doc.pages), 1)
        self.assertIn("Hello world", doc.full_text)

    def test_empty_txt_raises(self):
        with self.assertRaises(ExtractionError):
            extract_document("empty.txt", b"   \n  ")

    def test_unsupported_type_raises(self):
        with self.assertRaises(ExtractionError):
            extract_document("file.xls", b"data")

    def test_docx_extraction(self):
        xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>Observed pain during lifting.</w:t></w:r></w:p>"
            "</w:body></w:document>"
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("word/document.xml", xml)
        doc = extract_document("record.docx", buffer.getvalue())
        self.assertIn("Observed pain", doc.full_text)

    def test_page_labelled_text(self):
        doc = extract_document("note.txt", b"Body text here.")
        self.assertIn("page 1", doc.page_labelled_text())


class TestLocalPathLoading(unittest.TestCase):
    def _make_tmp(self, files: dict[str, bytes]):
        import tempfile

        tmp = tempfile.mkdtemp()
        for name, data in files.items():
            p = Path(tmp) / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        return tmp

    def test_loads_single_file(self):
        tmp = self._make_tmp({"note.txt": b"Knee pain noted during visit."})
        docs = records_from_local_path(str(Path(tmp) / "note.txt"))
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].filename, "note.txt")
        self.assertIn("Knee pain", docs[0].full_text)

    def test_loads_folder_recursively_and_sorts(self):
        tmp = self._make_tmp(
            {
                "b/note.txt": b"Second record.",
                "a/note.md": b"First record.",
                "ignore.csv": b"not supported",
                "nested/deep/note.txt": b"Third record.",
            }
        )
        docs = records_from_local_path(tmp)
        self.assertEqual([d.filename for d in docs], ["note.md", "note.txt", "note.txt"])
        self.assertIn("Third record", docs[-1].full_text)
        self.assertTrue(all("not supported" not in d.full_text for d in docs))

    def test_missing_path_raises(self):
        with self.assertRaises(ExtractionError):
            records_from_local_path("/nonexistent/records/folder")

    def test_empty_folder_raises(self):
        tmp = self._make_tmp({})
        with self.assertRaises(ExtractionError):
            records_from_local_path(tmp)

    def test_no_supported_files_raises(self):
        tmp = self._make_tmp({"data.csv": b"a,b\n1,2\n"})
        with self.assertRaises(ExtractionError):
            records_from_local_path(tmp)

    def test_expands_user_home(self):
        import os

        tmp = self._make_tmp({"note.txt": b"Home record."})
        old_home = os.environ.get("HOME")
        try:
            os.environ["HOME"] = tmp
            docs = records_from_local_path("~/note.txt")
        finally:
            if old_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old_home
        self.assertEqual(len(docs), 1)


class TestChunking(unittest.TestCase):
    def test_small_text_single_chunk(self):
        chunks = chunk_page_labelled_text("short text")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].label, "chunk 1/1")

    def test_large_text_multiple_chunks(self):
        text = "\n\n".join(
            f"Paragraph number {i} contains medical information." for i in range(800)
        )
        chunks = chunk_page_labelled_text(text, max_chars=3000)
        self.assertGreater(len(chunks), 2)
        self.assertEqual(chunks[0].index, 1)
        self.assertEqual(chunks[-1].total, len(chunks))

    def test_content_preserved(self):
        text = "abcdefghij " * 2000
        chunks = chunk_page_labelled_text(text, max_chars=5000)
        union = "".join(c.text for c in chunks)
        for token in text.split()[:100]:
            self.assertIn(token, union)


class TestJsonParsing(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(_parse_json('{"a": 1}'), {"a": 1})

    def test_fenced_json(self):
        self.assertEqual(_parse_json('```json\n{"a": 2}\n```'), {"a": 2})

    def test_json_inside_prose(self):
        self.assertEqual(_parse_json('Sure! Here it is: {"a": 3} hope that helps'), {"a": 3})

    def test_list_json(self):
        self.assertEqual(_parse_json("result: [1, 2]"), [1, 2])

    def test_bad_json_raises(self):
        from app.llm import LLMError

        with self.assertRaises(LLMError):
            _parse_json("no json at all")


# ---------------------------------------------------------------------------
# Large-record pipeline tests (parallel review, dedup, hierarchical merge)
# using a deterministic fake LLM — no network required.
# ---------------------------------------------------------------------------
class _FakeSettings:
    model_fast = "fake-fast"
    model_main = "fake-main"


class FakeLLM:
    """Deterministic stand-in for LLMClient."""

    def __init__(self, fail_digest_once: bool = False) -> None:
        self._settings = _FakeSettings()
        self.digest_calls = 0
        self.merge_calls = 0
        self.chat_calls = 0
        self._failed = False
        self._fail_digest_once = fail_digest_once

    def chat_json(self, system, user, **kwargs):
        if "CHUNK TEXT" in user:  # digest prompt
            self.digest_calls += 1
            if self._fail_digest_once and not self._failed:
                self._failed = True
                from app.llm import LLMError

                raise LLMError("simulated transient failure")
            facts = []
            for line in user.splitlines():
                line = line.strip()
                if line.startswith("EVT "):
                    facts.append(
                        {
                            "date": "2020-01",
                            "type": "symptom",
                            "description": line,
                            "source": "",
                            "quote": line,
                        }
                    )
            return {
                "facts": facts,
                "conditions_mentioned": ["knee pain"],
                "providers_and_facilities": ["Dr. Smith (ortho)"],
                "notes": "",
            }
        # merge prompt: pass facts through unchanged (payload follows the intro)
        self.merge_calls += 1
        import json as _json

        return {"facts": _json.loads(user.split("\n\n", 1)[1])}

    def chat(self, system, user, **kwargs):
        self.chat_calls += 1
        return "Summary of records."


class TestLargeRecordPipeline(unittest.TestCase):
    def test_pipeline_order_page_dedup_and_stats(self):
        docs = [
            extract_document(
                "a.txt", b"EVT one knee pain noted.\n\nEVT two brace prescribed."
            ),
            # Same content as a.txt — must be skipped as a duplicate page.
            extract_document(
                "b.txt", b"EVT one knee pain noted.\n\nEVT two brace prescribed."
            ),
            extract_document("c.txt", b"EVT three tinnitus reported."),
        ]
        llm = FakeLLM()
        digest = review_medical_records(llm, docs)
        self.assertEqual(digest.pages_reviewed, 3)
        self.assertEqual(digest.duplicates_skipped, 1)
        self.assertEqual(digest.summary, "Summary of records.")
        self.assertEqual(
            [f.description for f in digest.facts],
            [
                "EVT one knee pain noted.",
                "EVT two brace prescribed.",
                "EVT three tinnitus reported.",
            ],
        )
        self.assertIn("knee pain", digest.conditions)
        self.assertIn("Dr. Smith (ortho)", digest.providers)

    def test_pipeline_retries_transient_chunk_failure(self):
        llm = FakeLLM(fail_digest_once=True)
        docs = [extract_document("a.txt", b"EVT one event recorded here.")]
        digest = review_medical_records(llm, docs)
        self.assertEqual(len(digest.facts), 1)
        self.assertEqual(llm.digest_calls, 2)  # failed once, succeeded on retry

    def test_page_cap_enforced(self):
        original = config.MAX_RECORD_PAGES
        config.MAX_RECORD_PAGES = 1
        try:
            docs = [
                extract_document("a.txt", b"EVT one"),
                extract_document("b.txt", b"EVT two"),
            ]
            with self.assertRaises(ValueError):
                review_medical_records(FakeLLM(), docs)
        finally:
            config.MAX_RECORD_PAGES = original


    def test_hierarchical_merge_batches_large_fact_lists(self):
        facts = [
            MedicalFact(
                date=f"2020-{(i % 12) + 1:02d}",
                type="symptom",
                description=f"fact number {i}",
                source="chunk 1/9",
            )
            for i in range(600)
        ]
        digest = MedicalDigest(facts=facts)
        llm = FakeLLM()
        merged = _merge_facts(llm, digest)
        # 600 facts -> at least 3 parallel merge batches of 200.
        self.assertGreaterEqual(llm.merge_calls, 3)
        # Pass-through fake must not lose any distinct facts.
        self.assertEqual(len(merged), 600)
        self.assertLessEqual(len(merged), config.MAX_DIGEST_FACTS)

    def test_dedupe_facts_mechanical(self):
        facts = [
            MedicalFact("2020-01", "symptom", "Knee pain.", "a"),
            MedicalFact("2020-01", "symptom", "knee   pain.", "b"),
            MedicalFact("2021-02", "symptom", "Knee pain.", "c"),
        ]
        unique = _dedupe_facts(facts)
        self.assertEqual(len(unique), 2)
        self.assertEqual(unique[0].source, "a")
        self.assertEqual(unique[1].date, "2021-02")

    def test_relevant_facts_text_ranks_matches_first(self):
        digest = MedicalDigest(
            facts=[
                MedicalFact("2019-03", "diagnosis", "tinnitus diagnosed", "chunk 1/2"),
                MedicalFact(
                    "2020-05",
                    "symptom",
                    "knee pain worsened while walking",
                    "chunk 2/2",
                    "reported knee pain",
                ),
            ]
        )
        text = digest.relevant_facts_text("veteran reports knee pain when walking")
        self.assertIn("2 of 2", text)
        self.assertLess(
            text.index("knee pain worsened"), text.index("tinnitus diagnosed")
        )

    def test_condensed_timeline_covers_full_range(self):
        facts = [
            MedicalFact(str(i), "other", f"fact-{i}", "src") for i in range(1000)
        ]
        digest = MedicalDigest(facts=facts)
        lines = digest.condensed_timeline(max_entries=100).splitlines()
        self.assertEqual(len(lines), 100)
        self.assertTrue(lines[0].startswith("[0]"))
        self.assertIn("fact-999", lines[-1])

    def test_paragraph_index_is_cached(self):
        doc = extract_document(
            "note.txt",
            b"First paragraph about knee pain and swelling.\n\n"
            b"Second paragraph about medication refills.",
        )
        first = paragraph_index(doc)
        second = paragraph_index(doc)
        self.assertIs(first, second)
        self.assertEqual(len(first), 2)


if __name__ == "__main__":
    unittest.main()
