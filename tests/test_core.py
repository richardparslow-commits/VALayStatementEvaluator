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
    extract_uploaded_documents,
    paragraph_index,
    records_from_local_path,
)
from app.llm import LLMClient, _parse_json  # noqa: E402
from app import config  # noqa: E402
from app import watchdog  # noqa: E402
from app.usage import UsageTracker, estimate_tokens  # noqa: E402
from app.llm import _usage_tokens  # noqa: E402
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

    def _make_docx_bytes(self, text: str) -> bytes:
        xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p>"
            "</w:body></w:document>"
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("word/document.xml", xml)
        return buffer.getvalue()

    def test_loads_single_file(self):
        tmp = self._make_tmp({"note.txt": b"Knee pain noted during visit."})
        docs, skipped = records_from_local_path(str(Path(tmp) / "note.txt"))
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].filename, "note.txt")
        self.assertIn("Knee pain", docs[0].full_text)
        self.assertEqual(skipped, [])

    def test_loads_docx_with_filename_preserved(self):
        docx = self._make_docx_bytes("Treated with sertraline.")
        tmp = self._make_tmp({"records/medication.docx": docx})
        docs, skipped = records_from_local_path(tmp)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].filename, "records/medication.docx")
        self.assertIn("sertraline", docs[0].full_text)
        self.assertEqual(skipped, [])

    def test_loads_folder_recursively_and_sorts(self):
        tmp = self._make_tmp(
            {
                "b/note.txt": b"Second record.",
                "a/note.md": b"First record.",
                "ignore.csv": b"not supported",
                "nested/deep/note.txt": b"Third record.",
            }
        )
        docs, _ = records_from_local_path(tmp)
        self.assertEqual(
            [d.filename for d in docs], ["a/note.md", "b/note.txt", "nested/deep/note.txt"]
        )
        self.assertIn("Third record", docs[-1].full_text)
        self.assertTrue(all("not supported" not in d.full_text for d in docs))

    def test_same_named_files_in_different_folders_do_not_collide(self):
        tmp = self._make_tmp(
            {
                "2023/note.txt": b"First year.",
                "2024/note.txt": b"Second year.",
            }
        )
        docs, _ = records_from_local_path(tmp)
        self.assertEqual(
            [d.filename for d in docs], ["2023/note.txt", "2024/note.txt"]
        )
        self.assertEqual(
            {d.full_text for d in docs}, {"First year.", "Second year."}
        )

    def test_unreadable_files_are_reported_not_fatal(self):
        # A scanned/image-only PDF fails extraction but must not sink the load.
        tmp = self._make_tmp(
            {
                "good.txt": b"Knee pain noted.",
                "scanned.pdf": b"%PDF-1.4 completely unreadable scanned image data",
            }
        )
        docs, skipped = records_from_local_path(tmp)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].filename, "good.txt")
        self.assertEqual(len(skipped), 1)
        self.assertIn("scanned.pdf", skipped[0])

    def test_all_files_unreadable_raises(self):
        tmp = self._make_tmp(
            {"scanned.pdf": b"%PDF-1.4 unreadable scanned image data"}
        )
        with self.assertRaises(ExtractionError):
            records_from_local_path(tmp)

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
            docs, _ = records_from_local_path("~/note.txt")
        finally:
            if old_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old_home
        self.assertEqual(len(docs), 1)


class TestUploadedExtraction(unittest.TestCase):
    class _FakeUploaded:
        """Minimal stand-in for a Streamlit UploadedFile."""

        def __init__(self, name: str, data: bytes):
            self.name = name
            self.size = len(data)
            self._data = data

        def getvalue(self) -> bytes:
            return self._data

    def test_good_and_bad_files_split_into_docs_and_skipped(self):
        files = [
            self._FakeUploaded("good.txt", b"Knee pain noted."),
            self._FakeUploaded("bad.pdf", b"%PDF-1.4 broken scan data"),
            self._FakeUploaded("other.txt", b"Tinnitus reported."),
        ]
        docs, skipped = extract_uploaded_documents(files)
        self.assertEqual([d.filename for d in docs], ["good.txt", "other.txt"])
        self.assertEqual(len(skipped), 1)
        self.assertIn("bad.pdf", skipped[0])

    def test_all_bad_returns_no_docs_and_all_skipped(self):
        files = [
            self._FakeUploaded("a.pdf", b"%PDF broken"),
            self._FakeUploaded("b.pdf", b"%PDF broken"),
        ]
        docs, skipped = extract_uploaded_documents(files)
        self.assertEqual(docs, [])
        self.assertEqual(len(skipped), 2)

    def test_empty_input(self):
        docs, skipped = extract_uploaded_documents([])
        self.assertEqual(docs, [])
        self.assertEqual(skipped, [])


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


class TestUsageWatchdog(unittest.TestCase):
    def test_persistence_round_trip(self):
        import tempfile

        tmp = tempfile.mkdtemp()
        path = str(Path(tmp) / "hist.json")
        history = watchdog.UsageHistory()
        watchdog.record_run(history, prompt_tokens=100, completion_tokens=50, calls=3)
        watchdog.record_calibration(history, credits=120.5)
        watchdog.save_history(history, path)

        loaded = watchdog.load_history(path)
        self.assertEqual(len(loaded.runs), 1)
        self.assertEqual(loaded.runs[0].prompt_tokens, 100)
        self.assertEqual(loaded.runs[0].completion_tokens, 50)
        self.assertEqual(len(loaded.calibrations), 1)
        self.assertAlmostEqual(loaded.calibrations[0].credits, 120.5)

    def test_cumulative_tokens(self):
        history = watchdog.UsageHistory()
        watchdog.record_run(history, prompt_tokens=100, completion_tokens=50, calls=1)
        watchdog.record_run(history, prompt_tokens=200, completion_tokens=100, calls=1)
        totals = history.cumulative_tokens()
        self.assertEqual(totals["prompt"], 300)
        self.assertEqual(totals["completion"], 150)
        # up to run 0 only
        totals0 = history.cumulative_tokens(0)
        self.assertEqual(totals0["prompt"], 100)

    def test_fit_single_interval_blended_rate(self):
        history = watchdog.UsageHistory()
        # run1: 150 tokens total, baseline reading taken right after it.
        watchdog.record_run(history, prompt_tokens=100, completion_tokens=50, calls=2)
        watchdog.record_calibration(history, credits=0.0, ts=1.0)
        # run2: another 350 tokens (500 total now)
        watchdog.record_run(history, prompt_tokens=200, completion_tokens=150, calls=3)
        watchdog.record_calibration(history, credits=0.6, ts=2.0)
        fit = watchdog.fit_effective_rate(history)
        self.assertTrue(fit.any_rate())
        # Interval spans run 1 only = 350 new tokens => 0.6 credits.
        self.assertAlmostEqual(fit.blended_rate, 0.6 / 350 * 1e6, delta=1e-6)
        self.assertEqual(fit.intervals, 1)

    def test_fit_token_weighted_across_intervals(self):
        history = watchdog.UsageHistory()
        # run0 baseline.
        watchdog.record_run(history, prompt_tokens=1000, completion_tokens=0, calls=1)
        watchdog.record_calibration(history, credits=0.0, ts=1.0)
        # run1: 1000 more tokens -> +1 credit (1000/1M)
        watchdog.record_run(history, prompt_tokens=1000, completion_tokens=0, calls=1)
        watchdog.record_calibration(history, credits=1.0, ts=2.0)
        # run2: 3000 more tokens -> +6 credits (2000/1M)
        watchdog.record_run(history, prompt_tokens=3000, completion_tokens=0, calls=1)
        watchdog.record_calibration(history, credits=7.0, ts=3.0)
        fit = watchdog.fit_effective_rate(history)
        expected = (1000 / 4000) * 1000 + (3000 / 4000) * 2000  # = 1750
        self.assertAlmostEqual(fit.blended_rate, expected, delta=1e-6)
        self.assertEqual(fit.intervals, 2)
        self.assertEqual(fit.observed_credits, 7.0)

    def test_no_fit_without_calibrations(self):
        history = watchdog.UsageHistory()
        watchdog.record_run(history, prompt_tokens=100, completion_tokens=50, calls=1)
        fit = watchdog.fit_effective_rate(history)
        self.assertFalse(fit.any_rate())
        self.assertEqual(fit.intervals, 0)

    def test_no_fit_with_single_calibration(self):
        history = watchdog.UsageHistory()
        watchdog.record_run(history, prompt_tokens=100, completion_tokens=0, calls=1)
        watchdog.record_calibration(history, credits=0.5, ts=1.0)
        self.assertFalse(watchdog.fit_effective_rate(history).any_rate())

    def test_persistence_round_trip_with_by_role(self):
        import tempfile

        tmp = tempfile.mkdtemp()
        path = str(Path(tmp) / "hist.json")
        history = watchdog.UsageHistory()
        watchdog.record_run(
            history,
            prompt_tokens=1000,
            completion_tokens=2000,
            calls=5,
            by_role={"main": 1200, "fast": 1800},
        )
        watchdog.save_history(history, path)
        loaded = watchdog.load_history(path)
        self.assertEqual(loaded.runs[0].by_role, {"main": 1200, "fast": 1800})
        # Legacy records without by_role still load.
        watchdog.record_run(history, prompt_tokens=10, completion_tokens=5, calls=1)
        watchdog.save_history(history, path)
        loaded2 = watchdog.load_history(path)
        self.assertEqual(loaded2.runs[1].by_role, {})

    def test_per_model_least_squares_separates_rates(self):
        history = watchdog.UsageHistory()
        # Baseline run + reading at rate 0.
        watchdog.record_run(history, prompt_tokens=100, completion_tokens=0, calls=1)
        watchdog.record_calibration(history, credits=0.0, ts=1.0)
        # Interval 1: only the fast model runs (2M tokens -> 100 credits).
        watchdog.record_run(
            history, prompt_tokens=2_000_000, completion_tokens=0, calls=1,
            by_role={"fast": 2_000_000},
        )
        watchdog.record_calibration(history, credits=100.0, ts=2.0)
        # Interval 2: only the main model runs (2M tokens -> 1600 credits).
        watchdog.record_run(
            history, prompt_tokens=2_000_000, completion_tokens=0, calls=1,
            by_role={"main": 2_000_000},
        )
        watchdog.record_calibration(history, credits=1700.0, ts=3.0)

        fit = watchdog.fit_effective_rate(history)
        self.assertIsNotNone(fit.main_rate)
        self.assertIsNotNone(fit.fast_rate)
        # fast = 100 credits / 2M tokens * 1e6 = 50; main = 1600/2M*1e6 = 800.
        self.assertAlmostEqual(fit.fast_rate, 50.0, delta=1e-6)
        self.assertAlmostEqual(fit.main_rate, 800.0, delta=1e-6)
        self.assertNotEqual(fit.main_rate, fit.fast_rate)
        # Blended is still reported as an overall figure.
        self.assertAlmostEqual(fit.blended_rate, 1700.0 / 4_000_000 * 1e6, delta=1e-6)

    def test_per_model_falls_back_to_blended_when_single_model(self):
        """All-main data cannot separate rates; both slots use the blended rate."""
        history = watchdog.UsageHistory()
        watchdog.record_run(history, prompt_tokens=1000, completion_tokens=0, calls=1)
        watchdog.record_calibration(history, credits=0.0, ts=1.0)
        watchdog.record_run(
            history, prompt_tokens=1_000_000, completion_tokens=0, calls=1,
            by_role={"main": 1_000_000},
        )
        watchdog.record_calibration(history, credits=500.0, ts=2.0)
        fit = watchdog.fit_effective_rate(history)
        self.assertEqual(fit.main_rate, fit.fast_rate)
        self.assertAlmostEqual(fit.main_rate, 500.0, delta=1e-6)

    def test_per_role_tokens_classifies_by_phase(self):
        from app.usage import UsageTracker

        tracker = UsageTracker()
        tracker.record(model="m", phase="records:digest", system="s", user="u",
                       content="o", prompt_tokens=100, completion_tokens=50)
        tracker.record(model="m", phase="claims", system="s", user="u",
                       content="o", prompt_tokens=10, completion_tokens=20)
        roles = tracker.per_role_tokens()
        self.assertEqual(roles["fast"], 150)
        self.assertEqual(roles["main"], 30)

    def test_negative_credit_delta_skipped(self):
        history = watchdog.UsageHistory()
        watchdog.record_run(history, prompt_tokens=1000, completion_tokens=0, calls=1)
        watchdog.record_calibration(history, credits=10.0, ts=1.0)
        watchdog.record_run(history, prompt_tokens=1000, completion_tokens=0, calls=1)
        # console shows fewer credits later (reset) => invalid interval, skipped
        watchdog.record_calibration(history, credits=5.0, ts=2.0)
        fit = watchdog.fit_effective_rate(history)
        self.assertFalse(fit.any_rate())

    def test_duplicate_readings_same_run_ignored(self):
        history = watchdog.UsageHistory()
        watchdog.record_run(history, prompt_tokens=1000, completion_tokens=0, calls=1)
        watchdog.record_calibration(history, credits=1.0, ts=1.0)
        # second reading with no new runs between -> no interval
        watchdog.record_calibration(history, credits=2.0, ts=1.5)
        self.assertFalse(watchdog.fit_effective_rate(history).any_rate())


class TestUsageTracker(unittest.TestCase):
    def test_estimate_tokens_never_zero_for_nonempty(self):
        self.assertGreater(estimate_tokens("x"), 0)
        self.assertEqual(estimate_tokens(""), 0)

    def test_aggregates_by_phase(self):
        tracker = UsageTracker()
        tracker.record(model="m", phase="digest", system="s", user="u",
                       content="o", prompt_tokens=10, completion_tokens=20)
        tracker.record(model="m", phase="digest", system="s", user="u",
                       content="o", prompt_tokens=15, completion_tokens=25)
        tracker.record(model="m", phase="verify", system="s", user="u",
                       content="o", prompt_tokens=8, completion_tokens=2)
        per = tracker.per_phase()
        self.assertEqual(per["digest"].calls, 2)
        self.assertEqual(per["digest"].prompt_tokens, 25)
        self.assertEqual(per["digest"].completion_tokens, 45)
        self.assertEqual(per["verify"].calls, 1)
        self.assertEqual(tracker.totals().calls, 3)
        self.assertEqual(tracker.totals().total_tokens, 80)

    def test_falls_back_to_character_estimate_without_usage(self):
        tracker = UsageTracker()
        tracker.record(model="m", phase="p", system="a" * 40, user="b" * 40,
                       content="c" * 40, prompt_tokens=None, completion_tokens=None)
        # 80 prompt chars /4 = 20; 40 completion chars /4 = 10.
        self.assertEqual(tracker.totals().prompt_tokens, 20)
        self.assertEqual(tracker.totals().completion_tokens, 10)

    def test_phase_counts_models(self):
        tracker = UsageTracker()
        tracker.record(model="fast", phase="digest", system="s", user="u",
                       content="o", prompt_tokens=1, completion_tokens=1)
        tracker.record(model="main", phase="digest", system="s", user="u",
                       content="o", prompt_tokens=1, completion_tokens=1)
        stats = tracker.per_phase()["digest"]
        self.assertEqual(stats.models["fast"], 1)
        self.assertEqual(stats.models["main"], 1)

    def test_credit_estimate_none_without_rates(self):
        tracker = UsageTracker()
        tracker.record(model="qwen3.7-max", phase="p", system="s", user="u",
                       content="o", prompt_tokens=1_000_000)
        self.assertIsNone(tracker.credit_estimate({}))
        # Unrated model still has no known rate => None.
        self.assertIsNone(
            tracker.credit_estimate({"qwen3.7-flash": 200.0})
        )

    def test_credit_estimate_converts_when_rate_known(self):
        tracker = UsageTracker()
        # 2M total tokens on the main model at 800 credits/1M => 1600 credits.
        tracker.record(model="qwen3.7-max", phase="p", system="s", user="u",
                       content="o", prompt_tokens=1_500_000, completion_tokens=500_000)
        credits = tracker.credit_estimate({"qwen3.7-max": 800.0, "qwen3.7-flash": 200.0})
        self.assertAlmostEqual(credits, 1600.0)

    def test_live_line_empty_when_no_calls(self):
        self.assertEqual(UsageTracker().live_line(), "")

    def test_live_line_counts_calls(self):
        tracker = UsageTracker()
        tracker.record(model="m", phase="p", system="s", user="u", content="o",
                       prompt_tokens=1000, completion_tokens=500)
        line = tracker.live_line()
        self.assertIn("1 call", line)
        self.assertIn("1,000 in", line)
        self.assertIn("500 out", line)

    def test_usage_tokens_parses_provider_metadata(self):
        class _Usage:
            prompt_tokens = 11
            completion_tokens = 22

        class _Resp:
            usage = _Usage()

        self.assertEqual(_usage_tokens(_Resp()), (11, 22))

    def test_usage_tokens_none_without_metadata(self):
        class _Resp:
            usage = None

        self.assertEqual(_usage_tokens(_Resp()), (None, None))

    def test_llm_client_constructs_usage_tracker(self):
        class _S:
            configured = True
            api_key = "k"
            base_url = "http://example.invalid"
            model_main = "m"

        client = LLMClient(_S())
        self.assertIsInstance(client.usage, UsageTracker)


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
