"""Offline unit tests: extraction, chunking, JSON parsing. No network needed.

Run from project root: .venv/bin/python -m unittest discover -s tests -v
"""
import io
import json
import os
import sys
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app.documents import (  # noqa: E402
    ExtractionError,
    _read_docx_member_limited,
    chunk_page_labelled_text,
    iter_page_labelled_chunks,
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
    _norm_key,
    review_medical_records,
)


class TestExtraction(unittest.TestCase):
    def _make_docx_bytes(
        self,
        text: str = "Observed pain during lifting.",
        extra_entries: dict[str, bytes] | None = None,
    ) -> bytes:
        xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p>"
            "</w:body></w:document>"
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("word/document.xml", xml)
            for name, payload in (extra_entries or {}).items():
                archive.writestr(name, payload)
        return buffer.getvalue()

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
        doc = extract_document("record.docx", self._make_docx_bytes())
        self.assertIn("Observed pain", doc.full_text)

    def test_docx_rejects_member_over_max_uncompressed_size(self):
        docx = self._make_docx_bytes(text="A" * 300)
        with patch.object(config, "DOCX_MAX_INTERNAL_FILE_BYTES", 200), patch.object(
            config, "DOCX_MAX_TOTAL_UNCOMPRESSED_BYTES", 10_000
        ):
            with self.assertRaises(ExtractionError) as exc:
                extract_document("oversized-member.docx", docx)
        self.assertIn("exceeds max uncompressed size", str(exc.exception))

    def test_docx_rejects_total_uncompressed_size_over_limit(self):
        docx = self._make_docx_bytes(extra_entries={"customXml/item1.xml": b"A" * 300})
        with patch.object(config, "DOCX_MAX_INTERNAL_FILE_BYTES", 500), patch.object(
            config, "DOCX_MAX_TOTAL_UNCOMPRESSED_BYTES", 400
        ):
            with self.assertRaises(ExtractionError) as exc:
                extract_document("oversized-total.docx", docx)
        self.assertIn("total uncompressed size exceeds limit", str(exc.exception))

    def test_docx_extracts_successfully_when_under_limits(self):
        docx = self._make_docx_bytes(extra_entries={"docProps/core.xml": b"<p>ok</p>"})
        with patch.object(config, "DOCX_MAX_INTERNAL_FILE_BYTES", 10_000), patch.object(
            config, "DOCX_MAX_TOTAL_UNCOMPRESSED_BYTES", 20_000
        ):
            doc = extract_document("under-limit.docx", docx)
        self.assertIn("Observed pain", doc.full_text)

    def test_docx_rejects_too_many_internal_files(self):
        extra = {f"customXml/item{i}.xml": b"x" for i in range(5)}
        docx = self._make_docx_bytes(extra_entries=extra)
        with patch.object(config, "DOCX_MAX_INTERNAL_FILE_COUNT", 3):
            with self.assertRaises(ExtractionError) as exc:
                extract_document("too-many-members.docx", docx)
        self.assertIn("too many internal files", str(exc.exception))

    def test_docx_member_runtime_overflow_guard(self):
        class _FakeStream:
            def __init__(self):
                self._chunks = [b"abc", b"def", b"ghi", b""]

            def read(self, _size: int) -> bytes:
                return self._chunks.pop(0)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class _FakeArchive:
            def getinfo(self, _name: str):
                return type("Info", (), {"file_size": 4})()

            def open(self, _info, _mode: str):
                return _FakeStream()

        with self.assertRaises(ExtractionError) as exc:
            _read_docx_member_limited(
                "runtime-overflow.docx",
                archive=_FakeArchive(),
                member_name="word/document.xml",
                max_member_bytes=5,
                max_total_bytes=50,
                existing_total_bytes=0,
            )
        self.assertIn("exceeded max uncompressed size while reading", str(exc.exception))

    def test_docx_total_runtime_overflow_guard(self):
        class _FakeStream:
            def __init__(self):
                self._chunks = [b"abc", b"def", b""]

            def read(self, _size: int) -> bytes:
                return self._chunks.pop(0)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class _FakeArchive:
            def getinfo(self, _name: str):
                return type("Info", (), {"file_size": 3})()

            def open(self, _info, _mode: str):
                return _FakeStream()

        with self.assertRaises(ExtractionError) as exc:
            _read_docx_member_limited(
                "runtime-total-overflow.docx",
                archive=_FakeArchive(),
                member_name="word/document.xml",
                max_member_bytes=100,
                max_total_bytes=5,
                existing_total_bytes=0,
            )
        self.assertIn("total uncompressed size exceeded while reading", str(exc.exception))

    def test_docx_total_runtime_overflow_guard_with_existing_members(self):
        class _FakeStream:
            def __init__(self):
                self._chunks = [b"abc", b"def", b""]

            def read(self, _size: int) -> bytes:
                return self._chunks.pop(0)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class _FakeArchive:
            def getinfo(self, _name: str):
                return type("Info", (), {"file_size": 3})()

            def open(self, _info, _mode: str):
                return _FakeStream()

        with self.assertRaises(ExtractionError) as exc:
            _read_docx_member_limited(
                "runtime-total-existing-overflow.docx",
                archive=_FakeArchive(),
                member_name="word/document.xml",
                max_member_bytes=100,
                max_total_bytes=10,
                existing_total_bytes=6,
            )
        self.assertIn("total uncompressed size exceeded while reading", str(exc.exception))

    def test_docx_member_runtime_guard_wins_with_existing_total(self):
        class _FakeStream:
            def __init__(self):
                self._chunks = [b"abc", b"def", b""]

            def read(self, _size: int) -> bytes:
                return self._chunks.pop(0)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class _FakeArchive:
            def getinfo(self, _name: str):
                return type("Info", (), {"file_size": 4})()

            def open(self, _info, _mode: str):
                return _FakeStream()

        with self.assertRaises(ExtractionError) as exc:
            _read_docx_member_limited(
                "runtime-member-existing-total.docx",
                archive=_FakeArchive(),
                member_name="word/document.xml",
                max_member_bytes=5,
                max_total_bytes=100,
                existing_total_bytes=40,
            )
        self.assertIn("member 'word/document.xml' exceeded max uncompressed size", str(exc.exception))

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


class TestStreamingChunking(unittest.TestCase):
    """The streaming chunker must produce byte-identical chunks to the joined
    implementation while never materializing the joined corpus."""

    @staticmethod
    def _docs(page_specs: list[tuple[str, int, str]]) -> list:
        from app.documents import DocumentPage, ExtractedDocument

        docs = {}
        for filename, page, text in page_specs:
            docs.setdefault(filename, []).append(DocumentPage(filename, page, text))
        return [ExtractedDocument(name, pages) for name, pages in docs.items()]

    def test_byte_identical_to_joined_implementation(self):
        """Same chunks (text, pages, numbering) as the joined path — property
        over randomized page sets with trap boundaries (long runs of blank
        lines, trailing spaces before paragraph breaks, '. ' sentence cuts).

        Page texts are passed through ``clean_text`` exactly as every real
        extractor does before handing pages over (PDF, TXT, DOCX paths all
        pre-clean), because the per-part-equals-joined equivalence the
        streaming cutter relies on is stated for pre-cleaned parts: each part
        then starts and ends with a non-whitespace character, so none of
        ``clean_text``'s three operations can act across a seam.
        """
        import random

        from app.documents import ExtractedDocument, clean_text

        rng = random.Random(20260921)
        for trial in range(40):
            n_pages = rng.randint(0, 25)
            specs = []
            for i in range(n_pages):
                if rng.random() < 0.1:
                    text = ""  # image-only page: skipped by both paths
                else:
                    n_paras = rng.randint(1, 6)
                    paras = []
                    for _ in range(n_paras):
                        n_sent = rng.randint(1, 8)
                        body = ". ".join(
                            f"Finding {rng.randint(0, 999)} noted by clinician {j}"
                            for j in range(n_sent)
                        )
                        body += "."
                        # Trap: trailing spaces before the paragraph break, and
                        # occasional long blank-line runs inside a page.
                        if rng.random() < 0.4:
                            body += "   "
                        if rng.random() < 0.15:
                            body += "\n\n\n\n"
                        paras.append(body)
                    text = clean_text("\n\n".join(paras))
                specs.append((f"rec{i // 8}.pdf", i % 8 + 1, text))
            docs = self._docs(specs)
            if not docs:
                docs = [ExtractedDocument("empty.pdf", [])]

            # The reference corpus is exactly what the pre-streaming caller
            # built: the flat "\n\n" join of each page's "marker\ntext" part,
            # composed here through the real page_labelled_text so the marker
            # format can never drift between this test and the cutter.
            joined = "\n\n".join(doc.page_labelled_text() for doc in docs)
            reference = chunk_page_labelled_text(joined)
            streamed = list(
                iter_page_labelled_chunks(
                    page for doc in docs for page in doc.pages
                )
            )
            self.assertEqual(
                len(reference),
                len(streamed),
                msg=f"trial {trial}: chunk count differs",
            )
            for k, (r, s) in enumerate(zip(reference, streamed)):
                self.assertEqual(r.text, s.text, msg=f"trial {trial} chunk {k}: text differs")
                self.assertEqual(
                    r.pages, s.pages, msg=f"trial {trial} chunk {k}: pages differ"
                )
                self.assertEqual(r.index, s.index)
                self.assertEqual(r.total, s.total)

    def test_never_materializes_the_joined_corpus(self):
        """Streaming peak stays at least one corpus copy below the joined
        path's peak, measured under the same tracemalloc window.

        Self-calibrating on purpose: an absolute bound would have to price in
        per-chunk metadata and tracemalloc's per-allocation overhead, both of
        which the joined path pays too. The joined path's traced delta over
        the streamer's is exactly the copies streaming eliminates — the
        ``clean_text`` copy of the corpus and the join itself — so the pinned
        property is: joined_peak - streamed_peak >= one corpus. Any regression
        that re-materializes the join adds a corpus copy to the streamer and
        collapses the margin.
        """
        import tracemalloc

        n_pages = 800
        specs = [
            (f"rec{i // 8}.pdf", i % 8 + 1, f"Finding {i} " + "x" * 180)
            for i in range(n_pages)
        ]
        docs = self._docs(specs)
        # The pre-built joined corpus is allocated before tracing starts, so
        # it counts against neither pass — both paths receive the same string.
        joined_corpus = "\n\n".join(doc.page_labelled_text() for doc in docs)
        corpus_chars = len(joined_corpus)
        page_iter = (page for doc in docs for page in doc.pages)

        tracemalloc.start()
        joined_chunks = chunk_page_labelled_text(joined_corpus)
        _, joined_peak = tracemalloc.get_traced_memory()
        del joined_chunks  # free pass 1 so it cannot inflate pass 2's peak
        tracemalloc.reset_peak()
        streamed = list(iter_page_labelled_chunks(page_iter))
        _, streamed_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.assertGreater(len(streamed), 2)
        self.assertLess(
            streamed_peak,
            joined_peak,
            msg="streaming peak reached the joined path's peak",
        )
        self.assertGreaterEqual(
            joined_peak - streamed_peak,
            corpus_chars * 0.8,
            msg="streaming no longer saves a full corpus copy — the join may "
            "be materializing",
        )

    def test_empty_pages_yield_single_empty_chunk_like_joined_path(self):
        """Edge parity: all-empty pages produce the same single empty chunk the
        joined path produces from an empty string."""
        docs = self._docs([("a.pdf", 1, ""), ("b.pdf", 1, "")])
        streamed = list(iter_page_labelled_chunks(page for d in docs for page in d.pages))
        reference = chunk_page_labelled_text("")
        self.assertEqual(len(streamed), len(reference))
        self.assertEqual(len(streamed), 1)
        self.assertEqual(streamed[0].text, reference[0].text)
        self.assertEqual(streamed[0].pages, ())

    def test_hard_cut_landing_at_end_of_corpus_emits_no_extra_chunk(self):
        """A cut that consumes the corpus exactly must not be followed by the
        overlap-tail iteration — the joined path breaks on ``end >= len(text)``
        and the streamer must too, or it emits a phantom 400-char final chunk.

        Constructed so the virtual corpus (marker + newline + text) is exactly
        ``max_chars + 1`` characters with no paragraph or sentence cut inside
        the window: the only possible cut is the hard cut, and it lands one
        past the corpus end.
        """
        from app.documents import DocumentPage

        marker_probe = DocumentPage("x.pdf", 1, "")
        text = "A" * (8001 - len(marker_probe.marker) - 1)
        page = DocumentPage("x.pdf", 1, text)
        chunks = list(iter_page_labelled_chunks([page]))
        reference = chunk_page_labelled_text(page.marker + chr(10) + text)
        self.assertEqual([c.text for c in chunks], [c.text for c in reference])
        self.assertEqual(len(chunks), 1)


class TestConfigParsing(unittest.TestCase):
    def test_positive_int_env_uses_default_when_unset_invalid_or_non_positive(self):
        with patch.dict("os.environ", {}, clear=False):
            self.assertEqual(config._positive_int_env("VA_LSE_TEST_POS_INT", 123), 123)
        with patch.dict("os.environ", {"VA_LSE_TEST_POS_INT": "bad"}, clear=False):
            self.assertEqual(config._positive_int_env("VA_LSE_TEST_POS_INT", 123), 123)
        with patch.dict("os.environ", {"VA_LSE_TEST_POS_INT": "0"}, clear=False):
            self.assertEqual(config._positive_int_env("VA_LSE_TEST_POS_INT", 123), 123)
        with patch.dict("os.environ", {"VA_LSE_TEST_POS_INT": "-5"}, clear=False):
            self.assertEqual(config._positive_int_env("VA_LSE_TEST_POS_INT", 123), 123)

    def test_positive_int_env_accepts_valid_positive_int(self):
        with patch.dict("os.environ", {"VA_LSE_TEST_POS_INT": "456"}, clear=False):
            self.assertEqual(config._positive_int_env("VA_LSE_TEST_POS_INT", 123), 456)


class TestStreamlitSecrets(unittest.TestCase):
    """Hosted deployments (Streamlit Community Cloud) ship no .env, so the key
    and endpoint must be readable from st.secrets; environment still wins."""

    _ENV_KEYS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "LLM_MODEL_MAIN", "LLM_MODEL_FAST")

    def setUp(self):
        # streamlit_secrets() memoizes per process — clear it so a test can patch
        # in its own mapping and read it back.
        cache_reset = patch.object(config, "_SECRETS_CACHE", None)
        cache_reset.start()
        self.addCleanup(cache_reset.stop)

    def _load(self, secrets, env=None):
        """load_settings() with the secrets manager faked and these env vars blank.

        Blanking (rather than clearing os.environ) keeps a developer's real
        .env from leaking the answer into a secrets-resolution test.
        """
        environ = {name: "" for name in self._ENV_KEYS}
        environ.update(env or {})
        with patch.object(config, "streamlit_secrets", return_value=secrets):
            with patch.dict("os.environ", environ, clear=False):
                return config.load_settings()

    def test_secrets_fill_settings_and_are_recorded(self):
        settings = self._load(
            {
                "OPENAI_API_KEY": "test-key-from-secrets",
                "OPENAI_BASE_URL": "https://ws-example.example.com/v1",
                "LLM_MODEL_MAIN": "qwen-from-secrets",
            }
        )
        self.assertEqual(settings.api_key, "test-key-from-secrets")
        self.assertEqual(settings.base_url, "https://ws-example.example.com/v1")
        self.assertEqual(settings.model_main, "qwen-from-secrets")
        # Not in secrets and not in the (blanked) environment → code default.
        self.assertEqual(settings.model_fast, config.DEFAULT_MODEL_FAST)
        self.assertEqual(
            settings.from_secrets,
            frozenset({"OPENAI_API_KEY", "OPENAI_BASE_URL", "LLM_MODEL_MAIN"}),
        )

    def test_environment_overrides_secrets(self):
        settings = self._load(
            {"OPENAI_API_KEY": "test-key-from-secrets", "OPENAI_BASE_URL": "https://secret/v1"},
            env={"OPENAI_API_KEY": "test-key-from-env"},
        )
        self.assertEqual(settings.api_key, "test-key-from-env")
        self.assertEqual(settings.base_url, "https://secret/v1")
        self.assertNotIn("OPENAI_API_KEY", settings.from_secrets)
        self.assertIn("OPENAI_BASE_URL", settings.from_secrets)

    def test_lowercase_secret_keys_supported(self):
        settings = self._load({"openai_api_key": "test-key-lowercase"})
        self.assertEqual(settings.api_key, "test-key-lowercase")
        self.assertEqual(settings.from_secrets, frozenset({"OPENAI_API_KEY"}))

    def test_blank_or_nested_secrets_are_ignored(self):
        settings = self._load({"OPENAI_API_KEY": "   ", "LLM_MODEL_MAIN": {"nested": 1}})
        self.assertEqual(settings.api_key, "")
        self.assertEqual(settings.model_main, config.DEFAULT_MODEL_MAIN)
        self.assertEqual(settings.from_secrets, frozenset())

    def test_no_secrets_uses_defaults(self):
        settings = self._load({})
        self.assertFalse(settings.configured)
        self.assertEqual(settings.base_url, config.DEFAULT_BASE_URL)
        self.assertEqual(settings.from_secrets, frozenset())

    def test_missing_secrets_file_is_not_an_error(self):
        """Local runs (no .streamlit/secrets.toml) must not crash on import-time reads."""
        import streamlit as st

        broken = MagicMock()
        broken.to_dict.side_effect = RuntimeError("no secrets file")
        with patch.object(st, "secrets", broken):
            self.assertEqual(config._read_streamlit_secrets(), {})


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
            model_fast = "f"

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

    fast_model = _FakeSettings.model_fast  # "fake-fast"

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
    @staticmethod
    def _multi_chunk_docs():
        """One record whose text exceeds the digest chunk budget several times."""
        from app.documents import DEFAULT_CHUNK_CHARS

        parts = [
            f"SECTION{i} " + (f"word{i} " * ((DEFAULT_CHUNK_CHARS + 800) // 8))
            for i in range(4)
        ]
        return [extract_document("big.txt", ("\n\n".join(parts)).encode())]

    @staticmethod
    def _chunk_total(messages):
        import re

        for _, message in messages:
            match = re.search(r"in (\d+) chunk", message)
            if match:
                return int(match.group(1))
        raise AssertionError(f"no chunk-count message in {messages!r}")

    def test_deterministic_rejection_skips_the_doomed_retry_round(self):
        """A non-retriable rejection is not re-attempted; a retry cannot change it.

        Uses a 400, not a refusal status: 401/403/404 end the pass early (see
        the stop test below), while this one needs the pass to run to its end so
        that the missing second attempt is what the counts prove.
        """
        from app.llm import LLMError, LLMUpstreamError

        class RefusingLLM(FakeLLM):
            def __init__(self):
                super().__init__()
                self.attempts = 0

            def chat_json(self, system, user, **kwargs):
                if "CHUNK TEXT" in user:
                    self.attempts += 1
                    raise LLMUpstreamError(
                        "LLM provider rejected the request (Error code: 400)",
                        retriable=False,
                        status_code=400,
                    )
                return super().chat_json(system, user, **kwargs)

        llm = RefusingLLM()
        messages = []
        with self.assertRaises(LLMError) as ctx:
            review_medical_records(
                llm, self._multi_chunk_docs(), progress=lambda f, m: messages.append((f, m))
            )
        chunk_total = self._chunk_total(messages)
        self.assertGreaterEqual(chunk_total, 2)
        self.assertEqual(llm.attempts, chunk_total)  # one attempt per chunk: no retry pass
        self.assertNotIn("Retrying", "\n".join(m for _, m in messages))
        error = str(ctx.exception)
        self.assertIn("which retrying cannot fix", error)
        self.assertIn(f"Chunks affected: {chunk_total} of {chunk_total}", error)
        # A 400 is per-request, not a refusal of the configuration: the pass
        # runs to the end and nothing is reported as never attempted.
        self.assertNotIn("not attempted", error)
        self.assertNotIn("Stopped early", "\n".join(m for _, m in messages))

    def test_refusal_stops_the_pass_and_counts_what_was_never_attempted(self):
        """401/403/404 end the pass: queued chunks are canceled, not visited."""
        import time

        from app import config as _config
        from app.llm import LLMError, LLMUpstreamError

        class RefusingLLM(FakeLLM):
            def __init__(self):
                super().__init__()
                self.attempts = 0

            def chat_json(self, system, user, **kwargs):
                if "CHUNK TEXT" in user:
                    self.attempts += 1
                    time.sleep(0.2)  # one worker; keep it from racing the cancel
                    raise LLMUpstreamError(
                        "LLM provider rejected the request (Error code: 403)",
                        retriable=False,
                        status_code=403,
                    )
                return super().chat_json(system, user, **kwargs)

        original = _config.RECORDS_CONCURRENCY
        _config.RECORDS_CONCURRENCY = 1
        try:
            llm = RefusingLLM()
            messages = []
            with self.assertRaises(LLMError) as ctx:
                review_medical_records(
                    llm, self._multi_chunk_docs(), progress=lambda f, m: messages.append((f, m))
                )
        finally:
            _config.RECORDS_CONCURRENCY = original
        rendered = "\n".join(m for _, m in messages)
        chunk_total = self._chunk_total(messages)
        self.assertEqual(chunk_total, 4)
        # The worker raises once and the queued chunks are canceled; at most one
        # call already under way can land before the cancel takes effect.
        self.assertLessEqual(llm.attempts, 2)
        not_attempted = chunk_total - llm.attempts
        self.assertGreaterEqual(not_attempted, 2)
        self.assertNotIn(f"— {chunk_total}/{chunk_total} chunks done", rendered)
        self.assertIn(
            f"Stopped early — {not_attempted} chunk(s) not attempted after the endpoint "
            "refused the run.",
            rendered,
        )
        error = str(ctx.exception)
        self.assertIn("HTTP 403", error)
        self.assertIn(f"Chunks affected: {llm.attempts} of {chunk_total}", error)
        self.assertIn(f"{not_attempted} more chunk(s) were not attempted", error)

    def test_transient_failures_retry_with_round_local_counts(self):
        """Every chunk retries once, and no progress line counts past its round."""
        import re

        from app.llm import LLMUpstreamError

        class FlakyOnceLLM(FakeLLM):
            def __init__(self):
                super().__init__()
                self.attempts = 0
                self._seen = set()

            def chat_json(self, system, user, **kwargs):
                if "CHUNK TEXT" in user:
                    self.attempts += 1
                    if user not in self._seen:
                        self._seen.add(user)
                        raise LLMUpstreamError(
                            "transient rate limit", retriable=True, status_code=429
                        )
                return super().chat_json(system, user, **kwargs)

        llm = FlakyOnceLLM()
        messages = []
        review_medical_records(
            llm, self._multi_chunk_docs(), progress=lambda f, m: messages.append((f, m))
        )
        chunk_total = self._chunk_total(messages)
        self.assertEqual(llm.attempts, 2 * chunk_total)  # first pass + one retry each
        retry_lines = [m for _, m in messages if m.startswith("Retrying failed chunks")]
        self.assertEqual(len(retry_lines), chunk_total)
        counts = [
            (int(done), int(total))
            for _, message in messages
            for done, total in re.findall(r"— (\d+)/(\d+) ", message)
        ]
        self.assertTrue(all(done <= total for done, total in counts))
        self.assertEqual(max(done for done, _ in counts), chunk_total)
        fractions = [fraction for fraction, _ in messages]
        self.assertEqual(fractions, sorted(fractions))  # progress never goes backwards

    def test_fail_fast_rejections_are_skipped_but_still_counted(self):
        """Breaker refusals are not re-attempted; the summary still counts them."""
        from app.circuit_breaker import CircuitBreakerOpenError
        from app.llm import LLMError, LLMUpstreamError

        class MixedLLM(FakeLLM):
            def __init__(self):
                super().__init__()
                self.attempts = 0
                self._seen = set()

            def chat_json(self, system, user, **kwargs):
                if "CHUNK TEXT" in user:
                    self.attempts += 1
                    if self.attempts <= 2:  # first two calls: refused before a request
                        raise CircuitBreakerOpenError("Circuit breaker 'llm' is OPEN (test)")
                    if user not in self._seen:
                        self._seen.add(user)
                        raise LLMUpstreamError(
                            "transient upstream error", retriable=True, status_code=503
                        )
                return super().chat_json(system, user, **kwargs)

        llm = MixedLLM()
        messages = []
        with self.assertRaises(LLMError) as ctx:
            review_medical_records(
                llm, self._multi_chunk_docs(), progress=lambda f, m: messages.append((f, m))
            )
        chunk_total = self._chunk_total(messages)
        self.assertEqual(chunk_total, 4)  # fixture is four sections
        # Two chunks were refused before a request; two failed transiently and were
        # retried. The refused two are not re-attempted: 4 + 2 attempts, not 8.
        self.assertEqual(llm.attempts, chunk_total + 2)
        self.assertEqual(
            [m for _, m in messages if m.startswith("Retrying failed chunks")],
            [f"Retrying failed chunks — {i}/{chunk_total - 2} done…" for i in (1, 2)],
        )
        error = str(ctx.exception)
        self.assertIn(f"Chunks affected: {chunk_total - 2} of {chunk_total}", error)

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
            for i in range(1501)
        ]
        digest = MedicalDigest(facts=facts)
        llm = FakeLLM()
        merged = _merge_facts(llm, digest)
        # 1,501 facts -> at least 8 parallel merge batches of 200.
        self.assertGreaterEqual(llm.merge_calls, 8)
        # Pass-through fake must not lose any distinct facts.
        self.assertEqual(merged, facts)

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

    def test_norm_key_collapses_every_whitespace_class(self):
        """``_norm_key`` is the dedupe/merge comparison key, so its normalization
        must cover the whitespace real extraction emits: tabs and runs of spaces,
        paragraph breaks, and the non-breaking/thin spaces that arrive from PDFs
        and copied web pages.
        """
        self.assertEqual(
            _norm_key("  Knee\t\tpAIN.\n\n rated   10\u00a0degrees "),
            "knee pain. rated 10 degrees",
        )
        self.assertEqual(_norm_key("a\u2009b"), "a b")  # thin space
        self.assertEqual(_norm_key("a\x0bb"), "a b")  # vertical tab
        self.assertEqual(_norm_key(""), "")

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


class TestMergeOutputBudget(unittest.TestCase):
    """Merge batches must fit the model's OUTPUT budget, not just its input.

    _merge_once asks the model to echo every distinct fact of the batch as
    JSON. With the old 200-fact batches the echo demanded ~17k output tokens
    against max_tokens=8000, so every response truncated mid-JSON and every
    merge call failed (live 2026-09-21: 13/13 unparseable). The batch sizing
    must keep the expected echo within the output budget.
    """

    def test_merge_batches_never_exceed_output_budget(self):
        # Live-run shape: near-duplicate variants of the same underlying
        # events, so the model has real consolidation work and the batch
        # input is realistically large (~140k chars at the old size).
        facts = [
            MedicalFact(
                date=f"2024-{(i % 12) + 1:02d}-1{i % 9}",
                type="symptom",
                description=f"Nightmare episode variant {i}: woke thrashing, "
                f"disoriented, removed CPAP; grounded within {5 + i % 20} minutes.",
                source=f"9:18:26_Part{i % 91 + 1}.pdf p.{i % 20 + 1}",
                quote="Quote text sized like a real extraction " * 2,
            )
            for i in range(1501)
        ]
        digest = MedicalDigest(facts=facts)

        class SizeRecordingLLM(FakeLLM):
            """Pass-through merge that records the input size of each call."""

            def __init__(self):
                super().__init__()
                self.merge_sizes: list[int] = []

            def _merge_facts(self, facts_list):  # noqa: ARG002 - signature only
                return facts_list

            def chat_json(self, system, user, **kwargs):
                if "Deduplicate and consolidate" in user:
                    # Mirrors _merge_once's parsing: strip the prompt header.
                    payload = user.rsplit("\n\n", 1)[-1]
                    self.merge_sizes.append(len(payload))
                    try:
                        return {"facts": json.loads(payload)}
                    except json.JSONDecodeError:
                        return {"facts": []}
                return super().chat_json(system, user, **kwargs)

        llm = SizeRecordingLLM()
        merged = _merge_facts(llm, digest)
        self.assertEqual(len(merged), 1501)  # pass-through loses nothing
        self.assertGreater(len(llm.merge_sizes), 0)
        for size in llm.merge_sizes:
            # ~48 tokens per echoed fact against the 8,000-token output
            # budget -> the payload must stay far below what the old
            # 200-fact batches produced (~140k chars, 100% truncation).
            self.assertLess(size, 30_000)


if __name__ == "__main__":
    unittest.main()
