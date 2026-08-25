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
)
from app.llm import _parse_json  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
