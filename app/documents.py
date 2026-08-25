"""Document text extraction and chunking utilities."""
from __future__ import annotations

import io
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md", ".docx")

# Chunk size chosen so a digest prompt (system + knowledge + chunk) stays well under
# typical 32k-token context windows of fast models.
DEFAULT_CHUNK_CHARS = 12_000
CHUNK_OVERLAP_CHARS = 400
MAX_STATEMENT_CHARS = 60_000


class ExtractionError(RuntimeError):
    """Raised when a document cannot be read or contains no extractable text."""


@dataclass
class DocumentPage:
    """One page (or one text block) from an uploaded document."""

    filename: str
    page: int  # 1-based; text files use page 1
    text: str

    @property
    def label(self) -> str:
        return f"{self.filename} p.{self.page}"


@dataclass
class ExtractedDocument:
    """A fully extracted document."""

    filename: str
    pages: list[DocumentPage] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)

    def page_labelled_text(self) -> str:
        """Full text with page markers so LLM citations can reference pages."""
        parts = [
            f"[{p.filename} — page {p.page}]\n{p.text}" for p in self.pages if p.text.strip()
        ]
        return "\n\n".join(parts)


# ---------------------------------------------------------------- extraction
def extract_document(filename: str, data: bytes) -> ExtractedDocument:
    """Extract text from an uploaded file based on its extension."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return _extract_pdf(filename, data)
    if lower.endswith((".txt", ".md")):
        return document_from_text(filename, data.decode("utf-8", errors="replace"))
    if lower.endswith(".docx"):
        return _extract_docx(filename, data)
    raise ExtractionError(
        f"{filename}: unsupported file type. Use PDF, TXT, MD, or DOCX."
    )


def document_from_text(filename: str, text: str) -> ExtractedDocument:
    """Create a single-page extracted document from plain text."""
    text = clean_text(text)
    if not text:
        raise ExtractionError(f"{filename}: file is empty.")
    return ExtractedDocument(filename=filename, pages=[DocumentPage(filename, 1, text)])


def _extract_pdf(filename: str, data: bytes) -> ExtractedDocument:
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"{filename}: could not read PDF ({exc})") from exc

    doc = ExtractedDocument(filename=filename)
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:  # noqa: BLE001 - skip unreadable page, keep going
            text = ""
        if text:
            doc.pages.append(DocumentPage(filename, index, text))

    if not doc.pages or doc.char_count < 20:
        raise ExtractionError(
            f"{filename}: no extractable text. The PDF may be scanned/image-only; "
            "please upload a text-based PDF or OCR it first."
        )
    return doc


def _extract_docx(filename: str, data: bytes) -> ExtractedDocument:
    """Minimal DOCX text extraction without external dependencies."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
        xml_bytes = archive.read("word/document.xml")
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"{filename}: could not read DOCX ({exc})") from exc

    root = ET.fromstring(xml_bytes)
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs: list[str] = []
    for para in root.iter(f"{namespace}p"):
        runs = [node.text or "" for node in para.iter(f"{namespace}t")]
        line = "".join(runs).strip()
        if line:
            paragraphs.append(line)
    text = "\n".join(paragraphs).strip()
    if not text:
        raise ExtractionError(f"{filename}: DOCX contains no readable text.")
    return ExtractedDocument(filename=filename, pages=[DocumentPage(filename, 1, text)])


def clean_text(text: str) -> str:
    """Normalize whitespace without destroying paragraph structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ------------------------------------------------------------------ chunking
@dataclass
class Chunk:
    """A slice of page-labelled medical-record text for one digest pass."""

    index: int  # 1-based chunk number
    total: int  # total chunks
    text: str

    @property
    def label(self) -> str:
        return f"chunk {self.index}/{self.total}"


def chunk_page_labelled_text(text: str, max_chars: int = DEFAULT_CHUNK_CHARS) -> list[Chunk]:
    """Split page-labelled text into overlapping chunks at paragraph boundaries."""
    text = clean_text(text)
    if len(text) <= max_chars:
        return [Chunk(1, 1, text)]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # Prefer cutting at a paragraph, then sentence, then hard cut.
            window = text[start:end]
            cut = window.rfind("\n\n")
            if cut < max_chars // 2:
                cut = window.rfind(". ")
            if cut < max_chars // 2:
                cut = max_chars
            end = start + cut + 1
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP_CHARS, start + 1)

    total = len(chunks)
    return [Chunk(i, total, c) for i, c in enumerate(chunks, start=1)]
