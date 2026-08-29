"""Document text extraction and chunking utilities."""
from __future__ import annotations

import io
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from . import config

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md", ".docx")

# Chunk size chosen so a digest prompt (system + knowledge + chunk) stays well under
# typical context windows while small enough that dense pages are never truncated
# mid-extraction. Overridable via VA_LSE_DIGEST_CHUNK_CHARS.
DEFAULT_CHUNK_CHARS = config.DIGEST_CHUNK_CHARS
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


def records_from_local_path(path: str) -> list[ExtractedDocument]:
    """Read supported record files directly from a local file or folder path.

    Only meaningful when the app runs on the same machine as the records
    (local Streamlit run) — the UI gates this behind a local-run check.

    Raises ExtractionError if the path does not exist, contains no supported
    files, or none of the files could be extracted.
    """
    root = Path(path).expanduser()
    if not root.exists():
        raise ExtractionError(f"Local path not found: {root}")

    if root.is_file():
        files = [root]
    else:
        files = sorted(
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    if not files:
        raise ExtractionError(
            f"No supported record files (.pdf/.txt/.md/.docx) found in: {root}"
        )

    documents: list[ExtractedDocument] = []
    errors: list[str] = []
    for file in files:
        try:
            documents.append(extract_document(file.name, file.read_bytes()))
        except ExtractionError as exc:
            errors.append(str(exc))
    if not documents:
        detail = f" ({'; '.join(errors)}) " if errors else " "
        raise ExtractionError(f"Could not load any records from {root}{detail}")
    return documents


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


# ------------------------------------------------------- paragraph retrieval
@dataclass
class Paragraph:
    """One paragraph of record text with its page label, ready for scoring."""

    label: str
    text: str


_PARAGRAPH_CACHE: dict[tuple, list[Paragraph]] = {}


def paragraph_index(doc: ExtractedDocument, min_chars: int = 40) -> list[Paragraph]:
    """Split a document into scannable paragraphs, cached per document.

    Large record sets (1,000+ pages) are searched once per claim batch, so the
    split/tokenize work is memoized instead of repeated for every query.
    """
    key = (doc.filename, len(doc.pages), doc.char_count)
    cached = _PARAGRAPH_CACHE.get(key)
    if cached is not None:
        return cached
    paragraphs: list[Paragraph] = []
    for page in doc.pages:
        for block in re.split(r"\n{2,}", page.text):
            block = block.strip()
            if len(block) >= min_chars:
                paragraphs.append(Paragraph(page.label, block))
    if len(_PARAGRAPH_CACHE) > 64:  # keep the cache bounded
        _PARAGRAPH_CACHE.clear()
    _PARAGRAPH_CACHE[key] = paragraphs
    return paragraphs
