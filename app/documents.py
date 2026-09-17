"""Document text extraction and chunking utilities."""
from __future__ import annotations

import csv
import io
import json
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from typing import Protocol

from pypdf import PdfReader

from . import config


class UploadedFile(Protocol):
    """Minimal surface of ``streamlit.runtime.uploaded_file_manager.UploadedFile`` needed here."""

    name: str
    size: int

    def getvalue(self) -> bytes: ...

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md", ".docx")

# Chunk size chosen so a digest prompt (system + knowledge + chunk) stays well under
# typical context windows while small enough that dense pages are never truncated
# mid-extraction. Overridable via VA_LSE_DIGEST_CHUNK_CHARS.
DEFAULT_CHUNK_CHARS = config.DIGEST_CHUNK_CHARS
CHUNK_OVERLAP_CHARS = 400
MAX_STATEMENT_CHARS = 60_000
# Max observations chars for the draft pathway (witness observations)
MAX_OBSERVATIONS_CHARS = 60_000
# Internal pipeline limits — raised to 80k so the upfront MAX_* checks are the
# first line of defense; no silent 30k/40k truncation for statements that
# passed the 60k UI gate. Bypass callers still get bounded prompts.
EVALUATE_INTERNAL_MAX_CHARS = 80_000
DRAFT_INTERNAL_MAX_CHARS = 80_000
DOCX_READ_CHUNK_BYTES = 64 * 1024


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
    max_member_bytes = config.DOCX_MAX_INTERNAL_FILE_BYTES
    max_total_bytes = config.DOCX_MAX_TOTAL_UNCOMPRESSED_BYTES
    max_member_count = config.DOCX_MAX_INTERNAL_FILE_COUNT
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            existing_total_bytes = _validate_docx_uncompressed_sizes(
                filename,
                archive=archive,
                max_member_bytes=max_member_bytes,
                max_total_bytes=max_total_bytes,
                max_member_count=max_member_count,
                target_member_name="word/document.xml",
            )
            xml_bytes = _read_docx_member_limited(
                filename,
                archive=archive,
                member_name="word/document.xml",
                max_member_bytes=max_member_bytes,
                max_total_bytes=max_total_bytes,
                existing_total_bytes=existing_total_bytes,
            )
    except ExtractionError:
        raise
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


def _validate_docx_uncompressed_sizes(
    filename: str,
    archive: zipfile.ZipFile,
    max_member_bytes: int,
    max_total_bytes: int,
    max_member_count: int,
    target_member_name: str,
) -> int:
    total_uncompressed = 0
    member_count = 0
    non_target_total = 0
    for info in archive.infolist():
        if info.is_dir():
            continue
        member_count += 1
        if member_count > max_member_count:
            raise ExtractionError(
                f"{filename}: DOCX has too many internal files "
                f"({member_count} > {max_member_count})."
            )
        if info.file_size > max_member_bytes:
            raise ExtractionError(
                f"{filename}: DOCX member '{info.filename}' exceeds max uncompressed "
                f"size ({info.file_size} bytes > {max_member_bytes} bytes)."
            )
        total_uncompressed += info.file_size
        if info.filename != target_member_name:
            non_target_total += info.file_size
        if total_uncompressed > max_total_bytes:
            raise ExtractionError(
                f"{filename}: DOCX total uncompressed size exceeds limit "
                f"({total_uncompressed} bytes > {max_total_bytes} bytes)."
            )
    return non_target_total


def _read_docx_member_limited(
    filename: str,
    archive: zipfile.ZipFile,
    member_name: str,
    max_member_bytes: int,
    max_total_bytes: int,
    existing_total_bytes: int,
) -> bytes:
    try:
        info = archive.getinfo(member_name)
    except KeyError as exc:
        raise ExtractionError(f"{filename}: missing DOCX content ({member_name}).") from exc
    if info.file_size > max_member_bytes:
        raise ExtractionError(
            f"{filename}: DOCX member '{member_name}' exceeds max uncompressed size "
            f"({info.file_size} bytes > {max_member_bytes} bytes)."
        )

    output = bytearray()
    with archive.open(info, "r") as stream:
        while True:
            chunk = stream.read(DOCX_READ_CHUNK_BYTES)
            if not chunk:
                break
            output.extend(chunk)
            if len(output) > max_member_bytes:
                raise ExtractionError(
                    f"{filename}: DOCX member '{member_name}' exceeded max uncompressed "
                    f"size while reading ({len(output)} bytes > {max_member_bytes} bytes)."
                )
            if existing_total_bytes + len(output) > max_total_bytes:
                raise ExtractionError(
                    f"{filename}: DOCX total uncompressed size exceeded while reading "
                    f"({existing_total_bytes + len(output)} bytes > {max_total_bytes} bytes)."
                )
    return bytes(output)


def extract_uploaded_documents(
    files: "list[UploadedFile]",
) -> tuple[list[ExtractedDocument], list[str]]:
    """Extract uploaded files; returns ``(documents, skipped)``.

    ``skipped`` holds one message per file that could not be extracted (e.g.
    an image-only PDF), so callers can surface them as warnings instead of
    letting unreadable uploads silently vanish.
    """
    documents: list[ExtractedDocument] = []
    skipped: list[str] = []
    for uploaded in files:
        try:
            documents.append(extract_document(uploaded.name, uploaded.getvalue()))
        except ExtractionError as exc:
            skipped.append(str(exc))
    return documents, skipped


def records_from_local_path(path: str) -> tuple[list[ExtractedDocument], list[str]]:
    """Read supported record files directly from a local file or folder path.

    Returns ``(documents, skipped)`` where ``skipped`` lists one message per
    file that could not be extracted (e.g. image-only PDF) so the caller can
    surface them as warnings instead of failing the whole load. Filenames are
    preserved; files inside nested subfolders keep their relative path so
    same-named files in different folders do not collide.

    Raises ExtractionError if the path does not exist, contains no supported
    files, or none of the files could be extracted.
    """
    root = Path(path).expanduser()
    if not root.exists():
        raise ExtractionError(f"Local path not found: {root}")

    if root.is_file():
        files = [root]
        label_for = lambda file: file.name  # noqa: E731 - single file keeps bare name
    else:
        files = sorted(
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        label_for = lambda file: str(file.relative_to(root))  # noqa: E731
    if not files:
        raise ExtractionError(
            f"No supported record files (.pdf/.txt/.md/.docx) found in: {root}"
        )

    documents: list[ExtractedDocument] = []
    skipped: list[str] = []
    for file in files:
        try:
            documents.append(extract_document(label_for(file), file.read_bytes()))
        except ExtractionError as exc:
            skipped.append(str(exc))
    if not documents:
        detail = f" ({'; '.join(skipped)}) " if skipped else " "
        raise ExtractionError(f"Could not load any records from {root}{detail}")
    return documents, skipped


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


_PARAGRAPH_CACHE: dict[tuple[str, int, int], list[Paragraph]] = {}


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


# ------------------------------------------------------------ TF-IDF search
# Feature: Medical Record Search & Citation Index
_LABEL_PAGE_RE = re.compile(r"^(?P<filename>.*) p\.(?P<page>\d+)$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SEARCH_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b")
SEARCH_RESULT_LIMIT = 20
SEARCH_EXCERPT_MAX_CHARS = 400


@dataclass
class SearchResult:
    """One ranked medical-record passage matching a search query."""

    label: str
    excerpt: str
    score: float
    filename: str
    page: int


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_inverted_index(paragraphs: list[Paragraph]) -> dict[str, dict[int, int]]:
    """Build a token -> {paragraph_index: term_frequency} inverted index.

    Kept as a standalone step (rather than inlined in ``search_records``) so
    the ranking step can be unit tested against a hand-built index.
    """
    index: dict[str, dict[int, int]] = {}
    for i, paragraph in enumerate(paragraphs):
        counts = Counter(_tokenize(paragraph.text))
        for token, count in counts.items():
            index.setdefault(token, {})[i] = count
    return index


def _parse_search_date(raw: str) -> date | None:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _paragraph_dates(text: str) -> list[date]:
    dates: list[date] = []
    for match in _SEARCH_DATE_RE.finditer(text):
        parsed = _parse_search_date(match.group(1))
        if parsed:
            dates.append(parsed)
    return dates


def _paragraph_in_date_range(text: str, date_from: date | None, date_to: date | None) -> bool:
    """True when at least one date mentioned in the paragraph falls in range."""
    dates = _paragraph_dates(text)
    if not dates:
        return False
    for found in dates:
        if date_from and found < date_from:
            continue
        if date_to and found > date_to:
            continue
        return True
    return False


def _highlight(text: str, tokens: list[str], max_chars: int = SEARCH_EXCERPT_MAX_CHARS) -> str:
    """Bold every whole-word query-token match in a bounded excerpt (markdown)."""
    excerpt = text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"
    unique_tokens = sorted({t for t in tokens if len(t) > 1}, key=len, reverse=True)
    for token in unique_tokens:
        excerpt = re.sub(rf"(?i)\b({re.escape(token)})\b", r"**\1**", excerpt)
    return excerpt


def _split_label(label: str) -> tuple[str, int]:
    match = _LABEL_PAGE_RE.match(label)
    if not match:
        return label, 1
    return match.group("filename"), int(match.group("page"))


def search_records(
    documents: list[ExtractedDocument],
    query: str,
    *,
    date_from: date | None = None,
    date_to: date | None = None,
    provider: str | None = None,
    limit: int = SEARCH_RESULT_LIMIT,
) -> list[SearchResult]:
    """Rank paragraphs across ``documents`` against ``query`` via TF-IDF.

    Supports an optional inclusive date-range filter (matched against any
    date mentioned in the paragraph text) and an optional provider filter
    (case-insensitive substring match against the paragraph text). Returns
    the top ``limit`` results, highest score first.
    """
    query = (query or "").strip()
    if not query:
        return []

    paragraphs: list[Paragraph] = []
    for doc in documents:
        paragraphs.extend(paragraph_index(doc))
    if not paragraphs:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    index = build_inverted_index(paragraphs)
    n_paragraphs = len(paragraphs)
    scores = [0.0] * n_paragraphs
    for token in set(query_tokens):
        postings = index.get(token)
        if not postings:
            continue
        # Smoothed inverse-document-frequency: common across the record set
        # weighs less than a term that appears in only a few paragraphs.
        idf = math.log((n_paragraphs + 1) / (len(postings) + 1)) + 1.0
        for paragraph_index_, term_frequency in postings.items():
            scores[paragraph_index_] += term_frequency * idf

    ranked_indices = sorted(
        (i for i, s in enumerate(scores) if s > 0), key=lambda i: scores[i], reverse=True
    )

    provider_query = (provider or "").strip().lower()
    results: list[SearchResult] = []
    for i in ranked_indices:
        paragraph = paragraphs[i]
        if provider_query and provider_query not in paragraph.text.lower():
            continue
        if (date_from or date_to) and not _paragraph_in_date_range(
            paragraph.text, date_from, date_to
        ):
            continue
        filename, page = _split_label(paragraph.label)
        results.append(
            SearchResult(
                label=paragraph.label,
                excerpt=_highlight(paragraph.text, query_tokens),
                score=scores[i],
                filename=filename,
                page=page,
            )
        )
        if len(results) >= limit:
            break
    return results


# ------------------------------------------------------------ citation index
# Feature: Medical Record Search & Citation Index (F2.S2)
_CITATION_EXPORT_FIELDS = ("excerpt", "source")
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _csv_formula_guard(value: str) -> str:
    """Neutralize CSV/spreadsheet formula injection (OWASP CSV Injection).

    Excerpt/source text originates from uploaded medical-record content,
    which is untrusted input. If a cell's value begins with ``=``, ``+``,
    ``-``, or ``@`` (or a leading tab/CR), spreadsheet applications such as
    Excel or Google Sheets may interpret it as a formula when the exported
    file is opened, enabling formula-injection attacks against whoever opens
    the download. Prefixing with a single quote forces the cell to be
    treated as literal text while leaving the visible content unchanged.
    """
    if value.startswith(_CSV_FORMULA_PREFIXES):
        return "'" + value
    return value


def export_citation_index(citations: list[dict[str, str]], fmt: str) -> bytes:
    """Serialize the citation index (excerpt + source per entry) to bytes.

    ``fmt`` is ``"csv"`` or ``"json"`` (case-insensitive); raises
    :class:`ValueError` for anything else so a caller-facing error message
    can be shown instead of silently exporting the wrong format.
    """
    normalized = (fmt or "").strip().lower()
    if normalized == "json":
        return json.dumps(citations, indent=2, ensure_ascii=False).encode("utf-8")
    if normalized == "csv":
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(_CITATION_EXPORT_FIELDS))
        writer.writeheader()
        for citation in citations:
            writer.writerow(
                {
                    "excerpt": _csv_formula_guard(str(citation.get("excerpt", ""))),
                    "source": _csv_formula_guard(str(citation.get("source", ""))),
                }
            )
        return buffer.getvalue().encode("utf-8")
    raise ValueError(f"Unsupported citation export format: {fmt!r} (expected 'csv' or 'json')")
