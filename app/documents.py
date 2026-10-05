"""Document text extraction and chunking utilities."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import threading
import zipfile
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from typing import Any, Iterable, Iterator, Protocol, Sequence

from pypdf import PdfReader

from . import config
from . import ingestion_policy as policy


class UploadedFile(Protocol):
    """Minimal surface of ``streamlit.runtime.uploaded_file_manager.UploadedFile`` needed here."""

    name: str
    size: int

    def getvalue(self) -> bytes: ...


class RecordExtractor(Protocol):
    """How one record file's bytes become documents.

    Both ingestion paths — the uploader and the local-folder reader — funnel
    through ``_expand_and_extract``, so this port is the app's single seam
    between "a file arrived" and "its text is in memory". Keeping the seam
    narrow is the point: the reader in this module is the only PDF/DOCX parser
    in the project (page markers, running-header stripping and chunk boundaries
    all depend on it), so an adapter is allowed to change *where* the bytes are
    read, never *how* the text is shaped.

    Implementations live here (``InProcessExtractor``, the default) and in
    ``app/extractors.py`` (a sandbox that can OCR a scan, which this process
    deliberately cannot). Which one runs is a config choice, installed once by
    ``install_configured_extractor()``; the views never see the difference.
    """

    def extract(self, label: str, data: bytes) -> tuple[list[ExtractedDocument], list[str]]:
        """Return ``(documents, skipped)`` for one file, as the uploader expects.

        ``label`` is the name the user uploaded (or the file's relative path),
        and it is the name every page marker, citation and skip message must
        carry — an adapter reading the file elsewhere still answers under the
        label it was handed.
        """
        ...

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md", ".docx")
# Uploading the folder you were given is the natural thing to do, so archives are
# accepted and expanded (bounded — see ``archive_members``) rather than rejected.
ARCHIVE_EXTENSIONS = (".zip",)

# How a document's text is addressed in citations. "page" is a real PDF page
# (or a single-page file); "block" means the source has no page numbers at all
# (a long .txt/.md/.docx) and the number is a synthesized block. Chunks and facts
# cite whichever applies, so a 200-page DOCX no longer reports every fact as
# "p.1" while the file genuinely has no pages to point at.
PAGE = "page"
BLOCK = "block"

# Marker emitted by ``ExtractedDocument.page_labelled_text`` and parsed back out
# of a chunk to recover which pages a chunk covers. Defined once: the digest's
# fact citations are derived from these markers, so a format change must not be
# able to drift between the writer and the reader.
_PAGE_MARKER_RE = re.compile(
    r"\[(?P<file>[^\[\]]+?) — (?P<kind>page|block) (?P<num>\d+)\]"
)


def page_marker(filename: str, kind: str, number: int) -> str:
    """Return the in-text marker that identifies one page/block of a document."""
    return f"[{filename} — {kind} {number}]"


def chunk_source_hint(pages: Sequence[tuple[str, str, int]]) -> str:
    """Compress ``(filename, kind, number)`` occurrences into a citation range.

    A chunk is a contiguous slice of the concatenated record text, so the pages
    it covers form runs: ``clinic.pdf p.3–p.9`` (one file) or
    ``clinic.pdf p.9; labs.pdf p.1–p.2`` (straddling a file boundary). Returns
    an empty string when the chunk carries no markers (e.g. hand-built text).
    """
    runs: list[tuple[str, str, int, int]] = []
    for filename, kind, number in pages:
        if runs and runs[-1][0] == filename and runs[-1][1] == kind and runs[-1][3] + 1 == number:
            previous = runs[-1]
            runs[-1] = (previous[0], previous[1], previous[2], number)
        else:
            runs.append((filename, kind, number, number))
    labels: list[str] = []
    for filename, kind, first, last in runs:
        prefix = "p." if kind == PAGE else "b."
        span = f"{prefix}{first}" if first == last else f"{prefix}{first}-{prefix}{last}"
        labels.append(f"{filename} {span}")
    return "; ".join(labels)

# Chunk size chosen so a digest prompt (system + knowledge + chunk) stays well under
# typical context windows while small enough that dense pages are never truncated
# mid-extraction. Overridable via VA_LSE_DIGEST_CHUNK_CHARS (clamped there too).
DEFAULT_CHUNK_CHARS = config.DIGEST_CHUNK_CHARS
CHUNK_OVERLAP_CHARS = 400
# Floor for any caller-supplied chunk budget. A budget at or below
# ``CHUNK_OVERLAP_CHARS`` is not a smaller chunk — it is a chunk *explosion*, and
# the mechanism is the overlap, not the cut: after a hard cut the next chunk starts
# at ``end - CHUNK_OVERLAP_CHARS``, so with ``max_chars <= overlap`` every chunk
# advances the cursor by a single character. Measured on a 0.38 MB corpus:
# max_chars=8 produced 399,992 chunks and max_chars=64 produced 399,936 — exactly
# 1,048,5xx chunks per MB, i.e. one chunk (up to max_chars long) per input
# character, which on the 5,000-page bundle is ~12 million chunks. The floor keeps
# the progress per chunk at ``max_chars - overlap + 1`` >= 601 characters, which
# bounds a pathological budget to a slow run instead of an OOM.
MIN_CHUNK_CHARS = 4 * CHUNK_OVERLAP_CHARS  # 1600: comfortably above the overlap
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
    page: int  # 1-based; "block" documents number synthesized blocks instead
    text: str
    kind: str = PAGE  # PAGE for a real page, BLOCK when the source has none
    source_part: str = ""  # DOCX package part; never a filesystem path to open

    @property
    def label(self) -> str:
        prefix = "p." if self.kind == PAGE else "b."
        return f"{self.filename} {prefix}{self.page}"

    @property
    def marker(self) -> str:
        return page_marker(self.filename, self.kind, self.page)


@dataclass
class ExtractedDocument:
    """A fully extracted document."""

    filename: str
    pages: list[DocumentPage] = field(default_factory=list)
    # Pages present in the *source file*, including pages whose text could not be
    # extracted. A 400-page scan bundle with 30 text pages must not report itself
    # as a 30-page record set, so the two counts are tracked separately.
    total_pages: int = 0
    # Source page numbers that yielded no text (image-only scans, unreadable
    # pages). Named rather than counted: the user needs to know *which* pages of
    # their records were never read.
    unreadable_pages: list[int] = field(default_factory=list)
    pagination: str = PAGE
    coverage_known: bool = True
    source_sha256: str = ""
    extraction_method: str = ""
    text_encoding: str = ""

    @property
    def source_page_count(self) -> int:
        """Pages in the source file (falls back to extracted pages when unknown)."""
        return self.total_pages or len(self.pages)

    @property
    def unreadable_count(self) -> int:
        return len(self.unreadable_pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)

    def page_labelled_text(self) -> str:
        """Full text with page markers so LLM citations can reference pages.

        Each page's text is stripped, so every part starts with ``[`` and ends with
        a non-whitespace character. That is what makes ``clean_text`` part-local:
        no part edge can interact with a ``\n\n`` seam, which is the property the
        streaming cutter's per-page cleaning relies on to produce the same bytes as
        cleaning this joined string (``iter_page_labelled_chunks``). Without the
        strip, a page extracted with a trailing space (PDF extraction emits them)
        gave the joined path ``"…body \n\n[marker]"`` and the streamer
        ``"…body\n\n[marker]"`` — a silent byte difference between the two cutters,
        so the equivalence held only for pages an extractor happened to pre-clean.
        """
        parts = [f"{p.marker}\n{p.text.strip()}" for p in self.pages if p.text.strip()]
        return "\n\n".join(parts)

    def page_records(self) -> list[tuple[str, str, int]]:
        """``(filename, kind, page)`` for every page, for chunk-span recovery."""
        return [(p.filename, p.kind, p.page) for p in self.pages]


# ---------------------------------------------------------------- extraction
def _decode_text(data: bytes) -> str:
    """Decode supported Unicode; never guess or replace source bytes."""
    try:
        return policy.decode_text(data)
    except policy.IngestionRefused as exc:
        raise ExtractionError(str(exc)) from exc


def extract_document(filename: str, data: bytes) -> ExtractedDocument:
    """Validate identity, signature and structure before extracting source text."""
    try:
        policy.validate_label(filename)
        policy.validate_signature(Path(filename).suffix.lower(), data)
    except policy.IngestionRefused as exc:
        raise ExtractionError(str(exc)) from exc
    lower = filename.lower()
    if lower.endswith(".pdf"):
        doc = _extract_pdf(filename, data)
        doc.extraction_method = "pdf-text-unreviewed"
    elif lower.endswith((".txt", ".md")):
        doc = document_from_text(filename, _decode_text(data))
        doc.extraction_method = "strict-unicode"
        doc.text_encoding = ("utf-32-bom" if data.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff"))
                             else "utf-16-bom" if data.startswith((b"\xff\xfe", b"\xfe\xff"))
                             else "utf-8-sig" if data.startswith(b"\xef\xbb\xbf") else "utf-8")
    elif lower.endswith(".docx"):
        doc = _extract_docx(filename, data)
        doc.extraction_method = "docx-stories-unreviewed"
    else:
        raise ExtractionError(f"{filename}: unsupported file type. Use PDF, TXT, MD, or DOCX.")
    doc.source_sha256 = hashlib.sha256(data).hexdigest()
    return doc


def _blocks_from_text(
    filename: str, text: str, block_chars: int | None = None
) -> list[DocumentPage]:
    """Split paged-less text (.txt/.md/.docx) into addressable blocks.

    A .txt/.docx has no page numbers, so every fact extracted from one used to be
    cited "p.1" — provenance that is false for a 200-page document. Long text is
    split at paragraph boundaries into blocks of roughly
    ``config.DOCUMENT_BLOCK_CHARS`` and labelled ``b.1``, ``b.2``… (see
    ``DocumentPage.label``); short text stays a single page so the common case is
    unchanged.
    """
    limit = config.DOCUMENT_BLOCK_CHARS if block_chars is None else block_chars
    text = clean_text(text)
    if len(text) <= limit:
        return [DocumentPage(filename, 1, text, PAGE)]

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    blocks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > limit:
            blocks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
        # A single oversized paragraph (a table dump, a pasted export) still has to
        # be cut, or one block would hold the whole file and defeat the point.
        while len(current) > limit:
            blocks.append(current[:limit])
            current = current[limit:]
    if current:
        blocks.append(current)
    return [DocumentPage(filename, i, block, BLOCK) for i, block in enumerate(blocks, start=1)]


def document_from_text(filename: str, text: str) -> ExtractedDocument:
    """Create an extracted document from plain text, blocked when it is long."""
    text = clean_text(text)
    if not text:
        raise ExtractionError(f"{filename}: file is empty.")
    pages = _blocks_from_text(filename, text)
    return ExtractedDocument(
        filename=filename,
        pages=pages,
        total_pages=len(pages),
        pagination=pages[0].kind if pages else PAGE,
    )


_DATE_TOKEN_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b"
)
# A page this short is either a mostly-blank page or a layout the extractor only
# half-recovered; it is worth trying the other extraction mode on it.
_WEAK_PAGE_CHARS = 400


def _extraction_score(text: str) -> tuple[int, int]:
    """Rank an extraction of the same page: dates first, then completeness.

    Print-to-PDF clinical pages are multi-column (date | provider | value), and
    pypdf's default reader interleaves those columns while ``layout`` mode keeps
    rows intact. The columns are what the LLM needs to attach the right date to
    the right result, so the extraction that preserves the most date tokens wins;\n    text length only breaks ties.
    """
    return (len(_DATE_TOKEN_RE.findall(text)), len(text))


def _page_text(page: Any) -> str:
    """Best-effort text for one PDF page, preferring the layout-preserving read.

    Tries the default extraction and, when the page looks weak or the default
    read has no dates, also ``extraction_mode="layout"``; keeps whichever scores
    higher. Layout mode is unavailable on older pypdf releases, so a ``TypeError``
    from the unexpected keyword simply leaves the default read in place.
    """
    default = ""
    try:
        default = (page.extract_text() or "").strip()
    except Exception:  # noqa: BLE001 - unreadable page, keep going
        default = ""
    if not config.PDF_LAYOUT_EXTRACTION:
        return normalize_tabular_rows(_dehyphenate(default))
    if default and len(default) >= _WEAK_PAGE_CHARS and _extraction_score(default)[0] > 0:
        return normalize_tabular_rows(_dehyphenate(default))
    try:
        layout = (page.extract_text(extraction_mode="layout") or "").strip()
    except TypeError:  # pypdf too old for layout mode
        return normalize_tabular_rows(_dehyphenate(default))
    except Exception:  # noqa: BLE001 - layout pass failed, keep the default read
        return normalize_tabular_rows(_dehyphenate(default))
    best = layout if _extraction_score(layout) > _extraction_score(default) else default
    return normalize_tabular_rows(_dehyphenate(best))


def _page_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


# A word broken across a line by the PDF's own wrapping: "hyper-\ntension".
_HYPHEN_BREAK_RE = re.compile(r"(?<=[A-Za-z])-\n(?=[a-z])")


def _dehyphenate(text: str) -> str:
    """Rejoin words the PDF split at a line break.

    Without this, ``hyper-\ntension`` is two tokens and never matches a claim
    about hypertension — the wrap, not the content, decides whether evidence is
    found. The trade-off is a genuinely hyphenated compound split at a line break
    (``well-\nbeing``) being joined; for retrieval purposes a joined compound
    still matches on the shared tokens, while a split one matches neither.
    """
    return _HYPHEN_BREAK_RE.sub("", text)


# Column gaps inside a table row. Two or more spaces is what print-to-PDF output
# pads between columns; a single space is ordinary prose.
_TABLE_GAP_RE = re.compile(r"\s{2,}")
_TABLE_MAX_FIELDS = 8
_TABLE_MAX_LINE_CHARS = 200


def normalize_tabular_rows(text: str) -> str:
    """Rewrite column-aligned table rows as ``field | field | value`` lines.

    Lab and vitals tables arrive as padded columns, so the model sees a line of
    loosely related tokens and has to guess which value belongs to which label —
    the failure mode behind a fact carrying the wrong date or dose. Splitting on
    the column gaps and joining with ``|`` keeps a row's fields adjacent and in
    order. Nothing is added or reordered, and only lines with at least three
    non-empty fields, a digit, and no sentence-ending period are touched, so prose
    (which does not survive those tests) is left alone.
    """
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        fields = [field.strip() for field in _TABLE_GAP_RE.split(stripped) if field.strip()]
        if (
            3 <= len(fields) <= _TABLE_MAX_FIELDS
            and len(stripped) <= _TABLE_MAX_LINE_CHARS
            and not stripped.endswith(".")
            and any(any(ch.isdigit() for ch in field) for field in fields)
        ):
            out.append(" | ".join(fields))
        else:
            out.append(line)
    return "\n".join(out)


def strip_running_headers(pages: list[DocumentPage]) -> list[DocumentPage]:
    """Preserve source evidence: repetition alone cannot identify boilerplate.

    Running headers may be suppressed in a derived view with a verified format
    detector, but source pages and clinical statements must remain immutable.
    """
    return pages


def _extract_pdf(filename: str, data: bytes) -> ExtractedDocument:
    try:
        reader = PdfReader(io.BytesIO(data), strict=True)
        encrypted = bool(getattr(reader, "is_encrypted", False))
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"{filename}: could not read PDF structure.") from exc

    # A password-protected export (My HealtheVet offers one) parses fine and then
    # raises FileNotDecryptedError the moment its pages are touched. Left uncaught
    # that escapes the caller's ``except ExtractionError`` and takes down the whole
    # upload with a pypdf traceback, so name it instead.
    if encrypted:
        raise ExtractionError(
            f"{filename}: this PDF is password-protected, so its text cannot be read. "
            "Open it with the password and re-save (or print to PDF) without one, then "
            "upload that copy."
        )
    try:
        policy.validate_pdf(reader)
        source_pages = list(reader.pages)
    except policy.IngestionRefused as exc:
        raise ExtractionError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - damaged or unsupported structure
        raise ExtractionError(f"{filename}: could not validate PDF structure or read its pages.") from exc

    if len(source_pages) > config.MAX_RECORD_PAGES:
        raise ExtractionError("PDF exceeds the physical page limit before text extraction.")
    doc = ExtractedDocument(filename=filename, total_pages=len(source_pages))
    extracted_chars = 0
    for index, page in enumerate(source_pages, start=1):
        text = _page_text(page)
        if text:
            extracted_chars += len(text)
            if extracted_chars > 20 * 1024 * 1024:
                raise ExtractionError("Extracted PDF text exceeds the processing limit.")
            doc.pages.append(DocumentPage(filename, index, text))
        else:
            doc.unreadable_pages.append(index)

    if not doc.pages or not doc.char_count:
        raise ExtractionError(
            f"{filename}: no extractable text in {doc.total_pages:,} page(s). The PDF "
            "may be scanned/image-only. Use the operator's reviewed OCR workflow; "
            "the standalone scripts/ocr_records.py examples accept synthetic data only."
        )
    doc.pages = strip_running_headers(doc.pages)
    return doc


def _extract_docx(filename: str, data: bytes) -> ExtractedDocument:
    """Extract all supported Word stories as addressable source-part blocks.

    This is text extraction, not Word rendering. Refuse ambiguous revisions,
    drawings, equations and field-generated text instead of omitting them.
    """
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    stories: list[tuple[str, str]] = []
    text_chars = 0

    def read_story(name: str, root: Any) -> None:
        nonlocal text_chars
        unsupported = {namespace + n for n in ("drawing", "pict", "sym", "del", "ins", "moveFrom", "moveTo", "instrText", "fldSimple")}
        if any(n.tag in unsupported or n.tag.endswith("}oMath") or n.tag.endswith("}AlternateContent") for n in root.iter()):
            raise ExtractionError("DOCX contains visual, revised, or generated content. Export and review a complete text copy before analysis.")
        paragraphs: list[str] = []
        units = list(root) if root.tag in (namespace + "footnotes", namespace + "endnotes", namespace + "comments") else [root]
        for unit in units:
            if unit is not root:
                identity = unit.get(namespace + "id", "")
                author = unit.get(namespace + "author", "")
                date = unit.get(namespace + "date", "")
                paragraphs.append(f"[{unit.tag.split('}')[-1]} id={identity}; author={author}; date={date}]")
            for para in unit.iter(namespace + "p"):
                paragraphs.append(_word_paragraph_text(para, namespace))
        text = "\n\n".join(line for line in paragraphs if line)
        text_chars += len(text)
        if text_chars > 20 * 1024 * 1024:
            raise ExtractionError("DOCX extracted text exceeds the processing limit.")
        if text:
            stories.append((name, text))

    def _word_paragraph_text(para: Any, namespace: str) -> str:
        runs: list[str] = []
        for node in para.iter():
            if node.tag == namespace + "t":
                runs.append(node.text or "")
            elif node.tag == namespace + "tab":
                runs.append("\t")
            elif node.tag in (namespace + "br", namespace + "cr"):
                runs.append("\n")
            elif node.tag == namespace + "noBreakHyphen":
                runs.append("\u2011")
            elif node.tag == namespace + "softHyphen":
                runs.append("\u00ad")
            elif node.tag in (namespace + "footnoteReference", namespace + "endnoteReference", namespace + "commentReference"):
                runs.append(f" [{node.tag.split('}')[1]} {node.get(namespace + 'id', '')}] ")
        return "".join(runs).strip()
    max_member_bytes = config.DOCX_MAX_INTERNAL_FILE_BYTES
    max_total_bytes = config.DOCX_MAX_TOTAL_UNCOMPRESSED_BYTES
    max_member_count = config.DOCX_MAX_INTERNAL_FILE_COUNT
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            _validate_docx_uncompressed_sizes(
                filename,
                archive=archive,
                max_member_bytes=max_member_bytes,
                max_total_bytes=max_total_bytes,
                max_member_count=max_member_count,
                target_member_name="word/document.xml",
            )
            expanded = 0

            def read_part(name: str) -> bytes:
                nonlocal expanded
                body = _read_docx_member_limited(
                    filename, archive=archive, member_name=name,
                    max_member_bytes=max_member_bytes, max_total_bytes=max_total_bytes,
                    existing_total_bytes=expanded,
                )
                expanded += len(body)
                return body

            policy.validate_docx(archive, read_part, read_story)
    except policy.IngestionRefused as exc:
        raise ExtractionError(str(exc)) from exc
    except ExtractionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError("Could not read a structurally valid DOCX package.") from exc

    if not stories:
        raise ExtractionError(f"{filename}: DOCX contains no readable text.")
    pages: list[DocumentPage] = []
    for name, text in sorted(stories, key=lambda s: (s[0] != "word/document.xml", s[0])):
        for block in _blocks_from_text(filename, text):
            pages.append(DocumentPage(filename, len(pages) + 1, block.text, BLOCK, name))
    return ExtractedDocument(
        filename=filename,
        pages=pages,
        total_pages=len(pages),
        pagination=BLOCK,
    )


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


def iter_archive_members(
    filename: str, data: bytes, skipped: list[str]
) -> Iterator[tuple[str, bytes]]:
    """Validate metadata first, then expand one member with actual-byte quotas."""
    try:
        policy.validate_label(filename)
        policy.validate_signature(".zip", data)
        archive = zipfile.ZipFile(io.BytesIO(data))
    except policy.IngestionRefused as exc:
        raise ExtractionError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - named, not a bare zip traceback
        raise ExtractionError("Could not read a structurally valid ZIP archive.") from exc

    # The archive stays open for the whole function: reading a member after the
    # ``with`` block closes it raises "ZIP archive that was already closed", which
    # would leave every archive uploading as a list of skips.
    with archive:
        try:
            policy.validate_zip_entries(archive)
        except policy.IngestionRefused as exc:
            raise ExtractionError(str(exc)) from exc
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if len(infos) > config.ZIP_MAX_MEMBERS:
            raise ExtractionError(
                f"{filename}: archive holds {len(infos):,} files, over the "
                f"{config.ZIP_MAX_MEMBERS:,} supported. Extract it yourself and upload "
                "the records files directly."
            )

        label_prefix = Path(filename).stem or "archive"
        selected: list[tuple[str, str, zipfile.ZipInfo]] = []
        for info in infos:
            relative = info.filename
            base = Path(relative).name
            # Editor/OS bookkeeping that ends up in most zips; never record content.
            if not base or base.startswith(".") or "__MACOSX/" in f"{relative}/":
                continue
            label = f"{label_prefix}/{relative}"
            try:
                policy.validate_label(label)
            except policy.IngestionRefused as exc:
                raise ExtractionError(str(exc)) from exc
            if not relative.lower().endswith(SUPPORTED_EXTENSIONS):
                nested = relative.lower().endswith(ARCHIVE_EXTENSIONS)
                reason = "nested archive, not expanded" if nested else "unsupported file type"
                skipped.append(f"✖️ {filename}: skipped {relative} ({reason})")
                continue
            if info.file_size > config.ZIP_MAX_MEMBER_BYTES:
                skipped.append(
                    f"✖️ {filename}: skipped {relative} "
                    f"({info.file_size // 1_048_576} MB member exceeds the "
                    f"{config.ZIP_MAX_MEMBER_BYTES // 1_048_576} MB limit)"
                )
                continue
            ratio = info.file_size / max(info.compress_size, 1)
            if ratio > config.ZIP_MAX_COMPRESSION_RATIO:
                skipped.append(
                    f"✖️ {filename}: skipped {relative} (compression ratio {ratio:.0f}:1 is "
                    "not a plausible records file)"
                )
                continue
            selected.append((label, relative, info))

        # The total bound is checked before anything is read, so an archive that
        # expands past the limit never has its members held in memory at all.
        expands_to = sum(info.file_size for _, _, info in selected)
        if expands_to > config.ZIP_MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise ExtractionError(
                f"{filename}: archive expands to more than "
                f"{config.ZIP_MAX_TOTAL_UNCOMPRESSED_BYTES // 1_048_576} MB, which is "
                "more than one run can review. Upload the files it contains in smaller "
                "batches."
            )

        expanded = 0
        yielded = False
        for label, relative, info in selected:
            try:
                member = bytearray()
                with archive.open(info) as stream:
                    while True:
                        allowance = min(config.ZIP_MAX_MEMBER_BYTES - len(member),
                                        config.ZIP_MAX_TOTAL_UNCOMPRESSED_BYTES - expanded)
                        part = stream.read(min(65536, max(0, allowance) + 1))
                        if not part:
                            break
                        expanded += len(part)
                        if len(member) + len(part) > config.ZIP_MAX_MEMBER_BYTES or expanded > config.ZIP_MAX_TOTAL_UNCOMPRESSED_BYTES:
                            raise ExtractionError("Archive actual expansion exceeds its limit.")
                        member.extend(part)
                yielded = True
                yield label, bytes(member)
            except ExtractionError as exc:
                skipped.append(str(exc))
            except Exception as exc:  # noqa: BLE001 - one bad member must not lose the rest
                skipped.append(f"✖️ {filename}: could not read archive member {relative}.")
        if not yielded and not skipped:
            raise ExtractionError(f"{filename}: archive contains no record files.")


def archive_members(filename: str, data: bytes) -> tuple[list[tuple[str, bytes]], list[str]]:
    """Compatibility list API; ingestion uses the incremental member iterator."""
    skipped: list[str] = []
    return list(iter_archive_members(filename, data, skipped)), skipped


class InProcessExtractor:
    """The default ``RecordExtractor``: this module's reader, nothing external.

    The isolated parser uses this same passive-format policy. Pilot admission
    still requires OS isolation; these checks do not make in-process parsing safe.
    """

    def extract(self, label: str, data: bytes) -> tuple[list[ExtractedDocument], list[str]]:
        """Extract one file, expanding an archive into its members.

        Shared by the uploader and the local-folder reader so a zip behaves the
        same however it arrives.
        """
        if label.lower().endswith(ARCHIVE_EXTENSIONS):
            skipped: list[str] = []
            documents: list[ExtractedDocument] = []
            for member_label, member_bytes in iter_archive_members(label, data, skipped):
                try:
                    documents.append(extract_document(member_label, member_bytes))
                except ExtractionError as exc:
                    skipped.append(str(exc))
            return documents, skipped
        try:
            return [extract_document(label, data)], []
        except ExtractionError as exc:
            return [], [str(exc)]


# The extractor every ingestion path uses. Swapped once at startup by
# ``app.extractors.install_configured_extractor()`` (and by tests); ``None``
# always means the in-process reader above, never "no extractor".
_ACTIVE_EXTRACTOR: RecordExtractor = InProcessExtractor()


def active_extractor() -> RecordExtractor:
    """The extractor the uploader and the local-folder reader currently use."""
    return _ACTIVE_EXTRACTOR


def set_active_extractor(extractor: RecordExtractor | None) -> RecordExtractor:
    """Point every ingestion path at *extractor* (``None`` restores the default).

    Returns the extractor now active, so a caller can log which one it got.
    """
    global _ACTIVE_EXTRACTOR
    _ACTIVE_EXTRACTOR = extractor if extractor is not None else InProcessExtractor()
    return _ACTIVE_EXTRACTOR


def _expand_and_extract(
    label: str, data: bytes, extractor: RecordExtractor | None = None
) -> tuple[list[ExtractedDocument], list[str]]:
    """Extract one file through the active extractor (or *extractor* when given)."""
    try:
        return (extractor or _ACTIVE_EXTRACTOR).extract(label, data)
    except ExtractionError:
        # Per-file rejection is visible and does not abort other uploads. This
        # never substitutes an in-process parser for a refused protected parse.
        return [], [f"✖️ {label}: protected parsing was refused or is busy. Try this file again or ask the operator."]


def extract_uploaded_documents(
    files: "list[UploadedFile]",
) -> tuple[list[ExtractedDocument], list[str]]:
    """Extract uploaded files; returns ``(documents, skipped)``.

    ``skipped`` holds one message per file that could not be extracted (e.g.
    an image-only PDF), so callers can surface them as warnings instead of
    letting unreadable uploads silently vanish. A ``.zip`` upload contributes its
    members as separate documents, each extracted like a standalone upload.
    """
    documents: list[ExtractedDocument] = []
    skipped: list[str] = []
    for uploaded in files:
        extracted, problems = _expand_and_extract(uploaded.name, uploaded.getvalue())
        documents.extend(extracted)
        skipped.extend(problems)
    return documents, skipped


def _page_limit_reason() -> str:
    """The deployment-aware remedy sentence for a record-page-cap rejection.

    Reads the environment at call time so a test (or an operator flipping
    ``VA_LSE_ALLOW_LOCAL_PATHS``) changes the message without a restart.
    """
    import os

    if os.getenv("VA_LSE_ALLOW_LOCAL_PATHS", "").strip() == "1":
        return (
            "To change this limit, set VA_LSE_MAX_RECORD_PAGES in your .env and restart."
        )
    return (
        "This deployment is hosted, so the limit cannot be changed from the app. "
        "For a record set this size, download and run the app locally — the local "
        "tier keeps your records on your machine and has no fixed page cap "
        "(see the project README; scripts/batch_draft.py handles very large "
        "record sets there)."
    )


def page_limit_message(actual_pages: int, limit: int | None = None) -> str:
    """One shared rejection for the record-page cap (hosted-aware).

    The upload gate, the volume warning, and the run-time backstop all show
    this, so a hosted user is always pointed at the local tier instead of an
    operator environment variable they cannot touch.
    """
    limit_n = config.MAX_RECORD_PAGES if limit is None else limit
    return (
        f"Record set is {actual_pages:,} pages, over the limit of {limit_n:,} pages "
        f"for this deployment — the run will be refused. {_page_limit_reason()}"
    )


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
            if p.is_file() and p.suffix.lower() in (*SUPPORTED_EXTENSIONS, *ARCHIVE_EXTENSIONS)
        )
        label_for = lambda file: str(file.relative_to(root))  # noqa: E731
    if not files:
        raise ExtractionError(
            f"No supported record files (.pdf/.txt/.md/.docx/.zip) found in: {root}"
        )

    documents: list[ExtractedDocument] = []
    skipped: list[str] = []
    for file in files:
        extracted, problems = _expand_and_extract(label_for(file), file.read_bytes())
        documents.extend(extracted)
        skipped.extend(problems)
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
    # (filename, kind, page) for every page marker found inside ``text``. This is
    # what turns a chunk into a citation: without it the digest could only say
    # "chunk 7/40", which resolves to no page of any file, because chunks are cut
    # from the concatenation of every uploaded document.
    pages: tuple[tuple[str, str, int], ...] = ()

    @property
    def label(self) -> str:
        return f"chunk {self.index}/{self.total}"

    @property
    def source_hint(self) -> str:
        """Citation range for this chunk, e.g. ``clinic.pdf p.3-p.9``."""
        return chunk_source_hint(list(self.pages)) or self.label

    @property
    def page_numbers(self) -> tuple[int, ...]:
        return tuple(sorted({page for _, _, page in self.pages}))


def _chunk_pages(text: str) -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (match.group("file"), match.group("kind"), int(match.group("num")))
        for match in _PAGE_MARKER_RE.finditer(text)
    )


def chunk_page_labelled_text(text: str, max_chars: int = DEFAULT_CHUNK_CHARS) -> list[Chunk]:
    """Split page-labelled text into overlapping chunks at paragraph boundaries.

    ``max_chars`` is floored at :data:`MIN_CHUNK_CHARS` — at or below the overlap the
    cutter advances one character per chunk. :func:`iter_page_labelled_chunks`
    applies the same floor, so a budget too small to cut with behaves identically in
    both (their byte-equality for pre-cleaned pages is pinned by the property test in
    ``tests/test_core.py``).
    """
    max_chars = max(max_chars, MIN_CHUNK_CHARS)
    text = clean_text(text)
    if len(text) <= max_chars:
        return [Chunk(1, 1, text, _chunk_pages(text))]

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
    return [
        Chunk(i, total, c, _chunk_pages(c)) for i, c in enumerate(chunks, start=1)
    ]


# ------------------------------------------------------------ streaming chunks
class _StreamingCutter:
    """Cut the virtual page-labelled corpus into chunks without joining it.

    The joined implementation (:func:`chunk_page_labelled_text`) materializes
    the whole ``"\n\n".join(...)`` corpus and slices it; on a 5,000-page record
    set that join plus its slicing churn peaked at 37.7 MB traced — text that
    exists only to be cut. This cutter instead pulls one cleaned page part at a
    time from the caller's page iterator, keeps only the parts overlapping the
    current cutting window plus the overlap look-back, and accumulates just the
    resulting chunk texts. Peak memory: the caller's pages, the chunk texts (the
    caller retains those anyway), one ``max_chars`` window, and the largest
    single page — never the joined corpus, never a second cleaned copy of it.

    Reads see the same *virtual* text the joined implementation would clean:
    every extractor hands over already-cleaned page text, ``clean_text`` is
    idempotent, and its three operations are character-, line- and run-local —    so cleaning each ``marker\ntext`` part independently and joining the cleaned
    parts with ``\n\n`` equals cleaning the one joined string, because each
    part starts and ends with a non-whitespace character and each seam
    (``\n\n``) therefore cannot interact with any ``clean_text`` operation.
    The seams are part of this cutter's virtual text as their own deque
    entries. The property test in ``tests/test_core.py`` pins byte equality
    over randomized page sets.
    """

    def __init__(self, pages: Iterable[DocumentPage], max_chars: int) -> None:
        self._pages = iter(pages)
        self._max_chars = max_chars
        self._parts: deque[str] = deque()
        self._base = 0  # virtual index of self._parts[0][0]
        self._filled = 0  # virtual index one past the last pulled character
        self._exhausted = False

    def _append_part(self, page: DocumentPage) -> None:
        """Add one cleaned page part to the virtual text, with its seam.

        The joined implementation builds a ``"\n\n".join(...)``; here the seam
        is materialized as its own deque entry between parts, so reads and cut
        positions see the exact bytes the joined path would.
        """
        if self._filled > 0:
            self._parts.append("\n\n")
            self._filled += 2
        part = clean_text(f"{page.marker}\n{page.text}")
        self._parts.append(part)
        self._filled += len(part)

    def _pull_one(self) -> bool:
        """Pull the next non-blank page into the stream; ``False`` at the end.

        Whitespace-only pages (blank or image-only) are skipped: the joined
        path drops them in ``page_labelled_text`` before the join, so they
        contribute neither text nor a seam.
        """
        while True:
            page = next(self._pages, None)
            if page is None:
                self._exhausted = True
                return False
            if not page.text.strip():
                continue
            self._append_part(page)
            return True

    def _fill_to(self, length: int) -> None:
        """Pull cleaned parts until the virtual text reaches *length* or ends.

        When the text lands exactly on *length*, one more content part is
        pulled (or exhaustion confirmed), so the caller can always tell "the
        corpus ends exactly here" from "more remains" — the distinction the
        joined path gets for free by branching on ``end < len(text)`` of the
        full string. Without the lookahead, a cut landing exactly at
        end-of-corpus would be followed by an overlap-tail iteration and emit
        a chunk the joined implementation never produces.
        """
        while not self._exhausted and self._filled < length:
            self._pull_one()
        if not self._exhausted and self._filled == length:
            self._pull_one()

    def _read(self, start: int, end: int) -> str:
        """The virtual text from *start* (inclusive) to *end* (exclusive)."""
        if start < self._base:
            raise ValueError("read before the retained window")
        pieces: list[str] = []
        pos = self._base
        for part in self._parts:
            part_end = pos + len(part)
            if part_end > start:
                lo = max(start - pos, 0)
                hi = min(end - pos, len(part))
                if hi > lo:
                    pieces.append(part[lo:hi])
            if part_end >= end:
                break
            pos = part_end
        return "".join(pieces)

    def _drop_before(self, pos: int) -> None:
        """Release parts that end at or before *pos* (the cursor passed them)."""
        while self._parts and self._base + len(self._parts[0]) <= pos:
            self._base += len(self._parts.popleft())


def _iter_chunk_text(pages: Iterable[DocumentPage], max_chars: int) -> Iterator[str]:
    """Cut identical windows with the current window/page, not all chunk texts."""
    max_chars = max(max_chars, MIN_CHUNK_CHARS)
    cutter = _StreamingCutter(pages, max_chars)
    start = 0
    while True:
        # Fill one past the window: the original branches on ``end < len(text)``
        # of the joined string, so "exactly max_chars remain" must be
        # distinguishable from "more remains".
        cutter._fill_to(start + max_chars + 1)
        window = cutter._read(start, start + max_chars)
        if cutter._filled <= start + max_chars:
            # Stream ended inside this window: the whole rest is the final
            # chunk. An all-empty corpus is special-cased for parity with the
            # joined implementation, whose clean_text("") yields one empty
            # chunk (and therefore one — degenerate — digest call).
            if start == 0 and cutter._filled == 0:
                yield ""
            else:
                yield window
            break
        # Prefer cutting at a paragraph, then sentence, then hard cut.
        cut = window.rfind("\n\n")
        if cut < max_chars // 2:
            cut = window.rfind(". ")
        if cut < max_chars // 2:
            cut = max_chars
        end = start + cut + 1
        yield cutter._read(start, end)
        if cutter._exhausted and end >= cutter._filled:
            break  # the cut consumed the corpus: no overlap-tail iteration
        next_start = max(end - CHUNK_OVERLAP_CHARS, start + 1)
        cutter._drop_before(next_start)
        start = next_start


def iter_page_labelled_chunks(
    pages: Iterable[DocumentPage], max_chars: int = DEFAULT_CHUNK_CHARS
) -> Iterator[Chunk]:
    """Compatibility adapter for one-shot page iterators with an upfront total.

    Production medical review uses ChunkPlan: it counts in one bounded pass and
    yields in another, so no complete chunk-text list is retained.
    """
    pieces = list(_iter_chunk_text(pages, max_chars))
    for index, text in enumerate(pieces, start=1):
        yield Chunk(index, len(pieces), text, _chunk_pages(text))


class ChunkPlan:
    """Replayable, bounded-memory chunks over already retained source pages."""
    def __init__(self, pages: Iterable[DocumentPage], max_chars: int = DEFAULT_CHUNK_CHARS,
                 *, total: int | None = None, indices: frozenset[int] | None = None) -> None:
        self.pages = tuple(pages)
        self.max_chars = max_chars
        self.total = sum(1 for _ in _iter_chunk_text(self.pages, max_chars)) if total is None else total
        self.indices = indices

    def __len__(self) -> int:
        return self.total if self.indices is None else len(self.indices)

    def __iter__(self) -> Iterator[Chunk]:
        for index, text in enumerate(_iter_chunk_text(self.pages, self.max_chars), start=1):
            if self.indices is None or index in self.indices:
                yield Chunk(index, self.total, text, _chunk_pages(text))

    def select(self, indices: Iterable[int]) -> ChunkPlan:
        selected = frozenset(indices)
        if any(not 1 <= i <= self.total for i in selected):
            raise ValueError("Invalid chunk selection.")
        return ChunkPlan(self.pages, self.max_chars, total=self.total, indices=selected)


# ------------------------------------------------------- paragraph retrieval
@dataclass
class Paragraph:
    """One paragraph of record text with its page label, ready for scoring."""

    label: str
    text: str


# A retrieved block longer than this is split further, on single newlines.
PARAGRAPH_MAX_CHARS = config.PARAGRAPH_MAX_CHARS


def _split_oversized(block: str, limit: int = PARAGRAPH_MAX_CHARS) -> list[str]:
    """Break a block that has no blank lines inside it into line-grouped pieces.

    PDF text extraction often returns a whole page with no blank line, which would
    otherwise become a single retrieval unit for that page: one query score for
    hundreds of lines, so a paragraph buried in the middle can never outrank the
    page as a whole.
    """
    if len(block) <= limit:
        return [block]
    pieces: list[str] = []
    current: list[str] = []
    used = 0
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if current and used + len(stripped) > limit:
            pieces.append("\n".join(current))
            current, used = [], 0
        current.append(stripped)
        used += len(stripped) + 1
    if current:
        pieces.append("\n".join(current))
    return pieces


# Paragraph indexes are memoized per document: a large record set is searched once
# per claim batch, so the split work must not repeat per query.
#
# Bounded by *text*, not only by entry count. One 5,000-page document indexes to
# ~12 MB of paragraph strings (measured), so an entry-count-only cap of 64 retains
# hundreds of MB in a long-lived server process — the difference between a session
# that survives a week and one the OOM killer takes. Oldest entries are evicted
# once either bound is exceeded; oversized indexes are returned to the caller
# without being retained in this process-wide cache.
_PARAGRAPH_CACHE: OrderedDict[tuple[str, int, int], list[Paragraph]] = OrderedDict()
_PARAGRAPH_CACHE_LOCK = threading.Lock()
_PARAGRAPH_CACHE_MAX_ENTRIES = 64
_PARAGRAPH_CACHE_MAX_CHARS = 4_000_000


def _cache_get(key: tuple[str, int, int]) -> list[Paragraph] | None:
    """Cached index for *key*, refreshed as most-recently-used, or ``None``."""
    with _PARAGRAPH_CACHE_LOCK:
        cached = _PARAGRAPH_CACHE.get(key)
        if cached is not None:
            _PARAGRAPH_CACHE.move_to_end(key)
        return cached


def _cache_put(key: tuple[str, int, int], paragraphs: list[Paragraph]) -> None:
    """Store *key* and evict oldest entries until both bounds hold.

    The total is measured once and then decremented per eviction rather than
    recomputed inside the loop: a document large enough to matter is exactly the one
    whose paragraph count would make the recompute expensive, and eviction is the
    path it triggers.
    """
    with _PARAGRAPH_CACHE_LOCK:
        if key in _PARAGRAPH_CACHE:
            _PARAGRAPH_CACHE.move_to_end(key)
            return
        if sum(len(p.text) for p in paragraphs) > _PARAGRAPH_CACHE_MAX_CHARS:
            return
        _PARAGRAPH_CACHE[key] = paragraphs
        total = _cached_paragraph_chars()
        while len(_PARAGRAPH_CACHE) > 1 and (
            len(_PARAGRAPH_CACHE) > _PARAGRAPH_CACHE_MAX_ENTRIES
            or total > _PARAGRAPH_CACHE_MAX_CHARS
        ):
            _, evicted = _PARAGRAPH_CACHE.popitem(last=False)
            total -= sum(len(paragraph.text) for paragraph in evicted)


def _cached_paragraph_chars() -> int:
    """Characters of paragraph text the cache holds (caller holds the lock)."""
    return sum(
        sum(len(paragraph.text) for paragraph in paragraphs)
        for paragraphs in _PARAGRAPH_CACHE.values()
    )


def paragraph_index(doc: ExtractedDocument, min_chars: int = 1) -> list[Paragraph]:
    """Split a document into scannable paragraphs, cached per document.

    Large record sets (1,000+ pages) are searched once per claim batch, so the
    split/tokenize work is memoized instead of repeated for every query.
    """
    # Snapshot the inputs used by both hashing and extraction. Names and counts
    # are not identity; labels, page boundaries and ordering affect citations.
    pages = tuple((page.label, page.text) for page in doc.pages)
    fingerprint = hashlib.sha256()
    for page_input in pages:
        fingerprint.update(json.dumps(page_input, ensure_ascii=True).encode("ascii"))
    split_limit = PARAGRAPH_MAX_CHARS
    key = (fingerprint.hexdigest(), min_chars, split_limit)
    from . import pilot
    cache_allowed = not pilot.enabled()
    cached = _cache_get(key) if cache_allowed else None
    if cached is not None:
        return cached
    paragraphs: list[Paragraph] = []
    for label, text in pages:
        for block in re.split(r"\n{2,}", text):
            block = block.strip()
            if len(block) < min_chars:
                continue
            for piece in _split_oversized(block, limit=split_limit):
                paragraphs.append(Paragraph(label, piece))
    if cache_allowed:
        _cache_put(key, paragraphs)
    return paragraphs


# ------------------------------------------------------------ TF-IDF search
# Feature: Medical Record Search & Citation Index
_LABEL_PAGE_RE = re.compile(
    r"^(?P<filename>.*) (?P<prefix>[pb])\.(?P<page>\d+)$"
)
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
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
    import unicodedata
    return _TOKEN_RE.findall(unicodedata.normalize("NFC", text).casefold())


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
    from .text_fidelity import evidence_excerpt
    excerpt = evidence_excerpt(text, " ".join(tokens), max_chars)
    if not excerpt:
        return "[Matching source sentence exceeds the excerpt limit; open the complete source.]"
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
