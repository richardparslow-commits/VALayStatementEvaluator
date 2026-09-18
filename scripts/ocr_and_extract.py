#!/usr/bin/env python3
"""OCR a record bundle inside the sandbox, then extract it with the app's own reader.

Why this exists: the app has no OCR dependency and never shells out (see
``scripts/ocr_records.py``), so a page that is a scan has no text to extract — the
results panel counts it and asks the operator to OCR the file and upload it again.
That is the right shape for a deployment that cannot have Tesseract, but it means
the *work* of reading a records bundle happens before the upload, on whichever
machine holds the file. VA.gov exports and clinic records are routinely image-only
pages, so this is not a corner case.

The sandbox is the machine that can do it: ``tesseract-ocr``, ``poppler-utils``,
``ghostscript`` and ``qpdf`` are installed in the image's sandbox stage (and in no
other image — the deployment must not carry them), and a microVM is the right
place to run a parser and an OCR engine over records that arrived from somewhere
else. This script is the entrypoint for that box:

    python scripts/ocr_and_extract.py /work/records --out /work/bundle.json

What it does per file: if the file is a PDF with pages that yield no text, it OCRs
those pages (ocrmypdf, or Poppler + Tesseract when ocrmypdf is absent) into a
working directory, then extracts the OCR'd copy **under the original file's name**
— citations have to point at the record the user has, never at ``.ocr.pdf`` — and
reports how many pages were unreadable before and after. Files with text are
extracted directly. ``.zip`` bundles are expanded the same way the app expands an
uploaded archive (``app.documents.archive_members``), so a bundle and an upload of
the same records produce the same documents.

Two deliberate decisions worth knowing:

* **The app's reader does the extraction.** ``app.documents.extract_document`` is
  still the only PDF/DOCX parser in this project, so a page's text in the box is
  the page's text after upload — same page markers, same running-header stripping,
  same chunk boundaries. A second reader in another language would drift.
* **The output is the queue's document JSON.** ``app.job_payload.documents_to_json``
  is what a worker pod already reads, so the box's answer can be handed to the app
  (or to ``documents_from_json``) without a second schema to keep in sync. The
  ``files`` block around it is evidence — page counts, page numbers still
  unreadable, seconds spent — the parts of the run the app reports on.

It is honest about what it could not do: with no OCR tooling installed, the text
pages are still extracted, every scan is named, and the exit status says so rather
than reporting success for a bundle the app will read as empty.

Exit codes: 0 = extracted (OCR done where it was needed), 1 = no supported record
files found, 2 = scans were found but no OCR tooling is installed, 3 = bad input
(missing path, unwritable output, or nothing extractable at all).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import config  # noqa: E402
from app.documents import (  # noqa: E402
    ARCHIVE_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    ExtractedDocument,
    ExtractionError,
    archive_members,
    extract_document,
)
from app.job_payload import documents_to_json  # noqa: E402
from scripts import ocr_records  # noqa: E402

EXIT_OK = 0
EXIT_NOTHING_TO_DO = 1
EXIT_NO_TOOLING = 2
EXIT_BAD_INPUT = 3

#: Where the OCR copies are written, relative to a bundle directory. Named rather
#: than random so a second run reuses them instead of re-OCR-ing a large bundle,
#: and skipped during discovery so the copies are never read as records.
WORK_DIR_NAME = ".ocr-work"

RECORD_SUFFIXES = (*SUPPORTED_EXTENSIONS, *ARCHIVE_EXTENSIONS)


def _out(text: str, *, err: bool = False) -> None:
    print(text, file=sys.stderr if err else sys.stdout)


def _fail(label: str, message: str) -> str:
    """One skipped-file message, in the shape the app's uploader uses."""
    return f"✖️ {label}: {message}"


# ------------------------------------------------------------------- discovery
def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def discover(paths: Sequence[Path], *, work_dir: Path) -> list[Path]:
    """Every record file under *paths*, in a stable order, minus our own output.

    A directory is walked recursively and only recognised suffixes are taken: an
    operator's folder holds a README, a screenshot of the portal, a consent form
    as ``.docx`` — picking those up as records would put non-records in the run.
    """
    found: list[Path] = []
    resolved_work = work_dir.resolve()
    for entry in paths:
        if entry.is_file():
            found.append(entry)
            continue
        for candidate in sorted(entry.rglob("*")):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in RECORD_SUFFIXES:
                continue
            if _is_hidden(candidate.relative_to(entry)):
                continue
            if resolved_work in candidate.resolve().parents:
                continue  # our own OCR copies
            found.append(candidate)
    # Deduplicate while keeping the first occurrence: the same file passed twice
    # (a file inside a directory that was also named) must not be counted twice.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in found:
        key = path.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def default_work_dir(paths: Sequence[Path]) -> Path:
    """Where OCR copies go: beside the bundle when it can be written to."""
    first = next((p for p in paths if p.exists()), PROJECT_ROOT)
    base = first if first.is_dir() else first.parent
    target = (base / WORK_DIR_NAME) if base.is_dir() else (base.parent / WORK_DIR_NAME)
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".writable"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return target
    except OSError as exc:  # pragma: no cover - a read-only mount
        fallback = Path(tempfile.mkdtemp(prefix="ocr-work-"))
        _out(
            f"! {target} is not writable ({exc}); using {fallback} instead",
            err=True,
        )
        return fallback


# ------------------------------------------------------------------ per file
def _label_for(path: Path, roots: Sequence[Path]) -> str:
    """The document label: relative to its bundle, so two folders cannot collide."""
    for root in roots:
        if root.is_dir():
            try:
                return path.relative_to(root).as_posix()
            except ValueError:
                continue
    return path.name


def _safe_stem(label: str) -> str:
    """A filename stem for the OCR copy: the label, without anything path-like.

    The folder is kept (``a/scan.pdf`` → ``a__scan``) because two bundles can hold
    the same file name, and the extension is dropped because these copies are
    always PDFs — the caller appends it.
    """
    name = label.replace("\\", "__").replace("/", "__")
    return name[:-4] if name.lower().endswith(".pdf") else name


def prepare(
    label: str,
    data: bytes,
    *,
    work_dir: Path,
    dpi: int,
    use_ocr: bool,
) -> tuple[bytes, dict[str, Any]]:
    """Return ``(bytes to extract, what happened)`` for one record file.

    For a PDF with unreadable pages this OCRs a copy and returns *that* — the
    label stays the original file's, because that is the name the app's citations
    will carry and the name the user will look for. For anything else, or for a
    PDF that is fully readable, the bytes come back untouched.
    """
    report: dict[str, Any] = {
        "label": label,
        "bytes": len(data),
        "pages": 0,
        "image_only_before": [],
        "image_only_after": [],
        "ocr": "not_needed",
        "ocr_output": None,
        "ocr_seconds": 0.0,
        "notes": [],
    }
    if not label.lower().endswith(".pdf"):
        return data, report

    work_dir.mkdir(parents=True, exist_ok=True)
    source = work_dir / f"{_safe_stem(label)}.pdf"
    source.write_bytes(data)
    try:
        total, image_only = ocr_records.inspect_pdf(source)
    except Exception as exc:  # noqa: BLE001 - reported verbatim, like the sibling script
        report["notes"].append(str(exc))
        return data, report
    report["pages"] = total
    report["image_only_before"] = list(image_only)

    if not image_only:
        return data, report
    if not use_ocr:
        report["ocr"] = "skipped"
        return data, report

    backend = ocr_records.find_backend()
    if backend is None:
        report["ocr"] = "unavailable"
        report["notes"].append(
            "no OCR tooling on PATH; pages "
            f"{', '.join(str(p) for p in image_only[:20])}"
            f"{' …' if len(image_only) > 20 else ''} have no text to extract"
        )
        return data, report

    destination = work_dir / f"{_safe_stem(label)}.ocr.pdf"
    started = time.monotonic()
    try:
        if backend == "ocrmypdf":
            ocr_records.ocr_with_ocrmypdf(source, destination)
        else:
            ocr_records.ocr_with_tesseract(source, destination, dpi=dpi)
    except RuntimeError as exc:
        report["ocr"] = f"{backend}-failed"
        report["notes"].append(str(exc))
        return data, report
    report["ocr_seconds"] = round(time.monotonic() - started, 3)
    report["ocr"] = backend
    report["ocr_output"] = str(destination)

    try:
        after_total, remaining = ocr_records.inspect_pdf(destination)
    except Exception as exc:  # noqa: BLE001 - the engine produced something unreadable
        report["notes"].append(f"could not re-read the OCR output: {exc}")
        return data, report
    report["pages"] = after_total
    report["image_only_after"] = list(remaining)
    if remaining:
        report["notes"].append(
            f"{len(remaining)} page(s) are still image-only after OCR"
        )
    return destination.read_bytes(), report


# ------------------------------------------------------------------ the bundle
def process(
    files: Sequence[Path],
    *,
    roots: Sequence[Path],
    work_dir: Path,
    dpi: int = 300,
    use_ocr: bool = True,
    on_file: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """OCR and extract *files*, returning the report the CLI writes as JSON."""
    documents: list[ExtractedDocument] = []
    skipped: list[str] = []
    reports: list[dict[str, Any]] = []
    extract_seconds = 0.0

    def handle(label: str, data: bytes) -> None:
        nonlocal extract_seconds
        prepared, report = prepare(
            label, data, work_dir=work_dir, dpi=dpi, use_ocr=use_ocr
        )
        started = time.monotonic()
        try:
            document = extract_document(label, prepared)
        except ExtractionError as exc:
            skipped.append(str(exc))
            report["extracted"] = False
        else:
            documents.append(document)
            report["extracted"] = True
            report["pages"] = document.source_page_count
            report["unreadable_pages"] = list(document.unreadable_pages)
        extract_seconds += time.monotonic() - started
        reports.append(report)
        if on_file is not None:
            on_file(report)

    for path in files:
        label = _label_for(path, roots)
        data = path.read_bytes()
        if path.suffix.lower() in ARCHIVE_EXTENSIONS:
            # An archive's members are records too, and the app expands them the
            # same way on upload — so a bundle and an upload agree.
            try:
                members, archive_skips = archive_members(label, data)
            except ExtractionError as exc:
                skipped.append(str(exc))
                continue
            skipped.extend(archive_skips)
            for member_label, member_data in members:
                handle(member_label, member_data)
        else:
            handle(label, data)

    pages = sum(doc.source_page_count for doc in documents)
    unreadable = sum(len(doc.unreadable_pages) for doc in documents)
    return {
        "documents": documents_to_json(documents),
        "skipped": skipped,
        "files": reports,
        "totals": {
            "files": len(files),
            "documents": len(documents),
            "pages": pages,
            "pages_unreadable": unreadable,
            "chars": sum(doc.char_count for doc in documents),
            "ocr_seconds": round(sum(r["ocr_seconds"] for r in reports), 3),
            "extract_seconds": round(extract_seconds, 3),
            "over_page_cap": pages > config.MAX_RECORD_PAGES,
            "page_cap": config.MAX_RECORD_PAGES,
        },
    }


# ----------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "OCR a record bundle in the sandbox and extract it with the app's own "
            "reader, emitting the queue's document JSON."
        )
    )
    parser.add_argument(
        "bundle",
        nargs="+",
        type=Path,
        help="record files, or the directory holding them (walked recursively)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"where to write the JSON (default: <work-dir>/bundle.json)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help=f"where OCR copies are written (default: <bundle>/{WORK_DIR_NAME})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="rasterisation DPI for the tesseract backend (default 300)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="say which pages need OCR and write nothing",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="extract without OCR, so the result is what the app would see today",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing --out file",
    )
    return parser


def _summarize(report: dict[str, Any], destination: Path | None) -> None:
    totals = report["totals"]
    for entry in report["files"]:
        before = len(entry["image_only_before"])
        after = len(entry["image_only_after"])
        if entry["ocr"] == "not_needed":
            detail = f"{entry['pages']:,} page(s), all readable"
        elif entry["ocr"] == "skipped":
            detail = (
                f"{entry['pages']:,} page(s), {before:,} image-only — OCR not attempted"
            )
        elif entry["ocr"] == "unavailable":
            detail = (
                f"{entry['pages']:,} page(s), {before:,} image-only — no OCR tooling"
            )
        else:
            detail = (
                f"{entry['pages']:,} page(s), {before:,} image-only → {after:,} "
                f"after OCR ({entry['ocr']}, {entry['ocr_seconds']}s)"
            )
        _out(f"  {entry['label']}: {detail}")
        for note in entry["notes"]:
            _out(f"      {note}", err=True)
    for message in report["skipped"]:
        _out(f"  {message}", err=True)
    _out("")
    _out(
        f"{totals['documents']:,} document(s), {totals['pages']:,} page(s), "
        f"{totals['pages_unreadable']:,} still without text, "
        f"{totals['chars']:,} characters."
    )
    if totals["over_page_cap"]:
        _out(
            f"! {totals['pages']:,} pages is over VA_LSE_MAX_RECORD_PAGES "
            f"({totals['page_cap']:,}) — the app will refuse a run with these records.",
            err=True,
        )
    if destination is not None:
        _out(f"JSON: {destination}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle: list[Path] = list(args.bundle)
    missing = [str(path) for path in bundle if not path.exists()]
    if missing:
        _out(f"✖ not found: {', '.join(missing)}", err=True)
        return EXIT_BAD_INPUT

    work_dir: Path = args.work_dir or default_work_dir(bundle)
    work_dir.mkdir(parents=True, exist_ok=True)
    files = discover(bundle, work_dir=work_dir)
    if not files:
        _out(
            "✖ No supported record files found (.pdf/.txt/.md/.docx/.zip). "
            "Nothing to do.",
            err=True,
        )
        return EXIT_NOTHING_TO_DO

    destination: Path | None = None if args.report_only else (args.out or work_dir / "bundle.json")
    if destination is not None and destination.exists() and not args.force:
        _out(f"✖ {destination} exists. Pass --force to overwrite.", err=True)
        return EXIT_BAD_INPUT

    _out(
        f"{len(files)} record file(s) from {', '.join(str(p) for p in bundle)}; "
        f"{'inspecting only' if args.report_only else 'OCR + extract'}"
        + (" (OCR off)" if args.no_ocr else "")
    )
    report = process(
        files,
        roots=bundle,
        work_dir=work_dir,
        dpi=args.dpi,
        use_ocr=not (args.no_ocr or args.report_only),
        on_file=None,
    )
    _summarize(report, destination)

    if destination is not None:
        payload = {"version": 1, "bundle": [str(p) for p in bundle], **report}
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.report_only:
        return EXIT_OK

    totals = report["totals"]
    # The missing-tooling answer comes first on purpose: for an all-scan bundle it
    # is the only actionable one, and "nothing extractable" would send the operator
    # looking at their files instead of at the image they are running in.
    if any(entry["ocr"] == "unavailable" for entry in report["files"]):
        _out(
            "✖ Scans are present and no OCR tooling is installed, so those pages "
            "have no text to extract. Install ocrmypdf (or poppler + tesseract) and "
            "re-run — the sandbox image installs both (see the Dockerfile's sandbox "
            f"stage). Pages still unreadable: {totals['pages_unreadable']:,}.",
            err=True,
        )
        return EXIT_NO_TOOLING
    if totals["documents"] == 0:
        _out("✖ Nothing extractable — every file was skipped.", err=True)
        return EXIT_BAD_INPUT
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
