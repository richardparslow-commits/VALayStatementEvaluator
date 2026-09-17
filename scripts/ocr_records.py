#!/usr/bin/env python3
"""OCR a scanned medical-records PDF so the app can actually read its pages.

Why this exists: a VA.gov export or a clinic's records are routinely a mix of
text pages and image-only scans, and an image page has no text to extract. The
app is honest about that — the results panel's "Record coverage" section names how
many pages had no extractable text — but it cannot read them. OCR has to happen on
the machine that holds the file, before the upload, which is what this script is
for.

Backends, in order of preference:

1. ``ocrmypdf`` (best): keeps the original pages and adds a text layer, so the
   output still looks like the records.  ``pip install -r requirements-local.txt``.
2. ``pdftoppm`` (Poppler) + ``tesseract``: rasterises each image-only page and
   OCRs it, then writes a text PDF containing the OCR text and the pages that
   already had text. Page images are not preserved in this mode.

Neither backend is a dependency of the app, and neither is used at runtime — the
app never shells out. If neither is installed the script says so and exits 2.

Exit codes: 0 = wrote an OCR'd copy, 1 = nothing to do, 2 = no OCR tooling,
3 = bad input or refused to overwrite.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.documents import _page_text  # noqa: E402  (documented reuse of the app's reader)

EXIT_OK = 0
EXIT_NOTHING_TO_DO = 1
EXIT_NO_TOOLING = 2
EXIT_BAD_INPUT = 3


def _out(text: str, *, err: bool = False) -> None:
    print(text, file=sys.stderr if err else sys.stdout)


def inspect_pdf(path: Path) -> tuple[int, list[int]]:
    """Return ``(page_count, image_only_page_numbers)`` for ``path``.

    Uses the same page reader as the app, so "image-only" here means exactly what
    the app will report after upload.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if getattr(reader, "is_encrypted", False):
        raise ValueError(
            f"{path} is password-protected. Remove the password before OCR (open it "
            "and re-save, or print to PDF)."
        )
    pages = list(reader.pages)
    image_only = [
        number
        for number, page in enumerate(pages, start=1)
        if not _page_text(page).strip()
    ]
    return len(pages), image_only


def find_backend() -> str | None:
    """Which OCR route is available on this machine (``"ocrmypdf"``/``"tesseract"``)."""
    if shutil.which("ocrmypdf"):
        return "ocrmypdf"
    if shutil.which("pdftoppm") and shutil.which("tesseract"):
        return "tesseract"
    return None


def default_output_path(source: Path) -> Path:
    """``records.pdf`` -> ``records.ocr.pdf`` (never the input file itself)."""
    return source.with_name(f"{source.stem}.ocr{source.suffix or '.pdf'}")


def _run(command: list[str], *, what: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[:500]
        raise RuntimeError(f"{what} failed (exit {result.returncode}): {detail}")


def ocr_with_ocrmypdf(source: Path, destination: Path) -> None:
    """Add a text layer with ocrmypdf, leaving pages that already have text alone."""
    _run(
        [
            "ocrmypdf",
            "--skip-text",  # keep existing text pages untouched, OCR only images
            "--rotate-pages",
            "--deskew",
            "--output-type",
            "pdf",
            str(source),
            str(destination),
        ],
        what="ocrmypdf",
    )


def ocr_with_tesseract(source: Path, destination: Path, *, dpi: int = 300) -> None:
    """OCR image-only pages with Poppler + Tesseract into a new text PDF.

    Page images are not preserved: the output is a text document that the app can
    read (and the original stays where it is). ``reportlab`` is already an app
    dependency, so this path adds no package of its own.
    """
    import tempfile

    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from pypdf import PdfReader

    reader = PdfReader(str(source))
    with tempfile.TemporaryDirectory() as workdir:
        work = Path(workdir)
        pdf = canvas.Canvas(str(destination), pagesize=letter)
        width, height = letter
        for number, page in enumerate(reader.pages, start=1):
            text = _page_text(page)
            if not text.strip():
                prefix = work / f"page-{number:05d}"
                _run(
                    [
                        "pdftoppm",
                        "-r",
                        str(dpi),
                        "-f",
                        str(number),
                        "-l",
                        str(number),
                        "-png",
                        str(source),
                        str(prefix),
                    ],
                    what=f"pdftoppm page {number}",
                )
                rasterised = sorted(work.glob(f"page-{number:05d}*.png"))
                if not rasterised:
                    _out(f"  page {number}: could not rasterise, left blank", err=True)
                    pdf.showPage()
                    continue
                result = subprocess.run(
                    ["tesseract", str(rasterised[0]), "stdout", "--psm", "6"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    _out(f"  page {number}: tesseract failed, left blank", err=True)
                    pdf.showPage()
                    continue
                text = result.stdout
            _write_text_page(pdf, text, height, width)
        pdf.save()


def _write_text_page(pdf: Any, text: str, height: float, width: float) -> None:
    """Write one page of text at a readable size, wrapping at the page width."""
    from reportlab.pdfbase.pdfmetrics import stringWidth

    font, size = "Helvetica", 9.0
    line_height = size * 1.35
    max_width = width - 96
    y = height - 72
    for raw_line in (text or "").splitlines() or [""]:
        words = raw_line.split()
        line = ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if stringWidth(candidate, font, size) <= max_width:
                line = candidate
            else:
                pdf.setFont(font, size)
                pdf.drawString(48, y, line)
                y -= line_height
                line = word
                if y < 72:  # ran off the page: this path is best-effort, not a renderer
                    break
        pdf.setFont(font, size)
        pdf.drawString(48, y, line)
        y -= line_height
        if y < 72:
            break
    pdf.showPage()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Add a text layer to an image-only medical-records PDF so the app can read it."
        )
    )
    parser.add_argument("pdf", type=Path, help="the scanned record PDF")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path (default: alongside the input as <name>.ocr.pdf)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="just report how many pages are image-only, then exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing output file",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="rasterisation DPI for the tesseract backend (default 300)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source: Path = args.pdf
    if not source.is_file():
        _out(f"✖ {source}: not a file", err=True)
        return EXIT_BAD_INPUT

    try:
        total, image_only = inspect_pdf(source)
    except Exception as exc:  # noqa: BLE001 - reported to the operator verbatim
        _out(f"✖ {source}: {exc}", err=True)
        return EXIT_BAD_INPUT

    _out(
        f"{source.name}: {total:,} page(s), {total - len(image_only):,} with text, "
        f"{len(image_only):,} image-only."
    )
    if not image_only:
        _out("✔ Nothing to OCR — every page already has extractable text.")
        return EXIT_NOTHING_TO_DO
    if args.report_only:
        return EXIT_OK

    backend = find_backend()
    if backend is None:
        _out(
            "✖ No OCR tooling found. Install one of:\n"
            "    pip install -r requirements-local.txt   # ocrmypdf (preferred)\n"
            "    macOS: brew install tesseract poppler   # or your package manager\n"
            f"  Pages needing OCR: {', '.join(str(p) for p in image_only[:20])}"
            f"{' …' if len(image_only) > 20 else ''}",
            err=True,
        )
        return EXIT_NO_TOOLING

    destination: Path = args.out or default_output_path(source)
    if destination.resolve() == source.resolve():
        _out("✖ Refusing to overwrite the input file — pass --out.", err=True)
        return EXIT_BAD_INPUT
    if destination.exists() and not args.force:
        _out(f"✖ {destination} exists. Pass --force to overwrite.", err=True)
        return EXIT_BAD_INPUT

    _out(f"Running {backend} → {destination} …")
    try:
        if backend == "ocrmypdf":
            ocr_with_ocrmypdf(source, destination)
        else:
            ocr_with_tesseract(source, destination, dpi=args.dpi)
    except RuntimeError as exc:
        _out(f"✖ {exc}", err=True)
        return EXIT_BAD_INPUT

    try:
        total_after, remaining = inspect_pdf(destination)
    except Exception as exc:  # noqa: BLE001 - the OCR output may not even be a PDF
        _out(f"✖ wrote {destination} but could not re-read it: {exc}", err=True)
        return EXIT_BAD_INPUT
    _out(
        f"✔ Wrote {destination} ({destination.stat().st_size:,} bytes). "
        f"{total_after - len(remaining):,} of {total_after:,} page(s) now have text."
    )
    if remaining:
        _out(
            f"  {len(remaining)} page(s) are still image-only — upload anyway and check "
            "the app's 'Record coverage' panel, which will name them."
        )
    _out("  Upload this file as a record source; keep the original unchanged.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
