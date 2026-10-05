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

# This standalone tool accepts only explicitly declared synthetic inputs.
# Real records require a separately accepted isolation/retention workflow.

import argparse
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence
from collections.abc import Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.documents import _page_text  # noqa: E402  (documented reuse of the app's reader)
from app.synthetic_tools import require_synthetic_tool  # noqa: E402

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
    require_synthetic_tool()
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
    from app.child_process import run_bounded
    try:
        result = run_bounded(command)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{what} exceeded the processing deadline.") from None
    if result.returncode != 0:
        raise RuntimeError(f"{what} failed (exit {result.returncode}); raw tool output was suppressed.")


def ocr_with_ocrmypdf(source: Path, destination: Path) -> None:
    """Add a text layer with ocrmypdf, leaving pages that already have text alone."""
    with _atomic_output(source, destination) as temporary:
        _run(
        [
            "ocrmypdf",
            "--skip-text",  # keep existing text pages untouched, OCR only images
            "--rotate-pages",
            "--deskew",
            "--output-type",
            "pdf",
            str(source),
            str(temporary),
        ],
        what="ocrmypdf",
        )


@contextmanager
def _atomic_output(source: Path, destination: Path) -> Iterator[Path]:
    """Publish only a fully completed OCR artifact; preserve existing output on failure."""
    import os
    import tempfile
    require_synthetic_tool()
    if destination.is_symlink():
        raise ValueError("Refusing a symbolic OCR output destination.")
    if source.resolve() == destination.resolve() or (destination.exists() and source.samefile(destination)):
        raise ValueError("OCR output must be a separate file from the original source.")
    with tempfile.TemporaryDirectory(prefix=".ocr-", dir=str(destination.parent)) as directory:
        temporary = Path(directory) / "completed.pdf"
        yield temporary
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise ValueError("OCR did not produce a complete output file.")
        before, _ = inspect_pdf(source)
        after, _ = inspect_pdf(temporary)
        if after != before:
            raise ValueError("OCR changed source page coverage; no partial output was published.")
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)


def _ocr_page(source: Path, number: int, dpi: int, workdir: Path) -> str | None:
    """Rasterize and OCR one image-only page; ``None`` on failure (reported)."""
    prefix = workdir / f"page-{number:05d}"
    try:
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
    except RuntimeError:
        _out(f"  page {number}: could not rasterise, left blank", err=True)
        return None
    rasterised = sorted(workdir.glob(f"page-{number:05d}*.png"))
    if not rasterised:
        _out(f"  page {number}: could not rasterise, left blank", err=True)
        return None
    from app.child_process import run_bounded
    result = run_bounded(["tesseract", str(rasterised[0]), "stdout", "--psm", "6"])
    if result.returncode != 0:
        _out(f"  page {number}: tesseract failed, left blank", err=True)
        return None
    return result.stdout


def _ocr_pages_parallel(
    source: Path, numbers: list[int], *, dpi: int, workdir: Path, jobs: int
) -> dict[int, str | None]:
    """OCR *numbers* concurrently; each worker runs real subprocesses, so the
    GIL never binds this — the speedup tracks the worker count. Failures are
    per-page and reported, never fatal: a page that fails alone is left blank
    rather than costing its siblings.
    """
    from concurrent.futures import ThreadPoolExecutor

    results: dict[int, str | None] = {}
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        from app.pipeline_guard import bounded_pipeline_futures
        for future, number in bounded_pipeline_futures(
            pool, lambda number: _ocr_page(source, number, dpi, workdir),
            numbers, max(1, jobs * 2),
        ):
            try:
                results[number] = future.result()
            except Exception:  # noqa: BLE001 - one page must not stop the run
                _out(f"  page {number}: OCR failed — left blank", err=True)
                results[number] = None
    return results


def ocr_with_tesseract(
    source: Path, destination: Path, *, dpi: int = 300, jobs: int = 0
) -> None:
    """OCR image-only pages with Poppler + Tesseract into a new text PDF.

    Page images are not preserved: the output is a text document that the app can
    read (and the original stays where it is). ``reportlab`` is already an app
    dependency, so this path adds no package of its own.

    Image-only pages are OCRd in parallel — a 1,000-page scan is hours of
    sequential subprocess calls otherwise. ``jobs`` caps the worker count;
    ``0`` (default) picks one worker per CPU up to 8, and ``1`` restores the
    old sequential behavior.
    """
    import os
    import tempfile

    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    total, image_only = inspect_pdf(source)
    if jobs <= 0:
        jobs = min(8, os.cpu_count() or 4)
    jobs = max(1, min(jobs, max(1, len(image_only))))
    with _atomic_output(source, destination) as temporary, tempfile.TemporaryDirectory() as workdir:
        work = Path(workdir)
        ocr_results = _ocr_pages_parallel(
            source, image_only, dpi=dpi, workdir=work, jobs=jobs
        )
        from pypdf import PdfReader

        reader = PdfReader(str(source))
        pdf = canvas.Canvas(str(temporary), pagesize=letter)
        width, height = letter
        expected: list[str] = []
        for number in range(1, total + 1):
            text = ocr_results.get(number)
            if text is None:
                text = _page_text(reader.pages[number - 1])
            expected.append(text)
            _write_text_page(pdf, text, height, width)
        pdf.save()
        rendered = PdfReader(str(temporary), strict=True)
        if len(rendered.pages) != len(expected) or any(
            " ".join(_page_text(page).split()) != " ".join(text.split())
            for page, text in zip(rendered.pages, expected)
        ):
            raise ValueError("OCR transcript could not be extracted without text changes; no partial output was published.")


def _write_text_page(pdf: Any, text: str, height: float, width: float) -> None:
    """Write every word or refuse before drawing; never hide overflow or lost glyphs."""
    from reportlab.pdfbase.pdfmetrics import stringWidth

    font, size = "Helvetica", 9.0
    line_height = size * 1.35
    max_width = width - 96
    y = height - 72
    try:
        text.encode("cp1252", errors="strict")
    except UnicodeError as exc:
        raise ValueError("OCR transcript contains characters this PDF font cannot preserve. Use a reviewed OCR backend that retains the original images and Unicode text.") from exc
    lines: list[str] = []
    for raw_line in (text or "").splitlines() or [""]:
        words = raw_line.split()
        line = ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if stringWidth(candidate, font, size) <= max_width:
                line = candidate
            else:
                if stringWidth(word, font, size) > max_width:
                    raise ValueError("OCR transcript contains a word wider than the page; no partial output was published.")
                lines.append(line)
                line = word
        lines.append(line)
    if (len(lines) - 1) * line_height > height - 144:
        raise ValueError("OCR transcript exceeds one source page; no text was dropped. Use an OCR backend that retains original images and aligned text.")
    for line in lines:
        pdf.setFont(font, size)
        pdf.drawString(48, y, line)
        y -= line_height
    pdf.showPage()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Add a text layer to an image-only medical-records PDF so the app can read it."
        )
    )
    parser.add_argument("pdf", type=Path, help="the scanned record PDF")
    parser.add_argument("--data-class", required=True, choices=("synthetic", "sensitive"),
                        help="declare the input class; sensitive records require a separately approved workflow")
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
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help=(
            "parallel OCR workers for the tesseract backend "
            "(0 = one per CPU up to 8; 1 = sequential)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.data_class != "synthetic":
        _out("Sensitive records require a separately approved isolated workflow.", err=True)
        return EXIT_BAD_INPUT
    try:
        require_synthetic_tool()
    except ValueError as exc:
        _out(str(exc), err=True)
        return EXIT_BAD_INPUT
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
            ocr_with_tesseract(source, destination, dpi=args.dpi, jobs=args.jobs)
    except (RuntimeError, ValueError) as exc:
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
