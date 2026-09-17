#!/usr/bin/env python3
"""
Split large medical record files into smaller chunks for processing.

This script helps work around LLM endpoint issues by breaking large record sets
into smaller files that can be processed independently. This is useful when:
- The circuit breaker keeps opening due to sustained endpoint issues
- You want to reduce the blast radius of failures on large record sets
- Processing very large bundles (1000+ pages) is timing out or failing

Usage:
    python scripts/split_records.py input.pdf [--output-dir ./split_records] [--max-pages 100]

The output files are named sequentially (records_part_01.pdf, records_part_02.pdf, etc.)
and can be uploaded to the app individually or in smaller batches.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import BinaryIO

# PDF handling - try pypdf first (lighter), fall back to PyPDF2 or pikepdf
try:
    from pypdf import PdfReader, PdfWriter
    PDF_LIB = "pypdf"
except ImportError:
    try:
        from PyPDF2 import PdfReader, PdfWriter
        PDF_LIB = "PyPDF2"
    except ImportError:
        try:
            import pikepdf
            PDF_LIB = "pikepdf"
        except ImportError:
            PDF_LIB = None


def split_pdf(input_path: Path, output_dir: Path, max_pages: int) -> list[Path]:
    """Split a PDF into chunks of at most `max_pages` pages each."""
    if PDF_LIB is None:
        raise RuntimeError(
            "No PDF library available. Install one of: pypdf, PyPDF2, or pikepdf"
        )

    if PDF_LIB == "pikepdf":
        return _split_pikepdf(input_path, output_dir, max_pages)
    else:
        return _split_pypdf(input_path, output_dir, max_pages)


def _split_pypdf(input_path: Path, output_dir: Path, max_pages: int) -> list[Path]:
    """Split using pypdf or PyPDF2 (same API)."""
    reader = PdfReader(str(input_path))
    total_pages = len(reader.pages)
    output_files = []

    part_num = 0
    for start in range(0, total_pages, max_pages):
        part_num += 1
        writer = PdfWriter()
        end = min(start + max_pages, total_pages)

        for i in range(start, end):
            writer.add_page(reader.pages[i])

        # Preserve metadata from original if available
        if hasattr(reader, 'metadata') and reader.metadata:
            writer.metadata = reader.metadata

        output_path = output_dir / f"records_part_{part_num:02d}.pdf"
        with open(output_path, "wb") as f:
            writer.write(f)

        output_files.append(output_path)
        print(f"  Created: {output_path.name} ({end - start} pages)")

    return output_files


def _split_pikepdf(input_path: Path, output_dir: Path, max_pages: int) -> list[Path]:
    """Split using pikepdf (better for complex PDFs)."""
    import pikepdf

    pdf = pikepdf.Pdf.open(str(input_path))
    total_pages = len(pdf.pages)
    output_files = []

    part_num = 0
    for start in range(0, total_pages, max_pages):
        part_num += 1
        end = min(start + max_pages, total_pages)

        new_pdf = pikepdf.Pdf.new()
        new_pdf.pages.extend(pdf.pages[start:end])

        # Copy metadata
        if pdf.metadata:
            new_pdf.metadata.update(pdf.metadata)

        output_path = output_dir / f"records_part_{part_num:02d}.pdf"
        new_pdf.save(str(output_path))
        new_pdf.close()

        output_files.append(output_path)
        print(f"  Created: {output_path.name} ({end - start} pages)")

    pdf.close()
    return output_files


def split_text_file(input_path: Path, output_dir: Path, max_lines: int) -> list[Path]:
    """Split a text file into chunks of at most `max_lines` lines each."""
    output_files = []
    part_num = 0

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_lines = len(lines)

    for start in range(0, total_lines, max_lines):
        part_num += 1
        end = min(start + max_lines, total_lines)
        chunk_lines = lines[start:end]

        output_path = output_dir / f"records_part_{part_num:02d}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(chunk_lines)

        output_files.append(output_path)
        print(f"  Created: {output_path.name} ({end - start} lines)")

    return output_files


def split_docx(input_path: Path, output_dir: Path, max_paragraphs: int) -> list[Path]:
    """Split a DOCX file by paragraph count (approximate)."""
    try:
        from docx import Document
    except ImportError:
        raise RuntimeError("python-docx required for DOCX splitting: pip install python-docx")

    doc = Document(str(input_path))
    total_paragraphs = len(doc.paragraphs)
    output_files = []

    part_num = 0
    for start in range(0, total_paragraphs, max_paragraphs):
        part_num += 1
        end = min(start + max_paragraphs, total_paragraphs)

        new_doc = Document()
        # Copy styles and sections from original
        for section in doc.sections:
            new_doc.add_section()

        for i in range(start, end):
            new_doc.add_paragraph(doc.paragraphs[i].text, style=doc.paragraphs[i].style.name if doc.paragraphs[i].style else None)

        output_path = output_dir / f"records_part_{part_num:02d}.docx"
        new_doc.save(str(output_path))

        output_files.append(output_path)
        print(f"  Created: {output_path.name} (~{end - start} paragraphs)")

    return output_files


def get_file_info(path: Path) -> dict:
    """Get information about a file for reporting."""
    stat = path.stat()
    return {
        "name": path.name,
        "size_bytes": stat.st_size,
        "size_human": _human_size(stat.st_size),
    }


def _human_size(bytes_val: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split large medical record files into smaller chunks for processing."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input file to split (PDF, TXT, or DOCX)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for split files (default: ./split_<filename>)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=100,
        help="Max pages per output file for PDFs (default: 100)",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=5000,
        help="Max lines per output file for TXT files (default: 5000)",
    )
    parser.add_argument(
        "--max-paragraphs",
        type=int,
        default=200,
        help="Max paragraphs per output file for DOCX files (default: 200)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually splitting",
    )

    args = parser.parse_args()

    input_path = args.input.resolve()

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        output_dir = input_path.parent / f"split_{input_path.stem}"

    # Get file info
    file_info = get_file_info(input_path)
    print(f"\nInput file: {file_info['name']}")
    print(f"Size: {file_info['size_human']}")

    # Determine file type and appropriate split strategy
    suffix = input_path.suffix.lower()

    if args.dry_run:
        print("\n[DRY RUN] Would split as follows:")
    else:
        print("\nSplitting...")

    output_files = []

    if suffix == ".pdf":
        total_pages = _count_pdf_pages(input_path)
        print(f"Pages: {total_pages}")
        num_parts = (total_pages + args.max_pages - 1) // args.max_pages
        print(f"Will create {num_parts} file(s) with max {args.max_pages} pages each")

        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_files = split_pdf(input_path, output_dir, args.max_pages)

    elif suffix == ".txt" or suffix == ".md":
        with open(input_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
        print(f"Lines: {total_lines}")
        num_parts = (total_lines + args.max_lines - 1) // args.max_lines
        print(f"Will create {num_parts} file(s) with max {args.max_lines} lines each")

        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_files = split_text_file(input_path, output_dir, args.max_lines)

    elif suffix == ".docx":
        try:
            from docx import Document
            doc = Document(str(input_path))
            total_paragraphs = len(doc.paragraphs)
            print(f"Paragraphs: {total_paragraphs}")
            num_parts = (total_paragraphs + args.max_paragraphs - 1) // args.max_paragraphs
            print(f"Will create {num_parts} file(s) with max {args.max_paragraphs} paragraphs each")

            if not args.dry_run:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_files = split_docx(input_path, output_dir, args.max_paragraphs)
        except ImportError:
            print("Error: python-docx not installed. Install with: pip install python-docx", file=sys.stderr)
            sys.exit(1)

    else:
        print(f"Error: Unsupported file type: {suffix}", file=sys.stderr)
        print("Supported types: .pdf, .txt, .md, .docx", file=sys.stderr)
        sys.exit(1)

    if args.dry_run or not output_files:
        print()
        sys.exit(0)

    # Summary
    total_size = sum(f.stat().st_size for f in output_files)
    print(f"\n✓ Split complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Files created: {len(output_files)}")
    print(f"  Total size: {_human_size(total_size)}")

    # Show how to use with the app
    print("\n📋 Next steps:")
    print("  1. Upload the split files to the app individually or in small batches")
    print("  2. Process each batch separately to avoid circuit breaker issues")
    print("  3. Combine the results manually if needed")

    print("\n💡 Tip: Start with smaller batches (2-3 files) to test the LLM endpoint")


def _count_pdf_pages(path: Path) -> int:
    """Count pages in a PDF without loading the full content."""
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(path)).pages)
    except ImportError:
        try:
            from PyPDF2 import PdfReader
            return len(PdfReader(str(path)).pages)
        except ImportError:
            import pikepdf
            with pikepdf.Pdf.open(str(path)) as pdf:
                return len(pdf.pages)


if __name__ == "__main__":
    main()
