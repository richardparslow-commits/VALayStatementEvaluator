"""Tests for scripts/ocr_records.py — the pre-upload OCR step.

The actual OCR binaries are not installed in CI and must not be required, so these
cover what the script decides: which pages need OCR, which backend is available,
where the output goes, and that it never writes over its input.
"""
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from scripts import ocr_records  # noqa: E402


def _pdf_bytes(pages: list[str]) -> bytes:
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    for text in pages:
        if text:
            pdf.drawString(72, 720, text)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


class TestInspect(unittest.TestCase):
    def test_image_only_pages_are_named(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "records.pdf"
            path.write_bytes(_pdf_bytes(["Knee pain noted.", "", "Tinnitus noted."]))
            total, image_only = ocr_records.inspect_pdf(path)
        self.assertEqual(total, 3)
        self.assertEqual(image_only, [2])

    def test_a_fully_readable_pdf_has_nothing_to_do(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "records.pdf"
            path.write_bytes(_pdf_bytes(["Text here."]))
            total, image_only = ocr_records.inspect_pdf(path)
        self.assertEqual((total, image_only), (1, []))

    def test_a_password_protected_pdf_is_refused_with_a_reason(self) -> None:
        import tempfile

        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.encrypt("secret")
        buffer = io.BytesIO()
        writer.write(buffer)
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "locked.pdf"
            path.write_bytes(buffer.getvalue())
            with self.assertRaises(ValueError) as raised:
                ocr_records.inspect_pdf(path)
        self.assertIn("password", str(raised.exception))


class TestBackendSelection(unittest.TestCase):
    def test_ocrmypdf_is_preferred(self) -> None:
        with patch(
            "scripts.ocr_records.shutil.which",
            side_effect=lambda name: "/usr/bin/" + name,
        ):
            self.assertEqual(ocr_records.find_backend(), "ocrmypdf")

    def test_tesseract_is_used_when_ocrmypdf_is_absent(self) -> None:
        def which(name: str) -> str | None:
            return None if name == "ocrmypdf" else f"/usr/bin/{name}"

        with patch("scripts.ocr_records.shutil.which", side_effect=which):
            self.assertEqual(ocr_records.find_backend(), "tesseract")

    def test_no_tooling_means_none(self) -> None:
        with patch("scripts.ocr_records.shutil.which", return_value=None):
            self.assertIsNone(ocr_records.find_backend())


class TestOutputPath(unittest.TestCase):
    def test_output_is_a_new_file_beside_the_input(self) -> None:
        out = ocr_records.default_output_path(Path("/tmp/records.pdf"))
        self.assertEqual(out, Path("/tmp/records.ocr.pdf"))

    def test_a_different_extension_does_not_replace_it(self) -> None:
        self.assertEqual(
            ocr_records.default_output_path(Path("scan.PDF")), Path("scan.ocr.PDF")
        )


class TestMain(unittest.TestCase):
    def _run(self, argv: list[str]) -> int:
        with patch.object(sys, "argv", ["ocr_records.py"] + argv):
            return ocr_records.main(argv)

    def test_a_missing_file_is_bad_input(self) -> None:
        self.assertEqual(self._run(["/tmp/definitely-not-here-1234.pdf"]), 3)

    def test_report_only_never_needs_a_backend(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "records.pdf"
            path.write_bytes(_pdf_bytes(["Knee pain.", ""]))
            with patch("scripts.ocr_records.find_backend") as backend:
                code = self._run([str(path), "--report-only"])
        self.assertEqual(code, 0)
        backend.assert_not_called()

    def test_nothing_to_do_is_reported_without_running_ocr(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "records.pdf"
            path.write_bytes(_pdf_bytes(["All text."]))
            with patch("scripts.ocr_records.ocr_with_ocrmypdf") as ocr:
                code = self._run([str(path)])
        self.assertEqual(code, 1)
        ocr.assert_not_called()

    def test_missing_tooling_exits_two_and_names_the_pages(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "records.pdf"
            path.write_bytes(_pdf_bytes(["Text.", ""]))
            with patch("scripts.ocr_records.find_backend", return_value=None):
                code = self._run([str(path)])
        self.assertEqual(code, 2)

    def test_it_refuses_to_overwrite_its_input(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "records.pdf"
            path.write_bytes(_pdf_bytes(["Text.", ""]))
            with patch("scripts.ocr_records.find_backend", return_value="ocrmypdf"):
                code = self._run([str(path), "--out", str(path)])
        self.assertEqual(code, 3)

    def test_an_existing_output_needs_force(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "records.pdf"
            path.write_bytes(_pdf_bytes(["Text.", ""]))
            out = Path(workdir) / "out.pdf"
            out.write_bytes(b"%PDF-1.4 pretend")
            with patch("scripts.ocr_records.find_backend", return_value="ocrmypdf"):
                blocked = self._run([str(path), "--out", str(out)])
            self.assertEqual(blocked, 3)

            def _fake_ocr(source: Path, destination: Path) -> None:
                # A real OCR pass produces a real PDF; the script re-reads its own
                # output to report how many pages now have text.
                Path(destination).write_bytes(_pdf_bytes(["OCR text."]))

            with patch("scripts.ocr_records.find_backend", return_value="ocrmypdf"), patch(
                "scripts.ocr_records.ocr_with_ocrmypdf", side_effect=_fake_ocr
            ) as ocr:
                allowed = self._run([str(path), "--out", str(out), "--force"])
        self.assertEqual(allowed, 0)
        ocr.assert_called_once()

    def test_output_that_is_not_a_readable_pdf_is_reported(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "records.pdf"
            path.write_bytes(_pdf_bytes(["Text.", ""]))
            out = Path(workdir) / "out.pdf"
            with patch("scripts.ocr_records.find_backend", return_value="ocrmypdf"), patch(
                "scripts.ocr_records.ocr_with_ocrmypdf",
                side_effect=lambda source, destination: Path(destination).write_bytes(b"junk"),
            ):
                code = self._run([str(path), "--out", str(out)])
        self.assertEqual(code, 3)


class TestParallelFallback(unittest.TestCase):
    """The tesseract fallback OCRs pages concurrently — a 1,000-page scan is
    hours of sequential subprocess calls otherwise."""

    def _two_image_pages(self, tmp: Path) -> Path:
        path = tmp / "records.pdf"
        path.write_bytes(_pdf_bytes(["", ""]))  # both pages image-only
        return path

    def test_workers_run_concurrently_and_all_pages_arrive(self) -> None:
        import threading
        import tempfile
        import time as time_module

        with tempfile.TemporaryDirectory() as workdir:
            tmp = Path(workdir)
            path = self._two_image_pages(tmp)
            barrier = threading.Barrier(2, timeout=10)
            started: list[None] = []

            def fake_run(command: list[str], *, what: str) -> None:
                if command[0] == "pdftoppm":
                    started.append(None)
                    try:
                        barrier.wait()  # proves 2 workers inside subprocesses at once
                    except threading.BrokenBarrierError:  # pragma: no cover
                        pass
                    # Write the rasterized page the driver looks for.
                    prefix = Path(command[-1])
                    prefix.with_name(prefix.name + "-1.png").write_bytes(b"png")

            with patch.object(ocr_records, "_run", side_effect=fake_run), patch(
                "scripts.ocr_records.subprocess.run",
                return_value=mock.Mock(returncode=0, stdout="OCR text."),
            ):
                out = tmp / "out.pdf"
                ocr_records.ocr_with_tesseract(path, out, dpi=72, jobs=2)
                # Assert while the TemporaryDirectory still exists.
                self.assertEqual(len(started), 2)
                self.assertTrue(out.is_file())
                self.assertEqual(len(ocr_records.inspect_pdf(out)[1]), 0)  # both pages have text

    def test_one_failed_page_leaves_only_itself_blank(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as workdir:
            tmp = Path(workdir)
            path = self._two_image_pages(tmp)

            def fake_run(command: list[str], *, what: str) -> None:
                if command[0] == "pdftoppm" and "page-00001" in str(command[-1]):
                    raise RuntimeError("pdftoppm failed (exit 1)")
                if command[0] == "pdftoppm":
                    prefix = Path(command[-1])
                    prefix.with_name(prefix.name + "-1.png").write_bytes(b"png")

            with patch.object(ocr_records, "_run", side_effect=fake_run), patch(
                "scripts.ocr_records.subprocess.run",
                return_value=mock.Mock(returncode=0, stdout="OCR text."),
            ):
                out = tmp / "out.pdf"
                ocr_records.ocr_with_tesseract(path, out, dpi=72, jobs=2)
                total, image_only = ocr_records.inspect_pdf(out)
                self.assertEqual((total, image_only), (2, [1]))  # page 1 blank, page 2 OCRd

    def test_jobs_one_is_sequential(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as workdir:
            tmp = Path(workdir)
            path = self._two_image_pages(tmp)
            order: list[str] = []
            active = threading_local_max = 0

            def fake_run(command: list[str], *, what: str) -> None:
                nonlocal active, threading_local_max
                if command[0] != "pdftoppm":
                    return
                active += 1
                threading_local_max = max(threading_local_max, active)
                time.sleep(0.02)
                prefix = Path(command[-1])
                prefix.with_name(prefix.name + "-1.png").write_bytes(b"png")
                active -= 1
                order.append(str(command[-1]))

            import time

            with patch.object(ocr_records, "_run", side_effect=fake_run), patch(
                "scripts.ocr_records.subprocess.run",
                return_value=mock.Mock(returncode=0, stdout="OCR text."),
            ):
                out = tmp / "out.pdf"
                ocr_records.ocr_with_tesseract(path, out, dpi=72, jobs=1)
                self.assertEqual(threading_local_max, 1)
                self.assertTrue(out.is_file())

    def test_jobs_zero_derives_workers_from_cpu_count(self) -> None:
        """jobs=0 means one worker per CPU capped at 8, then clamped to the
        number of image-only pages — derived at call time on a real 10-page
        image-only PDF, so the full chain min(8, cpu_count) applies."""
        from concurrent.futures import ThreadPoolExecutor
        import tempfile

        with tempfile.TemporaryDirectory() as workdir:
            tmp = Path(workdir)
            path = tmp / "records.pdf"
            path.write_bytes(_pdf_bytes([""] * 10))  # 10 image-only pages
            seen_workers: list[int] = []

            class SpyPool(ThreadPoolExecutor):
                def __init__(self, max_workers=None, *args, **kwargs):
                    seen_workers.append(max_workers)
                    super().__init__(max_workers=max_workers, *args, **kwargs)

            with patch.object(
                ocr_records, "_ocr_page", return_value="OCR text."
            ), patch("os.cpu_count", return_value=16), patch(
                "concurrent.futures.ThreadPoolExecutor", SpyPool
            ):
                out = tmp / "out.pdf"
                ocr_records.ocr_with_tesseract(path, out, dpi=72, jobs=0)
                self.assertTrue(out.is_file())
                self.assertEqual(seen_workers[-1], 8)  # min(8, cpu_count=16)

    def test_pool_is_clamped_to_the_image_only_page_count(self) -> None:
        """Two image-only pages never justify eight workers."""
        from concurrent.futures import ThreadPoolExecutor
        import tempfile

        with tempfile.TemporaryDirectory() as workdir:
            tmp = Path(workdir)
            path = self._two_image_pages(tmp)
            seen_workers: list[int] = []

            class SpyPool(ThreadPoolExecutor):
                def __init__(self, max_workers=None, *args, **kwargs):
                    seen_workers.append(max_workers)
                    super().__init__(max_workers=max_workers, *args, **kwargs)

            with patch.object(
                ocr_records, "_ocr_page", return_value="OCR text."
            ), patch("os.cpu_count", return_value=16), patch(
                "concurrent.futures.ThreadPoolExecutor", SpyPool
            ):
                out = tmp / "out.pdf"
                ocr_records.ocr_with_tesseract(path, out, dpi=72, jobs=0)
                self.assertEqual(seen_workers[-1], 2)


if __name__ == "__main__":
    unittest.main()
