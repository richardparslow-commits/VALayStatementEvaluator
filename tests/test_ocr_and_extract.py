"""Tests for scripts/ocr_and_extract.py — the sandbox's OCR + extraction entrypoint.

The OCR binaries are not installed in CI and must not be required, so the backend
is faked the way an engine would behave (it writes a text PDF) and everything else
is exercised for real: the app's own reader parses the bytes, the JSON is the queue's
document shape, and the exit status distinguishes "read the scans" from "there was
nothing to read them with".
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app.job_payload import documents_from_json  # noqa: E402
from scripts import ocr_and_extract, ocr_records  # noqa: E402


def _pdf_bytes(pages: list[str]) -> bytes:
    """A PDF whose pages carry *pages* as text; an empty string is a blank page."""
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    for text in pages:
        if text:
            pdf.drawString(72, 720, text)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


#: Sentences, not placeholders: the app refuses a document whose extracted text is
#: under 20 characters (``app/documents.py``), so a fixture of "a" would trip the
#: app's own rule and hide whatever the test meant to check.
TYPED = "Knee pain noted on examination, December 2024."


def _fake_ocrmypdf(source: Path, destination: Path) -> None:
    """Stand in for ocrmypdf: a text layer over pages that had none.

    The engine's *output* is what this entrypoint reasons about, so the fake has to
    produce the real shape — a PDF whose previously image-only pages now have text.
    """
    total, _image_only = ocr_records.inspect_pdf(source)
    # Every page ends up with text: ocrmypdf leaves the pages that already had text
    # alone and adds a layer to the rest, so the copy is readable end to end.
    destination.write_bytes(
        _pdf_bytes([f"Scanned page {number}: knee pain noted in 2024." for number in range(1, total + 1)])
    )


def _fake_ocrmypdf_that_fails(source: Path, destination: Path) -> None:
    raise RuntimeError("ocrmypdf failed (exit 1): mocked refusal")


class BundleTestCase(unittest.TestCase):
    """A temporary bundle, so every test reads a real directory."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.bundle = self.root / "records"
        self.bundle.mkdir()
        self.work = self.root / "work"

    def write(self, relative: str, data: bytes) -> Path:
        path = self.bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def ocrmypdf_available(self):
        """Patch the tooling so `find_backend()` answers "ocrmypdf" and the engine is faked."""
        return (
            patch(
                "scripts.ocr_records.shutil.which",
                side_effect=lambda name: f"/usr/bin/{name}",
            ),
            patch("scripts.ocr_records.ocr_with_ocrmypdf", side_effect=_fake_ocrmypdf),
        )


class TestDiscovery(BundleTestCase):
    def test_it_walks_a_bundle_and_keeps_relative_labels(self) -> None:
        self.write("scan.pdf", _pdf_bytes(["text"]))
        self.write("nested/notes.txt", b"hello")
        found = ocr_and_extract.discover([self.bundle], work_dir=self.work)
        names = [path.relative_to(self.bundle).as_posix() for path in found]
        self.assertEqual(names, ["nested/notes.txt", "scan.pdf"])

    def test_it_ignores_files_the_app_could_not_ingest(self) -> None:
        """A folder holds a README, a portal screenshot, a JSON dump — not records."""
        self.write("scan.pdf", _pdf_bytes(["text"]))
        self.write("portal.png", b"\x89PNG")
        self.write("export.json", b"{}")
        found = ocr_and_extract.discover([self.bundle], work_dir=self.work)
        self.assertEqual([p.name for p in found], ["scan.pdf"])

    def test_it_skips_hidden_files_and_directories(self) -> None:
        self.write("scan.pdf", _pdf_bytes(["text"]))
        self.write(".DS_Store", b"junk")
        self.write(".cache/big.pdf", _pdf_bytes(["text"]))
        found = ocr_and_extract.discover([self.bundle], work_dir=self.work)
        self.assertEqual([p.name for p in found], ["scan.pdf"])

    def test_its_own_ocr_copies_are_not_records(self) -> None:
        """Otherwise a second run would extract the OCR'd copies alongside the sources."""
        self.write("scan.pdf", _pdf_bytes(["text"]))
        copies = self.bundle / ocr_and_extract.WORK_DIR_NAME
        copies.mkdir()
        (copies / "scan.ocr.pdf").write_bytes(_pdf_bytes(["text"]))
        found = ocr_and_extract.discover([self.bundle], work_dir=copies)
        self.assertEqual([p.name for p in found], ["scan.pdf"])

    def test_the_same_file_named_twice_is_counted_once(self) -> None:
        scan = self.write("scan.pdf", _pdf_bytes(["text"]))
        found = ocr_and_extract.discover([self.bundle, scan], work_dir=self.work)
        self.assertEqual(len(found), 1)


class TestPrepare(BundleTestCase):
    def test_a_readable_pdf_is_not_sent_to_ocr(self) -> None:
        self.work.mkdir(parents=True, exist_ok=True)
        original = _pdf_bytes(["Knee pain noted."])
        data, report = ocr_and_extract.prepare(
            "records.pdf", original, work_dir=self.work, dpi=300, use_ocr=True
        )
        self.assertEqual(report["ocr"], "not_needed")
        self.assertEqual(report["image_only_before"], [])
        self.assertEqual(data, original, "the bytes should come back untouched")

    def test_a_scan_is_ocrd_and_the_copy_is_what_gets_extracted(self) -> None:
        self.work.mkdir(parents=True, exist_ok=True)
        a, b = self.ocrmypdf_available()
        with a, b:
            data, report = ocr_and_extract.prepare(
                "scan.pdf", _pdf_bytes(["", "", "text page"]), work_dir=self.work, dpi=300, use_ocr=True
            )
        self.assertEqual(report["ocr"], "ocrmypdf")
        self.assertEqual(report["image_only_before"], [1, 2])
        self.assertEqual(report["image_only_after"], [])
        self.assertTrue(report["ocr_output"].endswith("scan.ocr.pdf"))
        # The extracted bytes are the OCR copy: pages 1 and 2 now carry text.
        self.assertNotEqual(data, _pdf_bytes(["", "", "text page"]))
        self.assertEqual(ocr_records.inspect_pdf(Path(report["ocr_output"]))[1], [])

    def test_the_ocr_copy_keeps_the_original_label(self) -> None:
        """Citations must point at the file the user has, not at `.ocr.pdf`.

        The document's pages are the OCR copy's text, but its name — and so every
        page marker the digest quotes — has to stay the record that was uploaded.
        """
        self.work.mkdir(parents=True, exist_ok=True)
        self.write("a/scan.pdf", _pdf_bytes(["", ""]))
        a, b = self.ocrmypdf_available()
        with a, b:
            report = ocr_and_extract.process(
                [self.bundle / "a" / "scan.pdf"], roots=[self.bundle], work_dir=self.work
            )
        self.assertEqual([doc["filename"] for doc in report["documents"]], ["a/scan.pdf"])
        self.assertTrue(
            all(page["text"].strip() for page in report["documents"][0]["pages"]),
            "the OCR copy's text is what should be in the document",
        )
        self.assertTrue(report["files"][0]["ocr_output"].endswith("a__scan.ocr.pdf"))

    def test_no_tooling_leaves_the_scan_alone_and_says_so(self) -> None:
        self.work.mkdir(parents=True, exist_ok=True)
        original = _pdf_bytes(["", "text"])
        with patch("scripts.ocr_records.shutil.which", return_value=None):
            data, report = ocr_and_extract.prepare(
                "scan.pdf", original, work_dir=self.work, dpi=300, use_ocr=True
            )
        self.assertEqual(report["ocr"], "unavailable")
        self.assertEqual(report["image_only_before"], [1])
        self.assertIn("1", report["notes"][0])
        self.assertEqual(data, original, "no copy was made, so the bytes are the source's")

    def test_use_ocr_false_records_a_skip_rather_than_a_failure(self) -> None:
        self.work.mkdir(parents=True, exist_ok=True)
        with patch("scripts.ocr_records.shutil.which", side_effect=lambda name: f"/usr/bin/{name}"):
            _, report = ocr_and_extract.prepare(
                "scan.pdf", _pdf_bytes([""]), work_dir=self.work, dpi=300, use_ocr=False
            )
        self.assertEqual(report["ocr"], "skipped")

    def test_the_tesseract_fallback_is_used_when_ocrmypdf_is_absent(self) -> None:
        self.work.mkdir(parents=True, exist_ok=True)
        calls: list[str] = []

        def fake_tesseract(source: Path, destination: Path, *, dpi: int = 300) -> None:
            calls.append(f"tesseract:{dpi}")
            destination.write_bytes(_pdf_bytes(["read by tesseract"]))

        def which(name: str) -> str | None:
            return None if name == "ocrmypdf" else f"/usr/bin/{name}"

        with patch("scripts.ocr_records.shutil.which", side_effect=which), patch(
            "scripts.ocr_records.ocr_with_tesseract", side_effect=fake_tesseract
        ):
            _, report = ocr_and_extract.prepare(
                "scan.pdf", _pdf_bytes([""]), work_dir=self.work, dpi=200, use_ocr=True
            )
        self.assertEqual(report["ocr"], "tesseract")
        self.assertEqual(calls, ["tesseract:200"])

    def test_a_refused_engine_is_reported_not_raised(self) -> None:
        self.work.mkdir(parents=True, exist_ok=True)
        with patch(
            "scripts.ocr_records.shutil.which", side_effect=lambda name: f"/usr/bin/{name}"
        ), patch(
            "scripts.ocr_records.ocr_with_ocrmypdf", side_effect=_fake_ocrmypdf_that_fails
        ):
            data, report = ocr_and_extract.prepare(
                "scan.pdf", _pdf_bytes([""]), work_dir=self.work, dpi=300, use_ocr=True
            )
        self.assertEqual(report["ocr"], "ocrmypdf-failed")
        self.assertIn("mocked refusal", report["notes"][0])
        self.assertTrue(data)  # the original bytes, still extractable for its text pages

    def test_an_engine_that_leaves_a_page_unreadable_says_which(self) -> None:
        self.work.mkdir(parents=True, exist_ok=True)

        def useless_engine(source: Path, destination: Path) -> None:
            destination.write_bytes(_pdf_bytes([""]))

        with patch(
            "scripts.ocr_records.shutil.which", side_effect=lambda name: f"/usr/bin/{name}"
        ), patch("scripts.ocr_records.ocr_with_ocrmypdf", side_effect=useless_engine):
            _, report = ocr_and_extract.prepare(
                "scan.pdf", _pdf_bytes([""]), work_dir=self.work, dpi=300, use_ocr=True
            )
        self.assertEqual(report["image_only_after"], [1])
        self.assertIn("still image-only", report["notes"][0])

    def test_a_page_of_trivial_text_is_refused_by_the_apps_own_rule(self) -> None:
        """The box inherits the app's floor (20 extracted characters), rather than
        inventing a friendlier one — a two-word page must not become a document here
        and vanish after the upload."""
        self.write("stub.pdf", _pdf_bytes(["a", "b", "c"]))
        report = ocr_and_extract.process(
            [self.bundle / "stub.pdf"], roots=[self.bundle], work_dir=self.work
        )
        self.assertEqual(report["documents"], [])
        self.assertTrue(
            any("no extractable text" in message for message in report["skipped"]),
            report["skipped"],
        )

    def test_a_non_pdf_is_passed_through_untouched(self) -> None:
        self.work.mkdir(parents=True, exist_ok=True)
        data, report = ocr_and_extract.prepare(
            "notes.txt", b"plain text", work_dir=self.work, dpi=300, use_ocr=True
        )
        self.assertEqual(data, b"plain text")
        self.assertEqual(report["ocr"], "not_needed")
        self.assertEqual(report["pages"], 0)


class TestBundle(BundleTestCase):
    def test_an_ocrd_bundle_extracts_with_text_on_every_page(self) -> None:
        self.write("a/scan.pdf", _pdf_bytes(["", ""]))
        self.write("b/notes.txt", TYPED.encode("utf-8"))
        a, b = self.ocrmypdf_available()
        with a, b:
            report = ocr_and_extract.process(
                ocr_and_extract.discover([self.bundle], work_dir=self.work),
                roots=[self.bundle],
                work_dir=self.work,
            )
        self.assertEqual(report["totals"]["documents"], 2)
        self.assertEqual(report["totals"]["pages"], 3)
        self.assertEqual(report["totals"]["pages_unreadable"], 0)
        self.assertEqual(sorted(d["filename"] for d in report["documents"]), ["a/scan.pdf", "b/notes.txt"])

    def test_the_documents_round_trip_through_the_queue_shape(self) -> None:
        """The point of emitting `documents`: a worker pod can read it, unchanged."""
        self.write("scan.pdf", _pdf_bytes([""]))
        a, b = self.ocrmypdf_available()
        with a, b:
            report = ocr_and_extract.process(
                [self.bundle / "scan.pdf"], roots=[self.bundle], work_dir=self.work
            )
        documents = documents_from_json(report["documents"])
        self.assertEqual([doc.filename for doc in documents], ["scan.pdf"])
        self.assertTrue(all(page.text.strip() for page in documents[0].pages))

    def test_a_bundle_without_tooling_still_extracts_what_has_text(self) -> None:
        self.write("scan.pdf", _pdf_bytes(["", TYPED]))
        with patch("scripts.ocr_records.shutil.which", return_value=None):
            report = ocr_and_extract.process(
                ocr_and_extract.discover([self.bundle], work_dir=self.work),
                roots=[self.bundle],
                work_dir=self.work,
            )
        self.assertEqual(report["totals"]["pages"], 2)
        self.assertEqual(report["totals"]["pages_unreadable"], 1)
        self.assertEqual(report["files"][0]["ocr"], "unavailable")

    def test_a_zip_is_expanded_the_way_an_upload_is(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("scan.pdf", _pdf_bytes([""]))
            archive.writestr("notes.txt", TYPED)
            archive.writestr("portal.png", b"\x89PNG")
        self.write("bundle.zip", buffer.getvalue())
        a, b = self.ocrmypdf_available()
        with a, b:
            report = ocr_and_extract.process(
                [self.bundle / "bundle.zip"], roots=[self.bundle], work_dir=self.work
            )
        self.assertEqual(
            sorted(entry["filename"] for entry in report["documents"]),
            ["bundle/notes.txt", "bundle/scan.pdf"],
        )
        self.assertTrue(
            any("unsupported file type" in message for message in report["skipped"]),
            f"the PNG member should be named as skipped: {report['skipped']}",
        )

    def test_an_unreadable_file_is_skipped_with_its_reason(self) -> None:
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.encrypt("secret")
        locked = io.BytesIO()
        writer.write(locked)
        self.write("locked.pdf", locked.getvalue())
        report = ocr_and_extract.process(
            [self.bundle / "locked.pdf"], roots=[self.bundle], work_dir=self.work
        )
        self.assertEqual(report["documents"], [])
        self.assertTrue(any("password" in message for message in report["skipped"]), report["skipped"])

    def test_the_totals_name_the_page_cap_it_breaks(self) -> None:
        import app.config as config

        self.write("scan.pdf", _pdf_bytes([TYPED, TYPED, TYPED]))
        with patch.object(config, "MAX_RECORD_PAGES", 2):
            report = ocr_and_extract.process(
                [self.bundle / "scan.pdf"], roots=[self.bundle], work_dir=self.work
            )
        self.assertTrue(report["totals"]["over_page_cap"])
        self.assertEqual(report["totals"]["page_cap"], 2)


class TestMain(BundleTestCase):
    def _run(self, argv: list[str]) -> int:
        return ocr_and_extract.main([str(self.bundle), "--work-dir", str(self.work)] + argv)

    def test_a_missing_path_is_bad_input(self) -> None:
        self.assertEqual(
            ocr_and_extract.main([str(self.root / "nope")]), ocr_and_extract.EXIT_BAD_INPUT
        )

    def test_a_bundle_with_no_records_reports_nothing_to_do(self) -> None:
        self.write("portal.png", b"\x89PNG")
        self.assertEqual(self._run([]), ocr_and_extract.EXIT_NOTHING_TO_DO)

    def test_a_successful_run_writes_the_json_and_exits_zero(self) -> None:
        self.write("notes.txt", TYPED.encode("utf-8"))
        out = self.root / "bundle.json"
        self.work.mkdir(parents=True, exist_ok=True)
        self.assertEqual(self._run(["--out", str(out)]), ocr_and_extract.EXIT_OK)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["version"], 1)
        self.assertEqual(payload["totals"]["documents"], 1)
        self.assertTrue(payload["bundle"])

    def test_an_existing_output_is_not_overwritten_without_force(self) -> None:
        self.write("notes.txt", TYPED.encode("utf-8"))
        out = self.root / "bundle.json"
        out.write_text("{}", encoding="utf-8")
        self.work.mkdir(parents=True, exist_ok=True)
        self.assertEqual(self._run(["--out", str(out)]), ocr_and_extract.EXIT_BAD_INPUT)
        self.assertEqual(out.read_text(encoding="utf-8"), "{}")
        self.assertEqual(
            self._run(["--out", str(out), "--force"]), ocr_and_extract.EXIT_OK
        )

    def test_scans_without_tooling_exit_two_and_still_write_the_report(self) -> None:
        self.write("scan.pdf", _pdf_bytes(["", TYPED]))
        out = self.root / "bundle.json"
        self.work.mkdir(parents=True, exist_ok=True)
        with patch("scripts.ocr_records.shutil.which", return_value=None):
            code = self._run(["--out", str(out)])
        self.assertEqual(code, ocr_and_extract.EXIT_NO_TOOLING)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["files"][0]["ocr"], "unavailable")
        self.assertEqual(payload["totals"]["pages_unreadable"], 1)

    def test_an_all_scan_bundle_without_tooling_names_the_missing_tooling(self) -> None:
        """The actionable answer, not "nothing extractable": the operator has files
        that need OCR, and the box they are in has none installed. The app's own
        reader refuses every page, so documents is 0 — which is exactly why the
        missing tooling has to be the stated reason."""
        self.write("scan.pdf", _pdf_bytes(["", ""]))
        out = self.root / "bundle.json"
        self.work.mkdir(parents=True, exist_ok=True)
        with patch("scripts.ocr_records.shutil.which", return_value=None):
            code = self._run(["--out", str(out)])
        self.assertEqual(code, ocr_and_extract.EXIT_NO_TOOLING)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["totals"]["documents"], 0)
        self.assertEqual(payload["files"][0]["ocr"], "unavailable")

    def test_report_only_writes_nothing(self) -> None:
        self.write("scan.pdf", _pdf_bytes([""]))
        self.work.mkdir(parents=True, exist_ok=True)
        self.assertEqual(self._run(["--report-only"]), ocr_and_extract.EXIT_OK)
        self.assertFalse((self.work / "bundle.json").exists())

    def test_no_ocr_reproduces_what_the_app_sees_today(self) -> None:
        """A fully scanned bundle with OCR off is what the uploader gives the app:
        no documents, one reason per file — and the report still lands on disk, so
        the failure can be read after the run rather than only on the console."""
        self.write("scan.pdf", _pdf_bytes(["", ""]))
        out = self.root / "bundle.json"
        self.work.mkdir(parents=True, exist_ok=True)
        code = self._run(["--no-ocr", "--out", str(out)])
        self.assertEqual(code, ocr_and_extract.EXIT_BAD_INPUT)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["files"][0]["ocr"], "skipped")
        self.assertEqual(payload["totals"]["documents"], 0)
        self.assertTrue(payload["skipped"], "the reason has to be recorded")


if __name__ == "__main__":
    unittest.main()
