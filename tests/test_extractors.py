"""Tests for app/extractors.py — the swap between readers, and its fallback.

The box is not available in CI (no Docker, no Vercel sandbox), so the runner is a
fake with a real contract: it is handed a staged directory and must answer with the
report JSON ``scripts/ocr_and_extract.py`` writes. Everything else runs for real —
the app's own reader parses the bytes, staging and the blob store are exercised on
disk, and the fail-open path is what is asserted, because that is the property that
keeps a misconfigured box from costing the user a run.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app import config, documents as documents_mod, error_report, extractors, pipeline_guard  # noqa: E402
from app.blob_store import BlobRef, BlobStore, BlobStoreError  # noqa: E402
from app.documents import extract_document, set_active_extractor  # noqa: E402
from app.job_payload import documents_to_json  # noqa: E402

TYPED = (
    "Knee pain noted on examination, December 2024. Range of motion 100 degrees, "
    "painful motion, no ankylosis reported by the examiner."
)


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


class FakeRunner:
    """A runner that answers with whatever the test wants the box to have said."""

    def __init__(self, answer: str | Exception) -> None:
        self.answer = answer
        self.calls: list[tuple[str, Path, float]] = []

    def run(self, staged: extractors.StagedFile, work_dir: Path, timeout: float) -> str:
        self.calls.append((staged.label, work_dir, timeout))
        self.staged = staged
        self.work_dir = work_dir
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class ScriptRunner:
    """The honest fake: run the real entrypoint over the staged directory.

    This is what the operator's command does on the box, minus the trip to the VM,
    so the swap is tested end to end — the app's own reader on the box's side, the
    queue JSON in the middle, the app's types on this side.
    """

    def __init__(self, *, no_ocr: bool = False) -> None:
        self.no_ocr = no_ocr
        self.timeout: float | None = None

    def run(self, staged: extractors.StagedFile, work_dir: Path, timeout: float) -> str:
        from scripts import ocr_and_extract

        self.timeout = timeout
        out = work_dir / "bundle.json"
        argv = [str(work_dir), "--out", str(out), "--force"]
        if self.no_ocr:
            argv.append("--no-ocr")
        with patch("sys.stdout", io.StringIO()), patch("sys.stderr", io.StringIO()):
            ocr_and_extract.main(argv)
        report = json.loads(out.read_text(encoding="utf-8"))
        # The entrypoint parses its arguments, so the staged file has to keep the
        # name the app handed over: a `.pdf` that is not a PDF would be skipped.
        assert staged.path.exists(), "the runner must be able to see the staged file"
        return json.dumps(report)


class MemoryStore(BlobStore):
    """A blob store that records what the adapter staged and kept."""

    name = "memory"

    def __init__(self) -> None:
        self.blobs: list[bytes] = []

    def put(self, data: bytes) -> BlobRef:
        self.blobs.append(data)
        digest = hashlib.sha256(data).hexdigest()
        return BlobRef(
            key=f"blobs/{digest[:2]}/{digest}.json",
            sha256=digest,
            size=len(data),
            backend=self.name,
        )

    def get(self, ref: BlobRef) -> bytes:  # pragma: no cover - the adapter never reads back
        raise AssertionError("the adapter should not read blobs back")


class BrokenStore(MemoryStore):
    """A store whose writes fail — staging must not be able to stop a read."""

    def put(self, data: bytes) -> BlobRef:
        raise BlobStoreError("blob store is having a bad day")


class ExtractorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.addCleanup(lambda: set_active_extractor(None))
        # ``report_failure(once=True)`` remembers (phase, message) for the life of
        # the process, which is the behavior under test — so a test that asserts
        # "logged once" has to start from a clean slate rather than depending on
        # which tests ran before it.
        error_report._ONCE_SEEN.clear()
        self.addCleanup(error_report._ONCE_SEEN.clear)
        # The extraction status is process-wide too (it is what /health and the
        # sidebar report), and the fallback tests below write it on purpose — so
        # clean it on both sides and leave no evidence for another module to find.
        extractors.reset_extraction_status()
        self.addCleanup(extractors.reset_extraction_status)

    def sandbox(self, runner: Any, **kwargs: Any) -> extractors.SandboxExtractor:
        return extractors.SandboxExtractor(
            runner, work_root=self.root, timeout_seconds=kwargs.pop("timeout_seconds", 30), **kwargs
        )

    def report_for(self, label: str, documents: list[Any], skipped: list[str] | None = None) -> str:
        return json.dumps(
            {"version": 1, "documents": documents_to_json(documents), "skipped": skipped or []}
        )


class TestSandboxExtraction(ExtractorTestCase):
    def test_it_reads_a_file_on_the_box_and_keeps_this_apps_types(self) -> None:
        data = _pdf_bytes([TYPED])
        expected = extract_document("records.pdf", data)
        extractor = self.sandbox(FakeRunner(self.report_for("records.pdf", [expected])))

        documents, skipped = extractor.extract("records.pdf", data)

        self.assertEqual(skipped, [])
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].filename, "records.pdf")
        self.assertEqual(documents[0].full_text, expected.full_text)
        self.assertEqual(documents[0].page_labelled_text(), expected.page_labelled_text())

    def test_the_real_entrypoint_answers_identically_to_the_in_process_reader(self) -> None:
        """The swap's whole claim: same reader, same text, same markers."""
        data = _pdf_bytes([TYPED, "Audiogram: mild high-frequency loss, right ear."])
        in_process = extract_document("records.pdf", data)

        documents, skipped = self.sandbox(ScriptRunner()).extract("records.pdf", data)

        self.assertEqual(skipped, [])
        self.assertEqual([d.filename for d in documents], ["records.pdf"])
        self.assertEqual(documents[0].page_labelled_text(), in_process.page_labelled_text())
        self.assertEqual(documents[0].page_records(), in_process.page_records())

    def test_a_scanned_file_reports_its_pages_as_unreadable_without_ocr(self) -> None:
        scan = _pdf_bytes(["", ""])

        documents, skipped = self.sandbox(ScriptRunner(no_ocr=True)).extract("scan.pdf", scan)

        self.assertEqual(documents, [])
        self.assertTrue(skipped, "a scan the box could not read must say so")
        self.assertIn("ocr", skipped[0].lower())

    def test_the_label_travels_in_the_manifest_and_the_suffix_on_disk(self) -> None:
        class Recorder:
            def run(self, staged: extractors.StagedFile, work_dir: Path, timeout: float) -> str:
                manifest = json.loads((work_dir / "manifest.json").read_text(encoding="utf-8"))
                self.manifest = manifest
                self.path = staged.path
                self.data = staged.path.read_bytes()
                return json.dumps(
                    {"documents": documents_to_json([extract_document(staged.label, staged.path.read_bytes())])}
                )

        recorder = Recorder()
        data = _pdf_bytes([TYPED])
        self.sandbox(recorder).extract("nested/reports/records.pdf", data)

        self.assertEqual(recorder.manifest["file"]["label"], "nested/reports/records.pdf")
        self.assertEqual(recorder.path.name, "records.pdf")
        self.assertEqual(recorder.path.suffix, ".pdf")
        self.assertEqual(recorder.data, data)
        self.assertEqual(
            recorder.manifest["file"]["sha256"],
            __import__("hashlib").sha256(data).hexdigest(),
        )

    def test_the_working_directory_is_removed_even_when_the_box_fails(self) -> None:
        extractor = self.sandbox(FakeRunner(extractors.SandboxUnavailable("no")))
        with self.assertRaises(extractors.SandboxUnavailable):
            extractor.extract("records.pdf", _pdf_bytes([TYPED]))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_the_staged_file_and_the_answer_are_kept_in_the_blob_store(self) -> None:
        store = MemoryStore()
        data = _pdf_bytes([TYPED])
        extractor = self.sandbox(
            FakeRunner(self.report_for("records.pdf", [extract_document("records.pdf", data)])),
            store=store,
        )

        extractor.extract("records.pdf", data)

        self.assertEqual(len(store.blobs), 2, "one blob for the file, one for the box's answer")
        staged = json.loads(store.blobs[0])
        self.assertEqual(staged["kind"], "record-file")
        self.assertEqual(staged["label"], "records.pdf")
        self.assertEqual(base64.b64decode(staged["file_b64"]), data)

    def test_a_store_that_refuses_writes_does_not_stop_the_read(self) -> None:
        data = _pdf_bytes([TYPED])
        extractor = self.sandbox(
            FakeRunner(self.report_for("records.pdf", [extract_document("records.pdf", data)])),
            store=BrokenStore(),
        )

        documents, skipped = extractor.extract("records.pdf", data)

        self.assertEqual(len(documents), 1)
        self.assertEqual(skipped, [])


class TestSandboxRefusals(ExtractorTestCase):
    def test_an_answer_under_a_name_the_user_never_had_is_refused(self) -> None:
        data = _pdf_bytes([TYPED])
        invented = extract_document("records.ocr.pdf", data)
        extractor = self.sandbox(FakeRunner(self.report_for("records.pdf", [invented])))

        with self.assertRaises(extractors.SandboxUnavailable) as caught:
            extractor.extract("records.pdf", data)

        self.assertIn("records.ocr.pdf", str(caught.exception))
        self.assertIn("Citations must", str(caught.exception))

    def test_a_zip_member_of_a_staged_archive_is_allowed(self) -> None:
        """Archive members legitimately extend the staged label."""
        data = _pdf_bytes([TYPED])
        member = extract_document("bundle.zip: records.pdf", data)
        extractor = self.sandbox(FakeRunner(self.report_for("bundle.zip", [member])))

        documents, _ = extractor.extract("bundle.zip", data)

        self.assertEqual([d.filename for d in documents], ["bundle.zip: records.pdf"])

    def test_stdout_that_is_not_json_is_refused_with_its_size(self) -> None:
        extractor = self.sandbox(FakeRunner("Traceback (most recent call last): …"))

        with self.assertRaises(extractors.SandboxUnavailable) as caught:
            extractor.extract("records.pdf", _pdf_bytes([TYPED]))

        self.assertIn("was not JSON", str(caught.exception))

    def test_json_without_a_documents_list_is_refused(self) -> None:
        extractor = self.sandbox(FakeRunner(json.dumps({"totals": {"pages": 3}})))

        with self.assertRaises(extractors.SandboxUnavailable) as caught:
            extractor.extract("records.pdf", _pdf_bytes([TYPED]))

        self.assertIn("no 'documents' list", str(caught.exception))

    def test_an_answer_with_nothing_in_it_is_refused(self) -> None:
        extractor = self.sandbox(FakeRunner(json.dumps({"documents": [], "skipped": []})))

        with self.assertRaises(extractors.SandboxUnavailable) as caught:
            extractor.extract("records.pdf", _pdf_bytes([TYPED]))

        self.assertIn("returned nothing at all", str(caught.exception))

    def test_a_box_that_says_why_it_skipped_a_file_is_believed(self) -> None:
        answer = json.dumps({"documents": [], "skipped": ["records.pdf: unsupported file type"]})

        documents, skipped = self.sandbox(FakeRunner(answer)).extract("records.pdf", b"%PDF-")

        self.assertEqual(documents, [])
        self.assertEqual(skipped, ["records.pdf: unsupported file type"])

    def test_a_cancelled_run_is_not_swallowed_by_the_fallback(self) -> None:
        extractor = extractors.FailOpenExtractor(self.sandbox(FakeRunner("{}")))
        with patch.object(
            pipeline_guard, "check_pipeline_cancelled", side_effect=pipeline_guard.PipelineCancelledError
        ):
            with self.assertRaises(pipeline_guard.PipelineCancelledError):
                extractor.extract("records.pdf", _pdf_bytes([TYPED]))

    def test_no_runner_is_configured_at_all(self) -> None:
        with self.assertRaises(extractors.SandboxUnavailable) as caught:
            extractors.CommandBoxRunner("   ")
        self.assertIn("No runner command configured", str(caught.exception))

    def test_a_runner_command_that_does_not_exist_says_so(self) -> None:
        runner = extractors.CommandBoxRunner("definitely-not-installed-anywhere {work}")

        with self.assertRaises(extractors.SandboxUnavailable) as caught:
            runner.run(
                extractors.StagedFile(label="records.pdf", path=self.root / "x.pdf", sha256="", size=0),
                self.root,
                5,
            )

        self.assertIn("does not exist", str(caught.exception))
        self.assertIn("VA_LSE_EXTRACTOR_RUNNER", str(caught.exception))

    def test_a_runner_that_exits_non_zero_is_refused_with_its_last_line(self) -> None:
        runner = extractors.CommandBoxRunner(
            "sh -c 'echo \"Scans are present and no OCR tooling is installed\" >&2; exit 2'"
        )

        with self.assertRaises(extractors.SandboxUnavailable) as caught:
            runner.run(
                extractors.StagedFile(label="records.pdf", path=self.root / "x.pdf", sha256="", size=0),
                self.root,
                5,
            )

        self.assertIn("exit 2", str(caught.exception))
        self.assertIn("no OCR tooling", str(caught.exception))

    def test_the_work_placeholder_is_replaced_and_nothing_else_is(self) -> None:
        runner = extractors.CommandBoxRunner("box-run --dir {work} --label literal")
        self.assertEqual(
            runner.command_for(Path("/tmp/staged")),
            ["box-run", "--dir", "/tmp/staged", "--label", "literal"],
        )


class TestRunnerInterpreter(unittest.TestCase):
    """The documented runner command has to start on a host that only has ``python3``.

    ``VA_LSE_EXTRACTOR_RUNNER="python scripts/vercel_sandbox_runner.py {work}"`` is
    the line ``.env.example`` and both docs carry; before
    :func:`app.extractors.resolve_python_interpreter` it failed on macOS, most Debian
    images and this repo's own virtualenv with "the runner command does not exist
    (python)", so the first real attempt at a box fell open for a reason that says
    nothing about the box. A configured interpreter *path* that has moved is the same
    failure one spelling over, so both are pinned here.
    """

    def test_a_python_the_host_lacks_becomes_the_running_interpreter(self) -> None:
        runner = extractors.CommandBoxRunner("python scripts/vercel_sandbox_runner.py {work}")

        with patch("app.extractors.shutil.which", return_value=None):
            with self.assertLogs("app.extractors", level="INFO") as logged:
                argv = runner.command_for(Path("/tmp/staged"))

        self.assertEqual(
            argv, [sys.executable, "scripts/vercel_sandbox_runner.py", "/tmp/staged"]
        )
        self.assertIn("'python'", "\n".join(logged.output))
        self.assertIn("not on PATH", "\n".join(logged.output))
        self.assertIn(sys.executable, "\n".join(logged.output))

    def test_a_configured_interpreter_path_that_is_not_there_is_a_warning(self) -> None:
        """The stale-``.venv`` case, which is a broken config rather than a spelling."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        missing = str(Path(tmp.name) / ".venv-sandbox" / "bin" / "python")
        runner = extractors.CommandBoxRunner(f"{missing} scripts/vercel_sandbox_runner.py {{work}}")

        with self.assertLogs("app.extractors", level="WARNING") as logged:
            argv = runner.command_for(Path("/tmp/staged"))

        self.assertEqual(
            argv, [sys.executable, "scripts/vercel_sandbox_runner.py", "/tmp/staged"]
        )
        self.assertIn("not on this host", "\n".join(logged.output))
        self.assertIn(sys.executable, "\n".join(logged.output))

    def test_an_interpreter_path_that_exists_is_left_alone(self) -> None:
        runner = extractors.CommandBoxRunner(f"{sys.executable} scripts/runner.py {{work}}")

        self.assertEqual(
            runner.command_for(Path("/w")), [sys.executable, "scripts/runner.py", "/w"]
        )

    def test_a_missing_file_that_is_not_an_interpreter_is_not_rewritten(self) -> None:
        """The path branch matches the interpreter, not every ``…/bin/…`` that is gone."""
        for program in ("~/bin/sandbox-run", "/opt/python-tools/runner"):
            runner = extractors.CommandBoxRunner(f"{program} {{work}}")
            self.assertEqual(runner.command_for(Path("/w")), [program, "/w"])

    def test_a_windows_launcher_and_a_pinned_version_are_recognized(self) -> None:
        for written in ("py", "python.exe", "python3.12.exe"):
            runner = extractors.CommandBoxRunner(f"{written} -m box {{work}}")
            with patch("app.extractors.shutil.which", return_value=None):
                argv = runner.command_for(Path("/w"))
            self.assertEqual(argv, [sys.executable, "-m", "box", "/w"])

    def test_the_documented_command_really_runs_where_there_is_no_python(self) -> None:
        """End to end: a real subprocess on a PATH with no ``python`` on it."""
        script = self._box_script()  # writes the report JSON the adapter parses
        runner = extractors.CommandBoxRunner(f"python {shlex.quote(str(script))} {{work}}")
        empty_bin = script.parent / "no-python-here"
        empty_bin.mkdir()

        with patch.dict(os.environ, {"PATH": str(empty_bin)}):
            with self.assertLogs("app.extractors", level="INFO"):
                output = runner.run(
                    extractors.StagedFile(
                        label="records.pdf", path=script, sha256="", size=0
                    ),
                    script.parent,
                    30,
                )

        # The argv the box saw is proof the interpreter was real: a missing one
        # cannot even reach the script that prints this.
        self.assertEqual(json.loads(output), {"documents": [], "argv": [str(script.parent)]})

    def test_a_python_the_host_has_is_left_exactly_as_written(self) -> None:
        runner = extractors.CommandBoxRunner("python3.12 -m box {work}")

        with patch("app.extractors.shutil.which", return_value="/usr/bin/python3.12"):
            self.assertEqual(runner.command_for(Path("/w")), ["python3.12", "-m", "box", "/w"])

    def test_a_leading_assignment_does_not_hide_the_interpreter(self) -> None:
        runner = extractors.CommandBoxRunner("PYTHONPATH=. python scripts/runner.py {work}")

        with patch("app.extractors.shutil.which", return_value=None):
            argv = runner.command_for(Path("/w"))

        self.assertEqual(argv, ["PYTHONPATH=.", sys.executable, "scripts/runner.py", "/w"])

    def test_an_empty_running_interpreter_never_rewrites_the_command(self) -> None:
        """An embedded interpreter has no path to offer, so the error stays the truth."""
        runner = extractors.CommandBoxRunner("python {work}")

        with patch("app.extractors.shutil.which", return_value=None):
            with patch.object(sys, "executable", ""):
                self.assertEqual(runner.command_for(Path("/w")), ["python", "/w"])

    def test_a_missing_program_that_is_not_python_is_still_a_missing_program(self) -> None:
        """Resolution is narrow on purpose: a typo must keep failing, and says so."""
        with patch("app.extractors.shutil.which", return_value=None):
            runner = extractors.CommandBoxRunner("definitely-not-installed-anywhere {work}")
            self.assertEqual(
                runner.command_for(Path("/w")), ["definitely-not-installed-anywhere", "/w"]
            )

    def _box_script(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        script = Path(tmp.name) / "box_runner.py"
        script.write_text(
            "import json, sys\n"
            "print(json.dumps({'documents': [], 'argv': sys.argv[1:]}))\n",
            encoding="utf-8",
        )
        return script


class TestTimeouts(ExtractorTestCase):
    def test_a_spent_run_budget_does_not_start_the_box(self) -> None:
        runner = ScriptRunner()
        extractor = self.sandbox(runner)
        with patch.object(pipeline_guard, "pipeline_remaining_seconds", return_value=0.0):
            with self.assertRaises(extractors.SandboxUnavailable) as caught:
                extractor.extract("records.pdf", _pdf_bytes([TYPED]))

        self.assertIn("budget is spent", str(caught.exception))

    def test_the_box_gets_whatever_is_left_of_the_run_and_no_more(self) -> None:
        runner = ScriptRunner()
        extractor = self.sandbox(runner, timeout_seconds=900)
        with patch.object(pipeline_guard, "pipeline_remaining_seconds", return_value=42.0):
            extractor.extract("records.pdf", _pdf_bytes([TYPED]))

        self.assertEqual(runner.timeout, 42.0)

    def test_the_configured_timeout_is_the_ceiling_when_the_run_has_room(self) -> None:
        runner = ScriptRunner()
        extractor = self.sandbox(runner, timeout_seconds=120)
        with patch.object(pipeline_guard, "pipeline_remaining_seconds", return_value=3600.0):
            extractor.extract("records.pdf", _pdf_bytes([TYPED]))

        self.assertEqual(runner.timeout, 120)

    def test_a_runner_that_hangs_is_cut_off_by_its_own_timeout(self) -> None:
        runner = extractors.CommandBoxRunner("sh -c 'sleep 30'")

        with self.assertRaises(extractors.SandboxUnavailable) as caught:
            runner.run(
                extractors.StagedFile(label="records.pdf", path=self.root / "x.pdf", sha256="", size=0),
                self.root,
                0.5,
            )

        self.assertIn("did not answer within", str(caught.exception))
        self.assertIn("VA_LSE_EXTRACTOR_TIMEOUT_SECONDS", str(caught.exception))


class TestFailOpen(ExtractorTestCase):
    def test_a_box_failure_reads_the_file_in_process_and_logs_it(self) -> None:
        data = _pdf_bytes([TYPED])
        extractor = extractors.FailOpenExtractor(
            self.sandbox(FakeRunner(extractors.SandboxUnavailable("the box is asleep")))
        )

        with self.assertLogs("app.error_report", level="WARNING") as logged:
            with self.assertLogs("app.extractors", level="INFO") as info:
                with patch("app.error_report.ensure_request_id", return_value="req-1"):
                    documents, skipped = extractor.extract("records.pdf", data)

        self.assertEqual(skipped, [])
        self.assertEqual(
            documents[0].page_labelled_text(),
            extract_document("records.pdf", data).page_labelled_text(),
        )
        self.assertTrue(any("box is asleep" in line for line in logged.output))
        self.assertTrue(any("Reading records in-process" in line for line in logged.output))
        self.assertTrue(any("records.pdf" in line for line in info.output))

    def test_the_same_reason_is_reported_once_per_process(self) -> None:
        """A box that is down must not turn a 20-file bundle into 20 tracebacks."""
        extractor = extractors.FailOpenExtractor(
            self.sandbox(FakeRunner(extractors.SandboxUnavailable("the box is asleep")))
        )
        data = _pdf_bytes([TYPED])

        with self.assertLogs("app.error_report", level="WARNING"):
            extractor.extract("one.pdf", data)
        with self.assertRaises(AssertionError):  # assertLogs fails when nothing is logged
            with self.assertLogs("app.error_report", level="WARNING"):
                extractor.extract("two.pdf", data)

    def test_a_different_reason_is_reported_separately(self) -> None:
        """A timeout and a refusal are different things to fix, so both are logged."""
        data = _pdf_bytes([TYPED])
        asleep = extractors.FailOpenExtractor(
            self.sandbox(FakeRunner(extractors.SandboxUnavailable("the box is asleep")))
        )
        with self.assertLogs("app.error_report", level="WARNING"):
            asleep.extract("one.pdf", data)

        slow = extractors.FailOpenExtractor(
            self.sandbox(FakeRunner(extractors.SandboxUnavailable("the box did not answer within 900s")))
        )
        with self.assertLogs("app.error_report", level="WARNING") as logged:
            slow.extract("two.pdf", data)

        self.assertTrue(any("did not answer within" in line for line in logged.output))

    def test_a_successful_box_answer_never_touches_the_fallback(self) -> None:
        data = _pdf_bytes([TYPED])

        class Exploding:
            def extract(self, label: str, data: bytes) -> Any:
                raise AssertionError("the fallback must not run when the box answered")

        extractor = extractors.FailOpenExtractor(
            self.sandbox(FakeRunner(self.report_for("records.pdf", [extract_document("records.pdf", data)]))),
            fallback=Exploding(),
        )

        documents, _ = extractor.extract("records.pdf", data)

        self.assertEqual(len(documents), 1)


class TestWhatIsReported(ExtractorTestCase):
    """The status ``/health`` and the sidebar read: which reader, and how it went.

    The failure this exists for is invisible from the outside — the sandbox falls back
    per file, so a finished run is exactly what a broken box produces — which is why the
    record has to be written where the decision and the fallback are made, not
    reconstructed later from configuration.
    """

    def test_nothing_configured_means_in_process_and_nothing_failed(self) -> None:
        status = extractors.extraction_status()

        self.assertEqual(status.mode, "in-process")
        self.assertFalse(status.on_the_box)
        self.assertEqual(status.fallbacks, 0)
        self.assertIsNone(status.last_fallback)
        self.assertEqual(status.problem, "")

    def test_a_sandbox_configuration_is_recorded_with_its_runner_and_timeout(self) -> None:
        with (
            patch.object(config, "EXTRACTOR_MODE", "sandbox"),
            patch.object(config, "EXTRACTOR_RUNNER", "box-run {work}"),
        ):
            extractors.build_extractor()

        status = extractors.extraction_status()
        self.assertEqual(status.mode, "sandbox")
        self.assertTrue(status.on_the_box)
        self.assertEqual(status.runner, "box-run {work}")
        self.assertEqual(status.timeout_seconds, config.EXTRACTOR_TIMEOUT_SECONDS)
        self.assertEqual(status.problem, "")

    def test_a_sandbox_without_a_runner_is_recorded_as_a_problem(self) -> None:
        """Visible before a run, which is the entire point: this one costs every file."""
        with (
            patch.object(config, "EXTRACTOR_MODE", "sandbox"),
            patch.object(config, "EXTRACTOR_RUNNER", ""),
            patch("app.error_report.ensure_request_id", return_value="req-2"),
        ):
            extractors.build_extractor()

        status = extractors.extraction_status()
        self.assertEqual(status.mode, "sandbox", "the mode is still what was asked for")
        self.assertIn("no runner is set", status.problem)
        self.assertIn("VA_LSE_EXTRACTOR_RUNNER", status.problem)

    def test_an_unknown_mode_is_recorded_as_a_problem_too(self) -> None:
        with patch.object(config, "EXTRACTOR_MODE", "kubernetes"):
            extractors.build_extractor()

        status = extractors.extraction_status()
        self.assertEqual(status.mode, "in-process")
        self.assertIn("kubernetes", status.problem)

    def test_a_fallback_is_counted_with_the_boxes_own_reason(self) -> None:
        extractor = extractors.FailOpenExtractor(
            self.sandbox(FakeRunner(extractors.SandboxUnavailable("the box is asleep")))
        )

        with self.assertLogs("app.error_report", level="WARNING"):
            extractor.extract("one.pdf", _pdf_bytes([TYPED]))
            extractor.extract("two.pdf", _pdf_bytes([TYPED]))

        status = extractors.extraction_status()
        self.assertEqual(status.fallbacks, 2)
        self.assertIsNotNone(status.last_fallback)
        assert status.last_fallback is not None  # for the type checker
        self.assertEqual(status.last_fallback.label, "two.pdf", "the last one is the one kept")
        self.assertIn("the box is asleep", status.last_fallback.reason)
        self.assertTrue(status.last_fallback.at.endswith("Z"))

    def test_a_later_success_does_not_erase_the_evidence(self) -> None:
        """A box that recovered for one file still fell back for the one before it."""
        data = _pdf_bytes([TYPED])
        extractor = extractors.FailOpenExtractor(
            self.sandbox(FakeRunner(extractors.SandboxUnavailable("the box is asleep")))
        )
        with self.assertLogs("app.error_report", level="WARNING"):
            extractor.extract("one.pdf", data)

        working = self.sandbox(FakeRunner(self.report_for("two.pdf", [extract_document("two.pdf", data)])))
        working.extract("two.pdf", data)

        status = extractors.extraction_status()
        self.assertEqual(status.fallbacks, 1)
        self.assertEqual(status.last_fallback.label if status.last_fallback else "", "one.pdf")

    def test_the_health_shape_is_the_same_record(self) -> None:
        extractors.record_configuration(mode="sandbox", runner="box-run {work}", timeout_seconds=30)
        extractors.record_fallback("no runner command configured", "records.pdf")

        payload = extractors.extraction_health()

        self.assertEqual(payload["mode"], "sandbox")
        self.assertEqual(payload["runner"], "box-run {work}")
        self.assertEqual(payload["timeout_seconds"], 30)
        self.assertEqual(payload["fallbacks"], 1)
        self.assertEqual(payload["last_fallback"]["label"], "records.pdf")

    def test_re_installing_the_configuration_keeps_the_evidence(self) -> None:
        """A Streamlit reload re-imports the entry module: that must not erase a fallback."""
        extractors.record_configuration(mode="sandbox", runner="box-run {work}")
        extractors.record_fallback("the box is asleep", "one.pdf")

        extractors.record_configuration(mode="sandbox", runner="box-run {work}")

        status = extractors.extraction_status()
        self.assertEqual(status.fallbacks, 1)
        self.assertEqual(status.runner, "box-run {work}")

    def test_the_record_can_be_cleared(self) -> None:
        extractors.record_fallback("the box is asleep", "one.pdf")
        extractors.reset_extraction_status()

        self.assertEqual(extractors.extraction_health()["fallbacks"], 0)
        self.assertIsNone(extractors.extraction_health()["last_fallback"])


class TestConfiguration(ExtractorTestCase):
    def test_the_default_mode_is_the_reader_in_this_process(self) -> None:
        with patch.object(config, "EXTRACTOR_MODE", "in-process"):
            self.assertIsNone(extractors.build_extractor())

    def test_an_unknown_mode_warns_and_stays_in_process(self) -> None:
        with patch.object(config, "EXTRACTOR_MODE", "kubernetes"):
            with self.assertLogs("app.extractors", level="WARNING") as logged:
                self.assertIsNone(extractors.build_extractor())

        self.assertTrue(any("not a mode this app knows" in line for line in logged.output))

    def test_sandbox_mode_without_a_runner_is_in_process_and_says_why(self) -> None:
        with (
            patch.object(config, "EXTRACTOR_MODE", "sandbox"),
            patch.object(config, "EXTRACTOR_RUNNER", ""),
            patch("app.error_report.ensure_request_id", return_value="req-2"),
        ):
            with self.assertLogs("app.error_report", level="WARNING") as logged:
                self.assertIsNone(extractors.build_extractor())

        self.assertTrue(any("VA_LSE_EXTRACTOR_RUNNER" in line for line in logged.output))

    def test_sandbox_mode_with_a_runner_builds_the_fail_open_extractor(self) -> None:
        with (
            patch.object(config, "EXTRACTOR_MODE", "sandbox"),
            patch.object(config, "EXTRACTOR_RUNNER", "box-run {work}"),
        ):
            extractor = extractors.build_extractor()

        self.assertIsInstance(extractor, extractors.FailOpenExtractor)
        self.assertIsInstance(extractor.box, extractors.SandboxExtractor)

    def test_installing_it_swaps_what_the_uploader_uses(self) -> None:
        """The swap, at the seam the views actually call."""
        class Stub:
            def __init__(self) -> None:
                self.labels: list[str] = []

            def extract(self, label: str, data: bytes) -> tuple[list[Any], list[str]]:
                self.labels.append(label)
                return [], []

        stub = Stub()
        with patch.object(extractors, "build_extractor", return_value=stub):
            active = extractors.install_configured_extractor()

        self.assertIs(active, stub)
        self.assertIs(documents_mod.active_extractor(), stub)
        documents_mod.extract_uploaded_documents([_FakeUpload("records.pdf", b"x")])
        self.assertEqual(stub.labels, ["records.pdf"])

    def test_restoring_the_default_gives_the_in_process_reader_back(self) -> None:
        class Stub:
            def extract(self, label: str, data: bytes) -> tuple[list[Any], list[str]]:
                raise AssertionError("should not be called after restoring")

        documents_mod.set_active_extractor(Stub())
        self.assertIsInstance(documents_mod.set_active_extractor(None), documents_mod.InProcessExtractor)


class _FakeUpload:
    """The ``UploadedFile`` surface the uploader needs."""

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self.size = len(data)
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


if __name__ == "__main__":
    unittest.main()
