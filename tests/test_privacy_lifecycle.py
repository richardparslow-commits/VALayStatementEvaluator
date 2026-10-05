"""Synthetic privacy canaries: logging fanout and owned OCR scratch lifetime."""
from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import hermetic  # noqa: F401
from app import documents, medical_review, pilot, privacy_logging, synthetic_tools
from scripts import ocr_and_extract, ocr_records

CANARY = "SYNTHETIC_ONLY_PRIVACY_CANARY_92714"


class PrivacyLoggingTests(unittest.TestCase):
    def setUp(self):
        privacy_logging.install()
        self.stream = io.StringIO()
        self.logger = logging.getLogger("privacy.synthetic.thirdparty")
        self.previous = (self.logger.handlers[:], self.logger.propagate, self.logger.level)
        self.logger.handlers = [logging.StreamHandler(self.stream)]
        self.logger.propagate = False
        self.logger.setLevel(logging.DEBUG)
        self.addCleanup(self.restore)

    def restore(self):
        self.logger.handlers, self.logger.propagate, self.logger.level = self.previous

    def test_messages_args_exceptions_and_dynamic_names_are_suppressed(self):
        with patch.object(pilot, "enabled", return_value=True):
            try:
                raise ValueError(CANARY)
            except ValueError:
                self.logger.exception("request %s", CANARY, extra={"source": CANARY})
        self.assertNotIn(CANARY, self.stream.getvalue())
        self.assertEqual(self.stream.getvalue().strip(), "Application event")

    def test_root_and_later_handlers_receive_only_sanitized_fields(self):
        records = []
        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record.__dict__.copy())
        root = logging.getLogger()
        handler = Capture()
        root.addHandler(handler)
        self.addCleanup(root.removeHandler, handler)
        self.logger.handlers.append(Capture())
        with patch.object(pilot, "enabled", return_value=True):
            self.logger.warning(CANARY, extra={"record_pages": 3, "request_id": "req_123456789abc",
                                               CANARY: CANARY, "audit_payload": {"source": CANARY, "record_files": 2}})
            root.warning(CANARY)
        self.assertEqual(len(records), 2)
        self.assertNotIn(CANARY, repr(records))
        self.assertEqual(records[0]["record_pages"], 3)
        self.assertEqual(records[0]["request_id"], "req_123456789abc")
        self.assertEqual(records[0]["audit_payload"]["record_files"], 2)
        self.assertIsNone(records[0]["exc_info"])
        self.assertEqual(records[0]["name"], "pilot")

    def test_raw_objects_are_never_formatted_and_bad_metadata_fails_closed(self):
        class Dangerous:
            def __str__(self):
                raise AssertionError("must not format raw source objects")
        with patch.object(pilot, "enabled", return_value=True):
            self.logger.warning(Dangerous(), extra={"request_id": Dangerous()})
            with patch.object(pilot, "safe_metadata", side_effect=ValueError(CANARY)):
                self.logger.warning(CANARY)
        self.assertNotIn(CANARY, self.stream.getvalue())
        self.assertEqual(self.stream.getvalue().count("Application event"), 2)

    def test_synthetic_logging_keeps_existing_diagnostics(self):
        with patch.object(pilot, "enabled", return_value=False):
            self.logger.warning("synthetic %s", CANARY)
        self.assertIn(CANARY, self.stream.getvalue())

    def test_count_fields_refuse_objects_nonfinite_values_and_foreign_references(self):
        class Number(int):
            pass
        safe = pilot.safe_metadata({"pages": Number(3), "facts": float("nan"),
                                    "calls": float("inf"), "request_id": object(),
                                    "record_files": 2, "status": CANARY})
        self.assertEqual(safe, {"record_files": 2})


class SourceLifetimeTests(unittest.TestCase):
    def test_case_clear_does_not_retain_pilot_text_or_clear_another_synthetic_cache(self):
        import streamlit as st
        medical_review._tokens_cached.cache_clear()
        self.addCleanup(medical_review._tokens_cached.cache_clear)
        with patch.object(documents, "_PARAGRAPH_CACHE", documents.OrderedDict()):
            with patch.object(pilot, "enabled", return_value=False):
                documents.paragraph_index(documents.document_from_text("other.txt", "Unrelated synthetic record."))
                medical_review._tokens("Unrelated synthetic record.")
            with patch.object(pilot, "enabled", return_value=True), patch.object(st, "session_state", {"statement": CANARY}), \
                    patch("streamlit.runtime.scriptrunner_utils.script_run_context.get_script_run_ctx", return_value=None):
                for text in (CANARY, CANARY + " second case"):
                    documents.paragraph_index(documents.document_from_text("case.txt", text))
                    medical_review._tokens(text)
                pilot.clear_case()
                self.assertFalse(st.session_state)
            self.assertEqual(len(documents._PARAGRAPH_CACHE), 1)
            self.assertNotIn(CANARY, repr(documents._PARAGRAPH_CACHE))
            self.assertEqual(medical_review._tokens_cached.cache_info().currsize, 1)


class StandalonePrivacyTests(unittest.TestCase):
    def test_class_declaration_is_required_before_any_file_access(self):
        for tool in (ocr_records, ocr_and_extract):
            with self.subTest(tool=tool.__name__), patch.object(Path, "exists") as exists, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    tool.main(["source.pdf"])
                exists.assert_not_called()

    def test_sensitive_declaration_and_pilot_mode_refuse_before_file_access(self):
        for tool in (ocr_records, ocr_and_extract):
            for classification, enabled in (("sensitive", False), ("synthetic", True)):
                with self.subTest(tool=tool.__name__, classification=classification, enabled=enabled), \
                        patch.object(pilot, "enabled", return_value=enabled), \
                        patch.object(Path, "exists") as exists, patch.object(Path, "is_file") as is_file, \
                        contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(tool.main(["--data-class", classification, CANARY + ".pdf"]), 3)
                    exists.assert_not_called()
                    is_file.assert_not_called()

    def test_programmatic_sensitive_routes_refuse_before_read_or_write(self):
        with patch.object(pilot, "enabled", return_value=True), patch.object(Path, "read_bytes") as read, \
                patch.object(Path, "write_bytes") as write:
            for call in (lambda: ocr_records.inspect_pdf(Path("source.pdf")),
                         lambda: ocr_and_extract.process([], roots=[], work_dir=Path("scratch")),
                         lambda: ocr_and_extract.prepare("source.pdf", b"data", work_dir=Path("scratch"), dpi=300, use_ocr=True)):
                with self.assertRaises(ValueError):
                    call()
            read.assert_not_called(); write.assert_not_called()

    def test_scratch_is_private_and_removed_on_success_exception_and_system_exit(self):
        with tempfile.TemporaryDirectory() as parent:
            for exception in (None, RuntimeError, SystemExit):
                path = None
                try:
                    with synthetic_tools.temporary_work(Path(parent)) as path:
                        self.assertEqual(path.stat().st_mode & 0o777, 0o700)
                        (path / "source.txt").write_text(CANARY)
                        if exception:
                            raise exception()
                except (RuntimeError, SystemExit):
                    pass
                self.assertIsNotNone(path)
                self.assertFalse(path.exists())
            self.assertEqual(list(Path(parent).iterdir()), [])

    def test_bundle_output_is_private_atomic_and_scratch_is_not_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source.txt"; destination = root / "bundle.json"; scratch = root / "scratch"
            source.write_text(CANARY)
            with contextlib.redirect_stdout(io.StringIO()):
                result = ocr_and_extract.main(["--data-class", "synthetic", str(source), "--out", str(destination), "--work-dir", str(scratch)])
            self.assertEqual(result, 0)
            self.assertEqual(destination.stat().st_mode & 0o777, 0o600)
            self.assertFalse(json.loads(destination.read_text())["work_artifacts_retained"])
            self.assertEqual(list(scratch.iterdir()), [])
            self.assertEqual(source.read_text(), CANARY)

    def test_failed_processing_removes_owned_scratch_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source.txt"; output = root / "bundle.json"; scratch = root / "scratch"
            source.write_text(CANARY); output.write_text("previous complete output")
            def fail(*args, **kwargs):
                (kwargs["work_dir"] / "copy.txt").write_text(CANARY)
                raise ValueError(CANARY)
            with patch.object(ocr_and_extract, "process", side_effect=fail), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
                result = ocr_and_extract.main(["--data-class", "synthetic", str(source), "--out", str(output), "--force", "--work-dir", str(scratch)])
            self.assertEqual(result, 3)
            self.assertEqual(output.read_text(), "previous complete output")
            self.assertEqual(list(scratch.iterdir()), [])
            self.assertNotIn(CANARY, errors.getvalue())

    def test_symlink_destinations_and_source_aliases_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/"source.txt"; link=root/"link.txt"
            source.write_text(CANARY); link.symlink_to(source)
            with self.assertRaises(ValueError):
                synthetic_tools.write_private(link, b"replacement")
            for output in (source, link):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(ocr_and_extract.main(["--data-class", "synthetic", str(source), "--out", str(output), "--force"]), 3)
            self.assertEqual(source.read_text(), CANARY)

    def test_cleanup_failure_blocks_publication_and_preserves_prior_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/"source.txt"; output=root/"bundle.json"; scratch=root/"scratch"
            source.write_text(CANARY); output.write_text("previous complete bundle")
            @contextlib.contextmanager
            def failed_cleanup(parent):
                scratch.mkdir(mode=0o700)
                yield scratch
                (scratch/"retained-copy.txt").write_text(CANARY)
                raise OSError("synthetic cleanup failure")
            with patch.object(ocr_and_extract, "temporary_work", failed_cleanup), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result=ocr_and_extract.main(["--data-class", "synthetic", str(source), "--out", str(output), "--force"])
            self.assertEqual(result, 3)
            self.assertEqual(output.read_text(), "previous complete bundle")
            self.assertTrue((scratch/"retained-copy.txt").exists())

    def test_nonoverwrite_publication_refuses_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"complete.json"; output.write_text("previous")
            with self.assertRaises(FileExistsError):
                synthetic_tools.write_private(output, b"new", overwrite=False)
            self.assertEqual(output.read_text(), "previous")
            self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_tool_error_never_echoes_subprocess_output(self):
        result = type("Result", (), {"returncode": 1, "stderr": CANARY, "stdout": CANARY})()
        with patch("app.child_process.run_bounded", return_value=result):
            with self.assertRaises(RuntimeError) as raised:
                ocr_records._run(["synthetic-tool"], what="OCR")
        self.assertNotIn(CANARY, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
