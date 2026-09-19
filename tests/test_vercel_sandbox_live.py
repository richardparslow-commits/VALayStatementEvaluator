"""Live check of the Vercel Sandbox path — opt-in, skipped without a credential.

`tests/test_vercel_sandbox_runner.py` fakes the CLI, so it pins this repository's half
of the contract but says nothing about the CLI's: a real ``create``'s name semantics, a
real ``copy``'s path rules, whether ``remove`` accepts the auth flags the runner passes.
Those were wrong twice in development, and only a real account can settle them. So this
runs the runner's own ``SandboxCli`` — the same argv, the same stages — against Vercel:

    VA_LSE_TEST_VERCEL_SANDBOX_TOKEN=vcp_... \
    VA_LSE_TEST_VERCEL_SANDBOX_SCOPE=<team> VA_LSE_TEST_VERCEL_SANDBOX_PROJECT=<project> \
    [VA_LSE_TEST_VERCEL_SANDBOX_CLI='npx -y sandbox'] \
    python -m unittest tests.test_vercel_sandbox_live

The knobs arrive under the ``VA_LSE_TEST_*`` prefix and are injected as the app's own
names, because ``tests/hermetic.py`` strips ambient ``VA_LSE_*`` configuration (that
prefix is the session's one sanctioned exception for a runner opting a test into real
setup). It skips without the token, so CI is unaffected. Two tests, priced accordingly:

* the first only needs a credential. It creates a box on the CLI's **default runtime**
  (``VA_LSE_SANDBOX_IMAGE=none``), makes a directory, copies a file in and back out,
  and removes the box — the credential, the transport and the lifecycle, for a fraction
  of a cent and no registry write.
* the second runs the whole runner (``run_for_one_file``) so the entrypoint reads a real
  staged record on a real box. It skips, naming the build command, while the custom
  image is not in the registry — which is the state of a fresh clone, because the image
  is built by hand (DEPLOYMENT.md §6).

Neither test asserts a *result* it cannot stand behind: the box's answer is compared
against ``InProcessExtractor``, exactly as the offline parity test does, so a difference
is a finding rather than a tolerance.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
import uuid
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path, PurePosixPath
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from scripts import vercel_sandbox_runner as runner  # noqa: E402

TOKEN_ENV = "VA_LSE_TEST_VERCEL_SANDBOX_TOKEN"
RUNNER = PROJECT_ROOT / "scripts" / "vercel_sandbox_runner.py"
TYPED = (
    "Knee pain noted on examination, December 2024. Range of motion 100 degrees, "
    "painful motion, no ankylosis reported by the examiner."
)

#: How the CLI reports an image that is not in the registry (measured: a bare
#: ``404 Not Found``, with "Image not found" left in its response buffer). This is how
#: the test knows to skip rather than fail — an unpushed image is a deployment step,
#: not a bug in the runner.
IMAGE_MISSING = ("not in the registry", "404", "image not found")


def _pdf_bytes(page: str) -> bytes:
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(72, 720, page)
    pdf.save()
    return buffer.getvalue()


#: Live knobs are read under the hermetic session's opt-in prefix and mapped here;
#: the sandbox runner's own names are stripped from the ambient environment.
KNOBS = (
    ("VA_LSE_TEST_VERCEL_SANDBOX_CLI", "VA_LSE_SANDBOX_CLI"),
    ("VA_LSE_TEST_VERCEL_SANDBOX_SCOPE", "VA_LSE_SANDBOX_SCOPE"),
    ("VA_LSE_TEST_VERCEL_SANDBOX_PROJECT", "VA_LSE_SANDBOX_PROJECT"),
)


class LiveSandboxTestCase(unittest.TestCase):
    """Real CLI, real account, one box per test, always removed."""

    def setUp(self) -> None:
        self.token = (os.getenv(TOKEN_ENV) or "").strip()
        if not self.token:
            self.skipTest(f"set {TOKEN_ENV} to check the real Vercel Sandbox CLI")
        self.overrides = {
            "VA_LSE_SANDBOX_TOKEN": self.token,
            "VA_LSE_SANDBOX_IMAGE": "none",  # default runtime unless a test says otherwise
            **{
                app_name: value
                for test_name, app_name in KNOBS
                if (value := (os.getenv(test_name) or "").strip())
            },
        }
        self.settings = self._settings()
        self.boxes: list[str] = []
        self.addCleanup(self._remove_leftovers)

    def _settings(self, **overrides: str) -> runner.Settings:
        with patch.dict(os.environ, {**self.overrides, **overrides}, clear=False):
            return runner.Settings.from_env()

    def _box(self, settings: runner.Settings | None = None) -> tuple[str, runner.SandboxCli]:
        settings = settings or self.settings
        name = f"va-lse-live-{uuid.uuid4().hex[:10]}"
        cli = runner.SandboxCli(settings)
        cli.create(name)
        self.boxes.append(name)
        return name, cli

    def _remove_leftovers(self) -> None:
        """Nothing outlives a failed assertion: a leaked microVM bills to its timeout."""
        for name in self.boxes:
            try:
                runner.SandboxCli(self.settings).remove(name)
            except runner.SandboxError as exc:  # pragma: no cover - cleanup of cleanup
                print(f"could not remove {name}: {exc}", file=sys.stderr)


class TestACredentialAndABox(LiveSandboxTestCase):
    def test_a_file_round_trips_through_a_real_box(self) -> None:
        name, cli = self._box()
        with tempfile.TemporaryDirectory(prefix="va-lse-live-") as tmp:
            source = Path(tmp) / "record.txt"
            source.write_text(TYPED, encoding="utf-8")
            back = Path(tmp) / "back.txt"
            remote = PurePosixPath("/work/bundle/note.txt")

            cli.mkdir(name, remote.parent)
            cli.copy(str(source), f"{name}:{remote}", "copy in")
            cli.copy(f"{name}:{remote}", str(back), "copy out")

            self.assertEqual(back.read_text(encoding="utf-8"), TYPED)

        cli.remove(name)
        self.boxes.remove(name)


class TestTheRunnerAgainstARealBox(LiveSandboxTestCase):
    def test_the_entrypoint_reads_a_staged_record_on_a_real_box(self) -> None:
        import hashlib

        from app.documents import InProcessExtractor
        from app.job_payload import documents_from_json

        settings = self._settings(VA_LSE_SANDBOX_IMAGE=runner.DEFAULT_IMAGE)
        with tempfile.TemporaryDirectory(prefix="va-lse-live-") as tmp:
            work = Path(tmp) / "work"
            work.mkdir()
            data = _pdf_bytes(TYPED)
            record = work / "progress_note.pdf"
            record.write_bytes(data)
            (work / "manifest.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "file": {
                            "label": "progress_note.pdf",
                            "path": str(record),
                            "sha256": hashlib.sha256(data).hexdigest(),
                            "size": len(data),
                            "blob_key": None,
                        },
                    }
                ),
                encoding="utf-8",
            )

            out, err = io.StringIO(), io.StringIO()
            try:
                with redirect_stdout(out), redirect_stderr(err):
                    report_text = runner.run_for_one_file(work, settings)
            except runner.SandboxError as exc:
                if any(marker in str(exc).lower() for marker in IMAGE_MISSING):
                    self.skipTest(f"{runner.DEFAULT_IMAGE} is not in the registry yet: {exc}")
                raise
            finally:
                self._remove_leftovers()
                self.boxes.clear()

        expected, _skipped = InProcessExtractor().extract("progress_note.pdf", data)
        from_box = documents_from_json(json.loads(report_text)["documents"])

        self.assertEqual([doc.filename for doc in from_box], [doc.filename for doc in expected])
        self.assertEqual([doc.full_text for doc in from_box], [doc.full_text for doc in expected])


class TestTheAppsExtractorAgainstARealBox(LiveSandboxTestCase):
    """The whole path, live: ``app/extractors.py`` stages the record, the runner boots a
    box from the pushed image, the entrypoint reads it, and the report is mapped back
    through the app's own document JSON. The offline twin of this test fakes the CLI, so
    this is the only place that contract meets a real microVM.

    ``SandboxExtractor`` is fail-open, so a silent fallback would make this pass while
    proving nothing: the assertion is that no ``extractor_sandbox`` failure was recorded,
    and an image that is not in the registry skips rather than fails — the same condition
    the runner's own live test skips on.
    """

    def test_a_record_is_read_on_the_box_and_mapped_back(self) -> None:
        from app import error_report
        from app.documents import InProcessExtractor
        from app.extractors import CommandBoxRunner, SandboxExtractor

        label, data = "progress_note.pdf", _pdf_bytes(TYPED)
        settings = self._settings(VA_LSE_SANDBOX_IMAGE=runner.DEFAULT_IMAGE)
        error_report._ONCE_SEEN.clear()
        self.addCleanup(error_report._ONCE_SEEN.clear)

        with patch.dict(
            os.environ, {**self.overrides, "VA_LSE_SANDBOX_IMAGE": settings.image}, clear=False
        ):
            box = SandboxExtractor(
                CommandBoxRunner(f"{sys.executable} {RUNNER} {{work}}")
            )
            documents, skipped = box.extract(label, data)

        failures = [
            message
            for phase, message in error_report._ONCE_SEEN
            if phase == "extractor_sandbox"
        ]
        if any(marker in " ".join(failures).lower() for marker in IMAGE_MISSING):
            self.skipTest(f"{settings.image} is not in the registry yet: {failures[0]}")
        self.assertEqual(
            failures, [], "the app fell back to the in-process reader instead of the box"
        )

        expected, _expected_skipped = InProcessExtractor().extract(label, data)
        self.assertEqual([doc.filename for doc in documents], [doc.filename for doc in expected])
        self.assertEqual([doc.full_text for doc in documents], [doc.full_text for doc in expected])
        self.assertEqual(skipped, [])


if __name__ == "__main__":
    unittest.main()
