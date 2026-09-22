"""Tests for scripts/vercel_sandbox_runner.py, with the Sandbox CLI faked.

The real Vercel Sandbox CLI is not installed in CI and there is no account to point
it at, so the binary the runner invokes is ``tests/fake_sandbox_cli.py``:
``create``/``exec``/``copy``/``remove`` run against a directory standing in for the
microVMs, and the exec step runs the *real* ``scripts/ocr_and_extract.py`` over the
file the runner actually copied in. Everything on this side of the CLI boundary is
therefore exercised for real — the manifest, the label-to-path mapping, the exec
argv, the copy-back, the JSON on stdout, the ``finally`` that removes the box — and
one test drives the whole thing through ``app/extractors.py`` so the contract the
app parses is the contract the runner prints.

What the fake cannot prove is the CLI's own behavior (a real ``create``'s name
semantics, a real ``copy``'s path rules). Those are pinned to the published CLI
reference; the point here is that a change to *this* repository's half of the
contract fails a test rather than a run.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app import error_report  # noqa: E402
from app.documents import InProcessExtractor  # noqa: E402
from app.extractors import CommandBoxRunner, SandboxExtractor, SandboxUnavailable  # noqa: E402
from scripts import vercel_sandbox_runner as runner  # noqa: E402

FAKE_CLI = PROJECT_ROOT / "tests" / "fake_sandbox_cli.py"
RUNNER = PROJECT_ROOT / "scripts" / "vercel_sandbox_runner.py"

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


def _zip_bytes(data: bytes, *, member: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(member, data)
    return buffer.getvalue()


class RunnerTestCase(unittest.TestCase):
    """One staged file, one fake box, one journal of what the runner did."""

    def setUp(self) -> None:
        tmp = tempfile.mkdtemp(prefix="va-lse-vercel-runner-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        self.tmp = Path(tmp)
        self.work = self.tmp / "work"
        self.work.mkdir()
        self.box = self.tmp / "box"
        self.box.mkdir()
        self.journal = self.tmp / "journal.jsonl"
        # The runner reads these from the environment at call time, and so does
        # the fake it invokes; the app-level tests pass the same set through
        # CommandBoxRunner's process environment.
        self.env = {
            "VA_LSE_SANDBOX_CLI": f"{shlex.quote(sys.executable)} {shlex.quote(str(FAKE_CLI))}",
            "FAKE_SANDBOX_BOX": str(self.box),
            "FAKE_SANDBOX_JOURNAL": str(self.journal),
            "FAKE_SANDBOX_PROJECT": str(PROJECT_ROOT),
        }
        # report_failure(once=True) remembers (phase, message) for the process,
        # and the fail-open test asserts the app speaks up — so start clean.
        error_report._ONCE_SEEN.clear()
        self.addCleanup(error_report._ONCE_SEEN.clear)

    # -- staging -------------------------------------------------------------
    def stage(self, label: str, data: bytes, **overrides: Any) -> Path:
        """Write what app/extractors.py stages: manifest.json plus the file."""
        path = self.work / (Path(label.replace("\\", "/")).name or "record")
        path.write_bytes(data)
        manifest = {
            "version": 1,
            "file": {
                "label": label,
                "path": str(path),
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
                "blob_key": None,
                **overrides,
            },
        }
        (self.work / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path

    # -- running -------------------------------------------------------------
    def command(self) -> str:
        """The operator's command, as DEPLOYMENT.md documents it."""
        return f"{shlex.quote(sys.executable)} {shlex.quote(str(RUNNER))} {{work}}"

    def run_runner(self, env: dict[str, str] | None = None) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, {**self.env, **(env or {})}, clear=False), redirect_stdout(
            out
        ), redirect_stderr(err):
            code = runner.main([str(self.work)])
        return code, out.getvalue(), err.getvalue()

    def run_check(
        self, env: dict[str, str] | None = None, argv: list[str] | None = None
    ) -> tuple[int, str, str]:
        """``--check``, which creates nothing — so its stdout is a verdict, not a report."""
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, {**self.env, **(env or {})}, clear=False), redirect_stdout(
            out
        ), redirect_stderr(err):
            code = runner.main(argv if argv is not None else ["--check"])
        return code, out.getvalue(), err.getvalue()

    def verdict(self, env: dict[str, str] | None = None) -> tuple[int, dict[str, Any], str]:
        code, out, err = self.run_check(env)
        return code, json.loads(out) if out.strip() else {}, err

    def check_named(self, payload: dict[str, Any], name: str) -> dict[str, Any]:
        for check in payload.get("checks", []):
            if check["name"] == name:
                return check
        raise AssertionError(f"no {name} check in {[c['name'] for c in payload.get('checks', [])]}")

    def calls(self) -> list[dict[str, Any]]:
        if not self.journal.exists():
            return []
        return [
            json.loads(line)
            for line in self.journal.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def steps(self) -> list[str]:
        return [str(call.get("step") or call["command"]) for call in self.calls()]

    def call_for(self, step: str) -> dict[str, Any]:
        for call in self.calls():
            if call.get("step") == step or call["command"] == step:
                return call
        raise AssertionError(f"no {step} call in {self.steps()}")


class TestOneFileThroughTheBox(RunnerTestCase):
    def test_stdout_is_the_report_and_nothing_else(self) -> None:
        """The app parses the runner's *whole* stdout as the report, so any
        progress line on stdout would be an unparseable answer."""
        label = "progress_note.pdf"
        self.stage(label, _pdf_bytes([TYPED]))

        code, out, err = self.run_runner()

        self.assertEqual(code, 0, err)
        report = json.loads(out)
        self.assertEqual([entry["filename"] for entry in report["documents"]], [label])
        self.assertIn("Knee pain", json.dumps(report["documents"]))
        self.assertEqual(
            self.steps(), ["create", "mkdir", "copy-in", "entrypoint", "copy-out", "remove"]
        )
        self.assertEqual(list(self.box.iterdir()), [], "the box was not removed")
        self.assertTrue((self.work / "report.json").is_file())

    def test_a_label_with_directories_comes_back_under_the_same_name(self) -> None:
        """The label *is* the path: a citation has to name the file the user has,
        and app/extractors.py refuses any answer under another name."""
        label = "records/2024/visit note.pdf"
        self.stage(label, _pdf_bytes([TYPED]))

        code, out, err = self.run_runner()

        self.assertEqual(code, 0, err)
        report = json.loads(out)
        self.assertEqual([entry["filename"] for entry in report["documents"]], [label])
        self.assertTrue(self.call_for("copy-in")["remote"].endswith(f"/bundle/{label}"))
        self.assertTrue(self.call_for("mkdir")["directory"].endswith("/bundle/records/2024"))

    def test_the_default_runtime_can_be_asked_for_by_name(self) -> None:
        """``VA_LSE_SANDBOX_IMAGE=none`` drops ``--image`` so the CLI boots its own
        runtime — the only way to prove a credential and the create/exec/copy/remove
        cycle when no VCR image has been pushed yet (see the live test)."""
        self.stage("note.pdf", _pdf_bytes([TYPED]))

        code, _out, err = self.run_runner({"VA_LSE_SANDBOX_IMAGE": "none"})

        self.assertEqual(code, 0, err)
        self.assertNotIn("--image", self.call_for("create")["argv"])

    def test_the_box_is_created_from_the_documented_image_and_removed_last(self) -> None:
        self.stage("progress_note.pdf", _pdf_bytes([TYPED]))

        code, _, err = self.run_runner()

        self.assertEqual(code, 0, err)
        create = self.call_for("create")["argv"]
        self.assertEqual(create[create.index("--image") + 1], "va-lse-sandbox:latest")
        self.assertIn("--non-persistent", create)
        self.assertIn("--silent", create)
        self.assertEqual(create[create.index("--timeout") + 1], "20m")

    def test_settings_reach_the_cli_and_the_token_never_reaches_a_log(self) -> None:
        self.stage("progress_note.pdf", _pdf_bytes([TYPED]))

        code, _, err = self.run_runner(
            {
                "VA_LSE_SANDBOX_SCOPE": "my-team",
                "VA_LSE_SANDBOX_PROJECT": "my-project",
                "VA_LSE_SANDBOX_TIMEOUT": "45m",
                "VERCEL_TOKEN": "fake-token-value-9173",
            }
        )

        self.assertEqual(code, 0, err)
        self.assertNotIn("fake-token-value-9173", err, "the token reached the log")
        create = self.call_for("create")["argv"]
        self.assertEqual(create[create.index("--scope") + 1], "my-team")
        self.assertEqual(create[create.index("--project") + 1], "my-project")
        self.assertEqual(create[create.index("--token") + 1], "fake-token-value-9173")
        self.assertEqual(create[create.index("--timeout") + 1], "45m")
        # Cleanup authenticates too: Vercel's reference lists no options for
        # `remove`, but the CLI takes them, and without them a token-authenticated
        # deployment could never remove its own box.
        remove = self.calls()[-1]["argv"]
        self.assertIn("--token", remove)
        self.assertIn("--scope", remove)
        self.assertIn("--project", remove)

    def test_cleanup_that_fails_does_not_fail_the_file(self) -> None:
        """A box nobody removed stops itself at --timeout; the report is already
        in hand, so this is a warning rather than a failed read."""
        self.stage("progress_note.pdf", _pdf_bytes([TYPED]))

        code, out, err = self.run_runner({"FAKE_SANDBOX_FAIL": "remove"})

        self.assertEqual(code, 0, err)
        json.loads(out)
        self.assertIn("stops itself at its --timeout", err)


class TestTheAppReadsThroughTheRunner(RunnerTestCase):
    """The end-to-end path, minus the VM: CommandBoxRunner runs the shipped
    command, the command runs the shipped entrypoint over the copied file."""

    def box_documents(self, label: str, data: bytes) -> tuple[list[Any], list[str]]:
        with patch.dict(os.environ, self.env, clear=False):
            return SandboxExtractor(CommandBoxRunner(self.command())).extract(label, data)

    def test_a_pdf_from_the_box_is_the_pdf_the_app_reads_in_process(self) -> None:
        label, data = "progress_note.pdf", _pdf_bytes([TYPED])
        self.stage(label, data)

        documents, skipped = self.box_documents(label, data)

        expected, expected_skipped = InProcessExtractor().extract(label, data)
        self.assertEqual(skipped, expected_skipped)
        self.assertEqual([d.filename for d in documents], [d.filename for d in expected])
        self.assertEqual([d.full_text for d in documents], [d.full_text for d in expected])
        self.assertEqual(
            [d.source_page_count for d in documents],
            [d.source_page_count for d in expected],
        )

    def test_an_uploaded_zip_is_answered_for_its_members(self) -> None:
        """The member labels the box answers with are the ones
        app.documents.archive_members produces in-process, and the app must
        accept them — a mismatch here means every archive silently falls back."""
        label = "records.zip"
        data = _zip_bytes(TYPED.encode(), member="2024/visit.txt")
        self.stage(label, data)

        documents, skipped = self.box_documents(label, data)

        expected, expected_skipped = InProcessExtractor().extract(label, data)
        self.assertEqual(skipped, expected_skipped)
        self.assertEqual(
            [d.filename for d in documents], [d.filename for d in expected]
        )
        self.assertEqual([d.filename for d in documents], ["records/2024/visit.txt"])
        self.assertEqual([d.full_text for d in documents], [d.full_text for d in expected])

    def test_a_box_that_cannot_start_is_a_sandbox_unavailable(self) -> None:
        label, data = "progress_note.pdf", _pdf_bytes([TYPED])
        self.stage(label, data)

        with patch.dict(os.environ, {**self.env, "FAKE_SANDBOX_FAIL": "create"}, clear=False):
            with self.assertRaises(SandboxUnavailable) as caught:
                SandboxExtractor(CommandBoxRunner(self.command())).extract(label, data)

        self.assertIn("failed on purpose", str(caught.exception))
        self.assertIn("create va-lse-ocr-", str(caught.exception))


class TestRefusals(RunnerTestCase):
    def test_the_boxes_own_sentence_is_forwarded_even_when_the_read_fails(self) -> None:
        """The diagnosis of a failed read *is* the box's output, so it has to reach
        stderr before the exit code is judged."""
        self.stage("scan.pdf", _pdf_bytes([TYPED]))

        code, _out, err = self.run_runner({"FAKE_SANDBOX_FAIL": "entrypoint"})

        self.assertEqual(code, 1)
        self.assertIn("  | ✖ Scans are present and no OCR tooling is installed", err)

    def test_a_box_that_cannot_be_created_never_fails_the_file_itself(self) -> None:
        self.stage("progress_note.pdf", _pdf_bytes([TYPED]))

        code, out, err = self.run_runner({"FAKE_SANDBOX_FAIL": "create"})

        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("create va-lse-ocr-", err)
        self.assertIn("failed on purpose", err)
        self.assertEqual(self.steps(), ["create"], "nothing was created, so nothing to remove")

    def test_a_refused_file_keeps_the_boxes_own_sentence_and_still_removes_it(self) -> None:
        self.stage("scan.pdf", _pdf_bytes([TYPED]))

        code, out, err = self.run_runner({"FAKE_SANDBOX_FAIL": "entrypoint"})

        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("no OCR tooling is installed", err)
        self.assertEqual(self.steps()[-1], "remove", "a failed read must not leak the box")
        self.assertEqual(list(self.box.iterdir()), [])

    def test_a_missing_cli_names_the_binary_to_install(self) -> None:
        self.stage("progress_note.pdf", _pdf_bytes([TYPED]))

        code, out, err = self.run_runner({"VA_LSE_SANDBOX_CLI": "no-such-sandbox-cli-9173"})

        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("no-such-sandbox-cli-9173", err)
        self.assertIn("npm i -g sandbox", err)

    def test_a_report_that_is_not_the_report_is_refused(self) -> None:
        self.stage("progress_note.pdf", _pdf_bytes([TYPED]))

        for corrupt, expected in (("garbage", "not JSON"), ("no-documents", "no 'documents' list")):
            with self.subTest(report=corrupt):
                code, out, err = self.run_runner({"FAKE_SANDBOX_REPORT": corrupt})
                self.assertEqual(code, 1)
                self.assertEqual(out, "")
                self.assertIn(expected, err)
                self.assertEqual(self.steps()[-1], "remove")

    def test_a_missing_or_mismatched_manifest_is_refused_before_a_box_exists(self) -> None:
        code, out, err = self.run_runner()

        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("manifest.json is missing", err)
        self.assertEqual(self.calls(), [])

        self.stage("progress_note.pdf", _pdf_bytes([TYPED]), sha256="0" * 64)
        code, out, err = self.run_runner()

        self.assertEqual(code, 1)
        self.assertIn("does not match the sha256", err)
        self.assertEqual(self.calls(), [])

    def test_a_label_that_climbs_out_of_the_bundle_is_refused(self) -> None:
        self.stage("../../etc/passwd", b"not a record")

        code, out, err = self.run_runner()

        self.assertEqual(code, 1)
        self.assertIn("climbs out of the bundle", err)
        self.assertEqual(self.calls(), [], "a refused label must not boot a box")


class TestATerminatedRunnerStillRemovesTheBox(RunnerTestCase):
    def test_sigterm_runs_the_finally(self) -> None:
        """A user Cancel must not leave a microVM billing: the app kills the
        runner, and the runner's cleanup still runs while the box is mid-read."""
        self.stage("progress_note.pdf", _pdf_bytes([TYPED]))
        env = {**os.environ, **self.env, "FAKE_SANDBOX_HOLD": "entrypoint"}
        proc = subprocess.Popen(
            [sys.executable, str(RUNNER), str(self.work)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        self.addCleanup(self._kill_group, proc)

        holding = self.box / "holding"
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if holding.exists():
                break
            if proc.poll() is not None:
                self.fail(f"the runner exited early: {proc.communicate()[1]}")
            time.sleep(0.05)
        else:
            self.fail("the runner never reached the entrypoint step")

        proc.send_signal(signal.SIGTERM)
        _, err = proc.communicate(timeout=30)

        self.assertEqual(proc.returncode, 143, err)
        self.assertEqual(self.steps()[-1], "remove")
        self.assertEqual(
            list(self.box.glob("va-lse-ocr-*")), [], "the box survived the SIGTERM"
        )

    #: The fake's held step is still polling after the runner exits; killing the
    #: session the runner started takes the orphan with it.
    @staticmethod
    def _kill_group(proc: subprocess.Popen[str]) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


class TestTheFailureDetail(unittest.TestCase):
    """What an operator reads when the box says no.

    The real CLI appends noise after the reason — a failed ``create`` ends with
    ``╰▶ hint: the full response buffer is stored in /tmp/…`` — so the last line is
    not the reason, and a runner that quotes it hides the actual error (measured
    against 4.4.0: an unpushed image reported a temp path instead of "Image not
    found").
    """

    def _detail(self, stdout: str = "", stderr: str = "") -> str:
        return runner._detail(
            subprocess.CompletedProcess(args=["sandbox"], returncode=1, stdout=stdout, stderr=stderr)
        )

    def test_a_trailing_hint_does_not_hide_the_reason(self) -> None:
        stderr = (
            '{"error":{"message":"Image not found"}}\n'
            "╰▶ hint: the full response buffer is stored in /tmp/sandbox-cli-response-x.http\n"
        )

        self.assertIn("Image not found", self._detail(stderr=stderr))

    def test_stderr_is_read_before_stdout(self) -> None:
        self.assertIn("the reason", self._detail(stdout="stdout noise", stderr="the reason"))

    def test_a_quiet_failure_still_says_something(self) -> None:
        self.assertEqual(self._detail(), "no output")

    def test_a_404_names_the_step_that_fixes_it(self) -> None:
        """The image is built by hand, so this is the likeliest first failure — and the
        CLI reports it as a bare status, which says nothing about what to do."""
        settings = runner.Settings(
            cli=("sandbox",), image="va-lse-sandbox:latest", timeout="20m", scope="", project="", token=""
        )
        cli = runner.SandboxCli(settings)

        advice = cli._image_advice(runner.SandboxError("create x failed (exit 1): status 404"))

        self.assertIn("not in the registry", advice)
        self.assertIn("DEPLOYMENT.md", advice)
        self.assertIn("VA_LSE_SANDBOX_IMAGE=none", advice)

    def test_advice_is_silent_for_other_failures_and_for_the_default_runtime(self) -> None:
        settings = runner.Settings(
            cli=("sandbox",), image="va-lse-sandbox:latest", timeout="20m", scope="", project="", token=""
        )
        probe = runner.Settings(
            cli=("sandbox",), image="none", timeout="20m", scope="", project="", token=""
        )

        self.assertEqual(runner.SandboxCli(settings)._image_advice(runner.SandboxError("boom")), "")
        self.assertEqual(
            runner.SandboxCli(probe)._image_advice(runner.SandboxError("404 not found")), ""
        )

    def test_noise_with_nothing_else_falls_back_to_it(self) -> None:
        """Better to show the hint than to show nothing at all."""
        self.assertIn("hint:", self._detail(stderr="╰▶ hint: the buffer is in /tmp/x.http"))


class TestVercelSandboxCredentials(RunnerTestCase):
    """Sandbox auth is a Vercel access token or a Function's OIDC token — never the
    AI Gateway key, which is the app's LLM credential for a different product."""

    def token_env(self, **values: str) -> dict[str, str]:
        """Every credential name, explicitly: an ambient VERCEL_* must not decide a test.

        The precedence here is measured against the CLI rather than guessed from
        the docs — ``VERCEL_AUTH_TOKEN`` is the name the Sandbox CLI itself reads,
        and it does not read ``VERCEL_TOKEN`` at all.
        """
        env = {
            name: ""
            for name in (
                "VA_LSE_SANDBOX_TOKEN",
                "VERCEL_AUTH_TOKEN",
                "VERCEL_OIDC_TOKEN",
                "VERCEL_TOKEN",
            )
        }
        env.update(values)
        return env

    def token_used(self) -> str | None:
        argv = self.call_for("create")["argv"]
        return argv[argv.index("--token") + 1] if "--token" in argv else None

    def test_a_gateway_key_in_the_token_slot_is_refused_by_name(self) -> None:
        self.stage("progress_note.pdf", _pdf_bytes([TYPED]))

        code, out, err = self.run_runner(
            self.token_env(VERCEL_TOKEN="vck_example-not-a-sandbox-token")
        )

        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("Vercel AI Gateway key", err)
        self.assertIn("Account Settings", err)
        self.assertEqual(self.calls(), [], "a refused credential must not reach the CLI")

    def test_the_clis_own_variable_name_is_read(self) -> None:
        self.stage("progress_note.pdf", _pdf_bytes([TYPED]))

        code, _, err = self.run_runner(self.token_env(VERCEL_AUTH_TOKEN="auth-token-example"))

        self.assertEqual(code, 0, err)
        self.assertEqual(self.token_used(), "auth-token-example")

    def test_the_oidc_token_stands_in_for_an_access_token(self) -> None:
        """Inside a Function the OIDC token is already there, so a deployment needs no
        long-lived secret."""
        self.stage("progress_note.pdf", _pdf_bytes([TYPED]))

        code, _, err = self.run_runner(self.token_env(VERCEL_OIDC_TOKEN="oidc-example"))

        self.assertEqual(code, 0, err)
        self.assertEqual(self.token_used(), "oidc-example")

    def test_this_apps_own_variable_wins_the_precedence(self) -> None:
        self.stage("progress_note.pdf", _pdf_bytes([TYPED]))

        code, _, err = self.run_runner(
            self.token_env(
                VA_LSE_SANDBOX_TOKEN="explicit-example",
                VERCEL_AUTH_TOKEN="auth-token-example",
                VERCEL_OIDC_TOKEN="oidc-example",
                VERCEL_TOKEN="access-example",
            )
        )

        self.assertEqual(code, 0, err)
        self.assertEqual(self.token_used(), "explicit-example")

    def test_no_credential_leaves_the_clis_own_login_in_charge(self) -> None:
        self.stage("progress_note.pdf", _pdf_bytes([TYPED]))

        code, _, err = self.run_runner(self.token_env())

        self.assertEqual(code, 0, err)
        self.assertIsNone(self.token_used())


class TestBoxFreeCheck(RunnerTestCase):
    """``--check``: what a run would reach, answered without spending a box.

    The promise to an operator is "run this first and it costs nothing", so the
    load-bearing assertion is the journal: the only subcommand a check may reach is
    ``list`` — never ``create``, never an ``exec``, never a ``copy``.
    """

    def test_it_reports_the_cli_the_credential_and_the_image_and_creates_nothing(self) -> None:
        code, payload, err = self.verdict()

        self.assertEqual(code, 0, err)
        self.assertTrue(payload["reachable"])
        self.assertEqual([c["name"] for c in payload["checks"]], ["cli", "credential", "image"])
        self.assertEqual(self.steps(), ["list"], "a check ran something other than list")
        self.assertNotIn("create", json.dumps(self.calls()))

    def test_the_verdict_names_the_slot_the_scope_and_the_project(self) -> None:
        code, payload, err = self.verdict(
            {
                "VA_LSE_SANDBOX_TOKEN": "example-token-value",
                "VA_LSE_SANDBOX_SCOPE": "my-team",
                "VA_LSE_SANDBOX_PROJECT": "my-project",
            }
        )

        self.assertEqual(code, 0, err)
        detail = self.check_named(payload, "credential")["detail"]
        self.assertIn("VA_LSE_SANDBOX_TOKEN", detail)
        self.assertIn("scope my-team", detail)
        self.assertIn("project my-project", detail)
        self.assertNotIn("example-token-value", json.dumps(payload), "the token reached the verdict")

    def test_the_authenticated_question_carries_the_scope_the_project_and_the_token(self) -> None:
        _, _, err = self.verdict(
            {
                "VA_LSE_SANDBOX_TOKEN": "example-token-value",
                "VA_LSE_SANDBOX_SCOPE": "my-team",
                "VA_LSE_SANDBOX_PROJECT": "my-project",
            }
        )

        argv = self.call_for("list")["argv"]
        self.assertEqual(argv[argv.index("--scope") + 1], "my-team")
        self.assertEqual(argv[argv.index("--project") + 1], "my-project")
        self.assertEqual(argv[argv.index("--token") + 1], "example-token-value")
        self.assertNotIn("example-token-value", err, "the token reached the log")

    def test_a_missing_cli_is_reported_with_its_remedy_and_asks_nothing(self) -> None:
        code, payload, err = self.verdict(
            {"VA_LSE_SANDBOX_CLI": "definitely-not-installed-anywhere"}
        )

        self.assertEqual(code, 1, err)
        cli = self.check_named(payload, "cli")
        self.assertEqual(cli["status"], "failed")
        self.assertIn("npm i -g sandbox", cli["remedy"])
        # Nothing is asked of a CLI that is not here: the second question could only
        # repeat the first answer.
        self.assertEqual([c["name"] for c in payload["checks"]], ["cli", "image"])
        self.assertEqual(self.calls(), [])

    def test_a_refused_credential_quotes_the_cli_and_names_what_to_fix(self) -> None:
        code, payload, _err = self.verdict({"FAKE_SANDBOX_FAIL": "list"})

        self.assertEqual(code, 1)
        credential = self.check_named(payload, "credential")
        self.assertEqual(credential["status"], "failed")
        self.assertIn("list failed on purpose", credential["detail"])
        self.assertIn("VA_LSE_SANDBOX_TOKEN", credential["remedy"])
        self.assertIn("AI Gateway key does not authenticate one", credential["remedy"])

    def test_a_cli_waiting_for_a_login_prompt_does_not_hang_the_check(self) -> None:
        """With no credential the CLI prompts to log in; there is no terminal here."""
        settings = runner.Settings(
            cli=("sh", "-c", "sleep 30"),
            image="none",
            timeout="20m",
            scope="",
            project="",
            token="",
        )

        result = runner.self_check(settings, timeout=0.5)

        credential = next(c for c in result.checks if c.name == "credential")
        self.assertEqual(credential.status, "failed")
        self.assertIn("did not answer within", credential.detail)
        self.assertIn("sandbox login", credential.remedy)
        self.assertFalse(result.reachable)

    def test_a_gateway_key_is_refused_by_the_check_too(self) -> None:
        code, out, err = self.run_check({"VERCEL_TOKEN": "vck_example-not-a-sandbox-token"})

        self.assertEqual(code, 1)
        self.assertEqual(out, "", "a refused credential still printed a verdict")
        self.assertIn("Vercel AI Gateway key", err)
        self.assertEqual(self.calls(), [], "a refused credential must not reach the CLI")

    def test_the_image_is_unproven_with_both_ways_to_settle_it(self) -> None:
        _, payload, _err = self.verdict()

        image = self.check_named(payload, "image")
        self.assertEqual(image["status"], "unproven")
        self.assertIn("vercel vcr image ls va-lse-sandbox", image["remedy"])
        self.assertIn("test_vercel_sandbox_live", image["remedy"])

    def test_the_default_runtime_needs_no_registry_image(self) -> None:
        _, payload, _err = self.verdict({"VA_LSE_SANDBOX_IMAGE": "none"})

        image = self.check_named(payload, "image")
        self.assertEqual(image["status"], "ok")
        self.assertIn("default runtime", image["detail"])

    def test_check_wins_over_a_staged_directory_that_is_right_there(self) -> None:
        """A diagnostic asked for must not spend a box because a file was waiting."""
        self.stage("progress_note.pdf", _pdf_bytes([TYPED]))

        code, _out, err = self.run_check(argv=["--check", str(self.work)])

        self.assertEqual(code, 0, err)
        self.assertEqual(self.steps(), ["list"])

    def test_the_staging_directory_is_still_required_for_a_real_run(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            self.run_check(argv=[])

        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
