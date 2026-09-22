"""Tests for scripts/check_sandbox.py — the one-verdict sandbox diagnostic.

The Vercel side is faked (``tests/fake_sandbox_cli.py``) and the app's configuration is
patched, because what this script *is* is a composition: the runner line the app is
configured with, resolved the way a run resolves it, plus the runner's own box-free
probe. Two properties carry the tests:

* **it creates nothing** — the fake CLI journals every call, and the only subcommand a
  check may reach is ``list``. That is the whole promise made to anyone who runs this
  before they trust a box, so it is asserted rather than assumed: with a record already
  staged and waiting, ``--check`` still only lists.
* **it disagrees with nobody** — a bare ``python`` on a host without one is reported as
  the substitution a run would make (not as a missing interpreter), a wrapper runner is
  reported as unproven (not as a broken line), and a line that genuinely cannot run is
  reported once, with the remedy the app would log.
"""
from __future__ import annotations

import io
import json
import os
import shlex
import shutil
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from scripts import check_sandbox  # noqa: E402

FAKE_CLI = PROJECT_ROOT / "tests" / "fake_sandbox_cli.py"
RUNNER = PROJECT_ROOT / "scripts" / "vercel_sandbox_runner.py"

#: A host where a bare ``python`` is not on ``PATH`` — the case the resolver was added
#: for. Only ``app.extractors``'s view of ``shutil`` is replaced, so this script's own
#: lookups (of the interpreter it resolved to) still behave like the real host.
class _NoPythonOnPath:
    @staticmethod
    def which(_name: str) -> None:
        return None


class CheckSandboxTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.mkdtemp(prefix="va-lse-check-sandbox-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        self.tmp = Path(tmp)
        self.box = self.tmp / "box-for-the-fake-cli"
        self.box.mkdir()
        self.journal = self.tmp / "journal.jsonl"
        self.env = {
            "VA_LSE_SANDBOX_CLI": f"{shlex.quote(sys.executable)} {shlex.quote(str(FAKE_CLI))}",
            "FAKE_SANDBOX_BOX": str(self.box),
            "FAKE_SANDBOX_JOURNAL": str(self.journal),
            "FAKE_SANDBOX_PROJECT": str(PROJECT_ROOT),
            "VA_LSE_SANDBOX_TOKEN": "example-token-value",
            "VA_LSE_SANDBOX_SCOPE": "my-team",
            "VA_LSE_SANDBOX_PROJECT": "my-project",
            "VA_LSE_SANDBOX_IMAGE": "va-lse-sandbox:latest",
        }

    # -- running -------------------------------------------------------------
    def documented_line(self) -> str:
        """The command DEPLOYMENT.md shows, with a real interpreter in place of ``python``."""
        return f"{shlex.quote(sys.executable)} {shlex.quote(str(RUNNER))} {{work}}"

    def run_check(
        self,
        *,
        argv: list[str] | None = None,
        env: dict[str, str] | None = None,
        mode: str = "sandbox",
        line: str | None = None,
        shutil_stub: Any = None,
    ) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(os.environ, {**self.env, **(env or {})}, clear=False)
            )
            stack.enter_context(patch.object(check_sandbox.config, "EXTRACTOR_MODE", mode))
            stack.enter_context(
                patch.object(
                    check_sandbox.config,
                    "EXTRACTOR_RUNNER",
                    self.documented_line() if line is None else line,
                )
            )
            if shutil_stub is not None:
                stack.enter_context(
                    patch.object(check_sandbox.extractors, "shutil", shutil_stub)
                )
            stack.enter_context(redirect_stdout(out))
            stack.enter_context(redirect_stderr(err))
            code = check_sandbox.main(list(argv or []))
        return code, out.getvalue(), err.getvalue()

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


class TestOneVerdict(CheckSandboxTestCase):
    def test_a_working_setup_is_one_verdict_and_touches_nothing_but_list(self) -> None:
        code, out, err = self.run_check()

        self.assertEqual(code, 0, out + err)
        self.assertIn("a box can be reached from here", out)
        self.assertEqual(self.steps(), ["list"], "the diagnostic did more than read")
        self.assertNotIn("create", json.dumps(self.calls()))

    def test_the_json_is_the_same_verdict_for_a_script_to_read(self) -> None:
        code, out, _err = self.run_check(argv=["--json"])

        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertTrue(payload["reachable"])
        self.assertEqual(
            [check["name"] for check in payload["checks"]],
            ["mode", "interpreter", "runner", "cli", "credential", "image"],
        )

    def test_every_answer_the_verdict_can_be_read_from_is_named(self) -> None:
        _code, out, _err = self.run_check()

        self.assertIn("✓ mode: VA_LSE_EXTRACTOR=sandbox", out)
        self.assertIn("✓ credential:", out)
        self.assertIn("? image:", out)
        self.assertIn("vercel vcr image ls va-lse-sandbox", out)

    def test_a_refused_credential_is_the_verdict_and_quotes_the_cli(self) -> None:
        code, out, _err = self.run_check(env={"FAKE_SANDBOX_FAIL": "list"})

        self.assertEqual(code, 1)
        self.assertIn("✗ credential:", out)
        self.assertIn("list failed on purpose", out)
        self.assertIn("VA_LSE_SANDBOX_TOKEN", out)


class TestTheRunnerLine(CheckSandboxTestCase):
    def test_a_line_that_cannot_run_is_reported_once_with_its_remedy(self) -> None:
        code, out, _err = self.run_check(line="definitely-not-installed-anywhere {work}")

        self.assertEqual(code, 1)
        self.assertIn("✗ runner:", out)
        self.assertIn("is not on PATH", out)
        self.assertIn(sys.executable, out, "the remedy must name the interpreter that exists")

    def test_a_bare_python_is_the_substitution_a_run_would_make(self) -> None:
        """Not a failure: this is the documented command on a host with only python3."""
        code, out, _err = self.run_check(
            line="python scripts/vercel_sandbox_runner.py {work}", shutil_stub=_NoPythonOnPath
        )

        self.assertEqual(code, 0, out)
        self.assertIn("'python' is not on PATH; a run resolves it to", out)
        self.assertNotIn("✗ interpreter", out)

    def test_a_line_without_the_file_placeholder_cannot_work(self) -> None:
        code, out, _err = self.run_check(line=self.documented_line().replace(" {work}", ""))

        self.assertEqual(code, 1)
        self.assertIn("{work}", out)
        self.assertIn("✗ runner:", out)

    def test_a_wrapper_is_unproven_rather_than_broken(self) -> None:
        """``my-box-run.sh {work}`` is a documented shape: nothing here may call it wrong."""
        wrapper = self.tmp / "my-box-run.sh"
        wrapper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        wrapper.chmod(0o755)

        code, out, _err = self.run_check(line=f"{shlex.quote(str(wrapper))} {{work}}")

        self.assertEqual(code, 0, out)
        self.assertIn("? runner:", out)
        self.assertIn("is the runner itself", out)
        # The box side is still answered, in-process, because only the wrapper is unknown.
        self.assertIn("✓ cli:", out)
        self.assertIn("✓ credential:", out)
        self.assertEqual(self.steps(), ["list"])

    def test_a_foreign_python_runner_is_executed_and_its_own_verdict_is_believed(self) -> None:
        """The app may be pointed at another Python runner; its answer is the answer."""
        foreign = self.tmp / "my_runner.py"
        foreign.write_text(
            "import json, sys\n"
            "json.dump({'version': 1, 'reachable': True, 'checks': [{\n"
            "    'name': 'canned', 'status': 'ok',\n"
            "    'detail': 'argv: ' + ' '.join(sys.argv[1:]), 'remedy': '',\n"
            "}]}, sys.stdout)\n"
            "sys.stdout.write('\\n')\n",
            encoding="utf-8",
        )

        code, out, _err = self.run_check(
            line=f"{shlex.quote(sys.executable)} {shlex.quote(str(foreign))} {{work}}"
        )

        self.assertEqual(code, 0, out)
        self.assertIn("canned", out)
        self.assertIn("argv: --check", out, "the placeholder must not be handed to --check")
        self.assertEqual(self.steps(), [], "a foreign runner should not touch the CLI")

    def test_a_foreign_runner_that_answers_nothing_is_reported_as_failed(self) -> None:
        foreign = self.tmp / "mute_runner.py"
        foreign.write_text("import sys\nprint('boom', file=sys.stderr)\n", encoding="utf-8")

        code, out, _err = self.run_check(
            line=f"{shlex.quote(sys.executable)} {shlex.quote(str(foreign))} {{work}}"
        )

        self.assertEqual(code, 1)
        self.assertIn("without a verdict", out)
        self.assertIn("boom", out)


class TestModes(CheckSandboxTestCase):
    def test_an_app_that_reads_in_process_is_not_a_broken_box(self) -> None:
        """A box that is configured off cannot fail a run, so it does not fail the exit."""
        code, out, _err = self.run_check(
            mode="in-process", env={"VA_LSE_SANDBOX_CLI": "definitely-not-installed-anywhere"}
        )

        self.assertEqual(code, 0)
        self.assertIn("· mode:", out)
        self.assertIn("every record is read in-process", out)
        self.assertIn("no box would be used", out)
        self.assertIn("✗ cli:", out, "the box side should still be reported")

    def test_the_apps_own_warning_points_at_this_check(self) -> None:
        """The remedy an operator reads *after* a run must name the check that runs before
        one — otherwise the diagnostic exists and nobody finds it."""
        from app import extractors

        self.assertIn("check_sandbox.py", extractors._RUNNER_HELP)

    def test_sandbox_mode_that_cannot_reach_a_box_exits_non_zero(self) -> None:
        code, out, _err = self.run_check(
            mode="sandbox", env={"VA_LSE_SANDBOX_CLI": "definitely-not-installed-anywhere"}
        )

        self.assertEqual(code, 1)
        self.assertIn("a box would be skipped and records read in-process", out)


if __name__ == "__main__":
    unittest.main()
