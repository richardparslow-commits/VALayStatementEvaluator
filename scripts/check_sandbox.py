#!/usr/bin/env python3
"""Answer "would the sandbox work from this host?" in one verdict, creating nothing.

Why this exists
---------------
The app already answers half of it at run time, and only after a file has been
uploaded: ``app/extractors.py`` spawns the configured runner, and when that cannot
start it reports ``the runner command does not exist (python)`` and reads the record
in-process. That is the right behavior and a poor diagnostic — it arrives after the
work, it names one problem, and it says nothing about the credential, the scope, the
project or the image. Every one of those fails the same quiet way: a warning, and a
run that reads records without OCR.

This is the diagnostic instead, and it runs before anything is uploaded:

    .venv/bin/python scripts/check_sandbox.py           # one verdict, human readable
    .venv/bin/python scripts/check_sandbox.py --json    # the same, for a script

What it checks, and how each one can be sure:

* **the runner line** — ``VA_LSE_EXTRACTOR_RUNNER`` exactly as a run resolves it
  (``app.extractors.resolve_python_interpreter``, so a bare ``python`` on a host that
  only has ``python3`` is reported as the substitution a run would make, not as a
  missing interpreter), then whether the interpreter and the script it names are here;
* **the line actually runs** — when the runner is this repository's script, the
  configured command is executed with ``--check``, so "the interpreter exists" becomes
  "this command exits 0";
* **the CLI** — ``VA_LSE_SANDBOX_CLI`` splits and its first word is on ``PATH``;
* **the credential, scope and project** — one authenticated ``sandbox list``, which
  proves the credential is accepted and the team and project are visible to it;
* **the image** — *unproven*, on purpose. The Sandbox CLI has no command that lists
  Vercel Container Registry images, so only a box (or ``vercel vcr image ls``) settles
  it; this says so rather than guessing, because a guessed "pushed" sends you to a run
  that falls back and a guessed "missing" sends you to rebuild an image that is there.

Nothing here creates a box, and ``tests/test_check_sandbox.py`` asserts it: run against
a CLI that journals every call, the only subcommand this can reach is ``list``.

Read ``.env`` the way the app does (importing ``app.config`` is what loads it, with
``override=True``), so what this reports is what a run would use — not what the
current shell happens to export.
"""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import config, extractors  # noqa: E402  (importing config loads .env, like the app)
from scripts import vercel_sandbox_runner as box  # noqa: E402

EXIT_OK = 0
EXIT_FAILED = 1

#: The mode name that means "records are read on the box".
SANDBOX_MODE = "sandbox"

#: The placeholder the app replaces with the directory it staged a file into.
WORK_PLACEHOLDER = "{work}"

#: How long the runner's own probe may take. It is the CLI's ``list`` that can hang
#: (a CLI with no credential prompts to log in), and this is the ceiling on that.
CHECK_TIMEOUT_SECONDS = 90.0


@dataclass
class Line:
    """The configured runner line, as a run would resolve it.

    ``swap`` is the interpreter substitution a run would make (``app.extractors``
    does the resolving here too, so this reports the same thing the app would log);
    ``problem`` is set when the line cannot be run at all.
    """

    written: str
    argv: list[str] = field(default_factory=list)
    swap: extractors.InterpreterSwap | None = None
    script: Path | None = None
    #: True for ``my-box-run.sh {work}``: the program *is* the runner, so there is no
    #: script of ours to find and no ``--check`` it promised to understand.
    wrapper: bool = False
    problem: str = ""
    remedy: str = ""


def _line_argv(written: str) -> list[str]:
    """The line as argv, or [] when it is not a command line at all."""
    try:
        return shlex.split(written)
    except ValueError:
        return []


def parse_line(written: str) -> Line:
    """What the app would do with *written* — including where it would look for it.

    The checks mirror ``app/extractors.CommandBoxRunner.command_for`` (the interpreter
    substitution) and its failure mode (a first word that does not exist), because a
    diagnostic that disagrees with the thing it is diagnosing is worse than none.
    """
    line = Line(written=written)
    if not written.strip():
        line.problem = "VA_LSE_EXTRACTOR_RUNNER is empty, so there is no runner to run"
        line.remedy = (
            'set it to the documented command, "python scripts/vercel_sandbox_runner.py '
            '{work}" (see DEPLOYMENT.md → Sandbox)'
        )
        return line
    argv = _line_argv(written)
    if not argv:
        line.problem = "VA_LSE_EXTRACTOR_RUNNER is not a command line (unbalanced quotes)"
        line.remedy = "quote the whole value, as .env.example does"
        return line
    argv, line.swap = extractors.resolve_python_interpreter(argv)
    line.argv = argv
    if WORK_PLACEHOLDER not in written:
        line.problem = (
            f"the line has no {WORK_PLACEHOLDER} placeholder, so the app cannot tell it "
            "which file to read"
        )
        line.remedy = (
            f'add it: "python scripts/vercel_sandbox_runner.py {WORK_PLACEHOLDER}"'
        )
    program = argv[0]
    if shutil.which(program) is None:
        line.problem = f"the interpreter/program {program!r} is not on PATH"
        line.remedy = (
            f"this app runs under {sys.executable or 'an interpreter with no path'}; write "
            "the command with a bare `python` and it is resolved to that, or name an "
            "interpreter that exists"
        )
        return line
    token = _script_token(argv)
    if token is None:
        # ``my-box-run.sh {work}`` and friends: a real shape (DEPLOYMENT.md shows one),
        # where the program is the runner and nothing here may second-guess it.
        line.wrapper = True
        return line
    line.script = _resolve_script(token)
    if line.script is None:
        line.problem = f"the runner script {token!r} is not there"
        line.remedy = (
            f"it is resolved against the directory the app is started from (now {Path.cwd()}); "
            f"it exists at {PROJECT_ROOT / token} — start the app from the project root, or "
            "name the script by an absolute path"
        )
    return line


def _script_token(argv: list[str]) -> str | None:
    """The script in the command line: the first word after the program that is a path.

    Returns ``None`` when the file placeholder comes first, which is the wrapper shape
    (``my-box-run.sh {work}``) — the caller treats that as "the program is the runner",
    never as a broken line, because DEPLOYMENT.md documents exactly that shape for an
    operator with their own wrapper.
    """
    for token in argv[1:]:
        if token == WORK_PLACEHOLDER:
            return None
        if token.startswith("-"):
            continue
        return token
    return None


def _resolve_script(token: str) -> Path | None:
    """Where the script is, resolved the way the subprocess would resolve it."""
    path = Path(token).expanduser()
    if path.is_absolute():
        return path if path.is_file() else None
    for base in (Path.cwd(), PROJECT_ROOT):
        candidate = (base / path).resolve()
        if candidate.is_file():
            return candidate
    return None


def check_line(line: Line) -> list[box.Check]:
    """The app-side checks: is the configured line runnable at all, and does it run?"""
    if line.problem:
        return [box.Check("runner", "failed", line.problem, line.remedy)]
    checks: list[box.Check] = []
    if line.swap is not None:
        detail = (
            f"{line.swap.written!r} is not on PATH; a run resolves it to "
            f"{line.swap.resolved} (logged at {line.swap.severity})"
        )
        checks.append(box.Check("interpreter", "ok", detail))
    else:
        checks.append(
            box.Check("interpreter", "ok", f"{line.argv[0]} → {shutil.which(line.argv[0])}")
        )
    if line.wrapper:
        checks.append(
            box.Check(
                "runner",
                "unproven",
                f"{line.argv[0]} is the runner itself (a wrapper or a native command), so "
                "only a run proves it works",
                remedy="run it by hand once with a staged directory, or point "
                "VA_LSE_EXTRACTOR_RUNNER at scripts/vercel_sandbox_runner.py",
            )
        )
        return checks
    checks.append(box.Check("runner", "ok", f"{line.script} (from {line.written!r})"))
    return checks


def run_line(line: Line, *, timeout: float = CHECK_TIMEOUT_SECONDS) -> box.SelfCheck | None:
    """Execute the configured command with ``--check``; None when it cannot be run.

    Only this repository's own runner is invoked this way. The word after the
    interpreter has to be a ``.py`` file *and* the placeholder has to be the last
    argument before it, so a wrapper — or a line that does something else after
    staging a file — is never run with a flag it does not know. Reporting "not run"
    beats running the operator's command with an argument they never wrote.
    """
    if line.problem or line.wrapper or line.script is None or line.script.suffix != ".py":
        return None
    argv = [token for token in line.argv if token != WORK_PLACEHOLDER] + ["--check"]
    try:
        proc = subprocess.run(  # noqa: S603 - the operator's own argv, no shell
            argv,
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return box.SelfCheck(
            [box.Check("line", "failed", f"the configured command could not be run: {exc}")]
        )
    try:
        payload = json.loads(proc.stdout)
        checks = [box.Check(**entry) for entry in payload["checks"]]
    except (ValueError, KeyError, TypeError):
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return box.SelfCheck(
            [
                box.Check(
                    "line",
                    "failed",
                    f"the configured command exited {proc.returncode} without a verdict: "
                    f"{tail[-1][:200] if tail else 'no output'}",
                )
            ]
        )
    if proc.returncode != 0 and not any(check.status == "failed" for check in checks):
        checks.append(
            box.Check("line", "failed", f"the configured command exited {proc.returncode}")
        )
    return box.SelfCheck(checks)


def in_process_probe(*, timeout: float = CHECK_TIMEOUT_SECONDS) -> box.SelfCheck:
    """The box-side checks without running the configured line (a wrapper, say).

    Same function the runner's ``--check`` calls, in this process, reading the same
    environment ``app.config`` just loaded — so a wrapper command still gets a verdict
    on its CLI and credential, minus the proof that the wrapper itself runs.
    """
    try:
        settings = box.Settings.from_env()
    except box.SandboxError as exc:
        return box.SelfCheck([box.Check("settings", "failed", str(exc))])
    return box.self_check(settings, timeout=timeout)


def mode_checks() -> list[box.Check]:
    """Whether the app would use a box at all, which is the first thing to know."""
    mode = (config.EXTRACTOR_MODE or "in-process").strip().lower()
    if mode == SANDBOX_MODE:
        return [
            box.Check(
                "mode",
                "ok",
                f"VA_LSE_EXTRACTOR={config.EXTRACTOR_MODE} — records are read on the box",
            )
        ]
    return [
        box.Check(
            "mode",
            "off",
            f"VA_LSE_EXTRACTOR={config.EXTRACTOR_MODE or '(unset)'} — every record is read "
            "in-process, so none of the below affects a run",
            remedy="set VA_LSE_EXTRACTOR=sandbox when you want the box used",
        )
    ]


def diagnose(*, timeout: float = CHECK_TIMEOUT_SECONDS) -> box.SelfCheck:
    """Every check this host can make, in the order a run would hit them."""
    checks = mode_checks()
    line = parse_line(config.EXTRACTOR_RUNNER or "")
    checks.extend(check_line(line))
    probe = run_line(line, timeout=timeout)
    if probe is None:
        if not line.problem and not line.wrapper and line.script is not None:
            checks.append(
                box.Check(
                    "line",
                    "unproven",
                    f"{line.script.name} is not this repository's runner, so it was not run "
                    "with --check: only a run proves it works",
                )
            )
        checks.extend(in_process_probe(timeout=timeout).checks)
    else:
        checks.extend(probe.checks)
    return box.SelfCheck(checks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report whether a Vercel Sandbox can actually be reached from this host: the "
            "runner line, the CLI, the credential with its scope and project, and what "
            "cannot be known locally. Creates nothing."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the verdict as JSON instead of the lines above it",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=CHECK_TIMEOUT_SECONDS,
        help=f"seconds any single probe may take (default {CHECK_TIMEOUT_SECONDS:.0f})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = diagnose(timeout=args.timeout)
    if args.json:
        sys.stdout.write(json.dumps(result.to_json(), indent=2) + "\n")
    else:
        for check in result.checks:
            print(f"  {check.render()}")
        print(f"\nverdict: {result.verdict()}")
        print("         (nothing was created, and no record was read)")
    # A box that is configured off cannot fail a run: the mode check says so and the
    # exit code agrees, so "not using a box" is not reported as a broken box.
    if not result.reachable and _mode_is_sandbox():
        return EXIT_FAILED
    return EXIT_OK


def _mode_is_sandbox() -> bool:
    return (config.EXTRACTOR_MODE or "").strip().lower() == SANDBOX_MODE


if __name__ == "__main__":
    raise SystemExit(main())
