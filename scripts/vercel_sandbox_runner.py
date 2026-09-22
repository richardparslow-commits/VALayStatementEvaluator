#!/usr/bin/env python3
"""Run ``scripts/ocr_and_extract.py`` for one staged file on a Vercel Sandbox.

Why this exists
---------------
``app/extractors.py`` gives the app a ``RecordExtractor`` port and ships
``CommandBoxRunner``: ``VA_LSE_EXTRACTOR=sandbox`` plus a command, with ``{work}``
standing for the directory the app staged one file into. The ``sandbox`` image
(Dockerfile's ``sandbox`` stage) is the only image in this project that carries
OCR tooling, and ``scripts/ocr_and_extract.py`` is its entrypoint — but nothing in
the repository knew how to reach a box, so the command was the operator's to
invent. This is that command, for Vercel Sandbox:

    VA_LSE_EXTRACTOR=sandbox
    VA_LSE_EXTRACTOR_RUNNER="python scripts/vercel_sandbox_runner.py {work}"

That first word may be a ``python`` this host does not have — macOS ships ``python3``
and no ``python`` — so ``app.extractors.resolve_python_interpreter`` replaces a
Python interpreter the host lacks (that spelling, or a venv path that has moved)
with the interpreter running the app, and logs which one took over. The command
above is therefore the supported spelling on every host; an absolute path is only
needed to pin a *particular* Python.

``--check`` answers a different question with no box at all: can a box be reached from
this host. It reports the CLI it would invoke, asks the account one authenticated
question (``list``, the cheapest read in the CLI reference), and says plainly what it
*cannot* know from here — whether the image is in the registry — instead of guessing.
``scripts/check_sandbox.py`` is the operator-facing half of that: it checks the runner
line the app is configured with and folds this verdict in, as one answer.

What one invocation does — one file, one box, removed in a ``finally``:

    sandbox create --name va-lse-ocr-<id> --image va-lse-sandbox:latest \\
        --timeout 20m --non-persistent --silent
    sandbox exec <name> -- mkdir -p /work/bundle/<label's directory>
    sandbox copy <work>/<file> <name>:/work/bundle/<label>
    sandbox exec <name> -- python3 /app/scripts/ocr_and_extract.py /work/bundle \\
        --out /work/report.json --force
    sandbox copy <name>:/work/report.json <work>/report.json
    sandbox remove <name>

Five decisions worth knowing, because each is a way this goes wrong:

* **The report comes back as a file, not as exec output.** ``sandbox exec`` is a
  session, not a pipe — its stdout can carry connection and progress lines — and
  ``app/extractors.py`` parses the runner's *whole stdout* as the report JSON. So
  the box writes ``/work/report.json``, this script copies that file back and
  prints its bytes as the only thing on stdout. Everything else, including the
  entrypoint's own summary and the CLI's output, goes to stderr.
* **The label is the path.** The staged file is copied to
  ``/work/bundle/<label>``, not to a flat path, so the entrypoint's
  ``_label_for`` answers under exactly the name the app handed over. Citations
  point at ``records/2024/visit.pdf``, and ``app/extractors.py`` refuses any
  answer under a name it did not stage — a different name here would silently
  fall back, so a label that climbs out of the bundle is refused rather than
  sanitised.
* **The box is ephemeral and always removed.** One microVM per file (that is the
  port's shape: one file in, documents out), ``--non-persistent`` so no snapshot
  accumulates, and ``remove`` in a ``finally`` — including on SIGTERM, which this
  script turns into an exception so the cleanup runs. SIGKILL cannot be caught,
  and that is exactly what the app's own per-file timeout sends, so the box also
  carries a ``--timeout`` backstop: a box nobody removed stops itself.
* **Vercel's two credentials are not interchangeable, and this says so.** A sandbox
  takes a Vercel *access token* (dashboard → Account Settings → Tokens, scoped to
  the team) or the ``VERCEL_OIDC_TOKEN`` a Function is handed — read here from
  ``VA_LSE_SANDBOX_TOKEN``, then ``VERCEL_AUTH_TOKEN`` (the name the Sandbox CLI
  itself reads), then ``VERCEL_OIDC_TOKEN``, then ``VERCEL_TOKEN``. The
  **AI Gateway** key (``vck_…``) that drives the app's own LLM calls is a different
  product and never authenticates a sandbox, so a gateway-shaped value in one of
  those slots is refused by name instead of being handed to the CLI for an opaque
  401.
* **The app is not imported and Vercel is not baked in.** This is operator
  tooling: stdlib only, no ``app`` import, and the CLI is invoked as a command
  line (``VA_LSE_SANDBOX_CLI``) so a stored ``sandbox login``, a team scope, or a
  wrapper an operator already has keeps working. ``sandbox``-mode failures are
  fail-open by design — a non-zero exit here means the app reads that file
  in-process and logs this script's last stderr line.

Exit codes: 0 = the report JSON is on stdout; 1 = anything else, with the reason
on stderr (the app reports it and reads the file in-process).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

EXIT_OK = 0
EXIT_FAILED = 1

#: Defaults for the knobs in README → Environment variables and DEPLOYMENT.md →
#: Sandbox target. All but ``DEFAULT_CLI`` name things this repository builds.
DEFAULT_CLI = "sandbox"
DEFAULT_IMAGE = "va-lse-sandbox:latest"
DEFAULT_TIMEOUT = "20m"

#: Where the box keeps one file's work — absolute, so no ``--workdir`` is needed
#: and a resumed box cannot make the answer depend on a session's cwd.
REMOTE_BUNDLE = PurePosixPath("/work/bundle")
REMOTE_REPORT = PurePosixPath("/work/report.json")

#: The image is this repository's own ``sandbox`` stage (WORKDIR /app, scripts/
#: copied in), so the interpreter and entrypoint paths are constants, not knobs.
BOX_PYTHON = "python3"
BOX_ENTRYPOINT = "/app/scripts/ocr_and_extract.py"

#: The file the app stages beside the record: app/extractors.py writes it, and it
#: carries the label the box must answer under.
MANIFEST_NAME = "manifest.json"
REPORT_NAME = "report.json"

#: Where a sandbox credential is read from, in order. ``VA_LSE_SANDBOX_TOKEN`` is
#: this repository's own name for it; ``VERCEL_AUTH_TOKEN`` is the name the Sandbox
#: CLI itself reads (its help: "the token stored in your system from
#: VERCEL_AUTH_TOKEN"); ``VERCEL_OIDC_TOKEN`` is what a Vercel Function can use
#: instead, Vercel's recommendation because nothing long-lived has to be stored;
#: ``VERCEL_TOKEN`` is the Vercel REST API's convention, which the Sandbox CLI does
#: *not* read — the runner passes it as ``--token``, which every subcommand takes
#: (verified against `sandbox <subcommand> --help`, 4.4.0). Unset everywhere means
#: the CLI's stored ``sandbox login`` session is used.
TOKEN_ENV_NAMES = (
    "VA_LSE_SANDBOX_TOKEN",
    "VERCEL_AUTH_TOKEN",
    "VERCEL_OIDC_TOKEN",
    "VERCEL_TOKEN",
)

#: Vercel AI Gateway keys. They carry Vercel's name and nothing else: they
#: authenticate the gateway's OpenAI-compatible LLM API (the app's
#: ``OPENAI_API_KEY``), never a sandbox. Recognised here so the mistake is refused
#: with the remedy rather than surfacing as an opaque 401 from the CLI.
GATEWAY_KEY_PREFIXES = ("vck_", "vcg_")


class SandboxError(RuntimeError):
    """Anything that means this file must be read in-process instead."""


@dataclass(frozen=True)
class Staged:
    """One file the app handed over, as ``manifest.json`` describes it."""

    label: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class Settings:
    """What one invocation needs, read from the environment at call time."""

    cli: tuple[str, ...]
    image: str
    timeout: str
    scope: str
    project: str
    token: str

    @classmethod
    def from_env(cls) -> "Settings":
        cli = os.getenv("VA_LSE_SANDBOX_CLI", DEFAULT_CLI).strip() or DEFAULT_CLI
        try:
            parts = tuple(shlex.split(cli))
        except ValueError as exc:
            raise SandboxError(f"VA_LSE_SANDBOX_CLI is not a command line: {exc}") from exc
        if not parts:
            raise SandboxError("VA_LSE_SANDBOX_CLI is empty — set it to the sandbox CLI")
        return cls(
            cli=parts,
            image=os.getenv("VA_LSE_SANDBOX_IMAGE", DEFAULT_IMAGE).strip() or DEFAULT_IMAGE,
            timeout=os.getenv("VA_LSE_SANDBOX_TIMEOUT", DEFAULT_TIMEOUT).strip()
            or DEFAULT_TIMEOUT,
            scope=os.getenv("VA_LSE_SANDBOX_SCOPE", "").strip(),
            project=os.getenv("VA_LSE_SANDBOX_PROJECT", "").strip(),
            token=sandbox_token_from_env(),
        )


def sandbox_token_from_env() -> str:
    """The sandbox credential, or "" to let the CLI use its stored session.

    OIDC first among the Vercel names because that is the recommended path: inside a
    Function or a Vercel-wired CI job the token is already provisioned, so a
    deployment needs no long-lived secret. A gateway key in any of these slots is
    refused here, by name, rather than passed to a CLI that can only answer 401.
    """
    for name in TOKEN_ENV_NAMES:
        value = os.getenv(name, "").strip()
        if not value:
            continue
        if value.startswith(GATEWAY_KEY_PREFIXES):
            raise SandboxError(
                f"{name} holds what looks like a Vercel AI Gateway key ({value[:4]}…) — that "
                "authenticates the gateway's LLM API (set it as the app's OPENAI_API_KEY), not a "
                "sandbox. Vercel Sandbox takes an access token (dashboard → Account Settings → "
                "Tokens, scoped to the team) or a Function's VERCEL_OIDC_TOKEN; unset this "
                "variable to fall back to the CLI's `sandbox login` session"
            )
        return value
    return ""


def _log(text: str) -> None:
    """Progress and failure detail go to stderr; stdout is the report and only
    the report (app/extractors.py parses the runner's whole stdout)."""
    print(text, file=sys.stderr)


#: Lines the CLI appends *after* the reason. Measured against 4.4.0: a failed
#: ``create`` ends with ``╰▶ hint: the full response buffer is stored in /tmp/…``,
#: so taking the last line verbatim reports a temp-file path instead of the reason
#: ("Image not found"). These are skipped, and never chosen over a real message.
_NOISE_MARKERS = ("hint:", "response buffer")

def _detail(proc: subprocess.CompletedProcess[str]) -> str:
    """The one line an operator needs: the CLI's own reason, not a wall or a hint.

    The *last* line is not it. A failed ``create`` ends with
    ``╰▶ hint: the full response buffer is stored in /tmp/…``, so quoting the last line
    hides the reason (measured: an unpushed image reported a temp path instead of
    "Image not found"). Hints are dropped and the last line that remains is the CLI's
    final word — which is the box's own sentence where there is one.
    """
    fallback = ""
    for stream in (proc.stderr, proc.stdout):
        lines = [line.strip() for line in (stream or "").splitlines() if line.strip()]
        if not lines:
            continue
        reasons = [line for line in lines if not any(m in line.lower() for m in _NOISE_MARKERS)]
        if reasons:
            return reasons[-1][:300]
        if not fallback:  # nothing but hints: a hint beats silence
            fallback = lines[-1][:300]
    return fallback or "no output"


def _masked(argv: Sequence[str]) -> str:
    """The command line as logged — the token's value never reaches a log."""
    shown: list[str] = []
    hide_next = False
    for arg in argv:
        if hide_next:
            shown.append("…")
            hide_next = False
            continue
        shown.append(arg)
        hide_next = arg in ("--token",)
    return " ".join(shown)


class SandboxCli:
    """The sandbox CLI, one argv at a time — the only thing tests replace.

    Wrapping the supported command line rather than the HTTP API is deliberate:
    ``sandbox login`` stores a session, a team scope is one flag, and an operator
    with a wrapper script keeps it by pointing ``VA_LSE_SANDBOX_CLI`` at it.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # -- plumbing ------------------------------------------------------------
    def common(self) -> list[str]:
        """The flags every subcommand that takes options accepts."""
        flags: list[str] = []
        if self.settings.token:
            flags += ["--token", self.settings.token]
        if self.settings.scope:
            flags += ["--scope", self.settings.scope]
        if self.settings.project:
            flags += ["--project", self.settings.project]
        return flags

    def _run(
        self, argv: list[str], purpose: str, *, forward: bool = False
    ) -> subprocess.CompletedProcess[str]:
        """One CLI invocation. ``forward`` echoes the box's own output to our stderr
        *before* the return code is judged, so a step that fails still shows what the
        box said — for the entrypoint that output is the diagnosis ("Scans are present
        and no OCR tooling is installed"), and dropping it on failure was leaving the
        operator with a single line instead of the reason."""
        full = [*self.settings.cli, *argv]
        _log(f"  $ {_masked(full)}")
        try:
            proc = subprocess.run(  # noqa: S603 - argv, no shell; the CLI is the operator's
                full,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise SandboxError(
                f"the sandbox CLI was not found ({exc.filename or self.settings.cli[0]}). "
                "Install it with `npm i -g sandbox` and log in, or point "
                "VA_LSE_SANDBOX_CLI at it"
            ) from exc
        if forward:
            for line in f"{proc.stdout or ''}{proc.stderr or ''}".splitlines():
                if line.strip():
                    _log(f"  | {line.strip()}")
        if proc.returncode != 0:
            raise SandboxError(f"{purpose} failed (exit {proc.returncode}): {_detail(proc)}")
        return proc

    # -- steps ---------------------------------------------------------------
    def create(self, name: str) -> None:
        """Boot one box from the pushed image.

        ``--silent`` because the name is ours, not the CLI's, and
        ``--non-persistent`` because this box has exactly one job and no state
        worth snapshotting.

        ``VA_LSE_SANDBOX_IMAGE=none`` boots the CLI's *default runtime* instead of a
        VCR image — the escape hatch for probing a credential and the create/exec/
        copy/remove cycle without an image in the registry (measured: 4.4.0's
        default runtime is Python 3.14, which this app's hash-pinned lock refuses,
        so it is a probe and not a way to read records).
        """
        argv = ["create", *self.common(), "--name", name]
        if self.settings.image.lower() not in ("", "none"):
            argv += ["--image", self.settings.image]
        argv += ["--timeout", self.settings.timeout, "--non-persistent", "--silent"]
        try:
            self._run(argv, f"create {name}")
        except SandboxError as exc:
            raise SandboxError(f"{exc}{self._image_advice(exc)}") from exc

    def _image_advice(self, exc: SandboxError) -> str:
        """Turn a 404 into the step that fixes it.

        The image is built by hand (DEPLOYMENT.md §6), so "not in the registry" is the
        likely first failure for a fresh clone, and the CLI reports it as a bare status
        — the "Image not found" message itself stays in its response buffer (measured
        against 4.4.0). Naming the remedy here beats an operator decoding a 404.
        """
        if self.settings.image.lower() in ("", "none"):
            return ""
        if not any(marker in str(exc).lower() for marker in ("404", "not found")):
            return ""
        return (
            f"\n  if that is about the image: {self.settings.image} is not in the "
            "registry yet. Build and push it (DEPLOYMENT.md §6), or set "
            "VA_LSE_SANDBOX_IMAGE=none to boot the CLI's default runtime and check "
            "the credential on its own"
        )

    def mkdir(self, name: str, directory: PurePosixPath) -> None:
        """Make the label's directory: ``copy`` writes a file, not a tree."""
        self._run(
            ["exec", *self.common(), name, "--", "mkdir", "-p", str(directory)],
            f"prepare {directory} on {name}",
        )

    def copy(self, source: str, destination: str, purpose: str) -> None:
        """One file, one direction — the CLI transfers one file per invocation."""
        self._run(["copy", *self.common(), source, destination], purpose)

    def extract(self, name: str, bundle: PurePosixPath, report: PurePosixPath) -> None:
        """Run the entrypoint in the box; its summary is forwarded to stderr.

        ``--force`` is free insurance: a fresh box has no ``report.json``, but a
        resumed one (a future reuse mode, or a name that collided) must not make
        this step fail on the entrypoint's overwrite refusal.
        """
        self._run(
            [
                "exec",
                *self.common(),
                name,
                "--",
                BOX_PYTHON,
                BOX_ENTRYPOINT,
                str(bundle),
                "--out",
                str(report),
                "--force",
            ],
            f"read {bundle} on {name}",
            forward=True,
        )

    def remove(self, name: str) -> None:
        """Best effort: a box that will not die is the box's own problem.

        It cannot outlive its ``--timeout``, and ``--non-persistent`` means it
        leaves no snapshot behind, so a failed cleanup is a warning rather than a
        failed file.

        The auth flags go here too. Vercel's CLI reference lists no options for
        ``remove``, but the CLI itself takes them (``--token``/``--scope``/
        ``--project``; measured against 4.4.0) — and without them a cleanup would
        fail for every deployment that authenticates with a token rather than a
        stored ``sandbox login``, which is every deployment there is.
        """
        try:
            self._run(["remove", *self.common(), name], f"remove {name}")
        except SandboxError as exc:
            _log(f"! {exc} — the box stops itself at its --timeout ({self.settings.timeout})")


# -- what can be known without a box -----------------------------------------

#: The command ``--check`` uses to ask the account a question: the cheapest one in
#: the published reference, an authenticated read, and it creates nothing. Kept as
#: a constant so a test can assert this is the only subcommand a check ever runs.
CHECK_SUBCOMMAND = "list"

#: How long that question may take. With no credential at all the CLI *prompts to
#: log in* (its reference: "we'll use a stored token or prompt you to log in"), and a
#: prompt with no terminal to read must not be able to hang a diagnostic whose whole
#: purpose is to answer before a run does.
CHECK_TIMEOUT_SECONDS = 60.0

#: One glyph per status, so ``--check`` and ``scripts/check_sandbox.py`` print the
#: same marks for the same findings. ``off`` is not a verdict about the box at all —
#: it is configuration that says no box will be used — which is why it is its own
#: status rather than a failure (nothing is broken) or an ``ok`` (nothing was proven).
CHECK_GLYPHS = {"ok": "✓", "failed": "✗", "unproven": "?", "off": "·"}


@dataclass(frozen=True)
class Check:
    """One thing a box-free probe can say — including what it cannot say.

    ``status`` is ``ok`` (proven to work from here), ``failed`` (proven not to), or
    ``unproven``, which is a real answer rather than a shrug: when no local probe can
    settle something, saying so and naming what *would* settle it is the only honest
    verdict, and it must not be dressed up as either of the others.
    """

    name: str
    status: str
    detail: str
    remedy: str = ""

    def to_json(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "remedy": self.remedy,
        }

    def render(self) -> str:
        """The check as one operator-facing paragraph, remedy included."""
        line = f"{CHECK_GLYPHS.get(self.status, '?')} {self.name}: {self.detail}"
        if self.status != "ok" and self.remedy:
            line += f"\n    → {self.remedy}"
        return line


@dataclass(frozen=True)
class SelfCheck:
    """The box-free verdict: every check, and whether a box would be reached."""

    checks: list[Check]

    @property
    def failed(self) -> list[Check]:
        return [check for check in self.checks if check.status == "failed"]

    @property
    def unproven(self) -> list[Check]:
        return [check for check in self.checks if check.status == "unproven"]

    @property
    def off(self) -> list[Check]:
        return [check for check in self.checks if check.status == "off"]

    @property
    def reachable(self) -> bool:
        """True when nothing that *can* be checked would make a run fall back."""
        return not self.failed

    def to_json(self) -> dict[str, Any]:
        return {
            "version": 1,
            "reachable": self.reachable,
            "checks": [check.to_json() for check in self.checks],
        }

    def verdict(self) -> str:
        """The one sentence the operator reads, and the reason it says what it says."""
        if self.off:
            return (
                "no box would be used: the app is configured to read records in-process, so "
                "nothing below can affect a run"
            )
        if self.failed:
            names = ", ".join(check.name for check in self.failed)
            return (
                f"a box would be skipped and records read in-process: {names} "
                f"{'is' if len(self.failed) == 1 else 'are'} wrong for a run from here"
            )
        if self.unproven:
            names = ", ".join(check.name for check in self.unproven)
            return (
                "a box can be reached from here as far as anything local can tell; "
                f"{names} can only be proven by a box"
            )
        return "a box can be reached from here"


def token_source() -> str:
    """Which variable the credential came from, or "" for the CLI's stored session."""
    for name in TOKEN_ENV_NAMES:
        if os.getenv(name, "").strip():
            return name
    return ""


def self_check(settings: Settings, *, timeout: float = CHECK_TIMEOUT_SECONDS) -> SelfCheck:
    """What this host can and cannot prove about a box, without creating one.

    Three questions are answerable away from Vercel: is the CLI the operator named
    actually here, does it accept the credential for this scope and project, and is
    the image in the registry. The third has no local answer (see :func:`_check_image`)
    and is reported as unproven rather than guessed.

    Answering the second question costs one command — ``list`` — and that is the
    point: an unaccepted token, a team the token cannot see, or a project that does
    not exist all surface *before* a file is uploaded, instead of as one warning per
    run afterwards. When the CLI is not here the second question is not asked: it
    could only repeat the first answer.
    """
    checks = [_check_cli(settings)]
    if checks[0].status == "ok":
        checks.append(_check_account(settings, timeout=timeout))
    checks.append(_check_image(settings))
    return SelfCheck(checks)


def _check_cli(settings: Settings) -> Check:
    """Is the CLI the operator named on this host at all?"""
    program = settings.cli[0]
    found = shutil.which(program)
    if found is None:
        return Check(
            "cli",
            "failed",
            f"{program!r} (VA_LSE_SANDBOX_CLI={shlex.join(settings.cli)}) is not on PATH",
            remedy=(
                "install it (`npm i -g sandbox`) and log in, or point "
                "VA_LSE_SANDBOX_CLI at the command that works here"
            ),
        )
    return Check("cli", "ok", f"{shlex.join(settings.cli)} → {found}")


def _check_account(settings: Settings, *, timeout: float) -> Check:
    """Ask the account one authenticated question: does ``list`` answer?

    This is what separates "the CLI is installed" from "the CLI can do anything for
    this deployment": a refused token, a team it cannot see and a project that does
    not exist each fail here, and each one is exactly what would turn every upload
    into an in-process fallback.
    """
    argv = [*settings.cli, CHECK_SUBCOMMAND, *SandboxCli(settings).common(), "--limit", "1"]
    _log(f"  $ {_masked(argv)}")
    try:
        proc = subprocess.run(  # noqa: S603 - argv, no shell; the CLI is the operator's
            argv,
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return Check("credential", "failed", f"the sandbox CLI could not be run: {exc}")
    except subprocess.TimeoutExpired:
        return Check(
            "credential",
            "failed",
            f"`{shlex.join(settings.cli)} {CHECK_SUBCOMMAND}` did not answer within "
            f"{timeout:.0f}s",
            remedy=(
                "a CLI with no credential prompts to log in and there is no terminal here "
                "to answer it: run `sandbox login` once, or set VA_LSE_SANDBOX_TOKEN"
            ),
        )
    if proc.returncode != 0:
        return Check(
            "credential",
            "failed",
            f"`{CHECK_SUBCOMMAND}` was refused (exit {proc.returncode}): {_detail(proc)}",
            remedy=_credential_remedy(settings),
        )
    return Check("credential", "ok", _account_detail(settings))


def _credential_remedy(settings: Settings) -> str:
    """What to fix when the account refuses to answer, in the order to fix it."""
    named = ""
    where = [
        part
        for part in (
            f"scope {settings.scope}" if settings.scope else "",
            f"project {settings.project}" if settings.project else "",
        )
        if part
    ]
    if where:
        named = f" — this check used {' and '.join(where)}"
    return (
        "a sandbox takes a Vercel access token (dashboard → Account Settings → Tokens, "
        "scoped to the team) in VA_LSE_SANDBOX_TOKEN, or a `sandbox login` session; an "
        "AI Gateway key does not authenticate one. Then confirm VA_LSE_SANDBOX_SCOPE and "
        f"VA_LSE_SANDBOX_PROJECT name a team and project the credential can see{named}"
    )


def _account_detail(settings: Settings) -> str:
    """What answered, naming the settings a later failure would blame."""
    source = token_source()
    how = source if source else "the CLI's stored `sandbox login` session"
    where = f"scope {settings.scope}" if settings.scope else "the account's default scope"
    project = (
        f"project {settings.project}" if settings.project else "the linked/default project"
    )
    return f"`{CHECK_SUBCOMMAND}` answered for {where} and {project}, authenticated by {how}"


def _check_image(settings: Settings) -> Check:
    """The one question no local probe can settle, said honestly.

    The Sandbox CLI has no command that lists Vercel Container Registry images (its
    published reference, 4.4.0: list/create/fork/run/exec/connect/copy/stop/remove/
    config/sessions/snapshot/snapshots/drives/login/logout), so "is
    ``va-lse-sandbox:latest`` pushed?" is answerable only by reading the registry or by
    booting a box. Reported as ``unproven`` with both ways to settle it, because a
    guessed "✓ pushed" sends the operator to a run that falls back and a guessed
    "✗ missing" sends them off to rebuild an image that was already there.
    """
    if settings.image.lower() in ("", "none"):
        return Check(
            "image",
            "ok",
            "VA_LSE_SANDBOX_IMAGE=none — the box boots the CLI's default runtime, so no "
            "registry image is involved",
        )
    repository = settings.image.split(":", 1)[0]
    return Check(
        "image",
        "unproven",
        f"{settings.image} is or is not in Vercel Container Registry, and nothing on "
        "this host can tell: the Sandbox CLI has no command that lists images",
        remedy=(
            f"read the registry (`vercel vcr image ls {repository}` — the Vercel CLI, not "
            "the sandbox one), build and push it (DEPLOYMENT.md §6), or prove it end to "
            "end with the opt-in live test (`VA_LSE_TEST_VERCEL_SANDBOX_TOKEN=… python -m "
            "unittest tests.test_vercel_sandbox_live`)"
        ),
    )


def read_manifest(work_dir: Path) -> Staged:
    """What the app staged: a label to answer under, and the bytes to read."""
    manifest = work_dir / MANIFEST_NAME
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SandboxError(
            f"{manifest} is missing — it is what app/extractors.py stages next to the "
            "record, and the runner is given that directory's path"
        ) from exc
    except (OSError, ValueError) as exc:
        raise SandboxError(f"{manifest} cannot be read: {exc}") from exc
    file_info = raw.get("file") if isinstance(raw, dict) else None
    if not isinstance(file_info, dict) or not file_info.get("label") or not file_info.get("path"):
        raise SandboxError(
            f"{manifest} has no file.label/file.path — it is not the manifest "
            "app/extractors.py writes"
        )
    staged = Staged(
        label=str(file_info["label"]),
        path=Path(str(file_info["path"])),
        sha256=str(file_info.get("sha256") or ""),
    )
    if not staged.path.is_file():
        raise SandboxError(f"the staged file {staged.path} is gone (or is not a file)")
    if staged.sha256:
        actual = hashlib.sha256(staged.path.read_bytes()).hexdigest()
        if actual != staged.sha256:
            raise SandboxError(
                f"the staged file {staged.path} does not match the sha256 in the "
                f"manifest ({actual} vs {staged.sha256}) — refusing to read different "
                "bytes than the app staged"
            )
    return staged


def relative_label(label: str) -> PurePosixPath:
    """The label as a path inside the bundle, or a refusal.

    The label *is* the path: the entrypoint names every document relative to the
    directory it walks, so copying ``records/2024/visit.pdf`` to
    ``/work/bundle/records/2024/visit.pdf`` is what makes the box answer under the
    name the app will cite. A label that climbs out of the bundle is refused here
    rather than rewritten: a different name would be refused by
    ``app/extractors.py`` anyway, and a silent fallback is worse than this line.
    """
    parts = [
        part
        for part in PurePosixPath(label.replace("\\", "/")).parts
        if part not in ("", ".", "/")
    ]
    if not parts:
        raise SandboxError(f"the staged label {label!r} is empty")
    if any(part == ".." for part in parts):
        raise SandboxError(f"the staged label {label!r} climbs out of the bundle directory")
    return PurePosixPath(*parts)


def run_for_one_file(work_dir: Path, settings: Settings) -> str:
    """Move one staged file through a box and return the report JSON text."""
    staged = read_manifest(work_dir)
    remote_record = REMOTE_BUNDLE / relative_label(staged.label)
    name = f"va-lse-ocr-{uuid.uuid4().hex[:12]}"
    cli = SandboxCli(settings)
    _log(
        f"box {name}: {staged.label} → {settings.image} "
        f"(timeout {settings.timeout}, ephemeral)"
    )

    created = False
    try:
        cli.create(name)
        created = True
        cli.mkdir(name, remote_record.parent)
        cli.copy(
            str(staged.path),
            f"{name}:{remote_record}",
            f"copy {staged.label} into {name}",
        )
        cli.extract(name, REMOTE_BUNDLE, REMOTE_REPORT)
        report_path = work_dir / REPORT_NAME
        cli.copy(
            f"{name}:{REMOTE_REPORT}",
            str(report_path),
            f"copy the report out of {name}",
        )
        return report_path.read_text(encoding="utf-8")
    finally:
        if created:
            cli.remove(name)


def _handle_termination(signum: int, _frame: Any) -> None:
    """Turn SIGTERM/SIGHUP into an exception so the box's ``finally`` runs.

    SIGKILL (what the app's own per-file timeout sends) cannot be caught; the
    box's ``--timeout`` is the backstop for that, and ``--non-persistent`` means
    nothing is left behind either way.
    """
    raise SystemExit(128 + signum)


def _install_cleanup_on_termination() -> None:
    for name in ("SIGTERM", "SIGHUP"):
        value = getattr(signal, name, None)
        if value is None:
            continue
        try:
            signal.signal(value, _handle_termination)
        except (ValueError, OSError):  # not the main thread — nothing to install
            return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read one file the app staged ({work}) on a Vercel Sandbox and print the "
            "report JSON scripts/ocr_and_extract.py writes. Used as "
            'VA_LSE_EXTRACTOR_RUNNER="python scripts/vercel_sandbox_runner.py {work}".'
        )
    )
    parser.add_argument(
        "work",
        type=Path,
        nargs="?",
        help="the staging directory the app handed over (holds manifest.json)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "report whether a box can be reached from this host and exit: the CLI, the "
            "credential with this scope and project, and whether the image can be "
            "known. Creates nothing; prints its verdict as JSON on stdout"
        ),
    )
    return parser


def _run_check() -> int:
    """``--check``: the box-free verdict, as JSON on stdout and words on stderr."""
    _log("sandbox self-check — no box is created, nothing is uploaded")
    try:
        result = self_check(Settings.from_env())
    except SandboxError as exc:
        _log(f"✗ {exc}")
        return EXIT_FAILED
    for check in result.checks:
        _log(f"  {check.render()}")
    _log(f"verdict: {result.verdict()}")
    sys.stdout.write(json.dumps(result.to_json(), indent=2) + "\n")
    sys.stdout.flush()
    return EXIT_OK if result.reachable else EXIT_FAILED


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.check:
        return _run_check()
    if args.work is None:
        parser.error("the staging directory is required unless --check is used")
    _install_cleanup_on_termination()
    try:
        settings = Settings.from_env()
        report_text = run_for_one_file(args.work, settings)
        report = json.loads(report_text)
        if not isinstance(report, dict) or not isinstance(report.get("documents"), list):
            raise SandboxError(
                f"the box's {REPORT_NAME} has no 'documents' list — it is not the "
                "report scripts/ocr_and_extract.py --out writes"
            )
    except SandboxError as exc:
        _log(f"✖ {exc}")
        return EXIT_FAILED
    except ValueError as exc:
        _log(f"✖ the box's {REPORT_NAME} is not JSON: {exc}")
        return EXIT_FAILED
    # stdout carries the report and nothing else: app/extractors.py parses the
    # runner's whole stdout as this JSON.
    sys.stdout.write(report_text if report_text.endswith("\n") else report_text + "\n")
    sys.stdout.flush()
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
