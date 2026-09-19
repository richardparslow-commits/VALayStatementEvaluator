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

What one invocation does — one file, one box, removed in a ``finally``:

    sandbox create --name va-lse-ocr-<id> --image va-lse-sandbox:latest \\
        --timeout 20m --non-persistent --silent
    sandbox exec <name> -- mkdir -p /work/bundle/<label's directory>
    sandbox copy <work>/<file> <name>:/work/bundle/<label>
    sandbox exec <name> -- python3 /app/scripts/ocr_and_extract.py /work/bundle \\
        --out /work/report.json --force
    sandbox copy <name>:/work/report.json <work>/report.json
    sandbox remove <name>

Four decisions worth knowing, because each is a way this goes wrong:

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
            token=os.getenv("VERCEL_TOKEN", "").strip(),
        )


def _log(text: str) -> None:
    """Progress and failure detail go to stderr; stdout is the report and only
    the report (app/extractors.py parses the runner's whole stdout)."""
    print(text, file=sys.stderr)


def _detail(proc: subprocess.CompletedProcess[str]) -> str:
    """The one line an operator needs: the CLI's own last word, not a wall."""
    for stream in (proc.stderr, proc.stdout):
        lines = [line.strip() for line in (stream or "").splitlines() if line.strip()]
        if lines:
            return lines[-1][:300]
    return "no output"


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

    def _run(self, argv: list[str], purpose: str) -> subprocess.CompletedProcess[str]:
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
        if proc.returncode != 0:
            raise SandboxError(f"{purpose} failed (exit {proc.returncode}): {_detail(proc)}")
        return proc

    # -- steps ---------------------------------------------------------------
    def create(self, name: str) -> None:
        """Boot one box from the pushed image.

        ``--silent`` because the name is ours, not the CLI's, and
        ``--non-persistent`` because this box has exactly one job and no state
        worth snapshotting.
        """
        self._run(
            [
                "create",
                *self.common(),
                "--name",
                name,
                "--image",
                self.settings.image,
                "--timeout",
                self.settings.timeout,
                "--non-persistent",
                "--silent",
            ],
            f"create {name}",
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
        proc = self._run(
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
        )
        for line in f"{proc.stdout or ''}{proc.stderr or ''}".splitlines():
            if line.strip():
                _log(f"  | {line.strip()}")

    def remove(self, name: str) -> None:
        """Best effort: a box that will not die is the box's own problem.

        It cannot outlive its ``--timeout``, and ``--non-persistent`` means it
        leaves no snapshot behind, so a failed cleanup is a warning rather than a
        failed file.
        """
        try:
            self._run(["remove", name], f"remove {name}")
        except SandboxError as exc:
            _log(f"! {exc} — the box stops itself at its --timeout ({self.settings.timeout})")


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
        help="the staging directory the app handed over (holds manifest.json)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
