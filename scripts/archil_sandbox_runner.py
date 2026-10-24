#!/usr/bin/env python3
"""Run ``scripts/ocr_and_extract.py`` for one staged file on an Archil Serverless Disk/MicroVM.

Why this exists
---------------
``app/extractors.py`` defines the ``RecordExtractor`` port and ``CommandBoxRunner``:
setting ``VA_LSE_EXTRACTOR=sandbox`` and ``VA_LSE_EXTRACTOR_RUNNER`` delegates PDF/DOCX/scan
extraction to an isolated Linux microVM carrying ``tesseract-ocr``, ``ocrmypdf``,
``poppler-utils``, ``ghostscript``, and ``qpdf`` (the ``sandbox`` stage of ``Dockerfile``).

While ``scripts/vercel_sandbox_runner.py`` implements that contract for Vercel Sandbox
(requiring 5 sequential CLI calls per file: ``create``, ``mkdir``, ``copy`` in, ``exec``,
``copy`` out, ``remove``), Archil (https://console.archil.com/) provides POSIX disks
colocated with serverless Linux microVMs and copy-on-write (CoW) branching:

    VA_LSE_EXTRACTOR=sandbox
    VA_LSE_EXTRACTOR_RUNNER="python scripts/archil_sandbox_runner.py {work}"

Two execution paths are supported automatically:

1. **Zero-copy shared-disk path (when ``VA_LSE_ARCHIL_MOUNT_DIR`` is mounted locally):**
   ``app/extractors.py`` already stages files into ``work_dir`` and ``app/blob_store.py``.
   When the host and the Archil microVM share a mounted Archil disk, the file is staged
   directly on the mounted POSIX path and ``disk exec`` runs ``ocr_and_extract.py`` in a
   colocated microVM with zero network copy overhead.
2. **Ephemeral CoW branch path (for Streamlit Community Cloud / unmounted hosts):**
   Forks an instant copy-on-write branch of ``VA_LSE_ARCHIL_DISK``
   (``disk branch create <disk> va-lse-ocr-<id>``), stages the file under its exact
   citation label (``/work/bundle/<label>``), runs ``ocr_and_extract.py`` via serverless
   execution (``disk exec``), reads back ``/work/report.json`` to stdout, and removes the
   CoW branch in a ``finally`` block (including on ``SIGTERM``/``SIGHUP``).

``--check`` implements the box-free diagnostic contract consumed by
``scripts/check_sandbox.py``: it verifies the Archil CLI (``VA_LSE_ARCHIL_CLI``, default
``disk``), checks the credential in ``VA_LSE_ARCHIL_TOKEN`` / ``ARCHIL_API_KEY`` /
``ARCHIL_TOKEN`` with a read-only ``disk list`` call, and reports ``VA_LSE_ARCHIL_DISK``
and ``VA_LSE_ARCHIL_IMAGE`` in the exact ``{"version": 1, "reachable": ..., "checks": [...]}``
JSON format ``scripts/check_sandbox.py`` expects.
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

DEFAULT_CLI = "disk"
DEFAULT_DISK = "valay-eval-sandbox"
DEFAULT_IMAGE = "va-lse-sandbox:latest"
DEFAULT_TIMEOUT = "15m"

REMOTE_BUNDLE = PurePosixPath("/work/bundle")
REMOTE_REPORT = PurePosixPath("/work/report.json")

BOX_PYTHON = "python3"
BOX_ENTRYPOINT = "/app/scripts/ocr_and_extract.py"

MANIFEST_NAME = "manifest.json"
REPORT_NAME = "report.json"

TOKEN_ENV_NAMES = (
    "VA_LSE_ARCHIL_TOKEN",
    "ARCHIL_API_KEY",
    "ARCHIL_TOKEN",
)

# Prevent accidental use of LLM provider keys in the Archil storage/compute token slot.
NON_ARCHIL_KEY_PREFIXES = ("pplx-", "sk-", "vck_", "vcg_")

CHECK_SUBCOMMAND = "list"
CHECK_TIMEOUT_SECONDS = 60.0
CHECK_GLYPHS = {"ok": "✓", "failed": "✗", "unproven": "?", "off": "·"}


class SandboxError(RuntimeError):
    """Raised when the Archil microVM cannot read the file (triggers fail-open in-process)."""


@dataclass(frozen=True)
class Staged:
    """One file staged by ``app/extractors.py`` as described in ``manifest.json``."""

    label: str
    path: Path
    sha256: str
    blob_key: str | None = None


@dataclass(frozen=True)
class Settings:
    """Archil runner configuration resolved from environment variables."""

    cli: tuple[str, ...]
    disk: str
    image: str
    timeout: str
    region: str
    token: str
    mount_dir: Path | None

    @classmethod
    def from_env(cls) -> "Settings":
        cli_raw = os.getenv("VA_LSE_ARCHIL_CLI", DEFAULT_CLI).strip() or DEFAULT_CLI
        try:
            parts = tuple(shlex.split(cli_raw))
        except ValueError as exc:
            raise SandboxError(f"VA_LSE_ARCHIL_CLI is not a valid command line: {exc}") from exc
        if not parts:
            raise SandboxError("VA_LSE_ARCHIL_CLI is empty — set it to `disk` or `npx -y disk`")
        mount_raw = os.getenv("VA_LSE_ARCHIL_MOUNT_DIR", "").strip()
        return cls(
            cli=parts,
            disk=os.getenv("VA_LSE_ARCHIL_DISK", DEFAULT_DISK).strip() or DEFAULT_DISK,
            image=os.getenv("VA_LSE_ARCHIL_IMAGE", DEFAULT_IMAGE).strip() or DEFAULT_IMAGE,
            timeout=os.getenv("VA_LSE_ARCHIL_TIMEOUT", DEFAULT_TIMEOUT).strip() or DEFAULT_TIMEOUT,
            region=os.getenv("VA_LSE_ARCHIL_REGION", "").strip(),
            token=archil_token_from_env(),
            mount_dir=Path(mount_raw).expanduser() if mount_raw else None,
        )


def archil_token_from_env() -> str:
    """Return the Archil API token from the environment, refusing LLM keys by prefix."""
    for name in TOKEN_ENV_NAMES:
        value = os.getenv(name, "").strip()
        if not value:
            continue
        if value.startswith(NON_ARCHIL_KEY_PREFIXES):
            raise SandboxError(
                f"{name} holds what looks like an LLM/Gateway API key ({value[:5]}…) — "
                "Archil requires an API token from https://console.archil.com/ (Access Tokens)"
            )
        return value
    return ""


def token_source() -> str:
    """Return the environment variable name that supplied the Archil token, or ''."""
    for name in TOKEN_ENV_NAMES:
        if os.getenv(name, "").strip():
            return name
    return ""


def _log(text: str) -> None:
    """Write progress and diagnostic messages to stderr so stdout stays pure JSON."""
    print(text, file=sys.stderr)


def _masked(argv: Sequence[str]) -> str:
    """Mask token values before logging CLI invocations."""
    shown: list[str] = []
    hide_next = False
    for arg in argv:
        if hide_next:
            shown.append("…")
            hide_next = False
            continue
        shown.append(arg)
        hide_next = arg in ("--token", "--api-key")
    return " ".join(shown)


def _detail(proc: subprocess.CompletedProcess[str]) -> str:
    """Extract the most actionable failure line from CLI stderr/stdout."""
    for stream in (proc.stderr, proc.stdout):
        lines = [line.strip() for line in (stream or "").splitlines() if line.strip()]
        if lines:
            return lines[-1][:300]
    return "no output"


@dataclass(frozen=True)
class Check:
    """Single diagnostic check item compatible with ``scripts/check_sandbox.py``."""

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
        line = f"{CHECK_GLYPHS.get(self.status, '?')} {self.name}: {self.detail}"
        if self.status != "ok" and self.remedy:
            line += f"\n    → {self.remedy}"
        return line


@dataclass(frozen=True)
class SelfCheck:
    """Aggregate box-free check verdict compatible with ``scripts/check_sandbox.py``."""

    checks: list[Check]

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == "failed"]

    @property
    def unproven(self) -> list[Check]:
        return [c for c in self.checks if c.status == "unproven"]

    @property
    def reachable(self) -> bool:
        return not self.failed

    def to_json(self) -> dict[str, Any]:
        return {
            "version": 1,
            "reachable": self.reachable,
            "checks": [c.to_json() for c in self.checks],
        }

    def verdict(self) -> str:
        if self.failed:
            names = ", ".join(c.name for c in self.failed)
            return (
                f"Archil sandbox would be skipped and records read in-process: {names} "
                f"{'is' if len(self.failed) == 1 else 'are'} not ready"
            )
        if self.unproven:
            names = ", ".join(c.name for c in self.unproven)
            return (
                "Archil serverless execution is reachable from this host; "
                f"{names} will be verified on first microVM execution"
            )
        return "Archil disk and serverless execution are reachable from this host"


class ArchilCli:
    """Wrapper around the Archil ``disk`` CLI for CoW branching and serverless exec."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def common(self) -> list[str]:
        flags: list[str] = []
        if self.settings.token:
            flags += ["--token", self.settings.token]
        if self.settings.region:
            flags += ["--region", self.settings.region]
        return flags

    def _run(
        self, argv: list[str], purpose: str, *, forward: bool = False
    ) -> subprocess.CompletedProcess[str]:
        full = [*self.settings.cli, *argv]
        _log(f"  $ {_masked(full)}")
        try:
            proc = subprocess.run(  # noqa: S603 - operator-configured argv, no shell
                full,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise SandboxError(
                f"the Archil CLI was not found ({exc.filename or self.settings.cli[0]}). "
                "Install it with `npm i -g disk` or set VA_LSE_ARCHIL_CLI='npx -y disk'"
            ) from exc
        if forward:
            for line in f"{proc.stdout or ''}{proc.stderr or ''}".splitlines():
                if line.strip():
                    _log(f"  | {line.strip()}")
        if proc.returncode != 0:
            raise SandboxError(f"{purpose} failed (exit {proc.returncode}): {_detail(proc)}")
        return proc

    def create_branch(self, branch_name: str) -> None:
        """Create an isolated copy-on-write branch on the configured Archil disk."""
        self._run(
            ["branch", "create", *self.common(), self.settings.disk, branch_name],
            f"create CoW branch {branch_name} on {self.settings.disk}",
        )

    def mkdir(self, branch_name: str, directory: PurePosixPath) -> None:
        """Create the parent directory structure inside the branch."""
        self._run(
            ["exec", *self.common(), "--disk", f"{self.settings.disk}:{branch_name}", "--", "mkdir", "-p", str(directory)],
            f"prepare {directory} on {branch_name}",
        )

    def put_file(self, branch_name: str, source: Path, remote_path: PurePosixPath, label: str) -> None:
        """Upload the staged file onto the CoW branch under its exact citation label."""
        self._run(
            ["cp", *self.common(), str(source), f"{self.settings.disk}:{branch_name}:{remote_path}"],
            f"stage {label} into {branch_name}",
        )

    def extract(self, branch_name: str, bundle: PurePosixPath, report: PurePosixPath) -> None:
        """Execute ``scripts/ocr_and_extract.py`` in an Archil serverless microVM colocated with the disk."""
        image_flags = (
            ["--image", self.settings.image]
            if self.settings.image.lower() not in ("", "none")
            else []
        )
        self._run(
            [
                "exec",
                *self.common(),
                "--disk",
                f"{self.settings.disk}:{branch_name}",
                *image_flags,
                "--timeout",
                self.settings.timeout,
                "--",
                BOX_PYTHON,
                BOX_ENTRYPOINT,
                str(bundle),
                "--data-class",
                "synthetic",
                "--out",
                str(report),
                "--force",
            ],
            f"run ocr_and_extract on {branch_name}",
            forward=True,
        )

    def extract_on_mount(self, bundle: PurePosixPath, report: PurePosixPath) -> None:
        """Execute ``scripts/ocr_and_extract.py`` against a zero-copy shared Archil disk mount."""
        image_flags = (
            ["--image", self.settings.image]
            if self.settings.image.lower() not in ("", "none")
            else []
        )
        self._run(
            [
                "exec",
                *self.common(),
                "--disk",
                self.settings.disk,
                *image_flags,
                "--timeout",
                self.settings.timeout,
                "--",
                BOX_PYTHON,
                BOX_ENTRYPOINT,
                str(bundle),
                "--data-class",
                "synthetic",
                "--out",
                str(report),
                "--force",
            ],
            f"run zero-copy ocr_and_extract on {self.settings.disk}",
            forward=True,
        )

    def get_file(self, branch_name: str, remote_path: PurePosixPath, destination: Path) -> None:
        """Copy the generated report.json from the CoW branch back to local work_dir."""
        self._run(
            ["cp", *self.common(), f"{self.settings.disk}:{branch_name}:{remote_path}", str(destination)],
            f"copy report out of {branch_name}",
        )

    def remove_branch(self, branch_name: str) -> None:
        """Best-effort cleanup of the ephemeral CoW branch."""
        try:
            self._run(
                ["branch", "delete", *self.common(), self.settings.disk, branch_name, "--force"],
                f"delete CoW branch {branch_name}",
            )
        except SandboxError as exc:
            _log(f"! {exc} — branch cleanup failed; prune stale va-lse-ocr-* branches in console.archil.com")


def _check_mount_dir(settings: Settings) -> Check:
    assert settings.mount_dir is not None
    if not settings.mount_dir.exists():
        return Check(
            "mount_dir",
            "failed",
            f"VA_LSE_ARCHIL_MOUNT_DIR={str(settings.mount_dir)!r} does not exist on this host",
            remedy="mount the Archil disk at VA_LSE_ARCHIL_MOUNT_DIR or unset VA_LSE_ARCHIL_MOUNT_DIR to use CoW branches",
        )
    if not settings.mount_dir.is_dir():
        return Check(
            "mount_dir",
            "failed",
            f"VA_LSE_ARCHIL_MOUNT_DIR={str(settings.mount_dir)!r} is not a directory",
            remedy="point VA_LSE_ARCHIL_MOUNT_DIR at the mounted Archil POSIX directory",
        )
    return Check("mount_dir", "ok", f"zero-copy POSIX mount ready at {settings.mount_dir}")


def self_check(settings: Settings, *, timeout: float = CHECK_TIMEOUT_SECONDS) -> SelfCheck:
    """Box-free verification of Archil CLI, credential, disk, optional mount, and sandbox image."""
    checks = [_check_cli(settings)]
    if checks[0].status == "ok":
        checks.append(_check_account(settings, timeout=timeout))
    if settings.mount_dir is not None:
        checks.append(_check_mount_dir(settings))
    checks.append(_check_image(settings))
    return SelfCheck(checks)


def _check_cli(settings: Settings) -> Check:
    program = settings.cli[0]
    found = shutil.which(program)
    if found is None:
        return Check(
            "cli",
            "failed",
            f"{program!r} (VA_LSE_ARCHIL_CLI={shlex.join(settings.cli)}) is not on PATH",
            remedy="install the Archil CLI (`npm i -g disk`) or set VA_LSE_ARCHIL_CLI='npx -y disk'",
        )
    return Check("cli", "ok", f"{shlex.join(settings.cli)} → {found}")


def _check_account(settings: Settings, *, timeout: float) -> Check:
    argv = [*settings.cli, CHECK_SUBCOMMAND, *ArchilCli(settings).common()]
    _log(f"  $ {_masked(argv)}")
    try:
        proc = subprocess.run(  # noqa: S603
            argv,
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return Check("credential", "failed", f"the Archil CLI could not be run: {exc}")
    except subprocess.TimeoutExpired:
        return Check(
            "credential",
            "failed",
            f"`{shlex.join(settings.cli)} {CHECK_SUBCOMMAND}` timed out after {timeout:.0f}s",
            remedy="create an API token at https://console.archil.com/ and set ARCHIL_API_KEY or VA_LSE_ARCHIL_TOKEN",
        )
    if proc.returncode != 0:
        return Check(
            "credential",
            "failed",
            f"`{CHECK_SUBCOMMAND}` was refused (exit {proc.returncode}): {_detail(proc)}",
            remedy="verify ARCHIL_API_KEY / VA_LSE_ARCHIL_TOKEN and confirm disk "
            f"{settings.disk!r} exists in https://console.archil.com/",
        )
    source = token_source() or "stored CLI session"
    return Check(
        "credential",
        "ok",
        f"`{CHECK_SUBCOMMAND}` succeeded for disk={settings.disk!r}, authenticated via {source}",
    )


def _check_image(settings: Settings) -> Check:
    if settings.image.lower() in ("", "none"):
        return Check(
            "image",
            "ok",
            "VA_LSE_ARCHIL_IMAGE=none — microVM uses default Archil Linux image",
        )
    return Check(
        "image",
        "unproven",
        f"{settings.image} will be pulled by the Archil serverless runtime on execution",
        remedy="build and publish the `sandbox` stage of Dockerfile (`docker build --target sandbox -t va-lse-sandbox:latest .`)",
    )


def read_manifest(work_dir: Path) -> Staged:
    """Validate and load the staged file metadata written by ``app/extractors.py``."""
    manifest = work_dir / MANIFEST_NAME
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SandboxError(
            f"{manifest} is missing — app/extractors.py writes it beside the staged record"
        ) from exc
    except (OSError, ValueError) as exc:
        raise SandboxError(f"{manifest} cannot be read: {exc}") from exc
    file_info = raw.get("file") if isinstance(raw, dict) else None
    if not isinstance(file_info, dict) or not file_info.get("label") or not file_info.get("path"):
        raise SandboxError(f"{manifest} has no file.label/file.path")
    staged = Staged(
        label=str(file_info["label"]),
        path=Path(str(file_info["path"])),
        sha256=str(file_info.get("sha256") or ""),
        blob_key=str(file_info["blob_key"]) if file_info.get("blob_key") else None,
    )
    if not staged.path.is_file():
        raise SandboxError(f"the staged file {staged.path} does not exist")
    if staged.sha256:
        actual = hashlib.sha256(staged.path.read_bytes()).hexdigest()
        if actual != staged.sha256:
            raise SandboxError(
                f"the staged file {staged.path} sha256 ({actual}) does not match manifest ({staged.sha256})"
            )
    return staged


def relative_label(label: str) -> PurePosixPath:
    """Preserve the exact relative path for citation integrity while refusing traversal."""
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


def _run_zero_copy_mount(work_dir: Path, staged: Staged, settings: Settings) -> str:
    """Zero-copy shared-disk execution when ``VA_LSE_ARCHIL_MOUNT_DIR`` is mounted locally."""
    assert settings.mount_dir is not None
    if not settings.mount_dir.is_dir():
        raise SandboxError(
            f"VA_LSE_ARCHIL_MOUNT_DIR={str(settings.mount_dir)!r} is not a mounted directory"
        )
    job_id = f"va-lse-ocr-{uuid.uuid4().hex[:12]}"
    job_dir = settings.mount_dir / "jobs" / job_id
    bundle_dir = job_dir / "bundle"
    staged_target = bundle_dir / Path(*relative_label(staged.label).parts)
    report_target = job_dir / REPORT_NAME
    remote_job = PurePosixPath("/work/jobs") / job_id
    remote_bundle = remote_job / "bundle"
    remote_report = remote_job / REPORT_NAME

    cli = ArchilCli(settings)
    _log(
        f"archil zero-copy mount {settings.mount_dir} ({job_id}): {staged.label} → {settings.image} "
        f"(timeout {settings.timeout})"
    )
    try:
        staged_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged.path, staged_target)
        cli.extract_on_mount(remote_bundle, remote_report)
        if not report_target.is_file():
            raise SandboxError(f"zero-copy run did not produce {report_target}")
        report_text = report_target.read_text(encoding="utf-8")
        (work_dir / REPORT_NAME).write_text(report_text, encoding="utf-8")
        return report_text
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


def run_for_one_file(work_dir: Path, settings: Settings) -> str:
    """Stage one file onto an Archil mount or ephemeral CoW branch, run OCR/extraction, and return JSON."""
    staged = read_manifest(work_dir)
    if settings.mount_dir is not None:
        return _run_zero_copy_mount(work_dir, staged, settings)

    remote_record = REMOTE_BUNDLE / relative_label(staged.label)
    branch_name = f"va-lse-ocr-{uuid.uuid4().hex[:12]}"
    cli = ArchilCli(settings)
    _log(
        f"archil branch {settings.disk}:{branch_name}: {staged.label} → {settings.image} "
        f"(timeout {settings.timeout}, CoW ephemeral)"
    )

    created = False
    try:
        cli.create_branch(branch_name)
        created = True
        cli.mkdir(branch_name, remote_record.parent)
        cli.put_file(branch_name, staged.path, remote_record, staged.label)
        cli.extract(branch_name, REMOTE_BUNDLE, REMOTE_REPORT)
        report_path = work_dir / REPORT_NAME
        cli.get_file(branch_name, REMOTE_REPORT, report_path)
        return report_path.read_text(encoding="utf-8")
    finally:
        if created:
            cli.remove_branch(branch_name)


def _handle_termination(signum: int, _frame: Any) -> None:
    raise SystemExit(128 + signum)


def _install_cleanup_on_termination() -> None:
    for name in ("SIGTERM", "SIGHUP"):
        value = getattr(signal, name, None)
        if value is None:
            continue
        try:
            signal.signal(value, _handle_termination)
        except (ValueError, OSError):
            return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read one staged file ({work}) in an Archil Serverless microVM and print the "
            "report JSON written by scripts/ocr_and_extract.py. Configure with "
            'VA_LSE_EXTRACTOR_RUNNER="python scripts/archil_sandbox_runner.py {work}".'
        )
    )
    parser.add_argument(
        "work",
        type=Path,
        nargs="?",
        help="the staging directory containing manifest.json",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="run box-free diagnostics and print JSON verdict to stdout for scripts/check_sandbox.py",
    )
    return parser


def _run_check() -> int:
    _log("archil sandbox self-check — no branch or microVM is created")
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
                f"the Archil microVM's {REPORT_NAME} has no 'documents' list — it is not "
                "the report scripts/ocr_and_extract.py --out writes"
            )
    except SandboxError as exc:
        _log(f"✖ {exc}")
        return EXIT_FAILED
    except ValueError as exc:
        _log(f"✖ the Archil microVM's {REPORT_NAME} is not JSON: {exc}")
        return EXIT_FAILED
    sys.stdout.write(report_text if report_text.endswith("\n") else report_text + "\n")
    sys.stdout.flush()
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())