"""Extraction adapters: the reader in this process, or a sandbox that can OCR.

Why this module exists
----------------------
``app/documents.py`` extracts text and never shells out, and that is on purpose:
the deployment image must not carry a PDF renderer and an OCR engine, so a page
that is a *scan* has no text to extract — the uploader counts it
(``ExtractedDocument.unreadable_pages``), warns the user, and asks them to run
``scripts/ocr_records.py`` on the file and upload it again. Records from a
records request are routinely half scans, so that gap is real.

The sandbox image is the machine that can close it: ``tesseract-ocr``,
``poppler-utils``, ``ghostscript``, ``qpdf`` and ``ocrmypdf`` are installed in
the Dockerfile's ``sandbox`` stage and in no other, and
``scripts/ocr_and_extract.py`` is its entrypoint (OCR every image-only page, then
extract with *this app's own reader*, then emit the queue's document JSON).

This module is the swap, not a rewrite: :class:`SandboxExtractor` implements the
same ``RecordExtractor`` port as ``InProcessExtractor``, and
:class:`FailOpenExtractor` puts the in-process reader underneath it so the app is
never worse off than it is today. Selecting it is configuration
(``VA_LSE_EXTRACTOR=sandbox`` plus a runner command), installed once at startup
by :func:`install_configured_extractor` — the uploader and the local-folder
reader call the same functions they always did.

Two rules the code below enforces, because both are how this goes wrong:

* **Labels are the citation backbone.** The box answers under the name the user
  has, never ``.ocr.pdf`` or a staging filename; a document that comes back under
  a name this app never handed over is refused (and the run falls back), because
  a citation pointing at a file the user does not have is worse than a slow read.
* **Fail open, loudly.** No runner, no tooling on the box, a refused engine, a
  timeout, unparseable JSON — each one is logged once through
  ``app.error_report`` with a correlation id, and the file is read in-process
  instead. The user sees today's behavior plus a warning they can quote; nothing
  is silently lost.

Transfer reuses what the queue already has: the file is staged into a working
directory (what a runner on this machine reads directly) *and*, best effort, into
``app.blob_store`` (content-addressed, already shared with worker pods) so a box
that cannot see this filesystem can fetch it by key. The box's answer is stored
the same way, so what it said is a blob a worker could read — the queue's
document JSON, ``app.job_payload.documents_from_json``, is the only schema on the
wire.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from . import config
from . import documents as documents_mod
from . import pipeline_guard
from .blob_store import BlobStore, BlobStoreError, dumps_documents
from .documents import ExtractedDocument, InProcessExtractor, RecordExtractor
from .error_report import report_failure

logger = logging.getLogger("app.extractors")

#: Remedy text shared by every "the box did not work" path, so the operator sees
#: the same sentence wherever the fallback is reported.
_RUNNER_HELP = (
    "Set VA_LSE_EXTRACTOR_RUNNER to the command that runs "
    "scripts/ocr_and_extract.py in the box (see DEPLOYMENT.md → Sandbox), or "
    "leave VA_LSE_EXTRACTOR unset to read records in-process. "
    "`python scripts/check_sandbox.py` answers the same question before a run does, "
    "without creating a box."
)

#: How a bare Python interpreter is spelled in a runner command: ``python``,
#: ``python3``, ``python3.12``, or Windows' ``py`` launcher, with or without ``.exe``.
_PYTHON_PROGRAM = re.compile(r"^(?:python(?:3(?:\.\d+)?)?|py)(?:\.exe)?$", re.IGNORECASE)

#: A Python interpreter named by *path* (``…/.venv/bin/python``), as opposed to a
#: bare name. Matched on the last component so a directory called ``python-bin``
#: or a runner named ``python-lint`` is not mistaken for an interpreter.
_PYTHON_INTERPRETER_PATH = re.compile(r"[/\\](?:python(?:3(?:\.\d+)?)?|py)(?:\.exe)?$", re.IGNORECASE)

#: A leading ``VAR=value`` word (``PYTHONPATH=. python scripts/… {work}``), which
#: ``shlex.split`` hands over as an ordinary token.
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


@dataclass(frozen=True)
class InterpreterSwap:
    """A first word that named an interpreter this host does not have, and its stand-in.

    ``severity`` is what the operator needs to hear, and the two reasons differ: a
    bare ``python`` on a host that spells it ``python3`` is normal and is news at
    INFO, while a *configured path* that is not on disk is a broken deployment and
    deserves a WARNING before it costs a run. Both say which interpreter took over,
    so the line is actionable either way.
    """

    written: str
    resolved: str
    reason: str
    severity: str

    def message(self) -> str:
        return (
            f"the runner command starts with {self.written!r}, which is {self.reason} — "
            f"running it with {self.resolved}"
        )


def resolve_python_interpreter(argv: list[str]) -> tuple[list[str], InterpreterSwap | None]:
    """Use this process's interpreter when a runner command names one the host lacks.

    ``VA_LSE_EXTRACTOR_RUNNER="python scripts/vercel_sandbox_runner.py {work}"`` is
    the command ``.env.example`` and the docs show, and on a host that exposes only
    ``python3`` — macOS, most Debian images, and this repository's own virtualenv —
    it used to die in ``subprocess.run`` with ``the runner command does not exist
    (python)``, so the very first attempt on a real box fell back for a reason that
    had nothing to do with the box. The same happens to a configured interpreter
    path that has moved or been deleted (``…/.venv-sandbox/bin/python``).

    The app is already running under an interpreter that exists, and the runner is
    stdlib-only operator tooling, so in both cases that interpreter is what the
    command means. A spelling that *is* on ``PATH`` — or a path that *is* on disk —
    is left exactly as written (the operator picked it, version pin included), and a
    word that names no interpreter is never touched: a non-Python runner, an
    environment assignment in front, and ``PYTHONPATH=. python3 …`` all behave as
    before.

    Returns the argv to run and what was swapped (with the sentence to log), or
    ``None`` when the command was already runnable as written.
    """
    program = 0
    while program < len(argv) and _ENV_ASSIGNMENT.match(argv[program]):
        program += 1
    if program >= len(argv):
        return argv, None
    # ``sys.executable`` is empty in embedded interpreters, which leaves the command
    # alone: the FileNotFoundError message then names the word the operator wrote,
    # which is the truth about that command, and nothing here could improve on it.
    if not sys.executable:
        return argv, None
    written = argv[program]
    if _PYTHON_PROGRAM.match(written):
        if shutil.which(written) is not None:
            return argv, None
        reason, severity = "not on PATH", "INFO"
    elif _PYTHON_INTERPRETER_PATH.search(written):
        if Path(written).exists():
            return argv, None
        reason, severity = "not on this host", "WARNING"
    else:
        return argv, None
    resolved = list(argv)
    resolved[program] = sys.executable
    return resolved, InterpreterSwap(written, sys.executable, reason, severity)


class SandboxUnavailable(RuntimeError):
    """Raised when a box cannot be used, with the reason the user will read."""


@dataclass
class StagedFile:
    """What the box is handed: a name it must answer under, and where the bytes are."""

    label: str
    path: Path
    sha256: str
    size: int
    blob_key: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "path": str(self.path),
            "sha256": self.sha256,
            "size": self.size,
            "blob_key": self.blob_key,
        }


class BoxRunner(Protocol):
    """How a staged file reaches the box and how its answer comes back.

    One method on purpose: everything else — staging, validation, mapping to this
    app's types, falling back — is the same for a subprocess on this machine and
    for a CLI that copies files into a microVM. Implementations return the box's
    report JSON **text**; parse failures are handled here, not by the runner.
    """

    def run(self, staged: StagedFile, work_dir: Path, timeout: float) -> str:
        """Run the entrypoint over *staged* and return its report JSON text."""
        ...


class CommandBoxRunner:
    """Run a command per file, with ``{work}`` replaced by the staging directory.

    The command is the operator's, not this app's invention:

        VA_LSE_EXTRACTOR_RUNNER="python scripts/ocr_and_extract.py {work} \\
            --out {work}/bundle.json && cat {work}/bundle.json"

    The last thing on stdout must be the report JSON (the shape
    ``scripts/ocr_and_extract.py --out`` writes). A command that exits non-zero
    is a box that failed, and the app falls back rather than guessing — including
    the entrypoint's own exit 2, "scans are present and no OCR tooling is
    installed", which is exactly a box that cannot do this job.
    """

    def __init__(self, template: str) -> None:
        self.template = template.strip()
        if not self.template:
            raise SandboxUnavailable(f"No runner command configured. {_RUNNER_HELP}")

    def command_for(self, work_dir: Path) -> list[str]:
        """The argv for one file, with ``{work}`` substituted and nothing else.

        The first word can be normalized: a bare ``python``/``python3`` this host
        does not expose becomes the interpreter running this app
        (:func:`resolve_python_interpreter`), because the documented runner command
        says ``python`` and hosts that only have ``python3`` are the common case.
        """
        argv, swapped = resolve_python_interpreter(shlex.split(self.template))
        if swapped is not None:
            getattr(logger, swapped.severity.lower())("%s", swapped.message())
        return [part.replace("{work}", str(work_dir)) for part in argv]

    def run(self, staged: StagedFile, work_dir: Path, timeout: float) -> str:
        try:
            proc = subprocess.run(  # noqa: S603 - operator-supplied argv, no shell
                self.command_for(work_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise SandboxUnavailable(
                f"the runner command does not exist ({exc.filename or 'unknown'}). {_RUNNER_HELP}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise SandboxUnavailable(
                f"the box did not answer within {timeout:.0f}s for {staged.label} — "
                "raise VA_LSE_EXTRACTOR_TIMEOUT_SECONDS if this bundle is legitimately slow"
            ) from exc
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            detail = tail[-1][:200] if tail else "no output"
            raise SandboxUnavailable(
                f"the box refused {staged.label} (exit {proc.returncode}): {detail}"
            )
        return proc.stdout


class SandboxExtractor:
    """``RecordExtractor`` backed by a box running ``scripts/ocr_and_extract.py``.

    A box is per *file*, not per bundle: the app's port is one file in, documents
    out, and the entrypoint handles exactly that. Files with text cost no more
    than today (they are read on the box and the text comes back), scans come back
    with text where the app would have counted a blank page.
    """

    def __init__(
        self,
        runner: BoxRunner,
        *,
        store: BlobStore | None = None,
        timeout_seconds: float | None = None,
        work_root: Path | None = None,
    ) -> None:
        self.runner = runner
        self.store = store
        self.timeout_seconds = timeout_seconds or config.EXTRACTOR_TIMEOUT_SECONDS
        self.work_root = work_root

    # -- the port ------------------------------------------------------------
    def extract(self, label: str, data: bytes) -> tuple[list[ExtractedDocument], list[str]]:
        """Extract one file on the box, refusing to guess about its answer."""
        # A user Cancel must stop box work too. Raises PipelineCancelledError,
        # which is a BaseException precisely so no ``except Exception`` here eats it.
        pipeline_guard.check_pipeline_cancelled()

        work_dir = Path(tempfile.mkdtemp(prefix="va-lse-box-", dir=self.work_root))
        try:
            staged = self._stage(label, data, work_dir)
            (work_dir / "manifest.json").write_text(
                json.dumps({"version": 1, "file": staged.to_json()}, indent=2),
                encoding="utf-8",
            )
            report = self._run(staged, work_dir)
            documents, skipped = self._documents_from(report, staged)
            self._keep_report(report)
            return documents, skipped
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    # -- internals -----------------------------------------------------------
    def _stage(self, label: str, data: bytes, work_dir: Path) -> StagedFile:
        """Put the bytes where the box can read them, under their own name.

        The suffix is preserved because both the entrypoint and this app's reader
        choose a parser from it; the directory part of ``label`` is not, because
        the box has no such tree and the label travels in the manifest instead.
        """
        name = Path(label.replace("\\", "/")).name or "record"
        staged = StagedFile(
            label=label,
            path=work_dir / name,
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
        )
        staged.path.write_bytes(data)
        staged.blob_key = self._blob_put(label, data)
        return staged

    def _blob_put(self, label: str, data: bytes) -> str | None:
        """Best-effort: a key a box that cannot see this filesystem can fetch.

        Deliberately not fatal. A store in ``none`` mode, a file over the queue's
        size ceiling, an unreachable S3 endpoint — none of those should stop a
        read that a local runner can do, so the failure is logged and the
        manifest says ``blob_key: null``.
        """
        if self.store is None:
            return None
        envelope = {
            "version": 1,
            "kind": "record-file",
            "label": label,
            "file_b64": _b64(data),
        }
        try:
            ref = self.store.put(dumps_documents(envelope))
        except (BlobStoreError, NotImplementedError, OSError) as exc:
            logger.warning("could not stage %s in the blob store: %s", label, exc)
            return None
        return ref.key

    def _keep_report(self, report: dict[str, Any]) -> None:
        """Store what the box said, so a worker could read it (best effort)."""
        if self.store is None:
            return
        try:
            self.store.put(dumps_documents(report))
        except (BlobStoreError, NotImplementedError, OSError) as exc:
            logger.warning("could not store the box's report: %s", exc)

    def _remaining_seconds(self) -> float:
        """Per-box timeout: this app's own budget, and never past the run's."""
        timeout = float(self.timeout_seconds)
        remaining = pipeline_guard.pipeline_remaining_seconds()
        if remaining is not None:
            if remaining <= 0:
                raise SandboxUnavailable(
                    "the run's time budget is spent, so the box is not started for this file"
                )
            timeout = min(timeout, remaining)
        return max(1.0, timeout)

    def _run(self, staged: StagedFile, work_dir: Path) -> dict[str, Any]:
        raw = self.runner.run(staged, work_dir, self._remaining_seconds())
        try:
            report = json.loads(raw)
        except ValueError as exc:
            raise SandboxUnavailable(
                f"the box's answer for {staged.label} was not JSON "
                f"({len(raw):,} bytes on stdout): {exc}"
            ) from exc
        if not isinstance(report, dict) or not isinstance(report.get("documents"), list):
            raise SandboxUnavailable(
                f"the box's answer for {staged.label} has no 'documents' list — it is "
                "not the report scripts/ocr_and_extract.py --out writes"
            )
        return report

    def _documents_from(
        self, report: dict[str, Any], staged: StagedFile
    ) -> tuple[list[ExtractedDocument], list[str]]:
        """Rebuild this app's documents from the queue JSON the box produced."""
        from .job_payload import documents_from_json  # local: keeps import cost off the default path

        documents = documents_from_json(report["documents"])
        documents = [self._check_labels(staged, document) for document in documents]
        skipped = [str(message) for message in report.get("skipped", []) if str(message).strip()]
        if not documents and not skipped:
            raise SandboxUnavailable(
                f"the box returned nothing at all for {staged.label} — no documents and no "
                "reason, which is not an answer this app can use"
            )
        return documents, skipped

    def _check_labels(self, staged: StagedFile, document: ExtractedDocument) -> ExtractedDocument:
        """Refuse a document the box answers under a name the user never had.

        ``extract_document`` keeps the name it is given, so every legitimate label
        here either is the staged label or extends it — and *how* it extends it is
        the archive shape: ``app.documents.archive_members`` names a ``.zip``'s
        members ``<archive stem>/<member>``, so ``records.zip`` legitimately answers
        as ``records/visit.pdf``. (``<label>:`` is also accepted because a runner
        may pass an explicit separator through.) Anything else means the box
        invented a name — usually a staging path or an ``.ocr.pdf`` copy — and those
        must not reach citations.
        """
        name = document.filename
        archive_prefix = f"{Path(staged.label).stem or 'archive'}/"
        if name == staged.label or name.startswith((f"{staged.label}:", archive_prefix)):
            return document
        raise SandboxUnavailable(
            f"the box answered for {staged.label} under the name {name!r}. Citations must "
            "point at the file the user has, so this answer is refused and the file is read "
            "in-process instead (see scripts/ocr_and_extract.py — it extracts under the "
            "original file name on purpose)"
        )


# -- which reader this process runs, and how that has gone ---------------------


def _now() -> str:
    """Wall clock in the shape ``/health`` uses: UTC, second resolution."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass(frozen=True)
class Fallback:
    """One file the box could not read, and the box's own reason for it."""

    label: str
    reason: str
    at: str

    def to_json(self) -> dict[str, str]:
        return {"label": self.label, "reason": self.reason, "at": self.at}


@dataclass(frozen=True)
class ExtractionStatus:
    """Which reader this process runs, and whether it has already fallen back.

    Reported by ``/health`` and by the sidebar, because otherwise the failure this
    describes is invisible: ``sandbox`` mode falls back *per file* by design, so a box
    that cannot be reached produces a finished run whose scans simply have no text,
    and the only evidence is one warning in the log.

    Configuration plus outcome, no probing: the image, the credential and the scope
    belong to the runner (``scripts/vercel_sandbox_runner.py`` reads them), and
    ``scripts/check_sandbox.py`` reports those. This is what *this* process decided and
    what has happened since — the two things a run cannot be trusted to tell you.
    """

    mode: str = "in-process"
    runner: str = ""
    timeout_seconds: float = 0.0
    problem: str = ""
    fallbacks: int = 0
    last_fallback: Fallback | None = None

    @property
    def on_the_box(self) -> bool:
        """True when the app *means* to read on a box; ``problem`` says whether it can."""
        return self.mode == "sandbox"

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "runner": self.runner,
            "timeout_seconds": self.timeout_seconds,
            "problem": self.problem,
            "fallbacks": self.fallbacks,
            "last_fallback": self.last_fallback.to_json() if self.last_fallback else None,
        }


#: The one status this process has. Written where the decision is made
#: (:func:`build_extractor`) and where the fallback happens (:class:`FailOpenExtractor`),
#: under a lock because Streamlit runs one thread per browser session and the health
#: sidecar is a third.
_status_lock = threading.Lock()
_status = ExtractionStatus()


def extraction_status() -> ExtractionStatus:
    """A consistent snapshot, for anything that reports it (``/health``, the sidebar)."""
    with _status_lock:
        return _status


def extraction_health() -> dict[str, Any]:
    """The ``/health`` shape of :func:`extraction_status` — configuration, no probe."""
    return extraction_status().to_json()


def record_configuration(
    *, mode: str, runner: str = "", timeout_seconds: float = 0.0, problem: str = ""
) -> None:
    """What this process decided to read records with, recorded where it decided it.

    Called by :func:`build_extractor` on every branch, including the two broken ones, so
    "the app is configured for a sandbox it cannot reach" is visible *before* a run
    rather than as a warning after one.

    Re-recording keeps the fallback history: the entry module can be imported more than
    once in a process (a Streamlit reload, tests), and a re-install that silently wiped
    the evidence of a box that already failed would hide exactly what this is for. Only
    :func:`reset_extraction_status` clears it.
    """
    global _status
    with _status_lock:
        _status = replace(
            _status,
            mode=mode,
            runner=runner,
            timeout_seconds=timeout_seconds,
            problem=problem,
        )


def record_fallback(reason: str, label: str) -> None:
    """One file read here instead of on the box: counted, with the last reason kept.

    Only the last reason is kept on purpose — the sidebar cannot show twenty, the count
    says how widespread it is, and the log has every one of them (``report_failure``,
    once per distinct reason).
    """
    global _status
    with _status_lock:
        _status = replace(
            _status,
            fallbacks=_status.fallbacks + 1,
            last_fallback=Fallback(label=label, reason=reason, at=_now()),
        )


def reset_extraction_status() -> None:
    """Back to "in-process, nothing has failed" — for tests, and for a fresh start."""
    global _status
    with _status_lock:
        _status = ExtractionStatus()


class FailOpenExtractor:
    """``RecordExtractor`` that tries the box and falls back to this process.

    Wrapping rather than embedding is what makes the fallback total: every way the
    box can fail arrives here as one exception type, is reported once with a
    correlation id, and then the reader that ships with the app does the work.
    """

    def __init__(self, box: RecordExtractor, fallback: RecordExtractor | None = None) -> None:
        self.box = box
        self.fallback = fallback or InProcessExtractor()

    def extract(self, label: str, data: bytes) -> tuple[list[ExtractedDocument], list[str]]:
        try:
            return self.box.extract(label, data)
        except SandboxUnavailable as exc:
            # Recorded before it is reported: the warning is once per distinct reason
            # (below), and "the sandbox is failing" has to be visible without the log —
            # on /health and in the sidebar — because the run itself looks fine.
            record_fallback(str(exc), label)
            # One warning per distinct reason, not per file: a box that is down
            # turns a 20-file bundle into 20 identical failures, and the point of
            # the line is that the operator finds it, once. The file that fell back
            # is on the info line below, so nothing is unattributed either way.
            report_failure(
                f"⚠️ Reading records in-process instead of on the sandbox — {exc}",
                phase="extractor_sandbox",
                exc=exc,
                severity="warning",
                once=True,
            )
            logger.info("sandbox extraction fell back to in-process for %s", label)
            return self.fallback.extract(label, data)


def build_extractor() -> RecordExtractor | None:
    """The extractor configuration asks for, or ``None`` for the in-process default.

    ``None`` is meaningful: ``set_active_extractor(None)`` restores the reader in
    ``app/documents.py``, so an unset or unrecognized mode cannot end up with a
    box-shaped object that fails every file.
    """
    mode = (config.EXTRACTOR_MODE or "in-process").strip().lower()
    if mode in ("", "in-process", "inprocess", "local"):
        record_configuration(mode="in-process", timeout_seconds=config.EXTRACTOR_TIMEOUT_SECONDS)
        return None
    if mode != "sandbox":
        logger.warning(
            "VA_LSE_EXTRACTOR=%r is not a mode this app knows (in-process | sandbox); "
            "reading records in-process",
            mode,
        )
        record_configuration(
            mode="in-process",
            timeout_seconds=config.EXTRACTOR_TIMEOUT_SECONDS,
            problem=(
                f"VA_LSE_EXTRACTOR={mode!r} is not a mode this app knows "
                "(in-process | sandbox) — records are read in-process"
            ),
        )
        return None
    from .blob_store import get_blob_store  # local: only a sandbox needs the store

    runner_template = (config.EXTRACTOR_RUNNER or "").strip()
    if not runner_template:
        # Reported, not raised: an app that will not start because a box is
        # misconfigured is worse than an app that reads records itself and says so.
        record_configuration(
            mode="sandbox", problem=f"VA_LSE_EXTRACTOR=sandbox but no runner is set. {_RUNNER_HELP}"
        )
        report_failure(
            f"⚠️ VA_LSE_EXTRACTOR=sandbox but no runner is configured. {_RUNNER_HELP}",
            phase="extractor_sandbox",
            severity="warning",
            once=True,
        )
        return None
    try:
        runner = CommandBoxRunner(runner_template)
    except SandboxUnavailable as exc:
        record_configuration(mode="sandbox", runner=runner_template, problem=str(exc))
        report_failure(
            f"⚠️ {exc}", phase="extractor_sandbox", exc=exc, severity="warning", once=True
        )
        return None

    store = get_blob_store()
    record_configuration(
        mode="sandbox",
        runner=runner_template,
        timeout_seconds=config.EXTRACTOR_TIMEOUT_SECONDS,
    )
    logger.info(
        "extract records on the sandbox (timeout %ss, blob store %s)",
        config.EXTRACTOR_TIMEOUT_SECONDS,
        store.name,
    )
    return FailOpenExtractor(
        SandboxExtractor(
            runner,
            store=None if store.name == "none" else store,
            timeout_seconds=config.EXTRACTOR_TIMEOUT_SECONDS,
        )
    )


def install_configured_extractor() -> RecordExtractor:
    """Point ``app.documents`` at the configured extractor; return what is active.

    Called once from the app entry point. Idempotent, because the entry module can
    be imported more than once in a process (tests, ``streamlit run`` reloads) and
    installing twice must not stack wrappers around wrappers.
    """
    extractor = build_extractor()
    active = documents_mod.set_active_extractor(extractor)
    return active


def _b64(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii")
