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
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
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
    "leave VA_LSE_EXTRACTOR unset to read records in-process."
)


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
        """The argv for one file, with ``{work}`` substituted and nothing else."""
        return [part.replace("{work}", str(work_dir)) for part in shlex.split(self.template)]

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

        ``extract_document`` keeps the name it is given (``.zip`` members keep the
        archive's name and a separator), so every legitimate label here either is
        the staged label or extends it. Anything else means the box invented a
        name — usually a staging path or an ``.ocr.pdf`` copy — and those must not
        reach citations.
        """
        name = document.filename
        if name == staged.label or name.startswith(f"{staged.label}:"):
            return document
        raise SandboxUnavailable(
            f"the box answered for {staged.label} under the name {name!r}. Citations must "
            "point at the file the user has, so this answer is refused and the file is read "
            "in-process instead (see scripts/ocr_and_extract.py — it extracts under the "
            "original file name on purpose)"
        )


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
        return None
    if mode != "sandbox":
        logger.warning(
            "VA_LSE_EXTRACTOR=%r is not a mode this app knows (in-process | sandbox); "
            "reading records in-process",
            mode,
        )
        return None
    from .blob_store import get_blob_store  # local: only a sandbox needs the store

    runner_template = (config.EXTRACTOR_RUNNER or "").strip()
    if not runner_template:
        # Reported, not raised: an app that will not start because a box is
        # misconfigured is worse than an app that reads records itself and says so.
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
        report_failure(
            f"⚠️ {exc}", phase="extractor_sandbox", exc=exc, severity="warning", once=True
        )
        return None

    store = get_blob_store()
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
