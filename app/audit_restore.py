"""Restore and integrity-verify backed-up audit logs.

Why this exists
---------------
A backup nobody has read back is a hypothesis, not a backup. ``app/audit_backup.py``
proves it *uploads*; nothing proved the objects are intact, complete, or
reassemblable into the stream an investigator would want. That gap is the usual
reason a compliance backup turns out to be useless at the moment it is needed:
the job has been green for months while the data sits in an unusable shape.

Three questions this answers
----------------------------
1. **Is the data intact?** Every object key already ends in
   ``sha256(content)[:12]``, so integrity checking needs no side manifest and
   cannot drift from the data: the expected hash travels *with* the object. A
   downloaded object whose bytes do not hash to the value in its own key is
   reported as corrupt.
2. **Is anything missing?** Live-log windows are byte ranges. Walking them in
   upload order within each file generation reveals holes — the bytes written
   while the backup job was failing. A gap is reported with its exact offsets,
   which is what tells an investigator whether a specific time window survived.
3. **Can the stream be rebuilt?** :func:`restore_backup` writes the windows back
   out as one chronological ``restored.jsonl`` plus the rotated-file snapshots and
   a manifest describing what was and was not recoverable.

Generations, and why the key carries one
----------------------------------------
An ``audit.log`` is rotated, not appended forever, so the stream is a sequence of
*files*, each numbered from offset 0. Grouping windows by filename and offset is
therefore ambiguous exactly when it matters — right after a rotation, when the new
file's window and a retry of the old file's window are both ``0-N``. That is why
the object key records a generation tag (see
:func:`app.audit_backup.generation_tag`). Windows from a key without one are
reported as generation ``unknown`` rather than merged into a neighbour.

Deliberate limits
-----------------
This module **reads**; it never writes to the destination and never mutates the
backup state file. Restoring is an operator action, and a tool that can delete
objects during an incident is a tool that can destroy the only copy.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import config
from .audit_backup import (
    BackupDestination,
    BackupError,
    NullDestination,
    _sha12,
    build_destination,
    destination_summary,
)

logger = logging.getLogger("app.audit_restore")

# A live window whose key carries no generation tag: written by an older version,
# or by a platform that exposes no file identity (see generation_tag).
UNKNOWN_GENERATION = "unknown"

_LIVE_RE = re.compile(
    # <kind>-<filename>-g<tag>-<start>-<end>-<sha12>.jsonl
    r"^audit/(?P<date>\d{4}/\d{2}/\d{2})/"
    r"live-(?P<filename>.+?)-g(?P<generation>[0-9a-f]+)"
    r"-(?P<start>\d+)-(?P<end>\d+)-(?P<sha>[0-9a-f]{12})\.jsonl$"
)
# The pre-generation key shape: no g segment.
_LIVE_LEGACY_RE = re.compile(
    r"^audit/(?P<date>\d{4}/\d{2}/\d{2})/"
    r"live-(?P<filename>.+?)-(?P<start>\d+)-(?P<end>\d+)-(?P<sha>[0-9a-f]{12})\.jsonl$"
)
_ROTATED_RE = re.compile(
    # <kind>-<filename>-<size>-<sha12>.jsonl
    r"^audit/(?P<date>\d{4}/\d{2}/\d{2})/"
    r"rotated-(?P<filename>.+?)-(?P<size>\d+)-(?P<sha>[0-9a-f]{12})\.jsonl$"
)


@dataclass(frozen=True)
class ObjectRef:
    """A parsed backup object key."""

    key: str
    kind: str  # "live" | "rotated"
    date: str
    filename: str
    start: int
    end: int
    sha12: str
    size: int
    generation: str
    last_modified: datetime | None = None

    @property
    def range_bytes(self) -> int:
        return max(0, self.end - self.start)


def parse_object_key(key: str) -> ObjectRef | None:
    """Parse a backup object key, or return None when it is not one of ours.

    Unrecognised keys are collected by the caller rather than dropped: an object
    in the destination this tool cannot explain is itself worth reporting during a
    forensic review.
    """
    match = _ROTATED_RE.match(key)
    if match:
        size = int(match.group("size"))
        return ObjectRef(
            key=key,
            kind="rotated",
            date=match.group("date").replace("/", "-"),
            filename=match.group("filename"),
            start=0,
            end=size,
            sha12=match.group("sha"),
            size=size,
            generation=UNKNOWN_GENERATION,
        )
    # Order matters: _LIVE_LEGACY_RE also matches a new-shape key (it absorbs the
    # ``g<tag>`` segment into the filename), so the tagged pattern is tried first.
    match = _LIVE_RE.match(key)
    if match:
        start = int(match.group("start"))
        end = int(match.group("end"))
        return ObjectRef(
            key=key,
            kind="live",
            date=match.group("date").replace("/", "-"),
            filename=match.group("filename"),
            start=start,
            end=end,
            sha12=match.group("sha"),
            size=max(0, end - start),
            generation=match.group("generation"),
        )
    legacy = _LIVE_LEGACY_RE.match(key)
    if legacy:
        start = int(legacy.group("start"))
        end = int(legacy.group("end"))
        return ObjectRef(
            key=key,
            kind="live",
            date=legacy.group("date").replace("/", "-"),
            filename=legacy.group("filename"),
            start=start,
            end=end,
            sha12=legacy.group("sha"),
            size=max(0, end - start),
            generation=UNKNOWN_GENERATION,
        )
    return None


# ------------------------------------------------------------------ structures


@dataclass
class ObjectCheck:
    """Outcome of downloading and hashing one object."""

    key: str
    ok: bool
    # Named ``size`` rather than ``bytes``: a field called ``bytes`` shadows the
    # builtin inside the class body and breaks the annotation on the next line.
    size: int = 0
    expected_sha12: str = ""
    actual_sha12: str = ""
    error: str = ""
    lines: int = 0
    malformed_lines: int = 0
    # Kept so line counting does not re-download the object. Never serialised.
    data: bytes | None = field(default=None, repr=False)


@dataclass
class Gap:
    """A byte range that was never uploaded — the records lost in it."""

    filename: str
    generation: str
    start: int
    end: int

    @property
    def missing_bytes(self) -> int:
        return max(0, self.end - self.start)

    def to_json(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "generation": self.generation,
            "start": self.start,
            "end": self.end,
            "missing_bytes": self.missing_bytes,
        }


@dataclass
class Generation:
    """One ``audit.log`` file's worth of windows, in offset order."""

    filename: str
    generation: str
    index: int
    refs: list[ObjectRef] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    start: int = 0
    end: int = 0

    @property
    def bytes(self) -> int:
        """The interval the generation spans — **not** the bytes it actually holds."""
        return max(0, self.end - self.start)

    @property
    def held_bytes(self) -> int:
        """Bytes of the stream this generation can actually reproduce."""
        return max(0, self.bytes - self.missing_bytes)

    @property
    def missing_bytes(self) -> int:
        return sum(gap.missing_bytes for gap in self.gaps)

    @property
    def complete(self) -> bool:
        """True when coverage begins at offset 0 — nothing was skipped at the head."""
        return self.start == 0

    @property
    def first_upload(self) -> datetime | None:
        stamps = [ref.last_modified for ref in self.refs if ref.last_modified is not None]
        return min(stamps) if stamps else None

    @property
    def last_upload(self) -> datetime | None:
        stamps = [ref.last_modified for ref in self.refs if ref.last_modified is not None]
        return max(stamps) if stamps else None

    def to_json(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "generation": self.generation,
            "start": self.start,
            "end": self.end,
            "bytes": self.bytes,
            "windows": len(self.refs),
            "held_bytes": self.held_bytes,
            "missing_bytes": self.missing_bytes,
            "complete": self.complete,
            "first_upload_utc": self.first_upload.isoformat() if self.first_upload else None,
            "last_upload_utc": self.last_upload.isoformat() if self.last_upload else None,
            "keys": [ref.key for ref in self.refs],
        }


@dataclass
class StreamPlan:
    """Which windows to write and in what order, once generations are known."""

    generations: list[Generation] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    windows: list[ObjectRef] = field(default_factory=list)
    superseded: int = 0
    unknown_generation_windows: int = 0
    # Set when two generations share an upload time, or the destination reported no
    # times at all. See plan_stream for why that makes the stream order unknowable.
    order_uncertain: bool = False


@dataclass
class VerifyReport:
    """Everything :func:`verify_backup` learned about one destination."""

    destination: dict[str, Any]
    listed: bool = True
    list_error: str = ""
    hashed: bool = True
    objects: int = 0
    bytes: int = 0
    unrecognized: list[str] = field(default_factory=list)
    corrupt: list[ObjectCheck] = field(default_factory=list)
    unreadable: list[ObjectCheck] = field(default_factory=list)
    verified_objects: int = 0
    verified_bytes: int = 0
    plan: StreamPlan = field(default_factory=StreamPlan)
    # Rotated objects are tracked as refs *and* as checks. The refs are what a
    # restore iterates over, so they must be populated even when the objects are
    # not downloaded (``--no-hash``), or the snapshots — the only copy of any range
    # the live watermark never reached — would be silently left out.
    rotated_refs: list[ObjectRef] = field(default_factory=list)
    rotated: list[ObjectCheck] = field(default_factory=list)
    checks: dict[str, ObjectCheck] = field(default_factory=dict)
    total_lines: int = 0
    malformed_lines: int = 0

    @property
    def generation_runs(self) -> Sequence[Generation]:
        return self.plan.generations

    @property
    def gaps(self) -> Sequence[Gap]:
        return self.plan.gaps

    @property
    def missing_bytes(self) -> int:
        return sum(gap.missing_bytes for gap in self.plan.gaps)

    @property
    def ok(self) -> bool:
        """True when the backup is readable, intact, and has no holes."""
        return (
            self.listed
            and not self.list_error
            and not self.corrupt
            and not self.unreadable
            and not self.plan.gaps
        )

    @property
    def needs_attention(self) -> bool:
        """``ok`` plus "the stream can be rebuilt, but possibly out of order".

        Kept separate from :attr:`ok` so both facts stay visible: the objects are
        intact and complete, *and* their chronological order could not be
        established. Callers deciding whether to trust the output want the second
        answer too, and it is the one that has no other symptom.
        """
        return (not self.ok) or self.plan.order_uncertain

    def to_json(self) -> dict[str, Any]:
        return {
            "destination": self.destination,
            "ok": self.ok,
            "listed": self.listed,
            "list_error": self.list_error or None,
            "hashed": self.hashed,
            "objects": self.objects,
            "bytes": self.bytes,
            "unrecognized": self.unrecognized,
            "verified_objects": self.verified_objects,
            "verified_bytes": self.verified_bytes,
            "corrupt": [
                {
                    "key": check.key,
                    "expected_sha12": check.expected_sha12,
                    "actual_sha12": check.actual_sha12,
                }
                for check in self.corrupt
            ],
            "unreadable": [{"key": c.key, "error": c.error} for c in self.unreadable],
            "rotated_objects": len(self.rotated_refs),
            "generations": [gen.to_json() for gen in self.plan.generations],
            "gaps": [gap.to_json() for gap in self.plan.gaps],
            "missing_bytes": self.missing_bytes,
            "unknown_generation_windows": self.plan.unknown_generation_windows,
            "superseded_windows": self.plan.superseded,
            "order_uncertain": self.plan.order_uncertain,
            "total_lines": self.total_lines,
            "malformed_lines": self.malformed_lines,
        }


@dataclass
class RestoreReport:
    """Outcome of writing a backup back to disk."""

    target: str
    stream_path: str = ""
    manifest_path: str = ""
    rotated_paths: list[str] = field(default_factory=list)
    wrote_bytes: int = 0
    wrote_lines: int = 0
    included: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    verify: VerifyReport | None = None
    error: str = ""
    # Set when the run stopped on purpose rather than failing — a stale target
    # directory, say. A caller (the CLI) needs to tell "you asked for something
    # unsafe" from "the backup is broken", and matching on the error *message* to
    # do it would break the first time the wording changed.
    refused: bool = False

    @property
    def ok(self) -> bool:
        return not self.error

    def to_json(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "stream_path": self.stream_path or None,
            "manifest_path": self.manifest_path or None,
            "rotated_paths": self.rotated_paths,
            "wrote_bytes": self.wrote_bytes,
            "wrote_lines": self.wrote_lines,
            "objects_included": len(self.included),
            "objects_skipped": self.skipped,
            "refused": self.refused,
            "error": self.error or None,
            "verify": self.verify.to_json() if self.verify else None,
        }


# -------------------------------------------------------------------- analysis


def _epoch(moment: datetime | None) -> float:
    return moment.timestamp() if moment is not None else 0.0


def _count_lines(data: bytes) -> tuple[int, int]:
    """Return ``(lines, malformed)`` for a chunk of the audit JSON-lines stream."""
    lines = 0
    malformed = 0
    for raw in data.split(b"\n"):
        if not raw.strip():
            continue
        lines += 1
        try:
            json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            malformed += 1
    return lines, malformed


def plan_stream(refs: Sequence[ObjectRef]) -> StreamPlan:
    """Group live windows into generations, find holes, and order a restore.

    Within a generation, windows are ordered by byte offset and must tile
    ``[0, end)``. A window whose range is already covered is *superseded* — that is
    the crash-retry case, where the same start was re-uploaded after more lines
    arrived — so the longer window wins and no gap is reported. A window starting
    beyond the current coverage is a real hole: the writer advanced while the
    backup was not shipping.
    """
    plan = StreamPlan()
    ordered = sorted(
        (ref for ref in refs if ref.kind == "live"),
        key=lambda ref: (_epoch(ref.last_modified), ref.start, ref.end),
    )
    current: Generation | None = None

    for ref in ordered:
        if ref.generation == UNKNOWN_GENERATION:
            plan.unknown_generation_windows += 1
        if (
            current is None
            or current.filename != ref.filename
            or current.generation != ref.generation
        ):
            current = Generation(
                filename=ref.filename,
                generation=ref.generation,
                index=len(plan.generations),
                refs=[ref],
                start=ref.start,
                end=ref.end,
            )
            plan.generations.append(current)
            continue

        if ref.start > current.end:
            # A hole: the writer advanced while the backup was not shipping.
            gap = Gap(
                filename=ref.filename,
                generation=ref.generation,
                start=current.end,
                end=ref.start,
            )
            plan.gaps.append(gap)
            current.gaps.append(gap)
            current.end = ref.end
        elif ref.start == current.end:
            # A clean continuation: this window picks up exactly where the last
            # one ended. Not a supersede — nothing is covered twice.
            current.end = ref.end
        else:
            # Overlapping: the same start re-uploaded after more lines arrived (a
            # crash-retry), or a re-upload of a range already held. The later
            # upload is authoritative for the bytes it covers.
            plan.superseded += 1
            current.end = max(current.end, ref.end)
        current.refs.append(ref)

    # Windows to write are derived after the walk, not during it: a window cannot
    # be known to be superseded until a later upload has been seen. Sorting by
    # (start, longest first) makes the greedy pass pick the largest copy of each
    # range, so a retried range is written once instead of duplicated.
    for generation in plan.generations:
        generation.refs.sort(key=lambda ref: (ref.start, ref.end))
        covered = 0
        for ref in sorted(generation.refs, key=lambda ref: (ref.start, -ref.end)):
            if ref.end <= covered:
                continue
            plan.windows.append(ref)
            covered = max(covered, ref.end)

    # Ordering *within* a generation is by byte offset, which is exact. Ordering
    # *between* generations has no such anchor: every generation is a fresh file
    # numbered from 0, so the only cross-generation signal is when each was
    # uploaded. If two generations share that timestamp — or the destination
    # reported none — their relative order is genuinely unknowable, and the walk's
    # tie-break (longest first) is arbitrary. Flagged rather than silently
    # producing a mis-ordered forensic record.
    stamps = [_epoch(generation.first_upload) for generation in plan.generations]
    if len(stamps) != len(set(stamps)):
        plan.order_uncertain = True
    return plan


# -------------------------------------------------------------------- fetching


def _fetch(destination: BackupDestination, key: str) -> ObjectCheck:
    """Download one object and hash it against the value in its own key."""
    parsed = parse_object_key(key)
    check = ObjectCheck(
        key=key,
        ok=False,
        expected_sha12=parsed.sha12 if parsed else "",
    )
    try:
        data = destination.get_object(key)
    except (BackupError, OSError) as exc:
        check.error = f"{type(exc).__name__}: {exc}"
        return check
    check.data = data
    check.size = len(data)
    check.actual_sha12 = _sha12(data)
    check.ok = check.actual_sha12 == check.expected_sha12
    if not check.ok:
        check.error = (
            f"content hash {check.actual_sha12} does not match the key's {check.expected_sha12}"
        )
    return check


def _load_objects(
    destination: BackupDestination, *, hash_objects: bool, check_lines: bool
) -> VerifyReport:
    """List and (optionally) download everything, returning a bare report."""
    report = VerifyReport(destination=destination.describe(), hashed=hash_objects)
    try:
        listed = destination.list_objects("")
    except (BackupError, OSError) as exc:
        report.listed = False
        report.list_error = f"{type(exc).__name__}: {exc}"
        return report

    report.objects = len(listed)
    report.bytes = sum(obj.size for obj in listed)

    refs: list[ObjectRef] = []
    for obj in listed:
        ref = parse_object_key(obj.key)
        if ref is None:
            report.unrecognized.append(obj.key)
            continue
        ref = replace(ref, last_modified=obj.last_modified)
        if not hash_objects:
            report.verified_objects += 1
            report.verified_bytes += ref.size
            refs.append(ref)
            if ref.kind == "rotated":
                report.rotated_refs.append(ref)
            continue
        check = _fetch(destination, obj.key)
        report.checks[obj.key] = check
        if check.ok:
            report.verified_objects += 1
            report.verified_bytes += check.size
        elif check.size:
            report.corrupt.append(check)
        else:
            report.unreadable.append(check)
        if check_lines and check.data:
            check.lines, check.malformed_lines = _count_lines(check.data)
            report.total_lines += check.lines
            report.malformed_lines += check.malformed_lines
        # A corrupt object is still walked: its offsets are real information, and
        # dropping it would invent a gap that may not exist. The verdict comes from
        # `corrupt`, not from the coverage walk.
        refs.append(ref)
        if ref.kind == "rotated":
            report.rotated_refs.append(ref)
            report.rotated.append(check)

    report.plan = plan_stream(refs)
    return report


def verify_backup(
    destination: BackupDestination | None = None,
    *,
    hash_objects: bool = True,
    check_lines: bool = True,
) -> VerifyReport:
    """List, download, hash, and reassemble a backup. Never raises."""
    try:
        dest = destination if destination is not None else build_destination()
    except BackupError as exc:
        return VerifyReport(
            destination=destination_summary(), listed=False, list_error=str(exc)
        )

    if isinstance(dest, NullDestination):
        return VerifyReport(
            destination=dest.describe(),
            listed=False,
            list_error=(
                "no backup destination is configured (set "
                "VA_LSE_AUDIT_BACKUP_DESTINATION, or pass a destination override)"
            ),
        )
    return _load_objects(dest, hash_objects=hash_objects, check_lines=check_lines)


# ------------------------------------------------------------------- restoring


def restore_backup(
    destination: BackupDestination | None = None,
    *,
    target: str | Path,
    hash_objects: bool = True,
    verify: bool = True,
    force: bool = False,
) -> RestoreReport:
    """Write a backup back to ``target`` as a readable stream plus a manifest.

    Produces:

    * ``restored.jsonl`` — the live stream, reassembled in upload order across
      generations. This is the chronological record.
    * ``rotated/<name>-<size>-<sha12>.jsonl`` — every distinct rotated-file
      snapshot. These overlap the stream by design (a file is shipped live *and*
      again in full once it rotates), and they are the only copy of any range the
      live watermark never reached, so they are written out rather than folded in.
    * ``manifest.json`` — the verification report, so the operator has a dated,
      machine-readable statement of what the restoration does and does not contain.

    Restoring never deletes or modifies anything in the destination, and it refuses to
    overwrite an existing ``restored.jsonl`` unless ``force`` is set: a second run
    silently replacing a previous restoration (and its manifest) is the kind of thing
    that quietly destroys the record an operator was about to hand to an auditor.
    """
    try:
        dest = destination if destination is not None else build_destination()
    except BackupError as exc:
        return RestoreReport(target=str(target), error=str(exc))

    if isinstance(dest, NullDestination):
        # Nothing configured: say so rather than failing on a NotImplementedError
        # from the no-op destination's list_objects.
        return RestoreReport(
            target=str(target),
            error=(
                "no backup destination is configured (set "
                "VA_LSE_AUDIT_BACKUP_DESTINATION, or pass a destination override)"
            ),
        )

    report = (
        _load_objects(dest, hash_objects=hash_objects, check_lines=True) if verify else None
    )
    if report is not None and not report.listed:
        return RestoreReport(target=str(target), error=report.list_error, verify=report)

    root = Path(target).expanduser()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return RestoreReport(target=str(root), error=f"could not create {root}: {exc}")

    if not force and (root / "restored.jsonl").exists():
        return RestoreReport(
            target=str(root),
            refused=True,
            error=(
                f"{root / 'restored.jsonl'} already exists; pass --force to replace it, "
                "or restore into an empty directory"
            ),
            verify=report,
        )

    out = RestoreReport(
        target=str(root), manifest_path=str(root / "manifest.json"), verify=report
    )

    if report is not None:
        windows = report.plan.windows
        rotated_keys = [ref.key for ref in report.rotated_refs]
    else:
        windows, rotated_keys = _list_for_restore(dest)

    stream = root / "restored.jsonl"
    try:
        with stream.open("wb") as handle:
            for window in windows:
                data = _read_for_restore(dest, window.key, out)
                if data is None:
                    continue
                handle.write(data)
                out.wrote_bytes += len(data)
                out.wrote_lines += data.count(b"\n")
                out.included.append(window.key)
    except OSError as exc:
        out.error = f"could not write {stream}: {exc}"
        return out
    out.stream_path = str(stream)

    rotated_dir = root / "rotated"
    for key in rotated_keys:
        ref = parse_object_key(key)
        data = _read_for_restore(dest, key, out)
        if data is None or ref is None:
            continue
        # Named by content so two snapshots of one filename (before and after it
        # rotated) cannot overwrite each other in the restored output.
        path = rotated_dir / f"{ref.filename}-{ref.size}-{ref.sha12}.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        except OSError as exc:
            out.skipped.append({"key": key, "reason": str(exc)})
            continue
        out.rotated_paths.append(str(path))
        out.included.append(key)

    try:
        (root / "manifest.json").write_text(
            json.dumps(out.to_json(), indent=2, sort_keys=True), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("could not write restore manifest: %s", exc)
        out.manifest_path = ""
    return out


def _read_for_restore(
    destination: BackupDestination, key: str, out: RestoreReport
) -> bytes | None:
    try:
        return destination.get_object(key)
    except (BackupError, OSError) as exc:
        out.skipped.append({"key": key, "reason": str(exc)})
        return None


def _list_for_restore(
    destination: BackupDestination,
) -> tuple[list[ObjectRef], list[str]]:
    """Planless path used by ``verify=False``: order by upload time alone."""
    refs: list[ObjectRef] = []
    rotated: list[str] = []
    try:
        listed = destination.list_objects("")
    except (BackupError, OSError) as exc:
        logger.warning("could not list the destination: %s", exc)
        return [], []
    for obj in listed:
        ref = parse_object_key(obj.key)
        if ref is None:
            continue
        ref = replace(ref, last_modified=obj.last_modified)
        if ref.kind == "rotated":
            rotated.append(ref.key)
        else:
            refs.append(ref)
    plan = plan_stream(refs)
    return plan.windows, rotated


# ------------------------------------------------------------------- rendering


def render_verify_text(report: VerifyReport) -> str:
    """Human-readable verification summary for the CLI and for logs."""
    dest = report.destination
    lines = [f"destination: {dest.get('backend', 'unknown')} (off_pod={dest.get('off_pod')})"]
    if report.list_error:
        lines.append(f"FAILED to list the destination: {report.list_error}")
        return "\n".join(lines)
    lines.append(f"objects: {report.objects:,} ({report.bytes:,} bytes)")
    if report.unrecognized:
        lines.append(f"unrecognized objects (not written by this tool): {len(report.unrecognized)}")
        for key in report.unrecognized[:5]:
            lines.append(f"  - {key}")
    if report.hashed:
        lines.append(f"hash-verified: {report.verified_objects:,} of {report.objects:,}")
        if report.corrupt:
            lines.append(f"CORRUPT ({len(report.corrupt)}):")
            for check in report.corrupt:
                lines.append(f"  - {check.key}: {check.error}")
        if report.unreadable:
            lines.append(f"UNREADABLE ({len(report.unreadable)}):")
            for check in report.unreadable:
                lines.append(f"  - {check.key}: {check.error}")
    else:
        lines.append("hash verification: skipped (--no-hash)")

    lines.append(f"file generations: {len(report.plan.generations)}")
    for generation in report.plan.generations:
        flags = []
        if not generation.complete:
            flags.append("starts mid-file: head is missing")
        if generation.gaps:
            flags.append(f"{generation.missing_bytes:,} bytes missing inside")
        suffix = f"  [{'; '.join(flags)}]" if flags else ""
        lines.append(
            f"  - {generation.filename} gen={generation.generation} "
            f"span {generation.start:,}-{generation.end:,} "
            f"({len(generation.refs)} window(s), holding {generation.held_bytes:,} bytes){suffix}"
        )
    if report.plan.superseded:
        lines.append(
            f"superseded windows (a longer re-upload of the same range): {report.plan.superseded}"
        )
    if report.plan.unknown_generation_windows:
        lines.append(
            f"windows with no generation tag: {report.plan.unknown_generation_windows} "
            "(written by an older version, or the platform exposes no file identity)"
        )

    if report.plan.gaps:
        lines.append(
            f"GAPS ({len(report.plan.gaps)}) — {report.missing_bytes:,} bytes never uploaded:"
        )
        for gap in report.plan.gaps:
            lines.append(
                f"  - {gap.filename} gen={gap.generation} offsets "
                f"{gap.start:,}-{gap.end:,} ({gap.missing_bytes:,} bytes)"
            )
    else:
        lines.append("gaps: none — every uploaded range tiles the stream with no holes")

    if report.plan.gaps and report.rotated_refs:
        lines.append(
            f"note: {len(report.rotated_refs)} rotated snapshot(s) are present and may cover bytes "
            "the live watermark never reached; check the manifest before concluding a range "
            "is unrecoverable"
        )
    if any(not gen.complete for gen in report.plan.generations):
        lines.append(
            "note: a generation that starts above offset 0 has a missing head — the bytes "
            "before its first window were written while no backup was running"
        )
    if report.plan.order_uncertain:
        lines.append(
            "WARNING: file generations could not be put in chronological order — two or "
            "more share an upload time, or the destination reported none. Ordering within "
            "each generation is exact (byte offsets), but the order *between* them is not: "
            "every generation starts at offset 0, so only upload time distinguishes them. "
            "The restored stream may interleave files incorrectly; sort by the timestamp "
            "field inside each record if it matters."
        )
    lines.append(f"lines: {report.total_lines:,} (malformed: {report.malformed_lines:,})")
    lines.append(f"VERDICT: {'ok' if not report.needs_attention else 'attention needed'}")
    return "\n".join(lines)


def render_restore_text(report: RestoreReport) -> str:
    """Human-readable restore summary."""
    lines: list[str] = []
    if report.error:
        lines.append(f"restore FAILED: {report.error}")
    lines.append(f"target: {report.target}")
    if report.stream_path:
        lines.append(
            f"stream: {report.stream_path} ({report.wrote_bytes:,} bytes, "
            f"{report.wrote_lines:,} lines)"
        )
    lines.append(f"objects written: {len(report.included)}")
    if report.rotated_paths:
        lines.append(f"rotated snapshots: {len(report.rotated_paths)}")
    if report.skipped:
        lines.append(f"objects skipped ({len(report.skipped)}):")
        for entry in report.skipped[:5]:
            lines.append(f"  - {entry['key']}: {entry['reason']}")
    if report.manifest_path:
        lines.append(f"manifest: {report.manifest_path}")
    if report.verify is not None and report.verify.plan.gaps:
        lines.append(
            f"WARNING: the restored stream has {len(report.verify.plan.gaps)} gap(s) "
            f"totalling {report.verify.missing_bytes:,} bytes — see the manifest"
        )
    if report.verify is not None and report.verify.plan.order_uncertain:
        lines.append(
            "WARNING: file generations could not be ordered chronologically, so the order "
            "of restored.jsonl across file boundaries is not trustworthy — see the manifest"
        )
    return "\n".join(lines)


def config_overrides_from_args(
    *,
    destination: str = "",
    bucket: str = "",
    prefix: str = "",
    path: str = "",
    container: str = "",
    endpoint_url: str = "",
) -> Mapping[str, str]:
    """Translate CLI flags into ``config`` attribute overrides for the destination.

    Lets recovery read from a different bucket, prefix, or endpoint than the
    deployment writes to — the common case when the writer's credentials are
    scope-limited to writing.
    """
    overrides: dict[str, str] = {}
    if destination:
        overrides["AUDIT_BACKUP_DESTINATION"] = destination
    if bucket:
        overrides["AUDIT_BACKUP_S3_BUCKET"] = bucket
        overrides["AUDIT_BACKUP_GCS_BUCKET"] = bucket
    if prefix:
        overrides["AUDIT_BACKUP_S3_PREFIX"] = prefix
        overrides["AUDIT_BACKUP_GCS_PREFIX"] = prefix
        overrides["AUDIT_BACKUP_AZURE_PREFIX"] = prefix
    if path:
        overrides["AUDIT_BACKUP_DIR"] = path
    if container:
        overrides["AUDIT_BACKUP_AZURE_CONTAINER"] = container
    if endpoint_url:
        overrides["AUDIT_BACKUP_S3_ENDPOINT_URL"] = endpoint_url
    return overrides


def audit_integrity_health() -> dict[str, Any]:
    """Config-only summary for ``/health`` — no listing, no network.

    Restoring is an on-demand operator action, but the operator still needs to see
    whether the configuration they would restore *with* is coherent, and from the
    same endpoint they already poll.
    """
    summary = destination_summary()
    payload: dict[str, Any] = {
        "restore_available": summary.get("backend") not in ("", "none", None),
        "destination": summary.get("backend"),
        "off_pod": summary.get("off_pod"),
        "tool": "scripts/restore_audit_logs.py",
        "local_retention_days": config.AUDIT_RETENTION_DAYS,
    }
    if summary.get("warning"):
        payload["warning"] = summary["warning"]
    return payload
