"""Batched full draft over large record sets, against the configured endpoint.

Why batched: one 1,814-page run digests ~329 chunks; measured 2026-09-20
(req_279c83c8a6c9) the digest sustains ~3.3 chunks/min, so a single run needs
~100 min and dies at the pipeline timeout (``VA_LSE_PIPELINE_TIMEOUT_SECONDS``,
default 1800s). Batching splits the DIGEST into runs that each fit the timeout,
then drafts once over the combined evidence — the same shape a single
in-process run would have produced.

Structure (uses the app's own pipeline code, not a reimplementation):
    per batch: review_medical_records()  -> MedicalDigest
               (chunk + parallel digest + dedupe + hierarchical merge),
               wrapped in run_with_timeout, state saved resumably per batch
    combined:  _dedupe_facts + _merge_facts over the union (cross-batch dupes)
               _summarize -> record summary for the draft prompt
    once:      grounding -> draft -> review, the same prompts, budgets and
               normalization as app/draft.py

Failure policy: ``review_medical_records`` raises when ANY chunk fails after
its retry round, discarding the whole round's work. For a batched offline run
that is the wrong granularity — one dense chunk whose digest response overflows
the output cap (measured: 21,339 chars, JSON truncated mid-string) must not
quarantine its 12 siblings. So on failure the group bisects in half and each
half retries; a file that fails ALONE is quarantined (excluded, recorded in
state) and the rest of its group survives. The review pass is treated the way
``run_draft`` treats it: a failure there keeps the finished draft and records
the miss instead of discarding hours of work.

State is resumable: every completed batch is persisted as its own gzipped
shard under ``<out>/state/`` (plus a human-readable ``index.json`` summary),
each written atomically — a crash costs at most the shard being written,
never the whole run. Re-runs read the shards back and skip completed
batches; a legacy monolithic ``state.json`` from an older run still loads.
The final phase's hierarchical merge persists its own checkpoint
(``state/merge_ckpt.json.gz``, written after every completed merge round and
keyed to a fingerprint of the merge's input facts): a relaunch resumes the
consolidation from the last completed round instead of redoing an hour of
rounds. Delete the ``state/`` directory (or pass ``--fresh``) to start over.

Run:
    .venv/bin/python scripts/batch_draft.py --records "path/to/records" \\
        --condition PTSD --claim-type "Initial claim - service connection" \\
        --witness-json witness.json --out outputs/batch-run

The witness file is JSON: ``{"witness": {name, relationship, known_since,
contact_frequency, veteran_name, witnessed_event, ...credential fields},
"condition": "...", "claim_type": "...", "observations": "..."}``. Anything a
field omits falls back to the same placeholder the Draft tab uses.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import json
import os
import shutil
import string
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from app.medical_review import MedicalDigest

# Optional newer-app slot (A&A intake); absent on earlier app versions. The
# script runs against whatever app version is checked out, so this is a
# capability probe, not a hard dependency.
_CARE_BLOCK: Callable[[dict[str, str]], str] | None
try:
    from app.aa_intake import care_observation_block as _CARE_BLOCK
except ImportError:
    _CARE_BLOCK = None


def _care_block(witness: dict[str, str]) -> str:
    """The A&A intake block when the installed app provides it, else empty."""
    return _CARE_BLOCK(witness) if _CARE_BLOCK else ""

DEFAULT_GLOB = "Part*.pdf"
DEFAULT_PARTS_PER_BATCH = 13  # 91 parts -> 7 batches (13 each, ~260 pages)
# Keep both under the pipeline timeout (VA_LSE_PIPELINE_TIMEOUT_SECONDS, 1800s
# default): a batch that outlives the app's own guard would be killed the same
# way the unbatched run was.
DEFAULT_BATCH_TIMEOUT_S = 1740
DEFAULT_FINAL_TIMEOUT_S = 1500  # merge + summarize + grounding + draft + review


@dataclass
class BatchConfig:
    """Everything one invocation needs; also what the tests construct."""

    records_dir: Path
    out_dir: Path
    condition: str
    claim_type: str
    witness: dict[str, str]
    observations: str
    part_glob: str = DEFAULT_GLOB
    parts_per_batch: int = DEFAULT_PARTS_PER_BATCH
    batch_timeout_s: int = DEFAULT_BATCH_TIMEOUT_S
    final_timeout_s: int = DEFAULT_FINAL_TIMEOUT_S
    run_final: bool = True

    @property
    def state_path(self) -> Path:
        return self.out_dir / "state.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _format_with(template: str, **kwargs: str) -> str:
    """Format *template* with only the placeholders it actually declares.

    This script mirrors app/draft.py's prompt construction, and those templates
gain slots over time (the A&A ``care_block`` addition landed mid-development
while a run was in flight). Passing an undeclared kwarg raises KeyError;
supplying only declared ones keeps the script working across app versions —
and any field the template declares that we cannot supply is logged instead
of silently left as a literal ``{name}`` in the prompt.
    """
    fields = {name for _, name, _, _ in string.Formatter().parse(template) if name}
    missing = fields - kwargs.keys()
    if missing:
        log(f"WARNING: prompt template has unsupplied fields: {', '.join(sorted(missing))}")
    return template.format(**{k: v for k, v in kwargs.items() if k in fields})


def _facts_text_kwargs(digest_obj: Any) -> dict[str, Any]:
    """Call relevant_facts_text with the ordering the installed app supports.

    Newer app versions accept ``sort_dates=True`` (chronological presentation
    for drafting); older ones reject the kwarg. Detection, not version pinning.
    """
    params = inspect.signature(type(digest_obj).relevant_facts_text).parameters
    kwargs = {"max_facts": 150}
    if "sort_dates" in params:
        kwargs["sort_dates"] = True
    return kwargs


STATE_SCHEMA = 2
SHARD_DIR_NAME = "state"
FINAL_SHARD_NAME = "final.json.gz"


def _fsync_path(path: Path) -> None:
    """Flush one file's contents to disk before it is renamed into place."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_bytes(destination: Path, payload: bytes) -> None:
    """Write *payload* so *destination* is never a partial file.

    The write goes to a sibling temp file, is fsynced, and is renamed over the
    destination — ``os.replace`` is atomic on POSIX and Windows. A crash mid-
    write therefore leaves the previous good file intact, which is the whole
    point: this state is hours of LLM spend, and a truncated rewrite used to
    destroy every completed batch at once.
    """
    tmp = destination.with_name(f".{destination.name}.tmp")
    tmp.write_bytes(payload)
    _fsync_path(tmp)
    os.replace(tmp, destination)


def _shard_payload(key: str, batch: dict) -> bytes:
    body = json.dumps(
        {"schema": STATE_SCHEMA, "key": key, "batch": batch},
        separators=(",", ":"),
    ).encode("utf-8")
    return gzip.compress(body, mtime=0)


def _read_shard(path: Path) -> tuple[str, dict] | None:
    """Decode one shard; a corrupt shard reads as absent, never fatal."""
    try:
        data = json.loads(gzip.decompress(path.read_bytes()))
        if isinstance(data, dict) and isinstance(data.get("batch"), dict):
            return str(data.get("key") or path.stem), data["batch"]
    except (OSError, ValueError):
        return None
    return None


def save_state(cfg: BatchConfig, state: dict) -> None:
    """Persist run state as one gzip shard per batch plus ``index.json``.

    The shards (``<out>/state/batch_*.json.gz``, ``<out>/state/final.json.gz``)
    are the authoritative copy: each is written atomically, so a crash can
    cost at most the shard being written, never the whole run. ``index.json``
    is a human-readable summary, not a source of truth — ``load_state`` never
    reads it. A pre-sharding ``state.json`` is left untouched; ``load_state``
    still reads it so older runs resume, and the next save re-shards them.
    """
    shards = cfg.out_dir / SHARD_DIR_NAME
    shards.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {"schema": STATE_SCHEMA, "batches": {}, "final": False}
    for key, batch in state.get("batches", {}).items():
        if not isinstance(batch, dict) or not ("facts" in batch or "error" in batch):
            continue
        _atomic_write_bytes(shards / f"{key}.json.gz", _shard_payload(key, batch))
        index["batches"][key] = {
            "facts": len(batch.get("facts", [])) if "facts" in batch else None,
            "error": batch.get("error"),
        }
    final = state.get("final")
    if final is not None:
        _atomic_write_bytes(
            shards / FINAL_SHARD_NAME,
            gzip.compress(
                json.dumps({"schema": STATE_SCHEMA, "final": final}, separators=(",", ":")).encode("utf-8"),
                mtime=0,
            ),
        )
        index["final"] = True
    index["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _atomic_write_bytes(
        shards / "index.json",
        json.dumps(index, indent=1, sort_keys=True).encode("utf-8"),
    )


def load_state(cfg: BatchConfig, fresh: bool = False) -> dict:
    """Read run state back; shards win, legacy ``state.json`` is the fallback.

    A corrupt or truncated shard is skipped — the resume then redigests that
    one batch instead of failing the run. A legacy monolithic ``state.json``
    (written by the pre-shard format, including runs interrupted before this
    change) loads as before when no shards exist.
    """
    if fresh:
        return {"batches": {}, "final": None}
    shards = cfg.out_dir / SHARD_DIR_NAME
    batches: dict[str, dict] = {}
    final: dict | None = None
    if shards.is_dir():
        for path in sorted(shards.glob("batch_*.json.gz")):
            decoded = _read_shard(path)
            if decoded is not None:
                batches[decoded[0]] = decoded[1]
        final_path = shards / FINAL_SHARD_NAME
        if final_path.exists():
            try:
                data = json.loads(gzip.decompress(final_path.read_bytes()))
                if isinstance(data, dict) and data.get("final") is not None:
                    final = data["final"]
            except (OSError, ValueError):
                final = None
    if batches or final is not None:
        return {"batches": batches, "final": final}
    if cfg.state_path.exists():
        try:
            loaded: dict = json.loads(cfg.state_path.read_text(encoding="utf-8"))
            return loaded
        except (OSError, ValueError):
            # A truncated legacy file must read as "start clean", not crash the
            # resume: the operator loses the run either way, but a crash also
            # hides the message that would have said so.
            return {"batches": {}, "final": None}
    return {"batches": {}, "final": None}


MERGE_CKPT_NAME = "merge_ckpt.json.gz"
_MERGE_CKPT_SCHEMA = 1


def _merge_ckpt_path(cfg: BatchConfig) -> Path:
    return cfg.out_dir / SHARD_DIR_NAME / MERGE_CKPT_NAME


def _merge_fingerprint(facts: list[Any]) -> str:
    """A stable digest of the merge's *input* facts.

    The merge-round checkpoint is only valid for the exact fact list that
    produced it: the resume must not continue a consolidation whose input was
    different (digests re-run, flags change), so the checkpoint stores this
    fingerprint and a loader refuses any mismatch. Content-derived, not
    count-derived — a same-length list with different facts must not match.
    """
    h = hashlib.sha256()
    for f in facts:
        h.update(f"{getattr(f, 'date', '')}\x1f{getattr(f, 'type', '')}"
                 f"\x1f{getattr(f, 'description', '')}\x1e".encode("utf-8", "replace"))
    return h.hexdigest()[:16]


def load_merge_checkpoint(cfg: BatchConfig, fingerprint: str) -> list[Any] | None:
    """The stored post-round fact list for *fingerprint*, or None.

    A missing, corrupt, stale-schema, or fingerprint-mismatched checkpoint all
    read as None — the merge then simply starts from round 1, the same start a
    run without this feature would take. Facts are rebuilt field-wise exactly
    like ``digest_from_state`` so a schema-grown MedicalFact stays loadable.
    """
    try:
        data = json.loads(gzip.decompress(_merge_ckpt_path(cfg).read_bytes()))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("schema") != _MERGE_CKPT_SCHEMA:
        return None
    if data.get("fingerprint") != fingerprint:
        return None
    raw = data.get("facts")
    if not isinstance(raw, list):
        return None
    from app.medical_review import MedicalFact

    fields = MedicalFact.__dataclass_fields__
    return [MedicalFact(**{k: f[k] for k in fields if k in f})
            for f in raw if isinstance(f, dict)]


def save_merge_checkpoint(
    cfg: BatchConfig, fingerprint: str, round_no: int, facts: list[Any]
) -> None:
    """Atomically persist one completed merge round's output.

    Same durability contract as the batch shards (temp file, fsync, atomic
    rename): a crash mid-write leaves the previous round's checkpoint intact,
    so a resume loses at most the interrupted round — never a half-written
    fact list.
    """
    body = json.dumps(
        {
            "schema": _MERGE_CKPT_SCHEMA,
            "fingerprint": fingerprint,
            "round": round_no,
            "facts": [vars(f) for f in facts],
        },
        separators=(",", ":"),
    ).encode("utf-8")
    shards = cfg.out_dir / SHARD_DIR_NAME
    shards.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(shards / MERGE_CKPT_NAME, gzip.compress(body, mtime=0))


def digest_to_state(digest: Any) -> dict:
    return {
        "facts": [vars(f) for f in digest.facts],
        "conditions": digest.conditions,
        "providers": digest.providers,
        "pages_reviewed": digest.pages_reviewed,
        "chunks_reviewed": digest.chunks_reviewed,
        "duplicates_skipped": digest.duplicates_skipped,
        "unreadable_pages": digest.unreadable_pages,
        "pages_in_files": digest.pages_in_files,
        "coverage_ratio": round(digest.coverage_ratio, 4),
        "chunks_without_facts": digest.chunks_without_facts,
    }


def digest_from_state(data: dict) -> "MedicalDigest":
    from app.medical_review import MedicalDigest, MedicalFact

    fields = MedicalFact.__dataclass_fields__
    return MedicalDigest(
        facts=[MedicalFact(**{k: f[k] for k in fields if k in f}) for f in data["facts"]],
        conditions=list(data.get("conditions", [])),
        providers=list(data.get("providers", [])),
        pages_reviewed=data.get("pages_reviewed", 0),
        chunks_reviewed=data.get("chunks_reviewed", 0),
        duplicates_skipped=data.get("duplicates_skipped", 0),
        unreadable_pages=data.get("unreadable_pages", 0),
        pages_in_files=data.get("pages_in_files", 0),
        chunks_without_facts=data.get("chunks_without_facts", 0),
    )


def progress_cb(label: str) -> Callable[[float, str], None]:
    def cb(frac: float, msg: str) -> None:
        log(f"  [{label}] {frac:5.0%} {msg[:110]}")
    return cb


def plan_batches(part_files: list[Path], per_batch: int) -> list[list[Path]]:
    """Group the parts into digest batches of at most *per_batch* files."""
    if per_batch < 1:
        raise ValueError(f"parts per batch must be >= 1, got {per_batch}")
    return [part_files[i:i + per_batch] for i in range(0, len(part_files), per_batch)]


def merge_states(a: dict, b: dict) -> dict:
    """Union two digest-state dicts (facts re-merge in the final phase)."""
    if "facts" not in a:
        return b
    if "facts" not in b:
        return a
    m = dict(a)
    m["facts"] = a["facts"] + b["facts"]
    for k in ("conditions", "providers"):
        m[k] = sorted(set(a.get(k, [])) | set(b.get(k, [])))
    for k in ("pages_reviewed", "chunks_reviewed", "duplicates_skipped",
              "unreadable_pages", "pages_in_files", "chunks_without_facts"):
        m[k] = a.get(k, 0) + b.get(k, 0)
    m["coverage_ratio"] = round(
        (1 - m["unreadable_pages"] / m["pages_in_files"]) if m["pages_in_files"] else 1.0, 4)
    m["duration_s"] = round(a.get("duration_s", 0) + b.get("duration_s", 0), 1)
    m["files"] = a.get("files", 0) + b.get("files", 0)
    m["pages"] = a.get("pages", 0) + b.get("pages", 0)
    m["quarantined"] = sorted(set(a.get("quarantined", [])) | set(b.get("quarantined", [])))
    return m


EMPTY_DIGEST_STATE = {
    "facts": [], "conditions": [], "providers": [], "pages_reviewed": 0,
    "chunks_reviewed": 0, "duplicates_skipped": 0, "unreadable_pages": 0,
    "pages_in_files": 0, "coverage_ratio": 1.0, "chunks_without_facts": 0,
}


def _fresh_empty_state() -> dict:
    """A per-call copy of EMPTY_DIGEST_STATE with fresh list values.

    ``dict(EMPTY_DIGEST_STATE)`` shallow-copies: every empty/quarantined batch
    would alias the constant's ``facts``/``conditions``/``providers`` lists, and
    one in-place ``.append`` would cross-contaminate all of them *and the module
    constant* the tests pin. Keep the constant (tests reference
    ``batch_draft.EMPTY_DIGEST_STATE``); copy through this helper.
    """
    return {k: (list(v) if isinstance(v, list) else v)
            for k, v in EMPTY_DIGEST_STATE.items()}


def digest_group(llm: Any, cfg: BatchConfig, group_label: str,
                 files: list[Path], depth: int = 0) -> tuple[dict, list[str]]:
    """Digest a group of staged part files, bisecting on an isolated failure.

    Returns the merged digest-state dict and the list of file names that were
    quarantined. A file that fails alone is excluded and recorded; its group
    survives. See the module docstring for why this overrides the all-or-
    nothing granularity of ``review_medical_records``.
    """
    from app.documents import records_from_local_path
    from app.llm import LLMAuthError
    from app.medical_review import review_medical_records
    from app.pipeline_guard import run_with_timeout

    # Guard against empty file list to prevent infinite recursion in bisection.
    # This should never happen in practice (batch_files() never produces empty
    # batches), but defensive programming prevents theoretical edge cases.
    if not files:
        log(f"{group_label}: EMPTY — no files to process")
        return _fresh_empty_state(), []

    sdir = cfg.out_dir / "staging" / group_label
    sdir.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst = sdir / src.name
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
    docs, _skipped = records_from_local_path(str(sdir))
    pages = sum(d.source_page_count for d in docs)
    t0 = time.time()
    try:
        digest = run_with_timeout(
            review_medical_records, llm, docs,
            timeout_seconds=cfg.batch_timeout_s, progress=progress_cb(group_label),
        )
    except LLMAuthError as exc:
        # A credential refusal is not a property of any file: every member of
        # the group — and every other batch — would be refused identically.
        # Bisecting cannot help (the 2026-09-22 dead-key incident bisected five
        # levels deep and quarantined nine healthy files), so the only correct
        # move is to stop the run, keep every completed shard (save_state is
        # the caller's job; nothing here has dirtied state), and tell the
        # operator the one thing that fixes it. SystemExit(2): distinct from
        # the argument/validation code, and BaseException so the per-batch
        # "one bad batch must not kill the run" handler lets it through.
        log(f"FATAL: {exc}")
        log("Run stopped before further spend. Fix the credential, then re-run "
            "the same command — completed batches resume from state.")
        raise SystemExit(2) from exc
    except Exception as exc:  # noqa: BLE001 — bisect or quarantine, per policy above
        msg = f"{type(exc).__name__}: {str(exc)[:200]}"
        if len(files) == 1 or depth >= 6:
            log(f"{group_label}: QUARANTINED {len(files)} file(s) "
                f"[{', '.join(f.name for f in files)}] — {msg}")
            state = _fresh_empty_state()
            state.update({"quarantined": [f.name for f in files], "error": msg,
                          "files": len(files), "pages": pages, "duration_s": 0.0})
            return state, [f.name for f in files]
        mid = len(files) // 2
        log(f"{group_label}: FAILED ({type(exc).__name__}) — bisecting {len(files)} files "
            f"into {mid}+{len(files) - mid}")
        a, ea = digest_group(llm, cfg, f"{group_label}a", files[:mid], depth + 1)
        b, eb = digest_group(llm, cfg, f"{group_label}b", files[mid:], depth + 1)
        merged = merge_states(a, b)
        merged["quarantined"] = sorted(set(merged.get("quarantined", [])) | set(ea) | set(eb))
        return merged, ea + eb
    dur = time.time() - t0
    state = digest_to_state(digest)
    state.update({"duration_s": round(dur, 1), "files": len(files), "pages": pages,
                  "quarantined": []})
    log(f"{group_label}: OK in {dur:.0f}s — facts={len(digest.facts)} "
        f"chunks={digest.chunks_reviewed} pages={pages} cov={digest.coverage_ratio:.0%}")
    return state, []


def _retry_phase(fn: Callable[[], Any], label: str, *, attempts: int = 8,
                 base_wait_s: float = 30.0) -> Any:
    """Run one final-phase LLM call across provider degradation bursts.

    The per-call retry ladder spans seconds; the 2026-09-23 degradation came in
    minutes-long bursts (status: incomplete / empty responses, then 429s) that
    outlast it, and the single calls after the merge have no keep-raw-facts
    fallback — one exhausted ladder discarded the whole final phase's work.
    This wraps them with a patient outer loop: wait out each burst, retry the
    whole call. The breaker-wait gate runs first so an OPEN breaker (fed by the
    burst) is waited out before the attempt, not burned as one.
    """
    from app.llm import LLMAuthError

    for attempt in range(1, attempts + 1):
        _wait_for_breaker(f"the {label} call")
        try:
            return fn()
        except LLMAuthError:
            raise  # a credential verdict never heals
        except Exception as exc:  # noqa: BLE001 — degraded bursts are the case this exists for
            if attempt == attempts:
                raise
            wait = min(base_wait_s * attempt, 180.0)
            log(f"{label} call failed (attempt {attempt}/{attempts}) — "
                f"{type(exc).__name__}: {exc}; waiting {wait:.0f}s before retrying")
            time.sleep(wait)
    raise AssertionError("unreachable")


def _wait_for_breaker(source: str) -> None:
    """Block until the LLM circuit breaker will admit a call.

    The merge's keep-raw-facts fallback absorbs a breaker OPEN during a
    provider degradation (that is what saved the 2026-09-23 run), but the
    single calls that follow it — summarize, grounding, draft — have no
    fallback, and starting one with the breaker still OPEN killed the whole
    final phase instantly, twice. The breaker's own recovery clock is the
    right wait: it re-probes the provider at the configured cadence, so this
    returns as soon as a call would be admitted.
    """
    from app.circuit_breaker import get_llm_breaker

    breaker = get_llm_breaker()
    poll_s = 5.0
    polls = 0
    while breaker.state == "OPEN" and not breaker.allow_request():
        if polls == 0:
            log(f"breaker OPEN after {source} — waiting out its recovery window")
        time.sleep(poll_s)
        polls += 1
        if polls % 12 == 0:
            log(f"still waiting on breaker recovery ({polls * poll_s:.0f}s so far)")
    if polls:
        log(f"breaker recovered after {polls * poll_s:.0f}s — resuming {source}")


def final_phase(llm: Any, cfg: BatchConfig, batch_states: dict[str, dict]) -> dict:
    """Merge cross-batch, summarize, then grounding -> draft -> review.

    Mirrors app/draft.py's prompt construction exactly (same placeholders,
    budgets, grounding normalization, credential scope block) over the
    combined digest, and treats a failed review pass as non-fatal the way
    ``run_draft`` does.
    """
    from app.config import load_knowledge
    from app.documents import DRAFT_INTERNAL_MAX_CHARS, MAX_OBSERVATIONS_CHARS
    from app.draft import (
        DRAFT_SYSTEM_TEMPLATE,
        DRAFT_USER,
        GROUNDING_SYSTEM,
        GROUNDING_USER,
        REVIEW_MAX_CHARS,
        REVIEW_SYSTEM,
        REVIEW_USER,
        DraftResult,
        _grounding_for_prompt,
        _normalize_grounding,
        _review_rejection_reason,
        _truncate_for_prompt,
        grounding_markdown,
        witness_credentials_block,
        witness_scope_directive,
    )
    from app.medical_review import (
        MedicalDigest,
        _dedupe_facts,
        _merge_facts,
        _summarize,
    )
    from app.pipeline_guard import run_with_timeout
    from app.prompt_sanitize import GUARD_NOTE, sanitize_digest_text, sanitize_for_prompt

    def work() -> dict:
        parts = [digest_from_state(s) for s in batch_states.values()]
        all_facts = [f for d in parts for f in d.facts]
        log(f"combined: {len(all_facts)} facts from {len(parts)} batches -> dedupe")
        # Memory pressure warning: at the 6,000+ fact design scale (5,000-page
        # bundles) the dedupe and merge phases create additional copies of the
        # fact list, and peak memory during final_phase climbs toward 20-30 MB
        # by ~10,000 facts. The threshold matches the design scale so the
        # warning actually fires on production-sized runs — at the old 10,000
        # mark it never could. Log-only: counts, no PHI.
        if len(all_facts) > 6_000:
            log(f"WARNING: High fact count ({len(all_facts):,}) — final phase memory "
                f"usage may spike. Consider reducing batch size or increasing worker "
                f"memory limits if the system is under pressure.")
        deduped = _dedupe_facts(all_facts)
        log(f"combined: {len(deduped)} facts after mechanical dedupe -> hierarchical merge")
        fingerprint = _merge_fingerprint(deduped)
        combined = MedicalDigest(
            facts=deduped,
            conditions=sorted({c for d in parts for c in d.conditions}),
            providers=sorted({p for d in parts for p in d.providers}),
            pages_reviewed=sum(d.pages_reviewed for d in parts),
            chunks_reviewed=sum(d.chunks_reviewed for d in parts),
            duplicates_skipped=sum(d.duplicates_skipped for d in parts),
            unreadable_pages=sum(d.unreadable_pages for d in parts),
            pages_in_files=sum(d.pages_in_files for d in parts),
            chunks_without_facts=sum(d.chunks_without_facts for d in parts),
        )
        merged = _merge_facts(
            llm,
            combined,
            progress=progress_cb("merge"),
            # Merge-round checkpointing: every completed round is persisted
            # (keyed to this exact input's fingerprint), and a relaunch resumes
            # the consolidation instead of redoing an hour of rounds. The
            # hooks never raise into the merge (both sides guard); a broken
            # or stale checkpoint degrades to a from-scratch merge.
            on_round=lambda round_no, facts: save_merge_checkpoint(
                cfg, fingerprint, round_no, facts),
            resume=lambda: load_merge_checkpoint(cfg, fingerprint),
        )
        combined.facts = merged
        log(f"combined: {len(merged)} facts after merge -> summarize")
        combined.summary = _retry_phase(lambda: _summarize(llm, combined), "summarize")
        log("combined: summary done")

        obs_for_prompt, removed = _truncate_for_prompt(cfg.observations, DRAFT_INTERNAL_MAX_CHARS)
        if removed or len(cfg.observations) > MAX_OBSERVATIONS_CHARS:
            log(f"WARNING: observations are {len(cfg.observations):,} chars — "
                f"{removed:,} truncated for prompts; details at the end may be missed")
        grounding_query = f"{cfg.condition} {obs_for_prompt}"

        log("grounding call…")
        care_block = _care_block(cfg.witness)
        # Normalization runs INSIDE the retry: chat_json guarantees parseable
        # JSON, not the object-of-lists shape grounding needs, and a
        # shape-invalid answer (2026-09-23 23:21: a bare array) died outside
        # the wrapper one second after a 4-minute call, discarding the merge.
        grounding = _retry_phase(
            lambda: _normalize_grounding(llm.chat_json(
                GROUNDING_SYSTEM,
                _format_with(
                    GROUNDING_USER,
                    condition=sanitize_for_prompt(cfg.condition, max_chars=500),
                    claim_type=sanitize_for_prompt(cfg.claim_type, max_chars=500),
                    relationship=sanitize_for_prompt(
                        cfg.witness.get("relationship", "not specified"), max_chars=500),
                    credentials_block=witness_credentials_block(cfg.witness),
                    care_block=care_block,
                    observations=sanitize_for_prompt(obs_for_prompt, max_chars=DRAFT_INTERNAL_MAX_CHARS),
                    digest=sanitize_digest_text(
                        combined.relevant_facts_text(
                            grounding_query, **_facts_text_kwargs(combined)),
                        max_chars=120_000,
                    ),
                    checklist=load_knowledge("topic_checklist.md"),
                    guard_note=GUARD_NOTE,
                ),
                phase="grounding",
            )),
            "grounding",
        )
        log(f"grounding done: {len(grounding)} keys")

        # Self-contained scope block, exactly as run_draft builds it: heading
        # included only when a credential rule exists.
        scope_block = witness_scope_directive(cfg.witness)
        if scope_block:
            scope_block = f"CREDENTIAL SCOPE:\n{scope_block}"

        log("draft call…")
        draft = _retry_phase(lambda: llm.chat(
            _format_with(
                DRAFT_SYSTEM_TEMPLATE,
                guide=load_knowledge("drafting_guide.md"),
                checklist=load_knowledge("topic_checklist.md"),
                credential_scope=scope_block,
            ),
            _format_with(
                DRAFT_USER,
                witness_name=sanitize_for_prompt(
                    cfg.witness.get("name", "[Witness Name]"), max_chars=500),
                relationship=sanitize_for_prompt(
                    cfg.witness.get("relationship", "[relationship]"), max_chars=500),
                known_since=sanitize_for_prompt(
                    cfg.witness.get("known_since", "[how long known]"), max_chars=500),
                contact_frequency=sanitize_for_prompt(
                    cfg.witness.get("contact_frequency", "[frequency of contact]"), max_chars=500),
                veteran_name=sanitize_for_prompt(
                    cfg.witness.get("veteran_name", "[Veteran Name]"), max_chars=500),
                condition=sanitize_for_prompt(cfg.condition, max_chars=500),
                claim_type=sanitize_for_prompt(cfg.claim_type, max_chars=500),
                witnessed_event=sanitize_for_prompt(
                    cfg.witness.get("witnessed_event", "unknown"), max_chars=500),
                credentials_block=witness_credentials_block(cfg.witness),
                care_block=care_block,
                observations=sanitize_for_prompt(obs_for_prompt, max_chars=DRAFT_INTERNAL_MAX_CHARS),
                grounding=_grounding_for_prompt(grounding),
                digest_summary=sanitize_digest_text(combined.summary or "(no summary)", max_chars=20_000),
                guard_note=GUARD_NOTE,
            ),
            max_tokens=6000,
            phase="draft",
        ), "draft")
        log(f"draft done: {len(draft):,} chars")

        issues: list[str] = []
        final = draft.strip()
        # Escaping can expand text; sanitize without truncation, then check the
        # actual review input against its budget — as run_draft does.
        review_draft = sanitize_for_prompt(draft, max_chars=2 * len(draft) + 1)
        if len(review_draft) > REVIEW_MAX_CHARS:
            issues.append(
                f"Self-review was skipped because the full statement exceeds the "
                f"{REVIEW_MAX_CHARS:,}-character review limit. The complete original "
                "draft is preserved; review it manually before signing."
            )
            log(f"review skipped: draft exceeds {REVIEW_MAX_CHARS:,} chars")
        else:
            log("review call…")
            try:
                # The review is polish, not substance: fewer attempts than the
                # mandatory calls, because its designed degradation (ship the
                # unreviewed draft with a note) beats burning 20 minutes of
                # waits for a second opinion.
                review = _retry_phase(lambda: llm.chat_json(
                    REVIEW_SYSTEM,
                    _format_with(
                        REVIEW_USER,
                        draft=review_draft,
                        guide=load_knowledge("drafting_guide.md")[:6000],
                        checklist=load_knowledge("topic_checklist.md")[:6000],
                        credential_scope=scope_block,
                        care_block=care_block,
                        guard_note=GUARD_NOTE,
                    ),
                    phase="review",
                ), "review", attempts=2, base_wait_s=10.0)
            except Exception as exc:  # noqa: BLE001 — keep the finished draft
                review = None
                issues.append(
                    "Self-review pass was skipped (the model call failed) — the statement "
                    "below is the unreviewed draft. Re-run to get the polished version."
                )
                log(f"review call FAILED ({type(exc).__name__}) — keeping unreviewed draft")
            if isinstance(review, dict):
                raw = review.get("issues_found", [])
                issues = [i for i in raw if isinstance(i, str)] if isinstance(raw, list) else []
                improved = review.get("improved_statement", "")
                rejection = _review_rejection_reason(draft, improved)
                if rejection:
                    issues.append(f"Self-review not applied: {rejection}.")
                else:
                    final = improved.strip()
            log(f"review done: {len(issues)} issue(s)")

        return {
            "statement": final,
            "grounding_markdown": grounding_markdown(
                DraftResult(grounding=grounding)
            ) if isinstance(grounding, dict) else "",
            "grounding_raw": grounding if isinstance(grounding, dict) else {},
            "review_issues": issues,
            "facts_total": len(combined.facts),
            "facts_pre_merge": len(deduped),
            "summary": combined.summary,
        }

    return run_with_timeout(work, timeout_seconds=cfg.final_timeout_s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="batch_draft.py",
        description="Digest a large record set in timeout-sized batches, then draft "
                    "one lay statement over the combined evidence (resumable).",
    )
    p.add_argument("--records", type=Path, required=True,
                   help="directory containing the record part files")
    p.add_argument("--glob", default=DEFAULT_GLOB, metavar="PAT",
                   help=f"glob for part files inside --records (default: {DEFAULT_GLOB})")
    p.add_argument("--out", type=Path, default=Path("outputs/batch-draft"), metavar="DIR",
                   help="output directory for state.json, statement.md, grounding.md")
    p.add_argument("--condition", help="claimed condition, e.g. PTSD "
                   "(or supply it via --witness-json)")
    p.add_argument("--claim-type", dest="claim_type",
                   help='e.g. "Initial claim - service connection" '
                        '(or supply it via --witness-json)')
    p.add_argument("--witness-json", type=Path, metavar="FILE",
                   help="JSON with {witness: {...}, condition, claim_type, observations}; "
                        "per-flag values override it")
    p.add_argument("--observations", type=Path, metavar="FILE",
                   help="file containing the witness's observations text")
    p.add_argument("--parts-per-batch", type=int, default=DEFAULT_PARTS_PER_BATCH,
                   metavar="N", help=f"default: {DEFAULT_PARTS_PER_BATCH}")
    p.add_argument("--batch-timeout", type=int, default=DEFAULT_BATCH_TIMEOUT_S, metavar="S",
                   help="digest timeout per batch (default: %(default)s)")
    p.add_argument("--final-timeout", type=int, default=DEFAULT_FINAL_TIMEOUT_S, metavar="S",
                   help="timeout for merge+summarize+grounding+draft+review (default: %(default)s)")
    p.add_argument("--no-final", action="store_true",
                   help="digest only; skip the grounding/draft/review final phase")
    p.add_argument("--fresh", action="store_true",
                   help="ignore any existing state.json and start over")
    return p


def config_from_args(args: argparse.Namespace) -> BatchConfig:
    """Assemble the run configuration, with the witness JSON supplying defaults."""
    witness: dict[str, str] = {}
    condition = ""
    claim_type = ""
    observations = ""
    if args.witness_json:
        data = json.loads(args.witness_json.read_text(encoding="utf-8"))
        witness = {str(k): str(v) for k, v in (data.get("witness") or {}).items()}
        condition = str(data.get("condition") or "")
        claim_type = str(data.get("claim_type") or "")
        observations = str(data.get("observations") or "")
    if args.observations:
        observations = args.observations.read_text(encoding="utf-8").strip()
    return BatchConfig(
        records_dir=args.records,
        out_dir=args.out,
        condition=args.condition or condition,
        claim_type=args.claim_type or claim_type,
        witness=witness,
        observations=observations,
        part_glob=args.glob,
        parts_per_batch=args.parts_per_batch,
        batch_timeout_s=args.batch_timeout,
        final_timeout_s=args.final_timeout,
        run_final=not args.no_final,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)

    if not cfg.records_dir.is_dir():
        log(f"FATAL: records directory not found: {cfg.records_dir}")
        return 1
    if not cfg.condition or not cfg.claim_type:
        log("FATAL: --condition and --claim-type are required "
            "(directly or via --witness-json)")
        return 2
    if not cfg.observations:
        log("FATAL: observations are required (--observations FILE, "
            "or 'observations' in --witness-json)")
        return 2

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(cfg, fresh=args.fresh)
    if args.fresh:
        # --fresh means start over: a merge checkpoint from the discarded run
        # must not survive to short-circuit the new run's consolidation (its
        # fingerprint would not match anyway, but deleting it keeps the state
        # directory honest about what it holds).
        try:
            _merge_ckpt_path(cfg).unlink()
        except OSError:
            pass

    part_files = sorted(cfg.records_dir.glob(cfg.part_glob))
    if not part_files:
        log(f"FATAL: no parts match {cfg.records_dir / cfg.part_glob}")
        return 1
    batches = plan_batches(part_files, cfg.parts_per_batch)
    log(f"{len(part_files)} parts -> {len(batches)} batches of ≤{cfg.parts_per_batch}")

    from app.config import load_settings
    from app.llm import LLMClient, probe_chat
    from app.preflight import REFUSAL_STATUSES

    settings = load_settings()
    log(f"endpoint={settings.base_url} main={settings.model_main} fast={settings.model_fast}")

    # Startup credential gate (one real call, ~600 ms when refused — a 401
    # spends nothing): the 2026-09-22 incident spent 17 minutes discovering a
    # dead key one chunk at a time. REFUSAL_STATUSES (401/403/404) here are
    # the same deterministic refusals the UI preflight blocks on; anything
    # ambiguous (no response, 429, 5xx) lets the run proceed and leaves the
    # verdict to the run's own retry machinery — a gate that refuses a working
    # run is worse than the failure it prevents. Deliberately before the
    # client is constructed: there is no reason to build one on a dead key.
    probe = probe_chat(settings.base_url, settings.api_key, settings.model_fast)
    if probe.status in REFUSAL_STATUSES:
        log(f"FATAL: the endpoint refused the credential check (HTTP {probe.status})"
            f"{': ' + probe.error if probe.error else ''}")
        log("Nothing was run and nothing was spent. Regenerate the API key / check the "
            "credit balance, then re-run this command.")
        return 1
    llm = LLMClient(settings)

    t_all = time.time()
    for idx, chunk in enumerate(batches, start=1):
        key = f"batch_{idx:02d}"
        if key in state["batches"] and "facts" in state["batches"][key]:
            done = state["batches"][key]
            log(f"{key}: already done ({len(done['facts'])} facts, "
                f"{len(done.get('quarantined', []))} quarantined) — skip")
            continue
        try:
            batch_state, excluded = digest_group(llm, cfg, key, chunk)
            batch_state["excluded_files"] = excluded
            state["batches"][key] = batch_state
        except Exception as exc:  # noqa: BLE001 — one bad batch must not kill the run
            state["batches"][key] = {"error": f"{type(exc).__name__}: {exc}"}
            log(f"{key}: FAILED — {type(exc).__name__}: {exc}")
            traceback.print_exc()
        save_state(cfg, state)

    ok = {k: v for k, v in state["batches"].items() if "facts" in v}
    failed = {k: v for k, v in state["batches"].items() if "error" in v}
    log(f"digest complete: {len(ok)} batches ok, {len(failed)} failed "
        f"({sum(len(v['facts']) for v in ok.values())} facts total)")
    quarantined = sorted(q for v in ok.values() for q in v.get("quarantined", []))
    if quarantined:
        log(f"quarantined files ({len(quarantined)}): {', '.join(quarantined)}")

    if not ok:
        log("FATAL: no batch succeeded; cannot draft.")
        return 1
    if not cfg.run_final:
        log("digest-only run (--no-final): statement not drafted.")
        return 0

    final = state.get("final")
    if final is not None and "error" in final:
        # A stored error from a previous attempt is a resume point, not a
        # verdict: the digest shards behind it represent hours of paid API
        # work, and the failure that leaves one here (breaker opened mid-merge
        # during a provider degradation) is transient. Re-run the final phase
        # on every resume instead of dying with the run unrecoverable.
        log(f"final phase previously failed — retrying: {final['error']}")
        state["final"] = None

    if state.get("final") is None:
        try:
            state["final"] = final_phase(llm, cfg, ok)
            save_state(cfg, state)
        except Exception as exc:  # noqa: BLE001
            state["final"] = {"error": f"{type(exc).__name__}: {exc}"}
            save_state(cfg, state)
            log(f"final phase FAILED — {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return 1

    final = state["final"]
    if "error" in final:
        log(f"final phase reported an error: {final['error']}")
        return 1

    (cfg.out_dir / "statement.md").write_text(
        f"# Draft lay statement — {cfg.condition}\n\n{final['statement']}\n",
        encoding="utf-8",
    )
    (cfg.out_dir / "grounding.md").write_text(final["grounding_markdown"], encoding="utf-8")
    log(f"ALL DONE in {(time.time() - t_all) / 60:.1f} min — statement "
        f"{len(final['statement']):,} chars, {final['facts_total']} facts "
        f"(pre-merge {final['facts_pre_merge']}), review issues {len(final['review_issues'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
