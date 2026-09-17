"""Exhaustive medical-record review built for very large record sets (1 to ~5,000 pages).

Pipeline:
  documents -> overlapping chunks -> duplicate-chunk skip -> PARALLEL LLM fact
  extraction (with retry) -> mechanical fact dedup -> hierarchical LLM merge ->
  capped, ordered digest -> full-coverage narrative summary.

Both pathways (evaluate and draft) rely on this module to build a structured,
citable digest of every uploaded medical document.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import contextvars
import logging
import time

from . import config, tracing
from .documents import Chunk, ExtractedDocument, chunk_page_labelled_text, paragraph_index
from .llm import LLMClient, LLMError
from .logging_config import PhaseTimer, get_request_id
from .profiler import get_current_run_profiler, worker_timer
from .prompt_sanitize import GUARD_NOTE, sanitize_for_prompt

logger = logging.getLogger("app.medical_review")

ProgressCallback = Callable[[float, str], None]

DIGEST_SYSTEM = """You are a meticulous medical-records analyst supporting the review of \
VA disability claims. Your job is to extract EVERY medically or factually relevant detail \
from a chunk of medical records. You must be exhaustive and precise; downstream legal \
analysis depends on you capturing dates, names, facilities, diagnoses, symptoms, treatments, \
medications, test results, and provider statements verbatim-faithfully. Never invent facts. \
If the chunk is administrative or contains no medical content, say so with an empty fact list.

Because this digest feeds lay/witness-statement work, give special attention to details a \
witness could corroborate or expand on: observed symptoms and behaviors (pain, limping, \
cognitive changes, mood changes), functional limitations (work, driving, household tasks, \
self-care), witnessed incidents or injuries, medication changes and side effects, before/after \
progression markers, hospitalizations, and provider statements about prognosis or the need \
for assistance. Capture these as facts even when phrased informally."""

DIGEST_USER_TEMPLATE = """Extract all medically and factually relevant information from this \
medical-record chunk ({label}).

Return JSON with this exact shape:
{{
  "facts": [
    {{
      "date": "YYYY-MM or YYYY-MM-DD or approximate (e.g., 'circa 2019', 'unknown')",
      "type": "diagnosis | symptom | treatment | medication | test_result | hospitalization | provider_visit | in_service_event | functional_limitation | observable_behavior | administrative | other",
      "description": "concise factual description",
      "source": "{source_hint}",
      "quote": "short verbatim quote from the chunk supporting this fact"
    }}
  ],
  "conditions_mentioned": ["condition 1", "condition 2"],
  "providers_and_facilities": ["name (role/facility)"],
  "notes": "anything unusual about this chunk (illegible, incomplete, contradictory)"
}}

Rules:
- Capture every distinct fact; do NOT summarize multiple events into one unless identical.
- Preserve exact dates, dosages, pain scores, and proper nouns.
- 'quote' must be a real excerpt from the chunk, <= 40 words.
- Prefer many precise facts over few broad ones; when in doubt about relevance, include the fact.

CHUNK TEXT:
<<<
{chunk_text}
>>>

{guard_note}"""

MERGE_SYSTEM = """You are consolidating extracted medical facts from multiple chunks of the \
same record set into one authoritative digest. Deduplicate identical facts, keep every \
distinct fact, resolve trivially different phrasings, and keep all source citations. \
Do not add facts that were not provided. Output JSON only."""


@dataclass
class MedicalFact:
    date: str
    type: str
    description: str
    source: str
    quote: str = ""


@dataclass
class MedicalDigest:
    """Structured result of the exhaustive record review."""

    facts: list[MedicalFact] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)
    summary: str = ""
    pages_reviewed: int = 0
    chunks_reviewed: int = 0
    duplicates_skipped: int = 0

    def as_json_text(self, max_facts: int | None = None) -> str:
        limit = config.MAX_DIGEST_FACTS if max_facts is None else max_facts
        return json.dumps(
            {
                "facts": [vars(f) for f in self.facts[:limit]],
                "conditions_mentioned": self.conditions,
                "providers_and_facilities": self.providers,
            },
            indent=1,
        )

    def timeline_text(self) -> str:
        lines = [
            f"[{f.date}] ({f.type}) {f.description}  — {f.source}" for f in self.facts
        ]
        return "\n".join(lines)

    def condensed_timeline(self, max_entries: int = 400) -> str:
        """Evenly sample the full timeline so summaries cover the whole record set.

        With thousands of facts, taking only the head of the timeline would bias
        summaries toward the earliest documents; striding keeps every era visible.
        """
        facts = self.facts
        if len(facts) <= max_entries:
            return self.timeline_text()
        step = len(facts) / max_entries
        picked = [facts[int(i * step)] for i in range(max_entries)]
        if facts[-1] is not picked[-1]:
            picked[-1] = facts[-1]
        lines = [f"[{f.date}] ({f.type}) {f.description}  — {f.source}" for f in picked]
        return "\n".join(lines)

    def relevant_facts_text(
        self,
        query: str,
        *,
        max_facts: int = 150,
        budget_chars: int = 90_000,
        always_include_types: tuple[str, ...] = ("in_service_event", "hospitalization"),
    ) -> str:
        """Return digest facts ranked by relevance to a claim/observation query.

        Replaces naive head-truncation of the full JSON digest: for large record
        sets the evidence relevant to a given claim can sit anywhere in thousands
        of facts, so each verification/grounding prompt receives the facts that
        actually match it (plus all high-priority event types), within a budget.
        """
        query_tokens = set(_tokens(query))
        if not self.facts:
            return "(no facts extracted from records)"

        # IDF-weighted scoring so distinctive terms (dates, names, numbers)
        # outrank boilerplate shared by every fact — critical when thousands
        # of facts repeat similar wording.
        fact_tokens: list[frozenset[str]] = []
        df: Counter[str] = Counter()
        for fact in self.facts:
            tokens = _tokens(f"{fact.date} {fact.description} {fact.quote}")
            fact_tokens.append(tokens)
            for token in tokens:
                df[token] += 1
        total = len(self.facts)

        def weight(token: str) -> float:
            return math.log((total + 1) / (df.get(token, 0) + 1)) + 1.0

        query_weight = sum(weight(t) for t in query_tokens) or 1.0

        scored: list[tuple[float, int, MedicalFact]] = []
        for index, fact in enumerate(self.facts):
            tokens = fact_tokens[index]
            overlap = sum(weight(t) for t in query_tokens & tokens) / query_weight
            if fact.type in always_include_types:
                overlap = max(overlap, 0.05)  # keep anchor events visible
            scored.append((overlap, index, fact))

        # All matching facts first (best score first), then best non-matches to fill.
        matches = [item for item in scored if item[0] >= 0.15]
        matches.sort(key=lambda item: (-item[0], item[1]))
        fillers = [item for item in scored if item[0] < 0.15]
        fillers.sort(key=lambda item: (-item[0], item[1]))

        lines: list[str] = []
        used = 0
        count = 0
        for _, _, fact in matches + fillers:
            if count >= max_facts:
                break
            line = f"[{fact.date}] ({fact.type}) {fact.description} — {fact.source}"
            if fact.quote:
                line += f" | quote: \"{fact.quote}\""
            if used + len(line) > budget_chars:
                break
            lines.append(line)
            used += len(line) + 1
            count += 1
        header = (
            f"({len(lines)} of {len(self.facts)} extracted facts, selected for relevance "
            f"to the items below; the full digest was reviewed during record analysis)"
        )
        return header + "\n" + "\n".join(lines)


def _fact_from_raw(raw: dict, fallback_source: str) -> MedicalFact | None:
    description = str(raw.get("description", "") or "").strip()
    if not description:
        return None
    return MedicalFact(
        date=str(raw.get("date", "unknown") or "unknown").strip(),
        type=str(raw.get("type", "other") or "other").strip().lower(),
        description=description,
        source=str(raw.get("source", "") or fallback_source).strip() or fallback_source,
        quote=str(raw.get("quote", "") or "").strip(),
    )


def _norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _dedupe_facts(facts: list[MedicalFact]) -> list[MedicalFact]:
    """Mechanically drop repeated facts (same date + description), keeping order."""
    seen: set[str] = set()
    unique: list[MedicalFact] = []
    for fact in facts:
        key = f"{_norm_key(fact.date)}|{_norm_key(fact.description)}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(fact)
    return unique


def review_medical_records(
    llm: LLMClient,
    documents: list[ExtractedDocument],
    progress: ProgressCallback | None = None,
) -> MedicalDigest:
    """Run the full exhaustive review over all uploaded records.

    Scales to thousands of pages: chunks are digested in parallel, duplicate
    chunks are skipped, failed chunks are retried, and large fact lists are
    merged hierarchically instead of in one oversized call.
    """
    if not documents:
        raise ValueError("No medical records provided.")

    pages = sum(len(doc.pages) for doc in documents)
    if pages > config.MAX_RECORD_PAGES:
        raise ValueError(
            f"Record set is {pages:,} pages, over the configured limit of "
            f"{config.MAX_RECORD_PAGES:,}. Split the records into smaller sets or raise "
            "VA_LSE_MAX_RECORD_PAGES."
        )

    # Drop duplicate pages (across and within files) BEFORE chunking: record
    # bundles frequently repeat the same pages, and re-digesting them wastes
    # hours on very large sets without adding evidence.
    seen_page_hashes: set[str] = set()
    unique_docs: list[ExtractedDocument] = []
    duplicates_skipped = 0
    for doc in documents:
        kept = ExtractedDocument(filename=doc.filename)
        for page in doc.pages:
            page_hash = hashlib.sha1(_norm_key(page.text).encode("utf-8")).hexdigest()
            if page_hash in seen_page_hashes:
                duplicates_skipped += 1
                continue
            seen_page_hashes.add(page_hash)
            kept.pages.append(page)
        if kept.pages:
            unique_docs.append(kept)
    if not unique_docs:
        raise ValueError("Records contain no extractable unique text.")

    full_text = "\n\n".join(doc.page_labelled_text() for doc in unique_docs)
    chunks = chunk_page_labelled_text(full_text)
    total_units = len(chunks)

    if progress:
        dup_note = (
            f" ({duplicates_skipped} duplicate page(s) skipped)" if duplicates_skipped else ""
        )
        progress(
            0.05,
            f"Reviewing {pages:,} pages in {len(chunks)} chunk(s){dup_note} using "
            f"{config.RECORDS_CONCURRENCY} parallel worker(s)…",
        )

    _ctx_request_id = get_request_id()  # capture for worker threads
    _ctx_profiler_run = get_current_run_profiler()
    # Spans do not follow contextvars into a thread pool either, so the digest
    # thread re-attaches this parent context — otherwise every chunk span would
    # start a trace of its own instead of nesting under the run.
    _ctx_span = tracing.current_span_context()

    def digest_chunk(chunk: Chunk) -> dict[str, object]:
        # Propagate the run's correlation id into the worker thread.
        from .logging_config import _request_id_var  # local import to avoid cycle at import time
        from .profiler import _current_run_var  # local import to avoid cycle at import time

        token = _request_id_var.set(_ctx_request_id)
        profiler_token = _current_run_var.set(_ctx_profiler_run)
        t0 = time.perf_counter()
        try:
            # Adopt the run's span for this thread, then (optionally) open a chunk
            # span nested under it. Per-chunk spans are opt-in
            # (VA_LSE_TRACE_CHUNK_SPANS): a 5,000-page bundle is hundreds of
            # chunks, which would swamp every other span in the trace.
            with (
                tracing.use_parent(_ctx_span),
                tracing.phase_span(
                    "records:digest.chunk",
                    enabled=config.TRACING_CHUNK_SPANS,
                    chunk=chunk.index,
                ),
                worker_timer("digest", index=chunk.index),
            ):
                data = llm.chat_json(
                    DIGEST_SYSTEM,
                    DIGEST_USER_TEMPLATE.format(
                        label=sanitize_for_prompt(chunk.label, max_chars=200),
                        source_hint=sanitize_for_prompt(chunk.label, max_chars=200),
                        chunk_text=sanitize_for_prompt(chunk.text, max_chars=1_000_000),
                        guard_note=GUARD_NOTE,
                    ),
                    model=llm._settings.model_fast,
                    max_tokens=8000,
                    phase="records:digest",
                )
            duration_ms = int((time.perf_counter() - t0) * 1000)
            logger.debug(
                "chunk digest ok label=%s facts=%d duration_ms=%d",
                chunk.label,
                len(data.get("facts", []) or []),
                duration_ms,
                extra={
                    "request_id": _ctx_request_id or "-",
                    "phase": "records:digest",
                    "status": "ok",
                    "duration_ms": duration_ms,
                    "chunks": chunk.label,
                },
            )
            return data  # type: ignore[no-any-return]  # LLM JSON is untyped dict
        except Exception as exc:  # noqa: BLE001
            duration_ms = int((time.perf_counter() - t0) * 1000)
            logger.warning(
                "chunk digest error label=%s duration_ms=%d error=%s",
                chunk.label,
                duration_ms,
                f"{type(exc).__name__}: {exc}",
                extra={
                    "request_id": _ctx_request_id or "-",
                    "phase": "records:digest",
                    "status": "error",
                    "duration_ms": duration_ms,
                    "error_class": type(exc).__name__,
                },
            )
            raise
        finally:
            try:
                _request_id_var.reset(token)
                _current_run_var.reset(profiler_token)
            except ValueError:
                pass

    # -------------------------------------------- parallel chunk extraction
    results: dict[int, dict] = {}
    failed: dict[int, str] = {}
    completed = 0
    _review_t0 = time.perf_counter()
    rid = get_request_id() or "-"
    logger.info(
        "records review start pages=%d chunks=%d concurrency=%d duplicates_skipped=%d",
        pages,
        total_units,
        config.RECORDS_CONCURRENCY,
        duplicates_skipped,
        extra={
            "request_id": rid,
            "phase": "records:review",
            "status": "start",
            "pages": pages,
            "chunks": total_units,
        },
    )

    def run_round(pending: list[Chunk]) -> None:
        nonlocal completed
        with ThreadPoolExecutor(max_workers=config.RECORDS_CONCURRENCY) as pool:
            future_map = {pool.submit(digest_chunk, c): c for c in pending}
            for future in as_completed(future_map):
                chunk = future_map[future]
                completed += 1
                try:
                    results[chunk.index] = future.result()
                except Exception as exc:  # noqa: BLE001 - record and retry later
                    failed[chunk.index] = str(exc)
                if progress:
                    progress(
                        0.05 + 0.55 * completed / max(total_units, 1),
                        f"Extracting facts — {completed}/{total_units} chunks done…",
                    )

    with tracing.phase_span(
        "records:digest", chunks=total_units, concurrency=config.RECORDS_CONCURRENCY, pages=pages
    ):
        run_round(chunks)

    # Retry failed chunks once; parallel bursts can hit transient rate limits.
    if failed:
        retry_targets = [c for c in chunks if c.index in failed]
        logger.warning(
            "records digest retry pending=%d failed=%s",
            len(retry_targets),
            sorted(failed.keys()),
            extra={
                "request_id": rid,
                "phase": "records:digest",
                "status": "retry",
                "chunks": len(retry_targets),
            },
        )
        failed.clear()
        if progress:
            progress(0.62, f"Retrying {len(retry_targets)} failed chunk(s)…")
        with tracing.phase_span("records:digest", chunks=len(retry_targets), retry=True):
            run_round(retry_targets)

    if failed:
        labels = ", ".join(f"chunk {i}" for i in sorted(failed))
        first_error = failed[sorted(failed)[0]][:200]
        logger.error(
            "records review failed chunks=%s error=%s",
            sorted(failed.keys()),
            first_error,
            extra={
                "request_id": rid,
                "phase": "records:review",
                "status": "error",
                "chunks": len(failed),
                "error_class": "LLMError",
            },
        )
        raise LLMError(
            f"Record review failed: could not digest {labels} after a retry "
            f"({first_error}). Re-run the review; if it persists, split the record "
            "set into smaller files."
        )

    # -------------------------------------------- collect facts in doc order
    all_facts: list[MedicalFact] = []
    conditions: Counter[str] = Counter()
    providers: Counter[str] = Counter()
    for index in sorted(results):
        data = results[index]
        fallback = f"chunk {index}/{len(chunks)}"
        for raw in data.get("facts", []) or []:
            if not isinstance(raw, dict):
                continue
            fact = _fact_from_raw(raw, fallback)
            if fact:
                all_facts.append(fact)
        for name in data.get("conditions_mentioned", []) or []:
            conditions[str(name).strip()] += 1
        for name in data.get("providers_and_facilities", []) or []:
            providers[str(name).strip()] += 1

    all_facts = _dedupe_facts(all_facts)

    # Memory checkpoint: after chunk extraction + dedup (peak before merge)
    try:
        from .pipeline_guard import memory_checkpoint as _mem_cp

        _mem_cp("records:post_dedup")
    except Exception:  # noqa: BLE001
        pass

    if progress:
        progress(0.65, f"Consolidating {len(all_facts):,} extracted facts…")

    digest = MedicalDigest(
        facts=all_facts,
        conditions=[c for c, _ in conditions.most_common(40) if c],
        providers=[p for p, _ in providers.most_common(40) if p],
        pages_reviewed=pages,
        chunks_reviewed=len(chunks),
        duplicates_skipped=duplicates_skipped,
    )

    with (
        tracing.phase_span("records:merge", facts=len(all_facts)),
        PhaseTimer(logger, "records:merge", request_id=rid, facts=len(all_facts)),
    ):
        digest.facts = _merge_facts(llm, digest, progress)
    # Memory checkpoint: after merge (fact list may have shrunk)
    try:
        _mem_cp("records:post_merge")
    except Exception:  # noqa: BLE001
        pass
    with tracing.phase_span("records:summary"), PhaseTimer(logger, "records:summary", request_id=rid):
        digest.summary = _summarize(llm, digest)
    duration_ms = int((time.perf_counter() - _review_t0) * 1000)
    logger.info(
        "records review done pages=%d chunks=%d facts=%d duplicates_skipped=%d duration_ms=%d",
        pages,
        len(chunks),
        len(digest.facts),
        duplicates_skipped,
        duration_ms,
        extra={
            "request_id": rid,
            "phase": "records:review",
            "status": "ok",
            "duration_ms": duration_ms,
            "pages": pages,
            "chunks": len(chunks),
            "facts": len(digest.facts),
        },
    )
    if progress:
        progress(
            0.8,
            f"Record review complete: {len(digest.facts):,} facts extracted from "
            f"{pages:,} pages ({digest.chunks_reviewed} chunks).",
        )
    return digest


MERGE_BATCH_SIZE = 200
MERGE_SINGLE_LIMIT = 250


def _merge_facts(
    llm: LLMClient,
    digest: MedicalDigest,
    progress: ProgressCallback | None = None,
) -> list[MedicalFact]:
    """Consolidate facts, hierarchically when the list is too large for one call.

    A single merge call cannot hold thousands of facts, so oversized lists are
    merged in batches (in parallel), and the merged results are re-merged until
    the list fits one call or stops shrinking. Mechanical dedup runs between
    rounds so facts resolving to the same date+description collapse.
    """
    facts = _dedupe_facts(digest.facts)
    if len(facts) <= MERGE_SINGLE_LIMIT:
        try:
            return _merge_once(llm, facts) or facts
        except LLMError:
            return facts

    current = facts
    _merge_rid = get_request_id() or "-"
    for round_no in range(1, 4):
        batches = [
            current[i : i + MERGE_BATCH_SIZE]
            for i in range(0, len(current), MERGE_BATCH_SIZE)
        ]
        logger.info(
            "merge round start round=%d facts=%d batches=%d",
            round_no,
            len(current),
            len(batches),
            extra={
                "request_id": _merge_rid,
                "phase": "records:merge",
                "status": "start",
                "facts": len(current),
                "chunks": len(batches),
            },
        )
        if progress:
            progress(
                0.66,
                f"Consolidating facts — merge round {round_no}, "
                f"{len(current):,} facts in {len(batches)} batch(es)…",
            )
        merged_by_batch: dict[int, list[MedicalFact]] = {}
        # Capture correlation id for merge workers as well.
        _merge_ctx = _merge_rid
        def _merge_with_ctx(batch: list[MedicalFact]) -> list[MedicalFact]:
            from .logging_config import _request_id_var as _rid_var

            tok = _rid_var.set(_merge_ctx)
            try:
                return _merge_once(llm, batch)
            finally:
                try:
                    _rid_var.reset(tok)
                except ValueError:
                    pass

        with ThreadPoolExecutor(max_workers=config.RECORDS_CONCURRENCY) as pool:
            future_map = {
                pool.submit(_merge_with_ctx, batch): batch_index
                for batch_index, batch in enumerate(batches)
            }
            for future in as_completed(future_map):
                batch_index = future_map[future]
                try:
                    merged_by_batch[batch_index] = future.result() or batches[batch_index]
                except LLMError as exc:
                    logger.warning(
                        "merge batch failed round=%d batch=%d error=%s",
                        round_no,
                        batch_index,
                        f"{type(exc).__name__}: {exc}",
                        extra={
                            "request_id": _merge_rid,
                            "phase": "records:merge",
                            "status": "error",
                            "error_class": type(exc).__name__,
                        },
                    )
                    merged_by_batch[batch_index] = batches[batch_index]  # keep raw facts

        merged: list[MedicalFact] = []
        for batch_index in sorted(merged_by_batch):
            merged.extend(merged_by_batch[batch_index])
        merged = _dedupe_facts(merged)

        if len(merged) <= MERGE_SINGLE_LIMIT or len(merged) >= len(current):
            current = merged
            break
        current = merged

    return current[: config.MAX_DIGEST_FACTS]


def _merge_once(llm: LLMClient, facts: list[MedicalFact]) -> list[MedicalFact]:
    """One merge call over a batch-sized fact list (fast model)."""
    data = llm.chat_json(
        MERGE_SYSTEM,
        "Deduplicate and consolidate these extracted medical facts. Keep every DISTINCT fact "
        "with its source. Return JSON: {\"facts\": [{\"date\",\"type\",\"description\",\"source\",\"quote\"}]}\n\n"
        + json.dumps([vars(f) for f in facts]),
        model=llm._settings.model_fast,
        max_tokens=8000,
        phase="records:merge",
    )
    merged: list[MedicalFact] = []
    for raw in data.get("facts", []) or []:
        if isinstance(raw, dict):
            fact = _fact_from_raw(raw, "records")
            if fact:
                merged.append(fact)
    return merged


def _summarize(llm: LLMClient, digest: MedicalDigest) -> str:
    return llm.chat(
        "You are a medical-records analyst. Write a concise narrative summary (max 250 words) "
        "of the record set: key diagnoses, treatment history, notable events, and current "
        "status. Plain text only.",
        "Extracted facts (sampled evenly across the full timeline):\n"
        + digest.condensed_timeline(max_entries=400)[:16000],
        phase="records:summary",
    )


# ----------------------------------------------------------- relevance search
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for", "with", "by",
    "from", "is", "was", "were", "are", "been", "he", "she", "his", "her", "him", "i",
    "my", "me", "that", "this", "it", "as", "has", "have", "had", "be", "will", "since",
    "during", "about", "into", "their", "they", "them", "you", "your", "we", "our",
}


@lru_cache(maxsize=10_000)
def _tokens(text: str) -> frozenset[str]:
    """Tokenize text into content words; LRU-cached (bounded at 10k entries)
    because facts and paragraphs are scored repeatedly during verification of
    many claims. Bounded LRU prevents unbounded memory growth on large record
    sets while retaining high hit rates for repeated phrases."""
    return frozenset(
        token
        for token in re.findall(r"[a-z0-9]{3,}", text.lower())
        if token not in _STOPWORDS
    )


# Backwards-compat alias: some tooling/tests may import _TOKEN_CACHE.
# Expose the underlying cache mapping via the LRU wrapper's cache_info.
_TOKEN_CACHE: dict[str, frozenset[str]] = {}  # deprecated; _tokens is now LRU-bounded


# ------------------------------------------------------------------- timeline


@dataclass
class TimelineEvent:
    """A single event on the medical timeline, derived from a MedicalFact.

    Used by the timeline UI to render events with their date, type, description,
    and source citation. Events are sorted by date for chronological display.
    """

    date: str  # ISO date (YYYY-MM-DD) or partial (YYYY-MM) or qualifier (e.g., "unknown")
    date_sortable: str  # Normalized date string for sorting (YYYY-MM-DD or YYYY-MM or "9999-99")
    type: str  # fact type: diagnosis, symptom, treatment, medication, etc.
    description: str  # Short description (<=100 chars for display)
    full_description: str  # Full fact description (for expandable details)
    source: str  # Source citation (e.g., "records p.5" or "chunk 3/50")
    quote: str  # Supporting quote from the record
    index: int  # Position in the original facts list (for stable ordering)


@dataclass
class TimelineGap:
    """A detected gap in the medical record timeline.

    Represents a date range with no recorded medical events, which may indicate
    missing records or periods the user should address in their statement.
    """

    start_date: str  # End of previous event (exclusive)
    end_date: str  # Start of next event (exclusive)
    duration_months: int  # Approximate duration of the gap in months
    note: str  # Suggested note for the user


def build_timeline_events(digest: MedicalDigest) -> list[TimelineEvent]:
    """Convert a MedicalDigest into a sorted list of TimelineEvents.

    Events are sorted chronologically by their date_sortable value. Facts with
    the same date maintain their original order from the digest.
    """
    events: list[TimelineEvent] = []
    for index, fact in enumerate(digest.facts):
        event = TimelineEvent(
            date=fact.date,
            date_sortable=_normalize_date_for_sort(fact.date),
            type=fact.type,
            description=_truncate_description(fact.description, 100),
            full_description=fact.description,
            source=fact.source,
            quote=fact.quote,
            index=index,
        )
        events.append(event)

    # Sort by date_sortable, then by original index for stable ordering
    events.sort(key=lambda e: (e.date_sortable, e.index))
    return events


def _normalize_date_for_sort(date_str: str) -> str:
    """Normalize a date string for sorting.

    Handles various date formats extracted by the LLM:
    - Full dates: "2023-05-15" -> "2023-05-15"
    - Month/year: "2023-05" -> "2023-05"
    - Year only: "2023" -> "2023"
    - Qualifiers: "unknown", "circa 2019" -> "9999" (sort last)
    """
    if not date_str or date_str.lower() in ("unknown", "n/a", "none", ""):
        return "9999-99-99"

    date_lower = date_str.lower().strip()

    # Handle qualifiers like "circa 2019", "approx 2020"
    import re

    circa_match = re.match(r"(circa|approx(?:imately)?)\s*(\d{4})", date_lower)
    if circa_match:
        year = circa_match.group(2)
        return f"{year}-06-15"  # Mid-year for approximate dates

    # Try to extract a 4-digit year
    year_match = re.search(r"\b(\d{4})\b", date_str)
    if not year_match:
        return "9999-99-99"

    year = year_match.group(1)

    # Look for month
    month_match = re.search(r"\b(0?\d|1[0-2])\b", date_str)
    month = month_match.group(1).zfill(2) if month_match else "06"  # Default to June

    # Look for day
    day_match = re.search(r"\b(0?[1-9]|[12]\d|3[01])\b", date_str)
    day = day_match.group(1).zfill(2) if day_match else "15"  # Default to mid-month

    return f"{year}-{month}-{day}"


def _truncate_description(text: str, max_length: int) -> str:
    """Truncate a description to max_length characters, adding ellipsis if needed."""
    text = text.strip()
    if len(text) <= max_length:
        return text
    truncated = text[: max_length - 3].rsplit(" ", 1)[0]
    return f"{truncated}..."


def detect_timeline_gaps(
    events: list[TimelineEvent],
    *,
    min_gap_months: int = 6,
) -> list[TimelineGap]:
    """Detect significant gaps in the timeline where no medical events are recorded.

    A gap is flagged when there are at least `min_gap_months` between consecutive
    events. This helps users identify periods that may need additional documentation
    or explanation in their statement.
    """
    if len(events) < 2:
        return []

    gaps: list[TimelineGap] = []
    for i in range(len(events) - 1):
        current = events[i]
        next_event = events[i + 1]

        # Skip events without sortable dates
        if current.date_sortable == "9999-99-99" or next_event.date_sortable == "9999-99-99":
            continue

        try:
            from datetime import datetime

            current_date = datetime.strptime(current.date_sortable, "%Y-%m-%d")
            next_date = datetime.strptime(next_event.date_sortable, "%Y-%m-%d")
            gap_days = (next_date - current_date).days
            gap_months = gap_days // 30  # Approximate

            if gap_months >= min_gap_months:
                gaps.append(
                    TimelineGap(
                        start_date=current.date,
                        end_date=next_event.date,
                        duration_months=gap_months,
                        note=(f"No recorded medical events for approximately {gap_months} months "
                        f"({current.date} to {next_event.date}). "
                        f"Consider whether treatment continued during this period or if records "
                        f"are missing."
                        ),
                    )
                )
        except ValueError:
            continue

    return gaps


def get_timeline_type_counts(events: list[TimelineEvent]) -> dict[str, int]:
    """Count events by type for the filter UI."""
    counts: dict[str, int] = {}
    for event in events:
        counts[event.type] = counts.get(event.type, 0) + 1
    return counts


def get_timeline_providers(events: list[TimelineEvent]) -> list[str]:
    """Extract unique provider/facility names from event sources.

    Parses provider names from source strings like "VCU Medical Center (provider)" or
    "VA Hospital Richmond (facility)".
    """
    providers: set[str] = set()
    for event in events:
        source = event.source
        # Extract provider name from parenthetical if present
        import re

        match = re.match(r"^(.+?)\s*\((?:provider|facility|role).*?\)$", source)
        if match:
            providers.add(match.group(1).strip())
        else:
            # Use the source as-is if no parenthetical
            if source and source != "records":
                providers.add(source)

    return sorted(providers)


def filter_timeline_events(
    events: list[TimelineEvent],
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    fact_types: set[str] | None = None,
    providers: set[str] | None = None,
) -> list[TimelineEvent]:
    """Filter timeline events by date range, fact type, and/or provider.

    Args:
        events: The full list of timeline events.
        date_from: Filter events on or after this date (YYYY-MM-DD format).
        date_to: Filter events on or before this date (YYYY-MM-DD format).
        fact_types: Set of fact types to include (None = all types).
        providers: Set of provider names to include (None = all providers).

    Returns:
        Filtered list of events.
    """
    if not events:
        return []

    filtered: list[TimelineEvent] = []
    for event in events:
        # Date range filter
        if date_from and event.date_sortable != "9999-99-99":
            if event.date_sortable < date_from:
                continue
        if date_to and event.date_sortable != "9999-99-99":
            if event.date_sortable > date_to:
                continue

        # Fact type filter
        if fact_types and event.type not in fact_types:
            continue

        # Provider filter
        if providers:
            event_provider = _extract_provider_from_source(event.source)
            if event_provider and event_provider not in providers:
                continue

        filtered.append(event)

    return filtered


def _extract_provider_from_source(source: str) -> str | None:
    """Extract provider/facility name from a source string.

    Returns the provider name if found, None otherwise.
    """
    import re

    # Match patterns like "VCU Medical Center (provider)" or "Dr. Smith (provider)"
    match = re.match(r"^(.+?)\s*\((?:provider|facility|role).*?\)$", source)
    if match:
        return match.group(1).strip()
    return None


def render_timeline_markdown(events: list[TimelineEvent]) -> str:
    """Render timeline events as markdown for PDF export or text display.

    Produces a clean, chronological timeline suitable for printing or inclusion
    in VA form attachments.
    """
    if not events:
        return "No medical events recorded in the timeline.\n"

    lines: list[str] = []
    lines.append("# Medical Record Timeline")
    lines.append("")
    lines.append(f"**Total events:** {len(events)}")
    lines.append("")

    current_year = None
    for event in events:
        # Group by year for readability
        year = event.date_sortable[:4] if event.date_sortable != "9999-99-99" else "Unknown"
        if year != current_year:
            current_year = year
            lines.append(f"## {year}")
            lines.append("")

        # Format the date for display
        date_display = event.date if event.date else "Unknown date"
        type_label = event.type.replace("_", " ").title()

        lines.append(f"### {date_display} — {type_label}")
        lines.append("")
        lines.append(f"{event.full_description}")
        if event.quote:
            lines.append("")
            lines.append(f"> \"{event.quote}\"")
        lines.append("")
        lines.append(f"*Source: {event.source}*")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def find_relevant_excerpts(
    documents: list[ExtractedDocument],
    query: str,
    *,
    top_k: int = 5,
    excerpt_chars: int = 700,
) -> str:
    """Cheap keyword-overlap retrieval of raw record excerpts relevant to a claim.

    Returns labelled excerpts for inclusion in verification prompts. Deterministic and
    dependency-free, so verification always has raw-source context, not just the digest.
    Paragraph splitting is cached per document so very large record sets are only
    parsed once across all claim batches.
    """
    query_tokens = set(_tokens(query))
    if not query_tokens:
        return ""

    # Build the paragraph corpus first so token weights reflect how common each
    # term is across ALL records (IDF). Distinctive terms then dominate the
    # ranking instead of boilerplate repeated on every page.
    corpus: list[tuple[str, str, frozenset[str]]] = []
    df: Counter[str] = Counter()
    for doc in documents:
        for paragraph in paragraph_index(doc):
            tokens = _tokens(paragraph.text)
            if tokens:
                corpus.append((paragraph.label, paragraph.text, tokens))
                for token in tokens:
                    df[token] += 1
    total = len(corpus)
    if not total:
        return ""

    def weight(token: str) -> float:
        return math.log((total + 1) / (df.get(token, 0) + 1)) + 1.0

    query_weight = sum(weight(t) for t in query_tokens) or 1.0

    scored: list[tuple[float, str]] = []
    for label, text, tokens in corpus:
        overlap = sum(weight(t) for t in query_tokens & tokens) / query_weight
        if overlap >= 0.15:
            scored.append((overlap, f"[{label}]\n{text[:excerpt_chars]}"))

    scored.sort(key=lambda item: item[0], reverse=True)
    seen: set[str] = set()
    unique: list[str] = []
    for _, excerpt in scored:
        key = excerpt[:120]
        if key not in seen:
            seen.add(key)
            unique.append(excerpt)
        if len(unique) >= top_k:
            break
    return "\n\n---\n\n".join(unique)
