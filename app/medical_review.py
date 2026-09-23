"""Exhaustive medical-record review built for very large record sets (1 to ~5,000 pages).

Pipeline:
  documents -> overlapping chunks -> duplicate-chunk skip -> PARALLEL LLM fact
  extraction (with one retry, transient failures only) -> mechanical fact dedup ->
  hierarchical LLM merge -> capped, ordered digest -> full-coverage narrative summary.

Both pathways (evaluate and draft) rely on this module to build a structured,
citable digest of every uploaded medical document.
"""
from __future__ import annotations

import datetime
import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import astuple, dataclass, field, replace
from functools import lru_cache
from typing import Any, Callable

import contextvars
import logging
import time

from . import config, tracing
from .agiloop_telemetry import track_feature_error, track_goal
from .documents import (
    Chunk,
    ExtractedDocument,
    chunk_page_labelled_text,
    iter_page_labelled_chunks,
    page_limit_message,
    paragraph_index,
)
from .llm import (
    CircuitBreakerOpenError,
    LLMAuthError,
    LLMClient,
    LLMError,
    LLMService,
    LLMTimeoutError,
    LLMUpstreamError,
    QueueFullError,
)
from .logging_config import PhaseTimer, get_request_id
from .pipeline_guard import check_pipeline_cancelled, pipeline_as_completed
from .preflight import REFUSAL_STATUSES
from .profiler import get_current_run_profiler, worker_timer
from .prompt_sanitize import GUARD_NOTE, sanitize_for_prompt
from .va_gov_export import section_map

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
      "source": "the file and page you took this from, exactly as the [file — page N] markers spell it, e.g. 'clinic.pdf p.7'",
      "quote": "short verbatim quote from the chunk supporting this fact",
      "section": "the record section this fact came from (e.g. Problem list, Medications, Lab results) or 'unknown'"
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
- 'source' must name the file and page the quoted text sits under, copied from the
  nearest [file — page N] marker above it. Never answer 'chunk', 'records' or a
  bare page number: a citation that cannot be checked is useless downstream.
- Prefer the exact date printed next to the quote over an inferred one. If a date is
  ambiguous, prefer the dates listed for this chunk below.
- Prefer many precise facts over few broad ones; when in doubt about relevance, include the fact.

This chunk covers: {source_hint}
Record sections present in this chunk: {section_hint}
Dates printed on these pages: {page_dates}

CHUNK TEXT:
<<<
{chunk_text}
>>>

{guard_note}"""

MERGE_SYSTEM = (
    """You are consolidating extracted medical facts from multiple chunks of the \
same record set into one authoritative digest. Deduplicate identical facts, keep every \
distinct fact, resolve trivially different phrasings, and keep all source citations \
(the file and page each fact came from — never replace them with a chunk number). \
Do not add facts that were not provided. Output JSON only.

"""
    # These facts were extracted from untrusted record text, so their description
    # and quote fields carry whatever the record contained — including text shaped
    # like instructions. This is the second-order injection path: a PDF's payload
    # launders through the digest model's JSON and lands here, where it used to
    # arrive with no guard at all. The note goes in the *system* message (the
    # stronger role), and the payload is escaped by ``sanitize_for_prompt`` at the
    # call site, which is the mechanical half.
    + GUARD_NOTE
)

# The elements a VA lay/witness statement has to establish, in the order a rating
# decision reads them. A digest fact is only useful to the drafting pass if it can
# be attached to one of these, so the mapping is explicit rather than left to the
# prose of a prompt.
STATEMENT_ELEMENTS: tuple[str, ...] = (
    "in_service_event",
    "current_diagnosis",
    "nexus",
    "functional_impact",
    "severity_frequency",
    "treatment_history",
    "buddy_observable",
    "other",
)

# Digest ``type`` -> statement element. Types the digest prompt asks for are the
# keys; anything unknown falls through to the nexus text check below, then "other".
_TYPE_TO_ELEMENT: dict[str, str] = {
    "in_service_event": "in_service_event",
    "diagnosis": "current_diagnosis",
    "symptom": "severity_frequency",
    "test_result": "severity_frequency",
    "treatment": "treatment_history",
    "medication": "treatment_history",
    "hospitalization": "treatment_history",
    "provider_visit": "treatment_history",
    "functional_limitation": "functional_impact",
    "observable_behavior": "buddy_observable",
    "administrative": "other",
    "other": "other",
}

# A medical opinion on causation is the nexus evidence itself, whatever ``type``
# the model labelled it with — a diagnosis line that quotes "at least as likely as
# not ... caused by service" is worth more to the statement than the diagnosis.
_NEXUS_TEXT_RE = re.compile(
    r"\b(as likely as not|at least as likely|more likely than not|less likely than not|"
    r"secondary to|caused by|causally related|aggravat\w*|service[- ]connected|"
    r"nexus|incurred in service|arose during service)\b",
    re.IGNORECASE,
)


def statement_element_for(fact: "MedicalFact") -> str:
    """Map one digest fact onto the lay-statement element it can support.

    Deterministic and dependency-free, so the drafting/verification prompts and
    the coverage panel all agree on what the record set does and does not
    support (see ``MedicalDigest.element_coverage``).
    """
    if _NEXUS_TEXT_RE.search(f"{fact.description} {fact.quote}"):
        return "nexus"
    return _TYPE_TO_ELEMENT.get(_norm_key(fact.type), "other")


@dataclass
class MedicalFact:
    date: str
    type: str
    description: str
    source: str
    quote: str = ""
    # Resolved citation. ``source`` stays the human-readable string the model
    # produced (and is what the export shows); these two are the machine-checkable
    # version, filled from the chunk's own page markers when the model's answer
    # could not be resolved to a page.
    document: str = ""
    page: int = 0
    section: str = ""


@dataclass
class MedicalDigest:
    """Complete extracted evidence plus a separate, lossy narrative summary.

    ``facts`` is authoritative for retrieval, exports, and saved results. Only
    exact duplicates are removed; prompt budgets never truncate this store.
    """

    facts: list[MedicalFact] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)
    summary: str = ""
    pages_reviewed: int = 0
    chunks_reviewed: int = 0
    duplicates_skipped: int = 0
    # Coverage: what the source files contained versus what was actually read.
    # Real record bundles are routinely part scan, so a report that claims
    # "reviewed 3,000 pages" while 1,200 of them were image-only is worse than one
    # that says so — the user can then OCR those pages and re-run.
    pages_in_files: int = 0
    unreadable_pages: int = 0
    chunks_without_facts: int = 0
    duplicate_pages: list[dict[str, Any]] = field(default_factory=list)
    # Duplicated pages whose copy came from a *different* file. A page re-printed
    # inside one bundle is a duplicate; the same page arriving from two sources is
    # corroboration, and a statement can lean harder on the second.
    corroborated_pages: list[dict[str, Any]] = field(default_factory=list)
    files: list[dict[str, Any]] = field(default_factory=list)
    # Legacy saved results may have lost facts to the former storage cap. Keep
    # their warning on reload; new reviews retain evidence and leave this at zero.
    facts_dropped_by_cap: int = 0
    # Result of checking each fact's quote against the page it cites (see
    # ``verify_citations``): the one measurement that turns "the report cites
    # page 7" into "page 7 really says this".
    citation_check: dict[str, Any] = field(default_factory=dict)

    @property
    def coverage_ratio(self) -> float:
        """Share of source pages whose text was actually extracted (1.0 = all)."""
        if not self.pages_in_files:
            return 1.0
        return max(0.0, min(1.0, (self.pages_in_files - self.unreadable_pages)
                             / self.pages_in_files))

    def keyword_overlap(self, query: str) -> float:
        """Share of the query's content words that appear anywhere in the digest.

        A cheap, non-IDF signal used to decide whether "no raw record text matched"
        really means the record set is silent about a claim, or only that the claim
        and the record word the same thing differently — the digest paraphrases the
        records, so it is the wider net for that case.
        """
        query_tokens = set(_tokens(query))
        if not query_tokens or not self.facts:
            return 0.0
        fact_tokens = _tokens(
            " ".join(f"{fact.description} {fact.quote}" for fact in self.facts)
        )
        return len(query_tokens & set(fact_tokens)) / len(query_tokens)

    def element_coverage(self) -> dict[str, int]:
        """Count facts per lay-statement element (see ``statement_element_for``)."""
        counts: Counter[str] = Counter(statement_element_for(fact) for fact in self.facts)
        return {element: counts.get(element, 0) for element in STATEMENT_ELEMENTS}

    def as_json_text(self, max_facts: int | None = None) -> str:
        """Bounded prompt view, not the serialization used for saved results."""
        limit = max(0, config.MAX_DIGEST_FACTS if max_facts is None else max_facts)
        selected = self.facts[:limit]
        return json.dumps(
            {
                "facts": [vars(f) for f in selected],
                "total_facts": len(self.facts),
                "selection_limited": len(selected) < len(self.facts),
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
        """Chronological, evenly strided sample of the timeline for summaries.

        Facts are ordered by date first (undated last, stable for ties), so a
        summary model reads the record set as a progression rather than in
        upload order. With thousands of facts, taking only the head of the
        timeline would bias summaries toward the earliest documents; striding
        keeps every era visible.
        """
        ordered = sorted(self.facts, key=lambda f: _normalize_date_for_sort(f.date))

        def _lines(facts: list[MedicalFact]) -> list[str]:
            return [f"[{f.date}] ({f.type}) {f.description}  — {f.source}" for f in facts]

        if len(ordered) <= max_entries:
            return "\n".join(_lines(ordered))
        step = len(ordered) / max_entries
        picked = [ordered[int(i * step)] for i in range(max_entries)]
        if ordered[-1] is not picked[-1]:
            picked[-1] = ordered[-1]
        return "\n".join(_lines(picked))

    def relevant_facts_text(
        self,
        query: str,
        *,
        max_facts: int = 150,
        budget_chars: int = 90_000,
        always_include_types: tuple[str, ...] = ("in_service_event", "hospitalization"),
        sort_dates: bool = False,
    ) -> str:
        """Return digest facts ranked by relevance to a claim/observation query.

        Replaces naive head-truncation of the full JSON digest: for large record
        sets the evidence relevant to a given claim can sit anywhere in thousands
        of facts, so each verification/grounding prompt receives the facts that
        actually match it (plus all high-priority event types), within a budget.

        With ``sort_dates=True`` the retained facts are *presented* in
        chronological order (undated last; relevance order breaks ties), while
        selection is still relevance-ranked — the right facts are kept, and a
        drafting prompt reads them as a timeline. Verification prompts keep the
        default relevance order: for judging a claim, best evidence first.
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

        limit = max(0, min(max_facts, config.MAX_DIGEST_FACTS, len(self.facts)))

        def selection_header(count: int, *, chronological: bool = False) -> str:
            order = ", presented in chronological order" if chronological else ""
            return (
                f"({count} of {len(self.facts)} retained facts selected for relevance{order}; "
                "this bounded selection is not the full evidence store)"
            )

        # Budget accounting uses the worst-case header so the chronological
        # variant can never push the body past the budget before the final slice.
        lines: list[str] = []
        used = len(selection_header(limit, chronological=sort_dates)) + 1
        entries: list[tuple[str, str, int]] = []  # (date_sort_key, line, selection_rank)
        count = 0
        for rank, (_, _, fact) in enumerate(matches + fillers):
            if count >= limit:
                break
            line = f"[{fact.date}] ({fact.type}) {fact.description} — {fact.source}"
            if fact.quote:
                line += f" | quote: \"{fact.quote}\""
            if used + len(line) + 1 > budget_chars:
                continue
            used += len(line) + 1
            count += 1
            entries.append((_normalize_date_for_sort(fact.date), line, rank))
        if sort_dates:
            entries.sort(key=lambda item: (item[0], item[2]))
        lines = [line for _, line, _ in entries]
        header = selection_header(len(lines), chronological=sort_dates)
        return (header + "\n" + "\n".join(lines))[:max(0, budget_chars)]


# "[clinic.pdf — page 7]" / "[clinic.pdf — block 3]" as a whole citation.
_MARKER_CITATION_RE = re.compile(
    r"^\[?(?P<doc>[^\[\]]+?)\s*—\s*(?:page|block)\s*(?P<num>\d+)\]?$", re.IGNORECASE
)
# "clinic.pdf p.7" / "notes.docx b.3" / the chunk-span form "clinic.pdf p.7-p.9".
_LABEL_CITATION_RE = re.compile(
    r"^(?P<doc>.+?)\s+[pb]\.(?P<num>\d+)(?:\s*[-–]\s*[pb]\.\d+)?$", re.IGNORECASE
)


def _parse_citation(source: str) -> tuple[str, int]:
    """Split a citation string into ``(document, first_page)``; ``("", 0)`` if none.

    Deliberately strict: a free-text source like ``"chunk 2/9"`` or a multi-file
    span (``"a.pdf p.3; b.pdf p.1"``) returns no page rather than a guessed one,
    and the caller then falls back to the chunk's own page span.
    """
    text = (source or "").strip()
    if not text:
        return "", 0
    match = _MARKER_CITATION_RE.match(text) or _LABEL_CITATION_RE.match(text)
    if not match:
        return "", 0
    document = match.group("doc").strip()
    if not document or ";" in document:
        return "", 0
    return document, int(match.group("num"))


def _fact_from_raw(
    raw: dict,
    fallback_source: str,
    *,
    document: str = "",
    page: int = 0,
    section: str = "",
) -> MedicalFact | None:
    """Build a fact, resolving its citation against the chunk it came from.

    ``document``/``page``/``section`` are the chunk's own page markers: used when
    the model's ``source`` cannot be resolved to a page, so a fact still points at
    the pages it was read from instead of at a chunk number that maps to nothing.
    """
    description = str(raw.get("description", "") or "").strip()
    if not description:
        return None
    model_source = str(raw.get("source", "") or "").strip()
    cited_document, cited_page = _parse_citation(model_source)
    raw_section = str(raw.get("section", "") or "").strip()
    if _norm_key(raw_section) in ("", "unknown", "n/a", "none"):
        raw_section = ""
    # The merge pass is handed ``vars(fact)`` and often echoes the structured
    # citation back; accept it, so a merged fact does not lose the page the
    # extraction pass worked to resolve.
    raw_document = str(raw.get("document", "") or "").strip()
    raw_page = raw.get("page")
    model_page = raw_page if isinstance(raw_page, int) else 0
    if not model_page and isinstance(raw_page, str) and raw_page.isdigit():
        model_page = int(raw_page)
    return MedicalFact(
        date=str(raw.get("date", "unknown") or "unknown").strip(),
        type=str(raw.get("type", "other") or "other").strip().lower(),
        description=description,
        source=model_source or fallback_source,
        quote=str(raw.get("quote", "") or "").strip(),
        document=cited_document or raw_document or document,
        page=cited_page or model_page or page,
        section=raw_section or section,
    )


# How many leading words of a quote are matched against the page text. A whole
# quote can straddle a page break or contain the model's ellipsis; a long prefix
# is enough to prove the fact came from the page it cites.
_CITATION_PROBE_WORDS = 12
# Quotes shorter than this cannot be checked meaningfully — four words appear on
# hundreds of pages, so "not found on this page" would be noise.
_CITATION_MIN_PROBE_WORDS = 4


def _citation_probe(quote: str) -> str:
    """Normalized leading words of a quote, or "" when too short to check."""
    words = re.findall(r"[a-z0-9]+", quote.lower())
    if len(words) < _CITATION_MIN_PROBE_WORDS:
        return ""
    return " ".join(words[:_CITATION_PROBE_WORDS])


def verify_citations(
    facts: list[MedicalFact], documents: list[ExtractedDocument]
) -> dict[str, Any]:
    """Check that each fact's quote actually occurs on the page the fact cites.

    The digest is produced by an LLM reading overlapping chunks, so a fact can
    cite a page it does not appear on — the quote came from the neighbouring
    chunk, or the model reached for a plausible page number. The check is cheap
    because both sides are already in memory, and it is the only thing standing
    between a citation and an assertion: a report a veteran signs should say how
    many of its citations were actually verified.

    Facts without a resolved page, or with a quote too short to be meaningful,
    are counted as ``skipped`` rather than guessed at.
    """
    page_texts: dict[tuple[str, int], str] = {}
    for doc in documents:
        for page in doc.pages:
            page_texts[(doc.filename, page.page)] = re.sub(
                r"\s+", " ", page.text.lower()
            )
    checked = 0
    skipped = 0
    missing: list[dict[str, Any]] = []
    for fact in facts:
        probe = _citation_probe(fact.quote)
        text = page_texts.get((fact.document, fact.page)) if fact.document else None
        if not probe or text is None:
            skipped += 1
            continue
        checked += 1
        if probe not in text:
            missing.append(
                {
                    "document": fact.document,
                    "page": fact.page,
                    "description": fact.description[:160],
                    "quote": fact.quote[:160],
                }
            )
    result: dict[str, Any] = {
        "checked": checked,
        "missing": len(missing),
        "skipped": skipped,
        "examples": missing[:5],
    }
    if checked:
        result["verified_ratio"] = round((checked - len(missing)) / checked, 3)
    logger.info(
        "citation check checked=%d missing=%d skipped=%d",
        checked,
        len(missing),
        skipped,
        extra={
            "request_id": get_request_id() or "-",
            "phase": "records:citations",
            "status": "ok" if not missing else "mismatch",
            "checked": checked,
            "missing": len(missing),
        },
    )
    return result


def _restore_citations(
    merged: list[MedicalFact], sources: list[MedicalFact]
) -> list[MedicalFact]:
    """Re-attach the citation the merge model dropped when it reworded a fact.

    The merge pass rewrites descriptions, so a fact that comes back without a
    document/page is matched back to the pre-merge fact it consolidated. Without
    this, every merged fact loses its page and the report's citations degrade to
    "records" — exactly the property the extraction pass establishes.
    """
    if not merged:
        return merged
    by_description: dict[str, MedicalFact] = {}
    for fact in sources:
        by_description.setdefault(_norm_key(fact.description), fact)
        by_description.setdefault(f"{_norm_key(fact.date)}|{_norm_key(fact.description)}", fact)
    for fact in merged:
        if fact.document and fact.page:
            continue
        original = by_description.get(_norm_key(fact.description)) or by_description.get(
            f"{_norm_key(fact.date)}|{_norm_key(fact.description)}"
        )
        if original is None:
            continue
        fact.document = fact.document or original.document
        fact.page = fact.page or original.page
        fact.section = fact.section or original.section
    return merged


def _norm_key(text: str) -> str:
    """Whitespace-collapsed, case-folded comparison key.

    ``" ".join(text.split())`` rather than ``re.sub(r"\s+", " ", text).strip()``:
    identical output for every whitespace class (checked against NBSP, thin space,
    vertical tab and tabs), 3-4x faster, and this runs twice per fact inside
    ``_dedupe_facts`` — which is on the merge path of every run, once per merge
    round over the whole fact list (measured 496 ms for 50,000 facts before, and
    the regex was nearly all of it).
    """
    return " ".join(text.split()).lower()


# Matches a source string that is a citation rather than an entity name. Used to
# keep page citations out of anything that lists providers/facilities.
_CITATION_SOURCE_RE = re.compile(r"\b[pb]\.\d|\b(?:page|block)\s*\d", re.IGNORECASE)


# ------------------------------------------------------- duplicate page detection
def _cross_source_corroborations(
    duplicates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Duplicates whose copy lives in a different file (see ``MedicalDigest``)."""
    corroborations: list[dict[str, Any]] = []
    for entry in duplicates:
        origin = str(entry.get("duplicate_of", ""))
        document = str(entry.get("document", ""))
        origin_file = re.split(r"\s+[pb]\.\d+\s*$", origin)[0].strip() if origin else ""
        if origin_file and document and origin_file != document:
            corroborations.append(
                {"document": document, "page": entry.get("page"), "also_in": origin_file}
            )
    return corroborations


def _dedupe_pages(
    documents: list[ExtractedDocument],
) -> tuple[list[ExtractedDocument], list[dict[str, Any]]]:
    """Drop only exact text repetitions, naming every page that was dropped.

    Normalize newline encodings only. Numbers, punctuation, case, whitespace and
    the full page tail can carry clinical meaning, even on near-identical forms.
    Dictionary keys compare the complete text, not a lossy similarity signature.
    """
    seen_pages: dict[str, str] = {}
    unique_docs: list[ExtractedDocument] = []
    duplicates: list[dict[str, Any]] = []

    for doc in documents:
        kept = ExtractedDocument(
            filename=doc.filename,
            total_pages=doc.total_pages,
            unreadable_pages=list(doc.unreadable_pages),
            pagination=doc.pagination,
        )
        for page in doc.pages:
            check_pipeline_cancelled()
            page_text = page.text.replace("\r\n", "\n").replace("\r", "\n")
            origin = seen_pages.get(page_text)
            if origin is not None:
                duplicates.append(
                    {
                        "document": doc.filename,
                        "page": page.page,
                        "duplicate_of": origin,
                    }
                )
                continue
            seen_pages[page_text] = page.label
            kept.pages.append(page)
        if kept.pages:
            unique_docs.append(kept)
    return unique_docs, duplicates


def _dedupe_evidence(facts: list[MedicalFact]) -> list[MedicalFact]:
    """Remove exact repetitions without losing different quotes or provenance."""
    seen: set[tuple[Any, ...]] = set()
    unique: list[MedicalFact] = []
    for fact in facts:
        check_pipeline_cancelled()
        key = astuple(fact)
        if key not in seen:
            seen.add(key)
            unique.append(fact)
    return unique


def _dedupe_facts(facts: list[MedicalFact]) -> list[MedicalFact]:
    """Mechanically drop repeated facts (same date + description), keeping order."""
    seen: set[str] = set()
    unique: list[MedicalFact] = []
    for fact in facts:
        check_pipeline_cancelled()
        key = f"{_norm_key(fact.date)}|{_norm_key(fact.description)}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(fact)
    return unique


def review_medical_records(
    llm: LLMService,
    documents: list[ExtractedDocument],
    progress: ProgressCallback | None = None,
) -> MedicalDigest:
    """Run the full exhaustive review over all uploaded records.

    Scales to thousands of pages: chunks are digested in parallel, duplicate
    chunks are skipped, transient failures are retried, and large fact lists are
    merged hierarchically instead of in one oversized call. A rejection that
    proves every remaining call would fail the same way (a refusal status) ends
    the pass early, and the chunks that were never sent are reported apart from
    the ones that failed.
    """
    check_pipeline_cancelled()
    if not documents:
        raise ValueError("No medical records provided.")

    pages = sum(len(doc.pages) for doc in documents)
    # The cap is measured on the pages the *files* contain, not the pages that
    # yielded text: a bundle of scans has few readable pages and would otherwise
    # slip past the limit and spend hours of LLM calls on it.
    pages_in_files = sum(doc.source_page_count for doc in documents)
    if pages_in_files > config.MAX_RECORD_PAGES:
        raise ValueError(
            page_limit_message(pages_in_files, config.MAX_RECORD_PAGES)
        )

    # Drop duplicate pages (across and within files) BEFORE chunking: record
    # bundles frequently repeat the same pages, and re-digesting them wastes
    # hours on very large sets without adding evidence. Every skip is recorded
    # with the page it duplicated, so the coverage report can name them.
    unique_docs, duplicate_pages = _dedupe_pages(documents)
    duplicates_skipped = len(duplicate_pages)
    if not unique_docs:
        raise ValueError("Records contain no extractable unique text.")

    unreadable_pages = sum(doc.unreadable_count for doc in documents)

    # Streaming chunking: cut chunks directly off the page objects instead of
    # materializing the joined corpus (measured 2026-09-21: the join plus its
    # slicing churn peaked at 37.7 MB traced on a 5,000-page set — 2.5x what the
    # pipeline retains — for text that exists only to be sliced). The iterator
    # produces byte-identical chunks to chunk_page_labelled_text; see its
    # docstring for the locality argument and the property test that pins it.
    chunks = list(
        iter_page_labelled_chunks(
            page for doc in unique_docs for page in doc.pages
        )
    )
    total_units = len(chunks)
    # Deterministic per-page context for the digest prompt: the dates actually
    # printed on the page (an anchor for the fact's own date) and the VA.gov
    # section the page belongs to. Both beat asking the model to remember them.
    page_dates: dict[str, str] = {}
    page_sections: dict[str, str] = {}
    for doc in unique_docs:
        sections = section_map(doc)
        for doc_page in doc.pages:
            check_pipeline_cancelled()
            dates = _dates_in_text(doc_page.text)
            if dates:
                page_dates[doc_page.label] = ", ".join(dates)
            section = sections.get(doc_page.page, "")
            if section:
                page_sections[doc_page.label] = section

    if progress:
        dup_note = (
            f" ({duplicates_skipped} duplicate page(s) skipped)" if duplicates_skipped else ""
        )
        unreadable_note = (
            f" {unreadable_pages:,} page(s) have no extractable text"
            if unreadable_pages
            else ""
        )
        progress(
            0.05,
            f"Reviewing {pages:,} pages in {len(chunks)} chunk(s){dup_note}{unreadable_note} "
            f"using {config.RECORDS_CONCURRENCY} parallel worker(s)…",
        )

    _ctx_request_id = get_request_id()  # capture for worker threads
    _ctx_profiler_run = get_current_run_profiler()
    # Spans do not follow contextvars into a thread pool either, so the digest
    # thread re-attaches this parent context — otherwise every chunk span would
    # start a trace of its own instead of nesting under the run.
    _ctx_span = tracing.current_span_context()

    def digest_chunk(chunk: Chunk) -> dict[str, object]:
        check_pipeline_cancelled()
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
                        source_hint=sanitize_for_prompt(chunk.source_hint, max_chars=200),
                        section_hint=sanitize_for_prompt(
                            _chunk_sections(chunk, page_sections), max_chars=300
                        ),
                        page_dates=sanitize_for_prompt(
                            _chunk_dates(chunk, page_dates), max_chars=400
                        ),
                        chunk_text=sanitize_for_prompt(chunk.text, max_chars=1_000_000),
                        guard_note=GUARD_NOTE,
                    ),
                    model=llm.fast_model,
                    max_tokens=8000,
                    phase="records:digest",
                )
            check_pipeline_cancelled()
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
    failed: dict[int, BaseException] = {}
    # The first failure that carried a *reason*, kept across the retry round below.
    # That round clears `failed` and refills it, and once the breaker is OPEN every
    # refilled entry is its fail-fast rejection — a symptom shared by all the
    # chunks, which says nothing about what to fix. This is the error that does.
    cause: BaseException | None = None
    fail_fast_chunks = 0
    # Calls never made: the pass stopped when the endpoint refused the run, so
    # these are reported apart from the chunks that actually failed.
    not_attempted: set[int] = set()
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

    def run_round(pending: list[Chunk], *, retry: bool = False) -> None:
        nonlocal cause, fail_fast_chunks
        done = 0
        stopped = False
        pool = ThreadPoolExecutor(max_workers=config.RECORDS_CONCURRENCY)
        try:
            future_map = {
                pool.submit(contextvars.copy_context().run, digest_chunk, c): c
                for c in pending
            }
            for future in pipeline_as_completed(future_map):
                chunk = future_map[future]
                if future.cancelled():
                    # Never sent: the pass stopped at a refusal (below). Counted
                    # apart from failures, and it does not advance the progress
                    # count, which reports calls actually made.
                    not_attempted.add(chunk.index)
                    continue
                done += 1
                refused = False
                try:
                    results[chunk.index] = future.result()
                except Exception as exc:  # noqa: BLE001 - record and retry later
                    failed[chunk.index] = exc
                    if isinstance(exc, (CircuitBreakerOpenError, QueueFullError)):
                        # Not an endpoint fault: the call was refused before it was
                        # made. Counted separately so the summary can say how much
                        # of the failure is cascade, and never reported as the cause.
                        fail_fast_chunks += 1
                    elif isinstance(exc, LLMAuthError):
                        # A credential verdict is not a property of any chunk: the
                        # same refusal awaits every other file, so bisecting and
                        # quarantining can only manufacture failures (measured
                        # 2026-09-22: one 401 bisected five levels deep and
                        # quarantined nine healthy files). Bail out immediately —
                        # the raise propagates out of run_round past the retry
                        # round into review_medical_records' own raise.
                        raise LLMAuthError(
                            f"The endpoint refused the credentials — this is fatal for the "
                            f"whole run, not for chunk {chunk.index}: no file can succeed "
                            "until the API key is fixed. Fix the key or credit balance and "
                            "re-run; completed work resumes from state."
                        ) from exc
                    else:
                        refused = _refused_as_configured(exc)
                        if refused and not _refused_as_configured(cause):
                            # The refusal stopped the run, so it — not an earlier
                            # transient failure the map kept — is the error to quote.
                            cause = exc
                        elif cause is None:
                            cause = exc
                if progress:
                    # Each round counts against its own targets. Counting the retry
                    # pass against the original total produced "654/327 chunks done…"
                    # on a full-bundle failure (measured live), which reads as
                    # progress past the end of the run.
                    if retry:
                        progress(
                            0.62 + 0.03 * done / max(len(pending), 1),
                            f"Retrying failed chunks — {done}/{len(pending)} done…",
                        )
                    else:
                        progress(
                            0.05 + 0.55 * done / max(total_units, 1),
                            f"Extracting facts — {done}/{total_units} chunks done…",
                        )
                if refused and not stopped:
                    # A refusal status means every other chunk would be refused the
                    # same way — the evidence the endpoint preflight blocks on — so
                    # the queued calls are canceled instead of visited. Calls already
                    # in flight (at most workers - 1 of them) still report their own
                    # outcome.
                    stopped = True
                    for queued in future_map:
                        if queued is not future and not queued.done():
                            queued.cancel()
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        if stopped and not_attempted:
            logger.warning(
                "records digest stopped early round=%s attempted=%d not_attempted=%d error=%s",
                "retry" if retry else "first",
                done,
                len(not_attempted),
                _failure_summary(cause) if cause is not None else "-",
                extra={
                    "request_id": rid,
                    "phase": "records:digest",
                    "status": "stopped",
                    "attempted": done,
                    "not_attempted": len(not_attempted),
                },
            )
            if progress:
                progress(
                    0.65 if retry else 0.60,
                    f"Stopped early — {len(not_attempted)} chunk(s) not attempted after "
                    "the endpoint refused the run.",
                )

    with tracing.phase_span(
        "records:digest", chunks=total_units, concurrency=config.RECORDS_CONCURRENCY, pages=pages
    ):
        run_round(chunks)

    # Retry failed chunks once; parallel bursts can hit transient rate limits — but
    # only failures a retry could plausibly change are re-attempted. A call the
    # breaker refused *before making it* meets the same open breaker, and a rejection
    # the pipeline marked non-retriable (a rejected key, a model the account cannot
    # use) reproduces itself exactly. Re-attempting those cost a full second pass
    # over every chunk — hundreds of fail-fast rejections and their warnings — to
    # reach the conclusion the first pass already had.
    if failed:
        retry_targets = [
            c for c in chunks if c.index in failed and _retriable_failure(failed[c.index])
        ]
        not_retried = {i: exc for i, exc in failed.items() if not _retriable_failure(exc)}
        logger.warning(
            "records digest retry pending=%d not_retried=%d failed=%s",
            len(retry_targets),
            len(not_retried),
            sorted(failed.keys()),
            extra={
                "request_id": rid,
                "phase": "records:digest",
                "status": "retry",
                "chunks": len(retry_targets),
                "not_retried": len(not_retried),
            },
        )
        failed.clear()
        if retry_targets:
            if progress:
                progress(0.62, f"Retrying {len(retry_targets)} failed chunk(s)…")
            with tracing.phase_span("records:digest", chunks=len(retry_targets), retry=True):
                run_round(retry_targets, retry=True)
        # Failures the retry round could not improve stay counted — including the
        # ones it skipped — so the summary reports what failed, not what ran twice.
        failed.update(not_retried)

    if failed:
        reason_exc = cause if cause is not None else failed[sorted(failed)[0]]
        first_error = _failure_summary(reason_exc)
        logger.error(
            "records review failed chunks=%s error=%s fail_fast_chunks=%d not_attempted=%d",
            sorted(failed.keys()),
            first_error,
            fail_fast_chunks,
            len(not_attempted),
            extra={
                "request_id": rid,
                "phase": "records:review",
                "status": "error",
                "chunks": len(failed),
                "fail_fast_chunks": fail_fast_chunks,
                "not_attempted": len(not_attempted),
                "error_class": type(reason_exc).__name__,
            },
        )
        # Cause first, then the advice for it, then the scale. The order is the
        # whole point: this string is what the user reads in the red box, and a
        # 2,000-page bundle fails as 300+ chunks, so leading with the labels pushed
        # the actual error and its fix past the end of a ~2,300-character wall.
        if isinstance(reason_exc, LLMAuthError):
            # Deliberately not wrapped into the "Record review failed: …" frame:
            # the batch runner's stop-the-pipeline branch keys on this class, and
            # the credential advice is the whole message a user needs here — a
            # chunk census would imply files were at fault when none are.
            raise reason_exc
        raise LLMError(
            f"Record review failed: {first_error}. "
            f"{_digest_failure_advice(reason_exc, fail_fast_chunks=fail_fast_chunks)} "
            f"{_failed_chunk_summary(failed, total_units, not_attempted=len(not_attempted))}"
        )

    # -------------------------------------------- collect facts in doc order
    all_facts: list[MedicalFact] = []
    conditions: Counter[str] = Counter()
    providers: Counter[str] = Counter()
    chunks_without_facts = 0
    chunks_by_index = {chunk.index: chunk for chunk in chunks}
    for index in sorted(results):
        check_pipeline_cancelled()
        data = results[index]
        chunk = chunks_by_index.get(index)
        fallback = chunk.source_hint if chunk is not None else f"chunk {index}/{len(chunks)}"
        document = chunk.pages[0][0] if chunk is not None and chunk.pages else ""
        page = chunk.pages[0][2] if chunk is not None and chunk.pages else 0
        section = _chunk_sections(chunk, page_sections, default="") if chunk is not None else ""
        facts_in_chunk = 0
        for raw in data.get("facts", []) or []:
            if not isinstance(raw, dict):
                continue
            fact = _fact_from_raw(
                raw, fallback, document=document, page=page, section=section
            )
            if fact:
                all_facts.append(fact)
                facts_in_chunk += 1
        if facts_in_chunk == 0:
            # A chunk that yielded nothing is indistinguishable from a chunk the
            # model skimmed; counted so the report can say how many there were.
            chunks_without_facts += 1
        for name in data.get("conditions_mentioned", []) or []:
            conditions[str(name).strip()] += 1
        for name in data.get("providers_and_facilities", []) or []:
            providers[str(name).strip()] += 1

    all_facts = _dedupe_evidence(all_facts)

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
        pages_in_files=pages_in_files,
        unreadable_pages=unreadable_pages,
        chunks_without_facts=chunks_without_facts,
        duplicate_pages=duplicate_pages,
        corroborated_pages=_cross_source_corroborations(duplicate_pages),
        files=[_file_coverage(doc) for doc in documents],
    )

    with (
        tracing.phase_span("records:merge", facts=len(all_facts)),
        PhaseTimer(logger, "records:merge", request_id=rid, facts=len(all_facts)),
    ):
        # Model consolidation is a lossy summary view, never the evidence store.
        summary_facts = _merge_facts(llm, digest, progress)
    # Both the evidence store and summary view are retained during summarization.
    try:
        _mem_cp("records:post_merge")
    except Exception:  # noqa: BLE001
        pass
    check_pipeline_cancelled()
    digest.citation_check = verify_citations(digest.facts, documents)
    with tracing.phase_span("records:summary"), PhaseTimer(logger, "records:summary", request_id=rid):
        digest.summary = _summarize(llm, replace(digest, facts=summary_facts))
    check_pipeline_cancelled()
    duration_ms = int((time.perf_counter() - _review_t0) * 1000)
    logger.info(
        "records review done pages=%d pages_in_files=%d unreadable_pages=%d chunks=%d "
        "chunks_without_facts=%d facts=%d duplicates_skipped=%d duration_ms=%d",
        pages,
        pages_in_files,
        unreadable_pages,
        len(chunks),
        chunks_without_facts,
        len(digest.facts),
        duplicates_skipped,
        duration_ms,
        extra={
            "request_id": rid,
            "phase": "records:review",
            "status": "ok",
            "duration_ms": duration_ms,
            "pages": pages,
            "pages_in_files": pages_in_files,
            "unreadable_pages": unreadable_pages,
            "chunks": len(chunks),
            "chunks_without_facts": chunks_without_facts,
            "facts": len(digest.facts),
        },
    )
    if progress:
        unreadable_note = (
            f" {unreadable_pages:,} page(s) could not be read (image-only) and are not "
            "covered — OCR them and re-run for a complete review."
            if unreadable_pages
            else ""
        )
        progress(
            0.8,
            f"Record review complete: {len(digest.facts):,} facts extracted from "
            f"{pages:,} pages ({digest.chunks_reviewed} chunks).{unreadable_note}",
        )
    return digest


# --------------------------------------------------- digest failure reporting

# How many failed chunk indices the digest error names before eliding the rest.
# Twelve is enough to show the spread across a multi-file bundle (the labels read
# "chunk 7, chunk 88, chunk 143") while keeping the whole message readable.
MAX_FAILED_CHUNK_LABELS = 12


def _refused_as_configured(exc: BaseException | None) -> bool:
    """Whether this failure proves every remaining chunk would be refused too.

    The refusal statuses are the endpoint preflight's own (`app.preflight`): a
    rejected key, a model the account cannot use, a base URL with no route. One
    chunk answered that way means the others will, so the pass can stop. A
    content-moderation 400 is deliberately not here — that rejection is about one
    chunk's wording, and the next chunk can pass.
    """
    return (
        isinstance(exc, LLMUpstreamError)
        and isinstance(exc.status_code, int)
        and exc.status_code in REFUSAL_STATUSES
    )


def _retriable_failure(exc: BaseException) -> bool:
    """Whether one more attempt could plausibly change this chunk's outcome.

    The retry round exists for transient bursts. Two failures cannot improve on a
    retry: a call the breaker refused *before making it* meets the same open
    breaker, and a provider rejection the pipeline marked non-retriable — a
    rejected key, a model the account cannot use — reproduces itself exactly.
    Everything else (a timeout, a 5xx, a rate limit, an unparsable reply) is
    retried as before.
    """
    if isinstance(exc, CircuitBreakerOpenError):
        return False
    return bool(getattr(exc, "retriable", True))


def _failed_chunk_summary(
    failed: Mapping[int, BaseException], total_units: int, *, not_attempted: int = 0
) -> str:
    """How much of the record set failed, without letting the list dominate.

    The count answers the question the label list used to: "is this one chunk or
    the whole bundle?" The labels stay, capped, because they say *where* to look.
    Chunks are listed by index across the whole run, so an ellipsis is not hiding
    anything the user would act on differently. A stopped pass adds how many
    chunks were never sent, so "all of them" can never be read from a count that
    only covers the calls that were made.
    """
    indexes = sorted(failed)
    shown = ", ".join(f"chunk {i}" for i in indexes[:MAX_FAILED_CHUNK_LABELS])
    if len(indexes) > MAX_FAILED_CHUNK_LABELS:
        shown += f", … (+{len(indexes) - MAX_FAILED_CHUNK_LABELS} more)"
    summary = f"Chunks affected: {len(indexes)} of {total_units} ({shown})."
    if not_attempted:
        summary += (
            f" {not_attempted} more chunk(s) were not attempted after the endpoint "
            "refused the run."
        )
    return summary


def _failure_summary(exc: BaseException) -> str:
    """One bounded line naming a failure: class first, then its message.

    The class name leads because it is the token that appears in the log, in this
    file, and in `TROUBLESHOOTING.md`, so a user quoting the message can be matched
    to a documented cause. The length cap is the message's, not ours: a provider
    error can be long, and the advice that follows it has to stay reachable.
    """
    return f"{type(exc).__name__}: {exc}"[:200]


def _digest_failure_advice(exc: BaseException, *, fail_fast_chunks: int = 0) -> str:
    """What the user should do about a failed digest, chosen from its cause.

    The unconditional "split the record set into smaller files" was wrong for every
    failure that size had nothing to do with, which is most of them: a rejected key
    or an unusable model id fails identically on a one-page record. Worse, following
    it costs the user real work (chopping a bundle) and changes nothing.
    """
    if isinstance(exc, LLMAuthError):
        return (
            "The endpoint refused the credentials (HTTP 401/403) — every call in the run "
            "would be refused identically, so this is fatal for the run rather than a "
            "problem with any file. Regenerate the API key, check the account's credit "
            "balance, then re-run: completed work resumes from state."
        )
    if isinstance(exc, CircuitBreakerOpenError):
        return (
            "The endpoint was already marked unhealthy when these chunks ran, so they were "
            "refused without a request being sent — the real error is the one quoted above. "
            "Resolve that and re-run; the record set does not need to be smaller."
        )
    if isinstance(exc, QueueFullError):
        return (
            "This app's own concurrent-call cap was reached, not the provider's. Re-run when "
            "the queue is idle, or raise VA_LSE_MAX_CONCURRENT_LLM_CALLS / "
            "VA_LSE_LLM_QUEUE_MAX_DEPTH for the load you are running."
        )
    status = getattr(exc, "status_code", None)
    if isinstance(exc, LLMUpstreamError) and isinstance(status, int):
        if 400 <= status < 500 and status != 429:
            return (
                f"The endpoint rejected the request (HTTP {status}), which retrying cannot fix: "
                "check the base URL, API key and model names in the sidebar (use Test connection), "
                "then re-run."
            )
        if status == 429:
            return (
                "The endpoint rate-limited these calls (HTTP 429) after their retries were spent. "
                "Re-run shortly; if it recurs, lower VA_LSE_MAX_CONCURRENT_LLM_CALLS or split the "
                "record set into smaller files."
            )
        return (
            f"The endpoint returned HTTP {status}. Re-run; if it persists, split the record set "
            "into smaller files."
        )
    if isinstance(exc, LLMTimeoutError):
        return (
            "These calls timed out. Re-run; if it recurs, raise "
            "VA_LSE_LLM_CALL_TIMEOUT_SECONDS or split the record set into smaller files."
        )
    if fail_fast_chunks:
        return (
            "Some chunks were refused while the endpoint was marked unhealthy. Re-run; if it "
            "persists, split the record set into smaller files."
        )
    return "Re-run the review; if it persists, split the record set into smaller files."


def _file_coverage(doc: ExtractedDocument) -> dict[str, Any]:
    """Per-file coverage row for the digest's coverage report."""
    return {
        "filename": doc.filename,
        "pages_in_file": doc.source_page_count,
        "pages_read": len(doc.pages),
        "unreadable_pages": doc.unreadable_count,
        "pagination": doc.pagination,
        "characters": doc.char_count,
    }


def _chunk_dates(chunk: Chunk, page_dates: dict[str, str]) -> str:
    """Dates printed on this chunk's pages, as an anchor for the fact dates."""
    found: list[str] = []
    for filename, kind, number in chunk.pages:
        label = f"{filename} {'p.' if kind == 'page' else 'b.'}{number}"
        for date in page_dates.get(label, "").split(", "):
            if date and date not in found:
                found.append(date)
    return ", ".join(found[:12]) if found else "none detected on these pages"


def _chunk_sections(
    chunk: Chunk, page_sections: dict[str, str], *, default: str = "unknown"
) -> str:
    """Record sections this chunk touches (VA.gov exports are section-structured)."""
    found: list[str] = []
    for filename, kind, number in chunk.pages:
        label = f"{filename} {'p.' if kind == 'page' else 'b.'}{number}"
        section = page_sections.get(label, "")
        if section and section not in found:
            found.append(section)
    return ", ".join(found[:6]) if found else default


# Merge batches must fit the model's OUTPUT budget, not just its input window:
# _merge_once asks the model to echo every distinct fact of the batch as JSON,
# and a real fact renders to roughly 180 chars (~48 tokens) of output. At the
# old batch size of 200 that demanded ~17k output tokens against a max_tokens
# of 8000 — every response truncated mid-JSON and 100% of merge calls failed
# (observed live 2026-09-21: 13/13 batches unparseable, ~10k chars each). At 48
# facts the echo is ~2.3k tokens: comfortably inside the budget.
# Bound for prompts whose payload is *derived* from record text: the merge batch
# (48 facts declared below), the sampled timeline in the summary, and the undated
# date-inference rows. Generous on purpose — the batch sizes bound the payload, and
# this is only here so a single field cannot flood the context window. Escaping is
# what matters at this size, not truncation.
DERIVED_TEXT_MAX_CHARS = 200_000

MERGE_BATCH_SIZE = 48
# Same arithmetic for the single-call path: 8,000 output tokens / ~48 tokens
# per fact ≈ 165 facts maximum; 120 leaves margin for prompt preamble and
# model verbosity.
MERGE_SINGLE_LIMIT = 120


def _merge_facts(
    llm: LLMService,
    digest: MedicalDigest,
    progress: ProgressCallback | None = None,
) -> list[MedicalFact]:
    """Consolidate facts, hierarchically when the list is too large for one call.

    A single merge call cannot hold thousands of facts, so oversized lists are
    merged in batches (in parallel), and the merged results are re-merged until
    the list fits one call or stops shrinking. Mechanical dedup runs between
    rounds so facts resolving to the same date+description collapse.
    """
    check_pipeline_cancelled()
    facts = _dedupe_facts(digest.facts)
    if len(facts) <= MERGE_SINGLE_LIMIT:
        try:
            consolidated = _restore_citations(_merge_once(llm, facts), facts) or facts
        except LLMError:
            consolidated = facts
        return consolidated

    current = facts
    _merge_rid = get_request_id() or "-"
    for round_no in range(1, 4):
        check_pipeline_cancelled()
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
            check_pipeline_cancelled()
            from .logging_config import _request_id_var as _rid_var

            tok = _rid_var.set(_merge_ctx)
            try:
                merged = _restore_citations(_merge_once(llm, batch), batch)
                check_pipeline_cancelled()
                return merged
            finally:
                try:
                    _rid_var.reset(tok)
                except ValueError:
                    pass

        pool = ThreadPoolExecutor(max_workers=config.RECORDS_CONCURRENCY)
        try:
            future_map = {
                pool.submit(contextvars.copy_context().run, _merge_with_ctx, batch): batch_index
                for batch_index, batch in enumerate(batches)
            }
            for future in pipeline_as_completed(future_map):
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
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        merged: list[MedicalFact] = []
        for batch_index in sorted(merged_by_batch):
            merged.extend(merged_by_batch[batch_index])
        merged = _dedupe_facts(merged)

        if len(merged) <= MERGE_SINGLE_LIMIT or len(merged) >= len(current):
            current = merged
            break
        current = merged

    return current


def _merge_once(llm: LLMService, facts: list[MedicalFact]) -> list[MedicalFact]:
    """One merge call over a batch-sized fact list (fast model).

    The payload is escaped: these facts were extracted from untrusted record text,
    so a record's own words — including text shaped like instructions — reach this
    prompt through the digest model's JSON. The guard note rides in the system
    message (see ``MERGE_SYSTEM``). The payload stays the last block of the user
    message, right after the ``\\n\\n`` intro, because callers locate it there.
    """
    data = llm.chat_json(
        MERGE_SYSTEM,
        "Deduplicate and consolidate these extracted medical facts. Keep every DISTINCT fact "
        "with its source. Return JSON: {\"facts\": [{\"date\",\"type\",\"description\",\"source\",\"quote\"}]}\n\n"
        + sanitize_for_prompt(
            json.dumps([vars(f) for f in facts]), max_chars=DERIVED_TEXT_MAX_CHARS
        ),
        model=llm.fast_model,
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


def _summarize(llm: LLMService, digest: MedicalDigest) -> str:
    """Narrative summary over the timeline — record-derived text, so guarded.

    Same second-order path as the merge: these descriptions came out of the digest
    model's reading of untrusted pages, and a summary prompt that carried them
    unguarded is a place an injected sentence could come back as an instruction.
    """
    return llm.chat(
        "You are a medical-records analyst. Write a concise narrative summary (max 250 words) "
        "of the record set: key diagnoses, treatment history, notable events, and current "
        "status. Plain text only.\n\n" + GUARD_NOTE,
        "Extracted facts (sampled evenly across the full timeline):\n"
        + sanitize_for_prompt(
            digest.condensed_timeline(max_entries=400)[:16000],
            max_chars=DERIVED_TEXT_MAX_CHARS,
        ),
        phase="records:summary",
    )


# ----------------------------------------------------------- relevance search
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for", "with", "by",
    "from", "is", "was", "were", "are", "been", "he", "she", "his", "her", "him", "i",
    "my", "me", "that", "this", "it", "as", "has", "have", "had", "be", "will", "since",
    "during", "about", "into", "their", "they", "them", "you", "your", "we", "our",
}


# Two-letter clinical abbreviations that matter (GI bleed, CT scan, IV
# antibiotics, EKG) even though the general rule ignores tokens this short.
_SHORT_CLINICAL_TOKENS = frozenset(
    {"ct", "gi", "iv", "er", "pt", "mi", "tb", "ekg", "ecg", "bp", "hr", "ent", "rbc"}
)

# Lay term -> the word the record set is likely to use. A veteran writes "my neck
# and lower back"; the record writes "cervical and lumbar strain". Without a
# bridge those two share no token, and a verification against those records
# reports no evidence for a claim the records plainly document. The mapping is
# additive (the original token is kept), symmetric in effect (both sides expand),
# and deliberately small: it covers the lay/clinical pairs that recur in VA
# claims, not a general medical ontology.
_CLINICAL_SYNONYMS: dict[str, str] = {
    "neck": "cervical",
    "cervical": "cervical",
    "back": "lumbar",
    "lumbar": "lumbar",
    "dorsal": "lumbar",
    "thoracic": "thoracic",
    "sob": "dyspnea",
    "breath": "dyspnea",
    "breathing": "dyspnea",
    "breathless": "dyspnea",
    "shortness": "dyspnea",
    "dyspnea": "dyspnea",
    "dyspneic": "dyspnea",
    "ringing": "tinnitus",
    "tinnitus": "tinnitus",
    "hearing": "audiology",
    "audiology": "audiology",
    "audiogram": "audiology",
    "numb": "paresthesia",
    "numbness": "paresthesia",
    "tingling": "paresthesia",
    "paresthesia": "paresthesia",
    "paresthesias": "paresthesia",
    "dizzy": "vertigo",
    "dizziness": "vertigo",
    "lightheaded": "vertigo",
    "vertigo": "vertigo",
    "depressed": "depressive",
    "depression": "depressive",
    "depressive": "depressive",
    "anxiety": "anxious",
    "anxious": "anxious",
    "stomach": "gastric",
    "gastric": "gastric",
    "abdomen": "abdominal",
    "abdominal": "abdominal",
    "kidney": "renal",
    "renal": "renal",
    "heart": "cardiac",
    "cardiac": "cardiac",
    "sleep": "insomnia",
    "sleeping": "insomnia",
    "insomnia": "insomnia",
    "swelling": "edema",
    "swollen": "edema",
    "edema": "edema",
    "migraine": "headache",
    "migraines": "headache",
    "headaches": "headache",
}

# Suffix stripping, longest first so "ies"/"ing" win over a bare "s". Only used
# to add an extra token (never to replace one), so an over-eager stem can widen a
# match but can never lose one.
_STEM_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("ingly", ""),
    ("edly", ""),
    ("ing", ""),
    ("ied", "y"),
    ("ies", "y"),
    ("ed", ""),
    ("es", ""),
    ("s", ""),
)


def _stem(token: str) -> str:
    """Crude suffix stem: reports/reported/reporting -> report."""
    if len(token) <= 4:
        return token
    for suffix, replacement in _STEM_SUFFIXES:
        if not token.endswith(suffix):
            continue
        stem = token[: -len(suffix)]
        if len(stem) < 3:
            continue
        stem += replacement
        if suffix in ("ing", "ed") and len(stem) > 3 and stem[-1] == stem[-2]:
            stem = stem[:-1]  # running -> run, stopped -> stop
        return stem
    return token


@lru_cache(maxsize=10_000)
def _tokens(text: str) -> frozenset[str]:
    """Tokenize text into content words; LRU-cached (bounded at 10k entries)
    because facts and paragraphs are scored repeatedly during verification of
    many claims. Bounded LRU prevents unbounded memory growth on large record
    sets while retaining high hit rates for repeated phrases.

    Each token is emitted alongside its stem and its clinical synonym, so
    "reports neck pain" and "cervical pain reported" share tokens. Expansion is
    additive: scoring only ever gains overlap, never loses a literal match.
    """
    tokens: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token in _STOPWORDS or len(token) < 2:
            continue
        if len(token) == 2 and token not in _SHORT_CLINICAL_TOKENS:
            continue
        tokens.add(token)
        stem = _stem(token)
        if stem != token:
            tokens.add(stem)
        synonym = _CLINICAL_SYNONYMS.get(token) or _CLINICAL_SYNONYMS.get(stem)
        if synonym:
            tokens.add(synonym)
    return frozenset(tokens)


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
    "VA Hospital Richmond (facility)". A source that is a page citation
    ("clinic.pdf p.3-p.9") is not a provider and is skipped — otherwise the
    timeline's provider filter fills up with file names now that every fact
    carries a page citation instead of the old free-text source.
    """
    providers: set[str] = set()
    for event in events:
        source = event.source
        if _CITATION_SOURCE_RE.search(source):
            continue
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


def query_has_content_words(text: str) -> bool:
    """True when a query carries words worth matching (not only labels/stopwords).

    A claim with no content words ("c1", "item 2") gives retrieval nothing to
    judge, and an empty-evidence verdict must not be inferred from it.
    """
    return bool(_tokens(text))


@dataclass
class RetrievedEvidence:
    """Raw record excerpts for one query, with the signal needed to judge them.

    ``best_overlap`` is the honest half of the result: when nothing in the record
    set overlaps the query above the relevance floor, an absent record is a
    *coverage gap*, not a contradiction, and the caller must be able to tell
    which of the two it is holding before it reports a verdict to the user.
    """

    text: str = ""
    excerpts: int = 0
    best_overlap: float = 0.0
    corpus_size: int = 0

    @property
    def weak(self) -> bool:
        """True when nothing retrieved cleared ``config.EVIDENCE_WEAK_OVERLAP``."""
        return self.best_overlap < config.EVIDENCE_WEAK_OVERLAP


def _rank_paragraphs(
    documents: list[ExtractedDocument], query_tokens: set[str]
) -> tuple[list[tuple[float, str, str]], int]:
    """Rank paragraphs against query tokens; returns ``(ranked, corpus_size)``.

    Every paragraph is ranked, not just the ones clearing a threshold: ranking
    without gating guarantees the verification prompt receives the best raw
    context the record set can offer, and the threshold becomes a *signal* about
    that context (``RetrievedEvidence.weak``) rather than a silent filter.
    """
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
        return [], 0

    def weight(token: str) -> float:
        return math.log((total + 1) / (df.get(token, 0) + 1)) + 1.0

    query_weight = sum(weight(t) for t in query_tokens) or 1.0
    scored = [
        (sum(weight(t) for t in query_tokens & tokens) / query_weight, label, text)
        for label, text, tokens in corpus
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored, total


def retrieve_evidence(
    documents: list[ExtractedDocument],
    query: str,
    *,
    top_k: int = 8,
    excerpt_chars: int = 700,
) -> RetrievedEvidence:
    """Retrieve the raw record excerpts most relevant to a claim, ranked.

    Deterministic and dependency-free, so verification always has raw-source
    context rather than only the digest. Paragraph splitting is cached per
    document, so a very large record set is parsed once across all claim batches.
    """
    query_tokens = set(_tokens(query))
    if not query_tokens:
        return RetrievedEvidence()
    ranked, corpus_size = _rank_paragraphs(documents, query_tokens)
    seen: set[str] = set()
    unique: list[str] = []
    for _, label, text in ranked:
        excerpt = f"[{label}]\n{text[:excerpt_chars]}"
        key = excerpt[:120]
        if key in seen:
            continue
        seen.add(key)
        unique.append(excerpt)
        if len(unique) >= top_k:
            break
    return RetrievedEvidence(
        text="\n\n---\n\n".join(unique),
        excerpts=len(unique),
        best_overlap=ranked[0][0] if ranked else 0.0,
        corpus_size=corpus_size,
    )


def find_relevant_excerpts(
    documents: list[ExtractedDocument],
    query: str,
    *,
    top_k: int = 5,
    excerpt_chars: int = 700,
) -> str:
    """Ranked raw record excerpts as text (thin wrapper over ``retrieve_evidence``).

    Kept because existing callers and tests expect a plain string; new callers
    should use ``retrieve_evidence`` so they also see ``weak``/``best_overlap``.
    """
    return retrieve_evidence(
        documents, query, top_k=top_k, excerpt_chars=excerpt_chars
    ).text


# --------------------------------------------------------- timeline extraction
# F7.S1 (Medical Event Timeline Visualization, feature id
# 222efbdb-50be-4ff7-a384-1595d543c842): parse dates out of each `MedicalFact`
# via regex first, fall back to a small LLM inference pass for facts regex
# cannot date, group everything by date (or an 'undated' bucket), and detect
# gap periods (stretches with no records) across the dated events. Pure
# reuse of `MedicalFact`/`MedicalDigest` — no schema changes.

DIAGNOSTIC_FACT_TYPES = frozenset({"diagnosis", "test_result", "symptom", "provider_visit"})
TREATMENT_FACT_TYPES = frozenset({"treatment", "medication", "hospitalization"})

_MONTH_NAMES: dict[str, int] = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTH_ALTERNATION = "|".join(sorted(_MONTH_NAMES, key=len, reverse=True))

_ISO_DAY_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_ISO_MONTH_RE = re.compile(r"\b(\d{4})-(\d{1,2})\b")
_MONTH_YEAR_RE = re.compile(
    rf"\b({_MONTH_ALTERNATION})\.?\s+(\d{{4}})\b", re.IGNORECASE
)
_CIRCA_YEAR_RE = re.compile(
    r"\b(?:circa|c\.|around|approx\.?|approximately)\s*['’]?(\d{4})\b", re.IGNORECASE
)
_BARE_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

_TIMELINE_DEFAULT_GAP_DAYS = 180

_UNDATED_LLM_SYSTEM = (
    "You infer approximate dates for medical-record facts whose date field could not be "
    "parsed by pattern matching. For each fact, use its description and quote plus general "
    "context clues (referenced ages, seasons, nearby events) to infer the most likely "
    "calendar year, and month if evident. Never invent a specific day — return only a year "
    "('YYYY') or year-month ('YYYY-MM'). If there is truly no inferable date, return null for "
    "that fact. Output JSON only.\n\n" + GUARD_NOTE
)


# One pass, longest alternative first, so "2019-04-17" is taken as a day rather
# than also yielding a truncated "2019-04" month entry (the precision-specific
# patterns below each match a prefix of a longer date).
_ANY_DATE_RE = re.compile(
    rf"\b(?:\d{{4}}-\d{{1,2}}-\d{{1,2}}|\d{{4}}-\d{{1,2}}|(?:{_MONTH_ALTERNATION})\.?\s+\d{{4}}|"
    rf"\d{{1,2}}/\d{{1,2}}/\d{{2,4}}|(?:circa|approx\.?|approximately)\s*['’]?\d{{4}})\b",
    re.IGNORECASE,
)


def _dates_in_text(text: str, limit: int = 6) -> list[str]:
    """Distinct ISO dates printed in a page of text (at most ``limit``).

    Fed to the digest prompt as the dates that actually appear on this chunk's
    pages, so a fact's date is anchored to what the record prints rather than
    inferred from prose — the difference between "circa 2019" and 2019-04-17 in
    the timeline the statement is built from.
    """
    found: list[str] = []
    for match in _ANY_DATE_RE.finditer(text):
        parsed = _regex_extract_date(match.group(0))
        if parsed and parsed[0] not in found:
            found.append(parsed[0])
            if len(found) >= limit:
                return found
    return found


def _regex_extract_date(text: str) -> tuple[str, str] | None:
    """Best-effort (iso_date, precision) parsed from free text via regex only.

    Tries progressively looser patterns (exact day -> month -> named-month/year
    -> circa-year -> bare year) and returns the first successful match, or
    ``None`` when no date-like pattern is found at all.
    """
    if not text:
        return None
    match = _ISO_DAY_RE.search(text)
    if match:
        year, month, day = (int(v) for v in match.groups())
        try:
            return (datetime.date(year, month, day).isoformat(), "day")
        except ValueError:
            pass
    match = _ISO_MONTH_RE.search(text)
    if match:
        year, month = (int(v) for v in match.groups())
        try:
            return (datetime.date(year, month, 1).isoformat(), "month")
        except ValueError:
            pass
    match = _MONTH_YEAR_RE.search(text)
    if match:
        month_name, month_year = match.groups()
        named_month = _MONTH_NAMES.get(month_name.lower())
        if named_month:
            try:
                return (datetime.date(int(month_year), named_month, 1).isoformat(), "month")
            except ValueError:
                pass
    match = _CIRCA_YEAR_RE.search(text)
    if match:
        try:
            return (datetime.date(int(match.group(1)), 1, 1).isoformat(), "year")
        except ValueError:
            pass
    match = _BARE_YEAR_RE.search(text)
    if match:
        try:
            return (datetime.date(int(match.group(0)), 1, 1).isoformat(), "year")
        except ValueError:
            pass
    return None


def _categorize_fact_type(fact_type: str) -> str:
    """Map a raw `MedicalFact.type` value to a coarse timeline filter category."""
    normalized = (fact_type or "").strip().lower()
    if normalized in DIAGNOSTIC_FACT_TYPES:
        return "diagnostic"
    if normalized in TREATMENT_FACT_TYPES:
        return "treatment"
    return "other"


def _event_from_fact(fact: MedicalFact, *, iso_date: str | None, precision: str) -> dict[str, Any]:
    return {
        "date_iso": iso_date,
        "date_label": fact.date or "unknown",
        "precision": precision,
        "bucket": iso_date or "undated",
        "type": fact.type,
        "category": _categorize_fact_type(fact.type),
        "description": fact.description,
        "source": fact.source,
        "quote": fact.quote,
        "date_source": "regex" if iso_date else "none",
    }


def _infer_dates_once(
    llm: LLMService, batch: list[MedicalFact]
) -> list[Any] | None:
    """One inference call for a batch; ``None`` when the call itself failed."""
    items = [
        {"index": i, "description": f.description[:300], "quote": f.quote[:200]}
        for i, f in enumerate(batch)
    ]
    try:
        data = llm.chat_json(
            _UNDATED_LLM_SYSTEM,
            "Infer dates for these facts. Return JSON exactly as: "
            '{"dates": [{"index": <int>, "date": "YYYY" | "YYYY-MM" | null}]}\n\n'
            + sanitize_for_prompt(json.dumps(items), max_chars=DERIVED_TEXT_MAX_CHARS),
            model=llm.fast_model,
            # Budget per fact, not a flat cap: a fixed 2000-token ceiling over a
            # few hundred undated facts truncates the JSON mid-array and loses the
            # entire batch.
            max_tokens=max(2000, 60 * len(batch)),
            phase="timeline:llm_date_extraction",
        )
    except Exception as exc:  # noqa: BLE001 - fallback pass must never break extraction
        logger.warning(
            "timeline date inference call failed error=%s",
            f"{type(exc).__name__}: {exc}",
            extra={
                "request_id": get_request_id() or "-",
                "phase": "timeline:llm_date_extraction",
                "status": "error",
                "error_class": type(exc).__name__,
            },
        )
        return None
    rows = data.get("dates") if isinstance(data, dict) else None
    return rows if isinstance(rows, list) else None


def _llm_infer_undated(
    llm: LLMService, facts: list[MedicalFact]
) -> dict[int, tuple[str, str] | None]:
    """Best-effort LLM date inference for facts regex could not date.

    Returns a mapping of the input list's index to a parsed (iso_date,
    precision) tuple, or ``None`` when the model also could not infer a date.
    Never raises — a failed call simply yields no inferences for that batch and
    those facts stay in the 'undated' bucket.

    Batched, and tolerant per row: one call over every undated fact in a large
    bundle both overflows the output budget (losing the whole batch, and with it
    the chronology the statement depends on) and lets a single malformed row cost
    every other date. Each batch is retried once; rows are parsed individually.
    """
    if not facts:
        return {}
    batch_size = max(1, config.UNDATED_FACT_BATCH_SIZE)
    inferred: dict[int, tuple[str, str] | None] = {}
    for start in range(0, len(facts), batch_size):
        batch = facts[start : start + batch_size]
        for _attempt in (1, 2):
            rows = _infer_dates_once(llm, batch)
            if rows is not None:
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    index = row.get("index")
                    if not isinstance(index, int) or not 0 <= index < len(batch):
                        continue
                    raw_date = row.get("date")
                    inferred[start + index] = (
                        _regex_extract_date(str(raw_date)) if raw_date else None
                    )
                break
        else:
            logger.warning(
                "timeline date inference lost a batch facts=%d start=%d",
                len(batch),
                start,
                extra={
                    "request_id": get_request_id() or "-",
                    "phase": "timeline:llm_date_extraction",
                    "status": "dropped",
                    "facts": len(batch),
                },
            )
    return inferred


def _detect_timeline_gaps(
    dated_iso: list[str], *, threshold_days: int = _TIMELINE_DEFAULT_GAP_DAYS
) -> list[dict[str, Any]]:
    """Identify date ranges with no records among the already-dated events.

    A "gap" is any stretch between two consecutive distinct dated events that
    exceeds ``threshold_days``. Returns an empty list when fewer than two
    distinct dates are present (nothing to compare).
    """
    unique_sorted = sorted({d for d in dated_iso if d})
    if len(unique_sorted) < 2:
        return []
    gaps: list[dict[str, Any]] = []
    for previous, current in zip(unique_sorted, unique_sorted[1:]):
        previous_date = datetime.date.fromisoformat(previous)
        current_date = datetime.date.fromisoformat(current)
        span_days = (current_date - previous_date).days
        if span_days > threshold_days:
            gaps.append({"start": previous, "end": current, "days": span_days})
    return gaps


def _empty_timeline_data(request_id: str, *, error: bool = False) -> dict[str, Any]:
    data: dict[str, Any] = {
        "events": [],
        "grouped": {},
        "gaps": [],
        "dated_count": 0,
        "undated_count": 0,
        "gap_count": 0,
        "request_id": request_id,
    }
    if error:
        data["error"] = True
    return data


def build_timeline_data(
    digest: MedicalDigest,
    llm: LLMService | None = None,
    *,
    feature_id: str | None = None,
) -> dict[str, Any]:
    """Build the vertical-timeline dataset from a reviewed `MedicalDigest`.

    Groups every fact from the digest by its best-effort parsed date, falling
    back to an ``'undated'`` bucket when no date can be recovered via regex
    or (if an ``llm`` client is supplied) a small LLM inference pass. Runs
    gap detection over the dated events afterward. Only facts belonging to
    the supplied digest are processed — callers must pass the digest that
    belongs to the *current* run/request id (see
    `app/views/evaluate_view.py::_render_medical_timeline`, which keys the
    cached result by the current run's request id so a stale digest from a
    previous run is never reused).

    Never raises: any failure is tracked via telemetry and this returns a
    well-formed, empty timeline so the UI can render a friendly empty state
    instead of crashing the Evaluate tab. `feature_id` is accepted as a
    parameter (never hardcoded here — this module is shared pipeline logic)
    so the caller controls which feature the telemetry event is attributed
    to.
    """
    request_id = get_request_id() or "-"
    try:
        events: list[dict[str, Any]] = []
        undated_indices: list[int] = []
        undated_facts: list[MedicalFact] = []

        for fact in digest.facts:
            parsed = _regex_extract_date(fact.date) or _regex_extract_date(fact.quote)
            if parsed is None:
                undated_indices.append(len(events))
                undated_facts.append(fact)
                events.append(_event_from_fact(fact, iso_date=None, precision="none"))
            else:
                iso_date, precision = parsed
                events.append(_event_from_fact(fact, iso_date=iso_date, precision=precision))

        if undated_facts and llm is not None:
            inferred = _llm_infer_undated(llm, undated_facts)
            for local_index, event_index in enumerate(undated_indices):
                result = inferred.get(local_index)
                if result is None:
                    continue
                iso_date, precision = result
                event = events[event_index]
                event["date_iso"] = iso_date
                event["precision"] = precision
                event["bucket"] = iso_date
                event["date_source"] = "llm"

        grouped: dict[str, list[dict[str, Any]]] = {}
        for event in events:
            grouped.setdefault(event["bucket"], []).append(event)

        dated_iso = [e["date_iso"] for e in events if e["date_iso"]]
        gaps = _detect_timeline_gaps(dated_iso)

        dated_count = len(dated_iso)
        undated_count = len(events) - dated_count

        timeline_data: dict[str, Any] = {
            "events": events,
            "grouped": grouped,
            "gaps": gaps,
            "dated_count": dated_count,
            "undated_count": undated_count,
            "gap_count": len(gaps),
            "request_id": request_id,
        }

        logger.info(
            "timeline data built dated=%d undated=%d gaps=%d",
            dated_count,
            undated_count,
            len(gaps),
            extra={
                "request_id": request_id,
                "phase": "timeline:build",
                "status": "ok",
                "dated_count": dated_count,
                "undated_count": undated_count,
                "gap_count": len(gaps),
            },
        )

        if feature_id:
            try:
                track_goal(
                    feature_id,
                    "timeline data extracted",
                    dated_event_count=dated_count,
                    undated_count=undated_count,
                    gap_count=len(gaps),
                )
            except Exception:  # noqa: BLE001 - telemetry must never break the pipeline
                pass

        return timeline_data
    except Exception as exc:  # noqa: BLE001 - extraction must degrade gracefully, never crash
        logger.error(
            "timeline data extraction failed error=%s",
            exc,
            exc_info=True,
            extra={"request_id": request_id, "phase": "timeline:build", "status": "error"},
        )
        if feature_id:
            try:
                track_feature_error(
                    feature_id, exc, phase="build_timeline_data", error_type=type(exc).__name__
                )
            except Exception:  # noqa: BLE001 - telemetry must never break the pipeline
                pass
        return _empty_timeline_data(request_id, error=True)
