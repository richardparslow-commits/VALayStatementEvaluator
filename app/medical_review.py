"""Exhaustive medical-record review: chunked extraction -> merged facts digest.

Both pathways (evaluate and draft) rely on this module to build a structured,
citable digest of every uploaded medical document.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from .documents import ExtractedDocument, chunk_page_labelled_text
from .llm import LLMClient

ProgressCallback = Callable[[float, str], None]

DIGEST_SYSTEM = """You are a meticulous medical-records analyst supporting the review of \
VA disability claims. Your job is to extract EVERY medically or factually relevant detail \
from a chunk of medical records. You must be exhaustive and precise; downstream legal \
analysis depends on you capturing dates, names, facilities, diagnoses, symptoms, treatments, \
medications, test results, and provider statements verbatim-faithfully. Never invent facts. \
If the chunk is administrative or contains no medical content, say so with an empty fact list."""

DIGEST_USER_TEMPLATE = """Extract all medically and factually relevant information from this \
medical-record chunk ({label}).

Return JSON with this exact shape:
{{
  "facts": [
    {{
      "date": "YYYY-MM or YYYY-MM-DD or approximate (e.g., 'circa 2019', 'unknown')",
      "type": "diagnosis | symptom | treatment | medication | test_result | hospitalization | provider_visit | in_service_event | administrative | other",
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

CHUNK TEXT:
<<<
{chunk_text}
>>>"""

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

    def as_json_text(self, max_facts: int = 400) -> str:
        return json.dumps(
            {
                "facts": [vars(f) for f in self.facts[:max_facts]],
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


def review_medical_records(
    llm: LLMClient,
    documents: list[ExtractedDocument],
    progress: ProgressCallback | None = None,
) -> MedicalDigest:
    """Run the full exhaustive review over all uploaded records."""
    if not documents:
        raise ValueError("No medical records provided.")

    full_text = "\n\n".join(doc.page_labelled_text() for doc in documents)
    pages = sum(len(doc.pages) for doc in documents)
    chunks = chunk_page_labelled_text(full_text)

    if progress:
        progress(0.05, f"Reviewing {pages} pages in {len(chunks)} chunk(s)…")

    all_facts: list[MedicalFact] = []
    conditions: Counter[str] = Counter()
    providers: Counter[str] = Counter()

    for chunk in chunks:
        if progress:
            progress(
                0.05 + 0.55 * (chunk.index - 1) / len(chunks),
                f"Extracting facts from {chunk.label}…",
            )
        data = llm.chat_json(
            DIGEST_SYSTEM,
            DIGEST_USER_TEMPLATE.format(
                label=chunk.label,
                source_hint=chunk.label,
                chunk_text=chunk.text,
            ),
            model=llm._settings.model_fast,
        )
        for raw in data.get("facts", []):
            all_facts.append(
                MedicalFact(
                    date=str(raw.get("date", "unknown")),
                    type=str(raw.get("type", "other")),
                    description=str(raw.get("description", "")),
                    source=str(raw.get("source", chunk.label)),
                    quote=str(raw.get("quote", "")),
                )
            )
        for name in data.get("conditions_mentioned", []):
            conditions[str(name).strip()] += 1
        for name in data.get("providers_and_facilities", []):
            providers[str(name).strip()] += 1


    if progress:
        progress(0.65, "Consolidating extracted facts…")

    digest = MedicalDigest(
        facts=all_facts,
        conditions=[c for c, _ in conditions.most_common(40) if c],
        providers=[p for p, _ in providers.most_common(40) if p],
        pages_reviewed=pages,
        chunks_reviewed=len(chunks),
    )

    if len(all_facts) > 150:
        digest.facts = _merge_facts(llm, digest)

    digest.summary = _summarize(llm, digest)
    if progress:
        progress(0.8, f"Record review complete: {len(digest.facts)} facts extracted.")
    return digest


def _merge_facts(llm: LLMClient, digest: MedicalDigest) -> list[MedicalFact]:
    """Deduplicate large fact lists via the fast model."""
    data = llm.chat_json(
        MERGE_SYSTEM,
        "Deduplicate and consolidate these extracted medical facts. Keep every DISTINCT fact "
        "with its source. Return JSON: {\"facts\": [{\"date\",\"type\",\"description\",\"source\",\"quote\"}]}\n\n"
        + json.dumps([vars(f) for f in digest.facts]),
        model=llm._settings.model_fast,
    )
    merged = [
        MedicalFact(
            date=str(raw.get("date", "unknown")),
            type=str(raw.get("type", "other")),
            description=str(raw.get("description", "")),
            source=str(raw.get("source", "")),
            quote=str(raw.get("quote", "")),
        )
        for raw in data.get("facts", [])
    ]
    return merged or digest.facts


def _summarize(llm: LLMClient, digest: MedicalDigest) -> str:
    return llm.chat(
        "You are a medical-records analyst. Write a concise narrative summary (max 250 words) "
        "of the record set: key diagnoses, treatment history, notable events, and current "
        "status. Plain text only.",
        "Extracted facts:\n" + digest.timeline_text()[:12000],
    )


# ----------------------------------------------------------- relevance search
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for", "with", "by",
    "from", "is", "was", "were", "are", "been", "he", "she", "his", "her", "him", "i",
    "my", "me", "that", "this", "it", "as", "has", "have", "had", "be", "will", "since",
    "during", "about", "into", "their", "they", "them", "you", "your", "we", "our",
}


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]{3,}", text.lower())
        if token not in _STOPWORDS
    ]


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
    """
    query_tokens = set(_tokens(query))
    if not query_tokens:
        return ""

    scored: list[tuple[float, str]] = []
    for doc in documents:
        for page in doc.pages:
            paragraphs = [p for p in re.split(r"\n{2,}", page.text) if len(p.strip()) > 40]
            for paragraph in paragraphs:
                tokens = set(_tokens(paragraph))
                if not tokens:
                    continue
                overlap = len(query_tokens & tokens) / len(query_tokens)
                if overlap >= 0.2:
                    scored.append((overlap, f"[{page.label}]\n{paragraph[:excerpt_chars]}"))

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
