"""Bind drafting's record rows to retained facts and complete uploaded excerpts.

This is provenance validation, not a semantic check of a medical description or
of whether a witness observation is corroborated by the cited passage.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .documents import DocumentPage, ExtractedDocument
from .llm import LLMParseError
from .medical_review import MedicalDigest, _normalize_date_for_sort, quote_matches_source
from .prompt_sanitize import sanitize_for_prompt
from .source_validation import build_source_index, source_reference_key

GROUNDING_SOURCE_POLICY = "retained_fact_full_quote_source_unit_v1"
LEGACY_GROUNDING_SOURCE_NOTICE = (
    "This saved analysis has no recognized drafting source validation. "
    "Re-run with the original records before relying on record support or conflicts."
)
_RECORD_FIELDS = {
    "supported_observations": "record_support",
    "conflicts": "record_fact",
    "suggested_inclusions": "fact",
}


def _text_key(text: str) -> str:
    return " ".join(text.casefold().split())


def grounding_catalog(
    digest: MedicalDigest, records: list[ExtractedDocument], query: str, *,
    max_facts: int = 150, budget_chars: int = 90_000,
) -> tuple[dict[str, dict[str, Any]], str]:
    """Present only complete, resolvable facts; omitted facts stay in the digest.

    IDs bind an immutable fact snapshot to one typed uploaded address. Prompt
    escaping that changes a field makes that fact unsuitable for verbatim copying;
    exclude it rather than bless an altered excerpt. Budgets drop whole entries.
    """
    index = build_source_index(records)
    catalog: dict[str, dict[str, Any]] = {}
    used = 2  # JSON array brackets
    for fact in digest.ranked_facts(query):
        if len(catalog) >= max(0, max_facts):
            break
        unit = index.get(source_reference_key(fact.source))
        if unit is None or not fact.description.strip() or not quote_matches_source(fact.quote, unit.text):
            continue
        if (fact.document and fact.document != unit.filename) or (fact.page and fact.page != unit.page):
            continue
        address = {"filename": unit.filename, "kind": unit.kind, "number": unit.page}
        fingerprint = json.dumps([address, vars(fact)], sort_keys=True, ensure_ascii=True)
        fact_id = "fact-" + hashlib.sha256(fingerprint.encode()).hexdigest()
        entry = {
            "fact_id": fact_id, "source": unit.label, "source_unit": address,
            "description": fact.description, "quote": fact.quote,
            "date": fact.date, "type": fact.type,
        }
        serialized = json.dumps(entry, ensure_ascii=False)
        if sanitize_for_prompt(serialized, max_chars=2 * len(serialized) + 1) != serialized:
            continue
        if fact_id in catalog or used + len(serialized) + 2 > max(0, budget_chars):
            continue
        catalog[fact_id] = entry
        used += len(serialized) + 2
    ordered = sorted(catalog.values(), key=lambda entry: _normalize_date_for_sort(entry["date"]))
    return catalog, json.dumps(ordered, ensure_ascii=False)


def validate_grounding_sources(
    grounding: dict[str, Any], catalog: dict[str, dict[str, Any]],
    source_index: dict[str, DocumentPage | None],
) -> dict[str, Any]:
    """Fail before generation on any fabricated or altered record provenance.

    Witness-only rows need no citation. Error text never echoes private fields.
    Canonical fields come from the trusted catalog, never from model prose.
    """
    for section, description_field in _RECORD_FIELDS.items():
        for row in grounding[section]:
            fact_id = row.get("fact_id")
            entry = catalog.get(fact_id) if isinstance(fact_id, str) else None
            source = row.get("source")
            quote = row.get("quote")
            address = row.get("source_unit")
            unit = source_index.get(source_reference_key(source)) if isinstance(source, str) else None
            if (
                entry is None or unit is None or not isinstance(quote, str)
                or not isinstance(address, dict)
                or type(address.get("number")) is not int
                or address != entry["source_unit"] or unit.label != entry["source"]
                or _text_key(quote) != _text_key(entry["quote"])
                or _text_key(row[description_field]) != _text_key(entry["description"])
                or not quote_matches_source(quote, unit.text)
            ):
                raise LLMParseError(
                    "Grounding record evidence must copy a supplied fact ID, description, "
                    "typed source address and complete quote from one readable, unambiguous uploaded unit."
                )
            row.update(fact_id=entry["fact_id"], source=entry["source"],
                       source_unit=dict(entry["source_unit"]), quote=entry["quote"])
            row[description_field] = entry["description"]
    return grounding
