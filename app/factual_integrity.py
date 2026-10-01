"""Conservative factual comparison against original inputs, never model truth scoring.

Literal matches provide source links, not semantic proof. Other sentences need
human source selection. Critical changes cannot be waived by a review checkbox.
Human approval is session-only and bound to the exact text and source context.
"""
from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_left, bisect_right
from itertools import islice
from collections.abc import Iterator
from typing import Any

from .documents import DocumentPage, ExtractedDocument
from . import config
from .grounding_sources import grounding_catalog
from .medical_review import MedicalDigest

FACTUAL_POLICY = "original_account_source_comparison_v1"
MAX_SPANS = 4000
MAX_OUTPUT_CHARS = 120_000
_WORDS = re.compile(r"[\w]+(?:['’][\w]+)?", re.UNICODE)
_NUMBERS = re.compile(r"(?<!\w)-?\d+(?:[.,]\d+)*(?!\w)")
_NUMBER_WORDS = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million".split(),
    "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 30 40 50 60 70 80 90 100 1000 1000000".split(),
))
_HEADINGS = {
    "header", "introduction", "introduction & credentials of observation",
    "in-service event / onset", "observed symptoms and their progression",
    "observed symptoms", "functional impact", "continuity", "continuity statement",
    "closing", "certification", "closing & certification", "signature",
}
_CERTIFICATION = "i certify that this statement is true and correct to the best of my knowledge and belief"
_RECORD_ATTRIBUTION = re.compile(r"\b(?:records? (?:show|state|report|document|note)|according to (?:the )?(?:medical )?records?|(?:medical|clinical) (?:record|note|report))\b", re.I)
_GRAMMAR = set("i me my we us our he she him her his they them their it its the a an this that these those and or but as is are was were be been being am have has had do does did to of from in on at for with by after before since during until which who whom what when where than then so also can will would should medical clinical record records show shows state states note notes according".split())
_ALIASES = {word: "observe" for word in ("observed", "observe", "observing", "saw", "seen", "see", "watched")}
_ALIASES.update({word: "report" for word in ("reported", "reports", "report")})
_FIELD_LABELS = {"name": "Witness name", "veteran_name": "Veteran name", "relationship": "Witness relationship",
                 "known_since": "Known the veteran since / for", "contact_frequency": "Opportunity to observe"}


def _content_word(word: str) -> str:
    if word in _ALIASES:
        return _ALIASES[word]
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("s"):
        return word[:-1]
    return word


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _key(text: str) -> str:
    return " ".join(_WORDS.findall(text.casefold()))


def _spans(text: str) -> Iterator[dict[str, Any]]:
    for match in re.finditer(r".+?(?:[.!?](?=\s|$)|\n+|$)", text, re.S):
        raw = match.group()
        clean = raw.strip()
        if clean:
            start = match.start() + len(raw) - len(raw.lstrip())
            yield {"start": start, "end": start + len(clean), "text": clean}


def _number_associations(text: str) -> list[str]:
    """Bind each occurrence to nearby factual wording, not just a number set."""
    lowered = text.casefold().replace("’", "'")
    tokens = list(_WORDS.finditer(lowered))
    meaningful = [(m.start(), m.end(), _content_word(m.group())) for m in tokens
                  if m.group() not in _GRAMMAR | {"known", "know", "name"}
                  and m.group() not in _NUMBER_WORDS and not m.group().isdecimal()]
    occurrences = [(m.start(), m.end(), m.group()) for m in _NUMBERS.finditer(lowered)]
    occurrences += [(m.start(), m.end(), _NUMBER_WORDS[m.group()]) for m in tokens if m.group() in _NUMBER_WORDS]
    result = []
    counts: dict[str, int] = {}
    starts = [start for start, _, _ in meaningful]
    ends = [end for _, end, _ in meaningful]
    boundaries = list(re.finditer(r"[;!?]|\b(?:and|or|but)\b|(?<!\d)[,.]|[,.](?!\d)", lowered))
    boundary_starts = [m.start() for m in boundaries]
    boundary_ends = [m.end() for m in boundaries]
    for start, end, number in sorted(occurrences):
        prior, following = bisect_right(ends, start), bisect_left(starts, end)
        left, right = bisect_right(boundary_ends, start), bisect_left(boundary_starts, end)
        clause_start = boundary_ends[left - 1] if left else 0
        clause_end = boundary_starts[right] if right < len(boundaries) else len(lowered)
        first, last = bisect_left(starts, clause_start), bisect_right(ends, clause_end)
        before = [word for _, _, word in meaningful[max(first, prior - 2):prior]]
        after = [word for _, _, word in meaningful[following:min(last, following + 2)]]
        binding = fingerprint([before, number, after])
        counts[binding] = counts.get(binding, 0) + 1
        result.append(f"{binding}:{counts[binding]}")
    return sorted(result)


def features(text: str) -> dict[str, list[str]]:
    words = set(_WORDS.findall(text.casefold().replace("’", "'")))
    numbers = set(_NUMBERS.findall(text)) | {_NUMBER_WORDS[w] for w in words if w in _NUMBER_WORDS}
    return {
        # A deliberately conservative lexical screen catches novel diagnoses
        # and other changed content even when no medical vocabulary is known.
        # It can flag legitimate paraphrases; humans correct the output or
        # update the witness's original account rather than waiving new facts.
        "factual wording": sorted({_content_word(w) for w in words - _GRAMMAR
                                   if w not in _NUMBER_WORDS and not w.isdecimal()}),
        "dates/numbers": sorted(numbers),
        "number-to-claim associations": _number_associations(text),
        "calendar dates": sorted(words & {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}),
        "chronology": sorted(words & {"before", "after", "since", "during", "until", "earlier", "later", "prior", "following"}),
        "speaker": sorted((({"first person"} if words & {"i", "me", "my", "we", "our", "us"} else set()) |
                           ({"third person"} if words & {"he", "she", "him", "her", "his", "they", "them", "their"} else set()))),
        "negation": ["negated"] if words & {"no", "not", "never", "denies", "denied", "without", "cannot", "can't", "didn't", "isn't", "wasn't", "doesn't", "hasn't", "haven't", "couldn't", "won't", "unable", "negative"} else [],
        "date precision": ["approximate"] if words & {"around", "about", "approximately", "circa", "roughly"} else [],
        "uncertainty": ["uncertain"] if words & {"maybe", "possibly", "possible", "might", "could", "uncertain", "unsure", "believe", "think", "suspect"} else [],
        "laterality": sorted(words & {"left", "right", "bilateral", "both"}),
        "frequency": sorted(words & {"daily", "weekly", "monthly", "hourly", "occasionally", "sometimes", "constantly", "always", "every"}),
        "attribution": sorted((({"reported"} if words & {"told", "reported", "reports", "said", "says"} else set()) |
                               ({"observed"} if words & {"observed", "saw", "seen", "watched", "personally"} else set()))),
        "diagnosis/nexus": sorted((({"diagnosis"} if words & {"diagnosed", "diagnosis", "diagnoses"} else set()) |
                                   ({"causation"} if re.search(r"\b(?:caused by|due to|service[- ]connected|nexus)\b", text, re.I) else set()))),
    }


def build_context(account: str, witness: dict[str, str], digest: MedicalDigest | None,
                  records: list[ExtractedDocument]) -> dict[str, Any]:
    def source_spans() -> Iterator[dict[str, Any]]:
        for index, span in enumerate(_spans(account), 1):
            yield {**span, "kind": "witness_account", "label": f"Original witness account — passage {index}"}
        for name, value in sorted(witness.items()):
            # Claim labels describe the requested work, not observed evidence.
            # Retain them in the context fingerprint but never offer them as
            # support, including in a selection mixed with witness passages.
            if name not in {"Claimed condition", "Claim type"} and value.strip():
                for index, span in enumerate(_spans(value), 1):
                    yield {**span, "kind": "witness_field", "field": name,
                           "label": f"{_FIELD_LABELS.get(name, name)} — passage {index}"}
        if digest:
            catalog, _ = grounding_catalog(digest, records, "", max_facts=len(digest.facts), budget_chars=4_000_000)
            for entry in catalog.values():
                # Raw quotes, never model descriptions or inferred dates.
                quote_hash = fingerprint(entry["quote"])
                for span in _spans(entry["quote"]):
                    yield {**span, "kind": "record_quote", "label": entry["source"],
                           "quote_hash": quote_hash, "fact_id": entry["fact_id"], "source_unit": entry["source_unit"]}
    sources = list(islice(source_spans(), MAX_SPANS + 1))
    complete = len(sources) <= MAX_SPANS
    sources = sources[:MAX_SPANS]
    for source in sources:
        source["id"] = "source-" + fingerprint(source)
        source["features"] = features(source["text"])
    pages = [[doc.filename, doc.pagination, doc.unreadable_pages,
              [[p.page, p.kind, p.text] for p in doc.pages]] for doc in records]
    return {"policy": FACTUAL_POLICY, "sources": sources, "complete": complete,
            "hash": fingerprint([account, witness, pages, sources, complete])}


def retained_inputs(account: str, witness: dict[str, str], records: list[ExtractedDocument]) -> dict[str, Any]:
    """Preserve unavailable source addresses as well as readable page text."""
    unavailable = [{"filename": doc.filename, "kind": doc.pagination, "page": number}
                   for doc in records for number in doc.unreadable_pages]
    return {"policy": FACTUAL_POLICY, "account": account, "witness": dict(witness),
            "unreadable_units": unavailable}


def context_for_result(result: Any) -> dict[str, Any] | None:
    """Rebuild saved contexts; never trust a serialized successful comparison."""
    raw = getattr(result, "factual_inputs", None)
    if not isinstance(raw, dict) or raw.get("policy") != FACTUAL_POLICY:
        return None
    account, witness = raw.get("account"), raw.get("witness")
    unreadable = raw.get("unreadable_units")
    evidence = getattr(result, "evidence_source", None)
    if (not isinstance(account, str) or not account.strip() or len(account) > MAX_OUTPUT_CHARS
            or not isinstance(witness, dict) or len(witness) > 66
            or any(not isinstance(k, str) or not isinstance(v, str) or len(v) > 4000 for k, v in witness.items())
            or not isinstance(unreadable, list) or len(unreadable) > config.MAX_RECORD_PAGES
            or not isinstance(evidence, list) or not evidence):
        return None
    records = []
    for page in evidence:
        if (not isinstance(page, dict) or not isinstance(page.get("filename"), str)
                or not isinstance(page.get("text"), str) or type(page.get("page")) is not int
                or page["page"] < 1 or page.get("kind") not in ("page", "block")):
            return None
        # Recombine pages by filename so the shared source index detects
        # ambiguous addresses instead of giving a fabricated quote a binding.
        records.append(ExtractedDocument(page["filename"], [DocumentPage(page["filename"], page["page"], page["text"], page["kind"])], pagination=page["kind"]))
    for unit in unreadable:
        if (not isinstance(unit, dict) or not isinstance(unit.get("filename"), str)
                or unit.get("kind") not in ("page", "block") or type(unit.get("page")) is not int
                or unit["page"] < 1):
            return None
        records.append(ExtractedDocument(unit["filename"], unreadable_pages=[unit["page"]], pagination=unit["kind"]))
    context = build_context(account, witness, getattr(result, "digest", None), records)
    # Hash complete source snapshots, including unresolved facts, so changing
    # digest evidence cannot leave an old session approval applicable.
    digest = getattr(result, "digest", None)
    context["hash"] = fingerprint([context["hash"], [vars(f) for f in digest.facts] if digest else None])
    return context


def _structural(text: str) -> bool:
    label = text.strip().strip("#*: ").casefold().rstrip(".")
    return label in _HEADINGS or _key(text) == _CERTIFICATION


def compare(text: str, context: dict[str, Any] | None,
            links: dict[str, list[str]] | None = None, *, require_account_coverage: bool = True) -> dict[str, Any]:
    """Whole-output comparison. Suggestions alone cannot resolve any assertion."""
    context_hash = context["hash"] if context else ""
    result: dict[str, Any] = {"policy": FACTUAL_POLICY, "text_hash": fingerprint(text) if isinstance(text, str) else "",
                              "context_hash": context_hash, "status": "blocked", "rows": [], "issues": [],
                              "ledger": context["sources"] if context else []}
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_OUTPUT_CHARS:
        result["issues"] = ["The complete output must be nonblank text within the review limit."]
        return result
    if context is None or not context.get("complete"):
        result["issues"] = ["Original inputs are unavailable or exceed the comparison limit. Re-run with the original inputs before marking this text reviewed."]
        return result
    sources = {s["id"]: s for s in context["sources"]}
    source_keys = {sid: _key(s["text"]) for sid, s in sources.items()}
    exact_index: dict[str, list[str]] = {}
    inverted: dict[str, set[str]] = {}
    for sid, key in source_keys.items():
        exact_index.setdefault(key, []).append(sid)
        for word in set(key.split()):
            inverted.setdefault(word, set()).add(sid)
    used: set[str] = set()
    for ordinal, span in enumerate(_spans(text)):
        if ordinal >= MAX_SPANS:
            result["issues"].append("Output exceeds the sentence comparison limit. Shorten and re-run.")
            break
        sentence, key = span["text"], _key(span["text"])
        if _structural(sentence):
            continue
        row_id = fingerprint([span["start"], span["end"], sentence])
        exact = exact_index.get(key, [])
        if not exact and len(key.split()) >= 4:
            # Every word must be present before testing contiguous containment.
            # Exact matches use a direct lookup and never scan the whole ledger.
            matches = [inverted.get(word, set()) for word in set(key.split())]
            possible = set.intersection(*matches) if matches else set()
            exact = [sid for sid in sorted(possible) if " " + key + " " in " " + source_keys[sid] + " "]
        if exact:
            candidates = exact[:3]
        else:
            scores: dict[str, int] = {}
            for word in set(key.split()):
                for sid in inverted.get(word, ()):
                    scores[sid] = scores.get(sid, 0) + 1
            candidates = sorted(scores, key=lambda sid: (-scores[sid], sid))[:3]
        default_source = next((sid for sid in exact if sid not in used), exact[0] if exact else None)
        chosen = (links or {}).get(row_id, [default_source] if default_source else [])
        reasons = []
        if not isinstance(chosen, list) or len(chosen) > 3 or any(sid not in sources for sid in chosen):
            chosen = []; reasons.append("Invalid source selection.")
        if not chosen:
            reasons.append("No source is linked; select and review supporting original evidence.")
        selected = [sources[sid] for sid in chosen]
        # Candidate comparisons flag critical changes immediately even before
        # the user supplies a link. A lexical suggestion is never accepted proof.
        baseline = selected or [sources[sid] for sid in candidates[:1]]
        actual = features(sentence)
        for name, values in actual.items():
            expected = {v for s in baseline for v in s["features"][name]}
            missing = expected - set(values)
            added = set(values) - expected
            fields = {s.get("field") for s in baseline if s["kind"] == "witness_field"}
            # Identity/opportunity fields supply structured meaning even when
            # their value is just "Alex", "2010" or "weekly". Permit those
            # narrow wrappers, never diagnoses, new numbers or missing content.
            if fields:
                if (name == "speaker" and all(s["kind"] == "witness_field" for s in baseline)
                        and fields <= {"name", "relationship", "known_since", "contact_frequency"}):
                    added = set()
                if name == "chronology" and "known_since" in fields:
                    added -= {"since"}
                if name == "attribution" and "contact_frequency" in fields:
                    added -= {"observed"}
                if name == "factual wording":
                    if "name" in fields:
                        added -= {"name"}
                    if "known_since" in fields:
                        added -= {"know", "known"}
                    if "contact_frequency" in fields:
                        added -= {"observe"}
            if added or missing:
                reasons.append(f"Changed or unsupported {name}; compare with the original passage.")
        if any(s["kind"] == "record_quote" for s in selected) and not _RECORD_ATTRIBUTION.search(sentence):
            reasons.append("Record evidence must remain attributed to the records; it cannot become a firsthand witness account.")
        if re.search(r"\[[^\[\]]+\]", sentence):
            reasons.append("Resolve the placeholder with the witness before review approval.")
        if not reasons:
            used.update(chosen)
        result["rows"].append({**span, "id": row_id, "sources": chosen, "candidates": candidates,
                               "features": actual, "issues": reasons})
    for index, (sid, source) in enumerate(sources.items(), 1):
        if (require_account_coverage and source["kind"] == "witness_account"
                and not _structural(source["text"]) and sid not in used):
            result["issues"].append(f"Original witness passage {index} is not preserved or linked to a supported sentence.")
    if not result["rows"]:
        result["issues"].append("No factual sentences are available for source and witness review.")
    if not result["issues"] and all(not row["issues"] for row in result["rows"]):
        result["status"] = "review_required"
    return result


def attach_review(result: Any, text: str) -> None:
    context = context_for_result(result)
    result.factual_review = compare(text, context)
    edits = []
    for i, change in enumerate(getattr(result, "revision_changes", []), 1):
        if isinstance(change, dict) and isinstance(change.get("revised"), str):
            comparison = compare(change["revised"], context, require_account_coverage=False)
            # One shared ledger per result, not one full copy per proposed edit.
            comparison.pop("ledger")
            edits.append({"index": i, "comparison": comparison})
    result.factual_review["proposed_edits"] = edits


def review_notice(review: dict[str, Any] | None) -> str:
    if not review or review.get("policy") != FACTUAL_POLICY:
        return "Factual comparison unavailable for this saved output. Re-run with the original inputs; this text is unreviewed."
    count = len(review.get("issues", [])) + sum(bool(row.get("issues")) for row in review.get("rows", []))
    return (f"Factual comparison: {count} unresolved review item(s). Compare every sentence with the original account and source quotations. "
            "Model self-review is not source or witness approval. This text is unreviewed.")


def review_markdown(review: dict[str, Any] | None) -> str:
    lines = ["## Factual comparison — unreviewed output", "", review_notice(review), ""]
    if review:
        ledger = {s["id"]: s for s in review.get("ledger", [])}
        for row in review.get("rows", []):
            lines.append(f"Sentence at characters {row['start']}–{row['end']}; sources: {', '.join(row['sources']) or 'none'}")
            lines.extend("    " + line for line in row["text"].splitlines())
            lines.extend("- " + reason for reason in row["issues"])
            for sid in row["sources"] or row["candidates"]:
                source = ledger.get(sid)
                if source:
                    lines.append("Selected evidence:" if sid in row["sources"] else "Comparison candidate; not accepted support:")
                    lines.extend("    " + line for line in (source["label"] + "\n" + source["text"][:1000]).splitlines())
                    if len(source["text"]) > 1000:
                        lines.append("    Display excerpt shortened; consult the complete original source.")
            lines.append("")
        lines.extend("- " + reason for reason in review.get("issues", []))
        for edit in review.get("proposed_edits", []):
            child = {**edit["comparison"], "ledger": review.get("ledger", [])}
            lines.extend(["", f"### Proposed edit {edit['index']}", review_markdown(child)])
    return "\n".join(lines)
