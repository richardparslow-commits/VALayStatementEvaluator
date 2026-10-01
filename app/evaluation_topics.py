"""Complete typed evaluation topic analysis; structural checks, not semantic proof."""
from __future__ import annotations

import re
from typing import Any

from .condition_selector import TOPIC_LABELS
from .rubric_validation import rubric_is_complete

TOPIC_POLICY = "complete_evaluation_topics_v1"
TOPIC_MAX_ATTEMPTS = 3
TOPIC_ORDER = tuple(TOPIC_LABELS)
TOPIC_INCOMPLETE_NOTICE = (
    "Topic coverage analysis unavailable after three attempts. Completed record, "
    "claim and rubric review is retained as a partial evaluation. Topic coverage "
    "counts, follow-up conclusions, recommendations and the proposed rewrite are "
    "withheld. Re-run before relying on topic coverage."
)
TOPIC_UNVALIDATED_NOTICE = (
    "Topic coverage is unvalidated: this result has no recognized complete topic "
    "validation or its topic data is invalid. Coverage counts, follow-up conclusions, "
    "recommendations and the proposed rewrite are withheld. Re-run the evaluation."
)


class TopicValidationError(ValueError):
    """Validation errors contain no model output or private statement text."""


def _text(raw: Any, *, nonempty: bool = False) -> str:
    if not isinstance(raw, str) or (nonempty and not raw.strip()):
        raise TopicValidationError("Topic text fields must have the required string type and content.")
    return raw.strip()


def _letter(raw: str) -> str:
    match = re.match(r"^([A-O])(?:\s*[.():\-–—]|\s|$)", raw)
    if not match:
        raise TopicValidationError("Topic labels must identify a checklist topic A through O.")
    return match[1]


def _label(raw: Any) -> str:
    text = _text(raw, nonempty=True)
    letter = _letter(text)
    suffix = text[1:].lstrip(" .():-–—").strip()
    # The knowledge headings include explanatory parenthetical suffixes. The
    # identity must still match the existing catalogue, not an invented label.
    suffix = re.sub(r"\s*\([^)]*\)\s*$", "", suffix).strip()
    if suffix and " ".join(suffix.replace("&", "and").casefold().split()) != TOPIC_LABELS[letter].casefold():
        raise TopicValidationError("Unknown checklist topic label.")
    return letter


def normalize_topics(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.keys() != {"claim_focus", "topics", "critical_gaps", "notes"}:
        raise TopicValidationError("Topic response must contain exactly all required fields.")
    focus = _text(raw["claim_focus"], nonempty=True)
    notes = _text(raw["notes"])
    rows = raw["topics"]
    if not isinstance(rows, list) or len(rows) != len(TOPIC_ORDER):
        raise TopicValidationError("Exactly one entry per checklist topic A through O is required.")
    topics: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.keys() != {"topic", "applicable", "coverage", "evidence", "gap_note"}:
            raise TopicValidationError("Topic rows must contain exactly all required fields.")
        letter = _label(row["topic"])
        if letter in topics:
            raise TopicValidationError("Duplicate checklist topic.")
        if type(row["applicable"]) is not bool:
            raise TopicValidationError("Topic applicability must be a JSON boolean.")
        coverage = _text(row["coverage"])
        if coverage not in {"covered", "partial", "absent", "not applicable"}:
            raise TopicValidationError("Unknown topic coverage value.")
        evidence = _text(row["evidence"])
        gap = _text(row["gap_note"])
        applicable = row["applicable"]
        if applicable == (coverage == "not applicable"):
            raise TopicValidationError("Topic applicability and coverage disagree.")
        needs_gap = applicable and coverage in {"partial", "absent"}
        if bool(gap) != needs_gap:
            raise TopicValidationError("Topic gap wording must agree with coverage.")
        if needs_gap and (len(gap) < 4 or not any(c.isalpha() for c in gap)
                          or gap.casefold() in {"none", "n/a", "unknown", "tbd", "not applicable"}):
            raise TopicValidationError("Weak or absent applicable topics need non-placeholder gap wording.")
        if bool(evidence) != (applicable and coverage in {"covered", "partial"}):
            raise TopicValidationError("Topic evidence must agree with coverage.")
        topics[letter] = {"topic": f"{letter}. {TOPIC_LABELS[letter].title()}",
                          "applicable": applicable, "coverage": coverage,
                          "evidence": evidence, "gap_note": gap}
    gaps = raw["critical_gaps"]
    if not isinstance(gaps, list) or len(gaps) > 5:
        raise TopicValidationError("Critical gaps must be a JSON list with at most five entries.")
    critical = []
    seen: set[str] = set()
    for raw_gap in gaps:
        gap = _text(raw_gap, nonempty=True)
        letter = _letter(gap)
        if (letter in seen or not topics[letter]["applicable"]
                or topics[letter]["coverage"] not in {"partial", "absent"} or len(gap) < 4):
            raise TopicValidationError("Critical gaps must identify distinct weak applicable topics.")
        seen.add(letter)
        critical.append(gap)
    return {"claim_focus": focus, "topics": [topics[key] for key in TOPIC_ORDER],
            "critical_gaps": critical, "notes": notes}


def topics_are_complete(result: Any) -> bool:
    if (getattr(result, "topic_policy", "") != TOPIC_POLICY
            or getattr(result, "topic_status", "") != "complete"):
        return False
    try:
        normalize_topics({"claim_focus": getattr(result, "topic_focus", None),
                          "topics": getattr(result, "topic_rows", None),
                          "critical_gaps": getattr(result, "topic_critical_gaps", None),
                          "notes": getattr(result, "topic_notes", None)})
    except TopicValidationError:
        return False
    return True


def evaluation_is_complete(result: Any) -> bool:
    return rubric_is_complete(result) and topics_are_complete(result)


def topic_notice(result: Any) -> str:
    if topics_are_complete(result):
        return "Topic analysis complete: all fifteen checklist topics passed response validation."
    if (getattr(result, "topic_policy", "") == TOPIC_POLICY
            and getattr(result, "topic_status", "") == "incomplete"):
        return TOPIC_INCOMPLETE_NOTICE
    return TOPIC_UNVALIDATED_NOTICE
