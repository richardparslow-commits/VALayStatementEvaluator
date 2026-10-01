"""Validate the complete scoring response before publishing any derived rating."""
from __future__ import annotations

import math
from typing import Any

DIMENSION_LABELS = {
    "factual_accuracy": "Factual Accuracy vs. Records",
    "specificity_detail": "Specificity & Detail",
    "lay_competence": "Lay Competence Boundaries",
    "condition_connection": "Connection to Claimed Condition",
    "continuity_timeline": "Continuity & Timeline",
    "functional_impact": "Functional Impact",
    "credibility_consistency": "Credibility & Consistency",
    "form_completeness": "Form & Completeness",
}
RUBRIC_POLICY = "complete_rubric_v1"
RUBRIC_MAX_ATTEMPTS = 3
RUBRIC_INCOMPLETE_NOTICE = (
    "Scoring incomplete: a complete, valid rubric response was unavailable after "
    "three attempts. Completed record and claim review is retained as a partial "
    "evaluation. Ratings, effectiveness score, scoring recommendations, and the "
    "proposed rewrite are withheld. Re-run the evaluation before relying on scoring."
)
RUBRIC_UNVALIDATED_NOTICE = (
    "Scoring unavailable: this result has no recognized complete rubric validation "
    "or its scoring data is invalid. Ratings, effectiveness score, scoring "
    "recommendations, and the proposed rewrite are withheld. Re-run the evaluation."
)


class RubricValidationError(ValueError):
    """Static errors must never include model output or private record text."""


def _text(value: Any, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise RubricValidationError("Rubric text fields must be nonempty strings.")
    return value.strip()


def validate_scores(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict) or raw.keys() != DIMENSION_LABELS.keys():
        raise RubricValidationError("Rubric scores must contain exactly the eight dimensions.")
    scores: dict[str, float] = {}
    for key in DIMENSION_LABELS:
        value = raw[key]
        # Check bounds before conversion so even arbitrarily large JSON integers
        # fail safely. bool is an int subclass, but is not a JSON number.
        if type(value) not in (int, float) or not 0 <= value <= 10 or not math.isfinite(value):
            raise RubricValidationError("Rubric scores must be finite JSON numbers from 0 to 10.")
        scores[key] = float(value)
    return scores


def normalize_rubric(raw: Any) -> dict[str, Any]:
    required = {"scores", "rationales", "improvements", "omitted_record_facts", "executive_summary"}
    if not isinstance(raw, dict) or raw.keys() != required:
        raise RubricValidationError("Rubric response must contain all required fields and no others.")
    scores = validate_scores(raw["scores"])
    reasons = raw["rationales"]
    if not isinstance(reasons, dict) or reasons.keys() != DIMENSION_LABELS.keys():
        raise RubricValidationError("Rubric rationales must contain exactly the eight dimensions.")
    rationales = {key: _text(reasons[key]) for key in DIMENSION_LABELS}
    improvements = raw["improvements"]
    if not isinstance(improvements, list):
        raise RubricValidationError("Rubric improvements must be a JSON list.")
    normalized_improvements = []
    for row in improvements:
        if not isinstance(row, dict) or row.keys() != {"priority", "problem", "suggestion", "example_rewrite"}:
            raise RubricValidationError("Rubric improvement rows must contain all required fields.")
        priority = row["priority"]
        if type(priority) is not int or priority < 1:
            raise RubricValidationError("Rubric improvement priority must be a positive JSON integer.")
        normalized_improvements.append({
            "priority": priority, "problem": _text(row["problem"]),
            "suggestion": _text(row["suggestion"]),
            "example_rewrite": _text(row["example_rewrite"], empty=True),
        })
    omitted = raw["omitted_record_facts"]
    if not isinstance(omitted, list):
        raise RubricValidationError("Rubric omitted facts must be a JSON list.")
    normalized_omitted = []
    for row in omitted:
        if not isinstance(row, dict) or row.keys() != {"fact", "source"}:
            raise RubricValidationError("Rubric omitted-fact rows must contain fact and source.")
        normalized_omitted.append({"fact": _text(row["fact"]), "source": _text(row["source"])})
    return {
        "scores": scores, "rationales": rationales,
        "improvements": normalized_improvements, "omitted_record_facts": normalized_omitted,
        "executive_summary": _text(raw["executive_summary"]),
    }


def rubric_is_complete(result: Any) -> bool:
    if (getattr(result, "scoring_policy", "") != RUBRIC_POLICY
            or getattr(result, "scoring_status", "") != "complete"):
        return False
    try:
        normalize_rubric({key: getattr(result, key, None) for key in (
            "scores", "rationales", "improvements", "omitted_record_facts", "executive_summary"
        )})
    except RubricValidationError:
        return False
    return True


def scoring_notice(result: Any) -> str:
    if rubric_is_complete(result):
        return "Scoring complete: all eight rubric scores and required response fields passed validation."
    if (getattr(result, "scoring_policy", "") == RUBRIC_POLICY
            and getattr(result, "scoring_status", "") == "incomplete"):
        return RUBRIC_INCOMPLETE_NOTICE
    return RUBRIC_UNVALIDATED_NOTICE
