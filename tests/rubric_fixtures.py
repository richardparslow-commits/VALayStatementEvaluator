"""Complete synthetic scoring fixtures; no provider calls or real records."""
from tests import hermetic  # noqa: F401
from app.rubric_validation import DIMENSION_LABELS, RUBRIC_POLICY


def complete_rubric(scores=None):
    return {
        "scores": dict(scores) if scores is not None else {key: 6.0 for key in DIMENSION_LABELS},
        "rationales": {key: "Synthetic rationale for " + key for key in DIMENSION_LABELS},
        "improvements": [{"priority": 1, "problem": "Vague timeline", "suggestion": "Ask the witness for dates.", "example_rewrite": ""}],
        "omitted_record_facts": [],
        "executive_summary": "Synthetic scoring assessment; source and witness review still required.",
    }


def scored_result_fields(scores=None):
    return {**complete_rubric(scores), "scoring_policy": RUBRIC_POLICY, "scoring_status": "complete"}
