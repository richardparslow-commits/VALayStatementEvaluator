"""Complete synthetic topic fixtures; no provider calls or real records."""
from tests import hermetic  # noqa: F401
from app.condition_selector import TOPIC_LABELS
from app.evaluation_topics import TOPIC_POLICY


def complete_topics():
    rows = [{"topic": f"{letter}. {label.title()}", "applicable": False,
             "coverage": "not applicable", "evidence": "", "gap_note": ""}
            for letter, label in TOPIC_LABELS.items()]
    rows[0].update(applicable=True, coverage="partial", evidence="Stove left on.",
                   gap_note="Describe a specific near-miss incident.")
    return {"claim_focus": "knee condition - increased rating", "topics": rows,
            "critical_gaps": ["A. Hazards and Dangers — needs incident detail"], "notes": "Synthetic topic review."}


def topic_result_fields(raw=None):
    data = complete_topics() if raw is None else raw
    return {"topic_policy": TOPIC_POLICY, "topic_status": "complete",
            "topic_focus": data["claim_focus"], "topic_rows": data["topics"],
            "topic_critical_gaps": data["critical_gaps"], "topic_notes": data["notes"]}
