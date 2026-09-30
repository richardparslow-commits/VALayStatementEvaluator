"""Synthetic, structurally complete grounding responses for pipeline fixtures."""


def complete_grounding(observation="Daily knee pain observed."):
    return {
        "supported_observations": [],
        "unverified_observations": [{
            "observation": observation,
            "action": "Keep as the witness's account; confirm before signing.",
        }] if observation else [],
        "conflicts": [],
        "suggested_inclusions": [],
        "strengthening_questions": [],
        "topic_coverage": [{
            "topic": f"{label}. Synthetic topic {label}",
            "applicable": False,
            "covered": False,
            "prompt_for_witness": "",
        } for label in "ABCDEFGHIJKLMNO"],
    }
