"""Claimed-condition selector (feature: Condition-Specific Templates).

Renders body-system radio buttons followed by a searchable, multi-select
condition dropdown immediately after record-source selection in both Evaluate
and Draft modes (FR-1/FR-2/FR-3). Selecting condition(s) automatically
pre-selects the union of relevant topics from the 12-topic checklist (FR-4,
FR-7), with a fallback default set for conditions without a predefined
mapping (FR-8). A dedicated Aid & Attendance / SMC-L toggle forces topics
B, C, E, and J as mandatory (FR-9/FR-10), while users may still manually
adjust the non-mandatory topics (FR-6).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import streamlit as st

from .agiloop_telemetry import track_feature_error, track_goal, track_impression, track_interaction

_CONDITION_TOPICS_PATH = Path(__file__).parent / "condition_topics.json"

# FR-9 / FR-10 — Aid & Attendance / SMC-L forces these topics as mandatory,
# regardless of any condition-based mapping. Condition-based topics remain
# optional suggestions the user can still toggle off.
AA_FORCED_TOPICS: tuple[str, ...] = ("B", "C", "E", "J")

# FR-8 — fallback for conditions without a predefined mapping: hazards (A),
# before/after progression & worsening timeline (G), and functional
# impairment (K). Fully manually adjustable afterward.
DEFAULT_FALLBACK_TOPICS: tuple[str, ...] = ("A", "G", "K")

TOPIC_LABELS: dict[str, str] = {
    "A": "Hazards and dangers",
    "B": "Caregiver burden and necessity of care",
    "C": "Basic personal care and hygiene",
    "D": "Medication and financial management",
    "E": "Household safety and hazard protection",
    "F": "Routine errands and chores",
    "G": "Context and symptom progression",
    "H": "Daily life and observable behaviors",
    "I": "Impact on family dynamics",
    "J": "Physical side effects and secondary conditions from medications",
    "K": "Functional impairments from medication side effects",
    "L": "Formatting and certification",
}


@lru_cache(maxsize=1)
def _load_condition_topics() -> dict[str, Any]:
    with _CONDITION_TOPICS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _body_systems() -> list[str]:
    return list(_load_condition_topics().get("body_systems", {}).keys())


def _conditions_for_system(body_system: str) -> list[str]:
    systems = _load_condition_topics().get("body_systems", {})
    conditions = systems.get(body_system, {}).get("conditions", {})
    return list(conditions.keys())


def _topics_for_condition(body_system: str, condition: str) -> list[str]:
    systems = _load_condition_topics().get("body_systems", {})
    conditions = systems.get(body_system, {}).get("conditions", {})
    topics = conditions.get(condition)
    if not topics:
        return list(DEFAULT_FALLBACK_TOPICS)
    return list(topics)


def get_preselected_topics(
    conditions: list[tuple[str, str]], aa_toggle: bool
) -> tuple[list[str], list[str]]:
    """Return `(preselected_topics, forced_topics)` for the given selections.

    `conditions` is a list of `(body_system, condition_name)` pairs. Per FR-7
    the pre-selection is the UNION of each condition's mapped topics. Per
    FR-9/FR-10, when `aa_toggle` is active, `AA_FORCED_TOPICS` are added as
    mandatory (forced) while condition-based topics remain optional
    suggestions.
    """
    union: set[str] = set()
    for body_system, condition in conditions:
        union.update(_topics_for_condition(body_system, condition))

    forced: list[str] = list(AA_FORCED_TOPICS) if aa_toggle else []
    union.update(forced)

    ordered = [topic for topic in TOPIC_LABELS if topic in union]
    return ordered, forced


def render_condition_selector(slot: str, feature_id: str) -> dict[str, Any]:
    """Render the claimed-condition selector for `slot` (`"eval"` or `"draft"`).

    Must be called immediately after record-source selection in both modes.
    Returns the current selection state:

        {
          "conditions": [(body_system, condition_name), ...],
          "aa_toggle": bool,
          "preselected_topics": [...letters...],
          "forced_topics": [...letters...],
          "proceeded": bool,
        }

    The final selection is written to `st.session_state` under
    `selected_conditions_{slot}`, `aa_toggle_{slot}`,
    `preselected_topics_{slot}`, and `forced_topics_{slot}` only once the
    user clicks "Proceed" (AC: "when proceeding, then selection is stored in
    session state").
    """
    empty_state: dict[str, Any] = {
        "conditions": [],
        "aa_toggle": False,
        "preselected_topics": [],
        "forced_topics": [],
        "proceeded": bool(st.session_state.get(f"preselected_topics_{slot}")),
    }

    impression_key = f"cs_impression_sent_{slot}"
    if not st.session_state.get(impression_key):
        try:
            track_impression(feature_id, entry_point=f"{slot}_record_source")
        except Exception:  # noqa: BLE001 - telemetry must never break the UI
            pass
        st.session_state[impression_key] = True

    st.markdown("##### Claimed condition")
    st.caption(
        "Select the body system, then the specific condition(s), so the app can "
        "pre-select the relevant topics from the 12-topic checklist."
    )

    try:
        systems = _body_systems()
    except Exception as exc:  # noqa: BLE001 - keep the app usable on data errors
        track_feature_error(feature_id, exc)
        st.error("Could not load condition mappings; topic pre-selection is unavailable.")
        return empty_state

    if not systems:
        return empty_state

    body_system_key = f"cs_body_system_{slot}"
    body_system = st.radio(
        "Body system",
        systems,
        key=body_system_key,
        horizontal=True,
    )
    last_system_key = f"cs_last_body_system_{slot}"
    if body_system and st.session_state.get(last_system_key) != body_system:
        st.session_state[last_system_key] = body_system
        track_interaction(feature_id, action="body_system_selected", body_system=body_system)

    conditions_key = f"cs_conditions_{slot}"
    options = _conditions_for_system(body_system) if body_system else []
    selected = st.multiselect(
        "Claimed condition(s) — type to search",
        options,
        key=conditions_key,
        help="Type to filter. Select one or more; pre-selected topics are the "
        "union across all selected conditions.",
    )

    last_conditions_key = f"cs_last_conditions_{slot}"
    if selected != st.session_state.get(last_conditions_key):
        st.session_state[last_conditions_key] = list(selected)
        if selected:
            track_interaction(
                feature_id,
                action="condition_selected",
                body_system=body_system,
                condition_count=len(selected),
            )

    aa_key = f"cs_aa_toggle_{slot}"
    aa_toggle = st.checkbox(
        "This is an Aid & Attendance (A&A) / SMC-L claim",
        key=aa_key,
        help="Forces the caregiver-necessity, personal-care, household-safety, "
        "and medication-side-effect topics (B, C, E, J) as mandatory.",
    )
    last_aa_key = f"cs_last_aa_{slot}"
    if aa_toggle != st.session_state.get(last_aa_key, False):
        st.session_state[last_aa_key] = aa_toggle
        track_interaction(feature_id, action="aa_toggle", aa_toggle=aa_toggle)

    conditions_pairs = [(body_system, condition) for condition in selected] if body_system else []
    preselected_topics, forced_topics = get_preselected_topics(conditions_pairs, aa_toggle)

    if not preselected_topics:
        return {
            "conditions": conditions_pairs,
            "aa_toggle": aa_toggle,
            "preselected_topics": [],
            "forced_topics": forced_topics,
            "proceeded": False,
        }

    all_topics = list(TOPIC_LABELS.keys())
    with st.container(border=True):
        st.success(
            f"**{len(preselected_topics)} topic(s) pre-selected**: "
            + ", ".join(f"{topic} — {TOPIC_LABELS[topic]}" for topic in preselected_topics)
        )
        if forced_topics:
            st.warning(
                "Mandatory (A&A/SMC-L, cannot be removed): "
                + ", ".join(f"{topic} — {TOPIC_LABELS[topic]}" for topic in forced_topics)
            )

        adjust_key = f"cs_adjusted_topics_{slot}"
        default_value = st.session_state.get(adjust_key, preselected_topics)
        # Keep forced topics present in the default even after manual edits.
        merged_default = [t for t in all_topics if t in set(default_value) | set(forced_topics)]
        adjusted = st.multiselect(
            "Adjust pre-selected topics",
            all_topics,
            default=merged_default,
            format_func=lambda topic: f"{topic} — {TOPIC_LABELS[topic]}",
            key=adjust_key,
            help="Mandatory A&A/SMC-L topics are always re-added even if removed here.",
        )
        final_topics = [t for t in all_topics if t in set(adjusted) | set(forced_topics)]

        col_proceed, col_adjust = st.columns(2)
        proceeded = False
        if col_proceed.button("Proceed", key=f"cs_proceed_{slot}", type="primary"):
            st.session_state[f"selected_conditions_{slot}"] = conditions_pairs
            st.session_state[f"aa_toggle_{slot}"] = aa_toggle
            st.session_state[f"preselected_topics_{slot}"] = final_topics
            st.session_state[f"forced_topics_{slot}"] = forced_topics
            try:
                track_interaction(
                    feature_id,
                    action="proceed",
                    body_system=body_system,
                    condition_count=len(selected),
                )
                track_goal(
                    feature_id,
                    "preselection_confirmed",
                    preselected_topic_count=len(final_topics),
                )
            except Exception as exc:  # noqa: BLE001 - telemetry must never break the UI
                track_feature_error(feature_id, exc)
            proceeded = True
        if col_adjust.button("Adjust", key=f"cs_adjust_{slot}"):
            st.session_state.pop(f"preselected_topics_{slot}", None)

    return {
        "conditions": conditions_pairs,
        "aa_toggle": aa_toggle,
        "preselected_topics": final_topics,
        "forced_topics": forced_topics,
        "proceeded": proceeded or bool(st.session_state.get(f"preselected_topics_{slot}")),
    }
