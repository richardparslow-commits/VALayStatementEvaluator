"""The Aid & Attendance intake wizard, shared by the Draft and Evaluate tabs.

Renders the sixteen structured questions (``app.aa_intake``) as numbered
**form steps** — one ``st.form`` per stage — and collects the answers from
session state into the ``aa_*`` witness-dict convention (see
``answers_from_witness``).

Why a durable store instead of widget return values
---------------------------------------------------
Streamlit keeps widget state only for widgets rendered in the *current* run.
A multi-step wizard does not render steps the user has navigated past, so
reading widget keys directly would silently forget every earlier step. Each
step therefore lives in a ``st.form`` whose submit path copies its answers
into ``st.session_state["<prefix>_aa_store"]`` (a plain dict) before the step
advances. Collection reads the store — never the widgets — so the answers
survive navigation, collapse, and Streamlit's rerun-everything model.

Key shapes (prefix is per-tab, ``"draft"`` / ``"eval"``, so the two tabs
never collide):

* ``<prefix>_aa_<slug>``  — one widget per question
* ``<prefix>_aa_store``   — ``{slug: answer}`` for every saved answer
* ``<prefix>_aa_step``    — current step index (mutated only in callbacks /
  the submit path)

Form semantics, stated once: typed-but-unsaved answers are not collected
until the step's button is pressed — the same contract as every other form
in this app.
"""
from __future__ import annotations

from typing import Callable, Mapping

import streamlit as st

from ..aa_intake import (
    INTAKE_QUESTIONS,
    KEY_PREFIX,
    answers_from_witness,
)

# Wizard steps, in order — contiguous runs of the canonical INTAKE_QUESTIONS
# order (the classic intake numbering, kept stable so "Question 7 of 16"
# always means the same question). Every slug appears exactly once; the
# module assert below enforces the partition.
_STEPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("The caregiver's role", ("care_provider_frequency",)),
    (
        "Daily living & appliances",
        ("cleanliness", "dressing", "eating", "wants_of_nature", "prosthetics"),
    ),
    (
        "Supervision & bed status",
        ("hazard_supervision", "bedridden", "healthcare_supervision"),
    ),
    (
        "Consequences & daily realities",
        (
            "no_assistance_consequence",
            "adl_reminders",
            "flashback_care",
            "medication_supervision",
            "eating_safety",
            "care_substitute",
            "left_alone",
        ),
    ),
)

_BY_SLUG = {q.slug: q for q in INTAKE_QUESTIONS}
assert tuple(slug for _, slugs in _STEPS for slug in slugs) == tuple(
    q.slug for q in INTAKE_QUESTIONS
), "wizard steps must partition the intake questions in order"

_INTRO = (
    "These 16 questions are the Aid & Attendance evidence matrix — the "
    "observations a rater looks for. Answer what you have personally seen; "
    "leave blank anything that does not apply or you have not observed. "
    "Answers become first-hand observations woven into the statement — the "
    "app never turns them into legal conclusions."
)


def _store_key(prefix: str) -> str:
    return f"{prefix}_{KEY_PREFIX}store"


def _step_key(prefix: str) -> str:
    return f"{prefix}_{KEY_PREFIX}step"


def read_store(prefix: str) -> dict[str, str]:
    """The durable per-step answers for this tab (a copy; empty if none yet)."""
    store = st.session_state.get(_store_key(prefix))
    return dict(store) if isinstance(store, dict) else {}


def apply_step_answers(
    store: dict[str, str],
    widget_state: Mapping[str, object],
    slugs: tuple[str, ...],
    prefix: str,
) -> dict[str, str]:
    """Commit one step's widget values into the durable store.

    The persistence contract, in one place: non-empty answers are saved,
    cleared answers are removed (a deliberate negative choice like "No" is
    a non-empty answer and is kept — see ``completed_count``), the updated
    store is written back to session state, and the step pointer advances
    (clamped at the last step). *store* is treated as a working copy — the
    caller passes :func:`read_store`'s copy and this function owns writing
    it back. Pure with respect to the passed mapping, so tests can drive
    the contract without a Streamlit runtime.
    """
    for slug in slugs:
        value = widget_state.get(f"{prefix}_{KEY_PREFIX}{slug}")
        if isinstance(value, str) and value.strip():
            store[slug] = value.strip()
        else:
            store.pop(slug, None)
    st.session_state[_store_key(prefix)] = store
    st.session_state[_step_key(prefix)] = min(
        int(st.session_state.get(_step_key(prefix), 0) or 0) + 1, len(_STEPS) - 1
    )
    return store


def _goto_step(prefix: str, step: int) -> "Callable[[], None]":
    """A callback moving the wizard to *step* (used by Back)."""

    def _cb() -> None:
        st.session_state[_step_key(prefix)] = step

    return _cb


def render_aa_intake_wizard(prefix: str, *, expanded: bool = False) -> None:
    """Render the 16-question wizard as numbered form steps.

    *prefix* namespaces every widget key (``"draft"`` / ``"eval"``). Optional
    questions stay optional: everything defaults to unanswered and the
    pipeline treats an all-empty wizard exactly like no intake at all.
    """
    store = read_store(prefix)
    answered = len(store)
    label = (
        f"Aid & Attendance intake — 16 questions ({answered} of 16 answered)"
        if answered
        else "Aid & Attendance intake — 16 questions (optional, but they make the statement stronger)"
    )
    with st.expander(label, expanded=expanded):
        st.caption(_INTRO)

        step_index = max(0, min(int(st.session_state.get(_step_key(prefix), 0) or 0), len(_STEPS) - 1))
        title, slugs = _STEPS[step_index]
        # Canonical question numbering across steps ("Question 6 of 16").
        offset = sum(len(s) for _, s in _STEPS[:step_index])

        st.markdown(f"**Step {step_index + 1} of {len(_STEPS)} — {title}**")

        form_key = f"{prefix}_aa_form_{step_index}"
        with st.form(form_key, clear_on_submit=False):
            for i, slug in enumerate(slugs):
                question = _BY_SLUG[slug]
                number = offset + i + 1
                key = f"{prefix}_{KEY_PREFIX}{slug}"
                # Seed the widget from the durable store when its live state
                # is gone (the user navigated away and back). This must happen
                # before the widget is created in this run — Streamlit only
                # honours a session_state assignment made pre-instantiation.
                if slug in store and key not in st.session_state:
                    st.session_state[key] = store[slug]
                help_text = (
                    f"Question {number} of 16"
                    + (f" — {question.help}" if question.help else "")
                )
                if question.kind == "detail":
                    st.text_area(
                        f"{number}. {question.text}",
                        key=key,
                        height=90,
                        help=help_text,
                    )
                else:
                    st.selectbox(
                        f"{number}. {question.text}",
                        options=("", *question.choices),
                        key=key,
                        format_func=lambda option: (
                            "— not answered —" if option == "" else option
                        ),
                        help=help_text,
                    )

            submitted = st.form_submit_button(
                "Save & continue" if step_index < len(_STEPS) - 1 else "Save & finish",
                key=f"{prefix}_aa_submit_{step_index}",
            )

        if submitted:
            # SessionStateProxy satisfies the Mapping contract at runtime;
            # iterating it for a copy can surface widget-id entries, so pass
            # it as-is rather than dict()-ing it.
            apply_step_answers(store, st.session_state, slugs, prefix)  # type: ignore[arg-type]
            st.rerun()

        # Back navigation lives outside the form so it never requires saving.
        if step_index > 0:
            st.button(
                "← Back",
                key=f"{prefix}_aa_back_{step_index}",
                on_click=_goto_step(prefix, step_index - 1),
            )
        if answered:
            st.caption(
                "Answers are saved per step. Reopen any earlier step to change them."
            )


def collect_aa_answers(prefix: str) -> dict[str, str]:
    """Read the wizard's saved answers out of session state as a witness dict.

    Returns only non-empty answers, keyed ``aa_<slug>`` — the exact shape
    ``care_observation_block``/``care_gaps_text`` consume and the shape the
    job queue already serializes for the draft pathway. A fresh or fully
    blank wizard yields ``{}``, which makes the whole feature a no-op.
    """
    store = read_store(prefix)
    return {f"{KEY_PREFIX}{slug}": answer for slug, answer in store.items()}


def collect_aa_answers_offsession(witness_state: dict) -> dict[str, str]:
    """Non-UI collector for tests and scripts: filter a flat witness dict.

    Same contract as :func:`collect_aa_answers` — takes a flat dict in the
    witness-dict shape and returns its non-empty ``aa_*`` entries — but reads
    a plain mapping, so unit tests (and AppTest assertions) can verify the
    aa_* convention without a Streamlit session.
    """
    return {
        f"{KEY_PREFIX}{slug}": answer
        for slug, answer in answers_from_witness(witness_state).items()
    }
