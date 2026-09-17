"""Shared follow-up question helpers for Evaluate and Draft results."""
from __future__ import annotations

from typing import Any

import streamlit as st


def draft_follow_up_questions(result: Any) -> list[dict[str, str]]:
    """Return uncovered checklist-topic questions from a draft result."""
    grounding = getattr(result, "grounding", {})
    if not isinstance(grounding, dict):
        return []
    return _draft_topic_questions(grounding.get("topic_coverage", []))


def evaluate_follow_up_questions(result: Any) -> list[dict[str, str]]:
    """Return uncovered checklist-topic questions from an evaluation result."""
    claim_focus = getattr(result, "topic_focus", "")
    return _evaluation_topic_questions(getattr(result, "topic_rows", []), claim_focus=claim_focus)


def compose_follow_up_appendix(slot: str) -> str:
    """Serialize saved follow-up answers for inclusion in the next run."""
    answers = _saved_answers(slot)
    if not answers:
        return ""
    lines = ["Additional follow-up details confirmed after the last run:"]
    for item in answers:
        topic = _clean_text(item.get("topic"))
        question = _clean_text(item.get("question"))
        answer = _clean_text(item.get("answer"))
        if not answer:
            continue
        prefix = f"- {topic}: " if topic else "- "
        if question:
            lines.append(f"{prefix}{question} Answer: {answer}")
        else:
            lines.append(f"{prefix}{answer}")
    return "\n".join(lines) if len(lines) > 1 else ""


def append_follow_up_answers(text: str, *, slot: str) -> str:
    """Append accepted follow-up answers to the next prompt input, if any."""
    appendix = compose_follow_up_appendix(slot)
    if not appendix:
        return text
    base = text.rstrip()
    if not base:
        return appendix
    return f"{base}\n\n{appendix}"


def render_follow_up_questions(
    *,
    slot: str,
    source_id: str,
    questions: list[dict[str, str]],
    empty_message: str,
    next_run_label: str,
) -> None:
    """Render one follow-up question at a time with accept/skip controls."""
    _ensure_state(slot, source_id, questions)
    saved = _saved_answers(slot)
    skipped = _saved_skips(slot)
    index = int(st.session_state.get(_key(slot, "index"), 0) or 0)

    with st.expander("🤖 Automated follow-up question generator", expanded=bool(questions)):
        notice = _clean_text(st.session_state.get(_key(slot, "notice"), ""))
        if notice:
            st.success(notice)

        if not questions:
            st.info(empty_message)
            return

        if saved:
            st.caption(
                f"{len(saved)} accepted answer(s) will be included automatically in the next "
                f"{next_run_label} run."
            )
            for item in saved:
                topic = _clean_text(item.get("topic")) or "Follow-up"
                answer = _clean_text(item.get("answer"))
                st.write(f"- **{topic}** — {answer}")

        if index >= len(questions):
            if skipped:
                st.caption(f"Skipped questions: {len(skipped)}")
            st.success("You have reviewed every generated follow-up question for this run.")
            return

        current = questions[index]
        st.caption(f"Question {index + 1} of {len(questions)} — {current.get('topic', 'Checklist topic')}")
        question_text = st.text_area(
            "Follow-up question (editable before asking)",
            value=current.get("question", ""),
            height=110,
            key=_key(slot, f"question_{index}"),
        )
        answer_text = st.text_area(
            "Witness answer",
            value="",
            height=160,
            key=_key(slot, f"answer_{index}"),
        )
        accept = st.button("Save answer and continue", key=_key(slot, f"accept_{index}"))
        skip = st.button("Skip this question", key=_key(slot, f"skip_{index}"))

        if accept:
            cleaned_answer = _clean_text(answer_text)
            if not cleaned_answer:
                st.warning("Enter the witness's answer before saving it, or skip this question.")
                return
            saved.append(
                {
                    "topic": _clean_text(current.get("topic")),
                    "question": _clean_text(question_text) or _clean_text(current.get("question")),
                    "answer": cleaned_answer,
                }
            )
            st.session_state[_key(slot, "saved")] = saved
            st.session_state[_key(slot, "index")] = index + 1
            st.session_state[_key(slot, "notice")] = (
                f"Saved follow-up answer for {_clean_text(current.get('topic')) or 'the current topic'}."
            )
            st.rerun()
        if skip:
            skipped.append(
                {
                    "topic": _clean_text(current.get("topic")),
                    "question": _clean_text(question_text) or _clean_text(current.get("question")),
                }
            )
            st.session_state[_key(slot, "skipped")] = skipped
            st.session_state[_key(slot, "index")] = index + 1
            st.session_state[_key(slot, "notice")] = (
                f"Skipped {_clean_text(current.get('topic')) or 'the current topic'}."
            )
            st.rerun()


def _draft_topic_questions(topic_rows: Any) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []
    if not isinstance(topic_rows, list):
        return questions
    for row in topic_rows:
        if not isinstance(row, dict) or not row.get("applicable") or row.get("covered"):
            continue
        topic = _clean_text(row.get("topic"))
        question = _clean_text(row.get("prompt_for_witness")) or _default_question(topic)
        if question:
            questions.append({"topic": topic or "Checklist topic", "question": question})
    return questions


def _evaluation_topic_questions(topic_rows: Any, *, claim_focus: Any) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []
    if not isinstance(topic_rows, list):
        return questions
    focus = _clean_text(claim_focus)
    for row in topic_rows:
        if not isinstance(row, dict) or not row.get("applicable"):
            continue
        coverage = _clean_text(row.get("coverage")).lower()
        if coverage == "covered":
            continue
        topic = _clean_text(row.get("topic"))
        gap_note = _clean_text(row.get("gap_note"))
        question = _gap_note_to_question(topic, gap_note, claim_focus=focus)
        if question:
            questions.append({"topic": topic or "Checklist topic", "question": question})
    return questions


def _gap_note_to_question(topic: str, gap_note: str, *, claim_focus: str = "") -> str:
    if gap_note.endswith("?"):
        return gap_note
    if gap_note:
        if claim_focus:
            return (
                f"For this {claim_focus} statement, what should the witness add about {topic or 'this topic'} "
                f"if it is true? {gap_note.rstrip('.')}."
            )
        return f"For {topic or 'this topic'}, what details should the witness add if true? {gap_note.rstrip('.')}."
    return _default_question(topic)


def _default_question(topic: str) -> str:
    return f"What has the witness personally observed about {topic or 'this checklist topic'} that should be added if true?"


def _saved_answers(slot: str) -> list[dict[str, str]]:
    raw = st.session_state.get(_key(slot, "saved"), [])
    return list(raw) if isinstance(raw, list) else []


def _saved_skips(slot: str) -> list[dict[str, str]]:
    raw = st.session_state.get(_key(slot, "skipped"), [])
    return list(raw) if isinstance(raw, list) else []


def _ensure_state(slot: str, source_id: str, questions: list[dict[str, str]]) -> None:
    normalized = [
        {"topic": _clean_text(item.get("topic")), "question": _clean_text(item.get("question"))}
        for item in questions
        if isinstance(item, dict)
    ]
    signature = tuple((item["topic"], item["question"]) for item in normalized)
    signature_key = _key(slot, "signature")
    if (
        st.session_state.get(_key(slot, "source_id")) != source_id
        or st.session_state.get(signature_key) != signature
    ):
        st.session_state[_key(slot, "source_id")] = source_id
        st.session_state[signature_key] = signature
        st.session_state[_key(slot, "index")] = 0
        st.session_state[_key(slot, "saved")] = []
        st.session_state[_key(slot, "skipped")] = []
        st.session_state[_key(slot, "notice")] = ""


def _key(slot: str, suffix: str) -> str:
    return f"{slot}_follow_up_{suffix}"


def _clean_text(value: Any) -> str:
    return str(value).strip() if isinstance(value, str) else str(value).strip() if value is not None else ""
