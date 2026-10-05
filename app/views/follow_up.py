"""Shared follow-up question helpers for Evaluate and Draft results."""
from __future__ import annotations

from .. import pilot
from ..evaluation_topics import topics_are_complete

import hashlib
import json
import re
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
    if not topics_are_complete(result):
        return []
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


def evaluation_input_key(statement: str, records: Any, witness: Any, statement_source: Any = None) -> str:
    """Session-only binding to exact raw inputs; malformed/legacy inputs fail closed."""
    from ..documents import ExtractedDocument
    from ..job_payload import documents_to_json, document_to_json

    if (not isinstance(statement, str) or not isinstance(records, list)
            or not all(isinstance(doc, ExtractedDocument) for doc in records)
            or not isinstance(witness, dict)):
        return ""
    try:
        payload = {"statement": statement.strip(), "records": documents_to_json(records),
                   "witness": witness}
        if statement_source is not None:
            if not isinstance(statement_source, ExtractedDocument):
                return ""
            payload["statement_source"] = document_to_json(statement_source)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, AttributeError):
        return ""
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _valid_input_key(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def remember_evaluation_inputs(input_key: str, *, source_id: str = "") -> None:
    """Bind questions from a returned direct result; discard prior-case state."""
    previous = st.session_state.get(_key("eval", "input_key"), "")
    if not _valid_input_key(input_key) or previous != input_key:
        for suffix in ("saved", "skipped", "applied_saved", "applied_skipped"):
            st.session_state[_key("eval", suffix)] = []
        st.session_state[_key("eval", "index")] = 0
        st.session_state[_key("eval", "notice")] = ""
    st.session_state[_key("eval", "input_key")] = input_key if _valid_input_key(input_key) else ""
    st.session_state[_key("eval", "input_source_id")] = source_id
    st.session_state[_key("eval", "questions_bound")] = _valid_input_key(input_key) and bool(source_id)


def append_follow_up_answers(text: str, *, slot: str, input_key: str | None = None) -> str:
    """Append accepted answers; evaluation reuse requires matching exact inputs."""
    if slot == "eval" and (not _valid_input_key(input_key)
            or st.session_state.get(_key(slot, "input_key"), "") != input_key
            or not st.session_state.get(_key(slot, "input_source_id"))):
        if _saved_answers(slot):
            pilot.display("Saved follow-up answers belong to different or unrecognized inputs "
                          "and were not included. Review the statement and collect new answers "
                          "from its evaluation.", container=st, method="warning")
        return text
    appendix = compose_follow_up_appendix(slot)
    if not appendix:
        return text
    base = text.rstrip()
    if not base:
        return appendix
    return f"{base}\n\n{appendix}"


def mark_follow_up_answers_consumed(slot: str) -> None:
    """Mark saved answers as consumed by a successful rerun."""
    saved = _saved_answers(slot)
    skipped = _saved_skips(slot)
    if saved:
        st.session_state[_key(slot, "applied_saved")] = saved
    if skipped:
        st.session_state[_key(slot, "applied_skipped")] = skipped
    st.session_state[_key(slot, "saved")] = []
    st.session_state[_key(slot, "skipped")] = []
    st.session_state[_key(slot, "index")] = 0
    st.session_state[_key(slot, "notice")] = ""


def render_follow_up_questions(
    *,
    slot: str,
    source_id: str,
    questions: list[dict[str, str]],
    empty_message: str,
    next_run_label: str,
) -> None:
    """Render one follow-up question at a time with accept/skip controls."""
    _ensure_state(slot, source_id)
    if slot == "eval" and questions and (
            not _valid_input_key(st.session_state.get(_key(slot, "input_key")))
            or st.session_state.get(_key(slot, "input_source_id")) != source_id
            or st.session_state.get(_key(slot, "questions_bound")) is not True):
        questions = []
        empty_message = (
            "New follow-up questions are unavailable for this saved or queued result. "
            "Run a direct evaluation to bind new answers to its inputs. Earlier saved "
            "answers remain available and can be reused only with their matching inputs."
        )
    questions = _filter_handled_questions(slot, questions)
    saved = _saved_answers(slot)
    skipped = _saved_skips(slot)
    applied_saved = _applied_saved_answers(slot)
    applied_skipped = _applied_skips(slot)
    index = int(st.session_state.get(_key(slot, "index"), 0) or 0)
    if index > len(questions):
        index = len(questions)
        st.session_state[_key(slot, "index")] = index

    with st.expander("🤖 Automated follow-up question generator",
                     expanded=bool(questions or saved or skipped or applied_saved or applied_skipped)):
        notice = _clean_text(st.session_state.get(_key(slot, "notice"), ""))
        if notice:
            pilot.display(notice, container=st, method="success")

        if saved:
            pilot.display(
                (f"{len(saved)} accepted answer(s) will be included only when the statement, "
                 "records and witness inputs match their source evaluation."
                 if slot == "eval" and _valid_input_key(st.session_state.get(_key(slot, "input_key")))
                 else "These saved answers are unbound and will not be included in another evaluation. "
                 "Clear them and collect new answers." if slot == "eval"
                 else f"{len(saved)} accepted answer(s) will be included automatically in the next "
                 f"{next_run_label} run.")
            , container=st, method="caption")
            for item in saved:
                topic = _clean_text(item.get("topic")) or "Follow-up"
                answer = _clean_text(item.get("answer"))
                pilot.display(f"- **{topic}** — {answer}", container=st, method="write")
        elif applied_saved:
            pilot.display(
                f"{len(applied_saved)} accepted answer(s) were already included in the most recent "
                f"{next_run_label} run."
            , container=st, method="caption")
            for item in applied_saved:
                topic = _clean_text(item.get("topic")) or "Follow-up"
                answer = _clean_text(item.get("answer"))
                pilot.display(f"- **{topic}** — {answer}", container=st, method="write")
        if applied_skipped and not skipped:
            pilot.display(f"Previously skipped questions in the last cycle: {len(applied_skipped)}", container=st, method="caption")
        if saved or skipped or applied_saved or applied_skipped:
            if st.button(
                "Clear saved follow-up answers and skipped questions",
                key=_key(slot, "clear"),
            ):
                st.session_state[_key(slot, "saved")] = []
                st.session_state[_key(slot, "skipped")] = []
                st.session_state[_key(slot, "applied_saved")] = []
                st.session_state[_key(slot, "applied_skipped")] = []
                st.session_state[_key(slot, "index")] = 0
                st.session_state[_key(slot, "notice")] = "Cleared the saved follow-up state for this tab."
                st.rerun()

        if not questions:
            pilot.display(empty_message, container=st, method="info")
            return

        if index >= len(questions):
            if skipped:
                pilot.display(f"Skipped questions: {len(skipped)}", container=st, method="caption")
            pilot.display("You have reviewed every generated follow-up question for this run.", container=st, method="success")
            return

        current = questions[index]
        pilot.display(f"Question {index + 1} of {len(questions)} — {current.get('topic', 'Checklist topic')}", container=st, method="caption")
        with st.form(key=_key(slot, f"form_{index}")):
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
            accept = st.form_submit_button(
                "Save answer and continue", type="primary"
            )
            skip = st.form_submit_button("Skip this question")

        if accept:
            cleaned_answer = _clean_text(answer_text)
            if not cleaned_answer:
                pilot.display("Enter the witness's answer before saving it, or skip this question.", container=st, method="warning")
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
        if not isinstance(row, dict) or row.get("applicable") is not True or row.get("covered") is not False:
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
        if not isinstance(row, dict) or row.get("applicable") is not True:
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


def _applied_saved_answers(slot: str) -> list[dict[str, str]]:
    raw = st.session_state.get(_key(slot, "applied_saved"), [])
    return list(raw) if isinstance(raw, list) else []


def _applied_skips(slot: str) -> list[dict[str, str]]:
    raw = st.session_state.get(_key(slot, "applied_skipped"), [])
    return list(raw) if isinstance(raw, list) else []


def _filter_handled_questions(slot: str, questions: list[dict[str, str]]) -> list[dict[str, str]]:
    handled = {
        (_clean_text(item.get("topic")), _clean_text(item.get("question")))
        for item in [*_saved_answers(slot), *_saved_skips(slot)]
        if isinstance(item, dict)
    }
    filtered: list[dict[str, str]] = []
    for item in questions:
        if not isinstance(item, dict):
            continue
        normalized = {
            "topic": _clean_text(item.get("topic")) or "Checklist topic",
            "question": _clean_text(item.get("question")),
        }
        if not normalized["question"]:
            continue
        if (normalized["topic"], normalized["question"]) in handled:
            continue
        filtered.append(normalized)
    return filtered


def _ensure_state(slot: str, source_id: str) -> None:
    source_key = _key(slot, "source_id")
    if source_key not in st.session_state:
        st.session_state[_key(slot, "source_id")] = source_id
        st.session_state[_key(slot, "index")] = 0
        for suffix in ("saved", "skipped", "applied_saved", "applied_skipped"):
            st.session_state.setdefault(_key(slot, suffix), [])
        st.session_state[_key(slot, "notice")] = ""
        return
    if st.session_state.get(source_key) != source_id:
        st.session_state[source_key] = source_id
        st.session_state[_key(slot, "index")] = 0
        # Saved answers are still pending until explicit consumption or clearing.
        # Direct evaluation reuse also requires the matching input binding.
        st.session_state[_key(slot, "notice")] = ""


def _key(slot: str, suffix: str) -> str:
    return f"{slot}_follow_up_{suffix}"


def _clean_text(value: Any) -> str:
    return str(value).strip() if isinstance(value, str) else str(value).strip() if value is not None else ""
