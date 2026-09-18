"""Offline tests for automated follow-up question helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app.draft import DraftResult  # noqa: E402
from app.evaluate import EvaluationResult  # noqa: E402


def _fake_streamlit() -> tuple[MagicMock, dict]:
    st_mock = MagicMock()

    class _SessionState(dict):
        def __getattr__(self, name: str):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name: str, value) -> None:
            self[name] = value

    session: dict = _SessionState()
    st_mock.session_state = session
    return st_mock, session


class TestFollowUpQuestionExtraction(unittest.TestCase):
    def test_draft_extracts_only_uncovered_applicable_topics(self) -> None:
        from app.views.follow_up import draft_follow_up_questions

        result = DraftResult(
            grounding={
                "topic_coverage": [
                    {"topic": "A. Hazards", "applicable": True, "covered": True, "prompt_for_witness": ""},
                    {"topic": "B. Caregiver Burden", "applicable": True, "covered": False, "prompt_for_witness": "Who helps each day?"},
                    {"topic": "C. Personal Care", "applicable": False, "covered": False, "prompt_for_witness": "Ignored"},
                ]
            }
        )
        self.assertEqual(
            draft_follow_up_questions(result),
            [{"topic": "B. Caregiver Burden", "question": "Who helps each day?"}],
        )

    def test_evaluate_extracts_partial_and_absent_topics(self) -> None:
        from app.views.follow_up import evaluate_follow_up_questions

        result = EvaluationResult(
            topic_focus="PTSD caregiver support",
            topic_rows=[
                {"topic": "A. Hazards", "applicable": True, "coverage": "covered", "gap_note": "ignored"},
                {"topic": "B. Caregiver Burden", "applicable": True, "coverage": "partial", "gap_note": "Describe what happens if you are away for a day"},
                {"topic": "C. Personal Care", "applicable": True, "coverage": "absent", "gap_note": ""},
                {"topic": "D. Other", "applicable": False, "coverage": "absent", "gap_note": "ignored"},
            ],
        )
        questions = evaluate_follow_up_questions(result)
        self.assertEqual(len(questions), 2)
        self.assertEqual(questions[0]["topic"], "B. Caregiver Burden")
        self.assertIn("PTSD caregiver support", questions[0]["question"])
        self.assertEqual(questions[1]["topic"], "C. Personal Care")
        self.assertIn("What has the witness personally observed", questions[1]["question"])

    def test_malformed_topic_data_is_ignored_safely(self) -> None:
        from app.views.follow_up import draft_follow_up_questions, evaluate_follow_up_questions

        draft_result = DraftResult(
            grounding={"topic_coverage": ["bad", None, {"topic": "A", "applicable": True, "covered": False}]}
        )
        eval_result = EvaluationResult(
            topic_rows=["bad", {"topic": "A", "applicable": True, "coverage": None, "gap_note": None}]
        )

        self.assertEqual(
            draft_follow_up_questions(draft_result),
            [{"topic": "A", "question": "What has the witness personally observed about A that should be added if true?"}],
        )
        self.assertEqual(
            evaluate_follow_up_questions(eval_result),
            [{"topic": "A", "question": "What has the witness personally observed about A that should be added if true?"}],
        )


class TestFollowUpStateAndRendering(unittest.TestCase):
    def test_saved_answers_are_appended_for_next_run(self) -> None:
        import app.views.follow_up as follow_up

        st_mock, session = _fake_streamlit()
        session["draft_follow_up_saved"] = [
            {"topic": "B. Caregiver Burden", "question": "Who helps?", "answer": "I help him shower and dress each morning."}
        ]
        with patch.object(follow_up, "st", st_mock):
            text = follow_up.append_follow_up_answers("Original observations.", slot="draft")
        self.assertIn("Original observations.", text)
        self.assertIn("Additional follow-up details confirmed after the last run:", text)
        self.assertIn("I help him shower and dress each morning.", text)

    def test_accept_advances_to_next_question_and_saves_edits(self) -> None:
        import app.views.follow_up as follow_up

        st_mock, session = _fake_streamlit()
        st_mock.text_area.side_effect = ["Edited question?", "Edited answer."]
        st_mock.form_submit_button.side_effect = [True, False]
        with patch.object(follow_up, "st", st_mock):
            follow_up.render_follow_up_questions(
                slot="eval",
                source_id="req_1",
                questions=[
                    {"topic": "A. Hazards", "question": "Original question?"},
                    {"topic": "B. Caregiver Burden", "question": "Second question?"},
                ],
                empty_message="none",
                next_run_label="evaluation",
            )
        self.assertEqual(session["eval_follow_up_index"], 1)
        self.assertEqual(
            session["eval_follow_up_saved"],
            [{"topic": "A. Hazards", "question": "Edited question?", "answer": "Edited answer."}],
        )
        st_mock.rerun.assert_called_once()

    def test_skip_advances_without_saving_answer(self) -> None:
        import app.views.follow_up as follow_up

        st_mock, session = _fake_streamlit()
        session["draft_follow_up_source_id"] = "req_1"
        st_mock.text_area.side_effect = ["Question one?", ""]
        st_mock.form_submit_button.side_effect = [False, True]
        with patch.object(follow_up, "st", st_mock):
            follow_up.render_follow_up_questions(
                slot="draft",
                source_id="req_1",
                questions=[{"topic": "A. Hazards", "question": "Question one?"}],
                empty_message="none",
                next_run_label="draft",
            )
        self.assertEqual(session["draft_follow_up_index"], 1)
        self.assertEqual(session.get("draft_follow_up_saved", []), [])
        self.assertEqual(
            session["draft_follow_up_skipped"],
            [{"topic": "A. Hazards", "question": "Question one?"}],
        )
        st_mock.rerun.assert_called_once()

    def test_no_questions_renders_empty_state(self) -> None:
        import app.views.follow_up as follow_up

        st_mock, _session = _fake_streamlit()
        with patch.object(follow_up, "st", st_mock):
            follow_up.render_follow_up_questions(
                slot="eval",
                source_id="req_2",
                questions=[],
                empty_message="No follow-up questions needed.",
                next_run_label="evaluation",
            )
        st_mock.info.assert_called_once_with("No follow-up questions needed.")

    def test_new_source_clears_consumed_answers(self) -> None:
        import app.views.follow_up as follow_up

        st_mock, session = _fake_streamlit()
        session["eval_follow_up_source_id"] = "req_old"
        session["eval_follow_up_saved"] = [
            {"topic": "A. Hazards", "question": "Q?", "answer": "A."}
        ]
        session["eval_follow_up_pending_apply"] = False
        with patch.object(follow_up, "st", st_mock):
            follow_up.render_follow_up_questions(
                slot="eval",
                source_id="req_new",
                questions=[],
                empty_message="No follow-up questions needed.",
                next_run_label="evaluation",
            )
        self.assertEqual(session["eval_follow_up_saved"], [])

    def test_successful_consumption_clears_saved_follow_up_state(self) -> None:
        import app.views.follow_up as follow_up

        st_mock, session = _fake_streamlit()
        session["eval_follow_up_saved"] = [
            {"topic": "A. Hazards", "question": "Q?", "answer": "A."}
        ]
        session["eval_follow_up_skipped"] = [
            {"topic": "B. Caregiver Burden", "question": "Skipped?"}
        ]
        with patch.object(follow_up, "st", st_mock):
            follow_up.mark_follow_up_answers_consumed("eval")
        self.assertEqual(session["eval_follow_up_saved"], [])
        self.assertEqual(
            session["eval_follow_up_applied_saved"],
            [{"topic": "A. Hazards", "question": "Q?", "answer": "A."}],
        )
        self.assertEqual(
            session["eval_follow_up_applied_skipped"],
            [{"topic": "B. Caregiver Burden", "question": "Skipped?"}],
        )


class TestFollowUpRunIntegration(unittest.TestCase):
    def test_evaluation_flow_appends_saved_answers(self) -> None:
        import app.views.evaluate_view as evaluate_view
        import app.views.follow_up as follow_up

        st_mock, session = _fake_streamlit()
        session["eval_follow_up_saved"] = [
            {"topic": "A. Hazards", "question": "What happened near the stove?", "answer": "He left the burner on twice last month."}
        ]
        llm = MagicMock()
        llm.usage.totals.return_value = type("Totals", (), {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})()
        with (
            patch.object(follow_up, "st", st_mock),
            patch.object(evaluate_view, "st", st_mock),
            patch.object(evaluate_view, "new_run_request_id", return_value="req_eval_1"),
            patch.object(evaluate_view, "get_llm", return_value=llm),
            patch.object(evaluate_view, "check_shutdown_gate", return_value=True),
            patch.object(evaluate_view, "enter_run", return_value=True),
            patch.object(evaluate_view, "exit_run"),
            patch.object(evaluate_view, "progress_widgets", return_value=(MagicMock(), lambda *a, **k: None)),
            patch.object(evaluate_view, "check_memory_before_run"),
            patch.object(
                evaluate_view,
                "run_with_timeout",
                return_value=EvaluationResult(claims=[{"id": 1}], scores={"factual_accuracy": 7.0}),
            ) as run_with_timeout,
            patch.object(evaluate_view, "get_profiler", return_value=None),
            patch.object(evaluate_view, "audit_record_meta", return_value=(["Upload"], 1, 1)),
            patch.object(evaluate_view, "audit_condition_for_slot", return_value=""),
            patch.object(evaluate_view, "record_watchdog_run"),
            patch.object(evaluate_view, "audit_log"),
            patch.object(evaluate_view, "run_log_event"),
        ):
            evaluate_view._run_evaluation_flow("Original statement.", [MagicMock(pages=["p1"])])

        appended_statement = run_with_timeout.call_args.args[2]
        self.assertIn("Original statement.", appended_statement)
        self.assertIn("Additional follow-up details confirmed after the last run:", appended_statement)
        self.assertIn("He left the burner on twice last month.", appended_statement)
        self.assertEqual(session["eval_follow_up_saved"], [])
        self.assertEqual(
            session["eval_follow_up_applied_saved"],
            [
                {
                    "topic": "A. Hazards",
                    "question": "What happened near the stove?",
                    "answer": "He left the burner on twice last month.",
                }
            ],
        )

    def test_draft_flow_appends_saved_answers(self) -> None:
        import app.views.draft_view as draft_view
        import app.views.follow_up as follow_up

        st_mock, session = _fake_streamlit()
        session["draft_follow_up_saved"] = [
            {"topic": "B. Caregiver Burden", "question": "Who helps him bathe?", "answer": "I help him shower every morning."}
        ]
        llm = MagicMock()
        llm.usage.totals.return_value = type("Totals", (), {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})()
        with (
            patch.object(follow_up, "st", st_mock),
            patch.object(draft_view, "st", st_mock),
            patch.object(draft_view, "get_llm", return_value=llm),
            patch.object(draft_view, "check_shutdown_gate", return_value=True),
            patch.object(draft_view, "enter_run", return_value=True),
            patch.object(draft_view, "exit_run"),
            patch.object(draft_view, "progress_widgets", return_value=(MagicMock(), lambda *a, **k: None)),
            patch.object(draft_view, "check_memory_before_run"),
            patch.object(draft_view, "run_with_timeout", return_value=DraftResult(draft="draft text")) as run_with_timeout,
            patch.object(draft_view, "get_profiler", return_value=None),
            patch.object(draft_view, "audit_record_meta", return_value=(["Upload"], 1, 1)),
            patch.object(draft_view, "audit_condition_for_slot", return_value=""),
            patch.object(draft_view, "record_watchdog_run"),
            patch.object(draft_view, "audit_log"),
            patch.object(draft_view, "run_log_event"),
        ):
            draft_view._run_draft_flow(
                rid="req_draft_1",
                records=[MagicMock(pages=["p1"])],
                condition="PTSD",
                claim_type="Service connection",
                relationship="Spouse",
                witness_name="Jane Doe",
                veteran_name="John Doe",
                known_since="2010",
                contact_frequency="daily",
                witnessed_event="No",
                observations="Original observations.",
            )

        appended_observations = run_with_timeout.call_args.args[4]
        self.assertIn("Original observations.", appended_observations)
        self.assertIn("Additional follow-up details confirmed after the last run:", appended_observations)
        self.assertIn("I help him shower every morning.", appended_observations)
        self.assertEqual(session["draft_follow_up_saved"], [])
        self.assertEqual(
            session["draft_follow_up_applied_saved"],
            [
                {
                    "topic": "B. Caregiver Burden",
                    "question": "Who helps him bathe?",
                    "answer": "I help him shower every morning.",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
