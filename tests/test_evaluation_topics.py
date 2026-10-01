"""Synthetic R03 regression checks: topic coverage cannot silently become complete."""
from copy import deepcopy
import json
import unittest
from unittest.mock import MagicMock, patch

from tests import hermetic  # noqa: F401
from tests.topic_fixtures import complete_topics, topic_result_fields
from tests.rubric_fixtures import scored_result_fields
from tests.test_evaluate import _FakeLLM, _fake_digest, _doc
from tests.test_views import _fake_streamlit, _patch_st
from app.evaluate import EvaluationResult, run_evaluation, build_report, evaluation_report_markdown, _score_and_recommend, _draft_revision
from app.evaluation_topics import TOPIC_POLICY, TOPIC_ORDER, TOPIC_MAX_ATTEMPTS, TOPIC_INCOMPLETE_NOTICE, TOPIC_UNVALIDATED_NOTICE, TopicValidationError, normalize_topics, topics_are_complete, evaluation_is_complete
from app.job_payload import evaluation_to_json, evaluation_from_json, RunResult, encode_result, decode_result
from app.llm import LLMError, LLMParseError, LLMAuthError, LLMConfigurationError
from app.pipeline_guard import PipelineCancelledError, PipelineTimeoutError
from app.rubric_validation import rubric_is_complete
from app.usage import UsageTracker


def valid_result():
    return EvaluationResult(**scored_result_fields(), **topic_result_fields(),
        claims=[{"id": 1, "text": "Synthetic witnessed symptom."}],
        verifications=[{"id": 1, "verdict": "NOT FOUND", "record_reference": "", "note": "Retained finding."}],
        revised_statement="PRIVATE_REWRITE_SENTINEL", revision_notes="PRIVATE_REVISION_SENTINEL",
        recommendations=[{"title": "PRIVATE_RECOMMENDATION_SENTINEL"}],
        report_citations=[{"source": "synthetic.pdf p.1", "excerpt": "Retained synthetic source."}])


class TestTopicResponse(unittest.TestCase):
    def test_valid_complete_response_is_sorted_and_retains_gaps(self):
        raw = complete_topics(); raw["topics"].reverse()
        result = normalize_topics(raw)
        self.assertEqual([row["topic"][0] for row in result["topics"]], list(TOPIC_ORDER))
        self.assertEqual(result["critical_gaps"], raw["critical_gaps"])
        self.assertIn("near-miss", result["topics"][0]["gap_note"])

    def test_documented_checklist_headings_and_bare_letters_work(self):
        from app.config import load_knowledge
        from app.knowledge_currency import parse_topic_sections
        raw = complete_topics()
        headings = parse_topic_sections(load_knowledge("topic_checklist.md"))
        for row, section in zip(raw["topics"], headings):
            row["topic"] = f"{section.letter}. {section.title}"
        self.assertEqual(len(normalize_topics(raw)["topics"]), 15)
        for row in raw["topics"]:
            row["topic"] = row["topic"][0]
        self.assertEqual(len(normalize_topics(raw)["topics"]), 15)

    def test_missing_duplicate_and_extra_topics_are_rejected(self):
        base = complete_topics()
        for rows in ([], base["topics"][:-1], base["topics"] + [base["topics"][0]],
                     [base["topics"][0]] + base["topics"][:-1]):
            with self.subTest(length=len(rows)), self.assertRaises(TopicValidationError):
                normalize_topics({**base, "topics": rows})

    def test_invented_unknown_and_ambiguous_labels_are_rejected(self):
        for label in ("P", "a", "AA", "A/B", "A. Made up", "Topic A", None, 1):
            raw = complete_topics(); raw["topics"][0]["topic"] = label
            with self.subTest(label=label), self.assertRaises(TopicValidationError):
                normalize_topics(raw)

    def test_boolean_strings_and_numbers_never_count_as_applicable(self):
        for value in ("false", "true", 0, 1, None, [], {}):
            raw = complete_topics(); raw["topics"][0]["applicable"] = value
            with self.subTest(value=value), self.assertRaises(TopicValidationError):
                normalize_topics(raw)

    def test_only_the_documented_coverage_states_are_accepted(self):
        for value in ("complete", "Covered", "missing", "N/A", "", None, True, 1):
            raw = complete_topics(); raw["topics"][0]["coverage"] = value
            with self.subTest(value=value), self.assertRaises(TopicValidationError):
                normalize_topics(raw)

    def test_inapplicable_topics_cannot_be_marked_covered_partial_or_absent(self):
        for coverage in ("covered", "partial", "absent"):
            raw = complete_topics(); raw["topics"][1]["coverage"] = coverage
            with self.subTest(coverage=coverage), self.assertRaises(TopicValidationError):
                normalize_topics(raw)
        raw = complete_topics(); raw["topics"][0]["coverage"] = "not applicable"
        with self.assertRaises(TopicValidationError): normalize_topics(raw)

    def test_partial_and_absent_rows_need_non_placeholder_gap_wording(self):
        for coverage in ("partial", "absent"):
            for gap in ("", " ", "none", "n/a", "unknown", "tbd", "not applicable", "1234", True, None):
                raw = complete_topics(); raw["topics"][0].update(coverage=coverage, gap_note=gap,
                    evidence="Synthetic detail." if coverage == "partial" else "")
                with self.subTest(coverage=coverage, gap=gap), self.assertRaises(TopicValidationError):
                    normalize_topics(raw)

    def test_evidence_and_gaps_must_match_the_coverage(self):
        for changes in ({"coverage": "covered"}, {"evidence": ""}, {"coverage": "absent"},
                        {"gap_note": []}, {"evidence": []}):
            raw = complete_topics(); raw["topics"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(TopicValidationError): normalize_topics(raw)
        raw = complete_topics(); raw["topics"][1]["evidence"] = "Unexpected evidence."
        with self.assertRaises(TopicValidationError): normalize_topics(raw)
        raw = complete_topics(); raw["topics"][1]["gap_note"] = "Unexpected gap."
        with self.assertRaises(TopicValidationError): normalize_topics(raw)

    def test_critical_gaps_reference_distinct_weak_applicable_topics(self):
        for gaps in (["B. Not applicable"], ["P. Unknown"], ["A. Gap", "A. Duplicate"],
                     [None], "A. Gap", ["Unlabeled gap"], ["A. Gap"] * 6):
            raw = complete_topics(); raw["critical_gaps"] = gaps
            with self.subTest(gaps=gaps), self.assertRaises(TopicValidationError): normalize_topics(raw)
        raw = complete_topics(); raw["topics"][0].update(coverage="covered", gap_note="")
        with self.assertRaises(TopicValidationError): normalize_topics(raw)

    def test_raw_schema_fields_are_not_coerced_or_filtered(self):
        base = complete_topics()
        for field in base:
            raw = deepcopy(base); del raw[field]
            with self.subTest(field=field), self.assertRaises(TopicValidationError): normalize_topics(raw)
        for raw in (None, [], {**base, "extra": "PRIVATE_SENTINEL"}, {**base, "claim_focus": ""},
                    {**base, "notes": 1}, {**base, "topics": base["topics"][:-1] + [None]}):
            with self.assertRaises(TopicValidationError): normalize_topics(raw)
        for field in base["topics"][0]:
            raw = deepcopy(base); del raw["topics"][0][field]
            with self.assertRaises(TopicValidationError): normalize_topics(raw)
        raw = deepcopy(base); raw["topics"][0]["extra"] = True
        with self.assertRaises(TopicValidationError): normalize_topics(raw)


class TestTopicPipeline(unittest.TestCase):
    def _run(self, response, progress=None):
        llm = _FakeLLM(overrides={"topic": response})
        with patch("app.evaluate.review_medical_records", return_value=_fake_digest()), patch("app.evaluate.load_knowledge", return_value="Synthetic checklist"):
            result = run_evaluation(llm, "Synthetic witnessed symptoms.", [_doc()], progress=progress)
        return result, llm

    def test_false_string_cannot_produce_a_completed_evaluation(self):
        raw = complete_topics(); raw["topics"][0]["applicable"] = "false"
        progress = MagicMock(); result, llm = self._run(raw, progress)
        self.assertFalse(evaluation_is_complete(result))
        self.assertTrue(rubric_is_complete(result))
        self.assertEqual(result.topic_status, "incomplete")
        self.assertEqual(result.topic_rows, [])
        self.assertEqual(llm.calls.count(("chat_json", "topic")), TOPIC_MAX_ATTEMPTS)
        self.assertIsNotNone(result.digest)
        self.assertEqual(len(result.claims), 2)
        self.assertEqual(len(result.verifications), 2)
        self.assertEqual(result.overall_rating, "Adequate")
        self.assertIsInstance(result.effectiveness_score, int)
        self.assertEqual(result.revised_statement, "")
        self.assertEqual(result.recommendations, [])
        for phase in ("revision", "recommendations"):
            self.assertNotIn(("chat_json", phase), llm.calls)
        self.assertIn(TOPIC_INCOMPLETE_NOTICE, result.report_markdown)
        self.assertNotIn("## Topic Coverage", result.report_markdown)
        self.assertIn("Claim-by-Claim Verification", result.report_markdown)
        self.assertIn("Partial evaluation", progress.call_args.args[1])

    def test_retry_recovers_only_with_a_complete_valid_response(self):
        prompts = []
        def response(_system, user, _kwargs):
            prompts.append(user)
            return {"secret": "PRIVATE_RESPONSE_SENTINEL"} if len(prompts) == 1 else complete_topics()
        result, llm = self._run(response)
        self.assertTrue(evaluation_is_complete(result))
        self.assertEqual(len(prompts), 2)
        self.assertNotIn("PRIVATE_RESPONSE_SENTINEL", prompts[1] + result.report_markdown)
        self.assertIn(("chat_json", "revision"), llm.calls)
        self.assertIn(("chat_json", "recommendations"), llm.calls)

    def test_parse_and_provider_failures_are_bounded_and_private(self):
        for error in (LLMParseError("PRIVATE_RESPONSE_SENTINEL"), LLMError("PRIVATE_RESPONSE_SENTINEL")):
            with self.subTest(type=type(error).__name__), self.assertLogs("app.evaluate", level="WARNING") as logs:
                result, llm = self._run(error)
            self.assertEqual(llm.calls.count(("chat_json", "topic")), 3)
            self.assertNotIn("PRIVATE_RESPONSE_SENTINEL", " ".join(logs.output) + result.report_markdown)

    def test_permanent_provider_errors_stop_immediately(self):
        for error in (LLMAuthError("Synthetic rejection", status_code=401), LLMConfigurationError("Synthetic configuration")):
            llm = _FakeLLM(overrides={"topic": error})
            with patch("app.evaluate.review_medical_records", return_value=_fake_digest()), patch("app.evaluate.load_knowledge", return_value="Synthetic checklist"):
                with self.assertRaises(type(error)): run_evaluation(llm, "Synthetic statement.", [_doc()])
            self.assertEqual(llm.calls.count(("chat_json", "topic")), 1)
            self.assertNotIn(("chat_json", "revision"), llm.calls)

    def test_cancellation_and_timeout_are_never_swallowed(self):
        for error in (PipelineCancelledError(), PipelineTimeoutError(2, 1)):
            with self.subTest(type=type(error).__name__), self.assertRaises(type(error)):
                self._run(lambda *_: (_ for _ in ()).throw(error))

    def test_direct_dependent_phase_calls_cannot_bypass_the_gate(self):
        result = valid_result(); result.topic_rows.pop()
        llm = _FakeLLM()
        _draft_revision(llm, result, "Synthetic statement.", MagicMock())
        _score_and_recommend(llm, result, MagicMock())
        self.assertEqual(llm.calls, [])
        self.assertEqual(result.revised_statement, "")
        self.assertEqual(result.recommendations, [])
        self.assertIsInstance(result.effectiveness_score, int)


class TestTopicSavedResults(unittest.TestCase):
    def test_complete_results_survive_json_and_queue_roundtrips(self):
        result = valid_result()
        restored = evaluation_from_json(json.loads(json.dumps(evaluation_to_json(result), allow_nan=False)))
        self.assertTrue(evaluation_is_complete(restored))
        self.assertEqual(restored.topic_rows, result.topic_rows)
        self.assertEqual(restored.revised_statement, result.revised_statement)
        queued = decode_result(encode_result(RunResult(kind="evaluate", result=result, usage=UsageTracker(), request_id="synthetic-r03"))).result
        self.assertTrue(topics_are_complete(queued))

    def test_saved_flags_are_required_even_with_valid_rows(self):
        for flags in ({"topic_policy": ""}, {"topic_policy": "future"}, {"topic_status": ""}, {"topic_status": "incomplete"}):
            restored = evaluation_from_json({**evaluation_to_json(valid_result()), **flags})
            self.assertFalse(topics_are_complete(restored))
            self.assertEqual(restored.topic_rows, [])
            self.assertEqual(restored.revised_statement, "")
            self.assertEqual(restored.recommendations, [])
            self.assertTrue(rubric_is_complete(restored))

    def test_raw_saved_rows_are_validated_before_filtering(self):
        for value in ("false", 0, 1, None):
            payload = evaluation_to_json(valid_result()); payload["topic_rows"][0]["applicable"] = value
            restored = evaluation_from_json(payload)
            self.assertEqual(restored.topic_status, "invalid")
            self.assertEqual(restored.topic_rows, [])
        for field, value in (("topic_rows", []), ("topic_rows", complete_topics()["topics"] + [None]),
                             ("topic_critical_gaps", [None]), ("topic_notes", 1), ("topic_focus", [])):
            restored = evaluation_from_json({**evaluation_to_json(valid_result()), field: value})
            self.assertEqual(restored.topic_status, "invalid")
            self.assertEqual(restored.revised_statement, "")

    def test_invalid_cached_report_and_dependent_outputs_are_withheld(self):
        result = valid_result(); result.topic_rows[0]["applicable"] = "false"
        result.report_markdown = "ALL_COVERED_SENTINEL\nPRIVATE_REWRITE_SENTINEL"
        report = evaluation_report_markdown(result)
        self.assertIn(TOPIC_UNVALIDATED_NOTICE, report)
        self.assertIn("Retained synthetic source.", report)
        self.assertIn("Synthetic witnessed symptom.", report)
        payload = evaluation_to_json(result)
        for rendered in (report, payload["report_markdown"]):
            for text in ("ALL_COVERED_SENTINEL", "PRIVATE_REWRITE_SENTINEL", "PRIVATE_RECOMMENDATION_SENTINEL", "## Topic Coverage", "Proposed Rewrite"):
                self.assertNotIn(text, rendered)
        self.assertEqual(payload["topic_rows"], [])
        self.assertEqual(payload["revised_statement"], "")
        self.assertEqual(payload["recommendations"], [])
        self.assertEqual(len(payload["scores"]), 8)

    def test_partial_topic_failure_survives_queue_with_valid_review(self):
        result = valid_result(); result.topic_status = "incomplete"
        restored = decode_result(encode_result(RunResult(kind="evaluate", result=result, usage=UsageTracker(), request_id="synthetic-r03"))).result
        self.assertEqual(restored.topic_status, "incomplete")
        self.assertFalse(evaluation_is_complete(restored))
        self.assertTrue(rubric_is_complete(restored))
        self.assertEqual(restored.claims, result.claims)
        self.assertIn(TOPIC_INCOMPLETE_NOTICE, evaluation_report_markdown(restored))

    def test_worker_outcome_and_pilot_metadata_classify_topic_failure(self):
        from app.worker import _outcome_for
        from app.pilot import safe_metadata
        result = valid_result(); result.topic_rows.pop()
        outcome = _outcome_for("evaluate", result)
        self.assertEqual(outcome["topic_status"], "incomplete")
        self.assertEqual(outcome["scoring_status"], "complete")
        self.assertEqual(safe_metadata({"status": "partial", **outcome})["topic_status"], "incomplete")
        self.assertNotIn("topic_status", safe_metadata({"topic_status": "PRIVATE_SENTINEL"}))


class TestTopicUI(unittest.TestCase):
    def test_invalid_topics_have_no_count_table_followups_or_rewrite(self):
        from app.views import evaluate_view as view
        result = valid_result(); result.topic_rows[0]["applicable"] = "false"
        st_mock, _ = _fake_streamlit(); columns = tuple(MagicMock() for _ in range(4))
        st_mock.columns.return_value = columns
        with _patch_st(view, st_mock), patch.object(view, "render_follow_up_questions") as followups:
            view._render_evaluation_results(result)
        columns[3].metric.assert_called_once_with("Topics covered", "—")
        self.assertIn(TOPIC_UNVALIDATED_NOTICE, [c.args[0] for c in st_mock.warning.call_args_list])
        followups.assert_called_once()
        self.assertEqual(followups.call_args.kwargs["questions"], [])
        self.assertIn("unavailable", followups.call_args.kwargs["empty_message"])
        st_mock.text_area.assert_not_called()
        self.assertFalse(any("Topic coverage" in str(c) for c in st_mock.expander.call_args_list))
        rendered = str(st_mock.mock_calls)
        self.assertNotIn("PRIVATE_RECOMMENDATION_SENTINEL", rendered)
        reports = [c.kwargs["data"].decode() for c in st_mock.download_button.call_args_list if c.kwargs.get("file_name") == "lay_statement_evaluation.md"]
        self.assertIn(TOPIC_UNVALIDATED_NOTICE, reports[0])
        self.assertNotIn("PRIVATE_REWRITE_SENTINEL", reports[0])

    def test_pending_answers_survive_partial_reference_changes_and_remain_clearable(self):
        from app.views import evaluate_view as view, follow_up
        for old_source in (None, "req_old"):
            for clear in (False, True):
                with self.subTest(old_source=old_source, clear=clear):
                    result = valid_result(); result.topic_rows.pop()
                    st_mock, session = _fake_streamlit()
                    st_mock.columns.return_value = tuple(MagicMock() for _ in range(4))
                    session["eval_request_id"] = "req_new"
                    session["eval_follow_up_saved"] = [{"topic": "A", "question": "What happened?", "answer": "Synthetic pending answer."}]
                    if old_source is not None: session["eval_follow_up_source_id"] = old_source
                    st_mock.button.side_effect = lambda label, **kw: clear if kw.get("key") == "eval_follow_up_clear" else False
                    with _patch_st(view, st_mock), _patch_st(follow_up, st_mock):
                        view._render_evaluation_results(result)
                        next_input = follow_up.append_follow_up_answers("Changed synthetic statement.", slot="eval")
                    st_mock.button.assert_any_call("Clear saved follow-up answers and skipped questions", key="eval_follow_up_clear")
                    self.assertTrue(any("Synthetic pending answer." in str(c) for c in st_mock.write.call_args_list))
                    st_mock.form.assert_not_called()
                    if clear:
                        self.assertEqual(session["eval_follow_up_saved"], [])
                        self.assertNotIn("Synthetic pending answer.", next_input)
                        st_mock.rerun.assert_called_once()
                    else:
                        self.assertIn("Synthetic pending answer.", next_input)
                        self.assertEqual(len(session["eval_follow_up_saved"]), 1)

    def test_all_handled_questions_still_show_pending_answers_and_clear_control(self):
        from app.views import follow_up
        result = valid_result()
        question = follow_up.evaluate_follow_up_questions(result)[0]
        st_mock, session = _fake_streamlit()
        session["eval_follow_up_source_id"] = "req_current"
        session["eval_follow_up_saved"] = [{**question, "answer": "Synthetic accepted answer."}]
        st_mock.button.return_value = False
        with _patch_st(follow_up, st_mock):
            follow_up.render_follow_up_questions(slot="eval", source_id="req_current",
                questions=[question], empty_message="No remaining questions.", next_run_label="evaluation")
        st_mock.button.assert_called_once_with("Clear saved follow-up answers and skipped questions", key="eval_follow_up_clear")
        self.assertTrue(any("Synthetic accepted answer." in str(c) for c in st_mock.write.call_args_list))
        st_mock.form.assert_not_called()

    def test_validated_empty_followups_never_claim_all_topics_covered(self):
        from app.views import evaluate_view as view
        result = valid_result(); st_mock, _ = _fake_streamlit()
        st_mock.columns.side_effect = lambda count: tuple(MagicMock() for _ in range(count))
        with _patch_st(view, st_mock), patch.object(view, "render_follow_up_questions") as followups, patch.object(view, "_render_framework_currency_flags"):
            view._render_evaluation_results(result)
        message = followups.call_args.kwargs["empty_message"]
        self.assertIn("saved or skipped", message)
        self.assertNotIn("every applicable", message)

    def test_unvalidated_rows_never_generate_followup_questions(self):
        from app.views.follow_up import evaluate_follow_up_questions
        result = valid_result(); result.topic_policy = ""
        self.assertEqual(evaluate_follow_up_questions(result), [])
        result.topic_policy = TOPIC_POLICY; result.topic_rows[0]["applicable"] = "false"
        self.assertEqual(evaluate_follow_up_questions(result), [])

    def test_partial_direct_run_logs_topic_failure_and_keeps_answers_pending(self):
        from app.views import evaluate_view as view
        from tests.test_views import TestEvaluateRunBookkeeping
        result = valid_result(); result.topic_rows.pop()
        with patch.object(view, "mark_follow_up_answers_consumed") as consume:
            events, raised = TestEvaluateRunBookkeeping()._drive(result=result)
        self.assertIsNone(raised)
        partial = next(details for _, status, details in events if status == "partial")
        self.assertEqual(partial["scoring_status"], "complete")
        self.assertEqual(partial["topic_status"], "incomplete")
        self.assertNotIn("ok", [status for _, status, _ in events])
        consume.assert_not_called()

    def test_result_derived_currency_topics_require_validation(self):
        from app.views import evaluate_view as view
        result = valid_result(); st_mock, _ = _fake_streamlit()
        with _patch_st(view, st_mock):
            self.assertEqual(view._case_topic_letters(result), ["A"])
            result.topic_rows[0]["applicable"] = "false"
            self.assertEqual(view._case_topic_letters(result), [])
