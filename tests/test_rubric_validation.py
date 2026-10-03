"""Synthetic regression coverage for R02: complete scoring or an explicit partial result."""
from copy import deepcopy
import json
import unittest
from unittest.mock import MagicMock, patch

from tests import hermetic  # noqa: F401
from tests.rubric_fixtures import complete_rubric, scored_result_fields
from tests.topic_fixtures import topic_result_fields
from tests.test_evaluate import _FakeLLM, _fake_digest, _doc
from tests.test_views import _fake_streamlit, _patch_st
from app.evaluate import EvaluationResult, run_evaluation, build_report, evaluation_report_markdown, compute_effectiveness_score, _score_and_recommend
from app.job_payload import evaluation_from_json, evaluation_to_json, RunResult, encode_result, decode_result
from app.llm import LLMParseError, LLMError, LLMAuthError, LLMConfigurationError
from app.pipeline_guard import PipelineCancelledError, PipelineTimeoutError
from app.rubric_validation import (
    DIMENSION_LABELS, RUBRIC_POLICY, RUBRIC_MAX_ATTEMPTS, RUBRIC_INCOMPLETE_NOTICE,
    RUBRIC_UNVALIDATED_NOTICE, RubricValidationError, normalize_rubric, rubric_is_complete,
)


class TestRubricSchema(unittest.TestCase):
    def test_valid_integer_and_fractional_boundaries(self):
        for value in (0, 10, 0.0, 10.0, 4.25):
            with self.subTest(value=value):
                result = normalize_rubric(complete_rubric({key: value for key in DIMENSION_LABELS}))
                self.assertEqual(set(result["scores"].values()), {float(value)})

    def test_invalid_numbers_are_rejected_without_clamping(self):
        for value in (-1, 10.0001, 999, 10**1000, True, False, "9", None,
                      float("nan"), float("inf"), -float("inf"), [], {}):
            with self.subTest(type=type(value).__name__):
                raw = complete_rubric()
                raw["scores"]["factual_accuracy"] = value
                with self.assertRaises(RubricValidationError):
                    normalize_rubric(raw)

    def test_exact_dimensions_required_for_scores_and_rationales(self):
        for field in ("scores", "rationales"):
            for mutation in ("missing", "unknown", "nonobject"):
                with self.subTest(field=field, mutation=mutation):
                    raw = complete_rubric()
                    if mutation == "missing":
                        del raw[field]["factual_accuracy"]
                    elif mutation == "unknown":
                        raw[field]["invented_dimension"] = 9
                    else:
                        raw[field] = []
                    with self.assertRaises(RubricValidationError):
                        normalize_rubric(raw)

    def test_every_rationale_is_a_nonempty_string(self):
        for value in ("", " \n ", None, 7, True, [], {}):
            raw = complete_rubric()
            raw["rationales"]["factual_accuracy"] = value
            with self.subTest(value=value), self.assertRaises(RubricValidationError):
                normalize_rubric(raw)

    def test_root_requires_complete_schema(self):
        for field in complete_rubric():
            raw = complete_rubric()
            del raw[field]
            with self.subTest(field=field), self.assertRaises(RubricValidationError):
                normalize_rubric(raw)
        for raw in (None, [], "private medical text", {**complete_rubric(), "other": "private text"}):
            with self.assertRaises(RubricValidationError):
                normalize_rubric(raw)

    def test_improvement_rows_have_required_types(self):
        original = complete_rubric()["improvements"][0]
        bad_rows = [None, "private text", {}, {**original, "unknown": "x"}]
        bad_rows += [{k: v for k, v in original.items() if k != missing} for missing in original]
        bad_rows += [{**original, "priority": value} for value in (True, "1", 1.0, 0, -1, None)]
        bad_rows += [{**original, field: value} for field in ("problem", "suggestion") for value in ("", " ", 1, None)]
        bad_rows += [{**original, "example_rewrite": value} for value in (None, 4, {})]
        for row in bad_rows:
            raw = complete_rubric(); raw["improvements"] = [row]
            with self.subTest(row=row), self.assertRaises(RubricValidationError):
                normalize_rubric(raw)
        for value in (None, {}, "text"):
            raw = complete_rubric(); raw["improvements"] = value
            with self.assertRaises(RubricValidationError):
                normalize_rubric(raw)

    def test_omitted_fact_rows_have_required_types(self):
        for row in (None, "text", {}, {"fact": "x"}, {"fact": "x", "source": "p.1", "other": 1}):
            raw = complete_rubric(); raw["omitted_record_facts"] = [row]
            with self.assertRaises(RubricValidationError):
                normalize_rubric(raw)
        for field in ("fact", "source"):
            for value in ("", " ", None, 42, [], True):
                raw = complete_rubric(); raw["omitted_record_facts"] = [{"fact": "Synthetic observation", "source": "a.txt p.1", field: value}]
                with self.subTest(field=field, value=value), self.assertRaises(RubricValidationError):
                    normalize_rubric(raw)
        for value in (None, {}, "text"):
            raw = complete_rubric(); raw["omitted_record_facts"] = value
            with self.assertRaises(RubricValidationError):
                normalize_rubric(raw)

    def test_executive_summary_required_nonempty_string(self):
        for value in ("", " ", None, 7, [], True):
            raw = complete_rubric(); raw["executive_summary"] = value
            with self.subTest(value=value), self.assertRaises(RubricValidationError):
                normalize_rubric(raw)

    def test_empty_fact_and_improvement_lists_are_valid(self):
        raw = complete_rubric(); raw["improvements"] = []
        self.assertEqual(normalize_rubric(raw)["improvements"], [])

    def test_normalization_does_not_mutate_input(self):
        raw = complete_rubric(); raw["rationales"]["factual_accuracy"] = "  Rationale  "
        before = deepcopy(raw)
        self.assertEqual(normalize_rubric(raw)["rationales"]["factual_accuracy"], "Rationale")
        self.assertEqual(raw, before)

    def test_invalid_scores_cannot_be_clamped_into_effectiveness(self):
        result = EvaluationResult(scores={key: 999 for key in DIMENSION_LABELS})
        with self.assertRaises(RubricValidationError):
            compute_effectiveness_score(result)

    def test_incomplete_scoring_cannot_trigger_recommendation_calls(self):
        result = EvaluationResult(scoring_policy=RUBRIC_POLICY, scoring_status="incomplete")
        llm = _FakeLLM()
        _score_and_recommend(llm, result, MagicMock())
        self.assertEqual(llm.calls, [])
        self.assertIsNone(result.effectiveness_score)


class TestRubricPipeline(unittest.TestCase):
    def _run(self, response, progress=None):
        llm = _FakeLLM(overrides={"rubric": response})
        with patch("app.evaluate.review_medical_records", return_value=_fake_digest()), patch("app.evaluate.load_knowledge", return_value="Synthetic knowledge"):
            result = run_evaluation(llm, "Synthetic witnessed knee symptoms.", [_doc()], progress=progress)
        return result, llm

    def test_999_scores_preserve_review_but_never_grade_or_rewrite(self):
        response = complete_rubric({key: 999 for key in DIMENSION_LABELS})
        progress = MagicMock()
        result, llm = self._run(response, progress)
        self.assertEqual(result.scoring_status, "incomplete")
        self.assertFalse(rubric_is_complete(result))
        self.assertIsNotNone(result.digest)
        self.assertEqual(len(result.claims), 2)
        self.assertEqual(len(result.verifications), 2)
        self.assertEqual(result.scores, {})
        self.assertEqual(result.rationales, {})
        self.assertEqual(result.overall_rating, "Not scored")
        self.assertIsNone(result.effectiveness_score)
        self.assertEqual(result.score_band, "unavailable")
        self.assertEqual(result.recommendations, [])
        self.assertEqual(result.revised_statement, "")
        self.assertEqual(llm.calls.count(("chat_json", "rubric")), RUBRIC_MAX_ATTEMPTS)
        for phase in ("revision", "recommendations"):
            self.assertNotIn(("chat_json", phase), llm.calls)
        self.assertIn(RUBRIC_INCOMPLETE_NOTICE, result.report_markdown)
        self.assertIn("Claim-by-Claim Verification", result.report_markdown)
        for text in ("Excellent", "Rubric Scores", "/100", "999", "Proposed Rewrite"):
            self.assertNotIn(text, result.report_markdown)
        self.assertIn("Partial evaluation", progress.call_args.args[1])

    def test_bad_rationale_with_good_scores_is_still_incomplete(self):
        response = complete_rubric(); response["rationales"] = {}
        result, _ = self._run(response)
        self.assertEqual(result.scores, {})
        self.assertEqual(result.overall_rating, "Not scored")

    def test_retry_recovers_with_only_the_complete_response(self):
        calls = []
        def response(_system, user, _kwargs):
            calls.append(user)
            if len(calls) == 1:
                return {"scores": {key: 999 for key in DIMENSION_LABELS}, "secret": "PRIVATE_RESPONSE_SENTINEL"}
            return complete_rubric()
        result, llm = self._run(response)
        self.assertTrue(rubric_is_complete(result))
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.overall_rating, "Adequate")
        self.assertIsInstance(result.effectiveness_score, int)
        self.assertIn(("chat_json", "revision"), llm.calls)
        self.assertNotIn("PRIVATE_RESPONSE_SENTINEL", calls[1])
        self.assertNotIn("PRIVATE_RESPONSE_SENTINEL", result.report_markdown)

    def test_parse_and_provider_errors_are_bounded_without_private_error_output(self):
        for error in (LLMParseError("PRIVATE_RESPONSE_SENTINEL"), LLMError("PRIVATE_RESPONSE_SENTINEL")):
            with self.subTest(error=type(error).__name__), self.assertLogs("app.evaluate", level="WARNING") as logs:
                result, llm = self._run(error)
            self.assertEqual(llm.calls.count(("chat_json", "rubric")), 3)
            self.assertEqual(result.scoring_status, "incomplete")
            self.assertNotIn("PRIVATE_RESPONSE_SENTINEL", " ".join(logs.output) + result.report_markdown)

    def test_cancellation_is_never_swallowed_or_retried(self):
        with self.assertRaises(PipelineCancelledError):
            self._run(lambda *_: (_ for _ in ()).throw(PipelineCancelledError()))

    def test_pipeline_timeout_is_never_swallowed(self):
        with self.assertRaises(PipelineTimeoutError):
            self._run(PipelineTimeoutError(2, 1))

    def test_valid_boundary_scores_produce_expected_ratings(self):
        for value, rating in ((0, "Needs Substantial Work"), (10, "Excellent")):
            result, llm = self._run(complete_rubric({key: value for key in DIMENSION_LABELS}))
            self.assertTrue(rubric_is_complete(result))
            self.assertEqual(result.overall_rating, rating)
            self.assertEqual(llm.calls.count(("chat_json", "rubric")), 1)

    def test_permanent_provider_failures_stop_after_one_attempt(self):
        for error in (LLMAuthError("Synthetic credential rejection", status_code=401),
                      LLMConfigurationError("Synthetic configuration error")):
            llm = _FakeLLM(overrides={"rubric": error})
            with patch("app.evaluate.review_medical_records", return_value=_fake_digest()), patch("app.evaluate.load_knowledge", return_value="Synthetic knowledge"):
                with self.subTest(error=type(error).__name__), self.assertRaises(type(error)):
                    run_evaluation(llm, "Synthetic statement.", [_doc()])
            self.assertEqual(llm.calls.count(("chat_json", "rubric")), 1)
            self.assertNotIn(("chat_json", "topic"), llm.calls)
            self.assertNotIn(("chat_json", "revision"), llm.calls)

    def test_partial_report_preserves_independent_source_snapshot(self):
        citations = [{"source": "synthetic.pdf p.7", "excerpt": "Synthetic source appendix excerpt."}]
        with patch("app.evaluate.st") as st_mock:
            st_mock.session_state.get.return_value = citations
            result, _ = self._run(complete_rubric({key: 999 for key in DIMENSION_LABELS}))
        citations[0]["source"] = "CHANGED_SOURCE_SENTINEL"
        report = evaluation_report_markdown(result)
        self.assertIn("## Sources", report)
        self.assertIn("synthetic.pdf p.7", report)
        self.assertIn("Synthetic source appendix excerpt.", report)
        self.assertNotIn("CHANGED_SOURCE_SENTINEL", report)
        restored = evaluation_from_json(evaluation_to_json(result))
        self.assertEqual(restored.report_citations, result.report_citations)
        self.assertIn("Synthetic source appendix excerpt.", evaluation_report_markdown(restored))


class TestRubricSavedResults(unittest.TestCase):
    def _valid(self):
        return EvaluationResult(**scored_result_fields(), **topic_result_fields(), claims=[{"id": 1, "text": "Synthetic claim."}], verifications=[{"id": 1, "verdict": "NOT FOUND"}], revised_statement="Synthetic reviewed rewrite.")

    def test_complete_json_roundtrip_retains_status_and_effectiveness(self):
        result = self._valid()
        payload = evaluation_to_json(result)
        restored = evaluation_from_json(json.loads(json.dumps(payload, allow_nan=False)))
        self.assertTrue(rubric_is_complete(restored))
        self.assertEqual(restored.scoring_policy, RUBRIC_POLICY)
        self.assertEqual(restored.overall_rating, "Adequate")
        self.assertEqual(restored.effectiveness_score, payload["effectiveness_score"])
        self.assertEqual(restored.revised_statement, result.revised_statement)

    def test_invalid_saved_scores_are_checked_before_coercion(self):
        for value in (True, "9", 999, float("nan"), float("inf")):
            payload = evaluation_to_json(self._valid())
            payload["scores"]["factual_accuracy"] = value
            with self.subTest(value=value):
                restored = evaluation_from_json(payload)
                self.assertFalse(rubric_is_complete(restored))
                self.assertEqual(restored.scores, {})
                self.assertIsNone(restored.effectiveness_score)
                self.assertEqual(restored.revised_statement, "")
                self.assertEqual(restored.overall_rating, "Not scored")

    def test_saved_incomplete_or_unknown_policy_never_promotes_to_complete(self):
        for flags in ({"scoring_status": "incomplete"}, {"scoring_policy": "future"}, {"scoring_policy": ""}, {"scoring_status": ""}):
            payload = {**evaluation_to_json(self._valid()), **flags}
            restored = evaluation_from_json(payload)
            self.assertFalse(rubric_is_complete(restored))
            self.assertEqual(restored.overall_rating, "Not scored")
            self.assertIsNone(restored.effectiveness_score)
            self.assertEqual(restored.revised_statement, "")

    def test_malformed_saved_rationales_and_rows_are_not_filtered_into_success(self):
        for field, value in (("rationales", {}), ("improvements", ["invalid"]), ("omitted_record_facts", [None]), ("executive_summary", 10)):
            payload = {**evaluation_to_json(self._valid()), field: value}
            self.assertFalse(rubric_is_complete(evaluation_from_json(payload)))

    def test_cached_invalid_report_is_rebuilt_without_grades_or_rewrite(self):
        result = self._valid(); result.scores["factual_accuracy"] = 999
        result.report_markdown = "Overall rating: Excellent\nEffectiveness score: 100/100\nPRIVATE_REWRITE_SENTINEL"
        report = evaluation_report_markdown(result)
        self.assertIn(RUBRIC_UNVALIDATED_NOTICE, report)
        self.assertIn("Synthetic claim.", report)  # Retained review data remains available.
        for text in ("Excellent", "100/100", "PRIVATE_REWRITE_SENTINEL", "Proposed Rewrite"):
            self.assertNotIn(text, report)
        payload = evaluation_to_json(result)
        self.assertIsNone(payload["effectiveness_score"])
        self.assertEqual(payload["scores"], {})
        self.assertEqual(payload["revised_statement"], "")
        self.assertNotIn("PRIVATE_REWRITE_SENTINEL", payload["report_markdown"])

    def test_queue_envelope_preserves_partial_review(self):
        result = EvaluationResult(scoring_policy=RUBRIC_POLICY, scoring_status="incomplete", claims=[{"id": 1, "text": "Synthetic claim."}], digest=_fake_digest())
        from app.llm import UsageTracker
        restored = decode_result(encode_result(RunResult(kind="evaluate", result=result, usage=UsageTracker(), request_id="synthetic-r02"))).result
        self.assertEqual(restored.scoring_status, "incomplete")
        self.assertEqual(restored.claims, result.claims)
        self.assertEqual(restored.digest.summary, result.digest.summary)
        self.assertIsNone(restored.effectiveness_score)

    def test_raw_result_mutation_cannot_restore_a_rating(self):
        for field, value in (("scores", {key: 999 for key in DIMENSION_LABELS}), ("rationales", {}), ("scoring_status", "incomplete")):
            result = self._valid(); setattr(result, field, value)
            self.assertEqual(result.overall_rating, "Not scored")
            self.assertEqual(result.score_band, "unavailable")

    def test_worker_outcome_does_not_call_invalid_complete_flags_successful(self):
        from app.worker import _outcome_for
        result = self._valid()
        result.scores["factual_accuracy"] = 999
        outcome = _outcome_for("evaluate", result)
        self.assertEqual(outcome["scoring_status"], "incomplete")
        self.assertEqual(outcome["overall_rating"], "Not scored")

    def test_malformed_historical_claims_do_not_break_safe_report_export(self):
        result = evaluation_from_json({
            "claims": [{"text": "Synthetic claim without ID."}, {"id": [], "text": "Synthetic bad ID."},
                       {"id": 1, "text": {"invalid": "shape"}}],
            "verifications": [{"id": 1, "verdict": "NOT FOUND", "record_reference": "", "note": "Synthetic note."}],
            "report_markdown": "Excellent rating from historical cached markdown",
            "report_citations": [{"source": "synthetic.pdf p.7", "excerpt": "Retained source excerpt."}],
        })
        report = evaluation_report_markdown(result)
        self.assertIn("cannot be linked to findings", report)
        self.assertIn("Synthetic claim without ID.", report)
        self.assertIn("Synthetic bad ID.", report)
        self.assertIn("Retained source excerpt.", report)
        self.assertIn("| # | Claim | Verdict", report)
        self.assertNotIn("Excellent", report)
        self.assertEqual(evaluation_to_json(result)["report_markdown"], report)

    def test_saved_citation_rows_require_string_source_and_excerpt(self):
        result = evaluation_from_json({"report_citations": [None, {}, {"source": 1, "excerpt": "x"},
                  {"source": "s", "excerpt": []}, {"source": "s", "excerpt": "Retained excerpt"}]})
        self.assertEqual(result.report_citations, [{"source": "s", "excerpt": "Retained excerpt"}])


class TestRubricUI(unittest.TestCase):
    def test_partial_result_has_warning_without_grades_charts_or_rewrite(self):
        import app.views.evaluate_view as view
        result = EvaluationResult(scoring_policy=RUBRIC_POLICY, scoring_status="incomplete", claims=[{"id": 1, "text": "Synthetic claim."}], scores={key: 999 for key in DIMENSION_LABELS}, revised_statement="PRIVATE_REWRITE_SENTINEL", effectiveness_score=100)
        st_mock, _ = _fake_streamlit()
        st_mock.columns.return_value = tuple(MagicMock() for _ in range(4))
        with _patch_st(view, st_mock), patch.object(view, "track_impression") as impressions:
            view._render_evaluation_results(result)
        self.assertIn(RUBRIC_INCOMPLETE_NOTICE, [c.args[0] for c in st_mock.warning.call_args_list])
        st_mock.bar_chart.assert_not_called()
        st_mock.metric.assert_not_called()
        st_mock.text_area.assert_not_called()
        st_mock.success.assert_not_called()
        impressions.assert_not_called()
        reports = [c.kwargs["data"].decode() for c in st_mock.download_button.call_args_list if c.kwargs.get("file_name") == "lay_statement_evaluation.md"]
        self.assertEqual(len(reports), 1)
        self.assertNotIn("PRIVATE_REWRITE_SENTINEL", reports[0])
        self.assertIn(RUBRIC_INCOMPLETE_NOTICE, reports[0])


    def test_malformed_legacy_claims_render_and_download_without_scoring(self):
        import app.views.evaluate_view as view
        result = evaluation_from_json({"claims": [{"text": "Synthetic unlinked claim."}, {"id": []}],
            "verifications": [{"id": 1, "verdict": "NOT FOUND"}],
            "report_citations": [{"source": "synthetic.pdf p.1", "excerpt": "Synthetic retained excerpt."}]})
        st_mock, _ = _fake_streamlit()
        st_mock.columns.return_value = tuple(MagicMock() for _ in range(4))
        with _patch_st(view, st_mock):
            view._render_evaluation_results(result)
        warnings = [str(c.args[0]) for c in st_mock.warning.call_args_list]
        self.assertTrue(any("cannot be linked" in warning for warning in warnings))
        reports = [c.kwargs["data"].decode() for c in st_mock.download_button.call_args_list
                   if c.kwargs.get("file_name") == "lay_statement_evaluation.md"]
        self.assertEqual(len(reports), 1)
        self.assertIn("Synthetic unlinked claim.", reports[0])
        self.assertIn("Synthetic retained excerpt.", reports[0])
        st_mock.metric.assert_not_called()

    def test_legacy_score_badge_is_withheld(self):
        import app.views.evaluate_view as view
        result = EvaluationResult(scores={key: 10 for key in DIMENSION_LABELS}, effectiveness_score=100)
        st_mock, _ = _fake_streamlit()
        with _patch_st(view, st_mock):
            view._render_effectiveness_score(result)
        st_mock.metric.assert_not_called()
        st_mock.success.assert_not_called()
        self.assertIn(RUBRIC_UNVALIDATED_NOTICE, st_mock.warning.call_args.args[0])


class TestRubricPilotDiagnostics(unittest.TestCase):
    def test_pilot_metadata_keeps_only_recognized_scoring_classifications(self):
        from app import pilot
        for classification in ("complete", "incomplete", "invalid", "unvalidated"):
            saved = pilot.safe_metadata({"action": "evaluate", "status": "partial",
                "scoring_status": classification, "patient": "PRIVATE_METADATA_SENTINEL"})
            self.assertEqual(saved["status"], "partial")
            self.assertEqual(saved["scoring_status"], classification)
            self.assertNotIn("patient", saved)
        self.assertEqual(pilot.safe_metadata({"status": "PRIVATE_METADATA_SENTINEL",
                         "scoring_status": "PRIVATE_METADATA_SENTINEL"}), {})

    def test_pilot_partial_status_survives_persistence_and_ops_display(self):
        from pathlib import Path
        import tempfile
        from app import pilot, run_log
        from app.views.ops import _event_row
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "runs.jsonl"
            with patch.dict("os.environ", {"VA_LSE_PILOT_LOG_RETENTION_DAYS": "7"}), \
                    patch.object(pilot, "enabled", return_value=True), patch.object(run_log, "_resolve_log_path", return_value=path):
                run_log.run_log_event("evaluate", "partial", request_id="req_0123456789ab",
                    scoring_status="incomplete", error="PRIVATE_METADATA_SENTINEL", patient="PRIVATE_METADATA_SENTINEL")
            persisted = path.read_text()
            event = json.loads(persisted)
        self.assertNotIn("PRIVATE_METADATA_SENTINEL", persisted)
        self.assertEqual(event["status"], "partial")
        self.assertEqual(event["scoring_status"], "incomplete")
        self.assertEqual(_event_row(event)["Status"], "⚠️ partial")
