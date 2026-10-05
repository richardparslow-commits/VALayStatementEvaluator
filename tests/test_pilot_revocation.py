"""Synthetic access changes while a provider request or pipeline is running."""
from __future__ import annotations

import os
import threading
import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from tests import hermetic  # noqa: F401
from tests.test_controlled_pilot import approval
from app import llm, pilot
from app.documents import document_from_text


class TestRevocationDuringWork(unittest.TestCase):
    def setUp(self):
        import streamlit as st
        from app import documents
        self.addCleanup(documents.set_active_extractor, documents._ACTIVE_EXTRACTOR)
        self.data = approval()
        from tests.pilot_budget_fixtures import install_budget
        install_budget(self, self.data)
        llm._sdk_name("NOT_GIVEN")  # Normal client construction binds this SDK sentinel.
        self.now = 1000
        self.claims = {"is_logged_in": True, "iss": self.data["issuer"],
                       "sub": "participant", "iat": 950, "exp": 1100}
        for context in (
            patch.dict(os.environ, {"VA_LSE_MODE": "controlled-pilot", "VA_LSE_PILOT_LOG_RETENTION_DAYS": "7"}),
            patch.object(pilot, "load_approval", side_effect=lambda: self.data),
            patch.object(pilot.time, "time", side_effect=lambda: self.now),
            patch.object(st, "user", self.claims),
            patch.object(pilot, "require_destination"),
            patch.object(pilot, "require_consent", side_effect=lambda owner: pilot.notice_binding(owner, self.data)),
        ):
            context.start()
            self.addCleanup(context.stop)
        self.client = llm.LLMClient.__new__(llm.LLMClient)
        self.client._pilot_calls = 0
        self.client._pilot_prompt_chars = 0
        self.client._pilot_call_lock = threading.Lock()
        self.client._client = MagicMock()
        self.client._fallback_client = None
        self.client._settings = MagicMock(base_url=self.data["provider_base_url"], model_main="test-model")
        self.client.usage = MagicMock()
        self.docs = [document_from_text("record.txt", "Synthetic knee observation.")]

    def wire(self, response, *, responses=False):
        self.client._client.responses.create.side_effect = response
        self.client._client.chat.completions.create.side_effect = response
        with patch.object(llm, "_uses_responses_schema", return_value=responses):
            with pilot.action_budget(self.docs):
                return self.client._call_openai("primary", "test-model", "system", "synthetic account", 0.2, 100, None)

    def revoke(self, **_):
        self.data["subjects"].remove("participant")
        return MagicMock()

    def expire(self, **_):
        self.now = self.claims["exp"]
        return MagicMock()

    def test_chat_response_is_refused_after_invitation_removal(self):
        with self.assertRaises(pilot.PilotBlocked):
            self.wire(self.revoke)
        self.client._client.chat.completions.create.assert_called_once()

    def test_responses_schema_is_refused_after_invitation_removal(self):
        with self.assertRaises(pilot.PilotBlocked):
            self.wire(self.revoke, responses=True)
        self.client._client.responses.create.assert_called_once()

    def test_response_is_refused_at_exact_token_expiry(self):
        with self.assertRaises(pilot.PilotBlocked):
            self.wire(self.expire)

    def test_response_is_refused_if_approval_disappears(self):
        def disappear(**_):
            pilot.load_approval.side_effect = pilot.PilotBlocked("Approval unavailable.")
            return MagicMock()
        with self.assertRaises(pilot.PilotBlocked):
            self.wire(disappear)

    def test_response_cannot_change_to_another_invited_identity(self):
        def change(**_):
            self.claims["sub"] = "operator"
            return MagicMock()
        with self.assertRaises(pilot.PilotBlocked):
            self.wire(change)

    def test_destination_removal_during_request_refuses_response(self):
        pilot.require_destination.side_effect = [None, pilot.PilotBlocked("Destination no longer approved.")]
        with self.assertRaises(pilot.PilotBlocked):
            self.wire(lambda **_: MagicMock())

    def test_authorized_response_still_completes(self):
        response = MagicMock(model="test-model")
        self.assertEqual(self.wire(lambda **_: response), (response, False))

    def test_run_cannot_complete_after_revocation(self):
        with self.assertRaises(pilot.PilotBlocked):
            with pilot.action_budget(self.docs):
                self.revoke()
        self.assertEqual(pilot._active, 0)
        self.assertIsNone(pilot._run_claims.get())

    def test_run_cannot_complete_at_token_expiry(self):
        with self.assertRaises(pilot.PilotBlocked):
            with pilot.action_budget(self.docs):
                self.expire()
        self.assertEqual(pilot._active, 0)
        self.assertIsNone(pilot._run_claims.get())

    def test_pipeline_error_is_not_masked_and_quota_slot_is_released(self):
        with self.assertRaisesRegex(ValueError, "synthetic failure"):
            with pilot.action_budget(self.docs):
                self.revoke()
                raise ValueError("synthetic failure")
        self.assertEqual(pilot._active, 0)
        self.assertIsNone(pilot._run_claims.get())

    def test_access_refusal_is_not_retried_or_counted_as_provider_failure(self):
        breaker, limiter, rate_gate = MagicMock(), MagicMock(), MagicMock()
        rate_gate.enabled = False
        with patch.object(llm, "get_llm_breaker", return_value=breaker), \
                patch.object(llm, "get_llm_limiter", return_value=limiter), \
                patch.object(llm, "get_llm_rate_gate", return_value=rate_gate), \
                patch.object(llm, "_stall_watchdog_seconds", return_value=0), \
                patch.object(self.client, "_call_openai", side_effect=pilot.PilotBlocked("Access ended.")) as call:
            with self.assertRaises(pilot.PilotBlocked):
                self.client._chat_on_endpoint("primary", "system", "synthetic account")
        call.assert_called_once()
        self.client.usage.record.assert_not_called()
        breaker.record_failure.assert_not_called()
        breaker.record_success.assert_not_called()
        limiter.release.assert_called_once()

    def test_session_rechecks_identity_before_delayed_results_are_saved(self):
        import streamlit as st
        owner = pilot.current_owner()
        with patch.object(st, "session_state", {"_pilot_owner": owner}):
            pilot.require_session_access()
            self.claims["sub"] = "operator"
            with self.assertRaises(pilot.PilotBlocked):
                pilot.require_session_access()

    def test_session_refuses_revoked_expired_or_missing_admission(self):
        import streamlit as st
        owner = pilot.current_owner()
        with patch.object(st, "session_state", {}), self.assertRaises(pilot.PilotBlocked):
            pilot.require_session_access()
        with patch.object(st, "session_state", {"_pilot_owner": owner}):
            self.now = self.claims["exp"]
            with self.assertRaises(pilot.PilotBlocked):
                pilot.require_session_access()
            self.now = 1000
            self.revoke()
            with self.assertRaises(pilot.PilotBlocked):
                pilot.require_session_access()

    def test_synthetic_mode_does_not_require_an_oidc_session(self):
        with patch.dict(os.environ, {"VA_LSE_MODE": "synthetic"}):
            pilot.require_session_access()
            pilot.recheck_owner("synthetic-owner")

    def test_ui_clears_case_and_stops_other_views_after_access_refusal(self):
        import streamlit as st
        from pathlib import Path
        from streamlit.testing.v1 import AppTest
        from app import main
        class VerifiedUser(dict):
            is_logged_in = True
        user = VerifiedUser(self.claims)
        root = Path(__file__).resolve().parents[1]
        with patch.object(pilot, "validate_configuration", return_value=self.data), \
                patch("app.upload_admission.bind_session"), \
                patch.object(st, "user", user), \
                patch.object(pilot, "require_consent", side_effect=lambda owner: st.session_state["_pilot_consent_grant"]), \
                patch.object(main, "render_evaluate_tab", side_effect=pilot.PilotBlocked("Access ended.")), \
                patch.object(main, "render_draft_tab") as draft_view, \
                patch.object(main, "render_failure_detail") as diagnostics:
            at = AppTest.from_file(str(root / "run_app.py"))
            at.session_state["_pilot_notice_consent"] = pilot.notice_binding(pilot.current_owner(), self.data)
            at.session_state["_pilot_consent_grant"] = pilot.ConsentGrant(at.session_state["_pilot_notice_consent"])
            at.session_state["private_case"] = "SYNTHETIC_CASE_CANARY"
            at.run()
        self.assertFalse(at.exception)
        self.assertTrue(any(error.value == "Access ended." for error in at.error))
        self.assertNotIn("private_case", at.session_state)
        self.assertNotIn("_pilot_owner", at.session_state)
        draft_view.assert_not_called()
        diagnostics.assert_not_called()

    def test_drafting_error_mapping_preserves_access_refusal(self):
        from app.drafting_service import map_drafting_exception
        refusal = pilot.PilotBlocked("Access ended.")
        with self.assertRaises(pilot.PilotBlocked) as caught:
            map_drafting_exception(refusal, request_id="synthetic", phase="draft")
        self.assertIs(caught.exception, refusal)

    def test_drafting_pipeline_preserves_access_refusal(self):
        from app import draft
        refusal = pilot.PilotBlocked("Access ended.")
        with patch.object(draft, "_run_draft", side_effect=refusal), \
                self.assertRaises(pilot.PilotBlocked) as caught:
            draft.run_draft(self.client, self.docs,
                            {"witness_name": "Synthetic Witness", "relationship": "Friend"},
                            "I observed knee pain during a walk.", "Knee pain", "Original claim")
        self.assertIs(caught.exception, refusal)
        self.assertEqual(pilot._active, 0)

    def test_parallel_record_digest_does_not_retry_access_refusal(self):
        from app.medical_review import review_medical_records
        service = MagicMock()
        service.chat_json.side_effect = pilot.PilotBlocked("Access ended.")
        with self.assertRaises(pilot.PilotBlocked):
            review_medical_records(service, self.docs)
        service.chat_json.assert_called_once()

    def test_self_review_cannot_keep_a_draft_after_access_refusal(self):
        from app import draft
        from app.medical_review import MedicalDigest
        from tests.grounding_fixtures import complete_grounding
        service = MagicMock()
        refusal = pilot.PilotBlocked("Access ended.")
        service.chat_json.side_effect = [complete_grounding(), refusal]
        service.chat.return_value = "I observed knee pain during a walk."
        with patch.object(draft, "review_medical_records", return_value=MedicalDigest()), \
                self.assertRaises(pilot.PilotBlocked) as caught:
            draft.run_draft(service, self.docs, {"name": "Synthetic Witness", "relationship": "Friend"},
                            "I observed knee pain during a walk.", "Knee pain", "Original claim")
        self.assertIs(caught.exception, refusal)
        self.assertEqual(service.chat_json.call_count, 2)
        self.assertEqual(pilot._active, 0)

    def test_recommendations_cannot_fall_back_after_destination_refusal(self):
        from app import evaluate
        result = evaluate.EvaluationResult()
        refusal = pilot.PilotBlocked("Destination no longer approved.")
        with patch.object(evaluate, "rubric_is_complete", return_value=True), \
                patch.object(evaluate, "topics_are_complete", return_value=True), \
                patch.object(evaluate, "compute_effectiveness_score", return_value=70), \
                patch.object(evaluate, "generate_improvement_recommendations", side_effect=refusal), \
                patch.object(evaluate, "_fallback_recommendations") as fallback, \
                self.assertRaises(pilot.PilotBlocked) as caught:
            evaluate._score_and_recommend(self.client, result, MagicMock())
        self.assertIs(caught.exception, refusal)
        fallback.assert_not_called()

    def test_date_inference_cannot_retry_an_admission_refusal(self):
        from app import medical_review
        service = MagicMock()
        service.chat_json.side_effect = pilot.PilotBlocked("Destination no longer approved.")
        facts = [medical_review.MedicalFact("", "symptom", "Synthetic knee pain", "record.txt b.1")]
        with self.assertRaises(pilot.PilotBlocked):
            medical_review._llm_infer_undated(service, facts)
        service.chat_json.assert_called_once()

    def test_timeline_builder_does_not_mask_an_admission_refusal(self):
        from app import medical_review
        service = MagicMock()
        refusal = pilot.PilotBlocked("Destination no longer approved.")
        service.chat_json.side_effect = refusal
        digest = medical_review.MedicalDigest(facts=[
            medical_review.MedicalFact("", "symptom", "Synthetic knee pain", "record.txt b.1")])
        with patch.dict(os.environ, {"VA_LSE_MODE": "synthetic"}), self.assertRaises(pilot.PilotBlocked) as caught:
            medical_review.build_timeline_data(digest, service)
        self.assertIs(caught.exception, refusal)
        service.chat_json.assert_called_once()

    def _assert_refused_flow_has_terminal_events(self, kind, phase):
        from app.views import draft_view, evaluate_view
        view = draft_view if kind == "draft" else evaluate_view
        refusal = pilot.PilotBlocked("PRIVATE_REFUSAL_CANARY")
        streamlit = MagicMock()
        streamlit.session_state = {}
        bar = MagicMock()
        rid = "req_aaaaaaaaaaaa"
        with ExitStack() as stack:
            def patched(target, name, **kwargs):
                return stack.enter_context(patch.object(target, name, **kwargs))
            patched(view, "st", new=streamlit)
            patched(view, "_validate_draft_inputs" if kind == "draft" else "_validate_evaluate_inputs", return_value=True)
            patched(view.job_runner, "queue_mode_active", return_value=False)
            patched(view, "append_follow_up_answers", return_value="Synthetic account.")
            patched(view, "check_endpoint_gate", return_value=True)
            patched(view, "get_llm", return_value=self.client)
            patched(view, "check_shutdown_gate", return_value=True)
            patched(view, "enter_run", return_value=True)
            exit_run = patched(view, "exit_run")
            patched(view, "audit_record_meta", return_value=([], 1, 1))
            patched(view, "audit_condition_for_slot", return_value=None)
            patched(view, "progress_widgets", return_value=(bar, MagicMock()))
            patched(view, "get_profiler", return_value=None)
            patched(view, "check_memory_before_run")
            patched(view, "run_with_timeout", **({"side_effect": refusal} if phase == "pipeline" else {"return_value": MagicMock()}))
            patched(pilot, "require_session_access", side_effect=refusal)
            start = patched(view.audit_log, "audit_" + kind + "_start")
            audit = patched(view.audit_log, "audit_" + kind + "_error")
            log = patched(view, "run_log_event")
            if kind == "evaluate":
                patched(view, "new_run_request_id", return_value=rid)
            with self.assertRaises(pilot.PilotBlocked) as caught:
                if kind == "draft":
                    view._run_draft_flow(rid=rid, records=self.docs, condition="Knee pain", claim_type="Original claim",
                                         witness={}, observations="Synthetic account.")
                else:
                    view._run_evaluation_flow("Synthetic account.", self.docs)
        self.assertIs(caught.exception, refusal)
        start.assert_called_once()
        audit.assert_called_once()
        self.assertEqual(audit.call_args.kwargs["request_id"], rid)
        self.assertEqual(str(audit.call_args.kwargs["error"]), "Pilot access refused.")
        self.assertTrue(any(call.args[:2] == (kind, "rejected") for call in log.call_args_list))
        self.assertNotIn("PRIVATE_REFUSAL_CANARY", repr(audit.call_args_list) + repr(log.call_args_list))
        self.assertNotIn("draft_result" if kind == "draft" else "eval_result", streamlit.session_state)
        exit_run.assert_called_once()
        bar.empty.assert_called_once()

    def test_refused_evaluation_records_terminal_count_only_events(self):
        for phase in ("pipeline", "pre_storage"):
            with self.subTest(phase=phase):
                self._assert_refused_flow_has_terminal_events("evaluate", phase)

    def test_refused_draft_records_terminal_count_only_events(self):
        for phase in ("pipeline", "pre_storage"):
            with self.subTest(phase=phase):
                self._assert_refused_flow_has_terminal_events("draft", phase)
