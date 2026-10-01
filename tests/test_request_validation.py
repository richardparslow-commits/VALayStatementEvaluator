"""Required-input regressions: rejected inputs spend/queue nothing, using only fakes."""
from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

from tests import hermetic  # noqa: F401
from app import config, draft, evaluate, pilot, worker
from app.documents import DocumentPage, ExtractedDocument, document_from_text
from app.drafting_service import DraftingPayloadError
from app.job_payload import DraftJob, EvaluateJob, PayloadError, decode_job, encode_job, documents_bundle
from app.job_queue import InProcessJobBackend, STATUS_ERROR
from app.request_validation import (
    MAX_OBSERVATIONS_PAYLOAD_CHARS, MAX_STATEMENT_PAYLOAD_CHARS, RequestValidationError,
    validate_draft_request, validate_evaluation_request, validate_records,
)
from tests.test_views import _fake_streamlit


def docs():
    return [document_from_text("synthetic.txt", "Synthetic knee pain recorded.")]


def evaluation_request():
    return dict(statement_text="I observed knee pain.", records=docs(), witness=None)


def draft_request():
    return dict(observations="I observed knee pain.", condition="Knee pain",
                claim_type="Service connection", witness={}, records=docs())


def invalid_values(limit):
    return ("", " \t\n\u2003", "\u200b\ufeff", "\x00\x01\x7f\x80", "\u061c", None, 7, True, [], {}, "x" * (limit + 1))


class TestInputContract(unittest.TestCase):
    def test_minimal_requests_keep_witness_fields_optional(self):
        validate_evaluation_request(**evaluation_request())
        validate_draft_request(**draft_request())
        validate_draft_request(**{**draft_request(), "witness": {"name": "", "aa_bathing": ""}})

    def test_narrative_boundary_includes_followup_headroom(self):
        validate_evaluation_request(**{**evaluation_request(), "statement_text": "x" * MAX_STATEMENT_PAYLOAD_CHARS})
        validate_draft_request(**{**draft_request(), "observations": "x" * MAX_OBSERVATIONS_PAYLOAD_CHARS})

    def test_optional_metadata_still_has_types_and_bounds(self):
        for witness in ([], "name", {"name": None}, {"name": 1}, {"name": "x" * 501},
                        {"aa_bathing": "x" * 4001}, {1: "text"}, {"x" * 101: ""},
                        {str(i): "" for i in range(65)}):
            with self.subTest(witness_type=type(witness).__name__):
                for validator, request in ((validate_evaluation_request, evaluation_request()),
                                           (validate_draft_request, draft_request())):
                    with self.assertRaises(RequestValidationError):
                        validator(**{**request, "witness": witness})

    def test_invalid_record_types_and_empty_text_are_rejected(self):
        for records in (None, {}, (), [], [None], ["text"], [ExtractedDocument("a.txt")],
                        [ExtractedDocument("a.txt", [DocumentPage("a.txt", 1, " \n ")])],
                        [ExtractedDocument("a.txt", [DocumentPage("a.txt", 1, 42)])],
                        [ExtractedDocument("a.txt", [DocumentPage("b.txt", 1, "text")])],
                        [ExtractedDocument("a.txt", [DocumentPage("a.txt", True, "text")])]):
            with self.subTest(records_type=type(records).__name__), self.assertRaises(RequestValidationError):
                validate_records(records)

    def test_record_limits_apply_to_direct_requests(self):
        with patch.object(config, "MAX_RECORD_PAGES", 1):
            record = docs()[0]; record.total_pages = 2
            with self.assertRaises(RequestValidationError): validate_records([record])
        with patch.object(config, "MAX_TOTAL_UPLOAD_BYTES", 3):
            with self.assertRaises(RequestValidationError):
                validate_records([document_from_text("a.txt", "éé")])

    def test_errors_do_not_echo_submitted_text(self):
        for request in ({**draft_request(), "condition": "PRIVATE_SENTINEL" * 100},
                        {**draft_request(), "witness": {"name": "PRIVATE_SENTINEL```"}},
                        {**draft_request(), "observations": "PRIVATE_SENTINEL" * 10000}):
            with self.assertRaises(RequestValidationError) as caught:
                validate_draft_request(**request)
            self.assertNotIn("PRIVATE_SENTINEL", str(caught.exception))

    def test_invalid_unicode_is_a_clear_input_error(self):
        with self.assertRaises(RequestValidationError):
            validate_evaluation_request(**{**evaluation_request(), "statement_text": "\ud800"})

    def test_record_admission_counts_utf8_once_per_page(self):
        class CountedText(str):
            encodes = 0
            def encode(self, *args, **kwargs):
                self.encodes += 1
                return super().encode(*args, **kwargs)
        text = CountedText("é" * 100_000)
        record = ExtractedDocument("synthetic.txt", [DocumentPage("synthetic.txt", 1, text)])
        with patch("app.prompt_sanitize.sanitize_for_prompt", side_effect=AssertionError("Full page sanitized")):
            self.assertEqual(validate_records([record]), (1, 200_000))
        self.assertEqual(text.encodes, 1)


class TestDirectBoundaries(unittest.TestCase):
    def test_invalid_statement_never_reserves_quota_or_calls_model(self):
        for value in invalid_values(MAX_STATEMENT_PAYLOAD_CHARS):
            llm = MagicMock()
            with self.subTest(value_type=type(value).__name__), patch.object(pilot, "action_budget") as budget, \
                    patch.object(evaluate, "review_medical_records") as review:
                with self.assertRaises(RequestValidationError):
                    evaluate.run_evaluation(llm, value, docs())
                budget.assert_not_called(); review.assert_not_called()
            llm.chat.assert_not_called(); llm.chat_json.assert_not_called()

    def test_each_required_draft_field_stops_before_quota_and_models(self):
        for field, limit in (("observations", MAX_OBSERVATIONS_PAYLOAD_CHARS), ("condition", 500), ("claim_type", 500)):
            for value in invalid_values(limit):
                llm = MagicMock(); request = {**draft_request(), field: value}
                with self.subTest(field=field, value_type=type(value).__name__), \
                        patch.object(pilot, "action_budget") as budget, patch.object(draft, "review_medical_records") as review:
                    with self.assertRaises(DraftingPayloadError): draft.run_draft(llm, **request)
                    budget.assert_not_called(); review.assert_not_called()
                llm.chat.assert_not_called(); llm.chat_json.assert_not_called()

    def test_invalid_records_and_witness_stop_before_quota(self):
        for fields in ({"records": None}, {"records": []}, {"records": [MagicMock()]}, {"witness": []}):
            with self.subTest(fields=list(fields)), patch.object(pilot, "action_budget") as budget:
                with self.assertRaises(RequestValidationError): evaluate.run_evaluation(MagicMock(), **{**evaluation_request(), **fields})
                with self.assertRaises(DraftingPayloadError): draft.run_draft(MagicMock(), **{**draft_request(), **fields})
                budget.assert_not_called()
        with patch.object(pilot, "action_budget") as budget, self.assertRaises(DraftingPayloadError):
            draft.run_draft(MagicMock(), **{**draft_request(), "witness": None})
        budget.assert_not_called()

    def test_valid_minimal_inputs_reach_pipeline_with_optional_witness_empty(self):
        from tests.test_evaluate import _FakeLLM, _fake_digest, _doc
        from tests.test_draft import _FakeLLM as DraftLLM, _fake_digest as draft_digest, _doc as draft_doc
        with patch.object(evaluate, "review_medical_records", return_value=_fake_digest()), \
                patch.object(evaluate, "load_knowledge", return_value="Synthetic guidance."):
            result = evaluate.run_evaluation(_FakeLLM(), **{**evaluation_request(), "records": [_doc()]})
        self.assertTrue(result.claims)
        with patch.object(draft, "review_medical_records", return_value=draft_digest()), \
                patch.object(draft, "load_knowledge", return_value="Synthetic guidance."):
            result = draft.run_draft(DraftLLM(), **{**draft_request(), "records": [draft_doc()]})
        self.assertTrue(result.draft)


class TestQueuedBoundaries(unittest.TestCase):
    def test_invalid_inputs_cannot_be_encoded_or_written_to_blob(self):
        from app.views import job_runner
        for kind, job in (("evaluate", EvaluateJob(" ", docs())),
                          ("draft", DraftJob(**{**draft_request(), "observations": " "})),
                          ("draft", DraftJob(**{**draft_request(), "condition": None}))):
            with self.subTest(kind=kind), patch.object(job_runner, "get_blob_store") as store:
                with self.assertRaises(PayloadError): job_runner._encode_payload(kind, job)
                with self.assertRaises(PayloadError): documents_bundle(job)
                store.assert_not_called()

    def test_decoding_preserves_raw_types_until_validation(self):
        for kind, job, fields in (("evaluate", EvaluateJob("statement", docs()), ("statement_text",)),
                                  ("draft", DraftJob(**draft_request()), ("observations", "condition", "claim_type"))):
            body = json.loads(encode_job(kind, job))
            for field in fields:
                limit = MAX_STATEMENT_PAYLOAD_CHARS if field == "statement_text" else MAX_OBSERVATIONS_PAYLOAD_CHARS if field == "observations" else 500
                for value in invalid_values(limit):
                    raw = {**body, field: value}
                    with self.subTest(kind=kind, field=field, value_type=type(value).__name__), self.assertRaises(PayloadError):
                        decode_job(kind, json.dumps(raw))

    def test_missing_required_fields_cannot_decode(self):
        body = json.loads(encode_job("draft", DraftJob(**draft_request())))
        for field in ("observations", "condition", "claim_type"):
            raw = dict(body); raw.pop(field)
            with self.subTest(field=field), self.assertRaises(PayloadError): decode_job("draft", json.dumps(raw))

    def test_malformed_citation_units_cannot_be_coerced_to_page(self):
        body = json.loads(encode_job("evaluate", EvaluateJob("statement", docs())))
        for value in (None, 7, False, {}, [], "", "chapter"):
            for field in ("kind", "pagination"):
                raw = deepcopy(body); record = raw["documents"][0]
                (record["pages"][0] if field == "kind" else record)[field] = value
                with self.subTest(field=field, value_type=type(value).__name__), self.assertRaises(PayloadError):
                    decode_job("evaluate", json.dumps(raw))
        # Older requests omitted citation units; their existing page default remains.
        legacy = deepcopy(body); record = legacy["documents"][0]
        record.pop("pagination"); record["pages"][0].pop("kind")
        self.assertEqual(decode_job("evaluate", json.dumps(legacy)).records[0].pages[0].kind, "page")

    def test_wrong_witness_types_cannot_be_coerced_to_optional_empty_fields(self):
        for kind, job in (("evaluate", EvaluateJob("statement", docs())), ("draft", DraftJob(**draft_request()))):
            body = json.loads(encode_job(kind, job))
            for witness in ([], {"name": 7}, {"aa_bathing": None}):
                with self.subTest(kind=kind), self.assertRaises(PayloadError):
                    decode_job(kind, json.dumps({**body, "witness": witness}))

    def test_malformed_record_members_cannot_be_silently_dropped(self):
        body = json.loads(encode_job("evaluate", EvaluateJob("statement", docs())))
        bad_documents = [None, {"filename": "bad.txt", "pages": []},
                         {"filename": "bad.txt", "pages": [{"page": 1, "text": 123}]},
                         {"filename": "bad.txt", "pages": [{"page": "1", "text": "text"}]}]
        for bad in bad_documents:
            raw = deepcopy(body); raw["documents"].append(bad)
            with self.subTest(bad_type=type(bad).__name__), self.assertRaises(PayloadError):
                decode_job("evaluate", json.dumps(raw))

    def test_invalid_worker_requests_fail_before_client_or_pipeline(self):
        body = json.loads(encode_job("draft", DraftJob(**draft_request())))
        for field in ("observations", "condition", "claim_type", "witness"):
            backend = InProcessJobBackend(prefix="input-test", ttl_seconds=60)
            raw = {**body, field: []}
            record = backend.enqueue("draft", json.dumps(raw))
            claimed, payload = backend.claim(["draft"], worker_id="synthetic")
            with patch.object(worker, "build_llm") as client, patch.object(worker, "_run_pipeline") as pipeline, \
                    patch.object(worker, "_fail", wraps=worker._fail), patch.object(worker, "run_log_event"), \
                    patch.object(worker, "audit_log"):
                self.assertFalse(worker.execute_job(claimed, payload, backend))
                client.assert_not_called(); pipeline.assert_not_called()
            self.assertEqual(backend.get(record.job_id).status, STATUS_ERROR)
            self.assertIsNone(backend.get_result(record.job_id))

    def test_invalid_submission_and_saved_retry_do_not_enqueue(self):
        from app.views import job_runner
        st_mock, session = _fake_streamlit(); backend = MagicMock(); store = MagicMock()
        bad_job = DraftJob(**{**draft_request(), "observations": " "})
        with patch.object(job_runner, "st", st_mock), patch.object(job_runner, "get_job_backend", return_value=backend), \
                patch.object(job_runner, "get_blob_store", return_value=store), patch.object(job_runner, "run_log_event"), \
                patch.object(pilot, "current_owner", return_value="synthetic"):
            self.assertIsNone(job_runner.submit_job(slot="draft", job=bad_job, request_id="synthetic", condition=None,
                                                   sources=[], files=1, pages=1, action_label="Draft", wait_for_result=False))
            body = json.loads(encode_job("draft", DraftJob(**draft_request()))); body["condition"] = " "
            session["draft_queue_submission"] = dict(kind="draft", owner_id="synthetic", payload=json.dumps(body))
            self.assertIsNone(job_runner.submit_job(slot="draft", job=None, request_id="synthetic", condition=None,
                                                   sources=[], files=1, pages=1, action_label="Draft", wait_for_result=False))
        backend.enqueue.assert_not_called(); store.put.assert_not_called()


class TestUIBoundaries(unittest.TestCase):
    def test_saved_answers_fit_prompt_or_stop_without_consumption(self):
        from app.views import draft_view, evaluate_view, follow_up
        for view, slot in ((draft_view, "draft"), (evaluate_view, "eval")):
            for total in (60_001, 80_000, 80_001):
                for confirmed in (False, True):
                    st_mock, session = _fake_streamlit()
                    original = "x" * 60_000
                    request = ({**draft_request(), "observations": original} if slot == "draft" else
                               {**evaluation_request(), "statement_text": original})
                    answer = {"topic": "Impact", "question": "What changed?", "answer": "a"}
                    session[f"{slot}_follow_up_saved"] = [answer]
                    session[f"{slot}_confirm_oversize"] = confirmed
                    session["eval_follow_up_input_key"] = follow_up.evaluation_input_key(original, request["records"], {})
                    session["eval_follow_up_input_source_id"] = "synthetic-result"
                    with patch.object(follow_up, "st", st_mock):
                        overhead = len(follow_up.compose_follow_up_appendix(slot)) - 1
                        # Keep the original under the recommended limit; accepted
                        # answers alone cross the thresholds under test.
                        base_length = min(60_000, total - overhead - 3)
                        original = original[:base_length]
                        request["observations" if slot == "draft" else "statement_text"] = original
                        session["eval_follow_up_input_key"] = follow_up.evaluation_input_key(original, request["records"], {})
                        answer["answer"] = "a" * (total - base_length - 2 - overhead)
                        with self.subTest(slot=slot, total=total, confirmed=confirmed), \
                                patch.object(view, "st", st_mock), patch.object(view, "run_log_event") as log, \
                                patch.object(view, "check_endpoint_gate", return_value=False) as gate, \
                                patch.object(view, "get_llm") as client, \
                                patch.object(view.job_runner, "queue_mode_active", return_value=False):
                            if slot == "draft":
                                view._run_draft_flow(rid="synthetic", **request)
                            else:
                                with patch.object(view, "ensure_request_id", return_value="synthetic"):
                                    view._run_evaluation_flow(**request)
                            if total <= 80_000:
                                gate.assert_called_once()
                            else:
                                gate.assert_not_called()
                                self.assertEqual(log.call_args.kwargs["reason"], "payload_too_large")
                                self.assertIn("Shorten", log.call_args.kwargs["error"])
                            client.assert_not_called()
                            self.assertEqual(session[f"{slot}_follow_up_saved"], [answer])

    def test_original_long_text_confirmation_still_works_without_saved_answers(self):
        from app.views import draft_view, evaluate_view, follow_up
        for view, slot in ((draft_view, "draft"), (evaluate_view, "eval")):
            st_mock, session = _fake_streamlit()
            session[f"{slot}_confirm_oversize"] = True
            with patch.object(view, "st", st_mock), patch.object(follow_up, "st", st_mock), \
                    patch.object(view, "run_log_event"), \
                    patch.object(view.job_runner, "queue_mode_active", return_value=False), \
                    patch.object(view, "check_endpoint_gate", return_value=False) as gate:
                if slot == "draft":
                    view._run_draft_flow(rid="synthetic", **{**draft_request(), "observations": "x" * 80_001})
                else:
                    with patch.object(view, "ensure_request_id", return_value="synthetic"):
                        view._run_evaluation_flow(**{**evaluation_request(), "statement_text": "x" * 80_001})
            gate.assert_called_once()

    def test_saved_answers_can_cross_recommended_limit_without_missing_checkbox(self):
        from app.views import draft_view, evaluate_view
        for view in (draft_view, evaluate_view):
            st_mock, session = _fake_streamlit()
            with patch.object(view, "st", st_mock), patch.object(view, "run_log_event"), \
                    patch.object(view, "check_endpoint_gate", return_value=False) as gate, \
                    patch.object(view.job_runner, "queue_mode_active", return_value=False), \
                    patch.object(view, "append_follow_up_answers", return_value="x" * 60_001):
                if view is draft_view:
                    view._run_draft_flow(rid="synthetic", **draft_request())
                else:
                    with patch.object(view, "ensure_request_id", return_value="synthetic"):
                        view._run_evaluation_flow(**evaluation_request())
            self.assertNotIn("draft_confirm_oversize" if view is draft_view else "eval_confirm_oversize", session)
            gate.assert_called_once()

    def test_rejection_logs_preserve_capacity_invalid_and_missing_reasons(self):
        from app.views import draft_view, evaluate_view
        cases = ((evaluate_view, {"statement_text": "x" * (MAX_STATEMENT_PAYLOAD_CHARS + 1)}, "payload_too_large"),
                 (evaluate_view, {"statement_text": 7}, "payload_invalid"),
                 (evaluate_view, {"statement_text": " "}, "no_statement"),
                 (draft_view, {"observations": "x" * (MAX_OBSERVATIONS_PAYLOAD_CHARS + 1)}, "payload_too_large"),
                 (draft_view, {"condition": "x" * 501}, "payload_too_large"),
                 (draft_view, {"observations": 7}, "payload_invalid"),
                 (draft_view, {"observations": " "}, "missing_observations"))
        for view, fields, reason in cases:
            st_mock, _ = _fake_streamlit()
            with self.subTest(reason=reason, fields=list(fields)), patch.object(view, "st", st_mock), \
                    patch.object(view, "run_log_event") as log:
                if view is draft_view:
                    request = {**draft_request(), **fields}
                    self.assertFalse(view._validate_draft_inputs(request["records"], request["observations"], request["condition"],
                                                                "synthetic", claim_type=request["claim_type"], witness=request["witness"]))
                else:
                    with patch.object(view, "ensure_request_id", return_value="synthetic"):
                        self.assertFalse(view._validate_evaluate_inputs(**{**evaluation_request(), **fields}))
            self.assertEqual(log.call_args.kwargs["reason"], reason)
        for view in (draft_view, evaluate_view):
            st_mock, _ = _fake_streamlit()
            with patch.object(view, "st", st_mock), patch.object(view, "run_log_event") as log, patch.object(config, "MAX_RECORD_PAGES", 0):
                if view is draft_view:
                    self.assertFalse(view._validate_draft_inputs(docs(), "obs", "cond", "synthetic"))
                else:
                    with patch.object(view, "ensure_request_id", return_value="synthetic"):
                        self.assertFalse(view._validate_evaluate_inputs("statement", docs()))
            self.assertEqual(log.call_args.kwargs["reason"], "payload_too_large")

    def test_each_flow_checks_record_text_once_before_endpoint_gate(self):
        from app.views import draft_view, evaluate_view
        for view in (draft_view, evaluate_view):
            st_mock, _ = _fake_streamlit()
            with patch.object(view, "st", st_mock), patch.object(view, "run_log_event"), \
                    patch.object(view, "check_endpoint_gate", return_value=False), \
                    patch.object(view.job_runner, "queue_mode_active", return_value=False), \
                    patch("app.request_validation.validate_records", wraps=validate_records) as check, \
                    patch.object(view, "append_follow_up_answers", side_effect=lambda text, **kw: text):
                if view is draft_view:
                    view._run_draft_flow(rid="synthetic", **draft_request())
                else:
                    with patch.object(view, "ensure_request_id", return_value="synthetic"):
                        view._run_evaluation_flow(**evaluation_request())
            check.assert_called_once()

    def test_invalid_evaluation_flow_stops_before_preflight_or_queue(self):
        from app.views import evaluate_view as view
        for fields in ({"statement_text": " "}, {"statement_text": 42}, {"statement_text": "x" * (MAX_STATEMENT_PAYLOAD_CHARS + 1)},
                       {"records": []}, {"witness": {"aa_bathing": 7}}):
            st_mock, _ = _fake_streamlit()
            with patch.object(view, "st", st_mock), patch.object(view, "ensure_request_id", return_value="synthetic"), \
                    patch.object(view, "run_log_event"), patch.object(view, "check_endpoint_gate") as probe, \
                    patch.object(view, "get_llm") as llm, patch.object(view, "_run_evaluation_queued") as queue:
                view._run_evaluation_flow(**{**evaluation_request(), **fields})
                probe.assert_not_called(); llm.assert_not_called(); queue.assert_not_called()

    def test_invalid_draft_flow_stops_before_preflight_or_queue(self):
        from app.views import draft_view as view
        for fields in ({"observations": " "}, {"condition": " "}, {"claim_type": " "}, {"records": []},
                       {"observations": []}, {"condition": "x" * 501}, {"witness": None}):
            st_mock, _ = _fake_streamlit()
            with patch.object(view, "st", st_mock), patch.object(view, "run_log_event"), \
                    patch.object(view, "check_endpoint_gate") as probe, patch.object(view, "get_llm") as llm, \
                    patch.object(view, "_run_draft_queued") as queue:
                view._run_draft_flow(rid="synthetic", **{**draft_request(), **fields})
                probe.assert_not_called(); llm.assert_not_called(); queue.assert_not_called()

    def test_oversized_appended_answers_are_checked_before_preflight(self):
        from app.views import draft_view as view
        st_mock, _ = _fake_streamlit()
        with patch.object(view, "st", st_mock), patch.object(view, "run_log_event"), \
                patch.object(view, "append_follow_up_answers", return_value="x" * (MAX_OBSERVATIONS_PAYLOAD_CHARS + 1)), \
                patch.object(view, "check_endpoint_gate") as probe:
            view._run_draft_flow(rid="synthetic", **draft_request())
        probe.assert_not_called()


class TestBatchBoundaries(unittest.TestCase):
    def test_mixed_readable_and_blank_parts_keep_readable_batch_review(self):
        from tests.test_batch_draft import batch_draft, _main_argv
        from app.llm import ChatProbe
        from app.medical_review import MedicalDigest
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); arguments = _main_argv(root)
            (root / "records" / "Part2.txt").write_text(" \n ")
            with patch("app.config.load_settings", return_value=MagicMock()), \
                    patch("app.llm.probe_chat", return_value=ChatProbe(200, "", "OK")) as probe, \
                    patch("app.llm.LLMClient"), patch.object(batch_draft, "log"), \
                    patch("app.medical_review.review_medical_records", return_value=MedicalDigest(pages_reviewed=1)) as review:
                self.assertEqual(batch_draft.main(arguments), 0)
            probe.assert_called_once(); review.assert_called_once()
            self.assertEqual([d.filename for d in review.call_args.args[1]], ["Part1.txt"])

    def test_legacy_unbound_facts_can_finalize_with_current_usable_inputs(self):
        from tests.test_batch_draft import TestFinalPhaseSemantics
        result = TestFinalPhaseSemantics()._run_final("ok", legacy_unknown_only=True)
        self.assertEqual(result["legacy_source_facts_unresolved"], 1)
        self.assertIn("Legacy source coverage", result["grounding_markdown"])
        self.assertTrue(result["statement"])

    def test_legacy_unbound_facts_without_usable_inputs_never_call_model(self):
        from tests.test_batch_draft import batch_draft, _make_cfg
        from app.medical_review import MedicalDigest, MedicalFact
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(Path(tmp)); llm = MagicMock()
            state = batch_draft.digest_to_state(MedicalDigest(facts=[MedicalFact("2023", "symptom", "Legacy fact", "p. 1")]))
            with self.assertRaises(RequestValidationError):
                batch_draft.final_phase(llm, cfg, {"legacy": state})
            llm.chat.assert_not_called(); llm.chat_json.assert_not_called()

    def test_invalid_batch_config_stops_both_programmatic_paths(self):
        from tests.test_batch_draft import batch_draft, _make_cfg
        with tempfile.TemporaryDirectory() as tmp:
            for fields in ({"observations": " "}, {"condition": " "}, {"claim_type": []}, {"witness": {"name": 1}}):
                cfg = _make_cfg(Path(tmp), **fields); llm = MagicMock()
                with self.subTest(fields=list(fields)), patch("app.documents.records_from_local_path") as extract:
                    with self.assertRaises(RequestValidationError): batch_draft.digest_group(llm, cfg, "synthetic", [Path("a.txt")])
                    with self.assertRaises(RequestValidationError): batch_draft.final_phase(llm, cfg, {})
                    extract.assert_not_called()
                llm.chat.assert_not_called(); llm.chat_json.assert_not_called()

    def test_wrong_json_types_and_blank_required_fields_stop_cli_before_probe(self):
        from tests.test_batch_draft import batch_draft
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); records = root / "records"; records.mkdir(); source = root / "request.json"
            for field, value in (("observations", " "), ("observations", 7), ("condition", " "),
                                 ("claim_type", None), ("witness", {"name": False})):
                source.write_text(json.dumps(dict(observations="Observed pain.", condition="Knee", claim_type="Service", witness={}) | {field: value}))
                with self.subTest(field=field), patch("app.llm.probe_chat") as probe, patch.object(batch_draft, "log"):
                    code = batch_draft.main(["--records", str(records), "--out", str(root / "out"), "--witness-json", str(source)])
                    self.assertEqual(code, 2); probe.assert_not_called()
                self.assertFalse((root / "out").exists())

    def test_blank_cli_record_never_probes_provider(self):
        from tests.test_batch_draft import batch_draft
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); records = root / "records"; records.mkdir()
            (records / "Part1.txt").write_text(" ")
            observations = root / "observations.txt"; observations.write_text("Observed pain.")
            with patch("app.llm.probe_chat") as probe, patch.object(batch_draft, "log"):
                code = batch_draft.main(["--records", str(records), "--glob", "*.txt", "--out", str(root / "out"),
                                         "--condition", "Knee", "--claim-type", "Service", "--observations", str(observations)])
            self.assertEqual(code, 2); probe.assert_not_called()

    def test_empty_extracted_batch_stops_before_model_review(self):
        from tests.test_batch_draft import batch_draft, _make_cfg
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(Path(tmp)); source = cfg.records_dir / "Part1.txt"; source.write_text(" ")
            with patch("app.documents.records_from_local_path", return_value=([], [])), \
                    patch("app.medical_review.review_medical_records") as review:
                with self.assertRaises(RequestValidationError): batch_draft.digest_group(MagicMock(), cfg, "synthetic", [source])
                review.assert_not_called()


if __name__ == "__main__":
    unittest.main()
