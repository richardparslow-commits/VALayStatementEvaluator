"""R15 regressions: excluded modes, stale replies and submission/result association."""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import unittest
from contextlib import ExitStack, redirect_stderr
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from tests import hermetic  # noqa: F401
from tests import test_job_queue as fixtures
from tests import test_job_runner as ui_fixtures
from app import config, health, job_payload as payload, job_queue as queue, pilot, worker
from app.queue_policy import REFUSAL, synthetic_queue
from app.views import job_runner as ui

ROOT = Path(__file__).resolve().parent.parent
EXCLUDED = ("controlled-pilot", "controlled-pilto", "", "production")


class PilotQueueTests(unittest.TestCase):
    def setUp(self):
        self.mode = patch.dict(os.environ, VA_LSE_MODE="synthetic")
        self.mode.start()
        self.addCleanup(self.mode.stop)
        self.saved = dict(ui.st.session_state)
        self.addCleanup(self.restore_session)

    def restore_session(self):
        ui.st.session_state.clear()
        ui.st.session_state.update(self.saved)

    def refuse(self, function):
        with self.assertRaises(pilot.PilotBlocked) as caught:
            function()
        self.assertEqual(str(caught.exception), REFUSAL)

    def test_constructors_and_cached_factory_refuse_before_transport(self):
        cached = queue.InProcessJobBackend(prefix="invented", ttl_seconds=60)
        with patch.object(queue, "_backend", cached), patch("redis.Redis.from_url") as connection, \
                patch("urllib.request.urlopen") as http:
            for mode in EXCLUDED:
                with self.subTest(mode=mode), patch.dict(os.environ, VA_LSE_MODE=mode):
                    for call in (queue.get_job_backend, queue.build_job_backend,
                                 lambda: queue.InProcessJobBackend(prefix="x", ttl_seconds=60),
                                 lambda: queue.RedisJobBackend("redis://invented.invalid", prefix="x", ttl_seconds=60),
                                 lambda: queue.UpstashJobBackend("https://invented.invalid", "invented", prefix="x",
                                                               ttl_seconds=60, timeout_seconds=1)):
                        self.refuse(call)
            connection.assert_not_called()
            http.assert_not_called()

    def test_existing_backends_refuse_all_job_operations_without_changes(self):
        redis = fixtures._FakeRedisClient()
        transport = fixtures._FakeUpstashTransport()
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, redis=fixtures._FakeRedisModule(redis)))
            stack.enter_context(patch("urllib.request.urlopen", transport.urlopen))
            backends = [queue.InProcessJobBackend(prefix="invented", ttl_seconds=60),
                        queue.RedisJobBackend("redis://invented.invalid", prefix="invented", ttl_seconds=60),
                        queue.UpstashJobBackend("https://invented.invalid", "invented", prefix="invented",
                                                ttl_seconds=60, timeout_seconds=1)]
            for backend in backends:
                with self.subTest(backend=backend.name):
                    record = backend.enqueue("evaluate", "INVENTED_INPUT_CANARY", request_id="req_invented")
                    with patch.object(config, "JOB_QUEUE_CLAIM_TIMEOUT_SECONDS", .01):
                        claimed, _ = backend.claim(["evaluate"], worker_id="invented")
                    backend.store_result(record.job_id, "INVENTED_RESULT_CANARY", claim_token=claimed.claim_token)
                    commands = len(transport.commands)
                    calls = [lambda: backend.enqueue("evaluate", "new"),
                             lambda: backend.claim(["evaluate"], worker_id="new"),
                             lambda: backend.get(record.job_id), lambda: backend.get_result(record.job_id),
                             lambda: backend.set_progress(record.job_id, .5, "private", claim_token=claimed.claim_token),
                             lambda: backend.store_result(record.job_id, "new", claim_token=claimed.claim_token),
                             lambda: backend.complete(record.job_id, claim_token=claimed.claim_token),
                             lambda: backend.fail(record.job_id, error="private", claim_token=claimed.claim_token),
                             lambda: backend.requeue(record.job_id, claim_token=claimed.claim_token),
                             backend.requeue_stale, backend.depth, backend.retained_blob_keys, backend.ping,
                             lambda: backend.lookup_by_request_id("req_invented"),
                             lambda: backend.set_recovery_index("req_new", record.job_id)]
                    for mode in EXCLUDED:
                        with patch.dict(os.environ, VA_LSE_MODE=mode):
                            for call in calls:
                                self.refuse(call)
                            self.assertEqual(backend.health(probe=True)["backend"], "excluded")
                    self.assertEqual(backend.get_result(record.job_id), "INVENTED_RESULT_CANARY")
                    self.assertEqual(backend.get(record.job_id).claim_token, claimed.claim_token)
                    self.assertEqual(len(transport.commands), commands + (2 if backend.name == "upstash_rest" else 0))

    def test_serialization_refuses_before_blob_or_case_access(self):
        blob = MagicMock()
        calls = [lambda: payload.validate_job("evaluate", None), lambda: payload.documents_bundle(None),
                 lambda: payload.encode_job("evaluate", None), lambda: payload.payload_needs_blob("evaluate", None),
                 lambda: payload.encode_job_with_blob("evaluate", None, None),
                 lambda: payload.decode_job("evaluate", "INVENTED_PRIVATE_CANARY", blob_store=blob),
                 lambda: payload.encode_result(None), lambda: payload.decode_result("INVENTED_PRIVATE_CANARY")]
        for mode in EXCLUDED:
            with patch.dict(os.environ, VA_LSE_MODE=mode, VA_LSE_JOB_QUEUE="1", VA_LSE_QUEUE_APPROVED="1"):
                for call in calls:
                    self.refuse(call)
        self.assertEqual(blob.mock_calls, [])

    def test_worker_direct_entrypoints_and_progress_refuse_injected_clients(self):
        backend, llm = MagicMock(), MagicMock()
        record = queue.JobRecord(job_id="invented", kind="evaluate")
        calls = [worker.build_llm, lambda: worker.execute_job(record, "private", backend, llm=llm),
                 lambda: worker.run_worker(once=True, backend=backend, llm=llm),
                 lambda: worker.drain_stale(backend), worker._start_health_server,
                 lambda: worker._progress_callback(backend, record)(1, "private")]
        with patch.dict(os.environ, VA_LSE_MODE="controlled-pilot"):
            for call in calls:
                self.refuse(call)
        self.assertEqual(backend.mock_calls + llm.mock_calls, [])

    def test_worker_cli_refuses_before_logging_tracing_health_or_backend(self):
        names = ("configure_logging", "get_job_backend", "_start_health_server", "build_llm")
        with ExitStack() as stack:
            mocks = [stack.enter_context(patch.object(worker, name)) for name in names]
            mocks.append(stack.enter_context(patch.object(worker.tracing, "setup_tracing")))
            for mode in EXCLUDED:
                with patch.dict(os.environ, VA_LSE_MODE=mode):
                    for args in (["--once"], ["--drain-stale"], []):
                        out = io.StringIO()
                        with redirect_stderr(out):
                            self.assertEqual(worker.main(args), 2)
                        self.assertEqual(out.getvalue().strip(), REFUSAL)
            for mock in mocks:
                mock.assert_not_called()

    def test_ui_sensitive_entrypoints_refuse_without_hydration_or_queue_access(self):
        backend, run = MagicMock(), MagicMock()
        record = queue.JobRecord(job_id="invented", kind="evaluate")
        calls = [lambda: ui._hydrate("eval", run), lambda: ui._fetch_outcome(backend, record, "eval"),
                 lambda: ui._poll(backend, "invented", "eval", wait_seconds=1),
                 lambda: ui._encode_payload("evaluate", None), ui.worker_config_error,
                 lambda: ui.recover_job_by_request_id("eval", "req_invented", action_label="Evaluation"),
                 lambda: ui._render_uncertain_submission("eval"),
                 lambda: ui._render_failure(ui.QueueOutcome(ok=False, error="private"), "Evaluation"),
                 lambda: ui.submit_job(slot="eval", job=None, request_id="req_invented", condition=None,
                                       sources=[], files=0, pages=0, action_label="Evaluation")]
        before = dict(ui.st.session_state)
        with patch.dict(os.environ, VA_LSE_MODE="controlled-pilot"), patch.object(ui, "get_job_backend") as get, \
                patch.object(ui, "get_blob_store") as blobs, patch.object(pilot, "display") as display:
            for call in calls:
                self.refuse(call)
            self.assertFalse(ui.queue_mode_active())
            ui.resume_pending_job("eval", action_label="Evaluation")
            ui.render_recovery_form("eval", action_label="Evaluation")
            self.assertEqual(ui.queue_status_line(), "queued processing excluded")
            get.assert_not_called()
            blobs.assert_not_called()
            display.assert_not_called()
        self.assertEqual(dict(ui.st.session_state), before)
        self.assertEqual(backend.mock_calls, [])

    def test_health_never_initializes_or_probes_excluded_queue(self):
        with patch.dict(os.environ, VA_LSE_MODE="controlled-pilot"), patch.object(queue, "get_job_backend") as get:
            status = health._health_payload(probe_queue=True)["job_queue"]
        self.assertEqual(status["backend"], "excluded")
        self.assertFalse(status["enabled"])
        self.assertIsNone(status["depth"])
        get.assert_not_called()

    def test_late_reply_and_exception_replace_private_contents_with_fixed_refusal(self):
        for error in (False, True):
            @synthetic_queue
            def delayed():
                os.environ["VA_LSE_MODE"] = "controlled-pilot"
                if error:
                    raise RuntimeError("INVENTED_PRIVATE_CANARY")
                return "INVENTED_PRIVATE_CANARY"
            with patch.dict(os.environ, VA_LSE_MODE="synthetic"):
                self.refuse(delayed)

    def test_late_result_is_not_decoded_or_hydrated(self):
        backend = MagicMock()
        record = queue.JobRecord(job_id="invented", kind="evaluate", status="done")
        def late(_):
            os.environ["VA_LSE_MODE"] = "controlled-pilot"
            return "INVENTED_PRIVATE_CANARY"
        backend.get_result.side_effect = late
        with patch.object(pilot, "owns", return_value=True), patch.object(ui, "decode_result") as decode, \
                patch.object(ui, "_hydrate") as hydrate:
            self.refuse(lambda: ui._fetch_outcome(backend, record, "eval"))
            decode.assert_not_called()
            hydrate.assert_not_called()

    def test_late_poll_error_is_not_retried_or_displayed(self):
        backend = MagicMock()
        def late(_):
            os.environ["VA_LSE_MODE"] = "controlled-pilot"
            raise RuntimeError("INVENTED_PRIVATE_CANARY")
        backend.get.side_effect = late
        with patch.object(ui, "run_log_event") as logs, patch.object(ui.time, "sleep") as sleep:
            self.refuse(lambda: ui._poll(backend, "invented", "eval", wait_seconds=1))
            self.assertEqual(backend.get.call_count, 1)
            logs.assert_not_called()
            sleep.assert_not_called()

    def test_late_worker_claim_and_progress_errors_are_not_logged_or_executed(self):
        record = queue.JobRecord(job_id="invented", kind="evaluate")
        for operation in ("claim", "set_progress", "requeue_stale"):
            with patch.dict(os.environ, VA_LSE_MODE="synthetic"):
                backend = MagicMock()
                def late(*a, **k):
                    os.environ["VA_LSE_MODE"] = "controlled-pilot"
                    raise queue.JobQueueError("INVENTED_PRIVATE_CANARY")
                getattr(backend, operation).side_effect = late
                with patch.object(worker.logger, "error") as errors, patch.object(worker.logger, "warning") as warnings, \
                        patch.object(worker, "execute_job") as execute:
                    call = (lambda: worker.run_worker(once=True, backend=backend)) if operation == "claim" else (
                        lambda: worker._progress_callback(backend, record)(1, "private")) if operation == "set_progress" else (
                        lambda: worker.drain_stale(backend))
                    self.refuse(call)
                    errors.assert_not_called()
                    warnings.assert_not_called()
                    execute.assert_not_called()

    def test_late_redis_reply_refuses_existing_transport(self):
        client = fixtures._FakeRedisClient()
        with patch.dict(sys.modules, redis=fixtures._FakeRedisModule(client)):
            backend = queue.RedisJobBackend("redis://invented.invalid", prefix="invented", ttl_seconds=60)
        def late(*a, **k):
            os.environ["VA_LSE_MODE"] = "controlled-pilot"
            return "INVENTED_PRIVATE_CANARY"
        with patch.object(client, "get", side_effect=late):
            self.refuse(lambda: backend.get_result("invented"))

    def test_late_claim_success_cannot_start_pipeline(self):
        backend = MagicMock()
        def late(*a, **k):
            os.environ["VA_LSE_MODE"] = "controlled-pilot"
            return queue.JobRecord(job_id="invented", kind="evaluate"), "INVENTED_PRIVATE_CANARY"
        backend.claim.side_effect = late
        with patch.object(worker, "execute_job") as execute:
            self.refuse(lambda: worker.run_worker(once=True, backend=backend))
        execute.assert_not_called()

    def test_late_enqueue_failure_cannot_display_or_log_private_error(self):
        backend = MagicMock()
        def late(*a, **k):
            os.environ["VA_LSE_MODE"] = "controlled-pilot"
            raise RuntimeError("INVENTED_PRIVATE_CANARY")
        backend.enqueue.side_effect = late
        with patch.object(ui, "get_job_backend", return_value=backend), \
                patch.object(ui, "run_log_event") as logs, patch.object(ui, "report_failure") as report:
            self.refuse(lambda: ui.submit_job(slot="eval", job=ui_fixtures._job("req_invented"),
                        request_id="req_invented", condition=None, sources=[], files=0, pages=0,
                        action_label="Evaluation"))
        logs.assert_not_called()
        report.assert_not_called()

    def test_bad_submission_associations_fail_before_blob_decoding_or_provider(self):
        valid = payload.encode_job("evaluate", ui_fixtures._job("req_invented"))
        base = queue.JobRecord(job_id="invented", kind="evaluate", request_id="req_invented",
                               submission_digest=hashlib.sha256(valid.encode()).hexdigest())
        cases = [(base, valid + " "), (replace(base, submission_digest=""), valid)]
        for field, value in (("request_id", "req_other"), ("version", True), ("version", 2), ("kind", "draft")):
            envelope = json.loads(valid)
            envelope[field] = value
            raw = json.dumps(envelope)
            cases.append((replace(base, submission_digest=hashlib.sha256(raw.encode()).hexdigest()), raw))
        with patch.object(worker, "decode_job") as decode, patch.object(worker, "get_blob_store") as blobs, \
                patch.object(worker, "build_llm") as llm, patch.object(worker, "_fail") as fail:
            for record, raw in cases:
                self.assertFalse(worker.execute_job(record, raw, MagicMock()))
                self.assertIsInstance(fail.call_args.args[2], payload.PayloadError)
            decode.assert_not_called()
            blobs.assert_not_called()
            llm.assert_not_called()
        worker.verify_submission(base, valid)

    def test_wrong_slot_or_result_reference_never_hydrates(self):
        record = queue.JobRecord(job_id="invented", kind="evaluate", request_id="req_invented", status="done")
        backend = MagicMock()
        with patch.object(pilot, "owns", return_value=True), patch.object(ui, "_hydrate") as hydrate:
            for slot in ("draft", "unknown"):
                self.assertFalse(ui._fetch_outcome(backend, record, slot).ok)
            backend.get_result.assert_not_called()
            backend.get_result.return_value = ui_fixtures._result_json("req_other")
            self.assertEqual(ui._fetch_outcome(backend, record, "eval").error_class, "PayloadError")
            wrong_kind = json.loads(ui_fixtures._result_json("req_invented"))
            wrong_kind["kind"] = "draft"
            backend.get_result.return_value = json.dumps(wrong_kind)
            self.assertEqual(ui._fetch_outcome(backend, record, "eval").error_class, "PayloadError")
            backend.get_result.return_value = ui_fixtures._result_json("req_invented")
            self.assertTrue(ui._fetch_outcome(backend, record, "eval").ok)
            hydrate.assert_called_once()

    def test_other_session_is_refused_before_stored_result_read(self):
        backend = MagicMock()
        with patch.object(pilot, "owns", return_value=False):
            out = ui._fetch_outcome(backend, queue.JobRecord(job_id="invented", kind="evaluate"), "eval")
        self.assertEqual(out.error_class, "AccessDenied")
        backend.get_result.assert_not_called()

    def test_mode_change_in_advisory_watchdog_prevents_session_writes(self):
        before = dict(ui.st.session_state)
        def late(_):
            os.environ["VA_LSE_MODE"] = "controlled-pilot"
        with patch("app.views.usage.record_watchdog_run", side_effect=late):
            self.refuse(lambda: ui._hydrate("eval", MagicMock()))
        self.assertEqual(dict(ui.st.session_state), before)

    def test_policy_import_has_no_application_clients_or_ui_side_effects(self):
        result = subprocess.run([sys.executable, "-c",
            'import sys; import app.queue_policy; assert not {"app.config", "app.llm", "streamlit", "openai", "app.telemetry"}.intersection(sys.modules)'],
            cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
