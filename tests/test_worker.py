"""Offline tests for the worker execution path (app/worker.py).

A deterministic fake LLM stands in for LLMClient (the same phase-dispatch stub
tests/test_evaluate.py uses, wrapped so it also records usage — the worker ships
usage back to the UI). Tests cover the whole job lifecycle: success, payload
failure, pipeline failure, worker draining, and the stale-lease sweep.

Uses ``tests/test_evaluate.py``'s stub rather than a second copy of the phase
responses, so a change to the evaluate pipeline's prompt phases breaks one place.
"""
import json
import os
import sys
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from log_isolation import isolate_app_logs  # noqa: E402

from app import config  # noqa: E402
from app import shutdown  # noqa: E402
from app import worker  # noqa: E402
from app.documents import document_from_text  # noqa: E402
from app.job_payload import (  # noqa: E402
    KIND_DRAFT,
    KIND_EVALUATE,
    DraftJob,
    EvaluateJob,
    decode_result,
    encode_job,
)
from app.job_queue import (  # noqa: E402
    STATUS_DONE,
    STATUS_ERROR,
    InProcessJobBackend,
)
from app.usage import UsageTracker  # noqa: E402


def _base_stub() -> Any:
    """The phase-dispatch LLM stub from tests/test_evaluate.py."""
    from test_evaluate import _FakeLLM

    return _FakeLLM()


class _UsageStub:
    """Wraps the evaluate stub so it also records usage like LLMClient does."""

    def __init__(self, base: Any | None = None) -> None:
        self._base = base if base is not None else _base_stub()
        # The record-review path reads llm._settings.model_fast to pick the cheap
        # digest model, so a stub without it fails every job.
        self._settings = MagicMock(model_fast="fake-fast", model_main="fake-main")
        self.usage = UsageTracker()
        self.fail_with: BaseException | None = None

    def _record(self, system: str, user: str, kwargs: dict, content: str) -> None:
        self.usage.record(
            model=kwargs.get("model") or "fake-fast",
            phase=kwargs.get("phase", "general"),
            system=system,
            user=user,
            content=content,
            prompt_tokens=None,
            completion_tokens=None,
        )

    def chat_json(self, system: str, user: str, **kwargs: Any) -> Any:
        if self.fail_with is not None:
            raise self.fail_with
        out = self._base.chat_json(system, user, **kwargs)
        self._record(system, user, kwargs, json.dumps(out, default=str))
        return out

    def chat(self, system: str, user: str, **kwargs: Any) -> Any:
        if self.fail_with is not None:
            raise self.fail_with
        out = self._base.chat(system, user, **kwargs)
        self._record(system, user, kwargs, str(out))
        return out


def _docs():
    return [document_from_text("a.txt", "EVT knee pain noted during service.\n\nEVT brace prescribed.")]


def _evaluate_job(request_id: str = "req_job1") -> EvaluateJob:
    return EvaluateJob(
        statement_text="I injured my knee lifting a pallet in 2014.",
        records=_docs(),
        request_id=request_id,
        record_sources=["Upload"],
    )


def _draft_job(request_id: str = "req_job2") -> DraftJob:
    return DraftJob(
        records=_docs(),
        witness={"name": "Jane Doe", "relationship": "Spouse"},
        observations="Daily knee pain.",
        condition="knee strain",
        claim_type="Service connection (new claim)",
        request_id=request_id,
        record_sources=["Upload"],
    )


@contextmanager
def _fast_queue():
    """Make claims non-blocking so the suite never sleeps on the real default."""
    with patch.object(config, "JOB_QUEUE_CLAIM_TIMEOUT_SECONDS", 0.05), patch.object(
        config, "JOB_QUEUE_POLL_SECONDS", 0.01
    ), patch.object(config, "JOB_QUEUE_LEASE_SECONDS", 60):
        yield


class _WorkerCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = isolate_app_logs(self)
        shutdown.reset_for_tests()
        self.addCleanup(shutdown.reset_for_tests)
        self.backend = InProcessJobBackend(prefix="test", ttl_seconds=60)


class TestExecuteJobSuccess(_WorkerCase):
    def test_progress_stops_when_the_attempt_is_reclaimed(self):
        self.backend.enqueue(KIND_EVALUATE, "{}")
        old, _ = self.backend.claim([KIND_EVALUATE], worker_id="old")
        callback = worker._progress_callback(self.backend, old)
        with patch.object(config, "JOB_QUEUE_LEASE_SECONDS", 0):
            self.backend.requeue_stale()
        current, _ = self.backend.claim([KIND_EVALUATE], worker_id="new")
        with self.assertRaises(worker.JobLeaseLost):
            callback(1.0, "late progress")
        self.assertEqual(self.backend.get(old.job_id).claim_token, current.claim_token)

    def test_lost_ownership_at_completion_is_not_reported_as_success(self):
        job = _evaluate_job()
        self.backend.enqueue(KIND_EVALUATE, encode_job(KIND_EVALUATE, job))
        old, payload = self.backend.claim([KIND_EVALUATE], worker_id="old")
        original = self.backend.complete

        def reclaimed_complete(job_id, **kwargs):
            with patch.object(config, "JOB_QUEUE_LEASE_SECONDS", 0):
                self.backend.requeue_stale()
            self.backend.claim([KIND_EVALUATE], worker_id="new")
            original(job_id, **kwargs)

        with patch.object(self.backend, "complete", side_effect=reclaimed_complete):
            self.assertFalse(worker.execute_job(old, payload, self.backend, llm=_UsageStub()))
        self.assertEqual(self.backend.get(old.job_id).worker_id, "new")
        self.assertEqual(self.backend.get(old.job_id).status, "running")
        self.assertIsNone(self.backend.get_result(old.job_id))

    def test_evaluate_job_stores_a_decodable_result(self):
        job = _evaluate_job()
        record = self.backend.enqueue(KIND_EVALUATE, encode_job(KIND_EVALUATE, job), request_id=job.request_id)
        claimed = self.backend.claim([KIND_EVALUATE], worker_id="w1")
        self.assertIsNotNone(claimed)

        ok = worker.execute_job(claimed[0], claimed[1], self.backend, llm=_UsageStub())

        self.assertTrue(ok)
        done = self.backend.get(record.job_id)
        self.assertEqual(done.status, STATUS_DONE)
        self.assertEqual(done.progress, 1.0)
        run = decode_result(self.backend.get_result(record.job_id))
        self.assertEqual(run.kind, KIND_EVALUATE)
        self.assertEqual(run.request_id, "req_job1")
        self.assertEqual(run.result.overall_rating, "Adequate")
        # Usage must travel with the result: the results panel reports on it.
        self.assertGreater(run.usage.totals().calls, 0)

    def test_draft_job_stores_a_decodable_result(self):
        job = _draft_job()
        record = self.backend.enqueue(KIND_DRAFT, encode_job(KIND_DRAFT, job), request_id=job.request_id)
        claimed = self.backend.claim([KIND_DRAFT], worker_id="w1")

        ok = worker.execute_job(claimed[0], claimed[1], self.backend, llm=_UsageStub())

        self.assertTrue(ok)
        self.assertEqual(self.backend.get(record.job_id).status, STATUS_DONE)
        run = decode_result(self.backend.get_result(record.job_id))
        self.assertEqual(run.kind, KIND_DRAFT)
        self.assertTrue(run.result.output_statement)

    def test_progress_is_persisted_while_running(self):
        job = _evaluate_job()
        record = self.backend.enqueue(KIND_EVALUATE, encode_job(KIND_EVALUATE, job))
        claimed = self.backend.claim([KIND_EVALUATE], worker_id="w1")

        seen: list[float] = []
        real_set_progress = self.backend.set_progress

        def spy(job_id: str, progress: float, message: str, *, claim_token: str) -> None:
            seen.append(progress)
            real_set_progress(job_id, progress, message, claim_token=claim_token)

        self.backend.set_progress = spy  # type: ignore[method-assign]
        worker.execute_job(claimed[0], claimed[1], self.backend, llm=_UsageStub())

        self.assertTrue(seen, "expected the pipeline to report progress")
        # Progress must advance (the phase offsets push it past the digest start)
        # and the record must land at 1.0 so the polling UI never sticks.
        self.assertGreater(max(seen), min(seen))
        self.assertEqual(self.backend.get(record.job_id).progress, 1.0)


class TestExecuteJobFailures(_WorkerCase):
    def test_timeout_fails_promptly_and_rejects_late_progress_and_result(self):
        job = _evaluate_job()
        record = self.backend.enqueue(KIND_EVALUATE, encode_job(KIND_EVALUATE, job))
        claimed = self.backend.claim([KIND_EVALUATE], worker_id="w1")
        release = threading.Event()
        finished = threading.Event()

        def blocked(kind, job, llm, progress):
            try:
                release.wait(2)
                progress(1.0, "late success")
                return MagicMock()
            finally:
                finished.set()

        with (
            patch.object(worker, "_run_pipeline", side_effect=blocked),
            patch("app.pipeline_guard._pipeline_timeout_seconds", return_value=0.1),
            patch.object(self.backend, "set_progress", wraps=self.backend.set_progress) as progress,
            patch.object(self.backend, "store_result", wraps=self.backend.store_result) as store,
            patch.object(self.backend, "complete", wraps=self.backend.complete) as complete,
        ):
            try:
                t0 = time.monotonic()
                ok = worker.execute_job(claimed[0], claimed[1], self.backend, llm=_UsageStub())
                self.assertLess(time.monotonic() - t0, 0.5)
                self.assertFalse(ok)
                self.assertEqual(self.backend.get(record.job_id).error_class, "PipelineTimeoutError")
            finally:
                release.set()
                self.assertTrue(finished.wait(2))
            progress.assert_not_called()
            store.assert_not_called()
            complete.assert_not_called()
        self.assertEqual(self.backend.get(record.job_id).status, STATUS_ERROR)
        self.assertIsNone(self.backend.get_result(record.job_id))

    def test_payload_that_cannot_decode_fails_the_job(self):
        record = self.backend.enqueue(KIND_EVALUATE, "{not a payload}")
        claimed = self.backend.claim([KIND_EVALUATE], worker_id="w1")

        ok = worker.execute_job(claimed[0], claimed[1], self.backend, llm=_UsageStub())

        self.assertFalse(ok)
        failed = self.backend.get(record.job_id)
        self.assertEqual(failed.status, STATUS_ERROR)
        self.assertEqual(failed.error_class, "PayloadError")
        self.assertIsNone(self.backend.get_result(record.job_id))

    def test_pipeline_exception_fails_the_job_and_is_recorded(self):
        from app.llm import LLMError

        job = _evaluate_job()
        record = self.backend.enqueue(KIND_EVALUATE, encode_job(KIND_EVALUATE, job))
        claimed = self.backend.claim([KIND_EVALUATE], worker_id="w1")
        stub = _UsageStub()
        stub.fail_with = LLMError("endpoint down")

        ok = worker.execute_job(claimed[0], claimed[1], self.backend, llm=stub)

        self.assertFalse(ok)
        failed = self.backend.get(record.job_id)
        self.assertEqual(failed.status, STATUS_ERROR)
        self.assertEqual(failed.error_class, "LLMError")
        self.assertIn("endpoint down", failed.error)

    def test_failure_is_written_to_the_run_log(self):
        job = _evaluate_job(request_id="req_failing")
        record = self.backend.enqueue(KIND_EVALUATE, encode_job(KIND_EVALUATE, job))
        claimed = self.backend.claim([KIND_EVALUATE], worker_id="w1")
        stub = _UsageStub()
        stub.fail_with = RuntimeError("pipeline exploded")

        worker.execute_job(claimed[0], claimed[1], self.backend, llm=stub)

        runs_log = Path(self._tmpdir) / "runs.jsonl"
        self.assertTrue(runs_log.exists())
        events = [json.loads(line) for line in runs_log.read_text().splitlines() if line.strip()]
        matching = [e for e in events if e.get("request_id") == "req_failing"]
        self.assertTrue(matching, f"expected a run-log event, got {events}")
        self.assertEqual(matching[-1]["status"], "error")

    def test_unconfigured_llm_is_a_clear_job_failure(self):
        job = _evaluate_job()
        record = self.backend.enqueue(KIND_EVALUATE, encode_job(KIND_EVALUATE, job))
        claimed = self.backend.claim([KIND_EVALUATE], worker_id="w1")
        settings = MagicMock(configured=False)

        with patch.object(config, "load_settings", return_value=settings):
            ok = worker.execute_job(claimed[0], claimed[1], self.backend)

        self.assertFalse(ok)
        failed = self.backend.get(record.job_id)
        self.assertEqual(failed.error_class, "WorkerConfigError")
        self.assertIn("queue mode cannot use a key typed into the sidebar", failed.error.lower())

    def test_store_failure_fails_the_job_rather_than_reporting_success(self):
        job = _evaluate_job()
        record = self.backend.enqueue(KIND_EVALUATE, encode_job(KIND_EVALUATE, job))
        claimed = self.backend.claim([KIND_EVALUATE], worker_id="w1")

        def boom(*_args: Any, **_kwargs: Any) -> None:
            raise OSError("redis went away")

        with patch.object(self.backend, "store_result", boom):
            ok = worker.execute_job(claimed[0], claimed[1], self.backend, llm=_UsageStub())

        self.assertFalse(ok)
        self.assertEqual(self.backend.get(record.job_id).status, STATUS_ERROR)


class TestRunWorkerLoop(_WorkerCase):
    def test_once_returns_immediately_when_idle(self):
        with _fast_queue():
            stats = worker.run_worker(once=True, backend=self.backend, llm=_UsageStub())
        self.assertEqual(stats.claimed, 0)
        self.assertEqual(stats.completed, 0)

    def test_worker_drains_one_enqueued_job(self):
        job = _evaluate_job()
        record = self.backend.enqueue(KIND_EVALUATE, encode_job(KIND_EVALUATE, job))
        with _fast_queue():
            stats = worker.run_worker(
                backend=self.backend, llm=_UsageStub(), max_jobs=1
            )
        self.assertEqual(stats.claimed, 1)
        self.assertEqual(stats.completed, 1)
        self.assertEqual(self.backend.get(record.job_id).status, STATUS_DONE)

    def test_worker_serves_both_job_kinds(self):
        eval_job = _evaluate_job()
        draft_job = _draft_job()
        self.backend.enqueue(KIND_EVALUATE, encode_job(KIND_EVALUATE, eval_job))
        self.backend.enqueue(KIND_DRAFT, encode_job(KIND_DRAFT, draft_job))
        with _fast_queue():
            stats = worker.run_worker(backend=self.backend, llm=_UsageStub(), max_jobs=2)
        self.assertEqual(stats.claimed, 2)
        self.assertEqual(stats.failed, 0)

    def test_draining_worker_requeues_instead_of_failing(self):
        """A job claimed while draining must reach a healthy worker untouched."""
        job = _evaluate_job()
        record = self.backend.enqueue(KIND_EVALUATE, encode_job(KIND_EVALUATE, job))
        with _fast_queue(), patch.object(shutdown, "enter_run", return_value=False):
            stats = worker.run_worker(backend=self.backend, llm=_UsageStub(), max_jobs=1)

        self.assertEqual(stats.claimed, 1)
        self.assertEqual(stats.requeued, 1)
        requeued = self.backend.get(record.job_id)
        self.assertNotEqual(requeued.status, STATUS_ERROR)
        # Re-queueing gives the attempt back: the job never actually ran here,
        # so a worker restart must not burn its retry budget.
        self.assertEqual(requeued.attempts, 0)
        self.assertEqual(self.backend.depth(), 1)

    def test_worker_stops_when_the_stop_event_is_set(self):
        stopper = threading.Event()
        stopper.set()
        with _fast_queue():
            stats = worker.run_worker(
                backend=self.backend, llm=_UsageStub(), stop_event=stopper
            )
        self.assertEqual(stats.claimed, 0)

    def test_claim_transport_failure_is_retried_not_fatal(self):
        """A Redis blip must not kill the worker's claim loop.

        redis-py raises its own exception tree; the backend normalizes it to
        JobQueueError precisely so this path logs and keeps retrying instead of
        exiting with a traceback and leaving the queue unconsumed.
        """
        from app.job_queue import JobQueueError

        calls = {"n": 0}
        real_claim = self.backend.claim

        def flaky(kinds, *, worker_id):  # noqa: ANN001, ANN202
            calls["n"] += 1
            if calls["n"] >= 2:
                raise JobQueueError("redis claim failed: ConnectionError: refused")
            return real_claim(kinds, worker_id=worker_id)

        stopper = threading.Event()

        def stop_after_a_moment() -> None:
            time.sleep(0.4)
            stopper.set()

        with patch.object(self.backend, "claim", flaky), _fast_queue(), self.assertLogs(
            "app.worker", level="ERROR"
        ) as captured:
            threading.Thread(target=stop_after_a_moment, daemon=True).start()
            stats = worker.run_worker(backend=self.backend, llm=_UsageStub(), stop_event=stopper)

        self.assertGreaterEqual(calls["n"], 2, "the worker should have retried")
        self.assertTrue(any("claim failed" in line for line in captured.output))
        self.assertTrue(stats.errors, "the failure should be recorded on the stats")

    def test_claim_error_list_is_bounded(self):
        """A permanently broken backend must not grow the error list forever."""
        from app.job_queue import JobQueueError

        def always_fails(kinds, *, worker_id):  # noqa: ANN001, ANN202
            raise JobQueueError("redis claim failed: ConnectionError: refused")

        stopper = threading.Event()

        def stop_after_a_moment() -> None:
            time.sleep(0.3)
            stopper.set()

        with patch.object(self.backend, "claim", always_fails), _fast_queue():
            threading.Thread(target=stop_after_a_moment, daemon=True).start()
            stats = worker.run_worker(backend=self.backend, llm=_UsageStub(), stop_event=stopper)
        self.assertLessEqual(len(stats.errors), 20)

    def test_non_distributed_backend_logs_a_warning(self):
        with _fast_queue(), self.assertLogs("app.worker", level="WARNING") as captured:
            worker.run_worker(once=True, backend=self.backend, llm=_UsageStub())
        self.assertTrue(any("non-distributed backend" in line for line in captured.output))


class TestBlobReferencedJob(_WorkerCase):
    """A job whose records were externalized must be resolvable on the worker."""

    def _externalize(self, job, tmp_root):
        from app.blob_store import FilesystemBlobStore
        from app.job_payload import documents_bundle, encode_job_with_blob

        store = FilesystemBlobStore(tmp_root)
        ref = store.put(documents_bundle(job))
        return store, encode_job_with_blob(KIND_EVALUATE, job, ref)

    def test_worker_resolves_documents_from_the_blob_store(self):
        from tempfile import TemporaryDirectory

        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, payload = self._externalize(_evaluate_job(), tmp.name)
        record = self.backend.enqueue(KIND_EVALUATE, payload, request_id="req_blobjob")
        claimed = self.backend.claim([KIND_EVALUATE], worker_id="w1")

        with patch.object(worker, "get_blob_store", return_value=store):
            ok = worker.execute_job(claimed[0], claimed[1], self.backend, llm=_UsageStub())

        self.assertTrue(ok)
        self.assertEqual(self.backend.get(record.job_id).status, STATUS_DONE)
        run = decode_result(self.backend.get_result(record.job_id))
        # The digest ran over the real records, not an empty set.
        self.assertTrue(run.result.claims)
        self.assertIsNotNone(run.result.digest)

    def test_missing_blob_fails_the_job_with_the_sharing_hint(self):
        from tempfile import TemporaryDirectory

        from app.blob_store import FilesystemBlobStore

        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        _store, payload = self._externalize(_evaluate_job(), tmp.name)
        empty = TemporaryDirectory()
        self.addCleanup(empty.cleanup)
        record = self.backend.enqueue(KIND_EVALUATE, payload)
        claimed = self.backend.claim([KIND_EVALUATE], worker_id="w1")

        with patch.object(
            worker, "get_blob_store", return_value=FilesystemBlobStore(empty.name)
        ):
            ok = worker.execute_job(claimed[0], claimed[1], self.backend, llm=_UsageStub())

        self.assertFalse(ok)
        failed = self.backend.get(record.job_id)
        self.assertEqual(failed.status, STATUS_ERROR)
        self.assertIn("shared", failed.error)

    def test_no_blob_store_configured_never_runs_with_zero_records(self):
        """Silently digesting nothing would produce a confident empty report."""
        from app.blob_store import NullBlobStore

        from tempfile import TemporaryDirectory

        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        _store, payload = self._externalize(_evaluate_job(), tmp.name)
        record = self.backend.enqueue(KIND_EVALUATE, payload)
        claimed = self.backend.claim([KIND_EVALUATE], worker_id="w1")

        with patch.object(worker, "get_blob_store", return_value=NullBlobStore()):
            ok = worker.execute_job(claimed[0], claimed[1], self.backend, llm=_UsageStub())

        self.assertFalse(ok)
        failed = self.backend.get(record.job_id)
        self.assertEqual(failed.error_class, "PayloadError")
        self.assertIn("no blob store is configured", failed.error)


class TestWorkerContinuesTheSubmitTrace(_WorkerCase):
    """Pattern C moves the run to another pod — the trace has to move with it.

    Simulates both halves: the web pod encoding inside its ``queue:submit`` span,
    then a worker claiming that payload and executing it. Without the trace
    context in the envelope the worker's run span would be a second, unrelated
    trace, which is the difference between "APM per process" and distributed
    tracing.
    """

    def _tracing(self):
        import os
        from unittest.mock import patch as _patch

        from app import tracing

        try:
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                InMemorySpanExporter,
            )
        except Exception:  # pragma: no cover - CI installs the SDK
            self.skipTest("opentelemetry-sdk is not installed")
        exporter = InMemorySpanExporter()
        patches = [
            _patch.object(config, "TRACING_ENABLED", True),
            _patch.dict(os.environ, {"OTEL_SDK_DISABLED": ""}),
            _patch.object(tracing, "_build_exporter", return_value=exporter),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        tracing.reset_tracing_for_tests()
        self.addCleanup(tracing.reset_tracing_for_tests)
        self.assertTrue(tracing.setup_tracing(role="test"))
        return tracing, exporter

    def test_worker_run_span_joins_the_submit_trace(self):
        from app import tracing

        tracing_mod, exporter = self._tracing()
        job = _evaluate_job(request_id="req_traced")

        # --- web pod: encode inside the submit span, then enqueue
        with tracing_mod.phase_span("queue:submit", kind=KIND_EVALUATE) as submit:
            payload = encode_job(KIND_EVALUATE, job)
        self.assertIn("traceparent", json.loads(payload)["trace_context"])
        self.backend.enqueue(KIND_EVALUATE, payload, request_id=job.request_id)
        claimed = self.backend.claim([KIND_EVALUATE], worker_id="w1")

        # --- worker pod: claim and execute
        ok = worker.execute_job(claimed[0], claimed[1], self.backend, llm=_UsageStub())
        self.assertTrue(ok)
        tracing_mod.flush_tracing()

        spans = {s.name: s for s in exporter.get_finished_spans()}
        run = spans["run:evaluate"]
        self.assertEqual(run.context.trace_id, submit.context.trace_id)
        self.assertEqual(run.parent.span_id, submit.context.span_id)
        # The phase spans hang under the run, on the worker's side of the trace.
        for phase in ("records:review", "claims", "rubric"):
            self.assertEqual(spans[phase].context.trace_id, submit.context.trace_id)

    def test_untraced_submit_still_runs_the_job(self):
        """A payload without a trace context must not change worker behaviour."""
        job = _evaluate_job(request_id="req_untraced")
        with patch.object(config, "TRACING_ENABLED", False):
            payload = encode_job(KIND_EVALUATE, job)
        self.assertNotIn("trace_context", payload)
        record = self.backend.enqueue(KIND_EVALUATE, payload, request_id=job.request_id)
        claimed = self.backend.claim([KIND_EVALUATE], worker_id="w1")

        ok = worker.execute_job(claimed[0], claimed[1], self.backend, llm=_UsageStub())

        self.assertTrue(ok)
        self.assertEqual(self.backend.get(record.job_id).status, STATUS_DONE)


class TestStaleSweep(_WorkerCase):
    def test_drain_stale_requeues_an_abandoned_job(self):
        job = _evaluate_job()
        record = self.backend.enqueue(KIND_EVALUATE, encode_job(KIND_EVALUATE, job))
        self.backend.claim([KIND_EVALUATE], worker_id="dead-worker")

        with patch.object(config, "JOB_QUEUE_LEASE_SECONDS", 0):
            count = worker.drain_stale(self.backend)

        self.assertEqual(count, 1)
        self.assertEqual(self.backend.get(record.job_id).status, "queued")
        # And the recovered job then completes normally.
        with _fast_queue():
            stats = worker.run_worker(backend=self.backend, llm=_UsageStub(), max_jobs=1)
        self.assertEqual(stats.completed, 1)


class TestWorkerCli(_WorkerCase):
    def test_drain_stale_flag_exits_without_claiming(self):
        record = self.backend.enqueue(KIND_EVALUATE, encode_job(KIND_EVALUATE, _evaluate_job()))
        self.backend.claim([KIND_EVALUATE], worker_id="dead")
        with patch.object(worker, "get_job_backend", return_value=self.backend), patch.object(
            config, "JOB_QUEUE_LEASE_SECONDS", 0
        ):
            code = worker.main(["--drain-stale"])
        self.assertEqual(code, 0)
        self.assertEqual(self.backend.get(record.job_id).status, "queued")

    def test_once_flag_processes_at_most_one_job(self):
        self.backend.enqueue(KIND_EVALUATE, encode_job(KIND_EVALUATE, _evaluate_job()))
        self.backend.enqueue(KIND_EVALUATE, encode_job(KIND_EVALUATE, _evaluate_job()))
        with _fast_queue(), patch.object(
            worker, "get_job_backend", return_value=self.backend
        ), patch.object(worker, "build_llm", return_value=_UsageStub()):
            code = worker.main(["--once"])
        self.assertEqual(code, 0)
        self.assertEqual(self.backend.depth(), 1)


class TestWorkerIdentity(unittest.TestCase):
    def test_env_override_wins(self):
        with patch.object(config, "JOB_QUEUE_WORKER_ID", "worker-a"):
            self.assertEqual(worker.default_worker_id(), "worker-a")

    def test_default_is_host_and_pid(self):
        with patch.object(config, "JOB_QUEUE_WORKER_ID", ""):
            identity = worker.default_worker_id()
        self.assertIn(":", identity)
        self.assertIn(str(os.getpid()), identity)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
