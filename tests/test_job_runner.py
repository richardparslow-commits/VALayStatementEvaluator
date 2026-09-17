"""Tests for queue-mode run orchestration (app/views/job_runner.py).

Runs in Streamlit "bare mode" (Streamlit is imported but no script run exists),
which is how the other view-layer tests exercise this code: element calls are
no-ops, but the control flow — submit, poll, decode, hydrate, give up — is real.

A preset backend lets a job appear already finished or already running, so the
poll loop's terminal, timeout, and expired-result branches are all covered
without a worker process.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from app import config  # noqa: E402
from app.documents import document_from_text  # noqa: E402
from app.evaluate import DIMENSION_LABELS, EvaluationResult  # noqa: E402
from app.job_payload import (  # noqa: E402
    KIND_EVALUATE,
    EvaluateJob,
    RunResult,
    encode_job,
    encode_result,
)
from app.job_queue import (  # noqa: E402
    STATUS_DONE,
    InProcessJobBackend,
)
from app.usage import UsageTracker  # noqa: E402
from app.views import job_runner  # noqa: E402


def _docs():
    return [document_from_text("a.txt", "EVT knee pain noted.")]


def _job(request_id: str = "req_q1") -> EvaluateJob:
    return EvaluateJob(
        statement_text="I injured my knee.",
        records=_docs(),
        request_id=request_id,
        record_sources=["Upload"],
    )


def _result_json(request_id: str = "req_q1") -> str:
    tracker = UsageTracker()
    tracker.record(
        model="fake-fast",
        phase="records:digest",
        system="s",
        user="u",
        content="c",
        prompt_tokens=10,
        completion_tokens=5,
    )
    result = EvaluationResult(
        claimed_condition="knee",
        claims=[{"id": 1, "text": "c"}],
        scores={k: 7.0 for k in DIMENSION_LABELS},
        report_markdown="# Report",
        digest=None,
    )
    return encode_result(
        RunResult(kind=KIND_EVALUATE, result=result, usage=tracker, request_id=request_id)
    )


class _PresetBackend(InProcessJobBackend):
    """Distributed-looking backend whose jobs are already in a chosen state."""

    is_distributed = True
    name = "preset"

    def __init__(self, *, finish: bool = True, result: str | None = None) -> None:
        super().__init__(prefix="test", ttl_seconds=60)
        self._finish = finish
        self._result = result if result is not None else _result_json()
        self.last_payload: str | None = None

    def enqueue(self, kind: str, payload: str, *, request_id: str = ""):  # noqa: ANN201
        record = super().enqueue(kind, payload, request_id=request_id)
        self.last_payload = payload
        if self._finish:
            self.store_result(record.job_id, self._result)
            self.complete(record.job_id)
        else:
            # Simulate a worker holding the job without finishing.
            claimed = self.claim([kind], worker_id="w1")
            assert claimed is not None
            self.set_progress(record.job_id, 0.4, "digesting records")
        return record


class TestQueueModeDetection(unittest.TestCase):
    def test_disabled_by_default(self):
        with patch.object(config, "JOB_QUEUE_ENABLED", False):
            self.assertFalse(job_runner.queue_mode_active())

    def test_enabled_with_distributed_backend_is_active(self):
        with patch.object(config, "JOB_QUEUE_ENABLED", True), patch.object(
            job_runner, "get_job_backend", return_value=_PresetBackend()
        ):
            self.assertTrue(job_runner.queue_mode_active())

    def test_enabled_with_inprocess_backend_is_not_active(self):
        """A single-process queue must not pretend to distribute work."""
        with patch.object(config, "JOB_QUEUE_ENABLED", True), patch.object(
            job_runner, "get_job_backend", return_value=InProcessJobBackend(prefix="t", ttl_seconds=60)
        ):
            self.assertFalse(job_runner.queue_mode_active())

    def test_backend_failure_falls_back_to_inprocess(self):
        with patch.object(config, "JOB_QUEUE_ENABLED", True), patch.object(
            job_runner, "get_job_backend", side_effect=RuntimeError("no backend")
        ):
            self.assertFalse(job_runner.queue_mode_active())

    def test_status_line_describes_the_backend(self):
        backend = _PresetBackend()
        backend.enqueue(KIND_EVALUATE, "{}")
        with patch.object(job_runner, "get_job_backend", return_value=backend):
            line = job_runner.queue_status_line()
        self.assertIn("preset", line)


class TestWorkerConfigError(unittest.TestCase):
    def test_reports_a_missing_env_key(self):
        with patch.object(config, "load_settings", return_value=MagicMock(configured=False)):
            message = job_runner.worker_config_error()
        self.assertIn("cannot read a key typed into the sidebar", message)

    def test_empty_when_configured(self):
        with patch.object(config, "load_settings", return_value=MagicMock(configured=True)):
            self.assertEqual(job_runner.worker_config_error(), "")

    def test_unreadable_config_is_reported_not_raised(self):
        with patch.object(config, "load_settings", side_effect=RuntimeError("boom")):
            self.assertIn("RuntimeError", job_runner.worker_config_error())


class TestSubmitAndPoll(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = dict(st.session_state)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        for key in list(st.session_state.keys()):
            if key not in self._saved:
                del st.session_state[key]
        for key, value in self._saved.items():
            st.session_state[key] = value

    def _submit(self, backend, **overrides):  # noqa: ANN201
        with patch.object(job_runner, "get_job_backend", return_value=backend), patch.object(
            config, "JOB_QUEUE_UI_POLL_SECONDS", 0.01
        ):
            return job_runner.submit_job(
                slot="eval",
                job=_job(),
                request_id="req_q1",
                condition="knee strain",
                sources=["Upload"],
                files=1,
                pages=1,
                action_label="Evaluation",
                **overrides,
            )

    def test_completed_job_is_decoded_and_hydrated(self):
        backend = _PresetBackend()
        outcome = self._submit(backend)

        self.assertIsNotNone(outcome)
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.request_id, "req_q1")
        self.assertIsNotNone(st.session_state.get("eval_result"))
        self.assertEqual(st.session_state.get("eval_request_id"), "req_q1")
        # The pending marker must be cleared so the tab does not re-attach.
        self.assertNotIn("va_lse_pending_job_eval", st.session_state)

    def test_queued_payload_carries_the_run_inputs(self):
        backend = _PresetBackend()
        self._submit(backend)
        self.assertIn("I injured my knee.", backend.last_payload)
        self.assertIn("EVT knee pain noted.", backend.last_payload)

    def test_failed_job_reports_the_worker_error(self):
        backend = _PresetBackend(finish=False)
        record = backend.enqueue(KIND_EVALUATE, encode_job(KIND_EVALUATE, _job()))
        backend.fail(record.job_id, error="LLMError: endpoint down", error_class="LLMError")

        with patch.object(job_runner, "get_job_backend", return_value=backend), patch.object(
            config, "JOB_QUEUE_UI_POLL_SECONDS", 0.01
        ), patch.object(backend, "enqueue", return_value=backend.get(record.job_id)):
            outcome = job_runner.submit_job(
                slot="eval", job=_job(), request_id="req_q1", condition=None,
                sources=["Upload"], files=1, pages=1, action_label="Evaluation",
            )

        self.assertFalse(outcome.ok)
        self.assertIn("endpoint down", outcome.error)
        self.assertEqual(outcome.error_class, "LLMError")
        self.assertNotIn("va_lse_pending_job_eval", st.session_state)

    def test_expired_result_is_explained_not_crashed(self):
        backend = _PresetBackend()
        original = backend.get_result

        def gone(job_id: str):  # noqa: ANN201
            return None

        with patch.object(backend, "get_result", gone):
            outcome = self._submit(backend)
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.error_class, "ResultExpired")
        self.assertIn("expire", outcome.error)
        original("unused")

    def test_corrupt_result_is_a_clear_failure(self):
        backend = _PresetBackend(result="{not the right shape")
        outcome = self._submit(backend)
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.error_class, "PayloadError")

    def test_still_running_job_leaves_a_resume_marker(self):
        backend = _PresetBackend(finish=False)
        with patch.object(config, "PIPELINE_TIMEOUT_SECONDS", 1), patch.object(
            job_runner, "UI_WAIT_SLACK_SECONDS", 0.0
        ):
            outcome = self._submit(backend)

        self.assertFalse(outcome.ok)
        self.assertTrue(outcome.still_running)
        # The marker stays so a later render can re-attach to the running job.
        self.assertIn("va_lse_pending_job_eval", st.session_state)

    def test_a_lost_queue_is_reported_without_losing_the_job(self):
        backend = _PresetBackend(finish=False)

        def broken(_job_id: str):  # noqa: ANN201
            raise RuntimeError("connection reset")

        with patch.object(backend, "get", broken):
            outcome = self._submit(backend)
        self.assertFalse(outcome.ok)
        self.assertTrue(outcome.still_running)
        self.assertEqual(outcome.error_class, "JobQueueError")

    def test_removed_job_is_reported_as_expired(self):
        backend = _PresetBackend(finish=False)
        with patch.object(backend, "get", return_value=None):
            outcome = self._submit(backend)
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.error_class, "JobExpired")

    def test_oversize_payload_is_refused_before_enqueue(self):
        backend = _PresetBackend()
        with patch.object(config, "JOB_QUEUE_MAX_PAYLOAD_BYTES", 64):
            outcome = self._submit(backend)
        self.assertIsNone(outcome)
        self.assertEqual(backend.depth(), 0)

    def test_second_submission_is_refused_while_one_is_running(self):
        """Guard against queuing a duplicate digest after giving up on a slow run."""
        backend = _PresetBackend(finish=False)
        first = backend.enqueue(KIND_EVALUATE, encode_job(KIND_EVALUATE, _job()))
        st.session_state["va_lse_pending_job_eval"] = first.job_id

        outcome = self._submit(backend)

        self.assertIsNone(outcome)
        # Nothing new was queued on top of the in-flight job.
        self.assertEqual(backend.depth(), 0)
        self.assertEqual(st.session_state.get("va_lse_pending_job_eval"), first.job_id)

    def test_finished_pending_job_does_not_block_a_new_run(self):
        backend = _PresetBackend()
        outcome = self._submit(backend)
        self.assertTrue(outcome.ok)
        # A stale marker for a *finished* job must not lock the tab.
        record = backend.enqueue(KIND_EVALUATE, encode_job(KIND_EVALUATE, _job()))
        backend.complete(record.job_id)
        st.session_state["va_lse_pending_job_eval"] = record.job_id
        self.assertTrue(self._submit(backend).ok)

    def test_large_job_is_externalized_to_the_blob_store(self):
        """A big record bundle must reach the worker as a small reference."""
        from tempfile import TemporaryDirectory

        from app.blob_store import FilesystemBlobStore

        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = FilesystemBlobStore(tmp.name)
        backend = _PresetBackend()
        big = EvaluateJob(
            statement_text="s",
            records=[document_from_text("a.txt", "EVT " + ("pain noted. " * 5000))],
            request_id="req_q1",
        )
        with patch.object(job_runner, "get_job_backend", return_value=backend), patch.object(
            job_runner, "get_blob_store", return_value=store
        ), patch.object(config, "JOB_QUEUE_INLINE_MAX_BYTES", 1024), patch.object(
            config, "JOB_QUEUE_UI_POLL_SECONDS", 0.01
        ):
            outcome = job_runner.submit_job(
                slot="eval", job=big, request_id="req_q1", condition=None,
                sources=["Upload"], files=1, pages=1, action_label="Evaluation",
            )

        self.assertTrue(outcome.ok)
        self.assertIn("documents_ref", backend.last_payload)
        self.assertNotIn("pain noted", backend.last_payload)
        # One blob written, and the queue payload stayed small.
        self.assertEqual(len(list((Path(tmp.name) / "blobs").glob("*/*.json"))), 1)
        self.assertLess(len(backend.last_payload), 4096)

    def test_externalization_without_a_blob_store_fails_clearly(self):
        from app.blob_store import NullBlobStore

        backend = _PresetBackend()
        big = EvaluateJob(
            statement_text="s",
            records=[document_from_text("a.txt", "EVT " + ("pain noted. " * 5000))],
        )
        with patch.object(job_runner, "get_job_backend", return_value=backend), patch.object(
            job_runner, "get_blob_store", return_value=NullBlobStore()
        ), patch.object(config, "JOB_QUEUE_INLINE_MAX_BYTES", 1024):
            outcome = job_runner.submit_job(
                slot="eval", job=big, request_id="req_q1", condition=None,
                sources=["Upload"], files=1, pages=1, action_label="Evaluation",
            )

        self.assertIsNone(outcome)
        self.assertEqual(backend.depth(), 0)
        self.assertNotIn("va_lse_pending_job_eval", st.session_state)

    def test_enqueue_failure_is_reported_and_logged(self):
        backend = _PresetBackend()

        def boom(*_args, **_kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("redis down")

        with patch.object(backend, "enqueue", boom):
            outcome = self._submit(backend)
        self.assertIsNone(outcome)
        self.assertNotIn("va_lse_pending_job_eval", st.session_state)


class TestResume(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = dict(st.session_state)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        for key in list(st.session_state.keys()):
            if key not in self._saved:
                del st.session_state[key]
        for key, value in self._saved.items():
            st.session_state[key] = value

    def test_noop_when_queue_mode_is_off(self):
        st.session_state["va_lse_pending_job_eval"] = "job_x"
        with patch.object(config, "JOB_QUEUE_ENABLED", False):
            job_runner.resume_pending_job("eval", action_label="Evaluation")
        # Untouched: turning queue mode off must not silently drop a marker.
        self.assertIn("va_lse_pending_job_eval", st.session_state)

    def test_noop_without_a_pending_job(self):
        st.session_state.pop("va_lse_pending_job_eval", None)
        with patch.object(config, "JOB_QUEUE_ENABLED", True), patch.object(
            job_runner, "get_job_backend", return_value=_PresetBackend()
        ):
            job_runner.resume_pending_job("eval", action_label="Evaluation")

    def test_finished_job_is_hydrated_on_resume(self):
        backend = _PresetBackend()
        record = backend.enqueue(KIND_EVALUATE, encode_job(KIND_EVALUATE, _job()))
        st.session_state["va_lse_pending_job_eval"] = record.job_id

        with patch.object(config, "JOB_QUEUE_ENABLED", True), patch.object(
            job_runner, "get_job_backend", return_value=backend
        ):
            job_runner.resume_pending_job("eval", action_label="Evaluation")

        self.assertIsNotNone(st.session_state.get("eval_result"))
        self.assertNotIn("va_lse_pending_job_eval", st.session_state)

    def test_vanished_job_clears_the_marker(self):
        backend = _PresetBackend()
        st.session_state["va_lse_pending_job_eval"] = "job_gone"

        with patch.object(config, "JOB_QUEUE_ENABLED", True), patch.object(
            job_runner, "get_job_backend", return_value=backend
        ):
            job_runner.resume_pending_job("eval", action_label="Evaluation")

        self.assertNotIn("va_lse_pending_job_eval", st.session_state)

    def test_running_job_keeps_the_marker_and_offers_to_wait(self):
        backend = _PresetBackend(finish=False)
        record = backend.enqueue(KIND_EVALUATE, encode_job(KIND_EVALUATE, _job()))
        st.session_state["va_lse_pending_job_eval"] = record.job_id

        with patch.object(config, "JOB_QUEUE_ENABLED", True), patch.object(
            job_runner, "get_job_backend", return_value=backend
        ):
            job_runner.resume_pending_job("eval", action_label="Evaluation")

        self.assertEqual(st.session_state.get("va_lse_pending_job_eval"), record.job_id)

    def test_status_lookup_failure_never_breaks_the_tab(self):
        st.session_state["va_lse_pending_job_eval"] = "job_x"
        with patch.object(config, "JOB_QUEUE_ENABLED", True), patch.object(
            job_runner, "get_job_backend", side_effect=RuntimeError("backend gone")
        ):
            job_runner.resume_pending_job("eval", action_label="Evaluation")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
