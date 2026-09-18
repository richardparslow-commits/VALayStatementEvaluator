"""Deadline/continuation regressions; all provider calls are offline stubs."""
from __future__ import annotations

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app import circuit_breaker, config, medical_review
from app.documents import Chunk, document_from_text
from app.llm import LLMClient, LLMUpstreamError
from app.medical_review import MedicalDigest, MedicalFact
from app.pipeline_guard import PipelineTimeoutError, run_with_timeout


class TestModelCancellation(unittest.TestCase):
    def setUp(self):
        circuit_breaker.reset_all_for_tests()
        self.addCleanup(circuit_breaker.reset_all_for_tests)
        settings = SimpleNamespace(
            configured=True,
            api_key="test-key", base_url="https://primary.invalid/v1",
            model_main="main", model_fast="fast",
            fallback_base_url="https://fallback.invalid/v1",
            fallback_api_key="fallback-key", fallback_model_main="main",
            fallback_model_fast="fast",
        )
        with patch("app.llm.OpenAI") as sdk:
            self.client = LLMClient(settings)
        self.assertEqual(sdk.call_count, 2)
        for call in sdk.call_args_list:
            self.assertEqual(call.kwargs["max_retries"], 0)
        self.client._client = MagicMock()
        self.client._fallback_client = MagicMock()

    def test_late_response_or_error_starts_no_more_calls(self):
        for outcome in ("response", "error"):
            with self.subTest(outcome=outcome):
                release = threading.Event()
                finished = threading.Event()
                started = threading.Event()

                def create(**kwargs):
                    started.set()
                    release.wait(2)
                    if outcome == "error":
                        raise LLMUpstreamError("temporary failure", retriable=True)
                    return MagicMock(choices=[MagicMock(message=MagicMock(content="late"))])

                self.client._client.chat.completions.create = MagicMock(side_effect=create)

                def pipeline():
                    try:
                        self.client.chat("system", "first")
                        self.client.chat("system", "must not run")
                    finally:
                        finished.set()

                try:
                    with patch.object(self.client, "_endpoint_candidates", return_value=["primary", "fallback"]):
                        t0 = time.monotonic()
                        with self.assertRaises(PipelineTimeoutError):
                            run_with_timeout(pipeline, timeout_seconds=0.1)
                        self.assertLess(time.monotonic() - t0, 0.5)
                        self.assertTrue(started.is_set())
                        self.assertFalse(finished.is_set())
                        release.set()
                        self.assertTrue(finished.wait(1))
                finally:
                    release.set()
                    self.assertTrue(finished.wait(2))
                create_mock = self.client._client.chat.completions.create
                create_mock.assert_called_once()
                self.assertGreater(create_mock.call_args.kwargs["timeout"], 0)
                self.assertLessEqual(create_mock.call_args.kwargs["timeout"], 0.1)
                self.client._fallback_client.chat.completions.create.assert_not_called()

    def test_retry_backoff_is_interrupted(self):
        finished = threading.Event()
        create = self.client._client.chat.completions.create
        create.side_effect = LLMUpstreamError("temporary failure", retriable=True)

        def pipeline():
            try:
                self.client.chat("system", "user")
            finally:
                finished.set()

        with patch("app.llm._retry_backoff_seconds", return_value=2):
            try:
                with self.assertRaises(PipelineTimeoutError):
                    run_with_timeout(pipeline, timeout_seconds=0.1)
                self.assertTrue(finished.wait(0.5), "backoff must not retain the worker")
                create.assert_called_once()
            finally:
                self.assertTrue(finished.wait(3))

    def test_concurrency_queue_wait_respects_deadline(self):
        limiter = circuit_breaker.ConcurrencyLimiter(max_concurrent=1, queue_timeout=30)
        limiter.acquire()
        finished = threading.Event()

        def pipeline():
            try:
                self.client.chat("system", "user")
            finally:
                finished.set()

        with patch("app.llm.get_llm_limiter", return_value=limiter):
            try:
                with self.assertRaises(PipelineTimeoutError):
                    run_with_timeout(pipeline, timeout_seconds=0.1)
                self.assertTrue(finished.wait(0.5))
                self.client._client.chat.completions.create.assert_not_called()
            finally:
                limiter.release()
                self.assertTrue(finished.wait(2))


class TestRecordCancellation(unittest.TestCase):
    def test_digest_and_merge_cancel_queued_work_and_unwind_parent(self):
        for phase in ("digest", "merge"):
            with self.subTest(phase=phase):
                release = threading.Event()
                started = threading.Event()
                finished = threading.Event()
                pools = []
                progress = MagicMock()

                def pool_factory(**kwargs):
                    pool = ThreadPoolExecutor(**kwargs)
                    pools.append(pool)
                    return pool

                def blocked(*args, **kwargs):
                    started.set()
                    release.wait(2)
                    return {"facts": []} if phase == "digest" else []

                llm = MagicMock()
                llm._settings.model_fast = "fast"
                llm.chat_json.side_effect = blocked
                chunks = [Chunk(i, 5, f"record {i}") for i in range(1, 6)]
                facts = [MedicalFact("unknown", "symptom", f"fact {i}", "records.txt") for i in range(600)]

                def pipeline():
                    try:
                        if phase == "digest":
                            return medical_review.review_medical_records(
                                llm, [document_from_text("records.txt", "medical evidence")], progress
                            )
                        return medical_review._merge_facts(llm, MedicalDigest(facts=facts), progress)
                    finally:
                        finished.set()

                with (
                    patch.object(config, "RECORDS_CONCURRENCY", 1),
                    patch.object(medical_review, "ThreadPoolExecutor", side_effect=pool_factory),
                    patch.object(medical_review, "chunk_page_labelled_text", return_value=chunks),
                    patch.object(medical_review, "_merge_once", side_effect=blocked) as merge,
                    patch.object(medical_review, "_summarize") as summary,
                ):
                    try:
                        with self.assertRaises(PipelineTimeoutError):
                            run_with_timeout(pipeline, timeout_seconds=0.1)
                        self.assertTrue(started.is_set())
                        self.assertTrue(finished.wait(0.5), "parent must not wait for a blocked child")
                        progress_count = progress.call_count
                    finally:
                        release.set()
                        for pool in pools:
                            pool.shutdown(wait=True, cancel_futures=True)
                        self.assertTrue(finished.wait(2))
                    self.assertEqual(progress.call_count, progress_count)
                    summary.assert_not_called()
                    if phase == "digest":
                        llm.chat_json.assert_called_once()
                        merge.assert_not_called()
                    else:
                        merge.assert_called_once()


class TestProgressCancellation(unittest.TestCase):
    def test_ui_progress_rejects_late_updates(self):
        from app.views.shared import progress_widgets

        release = threading.Event()
        finished = threading.Event()
        with patch("app.views.shared.st") as st:
            bar, progress = progress_widgets()

            def pipeline():
                try:
                    release.wait(2)
                    progress(1.0, "late success")
                finally:
                    finished.set()

            try:
                with self.assertRaises(PipelineTimeoutError):
                    run_with_timeout(pipeline, timeout_seconds=0.05)
            finally:
                release.set()
                self.assertTrue(finished.wait(2))
            bar.progress.assert_not_called()


if __name__ == "__main__":
    unittest.main()
