"""Job-queue tests against a REAL Redis server.

Everything else in the suite fakes the transport, which verifies the queue logic
but not that redis-py's actual commands, argument forms, and reply shapes match
what ``RedisJobBackend`` expects — a rename or a changed kwarg would pass the
fakes and fail in production. This module closes that gap.

Skipped unless ``VA_LSE_TEST_REDIS_URL`` is set, so local runs without Redis stay
green; CI sets it against a Redis service container (see
``.github/workflows/test.yml`` → queue-integration). Run it locally with:

    docker run --rm -p 6379:6379 redis:7-alpine
    VA_LSE_TEST_REDIS_URL=redis://localhost:6379/0 python -m unittest tests.test_job_queue_redis_live

Each test run uses a unique key prefix and deletes only its own keys, so pointing
this at a shared Redis cannot disturb other data.
"""
import json
import os
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from log_isolation import isolate_app_logs  # noqa: E402

from app import config  # noqa: E402
from app import shutdown  # noqa: E402
from app import job_queue, worker  # noqa: E402
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
    STATUS_QUEUED,
    RedisJobBackend,
)
from app.usage import UsageTracker  # noqa: E402

REDIS_URL = os.getenv("VA_LSE_TEST_REDIS_URL", "").strip()

# Claims must not block the suite on a real socket either.
FAST_CLAIM = 0.2


class _UsageStub:
    """Minimal pipeline stub that records usage."""

    def __init__(self) -> None:
        from test_evaluate import _FakeLLM

        self._settings = MagicMock(model_fast="fake-fast", model_main="fake-main")
        self.usage = UsageTracker()
        self._base = _FakeLLM()

    def _record(self, system: str, user: str, kwargs: dict, content: str) -> None:
        self.usage.record(
            model="fake-fast",
            phase=kwargs.get("phase", "general"),
            system=system,
            user=user,
            content=content,
            prompt_tokens=None,
            completion_tokens=None,
        )

    def chat_json(self, system: str, user: str, **kwargs: object) -> object:
        out = self._base.chat_json(system, user, **kwargs)
        self._record(system, user, kwargs, json.dumps(out, default=str))
        return out

    def chat(self, system: str, user: str, **kwargs: object) -> object:
        out = self._base.chat(system, user, **kwargs)
        self._record(system, user, kwargs, str(out))
        return out


@unittest.skipUnless(REDIS_URL, "set VA_LSE_TEST_REDIS_URL to run against a real Redis")
class TestRedisBackendLive(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = isolate_app_logs(self)
        shutdown.reset_for_tests()
        self.addCleanup(shutdown.reset_for_tests)
        self.prefix = f"va_lse_test_{uuid.uuid4().hex[:10]}"
        self.backend = RedisJobBackend(
            REDIS_URL, prefix=self.prefix, ttl_seconds=120, timeout_seconds=5.0
        )
        self.addCleanup(self._cleanup)

    def _cleanup(self) -> None:
        client = self.backend._client  # test-only access to the raw client
        for key in client.scan_iter(match=f"{self.prefix}*"):
            client.delete(key)

    def _client(self) -> RedisJobBackend:
        return RedisJobBackend(
            REDIS_URL, prefix=self.prefix, ttl_seconds=120, timeout_seconds=5.0
        )

    def test_server_is_reachable(self):
        self.assertTrue(self.backend.ping())

    def test_enqueue_claim_complete_roundtrip(self):
        record = self.backend.enqueue(KIND_EVALUATE, '{"hello": 1}', request_id="req_live")
        self.assertEqual(self.backend.depth(), 1)

        with patch.object(config, "JOB_QUEUE_CLAIM_TIMEOUT_SECONDS", FAST_CLAIM):
            claimed = self.backend.claim([KIND_EVALUATE], worker_id="live-w1")
        self.assertIsNotNone(claimed)
        claimed_record, payload = claimed
        self.assertEqual(claimed_record.job_id, record.job_id)
        self.assertEqual(payload, '{"hello": 1}')
        self.assertEqual(claimed_record.status, job_queue.STATUS_RUNNING)
        self.assertEqual(claimed_record.worker_id, "live-w1")

        self.backend.set_progress(record.job_id, 0.42, "digesting", claim_token=claimed_record.claim_token)
        running = self.backend.get(record.job_id)
        self.assertAlmostEqual(running.progress, 0.42)
        self.assertEqual(running.message, "digesting")

        self.backend.store_result(record.job_id, '{"ok": true}', claim_token=claimed_record.claim_token)
        self.backend.complete(record.job_id, claim_token=claimed_record.claim_token, message="done")
        finished = self.backend.get(record.job_id)
        self.assertEqual(finished.status, STATUS_DONE)
        self.assertEqual(self.backend.get_result(record.job_id), '{"ok": true}')

    def test_records_carry_a_ttl(self):
        record = self.backend.enqueue(KIND_EVALUATE, "{}")
        ttl = self.backend._client.ttl(f"{self.prefix}:job:{record.job_id}:meta")
        self.assertGreater(ttl, 0)
        self.assertLessEqual(ttl, 120)

    def test_requeue_restores_a_running_job(self):
        record = self.backend.enqueue(KIND_EVALUATE, "{}")
        with patch.object(config, "JOB_QUEUE_CLAIM_TIMEOUT_SECONDS", FAST_CLAIM):
            claimed, _ = self.backend.claim([KIND_EVALUATE], worker_id="w1")
        self.assertTrue(self.backend.requeue(record.job_id, claim_token=claimed.claim_token, reason="draining"))
        self.assertEqual(self.backend.get(record.job_id).status, STATUS_QUEUED)
        self.assertEqual(self.backend.depth(), 1)
        # Only a running job can be handed back.
        self.assertFalse(self.backend.requeue(record.job_id, claim_token=claimed.claim_token, reason="again"))

    def test_second_client_cannot_double_claim(self):
        record = self.backend.enqueue(KIND_EVALUATE, "{}")
        other = self._client()
        with patch.object(config, "JOB_QUEUE_CLAIM_TIMEOUT_SECONDS", FAST_CLAIM), patch.object(
            config, "JOB_QUEUE_POLL_SECONDS", 0.01
        ):
            self.assertIsNotNone(other.claim([KIND_EVALUATE], worker_id="w1"))
            other._client.lpush(f"{self.prefix}:jobs:evaluate", record.job_id)
            self.assertIsNone(other.claim([KIND_EVALUATE], worker_id="w2"))

    def test_requeue_stale_recovers_an_abandoned_job(self):
        record = self.backend.enqueue(KIND_EVALUATE, "{}")
        other = self._client()
        with patch.object(config, "JOB_QUEUE_CLAIM_TIMEOUT_SECONDS", FAST_CLAIM):
            self.assertIsNotNone(other.claim([KIND_EVALUATE], worker_id="doomed"))

        with patch.object(config, "JOB_QUEUE_LEASE_SECONDS", 0):
            self.assertEqual(self.backend.requeue_stale(), 1)
        recovered = self.backend.get(record.job_id)
        self.assertEqual(recovered.status, STATUS_QUEUED)
        self.assertEqual(self.backend.depth(), 1)
        # The attempt is KEPT here, unlike the draining requeue that gives it back:
        # a worker that dies mid-job may have already spent LLM calls, and this is
        # what stops a poison job (one that kills every worker) from looping
        # forever — it gets parked as JobAbandoned after MAX_ATTEMPTS.
        self.assertEqual(recovered.attempts, 1)
        # The lease entry is gone, so a second sweep is a no-op.
        with patch.object(config, "JOB_QUEUE_LEASE_SECONDS", 0):
            self.assertEqual(self.backend.requeue_stale(), 0)

    def test_abandoned_job_is_parked_after_max_attempts(self):
        record = self.backend.enqueue(KIND_EVALUATE, "{}")
        with patch.object(config, "JOB_QUEUE_CLAIM_TIMEOUT_SECONDS", FAST_CLAIM):
            for _ in range(job_queue.MAX_ATTEMPTS):
                claimed = self.backend.claim([KIND_EVALUATE], worker_id="flaky")
                self.assertIsNotNone(claimed)
                with patch.object(config, "JOB_QUEUE_LEASE_SECONDS", 0):
                    self.backend.requeue_stale()
        parked = self.backend.get(record.job_id)
        self.assertEqual(parked.status, STATUS_ERROR)
        self.assertEqual(parked.error_class, "JobAbandoned")

    def test_health_reports_redis(self):
        health = self.backend.health()
        self.assertEqual(health["backend"], "redis")
        self.assertTrue(health["is_distributed"])

    def test_worker_executes_a_job_end_to_end(self):
        """The full Pattern C hop against a real server: submit → worker → result."""
        producer = self._client()
        job = EvaluateJob(
            statement_text="I injured my knee lifting a pallet.",
            records=[document_from_text("a.txt", "EVT knee pain noted during service.")],
            request_id="req_redis_e2e",
            record_sources=["Upload"],
        )
        record = producer.enqueue(
            KIND_EVALUATE, encode_job(KIND_EVALUATE, job), request_id=job.request_id
        )

        consumer = self._client()
        with patch.object(config, "JOB_QUEUE_CLAIM_TIMEOUT_SECONDS", 2), patch.object(
            config, "JOB_QUEUE_POLL_SECONDS", 0.05
        ):
            stats = worker.run_worker(
                once=True, backend=consumer, worker_id="live-worker", llm=_UsageStub()
            )

        self.assertEqual(stats.claimed, 1)
        self.assertEqual(stats.completed, 1)
        self.assertEqual(producer.get(record.job_id).status, STATUS_DONE)
        run = decode_result(producer.get_result(record.job_id))
        self.assertEqual(run.request_id, "req_redis_e2e")
        self.assertTrue(run.result.report_markdown)

    def test_draft_job_roundtrips(self):
        job = DraftJob(
            records=[document_from_text("a.txt", "EVT knee pain noted.")],
            witness={"name": "Jane Doe"},
            observations="Daily knee pain.",
            condition="knee strain",
            claim_type="Service connection (new claim)",
            request_id="req_redis_draft",
        )
        record = self.backend.enqueue(KIND_DRAFT, encode_job(KIND_DRAFT, job))
        with patch.object(config, "JOB_QUEUE_CLAIM_TIMEOUT_SECONDS", FAST_CLAIM):
            claimed = self.backend.claim([KIND_DRAFT], worker_id="w1")
        self.assertTrue(worker.execute_job(claimed[0], claimed[1], self.backend, llm=_UsageStub()))
        self.assertEqual(self.backend.get(record.job_id).status, STATUS_DONE)

    def test_get_and_result_tolerate_a_missing_job(self):
        self.assertIsNone(self.backend.get("job_does_not_exist"))
        self.assertIsNone(self.backend.get_result("job_does_not_exist"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
