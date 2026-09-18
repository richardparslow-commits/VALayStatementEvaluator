"""Offline tests for the distributed job queue (app/job_queue.py).

The Redis and Upstash backends are exercised against shared in-memory fakes of
their transports, so the key layout, claim/lease bookkeeping, stale re-queue, and
command shaping are all verified without any external service. The Upstash fake
also pins the JSON-command API shape (a JSON array POSTed to the database root),
because the sibling reference-cache client in app/shared_cache.py speaks the
older path-based API and it would be easy to drift into the wrong one.
"""
import base64
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app import config  # noqa: E402
from app import job_queue  # noqa: E402
from app.job_queue import (  # noqa: E402
    JobBackend,  # noqa: E402
    KIND_DRAFT,
    KIND_EVALUATE,
    InProcessJobBackend,
    JobRecord,
    RedisJobBackend,
    UpstashJobBackend,
    build_job_backend,
)

# Claims must not block the suite; the real default is 5s.
FAST_CLAIM = 0.05


# ------------------------------------------------------------------ test doubles
class _FakeRedisClient:
    """Minimal in-memory stand-in for a redis-py client."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}
        self.zsets: dict[str, dict[str, float]] = {}

    def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self.store[key] = value
        return True

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def lpush(self, key: str, value: str) -> int:
        self.lists.setdefault(key, []).insert(0, value)
        return len(self.lists[key])

    def rpop(self, key: str) -> str | None:
        items = self.lists.get(key) or []
        return items.pop() if items else None

    def llen(self, key: str) -> int:
        return len(self.lists.get(key) or [])

    def zadd(self, key: str, mapping: dict[str, float]) -> int:
        target = self.zsets.setdefault(key, {})
        target.update(mapping)
        return len(mapping)

    def zrem(self, key: str, member: str) -> int:
        return 1 if self.zsets.get(key, {}).pop(member, None) is not None else 0

    def zrangebyscore(self, key: str, _min: str, max_score: str) -> list[str]:
        exclusive = str(max_score).startswith("(")
        bound = float(str(max_score).lstrip("("))
        out = []
        for member, score in sorted(self.zsets.get(key, {}).items(), key=lambda kv: kv[1]):
            if score < bound if exclusive else score <= bound:
                out.append(member)
        return out

    def ping(self) -> bool:
        return True


class _FakeRedisModule:
    """Stand-in for the ``redis`` package (``Redis.from_url``)."""

    def __init__(self, client: _FakeRedisClient) -> None:
        self._client = client
        self.Redis = self  # so ``redis.Redis.from_url`` resolves

    def from_url(self, *_args, **_kwargs) -> _FakeRedisClient:
        return self._client


class _FakeUpstashResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeUpstashResponse":
        return self

    def __exit__(self, *_exc) -> bool:
        return False


class _FakeUpstashTransport:
    """Records commands and replies like the Upstash REST API."""

    def __init__(self) -> None:
        self.commands: list[list[object]] = []
        self.headers: list[dict] = []
        self.store: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}
        self.zsets: dict[str, dict[str, float]] = {}

    def urlopen(self, request, timeout=None):  # noqa: ANN001, ARG002
        body = json.loads(request.data.decode("utf-8"))
        assert isinstance(body, list)
        self.commands.append(body)
        self.headers.append(dict(request.headers))
        return _FakeUpstashResponse(json.dumps({"result": self._run(body)}).encode("utf-8"))

    def _run(self, body: list) -> object:
        op = str(body[0]).upper()
        if op == "SET":
            self.store[str(body[1])] = str(body[2])
            return "OK"
        if op == "GET":
            return self.store.get(str(body[1]))
        if op == "LPUSH":
            self.lists.setdefault(str(body[1]), []).insert(0, str(body[2]))
            return len(self.lists[str(body[1])])
        if op == "RPOP":
            items = self.lists.get(str(body[1])) or []
            return items.pop() if items else None
        if op == "LLEN":
            return len(self.lists.get(str(body[1])) or [])
        if op == "ZADD":
            self.zsets.setdefault(str(body[1]), {})[str(body[3])] = float(body[2])
            return 1
        if op == "ZREM":
            return 1 if self.zsets.get(str(body[1]), {}).pop(str(body[2]), None) else 0
        if op == "ZRANGEBYSCORE":
            exclusive = str(body[3]).startswith("(")
            bound = float(str(body[3]).lstrip("("))
            return [
                member
                for member, score in sorted(
                    self.zsets.get(str(body[1]), {}).items(), key=lambda kv: kv[1]
                )
                if (score < bound if exclusive else score <= bound)
            ]
        if op == "PING":
            return "PONG"
        raise AssertionError(f"unexpected command {op}")


# --------------------------------------------------------------------- basetest
class _BackendCase:
    """Assertions every backend must satisfy identically."""

    def make_backend(self):  # noqa: ANN201
        raise NotImplementedError

    def test_enqueue_then_claim_roundtrip(self):
        backend = self.make_backend()
        record = backend.enqueue(KIND_EVALUATE, '{"hello": 1}', request_id="req_1")
        self.assertEqual(record.status, job_queue.STATUS_QUEUED)
        self.assertEqual(backend.depth(), 1)

        claimed = backend.claim([KIND_EVALUATE], worker_id="w1")
        self.assertIsNotNone(claimed)
        claimed_record, payload = claimed
        self.assertEqual(claimed_record.job_id, record.job_id)
        self.assertEqual(payload, '{"hello": 1}')
        self.assertEqual(claimed_record.status, job_queue.STATUS_RUNNING)
        self.assertEqual(claimed_record.worker_id, "w1")
        self.assertEqual(claimed_record.attempts, 1)
        self.assertEqual(backend.depth(), 0)

    def test_progress_complete_and_result(self):
        backend = self.make_backend()
        record = backend.enqueue(KIND_EVALUATE, '{"k": 1}')
        backend.claim([KIND_EVALUATE], worker_id="w1")
        backend.set_progress(record.job_id, 0.5, "half way")
        running = backend.get(record.job_id)
        self.assertAlmostEqual(running.progress, 0.5)
        self.assertEqual(running.message, "half way")

        backend.store_result(record.job_id, '{"result": true}')
        backend.complete(record.job_id, message="completed in 3s")
        done = backend.get(record.job_id)
        self.assertEqual(done.status, job_queue.STATUS_DONE)
        self.assertTrue(done.is_terminal)
        self.assertEqual(done.progress, 1.0)
        self.assertEqual(done.message, "completed in 3s")
        self.assertEqual(backend.get_result(record.job_id), '{"result": true}')

    def test_fail_records_error(self):
        backend = self.make_backend()
        record = backend.enqueue(KIND_DRAFT, '{"k": 1}')
        backend.claim([KIND_DRAFT], worker_id="w1")
        backend.fail(record.job_id, error="boom", error_class="RuntimeError")
        failed = backend.get(record.job_id)
        self.assertEqual(failed.status, job_queue.STATUS_ERROR)
        self.assertEqual(failed.error, "boom")
        self.assertEqual(failed.error_class, "RuntimeError")

    def test_requeue_restores_job_without_burning_attempt(self):
        backend = self.make_backend()
        record = backend.enqueue(KIND_EVALUATE, '{"k": 1}')
        backend.claim([KIND_EVALUATE], worker_id="w1")
        self.assertTrue(backend.requeue(record.job_id, reason="draining"))
        requeued = backend.get(record.job_id)
        self.assertEqual(requeued.status, job_queue.STATUS_QUEUED)
        self.assertEqual(requeued.attempts, 0)
        self.assertEqual(backend.depth(), 1)
        self.assertFalse(backend.requeue(record.job_id, reason="again"))

    def test_requeue_stale_recovers_abandoned_job(self):
        backend = self.make_backend()
        record = backend.enqueue(KIND_EVALUATE, '{"k": 1}')
        backend.claim([KIND_EVALUATE], worker_id="dead-worker")
        with patch.object(config, "JOB_QUEUE_LEASE_SECONDS", 0):
            self.assertEqual(backend.requeue_stale(), 1)
        recovered = backend.get(record.job_id)
        self.assertEqual(recovered.status, job_queue.STATUS_QUEUED)
        self.assertEqual(backend.depth(), 1)
        # Stale recovery KEEPS the attempt, unlike the draining requeue which gives
        # it back: the job did start here (possibly spending LLM calls), and the
        # attempt counter is what eventually parks a poison job that kills every
        # worker instead of retrying it forever.
        self.assertEqual(recovered.attempts, 1)
        # The lease entry is gone, so a second sweep finds nothing to do.
        with patch.object(config, "JOB_QUEUE_LEASE_SECONDS", 0):
            self.assertEqual(backend.requeue_stale(), 0)

    def test_abandoned_job_is_parked_after_max_attempts(self):
        backend = self.make_backend()
        record = backend.enqueue(KIND_EVALUATE, '{"k": 1}')
        for _ in range(job_queue.MAX_ATTEMPTS):
            claimed = backend.claim([KIND_EVALUATE], worker_id="flaky")
            self.assertIsNotNone(claimed)
            with patch.object(config, "JOB_QUEUE_LEASE_SECONDS", 0):
                backend.requeue_stale()
        parked = backend.get(record.job_id)
        self.assertEqual(parked.status, job_queue.STATUS_ERROR)
        self.assertEqual(parked.error_class, "JobAbandoned")

    def test_claim_returns_none_when_idle(self):
        backend = self.make_backend()
        with patch.object(config, "JOB_QUEUE_CLAIM_TIMEOUT_SECONDS", FAST_CLAIM):
            self.assertIsNone(backend.claim([KIND_EVALUATE], worker_id="w1"))

    def test_claim_ignores_terminal_jobs(self):
        backend = self.make_backend()
        record = backend.enqueue(KIND_EVALUATE, '{"k": 1}')
        backend.fail(record.job_id, error="pre-failed")
        # Re-queuing a terminal job must not resurrect it.
        with patch.object(config, "JOB_QUEUE_CLAIM_TIMEOUT_SECONDS", FAST_CLAIM):
            self.assertIsNone(backend.claim([KIND_EVALUATE], worker_id="w1"))


class TestInProcessBackend(_BackendCase, unittest.TestCase):
    def make_backend(self):  # noqa: ANN201
        return InProcessJobBackend(prefix="t", ttl_seconds=60)


class TestRedisBackend(_BackendCase, unittest.TestCase):
    def make_backend(self):  # noqa: ANN201
        self.client = _FakeRedisClient()
        module = _FakeRedisModule(self.client)
        self._patcher = patch.dict(sys.modules, {"redis": module})
        self._patcher.start()
        self.addCleanup(self._patcher.stop)
        self._config_patch = patch.object(config, "JOB_QUEUE_CLAIM_TIMEOUT_SECONDS", FAST_CLAIM)
        self._config_patch.start()
        self.addCleanup(self._config_patch.stop)
        return RedisJobBackend("redis://localhost:6379/0", prefix="t", ttl_seconds=60)

    def test_keys_use_the_configured_prefix(self):
        backend = self.make_backend()
        record = backend.enqueue(KIND_EVALUATE, "{}")
        self.assertIn(f"t:job:{record.job_id}:meta", self.client.store)
        self.assertIn(f"t:job:{record.job_id}:payload", self.client.store)
        self.assertIn("t:jobs:evaluate", self.client.lists)

    def test_missing_redis_package_raises_unavailable(self):
        with patch.dict(sys.modules, {"redis": None}):
            with self.assertRaises(job_queue.JobQueueUnavailable):
                RedisJobBackend("redis://localhost:6379/0", prefix="t", ttl_seconds=60)


class TestUpstashBackend(_BackendCase, unittest.TestCase):
    def make_backend(self):  # noqa: ANN201
        self.transport = _FakeUpstashTransport()
        self._patcher = patch("urllib.request.urlopen", self.transport.urlopen)
        self._patcher.start()
        self.addCleanup(self._patcher.stop)
        self._config_patch = patch.object(config, "JOB_QUEUE_CLAIM_TIMEOUT_SECONDS", FAST_CLAIM)
        self._config_patch.start()
        self.addCleanup(self._config_patch.stop)
        # Upstash polling is slower by design (REST has no blocking pop).
        self._poll_patch = patch.object(config, "JOB_QUEUE_POLL_SECONDS", 0.01)
        self._poll_patch.start()
        self.addCleanup(self._poll_patch.stop)
        return UpstashJobBackend(
            "https://unit-test.upstash.io", "token-123", prefix="t", ttl_seconds=60, timeout_seconds=1.0
        )

    def test_uses_json_command_api_with_basic_auth(self):
        backend = self.make_backend()
        backend.enqueue(KIND_EVALUATE, '{"k": 1}')
        ops = [str(cmd[0]).upper() for cmd in self.transport.commands]
        self.assertIn("SET", ops)
        self.assertIn("LPUSH", ops)
        expected = "Basic " + base64.b64encode(b"token-123").decode("utf-8")
        self.assertEqual(self.transport.headers[0]["Authorization"], expected)

    def test_set_records_carry_a_ttl(self):
        backend = self.make_backend()
        record = backend.enqueue(KIND_EVALUATE, '{"k": 1}')
        set_commands = [c for c in self.transport.commands if str(c[0]).upper() == "SET"]
        meta_set = [c for c in set_commands if c[1] == f"t:job:{record.job_id}:meta"]
        self.assertTrue(meta_set)
        self.assertEqual(str(meta_set[0][3]).upper(), "EX")
        self.assertEqual(meta_set[0][4], 60)

    def test_upstash_error_payload_raises(self):
        backend = self.make_backend()

        def failing(*_args, **_kwargs):
            return _FakeUpstashResponse(json.dumps({"error": "ERR bad command"}).encode("utf-8"))

        with patch("urllib.request.urlopen", failing):
            with self.assertRaises(job_queue.JobQueueError):
                backend.enqueue(KIND_EVALUATE, "{}")

    def test_depth_and_ping_tolerate_a_broken_backend(self):
        """Health reporting must degrade, not raise (it runs on /health)."""
        backend = self.make_backend()

        def failing(*_args, **_kwargs):
            raise OSError("network down")

        with patch("urllib.request.urlopen", failing):
            self.assertEqual(backend.depth(), 0)
            self.assertFalse(backend.ping())


# ------------------------------------------------------------------- selection
class TestBuildJobBackend(unittest.TestCase):
    def test_disabled_by_default_uses_inprocess(self):
        with patch.object(config, "JOB_QUEUE_ENABLED", False):
            self.assertIsInstance(build_job_backend(), InProcessJobBackend)

    def test_redis_url_wins_over_shared_cache(self):
        client = _FakeRedisClient()
        with patch.dict(sys.modules, {"redis": _FakeRedisModule(client)}), patch.object(
            config, "JOB_QUEUE_ENABLED", True
        ), patch.object(config, "JOB_QUEUE_REDIS_URL", "redis://redis:6379/0"), patch.object(
            config, "SHARED_CACHE_URL", "https://x.upstash.io"
        ), patch.object(config, "SHARED_CACHE_TOKEN", "tok"):
            self.assertIsInstance(build_job_backend(), RedisJobBackend)

    def test_shared_cache_credentials_select_upstash(self):
        with patch.object(config, "JOB_QUEUE_ENABLED", True), patch.object(
            config, "JOB_QUEUE_REDIS_URL", ""
        ), patch.object(config, "SHARED_CACHE_URL", "https://x.upstash.io"), patch.object(
            config, "SHARED_CACHE_TOKEN", "tok"
        ):
            self.assertIsInstance(build_job_backend(), UpstashJobBackend)

    def test_enabled_without_backend_warns_and_falls_back(self):
        with patch.object(config, "JOB_QUEUE_ENABLED", True), patch.object(
            config, "JOB_QUEUE_REDIS_URL", ""
        ), patch.object(config, "SHARED_CACHE_URL", ""), patch.object(
            config, "SHARED_CACHE_TOKEN", ""
        ), self.assertLogs("app.job_queue", level="WARNING") as captured:
            self.assertIsInstance(build_job_backend(), InProcessJobBackend)
        self.assertTrue(any("no shared backend" in line for line in captured.output))

    def test_get_job_backend_is_cached_until_reset(self):
        job_queue.reset_job_backend_for_tests()
        try:
            with patch.object(config, "JOB_QUEUE_ENABLED", False):
                first = job_queue.get_job_backend()
                second = job_queue.get_job_backend()
            self.assertIs(first, second)
            self.assertFalse(job_queue.queue_is_distributed())
        finally:
            job_queue.reset_job_backend_for_tests()


class TestBackendHealth(unittest.TestCase):
    """``/health`` must not do network I/O: kubelet polls it every 10s.

    ``depth()`` is ``LLEN`` per kind on Redis and two HTTP requests on Upstash, and
    on an unreachable tier it returns 0 rather than raising — so reporting it by
    default would both block the liveness probe and, worse, report an empty healthy
    backlog exactly when the truth was unknown.
    """

    class _Remote(JobBackend):
        name = "redis"
        is_distributed = True

        def __init__(self):
            self.reads = 0

        def depth(self):
            self.reads += 1
            return 5

    def test_local_backend_reports_depth_without_being_asked(self):
        backend = InProcessJobBackend(prefix="t", ttl_seconds=60)
        payload = backend.health()
        self.assertEqual(payload["depth"], 0)
        self.assertEqual(payload["depth_source"], "local")
        self.assertNotIn("depth_note", payload)

    def test_remote_backend_omits_depth_by_default(self):
        backend = self._Remote()
        payload = backend.health()
        self.assertIsNone(payload["depth"])
        self.assertEqual(payload["depth_source"], "not_probed")
        self.assertEqual(backend.reads, 0)
        # None, not 0: an unread backlog is not an empty one.
        self.assertIsNot(payload["depth"], 0)

    def test_remote_backend_reports_depth_when_probed(self):
        backend = self._Remote()
        payload = backend.health(probe=True)
        self.assertEqual(payload["depth"], 5)
        self.assertEqual(payload["depth_source"], "probed")
        self.assertEqual(backend.reads, 1)

    def test_health_still_reports_the_backend_identity_when_not_probed(self):
        payload = self._Remote().health()
        self.assertEqual(payload["backend"], "redis")
        self.assertTrue(payload["is_distributed"])


class TestJobRecord(unittest.TestCase):
    def test_roundtrip(self):
        record = JobRecord(job_id="j1", kind=KIND_EVALUATE, request_id="req_1")
        again = JobRecord.from_json(record.to_json())
        self.assertEqual(again.job_id, "j1")
        self.assertEqual(again.request_id, "req_1")

    def test_tolerates_corrupt_and_foreign_fields(self):
        self.assertIsNone(JobRecord.from_json(None))
        self.assertIsNone(JobRecord.from_json(""))
        self.assertIsNone(JobRecord.from_json("{not json"))
        self.assertIsNone(JobRecord.from_json("[1, 2]"))
        # A field written by a newer pod must not break this one.
        record = JobRecord.from_json('{"job_id": "j2", "kind": "draft", "future_field": 7}')
        self.assertEqual(record.job_id, "j2")
        self.assertFalse(hasattr(record, "future_field"))

    def test_in_progress_flag(self):
        record = JobRecord(job_id="j3", kind=KIND_EVALUATE)
        self.assertFalse(record.is_terminal)

    def test_job_ids_are_unique_and_prefixed(self):
        ids = {job_queue.new_job_id() for _ in range(50)}
        self.assertEqual(len(ids), 50)
        self.assertTrue(all(i.startswith("job_") for i in ids))


class TestInProcessBackendBlockingClaim(unittest.TestCase):
    def test_claim_waits_for_a_job_enqueued_by_another_thread(self):
        import threading
        import time

        backend = InProcessJobBackend(prefix="t", ttl_seconds=60)
        result: list = []

        def claim() -> None:
            result.append(backend.claim([KIND_EVALUATE], worker_id="w1"))

        with patch.object(config, "JOB_QUEUE_CLAIM_TIMEOUT_SECONDS", 5):
            thread = threading.Thread(target=claim, daemon=True)
            thread.start()
            time.sleep(0.1)
            backend.enqueue(KIND_EVALUATE, '{"k": 1}')
            thread.join(timeout=5)
        self.assertTrue(result)
        self.assertIsNotNone(result[0])
        self.assertEqual(result[0][1], '{"k": 1}')


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
