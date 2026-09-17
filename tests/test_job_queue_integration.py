"""End-to-end test of the Pattern C flow across two independent clients.

The unit tests in ``test_job_queue.py`` exercise each backend against an
in-memory fake. That verifies the logic but not the actual claim this feature
rests on: that a job enqueued by one process (the Streamlit web pod) is claimed
by *another* (a worker) purely through the shared backend, and that the result
comes back decodable.

This runs the Upstash REST protocol against a real ``ThreadingHTTPServer`` on a
loopback port, with a producer backend and a consumer backend constructed
separately — so nothing is shared except the HTTP contract. It deliberately uses
the Upstash tier because it is stdlib-only: no Redis server, no extra package, so
CI and local runs behave identically.
"""
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
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
    JobQueueError,
    UpstashJobBackend,
)
from app.usage import UsageTracker  # noqa: E402


class _FakeUpstashHandler(BaseHTTPRequestHandler):
    """Serves the Upstash JSON command API over an in-memory store.

    Implements exactly the commands app/job_queue.py issues, so a divergence
    between the client and the documented protocol shows up as a failed command
    rather than a silently green test.
    """

    store: dict[str, str] = {}
    lists: dict[str, list[str]] = {}
    zsets: dict[str, dict[str, float]] = {}
    lock = threading.Lock()
    commands: list[list] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        with self.lock:
            self.commands.append(body)
            try:
                result = self._run(body)
            except Exception as exc:  # noqa: BLE001 - surface as a Redis-style error
                payload = {"error": f"ERR {exc}"}
            else:
                payload = {"result": result}
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

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
        raise ValueError(f"unsupported command {op}")

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class _UsageStub:
    """Minimal pipeline stub: records usage and answers every phase."""

    def __init__(self) -> None:
        self._settings = MagicMock(model_fast="fake-fast", model_main="fake-main")
        self.usage = UsageTracker()
        from test_evaluate import _FakeLLM

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


class TestTwoClientQueueFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstashHandler)
        cls.server.daemon_threads = True
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        _FakeUpstashHandler.store.clear()
        _FakeUpstashHandler.lists.clear()
        _FakeUpstashHandler.zsets.clear()
        _FakeUpstashHandler.commands.clear()
        self._tmpdir = isolate_app_logs(self)
        shutdown.reset_for_tests()
        self.addCleanup(shutdown.reset_for_tests)

    def _client(self) -> UpstashJobBackend:
        """A fresh backend instance — as a separate process would build."""
        return UpstashJobBackend(
            self.url, "integration-token", prefix="va_lse", ttl_seconds=600, timeout_seconds=5.0
        )

    def test_job_submitted_by_one_client_completes_on_another(self):
        producer = self._client()
        consumer = self._client()
        self.assertIsNot(producer, consumer)

        job = EvaluateJob(
            statement_text="I injured my knee lifting a pallet.",
            records=[document_from_text("a.txt", "EVT knee pain noted during service.")],
            request_id="req_xproc",
            record_sources=["Upload"],
        )
        record = producer.enqueue(
            KIND_EVALUATE, encode_job(KIND_EVALUATE, job), request_id=job.request_id
        )
        self.assertEqual(producer.depth(), 1)

        # The consumer sees the job through the shared backend only.
        with patch.object(config, "JOB_QUEUE_CLAIM_TIMEOUT_SECONDS", 2), patch.object(
            config, "JOB_QUEUE_POLL_SECONDS", 0.01
        ):
            claimed = consumer.claim([KIND_EVALUATE], worker_id="worker-1")
        self.assertIsNotNone(claimed)
        claimed_record, payload = claimed
        self.assertEqual(claimed_record.job_id, record.job_id)
        self.assertIn("knee pain noted during service", payload)

        self.assertTrue(worker.execute_job(claimed_record, payload, consumer, llm=_UsageStub()))

        # ...and the producer can read the finished result back.
        finished = producer.get(record.job_id)
        self.assertEqual(finished.status, STATUS_DONE)
        self.assertEqual(finished.request_id, "req_xproc")
        run = decode_result(producer.get_result(record.job_id))
        self.assertEqual(run.kind, KIND_EVALUATE)
        self.assertEqual(run.request_id, "req_xproc")
        self.assertTrue(run.result.report_markdown)
        self.assertGreater(run.usage.totals().calls, 0)

    def test_worker_once_drains_a_job_queued_by_another_client(self):
        producer = self._client()
        job = DraftJob(
            records=[document_from_text("a.txt", "EVT knee pain noted.")],
            witness={"name": "Jane Doe"},
            observations="Daily knee pain.",
            condition="knee strain",
            claim_type="Service connection (new claim)",
            request_id="req_draft_xproc",
        )
        record = producer.enqueue(KIND_DRAFT, encode_job(KIND_DRAFT, job))

        consumer = self._client()
        with patch.object(config, "JOB_QUEUE_CLAIM_TIMEOUT_SECONDS", 2), patch.object(
            config, "JOB_QUEUE_POLL_SECONDS", 0.01
        ):
            stats = worker.run_worker(
                once=True, backend=consumer, worker_id="worker-once", llm=_UsageStub()
            )

        self.assertEqual(stats.claimed, 1)
        self.assertEqual(stats.completed, 1)
        self.assertEqual(producer.get(record.job_id).status, STATUS_DONE)
        self.assertTrue(decode_result(producer.get_result(record.job_id)).result.output_statement)

    def test_second_client_cannot_double_claim_a_running_job(self):
        """A duplicate queue entry must not start a second concurrent run."""
        producer = self._client()
        consumer = self._client()
        record = producer.enqueue(KIND_EVALUATE, "{}")
        with patch.object(config, "JOB_QUEUE_CLAIM_TIMEOUT_SECONDS", 1), patch.object(
            config, "JOB_QUEUE_POLL_SECONDS", 0.01
        ):
            first = consumer.claim([KIND_EVALUATE], worker_id="w1")
            self.assertIsNotNone(first)
            # Force the id back onto the queue, as a race would.
            consumer._rest.command("LPUSH", "va_lse:jobs:evaluate", record.job_id)
            second = consumer.claim([KIND_EVALUATE], worker_id="w2")
        self.assertIsNone(second, "a job already running must not be claimed again")

    def test_abandoned_job_is_recovered_by_another_client(self):
        producer = self._client()
        consumer = self._client()
        record = producer.enqueue(KIND_EVALUATE, "{}")
        with patch.object(config, "JOB_QUEUE_CLAIM_TIMEOUT_SECONDS", 2), patch.object(
            config, "JOB_QUEUE_POLL_SECONDS", 0.01
        ):
            self.assertIsNotNone(consumer.claim([KIND_EVALUATE], worker_id="doomed"))

        # The worker died; its lease expires and a peer re-queues the job.
        with patch.object(config, "JOB_QUEUE_LEASE_SECONDS", 0):
            self.assertEqual(producer.requeue_stale(), 1)
        recovered = producer.get(record.job_id)
        self.assertEqual(recovered.status, "queued")
        self.assertEqual(producer.depth(), 1)

    def test_health_reports_the_distributed_backend(self):
        backend = self._client()
        health = backend.health()
        self.assertEqual(health["backend"], "upstash_rest")
        self.assertTrue(health["is_distributed"])

    def test_unreachable_backend_degrades_without_raising(self):
        """A dead queue must not take down a page render or a health probe.

        Reads and health probes swallow the failure and report nothing available;
        a write must raise so the submit path can tell the user the run was not
        queued instead of leaving them waiting on a job that does not exist.
        """
        dead = UpstashJobBackend(
            "http://127.0.0.1:1", "token", prefix="va_lse", ttl_seconds=60, timeout_seconds=0.5
        )
        self.assertEqual(dead.depth(), 0)
        self.assertFalse(dead.ping())
        self.assertIsNone(dead.get("job_missing"))
        self.assertIsNone(dead.get_result("job_missing"))
        with self.assertRaises(JobQueueError):
            dead.enqueue(KIND_EVALUATE, "{}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
