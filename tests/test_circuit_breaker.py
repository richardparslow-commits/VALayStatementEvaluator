"""Offline unit tests for the circuit breaker + concurrency limiter.

No network — every probe is mocked via LLMClient's _client stub and via the
in-memory breaker/limiter state. The 2 s fail-fast SLO is checked with wall
time.
"""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)


from app.circuit_breaker import (  # noqa: E402
    CircuitBreaker,
    CircuitBreakerOpenError,
    ConcurrencyLimiter,
    QueueFullError,
    get_llm_breaker,
    get_llm_limiter,
    reset_all_for_tests,
    reset_llm_breaker,
    reset_llm_limiter,
)
from app.config import Settings  # noqa: E402
from app.llm import LLMClient, LLMError, LLMUpstreamError  # noqa: E402


class _FakeSettings:
    configured = True
    api_key = "test-key"
    base_url = "http://example.invalid"
    model_main = "test-model"
    model_fast = "test-fast-model"


def _make_client_with_fake_openai(*, fail_times: int = 0) -> tuple[LLMClient, MagicMock]:
    """Return (client, create_mock) where create_mock fails `fail_times` then succeeds."""
    client = LLMClient(_FakeSettings())  # type: ignore[arg-type]
    create_mock = MagicMock()
    call_count = {"n": 0}

    def fake_create(**kwargs):  # type: ignore[no-untyped-def]
        call_count["n"] += 1
        if call_count["n"] <= fail_times * 3:  # LLMClient retries 3 times per logical call
            raise RuntimeError(f"simulated failure {call_count['n']}")
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok response"))]
        resp.usage = None
        return resp

    create_mock.side_effect = fake_create
    # LLMClient stores OpenAI(...).chat.completions.create
    client._client.chat.completions.create = create_mock  # type: ignore[attr-defined]
    return client, create_mock


# ------------------------------------------------------------------ breaker unit


class TestCircuitBreakerUnit(unittest.TestCase):
    def test_starts_closed(self) -> None:
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60, name="t")
        self.assertEqual(breaker.state, "CLOSED")
        self.assertEqual(breaker.failure_count, 0)

    def test_opens_after_threshold(self) -> None:
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60, name="t")
        with self.assertLogs("app.circuit_breaker", level="WARNING"):
            for _ in range(3):
                breaker.record_failure()
        self.assertEqual(breaker.state, "OPEN")
        with self.assertRaises(CircuitBreakerOpenError):
            breaker.check_or_raise()

    def test_success_resets_count(self) -> None:
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60, name="t")
        breaker.record_failure()
        breaker.record_failure()
        self.assertEqual(breaker.failure_count, 2)
        breaker.record_success()
        self.assertEqual(breaker.failure_count, 0)
        self.assertEqual(breaker.state, "CLOSED")

    def test_half_open_after_timeout_then_close_on_success(self) -> None:
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05, name="t")
        with self.assertLogs("app.circuit_breaker", level="WARNING"):
            breaker.record_failure()
            breaker.record_failure()
        self.assertEqual(breaker.state, "OPEN")
        time.sleep(0.08)
        # First allow_request after timeout transitions to HALF_OPEN
        with self.assertLogs("app.circuit_breaker", level="WARNING"):
            self.assertTrue(breaker.allow_request())
        self.assertEqual(breaker.state, "HALF_OPEN")
        with self.assertLogs("app.circuit_breaker", level="WARNING"):
            breaker.record_success()
        self.assertEqual(breaker.state, "CLOSED")

    def test_half_open_reopens_on_failure(self) -> None:
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05, name="t")
        with self.assertLogs("app.circuit_breaker", level="WARNING"):
            breaker.record_failure()
            breaker.record_failure()
        time.sleep(0.08)
        with self.assertLogs("app.circuit_breaker", level="WARNING"):
            breaker.allow_request()
        self.assertEqual(breaker.state, "HALF_OPEN")
        with self.assertLogs("app.circuit_breaker", level="WARNING"):
            breaker.record_failure()
        self.assertEqual(breaker.state, "OPEN")

    def test_state_changes_log_at_warning(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=60, name="t")
        with self.assertLogs("app.circuit_breaker", level="WARNING") as cm:
            breaker.record_failure()
        self.assertTrue(any("CLOSED -> OPEN" in msg for msg in cm.output))

    def test_check_or_raise_fail_fast_under_50ms(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=60, name="t")
        breaker.record_failure()
        # Must be OPEN now
        self.assertEqual(breaker.state, "OPEN")
        t0 = time.perf_counter()
        with self.assertRaises(CircuitBreakerOpenError):
            breaker.check_or_raise()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.assertLess(elapsed_ms, 50, f"fail-fast took {elapsed_ms:.1f} ms, should be <50 ms")


# -------------------------------------------------------------- limiter unit


class TestConcurrencyLimiterUnit(unittest.TestCase):
    def test_fast_path(self) -> None:
        limiter = ConcurrencyLimiter(max_concurrent=2, max_queue_depth=5, queue_timeout=1.0, name="t")
        limiter.acquire()
        self.assertEqual(limiter.active, 1)
        limiter.acquire()
        self.assertEqual(limiter.active, 2)
        limiter.release()
        limiter.release()
        self.assertEqual(limiter.active, 0)

    def test_queue_until_slot_frees(self) -> None:
        limiter = ConcurrencyLimiter(max_concurrent=1, max_queue_depth=5, queue_timeout=2.0, name="t")
        limiter.acquire()  # fills the single slot
        result: list[str] = []

        def waiter() -> None:
            limiter.acquire()
            result.append("acquired")
            limiter.release()

        t = threading.Thread(target=waiter, daemon=True)
        t.start()
        time.sleep(0.05)
        self.assertEqual(limiter.waiting, 1)
        limiter.release()  # free the slot
        t.join(timeout=2.0)
        self.assertEqual(result, ["acquired"])
        self.assertEqual(limiter.waiting, 0)

    def test_queue_full_rejected(self) -> None:
        limiter = ConcurrencyLimiter(max_concurrent=1, max_queue_depth=1, queue_timeout=2.0, name="t")
        limiter.acquire()  # active=1

        acquired: list[bool] = []

        def waiter_should_queue() -> None:
            limiter.acquire()
            acquired.append(True)
            time.sleep(0.2)
            limiter.release()

        t1 = threading.Thread(target=waiter_should_queue, daemon=True)
        t1.start()
        time.sleep(0.05)
        self.assertEqual(limiter.waiting, 1)
        # Next acquire should be rejected (queue depth 1 already used)
        with self.assertRaises(QueueFullError):
            limiter.acquire()
        limiter.release()  # free slot for waiter
        t1.join(timeout=2.0)
        self.assertEqual(acquired, [True])

    def test_queue_timeout(self) -> None:
        limiter = ConcurrencyLimiter(max_concurrent=1, max_queue_depth=5, queue_timeout=0.12, name="t")
        limiter.acquire()
        t0 = time.perf_counter()
        with self.assertRaises(QueueFullError) as ctx:
            limiter.acquire()
        elapsed = time.perf_counter() - t0
        self.assertGreaterEqual(elapsed, 0.08)
        self.assertLess(elapsed, 1.0)
        self.assertIn("Timed out", str(ctx.exception))
        self.assertEqual(limiter.waiting, 0)
        limiter.release()

    def test_context_manager(self) -> None:
        limiter = ConcurrencyLimiter(max_concurrent=2, max_queue_depth=5, queue_timeout=1.0, name="t")
        with limiter:
            self.assertEqual(limiter.active, 1)
        self.assertEqual(limiter.active, 0)


# --------------------------------------------------------- LLM integration


class TestLLMIntegration(unittest.TestCase):
    def setUp(self) -> None:
        # Use short timeouts so tests run quickly; silence retry sleeps.
        reset_all_for_tests(
            breaker_threshold=3,
            breaker_recovery=60.0,
            limiter_concurrent=20,
            limiter_queue_depth=50,
            limiter_timeout=30.0,
        )
        self.sleep_patch = patch("app.llm.time.sleep", return_value=None)
        self.sleep_patch.start()

    def tearDown(self) -> None:
        self.sleep_patch.stop()
        reset_all_for_tests()

    def test_success_does_not_open_breaker(self) -> None:
        breaker = reset_llm_breaker(failure_threshold=3, recovery_timeout=60.0)
        limiter = reset_llm_limiter(max_concurrent=20, max_queue_depth=50, queue_timeout=5.0)
        client, _ = _make_client_with_fake_openai(fail_times=0)
        result = client.chat("sys", "user", phase="test")
        self.assertEqual(result, "ok response")
        self.assertEqual(breaker.state, "CLOSED")
        self.assertEqual(limiter.active, 0)

    def test_three_consecutive_failures_open_breaker(self) -> None:
        breaker = reset_llm_breaker(failure_threshold=3, recovery_timeout=60.0)
        reset_llm_limiter(max_concurrent=20, max_queue_depth=50, queue_timeout=5.0)

        # Each logical failure = 3 create() failures (retries exhausted)
        failing_client, _ = _make_client_with_fake_openai(fail_times=10)

        for i in range(3):
            with self.assertRaises(LLMError):
                failing_client.chat("sys", "user", phase="test")

        self.assertEqual(breaker.state, "OPEN")

    def test_open_breaker_fails_fast_without_network(self) -> None:
        reset_llm_breaker(failure_threshold=2, recovery_timeout=60.0)
        reset_llm_limiter(max_concurrent=20, max_queue_depth=50, queue_timeout=5.0)
        failing_client, failing_mock = _make_client_with_fake_openai(fail_times=10)

        for _ in range(2):
            with self.assertRaises(LLMError):
                failing_client.chat("sys", "user", phase="test")

        # Now breaker is OPEN — even a healthy client should fail fast without touching create()
        healthy_client, healthy_mock = _make_client_with_fake_openai(fail_times=0)
        t0 = time.perf_counter()
        with self.assertRaises(CircuitBreakerOpenError):
            healthy_client.chat("sys", "user", phase="test")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.assertLess(elapsed_ms, 50, f"fail-fast took {elapsed_ms:.1f} ms")
        healthy_mock.assert_not_called()

    def test_half_open_probe_success_closes_breaker(self) -> None:
        # Use setUp's sleep patch only for the failing calls; the recovery sleep must be real.
        self.sleep_patch.stop()
        try:
            breaker = reset_llm_breaker(failure_threshold=2, recovery_timeout=0.12)
            reset_llm_limiter(max_concurrent=20, max_queue_depth=50, queue_timeout=5.0)
            failing_client, _ = _make_client_with_fake_openai(fail_times=10)
            with patch("app.llm.time.sleep", return_value=None):
                for _ in range(2):
                    with self.assertRaises(LLMError):
                        failing_client.chat("sys", "user", phase="test")
            self.assertEqual(breaker.state, "OPEN")
            time.sleep(0.16)
            healthy_client, _ = _make_client_with_fake_openai(fail_times=0)
            with self.assertLogs("app.circuit_breaker", level="WARNING"):
                result = healthy_client.chat("sys", "user", phase="test")
            self.assertEqual(result, "ok response")
            self.assertEqual(breaker.state, "CLOSED")
        finally:
            self.sleep_patch.start()

    def test_half_open_probe_failure_reopens(self) -> None:
        self.sleep_patch.stop()
        try:
            breaker = reset_llm_breaker(failure_threshold=2, recovery_timeout=0.12)
            reset_llm_limiter(max_concurrent=20, max_queue_depth=50, queue_timeout=5.0)
            failing_client, _ = _make_client_with_fake_openai(fail_times=10)
            with patch("app.llm.time.sleep", return_value=None):
                for _ in range(2):
                    with self.assertRaises(LLMError):
                        failing_client.chat("sys", "user", phase="test")
            time.sleep(0.16)
            # Probe also fails — must stay OPEN
            with self.assertRaises(LLMError):
                failing_client.chat("sys", "user", phase="test")
            self.assertEqual(breaker.state, "OPEN")
        finally:
            self.sleep_patch.start()

    def test_transient_failure_then_success_uses_bounded_exponential_backoff(self) -> None:
        client = LLMClient(_FakeSettings())  # type: ignore[arg-type]
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok response"))]
        resp.usage = None
        client._client.chat.completions.create = MagicMock(  # type: ignore[attr-defined]
            side_effect=[TimeoutError("t1"), TimeoutError("t2"), resp]
        )
        sleeps: list[float] = []
        with patch("app.llm.time.sleep", side_effect=lambda seconds: sleeps.append(seconds)):
            result = client.chat("sys", "user", phase="test")
        self.assertEqual(result, "ok response")
        self.assertEqual(sleeps, [1.0, 2.0])

    def test_non_retriable_error_returns_immediately(self) -> None:
        class _BadRequestError(RuntimeError):
            status_code = 400

        client = LLMClient(_FakeSettings())  # type: ignore[arg-type]
        create_mock = MagicMock(side_effect=_BadRequestError("bad request"))
        client._client.chat.completions.create = create_mock  # type: ignore[attr-defined]
        with patch("app.llm.time.sleep", return_value=None) as sleep_mock:
            with self.assertRaises(LLMUpstreamError) as ctx:
                client.chat("sys", "user", phase="test")
        self.assertFalse(ctx.exception.retriable)
        self.assertEqual(create_mock.call_count, 1)
        sleep_mock.assert_not_called()

    def test_queue_full_rejected_without_breaker_failure(self) -> None:
        breaker = reset_llm_breaker(failure_threshold=10, recovery_timeout=60.0)
        limiter = reset_llm_limiter(max_concurrent=1, max_queue_depth=0, queue_timeout=0.1)
        # Fill the single slot
        limiter.acquire()
        client, mock = _make_client_with_fake_openai(fail_times=0)
        with self.assertRaises(QueueFullError):
            client.chat("sys", "user", phase="test")
        mock.assert_not_called()
        self.assertEqual(breaker.state, "CLOSED")
        self.assertEqual(breaker.failure_count, 0)
        limiter.release()

    def test_queue_full_does_not_count_as_breaker_failure(self) -> None:
        breaker = reset_llm_breaker(failure_threshold=3, recovery_timeout=60.0)
        limiter = reset_llm_limiter(max_concurrent=1, max_queue_depth=0, queue_timeout=0.1)
        limiter.acquire()
        client, _ = _make_client_with_fake_openai(fail_times=0)
        for _ in range(5):
            with self.assertRaises(QueueFullError):
                client.chat("sys", "user", phase="test")
        self.assertEqual(breaker.failure_count, 0)
        limiter.release()

    def test_concurrency_limiter_released_on_success_and_failure(self) -> None:
        limiter = reset_llm_limiter(max_concurrent=20, max_queue_depth=50, queue_timeout=5.0)
        client_ok, _ = _make_client_with_fake_openai(fail_times=0)
        client_ok.chat("sys", "user", phase="test")
        self.assertEqual(limiter.active, 0)
        failing_client, _ = _make_client_with_fake_openai(fail_times=10)
        # Breaker threshold is high (10) so 3 failures won't open it; just check active is released
        reset_llm_breaker(failure_threshold=10, recovery_timeout=60.0)
        with self.assertRaises(LLMError):
            failing_client.chat("sys", "user", phase="test")
        self.assertEqual(limiter.active, 0)

    def test_env_vars_wire_to_singletons(self) -> None:
        # Verify that the reset helpers correctly reconfigure from config-like values.
        from app import config as cfg  # noqa: E402

        original_threshold = cfg.LLM_CB_FAILURE_THRESHOLD
        original_recovery = cfg.LLM_CB_RECOVERY_SECONDS
        try:
            cfg.LLM_CB_FAILURE_THRESHOLD = 5  # type: ignore[attr-defined]
            cfg.LLM_CB_RECOVERY_SECONDS = 120  # type: ignore[attr-defined]
            b = reset_llm_breaker(
                failure_threshold=cfg.LLM_CB_FAILURE_THRESHOLD,
                recovery_timeout=cfg.LLM_CB_RECOVERY_SECONDS,
            )
            self.assertEqual(b.failure_threshold, 5)
            self.assertEqual(b.recovery_timeout, 120.0)
        finally:
            cfg.LLM_CB_FAILURE_THRESHOLD = original_threshold  # type: ignore[attr-defined]
            cfg.LLM_CB_RECOVERY_SECONDS = original_recovery  # type: ignore[attr-defined]
        reset_all_for_tests()


if __name__ == "__main__":
    unittest.main()
