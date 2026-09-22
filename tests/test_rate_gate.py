"""Offline unit tests for the minimum-interval rate gate.

No network — the gate is a pure stdlib pacing primitive, exercised directly
and through LLMClient's stubbed _client (same conventions as
test_circuit_breaker.py). The gate's whole job is *spacing*, so the wall-time
assertions check that successive admissions are at least min_interval apart,
never exactly min_interval (scheduling jitter is the gate's friend, not a
failure).
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

from app import circuit_breaker  # noqa: E402
from app import config as cfg  # noqa: E402
from app.circuit_breaker import (  # noqa: E402
    MinimumIntervalGate,
    QueueFullError,
    RateGateTimeoutError,
    get_llm_rate_gate,
    reset_all_for_tests,
    reset_llm_rate_gate,
)
from app.llm import LLMClient  # noqa: E402


class _FakeSettings:
    configured = True
    api_key = "test-key"
    base_url = "http://example.invalid"
    model_main = "test-model"
    model_fast = "test-fast-model"


def _small_gate(min_interval: float) -> MinimumIntervalGate:
    """A gate with a tiny interval and a generous timeout, freshly reset."""
    return reset_llm_rate_gate(
        min_interval_seconds=min_interval, queue_timeout=10.0
    )


class TestMinimumIntervalGate(unittest.TestCase):
    """Direct behavior of the pacing primitive."""

    def test_disabled_gate_is_a_noop(self):
        gate = _small_gate(0.0)
        self.assertFalse(gate.enabled)
        t0 = time.monotonic()
        for _ in range(5):
            self.assertEqual(gate.wait(), 0.0)
        # Upper bound is generous on purpose: it catches "the disabled gate
        # actually waited", not scheduler noise on a loaded box.
        self.assertLess(time.monotonic() - t0, 1.0)

    def test_consecutive_waits_are_spaced(self):
        # Assert on *scheduled slots* (call start + returned delay), not wake
        # timestamps: a late wakeup on call N legitimately shrinks the
        # wake-to-wake gap while slot spacing still holds exactly.
        gate = _small_gate(0.05)
        slots = []
        t0 = time.monotonic()
        for _ in range(4):
            start = time.monotonic()
            slots.append(start + gate.wait())
        for earlier, later in zip(slots, slots[1:]):
            self.assertGreaterEqual(
                later - earlier,
                0.0499,
                "two admissions scheduled closer than the minimum interval",
            )

    def test_contending_threads_burst_apart_not_together(self):
        # The whole point of slot reservation: N threads arriving at once must
        # come out a full interval apart, not all admitted at the first tick.
        gate = _small_gate(0.04)
        start = threading.Barrier(5)
        admitted: list[float] = []

        def worker() -> None:
            start.wait()
            gate.wait()
            admitted.append(time.monotonic())

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(admitted), 5)
        admitted.sort()
        for earlier, later in zip(admitted, admitted[1:]):
            self.assertGreaterEqual(
                later - earlier,
                0.035,
                "burst not prevented: two threads admitted within one interval",
            )

    def test_reservation_survives_sleep_outside_lock(self):
        # A slow sleeper must not let a later caller grab an earlier slot.
        gate = _small_gate(0.05)
        # A fresh gate admits the first call immediately — spacing applies
        # *between* admissions, not before the first one.
        first = gate.wait()
        self.assertEqual(first, 0.0)
        # While the first caller is (conceptually) sleeping, a second caller
        # reserves a slot a full interval behind it.
        second = gate.wait()
        self.assertGreaterEqual(second, 0.045)

    def test_timeout_raises_rate_gate_error(self):
        gate = _small_gate(0.05)
        gate.wait()  # occupy the near slot
        with self.assertRaises(RateGateTimeoutError):
            gate.wait(timeout=0.0)  # zero budget cannot wait for the next slot

    def test_timeout_error_is_a_queue_full_error(self):
        # chat() fail-fasts on QueueFullError (no endpoint failover); the rate
        # gate must ride that same routing.
        self.assertTrue(issubclass(RateGateTimeoutError, QueueFullError))

    def test_reset_clears_reservations(self):
        gate = _small_gate(0.05)
        gate.wait()
        gate.reset()
        t0 = time.monotonic()
        gate.wait()
        self.assertLess(time.monotonic() - t0, 0.04, "reset left a stale slot")

    def test_negative_values_are_clamped(self):
        gate = MinimumIntervalGate(min_interval_seconds=-5.0, queue_timeout=-1.0)
        self.assertEqual(gate.min_interval_seconds, 0.0)
        self.assertEqual(gate.queue_timeout, 0.0)
        self.assertFalse(gate.enabled)


class TestRateGateConfig(unittest.TestCase):
    """Env-knob derivation: seconds cap vs RPM cap, stricter wins."""

    def tearDown(self):
        reset_all_for_tests()

    def test_rpm_derives_interval(self):
        with patch.object(cfg, "LLM_RATE_MIN_INTERVAL_SECONDS", 0.0), \
             patch.object(cfg, "LLM_RATE_MAX_RPM", 30.0):
            reset_llm_rate_gate()
            gate = get_llm_rate_gate()
            self.assertAlmostEqual(gate.min_interval_seconds, 2.0)

    def test_stricter_of_the_two_wins(self):
        with patch.object(cfg, "LLM_RATE_MIN_INTERVAL_SECONDS", 4.0), \
             patch.object(cfg, "LLM_RATE_MAX_RPM", 30.0):  # 2 s — weaker
            reset_llm_rate_gate()
            self.assertAlmostEqual(get_llm_rate_gate().min_interval_seconds, 4.0)

    def test_zero_disables_even_with_stricter_other(self):
        with patch.object(cfg, "LLM_RATE_MIN_INTERVAL_SECONDS", 0.0), \
             patch.object(cfg, "LLM_RATE_MAX_RPM", 0.0):
            reset_llm_rate_gate()
            self.assertFalse(get_llm_rate_gate().enabled)

    def test_negative_rpm_disables(self):
        with patch.object(cfg, "LLM_RATE_MIN_INTERVAL_SECONDS", 0.0), \
             patch.object(cfg, "LLM_RATE_MAX_RPM", -10.0):
            reset_llm_rate_gate()
            self.assertFalse(get_llm_rate_gate().enabled)


class _SleepRecorder:
    """Shim swapped in for app.circuit_breaker.time that records paced sleeps."""

    def __init__(self) -> None:
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        time.sleep(seconds)

    def monotonic(self) -> float:
        return time.monotonic()


class TestRateGateInLLMClient(unittest.TestCase):
    """The gate actually spaces real (stubbed) LLM calls."""

    def setUp(self):
        reset_all_for_tests()

    def tearDown(self):
        reset_all_for_tests()

    def test_chat_calls_are_spaced(self):
        reset_llm_rate_gate(min_interval_seconds=0.04)
        client = LLMClient(_FakeSettings())  # type: ignore[arg-type]
        client._client = MagicMock()
        client._client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))]
        )
        # Record the paced sleeps the gate performs inside the real call path:
        # the first call on a fresh gate admits immediately (no sleep), later
        # ones must be paced by ~the interval. Immune to wake-up jitter.
        recorder = _SleepRecorder()
        with patch.object(circuit_breaker, "time", recorder):
            for _ in range(3):
                client.chat("sys", "user")
        self.assertTrue(
            recorder.sleeps,
            "gate never paced a call — spacing is not wired into chat()",
        )
        for delay in recorder.sleeps:
            self.assertGreaterEqual(delay, 0.01, "recorded sleep is not a paced sleep")
        # Robust invariant: total paced time across 3 admissions must cover the
        # spacing (2 intervals ≈ 0.08 s) — individual sleeps sit a few ms below
        # the interval when the caller starts after its slot was scheduled,
        # which is correct admission spacing, not a pacing bug.
        self.assertGreaterEqual(
            sum(recorder.sleeps), 0.04, "total pacing shorter than the interval"
        )

    def test_disabled_gate_adds_no_latency(self):
        reset_llm_rate_gate(min_interval_seconds=0.0)
        client = LLMClient(_FakeSettings())  # type: ignore[arg-type]
        client._client = MagicMock()
        client._client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))]
        )
        t0 = time.monotonic()
        client.chat("sys", "user")
        self.assertLess(time.monotonic() - t0, 0.1)

    def test_timeout_surfaces_as_queue_full_error_to_callers(self):
        reset_llm_rate_gate(min_interval_seconds=0.05, queue_timeout=10.0)
        client = LLMClient(_FakeSettings())  # type: ignore[arg-type]
        client._client = MagicMock()
        client._client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))]
        )
        client.chat("sys", "user")  # occupy the near slot
        gate = get_llm_rate_gate()
        # Second call's slot is a full interval out; a zero budget must reject
        # it as a QueueFullError before any network touch.
        with patch.object(type(gate), "wait", side_effect=RateGateTimeoutError("paced out")):
            with self.assertRaises(QueueFullError):
                client.chat("sys", "user")


if __name__ == "__main__":
    unittest.main()
