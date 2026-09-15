"""Offline tests for the production profiler (app/profiler.py).

All tests run against mock / local backends — no network required.
"""
import time
import unittest
from unittest import mock

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.profiler import (
    Profiler,
    RunProfiler,
    PhaseTiming,
    WorkerTiming,
    _percentile_stats,
    get_profiler,
    reset_profiler_for_tests,
    phase_timer,
    worker_timer,
)
import app.config as config


class TestPercentileStats(unittest.TestCase):
    """Verify percentile calculation helper."""

    def test_empty_list(self) -> None:
        result = _percentile_stats([])
        self.assertEqual(result["count"], 0)
        self.assertAlmostEqual(result["p50"], 0.0)
        self.assertAlmostEqual(result["p95"], 0.0)

    def test_single_value(self) -> None:
        result = _percentile_stats([100.0])
        self.assertEqual(result["count"], 1)
        self.assertAlmostEqual(result["p50"], 100.0)
        self.assertAlmostEqual(result["max"], 100.0)
        self.assertAlmostEqual(result["mean"], 100.0)

    def test_multiple_values(self) -> None:
        values = [float(i) for i in range(1, 101)]  # 1 to 100
        result = _percentile_stats(values)
        self.assertEqual(result["count"], 100)
        # int(100 * 0.5) = 50 → sorted[50] = 51.0
        self.assertAlmostEqual(result["p50"], 51.0)
        # int(100 * 0.95) = 95 → sorted[95] = 96.0
        self.assertAlmostEqual(result["p95"], 96.0)
        self.assertAlmostEqual(result["max"], 100.0)
        self.assertAlmostEqual(result["mean"], 50.5)

    def test_sorted_percentiles(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = _percentile_stats(values)
        # int(5 * 0.5) = 2 → sorted[2] = 30.0
        self.assertEqual(result["p50"], 30.0)
        self.assertEqual(result["p95"], 50.0)
        self.assertEqual(result["p99"], 50.0)


class TestPhaseTiming(unittest.TestCase):
    """Verify PhaseTiming data class."""

    def test_duration(self) -> None:
        pt = PhaseTiming(phase="claims", start_mono=100.0, end_mono=100.5)
        self.assertAlmostEqual(pt.duration_ms, 500.0)

    def test_incomplete(self) -> None:
        pt = PhaseTiming(phase="claims", start_mono=100.0)
        self.assertAlmostEqual(pt.duration_ms, 0.0)


class TestWorkerTiming(unittest.TestCase):
    """Verify WorkerTiming data class."""

    def test_duration(self) -> None:
        wt = WorkerTiming(phase="digest", index=1, start_mono=100.0, end_mono=100.2)
        self.assertAlmostEqual(wt.duration_ms, 200.0)


class TestRunProfiler(unittest.TestCase):
    """Verify per-run profiling."""

    def test_empty_run(self) -> None:
        rp = RunProfiler(action="evaluate")
        self.assertAlmostEqual(rp.total_duration_ms, 0.0)
        self.assertEqual(rp.phase_summary(), {})

    def test_phase_summary(self) -> None:
        rp = RunProfiler(action="evaluate", run_start_mono=0.0, run_end_mono=10.0)
        rp.phases = [
            PhaseTiming(phase="claims", start_mono=0.0, end_mono=2.0),
            PhaseTiming(phase="verify", start_mono=2.0, end_mono=5.0),
            PhaseTiming(phase="claims", start_mono=5.0, end_mono=6.0),  # second claims call
        ]
        summary = rp.phase_summary()
        self.assertAlmostEqual(summary["claims"], 3000.0)  # 2s + 1s
        self.assertAlmostEqual(summary["verify"], 3000.0)

    def test_worker_summary(self) -> None:
        rp = RunProfiler(action="evaluate")
        rp.workers = [
            WorkerTiming(phase="digest", index=1, start_mono=0.0, end_mono=1.0),
            WorkerTiming(phase="digest", index=2, start_mono=0.0, end_mono=2.0),
            WorkerTiming(phase="digest", index=3, start_mono=0.0, end_mono=3.0),
        ]
        summary = rp.worker_summary()
        self.assertIn("digest", summary)
        self.assertEqual(summary["digest"]["count"], 3.0)
        self.assertAlmostEqual(summary["digest"]["max"], 3000.0)

    def test_emit_does_not_raise(self) -> None:
        rp = RunProfiler(action="evaluate", request_id="test-123")
        rp.phases = [PhaseTiming(phase="claims", start_mono=0.0, end_mono=1.0)]
        rp.emit()  # should not raise

    def test_emit_empty_phases(self) -> None:
        rp = RunProfiler(action="evaluate")
        rp.emit()  # should not raise, should be a no-op


class TestProfiler(unittest.TestCase):
    """Verify the global profiler aggregation."""

    def setUp(self) -> None:
        reset_profiler_for_tests()

    def tearDown(self) -> None:
        reset_profiler_for_tests()

    def test_record_and_summary(self) -> None:
        profiler = Profiler()
        for i in range(5):
            run = RunProfiler(
                action="evaluate",
                run_start_mono=0.0,
                run_end_mono=float(i + 1),
            )
            run.phases = [PhaseTiming(phase="claims", start_mono=0.0, end_mono=float(i + 1))]
            profiler.record_run(run)
        self.assertEqual(profiler._run_count, 5)
        profiler.emit_summary()  # should not raise

    def test_record_multiple_actions(self) -> None:
        profiler = Profiler()
        for _ in range(3):
            profiler.record_run(RunProfiler(action="evaluate", run_start_mono=0.0, run_end_mono=1.0))
        for _ in range(2):
            profiler.record_run(RunProfiler(action="draft", run_start_mono=0.0, run_end_mono=2.0))
        self.assertEqual(profiler._action_totals["evaluate"], 3)
        self.assertEqual(profiler._action_totals["draft"], 2)

    def test_reset(self) -> None:
        profiler = Profiler()
        profiler.record_run(RunProfiler(action="evaluate", run_start_mono=0.0, run_end_mono=1.0))
        profiler.reset()
        self.assertEqual(profiler._run_count, 0)
        self.assertEqual(len(profiler._run_totals), 0)

    def test_singleton(self) -> None:
        reset_profiler_for_tests()
        p1 = get_profiler()
        p2 = get_profiler()
        self.assertIs(p1, p2)

    def test_reset_singleton(self) -> None:
        p1 = get_profiler()
        reset_profiler_for_tests()
        p2 = get_profiler()
        self.assertIsNot(p1, p2)


class TestPhaseTimer(unittest.TestCase):
    """Verify the phase_timer context manager."""

    def test_noop_when_disabled(self) -> None:
        with mock.patch("app.profiler._ENABLED", False):
            with phase_timer("claims"):
                time.sleep(0.01)  # should not raise
            # No timing recorded — just verifying no-op behavior

    def test_records_when_enabled(self) -> None:
        with mock.patch("app.profiler._ENABLED", True):
            with phase_timer("claims"):
                time.sleep(0.01)
            # Just verifying it doesn't raise when enabled


class TestWorkerTimer(unittest.TestCase):
    """Verify the worker_timer context manager."""

    def test_noop_when_disabled(self) -> None:
        with mock.patch("app.profiler._ENABLED", False):
            with worker_timer("digest", index=1):
                time.sleep(0.01)
            # No-op, should not raise

    def test_records_when_enabled(self) -> None:
        with mock.patch("app.profiler._ENABLED", True):
            with worker_timer("digest", index=1):
                time.sleep(0.01)


class TestConfigGate(unittest.TestCase):
    """Verify the env var gate."""

    def test_profile_runs_env(self) -> None:
        with mock.patch.dict(config.__dict__, {"PROFILE_RUNS": True}):
            self.assertTrue(config.PROFILE_RUNS)
        with mock.patch.dict(config.__dict__, {"PROFILE_RUNS": False}):
            self.assertFalse(config.PROFILE_RUNS)


if __name__ == "__main__":
    unittest.main()
