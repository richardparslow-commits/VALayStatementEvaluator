"""Lightweight production profiler for Evaluate/Draft pipeline phases.

Enabled when ``VA_LSE_PROFILE_RUNS=1``.  Collects wall-clock timing per
phase, per-worker durations during parallel record digestion, and
aggregates p50/p95/p99 latency across multiple runs.  All output goes to
the structured logger (``phase=profiler``) — no external dependencies.

Usage::

    from .profiler import phase_timer, worker_timer, get_profiler

    # In a pipeline phase:
    with phase_timer("claims"):
        claims_data = llm.chat_json(...)

    # Around a parallel worker:
    with worker_timer("digest", chunk_index=idx):
        data = digest_chunk(chunk)

    # After the run — emit summary:
    get_profiler().emit_summary()
"""
from __future__ import annotations

import contextvars
import logging
import os
import statistics
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

logger = logging.getLogger("app.profiler")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ENABLED: bool = os.getenv("VA_LSE_PROFILE_RUNS", "").strip() in ("1", "true", "True", "yes")
_current_run_var: contextvars.ContextVar["RunProfiler | None"] = contextvars.ContextVar(
    "va_lse_run_profiler",
    default=None,
)


def is_enabled() -> bool:
    """Return True when profiling is active (VA_LSE_PROFILE_RUNS=1)."""
    try:
        from . import config as _config

        return bool(getattr(_config, "PROFILE_RUNS", False) or _ENABLED)
    except Exception:  # noqa: BLE001
        return _ENABLED


# ---------------------------------------------------------------------------
# Per-run phase timing
# ---------------------------------------------------------------------------

@dataclass
class PhaseTiming:
    """Timing data for a single phase within one run."""

    phase: str
    start_mono: float
    end_mono: float = 0.0

    @property
    def duration_ms(self) -> float:
        if self.end_mono == 0.0:
            return 0.0
        return (self.end_mono - self.start_mono) * 1000


@dataclass
class WorkerTiming:
    """Timing data for a single worker invocation."""

    phase: str
    index: int
    start_mono: float
    end_mono: float = 0.0

    @property
    def duration_ms(self) -> float:
        if self.end_mono == 0.0:
            return 0.0
        return (self.end_mono - self.start_mono) * 1000


# ---------------------------------------------------------------------------
# Run profiler (single run)
# ---------------------------------------------------------------------------

@dataclass
class RunProfiler:
    """Accumulates timing data for a single Evaluate or Draft run."""

    action: str  # "evaluate" or "draft"
    request_id: str = "-"
    phases: list[PhaseTiming] = field(default_factory=list)
    workers: list[WorkerTiming] = field(default_factory=list)
    run_start_mono: float = 0.0
    run_end_mono: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def total_duration_ms(self) -> float:
        if self.run_end_mono == 0.0 or self.run_start_mono == 0.0:
            return 0.0
        return (self.run_end_mono - self.run_start_mono) * 1000

    def phase_summary(self) -> dict[str, float]:
        """Return {phase_name: total_duration_ms}."""
        by_phase: dict[str, float] = defaultdict(float)
        for pt in self.phases:
            by_phase[pt.phase] += pt.duration_ms
        return dict(by_phase)

    def worker_summary(self) -> dict[str, dict[str, float]]:
        """Return {phase: {count, total_ms, p50, p95, p99, max}}."""
        by_phase: dict[str, list[float]] = defaultdict(list)
        for wt in self.workers:
            by_phase[wt.phase].append(wt.duration_ms)
        result: dict[str, dict[str, float]] = {}
        for phase, durations in by_phase.items():
            result[phase] = _percentile_stats(durations)
        return result

    def add_phase_timing(self, phase: str, start_mono: float, end_mono: float) -> None:
        with self._lock:
            self.phases.append(
                PhaseTiming(phase=phase, start_mono=start_mono, end_mono=end_mono)
            )

    def add_worker_timing(self, phase: str, index: int, start_mono: float, end_mono: float) -> None:
        with self._lock:
            self.workers.append(
                WorkerTiming(
                    phase=phase,
                    index=index,
                    start_mono=start_mono,
                    end_mono=end_mono,
                )
            )

    def emit(self) -> None:
        """Log the full profiling summary at INFO level."""
        if not self.phases:
            return
        phase_ms = self.phase_summary()
        worker_ms = self.worker_summary()
        total = self.total_duration_ms

        # Build a concise summary line.
        phase_parts = [f"{p}={v:.0f}ms" for p, v in sorted(phase_ms.items())]
        worker_parts: list[str] = []
        for p, stats in sorted(worker_ms.items()):
            worker_parts.append(
                f"{p}:n={stats['count']:.0f} p50={stats['p50']:.0f}ms "
                f"p95={stats['p95']:.0f}ms max={stats['max']:.0f}ms"
            )

        summary = (
            f"profile {self.action} total={total:.0f}ms "
            + " ".join(phase_parts)
        )
        if worker_parts:
            summary += " | workers: " + "; ".join(worker_parts)

        logger.info(
            summary,
            extra={
                "request_id": self.request_id,
                "phase": "profiler",
                "status": "ok",
                "action": self.action,
                "total_ms": round(total),
                "phase_breakdown": phase_ms,
                "worker_breakdown": worker_ms,
            },
        )


# ---------------------------------------------------------------------------
# Global profiler (aggregates across multiple runs)
# ---------------------------------------------------------------------------

class Profiler:
    """Process-global profiler that accumulates timing across runs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._phase_durations: dict[str, list[float]] = defaultdict(list)
        self._worker_durations: dict[str, list[float]] = defaultdict(list)
        self._run_count: int = 0
        self._run_totals: list[float] = []
        self._action_totals: dict[str, int] = defaultdict(int)

    def record_run(self, run: RunProfiler) -> None:
        """Record a completed run's timings."""
        with self._lock:
            self._run_count += 1
            self._run_totals.append(run.total_duration_ms)
            self._action_totals[run.action] += 1
            for pt in run.phases:
                self._phase_durations[pt.phase].append(pt.duration_ms)
            for wt in run.workers:
                self._worker_durations[wt.phase].append(wt.duration_ms)

    def emit_summary(self) -> None:
        """Log aggregated p50/p95/p99 across all recorded runs."""
        with self._lock:
            if self._run_count == 0:
                return

            run_stats = _percentile_stats(self._run_totals)
            parts = [
                f"runs={self._run_count}",
                f"total p50={run_stats['p50']:.0f}ms p95={run_stats['p95']:.0f}ms "
                f"p99={run_stats['p99']:.0f}ms max={run_stats['max']:.0f}ms",
            ]
            action_parts = [f"{a}={n}" for a, n in sorted(self._action_totals.items())]
            if action_parts:
                parts.append(" ".join(action_parts))

            phase_parts: list[str] = []
            for phase in sorted(self._phase_durations):
                stats = _percentile_stats(self._phase_durations[phase])
                phase_parts.append(
                    f"{phase}: p50={stats['p50']:.0f}ms p95={stats['p95']:.0f}ms "
                    f"n={stats['count']:.0f}"
                )
            if phase_parts:
                parts.append("phases: " + "; ".join(phase_parts))

            worker_parts: list[str] = []
            for phase in sorted(self._worker_durations):
                stats = _percentile_stats(self._worker_durations[phase])
                worker_parts.append(
                    f"{phase}: n={stats['count']:.0f} p50={stats['p50']:.0f}ms "
                    f"p95={stats['p95']:.0f}ms max={stats['max']:.0f}ms"
                )
            if worker_parts:
                parts.append("workers: " + "; ".join(worker_parts))

            summary = "profile aggregate " + " | ".join(parts)
            logger.info(
                summary,
                extra={
                    "phase": "profiler",
                    "status": "aggregate",
                    "run_count": self._run_count,
                    "run_stats": run_stats,
                    "phase_durations": {
                        p: _percentile_stats(d) for p, d in self._phase_durations.items()
                    },
                    "worker_durations": {
                        p: _percentile_stats(d) for p, d in self._worker_durations.items()
                    },
                },
            )

    def reset(self) -> None:
        """Clear all accumulated data."""
        with self._lock:
            self._phase_durations.clear()
            self._worker_durations.clear()
            self._run_count = 0
            self._run_totals.clear()
            self._action_totals.clear()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_profiler: Profiler | None = None
_lock = threading.Lock()


def get_profiler() -> Profiler:
    """Return (and lazily create) the process-global profiler."""
    global _profiler  # noqa: PLW0603
    if _profiler is not None:
        return _profiler
    with _lock:
        if _profiler is not None:
            return _profiler
        _profiler = Profiler()
        return _profiler


def reset_profiler_for_tests() -> None:
    """Reset the singleton so tests get a fresh profiler."""
    global _profiler  # noqa: PLW0603
    with _lock:
        _profiler = None
    _current_run_var.set(None)


def get_current_run_profiler() -> RunProfiler | None:
    """Return the currently bound run profiler, if any."""
    return _current_run_var.get()


@contextmanager
def bind_run_profiler(run: RunProfiler | None) -> Generator[None, None, None]:
    """Bind a run profiler to the current context for nested timers."""
    token = _current_run_var.set(run)
    try:
        yield
    finally:
        _current_run_var.reset(token)


# ---------------------------------------------------------------------------
# Context managers for phase and worker timing
# ---------------------------------------------------------------------------

@contextmanager
def phase_timer(phase: str) -> Generator[None, None, None]:
    """Time a pipeline phase.  Only active when ``VA_LSE_PROFILE_RUNS=1``."""
    if not is_enabled():
        yield
        return
    start = time.monotonic()
    yield
    end = time.monotonic()
    run = get_current_run_profiler()
    if run is not None:
        run.add_phase_timing(phase, start, end)
    try:
        from .logging_config import get_request_id
        rid = get_request_id() or "-"
    except Exception:  # noqa: BLE001
        rid = "-"
    logger.debug(
        "phase %s duration_ms=%.0f",
        phase,
        (end - start) * 1000,
        extra={
            "request_id": rid,
            "phase": f"profiler:{phase}",
            "status": "ok",
            "duration_ms": round((end - start) * 1000),
        },
    )


@contextmanager
def worker_timer(phase: str, *, index: int = 0) -> Generator[None, None, None]:
    """Time a single worker invocation (e.g. one chunk digest)."""
    if not is_enabled():
        yield
        return
    start = time.monotonic()
    yield
    end = time.monotonic()
    run = get_current_run_profiler()
    if run is not None:
        run.add_worker_timing(phase, index, start, end)
    try:
        from .logging_config import get_request_id
        rid = get_request_id() or "-"
    except Exception:  # noqa: BLE001
        rid = "-"
    logger.debug(
        "worker %s[%d] duration_ms=%.0f",
        phase,
        index,
        (end - start) * 1000,
        extra={
            "request_id": rid,
            "phase": f"profiler:worker:{phase}",
            "status": "ok",
            "duration_ms": round((end - start) * 1000),
            "worker_index": index,
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile_stats(durations: list[float]) -> dict[str, float]:
    """Compute p50/p95/p99/max/count for a list of durations."""
    if not durations:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    sorted_d = sorted(durations)
    n = len(sorted_d)
    return {
        "count": float(n),
        "p50": sorted_d[int(n * 0.5)] if n > 1 else sorted_d[0],
        "p95": sorted_d[min(int(n * 0.95), n - 1)],
        "p99": sorted_d[min(int(n * 0.99), n - 1)],
        "max": sorted_d[-1],
        "mean": statistics.mean(durations),
    }
