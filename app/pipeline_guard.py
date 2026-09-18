"""Pipeline-level timeout and memory monitoring for Evaluate/Draft runs.

Wraps the long-running ``run_evaluation`` and ``run_draft`` calls with:

* **Timeout** — a configurable wall-clock limit (default 30 minutes) enforced
  via ``concurrent.futures.ThreadPoolExecutor``.  When the deadline expires the
  caller receives a clear ``PipelineTimeoutError`` with the elapsed time and
  the configured limit, which ``app/main.py`` surfaces as a user-visible
  message.  ``signal.alarm`` is deliberately avoided because it only works on
  the main thread and has platform quirks; ``ThreadPoolExecutor`` works from
  any thread and on every OS.

* **Memory pre-check** — before the pipeline starts, the *available system
  memory* is sampled.  If it falls below ``VA_LSE_MEMORY_WARN_MB`` (default
  500 MB) the run proceeds but the operator is warned that the host is under
  memory pressure.  If available memory is critically low (< 200 MB) the run
  is aborted with ``MemoryError``.  The check is advisory on platforms where
  the figure is unavailable.

* **Memory checkpoints** — ``memory_checkpoint`` is called at key pipeline
  stages (after chunk extraction, after digest merge, after summary).  It
  samples current RSS, logs at INFO with phase/duration, and emits a warning
  if peak usage exceeds ``VA_LSE_MEMORY_WARN_MB`` (adjustable).  This makes
  memory growth visible in structured logs without adding a dependency.

Cancellation is cooperative: checkpoints stop further work, but cannot kill a
thread blocked inside a library call. A hard CPU/memory cutoff requires process
isolation. Memory checks gracefully degrade on unsupported platforms.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import logging
import os
import threading
import time
from typing import Any, Callable, TypeVar
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

logger = logging.getLogger("app.pipeline_guard")

T = TypeVar("T")

# ------------------------------------------------------------------ config


def _pipeline_timeout_seconds() -> int:
    """Read VA_LSE_PIPELINE_TIMEOUT_SECONDS (default 1800 = 30 min)."""
    try:
        from . import config as _cfg

        val = int(getattr(_cfg, "PIPELINE_TIMEOUT_SECONDS", 1800))
        return max(60, val)  # floor at 60s to avoid accidental misconfiguration
    except Exception:  # noqa: BLE001
        return 1800


def _memory_warn_mb() -> int:
    """Read VA_LSE_MEMORY_WARN_MB (default 500)."""
    try:
        from . import config as _cfg

        val = int(getattr(_cfg, "MEMORY_WARN_MB", 500))
        return max(50, val)
    except Exception:  # noqa: BLE001
        return 500


# --------------------------------------------------------------- exceptions


class PipelineTimeoutError(RuntimeError):
    """Raised when a pipeline run exceeds the configured timeout."""

    def __init__(self, elapsed_seconds: float, limit_seconds: float) -> None:
        self.elapsed_seconds = elapsed_seconds
        self.limit_seconds = limit_seconds
        super().__init__(
            f"Pipeline timed out after {elapsed_seconds:.0f}s "
            f"(limit: {limit_seconds}s / {limit_seconds // 60} min). "
            f"The record set may be too large for the configured timeout. "
            f"Split the records into smaller files or raise VA_LSE_PIPELINE_TIMEOUT_SECONDS."
        )


class PipelineCancelledError(BaseException):
    """Internal control flow; must bypass provider retries and best-effort fallbacks."""


@dataclass
class _PipelineRun:
    deadline: float
    cancelled: threading.Event = field(default_factory=threading.Event)


_pipeline_run: contextvars.ContextVar[_PipelineRun | None] = contextvars.ContextVar(
    "pipeline_run", default=None
)


def pipeline_remaining_seconds() -> float | None:
    """Remaining run budget, or None outside a guarded run; raises on cancellation."""
    run = _pipeline_run.get()
    if run is None:
        return None
    remaining = run.deadline - time.monotonic()
    if run.cancelled.is_set() or remaining <= 0:
        raise PipelineCancelledError("Pipeline deadline exceeded or run cancelled.")
    return remaining


def check_pipeline_cancelled() -> None:
    """Stop at a safe boundary before starting work or publishing progress."""
    pipeline_remaining_seconds()


def wait_with_cancellation(seconds: float) -> None:
    """Interrupt retry backoff when the run is cancelled."""
    remaining = pipeline_remaining_seconds()
    run = _pipeline_run.get()
    if run is None or remaining is None:
        time.sleep(seconds)
        return
    run.cancelled.wait(min(seconds, remaining))
    check_pipeline_cancelled()


def pipeline_as_completed(
    futures: Iterable[concurrent.futures.Future[T]],
) -> Iterator[concurrent.futures.Future[T]]:
    """Wait for child work without blocking cancellation on a stalled child."""
    pending = set(futures)
    while pending:
        check_pipeline_cancelled()
        done, pending = concurrent.futures.wait(
            pending, timeout=0.05, return_when=concurrent.futures.FIRST_COMPLETED
        )
        for future in done:
            check_pipeline_cancelled()
            yield future


# --------------------------------------------------------- memory helpers


def _read_available_memory_mb() -> float | None:
    """Return available system memory in MB, or None if unavailable.

    The pre-run guard answers "can this machine absorb a long pipeline?" —
    that is *available* memory, not the process's own RSS. A small/idle
    process is healthy, and aborting because RSS is low (the old behavior)
    falsely rejected runs on quiet machines while doing nothing when the
    system was actually exhausted.

    - Linux: ``/proc/meminfo`` MemAvailable (exact; no dependencies).
    - macOS: ``vm_stat`` page counts via stdlib subprocess (free + inactive
      + speculative pages — Apple keeps caches warm, so this is the honest
      "truly free" figure). Purgeable/swap headroom is ignored on purpose:
      this is a conservative gate, not a swappiness model.
    - Everything else: ``None`` → the guard degrades to advisory-only.
    """
    # Linux: MemAvailable from /proc/meminfo.
    try:
        with open("/proc/meminfo", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) / 1024.0  # kB → MB
    except (OSError, ValueError):
        pass

    # macOS: vm_stat free+inactive+speculative pages.
    try:
        import subprocess

        proc = subprocess.run(  # noqa: S603, S607 - fixed argv, no shell
            ["vm_stat"], capture_output=True, text=True, timeout=2
        )
        if proc.returncode == 0:
            page_size = 4096
            free = inactive = speculative = 0
            for line in proc.stdout.splitlines():
                if line.startswith("page size of"):
                    try:
                        page_size = int(line.split()[-2])
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("Pages free:"):
                    free = int(line.split()[-1].rstrip("."))
                elif line.startswith("Pages inactive:"):
                    inactive = int(line.split()[-1].rstrip("."))
                elif line.startswith("Pages speculative:"):
                    speculative = int(line.split()[-1].rstrip("."))
            return (free + inactive + speculative) * page_size / (1024 * 1024)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass

    return None


def _read_rss_mb() -> float | None:
    """Return current process RSS in MB, or None if unavailable.

    Kept for ``memory_checkpoint`` observability: checkpoint logging wants to
    see how big *the app itself* is getting as records are digested.
    Uses /proc/self/status on Linux (no dependencies), resource.getrusage on
    macOS (peak RSS, not current — less useful but better than nothing), and
    falls back to None on platforms where neither is available.
    """
    # Linux: /proc/self/status VmRSS line
    try:
        with open("/proc/self/status", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # "VmRSS:    123456 kB"
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) / 1024.0  # kB → MB
    except (OSError, ValueError):
        pass

    # macOS/Unix: resource.getrusage (peak RSS, not current)
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in bytes on Linux, kilobytes on macOS/BSD
        maxrss = usage.ru_maxrss
        if maxrss > 1_000_000:  # looks like bytes (Linux)
            return maxrss / (1024 * 1024)
        return maxrss / 1024.0  # kilobytes (macOS)
    except (ImportError, OSError):
        pass

    return None


def check_memory_before_run() -> None:
    """Check *available system memory* before starting a pipeline run.

    Logs the current availability. If below 200 MB, raises MemoryError to
    abort. If above VA_LSE_MEMORY_WARN_MB, logs a warning that the host is
    already under memory pressure (the run proceeds). Advisory no-op on
    platforms where the figure is unavailable.

    Historically this checked the *process's own* RSS, which inverted the
    intent: an idle app (low RSS) was "critical" while a bloated one on an
    exhausted host sailed through. That false abort killed Evaluate/Draft
    runs in fresh AppTest/test processes (~100 MB RSS) and lightweight
    deployments.
    """
    avail_mb = _read_available_memory_mb()
    if avail_mb is None:
        logger.debug("memory check skipped: available-memory figure unavailable on this platform")
        return

    logger.info(
        "pipeline memory pre-check available_mb=%.0f",
        avail_mb,
        extra={"phase": "pipeline_guard", "status": "ok", "available_mb": round(avail_mb)},
    )

    if avail_mb < 200:
        raise MemoryError(
            f"Critical memory shortage: only {avail_mb:.0f} MB of system memory available. "
            f"Free memory (close other apps / reduce other workloads) or reduce the record set. "
            f"(threshold: 200 MB minimum available)"
        )

    warn_mb = _memory_warn_mb()
    if avail_mb < warn_mb:
        logger.warning(
            "pipeline memory low available_mb=%.0f warn_threshold_mb=%d",
            avail_mb,
            warn_mb,
            extra={
                "phase": "pipeline_guard",
                "status": "warning",
                "available_mb": round(avail_mb),
                "warn_threshold_mb": warn_mb,
            },
        )


def memory_checkpoint(phase: str) -> None:
    """Log current RSS at a pipeline checkpoint.

    Called after major stages (chunk extraction, digest merge, summary) so
    operators can see memory growth in structured logs.  Warns when peak
    exceeds VA_LSE_MEMORY_WARN_MB.
    """
    rss_mb = _read_rss_mb()
    if rss_mb is None:
        return

    warn_mb = _memory_warn_mb()
    level = logging.WARNING if rss_mb > warn_mb else logging.INFO
    logger.log(
        level,
        "memory checkpoint phase=%s rss_mb=%.0f",
        phase,
        rss_mb,
        extra={
            "phase": "pipeline_guard",
            "checkpoint": phase,
            "status": "high" if rss_mb > warn_mb else "ok",
            "rss_mb": round(rss_mb),
        },
    )


# -------------------------------------------------------- timeout wrapper


def _propagate_streamlit_ctx(fn: Callable[..., T], *args: Any, **kwargs: Any) -> Callable[..., T]:
    """Wrap *fn* so the pool worker inherits the caller's Streamlit context.

    The pipeline calls ``st.progress``/``st.empty`` from inside the worker
    thread (progress callbacks). Without the caller's ``ScriptRunContext``
    those calls raise ``NoSessionContext`` and kill an otherwise healthy run —
    under ``AppTest`` and in real deployments alike (regression introduced
    when the timeout wrapper moved pipelines off the script thread).

    The context is captured in the *calling* thread (the wrapper is built
    there) and self-attached inside the worker via the documented
    ``add_script_run_ctx`` pattern. Best-effort: no-op when no context
    exists (bare scripts/tests).

    The worker also runs inside a copy of the caller's ``contextvars``, because
    those do not cross threads by themselves. Everything the pipeline attaches to
    them — the request id (``app/logging_config``), the run profiler
    (``app/profiler``), and the active trace span (``app/tracing``) — would
    otherwise silently vanish at this boundary: logs from the pipeline would lose
    their correlation id, and (worse) every run would start a second, unrelated
    trace inside this thread.
    """
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import (
            get_script_run_ctx,
        )

        ctx = get_script_run_ctx(suppress_warning=True)
    except Exception:  # noqa: BLE001 - context propagation is best-effort
        ctx = None
    ctx_vars = contextvars.copy_context()

    def _wrapped() -> T:
        def _in_context() -> T:
            if ctx is not None:
                try:
                    from streamlit.runtime.scriptrunner_utils.script_run_context import (
                        add_script_run_ctx,
                    )

                    add_script_run_ctx(ctx=ctx)
                except Exception:  # noqa: BLE001 - context propagation is best-effort
                    pass
            return fn(*args, **kwargs)

        return ctx_vars.run(_in_context)

    return _wrapped


def run_with_timeout(
    fn: Callable[..., T],
    *args: Any,
    timeout_seconds: float | None = None,
    **kwargs: Any,
) -> T:
    """Run *fn* in a worker thread with a wall-clock timeout.

    Returns the result of ``fn(*args, **kwargs)``.  Raises
    ``PipelineTimeoutError`` if the function does not complete within the
    timeout. The caller never waits for executor shutdown on timeout. A per-run
    cancellation signal stops further phases, retries, and progress updates.
    Already-running library calls cannot be forcibly stopped by a thread guard;
    they must return or hit their own timeout before their thread exits.

    The worker inherits the caller's Streamlit ``ScriptRunContext`` and
    ``contextvars`` (see :func:`_propagate_streamlit_ctx`) so pipeline progress
    callbacks using ``st.*`` keep working from the pool thread and the run's
    request id, profiler, and trace span stay attached to it.

    The timeout defaults to ``VA_LSE_PIPELINE_TIMEOUT_SECONDS`` (30 min).
    """
    if timeout_seconds is None:
        timeout_seconds = _pipeline_timeout_seconds()

    t0 = time.perf_counter()

    # Log the timeout budget at the start so operators can see it.
    rid = _get_request_id()
    logger.info(
        "pipeline timeout armed seconds=%d phase=%s",
        timeout_seconds,
        getattr(fn, "__name__", "unknown"),
        extra={
            "request_id": rid,
            "phase": "pipeline_guard",
            "status": "armed",
            "timeout_seconds": timeout_seconds,
        },
    )

    run = _PipelineRun(deadline=time.monotonic() + timeout_seconds)

    def checked() -> T:
        check_pipeline_cancelled()
        result = fn(*args, **kwargs)
        check_pipeline_cancelled()
        return result

    token = _pipeline_run.set(run)
    try:
        wrapped = _propagate_streamlit_ctx(checked)
    finally:
        _pipeline_run.reset(token)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(wrapped)
        done, _ = concurrent.futures.wait(
            [future], timeout=max(0.0, run.deadline - time.monotonic())
        )
        if not done:
            raise PipelineCancelledError()
        # A TimeoutError raised by fn is its own error, not a wait timeout.
        result = future.result()
        elapsed = time.perf_counter() - t0
        logger.info(
            "pipeline completed within timeout elapsed=%.0fs limit=%ds",
            elapsed, timeout_seconds,
            extra={"request_id": rid, "phase": "pipeline_guard", "status": "ok",
                   "elapsed_s": round(elapsed)},
        )
        return result
    except PipelineCancelledError:
        run.cancelled.set()
        elapsed = time.perf_counter() - t0
        logger.error(
            "pipeline timeout elapsed=%.0fs limit=%ds",
            elapsed, timeout_seconds,
            extra={"request_id": rid, "phase": "pipeline_guard", "status": "timeout",
                   "elapsed_s": round(elapsed), "timeout_seconds": timeout_seconds},
        )
        raise PipelineTimeoutError(elapsed, timeout_seconds) from None
    finally:
        run.cancelled.set()
        pool.shutdown(wait=False, cancel_futures=True)


def _get_request_id() -> str:
    """Best-effort request id for log correlation."""
    try:
        from .logging_config import get_request_id

        return get_request_id() or "-"
    except Exception:  # noqa: BLE001
        return "-"
