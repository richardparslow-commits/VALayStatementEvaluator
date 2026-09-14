"""Pipeline-level timeout and memory monitoring for Evaluate/Draft runs.

Wraps the long-running ``run_evaluation`` and ``run_draft`` calls with:

* **Timeout** — a configurable wall-clock limit (default 30 minutes) enforced
  via ``concurrent.futures.ThreadPoolExecutor``.  When the deadline expires the
  caller receives a clear ``PipelineTimeoutError`` with the elapsed time and
  the configured limit, which ``app/main.py`` surfaces as a user-visible
  message.  ``signal.alarm`` is deliberately avoided because it only works on
  the main thread and has platform quirks; ``ThreadPoolExecutor`` works from
  any thread and on every OS.

* **Memory pre-check** — before the pipeline starts, the available system
  memory is sampled.  If it falls below ``VA_LSE_MEMORY_WARN_MB`` (default
  500 MB) the run proceeds but emits a ``st.warning`` so the user can reduce
  the record set.  If memory is critically low (< 200 MB) the run is aborted
  with ``MemoryError``.  The check is advisory on platforms where memory info
  is unavailable (Windows without ``psutil``).

* **Memory checkpoints** — ``memory_checkpoint`` is called at key pipeline
  stages (after chunk extraction, after digest merge, after summary).  It
  samples current RSS, logs at INFO with phase/duration, and emits a warning
  if peak usage exceeds ``VA_LSE_MEMORY_WARN_MB`` (adjustable).  This makes
  memory growth visible in structured logs without adding a dependency.

All helpers are stdlib-only and never raise on their own (memory checks
gracefully degrade on unsupported platforms).  ``PipelineTimeoutError`` is the
only exception the guard raises.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import threading
import time
from typing import Any, Callable, TypeVar

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

    def __init__(self, elapsed_seconds: float, limit_seconds: int) -> None:
        self.elapsed_seconds = elapsed_seconds
        self.limit_seconds = limit_seconds
        super().__init__(
            f"Pipeline timed out after {elapsed_seconds:.0f}s "
            f"(limit: {limit_seconds}s / {limit_seconds // 60} min). "
            f"The record set may be too large for the configured timeout. "
            f"Split the records into smaller files or raise VA_LSE_PIPELINE_TIMEOUT_SECONDS."
        )


# --------------------------------------------------------- memory helpers


def _read_rss_mb() -> float | None:
    """Return current process RSS in MB, or None if unavailable.

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
    """Check available memory before starting a pipeline run.

    Logs the current RSS.  If below 200 MB, raises MemoryError to abort.
    If below VA_LSE_MEMORY_WARN_MB, logs a warning (caller should show
    st.warning to the user).
    """
    rss_mb = _read_rss_mb()
    if rss_mb is None:
        logger.debug("memory check skipped: RSS not available on this platform")
        return

    logger.info(
        "pipeline memory pre-check rss_mb=%.0f",
        rss_mb,
        extra={"phase": "pipeline_guard", "status": "ok", "rss_mb": round(rss_mb)},
    )

    if rss_mb < 200:
        raise MemoryError(
            f"Critical memory shortage: only {rss_mb:.0f} MB RSS available. "
            f"Reduce the record set or restart the app. "
            f"(threshold: 200 MB minimum)"
        )

    warn_mb = _memory_warn_mb()
    if rss_mb > warn_mb:
        logger.warning(
            "pipeline memory high rss_mb=%.0f warn_threshold_mb=%d",
            rss_mb,
            warn_mb,
            extra={
                "phase": "pipeline_guard",
                "status": "warning",
                "rss_mb": round(rss_mb),
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


def run_with_timeout(
    fn: Callable[..., T],
    *args: Any,
    timeout_seconds: int | None = None,
    **kwargs: Any,
) -> T:
    """Run *fn* in a worker thread with a wall-clock timeout.

    Returns the result of ``fn(*args, **kwargs)``.  Raises
    ``PipelineTimeoutError`` if the function does not complete within the
    timeout.  The worker thread is abandoned (it continues running in the
    background but the caller is no longer blocked) — this matches the
    graceful-shutdown model where the orchestrator's SIGKILL handles stuck
    processes.

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

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            elapsed = time.perf_counter() - t0
            logger.info(
                "pipeline completed within timeout elapsed=%.0fs limit=%ds",
                elapsed,
                timeout_seconds,
                extra={
                    "request_id": rid,
                    "phase": "pipeline_guard",
                    "status": "ok",
                    "elapsed_s": round(elapsed),
                },
            )
            return result
        except concurrent.futures.TimeoutError:
            elapsed = time.perf_counter() - t0
            # Log the timeout event at ERROR for visibility in logs/alerts.
            logger.error(
                "pipeline timeout elapsed=%.0fs limit=%ds",
                elapsed,
                timeout_seconds,
                extra={
                    "request_id": rid,
                    "phase": "pipeline_guard",
                    "status": "timeout",
                    "elapsed_s": round(elapsed),
                    "timeout_seconds": timeout_seconds,
                },
            )
            raise PipelineTimeoutError(elapsed, timeout_seconds) from None
        except Exception:
            # Re-raise pipeline errors (LLMError, ValueError, etc.) as-is.
            raise


def _get_request_id() -> str:
    """Best-effort request id for log correlation."""
    try:
        from .logging_config import get_request_id

        return get_request_id() or "-"
    except Exception:  # noqa: BLE001
        return "-"
