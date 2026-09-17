"""Graceful shutdown handling for container orchestration.

Kubernetes (and similar orchestrators) send SIGTERM to a pod before
SIGKILL. Without a handler the process dies mid-request, aborting any
in-flight LLM calls and losing the user's work. This module gives
in-flight Evaluate/Draft runs up to VA_LSE_SHUTDOWN_GRACE_SECONDS (default
30 s) to finish cleanly, while making the container fall out of the load-
balancer pool immediately so no new work is routed to it.

Design
------
* A process-wide ``Event`` (``_shutdown_requested``) is the single source
  of truth — ``is_shutting_down()`` is checked by:
  - :mod:`app.health` — ``GET /ready`` flips to 503 while draining so the
    orchestrator stops routing to this instance.
  - :mod:`app.main` — new Evaluate/Draft runs are rejected with a
    user-visible warning once shutdown is in progress.
* An inflight counter (``_inflight``) is bumped by each Evaluate/Draft run
  via :func:`enter_run` / :func:`exit_run` (always paired, even on error).
* :func:`install_signal_handlers` wires SIGTERM + SIGINT to
  :func:`request_shutdown`, which sets the flag, logs at WARNING, waits up
  to ``grace_seconds`` for the counter to drain, then logs the outcome.
  After the grace window expires a final WARNING is emitted; the process is
  **not** force-killed here — the orchestrator's SIGKILL handles that —
  but the log line makes the forced exit visible for post-mortem.
* Per-LLM-call timeout is enforced in :mod:`app.llm` via the OpenAI
  ``timeout`` (``VA_LSE_LLM_CALL_TIMEOUT_SECONDS``, default 300 s / 5 min).
  A call that exceeds it raises ``LLMError`` with a user-visible message.

All public helpers are thread-safe and safe to call from tests.  The module
is stdlib-only and mypy-strict.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from types import FrameType

logger = logging.getLogger("app.shutdown")

# ------------------------------------------------------------------ state

_shutdown_requested = threading.Event()
_inflight_lock = threading.Lock()
_inflight: int = 0
_handlers_installed: bool = False
_install_lock = threading.Lock()


# --------------------------------------------------------------- public API


def is_shutting_down() -> bool:
    """Return True after SIGTERM/SIGINT has been received."""
    return _shutdown_requested.is_set()


def inflight_count() -> int:
    """Return the number of Evaluate/Draft runs currently in flight."""
    with _inflight_lock:
        return _inflight


def enter_run() -> bool:
    """Try to register a new Evaluate/Draft run.

    Returns False (and does NOT increment) when shutdown is already in
    progress — the caller should reject the run with a user-visible warning.
    Returns True when the slot was acquired (caller must call :func:`exit_run`
    in a finally block).
    """
    if _shutdown_requested.is_set():
        return False
    with _inflight_lock:
        # Re-check under lock to close the race where shutdown arrived
        # between the first check and acquiring the lock.
        if _shutdown_requested.is_set():
            return False
        global _inflight
        _inflight += 1
        return True


def exit_run() -> None:
    """Deregister a finished run (always call, even on error)."""
    global _inflight
    with _inflight_lock:
        if _inflight > 0:
            _inflight -= 1
        else:
            _inflight = 0


def _wait_for_drain(grace_seconds: float) -> bool:
    """Wait up to grace_seconds for inflight to reach 0. Return True if drained."""
    deadline = time.monotonic() + max(0.0, grace_seconds)
    while time.monotonic() < deadline:
        if inflight_count() == 0:
            return True
        time.sleep(0.2)
    return inflight_count() == 0


def request_shutdown(
    *,
    source: str = "signal",
    grace_seconds: float | None = None,
) -> bool:
    """Initiate graceful shutdown: set flag, wait for inflight to drain.

    Idempotent — the second call is a no-op (returns the current drained
    state). Always logs at WARNING so operators can trace pod lifecycle.

    Returns True when all inflight runs drained within the grace window,
    False when the window expired while work was still in flight.
    """
    if grace_seconds is None:
        try:
            from . import config as _cfg

            grace_seconds = float(getattr(_cfg, "SHUTDOWN_GRACE_SECONDS", 30))
        except Exception:  # noqa: BLE001
            grace_seconds = 30.0
    grace_seconds = max(0.0, float(grace_seconds))

    already = _shutdown_requested.is_set()
    _shutdown_requested.set()
    if already:
        # Handler re-entered (e.g. second SIGTERM) — do not restart the wait.
        return inflight_count() == 0

    count = inflight_count()
    logger.warning(
        "shutdown requested via %s: draining up to %.0fs (inflight=%d)",
        source,
        grace_seconds,
        count,
        extra={"phase": "shutdown", "status": "draining", "inflight": count},
    )

    drained = _wait_for_drain(grace_seconds)
    remaining = inflight_count()
    # Flush traces only after the drain: buffered spans belong to the runs that
    # just finished, and losing them would lose the trace of the very run an
    # operator is trying to explain. No-op when tracing is off.
    _flush_traces()
    if drained:
        logger.warning(
            "shutdown drained cleanly (inflight=0) after %s",
            source,
            extra={"phase": "shutdown", "status": "drained", "inflight": 0},
        )
    else:
        logger.warning(
            "shutdown grace period expired (%.0fs) with %d inflight run(s) still active — "
            "orchestrator SIGKILL will force-exit; users may see a failure and should retry",
            grace_seconds,
            remaining,
            extra={"phase": "shutdown", "status": "force_exit", "inflight": remaining},
        )
    return drained


def _flush_traces() -> None:
    """Flush buffered spans during shutdown. Never raises, never blocks for long."""
    try:
        from .tracing import shutdown_tracing

        shutdown_tracing()
    except Exception as exc:  # noqa: BLE001 - telemetry must not break shutdown
        logger.debug("could not flush traces during shutdown: %s", exc)


def _signal_handler(signum: int, _frame: FrameType | None) -> None:
    name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    # Do not block the signal handler thread for the full grace window —
    # spawn a daemon waiter so the interpreter can still deliver a second
    # signal while we drain. The waiter does the actual _wait_for_drain.
    threading.Thread(
        target=request_shutdown,
        kwargs={"source": name, "grace_seconds": None},
        name="va-lse-shutdown-drain",
        daemon=True,
    ).start()


def install_signal_handlers(*, grace_seconds: float | None = None) -> bool:
    """Install SIGTERM + SIGINT handlers for graceful drain (idempotent).

    Returns True when handlers were installed (or already installed), False
    when the platform does not support the signals (e.g. Windows without
    SIGTERM). Never raises — a failure to install is logged at WARNING.
    The *grace_seconds* argument is accepted for test injection but the
    runtime grace is always read from config at signal time so an env change
    does not require re-installation.
    """
    global _handlers_installed
    with _install_lock:
        if _handlers_installed:
            return True
        try:
            # Only the main thread may set signal handlers.
            if threading.current_thread() is not threading.main_thread():
                logger.debug(
                    "shutdown handlers not installed: not on main thread",
                    extra={"phase": "shutdown", "status": "skip"},
                )
                return False
            signal.signal(signal.SIGTERM, _signal_handler)
            # SIGINT covers Ctrl+C / docker stop without --time; keep it too.
            try:
                signal.signal(signal.SIGINT, _signal_handler)
            except (ValueError, OSError) as exc:  # pragma: no cover - platform-specific
                logger.debug("SIGINT handler not installed: %s", exc)
            _handlers_installed = True
            logger.info(
                "shutdown handlers installed (SIGTERM/SIGINT, grace from VA_LSE_SHUTDOWN_GRACE_SECONDS)",
                extra={"phase": "shutdown", "status": "armed"},
            )
            return True
        except (ValueError, OSError, AttributeError) as exc:
            logger.warning(
                "could not install shutdown handlers: %s",
                exc,
                extra={"phase": "shutdown", "status": "error", "error_class": type(exc).__name__},
            )
            return False


def reset_for_tests() -> None:
    """Reset shutdown state (tests only). Does not uninstall signal handlers."""
    global _inflight
    _shutdown_requested.clear()
    with _inflight_lock:
        _inflight = 0
