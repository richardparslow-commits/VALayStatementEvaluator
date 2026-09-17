"""Circuit breaker and concurrency limiter for LLM calls.

Stdlib-only implementation that mirrors the behaviour the spec asks for
(``pybreaker`` style) without adding a new dependency.

* **Circuit breaker** — per-process, thread-safe. Opens after
  ``VA_LSE_CB_FAILURE_THRESHOLD`` (default 3) consecutive *logical* LLM call
  failures (a call that exhausts its own retries counts as one failure). While
  OPEN every call fails fast (``CircuitBreakerOpenError``) in <2 s without
  touching the network — this stops 100 concurrent users from hammering a
  degraded endpoint. After ``VA_LSE_CB_RECOVERY_SECONDS`` (default 60 s) the
  breaker moves to HALF_OPEN and lets one probe through; a success closes it,
  a failure re-opens it. All state changes log at WARNING.

* **Concurrency limiter** — global semaphore with an in-memory queue.
  ``VA_LSE_MAX_CONCURRENT_LLM_CALLS`` (default 20) caps the number of LLM
  calls executing at once. When the cap is hit, callers are queued up to
  ``VA_LSE_LLM_QUEUE_MAX_DEPTH`` (default 50); beyond that the call is
  rejected immediately with ``QueueFullError``. Waiting callers block up to
  ``VA_LSE_LLM_QUEUE_TIMEOUT_SECONDS`` (default 30 s) before timing out.

The breaker is **per endpoint** (``get_llm_breaker``), because a breaker is a
statement about one endpoint's health: a shared one would let a healthy fallback's
successes close the primary's breaker and hide an ongoing outage, and let the
primary's failures refuse calls the fallback could have served. The concurrency
limiter stays global — it bounds *this process's* egress, not one provider's.

Each breaker also tracks how long its endpoint has been *continuously*
unhealthy (``unhealthy_for_seconds``), which is the failover trigger. That is a
different clock from the probe countdown: ``_opened_at`` is reset by every failed
probe, so it never ages past one ``recovery_timeout`` while traffic keeps probing.
Test helpers ``reset_llm_breaker`` / ``reset_llm_limiter`` allow tests to
reconfigure them without restarting the process.
"""

from __future__ import annotations

import logging
import threading
import time
from types import TracebackType
from typing import Literal

logger = logging.getLogger("app.circuit_breaker")

CircuitState = Literal["CLOSED", "OPEN", "HALF_OPEN"]


class CircuitBreakerError(RuntimeError):
    """Base for circuit-breaker rejections."""


class CircuitBreakerOpenError(CircuitBreakerError):
    """Raised when the breaker is OPEN and the call is rejected fast."""


class ConcurrencyLimitError(RuntimeError):
    """Base for concurrency/queue rejections."""


class QueueFullError(ConcurrencyLimitError):
    """Raised when the concurrency queue is at capacity."""


class CircuitBreaker:
    """Thread-safe consecutive-failure circuit breaker.

    Counts only *logical* failures (one per ``LLMClient.chat`` that exhausts
    its retries). Not per-retry attempt.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        name: str = "llm",
    ) -> None:
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_timeout = max(0.0, float(recovery_timeout))
        self.name = name
        self._state: CircuitState = "CLOSED"
        self._failure_count: int = 0
        self._opened_at: float | None = None
        self._unhealthy_since: float | None = None
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- state

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    def unhealthy_for_seconds(self) -> float | None:
        """How long this endpoint has been continuously unhealthy, else ``None``.

        Set when the breaker first opens and cleared only by a genuine recovery
        (a transition back to CLOSED) — *not* by a failed probe. This is the
        failover trigger, and it is deliberately not ``_opened_at``: a failed
        probe resets that one, so with a 60 s recovery timeout and continuous
        traffic it never ages past 60 s, and a failover rule built on it would
        never fire under exactly the load it exists to handle.
        """
        with self._lock:
            if self._unhealthy_since is None:
                return None
            return max(0.0, time.monotonic() - self._unhealthy_since)

    def _transition(self, new_state: CircuitState, *, reason: str) -> None:
        old = self._state
        if old == new_state:
            return
        self._state = new_state
        if new_state == "OPEN":
            # First open wins: HALF_OPEN -> OPEN (a failed probe) must not restart
            # the clock, or a long outage would look like it began a minute ago.
            if self._unhealthy_since is None:
                self._unhealthy_since = time.monotonic()
        elif new_state == "CLOSED":
            self._unhealthy_since = None
        # Imported here rather than at module scope: the breaker is created during
        # `get_llm_breaker()` from inside a call path, and a metrics import that
        # pulled in config (or anything else) at module load would add a startup
        # dependency to a primitive that must stay stdlib-only.
        try:
            from .metrics import observe_breaker_transition

            observe_breaker_transition(self.name, old, new_state)
        except Exception:  # noqa: BLE001 - instrumentation is best-effort by design
            pass
        logger.warning(
            "circuit breaker '%s' %s -> %s (%s)",
            self.name,
            old,
            new_state,
            reason,
            extra={
                "phase": "circuit_breaker",
                "status": new_state.lower(),
                "breaker": self.name,
                "old_state": old,
                "new_state": new_state,
            },
        )

    # ----------------------------------------------------------------- gating

    def allow_request(self) -> bool:
        """Return True if a call may proceed, False if it must fail fast.

        Handles the OPEN -> HALF_OPEN time-based transition.
        """
        with self._lock:
            if self._state == "CLOSED":
                return True
            if self._state == "HALF_OPEN":
                return True
            assert self._state == "OPEN"
            if self._opened_at is None:
                return False
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self.recovery_timeout:
                self._transition("HALF_OPEN", reason=f"recovery timeout {self.recovery_timeout}s elapsed")
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            if self._state == "HALF_OPEN":
                self._failure_count = 0
                self._opened_at = None
                self._transition("CLOSED", reason="probe succeeded")
            elif self._state == "CLOSED":
                if self._failure_count != 0:
                    self._failure_count = 0
            else:  # OPEN — should not happen (calls are blocked), but handle
                self._failure_count = 0
                self._opened_at = None
                self._transition("CLOSED", reason="success while open (unexpected)")

    def record_failure(self) -> None:
        with self._lock:
            if self._state == "HALF_OPEN":
                self._opened_at = time.monotonic()
                self._transition("OPEN", reason="probe failed")
                self._failure_count = self.failure_threshold
                return
            if self._state == "OPEN":
                self._opened_at = time.monotonic()
                return
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._opened_at = time.monotonic()
                self._transition(
                    "OPEN",
                    reason=f"{self._failure_count} consecutive failures >= threshold {self.failure_threshold}",
                )

    # ---------------------------------------------------------------- helpers

    def check_or_raise(self) -> None:
        """Fail fast with CircuitBreakerOpenError if the breaker is OPEN."""
        if not self.allow_request():
            remaining = 0.0
            with self._lock:
                if self._opened_at is not None:
                    remaining = max(0.0, self.recovery_timeout - (time.monotonic() - self._opened_at))
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN — LLM endpoint temporarily unavailable. "
                f"Failing fast to protect the endpoint (retry in {remaining:.0f}s). "
                f"After {self.failure_threshold} consecutive failures the breaker opened for "
                f"{self.recovery_timeout:.0f}s."
            )

    def reset(self) -> None:
        """Force back to CLOSED (used in tests)."""
        with self._lock:
            self._state = "CLOSED"
            self._failure_count = 0
            self._opened_at = None
            self._unhealthy_since = None


# ------------------------------------------------------------------ limiter


class ConcurrencyLimiter:
    """Global semaphore + bounded queue for LLM calls.

    ``max_concurrent`` caps the number of calls executing at once.
    ``max_queue_depth`` caps the number of callers waiting for a slot.
    ``queue_timeout`` caps how long a queued caller will block.
    """

    def __init__(
        self,
        max_concurrent: int = 20,
        max_queue_depth: int = 50,
        queue_timeout: float = 30.0,
        name: str = "llm",
    ) -> None:
        self.max_concurrent = max(1, int(max_concurrent))
        self.max_queue_depth = max(0, int(max_queue_depth))
        self.queue_timeout = max(0.0, float(queue_timeout))
        self.name = name
        self._active: int = 0
        self._waiting: int = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    # --------------------------------------------------------------- stats

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    @property
    def waiting(self) -> int:
        with self._lock:
            return self._waiting

    # -------------------------------------------------------------- acquire

    def acquire(self, *, timeout: float | None = None) -> None:
        """Acquire a concurrency slot, queuing if necessary.

        Raises QueueFullError if the waiting queue is at capacity, or if the
        wait times out. Always call ``release`` after a successful acquire.
        """
        effective_timeout = self.queue_timeout if timeout is None else max(0.0, float(timeout))
        with self._cond:
            if self._active < self.max_concurrent:
                self._active += 1
                return

            if self._waiting >= self.max_queue_depth:
                logger.warning(
                    "concurrency limiter '%s' queue full (waiting=%d max=%d active=%d) — rejecting call",
                    self.name,
                    self._waiting,
                    self.max_queue_depth,
                    self._active,
                    extra={
                        "phase": "concurrency",
                        "status": "queue_full",
                        "limiter": self.name,
                        "active": self._active,
                        "waiting": self._waiting,
                    },
                )
                raise QueueFullError(
                    f"Too many concurrent LLM requests — queue full "
                    f"({self._waiting}/{self.max_queue_depth} waiting, "
                    f"{self._active}/{self.max_concurrent} active). "
                    f"Try again shortly or raise VA_LSE_MAX_CONCURRENT_LLM_CALLS / "
                    f"VA_LSE_LLM_QUEUE_MAX_DEPTH."
                )

            self._waiting += 1
            acquired = False
            try:
                deadline = time.monotonic() + effective_timeout
                while self._active >= self.max_concurrent:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        logger.warning(
                            "concurrency limiter '%s' queue timeout after %.1fs (waiting=%d active=%d)",
                            self.name,
                            effective_timeout,
                            self._waiting,
                            self._active,
                            extra={
                                "phase": "concurrency",
                                "status": "queue_timeout",
                                "limiter": self.name,
                                "active": self._active,
                                "waiting": self._waiting,
                            },
                        )
                        raise QueueFullError(
                            f"Timed out waiting for an LLM concurrency slot after {effective_timeout:.0f}s "
                            f"({self._waiting} waiting, {self._active}/{self.max_concurrent} active). "
                            f"The endpoint may be saturated."
                        )
                    self._cond.wait(timeout=remaining)
                self._waiting -= 1
                self._active += 1
                acquired = True
            finally:
                if not acquired:
                    if self._waiting > 0:
                        self._waiting -= 1
                    if self._waiting < 0:
                        self._waiting = 0

    def release(self) -> None:
        with self._cond:
            if self._active > 0:
                self._active -= 1
            else:
                self._active = 0
            self._cond.notify()

    # ------------------------------------------------------------ context

    def __enter__(self) -> ConcurrencyLimiter:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        self.release()
        return False

    def reset(self) -> None:
        """Reset counts (used in tests)."""
        with self._cond:
            self._active = 0
            self._waiting = 0
            self._cond.notify_all()


# ---------------------------------------------------------- singletons

# Breaker names. The name is also the `breaker` label on the circuit-breaker
# metrics, so an operator can tell a primary outage from a fallback one.
LLM_BREAKER_NAME = "llm"
LLM_FALLBACK_BREAKER_NAME = "llm-fallback"

_llm_breakers: dict[str, CircuitBreaker] = {}
_llm_breaker_lock = threading.RLock()

_llm_limiter: ConcurrencyLimiter | None = None
_llm_limiter_lock = threading.RLock()


def _breaker_config() -> tuple[int, float]:
    """Threshold/recovery from config, with safe fallbacks if config is absent."""
    try:
        from . import config as _cfg  # pylint: disable=import-outside-toplevel

        return (
            int(getattr(_cfg, "LLM_CB_FAILURE_THRESHOLD", 3)),
            float(getattr(_cfg, "LLM_CB_RECOVERY_SECONDS", 60)),
        )
    except Exception:  # noqa: BLE001
        return 3, 60.0


def get_llm_breaker(name: str = LLM_BREAKER_NAME) -> CircuitBreaker:
    """Return the process-wide breaker for one LLM endpoint (lazy, env-configured).

    Keyed by endpoint name rather than one global instance — see the module
    docstring for why a shared breaker is wrong once there is a fallback.
    """
    existing = _llm_breakers.get(name)
    if existing is not None:
        return existing
    with _llm_breaker_lock:
        existing = _llm_breakers.get(name)
        if existing is not None:
            return existing
        threshold, recovery = _breaker_config()
        breaker = CircuitBreaker(
            failure_threshold=threshold,
            recovery_timeout=recovery,
            name=name,
        )
        _llm_breakers[name] = breaker
        return breaker


def iter_llm_breakers() -> tuple[CircuitBreaker, ...]:
    """Breakers that already exist, in name order — does not create any.

    Used by ``/metrics``: instantiating a breaker for an endpoint that was never
    called would report a state for something that has never been tested.
    """
    with _llm_breaker_lock:
        return tuple(_llm_breakers[name] for name in sorted(_llm_breakers))


def get_llm_limiter() -> ConcurrencyLimiter:
    """Return the process-wide LLM concurrency limiter (lazy, env-configured)."""
    global _llm_limiter
    if _llm_limiter is not None:
        return _llm_limiter
    with _llm_limiter_lock:
        if _llm_limiter is not None:
            return _llm_limiter
        try:
            from . import config as _cfg  # pylint: disable=import-outside-toplevel

            max_conc = int(getattr(_cfg, "LLM_MAX_CONCURRENT", 20))
            max_q = int(getattr(_cfg, "LLM_QUEUE_MAX_DEPTH", 50))
            timeout = float(getattr(_cfg, "LLM_QUEUE_TIMEOUT_SECONDS", 30))
        except Exception:  # noqa: BLE001
            max_conc = 20
            max_q = 50
            timeout = 30.0
        _llm_limiter = ConcurrencyLimiter(
            max_concurrent=max_conc,
            max_queue_depth=max_q,
            queue_timeout=timeout,
            name="llm",
        )
        return _llm_limiter


def reset_llm_breaker(
    *,
    name: str = LLM_BREAKER_NAME,
    failure_threshold: int | None = None,
    recovery_timeout: float | None = None,
) -> CircuitBreaker:
    """Reset (or reconfigure) one endpoint's LLM breaker — intended for tests."""
    with _llm_breaker_lock:
        breaker = get_llm_breaker(name)
        if failure_threshold is not None:
            breaker.failure_threshold = max(1, int(failure_threshold))
        if recovery_timeout is not None:
            breaker.recovery_timeout = max(0.0, float(recovery_timeout))
        breaker.reset()
        return breaker


def reset_llm_limiter(
    *,
    max_concurrent: int | None = None,
    max_queue_depth: int | None = None,
    queue_timeout: float | None = None,
) -> ConcurrencyLimiter:
    """Reset (or reconfigure) the global LLM limiter — intended for tests."""
    global _llm_limiter
    with _llm_limiter_lock:
        if _llm_limiter is None:
            limiter = get_llm_limiter()
            _llm_limiter = limiter
        assert _llm_limiter is not None
        if max_concurrent is not None:
            _llm_limiter.max_concurrent = max(1, int(max_concurrent))
        if max_queue_depth is not None:
            _llm_limiter.max_queue_depth = max(0, int(max_queue_depth))
        if queue_timeout is not None:
            _llm_limiter.queue_timeout = max(0.0, float(queue_timeout))
        _llm_limiter.reset()
        return _llm_limiter


def reset_all_for_tests(
    *,
    breaker_threshold: int = 3,
    breaker_recovery: float = 60.0,
    limiter_concurrent: int = 20,
    limiter_queue_depth: int = 50,
    limiter_timeout: float = 30.0,
) -> tuple[CircuitBreaker, ConcurrencyLimiter]:
    """Convenience: reset every breaker plus the limiter to known test defaults."""
    # Every *existing* breaker, not just the primary: a test that armed failover
    # must not leak an OPEN fallback into the next test.
    with _llm_breaker_lock:
        names = set(_llm_breakers) | {LLM_BREAKER_NAME}
    breaker = reset_llm_breaker(
        name=LLM_BREAKER_NAME,
        failure_threshold=breaker_threshold,
        recovery_timeout=breaker_recovery,
    )
    for name in sorted(names - {LLM_BREAKER_NAME}):
        reset_llm_breaker(
            name=name,
            failure_threshold=breaker_threshold,
            recovery_timeout=breaker_recovery,
        )
    limiter = reset_llm_limiter(
        max_concurrent=limiter_concurrent,
        max_queue_depth=limiter_queue_depth,
        queue_timeout=limiter_timeout,
    )
    return breaker, limiter
