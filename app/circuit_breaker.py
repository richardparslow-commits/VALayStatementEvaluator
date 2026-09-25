"""Circuit breaker and concurrency limiter for LLM calls.

Stdlib-only implementation that mirrors the behaviour the spec asks for
(``pybreaker`` style) without adding a new dependency.

* **Circuit breaker** — per-process, thread-safe. Opens after
  ``VA_LSE_CB_FAILURE_THRESHOLD`` (default 3) consecutive *logical* LLM call
  failures (a call that exhausts its own retries counts as one failure). While
  OPEN every call fails fast (``CircuitBreakerOpenError``) in <2 s without
  touching the network — this stops 100 concurrent users from hammering a
  degraded endpoint. After ``VA_LSE_CB_RECOVERY_SECONDS`` (default 60 s) the
  breaker moves to HALF_OPEN and admits calls again; the first reported
  verdict ends the trial — a success closes it, a failure re-opens it for
  another full window. HALF_OPEN deliberately admits *every* caller instead
  of metering a single probe slot: a slot whose holder never reports back (a
  cancelled or crashed call) would wedge the breaker shut for the life of the
  process, and the recovery burst is already bounded by the concurrency
  limiter and rate gate below. All state changes log at WARNING.

* **Concurrency limiter** — global semaphore with an in-memory queue.
  ``VA_LSE_MAX_CONCURRENT_LLM_CALLS`` (default 20) caps the number of LLM
  calls executing at once. When the cap is hit, callers are queued up to
  ``VA_LSE_LLM_QUEUE_MAX_DEPTH`` (default 50); beyond that the call is
  rejected immediately with ``QueueFullError``. Waiting callers block up to
  ``VA_LSE_LLM_QUEUE_TIMEOUT_SECONDS`` (default 30 s) before timing out.

* **Rate gate** — optional minimum-interval pacing between LLM call
  admissions. ``VA_LSE_LLM_MIN_INTERVAL_SECONDS`` (default 0 = off) forces
  at least that many seconds between the starts of successive calls in this
  process; ``VA_LSE_LLM_MAX_RPM`` expresses the same thing as a
  requests-per-minute cap (when both are set the stricter wins). A caller
  whose reserved slot lies beyond its wait budget is rejected with
  ``RateGateTimeoutError`` (a ``QueueFullError`` subclass, so chat()'s
  fail-fast routing applies). Unlike the breaker and retries, spacing is
  *preventive*: its goal is that the provider's 429s never happen, where
  retries can only absorb the ones that do.

The breaker is **per endpoint** (``get_llm_breaker``), because a breaker is a
statement about one endpoint's health: a shared one would let a healthy fallback's
successes close the primary's breaker and hide an ongoing outage, and let the
primary's failures refuse calls the fallback could have served. The concurrency
limiter stays global — it bounds *this process's* egress, not one provider's.

Each breaker also tracks how long its endpoint has been *continuously*
unhealthy (``unhealthy_for_seconds``), which is the failover trigger. That is a
different clock from the probe countdown: ``_opened_at`` is reset by every failed
probe, so it never ages past one ``recovery_timeout`` while traffic keeps probing.
Test helpers ``reset_llm_breaker`` / ``reset_llm_limiter`` /
``reset_llm_rate_gate`` allow tests to reconfigure them without restarting
the process.
"""

from __future__ import annotations

import logging
import threading
import time
from types import TracebackType
from typing import Literal

logger = logging.getLogger("app.circuit_breaker")

CircuitState = Literal["CLOSED", "OPEN", "HALF_OPEN"]

# How much of the recorded failure reason is quoted back to the user. The reason
# is a provider error string, which is already user-facing, but a long one would
# bury the sentence that says what to do about it.
MAX_FAILURE_REASON_CHARS = 300


class CircuitBreakerError(RuntimeError):
    """Base for circuit-breaker rejections."""


class CircuitBreakerOpenError(CircuitBreakerError):
    """Raised when the breaker is OPEN and the call is rejected fast."""


class ConcurrencyLimitError(RuntimeError):
    """Base for concurrency/queue rejections."""


class QueueFullError(ConcurrencyLimitError):
    """Raised when the concurrency queue is at capacity."""


class RateGateTimeoutError(QueueFullError):
    """Raised when a caller's pacing-wait budget is exhausted before its slot.

    Deliberately a ``QueueFullError`` subclass: ``LLMClient.chat`` treats
    queue rejections as fail-fast (never retried on the other endpoint), and
    a pacing timeout is the same kind of rejection — the provider is fine,
    this process is simply asking faster than its configured spacing allows,
    and switching endpoints would not help because the gate is process-wide.
    """


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
        # Why the most recent counted failure happened, and whether retrying it
        # could ever help. The breaker raises its own message while OPEN, and
        # without these it can only say "the endpoint is unavailable" — a guess
        # that is wrong for the failures that never reach the endpoint at all
        # (a rejected key, a model id the account cannot use), which are exactly
        # the ones a user needs named.
        self._last_failure_reason: str = ""
        self._last_failure_retriable: bool = True
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

    @property
    def last_failure_reason(self) -> str:
        """Why the most recent counted failure happened (``""`` when none)."""
        with self._lock:
            return self._last_failure_reason

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

        Handles the OPEN -> HALF_OPEN time-based transition. HALF_OPEN admits
        every caller — there is no single-probe slot to leak (see the module
        docstring); the first ``record_success``/``record_failure`` ends the
        trial.
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
            # A success retires the recorded reason in every branch below: it
            # describes a failure that this endpoint has since recovered from,
            # and a stale reason would be quoted by the *next* breaker-opening
            # message as though it were the cause of that one.
            self._last_failure_reason = ""
            self._last_failure_retriable = True
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

    def record_failure(self, *, reason: str = "", retriable: bool = True) -> None:
        """Count one failed logical call, remembering *why* it failed.

        ``reason`` is what :meth:`check_or_raise` quotes afterwards, and
        ``retriable`` says whether an identical attempt could ever succeed. Both
        are optional so that a caller with nothing to add (and every existing
        test) behaves exactly as before.
        """
        with self._lock:
            if reason:
                self._last_failure_reason = reason.strip()[:MAX_FAILURE_REASON_CHARS]
                self._last_failure_retriable = retriable
            if self._state == "HALF_OPEN":
                self._opened_at = time.monotonic()
                self._transition(
                    "OPEN",
                    reason=f"probe failed: {self._last_failure_reason or 'reason not reported'}",
                )
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
                    reason=(
                        f"{self._failure_count} consecutive failures >= threshold "
                        f"{self.failure_threshold}: "
                        f"{self._last_failure_reason or 'reason not reported'}"
                    ),
                )

    # ---------------------------------------------------------------- helpers

    def check_or_raise(self) -> None:
        """Fail fast with CircuitBreakerOpenError if the breaker is OPEN."""
        if not self.allow_request():
            remaining = 0.0
            with self._lock:
                if self._opened_at is not None:
                    remaining = max(0.0, self.recovery_timeout - (time.monotonic() - self._opened_at))
                reason = self._last_failure_reason
                retriable = self._last_failure_retriable
            # The reason is named, not guessed. "endpoint temporarily unavailable"
            # was wrong for every failure that never reached the endpoint (a
            # rejected key, a model id this account cannot use): those fail fast
            # here exactly like an outage does, and the user was told to wait for
            # a recovery that could never arrive.
            detail = (
                f" Last failure: {reason}" if reason else " The endpoint gave no reason."
            )
            if reason and not retriable:
                diagnosis = (
                    " That failure is deterministic, so retrying it unchanged will reproduce it — "
                    "fix the request (API key, model id, or endpoint) rather than waiting."
                )
            else:
                diagnosis = (
                    f" The breaker probes again in {self.recovery_timeout:.0f}s and closes itself "
                    "if the endpoint has recovered."
                )
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN — {self.failure_threshold} consecutive "
                f"LLM calls failed, so calls to this endpoint fail fast (no network) for another "
                f"{remaining:.0f}s.{detail}{diagnosis}"
            )

    def reset(self) -> None:
        """Force back to CLOSED (used in tests)."""
        with self._lock:
            self._state = "CLOSED"
            self._failure_count = 0
            self._opened_at = None
            self._unhealthy_since = None
            self._last_failure_reason = ""
            self._last_failure_retriable = True


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


# ---------------------------------------------------------------- rate gate


class MinimumIntervalGate:
    """Thread-safe minimum-spacing gate for LLM call admissions.

    Guarantees at least ``min_interval_seconds`` between the *starts* of
    successive admitted calls in this process, even under thread contention:
    each caller reserves the next free slot under the lock (first-come
    first-served), then sleeps until its slot outside the lock. Reservation —
    not read-then-sleep — is what prevents bursts: five threads arriving
    together are scheduled a full interval apart, not all woken at the same
    instant to race the provider's rate limiter.

    ``min_interval_seconds <= 0`` disables the gate entirely (``wait`` is a
    no-op), which is the default: spacing is opt-in because the right value
    depends on the provider tier.

    Unlike the concurrency limiter this does not bound simultaneous calls —
    it bounds their *rate*. The two compose: the gate spaces admissions, the
    limiter caps how many of the admitted calls overlap.
    """

    def __init__(
        self,
        min_interval_seconds: float = 0.0,
        queue_timeout: float = 30.0,
        name: str = "llm-rate",
    ) -> None:
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self.queue_timeout = max(0.0, float(queue_timeout))
        self.name = name
        self._next_free = 0.0  # monotonic timestamp of the next free slot
        self._last_admitted = 0.0  # monotonic timestamp of the last admission
        self._lock = threading.Lock()

    # --------------------------------------------------------------- stats

    @property
    def enabled(self) -> bool:
        return self.min_interval_seconds > 0.0

    @property
    def last_admitted(self) -> float:
        """Monotonic timestamp of the most recent admission (0 before any)."""
        with self._lock:
            return self._last_admitted

    # ---------------------------------------------------------------- wait

    def wait(self, *, timeout: float | None = None) -> float:
        """Reserve the next admission slot and block until it arrives.

        Returns the seconds actually waited (0.0 when the gate is disabled or
        the slot is already due). Raises ``RateGateTimeoutError`` when the
        reservation lies further out than the caller's wait budget —
        ``timeout`` if given, else ``queue_timeout``.
        """
        if not self.enabled:
            return 0.0
        effective_timeout = self.queue_timeout if timeout is None else max(0.0, float(timeout))
        now = time.monotonic()
        with self._lock:
            scheduled = max(now, self._next_free)
            self._next_free = scheduled + self.min_interval_seconds
        delay = scheduled - now
        if delay <= 0.0:
            with self._lock:
                self._last_admitted = time.monotonic()
            return 0.0
        if delay > effective_timeout:
            logger.warning(
                "rate gate '%s' timeout: slot %.1fs out exceeds %.1fs budget "
                "(min_interval=%.2fs) — rejecting call",
                self.name,
                delay,
                effective_timeout,
                self.min_interval_seconds,
                extra={
                    "phase": "rate_gate",
                    "status": "timeout",
                    "gate": self.name,
                    "delay_seconds": round(delay, 3),
                    "min_interval": self.min_interval_seconds,
                },
            )
            raise RateGateTimeoutError(
                f"LLM rate gate '{self.name}' would wait {delay:.0f}s for a spaced "
                f"slot, over the {effective_timeout:.0f}s budget. This process is "
                f"asking faster than its spacing allows — raise the wait budget "
                f"(VA_LSE_LLM_QUEUE_TIMEOUT_SECONDS), relax the spacing "
                f"(VA_LSE_LLM_MIN_INTERVAL_SECONDS / VA_LSE_LLM_MAX_RPM), or slow "
                f"the callers (VA_LSE_RECORDS_CONCURRENCY)."
            )
        if delay > 0.05:
            logger.debug(
                "rate gate '%s': pacing call by %.2fs (min_interval=%.2fs)",
                self.name,
                delay,
                self.min_interval_seconds,
            )
        # Bounded sleep: delay <= effective_timeout, which the caller caps at
        # the pipeline's remaining budget. check_pipeline_cancelled() runs in
        # llm.py right after the wait returns, so a cancellation that lands
        # mid-sleep is honored within at most one budget — and the wait itself
        # stays short by construction (spacing is seconds, never minutes).
        time.sleep(delay)
        with self._lock:
            self._last_admitted = time.monotonic()
        return delay

    def reset(self) -> None:
        """Forget reserved slots and history (used in tests)."""
        with self._lock:
            self._next_free = 0.0
            self._last_admitted = 0.0


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


_llm_rate_gate: MinimumIntervalGate | None = None
_llm_rate_gate_lock = threading.RLock()


def _rate_gate_interval_from_config() -> float:
    """Effective spacing from config; stricter of the seconds-cap and RPM-cap.

    Both knobs express the same constraint, so when both are set the *larger*
    interval (the stricter cap) wins; a non-positive value means "unset".
    """
    try:
        from . import config as _cfg  # pylint: disable=import-outside-toplevel

        seconds = float(getattr(_cfg, "LLM_RATE_MIN_INTERVAL_SECONDS", 0.0) or 0.0)
        rpm = float(getattr(_cfg, "LLM_RATE_MAX_RPM", 0.0) or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0
    return max(0.0, max(seconds, (60.0 / rpm) if rpm > 0 else 0.0))


def get_llm_rate_gate() -> MinimumIntervalGate:
    """Return the process-wide LLM rate gate (lazy, env-configured)."""
    global _llm_rate_gate
    if _llm_rate_gate is not None:
        return _llm_rate_gate
    with _llm_rate_gate_lock:
        if _llm_rate_gate is not None:
            return _llm_rate_gate
        try:
            from . import config as _cfg  # pylint: disable=import-outside-toplevel

            timeout = float(getattr(_cfg, "LLM_QUEUE_TIMEOUT_SECONDS", 30))
        except Exception:  # noqa: BLE001
            timeout = 30.0
        _llm_rate_gate = MinimumIntervalGate(
            min_interval_seconds=_rate_gate_interval_from_config(),
            queue_timeout=timeout,
            name="llm-rate",
        )
        return _llm_rate_gate


def reset_llm_rate_gate(
    *,
    min_interval_seconds: float | None = None,
    max_rpm: float | None = None,
    queue_timeout: float | None = None,
) -> MinimumIntervalGate:
    """Reset (or reconfigure) the global LLM rate gate — intended for tests.

    ``max_rpm`` sets the interval from a requests-per-minute cap (0 disables);
    ``min_interval_seconds``, when also given, overrides it.
    """
    global _llm_rate_gate
    with _llm_rate_gate_lock:
        if _llm_rate_gate is None:
            _llm_rate_gate = get_llm_rate_gate()
        assert _llm_rate_gate is not None
        if min_interval_seconds is None and max_rpm is None:
            # No explicit interval: re-derive from config, so a test (or an
            # operator hot-reloading settings) gets the current env values
            # rather than whatever the singleton was first built with.
            _llm_rate_gate.min_interval_seconds = _rate_gate_interval_from_config()
        else:
            if max_rpm is not None:
                rpm = max(0.0, float(max_rpm))
                _llm_rate_gate.min_interval_seconds = (60.0 / rpm) if rpm > 0 else 0.0
            if min_interval_seconds is not None:
                _llm_rate_gate.min_interval_seconds = max(0.0, float(min_interval_seconds))
        if queue_timeout is not None:
            _llm_rate_gate.queue_timeout = max(0.0, float(queue_timeout))
        _llm_rate_gate.reset()
        return _llm_rate_gate


def reset_all_for_tests(
    *,
    breaker_threshold: int = 3,
    breaker_recovery: float = 60.0,
    limiter_concurrent: int = 20,
    limiter_queue_depth: int = 50,
    limiter_timeout: float = 30.0,
) -> tuple[CircuitBreaker, ConcurrencyLimiter]:
    """Convenience: reset every breaker, the limiter, and the rate gate.

    The rate gate is reset to *disabled* (spacing is opt-in) so a test that
    armed pacing cannot throttle an unrelated later test.
    """
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
    reset_llm_rate_gate(min_interval_seconds=0.0)
    return breaker, limiter
