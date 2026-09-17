"""Prometheus text-format exposition for the health sidecar.

Why this exists
---------------
``/health`` answers "is this process alive and can it serve a run". It is the
wrong instrument for "has the audit backup been failing for three days" or "how
close is the log volume to the floor" — those are questions about a *trend*, and
answering them by polling a JSON endpoint and diffing snapshots is how alerting
gets built badly. A Prometheus scrape endpoint is the standard answer, and it is
what turns the values ``/health`` already computes into something an alert rule
can reference.

Why it is a pure renderer
-------------------------
This module performs **no I/O at all**: it takes the payload
:func:`app.health._health_payload` already builds and formats it. That is
deliberate, because /metrics is scraped every 15s by default and shares a port
with the liveness probe. If gathering a metric required a network round trip,
a slow Redis tier would take down the readiness probe — the queue depth problem
described in :func:`app.job_queue.JobBackend.health`. Keeping the renderer pure
means every value is as cheap as ``/health`` is, and the one expensive value
(queue depth on a remote backend) is reported as absent rather than probed.
``GET /metrics?probe=1`` is the documented opt-in for forcing that read.

Metric names are prefixed ``va_lse_`` and follow Prometheus conventions: base
units in the name (``_seconds``, ``_bytes``, ``_total`` for counters), and a
single labelling scheme per family.

Two kinds of value
------------------
**Live values** are resolved at scrape time from their owner (the circuit
breaker's state, the limiter's queue depth, in-flight runs, the session count).
A mirrored copy would go stale — the breaker changes state in whichever thread
hit the failure — so the renderer reads the source of truth. Those reads are all
in-process.

**Accumulated values** (call durations, phase durations, transition counts) are
kept in the registry below. A histogram cannot be reconstructed from current
state, so the app records observations as they happen and this module renders
them. Recording is a few dict operations under a lock and never raises: a
metrics bug must not be able to fail a user's run.

Declared means visible
----------------------
Every histogram and counter is declared eagerly at import, so its ``_count`` /
``_total`` series exists with value 0 before the first observation. A dashboard
panel that only appears once traffic arrives is a panel that is blank during the
incident you are trying to diagnose. Labelled families are the exception — a
label combination exists only once it has been observed, which is inherent, so
alert rules over them use ``sum()`` or ``absent()``.

Label cardinality is bounded on purpose (see ``MAX_SERIES_PER_METRIC``). An
unbounded label is the classic way to take down the monitoring system that was
supposed to protect you, so new label combinations beyond the cap are dropped
and counted in ``va_lse_metrics_series_dropped_total`` rather than accepted.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

logger = logging.getLogger("app.metrics")

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# Enum encoding for va_lse_audit_backup_state. A single numeric series is far
# easier to alert on ("state > 1 for 30m") than a set of per-state boolean
# gauges, and the mapping is documented in the metric's HELP text.
BACKUP_STATE_VALUES: dict[str, int] = {
    "ok": 0,
    "disabled": 1,
    "never_ran": 2,
    "stale": 3,
    "error": 4,
    "unavailable": 5,
}

# Readiness as a gauge, plus cache backend health.
_STATE_HELP = (
    "Audit backup state as an enum: "
    + ", ".join(f"{value}={name}" for name, value in BACKUP_STATE_VALUES.items())
    + ". 0 is healthy; disabled(1) means no destination is configured."
)


def _escape_label(value: str) -> str:
    """Escape a label value per the Prometheus text format."""
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _escape_help(text: str) -> str:
    """Escape HELP text: backslashes and newlines only (it is not quoted)."""
    return text.replace("\\", "\\\\").replace("\n", "\\n")


def _render_labels(labels: Mapping[str, str], *, le: str | None = None) -> str:
    """Render a label block, adding ``le`` last (the convention for buckets)."""
    pairs = [(key, value) for key, value in sorted(labels.items())]
    if le is not None:
        pairs.append(("le", le))
    if not pairs:
        return ""
    rendered = ",".join(f'{key}="{_escape_label(str(value))}"' for key, value in pairs)
    return "{" + rendered + "}"


def _format_value(value: Any) -> str | None:
    """Render a number in the exposition format, or None when not representable.

    Returning None (rather than 0) for a missing value is the important part:
    emitting 0 for "unknown" would make a dashboard show a healthy disk and an
    empty queue at exactly the moment the value could not be read.
    """
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "+Inf" if value > 0 else "-Inf"
        return repr(value)
    return None


# --------------------------------------------------------------------- registry

# Cap on distinct label combinations per metric family. Reaching this means a
# label is carrying unbounded data (a request id, a user id, an error string),
# which is a bug in the caller — but the failure it causes is in *this* system, so
# the guard is here: extra combinations are dropped and reported.
MAX_SERIES_PER_METRIC = 200

# Cap on tracked browser sessions, so a client minting session ids cannot grow
# process memory without bound. Sessions also expire (see session_count).
MAX_TRACKED_SESSIONS = 10_000

# Latency ladders. The wide top end is not padding: a 2,000-page run legitimately
# takes tens of minutes, and a histogram that saturates at 30s would show every
# real run in the +Inf bucket and answer nothing.
_MS_BUCKETS_LLM: tuple[float, ...] = (
    50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000, 30000, 60000, 120000, 300000,
)
_MS_BUCKETS_PHASE: tuple[float, ...] = (
    100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000, 120000, 300000, 600000,
    1200000, 2400000,
)


@dataclass
class Counter:
    """Thread-safe labelled monotonic counter."""

    name: str
    help_text: str
    _values: dict[tuple[tuple[str, str], ...], float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def inc(self, **labels: Any) -> None:
        """Add one to a label combination. Never raises.

        No ``amount`` parameter on purpose: callers pass label names positionally
        impossible, and a keyword that could collide with a label name is a bug
        waiting to happen.
        """
        try:
            key = _label_key(labels)
            with self._lock:
                if not _admit(self.name, key, self._values):
                    return
                self._values[key] = self._values.get(key, 0.0) + 1.0
        except Exception:  # noqa: BLE001 - instrumentation must never break a run
            logger.debug("counter %s rejected an observation", self.name, exc_info=True)

    def samples(self) -> list[tuple[dict[str, str], float]]:
        with self._lock:
            return [(dict(key), value) for key, value in sorted(self._values.items())]

    def reset(self) -> None:
        with self._lock:
            self._values.clear()


@dataclass
class Histogram:
    """Thread-safe labelled histogram with fixed cumulative buckets."""

    name: str
    help_text: str
    buckets: tuple[float, ...] = _MS_BUCKETS_LLM
    _state: dict[tuple[tuple[str, str], ...], list[float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def observe(self, value: float, **labels: Any) -> None:
        """Record one observation. Never raises.

        Negative values are clamped to 0: a negative duration means a clock went
        backwards, and a bucket ladder cannot represent it.
        """
        try:
            if value is None:
                return
            observed = max(0.0, float(value))
            key = _label_key(labels)
            with self._lock:
                # One extra slot at the end for the running sum, so a lone dict
                # lookup covers the whole update.
                if not _admit(self.name, key, self._state):
                    return
                entry = self._state.get(key)
                if entry is None:
                    entry = [0.0] * (len(self.buckets) + 2)
                    self._state[key] = entry
                entry[-2] += 1.0  # count
                entry[-1] += observed  # sum
                for index, bound in enumerate(self.buckets):
                    if observed <= bound:
                        entry[index] += 1.0
                        break
        except Exception:  # noqa: BLE001 - instrumentation must never break a run
            logger.debug("histogram %s rejected an observation", self.name, exc_info=True)

    def samples(self) -> list[tuple[dict[str, str], list[float]]]:
        """Return ``{labels: [per-bucket counts..., count, sum]}``, stable order."""
        with self._lock:
            return [
                (dict(key), list(value)) for key, value in sorted(self._state.items())
            ]

    @property
    def observed(self) -> float:
        """Total observations across every label combination."""
        with self._lock:
            return sum(entry[-2] for entry in self._state.values())

    def reset(self) -> None:
        with self._lock:
            self._state.clear()


def _label_key(labels: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Canonical, hashable form of a label set (sorted so order never matters)."""
    return tuple(sorted((str(name), str(value)) for name, value in labels.items()))


def _admit(
    metric: str, key: tuple[tuple[str, str], ...], store: Mapping[Any, Any]
) -> bool:
    """Whether a new label combination may be recorded.

    Existing combinations are always admitted — the cap only stops the *set* from
    growing, so a busy label keeps counting and a new noisy one is refused.
    """
    if key in store or len(store) < MAX_SERIES_PER_METRIC:
        return True
    if metric == _DROPPED_NAME:
        # The drop counter is itself full. Stop here rather than recursing — the
        # cap is already reporting itself as saturated, which is the useful signal.
        return False
    _series_dropped.inc(metric=metric)
    return False


# Counter incremented when a family hits the cardinality cap. It is declared here
# rather than beside the other families so `_admit` can reach it without a forward
# reference; it is still emitted by `metric_names()`.
_DROPPED_NAME = "va_lse_metrics_series_dropped_total"

_series_dropped = Counter(
    name=_DROPPED_NAME,
    help_text=(
        "Observations refused because a metric family reached its label-combination "
        "cap. Non-zero means a label is carrying unbounded data."
    ),
)

# ------------------------------------------------------------- declared families

phase_duration_ms = Histogram(
    name="va_lse_phase_duration_ms",
    help_text=(
        "Pipeline phase duration in milliseconds, recorded at the same call sites "
        "that log phase= so a span, a log line, and a bucket share one vocabulary."
    ),
    buckets=_MS_BUCKETS_PHASE,
)

llm_call_duration_ms = Histogram(
    name="va_lse_llm_call_duration_ms",
    help_text=(
        "Duration of a logical LLM call (retries included) in milliseconds, by phase "
        "and outcome. This is what a user waits for, not one HTTP attempt."
    ),
    buckets=_MS_BUCKETS_LLM,
)

llm_calls_total = Counter(
    name="va_lse_llm_calls_total",
    help_text="Logical LLM calls by phase and outcome (ok|error).",
)

llm_attempts_total = Counter(
    name="va_lse_llm_attempts_total",
    help_text=(
        "Individual provider attempts. Every attempt is counted, including the one "
        "that succeeded, so attempts/calls is 1.0 with no retries and the excess "
        "above 1.0 is the retry overhead the endpoint is costing you."
    ),
)

llm_errors_total = Counter(
    name="va_lse_llm_errors_total",
    help_text="Failed LLM calls by error category (client|retry|moderation|moderation_nudge).",
)

llm_endpoint_duration_ms = Histogram(
    name="va_lse_llm_endpoint_duration_ms",
    help_text=(
        "Logical LLM call duration in milliseconds by serving endpoint, so a "
        "fallback that is slower (or faster) than the primary is visible while "
        "failover is engaged. Same buckets as va_lse_llm_call_duration_ms."
    ),
    buckets=_MS_BUCKETS_LLM,
)

llm_endpoint_calls_total = Counter(
    name="va_lse_llm_endpoint_calls_total",
    help_text=(
        "Logical LLM calls by serving endpoint and outcome. Split out from "
        "va_lse_llm_calls_total so existing phase dashboards keep working while "
        "failover is still observable."
    ),
)

llm_failover_total = Counter(
    name="va_lse_llm_failover_total",
    help_text=(
        "Calls served by the fallback endpoint because the primary failed. A "
        "non-zero rate means users are being served by the backup provider."
    ),
)

breaker_rejections_total = Counter(
    name="va_lse_circuit_breaker_rejections_total",
    help_text=(
        "Calls failed fast without touching the network because the breaker was OPEN. "
        "These are requests users were told to retry, not endpoint failures."
    ),
)

breaker_transitions_total = Counter(
    name="va_lse_circuit_breaker_transitions_total",
    help_text="Circuit-breaker state transitions, by source and destination state.",
)

# -------------------------------------------------------------- session tracking

_session_lock = threading.Lock()
_sessions: dict[str, float] = {}


@dataclass
class SessionTracker:
    """Counts browser sessions seen within a TTL.

    Streamlit exposes no session registry that is safe to read from a background
    thread, so the app reports its own: every script run stamps the session id it
    already mints for audit correlation. A session that stops re-running (tab
    closed, socket dropped) ages out of the window.

    This is deliberately **not** called "connected users". Streamlit re-executes
    the script on every widget interaction, so what it measures is "sessions that
    did something in the last TTL" — which is the number that matters for capacity,
    and is a smaller number than "sockets open" when users are idle.
    """

    ttl_seconds: float = 300.0

    def touch(self, session_id: str) -> None:
        """Record activity for a session. Never raises."""
        try:
            # Inside the guard: an untrusted id can raise from ``__bool__`` or
            # ``__hash__``, and this is called from the app's startup path.
            if not session_id:
                return
            now = time.monotonic()
            with _session_lock:
                if len(_sessions) >= MAX_TRACKED_SESSIONS and session_id not in _sessions:
                    # Evict the stalest entry rather than refuse the new session:
                    # refusing would under-report exactly when traffic is highest.
                    oldest = min(_sessions, key=_sessions.__getitem__)
                    _sessions.pop(oldest, None)
                _sessions[session_id] = now
        except Exception:  # noqa: BLE001 - instrumentation must never break a run
            logger.debug("session touch failed", exc_info=True)

    def count(self, *, ttl_seconds: float | None = None, now: float | None = None) -> int:
        """Sessions seen within the TTL, pruning anything older."""
        ttl = self.ttl_seconds if ttl_seconds is None else ttl_seconds
        current = time.monotonic() if now is None else now
        with _session_lock:
            stale = [sid for sid, seen in _sessions.items() if current - seen > ttl]
            for sid in stale:
                _sessions.pop(sid, None)
            return len(_sessions)

    def reset(self) -> None:
        with _session_lock:
            _sessions.clear()


sessions = SessionTracker()

# The TTL is read once at import, not per scrape: it is configuration, and pulling
# it in here keeps the web pod and a worker (which serves /metrics too but never
# imports app.main) reporting the same window without any wiring in either entry
# point. Guarded so this module stays importable on its own.
try:  # pragma: no cover - the fallback is the only unreachable path
    from . import config as _config

    sessions.ttl_seconds = float(getattr(_config, "METRICS_SESSION_TTL_SECONDS", 300))
except Exception:  # noqa: BLE001
    pass


def touch_session(session_id: str) -> None:
    """Record that a browser session is active (called once per script run)."""
    sessions.touch(session_id)


def session_count(*, ttl_seconds: float | None = None, now: float | None = None) -> int:
    """Sessions active within the window (see :class:`SessionTracker`)."""
    return sessions.count(ttl_seconds=ttl_seconds, now=now)


# ------------------------------------------------------------- recording hooks


def observe_llm_call(
    phase: str, outcome: str, duration_ms: float, *, endpoint: str | None = None
) -> None:
    """Record one logical LLM call. Called by ``app.llm`` on the hot path.

    ``endpoint`` additionally records the same call against the per-endpoint
    families. Optional so the phase/outcome families stay the primary view (and
    so a caller that has no endpoint still records something).
    """
    labels = {"phase": phase or "general", "outcome": outcome}
    llm_call_duration_ms.observe(duration_ms, **labels)
    llm_calls_total.inc(**labels)
    if endpoint:
        llm_endpoint_duration_ms.observe(duration_ms, endpoint=endpoint, outcome=outcome)
        llm_endpoint_calls_total.inc(endpoint=endpoint, outcome=outcome)


def observe_llm_failover(reason: str) -> None:
    """Record one call that moved from the primary to the fallback endpoint."""
    llm_failover_total.inc(reason=reason or "unknown")


def observe_llm_attempt(phase: str, outcome: str) -> None:
    """Record one provider attempt, including a successful one.

    ``outcome`` is ``ok`` | ``retry`` | ``error``. Because the successful attempt
    is recorded too, ``attempts / calls >= 1`` always holds and the ratio is
    readable as retry overhead rather than as a fraction of failures.
    """
    llm_attempts_total.inc(phase=phase or "general", outcome=outcome)


def observe_llm_error(category: str) -> None:
    """Record a failed call by category (a bounded vocabulary from ``app.llm``)."""
    llm_errors_total.inc(category=category or "unknown")


def observe_phase(phase: str, outcome: str, duration_ms: float) -> None:
    """Record one pipeline phase. Called by ``PhaseTimer``."""
    phase_duration_ms.observe(duration_ms, phase=phase or "unknown", outcome=outcome)


def observe_breaker_transition(breaker: str, old_state: str, new_state: str) -> None:
    breaker_transitions_total.inc(breaker=breaker, from_state=old_state, to_state=new_state)


def observe_breaker_rejection(breaker: str) -> None:
    breaker_rejections_total.inc(breaker=breaker)


def reset_for_tests() -> None:
    """Clear every accumulated series (tests only)."""
    for family in (
        phase_duration_ms,
        llm_call_duration_ms,
        llm_calls_total,
        llm_attempts_total,
        llm_errors_total,
        llm_endpoint_duration_ms,
        llm_endpoint_calls_total,
        llm_failover_total,
        breaker_rejections_total,
        breaker_transitions_total,
        _series_dropped,
    ):
        family.reset()
    sessions.reset()


def _epoch_seconds(raw: Any) -> float | None:
    """Convert an ISO-8601 timestamp from /health into a Unix timestamp."""
    if not isinstance(raw, str) or not raw:
        return None
    text = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


class _Writer:
    """Accumulates metric families, emitting HELP/TYPE once per name."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._declared: set[str] = set()

    def metric(
        self,
        name: str,
        value: Any,
        *,
        kind: str = "gauge",
        help_text: str = "",
        labels: Mapping[str, Any] | None = None,
    ) -> None:
        """Emit one sample. A value of None emits nothing at all."""
        rendered = _format_value(value)
        if rendered is None:
            return
        if name not in self._declared:
            self._lines.append(f"# HELP {name} {_escape_help(help_text)}")
            self._lines.append(f"# TYPE {name} {kind}")
            self._declared.add(name)
        if labels:
            pairs = ",".join(
                f'{key}="{_escape_label(str(raw))}"' for key, raw in sorted(labels.items())
            )
            self._lines.append(f"{name}{{{pairs}}} {rendered}")
        else:
            self._lines.append(f"{name} {rendered}")

    def info(self, name: str, labels: Mapping[str, Any], *, help_text: str = "") -> None:
        """Emit an info-style family (always-valued 1 with descriptive labels)."""
        self.metric(name, 1, help_text=help_text, labels=labels)

    def _declare(self, name: str, kind: str, help_text: str) -> None:
        if name in self._declared:
            return
        self._lines.append(f"# HELP {name} {_escape_help(help_text)}")
        self._lines.append(f"# TYPE {name} {kind}")
        self._declared.add(name)

    def histogram(
        self,
        family: Histogram,
        *,
        zero_when_empty: bool = False,
    ) -> None:
        """Emit a histogram family in exposition format.

        Buckets are already cumulative per label set (an observation increments
        every bucket it falls into and then stops), so they are written out as-is.
        ``zero_when_empty`` emits the unlabelled ``_count``/``_sum`` of 0 that an
        unobserved histogram would have, so the series is present for a ``rate()``
        or an alert to bind to before the first call arrives.
        """
        samples = family.samples()
        if not samples:
            if not zero_when_empty:
                return
            self._declare(family.name, "histogram", family.help_text)
            for bound in family.buckets:
                self._lines.append(
                    f'{family.name}_bucket{{le="{_format_value(bound)}"}} 0'
                )
            self._lines.append(f'{family.name}_bucket{{le="+Inf"}} 0')
            self._lines.append(f"{family.name}_sum 0")
            self._lines.append(f"{family.name}_count 0")
            return
        self._declare(family.name, "histogram", family.help_text)
        for labels, entry in samples:
            counts = entry[:-2]
            total = entry[-2]
            running = 0.0
            cumulative: list[float] = []
            # Buckets are stored non-cumulatively by the observe() fast path, so
            # accumulate here — exposition format requires `le` buckets to be
            # cumulative.
            for count in counts:
                running += count
                cumulative.append(running)
            for bound, count in zip(family.buckets, cumulative, strict=False):
                self._lines.append(
                    f"{family.name}_bucket{_render_labels(labels, le=str(bound))} "
                    f"{_format_value(count)}"
                )
            # The +Inf bucket is the observation count by definition; taking it
            # from the running total keeps it correct even for values above the top
            # explicit bucket.
            self._lines.append(
                f'{family.name}_bucket{_render_labels(labels, le="+Inf")} '
                f"{_format_value(total)}"
            )
            self._lines.append(
                f"{family.name}_sum{_render_labels(labels)} {_format_value(entry[-1])}"
            )
            self._lines.append(
                f"{family.name}_count{_render_labels(labels)} {_format_value(total)}"
            )

    def counter(self, family: Counter, *, zero_when_empty: bool = False) -> None:
        """Emit a counter family.

        The declared name ends in ``_total``, so it is emitted verbatim; Prometheus
        does not add a suffix the way it does for histogram bucket children.
        """
        samples = family.samples()
        if not samples and zero_when_empty:
            self.metric(family.name, 0, kind="counter", help_text=family.help_text)
            return
        for labels, value in samples:
            self.metric(family.name, value, kind="counter", help_text=family.help_text, labels=labels)

    def text(self) -> str:
        # A trailing newline terminates the final sample as the format requires.
        return "\n".join(self._lines) + "\n" if self._lines else ""


def _emit_process(writer: _Writer, payload: Mapping[str, Any]) -> None:
    writer.metric(
        "va_lse_up",
        1,
        help_text="1 while the health sidecar is serving. Absence means the process is down.",
    )
    writer.metric(
        "va_lse_uptime_seconds",
        payload.get("uptime_s"),
        help_text="Seconds since the health sidecar started.",
    )
    ready = payload.get("ready")
    if isinstance(ready, bool):
        writer.metric(
            "va_lse_ready",
            ready,
            help_text="1 when readiness checks pass (/ready returns 200), 0 otherwise.",
        )


def _emit_cache(writer: _Writer, payload: Mapping[str, Any]) -> None:
    cache = payload.get("cache")
    if not isinstance(cache, Mapping):
        return
    labels = {
        "backend": str(cache.get("backend") or "unknown"),
    }
    writer.info(
        "va_lse_cache_info",
        labels,
        help_text="Active shared-cache backend, as an info metric with the backend name.",
    )


def _emit_tracing(writer: _Writer, payload: Mapping[str, Any]) -> None:
    tracing = payload.get("tracing")
    if not isinstance(tracing, Mapping):
        return
    writer.metric(
        "va_lse_tracing_enabled",
        1 if tracing.get("enabled") else 0,
        help_text="1 when OpenTelemetry tracing is configured (VA_LSE_TRACING=1).",
    )
    writer.metric(
        "va_lse_tracing_active",
        1 if tracing.get("active") else 0,
        help_text="1 when spans are actually being recorded; 0 with a reason in /health.",
    )


def _emit_audit(writer: _Writer, payload: Mapping[str, Any]) -> None:
    audit = payload.get("audit")
    if not isinstance(audit, Mapping):
        return
    writer.metric(
        "va_lse_audit_configured",
        1 if audit.get("configured") else 0,
        help_text="1 when the audit stream is writing to disk.",
    )
    failures = audit.get("write_failures")
    if isinstance(failures, int):
        writer.metric(
            "va_lse_audit_write_failures_total",
            failures,
            kind="counter",
            help_text=(
                "Audit records lost to failed writes since process start. Non-zero means "
                "compliance records were dropped — usually a full or read-only volume."
            ),
        )


def _emit_audit_backup(writer: _Writer, payload: Mapping[str, Any]) -> None:
    backup = payload.get("audit_backup")
    if not isinstance(backup, Mapping):
        return
    state = str(backup.get("status") or "unavailable")
    writer.metric(
        "va_lse_audit_backup_state",
        BACKUP_STATE_VALUES.get(state, BACKUP_STATE_VALUES["unavailable"]),
        help_text=_STATE_HELP,
    )
    writer.metric(
        "va_lse_audit_backup_configured",
        1 if backup.get("configured") else 0,
        help_text="1 when a backup destination is configured (VA_LSE_AUDIT_BACKUP_DESTINATION).",
    )
    off_pod = backup.get("off_pod")
    if isinstance(off_pod, bool):
        writer.metric(
            "va_lse_audit_backup_off_pod",
            1 if off_pod else 0,
            help_text=(
                "1 when the destination is genuinely off the pod. 0 means it shares the "
                "audit log's volume and cannot survive the failure a backup exists for."
            ),
        )
    writer.metric(
        "va_lse_audit_backup_last_success_timestamp_seconds",
        _epoch_seconds(backup.get("last_success_utc")),
        help_text="Unix timestamp of the last successful backup pass.",
    )
    writer.metric(
        "va_lse_audit_backup_age_seconds",
        backup.get("age_seconds"),
        help_text="Seconds since the last successful backup pass.",
    )
    writer.metric(
        "va_lse_audit_backup_pending_bytes",
        backup.get("pending_bytes"),
        help_text="Audit bytes on this volume not yet shipped — what a pod death would lose.",
    )
    writer.metric(
        "va_lse_audit_backup_uploaded_objects_total",
        backup.get("uploaded_objects"),
        kind="counter",
        help_text="Backup objects uploaded since the state file was created.",
    )
    writer.metric(
        "va_lse_audit_backup_uploaded_bytes_total",
        backup.get("uploaded_bytes"),
        kind="counter",
        help_text="Audit bytes uploaded since the state file was created.",
    )
    writer.metric(
        "va_lse_audit_backup_runs_total",
        backup.get("runs"),
        kind="counter",
        help_text="Backup passes attempted since the state file was created.",
    )
    writer.metric(
        "va_lse_audit_backup_interval_seconds",
        _hours_to_seconds(backup.get("interval_hours")),
        help_text="Configured interval between backup passes.",
    )
    writer.metric(
        "va_lse_audit_retention_days",
        backup.get("local_retention_days"),
        help_text="Age after which rotated audit files are deleted locally.",
    )
    writer.metric(
        "va_lse_audit_backup_cloud_retention_days",
        backup.get("cloud_retention_days"),
        help_text="Age after which backed-up objects are pruned from the destination.",
    )
    writer.info(
        "va_lse_audit_backup_info",
        {
            "status": state,
            "destination": str(backup.get("destination") or "none"),
        },
        help_text="Audit backup status and destination, as an info metric.",
    )


def _hours_to_seconds(hours: Any) -> float | None:
    if isinstance(hours, bool) or not isinstance(hours, (int, float)):
        return None
    return float(hours) * 3600.0


def _emit_disk(writer: _Writer, payload: Mapping[str, Any]) -> None:
    disk = payload.get("disk")
    if not isinstance(disk, Mapping) or not disk.get("checked"):
        return
    writer.metric(
        "va_lse_disk_free_bytes",
        disk.get("free_bytes"),
        help_text="Free bytes on the volume holding the log streams.",
    )
    writer.metric(
        "va_lse_disk_total_bytes",
        disk.get("total_bytes"),
        help_text="Total bytes on the volume holding the log streams.",
    )
    writer.metric(
        "va_lse_disk_used_percent",
        disk.get("used_percent"),
        help_text="Percentage of the log volume in use.",
    )
    writer.metric(
        "va_lse_disk_below_floor",
        1 if disk.get("below_floor") else 0,
        help_text=(
            "1 when free space is under VA_LSE_DISK_MIN_FREE_BYTES. The configured floor "
            "is exported as va_lse_disk_min_free_bytes."
        ),
    )
    writer.metric(
        "va_lse_disk_min_free_bytes",
        disk.get("min_free_bytes"),
        help_text="Configured free-space floor for the log volume.",
    )


def _emit_job_queue(writer: _Writer, payload: Mapping[str, Any]) -> None:
    queue = payload.get("job_queue")
    if not isinstance(queue, Mapping):
        return
    writer.metric(
        "va_lse_job_queue_enabled",
        1 if queue.get("enabled") else 0,
        help_text="1 when runs are submitted to the queue instead of running in-process.",
    )
    writer.metric(
        "va_lse_job_queue_distributed",
        1 if queue.get("is_distributed") else 0,
        help_text=(
            "1 when the active backend can serve a worker in another process. 0 with the "
            "queue enabled means jobs can never be claimed by a separate worker."
        ),
    )
    # Depth is only present when it was free to read. On a remote backend the
    # liveness path deliberately skips the round trip (see the module docstring),
    # so the series is simply absent rather than reported as zero.
    if isinstance(queue.get("depth"), int):
        writer.metric(
            "va_lse_job_queue_depth",
            queue["depth"],
            help_text="Jobs waiting for a worker, summed across kinds.",
        )
    writer.info(
        "va_lse_job_queue_info",
        {"backend": str(queue.get("backend") or "unknown")},
        help_text="Active job-queue backend, as an info metric with the backend name.",
    )


# Breaker state as an enum gauge. Numeric with a documented mapping rather than
# one boolean per state, so a single series can be alerted on consistently.
_BREAKER_STATE_VALUES: dict[str, int] = {"CLOSED": 0, "HALF_OPEN": 1, "OPEN": 2, "UNKNOWN": 3}
_BREAKER_STATE_HELP = (
    "Circuit-breaker state as an enum: "
    + ", ".join(f"{value}={name}" for name, value in _BREAKER_STATE_VALUES.items())
    + ". 0 is healthy; 2 means calls are failing fast."
)


def _emit_phases(writer: _Writer) -> None:
    """Pipeline phase latency — the p50/p95/p99 a dashboard actually plots."""
    writer.histogram(phase_duration_ms, zero_when_empty=True)


def _emit_llm(writer: _Writer) -> None:
    """LLM call latency, throughput, retry overhead, error mix, and failover."""
    writer.histogram(llm_call_duration_ms, zero_when_empty=True)
    writer.counter(llm_calls_total, zero_when_empty=True)
    writer.counter(llm_attempts_total, zero_when_empty=True)
    writer.counter(llm_errors_total, zero_when_empty=True)
    writer.histogram(llm_endpoint_duration_ms, zero_when_empty=True)
    writer.counter(llm_endpoint_calls_total, zero_when_empty=True)
    writer.counter(llm_failover_total, zero_when_empty=True)
    writer.counter(breaker_rejections_total, zero_when_empty=True)
    writer.counter(breaker_transitions_total, zero_when_empty=True)
    writer.counter(_series_dropped, zero_when_empty=True)


def _emit_runtime(writer: _Writer) -> None:
    """Live in-process values, read from their owner at scrape time.

    Every read here is local (a lock and a counter), which is what lets /metrics
    stay off the network. Anything that could block belongs on the ``?probe=1``
    path instead, not here.

    Each block is independently guarded: one failing source must not blank the
    rest of the exposition, because the metrics that survive are exactly the ones
    an operator needs while something is wrong.
    """
    breakers: tuple[Any, ...] = ()
    limiter = None
    try:
        from .circuit_breaker import get_llm_breaker, get_llm_limiter, iter_llm_breakers

        # Ensure the primary exists so its series are always reported; the
        # fallback (and any other endpoint) appears once it has been used. This is
        # the one place a breaker may be created — a state gauge is worth it.
        get_llm_breaker()
        breakers = iter_llm_breakers()
        limiter = get_llm_limiter()
    except Exception:  # noqa: BLE001 - never blank the exposition
        logger.debug("circuit breaker unavailable for metrics", exc_info=True)

    for breaker in breakers:
        try:
            state = str(breaker.state)
            writer.metric(
                "va_lse_circuit_breaker_state",
                _BREAKER_STATE_VALUES.get(state, _BREAKER_STATE_VALUES["UNKNOWN"]),
                labels={"breaker": breaker.name},
                help_text=_BREAKER_STATE_HELP,
            )
            writer.metric(
                "va_lse_circuit_breaker_consecutive_failures",
                breaker.failure_count,
                labels={"breaker": breaker.name},
                help_text=(
                    "Consecutive logical LLM failures counted toward the threshold. "
                    "Resets on success; a value at the threshold means the breaker "
                    "is about to open (or already has)."
                ),
            )
            writer.metric(
                "va_lse_circuit_breaker_open",
                1 if state == "OPEN" else 0,
                labels={"breaker": breaker.name},
                help_text=(
                    "1 while the breaker is OPEN and calls are failing fast without "
                    "touching the endpoint. Alert on this, not on the enum."
                ),
            )
            # How long this endpoint has been continuously unhealthy — the same
            # clock the failover rule uses, so an operator can see the threshold
            # approaching. 0 (not absent) when healthy: that is a known value.
            unhealthy = breaker.unhealthy_for_seconds()
            writer.metric(
                "va_lse_circuit_breaker_unhealthy_seconds",
                0.0 if unhealthy is None else unhealthy,
                labels={"breaker": breaker.name},
                help_text=(
                    "Seconds this endpoint has been continuously unhealthy, or 0 "
                    "while healthy. Unlike the probe countdown this is not reset by "
                    "a failed probe, so it keeps growing through a long outage."
                ),
            )
        except Exception:  # noqa: BLE001
            logger.debug("circuit breaker metrics failed", exc_info=True)

    # Failover configuration and whether it is currently in use. Read from config
    # + the primary breaker, never remembered client-side (a client is rebuilt on
    # every Streamlit rerun, so a flag on it would forget an ongoing outage).
    try:
        from .llm import failover_status

        status = failover_status()
        unhealthy = status.get("primary_unhealthy_seconds")
        writer.metric(
            "va_lse_llm_failover_enabled",
            1 if status.get("configured") else 0,
            help_text=(
                "1 when a second LLM endpoint is configured (OPENAI_BASE_URL_FALLBACK). "
                "0 means this deployment runs on one endpoint with no failover."
            ),
        )
        writer.metric(
            "va_lse_llm_failover_active",
            1 if status.get("active") else 0,
            help_text=(
                "1 while calls are being served by the fallback endpoint because the "
                "primary has been unhealthy past the grace period. This is the "
                "\"running on the backup provider\" signal."
            ),
        )
        writer.metric(
            "va_lse_llm_failover_after_seconds",
            status.get("after_seconds"),
            help_text=(
                "Grace period the primary must stay unhealthy for before failover "
                "engages (LLM_ENDPOINT_FALLBACK_TIMEOUT_SECONDS)."
            ),
        )
        writer.metric(
            "va_lse_llm_primary_unhealthy_seconds",
            0.0 if unhealthy is None else unhealthy,
            help_text=(
                "Seconds the primary endpoint has been continuously unhealthy; 0 "
                "while it is healthy."
            ),
        )
    except Exception:  # noqa: BLE001
        logger.debug("failover metrics failed", exc_info=True)

    if limiter is not None:
        try:
            writer.metric(
                "va_lse_llm_active_calls",
                limiter.active,
                help_text="LLM calls executing right now, across all sessions.",
            )
            writer.metric(
                "va_lse_llm_queued_calls",
                limiter.waiting,
                help_text=(
                    "Callers waiting for an LLM concurrency slot. Sustained non-zero "
                    "means the concurrency cap is the bottleneck."
                ),
            )
            writer.metric(
                "va_lse_llm_max_concurrent_calls",
                limiter.max_concurrent,
                help_text="Configured concurrency cap (VA_LSE_MAX_CONCURRENT_LLM_CALLS).",
            )
        except Exception:  # noqa: BLE001
            logger.debug("limiter metrics failed", exc_info=True)

    try:
        from .shutdown import inflight_count

        writer.metric(
            "va_lse_active_requests",
            inflight_count(),
            help_text=(
                "Evaluate/Draft runs in flight in this process. This is the number "
                "that must reach zero before a SIGTERM drain completes."
            ),
        )
    except Exception:  # noqa: BLE001
        logger.debug("inflight metrics failed", exc_info=True)

    try:
        from .health import cached_readiness_state

        # The cached verdict only — /metrics must not probe the LLM gateway.
        # Absent until something has actually checked, so an unattended deployment
        # does not report readiness it never verified.
        ready = cached_readiness_state()
        if ready is not None:
            writer.metric(
                "va_lse_ready",
                1 if ready else 0,
                help_text=(
                    "Last cached readiness verdict: 1 when the LLM endpoint and both "
                    "configured models are reachable. Absent until a readiness check "
                    "has run, because /metrics never probes the network itself."
                ),
            )
    except Exception:  # noqa: BLE001
        logger.debug("readiness metric failed", exc_info=True)

    try:
        writer.metric(
            "va_lse_session_count",
            sessions.count(),
            help_text=(
                "Browser sessions seen within VA_LSE_METRICS_SESSION_TTL_SECONDS. Not "
                "\"sockets open\": Streamlit re-runs the script on each interaction, "
                "so this counts sessions that were active recently."
            ),
        )
        writer.metric(
            "va_lse_session_ttl_seconds",
            sessions.ttl_seconds,
            help_text="Window used by va_lse_session_count.",
        )
    except Exception:  # noqa: BLE001
        logger.debug("session metrics failed", exc_info=True)


def render_prometheus(payload: Mapping[str, Any]) -> str:
    """Render a ``/health`` payload as Prometheus text exposition format.

    Combines the supplied payload (config-derived values, whose source of truth is
    the ``/health`` assembler) with the registry's accumulated series and a handful
    of live in-process reads. Performing no **I/O** is the invariant that matters:
    no network, no destination clients, no config reloads. A value that could not be
    read is omitted rather than defaulted to 0, so an absent series means "not
    measurable right now" instead of "healthy".
    """
    writer = _Writer()
    _emit_process(writer, payload)
    _emit_runtime(writer)
    _emit_phases(writer)
    _emit_llm(writer)
    _emit_cache(writer, payload)
    _emit_tracing(writer, payload)
    _emit_audit(writer, payload)
    _emit_audit_backup(writer, payload)
    _emit_disk(writer, payload)
    _emit_job_queue(writer, payload)
    return writer.text()


def metric_names() -> Sequence[str]:
    """Every metric family this renderer can emit (used by tests and docs)."""
    return (
        "va_lse_up",
        "va_lse_uptime_seconds",
        "va_lse_ready",
        "va_lse_circuit_breaker_state",
        "va_lse_circuit_breaker_open",
        "va_lse_circuit_breaker_consecutive_failures",
        "va_lse_circuit_breaker_unhealthy_seconds",
        "va_lse_llm_failover_enabled",
        "va_lse_llm_failover_active",
        "va_lse_llm_failover_after_seconds",
        "va_lse_llm_primary_unhealthy_seconds",
        "va_lse_llm_endpoint_duration_ms",
        "va_lse_llm_endpoint_calls_total",
        "va_lse_llm_failover_total",
        "va_lse_llm_active_calls",
        "va_lse_llm_queued_calls",
        "va_lse_llm_max_concurrent_calls",
        "va_lse_active_requests",
        "va_lse_session_count",
        "va_lse_session_ttl_seconds",
        "va_lse_phase_duration_ms",
        "va_lse_llm_call_duration_ms",
        "va_lse_llm_calls_total",
        "va_lse_llm_attempts_total",
        "va_lse_llm_errors_total",
        "va_lse_circuit_breaker_rejections_total",
        "va_lse_circuit_breaker_transitions_total",
        "va_lse_metrics_series_dropped_total",
        "va_lse_cache_info",
        "va_lse_tracing_enabled",
        "va_lse_tracing_active",
        "va_lse_audit_configured",
        "va_lse_audit_write_failures_total",
        "va_lse_audit_backup_state",
        "va_lse_audit_backup_configured",
        "va_lse_audit_backup_off_pod",
        "va_lse_audit_backup_last_success_timestamp_seconds",
        "va_lse_audit_backup_age_seconds",
        "va_lse_audit_backup_pending_bytes",
        "va_lse_audit_backup_uploaded_objects_total",
        "va_lse_audit_backup_uploaded_bytes_total",
        "va_lse_audit_backup_runs_total",
        "va_lse_audit_backup_interval_seconds",
        "va_lse_audit_retention_days",
        "va_lse_audit_backup_cloud_retention_days",
        "va_lse_audit_backup_info",
        "va_lse_disk_free_bytes",
        "va_lse_disk_total_bytes",
        "va_lse_disk_used_percent",
        "va_lse_disk_below_floor",
        "va_lse_disk_min_free_bytes",
        "va_lse_job_queue_enabled",
        "va_lse_job_queue_distributed",
        "va_lse_job_queue_depth",
        "va_lse_job_queue_info",
    )
