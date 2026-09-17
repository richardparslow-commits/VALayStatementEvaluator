"""Distributed tracing (OpenTelemetry) for Evaluate/Draft runs.

The app already has the two halves of observability that do not need a backend:
structured JSON logs carrying a ``request_id`` (``app/logging_config.py``) and a
per-phase profiler that logs p50/p95/p99 (``app/profiler.py``). What neither
gives you is *span structure* — which phase was slow for **this** run — or a
trace that survives a process boundary. This module adds both.

Design
------
* **Opt-in and dependency-light.** Traces are emitted only when
  ``VA_LSE_TRACING=1``, and every entry point here degrades to a no-op when the
  OpenTelemetry packages are absent (``requirements-otel.txt``). The default
  install, the tests, and the single-user path behave exactly as before.
* **One trace per run, named after the phases you already know.** Span names are
  the same strings that appear as ``phase=`` in the structured log —
  ``records:review``, ``claims``, ``verify``, ``rubric``, ``topic``,
  ``revision``, ``report`` (Evaluate) and ``records:review``, ``grounding``,
  ``draft``, ``review`` (Draft) — so a slow span and its log lines line up
  without a translation table.
* **The queue boundary is traced.** In Pattern C the work moves to another pod,
  so the web pod injects W3C trace context into the job payload and the worker
  adopts it. The trace then reads: ``queue:submit`` (web) → ``run:evaluate``
  (worker) → phase spans, instead of two unrelated traces.
* **OTLP only, no vendor SDK.** Jaeger, Grafana Tempo, Datadog, New Relic and
  Honeycomb all ingest OTLP, so the exporter is configured with the standard
  ``OTEL_*`` variables and switching backends is an env change.
* **No PII, ever.** Statement text, observations, record text, and prompts are
  never attached to spans — the same rule ``logging_config`` enforces for logs.
  :func:`_screen` actively drops attributes whose name says they are free text,
  including any ``*_text`` key, so a future call site cannot leak by accident.
* **Cardinality is bounded on purpose.** The bulk digest makes one LLM call per
  chunk, so per-chunk and per-call spans are *off* by default
  (``VA_LSE_TRACE_CHUNK_SPANS`` / ``VA_LSE_TRACE_LLM_CALLS``): a 5,000-page run is
  hundreds of chunks, which would dwarf every other span in the trace. The
  fan-out is visible instead as a single ``records:digest`` span carrying
  ``records.chunks``/``records.concurrency``.

Threads: pipeline work runs in a ``ThreadPoolExecutor``, and a
:class:`contextvars.ContextVar` does not cross into those threads by itself. The
digest path captures :func:`current_span_context` before submitting and passes
it to :func:`phase_span` via *parent*, which is why per-chunk spans (when
enabled) still nest under the run instead of starting a new trace each.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, suppress
from pathlib import Path
from typing import Any, Protocol

from . import config

logger = logging.getLogger("app.tracing")

# --------------------------------------------------------------- optional dep

try:  # pragma: no cover - exercised by whichever branch the environment has
    from opentelemetry import trace as _otel_trace
    from opentelemetry.context import attach as _otel_attach
    from opentelemetry.context import detach as _otel_detach
    from opentelemetry.propagate import extract as _otel_extract
    from opentelemetry.propagate import inject as _otel_inject
    from opentelemetry.trace import NonRecordingSpan as _NonRecordingSpan
    from opentelemetry.trace import ProxyTracerProvider as _ProxyTracerProvider
    from opentelemetry.trace import set_span_in_context as _set_span_in_context

    _IMPORT_ERROR: str | None = None
except Exception as exc:  # noqa: BLE001 - any import failure must degrade, not crash
    _otel_trace = None  # type: ignore[assignment]
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

#: Attribute names that would put free text (and therefore potentially PHI) into
#: a span. Kept as an explicit set rather than a substring match so legitimate
#: names like ``records.pages`` survive. Any ``*_text`` suffix is dropped too.
_FORBIDDEN_ATTR_KEYS = frozenset(
    {
        "text",
        "chunk_text",
        "statement",
        "statement_text",
        "observations",
        "observation",
        "prompt",
        "system_prompt",
        "user_prompt",
        "content",
        "body",
        "witness",
        "records",
    }
)

_MAX_ATTR_CHARS = 200
_MAX_LIST_ATTR_ITEMS = 20


class SpanLike(Protocol):
    """The subset of the OTel ``Span`` API this app uses."""

    def set_attribute(self, key: str, value: Any) -> Any: ...
    def set_attributes(self, attributes: Mapping[str, Any]) -> Any: ...
    def add_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> Any: ...
    def record_exception(self, exception: BaseException) -> Any: ...
    def set_status(self, status: Any) -> Any: ...
    def is_recording(self) -> bool: ...


class _NullSpan:
    """Stand-in span used whenever tracing is off. Every method is a no-op."""

    __slots__ = ()

    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        return None

    def add_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> None:
        return None

    def record_exception(self, exception: BaseException) -> None:
        return None

    def set_status(self, status: Any) -> None:
        return None

    def is_recording(self) -> bool:
        return False


_NULL_SPAN = _NullSpan()

_provider: Any | None = None
_active_role: str = ""
_setup_error: str | None = None
_lock = threading.Lock()
_dropped_attr_keys: set[str] = set()


# ------------------------------------------------------------------ enablement


def is_enabled() -> bool:
    """True when tracing is asked for and the SDK can actually be imported.

    ``VA_LSE_TRACING`` alone is not enough: the OpenTelemetry packages have to
    be installed (``requirements-otel.txt``). ``OTEL_SDK_DISABLED=true`` — the
    OpenTelemetry-wide kill switch — also disables tracing, so an operator can
    turn it off without touching ``VA_LSE_TRACING``.
    """
    if _otel_trace is None:
        return False
    if os.getenv("OTEL_SDK_DISABLED", "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    return bool(config.TRACING_ENABLED)


def is_active() -> bool:
    """True once :func:`setup_tracing` has produced a provider in this process."""
    return _provider is not None


def unavailable_reason() -> str:
    """Why tracing is not active — for ``/health`` and the ops panel."""
    if _IMPORT_ERROR is not None:
        return f"opentelemetry packages not importable ({_IMPORT_ERROR}); install requirements-otel.txt"
    if not config.TRACING_ENABLED:
        return "disabled (set VA_LSE_TRACING=1 to enable)"
    if os.getenv("OTEL_SDK_DISABLED", "").strip().lower() in ("1", "true", "yes", "on"):
        return "disabled by OTEL_SDK_DISABLED"
    if _setup_error:
        return f"setup failed: {_setup_error}"
    if _provider is None:
        return "not initialised (call setup_tracing at the entry point)"
    return ""


def exporter_endpoint() -> str:
    """The OTLP endpoint in use, or the exporter's own default."""
    explicit = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "").strip() or os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", ""
    ).strip()
    return explicit or "http://localhost:4318/v1/traces (OTLP default)"


def _sampler() -> Any:
    """Build a sampler from ``OTEL_TRACES_SAMPLER`` when set, else from config.

    Only the standard sampler names are recognised; anything else is logged and
    ignored rather than silently changing trace volume.
    """
    from opentelemetry.sdk.trace.sampling import (
        ALWAYS_OFF,
        ALWAYS_ON,
        ParentBased,
        TraceIdRatioBased,
    )

    ratio = config.TRACING_SAMPLE_RATIO
    name = os.getenv("OTEL_TRACES_SAMPLER", "").strip().lower()
    raw_arg = os.getenv("OTEL_TRACES_SAMPLER_ARG", "").strip()
    arg: float
    try:
        arg = float(raw_arg) if raw_arg else ratio
    except ValueError:
        logger.warning("OTEL_TRACES_SAMPLER_ARG=%r is not a number; using %.3f", raw_arg, ratio)
        arg = ratio
    arg = min(1.0, max(0.0, arg))

    known = {
        "always_on": lambda: ParentBased(ALWAYS_ON),
        "parentbased_always_on": lambda: ParentBased(ALWAYS_ON),
        "always_off": lambda: ParentBased(ALWAYS_OFF),
        "parentbased_always_off": lambda: ParentBased(ALWAYS_OFF),
        "traceidratio": lambda: TraceIdRatioBased(arg),
        "parentbased_traceidratio": lambda: ParentBased(TraceIdRatioBased(arg)),
    }
    if name:
        factory = known.get(name)
        if factory is not None:
            return factory()
        logger.warning(
            "unsupported OTEL_TRACES_SAMPLER=%r; falling back to VA_LSE_TRACE_SAMPLE_RATIO=%.3f",
            name,
            ratio,
        )
    return ParentBased(TraceIdRatioBased(ratio)) if ratio < 1.0 else ParentBased(ALWAYS_ON)


def _build_exporter() -> Any | None:
    """Build the span exporter: OTLP over HTTP unless configured otherwise.

    ``OTEL_TRACES_EXPORTER`` follows the OpenTelemetry convention for the two
    names that are useful here: ``console`` (spans to stderr, for a quick local
    check) and ``none`` (record spans, export nothing). Anything else — including
    the default ``otlp`` — uses the OTLP exporter, whose endpoint/headers/timeout
    come from the standard ``OTEL_EXPORTER_OTLP_*`` variables.
    """
    name = os.getenv("OTEL_TRACES_EXPORTER", "").strip().lower() or "otlp"
    if name == "none":
        return None
    if name == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        return ConsoleSpanExporter()
    if name not in ("otlp", "otlp_proto_http", "otlp_http"):
        logger.warning(
            "unsupported OTEL_TRACES_EXPORTER=%r; using the OTLP HTTP exporter", name
        )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter()


def setup_tracing(*, role: str | None = None) -> bool:
    """Install the global tracer provider once. Returns True when tracing is live.

    Idempotent and never raises: a half-configured exporter must not take the
    app down, so any failure is logged at WARNING and tracing stays off.
    """
    global _provider, _setup_error, _active_role

    if not is_enabled():
        return False
    with _lock:
        if _provider is not None:
            return True
        try:
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            resource = Resource.create(
                {
                    "service.name": config.TRACING_SERVICE_NAME,
                    "service.instance.id": f"{socket.gethostname()}:{os.getpid()}",
                    "va_lse.role": role or config.TRACING_ROLE,
                }
            )
            provider = TracerProvider(resource=resource, sampler=_sampler())
            exporter = _build_exporter()
            if exporter is not None:
                provider.add_span_processor(BatchSpanProcessor(exporter))
            # Only claim the global provider if nothing else has. Re-setting it is
            # rejected (with a warning) by the SDK, and this module always uses its
            # own provider rather than the global one.
            if isinstance(_otel_trace.get_tracer_provider(), _ProxyTracerProvider):
                _otel_trace.set_tracer_provider(provider)
        except Exception as exc:  # noqa: BLE001 - tracing must never break a run
            _setup_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "tracing is enabled but could not be started (%s); continuing untraced",
                _setup_error,
                extra={"phase": "tracing", "status": "error", "error_class": type(exc).__name__},
            )
            return False
        _provider = provider
        _active_role = role or config.TRACING_ROLE

    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip() and not os.getenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", ""
    ).strip():
        # Not fatal — the exporter has a sensible localhost default — but a
        # deployment that expects a collector needs to know it is unset.
        logger.warning(
            "tracing enabled without OTEL_EXPORTER_OTLP_ENDPOINT; spans will be sent to %s",
            exporter_endpoint(),
            extra={"phase": "tracing", "status": "warn"},
        )
    logger.info(
        "tracing enabled service=%s role=%s endpoint=%s sample_ratio=%.3f",
        config.TRACING_SERVICE_NAME,
        _active_role,
        exporter_endpoint(),
        config.TRACING_SAMPLE_RATIO,
        extra={"phase": "tracing", "status": "armed"},
    )
    return True


def flush_tracing(timeout_millis: int = 5_000) -> bool:
    """Export anything buffered right now. Returns True when it went out.

    Useful for a one-shot process (``python -m app.worker --once``) and for
    checking a new collector without waiting for the batch timer.
    """
    provider = _provider
    if provider is None:
        return False
    try:
        return bool(provider.force_flush(timeout_millis))
    except Exception as exc:  # noqa: BLE001
        logger.debug("force_flush failed: %s", exc)
        return False


def shutdown_tracing() -> None:
    """Flush buffered spans and stop the exporter. Safe to call any number of times.

    Called from the graceful-shutdown drain: the whole point of tracing a
    2,000-page run is that the spans for it exist, and a ``BatchSpanProcessor``
    holds the most recent ones in memory until it flushes. Without this, a
    SIGTERM loses exactly the trace an operator is looking for.
    """
    global _provider

    with _lock:
        provider = _provider
        _provider = None
    if provider is None:
        return
    try:
        provider.shutdown()
    except Exception as exc:  # noqa: BLE001 - shutdown must not raise
        logger.warning(
            "flushing traces on shutdown failed: %s",
            exc,
            extra={"phase": "tracing", "status": "error", "error_class": type(exc).__name__},
        )
        return
    logger.info(
        "tracing shut down (buffered spans flushed)",
        extra={"phase": "tracing", "status": "stopped"},
    )


# ---------------------------------------------------------------------- spans


def _get_tracer() -> Any | None:
    if _provider is None:
        setup_tracing()
    provider = _provider
    if provider is None or _otel_trace is None:
        return None
    # Deliberately this process's provider rather than the global one: the SDK
    # refuses to replace a global provider, so re-configuring (tests, or a
    # changed endpoint) would otherwise silently keep exporting to the old one.
    return provider.get_tracer("app")


def current_span_context() -> Any | None:
    """The active span's context, for passing a parent into another thread.

    Returns ``None`` when tracing is off or no span is recording, so callers can
    hand it straight to :func:`phase_span` without a conditional.
    """
    if _provider is None or _otel_trace is None:
        return None
    try:  # a broken vendor SDK must not break a run
        ctx = _otel_trace.get_current_span().get_span_context()
    except Exception:  # noqa: BLE001
        return None
    return ctx if ctx is not None and getattr(ctx, "is_valid", False) else None


@contextmanager
def use_parent(span_context: Any | None) -> Iterator[None]:
    """Adopt *span_context* as this thread's parent for nested spans.

    Needed because pipeline work runs in a ``ThreadPoolExecutor``: the run's span
    is not in the worker thread's context, so without this every span created
    there (a chunk span, an LLM-call span) would start a brand-new trace.
    """
    if span_context is None or _provider is None:
        yield
        return
    token: Any | None = None
    try:
        token = _otel_attach(_set_span_in_context(_NonRecordingSpan(span_context)))
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not adopt parent span: %s", exc)
        yield
        return
    try:
        yield
    finally:
        with suppress(Exception):
            _otel_detach(token)


@contextmanager
def phase_span(name: str, *, enabled: bool = True, **attrs: Any) -> Iterator[SpanLike]:
    """Time one pipeline phase as a span. No-op when tracing is off.

    ``request.id`` is attached automatically so a span can be matched to its log
    lines, and *attrs* are screened by :func:`_screen` before they reach the SDK.

    *enabled* lets a call site that runs in a hot loop (one span per record chunk)
    opt out without a conditional at the call site.
    """
    tracer = _get_tracer() if enabled else None
    if tracer is None:
        yield _NULL_SPAN
        return

    attributes = _screen(attrs)
    request_id = _current_request_id()
    if request_id:
        attributes.setdefault("request.id", request_id)
    try:
        manager = tracer.start_as_current_span(name)
        span = manager.__enter__()
    except Exception as exc:  # noqa: BLE001 - never let telemetry break a run
        logger.debug("could not start span %s: %s", name, exc)
        yield _NULL_SPAN
        return
    if attributes:
        _set_attributes(span, attributes)
    try:
        yield span
    except BaseException as exc:
        # The span context manager records the exception and sets ERROR status on
        # exit; annotate the type so dashboards can group by it, then re-raise.
        _annotate_error(span, exc)
        _exit_span(manager, exc)
        raise
    else:
        _exit_span(manager, None)


@contextmanager
def run_span(action: str, **attrs: Any) -> Iterator[SpanLike]:
    """The root span for one Evaluate/Draft run (``run:evaluate``)."""
    with phase_span(f"run:{action}", **attrs) as span:
        yield span


def llm_call_span(
    phase: str, *, model: str = "", attempt: int = 1, endpoint: str = ""
) -> AbstractContextManager[SpanLike]:
    """A span for one LLM provider call, or a no-op when that is switched off.

    Returns a context manager rather than being a generator so ``app/llm.py`` can
    use it in a ``with`` statement around the network call. ``endpoint`` records
    which of the configured endpoints served the call (``primary``/``fallback``),
    so a trace from a failed-over run is readable without cross-referencing logs.
    """
    return phase_span(
        f"llm:{phase}",
        enabled=bool(config.TRACING_LLM_CALLS),
        model=model,
        attempt=attempt,
        endpoint=endpoint,
    )


@contextmanager
def attach_trace_context(carrier: Mapping[str, str] | None) -> Iterator[None]:
    """Continue the trace described by *carrier* (a queued job's payload).

    Used by the worker so its spans are children of the web pod's submit span.
    A payload without a carrier (tracing off at submit time, or an older web pod)
    simply starts a new trace.
    """
    if not carrier or _provider is None:
        yield
        return
    token: Any | None = None
    try:
        parent = _otel_extract(dict(carrier))
        token = _otel_attach(parent)
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not adopt trace context %r: %s", sorted(carrier), exc)
        yield
        return
    try:
        yield
    finally:
        try:
            _otel_detach(token)
        except Exception:  # noqa: BLE001
            pass


def inject_trace_context() -> dict[str, str]:
    """W3C ``traceparent``/``tracestate`` for the active span, or ``{}``.

    Returns empty when no span is recording, so a payload only grows when there
    is genuinely a trace to continue.
    """
    if _provider is None or _otel_trace is None:
        return {}
    try:
        if not _otel_trace.get_current_span().get_span_context().is_valid:
            return {}
        carrier: dict[str, str] = {}
        _otel_inject(carrier)
        return carrier
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not inject trace context: %s", exc)
        return {}


def tracing_health() -> dict[str, Any]:
    """Tracing status for ``GET /health`` and the sidebar ops panel.

    Deliberately does no network I/O: ``/health`` is polled by probes, and a
    blocked connect to a down collector would make the pod look unready.
    """
    return {
        "enabled": bool(config.TRACING_ENABLED),
        "active": is_active(),
        "exporter": os.getenv("OTEL_TRACES_EXPORTER", "").strip().lower() or "otlp-http",
        "endpoint": exporter_endpoint(),
        "service_name": config.TRACING_SERVICE_NAME,
        "role": _active_role or config.TRACING_ROLE,
        "sample_ratio": config.TRACING_SAMPLE_RATIO,
        "chunk_spans": bool(config.TRACING_CHUNK_SPANS),
        "llm_call_spans": bool(config.TRACING_LLM_CALLS),
        "reason": unavailable_reason(),
    }


def reset_tracing_for_tests() -> None:
    """Drop the provider so tests can reconfigure tracing from scratch."""
    global _provider, _setup_error, _active_role

    with _lock:
        provider = _provider
        _provider = None
        _setup_error = None
        _active_role = ""
    if provider is not None:
        try:
            provider.shutdown()
        except Exception:  # noqa: BLE001
            pass
    _dropped_attr_keys.clear()


# -------------------------------------------------------------------- helpers


def _current_request_id() -> str:
    try:
        from .logging_config import get_request_id

        return get_request_id() or ""
    except Exception:  # noqa: BLE001
        return ""


def _annotate_error(span: SpanLike, exc: BaseException) -> None:
    """Tag the failing span with the error class so traces can be grouped by it.

    The exception itself (message + stack) is recorded by the span's own exit
    path, which is also what flips the span status to ERROR. Only the class name
    becomes an attribute, keeping the attribute set free of message text.
    """
    try:
        span.set_attribute("error.class", type(exc).__name__)
    except Exception:  # noqa: BLE001
        pass


def _set_attributes(span: SpanLike, attributes: Mapping[str, Any]) -> None:
    try:
        span.set_attributes(attributes)
    except Exception:  # noqa: BLE001
        pass


def _exit_span(manager: Any, exc: BaseException | None) -> None:
    try:
        if exc is None:
            manager.__exit__(None, None, None)
        else:
            # Suppress: the caller re-raises the original exception.
            manager.__exit__(type(exc), exc, exc.__traceback__)
    except Exception:  # noqa: BLE001
        pass


def _screen(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Drop free-text attributes and coerce the rest into OTel-safe values.

    Two guards, because span attributes leave the deployment and this app
    handles medical records: known free-text names (plus any ``*_text`` key) are
    removed outright, and values are trimmed to primitives so a caller cannot
    smuggle a document through a list.
    """
    safe: dict[str, Any] = {}
    for key, value in attrs.items():
        if value is None:
            continue
        lowered = key.lower()
        if lowered in _FORBIDDEN_ATTR_KEYS or lowered.endswith("_text"):
            if key not in _dropped_attr_keys:
                _dropped_attr_keys.add(key)
                logger.debug(
                    "span attribute %r dropped: free-text attributes are not traced (PII rule)",
                    key,
                )
            continue
        safe[key] = _coerce(value)
    return safe


def _coerce(value: Any) -> Any:
    if isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value[: _MAX_ATTR_CHARS]
    if isinstance(value, Path):
        return value.name
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_coerce(item) for item in list(value)[:_MAX_LIST_ATTR_ITEMS]]
        return [item for item in items if isinstance(item, bool | int | float | str)]
    return type(value).__name__
