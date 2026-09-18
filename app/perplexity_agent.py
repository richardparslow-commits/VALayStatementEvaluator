"""Web-grounded research through the Perplexity Agent API.

Why this is its own module rather than another endpoint inside ``app/llm.py``: the
two speak different schemas, for different jobs.

``app.llm.LLMClient`` is an OpenAI **Chat Completions** client
(``chat.completions.create``) and carries the adjudication pipeline's hardened call
path — per-endpoint circuit breakers, a concurrency limiter sized to a credit quota,
retry/backoff, per-call failover (ADR-012), and usage split across a cheap/heavy model
pair. Every one of those exists to make *hundreds* of cheap structured extraction calls
survive rate limits without burning a metered budget.

The Agent API is **Responses**-shaped, web-grounded, and billed per model token *and per
tool invocation*. Its value here is the opposite shape of work: a handful of
citation-bearing lookups where the live web is the whole point. The bulk of this app
reads medical records the user uploaded, and none of that is on the web — so grounding
the digest path would add tool-call spend to every chunk with no chance of improving it
(see ARCHITECTURE.md §4, "Why hierarchical fact merging instead of one mega-call").

This module is therefore additive. Nothing in the evaluate/draft pipeline imports it, and
with no API key or no SDK the app behaves exactly as it did before.

Two published constraints shape the code below:

1. **Never ask the model for URLs inside structured output.** The docs are explicit that
   a model emitting links as part of a JSON schema can produce malformed or fabricated
   ones, and that citations should be read from the response's own ``search_results`` /
   ``fetch_url_results`` items. :func:`_collect_citations` is accordingly the only source
   of links here, and the schemas this module builds never contain a URL field.
2. **Imports of the SDK are function-local, and a missing package is a configuration
   problem.** ``perplexityai`` *is* in the deployed dependency set (``requirements.txt``
   and the hash-pinned ``requirements.lock``), because a host that installs from
   ``requirements.txt`` — Streamlit Community Cloud — has no shell and could otherwise
   never switch this feature on. The graceful-degradation contract is kept anyway, for
   the same reason ``app/blob_store.py`` and ``app/audit_backup.py`` keep it for boto3:
   an install can legitimately be partial (a slim image, a contributor's venv, a future
   decision to drop a provider), and the failure has to read as *"here is what is
   missing"* rather than an ``ImportError`` traceback at import time. See
   ``requirements-perplexity.txt`` for the standalone install and the cost controls.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from .circuit_breaker import get_llm_breaker
from .config import PERPLEXITY_PRESETS, Settings
from .logging_config import get_logger, get_request_id
from .pipeline_guard import check_pipeline_cancelled

logger = get_logger("app.perplexity_agent")

# One breaker per endpoint (ADR-012). Perplexity is a distinct provider from the
# adjudication gateway, so it needs its own name: sharing the primary's breaker would
# let a Perplexity outage trip the pipeline, and vice versa.
BREAKER_NAME = "perplexity-agent"

API_KEY_ENV = "PERPLEXITY_API_KEY"
SDK_PACKAGE = "perplexity"
SDK_REQUIREMENTS_FILE = "requirements-perplexity.txt"

# Attempts per logical call (the breaker above counts *logical* failures, not attempts,
# matching app/llm.py). Transient statuses are retried; a 4xx that is not 429 is not,
# because a malformed request will fail identically three times.
MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 1.5
# Ceiling on a server-supplied Retry-After. A research panel is an interactive click, so
# waiting minutes for one answer is worse than reporting the rate limit and stopping.
MAX_RETRY_AFTER_SECONDS = 30.0
# Bounded because a wide-research run can return hundreds of hits and the UI renders all
# of them; the answer text is where the substance lives.
MAX_CITATIONS = 40


class PerplexityError(RuntimeError):
    """Base class for Agent API failures (mirrors app.llm.LLMError's role)."""


class PerplexityConfigurationError(PerplexityError):
    """The integration is not usable: no key, or the optional SDK is not installed."""


class PerplexityUpstreamError(PerplexityError):
    """The API was reached but the call failed (rate limit, auth, 5xx, transport)."""

    def __init__(
        self,
        message: str,
        *,
        retriable: bool = False,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retriable = retriable
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class PerplexityParseError(PerplexityError):
    """A structured-output call returned text that is not valid JSON."""


# --------------------------------------------------------------------- availability


def sdk_installed() -> bool:
    """Whether the optional ``perplexityai`` package can be imported.

    Checked with ``find_spec`` rather than a try/except import so asking the question
    has no side effect and costs nothing on every rerun of the research tab.
    """
    try:
        return importlib.util.find_spec(SDK_PACKAGE) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken/namespace-path weirdness
        return False


def unavailable_reason(settings: Settings) -> str | None:
    """Why the research panel cannot run, or ``None`` when it can.

    Returns user-facing text naming the *specific* missing piece, because "it didn't
    work" is not actionable when the remedies differ (install a package vs. set a key).
    The key is only ever checked for presence — never read, printed, or logged.
    """
    if not settings.perplexity_configured:
        return (
            f"Perplexity research is not configured. Create an API key at "
            f"https://console.perplexity.ai and set {API_KEY_ENV} in your environment "
            f"or this project's .env file (which is git-ignored), then restart the app. "
            f"One Perplexity key serves both APIs, so when the base URL is Perplexity's, "
            f"the same key as OPENAI_API_KEY is used automatically."
        )
    if not sdk_installed():
        return (
            f"Perplexity research needs the official SDK, which is an optional install. "
            f"Run:  pip install -r {SDK_REQUIREMENTS_FILE}"
        )
    return None


def configured_preset(settings: Settings) -> str:
    """The preset to send, guarded against an unknown name.

    ``app.config`` already validates ``PERPLEXITY_PRESET``, but a ``Settings`` object can
    also be built or edited by hand (the sidebar mutates it in place), so the value is
    re-checked at the call site rather than trusted.
    """
    preset = settings.perplexity_preset.strip()
    return preset if preset in PERPLEXITY_PRESETS else PERPLEXITY_PRESETS[0]


# ------------------------------------------------------------------------ results


@dataclass(frozen=True)
class Citation:
    """One source the answer was grounded in, taken from the response, never the model."""

    url: str
    title: str = ""
    date: str | None = None
    snippet: str = ""

    def label(self) -> str:
        """A one-line label for the UI (falls back to the URL when untitled)."""
        return self.title.strip() or self.url


@dataclass
class GroundedAnswer:
    """A completed research call: prose, its sources, and the call's own metadata."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    # Parsed structured output when a JSON schema was requested, else None.
    findings: dict[str, Any] | None = None
    preset: str | None = None
    model: str | None = None
    response_id: str = ""
    latency_ms: int = 0
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def citation_count(self) -> int:
        return len(self.citations)


# ------------------------------------------------------------------- tool helpers


def web_search_tool(domains: list[str] | None = None, *, context_size: str = "medium") -> dict[str, Any]:
    """Build the ``web_search`` tool entry, optionally domain-restricted.

    Restricting to primary sources is what makes an answer *checkable* rather than merely
    plausible, so the panel offers it as a toggle; ``search_domain_filter`` is the
    documented filter name. An empty domain list means the whole web, which is honest —
    sending an empty list is not the same as sending no filter, so it is omitted.
    """
    tool: dict[str, Any] = {"type": "web_search", "search_context_size": context_size}
    cleaned = [d.strip() for d in (domains or []) if d.strip()]
    if cleaned:
        tool["filters"] = {"search_domain_filter": cleaned}
    return tool


def condition_audit_schema() -> dict[str, Any]:
    """JSON schema for a condition audit, deliberately containing **no URL fields**.

    The docs warn that links requested inside structured output can come back malformed
    or fabricated; sources are read from the response's ``search_results`` items instead.
    Any field that must be cited is therefore a claim plus a short source *title*, with
    the link resolvable from the response-level citation list.

    ``additionalProperties: False`` and an explicit ``required`` list are set because the
    documented examples do the same and it materially improves adherence.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "condition_audit",
            "schema": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string"},
                    "summary": {"type": "string"},
                    "rating_criteria": {"type": "string"},
                    "presumptive": {
                        "type": "string",
                        "description": (
                            "Whether the condition is presumptive (including PACT Act or "
                            "other qualifying exposures), and under what authority."
                        ),
                    },
                    "evidence_expectations": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "lay_observations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Observations a lay witness could competently describe for this "
                            "condition (functional impact), never diagnoses or causation."
                        ),
                    },
                    "framework_findings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "topic": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["current", "changed", "unclear"],
                                },
                                "note": {"type": "string"},
                            },
                            "required": ["topic", "status", "note"],
                        },
                        "description": (
                            "How this condition maps onto the app's committed topic "
                            "checklist, and whether that mapping still looks current."
                        ),
                    },
                    "caveats": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["condition", "summary", "rating_criteria"],
                "additionalProperties": False,
            },
        },
    }


def framework_currency_schema(topics: Sequence[str]) -> dict[str, Any]:
    """JSON schema for a per-topic currency verdict over the committed checklist.

    ``topics`` are the checklist letters actually being reviewed, and they become the
    ``enum`` for the ``topic`` field. That constraint is the point: it makes it
    impossible to get back a verdict about a topic that was not sent, which is what keeps
    a stored report interpretable later — every verdict maps onto text this app has.

    ``authority`` is a *short title* ("38 C.F.R. § 4.130"), never a URL, for the same
    reason :func:`condition_audit_schema` asks for none: the docs warn that links
    requested inside structured output can come back malformed or fabricated, and an
    audit whose citation cannot be trusted is worse than one with no citation. Real
    sources come from the response's own ``search_results`` items.
    """
    letters = [str(topic).strip().upper() for topic in topics if str(topic).strip()]
    if not letters:
        raise ValueError("framework_currency_schema needs at least one topic letter")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "framework_currency",
            "schema": {
                "type": "object",
                "properties": {
                    "verdicts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "topic": {"type": "string", "enum": letters},
                                "status": {
                                    "type": "string",
                                    "enum": ["current", "changed", "unclear"],
                                    "description": (
                                        "current = the committed text still matches current "
                                        "law; changed = current law differs or the text now "
                                        "misleads; unclear = primary sources conflict or "
                                        "you could not confirm."
                                    ),
                                },
                                "note": {
                                    "type": "string",
                                    "description": (
                                        "What changed, or why the text is still accurate. "
                                        "Concrete, and specific to the topic text below."
                                    ),
                                },
                                "authority": {
                                    "type": "string",
                                    "description": (
                                        "Short title of the controlling source, e.g. "
                                        "'38 C.F.R. § 4.130'. Never a URL."
                                    ),
                                },
                            },
                            "required": ["topic", "status", "note", "authority"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["verdicts"],
                "additionalProperties": False,
            },
        },
    }


# ------------------------------------------------------------------ extraction


def _as_str(value: Any) -> str:
    """Coerce an SDK field to ``str`` (empty for None) without leaking ``Any``."""
    return value.strip() if isinstance(value, str) else ""


def _field(obj: Any, name: str) -> Any:
    """Read a field from an SDK response object *or* a plain mapping.

    The SDK is not consistent about this, and the inconsistency is not theoretical: with
    identical request shapes, the rows inside a ``search_results`` item have been observed
    coming back as typed pydantic models in one response and as plain ``dict``s in another
    (``output`` items themselves stay typed). Reading citations with a bare ``getattr``
    therefore returns ``None`` for every row and yields an *empty source list rather than
    an error* — the worst possible failure for a feature whose entire purpose is showing
    where an answer came from, because an unsourced answer looks like a confident one.
    Handling both shapes here keeps extraction independent of which one arrives.
    """
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _items(response: Any) -> list[Any]:
    """The response's ``output`` list, or an empty list when absent/unexpected."""
    output = _field(response, "output")
    return list(output) if isinstance(output, (list, tuple)) else []


def _collect_citations(response: Any) -> list[Citation]:
    """Sources from the response's own tool-result items, deduplicated by URL.

    Reads all three shapes the docs describe: ``search_results`` (each carrying
    ``results``), ``fetch_url_results`` (each carrying ``contents``), and URL annotations
    attached to a text block. Deduplication is by URL because the same source is
    routinely returned by two queries in one run, and a repeated link in a "sources" list
    reads as two independent confirmations.
    """
    found: list[Citation] = []
    seen: set[str] = set()

    def add(url: Any, title: Any, date: Any, snippet: Any) -> None:
        cleaned = _as_str(url)
        if not cleaned or cleaned in seen or len(found) >= MAX_CITATIONS:
            return
        seen.add(cleaned)
        found.append(
            Citation(
                url=cleaned,
                title=_as_str(title),
                date=_as_str(date) or None,
                snippet=_as_str(snippet)[:400],
            )
        )

    for item in _items(response):
        item_type = _as_str(_field(item, "type"))
        if item_type == "search_results":
            for row in _field(item, "results") or []:
                add(
                    _field(row, "url"),
                    _field(row, "title"),
                    _field(row, "date"),
                    _field(row, "snippet"),
                )
        elif item_type == "fetch_url_results":
            for row in _field(item, "contents") or []:
                add(
                    _field(row, "url"),
                    _field(row, "title"),
                    None,
                    _field(row, "snippet"),
                )
        elif item_type == "message":
            for block in _field(item, "content") or []:
                for ann in _field(block, "annotations") or []:
                    add(
                        _field(ann, "url"),
                        _field(ann, "title"),
                        _field(ann, "date"),
                        None,
                    )
    return found


def _usage_summary(response: Any) -> dict[str, Any]:
    """A small, JSON-safe usage record (token counts and cost when reported).

    Kept deliberately narrow: this lands in the audit stream, whose contract is counts
    and classifications only.
    """
    usage = _field(response, "usage")
    if usage is None:
        return {}
    summary: dict[str, Any] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = _field(usage, key)
        if isinstance(value, int):
            summary[key] = value
    # Per-call cost is reported by the Agent API and includes the per-tool-invocation
    # charges, which is the part a caller cannot infer from token counts alone.
    total_cost = _field(_field(usage, "cost"), "total_cost")
    if isinstance(total_cost, (int, float)):
        summary["total_cost_usd"] = round(float(total_cost), 6)
    return summary


def _parse_findings(text: str) -> dict[str, Any]:
    """Parse a structured-output response body into a dict."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PerplexityParseError(
            f"The research response was not valid JSON (chars={len(text)})."
        ) from exc
    if not isinstance(parsed, dict):
        raise PerplexityParseError("The research response was valid JSON but not an object.")
    return parsed


# ------------------------------------------------------------------ error mapping


def _status_code(exc: BaseException) -> int | None:
    """HTTP status from an SDK exception, when it exposes one."""
    for attr in ("status_code", "http_status", "code"):
        value = _field(exc, attr)
        if isinstance(value, int):
            return value
    value = _field(_field(exc, "response"), "status_code")
    if isinstance(value, int):
        return value
    return None


def _retry_after_seconds(exc: BaseException) -> float | None:
    """``Retry-After`` from a rate-limit response, in seconds, clamped.

    The docs direct callers to honor ``Retry-After`` on a 429. The header may also be an
    HTTP date, which is only used when it parses to a sane positive number, and the value
    is clamped so a large server-supplied wait degrades into a clear message instead of
    hanging an interactive panel.
    """
    headers = _field(_field(exc, "response"), "headers")
    raw: Any = None
    if headers is not None:
        try:
            raw = headers.get("retry-after") or headers.get("Retry-After")
        except Exception:  # noqa: BLE001 - a mapping that insists on str keys
            raw = None
    if raw is None:
        raw = _field(exc, "retry_after")
    if raw is None:
        return None
    try:
        seconds = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return min(seconds, MAX_RETRY_AFTER_SECONDS)


def _is_transient(exc: BaseException, status: int | None) -> bool:
    """Whether re-sending the identical request could plausibly succeed.

    Rates limit and server faults qualify; a 4xx that is not 429 does not, because a
    malformed or unauthorized request fails the same way three times and turns one error
    into three log lines.
    """
    name = type(exc).__name__
    if "timeout" in name.lower() or "connection" in name.lower():
        return True
    if status is None:
        # No status at all means the failure happened before a response existed
        # (DNS, TLS, transport) — retriable.
        return True
    if status == 429 or status >= 500:
        return True
    return False


def _normalize_error(exc: Exception) -> PerplexityUpstreamError:
    """Turn an SDK exception into this module's error type, with an actionable message."""
    status = _status_code(exc)
    if status == 401:
        message = (
            f"Perplexity rejected the API key (401). Check {API_KEY_ENV}; if it was ever "
            "exposed, rotate it at https://console.perplexity.ai."
        )
    elif status == 403:
        message = "Perplexity refused the request (403) — the key may lack access to this API."
    elif status == 429:
        message = "Perplexity rate limit reached (429). Wait and try again."
    elif status is not None and status >= 500:
        message = f"Perplexity reported a server error ({status})."
    else:
        message = f"Perplexity request failed: {type(exc).__name__}."
    return PerplexityUpstreamError(
        message,
        retriable=_is_transient(exc, status),
        status_code=status,
        retry_after_seconds=_retry_after_seconds(exc),
    )


# ------------------------------------------------------------------------- call


def _timeout_seconds() -> float:
    """Per-call timeout, reusing ``VA_LSE_LLM_CALL_TIMEOUT_SECONDS``.

    Deliberately the same knob as ``app/llm.py``: one deployment-wide setting should
    govern how long any single provider call may hang, so an operator does not have to
    discover a second variable that only one provider honours.
    """
    try:
        return max(1.0, float(os.getenv("VA_LSE_LLM_CALL_TIMEOUT_SECONDS", "") or 300.0))
    except ValueError:
        return 300.0


def _build_client(api_key: str) -> Any:
    """Instantiate the official SDK client (imported here, not at module scope).

    The key is passed into the constructor and never stored on this module, logged, or
    included in an exception message. The import is local so the module stays importable
    — and testable — without the optional package.

    ``max_retries=0`` for the same reason ``app/llm.py`` sets it: the SDK's internal
    retries are invisible to this app's cancellation checks, and they would multiply with
    the retry loop in :func:`research`, turning one logical failure into several attempts
    against a rate-limited API that is already asking us to slow down.
    """
    from perplexity import Perplexity  # noqa: PLC0415 - optional dependency, by design

    return Perplexity(api_key=api_key, timeout=_timeout_seconds(), max_retries=0)


def research(
    question: str,
    *,
    settings: Settings,
    instructions: str = "",
    domains: list[str] | None = None,
    schema: dict[str, Any] | None = None,
    context_size: str = "medium",
) -> GroundedAnswer:
    """Run one web-grounded research request and return the answer plus its sources.

    ``schema`` is a ``response_format`` object (see :func:`condition_audit_schema`); when
    given, the response body is parsed into :attr:`GroundedAnswer.findings`.

    Gated by this provider's own circuit breaker and the pipeline cancellation check, and
    retried only on transient failures. Raises :class:`PerplexityConfigurationError` when
    the integration is not usable, ``CircuitBreakerOpenError`` (from app/circuit_breaker)
    when the breaker is open — deliberately propagated untouched, as ``app/llm.py`` does,
    so callers can fail fast with their own wording — and
    :class:`PerplexityUpstreamError` / :class:`PerplexityParseError` otherwise.
    """
    check_pipeline_cancelled()
    reason = unavailable_reason(settings)
    if reason is not None:
        raise PerplexityConfigurationError(reason)

    prompt = question.strip()
    if not prompt:
        raise PerplexityConfigurationError("Enter a research question first.")

    breaker = get_llm_breaker(BREAKER_NAME)
    breaker.check_or_raise()

    client = _build_client(settings.perplexity_api_key.strip())
    request: dict[str, Any] = {
        "input": prompt,
        "tools": [web_search_tool(domains, context_size=context_size)],
        "max_output_tokens": max(1, settings.perplexity_max_output_tokens),
    }
    # A preset supplies model, tools, and limits, and is the documented way to inherit
    # Perplexity's improvements without a code change. An explicit PERPLEXITY_MODEL
    # freezes the model while keeping the rest of the preset's configuration.
    preset = configured_preset(settings)
    request["preset"] = preset
    model_override = settings.perplexity_model.strip()
    if model_override:
        request["model"] = model_override
    if instructions.strip():
        request["instructions"] = instructions.strip()
    if schema is not None:
        request["response_format"] = schema

    started = time.monotonic()
    last_error: PerplexityUpstreamError | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        check_pipeline_cancelled()
        try:
            response = client.responses.create(**request)
        except Exception as exc:  # noqa: BLE001 - SDK error types are mapped below
            mapped = _normalize_error(exc)
            last_error = mapped
            if not mapped.retriable or attempt >= MAX_ATTEMPTS:
                breaker.record_failure()
                _log_failure(mapped, phase="research", attempts=attempt, preset=preset)
                raise mapped from exc
            wait = mapped.retry_after_seconds or _backoff_seconds(attempt)
            logger.warning(
                "perplexity: transient failure, retrying in %.1fs",
                wait,
                extra={
                    "request_id": get_request_id() or "-",
                    "phase": "research",
                    "status": "retry",
                    "attempt": attempt,
                    "error_class": type(exc).__name__,
                },
            )
            # Cancellation-aware wait: a server-supplied Retry-After can be tens of
            # seconds, and a shutdown request during it must still be honoured. Counted
            # in elapsed local steps rather than against a wall-clock deadline so the
            # loop always terminates even if time.sleep is patched (tests).
            waited = 0.0
            while waited < wait:
                check_pipeline_cancelled()
                step = min(0.25, wait - waited)
                time.sleep(step)
                waited += step
            continue

        answer = _answer_from_response(response, preset=preset, started=started)
        breaker.record_success()
        logger.info(
            "perplexity research ok",
            extra={
                "request_id": get_request_id() or "-",
                "phase": "research",
                "status": "ok",
                "duration_ms": answer.latency_ms,
                "preset": preset,
                "model": answer.model or "-",
                "citations": answer.citation_count,
                "structured": schema is not None,
            },
        )
        return answer

    assert last_error is not None  # pragma: no cover - loop always returns or raises
    raise last_error


def _answer_from_response(response: Any, *, preset: str, started: float) -> GroundedAnswer:
    """Assemble a :class:`GroundedAnswer` from an SDK response object."""
    text = _as_str(_field(response, "output_text"))
    status = _as_str(_field(response, "status"))
    if status and status != "completed":
        # An incomplete run still carries whatever text it produced, but presenting a
        # truncated research answer as final is exactly the kind of unverified claim this
        # app exists to prevent, so it is surfaced as a failure instead.
        raise PerplexityUpstreamError(
            f"The research run did not complete (status: {status}).",
            retriable=False,
        )
    findings = _parse_findings(text) if text.strip().startswith("{") else None
    return GroundedAnswer(
        text=text,
        citations=_collect_citations(response),
        findings=findings,
        preset=preset,
        model=_as_str(_field(response, "model")) or None,
        response_id=_as_str(_field(response, "id")),
        latency_ms=int((time.monotonic() - started) * 1000),
        usage=_usage_summary(response),
    )


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter, for the retry path with no server hint.

    ``2.0 ** n`` rather than ``2 ** n`` deliberately: mypy types an ``int ** int``
    expression as ``Any`` (a negative exponent could yield a complex number), which
    would silently untype everything downstream of ``base``.
    """
    base = BACKOFF_BASE_SECONDS * (2.0 ** (attempt - 1))
    return min(MAX_RETRY_AFTER_SECONDS, base) * (0.75 + random.random() * 0.5)


def _log_failure(
    error: PerplexityUpstreamError, *, phase: str, attempts: int, preset: str
) -> None:
    """Log a terminal research failure with counts only — never the question or answer."""
    logger.warning(
        "perplexity research failed: %s",
        error,
        extra={
            "request_id": get_request_id() or "-",
            "phase": phase,
            "status": "error",
            "attempts": attempts,
            "preset": preset,
            "status_code": error.status_code,
        },
    )
