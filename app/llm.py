"""OpenAI-compatible LLM client with retries, JSON-mode helpers, circuit breaker and concurrency limiting."""

from __future__ import annotations

import json
import logging
import random
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, cast, runtime_checkable
from urllib.parse import urlparse

from . import tracing
from . import metrics
from .circuit_breaker import (
    CircuitBreakerOpenError,
    QueueFullError,
    RateGateTimeoutError,
    get_llm_breaker,
    get_llm_limiter,
    get_llm_rate_gate,
)
from .config import (
    DEFAULT_BASE_URL,
    FALLBACK_ENDPOINT,
    PRIMARY_ENDPOINT,
    PERPLEXITY_HOST,
    Settings,
)
from .logging_config import get_request_id
from .pipeline_guard import (
    check_pipeline_cancelled, pipeline_remaining_seconds, wait_with_cancellation,
)
from .prompt_sanitize import validate_model_name
from .usage import UsageTracker

try:
    import httpx as _httpx  # noqa: F401  # optional; only for TimeoutException isinstance check

    _HttpxTimeoutError: tuple[type[BaseException], ...] = (_httpx.TimeoutException,)
except ImportError:  # pragma: no cover - httpx not always installed in tests
    _HttpxTimeoutError = ()

logger = logging.getLogger("app.llm")

# Static-only names from the OpenAI SDK: at runtime the SDK is imported
# lazily (module ``__getattr__`` below) on the first client construction or
# ``NOT_GIVEN`` use. Importing this module must not pay the SDK's import
# floor — measured 2026-09-21 at ~32 MB of peak RSS on the import ladder
# (.freebuff/profile/_memprofile.py --ladder: the ``import app.llm`` rung
# dropped 103.2 -> 70.9 MB) — because every launch pays that floor while only
if TYPE_CHECKING:
    from openai import NOT_GIVEN, OpenAI


def _sdk_name(name: str) -> Any:
    """Resolve ``OpenAI``/``NOT_GIVEN`` through the module global, lazily.

    The module global is consulted FIRST, not last: an active
    ``patch("app.llm.OpenAI")`` must win, exactly as it did when the import
    was eager. Only when the global is unbound does this import the SDK and
    cache the real object, so subsequent lookups are ordinary and free.
    Internal ``LOAD_GLOBAL`` lookups never reach the module ``__getattr__``
    (PEP 562 serves external attribute access only), so runtime use sites go
    through this helper instead. Returns ``Any`` on purpose: the cached value
    is whatever the SDK (or a test patch) put in the global slot, and callers
    that hand it to typed surfaces cast at their own call site.
    """
    value = globals().get(name)
    if value is None:
        import openai

        value = getattr(openai, name)
        globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    """Resolve ``OpenAI`` / ``NOT_GIVEN`` from the SDK on first use (PEP 562).

    This serves *external* attribute access — ``from app.llm import OpenAI``,
    ``patch("app.llm.OpenAI")``'s getattr, ``app.llm.NOT_GIVEN``. Internal
    use sites call :func:`_sdk_name` instead. The resolved value is written
    back into the module globals, so lookups after the first are ordinary
    (fast) and the patch/restore cycle behaves exactly as it did when the
    import was eager.
    """
    if name in ("OpenAI", "NOT_GIVEN"):
        return _sdk_name(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Re-export for callers that want to catch these specifically.
__all__ = [
    "LLMClient",
    "LLMService",
    "LLMError",
    "LLMConfigurationError",
    "LLMUpstreamError",
    "LLMTimeoutError",
    "LLMParseError",
    "CircuitBreakerOpenError",
    "QueueFullError",
    "RateGateTimeoutError",
    "ModelProbe",
    "ChatProbe",
    "check_model_availability",
    "probe_models",
    "probe_chat",
]

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.0
MAX_RETRY_BACKOFF_SECONDS = 4.0
MODELS_ENDPOINT_TIMEOUT_SECONDS = 6

# The short chat call the preflight makes after the model listing. A two-character
# prompt, so the request cannot be mistaken for a run — and a small-but-not-one-token
# budget, because a reasoning model spends tokens thinking before it writes anything:
# at max_tokens=1 the measured main model (alibaba/qwen3.7-flash, Vercel AI Gateway)
# answered 200 with an empty message, finish_reason "length" and 230 reasoning tokens,
# which reads as "the endpoint answered, but not usefully" when the endpoint is fine.
# The ceiling is longer than the listing's, because a listing is a database read while
# a call has to reach a model (and wake a cold one): a 6-second ceiling would report
# "no response" for an endpoint that is merely slow, and the run would start unverified.
CHAT_PROBE_MAX_TOKENS = 64
CHAT_PROBE_PROMPT = "hi"
CHAT_PROBE_TIMEOUT_SECONDS = 20

# --------------------------------------------------------------- endpoints
#
# "Endpoint" means one configured (base_url, api_key, model names) triple. There
# is exactly one unless OPENAI_BASE_URL_FALLBACK is set, in which case calls can
# be served by either. See `failover_status` for the live state and
# `_endpoint_candidates` for the routing rule. The names live in config so the
# usage record and the queued-job payload use the same strings.


def _endpoint_breaker_name(endpoint: str) -> str:
    """The circuit-breaker name for an endpoint (also its metrics label)."""
    from .circuit_breaker import LLM_BREAKER_NAME, LLM_FALLBACK_BREAKER_NAME

    return LLM_FALLBACK_BREAKER_NAME if endpoint == FALLBACK_ENDPOINT else LLM_BREAKER_NAME


class _FallbackTarget(NamedTuple):
    """The resolved failover endpoint: URL, key, and the two model names."""

    base_url: str
    api_key: str
    model_main: str
    model_fast: str

    @property
    def configured(self) -> bool:
        return bool(self.base_url)


def _as_text(raw: Any) -> str:
    """A configuration value as a stripped string, or "" if it is not a string.

    Only a ``str`` counts. A non-string is not a URL, key, or model name, and
    treating one as if it were (``str(mock)`` is non-empty) would make an object
    that merely *has* such an attribute look like a configured second endpoint.
    """
    return raw.strip() if isinstance(raw, str) else ""


def _fallback_target(settings: Settings) -> _FallbackTarget:
    """Resolve the failover endpoint from *settings*, tolerating settings without one.

    Read through ``getattr``/``hasattr`` because this is reached from the
    ``LLMClient`` constructor, which test doubles and older callers also invoke: an
    object that declares no fallback URL has no fallback, which is exactly the
    behaviour before this feature existed, rather than an ``AttributeError``. The
    ``Settings`` accessors are preferred when present so the inheritance rules
    ("an unset fallback key/model means the primary's") live in one place.
    """
    base_url = _as_text(getattr(settings, "fallback_base_url", ""))
    if hasattr(settings, "fallback_api_key_or_primary"):
        api_key = _as_text(settings.fallback_api_key_or_primary())
        model_main = _as_text(settings.fallback_model_main_or_primary())
        model_fast = _as_text(settings.fallback_model_fast_or_primary())
    else:
        api_key = _as_text(getattr(settings, "fallback_api_key", "")) or _as_text(
            getattr(settings, "api_key", "")
        )
        model_main = _as_text(getattr(settings, "fallback_model_main", "")) or _as_text(
            getattr(settings, "model_main", "")
        )
        model_fast = _as_text(getattr(settings, "fallback_model_fast", "")) or _as_text(
            getattr(settings, "model_fast", "")
        )
    return _FallbackTarget(base_url, api_key, model_main, model_fast)


def failover_after_seconds() -> float:
    """Grace period the primary must be unhealthy for before calls move over.

    Named for ``LLM_ENDPOINT_FALLBACK_TIMEOUT_SECONDS``; it is not an HTTP
    timeout. Returns a large number when config is unavailable, so a broken
    config fails *closed* (never failover) rather than failing over eagerly.
    """
    try:
        from . import config as _cfg

        return max(0.0, float(getattr(_cfg, "LLM_FAILOVER_AFTER_SECONDS", 300)))
    except Exception:  # noqa: BLE001
        return 300.0


def failover_status() -> dict[str, Any]:
    """Live failover state for /health and /metrics. Performs no I/O.

    Read from the primary's breaker rather than remembered anywhere: the breaker
    is the process-wide owner of "how long has this endpoint been unhealthy", and
    a client instance is rebuilt on every Streamlit rerun, so a client-side flag
    would forget an outage between reruns.
    """
    try:
        from . import config as _cfg

        configured = _fallback_target(_cfg.load_settings()).configured
    except Exception:  # noqa: BLE001
        configured = False

    from .circuit_breaker import get_llm_breaker

    threshold = failover_after_seconds()
    unhealthy = get_llm_breaker(_endpoint_breaker_name(PRIMARY_ENDPOINT)).unhealthy_for_seconds()
    active = bool(configured and unhealthy is not None and unhealthy >= threshold)
    return {
        "configured": configured,
        "active": active,
        "after_seconds": threshold,
        "primary_unhealthy_seconds": unhealthy,
    }

# ---------------------------------------------------------------- moderation filter
#
# Some OpenAI-compatible gateways (QwenCloud / Aliyun MaaS in particular) run a
# content-moderation filter over the MODEL OUTPUT and reject the whole call with
# HTTP 400 ``data_inspection_failed`` ("Output data may contain inappropriate
# content").  Medical/trauma/veteran-claim drafting trips this stochastically —
# identical prompts can pass one run and fail the next.  Two properties matter:
#
# 1. **Retry cannot help** for a deterministic-400 body like this, and blind
#    retries burn credits and add ~12 s of backoff before the user sees the
#    generic failure.  (We still retry *some* 400-shaped errors because
#    gateways vary; the moderation filter is matched by code/message.)
# 2. The error must surface as a clear, actionable message — not the generic
#    "Drafting failed" the UI used to show for every exception.
# 3. Because the filter samples *output* stochastically, one rephrased retry
#    (clinical-tone nudge appended to the user prompt) clears many runs that
#    would otherwise fail. See ``_moderation_nudge_user`` for the exact scope:
#    one attempt, small prompts only, input-side rejections never nudged.
#
# `_ModerationFilteredError` (defined below, after ``LLMError``) carries the
# original provider detail so logs keep the underlying cause without leaking
# it to the UI.
_MODERATION_MARKERS = (
    "data_inspection_failed",
    "inappropriate content",
    "content filter",
    "content moderation",
    "sensitive content",
    "output data may contain",
    "input data may contain",
)


def _error_status_code(exc: BaseException) -> int | None:
    """Best-effort HTTP status extraction from an OpenAI SDK exception."""
    status = _provider_status_code(exc)
    if status is not None:
        return status
    return None


def _is_moderation_filtered(exc: BaseException) -> bool:
    """True when the provider's content-moderation filter rejected the call.

    Matched by the gateway's error code/message (``data_inspection_failed``,
    "inappropriate content", …) rather than status alone — see the module
    comment above ``_MODERATION_MARKERS`` for why.
    """
    text = str(exc).lower()
    return any(marker in text for marker in _MODERATION_MARKERS)



# One output-filter rejection gets exactly one rephrased retry: the filter
# samples model *output* stochastically, so nudging the model toward a strictly
# clinical, factual tone clears many runs that would otherwise fail. Input-side
# rejections and oversized prompts skip the nudge — re-running a
# multi-hundred-kB record prompt to chase a filter flake is neither effective
# nor cheap, and a nudge cannot fix content the user supplied verbatim.
MODERATION_NUDGE_MAX_USER_CHARS = 4_000

_MODERATION_NUDGE_SUFFIX = (
    "\n\nTONE REQUIREMENT (provider content policy): use a strictly clinical, "
    "factual, respectful tone. Describe events and symptoms plainly without "
    "graphic, gory, or violent detail; omit vivid injury descriptions and keep "
    "wording neutral (e.g. 'sustained a head injury' rather than a graphic "
    "narrative). Keep every fact, name, date, and required element — this is a "
    "formal benefits document, not creative writing."
)


def _moderation_nudge_user(user: str) -> str | None:
    """Return the nudged prompt for one output-filter retry, or None.

    ``None`` means "do not nudge": the prompt is empty or too large for a
    rephrased retry to be worth a full re-run (large prompts are usually
    record-driven, i.e. the filter flagged supplied content, which a tone
    nudge cannot change).
    """
    if not user or len(user) > MODERATION_NUDGE_MAX_USER_CHARS:
        return None
    return user + _MODERATION_NUDGE_SUFFIX


class ModelProbe(NamedTuple):
    """Outcome of a best-effort ``GET {base_url}/models`` check.

    ``models`` is ``None`` whenever the check did not produce a list, and
    ``status``/``error`` say why. The status is the whole point: a rejected key
    (401), a path that does not exist (404), and a host that does not resolve are
    three different problems with three different fixes, and the previous
    ``set | None`` collapsed them into one sentence that named none of them.

    ``ok`` is defined on the presence of a model list, not on an HTTP status: a
    host that answers 200 with something that is not a model list has still
    failed this check.
    """

    models: set[str] | None
    status: int | None
    error: str

    @property
    def ok(self) -> bool:
        return self.models is not None


def probe_models(base_url: str, api_key: str) -> ModelProbe:
    """GET {base_url}/models, reporting the model ids *or* why there are none.

    Best-effort and never raises: callers decide how loud to be. Without an API
    key the check is not attempted. Uses only stdlib (urllib) so no extra deps.
    """
    if not api_key or not api_key.strip():
        return ModelProbe(None, None, "no API key was supplied")
    url = base_url.rstrip("/") + "/models"
    try:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key.strip()}"})
        with urllib.request.urlopen(req, timeout=MODELS_ENDPOINT_TIMEOUT_SECONDS) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - availability check is advisory only
        return ModelProbe(None, _http_status(exc), _probe_error_text(exc))
    rows = data.get("data", []) if isinstance(data, dict) else []
    ids = {
        row["id"].strip()
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"].strip()
    }
    if not ids:
        return ModelProbe(None, 200, f"{url} answered but published no model ids")
    return ModelProbe(ids, 200, "")


# ---------------------------------------------------------------- wire formats
#
# Two schemas share the ``LLMClient.chat`` signature. Which one an endpoint speaks
# is decided once, from its base URL:
#
# * **Responses** (Perplexity's Agent API, the default endpoint): POST
#   {base_url}/responses with ``input`` / ``instructions`` / ``max_output_tokens``
#   (https://docs.perplexity.ai/docs/agent-api/openai-compatibility). ``input``
#   accepts a plain string, which the service treats as a single user turn; the
#   system prompt travels in ``instructions``.
# * **Chat Completions** (every other OpenAI-compatible endpoint): POST
#   {base_url}/chat/completions with ``messages`` / ``max_tokens``.
#
# ``/v1/agent`` is the same service as ``/v1/responses``; the SDK alias is what a
# base URL of ``.../v1`` hits.

def _is_perplexity_base_url(base_url: str) -> bool:
    """True when *base_url* is any Perplexity host (main or fallback endpoint).

    The path is not inspected: ``.../v1`` and ``.../router/v1`` both name the same
    provider, and the Router host also serves Chat Completions, so only the host
    identifies it. Comparing registered domains (not ``in``) keeps a hostile lookalike
    like ``api.perplexity.ai.evil.test`` from being classed as Perplexity.
    """
    host = (urlparse(base_url.strip()).hostname or "").lower()
    if not host:
        return False
    return host == PERPLEXITY_HOST or host.endswith(f".{PERPLEXITY_HOST}")


def _head_route_exists(url: str) -> bool | None:
    """Whether a route exists at *url*, or ``None`` when the probe cannot tell.

    A ``HEAD`` is the cheapest existence question. The statuses that answer it
    were calibrated live against the working Agent API (2026-09-22):
    ``HEAD /responses`` → ``405`` (the route exists; HEAD is just not an allowed
    method on it), ``HEAD /chat/completions`` → ``404``, ``HEAD /nope`` → ``404``.
    So ``404/410`` is a definitive no and ``405`` a definitive yes; auth and rate
    statuses say nothing about the route's shape, and network errors say nothing
    at all — all of those return ``None`` rather than guessing.
    """
    try:
        req = urllib.request.Request(
            url, method="HEAD", headers={"Authorization": "Bearer probe"}
        )
        with urllib.request.urlopen(req, timeout=3) as resp:  # noqa: S310 - operator-configured endpoint
            return resp.status not in (404, 410)
    except urllib.error.HTTPError as exc:
        exc.close()
        if exc.code in (404, 410):
            return False
        if exc.code == 405:
            return True
        return None
    except (urllib.error.URLError, OSError, ValueError):
        return None


#: Resolved schema per base URL. Populated once per process per endpoint: the
#: probe is a startup-time cost and the answer does not change mid-run.
_SCHEMA_CACHE: dict[str, str] = {}
_SCHEMA_CACHE_LOCK = threading.Lock()


def _endpoint_schema(base_url: str) -> str:
    """``"responses"`` or ``"chat"`` — the wire schema *base_url* actually serves.

    The host classification stays as the fast, offline-safe guess, but it no
    longer has the final say: one cached HEAD probe asks the endpoint which
    routes it serves, and affirmative evidence from that probe outranks the
    host sniff. This is deliberate capability routing, not model-string
    sniffing — a provider may legally serve models under prefixed ids on its
    own host (Perplexity's ``perplexity/kimi-k3`` does), so the model name
    carries no routing signal, while the route table does.

    ``VA_LSE_LLM_ENDPOINT_SCHEMA`` (``responses`` | ``chat``) overrides the
    resolution entirely — the escape hatch for a proxy that mishandles HEAD
    probes, and the deterministic knob tests use.
    """
    override = (getattr(_config_module(), "LLM_ENDPOINT_SCHEMA", "") or "").strip().lower()
    if override in ("responses", "chat"):
        return override
    # Fast path: check cache without lock (read is atomic for dict.get)
    cached = _SCHEMA_CACHE.get(base_url)
    if cached:
        return cached
    # Slow path: acquire lock and re-check before probing
    with _SCHEMA_CACHE_LOCK:
        # Another thread may have populated the cache while we waited
        cached = _SCHEMA_CACHE.get(base_url)
        if cached:
            return cached
        guess = "responses" if _is_perplexity_base_url(base_url) else "chat"
        root = base_url.rstrip("/")
        responses_route = _head_route_exists(f"{root}/responses")
        chat_route = _head_route_exists(f"{root}/chat/completions")
        schema: str | None = None
        if responses_route and chat_route is False:
            schema = "responses"
        elif chat_route and responses_route is False:
            schema = "chat"
        elif responses_route and chat_route:
            # Both routes exist: the host's documented shape wins.
            schema = guess
        resolved = schema or guess
        _SCHEMA_CACHE[base_url] = resolved
        return resolved


def _config_module() -> Any:
    """The config module, imported lazily to keep the import ladder light."""
    from . import config as _cfg

    return _cfg


def _uses_responses_schema(base_url: str) -> bool:
    """Whether this endpoint takes the Responses wire schema.

    Resolution is capability-based (:func:`_endpoint_schema`); the host split
    only supplies the default guess when the endpoint's route table cannot be
    observed. A custom deployment that wants Chat Completions on a Perplexity
    host no longer needs the old IP/proxy workaround — the probe sees which
    routes exist — but the env override is there for anything the probe cannot
    see through.
    """
    return _endpoint_schema(base_url) == "responses"


def _responses_input(system: str, user: str) -> str:
    """Serialize the user turn into the stateless ``input`` string.

    The system prompt does NOT travel here: it rides in ``instructions``, the
    Responses schema's dedicated field, sent by ``_call_openai``. It used to be
    joined into ``input`` as well — the endpoint accepted the duplication, but
    it billed the system block twice on every call (this app's digest prompt
    re-sends the same rubric preamble hundreds of times per run, so the waste
    was real money at scale). Verified 2026-09-22 against the live endpoint:
    ``instructions``-only payloads serve correctly, with and without the field.
    """
    return user


def _responses_field(obj: Any, name: str) -> Any:
    """Read *name* off a Responses payload, which arrives as two shapes.

    The SDK's ``responses.create`` returns a typed object with attributes; the
    preflight probe parses the same JSON by hand and holds dicts. Reading both
    here keeps one parser honest for the two callers.
    """
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _responses_output_text(body: Any) -> str:
    """The answer text in a Responses payload, or "" when there is none.

    Reads ``output_text`` (the SDK's convenience aggregation, and what Perplexity's
    own examples read), then walks ``output`` items — an ``output_text`` content part
    on any item — so a service that omits the aggregation still yields its text.
    Accepts the SDK's typed response object or a raw parsed dict.
    """
    text = _responses_field(body, "output_text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    items = _responses_field(body, "output")
    if not isinstance(items, (list, tuple)):
        return ""
    for item in items:
        content = _responses_field(item, "content")
        if not isinstance(content, (list, tuple)):
            continue
        for part in content:
            if _responses_field(part, "type") != "output_text":
                continue
            text = _responses_field(part, "text")
            if isinstance(text, str) and text.strip():
                return text.strip()
    return ""


def _responses_status(body: Any) -> str:
    """The run status of a Responses payload ("completed", "failed", ...), or "".

    The Agent API returns HTTP 200 for runs that then failed server-side: the
    status and the reason live in the body (``status`` + ``error.message``).
    """
    status = _responses_field(body, "status")
    return status if isinstance(status, str) else ""


def _responses_error_message(body: Any) -> str:
    """The provider's own error text in a Responses payload, or "".

    A 200 body that reports a failed run carries the reason in
    ``error.message`` — the actionable part, same as a refusal's body. The
    error object arrives as a dict from the hand-parsed probe and a typed
    object from the SDK, so both reads go through :func:`_responses_field`.
    """
    err = _responses_field(body, "error")
    message = _responses_field(err, "message")
    return message.strip() if isinstance(message, str) and message.strip() else ""


def _responses_empty_error(body: Any) -> LLMError:
    """The error for a Responses answer with no visible text, with its real cause.

    A 200 body can carry ``status: failed`` with the provider's own reason in
    ``error.message`` — reporting that beats a generic "empty response" for
    the same wire event. A completed run with no text is the reasoning-model
    case the probe already treats as a served call; at run time, empty text
    cannot feed a pipeline, so it stays an error, but a status-aware one.

    ``status: incomplete`` is classified retriable, not deterministic: live
    evidence 2026-09-23 (provider degradation, ~470 absorbed merge failures in
    one run) shows the same call succeeding minutes later with nothing changed
    but time. As a plain ``LLMError`` it fell out of the retry ladder in one
    attempt and each failure told the breaker to record a *deterministic*
    failure — whose advice ("fix the request rather than waiting") is exactly
    wrong for a condition waiting actually fixes.
    """
    status = _responses_status(body)
    if status and status != "completed":
        detail = _responses_error_message(body) or "no details given"
        message = f"The Responses run did not complete (status: {status}): {detail}"
        if status == "incomplete":
            return LLMUpstreamError(message, retriable=True)
        return LLMError(message)
    # A completed run with no text rides the same burst signature (empties
    # interleaved with incompletes in the same degradation window), so it gets
    # the ladder too — the moderation nudge and parse re-ask live there.
    return LLMUpstreamError("Model returned an empty response.", retriable=True)


def _responses_usage_tokens(body: Any) -> tuple[int | None, int | None]:
    """``usage.input_tokens`` / ``usage.output_tokens`` from a Responses payload.

    The Responses schema renames the Chat usage fields; both are optional. Accepts
    the SDK's typed response object or a raw parsed dict.
    """
    usage = _responses_field(body, "usage")
    prompt = _responses_field(usage, "input_tokens")
    completion = _responses_field(usage, "output_tokens")
    if not isinstance(prompt, int) or prompt < 0:
        prompt = None
    if not isinstance(completion, int) or completion < 0:
        completion = None
    return prompt, completion


def _chat_output_text(response: Any) -> str:
    """Answer text from an SDK Chat Completions response, or "" when there is none."""
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return ""
    return content.strip() if isinstance(content, str) else ""


def _endpoint_request_url(base_url: str, *, responses: bool) -> str:
    """The URL the wire call (or a probe log line) will actually hit."""
    return base_url.rstrip("/") + ("/responses" if responses else "/chat/completions")


class ChatProbe(NamedTuple):
    """Outcome of one short real-call check against the configured endpoint.

    A model *listing* is not a promise. Perplexity's Router API answered
    ``GET /models`` with its ids and then refused every completion with ``403 The
    Router API is currently in limited preview`` (measured), which is indistinguishable
    from a healthy endpoint until something actually calls it — so the preflight
    follows the listing with one real call.

    ``ok`` means a completion came back — visible text (``reply``), or a completion
    object the model left empty (``silent``). Either one proves what this probe is for:
    the endpoint served the call. ``status`` is the HTTP status when one arrived, and
    ``None`` when the request got no response at all (DNS, refused, timeout). ``error``
    carries the provider's own words, which is what makes a refusal actionable rather
    than a status code to look up.
    """

    status: int | None
    error: str
    reply: str = ""
    #: True when a completion object came back with no visible text. A reasoning model
    #: can spend the probe's whole budget thinking and finish with an empty message
    #: (``finish_reason: "length"``), which is a served call, not a silent endpoint —
    #: reporting it as "no completion" blamed the endpoint for the probe's budget.
    silent: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.reply) or self.silent


def probe_chat(base_url: str, api_key: str, model: str) -> ChatProbe:
    """Ask the endpoint for a short answer, reporting the reply *or* why there is none.

    Best-effort and never raises, like :func:`probe_models`. Deliberately one raw
    request rather than ``LLMClient``: this runs *before* a run, and the client's
    retries plus shared circuit breaker would let a dead endpoint open the breaker
    during the check and poison the run the check exists to protect.

    Stdlib only (urllib), so it adds no dependency and no client to keep in sync.

    The request body matches the endpoint's schema: Responses (``input`` /
    ``max_output_tokens``) on a Perplexity base URL, Chat Completions
    (``messages`` / ``max_tokens``) everywhere else — the same split the run-time
    client uses, so the probe practices exactly what a run will do.
    """
    if not api_key or not api_key.strip():
        return ChatProbe(None, "no API key was supplied")
    if not model or not model.strip():
        return ChatProbe(None, "no model was configured to call")
    responses = _uses_responses_schema(base_url)
    url = _endpoint_request_url(base_url, responses=responses)
    if responses:
        payload = json.dumps(
            {
                "model": model.strip(),
                "input": CHAT_PROBE_PROMPT,
                "max_output_tokens": CHAT_PROBE_MAX_TOKENS,
                "temperature": 0,
                # The probe says only "the endpoint served this call"; it stores
                # nothing, so the check never creates retrievable run state.
                "store": False,
            }
        ).encode("utf-8")
    else:
        payload = json.dumps(
            {
                "model": model.strip(),
                "messages": [{"role": "user", "content": CHAT_PROBE_PROMPT}],
                "max_tokens": CHAT_PROBE_MAX_TOKENS,
                "temperature": 0,
            }
        ).encode("utf-8")
    try:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(  # noqa: S310 - the operator's own endpoint
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=CHAT_PROBE_TIMEOUT_SECONDS) as resp:  # noqa: S310
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - availability check is advisory only
        return ChatProbe(_http_status(exc), _probe_error_text(exc))
    if responses:
        reply = _responses_output_text(body)
        if reply:
            return ChatProbe(200, "", reply)
        status = _responses_status(body)
        if status == "completed":
            # The provider served the call and reports the run completed; the model
            # spent the probe's budget on hidden reasoning (see ``ChatProbe.silent``).
            return ChatProbe(200, "", silent=True)
        if isinstance(body, dict) and ("output" in body or "output_text" in body):
            # A Responses-shaped answer object with no text in it. A failed or
            # incomplete run is reported as the error it is, with the provider's
            # own words (the body arrives with HTTP 200); anything else still
            # proves the endpoint served a Responses run.
            if status and status != "completed":
                detail = _responses_error_message(body)
                suffix = f": {detail}" if detail else ""
                return ChatProbe(200, f"the run did not complete (status: {status}){suffix}")
            return ChatProbe(200, "", silent=True)
        return ChatProbe(200, f"{url} answered without a completion")
    reply = _completion_text(body)
    if reply:
        return ChatProbe(200, "", reply)
    if _has_completion_object(body):
        # The provider served the call; the model just wrote nothing visible and used
        # the budget on hidden reasoning instead. That still proves the key, endpoint
        # and model id can call — see ``ChatProbe.silent``.
        return ChatProbe(200, "", silent=True)
    return ChatProbe(200, f"{url} answered without a completion")


def _completion_text(body: Any) -> str:
    """The assistant text in a Chat Completions response, or "" when there is none.

    Accepts both shapes providers return: ``choices[0].message.content`` (a string, or
    the list of content parts some servers send) and the legacy ``choices[0].text``.
    """
    rows = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        return ""
    choice = rows[0]
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        return "".join(parts).strip()
    text = choice.get("text")
    return text.strip() if isinstance(text, str) else ""


def _has_completion_object(body: Any) -> bool:
    """Whether the response carries a completion object, even with no visible text.

    The shape a budget-starved reasoning model answers with: one choice whose
    ``message`` is present and whose ``content`` is empty. Reading that as "no
    completion" reported the endpoint as silent when it had served the call.
    """
    rows = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        return False
    choice = rows[0]
    return isinstance(choice.get("message"), dict) or isinstance(choice.get("text"), str)


def _http_status(exc: BaseException) -> int | None:
    """The HTTP status carried by a urllib failure, if it has one."""
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


def _probe_error_text(exc: BaseException) -> str:
    """One short line naming why the model listing failed."""
    status = _http_status(exc)
    reason = getattr(exc, "reason", None)
    detail = str(reason if reason is not None else exc).strip()
    if status is not None:
        # HTTPError's own str() is just the status line, which says nothing the
        # status does not; the *body* is where a provider explains itself, and it
        # is what tells a rejected key apart from a path that does not exist.
        body = ""
        read = getattr(exc, "read", None)
        if callable(read):
            try:
                body = read().decode("utf-8", "replace").strip()
            except Exception:  # noqa: BLE001 - the body is a bonus, not the diagnosis
                body = ""
        return f"HTTP {status}: {(body or detail)[:200]}"
    return f"{type(exc).__name__}: {detail}"[:200]


def check_model_availability(base_url: str, api_key: str) -> set[str] | None:
    """The model ids at ``GET {base_url}/models``, or ``None`` on any failure.

    Advisory-only wrapper over :func:`probe_models`, kept because callers that
    only want the warning-suppression behaviour should not have to unpack a
    reason they are not going to show anyone.
    """
    return probe_models(base_url, api_key).models


def _validate_fallback_settings(settings: Settings) -> None:
    """Validate the failover endpoint — only called when one is configured."""
    target = _fallback_target(settings)
    if not target.configured:
        return
    base_url = target.base_url
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise LLMConfigurationError(
            "OPENAI_BASE_URL_FALLBACK must be a valid http(s) URL for an "
            "OpenAI-compatible endpoint."
        )
    if base_url.rstrip("/") == (settings.base_url or "").strip().rstrip("/"):
        # Not fatal, but it is certainly not a failover: the same endpoint will
        # fail the same way, and the logs would claim two providers were tried.
        logger.warning(
            "OPENAI_BASE_URL_FALLBACK is the same URL as OPENAI_BASE_URL — "
            "failover to it cannot help (no second endpoint is configured).",
            extra={"phase": "llm_config", "status": "warn"},
        )
    # No key check here: ``target.api_key`` is ``OPENAI_API_KEY_FALLBACK or
    # OPENAI_API_KEY``, and ``_validate_settings`` has already rejected an empty
    # primary key, so a missing key cannot reach this point.
    for label, model in (
        ("Fallback main model", target.model_main),
        ("Fallback fast model", target.model_fast),
    ):
        msg = validate_model_name(str(model or ""))
        if msg:
            raise LLMConfigurationError(f"{label}: {msg}")


def _usage_tokens(response: Any) -> tuple[int | None, int | None]:
    """Pull prompt/completion tokens from a response if the provider reports them.

    Some OpenAI-compatible gateways omit usage entirely; we fall back to a
    character-based estimate in that case.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if not isinstance(prompt, int) or prompt < 0:
        prompt = None
    if not isinstance(completion, int) or completion < 0:
        completion = None
    return prompt, completion


class LLMError(RuntimeError):
    """Raised when the LLM call ultimately fails or returns unusable output."""


class LLMConfigurationError(LLMError):
    """Raised when runtime LLM settings are missing or malformed."""


class LLMUpstreamError(LLMError):
    """Raised when the upstream LLM provider fails or rejects the request."""

    def __init__(
        self,
        message: str,
        *,
        retriable: bool = False,
        status_code: int | None = None,
        upstream_request_id: str = "",
    ) -> None:
        super().__init__(message)
        self.retriable = retriable
        self.status_code = status_code
        self.upstream_request_id = upstream_request_id


class LLMAuthError(LLMUpstreamError):
    """Fatal credential/authorization refusal (HTTP 401/403).

    A subclass on purpose, so every ``except LLMUpstreamError`` keeps working,
    but a distinct one because the retry/bisect machinery must not treat it
    like a payload problem: the credential verdict applies to every future
    call identically, so it takes exactly one attempt (never retried — see
    :func:`_normalize_provider_error`), never failover (the fallback would be
    refused the same way — it cannot share this credential's fate differently),
    and the batch runner treats it as a stop-the-run signal rather than a file
    to quarantine. Raised for a rejected key, an account without entitlement,
    or a revoked/deactivated credential — never for a rate limit or an outage.
    """


class LLMTimeoutError(LLMUpstreamError):
    """Raised when the upstream LLM call times out."""


class LLMParseError(LLMError):
    """Raised when the model response cannot be parsed into the expected format.

    The model's output is sampled stochastically — the identical prompt can yield
    valid JSON on one call and unparseable text on the next (measured 2026-09-20:
    5 of 149 digest chunks in one run) — so :meth:`LLMClient.chat_json` re-asks
    once before raising. The failure stays logged with the raw shape for
    diagnosis, and one re-ask cannot recurse.
    """


class _ModerationFilteredError(LLMUpstreamError):
    """Provider content filter rejected the request or output (HTTP 400).

    Non-retryable as-is: identical input reproduces the rejection. The retry
    loop may still take the one-shot clinical-tone nudge (see
    ``_moderation_nudge_user``), which rephrases the prompt and retries.
    Carries an actionable, PII-free message for the UI.
    """


# ------------------------------------------------------------------ config helpers
def _configured_timeout_seconds() -> float:
    try:
        from . import config as _cfg

        timeout_seconds = float(getattr(_cfg, "LLM_CALL_TIMEOUT_SECONDS", 300))
    except (TypeError, ValueError):
        raise LLMConfigurationError(
            "LLM call timeout must be a positive number (VA_LSE_LLM_CALL_TIMEOUT_SECONDS)."
        ) from None
    if timeout_seconds <= 0:
        raise LLMConfigurationError(
            "LLM call timeout must be greater than 0 seconds (VA_LSE_LLM_CALL_TIMEOUT_SECONDS)."
        )
    return max(1.0, timeout_seconds)


def _stall_watchdog_seconds() -> float:
    """Wall-clock budget for one provider call, or ``0.0`` to disable the watchdog.

    ``multiplier ×`` the configured per-call timeout (via
    ``VA_LSE_LLM_STALL_WATCHDOG_MULTIPLIER``), so the watchdog is always the
    *second* line of defense: the HTTP timeout should fire first for a slow
    provider; the watchdog only saves a call whose connection died silently —
    with a warm keep-alive socket, "no bytes for N seconds" never accumulates
    and no read timeout ever fires (the 2026-09-22 five-hour hang).
    """
    try:
        from . import config as _cfg

        multiplier = float(getattr(_cfg, "LLM_STALL_WATCHDOG_MULTIPLIER", 2.0))
    except (TypeError, ValueError):
        multiplier = 2.0
    if multiplier <= 0:
        return 0.0  # disabled
    return multiplier * _configured_timeout_seconds()


def _shutdown_pool_sockets(client: OpenAI) -> int:
    """Force-close the raw sockets beneath *client*'s connection pool.

    The SDK's ``close()`` releases *idle* connections but does not unblock a
    request already parked in a socket read (measured 2026-09-23 against the
    vendored openai 3.13/httpx2 stack: a hung call stayed parked through
    ``close()``). The stall watchdog therefore walks the pool's connection
    objects to the underlying sockets and ``shutdown()``s them — the parked
    read fails immediately with a connection error the retry ladder already
    classifies as retriable. Best-effort by design: returns the number of
    sockets touched, and any object shape this walk cannot parse simply means
    no rescue fires for that call shape.
    """
    try:
        pool = client._client._transport._pool  # type: ignore[attr-defined]
        frontier: list[Any] = list(getattr(pool, "connections", []) or [])
    except AttributeError:
        return 0
    seen: set[int] = set()
    count = 0
    depth = 0
    while frontier and depth < 8:
        nxt: list[Any] = []
        for obj in frontier:
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            for value in list(vars(obj).values()) if hasattr(obj, "__dict__") else []:
                if isinstance(value, socket.socket):
                    try:
                        value.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass  # already gone — that is the rescue working
                    try:
                        value.close()
                    except OSError:
                        pass
                    count += 1
                elif hasattr(value, "__dict__"):
                    nxt.append(value)
                elif isinstance(value, (list, tuple)):
                    nxt.extend(v for v in value if hasattr(v, "__dict__"))
        frontier = nxt
        depth += 1
    return count


def _validate_settings(settings: Settings) -> None:
    if not settings.configured:
        raise LLMConfigurationError(
            "No API key configured. Add your key in the sidebar or in a .env file."
        )
    base_url = (getattr(settings, "base_url", "") or "").strip()
    parsed = urlparse(base_url)
    if not base_url or parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise LLMConfigurationError(
            "Base URL must be a valid http(s) URL for an OpenAI-compatible endpoint."
        )
    for label, model in (
        ("Main model", getattr(settings, "model_main", "")),
        ("Fast model", getattr(settings, "model_fast", "")),
    ):
        msg = validate_model_name(str(model or ""))
        if msg:
            raise LLMConfigurationError(f"{label}: {msg}")


def _provider_status_code(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None)
    if value is None:
        return None
    try:
        status_code = int(value)
    except (TypeError, ValueError):
        return None
    return status_code if status_code > 0 else None


def _provider_request_id(exc: BaseException) -> str:
    for attr in ("request_id", "requestId"):
        value = getattr(exc, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        for key in ("x-request-id", "request-id", "openai-request-id"):
            value = headers.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _is_timeout_error(exc: BaseException) -> bool:
    if isinstance(exc, _HttpxTimeoutError) or isinstance(exc, TimeoutError):
        return True
    return type(exc).__name__.endswith("TimeoutError")


def _is_transient_status(status_code: int | None) -> bool:
    # 499 (nginx/Cloudflare "client closed request" / client_disconnected) is a
    # *transport* event — the connection died mid-flight, here after 166 s of a
    # live digest — not a judgment about the payload, so a retry of the identical
    # request plausibly succeeds. It must be listed explicitly: it is 4xx, which
    # the 5xx clause does not reach.
    return bool(
        status_code is not None
        and (status_code in {408, 409, 425, 429, 499} or 500 <= status_code < 600)
    )


def _is_transient_provider_error(exc: BaseException) -> bool:
    """True when retrying the same request could plausibly succeed."""
    status_code = _provider_status_code(exc)
    # Defensive, not dead code: _normalize_provider_error routes 401/403 to
    # LLMAuthError before consulting this predicate, but anything that builds
    # an upstream error directly (probes, tests, future call sites) must not
    # have an auth refusal reinterpreted as "retry may succeed".
    if status_code in (401, 403):
        return False
    if _is_timeout_error(exc):
        return True
    if isinstance(exc, (ConnectionError, OSError)):
        return True
    status_code = _provider_status_code(exc)
    if status_code is not None:
        return _is_transient_status(status_code)
    name = type(exc).__name__.lower()
    if any(token in name for token in ("badrequest", "authentication", "permission", "notfound", "unprocessable")):
        return False
    if any(token in name for token in ("timeout", "ratelimit", "apiconnection", "serviceunavailable", "internalserver")):
        return True
    return True


def _provider_details(exc: BaseException) -> str:
    details = [f"{type(exc).__name__}: {exc}"]
    status_code = _provider_status_code(exc)
    if status_code is not None:
        details.append(f"status={status_code}")
    upstream_request_id = _provider_request_id(exc)
    if upstream_request_id:
        details.append(f"upstream_request_id={upstream_request_id}")
    return "; ".join(details)


def _normalize_provider_error(exc: Exception) -> LLMError:
    """Map a raw provider exception onto the LLM error taxonomy.

    Content-filter rejections (``data_inspection_failed`` or explicit filter
    wording) map to :class:`_ModerationFilteredError` with an actionable,
    PII-free message and ``retriable=False`` — the retry loop may still take
    its one-shot tone nudge, but blind retries of a deterministic 400 are
    pointless. Everything else keeps the remote taxonomy: timeouts →
    ``LLMTimeoutError`` (retriable), transient statuses/connection errors →
    retriable ``LLMUpstreamError``, deterministic rejections →
    non-retriable ``LLMUpstreamError``.
    """
    status_code = _provider_status_code(exc)
    upstream_request_id = _provider_request_id(exc)
    details = _provider_details(exc)
    # A 401/403 is a verdict on the credential, not the payload: retrying the
    # identical request reproduces it exactly (the 2026-09-22 incident — a key
    # deactivated provider-side between 14:47 and 16:13 — burned 3 attempts and
    # opened the breaker before the bisect cascade quarantined healthy files).
    # One attempt, one actionable message, own class so the batch runner can
    # halt instead of bisect.
    if status_code in (401, 403):
        return LLMAuthError(
            f"Credentials refused by the LLM endpoint (HTTP {status_code}) — the API key "
            f"is invalid, revoked, or not entitled to this model/endpoint. Regenerate the "
            f"key, check the account's credit balance, and confirm base URL and key belong "
            f"to the same provider account. ({details})",
            retriable=False,
            status_code=status_code,
            upstream_request_id=upstream_request_id,
        )
    if _is_moderation_filtered(exc):
        return _ModerationFilteredError(
            "The LLM provider's content filter rejected this run "
            "(HTTP 400 data_inspection_failed). Medical/trauma wording in the "
            "draft triggers it intermittently — this is not a bug in your input. "
            "Retry the run; if it keeps failing, rewording the observations "
            "(fewer graphic injury details) usually clears it.",
            retriable=False,
            status_code=status_code,
            upstream_request_id=upstream_request_id,
        )
    if _is_timeout_error(exc):
        timeout_seconds = int(_configured_timeout_seconds())
        return LLMTimeoutError(
            f"LLM call timed out after {timeout_seconds}s — the endpoint did not respond in time. "
            f"Try again or raise VA_LSE_LLM_CALL_TIMEOUT_SECONDS. ({details})",
            retriable=True,
            status_code=status_code,
            upstream_request_id=upstream_request_id,
        )
    retriable = _is_transient_provider_error(exc)
    if retriable:
        return LLMUpstreamError(
            f"Transient LLM provider error — retry may succeed. ({details})",
            retriable=True,
            status_code=status_code,
            upstream_request_id=upstream_request_id,
        )
    return LLMUpstreamError(
        f"LLM provider rejected the request — check model, endpoint, and payload settings. ({details})",
        retriable=False,
        status_code=status_code,
        upstream_request_id=upstream_request_id,
    )


def _retry_backoff_seconds(attempt: int) -> float:
    return min(MAX_RETRY_BACKOFF_SECONDS, RETRY_BACKOFF_SECONDS * (2.0 ** attempt))


#: Jitter added to an honored Retry-After wait. A provider that throttles N
#: concurrent workers answers them with the *same* Retry-After value — without
#: jitter they would all wake at the same instant and re-form the herd the
#: header exists to disperse. Uniform 0.1–0.5 s is small next to any real
#: Retry-After and large enough to stagger thread wakes.
RETRY_AFTER_JITTER_SECONDS = (0.1, 0.5)


def _retry_after_hint(exc: BaseException | None) -> float | None:
    """The Retry-After demand carried by a raw provider error, or ``None``.

    The OpenAI SDK raises ``openai.RateLimitError`` (an ``APIStatusError``) for
    HTTP 429 and keeps the underlying ``httpx.Response`` on ``.response`` —
    whose ``.headers`` is a case-insensitive mapping, so ``get("retry-after")``
    matches ``Retry-After``, ``RETRY-AFTER``, and any casing the provider uses.
    Both header forms RFC 7231 allows are parsed: integer seconds (what
    Perplexity sends) and HTTP-date (resolved against the local clock; the
    second or so of parsing drift is bounded by the cap and jitter). Missing,
    empty, unparseable, or negative values all return ``None`` — the caller
    falls back to the ladder, never guesses.
    """
    if exc is None:
        return None
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after")
    # isinstance guard, not truthiness: stubbed responses in tests carry a
    # MagicMock whose .get() returns a MagicMock, and float(MagicMock())
    # silently returns 1.0 — a header that was never sent must not be honored.
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        seconds = None
    if seconds is not None:
        return seconds if seconds >= 0 else None
    # Not an integer: attempt the HTTP-date form. Imported lazily — the common
    # path (integer header, or no header at all) must not pay for email.utils.
    try:
        from datetime import datetime, timezone
        from email.utils import parsedate_to_datetime

        when = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    delta = (when - datetime.now(timezone.utc)).total_seconds()
    return delta if delta > 0 else 0.0


def _retry_wait_seconds(attempt: int, exc: BaseException | None) -> tuple[float, bool]:
    """``(seconds, honored)`` to wait before the next attempt after *exc*.

    Honors the provider's Retry-After when one is present and the cap allows,
    clamped to ``VA_LSE_LLM_RETRY_AFTER_MAX_SECONDS`` and jittered; missing,
    unparseable, or disabled (cap 0) falls back to the exponential ladder,
    exactly as before. Thread safety is inherited from the sleep, not added
    here: the wait runs through ``wait_with_cancellation`` inside the one
    worker thread that caught the 429 — the main thread and the other workers
    never block on it — and the jitter staggers wakes so the herd does not
    re-form at the header's expiry instant.
    """  # noqa: E501 - docstring width is not load-bearing
    hint = _retry_after_hint(exc)
    cap = getattr(_config_module(), "LLM_RETRY_AFTER_MAX_SECONDS", 60.0)
    if hint is not None and cap > 0:
        return min(max(hint, 0.0), cap) + random.uniform(*RETRY_AFTER_JITTER_SECONDS), True
    return _retry_backoff_seconds(attempt), False


def _failure_reason(error: BaseException | None) -> str:
    """One bounded line naming a failure for the breaker's OPEN message.

    The class name is kept because it is the part that survives grepping the docs
    and the log (``LLMUpstreamError`` vs ``LLMTimeoutError``), and the message is
    already user-facing, so nothing here is withheld from it.
    """
    if error is None:
        return ""
    return f"{type(error).__name__}: {error}"


@runtime_checkable
class LLMService(Protocol):
    """The surface every pipeline component consumes — the app's LLM boundary.

    Phase-1 abstraction: nothing outside this module may import an LLM SDK or
    speak HTTP to a provider. Components (draft, evaluate, medical_review,
    views, worker) type their ``llm`` parameter as ``LLMService`` and call only
    these methods, which is what makes the endpoint a *setting*: today the
    concrete implementation is :class:`LLMClient` (any OpenAI-compatible
    host — Perplexity Agent API, Azure OpenAI under a BAA, Ollama, a gateway),
    and a future backend (a brokered HIPAA service, a batch API) satisfies the
    same Protocol without a single call-site change.

    Keep this Protocol minimal on purpose: every method here is a promise a
    replacement backend must keep, including the JSON-repair behaviour of
    :meth:`chat_json` and the cancellation-aware retry semantics the pipeline
    guard relies on.
    """

    def chat(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 8000,
        phase: str = "general",
    ) -> str: ...

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 8000,
        phase: str = "general",
    ) -> Any: ...

    @property
    def fast_model(self) -> str: ...
    """Model id for bulk/cheap calls (the digest's workhorse).

    Part of the contract, not a setting leak: the pipeline's cost shape
    depends on bulk work going to the cheap model, so any backend must say
    which of its models plays that role.
    """


class LLMClient:
    """Thin wrapper around any OpenAI-compatible endpoint.

    Tracks an estimated-usage ``UsageTracker`` so callers can report per-phase
    token/call counts and (optionally) credit burn after a run.

    Each ``chat`` call is gated by a process-wide **circuit breaker** (opens
    after 3 consecutive logical failures, fail-fast in <2 s while open) and a
    **concurrency limiter** with a bounded queue (see ``app/circuit_breaker.py``
    and ``app/config.py``).
    """

    @property
    def fast_model(self) -> str:
        """The cheap model for bulk passes (the :class:`LLMService` contract)."""
        return self._settings.model_fast

    def __init__(self, settings: Settings) -> None:
        # First real client construction is the moment the SDK's import floor
        # is paid. Both names are bound into module globals here so every
        # later use site — including the bare NOT_GIVEN in the request loop —
        # resolves them the same way it did when the import was eager.
        OpenAI_cls = _sdk_name("OpenAI")
        _sdk_name("NOT_GIVEN")
        _validate_settings(settings)
        _validate_fallback_settings(settings)
        self._settings = settings
        _timeout_s = _configured_timeout_seconds()
        self._client = cast("type[OpenAI]", OpenAI_cls)(
            api_key=settings.api_key, base_url=settings.base_url, timeout=max(1.0, _timeout_s),
            max_retries=0,  # Application retries must observe pipeline cancellation.
        )
        # Built once here (not lazily on the failover path) so a bad fallback URL
        # or key is reported at startup rather than discovered mid-outage.
        self._fallback_client: OpenAI | None = None
        fallback = _fallback_target(settings)
        if fallback.configured:
            self._fallback_client = cast("type[OpenAI]", OpenAI_cls)(
                api_key=fallback.api_key,
                base_url=fallback.base_url,
                timeout=max(1.0, _timeout_s),
                max_retries=0,
            )
        self.usage = UsageTracker()

    # -------------------------------------------------------------- endpoints

    @property
    def fallback_available(self) -> bool:
        """Whether this client can serve a call from a second endpoint."""
        return self._fallback_client is not None

    def _endpoint_client(self, endpoint: str) -> OpenAI:
        if endpoint == FALLBACK_ENDPOINT:
            if self._fallback_client is None:
                raise LLMConfigurationError(
                    "The fallback endpoint was selected but is not configured."
                )
            return self._fallback_client
        return self._client

    def _rebuild_client(self, endpoint: str) -> None:
        """Replace *endpoint*'s client with one on a fresh connection pool.

        Called by the stall watchdog right after force-closing the old pool: the
        retry attempt must not land on the dead sockets, and the SDK exposes no
        way to swap the pool inside a live client. Mirrors ``__init__``'s
        construction exactly (same timeout, retries disabled) so a rebuilt
        client is indistinguishable from a fresh start.
        """
        OpenAI_cls = cast("type[OpenAI]", _sdk_name("OpenAI"))
        timeout = max(1.0, _configured_timeout_seconds())
        if endpoint == FALLBACK_ENDPOINT:
            target = _fallback_target(self._settings)
            self._fallback_client = OpenAI_cls(
                api_key=target.api_key,
                base_url=target.base_url,
                timeout=timeout,
                max_retries=0,  # Application retries must observe pipeline cancellation.
            )
            return
        self._client = OpenAI_cls(
            api_key=self._settings.api_key,
            base_url=self._settings.base_url,
            timeout=timeout,
            max_retries=0,  # Application retries must observe pipeline cancellation.
        )

    def _stall_watchdog_fired(
        self,
        endpoint: str,
        rid: str,
        phase: str,
        model: str,
        attempt: int,
        budget: float,
    ) -> None:
        """Watchdog callback (runs on its timer thread): rescue a stalled call.

        A call that outlived this budget is past its HTTP deadline twice over
        and its socket read is parked with no end in sight. The rescue is
        physical, not cooperative: the raw sockets beneath the pool are
        ``shutdown()`` so the parked read fails immediately, the client is
        closed, and a fresh client (new pool) is built so the retry does not
        land on the dead sockets. Races are harmless — a call that completed
        microseconds earlier had its timer cancelled already, and closing a
        pool whose only request just finished only costs the next call one
        new TCP handshake. Any call still alive at this point has already
        outlived its own deadline, so pool-wide collateral is bounded to calls
        that are themselves anomalous.
        """
        client = self._endpoint_client(endpoint)
        touched = _shutdown_pool_sockets(client)
        try:
            client.close()
        except Exception:  # noqa: BLE001 - the timer thread must never die loudly
            logger.debug(
                "llm stall watchdog could not close the %s client", endpoint, exc_info=True
            )
        try:
            self._rebuild_client(endpoint)
        except Exception:  # noqa: BLE001 - retry may still land on the closed pool once
            logger.warning(
                "llm stall watchdog could not rebuild the %s client — the next "
                "attempt may fail once more before recovering",
                endpoint,
                exc_info=True,
                extra={"request_id": rid, "phase": phase, "endpoint": endpoint},
            )
        logger.warning(
            "llm stall watchdog force-closed the %s connection pool after %.0fs "
            "with no response (attempt %d, %d socket(s) shut down) — the stalled "
            "call fails fast and retries on a fresh pool",
            endpoint,
            budget,
            attempt,
            touched,
            extra={
                "request_id": rid,
                "phase": phase,
                "status": "stall_watchdog",
                "model": model,
                "endpoint": endpoint,
                "attempt": attempt,
                "stall_budget_seconds": budget,
                "sockets_shutdown": touched,
            },
        )

    def _call_openai(
        self,
        endpoint: str,
        model: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        request_timeout: Any,
    ) -> tuple[Any, bool]:
        """One provider call in the endpoint's own schema; ``(response, is_responses)``.

        The OpenAI SDK's ``responses.create`` is used for the Responses schema so the
        SDK stays the single wire dependency (it posts ``{base_url}/responses`` and
        parses into objects); raw kwargs mirror what the SDK sends so the failover
        and retry paths behave identically across schemas. Returns the schema flag
        alongside the response because the answer and usage live at different keys
        in each shape, and the caller must not guess.
        """
        client = self._endpoint_client(endpoint)
        if _uses_responses_schema(self._endpoint_base_url(endpoint)):
            request: dict[str, Any] = {
                "model": model,
                "input": _responses_input(system, user),
                "max_output_tokens": max(1, max_tokens),
                # The endpoint is stateless for this app's use: every prompt is
                # fully self-contained, nothing is continued via
                # previous_response_id, and the payloads carry medical records.
                # Opting out of server-side storage keeps retrievable copies of
                # that content from accumulating on the provider — the
                # documented way to run once and leave nothing behind.
                "store": False,
            }
            if system.strip():
                request["instructions"] = system
            if temperature != 0.2:
                request["temperature"] = temperature
            if request_timeout is not NOT_GIVEN:
                request["timeout"] = request_timeout
            return client.responses.create(**request), True
        return client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            timeout=request_timeout,
        ), False

    def _endpoint_base_url(self, endpoint: str) -> str:
        """The base URL *endpoint* will be called with (schema is decided from it)."""
        if endpoint == FALLBACK_ENDPOINT:
            target = _fallback_target(self._settings)
            return target.base_url
        return self._settings.base_url

    def _resolve_model(self, endpoint: str, requested: str | None) -> str:
        """The model name to send to *endpoint* for the model the caller asked for.

        Callers name the primary's models (``settings.model_main`` / ``model_fast``
        or nothing at all). A fallback is usually a different provider whose model
        names differ, so those two roles are translated onto the fallback's names.
        A caller passing some *other* model name is naming one for this
        deployment; it is passed through untouched, because inventing an
        equivalent on another provider would be guesswork.
        """
        settings = self._settings
        requested_name = (requested or settings.model_main).strip()
        if endpoint != FALLBACK_ENDPOINT:
            return requested_name
        target = _fallback_target(settings)
        if requested_name == (settings.model_main or "").strip():
            return target.model_main
        if requested_name == (settings.model_fast or "").strip():
            return target.model_fast
        return requested_name

    def _endpoint_candidates(self) -> list[str]:
        """Which endpoint(s) a call may use, in the order they should be tried.

        * No fallback configured -> the primary only, exactly as before.
        * Primary healthy, **or** unhealthy for less than the grace period -> the
          primary only. A blip must not move the business onto another provider,
          and a short outage is better absorbed than routed around.
        * Primary unhealthy for at least the grace period -> the fallback. If the
          primary's own recovery window has elapsed, the primary is tried first so
          this call doubles as its probe: a failure falls through to the fallback
          (see :meth:`chat`) instead of failing the user's run.
        """
        if not self.fallback_available:
            return [PRIMARY_ENDPOINT]
        primary = get_llm_breaker(_endpoint_breaker_name(PRIMARY_ENDPOINT))
        unhealthy = primary.unhealthy_for_seconds()
        if unhealthy is None or unhealthy < failover_after_seconds():
            return [PRIMARY_ENDPOINT]
        if primary.allow_request():
            return [PRIMARY_ENDPOINT, FALLBACK_ENDPOINT]
        return [FALLBACK_ENDPOINT]

    def _failover_note(self) -> str:
        """Extra sentence for a fast-fail error when a fallback is armed."""
        if not self.fallback_available:
            return ""
        primary = get_llm_breaker(_endpoint_breaker_name(PRIMARY_ENDPOINT))
        unhealthy = primary.unhealthy_for_seconds()
        threshold = failover_after_seconds()
        if unhealthy is None:
            return ""
        remaining = max(0.0, threshold - unhealthy)
        return (
            f"A backup endpoint is configured: it will be used automatically once the "
            f"primary has been failing for {threshold:.0f}s (in {remaining:.0f}s)."
        )

    # ------------------------------------------------------------------ core
    def chat(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 8000,
        phase: str = "general",
    ) -> str:
        """Single-turn chat completion, with failover to a second endpoint.

        Tries the endpoints from :meth:`_endpoint_candidates` in order (one, unless
        failover is engaged and the primary's probe is due). A fast-fail rejection
        is never retried on the other endpoint — it means "do not call this
        provider right now" — but a genuine call failure is, so the primary's
        recovery probe costs the user nothing.
        """
        check_pipeline_cancelled()
        request_model = (model or self._settings.model_main).strip()
        candidates = self._endpoint_candidates()
        last_error: Exception | None = None
        for index, endpoint in enumerate(candidates):
            check_pipeline_cancelled()
            try:
                return self._chat_on_endpoint(
                    endpoint,
                    system,
                    user,
                    model=request_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    phase=phase,
                )
            except (CircuitBreakerOpenError, QueueFullError) as exc:
                # Fail fast all the way to the caller. If the primary is simply
                # still inside its grace period, say when failover will engage —
                # otherwise the message reads as "endpoint is down" with no hint
                # that a backup is coming.
                note = self._failover_note() if endpoint == PRIMARY_ENDPOINT else ""
                if not note:
                    raise
                raise type(exc)(f"{exc} {note}") from exc
            except LLMAuthError:
                # A credential verdict, not an endpoint outage: the fallback
                # would be refused the same way (its key defaults to this one,
                # and a second provider's key authenticates nothing at this
                # endpoint). One clean raise beats a guaranteed second 401 and
                # a metrics record that says "failover happened".
                raise
            except LLMError as exc:
                last_error = exc
                if index + 1 >= len(candidates):
                    raise
                metrics.observe_llm_failover(reason="primary_call_failed")
                logger.warning(
                    "failover: primary call failed (%s: %s) — serving this call from "
                    "the fallback endpoint",
                    type(exc).__name__,
                    exc,
                    extra={
                        "request_id": get_request_id() or "-",
                        "phase": phase,
                        "status": "failover",
                        "from_endpoint": endpoint,
                        "to_endpoint": candidates[index + 1],
                        "error_class": type(exc).__name__,
                    },
                )
        assert last_error is not None  # pragma: no cover - loop always returns/raises
        raise last_error

    # ------------------------------------------------------------------ core
    def _chat_on_endpoint(
        self,
        endpoint: str,
        system: str,
        user: str,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 8000,
        phase: str = "general",
    ) -> str:
        """One endpoint's attempt at a chat completion (retries, breaker, limiter).

        Raises ``CircuitBreakerOpenError`` (fail-fast, <2 s, no network) when this
        endpoint's breaker is OPEN, ``QueueFullError`` when the concurrency queue
        is full or times out or the rate gate's pacing budget is exceeded (a
        ``RateGateTimeoutError``, also a ``QueueFullError``), and ``LLMError``
        when the provider call exhausts its retries. The breaker counts only
        *logical* call failures (one per call that exhausts retries), not
        per-attempt retries.
        """
        model = self._resolve_model(endpoint, model)  # noqa: A001 - reassign param
        rid = get_request_id() or "-"

        # Fail fast before touching the limiter or the network.
        breaker = get_llm_breaker(_endpoint_breaker_name(endpoint))
        try:
            breaker.check_or_raise()
        except CircuitBreakerOpenError:
            # Counted separately from endpoint failures: these are calls the user
            # was told to retry, and a rising count is what makes an open breaker
            # visible to an alert rather than only to whoever reads logs.
            metrics.observe_breaker_rejection(breaker.name)
            raise

        limiter = get_llm_limiter()
        # Preventive pacing BEFORE taking a concurrency slot: the gate spaces
        # admissions so the provider's 429s are avoided rather than retried,
        # and its wait must not hold a limiter slot that other calls could use.
        # The wait budget is capped at the pipeline's remaining time exactly
        # like the limiter queue wait below, so a paced call can never be the
        # reason a pipeline run blows its own wall clock.
        rate_gate = get_llm_rate_gate()
        if rate_gate.enabled:
            remaining = pipeline_remaining_seconds()
            try:
                if remaining is None:
                    rate_gate.wait()
                else:
                    rate_gate.wait(timeout=min(rate_gate.queue_timeout, remaining))
            except RateGateTimeoutError:
                check_pipeline_cancelled()
                raise
        # Acquire a concurrency slot (queues up to max_queue_depth, else QueueFullError).
        remaining = pipeline_remaining_seconds()
        try:
            if remaining is None:
                limiter.acquire()
            else:
                limiter.acquire(timeout=min(limiter.queue_timeout, remaining))
        except QueueFullError:
            check_pipeline_cancelled()
            raise
        acquired = True
        try:
            check_pipeline_cancelled()
            # Re-check breaker after queuing — it may have opened while we waited.
            try:
                breaker.check_or_raise()
            except CircuitBreakerOpenError:
                metrics.observe_breaker_rejection(breaker.name)
                raise

            sys_len = len(system or "")
            user_len = len(user or "")
            last_error: Exception | None = None
            nudged = False
            t0 = time.perf_counter()
            # Watchdog budget for one wire call (see _stall_watchdog_fired);
            # 0 disables the watchdog. Computed once — it does not vary per
            # attempt.
            stall_budget = _stall_watchdog_seconds()
            for attempt in range(MAX_RETRIES):
                check_pipeline_cancelled()
                attempt_t0 = time.perf_counter()
                stall_timer: threading.Timer | None = None
                try:
                    # Opt-in span around the provider call itself
                    # (VA_LSE_TRACE_LLM_CALLS) — this is where endpoint latency
                    # and retries show up, as opposed to phase-level totals.
                    with tracing.llm_call_span(
                        phase, model=model, attempt=attempt + 1, endpoint=endpoint
                    ):
                        remaining = pipeline_remaining_seconds()
                        request_timeout = (
                            min(
                                max(1.0, _configured_timeout_seconds()), remaining
                            )
                            if remaining is not None else NOT_GIVEN
                        )
                        if stall_budget > 0:
                            # Armed only around the wire call: once _call_openai
                            # returns, the response is fully buffered and there is
                            # nothing left to stall. The timer fires on its own
                            # daemon thread and force-closes the pool (see
                            # _stall_watchdog_fired); cancelling here is cheap
                            # and races are harmless.
                            stall_timer = threading.Timer(
                                stall_budget,
                                self._stall_watchdog_fired,
                                args=(
                                    endpoint,
                                    rid,
                                    phase,
                                    model,
                                    attempt + 1,
                                    stall_budget,
                                ),
                            )
                            stall_timer.daemon = True
                            stall_timer.start()
                        try:
                            response, responses = self._call_openai(
                                endpoint, model, system, user, temperature, max_tokens,
                                request_timeout,
                            )
                        finally:
                            if stall_timer is not None:
                                stall_timer.cancel()
                    check_pipeline_cancelled()
                    content = (
                        _responses_output_text(response) if responses
                        else _chat_output_text(response)
                    )
                    if not content:
                        raise _responses_empty_error(response) if responses \
                            else LLMError("Model returned an empty response.")
                    prompt_tokens, completion_tokens = (
                        _responses_usage_tokens(response) if responses
                        else _usage_tokens(response)
                    )
                    self.usage.record(
                        model=model,
                        endpoint=endpoint,
                        phase=phase,
                        system=system,
                        user=user,
                        content=content,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                    duration_ms = int((time.perf_counter() - attempt_t0) * 1000)
                    total_ms = int((time.perf_counter() - t0) * 1000)
                    logger.info(
                        "llm call ok phase=%s model=%s attempt=%d/%d duration_ms=%d total_ms=%d tokens_in=%s tokens_out=%s sys_chars=%d user_chars=%d out_chars=%d",
                        phase,
                        model,
                        attempt + 1,
                        MAX_RETRIES,
                        duration_ms,
                        total_ms,
                        str(prompt_tokens) if prompt_tokens is not None else "est",
                        str(completion_tokens) if completion_tokens is not None else "est",
                        sys_len,
                        user_len,
                        len(content),
                        extra={
                            "request_id": rid,
                            "phase": phase,
                            "status": "ok",
                            "model": model,
                            "endpoint": endpoint,
                            "attempt": attempt + 1,
                            "retries": MAX_RETRIES,
                            "duration_ms": duration_ms,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                        },
                    )
                    breaker.record_success()
                    # Every provider attempt is counted, including the one that
                    # succeeded. Without this the attempts counter would only ever
                    # see failures, so `attempts / calls` would sit near zero on a
                    # healthy system and the retry-overhead alert could never fire.
                    metrics.observe_llm_attempt(phase, "ok")
                    # One observation per *logical* call: total_ms includes any
                    # retries that preceded this success, which is what a user
                    # actually waited through.
                    metrics.observe_llm_call(phase, "ok", total_ms, endpoint=endpoint)
                    return content
                except (CircuitBreakerOpenError, QueueFullError):
                    # Never count limiter/breaker rejections as endpoint failures.
                    raise
                except Exception as exc:  # noqa: BLE001 - retry on any provider error
                    check_pipeline_cancelled()
                    # A stall-watchdog rescue surfaces here as the SDK's
                    # APIConnectionError (the shutdown socket made the parked
                    # read fail); _normalize_provider_error already classifies
                    # connection errors as retriable, so the ordinary ladder
                    # absorbs it — no special case needed.
                    # Normalize onto the error taxonomy. Deterministic failures
                    # (moderation filter 400s, bad key/model, malformed request)
                    # come back with retriable=False — identical input would
                    # reproduce them, so retrying only burns credits and delays
                    # the user's error by seconds of backoff.
                    normalized = exc if isinstance(exc, LLMError) else _normalize_provider_error(exc)
                    retriable = bool(getattr(normalized, "retriable", False))

                    # One-shot moderation nudge (see _moderation_nudge_user):
                    # on the first output-filter rejection, retry once with a
                    # clinical-tone suffix appended instead of failing right
                    # away. Exactly one nudge per chat call, small prompts only.
                    nudge_next = False
                    if (
                        isinstance(normalized, _ModerationFilteredError)
                        and attempt < MAX_RETRIES - 1
                        and not nudged
                    ):
                        replacement = _moderation_nudge_user(user)
                        if replacement is not None:
                            user = replacement
                            user_len = len(user)
                            nudged = True
                            nudge_next = True
                            # The nudged retry is a genuine second chance, so
                            # the filter rejection itself stays retriable for
                            # exactly this one follow-up attempt.
                            retriable = True

                    last_error = normalized
                    duration_ms = int((time.perf_counter() - attempt_t0) * 1000)
                    is_last = attempt >= MAX_RETRIES - 1 or not retriable
                    metrics.observe_llm_attempt(phase, "error" if is_last else "retry")
                    if is_last and not nudge_next:
                        metrics.observe_llm_error(
                            "moderation" if isinstance(normalized, _ModerationFilteredError)
                            else "retry" if retriable else "client"
                        )
                    logger.log(
                        logging.ERROR if is_last and not nudge_next else logging.WARNING,
                        "llm call %s phase=%s model=%s attempt=%d/%d duration_ms=%d error=%s",
                        "failed" if is_last else "retry",
                        phase,
                        model,
                        attempt + 1,
                        MAX_RETRIES,
                        duration_ms,
                        f"{type(normalized).__name__}: {normalized}",
                        exc_info=exc if is_last and not nudge_next else None,
                        extra={
                            "request_id": rid,
                            "phase": phase,
                            "status": "error" if is_last and not nudge_next else "retry",
                            "model": model,
                            "endpoint": endpoint,
                            "attempt": attempt + 1,
                            "retries": MAX_RETRIES,
                            "duration_ms": duration_ms,
                            "error_class": type(normalized).__name__,
                            "error_category": (
                                "moderation_nudge" if nudge_next
                                else "moderation" if isinstance(normalized, _ModerationFilteredError)
                                else "retry" if retriable else "client"
                            ),
                            "retryable": retriable,
                            "status_code": getattr(normalized, "status_code", None),
                            "upstream_request_id": getattr(normalized, "upstream_request_id", ""),
                        },
                    )
                    if is_last and not nudge_next:
                        # Hand the breaker the *reason*, not just the count: while
                        # OPEN it raises its own message, and a generic "endpoint
                        # unavailable" misreports a rejected key or an unusable
                        # model id — failures that never reach the model at all.
                        breaker.record_failure(
                            reason=_failure_reason(normalized), retriable=retriable
                        )
                        # The logical failed call is recorded *here* as well as after
                        # the loop. A deterministic failure (bad key, moderation
                        # filter) raises from inside the loop, so recording only at
                        # the bottom would omit the failures the error-rate alert is
                        # actually watching, and inflate errors/calls above 1.
                        metrics.observe_llm_call(
                            phase,
                            "error",
                            int((time.perf_counter() - t0) * 1000),
                            endpoint=endpoint,
                        )
                        raise normalized
                    if not is_last:
                        # A nudge retry does not sleep: the rejection was
                        # instantaneous, not a load/rate-limit signal.
                        if not nudge_next:
                            wait, honored = _retry_wait_seconds(attempt, exc)
                            # A 429's method of waiting is a rate-limit signal the
                            # operator can act on: 'retry_after' means the provider
                            # named a time (header present and honored); 'backoff'
                            # means no usable header arrived and the ladder chose.
                            # This is what answers "does Perplexity ever send the
                            # header?" — without it, an all-'backoff' split is
                            # indistinguishable from an untested honored path.
                            if _provider_status_code(exc) == 429:
                                metrics.observe_llm_rate_limit_retry(
                                    "retry_after" if honored else "backoff"
                                )
                            if honored:
                                # Visible at INFO so an operator can tell a
                                # header-driven wait from the ladder in the log
                                # — this is the line that proves throttling is
                                # being *obeyed*, not just absorbed.
                                logger.info(
                                    "llm retry-after honored phase=%s model=%s "
                                    "attempt=%d/%d wait=%.2fs",
                                    phase,
                                    model,
                                    attempt + 1,
                                    MAX_RETRIES,
                                    wait,
                                    extra={
                                        "request_id": rid,
                                        "phase": phase,
                                        "status": "retry",
                                        "model": model,
                                        "endpoint": endpoint,
                                        "attempt": attempt + 1,
                                        "retry_after": True,
                                    },
                                )
                            wait_with_cancellation(wait)
            # Exhausted retries — counts as one logical failure for the breaker.
            breaker.record_failure(
                reason=_failure_reason(last_error),
                retriable=bool(getattr(last_error, "retriable", True)),
            )
            metrics.observe_llm_call(
                phase, "error", int((time.perf_counter() - t0) * 1000), endpoint=endpoint
            )
            if isinstance(last_error, LLMTimeoutError):
                raise LLMTimeoutError(
                    f"LLM call failed after {MAX_RETRIES} attempts: {last_error}",
                    retriable=True,
                    status_code=last_error.status_code,
                    upstream_request_id=last_error.upstream_request_id,
                )
            if isinstance(last_error, _ModerationFilteredError):
                raise last_error
            if isinstance(last_error, LLMUpstreamError):
                raise LLMUpstreamError(
                    f"LLM call failed after {MAX_RETRIES} attempts: {last_error}",
                    retriable=last_error.retriable,
                    status_code=last_error.status_code,
                    upstream_request_id=last_error.upstream_request_id,
                )
            raise LLMError(f"LLM call failed after {MAX_RETRIES} attempts: {last_error}")
        except (CircuitBreakerOpenError, QueueFullError):
            # Re-raise without counting as a breaker failure and without extra logging
            # (concurrency limiter and breaker already logged at WARNING).
            raise
        finally:
            if acquired:
                limiter.release()

    # ------------------------------------------------------------------ json

    #: Upper bound on the first response that may be re-asked for *stochastic*
    #: parse failures (malformed JSON, markdown fences, etc.). A ~16k-character
    #: response that fails to parse is more often the model writing an essay than
    #: sampling noise, and re-asking doubles the spend on a call likely to miss
    #: again. Truncated responses (hit max_tokens mid-structure) bypass this cap
    #: — they are deterministic failures worth recovering, not stochastic ones.
    JSON_REASK_MAX_CHARS = 32_768

    #: One bounded re-ask: stochastically malformed output is worth a second
    #: chance; identical input is not worth a third (the moderation-nudge precedent:
    #: one attempt, then surface the failure).
    JSON_REASK_ATTEMPTS = 1

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 8000,
        phase: str = "general",
    ) -> Any:
        """Chat completion that must return a JSON document; parses it.

        When the first response cannot be parsed, the call is retried once with a
        repair instruction appended — malformed JSON is a *stochastic* model
        failure (the same prompt yields valid JSON on most calls), so a second
        draw usually recovers a chunk that would otherwise be dropped from the
        run's evidence. Every :meth:`chat` call inside goes through the usual
        retry/breaker/cancellation machinery, and the re-ask is bounded: after
        one retry the original :class:`LLMParseError` is raised unchanged, and
        responses above ``JSON_REASK_MAX_CHARS`` are not re-asked at all (a
        16k-character essay is more likely a wrong output mode than noise).
        """
        json_system = system + "\n\nRespond with ONLY valid JSON — no markdown fences, no commentary."
        text = self.chat(
            json_system,
            user,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            phase=phase,
        )
        try:
            return _parse_json(text)
        except LLMParseError as exc:
            failure = LLMParseError(
                f"Could not parse JSON for phase '{phase}' (response chars={len(text)})."
            )
            # Detect truncation: response ends mid-structure or hit max_tokens.
            # Truncated responses are deterministic failures worth recovering,
            # not stochastic noise — always re-ask them regardless of size.
            stripped = text.rstrip()
            is_truncated = (
                not stripped.endswith(("}", "]"))
                or stripped.endswith((",", ":", "{", "["))
                or len(text) >= max_tokens * 3  # Heuristic: ~4 chars/token
            )
            if self.JSON_REASK_ATTEMPTS <= 0:
                raise failure from exc
            # For non-truncated responses above the cap, fail fast (likely an essay)
            if len(text) > self.JSON_REASK_MAX_CHARS and not is_truncated:
                raise failure from exc
            logger.warning(
                "json parse failed — one re-ask before giving up phase=%s chars=%d truncated=%s",
                phase,
                len(text),
                is_truncated,
                extra={"phase": phase, "status": "retry", "truncated": is_truncated},
            )
            reask_system = json_system + (
                "\n\nYour previous response was not valid JSON. Return ONLY the JSON "
                "document — no prose, no markdown fences, no trailing commentary — "
                "matching the structure asked for."
            )
            retry_text = self.chat(
                reask_system,
                user,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                phase=phase,
            )
            try:
                return _parse_json(retry_text)
            except LLMParseError as retry_exc:
                raise failure from retry_exc


def _parse_json(text: str) -> Any:
    """Parse JSON from a model response, tolerating fences/prose around it."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:]
        candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Last resort: locate the outermost braces/brackets.
        for opener, closer in (("{", "}"), ("[", "]")):
            start, end = candidate.find(opener), candidate.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(candidate[start : end + 1])
                except json.JSONDecodeError:
                    continue
    raise LLMParseError(f"Could not parse JSON from model output (chars={len(text)}).")
