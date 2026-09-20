"""OpenAI-compatible LLM client with retries, JSON-mode helpers, circuit breaker and concurrency limiting."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, NamedTuple
from urllib.parse import urlparse

from openai import NOT_GIVEN, OpenAI

from . import tracing
from . import metrics
from .circuit_breaker import CircuitBreakerOpenError, QueueFullError, get_llm_breaker, get_llm_limiter
from .config import FALLBACK_ENDPOINT, PRIMARY_ENDPOINT, Settings
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

# Re-export for callers that want to catch these specifically.
__all__ = [
    "LLMClient",
    "LLMError",
    "LLMConfigurationError",
    "LLMUpstreamError",
    "LLMTimeoutError",
    "LLMParseError",
    "CircuitBreakerOpenError",
    "QueueFullError",
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


class ChatProbe(NamedTuple):
    """Outcome of one short ``POST {base_url}/chat/completions`` check.

    A model *listing* is not a promise. Perplexity's Router API answers
    ``GET /models`` with its ids and then refuses every completion with ``403 The
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
    """
    if not api_key or not api_key.strip():
        return ChatProbe(None, "no API key was supplied")
    if not model or not model.strip():
        return ChatProbe(None, "no model was configured to call")
    url = base_url.rstrip("/") + "/chat/completions"
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


class LLMTimeoutError(LLMUpstreamError):
    """Raised when the upstream LLM call times out."""


class LLMParseError(LLMError):
    """Raised when the model response cannot be parsed into the expected format."""


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
    return bool(
        status_code is not None and (status_code in {408, 409, 425, 429} or 500 <= status_code < 600)
    )


def _is_transient_provider_error(exc: BaseException) -> bool:
    """True when retrying the same request could plausibly succeed."""
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


def _failure_reason(error: BaseException | None) -> str:
    """One bounded line naming a failure for the breaker's OPEN message.

    The class name is kept because it is the part that survives grepping the docs
    and the log (``LLMUpstreamError`` vs ``LLMTimeoutError``), and the message is
    already user-facing, so nothing here is withheld from it.
    """
    if error is None:
        return ""
    return f"{type(error).__name__}: {error}"


class LLMClient:
    """Thin wrapper around any OpenAI-compatible endpoint.

    Tracks an estimated-usage ``UsageTracker`` so callers can report per-phase
    token/call counts and (optionally) credit burn after a run.

    Each ``chat`` call is gated by a process-wide **circuit breaker** (opens
    after 3 consecutive logical failures, fail-fast in <2 s while open) and a
    **concurrency limiter** with a bounded queue (see ``app/circuit_breaker.py``
    and ``app/config.py``).
    """

    def __init__(self, settings: Settings) -> None:
        _validate_settings(settings)
        _validate_fallback_settings(settings)
        self._settings = settings
        _timeout_s = _configured_timeout_seconds()
        self._client = OpenAI(
            api_key=settings.api_key, base_url=settings.base_url, timeout=max(1.0, _timeout_s),
            max_retries=0,  # Application retries must observe pipeline cancellation.
        )
        # Built once here (not lazily on the failover path) so a bad fallback URL
        # or key is reported at startup rather than discovered mid-outage.
        self._fallback_client: OpenAI | None = None
        fallback = _fallback_target(settings)
        if fallback.configured:
            self._fallback_client = OpenAI(
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
        is full or times out, and ``LLMError`` when the provider call exhausts its
        retries. The breaker counts only *logical* call failures (one per call that
        exhausts retries), not per-attempt retries.
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
            for attempt in range(MAX_RETRIES):
                check_pipeline_cancelled()
                attempt_t0 = time.perf_counter()
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
                        response = self._endpoint_client(endpoint).chat.completions.create(
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            messages=[
                                {"role": "system", "content": system},
                                {"role": "user", "content": user},
                            ],
                            timeout=request_timeout,
                        )
                    check_pipeline_cancelled()
                    content = response.choices[0].message.content
                    if not content or not content.strip():
                        raise LLMError("Model returned an empty response.")
                    content = content.strip()
                    prompt_tokens, completion_tokens = _usage_tokens(response)
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
                            wait_with_cancellation(_retry_backoff_seconds(attempt))
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
        """Chat completion that must return a JSON document; parses it."""
        text = self.chat(
            system + "\n\nRespond with ONLY valid JSON — no markdown fences, no commentary.",
            user,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            phase=phase,
        )
        try:
            return _parse_json(text)
        except LLMParseError as exc:
            raise LLMParseError(
                f"Could not parse JSON for phase '{phase}' (response chars={len(text)})."
            ) from exc


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
