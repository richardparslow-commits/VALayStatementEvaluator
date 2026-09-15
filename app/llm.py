"""OpenAI-compatible LLM client with retries, JSON-mode helpers, circuit breaker and concurrency limiting."""

from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib.parse import urlparse

from openai import OpenAI

from .circuit_breaker import CircuitBreakerOpenError, QueueFullError, get_llm_breaker, get_llm_limiter
from .config import Settings
from .logging_config import get_request_id
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
    "check_model_availability",
]

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.0
MAX_RETRY_BACKOFF_SECONDS = 4.0
MODELS_ENDPOINT_TIMEOUT_SECONDS = 6

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


def check_model_availability(base_url: str, api_key: str) -> set[str] | None:
    """GET {base_url}/models and return the set of model ids, or None on failure.

    Best-effort only: any network, auth, or parse failure returns None so
    callers can silently skip the availability warning. Without an API key
    the check is not attempted. Uses only stdlib (urllib) so no extra deps.
    """
    if not api_key or not api_key.strip():
        return None
    url = base_url.rstrip("/") + "/models"
    try:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key.strip()}"})
        with urllib.request.urlopen(req, timeout=MODELS_ENDPOINT_TIMEOUT_SECONDS) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        rows = data.get("data", []) if isinstance(data, dict) else []
        ids: set[str] = set()
        for row in rows:
            mid = row.get("id") if isinstance(row, dict) else None
            if isinstance(mid, str) and mid.strip():
                ids.add(mid.strip())
        return ids
    except Exception:  # noqa: BLE001 - availability check is advisory only
        return None


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
        self._settings = settings
        _timeout_s = _configured_timeout_seconds()
        self._client = OpenAI(
            api_key=settings.api_key, base_url=settings.base_url, timeout=max(1.0, _timeout_s)
        )
        self.usage = UsageTracker()

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
        """Single-turn chat completion with retry, breaker, and concurrency guards.

        Raises ``CircuitBreakerOpenError`` (fail-fast, <2 s, no network) when
        the breaker is OPEN, ``QueueFullError`` when the concurrency queue is
        full or times out, and ``LLMError`` when the provider call exhausts its
        retries. The breaker counts only *logical* call failures (one per
        ``chat`` that exhausts retries), not per-attempt retries.
        """
        model = model or self._settings.model_main  # noqa: A001 - reassign param
        rid = get_request_id() or "-"

        # Fail fast before touching the limiter or the network.
        breaker = get_llm_breaker()
        breaker.check_or_raise()

        limiter = get_llm_limiter()
        # Acquire a concurrency slot (queues up to max_queue_depth, else QueueFullError).
        limiter.acquire()
        acquired = True
        try:
            # Re-check breaker after queuing — it may have opened while we waited.
            breaker.check_or_raise()

            sys_len = len(system or "")
            user_len = len(user or "")
            last_error: Exception | None = None
            nudged = False
            t0 = time.perf_counter()
            for attempt in range(MAX_RETRIES):
                attempt_t0 = time.perf_counter()
                try:
                    response = self._client.chat.completions.create(
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    )
                    content = response.choices[0].message.content
                    if not content or not content.strip():
                        raise LLMError("Model returned an empty response.")
                    content = content.strip()
                    prompt_tokens, completion_tokens = _usage_tokens(response)
                    self.usage.record(
                        model=model,
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
                            "attempt": attempt + 1,
                            "retries": MAX_RETRIES,
                            "duration_ms": duration_ms,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                        },
                    )
                    breaker.record_success()
                    return content
                except (CircuitBreakerOpenError, QueueFullError):
                    # Never count limiter/breaker rejections as endpoint failures.
                    raise
                except Exception as exc:  # noqa: BLE001 - retry on any provider error
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
                        breaker.record_failure()
                        raise normalized
                    if not is_last:
                        # A nudge retry does not sleep: the rejection was
                        # instantaneous, not a load/rate-limit signal.
                        if not nudge_next:
                            time.sleep(_retry_backoff_seconds(attempt))
            # Exhausted retries — counts as one logical failure for the breaker.
            breaker.record_failure()
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
