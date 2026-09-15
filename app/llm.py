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
    status_code = _provider_status_code(exc)
    upstream_request_id = _provider_request_id(exc)
    details = _provider_details(exc)
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
    return min(MAX_RETRY_BACKOFF_SECONDS, RETRY_BACKOFF_SECONDS * (2 ** attempt))


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
                    normalized = exc if isinstance(exc, LLMError) else _normalize_provider_error(exc)
                    retriable = bool(getattr(normalized, "retriable", False))
                    last_error = normalized
                    duration_ms = int((time.perf_counter() - attempt_t0) * 1000)
                    is_last = attempt >= MAX_RETRIES - 1 or not retriable
                    logger.log(
                        logging.ERROR if is_last else logging.WARNING,
                        "llm call %s phase=%s model=%s attempt=%d/%d duration_ms=%d error=%s",
                        "failed" if is_last else "retry",
                        phase,
                        model,
                        attempt + 1,
                        MAX_RETRIES,
                        duration_ms,
                        f"{type(normalized).__name__}: {normalized}",
                        exc_info=exc if is_last else None,
                        extra={
                            "request_id": rid,
                            "phase": phase,
                            "status": "error" if is_last else "retry",
                            "model": model,
                            "attempt": attempt + 1,
                            "retries": MAX_RETRIES,
                            "duration_ms": duration_ms,
                            "error_class": type(normalized).__name__,
                            "retryable": retriable,
                            "status_code": getattr(normalized, "status_code", None),
                            "upstream_request_id": getattr(normalized, "upstream_request_id", ""),
                        },
                    )
                    if is_last and not retriable:
                        breaker.record_failure()
                        raise normalized
                    if not is_last:
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
