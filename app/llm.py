"""OpenAI-compatible LLM client with retries and JSON-mode helpers."""
from __future__ import annotations

import json
import time
from typing import Any

from openai import BadRequestError, OpenAI
from openai import PermissionDeniedError

from .config import Settings

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0
# When set on a chat call, the client will raise LLMError on the first failure
# instead of retrying. Useful in batch jobs that want to fail fast and surface
# errors instead of burning through retry budget.
FAIL_FAST_NON_RETRYABLE: tuple[type[BaseException], ...] = (
    BadRequestError,
    PermissionDeniedError,
)


class LLMError(RuntimeError):
    """Raised when the LLM call ultimately fails or returns unusable output."""


class LLMClient:
    """Thin wrapper around any OpenAI-compatible endpoint.

    When `raise_for_status=True` is passed to `chat`, the client raises
    `LLMError` on the first failure instead of retrying — useful for batch
    callers that want to fail fast.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.configured:
            raise LLMError(
                "No API key configured. Add your key in the sidebar or in a .env file."
            )
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.api_key, base_url=settings.base_url, timeout=300.0
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
        raise_for_status: bool = False,
    ) -> str:
        """Single-turn chat completion with basic retry. Returns text.

        `raise_for_status=True` short-circuits retries and propagates the first
        provider error as `LLMError`. Non-retryable client errors (400/401/403)
        are never retried in either mode.
        """
        model = model or self._settings.model_main
        attempts = 1 if raise_for_status else MAX_RETRIES
        last_error: Exception | None = None
        for attempt in range(attempts):
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
                return content.strip()
            except LLMError:
                # Empty-response errors are deterministic; never retry them.
                raise
            except FAIL_FAST_NON_RETRYABLE as exc:
                raise LLMError(f"LLM call failed after 1 attempt(s) (non-retryable): {exc}") from exc
            except Exception as exc:  # noqa: BLE001 - retry on any provider error
                last_error = exc
                if attempt < attempts - 1:
                    time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
        raise LLMError(f"LLM call failed after {attempts} attempt(s): {last_error}")

    # ------------------------------------------------------------------ json
    def chat_json(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 8000,
        raise_for_status: bool = False,
    ) -> Any:
        """Chat completion that must return a JSON document; parses it."""
        text = self.chat(
            system + "\n\nRespond with ONLY valid JSON — no markdown fences, no commentary.",
            user,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            raise_for_status=raise_for_status,
        )
        return _parse_json(text)


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
    raise LLMError(f"Could not parse JSON from model output: {text[:300]}")
