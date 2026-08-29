"""Offline tests for the LLM client wrapper (retry, fail-fast, JSON parsing).

No network or API key required: the OpenAI client is replaced with a minimal
stub that simulates chat.completions.create behavior.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openai  # noqa: E402
from openai import BadRequestError, PermissionDeniedError  # noqa: E402

from app.config import Settings  # noqa: E402
from app.llm import LLMClient, LLMError  # noqa: E402


class _FakeRequest:
    """Minimal stand-in for httpx.Request that BadRequestError requires."""

    def __init__(self, method: str = "POST", url: str = "https://x/v1/chat/completions") -> None:
        self.method = method
        self.url = url

    @property
    def headers(self) -> dict[str, str]:
        return {}


class _FakeResponse:
    """Minimal stand-in for httpx.Response. The OpenAI exception base class reads
    `response.request` and `response.headers` in its constructor, so we mirror
    those attributes."""

    def __init__(self, status_code: int = 400, body: Any = None) -> None:
        self.status_code = status_code
        self._body = body
        self.request = _FakeRequest()
        self.headers: dict[str, str] = {}

    def json(self) -> Any:
        return self._body or {}


def _client_with_behavior(behavior: list[Exception | str]) -> LLMClient:
    """Build an LLMClient whose chat.completions.create returns the next item."""
    s = Settings(api_key="x", base_url="https://x", model_main="m", model_fast="m")
    c = LLMClient(s)
    calls = {"n": 0}

    class _Completions:
        def create(self, **kwargs: Any) -> Any:
            index = calls["n"]
            calls["n"] += 1
            outcome = behavior[min(index, len(behavior) - 1)]
            if isinstance(outcome, BaseException):
                raise outcome
            return type(
                "Resp",
                (),
                {"choices": [type("Choice", (), {"message": type("Msg", (), {"content": outcome})()})()]},
            )()

    class _Chat:
        completions = _Completions()

    c._client = type("Client", (), {"chat": _Chat()})()
    return c


class TestRetryBehavior(unittest.TestCase):
    def test_returns_on_first_success(self) -> None:
        c = _client_with_behavior(["hello"])
        self.assertEqual(c.chat("s", "u"), "hello")

    def test_retries_transient_then_succeeds(self) -> None:
        c = _client_with_behavior(
            [
                openai.APIConnectionError(request=_FakeRequest(), message="net"),
                "second try",
            ]
        )
        self.assertEqual(c.chat("s", "u"), "second try")

    def test_eventually_raises_after_three_attempts(self) -> None:
        c = _client_with_behavior(
            [
                openai.APIConnectionError(request=_FakeRequest(), message="x"),
                openai.APIConnectionError(request=_FakeRequest(), message="y"),
                openai.APIConnectionError(request=_FakeRequest(), message="z"),
            ]
        )
        with self.assertRaises(LLMError) as ctx:
            c.chat("s", "u")
        self.assertIn("after 3 attempt(s)", str(ctx.exception))


class TestFailFastBehavior(unittest.TestCase):
    def test_raise_for_status_short_circuits_retries(self) -> None:
        c = _client_with_behavior(
            [
                BadRequestError("bad", response=_FakeResponse(400), body={}),
                "should not be called",
            ]
        )
        with self.assertRaises(LLMError) as ctx:
            c.chat("s", "u", raise_for_status=True)
        self.assertIn("after 1 attempt(s)", str(ctx.exception))

    def test_non_retryable_auth_error_never_retries(self) -> None:
        c = _client_with_behavior(
            [
                PermissionDeniedError("denied", response=_FakeResponse(403), body={}),
                "should not be called",
            ]
        )
        with self.assertRaises(LLMError):
            c.chat("s", "u")
        # Even though raise_for_status is not set, the 403 should not be retried.

    def test_empty_response_is_deterministic(self) -> None:
        c = _client_with_behavior([""])
        with self.assertRaises(LLMError) as ctx:
            c.chat("s", "u")
        self.assertIn("empty response", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()