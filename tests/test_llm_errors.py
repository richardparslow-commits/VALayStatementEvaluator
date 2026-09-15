"""Offline tests for LLM provider-error classification and retry policy.

Regression coverage for the QwenCloud/Aliyun ``data_inspection_failed`` content
filter: it rejects the *model output* stochastically with HTTP 400, retrying
cannot fix it, and the user used to see only a generic "Drafting failed" error.
No network — the OpenAI client stub raises the simulated provider exceptions.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.circuit_breaker import reset_all_for_tests  # noqa: E402
from app.llm import (  # noqa: E402
    MODERATION_NUDGE_MAX_USER_CHARS,
    LLMClient,
    LLMError,
    _classify_provider_error,
    _error_status_code,
    _moderation_nudge_user,
)


class _FakeSettings:
    configured = True
    api_key = "test-key"
    base_url = "http://example.invalid"
    model_main = "test-model"
    model_fast = "test-fast"


class _ProviderError(Exception):
    """Shape-compatible stand-in for openai.StatusError / APIStatusError."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
        self.response = MagicMock(status_code=status_code)


def _client_with_create_raises(exc: Exception) -> tuple[LLMClient, MagicMock]:
    client = LLMClient(_FakeSettings())  # type: ignore[arg-type]
    create_mock = MagicMock(side_effect=exc)
    client._client.chat.completions.create = create_mock  # type: ignore[attr-defined]
    return client, create_mock


class TestClassifyProviderError(unittest.TestCase):
    def test_moderation_by_code(self):
        exc = _ProviderError(
            "Error code: 400 - {'error': {'code': 'data_inspection_failed', "
            "'message': 'Output data may contain inappropriate content.'}}",
            status_code=400,
        )
        self.assertEqual(_classify_provider_error(exc), "moderation")

    def test_moderation_by_message_without_status(self):
        exc = _ProviderError("Output data may contain inappropriate content.")
        self.assertEqual(_classify_provider_error(exc), "moderation")

    def test_client_4xx_without_429(self):
        self.assertEqual(_classify_provider_error(_ProviderError("bad key", 401)), "client")
        self.assertEqual(_classify_provider_error(_ProviderError("no model", 404)), "client")
        self.assertEqual(_classify_provider_error(_ProviderError("bad request", 400)), "client")

    def test_429_and_5xx_and_transport_retry(self):
        self.assertEqual(_classify_provider_error(_ProviderError("rate limited", 429)), "retry")
        self.assertEqual(_classify_provider_error(_ProviderError("server oops", 503)), "retry")
        self.assertEqual(_classify_provider_error(RuntimeError("connection reset")), "retry")
        self.assertEqual(_classify_provider_error(TimeoutError("timed out")), "retry")

    def test_status_extraction(self):
        self.assertEqual(_error_status_code(_ProviderError("x", 418)), 418)
        self.assertIsNone(_error_status_code(RuntimeError("no status")))


class TestModerationNudgeScope(unittest.TestCase):
    def test_small_prompt_gets_suffix(self):
        nudged = _moderation_nudge_user("user")
        assert nudged is not None
        self.assertTrue(nudged.startswith("user"))
        self.assertIn("TONE REQUIREMENT", nudged)

    def test_oversized_prompt_not_nudged(self):
        self.assertIsNone(_moderation_nudge_user("x" * (MODERATION_NUDGE_MAX_USER_CHARS + 1)))

    def test_empty_prompt_not_nudged(self):
        self.assertIsNone(_moderation_nudge_user(""))


class TestChatRetryPolicy(unittest.TestCase):
    """chat() must fail fast on deterministic 4xxs and keep its message."""

    def setUp(self) -> None:
        # Keep backoff sleeps out of the retry tests.
        self.sleep_patch = patch("app.llm.time.sleep", return_value=None)
        self.sleep_patch.start()

    def tearDown(self) -> None:
        self.sleep_patch.stop()
        reset_all_for_tests()

    def _run_chat(self, exc: Exception) -> str:
        client, _ = _client_with_create_raises(exc)
        try:
            client.chat("system", "user", phase="test")
        except Exception as caught:  # noqa: BLE001
            return f"{type(caught).__name__}: {caught}"
        return "no-error"

    def test_moderation_small_prompt_gets_one_nudge_then_fails(self):
        exc = _ProviderError(
            "Error code: 400 - {'error': {'code': 'data_inspection_failed', "
            "'message': 'Output data may contain inappropriate content.'}}",
            status_code=400,
        )
        client, create_mock = _client_with_create_raises(exc)
        with self.assertRaises(LLMError) as ctx:
            client.chat("system", "user", phase="draft")
        # Small prompt → exactly one clinical-tone nudge retry, then stop.
        self.assertEqual(create_mock.call_count, 2)
        # The nudged attempt appended the tone requirement.
        second_call_user = create_mock.call_args_list[1].kwargs["messages"][1]["content"]
        self.assertIn("TONE REQUIREMENT", second_call_user)
        self.assertTrue(second_call_user.endswith("formal benefits document, not creative writing."))
        # Actionable, non-generic message.
        self.assertIn("content filter", str(ctx.exception))
        self.assertIn("Retry", str(ctx.exception))
        # Sanity: the generic wrapper must NOT be used for this case.
        self.assertNotIn("LLM call failed after", str(ctx.exception))

    def test_moderation_large_prompt_fails_fast_without_nudge(self):
        exc = _ProviderError(
            "Error code: 400 - {'error': {'code': 'data_inspection_failed', "
            "'message': 'Output data may contain inappropriate content.'}}",
            status_code=400,
        )
        client, create_mock = _client_with_create_raises(exc)
        big_user = "x" * (MODERATION_NUDGE_MAX_USER_CHARS + 1)
        with self.assertRaises(LLMError) as ctx:
            client.chat("system", big_user, phase="draft")
        # Oversized prompt → no nudge; single attempt only.
        self.assertEqual(create_mock.call_count, 1)
        self.assertIn("content filter", str(ctx.exception))

    def test_moderation_nudge_retry_can_succeed(self):
        client = LLMClient(_FakeSettings())  # type: ignore[arg-type]
        create_mock = MagicMock()
        attempts = {"n": 0}

        def fake_create(**kwargs):  # type: ignore[no-untyped-def]
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise _ProviderError(
                    "Error code: 400 - data_inspection_failed: Output data may "
                    "contain inappropriate content.",
                    status_code=400,
                )
            resp = MagicMock()
            resp.choices = [MagicMock(message=MagicMock(content="ok response"))]
            resp.usage = None
            return resp

        create_mock.side_effect = fake_create
        client._client.chat.completions.create = create_mock  # type: ignore[attr-defined]
        result = client.chat("system", "user", phase="draft")
        self.assertEqual(result, "ok response")
        self.assertEqual(create_mock.call_count, 2)
        nudged_user = create_mock.call_args_list[1].kwargs["messages"][1]["content"]
        self.assertIn("TONE REQUIREMENT", nudged_user)

    def test_bad_key_401_fails_fast_and_keeps_provider_text(self):
        exc = _ProviderError("Error code: 401 - invalid api key", status_code=401)
        client, create_mock = _client_with_create_raises(exc)
        with self.assertRaises(LLMError) as ctx:
            client.chat("system", "user", phase="draft")
        self.assertEqual(create_mock.call_count, 1)
        self.assertIn("401", str(ctx.exception))

    def test_transient_5xx_still_retries_then_fails(self):
        exc = _ProviderError("Error code: 503 - overloaded", status_code=503)
        client, create_mock = _client_with_create_raises(exc)
        with self.assertRaises(LLMError):
            client.chat("system", "user", phase="draft")
        self.assertEqual(create_mock.call_count, 3)

    def test_transparent_success_after_transient_failure(self):
        client = LLMClient(_FakeSettings())  # type: ignore[arg-type]
        create_mock = MagicMock()
        attempts = {"n": 0}

        def fake_create(**kwargs):  # type: ignore[no-untyped-def]
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise _ProviderError("Error code: 503 - overloaded", status_code=503)
            resp = MagicMock()
            resp.choices = [MagicMock(message=MagicMock(content="ok response"))]
            resp.usage = None
            return resp

        create_mock.side_effect = fake_create
        client._client.chat.completions.create = create_mock  # type: ignore[attr-defined]
        self.assertEqual(client.chat("system", "user", phase="draft"), "ok response")
        self.assertEqual(create_mock.call_count, 3)


if __name__ == "__main__":
    unittest.main()
