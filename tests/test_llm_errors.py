"""Offline tests for LLM provider-error classification and retry policy.

Regression coverage for the QwenCloud/Aliyun ``data_inspection_failed`` content
filter: it rejects the *model output* stochastically with HTTP 400, retrying
cannot fix it, and the user used to see only a generic "Drafting failed" error.
No network — the OpenAI client stub raises the simulated provider exceptions.

Uses the merged error taxonomy (``LLMUpstreamError`` with ``retriable`` /
``status_code`` / ``upstream_request_id``) plus the moderation-specific
``_ModerationFilteredError`` and one-shot clinical-tone nudge.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app.circuit_breaker import reset_all_for_tests  # noqa: E402
from app.llm import (  # noqa: E402
    MODERATION_NUDGE_MAX_USER_CHARS,
    LLMClient,
    LLMError,
    LLMParseError,
    LLMTimeoutError,
    LLMUpstreamError,
    _is_moderation_filtered,
    _is_transient_provider_error,
    _is_transient_status,
    _moderation_nudge_user,
    _normalize_provider_error,
    _provider_status_code,
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


class TestChatJsonReask(unittest.TestCase):
    """One unparseable JSON response must not permanently drop a chunk.

    Model output is sampled stochastically — measured 2026-09-20, 5 of 149 digest
    chunks in one run returned ~16k characters the JSON parser rejected while the
    same prompt parsed on retry. ``chat_json`` therefore re-asks once with a
    repair instruction before raising: bounded, logged, cancellation-aware (the
    re-ask goes through the ordinary ``chat`` path), and skipped entirely for
    oversized first responses where a second full-payload call is more likely to
    miss again than to recover.
    """

    def setUp(self) -> None:
        reset_all_for_tests()
        self.addCleanup(reset_all_for_tests)

    def _client(self, responses: list[str]) -> LLMClient:
        """A client whose ``chat`` pops scripted responses in order."""
        client = LLMClient(_FakeSettings())  # type: ignore[arg-type]
        script = iter(responses)
        chat_mock = MagicMock(side_effect=lambda *a, **k: next(script))
        client.chat = chat_mock  # type: ignore[method-assign]
        return client

    def test_a_malformed_response_is_recovered_by_one_reask(self) -> None:
        client = self._client(
            ["Here is the analysis you asked for, in prose, not JSON.", '{"ok": true}']
        )
        self.assertEqual(client.chat_json("sys", "user", phase="records:digest"), {"ok": True})
        self.assertEqual(client.chat.call_count, 2)

    def test_the_reask_carries_a_repair_instruction(self) -> None:
        client = self._client(["garbage", "[]"])
        client.chat_json("sys", "user")
        second_system = client.chat.call_args_list[1][0][0]
        self.assertIn("not valid JSON", second_system)
        self.assertIn("ONLY", second_system)

    def test_two_bad_responses_raise_with_the_original_message(self) -> None:
        client = self._client(["garbage one", "garbage two"])
        with self.assertRaises(LLMParseError) as ctx:
            client.chat_json("sys", "user", phase="grounding")
        self.assertIn("phase 'grounding'", str(ctx.exception))
        self.assertEqual(client.chat.call_count, 2, "bounded: exactly one re-ask")

    def test_an_oversized_response_is_not_reasked(self) -> None:
        """A ~16k-character essay is a wrong output mode, not sampling noise."""
        big_garbage = "x" * (LLMClient.JSON_REASK_MAX_CHARS + 1)
        client = self._client([big_garbage, "should never be consumed"])
        with self.assertRaises(LLMParseError):
            client.chat_json("sys", "user")
        self.assertEqual(client.chat.call_count, 1)

    def test_a_response_at_the_cap_is_reasked(self) -> None:
        at_cap = "x" * LLMClient.JSON_REASK_MAX_CHARS
        client = self._client([at_cap, '{"ok": 1}'])
        self.assertEqual(client.chat_json("sys", "user"), {"ok": 1})

    def test_valid_json_on_the_first_try_is_not_reasked(self) -> None:
        client = self._client(['{"ok": true}'])
        self.assertEqual(client.chat_json("sys", "user"), {"ok": True})
        self.assertEqual(client.chat.call_count, 1)

    def test_the_reask_flows_through_the_real_chat_path(self) -> None:
        """The re-ask must observe cancellation and the breaker like any other call.

        Proven against the *real* ``chat`` machinery: the first provider response is
        unparseable prose, the second is valid JSON — and no stubbing of ``chat``
        itself, so the re-ask demonstrably passes through retries, the limiter and
        the breaker.
        """
        client = LLMClient(_FakeSettings())  # type: ignore[arg-type]
        responses = iter(["prose, not json", "[1, 2]"])
        create_mock = MagicMock(
            side_effect=lambda **kwargs: MagicMock(
                choices=[MagicMock(message=MagicMock(content=next(responses)))],
                usage=MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        )
        client._client.chat.completions.create = create_mock  # type: ignore[attr-defined]
        self.assertEqual(client.chat_json("sys", "user"), [1, 2])
        self.assertEqual(create_mock.call_count, 2)


class TestDigestSurvivesOneBadChunk(unittest.TestCase):
    """End to end through the digest worker: a parse-blip chunk is recovered."""

    def test_the_chunk_that_once_died_now_parses_on_the_reask(self) -> None:
        import json as _json
        from unittest.mock import MagicMock as _MagicMock

        from tests.test_core import FakeLLM
        from app.documents import extract_document
        from app.medical_review import review_medical_records

        payload = {
            "facts": [
                {
                    "date": "2020-01",
                    "type": "symptom",
                    "description": "EVT one knee pain noted.",
                    "source": "",
                    "quote": "EVT one knee pain noted.",
                }
            ],
            "conditions_mentioned": ["knee pain"],
            "providers_and_facilities": [],
            "notes": "",
        }

        class ParseBlipLLM(FakeLLM):
            """FakeLLM, but the first digest chunk runs through a *real* client

            whose provider responses are scripted: prose first (the measured
            failure), valid JSON second — so the production re-ask path, not a
            stub of it, is what recovers the chunk.
            """

            def __init__(self) -> None:
                super().__init__()
                self._blipped = False
                self._real = LLMClient(_FakeSettings())
                responses = iter(["prose essay, no json here", _json.dumps(payload)])
                self._real.chat = _MagicMock(side_effect=lambda *a, **k: next(responses))

            def chat_json(self, system, user, **kwargs):
                if "CHUNK TEXT" in user and not self._blipped:
                    self._blipped = True
                    return self._real.chat_json(system, user, **kwargs)
                return super().chat_json(system, user, **kwargs)

        llm = ParseBlipLLM()
        doc = extract_document("a.txt", b"EVT one knee pain noted.")
        digest = review_medical_records(llm, [doc])
        self.assertEqual(digest.pages_reviewed, 1)
        self.assertEqual(llm._real.chat.call_count, 2, "the blip forced exactly one re-ask")
        self.assertTrue(
            any("knee pain" in f.description for f in digest.facts),
            "the chunk's facts must survive in the digest",
        )


def _client_with_create_raises(exc: Exception) -> tuple[LLMClient, MagicMock]:
    client = LLMClient(_FakeSettings())  # type: ignore[arg-type]
    create_mock = MagicMock(side_effect=exc)
    client._client.chat.completions.create = create_mock  # type: ignore[attr-defined]
    return client, create_mock


class TestNormalizeProviderError(unittest.TestCase):
    """The merged taxonomy: moderation / timeout / transient / deterministic."""

    def test_moderation_by_code(self):
        exc = _ProviderError(
            "Error code: 400 - {'error': {'code': 'data_inspection_failed', "
            "'message': 'Output data may contain inappropriate content.'}}",
            status_code=400,
        )
        normalized = _normalize_provider_error(exc)
        self.assertIsInstance(normalized, LLMUpstreamError)
        self.assertFalse(normalized.retriable)
        self.assertTrue(_is_moderation_filtered(exc))
        self.assertIn("content filter", str(normalized))

    def test_moderation_by_message_without_status(self):
        exc = _ProviderError("Output data may contain inappropriate content.")
        self.assertTrue(_is_moderation_filtered(exc))

    def test_timeout_maps_to_llm_timeout_error(self):
        normalized = _normalize_provider_error(TimeoutError("timed out"))
        self.assertIsInstance(normalized, LLMTimeoutError)
        self.assertTrue(normalized.retriable)

    def test_deterministic_4xx_not_retriable(self):
        for status in (400, 401, 404):
            normalized = _normalize_provider_error(_ProviderError("nope", status))
            self.assertIsInstance(normalized, LLMUpstreamError)
            self.assertFalse(normalized.retriable)
            self.assertEqual(normalized.status_code, status)

    def test_429_and_5xx_and_transport_retriable(self):
        for exc in (
            _ProviderError("rate limited", 429),
            _ProviderError("server oops", 503),
            RuntimeError("connection reset"),
        ):
            normalized = _normalize_provider_error(exc)
            self.assertTrue(normalized.retriable, msg=str(exc))

    def test_transient_classification(self):
        self.assertTrue(_is_transient_provider_error(_ProviderError("x", 503)))
        self.assertTrue(_is_transient_provider_error(_ProviderError("x", 429)))
        self.assertFalse(_is_transient_provider_error(_ProviderError("x", 401)))
        self.assertFalse(_is_transient_provider_error(_ProviderError("x", 404)))

    def test_499_client_disconnected_is_retriable(self):
        # 2026-09-22, req 8ed45557: a digest call died 166 s in with 499
        # client_disconnected and was classified as deterministic — no retry,
        # a ~20-chunk batch discarded, 3 files re-digested. A dropped
        # connection is a transport event, not a property of the request;
        # retrying the identical request plausibly succeeds.
        self.assertTrue(_is_transient_status(499))
        normalized = _normalize_provider_error(_ProviderError("Request canceled", 499))
        self.assertIsInstance(normalized, LLMUpstreamError)
        self.assertTrue(normalized.retriable, msg=str(normalized))

    def test_status_extraction(self):
        self.assertEqual(_provider_status_code(_ProviderError("x", 418)), 418)
        self.assertIsNone(_provider_status_code(RuntimeError("no status")))


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

    def test_an_empty_response_is_deterministic_and_never_retried(self):
        # Empty and whitespace-only completions are deterministic: identical input
        # reproduces them, so retrying only burns credits and delays the error.
        # (The one surviving piece of the retired backup branch 678376d — its
        # fail-fast flag is unnecessary now that the error taxonomy classifies
        # deterministic 4xxs as non-retriable, and its stub-client tests are
        # superseded by this module and test_llm_failover.)
        for content in ("", "   "):
            client = LLMClient(_FakeSettings())  # type: ignore[arg-type]
            resp = MagicMock()
            resp.choices = [MagicMock(message=MagicMock(content=content))]
            resp.usage = None
            create_mock = MagicMock(return_value=resp)
            client._client.chat.completions.create = create_mock  # type: ignore[attr-defined]
            with self.assertRaises(LLMError) as ctx:
                client.chat("system", "user", phase="draft")
            self.assertIn("empty response", str(ctx.exception))
            self.assertEqual(create_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
