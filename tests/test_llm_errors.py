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
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app import config  # noqa: E402
from app.circuit_breaker import reset_all_for_tests  # noqa: E402
from app.llm import (  # noqa: E402
    MODERATION_NUDGE_MAX_USER_CHARS,
    LLMAuthError,
    LLMClient,
    LLMError,
    LLMParseError,
    LLMTimeoutError,
    LLMUpstreamError,
    _retry_after_hint,
    _retry_backoff_seconds,
    _retry_wait_seconds,
    _shutdown_pool_sockets,
    _stall_watchdog_seconds,
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


class TestAuthenticationClassification(unittest.TestCase):
    """A credential refusal is fatal and typed: exactly one attempt, own class.

    The 2026-09-22 incident: a key deactivated provider-side between two runs
    came back as 401 on every call. The retry loop burned all three attempts,
    the breaker opened on the third, and the batch runner's bisect — which had
    no way to know an auth refusal is not a file problem — quarantined nine
    healthy files. The fix is the type: LLMAuthError.
    """

    def test_401_normalizes_to_the_auth_type_not_generic_upstream(self) -> None:
        normalized = _normalize_provider_error(
            _ProviderError("Error code: 401 - invalid api key", 401)
        )
        self.assertIsInstance(normalized, LLMAuthError)
        # The subclass contract is load-bearing: every existing
        # `except LLMUpstreamError` (digest bookkeeping, advice selection) keeps
        # firing; only the fatal-policy branches key on the narrower type.
        self.assertIsInstance(normalized, LLMUpstreamError)
        self.assertFalse(normalized.retriable)
        self.assertEqual(normalized.status_code, 401)
        self.assertIn("401", str(normalized))
        self.assertIn("invalid api key", str(normalized))  # provider's own words survive

    def test_403_entitlement_refusal_is_the_same_fatal_type(self) -> None:
        normalized = _normalize_provider_error(_ProviderError("Error code: 403 - forbidden", 403))
        self.assertIsInstance(normalized, LLMAuthError)
        self.assertFalse(normalized.retriable)

    def test_auth_statuses_are_never_transient_even_outside_normalization(self) -> None:
        # _is_transient_provider_error is a seam other code paths build errors
        # through directly; it must not label 401/403 "retry may succeed".
        for status in (401, 403):
            self.assertFalse(_is_transient_provider_error(_ProviderError("x", status)))

    def test_rate_limit_and_outage_still_normalize_transient(self) -> None:
        for status in (429, 503):
            normalized = _normalize_provider_error(_ProviderError("x", status))
            self.assertNotIsInstance(normalized, LLMAuthError)
            self.assertTrue(normalized.retriable)


class TestAuthRefusalRetryPolicy(unittest.TestCase):
    """chat() spends exactly one attempt on a credential refusal — no ladder."""

    def setUp(self) -> None:
        self.sleep_patcher = patch("app.pipeline_guard.time.sleep", return_value=None)
        self.sleep_mock = self.sleep_patcher.start()
        self.addCleanup(self.sleep_patcher.stop)
        self.addCleanup(reset_all_for_tests)

    def test_401_takes_exactly_one_attempt_and_raises_the_auth_type(self) -> None:
        exc = _ProviderError("Error code: 401 - invalid api key", status_code=401)
        client, create_mock = _client_with_create_raises(exc)
        with self.assertRaises(LLMAuthError):
            client.chat("system", "user", phase="test")
        self.assertEqual(create_mock.call_count, 1)

    def test_a_429_still_takes_the_full_ladder(self) -> None:
        exc = _ProviderError("Error code: 429 - rate limited", status_code=429)
        client, create_mock = _client_with_create_raises(exc)
        with self.assertRaises(LLMUpstreamError) as ctx:
            client.chat("system", "user", phase="test")
        self.assertNotIsInstance(ctx.exception, LLMAuthError)
        self.assertEqual(create_mock.call_count, 3)


class TestRetryAfterHonoring(unittest.TestCase):
    """A 429's Retry-After demand is obeyed, not approximated by the ladder.

    The ladder (1s/2s/4s) under sustained throttling asks again before the
    provider said it would listen — burning attempts and tripping the breaker
    — so when the provider names a time, that time wins (capped, jittered).
    """

    def setUp(self) -> None:
        # wait_with_cancellation sleeps through pipeline_guard's time.sleep —
        # patch THERE or the header test takes the real 7-second wait.
        self.sleep_patcher = patch("app.pipeline_guard.time.sleep", return_value=None)
        self.sleep_mock = self.sleep_patcher.start()
        self.addCleanup(self.sleep_patcher.stop)
        self.addCleanup(reset_all_for_tests)

    @staticmethod
    def _ratelimit(value: str | None) -> Exception:
        """A shaped 429, with (or without) a Retry-After header."""
        exc = _ProviderError("Error code: 429 - rate limited", 429)
        exc.response = MagicMock(
            status_code=429, headers={} if value is None else {"retry-after": value}
        )
        return exc

    # ------------------------------------------------- header extraction/parsing

    def test_integer_seconds_is_parsed(self) -> None:
        self.assertEqual(_retry_after_hint(self._ratelimit("2")), 2.0)
        self.assertEqual(_retry_after_hint(self._ratelimit(" 12 ")), 12.0)
        self.assertEqual(_retry_after_hint(self._ratelimit("0")), 0.0)

    def test_http_date_form_is_parsed_against_the_local_clock(self) -> None:
        from email.utils import formatdate

        soon = formatdate(time.time() + 30, usegmt=True)
        hint = _retry_after_hint(self._ratelimit(soon))
        self.assertIsNotNone(hint)
        assert hint is not None
        self.assertGreater(hint, 20.0, "an HTTP-date 30s out must read as roughly 30s")
        self.assertLess(hint, 40.0)

    def test_a_stale_http_date_reads_as_zero_not_negative(self) -> None:
        from email.utils import formatdate

        stale = formatdate(time.time() - 60, usegmt=True)
        self.assertEqual(_retry_after_hint(self._ratelimit(stale)), 0.0)

    def test_unparseable_values_fall_through_to_none(self) -> None:
        self.assertIsNone(_retry_after_hint(self._ratelimit("soon")))
        self.assertIsNone(_retry_after_hint(self._ratelimit("")))
        self.assertIsNone(_retry_after_hint(self._ratelimit("-5")))

    def test_a_missing_header_or_missing_response_is_none(self) -> None:
        self.assertIsNone(_retry_after_hint(self._ratelimit(None)))
        self.assertIsNone(_retry_after_hint(Exception("no shape at all")))
        self.assertIsNone(_retry_after_hint(None))

    def test_a_mock_header_is_never_honored(self) -> None:
        # The MagicMock trap: a stub response's headers.get() returns a MagicMock,
        # and float(MagicMock()) would silently yield 1.0. Only a real string is
        # a header the provider actually sent.
        self.assertIsNone(_retry_after_hint(_ProviderError("x", 429)))

    # ----------------------------------------------------- the wait computation

    def test_honored_wait_is_capped_and_jittered(self) -> None:
        with patch.object(config, "LLM_RETRY_AFTER_MAX_SECONDS", 5.0):
            wait, honored = _retry_wait_seconds(0, self._ratelimit("30"))
        self.assertTrue(honored)
        self.assertGreaterEqual(wait, 5.1, "cap, plus at least the jitter floor")
        self.assertLessEqual(wait, 5.5, "cap, plus at most the jitter ceiling")

    def test_cap_zero_disables_honoring_entirely(self) -> None:
        with patch.object(config, "LLM_RETRY_AFTER_MAX_SECONDS", 0):
            wait, honored = _retry_wait_seconds(0, self._ratelimit("30"))
        self.assertFalse(honored)
        self.assertEqual(wait, _retry_backoff_seconds(0))

    def test_missing_header_falls_back_to_the_exponential_ladder(self) -> None:
        for attempt in (0, 1, 2):
            wait, honored = _retry_wait_seconds(attempt, self._ratelimit(None))
            self.assertFalse(honored)
            self.assertEqual(wait, _retry_backoff_seconds(attempt))

    # ------------------------------------------------------------ the live loop

    def test_the_retry_loop_sleeps_for_the_header_duration(self) -> None:
        exc = self._ratelimit("7")
        client = LLMClient(_FakeSettings())  # type: ignore[arg-type]
        ok = MagicMock()
        ok.choices = [MagicMock(message=MagicMock(content="ok response"))]
        ok.usage = None
        create_mock = MagicMock(side_effect=[exc, ok])
        client._client.chat.completions.create = create_mock  # type: ignore[attr-defined]

        with patch.object(config, "LLM_RETRY_AFTER_MAX_SECONDS", 60.0):
            out = client.chat("system", "user", phase="test")

        self.assertEqual(out, "ok response")
        self.assertEqual(create_mock.call_count, 2)
        self.sleep_mock.assert_called_once()
        waited = self.sleep_mock.call_args.args[0]
        self.assertGreaterEqual(waited, 7.1)
        self.assertLessEqual(waited, 7.5, "exactly the header (7s) plus jitter")

    def test_the_jitter_staggers_concurrent_wakes(self) -> None:
        # The thundering-herd case: N throttled workers receive the SAME header
        # and must not wake on the same instant. Jitter is per-call, so N
        # concurrently computed waits come out distinct.
        import threading

        barrier = threading.Barrier(12)
        waits: list[float] = []
        lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            wait, honored = _retry_wait_seconds(0, self._ratelimit("5"))
            with lock:
                waits.append(wait)

        with patch.object(config, "LLM_RETRY_AFTER_MAX_SECONDS", 60.0):
            threads = [threading.Thread(target=worker) for _ in range(12)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        self.assertEqual(len(waits), 12)
        self.assertGreaterEqual(
            len(set(waits)), 6, "concurrent wakes must be staggered, not synchronized"
        )


class TestStallWatchdog(unittest.TestCase):
    """The wall-clock rescue for a call whose connection died silently.

    With a warm keep-alive socket, "no bytes for N seconds" never accumulates,
    so no read timeout ever fires (2026-09-22: two digest workers parked in an
    SSL read for five hours while the sockets stayed ESTABLISHED). At
    multiplier × call-timeout the watchdog force-closes the pool's raw
    sockets, which turns the frozen read into the SDK's APIConnectionError —
    already classified retriable — and the ordinary ladder absorbs it.
    """

    def setUp(self) -> None:
        reset_all_for_tests()
        self.addCleanup(reset_all_for_tests)
        # The ladder's backoff sleeps through pipeline_guard's time.sleep —
        # patch THERE or the rescue test takes the real 1-second wait.
        self.sleep_patcher = patch("app.pipeline_guard.time.sleep", return_value=None)
        self.sleep_patcher.start()
        self.addCleanup(self.sleep_patcher.stop)

    @staticmethod
    def _ok() -> MagicMock:
        ok = MagicMock()
        ok.choices = [MagicMock(message=MagicMock(content="ok response"))]
        ok.usage = None
        return ok

    # ------------------------------------------------------------- the budget

    def test_budget_is_multiplier_times_call_timeout(self) -> None:
        with patch.object(config, "LLM_CALL_TIMEOUT_SECONDS", 300):
            with patch.object(config, "LLM_STALL_WATCHDOG_MULTIPLIER", 2.0):
                self.assertEqual(_stall_watchdog_seconds(), 600.0)
        with patch.object(config, "LLM_CALL_TIMEOUT_SECONDS", 100):
            with patch.object(config, "LLM_STALL_WATCHDOG_MULTIPLIER", 3.5):
                self.assertEqual(_stall_watchdog_seconds(), 350.0)

    def test_nonpositive_multiplier_disables_the_watchdog(self) -> None:
        with patch.object(config, "LLM_STALL_WATCHDOG_MULTIPLIER", 0):
            self.assertEqual(_stall_watchdog_seconds(), 0.0)

    def test_no_timer_is_armed_when_disabled(self) -> None:
        client = LLMClient(_FakeSettings())  # type: ignore[arg-type]
        armed = []

        class _FakeTimer:
            def __init__(self, *_args, **_kwargs) -> None:
                armed.append(self)

            def start(self) -> None: ...

            def cancel(self) -> None: ...

        with patch.object(config, "LLM_STALL_WATCHDOG_MULTIPLIER", 0):
            with patch("app.llm.threading.Timer", _FakeTimer):
                client._client.chat.completions.create = MagicMock(  # type: ignore[attr-defined]
                    return_value=self._ok()
                )
                out = client.chat("system", "user", phase="test")
        self.assertEqual(out, "ok response")
        self.assertEqual(armed, [], "a disabled watchdog must arm nothing")

    # ------------------------------------------------------------ the rescue

    def test_a_call_parked_past_the_budget_is_rescued_and_retried(self) -> None:
        # The parked read "dies" exactly when the watchdog closes the pool —
        # the physical sequence, replayed in miniature: timer fires → pool
        # closed → ConnectionError → ladder retries → success on the fresh
        # client.
        client = LLMClient(_FakeSettings())  # type: ignore[arg-type]
        rescued = threading.Event()
        calls = {"n": 0}

        def scripted(*_args, **_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                rescued.wait(timeout=5.0)  # parked until the pool is shut down
                raise ConnectionError("connection reset by pool shutdown")
            return self._ok()

        rebuild = MagicMock(name="rebuild_client")
        client._rebuild_client = rebuild  # type: ignore[method-assign]
        close_mock = MagicMock(side_effect=lambda: rescued.set())
        client._client.close = close_mock  # type: ignore[attr-defined]
        create_mock = MagicMock(side_effect=scripted)
        client._client.chat.completions.create = create_mock  # type: ignore[attr-defined]

        with patch.object(config, "LLM_CALL_TIMEOUT_SECONDS", 1):
            with patch.object(config, "LLM_STALL_WATCHDOG_MULTIPLIER", 0.2):
                out = client.chat("system", "user", phase="test")

        self.assertEqual(out, "ok response", "the ladder must absorb the rescue")
        # One arg, the endpoint: the retry must land on a fresh client.
        rebuild.assert_called_once_with("primary")
        close_mock.assert_called_once()
        self.assertEqual(create_mock.call_count, 2)

    def test_shutdown_pool_sockets_closes_the_raw_socket(self) -> None:
        # The real rescue mechanism: walking the pool object graph must reach
        # a socket.socket and shut it down (verified live 2026-09-23: this is
        # what unblocks a parked read; client.close() alone does not).
        import socket as _socket

        pair = _socket.socketpair()
        conn = MagicMock()
        conn._sock = pair[0]
        pool = MagicMock()
        pool.connections = [conn]
        transport = MagicMock()
        transport._pool = pool
        httpx_like = MagicMock()
        httpx_like._transport = transport
        holder = MagicMock()
        holder._client = httpx_like
        count = _shutdown_pool_sockets(holder)
        self.assertEqual(count, 1)
        self.assertEqual(pair[0].fileno(), -1, "the socket must be closed")
        pair[1].close()

    def test_unwalkable_pool_shapes_are_safely_ignored(self) -> None:
        weird = MagicMock(spec=["unrelated"])
        self.assertEqual(_shutdown_pool_sockets(weird), 0)
        self.assertEqual(_shutdown_pool_sockets(None), 0)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
