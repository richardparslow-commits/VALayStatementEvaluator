"""The Responses wire path: Perplexity's Agent API as a first-class endpoint.

The default endpoint is ``https://api.perplexity.ai/v1`` (the Agent API), which
speaks the OpenAI **Responses** schema — ``input``/``instructions``/``max_output_tokens``
at ``{base_url}/responses`` — not Chat Completions. ``app/llm.py`` picks the schema
from the base URL's host, so these tests pin the split: Perplexity hosts send
Responses (and parse Responses answers and usage), every other host sends Chat
Completions unchanged, and the preflight probe speaks whichever dialect the run
will speak.

Live wire shapes come from https://docs.perplexity.ai/docs/agent-api/openai-compatibility:
the answer text lives in ``output_text`` (or ``output[*].content[*].text`` with
``type == "output_text"``) and usage in ``usage.input_tokens`` / ``usage.output_tokens``.
"""

from __future__ import annotations

import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app import config, llm  # noqa: E402


# --------------------------------------------------------------- shape helpers

def _chat_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=text))]
    resp.usage = None
    return resp


def _responses_body(
    text: str = "agent answer",
    *,
    status: str = "completed",
    input_tokens: int | None = 11,
    output_tokens: int | None = 7,
    with_output_text_key: bool = True,
) -> dict:
    body: dict = {"status": status, "output": [
        {"type": "message", "content": [
            {"type": "output_text", "text": text},
        ]},
    ]}
    if with_output_text_key:
        body["output_text"] = text
    usage: dict = {}
    if input_tokens is not None:
        usage["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens
    if usage:
        body["usage"] = usage
    return body


def _http_error(status: int, body: str = "") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://api.perplexity.ai/v1/responses",
        code=status,
        msg=f"HTTP {status}",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(body.encode("utf-8")),
    )


class _Resp(io.BytesIO):
    """Minimal urlopen response: a context manager that reads like a file."""

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _ok(payload: bytes):
    return patch("urllib.request.urlopen", return_value=_Resp(payload))


# ---------------------------------------------------------------- schema split

class TestSchemaSelection(unittest.TestCase):
    def test_default_base_url_is_the_agent_api(self) -> None:
        self.assertEqual(config.DEFAULT_BASE_URL, "https://api.perplexity.ai/v1")

    def test_perplexity_hosts_use_responses(self) -> None:
        for url in (
            config.DEFAULT_BASE_URL,
            "https://api.perplexity.ai/v1/",
            "https://api.perplexity.ai/router/v1",  # the retired default still selects Responses
            "https://www.api.perplexity.ai/v1",
        ):
            self.assertTrue(llm._uses_responses_schema(url), url)

    def test_other_hosts_use_chat_completions(self) -> None:
        for url in (
            "https://api.openai.com/v1",
            "https://ai-gateway.vercel.sh/v1",
            "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
            "http://localhost:11434/v1",
            "",
        ):
            self.assertFalse(llm._uses_responses_schema(url), url)

    def test_lookalike_host_is_not_perplexity(self) -> None:
        # Suffix matching must be domain-bound: a hostile lookalike must fall to
        # Chat Completions, not inherit the Responses path.
        self.assertFalse(llm._uses_responses_schema("https://api.perplexity.ai.evil.test/v1"))
        self.assertFalse(llm._uses_responses_schema("https://notperplexity.ai/v1"))


# ------------------------------------------------------- Responses payload text

class TestResponsesOutputText(unittest.TestCase):
    def test_output_text_key_wins(self) -> None:
        self.assertEqual(llm._responses_output_text(_responses_body()), "agent answer")

    def test_output_items_are_walked_when_the_key_is_missing(self) -> None:
        body = _responses_body(with_output_text_key=False)
        self.assertEqual(llm._responses_output_text(body), "agent answer")

    def test_sdk_typed_response_object_is_read(self) -> None:
        # The SDK's ``responses.create`` returns a typed object with attributes,
        # not a dict — a parser that only reads dicts would call a healthy answer
        # empty (found by the live-wire harness, not by the dict stubs).
        from types import SimpleNamespace

        obj = SimpleNamespace(
            output_text="agent answer",
            output=[SimpleNamespace(content=[SimpleNamespace(type="output_text", text="x")])],
            usage=SimpleNamespace(input_tokens=5, output_tokens=2),
        )
        self.assertEqual(llm._responses_output_text(obj), "agent answer")
        self.assertEqual(llm._responses_usage_tokens(obj), (5, 2))

    def test_empty_and_malformed_payloads_yield_empty(self) -> None:
        self.assertEqual(llm._responses_output_text({}), "")
        self.assertEqual(llm._responses_output_text({"output": "nope"}), "")
        self.assertEqual(llm._responses_output_text("not a dict"), "")
        self.assertEqual(llm._responses_output_text({"output_text": "   "}), "")

    def test_a_failed_run_body_yields_the_provider_error_not_generic_empty(self) -> None:
        body = _responses_body(text="", status="failed", with_output_text_key=False)
        body["error"] = {"message": "quota exhausted"}
        err = llm._responses_empty_error(body)
        self.assertIn("failed", str(err))
        self.assertIn("quota exhausted", str(err))
        self.assertNotIn("empty response", str(err))

    def test_an_incomplete_run_is_retriable(self) -> None:
        # Live evidence 2026-09-23: during a provider degradation, "status:
        # incomplete" hit in bursts while identical calls succeeded minutes
        # later — transient, so it must ride the retry ladder instead of
        # dying in one attempt and telling the breaker it was deterministic.
        body = _responses_body(text="", status="incomplete", with_output_text_key=False)
        err = llm._responses_empty_error(body)
        self.assertIn("incomplete", str(err))
        self.assertIsInstance(err, llm.LLMUpstreamError)
        self.assertTrue(err.retriable)

    def test_a_completed_but_empty_run_is_retriable(self) -> None:
        # Empty completions interleaved with incompletes in the same
        # degradation window share the burst signature.
        err = llm._responses_empty_error(_responses_body(text="  "))
        self.assertIn("empty response", str(err))
        self.assertIsInstance(err, llm.LLMUpstreamError)
        self.assertTrue(err.retriable)

    def test_a_failed_run_stays_deterministic(self) -> None:
        # A failed run WITH a provider reason (bad model id, quota) is a
        # request problem, not a burst — the ladder must not burn attempts on it.
        body = _responses_body(text="", status="failed", with_output_text_key=False)
        body["error"] = {"message": "quota exhausted"}
        err = llm._responses_empty_error(body)
        self.assertFalse(getattr(err, "retriable", False))

    def test_a_completed_run_with_no_text_still_says_empty(self) -> None:
        err = llm._responses_empty_error(_responses_body(text="  "))
        self.assertIn("empty response", str(err))

    def test_the_error_object_is_read_off_the_sdk_typed_shape(self) -> None:
        # Through the real SDK the error object arrives as a typed attribute
        # object, not a dict — found by the live wire harness.
        from types import SimpleNamespace

        obj = SimpleNamespace(
            status="failed",
            error=SimpleNamespace(message="invalid model id: nope/bogus"),
        )
        self.assertEqual(llm._responses_error_message(obj), "invalid model id: nope/bogus")
        self.assertIn("nope/bogus", str(llm._responses_empty_error(obj)))

    def test_chat_output_text_reads_choices(self) -> None:
        self.assertEqual(llm._chat_output_text(_chat_response(" hi ")), "hi")
        self.assertEqual(llm._chat_output_text(MagicMock(choices=[])), "")
        self.assertEqual(llm._chat_output_text(object()), "")


class TestResponsesUsage(unittest.TestCase):
    def test_input_output_tokens_are_read(self) -> None:
        self.assertEqual(llm._responses_usage_tokens(_responses_body()), (11, 7))

    def test_missing_or_negative_usage_is_none(self) -> None:
        self.assertEqual(llm._responses_usage_tokens(_responses_body(
            input_tokens=None, output_tokens=None)), (None, None))
        self.assertEqual(llm._responses_usage_tokens(_responses_body(
            input_tokens=-1, output_tokens=-2)), (None, None))
        self.assertEqual(llm._responses_usage_tokens(_responses_body(
            input_tokens=None, output_tokens=None)), (None, None))

    def test_serialized_input_is_the_user_turn_with_system_in_instructions(self) -> None:
        # The system prompt travels in `instructions` only. It used to be joined
        # into `input` as well — the endpoint accepted the duplication but billed
        # the system block twice on every call, and the digest prompt re-sends
        # its rubric preamble hundreds of times per run.
        self.assertEqual(llm._responses_input("sys", "usr"), "usr")
        self.assertEqual(llm._responses_input("", "usr"), "usr")


# ----------------------------------------------------- wire-schema resolution


class TestEndpointSchemaResolution(unittest.TestCase):
    """The wire schema is resolved from the endpoint's capability, not just its host.

    Resolution is cached per base URL and hermetic.py neutralizes the route
    probe for the whole suite, so every test here sees the documented host
    guess (the probe returning "cannot tell") and no network. The override
    knob is the deterministic way to test the probe's decision itself.
    """

    def setUp(self) -> None:
        self._override = patch.object(config, "LLM_ENDPOINT_SCHEMA", "")
        self._override.start()
        self.addCleanup(self._override.stop)
        llm._SCHEMA_CACHE.clear()
        self.addCleanup(llm._SCHEMA_CACHE.clear)

    def test_the_perplexity_host_guess_is_responses(self) -> None:
        self.assertTrue(llm._uses_responses_schema("https://api.perplexity.ai/v1"))

    def test_any_other_host_guesses_chat(self) -> None:
        self.assertFalse(llm._uses_responses_schema("https://openai-compatible.invalid/v1"))

    def test_the_env_override_wins_over_the_host_guess(self) -> None:
        with patch.object(config, "LLM_ENDPOINT_SCHEMA", "chat"):
            self.assertFalse(llm._uses_responses_schema("https://api.perplexity.ai/v1"))
        with patch.object(config, "LLM_ENDPOINT_SCHEMA", "responses"):
            self.assertTrue(llm._uses_responses_schema("https://gateway.invalid/v1"))

    def test_an_unrecognized_override_is_ignored(self) -> None:
        with patch.object(config, "LLM_ENDPOINT_SCHEMA", "completions-ish"):
            self.assertTrue(llm._uses_responses_schema("https://api.perplexity.ai/v1"))


class TestHeadRouteProbe(unittest.TestCase):
    """The probe's own status vocabulary, with urllib stubbed out.

    Calibrated live 2026-09-22 against the Agent API: HEAD /responses -> 405
    (route exists, HEAD not an allowed method), HEAD /chat/completions -> 404,
    HEAD /nope -> 404. Auth and rate statuses say nothing about the route;
    neither does a dead network — "cannot tell" (None) is the only honest
    answer for those, and resolution falls back to the host guess.
    """

    @staticmethod
    def _urlopen_error(status: int) -> Exception:
        return urllib.error.HTTPError(
            url="https://x.invalid/responses", code=status, msg="x", hdrs={}, fp=io.BytesIO(b"")
        )

    def _resolved(self, code: int | None) -> bool | None:
        boom = (
            urllib.error.URLError("connection refused")
            if code is None
            else self._urlopen_error(code)
        )
        with patch.object(llm.urllib.request, "urlopen", side_effect=boom):
            # *_original: hermetic.py replaces the module attribute with a
            # never-network stub for the session; the vocabulary under test
            # here is the real implementation, with urllib stubbed instead.
            return llm._head_route_exists_original("https://x.invalid/responses")

    def test_405_means_the_route_exists(self) -> None:
        self.assertIs(self._resolved(405), True)

    def test_404_and_410_mean_it_does_not(self) -> None:
        self.assertIs(self._resolved(404), False)
        self.assertIs(self._resolved(410), False)

    def test_auth_rate_and_server_statuses_cannot_tell(self) -> None:
        for code in (401, 403, 429, 500):
            self.assertIsNone(self._resolved(code))

    def test_a_network_failure_cannot_tell(self) -> None:
        self.assertIsNone(self._resolved(None))

    def test_affirmative_probe_evidence_outranks_the_host_guess(self) -> None:
        # 405 on /responses and 404 on /chat/completions is affirmative evidence
        # — it must resolve "responses" even on a NON-Perplexity host, which is
        # the whole point of capability routing. "Cannot tell" (None) falls back
        # to the host guess instead.
        def fake_head(url: str) -> bool | None:
            if url.endswith("/responses"):
                return True
            if url.endswith("/chat/completions"):
                return False
            return None

        with patch.object(config, "LLM_ENDPOINT_SCHEMA", ""):
            with patch.object(llm, "_head_route_exists", side_effect=fake_head):
                self.assertTrue(
                    llm._uses_responses_schema("https://responses-only-provider.invalid/v1")
                )
            with patch.object(llm, "_head_route_exists", side_effect=lambda url: None):
                self.assertFalse(llm._uses_responses_schema("https://mystery-proxy.invalid/v1"))


# --------------------------------------------------------- run-time wire calls

class TestResponsesWireCall(unittest.TestCase):
    """``LLMClient.chat`` against a stubbed SDK client, for both schemas."""

    def _client(self, base_url: str) -> llm.LLMClient:
        settings = config.Settings(
            api_key="k",
            base_url=base_url,
            model_main="main-model",
            model_fast="fast-model",
            fetch_api_key="",
            fetch_base_url="",
            fetch_records_path="",
        )
        return llm.LLMClient(settings)

    def test_perplexity_endpoint_sends_a_responses_request(self) -> None:
        client = self._client("https://api.perplexity.ai/v1")
        responses = MagicMock()
        responses.create.return_value = _responses_body()
        client._client.responses = responses

        out = client.chat("be brief", "hello", phase="t")

        self.assertEqual(out, "agent answer")
        kwargs = responses.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "main-model")
        self.assertEqual(kwargs["input"], "hello")
        self.assertEqual(kwargs["instructions"], "be brief")
        self.assertEqual(kwargs["max_output_tokens"], 8000)
        self.assertNotIn("messages", kwargs)
        self.assertNotIn("max_tokens", kwargs)
        # Usage came from the Responses usage keys, proving the right parser ran.
        self.assertEqual(client.usage.summary()["prompt_tokens"], 11)

    def test_blank_system_prompt_omits_instructions(self) -> None:
        client = self._client("https://api.perplexity.ai/v1")
        responses = MagicMock()
        responses.create.return_value = _responses_body()
        client._client.responses = responses

        client.chat("", "hello", phase="t")

        kwargs = responses.create.call_args.kwargs
        self.assertEqual(kwargs["input"], "hello")
        self.assertNotIn("instructions", kwargs)

    def test_empty_visible_text_is_still_an_error_on_responses(self) -> None:
        client = self._client("https://api.perplexity.ai/v1")
        responses = MagicMock()
        responses.create.return_value = {"status": "completed", "output_text": "  "}
        client._client.responses = responses

        with self.assertRaises(llm.LLMError):
            client.chat("s", "u", phase="t")

    def test_runs_are_sent_with_store_false(self) -> None:
        # The app's payloads carry medical records; runs are not continued via
        # previous_response_id, so server-side storage is opted out of.
        client = self._client("https://api.perplexity.ai/v1")
        responses = MagicMock()
        responses.create.return_value = _responses_body()
        client._client.responses = responses

        client.chat("s", "u", phase="t")

        self.assertIs(responses.create.call_args.kwargs["store"], False)

    def test_chat_endpoint_sends_messages_unchanged(self) -> None:
        client = self._client("https://openai-compatible.invalid/v1")
        completions = MagicMock()
        completions.create.return_value = _chat_response("chat answer")
        client._client.chat.completions = completions

        out = client.chat("be brief", "hello", max_tokens=123, phase="t")

        self.assertEqual(out, "chat answer")
        kwargs = completions.create.call_args.kwargs
        self.assertEqual(kwargs["max_tokens"], 123)
        self.assertEqual(kwargs["messages"], [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hello"},
        ])
        self.assertNotIn("max_output_tokens", kwargs)


# ------------------------------------------------------------------- the probe

class TestProbeChatDialects(unittest.TestCase):
    """The preflight probe must speak the same schema the run will speak."""

    def test_perplexity_base_url_posts_to_responses_with_responses_payload(self) -> None:
        captured: dict = {}

        def fake_urlopen(req, timeout):  # noqa: ANN001
            captured["url"] = req.full_url
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return _Resp(json.dumps(_responses_body()).encode("utf-8"))

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            probe = llm.probe_chat("https://api.perplexity.ai/v1", "k", "perplexity/kimi-k3")

        self.assertTrue(probe.ok)
        self.assertEqual(probe.reply, "agent answer")
        self.assertEqual(captured["url"], "https://api.perplexity.ai/v1/responses")
        self.assertEqual(captured["payload"]["input"], llm.CHAT_PROBE_PROMPT)
        self.assertEqual(captured["payload"]["max_output_tokens"], llm.CHAT_PROBE_MAX_TOKENS)
        self.assertNotIn("messages", captured["payload"])
        self.assertNotIn("max_tokens", captured["payload"])

    def test_responses_run_that_completed_without_text_is_silent_ok(self) -> None:
        body = _responses_body(text="", with_output_text_key=False)
        with _ok(json.dumps(body).encode("utf-8")):
            probe = llm.probe_chat("https://api.perplexity.ai/v1", "k", "m")
        self.assertTrue(probe.ok)
        self.assertTrue(probe.silent)
        self.assertEqual(probe.reply, "")

    def test_responses_incomplete_status_is_reported_as_an_error(self) -> None:
        body = _responses_body(text="", status="incomplete", with_output_text_key=False)
        with _ok(json.dumps(body).encode("utf-8")):
            probe = llm.probe_chat("https://api.perplexity.ai/v1", "k", "m")
        self.assertFalse(probe.ok)
        self.assertIn("incomplete", probe.error)

    def test_chat_base_url_keeps_the_chat_payload_and_route(self) -> None:
        captured: dict = {}

        def fake_urlopen(req, timeout):  # noqa: ANN001
            captured["url"] = req.full_url
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return _Resp(json.dumps({"choices": [
                {"message": {"content": "chat answer"}},
            ]}).encode("utf-8"))

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            probe = llm.probe_chat("https://openai-compatible.invalid/v1", "k", "m")

        self.assertTrue(probe.ok)
        self.assertEqual(probe.reply, "chat answer")
        self.assertEqual(captured["url"], "https://openai-compatible.invalid/v1/chat/completions")
        self.assertIn("messages", captured["payload"])
        self.assertNotIn("input", captured["payload"])

    def test_403_on_responses_carries_the_provider_message(self) -> None:
        with patch("urllib.request.urlopen", side_effect=_http_error(
                403, '{"error": {"message": "no entitlement"}}')):
            probe = llm.probe_chat("https://api.perplexity.ai/v1", "k", "m")
        self.assertEqual(probe.status, 403)
        self.assertIn("no entitlement", probe.error)

    def test_a_failed_run_under_http_200_names_the_provider_error(self) -> None:
        # The Agent API returns 200 for a run that then failed server-side; the
        # reason lives in error.message. The probe must surface it, not report
        # a generic empty-completion.
        body = _responses_body(text="", status="failed", with_output_text_key=False)
        body["error"] = {"message": "model is over capacity"}
        with _ok(json.dumps(body).encode("utf-8")):
            probe = llm.probe_chat("https://api.perplexity.ai/v1", "k", "m")
        self.assertFalse(probe.ok)
        self.assertIn("failed", probe.error)
        self.assertIn("model is over capacity", probe.error)

    def test_the_probe_does_not_ask_the_endpoint_to_store_anything(self) -> None:
        captured: dict = {}

        def fake_urlopen(req, timeout):  # noqa: ANN001
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return _Resp(json.dumps(_responses_body()).encode("utf-8"))

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.probe_chat("https://api.perplexity.ai/v1", "k", "m")
        self.assertIs(captured["payload"].get("store"), False)


if __name__ == "__main__":
    unittest.main()
