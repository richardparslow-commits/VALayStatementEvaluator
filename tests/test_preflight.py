"""Tests for the endpoint preflight: policy, gate, and the notice it leaves.

Like every module in this suite, it imports the hermetic harness first: the app
freezes its configuration at import, and a test file that imports the app before
the harness tests whatever `.env` this machine happens to have. The settings here
are built explicitly and the probe is canned, so these tests never read ambient
configuration and never open a socket.
"""

from __future__ import annotations

import json
import sys
import types
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app.llm import CHAT_PROBE_MAX_TOKENS, ChatProbe, ModelProbe, probe_chat  # noqa: E402
from app import preflight  # noqa: E402


def _settings(**over: object) -> SimpleNamespace:
    base: dict[str, object] = {
        "base_url": "https://api.example.test/v1",
        "api_key": "test-key",
        "model_main": "model-main",
        "model_fast": "model-fast",
    }
    base.update(over)
    return SimpleNamespace(**base)


def _probe(models=None, status=None, error="") -> ModelProbe:
    return ModelProbe(set(models) if models else None, status, error)


def _chat(
    status=None,
    error="URLError: name or service not known",
    reply="",
    silent=False,
) -> ChatProbe:
    return ChatProbe(status, error, reply, silent)


#: A short chat call that answered. The commonest canned outcome in these tests, so it
#: has a name rather than a `_chat(200, "", "ok")` at every call site.
_ANSWERED = _chat(200, "", "ok")


def _check(
    probe_result: ModelProbe,
    settings: SimpleNamespace | None = None,
    *,
    chat: ChatProbe | None = None,
    chat_calls: list[str] | None = None,
) -> preflight.Verdict:
    """Run the policy against canned probe results.

    ``chat`` defaults to a call that got no response, which is the behaviour these
    tests had before the chat call existed: the model listing decides alone. Pass
    ``chat=`` for the second probe's outcome, and ``chat_calls`` to record which models
    it was asked about (an empty list is the assertion that it was not asked at all).
    """
    canned = _chat() if chat is None else chat

    def fake_chat(base_url: str, api_key: str, model: str) -> ChatProbe:
        if chat_calls is not None:
            chat_calls.append(model)
        return canned

    return preflight.check_endpoint(
        settings or _settings(),
        probe=lambda base_url, api_key: probe_result,
        chat_probe=fake_chat,
    )


class TestPreflightPolicy(unittest.TestCase):
    """Which outcomes stop a run, and which merely get reported."""

    def test_all_models_listed_is_a_pass(self) -> None:
        verdict = _check(_probe({"model-main", "model-fast", "other"}, 200))
        self.assertEqual(verdict.kind, preflight.OK)
        self.assertFalse(verdict.blocks)
        self.assertEqual(verdict.listed, 3)

    def test_a_model_the_endpoint_does_not_offer_blocks_the_run(self) -> None:
        verdict = _check(_probe({"model-main"}, 200))
        self.assertTrue(verdict.blocks)
        self.assertEqual(verdict.missing, ("model-fast",))
        self.assertIn("model-fast", verdict.headline)
        self.assertIn("Test connection", verdict.fix)

    def test_a_dated_variant_of_the_configured_id_counts_as_present(self) -> None:
        """Providers serve `qwen3.7-max` as `qwen3.7-max-2025-04-16`; that is the same model."""
        verdict = _check(
            _probe({"model-main-2025-04-16", "model-fast:latest"}, 200),
            _settings(model_main="model-main", model_fast="model-fast"),
        )
        self.assertEqual(verdict.kind, preflight.OK)

    def test_the_reverse_direction_does_not_count(self) -> None:
        """A configured id longer than anything listed is a different model."""
        verdict = _check(_probe({"model-main"}, 200), _settings(model_main="model-main-turbo"))
        self.assertTrue(verdict.blocks)

    def test_a_rejected_key_blocks_and_names_the_status(self) -> None:
        for status in (401, 403):
            with self.subTest(status=status):
                verdict = _check(_probe(None, status, f"HTTP {status}: nope"))
                self.assertTrue(verdict.blocks)
                self.assertIn(f"HTTP {status}", verdict.headline)
                self.assertIn("Router access", verdict.fix)
                # The key and the base URL are one credential pair: a mismatched
                # path reads as a rejected key, so say that before blaming the key.
                self.assertIn("same provider account", verdict.fix)

    def test_a_missing_models_route_is_reported_but_does_not_block(self) -> None:
        """A server can serve completions without listing models."""
        verdict = _check(_probe(None, 404, "HTTP 404: Not Found"))
        self.assertEqual(verdict.kind, preflight.UNVERIFIED)
        self.assertFalse(verdict.blocks)
        self.assertIn("404", verdict.headline)

    def test_no_response_does_not_block(self) -> None:
        verdict = _check(_probe(None, None, "URLError: name or service not known"))
        self.assertEqual(verdict.kind, preflight.UNVERIFIED)
        self.assertIn("no response", verdict.headline)
        self.assertIn("name or service not known", verdict.fix)
        # Point the blame at the host, not the key: one failed probe is not proof
        # that calls will fail, and the run stays allowed.
        self.assertIn("not the key", verdict.fix)

    def test_a_provider_side_failure_does_not_block(self) -> None:
        verdict = _check(_probe(None, 503, "HTTP 503: unavailable"))
        self.assertEqual(verdict.kind, preflight.UNVERIFIED)
        self.assertIn("HTTP 503", verdict.headline)

    def test_a_200_with_no_ids_does_not_block(self) -> None:
        verdict = _check(_probe(None, 200, "published no model ids"))
        self.assertEqual(verdict.kind, preflight.UNVERIFIED)

    def test_no_configured_models_does_not_block(self) -> None:
        """The app falls back to the provider default, so there is nothing to compare."""
        verdict = _check(_probe({"whatever"}, 200), _settings(model_main="", model_fast="  "))
        self.assertEqual(verdict.kind, preflight.UNVERIFIED)
        self.assertIn("No model names", verdict.headline)

    def test_a_blank_key_is_not_a_blocked_run(self) -> None:
        """Local OpenAI-compatible servers legitimately need no key at all."""
        verdict = _check(_probe(None, None, "no API key was supplied"), _settings(api_key=""))
        self.assertFalse(verdict.blocks)

    def test_only_the_blocked_verdict_blocks(self) -> None:
        self.assertTrue(preflight.Verdict(preflight.BLOCKED, headline="x").blocks)
        for kind in (preflight.OK, preflight.UNVERIFIED):
            with self.subTest(kind=kind):
                self.assertFalse(preflight.Verdict(kind, headline="x").blocks)


class TestTheChatCallAfterTheListing(unittest.TestCase):
    """The second probe: `/models` is a listing, a listing is not a promise.

    Perplexity's Router API is the endpoint that forced this. It answers the listing
    with its own ids and then refuses every completion with `403 The Router API is
    currently in limited preview` until the account is granted access (measured against
    the live app, where the listing alone reported a healthy endpoint and the run failed
    142 times). So the chat probe can only ever *strengthen* a verdict: a refusal blocks,
    an answer confirms, and anything ambiguous leaves the listing's verdict alone.
    """

    ROUTER = "https://api.perplexity.ai/router/v1"
    LISTING_OK = _probe({"model-main", "model-fast", "other"}, 200)

    def test_a_listed_model_that_refuses_every_call_blocks_the_run(self) -> None:
        verdict = _check(
            self.LISTING_OK,
            _settings(base_url=self.ROUTER),
            chat=_chat(403, "HTTP 403: The Router API is currently in limited preview."),
        )

        self.assertTrue(verdict.blocks)
        self.assertIn("refused a real call", verdict.headline)
        self.assertIn("403", verdict.headline)
        self.assertIn("limited preview", verdict.fix, "the provider's own words are the point")
        self.assertIn("api@perplexity.ai", verdict.fix, "and so is the remedy")

    def test_the_router_note_is_not_attached_to_another_endpoint(self) -> None:
        verdict = _check(self.LISTING_OK, chat=_chat(403, "HTTP 403: nope"))

        self.assertTrue(verdict.blocks)
        self.assertIn("nope", verdict.fix)
        self.assertNotIn("limited preview", verdict.fix)

    def test_a_refused_fast_model_blocks_even_when_the_main_one_answers(self) -> None:
        """The bulk digest runs on the fast model, so its refusal is fatal on its own."""
        asked: list[str] = []

        def per_model(base_url: str, api_key: str, model: str) -> ChatProbe:
            asked.append(model)
            return _ANSWERED if model == "model-main" else _chat(403, "HTTP 403: not entitled")

        verdict = preflight.check_endpoint(
            _settings(), probe=lambda base_url, api_key: self.LISTING_OK, chat_probe=per_model
        )

        self.assertTrue(verdict.blocks)
        self.assertIn("model-fast", verdict.headline)
        self.assertEqual(asked, ["model-main", "model-fast"])

    def test_an_answered_call_confirms_the_configuration(self) -> None:
        verdict = _check(self.LISTING_OK, chat=_ANSWERED)

        self.assertEqual(verdict.kind, preflight.OK)
        self.assertFalse(verdict.blocks)
        self.assertIn("offers every configured model", verdict.headline)
        self.assertIn("real call", verdict.headline)

    def test_a_silent_reasoning_answer_still_confirms_the_endpoint(self) -> None:
        """The model wrote nothing, but the call came back — that is what was checked.

        A reasoning model can spend the short call's budget thinking; the endpoint is
        proven by the call being served at all, and the headline has to say that rather
        than read like a failure.
        """
        verdict = _check(self.LISTING_OK, chat=_chat(200, "", silent=True))

        self.assertEqual(verdict.kind, preflight.OK)
        self.assertFalse(verdict.blocks)
        self.assertIn("was served", verdict.headline)
        self.assertIn("no visible text", verdict.headline)

    def test_a_rate_limit_is_not_a_refusal(self) -> None:
        """The key worked; a limit is the account's business and the run decides."""
        verdict = _check(self.LISTING_OK, chat=_chat(429, "HTTP 429: rate limited"))

        self.assertEqual(verdict.kind, preflight.OK)
        self.assertFalse(verdict.blocks)

    def test_a_payload_the_endpoint_rejects_leaves_the_listing_verdict(self) -> None:
        """A 400 may be about this probe's shape rather than about any real call."""
        verdict = _check(self.LISTING_OK, chat=_chat(400, "HTTP 400: bad request"))

        self.assertEqual(verdict.kind, preflight.OK)

    def test_no_response_leaves_the_listing_verdict_alone(self) -> None:
        verdict = _check(self.LISTING_OK, chat=_chat())

        self.assertEqual(verdict.kind, preflight.OK)
        self.assertNotIn("real call", verdict.headline)

    def test_the_call_upgrades_an_endpoint_that_publishes_no_listing(self) -> None:
        """No `/models` route but a real call works is better than "unverified"."""
        verdict = _check(_probe(None, 404, "HTTP 404: Not Found"), chat=_ANSWERED)

        self.assertEqual(verdict.kind, preflight.OK)
        self.assertFalse(verdict.blocks)

    def test_the_call_can_block_an_endpoint_with_no_listing_route(self) -> None:
        verdict = _check(_probe(None, 404, "HTTP 404: Not Found"), chat=_chat(403, "HTTP 403: refused"))

        self.assertTrue(verdict.blocks)
        self.assertIn("refused a real call", verdict.headline)
        self.assertNotIn(
            "models route answered",
            verdict.fix,
            "there was no listing here, so the fix must not claim one",
        )

    def test_a_missing_completions_route_blocks_too(self) -> None:
        """A 404 on the call is as deterministic as a rejected key, unlike one on `/models`."""
        verdict = _check(self.LISTING_OK, chat=_chat(404, "HTTP 404: Not Found"))

        self.assertTrue(verdict.blocks)
        self.assertIn("404", verdict.headline)
        self.assertEqual(verdict.listed, 3, "the listing is still the evidence it was")

    def test_a_blocked_listing_is_never_second_guessed_by_a_call(self) -> None:
        """One request is enough when the listing already proves the run cannot work."""
        asked: list[str] = []
        verdict = _check(_probe(None, 403, "HTTP 403: rejected"), chat_calls=asked)

        self.assertTrue(verdict.blocks)
        self.assertEqual(asked, [], "the gate is supposed to be cheap when it already knows")

    def test_no_configured_models_means_nothing_to_call(self) -> None:
        asked: list[str] = []
        verdict = _check(
            _probe({"whatever"}, 200),
            _settings(model_main="", model_fast="  "),
            chat_calls=asked,
        )

        self.assertEqual(verdict.kind, preflight.UNVERIFIED)
        self.assertEqual(asked, [])

    def test_the_same_model_configured_twice_is_called_once(self) -> None:
        asked: list[str] = []
        _check(
            _probe({"same"}, 200),
            _settings(model_main="same", model_fast="same"),
            chat_calls=asked,
        )

        self.assertEqual(asked, ["same"])


class TestTheChatProbeItself(unittest.TestCase):
    """``probe_chat``'s own mapping, over a stubbed transport — this suite opens no socket.

    The policy above is tested against canned outcomes; this is what turns a socket into
    those outcomes, so its shapes are worth pinning: which URL, what payload, and what
    each answer means.
    """

    def _stub(self, body: object, status: int = 200) -> MagicMock:
        payload = body if isinstance(body, bytes) else json.dumps(body).encode()
        response = MagicMock()
        response.read.return_value = payload
        response.status = status
        response.__enter__ = lambda _: response
        response.__exit__ = lambda *_: False
        return response

    def test_a_completion_is_the_proof(self) -> None:
        body = {"choices": [{"message": {"content": "ok"}}]}
        with patch("urllib.request.urlopen", return_value=self._stub(body)):
            result = probe_chat("https://api.example.test/v1", "example-key", "model-main")

        self.assertTrue(result.ok)
        self.assertEqual(result.status, 200)
        self.assertEqual(result.reply, "ok")

    def test_it_asks_for_a_short_answer_with_the_configured_model(self) -> None:
        body = {"choices": [{"message": {"content": "ok"}}]}
        with patch("urllib.request.urlopen", return_value=self._stub(body)) as urlopen:
            probe_chat("https://api.example.test/v1/", "example-key", "model-main")

        request = urlopen.call_args[0][0]
        self.assertEqual(request.full_url, "https://api.example.test/v1/chat/completions")
        self.assertEqual(request.get_method(), "POST")
        self.assertIn("Bearer example-key", request.headers.get("Authorization", ""))
        sent = json.loads(request.data.decode("utf-8"))
        self.assertEqual(sent["model"], "model-main")
        self.assertEqual(sent["max_tokens"], CHAT_PROBE_MAX_TOKENS)
        self.assertEqual(sent["messages"][0]["role"], "user")

    def test_the_budget_leaves_room_for_a_reasoning_model_to_answer(self) -> None:
        """A single token is spent entirely on hidden reasoning by some models.

        Measured on ``alibaba/qwen3.7-flash`` (Vercel AI Gateway): at ``max_tokens=1``
        the answer is 200 with an empty message, ``finish_reason`` "length" and 230
        reasoning tokens; the same ask answers normally once the budget allows for the
        thinking, so the probe has to budget for the thinking and not just the words.
        """
        self.assertGreater(CHAT_PROBE_MAX_TOKENS, 1)

    def test_a_refusal_keeps_the_providers_own_words(self) -> None:
        body = b'{"error":{"message":"The Router API is currently in limited preview."}}'
        error = HTTPError("https://api.example.test/v1/chat/completions", 403, "Forbidden", None, BytesIO(body))
        with patch("urllib.request.urlopen", side_effect=error):
            result = probe_chat("https://api.example.test/v1", "example-key", "model-main")

        self.assertFalse(result.ok)
        self.assertEqual(result.status, 403)
        self.assertIn("limited preview", result.error)

    def test_a_200_without_a_completion_is_not_a_pass(self) -> None:
        with patch("urllib.request.urlopen", return_value=self._stub({"choices": []})):
            result = probe_chat("https://api.example.test/v1", "example-key", "model-main")

        self.assertFalse(result.ok)
        self.assertFalse(result.silent)
        self.assertEqual(result.status, 200)
        self.assertIn("without a completion", result.error)

    def test_a_reasoning_model_with_no_visible_text_is_still_a_served_call(self) -> None:
        """The measured shape of a budget-starved reasoning answer.

        ``choices[0].message`` is present and empty, with ``finish_reason: "length"``:
        the endpoint served the call and the budget went to hidden reasoning, so this
        must not read as "answered without a completion".
        """
        body = {
            "choices": [
                {"finish_reason": "length", "message": {"role": "assistant", "content": ""}}
            ]
        }
        with patch("urllib.request.urlopen", return_value=self._stub(body)):
            result = probe_chat("https://api.example.test/v1", "example-key", "model-main")

        self.assertTrue(result.ok)
        self.assertTrue(result.silent)
        self.assertEqual(result.status, 200)
        self.assertEqual(result.reply, "")
        self.assertEqual(result.error, "")

    def test_content_parts_are_read_too(self) -> None:
        """Some servers answer with a list of content parts instead of a string."""
        body = {"choices": [{"message": {"content": [{"type": "text", "text": "ok"}]}}]}
        with patch("urllib.request.urlopen", return_value=self._stub(body)):
            result = probe_chat("https://api.example.test/v1", "example-key", "model-main")

        self.assertTrue(result.ok)

    def test_no_key_means_no_request(self) -> None:
        with patch("urllib.request.urlopen") as urlopen:
            result = probe_chat("https://api.example.test/v1", "", "model-main")

        urlopen.assert_not_called()
        self.assertEqual(result.status, None)
        self.assertIn("no API key", result.error)

    def test_a_dead_endpoint_has_no_status(self) -> None:
        with patch("urllib.request.urlopen", side_effect=Exception("name or service not known")):
            result = probe_chat("https://api.example.test/v1", "example-key", "model-main")

        self.assertEqual(result.status, None)
        self.assertIn("name or service not known", result.error)

    def test_the_timeout_is_longer_than_a_listing_s_ceiling(self) -> None:
        """A call has to reach a model, which is slower than reading a list."""
        from app.llm import CHAT_PROBE_TIMEOUT_SECONDS, MODELS_ENDPOINT_TIMEOUT_SECONDS

        self.assertGreater(CHAT_PROBE_TIMEOUT_SECONDS, MODELS_ENDPOINT_TIMEOUT_SECONDS)


class TestSignature(unittest.TestCase):
    def test_it_changes_with_every_setting_that_matters(self) -> None:
        base = preflight.signature(_settings())
        variants = {
            "base_url": _settings(base_url="https://other.test/v1"),
            "api_key": _settings(api_key="other-key"),
            "model_main": _settings(model_main="other-main"),
            "model_fast": _settings(model_fast="other-fast"),
        }
        for label, settings in variants.items():
            with self.subTest(changed=label):
                self.assertNotEqual(preflight.signature(settings), base)

    def test_it_is_stable_for_the_same_configuration(self) -> None:
        self.assertEqual(preflight.signature(_settings()), preflight.signature(_settings()))

    def test_a_trailing_slash_is_not_a_different_endpoint(self) -> None:
        self.assertEqual(
            preflight.signature(_settings(base_url="https://api.example.test/v1/")),
            preflight.signature(_settings()),
        )

    def test_the_key_itself_never_appears_in_the_signature(self) -> None:
        """The signature is a session key that may reach a log line."""
        secret = "pplx-super-secret-value"
        self.assertNotIn(secret, preflight.signature(_settings(api_key=secret)))

    def test_a_blank_key_is_still_identifiable(self) -> None:
        self.assertIn("no-key", preflight.signature(_settings(api_key="")))


class TestVerdictReuse(unittest.TestCase):
    """A verdict stands in for a probe only for its own configuration, and only briefly."""

    def _verdict(self) -> preflight.Verdict:
        return preflight.Verdict(preflight.OK, headline="all good")

    def _store(self, settings, verdict, *, now: float) -> dict:
        store: dict = {}
        preflight.remember_verdict(store, settings, verdict, now=now)
        return store

    def test_a_fresh_verdict_for_the_same_configuration_is_reused(self) -> None:
        settings = _settings()
        verdict = self._verdict()
        store = self._store(settings, verdict, now=1_000.0)

        self.assertIs(preflight.reusable_verdict(store, settings, now=1_010.0), verdict)

    def test_a_verdict_goes_stale_after_the_window(self) -> None:
        settings = _settings()
        verdict = self._verdict()
        store = self._store(settings, verdict, now=1_000.0)
        boundary = 1_000.0 + preflight.VERDICT_REUSE_SECONDS

        self.assertIs(preflight.reusable_verdict(store, settings, now=boundary), verdict)
        self.assertIsNone(preflight.reusable_verdict(store, settings, now=boundary + 0.001))

    def test_a_verdict_timestamped_in_the_future_is_not_reused(self) -> None:
        """A clock that went backwards is not evidence that anything was checked."""
        settings = _settings()
        store = self._store(settings, self._verdict(), now=1_010.0)

        self.assertIsNone(preflight.reusable_verdict(store, settings, now=1_000.0))

    def test_a_verdict_is_only_evidence_for_its_own_configuration(self) -> None:
        settings = _settings()
        store = self._store(settings, self._verdict(), now=1_000.0)

        for changed in (
            _settings(api_key="other-key"),
            _settings(base_url="https://elsewhere.example/v1"),
            _settings(model_main="other-main"),
            _settings(model_fast="other-fast"),
        ):
            with self.subTest(changed=preflight.signature(changed)):
                self.assertIsNone(preflight.reusable_verdict(store, changed, now=1_010.0))

    def test_a_missing_entry_is_simply_unreusable(self) -> None:
        self.assertIsNone(preflight.reusable_verdict({}, _settings(), now=1_000.0))

    def test_junk_in_the_store_is_ignored_rather_than_trusted(self) -> None:
        settings = _settings()
        key = preflight.VERDICT_SESSION_KEY
        for junk in (
            None,
            "not an entry",
            {},
            {"signature": preflight.signature(settings)},
            {"signature": preflight.signature(settings), "at": "yesterday", "verdict": self._verdict()},
            {"signature": preflight.signature(settings), "at": 1_000.0, "verdict": {"not": "a verdict"}},
        ):
            with self.subTest(junk=repr(junk)[:48]):
                self.assertIsNone(
                    preflight.reusable_verdict({key: junk}, settings, now=1_010.0)
                )

    def test_reading_a_verdict_does_not_refresh_it(self) -> None:
        """Only a probe refreshes the window: a read ages from when the probe ran,
        so a session that keeps checking cannot postpone a real check forever."""
        settings = _settings()
        store = self._store(settings, self._verdict(), now=1_000.0)
        boundary = 1_000.0 + preflight.VERDICT_REUSE_SECONDS

        self.assertIsNotNone(preflight.reusable_verdict(store, settings, now=1_010.0))
        self.assertIsNotNone(preflight.reusable_verdict(store, settings, now=boundary))
        self.assertEqual(store[preflight.VERDICT_SESSION_KEY]["at"], 1_000.0)
        self.assertIsNone(preflight.reusable_verdict(store, settings, now=boundary + 0.001))

    def test_remembering_again_replaces_the_earlier_verdict(self) -> None:
        settings = _settings()
        store = self._store(settings, self._verdict(), now=1_000.0)
        newer = preflight.Verdict(preflight.BLOCKED, headline="now refused")
        preflight.remember_verdict(store, settings, newer, now=1_005.0)

        self.assertIs(preflight.reusable_verdict(store, settings, now=1_006.0), newer)


class _FakeSession(dict):
    """Mirrors Streamlit's SessionState: item access and attribute access."""

    def __getattr__(self, name: str) -> object:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _fake_st() -> MagicMock:
    """A MagicMock `st` whose session_state is a real mapping.

    Locally defined rather than imported from tests.test_views: a test module that
    imports another test module inherits its import order and its fixtures, and this
    file needs one small fake, not that whole surface.
    """
    st_mock = MagicMock()
    st_mock.session_state = _FakeSession()
    st_mock.expander.return_value.__enter__.return_value = st_mock
    return st_mock


class TestEndpointGate(unittest.TestCase):
    """The gate itself: what it stores, what it logs, and what it clears."""

    def _run(self, shared, st_mock, verdict, *, action="draft", log_action="draft"):
        with patch.object(shared, "st", st_mock), patch.object(
            shared, "session_settings", return_value=_settings()
        ), patch.object(shared.preflight, "check_endpoint", return_value=verdict), patch.object(
            shared, "run_log_event"
        ) as log_event:
            allowed = shared.check_endpoint_gate(action, log_action=log_action, request_id="req_1")
        return allowed, log_event

    def test_a_pass_allows_the_run_and_clears_any_stale_block(self) -> None:
        import app.views.shared as shared

        st_mock = _fake_st()
        st_mock.session_state[shared._endpoint_block_key("draft")] = MagicMock()
        allowed, log_event = self._run(st_mock=st_mock, shared=shared, verdict=preflight.Verdict(preflight.OK, "fine"))
        self.assertTrue(allowed)
        self.assertNotIn(shared._endpoint_block_key("draft"), st_mock.session_state)
        log_event.assert_not_called()

    def test_a_block_refuses_the_run_logs_it_and_stores_the_verdict(self) -> None:
        import app.views.shared as shared

        st_mock = _fake_st()
        verdict = preflight.Verdict(preflight.BLOCKED, headline="The endpoint rejected this API key.")
        allowed, log_event = self._run(st_mock=st_mock, shared=shared, verdict=verdict)
        self.assertFalse(allowed)
        self.assertIs(st_mock.session_state[shared._endpoint_block_key("draft")], verdict)
        log_event.assert_called_once()
        action, status = log_event.call_args[0][0], log_event.call_args[0][1]
        self.assertEqual((action, status), ("draft", "rejected"))
        self.assertEqual(log_event.call_args[1]["reason"], "endpoint_preflight")
        self.assertEqual(log_event.call_args[1]["request_id"], "req_1")

    def test_the_log_action_is_separate_from_the_user_facing_one(self) -> None:
        """The Evaluate tab says "evaluation" to the user and "evaluate" in the log."""
        import app.views.shared as shared

        st_mock = _fake_st()
        _, log_event = self._run(
            shared=shared,
            st_mock=st_mock,
            verdict=preflight.Verdict(preflight.BLOCKED, headline="blocked"),
            action="evaluation",
            log_action="evaluate",
        )
        self.assertEqual(log_event.call_args[0][0], "evaluate")
        self.assertIn(shared._endpoint_block_key("evaluation"), st_mock.session_state)

    def test_a_fresh_test_connection_verdict_skips_the_probes(self) -> None:
        """The button already paid for this exact check; the gate must not pay twice."""
        import app.views.shared as shared

        st_mock = _fake_st()
        verdict = preflight.Verdict(preflight.OK, headline="fine")
        preflight.remember_verdict(st_mock.session_state, _settings(), verdict)
        with patch.object(shared, "st", st_mock), patch.object(
            shared, "session_settings", return_value=_settings()
        ), patch.object(
            shared.preflight, "check_endpoint", side_effect=AssertionError("probed anyway")
        ):
            allowed = shared.check_endpoint_gate("draft", log_action="draft", request_id="req_1")
        self.assertTrue(allowed)
        self.assertNotIn(shared._endpoint_block_key("draft"), st_mock.session_state)

    def test_a_verdict_for_another_configuration_does_not_skip_the_probes(self) -> None:
        import app.views.shared as shared

        st_mock = _fake_st()
        preflight.remember_verdict(
            st_mock.session_state,
            _settings(api_key="some-other-key"),
            preflight.Verdict(preflight.OK, headline="fine"),
        )
        with patch.object(shared, "st", st_mock), patch.object(
            shared, "session_settings", return_value=_settings()
        ), patch.object(
            shared.preflight, "check_endpoint", return_value=preflight.Verdict(preflight.OK, "fresh")
        ) as check:
            allowed = shared.check_endpoint_gate("draft", log_action="draft", request_id="req_1")
        self.assertTrue(allowed)
        check.assert_called_once()

    def test_the_gate_remembers_its_own_verdict_for_the_next_run(self) -> None:
        """The first run pays for the probes; the next attempt on the same
        configuration must not pay again while the evidence is fresh."""
        import app.views.shared as shared

        st_mock = _fake_st()
        fresh = preflight.Verdict(preflight.OK, headline="fresh")
        with patch.object(shared, "st", st_mock), patch.object(
            shared, "session_settings", return_value=_settings()
        ), patch.object(shared.preflight, "check_endpoint", return_value=fresh) as check:
            self.assertTrue(
                shared.check_endpoint_gate("draft", log_action="draft", request_id="req_1")
            )
            check.assert_called_once()

        with patch.object(shared, "st", st_mock), patch.object(
            shared, "session_settings", return_value=_settings()
        ), patch.object(
            shared.preflight, "check_endpoint", side_effect=AssertionError("probed again")
        ):
            for request_id in ("req_2", "req_3"):
                self.assertTrue(
                    shared.check_endpoint_gate("draft", log_action="draft", request_id=request_id)
                )

    def test_reusing_does_not_postpone_the_next_real_check(self) -> None:
        """The window runs from the probe, not from the last reuse."""
        import app.views.shared as shared

        st_mock = _fake_st()
        clock = {"t": 1_000.0}
        with patch.object(
            shared.preflight.time, "monotonic", side_effect=lambda: clock["t"]
        ), patch.object(shared, "st", st_mock), patch.object(
            shared, "session_settings", return_value=_settings()
        ), patch.object(
            shared.preflight, "check_endpoint", return_value=preflight.Verdict(preflight.OK, "fresh")
        ) as check:
            shared.check_endpoint_gate("draft", log_action="draft", request_id="req_1")
            self.assertEqual(check.call_count, 1)

            clock["t"] = 1_000.0 + preflight.VERDICT_REUSE_SECONDS - 1.0
            shared.check_endpoint_gate("draft", log_action="draft", request_id="req_2")
            check.assert_called_once()  # still fresh: reused, not re-probed

            clock["t"] = 1_000.0 + preflight.VERDICT_REUSE_SECONDS + 1.0
            shared.check_endpoint_gate("draft", log_action="draft", request_id="req_3")
            self.assertEqual(check.call_count, 2)  # stale on schedule: probed again

    def test_a_reused_block_still_refuses_the_run(self) -> None:
        import app.views.shared as shared

        st_mock = _fake_st()
        verdict = preflight.Verdict(preflight.BLOCKED, headline="The endpoint rejected this API key.")
        preflight.remember_verdict(st_mock.session_state, _settings(), verdict)
        with patch.object(shared, "st", st_mock), patch.object(
            shared, "session_settings", return_value=_settings()
        ), patch.object(
            shared.preflight, "check_endpoint", side_effect=AssertionError("probed anyway")
        ), patch.object(shared, "run_log_event") as log_event:
            allowed = shared.check_endpoint_gate("draft", log_action="draft", request_id="req_1")
        self.assertFalse(allowed)
        self.assertIs(st_mock.session_state[shared._endpoint_block_key("draft")], verdict)
        log_event.assert_called_once()

    def test_a_waiver_allows_that_configuration_and_clears_the_notice(self) -> None:
        import app.views.shared as shared

        st_mock = _fake_st()
        signature = preflight.signature(_settings())
        st_mock.session_state[shared.endpoint_waiver_key("draft", signature)] = True
        allowed, log_event = self._run(
            shared=shared,
            st_mock=st_mock,
            verdict=preflight.Verdict(preflight.BLOCKED, headline="blocked"),
        )
        self.assertTrue(allowed)
        self.assertNotIn(shared._endpoint_block_key("draft"), st_mock.session_state)
        log_event.assert_not_called()

    def test_a_waiver_does_not_leak_onto_a_different_configuration(self) -> None:
        """Changing the endpoint, key, or models has to ask the user again."""
        import app.views.shared as shared

        st_mock = _fake_st()
        st_mock.session_state[shared.endpoint_waiver_key("draft", "old|config|sig|abcd")] = True
        allowed, log_event = self._run(
            shared=shared,
            st_mock=st_mock,
            verdict=preflight.Verdict(preflight.BLOCKED, headline="blocked"),
        )
        self.assertFalse(allowed)
        log_event.assert_called_once()

    def test_an_unverified_result_allows_the_run(self) -> None:
        import app.views.shared as shared

        st_mock = _fake_st()
        allowed, log_event = self._run(
            shared=shared,
            st_mock=st_mock,
            verdict=preflight.Verdict(preflight.UNVERIFIED, headline="no answer"),
        )
        self.assertTrue(allowed)
        log_event.assert_not_called()


class TestEndpointNotice(unittest.TestCase):
    def test_a_matching_block_renders_the_reason_and_a_waiver(self) -> None:
        import app.views.shared as shared

        st_mock = _fake_st()
        signature = preflight.signature(_settings())
        st_mock.session_state[shared._endpoint_block_key("draft")] = preflight.Verdict(
            preflight.BLOCKED, headline="The endpoint does not offer `model-fast`.", fix="Fix the id."
        )
        st_mock.session_state[shared._endpoint_block_key("draft") + "_sig"] = signature
        with patch.object(shared, "st", st_mock), patch.object(
            shared, "session_settings", return_value=_settings()
        ):
            shared.render_endpoint_preflight_notice("draft")
        message = str(st_mock.error.call_args[0][0])
        self.assertIn("Run not started", message)
        self.assertIn("model-fast", message)
        self.assertIn("Fix the id.", message)
        self.assertEqual(
            st_mock.checkbox.call_args[1]["key"], shared.endpoint_waiver_key("draft", signature)
        )

    def test_a_block_from_a_different_configuration_is_dropped(self) -> None:
        import app.views.shared as shared

        st_mock = _fake_st()
        st_mock.session_state[shared._endpoint_block_key("draft")] = preflight.Verdict(
            preflight.BLOCKED, headline="stale"
        )
        st_mock.session_state[shared._endpoint_block_key("draft") + "_sig"] = "other|config"
        with patch.object(shared, "st", st_mock), patch.object(
            shared, "session_settings", return_value=_settings()
        ):
            shared.render_endpoint_preflight_notice("draft")
        st_mock.error.assert_not_called()
        self.assertNotIn(shared._endpoint_block_key("draft"), st_mock.session_state)

    def test_nothing_is_rendered_without_a_block(self) -> None:
        import app.views.shared as shared

        st_mock = _fake_st()
        with patch.object(shared, "st", st_mock), patch.object(
            shared, "session_settings", return_value=_settings()
        ):
            shared.render_endpoint_preflight_notice("draft")
        st_mock.error.assert_not_called()

    def test_junk_in_session_state_is_ignored_rather_than_trusted(self) -> None:
        import app.views.shared as shared

        st_mock = _fake_st()
        st_mock.session_state[shared._endpoint_block_key("draft")] = {"not": "a verdict"}
        with patch.object(shared, "st", st_mock), patch.object(
            shared, "session_settings", return_value=_settings()
        ):
            shared.render_endpoint_preflight_notice("draft")
        st_mock.error.assert_not_called()


class TestRunFlowsAreStopped(unittest.TestCase):
    """The promise itself: a blocked preflight must not reach the pipeline."""

    def test_a_blocked_draft_never_calls_the_pipeline(self) -> None:
        import app.views.draft_view as draft_view

        st_mock = _fake_st()
        with patch.object(draft_view, "st", st_mock), patch.object(
            draft_view, "check_endpoint_gate", return_value=False
        ), patch.object(draft_view, "new_run_request_id", return_value="req_blocked"), patch.object(
            draft_view, "_run_draft_queued"
        ) as queued, patch.object(draft_view, "get_llm") as get_llm:
            draft_view._run_draft_flow(
                rid="req_blocked",
                records=[MagicMock()],
                condition="knee",
                claim_type="personal",
                relationship="",
                witness_name="",
                veteran_name="",
                known_since="",
                contact_frequency="",
                witnessed_event="",
                observations="obs",
            )
        get_llm.assert_not_called()
        queued.assert_not_called()

    def test_a_blocked_evaluation_never_calls_the_pipeline(self) -> None:
        import app.views.evaluate_view as evaluate_view

        st_mock = _fake_st()
        with patch.object(evaluate_view, "st", st_mock), patch.object(
            evaluate_view, "check_endpoint_gate", return_value=False
        ), patch.object(evaluate_view, "_run_evaluation_queued") as queued, patch.object(
            evaluate_view, "new_run_request_id", return_value="req_blocked"
        ):
            evaluate_view._run_evaluation_flow("statement", [MagicMock()])
        queued.assert_not_called()


class TestVercelCredentials(unittest.TestCase):
    """Vercel's credentials are not interchangeable, and a rejection says which is which.

    An AI Gateway key is not a provider key, a provider key is not a gateway key, and
    neither is the access token the *Sandbox* product takes — which is the third
    thing carrying Vercel's name and must not be suggested as an LLM credential.
    """

    GATEWAY = preflight.VERCEL_GATEWAY_BASE_URL

    def test_a_gateway_key_against_another_endpoint_names_the_gateway(self) -> None:
        verdict = _check(
            _probe(None, 401, "HTTP 401: unauthorized"),
            _settings(api_key="vck_example", base_url="https://api.perplexity.ai/router/v1"),
        )
        self.assertTrue(verdict.blocks)
        self.assertIn("AI Gateway", verdict.fix)
        self.assertIn(self.GATEWAY, verdict.fix)
        self.assertIn("OPENAI_BASE_URL", verdict.fix)

    def test_the_gateway_url_with_a_provider_key_names_the_key_to_create(self) -> None:
        verdict = _check(
            _probe(None, 401, "HTTP 401: unauthorized"),
            _settings(api_key="pplx-example", base_url=self.GATEWAY),
        )
        self.assertTrue(verdict.blocks)
        self.assertIn("AI Gateway API key", verdict.fix)
        self.assertIn("Sandbox", verdict.fix)

    def test_the_shape_never_decides_a_verdict_on_its_own(self) -> None:
        """A proxy can front the gateway with the same key, so an answered probe wins."""
        verdict = _check(
            _probe({"model-main", "model-fast"}, 200),
            _settings(api_key="vck_example", base_url="https://llm.internal.test/v1"),
        )
        self.assertEqual(verdict.kind, preflight.OK)

    def test_a_missing_gateway_model_id_gets_the_catalog_note(self) -> None:
        verdict = _check(
            _probe({"moonshotai/kimi-k3", "alibaba/qwen3.7-flash"}, 200),
            _settings(
                base_url=self.GATEWAY,
                api_key="vck_example",
                model_main="perplexity/kimi-k3",
                model_fast="alibaba/qwen3.7-flash",
            ),
        )
        self.assertTrue(verdict.blocks)
        self.assertEqual(verdict.missing, ("perplexity/kimi-k3",))
        self.assertIn("its own catalog", verdict.fix)
        self.assertIn("moonshotai/kimi-k3", verdict.fix)

    def test_an_unrelated_rejection_keeps_the_generic_fix(self) -> None:
        verdict = _check(_probe(None, 401, "HTTP 401: unauthorized"))
        self.assertNotIn("AI Gateway", verdict.fix)


if __name__ == "__main__":
    unittest.main()
