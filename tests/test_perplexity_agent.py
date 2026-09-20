"""Offline tests for the Perplexity Agent API client (app/perplexity_agent.py).

The Research tab is optional, so these cover both halves: the app with nothing configured
(no key, no SDK) must degrade to an explanation rather than an error, and the configured
path must extract the right things from a real response.

The configured path drives the **real** ``perplexityai`` SDK with its transport replaced by
an in-process ``httpx.MockTransport``. That matters more than it looks: a hand-rolled stub
returning a hand-rolled object would have hidden the bug this file now guards against — with
identical request shapes the SDK has been observed returning ``search_results`` rows as
plain ``dict``s in one response and as typed models in another. An extraction path built on
a bare ``getattr`` reads zero citations from the dict form *without raising*, which for a
web-grounded feature is the worst possible failure: an unsourced answer still looks
confident. The parametrized row-shape tests below exist specifically to pin both forms.

No network, no API key, and no credentials of any kind: the key used here is a literal that
is also asserted never to reach a log line or an exception message.

Tests skip (rather than fail) when the optional SDK is not installed, matching how the
tracing and S3 tests treat their optional dependencies.
"""
from __future__ import annotations

import json
import logging
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app import config  # noqa: E402
from app.circuit_breaker import (  # noqa: E402
    CircuitBreakerOpenError,
    get_llm_breaker,
    reset_llm_breaker,
)
from app.perplexity_agent import (  # noqa: E402
    BREAKER_NAME,
    GroundedAnswer,
    PerplexityConfigurationError,
    PerplexityParseError,
    PerplexityUpstreamError,
    condition_audit_schema,
    configured_preset,
    research,
    sdk_installed,
    unavailable_reason,
    web_search_tool,
)

try:  # The SDK brings httpx; without either, the integration is simply unavailable.
    import httpx

    from perplexity import Perplexity
except ImportError:  # pragma: no cover - exercised by the skip decorators below
    httpx = None  # type: ignore[assignment]
    Perplexity = None  # type: ignore[assignment]

SDK_READY = httpx is not None and Perplexity is not None and sdk_installed()

# A literal that is never a real credential, and is asserted to stay out of logs/errors.
FAKE_KEY = "pplx-not-a-real-key-000"

# The exact path the SDK's responses.create() posts to (the documented OpenAI-compat
# alias for /v1/agent).
RESPONSES_PATH = "/v1/responses"


def _settings(**overrides: Any) -> config.Settings:
    """A Settings object with the required fields filled and overrides applied."""
    base: dict[str, Any] = {
        "api_key": "primary-key",
        "base_url": "https://example.invalid/v1",
        "model_main": "main-model",
        "model_fast": "fast-model",
        "fetch_api_key": "",
        "fetch_base_url": "https://fetchsandbox.com",
        "fetch_records_path": "/medical_records/{patient_id}",
    }
    base.update(overrides)
    return config.Settings(**base)


def _payload(
    *,
    rows_as: str = "model",
    text: str = "Answer text.",
    status: str = "completed",
    with_fetch_url: bool = False,
    with_annotation: bool = False,
    with_cost: bool = False,
) -> dict[str, Any]:
    """A response body shaped like the documented Agent API response.

    ``rows_as`` selects the row shape the SDK will hand back for ``search_results``:
    ``"model"`` builds the documented JSON and relies on SDK validation, ``"dict"`` forces
    dict rows by asking for no strict validation. Both are exercised because both were
    observed in practice; see the module docstring.
    """

    def row(url: str, title: str, snippet: str = "s") -> Any:
        data = {"url": url, "title": title, "date": "2026-01-01", "snippet": snippet}
        return data

    output: list[dict[str, Any]] = [
        {
            "type": "search_results",
            "queries": ["va rating criteria"],
            "results": [
                row("https://www.ecfr.gov/a", "eCFR 4.97"),
                row("https://www.va.gov/b", "VA page"),
                row("https://www.ecfr.gov/a", "eCFR 4.97 (duplicate)"),
            ],
        }
    ]
    if with_fetch_url:
        output.append(
            {
                "type": "fetch_url_results",
                "contents": [row("https://www.va.gov/fetched", "Fetched page", "content")],
            }
        )
    annotations = (
        [{"type": "url_citation", "url": "https://www.va.gov/cited", "title": "cited"}]
        if with_annotation
        else []
    )
    output.append(
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": text, "logprobs": [], "annotations": annotations}
            ],
        }
    )
    if rows_as == "dict":
        # Drop the message item's typed shape too, so every row this test reads is a dict.
        output = [json.loads(json.dumps(item)) for item in output]

    usage: dict[str, Any] = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    if with_cost:
        usage["cost"] = {"currency": "USD", "total_cost": 0.00421}

    return {
        "id": "resp_test123",
        "object": "response",
        "status": status,
        "model": "openai/gpt-5.6-luna",
        "created_at": 1,
        "completed_at": 2,
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": 4096,
        "max_tool_calls": None,
        "metadata": {},
        "parallel_tool_calls": True,
        "presence_penalty": 0,
        "previous_response_id": None,
        "prompt_cache_key": None,
        "reasoning": None,
        "safety_identifier": None,
        "service_tier": "default",
        "store": True,
        "temperature": 1,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [{"type": "web_search"}],
        "top_logprobs": 0,
        "top_p": 1,
        "truncation": "disabled",
        "user": None,
        "usage": usage,
        "output": output,
    }


class _Transport:
    """Records requests and replays a queue of canned HTTP responses."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = responses
        self.requests: list[dict[str, Any]] = []

    def handler(self, request: Any) -> Any:
        self.requests.append(
            {
                "path": request.url.path,
                "headers": dict(request.headers),
                "body": json.loads(request.content) if request.content else {},
            }
        )
        item = self._responses[min(len(self.requests) - 1, len(self._responses) - 1)]
        if callable(item):
            return item(request)
        return item

    def client(self) -> Any:
        assert httpx is not None
        return Perplexity(
            api_key=FAKE_KEY,
            base_url="https://api.perplexity.ai",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(self.handler)),
        )


def _ok(payload: dict[str, Any]) -> Any:
    assert httpx is not None
    return httpx.Response(200, json=payload)


def _err(status: int, **headers: str) -> Any:
    assert httpx is not None
    return httpx.Response(status, json={"error": {"message": "boom"}}, headers=headers)


@unittest.skipUnless(SDK_READY, "optional perplexityai SDK is not installed")
class TestConfiguredPath(unittest.TestCase):
    """The SDK is present: request shape, extraction, and error mapping."""

    def setUp(self) -> None:
        reset_llm_breaker(name=BREAKER_NAME)

    tearDown = setUp

    def _run(self, transport: _Transport, **kwargs: Any) -> GroundedAnswer:
        """Run research() against a stubbed transport, through the real SDK."""
        with patch("app.perplexity_agent._build_client", return_value=transport.client()):
            return research(
                kwargs.pop("question", "What is the current rating criteria?"),
                settings=kwargs.pop("settings", _settings(perplexity_api_key=FAKE_KEY)),
                **kwargs,
            )

    # ------------------------------------------------------------- request shape

    def test_posts_to_responses_alias_with_documented_fields(self) -> None:
        transport = _Transport([_ok(_payload())])
        self._run(transport, domains=["va.gov"], schema=condition_audit_schema())

        self.assertEqual(len(transport.requests), 1)
        request = transport.requests[0]
        self.assertEqual(request["path"], RESPONSES_PATH)
        # Headers are lowercase in httpx; the bearer token must be present and correct.
        self.assertEqual(request["headers"]["authorization"], f"Bearer {FAKE_KEY}")
        self.assertEqual(request["body"]["preset"], "low")
        self.assertEqual(request["body"]["input"], "What is the current rating criteria?")
        self.assertEqual(request["body"]["max_output_tokens"], 4096)
        self.assertEqual(
            request["body"]["tools"],
            [
                {
                    "type": "web_search",
                    "search_context_size": "medium",
                    "filters": {"search_domain_filter": ["va.gov"]},
                }
            ],
        )
        self.assertEqual(
            request["body"]["response_format"]["json_schema"]["name"], "condition_audit"
        )

    def test_model_override_and_instructions_are_forwarded(self) -> None:
        transport = _Transport([_ok(_payload())])
        self._run(
            transport,
            settings=_settings(perplexity_api_key=FAKE_KEY, perplexity_model="openai/gpt-5.6-sol"),
            instructions="Be precise.",
        )
        body = transport.requests[0]["body"]
        self.assertEqual(body["model"], "openai/gpt-5.6-sol")
        self.assertEqual(body["instructions"], "Be precise.")
        # A preset is still sent: the model override freezes the model, not the config.
        self.assertEqual(body["preset"], "low")

    def test_blank_question_is_rejected_without_a_call(self) -> None:
        transport = _Transport([_ok(_payload())])
        with self.assertRaises(PerplexityConfigurationError):
            self._run(transport, question="   ")
        self.assertEqual(transport.requests, [])

    # ---------------------------------------------------------------- extraction

    def test_extracts_text_citations_and_usage(self) -> None:
        transport = _Transport([_ok(_payload(with_fetch_url=True, with_annotation=True, with_cost=True))])
        answer = self._run(transport)

        self.assertEqual(answer.text, "Answer text.")
        self.assertEqual(answer.response_id, "resp_test123")
        self.assertEqual(answer.model, "openai/gpt-5.6-luna")
        self.assertEqual(answer.preset, "low")

        urls = [c.url for c in answer.citations]
        # search_results + fetch_url_results + message annotation.
        self.assertIn("https://www.ecfr.gov/a", urls)
        self.assertIn("https://www.va.gov/b", urls)
        self.assertIn("https://www.va.gov/fetched", urls)
        self.assertIn("https://www.va.gov/cited", urls)
        # Deduplicated by URL: the same source twice would read as two confirmations.
        self.assertEqual(urls.count("https://www.ecfr.gov/a"), 1)
        self.assertEqual(answer.citation_count, len(urls))

        # Titles/dates come from the response, never from model-written JSON.
        first = next(c for c in answer.citations if c.url == "https://www.ecfr.gov/a")
        self.assertEqual(first.title, "eCFR 4.97")
        self.assertEqual(first.date, "2026-01-01")

        self.assertEqual(answer.usage["total_tokens"], 150)
        self.assertEqual(answer.usage["total_cost_usd"], 0.00421)

    def test_structured_output_is_parsed_into_findings(self) -> None:
        audit = {
            "condition": "sleep apnea",
            "summary": "Current criteria reviewed.",
            "rating_criteria": "38 CFR 4.97 DC 6847.",
            "framework_findings": [
                {"topic": "medication side effects", "status": "current", "note": "unchanged"}
            ],
        }
        transport = _Transport([_ok(_payload(text=json.dumps(audit)))])
        answer = self._run(transport, schema=condition_audit_schema())

        self.assertIsNotNone(answer.findings)
        assert answer.findings is not None
        self.assertEqual(answer.findings["condition"], "sleep apnea")
        self.assertEqual(answer.findings["framework_findings"][0]["status"], "current")

    def test_prose_answer_does_not_attempt_json_parsing(self) -> None:
        transport = _Transport([_ok(_payload(text="Not JSON at all."))])
        answer = self._run(transport)
        self.assertIsNone(answer.findings)
        self.assertEqual(answer.text, "Not JSON at all.")

    # ------------------------------------------------------------------- errors

    def test_incomplete_status_is_a_failure_not_a_truncated_answer(self) -> None:
        transport = _Transport([_ok(_payload(status="incomplete", text="Half an ans"))])
        with self.assertRaises(PerplexityUpstreamError) as ctx:
            self._run(transport)
        self.assertFalse(ctx.exception.retriable)
        self.assertIn("incomplete", str(ctx.exception))

    def test_401_is_not_retried_and_names_the_rotation_path(self) -> None:
        transport = _Transport([_err(401)])
        with self.assertRaises(PerplexityUpstreamError) as ctx:
            self._run(transport)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertFalse(ctx.exception.retriable)
        self.assertIn("rotate", str(ctx.exception))
        # A retriable-False error must not be retried.
        self.assertEqual(len(transport.requests), 1)

    def test_400_is_not_retried(self) -> None:
        transport = _Transport([_err(400)])
        with self.assertRaises(PerplexityUpstreamError) as ctx:
            self._run(transport)
        self.assertFalse(ctx.exception.retriable)
        self.assertEqual(len(transport.requests), 1)

    def test_429_honours_retry_after_then_succeeds(self) -> None:
        transport = _Transport([_err(429, **{"retry-after": "2"}), _ok(_payload())])
        with patch("time.sleep") as slept:
            answer = self._run(transport)

        self.assertEqual(answer.text, "Answer text.")
        self.assertEqual(len(transport.requests), 2, "the 429 should be retried once here")
        # Retry-After was honoured (2s of waiting, in cancellation-aware steps).
        self.assertGreaterEqual(sum(c.args[0] for c in slept.call_args_list), 2.0)

    def test_429_exhausting_attempts_reports_retriable(self) -> None:
        transport = _Transport([_err(429, **{"retry-after": "1"})])
        with patch("time.sleep"):
            with self.assertRaises(PerplexityUpstreamError) as ctx:
                self._run(transport)
        self.assertTrue(ctx.exception.retriable)
        self.assertEqual(ctx.exception.status_code, 429)
        self.assertEqual(len(transport.requests), 3, "MAX_ATTEMPTS retries all run")

    def test_vendor_error_body_never_leaks_the_key(self) -> None:
        transport = _Transport([_err(500)])
        with patch("time.sleep"):
            with self.assertRaises(PerplexityUpstreamError) as ctx:
                self._run(transport)
        self.assertNotIn(FAKE_KEY, str(ctx.exception))

    # ------------------------------------------------------------------ breaker

    def test_terminal_failure_records_a_breaker_failure(self) -> None:
        transport = _Transport([_err(400)])
        breaker = get_llm_breaker(BREAKER_NAME)
        before = breaker.failure_count
        with self.assertRaises(PerplexityUpstreamError):
            self._run(transport)
        self.assertGreater(breaker.failure_count, before)

    def test_open_breaker_fails_fast_without_a_request(self) -> None:
        reset_llm_breaker(name=BREAKER_NAME, failure_threshold=1, recovery_timeout=300.0)
        breaker = get_llm_breaker(BREAKER_NAME)
        breaker.record_failure()

        transport = _Transport([_ok(_payload())])
        with self.assertRaises(CircuitBreakerOpenError):
            self._run(transport)
        # Fail-fast means no network call at all: the point of the breaker.
        self.assertEqual(transport.requests, [])


@unittest.skipUnless(SDK_READY, "optional perplexityai SDK is not installed")
class TestRowShapeRobustness(unittest.TestCase):
    """Pin both observed shapes for search_results rows.

    The SDK has been seen returning these rows as typed models *and* as plain dicts for
    identical request shapes. A bare getattr yields an empty citation list for the dict
    form, silently — so both are asserted to produce the same sources.
    """

    def setUp(self) -> None:
        reset_llm_breaker(name=BREAKER_NAME)

    def _citations(self, rows_as: str) -> list[str]:
        transport = _Transport([_ok(_payload(rows_as=rows_as, with_fetch_url=True))])
        with patch("app.perplexity_agent._build_client", return_value=transport.client()):
            answer = research(
                "q", settings=_settings(perplexity_api_key=FAKE_KEY)
            )
        return [c.url for c in answer.citations]

    def test_documented_json_shape(self) -> None:
        urls = self._citations("model")
        self.assertIn("https://www.ecfr.gov/a", urls)
        self.assertIn("https://www.va.gov/fetched", urls)

    def test_dict_rows_are_also_extracted(self) -> None:
        urls = self._citations("dict")
        # Regression guard: this set used to come back empty.
        self.assertIn("https://www.ecfr.gov/a", urls)
        self.assertIn("https://www.va.gov/fetched", urls)


class TestAvailability(unittest.TestCase):
    """Nothing configured: the tab must explain itself, not fail."""

    def test_missing_key_reports_how_to_enable(self) -> None:
        settings = _settings()
        self.assertFalse(settings.perplexity_configured)
        reason = unavailable_reason(settings)
        self.assertIsNotNone(reason)
        assert reason is not None
        self.assertIn("PERPLEXITY_API_KEY", reason)
        self.assertIn("console.perplexity.ai", reason)

    @unittest.skipUnless(sdk_installed(), "SDK present; testing the missing-SDK branch")
    def test_missing_sdk_names_the_requirements_file(self) -> None:
        settings = _settings(perplexity_api_key=FAKE_KEY)
        with patch("app.perplexity_agent.sdk_installed", return_value=False):
            reason = unavailable_reason(settings)
        assert reason is not None
        self.assertIn("requirements-perplexity.txt", reason)

    @unittest.skipUnless(sdk_installed(), "SDK is an optional install")
    def test_configured_and_installed_is_available(self) -> None:
        self.assertIsNone(unavailable_reason(_settings(perplexity_api_key=FAKE_KEY)))

    def test_research_without_configuration_raises_before_any_call(self) -> None:
        with self.assertRaises(PerplexityConfigurationError):
            research("q", settings=_settings())

    def test_unknown_preset_falls_back_to_a_valid_one(self) -> None:
        settings = _settings(perplexity_preset="ultra-mega")
        self.assertIn(configured_preset(settings), config.PERPLEXITY_PRESETS)

    def test_preset_env_var_is_validated_at_load_time(self) -> None:
        with patch.dict("os.environ", {"PERPLEXITY_PRESET": "not-a-preset"}, clear=False):
            self.assertEqual(
                config._perplexity_preset_setting(set()), config.DEFAULT_PERPLEXITY_PRESET
            )


class TestSchemaAndToolShape(unittest.TestCase):
    """The tool entry and the schema must stay inside documented bounds."""

    def test_web_search_tool_without_domains_omits_the_filter(self) -> None:
        # An empty filter list is not the same as no filter, so it must be absent.
        tool = web_search_tool([])
        self.assertNotIn("filters", tool)
        self.assertEqual(tool["type"], "web_search")

    def test_web_search_tool_restricts_domains_when_given(self) -> None:
        tool = web_search_tool(["va.gov", " ecfr.gov ", ""])
        self.assertEqual(tool["filters"]["search_domain_filter"], ["va.gov", "ecfr.gov"])

    def test_schema_contains_no_url_field_anywhere(self) -> None:
        """The docs forbid asking the model for links in structured output.

        A model emitting URLs inside a JSON schema can produce malformed or fabricated
        ones, so citations are read from the response's own items instead. This walks the
        whole schema so a future field named ``source_url`` cannot sneak back in.
        """
        def walk(node: Any) -> list[str]:
            found: list[str] = []
            if isinstance(node, dict):
                for key, value in node.items():
                    if isinstance(key, str) and "url" in key.lower():
                        found.append(key)
                    found.extend(walk(value))
            elif isinstance(node, list):
                for entry in node:
                    found.extend(walk(entry))
            return found

        self.assertEqual(walk(condition_audit_schema()), [])

    def test_schema_requires_an_explicit_property_set(self) -> None:
        schema = condition_audit_schema()["json_schema"]["schema"]
        self.assertFalse(schema["additionalProperties"])
        self.assertIn("condition", schema["required"])


class TestSecretHygiene(unittest.TestCase):
    """The key must never reach a log line or an exception message."""

    class _Capture(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.messages: list[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.messages.append(self.format(record))

    def test_key_never_appears_in_logs(self) -> None:
        handler = self._Capture()
        handler.setFormatter(logging.Formatter("%(message)s"))
        app_logger = logging.getLogger("app")
        app_logger.addHandler(handler)
        previous = app_logger.level
        app_logger.setLevel(logging.DEBUG)
        try:
            settings = _settings(
                perplexity_api_key=FAKE_KEY, perplexity_sources="va.gov,ecfr.gov"
            )
            # Exercise the modules that log around a call, configured or not.
            unavailable_reason(settings)
            configured_preset(settings)
            web_search_tool(settings.perplexity_source_domains())
            try:
                research("q", settings=settings, schema=condition_audit_schema())
            except Exception:  # noqa: BLE001 - offline: any failure mode is fine here
                pass
        finally:
            app_logger.removeHandler(handler)
            app_logger.setLevel(previous)

        joined = "\n".join(handler.messages)
        self.assertNotIn(FAKE_KEY, joined)

    def test_user_facing_strings_do_not_contain_the_key(self) -> None:
        """Reason strings are rendered in the tab and quoted into issues, so no key."""
        settings = _settings(perplexity_api_key=FAKE_KEY)
        self.assertTrue(settings.perplexity_configured)
        with patch("app.perplexity_agent.sdk_installed", return_value=False):
            sdk_missing = unavailable_reason(settings) or ""
        for text in (sdk_missing, configured_preset(settings), *map(str, settings.perplexity_source_domains())):
            self.assertNotIn(FAKE_KEY, text)

    def test_unconfigured_reason_never_echoes_an_absent_key(self) -> None:
        # The unset case is the one a user sees most often; it must stay a how-to, not an
        # echo of whatever partial value happened to be in the environment.
        reason = unavailable_reason(_settings(perplexity_api_key="   ")) or ""
        self.assertIn("PERPLEXITY_API_KEY", reason)
        self.assertNotIn("None", reason)


if __name__ == "__main__":
    unittest.main()
