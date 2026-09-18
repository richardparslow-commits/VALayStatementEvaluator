"""Tests for optional LLM endpoint failover (``OPENAI_BASE_URL_FALLBACK``).

Three properties carry this feature, and each has its own class below:

* **The failover trigger must be measurable.** ``CircuitBreaker._opened_at`` is
  reset by every *failed probe*, so with the default 60 s recovery timeout and
  continuous traffic it never ages past 60 s. A rule written as "the primary has
  been OPEN for more than N seconds" would therefore never fire under exactly the
  load it exists for — while looking implemented. ``unhealthy_for_seconds`` is the
  separate clock that only a genuine recovery clears. This is the regression the
  ``TestUnhealthyClock`` class pins.
* **One breaker per endpoint.** A shared breaker would let a healthy fallback's
  successes close the primary's and hide an ongoing outage.
* **A failover probe must not fail a user's run.** When the primary's recovery
  window elapses, the next real call is tried there first and falls through to the
  fallback if it fails, so recovery is detected without a user ever seeing the
  probe failure.
"""

from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app import circuit_breaker, config, health, llm, metrics  # noqa: E402
from app.circuit_breaker import (  # noqa: E402
    LLM_BREAKER_NAME,
    LLM_FALLBACK_BREAKER_NAME,
    CircuitBreaker,
    CircuitBreakerOpenError,
    QueueFullError,
)
from app.job_payload import usage_from_json, usage_to_json  # noqa: E402
from app.llm import LLMClient  # noqa: E402
from app.usage import UsageTracker  # noqa: E402


def _settings(**overrides) -> config.Settings:
    """A Settings with a primary endpoint and, by default, no fallback."""
    base = dict(
        api_key="primary-key",
        base_url="https://primary.invalid/v1",
        model_main="primary-main",
        model_fast="primary-fast",
        fetch_api_key="",
        fetch_base_url="",
        fetch_records_path="",
    )
    base.update(overrides)
    return config.Settings(**base)


def _armed(**overrides) -> config.Settings:
    """A Settings with a genuine second provider (different URL, key and models)."""
    base = dict(
        fallback_base_url="https://fallback.invalid/v1",
        fallback_api_key="fallback-key",
        fallback_model_main="fallback-main",
        fallback_model_fast="fallback-fast",
    )
    base.update(overrides)
    return _settings(**base)


def _response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=text))]
    resp.usage = None
    return resp


class _FailoverCase(unittest.TestCase):
    """Resets breaker/limiter/metrics state and the failover knobs per test."""

    def setUp(self) -> None:
        circuit_breaker.reset_all_for_tests()
        metrics.reset_for_tests()
        self._threshold = patch.object(config, "LLM_FAILOVER_AFTER_SECONDS", 300)
        self._threshold.start()
        self.addCleanup(self._threshold.stop)
        self.addCleanup(circuit_breaker.reset_all_for_tests)
        self.addCleanup(metrics.reset_for_tests)

    @staticmethod
    def _client(settings: config.Settings, *, primary_ok: bool = True, fallback_ok: bool = True) -> LLMClient:
        client = LLMClient(settings)
        if primary_ok:
            client._client.chat.completions.create = MagicMock(  # type: ignore[attr-defined]
                return_value=_response("from primary")
            )
        else:
            client._client.chat.completions.create = MagicMock(  # type: ignore[attr-defined]
                side_effect=Exception("primary boom")
            )
        if client._fallback_client is not None:
            if fallback_ok:
                client._fallback_client.chat.completions.create = MagicMock(  # type: ignore[attr-defined]
                    return_value=_response("from fallback")
                )
            else:
                client._fallback_client.chat.completions.create = MagicMock(  # type: ignore[attr-defined]
                    side_effect=Exception("fallback boom")
                )
        return client

    @staticmethod
    def _primary_breaker() -> CircuitBreaker:
        return circuit_breaker.get_llm_breaker(LLM_BREAKER_NAME)

    def _make_primary_unhealthy(self) -> CircuitBreaker:
        """Trip the primary into OPEN."""
        breaker = self._primary_breaker()
        for _ in range(breaker.failure_threshold):
            breaker.record_failure()
        assert breaker.state == "OPEN"
        return breaker


# --------------------------------------------------------------------- config


class TestFallbackConfiguration(unittest.TestCase):
    def test_absent_by_default_so_nothing_changes(self) -> None:
        settings = _settings()
        self.assertFalse(settings.fallback_configured)
        self.assertFalse(llm._fallback_target(settings).configured)

    def test_a_base_url_alone_arms_failover_and_inherits_the_rest(self) -> None:
        """A second gateway under the same account must need one variable, not four."""
        settings = _settings(fallback_base_url="https://second.invalid/v1")
        target = llm._fallback_target(settings)
        self.assertTrue(target.configured)
        self.assertEqual(target.base_url, "https://second.invalid/v1")
        self.assertEqual(target.api_key, "primary-key")
        self.assertEqual(target.model_main, "primary-main")
        self.assertEqual(target.model_fast, "primary-fast")

    def test_explicit_fallback_values_win(self) -> None:
        target = llm._fallback_target(_armed())
        self.assertEqual(target.api_key, "fallback-key")
        self.assertEqual(target.model_main, "fallback-main")
        self.assertEqual(target.model_fast, "fallback-fast")

    def test_settings_without_the_attribute_have_no_fallback(self) -> None:
        """A settings object predating this feature must not raise."""

        class _Old:
            api_key = "k"
            base_url = "https://p.invalid/v1"
            model_main = "m"
            model_fast = "f"

        self.assertFalse(llm._fallback_target(_Old()).configured)  # type: ignore[arg-type]

    def test_a_non_string_url_is_not_a_configured_endpoint(self) -> None:
        """A mock's auto-created attribute is not a URL; only a str can be one."""
        stub = MagicMock()
        stub.fallback_base_url = MagicMock()
        self.assertFalse(llm._fallback_target(stub).configured)

    def test_load_settings_reads_the_documented_env_vars(self) -> None:
        env = {
            "OPENAI_BASE_URL_FALLBACK": "https://api.openai.com/v1",
            "OPENAI_API_KEY_FALLBACK": "sk-fallback",
            "LLM_MODEL_MAIN_FALLBACK": "gpt-4o",
            "LLM_MODEL_FAST_FALLBACK": "gpt-4o-mini",
        }
        with patch.dict("os.environ", env, clear=False):
            settings = config.load_settings()
        self.assertTrue(settings.fallback_configured)
        self.assertEqual(settings.fallback_base_url, "https://api.openai.com/v1")
        self.assertEqual(settings.fallback_api_key_or_primary(), "sk-fallback")
        self.assertEqual(settings.fallback_model_main_or_primary(), "gpt-4o")
        self.assertEqual(settings.fallback_model_fast_or_primary(), "gpt-4o-mini")

    def test_the_grace_period_defaults_to_five_minutes(self) -> None:
        self.assertEqual(config.DEFAULT_FAILOVER_AFTER_SECONDS, 300)
        self.assertEqual(llm.failover_after_seconds(), 300.0)

    def test_the_documented_env_var_name_is_the_one_config_reads(self) -> None:
        """A rename here would silently ignore an operator's setting."""
        source = (Path(config.__file__)).read_text(encoding="utf-8")
        self.assertIn("LLM_ENDPOINT_FALLBACK_TIMEOUT_SECONDS", source)
        self.assertIn(
            'LLM_FAILOVER_AFTER_SECONDS = _non_negative_int_env(',
            source,
            "0 must stay a valid setting (it means \"fail over immediately\")",
        )

    def test_env_zero_means_fail_over_immediately(self) -> None:
        """`0` must mean "no wait", not "invalid, use the default".

        Checked in a subprocess because config reads the environment once at import:
        the in-process tests above patch the attribute, which is exactly why this
        one is here — a positive-int parser would silently turn an operator's
        deliberate `0` back into the 300s default.
        """
        import os
        import subprocess

        project_root = str(Path(__file__).resolve().parent.parent)
        code = (
            f"import sys; sys.path.insert(0, {project_root!r}); "
            "from app import config; print(config.LLM_FAILOVER_AFTER_SECONDS)"
        )
        for raw, expected in (("0", "0"), ("45", "45"), ("-5", "300"), ("x", "300")):
            env = dict(os.environ)
            env["LLM_ENDPOINT_FALLBACK_TIMEOUT_SECONDS"] = raw
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
            )
            self.assertEqual(result.stdout.strip(), expected, f"{raw!r} was not honoured")

    def test_the_grace_period_is_not_an_http_timeout(self) -> None:
        """The spec's variable name is ambiguous; make sure it did not get wired as one."""
        with patch.object(config, "LLM_FAILOVER_AFTER_SECONDS", 45):
            self.assertEqual(llm.failover_after_seconds(), 45.0)
        self.assertTrue(hasattr(config, "LLM_CALL_TIMEOUT_SECONDS"))


class TestFallbackValidation(unittest.TestCase):
    def test_a_non_url_is_rejected_at_construction(self) -> None:
        with self.assertRaises(llm.LLMConfigurationError) as ctx:
            LLMClient(_settings(fallback_base_url="not-a-url"))
        self.assertIn("OPENAI_BASE_URL_FALLBACK", str(ctx.exception))

    def test_a_scheme_less_url_is_rejected(self) -> None:
        with self.assertRaises(llm.LLMConfigurationError):
            LLMClient(_settings(fallback_base_url="api.openai.com/v1"))

    def test_an_invalid_fallback_model_name_is_rejected(self) -> None:
        with self.assertRaises(llm.LLMConfigurationError) as ctx:
            LLMClient(_armed(fallback_model_main="bad model name!"))
        self.assertIn("Fallback main model", str(ctx.exception))

    def test_a_fallback_key_is_optional_when_the_primary_has_one(self) -> None:
        """Same-account second gateway: one variable, and the key is inherited."""
        client = LLMClient(_settings(fallback_base_url="https://second.invalid/v1"))
        self.assertTrue(client.fallback_available)

    def test_no_key_at_all_is_rejected_by_the_primary_check(self) -> None:
        """A fallback cannot rescue a deployment with no credentials at all."""
        with self.assertRaises(llm.LLMConfigurationError) as ctx:
            LLMClient(_settings(api_key="", fallback_base_url="https://f.invalid/v1"))
        self.assertIn("No API key configured", str(ctx.exception))

    def test_pointing_the_fallback_at_the_primary_warns_but_is_allowed(self) -> None:
        """Not fatal, but it cannot be failover — say so rather than pretend."""
        same = "https://primary.invalid/v1"
        with self.assertLogs("app.llm", level="WARNING") as logs:
            LLMClient(_settings(fallback_base_url=same))
        self.assertTrue(any("same URL" in line for line in logs.output))


# -------------------------------------------------------------------- routing


class TestEndpointRouting(_FailoverCase):
    def test_no_fallback_configured_uses_the_primary_alone(self) -> None:
        client = self._client(_settings())
        self.assertFalse(client.fallback_available)
        self.assertEqual(client._endpoint_candidates(), [llm.PRIMARY_ENDPOINT])

    def test_a_healthy_primary_is_used(self) -> None:
        client = self._client(_armed())
        self.assertEqual(client._endpoint_candidates(), [llm.PRIMARY_ENDPOINT])

    def test_an_unhealthy_primary_inside_the_grace_period_is_still_used(self) -> None:
        """A blip must not move the business onto another provider."""
        self._make_primary_unhealthy()
        client = self._client(_armed())
        self.assertLess(
            self._primary_breaker().unhealthy_for_seconds(), config.LLM_FAILOVER_AFTER_SECONDS
        )
        self.assertEqual(client._endpoint_candidates(), [llm.PRIMARY_ENDPOINT])

    def test_past_the_grace_period_calls_move_to_the_fallback(self) -> None:
        self._make_primary_unhealthy()
        with patch.object(config, "LLM_FAILOVER_AFTER_SECONDS", 0):
            client = self._client(_armed())
            # The probe window has not elapsed, so the primary is not tried.
            self._primary_breaker().recovery_timeout = 3600
            self.assertEqual(client._endpoint_candidates(), [llm.FALLBACK_ENDPOINT])

    def test_the_primary_is_probed_once_its_recovery_window_elapses(self) -> None:
        self._make_primary_unhealthy()
        with patch.object(config, "LLM_FAILOVER_AFTER_SECONDS", 0):
            client = self._client(_armed())
            self._primary_breaker().recovery_timeout = 0
            self.assertEqual(
                client._endpoint_candidates(),
                [llm.PRIMARY_ENDPOINT, llm.FALLBACK_ENDPOINT],
            )

    def test_an_armed_but_unreachable_fallback_still_routes_somewhere(self) -> None:
        """A fallback client is built in the constructor, so routing can rely on it."""
        client = self._client(_armed())
        self.assertTrue(client.fallback_available)


# ----------------------------------------------------------------- chat paths


class TestChatFailover(_FailoverCase):
    def test_a_healthy_run_uses_the_primary_and_is_stamped_primary(self) -> None:
        client = self._client(_armed())
        self.assertEqual(client.chat("s", "u", phase="draft"), "from primary")
        self.assertEqual(client.usage.endpoints_used(), [llm.PRIMARY_ENDPOINT])
        self.assertFalse(client.usage.used_fallback)
        self.assertFalse(llm.failover_status()["active"])

    def test_a_failed_over_run_is_served_by_the_fallback_and_stamped(self) -> None:
        self._make_primary_unhealthy()
        with patch.object(config, "LLM_FAILOVER_AFTER_SECONDS", 0):
            client = self._client(_armed(), primary_ok=False)
            self._primary_breaker().recovery_timeout = 3600
            self.assertEqual(client.chat("s", "u", phase="draft"), "from fallback")
        self.assertEqual(client.usage.endpoints_used(), [llm.FALLBACK_ENDPOINT])
        self.assertTrue(client.usage.used_fallback)
        self.assertEqual(
            client.usage.summary()["fallback_calls"], 1, "the run record must show it"
        )

    def test_the_fallback_call_is_counted_under_its_own_endpoint_label(self) -> None:
        self._make_primary_unhealthy()
        with patch.object(config, "LLM_FAILOVER_AFTER_SECONDS", 0):
            client = self._client(_armed(), primary_ok=False)
            self._primary_breaker().recovery_timeout = 3600
            client.chat("s", "u", phase="draft")
        calls = {tuple(sorted(labels.items())): value for labels, value in metrics.llm_endpoint_calls_total.samples()}
        self.assertEqual(calls[(("endpoint", "fallback"), ("outcome", "ok"))], 1.0)
        self.assertEqual(metrics.llm_endpoint_duration_ms.observed, 1)

    def test_the_primary_probe_falls_through_to_the_fallback_without_an_error(self) -> None:
        """The user's run must not fail because we tested the primary."""
        self._make_primary_unhealthy()
        with patch.object(config, "LLM_FAILOVER_AFTER_SECONDS", 0):
            client = self._client(_armed(), primary_ok=False)
            self._primary_breaker().recovery_timeout = 0
            self.assertEqual(client._endpoint_candidates(), [llm.PRIMARY_ENDPOINT, llm.FALLBACK_ENDPOINT])
            # No exception reaches the caller.
            self.assertEqual(client.chat("s", "u", phase="draft"), "from fallback")
        self.assertEqual(
            {tuple(sorted(labels.items())): v for labels, v in metrics.llm_failover_total.samples()},
            {(("reason", "primary_call_failed"),): 1.0},
        )

    def test_a_successful_probe_returns_the_run_to_the_primary(self) -> None:
        """Fail-back is what keeps a single outage from pinning the whole month."""
        self._make_primary_unhealthy()
        with patch.object(config, "LLM_FAILOVER_AFTER_SECONDS", 0):
            client = self._client(_armed(), primary_ok=True)
            self._primary_breaker().recovery_timeout = 0
            self.assertEqual(client.chat("s", "u", phase="draft"), "from primary")
        self.assertEqual(self._primary_breaker().state, "CLOSED")
        self.assertIsNone(self._primary_breaker().unhealthy_for_seconds())
        # Served by the primary, and the fallback was never touched for this call.
        self.assertEqual(client.usage.endpoints_used(), [llm.PRIMARY_ENDPOINT])
        client._fallback_client.chat.completions.create.assert_not_called()  # type: ignore[union-attr]

    def test_both_endpoints_failing_surfaces_one_honest_error(self) -> None:
        self._make_primary_unhealthy()
        with patch.object(config, "LLM_FAILOVER_AFTER_SECONDS", 0):
            client = self._client(_armed(), primary_ok=False, fallback_ok=False)
            self._primary_breaker().recovery_timeout = 0
            with self.assertRaises(llm.LLMError):
                client.chat("s", "u", phase="draft")

    def test_a_fast_fail_is_not_retried_on_the_other_endpoint(self) -> None:
        """OPEN means \"do not call this provider\"; it is not a call failure."""
        self._make_primary_unhealthy()
        client = self._client(_armed())
        with self.assertRaises(CircuitBreakerOpenError):
            client.chat("s", "u", phase="draft")
        client._fallback_client.chat.completions.create.assert_not_called()  # type: ignore[union-attr]

    def test_the_fast_fail_message_says_a_backup_endpoint_is_coming(self) -> None:
        """Otherwise the message reads as \"down\" with no hint fault tolerance exists."""
        self._make_primary_unhealthy()
        client = self._client(_armed())
        with self.assertRaises(CircuitBreakerOpenError) as ctx:
            client.chat("s", "u", phase="draft")
        self.assertIn("backup endpoint", str(ctx.exception))

    def test_a_single_endpoint_fast_fail_message_is_unchanged(self) -> None:
        """No fallback configured means no promise about one."""
        self._make_primary_unhealthy()
        client = self._client(_settings())
        with self.assertRaises(CircuitBreakerOpenError) as ctx:
            client.chat("s", "u", phase="draft")
        self.assertNotIn("backup endpoint", str(ctx.exception))

    def test_queue_full_is_not_retried_on_the_other_endpoint(self) -> None:
        """Our own concurrency cap is not an endpoint fault."""
        self._make_primary_unhealthy()
        with patch.object(config, "LLM_FAILOVER_AFTER_SECONDS", 0):
            client = self._client(_armed())
            self._primary_breaker().recovery_timeout = 0
            with patch(
                "app.llm.get_llm_limiter",
                side_effect=lambda: MagicMock(acquire=MagicMock(side_effect=QueueFullError("full")), release=MagicMock()),
            ):
                with self.assertRaises(QueueFullError):
                    client.chat("s", "u", phase="draft")

    def test_chat_json_inherits_failover(self) -> None:
        self._make_primary_unhealthy()
        with patch.object(config, "LLM_FAILOVER_AFTER_SECONDS", 0):
            client = self._client(_armed(), primary_ok=False)
            self._primary_breaker().recovery_timeout = 3600
            client._fallback_client.chat.completions.create = MagicMock(  # type: ignore[union-attr]
                return_value=_response('{"ok": true}')
            )
            self.assertEqual(client.chat_json("s", "u", phase="topic"), {"ok": True})


class TestModelTranslation(_FailoverCase):
    def test_the_primary_models_pass_through(self) -> None:
        client = self._client(_armed())
        self.assertEqual(client._resolve_model(llm.PRIMARY_ENDPOINT, "primary-main"), "primary-main")
        self.assertEqual(client._resolve_model(llm.PRIMARY_ENDPOINT, None), "primary-main")

    def test_the_two_roles_map_onto_the_fallback_names(self) -> None:
        """A different provider has different model names for the same two roles."""
        client = self._client(_armed())
        self.assertEqual(client._resolve_model(llm.FALLBACK_ENDPOINT, "primary-main"), "fallback-main")
        self.assertEqual(client._resolve_model(llm.FALLBACK_ENDPOINT, "primary-fast"), "fallback-fast")
        self.assertEqual(client._resolve_model(llm.FALLBACK_ENDPOINT, None), "fallback-main")

    def test_an_unrelated_model_name_is_not_guessed_at(self) -> None:
        client = self._client(_armed())
        self.assertEqual(client._resolve_model(llm.FALLBACK_ENDPOINT, "some-other-model"), "some-other-model")

    def test_the_model_reaching_the_fallback_provider_is_the_fallback_model(self) -> None:
        self._make_primary_unhealthy()
        with patch.object(config, "LLM_FAILOVER_AFTER_SECONDS", 0):
            client = self._client(_armed(), primary_ok=False)
            self._primary_breaker().recovery_timeout = 3600
            client.chat("s", "u", phase="draft", model="primary-main")
        kwargs = client._fallback_client.chat.completions.create.call_args.kwargs  # type: ignore[union-attr]
        self.assertEqual(kwargs["model"], "fallback-main")


# ------------------------------------------------------- breaker independence


class TestBreakerIsolation(_FailoverCase):
    def test_the_fallback_has_its_own_breaker(self) -> None:
        primary = circuit_breaker.get_llm_breaker(LLM_BREAKER_NAME)
        fallback = circuit_breaker.get_llm_breaker(LLM_FALLBACK_BREAKER_NAME)
        self.assertIsNot(primary, fallback)
        self.assertEqual(primary.name, "llm")
        self.assertEqual(fallback.name, "llm-fallback")

    def test_a_fallback_success_does_not_close_the_primary_breaker(self) -> None:
        """Sharing one breaker would hide an ongoing primary outage."""
        self._make_primary_unhealthy()
        fallback = circuit_breaker.get_llm_breaker(LLM_FALLBACK_BREAKER_NAME)
        for _ in range(5):
            fallback.record_success()
        self.assertEqual(self._primary_breaker().state, "OPEN")
        self.assertIsNotNone(self._primary_breaker().unhealthy_for_seconds())

    def test_fallback_failures_do_not_count_toward_the_primary_threshold(self) -> None:
        fallback = circuit_breaker.get_llm_breaker(LLM_FALLBACK_BREAKER_NAME)
        for _ in range(fallback.failure_threshold):
            fallback.record_failure()
        self.assertEqual(fallback.state, "OPEN")
        self.assertEqual(self._primary_breaker().state, "CLOSED")
        self.assertIsNone(self._primary_breaker().unhealthy_for_seconds())

    def test_iter_does_not_create_breakers_that_were_never_used(self) -> None:
        """Reporting a state for an endpoint nothing has ever called is a lie."""
        names = {b.name for b in circuit_breaker.iter_llm_breakers()}
        self.assertNotIn("llm-tertiary", names)
        circuit_breaker.get_llm_breaker("llm-tertiary")
        self.assertIn("llm-tertiary", {b.name for b in circuit_breaker.iter_llm_breakers()})

    def test_reset_all_for_tests_also_resets_the_fallback(self) -> None:
        fallback = circuit_breaker.get_llm_breaker(LLM_FALLBACK_BREAKER_NAME)
        for _ in range(fallback.failure_threshold):
            fallback.record_failure()
        circuit_breaker.reset_all_for_tests()
        self.assertEqual(
            circuit_breaker.get_llm_breaker(LLM_FALLBACK_BREAKER_NAME).state, "CLOSED"
        )


class TestUnhealthyClock(_FailoverCase):
    """The clock the failover rule depends on.

    ``_opened_at`` cannot serve this purpose: it is reset by every failed probe,
    so with continuous traffic it never ages past one recovery timeout. A rule
    built on it would fire only on an idle system — never during the outage it
    exists for.
    """

    def test_a_healthy_breaker_reports_no_unhealthy_time(self) -> None:
        self.assertIsNone(self._primary_breaker().unhealthy_for_seconds())

    def test_the_clock_starts_when_the_breaker_first_opens(self) -> None:
        self._make_primary_unhealthy()
        unhealthy = self._primary_breaker().unhealthy_for_seconds()
        self.assertIsNotNone(unhealthy)
        self.assertLess(unhealthy, 1.0)

    def test_a_failed_probe_does_not_reset_the_clock(self) -> None:
        """The regression: this is what makes the spec's rule implementable."""
        breaker = self._make_primary_unhealthy()
        breaker.recovery_timeout = 0.0
        time.sleep(0.02)
        breaker.check_or_raise()  # OPEN -> HALF_OPEN: the probe is let through
        self.assertEqual(breaker.state, "HALF_OPEN")
        breaker.record_failure()  # probe failed -> OPEN, and `_opened_at` resets
        self.assertEqual(breaker.state, "OPEN")
        self.assertGreaterEqual(
            breaker.unhealthy_for_seconds(),
            0.02,
            "a failed probe reset the unhealthy clock, so failover could never engage",
        )

    def test_the_clock_keeps_growing_across_repeated_failed_probes(self) -> None:
        breaker = self._make_primary_unhealthy()
        breaker.recovery_timeout = 0.0
        for _ in range(3):
            time.sleep(0.02)
            breaker.check_or_raise()
            breaker.record_failure()
        self.assertGreater(breaker.unhealthy_for_seconds(), 0.05)

    def test_a_genuine_recovery_clears_the_clock(self) -> None:
        breaker = self._make_primary_unhealthy()
        breaker.recovery_timeout = 0.0
        time.sleep(0.01)
        breaker.check_or_raise()
        breaker.record_success()
        self.assertEqual(breaker.state, "CLOSED")
        self.assertIsNone(breaker.unhealthy_for_seconds())

    def test_reset_clears_the_clock(self) -> None:
        breaker = self._make_primary_unhealthy()
        breaker.reset()
        self.assertIsNone(breaker.unhealthy_for_seconds())

    def test_the_clock_survives_being_read_repeatedly(self) -> None:
        """Reading it is what the router does on every call."""
        breaker = self._make_primary_unhealthy()
        first = breaker.unhealthy_for_seconds()
        for _ in range(50):
            breaker.unhealthy_for_seconds()
        self.assertGreaterEqual(breaker.unhealthy_for_seconds(), first)


# --------------------------------------------------------------------- health


class TestFailoverStatus(_FailoverCase):
    def test_a_single_endpoint_deployment_reports_no_failover(self) -> None:
        with patch("app.config.load_settings", return_value=_settings()):
            status = llm.failover_status()
        self.assertFalse(status["configured"])
        self.assertFalse(status["active"])

    def test_configured_but_healthy_is_not_active(self) -> None:
        with patch("app.config.load_settings", return_value=_armed()):
            status = llm.failover_status()
        self.assertTrue(status["configured"])
        self.assertFalse(status["active"])

    def test_active_reports_the_elapsed_unhealthy_time(self) -> None:
        self._make_primary_unhealthy()
        with patch("app.config.load_settings", return_value=_armed()), patch.object(
            config, "LLM_FAILOVER_AFTER_SECONDS", 0
        ):
            status = llm.failover_status()
        self.assertTrue(status["active"])
        self.assertIsNotNone(status["primary_unhealthy_seconds"])

    def test_a_config_failure_does_not_raise(self) -> None:
        with patch("app.config.load_settings", side_effect=RuntimeError("boom")):
            self.assertFalse(llm.failover_status()["configured"])


class TestReadinessWithFallback(unittest.TestCase):
    """Ready means \"this instance can serve a run\"."""

    def setUp(self) -> None:
        health._cached_ready_at = 0.0

    def _probe(self, settings, primary_ok: bool, fallback_ok: bool):
        probed: list[str] = []

        def fake(base_url, api_key, models, *, timeout=None):
            probed.append(base_url)
            ok = primary_ok if base_url.startswith("https://primary") else fallback_ok
            return (True, "ready") if ok else (False, f"{base_url} unreachable")

        with patch("app.config.load_settings", return_value=settings), patch(
            "app.health._probe_endpoint_models", side_effect=fake
        ):
            ready, detail = health._probe_llm_readiness()
        return ready, detail, probed

    def test_a_single_endpoint_behaves_exactly_as_before(self) -> None:
        ready, detail, probed = self._probe(_settings(), True, True)
        self.assertTrue(ready)
        self.assertEqual(detail, "ready")
        self.assertEqual(len(probed), 1, "no second endpoint means no second probe")

    def test_a_single_endpoint_that_is_down_is_not_ready(self) -> None:
        ready, detail, probed = self._probe(_settings(), False, False)
        self.assertFalse(ready)
        self.assertEqual(len(probed), 1)

    def test_a_healthy_primary_does_not_probe_the_fallback(self) -> None:
        _, _, probed = self._probe(_armed(), True, True)
        self.assertEqual(probed, ["https://primary.invalid/v1"])

    def test_a_healthy_fallback_keeps_the_pod_ready(self) -> None:
        """Otherwise a clean failover would drain the pod and page on-call."""
        ready, detail, probed = self._probe(_armed(), False, True)
        self.assertTrue(ready)
        self.assertIn("primary endpoint unavailable", detail)
        self.assertIn("fallback endpoint ready", detail)
        self.assertEqual(probed, ["https://primary.invalid/v1", "https://fallback.invalid/v1"])

    def test_both_endpoints_down_is_not_ready(self) -> None:
        ready, detail, _ = self._probe(_armed(), False, False)
        self.assertFalse(ready)
        self.assertIn("primary:", detail)
        self.assertIn("fallback:", detail)

    def test_the_fallback_probe_uses_the_fallback_models(self) -> None:
        seen: dict[str, tuple] = {}

        def fake(base_url, api_key, models, *, timeout=None):
            seen[base_url] = models
            return (base_url.startswith("https://fallback"), "ready")

        with patch("app.config.load_settings", return_value=_armed()), patch(
            "app.health._probe_endpoint_models", side_effect=fake
        ):
            health._probe_llm_readiness()
        self.assertEqual(
            [model for _, model in seen["https://fallback.invalid/v1"]],
            ["fallback-main", "fallback-fast"],
        )

    def test_health_payload_reports_failover_state_without_network_io(self) -> None:
        with patch("app.config.load_settings", return_value=_armed()):
            payload = health._health_payload()
        self.assertIn("llm_failover", payload)
        self.assertTrue(payload["llm_failover"]["configured"])
        self.assertFalse(payload["llm_failover"]["active"])


# -------------------------------------------------------------------- metrics


class TestFailoverMetrics(_FailoverCase):
    def _scrape(self) -> dict:
        text = metrics.render_prometheus({"uptime_s": 1})
        out = {}
        for line in text.splitlines():
            if line and not line.startswith("#"):
                name, _, value = line.rpartition(" ")
                out[name] = value
        return out

    def test_the_failover_gauges_are_always_present(self) -> None:
        with patch("app.config.load_settings", return_value=_armed()):
            samples = self._scrape()
        self.assertEqual(samples["va_lse_llm_failover_enabled"], "1")
        self.assertEqual(samples["va_lse_llm_failover_active"], "0")
        self.assertEqual(samples["va_lse_llm_failover_after_seconds"], "300.0")

    def test_a_single_endpoint_reports_failover_disabled(self) -> None:
        with patch("app.config.load_settings", return_value=_settings()):
            samples = self._scrape()
        self.assertEqual(samples["va_lse_llm_failover_enabled"], "0")
        self.assertEqual(samples["va_lse_llm_failover_active"], "0")

    def test_active_becomes_one_only_past_the_threshold(self) -> None:
        self._make_primary_unhealthy()
        with patch("app.config.load_settings", return_value=_armed()):
            self.assertEqual(self._scrape()["va_lse_llm_failover_active"], "0")
        with patch("app.config.load_settings", return_value=_armed()), patch.object(
            config, "LLM_FAILOVER_AFTER_SECONDS", 0
        ):
            self.assertEqual(self._scrape()["va_lse_llm_failover_active"], "1")

    def test_a_healthy_primary_reports_zero_unhealthy_seconds_not_an_absent_series(self) -> None:
        """0 is a known value; absence would be indistinguishable from a scrape gap."""
        with patch("app.config.load_settings", return_value=_armed()):
            samples = self._scrape()
        self.assertEqual(samples["va_lse_llm_primary_unhealthy_seconds"], "0.0")
        self.assertEqual(samples['va_lse_circuit_breaker_unhealthy_seconds{breaker="llm"}'], "0.0")

    def test_the_breaker_gauge_is_labelled_per_endpoint(self) -> None:
        circuit_breaker.get_llm_breaker(LLM_FALLBACK_BREAKER_NAME)
        with patch("app.config.load_settings", return_value=_armed()):
            samples = self._scrape()
        self.assertIn('va_lse_circuit_breaker_state{breaker="llm"}', samples)
        self.assertIn('va_lse_circuit_breaker_state{breaker="llm-fallback"}', samples)

    def test_every_new_family_is_registered_in_the_documented_names(self) -> None:
        known = set(metrics.metric_names())
        for name in (
            "va_lse_llm_failover_enabled",
            "va_lse_llm_failover_active",
            "va_lse_llm_failover_after_seconds",
            "va_lse_llm_primary_unhealthy_seconds",
            "va_lse_llm_endpoint_duration_ms",
            "va_lse_llm_endpoint_calls_total",
            "va_lse_llm_failover_total",
            "va_lse_circuit_breaker_unhealthy_seconds",
        ):
            self.assertIn(name, known)

    def test_the_endpoint_families_are_reported_for_a_single_endpoint_run(self) -> None:
        """Every call has an endpoint, even when there is only one."""
        metrics.observe_llm_call("draft", "ok", 5, endpoint=llm.PRIMARY_ENDPOINT)
        samples = self._scrape()
        self.assertIn('va_lse_llm_endpoint_calls_total{endpoint="primary",outcome="ok"}', samples)
        self.assertEqual(
            samples['va_lse_llm_endpoint_calls_total{endpoint="primary",outcome="ok"}'], "1.0"
        )

    def test_calls_without_an_endpoint_do_not_invent_one(self) -> None:
        metrics.observe_llm_call("draft", "ok", 5)
        text = metrics.render_prometheus({"uptime_s": 1})
        self.assertNotIn("va_lse_llm_endpoint_calls_total{", text.split("# TYPE va_lse_llm_endpoint_calls_total")[1].split("#")[0] if "va_lse_llm_endpoint_calls_total" in text else "")
        self.assertIn('va_lse_llm_calls_total{outcome="ok",phase="draft"}', text)


# ------------------------------------------------------------------ stamping


class TestUsageStamping(unittest.TestCase):
    def test_a_single_endpoint_run_still_names_its_endpoint(self) -> None:
        tracker = UsageTracker()
        tracker.record(model="m", phase="p", system="s", user="u", content="c")
        self.assertEqual(tracker.endpoints_used(), [llm.PRIMARY_ENDPOINT])
        self.assertFalse(tracker.used_fallback)

    def test_both_endpoints_are_listed_in_a_stable_order(self) -> None:
        tracker = UsageTracker()
        tracker.record(
            model="m", phase="p", system="s", user="u", content="c", endpoint=llm.FALLBACK_ENDPOINT
        )
        tracker.record(model="m", phase="p", system="s", user="u", content="c")
        self.assertEqual(
            tracker.endpoints_used(), [llm.PRIMARY_ENDPOINT, llm.FALLBACK_ENDPOINT]
        )
        self.assertTrue(tracker.used_fallback)

    def test_the_summary_counts_fallback_calls(self) -> None:
        tracker = UsageTracker()
        for endpoint in (llm.FALLBACK_ENDPOINT, llm.FALLBACK_ENDPOINT, llm.PRIMARY_ENDPOINT):
            tracker.record(
                model="m", phase="p", system="s", user="u", content="c", endpoint=endpoint
            )
        summary = tracker.summary()
        self.assertEqual(summary["calls"], 3)
        self.assertEqual(summary["fallback_calls"], 2)
        self.assertEqual(summary["endpoints"], [llm.PRIMARY_ENDPOINT, llm.FALLBACK_ENDPOINT])

    def test_an_empty_run_reports_no_endpoints(self) -> None:
        self.assertEqual(UsageTracker().endpoints_used(), [])

    def test_the_endpoint_survives_a_queue_round_trip(self) -> None:
        """The worker's report is what the user reads — the stamp must get there."""
        tracker = UsageTracker()
        tracker.record(
            model="m", phase="p", system="s", user="u", content="c", endpoint=llm.FALLBACK_ENDPOINT
        )
        raw = json.loads(json.dumps(usage_to_json(tracker)))
        self.assertEqual(raw["entries"][0]["endpoint"], "fallback")
        restored = usage_from_json(raw)
        self.assertTrue(restored.used_fallback)
        self.assertEqual(restored.endpoints_used(), [llm.FALLBACK_ENDPOINT])

    def test_a_payload_from_an_older_pod_reads_as_primary(self) -> None:
        """Backward compatibility: a queued job submitted before this feature."""
        legacy = {
            "entries": [
                {"model": "m", "phase": "p", "prompt_tokens": 1, "completion_tokens": 1}
            ]
        }
        restored = usage_from_json(legacy)
        self.assertEqual(restored.endpoints_used(), [llm.PRIMARY_ENDPOINT])
        self.assertFalse(restored.used_fallback)


class TestOutageRehearsal(_FailoverCase):
    """A full outage walked end to end, asserting what an operator can observe.

    This deliberately does not re-test the routing table (``TestEndpointRouting``
    does that). It drives the real ``HealthHandler`` over a real socket and asserts
    the *observable* contract at each stage of an outage, in order, because that is
    the sequence someone actually depends on:

    healthy -> breaker opens -> grace period -> failover -> probe -> recovery.

    Every stage is asserted against the same three surfaces an operator has:
    ``/health``, ``/metrics``, and the audit record of a run.
    """

    def setUp(self) -> None:
        super().setUp()
        from app import health

        self.health = health
        self.health.stop_health_server()
        server = health.start_health_server(port=0, host="127.0.0.1")
        assert server is not None
        time.sleep(0.15)
        self.port = server.server_address[1]
        self.addCleanup(self.health.stop_health_server)

    def _get(self, path: str) -> dict:
        import urllib.request

        url = f"http://127.0.0.1:{self.port}{path}"
        return json.loads(urllib.request.urlopen(url, timeout=5).read())  # noqa: S310

    def _samples(self) -> dict:
        import urllib.request

        url = f"http://127.0.0.1:{self.port}/metrics"
        body = urllib.request.urlopen(url, timeout=5).read().decode()  # noqa: S310
        out: dict[str, str] = {}
        for line in body.splitlines():
            if line and not line.startswith("#"):
                name, _, value = line.rpartition(" ")
                out[name] = value
        return out

    def _break_primary(self) -> None:
        for _ in range(self._primary_breaker().failure_threshold):
            self._primary_breaker().record_failure()

    def test_a_single_endpoint_outage_reports_no_failover_at_any_stage(self) -> None:
        """Nothing about this feature may appear to be doing something."""
        with patch("app.config.load_settings", return_value=_settings()):
            self.assertEqual(self._get("/health")["llm_failover"]["configured"], False)
            self._break_primary()
            self.assertEqual(self._get("/health")["llm_failover"]["active"], False)
            samples = self._samples()
        self.assertEqual(samples["va_lse_llm_failover_enabled"], "0")
        self.assertEqual(samples["va_lse_llm_failover_active"], "0")
        self.assertEqual(samples["va_lse_circuit_breaker_open{breaker=\"llm\"}"], "1")

    def test_the_full_outage_timeline(self) -> None:
        armed = _armed()
        with patch("app.config.load_settings", return_value=armed):
            client = self._client(armed)

            # -- stage 1: healthy --------------------------------------------
            self.assertEqual(client.chat("s", "u", phase="draft"), "from primary")
            healthy = self._get("/health")["llm_failover"]
            self.assertTrue(healthy["configured"])
            self.assertFalse(healthy["active"])
            self.assertEqual(self._samples()["va_lse_llm_failover_active"], "0")

            # -- stage 2: the breaker opens, inside the grace period ---------
            self._break_primary()
            self._primary_breaker().recovery_timeout = 3600  # no probe yet
            grace = self._get("/health")["llm_failover"]
            self.assertFalse(grace["active"], "must not fail over during the grace window")
            self.assertIsNotNone(grace["primary_unhealthy_seconds"])
            samples = self._samples()
            self.assertEqual(samples["va_lse_circuit_breaker_open{breaker=\"llm\"}"], "1")
            self.assertGreater(float(samples["va_lse_llm_primary_unhealthy_seconds"]), 0.0)
            # The contract the spec asked for: no failover before the timeout.
            with self.assertRaises(CircuitBreakerOpenError):
                client.chat("s", "u", phase="draft")
            client._fallback_client.chat.completions.create.assert_not_called()  # type: ignore[union-attr]

            # -- stage 3: past the grace period ------------------------------
            with patch.object(config, "LLM_FAILOVER_AFTER_SECONDS", 0):
                active = self._get("/health")["llm_failover"]
                self.assertTrue(active["active"])
                self.assertEqual(self._samples()["va_lse_llm_failover_active"], "1")
                client._client.chat.completions.create = MagicMock(  # type: ignore[attr-defined]
                    side_effect=Exception("primary still down")
                )
                self.assertEqual(client.chat("s", "u", phase="draft"), "from fallback")
                self.assertTrue(client.usage.used_fallback)
                # Readiness must stay up: the pod can serve runs.
                import urllib.request

                ready_url = f"http://127.0.0.1:{self.port}/ready"
                with patch(
                    "app.health._probe_llm_readiness",
                    return_value=(True, "primary endpoint unavailable; fallback endpoint ready"),
                ):
                    self.assertEqual(  # noqa: S310
                        urllib.request.urlopen(ready_url, timeout=5).status, 200
                    )

            # -- stage 4: the primary recovers, detected by a real call ------
            self._primary_breaker().recovery_timeout = 0.0
            time.sleep(0.01)
            client._client.chat.completions.create = MagicMock(  # type: ignore[attr-defined]
                return_value=_response("from primary")
            )
            self.assertEqual(
                client.chat("s", "u", phase="draft"),
                "from primary",
                "a failed probe must not strand traffic on the backup",
            )
            self.assertEqual(self._primary_breaker().state, "CLOSED")

            # -- stage 5: back to normal ------------------------------------
            recovered = self._get("/health")["llm_failover"]
            self.assertFalse(recovered["active"])
            self.assertIsNone(recovered["primary_unhealthy_seconds"])
            self.assertEqual(self._samples()["va_lse_llm_failover_active"], "0")

    def test_a_probe_during_the_outage_does_not_fail_the_run(self) -> None:
        """Stage 4's counterpart: the probe is due *and* the primary is still down."""
        armed = _armed()
        with patch("app.config.load_settings", return_value=armed), patch.object(
            config, "LLM_FAILOVER_AFTER_SECONDS", 0
        ):
            client = self._client(armed, primary_ok=False)
            self._primary_breaker().recovery_timeout = 0.0
            for _ in range(self._primary_breaker().failure_threshold):
                self._primary_breaker().record_failure()
            self.assertEqual(client.chat("s", "u", phase="draft"), "from fallback")
        self.assertEqual(
            self._samples()["va_lse_llm_failover_total{reason=\"primary_call_failed\"}"],
            "1.0",
        )

    def test_the_metrics_never_claim_a_failover_that_is_not_happening(self) -> None:
        """`enabled` and `active` are different answers and must not be conflated."""
        with patch("app.config.load_settings", return_value=_armed()):
            samples = self._samples()
        self.assertEqual(samples["va_lse_llm_failover_enabled"], "1")
        self.assertEqual(samples["va_lse_llm_failover_active"], "0")


class TestAuditStamping(unittest.TestCase):
    """A failover changes who wrote the document, so it belongs in the audit record."""

    def setUp(self) -> None:
        import tempfile

        from app import audit

        self.audit = audit
        self._tmp = tempfile.TemporaryDirectory()
        audit._reset_for_tests()  # type: ignore[attr-defined]
        audit.configure_audit_logging(
            log_dir=self._tmp.name, log_file="audit.log", force=True
        )

    def tearDown(self) -> None:
        self.audit._reset_for_tests()  # type: ignore[attr-defined]
        self._tmp.cleanup()

    def _lines(self) -> list[dict]:
        path = Path(self._tmp.name) / "audit.log"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def test_a_failed_over_run_is_stamped_in_the_audit_record(self) -> None:
        self.audit.audit_evaluate_ok(
            request_id="req_failover",
            duration_ms=10,
            outcome={"claims": 1},
            llm_endpoints=[llm.PRIMARY_ENDPOINT, llm.FALLBACK_ENDPOINT],
        )
        self.assertEqual(self._lines()[0]["llm_endpoints"], ["primary", "fallback"])

    def test_a_normal_run_is_stamped_with_its_single_endpoint(self) -> None:
        self.audit.audit_draft_ok(
            request_id="req_normal", duration_ms=10, llm_endpoints=[llm.PRIMARY_ENDPOINT]
        )
        self.assertEqual(self._lines()[0]["llm_endpoints"], ["primary"])

    def test_the_field_is_absent_when_the_caller_does_not_supply_one(self) -> None:
        """Existing callers and older records keep their exact shape."""
        self.audit.audit_evaluate_ok(request_id="req_plain", duration_ms=10)
        self.assertNotIn("llm_endpoints", self._lines()[0])

    def test_the_stamp_is_bounded_and_truncated(self) -> None:
        self.audit.audit_evaluate_ok(
            request_id="req_wide",
            duration_ms=10,
            llm_endpoints=["a" * 200, "b", "c", "d", "e"],
        )
        stamps = self._lines()[0]["llm_endpoints"]
        self.assertEqual(len(stamps), 4, "a run cannot name more than a few endpoints")
        self.assertTrue(stamps[0].startswith("a" * 32))
        self.assertLessEqual(len(stamps[0]), 33, "truncated (plus the ellipsis)")

    def test_blank_names_are_dropped_rather_than_recorded(self) -> None:
        self.audit.audit_evaluate_ok(
            request_id="req_blank", duration_ms=10, llm_endpoints=["", "  "]
        )
        self.assertNotIn("llm_endpoints", self._lines()[0])
