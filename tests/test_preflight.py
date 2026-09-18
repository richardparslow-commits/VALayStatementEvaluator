"""Tests for the endpoint preflight: policy, gate, and the notice it leaves.

Like every module in this suite, it imports the hermetic harness first: the app
freezes its configuration at import, and a test file that imports the app before
the harness tests whatever `.env` this machine happens to have. The settings here
are built explicitly and the probe is canned, so these tests never read ambient
configuration and never open a socket.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app.llm import ModelProbe  # noqa: E402
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


def _check(probe_result: ModelProbe, settings: SimpleNamespace | None = None) -> preflight.Verdict:
    """Run the policy against a canned probe result."""
    return preflight.check_endpoint(
        settings or _settings(), probe=lambda base_url, api_key: probe_result
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


if __name__ == "__main__":
    unittest.main()
