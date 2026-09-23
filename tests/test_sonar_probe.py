"""Offline tests for scripts/sonar_probe.py — classification and exit codes.

The probe's value is entirely in its verdict mapping: a 400 "not supported"
means *not yet*, a 200 means *switch now*, and everything else must read as
inconclusive rather than as a yes or a no (a rate limit is not a verdict on
the model). These tests pin that table and the exit-code contract with no
network: ``_post`` is stubbed.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "sonar_probe", Path(__file__).resolve().parent.parent / "scripts" / "sonar_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


sonar_probe = _load_probe()


class TestVerdictClassification(unittest.TestCase):
    """The verdict table — the part an operator acts on."""

    def test_200_is_accepted(self) -> None:
        verdict, note = sonar_probe._classify(200, '{"id":"x"}')
        self.assertEqual(verdict, "accepted")
        self.assertTrue(note)

    def test_400_not_supported_is_rejected_not_yet(self) -> None:
        body = '{"error":{"message":"validation failed: model \\"sonar\\" is not supported"}}'
        self.assertEqual(sonar_probe._classify(400, body)[0], "rejected")

    def test_a_plain_400_is_inconclusive(self) -> None:
        # A different 400 (bad request shape) must NOT read as a model verdict.
        self.assertEqual(sonar_probe._classify(400, '{"error":{"message":"bad json"}}')[0], "inconclusive")

    def test_403_is_forbidden(self) -> None:
        self.assertEqual(sonar_probe._classify(403, '{"error":{"message":"no access"}}')[0], "forbidden")

    def test_401_is_auth(self) -> None:
        self.assertEqual(sonar_probe._classify(401, '{"error":{"code":401}}')[0], "auth")

    def test_429_is_rate_limited(self) -> None:
        self.assertEqual(sonar_probe._classify(429, "request rate limit exceeded")[0], "rate_limited")

    def test_no_status_is_unreachable(self) -> None:
        self.assertEqual(sonar_probe._classify(None, "timeout")[0], "unreachable")

    def test_404_is_unreachable(self) -> None:
        self.assertEqual(sonar_probe._classify(404, "")[0], "unreachable")

    # ------------------------------------------------- body-truncation hardening

    def test_a_marker_past_display_truncation_still_classifies(self) -> None:
        """Classification reads the whole body, not the first 200 characters.

        A rejection whose 'not supported' marker sits past the display
        truncation must still read as ``rejected`` — truncating before
        classification would misreport a real catalog verdict as
        inconclusive, and an operator could re-probe forever against a model
        that will never arrive.
        """
        pad = "x" * 500
        verdict, _ = sonar_probe._classify(400, pad + 'model "sonar" is not supported')
        self.assertEqual(verdict, "rejected")

        rate_pad = "y" * 500
        verdict, _ = sonar_probe._classify(429, rate_pad + "Request rate limit exceeded")
        self.assertEqual(verdict, "rate_limited")

    def test_a_structured_404_error_body_reads_as_rejected(self) -> None:
        """A 404 carrying the API's structured error object means the model is unserved.

        Several providers reject unknown models with 404-shaped structured
        bodies rather than 400s. Reading every 404 as 'route missing' would
        bury a catalog verdict under an unreachable-route story; the JSON
        ``error`` key separates the endpoint's own answer from a bare missing
        route. The failure direction stays conservative: rejected ("not yet"),
        never accepted.
        """
        body = '{"error": {"message": "model not found: sonar"}}'
        verdict, note = sonar_probe._classify(404, body)
        self.assertEqual(verdict, "rejected")
        self.assertIn("404", note)

    def test_a_bare_404_stays_unreachable(self) -> None:
        # No JSON error object: a missing route or proxy page, not a model verdict.
        self.assertEqual(sonar_probe._classify(404, "")[0], "unreachable")
        self.assertEqual(sonar_probe._classify(404, "<html>proxy page</html>")[0], "unreachable")

    def test_an_array_json_body_is_not_a_structured_error(self) -> None:
        # A JSON *array* has no 'error' key; json.loads succeeds but the
        # membership test must not raise — pin the non-dict shapes.
        self.assertEqual(sonar_probe._classify(404, '[1, 2, 3]')[0], "unreachable")
        self.assertEqual(sonar_probe._classify(404, '"just a string"')[0], "unreachable")


class _FakeSettings:
    configured = True
    api_key = "test-key"
    base_url = "https://api.perplexity.ai/v1"
    model_main = "perplexity/kimi-k3"
    model_fast = "perplexity/glm-5.3-flash"


class TestExitCodes(unittest.TestCase):
    """0 = switch now, 1 = not yet, 2 = inconclusive."""

    def _run(self, statuses_bodies: list[tuple[int | None, str]], models: str = "sonar"):
        seq = iter(statuses_bodies)
        with patch.object(sonar_probe, "load_settings", return_value=_FakeSettings()):
            with patch.object(sonar_probe, "_post", side_effect=lambda *_a, **_k: next(seq)):
                # The stub sequence maps 1:1 onto the model list, in order;
                # output is swallowed — these tests assert the exit code.
                with contextlib.redirect_stdout(io.StringIO()):
                    return sonar_probe.main(["--models", models])

    def test_any_acceptance_wins_exit_zero(self) -> None:
        # Even one accepted model among rejections means the switch is possible.
        self.assertEqual(
            self._run(
                [(400, "not supported"), (200, '{"ok":true}')],
                models="sonar-pro,sonar",
            ),
            0,
        )

    def test_all_rejected_is_exit_one(self) -> None:
        self.assertEqual(
            self._run([(400, "not supported"), (400, "not supported")]), 1
        )

    def test_rate_limit_is_inconclusive_not_no(self) -> None:
        self.assertEqual(
            self._run([(429, "rate limited"), (400, "not supported")]), 2
        )

    def test_unreachable_is_inconclusive(self) -> None:
        self.assertEqual(self._run([(None, "boom")]), 2)

    def test_no_api_key_is_inconclusive(self) -> None:
        settings = _FakeSettings()
        settings.api_key = ""
        with patch.object(sonar_probe, "load_settings", return_value=settings):
            self.assertEqual(sonar_probe.main([]), 2)

    def test_json_report_carries_all_fields(self) -> None:
        with patch.object(sonar_probe, "load_settings", return_value=_FakeSettings()):
            with patch.object(
                sonar_probe, "_post", return_value=(400, "model is not supported")
            ):
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    sonar_probe.main(["--json", "--models", "sonar"])
        report = json.loads(buffer.getvalue())
        self.assertEqual(report["base_url"], _FakeSettings.base_url)
        self.assertEqual(report["results"][0]["model"], "sonar")
        self.assertEqual(report["results"][0]["verdict"], "rejected")


if __name__ == "__main__":
    unittest.main()
