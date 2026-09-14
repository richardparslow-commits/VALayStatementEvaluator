"""Offline unit tests for the feature-id-neutral telemetry helper."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import telemetry  # noqa: E402


class TestTelemetryConfiguration(unittest.TestCase):
    def tearDown(self):
        telemetry.end_session()

    def test_mock_mode_when_env_vars_missing(self):
        with patch.dict(
            "os.environ",
            {"AGILOOP_INSPECT_API_KEY": "", "AGILOOP_PROJECT_ID": ""},
            clear=False,
        ):
            configured, reason = telemetry._inspect_configured()
        self.assertFalse(configured)
        self.assertIn("mock", reason)

    def test_real_mode_when_env_vars_present(self):
        with patch.dict(
            "os.environ",
            {"AGILOOP_INSPECT_API_KEY": "key123", "AGILOOP_PROJECT_ID": "proj-1"},
            clear=False,
        ):
            configured, reason = telemetry._inspect_configured()
        self.assertTrue(configured)
        self.assertEqual(reason, "real")

    def test_partial_configuration_is_still_reported_as_mock(self):
        with patch.dict(
            "os.environ",
            {"AGILOOP_INSPECT_API_KEY": "key123", "AGILOOP_PROJECT_ID": ""},
            clear=False,
        ):
            configured, reason = telemetry._inspect_configured()
        self.assertFalse(configured)
        self.assertIn("AGILOOP_PROJECT_ID", reason)


class TestTelemetryEventShape(unittest.TestCase):
    def tearDown(self):
        telemetry.end_session()

    def test_track_interaction_dispatches_feature_scoped_event(self):
        captured = []
        with patch.object(telemetry, "_dispatch", side_effect=lambda e: captured.append(e)):
            telemetry.track_interaction("feature-123", {"records_fetched": 2})
        self.assertEqual(len(captured), 1)
        event = captured[0]
        self.assertEqual(event["type"], "feature.interaction")
        self.assertEqual(event["featureId"], "feature-123")
        self.assertEqual(event["metadata"]["records_fetched"], 2)

    def test_track_feature_error_never_raises(self):
        with patch.object(telemetry, "_send", side_effect=RuntimeError("network down")):
            try:
                telemetry.track_feature_error("feature-123", ValueError("boom"), retry_attempt=1)
            except Exception as exc:  # noqa: BLE001
                self.fail(f"track_feature_error must never raise, got: {exc}")

    def test_init_telemetry_is_idempotent(self):
        telemetry.init_telemetry("user@example.com")
        first_session = telemetry._session_id
        telemetry.init_telemetry("user@example.com")
        self.assertEqual(first_session, telemetry._session_id)


if __name__ == "__main__":
    unittest.main()
