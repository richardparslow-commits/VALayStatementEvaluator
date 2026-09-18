"""Unit tests for the condition-specific-templates pre-selection engine and
its JSON data file (feature: Condition-Specific Templates).

Run from project root: .venv/bin/python -m unittest discover -s tests -v
"""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app.condition_selector import (  # noqa: E402
    AA_FORCED_TOPICS,
    DEFAULT_FALLBACK_TOPICS,
    TOPIC_LABELS,
    _CONDITION_TOPICS_PATH,
    _load_condition_topics,
    get_preselected_topics,
)


class TestConditionTopicsData(unittest.TestCase):
    """Validate the condition_topics.json structure (F2.S1.T3)."""

    def setUp(self):
        _load_condition_topics.cache_clear()

    def test_json_file_loads(self):
        with _CONDITION_TOPICS_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertIn("body_systems", data)
        self.assertGreaterEqual(len(data["body_systems"]), 3)

    def test_at_least_twenty_conditions_mapped(self):
        data = _load_condition_topics()
        total = sum(
            len(system.get("conditions", {}))
            for system in data["body_systems"].values()
        )
        self.assertGreaterEqual(total, 20)

    def test_every_topic_letter_is_valid(self):
        data = _load_condition_topics()
        valid_letters = set(TOPIC_LABELS.keys())
        for system in data["body_systems"].values():
            for topics in system.get("conditions", {}).values():
                if topics is None:
                    continue
                for letter in topics:
                    self.assertIn(letter, valid_letters)

    def test_every_body_system_has_fallback_condition(self):
        data = _load_condition_topics()
        for name, system in data["body_systems"].items():
            self.assertIn(
                "Other / not listed",
                system.get("conditions", {}),
                msg=f"{name} is missing an 'Other / not listed' fallback entry",
            )


class TestGetPreselectedTopics(unittest.TestCase):
    def test_single_condition_returns_its_mapped_topics(self):
        preselected, forced = get_preselected_topics(
            [("Musculoskeletal", "Lumbar strain / degenerative disc disease")], False
        )
        self.assertEqual(preselected, ["F", "G", "H", "K"])
        self.assertEqual(forced, [])

    def test_multiple_conditions_are_unioned(self):
        preselected, _ = get_preselected_topics(
            [
                ("Musculoskeletal", "Carpal tunnel syndrome"),  # F, H, K
                ("Mental Health", "PTSD (post-traumatic stress disorder)"),  # A, G, H, I
            ],
            False,
        )
        self.assertEqual(preselected, ["A", "F", "G", "H", "I", "K"])

    def test_unknown_condition_falls_back_to_default(self):
        preselected, _ = get_preselected_topics(
            [("Musculoskeletal", "Other / not listed")], False
        )
        self.assertEqual(preselected, sorted(DEFAULT_FALLBACK_TOPICS))

    def test_aa_toggle_forces_bcej_as_mandatory(self):
        preselected, forced = get_preselected_topics([], True)
        self.assertEqual(set(forced), set(AA_FORCED_TOPICS))
        for letter in AA_FORCED_TOPICS:
            self.assertIn(letter, preselected)

    def test_aa_toggle_merges_with_condition_topics(self):
        preselected, forced = get_preselected_topics(
            [("Musculoskeletal", "Lumbar strain / degenerative disc disease")], True
        )
        # Condition topics (F, G, H, K) plus forced A&A topics (B, C, E, J).
        self.assertEqual(preselected, ["B", "C", "E", "F", "G", "H", "J", "K"])
        self.assertEqual(set(forced), {"B", "C", "E", "J"})

    def test_no_conditions_and_no_aa_toggle_returns_empty(self):
        preselected, forced = get_preselected_topics([], False)
        self.assertEqual(preselected, [])
        self.assertEqual(forced, [])


class TestTelemetryMockMode(unittest.TestCase):
    """The telemetry module must never make network calls when unconfigured."""

    def test_send_event_noops_without_env_vars(self):
        from app import agiloop_telemetry

        with patch.dict("os.environ", {}, clear=False):
            for var in (
                "AGILOOP_INSPECT_API_KEY",
                "AGILOOP_INSPECT_URL",
                "AGILOOP_PROJECT_ID",
            ):
                import os

                os.environ.pop(var, None)
            with patch("urllib.request.urlopen") as mock_urlopen:
                agiloop_telemetry.track_impression("feature-id", "test")
                mock_urlopen.assert_not_called()

    def test_telemetry_module_has_no_hardcoded_feature_id(self):
        """Shared telemetry infra must never hardcode a feature id (neutrality rule)."""
        from app import agiloop_telemetry

        source = Path(agiloop_telemetry.__file__).read_text(encoding="utf-8")
        self.assertNotIn("02f0935a-ee5e-4083-88a2-10e11753ccc9", source)


if __name__ == "__main__":
    unittest.main()
