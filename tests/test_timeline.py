"""Unit tests for F7.S1 (Medical Event Timeline Visualization — data extraction).

Covers: date grouping via regex, 'undated' bucketing without error, LLM
fallback for otherwise-undated facts, and gap detection. No network needed.

Run from project root: .venv/bin/python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app.medical_review import (  # noqa: E402
    MedicalDigest,
    MedicalFact,
    _categorize_fact_type,
    _detect_timeline_gaps,
    _regex_extract_date,
    build_timeline_data,
)


class _FakeLLM:
    """Deterministic stub for the undated-date-inference LLM call."""

    fast_model = "fake-fast"

    def __init__(self, response: dict | Exception | None = None):
        self._settings = MagicMock(model_fast="fake-fast")
        self._response = response
        self.calls = 0

    def chat_json(self, system, user, **kwargs):
        self.calls += 1
        if isinstance(self._response, Exception):
            raise self._response
        return self._response if self._response is not None else {"dates": []}


class TestRegexExtractDate(unittest.TestCase):
    def test_exact_iso_date(self):
        self.assertEqual(_regex_extract_date("Seen on 2020-03-14 for follow-up."), ("2020-03-14", "day"))

    def test_year_month(self):
        self.assertEqual(_regex_extract_date("Admitted 2019-11."), ("2019-11-01", "month"))

    def test_month_name_year(self):
        self.assertEqual(_regex_extract_date("Diagnosed in March 2018."), ("2018-03-01", "month"))

    def test_circa_year(self):
        self.assertEqual(_regex_extract_date("Onset circa 2015."), ("2015-01-01", "year"))

    def test_bare_year(self):
        self.assertEqual(_regex_extract_date("Reported symptoms since 2021."), ("2021-01-01", "year"))

    def test_no_date(self):
        self.assertIsNone(_regex_extract_date("No dates in this text at all."))

    def test_empty_text(self):
        self.assertIsNone(_regex_extract_date(""))


class TestCategorizeFactType(unittest.TestCase):
    def test_diagnostic_types(self):
        self.assertEqual(_categorize_fact_type("diagnosis"), "diagnostic")
        self.assertEqual(_categorize_fact_type("test_result"), "diagnostic")

    def test_treatment_types(self):
        self.assertEqual(_categorize_fact_type("treatment"), "treatment")
        self.assertEqual(_categorize_fact_type("medication"), "treatment")

    def test_other_types(self):
        self.assertEqual(_categorize_fact_type("functional_limitation"), "other")
        self.assertEqual(_categorize_fact_type(""), "other")


class TestDetectTimelineGaps(unittest.TestCase):
    def test_no_gap_when_close_together(self):
        gaps = _detect_timeline_gaps(["2020-01-01", "2020-02-01"])
        self.assertEqual(gaps, [])

    def test_gap_detected_over_threshold(self):
        gaps = _detect_timeline_gaps(["2018-01-01", "2020-06-01"], threshold_days=180)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["start"], "2018-01-01")
        self.assertEqual(gaps[0]["end"], "2020-06-01")

    def test_fewer_than_two_dates_no_gaps(self):
        self.assertEqual(_detect_timeline_gaps(["2020-01-01"]), [])
        self.assertEqual(_detect_timeline_gaps([]), [])


class TestBuildTimelineData(unittest.TestCase):
    def _digest(self, facts):
        return MedicalDigest(facts=facts, pages_reviewed=1, chunks_reviewed=1)

    def test_groups_by_parsed_date_or_undated(self):
        digest = self._digest([
            MedicalFact("2020-01-15", "diagnosis", "Diagnosed with knee condition.", "a.pdf p1"),
            MedicalFact("unknown", "symptom", "Reports ongoing pain.", "b.pdf p2"),
        ])
        data = build_timeline_data(digest)
        self.assertEqual(len(data["events"]), 2)
        self.assertEqual(data["dated_count"], 1)
        self.assertEqual(data["undated_count"], 1)
        self.assertIn("2020-01-15", data["grouped"])
        self.assertIn("undated", data["grouped"])

    def test_missing_dates_bucketed_without_error_no_llm(self):
        digest = self._digest([
            MedicalFact("unknown", "treatment", "Some treatment noted.", "c.pdf p3", quote="no date here"),
        ])
        # No LLM client supplied -> must not raise, must bucket as undated.
        data = build_timeline_data(digest, llm=None)
        self.assertNotIn("error", data)
        self.assertEqual(data["undated_count"], 1)
        self.assertEqual(data["events"][0]["bucket"], "undated")

    def test_llm_fallback_infers_date_for_undated_fact(self):
        digest = self._digest([
            MedicalFact("unknown", "symptom", "Patient reported pain after an incident.", "d.pdf p4"),
        ])
        llm = _FakeLLM({"dates": [{"index": 0, "date": "2017"}]})
        data = build_timeline_data(digest, llm=llm)
        self.assertEqual(data["dated_count"], 1)
        self.assertEqual(data["undated_count"], 0)
        self.assertEqual(data["events"][0]["date_iso"], "2017-01-01")
        self.assertEqual(data["events"][0]["date_source"], "llm")
        self.assertEqual(llm.calls, 1)

    def test_llm_failure_falls_back_to_undated_without_error(self):
        digest = self._digest([
            MedicalFact("unknown", "symptom", "No inferable date here.", "e.pdf p5"),
        ])
        llm = _FakeLLM(RuntimeError("upstream failure"))
        data = build_timeline_data(digest, llm=llm)
        self.assertNotIn("error", data)
        self.assertEqual(data["undated_count"], 1)

    def test_gap_detection_returns_date_ranges(self):
        digest = self._digest([
            MedicalFact("2015-01-01", "diagnosis", "Initial diagnosis.", "f.pdf p1"),
            MedicalFact("2021-06-01", "treatment", "Resumed treatment.", "f.pdf p9"),
        ])
        data = build_timeline_data(digest)
        self.assertEqual(data["gap_count"], 1)
        self.assertEqual(data["gaps"][0]["start"], "2015-01-01")
        self.assertEqual(data["gaps"][0]["end"], "2021-06-01")

    def test_empty_digest_produces_well_formed_empty_timeline(self):
        data = build_timeline_data(self._digest([]))
        self.assertEqual(data["events"], [])
        self.assertEqual(data["dated_count"], 0)
        self.assertEqual(data["undated_count"], 0)
        self.assertEqual(data["gap_count"], 0)

    @patch("app.medical_review.track_goal")
    def test_goal_telemetry_fired_with_feature_id(self, mock_goal):
        digest = self._digest([
            MedicalFact("2020-01-01", "diagnosis", "Diagnosis noted.", "g.pdf p1"),
        ])
        build_timeline_data(digest, feature_id="feat-123")
        mock_goal.assert_called_once()
        args, kwargs = mock_goal.call_args
        self.assertEqual(args[0], "feat-123")
        self.assertEqual(kwargs.get("dated_event_count"), 1)
        self.assertEqual(kwargs.get("undated_count"), 0)
        self.assertEqual(kwargs.get("gap_count"), 0)

    @patch("app.medical_review.track_goal")
    def test_no_telemetry_when_feature_id_not_provided(self, mock_goal):
        digest = self._digest([
            MedicalFact("2020-01-01", "diagnosis", "Diagnosis noted.", "g.pdf p1"),
        ])
        build_timeline_data(digest)
        mock_goal.assert_not_called()

    @patch("app.medical_review._regex_extract_date", side_effect=RuntimeError("boom"))
    @patch("app.medical_review.track_feature_error")
    def test_error_boundary_tracks_error_and_returns_empty(self, mock_err, _mock_regex):
        digest = self._digest([
            MedicalFact("2020-01-01", "diagnosis", "Diagnosis noted.", "h.pdf p1"),
        ])
        data = build_timeline_data(digest, feature_id="feat-err")
        self.assertTrue(data.get("error"))
        self.assertEqual(data["events"], [])
        mock_err.assert_called_once()
        args, kwargs = mock_err.call_args
        self.assertEqual(args[0], "feat-err")
        self.assertEqual(kwargs.get("phase"), "build_timeline_data")
        self.assertEqual(kwargs.get("error_type"), "RuntimeError")


if __name__ == "__main__":
    unittest.main()
