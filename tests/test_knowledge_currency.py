"""Offline tests for framework currency (app/knowledge_currency.py).

This feature makes a claim about the app's own legal reference text, so the tests are
mostly about the *honesty* of that claim rather than about plumbing:

* a verdict must name exactly the topics that were checked, with missing topics filled in
  as unconfirmed rather than dropped — a report where four of twelve topics vanish must not
  read as twelve clean verdicts;
* a verdict must stop applying the moment the committed text changes, because otherwise
  editing the checklist would inherit a "still current" stamp for text nobody reviewed;
* absent, expired, and changed-text must all read as *unverified*, never as current.

The API is never reached: ``research`` is stubbed, and the checks that only read a stored
report are asserted to make no call at all.
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app import config  # noqa: E402
from app import knowledge_currency as currency  # noqa: E402
from app.perplexity_agent import (  # noqa: E402
    GroundedAnswer,
    PerplexityConfigurationError,
    PerplexityParseError,
)

NOW = datetime(2026, 9, 18, 12, 0, 0, tzinfo=timezone.utc)


def _settings(**overrides) -> config.Settings:
    base = dict(
        api_key="fake-key",
        base_url="https://primary.invalid/v1",
        model_main="qwen3.7-max",
        model_fast="qwen3.7-flash",
        fetch_api_key="",
        fetch_base_url="",
        fetch_records_path="",
    )
    base.update(overrides)
    return config.Settings(**base)


def _answer(findings=None, *, preset="low", model="gpt-5.6-sol", latency_ms=1234) -> GroundedAnswer:
    return GroundedAnswer(
        text="Because…",
        findings=findings,
        preset=preset,
        model=model,
        response_id="resp_abc",
        latency_ms=latency_ms,
    )


def _verdict_rows(*pairs) -> dict:
    return {
        "verdicts": [
            {"topic": topic, "status": status, "note": f"note for {topic}", "authority": "38 C.F.R. § 4.130"}
            for topic, status in pairs
        ]
    }


def _report(*pairs, checked_at=None, fingerprint=None, latency_ms=10) -> currency.CurrencyReport:
    return currency.CurrencyReport(
        checked_at=checked_at or NOW.isoformat(),
        fingerprint=fingerprint or currency.framework_fingerprint(),
        verdicts=tuple(
            currency.TopicVerdict(topic=topic, status=status, note=f"note for {topic}")
            for topic, status in pairs
        ),
        latency_ms=latency_ms,
    )


class TestParsingTheCommittedFramework(unittest.TestCase):
    def test_real_checklist_yields_every_lettered_topic(self) -> None:
        sections = currency.parse_topic_sections(config.load_knowledge("topic_checklist.md"))
        letters = [section.letter for section in sections]
        self.assertEqual(letters, list("ABCDEFGHIJKL"))
        for section in sections:
            self.assertTrue(section.title, msg=f"topic {section.letter} has no title")
            self.assertGreater(len(section.text), 40, msg=f"topic {section.letter} is empty")

    def test_unlettered_prose_sections_are_skipped(self) -> None:
        sections = currency.parse_topic_sections(config.load_knowledge("topic_checklist.md"))
        self.assertNotIn("Writer guidelines", [section.title for section in sections])

    def test_sections_do_not_overlap(self) -> None:
        sections = currency.parse_topic_sections(config.load_knowledge("topic_checklist.md"))
        titles = [section.title for section in sections]
        self.assertNotIn("##", " ".join(titles))

    def test_unknown_letters_and_duplicates_are_dropped(self) -> None:
        self.assertEqual(currency.normalize_topics(["a", "A", "zz", "", "L"]), ["A", "L"])
        self.assertEqual(currency.normalize_topics([1, None]), [])


class TestFingerprint(unittest.TestCase):
    def test_is_stable_for_unchanged_text(self) -> None:
        self.assertEqual(currency.framework_fingerprint(), currency.framework_fingerprint())

    def test_changes_when_the_checklist_changes(self) -> None:
        """The whole invalidation story: edited text must not keep a verdict alive."""

        def fake_load(name: str) -> str:
            text = config.load_knowledge(name)
            return text + "\n\n## M. A newly committed topic\nSomething new.\n" if name.endswith(
                "topic_checklist.md"
            ) else text

        with patch.object(currency, "load_knowledge", side_effect=fake_load):
            edited = currency.framework_fingerprint()
        self.assertNotEqual(edited, currency.framework_fingerprint())

    def test_unreadable_framework_cannot_validate_a_verdict(self) -> None:
        with patch.object(currency, "load_knowledge", side_effect=FileNotFoundError("gone")):
            self.assertEqual(currency.framework_fingerprint(), "missing")


class TestVerifyFrameworkCurrency(unittest.TestCase):
    def _verify(self, *, topics, findings, settings=None):
        captured: dict = {}

        def fake_research(question, **kwargs):
            captured["question"] = question
            captured.update(kwargs)
            return _answer(findings)

        with patch.object(currency, "research", side_effect=fake_research), patch.object(
            currency, "store_report"
        ) as stored:
            report = currency.verify_framework_currency(
                settings=settings or _settings(), topics=topics
            )
        captured["stored"] = stored
        return report, captured

    def test_sends_the_committed_topic_text_and_restricts_the_schema(self) -> None:
        _report, captured = self._verify(
            topics=["A", "D"], findings=_verdict_rows(("A", "current"), ("D", "changed"))
        )
        schema = captured["schema"]["json_schema"]["schema"]
        topics_enum = schema["properties"]["verdicts"]["items"]["properties"]["topic"]["enum"]
        self.assertEqual(topics_enum, ["A", "D"])
        instructions = captured["instructions"]
        # The committed text under review travels with the question, delimited as data.
        self.assertIn("Hazards and Dangers", instructions)
        self.assertIn("never a URL", instructions)
        self.assertIn("<<<", instructions)
        self.assertIn(">>>", instructions)

    def test_requests_only_the_selected_topics_text(self) -> None:
        _report, captured = self._verify(topics=["G"], findings=_verdict_rows(("G", "current")))
        self.assertNotIn("[A] Hazards", captured["instructions"])
        self.assertIn("[G] Context and Symptom Progression", captured["instructions"])

    def test_defaults_to_the_official_source_domains(self) -> None:
        sample = _settings(perplexity_sources="va.gov,ecfr.gov")
        _report, captured = self._verify(
            topics=["A"], findings=_verdict_rows(("A", "current")), settings=sample
        )
        self.assertEqual(captured["domains"], ["va.gov", "ecfr.gov"])

    def test_report_covers_every_requested_topic_in_order(self) -> None:
        report, _captured = self._verify(
            topics=["L", "A"],
            findings=_verdict_rows(("A", "current"), ("L", "changed")),
        )
        self.assertEqual(report.topics, ("A", "L"))
        self.assertEqual([verdict.status for verdict in report.verdicts], ["current", "changed"])

    def test_a_topic_the_model_skipped_is_unconfirmed_not_absent(self) -> None:
        """The one error this feature cannot afford is silence reading as approval."""
        report, _captured = self._verify(
            topics=["A", "B"], findings=_verdict_rows(("A", "current"))
        )
        verdicts = {verdict.topic: verdict for verdict in report.verdicts}
        self.assertEqual(sorted(verdicts), ["A", "B"])
        self.assertEqual(verdicts["B"].status, currency.STATUS_UNCONFIRMED)
        self.assertIn("no verdict", verdicts["B"].note)

    def test_unexpected_statuses_become_unclear_and_stale_aliases_become_changed(self) -> None:
        report, _captured = self._verify(
            topics=["A", "B"],
            findings={
                "verdicts": [
                    {"topic": "A", "status": "probably fine", "note": "", "authority": ""},
                    {"topic": "B", "status": "OUTDATED", "note": "", "authority": ""},
                ]
            },
        )
        statuses = {verdict.topic: verdict.status for verdict in report.verdicts}
        self.assertEqual(statuses["A"], currency.STATUS_UNCONFIRMED)
        self.assertEqual(statuses["B"], currency.STATUS_CHANGED)

    def test_report_records_the_call_and_the_text_it_was_about(self) -> None:
        report, captured = self._verify(
            topics=["A"], findings=_verdict_rows(("A", "current"))
        )
        self.assertEqual(report.fingerprint, currency.framework_fingerprint())
        self.assertEqual(report.model, "gpt-5.6-sol")
        self.assertEqual(report.latency_ms, 1234)
        self.assertEqual(report.request_id, "resp_abc")
        captured["stored"].assert_called_once()

    def test_empty_selection_is_a_configuration_error_and_makes_no_call(self) -> None:
        with patch.object(currency, "research") as research_call:
            with self.assertRaises(PerplexityConfigurationError):
                currency.verify_framework_currency(settings=_settings(), topics=[])
        research_call.assert_not_called()

    def test_no_structured_findings_is_a_parse_error(self) -> None:
        with self.assertRaises(PerplexityParseError):
            self._verify(topics=["A"], findings=None)

    def test_a_verdict_for_a_topic_that_was_not_asked_about_is_dropped(self) -> None:
        report, _captured = self._verify(
            topics=["A"], findings=_verdict_rows(("A", "current"), ("Z", "changed"))
        )
        self.assertEqual(report.topics, ("A",))


class TestReadingTheStoredVerdict(unittest.TestCase):
    def test_fresh_report_is_fresh(self) -> None:
        report = _report(("A", "current"))
        self.assertEqual(
            currency.freshness(report, ttl_days=30, now=NOW), currency.STATE_FRESH
        )

    def test_missing_report_is_never_checked(self) -> None:
        self.assertEqual(
            currency.freshness(None, ttl_days=30, now=NOW), currency.STATE_NEVER_CHECKED
        )

    def test_old_report_is_expired(self) -> None:
        old = _report(("A", "current"), checked_at=(NOW - timedelta(days=31)).isoformat())
        self.assertEqual(
            currency.freshness(old, ttl_days=30, now=NOW), currency.STATE_EXPIRED
        )

    def test_edited_framework_invalidates_even_a_recent_verdict(self) -> None:
        report = _report(("A", "current"), fingerprint="some-old-fingerprint")
        self.assertEqual(
            currency.freshness(report, ttl_days=30, now=NOW),
            currency.STATE_FRAMEWORK_CHANGED,
        )

    def test_flag_narrows_to_the_cases_own_topics(self) -> None:
        report = _report(("A", "changed"), ("J", "changed"), ("D", "unclear"))
        flag = currency.case_currency_flag(["D", "J"], ttl_days=30, report=report, now=NOW)
        self.assertEqual([v.topic for v in flag.stale], ["J"])
        self.assertEqual([v.topic for v in flag.unconfirmed], ["D"])
        self.assertTrue(flag.verified)

    def test_flag_without_a_report_makes_no_call_and_claims_nothing(self) -> None:
        # ``report=None`` means "read the stored one"; the cache is stubbed empty so this
        # is the no-verification case regardless of what the host process has cached.
        with patch.object(currency, "load_report", return_value=None), patch.object(
            currency, "research"
        ) as research_call:
            flag = currency.case_currency_flag(["A"], ttl_days=30, now=NOW)
        research_call.assert_not_called()
        self.assertEqual(flag.state, currency.STATE_NEVER_CHECKED)
        self.assertFalse(flag.verified)
        self.assertEqual(flag.stale, ())

    def test_expired_report_still_reports_its_stale_topics(self) -> None:
        """Expiry downgrades confidence; it must not hide a known-stale topic."""
        old = _report(("A", "changed"), checked_at=(NOW - timedelta(days=400)).isoformat())
        flag = currency.case_currency_flag(["A"], ttl_days=30, report=old, now=NOW)
        self.assertEqual(flag.state, currency.STATE_EXPIRED)
        self.assertEqual([v.topic for v in flag.stale], ["A"])

    def test_age_is_measured_from_the_check(self) -> None:
        report = _report(("A", "current"), checked_at=(NOW - timedelta(days=3)).isoformat())
        self.assertAlmostEqual(report.age_days(now=NOW) or 0.0, 3.0, places=3)

    def test_unreadable_timestamp_reads_as_stale_rather_than_crashing(self) -> None:
        report = _report(("A", "current"), checked_at="not a timestamp")
        self.assertIsNone(report.age_days(now=NOW))
        self.assertEqual(
            currency.freshness(report, ttl_days=30, now=NOW), currency.STATE_EXPIRED
        )


class TestReportSerialization(unittest.TestCase):
    def test_round_trip_preserves_everything_that_matters(self) -> None:
        original = _report(("A", "changed"), ("B", "unclear"))
        restored = currency.CurrencyReport.from_json(original.to_json())
        self.assertIsNotNone(restored)
        assert restored is not None  # narrow for the type checker
        self.assertEqual(restored.fingerprint, original.fingerprint)
        self.assertEqual(restored.checked_at, original.checked_at)
        self.assertEqual(restored.verdicts, original.verdicts)
        self.assertEqual(restored.latency_ms, original.latency_ms)

    def test_unusable_cache_values_read_as_no_report(self) -> None:
        for raw in ("", "{", "null", "[]", '{"checked_at": 1}', '{"verdicts": []}'):
            self.assertIsNone(currency.CurrencyReport.from_json(raw), msg=repr(raw))


if __name__ == "__main__":
    unittest.main()
