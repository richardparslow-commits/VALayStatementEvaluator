"""Real-runtime test for the framework-currency feature, on both surfaces it appears.

Drives the app with Streamlit's ``AppTest``. ``tests/test_views.py`` pins the logic against
a mocked ``st``; this file is what proves the two user-visible halves actually render:

* the Research tab offers the paid verification control, and reports an honest status when
  nothing has been checked;
* the *Evaluate* tab — the app's own output — carries the stale-topic flag, which is the
  point of the whole feature.

No network and no API call: the Evaluate-side test writes a stored verdict directly into
the cache, which is exactly how the app reads one, and the Research-side test only asserts
the control exists. The one credential is a literal placeholder used to get past the tab's
"not configured" guard.
"""
import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/ — log_isolation
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from app import knowledge_currency as currency  # noqa: E402
from log_isolation import isolate_app_logs  # noqa: E402

# Presence-only: never a real key, and never able to leave this process.
PLACEHOLDER_KEY = "pplx-placeholder-not-a-real-key"


def _stale_report():
    """A stored verdict saying topic A has moved, against the framework on disk."""
    return currency.CurrencyReport(
        checked_at=datetime.now(timezone.utc).isoformat(),
        fingerprint=currency.framework_fingerprint(),
        verdicts=(
            currency.TopicVerdict(
                topic="A",
                status=currency.STATUS_CHANGED,
                note="The hazard examples no longer match the current rating criteria.",
                authority="38 C.F.R. § 4.130",
            ),
            currency.TopicVerdict(topic="G", status=currency.STATUS_CURRENT, note="Still current."),
        ),
    )


def _eval_result():
    """A completed evaluation whose applicable topics include the stale one."""
    from app.evaluate import EvaluationResult

    return EvaluationResult(
        claimed_condition="post-traumatic stress disorder",
        executive_summary="Summary.",
        topic_rows=[
            {
                "topic": "A",
                "applicable": True,
                "coverage": "absent",
                "evidence": "",
                "gap_note": "Describe the stove incident.",
            },
            {
                "topic": "G",
                "applicable": True,
                "coverage": "partial",
                "evidence": "before and after",
                "gap_note": "Add dates.",
            },
        ],
        topic_focus="PTSD, need for regular assistance",
    )


class TestFrameworkCurrencyInTheApp(unittest.TestCase):
    def setUp(self) -> None:
        # AppTest runs the real app in-process — keep its run-log/audit-log writes out of
        # the developer's real logs/ (see tests/log_isolation.py), and start from a process
        # whose currency cache is empty so "not verified" means what it says.
        self._tmpdir = isolate_app_logs(self)
        currency.forget_report()
        self.addCleanup(currency.forget_report)

    def _app(self, **session):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(PROJECT_ROOT / "run_app.py"), default_timeout=30)
        for key, value in session.items():
            at.session_state[key] = value
        return at

    def test_research_tab_offers_the_paid_check_and_admits_it_has_not_run(self):
        at = self._app()
        with patch.dict(os.environ, {"PERPLEXITY_API_KEY": PLACEHOLDER_KEY}):
            at.run()

        self.assertFalse(
            at.exception, msg=f"app raised while rendering: {[e.value for e in at.exception]}"
        )
        self.assertIn(
            "Verify selected topics",
            [button.label for button in at.button],
            msg=f"expected the verify control, got: {[b.label for b in at.button]}",
        )
        captions = " ".join(str(element.value) for element in at.caption)
        self.assertIn("Not verified yet", captions)

    def test_evaluator_output_flags_the_stale_topic(self):
        currency.store_report(_stale_report())

        at = self._app(eval_result=_eval_result(), preselected_topics_eval=["A", "G"])
        with patch.dict(os.environ, {"PERPLEXITY_API_KEY": PLACEHOLDER_KEY}):
            at.run()

        self.assertFalse(
            at.exception, msg=f"app raised while rendering: {[e.value for e in at.exception]}"
        )
        warnings = " ".join(str(alert.value) for alert in at.warning)
        self.assertIn("changed under current VA law", warnings)

        detail = " ".join(str(element.value) for element in at.markdown)
        self.assertIn("38 C.F.R. § 4.130", detail)
        self.assertIn("The hazard examples no longer match", detail)

    def test_evaluator_output_does_not_flag_a_topic_the_case_does_not_touch(self):
        """Topic A is stale, but this run's topics are G only — no warning, no noise."""
        currency.store_report(_stale_report())

        at = self._app(eval_result=_eval_result(), preselected_topics_eval=["G"])
        with patch.dict(os.environ, {"PERPLEXITY_API_KEY": PLACEHOLDER_KEY}):
            at.run()

        warnings = " ".join(str(alert.value) for alert in at.warning)
        self.assertNotIn("changed under current VA law", warnings)
        self.assertIn("verified", " ".join(str(e.value) for e in at.caption).lower())

    def test_evaluator_output_says_unverified_when_nothing_was_checked(self):
        at = self._app(eval_result=_eval_result(), preselected_topics_eval=["A"])
        with patch.dict(os.environ, {"PERPLEXITY_API_KEY": PLACEHOLDER_KEY}):
            at.run()

        captions = " ".join(str(element.value) for element in at.caption)
        self.assertIn("has not been verified", captions)


if __name__ == "__main__":
    unittest.main()
