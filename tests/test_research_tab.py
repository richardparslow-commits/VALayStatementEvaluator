"""Real-runtime test for the Research tab's case-derived question panel.

Driven through Streamlit's ``AppTest`` because the two behaviours this feature leans on
are ones a mocked ``st`` accepts unconditionally: a radio whose options are ``dataclass``
instances, and a button that seeds widget keys *before* those widgets exist and then asks
for a rerun. ``TestResearchCaseContext`` in ``tests/test_views.py`` covers the logic with a
MagicMock; this file is what proves the panel actually renders in the app.

No network and no API call: the test stops at rendering the panel. The one credential here
is a literal placeholder, used only to get past the tab's "not configured" guard, and it is
asserted never to appear in any rendered element.
"""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/ — log_isolation
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from log_isolation import isolate_app_logs  # noqa: E402

# Presence-only: never a real key, and never able to leave this process.
PLACEHOLDER_KEY = "pplx-placeholder-not-a-real-key"


def _digest():
    """A digest shaped like one a completed review leaves behind."""
    from app.medical_review import MedicalDigest, MedicalFact

    return MedicalDigest(
        facts=[
            MedicalFact(
                date="2019-04-02",
                type="symptom",
                description="sleep apnea documented at the VA clinic",
                source="VA Records.pdf — page 3",
                quote="he sleeps two hours a night",
            )
        ],
        conditions=["sleep apnea", "tinnitus"],
        providers=["Dr. Aloysius Pendergast", "VA Medical Center"],
        summary="Narrative summary that must not be sent with a question.",
    )


class TestResearchTabCaseQuestions(unittest.TestCase):
    def setUp(self) -> None:
        # AppTest runs the real app in-process — keep its run-log/audit-log writes out of
        # the developer's real logs/ (see tests/log_isolation.py).
        self._tmpdir = isolate_app_logs(self)

    def _app(self):
        """The app, with a completed review in session state and research configured."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(PROJECT_ROOT / "run_app.py"), default_timeout=30)
        at.session_state["timeline_digest"] = _digest()
        return at

    def test_derived_questions_render_for_the_reviewed_case(self):
        from app.research_questions import derive_questions

        expected = derive_questions(_digest())
        self.assertTrue(expected)

        at = self._app()
        with patch.dict(os.environ, {"PERPLEXITY_API_KEY": PLACEHOLDER_KEY}):
            at.run()

        self.assertFalse(
            at.exception, msg=f"app raised while rendering: {[e.value for e in at.exception]}"
        )
        options = at.radio(key="research_case_question").options
        self.assertEqual(len(options), len(expected))
        self.assertIn(
            "Load into the form",
            [button.label for button in at.button],
            msg=f"expected the load button, got: {[b.label for b in at.button]}",
        )

    def test_loading_a_derived_question_fills_the_form(self):
        from app.research_questions import derive_questions

        first = derive_questions(_digest())[0]

        at = self._app()
        with patch.dict(os.environ, {"PERPLEXITY_API_KEY": PLACEHOLDER_KEY}):
            at.run()
            at.button(key="research_load_case_question").click().run()

        self.assertEqual(at.session_state["research_question"], first.text)
        self.assertEqual(at.text_area(key="research_question").value, first.text)
        self.assertEqual(at.session_state["research_condition"], first.condition)

    def test_the_panel_is_absent_when_no_review_has_run(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(PROJECT_ROOT / "run_app.py"), default_timeout=30)
        with patch.dict(os.environ, {"PERPLEXITY_API_KEY": PLACEHOLDER_KEY}):
            at.run()

        self.assertFalse(at.exception)
        self.assertEqual(
            [r for r in at.radio if r.key == "research_case_question"],
            [],
            msg="a case panel appeared with no case in session state",
        )

    def test_no_record_text_or_key_reaches_the_rendered_page(self):
        at = self._app()
        with patch.dict(os.environ, {"PERPLEXITY_API_KEY": PLACEHOLDER_KEY}):
            at.run()

        rendered = "\n".join(
            str(element.value)
            for element in list(at.markdown) + list(at.caption) + list(at.text)
        )
        for leaked in (
            PLACEHOLDER_KEY,
            "VA Records.pdf",
            "Dr. Aloysius Pendergast",
            "two hours a night",
            "Narrative summary",
        ):
            self.assertNotIn(leaked, rendered)


if __name__ == "__main__":
    unittest.main()
