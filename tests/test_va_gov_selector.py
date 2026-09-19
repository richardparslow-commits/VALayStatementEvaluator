"""UI-level tests: VA.gov appears in the record source selector and selecting
it surfaces the secure login/consent flow (F1.S2 acceptance criteria).
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/ — log_isolation
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from log_isolation import isolate_app_logs  # noqa: E402


class TestVaGovSelector(unittest.TestCase):
    def setUp(self) -> None:
        # AppTest runs the real app in-process — keep its run-log/audit-log writes
        # out of the developer's real logs/ (see tests/log_isolation.py).
        self._tmpdir = isolate_app_logs(self)

    def _app(self):
        from streamlit.testing.v1 import AppTest

        return AppTest.from_file(str(PROJECT_ROOT / "run_app.py"), default_timeout=20)

    def test_va_gov_option_present_in_evaluate_and_draft(self):
        at = self._app()
        at.run()
        eval_options = at.radio(key="records_source_eval").options
        draft_options = at.radio(key="records_source_draft").options
        self.assertIn("VA.gov", eval_options)
        self.assertIn("VA.gov", draft_options)

    def test_selecting_va_gov_shows_login_and_consent(self):
        at = self._app()
        at.run()
        at.radio(key="records_source_eval").set_value("VA.gov")
        at.run()
        checkbox_labels = [c.label for c in at.checkbox]
        self.assertTrue(
            any("consent" in label.lower() for label in checkbox_labels),
            msg=f"expected a VA.gov consent checkbox, got: {checkbox_labels}",
        )
        button_labels = [b.label for b in at.button]
        self.assertTrue(
            any("log in" in label.lower() for label in button_labels),
            msg=f"expected a VA.gov login button, got: {button_labels}",
        )

    def test_login_without_consent_warns_and_does_not_fetch(self):
        at = self._app()
        at.run()
        at.radio(key="records_source_eval").set_value("VA.gov")
        at.run()
        at.text_input(key="va_gov_user_eval").set_value("veteran1")
        at.text_input(key="va_gov_pass_eval").set_value("hunter2")
        at.button(key="va_gov_login_eval").click().run()
        warnings = [w.value for w in at.warning]
        self.assertTrue(any("onsent" in w for w in warnings), msg=warnings)

    def test_login_with_consent_fetches_mock_records_and_requires_confirmation(self):
        at = self._app()
        at.run()
        at.radio(key="records_source_eval").set_value("VA.gov")
        at.run()
        at.checkbox(key="va_gov_consent_eval").set_value(True)
        at.text_input(key="va_gov_user_eval").set_value("veteran1")
        at.text_input(key="va_gov_pass_eval").set_value("hunter2")
        with patch.dict("os.environ", {"VA_GOV_API_BASE_URL": ""}, clear=False):
            at.button(key="va_gov_login_eval").click().run()
        # Records were fetched (mock mode) and a merged-summary confirmation
        # checkbox is now shown before the records are returned to the pipeline.
        self.assertIn("va_gov_confirm_eval", at.session_state)

    def test_confirmation_summary_matches_retained_records_and_sources(self):
        from streamlit.testing.v1 import AppTest

        for same_text in (False, True):
            with self.subTest(same_text=same_text):
                fetched_text = "Patient A: asthma." if same_text else "Patient B: injury."
                at = AppTest.from_string(f'''
import streamlit as st
from app.documents import document_from_text
from app.va_gov_client import VaGovFetchResult
from app.views.records import _va_gov_records

st.session_state.source_records_eval = {{
    "Upload": [document_from_text("records.txt", "Patient A: asthma.")],
}}
st.session_state.va_gov_fetch_eval = VaGovFetchResult(
    documents=[document_from_text("records.txt", {fetched_text!r})],
    retrieved=1, expected=1, partial=False,
)
st.session_state.selected_records = _va_gov_records("eval")
''', default_timeout=20).run()
                self.assertFalse(at.exception)
                self.assertEqual(at.session_state.selected_records, [])
                rows = at.dataframe[0].value.to_dict("records")
                expected_count = 1 if same_text else 2
                self.assertEqual(len(rows), expected_count)
                if same_text:
                    self.assertEqual(rows[0]["Source"], "Upload, VA.gov")
                else:
                    self.assertEqual([row["Source"] for row in rows], ["Upload", "VA.gov"])
                at.checkbox(key="va_gov_confirm_eval").set_value(True).run()
                self.assertFalse(at.exception)
                selected = at.session_state.selected_records
                self.assertEqual(len(selected), expected_count)
                self.assertEqual([row["File"] for row in rows], [doc.filename for doc in selected])
                self.assertIn(fetched_text, [doc.full_text for doc in selected])


if __name__ == "__main__":
    unittest.main()
