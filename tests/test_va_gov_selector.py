"""UI-level tests: VA.gov appears in the record source selector and selecting
it surfaces the secure login/consent flow (F1.S2 acceptance criteria).
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestVaGovSelector(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
