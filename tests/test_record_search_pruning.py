"""UI-level test: replacing a record prunes its search results and citations.

Runs the real app through Streamlit's AppTest harness — real widgets, real
session state, real extraction — rather than a mocked ``st``. The panel's
pruning lives in the interaction between the uploader's reconcile call and the
citation index, and a mocked session cannot show whether the widgets the user
sees end up agreeing with the state that was pruned.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/ — log_isolation
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from log_isolation import isolate_app_logs  # noqa: E402


class TestRecordSearchPruning(unittest.TestCase):
    def setUp(self) -> None:
        # AppTest runs the real app in-process — keep its run-log/audit-log writes
        # out of the developer's real logs/ (see tests/log_isolation.py).
        self._tmpdir = isolate_app_logs(self)

    def _app(self):  # noqa: ANN202
        from streamlit.testing.v1 import AppTest

        return AppTest.from_file(str(PROJECT_ROOT / "run_app.py"), default_timeout=20)

    def _upload(self, at, text: str) -> None:  # noqa: ANN001
        at.file_uploader(key="files_eval").set_value(
            [("visit.txt", text.encode(), "text/plain")]
        )
        at.run()
        self.assertFalse(at.exception)

    def test_replacing_a_record_prunes_its_results_and_citations(self) -> None:
        at = self._app()
        at.run()
        self._upload(at, "Knee pain after lifting in 2014.")
        self.assertEqual(
            [doc.full_text for doc in at.session_state["source_records_eval"]["Upload"]],
            ["Knee pain after lifting in 2014."],
        )

        # Search for the symptom and export the hit into the citation index.
        at.text_input(key="search_query_eval").set_value("knee")
        at.run()
        at.button(key="search_submit_eval").click()
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.session_state["search_results_eval"]), 1)
        at.button(key="search_export_eval_0").click()
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(
            [citation["source"] for citation in at.session_state["citation_index"]],
            ["visit.txt p.1"],
        )
        self.assertIn("Knee", at.session_state["citation_index"][0]["excerpt"])

        # Replace the file's contents under the same name: the uploader's reconcile
        # must drop both the rendered hit and the citation it produced.
        self._upload(at, "Updated note: shoulder impingement only.")
        self.assertEqual(
            [doc.full_text for doc in at.session_state["source_records_eval"]["Upload"]],
            ["Updated note: shoulder impingement only."],
        )
        self.assertEqual(at.session_state["search_results_eval"], [])
        self.assertEqual(at.session_state["citation_index"], [])
        rendered = " ".join(element.value for element in at.markdown)
        self.assertNotIn("Knee pain after lifting", rendered)
        self.assertNotIn("Citation index", rendered)


if __name__ == "__main__":
    unittest.main()
