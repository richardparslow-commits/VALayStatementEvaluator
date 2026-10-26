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

    def _search_and_export(self, at, query: str, position: int) -> None:  # noqa: ANN001
        at.text_input(key="search_query_eval").set_value(query)
        at.run()
        at.button(key="search_submit_eval").click()
        at.run()
        self.assertFalse(at.exception)
        at.button(key=f"search_export_eval_{position}").click()
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
        self._search_and_export(at, "knee", 0)
        self.assertEqual(len(at.session_state["search_results_eval"]), 1)
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

    def test_replacing_an_upload_keeps_a_va_gov_citation(self) -> None:
        """A VA.gov citation must survive a replacement in the upload source.

        A VA.gov fetch needs the veteran's own login, so the test seeds what that
        successful fetch leaves behind — the stored records, the fetch marker that
        keeps the uploader from forgetting the source, and the citation the panel
        exported for it earlier in the session. From there every widget and every
        state change is the app's own.
        """
        from app.documents import document_from_text

        va_doc = document_from_text(
            "va_problem_list.txt", "VA problem list: chronic lumbar strain noted."
        )
        va_citation = {
            "source": "va_problem_list.txt p.1",
            "excerpt": "VA problem list: chronic lumbar strain noted.",
        }
        at = self._app()
        at.run()
        at.session_state["source_records_eval"] = {"VA.gov": [va_doc]}
        at.session_state["va_gov_fetch_eval"] = True
        at.session_state["citation_index"] = [dict(va_citation)]

        self._upload(at, "Knee pain after lifting in 2014.")
        self.assertEqual(at.session_state["citation_index"], [va_citation])

        self._search_and_export(at, "knee", 0)
        self.assertEqual(
            [citation["source"] for citation in at.session_state["citation_index"]],
            ["va_problem_list.txt p.1", "visit.txt p.1"],
        )

        # The upload's citation is stale now, the VA.gov one is not: reconcile must
        # keep the source the user still has loaded.
        self._upload(at, "Updated note: shoulder impingement only.")
        self.assertEqual(at.session_state["citation_index"], [va_citation])
        self.assertEqual(at.session_state["search_results_eval"], [])


if __name__ == "__main__":
    unittest.main()
