"""UI-level tests: record pruning and retention across both workflow slots.

Runs the real app through Streamlit's AppTest harness — real widgets, real
session state, real extraction — rather than a mocked ``st``. Pruning lives in
the interaction between the uploader's reconcile call and the citation index,
and retention depends on whether the uploader is still mounted on a run; a
mocked session can show neither, nor whether the widgets the user sees agree
with the state that was pruned. Evaluate and Draft are both covered: they share
``records_uploader`` but keep their state under different slot keys.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/ — log_isolation
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from log_isolation import isolate_app_logs  # noqa: E402

# A VA.gov "Download your medical records" export: VA.gov markers plus a
# records context, which is what the uploader's source detection keys on.
_VA_GOV_EXPORT = (
    "Download your medical records\nva.gov | My HealtheVet\n"
    "Facility: VA Medical Center\nProvider: Smith, John MD\n"
)


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

    def _switch_source(self, at, source: str, slot: str = "eval") -> None:  # noqa: ANN001
        at.radio(key=f"records_source_{slot}").set_value(source)
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

    def test_switching_source_away_and_back_keeps_records_and_citations(self) -> None:
        """Looking at another source must not unload the files already uploaded.

        Streamlit drops the value of a widget it stops rendering, so an uploader
        that is only built on one branch of the source radio comes back empty:
        the records, their search hits and their citations all vanish without the
        user touching the uploader.
        """
        at = self._app()
        at.run()
        self._upload(at, "Knee pain after lifting in 2014.")
        self._search_and_export(at, "knee", 0)

        self._switch_source(at, "VA.gov")
        self.assertEqual(
            [doc.full_text for doc in at.session_state["source_records_eval"]["Upload"]],
            ["Knee pain after lifting in 2014."],
        )
        self.assertEqual(
            [citation["source"] for citation in at.session_state["citation_index"]],
            ["visit.txt p.1"],
        )

        self._switch_source(at, "Upload files")
        self.assertEqual(
            [doc.full_text for doc in at.session_state["source_records_eval"]["Upload"]],
            ["Knee pain after lifting in 2014."],
        )
        self.assertEqual(len(at.session_state["search_results_eval"]), 1)
        self.assertEqual(
            [citation["source"] for citation in at.session_state["citation_index"]],
            ["visit.txt p.1"],
        )
        self.assertIn("Knee", at.session_state["citation_index"][0]["excerpt"])

    def test_switching_source_keeps_a_va_gov_export_grouping(self) -> None:
        """An uploaded VA.gov export stays labelled after a source detour."""
        at = self._app()
        at.run()
        at.file_uploader(key="files_eval").set_value(
            [("va_records.txt", _VA_GOV_EXPORT.encode(), "text/plain")]
        )
        at.run()
        self.assertIn("VA.gov", at.session_state["source_records_eval"])

        self._switch_source(at, "VA.gov")
        self._switch_source(at, "Upload files")

        store = at.session_state["source_records_eval"]
        self.assertEqual([doc.filename for doc in store["VA.gov"]], ["va_records.txt"])
        self.assertEqual([doc.filename for doc in store["Upload"]], ["va_records.txt"])

    def test_clearing_the_kept_uploads_panel_forgets_the_records(self) -> None:
        """Removing the files from the kept panel still clears what they produced."""
        at = self._app()
        at.run()
        self._upload(at, "Knee pain after lifting in 2014.")
        self._search_and_export(at, "knee", 0)
        self._switch_source(at, "VA.gov")
        kept = [block for block in at.expander if "kept" in (block.label or "")]
        self.assertTrue(kept, msg=[block.label for block in at.expander])

        at.file_uploader(key="files_eval").set_value([])
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["source_records_eval"], {})
        self.assertEqual(at.session_state["search_results_eval"], [])
        self.assertEqual(at.session_state["citation_index"], [])

    def test_switching_source_away_and_back_keeps_draft_uploads(self) -> None:
        """The Draft tab uses the same uploader, so the same retention holds there.

        Draft has no search panel, so the record store and the rendered "Loaded"
        summary are what proves the files survived; the VA.gov label is checked
        because it is derived from the upload itself.
        """
        at = self._app()
        at.run()
        at.file_uploader(key="files_draft").set_value(
            [("va_records.txt", _VA_GOV_EXPORT.encode(), "text/plain")]
        )
        at.run()
        self.assertEqual(
            [
                doc.filename
                for doc in at.session_state["source_records_draft"]["VA.gov"]
            ],
            ["va_records.txt"],
        )

        self._switch_source(at, "VA.gov", "draft")
        kept = [block for block in at.expander if "kept" in (block.label or "")]
        self.assertTrue(kept, msg=[block.label for block in at.expander])
        self.assertEqual(
            [
                doc.filename
                for doc in at.session_state["source_records_draft"]["VA.gov"]
            ],
            ["va_records.txt"],
        )

        self._switch_source(at, "Upload files", "draft")
        store = at.session_state["source_records_draft"]
        self.assertEqual([doc.filename for doc in store["Upload"]], ["va_records.txt"])
        self.assertEqual([doc.filename for doc in store["VA.gov"]], ["va_records.txt"])
        successes = " ".join(message.value for message in at.success)
        self.assertIn("Loaded 1 record file(s)", successes)

    def test_clearing_the_draft_kept_panel_forgets_its_records(self) -> None:
        """Removing the draft's files from the kept panel clears the draft store."""
        at = self._app()
        at.run()
        at.file_uploader(key="files_draft").set_value(
            [("draft_visit.txt", b"Knee pain after lifting in 2014.", "text/plain")]
        )
        at.run()
        self.assertIn("Upload", at.session_state["source_records_draft"])

        self._switch_source(at, "VA.gov", "draft")
        at.file_uploader(key="files_draft").set_value([])
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["source_records_draft"], {})


if __name__ == "__main__":
    unittest.main()
