"""UI-level tests: unreadable uploads surface per-file warnings in the app.

Uses Streamlit's AppTest harness to run the real app and drive the file
uploader, so the warning behavior is locked in end-to-end (not just in the
pure extraction helper).
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@unittest.skipUnless(
    (PROJECT_ROOT / ".venv").exists() or True,  # venv optional; deps needed either way
    "requires streamlit AppTest",
)
class TestUploadWarnings(unittest.TestCase):
    def _app(self):
        from streamlit.testing.v1 import AppTest

        return AppTest.from_file(str(PROJECT_ROOT / "run_app.py"), default_timeout=20)

    def test_bad_pdf_upload_surfaces_warning(self):
        at = self._app()
        at.run()
        at.file_uploader(key="files_eval").set_value(
            [
                ("good.txt", b"Knee pain noted during visit.", "text/plain"),
                ("bad.pdf", b"%PDF-1.4 broken scan data", "application/pdf"),
            ]
        )
        at.run()

        messages = [w.value for w in at.warning]
        self.assertTrue(
            any("bad.pdf" in m and "could not read PDF" in m for m in messages),
            msg=f"expected a bad.pdf warning, got: {messages}",
        )
        # The good file must still have loaded.
        successes = [s.value for s in at.success]
        self.assertTrue(
            any("good.txt" in s for s in successes),
            msg=f"expected good.txt to load, got: {successes}",
        )

    def test_warning_persists_across_reruns(self):
        at = self._app()
        at.run()
        at.file_uploader(key="files_eval").set_value(
            [("bad.pdf", b"%PDF-1.4 broken scan data", "application/pdf")]
        )
        at.run()
        at.run()  # unrelated rerun (e.g. another widget interaction)
        at.radio(key="eval_mode").set_value("Paste text")
        at.run()
        messages = [w.value for w in at.warning]
        self.assertTrue(any("bad.pdf" in m for m in messages), msg=messages)

    def test_warning_clears_when_bad_file_replaced(self):
        """Warnings are recomputed per run, so removing the bad file clears them."""
        at = self._app()
        at.run()
        at.file_uploader(key="files_eval").set_value(
            [("bad.pdf", b"%PDF-1.4 broken scan data", "application/pdf")]
        )
        at.run()
        self.assertTrue(any("bad.pdf" in w.value for w in at.warning))
        at.file_uploader(key="files_eval").set_value(
            [("good.txt", b"Knee pain noted during visit.", "text/plain")]
        )
        at.run()
        self.assertFalse(
            any("bad.pdf" in w.value for w in at.warning),
            msg=f"stale warning after replacing file: {[w.value for w in at.warning]}",
        )

    def test_bad_statement_upload_warns_in_step_1(self):
        """Unreadable statement files warn in the Step 1 uploader too."""
        at = self._app()
        at.run()
        at.file_uploader(key="eval_statement_file").set_value(
            ("bad_stmt.pdf", b"%PDF-1.4 broken scan data", "application/pdf")
        )
        at.run()
        messages = [w.value for w in at.warning]
        self.assertTrue(
            any("bad_stmt.pdf" in m and "could not read PDF" in m for m in messages),
            msg=f"expected a bad_stmt.pdf warning, got: {messages}",
        )
        # Replacing it with a readable statement clears the warning.
        at.file_uploader(key="eval_statement_file").set_value(
            ("good_stmt.txt", b"I observed the veteran limping.", "text/plain")
        )
        at.run()
        self.assertFalse(
            any("bad_stmt.pdf" in w.value for w in at.warning),
            msg=f"stale statement warning: {[w.value for w in at.warning]}",
        )


if __name__ == "__main__":
    unittest.main()
