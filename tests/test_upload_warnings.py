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


if __name__ == "__main__":
    unittest.main()
