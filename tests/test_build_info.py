"""The app can say which commit it is running — staleness must be visible.

Twice in one week a merged feature (the Agent API pivot, then the witness-
credentials step) was reported missing from the deployed app; both times the
code had been on ``main`` for hours and the deployment was simply stale. No
test can force a platform to redeploy, but the app can stop hiding its own
identity: ``app/build_info.py`` resolves a short SHA, and the About tab shows
it (or says honestly that it cannot know).

These tests pin both halves: the resolver's precedence (explicit environment
variable over git, over "unknown") and the About line's two renderings — a
known build stating its SHA, and an unknown build telling the reader what to
do about it (redeploy from ``main``) instead of showing nothing.
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app import build_info  # noqa: E402
from app.views import about_view  # noqa: E402


def _fake_streamlit() -> MagicMock:
    """A MagicMock ``st`` whose session_state mirrors Streamlit's dict."""
    st = MagicMock()

    class _Session(dict):
        def __getattr__(self, name: str):
            try:
                return self[name]
            except KeyError as exc:  # mirror attribute semantics
                raise AttributeError(name) from exc

        def __setattr__(self, name: str, value) -> None:
            self[name] = value

    st.session_state = _Session()
    return st


class TestBuildShaResolution(unittest.TestCase):
    """Precedence: explicit env var, then the git checkout, then unknown."""

    def tearDown(self) -> None:
        os.environ.pop(build_info.BUILD_SHA_ENV, None)

    def test_an_injected_sha_wins(self) -> None:
        with patch.object(build_info, "_sha_from_git", return_value="9999999"):
            with patch.dict(os.environ, {build_info.BUILD_SHA_ENV: "abc1234def"}):
                self.assertEqual(build_info.build_sha(), "abc1234def")
                self.assertEqual(build_info.build_source(), "environment")

    def test_a_non_sha_env_value_is_ignored_not_displayed(self) -> None:
        """``latest`` pasted into a build arg must not masquerade as a SHA."""
        with patch.object(build_info, "_sha_from_git", return_value=""):
            with patch.dict(os.environ, {build_info.BUILD_SHA_ENV: "latest"}):
                self.assertEqual(build_info.build_sha(), "")
                self.assertEqual(build_info.build_source(), "unknown")

    def test_the_git_checkout_is_used_without_an_env_override(self) -> None:
        fake = subprocess.CompletedProcess([], 0, stdout="abc1234\n", stderr="")
        with patch.object(build_info, "_sha_from_env", return_value=""):
            with patch.object(build_info.subprocess, "run", return_value=fake):
                self.assertEqual(build_info.build_sha(), "abc1234")
                self.assertEqual(build_info.build_source(), "git")

    def test_no_git_and_no_env_is_unknown_not_a_crash(self) -> None:
        """A Docker image has no .git; a locked-down box forbids subprocesses."""
        with patch.object(build_info, "_sha_from_env", return_value=""):
            with patch.object(
                build_info.subprocess,
                "run",
                side_effect=OSError("no git binary"),
            ):
                self.assertEqual(build_info.build_sha(), "")
                self.assertEqual(build_info.build_source(), "unknown")


class TestAboutTabShowsTheBuild(unittest.TestCase):
    """The build line renders where the user actually looks for answers."""

    def setUp(self) -> None:
        self.st = _fake_streamlit()

    def test_a_known_build_states_its_sha_and_origin(self) -> None:
        with patch.object(about_view, "st", self.st):
            with patch.object(about_view, "build_sha", return_value="abc1234def"):
                with patch.object(about_view, "build_source", return_value="git"):
                    about_view._render_build_identity()
        caption = self.st.caption.call_args[0][0]
        self.assertIn("abc1234def", caption)
        self.assertIn("git checkout", caption)

    def test_an_unknown_build_says_so_and_names_the_remedy(self) -> None:
        with patch.object(about_view, "st", self.st):
            with patch.object(about_view, "build_sha", return_value=""):
                with patch.object(about_view, "build_source", return_value="unknown"):
                    about_view._render_build_identity()
        caption = self.st.caption.call_args[0][0]
        self.assertIn("unknown", caption)
        # The remedy is the actual fix for every staleness report so far:
        self.assertIn("redeploy", caption.lower())
        self.assertIn("main", caption)


if __name__ == "__main__":
    unittest.main()
