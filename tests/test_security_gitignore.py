"""Verify .gitignore actually ignores secret-bearing paths and hook is present."""
import fnmatch
import os
import shutil
import stat
import sys
import subprocess
import tempfile
from pathlib import Path

import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GITIGNORE = PROJECT_ROOT / ".gitignore"
HOOK = PROJECT_ROOT / "scripts/hooks/pre-commit"
#: The harness-import rule the hook runs. A throwaway repository needs a copy of
#: it, because the hook resolves the module from the checkout it is committing to.
HARNESS_IMPORTS = PROJECT_ROOT / "tests" / "harness_imports.py"


def _gitignore_patterns() -> list[str]:
    patterns: list[str] = []
    for line in GITIGNORE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def _ignored_by_gitignore(rel_path: str, patterns: list[str]) -> bool:
    """Approximate gitignore matching for the patterns we care about.

    Uses fnmatch across the full relative path and against the basename, so
    both *.env.local and .env.local match at the repo root and in subdirs.
    """
    for pat in patterns:
        # git treats patterns without slash as basename-matched across dirs
        if "/" not in pat.strip("/"):
            if fnmatch.fnmatch(rel_path, pat) or fnmatch.fnmatch(
                os.path.basename(rel_path), pat
            ):
                return True
        else:
            # anchored-like patterns (e.g. .env)
            if fnmatch.fnmatch(rel_path, pat) or fnmatch.fnmatch(
                os.path.basename(rel_path), pat.strip("/")
            ):
                return True
            # fallback: exact rel match
            if rel_path == pat or rel_path == pat.lstrip("/"):
                return True
    return False


class TestGitignoreSecrets(unittest.TestCase):
    def test_env_variants_ignored(self):
        patterns = _gitignore_patterns()
        must_ignore = [
            ".env",
            ".env.local",
            ".env.dev.local",
            ".env.prod.local",
            "subdir/.env.local",
            "subdir/.env.staging.local",
        ]
        for path in must_ignore:
            self.assertTrue(
                _ignored_by_gitignore(path, patterns),
                msg=f"{path} should be ignored by .gitignore patterns {patterns}",
            )

    def test_env_example_not_ignored(self):
        patterns = _gitignore_patterns()
        self.assertFalse(_ignored_by_gitignore(".env.example", patterns))
        self.assertFalse(_ignored_by_gitignore("subdir/.env.example", patterns))

    def test_streamlit_secrets_ignored(self):
        patterns = _gitignore_patterns()
        self.assertTrue(
            _ignored_by_gitignore(".streamlit/secrets.toml", patterns)
            or _ignored_by_gitignore("secrets.toml", patterns)
        )

    def test_actual_git_check_ignore(self):
        """Ground truth: ask real git check-ignore for the paths that matter."""
        cases_should_ignore = [".env", ".env.local", ".env.dev.local"]
        for path in cases_should_ignore:
            result = subprocess.run(
                ["git", "check-ignore", "--quiet", path],
                cwd=str(PROJECT_ROOT),
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"git check-ignore should ignore {path} but didn't",
            )
        # .env.example must NOT be ignored
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", ".env.example"],
            cwd=str(PROJECT_ROOT),
        )
        self.assertNotEqual(
            result.returncode,
            0,
            msg=".env.example must not be ignored",
        )


class TestPreCommitHook(unittest.TestCase):
    def test_hook_exists_and_executable(self):
        self.assertTrue(HOOK.exists(), msg="scripts/hooks/pre-commit should exist")
        mode = HOOK.stat().st_mode
        self.assertTrue(mode & stat.S_IXUSR, msg="hook should be executable")

    def test_hook_blocks_env_file(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True)
            # Simulate the hook's file-pattern check by running it with staged files.
            # Use --no-verify to not trigger other repo's hooks.
            # Patch git diff via env: simplest is to stage a .env and run the hook directly.
            (repo / ".env").write_text("OPENAI_API_KEY=sk-sp-fake\n")
            subprocess.run(["git", "add", ".env"], cwd=str(repo), check=True)
            # Copy real hook into temp repo
            hook_src = HOOK.read_text()
            hook_dest = repo / "hook.sh"
            hook_dest.write_text(hook_src)
            hook_dest.chmod(0o755)
            # Run hook with git context pointed at temp repo
            result = subprocess.run(
                ["bash", str(hook_dest)],
                cwd=str(repo),
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(".env", result.stderr + result.stdout)

    def test_hook_blocks_sk_prefix_in_diff(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True)
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                cwd=str(repo),
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=str(repo),
                check=True,
                capture_output=True,
            )
            (repo / "README.md").write_text("hello\n")
            subprocess.run(["git", "add", "README.md"], cwd=str(repo), check=True)
            subprocess.run(
                ["git", "commit", "-m", "init", "--allow-empty"],
                cwd=str(repo),
                check=True,
                capture_output=True,
            )
            (repo / "README.md").write_text("OPENAI_API_KEY=sk-sp-abc123\n")
            subprocess.run(["git", "add", "README.md"], cwd=str(repo), check=True)
            hook_dest = repo / "hook.sh"
            hook_dest.write_text(HOOK.read_text())
            hook_dest.chmod(0o755)
            result = subprocess.run(
                ["bash", str(hook_dest)],
                cwd=str(repo),
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)

    def test_hook_allows_env_example(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True)
            (repo / ".env.example").write_text("OPENAI_API_KEY=\n")
            subprocess.run(["git", "add", ".env.example"], cwd=str(repo), check=True)
            hook_dest = repo / "hook.sh"
            hook_dest.write_text(HOOK.read_text())
            hook_dest.chmod(0o755)
            result = subprocess.run(
                ["bash", str(hook_dest)],
                cwd=str(repo),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0)


class TestTheHookEnforcesTheHarnessImport(unittest.TestCase):
    """The harness rule, at the moment the mistake is made rather than after a push.

    ``tests/test_job_queue_atomic.py`` reached main without the harness import,
    and the only thing that noticed was CI — on a *merge result*, one module late,
    in a build nobody could reproduce locally. The scan in
    ``tests/test_hermetic.py`` cannot see a module whose branch is not merged; the
    commit hook can, because it looks at what is being committed. Both call
    ``tests/harness_imports.py``, so they cannot disagree about what is required.

    These run the real hook in a throwaway repository, so they exercise the path a
    contributor's commit takes: the hook's interpreter choice, its staged-file
    read, and its exit status.
    """

    WIRED = (
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))\n"
        "from tests import hermetic  # noqa: E402,F401\n"
        "\n"
        "from app import config  # noqa: E402\n"
    )
    FORGOT = "import unittest\n\n\nclass T(unittest.TestCase):\n    pass\n"
    APP_FIRST = (
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))\n"
        "from app import config  # noqa: E402\n"
        "from tests import hermetic  # noqa: E402,F401\n"
    )

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name)
        subprocess.run(["git", "init", "-q"], cwd=str(self.repo), check=True)
        (self.repo / "tests").mkdir()
        (self.repo / "tests" / "harness_imports.py").write_text(
            HARNESS_IMPORTS.read_text(encoding="utf-8"), encoding="utf-8"
        )
        self.hook = self.repo / "hook.sh"
        self.hook.write_text(HOOK.read_text(encoding="utf-8"), encoding="utf-8")
        self.hook.chmod(0o755)

    def stage(self, relative: str, text: str) -> None:
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        subprocess.run(
            ["git", "add", relative],
            cwd=str(self.repo),
            check=True,
            capture_output=True,
        )

    def run_hook(self, env: dict | None = None) -> subprocess.CompletedProcess:
        # bash by absolute path: with ``env`` replaced, subprocess resolves the
        # executable against *that* PATH, and the fail-closed test deliberately
        # hands it a PATH with nothing but the tools the hook itself needs.
        bash = shutil.which("bash") or "bash"
        return subprocess.run(
            [bash, str(self.hook)],
            cwd=str(self.repo),
            capture_output=True,
            text=True,
            env=env,
        )

    def test_the_hook_runs_the_shared_rule(self) -> None:
        """A gate wired to nothing is a gate that cannot fail."""
        self.assertTrue(
            HARNESS_IMPORTS.exists(),
            "the rule the hook and the suite share is missing, so the hook either "
            "does nothing or carries a second copy that can drift from CI's",
        )
        hook = HOOK.read_text(encoding="utf-8")
        self.assertIn("tests.harness_imports", hook)
        self.assertIn("--staged", hook)

    def test_the_hook_blocks_a_staged_module_that_forgets(self) -> None:
        self.stage("tests/test_forgot.py", self.FORGOT)
        result = self.run_hook()
        self.assertNotEqual(result.returncode, 0, "the commit should have been refused")
        self.assertIn("test_forgot.py", result.stderr)
        self.assertIn("does not import the hermetic harness", result.stderr)
        self.assertIn("from tests import hermetic", result.stderr, "say how to fix it")

    def test_the_hook_blocks_a_module_that_imports_the_app_first(self) -> None:
        """The ordering half: importing the harness *after* the app is worse than not."""
        self.stage("tests/test_late.py", self.APP_FIRST)
        result = self.run_hook()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("test_late.py", result.stderr)
        self.assertIn("lands after the import on line", result.stderr)

    def test_the_hook_allows_a_wired_module(self) -> None:
        self.stage("tests/test_wired.py", self.WIRED)
        result = self.run_hook()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_the_hook_ignores_modules_that_are_not_tests(self) -> None:
        """Scope is ``tests/test_*.py``, the same glob the suite scans."""
        self.stage("tests/helper.py", self.FORGOT)
        self.stage("app/thing.py", self.FORGOT)
        result = self.run_hook()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_the_hook_judges_the_staged_copy_not_the_working_tree(self) -> None:
        """A commit is what is staged; a file edited afterwards is not in it.

        Both directions matter. A working tree broken after staging must not block
        the commit, and a working tree *fixed* after staging must not let a broken
        staged module through — that is the mistake the check exists to catch.
        """
        self.stage("tests/test_staged.py", self.WIRED)
        (self.repo / "tests" / "test_staged.py").write_text(
            self.FORGOT, encoding="utf-8"
        )
        self.assertEqual(
            self.run_hook().returncode, 0, "the staged copy is the wired one"
        )

        self.stage("tests/test_staged.py", self.FORGOT)
        (self.repo / "tests" / "test_staged.py").write_text(
            self.WIRED, encoding="utf-8"
        )
        broken = self.run_hook()
        self.assertNotEqual(
            broken.returncode,
            0,
            "the staged copy is unwired, so fixing the working tree cannot help",
        )
        self.assertIn("test_staged.py", broken.stderr)

    def test_the_check_fails_closed_when_no_interpreter_exists(self) -> None:
        """Not skipped, blocked — the one outcome this hook must never produce.

        PATH is reduced to wrappers for the few tools the hook itself needs: no
        python of any kind, and no ``.venv`` in this throwaway repository, which is
        the state of a machine before anything is installed.
        """
        binaries = self.repo / "bin"
        binaries.mkdir()
        for tool in ("git", "grep", "basename"):
            real = shutil.which(tool)
            self.assertIsNotNone(real, f"{tool} is needed to run the hook at all")
            wrapper = binaries / tool
            wrapper.write_text(f'#!/bin/sh\nexec "{real}" "$@"\n', encoding="utf-8")
            wrapper.chmod(0o755)
        self.stage("tests/test_wired.py", self.WIRED)
        env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
        env["PATH"] = str(binaries)
        result = self.run_hook(env)
        self.assertNotEqual(result.returncode, 0, "a check that cannot run must not pass")
        self.assertIn("no python3 on PATH", result.stderr)


if __name__ == "__main__":
    unittest.main()
