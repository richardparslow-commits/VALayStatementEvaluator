"""Verify .gitignore actually ignores secret-bearing paths and hook is present."""
import fnmatch
import os
import stat
import sys
import subprocess
import tempfile
from pathlib import Path

import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GITIGNORE = PROJECT_ROOT / ".gitignore"
HOOK = PROJECT_ROOT / "scripts/hooks/pre-commit"


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


if __name__ == "__main__":
    unittest.main()
