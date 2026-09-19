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
#: Every module the hook runs — each rule, plus the plumbing they share. A
#: throwaway repository needs copies of all of them, because the hook resolves each
#: one from the checkout it is committing to.
RULE_MODULES = (
    PROJECT_ROOT / "tests" / "staged_sources.py",
    PROJECT_ROOT / "tests" / "harness_imports.py",
    PROJECT_ROOT / "tests" / "streamlit_option_reads.py",
)


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

    def test_the_header_installs_it_in_a_way_that_does_not_go_stale(self):
        """A copy of the hook checks less than the hook, and reports green.

        Every rule below lives in a module the hook resolves from the checkout it
        is committing to, and this file is edited whenever one is added — so a
        checkout that holds a *copy* keeps running the version it was copied from,
        silently enforcing the rules of the day it was installed. ``README.md``
        and ``SECURITY.md`` therefore say to link it or to point
        ``core.hooksPath`` at ``scripts/hooks``; the header a contributor reads
        when their commit is refused has to say the same thing, because that is
        the moment they (re)install it.
        """
        header = HOOK.read_text(encoding="utf-8")
        self.assertIn(
            "core.hooksPath",
            header,
            "the header should name the install that keeps the hook current",
        )
        self.assertNotIn(
            "cp scripts/hooks/pre-commit",
            header,
            "the header must not recommend a copy, which goes stale",
        )

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


class HookRepository(unittest.TestCase):
    """A throwaway repository holding the real hook and a copy of every rule.

    Both rule classes below run the *actual* hook file from such a tree, with no
    venv and nothing installed, so what they exercise is the path a contributor's
    commit takes: the interpreter choice, the staged-file read, and the exit
    status.
    """

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name)
        subprocess.run(["git", "init", "-q"], cwd=str(self.repo), check=True)
        rules = self.repo / "tests"
        rules.mkdir()
        for module in RULE_MODULES:
            (rules / module.name).write_text(
                module.read_text(encoding="utf-8"), encoding="utf-8"
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


class TestTheHookReadsOddStagedPathsAsTheyAre(HookRepository):
    """A path is one path, and a fixture is not a source file.

    Both rules read the staged listing through ``tests/staged_sources.py``, whose
    names used to be split on whitespace: a staged ``tests/test_ two words.py``
    became several paths, the lookups for the others failed, and the hook refused
    a commit that broke no rule. Content used to be decoded strictly too, so a
    binary fixture under ``tests/`` or ``app/`` raised through the rule instead of
    being filtered out by the name rule that never meant to read it.
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

    def stage_bytes(self, relative: str, payload: bytes) -> None:
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        subprocess.run(
            ["git", "add", relative],
            cwd=str(self.repo),
            check=True,
            capture_output=True,
        )

    def test_the_hook_allows_a_wired_module_whose_name_has_a_space(self) -> None:
        self.stage("tests/test_wired with spaces.py", self.WIRED)
        result = self.run_hook()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_the_hook_refuses_an_unwired_module_whose_name_has_a_space(self) -> None:
        self.stage("tests/test_unwired with spaces.py", self.FORGOT)
        result = self.run_hook()
        self.assertNotEqual(result.returncode, 0, "the commit should have been refused")
        self.assertIn("test_unwired with spaces.py", result.stderr)
        self.assertIn("does not import the hermetic harness", result.stderr)
        self.assertNotIn("failed:", result.stderr, "the rule spoke, not a failed lookup")

    def test_a_binary_fixture_under_tests_is_not_a_source_file(self) -> None:
        self.stage_bytes("tests/fixtures/blob.bin", b"\x00\x01\xff\xfe\x89PNG")
        result = self.run_hook()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_a_binary_fixture_under_app_is_not_a_source_file(self) -> None:
        self.stage_bytes("app/static/blob.bin", b"\x00\x01\xff\xfe\x89PNG")
        result = self.run_hook()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Traceback", result.stderr)


class TestTheHookEnforcesTheHarnessImport(HookRepository):
    """The harness rule, at the moment the mistake is made rather than after a push.

    ``tests/test_job_queue_atomic.py`` reached main without the harness import,
    and the only thing that noticed was CI — on a *merge result*, one module late,
    in a build nobody could reproduce locally. The scan in
    ``tests/test_hermetic.py`` cannot see a module whose branch is not merged; the
    commit hook can, because it looks at what is being committed. Both call
    ``tests/harness_imports.py``, so they cannot disagree about what is required.
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

    def test_the_hook_runs_the_shared_rule(self) -> None:
        """A gate wired to nothing is a gate that cannot fail."""
        for module in RULE_MODULES:
            self.assertTrue(
                module.exists(),
                f"{module.name} is missing, so its rule either does nothing or "
                "carries a second copy that can drift from CI's",
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


class TestTheHookRefusesAStreamlitOptionRead(HookRepository):
    """The option rule, at the moment the mistake is made rather than after a push.

    ``app/`` must decide nothing from a Streamlit option's *value*: it comes from
    the machine and from the directory the process started in, so the same code
    behaves differently in two deployments — and for the hardening check in
    ``app/main.py`` it would answer the wrong question ("is XSRF on here?") instead
    of the one that matters ("does this deployment ship it?"). The scan in
    ``tests/test_hermetic.py`` cannot see an unmerged branch; the hook can, because
    it looks at what is being committed. Both call
    ``tests/streamlit_option_reads.py``, so they cannot disagree about the rule.
    """

    OPTION_READ = (
        "import streamlit as st\n"
        "\n"
        "\n"
        "def port() -> int:\n"
        "    return st.get_option('server.port')\n"
    )
    FILE_READ = (
        "from pathlib import Path\n"
        "\n"
        "CONFIG = Path(__file__).parent.parent / '.streamlit' / 'config.toml'\n"
        "TEXT = CONFIG.read_text(encoding='utf-8')\n"
    )

    def test_the_hook_runs_the_shared_rule(self) -> None:
        hook = HOOK.read_text(encoding="utf-8")
        self.assertIn("tests.streamlit_option_reads", hook)
        self.assertIn("--staged", hook)

    def test_the_hook_blocks_a_staged_app_module_that_reads_an_option(self) -> None:
        self.stage("app/views/thing.py", self.OPTION_READ)
        result = self.run_hook()
        self.assertNotEqual(result.returncode, 0, "the commit should have been refused")
        self.assertIn("app/views/thing.py", result.stderr)
        self.assertIn("get_option", result.stderr, "name the offending read")
        self.assertIn("config.toml", result.stderr, "say what to do instead")

    def test_the_hook_allows_reading_the_committed_file(self) -> None:
        """The *intended* shape: the deployment's own file, read as text."""
        self.stage("app/main.py", self.FILE_READ)
        result = self.run_hook()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_the_hook_ignores_modules_outside_the_app(self) -> None:
        """Scope is ``app/``, because ``tests/hermetic.py`` reads streamlit.config
        on purpose — it has to reach into internals no public API exposes."""
        self.stage("scripts/probe.py", self.OPTION_READ)
        self.stage("harness_probe.py", self.OPTION_READ)
        result = self.run_hook()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_both_rules_report_in_one_commit(self) -> None:
        """Neither rule may mask the other, in either direction.

        The rules are separate modules and the hook runs each in its own process,
        so a refusal from the first must not stop the second from reporting: the
        failure a contributor sees should list everything wrong with the commit.
        """
        self.stage("tests/test_forgot.py", "import unittest\n")
        self.stage("app/thing.py", self.OPTION_READ)
        result = self.run_hook()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not import the hermetic harness", result.stderr)
        self.assertIn("get_option", result.stderr)


if __name__ == "__main__":
    unittest.main()
