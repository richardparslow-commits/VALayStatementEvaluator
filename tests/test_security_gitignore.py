"""Verify .gitignore actually ignores secret-bearing paths and hook is present."""
import fnmatch
import os
import re
import shutil
import stat
import sys
import subprocess
import tempfile
from pathlib import Path

import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)
from tests import harness_imports, streamlit_option_reads  # noqa: E402

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


class TestTheHookNoticesAStaleInstall(HookRepository):
    """A copy of the hook checks less than it claims, and only its bytes can say so.

    The rules resolve from the checkout, but the hook around them is whichever file
    was installed: a copy frozen the day it was made, or — since every worktree
    shares the main checkout's ``.git/hooks`` — a link into another checkout. No
    rule can notice that, so the hook compares its own bytes with the checkout's
    ``scripts/hooks/pre-commit`` and refuses a mismatch. A checkout without that
    file (a branch from before the hook) has nothing to compare, and is not judged.
    """

    CANONICAL = "scripts/hooks/pre-commit"

    def install_canonical(self, text: str) -> None:
        path = self.repo / self.CANONICAL
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)

    def run_installed(self, relative: str) -> subprocess.CompletedProcess:
        """Run an installed path the way git would: bash <path>, from the root.

        Git invokes a hook by its path with the working directory at the checkout
        root — what lets a relative install (``core.hooksPath``) resolve, and what
        the hook's comparison of its own bytes relies on.
        """
        bash = shutil.which("bash") or "bash"
        return subprocess.run(
            [bash, relative],
            cwd=str(self.repo),
            capture_output=True,
            text=True,
        )

    def test_a_copy_that_does_not_match_is_refused(self) -> None:
        """The running bytes and the checkout's disagree: whichever side is behind,
        that side is checking less, and nothing in a commit can show which it is."""
        self.install_canonical("#!/usr/bin/env bash\n# an older hook\nexit 0\n")
        result = self.run_hook()
        self.assertNotEqual(result.returncode, 0, "the commit should have been refused")
        self.assertIn("not this checkout's hook", result.stderr)
        self.assertIn(self.CANONICAL, result.stderr, "name the file it should be")
        self.assertIn("core.hooksPath", result.stderr, "say how to install it")

    def test_the_checkouts_own_hook_passes(self) -> None:
        """The ``core.hooksPath`` install: git runs the checkout's file itself."""
        self.install_canonical(HOOK.read_text(encoding="utf-8"))
        result = self.run_installed(self.CANONICAL)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_a_link_to_the_checkouts_own_hook_passes(self) -> None:
        """The link install: a different path, the same bytes."""
        self.install_canonical(HOOK.read_text(encoding="utf-8"))
        link = self.repo / ".git" / "hooks" / "pre-commit"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(Path("../../scripts/hooks/pre-commit"))
        result = self.run_installed(".git/hooks/pre-commit")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_a_checkout_without_the_hook_file_is_not_refused(self) -> None:
        """A branch from before the hook has nothing to compare against.

        The comparison cannot be made there, and a gate must not invent a
        mismatch — the suite's throwaway repositories and older branches both run
        without the file.
        """
        result = self.run_hook()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("checkout's hook", result.stderr)


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


# Fixture keys assembled at runtime rather than written as one literal,
# deliberately: ``scripts/hooks/pre-commit`` blocks any added line containing an
# ``sk-`` token, so a secret-shaped literal would make this file uncommittable
# for anyone who has the hook installed. The secret rules are tested against the
# value, and the value is exactly provider-shaped.
_PROVIDER_KEY = "sk-" + "sp-abcdefgh12345678"
_KEY_ASSIGNMENT = "OPENAI_API_KEY=" + _PROVIDER_KEY
_EXTRA_KEY_ASSIGNMENT = "EXTRA_KEY=" + _PROVIDER_KEY


class TestTheHookRefusesASecretUnderARename(HookRepository):
    """A renamed path is a staged path: ``git mv notes.txt .env`` commits a .env.

    The name patterns and the content scan both read ``--diff-filter=ACMR``, so a
    rename arrives under the name it lands on. With ``ACM`` the destination was
    invisible to every check below the gate's name list, and the commit was
    accepted while ``SECURITY.md`` promises a staged ``.env`` or ``*.pem`` is
    refused.
    """

    def commit(self, relative: str, text: str) -> None:
        self.stage(relative, text)
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=str(self.repo),
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.invalid"],
            cwd=str(self.repo),
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-qm", f"add {relative}"],
            cwd=str(self.repo),
            check=True,
            capture_output=True,
        )

    def rename(self, source: str, destination: str) -> None:
        subprocess.run(
            ["git", "mv", source, destination],
            cwd=str(self.repo),
            check=True,
            capture_output=True,
        )

    def test_a_renamed_env_file_is_refused(self) -> None:
        self.commit("notes.txt", _KEY_ASSIGNMENT + "\n")
        self.rename("notes.txt", ".env")
        result = self.run_hook()
        self.assertNotEqual(result.returncode, 0, "the commit should have been refused")
        self.assertIn(".env", result.stderr)

    def test_a_renamed_pem_file_is_refused(self) -> None:
        self.commit("certificate.txt", "-----BEGIN PRIVATE KEY-----\n")
        self.rename("certificate.txt", "server.pem")
        result = self.run_hook()
        self.assertNotEqual(result.returncode, 0, "the commit should have been refused")
        self.assertIn("server.pem", result.stderr)

    def test_a_rename_that_adds_a_key_is_refused(self) -> None:
        """The content scan follows the rename too, not only the name patterns.

        The body is long enough that git calls this a rename *with changes* rather
        than an unrelated add and delete — the shape where the destination is the
        path a check has to look at.
        """
        body = "".join(f"line {number}\n" for number in range(15))
        self.commit("notes.txt", body)
        self.rename("notes.txt", "renamed.txt")
        (self.repo / "renamed.txt").write_text(
            body + _EXTRA_KEY_ASSIGNMENT + "\n", encoding="utf-8"
        )
        subprocess.run(
            ["git", "add", "renamed.txt"],
            cwd=str(self.repo),
            check=True,
            capture_output=True,
        )
        result = self.run_hook()
        self.assertNotEqual(result.returncode, 0, "the commit should have been refused")
        self.assertIn("sk-", result.stderr)

    def test_an_ordinary_rename_is_allowed(self) -> None:
        self.commit("notes.txt", "hello\n")
        self.rename("notes.txt", "renamed.txt")
        result = self.run_hook()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_a_rename_of_a_file_that_quotes_key_shapes_is_allowed(self) -> None:
        """A moved file is judged on what it *changes*, not on what it already says.

        Docs quote key shapes — ``MIGRATION.md`` and ``.env.example`` both do — and
        a moved file's unchanged lines are not new to the repository. Reading them
        as if they were added refused ordinary renames, which is why the content
        scan keeps each rename's source beside its destination.
        """
        quoted = "Set `" + _KEY_ASSIGNMENT + "` in `.env`.\n"
        self.commit("guide.md", quoted)
        self.rename("guide.md", "moved-guide.md")
        result = self.run_hook()
        self.assertEqual(result.returncode, 0, result.stderr)


class TestTheHookAndTheRulesStayInStep(HookRepository):
    """The hook lists staged files itself, so its filters can drift from the reader's.

    ``tests/staged_sources.py`` owns the decision about which change kinds travel —
    renames included, because a file must not be able to escape a rule by moving —
    and both rule modules read the index through it. The hook cannot import that
    decision for its own name and content scans, so it states the filter a second
    time, and the two have drifted once already: the reader read ``ACMR`` while the
    hook's listing read ``ACM``, and only the reader saw renames. Nothing failed in
    either direction, which is the point — the side left behind keeps reporting
    green while checking less than it claims.
    """

    READER = PROJECT_ROOT / "tests" / "staged_sources.py"
    WIRED = (
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))\n"
        "from tests import hermetic  # noqa: E402,F401\n"
        "\n"
        "from app import config  # noqa: E402\n"
    )
    FILE_READ = (
        "from pathlib import Path\n"
        "\n"
        "CONFIG = Path(__file__).parent.parent / '.streamlit' / 'config.toml'\n"
        "TEXT = CONFIG.read_text(encoding='utf-8')\n"
    )

    def commit(self, relative: str, text: str) -> None:
        self.stage(relative, text)
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=str(self.repo),
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.invalid"],
            cwd=str(self.repo),
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-qm", f"add {relative}"],
            cwd=str(self.repo),
            check=True,
            capture_output=True,
        )

    def rename(self, source: str, destination: str) -> None:
        subprocess.run(
            ["git", "mv", source, destination],
            cwd=str(self.repo),
            check=True,
            capture_output=True,
        )

    def stage_edit(self, relative: str, text: str) -> None:
        (self.repo / relative).write_text(text, encoding="utf-8")
        subprocess.run(
            ["git", "add", relative],
            cwd=str(self.repo),
            check=True,
            capture_output=True,
        )

    def test_the_hook_listing_reads_the_kinds_the_reader_reads(self) -> None:
        """One decision about change kinds, stated twice — so compare the statements.

        The common shapes (an add, a modify) exercise neither filter's edge; a kind
        dropped on one side is exactly the drift this test exists to stop, and it is
        invisible until someone stages that kind of change.
        """
        hook_filters = set(
            re.findall(r"--diff-filter=([A-Z]+)", HOOK.read_text(encoding="utf-8"))
        )
        reader_filters = set(
            re.findall(
                r"--diff-filter=([A-Z]+)", self.READER.read_text(encoding="utf-8")
            )
        )
        self.assertTrue(reader_filters, "the reader states no filter to compare")
        self.assertTrue(hook_filters, "the hook states no filter to compare")
        for stated in sorted(reader_filters):
            self.assertIn(
                stated,
                hook_filters,
                "the hook's own scans must read the same kinds of change as the "
                "shared reader; changed on one side only, that side checks less "
                "without failing anything",
            )
        self.assertIn(
            "R",
            set("".join(reader_filters)),
            "the reader must carry renames, or a rule can be escaped by moving",
        )
        self.assertIn(
            "R",
            set("".join(hook_filters)),
            "the hook must carry renames, or a staged rename is never scanned",
        )

    def test_the_rule_gates_are_the_scopes_the_modules_judge(self) -> None:
        """The hook's gates decide which rules run; the modules decide what they judge.

        The gates are deliberately coarser than the rules — a directory each, not the
        module's own glob — so that *which* files count has one answer, in the
        modules. A gate narrower than its module skips files the rule meant to judge;
        a missing gate skips the rule entirely.
        """
        gates = set(
            re.findall(r"grep -q '\^([^/']+)/'", HOOK.read_text(encoding="utf-8"))
        )
        scopes = {harness_imports.TESTS_DIR.name, streamlit_option_reads.APP_DIR.name}
        self.assertEqual(
            gates,
            scopes,
            "every rule's scope needs its gate, and every gate a rule — what a "
            "commit is checked against is whatever the two agree on",
        )

    def test_a_renamed_test_module_is_still_judged(self) -> None:
        """A rename with a small edit — the shape where a dropped filter goes unseen.

        A pure move of a wired module passes even when a filter has lost ``R``,
        because nothing was added for a rule to object to; the inserted line is what
        the modules are entitled to see.
        """
        filler = "".join(f"VALUE_{number} = {number}\n" for number in range(15))
        self.commit("tests/test_original.py", self.WIRED + filler)
        self.rename("tests/test_original.py", "tests/test_moved.py")
        self.stage_edit(
            "tests/test_moved.py",
            "from app import config\n" + self.WIRED + filler,
        )
        result = self.run_hook()
        self.assertNotEqual(
            result.returncode, 0, "a moved module is still a staged module"
        )
        self.assertIn("test_moved.py", result.stderr)
        self.assertIn("lands after the import on line", result.stderr)

    def test_a_renamed_app_module_is_still_judged(self) -> None:
        filler = "".join(f"VALUE_{number} = {number}\n" for number in range(15))
        self.commit("app/views/original.py", self.FILE_READ + filler)
        self.rename("app/views/original.py", "app/views/moved.py")
        self.stage_edit(
            "app/views/moved.py",
            "import streamlit as st\n"
            "PORT = st.get_option('server.port')\n" + self.FILE_READ + filler,
        )
        result = self.run_hook()
        self.assertNotEqual(
            result.returncode, 0, "a moved module is still a staged module"
        )
        self.assertIn("app/views/moved.py", result.stderr)
        self.assertIn("get_option", result.stderr)


if __name__ == "__main__":
    unittest.main()
