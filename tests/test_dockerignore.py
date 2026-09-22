"""The build context, checked in both directions.

`.dockerignore` is the only thing standing between a developer's checkout and a
builder: the context is uploaded, and the `sandbox` target's image can be pushed
to a registry and snapshotted, so a path the ignore file does not exclude is a
path a `COPY` *can* take — today, or after someone adds a directory copy later.
Two failure modes matter and they pull in opposite directions:

* **too little ignored** — a real `.env`, the audit trail, the blob store, an
  exported report, a private key. This is the one that costs something;
* **too much ignored** — a build that fails, or an image that silently lost a
  file a stage asked for. (Docker refuses a missing single-file source; a
  directory copy just ships less, which is worse because it is quiet.)

Both are asserted here, against the Dockerfile's own `COPY` lines *and* against a
simulated `COPY . .`, so what protects the image is the filter rather than
today's instructions. The matcher is self-tested against Docker's documented
semantics first, because a matcher that quietly matches nothing passes the second
class and fails the first, and one that matches everything does the reverse.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from tests import dockerfile  # noqa: E402

PROJECT_ROOT = dockerfile.PROJECT_ROOT


def _ignore() -> dockerfile.IgnoreFile:
    return dockerfile.IgnoreFile.for_repo()


#: Directories that only ever hold run output, secrets or local scratch. The
#: entries are prefixes of a repo-relative path, matched at any depth for the
#: names that appear in subdirectories too (``__pycache__``).
SENSITIVE_PREFIXES = (
    "logs/",
    "blobs/",
    "outputs/",
    ".git/",
    ".streamlit/secrets",
    ".streamlit/config.local",
    # VA.gov scraper outputs: signed-in medical-record pages, screenshots and
    # Chrome profiles holding the VA.gov session (PHI/PII — see the ignore
    # file's scraper section). The artifacts dir defaults to a relative path,
    # so it can genuinely materialize inside the checkout.
    "va_gov_download_artifacts/",
    ".va_lse_debug_chrome/",
    ".va_lse_va_gov_profile/",
)

#: Names that are sensitive wherever they appear.
SENSITIVE_NAMES = (".env", "usage_history.json", "secrets.toml")

#: Directory names that are sensitive as *any* path segment, not just at the
#: root — the VA.gov scraper can be run from any cwd, so its artifacts and
#: Chrome-profile directories can materialize at any depth.
SENSITIVE_DIR_NAMES = (
    "va_gov_download_artifacts",
    ".va_lse_debug_chrome",
    ".va_lse_va_gov_profile",
)

#: Suffixes that carry key material.
SENSITIVE_SUFFIXES = (".pem", ".key")

#: One path per category this guard exists for, so a future edit that keeps the
#: file non-empty but drops a category fails by name instead of by arithmetic.
SENSITIVE_PATH_EXAMPLES = (
    ".env",
    ".env.local",
    ".env.dev.local",
    ".streamlit/secrets.toml",
    ".streamlit/secrets.toml.bak",
    ".streamlit/config.local.toml",
    "deploy/tls/server.pem",
    "deploy/tls/server.key",
    "logs/audit.log",
    "logs/runs.jsonl",
    "logs/nested/2026/app.log",
    "blobs/ab/0123456789abcdef.json",
    "outputs/evaluation-report.docx",
    "usage_history.json",
    ".git/config",
    ".git/objects/ab/cdef",
    "app/__pycache__/main.cpython-312.pyc",
    # VA.gov scraper artifacts, at the root and nested (the tool can be run
    # from any cwd, so the artifacts dir can appear at any depth).
    "va_gov_download_artifacts/C.plist",
    "va_gov_download_artifacts/session-2026-09-21/records.pdf",
    "deploy/notes/va_gov_download_artifacts/screenshots/page.png",
    ".va_lse_debug_chrome/Default/Cookies",
    ".va_lse_va_gov_profile/Default/Preferences",
)

#: Files the stages ask for by name. An ignore file that swallowed one of these
#: ships an image that cannot run the suite, or breaks the build outright.
MUST_SURVIVE = (
    ".env.example",  # the re-include, and the reason `.env.*` may be broad
    ".gitignore",
    ".github/workflows/test.yml",
    "Dockerfile",
    "README.md",
    "docker-compose.yml",
    "pyproject.toml",
    "requirements.lock",
    "requirements-dev.txt",
    "run_app.py",
    "app/main.py",
    "app/health.py",
    ".streamlit/config.toml",
    "tests/test_health.py",
    "tests/dockerfile.py",
    "scripts/scale_sim.py",
    "scripts/va_records_download.py",
    "deploy/monitoring/prometheus.yml",
    "nginx/nginx.conf",
    "examples/sample_lay_statement.txt",
    "examples/sample_medical_records.txt",
)


def is_sensitive(path: str) -> bool:
    """Would this file be a problem if it reached an image or a builder?

    The single definition of that question, used by every check below. The
    placeholder ``.env.example`` is the documented exception: it is a template,
    not a credential, and the sandbox image copies it on purpose.
    """
    if path == ".env.example":
        return False
    parts = path.split("/")
    if parts[-1] == ".env" or parts[-1].startswith(".env."):
        return True
    if parts[-1] in SENSITIVE_NAMES:
        return True
    if parts[-1].endswith(SENSITIVE_SUFFIXES):
        return True
    if "__pycache__" in parts:
        return True
    if any(seg in SENSITIVE_DIR_NAMES for seg in parts[:-1]):
        return True
    for prefix in SENSITIVE_PREFIXES:
        if path == prefix.rstrip("/") or path.startswith(prefix):
            return True
    return False


class TestTheSensitivityRuleItself(unittest.TestCase):
    def test_the_examples_are_judged_sensitive(self) -> None:
        """Ties the rule to the examples, so neither can drift unnoticed."""
        missed = [path for path in SENSITIVE_PATH_EXAMPLES if not is_sensitive(path)]
        self.assertEqual(missed, [], "the rule misses: " + ", ".join(missed))

    def test_ordinary_files_are_not_judged_sensitive(self) -> None:
        for path in MUST_SURVIVE + ("docs/notes.md", "app/config.py"):
            with self.subTest(path=path):
                self.assertFalse(is_sensitive(path))


class TestTheMatcherMatchesDocker(unittest.TestCase):
    """Docker's documented semantics, so the coverage tests below mean something."""

    def test_a_star_does_not_cross_a_separator(self) -> None:
        ignore = dockerfile.IgnoreFile("*.md\n")
        self.assertTrue(ignore.ignored("README.md"))
        self.assertFalse(
            ignore.ignored("docs/README.md"),
            "`.dockerignore` patterns are anchored at the context root, unlike .gitignore",
        )

    def test_a_double_star_crosses_separators(self) -> None:
        ignore = dockerfile.IgnoreFile("**/*.md\n")
        for path in ("README.md", "docs/README.md", "docs/a/b/README.md"):
            with self.subTest(path=path):
                self.assertTrue(ignore.ignored(path))

    def test_question_mark_matches_one_character(self) -> None:
        ignore = dockerfile.IgnoreFile("temp?\n")
        self.assertTrue(ignore.ignored("tempa"))
        self.assertFalse(ignore.ignored("temp"))
        self.assertFalse(ignore.ignored("tempab"))

    def test_character_classes(self) -> None:
        ignore = dockerfile.IgnoreFile("run[0-9].log\n")
        self.assertTrue(ignore.ignored("run7.log"))
        self.assertFalse(ignore.ignored("runx.log"))

    def test_a_directory_pattern_excludes_its_contents(self) -> None:
        ignore = dockerfile.IgnoreFile("logs/\n")
        for path in ("logs", "logs/audit.log", "logs/2026/01/app.log"):
            with self.subTest(path=path):
                self.assertTrue(ignore.ignored(path))

    def test_the_last_matching_line_wins(self) -> None:
        ignore = dockerfile.IgnoreFile("logs/\n!logs/keep.txt\n")
        self.assertTrue(ignore.ignored("logs/audit.log"))
        self.assertFalse(ignore.ignored("logs/keep.txt"))

    def test_comments_and_blank_lines_are_not_patterns(self) -> None:
        ignore = dockerfile.IgnoreFile("# a comment\n\n\nlogs/\n")
        self.assertEqual(len(ignore.patterns), 1)
        self.assertFalse(ignore.ignored("# a comment"))

    def test_a_pattern_it_cannot_translate_is_refused_not_guessed(self) -> None:
        """An unparsed pattern would read as "not ignored" — the wrong default
        for a file whose job is keeping secrets out of an image."""
        for pattern in ("a**b\n", "[unclosed\n", "!\n"):
            with self.subTest(pattern=pattern):
                with self.assertRaises(dockerfile.UnsupportedPattern):
                    dockerfile.IgnoreFile(pattern)


class TestTheRepositorysIgnoreFile(unittest.TestCase):
    def setUp(self) -> None:
        self.ignore = _ignore()

    def test_it_exists_and_has_patterns(self) -> None:
        self.assertTrue(dockerfile.DOCKERIGNORE.exists(), ".dockerignore is missing")
        self.assertGreaterEqual(len(self.ignore.patterns), 10)

    def test_nothing_sensitive_would_be_uploaded(self) -> None:
        uploaded = [path for path in SENSITIVE_PATH_EXAMPLES if not self.ignore.ignored(path)]
        self.assertEqual(
            uploaded,
            [],
            "these would be uploaded to a builder and could reach an image: " + ", ".join(uploaded),
        )

    def test_each_named_category_is_covered(self) -> None:
        for path in (".env", ".streamlit/secrets.toml", "logs/audit.log", "blobs/ab/x.json", "outputs/x.pdf"):
            with self.subTest(path=path):
                self.assertTrue(self.ignore.ignored(path), f"{path} is not excluded")

    def test_the_placeholder_env_file_is_kept(self) -> None:
        self.assertFalse(
            self.ignore.ignored(".env.example"),
            "the sandbox image copies .env.example, so the `!.env.example` "
            "re-include has to come after `.env.*`",
        )

    def test_artifacts_this_checkout_really_has_are_excluded(self) -> None:
        """Ground truth over the patterns rather than over the rules: whatever is
        on disk right now that this guard calls sensitive must not be uploadable."""
        present = [
            path
            for path in (".git", "logs", "blobs", "outputs", "usage_history.json", ".env", ".venv")
            if (PROJECT_ROOT / path).exists()
        ]
        self.assertTrue(present, "no run artifacts exist here — the paths must be wrong")
        leaked = [path for path in present if not self.ignore.ignored(path)]
        self.assertEqual(leaked, [], "present in this checkout and uploadable: " + ", ".join(leaked))


class TestNoStageCanCarrySomethingSensitive(unittest.TestCase):
    def test_no_copy_source_reaches_a_sensitive_path(self) -> None:
        """Every file each COPY *could* take, not just the ones it takes today."""
        ignore = _ignore()
        stages = dockerfile.stages()
        self.assertIn("runtime", stages)
        self.assertIn("sandbox", stages)

        leaks: list[str] = []
        for stage, stage_instructions in stages.items():
            for source in dockerfile.copy_sources(stage_instructions):
                for path in dockerfile.expand(source):  # unfiltered by design
                    if is_sensitive(path) and not ignore.ignored(path):
                        leaks.append(f"{stage}: COPY {source} -> {path}")
        self.assertEqual(
            leaks,
            [],
            "sensitive paths a stage's COPY could reach:\n  " + "\n  ".join(leaks),
        )

    def test_a_whole_context_copy_would_be_safe(self) -> None:
        """What `COPY . .` would upload, minus .git and everything listed above.

        This is the assertion that makes the guard about the filter instead of
        about today's instructions: it passes whatever the COPY lines say.
        """
        uploaded = dockerfile.context_files(_ignore())
        self.assertTrue(uploaded, "the filtered context is empty — a pattern is wrong")
        leaked = [path for path in uploaded if is_sensitive(path)]
        self.assertEqual(leaked, [], "a whole-context copy would upload:\n  " + "\n  ".join(leaked))
        self.assertLess(
            len(uploaded),
            4000,
            "the context looks unfiltered — .git and the caches should not be here",
        )


class TestTheFilterDoesNotBreakABuild(unittest.TestCase):
    """The other direction: an ignore file that is too broad is a broken build."""

    def test_every_copy_source_still_matches_something(self) -> None:
        ignore = _ignore()
        empty: list[str] = []
        for stage, stage_instructions in dockerfile.stages().items():
            for source in dockerfile.copy_sources(stage_instructions):
                if not dockerfile.expand(source, ignore):
                    empty.append(f"{stage}: COPY {source}")
        self.assertEqual(
            empty,
            [],
            "these COPY sources would find nothing once filtered:\n  " + "\n  ".join(empty),
        )

    def test_the_files_the_stages_need_survive(self) -> None:
        ignore = _ignore()
        missing = [path for path in MUST_SURVIVE if ignore.ignored(path)]
        self.assertEqual(missing, [], "the ignore file would withhold: " + ", ".join(missing))

    def test_every_file_the_stages_need_exists(self) -> None:
        """So a renamed file fails here rather than in a build nobody runs."""
        missing = [path for path in MUST_SURVIVE if not (PROJECT_ROOT / path).exists()]
        self.assertEqual(missing, [], "MUST_SURVIVE names files that are gone: " + ", ".join(missing))


if __name__ == "__main__":
    unittest.main()
