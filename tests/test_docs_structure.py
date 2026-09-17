"""Structural invariants for the docs, the way ``test_security_gitignore`` guards git.

Two defects found while writing this file, both invisible to every other test in
the suite because they only exist in rendered markdown:

* a blockquote inserted **inside** the README's environment-variable table split
  it in two, so every row after the insertion rendered as literal pipe text; and
* ``scripts/live_draft_e2e.py`` was referenced by no document at all — a live
  check the repo ships and nobody could find.

Neither is a code bug, which is exactly why it needs a test: the failure mode is
silence. These checks are deliberately about *structure* (a table stayed one
table, a file is reachable from a page a human reads, a link points at a file that
exists) rather than about wording, so they do not need editing every time a
sentence changes.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The pages that constitute "documented". If a script is on none of them, an
# operator cannot find it, however good its own docstring is.
DOC_PAGES = (
    "README.md",
    "DEPLOYMENT.md",
    "ARCHITECTURE.md",
    "COMPATIBILITY.md",
    "deploy/k8s/README.md",
)

_SCRIPT_REF = re.compile(r"scripts/([A-Za-z0-9_]+\.py)")

# Markdown link targets that are local paths, e.g. `](scripts/foo.py)` or
# `](DEPLOYMENT.md#anchor)`. Anything with a scheme is skipped.
_MD_LINK = re.compile(r"\]\((?!/)([^)\s]+)\)")


def _doc_texts() -> dict[str, str]:
    return {page: (PROJECT_ROOT / page).read_text(encoding="utf-8") for page in DOC_PAGES}


def _prose_lines(page: str) -> list[tuple[int, str]]:
    """Lines outside fenced code blocks — the only place markdown is markdown.

    Log excerpts and shell transcripts are full of lines beginning with ``|`` that
    have nothing to do with tables, and flagging those would make this check cry
    wolf until someone deleted it.
    """
    out: list[tuple[int, str]] = []
    in_fence = False
    for number, line in enumerate(
        (PROJECT_ROOT / page).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append((number, line))
    return out


class TestTablesAreNotInterrupted(unittest.TestCase):
    """A `|` row whose previous line is ordinary text is orphaned pipe markup."""

    def test_no_table_row_follows_prose_without_a_blank_line(self) -> None:
        offenders: list[str] = []
        for page in DOC_PAGES:
            lines = _prose_lines(page)
            for index, (number, line) in enumerate(lines):
                if not line.startswith("|"):
                    continue
                previous = lines[index - 1][1] if index else ""
                if previous.strip() and not previous.startswith("|"):
                    offenders.append(f"{page}:{number}: {line[:60]}")
        self.assertEqual(
            offenders,
            [],
            "table rows separated from their table by prose. Insert the paragraph "
            "before or after the whole table, not inside it:\n  " + "\n  ".join(offenders),
        )


class TestEveryScriptIsFindable(unittest.TestCase):
    def test_each_script_is_named_on_a_documentation_page(self) -> None:
        scripts = sorted(p.name for p in (PROJECT_ROOT / "scripts").glob("*.py"))
        self.assertTrue(scripts, "no scripts found — the path must be wrong")
        documented = set()
        for text in _doc_texts().values():
            documented.update(_SCRIPT_REF.findall(text))
            # The README's directory tree lists names without the `scripts/`
            # prefix; that still tells a reader the file exists.
            documented.update(re.findall(r"\b([A-Za-z0-9_]+\.py)\b", text))
        missing = [name for name in scripts if name not in documented]
        self.assertEqual(
            missing,
            [],
            "these scripts are referenced by no doc page, so an operator has no way "
            "to discover them: " + ", ".join(missing),
        )

    def test_the_failover_drill_is_documented_with_its_exit_codes(self) -> None:
        """A check script is only usable if its exit codes are written down."""
        deployment = (PROJECT_ROOT / "DEPLOYMENT.md").read_text(encoding="utf-8")
        self.assertIn("scripts/rehearse_failover.py", deployment)
        for flag in ("--expect-idle", "--expect-active", "--json", "--watch"):
            with self.subTest(flag=flag):
                self.assertIn(flag, deployment)


class TestDocLinksResolve(unittest.TestCase):
    """A link to a deleted or renamed file reads as confidence and is a 404."""

    def test_relative_markdown_links_point_at_real_files(self) -> None:
        broken: list[str] = []
        for page in DOC_PAGES:
            base = (PROJECT_ROOT / page).parent
            for target in _MD_LINK.findall(
                (PROJECT_ROOT / page).read_text(encoding="utf-8")
            ):
                path = target.split("#", 1)[0]
                if not path or "{" in path:  # templated or anchor-only
                    continue
                if "://" in path or path.startswith("mailto:"):  # not a local file
                    continue
                if not (base / path).exists() and not (PROJECT_ROOT / path).exists():
                    broken.append(f"{page} -> {target}")
        self.assertEqual(broken, [], "broken relative links:\n  " + "\n  ".join(broken))

    def test_cross_page_anchors_exist_on_the_target_page(self) -> None:
        """`DEPLOYMENT.md#section` must actually land on that heading."""

        def slugs(text: str) -> set[str]:
            found = set()
            for line in text.splitlines():
                if not line.startswith("#"):
                    continue
                heading = line.lstrip("#").strip()
                # GitHub's heading slug rule, near enough for our headings.
                slug = re.sub(r"[^a-z0-9 -]", "", heading.lower())
                found.add(slug.replace(" ", "-"))
            return found

        broken: list[str] = []
        for page in DOC_PAGES:
            text = (PROJECT_ROOT / page).read_text(encoding="utf-8")
            for target in _MD_LINK.findall(text):
                if "#" not in target:
                    continue
                path, _, anchor = target.partition("#")
                target_page = (PROJECT_ROOT / path) if path else (PROJECT_ROOT / page)
                if not target_page.exists() or target_page.suffix != ".md":
                    continue
                if anchor not in slugs(target_page.read_text(encoding="utf-8")):
                    broken.append(f"{page} -> {target}")
        self.assertEqual(broken, [], "anchors that match no heading:\n  " + "\n  ".join(broken))


if __name__ == "__main__":
    unittest.main()
