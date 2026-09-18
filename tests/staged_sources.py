"""Read source files as a *commit* contains them, once, for every rule that needs it.

Two rules now have the same two consumers — the suite's scans and
``scripts/hooks/pre-commit`` — and both come down to the same question: *which*
copy of a file should the rule read? The answer is not the same one in both
places, and getting it wrong is silent:

* the **hook** must read the index. A commit is what is staged, so a file edited
  after ``git add`` is not in it — judging the working tree there would refuse a
  correct commit and pass an incorrect one, depending on which way the two copies
  drifted apart.
* the **suite** must read the working tree, because it is asserting a property of
  the repository the reader is looking at.

That distinction, the two ``git`` calls that implement it, and the choice of
rename/copy handling all live here rather than in each rule, so the rules cannot
disagree about what a commit contains. Everything is stdlib-only so the hook can
run it in a checkout where the app's dependencies are not installed.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout


def staged(pathspec: str) -> list[tuple[str, str]]:
    """``(repo-relative path, source)`` for what is staged under *pathspec*.

    Renames come along — a file must not be able to escape a rule by moving — and
    deletions do not, having no content to judge. The content is the index's, for
    the reason in this module's docstring.
    """
    paths = _git(
        "diff", "--cached", "--name-only", "--diff-filter=ACMR", "--", pathspec
    ).split()
    return [(path, _git("show", f":{path}")) for path in paths]


def worktree(root: Path) -> list[tuple[str, str]]:
    """``(repo-relative path, source)`` for every ``.py`` file under *root*."""
    return [
        (str(path.relative_to(PROJECT_ROOT)), path.read_text(encoding="utf-8"))
        for path in sorted(root.rglob("*.py"))
        if path.is_file()
    ]
