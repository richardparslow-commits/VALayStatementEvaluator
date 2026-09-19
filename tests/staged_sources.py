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

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> bytes:
    """Raw stdout of a ``git`` call — a path or a file need not be text."""
    result = subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        failure = result.stderr.decode("utf-8", errors="replace").strip()
        raise SystemExit(f"git {' '.join(args)} failed:\n{failure}")
    return result.stdout


def _index_source(path: str) -> str:
    """The index's copy of *path*, as text.

    Decoded with replacement rather than strictly: a binary fixture is not a source
    file any rule should read, and the rules filter by name before they judge
    content — so refusing to decode one turned a file no rule objected to into a
    traceback.
    """
    return _git("show", f":{path}").decode("utf-8", errors="replace")


def staged(pathspec: str) -> list[tuple[str, str]]:
    """``(repo-relative path, source)`` for what is staged under *pathspec*.

    Renames come along — a file must not be able to escape a rule by moving — and
    deletions do not, having no content to judge. The content is the index's, for
    the reason in this module's docstring.

    Names are read NUL-separated and as bytes. Git leaves a space in a path
    unquoted, so splitting on whitespace turned one path into several and the
    lookups for the others failed — refusing a commit no rule objected to.
    ``os.fsdecode`` also reverses exactly when the name is handed back to git, so
    even a path that is not UTF-8 still names its blob.
    """
    listing = _git(
        "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR", "--", pathspec
    )
    paths = [os.fsdecode(name) for name in listing.split(b"\0") if name]
    return [(path, _index_source(path)) for path in paths]


def worktree(root: Path) -> list[tuple[str, str]]:
    """``(repo-relative path, source)`` for every ``.py`` file under *root*."""
    return [
        (str(path.relative_to(PROJECT_ROOT)), path.read_text(encoding="utf-8"))
        for path in sorted(root.rglob("*.py"))
        if path.is_file()
    ]
