"""The harness-import rule, defined once for the suite and the commit hook.

Every test module has to import ``tests/hermetic.py`` **before** its first ``app``
import. ``app/config.py`` reads its constants once, at import, so a module that
imports the app first freezes whatever this machine had into the app for the rest
of the session, and a module that never imports the harness never empties that
configuration at all. Either way the module's result depends on the machine — the
failure the harness exists to prevent.

Two things enforce it, and they have to agree:

* ``tests/test_hermetic.py`` scans the working tree, and CI runs the suite, so a
  module that slips through fails the build. That is how it was found the first
  time: ``tests/test_job_queue_atomic.py`` landed on main while this harness was
  still unmerged, and the failure appeared on a *merge result* — one module late,
  in CI, as a red build nobody could reproduce locally.
* ``scripts/hooks/pre-commit`` runs this file over the **staged** modules, so the
  same mistake fails at the moment it is made rather than after a push.

The rule lives here rather than in the hook because a second copy in bash would
drift, and a drifted gate fails open: it keeps reporting green while checking less
than it claims. Run it by hand with::

    python -m tests.harness_imports --staged   # the staged test modules (the hook)
    python -m tests.harness_imports --all      # every test module in the tree

Exit status is 0 when everything is wired and 1 with one line per offender
otherwise. Nothing is printed on success, because a hook that announces itself on
every commit is noise.

Deliberately dependency-free — stdlib only, and no import of ``tests.hermetic``
— so the hook can run it in a checkout where the app's dependencies are not
installed and without paying for Streamlit's import.
"""
from __future__ import annotations

import ast
import fnmatch
import sys
import sysconfig
from collections.abc import Iterable
from pathlib import Path

from tests import staged_sources

PROJECT_ROOT = staged_sources.PROJECT_ROOT
TESTS_DIR = PROJECT_ROOT / "tests"

#: What counts as a test module: the same glob the suite's scan uses.
MODULE_GLOB = "test_*.py"

def _interpreter_stdlib_names() -> set[str]:
    """Top-level names this interpreter has: builtins, stdlib, ``lib-dynload``.

    The fallback for interpreters without ``sys.stdlib_module_names``, which
    ``Python 3.10`` added — and the hook does meet them: it prefers the project's
    venv, but with no venv it takes whatever ``python3`` is on ``PATH``, which on a
    Mac is the Command Line Tools 3.9 (measured). An incomplete set is not a
    harmless approximation here: a stdlib import misread as foreign refuses a
    *correctly wired* module, and the first version of this function did exactly
    that — ``import sys`` was flagged, because ``sys`` is compiled into the
    interpreter and has no file in the stdlib directory at all. So this asks three
    ways rather than curating a list.
    """
    names: set[str] = set(sys.builtin_module_names)
    stdlib = sysconfig.get_paths().get("stdlib")
    if not stdlib:  # pragma: no cover - an interpreter with no stdlib path
        return names
    root = Path(stdlib)
    for directory in (root, root / "lib-dynload"):
        if not directory.is_dir():
            continue
        for entry in directory.iterdir():
            module = entry.name.split(".")[0]
            if not module or module.startswith("_") or module in {
                "site-packages",
                "test",
            }:
                continue
            if entry.is_dir() or entry.suffix in {".py", ".pyc", ".so", ".pyd"}:
                names.add(module)
    return names


#: Top-level stdlib modules this project's Python has that an older interpreter
#: would not. The hook takes whatever ``python3`` is on ``PATH``, which can be the
#: 3.8/3.9 that macOS and CI images ship, while this project requires 3.12 — so the
#: question the fallback has to answer is "what is stdlib for *this code*", not
#: "what does this interpreter have". ``tomllib`` is the one that bit: a correctly
#: wired module that imports it was reported unwired, and the hook refused that
#: commit. ``tests/test_hermetic.py`` asserts every name here is in the interpreter
#: the suite runs on, so an obsolete entry fails a test rather than silently
#: misjudging a commit.
STDLIB_BEYOND_3_8 = {"graphlib", "tomllib", "zoneinfo"}


def stdlib_names() -> set[str]:
    """The stdlib roots *this project* recognises — exact where it can be.

    ``sys.stdlib_module_names`` is exact and it is what the suite runs with. When
    the interpreter is too old to have it, the interpreter's own names are widened
    by the ones added since 3.8 rather than answering the narrower question.
    """
    exact = set(getattr(sys, "stdlib_module_names", None) or ())
    if exact:
        return exact
    return _interpreter_stdlib_names() | STDLIB_BEYOND_3_8


STDLIB = stdlib_names() | {"__future__"}

#: One wording for the fix, used by the scan's assertion and the hook's refusal,
#: so the two cannot describe different remedies.
REMEDY = (
    "A test module that does not import the harness can read this machine's "
    "configuration; one that imports it after the app freezes that configuration "
    "into the app. Add, near the top and before any app import:\n"
    "  from tests import hermetic  # noqa: E402,F401  (hermetic test session; "
    "see tests/hermetic.py)"
)


def _import_roots(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name.split(".")[0] for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [(node.module or "").split(".")[0]]
    return []


def _first_foreign_import(tree: ast.Module, skip_line: int) -> int | None:
    """Line of the first module-level import that is not from the stdlib."""
    found = [
        int(node.lineno)
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and int(node.lineno) != skip_line
        and any(root and root not in STDLIB for root in _import_roots(node))
    ]
    return min(found) if found else None


def harness_import_offence(name: str, source: str) -> str | None:
    """Why ``source`` fails to protect itself, or None when it is fine.

    The ordering rule is the whole point: ``app/config.py`` reads its constants
    once, at import, so a module that imports the app *before* the harness freezes
    whatever the machine had into the app the rest of the session tests.
    """
    tree = ast.parse(source)
    harness = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "tests"
        and any(alias.name == "hermetic" for alias in node.names)
    ]
    if not harness:
        return f"{name}: does not import the hermetic harness"
    line = min(int(node.lineno) for node in harness)
    foreign = _first_foreign_import(tree, skip_line=line)
    if foreign is not None and foreign < line:
        return f"{name}:{line}: harness import lands after the import on line {foreign}"
    return None


def is_test_module(path: str | Path) -> bool:
    """Whether *path* is a test module, by the same rule the staged scan uses."""
    return fnmatch.fnmatch(Path(path).name, MODULE_GLOB)


def test_module_sources(directory: Path | None = None) -> list[tuple[str, str]]:
    """``(repo-relative path, source)`` for every test module — the working tree."""
    root = TESTS_DIR if directory is None else directory
    return [
        (path, source)
        for path, source in staged_sources.worktree(root)
        if is_test_module(path)
    ]


def staged_test_module_sources() -> list[tuple[str, str]]:
    """``(repo-relative path, source)`` for the test modules staged for commit.

    The **index** copy rather than the file on disk, because a commit is what is
    staged — see ``tests/staged_sources.py``, which owns that distinction for every
    rule rather than letting each one decide it.
    """
    return [
        (path, source)
        for path, source in staged_sources.staged("tests/")
        if is_test_module(path)
    ]


def offenders(sources: Iterable[tuple[str, str]]) -> list[str]:
    """One line per module in *sources* that is not wired into the harness."""
    return [
        offence
        for name, source in sources
        if (offence := harness_import_offence(name, source)) is not None
    ]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:]) if argv is None else list(argv)
    if args == ["--staged"]:
        sources = staged_test_module_sources()
    elif args == ["--all"]:
        sources = test_module_sources()
    else:
        print(
            "usage: python -m tests.harness_imports (--staged | --all)",
            file=sys.stderr,
        )
        return 2

    found = offenders(sources)
    if not found:
        return 0

    print("", file=sys.stderr)
    print(
        "✖  Blocked: these test modules are not wired into the hermetic harness:",
        file=sys.stderr,
    )
    for offence in found:
        print(f"   {offence}", file=sys.stderr)
    print("", file=sys.stderr)
    for line in REMEDY.splitlines():
        print(f"   {line}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
