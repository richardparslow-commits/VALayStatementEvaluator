"""The Streamlit-config-option rule, defined once for the suite and the commit hook.

``app/`` must decide nothing from the *value* of a Streamlit option. An option's
value comes from the machine and from the directory the process started in — a
``config.toml`` in ``~`` or in the working directory is enough — so a decision made
from one is ambient: two deployments of identical code behave differently, and the
suite cannot say which behaviour was tested.

The case that matters most is security posture. ``app/main.py`` checks that the
deployment ships the hardening by reading the committed file as **text**, because
Streamlit exposes no way to ask which source a value came from: asking for the
value answers "is XSRF on in this process?", which a machine's own config file can
answer wrongly, instead of "does the deployment ship the hardening?". A future
refactor that reaches for ``get_option`` would quietly restore that mistake.

Two things enforce it, and they have to agree:

* ``tests/test_hermetic.py`` scans the working tree, and CI runs the suite, so an
  app module that reads an option fails the build.
* ``scripts/hooks/pre-commit`` runs this file over the **staged** app modules, so
  the same mistake fails at the moment it is made rather than after a push.

The rule lives here rather than in the hook because a second copy in bash would
drift, and a drifted gate fails open: it keeps reporting green while checking less
than it claims. Run it by hand with::

    python -m tests.streamlit_option_reads --staged   # staged app modules (the hook)
    python -m tests.streamlit_option_reads --all      # every app module in the tree

Exit status is 0 when the app decides nothing from an option and 1 with one line
per offending read otherwise. Nothing is printed on success, because a hook that
announces itself on every commit is noise.

Deliberately dependency-free — stdlib only, and neither ``tests.hermetic`` nor
Streamlit itself — so the hook can run it in a checkout where the app's
dependencies are not installed. Reading this rule's *source* is not the same as
importing the config module it is about, and only the source is needed.
"""
from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from pathlib import Path

from tests import staged_sources

PROJECT_ROOT = staged_sources.PROJECT_ROOT
APP_DIR = PROJECT_ROOT / "app"

#: What counts as an app module: the same suffix the suite's scan uses.
MODULE_SUFFIX = ".py"

#: Ways a module could ask Streamlit for an option's value. Compiled once so the
#: scan, its self-tests and the hook cannot disagree about what is being matched —
#: these are the patterns that were in ``tests/test_hermetic.py``, the scan just
#: moved to a module both callers can reach.
OPTION_READ_PATTERNS = tuple(re.compile(pattern) for pattern in (
    r"\bget_option\s*\(",
    r"\bset_option\s*\(",
    r"\bst\.config\b",
    r"\bstreamlit\.config\b",
))

#: One wording for the fix, used by the scan's assertion and the hook's refusal,
#: so the two cannot describe different remedies.
REMEDY = (
    "An option's value comes from the machine and the working directory, so a "
    "decision made from one is ambient — the same code would behave differently "
    "in two deployments. Read the deployment's own file as text instead, the way "
    "the hardening check does (app/main.py:_check_streamlit_config_hardening):\n"
    "  text = (Path(__file__).parent.parent / '.streamlit' / 'config.toml')"
    ".read_text(encoding='utf-8')"
)


def option_read_offence(name: str | Path, source: str) -> list[str]:
    """Lines where *source* reads a Streamlit config option, as ``name:line``."""
    return [
        f"{name}:{number}: {line.strip()}"
        for number, line in enumerate(source.splitlines(), start=1)
        if any(pattern.search(line) for pattern in OPTION_READ_PATTERNS)
    ]


def is_app_module(path: str | Path) -> bool:
    """Whether *path* is an app module, by the same rule the staged scan uses.

    The scope is exactly ``app/``: ``tests/hermetic.py`` reaches into
    ``streamlit.config`` on purpose (``get_config_files``, ``get_option``,
    ``ConfigOption.sensitive`` — no public API covers those routes), so a rule
    that covered the whole repository would flag the harness that makes the suite
    hermetic in the first place.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(PROJECT_ROOT)
        except ValueError:
            return False
    return candidate.suffix == MODULE_SUFFIX and candidate.parts[:1] == (
        APP_DIR.name,
    )


def app_module_sources(directory: Path | None = None) -> list[tuple[str, str]]:
    """``(repo-relative path, source)`` for every app module — the working tree."""
    root = APP_DIR if directory is None else directory
    return [
        (path, source)
        for path, source in staged_sources.worktree(root)
        if is_app_module(path)
    ]


def staged_app_module_sources() -> list[tuple[str, str]]:
    """``(repo-relative path, source)`` for the app modules staged for commit.

    The **index** copy rather than the file on disk, because a commit is what is
    staged — see ``tests/staged_sources.py``, which owns that distinction for every
    rule rather than letting each one decide it.
    """
    return [
        (path, source)
        for path, source in staged_sources.staged(f"{APP_DIR.name}/")
        if is_app_module(path)
    ]


def offenders(sources: Iterable[tuple[str, str]]) -> list[str]:
    """One line per option read in *sources*; empty when the app decides nothing."""
    return [
        offence
        for path, source in sources
        for offence in option_read_offence(path, source)
    ]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:]) if argv is None else list(argv)
    if args == ["--staged"]:
        sources = staged_app_module_sources()
    elif args == ["--all"]:
        sources = app_module_sources()
    else:
        print(
            "usage: python -m tests.streamlit_option_reads (--staged | --all)",
            file=sys.stderr,
        )
        return 2

    found = offenders(sources)
    if not found:
        return 0

    print("", file=sys.stderr)
    print(
        "✖  Blocked: these app modules read Streamlit config options:",
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
