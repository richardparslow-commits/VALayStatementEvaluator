"""The Dockerfile and .dockerignore, read once for the tests that guard them.

No CI job builds an image, so the two files that decide what an image contains
are checked by reading them: ``tests/test_sandbox_image.py`` asserts the sandbox
stage can run the app and the suite, and ``tests/test_dockerignore.py`` asserts
nothing sensitive can reach a stage and nothing a stage needs is dropped.

Both need the same view of the Dockerfile — its stages, their instructions, their
``COPY`` sources — and both need the same answer to "would this path be uploaded
at all?". Two copies of either would drift, and drift here fails *open*: the
suite would keep reporting green while checking less than it claims. So the
parser and the matcher live here, and the test modules own only the assertions.

The matcher implements Docker's documented subset (Go's ``filepath.Match`` per
path segment, plus ``**`` for any depth, ``!`` to re-include, last match wins).
It refuses a construct it cannot translate instead of guessing: an unparsed
pattern would otherwise read as "not ignored", which is the wrong default for a
file whose whole job is to keep secrets out. One documented deviation, in the
stricter direction: a trailing ``/`` here means "this directory and everything
under it", while Docker additionally refuses to match a *file* of that name. That
can only make these checks reject more, never accept more.

Stdlib only, and no import of ``tests.hermetic``, so a throwaway script can use
it too — the verification runs in this repository's history did exactly that.
"""
from __future__ import annotations

import fnmatch
import glob
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = PROJECT_ROOT / "Dockerfile"
DOCKERIGNORE = PROJECT_ROOT / ".dockerignore"


class UnsupportedPattern(ValueError):
    """A .dockerignore pattern this matcher will not guess at."""


# --------------------------------------------------------------------- parsing

def instructions(text: str) -> list[str]:
    """Dockerfile instructions, with continuations joined and comments dropped.

    Line continuations matter: most of the interesting instructions in this file
    are multi-line, and a per-line scan would see ``COPY x \\`` and nothing else.
    """
    out: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not buffer and (not line or line.startswith("#")):
            continue
        buffer += line[:-1].strip() + " " if line.endswith("\\") else line
        if not raw.rstrip().endswith("\\"):
            out.append(buffer)
            buffer = ""
    return out


def stages(text: str | None = None) -> dict[str, list[str]]:
    """``stage name -> its instructions``, keyed by ``AS <name>``."""
    source = DOCKERFILE.read_text(encoding="utf-8") if text is None else text
    found: dict[str, list[str]] = {}
    current: str | None = None
    for instruction in instructions(source):
        if instruction.startswith("FROM "):
            tokens = instruction.split()
            current = tokens[3] if len(tokens) >= 4 and tokens[2].upper() == "AS" else tokens[-1]
            found[current] = []
        elif current is not None:
            found[current].append(instruction)
    return found


def copy_sources(stage: list[str]) -> list[str]:
    """The source paths of every ``COPY`` in *stage* (the destination dropped)."""
    sources: list[str] = []
    for instruction in stage:
        if instruction.startswith("COPY "):
            tokens = instruction.split()
            sources.extend(tokens[1:-1])
    return sources


def carries(stage: list[str], path: str) -> bool:
    """Would *stage* bring repo-relative *path* in, by its own ``COPY`` lines?"""
    for source in copy_sources(stage):
        if source == path:
            return True
        if source.endswith("/") and (path == source[:-1] or path.startswith(source)):
            return True
        # A glob source matches only within one directory: fnmatch's `*` crosses
        # `/` happily, so `*.md` would otherwise claim every nested page.
        if "/" not in source and "/" not in path and fnmatch.fnmatch(path, source):
            return True
    return False


def files_carried(stage: list[str]) -> set[str]:
    """The repo-relative *files* a stage's ``COPY`` lines bring in, expanded.

    Directories are expanded to the files under them, so "would
    ``requirements-dev.txt`` be in the image?" can be asked of a stage that copies
    a directory containing it.
    """
    found: set[str] = set()
    for source in copy_sources(stage):
        found.update(expand(source))
    return found


def files_read_before_being_carried(
    stage: list[str], inherited_stage: list[str] | None = None
) -> list[str]:
    """Files a ``RUN`` reads that the stage has not brought in *by that point*.

    Docker runs a stage top to bottom, so a RUN cannot read what a later COPY
    brings in. This is the bug ``pip install -r requirements-dev.txt`` had: the RUN
    line names the file, so ``carries`` — which asks whether the string appears in
    the stage — reported it present while the build failed at that step (measured:
    the first CI run of the sandbox-image job, before any image had ever been
    built). A test that asks about mentions cannot tell a COPY from a RUN.

    Includes are followed, because the file a RUN names is often not the file pip
    reads: ``requirements-dev.txt`` begins with ``-r requirements.txt``, so a stage
    that carries the first and not the second fails inside pip with a path nobody
    reading the Dockerfile can see (measured: the second CI run, after the first
    fix). One level of indirection is the whole trap, and it is free to follow.

    *inherited_stage* is the stage this one is built ``FROM``, whose files are
    present before this stage's first instruction.
    """
    available = files_carried(inherited_stage) if inherited_stage else set()
    missing: list[str] = []
    for instruction in stage:
        if instruction.startswith("COPY "):
            available |= files_carried([instruction])
            continue
        if not instruction.startswith("RUN ") or "pip install" not in instruction:
            # Only pip reads requirements files, and asking every RUN is not
            # harmless: `rm -rf /var/lib/apt/lists/*` reads as `-rf` to a flag
            # parser, which is how the attached form below reported a file `f`.
            continue
        named = requirement_files_named(instruction)
        for name in _requirement_closure(named):
            if name not in available:
                missing.append(name)
    return sorted(set(missing))


def requirement_files_named(text: str) -> list[str]:
    """The files a command's ``-r``/``--requirement`` flags point at.

    Covers the three forms pip accepts: ``-r file``, ``-rfile`` and
    ``--requirement=file``. The attached form additionally has to look like a path,
    so a short option cluster is not mistaken for a filename.
    """
    names: list[str] = []
    tokens = text.split()
    for index, token in enumerate(tokens):
        if token in ("-r", "--requirement"):
            if index + 1 < len(tokens):
                names.append(tokens[index + 1])
        elif token.startswith("--requirement="):
            names.append(token.split("=", 1)[1])
        elif token.startswith("-r") and len(token) > 2 and any(c in token for c in "./"):
            names.append(token[2:])
    return [name for name in names if not name.startswith("-")]


def _requirement_closure(names: list[str]) -> list[str]:
    """*names* plus everything those requirement files include, repo-relative.

    Paths inside a requirements file are relative to that file, which is why the
    queue carries the directory each name came from.
    """
    found: list[str] = []
    queue: list[tuple[str, str]] = [(name, "") for name in names]
    while queue:
        name, base = queue.pop(0)
        if name.startswith("/"):  # an absolute path is the image's business, not the repo's
            continue
        relative = f"{base}/{name}".lstrip("/") if base else name
        if relative in found:
            continue
        # Named but absent on disk is still reported: the caller decides whether the
        # stage carries it, and a typo in a requirements file is the same failure.
        found.append(relative)
        if not (PROJECT_ROOT / relative).is_file():
            continue
        for line in (PROJECT_ROOT / relative).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parent = str(Path(relative).parent)
            for include in requirement_files_named(stripped):
                queue.append((include, "" if parent == "." else parent))
    return found


# ------------------------------------------------------------ ignore patterns

class _Pattern:
    """One line of a .dockerignore, compiled."""

    def __init__(self, line: str) -> None:
        self.raw = line.strip()
        text = self.raw
        self.exclusion = text.startswith("!")
        if self.exclusion:
            text = text[1:].strip()
        # A trailing slash names a directory. Docker's use for that fact is to
        # refuse matching a *file* of the name; here it means "and its contents",
        # which can only reject more (see the module docstring).
        self.directory_only = text.endswith("/")
        text = text.rstrip("/").removeprefix("./")
        if not text:
            raise UnsupportedPattern(line)
        self._regex = re.compile("^" + "".join(self._translate(text)) + "$")

    @staticmethod
    def _translate(pattern: str) -> list[str]:
        """Translate a Docker pattern into regex pieces over a `/`-joined path."""
        body: list[str] = []
        index = 0
        while index < len(pattern):
            char = pattern[index]
            if char == "*":
                if pattern[index : index + 2] == "**":
                    before_ok = index == 0 or pattern[index - 1] == "/"
                    after = index + 2
                    after_ok = after == len(pattern) or pattern[after] == "/"
                    if not (before_ok and after_ok):
                        raise UnsupportedPattern(pattern)
                    if after < len(pattern):  # `**/` — any number of directories
                        body.append("(?:[^/]+/)*")
                        index = after + 1
                        continue
                    body.append(".*")  # trailing `**` — anything below
                    index = after
                    continue
                body.append("[^/]*")
                index += 1
                continue
            if char == "?":
                body.append("[^/]")
                index += 1
                continue
            if char == "[":
                end = pattern.find("]", index + 1)
                if end == -1:
                    raise UnsupportedPattern(pattern)
                body.append(pattern[index : end + 1])
                index = end + 1
                continue
            body.append(re.escape(char))
            index += 1
        return body

    def matches(self, path: str) -> bool:
        """True when this pattern names *path* or one of its parent directories."""
        if self._regex.match(path):
            return True
        parts = path.split("/")
        for depth in range(1, len(parts)):
            if self._regex.match("/".join(parts[:depth])):
                return True
        return False


class IgnoreFile:
    """A parsed .dockerignore, answering Docker's question: is this uploaded?"""

    def __init__(self, text: str) -> None:
        self.patterns: list[_Pattern] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            self.patterns.append(_Pattern(line))

    @classmethod
    def for_repo(cls) -> IgnoreFile:
        return cls(DOCKERIGNORE.read_text(encoding="utf-8"))

    def ignored(self, path: str) -> bool:
        """Docker's rule: evaluate in order, the last match decides.

        ``ignored`` on a directory is enough to exclude its contents — that is
        why ``_Pattern.matches`` also tests parent directories, and why callers
        can ask about a path that does not exist yet.
        """
        normalized = path.replace("\\", "/").removeprefix("./")
        verdict = False
        for pattern in self.patterns:
            if pattern.matches(normalized):
                verdict = not pattern.exclusion
        return verdict


# --------------------------------------------------- expanding a COPY source

def expand(source: str, ignore: IgnoreFile | None = None) -> list[str]:
    """The files (not directories) a ``COPY`` source would bring in.

    With *ignore*, anything the .dockerignore excludes is skipped, and an ignored
    directory is not descended into — which is what Docker does when it assembles
    the context, and what keeps a whole-context scan cheap enough to run in a
    unit test.
    """
    files_on_disk = sorted(glob.glob(str(PROJECT_ROOT / source), recursive=True))
    found: list[str] = []
    for match in files_on_disk:
        path = Path(match)
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if ignore is not None and ignore.ignored(relative):
            continue
        if path.is_file():
            found.append(relative)
        elif path.is_dir():
            found.extend(_files_under(path, ignore))
    return sorted(found)


def _files_under(directory: Path, ignore: IgnoreFile | None) -> list[str]:
    """Every file below *directory*, skipping what the ignore file excludes."""
    found: list[str] = []
    for root, _dirnames, filenames in _walk(directory, ignore):
        for name in filenames:
            relative = Path(root, name).relative_to(PROJECT_ROOT).as_posix()
            if ignore is not None and ignore.ignored(relative):
                continue
            found.append(relative)
    return found


def _walk(directory: Path, ignore: IgnoreFile | None):
    """``os.walk`` that prunes ignored directories, so `.git` is never entered."""
    for root, dirnames, filenames in os.walk(directory):
        if ignore is not None:
            keep = []
            for name in dirnames:
                relative = Path(root, name).relative_to(PROJECT_ROOT).as_posix()
                if not ignore.ignored(relative):
                    keep.append(name)
            dirnames[:] = keep
        yield root, dirnames, filenames


def context_files(ignore: IgnoreFile | None = None) -> list[str]:
    """Every file the build context would carry — the whole repository, filtered.

    This is the ``COPY . .`` case, and the reason the guard is about the filter
    rather than about today's ``COPY`` lines.
    """
    return sorted(_files_under(PROJECT_ROOT, ignore))
