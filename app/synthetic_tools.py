"""Standalone preprocessing is synthetic-only until separately accepted."""
from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def require_synthetic_tool() -> None:
    from . import pilot
    if pilot.enabled():
        raise ValueError("Standalone preprocessing is excluded from the controlled pilot. Use the separately approved isolated workflow.")


@contextmanager
def temporary_work(parent: Path | None = None) -> Iterator[Path]:
    """Own and remove only this invocation's private scratch directory."""
    require_synthetic_tool()
    if parent is not None:
        if parent.is_symlink():
            raise ValueError("The scratch parent must not be a symbolic link.")
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ocr-work-", dir=parent) as directory:
        yield Path(directory)


def write_private(path: Path, data: bytes, *, overwrite: bool = True) -> None:
    """Atomic owner-only derivative; refuse symbolic destinations."""
    require_synthetic_tool()
    if path.is_symlink():
        raise ValueError("Refusing a symbolic output destination.")
    descriptor, name = tempfile.mkstemp(prefix=".ocr-output-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # Publish without a check-then-overwrite race.
    finally:
        temporary.unlink(missing_ok=True)
