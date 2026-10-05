"""Explicit synthetic/reviewed-public extraction; never a pilot tool.

Run as `python -m scripts.extract_pdfs`. No default inputs or repository output.
The isolated parser must be configured; there is no in-process fallback.
"""
from __future__ import annotations
import argparse
import hashlib
import os
import re
import stat
import sys
from pathlib import Path
from typing import Sequence
from app import config, pilot
from app.bounded_json import decode_json
from app.isolated_extract import IsolatedExtractor

REPOSITORY = Path(__file__).resolve().parent.parent
REFUSAL = "Reference extraction refused; no derivative was published."


class PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        # argparse's default unknown-argument errors can echo private filenames.
        self.exit(2, REFUSAL + "\n")


def require_class(data_class: str) -> None:
    # Before ALL source/manifest/output access, including programmatic entry.
    # A declaration cannot detect misclassification; restrict operator access.
    if pilot.enabled() or data_class not in ("synthetic", "approved-public"):
        raise ValueError(REFUSAL)


def read_regular_bounded(path: Path, maximum: int) -> bytes:
    if not path.is_absolute() or not hasattr(os, "O_NOFOLLOW"):
        raise ValueError(REFUSAL)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ValueError(REFUSAL)
        raw = bytearray()
        while len(raw) <= maximum:
            chunk = os.read(fd, min(65536, maximum + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after, current = os.fstat(fd), path.lstat()
        def identity(value: os.stat_result) -> tuple[int, ...]:
            return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if len(raw) > maximum or len(raw) != before.st_size or identity(before) != identity(after) or identity(after) != identity(current):
            raise ValueError(REFUSAL)
        return bytes(raw)
    finally:
        os.close(fd)


def approved_hashes(manifest: Path) -> set[str]:
    value = decode_json(pilot._approval_bytes(manifest))
    if not isinstance(value, dict) or set(value) != {"schema_version", "approved_sha256"} or type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError(REFUSAL)
    hashes = value["approved_sha256"]
    if not isinstance(hashes, list) or not 0 < len(hashes) <= 200 or any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes) or len(set(hashes)) != len(hashes):
        raise ValueError(REFUSAL)
    return set(hashes)


def extract(path: Path, *, data_class: str, public_manifest: Path | None = None) -> str:
    require_class(data_class)
    try:
        if path.suffix.lower() != ".pdf":
            raise ValueError
        allowed = approved_hashes(public_manifest) if data_class == "approved-public" and public_manifest is not None else set()
        if data_class == "approved-public" and not allowed:
            raise ValueError
        data = read_regular_bounded(path, min(config.MAX_UPLOAD_BYTES, 50 * 1024 * 1024))
        if data_class == "approved-public" and hashlib.sha256(data).hexdigest() not in allowed:
            raise ValueError
        # Fixed label prevents filenames from entering parser errors/output.
        documents, skipped = IsolatedExtractor().extract("reference.pdf", data)
        if skipped or len(documents) != 1:
            raise ValueError
        document = documents[0]
        if not document.coverage_known or document.unreadable_pages or not document.pages or len(document.pages) != document.total_pages:
            raise ValueError
        return "".join(f"\n\n----- Page {p.page} -----\n{p.text}" for p in document.pages)
    except Exception:
        raise ValueError(REFUSAL) from None


def output_parent(path: Path) -> int:
    if not path.is_absolute() or path.parent.resolve(strict=True) != path.parent or path.is_relative_to(REPOSITORY):
        raise ValueError(REFUSAL)
    if any((p / ".git").exists() for p in (path.parent, *path.parent.parents)):
        raise ValueError(REFUSAL)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    info = os.fstat(fd)
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
        os.close(fd)
        raise ValueError(REFUSAL)
    return fd


def publish(path: Path, payload: bytes, *, data_class: str) -> None:
    require_class(data_class)
    parent = output_parent(path)
    try:
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
        except BaseException:
            os.unlink(path.name, dir_fd=parent)
            raise
    finally:
        os.close(parent)


def main(argv: Sequence[str] | None = None) -> int:
    parser = PrivateArgumentParser(description=__doc__)
    parser.add_argument("--data-class", required=True, choices=("synthetic", "approved-public", "sensitive"))
    parser.add_argument("--public-manifest", type=Path)
    parser.add_argument("--out", required=True, type=Path, help="new retained TXT in an existing private directory outside Git")
    parser.add_argument("source", type=Path)
    args = parser.parse_args(argv)
    try:
        require_class(args.data_class)
        parent = output_parent(args.out)
        os.close(parent)
        if args.out.exists() or args.out.is_symlink():
            raise ValueError
        text = extract(args.source, data_class=args.data_class, public_manifest=args.public_manifest)
        publish(args.out, text.encode("utf-8", "strict"), data_class=args.data_class)
    except Exception:
        print(REFUSAL, file=sys.stderr)
        return 3
    print("Reference derivative published with owner-only access. Apply the approved retention policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
