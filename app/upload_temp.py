"""Verify the effective private, bounded Linux tmpfs used for upload spools."""
from __future__ import annotations

import os
import stat
import sys
import tempfile
from pathlib import Path

from .pilot import PilotBlocked

UPLOAD_TEMP = Path("/run/upload-tmp")
MAX_TEMP_BYTES = 256 * 1024 * 1024


def validate() -> None:
    try:
        if (not sys.platform.startswith("linux") or os.geteuid() == 0
                or any(os.environ.get(key) != str(UPLOAD_TEMP) for key in ("TMPDIR", "TEMP", "TMP"))
                or UPLOAD_TEMP.resolve(strict=True) != UPLOAD_TEMP
                or tempfile.gettempdir() != str(UPLOAD_TEMP)):
            raise ValueError
        info = UPLOAD_TEMP.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError
        # mountinfo describes THIS process's effective namespace, not Compose's
        # intended configuration. An exact mount prevents a disk-backed parent
        # or a nested filesystem from silently satisfying the policy.
        with Path("/proc/self/mountinfo").open("rb") as mounts:
            raw = mounts.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError
        matches = []
        for line in raw.decode("utf-8", "strict").splitlines():
            before, after = line.split(" - ", 1)
            fields, filesystem = before.split(), after.split()
            if fields[4] == str(UPLOAD_TEMP):
                matches.append((fields, filesystem))
        if len(matches) != 1:
            raise ValueError
        fields, filesystem = matches[0]
        if filesystem[0] != "tmpfs" or not {"rw", "noexec", "nosuid", "nodev"}.issubset(fields[5].split(",")):
            raise ValueError
        capacity = os.statvfs(UPLOAD_TEMP)
        if not 0 < capacity.f_blocks * capacity.f_frsize <= MAX_TEMP_BYTES:
            raise ValueError
        # Verify usable owner-only spooling on the observed mount before listen.
        with tempfile.TemporaryFile(dir=UPLOAD_TEMP) as spool:
            if stat.S_IMODE(os.fstat(spool.fileno()).st_mode) != 0o600:
                raise ValueError
            spool.write(b"synthetic-startup-probe")
            spool.flush()
    except (OSError, ValueError, IndexError, UnicodeError) as exc:
        raise PilotBlocked("Pilot upload storage must be the reviewed private bounded tmpfs.") from exc
