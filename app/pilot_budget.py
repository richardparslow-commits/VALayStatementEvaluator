"""Private, single-host pilot control ledger. No case or provider content.

Every run reserves its entire reviewed attempt/charge envelope. Actual attempts
keep their full conservative charge, including errors or absent usage; only
unstarted attempts are released on normal unwinding. A crash keeps the entire
reservation. Dollar protection depends on the operator proving the charge
ceiling and an independent provider/account cutoff, not on token estimates.
"""
from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import stat
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from .pilot import PilotBlocked

REFUSAL = "Pilot budget control is unavailable or exhausted. Ask the operator to review it."


def _now() -> float:
    return datetime.now(timezone.utc).timestamp()


def policy(approval: Mapping[str, Any]) -> dict[str, Any]:
    """Explicit operator values; no pricing, limits or evidence are invented."""
    try:
        value = approval["quota_policy"]
        names = {"pilot_id", "expires_at", "participant_daily_starts", "run_attempts",
                 "run_prompt_chars", "attempt_output_tokens", "attempt_charge_microusd",
                 "pilot_total_microusd", "pilot_total_attempts"}
        if not isinstance(value, dict) or set(value) != names:
            raise ValueError
        if str(uuid.UUID(value["pilot_id"])) != value["pilot_id"]:
            raise ValueError
        expiry = datetime.fromisoformat(value["expires_at"].replace("Z", "+00:00"))
        if expiry.tzinfo is None or not _now() < expiry.timestamp() <= _now() + 30 * 86400:
            raise ValueError
        bounds = {"participant_daily_starts": (1, 10), "run_attempts": (1, 200),
                  "run_prompt_chars": (1, 2_000_000), "attempt_output_tokens": (1, 8192),
                  "attempt_charge_microusd": (1, 1_000_000_000),
                  "pilot_total_microusd": (1, 1_000_000_000_000),
                  "pilot_total_attempts": (1, 100_000)}
        for key, (low, high) in bounds.items():
            if type(value[key]) is not int or not low <= value[key] <= high:
                raise ValueError
        if (value["pilot_total_attempts"] < value["run_attempts"]
                or value["pilot_total_microusd"] < value["run_attempts"] * value["attempt_charge_microusd"]):
            raise ValueError
        if datetime.fromisoformat(approval["expires_at"].replace("Z", "+00:00")).timestamp() > expiry.timestamp():
            raise ValueError
        return dict(value)
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError) as exc:
        raise PilotBlocked(REFUSAL) from exc


def _private_directory(path: Path) -> None:
    info = path.parent.lstat()
    if (not path.is_absolute() or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError


def provision(path: Path, approval: Mapping[str, Any]) -> None:
    """Explicit offline provisioning only; admission never creates a ledger."""
    settings = policy(approval)
    _private_directory(path)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    os.close(descriptor)
    descriptor = os.open(Path(str(path) + ".lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    os.close(descriptor)
    # Directory is private; SQLite's journal contains only opaque control data.
    with sqlite3.connect(path) as db:
        db.executescript("""
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            PRAGMA secure_delete=ON;
            CREATE TABLE control (version INTEGER NOT NULL, policy TEXT NOT NULL,
                salt TEXT NOT NULL, charged INTEGER NOT NULL, reserved_attempts INTEGER NOT NULL,
                last_now REAL NOT NULL);
            CREATE TABLE runs (id TEXT PRIMARY KEY, owner TEXT NOT NULL, started REAL NOT NULL,
                state TEXT NOT NULL, attempted INTEGER NOT NULL, prompt_chars INTEGER NOT NULL);
            CREATE TABLE attempts (run TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                number INTEGER NOT NULL, reconciled INTEGER NOT NULL,
                input_tokens INTEGER, output_tokens INTEGER, PRIMARY KEY(run, number));
        """)
        db.execute("INSERT INTO control VALUES (1, ?, ?, 0, 0, ?)",
                   (json.dumps(settings, sort_keys=True), secrets.token_hex(32), _now()))


class Ledger:
    def __init__(self, path: Path, approval: Mapping[str, Any]) -> None:
        self.path = path
        self.lock_path = Path(str(path) + ".lock")
        self.settings = policy(approval)
        self.pid = os.getpid()
        self.lock = threading.RLock()
        self.fd = -1
        self.failed = False
        try:
            _private_directory(path)
            # Separate inode: BSD/macOS can merge flock and SQLite fcntl locks.
            # The process lease must never lock SQLite's own database inode.
            self.fd = os.open(self.lock_path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
            self.lock_identity = os.fstat(self.fd)
            self.identity = path.lstat()
            self._check_file()
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.transaction() as db:
                if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise ValueError
                # Old process is gone (exclusive lock acquired). Unknown in-flight
                # spend keeps the FULL reservation. Never refund on restart.
                db.execute("UPDATE runs SET state='abandoned' WHERE state='active'")
        except (OSError, ValueError, sqlite3.Error, PilotBlocked) as exc:
            self.close()
            raise PilotBlocked(REFUSAL) from exc

    def close(self) -> None:
        with self.lock:
            if self.fd >= 0:
                os.close(self.fd)
                self.fd = -1

    def _check_file(self) -> None:
        _private_directory(self.path)
        info = self.path.lstat()
        if (os.getpid() != self.pid or self.fd < 0 or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600
                or (info.st_dev, info.st_ino) != (self.identity.st_dev, self.identity.st_ino)
                or info.st_size > 8 * 1024 * 1024):
            raise ValueError
        lease = self.lock_path.lstat()
        held = os.fstat(self.fd)
        if (not stat.S_ISREG(lease.st_mode) or lease.st_nlink != 1 or lease.st_size != 0
                or lease.st_uid != os.geteuid() or stat.S_IMODE(lease.st_mode) != 0o600
                or (lease.st_dev, lease.st_ino) != (self.lock_identity.st_dev, self.lock_identity.st_ino)
                or (held.st_dev, held.st_ino) != (lease.st_dev, lease.st_ino)):
            raise ValueError
        for suffix in ("-journal", "-wal", "-shm"):
            candidate = Path(str(self.path) + suffix)
            if candidate.exists() or candidate.is_symlink():
                entry = candidate.lstat()
                if (not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1
                        or entry.st_uid != os.geteuid() or stat.S_IMODE(entry.st_mode) & 0o077):
                    raise ValueError

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            db = None
            try:
                if self.failed:
                    raise ValueError
                self._check_file()
                db = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=2)
                db.execute("PRAGMA foreign_keys=ON")
                db.execute("PRAGMA synchronous=FULL")
                db.execute("PRAGMA secure_delete=ON")
                if db.execute("PRAGMA journal_mode").fetchone()[0] != "delete":
                    raise ValueError
                db.execute("BEGIN IMMEDIATE")
                rows = db.execute("SELECT * FROM control").fetchall()
                if len(rows) != 1:
                    raise ValueError
                version, saved, salt, charged, attempts, last = rows[0]
                now = _now()
                if (version != 1 or json.loads(saved) != self.settings or len(bytes.fromhex(salt)) != 32
                        or type(charged) is not int or not 0 <= charged <= self.settings["pilot_total_microusd"]
                        or type(attempts) is not int or not 0 <= attempts <= self.settings["pilot_total_attempts"]
                        or now < last or now >= datetime.fromisoformat(self.settings["expires_at"].replace("Z", "+00:00")).timestamp()):
                    raise ValueError
                db.execute("UPDATE control SET last_now=?", (now,))
                db.execute("DELETE FROM runs WHERE state!='active' AND started<=?", (now - 86400,))
                yield db
                db.commit()
            except (OSError, ValueError, TypeError, sqlite3.Error) as exc:
                self.failed = True
                raise PilotBlocked(REFUSAL) from exc
            finally:
                if db is not None:
                    db.close()  # Rolls back failed transactions, including limit refusals.

    def verify(self, approval: Mapping[str, Any]) -> None:
        if policy(approval) != self.settings:
            raise PilotBlocked(REFUSAL)
        with self.transaction():
            pass

    def start(self, owner: str) -> str:
        with self.transaction() as db:
            salt, charged, attempts = db.execute("SELECT salt, charged, reserved_attempts FROM control").fetchone()
            opaque = hmac.new(bytes.fromhex(salt), owner.encode(), hashlib.sha256).hexdigest()
            now = _now()
            hourly, daily = db.execute("SELECT SUM(started>?), COUNT(*) FROM runs WHERE owner=? AND started>?",
                                      (now - 3600, opaque, now - 86400)).fetchone()
            capacity = self.settings["run_attempts"]
            cost = capacity * self.settings["attempt_charge_microusd"]
            if ((hourly or 0) >= 2 or daily >= self.settings["participant_daily_starts"]
                    or db.execute("SELECT 1 FROM runs WHERE state='active'").fetchone()
                    or charged + cost > self.settings["pilot_total_microusd"]
                    or attempts + capacity > self.settings["pilot_total_attempts"]):
                raise PilotBlocked(REFUSAL)
            run = uuid.uuid4().hex
            db.execute("INSERT INTO runs VALUES (?, ?, ?, 'active', 0, 0)", (run, opaque, now))
            db.execute("UPDATE control SET charged=charged+?, reserved_attempts=reserved_attempts+?", (cost, capacity))
            return run

    def attempt(self, run: str, prompt_chars: int) -> int:
        with self.transaction() as db:
            row = db.execute("SELECT attempted, prompt_chars FROM runs WHERE id=? AND state='active'", (run,)).fetchone()
            if (row is None or type(prompt_chars) is not int or prompt_chars < 0
                    or type(row[0]) is not int or not 0 <= row[0] <= self.settings["run_attempts"]
                    or type(row[1]) is not int or not 0 <= row[1] <= self.settings["run_prompt_chars"]
                    or row[0] >= self.settings["run_attempts"]
                    or row[1] + prompt_chars > self.settings["run_prompt_chars"]):
                raise PilotBlocked(REFUSAL)
            number = int(row[0]) + 1
            db.execute("UPDATE runs SET attempted=?, prompt_chars=prompt_chars+? WHERE id=?", (number, prompt_chars, run))
            db.execute("INSERT INTO attempts VALUES (?, ?, 0, NULL, NULL)", (run, number))
            return number

    def reconcile(self, run: str, number: int, input_tokens: Any, output_tokens: Any) -> None:
        # Missing/invalid usage remains NULL. Usage is evidence, not a refund or
        # a tokenizer/price guess; every attempt retains its full approved charge.
        tokens = [v if type(v) is int and 0 <= v <= 1_000_000_000 else None
                  for v in (input_tokens, output_tokens)]
        with self.transaction() as db:
            db.execute("UPDATE attempts SET reconciled=1, input_tokens=?, output_tokens=? "
                       "WHERE run=? AND number=? AND reconciled=0", (*tokens, run, number))

    def finish(self, run: str) -> None:
        with self.transaction() as db:
            row = db.execute("SELECT attempted FROM runs WHERE id=? AND state='active'", (run,)).fetchone()
            if row is None or type(row[0]) is not int or not 0 <= row[0] <= self.settings["run_attempts"]:
                raise PilotBlocked(REFUSAL)
            unused = self.settings["run_attempts"] - row[0]
            db.execute("UPDATE control SET charged=charged-?, reserved_attempts=reserved_attempts-?",
                       (unused * self.settings["attempt_charge_microusd"], unused))
            db.execute("UPDATE runs SET state='finished' WHERE id=?", (run,))


_instance: Ledger | None = None
_instance_lock = threading.Lock()


def get_ledger(approval: Mapping[str, Any]) -> Ledger:
    global _instance
    try:
        path = Path(os.environ["VA_LSE_PILOT_BUDGET_FILE"])
        with _instance_lock:
            if _instance is None:
                _instance = Ledger(path, approval)
            if path != _instance.path:
                raise PilotBlocked(REFUSAL)
            _instance.verify(approval)
            return _instance
    except (KeyError, OSError, ValueError) as exc:
        raise PilotBlocked(REFUSAL) from exc


def main() -> None:
    """Operator-only initial provisioning, never an automatic reset path."""
    import argparse
    from .pilot import load_approval
    parser = argparse.ArgumentParser(description="Provision a NEW reviewed pilot budget ledger offline.")
    parser.add_argument("command", choices=["init"])
    parser.parse_args()
    try:
        provision(Path(os.environ["VA_LSE_PILOT_BUDGET_FILE"]), load_approval())
    except (KeyError, OSError, ValueError, sqlite3.Error, PilotBlocked):
        parser.exit(2, REFUSAL + "\n")
    print("New pilot control ledger provisioned. Actual spending-cutoff acceptance is still required.")


if __name__ == "__main__":
    main()
