"""Single-process pilot log age limits, including active files and idle periods.

Whole files expire when their first event expires. Size rotation can remove them
sooner: this is a maximum retention, not a promise to preserve every event. The
writer's own lock protects cleanup; no external sweeper may race these files.
"""
from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import threading
import time
from datetime import datetime
from typing import Any
import weakref

SWEEP_SECONDS = 60
_handlers: weakref.WeakSet[PilotLogHandler] = weakref.WeakSet()
_registry_lock = threading.Lock()
_thread: threading.Thread | None = None


def retention_days() -> int:
    """No implicit real-data policy: the operator must select 1–30 days."""
    raw = os.getenv("VA_LSE_PILOT_LOG_RETENTION_DAYS", "").strip()
    if not raw.isdecimal() or not 1 <= int(raw) <= 30:
        raise ValueError("Pilot local log retention requires an explicit 1–30 day policy.")
    return int(raw)


def _first_event(path: Path, now: float) -> float | None:
    try:
        with path.open(encoding="utf-8") as file:
            line = file.readline(4097)
        if not line:
            return now  # Empty files hold no retained events.
        if len(line) > 4096 or not line.endswith("\n"):
            return None
        data = json.loads(line)
        stamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp.timestamp() > now:
            return None
        return stamp.timestamp()
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        # OSError must remain visible, rather than silently admitting a broken
        # retention sink. Malformed/legacy content has no provable age.
        if not path.exists():
            return now
        with path.open(encoding="utf-8"):
            pass
        return None


class PilotLogHandler(RotatingFileHandler):
    """Keep numbered size rotations, plus a first-event age ceiling."""

    def __init__(self, filename: str, *, maxBytes: int, backupCount: int,
                 encoding: str = "utf-8", days: int | None = None) -> None:
        self.days = retention_days() if days is None else days
        if not 1 <= self.days <= 30 or maxBytes <= 0 or backupCount < 1:
            raise ValueError("Pilot log bounds must be positive.")
        self.last_sweep: float | None = None
        self.failed = False
        self.write_failed = False
        self._closed_retention = False
        self._oldest: float | None = None
        super().__init__(filename, maxBytes=maxBytes, backupCount=backupCount,
                         encoding=encoding, delay=True)
        self.prune()
        global _thread
        with _registry_lock:
            _handlers.add(self)
            if _thread is None or not _thread.is_alive():
                _thread = threading.Thread(target=_sweep_loop, name="pilot-log-retention", daemon=True)
                _thread.start()

    def prune(self, *, now: float | None = None) -> None:
        self.acquire()
        try:
            if self._closed_retention:
                return
            instant = time.time() if now is None else now
            path = Path(self.baseFilename)
            # Include old numbered rotations outside the current backup count.
            candidates = [path, *(p for p in path.parent.glob(path.name + ".*")
                                  if p.name[len(path.name) + 1:].isdigit())]
            for candidate in candidates:
                if not candidate.exists():
                    continue
                first = self._oldest if candidate == path and self._oldest is not None else _first_event(candidate, instant)
                if first is None or first <= instant - self.days * 86400:
                    if candidate == path:
                        if self.stream is not None:
                            self.stream.close()
                            self.stream = None
                        self._oldest = None
                    candidate.unlink()
                elif candidate == path:
                    self._oldest = first if candidate.stat().st_size else None
            self.last_sweep = instant
            self.failed = False
        except OSError:
            self.failed = True
            raise
        finally:
            self.release()

    def emit(self, record: logging.LogRecord) -> None:
        # Handler.handle normally holds this lock; acquire also protects direct
        # emit callers and the idle sweeper. RLock is intentional.
        self.acquire()
        try:
            self.prune()
            if self.shouldRollover(record):
                self.doRollover()
            logging.FileHandler.emit(self, record)
            if self._oldest is None:
                self._oldest = _first_event(Path(self.baseFilename), time.time())
        except Exception:
            self.failed = True
            self.handleError(record)
        finally:
            self.release()

    def doRollover(self) -> None:
        super().doRollover()
        self._oldest = None

    def verify_sink(self) -> None:
        """Prove the configured file opens before case admission, not lazily."""
        self.acquire()
        try:
            if self._closed_retention:
                raise OSError("Pilot log sink is closed.")
            if self.stream is None:
                self.stream = self._open()
            self.stream.flush()
        except OSError:
            self.write_failed = True
            raise
        finally:
            self.release()

    def _open(self) -> Any:
        # The log volume belongs to the operator. Refuse symlinks and restrict
        # readable files rather than relying on the deployment's ambient umask.
        fd = os.open(self.baseFilename, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            os.fchmod(fd, 0o600)
            return os.fdopen(fd, self.mode, encoding=self.encoding, errors=self.errors)
        except BaseException:
            os.close(fd)
            raise

    def handleError(self, record: logging.LogRecord) -> None:
        self.failed = True
        self.write_failed = True
        # Fixed message only: logging's default stderr includes caller text and
        # exception details. Health/admission exposes failure without content.

    def close(self) -> None:
        self.acquire()
        try:
            self._closed_retention = True
            with _registry_lock:
                _handlers.discard(self)
            super().close()
        finally:
            self.release()


def sweep_registered() -> None:
    with _registry_lock:
        handlers = list(_handlers)
    for handler in handlers:
        try:
            handler.prune()
        except OSError:
            pass  # failed flag makes the failure observable and blocks admission


def _sweep_loop() -> None:
    waiter = threading.Event()
    while not waiter.wait(SWEEP_SECONDS):
        sweep_registered()


def retention_health() -> dict[str, Any]:
    with _registry_lock:
        handlers = list(_handlers)
        running = _thread is not None and _thread.is_alive()
    return {"active": running, "sweep_seconds": SWEEP_SECONDS,
            "streams": len(handlers), "failed": any(h.failed or h.write_failed for h in handlers),
            "stale": any(h.last_sweep is None or time.time() - h.last_sweep > 2 * SWEEP_SECONDS for h in handlers)}
