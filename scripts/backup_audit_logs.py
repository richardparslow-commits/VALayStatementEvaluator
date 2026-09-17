#!/usr/bin/env python
"""Ship the audit log off-pod and enforce its retention window.

Run this as a Kubernetes CronJob (one pass per invocation, the recommended shape)
or as a sidecar with ``--loop``. See ``app/audit_backup.py`` for the mechanics and
``DEPLOYMENT.md -> Audit log retention and backup`` for the operating procedure.

    # one pass (CronJob)
    python scripts/backup_audit_logs.py --once

    # sidecar
    python scripts/backup_audit_logs.py --loop --interval-hours 6

    # what does the operator-facing state look like right now?
    python scripts/backup_audit_logs.py --status

Exit codes (so a CronJob's failure is visible in ``kubectl get jobs``):

* ``0`` — pass completed, or backup is not configured and not required
* ``1`` — bad usage / unusable configuration
* ``2`` — a pass ran and failed (upload error, destination unreachable)

This script never raises a traceback at the operator: a failed backup must be a
readable one-line diagnosis, because it is read at 3am by whoever is on call.
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import audit, audit_backup, config  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402

logger = logging.getLogger("app.audit_backup.cli")

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_FAILED = 2

_stop = threading.Event()


def _out(text: str, *, err: bool = False) -> None:
    """Write a result line, always flushed.

    stdout is block-buffered when it is not a tty (i.e. exactly how a container
    runs this), so an unflushed line is lost if the process is SIGKILLed — which is
    what ``activeDeadlineSeconds`` does. For a log-shipping job the output *is* the
    deliverable, so it never sits in a buffer.
    """
    print(text, file=sys.stderr if err else sys.stdout, flush=True)


def _install_stop_handler() -> None:
    """Stop after the current pass on SIGTERM (sidecar/`docker stop`)."""

    def _handler(signum: int, _frame: object) -> None:
        name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        logger.warning("received %s: finishing the current pass, then exiting", name)
        _stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):  # pragma: no cover - not the main thread
            logger.debug("could not install a handler for %s", sig)


def _print_status() -> int:
    """Print the same payload ``/health`` exposes (state file, no network)."""
    payload = {
        "audit": audit.audit_health(),
        "audit_backup": audit_backup.audit_backup_health(),
        "disk": audit_backup.disk_status(),
        "resolved": {
            "log_dir": str(audit_backup.audit_log_dir()),
            "state_file": str(audit_backup.state_path()),
            "rotated_files": [p.name for p in audit_backup.rotated_audit_files()],
        },
    }
    _out(json.dumps(payload, indent=2, sort_keys=True))
    return EXIT_OK


def _one_pass(*, dry_run: bool, prune: bool, required: bool) -> int:
    """Run a single backup pass and map the outcome onto an exit code."""
    with audit_backup.BackupLock() as lock:  # noqa: F841 - held for the pass
        result = audit_backup.run_backup(dry_run=dry_run, prune_remote=prune)
    _report(result, dry_run=dry_run)
    if result.error:
        if not result.configured and not required:
            return EXIT_OK
        return EXIT_FAILED
    if not result.configured:
        if required:
            _out(
                "audit backup is required (VA_LSE_AUDIT_BACKUP_REQUIRED=1) but "
                "VA_LSE_AUDIT_BACKUP_DESTINATION is unset",
                err=True,
            )
            return EXIT_FAILED
        _out(
            "audit backup is not configured (VA_LSE_AUDIT_BACKUP_DESTINATION) — nothing to ship. "
            "Audit logs remain on this volume only"
            + (
                f"; local retention still ran ({result.pruned_local} file(s) expired)."
                if result.pruned_local
                else "."
            ),
            err=True,
        )
    return EXIT_OK


def _report(result: audit_backup.BackupResult, *, dry_run: bool) -> None:
    prefix = "[dry-run] " if dry_run else ""
    if result.error:
        _out(f"{prefix}audit backup FAILED: {result.error}", err=True)
        return
    if not result.configured:
        return
    _out(
        f"{prefix}audit backup ok: destination={result.destination} "
        f"uploaded={result.uploaded} object(s) ({result.uploaded_bytes:,} bytes) "
        f"already_shipped={result.skipped_rotated} "
        f"local_pruned={result.pruned_local} remote_pruned={result.pruned_remote}"
    )
    for line in result.detail:
        _out(f"{prefix}  {line}")
    state = audit_backup.load_state()
    _out(
        f"{prefix}  pending={audit_backup.pending_bytes(state):,} bytes "
        f"(would be lost if the pod died now)"
    )


def _loop(*, interval_hours: float, dry_run: bool, prune: bool, required: bool) -> int:
    interval = max(60.0, interval_hours * 3600)
    logger.info("audit backup loop starting (every %.1fh)", interval_hours)
    exit_code = EXIT_OK
    while not _stop.is_set():
        try:
            exit_code = _one_pass(dry_run=dry_run, prune=prune, required=required)
        except Exception as exc:  # noqa: BLE001 - a bad pass must not kill the loop
            logger.error("backup pass raised %s: %s", type(exc).__name__, exc)
            exit_code = EXIT_FAILED
        # Wait in short slices so SIGTERM is honoured promptly.
        deadline = time.monotonic() + interval
        while not _stop.is_set() and time.monotonic() < deadline:
            _stop.wait(1.0)
    logger.info("audit backup loop stopped")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Back up the audit log off-pod and enforce its retention window.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="run a single pass (default) and exit")
    mode.add_argument("--loop", action="store_true", help="run continuously (sidecar mode)")
    mode.add_argument("--status", action="store_true", help="print state and exit; uploads nothing")
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=None,
        help=f"hours between passes in --loop mode (default {config.AUDIT_BACKUP_INTERVAL_HOURS:g})",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help=(
            "also delete destination objects older than "
            f"VA_LSE_AUDIT_BACKUP_CLOUD_RETENTION_DAYS "
            f"({config.AUDIT_BACKUP_CLOUD_RETENTION_DAYS} days). A bucket lifecycle "
            "rule is the better instrument; this is for stores that cannot set one."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="report what would be shipped, change nothing")
    parser.add_argument(
        "--require-destination",
        action="store_true",
        help="exit 2 when no destination is configured (CronJobs should pass this)",
    )
    args = parser.parse_args(argv)

    configure_logging()
    if args.status:
        return _print_status()

    interval_hours = args.interval_hours if args.interval_hours is not None else config.AUDIT_BACKUP_INTERVAL_HOURS
    if interval_hours <= 0:
        _out("--interval-hours must be positive", err=True)
        return EXIT_USAGE

    required = args.require_destination or config.AUDIT_BACKUP_REQUIRED
    try:
        if args.loop:
            _install_stop_handler()
            return _loop(
                interval_hours=interval_hours, dry_run=args.dry_run, prune=args.prune, required=required
            )
        return _one_pass(dry_run=args.dry_run, prune=args.prune, required=required)
    except audit_backup.BackupError as exc:
        # Configuration problems (missing bucket, backend package absent, another
        # backup process holding the lock) are a usage error, not a data failure.
        _out(f"audit backup is misconfigured: {exc}", err=True)
        return EXIT_USAGE
    except KeyboardInterrupt:  # pragma: no cover - interactive use
        return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
