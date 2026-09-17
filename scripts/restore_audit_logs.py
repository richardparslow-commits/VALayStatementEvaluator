#!/usr/bin/env python
"""Read the audit-log backup back: verify it, or restore it to disk.

The companion to ``scripts/backup_audit_logs.py``. Verify first — a backup that
has never been read back is a hypothesis, and this is the script that turns it
into an answer. See ``app/audit_restore.py`` for the mechanics and
``DEPLOYMENT.md -> Audit log retention and backup`` for the procedure.

    # is the backup intact and complete? (downloads and hashes every object)
    python scripts/restore_audit_logs.py --verify

    # same, but skip the downloads (no integrity verdict, much faster)
    python scripts/restore_audit_logs.py --verify --no-hash

    # rebuild the stream for an investigator
    python scripts/restore_audit_logs.py --restore /tmp/audit-restore

    # read from a different bucket than the deployment writes to
    python scripts/restore_audit_logs.py --verify --bucket audit-archive --prefix cold

Exit codes:

* ``0`` — verification passed, or the restore completed
* ``1`` — a misconfiguration this script can describe: nothing configured, an
  unusable setting, an unknown backend name passed programmatically
* ``2`` — the backup is reachable but incomplete (gaps), corrupt, or unreachable.
  Also argparse's standard code for an invalid flag, so ``--destination dropbox``
  exits 2 with argparse's own message rather than being silently accepted.

``--restore`` exits 2 when the stream it wrote has gaps, because a restore that
silently produces a holey record is worse than one that says so.

This script never writes to the destination: recovery must not be able to damage
the only copy.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import audit_backup, audit_restore  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402

logger = logging.getLogger("app.audit_restore.cli")

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_FAILED = 2


def _out(text: str, *, err: bool = False) -> None:
    """Write a result line, always flushed (see backup_audit_logs.py)."""
    print(text, file=sys.stderr if err else sys.stdout, flush=True)


def _build_destination(args: argparse.Namespace) -> audit_backup.BackupDestination | None:
    """Resolve the destination from config plus CLI overrides, or None if unusable.

    "Nothing is configured" and "the configured store is unreachable" are different
    failures with different responses, so they get different exit codes: the first
    is a setup problem (usage, 1), the second is a backup problem (2).
    """
    overrides = audit_restore.config_overrides_from_args(
        destination=args.destination or "",
        bucket=args.bucket or "",
        prefix=args.prefix or "",
        path=args.path or "",
        container=args.container or "",
        endpoint_url=args.endpoint_url or "",
    )
    try:
        destination = audit_backup.build_destination(overrides=overrides)
    except audit_backup.BackupError as exc:
        _out(f"audit restore is misconfigured: {exc}", err=True)
        return None
    if isinstance(destination, audit_backup.NullDestination):
        _out(
            "no backup destination is configured, so there is nothing to read back. "
            "Set VA_LSE_AUDIT_BACKUP_DESTINATION (and run "
            "scripts/backup_audit_logs.py), or pass --destination/--bucket/--path.",
            err=True,
        )
        return None
    return destination


def _verify(args: argparse.Namespace) -> int:
    destination = _build_destination(args)
    if destination is None:
        return EXIT_USAGE

    report = audit_restore.verify_backup(
        destination, hash_objects=not args.no_hash, check_lines=not args.no_hash
    )
    if args.json:
        _out(json.dumps(report.to_json(), indent=2, sort_keys=True))
    else:
        _out(audit_restore.render_verify_text(report))
    if not report.listed:
        return EXIT_FAILED if report.list_error else EXIT_USAGE
    return EXIT_OK if not report.needs_attention else EXIT_FAILED


def _restore(args: argparse.Namespace) -> int:
    destination = _build_destination(args)
    if destination is None:
        return EXIT_USAGE

    report = audit_restore.restore_backup(
        destination,
        target=args.restore,
        hash_objects=not args.no_hash,
        verify=not args.no_verify,
        force=args.force,
    )
    if args.json:
        _out(json.dumps(report.to_json(), indent=2, sort_keys=True))
    else:
        _out(audit_restore.render_restore_text(report))
        if report.verify is not None and report.verify.needs_attention:
            _out("")
            _out(audit_restore.render_verify_text(report.verify))

    if report.error:
        # "Refused on purpose" (a stale target directory) is a usage problem to fix
        # and re-run; anything else is a broken backup.
        return EXIT_USAGE if report.refused else EXIT_FAILED
    if report.verify is not None and report.verify.needs_attention:
        # Covers gaps, corrupt/unreadable objects, and generations whose order could
        # not be established. A restore that is holey or mis-ordered must not exit 0.
        return EXIT_FAILED
    return EXIT_OK


def _status(args: argparse.Namespace) -> int:
    """Config-only view of what a restore would use (no network)."""
    payload = {
        "restore": audit_restore.audit_integrity_health(),
        "backup": audit_backup.audit_backup_health(),
        "resolved": {
            "log_dir": str(audit_backup.audit_log_dir()),
            "state_file": str(audit_backup.state_path()),
        },
    }
    del args
    _out(json.dumps(payload, indent=2, sort_keys=True))
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify or restore the backed-up audit log.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Verification downloads every object and checks it against the hash embedded "
            "in its own key, then walks the byte ranges to find holes. Restoring writes "
            "restored.jsonl, any rotated snapshots, and a manifest.json describing both."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--verify", action="store_true", help="check integrity and coverage, change nothing (default)"
    )
    mode.add_argument(
        "--restore",
        metavar="DIR",
        help="write the audit stream back into DIR (restored.jsonl + rotated/ + manifest.json)",
    )
    mode.add_argument("--status", action="store_true", help="print config and exit; no network")

    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help=(
            "skip downloading objects. Much faster, but there is then no integrity "
            "verdict and no gap analysis is possible from content — only keys."
        ),
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="with --restore, skip the verification pass before writing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="with --restore, replace an existing restored.jsonl instead of refusing",
    )

    overrides = parser.add_argument_group(
        "destination overrides", "read from a different store than the deployment writes to"
    )
    overrides.add_argument("--destination", choices=["filesystem", "s3", "gcs", "azure"])
    overrides.add_argument("--bucket", help="S3 or GCS bucket name")
    overrides.add_argument("--prefix", help="key prefix inside the bucket or container")
    overrides.add_argument("--path", help="root directory for a filesystem destination")
    overrides.add_argument("--container", help="Azure blob container name")
    overrides.add_argument("--endpoint-url", help="S3-compatible endpoint (MinIO, R2, Spaces)")

    args = parser.parse_args(argv)
    configure_logging()

    if args.status:
        return _status(args)
    if args.restore:
        return _restore(args)
    return _verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
