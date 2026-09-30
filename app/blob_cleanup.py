"""One filesystem retention pass, independent of web/worker traffic.

Run ``python -m app.blob_cleanup --dry-run`` before enabling a schedule. This
command does not manage S3 lifecycle, backups, downloads, or provider copies.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import config
from .blob_store import BlobStoreError, FilesystemBlobStore, build_blob_store
from .job_queue import JobQueueError, build_job_backend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="count expired files without deleting them")
    args = parser.parse_args(argv)
    try:
        store = build_blob_store()
        if not isinstance(store, FilesystemBlobStore):
            raise BlobStoreError("cleanup requires the configured filesystem blob backend")
        if not store.root.is_dir():
            raise BlobStoreError("configured blob directory is missing; check the shared mount")
        backend = build_job_backend(require_distributed=True)
        removed = store.sweep(dry_run=args.dry_run, retained_keys=backend.retained_blob_keys)
    except (BlobStoreError, JobQueueError, OSError):
        # Do not emit document keys, filesystem paths, credentials, or raw errors.
        print("filesystem blob cleanup failed; check queue configuration, backend, mount, permissions and locking", file=sys.stderr)
        return 2
    print(json.dumps({"dry_run": args.dry_run, "expired_files": removed,
                      "retention_seconds": config.JOB_QUEUE_TTL_SECONDS}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
