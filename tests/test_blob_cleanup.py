"""Independent, synthetic-data retention passes and visible failure semantics."""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tests import hermetic  # noqa: F401
from app import blob_cleanup
from app.blob_store import BlobStoreError, FilesystemBlobStore, NullBlobStore


class TestBlobCleanup(unittest.TestCase):
    def test_idle_store_expires_without_another_upload(self):
        with TemporaryDirectory() as tmp:
            store = FilesystemBlobStore(tmp, sweep_age_seconds=60)
            expired = store.put(b"expired synthetic canary")
            fresh = store.put(b"fresh synthetic canary")
            old = time.time() - 120
            os.utime(Path(tmp) / expired.key, (old, old))
            output = io.StringIO()
            with patch.object(blob_cleanup, "build_blob_store", return_value=store), \
                 contextlib.redirect_stdout(output):
                self.assertEqual(blob_cleanup.main(["--dry-run"]), 0)
                self.assertTrue((Path(tmp) / expired.key).exists())
                self.assertEqual(blob_cleanup.main([]), 0)
            self.assertFalse((Path(tmp) / expired.key).exists())
            self.assertEqual(store.get(fresh), b"fresh synthetic canary")
            reports = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual([r["expired_files"] for r in reports], [1, 1])
            self.assertEqual([r["dry_run"] for r in reports], [True, False])
            self.assertNotIn(expired.key, output.getvalue())
            self.assertNotIn("canary", output.getvalue())

    def test_unconfigured_or_non_filesystem_backend_fails(self):
        with patch.object(blob_cleanup, "build_blob_store", return_value=NullBlobStore()), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(blob_cleanup.main([]), 2)

    def test_missing_mount_fails_instead_of_reporting_success(self):
        with TemporaryDirectory() as tmp:
            store = FilesystemBlobStore(Path(tmp) / "missing")
            with patch.object(blob_cleanup, "build_blob_store", return_value=store), \
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(blob_cleanup.main([]), 2)
            self.assertFalse(store.root.exists())

    def test_permission_and_lock_failures_are_visible_without_raw_errors(self):
        for error in (OSError("private-path"), BlobStoreError("private-key")):
            with TemporaryDirectory() as tmp:
                store = FilesystemBlobStore(tmp)
                errors = io.StringIO()
                with patch.object(blob_cleanup, "build_blob_store", return_value=store), \
                     patch.object(store, "sweep", side_effect=error), \
                     contextlib.redirect_stderr(errors):
                    self.assertEqual(blob_cleanup.main([]), 2)
                self.assertIn("failed", errors.getvalue())
                self.assertNotIn("private", errors.getvalue())

    def test_delete_failure_is_not_counted_as_success(self):
        with TemporaryDirectory() as tmp:
            store = FilesystemBlobStore(tmp, sweep_age_seconds=60)
            ref = store.put(b"old synthetic record")
            path = Path(tmp) / ref.key
            old = time.time() - 120
            os.utime(path, (old, old))
            with patch.object(Path, "unlink", side_effect=PermissionError("denied")):
                with self.assertRaises(BlobStoreError):
                    store.sweep()
            self.assertTrue(path.exists())

    def test_directory_scan_failure_is_not_reported_as_empty_success(self):
        with TemporaryDirectory() as tmp:
            store = FilesystemBlobStore(tmp)
            store.put(b"synthetic record")
            with patch.object(os, "scandir", side_effect=PermissionError("private mount")):
                with self.assertRaises(BlobStoreError):
                    store.sweep()

    def test_actual_cli_cleans_an_idle_store(self):
        with TemporaryDirectory() as tmp:
            store = FilesystemBlobStore(tmp, sweep_age_seconds=60)
            ref = store.put(b"CLI synthetic canary")
            old = time.time() - 120
            os.utime(Path(tmp) / ref.key, (old, old))
            env = dict(os.environ, VA_LSE_BLOB_STORE="filesystem", VA_LSE_BLOB_DIR=tmp,
                       VA_LSE_JOB_QUEUE_TTL_SECONDS="60")
            result = subprocess.run([sys.executable, "-m", "app.blob_cleanup"], env=env,
                                    cwd=Path(__file__).resolve().parent.parent,
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), {"dry_run": False, "expired_files": 1,
                                                        "retention_seconds": 60})
            self.assertFalse((Path(tmp) / ref.key).exists())

    def test_symlinked_namespace_cannot_delete_outside_the_store(self):
        with TemporaryDirectory() as tmp, TemporaryDirectory() as outside:
            target = FilesystemBlobStore(outside)
            ref = target.put(b"outside synthetic record")
            old = time.time() - 120
            os.utime(Path(outside) / ref.key, (old, old))
            (Path(tmp) / "blobs").symlink_to(Path(outside) / "blobs", target_is_directory=True)
            store = FilesystemBlobStore(tmp, sweep_age_seconds=60)
            with self.assertRaises(BlobStoreError):
                store.sweep()
            with self.assertRaises(BlobStoreError):
                store.put(b"outside synthetic record")
            self.assertEqual(target.get(ref), b"outside synthetic record")

    def test_lock_symlink_fails_closed(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "foreign-lock"
            target.write_bytes(b"keep")
            (Path(tmp) / ".blob-store.lock").symlink_to(target)
            with self.assertRaises(BlobStoreError):
                FilesystemBlobStore(tmp).put(b"synthetic record")
            self.assertEqual(target.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
