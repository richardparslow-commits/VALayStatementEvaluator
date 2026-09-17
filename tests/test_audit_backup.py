"""Tests for audit log retention/backup (app/audit_backup.py, app/audit.py).

The load-bearing behaviours are the ones that decide whether a pod restart loses
audit records:

* the live ``audit.log`` window is shipped by byte watermark, cut to the last
  complete line, so a 10 MiB file that will not rotate for weeks is still off-pod;
* a retry cannot duplicate an object (content+range-addressed keys);
* rotation resets the watermark instead of skipping the new file's head;
* a hard destination failure is recorded in state and surfaced, not raised.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
import unittest
from pathlib import Path

from app import audit_backup, config


class _BackupCase(unittest.TestCase):
    """Base case: a private log dir + filesystem destination, isolated per test."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.logs = self.root / "logs"
        self.logs.mkdir()
        self.dest_root = self.root / "offsite"
        self._saved = {
            name: getattr(config, name)
            for name in (
                "AUDIT_LOG_DIR",
                "AUDIT_LOG_FILE",
                "AUDIT_LOG_MAX_BYTES",
                "AUDIT_LOG_BACKUPS",
                "AUDIT_RETENTION_DAYS",
                "AUDIT_BACKUP_DESTINATION",
                "AUDIT_BACKUP_DIR",
                "AUDIT_BACKUP_STATE_FILE",
                "AUDIT_BACKUP_INTERVAL_HOURS",
                "AUDIT_BACKUP_CLOUD_RETENTION_DAYS",
                "AUDIT_BACKUP_S3_BUCKET",
                "AUDIT_BACKUP_GCS_BUCKET",
                "AUDIT_BACKUP_AZURE_CONTAINER",
                "AUDIT_BACKUP_AZURE_ACCOUNT_URL",
                "AUDIT_ERROR_MESSAGES",
                "RUN_LOG_MAX_BYTES",
                "RUN_LOG_BACKUPS",
                "DISK_MIN_FREE_BYTES",
            )
        }
        config.AUDIT_LOG_DIR = str(self.logs)
        config.AUDIT_LOG_FILE = "audit.log"
        config.AUDIT_BACKUP_DESTINATION = "filesystem"
        config.AUDIT_BACKUP_DIR = str(self.dest_root)
        config.AUDIT_RETENTION_DAYS = 7
        config.AUDIT_BACKUP_CLOUD_RETENTION_DAYS = 90

    def tearDown(self) -> None:
        for name, value in self._saved.items():
            setattr(config, name, value)
        self._tmp.cleanup()

    # -- helpers -----------------------------------------------------------
    def live(self) -> Path:
        return self.logs / "audit.log"

    def append(self, text: str) -> None:
        with self.live().open("a", encoding="utf-8") as handle:
            handle.write(text)

    def rotate(self) -> None:
        """Mimic RotatingFileHandler: audit.log -> audit.log.1, .1 -> .2, …."""
        for index in range(config.AUDIT_LOG_BACKUPS, 0, -1):
            src = self.logs / f"audit.log.{index}"
            dst = self.logs / f"audit.log.{index + 1}"
            if src.exists():
                if index == config.AUDIT_LOG_BACKUPS:
                    src.unlink()
                else:
                    src.replace(dst)
        if self.live().exists():
            self.live().replace(self.logs / "audit.log.1")

    def dest(self) -> audit_backup.FilesystemDestination:
        return audit_backup.FilesystemDestination(self.dest_root)

    def objects(self) -> list[str]:
        return sorted(p.relative_to(self.dest_root).as_posix() for p in self.dest_root.rglob("*.jsonl"))

    def line(self, index: int) -> str:
        return json.dumps({"request_id": f"req_{index}", "status": "ok"}) + "\n"


class TestDestinationSelection(_BackupCase):
    def test_unconfigured_is_a_noop_not_an_error(self) -> None:
        config.AUDIT_BACKUP_DESTINATION = ""
        result = audit_backup.run_backup()
        self.assertFalse(result.configured)
        self.assertTrue(result.ok)
        self.assertEqual(result.uploaded, 0)

    def test_unknown_destination_reports_misconfiguration(self) -> None:
        config.AUDIT_BACKUP_DESTINATION = "dropbox"
        result = audit_backup.run_backup()
        self.assertTrue(result.configured)
        self.assertIn("dropbox", result.error)

    def test_filesystem_destination_requires_a_dir(self) -> None:
        config.AUDIT_BACKUP_DIR = ""
        result = audit_backup.run_backup()
        self.assertIn("VA_LSE_AUDIT_BACKUP_DIR", result.error)

    def test_same_volume_destination_is_reported_not_refused(self) -> None:
        config.AUDIT_BACKUP_DIR = str(self.logs / "backup")
        destination = audit_backup.build_destination()
        self.assertIsInstance(destination, audit_backup.FilesystemDestination)
        self.assertFalse(destination.off_pod)
        self.assertTrue(destination.describe()["same_volume"])

    def test_a_separate_dir_is_treated_as_off_pod(self) -> None:
        destination = audit_backup.build_destination()
        self.assertTrue(destination.off_pod)
        self.assertFalse(destination.describe()["same_volume"])


class TestLiveWindow(_BackupCase):
    def test_ships_the_live_file_without_waiting_for_rotation(self) -> None:
        """The point of the whole module: a file that never rotates still leaves."""
        for i in range(3):
            self.append(self.line(i))
        result = audit_backup.run_backup(destination=self.dest())
        self.assertTrue(result.ok)
        self.assertEqual(result.uploaded, 1)
        uploaded = self.objects()
        self.assertEqual(len(uploaded), 1)
        # The key carries the file generation between the name and the range, so a
        # restore can tell this file's windows from the next file's.
        self.assertRegex(uploaded[0], r"live-audit\.log-g[0-9a-f]+-0-\d+-[0-9a-f]{12}\.jsonl$")
        self.assertEqual((self.dest_root / uploaded[0]).read_text().count("\n"), 3)

    def test_second_pass_ships_only_new_bytes(self) -> None:
        self.append(self.line(0))
        audit_backup.run_backup(destination=self.dest())
        self.append(self.line(1))
        result = audit_backup.run_backup(destination=self.dest())
        self.assertEqual(result.uploaded, 1)
        self.assertEqual(len(self.objects()), 2)
        # The second window must start exactly where the first ended — no gap
        # (lost records) and no overlap (duplicate records).
        windows = sorted(self._live_windows())
        self.assertEqual(len(windows), 2)
        (first_start, first_end), (second_start, second_end) = windows
        self.assertEqual(first_start, 0)
        self.assertEqual(second_start, first_end)
        self.assertGreater(second_end, second_start)

    def _live_windows(self) -> list[tuple[int, int]]:
        """Parse ``(start, end)`` byte ranges out of live-window object names."""
        out: list[tuple[int, int]] = []
        for name in self.objects():
            match = re.search(
                r"live-audit\.log-g[0-9a-f]+-(\d+)-(\d+)-[0-9a-f]+\.jsonl$", name
            )
            if match:
                out.append((int(match.group(1)), int(match.group(2))))
        return out

    def test_no_new_bytes_means_no_upload(self) -> None:
        self.append(self.line(0))
        audit_backup.run_backup(destination=self.dest())
        second = audit_backup.run_backup(destination=self.dest())
        self.assertEqual(second.uploaded, 0)
        self.assertEqual(len(self.objects()), 1)

    def test_partial_trailing_line_is_never_uploaded(self) -> None:
        self.append(self.line(0))
        self.append('{"request_id": "req_half"')  # writer mid-line
        result = audit_backup.run_backup(destination=self.dest())
        self.assertEqual(result.uploaded, 1)
        content = (self.dest_root / self.objects()[0]).read_text()
        self.assertIn("req_0", content)
        self.assertNotIn("req_half", content)

    def test_the_partial_line_is_shipped_once_it_is_complete(self) -> None:
        self.append('{"request_id": "req_half"')
        audit_backup.run_backup(destination=self.dest())
        self.assertEqual(self.objects(), [])
        self.append("}\n")
        result = audit_backup.run_backup(destination=self.dest())
        self.assertEqual(result.uploaded, 1)
        self.assertIn("req_half", (self.dest_root / self.objects()[0]).read_text())

    def test_retry_after_a_crash_overwrites_the_same_object(self) -> None:
        """At-least-once delivery must not become duplicates."""
        self.append(self.line(0))
        state = audit_backup.load_state()
        audit_backup.run_backup(destination=self.dest(), state=state)
        before = self.objects()
        # Simulate a crash after upload but before the checkpoint was persisted:
        # the window is shipped again from the same watermark.
        stale = audit_backup.BackupState()
        result = audit_backup.run_backup(destination=self.dest(), state=stale)
        self.assertEqual(result.uploaded, 1)
        self.assertEqual(self.objects(), before)

    def test_rotation_resets_the_watermark_and_ships_the_new_file(self) -> None:
        self.append(self.line(0))
        audit_backup.run_backup(destination=self.dest())
        self.rotate()
        self.append(self.line(1))  # fresh audit.log, smaller than the watermark
        result = audit_backup.run_backup(destination=self.dest())
        self.assertTrue(result.ok)
        # One object for the rotated file, one for the new live window starting at 0.
        windows = [o for o in self.objects() if "live-audit.log-g" in o]
        self.assertEqual(len(windows), 2, self.objects())
        # The two windows are both "0-N" ranges but from *different files*. Without
        # the generation tag in the key a restore cannot tell that apart from a
        # retry of one window, so assert the tags actually differ.
        tags = {
            re.search(r"live-audit\.log-g([0-9a-f]+)-", name).group(1)  # type: ignore[union-attr]
            for name in windows
        }
        self.assertEqual(len(tags), 2, windows)

    def test_truncated_file_does_not_skip_content(self) -> None:
        long_line = json.dumps({"request_id": "req_long", "pad": "x" * 500}) + "\n"
        self.append(long_line)
        audit_backup.run_backup(destination=self.dest())
        # File rewritten smaller in place (log vacuum, not rotation).
        self.live().write_text(self.line(1), encoding="utf-8")
        result = audit_backup.run_backup(destination=self.dest())
        self.assertTrue(result.ok)
        contents = "".join((self.dest_root / o).read_text() for o in self.objects())
        self.assertIn("req_1", contents)


class TestRotatedFiles(_BackupCase):
    def test_rotated_file_is_uploaded_then_not_re_uploaded(self) -> None:
        self.append(self.line(0))
        self.rotate()
        first = audit_backup.run_backup(destination=self.dest())
        self.assertEqual(first.uploaded, 1)
        self.assertIn("rotated-audit.log.1-", self.objects()[0])
        second = audit_backup.run_backup(destination=self.dest())
        self.assertEqual(second.uploaded, 0)
        self.assertEqual(second.skipped_rotated, 1)

    def test_rotated_files_ship_oldest_first(self) -> None:
        self.append(self.line(0))
        self.rotate()
        self.append(self.line(1))
        self.rotate()
        self.append(self.line(2))
        result = audit_backup.run_backup(destination=self.dest())
        self.assertEqual(result.uploaded, 3)
        # Assert on the *order of the pass* (result.detail), not on a sorted
        # listing: sorting by path destroys exactly the ordering under test.
        # .2 holds the oldest data and must be shipped before .1, because rotation
        # deletes from the high end and a backlogged destination would otherwise
        # lose the oldest records first.
        order = [line for line in result.detail if line.startswith("rotated")]
        self.assertIn("rotated-audit.log.2", order[0])
        self.assertIn("rotated-audit.log.1", order[1])
        self.assertEqual(len(order), 2)

    def test_a_regenerated_file_with_new_content_is_uploaded_again(self) -> None:
        self.append(self.line(0))
        self.rotate()
        audit_backup.run_backup(destination=self.dest())
        # audit.log.1 is reused by the rotation scheme; new content must not be
        # mistaken for the old file just because the name matches.
        (self.logs / "audit.log.1").write_text(self.line(99), encoding="utf-8")
        result = audit_backup.run_backup(destination=self.dest())
        self.assertEqual(result.uploaded, 1)


class TestRetention(_BackupCase):
    def test_local_sweep_removes_old_rotated_files_only(self) -> None:
        self.append(self.line(0))
        self.rotate()
        self.append(self.line(1))  # the post-rotation live file
        old = self.logs / "audit.log.1"
        stale = time.time() - 9 * 86400
        os.utime(old, (stale, stale))
        removed = audit_backup.sweep_local_retention()
        self.assertEqual(removed, 1)
        self.assertFalse(old.exists())
        self.assertTrue(self.live().exists(), "the live audit log must never be swept")

    def test_local_sweep_keeps_files_inside_the_window(self) -> None:
        self.append(self.line(0))
        self.rotate()
        self.assertEqual(audit_backup.sweep_local_retention(), 0)
        self.assertTrue((self.logs / "audit.log.1").exists())

    def test_retention_zero_disables_the_sweep(self) -> None:
        self.append(self.line(0))
        self.rotate()
        old = self.logs / "audit.log.1"
        stale = time.time() - 400 * 86400
        os.utime(old, (stale, stale))
        self.assertEqual(audit_backup.sweep_local_retention(retention_days=0), 0)
        self.assertTrue(old.exists())

    def test_prune_removes_expired_remote_objects(self) -> None:
        self.append(self.line(0))
        audit_backup.run_backup(destination=self.dest())
        target = self.dest_root / self.objects()[0]
        stale = time.time() - 120 * 86400
        os.utime(target, (stale, stale))
        self.assertEqual(audit_backup.prune_cloud_retention(self.dest()), 1)
        self.assertEqual(self.objects(), [])

    def test_prune_spares_fresh_objects(self) -> None:
        self.append(self.line(0))
        audit_backup.run_backup(destination=self.dest())
        self.assertEqual(audit_backup.prune_cloud_retention(self.dest()), 0)
        self.assertEqual(len(self.objects()), 1)

    def test_run_backup_prunes_only_when_asked(self) -> None:
        self.append(self.line(0))
        audit_backup.run_backup(destination=self.dest())
        target = self.dest_root / self.objects()[0]
        stale = time.time() - 120 * 86400
        os.utime(target, (stale, stale))
        result = audit_backup.run_backup(destination=self.dest())
        self.assertEqual(result.pruned_remote, 0)
        self.assertEqual(len(self.objects()), 1)

    def test_local_retention_runs_even_with_no_destination(self) -> None:
        """Retention is not conditional on having somewhere to ship logs.

        Returning early when unconfigured silently disabled the retention policy
        in the default deployment, which is the one place it matters for keeping
        the log volume bounded.
        """
        config.AUDIT_BACKUP_DESTINATION = ""
        self.append(self.line(0))
        self.rotate()
        old = self.logs / "audit.log.1"
        stale = time.time() - 9 * 86400
        os.utime(old, (stale, stale))
        result = audit_backup.run_backup()
        self.assertFalse(result.configured)
        self.assertEqual(result.pruned_local, 1)
        self.assertFalse(old.exists())

    def test_dry_run_changes_nothing(self) -> None:
        self.append(self.line(0))
        result = audit_backup.run_backup(destination=self.dest(), dry_run=True)
        self.assertEqual(result.uploaded, 1)
        self.assertEqual(self.objects(), [])
        self.assertEqual(audit_backup.load_state().live_offset, 0)


class TestFailureHandling(_BackupCase):
    class _ExplodingDestination(audit_backup.BackupDestination):
        name = "boom"
        off_pod = True

        def put_object(self, key: str, data: bytes) -> None:
            raise audit_backup.BackupError("connection reset by peer")

    def test_upload_failure_is_recorded_and_not_raised(self) -> None:
        self.append(self.line(0))
        result = audit_backup.run_backup(destination=self._ExplodingDestination())
        self.assertFalse(result.ok)
        self.assertIn("connection reset by peer", result.error)
        state = audit_backup.load_state()
        self.assertIn("connection reset by peer", state.last_error)
        # The watermark must not advance on a failed upload, or the bytes are lost.
        self.assertEqual(state.live_offset, 0)

    def test_a_later_success_clears_the_error(self) -> None:
        self.append(self.line(0))
        audit_backup.run_backup(destination=self._ExplodingDestination())
        audit_backup.run_backup(destination=self.dest())
        state = audit_backup.load_state()
        self.assertEqual(state.last_error, "")
        self.assertTrue(state.last_success_utc)

    def test_state_file_is_not_mistaken_for_an_audit_log(self) -> None:
        self.append(self.line(0))
        audit_backup.run_backup(destination=self.dest())
        names = [p.name for p in audit_backup.rotated_audit_files()]
        self.assertEqual(names, [], "state/lock files must not look like rotated audit logs")

    def test_unwritable_state_file_is_not_fatal(self) -> None:
        state = audit_backup.BackupState()
        self.assertFalse(audit_backup.save_state(state, Path("/proc/definitely/not/writable/x.json")))


class TestHealth(_BackupCase):
    def test_reports_disabled_when_unconfigured(self) -> None:
        config.AUDIT_BACKUP_DESTINATION = ""
        payload = audit_backup.audit_backup_health()
        self.assertFalse(payload["configured"])
        self.assertEqual(payload["status"], "disabled")

    def test_reports_never_ran_before_the_first_pass(self) -> None:
        payload = audit_backup.audit_backup_health()
        self.assertTrue(payload["configured"])
        self.assertEqual(payload["status"], "never_ran")
        self.assertTrue(payload["stale"])

    def test_reports_ok_after_a_pass(self) -> None:
        self.append(self.line(0))
        audit_backup.run_backup(destination=self.dest())
        payload = audit_backup.audit_backup_health()
        self.assertEqual(payload["status"], "ok")
        self.assertFalse(payload["stale"])
        self.assertEqual(payload["uploaded_objects"], 1)
        self.assertGreater(payload["last_success_utc"], "")

    def test_reports_error_after_a_failed_pass(self) -> None:
        self.append(self.line(0))
        audit_backup.run_backup(destination=TestFailureHandling._ExplodingDestination())
        payload = audit_backup.audit_backup_health()
        self.assertEqual(payload["status"], "error")
        self.assertIn("connection reset", payload["reason"])

    def test_reports_stale_when_the_last_success_is_too_old(self) -> None:
        self.append(self.line(0))
        audit_backup.run_backup(destination=self.dest())
        state = audit_backup.load_state()
        state.last_success_utc = "2020-01-01T00:00:00+00:00"
        audit_backup.save_state(state)
        payload = audit_backup.audit_backup_health()
        self.assertEqual(payload["status"], "stale")
        self.assertTrue(payload["stale"])

    def test_pending_bytes_counts_what_a_pod_death_would_lose(self) -> None:
        self.append(self.line(0))
        self.assertEqual(audit_backup.pending_bytes(), len(self.line(0).encode()))
        audit_backup.run_backup(destination=self.dest())
        self.assertEqual(audit_backup.pending_bytes(), 0)

    def test_pending_bytes_does_not_hash_rotated_files(self) -> None:
        """The /health path must stay cheap; hashing is asserted against here."""
        calls = {"n": 0}
        original = audit_backup._file_sha12

        def _counting(path):  # noqa: ANN001
            calls["n"] += 1
            return original(path)

        self.append(self.line(0))
        self.rotate()
        audit_backup._file_sha12 = _counting  # type: ignore[assignment]
        try:
            audit_backup.pending_bytes()
        finally:
            audit_backup._file_sha12 = original  # type: ignore[assignment]
        self.assertEqual(calls["n"], 0, "pending_bytes must compare sizes, not hashes")

    def test_disk_status_reports_space_and_floor(self) -> None:
        status = audit_backup.disk_status()
        self.assertTrue(status["checked"])
        self.assertGreater(status["total_bytes"], 0)
        self.assertIn("below_floor", status)

    def test_disk_floor_flags_a_degraded_volume(self) -> None:
        config.DISK_MIN_FREE_BYTES = 10**18  # more free space than exists
        status = audit_backup.disk_status()
        self.assertTrue(status["below_floor"])


class TestCloudBackendConfiguration(_BackupCase):
    """GCS/Azure are optional SDKs: misconfiguration and absence must read clearly.

    These backends are not exercised against a live account (no credentials in this
    environment), so what is tested here is the part a first deployment actually
    hits: a missing bucket, a missing SDK, and a missing credential source. Each
    must produce a one-line diagnosis naming the fix — never an ImportError or a
    traceback from a library the operator did not install.
    """

    def test_gcs_without_a_bucket_names_the_variable(self) -> None:
        config.AUDIT_BACKUP_DESTINATION = "gcs"
        config.AUDIT_BACKUP_GCS_BUCKET = ""
        with self.assertRaises(audit_backup.BackupError) as ctx:
            audit_backup.build_destination()
        self.assertIn("VA_LSE_AUDIT_BACKUP_GCS_BUCKET", str(ctx.exception))

    def test_azure_without_a_container_names_the_variable(self) -> None:
        config.AUDIT_BACKUP_DESTINATION = "azure"
        config.AUDIT_BACKUP_AZURE_CONTAINER = ""
        with self.assertRaises(audit_backup.BackupError) as ctx:
            audit_backup.build_destination()
        self.assertIn("VA_LSE_AUDIT_BACKUP_AZURE_CONTAINER", str(ctx.exception))

    def test_azure_without_credentials_explains_both_options(self) -> None:
        """Either diagnosis is acceptable depending on whether the SDK is installed."""
        config.AUDIT_BACKUP_DESTINATION = "azure"
        config.AUDIT_BACKUP_AZURE_CONTAINER = "audit"
        config.AUDIT_BACKUP_AZURE_ACCOUNT_URL = ""
        previous = os.environ.pop("AZURE_STORAGE_CONNECTION_STRING", None)
        try:
            with self.assertRaises(audit_backup.BackupError) as ctx:
                audit_backup.build_destination()
        finally:
            if previous is not None:
                os.environ["AZURE_STORAGE_CONNECTION_STRING"] = previous
        message = str(ctx.exception)
        self.assertTrue(
            "AZURE_STORAGE_CONNECTION_STRING" in message or "requirements-backup.txt" in message,
            message,
        )

    def test_a_missing_sdk_is_reported_as_a_dependency_not_a_crash(self) -> None:
        """Simulate an image built without requirements-backup.txt."""
        import sys

        class _Blocked:
            """A meta-path finder that makes the cloud SDKs unimportable."""

            blocked = ("google.cloud.storage", "azure.storage.blob", "boto3")

            def find_module(self, fullname: str, _path: object = None) -> object:
                return self if fullname in self.blocked else None

            def find_spec(self, fullname: str, _path: object = None, _target: object = None) -> object:
                if fullname in self.blocked:
                    raise ImportError(f"blocked for test: {fullname}")
                return None

        finder = _Blocked()
        saved = {name: sys.modules.pop(name, None) for name in finder.blocked}
        sys.meta_path.insert(0, finder)
        try:
            for choice, attr, value in (
                ("s3", "AUDIT_BACKUP_S3_BUCKET", "bucket"),
                ("gcs", "AUDIT_BACKUP_GCS_BUCKET", "bucket"),
            ):
                config.AUDIT_BACKUP_DESTINATION = choice
                setattr(config, attr, value)
                with self.assertRaises(audit_backup.BackupError) as ctx:
                    audit_backup.build_destination()
                self.assertIn("requirements-backup.txt", str(ctx.exception))
                self.assertNotIsInstance(ctx.exception, ImportError)
        finally:
            sys.meta_path.remove(finder)
            for name, module in saved.items():
                if module is not None:
                    sys.modules[name] = module

    def test_an_unknown_backend_lists_the_valid_ones(self) -> None:
        config.AUDIT_BACKUP_DESTINATION = "dropbox"
        with self.assertRaises(audit_backup.BackupError) as ctx:
            audit_backup.build_destination()
        message = str(ctx.exception)
        for choice in ("filesystem", "s3", "gcs", "azure"):
            self.assertIn(choice, message)


class TestLock(_BackupCase):
    def test_second_holder_is_refused(self) -> None:
        first = audit_backup.BackupLock()
        self.assertTrue(first.acquire())
        try:
            with self.assertRaises(audit_backup.BackupError):
                with audit_backup.BackupLock():
                    pass
        finally:
            first.release()

    def test_lock_is_reusable_after_release(self) -> None:
        with audit_backup.BackupLock():
            pass
        with audit_backup.BackupLock() as lock:
            self.assertTrue(lock._held)

    def test_a_fresh_lock_is_respected(self) -> None:
        held = audit_backup.BackupLock()
        self.assertTrue(held.acquire())
        try:
            fresh = audit_backup.BackupLock()
            self.assertFalse(fresh.acquire())
        finally:
            held.release()

    def test_a_stale_lock_is_taken_over(self) -> None:
        """SIGKILL (the CronJob's activeDeadlineSeconds) leaves the lock behind.

        Without takeover, one hung upload would disable every later backup — a
        silently dead compliance job that still looked scheduled.
        """
        abandoned = audit_backup.BackupLock()
        self.assertTrue(abandoned.acquire())
        # Backdate the lock as if the owning process had been killed an hour ago.
        old = time.time() - 7200
        os.utime(abandoned._path, (old, old))
        takeover = audit_backup.BackupLock(stale_after_seconds=3600)
        try:
            self.assertTrue(takeover.acquire())
            self.assertTrue(takeover._held)
        finally:
            takeover.release()

    def test_a_recent_lock_is_not_taken_over(self) -> None:
        held = audit_backup.BackupLock()
        self.assertTrue(held.acquire())
        try:
            # One second old: another process may simply be mid-upload.
            os.utime(held._path, None)
            self.assertFalse(audit_backup.BackupLock(stale_after_seconds=3600).acquire())
        finally:
            held.release()


class TestAuditErrorScrubbing(unittest.TestCase):
    """``error_message`` is the one audit field outside the no-PII contract."""

    def test_pii_shaped_tokens_are_redacted(self) -> None:
        from app.audit import _scrub_error_message

        scrubbed = _scrub_error_message(
            "failed for ssn 123-45-6789, claim 98765432, user a.b@example.com"
        )
        self.assertNotIn("123-45-6789", scrubbed)
        self.assertNotIn("98765432", scrubbed)
        self.assertNotIn("a.b@example.com", scrubbed)
        self.assertIn("[redacted]", scrubbed)

    def test_newlines_are_collapsed_so_one_entry_stays_one_line(self) -> None:
        from app.audit import _scrub_error_message

        self.assertNotIn("\n", _scrub_error_message("line one\nline two\ttabbed"))

    def test_length_is_bounded(self) -> None:
        from app.audit import _scrub_error_message

        self.assertLessEqual(len(_scrub_error_message("x" * 5000)), 301)


if __name__ == "__main__":
    unittest.main()
