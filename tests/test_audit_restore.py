"""Tests for audit restore + integrity verification (app/audit_restore.py).

The load-bearing behaviours here are the ones that decide whether an operator can
trust what the backup says:

* every object is checked against the hash embedded in *its own key*, so a
  corrupted object is caught without any side manifest;
* windows that do not tile the stream are reported as gaps with exact offsets —
  a gap is the audit records that no longer exist anywhere;
* windows from different file generations are never merged, which is the failure
  a restore would otherwise not notice (every generation starts at offset 0);
* restoring writes a readable stream plus a manifest, and never modifies the
  destination.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)
from app import audit_backup, audit_restore, config


def _live_key(filename: str, start: int, data: bytes, generation: str) -> str:
    """Build a live-window key through the producer, so tests cannot drift from it."""
    return audit_backup.object_key(
        True,
        filename,
        start=start,
        end=start + len(data),
        sha12=audit_backup._sha12(data),
        generation=generation,
    )


def _rotated_key(filename: str, data: bytes) -> str:
    return audit_backup.object_key(
        False, filename, end=len(data), sha12=audit_backup._sha12(data)
    )


class _RestoreCase(unittest.TestCase):
    """A filesystem destination holding hand-built backup objects."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.dest_root = self.root / "offsite"
        self.dest_root.mkdir()
        self.logs = self.root / "logs"
        self.logs.mkdir()
        self._saved = {
            name: getattr(config, name)
            for name in (
                "AUDIT_LOG_DIR",
                "AUDIT_LOG_FILE",
                "AUDIT_BACKUP_DESTINATION",
                "AUDIT_BACKUP_DIR",
                "AUDIT_RETENTION_DAYS",
            )
        }
        config.AUDIT_LOG_DIR = str(self.logs)
        config.AUDIT_LOG_FILE = "audit.log"
        config.AUDIT_BACKUP_DESTINATION = "filesystem"
        config.AUDIT_BACKUP_DIR = str(self.dest_root)
        self._clock = 1_700_000_000.0

    def tearDown(self) -> None:
        for name, value in self._saved.items():
            setattr(config, name, value)
        self._tmp.cleanup()

    # -- builders ----------------------------------------------------------
    def dest(self) -> audit_backup.FilesystemDestination:
        return audit_backup.FilesystemDestination(self.dest_root)

    def put(self, key: str, data: bytes, *, mtime: float | None = None) -> str:
        """Write an object at ``key``, with an explicit mtime so upload order is real."""
        path = self.dest_root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self._clock += 10.0
        stamp = self._clock if mtime is None else mtime
        os.utime(path, (stamp, stamp))
        return key

    def put_window(
        self,
        start: int,
        data: bytes,
        *,
        generation: str = "aaaabbbbcccc",
        filename: str = "audit.log",
        mtime: float | None = None,
    ) -> str:
        return self.put(
            _live_key(filename, start, data, generation), data, mtime=mtime
        )

    def put_rotated(self, data: bytes, *, filename: str = "audit.log.1") -> str:
        return self.put(_rotated_key(filename, data), data)

    def line(self, index: int) -> bytes:
        return (json.dumps({"request_id": f"req_{index}", "status": "ok"}) + "\n").encode()


class TestParseObjectKey(_RestoreCase):
    def test_parses_a_live_window(self) -> None:
        key = self.put_window(0, self.line(0))
        ref = audit_restore.parse_object_key(key)
        assert ref is not None
        self.assertEqual(ref.kind, "live")
        self.assertEqual(ref.start, 0)
        self.assertEqual(ref.end, len(self.line(0)))
        self.assertEqual(ref.generation, "aaaabbbbcccc")

    def test_parses_a_rotated_file(self) -> None:
        key = self.put_rotated(self.line(0))
        ref = audit_restore.parse_object_key(key)
        assert ref is not None
        self.assertEqual(ref.kind, "rotated")
        self.assertEqual(ref.size, len(self.line(0)))
        self.assertEqual(ref.generation, audit_restore.UNKNOWN_GENERATION)

    def test_parses_a_legacy_key_without_a_generation(self) -> None:
        """Keys written before the generation tag existed must still be readable."""
        data = self.line(0)
        key = f"audit/2026/09/16/live-audit.log-0-{len(data)}-{audit_backup._sha12(data)}.jsonl"
        ref = audit_restore.parse_object_key(key)
        assert ref is not None
        self.assertEqual(ref.kind, "live")
        self.assertEqual(ref.generation, audit_restore.UNKNOWN_GENERATION)

    def test_rejects_an_unrelated_key(self) -> None:
        self.assertIsNone(audit_restore.parse_object_key("audit/notes.txt"))


class TestIntegrity(_RestoreCase):
    def test_intact_backup_verifies(self) -> None:
        self.put_window(0, self.line(0) + self.line(1))
        report = audit_restore.verify_backup(self.dest())
        self.assertTrue(report.ok, report.to_json())
        self.assertEqual(report.verified_objects, 1)
        self.assertEqual(report.gaps, [])

    def test_tampered_content_is_detected_by_the_key_embedded_hash(self) -> None:
        """No side manifest: the expected hash travels with the object."""
        key = self.put_window(0, self.line(0) + self.line(1))
        (self.dest_root / key).write_bytes(self.line(9))  # same store, different bytes
        report = audit_restore.verify_backup(self.dest())
        self.assertFalse(report.ok)
        self.assertEqual(len(report.corrupt), 1)
        self.assertIn("does not match", report.corrupt[0].error)
        self.assertIn(key, report.corrupt[0].key)

    def test_corrupt_objects_do_not_abort_verification(self) -> None:
        good = self.put_window(0, self.line(0))
        bad = self.put_window(len(self.line(0)), self.line(1))
        (self.dest_root / bad).write_bytes(b"{}")
        report = audit_restore.verify_backup(self.dest())
        self.assertFalse(report.ok)
        self.assertIn(good, report.checks)
        self.assertIn(bad, report.checks)

    def test_unrecognized_objects_are_reported_not_ignored(self) -> None:
        # `.jsonl` on purpose: the filesystem backend lists only that extension
        # (see FilesystemDestination.list_objects), so a non-jsonl object would not
        # reach this code path at all. On S3/GCS/Azure every object is listed.
        odd = self.put("audit/2026/09/16/backup-manifest.jsonl", b"{}")
        report = audit_restore.verify_backup(self.dest())
        self.assertEqual(report.unrecognized, [odd])
        self.assertIn("unrecognized objects", audit_restore.render_verify_text(report))

    def test_hash_check_can_be_skipped(self) -> None:
        key = self.put_window(0, self.line(0))
        (self.dest_root / key).write_bytes(self.line(9))
        report = audit_restore.verify_backup(self.dest(), hash_objects=False)
        self.assertFalse(report.hashed)
        self.assertEqual(report.corrupt, [])
        self.assertEqual(report.verified_objects, 1)


class TestCoverage(_RestoreCase):
    def test_tiled_windows_have_no_gaps(self) -> None:
        first = self.line(0)
        second = self.line(1)
        self.put_window(0, first)
        self.put_window(len(first), second)
        report = audit_restore.verify_backup(self.dest())
        self.assertEqual(report.gaps, [])
        self.assertTrue(report.ok)
        # One generation, both windows, in stream order.
        self.assertEqual(len(report.plan.generations), 1)
        self.assertEqual([w.start for w in report.plan.windows], [0, len(first)])
        generation = report.plan.generations[0]
        self.assertEqual((generation.start, generation.end), (0, len(first) + len(second)))

    def test_a_hole_between_windows_is_reported_with_exact_offsets(self) -> None:
        """The failure a naive verify misses: the pass that would have covered it failed."""
        first = self.line(0)
        self.put_window(0, first)
        # 500 bytes were written while the backup job was down.
        self.put_window(len(first) + 500, self.line(1))
        report = audit_restore.verify_backup(self.dest())
        self.assertFalse(report.ok)
        self.assertEqual(len(report.gaps), 1)
        gap = report.gaps[0]
        self.assertEqual(gap.start, len(first))
        self.assertEqual(gap.end, len(first) + 500)
        self.assertEqual(gap.missing_bytes, 500)
        self.assertEqual(report.missing_bytes, 500)

    def test_a_longer_reupload_of_the_same_range_is_not_a_gap(self) -> None:
        """The crash-retry case: same start, more lines arrived before the retry."""
        short = self.line(0)
        longer = self.line(0) + self.line(1)
        self.put_window(0, short)
        self.put_window(0, longer)
        report = audit_restore.verify_backup(self.dest())
        self.assertEqual(report.gaps, [])
        self.assertEqual(report.plan.superseded, 1)
        # The longer window is authoritative and is the one a restore writes.
        self.assertEqual(len(report.plan.windows), 1)
        self.assertEqual(report.plan.generations[0].end, len(longer))

    def test_generation_starting_above_zero_is_flagged_incomplete(self) -> None:
        self.put_window(900, self.line(0))
        report = audit_restore.verify_backup(self.dest())
        generation = report.plan.generations[0]
        self.assertEqual(generation.start, 900)
        self.assertFalse(generation.complete)
        self.assertIn("head is missing", audit_restore.render_verify_text(report))

    def test_two_generations_are_never_merged(self) -> None:
        """Both files start at offset 0 — the tag is what keeps them apart."""
        first = self.line(0)
        second = self.line(1)
        self.put_window(0, first, generation="111111111111")
        self.put_window(0, second, generation="222222222222")
        report = audit_restore.verify_backup(self.dest())
        self.assertEqual(len(report.plan.generations), 2)
        self.assertEqual([g.generation for g in report.plan.generations],
                         ["111111111111", "222222222222"])
        # Two files that each start at 0 must not be read as a 2x-overlapping gap.
        self.assertEqual(report.gaps, [])

    def test_generations_with_equal_upload_times_are_flagged_unordered(self) -> None:
        """The order between files has no anchor but upload time — say so, don't guess.

        Offsets cannot order generations (each starts at 0), so a destination that
        reports no modification times, or two that collide, leaves the chronological
        order unknowable. Silently emitting a mis-ordered forensic record is the
        failure this flag exists to prevent.
        """
        first = self.line(0)
        second = self.line(1)
        same = 1_700_000_000.0
        self.put_window(0, first, generation="111111111111", mtime=same)
        self.put_window(0, second, generation="222222222222", mtime=same)
        report = audit_restore.verify_backup(self.dest())
        self.assertTrue(report.plan.order_uncertain)
        # Intact and complete, but not trustworthy in order: both facts are reported.
        self.assertTrue(report.ok)
        self.assertTrue(report.needs_attention)
        self.assertIn("could not be put in chronological order",
                      audit_restore.render_verify_text(report))

    def test_generations_with_distinct_times_are_ordered(self) -> None:
        self.put_window(0, self.line(0), generation="111111111111")
        self.put_window(0, self.line(1), generation="222222222222")
        report = audit_restore.verify_backup(self.dest())
        self.assertFalse(report.plan.order_uncertain)
        self.assertFalse(report.needs_attention)

    def test_a_single_generation_is_never_flagged_unordered(self) -> None:
        self.put_window(0, self.line(0))
        report = audit_restore.verify_backup(self.dest())
        self.assertFalse(report.plan.order_uncertain)

    def test_untagged_windows_are_counted(self) -> None:
        data = self.line(0)
        key = f"audit/2026/09/16/live-audit.log-0-{len(data)}-{audit_backup._sha12(data)}.jsonl"
        self.put(key, data)
        report = audit_restore.verify_backup(self.dest())
        self.assertEqual(report.plan.unknown_generation_windows, 1)
        self.assertIn("no generation tag", audit_restore.render_verify_text(report))


class TestRetryDedupe(_RestoreCase):
    def test_a_retried_range_is_written_once_not_twice(self) -> None:
        """Regression: [0,40), [40,80), then a retry of 40 with a longer end.

        The middle window is wholly inside the retry, so a naive "keep every window
        whose end extends coverage" walk writes those 40 bytes twice — a duplicated
        audit record in the restored stream, which is exactly the kind of defect a
        reviewer would not notice.
        """
        a = self.line(0)
        b = self.line(1)
        c = self.line(2)
        self.put_window(0, a)
        self.put_window(len(a), b)
        self.put_window(len(a), b + c)  # crashed before checkpointing, then grew
        report = audit_restore.verify_backup(self.dest())
        self.assertEqual(report.gaps, [])
        self.assertEqual(report.plan.superseded, 1)
        self.assertEqual([w.start for w in report.plan.windows], [0, len(a)])
        self.assertEqual(
            [w.end for w in report.plan.windows], [len(a), len(a) + len(b) + len(c)]
        )

    def test_the_supersede_count_matches_what_is_actually_written(self) -> None:
        a = self.line(0)
        b = self.line(1)
        self.put_window(0, a)
        self.put_window(0, a + b)
        report = audit_restore.verify_backup(self.dest())
        windows_in_generations = sum(len(g.refs) for g in report.plan.generations)
        self.assertEqual(windows_in_generations - report.plan.superseded,
                         len(report.plan.windows))


class TestRoundTripWithTheBackupJob(_RestoreCase):
    """The producer/consumer contract: what run_backup writes, restore reads back.

    Building objects by hand can only test the reader against my understanding of
    the key format. This runs the real backup pass and reads the result back, which
    is what actually breaks when the two modules drift apart.
    """

    def live(self) -> Path:
        return self.logs / "audit.log"

    def append(self, index: int) -> bytes:
        data = self.line(index)
        with self.live().open("ab") as handle:
            handle.write(data)
        return data

    def rotate(self) -> None:
        """Mimic RotatingFileHandler: a rename, so the next audit.log is a new file."""
        self.live().replace(self.logs / "audit.log.1")

    def test_a_real_backup_verifies_and_restores_losslessly(self) -> None:
        first = self.append(0) + self.append(1)
        audit_backup.run_backup(destination=self.dest())
        self.rotate()
        second = self.append(2) + self.append(3)
        result = audit_backup.run_backup(destination=self.dest())
        self.assertTrue(result.ok, result.error)

        report = audit_restore.verify_backup(self.dest())
        self.assertTrue(report.ok, report.to_json())
        self.assertEqual(report.gaps, [])
        self.assertEqual(report.corrupt, [])
        self.assertEqual(report.malformed_lines, 0)

        restored = self.root / "restored"
        out = audit_restore.restore_backup(self.dest(), target=restored)
        self.assertTrue(out.ok, out.error)
        # The two files' windows, in upload order: this is the record a forensic
        # reader wants, and it is only correct because the key carries the
        # generation (both files start at offset 0).
        self.assertEqual((restored / "restored.jsonl").read_bytes(), first + second)
        self.assertEqual(len(report.plan.generations), 2)

    def test_a_reread_from_offset_zero_is_not_reported_as_a_false_gap(self) -> None:
        """A lost state file re-ships from 0. Two overlapping views, not a hole.

        This is the false positive that would make the whole tool untrustworthy: a
        verify that cries "missing bytes" whenever a watermark was lost would be
        ignored precisely when it was right.
        """
        self.append(0)
        audit_backup.run_backup(destination=self.dest())
        # The job is down for a while; the writer keeps going.
        self.append(1)
        self.append(2)
        self.append(3)
        # A later pass resumes from a *stale* watermark, as after a state-file loss.
        audit_backup.run_backup(destination=self.dest(), state=audit_backup.BackupState())
        report = audit_restore.verify_backup(self.dest())
        self.assertEqual(report.gaps, [])
        self.assertGreaterEqual(report.plan.superseded, 1)
        self.assertTrue(report.ok, report.to_json())

    def test_legacy_keys_from_a_previous_version_still_verify(self) -> None:
        """A destination written before the generation tag existed must stay readable."""
        data = self.line(0)
        key = f"audit/2026/09/16/live-audit.log-0-{len(data)}-{audit_backup._sha12(data)}.jsonl"
        self.put(key, data)
        report = audit_restore.verify_backup(self.dest())
        self.assertEqual(report.corrupt, [])
        self.assertEqual(report.plan.unknown_generation_windows, 1)
        self.assertTrue(report.ok, report.to_json())


class TestFailureModes(_RestoreCase):
    def test_unconfigured_destination_reports_clearly(self) -> None:
        config.AUDIT_BACKUP_DESTINATION = ""
        report = audit_restore.verify_backup()
        self.assertFalse(report.listed)
        self.assertIn("no backup destination is configured", report.list_error)
        self.assertFalse(report.ok)

    def test_unreachable_destination_does_not_raise(self) -> None:
        class Broken(audit_backup.BackupDestination):
            name = "broken"

            def list_objects(self, prefix: str = "") -> list[audit_backup.RemoteObject]:
                raise audit_backup.BackupError("connection refused")

            def describe(self) -> dict[str, object]:
                return {"backend": "broken"}

        report = audit_restore.verify_backup(Broken())
        self.assertFalse(report.listed)
        self.assertIn("connection refused", report.list_error)
        self.assertFalse(report.ok)

    def test_an_unreadable_object_is_distinguished_from_a_corrupt_one(self) -> None:
        key = self.put_window(0, self.line(0))

        class HalfBroken(audit_backup.BackupDestination):
            name = "half"

            def list_objects(self, prefix: str = "") -> list[audit_backup.RemoteObject]:
                return [audit_backup.RemoteObject(key=key, size=10, last_modified=None)]

            def get_object(self, key: str) -> bytes:
                raise audit_backup.BackupError("403 Forbidden")

            def describe(self) -> dict[str, object]:
                return {"backend": "half"}

        report = audit_restore.verify_backup(HalfBroken())
        self.assertEqual(len(report.unreadable), 1)
        self.assertEqual(report.corrupt, [])
        self.assertFalse(report.ok)


class TestRestore(_RestoreCase):
    def test_rebuilds_the_stream_in_upload_order(self) -> None:
        first = self.line(0) + self.line(1)
        second = self.line(2) + self.line(3)
        self.put_window(0, first, generation="111111111111")
        self.put_window(0, second, generation="222222222222")
        out_dir = self.root / "restored"
        report = audit_restore.restore_backup(self.dest(), target=out_dir)
        self.assertTrue(report.ok, report.to_json())
        stream = (out_dir / "restored.jsonl").read_bytes()
        self.assertEqual(stream, first + second)
        self.assertEqual(report.wrote_lines, 4)

    def test_writes_a_manifest_describing_the_result(self) -> None:
        self.put_window(0, self.line(0))
        out_dir = self.root / "restored"
        audit_restore.restore_backup(self.dest(), target=out_dir)
        manifest = json.loads((out_dir / "manifest.json").read_text())
        self.assertTrue(manifest["verify"]["ok"])
        self.assertEqual(manifest["objects_included"], 1)
        self.assertEqual(manifest["verify"]["missing_bytes"], 0)

    def test_manifest_records_gaps_so_a_holey_restore_cannot_pass_unnoticed(self) -> None:
        first = self.line(0)
        self.put_window(0, first)
        self.put_window(len(first) + 100, self.line(1))
        out_dir = self.root / "restored"
        report = audit_restore.restore_backup(self.dest(), target=out_dir)
        manifest = json.loads((out_dir / "manifest.json").read_text())
        self.assertEqual(manifest["verify"]["missing_bytes"], 100)
        self.assertFalse(manifest["verify"]["ok"])
        self.assertIn("gap", audit_restore.render_restore_text(report))

    def test_rotated_snapshots_are_written_under_content_addressed_names(self) -> None:
        snapshot = self.line(0)
        self.put_rotated(snapshot)
        self.put_rotated(snapshot)  # same content again -> same name, no collision
        out_dir = self.root / "restored"
        report = audit_restore.restore_backup(self.dest(), target=out_dir)
        self.assertEqual(len(report.rotated_paths), 1)
        self.assertEqual(Path(report.rotated_paths[0]).read_bytes(), snapshot)

    def test_two_snapshots_of_one_filename_do_not_overwrite_each_other(self) -> None:
        """audit.log.1 is reused by the rotation scheme, so the name alone is not unique."""
        first = self.line(0)
        second = self.line(1) + self.line(2)
        self.put_rotated(first)
        self.put_rotated(second)
        out_dir = self.root / "restored"
        report = audit_restore.restore_backup(self.dest(), target=out_dir)
        self.assertEqual(len(report.rotated_paths), 2)
        contents = {Path(p).read_bytes() for p in report.rotated_paths}
        self.assertEqual(contents, {first, second})

    def test_a_corrupt_object_is_restored_and_flagged_not_silently_dropped(self) -> None:
        key = self.put_window(0, self.line(0))
        (self.dest_root / key).write_bytes(b"{}")
        out_dir = self.root / "restored"
        report = audit_restore.restore_backup(self.dest(), target=out_dir)
        self.assertEqual(len(report.verify.corrupt), 1)  # type: ignore[union-attr]
        self.assertFalse(report.verify.ok)  # type: ignore[union-attr]
        # Written anyway: for forensics a flagged copy beats no copy.
        self.assertEqual((out_dir / "restored.jsonl").read_bytes(), b"{}")

    def test_restore_never_writes_to_the_destination(self) -> None:
        self.put_window(0, self.line(0))
        before = sorted(p.relative_to(self.dest_root).as_posix()
                        for p in self.dest_root.rglob("*"))
        audit_restore.restore_backup(self.dest(), target=self.root / "restored")
        after = sorted(p.relative_to(self.dest_root).as_posix()
                       for p in self.dest_root.rglob("*"))
        self.assertEqual(before, after)

    def test_a_second_restore_refuses_to_clobber_the_first(self) -> None:
        """A rerun must not quietly replace a record someone is about to hand over."""
        self.put_window(0, self.line(0))
        out_dir = self.root / "restored"
        self.assertTrue(audit_restore.restore_backup(self.dest(), target=out_dir).ok)
        first = (out_dir / "restored.jsonl").read_bytes()
        again = audit_restore.restore_backup(self.dest(), target=out_dir)
        self.assertFalse(again.ok)
        self.assertTrue(again.refused)
        self.assertIn("already exists", again.error)
        self.assertEqual((out_dir / "restored.jsonl").read_bytes(), first)

    def test_force_allows_a_deliberate_re_restore(self) -> None:
        """Re-restoring must pick up objects that arrived since the first attempt."""
        first = self.line(0)
        second = self.line(1)
        self.put_window(0, first)
        out_dir = self.root / "restored"
        audit_restore.restore_backup(self.dest(), target=out_dir)
        self.assertEqual((out_dir / "restored.jsonl").read_bytes(), first)
        # The file rotated and the new one was shipped after the first restore.
        self.put_window(0, second, generation="bbbbccccdddd")
        forced = audit_restore.restore_backup(self.dest(), target=out_dir, force=True)
        self.assertTrue(forced.ok, forced.error)
        self.assertEqual((out_dir / "restored.jsonl").read_bytes(), first + second)

    def test_no_hash_restore_still_writes_the_rotated_snapshots(self) -> None:
        """Regression: skipping the downloads dropped the snapshots from the restore.

        Those snapshots are the only copy of any range the live watermark never
        reached, so omitting them from a restore is the worst kind of quiet loss.
        """
        self.put_window(0, self.line(0))
        snapshot = self.line(0) + self.line(1)
        self.put_rotated(snapshot)
        out_dir = self.root / "restored"
        report = audit_restore.restore_backup(
            self.dest(), target=out_dir, hash_objects=False
        )
        self.assertTrue(report.ok, report.error)
        self.assertIsNotNone(report.verify)
        self.assertEqual(len(report.verify.rotated_refs), 1)  # type: ignore[union-attr]
        self.assertEqual(len(report.rotated_paths), 1)
        self.assertEqual(Path(report.rotated_paths[0]).read_bytes(), snapshot)

    def test_restore_without_verification_still_produces_a_stream(self) -> None:
        data = self.line(0)
        self.put_window(0, data)
        out_dir = self.root / "restored"
        report = audit_restore.restore_backup(self.dest(), target=out_dir, verify=False)
        self.assertTrue(report.ok)
        self.assertIsNone(report.verify)
        self.assertEqual((out_dir / "restored.jsonl").read_bytes(), data)

    def test_unconfigured_restore_reports_an_error(self) -> None:
        config.AUDIT_BACKUP_DESTINATION = ""
        report = audit_restore.restore_backup(target=self.root / "restored")
        self.assertFalse(report.ok)
        self.assertIn("no backup destination", report.error)


class TestDestinationOverrides(_RestoreCase):
    def test_overrides_let_recovery_read_from_another_store(self) -> None:
        overrides = audit_restore.config_overrides_from_args(
            destination="filesystem", path=str(self.dest_root)
        )
        destination = audit_backup.build_destination(overrides=overrides)
        self.assertIsInstance(destination, audit_backup.FilesystemDestination)
        self.assertEqual(destination.root, self.dest_root)

    def test_overrides_never_mutate_process_config(self) -> None:
        before = config.AUDIT_BACKUP_DIR
        audit_backup.build_destination(
            overrides={"AUDIT_BACKUP_DESTINATION": "filesystem", "AUDIT_BACKUP_DIR": "/tmp/elsewhere"}
        )
        self.assertEqual(config.AUDIT_BACKUP_DIR, before)

    def test_unknown_destination_via_override_is_rejected(self) -> None:
        with self.assertRaises(audit_backup.BackupError):
            audit_backup.build_destination(overrides={"AUDIT_BACKUP_DESTINATION": "dropbox"})


class TestIntegrityHealth(_RestoreCase):
    def test_reports_a_usable_configuration(self) -> None:
        payload = audit_restore.audit_integrity_health()
        self.assertTrue(payload["restore_available"])
        self.assertEqual(payload["destination"], "filesystem")

    def test_reports_unconfigured_without_touching_the_network(self) -> None:
        config.AUDIT_BACKUP_DESTINATION = ""
        payload = audit_restore.audit_integrity_health()
        self.assertFalse(payload["restore_available"])
        self.assertEqual(payload["destination"], "none")


class TestRestoreCli(_RestoreCase):
    """Exit codes, because this is run from a terminal during an incident.

    * 0 — verified or restored
    * 1 — nothing configured / bad usage (a setup problem)
    * 2 — the backup is reachable but incomplete or corrupt (a data problem)
    """

    def setUp(self) -> None:
        super().setUp()
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "restore_audit_logs", Path(__file__).resolve().parent.parent
            / "scripts" / "restore_audit_logs.py"
        )
        assert spec and spec.loader
        self.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.cli)

    def _run(self, argv: list[str]) -> tuple[int, str]:
        """Run the CLI with both streams captured, so test logs stay readable.

        Diagnostics go to stderr (the operator-facing channel) while results go to
        stdout, and both are asserted on somewhere below.
        """
        buffer = io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(buffer):
            code = self.cli.main(argv)
        return code, buffer.getvalue()

    def test_verify_exits_zero_on_an_intact_backup(self) -> None:
        self.put_window(0, self.line(0))
        code, out = self._run(["--verify"])
        self.assertEqual(code, 0, out)
        self.assertIn("VERDICT: ok", out)

    def test_verify_exits_two_on_a_corrupt_object(self) -> None:
        key = self.put_window(0, self.line(0))
        (self.dest_root / key).write_bytes(b"{}")
        code, out = self._run(["--verify"])
        self.assertEqual(code, 2)
        self.assertIn("CORRUPT", out)

    def test_verify_exits_two_on_a_gap(self) -> None:
        first = self.line(0)
        self.put_window(0, first)
        self.put_window(len(first) + 64, self.line(1))
        code, out = self._run(["--verify"])
        self.assertEqual(code, 2)
        self.assertIn("GAPS (1)", out)

    def test_verify_exits_two_when_generation_order_is_unknowable(self) -> None:
        same = 1_700_000_000.0
        self.put_window(0, self.line(0), generation="111111111111", mtime=same)
        self.put_window(0, self.line(1), generation="222222222222", mtime=same)
        code, out = self._run(["--verify"])
        self.assertEqual(code, 2)
        self.assertIn("chronological order", out)

    def test_verify_exits_one_when_nothing_is_configured(self) -> None:
        config.AUDIT_BACKUP_DESTINATION = ""
        code, _ = self._run(["--verify"])
        self.assertEqual(code, 1)

    def test_restore_exits_zero_and_writes_the_stream(self) -> None:
        data = self.line(0)
        self.put_window(0, data)
        target = self.root / "restored"
        code, out = self._run(["--restore", str(target)])
        self.assertEqual(code, 0, out)
        self.assertEqual((target / "restored.jsonl").read_bytes(), data)
        self.assertTrue((target / "manifest.json").exists())

    def test_restore_exits_two_when_the_stream_it_wrote_has_gaps(self) -> None:
        """A holey restore must not look like a clean success."""
        first = self.line(0)
        self.put_window(0, first)
        self.put_window(len(first) + 64, self.line(1))
        target = self.root / "restored"
        code, out = self._run(["--restore", str(target)])
        self.assertEqual(code, 2)
        self.assertTrue((target / "restored.jsonl").exists())
        self.assertIn("WARNING", out)

    def test_json_mode_emits_a_machine_readable_report(self) -> None:
        self.put_window(0, self.line(0))
        code, out = self._run(["--verify", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["verified_objects"], 1)

    def test_restore_refuses_to_clobber_and_says_so(self) -> None:
        self.put_window(0, self.line(0))
        target = self.root / "restored"
        self.assertEqual(self._run(["--restore", str(target)])[0], 0)
        code, out = self._run(["--restore", str(target)])
        self.assertEqual(code, 1)  # a usage problem: re-run with --force
        self.assertIn("--force", out)
        self.assertEqual(self._run(["--restore", str(target), "--force"])[0], 0)

    def test_status_needs_no_destination_and_no_network(self) -> None:
        code, out = self._run(["--status"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertTrue(payload["restore"]["restore_available"])

    def test_status_works_when_nothing_is_configured(self) -> None:
        config.AUDIT_BACKUP_DESTINATION = ""
        code, out = self._run(["--status"])
        self.assertEqual(code, 0)
        self.assertFalse(json.loads(out)["restore"]["restore_available"])

    def test_a_bad_destination_flag_is_rejected_not_accepted(self) -> None:
        """argparse owns invalid flags and exits 2; the point is that it is refused."""
        with self.assertRaises(SystemExit) as caught:
            self._run(["--verify", "--destination", "dropbox"])
        self.assertNotEqual(caught.exception.code, 0)

    def test_an_unknown_backend_from_config_exits_one(self) -> None:
        """Not reachable through argparse, but reachable through the environment."""
        config.AUDIT_BACKUP_DESTINATION = "dropbox"
        code, _ = self._run(["--verify"])
        self.assertEqual(code, 1)

    def test_no_hash_skips_downloads(self) -> None:
        """The fast path: usable on a large destination without the object reads."""
        self.put_window(0, self.line(0))
        calls = {"n": 0}
        real = audit_backup.FilesystemDestination.get_object

        def counting(inner_self, key):  # noqa: ANN001
            calls["n"] += 1
            return real(inner_self, key)

        with redirect_stdout(io.StringIO()):
            with mock.patch.object(
                audit_backup.FilesystemDestination, "get_object", counting
            ):
                code = self.cli.main(["--verify", "--no-hash"])
        self.assertEqual(code, 0)
        self.assertEqual(calls["n"], 0)


if __name__ == "__main__":
    unittest.main()
