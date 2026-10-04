"""Approval admission regressions; all identities/evidence are invented fixtures."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests import hermetic  # noqa: F401
from app import pilot
from tests.test_controlled_pilot import approval


class ApprovalTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "approval.json"
        self.data = approval()
        self.write()
        environment = patch.dict(os.environ, {
            "VA_LSE_PILOT_APPROVAL_FILE": str(self.path),
            "VA_LSE_BUILD_SHA": self.data["reviewed_revision"],
        })
        environment.start()
        self.addCleanup(environment.stop)

    def write(self, changes=None):
        self.path.write_text(json.dumps({**self.data, **(changes or {})}), encoding="utf-8")
        self.path.chmod(0o644)

    def refused(self):
        with self.assertRaises(pilot.PilotBlocked) as error:
            pilot.load_approval()
        self.assertIn("Pilot admission is closed", str(error.exception))
        self.assertNotIn(str(self.path), str(error.exception))

    def test_valid_owned_readonly_manifest_and_replacement_are_loaded(self):
        self.assertEqual(pilot.load_approval(), self.data)
        replacement = self.path.with_suffix(".new")
        replacement.write_text(json.dumps({**self.data, "subjects": ["operator"]}))
        replacement.chmod(0o444)
        replacement.replace(self.path)
        self.assertEqual(pilot.load_approval()["subjects"], ["operator"])

    def test_full_immutable_revision_and_exact_match_are_required(self):
        for revision in ("test-revision", "a" * 7, "A" * 40, "g" * 40, "a" * 41):
            with self.subTest(revision=revision):
                self.write({"reviewed_revision": revision})
                with patch.dict(os.environ, {"VA_LSE_BUILD_SHA": revision}):
                    self.refused()
        self.write({"reviewed_revision": "b" * 40})
        self.refused()

    def test_timestamp_types_timezones_and_windows_fail_closed(self):
        now = datetime.now(timezone.utc)
        for field in ("approved_at", "expires_at"):
            for value in (None, 1, True, [], {}, "2026-01-01T00:00:00", "invalid"):
                with self.subTest(field=field, value=value):
                    self.write({field: value})
                    self.refused()
        for changes in ({"approved_at": (now + timedelta(minutes=1)).isoformat()},
                        {"expires_at": (now - timedelta(minutes=1)).isoformat()},
                        {"expires_at": (now + timedelta(days=31)).isoformat()}):
            self.write(changes)
            self.refused()

    def test_duplicate_top_level_and_nested_fields_are_refused(self):
        valid = json.dumps(self.data)
        for raw in ('{"schema_version": 0,' + valid[1:],
                    valid.replace('"run_attempts": 50', '"run_attempts": 1, "run_attempts": 50')):
            self.path.write_text(raw)
            self.refused()

    def test_nonfinite_and_overflowing_json_numbers_are_refused(self):
        for value in ("NaN", "Infinity", "-Infinity", "1e9999"):
            self.path.write_text(json.dumps(self.data)[:-1] + ', "extra": ' + value + "}")
            self.refused()

    def test_schema_is_an_integer_and_every_evidence_reference_is_required(self):
        for value in (True, 1.0, "1", None):
            self.write({"schema_version": value})
            self.refused()
        self.assertEqual(len(pilot.EVIDENCE_FIELDS), 8)
        for field in pilot.EVIDENCE_FIELDS:
            for value in ("", " ", None, 1):
                with self.subTest(field=field, value=value):
                    self.write({field: value})
                    self.refused()

    def test_identity_and_model_lists_are_unique_and_unambiguous(self):
        for field in ("subjects", "operators", "models"):
            for values in ([], [True], [{}], [""], [" padded "], ["line\nbreak"],
                           ["control\x00byte"], ["same", "same"]):
                with self.subTest(field=field, values=values):
                    self.write({field: values})
                    self.refused()

    def test_size_encoding_and_deep_json_are_bounded(self):
        for raw in (b" " * 65537, b"\xff", b"[" * 2000 + b"]" * 2000):
            self.path.write_bytes(raw)
            self.refused()

    def test_relative_missing_directory_fifo_and_leaf_link_are_refused(self):
        fifo = self.path.with_name("pipe")
        os.mkfifo(fifo)
        link = self.path.with_name("link")
        link.symlink_to(self.path)
        for path in ("relative.json", self.path.with_name("missing"), self.path.parent, fifo, link):
            with self.subTest(path=path), patch.dict(os.environ, {"VA_LSE_PILOT_APPROVAL_FILE": str(path)}):
                self.refused()

    def test_group_or_world_writable_manifest_is_refused(self):
        for mode in (0o664, 0o646, 0o666):
            self.path.chmod(mode)
            self.refused()

    def test_hardlinked_manifest_is_refused(self):
        alias = self.path.with_name("writable-alias.json")
        os.link(self.path, alias)
        self.refused()
        alias.unlink()
        self.assertEqual(pilot.load_approval(), self.data)

    def test_foreign_owner_is_refused_and_root_readonly_mount_is_allowed(self):
        real_fstat = os.fstat
        def owner_stat(fd, owner):
            original = real_fstat(fd)
            return SimpleNamespace(**{key: getattr(original, key) for key in (
                "st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")}, st_uid=owner)
        with patch("os.fstat", side_effect=lambda fd: owner_stat(fd, os.geteuid() + 10000)):
            self.refused()
        original = self.path.lstat()
        root = SimpleNamespace(**{key: getattr(original, key) for key in (
            "st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")}, st_uid=0)
        with patch("os.fstat", side_effect=lambda fd: owner_stat(fd, 0)), patch.object(Path, "lstat", return_value=root):
            self.assertEqual(pilot.load_approval(), self.data)

    def test_replacement_or_mutation_during_read_is_refused(self):
        real_read = os.read
        for operation in ("replace", "append"):
            self.write()
            changed = False
            def read(fd, count):
                nonlocal changed
                result = real_read(fd, count)
                if not changed:
                    changed = True
                    if operation == "replace":
                        other = self.path.with_suffix(".new")
                        other.write_text(json.dumps(self.data))
                        other.replace(self.path)
                    else:
                        with self.path.open("ab") as stream:
                            stream.write(b" ")
                return result
            with patch("os.read", side_effect=read):
                self.refused()

    def test_actual_read_limit_does_not_trust_reported_size(self):
        self.path.write_bytes(b" " * 70000)
        original = self.path.stat()
        reported = SimpleNamespace(**{key: getattr(original, key) for key in (
            "st_dev", "st_ino", "st_mode", "st_uid", "st_nlink", "st_mtime_ns", "st_ctime_ns")}, st_size=1)
        counts = []
        real_read = os.read
        def read(fd, count):
            counts.append(count)
            return real_read(fd, count)
        with patch("os.fstat", return_value=reported), patch("os.read", side_effect=read):
            self.refused()
        self.assertEqual(sum(counts), 65537)
        self.assertLessEqual(max(counts), 8192)

    def test_unsupported_file_protection_fails_closed(self):
        real_hasattr = hasattr
        with patch("builtins.hasattr", side_effect=lambda obj, key:
                   False if obj is os and key == "O_NOFOLLOW" else real_hasattr(obj, key)):
            self.refused()


if __name__ == "__main__":
    unittest.main()
