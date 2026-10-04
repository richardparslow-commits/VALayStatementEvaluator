"""Offline R16 worksheet integrity using disposable committed Git fixtures."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests import hermetic  # noqa: F401
from app.accuracy_benchmark import BenchmarkInvalid, digest
from scripts.deletion_review import CHECKS, EXTRA, prepare_packet

ROOT = Path(__file__).resolve().parent.parent


class DeletionReviewTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.repo = Path(directory.name) / "source"
        self.repo.mkdir()
        for name in (*EXTRA, "requirements.lock", "app/pilot.py", "app/knowledge/legal_framework.md",
                     "run_app.py", "requirements-parser.lock", "scripts/backup_audit_logs.py"):
            target = self.repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        self.git("init", "-q")
        self.git("config", "user.name", "Invented Fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.git("add", ".")
        self.git("commit", "-qm", "Invented deletion fixture")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args],
                                       stderr=subprocess.PIPE, text=True).strip()

    def cli(self, output):
        return subprocess.run([sys.executable, str(ROOT / "scripts/deletion_review.py"),
                               "--repo", str(self.repo), "--out", str(output)],
                              text=True, capture_output=True, cwd=ROOT)

    def test_packet_starts_unapproved_with_20_unrun_checks_and_blank_signatures(self):
        packet = prepare_packet(self.repo)
        fingerprint = packet.pop("packet_sha256")
        self.assertEqual(digest(packet), fingerprint)
        self.assertEqual(packet["status"], "unapproved")
        self.assertEqual(packet["pilot_admission"], "not_authorized_by_this_tool")
        self.assertEqual(packet["decision_template"]["decision"], "NO-GO")
        self.assertTrue(all(not value for key, value in packet["decision_template"].items()
                            if key.endswith("signature_reference")))
        self.assertEqual({r["id"] for r in packet["checks"]}, {f"D{i:02}" for i in range(1, 21)})
        self.assertTrue(all(r["status"] == "not_run" and not r["observations"]
                            and not r["evidence_reference"] and not r["reviewer_signature_reference"]
                            for r in packet["checks"]))
        release = packet["design_template"]
        self.assertEqual(packet["durable_storage_activation"], "not_implemented_or_authorized_by_this_tool")
        self.assertEqual(packet["current_durable_storage_scope"], "synthetic_only")
        self.assertEqual(packet["case_deletion_execution"], "not_performed_by_this_tool")
        self.assertTrue(all(not value for value in release["accepted_prerequisite_references"].values()))
        self.assertFalse(release["operator_id"] or release["private_host_reference"])
        self.assertEqual(len(CHECKS), 20)

    def test_all_fixed_deployment_and_application_bytes_are_revision_bound(self):
        packet = prepare_packet(self.repo)
        self.assertEqual(packet["source"]["revision"], self.git("rev-parse", "HEAD"))
        self.assertEqual(packet["design_template"]["reviewed_revision"], packet["source"]["revision"])
        fingerprints = packet["source"]["file_sha256"]
        self.assertTrue(set(EXTRA).issubset(fingerprints))
        self.assertIn("app/pilot.py", fingerprints)
        self.assertIn("requirements.lock", fingerprints)
        for name in ("run_app.py", "requirements-parser.lock", "scripts/backup_audit_logs.py"):
            self.assertIn(name, fingerprints)
        self.assertEqual(packet, prepare_packet(self.repo))

    def test_dirty_or_untracked_source_is_refused(self):
        for name in ("app/storage_policy.py", "app/blob_cleanup.py", "docker-compose.pilot.yml", "new-untracked.txt"):
            with self.subTest(name=name):
                path = self.repo / name
                existed = path.exists()
                raw = path.read_bytes() if existed else b""
                path.write_text("Invented changed source")
                with self.assertRaises(BenchmarkInvalid):
                    prepare_packet(self.repo)
                if existed:
                    path.write_bytes(raw)
                else:
                    path.unlink()

    def test_hidden_index_changes_cannot_escape_source_binding(self):
        for name in ("app/storage_policy.py", "deploy/PILOT_DELETION_ACCEPTANCE.md", "run_app.py",
                     "requirements-parser.lock", "scripts/backup_audit_logs.py"):
            with self.subTest(name=name):
                self.git("update-index", "--assume-unchanged", name)
                path = self.repo / name
                raw = path.read_bytes()
                path.write_text("Invented hidden change")
                self.assertEqual(self.git("status", "--porcelain"), "")
                with self.assertRaises(BenchmarkInvalid):
                    prepare_packet(self.repo)
                path.write_bytes(raw)
                self.git("update-index", "--no-assume-unchanged", name)

    def test_binary_build_assets_are_hashed_and_compared_without_text_decoding(self):
        asset = self.repo / "release-asset.bin"
        asset.write_bytes(b"\xff\x00INVENTED_BINARY_FIXTURE")
        self.git("add", "-f", "release-asset.bin")
        self.git("commit", "-qm", "Invented binary build input")
        self.assertIn("release-asset.bin", prepare_packet(self.repo)["source"]["file_sha256"])
        self.git("update-index", "--assume-unchanged", "release-asset.bin")
        asset.write_bytes(b"\x80\x00INVENTED_CHANGED_ASSET")
        self.assertEqual(self.git("status", "--porcelain"), "")
        with self.assertRaises(BenchmarkInvalid):
            prepare_packet(self.repo)

    def test_missing_required_tracked_deployment_file_is_refused(self):
        self.git("rm", "-q", "deploy/monitoring/alertmanager.yml")
        self.git("commit", "-qm", "Invented incomplete release")
        with self.assertRaises((BenchmarkInvalid, subprocess.CalledProcessError)):
            prepare_packet(self.repo)

    def test_leaf_and_parent_links_are_refused_even_when_git_status_is_hidden(self):
        path = self.repo / "app/pilot.py"
        self.git("update-index", "--assume-unchanged", "app/pilot.py")
        other = self.repo.parent / "same-content.py"
        other.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(other)
        with self.assertRaises(BenchmarkInvalid):
            prepare_packet(self.repo)
        path.unlink()
        path.write_bytes(other.read_bytes())
        self.git("update-index", "--no-assume-unchanged", "app/pilot.py")
        names = self.git("ls-files", "deploy").splitlines()
        self.git("update-index", "--assume-unchanged", *names)
        deploy = self.repo / "deploy"
        outside = self.repo.parent / "same-deploy"
        deploy.rename(outside)
        deploy.symlink_to(outside, target_is_directory=True)
        # Hide the directory link as an ignored untracked entry, independently
        # of the assumed-unchanged tracked children. The byte/path check must
        # still reject it when Git's status looks clean.
        with (self.repo / ".git/info/exclude").open("a") as stream:
            stream.write("\n/deploy\n")
        self.assertEqual(self.git("status", "--porcelain"), "")
        with self.assertRaises(BenchmarkInvalid):
            prepare_packet(self.repo)

    def test_ignored_local_evidence_is_never_read_or_packaged(self):
        private = self.repo / "deletion-evidence"
        private.mkdir()
        canary = "INVENTED_PRIVATE_OPERATOR_CANARY_94831"
        (private / "accepted-release.json").write_text(canary)
        (self.repo / ".env").write_text(canary)
        self.assertEqual(self.git("status", "--porcelain"), "")
        self.assertNotIn(canary, json.dumps(prepare_packet(self.repo)))

    def test_import_loads_no_clients_settings_ui_or_telemetry(self):
        code = ('import sys; import scripts.deletion_review; '
                'assert not {"app.config", "app.llm", "app.telemetry", "app.perplexity_agent", '
                '"streamlit", "openai"}.intersection(sys.modules); print("offline")')
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "offline")

    def test_cli_creates_draft_and_preserves_existing_output(self):
        output = self.repo.parent / "draft.json"
        first = self.cli(output)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("NO-GO", first.stdout)
        raw = output.read_bytes()
        second = self.cli(output)
        self.assertEqual(second.returncode, 2)
        self.assertEqual(output.read_bytes(), raw)
        self.assertNotIn(str(output), second.stderr)

    def test_cli_failure_is_generic_and_writes_no_packet(self):
        (self.repo / "app/pilot.py").write_text("INVENTED_ERROR_CANARY")
        output = self.repo.parent / "not-created.json"
        result = self.cli(output)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(output.exists())
        self.assertNotIn("INVENTED_ERROR_CANARY", result.stderr)
        self.assertNotIn(str(self.repo), result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
