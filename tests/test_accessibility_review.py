"""Offline R13 worksheet integrity using disposable committed Git fixtures."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

from tests import hermetic  # noqa: F401
from app.accuracy_benchmark import BenchmarkInvalid, digest
from scripts.accessibility_review import CHECKS, EXTRA, prepare_packet
from tests import test_operations_review as operations_tests

ROOT = Path(__file__).resolve().parent.parent


class AccessibilityReviewTests(unittest.TestCase):
    git = operations_tests.OperationsReviewTests.git

    def setUp(self):
        operations_tests.OperationsReviewTests.setUp(self)
        for name in EXTRA:
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, path)
        self.git("add", ".")
        self.git("commit", "-qm", "Invented accessibility fixture")

    def cli(self, output):
        return subprocess.run([sys.executable, str(ROOT / "scripts/accessibility_review.py"),
                               "--repo", str(self.repo), "--out", str(output)],
                              text=True, capture_output=True, cwd=ROOT)

    def test_every_actual_check_is_unrun_and_no_approval_or_participant_facts_are_invented(self):
        packet = prepare_packet(self.repo)
        fingerprint = packet.pop("packet_sha256")
        self.assertEqual(digest(packet), fingerprint)
        self.assertEqual(packet["status"], "unapproved")
        self.assertEqual(packet["pilot_admission"], "not_authorized_by_this_tool")
        self.assertEqual(packet["decision_template"]["decision"], "NO-GO")
        self.assertEqual(len(CHECKS), 12)
        self.assertEqual({r["id"] for r in packet["checks"]}, {f"A{i:02}" for i in range(1, 13)})
        self.assertTrue(all(r["status"] == "not_run" and not r["observations"] and not r["evidence_reference"]
                            and not r["participant_alias_references"] and not r["matrix_row_references"]
                            and not r["reviewer_signature_reference"] for r in packet["checks"]))
        self.assertFalse(packet["release_template"]["browser_device_assistive_tool_matrix"])
        self.assertFalse(packet["release_template"]["operator_id"])
        self.assertFalse(packet["decision_template"]["operator_signature_reference"])
        self.assertFalse(packet["decision_template"]["reviewer_signature_reference"])
        self.assertEqual(packet["release_template"]["excluded_scope"], ["file_exports"])
        self.assertNotIn("O01", json.dumps(packet))

    def test_full_source_and_r13_assets_are_revision_bound_without_operational_approval(self):
        packet = prepare_packet(self.repo)
        self.assertEqual(packet["source"]["revision"], self.git("rev-parse", "HEAD"))
        self.assertEqual(packet["release_template"]["source_tree"], self.git("rev-parse", "HEAD^{tree}"))
        self.assertTrue(set(EXTRA).issubset(packet["source"]["file_sha256"]))
        self.assertIn("run_app.py", packet["source"]["file_sha256"])
        self.assertEqual(packet, prepare_packet(self.repo))

    def test_hidden_r13_changes_and_missing_required_protocol_are_refused(self):
        for name in ("app/accessibility.py", "app/views/factual_review.py", "deploy/PILOT_ACCESSIBILITY_ACCEPTANCE.md"):
            with self.subTest(name=name):
                self.git("update-index", "--assume-unchanged", name)
                path = self.repo / name
                raw = path.read_bytes()
                path.write_text("INVENTED_HIDDEN_CHANGE")
                self.assertEqual(self.git("status", "--porcelain"), "")
                with self.assertRaises(BenchmarkInvalid):
                    prepare_packet(self.repo)
                path.write_bytes(raw)
                self.git("update-index", "--no-assume-unchanged", name)
        self.git("rm", "-q", "deploy/PILOT_ACCESSIBILITY_ACCEPTANCE.md")
        self.git("commit", "-qm", "Invented incomplete source")
        with self.assertRaises((BenchmarkInvalid, subprocess.CalledProcessError)):
            prepare_packet(self.repo)

    def test_private_evidence_is_excluded_at_root_and_nested_and_never_read(self):
        canary = "INVENTED_PRIVATE_PARTICIPANT_CANARY_82143"
        for directory in ("accessibility-evidence", "nested/accessibility-evidence"):
            path = self.repo / directory
            path.mkdir(parents=True)
            (path / "observations.json").write_text(canary)
        self.assertEqual(self.git("status", "--porcelain"), "")
        self.assertNotIn(canary, json.dumps(prepare_packet(self.repo)))

    def test_import_loads_no_clients_ui_config_or_telemetry(self):
        code = ('import sys; import scripts.accessibility_review; '
                'assert not {"app.config", "app.llm", "app.telemetry", "streamlit", "openai"}.intersection(sys.modules)')
        run = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_cli_exclusive_output_preserves_existing_private_observations(self):
        output = self.repo.parent / "draft.json"
        first = self.cli(output)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("NO-GO", first.stdout)
        raw = output.read_bytes()
        second = self.cli(output)
        self.assertEqual(second.returncode, 2)
        self.assertEqual(output.read_bytes(), raw)
        self.assertNotIn(str(output), second.stderr)

    def test_cli_dirty_failure_is_generic_and_writes_nothing(self):
        (self.repo / "app/accessibility.py").write_text("INVENTED_FAILURE_CANARY")
        output = self.repo.parent / "not-created.json"
        run = self.cli(output)
        self.assertEqual(run.returncode, 2)
        self.assertFalse(output.exists())
        self.assertNotIn("INVENTED_FAILURE_CANARY", run.stderr)
        self.assertNotIn(str(self.repo), run.stderr)
        self.assertNotIn("Traceback", run.stderr)


if __name__ == "__main__":
    unittest.main()
