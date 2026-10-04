"""Source-bound review preparation using only disposable local Git fixtures."""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests import hermetic  # noqa: E402,F401 (isolated test configuration)

from app.accuracy_benchmark import BenchmarkInvalid, digest, read_json, source_snapshot
from scripts.legal_review import (
    EXTRA, KNOWLEDGE, POLICIES, REGISTER, SCENARIOS, committed_text,
    knowledge_units, prepare_packet, prompt_units, validate_seed,
)

ROOT = Path(__file__).resolve().parent.parent


class LegalReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "source"
        self.repo.mkdir()
        for name in {*EXTRA, *KNOWLEDGE, *POLICIES, "requirements.lock"}:
            dest = self.repo / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
        self.git("init", "-q")
        self.git("config", "user.name", "Invented Test Fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.git("add", ".")
        self.git("commit", "-qm", "Invented disposable fixture")
        self.register = read_json(self.repo / REGISTER)
        self.scenarios = read_json(self.repo / SCENARIOS)
        self.source = source_snapshot(self.repo)
        for name in EXTRA:
            committed_text(self.repo, name, self.source)

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args],
                                       stderr=subprocess.PIPE, text=True).strip()

    def validate(self, register=None, scenarios=None):
        validate_seed(register if register is not None else self.register,
                      scenarios if scenarios is not None else self.scenarios, self.source)

    def test_packet_is_unapproved_and_all_findings_blank(self):
        packet = prepare_packet(self.repo)
        fingerprint = packet.pop("packet_sha256")
        self.assertEqual(digest(packet), fingerprint)
        self.assertEqual(packet["status"], "unapproved")
        self.assertEqual(packet["pilot_admission"], "not_authorized_by_this_tool")
        ledger = packet["reviewer_template"]
        self.assertEqual(ledger["signature_reference"], "")
        self.assertTrue(all(r["status"] == "unverified" and r["rule_findings"] == []
                            for r in ledger["unit_findings"]))
        self.assertTrue(all(r["status"] == "not_run" for r in ledger["scenario_findings"]))
        self.assertEqual({r["unit_id"] for r in ledger["unit_findings"]},
                         {r["id"] for r in packet["review_units"]})
        self.assertIn("app/condition_topics.json", packet["source"]["file_sha256"])
        self.assertTrue(set(EXTRA).issubset(packet["source"]["file_sha256"]))

    def test_all_knowledge_bytes_and_dimensions_are_preserved(self):
        packet = prepare_packet(self.repo)
        for name in KNOWLEDGE:
            units = [r for r in packet["review_units"] if r["path"] == name]
            self.assertEqual("".join(r["text"] for r in units).encode(), (self.repo / name).read_bytes())
            self.assertEqual(units[0]["start_line"], 1)
            for first, second in zip(units, units[1:]):
                self.assertEqual(first["end_line"] + 1, second["start_line"])
        rubric = [r for r in packet["review_units"] if r["path"].endswith("evaluation_rubric.md")]
        self.assertTrue(any(r["text"].startswith("5. **Continuity") for r in rubric))
        self.assertTrue(any("Writer guidelines" in r["text"] for r in packet["review_units"]))
        sample = "Preamble\r\n## Test\r\n1. **Rule**\r\nExact text\r\n"
        self.assertEqual("".join(r["text"] for r in knowledge_units("sample", sample)), sample)

    def test_full_policy_files_cover_scoring_ui_and_condition_mapping(self):
        packet = prepare_packet(self.repo)
        files = {r["path"]: r for r in packet["review_units"] if r["kind"] == "complete_policy_file"}
        self.assertTrue(set(POLICIES).issubset(files))
        self.assertIn("weighted", files["app/evaluate.py"]["text"])
        self.assertIn("AA_FORCED_TOPICS", files["app/condition_selector.py"]["text"])

    def test_prompt_scan_never_evaluates_expressions_and_covers_new_modules(self):
        content = 'SYSTEM = dangerous_call()\n\ndef helper():\n    prompt = f"Dynamic {dangerous_call()}"\n'
        self.assertEqual(len(prompt_units("app/new.py", content)), 2)
        (self.repo / "app/new.py").write_text(content)
        self.git("add", "app/new.py")
        self.git("commit", "-qm", "New prompt fixture")
        packet = prepare_packet(self.repo)
        self.assertEqual(len([r for r in packet["review_units"] if r["path"] == "app/new.py"]), 2)

    def test_import_does_not_load_settings_clients_or_telemetry(self):
        code = ('import sys; import scripts.legal_review; '
                'assert not {"app.config", "app.llm", "app.telemetry", "app.perplexity_agent", '
                '"streamlit", "openai"}.intersection(sys.modules); print("offline")')
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True,
                                capture_output=True, env={**os.environ, "VA_LSE_PILOT": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "offline")

    def test_dirty_or_nonignored_untracked_source_is_rejected(self):
        (self.repo / KNOWLEDGE[0]).write_text("Invented changed guidance")
        with self.assertRaises(BenchmarkInvalid):
            prepare_packet(self.repo)
        self.git("restore", KNOWLEDGE[0])
        (self.repo / "untracked.txt").write_text("Invented source")
        with self.assertRaises(BenchmarkInvalid):
            prepare_packet(self.repo)

    def test_hidden_asset_change_is_rejected(self):
        name = "app/condition_topics.json"
        self.git("update-index", "--assume-unchanged", name)
        (self.repo / name).write_text('{"invented":true}')
        self.assertEqual(self.git("status", "--porcelain"), "")
        with self.assertRaises(BenchmarkInvalid):
            prepare_packet(self.repo)

    def test_hidden_register_change_is_rejected(self):
        self.git("update-index", "--assume-unchanged", REGISTER)
        (self.repo / REGISTER).write_text('{}')
        self.assertEqual(self.git("status", "--porcelain"), "")
        with self.assertRaises(BenchmarkInvalid):
            prepare_packet(self.repo)

    def test_primary_source_urls_and_unverified_interpretations_required(self):
        for value in ("https://example.invalid/legal", "http://www.va.gov/forms/", 123,
                      "https://secret@www.va.gov/forms/"):
            changed = copy.deepcopy(self.register)
            changed["authorities"][0]["url"] = value
            with self.subTest(value=value), self.assertRaises(BenchmarkInvalid):
                self.validate(changed)
        changed = copy.deepcopy(self.register)
        changed["authorities"][0]["interpretation_status"] = "accepted"
        with self.assertRaises(BenchmarkInvalid):
            self.validate(changed)
        changed = copy.deepcopy(self.register)
        changed["controls"][0]["status"] = "accepted"
        with self.assertRaises(BenchmarkInvalid):
            self.validate(changed)

    def test_unknown_or_missing_required_source_scope_is_rejected(self):
        changed = copy.deepcopy(self.register)
        changed["controls"][0]["paths"].append("../../private.txt")
        with self.assertRaises(BenchmarkInvalid):
            self.validate(changed)
        for missing in (KNOWLEDGE[0], "app/condition_topics.json", "app/views/about_view.py"):
            changed = copy.deepcopy(self.register)
            for row in changed["controls"]:
                row["paths"] = [p for p in row["paths"] if p != missing]
            with self.subTest(missing=missing), self.assertRaises(BenchmarkInvalid):
                self.validate(changed)

    def test_both_applicability_branches_and_all_controls_required(self):
        for field in ("applicable_example", "inapplicable_example"):
            changed = copy.deepcopy(self.scenarios)
            changed["cases"][0][field] = " "
            with self.subTest(field=field), self.assertRaises(BenchmarkInvalid):
                self.validate(scenarios=changed)
        changed = copy.deepcopy(self.scenarios)
        changed["cases"].pop()
        with self.assertRaises(BenchmarkInvalid):
            self.validate(scenarios=changed)
        changed = copy.deepcopy(self.register)
        changed["controls"].pop()
        with self.assertRaises(BenchmarkInvalid):
            self.validate(changed)

    def test_malformed_and_duplicate_rows_are_rejected(self):
        for key in ("authorities", "controls"):
            for value in (None, "bad", [], [None]):
                changed = copy.deepcopy(self.register)
                changed[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(BenchmarkInvalid):
                    self.validate(changed)
            changed = copy.deepcopy(self.register)
            changed[key].append(copy.deepcopy(changed[key][0]))
            with self.assertRaises(BenchmarkInvalid):
                self.validate(changed)
        changed = copy.deepcopy(self.scenarios)
        changed["cases"].append(changed["cases"][0])
        with self.assertRaises(BenchmarkInvalid):
            self.validate(scenarios=changed)

    def test_changed_committed_guidance_requires_a_new_packet(self):
        before = prepare_packet(self.repo)
        path = self.repo / KNOWLEDGE[0]
        path.write_bytes(path.read_bytes() + b'\n## Invented changed scope\nReview again.\n')
        self.git("add", KNOWLEDGE[0])
        self.git("commit", "-qm", "Changed knowledge fixture")
        after = prepare_packet(self.repo)
        self.assertNotEqual(before["packet_sha256"], after["packet_sha256"])
        self.assertNotEqual(before["source"]["tree"], after["source"]["tree"])
        self.assertTrue(any("Invented changed scope" in r["text"] for r in after["review_units"]))

    def test_cli_prepares_only_and_refuses_existing_output(self):
        out = Path(self.temp.name) / "packet.json"
        command = [sys.executable, str(ROOT / "scripts/legal_review.py"), "--repo", str(self.repo),
                   "--out", str(out)]
        first = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("NO-GO", first.stdout)
        preserved = out.read_bytes()
        second = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(second.returncode, 2)
        self.assertEqual(out.read_bytes(), preserved)
        self.assertNotIn("Traceback", second.stderr)

    def test_cli_error_does_not_expose_source_or_write_approval(self):
        (self.repo / "untracked.txt").write_text("invented-private-marker")
        out = Path(self.temp.name) / "packet.json"
        result = subprocess.run([sys.executable, str(ROOT / "scripts/legal_review.py"),
                                 "--repo", str(self.repo), "--out", str(out)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(out.exists())
        self.assertNotIn("invented-private-marker", result.stdout + result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_invalid_committed_python_fails_without_traceback_or_source_echo(self):
        (self.repo / "app/new.py").write_text('SYSTEM = (invented_private_marker\n')
        self.git("add", "app/new.py")
        self.git("commit", "-qm", "Invalid syntax fixture")
        out = Path(self.temp.name) / "packet.json"
        result = subprocess.run([sys.executable, str(ROOT / "scripts/legal_review.py"),
                                 "--repo", str(self.repo), "--out", str(out)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(out.exists())
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn("invented_private_marker", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
