"""Unit tests for ``scripts/archil_sandbox_runner.py`` and its integration with
``app/extractors.py`` and ``scripts/check_sandbox.py``."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests import hermetic  # noqa: E402,F401
from app.documents import extract_document  # noqa: E402
from app.extractors import CommandBoxRunner, SandboxExtractor  # noqa: E402
from scripts import archil_sandbox_runner as archil_runner  # noqa: E402
from scripts import check_sandbox  # noqa: E402


FAKE_DISK_CLI_SCRIPT = """#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from pathlib import Path

argv = sys.argv[1:]
journal = os.environ.get("FAKE_ARCHIL_JOURNAL", "")
if journal:
    with open(journal, "a", encoding="utf-8") as fh:
        fh.write(" ".join(argv) + "\\n")

state_root = Path(os.environ["FAKE_ARCHIL_ROOT"])
state_root.mkdir(parents=True, exist_ok=True)

# Strip global options (--token VAL, --region VAL)
clean = []
i = 0
while i < len(argv):
    if argv[i] in ("--token", "--region"):
        i += 2
        continue
    clean.append(argv[i])
    i += 1

if not clean:
    sys.exit(1)

cmd = clean[0]
if cmd == "list":
    if os.environ.get("FAKE_ARCHIL_REFUSE_CREDENTIAL") == "1":
        print("401 Unauthorized: invalid Archil token", file=sys.stderr)
        sys.exit(1)
    print("valay-eval-sandbox  10 GiB  ready")
    sys.exit(0)

if cmd == "branch" and len(clean) >= 4 and clean[1] == "create":
    disk, branch = clean[2], clean[3]
    (state_root / f"{disk}__{branch}").mkdir(parents=True, exist_ok=True)
    sys.exit(0)

if cmd == "branch" and len(clean) >= 4 and clean[1] == "delete":
    disk, branch = clean[2], clean[3]
    shutil.rmtree(state_root / f"{disk}__{branch}", ignore_errors=True)
    sys.exit(0)

if cmd == "cp" and len(clean) >= 3:
    src, dst = clean[1], clean[2]
    if ":" in dst and not Path(dst).is_absolute():
        disk, branch, remote_path = dst.split(":", 2)
        target = state_root / f"{disk}__{branch}" / remote_path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        sys.exit(0)
    if ":" in src and not Path(src).is_absolute():
        disk, branch, remote_path = src.split(":", 2)
        source = state_root / f"{disk}__{branch}" / remote_path.lstrip("/")
        shutil.copy2(source, dst)
        sys.exit(0)

if cmd == "exec":
    disk_spec = ""
    sep = clean.index("--")
    for idx in range(1, sep):
        if clean[idx] == "--disk":
            disk_spec = clean[idx + 1]
    if ":" in disk_spec:
        disk, branch = disk_spec.split(":", 1)
        box_root = state_root / f"{disk}__{branch}"
    else:
        mount_env = os.environ.get("VA_LSE_ARCHIL_MOUNT_DIR", "")
        box_root = Path(mount_env) if mount_env else state_root / disk_spec
    subcmd = clean[sep + 1:]
    if subcmd[:2] == ["mkdir", "-p"]:
        (box_root / subcmd[2].lstrip("/")).mkdir(parents=True, exist_ok=True)
        sys.exit(0)
    if subcmd[:2] == ["rm", "-rf"]:
        for target_rel in subcmd[2:]:
            shutil.rmtree(box_root / target_rel.lstrip("/"), ignore_errors=True)
        sys.exit(0)
    mapped = []
    for tok in subcmd:
        if tok == "/app/scripts/ocr_and_extract.py":
            mapped.append(str(Path(__file__).resolve().parent / "scripts" / "ocr_and_extract.py"))
        elif tok.startswith("/work/"):
            rel = tok[len("/work/"):]
            mapped.append(str(box_root / rel) if ":" not in disk_spec else str(box_root / tok.lstrip("/")))
        else:
            mapped.append(tok)
    mapped[0] = sys.executable
    proc = subprocess.run(mapped, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    sys.exit(proc.returncode)

sys.exit(1)
"""


class TestArchilSandboxRunner(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="va-lse-archil-test-")
        self.tmp = Path(self._tmp.name)
        self.fake_cli = self.tmp / "fake_disk_cli.py"
        (self.tmp / "scripts").symlink_to(PROJECT_ROOT / "scripts")
        self.fake_cli.write_text(FAKE_DISK_CLI_SCRIPT, encoding="utf-8")
        self.fake_cli.chmod(0o755)
        self.journal = self.tmp / "journal.log"
        self.state_root = self.tmp / "archil_state"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _env(self, **extra: str) -> dict[str, str]:
        base = {
            "VA_LSE_ARCHIL_CLI": f"{sys.executable} {self.fake_cli}",
            "VA_LSE_ARCHIL_DISK": "valay-eval-sandbox",
            "VA_LSE_ARCHIL_TOKEN": "archil_tok_test_secret_999",
            "FAKE_ARCHIL_JOURNAL": str(self.journal),
            "FAKE_ARCHIL_ROOT": str(self.state_root),
        }
        base.update(extra)
        return base

    def test_llm_keys_in_archil_token_slot_are_refused_by_name(self) -> None:
        for bad_key in ("pplx-12345", "sk-proj-12345", "vck_12345", "vcg_12345"):
            with patch.dict(os.environ, {"VA_LSE_ARCHIL_TOKEN": bad_key}, clear=False):
                with self.assertRaises(archil_runner.SandboxError) as ctx:
                    archil_runner.Settings.from_env()
                self.assertIn("console.archil.com", str(ctx.exception))

    def test_relative_label_blocks_traversal_and_preserves_nested_paths(self) -> None:
        self.assertEqual(
            str(archil_runner.relative_label("records/2024/clinic_note.pdf")),
            "records/2024/clinic_note.pdf",
        )
        for bad in ("", ".", "../secret.txt", "records/../../etc/passwd"):
            with self.assertRaises(archil_runner.SandboxError):
                archil_runner.relative_label(bad)

    def test_read_manifest_verifies_sha256_integrity(self) -> None:
        work = self.tmp / "work_sha"
        work.mkdir()
        record = work / "note.txt"
        record.write_bytes(b"original content")
        good_sha = hashlib.sha256(b"original content").hexdigest()
        manifest = work / "manifest.json"
        manifest.write_text(
            json.dumps({"file": {"label": "note.txt", "path": str(record), "sha256": good_sha}}),
            encoding="utf-8",
        )
        staged = archil_runner.read_manifest(work)
        self.assertEqual(staged.label, "note.txt")
        record.write_bytes(b"tampered content")
        with self.assertRaises(archil_runner.SandboxError) as ctx:
            archil_runner.read_manifest(work)
        self.assertIn("sha256", str(ctx.exception))

    def test_box_free_check_succeeds_and_never_creates_a_branch(self) -> None:
        with patch.dict(os.environ, self._env(), clear=False):
            settings = archil_runner.Settings.from_env()
            verdict = archil_runner.self_check(settings)
        self.assertTrue(verdict.reachable)
        calls = self.journal.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(calls), 1)
        self.assertIn("list", calls[0])

    def test_box_free_check_validates_mount_dir_when_configured(self) -> None:
        missing_mount = self.tmp / "nonexistent_mount"
        with patch.dict(
            os.environ,
            self._env(VA_LSE_ARCHIL_MOUNT_DIR=str(missing_mount)),
            clear=False,
        ):
            verdict = archil_runner.self_check(archil_runner.Settings.from_env())
        self.assertFalse(verdict.reachable)
        self.assertEqual([c.name for c in verdict.failed], ["mount_dir"])

        live_mount = self.tmp / "live_mount"
        live_mount.mkdir()
        with patch.dict(
            os.environ,
            self._env(VA_LSE_ARCHIL_MOUNT_DIR=str(live_mount)),
            clear=False,
        ):
            verdict_ok = archil_runner.self_check(archil_runner.Settings.from_env())
        self.assertTrue(verdict_ok.reachable)

    def test_check_sandbox_integrates_with_archil_runner_end_to_end(self) -> None:
        runner_cmd = f"{sys.executable} {PROJECT_ROOT / 'scripts' / 'archil_sandbox_runner.py'} {{work}}"
        with patch.dict(
            os.environ,
            self._env(VA_LSE_EXTRACTOR="sandbox", VA_LSE_EXTRACTOR_RUNNER=runner_cmd),
            clear=False,
        ):
            with patch.object(check_sandbox.config, "EXTRACTOR_MODE", "sandbox"), patch.object(
                check_sandbox.config, "EXTRACTOR_RUNNER", runner_cmd
            ):
                verdict = check_sandbox.diagnose()
        self.assertTrue(verdict.reachable, verdict.verdict())

    def test_end_to_end_extraction_through_cow_branch_path(self) -> None:
        runner_cmd = f"{sys.executable} {PROJECT_ROOT / 'scripts' / 'archil_sandbox_runner.py'} {{work}}"
        sample_bytes = b"Department of Veterans Affairs Progress Note: Knee pain since 2019."
        label = "records/2024/clinic_note.txt"
        with patch.dict(os.environ, self._env(), clear=False):
            extractor = SandboxExtractor(CommandBoxRunner(runner_cmd))
            docs, skipped = extractor.extract(label, sample_bytes)
        in_proc_doc = extract_document(label, sample_bytes)
        self.assertEqual(skipped, [])
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].filename, label)
        self.assertEqual(docs[0].full_text, in_proc_doc.full_text)
        remaining_branches = list(self.state_root.glob("*"))
        self.assertEqual(remaining_branches, [])
        self.assertEqual(archil_runner._masked(["disk", "list", "--token", "secret"]), "disk list --token …")

    def test_end_to_end_extraction_through_zero_copy_mount_path(self) -> None:
        mount_dir = self.tmp / "archil_mount"
        mount_dir.mkdir()
        runner_cmd = f"{sys.executable} {PROJECT_ROOT / 'scripts' / 'archil_sandbox_runner.py'} {{work}}"
        sample_bytes = b"VA Audiology Note: Bilateral tinnitus reported after artillery service."
        label = "records/audiology_2025.txt"
        with patch.dict(
            os.environ,
            self._env(VA_LSE_ARCHIL_MOUNT_DIR=str(mount_dir)),
            clear=False,
        ):
            extractor = SandboxExtractor(CommandBoxRunner(runner_cmd))
            docs, skipped = extractor.extract(label, sample_bytes)
        in_proc_doc = extract_document(label, sample_bytes)
        self.assertEqual(skipped, [])
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].filename, label)
        self.assertEqual(docs[0].full_text, in_proc_doc.full_text)
        # Verify zero-copy path did not invoke branch create or cp, and cleaned up its job dir
        calls = self.journal.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].startswith("exec "))
        self.assertEqual(list((mount_dir / "jobs").glob("*")), [])


if __name__ == "__main__":
    unittest.main()
