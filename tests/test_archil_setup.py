"""Archil references are explicitly allowlisted; mounted knowledge fails closed."""
from tests import hermetic  # noqa: F401

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.config import load_knowledge
from scripts.archil_setup import KNOWLEDGE, reference_release


class ArchilSetupTests(unittest.TestCase):
    def test_allowlist_excludes_private_files_and_release_tracks_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in KNOWLEDGE:
                path = root / 'app/knowledge' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('public')
            (root / 'benchmarks').mkdir()
            (root / 'benchmarks/synthetic-accuracy-v1.json').write_text('{}')
            (root / '.env').write_text('SECRET=never-upload')
            (root / 'statement.txt').write_text('private')
            release, files = reference_release(root)
            self.assertEqual(len(files), 6)
            self.assertNotIn('.env', files)
            self.assertNotIn('statement.txt', files)
            (root / 'app/knowledge/legal_framework.md').write_text('new rules')
            self.assertNotEqual(release, reference_release(root)[0])

    def test_missing_mounted_knowledge_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'VA_LSE_KNOWLEDGE_DIR': tmp}):
            with self.assertRaises(FileNotFoundError):
                load_knowledge('legal_framework.md')
            (Path(tmp) / 'legal_framework.md').write_text('mounted release')
            self.assertEqual(load_knowledge('legal_framework.md'), 'mounted release')
            with self.assertRaises(ValueError):
                load_knowledge('../legal_framework.md')

    def test_reference_symlinks_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'app/knowledge/legal_framework.md'
            path.parent.mkdir(parents=True)
            (root / 'private').write_text('private')
            path.symlink_to(root / 'private')
            with self.assertRaises(ValueError):
                reference_release(root)
