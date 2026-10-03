"""Explicit temporary synthetic control ledgers; never operator files."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from tests import hermetic  # noqa: F401
from app import pilot_budget


def install_budget(case, data):
    temporary = tempfile.TemporaryDirectory()
    case.addCleanup(temporary.cleanup)
    path = Path(temporary.name) / 'pilot-budget.sqlite3'
    pilot_budget.provision(path, data)
    env = patch.dict(os.environ, {'VA_LSE_PILOT_BUDGET_FILE': str(path)})
    env.start()
    case.addCleanup(env.stop)
    def release():
        with pilot_budget._instance_lock:
            if pilot_budget._instance is not None:
                pilot_budget._instance.close()
                pilot_budget._instance = None
    release()
    case.addCleanup(release)
    return path
