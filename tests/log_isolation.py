"""Log-directory isolation for AppTest-driven tests.

``streamlit.testing.v1.AppTest`` runs the *real* app in-process, so a
test-driven Evaluate/Draft run appends genuine-looking lifecycle events to the
developer's real ``logs/runs.jsonl`` and ``logs/audit.log``. Those synthetic
events are indistinguishable from real use afterwards — in the About tab's
"Recent runs" panel and, more importantly, in ``grep req_…`` forensics, which
is the app's entire support story for a user-visible ``req_…`` reference.

``isolate_app_logs`` moves all three persisted streams (run log, audit log,
usage watchdog history) into a per-test temp directory, and re-points the
already-configured audit logger at it. The audit logger is configured once per
process, so without the ``force`` reconfigure an early test would pin every
later run to the first test's directory (or, worse, to ``logs/``).
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock


def isolate_app_logs(case: unittest.TestCase) -> str:
    """Redirect run-log, audit-log, and watchdog writes to a temp dir.

    Call from ``setUp``; cleanup is registered on *case*. Returns the temp
    directory path (handy for assertions).
    """
    tmp = tempfile.TemporaryDirectory()
    case.addCleanup(tmp.cleanup)

    # Import here so callers keep their own sys.path setup at module import.
    from app import audit as audit_log
    from app.config import AUDIT_LOG_DIR

    # Resolved before the env patch, so this is the real configured directory.
    real_audit_dir = AUDIT_LOG_DIR

    patcher = mock.patch.dict(
        os.environ,
        {
            "VA_LSE_RUN_LOG_DIR": tmp.name,
            "VA_LSE_AUDIT_LOG_DIR": tmp.name,
            "VA_LSE_WATCHDOG_PATH": os.path.join(tmp.name, "usage_history.json"),
        },
    )
    patcher.start()
    case.addCleanup(patcher.stop)

    # Re-point the singleton audit logger at the temp dir, then put it back.
    audit_log.configure_audit_logging(log_dir=tmp.name, force=True)
    case.addCleanup(
        lambda: audit_log.configure_audit_logging(log_dir=real_audit_dir, force=True)
    )
    return tmp.name
