"""Tests for the persistent run log (app/run_log.py) and its view wiring.

The run log exists so every ``req_…`` reference shown to a user is
correlatable to a durable JSON line — including runs that fail *before* the
audit log's ``start`` event (input validation, LLM setup, shutdown gate).
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import run_log as run_log_mod
from app.run_log import read_recent_events, run_log_event


def _isolated_run_log():
    """Point the run log at a fresh temp dir and return its path.

    ``run_log_event`` resolves the dir per call, so patching the env for the
    duration of the test fully isolates it (no cross-test bleed)."""
    tmp = tempfile.TemporaryDirectory()
    patcher = mock.patch.dict(os.environ, {"VA_LSE_RUN_LOG_DIR": tmp.name})
    patcher.start()
    return tmp, patcher


class RunLogTests(unittest.TestCase):
    """run_log_event writes JSON lines to VA_LSE_RUN_LOG_DIR."""

    def setUp(self) -> None:
        self._tmp, self._patcher = _isolated_run_log()
        self.addCleanup(self._patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def _path(self) -> Path:
        return Path(self._tmp.name) / "runs.jsonl"

    def test_event_appends_json_line(self):
        run_log_event("draft", "rejected", request_id="req_test01", error="no records", reason="no_records")
        lines = self._path().read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        event = json.loads(lines[0])
        self.assertEqual(event["action"], "draft")
        self.assertEqual(event["status"], "rejected")
        self.assertEqual(event["request_id"], "req_test01")
        self.assertEqual(event["reason"], "no_records")
        self.assertIn("timestamp", event)

    def test_error_fields_recorded_and_truncated(self):
        run_log_event(
            "draft", "error", request_id="req_test02",
            error="ValueError: " + "x" * 900, error_class="ValueError",
            traceback="Traceback (most recent call last): ...",
        )
        event = json.loads(self._path().read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(event["error_class"], "ValueError")
        self.assertLessEqual(len(event["error"]), 300)
        self.assertIn("Traceback", event["traceback"])

    def test_none_extra_values_skipped(self):
        run_log_event("evaluate", "ok", request_id="req_test03", duration_ms=None, claims=2)
        event = json.loads(self._path().read_text(encoding="utf-8").splitlines()[0])
        self.assertNotIn("duration_ms", event)
        self.assertEqual(event["claims"], 2)

    def test_disabled_env_writes_nothing(self):
        self._patcher.stop()
        env = mock.patch.dict(os.environ, {"VA_LSE_RUN_LOG_DISABLED": "1"})
        env.start()
        self.addCleanup(env.stop)
        run_log_event("draft", "error", request_id="req_test04", error="boom")
        self.assertFalse(self._path().exists())

    def test_never_raises_on_bad_dir(self):
        env = mock.patch.dict(os.environ, {"VA_LSE_RUN_LOG_DIR": "/dev/null/not/a/dir"})
        env.start()
        self.addCleanup(env.stop)
        # Must not raise.
        run_log_event("draft", "error", request_id="req_test05", error="boom")

    def test_read_recent_events_roundtrip(self):
        for i in range(5):
            run_log_event("draft", "ok", request_id=f"req_rt{i}")
        events = read_recent_events(limit=3)
        self.assertEqual(len(events), 3)
        self.assertEqual(events[-1]["request_id"], "req_rt4")


class ViewWiringTests(unittest.TestCase):
    """Draft/Evaluate run paths emit run-log events for every terminal state."""

    def setUp(self) -> None:
        self._tmp, self._patcher = _isolated_run_log()
        self.addCleanup(self._patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def _events_for(self, rid: str) -> list[dict]:
        return [e for e in read_recent_events(limit=500) if e.get("request_id") == rid]

    def test_draft_validation_failure_logged_with_reference(self):
        """The pre-pipeline rejection path (which never reaches audit) is
        recorded so the shown reference is always resolvable."""
        import app.views.draft_view as dv

        rid = "req_valfail1"
        fake_st = mock.MagicMock()
        with mock.patch.object(dv.st, "error", fake_st.error), \
             mock.patch.object(dv.st, "session_state", mock.MagicMock(get=lambda k, d=None: None)):
            ok = dv._validate_draft_inputs(records=[], observations="", condition="", rid=rid)
        self.assertFalse(ok)
        fake_st.error.assert_called_once()
        events = self._events_for(rid)
        self.assertEqual([e["status"] for e in events], ["rejected"])
        self.assertEqual(events[0]["reason"], "no_records")

    def test_draft_error_path_emits_run_log_event(self):
        import app.views.draft_view as dv

        rid = "req_errpath1"
        exc = RuntimeError("llm exploded")
        with mock.patch.object(dv.audit_log, "audit_draft_error") as aerr, \
             mock.patch.object(dv.logger, "error"):
            dv._handle_draft_error(
                rid, exc, t0=0.0, condition=None, sources=None, files=0, pages=0,
            )
        aerr.assert_called_once()
        events = self._events_for(rid)
        self.assertEqual([e["status"] for e in events], ["error"])
        self.assertEqual(events[0]["error_class"], "RuntimeError")
        self.assertIn("llm exploded", events[0]["error"])

    def test_evaluate_validation_failure_logged(self):
        import app.views.evaluate_view as ev

        rid_holder = {"rid": "req_evalfail1"}
        with mock.patch.object(ev, "get_request_id", return_value=rid_holder["rid"]), \
             mock.patch.object(ev.st, "error") as err:
            ok = ev._validate_evaluate_inputs(statement_text="   ", records=[])
        self.assertFalse(ok)
        err.assert_called_once()
        events = self._events_for(rid_holder["rid"])
        reasons = {e["reason"] for e in events}
        self.assertIn("no_statement", reasons)


if __name__ == "__main__":
    unittest.main()
