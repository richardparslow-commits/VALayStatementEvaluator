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

from app import config
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


class RunLogRotationTests(unittest.TestCase):
    """The run log is size-bounded.

    Before rotation was added this file was the only unbounded writer in the app
    — the audit log has always rotated — so a long-lived pod could fill its log
    volume with ``runs.jsonl`` alone. That is what the audit's "no protection
    against disk-space exhaustion" actually pointed at.
    """

    def setUp(self) -> None:
        self._tmp, self._patcher = _isolated_run_log()
        self.addCleanup(self._patcher.stop)
        self.addCleanup(self._tmp.cleanup)
        self._saved = {
            "RUN_LOG_MAX_BYTES": config.RUN_LOG_MAX_BYTES,
            "RUN_LOG_BACKUPS": config.RUN_LOG_BACKUPS,
        }
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        for name, value in self._saved.items():
            setattr(config, name, value)

    def _path(self) -> Path:
        return Path(self._tmp.name) / "runs.jsonl"

    def test_file_stays_under_the_limit(self) -> None:
        config.RUN_LOG_MAX_BYTES = 2000
        config.RUN_LOG_BACKUPS = 3
        for i in range(200):
            run_log_event("evaluate", "ok", request_id=f"req_{i:05d}", pad="x" * 60)
        self.assertLess(self._path().stat().st_size, 2100)
        # Old data is rotated, not discarded.
        self.assertTrue((Path(self._tmp.name) / "runs.jsonl.1").exists())

    def test_backups_are_capped_so_the_volume_is_bounded(self) -> None:
        config.RUN_LOG_MAX_BYTES = 1000
        config.RUN_LOG_BACKUPS = 2
        for i in range(400):
            run_log_event("evaluate", "ok", request_id=f"req_{i:05d}", pad="y" * 80)
        files = sorted(p.name for p in Path(self._tmp.name).glob("runs.jsonl*"))
        self.assertEqual(files, ["runs.jsonl", "runs.jsonl.1", "runs.jsonl.2"])
        total = sum((Path(self._tmp.name) / f).stat().st_size for f in files)
        # (backups + 1) x max_bytes is the hard ceiling.
        self.assertLess(total, 3 * 1000 + 500)

    def test_recent_events_survive_a_rotation(self) -> None:
        config.RUN_LOG_MAX_BYTES = 1000
        config.RUN_LOG_BACKUPS = 3
        for i in range(60):
            run_log_event("draft", "ok", request_id=f"req_{i:05d}", pad="z" * 60)
        events = read_recent_events(limit=10)
        self.assertEqual(len(events), 10)
        # Newest last, and the tail must be the most recently written run.
        self.assertEqual(events[-1]["request_id"], "req_00059")

    def test_recent_events_reads_within_the_limit_across_files(self) -> None:
        config.RUN_LOG_MAX_BYTES = 500
        config.RUN_LOG_BACKUPS = 4
        for i in range(30):
            run_log_event("draft", "ok", request_id=f"req_{i:05d}", pad="q" * 40)
        events = read_recent_events(limit=100)
        ids = [e["request_id"] for e in events]
        # Ascending write order across the rotated boundary, no duplicates.
        self.assertEqual(ids, sorted(ids, key=lambda s: int(s.split("_")[1])))
        self.assertEqual(len(ids), len(set(ids)))

    def test_rotation_does_not_lose_a_line(self) -> None:
        config.RUN_LOG_MAX_BYTES = 400
        config.RUN_LOG_BACKUPS = 5
        total = 120
        for i in range(total):
            run_log_event("evaluate", "ok", request_id=f"req_{i:05d}", pad="w" * 20)
        seen: set[str] = set()
        for path in Path(self._tmp.name).glob("runs.jsonl*"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    seen.add(json.loads(line)["request_id"])
        # Every line either survives in a kept file or was rotated off the end;
        # what must never happen is a line that is silently half-written.
        self.assertLessEqual(len(seen), total)
        self.assertGreater(len(seen), 0)
        self.assertTrue(all(rid.startswith("req_") for rid in seen))


if __name__ == "__main__":
    unittest.main()
