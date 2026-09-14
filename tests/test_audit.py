"""Offline tests for app/audit.py — audit trail for Evaluate/Draft.

Covers:
* Required fields (timestamp/action/status/request_id/user_session_id)
* Dedicated `audit` logger separate from diagnostic `app.log`
* Session id stability and explicit override
* File output (JSON lines) to a tmp directory
* No PII leakage — only metadata/classifications, never statement/record text
* Wrapper helpers (audit_evaluate_*/audit_draft_*) emit correct action/status
* Condition truncation, record_sources, outcome, error fields
* Best-effort: audit never raises
"""

from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _capture_audit_payloads():
    """Install a capturing handler on the audit logger; return (records, handler)."""
    from app import audit as audit_mod

    audit_logger = audit_mod.get_audit_logger()
    captured: list[dict] = []

    class _Cap(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # noqa: A003
            payload = getattr(record, "audit_payload", None)
            if isinstance(payload, dict):
                captured.append(dict(payload))
            else:
                try:
                    captured.append(json.loads(record.getMessage()))
                except Exception:  # noqa: BLE001
                    captured.append({"message": record.getMessage()})

    handler = _Cap()
    handler.setLevel(logging.INFO)
    setattr(handler, "_va_lse_audit_test", True)  # type: ignore[attr-defined]
    audit_logger.addHandler(handler)
    return captured, handler


def _cleanup_capture(handler: logging.Handler) -> None:
    from app import audit as audit_mod

    audit_logger = logging.getLogger("audit")
    try:
        audit_logger.removeHandler(handler)
    except Exception:  # noqa: BLE001
        pass
    try:
        handler.close()
    except Exception:  # noqa: BLE001
        pass


class AuditLoggerTest(unittest.TestCase):
    def setUp(self) -> None:
        from app import audit as audit_mod

        audit_mod._reset_for_tests()  # type: ignore[attr-defined]
        # Ensure a clean tmp-isolated configuration for each test where needed.
        audit_mod.configure_audit_logging(log_dir=tempfile.mkdtemp(), force=True)

    def tearDown(self) -> None:
        from app import audit as audit_mod

        audit_mod._reset_for_tests()  # type: ignore[attr-defined]

    # -- basic fields -------------------------------------------------------

    def test_required_fields_present(self) -> None:
        from app import audit as audit_mod

        captured, h = _capture_audit_payloads()
        try:
            audit_mod.audit_event("evaluate", "start", request_id="req_abc123")
            self.assertEqual(len(captured), 1)
            payload = captured[0]
            self.assertEqual(payload.get("action"), "evaluate")
            self.assertEqual(payload.get("status"), "start")
            self.assertEqual(payload.get("request_id"), "req_abc123")
            self.assertIn("timestamp", payload)
            self.assertIn("user_session_id", payload)
            sess = str(payload.get("user_session_id", ""))
            self.assertTrue(sess.startswith("sess_"), sess)
        finally:
            _cleanup_capture(h)

    def test_user_session_id_explicit_overrides_auto(self) -> None:
        from app import audit as audit_mod

        captured, h = _capture_audit_payloads()
        try:
            audit_mod.audit_event(
                "draft", "start", request_id="req_x", user_session_id="sess_explicit"
            )
            self.assertEqual(captured[0].get("user_session_id"), "sess_explicit")
        finally:
            _cleanup_capture(h)

    def test_session_id_stable_without_streamlit(self) -> None:
        from app import audit as audit_mod

        audit_mod._reset_for_tests()  # type: ignore[attr-defined]
        a = audit_mod.get_audit_session_id()
        b = audit_mod.get_audit_session_id()
        self.assertEqual(a, b)
        self.assertTrue(a.startswith("sess_"))

    # -- separate stream from diagnostic logging ---------------------------

    def test_audit_logger_is_separate_from_app_logger(self) -> None:
        from app import audit as audit_mod

        audit_logger = audit_mod.get_audit_logger()
        app_logger = logging.getLogger("app")
        self.assertEqual(audit_logger.name, "audit")
        self.assertNotEqual(audit_logger.name, app_logger.name)
        # Audit logger does not propagate to app; it is an independent stream.
        self.assertFalse(audit_logger.propagate)

    # -- condition truncation & sources ------------------------------------

    def test_condition_truncated_to_120(self) -> None:
        from app import audit as audit_mod

        long_cond = "X" * 500
        captured, h = _capture_audit_payloads()
        try:
            audit_mod.audit_event("evaluate", "start", request_id="req_1", condition=long_cond)
            cond = str(captured[0].get("condition", ""))
            self.assertLessEqual(len(cond), 121)  # 120 + ellipsis
            self.assertTrue(cond.endswith("…") or len(cond) == 120)
        finally:
            _cleanup_capture(h)

    def test_record_sources_and_counts(self) -> None:
        from app import audit as audit_mod

        captured, h = _capture_audit_payloads()
        try:
            audit_mod.audit_event(
                "evaluate",
                "ok",
                request_id="req_2",
                record_sources=["Upload", "VA.gov"],
                record_files=3,
                record_pages=42,
                duration_ms=1234,
                outcome={"overall_rating": "Strong", "claims": 5},
            )
            p = captured[0]
            self.assertEqual(p.get("record_sources"), ["Upload", "VA.gov"])
            self.assertEqual(p.get("record_files"), 3)
            self.assertEqual(p.get("record_pages"), 42)
            self.assertEqual(p.get("duration_ms"), 1234)
            self.assertEqual(p.get("outcome", {}).get("overall_rating"), "Strong")
        finally:
            _cleanup_capture(h)

    def test_outcome_is_sanitized_to_serializable(self) -> None:
        from app import audit as audit_mod

        captured, h = _capture_audit_payloads()
        try:
            audit_mod.audit_event(
                "draft",
                "ok",
                request_id="req_3",
                outcome={"draft_chars": 1200, "nested": {"a": 1}},
            )
            self.assertIn("outcome", captured[0])
        finally:
            _cleanup_capture(h)

    # -- error fields ------------------------------------------------------

    def test_error_fields_present(self) -> None:
        from app import audit as audit_mod

        captured, h = _capture_audit_payloads()
        try:
            err = RuntimeError("model timeout after retries")
            audit_mod.audit_event(
                "evaluate", "error", request_id="req_4", error_class=type(err).__name__, error_message=str(err)
            )
            p = captured[0]
            self.assertEqual(p.get("error_class"), "RuntimeError")
            self.assertIn("model timeout", str(p.get("error_message", "")))
        finally:
            _cleanup_capture(h)

    def test_error_message_truncated(self) -> None:
        from app import audit as audit_mod

        captured, h = _capture_audit_payloads()
        try:
            audit_mod.audit_event(
                "draft", "error", request_id="req_5", error_message="E" * 1000
            )
            msg = str(captured[0].get("error_message", ""))
            self.assertLessEqual(len(msg), 301)
        finally:
            _cleanup_capture(h)

    # -- wrapper helpers ---------------------------------------------------

    def test_evaluate_wrappers_emit_correct_action_status(self) -> None:
        from app import audit as audit_mod

        captured, h = _capture_audit_payloads()
        try:
            audit_mod.audit_evaluate_start(request_id="req_e1", condition="PTSD")
            audit_mod.audit_evaluate_ok(request_id="req_e2", duration_ms=10, outcome={"claims": 2})
            audit_mod.audit_evaluate_error(request_id="req_e3", duration_ms=20, error=ValueError("bad input"))
            self.assertEqual(len(captured), 3)
            self.assertEqual(captured[0].get("action"), "evaluate")
            self.assertEqual(captured[0].get("status"), "start")
            self.assertEqual(captured[1].get("status"), "ok")
            self.assertEqual(captured[2].get("status"), "error")
            self.assertEqual(captured[2].get("error_class"), "ValueError")
        finally:
            _cleanup_capture(h)

    def test_draft_wrappers_emit_correct_action_status(self) -> None:
        from app import audit as audit_mod

        captured, h = _capture_audit_payloads()
        try:
            audit_mod.audit_draft_start(request_id="req_d1", condition="tinnitus")
            audit_mod.audit_draft_ok(request_id="req_d2", duration_ms=15)
            audit_mod.audit_draft_error(request_id="req_d3", duration_ms=25, error=RuntimeError("llm down"))
            self.assertEqual(captured[0].get("action"), "draft")
            self.assertEqual(captured[2].get("error_class"), "RuntimeError")
        finally:
            _cleanup_capture(h)

    # -- no PII ------------------------------------------------------------

    def test_no_pii_keys_in_payload(self) -> None:
        """Ensure audit payload never contains sensitive free-text fields."""
        from app import audit as audit_mod

        captured, h = _capture_audit_payloads()
        try:
            # Even if caller mistakenly puts large text in condition, it is truncated;
            # record text / statement text must never appear as a payload key.
            audit_mod.audit_evaluate_start(
                request_id="req_pii",
                condition="PTSD",
                record_sources=["Upload"],
                record_files=1,
                record_pages=5,
            )
            payload = captured[0]
            forbidden = {"statement", "observations", "record_text", "veteran_name", "witness_name", "file_content", "prompt", "body"}
            payload_keys = {k.lower() for k in payload.keys()}
            self.assertTrue(payload_keys.isdisjoint(forbidden), f"payload leaked PII keys: {payload_keys & forbidden}")
            # Condition is classification only, never full statement.
            self.assertNotIn("John Doe", json.dumps(payload))
        finally:
            _cleanup_capture(h)

    def test_audit_never_raises(self) -> None:
        from app import audit as audit_mod

        # Pass pathological values; audit_event must swallow exceptions.
        try:
            audit_mod.audit_event("evaluate", "start", request_id="")  # type: ignore[arg-type]
            audit_mod.audit_event("draft", "ok", request_id="req_ok", outcome=None)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            self.fail(f"audit_event raised: {exc}")

    # -- file output -------------------------------------------------------

    def test_file_output_is_json_lines_with_required_fields(self) -> None:
        from app import audit as audit_mod

        with tempfile.TemporaryDirectory() as tmp:
            audit_mod._reset_for_tests()  # type: ignore[attr-defined]
            audit_mod.configure_audit_logging(log_dir=tmp, log_file="audit.log", force=True)
            audit_mod.audit_evaluate_start(request_id="req_file1", condition="lumbar strain", record_sources=["Upload"], record_files=2, record_pages=10)
            audit_mod.audit_evaluate_ok(request_id="req_file1", duration_ms=999, outcome={"claims": 3, "overall_rating": "Adequate"})
            # Flush handlers to ensure file is written.
            for h in logging.getLogger("audit").handlers:
                try:
                    h.flush()  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass
            audit_path = Path(tmp) / "audit.log"
            self.assertTrue(audit_path.exists(), f"audit log not created at {audit_path}")
            lines = [ln.strip() for ln in audit_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertGreaterEqual(len(lines), 2)
            first = json.loads(lines[0])
            # File lines are the same structured payload as captured above.
            self.assertEqual(first.get("action"), "evaluate")
            self.assertEqual(first.get("status"), "start")
            self.assertIn("request_id", first)
            self.assertIn("user_session_id", first)
            self.assertIn("timestamp", first)
            # No PII in file either.
            self.assertNotIn("statement", json.dumps(first).lower())
            second = json.loads(lines[1])
            self.assertEqual(second.get("status"), "ok")
            self.assertIn("duration_ms", second)
            audit_mod._reset_for_tests()  # type: ignore[attr-defined]
            audit_mod.configure_audit_logging(log_dir=tempfile.mkdtemp(), force=True)

    def test_audit_log_configurable_via_env(self) -> None:
        from app import audit as audit_mod

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"VA_LSE_AUDIT_LOG_DIR": tmp, "VA_LSE_AUDIT_LOG_FILE": "custom_audit.log"}):
                audit_mod._reset_for_tests()  # type: ignore[attr-defined]
                audit_mod.configure_audit_logging(force=True)
                audit_mod.audit_event("draft", "start", request_id="req_env")
                for h in logging.getLogger("audit").handlers:
                    try:
                        h.flush()  # type: ignore[attr-defined]
                    except Exception:  # noqa: BLE001
                        pass
                custom_path = Path(tmp) / "custom_audit.log"
                self.assertTrue(custom_path.exists())
                audit_mod._reset_for_tests()  # type: ignore[attr-defined]
                audit_mod.configure_audit_logging(log_dir=tempfile.mkdtemp(), force=True)
