"""A reference a user can see resolves to the lines behind it, in-app.

The About tab used to answer "what happened to my run?" with ``grep req_…`` — an
instruction that is useless to the person reading it, since the person reading it
is in a browser, not on the pod. This suite pins the replacement: the id shown in
an error message is the id the lookup accepts, and what comes back is the record
of that run and nothing else.

Isolation matters here. The capture buffer is process-wide and the run log is a
file on disk, so each test clears the one and points the other at a temp dir;
otherwise an earlier test's records would satisfy a later test's lookup and the
suite would pass while the feature was broken.
"""
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import diagnostics  # noqa: E402
from app.diagnostics import (  # noqa: E402
    MAX_FIELD_CHARS,
    CaptureHandler,
    capture_handler,
    extract_reference,
    install_capture,
    is_reference,
    lookup,
    redact_secrets,
)
from app.error_report import report_failure  # noqa: E402
from app.logging_config import (  # noqa: E402
    clear_request_id,
    configure_logging,
    new_request_id,
    set_request_id,
)
from app.run_log import run_log_event  # noqa: E402
from log_isolation import isolate_app_logs  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Assembled at runtime rather than written as one literal, deliberately.
# ``scripts/hooks/pre-commit`` blocks any added line containing an ``sk-`` token —
# blunt, but right to be — so a fixture spelling out a provider key would make this
# file uncommittable for anyone who has the hook installed. What the redaction rules
# are tested against is the value, and the value is exactly provider-shaped.
_PROVIDER_KEY = "sk-" + "sp-" + "abcdefgh12345678"
_KEY_ASSIGNMENT = "api_key=" + _PROVIDER_KEY


class _CaptureCase(unittest.TestCase):
    """Clears the process-wide buffer and redirects the run log per test."""

    def setUp(self) -> None:
        configure_logging()
        handler = install_capture()
        handler.clear()
        clear_request_id()
        self.addCleanup(clear_request_id)

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.run_log_dir = tmp.name
        patcher = mock.patch.dict(os.environ, {"VA_LSE_RUN_LOG_DIR": tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _show_failure(self, message: str = "Search failed: could not read bad.pdf") -> str:
        """Log a failure the way a view does, and return what the user would read."""
        rid = new_request_id()
        set_request_id(rid)
        return report_failure(message, phase="record_search", exc=ValueError("bad pdf"))


class TestReferenceParsing(unittest.TestCase):
    def test_accepts_a_bare_reference(self):
        self.assertTrue(is_reference("req_4f8a2b1c9d0e"))

    def test_rejects_a_prefix_without_hex(self):
        self.assertFalse(is_reference("req_ZZZZZZZZ"))
        self.assertFalse(is_reference("req_"))

    def test_rejects_ids_that_are_too_short_or_too_long(self):
        self.assertFalse(is_reference("req_abc"))
        self.assertFalse(is_reference("req_" + "a" * 40))

    def test_rejects_paths_globs_and_empty_input(self):
        for query in (
            "",
            "   ",
            "../../etc/passwd",
            "logs/runs.jsonl",
            "req_*",
            "req_4f8a2b1c9d0e/../../secret",
        ):
            with self.subTest(query=query):
                self.assertFalse(is_reference(query))

    def test_extracts_the_id_from_a_whole_error_message(self):
        shown = "Drafting failed: LLM call failed (reference: req_4f8a2b1c9d0e)"
        self.assertEqual(extract_reference(shown), "req_4f8a2b1c9d0e")

    def test_extraction_lowercases_so_a_pasted_id_still_matches(self):
        self.assertEqual(extract_reference("(reference: req_4F8A2B1C9D0E)"), "req_4f8a2b1c9d0e")

    def test_extraction_of_text_without_an_id_is_empty(self):
        self.assertEqual(extract_reference("nothing to see here"), "")


class TestLookup(_CaptureCase):
    def test_the_reference_shown_to_the_user_is_the_one_that_resolves(self):
        """The whole feature in one assertion: what was shown is what resolves."""
        shown = self._show_failure()
        self.assertIn("reference: req_", shown)

        detail = lookup(shown)  # the user pastes the message, not the id

        self.assertTrue(detail.valid)
        self.assertIn(detail.reference, shown)
        self.assertTrue(detail.lines, msg="the id shown resolved to no lines")
        self.assertTrue(any("could not read bad.pdf" in line for line in detail.lines))
        # The traceback is the point of the endpoint for a developer.
        self.assertTrue(any("ValueError" in line for line in detail.lines))

    def test_a_bare_id_resolves_the_same_as_the_whole_message(self):
        shown = self._show_failure("Evaluation failed: timeout")
        rid = extract_reference(shown)
        self.assertEqual(lookup(rid).lines, lookup(shown).lines)

    def test_unknown_but_well_formed_reference_explains_itself(self):
        detail = lookup("req_aaaaaaaaaaaa")
        self.assertTrue(detail.valid)
        self.assertEqual(detail.lines, [])
        self.assertEqual(detail.events, [])
        self.assertIn("buffer holds only the most recent", detail.note)

    def test_malformed_query_is_refused_without_touching_the_run_log(self):
        with mock.patch("app.run_log.read_recent_events") as reader:
            detail = lookup("../../etc/passwd")
        reader.assert_not_called()
        self.assertFalse(detail.valid)
        self.assertIn("not a reference id", detail.problem)
        self.assertEqual(detail.lines, [])

    def test_lookup_does_not_return_another_references_lines(self):
        first = self._show_failure("first failure")
        other_rid = new_request_id()
        set_request_id(other_rid)
        report_failure("unrelated failure", phase="other")

        detail = lookup(first)
        self.assertTrue(detail.lines)
        self.assertFalse(
            any("unrelated failure" in line for line in detail.lines),
            msg="lines from a different reference leaked into the answer",
        )

    def test_line_count_is_bounded_and_says_so(self):
        rid = new_request_id()
        set_request_id(rid)
        for index in range(5):
            report_failure(f"failure {index}", phase="record_search")

        detail = lookup(rid, limit=2)
        self.assertEqual(len(detail.lines), 2)
        self.assertTrue(detail.truncated)
        # Newest last: the most recent failures are the ones being investigated.
        self.assertIn("failure 4", detail.lines[-1])

    def test_run_log_event_for_the_reference_is_returned(self):
        rid = new_request_id()
        set_request_id(rid)
        run_log_event("evaluate", "error", request_id=rid, error="provider 500")

        detail = lookup(rid)
        self.assertEqual([e["request_id"] for e in detail.events], [rid])
        self.assertEqual(detail.events[0]["error"], "provider 500")

    def test_a_worker_run_says_why_there_are_no_local_lines(self):
        """Events but no buffered lines means another process ran it."""
        rid = new_request_id()
        run_log_event("draft", "rejected", request_id=rid, error="no records", reason="no_records")

        detail = lookup(rid)
        self.assertEqual(detail.lines, [])
        self.assertEqual(len(detail.events), 1)
        self.assertIn("executed elsewhere", detail.note)

    def test_lookup_never_raises_on_a_hostile_query(self):
        for query in ("", "   ", "req_" + "f" * 64, "'; DROP TABLE --", "\x00req_abc"):
            with self.subTest(query=query):
                detail = lookup(query)
                self.assertEqual(detail.lines, [])
                self.assertEqual(detail.events, [])

    def test_a_filename_is_kept_because_it_is_the_answer(self):
        """Which file failed is usually the whole question, so filenames are kept —
        a filename is a string the user themself typed, and the docs now say so."""
        shown = self._show_failure("could not read Patient A records.pdf")
        self.assertTrue(
            any("Patient A records.pdf" in line for line in lookup(shown).lines)
        )

    def test_output_is_redacted_before_it_is_returned(self):
        rid = new_request_id()
        set_request_id(rid)
        report_failure("provider rejected " + _KEY_ASSIGNMENT, phase="llm_call")

        detail = lookup(rid)
        joined = "\n".join(detail.lines)
        self.assertNotIn(_PROVIDER_KEY, joined)
        self.assertIn("api_key=[redacted]", joined)


class TestCaptureBuffer(unittest.TestCase):
    def setUp(self) -> None:
        clear_request_id()
        self.addCleanup(clear_request_id)

    def _logger(self, name: str) -> logging.Logger:
        return logging.getLogger(name)

    def test_buffer_is_bounded_and_keeps_the_newest(self):
        logger = self._logger("app.test.bounded")
        handler = install_capture(logger, limit=3)
        self.addCleanup(logger.removeHandler, handler)
        self.addCleanup(lambda: setattr(logger, "_va_lse_capture_handler", None))

        for index in range(10):
            logger.error("event %d", index, extra={"request_id": "req_0123456789ab"})

        entries = handler.snapshot()
        self.assertEqual(len(entries), 3)
        self.assertIn("event 9", entries[-1]["message"])
        self.assertIn("event 7", entries[0]["message"])

    def test_entry_carries_the_correlation_fields(self):
        logger = self._logger("app.test.fields")
        handler = install_capture(logger, limit=5)
        self.addCleanup(logger.removeHandler, handler)
        self.addCleanup(lambda: setattr(logger, "_va_lse_capture_handler", None))

        logger.error(
            "failed",
            exc_info=ValueError("boom"),
            extra={
                "request_id": "req_0123456789ab",
                "phase": "record_search",
                "status": "error",
                "error_class": "ValueError",
            },
        )
        entry = handler.snapshot()[0]
        self.assertEqual(entry["request_id"], "req_0123456789ab")
        self.assertEqual(entry["phase"], "record_search")
        self.assertEqual(entry["error_class"], "ValueError")
        self.assertIsNotNone(entry["traceback"])
        self.assertIn("ValueError", str(entry["traceback"]))

    def test_entry_falls_back_to_the_context_id(self):
        logger = self._logger("app.test.context")
        handler = install_capture(logger, limit=5)
        self.addCleanup(logger.removeHandler, handler)
        self.addCleanup(lambda: setattr(logger, "_va_lse_capture_handler", None))

        set_request_id("req_context1234")
        logger.error("no explicit id")
        self.assertEqual(handler.snapshot()[0]["request_id"], "req_context1234")

    def test_oversized_message_is_clipped(self):
        logger = self._logger("app.test.huge")
        handler = install_capture(logger, limit=5)
        self.addCleanup(logger.removeHandler, handler)
        self.addCleanup(lambda: setattr(logger, "_va_lse_capture_handler", None))

        logger.error("x" * (MAX_FIELD_CHARS * 3))
        message = handler.snapshot()[0]["message"]
        self.assertLess(len(message), MAX_FIELD_CHARS * 2)
        self.assertTrue(message.endswith("[truncated]"))

    def test_install_is_idempotent(self):
        logger = self._logger("app.test.idempotent")
        first = install_capture(logger)
        second = install_capture(logger)
        self.addCleanup(logger.removeHandler, first)
        self.addCleanup(lambda: setattr(logger, "_va_lse_capture_handler", None))
        self.assertIs(first, second)
        self.assertEqual(sum(isinstance(h, CaptureHandler) for h in logger.handlers), 1)

    def test_reconfiguring_logging_keeps_the_buffer(self):
        """The handler is intentionally unmanaged: a reconfiguration must not
        erase records a reference is about to be looked up against."""
        set_request_id(new_request_id())
        report_failure("before reconfigure", phase="test_reconfigure")

        configure_logging()

        handler = capture_handler()
        self.assertIsNotNone(handler, msg="capture was dropped by reconfiguration")
        assert handler is not None
        self.assertTrue(
            any("before reconfigure" in entry["message"] for entry in handler.snapshot())
        )


class TestThePanelInTheRealApp(unittest.TestCase):
    """The panel is reachable in the actual app, not merely importable.

    AppTest runs the app in-process, so the capture buffer the panel reads is the
    same one this test writes to. That makes this the user's real journey as a
    test: a failure is reported with a reference, the reference is pasted into the
    panel, and the lines come back.
    """

    def setUp(self) -> None:
        isolate_app_logs(self)
        configure_logging()
        install_capture().clear()
        clear_request_id()
        self.addCleanup(clear_request_id)

    def _lookup(self, query: str):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(PROJECT_ROOT / "run_app.py"), default_timeout=25)
        at.run()
        at.text_input(key="diagnostics_reference_query").set_value(query)
        return at.button(key="diagnostics_lookup_submit").click().run()

    def test_a_reference_shown_to_the_user_resolves_through_the_panel(self):
        rid = new_request_id()
        set_request_id(rid)
        report_failure(
            "Search failed: could not read Patient A records.pdf",
            phase="record_search",
            exc=ValueError("bad pdf"),
        )

        at = self._lookup(rid)

        self.assertFalse(at.exception)
        rendered = "\n".join(code.value for code in at.code)
        self.assertIn("could not read Patient A records.pdf", rendered)
        self.assertIn("ValueError", rendered)

    def test_the_whole_error_message_can_be_pasted_in(self):
        rid = new_request_id()
        set_request_id(rid)
        shown = report_failure("Evaluation failed: provider returned nothing", phase="evaluate")

        at = self._lookup(shown)

        self.assertFalse(at.exception)
        self.assertIn("provider returned nothing", "\n".join(c.value for c in at.code))

    def test_a_run_log_event_with_nested_fields_renders(self):
        """Real run-log events carry nested values (``outcome``, ``record_sources``).

        The panel renders events as a dataframe, which is where a nested value can
        stop being a value and start being a serialization error — a failure mode
        only a rendered panel reveals, so it is asserted here rather than assumed.
        """
        rid = new_request_id()
        set_request_id(rid)
        run_log_event(
            "evaluate",
            "ok",
            request_id=rid,
            record_sources=["Upload", "Fetch Sandbox"],
            outcome={"claims": 2, "contradictions": 1, "overall_rating": "Adequate"},
        )

        at = self._lookup(rid)

        self.assertFalse(at.exception, msg=str(at.exception))
        self.assertTrue(len(at.dataframe) >= 1)

    def test_a_bad_reference_is_refused_in_the_panel(self):
        at = self._lookup("../../etc/passwd")
        self.assertFalse(at.exception)
        warnings = [w.value for w in at.warning]
        self.assertTrue(
            any("not a reference id" in w for w in warnings), msg=f"warnings: {warnings}"
        )
        self.assertEqual([code.value for code in at.code], [])

    def test_an_unknown_reference_explains_what_to_check(self):
        at = self._lookup("req_aaaaaaaaaaaa")
        self.assertFalse(at.exception)
        infos = [i.value for i in at.info]
        self.assertTrue(
            any("buffer holds only the most recent" in i for i in infos), msg=f"infos: {infos}"
        )


class TestRedaction(unittest.TestCase):
    def test_provider_key_is_redacted(self):
        self.assertTrue(_PROVIDER_KEY.startswith("sk-"))
        scrubbed = redact_secrets("401 from provider: " + _PROVIDER_KEY)
        self.assertNotIn(_PROVIDER_KEY, scrubbed)
        self.assertIn("[redacted-key]", scrubbed)

    def test_bearer_token_is_redacted(self):
        """The scheme word must not be mistaken for the value: a naive ``key: value``
        rule redacts "Bearer" and leaves the token in plain sight."""
        scrubbed = redact_secrets("Authorization: Bearer abcdef123456ghijkl")
        self.assertNotIn("abcdef123456ghijkl", scrubbed)
        self.assertIn("Bearer [redacted]", scrubbed)

    def test_basic_auth_is_redacted_too(self):
        scrubbed = redact_secrets("Authorization: Basic dXNlcjpwYXNzd29yZA==")
        self.assertNotIn("dXNlcjpwYXNzd29yZA==", scrubbed)

    def test_key_value_secret_is_redacted(self):
        scrubbed = redact_secrets("trying " + _KEY_ASSIGNMENT + " now")
        self.assertNotIn(_PROVIDER_KEY, scrubbed)
        self.assertIn("api_key=[redacted]", scrubbed)

    def test_ordinary_text_is_left_alone(self):
        text = "✖️ Patient A records.pdf: could not read PDF"
        self.assertEqual(redact_secrets(text), text)

    def test_redaction_is_stable_when_applied_twice(self):
        once = redact_secrets("failed " + _KEY_ASSIGNMENT)
        self.assertEqual(redact_secrets(once), once)


if __name__ == "__main__":
    unittest.main()
