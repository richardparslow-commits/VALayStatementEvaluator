"""Every user-facing failure carries a correlation id and a log line.

A failure the user can see is a failure somebody will have to find afterwards.
Two view paths used to render ``str(exc)`` with neither a reference nor a log
line — the VA.gov authentication failure and the settings validation errors —
so the only record of what happened was the sentence on screen. The extractor's
per-file skip messages were the same shape: built as strings, shown, never
logged, so "it said my file was skipped" had nothing behind it.

The static test below is the guard against that coming back: any ``st.error`` /
``st.warning`` whose text depends on an exception must obtain that text from a
reporter helper, which is what attaches the id and writes the traceback. A new
bare render fails the suite rather than shipping.
"""
import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.error_report import (  # noqa: E402
    ensure_request_id,
    format_error_for_user,
    reference_suffix,
    report_failure,
)
from app.logging_config import (  # noqa: E402
    clear_request_id,
    get_request_id,
    set_request_id,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"

# Anything that renders a failure must carry its reference through one of these.
REPORTERS = frozenset({"report_failure", "format_error_for_user", "reference_suffix"})


def _is_streamlit_render(node: ast.AST) -> bool:
    """``st.error(...)`` / ``st.warning(...)`` — the two user-facing channels."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "st"
        and node.func.attr in ("error", "warning")
    )


def _names_in(call: ast.Call) -> set[str]:
    return {n.id for n in ast.walk(call) if isinstance(n, ast.Name)}


def _looks_like_exception(name: str) -> bool:
    return name in ("exc", "err", "error", "exception") or name.endswith("_exc")


def _unattributable_renders() -> list[str]:
    """Renders whose text comes from an exception but not from a reporter."""
    offenders: list[str] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not _is_streamlit_render(node):
                continue
            assert isinstance(node, ast.Call)
            names = _names_in(node)
            exceptions = {n for n in names if _looks_like_exception(n)}
            if not exceptions or names & REPORTERS:
                continue
            rel = path.relative_to(PROJECT_ROOT)
            offenders.append(f"{rel}:{node.lineno} renders {sorted(exceptions)} with no reference")
    return offenders


class TestEveryFailureIsAttributable(unittest.TestCase):
    def test_exception_renders_route_through_a_reporter(self):
        offenders = _unattributable_renders()
        self.assertEqual(
            offenders,
            [],
            msg=(
                "These user-facing renders show an exception without a correlation "
                "id, so a user who hits them has nothing to quote and the log has "
                "no matching line. Wrap the text in report_failure(...) (adds the "
                "reference and logs the traceback) or format_error_for_user(exc, "
                "rid) when the run id is already known:\n  " + "\n  ".join(offenders)
            ),
        )

    def test_the_guard_detects_a_bare_render(self):
        """The scan must fail on the shape it forbids, not merely pass today."""
        tree = ast.parse('st.error(f"failed: {exc}")')
        render = tree.body[0].value
        assert isinstance(render, ast.Call)
        names = _names_in(render)
        self.assertIn("exc", {n for n in names if _looks_like_exception(n)})
        self.assertFalse(names & REPORTERS)

        routed = ast.parse('st.error(report_failure(f"failed: {exc}", phase="p", exc=exc))')
        routed_call = routed.body[0].value
        assert isinstance(routed_call, ast.Call)
        self.assertTrue(_names_in(routed_call) & REPORTERS)


class TestReportFailure(unittest.TestCase):
    def setUp(self) -> None:
        clear_request_id()
        # ``once`` dedupes per process; reset so tests do not see each other.
        from app import error_report

        error_report._ONCE_SEEN.clear()
        self.addCleanup(clear_request_id)

    def test_mints_a_reference_when_none_is_active(self):
        with self.assertLogs("app.error_report", level="ERROR") as captured:
            message = report_failure("queue unreachable", phase="job_queue")
        record = captured.records[0]
        rid = getattr(record, "request_id", "")
        self.assertTrue(rid.startswith("req_"), msg=f"no correlation id: {rid}")
        self.assertEqual(message, f"queue unreachable (reference: {rid})")
        # Activated, not merely printed: everything logged later in this request
        # shares the id the user was shown.
        self.assertEqual(get_request_id(), rid)

    def test_reuses_a_supplied_request_id(self):
        with self.assertLogs("app.error_report", level="ERROR") as captured:
            message = report_failure(
                "could not be queued", phase="draft_enqueue", request_id="req_known1234"
            )
        self.assertEqual(message, "could not be queued (reference: req_known1234)")
        self.assertEqual(getattr(captured.records[0], "request_id", ""), "req_known1234")

    def test_the_reference_returned_is_the_one_logged(self):
        """The whole point: the id shown resolves to the line written."""
        with self.assertLogs("app.error_report", level="ERROR") as captured:
            message = report_failure("boom", phase="p")
        self.assertIn("reference: ", message)
        rid = message.rsplit("reference: ", 1)[1].rstrip(")")
        self.assertEqual(rid, getattr(captured.records[0], "request_id", ""))

    def test_logs_a_traceback_when_an_exception_is_supplied(self):
        with self.assertLogs("app.error_report", level="ERROR") as captured:
            report_failure("failed", phase="p", exc=ValueError("bad"))
        self.assertIsNotNone(captured.records[0].exc_info)
        self.assertEqual(getattr(captured.records[0], "error_class", None), "ValueError")

    def test_warnings_log_at_warning_level(self):
        with self.assertLogs("app.error_report", level="WARNING") as captured:
            report_failure("skipped", phase="upload_extract", severity="warning")
        self.assertEqual(captured.records[0].levelname, "WARNING")

    def test_once_logs_the_first_occurrence_only(self):
        with self.assertLogs("app.error_report", level="ERROR") as captured:
            first = report_failure("persistent failure", phase="panel", once=True)
            second = report_failure("persistent failure", phase="panel", once=True)
        self.assertEqual(len(captured.records), 1)
        # The reference stays resolvable, so the user still gets a usable id.
        self.assertEqual(first, second)

    def test_once_is_per_phase(self):
        with self.assertLogs("app.error_report", level="ERROR") as captured:
            report_failure("same text", phase="panel_a", once=True)
            report_failure("same text", phase="panel_b", once=True)
        self.assertEqual(len(captured.records), 2)

    def test_once_registry_is_bounded(self):
        from app import error_report

        with self.assertLogs("app.error_report", level="ERROR"):
            for i in range(error_report._ONCE_LIMIT + 10):
                report_failure(f"failure {i}", phase="flood", once=True)
        self.assertLessEqual(len(error_report._ONCE_SEEN), error_report._ONCE_LIMIT)


class TestReferenceHelpers(unittest.TestCase):
    def test_reference_suffix_omits_missing_ids(self):
        self.assertEqual(reference_suffix(""), "")
        self.assertEqual(reference_suffix("-"), "")
        self.assertEqual(reference_suffix("req_abc"), " (reference: req_abc)")

    def test_ensure_request_id_activates_a_minted_id(self):
        clear_request_id()
        rid = ensure_request_id()
        self.assertTrue(rid.startswith("req_"))
        self.assertEqual(get_request_id(), rid)
        # Idempotent: a second call keeps the id the first one established.
        self.assertEqual(ensure_request_id(), rid)
        clear_request_id()

    def test_format_error_for_user_carries_the_id(self):
        self.assertEqual(
            format_error_for_user(ValueError("boom"), "req_abc"), "boom (reference: req_abc)"
        )
        self.assertEqual(format_error_for_user(ValueError("boom"), "-"), "boom")

    def test_the_context_id_is_used_when_present(self):
        set_request_id("req_context99")
        with self.assertLogs("app.error_report", level="ERROR") as captured:
            message = report_failure("failed", phase="p")
        self.assertEqual(message, "failed (reference: req_context99)")
        self.assertEqual(getattr(captured.records[0], "request_id", ""), "req_context99")
        clear_request_id()


if __name__ == "__main__":
    unittest.main()
