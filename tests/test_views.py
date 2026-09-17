"""Unit tests for the app/views layer — mocked Streamlit, no AppTest.

Locks the app↔views boundary without spinning up the whole Streamlit runtime:
pure helpers run directly; st-touching helpers run under a MagicMock ``st``
plus a fake session_state dict.
"""

from __future__ import annotations

import sys
import types
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.usage import UsageTracker  # noqa: E402


def _fake_streamlit() -> tuple[MagicMock, dict]:
    """Return (st_mock, session_dict) wired so st.session_state is a real dict.

    The dict subclass mirrors Streamlit's SessionState, which supports both
    item access (``st.session_state["k"]``) and attribute access
    (``st.session_state.settings``).
    """
    st_mock = MagicMock()

    class _SessionState(dict):
        def __getattr__(self, name: str):  # noqa: ANN202 - dict passthrough
            try:
                return self[name]
            except KeyError as exc:  # mirror attribute semantics
                raise AttributeError(name) from exc

        def __setattr__(self, name: str, value) -> None:  # noqa: ANN001
            self[name] = value

    session: dict = _SessionState()
    st_mock.session_state = session
    return st_mock, session


def _patch_st(module: types.ModuleType, st_mock: MagicMock):
    return patch.object(module, "st", st_mock)


class _UploadedFile:
    """Minimal UploadedFile stand-in (name + size, like Streamlit's)."""

    def __init__(self, name: str, size: int) -> None:
        self.name = name
        self.size = size


# ---------------------------------------------------------------- shared.py
class TestFormatErrorForUser(unittest.TestCase):
    def test_includes_reference_when_present(self) -> None:
        from app.views.shared import format_error_for_user

        msg = format_error_for_user(ValueError("boom"), "req_abc123")
        self.assertEqual(msg, "boom (reference: req_abc123)")

    def test_omits_reference_for_dash(self) -> None:
        from app.views.shared import format_error_for_user

        self.assertEqual(format_error_for_user(ValueError("boom"), "-"), "boom")

    def test_omits_reference_for_empty(self) -> None:
        from app.views.shared import format_error_for_user

        self.assertEqual(format_error_for_user(ValueError("boom"), ""), "boom")


class TestCorrelationIds(unittest.TestCase):
    def test_get_or_create_mints_once_and_reuses(self) -> None:
        import app.views.shared as shared

        st_mock, session = _fake_streamlit()
        with _patch_st(shared, st_mock):
            rid1 = shared.get_or_create_request_id()
            self.assertTrue(rid1.startswith("req_"))
            rid2 = shared.get_or_create_request_id()
            self.assertEqual(rid1, rid2)
            self.assertEqual(session[shared.REQUEST_ID_KEY], rid1)

    def test_new_run_request_id_rotates(self) -> None:
        import app.views.shared as shared

        st_mock, session = _fake_streamlit()
        with _patch_st(shared, st_mock):
            rid1 = shared.new_run_request_id()
            rid2 = shared.new_run_request_id()
            self.assertNotEqual(rid1, rid2)
            self.assertEqual(session[shared.REQUEST_ID_KEY], rid2)


class TestCheckShutdownGate(unittest.TestCase):
    def test_passes_when_not_shutting_down(self) -> None:
        import app.views.shared as shared

        st_mock, _ = _fake_streamlit()
        with _patch_st(shared, st_mock), patch.object(shared, "is_shutting_down", return_value=False):
            self.assertTrue(shared.check_shutdown_gate("draft"))
            st_mock.error.assert_not_called()

    def test_blocks_and_explains_when_draining(self) -> None:
        import app.views.shared as shared

        st_mock, _ = _fake_streamlit()
        with _patch_st(shared, st_mock), patch.object(shared, "is_shutting_down", return_value=True):
            self.assertFalse(shared.check_shutdown_gate("draft"))
            msg = str(st_mock.error.call_args[0][0])
            self.assertIn("shutting down", msg)
            self.assertIn("draft", msg)


class TestAuditMeta(unittest.TestCase):
    def test_audit_record_meta_counts_pages(self) -> None:
        import app.views.shared as shared

        st_mock, _ = _fake_streamlit()
        doc = MagicMock(pages=["p1", "p2", "p3"], __class__=MagicMock)
        records = [doc, MagicMock(pages=["p4"])]
        with _patch_st(shared, st_mock):
            sources, files, pages = shared.audit_record_meta("eval", records)
        self.assertEqual(files, 2)
        self.assertEqual(pages, 4)
        self.assertEqual(sources, ["Upload"])  # fallback label

    def test_audit_record_meta_uses_source_store(self) -> None:
        import app.views.shared as shared

        st_mock, session = _fake_streamlit()
        session["source_records_eval"] = {"VA.gov": [MagicMock(pages=["p1"])]}
        with _patch_st(shared, st_mock):
            sources, files, pages = shared.audit_record_meta("eval", [])
        self.assertEqual(sources, ["VA.gov"])
        self.assertEqual((files, pages), (0, 0))

    def test_audit_condition_for_slot_reads_selector_state(self) -> None:
        import app.views.shared as shared

        st_mock, session = _fake_streamlit()
        session["selected_conditions_draft"] = [["ptsd", "PTSD"], ["tinn", "Tinnitus"]]
        with _patch_st(shared, st_mock):
            self.assertEqual(shared.audit_condition_for_slot("draft"), "PTSD, Tinnitus")

    def test_audit_condition_for_slot_empty(self) -> None:
        import app.views.shared as shared

        st_mock, _ = _fake_streamlit()
        with _patch_st(shared, st_mock):
            self.assertEqual(shared.audit_condition_for_slot("draft"), "")


# ---------------------------------------------------------------- uploads.py
class TestCheckUploadLimits(unittest.TestCase):
    def setUp(self) -> None:
        import app.config as config

        self._per = config.MAX_UPLOAD_BYTES
        self._total = config.MAX_TOTAL_UPLOAD_BYTES

    def test_empty_returns_empty(self) -> None:
        from app.views.uploads import check_upload_limits

        self.assertEqual(check_upload_limits([]), ([], []))

    def test_oversized_file_rejected(self) -> None:
        from app.views.uploads import check_upload_limits
        import app.config as config

        with patch.object(config, "MAX_UPLOAD_BYTES", 10), patch.object(
            config, "MAX_TOTAL_UPLOAD_BYTES", 100
        ):
            accepted, msgs = check_upload_limits([_UploadedFile("big.pdf", 11)])
        self.assertEqual(accepted, [])
        self.assertEqual(len(msgs), 1)
        self.assertIn("big.pdf", msgs[0])

    def test_batch_total_drops_largest(self) -> None:
        from app.views.uploads import check_upload_limits
        import app.config as config

        files = [
            _UploadedFile("a.txt", 6),
            _UploadedFile("b.txt", 7),
            _UploadedFile("c.txt", 8),
        ]
        with patch.object(config, "MAX_UPLOAD_BYTES", 10), patch.object(
            config, "MAX_TOTAL_UPLOAD_BYTES", 13
        ):
            accepted, msgs = check_upload_limits(files)
        self.assertEqual([f.name for f in accepted], ["a.txt", "b.txt"])
        self.assertEqual(len(msgs), 1)
        self.assertIn("c.txt", msgs[0])

    def test_all_within_limits_pass(self) -> None:
        from app.views.uploads import check_upload_limits
        import app.config as config

        files = [_UploadedFile("a.txt", 3), _UploadedFile("b.txt", 4)]
        with patch.object(config, "MAX_UPLOAD_BYTES", 10), patch.object(
            config, "MAX_TOTAL_UPLOAD_BYTES", 13
        ):
            accepted, msgs = check_upload_limits(files)
        self.assertEqual(len(accepted), 2)
        self.assertEqual(msgs, [])


class TestExtractUploadsCaching(unittest.TestCase):
    def test_cache_hit_skips_extraction(self) -> None:
        import app.views.uploads as uploads

        st_mock, session = _fake_streamlit()
        cached_doc = MagicMock(filename="a.txt")
        session["slot:a.txt:10"] = cached_doc
        uploaded = _UploadedFile("a.txt", 10)

        with _patch_st(uploads, st_mock), patch.object(
            uploads, "extract_uploaded_documents"
        ) as extract_mock:
            docs = uploads.extract_uploads([uploaded], "slot")
        self.assertEqual(docs, [cached_doc])
        extract_mock.assert_not_called()

    def test_new_file_extracted_and_cached(self) -> None:
        import app.views.uploads as uploads

        st_mock, session = _fake_streamlit()
        doc = MagicMock(filename="b.txt")
        uploaded = _UploadedFile("b.txt", 20)

        with _patch_st(uploads, st_mock), patch.object(
            uploads, "extract_uploaded_documents", return_value=([doc], [])
        ):
            docs = uploads.extract_uploads([uploaded], "slot")
        self.assertEqual(docs, [doc])
        self.assertEqual(session["slot:b.txt:20"], doc)

    def test_failed_extraction_warns_and_summarizes(self) -> None:
        import app.views.uploads as uploads

        st_mock, _ = _fake_streamlit()
        uploaded = _UploadedFile("bad.pdf", 5)

        with _patch_st(uploads, st_mock), patch.object(
            uploads, "extract_uploaded_documents", return_value=([], ["✖️ bad.pdf: unreadable"])
        ):
            docs = uploads.extract_uploads([uploaded], "slot")
        self.assertEqual(docs, [])
        st_mock.warning.assert_called_once_with("✖️ bad.pdf: unreadable")
        caption_text = str(st_mock.caption.call_args[0][0])
        self.assertIn("0 of 1", caption_text)
        self.assertIn("1 skipped", caption_text)


# ----------------------------------------------------------------- records.py
class TestUploadedVaGovExportLabelling(unittest.TestCase):
    """A VA.gov export downloaded by the user and uploaded is a real record set:
    the uploader should label it as the VA.gov source rather than an anonymous file."""

    _VA_GOV_TEXT = (
        "Download your medical records\nva.gov | My HealtheVet\n"
        "Facility: VA Medical Center\nProvider: Smith, John MD\n"
    )
    _PRIVATE_TEXT = "Valley Regional Clinic\nProvider: Dr. Jane Doe\nMedications follow\n"

    def _run_uploader(self, filename: str, text: str):
        import app.views.records as records
        from app.documents import DocumentPage, ExtractedDocument

        doc = ExtractedDocument(
            filename=filename,
            pages=[DocumentPage(filename=filename, page=1, text=text)],
        )
        st_mock, session = _fake_streamlit()
        uploaded = _UploadedFile(filename, 2048)
        st_mock.file_uploader.return_value = [uploaded]
        with _patch_st(records, st_mock), patch.object(
            records, "check_upload_limits", return_value=([uploaded], [])
        ), patch.object(records, "extract_uploads", return_value=[doc]):
            records.records_uploader("eval")
        return st_mock, session, doc

    def test_va_gov_export_is_recorded_as_va_gov_source(self) -> None:
        st_mock, session, doc = self._run_uploader("va_records.pdf", self._VA_GOV_TEXT)
        store = session["source_records_eval"]
        self.assertEqual(store["VA.gov"], [doc])
        self.assertEqual(store["Upload"], [doc])
        captions = [str(call.args[0]) for call in st_mock.caption.call_args_list]
        self.assertTrue(
            any("Detected a VA.gov medical-records export" in c for c in captions),
            msg=captions,
        )

    def test_private_records_are_left_as_a_plain_upload(self) -> None:
        st_mock, session, doc = self._run_uploader("clinic.pdf", self._PRIVATE_TEXT)
        store = session["source_records_eval"]
        self.assertEqual(store["Upload"], [doc])
        self.assertNotIn("VA.gov", store)
        captions = [str(call.args[0]) for call in st_mock.caption.call_args_list]
        self.assertFalse(any("Detected a VA.gov" in c for c in captions), msg=captions)


class TestIsLocalRun(unittest.TestCase):
    def test_env_opt_in_forces_true(self) -> None:
        from app.views.records import is_local_run

        st_mock, _ = _fake_streamlit()
        with _patch_st(__import__("app.views.records", fromlist=["st"]), st_mock), patch.dict(
            "os.environ", {"VA_LSE_ALLOW_LOCAL_PATHS": "1"}
        ):
            self.assertTrue(is_local_run())

    def test_env_opt_out_falls_through_to_context(self) -> None:
        import app.views.records as records

        st_mock, _ = _fake_streamlit()
        ctx = MagicMock()
        ctx.headers = {"Host": "example.com:8501"}
        ctx.url = "https://example.com"
        st_mock.context = ctx
        with _patch_st(records, st_mock), patch.dict(
            "os.environ", {"VA_LSE_ALLOW_LOCAL_PATHS": ""}
        ):
            self.assertFalse(records.is_local_run())


class TestRememberSourceRecords(unittest.TestCase):
    def test_stores_docs_per_source_label(self) -> None:
        import app.views.records as records

        st_mock, session = _fake_streamlit()
        docs = [MagicMock()]
        with _patch_st(records, st_mock):
            records.remember_source_records("eval", "VA.gov", docs)
            records.remember_source_records("eval", "Upload", [MagicMock(), MagicMock()])
        store = session["source_records_eval"]
        self.assertEqual(store["VA.gov"], docs)
        self.assertEqual(len(store["Upload"]), 2)

    def test_empty_docs_do_not_overwrite(self) -> None:
        import app.views.records as records

        st_mock, session = _fake_streamlit()
        docs = [MagicMock()]
        with _patch_st(records, st_mock):
            records.remember_source_records("eval", "VA.gov", docs)
            records.remember_source_records("eval", "VA.gov", [])
        self.assertEqual(session["source_records_eval"]["VA.gov"], docs)


# ------------------------------------------------------------------ usage.py
class TestEffectiveCreditRates(unittest.TestCase):
    def test_explicit_env_rate_not_overwritten_by_watchdog(self) -> None:
        """Regression lock: the watchdog fills only the missing model's slot."""
        import app.config as config
        import app.views.usage as usage_view

        history = usage_view.watchdog.UsageHistory()
        usage_view.watchdog.record_run(history, prompt_tokens=1000, completion_tokens=0, calls=1)
        usage_view.watchdog.record_calibration(history, credits=0.0, ts=1.0)
        usage_view.watchdog.record_run(history, prompt_tokens=1000, completion_tokens=0, calls=1)
        usage_view.watchdog.record_calibration(history, credits=0.1, ts=2.0)  # 100 credits/1M

        st_mock, _ = _fake_streamlit()
        with _patch_st(usage_view, st_mock), patch.object(
            config, "CREDITS_PER_1M_MAIN", 800.0
        ), patch.object(config, "CREDITS_PER_1M_FAST", None), patch.object(
            usage_view.watchdog, "load_history", return_value=history
        ):
            rates, label = usage_view.effective_credit_rates()

        self.assertEqual(rates[config.DEFAULT_MODEL_MAIN], 800.0)
        self.assertAlmostEqual(rates[config.DEFAULT_MODEL_FAST], 100.0, delta=1e-6)
        self.assertIn("watchdog", label)

    def test_explicit_env_rates_win_immediately(self) -> None:
        import app.config as config
        import app.views.usage as usage_view

        st_mock, _ = _fake_streamlit()
        with _patch_st(usage_view, st_mock), patch.object(
            config, "CREDITS_PER_1M_MAIN", 800.0
        ), patch.object(config, "CREDITS_PER_1M_FAST", 200.0):
            rates, label = usage_view.effective_credit_rates()
        self.assertEqual(label, "configured in .env")
        self.assertEqual(rates[config.DEFAULT_MODEL_MAIN], 800.0)

    def test_record_watchdog_run_appends_totals(self) -> None:
        import app.views.usage as usage_view

        history = usage_view.watchdog.UsageHistory()
        tracker = MagicMock()
        totals = MagicMock(calls=2, prompt_tokens=100, completion_tokens=50)
        tracker.totals.return_value = totals
        tracker.per_role_tokens.return_value = {"main": 100, "fast": 50}

        st_mock, _ = _fake_streamlit()
        with _patch_st(usage_view, st_mock), patch.object(
            usage_view, "load_usage_history", return_value=history
        ), patch.object(usage_view, "save_usage_history") as save_mock:
            usage_view.record_watchdog_run(tracker)

        save_mock.assert_called_once()
        self.assertEqual(len(history.runs), 1)
        self.assertEqual(history.runs[0].prompt_tokens, 100)
        self.assertEqual(history.runs[0].completion_tokens, 50)

    def test_record_watchdog_run_skips_empty_usage(self) -> None:
        import app.views.usage as usage_view

        tracker = MagicMock()
        tracker.totals.return_value = MagicMock(calls=0)
        with patch.object(usage_view, "load_usage_history") as load_mock:
            usage_view.record_watchdog_run(tracker)
        load_mock.assert_not_called()


class TestRenderUsageSummary(unittest.TestCase):
    def test_no_usage_renders_nothing(self) -> None:
        import app.views.usage as usage_view

        st_mock, _ = _fake_streamlit()
        with _patch_st(usage_view, st_mock):
            usage_view.render_usage_summary(None)
            usage_view.render_usage_summary(UsageTracker())
        st_mock.expander.assert_not_called()

    def test_usage_renders_expander_with_totals(self) -> None:
        import app.views.usage as usage_view

        st_mock, _ = _fake_streamlit()
        expander_ctx = MagicMock()
        st_mock.expander.return_value = expander_ctx
        expander_ctx.__enter__ = MagicMock(return_value=None)
        expander_ctx.__exit__ = MagicMock(return_value=False)

        tracker = UsageTracker()
        tracker.record(model="m", phase="p", system="s", user="u", content="c")
        with _patch_st(usage_view, st_mock):
            usage_view.render_usage_summary(tracker)
        st_mock.expander.assert_called_once()
        self.assertIn("Estimated API usage", st_mock.expander.call_args[0][0])
        caption_text = " ".join(str(c.args[0]) for c in st_mock.caption.call_args_list)
        self.assertIn("Total:", caption_text)


# ------------------------------------------------------------------ get_llm
class TestGetLLM(unittest.TestCase):
    def test_returns_none_with_error_when_not_configured(self) -> None:
        import app.views.shared as shared

        st_mock, session = _fake_streamlit()
        settings = MagicMock(configured=False, api_key="")
        session["settings"] = settings
        with _patch_st(shared, st_mock):
            self.assertIsNone(shared.get_llm())
        msg = str(st_mock.error.call_args[0][0])
        self.assertIn("API key", msg)

    def test_returns_client_when_configured(self) -> None:
        import app.views.shared as shared

        st_mock, session = _fake_streamlit()
        settings = MagicMock(configured=True, api_key="key")
        session["settings"] = settings
        fake_client = MagicMock()
        with _patch_st(shared, st_mock), patch.object(
            shared, "LLMClient", return_value=fake_client
        ):
            self.assertIs(shared.get_llm(), fake_client)

    def test_llm_error_surfaces_message(self) -> None:
        import app.views.shared as shared
        from app.llm import LLMError

        st_mock, session = _fake_streamlit()
        settings = MagicMock(configured=True, api_key="key")
        session["settings"] = settings
        with _patch_st(shared, st_mock), patch.object(
            shared, "LLMClient", side_effect=LLMError("bad endpoint")
        ):
            self.assertIsNone(shared.get_llm())
        st_mock.error.assert_called_once_with("bad endpoint")


# ------------------------------------------------- empty-analysis guard
class TestEmptyAnalysisResults(unittest.TestCase):
    """An all-empty run must never render as a clean "Not scored" report."""

    def _st(self) -> MagicMock:
        st_mock, session = _fake_streamlit()
        # _render_evaluation_results lays out four metric columns.
        st_mock.columns.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        session["eval_request_id"] = "req_84ab42a65e24"
        return st_mock, session

    def test_is_empty_analysis_detects_blank_result(self) -> None:
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult

        self.assertTrue(evaluate_view._is_empty_analysis(EvaluationResult()))

    def test_is_empty_analysis_false_with_claims_or_scores(self) -> None:
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult

        with_claims = EvaluationResult(claims=[{"id": 1, "text": "Knee pain."}])
        with_scores = EvaluationResult(scores={"factual_accuracy": 5.0})
        self.assertFalse(evaluate_view._is_empty_analysis(with_claims))
        self.assertFalse(evaluate_view._is_empty_analysis(with_scores))

    def test_blank_result_renders_error_naming_the_reference(self) -> None:
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult

        st_mock, _ = self._st()
        with _patch_st(evaluate_view, st_mock):
            evaluate_view._render_evaluation_results(EvaluationResult())

        st_mock.error.assert_called_once()
        message = str(st_mock.error.call_args[0][0])
        self.assertIn("no usable analysis", message)
        self.assertIn("req_84ab42a65e24", message)
        # Which run the panel belongs to is stated, so a cached re-render cannot
        # pass for a fresh run.
        caption_text = " ".join(str(c.args[0]) for c in st_mock.caption.call_args_list)
        self.assertIn("req_84ab42a65e24", caption_text)
        self.assertIn("last completed run", caption_text)

    def test_populated_result_renders_without_the_empty_error(self) -> None:
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult

        st_mock, _ = self._st()
        result = EvaluationResult(
            claims=[{"id": 1, "text": "Knee pain since 2014."}],
            verifications=[{"id": 1, "verdict": "SUPPORTED"}],
            scores={"factual_accuracy": 7.0},
            executive_summary="Strong statement.",
        )
        with _patch_st(evaluate_view, st_mock):
            evaluate_view._render_evaluation_results(result)

        st_mock.error.assert_not_called()


# ------------------------------------------------- run bookkeeping
class TestEvaluateRunBookkeeping(unittest.TestCase):
    """Every run must leave a traceable completion — including interruptions.

    ``RerunException``/``StopException`` derive from ``BaseException``, so the
    ``except Exception`` handler must come first: if the control-flow clause is
    ordered ahead of it, ordinary failures (LLM errors, timeouts, bugs) get
    misreported as interruptions and re-raised instead of surfaced to the user.
    """

    def _drive(self, *, exc: BaseException | None = None, result=None):
        """Run _run_evaluation_flow with stubbed plumbing; return logged events."""
        import app.views.evaluate_view as evaluate_view
        from collections import namedtuple

        totals = namedtuple("Totals", "calls prompt_tokens completion_tokens")(0, 0, 0)
        st_mock, _session = _fake_streamlit()
        events: list[tuple[str, str, dict]] = []
        llm = MagicMock()
        llm.usage.totals.return_value = totals

        pipeline = (
            patch.object(evaluate_view, "run_with_timeout", side_effect=exc)
            if exc is not None
            else patch.object(evaluate_view, "run_with_timeout", return_value=result)
        )
        with _patch_st(evaluate_view, st_mock), patch.object(
            evaluate_view, "new_run_request_id", return_value="req_84ab42a65e24"
        ), patch.object(evaluate_view, "get_llm", return_value=llm), patch.object(
            evaluate_view, "check_shutdown_gate", return_value=True
        ), patch.object(evaluate_view, "enter_run", return_value=True), patch.object(
            evaluate_view, "exit_run"
        ), patch.object(
            evaluate_view,
            "progress_widgets",
            return_value=(MagicMock(), lambda *a, **k: None),
        ), patch.object(evaluate_view, "check_memory_before_run"), pipeline, patch.object(
            evaluate_view, "get_profiler", return_value=None
        ), patch.object(
            evaluate_view, "audit_record_meta", return_value=(["Upload"], 1, 1)
        ), patch.object(
            evaluate_view, "audit_condition_for_slot", return_value=""
        ), patch.object(
            evaluate_view,
            "run_log_event",
            side_effect=lambda action, status, **kw: events.append((action, status, kw)),
        ), patch.object(evaluate_view, "audit_log"):
            try:
                evaluate_view._run_evaluation_flow("I watched the veteran limp.", [MagicMock()])
            except BaseException as raised:  # noqa: BLE001 - control flow is re-raised
                return events, raised
        return events, None

    def test_runtime_failure_is_logged_as_error(self) -> None:
        events, _ = self._drive(exc=RuntimeError("boom"))
        statuses = [status for _, status, _ in events]
        self.assertIn("error", statuses)
        self.assertNotIn("interrupted", statuses)

    def test_control_flow_exception_is_logged_interrupted_and_reraised(self) -> None:
        class _ScriptTornDown(BaseException):  # mirrors Streamlit's control flow
            pass

        torn_down = _ScriptTornDown("rerun requested")
        events, raised = self._drive(exc=torn_down)
        self.assertIs(raised, torn_down)
        self.assertIn("interrupted", [status for _, status, _ in events])

    def test_empty_analysis_logs_empty_event(self) -> None:
        from app.evaluate import EvaluationResult

        events, raised = self._drive(result=EvaluationResult())
        self.assertIsNone(raised)
        statuses = [status for _, status, _ in events]
        self.assertIn("empty", statuses)
        self.assertNotIn("interrupted", statuses)

    def test_populated_analysis_logs_ok(self) -> None:
        from app.evaluate import EvaluationResult

        events, _ = self._drive(
            result=EvaluationResult(claims=[{"id": 1, "text": "Knee pain."}], scores={"a": 5.0})
        )
        statuses = [status for _, status, _ in events]
        self.assertIn("ok", statuses)
        self.assertNotIn("empty", statuses)


# ------------------------------------------------- sidebar settings guards
class TestSidebarSettingsGuards(unittest.TestCase):
    """The sidebar's key/URL/model fields are applied inconsistently by design —
    the guards make that visible instead of leaving a silent mismatch."""

    def _settings(self, **over):
        base = dict(
            api_key="test-workspace-key",
            base_url="https://ws-example.us-east-1.maas.aliyuncs.com",
            model_main="qwen3.7-max",
            model_fast="qwen3.7-flash",
        )
        base.update(over)
        return types.SimpleNamespace(**base)

    def test_pending_warning_lists_unapplied_fields(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, session = _fake_streamlit()
        session["base_url_input"] = "https://token-plan.example/v1"
        session["model_main_input"] = "qwen3.7-max"
        with _patch_st(sidebar, st_mock):
            sidebar._pending_settings_warning(self._settings())
        st_mock.warning.assert_called_once()
        msg = str(st_mock.warning.call_args[0][0])
        self.assertIn("base URL", msg)
        self.assertIn("Apply settings", msg)

    def test_pending_warning_silent_once_applied(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, session = _fake_streamlit()
        session["base_url_input"] = "https://ws-example.us-east-1.maas.aliyuncs.com"
        session["model_main_input"] = "qwen3.7-max"
        with _patch_st(sidebar, st_mock):
            sidebar._pending_settings_warning(self._settings())
        st_mock.warning.assert_not_called()

    def test_connection_success_names_models(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, session = _fake_streamlit()
        session["base_url_input"] = "https://ws-example.us-east-1.maas.aliyuncs.com"
        session["api_key_input"] = "test-workspace-key"
        with _patch_st(sidebar, st_mock), patch.object(
            sidebar, "check_model_availability", return_value={"qwen3.7-max", "qwen3.7-flash"}
        ):
            sidebar._test_connection_report(self._settings())
        st_mock.success.assert_called_once()
        self.assertIn("reachable", str(st_mock.success.call_args[0][0]))

    def test_connection_failure_explains_key_url_pairing(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, session = _fake_streamlit()
        session["base_url_input"] = "https://token-plan.example/v1"
        session["api_key_input"] = "test-workspace-key"
        with _patch_st(sidebar, st_mock), patch.object(
            sidebar, "check_model_availability", return_value=None
        ):
            sidebar._test_connection_report(self._settings())
        st_mock.error.assert_called_once()
        self.assertIn("same provider account", str(st_mock.error.call_args[0][0]))

    def test_connection_flags_models_the_endpoint_lacks(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, session = _fake_streamlit()
        session["api_key_input"] = "test-workspace-key"
        with _patch_st(sidebar, st_mock), patch.object(
            sidebar, "check_model_availability", return_value={"some-other-model"}
        ):
            sidebar._test_connection_report(self._settings())
        st_mock.warning.assert_called_once()
        msg = str(st_mock.warning.call_args[0][0])
        self.assertIn("qwen3.7-max", msg)
        self.assertIn("qwen3.7-flash", msg)

    def test_connection_requires_a_key(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        with _patch_st(sidebar, st_mock):
            sidebar._test_connection_report(self._settings(api_key=""))
        st_mock.error.assert_called_once_with("Enter an API key first, then test the connection.")

    def test_secrets_origin_is_captioned(self) -> None:
        """A hosted run pre-fills the fields from st.secrets — say so."""
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        settings = self._settings(
            from_secrets=frozenset({"OPENAI_API_KEY", "OPENAI_BASE_URL"})
        )
        with _patch_st(sidebar, st_mock):
            sidebar._secrets_source_note(settings)
        st_mock.caption.assert_called_once()
        msg = str(st_mock.caption.call_args[0][0])
        self.assertIn("API key", msg)
        self.assertIn("base URL", msg)
        self.assertIn("secrets", msg)

    def test_secrets_note_silent_without_secrets(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        with _patch_st(sidebar, st_mock):
            sidebar._secrets_source_note(self._settings())
        st_mock.caption.assert_not_called()


class _FakeBlobStore:
    """Minimal blob-store stand-in for the sidebar panel (name + is_shared)."""

    def __init__(self, name: str = "filesystem", is_shared: bool = True) -> None:
        self.name = name
        self.is_shared = is_shared


def _backup_health(**overrides) -> dict:  # noqa: ANN003 - test payload builder
    """An ``audit_backup_health()`` payload, healthy unless overridden."""
    payload = {
        "configured": True,
        "destination": "s3",
        "off_pod": True,
        "interval_hours": 6.0,
        "local_retention_days": 7,
        "cloud_retention_days": 90,
        "uploaded_objects": 4,
        "uploaded_bytes": 4096,
        "runs": 5,
        "last_success_utc": "2026-09-16T12:00:00+00:00",
        "last_error": None,
        "pending_bytes": 0,
        "age_seconds": 600,
        "stale": False,
        "status": "ok",
    }
    payload.update(overrides)
    return payload


def _disk_status(**overrides) -> dict:  # noqa: ANN003 - test payload builder
    """A ``disk_status()`` payload, comfortably above the floor by default."""
    payload = {
        "checked": True,
        "path": "/app/logs",
        "free_bytes": 10 * 1024**3,
        "total_bytes": 40 * 1024**3,
        "used_percent": 75.0,
        "min_free_bytes": 256 * 1024**2,
        "below_floor": False,
    }
    payload.update(overrides)
    return payload


class TestJobQueuePanel(unittest.TestCase):
    """Sidebar ops panel: tells the truth about where runs execute.

    The panel renders on every sidebar paint, so these tests also pin the rule
    that rendering performs no queue I/O — the backlog is fetched only when the
    operator asks, because the Upstash tier's ``depth()`` is two HTTP requests.
    """

    class _Queue:
        def __init__(
            self,
            *,
            name: str = "redis",
            distributed: bool = True,
            depth: int = 0,
            reachable: bool = True,
            boom: bool = False,
        ) -> None:
            self.name = name
            self.is_distributed = distributed
            self._depth = depth
            self._reachable = reachable
            self._boom = boom
            self.probes = 0

        def depth(self) -> int:
            self.probes += 1
            if self._boom:
                raise RuntimeError("connection refused")
            return self._depth

        def ping(self) -> bool:
            if self._boom:
                raise RuntimeError("connection refused")
            return self._reachable

    @contextmanager
    def _panel(
        self,
        st_mock: MagicMock,
        *,
        enabled: bool = True,
        queue=None,  # noqa: ANN001 - _Queue or a raising side effect
        blob=None,  # noqa: ANN001 - _FakeBlobStore or a raising side effect
    ):
        import app.blob_store as blob_store
        import app.job_queue as job_queue
        import app.views.sidebar as sidebar

        queue = queue if queue is not None else self._Queue()
        blob = blob if blob is not None else _FakeBlobStore()
        backend_patch = (
            patch.object(job_queue, "get_job_backend", side_effect=queue)
            if isinstance(queue, BaseException)
            else patch.object(job_queue, "get_job_backend", return_value=queue)
        )
        blob_patch = (
            patch.object(blob_store, "get_blob_store", side_effect=blob)
            if isinstance(blob, BaseException)
            else patch.object(blob_store, "get_blob_store", return_value=blob)
        )
        with ExitStack() as stack:
            stack.enter_context(_patch_st(sidebar, st_mock))
            stack.enter_context(
                patch.object(sidebar.config, "JOB_QUEUE_ENABLED", enabled, create=True)
            )
            stack.enter_context(backend_patch)
            stack.enter_context(blob_patch)
            yield queue

    @staticmethod
    def _rows(st_mock: MagicMock) -> dict:
        rows = st_mock.dataframe.call_args[0][0]
        return {row["Setting"]: row["Value"] for row in rows}

    def test_disabled_explains_in_process_execution(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        with self._panel(st_mock, enabled=False) as queue:
            sidebar._job_queue_panel()
        self.assertGreaterEqual(st_mock.caption.call_count, 2)
        self.assertIn("worker pool", str(st_mock.caption.call_args_list[-1][0][0]))
        st_mock.dataframe.assert_not_called()

    def test_disabled_does_not_touch_the_queue(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        with self._panel(st_mock, enabled=False) as queue:
            sidebar._job_queue_panel()
        self.assertEqual(queue.probes, 0)
        st_mock.warning.assert_not_called()
        st_mock.error.assert_not_called()

    def test_enabled_names_backend_and_shared_blob_store(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        st_mock.button.return_value = False
        with self._panel(st_mock) as queue:
            sidebar._job_queue_panel()
        self.assertIn(queue.name, str(st_mock.caption.call_args_list[0][0][0]))
        rows = self._rows(st_mock)
        self.assertEqual(rows["Backend"], "redis")
        self.assertEqual(rows["Distributed"], "yes")
        self.assertEqual(rows["Job documents"], "filesystem (shared)")
        st_mock.warning.assert_not_called()

    def test_local_only_blob_store_is_not_called_shared(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        st_mock.button.return_value = False
        with self._panel(st_mock, blob=_FakeBlobStore(name="s3", is_shared=False)):
            sidebar._job_queue_panel()
        self.assertEqual(self._rows(st_mock)["Job documents"], "s3")

    def test_single_process_backend_warns_workers_cannot_claim(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        st_mock.button.return_value = False
        with self._panel(st_mock, queue=self._Queue(name="inprocess", distributed=False)):
            sidebar._job_queue_panel()
        st_mock.warning.assert_called_once()
        msg = str(st_mock.warning.call_args[0][0])
        self.assertIn("inprocess", msg)
        self.assertIn("VA_LSE_REDIS_URL", msg)
        self.assertEqual(self._rows(st_mock)["Distributed"], "no")

    def test_backend_failure_is_reported_not_raised(self) -> None:
        import app.views.sidebar as sidebar
        from app.job_queue import JobQueueUnavailable

        st_mock, _ = _fake_streamlit()
        with self._panel(st_mock, queue=JobQueueUnavailable("redis package missing")):
            sidebar._job_queue_panel()
        st_mock.error.assert_called_once()
        msg = str(st_mock.error.call_args[0][0])
        self.assertIn("Job queue unavailable", msg)
        self.assertIn("redis package missing", msg)
        st_mock.dataframe.assert_not_called()

    def test_blob_failure_degrades_the_row(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        st_mock.button.return_value = False
        with self._panel(st_mock, blob=RuntimeError("no boto3")):
            sidebar._job_queue_panel()
        self.assertEqual(self._rows(st_mock)["Job documents"], "unavailable")

    def test_render_does_not_probe_the_queue(self) -> None:
        """Every sidebar paint must stay offline; the probe is button-driven."""
        import app.views.sidebar as sidebar

        st_mock, session = _fake_streamlit()
        st_mock.button.return_value = False
        with self._panel(st_mock, queue=self._Queue(depth=7)) as queue:
            sidebar._job_queue_panel()
        self.assertEqual(queue.probes, 0)
        self.assertNotIn("job_queue_probe_at", session)
        st_mock.info.assert_not_called()

    def test_backlog_probe_reports_waiting_jobs(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, session = _fake_streamlit()
        st_mock.button.return_value = True
        with self._panel(st_mock, queue=self._Queue(depth=3)) as queue:
            sidebar._job_queue_panel()
        self.assertEqual(queue.probes, 1)
        self.assertEqual(session["job_queue_depth"], 3)
        self.assertTrue(session["job_queue_reachable"])
        st_mock.info.assert_called_once()
        self.assertIn("3 job(s)", str(st_mock.info.call_args[0][0]))

    def test_backlog_probe_reports_empty_queue(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        st_mock.button.return_value = True
        with self._panel(st_mock, queue=self._Queue(depth=0)):
            sidebar._job_queue_panel()
        st_mock.info.assert_not_called()
        self.assertIn("No jobs waiting", str(st_mock.caption.call_args_list[-2][0][0]))

    def test_backlog_probe_flags_unreachable_queue(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, session = _fake_streamlit()
        st_mock.button.return_value = True
        with self._panel(st_mock, queue=self._Queue(reachable=False)):
            sidebar._job_queue_panel()
        self.assertFalse(session["job_queue_reachable"])
        self.assertIn("unreachable", str(st_mock.error.call_args[0][0]))

    def test_backlog_probe_failure_is_reported(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, session = _fake_streamlit()
        st_mock.button.return_value = True
        with self._panel(st_mock, queue=self._Queue(boom=True)):
            sidebar._job_queue_panel()
        self.assertIsNone(session["job_queue_depth"])
        self.assertFalse(session["job_queue_reachable"])
        self.assertIn("Queue probe failed", str(st_mock.error.call_args[0][0]))
        self.assertIn("connection refused", str(st_mock.error.call_args[0][0]))


class TestAuditBackupPanel(unittest.TestCase):
    """Sidebar compliance panel: is the audit log backed up, and is it surviving?

    The panel renders on every sidebar paint, so these tests also pin the rule that
    it performs no network I/O — everything it shows comes from the state file the
    backup process writes plus one ``statvfs``.
    """

    @contextmanager
    def _panel(self, st_mock: MagicMock, *, health=None, disk=None):  # noqa: ANN001
        """Patch the two functions the panel reads; capture what it renders."""
        import app.audit_backup as audit_backup
        import app.views.sidebar as sidebar

        health = health if health is not None else _backup_health()
        disk = disk if disk is not None else _disk_status()
        health_patch = (
            patch.object(audit_backup, "audit_backup_health", side_effect=health)
            if isinstance(health, BaseException)
            else patch.object(audit_backup, "audit_backup_health", return_value=health)
        )
        disk_patch = (
            patch.object(audit_backup, "disk_status", side_effect=disk)
            if isinstance(disk, BaseException)
            else patch.object(audit_backup, "disk_status", return_value=disk)
        )
        with ExitStack() as stack:
            stack.enter_context(_patch_st(sidebar, st_mock))
            stack.enter_context(health_patch)
            stack.enter_context(disk_patch)
            yield

    @staticmethod
    def _rows(st_mock: MagicMock) -> dict:
        rows = st_mock.dataframe.call_args[0][0]
        return {row["Setting"]: row["Value"] for row in rows}

    def test_unconfigured_warns_the_record_lives_only_here(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        with self._panel(st_mock, health=_backup_health(configured=False, status="disabled",
                                                        destination="none")):
            sidebar._audit_backup_panel()
        msg = str(st_mock.warning.call_args[0][0])
        self.assertIn("only on this volume", msg)
        self.assertEqual(self._rows(st_mock)["Status"], "disabled")

    def test_same_volume_destination_is_called_out(self) -> None:
        """A 'backup' on the volume it should survive is the dangerous config."""
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        with self._panel(st_mock, health=_backup_health(off_pod=False)):
            sidebar._audit_backup_panel()
        self.assertIn("does **not** survive", str(st_mock.warning.call_args[0][0]))

    def test_stale_backup_warns(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        stale = _backup_health(status="stale", reason="last successful backup was 72h ago")
        with self._panel(st_mock, health=stale):
            sidebar._audit_backup_panel()
        self.assertIn("Backups have stopped", str(st_mock.warning.call_args[0][0]))

    def test_failed_pass_shows_the_error_not_a_stale_timestamp(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        failed = _backup_health(status="error", reason="AccessDenied")
        with self._panel(st_mock, health=failed):
            sidebar._audit_backup_panel()
        self.assertIn("AccessDenied", str(st_mock.error.call_args[0][0]))

    def test_never_ran_warns_about_a_dead_cronjob(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        with self._panel(st_mock, health=_backup_health(status="never_ran")):
            sidebar._audit_backup_panel()
        self.assertIn("CronJob or sidecar", str(st_mock.warning.call_args[0][0]))

    def test_healthy_backup_reports_age_and_pending_bytes(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        healthy = _backup_health(age_seconds=1800, pending_bytes=2048)
        with self._panel(st_mock, health=healthy):
            sidebar._audit_backup_panel()
        st_mock.success.assert_called_once()
        self.assertIn("0.5h ago", str(st_mock.success.call_args[0][0]))
        self.assertIn("2 KB would be lost", self._rows(st_mock)["Not yet shipped"])

    def test_disk_below_floor_is_an_error(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        low = _disk_status(free_bytes=1024, below_floor=True)
        with self._panel(st_mock, disk=low):
            sidebar._audit_backup_panel()
        self.assertIn("VA_LSE_DISK_MIN_FREE_BYTES", str(st_mock.error.call_args[0][0]))

    def test_retention_and_volume_are_visible(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        with self._panel(st_mock):
            sidebar._audit_backup_panel()
        rows = self._rows(st_mock)
        self.assertEqual(rows["Retention"], "7d local / 90d cloud")
        self.assertIn("GB", rows["Log volume free"])

    def test_a_failing_health_call_is_reported_not_raised(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        with self._panel(st_mock, health=RuntimeError("state file unreadable")):
            sidebar._audit_backup_panel()
        st_mock.error.assert_called_once()
        self.assertIn("state file unreadable", str(st_mock.error.call_args[0][0]))
        st_mock.dataframe.assert_not_called()

    def test_panel_does_not_build_a_destination_client(self) -> None:
        """Rendering must not construct an S3/GCS client (no I/O, no credentials)."""
        import app.audit_backup as audit_backup
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        with self._panel(st_mock):
            with patch.object(audit_backup, "build_destination") as builder:
                sidebar._audit_backup_panel()
        builder.assert_not_called()


def _failover_status(**overrides):  # noqa: ANN003 - test payload builder
    base = {
        "configured": False,
        "active": False,
        "after_seconds": 300.0,
        "primary_unhealthy_seconds": None,
    }
    base.update(overrides)
    return base


class TestLlmFailoverPanel(unittest.TestCase):
    """Sidebar failover panel: which endpoint is serving, and what happens next.

    The countdown is the reason this panel exists. A failing primary inside the
    grace period produces fast-fail errors *by design*, and without a visible wait
    a user cannot tell a two-minute provider hiccup from a broken key.
    """

    @contextmanager
    def _panel(self, st_mock: MagicMock, status):  # noqa: ANN001
        import app.llm as llm_module
        import app.views.sidebar as sidebar

        status_patch = (
            patch.object(llm_module, "failover_status", side_effect=status)
            if isinstance(status, BaseException)
            else patch.object(llm_module, "failover_status", return_value=status)
        )
        with ExitStack() as stack:
            stack.enter_context(_patch_st(sidebar, st_mock))
            stack.enter_context(status_patch)
            yield

    @staticmethod
    def _rows(st_mock: MagicMock) -> dict:
        rows = st_mock.dataframe.call_args[0][0]
        return {row["Setting"]: row["Value"] for row in rows}

    def test_a_single_endpoint_deployment_explains_how_to_arm_one(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        with self._panel(st_mock, _failover_status()):
            sidebar._llm_failover_panel()
        caption = " ".join(str(c[0][0]) for c in st_mock.caption.call_args_list)
        self.assertIn("single endpoint", caption)
        self.assertIn("OPENAI_BASE_URL_FALLBACK", caption)
        self.assertEqual(self._rows(st_mock)["Backup configured"], "no")
        st_mock.warning.assert_not_called()

    def test_an_armed_but_unused_backup_reports_healthy(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        status = _failover_status(configured=True, primary_unhealthy_seconds=None)
        with self._panel(st_mock, status):
            sidebar._llm_failover_panel()
        st_mock.success.assert_called_once()
        rows = self._rows(st_mock)
        self.assertEqual(rows["Backup configured"], "yes")
        self.assertEqual(rows["Serving from backup"], "no")
        self.assertEqual(rows["Primary unhealthy for"], "healthy")

    def test_a_zero_unhealthy_time_reads_as_zero_not_healthy(self) -> None:
        """0.0 means "the clock just started" — which is not the same as healthy."""
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        status = _failover_status(configured=True, primary_unhealthy_seconds=0.0)
        with self._panel(st_mock, status):
            sidebar._llm_failover_panel()
        self.assertEqual(self._rows(st_mock)["Primary unhealthy for"], "0s")

    def test_being_served_by_the_backup_is_stated_plainly(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        status = _failover_status(configured=True, active=True, primary_unhealthy_seconds=900.0)
        with self._panel(st_mock, status):
            sidebar._llm_failover_panel()
        message = str(st_mock.warning.call_args[0][0])
        self.assertIn("backup endpoint", message)
        self.assertIn("llm_endpoints", message, "the stamp is how it is recovered later")
        self.assertEqual(self._rows(st_mock)["Serving from backup"], "yes")

    def test_the_grace_window_shows_a_countdown(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        status = _failover_status(configured=True, primary_unhealthy_seconds=120.0)
        with self._panel(st_mock, status):
            sidebar._llm_failover_panel()
        message = str(st_mock.warning.call_args[0][0])
        self.assertIn("about 180s", message, "300s threshold - 120s unhealthy")
        self.assertIn("120s", message)

    def test_the_countdown_window_names_the_escape_hatch(self) -> None:
        """The one setting that removes the wait must be where the wait is felt."""
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        status = _failover_status(configured=True, primary_unhealthy_seconds=120.0)
        with self._panel(st_mock, status):
            sidebar._llm_failover_panel()
        caption = " ".join(str(c[0][0]) for c in st_mock.caption.call_args_list)
        self.assertIn("LLM_ENDPOINT_FALLBACK_TIMEOUT_SECONDS=0", caption)

    def test_the_countdown_does_not_claim_every_call_is_failing(self) -> None:
        """The breaker still lets a retry through each recovery window, and a
        successful retry ends this state — saying otherwise sends an operator
        hunting for a problem that does not exist."""
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        status = _failover_status(configured=True, primary_unhealthy_seconds=120.0)
        with self._panel(st_mock, status):
            sidebar._llm_failover_panel()
        message = str(st_mock.warning.call_args[0][0])
        self.assertIn("re-tried every", message)

    def test_the_breaker_state_is_reported(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        status = _failover_status(configured=True, primary_unhealthy_seconds=5.0)
        with self._panel(st_mock, status):
            sidebar._llm_failover_panel()
        self.assertIn(self._rows(st_mock)["Primary breaker"], {"CLOSED", "OPEN", "HALF_OPEN"})

    def test_an_unknown_threshold_does_not_render_a_bogus_countdown(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        status = _failover_status(configured=True, after_seconds=None, primary_unhealthy_seconds=None)
        with self._panel(st_mock, status):
            sidebar._llm_failover_panel()
        self.assertEqual(self._rows(st_mock)["Failover after"], "unknown")

    def test_a_failing_status_call_is_reported_not_raised(self) -> None:
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        with self._panel(st_mock, RuntimeError("breaker unavailable")):
            sidebar._llm_failover_panel()
        st_mock.error.assert_called_once()
        self.assertIn("breaker unavailable", str(st_mock.error.call_args[0][0]))
        st_mock.dataframe.assert_not_called()

    def test_rendering_never_constructs_an_llm_client(self) -> None:
        """The sidebar paints on every rerun, so this must stay off the network."""
        import app.llm as llm_module
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        with self._panel(st_mock, _failover_status(configured=True)):
            with patch.object(llm_module, "LLMClient") as client:
                sidebar._llm_failover_panel()
        client.assert_not_called()

    def test_the_panel_says_why_the_backup_is_not_a_sidebar_setting(self) -> None:
        """Otherwise "why can't I set this here?" has no answer in the UI."""
        import app.views.sidebar as sidebar

        st_mock, _ = _fake_streamlit()
        with self._panel(st_mock, _failover_status(configured=True)):
            sidebar._llm_failover_panel()
        caption = " ".join(str(c[0][0]) for c in st_mock.caption.call_args_list)
        self.assertIn("worker", caption)


if __name__ == "__main__":
    unittest.main()
