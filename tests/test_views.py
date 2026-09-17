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


class TestRenderRecordSearch(unittest.TestCase):
    """F2.S1 — search widget: query+filters -> ranked/highlighted results;
    export adds to ``citation_index`` in session state."""

    def _widget_mock(self, query_value: str, button_hits: dict) -> tuple:
        import app.views.records as records
        from app.documents import SearchResult

        st_mock, session = _fake_streamlit()
        col_from, col_to, col_provider = MagicMock(), MagicMock(), MagicMock()
        col_from.date_input.return_value = None
        col_to.date_input.return_value = None
        col_provider.text_input.return_value = ""

        def columns_side_effect(spec, *args, **kwargs):
            n = spec if isinstance(spec, int) else len(spec)
            if n == 3:
                return (col_from, col_to, col_provider)
            return tuple(MagicMock() for _ in range(n))

        st_mock.columns.side_effect = columns_side_effect
        st_mock.text_input.return_value = query_value

        def button_side_effect(label, key=None, **kwargs):
            return button_hits.get(key, False)

        st_mock.button.side_effect = button_side_effect
        return records, st_mock, session, SearchResult

    def test_search_click_stores_results_and_fires_telemetry(self) -> None:
        records, st_mock, session, SearchResult = self._widget_mock(
            "asthma", {"search_submit_eval": True}
        )
        fake_result = SearchResult(
            label="clinic.pdf p.1", excerpt="**asthma** flare", score=1.0,
            filename="clinic.pdf", page=1,
        )
        with _patch_st(records, st_mock), patch.object(
            records, "search_records", return_value=[fake_result]
        ) as mock_search, patch.object(
            records, "track_impression"
        ) as mock_impression, patch.object(
            records, "track_interaction"
        ) as mock_interaction:
            records.render_record_search("eval", [MagicMock()])

        mock_search.assert_called_once()
        self.assertEqual(session["search_results_eval"], [fake_result])
        mock_impression.assert_called_once()
        self.assertEqual(mock_impression.call_args.kwargs.get("entry_point"), "eval")
        # one interaction for the search action
        actions = [c.kwargs.get("action") for c in mock_interaction.call_args_list]
        self.assertIn("search", actions)

    def test_search_failure_fires_error_telemetry_and_shows_message(self) -> None:
        records, st_mock, session, _SearchResult = self._widget_mock(
            "asthma", {"search_submit_eval": True}
        )
        with _patch_st(records, st_mock), patch.object(
            records, "search_records", side_effect=RuntimeError("boom")
        ), patch.object(records, "track_feature_error") as mock_error, patch.object(
            records, "track_impression"
        ), patch.object(records, "track_interaction"):
            records.render_record_search("eval", [MagicMock()])

        mock_error.assert_called_once()
        self.assertEqual(mock_error.call_args.kwargs.get("stage"), "search")
        self.assertEqual(session["search_results_eval"], [])
        st_mock.error.assert_called()

    def test_export_excerpt_adds_to_citation_index(self) -> None:
        records, st_mock, session, SearchResult = self._widget_mock(
            "", {"search_export_eval_0": True}
        )
        result = SearchResult(
            label="clinic.pdf p.2", excerpt="**knee** pain noted", score=2.0,
            filename="clinic.pdf", page=2,
        )
        session["search_results_eval"] = [result]
        with _patch_st(records, st_mock), patch.object(
            records, "track_impression"
        ), patch.object(records, "track_interaction") as mock_interaction:
            records.render_record_search("eval", [MagicMock()])

        self.assertEqual(
            session["citation_index"],
            [{"excerpt": "**knee** pain noted", "source": "clinic.pdf p.2"}],
        )
        export_calls = [
            c for c in mock_interaction.call_args_list
            if c.kwargs.get("action") == "export_excerpt"
        ]
        self.assertEqual(len(export_calls), 1)
        self.assertEqual(export_calls[0].kwargs.get("source"), "clinic.pdf p.2")

    def test_export_excerpt_does_not_duplicate(self) -> None:
        records, st_mock, session, SearchResult = self._widget_mock(
            "", {"search_export_eval_0": True}
        )
        result = SearchResult(
            label="clinic.pdf p.2", excerpt="**knee** pain noted", score=2.0,
            filename="clinic.pdf", page=2,
        )
        session["search_results_eval"] = [result]
        session["citation_index"] = [{"excerpt": "**knee** pain noted", "source": "clinic.pdf p.2"}]
        with _patch_st(records, st_mock), patch.object(
            records, "track_impression"
        ), patch.object(records, "track_interaction"):
            records.render_record_search("eval", [MagicMock()])

        self.assertEqual(len(session["citation_index"]), 1)

    def test_no_documents_renders_nothing(self) -> None:
        records, st_mock, _session, _SearchResult = self._widget_mock("", {})
        with _patch_st(records, st_mock):
            records.render_record_search("eval", [])
        st_mock.expander.assert_not_called()


class TestRenderCitationIndexExport(unittest.TestCase):
    """F2.S2 — CSV/JSON download buttons for the accumulated citation index."""

    def _run(self, citations, clicked_fmt: str | None):
        import app.views.records as records

        st_mock, session = _fake_streamlit()
        session["citation_index"] = citations
        col_csv, col_json = MagicMock(), MagicMock()
        col_csv.download_button.return_value = clicked_fmt == "csv"
        col_json.download_button.return_value = clicked_fmt == "json"
        st_mock.columns.return_value = (col_csv, col_json)
        with _patch_st(records, st_mock), patch.object(
            records, "track_interaction"
        ) as mock_interaction, patch.object(records, "track_feature_error") as mock_error:
            records._render_citation_index_export("eval")
        return st_mock, session, col_csv, col_json, mock_interaction, mock_error

    def test_no_export_buttons_when_index_empty(self) -> None:
        st_mock, *_ = self._run([], None)
        st_mock.columns.assert_not_called()

    def test_renders_both_download_buttons_when_citations_present(self) -> None:
        citations = [{"excerpt": "e", "source": "s"}]
        st_mock, session, col_csv, col_json, mock_interaction, mock_error = self._run(
            citations, None
        )
        col_csv.download_button.assert_called_once()
        col_json.download_button.assert_called_once()
        mock_interaction.assert_not_called()
        mock_error.assert_not_called()

    def test_csv_click_fires_interaction_telemetry(self) -> None:
        citations = [{"excerpt": "e", "source": "s"}]
        _st_mock, _session, _col_csv, _col_json, mock_interaction, _mock_error = self._run(
            citations, "csv"
        )
        mock_interaction.assert_called_once()
        self.assertEqual(mock_interaction.call_args.kwargs.get("action"), "export_citation_index")
        self.assertEqual(mock_interaction.call_args.kwargs.get("format"), "csv")
        self.assertEqual(mock_interaction.call_args.kwargs.get("citation_count"), 1)


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

        # The results panel also renders the effectiveness-score band banner,
        # which uses st.error for the RED band, so scope this assertion to the
        # empty-analysis message instead of counting every st.error call.
        empty_analysis_errors = [
            str(call.args[0])
            for call in st_mock.error.call_args_list
            if "no usable analysis" in str(call.args[0])
        ]
        self.assertEqual(len(empty_analysis_errors), 1)
        message = empty_analysis_errors[0]
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

        # Only the effectiveness-score band banner may use st.error here; the
        # empty-analysis guard must stay silent for a populated result.
        error_messages = [str(call.args[0]) for call in st_mock.error.call_args_list]
        self.assertFalse([m for m in error_messages if "no usable analysis" in m])

    def test_blank_result_renders_the_score_band_banner_alongside_the_guard(self) -> None:
        """The empty-analysis guard and the RED score band both use st.error.

        Merge regression guard: the effectiveness-score panel renders its RED
        band through ``st.error`` as well, so on a blank run the panel must
        emit both messages — the guard exactly once and still naming the
        reference — instead of one crowding the other out.
        """
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult

        st_mock, _ = self._st()
        with _patch_st(evaluate_view, st_mock), patch.object(
            evaluate_view, "track_impression"
        ):
            evaluate_view._render_evaluation_results(EvaluationResult())

        error_messages = [str(call.args[0]) for call in st_mock.error.call_args_list]
        guards = [m for m in error_messages if "no usable analysis" in m]
        band_banners = [m for m in error_messages if "(RED band)" in m]
        self.assertEqual(len(guards), 1)
        self.assertIn("req_84ab42a65e24", guards[0])
        self.assertEqual(len(band_banners), 1)

    def test_panel_renders_the_timeline_and_the_score_panels_exactly_once_each(self) -> None:
        """Merge regression guard for the timeline + effectiveness-score panels.

        Both branches appended their render function at the *same* anchor in
        the results panel, and git matched the identical telemetry boilerplate
        inside them — splitting each function in half. A resolution that simply
        concatenated both sides of the conflict produced interleaved, duplicated
        panels; this pins one of each on a fully populated run.
        """
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult
        from app.medical_review import MedicalDigest, MedicalFact

        st_mock, _ = self._st()
        st_mock.radio.return_value = "All"
        st_mock.selectbox.return_value = None
        st_mock.plotly_chart.return_value = MagicMock(selection=MagicMock(points=[]))
        result = EvaluationResult(
            claims=[{"id": 1, "text": "Knee pain since 2014."}],
            verifications=[{"id": 1, "verdict": "SUPPORTED"}],
            scores={"factual_accuracy": 7.0},
            executive_summary="Strong statement.",
            effectiveness_score=90,
            recommendations=[
                {"title": "Add dates", "impact": "+4 points", "explanation": "x"}
            ],
            digest=MedicalDigest(
                facts=[
                    MedicalFact(
                        date="2020-01-01",
                        type="diagnosis",
                        description="Diagnosed with knee condition.",
                        source="records.pdf p.3",
                    )
                ],
                pages_reviewed=3,
            ),
        )
        with _patch_st(evaluate_view, st_mock), patch.object(
            evaluate_view, "track_impression"
        ), patch.object(evaluate_view, "track_interaction"):
            evaluate_view._render_evaluation_results(result)

        expander_labels = [str(call.args[0]) for call in st_mock.expander.call_args_list]
        timelines = [label for label in expander_labels if "Medical event timeline" in label]
        self.assertEqual(len(timelines), 1)

        metric_labels = [call.args[0] for call in st_mock.metric.call_args_list]
        self.assertEqual(metric_labels.count("Effectiveness score"), 1)
        band_banners = [
            str(call.args[0])
            for call in st_mock.success.call_args_list
            if "(GREEN band)" in str(call.args[0])
        ]
        self.assertEqual(len(band_banners), 1)

        error_messages = [str(call.args[0]) for call in st_mock.error.call_args_list]
        self.assertFalse([m for m in error_messages if "no usable analysis" in m])


# ------------------------------------------------- fact citation exporter (F7.S1)
class TestRubricAndPositiveSources(unittest.TestCase):
    def _digest(self, facts):
        from app.medical_review import MedicalDigest

        return MedicalDigest(facts=facts)

    def _fact(self, **overrides):
        from app.medical_review import MedicalFact

        base = dict(
            date="2020-01-01", type="diagnosis", description="d", source="records.pdf p.3",
            quote="q",
        )
        base.update(overrides)
        return MedicalFact(**base)

    def test_no_digest_returns_empty_sets(self) -> None:
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult

        cited, positive = evaluate_view._rubric_and_positive_sources(EvaluationResult())
        self.assertEqual(cited, set())
        self.assertEqual(positive, set())

    def test_matches_fact_source_substring_in_record_reference(self) -> None:
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult

        fact = self._fact(source="records.pdf p.3")
        result = EvaluationResult(
            digest=self._digest([fact]),
            verifications=[
                {"id": 1, "verdict": "SUPPORTED", "record_reference": "records.pdf p.3, 2020-01-01"}
            ],
        )
        cited, positive = evaluate_view._rubric_and_positive_sources(result)
        self.assertEqual(cited, {"records.pdf p.3"})
        self.assertEqual(positive, {"records.pdf p.3"})

    def test_contradicted_verdict_is_cited_but_not_positive(self) -> None:
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult

        fact = self._fact(source="records.pdf p.5")
        result = EvaluationResult(
            digest=self._digest([fact]),
            verifications=[
                {"id": 1, "verdict": "CONTRADICTED", "record_reference": "records.pdf p.5"}
            ],
        )
        cited, positive = evaluate_view._rubric_and_positive_sources(result)
        self.assertEqual(cited, {"records.pdf p.5"})
        self.assertEqual(positive, set())

    def test_later_supportive_match_marks_source_positive(self) -> None:
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult

        fact = self._fact(source="records.pdf p.7")
        result = EvaluationResult(
            digest=self._digest([fact]),
            verifications=[
                {"id": 1, "verdict": "CONTRADICTED", "record_reference": "records.pdf p.7"},
                {
                    "id": 2,
                    "verdict": "PARTIALLY SUPPORTED",
                    "record_reference": "records.pdf p.7, 2020-01-01",
                },
            ],
        )
        cited, positive = evaluate_view._rubric_and_positive_sources(result)
        self.assertEqual(cited, {"records.pdf p.7"})
        self.assertEqual(positive, {"records.pdf p.7"})

    def test_fact_not_referenced_anywhere_is_excluded(self) -> None:
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult

        fact = self._fact(source="unreferenced.pdf p.1")
        result = EvaluationResult(
            digest=self._digest([fact]),
            verifications=[{"id": 1, "verdict": "SUPPORTED", "record_reference": "other.pdf p.9"}],
        )
        cited, positive = evaluate_view._rubric_and_positive_sources(result)
        self.assertEqual(cited, set())
        self.assertEqual(positive, set())


class TestRenderFactExportSection(unittest.TestCase):
    def _digest(self):
        from app.medical_review import MedicalDigest, MedicalFact

        fact = MedicalFact(
            date="2020-01-01", type="diagnosis", description="d",
            source="records.pdf p.3", quote="q",
        )
        return MedicalDigest(facts=[fact], conditions=["PTSD"], pages_reviewed=10)

    def test_no_digest_renders_nothing(self) -> None:
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult

        st_mock, _ = _fake_streamlit()
        with _patch_st(evaluate_view, st_mock):
            evaluate_view._render_fact_export_section(EvaluationResult())
        st_mock.expander.assert_not_called()

    def test_impression_fires_once_per_session(self) -> None:
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult

        st_mock, session = _fake_streamlit()
        st_mock.checkbox.return_value = False
        st_mock.button.return_value = False
        result = EvaluationResult(digest=self._digest())
        with _patch_st(evaluate_view, st_mock), patch.object(
            evaluate_view, "track_impression"
        ) as mock_impression:
            evaluate_view._render_fact_export_section(result)
            evaluate_view._render_fact_export_section(result)
        mock_impression.assert_called_once()
        self.assertEqual(
            mock_impression.call_args.args[0], evaluate_view.EXPORT_FACTS_FEATURE_ID
        )

    def test_export_click_generates_all_three_formats_and_fires_goal(self) -> None:
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult

        st_mock, _session = _fake_streamlit()
        st_mock.checkbox.return_value = False
        st_mock.button.return_value = True
        st_mock.columns.return_value = (MagicMock(), MagicMock(), MagicMock())
        result = EvaluationResult(digest=self._digest())
        with _patch_st(evaluate_view, st_mock), patch.object(
            evaluate_view, "export_facts", wraps=evaluate_view.export_facts
        ) as mock_export, patch.object(evaluate_view, "track_goal") as mock_goal, patch.object(
            evaluate_view, "track_interaction"
        ), patch.object(evaluate_view, "track_impression"):
            evaluate_view._render_fact_export_section(result)

        formats = {c.args[1] for c in mock_export.call_args_list}
        self.assertEqual(formats, {"csv", "pdf", "md"})
        mock_goal.assert_called_once()
        self.assertEqual(mock_goal.call_args.args[0], evaluate_view.EXPORT_FACTS_FEATURE_ID)

    def test_download_click_fires_interaction_telemetry(self) -> None:
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult

        st_mock, session = _fake_streamlit()
        st_mock.checkbox.return_value = False
        st_mock.button.return_value = False
        session["export_facts_files"] = {"csv": b"data", "pdf": b"%PDF", "md": b"# md"}
        col_csv, col_pdf, col_md = MagicMock(), MagicMock(), MagicMock()
        col_csv.download_button.return_value = True
        col_pdf.download_button.return_value = False
        col_md.download_button.return_value = False
        st_mock.columns.return_value = (col_csv, col_pdf, col_md)
        result = EvaluationResult(digest=self._digest())
        with _patch_st(evaluate_view, st_mock), patch.object(
            evaluate_view, "track_interaction"
        ) as mock_interaction, patch.object(evaluate_view, "track_impression"):
            evaluate_view._render_fact_export_section(result)

        mock_interaction.assert_called_once()
        self.assertEqual(mock_interaction.call_args.kwargs.get("action"), "download")
        self.assertEqual(mock_interaction.call_args.kwargs.get("format"), "csv")

    def test_export_error_in_one_format_does_not_block_others(self) -> None:
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult

        st_mock, _session = _fake_streamlit()
        st_mock.checkbox.return_value = False
        st_mock.button.return_value = True
        st_mock.columns.return_value = (MagicMock(), MagicMock(), MagicMock())
        result = EvaluationResult(digest=self._digest())

        def _flaky(digest, fmt, **kwargs):
            if fmt == "pdf":
                raise RuntimeError("boom")
            return b"ok"

        with _patch_st(evaluate_view, st_mock), patch.object(
            evaluate_view, "export_facts", side_effect=_flaky
        ), patch.object(evaluate_view, "track_goal"), patch.object(
            evaluate_view, "track_interaction"
        ), patch.object(evaluate_view, "track_impression"):
            evaluate_view._render_fact_export_section(result)

        st_mock.error.assert_called_once()
        self.assertIn("pdf", str(st_mock.error.call_args[0][0]))


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


# ------------------------------------------- medical event timeline (F7.S2)
class TestRenderMedicalTimeline(unittest.TestCase):
    """Unit tests for the Medical Event Timeline Visualization subsection."""

    def _digest(self):
        from app.medical_review import MedicalDigest, MedicalFact

        facts = [
            MedicalFact(
                date="2020-01-01", type="diagnosis", description="Diagnosed with knee condition.",
                source="records.pdf p.3", quote="knee condition confirmed",
            ),
            MedicalFact(
                date="2021-06-01", type="treatment", description="Began physical therapy.",
                source="records.pdf p.8", quote="PT referral",
            ),
        ]
        return MedicalDigest(facts=facts, conditions=["Knee"], pages_reviewed=10)

    def _timeline_data(self, digest):
        from app.medical_review import build_timeline_data

        return build_timeline_data(digest)

    def _st(self):
        st_mock, session = _fake_streamlit()
        st_mock.radio.return_value = "All"
        st_mock.selectbox.return_value = None
        st_mock.plotly_chart.return_value = MagicMock(selection=MagicMock(points=[]))
        return st_mock, session

    def test_no_digest_renders_nothing(self) -> None:
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult

        st_mock, _ = self._st()
        with _patch_st(evaluate_view, st_mock):
            evaluate_view._render_medical_timeline(EvaluationResult(), request_reference="req_1")
        st_mock.expander.assert_not_called()

    def test_builds_and_caches_timeline_data_by_request_reference(self) -> None:
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult

        st_mock, session = self._st()
        result = EvaluationResult(digest=self._digest())
        with _patch_st(evaluate_view, st_mock), patch.object(
            evaluate_view, "get_llm", return_value=None
        ), patch.object(
            evaluate_view, "build_timeline_data", wraps=evaluate_view.build_timeline_data
        ) as mock_build:
            evaluate_view._render_medical_timeline(result, request_reference="req_1")
            evaluate_view._render_medical_timeline(result, request_reference="req_1")
        mock_build.assert_called_once()
        self.assertEqual(session["timeline_request_id"], "req_1")

    def test_rebuilds_when_request_reference_changes(self) -> None:
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult

        st_mock, _session = self._st()
        result = EvaluationResult(digest=self._digest())
        with _patch_st(evaluate_view, st_mock), patch.object(
            evaluate_view, "get_llm", return_value=None
        ), patch.object(
            evaluate_view, "build_timeline_data", wraps=evaluate_view.build_timeline_data
        ) as mock_build:
            evaluate_view._render_medical_timeline(result, request_reference="req_1")
            evaluate_view._render_medical_timeline(result, request_reference="req_2")
        self.assertEqual(mock_build.call_count, 2)

    def test_impression_fires_once_per_request_reference(self) -> None:
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult

        st_mock, _session = self._st()
        result = EvaluationResult(digest=self._digest())
        with _patch_st(evaluate_view, st_mock), patch.object(
            evaluate_view, "get_llm", return_value=None
        ), patch.object(evaluate_view, "track_impression") as mock_impression:
            evaluate_view._render_medical_timeline(result, request_reference="req_1")
            evaluate_view._render_medical_timeline(result, request_reference="req_1")
        mock_impression.assert_called_once()
        self.assertEqual(mock_impression.call_args.args[0], evaluate_view.TIMELINE_FEATURE_ID)
        self.assertEqual(mock_impression.call_args.kwargs.get("timeline_event_count"), 2)

    def test_diagnostic_filter_narrows_events_and_tracks_interaction(self) -> None:
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult

        st_mock, _session = self._st()
        st_mock.radio.return_value = "Diagnostic only"
        result = EvaluationResult(digest=self._digest())
        with _patch_st(evaluate_view, st_mock), patch.object(
            evaluate_view, "get_llm", return_value=None
        ), patch.object(evaluate_view, "track_interaction") as mock_interaction, patch.object(
            evaluate_view, "track_impression"
        ):
            evaluate_view._render_medical_timeline(result, request_reference="req_1")

        mock_interaction.assert_any_call(
            evaluate_view.TIMELINE_FEATURE_ID, action="filter", filter="diagnostic"
        )
        # Only the diagnostic-typed event should reach the chart builder.
        chart_args = st_mock.plotly_chart.call_args
        self.assertIsNotNone(chart_args)
        figure = chart_args.args[0]
        self.assertEqual(len(figure.data), 1)
        self.assertEqual(figure.data[0].name, "Diagnostic")

    def test_event_click_via_selectbox_shows_details_and_tracks_interaction(self) -> None:
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult

        st_mock, _session = self._st()
        st_mock.selectbox.return_value = 0
        result = EvaluationResult(digest=self._digest())
        with _patch_st(evaluate_view, st_mock), patch.object(
            evaluate_view, "get_llm", return_value=None
        ), patch.object(evaluate_view, "track_interaction") as mock_interaction, patch.object(
            evaluate_view, "track_impression"
        ):
            evaluate_view._render_medical_timeline(result, request_reference="req_1")

        mock_interaction.assert_any_call(
            evaluate_view.TIMELINE_FEATURE_ID, action="event_click", filter="all"
        )
        st_mock.markdown.assert_any_call("**2020-01-01 — diagnosis**")
        st_mock.write.assert_any_call("Diagnosed with knee condition.")

    def test_chart_click_selection_shows_details(self) -> None:
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult

        st_mock, _session = self._st()
        st_mock.plotly_chart.return_value = MagicMock(
            selection=MagicMock(points=[{"customdata": [1]}])
        )
        result = EvaluationResult(digest=self._digest())
        with _patch_st(evaluate_view, st_mock), patch.object(
            evaluate_view, "get_llm", return_value=None
        ), patch.object(evaluate_view, "track_interaction") as mock_interaction, patch.object(
            evaluate_view, "track_impression"
        ):
            evaluate_view._render_medical_timeline(result, request_reference="req_1")

        mock_interaction.assert_any_call(
            evaluate_view.TIMELINE_FEATURE_ID, action="event_click", filter="all"
        )
        st_mock.markdown.assert_any_call("**2021-06-01 — treatment**")

    def test_no_events_renders_nothing(self) -> None:
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult
        from app.medical_review import MedicalDigest

        st_mock, _session = self._st()
        result = EvaluationResult(digest=MedicalDigest(facts=[]))
        with _patch_st(evaluate_view, st_mock), patch.object(
            evaluate_view, "get_llm", return_value=None
        ):
            evaluate_view._render_medical_timeline(result, request_reference="req_1")
        st_mock.expander.assert_not_called()

    def test_chart_build_failure_costs_only_the_chart_not_the_panel(self) -> None:
        """A broken chart library must not take the results panel down with it.

        The figure build used to sit outside the render guard, so an
        ImportError/AttributeError from a missing or broken plotly escaped
        ``_render_medical_timeline`` and killed the whole Evaluate results
        panel — the opposite of what this function promises.
        """
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult

        st_mock, _session = self._st()
        result = EvaluationResult(digest=self._digest())
        with _patch_st(evaluate_view, st_mock), patch.object(
            evaluate_view, "get_llm", return_value=None
        ), patch.object(evaluate_view, "go", None), patch.object(
            evaluate_view, "track_impression"
        ), patch.object(evaluate_view, "track_feature_error") as mock_error:
            evaluate_view._render_medical_timeline(result, request_reference="req_1")

        st_mock.plotly_chart.assert_not_called()
        # The event list still renders, so the facts stay reachable without the chart.
        st_mock.selectbox.assert_called_once()
        info_messages = [str(call.args[0]) for call in st_mock.info.call_args_list]
        self.assertTrue(
            any("timeline chart is unavailable" in message for message in info_messages)
        )
        mock_error.assert_called_once()
        self.assertEqual(mock_error.call_args.args[0], evaluate_view.TIMELINE_FEATURE_ID)
        self.assertEqual(mock_error.call_args.kwargs.get("phase"), "chart_render")

    def test_figure_build_exception_is_tracked_not_raised(self) -> None:
        import app.views.evaluate_view as evaluate_view
        from app.evaluate import EvaluationResult

        st_mock, _session = self._st()
        result = EvaluationResult(digest=self._digest())
        with _patch_st(evaluate_view, st_mock), patch.object(
            evaluate_view, "get_llm", return_value=None
        ), patch.object(
            evaluate_view, "_build_timeline_figure", side_effect=RuntimeError("no go")
        ), patch.object(evaluate_view, "track_impression"), patch.object(
            evaluate_view, "track_feature_error"
        ) as mock_error:
            evaluate_view._render_medical_timeline(result, request_reference="req_1")

        mock_error.assert_called_once()
        self.assertEqual(mock_error.call_args.kwargs.get("phase"), "chart_render")


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


# ------------------------------------------------ effectiveness score (F4.S2)
class TestRenderEffectivenessScore(unittest.TestCase):
    def _result(self, score: int = 80, recommendations=None, claims=None):
        from app.evaluate import EvaluationResult

        return EvaluationResult(
            effectiveness_score=score,
            recommendations=recommendations
            if recommendations is not None
            else [
                {"title": "Add supporting evidence", "impact": "+8 points", "explanation": "x", "claim_id": 1},
                {"title": "Clarify frequency", "impact": "+5 points", "explanation": "y", "claim_id": None},
                {"title": "Add functional impact", "impact": "+3 points", "explanation": "z", "claim_id": None},
            ],
            claims=claims if claims is not None else [{"id": 1, "text": "Injured knee lifting."}],
        )

    def test_green_band_uses_success_banner(self) -> None:
        import app.views.evaluate_view as evaluate_view

        st_mock, _session = _fake_streamlit()
        st_mock.columns.return_value = (MagicMock(), MagicMock())
        st_mock.button.return_value = False
        result = self._result(score=90)
        with _patch_st(evaluate_view, st_mock), patch.object(
            evaluate_view, "track_impression"
        ):
            evaluate_view._render_effectiveness_score(result)
        st_mock.success.assert_called_once()
        st_mock.error.assert_not_called()
        st_mock.warning.assert_not_called()

    def test_red_band_uses_error_banner_and_shows_all_recommendations(self) -> None:
        import app.views.evaluate_view as evaluate_view

        st_mock, _session = _fake_streamlit()
        st_mock.columns.return_value = (MagicMock(), MagicMock())
        st_mock.button.return_value = False
        result = self._result(score=20)
        with _patch_st(evaluate_view, st_mock), patch.object(
            evaluate_view, "track_impression"
        ):
            evaluate_view._render_effectiveness_score(result)
        st_mock.error.assert_called_once()
        # All 3 recommendations rendered (one markdown title line each).
        title_calls = [
            c for c in st_mock.markdown.call_args_list if "1." in str(c) or "2." in str(c) or "3." in str(c)
        ]
        self.assertGreaterEqual(len(title_calls), 3)

    def test_yellow_band_uses_warning_banner(self) -> None:
        import app.views.evaluate_view as evaluate_view

        st_mock, _session = _fake_streamlit()
        st_mock.columns.return_value = (MagicMock(), MagicMock())
        st_mock.button.return_value = False
        result = self._result(score=60)
        with _patch_st(evaluate_view, st_mock), patch.object(
            evaluate_view, "track_impression"
        ):
            evaluate_view._render_effectiveness_score(result)
        st_mock.warning.assert_called_once()

    def test_impression_fires_once_per_reference(self) -> None:
        import app.views.evaluate_view as evaluate_view

        st_mock, _session = _fake_streamlit()
        st_mock.columns.return_value = (MagicMock(), MagicMock())
        st_mock.button.return_value = False
        result = self._result()
        with _patch_st(evaluate_view, st_mock), patch.object(
            evaluate_view, "track_impression"
        ) as mock_impression:
            evaluate_view._render_effectiveness_score(result)
            evaluate_view._render_effectiveness_score(result)
        mock_impression.assert_called_once()
        self.assertEqual(
            mock_impression.call_args.args[0], evaluate_view.EFFECTIVENESS_SCORE_FEATURE_ID
        )

    def test_click_on_claim_linked_recommendation_fires_interaction_and_jumps(self) -> None:
        import app.views.evaluate_view as evaluate_view

        st_mock, _session = _fake_streamlit()
        st_mock.columns.return_value = (MagicMock(), MagicMock())
        # First recommendation's button click returns True, others False.
        st_mock.button.side_effect = [True, False, False]
        result = self._result()
        with _patch_st(evaluate_view, st_mock), patch.object(
            evaluate_view, "track_impression"
        ), patch.object(evaluate_view, "track_interaction") as mock_interaction:
            evaluate_view._render_effectiveness_score(result)
        mock_interaction.assert_called_once_with(
            evaluate_view.EFFECTIVENESS_SCORE_FEATURE_ID,
            recommendationIndex=1,
            action="jump_to_claim",
        )

    def test_click_on_generic_recommendation_triggers_rewrite(self) -> None:
        import app.views.evaluate_view as evaluate_view

        st_mock, _session = _fake_streamlit()
        st_mock.columns.return_value = (MagicMock(), MagicMock())
        # Second recommendation (claim_id=None) clicked.
        st_mock.button.side_effect = [False, True, False]
        result = self._result()
        with _patch_st(evaluate_view, st_mock), patch.object(
            evaluate_view, "track_impression"
        ), patch.object(evaluate_view, "track_interaction") as mock_interaction:
            evaluate_view._render_effectiveness_score(result)
        mock_interaction.assert_called_once_with(
            evaluate_view.EFFECTIVENESS_SCORE_FEATURE_ID,
            recommendationIndex=2,
            action="trigger_rewrite",
        )

    def test_no_recommendations_still_renders_score(self) -> None:
        import app.views.evaluate_view as evaluate_view

        st_mock, _session = _fake_streamlit()
        st_mock.columns.return_value = (MagicMock(), MagicMock())
        result = self._result(recommendations=[])
        with _patch_st(evaluate_view, st_mock), patch.object(
            evaluate_view, "track_impression"
        ):
            evaluate_view._render_effectiveness_score(result)
        st_mock.button.assert_not_called()


if __name__ == "__main__":
    unittest.main()
