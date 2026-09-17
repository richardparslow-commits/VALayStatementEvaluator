"""Unit tests for the app/views layer — mocked Streamlit, no AppTest.

Locks the app↔views boundary without spinning up the whole Streamlit runtime:
pure helpers run directly; st-touching helpers run under a MagicMock ``st``
plus a fake session_state dict.
"""

from __future__ import annotations

import sys
import types
import unittest
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


if __name__ == "__main__":
    unittest.main()
