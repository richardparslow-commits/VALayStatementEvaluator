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


if __name__ == "__main__":
    unittest.main()
