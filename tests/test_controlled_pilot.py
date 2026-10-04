"""Synthetic security regressions for the restricted real-information pilot."""
from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from tests import hermetic  # noqa: F401
from app import pilot
from app.documents import DocumentPage, ExtractedDocument, ExtractionError, document_from_text

CANARY = "SYNTHETIC_PERSON_RECORD_CANARY_84923"
_QUOTA_EXPIRY = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()


def approval() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "schema_version": 1, "reviewed_revision": "a" * 40,
        "approved_at": (now - timedelta(minutes=1)).isoformat(),
        "expires_at": (now + timedelta(days=1)).isoformat(),
        "issuer": "https://identity.example.test", "deployment_url": "https://pilot.example.test",
        "subjects": ["participant", "operator"], "operators": ["operator"],
        "models": ["test-model"], "provider_base_url": "https://provider.example.test/v1",
        "participant_notice": "SYNTHETIC NOTICE: test operator sends record and statement text to the synthetic test provider. Seven-day local count logs. Provider deletion is separate. Contact test operator.",
        "local_log_retention_days": 7,
        "quota_policy": {"pilot_id": "22f9db15-2a9a-4e00-83a5-c7bfe30062a4", "expires_at": _QUOTA_EXPIRY,
                         "participant_daily_starts": 2, "run_attempts": 50, "run_prompt_chars": 2_000_000,
                         "attempt_output_tokens": 8192, "attempt_charge_microusd": 1_000_000,
                         "pilot_total_microusd": 250_000_000, "pilot_total_attempts": 250},
        "single_instance": True, **{name: "synthetic evidence" for name in pilot.EVIDENCE_FIELDS},
    }


class TestAdmission(unittest.TestCase):
    def setUp(self):
        from tests.log_isolation import isolate_app_logs
        from app.logging_config import configure_logging
        directory = isolate_app_logs(self)
        context = patch.dict(os.environ, {"VA_LSE_LOG_DIR": directory})
        context.start()
        self.addCleanup(context.stop)
        self.addCleanup(lambda: configure_logging(log_dir="", force=True))
        from app import documents
        self.addCleanup(documents.set_active_extractor, documents._ACTIVE_EXTRACTOR)

    def test_operator_manifest_is_required_current_and_revision_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "approval.json"
            data = approval()
            with patch.dict(os.environ, {"VA_LSE_PILOT_APPROVAL_FILE": str(path), "VA_LSE_BUILD_SHA": "a" * 40}):
                with self.assertRaises(pilot.PilotBlocked):
                    pilot.load_approval()
                path.write_text(json.dumps(data))
                self.assertEqual(pilot.load_approval()["subjects"], data["subjects"])
                for field, value in (("privacy_review", ""), ("reviewed_revision", "old-revision"),
                                     ("expires_at", "2000-01-01T00:00:00Z"),
                                     ("single_instance", False), ("operators", ["outsider"]),
                                     ("participant_notice", ""), ("local_log_retention_days", True),
                                     ("local_log_retention_days", 31)):
                    with self.subTest(field=field):
                        path.write_text(json.dumps({**data, field: value}))
                        with self.assertRaises(pilot.PilotBlocked):
                            pilot.load_approval()

    def test_identity_requires_verified_invited_unexpired_short_lived_claims(self):
        claims = {"is_logged_in": True, "iss": approval()["issuer"], "sub": "participant", "iat": 950, "exp": 1100}
        owner = pilot.authorized_identity(claims, approval(), now=1000)
        self.assertEqual(len(owner), 64)
        self.assertEqual(owner, pilot.authorized_identity(claims, approval(), now=1001))
        for change in ({"is_logged_in": False}, {"iss": "https://evil.example"},
                       {"sub": "outsider"}, {"exp": 999}, {"exp": 99999},
                       {"iat": None}, {"exp": True}, {"iat": 1001}):
            with self.subTest(change=change), self.assertRaises(pilot.PilotBlocked):
                pilot.authorized_identity({**claims, **change}, approval(), now=1000)

    def test_default_closed_pilot_has_no_upload_or_action_controls(self):
        from streamlit.testing.v1 import AppTest
        root = Path(__file__).resolve().parents[1]
        with patch.dict(os.environ, {"VA_LSE_MODE": "controlled-pilot", "VA_LSE_PILOT_LOG_RETENTION_DAYS": "7", "VA_LSE_PILOT_APPROVAL_FILE": "/missing/approval.json"}):
            at = AppTest.from_file(str(root / "run_app.py")).run()
        self.assertFalse(at.exception)
        self.assertTrue(at.error)
        self.assertEqual(len(at.button), 0)
        self.assertEqual(len(at.text_area), 0)

    def test_destination_refuses_private_resolution_and_unapproved_urls(self):
        with patch.dict(os.environ, {"VA_LSE_MODE": "controlled-pilot", "VA_LSE_PILOT_LOG_RETENTION_DAYS": "7"}), patch.object(pilot, "load_approval", return_value=approval()):
            with self.assertRaises(pilot.PilotBlocked):
                pilot.require_destination("https://evil.example.test/v1")
            with self.assertRaises(pilot.PilotBlocked):
                pilot.require_destination(approval()["provider_base_url"], "unapproved-model")
            with patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("127.0.0.1", 443))]):
                with self.assertRaises(pilot.PilotBlocked):
                    pilot.require_destination(approval()["provider_base_url"])
            with patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("8.8.8.8", 443))]):
                pilot.require_destination(approval()["provider_base_url"], "test-model")

    def test_operator_key_is_not_a_widget_value_and_cannot_be_retargeted(self):
        from streamlit.testing.v1 import AppTest
        from app.views import shared
        from app.config import load_settings
        from dataclasses import replace
        root = Path(__file__).resolve().parents[1]
        with patch.dict(os.environ, {"OPENAI_API_KEY": CANARY}):
            at = AppTest.from_file(str(root / "run_app.py")).run()
        self.assertFalse(at.exception)
        self.assertFalse(any(CANARY in str(widget.value) for widget in at.text_input))
        self.assertFalse(any("API key" in widget.label for widget in at.text_input))
        state = MagicMock()
        state.get.return_value = "attacker value"
        managed = replace(load_settings(), api_key=CANARY, base_url="https://approved.example/v1")
        with patch.object(shared.st, "session_state", state), patch("app.config.load_settings", return_value=managed):
            self.assertIs(shared.session_settings(), managed)
        self.assertEqual(managed.api_key, CANARY)
        self.assertEqual(managed.base_url, "https://approved.example/v1")

    def test_valid_invited_profile_renders_without_managed_secret_widgets(self):
        from dataclasses import replace
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        from app import config
        root = Path(__file__).resolve().parents[1]
        data = approval()
        now = int(datetime.now(timezone.utc).timestamp())
        class VerifiedTestUser(dict):
            is_logged_in = True
        user = VerifiedTestUser(is_logged_in=True, iss=data["issuer"], sub="participant", iat=now - 1, exp=now + 300)
        from tests.pilot_budget_fixtures import install_budget
        install_budget(self, data)
        settings = replace(config.load_settings(), api_key=CANARY,
                           base_url=data["provider_base_url"], model_main="test-model", model_fast="test-model",
                           fallback_base_url="", fetch_api_key="")
        auth = {"redirect_uri": data["deployment_url"] + "/oauth2callback",
                "server_metadata_url": data["issuer"] + "/.well-known/openid-configuration",
                "client_id": "synthetic-client", "client_secret": "synthetic-secret",
                "cookie_secret": "a" * 64}
        with patch.dict(os.environ, {"VA_LSE_MODE": "controlled-pilot", "VA_LSE_PILOT_LOG_RETENTION_DAYS": "7"}), patch.object(pilot, "load_approval", return_value=data), \
                patch.object(config, "load_settings", return_value=settings), patch.object(st, "user", user), \
                patch.object(config, "MAX_RECORD_PAGES", 500), patch.object(config, "BLOB_STORE_MODE", "none"), \
                patch.object(config, "SHARED_CACHE_URL", ""), patch.object(config, "SHARED_CACHE_TOKEN", ""), \
                patch("sys.platform", "linux"), patch("os.geteuid", return_value=os.geteuid()), \
                patch("app.isolated_extract.parser_health"):
            at = AppTest.from_file(str(root / "run_app.py"))
            at.secrets["auth"] = auth
            at.run()
            self.assertEqual(len(at.tabs), 0)
            self.assertEqual(len(at.text_area), 0)
            at.checkbox[0].check().run()
            self.assertEqual(len(at.tabs), 4)
            at.session_state["private_case"] = CANARY
            data["participant_notice"] += " Updated notice."
            at.run()
            self.assertEqual(len(at.tabs), 0)
            self.assertNotIn("private_case", at.session_state)
            self.assertEqual(len(at.text_area), 0)
            at.checkbox[0].check().run()
        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        self.assertEqual(len(at.tabs), 4)
        self.assertFalse(any("API key" in item.label for item in at.text_input))

    def test_provider_budget_refuses_work_before_the_http_call(self):
        import threading
        from app.llm import LLMClient
        client = LLMClient.__new__(LLMClient)
        client._pilot_calls = 200
        client._pilot_prompt_chars = 0
        client._pilot_call_lock = threading.Lock()
        client._client = MagicMock()
        client._settings = MagicMock(base_url=approval()["provider_base_url"])
        with patch.dict(os.environ, {"VA_LSE_MODE": "controlled-pilot", "VA_LSE_PILOT_LOG_RETENTION_DAYS": "7"}), \
                patch.object(pilot, "current_owner", return_value="participant"), \
                patch.object(pilot, "require_destination"):
            with self.assertRaises(pilot.PilotBlocked):
                client._call_openai("primary", "test-model", "system", "synthetic input", 0, 100, None)
        client._client.chat.completions.create.assert_not_called()

    def test_verified_identity_reaches_digest_threads_and_is_rechecked(self):
        import contextvars
        import streamlit as st
        from concurrent.futures import ThreadPoolExecutor
        now = int(datetime.now(timezone.utc).timestamp())
        data = approval()
        claims = {"is_logged_in": True, "iss": data["issuer"], "sub": "participant", "iat": now - 1, "exp": now + 300}
        from tests.pilot_budget_fixtures import install_budget
        install_budget(self, data)
        docs = [document_from_text("a.txt", "Synthetic knee observation for a thread identity test.")]
        with patch.dict(os.environ, {"VA_LSE_MODE": "controlled-pilot", "VA_LSE_PILOT_LOG_RETENTION_DAYS": "7"}), \
                patch.object(pilot, "load_approval", return_value=data), patch.object(st, "user", claims), \
                patch.object(pilot, "require_consent", return_value="synthetic-consent"):
            with self.assertRaises(pilot.PilotBlocked), pilot.action_budget(docs):
                context = contextvars.copy_context()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    self.assertEqual(executor.submit(context.run, pilot.current_owner).result(),
                                     pilot.authorized_identity(claims, data))
                data["subjects"].remove("participant")
                with self.assertRaises(pilot.PilotBlocked):
                    context.run(pilot.current_owner)
        self.assertIsNone(pilot._run_claims.get())


class TestOwnershipAndPrivacy(unittest.TestCase):
    def test_reference_cannot_read_foreign_or_legacy_result(self):
        from app.job_queue import JobRecord, STATUS_DONE
        from app.views.job_runner import _fetch_outcome
        backend = MagicMock()
        for owner in ("", "other-participant"):
            record = JobRecord("job", "evaluate", owner_id=owner, status=STATUS_DONE)
            with patch.object(pilot, "current_owner", return_value="participant"):
                result = _fetch_outcome(backend, record, "eval")
            self.assertFalse(result.ok)
            self.assertEqual(result.error_class, "AccessDenied")
        backend.get_result.assert_not_called()

    def test_non_operator_cannot_read_shared_diagnostics(self):
        from app.views import ops
        with patch.object(pilot, "operator_allowed", return_value=False), patch.object(ops, "_cached_detail") as read:
            ops.render_failure_detail("req_aaaaaaaaaaaa")
        read.assert_not_called()

    def test_all_pilot_logging_sinks_drop_content_and_exception_chains(self):
        from app.logging_config import JsonFormatter, PlainFormatter
        from app.diagnostics import CaptureHandler
        from app import run_log, audit
        record = logging.LogRecord("app.test", logging.ERROR, CANARY, 1, CANARY, (), None)
        record.request_id = "req_aaaaaaaaaaaa"
        record.record_pages = 3
        record.diagnostics = {"name": CANARY}
        record.stack_info = CANARY
        with patch.dict(os.environ, {"VA_LSE_MODE": "controlled-pilot", "VA_LSE_PILOT_LOG_RETENTION_DAYS": "7"}):
            for formatter in (JsonFormatter(), PlainFormatter()):
                self.assertNotIn(CANARY, formatter.format(record))
            handler = CaptureHandler()
            handler.emit(record)
            self.assertNotIn(CANARY, repr(handler.snapshot()))
            with tempfile.TemporaryDirectory() as directory, patch.object(run_log, "_resolve_log_path", return_value=Path(directory) / "runs.jsonl"):
                run_log.run_log_event("evaluate", "error", request_id=record.request_id, error=CANARY, traceback=CANARY, patient=CANARY, pages=3)
                saved = (Path(directory) / "runs.jsonl").read_text()
                self.assertNotIn(CANARY, saved)
                self.assertIn("req_aaaaaaaaaaaa", saved)
            sink = MagicMock()
            with patch.object(audit, "get_audit_logger", return_value=sink), patch.object(audit, "get_audit_session_id", return_value="synthetic-session"):
                audit.audit_event("evaluate", "error", request_id=record.request_id, condition=CANARY, error=ValueError(CANARY), outcome={"name": CANARY, "nested": {"record": CANARY}})
            self.assertNotIn(CANARY, repr(sink.call_args_list))

    def test_telemetry_and_downloads_have_no_outbound_pilot_path(self):
        from app import telemetry, agiloop_telemetry
        container = MagicMock()
        with patch.dict(os.environ, {"VA_LSE_MODE": "controlled-pilot", "VA_LSE_PILOT_LOG_RETENTION_DAYS": "7"}), patch.object(telemetry, "HTTPSConnection") as network:
            telemetry._send({"record": CANARY})
            telemetry._dispatch({"record": CANARY})
            agiloop_telemetry.track_feature_error("test", ValueError(CANARY), source=CANARY)
            self.assertFalse(pilot.file_download("Export", data=CANARY, container=container))
        network.assert_not_called()
        container.download_button.assert_not_called()

    def test_record_markup_is_literal_and_dynamic_labels_are_escaped(self):
        container = MagicMock()
        content = f'![record](https://outside.invalid/{CANARY})<img src="https://outside.invalid/{CANARY}">'
        with patch.dict(os.environ, {"VA_LSE_MODE": "controlled-pilot", "VA_LSE_PILOT_LOG_RETENTION_DAYS": "7"}):
            for method in ("markdown", "write", "caption", "error", "warning"):
                pilot.display(content, container=container, method=method, unsafe_allow_html=True)
                getattr(container, method).assert_not_called()
                self.assertIn(content, container.text.call_args.args[0])
            self.assertIn(r"\!\[record\]\(", pilot.text_label(content))
        with patch.dict(os.environ, {"VA_LSE_MODE": "synthetic"}):
            pilot.display("synthetic report", container=container, method="markdown")
        container.markdown.assert_called_once_with("synthetic report")

    def test_real_http_clients_refuse_redirects_and_environment_proxies(self):
        import threading
        import urllib.error
        import urllib.request
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from app.llm import _pilot_transport, _probe_open
        received = []
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                received.append(self.path)
                self.send_response(302)
                self.send_header("Location", "/record-receiver")
                self.end_headers()
            def log_message(self, *_):
                pass
        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}/redirect"
            try:
                with patch.dict(os.environ, {"VA_LSE_MODE": "controlled-pilot", "VA_LSE_PILOT_LOG_RETENTION_DAYS": "7",
                                            "HTTP_PROXY": "http://127.0.0.1:1", "NO_PROXY": ""}), \
                        patch.object(pilot, "require_destination"):
                    client = _pilot_transport("https://approved.example.test/v1")["http_client"]
                    with client:
                        self.assertEqual(client.get(url, timeout=2).status_code, 302)
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        _probe_open(urllib.request.Request(url), timeout=2)
                    self.assertEqual(error.exception.code, 302)
            finally:
                server.shutdown()
                thread.join(2)
        self.assertEqual(received, ["/redirect", "/redirect"])

    def test_quota_limits_and_missing_coverage_fail_before_work(self):
        from tests.pilot_budget_fixtures import install_budget
        data = approval()
        install_budget(self, data)
        docs = [document_from_text("record.txt", "Synthetic medical text for a bounded test.")]
        with patch.dict(os.environ, {"VA_LSE_MODE": "controlled-pilot", "VA_LSE_PILOT_LOG_RETENTION_DAYS": "7"}), patch.object(pilot, "current_owner", return_value="quota-test"), \
                patch.object(pilot, "require_consent", return_value="synthetic-consent"), patch.object(pilot, "load_approval", return_value=data):
            with pilot.action_budget(docs):
                with self.assertRaises(pilot.PilotBlocked):
                    with pilot.action_budget(docs):
                        self.fail("concurrent work admitted")
            with pilot.action_budget(docs):
                pass
            with self.assertRaises(pilot.PilotBlocked):
                with pilot.action_budget(docs):
                    self.fail("third start admitted")
            docs[0].coverage_known = False
            with self.assertRaises(pilot.PilotBlocked):
                with pilot.action_budget(docs):
                    self.fail("unknown coverage admitted")


class TestEvidenceRegressions(unittest.TestCase):
    def test_archive_cache_returns_every_document_and_replays_rejections(self):
        from app.views import uploads
        docs = [document_from_text("a.txt", "First synthetic record."), document_from_text("b.txt", "Second synthetic record.")]
        upload = MagicMock(name="upload")
        upload.name = "records.zip"
        upload.size = 10
        upload.getvalue.return_value = b"synthetic archive bytes"
        state = {}
        streamlit = MagicMock()
        streamlit.session_state = state
        with patch.object(uploads, "st", streamlit), patch.object(uploads, "extract_uploaded_documents", return_value=(docs, ["unreadable member"])) as extractor:
            first = uploads.extract_uploads([upload], "eval")
            second = uploads.extract_uploads([upload], "eval")
        self.assertEqual(first, second)
        self.assertEqual(len(second), 2)
        self.assertEqual(extractor.call_count, 1)

    def test_serialization_preserves_scans_totals_and_citation_units(self):
        from app.job_payload import document_to_json, document_from_json
        doc = ExtractedDocument("record.pdf", [DocumentPage("record.pdf", 1, "synthetic text")], total_pages=3, unreadable_pages=[2, 3])
        saved = document_from_json(document_to_json(doc))
        self.assertEqual(saved.source_page_count, 3)
        self.assertEqual(saved.unreadable_pages, [2, 3])
        block = ExtractedDocument("record.txt", [DocumentPage("record.txt", 1, "Synthetic text record.", "block")], total_pages=1, pagination="block")
        self.assertEqual(document_from_json(document_to_json(block)).pages[0].kind, "block")
        legacy = document_from_json({"filename": "old.pdf", "pages": [{"page": 1, "text": "text"}]})
        self.assertFalse(legacy.coverage_known)

    def test_repeated_clinical_observations_are_preserved(self):
        from app.documents import strip_running_headers
        pages = [DocumentPage("a.pdf", n, "Patient denies chest pain.\nRepeated clinical assessment.") for n in range(1, 7)]
        self.assertEqual(strip_running_headers(pages), pages)

    def test_changed_facts_and_empty_grounding_cannot_be_accepted(self):
        from app.draft import _review_rejection_reason, _normalize_grounding
        from app.llm import LLMParseError
        original = ("I recall the onset was approximately 2019. I observed pain twice a week. " * 5)
        for change in (original.replace("2019", "2021"), original.replace("twice", "daily"),
                       original.replace("I recall", "I know")):
            self.assertTrue(_review_rejection_reason(original, change))
        with self.assertRaises(LLMParseError):
            _normalize_grounding({})

    def test_missing_source_is_invalid_instead_of_skipped(self):
        from app.medical_review import MedicalFact, verify_citations
        fact = MedicalFact(date="2020", type="symptom", description="synthetic", source="absent.pdf p.7", document="absent.pdf", page=7, quote="Synthetic source quotation.")
        check = verify_citations([fact], [])
        self.assertEqual((check["checked"], check["missing"], check["skipped"]), (1, 1, 0))

    def test_full_and_named_dates_preserve_calendar_order(self):
        from app.medical_review import _normalize_date_for_sort
        self.assertEqual(_normalize_date_for_sort("2019-11-03"), "2019-11-03")
        self.assertEqual(_normalize_date_for_sort("January 2020"), "2020-01-01")
        self.assertEqual(_normalize_date_for_sort("2020-02-31"), "9999-99-99")

    def test_partial_dates_do_not_create_exact_gap_durations(self):
        from app.medical_review import MedicalDigest, MedicalFact, build_timeline_data
        digest = MedicalDigest(summary="Synthetic dates", facts=[
            MedicalFact("2015", "diagnosis", "Synthetic observation", "a.pdf p1"),
            MedicalFact("June 2021", "treatment", "Synthetic observation", "a.pdf p2"),
        ])
        data = build_timeline_data(digest)
        self.assertEqual(data["dated_count"], 2)
        self.assertEqual(data["gaps"], [])

    def test_pilot_timeline_does_not_contact_an_optional_date_provider(self):
        from app.views import evaluate_view
        from app.medical_review import MedicalDigest, MedicalFact
        digest = MedicalDigest(summary="Synthetic undated fact", facts=[
            MedicalFact("unknown", "symptom", "Synthetic observation", "a.pdf p1"),
        ])
        result = MagicMock(digest=digest)
        with patch.dict(os.environ, {"VA_LSE_MODE": "controlled-pilot", "VA_LSE_PILOT_LOG_RETENTION_DAYS": "7"}), \
                patch.object(evaluate_view.st, "session_state", {}), \
                patch.object(evaluate_view, "get_llm") as provider, \
                patch.object(evaluate_view, "build_timeline_data", return_value={}) as build:
            evaluate_view._render_medical_timeline(result, request_reference="req_aaaaaaaaaaaa")
        provider.assert_not_called()
        self.assertIsNone(build.call_args.args[1])

    def test_protected_parser_preserves_archive_member_citations(self):
        from app.isolated_extract import _documents
        from app.documents import InProcessExtractor
        from app.job_payload import documents_to_json
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("one.txt", "First synthetic observation of knee pain.")
            archive.writestr("two.txt", "Second synthetic observation of limited walking.")
        # Synthetic schema fixture only; actual OS boundary runs in Linux CI.
        docs, skipped = InProcessExtractor().extract("records.zip", payload.getvalue())
        request = {"version": 1, "label": "records.zip", "sha256": "a" * 64, "nonce": "b" * 32}
        reply = {**request, "image": "sha256:" + "c" * 64, "documents": documents_to_json(docs), "skipped": skipped}
        docs, skipped = _documents(reply, request, reply["image"])
        self.assertEqual(len(docs), 2)
        self.assertFalse(skipped)
        self.assertTrue(all(doc.coverage_known for doc in docs))

    def test_parser_timeout_is_fail_closed_without_parent_fallback(self):
        from app.isolated_extract import IsolatedExtractor
        with patch.dict(os.environ, {"VA_LSE_PARSER_IMAGE": "sha256:" + "a" * 64}), \
                patch("app.isolated_extract.socket.socket") as constructor, patch("app.documents.InProcessExtractor.extract") as fallback:
            constructor.return_value.__enter__.return_value.connect.side_effect = TimeoutError("synthetic timeout")
            with self.assertRaises(ExtractionError):
                IsolatedExtractor(timeout=1).extract("record.txt", b"synthetic record")
        fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
