"""Invented signed-cookie HTTP and private export lifecycle tests; no provider calls."""
from __future__ import annotations

import asyncio
import gc
import json
import os
import subprocess
import sys
import time
import unittest
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from tests import hermetic  # noqa: F401
from tests.test_controlled_pilot import approval
from app import export_cookie, export_routes, pilot, text_exports
from app.factual_integrity import fingerprint
from app.text_exports import ExportLease, ExportStore, ExportUnavailable, policy_binding
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.testclient import TestClient
from streamlit.web.server.starlette.starlette_app_utils import create_signed_value

SECRET = "INVENTED_TEST_COOKIE_SECRET_73642_" * 2
ORIGIN = "https://pilot.example.invalid"
TEXT = 'Invented observation: I saw knee swelling around 2020. <img src="https://outside.invalid/CANARY">'


class TextExportTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {"VA_LSE_MODE": "controlled-pilot", "VA_LSE_PILOT_TEXT_EXPORTS": "1"}))
        self.data = approval()
        self.data.update(deployment_url=ORIGIN, subjects=["invented-a", "invented-b"], operators=["invented-a"],
                         text_export_policy={"enabled": True, "formats": ["txt"], "evidence_reference": "INVENTED_TEST_ONLY_REFERENCE"})
        self.stack.enter_context(patch.object(pilot, "load_approval", return_value=self.data))
        import streamlit as st
        self.stack.enter_context(patch.object(st, "secrets", {"auth": {
            "cookie_secret": SECRET, "redirect_uri": ORIGIN + "/oauth2callback",
            "server_metadata_url": self.data["issuer"] + "/.well-known/openid-configuration"}}))
        self.tick = 0.0
        self.store = ExportStore(clock=lambda: self.tick)
        self.store.start()
        self.addCleanup(self.store.close)
        self.stack.enter_context(patch.object(export_routes, "STORE", self.store))
        self.stack.enter_context(patch.object(text_exports, "STORE", self.store))
        self.stack.enter_context(patch("app.shutdown.is_shutting_down", return_value=False))
        self.owner = pilot.authorized_identity(self.identity(), self.data)
        self.consent = pilot.ConsentGrant(pilot.notice_binding(self.owner, self.data))
        self.lease = self.lease_for(TEXT)
        self.handle = self.store.create(self.lease, TEXT)
        self.client = TestClient(Starlette(routes=export_routes.routes()), base_url=ORIGIN)
        self.addCleanup(self.client.close)

    def identity(self, subject="invented-a", **overrides):
        now = time.time()
        return {"origin": ORIGIN, "is_logged_in": True, "iss": self.data["issuer"], "sub": subject,
                "iat": now - 1, "exp": now + 100, **overrides}

    def cookie(self, payload=None):
        value = json.dumps(self.identity() if payload is None else payload)
        return "_streamlit_user=" + create_signed_value(SECRET, "_streamlit_user", value).decode()

    def lease_for(self, text, scope="invented-review"):
        return ExportLease(self.owner, "draft", fingerprint(scope), fingerprint(text), policy_binding(self.data), self.consent)

    def get(self, *, cookie=None, handle=None, extra=None):
        headers = {"cookie": self.cookie() if cookie is None else cookie, **(extra or {})}
        return self.client.get("/pilot-exports/" + (self.handle if handle is None else handle), headers=headers)

    def refused(self, response):
        self.assertEqual(response.status_code, 404)
        self.assertNotIn(TEXT.encode(), response.content)
        self.assertEqual(response.text, "Download unavailable.")
        self.assertIn("no-store", response.headers["cache-control"])
        self.assertNotIn("content-disposition", response.headers)

    def test_owner_receives_exact_utf8_txt_and_fixed_private_headers(self):
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, TEXT.encode())
        self.assertEqual(response.headers["content-type"], "text/plain; charset=utf-8")
        self.assertEqual(response.headers["content-disposition"], 'attachment; filename="reviewed_statement.txt"')
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["referrer-policy"], "no-referrer")
        self.assertIn("sandbox", response.headers["content-security-policy"])
        self.assertIn("no-store", response.headers["cache-control"])
        self.assertEqual(response.headers["vary"], "Cookie")
        self.assertNotIn("etag", response.headers)
        self.assertNotIn("set-cookie", response.headers)

    def test_known_handle_is_not_a_bearer_credential(self):
        for cookie in ("", self.cookie(self.identity("invented-b")), self.cookie().replace("=", "=tamper", 1)):
            with self.subTest(cookie=cookie[:20]):
                self.refused(self.get(cookie=cookie))
        self.refused(self.get(handle="f" * 64))

    def test_expired_revoked_foreign_issuer_and_invalid_identity_fail_with_known_handle(self):
        for overrides in ({"exp": time.time() - 1}, {"iat": time.time() + 10}, {"exp": time.time() + 7200},
                          {"iss": "https://outside.invalid"}, {"sub": "uninvited"}, {"is_logged_in": False},
                          {"exp": True}, {"exp": float("nan")}, {"origin": "https://outside.invalid"}):
            with self.subTest(overrides=overrides):
                self.refused(self.get(cookie=self.cookie(self.identity(**overrides))))
        self.data["subjects"].remove("invented-a")
        self.refused(self.get())

    def test_clear_logout_consent_withdrawal_and_review_change_revoke_bytes(self):
        self.consent.revoked.set()
        self.refused(self.get())
        self.assertEqual(self.store.sweep()["bytes"], 0)
        self.consent = pilot.ConsentGrant(pilot.notice_binding(self.owner, self.data))
        self.lease = self.lease_for(TEXT)
        self.handle = self.store.create(self.lease, TEXT)
        self.store.revoke(self.lease)
        self.refused(self.get())
        self.assertEqual(self.store.sweep()["files"], 0)

    def test_losing_the_session_lease_drops_the_artifact(self):
        del self.lease
        gc.collect()
        self.refused(self.get())
        self.assertEqual(self.store.sweep(), {"files": 0, "bytes": 0})

    def test_expiry_is_not_extended_by_repeated_creation_or_download(self):
        self.tick = 299
        self.assertEqual(self.store.create(self.lease, TEXT), self.handle)
        self.assertEqual(self.get().status_code, 200)
        self.tick = 300
        self.refused(self.get())
        self.assertEqual(self.store.sweep()["bytes"], 0)

    def test_idle_expiry_runs_without_any_requests(self):
        self.tick = 301
        time.sleep(1.15)
        self.assertEqual(len(self.store._files), 0)

    def test_mode_scope_approval_and_current_notice_changes_close_delivery(self):
        for key, value in (("models", ["another-model"]), ("participant_notice", "Changed invented notice"),
                           ("deployment_validation", "Changed accepted-release reference")):
            with self.subTest(key=key):
                old = self.data[key]
                self.data[key] = value
                self.refused(self.get())
                self.data[key] = old
        for mode, flag in (("synthetic", "1"), ("controlled-pilot", "0")):
            with patch.dict(os.environ, {"VA_LSE_MODE": mode, "VA_LSE_PILOT_TEXT_EXPORTS": flag}):
                self.refused(self.get())
        with patch.object(text_exports.Path, "read_text", return_value="[server]\nbaseUrlPath='nested'\n"):
            self.refused(self.get())
        original = deepcopy(self.data["text_export_policy"])
        for policy in ({}, {**original, "enabled": False}, {**original, "evidence_reference": ""},
                       {**original, "formats": ["pdf"]}, {**original, "unknown": True}):
            self.data["text_export_policy"] = policy
            self.refused(self.get())

    def test_bad_origin_cross_site_range_query_path_and_head_are_refused(self):
        for headers in ({"host": "outside.invalid"}, {"origin": "https://outside.invalid"},
                        {"sec-fetch-site": "cross-site"}, {"range": "bytes=0-10"}):
            with self.subTest(headers=headers):
                self.refused(self.get(extra=headers))
        for handle in (self.handle + "?anything=1", "short", "../" + self.handle, self.handle + "/record.txt"):
            response = self.get(handle=handle)
            self.assertEqual(response.status_code, 404)
            self.assertNotIn(TEXT.encode(), response.content)
        head = self.client.head("/pilot-exports/" + self.handle, headers={"cookie": self.cookie()})
        self.assertEqual(head.status_code, 404)
        self.assertEqual(head.content, b"")

    def test_shutdown_and_stopped_cleanup_deny_delivery(self):
        with patch("app.shutdown.is_shutting_down", return_value=True):
            self.refused(self.get())
        self.store.close()
        self.refused(self.get())

    def test_a_response_built_before_revocation_rechecks_when_delivered(self):
        async def exercise():
            scope = {"type": "http", "method": "GET", "scheme": "https", "path": "/pilot-exports/" + self.handle,
                     "query_string": b"", "headers": [(b"host", b"pilot.example.invalid"), (b"cookie", self.cookie().encode())],
                     "path_params": {"handle": self.handle}}
            response = await export_routes.download(Request(scope))
            self.store.revoke(self.lease)
            messages = []
            async def receive():
                return {"type": "http.request"}
            async def send(message):
                messages.append(message)
            await response(scope, receive, send)
            self.assertEqual(messages[0]["status"], 404)
            self.assertNotIn(TEXT.encode(), b"".join(m.get("body", b"") for m in messages))
        asyncio.run(exercise())

    def test_bounded_size_capacity_and_exact_text_hash_are_enforced(self):
        with self.assertRaises(ExportUnavailable):
            self.store.create(self.lease, TEXT + " altered")
        for text in ("", "[Confirm: date]", "€" * (text_exports.MAX_BYTES // 2)):
            lease = self.lease_for(text)
            with self.assertRaises(ExportUnavailable):
                self.store.create(lease, text)
        with patch.object(text_exports, "MAX_TOTAL_BYTES", len(TEXT.encode()) * 2):
            other = self.lease_for(TEXT, "second scope")
            self.store.create(other, TEXT)
            third = self.lease_for(TEXT, "third scope")
            with self.assertRaises(ExportUnavailable):
                self.store.create(third, TEXT)
            self.store.revoke(other)
            self.store.create(third, TEXT)
        with patch.object(text_exports, "MAX_FILES", len(self.store._files)):
            fourth = self.lease_for(TEXT, "fourth scope")
            with self.assertRaises(ExportUnavailable):
                self.store.create(fourth, TEXT)

    def test_real_clear_case_revokes_the_live_lease_and_copied_consent(self):
        import streamlit as st
        state = {"_text_export_lease_draft": self.lease, "_text_export_handle_draft": self.handle,
                 "_pilot_consent_grant": self.consent, "invented_working_case": TEXT}
        with patch.object(st, "session_state", state):
            pilot.clear_case()
        self.assertEqual(state, {})
        self.assertTrue(self.consent.revoked.is_set())
        self.assertTrue(self.lease.revoked.is_set())
        self.refused(self.get())
        self.assertEqual(self.store.sweep(), {"files": 0, "bytes": 0})

    def test_source_selection_and_review_withdrawal_revoke_known_handles(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        self.stack.enter_context(patch.object(pilot, "current_owner", return_value=self.owner))
        code = '''
from app.views.factual_review import render_factual_review
from tests.test_factual_integrity import result, ACCOUNT
render_factual_review(result(), ACCOUNT, slot="draft")
'''
        for action in ("source", "review"):
            at = AppTest.from_string(code, default_timeout=20).run()
            at.checkbox[0].check().run()
            self.lease = self.lease_for(TEXT, action)
            self.handle = self.store.create(self.lease, TEXT)
            at.session_state["_text_export_lease_draft"] = self.lease
            at.session_state["_text_export_handle_draft"] = self.handle
            if action == "source":
                at.multiselect[0].set_value([]).run()
            else:
                at.checkbox[0].uncheck().run()
            self.assertFalse(at.exception)
            self.refused(self.get())

    def test_supported_asgi_entrypoint_delivers_and_closes_the_store(self):
        code = '''
from tests import hermetic
import os
os.environ["VA_LSE_MODE"] = "controlled-pilot"
os.environ["VA_LSE_PILOT_TEXT_EXPORTS"] = "1"
import streamlit as st
from unittest.mock import patch
from app.pilot_asgi import app
from app import pilot, text_exports
from tests.test_controlled_pilot import approval
from tests.test_text_exports import SECRET, ORIGIN, TEXT
from app.factual_integrity import fingerprint
from streamlit.web.server.starlette.starlette_app_utils import create_signed_value
from starlette.testclient import TestClient
import json, time
data = approval()
data.update(deployment_url=ORIGIN, text_export_policy={"enabled": True, "formats": ["txt"], "evidence_reference": "INVENTED_TEST_ONLY"})
identity = {"origin": ORIGIN, "is_logged_in": True, "iss": data["issuer"], "sub": data["subjects"][0], "iat": time.time()-1, "exp": time.time()+100}
owner = pilot.authorized_identity(identity, data)
consent = pilot.ConsentGrant(pilot.notice_binding(owner, data))
with patch.object(pilot, "load_approval", return_value=data), patch.object(st, "secrets", {"auth": {"cookie_secret": SECRET, "redirect_uri": ORIGIN+"/oauth2callback", "server_metadata_url": data["issuer"]+"/.well-known/openid-configuration"}}):
    text_exports.STORE.start()
    lease = text_exports.ExportLease(owner, "draft", fingerprint("invented scope"), fingerprint(TEXT), text_exports.policy_binding(data), consent)
    handle = text_exports.STORE.create(lease, TEXT)
    cookie = "_streamlit_user="+create_signed_value(SECRET, "_streamlit_user", json.dumps(identity)).decode()
    with TestClient(app, base_url=ORIGIN) as client:
        response = client.get("/pilot-exports/"+handle, headers={"cookie": cookie, "accept-encoding": "gzip"})
        assert response.status_code == 200, response.status_code
        assert response.content == TEXT.encode()
        assert response.headers["content-encoding"] == "identity"
        assert "no-store" in response.headers["cache-control"]
    assert not text_exports.STORE.active
    assert text_exports.STORE.sweep() == {"files": 0, "bytes": 0}
'''
        run = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parent.parent,
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_legacy_media_downloads_stay_disabled_even_when_txt_policy_is_accepted(self):
        from unittest.mock import MagicMock
        container = MagicMock()
        self.assertFalse(pilot.file_download("unsafe report", data=TEXT, container=container))
        container.download_button.assert_not_called()

    def test_default_disabled_renderer_retains_no_artifacts_or_links(self):
        import streamlit as st
        from app.views import text_export
        from unittest.mock import MagicMock
        state = {}
        container = MagicMock()
        with patch.dict(os.environ, {"VA_LSE_PILOT_TEXT_EXPORTS": "0"}), \
                patch.object(st, "session_state", state), patch.object(text_export, "st", container):
            text_export.render_text_export(TEXT, slot="draft", confirmed=True)
        self.assertEqual(state, {})
        container.button.assert_not_called()
        container.link_button.assert_not_called()

    def test_a_new_run_revokes_previous_links_before_validation_or_provider_work(self):
        import streamlit as st
        from app.views import draft_view, evaluate_view
        for slot in ("draft", "eval"):
            self.lease = ExportLease(self.owner, slot, fingerprint("run scope"), fingerprint(TEXT),
                                     policy_binding(self.data), self.consent)
            self.handle = self.store.create(self.lease, TEXT)
            state = {"_text_export_lease_" + slot: self.lease, "_text_export_handle_" + slot: self.handle}
            with patch.object(st, "session_state", state):
                if slot == "draft":
                    with patch.object(draft_view, "_validate_draft_inputs", return_value=False):
                        draft_view._run_draft_flow(rid="invented", records=[], condition="", claim_type="", witness={}, observations="")
                else:
                    with patch.object(evaluate_view, "_validate_evaluate_inputs", return_value=False):
                        evaluate_view._run_evaluation_flow("", [])
            self.assertTrue(self.lease.revoked.is_set())
            self.refused(self.get())

    def test_launcher_starts_expiry_only_with_the_accepted_pinned_scope(self):
        from app import pilot_server
        from unittest.mock import MagicMock
        store = MagicMock()
        with patch.object(text_exports, "STORE", store), patch.object(pilot, "validate_log_policy"), \
                patch("app.log_retention.retention_days", return_value=7), \
                patch("app.pilot_budget.get_ledger"), patch("app.shutdown.install_signal_handlers"), \
                patch.dict(os.environ, {"VA_LSE_HEALTH_PORT": "0"}):
            with patch.dict(os.environ, {"VA_LSE_PILOT_TEXT_EXPORTS": "0"}):
                pilot_server.initialize()
                store.start.assert_not_called()
            with patch("importlib.metadata.version", return_value="unreviewed"):
                with self.assertRaises(pilot.PilotBlocked):
                    pilot_server.initialize()
                store.start.assert_not_called()
            pilot_server.initialize()
            store.start.assert_called_once()

    def test_no_content_cookie_or_handle_reaches_application_logs_on_refusal(self):
        import logging
        with patch.object(logging.Logger, "error") as error, patch.object(logging.Logger, "exception") as exception:
            self.refused(self.get(cookie="_streamlit_user=INVENTED_BAD_COOKIE_CANARY"))
        error.assert_not_called()
        exception.assert_not_called()

    def test_cookie_duplicates_oversize_unsupported_runtime_and_authenticated_bomb_are_bounded(self):
        raw = self.cookie().split("=", 1)[1]
        for headers in ([self.cookie() + "; " + self.cookie()], [self.cookie(), self.cookie()],
                        ["_streamlit_user=" + "a" * 20000]):
            with self.assertRaises(ExportUnavailable):
                export_cookie.claims(headers, SECRET, ORIGIN)
        with patch.object(export_cookie, "version", return_value="unreviewed-version"):
            with self.assertRaises(ExportUnavailable):
                export_cookie.claims([self.cookie()], SECRET, ORIGIN)
        bomb = create_signed_value(SECRET, "_streamlit_user", "x" * 1_000_000).decode()
        with self.assertRaises(ExportUnavailable):
            export_cookie.claims(["_streamlit_user=" + bomb], SECRET, ORIGIN)
        self.refused(self.get(cookie="_streamlit_user=" + raw[:-1] + ("A" if raw[-1] != "A" else "B")))

    def test_actual_streamlit_chunked_cookie_protocol_and_duplicate_json_refusal(self):
        payload = json.dumps(self.identity())
        count = create_signed_value(SECRET, "_streamlit_user", "chunks-2").decode()
        first = create_signed_value(SECRET, "_streamlit_user_1", payload[:len(payload)//2]).decode()
        second = create_signed_value(SECRET, "_streamlit_user_2", payload[len(payload)//2:]).decode()
        cookie = f"_streamlit_user={count}; _streamlit_user_1={first}; _streamlit_user_2={second}"
        self.assertEqual(self.get(cookie=cookie).status_code, 200)
        self.refused(self.get(cookie=cookie.rsplit(";", 1)[0]))
        duplicate = payload[:-1] + ', "sub": "invented-b"}'
        self.refused(self.get(cookie="_streamlit_user=" + create_signed_value(SECRET, "_streamlit_user", duplicate).decode()))

    def test_import_has_no_ui_config_client_or_worker_side_effects(self):
        code = ('import sys; import app.text_exports, app.export_cookie, app.export_routes; '
                'assert not {"streamlit", "app.config", "app.llm", "app.telemetry", "openai"}.intersection(sys.modules)')
        run = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parent.parent,
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_real_widgets_invalidate_a_prepared_download_on_edit_and_revert(self):
        from streamlit.testing.v1 import AppTest
        from app.views import text_export
        self.stack.enter_context(patch.object(text_export, "STORE", self.store))
        self.stack.enter_context(patch.object(pilot, "current_owner", return_value=self.owner))
        self.stack.enter_context(patch.object(pilot, "require_session_access"))
        self.stack.enter_context(patch.object(pilot, "require_consent", return_value=self.consent))
        code = '''
import streamlit as st
from app import pilot
from app.views.factual_review import render_factual_review
from app.views.text_export import render_text_export
from tests.test_factual_integrity import result, ACCOUNT
text = st.text_area("Candidate", value=ACCOUNT, key="candidate", on_change=pilot.invalidate_exports, args=("draft",))
confirmed = render_factual_review(result(), text, slot="draft") and pilot.confirm_export(text)
render_text_export(text, slot="draft", confirmed=confirmed)
'''
        at = AppTest.from_string(code, default_timeout=20).run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.button), 0)
        at.checkbox[0].check().run()
        at.checkbox[1].check().run()
        self.assertFalse(at.exception)
        at.button[0].click().run()
        self.assertFalse(at.exception)
        handle = at.session_state["_text_export_handle_draft"]
        self.assertEqual(self.get(handle=handle).status_code, 200)
        at.text_area[0].set_value("I observed knee pain in 1995.").run()
        self.assertFalse(at.exception)
        self.refused(self.get(handle=handle))
        from tests.test_factual_integrity import ACCOUNT
        at.text_area[0].set_value(ACCOUNT).run()
        self.assertFalse(at.exception)
        self.assertFalse(at.checkbox[0].value)
        self.refused(self.get(handle=handle))


if __name__ == "__main__":
    unittest.main()
