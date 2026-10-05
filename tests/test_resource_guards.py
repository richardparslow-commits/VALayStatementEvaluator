"""Invented upload/body/worker boundary regressions; no external services."""
from __future__ import annotations

import asyncio
import gc
import io
import json
import os
import subprocess
import sys
import threading
import time
import tracemalloc
import unittest
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from tests import hermetic  # noqa: F401
from tests.test_controlled_pilot import approval
from app import pilot, pipeline_guard as guard, upload_admission as uploads
from app.child_process import run_bounded
from app.documents import ChunkPlan, DocumentPage, iter_page_labelled_chunks
from streamlit.runtime.memory_uploaded_file_manager import MemoryUploadedFileManager
from streamlit.runtime.uploaded_file_manager import UploadedFileRec
from streamlit.web.server.starlette.starlette_app_utils import create_signed_value, generate_xsrf_token_string
from app.private_uploads import create_upload_routes
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.testclient import TestClient

SECRET = "INVENTED_ONLY_UPLOAD_COOKIE_" * 3
ORIGIN = "https://pilot.example.test"


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {"VA_LSE_MODE": "controlled-pilot"}))
        self.data = approval()
        self.stack.enter_context(patch.object(pilot, "load_approval", return_value=self.data))
        import streamlit as st
        self.stack.enter_context(patch.object(st, "secrets", {"auth": {
            "cookie_secret": SECRET, "redirect_uri": ORIGIN + "/oauth2callback",
            "server_metadata_url": self.data["issuer"] + "/.well-known/openid-configuration"}}))
        self.stack.enter_context(patch("app.shutdown.is_shutting_down", return_value=False))
        self.stack.enter_context(patch.object(guard, "check_memory_before_run"))
        self.registry = uploads.UploadRegistry()
        self.stack.enter_context(patch.object(uploads, "UPLOADS", self.registry))
        self.manager = MemoryUploadedFileManager("/_stcore/upload_file")
        self.current = MagicMock(uploaded_file_mgr=self.manager)
        self.current.is_active_session.return_value = True
        self.stack.enter_context(patch.object(uploads, "runtime", return_value=self.current))
        self.claims = self.identity()
        self.owner = pilot.authorized_identity(self.claims, self.data)
        self.grant = pilot.ConsentGrant(pilot.notice_binding(self.owner, self.data))
        self.registry.bind("invented-session", self.owner, self.grant)
        self.token = generate_xsrf_token_string()
        self.client = TestClient(Starlette(
            routes=create_upload_routes(self.current, self.manager, None),
            middleware=[Middleware(uploads.PilotUploadMiddleware)]), base_url=ORIGIN)
        self.addCleanup(self.client.close)

    def identity(self, subject="participant", **changes):
        now = time.time()
        return {"origin": ORIGIN, "is_logged_in": True, "iss": self.data["issuer"],
                "sub": subject, "iat": now - 1, "exp": now + 300, **changes}

    def headers(self, claims=None):
        signed = create_signed_value(SECRET, "_streamlit_user", json.dumps(claims or self.claims)).decode()
        return {"Cookie": "_streamlit_user=" + signed + "; _streamlit_xsrf=" + self.token,
                "X-Xsrftoken": self.token, "Origin": ORIGIN}

    def put(self, file="invented-file", **kw):
        return self.client.put("/_stcore/upload_file/invented-session/" + file,
                               files={"file": ("invented.txt", b"Invented note.", "text/plain")},
                               headers=kw.pop("headers", self.headers()), **kw)

    def test_real_handler_accepts_owned_signed_consent_and_preserves_xsrf(self):
        self.assertEqual(self.put().status_code, 204)
        self.assertEqual(self.manager.get_files("invented-session", ["invented-file"])[0].data, b"Invented note.")
        self.assertEqual(self.put("second", headers={**self.headers(), "X-Xsrftoken": "bad"}).status_code, 403)
        self.assertEqual(len(self.registry._files), 1)
        self.assertFalse(self.registry._pending)

    def test_no_cookie_foreign_owner_expired_or_revoked_upload_is_refused(self):
        for headers in ({}, self.headers(self.identity("operator")), self.headers(self.identity(exp=time.time()-1)),
                        {**self.headers(), "Origin": "https://other.example.test"}):
            with self.subTest(headers=list(headers)):
                self.assertEqual(self.put(headers=headers).status_code, 403)
        self.grant.revoked.set()
        self.assertEqual(self.put().status_code, 403)
        self.assertEqual(self.manager._total_bytes, 0)

    def test_refusal_does_not_receive_the_body(self):
        received = 0
        messages = []
        async def receive():
            nonlocal received
            received += 1
            raise AssertionError("An unauthenticated body was read.")
        async def send(message):
            messages.append(message)
        scope = {"type": "http", "path": "/_stcore/upload_file/invented-session/invented-file",
                 "method": "PUT", "headers": [], "scheme": "https", "query_string": b"",
                 "server": ("pilot.example.test", 443)}
        middleware = uploads.PilotUploadMiddleware(MagicMock())
        asyncio.run(middleware(scope, receive, send))
        self.assertEqual(received, 0)
        self.assertEqual(messages[0]["status"], 403)

    def test_plain_runtime_cannot_admit_pilot_uploads(self):
        context = MagicMock(session_id="invented-session")
        with patch.object(uploads, "_GUARDED_RUNTIMES", uploads.weakref.WeakSet()), \
                patch("streamlit.runtime.scriptrunner_utils.script_run_context.get_script_run_ctx", return_value=context):
            with self.assertRaises(pilot.PilotBlocked):
                uploads.bind_session(self.owner, self.grant)

    def test_prefixed_upload_route_cannot_bypass_the_guard(self):
        self.assertEqual(self.client.put("/prefix/_stcore/upload_file/invented-session/new", headers=self.headers(), content=b"x").status_code, 403)

    def test_owner_cannot_delete_another_sessions_file(self):
        self.assertEqual(self.put().status_code, 204)
        path = "/_stcore/upload_file/invented-session/invented-file"
        self.assertEqual(self.client.delete(path, headers=self.headers(self.identity("operator"))).status_code, 403)
        self.assertEqual(self.manager._file_count, 1)
        self.assertEqual(self.client.delete(path, headers=self.headers()).status_code, 204)
        self.assertEqual(self.manager._file_count, 0)

    def test_transfer_has_an_absolute_deadline_and_no_orphan_reservation(self):
        async def endpoint(scope, receive, send):
            await receive()
        async def receive():
            await asyncio.sleep(1)
            return {"type": "http.request", "body": b"", "more_body": False}
        messages = []
        async def send(message):
            messages.append(message)
        scope = {"type": "http", "path": "/_stcore/upload_file/invented-session/new",
                 "method": "PUT", "headers": [(k.lower().encode(), v.encode()) for k,v in self.headers().items()]
                 + [(b"host", b"pilot.example.test")], "scheme": "https", "query_string": b""}
        with patch.object(uploads, "TRANSFER_SECONDS", .02):
            asyncio.run(uploads.PilotUploadMiddleware(endpoint)(scope, receive, send))
        self.assertEqual(messages[0]["status"], 408)
        self.assertFalse(self.registry._pending)

    def test_transfer_cannot_reset_deadline_by_trickling(self):
        async def endpoint(scope, receive, send):
            while True:
                await receive()
        async def receive():
            await asyncio.sleep(.005)
            return {"type": "http.request", "body": b"x", "more_body": True}
        messages = []
        async def send(message):
            messages.append(message)
        scope = {"type": "http", "path": "/_stcore/upload_file/invented-session/new",
                 "method": "PUT", "headers": [(k.lower().encode(), v.encode()) for k,v in self.headers().items()]
                 + [(b"host", b"pilot.example.test")], "scheme": "https", "query_string": b""}
        with patch.object(uploads, "TRANSFER_SECONDS", .03):
            asyncio.run(uploads.PilotUploadMiddleware(endpoint)(scope, receive, send))
        self.assertEqual(messages[0]["status"], 408)
        self.assertFalse(self.registry._pending)

    def test_reservations_enforce_owner_global_count_and_transfer_limits(self):
        from app import config
        with patch.object(config, "MAX_UPLOAD_BYTES", 16), patch.object(config, "MAX_TOTAL_UPLOAD_BYTES", 32):
            first = self.registry.reserve("invented-session", "one", self.owner, self.data)
            with self.assertRaises(uploads.UploadRefused):
                self.registry.reserve("invented-session", "two", self.owner, self.data)
            self.manager.add_file("invented-session", UploadedFileRec("one", "one.txt", "text/plain", b"x" * 16))
            self.registry.finish(first, self.manager, True)
            second = self.registry.reserve("invented-session", "two", self.owner, self.data)
            self.manager.add_file("invented-session", UploadedFileRec("two", "two.txt", "text/plain", b"x" * 16))
            self.registry.finish(second, self.manager, True)
            with self.assertRaises(uploads.UploadRefused):
                self.registry.reserve("invented-session", "three", self.owner, self.data)
            self.manager.remove_file("invented-session", "one")
            self.registry.sweep(self.manager, lambda _: True)
            with patch.object(uploads, "MAX_FILES", 1), self.assertRaises(uploads.UploadRefused):
                self.registry.reserve("invented-session", "three", self.owner, self.data)
            with patch.object(uploads, "MAX_PROCESS_BYTES", 20), self.assertRaises(uploads.UploadRefused):
                self.registry.reserve("invented-session", "three", self.owner, self.data)

    def test_oversized_actual_file_is_removed_before_reservation_release(self):
        from app import config
        with patch.object(config, "MAX_UPLOAD_BYTES", 4):
            lease = self.registry.reserve("invented-session", "one", self.owner, self.data)
            self.manager.add_file("invented-session", UploadedFileRec("one", "one.txt", "text/plain", b"x" * 5))
            with self.assertRaises(uploads.UploadRefused):
                self.registry.finish(lease, self.manager, True)
        self.assertEqual(self.manager._total_bytes, 0)
        self.assertFalse(self.registry._pending)

    def test_source_text_has_owner_and_process_budgets_until_grant_is_released(self):
        with patch.object(uploads, "MAX_OWNER_TEXT", 5), patch.object(uploads, "MAX_PROCESS_TEXT", 6):
            self.registry.claim_text("invented-session", "one", self.owner, self.data, 4)
            self.registry.claim_text("invented-session", "one", self.owner, self.data, 4)
            with self.assertRaises(pilot.PilotBlocked):
                self.registry.claim_text("invented-session", "two", self.owner, self.data, 2)
            other = pilot.authorized_identity(self.identity("operator"), self.data)
            grant = pilot.ConsentGrant(pilot.notice_binding(other, self.data))
            self.registry.bind("other", other, grant)
            self.grant.revoked.set()
            self.registry.revoke("invented-session")
            with self.assertRaises(pilot.PilotBlocked):
                self.registry.claim_text("other", "two", other, self.data, 3)
            self.grant = None
            gc.collect()
            self.registry.claim_text("other", "two", other, self.data, 3)

    def test_empty_text_entries_cannot_exhaust_registry_metadata(self):
        with patch.object(uploads, "MAX_FILES", 2):
            self.registry.claim_text("invented-session", "one", self.owner, self.data, 0)
            self.registry.claim_text("invented-session", "two", self.owner, self.data, 0)
            with self.assertRaises(pilot.PilotBlocked):
                self.registry.claim_text("invented-session", "three", self.owner, self.data, 0)

    def test_participant_upload_rate_survives_case_clear_and_expires(self):
        with patch.object(uploads, "MAX_OWNER_PUTS_PER_MINUTE", 1), \
                patch.object(uploads, "MAX_OWNER_PUTS_PER_HOUR", 2), \
                patch.object(uploads.time, "monotonic", return_value=1000) as clock:
            first = self.registry.reserve("invented-session", "one", self.owner, self.data)
            self.registry.finish(first, self.manager, False)
            self.registry.revoke("invented-session")
            self.registry.bind("invented-session", self.owner, self.grant)
            with self.assertRaises(uploads.UploadRefused) as error:
                self.registry.reserve("invented-session", "two", self.owner, self.data)
            self.assertEqual(error.exception.status, 429)
            clock.return_value = 1061
            second = self.registry.reserve("invented-session", "two", self.owner, self.data)
            self.registry.finish(second, self.manager, False)
            clock.return_value = 1122
            with self.assertRaises(uploads.UploadRefused):
                self.registry.reserve("invented-session", "three", self.owner, self.data)
            clock.return_value = 4601
            third = self.registry.reserve("invented-session", "three", self.owner, self.data)
            self.registry.finish(third, self.manager, False)

    def test_global_transfer_quota_is_reserved_before_body_receipt(self):
        other = pilot.authorized_identity(self.identity("operator"), self.data)
        grant = pilot.ConsentGrant(pilot.notice_binding(other, self.data))
        self.registry.bind("other-session", other, grant)
        first = self.registry.reserve("invented-session", "one", self.owner, self.data)
        second = self.registry.reserve("other-session", "two", other, self.data)
        self.registry.bind("third-session", other, grant)
        with self.assertRaises(uploads.UploadRefused) as error:
            self.registry.reserve("third-session", "three", other, self.data)
        self.assertEqual(error.exception.status, 429)
        self.registry.finish(first, self.manager, False)
        self.registry.finish(second, self.manager, False)

    def test_withdrawal_and_body_overflow_remove_already_written_files(self):
        from app import config
        for reason in ("withdrawal", "overflow"):
            self.grant.revoked.clear()
            self.registry.bind("invented-session", self.owner, self.grant)
            messages = []
            async def endpoint(scope, receive, send):
                self.manager.add_file("invented-session", UploadedFileRec("new", "new.txt", "text/plain", b"x"))
                await receive()
                await send({"type": "http.response.start", "status": 204, "headers": []})
                await send({"type": "http.response.body", "body": b""})
            async def receive():
                if reason == "withdrawal":
                    self.grant.revoked.set()
                return {"type": "http.request", "body": b"x" * (uploads.MIB + 5), "more_body": False}
            async def send(message):
                messages.append(message)
            scope = {"type": "http", "path": "/_stcore/upload_file/invented-session/new",
                     "method": "PUT", "headers": [(k.lower().encode(), v.encode()) for k,v in self.headers().items()]
                     + [(b"host", b"pilot.example.test")], "scheme": "https", "query_string": b""}
            with self.subTest(reason=reason), patch.object(config, "MAX_UPLOAD_BYTES", 4):
                # Withdrawal is rechecked at response commit; overflow at receive.
                async def small_receive():
                    self.grant.revoked.set()
                    return {"type": "http.request", "body": b"x", "more_body": False}
                asyncio.run(uploads.PilotUploadMiddleware(endpoint)(scope,
                            small_receive if reason == "withdrawal" else receive, send))
                self.assertEqual(messages[0]["status"], 403 if reason == "withdrawal" else 413)
                self.assertEqual(self.manager._total_bytes, 0)
                self.assertFalse(self.registry._pending)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        memory = patch.object(guard, "_read_available_memory_mb", return_value=8000)
        memory.start()
        self.addCleanup(memory.stop)

    def test_timed_out_worker_keeps_capacity_until_it_exits(self):
        release = threading.Event()
        finished = threading.Event()
        grant = pilot.ConsentGrant("invented")
        def work():
            try:
                release.wait(2)
            finally:
                finished.set()
        with patch.object(pilot, "enabled", return_value=True), patch.object(pilot, "current_owner", return_value="invented-owner"), patch.object(pilot, "require_consent", return_value=grant):
            try:
                with self.assertRaises(guard.PipelineTimeoutError):
                    guard.run_with_timeout(work, timeout_seconds=.02)
                with self.assertRaises(pilot.PilotBlocked):
                    guard.run_with_timeout(lambda: None, timeout_seconds=1)
            finally:
                release.set()
                self.assertTrue(finished.wait(2))
                for _ in range(100):
                    with guard._jobs_lock:
                        if "invented-owner" not in guard._jobs:
                            break
                    time.sleep(.005)
            guard.run_with_timeout(lambda: None, timeout_seconds=1)

    def test_child_future_keeps_reservation_after_parent_returns(self):
        child = Future()
        grant = pilot.ConsentGrant("invented")
        with patch.object(pilot, "enabled", return_value=True), patch.object(pilot, "current_owner", return_value="invented-child-owner"), patch.object(pilot, "require_consent", return_value=grant):
            try:
                guard.run_with_timeout(lambda: guard.track_pipeline_future(child), timeout_seconds=1)
                with self.assertRaises(pilot.PilotBlocked):
                    guard.require_work_capacity("invented-child-owner")
            finally:
                child.set_result(None)
            guard.require_work_capacity("invented-child-owner")

    def test_container_headroom_is_the_stricter_memory_figure(self):
        with patch.object(guard, "_read_available_memory_mb", wraps=_REAL_MEMORY_READER), patch.object(guard, "_read_host_available_memory_mb", return_value=8000), patch.object(guard, "_read_cgroup_available_memory_mb", return_value=100):
            self.assertEqual(guard._read_available_memory_mb(), 100)
            with self.assertRaises(MemoryError):
                guard.check_memory_before_run()

    def test_cgroup_v2_and_v1_available_bytes_are_parsed(self):
        for values in ({"/sys/fs/cgroup/memory.max": "104857600", "/sys/fs/cgroup/memory.current": "52428800"},
                       {"/sys/fs/cgroup/memory.max": "max", "/sys/fs/cgroup/memory/memory.limit_in_bytes": "104857600",
                        "/sys/fs/cgroup/memory/memory.usage_in_bytes": "157286400"}):
            def read(path):
                if str(path) not in values:
                    raise FileNotFoundError
                return values[str(path)]
            with self.subTest(values=values), patch.object(guard.Path, "read_text", read):
                self.assertEqual(guard._read_cgroup_available_memory_mb(), 50 if len(values) == 2 else 0)

    def test_executor_setup_failure_releases_the_owner_lease(self):
        with patch.object(pilot, "enabled", return_value=True), patch.object(pilot, "current_owner", return_value="setup-owner"), \
                patch.object(pilot, "require_consent", return_value=pilot.ConsentGrant("invented")), \
                patch.object(guard.concurrent.futures, "ThreadPoolExecutor", side_effect=RuntimeError("synthetic setup failure")):
            with self.assertRaises(RuntimeError):
                guard.run_with_timeout(lambda: None, timeout_seconds=1)
            guard.require_work_capacity("setup-owner")

    def test_bounded_futures_do_not_consume_the_whole_input(self):
        seen = []
        release = threading.Event()
        def items():
            for i in range(100):
                seen.append(i)
                yield i
        with ThreadPoolExecutor(max_workers=1) as pool:
            iterator = guard.bounded_pipeline_futures(pool, lambda i: (release.wait(.02), i)[1], items(), 2)
            first, value = next(iterator)
            self.assertEqual(len(seen), 2)
            release.set()
            rest = [f.result() for f, _ in iterator]
        self.assertEqual(sorted([first.result(), *rest]), list(range(100)))

    def test_chunk_plan_preserves_chunks_with_small_working_memory(self):
        pages = [DocumentPage("invented.pdf", n+1, ("Invented finding. " * 300)) for n in range(300)]
        reference = list(iter_page_labelled_chunks(pages))
        tracemalloc.start()
        count = 0
        for expected, actual in zip(reference, ChunkPlan(pages), strict=True):
            self.assertEqual(expected, actual)
            count += 1
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertEqual(count, len(reference))
        self.assertLess(peak, sum(len(c.text) for c in reference) // 2)

    def test_local_tool_deadline_and_output_cap(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            run_bounded([sys.executable, "-c", "import time; time.sleep(5)"], timeout=.05)
        with self.assertRaisesRegex(RuntimeError, "output"):
            run_bounded([sys.executable, "-c", "print('x'*100000)"], max_output=128)
        result = run_bounded([sys.executable, "-c", "print('invented')"])
        self.assertEqual((result.returncode, result.stdout), (0, "invented\n"))

    def test_exited_tool_parent_cannot_leave_descendants_holding_output_open(self):
        command = [sys.executable, "-c", "import os,time; child=os.fork(); time.sleep(5) if child == 0 else None"]
        start = time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired):
            run_bounded(command, timeout=.1)
        self.assertLess(time.monotonic() - start, 1)


_REAL_MEMORY_READER = guard._read_available_memory_mb


class UiResourceTests(unittest.TestCase):
    def test_pilot_rejects_entire_partial_batch_before_extraction(self):
        from app.views import records
        with patch.object(pilot, "enabled", return_value=True), patch.object(records, "_track_selector_impression"), \
                patch.object(records.st, "radio", return_value="Upload files"), patch.object(records.st, "checkbox", return_value=True), \
                patch.object(records.st, "file_uploader", return_value=[object(), object()]), \
                patch.object(records, "check_upload_limits", return_value=([object()], ["Invented size refusal"])), \
                patch.object(records, "extract_uploads") as extract, patch.object(pilot, "display"), patch.object(records, "report_failure"):
            self.assertEqual(records.records_uploader("invented"), [])
        extract.assert_not_called()

    def test_invalid_upload_sizes_are_rejected(self):
        from app.views.uploads import check_upload_limits
        for size in (-1, True, "1"):
            with self.subTest(size=size):
                accepted, refusals = check_upload_limits([MagicMock(size=size)])
                self.assertEqual(accepted, [])
                self.assertEqual(len(refusals), 1)

    def test_single_oversized_paragraph_index_is_not_retained(self):
        from app import documents
        paragraphs = [MagicMock(text="Invented paragraph")]
        with patch.object(documents, "_PARAGRAPH_CACHE", documents.OrderedDict()), \
                patch.object(documents, "_PARAGRAPH_CACHE_MAX_CHARS", 3):
            documents._cache_put(("invented", 1, 1), paragraphs)
            self.assertFalse(documents._PARAGRAPH_CACHE)

    def test_pilot_matching_preserves_results_without_shared_source_caches(self):
        from app import documents, medical_review
        text = "Invented veteran reported cervical pain and neck stiffness after walking."
        doc = documents.document_from_text("invented.txt", text)
        medical_review._tokens_cached.cache_clear()
        self.addCleanup(medical_review._tokens_cached.cache_clear)
        with patch.object(documents, "_PARAGRAPH_CACHE", documents.OrderedDict()):
            with patch.object(pilot, "enabled", return_value=False):
                expected_tokens = medical_review._tokens(text)
                expected_paragraphs = documents.paragraph_index(doc)
            medical_review._tokens_cached.cache_clear()
            documents._PARAGRAPH_CACHE.clear()
            with patch.object(pilot, "enabled", return_value=True):
                self.assertEqual(medical_review._tokens(text), expected_tokens)
                self.assertEqual(documents.paragraph_index(doc), expected_paragraphs)
                self.assertFalse(documents._PARAGRAPH_CACHE)
                self.assertEqual(medical_review._tokens_cached.cache_info().currsize, 0)

    def test_archive_members_are_parsed_before_the_next_body_is_expanded(self):
        from app import documents
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("one.txt", "Invented first note")
            archive.writestr("two.txt", "Invented second note")
        opened = []
        original_open = zipfile.ZipFile.open
        def record_open(archive, info, *args, **kwargs):
            opened.append(info.filename)
            return original_open(archive, info, *args, **kwargs)
        def parse(label, data):
            expected = ["one.txt"] if label.endswith("one.txt") else ["one.txt", "two.txt"]
            self.assertEqual(opened, expected)
            return documents.document_from_text(label, data.decode())
        with patch.object(zipfile.ZipFile, "open", record_open), patch.object(documents, "extract_document", side_effect=parse):
            parsed, skipped = documents.InProcessExtractor().extract("invented.zip", buffer.getvalue())
        self.assertEqual(len(parsed), 2)
        self.assertFalse(skipped)

    def test_forged_archive_metadata_cannot_bypass_actual_expansion_limits(self):
        from app import config, documents
        info = zipfile.ZipInfo("invented.txt")
        info.file_size = info.compress_size = 5
        archive = MagicMock()
        archive.__enter__.return_value = archive
        archive.infolist.return_value = [info]
        archive.open.return_value = io.BytesIO(b"x" * 100)
        skipped = []
        with patch.object(zipfile, "ZipFile", return_value=archive), \
                patch.object(config, "ZIP_MAX_MEMBER_BYTES", 8), patch.object(config, "ZIP_MAX_TOTAL_UNCOMPRESSED_BYTES", 16):
            self.assertEqual(list(documents.iter_archive_members("invented.zip", b"PK\x03\x04", skipped)), [])
        self.assertEqual(len(skipped), 1)
        self.assertIn("actual expansion", skipped[0])
