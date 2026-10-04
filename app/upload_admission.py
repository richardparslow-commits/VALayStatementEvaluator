"""Pre-body pilot upload ownership and bounded, process-local reservations.

The pilot is single-process. No document bytes, labels, or cookies enter this
registry. Streamlit owns the bytes; its public file manager owns their removal.
"""
from __future__ import annotations

import asyncio
import re
import threading
import time
import weakref
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlsplit

from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import pilot

MIB = 1024 * 1024
MAX_FILES = 32
MAX_SESSIONS = 20
MAX_PROCESS_BYTES = 512 * MIB
MAX_OWNER_TEXT = 20 * MIB
MAX_PROCESS_TEXT = 80 * MIB
MAX_TRANSFERS = 2
MAX_OWNER_PUTS_PER_MINUTE = 32
MAX_OWNER_PUTS_PER_HOUR = 128
TRANSFER_SECONDS = 60
_ROUTE = re.compile(r"^/_stcore/upload_file/([^/]{1,128})/([^/]{1,128})$")
HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}


class UploadRefused(Exception):
    def __init__(self, status: int = 403) -> None:
        self.status = status
        super().__init__("Upload unavailable.")


@dataclass(frozen=True)
class Binding:
    owner: str
    consent: weakref.ReferenceType[pilot.ConsentGrant]


@dataclass(frozen=True)
class Reservation:
    key: tuple[str, str]
    binding: Binding
    maximum: int


class UploadRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._bindings: dict[str, Binding] = {}
        self._files: dict[tuple[str, str], tuple[str, int]] = {}
        self._pending: dict[tuple[str, str], Reservation] = {}
        self._text: dict[tuple[str, str], tuple[weakref.ReferenceType[pilot.ConsentGrant], str, int]] = {}
        self._attempts: dict[str, deque[float]] = {}

    def bind(self, session: str, owner: str, grant: pilot.ConsentGrant) -> None:
        with self._lock:
            old = self._bindings.get(session)
            if old and (old.owner != owner or old.consent() is not grant):
                raise pilot.PilotBlocked("Clear this case before changing upload ownership.")
            if old is not None:
                return
            if session not in self._bindings and len(self._bindings) >= MAX_SESSIONS:
                raise pilot.PilotBlocked("Upload sessions are at capacity. Clear an unused case.")
            self._bindings[session] = Binding(owner, weakref.ref(grant))

    def revoke(self, session: str) -> None:
        # Pending requests retain their reservations until their finally block.
        with self._lock:
            self._bindings.pop(session, None)
            for key in list(self._files):
                if key[0] == session:
                    del self._files[key]

    def _binding(self, session: str, owner: str, approval: dict[str, Any]) -> Binding:
        binding = self._bindings.get(session)
        grant = binding.consent() if binding else None
        if (binding is None or binding.owner != owner or grant is None
                or grant.revoked.is_set()
                or grant.binding != pilot.notice_binding(owner, approval)):
            raise UploadRefused()
        return binding

    def sweep(self, manager: Any, active: Callable[[str], bool]) -> None:
        with self._lock:
            for session, binding in list(self._bindings.items()):
                grant = binding.consent()
                if grant is None or grant.revoked.is_set() or not active(session):
                    manager.remove_session_files(session)
                    self.revoke(session)
            for key in list(self._files):
                if not manager.get_files(key[0], [key[1]]):
                    del self._files[key]

    def claim_text(self, session: str, key: str, owner: str, approval: dict[str, Any], chars: int) -> None:
        with self._lock:
            for token, (ref, _, _) in list(self._text.items()):
                if ref() is None:
                    del self._text[token]
            binding = self._binding(session, owner, approval)
            grant = binding.consent()
            assert grant is not None
            token = (session, key)
            previous = self._text.get(token)
            if previous and (previous[0]() is not grant or previous[2] != chars):
                raise pilot.PilotBlocked("Clear the case before replacing retained source text.")
            if previous:
                return
            owned = sum(n for _, o, n in self._text.values() if o == owner)
            total = sum(n for _, _, n in self._text.values())
            count = sum(o == owner for _, o, _ in self._text.values())
            if (chars < 0 or owned + chars > MAX_OWNER_TEXT or total + chars > MAX_PROCESS_TEXT
                    or count >= MAX_FILES or len(self._text) >= MAX_FILES * MAX_SESSIONS):
                raise pilot.PilotBlocked("The retained record text is at capacity. Clear an unused case or select fewer records.")
            self._text[token] = (weakref.ref(grant), owner, chars)

    def reserve(self, session: str, file: str, owner: str,
                approval: dict[str, Any]) -> Reservation:
        from . import config
        from .pipeline_guard import check_memory_before_run, require_work_capacity
        with self._lock:
            binding = self._binding(session, owner, approval)
            now = time.monotonic()
            for identity, history in list(self._attempts.items()):
                while history and history[0] <= now - 3600:
                    history.popleft()
                if not history:
                    del self._attempts[identity]
            if owner not in self._attempts and len(self._attempts) >= MAX_SESSIONS:
                raise UploadRefused(429)
            history = self._attempts.setdefault(owner, deque())
            if (len(history) >= MAX_OWNER_PUTS_PER_HOUR
                    or sum(t > now - 60 for t in history) >= MAX_OWNER_PUTS_PER_MINUTE):
                raise UploadRefused(429)
            # Refused concurrent/size/capacity attempts count too. Clearing a
            # case must not reset this participant's upload rate envelope.
            history.append(now)
            key = (session, file)
            if (key in self._pending or len(self._pending) >= MAX_TRANSFERS
                    or any(r.binding.owner == owner for r in self._pending.values())):
                raise UploadRefused(429)
            # Count replacement's old bytes too until it is actually replaced.
            maximum = min(config.MAX_UPLOAD_BYTES, 50 * MIB)
            held = sum(n for _, n in self._files.values()) + sum(r.maximum for r in self._pending.values())
            owned = sum(n for o, n in self._files.values() if o == owner)
            owned += sum(r.maximum for r in self._pending.values() if r.binding.owner == owner)
            count = sum(o == owner for o, _ in self._files.values())
            count += sum(r.binding.owner == owner for r in self._pending.values())
            if (maximum <= 0 or owned + maximum > min(config.MAX_TOTAL_UPLOAD_BYTES, 200 * MIB)
                    or held + maximum > MAX_PROCESS_BYTES or count >= MAX_FILES):
                raise UploadRefused(413)
            try:
                require_work_capacity(owner)
                # Reserve conservative body + parsing/reply working headroom.
                check_memory_before_run(minimum_mb=384 * (len(self._pending) + 1))
            except (MemoryError, pilot.PilotBlocked):
                raise UploadRefused(503) from None
            reservation = Reservation(key, binding, maximum)
            self._pending[key] = reservation
            return reservation

    def check(self, reservation: Reservation, owner: str, approval: dict[str, Any]) -> None:
        with self._lock:
            if self._binding(reservation.key[0], owner, approval) is not reservation.binding:
                raise UploadRefused()

    def finish(self, reservation: Reservation, manager: Any, success: bool) -> None:
        with self._lock:
            session, file = reservation.key
            try:
                if success and self._bindings.get(session) is reservation.binding:
                    files = manager.get_files(session, [file])
                    if len(files) != 1 or len(files[0].data) > reservation.maximum:
                        manager.remove_file(session, file)
                        self._files.pop(reservation.key, None)
                        raise UploadRefused(413)
                    self._files[reservation.key] = (reservation.binding.owner, len(files[0].data))
                else:
                    manager.remove_file(session, file)
                    self._files.pop(reservation.key, None)
            finally:
                self._pending.pop(reservation.key, None)

    def delete(self, session: str, file: str, owner: str, approval: dict[str, Any],
               manager: Any) -> None:
        with self._lock:
            self._binding(session, owner, approval)
            if (session, file) in self._pending:
                raise UploadRefused(409)
            manager.remove_file(session, file)
            self._files.pop((session, file), None)


UPLOADS = UploadRegistry()
_GUARDED_RUNTIMES: weakref.WeakSet[Any] = weakref.WeakSet()


def runtime() -> Any:
    from streamlit.runtime import get_instance
    return get_instance()


def bind_session(owner: str, grant: pilot.ConsentGrant) -> None:
    from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
    context = get_script_run_ctx(suppress_warning=True)
    if context is not None:
        current = runtime()
        if current not in _GUARDED_RUNTIMES:
            raise pilot.PilotBlocked("The controlled-pilot upload service is unavailable. Contact the operator.")
        UPLOADS.sweep(current.uploaded_file_mgr, current.is_active_session)
        UPLOADS.bind(context.session_id, owner, grant)


def claim_documents(cache_key: str, documents: list[Any]) -> None:
    import hashlib
    from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
    context = get_script_run_ctx(suppress_warning=True)
    if context is None:
        raise pilot.PilotBlocked("A pilot session is required to retain uploaded records.")
    owner = pilot.current_owner()
    UPLOADS.claim_text(context.session_id, hashlib.sha256(cache_key.encode()).hexdigest(),
                      owner, pilot.load_approval(), sum(len(p.text) for d in documents for p in d.pages))


def identity(request: Request) -> tuple[str, dict[str, Any]]:
    from .export_cookie import claims
    from .shutdown import is_shutting_down
    if is_shutting_down():
        raise UploadRefused(503)
    approval = pilot.load_approval()
    parsed = urlsplit(pilot.https_url(approval["deployment_url"]))
    origin = "https://" + parsed.netloc
    if (parsed.path not in ("", "/") or request.url.query
            or request.headers.get("host") != parsed.netloc
            or request.headers.get("origin") != origin
            or request.headers.get("sec-fetch-site", "same-origin") != "same-origin"
            or (request.url.scheme != "https" and request.headers.get("x-forwarded-proto") != "https")):
        raise UploadRefused()
    import streamlit as st
    auth = st.secrets["auth"]
    if (auth["redirect_uri"] != origin + "/oauth2callback"
            or auth["server_metadata_url"] != pilot.https_url(approval["issuer"]) + "/.well-known/openid-configuration"):
        raise UploadRefused()
    owner = pilot.authorized_identity(claims(request.headers.getlist("cookie"), auth["cookie_secret"], origin), approval)
    return owner, approval


class PilotUploadMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not pilot.enabled():
            await self.app(scope, receive, send)
            return
        # Screen admission must prove this runtime has the HTTP guard installed;
        # a plain `streamlit run run_app.py` cannot open a pilot uploader.
        _GUARDED_RUNTIMES.add(runtime())
        if "/_stcore/upload_file" not in scope["path"]:
            await self.app(scope, receive, send)
            return
        reservation: Reservation | None = None
        manager: Any = None
        started = False
        success = False
        try:
            route = _ROUTE.fullmatch(scope["path"])
            if route is None or scope["method"] not in ("PUT", "DELETE", "OPTIONS"):
                raise UploadRefused()
            if scope["method"] == "OPTIONS":
                # Let Streamlit retain its origin/XSRF/CORS behavior.
                await self.app(scope, receive, send)
                return
            request = Request(scope)
            owner, approval = identity(request)
            current = runtime()
            manager = current.uploaded_file_mgr
            UPLOADS.sweep(manager, current.is_active_session)
            session, file = route.groups()
            if not current.is_active_session(session):
                raise UploadRefused()
            if scope["method"] == "DELETE":
                # Authorize first; leave actual deletion/XSRF checks to Streamlit.
                with UPLOADS._lock:
                    UPLOADS._binding(session, owner, approval)
                    if (session, file) in UPLOADS._pending:
                        raise UploadRefused(409)
                await self.app(scope, receive, send)
                UPLOADS.sweep(manager, current.is_active_session)
                return
            lease = UPLOADS.reserve(session, file, owner, approval)
            reservation = lease
            consumed = 0
            expires = time.monotonic() + TRANSFER_SECONDS

            async def bounded_receive() -> Message:
                nonlocal consumed
                fresh_owner, fresh_approval = identity(request)
                UPLOADS.check(lease, fresh_owner, fresh_approval)
                remaining = expires - time.monotonic()
                if remaining <= 0:
                    raise UploadRefused(408)
                message = await asyncio.wait_for(receive(), remaining)
                if message["type"] == "http.request":
                    consumed += len(message.get("body", b""))
                    if consumed > lease.maximum + MIB:
                        raise UploadRefused(413)
                return message

            async def checked_send(message: Message) -> None:
                nonlocal started, success
                if message["type"] == "http.response.start":
                    fresh_owner, fresh_approval = identity(request)
                    UPLOADS.check(lease, fresh_owner, fresh_approval)
                    success = message["status"] == 204
                    UPLOADS.finish(lease, manager, success)
                    started = True
                await send(message)

            async with asyncio.timeout(TRANSFER_SECONDS):
                await self.app(scope, bounded_receive, checked_send)
        except TimeoutError:
            if not started:
                await Response("Upload timed out. Try a smaller file.", status_code=408, headers=HEADERS)(scope, receive, send)
        except Exception as exc:
            if not started:
                code = exc.status if isinstance(exc, UploadRefused) else 403
                await Response("Upload unavailable. Refresh access or reduce the selected files.", status_code=code, headers=HEADERS)(scope, receive, send)
            else:
                raise
        finally:
            if reservation is not None and not started:
                UPLOADS.finish(reservation, manager, False)
