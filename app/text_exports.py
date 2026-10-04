"""Bounded, session/review-bound TXT artifacts; no disk, UI or network at import."""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
import time
import tomllib
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from . import pilot

MAX_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 8 * MAX_BYTES
MAX_FILES = 64
TTL_SECONDS = 300


class ExportUnavailable(ValueError):
    """A fixed refusal; no case, handle, cookie or exception detail."""


def policy_binding(approval: dict[str, Any]) -> str:
    """Explicit activation needs exact-release evidence and the fixed scope."""
    if not pilot.enabled() or os.getenv("VA_LSE_PILOT_TEXT_EXPORTS", "0") != "1":
        raise ExportUnavailable("Reviewed text downloads are unavailable.")
    policy = approval.get("text_export_policy")
    if (not isinstance(policy, dict) or set(policy) != {"enabled", "formats", "evidence_reference"}
            or policy["enabled"] is not True or policy["formats"] != ["txt"]
            or not isinstance(policy["evidence_reference"], str)
            or not policy["evidence_reference"].strip()):
        raise ExportUnavailable("Reviewed text downloads are unavailable.")
    url = urlsplit(pilot.https_url(approval["deployment_url"]))
    if url.path not in ("", "/"):
        raise ExportUnavailable("Reviewed text downloads require the reviewed root origin.")
    try:
        # The repository's hardening guard deliberately forbids ambient option
        # reads. Inspect the reviewed source configuration, independent of cwd.
        server = tomllib.loads((Path(__file__).resolve().parent.parent / ".streamlit/config.toml").read_text()).get("server", {})
        if not isinstance(server, dict) or server.get("baseUrlPath", "") != "":
            raise ValueError
    except (OSError, ValueError) as exc:
        raise ExportUnavailable("Reviewed text downloads require the reviewed root origin.") from exc
    return hashlib.sha256(json.dumps(approval, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ExportLease:
    owner: str
    slot: str
    scope: str
    text_hash: str
    approval_binding: str
    consent: pilot.ConsentGrant
    revoked: threading.Event = field(default_factory=threading.Event, compare=False)


@dataclass(frozen=True)
class _Artifact:
    lease: weakref.ReferenceType[ExportLease]
    content: bytes
    expires: float


class ExportStore:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._files: dict[str, _Artifact] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _purge(self) -> None:
        now = self._clock()
        for handle, artifact in list(self._files.items()):
            lease = artifact.lease()
            if (artifact.expires <= now or lease is None or lease.revoked.is_set()
                    or lease.consent.revoked.is_set()):
                del self._files[handle]

    def create(self, lease: ExportLease, text: str) -> str:
        from .factual_integrity import fingerprint
        if (not isinstance(text, str) or not text.strip() or len(text) > MAX_BYTES
                or re.search(r"\[(?:Confirm|Add if applicable|TODO)\b", text, re.IGNORECASE)
                or lease.text_hash != fingerprint(text) or not lease.owner
                or lease.slot not in ("draft", "eval") or not re.fullmatch(r"[0-9a-f]{64}", lease.scope)
                or lease.revoked.is_set() or lease.consent.revoked.is_set()):
            raise ExportUnavailable("Reviewed text downloads are unavailable.")
        content = text.encode("utf-8")
        if len(content) > MAX_BYTES:
            raise ExportUnavailable("The reviewed text exceeds the export limit.")
        with self._lock:
            self._purge()
            # Reuse the same reviewed bytes without extending their lifetime.
            for handle, artifact in self._files.items():
                if artifact.lease() is lease and artifact.content == content:
                    return handle
            if len(self._files) >= MAX_FILES or sum(len(a.content) for a in self._files.values()) + len(content) > MAX_TOTAL_BYTES:
                raise ExportUnavailable("The private export limit is reached. Try again after expiry.")
            handle = secrets.token_hex(32)
            self._files[handle] = _Artifact(weakref.ref(lease), content, self._clock() + TTL_SECONDS)
            return handle

    def read(self, handle: str, owner: str, approval: dict[str, Any]) -> bytes:
        binding = policy_binding(approval)
        with self._lock:
            self._purge()
            artifact = self._files.get(handle)
            lease = artifact.lease() if artifact else None
            if (artifact is None or lease is None or owner != lease.owner
                    or binding != lease.approval_binding
                    or lease.consent.binding != pilot.notice_binding(owner, approval)
                    or lease.revoked.is_set() or lease.consent.revoked.is_set()):
                raise ExportUnavailable("Download unavailable.")
            return artifact.content

    def revoke(self, lease: ExportLease) -> None:
        lease.revoked.set()
        with self._lock:
            self._purge()

    def sweep(self) -> dict[str, int]:
        with self._lock:
            self._purge()
            return {"files": len(self._files), "bytes": sum(len(a.content) for a in self._files.values())}

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            def clean() -> None:
                while not self._stop.wait(1):
                    self.sweep()
            self._thread = threading.Thread(target=clean, name="private-text-export-expiry", daemon=True)
            self._thread.start()

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2)
        with self._lock:
            self._files.clear()


# Allocation has no content and starts no thread; only the pilot launcher starts cleanup.
STORE = ExportStore()


def invalidate(slot: str | None = None) -> None:
    if not pilot.enabled():
        return
    import streamlit as st
    for name in ("draft", "eval") if slot is None else (slot,):
        lease = st.session_state.pop("_text_export_lease_" + name, None)
        if isinstance(lease, ExportLease):
            STORE.revoke(lease)
        st.session_state.pop("_text_export_handle_" + name, None)
