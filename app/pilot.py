"""Fail-closed admission for the single-instance, invite-only real-data pilot.

The manifest records operator attestations and their evidence references. It is
not a certification, a substitute for reviewing the evidence, or a secret.
Identity comes only from Streamlit's verified OIDC token, never a browser field.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import re
import socket
import stat
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping
from urllib.parse import urlsplit


class PilotBlocked(RuntimeError):
    """A pilot admission or privacy requirement has not been met."""


@dataclass
class ConsentGrant:
    binding: str
    revoked: threading.Event = field(default_factory=threading.Event)


_run_notice: ContextVar[ConsentGrant | None] = ContextVar("pilot_notice_consent", default=None)
_run_claims: ContextVar[dict[str, Any] | None] = ContextVar("pilot_verified_claims", default=None)


def enabled() -> bool:
    return os.getenv("VA_LSE_MODE", "synthetic").strip().lower() == "controlled-pilot"


EVIDENCE_FIELDS = (
    "provider_terms", "retention_policy", "privacy_review", "legal_review",
    "accuracy_validation", "deployment_validation", "incident_response", "spending_controls",
)


def https_url(value: Any) -> str:
    if not isinstance(value, str):
        raise PilotBlocked("The operator must configure an approved HTTPS destination.")
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment
            or parsed.port not in (None, 443)):
        raise PilotBlocked("The operator must configure an approved HTTPS destination.")
    return value.rstrip("/")


_APPROVAL_LIMIT = 65536


def _approval_bytes(path: Path) -> bytes:
    """Read a bounded, stable regular file without following the leaf link.

    A root-owned read-only mount can be readable by the non-root runtime. Host
    parent directories and replacement authority still require operator review.
    """
    if (not path.is_absolute() or not hasattr(os, "O_NOFOLLOW")
            or not hasattr(os, "O_NONBLOCK") or not hasattr(os, "geteuid")):
        raise ValueError
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid not in (0, os.geteuid())
                or before.st_nlink != 1 or before.st_mode & 0o022 or before.st_size > _APPROVAL_LIMIT):
            raise ValueError
        raw = bytearray()
        while len(raw) <= _APPROVAL_LIMIT:
            chunk = os.read(fd, min(8192, _APPROVAL_LIMIT + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(fd)
        current = path.lstat()
        def identity(value: os.stat_result) -> tuple[int, ...]:
            return (value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_nlink,
                    value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if (len(raw) > _APPROVAL_LIMIT or len(raw) != before.st_size
                or identity(before) != identity(after) or identity(after) != identity(current)):
            raise ValueError
        return bytes(raw)
    finally:
        os.close(fd)


def _approval_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _approval_constant(value: str) -> Any:
    raise ValueError


def _approval_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError
    return result


def _approval_time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError
    return result


def load_approval() -> dict[str, Any]:
    try:
        path = Path(os.environ["VA_LSE_PILOT_APPROVAL_FILE"])
        data = json.loads(_approval_bytes(path).decode("utf-8"), object_pairs_hook=_approval_pairs,
                          parse_constant=_approval_constant, parse_float=_approval_float)
        if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
                or data["schema_version"] != 1):
            raise ValueError
        now = datetime.now(timezone.utc)
        start = _approval_time(data["approved_at"])
        expiry = _approval_time(data["expires_at"])
        if not start <= now < expiry or (expiry - start).total_seconds() > 30 * 86400:
            raise ValueError
        for name in EVIDENCE_FIELDS:
            if not isinstance(data.get(name), str) or not data[name].strip():
                raise ValueError
        revision = os.getenv("VA_LSE_BUILD_SHA", "").strip()
        if not re.fullmatch(r"[0-9a-f]{40}", revision) or data.get("reviewed_revision") != revision:
            raise ValueError
        for name in ("deployment_url", "issuer", "provider_base_url"):
            https_url(data.get(name))
        for name in ("subjects", "operators", "models"):
            if (not isinstance(data.get(name), list) or not data[name]
                    or any(not isinstance(v, str) or not v or v != v.strip()
                           or any(ord(c) < 32 or ord(c) == 127 for c in v) for v in data[name])
                    or len(set(data[name])) != len(data[name])):
                raise ValueError
        if len(data["subjects"]) > 10 or not set(data["operators"]).issubset(data["subjects"]):
            raise ValueError
        if data.get("single_instance") is not True:
            raise ValueError
        notice = data.get("participant_notice")
        days = data.get("local_log_retention_days")
        if (not isinstance(notice, str) or not 80 <= len(notice.strip()) <= 16000
                or type(days) is not int or not 1 <= days <= 30):
            raise ValueError
        from .pilot_budget import policy
        policy(data)
        return data
    except (KeyError, ValueError, TypeError, OSError, OverflowError, RecursionError) as exc:
        raise PilotBlocked("Pilot admission is closed. The operator must supply current, "
                           "revision-specific review evidence and an invitation list.") from exc


def validate_log_policy(approval: Mapping[str, Any]) -> None:
    """Require all three local writers and a healthy matching age policy."""
    from .log_retention import retention_days, PilotLogHandler, retention_health
    try:
        if retention_days() != approval["local_log_retention_days"]:
            raise ValueError
        from .logging_config import configure_logging
        from .audit import configure_audit_logging
        from . import run_log
        handlers = [h for log in (configure_logging(), configure_audit_logging())
                    for h in log.handlers if isinstance(h, PilotLogHandler)]
        with run_log._LOCK:
            handlers.append(run_log.prepare_pilot_handler())
        if (len(handlers) != 3 or len({h.baseFilename for h in handlers}) != 3
                or any(h.days != retention_days() for h in handlers)):
            raise ValueError
        for handler in handlers:
            handler.verify_sink()
        health = retention_health()
        if not health["active"] or health["failed"] or health["stale"]:
            raise ValueError
    except (OSError, ValueError, KeyError) as exc:
        raise PilotBlocked("Pilot log retention must match the reviewed policy and be healthy.") from exc


def validate_configuration() -> dict[str, Any]:
    from . import config
    mode = os.getenv("VA_LSE_MODE", "synthetic").strip().lower()
    if mode not in ("synthetic", "controlled-pilot"):
        raise PilotBlocked("Unknown operating mode. Ask the operator to correct the configuration.")
    if not enabled():
        return {}
    approval = load_approval()
    import sys
    if not sys.platform.startswith("linux") or os.geteuid() == 0:
        raise PilotBlocked("The controlled pilot requires a non-root Linux runtime.")
    if (not 0 < config.MAX_RECORD_PAGES <= 500 or config.MAX_UPLOAD_BYTES > 50 * 1024 * 1024
            or config.MAX_TOTAL_UPLOAD_BYTES > 200 * 1024 * 1024):
        raise PilotBlocked("Pilot input limits require at most 500 pages, 50 MB per file, and 200 MB total.")
    from .isolated_extract import parser_health
    try:
        parser_health()
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise PilotBlocked("The reviewed protected parser service must be available before pilot admission.") from exc
    # This first pilot deliberately has no durable case storage or remote tools.
    # The disabled services need a separate privacy/ownership release review.
    if (config.JOB_QUEUE_ENABLED or config.BLOB_STORE_MODE != "none"
            or config.TRACING_ENABLED or config.SHARED_CACHE_URL or config.SHARED_CACHE_TOKEN
            or config.EXTRACTOR_MODE not in ("isolated", "in-process")):
        raise PilotBlocked("Pilot configuration requires no queue, no blob storage, "
                           "no tracing, and the protected parser service.")
    if any(os.getenv(key, "").strip() for key in (
        "INSPECT_API_KEY", "AGILOOP_API_KEY", "INSPECT_TELEMETRY_URL",
        "AGILOOP_TELEMETRY_URL", "OTEL_EXPORTER_OTLP_ENDPOINT",
        "AGILOOP_INSPECT_API_KEY", "AGILOOP_INSPECT_URL",
    )):
        raise PilotBlocked("External telemetry is outside the controlled pilot profile.")
    validate_log_policy(approval)
    from .pilot_budget import get_ledger
    get_ledger(approval)
    if config.AUDIT_BACKUP_DESTINATION not in ("", "none") or config.AUDIT_BACKUP_REQUIRED:
        raise PilotBlocked("Log backup destinations require a separate review and are disabled in this pilot.")
    import streamlit as st
    if st.get_option("server.disconnectedSessionTTL") > 60:
        raise PilotBlocked("Pilot disconnected sessions must expire within 60 seconds.")
    if os.getenv("VA_LSE_PILOT_TEXT_EXPORTS", "0") == "1":
        from importlib.metadata import version
        from .text_exports import STORE, ExportUnavailable, policy_binding
        try:
            policy_binding(approval)
            if not STORE.active or version("streamlit") != "1.63.0":
                raise ExportUnavailable("The private export service is unavailable.")
        except ExportUnavailable as exc:
            raise PilotBlocked("Reviewed TXT downloads require the accepted root-origin service and pinned runtime.") from exc
    settings = config.load_settings()
    if (not settings.api_key or settings.fallback_base_url or settings.fetch_api_key
            or https_url(settings.base_url) != https_url(approval["provider_base_url"])
            or not {settings.model_main, settings.model_fast}.issubset(set(approval["models"]))):
        raise PilotBlocked("Pilot services must use the operator's approved provider and models.")
    import streamlit as st
    try:
        auth = st.secrets["auth"]
        if (auth["redirect_uri"] != https_url(approval["deployment_url"]) + "/oauth2callback"
                or auth["server_metadata_url"] != https_url(approval["issuer"]) + "/.well-known/openid-configuration"
                or not auth["client_id"] or not auth["client_secret"]
                or len(auth["cookie_secret"]) < 32):
            raise ValueError
    except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        raise PilotBlocked("Pilot sign-in must match the reviewed HTTPS origin and issuer.") from exc
    if not st.get_option("server.enableXsrfProtection") or not st.get_option("server.enableCORS"):
        raise PilotBlocked("Pilot configuration requires XSRF and CORS protection.")
    return approval


def authorized_identity(claims: Mapping[str, Any], approval: Mapping[str, Any],
                        *, now: float | None = None) -> str:
    instant = time.time() if now is None else now
    expiry = claims.get("exp")
    issued = claims.get("iat")
    if (claims.get("is_logged_in") is not True or claims.get("iss") != approval["issuer"]
            or claims.get("sub") not in approval["subjects"]
            or isinstance(expiry, bool) or not isinstance(expiry, (int, float))
            or isinstance(issued, bool) or not isinstance(issued, (int, float))
            or not issued <= instant < expiry or not 0 < expiry - issued <= 3600):
        raise PilotBlocked("Sign in with a current invitation and a token valid for at most one hour.")
    return hashlib.sha256(f"{claims['iss']}\0{claims['sub']}".encode()).hexdigest()


def current_owner() -> str:
    import streamlit as st
    if enabled():
        # Digest workers copy contextvars, but do not have a Streamlit UI
        # context. Only action_budget installs this verified token snapshot.
        claims = _run_claims.get()
        return authorized_identity(claims if claims is not None else dict(st.user), load_approval())
    # Session-local ownership also protects synthetic queue results. A public
    # request ID is a reference, never a credential.
    if "_case_owner" not in st.session_state:
        st.session_state["_case_owner"] = uuid.uuid4().hex
    return str(st.session_state["_case_owner"])


def owns(record: Any) -> bool:
    return bool(getattr(record, "owner_id", "")) and record.owner_id == current_owner()


def notice_binding(owner: str, approval: Mapping[str, Any]) -> str:
    """Consent is session-only and specific to identity, notice and approvals."""
    data = [owner, approval["participant_notice"], approval["privacy_review"],
            approval["provider_terms"], approval["retention_policy"],
            approval["provider_base_url"], approval["models"], approval["local_log_retention_days"],
            approval["quota_policy"], approval["spending_controls"], approval.get("text_export_policy")]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def require_consent(owner: str) -> ConsentGrant:
    import streamlit as st
    expected = notice_binding(owner, load_approval())
    accepted = _run_notice.get()
    if accepted is None:
        accepted = st.session_state.get("_pilot_consent_grant")
    if (not isinstance(accepted, ConsentGrant) or accepted.binding != expected
            or accepted.revoked.is_set()):
        raise PilotBlocked("Read the current pilot privacy notice and give consent before continuing.")
    return accepted


def recheck_owner(owner: str) -> None:
    """Refuse delayed work after expiry, invitation removal or identity change."""
    if enabled():
        if current_owner() != owner:
            raise PilotBlocked("Access changed while this work was running. Sign in again.")
        require_consent(owner)


def require_session_access() -> None:
    """Check the UI identity again before storing or rendering delayed results."""
    if not enabled():
        return
    import streamlit as st
    owner = st.session_state.get("_pilot_owner")
    if not isinstance(owner, str) or not owner:
        raise PilotBlocked("Sign in again before accessing this case.")
    recheck_owner(owner)


def operator_allowed() -> bool:
    if not enabled():
        return os.getenv("VA_LSE_OPERATOR_DIAGNOSTICS", "") == "1"
    import streamlit as st
    approval = load_approval()
    authorized_identity(dict(st.user), approval)
    return st.user.get("sub") in approval["operators"]


def text_exports_enabled() -> bool:
    if not enabled() or os.getenv("VA_LSE_PILOT_TEXT_EXPORTS", "0") != "1":
        return False
    from .text_exports import ExportUnavailable, policy_binding
    try:
        policy_binding(load_approval())
        return True
    except ExportUnavailable:
        return False


def invalidate_exports(slot: str | None = None) -> None:
    from .text_exports import invalidate
    invalidate(slot)


def clear_case() -> None:
    import streamlit as st
    from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
    invalidate_exports()
    grant = st.session_state.get("_pilot_consent_grant")
    if isinstance(grant, ConsentGrant):
        grant.revoked.set()  # Shared object: copied worker contexts see this too.
    context = get_script_run_ctx(suppress_warning=True)
    if context is not None:
        context.uploaded_file_mgr.remove_session_files(context.session_id)
        from .upload_admission import UPLOADS
        UPLOADS.revoke(context.session_id)
    for key in list(st.session_state):
        del st.session_state[key]


def render_admission() -> None:
    import streamlit as st
    try:
        approval = validate_configuration()
        if not enabled():
            st.caption("Synthetic-data mode. Do not enter real veteran information.")
            return
        if not st.user.is_logged_in:
            st.title("Controlled pilot")
            st.write("Access is limited to invited participants.")
            if st.button("Sign in"):
                st.login()
            st.stop()
        owner = current_owner()  # Recheck expiry and revocation on every rerun.
        if st.session_state.get("_pilot_owner") not in (None, owner):
            clear_case()
        binding = notice_binding(owner, approval)
        accepted = st.session_state.get("_pilot_notice_consent")
        if accepted not in (None, binding):
            clear_case()
        st.session_state["_pilot_owner"] = owner
        with st.sidebar:
            st.caption("Controlled pilot · invited participant")
            if st.button("Clear case and sign out"):
                clear_case()
                st.logout()
                st.stop()
            if st.button("Clear this case"):
                clear_case()
                st.rerun()
            st.caption("Only separately accepted, reviewed .txt downloads are permitted; other file exports remain disabled."
                       if text_exports_enabled() else "File downloads are disabled for this pilot. Copy reviewed text "
                       "only to an approved destination.")
            st.caption("Clear case releases this session's working data. It cannot erase "
                       "downloads, provider copies, or records held outside this app.")
        grant = st.session_state.get("_pilot_consent_grant")
        if (st.session_state.get("_pilot_notice_consent") != binding
                or not isinstance(grant, ConsentGrant) or grant.revoked.is_set()):
            st.title("Pilot privacy notice")
            st.text(approval["participant_notice"])
            if st.checkbox("I have authority to use this information and consent to the uses described in this notice.",
                           key="notice_" + binding):
                st.session_state["_pilot_notice_consent"] = binding
                st.session_state["_pilot_consent_grant"] = ConsentGrant(binding)
                st.rerun()
            st.stop()
        with st.sidebar.expander("Pilot privacy notice"):
            st.text(approval["participant_notice"])
        from .upload_admission import bind_session
        bind_session(owner, require_consent(owner))
    except PilotBlocked as blocked:
        from .error_report import report_failure
        clear_case()
        st.error(report_failure(str(blocked), phase="pilot_admission"))
        if getattr(st.user, "is_logged_in", False) and st.button("Sign out to refresh access"):
            st.logout()
        st.stop()


def require_destination(base_url: str, model: str | None = None) -> None:
    if not enabled():
        return
    approval = load_approval()
    if https_url(base_url) != https_url(approval["provider_base_url"]):
        raise PilotBlocked("This service destination is not approved for the pilot.")
    if model is not None and model not in approval["models"]:
        raise PilotBlocked("This model is not approved for the pilot.")
    # Prevent a misconfigured provider address from targeting private services.
    host = urlsplit(base_url).hostname
    try:
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(row[4][0]).is_global for row in addresses):
            raise PilotBlocked("The approved provider must resolve to public service addresses.")
    except OSError as exc:
        raise PilotBlocked("The approved provider could not be resolved.") from exc


_lock = threading.Lock()
_active = 0
_run_budget: ContextVar[tuple[Any, str, str] | None] = ContextVar("pilot_run_budget", default=None)


def reserve_provider_attempt(owner: str, prompt_chars: int, max_tokens: int) -> tuple[Any, str, int, int]:
    """Durably debit immediately before EVERY SDK request, across all clients."""
    from .pilot_budget import REFUSAL
    budget = _run_budget.get()
    if budget is None or budget[2] != owner or type(max_tokens) is not int or max_tokens <= 0:
        raise PilotBlocked(REFUSAL)
    ledger, run, _ = budget
    ledger.verify(load_approval())
    number = ledger.attempt(run, prompt_chars)
    return ledger, run, number, min(max_tokens, ledger.settings["attempt_output_tokens"])


@contextmanager
def action_budget(records: list[Any]) -> Iterator[None]:
    """Reserve the full run envelope before any work; persistent rolling quotas."""
    global _active
    if not enabled():
        yield
        return
    owner = current_owner()
    notice = require_consent(owner)
    if _run_budget.get() is not None:
        raise PilotBlocked("A pilot run is already active in this context.")
    import streamlit as st
    claims = _run_claims.get()
    if claims is None:
        claims = dict(st.user)
    from . import config
    if sum(d.source_page_count for d in records) > min(config.MAX_RECORD_PAGES, 500):
        raise PilotBlocked("The pilot record set exceeds the page limit.")
    if (not records or any(not getattr(d, "coverage_known", False)
                           or not getattr(d, "pages", [])
                           or getattr(d, "unreadable_pages", []) for d in records)):
        raise PilotBlocked("Upload records with known, complete text coverage before a pilot run.")
    from .pilot_budget import get_ledger
    ledger = get_ledger(load_approval())
    run = ledger.start(owner)
    with _lock:
        _active += 1
    identity_token = _run_claims.set(claims)
    notice_token = _run_notice.set(notice)
    budget_token = _run_budget.set((ledger, run, owner))
    try:
        yield
        # Provider checks cannot cover work after the final call, cached/fake
        # backends or final CPU-bound processing. Reject the completed run too.
        recheck_owner(owner)
    finally:
        import sys
        unwinding = sys.exc_info()[0] is not None
        _run_budget.reset(budget_token)
        _run_notice.reset(notice_token)
        _run_claims.reset(identity_token)
        with _lock:
            _active -= 1
        try:
            ledger.finish(run)
        except PilotBlocked:
            # A failed ledger retains the full reservation and closes further
            # admission. Preserve an existing pipeline/access exception.
            if not unwinding:
                raise
    # Return to the parent/UI identity after resetting worker token snapshots.
    # A different signed-in user must not receive the old owner's completed run.
    recheck_owner(owner)


def safe_metadata(data: Mapping[str, Any]) -> dict[str, Any]:
    """Count-only persisted pilot diagnostics. No caller strings or exceptions."""
    allowed = {"duration_ms", "pages", "files", "record_files", "record_pages", "calls",
               "chunks", "facts", "attempt", "retries", "tokens_in", "tokens_out",
               "prompt_tokens", "completion_tokens", "total_tokens", "status_code"}
    result: dict[str, Any] = {key: value for key, value in data.items()
              if key in allowed and isinstance(value, (int, float, bool))}
    reference = str(data.get("request_id", ""))
    if re.fullmatch(r"(?:req_[a-f0-9]{12}|[a-f0-9-]{32,36})", reference):
        result["request_id"] = reference
    for key, choices in {"action": {"app", "evaluate", "draft"},
                         "status": {"start", "ok", "partial", "error", "timeout", "rejected", "queued", "done"},
                         "scoring_status": {"complete", "incomplete", "invalid", "unvalidated"},
                         "topic_status": {"complete", "incomplete", "invalid", "unvalidated"}}.items():
        if isinstance(data.get(key), str) and data[key] in choices:
            result[key] = data[key]
    return result


def confirm_export(text: str) -> bool:
    if not enabled():
        return True
    import streamlit as st
    current_owner()
    if re.search(r"\[(?:Confirm|Add if applicable|TODO)\b", text, re.IGNORECASE):
        st.warning("Resolve every confirmation placeholder with the witness before exporting.")
        return False
    st.warning("AI text is a review draft. Compare every factual sentence with the original "
               "records and witness account; preserve uncertainty and attribution.")
    digest = hashlib.sha256(text.encode()).hexdigest()
    return bool(st.checkbox("I reviewed this exact text against the sources and confirmed "
                            "all facts with the witness before export.", key="review_" + digest,
                            on_change=invalidate_exports))


def file_download(*args: Any, container: Any, **kwargs: Any) -> bool:
    """Do not create unauthenticated Streamlit media URLs for pilot records.

    Participants can copy reviewed text from their authorized session. File
    exports need an owner-authorized download service before a later release.
    """
    if enabled():
        return False
    return bool(container.download_button(*args, **kwargs))


def display(*args: Any, container: Any, method: str, **kwargs: Any) -> Any:
    """Render pilot content literally, without Markdown images or HTML assets.

    Record text and model output are untrusted. A Markdown image URL could send
    its embedded record text to another service from the participant's browser.
    The pilot accepts plain text in place of rich reports for this reason.
    Status messages use fixed, escaped markup to retain alert/status roles.
    """
    if not enabled():
        return getattr(container, method)(*args, **kwargs)
    result = None
    for value in args:
        if method in ("error", "warning", "info", "success"):
            from .accessibility import status_html
            # Keep status semantics without parsing caller Markdown/HTML.
            # JavaScript remains disabled by the Streamlit HTML default.
            result = container.html(status_html(method, value))
        else:
            result = container.text(str(value))
    return result


def dataframe(container: Any, data: Any, **kwargs: Any) -> Any:
    """Pilot tables have native headers, literal cells and no CSV download UI."""
    if not enabled():
        return container.dataframe(data, **kwargs)
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise TypeError("Pilot tables require row dictionaries.")
    if not data:
        return container.text("No rows.")
    from .accessibility import table_html
    return container.html(table_html(data))


def text_label(value: str) -> str:
    """Escape dynamic widget labels, which Streamlit also treats as Markdown."""
    if not enabled():
        return value
    return re.sub(r"([\\`*_{}\[\]()<>#!|])", r"\\\1", value)
