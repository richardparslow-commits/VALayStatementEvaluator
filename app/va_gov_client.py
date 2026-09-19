"""VA.gov medical-record source client — **sandbox/simulator only**.

Handles per-session authenticated login and automatic record retrieval against
an OpenAI-style authenticated records endpoint. Credentials and session tokens
live only in memory for the duration of the call chain (and, at the UI layer, in
``st.session_state``) — they are never written to disk, ``.env``, or logs.

Scope, stated plainly: **real VA.gov is not reachable this way.** Actual VA.gov
access is OAuth via ID.me with SMS multi-factor authentication, and VA.gov
publishes no patient-facing medical-records API, so ``VA_GOV_API_BASE_URL`` can
only ever point at a sandbox, simulator, or mock. Do not type real VA.gov
credentials into this form; use sandbox credentials.

Mock vs. configured: when ``VA_GOV_API_BASE_URL`` is unset, authentication and
fetch run against an in-memory mock so the golden-path flow (login -> fetch ->
merge -> confirm) works with zero env vars configured. When it is set, HTTPS
calls are made to that sandbox and retried with exponential backoff.

To obtain a real record set, download it from VA.gov instead — see
``scripts/va_records_download.py`` (local browser automation of the download
wizard, with the human doing the ID.me sign-in and SMS code) or the manual steps
in ``README.md -> VA.gov record source``.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, replace
from http.client import HTTPConnection, HTTPSConnection
from typing import Any
from urllib.parse import urlencode, urlparse

from .documents import ExtractedDocument, document_from_text
from .telemetry import track_feature_error, track_interaction

# Feature id from .implement/work-breakdown.json (vagov-record-source-integration).
# Feature-id-neutral shared modules (telemetry.py) never see this constant —
# it is only ever supplied at this feature's own call sites.
FEATURE_ID = "30296e18-b734-4765-a0ea-5e9219410249"

logger = logging.getLogger("app.va_gov_client")

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.5
REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_VA_GOV_MAX_RESPONSE_BYTES = 100 * 1024 * 1024
VA_GOV_READ_CHUNK_BYTES = 64 * 1024


class VaGovError(RuntimeError):
    """Raised when VA.gov authentication or record retrieval cannot proceed.

    Mirrors the ``FetchSandboxError`` pattern used by the existing Fetch
    Sandbox integration so both sources surface errors the same way in the UI.
    """

    def __init__(
        self,
        message: str,
        *,
        error_class: str = "unknown",
        partial: bool = False,
        retrieved: int = 0,
        expected: int = 0,
    ) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.partial = partial
        self.retrieved = retrieved
        self.expected = expected


@dataclass
class VaGovSession:
    """Per-session VA.gov auth state. Never persisted beyond the process/session."""

    token: str
    patient_id: str
    issued_at: float = field(default_factory=time.time)


@dataclass
class VaGovFetchResult:
    """Result of a fetch attempt, including partial-result metadata."""

    documents: list[ExtractedDocument]
    retrieved: int
    expected: int
    partial: bool
    error_message: str | None = None


@dataclass
class MergedRecordSummaryRow:
    """One retained document, with all contributing source labels."""

    source: str
    filename: str
    pages: int


@dataclass
class MergedRecordSet:
    """Result of merging documents from multiple record sources."""

    documents: list[ExtractedDocument]
    summary: list[MergedRecordSummaryRow]
    sources_merged: int


def _va_gov_base_url() -> str:
    return os.getenv("VA_GOV_API_BASE_URL", "").strip()


def _va_gov_configured() -> bool:
    return bool(_va_gov_base_url())


def va_gov_configured() -> bool:
    """True when ``VA_GOV_API_BASE_URL`` points at a sandbox/simulator endpoint.

    Never implies real VA.gov access: no patient-facing records API exists, and
    real access is ID.me + MFA protected (see the module docstring).
    """
    return _va_gov_configured()


def authenticate_va_gov(username: str, password: str, *, patient_id: str = "") -> VaGovSession:
    """Authenticate against the configured sandbox and return a per-session token.

    Configured mode requires ``VA_GOV_API_BASE_URL`` (a sandbox/simulator — real
    VA.gov does not work this way); when unset, a deterministic mock session is
    issued (no network call, nothing persisted) so the rest of the
    fetch/merge/confirm flow can be exercised without any env vars set.
    """
    username = (username or "").strip()
    password = (password or "").strip()
    if not username or not password:
        raise VaGovError(
            "Enter your VA.gov username and password to continue.",
            error_class="invalid_credentials",
        )

    resolved_patient_id = patient_id.strip() or username

    if not _va_gov_configured():
        token = f"mock-session-{uuid.uuid4()}"
        return VaGovSession(token=token, patient_id=resolved_patient_id)

    base_url = _va_gov_base_url()
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            token = _request_token(base_url, username, password)
            return VaGovSession(token=token, patient_id=resolved_patient_id)
        except VaGovError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize transport failures
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise VaGovError(
        f"Could not sign in to VA.gov: {last_error}", error_class="auth_failed"
    )


def fetch_va_records(session: VaGovSession) -> VaGovFetchResult:
    """Fetch all available VA.gov records for an authenticated session.

    Never raises for a partial fetch — it returns whatever records were
    retrieved plus error metadata (retrieved/expected counts) so the caller
    can offer a retry while still allowing the user to proceed with the
    partial set or other sources. Only a total failure (retries exhausted,
    zero records retrieved) is reported via ``error_message``.
    """
    if not _va_gov_configured():
        documents = _mock_documents(session.patient_id)
        result = VaGovFetchResult(
            documents=documents,
            retrieved=len(documents),
            expected=len(documents),
            partial=False,
        )
        track_interaction(
            FEATURE_ID,
            {
                "records_fetched": result.retrieved,
                "records_expected": result.expected,
                "sources_merged": 1,
                "confirmed": False,
            },
        )
        return result

    base_url = _va_gov_base_url()
    last_error: VaGovError | Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            payload = _request_records(base_url, session)
            result = _normalize_records_payload(payload)
            if result.partial:
                track_feature_error(
                    FEATURE_ID,
                    VaGovError(
                        result.error_message or "Partial VA.gov fetch",
                        error_class="partial_fetch",
                        partial=True,
                        retrieved=result.retrieved,
                        expected=result.expected,
                    ),
                    error_class="partial_fetch",
                    partial=True,
                    retrieved=result.retrieved,
                    expected=result.expected,
                    retry_attempt=attempt,
                )
            track_interaction(
                FEATURE_ID,
                {
                    "records_fetched": result.retrieved,
                    "records_expected": result.expected,
                    "sources_merged": 1,
                    "confirmed": False,
                },
            )
            return result
        except VaGovError as exc:
            last_error = exc
            track_feature_error(
                FEATURE_ID,
                exc,
                error_class=exc.error_class,
                partial=exc.partial,
                retrieved=exc.retrieved,
                expected=exc.expected,
                retry_attempt=attempt,
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            return VaGovFetchResult(
                documents=[],
                retrieved=0,
                expected=exc.expected,
                partial=True,
                error_message=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 - normalize any transport failure
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            error = VaGovError(str(exc), error_class="connection_error", partial=True)
            track_feature_error(
                FEATURE_ID,
                error,
                error_class=error.error_class,
                partial=True,
                retrieved=0,
                expected=0,
                retry_attempt=attempt,
            )
            return VaGovFetchResult(
                documents=[], retrieved=0, expected=0, partial=True, error_message=str(exc)
            )
    return VaGovFetchResult(
        documents=[], retrieved=0, expected=0, partial=True, error_message=str(last_error)
    )


def merge_records(sources: dict[str, list[ExtractedDocument]]) -> MergedRecordSet:
    """Merge documents from multiple record sources into one labeled set.

    Only exact copies of text, page addresses and coverage metadata are combined.
    Distinct documents keep unique citation names without mutating the originals.
    The summary has one row per retained document, naming every source of a copy.
    """
    seen: dict[tuple[Any, ...], int] = {}
    documents: list[ExtractedDocument] = []
    source_labels: list[list[str]] = []
    reserved_names = {doc.filename for docs in sources.values() for doc in docs}
    used_names: set[str] = set()
    sources_merged = 0
    for source_label, docs in sources.items():
        if not docs:
            continue
        sources_merged += 1
        for doc in docs:
            # Full immutable inputs, not filename/size heuristics. Tuple equality
            # verifies text as well as the citation and unreadable-page metadata.
            key = (
                doc.filename, doc.pagination, doc.total_pages, tuple(doc.unreadable_pages),
                tuple((p.filename, p.page, p.kind, p.text) for p in doc.pages),
            )
            existing = seen.get(key)
            if existing is not None:
                if source_label not in source_labels[existing]:
                    source_labels[existing].append(source_label)
                continue
            retained = doc
            if doc.filename in used_names:
                base_name = f"{source_label}/{doc.filename}"
                filename = base_name
                suffix = 2
                while filename in reserved_names or filename in used_names:
                    filename = f"{base_name} ({suffix})"
                    suffix += 1
                retained = replace(
                    doc, filename=filename,
                    pages=[replace(page, filename=filename) for page in doc.pages],
                    unreadable_pages=list(doc.unreadable_pages),
                )
            seen[key] = len(documents)
            used_names.add(retained.filename)
            documents.append(retained)
            source_labels.append([source_label])
    summary = [
        MergedRecordSummaryRow(source=", ".join(labels), filename=doc.filename, pages=len(doc.pages))
        for doc, labels in zip(documents, source_labels)
    ]
    return MergedRecordSet(documents=documents, summary=summary, sources_merged=sources_merged)


# ------------------------------------------------------------------ transport
def _headers(session: VaGovSession | None = None) -> dict[str, str]:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if session is not None:
        headers["Authorization"] = f"Bearer {session.token}"
    return headers


def _validated_https_url(base_url: str, path: str, query: dict[str, str] | None = None) -> str:
    parsed = urlparse(base_url.strip())
    if parsed.scheme != "https" or not parsed.netloc:
        raise VaGovError(
            "VA_GOV_API_BASE_URL must be an absolute HTTPS URL.", error_class="config_error"
        )
    full_path = f"{parsed.path.rstrip('/')}{path}"
    if query:
        full_path = f"{full_path}?{urlencode(query)}"
    return full_path


def _request_token(base_url: str, username: str, password: str) -> str:
    body, status = _https_request(
        base_url,
        "POST",
        "/session",
        headers=_headers(),
        body=json.dumps({"username": username, "password": password}),
    )
    if status == 401:
        raise VaGovError(
            "VA.gov rejected the username or password.", error_class="invalid_credentials"
        )
    if status >= 400:
        raise VaGovError(
            f"VA.gov sign-in failed with HTTP {status}.", error_class="auth_failed"
        )
    try:
        payload = json.loads(body.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise VaGovError("VA.gov returned a non-JSON sign-in response.", error_class="auth_failed") from exc
    token = payload.get("token") or payload.get("access_token")
    if not token:
        raise VaGovError("VA.gov sign-in response did not include a session token.", error_class="auth_failed")
    return str(token)


def _request_records(base_url: str, session: VaGovSession) -> Any:
    path = _validated_https_url(base_url, "/records", {"patient_id": session.patient_id})
    body, status = _https_request(base_url, "GET", path, headers=_headers(session), body=None, already_built_path=True)
    if status == 401:
        raise VaGovError("VA.gov session expired or was rejected.", error_class="invalid_credentials")
    if status >= 500:
        raise VaGovError(f"VA.gov record service error (HTTP {status}).", error_class="connection_error", partial=True)
    if status >= 400:
        raise VaGovError(f"VA.gov record request failed with HTTP {status}.", error_class="fetch_failed")
    try:
        return json.loads(body.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise VaGovError("VA.gov returned a non-JSON records response.", error_class="fetch_failed") from exc


def _https_request(
    base_url: str,
    method: str,
    path: str,
    *,
    headers: dict[str, str],
    body: str | None,
    already_built_path: bool = False,
) -> tuple[bytes, int]:
    parsed = urlparse(base_url.strip())
    if parsed.scheme != "https" or not parsed.netloc:
        raise VaGovError("VA_GOV_API_BASE_URL must be an absolute HTTPS URL.", error_class="config_error")
    if not parsed.hostname:
        raise VaGovError("VA_GOV_API_BASE_URL must include a hostname.", error_class="config_error")
    full_path = path if already_built_path else _validated_https_url(base_url, path)
    connection = HTTPSConnection(parsed.hostname, parsed.port, timeout=REQUEST_TIMEOUT_SECONDS)
    try:
        connection.request(method, full_path, body=body, headers=headers)
        response = connection.getresponse()
        data = _read_limited_response(response, _va_gov_max_response_bytes())
        return data, response.status
    except VaGovError:
        raise
    except OSError as exc:
        raise VaGovError(f"Could not reach VA.gov: {exc}", error_class="connection_error", partial=True) from exc
    finally:
        connection.close()


def _va_gov_max_response_bytes() -> int:
    raw = os.getenv("VA_GOV_MAX_RESPONSE_BYTES", "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return DEFAULT_VA_GOV_MAX_RESPONSE_BYTES


def _read_limited_response(response: Any, max_bytes: int) -> bytes:
    content_length = response.getheader("Content-Length")
    if content_length is not None:
        invalid_length_message = (
            "VA.gov response has invalid Content-Length " f"({content_length})."
        )
        try:
            declared_size = int(str(content_length))
        except (TypeError, ValueError):
            raise VaGovError(invalid_length_message, error_class="fetch_failed")
        if declared_size < 0:
            raise VaGovError(invalid_length_message, error_class="fetch_failed")
        if declared_size > max_bytes:
            raise VaGovError(
                "VA.gov response too large "
                f"(Content-Length {content_length} bytes, limit {max_bytes} bytes).",
                error_class="fetch_failed",
                partial=True,
            )
    total = 0
    data = bytearray()
    while True:
        chunk = response.read(VA_GOV_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise VaGovError(
                "VA.gov response exceeded maximum size "
                f"({total} bytes, limit {max_bytes} bytes).",
                error_class="fetch_failed",
                partial=True,
            )
        data.extend(chunk)
    return bytes(data)


def _normalize_records_payload(payload: Any) -> VaGovFetchResult:
    """Normalize a VA.gov records response into extracted documents.

    Accepts either a bare JSON array of record items, or an object of the
    shape ``{"records": [...], "expected": N, "partial": bool, "error": str}``.
    """
    if isinstance(payload, list):
        items = payload
        expected = len(items)
        partial = False
        error_message = None
    elif isinstance(payload, dict):
        items = payload.get("records") or payload.get("documents") or []
        expected = payload.get("expected", len(items))
        partial = bool(payload.get("partial", False)) or (len(items) < expected)
        error_message = payload.get("error")
    else:
        raise VaGovError("VA.gov records response must be a JSON object or array.", error_class="fetch_failed")

    documents: list[ExtractedDocument] = []
    for index, item in enumerate(items, start=1):
        documents.append(_normalize_record_item(item, index))

    return VaGovFetchResult(
        documents=documents,
        retrieved=len(documents),
        expected=max(expected, len(documents)),
        partial=partial,
        error_message=error_message,
    )


def _normalize_record_item(item: Any, index: int) -> ExtractedDocument:
    if isinstance(item, str):
        return document_from_text(f"va_gov_record_{index}.txt", item)
    if not isinstance(item, dict):
        raise VaGovError(
            f"VA.gov record #{index} must be a JSON object or text payload.", error_class="fetch_failed"
        )
    name = str(item.get("filename") or item.get("title") or item.get("id") or f"va_gov_record_{index}")
    for key in ("text", "content", "body"):
        text = item.get(key)
        if isinstance(text, str) and text.strip():
            filename = name if name.lower().endswith((".txt", ".md")) else f"{name}.txt"
            return document_from_text(filename, text)
    return document_from_text(f"{name}.json", json.dumps(item, indent=2, sort_keys=True))


def _mock_documents(patient_id: str) -> list[ExtractedDocument]:
    """Deterministic in-memory mock records so the golden path works with no env vars set."""
    return [
        document_from_text(
            "va_gov_record_1.txt",
            f"VA.gov mock record for {patient_id}: primary care visit note, "
            "no acute findings, follow-up recommended in 6 months.",
        ),
        document_from_text(
            "va_gov_record_2.txt",
            f"VA.gov mock record for {patient_id}: audiology consult, mild "
            "high-frequency hearing loss noted bilaterally.",
        ),
    ]
