"""Fetch Sandbox medical-record ingestion."""
from __future__ import annotations

import base64
import json
from email.message import Message
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import ParseResult, quote, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

from .config import Settings
from .documents import (
    ExtractionError,
    ExtractedDocument,
    document_from_text,
    extract_document,
)

DOCUMENT_COLLECTION_KEYS = ("documents", "records", "files", "items")
TEXT_KEYS = ("text", "content", "body", "markdown")
STRUCTURED_KEYS = ("json", "data", "payload", "resource")
URL_KEYS = ("download_url", "url", "file_url", "href")
BASE64_KEYS = ("base64", "data_base64", "file_base64", "content_base64")
NAME_KEYS = ("filename", "name", "title", "id")
TYPE_KEYS = ("content_type", "mime_type", "media_type", "type")
REQUEST_TIMEOUT_SECONDS = 60.0


class FetchSandboxError(RuntimeError):
    """Raised when Fetch Sandbox data cannot be imported."""


class FetchClient:
    """Fetch records from a sandbox endpoint and normalize them for the app."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        if not settings.fetch_configured:
            raise FetchSandboxError(
                "Enter the Fetch Sandbox base URL and records path in the sidebar first."
            )

    def fetch_documents(self, patient_id: str) -> list[ExtractedDocument]:
        """Fetch and normalize medical-record documents for one patient or record id."""
        patient_id = patient_id.strip()
        if not patient_id:
            raise FetchSandboxError("Enter a patient or record ID before importing.")
        payload = self._request_json(self._build_records_url(patient_id))
        documents = self._normalize_payload(payload, patient_id)
        if not documents:
            raise FetchSandboxError("Fetch Sandbox returned no usable medical records.")
        return documents

    def _build_records_url(self, patient_id: str) -> str:
        template = self._settings.fetch_records_path.strip()
        encoded_id = quote(patient_id, safe="")
        if "{patient_id}" in template:
            path = template.replace("{patient_id}", encoded_id)
            return self._validated_url(path)
        base = urljoin(
            self._settings.fetch_base_url.rstrip("/") + "/", template.lstrip("/")
        )
        separator = "&" if "?" in base else "?"
        return self._validated_url(f"{base}{separator}{urlencode({'patient_id': patient_id})}")

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        api_key = self._settings.fetch_api_key.strip()
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
            headers["X-API-Key"] = api_key
        return headers

    def _request_json(self, url: str) -> Any:
        request = Request(url, headers=self._headers(), method="GET")
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                body = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            if exc.code == 401:
                raise FetchSandboxError(
                    "Fetch Sandbox rejected the request (401). Check the API key or sandbox auth mode."
                ) from exc
            raise FetchSandboxError(
                f"Fetch Sandbox request failed with HTTP {exc.code}: {exc.reason}"
            ) from exc
        except URLError as exc:
            raise FetchSandboxError(f"Could not reach Fetch Sandbox: {exc.reason}") from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise FetchSandboxError(
                "Fetch Sandbox returned non-JSON data for the records endpoint."
            ) from exc

    def _download_document(self, url: str) -> tuple[bytes, str, str]:
        request = Request(self._validated_url(url), headers=self._headers(), method="GET")
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                data = response.read()
                content_type = response.headers.get_content_type()
                filename = self._filename_from_headers(response.headers)
                return data, content_type, filename
        except HTTPError as exc:
            raise FetchSandboxError(
                f"Fetch Sandbox document download failed with HTTP {exc.code}: {exc.reason}"
            ) from exc
        except URLError as exc:
            raise FetchSandboxError(f"Could not download Fetch Sandbox document: {exc.reason}") from exc

    def _normalize_payload(
        self, payload: Any, patient_id: str
    ) -> list[ExtractedDocument]:
        items = self._extract_items(payload)
        if not items:
            return []
        if len(items) == 1 and items[0] is payload and isinstance(payload, dict):
            return [self._structured_payload_document(payload, patient_id)]
        documents: list[ExtractedDocument] = []
        for index, item in enumerate(items, start=1):
            documents.append(self._normalize_item(item, patient_id, index))
        return documents

    def _extract_items(self, payload: Any) -> list[Any]:
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            raise FetchSandboxError("Fetch Sandbox records response must be a JSON object or array.")
        for key in DOCUMENT_COLLECTION_KEYS:
            items = payload.get(key)
            if isinstance(items, list):
                return items
        nested = payload.get("data")
        if isinstance(nested, dict):
            for key in DOCUMENT_COLLECTION_KEYS:
                items = nested.get(key)
                if isinstance(items, list):
                    return items
        return [payload]

    def _normalize_item(
        self, item: Any, patient_id: str, index: int
    ) -> ExtractedDocument:
        if isinstance(item, str):
            if self._looks_like_url(item):
                data, content_type, filename = self._download_document(item)
                return self._document_from_bytes(
                    filename or f"fetch_record_{index}", data, content_type
                )
            return document_from_text(f"fetch_record_{index}.txt", item)

        if not isinstance(item, dict):
            raise FetchSandboxError(
                f"Fetch Sandbox document #{index} must be a JSON object, string URL, or text payload."
            )

        name = self._first_string(item, NAME_KEYS) or f"fetch_record_{index}"
        content_type = (self._first_string(item, TYPE_KEYS) or "").lower()

        for key in BASE64_KEYS:
            encoded = item.get(key)
            if isinstance(encoded, str) and encoded.strip():
                try:
                    data = base64.b64decode(encoded)
                except ValueError as exc:
                    raise FetchSandboxError(
                        f"{name}: invalid base64 document payload from Fetch Sandbox."
                    ) from exc
                return self._document_from_bytes(name, data, content_type)

        for key in URL_KEYS:
            maybe_url = item.get(key)
            if isinstance(maybe_url, str) and maybe_url.strip():
                data, downloaded_type, downloaded_name = self._download_document(maybe_url)
                return self._document_from_bytes(
                    downloaded_name or name,
                    data,
                    downloaded_type or content_type,
                )

        for key in TEXT_KEYS:
            text = item.get(key)
            if isinstance(text, str) and text.strip():
                filename = self._text_filename(name, content_type)
                return document_from_text(filename, text)

        for key in STRUCTURED_KEYS:
            structured = item.get(key)
            if structured not in (None, ""):
                return document_from_text(
                    self._json_filename(name),
                    self._json_text(structured),
                )

        return document_from_text(self._json_filename(name), self._json_text(item))

    def _document_from_bytes(
        self, filename: str, data: bytes, content_type: str = ""
    ) -> ExtractedDocument:
        resolved_name = self._resolved_filename(filename, content_type, data)
        lower = resolved_name.lower()
        if lower.endswith(".json"):
            try:
                parsed = json.loads(data.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                parsed = data.decode("utf-8", errors="replace")
            return document_from_text(resolved_name, self._json_text(parsed))
        if lower.endswith((".txt", ".md", ".pdf", ".docx")):
            try:
                return extract_document(resolved_name, data)
            except ExtractionError as exc:
                raise FetchSandboxError(str(exc)) from exc
        try:
            return document_from_text(f"{filename}.txt", data.decode("utf-8", errors="replace"))
        except ExtractionError as exc:
            raise FetchSandboxError(
                f"{filename}: unsupported Fetch Sandbox document format."
            ) from exc

    def _structured_payload_document(
        self, payload: dict[str, Any], patient_id: str
    ) -> ExtractedDocument:
        return document_from_text(
            self._json_filename(f"{patient_id}_records"),
            self._json_text(payload),
        )

    def _resolved_filename(self, filename: str, content_type: str, data: bytes) -> str:
        lower = filename.lower()
        if lower.endswith((".txt", ".md", ".pdf", ".docx", ".json")):
            return filename
        if data.startswith(b"%PDF"):
            return f"{filename}.pdf"
        if "wordprocessingml.document" in content_type:
            return f"{filename}.docx"
        if "markdown" in content_type:
            return f"{filename}.md"
        if "json" in content_type:
            return self._json_filename(filename)
        return f"{filename}.txt"

    def _text_filename(self, name: str, content_type: str) -> str:
        lower = name.lower()
        if lower.endswith((".txt", ".md")):
            return name
        if "markdown" in content_type:
            return f"{name}.md"
        return f"{name}.txt"

    @staticmethod
    def _json_filename(name: str) -> str:
        return name if name.lower().endswith(".json") else f"{name}.json"

    @staticmethod
    def _json_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, indent=2, sort_keys=True)

    @staticmethod
    def _first_string(item: dict[str, Any], keys: tuple[str, ...]) -> str:
        for key in keys:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _looks_like_url(value: str) -> bool:
        parsed = urlparse(value)
        return bool(parsed.scheme and parsed.netloc)

    @staticmethod
    def _filename_from_headers(headers: Message) -> str:
        disposition = headers.get("Content-Disposition", "")
        for token in disposition.split(";"):
            token = token.strip()
            if token.startswith("filename="):
                return token.split("=", 1)[1].strip().strip('"')
        return ""

    def _validated_url(self, url: str) -> str:
        base = self._parsed_base_url()
        resolved = urljoin(self._settings.fetch_base_url.rstrip("/") + "/", url)
        parsed = urlparse(resolved)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise FetchSandboxError("Fetch Sandbox URLs must be absolute HTTP(S) endpoints.")
        if (parsed.scheme, parsed.netloc) != (base.scheme, base.netloc):
            raise FetchSandboxError(
                "Fetch Sandbox document URLs must stay on the configured Fetch Sandbox host."
            )
        return resolved

    def _parsed_base_url(self) -> ParseResult:
        parsed = urlparse(self._settings.fetch_base_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise FetchSandboxError(
                "Fetch Sandbox base URL must be an absolute HTTP(S) URL."
            )
        return parsed
