"""Fail-closed client of the OS-isolated parser; never starts a local parser."""
from __future__ import annotations

import hashlib
import os
import re
import socket
import time
import uuid
from pathlib import Path
from typing import Any

from .documents import (DOCUMENT_SCHEMA_VERSION, DocumentPage, ExtractedDocument,
                        ExtractionError, validate_source_spans)
from .ingestion_policy import IngestionRefused, validate_label
from .parser_engine import parser_engine_id
from .parser_protocol import (DEADLINE, MAX_HEADER, MAX_INPUT, MAX_OUTPUT, MAX_TEXT, MAX_PAGES,
                              SOCKET_PATH, ParserRefused, decode, encode, frame,
                              recv_frame, validate_request)

MAX_OUTPUT_BYTES = MAX_OUTPUT


def _unpack_reply(value: Any, image: str) -> dict[str, Any]:
    if (not isinstance(value, dict) or set(value) != {"image", "engine_id", "response"}
            or value["image"] != image or value["engine_id"] != parser_engine_id()
            or not isinstance(value["response"], dict)
            or "image" in value["response"]):
        raise ParserRefused("Invalid parser response envelope.")
    return {**value["response"], "image": image}


def parser_image() -> str:
    image = os.environ.get("VA_LSE_PARSER_IMAGE", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ParserRefused("The operator must configure the reviewed parser image.")
    return image


def parser_health() -> None:
    from . import config
    image, engine = parser_image(), parser_engine_id()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(12)
        connection.connect(SOCKET_PATH)
        connection.sendall(frame(encode({"operation": "health"})))
        reply = decode(recv_frame(connection, MAX_HEADER, time.monotonic() + 12))
    if (not isinstance(reply, dict) or set(reply) != {"ready", "image", "revision", "max_pages", "engine_id"}
            or reply["ready"] is not True or reply["image"] != image
            or reply["engine_id"] != engine
            or reply["revision"] != os.environ.get("VA_LSE_BUILD_SHA", "")
            or type(reply["max_pages"]) is not int
            or not max(1, min(config.MAX_RECORD_PAGES, MAX_PAGES)) <= reply["max_pages"] <= MAX_PAGES):
        raise ParserRefused("The reviewed parser service is unavailable.")


def _documents(reply: Any, request: dict[str, Any], image: str) -> tuple[list[ExtractedDocument], list[str]]:
    from . import config
    limit = max(1, min(MAX_PAGES, config.MAX_RECORD_PAGES, request.get("page_limit", MAX_PAGES)))
    if (not isinstance(reply, dict)
            or set(reply) != {"version", "nonce", "sha256", "label", "documents", "skipped", "image"}
            or type(reply["version"]) is not int or reply["version"] != 1
            or any(reply[key] != request[key] for key in ("nonce", "sha256", "label"))
            or reply["image"] != image):
        raise ParserRefused("Parser response integrity check failed.")
    raw = reply["documents"]
    skipped = reply["skipped"]
    if (not isinstance(raw, list) or len(raw) > limit or not isinstance(skipped, list)
            or len(skipped) > 1000 or any(not isinstance(x, str) or len(x) > 4096 for x in skipped)):
        raise ParserRefused("Invalid parser document list.")
    documents: list[ExtractedDocument] = []
    names: set[str] = set()
    total, chars = 0, 0
    for item in raw:
        if (not isinstance(item, dict) or set(item) != {"filename", "schema_version", "total_pages",
                "unreadable_pages", "pagination", "coverage_known", "pages",
                "source_sha256", "extraction_method", "text_encoding"}):
            raise ParserRefused("Invalid parser document schema.")
        name = item["filename"]
        try:
            validate_label(name, limit=2048)
        except IngestionRefused as exc:
            raise ParserRefused("Invalid parser document label.") from exc
        label = request["label"]
        # Archive members are citation labels, never paths that the client opens.
        prefix = (Path(label).stem or "archive") + "/"
        member = name[len(prefix):] if isinstance(name, str) and name.startswith(prefix) else ""
        if (not isinstance(name, str) or not 0 < len(name) <= 2048 or name in names
                or not (name == label or (label.lower().endswith(".zip") and member
                    and not member.startswith("/")))
                or any(ord(c) < 32 or ord(c) == 127 for c in name)
                or type(item["schema_version"]) is not int or item["schema_version"] != DOCUMENT_SCHEMA_VERSION
                or not isinstance(item["source_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["source_sha256"])
                or (name == label and item["source_sha256"] != request["sha256"])
                or item["extraction_method"] not in ("pdf-text-unreviewed", "docx-stories-unreviewed", "strict-unicode")
                or item["text_encoding"] not in ("", "utf-8", "utf-8-sig", "utf-16-bom", "utf-32-bom")
                or item["coverage_known"] is not True
                or item["pagination"] not in ("page", "block")
                or type(item["total_pages"]) is not int or not 0 < item["total_pages"] <= limit):
            raise ParserRefused("Invalid parser document identity or coverage.")
        names.add(name)
        total += item["total_pages"]
        pages, unreadable = item["pages"], item["unreadable_pages"]
        if (not isinstance(pages, list) or len(pages) > limit or not isinstance(unreadable, list)
                or len(unreadable) > limit or any(type(n) is not int or not 1 <= n <= item["total_pages"] for n in unreadable)
                or len(set(unreadable)) != len(unreadable)):
            raise ParserRefused("Invalid parser page coverage.")
        decoded: list[DocumentPage] = []
        seen = set(unreadable)
        for page in pages:
            if (not isinstance(page, dict) or set(page) != {"page", "text", "kind", "source_part", "source_start", "source_end"}
                    or type(page["page"]) is not int or not 1 <= page["page"] <= item["total_pages"]
                    or page["page"] in seen or page["kind"] != item["pagination"]
                    or not isinstance(page["text"], str) or not page["text"]):
                raise ParserRefused("Invalid parser page schema.")
            part = page["source_part"]
            if not isinstance(part, str) or len(part) > 1024:
                raise ParserRefused("Invalid parser source part.")
            if part:
                try:
                    validate_label(part)
                except IngestionRefused as exc:
                    raise ParserRefused("Invalid parser source part.") from exc
                if not name.lower().endswith(".docx") or not part.startswith("word/") or not part.endswith(".xml"):
                    raise ParserRefused("Invalid parser source part.")
            seen.add(page["page"])
            chars += len(page["text"])
            if item["extraction_method"] == "docx-stories-unreviewed" and not part:
                raise ParserRefused("Missing parser source story identity.")
            decoded.append(DocumentPage(name, page["page"], page["text"], page["kind"], part,
                                        page["source_start"], page["source_end"]))
        if len(seen) != item["total_pages"] or total > limit or chars > MAX_TEXT:
            raise ParserRefused("Parser response exceeds coverage or output bounds.")
        if [p.page for p in decoded] != sorted(p.page for p in decoded):
            raise ParserRefused("Parser pages are out of order.")
        needs_spans = item["extraction_method"] in ("strict-unicode", "docx-stories-unreviewed")
        try:
            validate_source_spans(decoded, required=needs_spans)
        except ExtractionError as exc:
            raise ParserRefused("Invalid parser source spans.") from exc
        if not needs_spans and any(p.source_start is not None or p.source_end is not None or not p.text.strip() for p in decoded):
            raise ParserRefused("Invalid physical-page source metadata.")
        documents.append(ExtractedDocument(name, decoded, item["total_pages"], unreadable,
                                           item["pagination"], True, item["source_sha256"],
                                           item["extraction_method"], item["text_encoding"]))
    return documents, skipped


class IsolatedExtractor:
    def __init__(self, timeout: float = DEADLINE + 30) -> None:
        self.timeout = min(timeout, DEADLINE + 30)

    def extract(self, label: str, data: bytes) -> tuple[list[ExtractedDocument], list[str]]:
        from . import config
        try:
            image = parser_image()
            parser_engine_id()  # Refuse missing configuration before transferring record bytes.
            if len(data) > min(MAX_INPUT, config.MAX_UPLOAD_BYTES):
                raise ParserRefused("Upload exceeds the parser input limit.")
            request = validate_request({"version": 1, "label": label, "size": len(data),
                                        "sha256": hashlib.sha256(data).hexdigest(), "nonce": uuid.uuid4().hex,
                                        "page_limit": max(1, min(MAX_PAGES, config.MAX_RECORD_PAGES))})
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                deadline = time.monotonic() + self.timeout
                connection.settimeout(self.timeout)
                connection.connect(SOCKET_PATH)
                connection.sendall(frame(encode(request)))
                admission = decode(recv_frame(connection, MAX_HEADER, deadline))
                if admission == {"busy": True}:
                    raise ExtractionError("The protected parser is busy. Try this file again after the current upload finishes.")
                if admission != {"accepted": True}:
                    raise ParserRefused("Parser admission was refused.")
                connection.settimeout(max(0.001, deadline - time.monotonic()))
                connection.sendall(data)
                reply = _unpack_reply(decode(recv_frame(connection, MAX_OUTPUT, deadline)), image)
            return _documents(reply, request, image)
        except (OSError, ValueError, KeyError, TypeError, RecursionError) as exc:
            raise ExtractionError("The protected parser refused this file. No local fallback was used.") from exc
