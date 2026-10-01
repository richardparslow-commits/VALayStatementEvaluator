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

from .documents import DocumentPage, ExtractedDocument, ExtractionError
from .parser_protocol import (DEADLINE, MAX_HEADER, MAX_INPUT, MAX_OUTPUT, MAX_TEXT,
                              SOCKET_PATH, ParserRefused, decode, encode, frame,
                              recv_frame, validate_request)

MAX_OUTPUT_BYTES = MAX_OUTPUT


def parser_image() -> str:
    image = os.environ.get("VA_LSE_PARSER_IMAGE", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ParserRefused("The operator must configure the reviewed parser image.")
    return image


def parser_health() -> None:
    image = parser_image()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(12)
        connection.connect(SOCKET_PATH)
        connection.sendall(frame(encode({"operation": "health"})))
        reply = decode(recv_frame(connection, MAX_HEADER, time.monotonic() + 12))
    if reply != {"ready": True, "image": image, "revision": os.environ.get("VA_LSE_BUILD_SHA", "")}:
        raise ParserRefused("The reviewed parser service is unavailable.")


def _documents(reply: Any, request: dict[str, Any], image: str) -> tuple[list[ExtractedDocument], list[str]]:
    if (not isinstance(reply, dict)
            or set(reply) != {"version", "nonce", "sha256", "label", "documents", "skipped", "image"}
            or type(reply["version"]) is not int or reply["version"] != 1
            or any(reply[key] != request[key] for key in ("nonce", "sha256", "label"))
            or reply["image"] != image):
        raise ParserRefused("Parser response integrity check failed.")
    raw = reply["documents"]
    skipped = reply["skipped"]
    if (not isinstance(raw, list) or len(raw) > 500 or not isinstance(skipped, list)
            or len(skipped) > 1000 or any(not isinstance(x, str) or len(x) > 4096 for x in skipped)):
        raise ParserRefused("Invalid parser document list.")
    documents: list[ExtractedDocument] = []
    names: set[str] = set()
    total, chars = 0, 0
    for item in raw:
        if (not isinstance(item, dict) or set(item) != {"filename", "schema_version", "total_pages",
                "unreadable_pages", "pagination", "coverage_known", "pages"}):
            raise ParserRefused("Invalid parser document schema.")
        name = item["filename"]
        label = request["label"]
        # Archive members are citation labels, never paths that the client opens.
        prefix = (Path(label).stem or "archive") + "/"
        member = name[len(prefix):] if isinstance(name, str) and name.startswith(prefix) else ""
        if (not isinstance(name, str) or not 0 < len(name) <= 2048 or name in names
                or not (name == label or (label.lower().endswith(".zip") and member
                    and not member.startswith("/") and ".." not in member.replace("\\", "/").split("/")))
                or any(ord(c) < 32 or ord(c) == 127 for c in name)
                or type(item["schema_version"]) is not int or item["schema_version"] != 2
                or item["coverage_known"] is not True
                or item["pagination"] not in ("page", "block")
                or type(item["total_pages"]) is not int or not 0 < item["total_pages"] <= 500):
            raise ParserRefused("Invalid parser document identity or coverage.")
        names.add(name)
        total += item["total_pages"]
        pages, unreadable = item["pages"], item["unreadable_pages"]
        if (not isinstance(pages, list) or len(pages) > 500 or not isinstance(unreadable, list)
                or len(unreadable) > 500 or any(type(n) is not int or not 1 <= n <= item["total_pages"] for n in unreadable)
                or len(set(unreadable)) != len(unreadable)):
            raise ParserRefused("Invalid parser page coverage.")
        decoded: list[DocumentPage] = []
        seen = set(unreadable)
        for page in pages:
            if (not isinstance(page, dict) or set(page) != {"page", "text", "kind"}
                    or type(page["page"]) is not int or not 1 <= page["page"] <= item["total_pages"]
                    or page["page"] in seen or page["kind"] != item["pagination"]
                    or not isinstance(page["text"], str) or not page["text"].strip()):
                raise ParserRefused("Invalid parser page schema.")
            seen.add(page["page"])
            chars += len(page["text"])
            decoded.append(DocumentPage(name, page["page"], page["text"], page["kind"]))
        if len(seen) != item["total_pages"] or total > 500 or chars > MAX_TEXT:
            raise ParserRefused("Parser response exceeds coverage or output bounds.")
        if [p.page for p in decoded] != sorted(p.page for p in decoded):
            raise ParserRefused("Parser pages are out of order.")
        documents.append(ExtractedDocument(name, decoded, item["total_pages"], unreadable,
                                           item["pagination"], True))
    return documents, skipped


class IsolatedExtractor:
    def __init__(self, timeout: float = DEADLINE + 30) -> None:
        self.timeout = min(timeout, DEADLINE + 30)

    def extract(self, label: str, data: bytes) -> tuple[list[ExtractedDocument], list[str]]:
        from . import config
        try:
            image = parser_image()
            if len(data) > min(MAX_INPUT, config.MAX_UPLOAD_BYTES):
                raise ParserRefused("Upload exceeds the parser input limit.")
            request = validate_request({"version": 1, "label": label, "size": len(data),
                                        "sha256": hashlib.sha256(data).hexdigest(), "nonce": uuid.uuid4().hex})
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                deadline = time.monotonic() + self.timeout
                connection.settimeout(self.timeout)
                connection.connect(SOCKET_PATH)
                connection.sendall(frame(encode(request)))
                connection.settimeout(max(0.001, deadline - time.monotonic()))
                connection.sendall(data)
                reply = decode(recv_frame(connection, MAX_OUTPUT, deadline))
            return _documents(reply, request, image)
        except (OSError, ValueError, KeyError, TypeError, RecursionError) as exc:
            raise ExtractionError("The protected parser refused this file. No local fallback was used.") from exc
