"""Bounded private parser protocol; contains no application settings or secrets."""
from __future__ import annotations

import hashlib
import json
import re
import socket
import struct
import time
from typing import Any, BinaryIO

from .ingestion_policy import IngestionRefused, validate_label

MAX_INPUT = 50 * 1024 * 1024
MAX_OUTPUT = 32 * 1024 * 1024
MAX_HEADER = 4096
MAX_TEXT = 20 * 1024 * 1024
MAX_PAGES = 5000
MAX_JSON_STRUCTURE = 200000
MAX_JSON_DEPTH = 16
SOCKET_PATH = "/run/parser/parser.sock"
DEADLINE = 60


class ParserRefused(ValueError):
    """No record content is included in error messages."""


def encode(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def validate_json_structure(data: bytes) -> None:
    # Our encoder always escapes non-ASCII characters. Refusing alternative
    # encodings prevents both scanner ambiguity and a 4x widened string allocation.
    if not data or not data.isascii() or b"\x00" in data:
        raise ParserRefused("Invalid parser JSON encoding.")
    # Bound allocations before json.loads builds a graph in the trusted process.
    # Punctuation inside record strings is data, not structure; escaped quotes
    # and backslashes must not let a hostile string evade this scan.
    depth = structure = 0
    quoted = escaped = False
    for value in data:
        if quoted:
            if escaped:
                escaped = False
            elif value == 92:
                escaped = True
            elif value == 34:
                quoted = False
            continue
        if value == 34:
            quoted = True
        elif value in (123, 91):
            depth += 1
            structure += 1
            if depth > MAX_JSON_DEPTH:
                raise ParserRefused("Parser JSON nesting exceeds its limit.")
        elif value in (125, 93):
            depth -= 1
            structure += 1
        elif value in (44, 58):
            structure += 1
        if structure > MAX_JSON_STRUCTURE:
            raise ParserRefused("Parser JSON structure exceeds its limit.")


def decode(data: bytes) -> Any:
    validate_json_structure(data)
    def unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ParserRefused("Duplicate parser fields.")
            result[key] = value
        return result
    return json.loads(data.decode("ascii"), object_pairs_hook=unique)


def validate_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"version", "label", "size", "sha256", "nonce", "page_limit"}:
        raise ParserRefused("Invalid parser request.")
    if (type(value["version"]) is not int or value["version"] != 1
            or type(value["size"]) is not int or not 0 < value["size"] <= MAX_INPUT
            or type(value["page_limit"]) is not int or not 0 < value["page_limit"] <= MAX_PAGES
            or not isinstance(value["label"], str) or not 0 < len(value["label"]) <= 1024
            or any(ord(c) < 32 or ord(c) == 127 for c in value["label"])
            or not isinstance(value["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
            or not isinstance(value["nonce"], str) or not re.fullmatch(r"[0-9a-f]{32}", value["nonce"])):
        raise ParserRefused("Invalid parser request.")
    try:
        validate_label(value["label"])
    except IngestionRefused as exc:
        raise ParserRefused("Invalid parser file label.") from exc
    return value


def bind_input(request: dict[str, Any], data: bytes) -> None:
    if len(data) != request["size"] or hashlib.sha256(data).hexdigest() != request["sha256"]:
        raise ParserRefused("Parser input integrity check failed.")


def recv_exact(connection: socket.socket, size: int, deadline: float | None = None) -> bytes:
    result = bytearray()
    while len(result) < size:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ParserRefused("Parser transport deadline exceeded.")
            connection.settimeout(remaining)
        part = connection.recv(min(65536, size - len(result)))
        if not part:
            raise ParserRefused("Incomplete parser response.")
        result.extend(part)
    return bytes(result)


def recv_frame(connection: socket.socket, limit: int, deadline: float | None = None) -> bytes:
    size = struct.unpack("!I", recv_exact(connection, 4, deadline))[0]
    if not 0 < size <= limit:
        raise ParserRefused("Parser frame exceeds its limit.")
    return recv_exact(connection, size, deadline)


def frame(data: bytes) -> bytes:
    return struct.pack("!I", len(data)) + data


def read_request(stream: BinaryIO) -> tuple[dict[str, Any], bytes]:
    prefix = stream.read(4)
    if len(prefix) != 4:
        raise ParserRefused("Incomplete parser request.")
    size = struct.unpack("!I", prefix)[0]
    if not 0 < size <= MAX_HEADER:
        raise ParserRefused("Parser header exceeds its limit.")
    request = validate_request(decode(stream.read(size)))
    data = stream.read(request["size"] + 1)
    bind_input(request, data)
    return request, data
