"""Bounded signed-cookie adapter for the locked Streamlit 1.63.0 protocol."""
from __future__ import annotations

import base64
import json
import re
import zlib
from importlib.metadata import version
from typing import Any

from itsdangerous import TimestampSigner, URLSafeTimedSerializer

from . import pilot
from .text_exports import ExportUnavailable

MAX_COOKIE = 32768
COOKIE = "_streamlit_user"


def _decode(raw: str, name: str, secret: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,16384}", raw):
        raise ExportUnavailable("Download unavailable.")
    # Authenticate BEFORE decompressing; do not use loads(), which has no output cap.
    signer = URLSafeTimedSerializer(secret, salt=name).make_signer()
    if not isinstance(signer, TimestampSigner):
        raise ExportUnavailable("Download unavailable.")
    payload = signer.unsign(raw, max_age=3600)
    compressed = payload.startswith(b".")
    if compressed:
        payload = payload[1:]
    decoded = base64.b64decode(payload + b"=" * (-len(payload) % 4), altchars=b"-_", validate=True)
    if compressed:
        decoder = zlib.decompressobj()
        decoded = decoder.decompress(decoded, MAX_COOKIE + 1)
        if decoder.unconsumed_tail or decoder.unused_data or not decoder.eof:
            raise ExportUnavailable("Download unavailable.")
    if len(decoded) > MAX_COOKIE:
        raise ExportUnavailable("Download unavailable.")
    value = json.loads(decoded)
    if not isinstance(value, str) or len(value) > MAX_COOKIE:
        raise ExportUnavailable("Download unavailable.")
    return value


def claims(headers: list[str], secret: str, origin: str) -> dict[str, Any]:
    if version("streamlit") != "1.63.0" or not isinstance(secret, str) or len(secret) < 32:
        raise ExportUnavailable("Download unavailable.")
    if len(headers) != 1 or len(headers[0]) > 16384:
        raise ExportUnavailable("Download unavailable.")
    cookies: dict[str, str] = {}
    for part in headers[0].split(";"):
        name, separator, value = part.strip().partition("=")
        if separator and re.fullmatch(r"_streamlit_user(?:_\d+)?", name):
            if name in cookies:
                raise ExportUnavailable("Download unavailable.")
            cookies[name] = value
    raw = cookies.get(COOKIE)
    if raw is None:
        raise ExportUnavailable("Download unavailable.")
    value = _decode(raw, COOKIE, secret)
    chunks = re.fullmatch(r"chunks-([1-4])", value)
    if chunks:
        count = int(chunks[1])
        if set(cookies) != {COOKIE, *(f"{COOKIE}_{i}" for i in range(1, count + 1))}:
            raise ExportUnavailable("Download unavailable.")
        value = "".join(_decode(cookies[f"{COOKIE}_{i}"], f"{COOKIE}_{i}", secret)
                        for i in range(1, count + 1))
    elif set(cookies) != {COOKIE}:
        raise ExportUnavailable("Download unavailable.")
    if len(value) > MAX_COOKIE:
        raise ExportUnavailable("Download unavailable.")
    result = json.loads(value, object_pairs_hook=pilot._approval_pairs,
                        parse_constant=pilot._approval_constant, parse_float=pilot._approval_float)
    if not isinstance(result, dict) or result.get("origin") != origin:
        raise ExportUnavailable("Download unavailable.")
    return result
