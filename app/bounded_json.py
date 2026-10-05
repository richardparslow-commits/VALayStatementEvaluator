"""Strict, bounded decoding of provider envelopes and model JSON documents.

The scanner bounds nesting and structural punctuation before graph allocation.
These are structural units, not a schema or a claim of clinical correctness.
"""
from __future__ import annotations

import json
import math
from typing import Any

MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_JSON_DEPTH = 64
MAX_JSON_ITEMS = 20_000
MAX_JSON_INTEGER_DIGITS = 4300


class JSONContractError(ValueError):
    """A hard limit or ambiguous value was refused; contains no response text."""


def response_bytes(text: str) -> bytes:
    # Check characters before encoding, then actual bytes, including wrappers.
    if not isinstance(text, str) or len(text) > MAX_RESPONSE_BYTES:
        raise JSONContractError("JSON byte limit exceeded.")
    try:
        raw = text.encode("utf-8", errors="strict")
    except UnicodeError:
        raise JSONContractError("Invalid JSON Unicode.") from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise JSONContractError("JSON byte limit exceeded.")
    return raw


def decode_json(raw: bytes, *, object_only: bool = False) -> Any:
    if not raw or len(raw) > MAX_RESPONSE_BYTES:
        raise JSONContractError("JSON byte limit exceeded.")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError:
        raise JSONContractError("Invalid JSON Unicode.") from None
    stack: list[str] = []
    quoted = escaped = False
    items = 0
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            stack.append(char)
            items += 1
        elif char in "]}":
            if not stack or stack.pop() != ("[" if char == "]" else "{"):
                raise json.JSONDecodeError("Unbalanced JSON document", text, 0)
        elif char in ",:":
            items += 1
        if len(stack) > MAX_JSON_DEPTH or items > MAX_JSON_ITEMS:
            raise JSONContractError("JSON structure limit exceeded.")
    # Incomplete syntax is left to loads so a bounded truncation can be re-asked.
    def pairs(entries: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in entries:
            if key in result:
                raise JSONContractError("Duplicate JSON field.")
            result[key] = value
        return result

    def constant(value: str) -> Any:
        raise JSONContractError("Non-finite JSON number.")

    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise JSONContractError("Non-finite JSON number.")
        return number

    def bounded_int(value: str) -> int:
        # Independent of PYTHONINTMAXSTRDIGITS or a changed interpreter setting.
        if len(value) - int(value.startswith("-")) > MAX_JSON_INTEGER_DIGITS:
            raise JSONContractError("JSON integer limit exceeded.")
        return int(value)

    try:
        result = json.loads(text, object_pairs_hook=pairs, parse_constant=constant,
                            parse_float=finite_float, parse_int=bounded_int)
    except json.JSONDecodeError:
        raise
    except (ValueError, RecursionError):
        raise JSONContractError("Invalid JSON value.") from None
    if not isinstance(result, dict if object_only else (dict, list)):
        raise JSONContractError("JSON document has an unsupported root.")
    # UTF-8 input may still encode a lone surrogate as a JSON escape. Validate
    # decoded values AND keys. Valid surrogate pairs decode to a scalar normally.
    pending = [result]
    try:
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                pending.extend(value.keys())
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
            elif isinstance(value, str):
                value.encode("utf-8", errors="strict")
    except UnicodeError:
        raise JSONContractError("Invalid JSON Unicode.") from None
    return result
