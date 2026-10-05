"""Reviewed pilot request upper bounds and pre-SDK response resource limits.

The byte-to-token bound is an operator-tested, model-specific contract, not a
universal tokenizer claim. Missing profiles close admission. No text is logged.
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping
from typing import Any

from .bounded_json import MAX_RESPONSE_BYTES, decode_json


def validate_profiles(approval: Mapping[str, Any]) -> None:
    profiles = approval.get("model_profiles")
    models = approval.get("models")
    if not isinstance(profiles, dict) or not isinstance(models, list) or set(profiles) != set(models):
        raise ValueError("Missing reviewed model profiles")
    for model, profile in profiles.items():
        if not isinstance(profile, dict) or set(profile) != {
            "model_version", "context_window_tokens", "max_output_tokens", "input_tokens_per_utf8_byte",
            "framing_token_reserve", "token_bound_evidence",
        }:
            raise ValueError("Invalid model profile")
        version = profile["model_version"]
        if not isinstance(version, str) or not 1 <= len(version) <= 256 or version != version.strip():
            raise ValueError("Missing exact returned model version")
        for field, minimum, maximum in (
            ("context_window_tokens", 1024, 2_000_000), ("max_output_tokens", 1, 100_000),
            ("input_tokens_per_utf8_byte", 1, 8), ("framing_token_reserve", 128, 65536),
        ):
            if type(profile[field]) is not int or not minimum <= profile[field] <= maximum:
                raise ValueError("Invalid model limit")
        if profile["max_output_tokens"] + profile["framing_token_reserve"] >= profile["context_window_tokens"]:
            raise ValueError("Invalid context allocation")
        if not isinstance(profile["token_bound_evidence"], str) or not 1 <= len(profile["token_bound_evidence"].strip()) <= 1024:
            raise ValueError("Missing tested token upper-bound evidence")


def check_request(model: str, body: dict[str, Any], output_tokens: int) -> None:
    """Bound the entire final wire body, including roles, escaping and re-asks."""
    from . import pilot
    if not pilot.enabled():
        return
    approval = pilot.load_approval()
    try:
        validate_profiles(approval)
        profile = approval["model_profiles"][model]
        raw = json.dumps(body, ensure_ascii=True, allow_nan=False).encode("utf-8")
        upper = len(raw) * profile["input_tokens_per_utf8_byte"] + profile["framing_token_reserve"]
        if (type(output_tokens) is not int or not 1 <= output_tokens <= profile["max_output_tokens"]
                or upper + output_tokens > profile["context_window_tokens"]):
            raise ValueError
    except (KeyError, ValueError, TypeError, UnicodeError) as exc:
        raise pilot.PilotBlocked("The complete request exceeds, or lacks, reviewed model context/output limits. Split the inputs; no source text was truncated.") from exc


def check_response_model(model: str, response: Any) -> None:
    from . import pilot
    if pilot.enabled():
        approval = pilot.load_approval()
        validate_profiles(approval)
        expected = approval["model_profiles"][model]["model_version"]
        if getattr(response, "model", None) != expected:
            raise pilot.PilotBlocked("The provider returned a different or missing approved model version. This result was refused.")


def validate_response_json(raw: bytes) -> None:
    """Bound JSON structure before the SDK creates its response object graph."""
    from . import pilot
    try:
        decode_json(raw, object_only=True)
    except (UnicodeError, ValueError, RecursionError):
        raise pilot.PilotBlocked("The provider returned an invalid or over-complex response. Its content was not logged or accepted.") from None


def bounded_transport(deadline_seconds: float, inner: Any = None) -> Any:
    """Use the installed SDK's httpx2 transport; refuse compression before decoding.

    Identity encoding avoids a decompressor allocating a bomb before a decoded
    byte counter can run. Both success and error bodies share the actual cap.
    """
    import httpx2
    from . import pilot

    class LimitedStream(httpx2.SyncByteStream):
        def __init__(self, stream: Any, deadline: float, json_required: bool) -> None:
            self.stream, self.deadline, self.json_required = stream, deadline, json_required

        def __iter__(self) -> Iterator[bytes]:
            body = bytearray()
            try:
                for chunk in self.stream:
                    if time.monotonic() > self.deadline or len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                        raise pilot.PilotBlocked("The provider response exceeded its byte or elapsed-time limit.")
                    body.extend(chunk)
                if time.monotonic() > self.deadline:
                    raise pilot.PilotBlocked("The provider response exceeded its elapsed-time limit.")
                if self.json_required:
                    validate_response_json(bytes(body))
                yield bytes(body)
            finally:
                self.close()

        def close(self) -> None:
            self.stream.close()

    class LimitedTransport(httpx2.BaseTransport):
        def __init__(self) -> None:
            self.inner = inner if inner is not None else httpx2.HTTPTransport(trust_env=False)

        def handle_request(self, request: Any) -> Any:
            deadline = time.monotonic() + deadline_seconds
            request.headers["Accept-Encoding"] = "identity"
            response = self.inner.handle_request(request)
            try:
                encoding = response.headers.get("content-encoding", "identity").strip().lower()
                length = response.headers.get("content-length")
                if encoding not in ("", "identity") or (length is not None and (not length.isdecimal() or int(length) > MAX_RESPONSE_BYTES)):
                    raise pilot.PilotBlocked("The provider returned compressed or oversized content; the response was refused before decoding.")
                if hasattr(response, "_content"):
                    if len(response.content) > MAX_RESPONSE_BYTES:
                        raise pilot.PilotBlocked("The provider response exceeded its byte limit.")
                    if request.method == "POST":
                        validate_response_json(response.content)
                response.stream = LimitedStream(response.stream, deadline, request.method == "POST")
                return response
            except BaseException:
                response.close()
                raise

        def close(self) -> None:
            self.inner.close()

    return LimitedTransport()
