"""Drafting-specific validation and error mapping helpers."""

from __future__ import annotations

from typing import Any

from .documents import DRAFT_INTERNAL_MAX_CHARS
from .llm import LLMConfigurationError, LLMError, LLMParseError, LLMTimeoutError, LLMUpstreamError
from .prompt_sanitize import validate_witness_field

MAX_DRAFT_OBSERVATIONS_PAYLOAD_CHARS = DRAFT_INTERNAL_MAX_CHARS + 40_000


class DraftingError(RuntimeError):
    """Drafting failure with user-safe messaging plus internal diagnostics."""

    def __init__(
        self,
        user_message: str,
        *,
        request_id: str = "",
        error_kind: str = "drafting",
        retryable: bool = False,
        diagnostics: str = "",
    ) -> None:
        super().__init__(diagnostics or user_message)
        self.user_message = user_message
        self.request_id = request_id
        self.error_kind = error_kind
        self.retryable = retryable
        self.diagnostics = diagnostics or user_message

    def format_for_user(self, request_id: str = "") -> str:
        rid = request_id or self.request_id
        suffix = f" (reference: {rid})" if rid and rid != "-" else ""
        return f"{self.user_message}{suffix}"


class DraftingConfigError(DraftingError):
    """Drafting cannot start because runtime configuration is invalid."""


class DraftingPayloadError(DraftingError):
    """Drafting input is malformed or too large to safely submit."""


class DraftingUpstreamError(DraftingError):
    """Drafting failed because the upstream model service failed."""


class DraftingParseError(DraftingError):
    """Drafting failed because the model output could not be parsed safely."""


def validate_drafting_request(
    *,
    observations: str,
    condition: str,
    claim_type: str,
    witness: dict[str, str],
) -> None:
    if not isinstance(observations, str):
        raise DraftingPayloadError(
            "Draft observations must be plain text before the request can be sent.",
            error_kind="payload_invalid",
            diagnostics=f"observations_type={type(observations).__name__}",
        )
    if len(observations) > MAX_DRAFT_OBSERVATIONS_PAYLOAD_CHARS:
        raise DraftingPayloadError(
            f"Draft observations are too large to send safely ({len(observations):,} chars). "
            f"Shorten or split them below {MAX_DRAFT_OBSERVATIONS_PAYLOAD_CHARS:,} characters and retry.",
            error_kind="payload_too_large",
            diagnostics=(
                f"observations_chars={len(observations)} "
                f"limit={MAX_DRAFT_OBSERVATIONS_PAYLOAD_CHARS}"
            ),
        )

    for field_name, value in (
        ("Claimed condition", condition),
        ("Claim type", claim_type),
        ("Witness name", witness.get("name", "")),
        ("Relationship", witness.get("relationship", "")),
        ("Known since / for", witness.get("known_since", "")),
        ("Opportunity to observe", witness.get("contact_frequency", "")),
        ("Veteran name", witness.get("veteran_name", "")),
        ("Witnessed event", witness.get("witnessed_event", "")),
        # Optional credential fields (empty is fine): capped and injection-screened
        # like every other witness field, on both the in-process and queued paths.
        ("Credential level", witness.get("credential_level", "")),
        ("Medical specialties", witness.get("medical_specialties", "")),
        ("Credentials & certifications", witness.get("credentials_detail", "")),
        ("Professional relevance", witness.get("credential_relevance", "")),
    ):
        msg = validate_witness_field(str(value or ""), field_name=field_name, max_chars=500)
        if msg:
            raise DraftingPayloadError(
                f"{field_name} is malformed for drafting. {msg}",
                error_kind="payload_invalid",
                diagnostics=f"{field_name}: {msg}",
            )


def map_drafting_exception(exc: Exception, *, request_id: str, phase: str) -> DraftingError:
    if isinstance(exc, DraftingError):
        if not exc.request_id:
            exc.request_id = request_id
        return exc
    if isinstance(exc, LLMConfigurationError):
        return DraftingConfigError(
            "Drafting is not configured correctly. Check the API key, base URL, model names, and timeout settings.",
            request_id=request_id,
            error_kind="config_error",
            diagnostics=f"{phase}: {type(exc).__name__}: {exc}",
        )
    if isinstance(exc, LLMTimeoutError):
        return DraftingUpstreamError(
            "The drafting service timed out. Please retry. If it keeps happening, shorten the input or increase the configured timeout.",
            request_id=request_id,
            error_kind="upstream_timeout",
            retryable=True,
            diagnostics=f"{phase}: {type(exc).__name__}: {exc}",
        )
    if isinstance(exc, LLMParseError):
        return DraftingParseError(
            "The drafting service returned an unreadable response. Please retry. If it repeats, use the reference when checking logs.",
            request_id=request_id,
            error_kind="parse_error",
            diagnostics=f"{phase}: {type(exc).__name__}: {exc}",
        )
    if isinstance(exc, LLMUpstreamError):
        return DraftingUpstreamError(
            (
                "The drafting service is temporarily unavailable. Please retry in a moment."
                if exc.retriable
                else "The drafting request was rejected before a statement could be produced. Check the configuration and input, then retry."
            ),
            request_id=request_id,
            error_kind="upstream_error",
            retryable=exc.retriable,
            diagnostics=f"{phase}: {type(exc).__name__}: {exc}",
        )
    if isinstance(exc, LLMError):
        return DraftingUpstreamError(
            "The drafting service failed before a statement could be produced. Please retry in a moment.",
            request_id=request_id,
            error_kind="upstream_error",
            retryable=False,
            diagnostics=f"{phase}: {type(exc).__name__}: {exc}",
        )
    return DraftingError(
        "An unexpected drafting error occurred. Please retry. If it continues, use the reference when checking logs.",
        request_id=request_id,
        error_kind="unexpected_error",
        retryable=False,
        diagnostics=f"{phase}: {type(exc).__name__}: {exc}",
    )


def format_error_for_user(exc: Exception, request_id: str) -> str:
    if isinstance(exc, DraftingError):
        return exc.format_for_user(request_id)
    rid_suffix = f" (reference: {request_id})" if request_id and request_id != "-" else ""
    return f"{exc}{rid_suffix}"


def error_extra(exc: DraftingError) -> dict[str, Any]:
    return {
        "error_kind": exc.error_kind,
        "retryable": exc.retryable,
        "diagnostics": exc.diagnostics,
    }
