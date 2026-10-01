"""Shared, side-effect-free input contract for evaluation and drafting.

Run this before quota reservation, endpoint probes, queue/blob writes or models.
Errors contain field labels and counts only, never submitted text.
"""
from __future__ import annotations

from typing import Any

from . import config
from .documents import DRAFT_INTERNAL_MAX_CHARS, EVALUATE_INTERNAL_MAX_CHARS, DocumentPage, ExtractedDocument
from .prompt_sanitize import sanitize_for_prompt, validate_witness_field

MAX_STATEMENT_PAYLOAD_CHARS = EVALUATE_INTERNAL_MAX_CHARS + 40_000
MAX_OBSERVATIONS_PAYLOAD_CHARS = DRAFT_INTERNAL_MAX_CHARS + 40_000
MAX_FIELD_CHARS = 500
MAX_INTAKE_CHARS = 4_000
MAX_WITNESS_FIELDS = 64


class RequestValidationError(ValueError):
    """A request cannot start; the message is safe to show to its caller."""

    def __init__(self, message: str, *, field: str, reason: str = "payload_invalid") -> None:
        super().__init__(message)
        self.field = field
        self.reason = reason


def validate_text(value: Any, *, field: str, label: str, limit: int, required: bool = True) -> None:
    if not isinstance(value, str):
        raise RequestValidationError(f"{label} must be plain text.", field=field)
    if len(value) > limit:
        raise RequestValidationError(
            f"{label} is too large to send safely. Shorten or split it to at most {limit:,} characters.",
            field=field, reason="payload_too_large",
        )
    if required and (not value.strip() or not sanitize_for_prompt(value, max_chars=limit).strip()):
        raise RequestValidationError(f"Provide {label.lower()} before starting the run.", field=field)
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RequestValidationError(f"{label} contains invalid text encoding.", field=field) from exc


def validate_witness(witness: Any, *, optional: bool = False) -> None:
    if witness is None and optional:
        return
    if not isinstance(witness, dict) or len(witness) > MAX_WITNESS_FIELDS:
        raise RequestValidationError("Witness details must be a bounded map of text fields.", field="witness")
    for key, value in witness.items():
        if not isinstance(key, str) or not key or len(key) > 100:
            raise RequestValidationError("Witness field names must be short text labels.", field="witness")
        validate_text(key, field="witness", label="Witness field name", limit=100)
        limit = MAX_INTAKE_CHARS if key.startswith("aa_") else MAX_FIELD_CHARS
        validate_text(value, field="witness", label="Witness details", limit=limit, required=False)
        message = validate_witness_field(value, field_name="Witness details", max_chars=limit)
        if message:
            raise RequestValidationError("Witness details are malformed. " + message, field="witness")


def validate_records(records: Any) -> None:
    if not isinstance(records, list) or not records:
        raise RequestValidationError("Upload at least one medical record file with extractable text.", field="records")
    source_pages = text_bytes = 0
    for doc in records:
        if not isinstance(doc, ExtractedDocument):
            raise RequestValidationError("Medical records must be extracted documents.", field="records")
        validate_text(doc.filename, field="records", label="Record filename", limit=MAX_FIELD_CHARS)
        if (not isinstance(doc.pages, list) or not doc.pages
                or type(doc.total_pages) is not int or doc.total_pages < 0
                or doc.pagination not in ("page", "block")):
            raise RequestValidationError("Each medical record needs valid pages with extractable text.", field="records")
        for page in doc.pages:
            if (not isinstance(page, DocumentPage) or page.filename != doc.filename
                    or type(page.page) is not int or page.page < 1 or page.kind not in ("page", "block")):
                raise RequestValidationError("Medical record page addresses are invalid.", field="records")
            validate_text(page.text, field="records", label="Record page text", limit=config.MAX_TOTAL_UPLOAD_BYTES)
            text_bytes += len(page.text.encode("utf-8"))
        source_pages += max(doc.source_page_count, len(doc.pages))
    if source_pages > config.MAX_RECORD_PAGES:
        raise RequestValidationError("Medical records exceed the configured page limit. Split the record set.",
                                     field="records", reason="payload_too_large")
    if text_bytes > config.MAX_TOTAL_UPLOAD_BYTES:
        raise RequestValidationError("Medical record text exceeds the configured total input limit. Split the record set.",
                                     field="records", reason="payload_too_large")


def validate_evaluation_request(*, statement_text: Any, records: Any, witness: Any = None) -> None:
    validate_text(statement_text, field="statement_text", label="The lay statement", limit=MAX_STATEMENT_PAYLOAD_CHARS)
    validate_records(records)
    validate_witness(witness, optional=True)


def validate_draft_request(*, observations: Any, condition: Any, claim_type: Any, witness: Any,
                           records: Any = None, validate_record_set: bool = True) -> None:
    validate_text(observations, field="observations", label="The witness's observations", limit=MAX_OBSERVATIONS_PAYLOAD_CHARS)
    for field, label, value in (("condition", "The claimed condition", condition), ("claim_type", "The claim type", claim_type)):
        validate_text(value, field=field, label=label, limit=MAX_FIELD_CHARS)
        message = validate_witness_field(value, field_name=label, max_chars=MAX_FIELD_CHARS)
        if message:
            raise RequestValidationError(label + " is malformed. " + message, field=field)
    validate_witness(witness)
    if validate_record_set:
        validate_records(records)
