"""Shared, side-effect-free input contract for evaluation and drafting.

Run this before quota reservation, endpoint probes, queue/blob writes or models.
Errors contain field labels and counts only, never submitted text.
"""
from __future__ import annotations

from typing import Any

from . import config
from .documents import DRAFT_INTERNAL_MAX_CHARS, EVALUATE_INTERNAL_MAX_CHARS, DocumentPage, ExtractedDocument
from .prompt_sanitize import has_prompt_text, validate_witness_field

MAX_STATEMENT_PAYLOAD_CHARS = EVALUATE_INTERNAL_MAX_CHARS
MAX_OBSERVATIONS_PAYLOAD_CHARS = DRAFT_INTERNAL_MAX_CHARS
MAX_FIELD_CHARS = 500
MAX_INTAKE_CHARS = 4_000
MAX_WITNESS_FIELDS = 64


class RequestValidationError(ValueError):
    """A request cannot start; the message is safe to show to its caller."""

    def __init__(self, message: str, *, field: str, reason: str = "payload_invalid") -> None:
        super().__init__(message)
        self.field = field
        self.reason = reason


def validate_text(value: Any, *, field: str, label: str, limit: int, required: bool = True) -> int:
    if not isinstance(value, str):
        raise RequestValidationError(f"{label} must be plain text.", field=field)
    if len(value) > limit:
        raise RequestValidationError(
            f"{label} is too large to send safely. Shorten or split it to at most {limit:,} characters.",
            field=field, reason="payload_too_large",
        )
    if required and not has_prompt_text(value):
        raise RequestValidationError(f"Provide {label.lower()} before starting the run.", field=field, reason="payload_missing")
    try:
        return len(value.encode("utf-8"))
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


def validate_follow_up_prompt_budget(original: str, submitted: str, *, field: str,
                                    label: str, limit: int) -> None:
    """Saved answers must fit; an earlier base-text waiver cannot discard them."""
    if submitted != original and len(submitted) > limit:
        raise RequestValidationError(
            f"{label} with saved follow-up answers exceeds the {limit:,}-character model limit. "
            f"Shorten {label.lower()} so all saved answers fit before trying again. "
            "Your saved answers have been kept.",
            field=field, reason="payload_too_large",
        )


def validate_records(records: Any) -> tuple[int, int]:
    if not isinstance(records, list):
        raise RequestValidationError("Medical records must be a list of extracted documents.", field="records")
    if not records:
        raise RequestValidationError("Upload at least one medical record file with extractable text.",
                                     field="records", reason="payload_missing")
    source_pages = text_bytes = text_chars = 0
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
            text_bytes += validate_text(page.text, field="records", label="Record page text", limit=config.MAX_TOTAL_UPLOAD_BYTES)
            text_chars += len(page.text)
        source_pages += max(doc.source_page_count, len(doc.pages))
    if source_pages > config.MAX_RECORD_PAGES:
        raise RequestValidationError("Medical records exceed the configured page limit. Split the record set.",
                                     field="records", reason="payload_too_large")
    if text_bytes > config.MAX_TOTAL_UPLOAD_BYTES:
        raise RequestValidationError("Medical record text exceeds the configured total input limit. Split the record set.",
                                     field="records", reason="payload_too_large")

    from . import pilot
    if pilot.enabled():
        require_complete_record_coverage(records)
    if pilot.enabled() and text_chars > 20 * 1024 * 1024:
        raise RequestValidationError("The pilot record text exceeds its processing limit.", field="records")
    return source_pages, text_bytes


RECORD_COVERAGE_POLICY = "complete_readable_source_units_v1"


def require_complete_record_coverage(records: list[ExtractedDocument]) -> None:
    """Full, known extraction coverage; this does not certify OCR/layout accuracy.

    Counts alone do not suffice: duplicate, missing, reordered or wrong-kind
    page/block addresses must not masquerade as all source units being present.
    """
    seen: set[tuple[str, str]] = set()
    for doc in records:
        identity = (doc.filename, doc.pagination)
        if (doc.coverage_known is not True or not isinstance(doc.unreadable_pages, list)
                or doc.unreadable_pages
                or type(doc.total_pages) is not int or doc.total_pages < 1
                or doc.total_pages != len(doc.pages) or identity in seen
                or any(type(page.page) is not int or page.page != index
                       or page.kind != doc.pagination or page.filename != doc.filename
                       or not isinstance(page.text, str) or not page.text.strip()
                       for index, page in enumerate(doc.pages, 1))):
            raise RequestValidationError(
                "Medical record coverage is incomplete or unknown. Review and supply a complete readable copy before a pilot run.",
                field="records", reason="coverage_incomplete",
            )
        seen.add(identity)


def validate_evaluation_request(*, statement_text: Any, records: Any, witness: Any = None,
                                validate_record_set: bool = True,
                                statement_source: ExtractedDocument | None = None) -> None:
    validate_text(statement_text, field="statement_text", label="The lay statement", limit=MAX_STATEMENT_PAYLOAD_CHARS)
    if statement_source is not None:
        if (not isinstance(statement_source, ExtractedDocument) or not statement_source.coverage_known
                or statement_source.unreadable_pages or not statement_source.pages
                or statement_source.source_page_count != len(statement_source.pages)):
            raise RequestValidationError("The uploaded statement has incomplete text coverage. Review and supply a complete readable copy.", field="statement_text")
        original = statement_source.full_text.strip()
        if not (statement_text.strip() == original or statement_text.strip().startswith(original + "\n")):
            raise RequestValidationError("The uploaded statement text no longer matches its source.", field="statement_text")
    if validate_record_set:
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
