"""Serialization of queued Evaluate/Draft jobs and their results.

The queue in :mod:`app.job_queue` moves opaque JSON strings between the web pod
and worker pods, so something has to turn this app's objects into JSON and back.
That is this module, and it is deliberately narrow:

* **Documents** — extracted page text travels with the job, so a worker does not
  need the original upload (and the web pod's extraction cache stays the only
  PDF/DOCX parser in the system).
* **Results** — the pipeline returns dataclasses (``EvaluationResult``,
  ``DraftResult``, ``MedicalDigest``); these are rebuilt on the reading pod so the
  results panel renders from the same types it always has.
* **Usage** — the token/call tracker the results panel reports on.

Round-tripping is defensive on the read path: a worker on a newer version can
write fields an older pod does not know, and a truncated Redis value must
degrade into a clear error rather than an ``AttributeError`` three frames deep in
the results panel.

The envelope also carries the submit span's W3C trace context when tracing is on
(``app/tracing.py``) — that is what stitches the web pod's span to the worker's
instead of leaving two unrelated traces.

This module is stdlib-only and holds no I/O.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from . import config
from .blob_store import BlobRef, BlobStore, BlobStoreError, dumps_documents, loads_documents
from .documents import DocumentPage, ExtractedDocument
from .draft import DraftResult
from .evaluate import EvaluationResult
from .medical_review import MedicalDigest, MedicalFact
from .config import PRIMARY_ENDPOINT
from .usage import UsageEntry, UsageTracker

PAYLOAD_VERSION = 1
KIND_EVALUATE = "evaluate"
KIND_DRAFT = "draft"


class PayloadError(RuntimeError):
    """Raised when a job payload or result cannot be (de)serialized."""


class PayloadTooLargeError(PayloadError):
    """Raised when a payload exceeds ``VA_LSE_JOB_QUEUE_MAX_PAYLOAD_BYTES``."""


# ------------------------------------------------------------------- coercion
def _as_str(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(k): v for k, v in value.items()}


def _dict_items(value: Any) -> list[dict[str, Any]]:
    return [x for x in _as_list(value) if isinstance(x, dict)]


def _str_items(value: Any) -> list[str]:
    return [x for x in _as_list(value) if isinstance(x, str)]


def _float_map(value: Any) -> dict[str, float]:
    return {str(k): _as_float(v) for k, v in _as_dict(value).items()}


def _str_map(value: Any) -> dict[str, str]:
    return {str(k): _as_str(v) for k, v in _as_dict(value).items()}


# ------------------------------------------------------------------ documents
def document_to_json(doc: ExtractedDocument) -> dict[str, Any]:
    return {
        "filename": doc.filename,
        "pages": [{"page": p.page, "text": p.text} for p in doc.pages],
    }


def document_from_json(raw: Any) -> ExtractedDocument | None:
    data = _as_dict(raw)
    filename = _as_str(data.get("filename"))
    if not filename:
        return None
    pages: list[DocumentPage] = []
    for entry in _dict_items(data.get("pages")):
        text = _as_str(entry.get("text"))
        if not text:
            continue
        pages.append(DocumentPage(filename, max(1, _as_int(entry.get("page"), 1)), text))
    if not pages:
        return None
    return ExtractedDocument(filename=filename, pages=pages)


def documents_to_json(records: list[ExtractedDocument]) -> list[dict[str, Any]]:
    return [document_to_json(doc) for doc in records]


def documents_from_json(raw: Any) -> list[ExtractedDocument]:
    docs: list[ExtractedDocument] = []
    for entry in _as_list(raw):
        doc = document_from_json(entry)
        if doc is not None:
            docs.append(doc)
    return docs


# --------------------------------------------------------------------- digest
def digest_to_json(digest: MedicalDigest | None) -> dict[str, Any] | None:
    if digest is None:
        return None
    return {
        "facts": [
            {
                "date": f.date,
                "type": f.type,
                "description": f.description,
                "source": f.source,
                "quote": f.quote,
                "document": f.document,
                "page": f.page,
                "section": f.section,
            }
            for f in digest.facts
        ],
        "conditions": list(digest.conditions),
        "providers": list(digest.providers),
        "summary": digest.summary,
        "pages_reviewed": digest.pages_reviewed,
        "chunks_reviewed": digest.chunks_reviewed,
        "duplicates_skipped": digest.duplicates_skipped,
        "pages_in_files": digest.pages_in_files,
        "unreadable_pages": digest.unreadable_pages,
        "chunks_without_facts": digest.chunks_without_facts,
        "duplicate_pages": list(digest.duplicate_pages),
        "corroborated_pages": list(digest.corroborated_pages),
        "files": list(digest.files),
        "citation_check": dict(digest.citation_check),
        "facts_dropped_by_cap": digest.facts_dropped_by_cap,
    }


def digest_from_json(raw: Any) -> MedicalDigest | None:
    if not isinstance(raw, dict):
        return None
    facts = [
        MedicalFact(
            date=_as_str(item.get("date")),
            type=_as_str(item.get("type")),
            description=_as_str(item.get("description")),
            source=_as_str(item.get("source")),
            quote=_as_str(item.get("quote")),
            document=_as_str(item.get("document")),
            page=_as_int(item.get("page")),
            section=_as_str(item.get("section")),
        )
        for item in _dict_items(raw.get("facts"))
    ]
    return MedicalDigest(
        facts=facts,
        conditions=_str_items(raw.get("conditions")),
        providers=_str_items(raw.get("providers")),
        summary=_as_str(raw.get("summary")),
        pages_reviewed=_as_int(raw.get("pages_reviewed")),
        chunks_reviewed=_as_int(raw.get("chunks_reviewed")),
        duplicates_skipped=_as_int(raw.get("duplicates_skipped")),
        pages_in_files=_as_int(raw.get("pages_in_files")),
        unreadable_pages=_as_int(raw.get("unreadable_pages")),
        chunks_without_facts=_as_int(raw.get("chunks_without_facts")),
        duplicate_pages=_dict_items(raw.get("duplicate_pages")),
        corroborated_pages=_dict_items(raw.get("corroborated_pages")),
        files=_dict_items(raw.get("files")),
        citation_check=_as_dict(raw.get("citation_check")),
        facts_dropped_by_cap=_as_int(raw.get("facts_dropped_by_cap")),
    )


# ---------------------------------------------------------------- results
def evaluation_to_json(result: EvaluationResult) -> dict[str, Any]:
    return {
        "claimed_condition": result.claimed_condition,
        "writer_role": result.writer_role,
        "claims": list(result.claims),
        "verifications": list(result.verifications),
        "scores": dict(result.scores),
        "rationales": dict(result.rationales),
        "improvements": list(result.improvements),
        "omitted_record_facts": list(result.omitted_record_facts),
        "evidence_gaps": list(result.evidence_gaps),
        "executive_summary": result.executive_summary,
        "topic_focus": result.topic_focus,
        "topic_rows": list(result.topic_rows),
        "topic_critical_gaps": list(result.topic_critical_gaps),
        "topic_notes": result.topic_notes,
        "revision_notes": result.revision_notes,
        "revision_changes": list(result.revision_changes),
        "revised_statement": result.revised_statement,
        "added_facts_to_verify": list(result.added_facts_to_verify),
        "digest": digest_to_json(result.digest),
        "report_markdown": result.report_markdown,
        "input_chars": result.input_chars,
        "truncated_chars": result.truncated_chars,
        "truncation_warning": result.truncation_warning,
        "evidence_source": list(result.evidence_source),
    }


def evaluation_from_json(raw: Any) -> EvaluationResult:
    data = _as_dict(raw)
    return EvaluationResult(
        claimed_condition=_as_str(data.get("claimed_condition")),
        writer_role=_as_str(data.get("writer_role")),
        claims=_dict_items(data.get("claims")),
        verifications=_dict_items(data.get("verifications")),
        scores=_float_map(data.get("scores")),
        rationales=_str_map(data.get("rationales")),
        improvements=_dict_items(data.get("improvements")),
        omitted_record_facts=_dict_items(data.get("omitted_record_facts")),
        evidence_gaps=_dict_items(data.get("evidence_gaps")),
        executive_summary=_as_str(data.get("executive_summary")),
        topic_focus=_as_str(data.get("topic_focus")),
        topic_rows=_dict_items(data.get("topic_rows")),
        topic_critical_gaps=_str_items(data.get("topic_critical_gaps")),
        topic_notes=_as_str(data.get("topic_notes")),
        revision_notes=_as_str(data.get("revision_notes")),
        revision_changes=_dict_items(data.get("revision_changes")),
        revised_statement=_as_str(data.get("revised_statement")),
        added_facts_to_verify=_str_items(data.get("added_facts_to_verify")),
        digest=digest_from_json(data.get("digest")),
        report_markdown=_as_str(data.get("report_markdown")),
        input_chars=_as_int(data.get("input_chars")),
        truncated_chars=_as_int(data.get("truncated_chars")),
        truncation_warning=_as_str(data.get("truncation_warning")),
        evidence_source=_dict_items(data.get("evidence_source")),
    )


def draft_to_json(result: DraftResult) -> dict[str, Any]:
    return {
        "grounding": dict(result.grounding) if isinstance(result.grounding, dict) else {},
        "draft": result.draft,
        "final_statement": result.final_statement,
        "review_issues": list(result.review_issues),
        "digest": digest_to_json(result.digest),
        "input_chars": result.input_chars,
        "truncated_chars": result.truncated_chars,
        "truncation_warning": result.truncation_warning,
        "evidence_source": list(result.evidence_source),
    }


def draft_from_json(raw: Any) -> DraftResult:
    data = _as_dict(raw)
    return DraftResult(
        grounding=_as_dict(data.get("grounding")),
        draft=_as_str(data.get("draft")),
        final_statement=_as_str(data.get("final_statement")),
        review_issues=_str_items(data.get("review_issues")),
        digest=digest_from_json(data.get("digest")),
        input_chars=_as_int(data.get("input_chars")),
        truncated_chars=_as_int(data.get("truncated_chars")),
        truncation_warning=_as_str(data.get("truncation_warning")),
        evidence_source=_dict_items(data.get("evidence_source")),
    )


# ---------------------------------------------------------------------- usage
def usage_to_json(usage: UsageTracker) -> dict[str, Any]:
    return {
        "entries": [
            {
                "model": e.model,
                "phase": e.phase,
                "prompt_tokens": e.prompt_tokens,
                "completion_tokens": e.completion_tokens,
                # Recorded so a run that failed over is still identifiable after
                # the round trip through the queue — the worker's report is what
                # the user reads, and a silently switched model must not vanish.
                "endpoint": e.endpoint,
            }
            for e in usage.entries
        ]
    }


def usage_from_json(raw: Any) -> UsageTracker:
    tracker = UsageTracker()
    for item in _dict_items(_as_dict(raw).get("entries")):
        tracker.entries.append(
            UsageEntry(
                model=_as_str(item.get("model")),
                phase=_as_str(item.get("phase")),
                prompt_tokens=_as_int(item.get("prompt_tokens")),
                completion_tokens=_as_int(item.get("completion_tokens")),
                # Missing (an older web pod's payload) reads as the primary, which
                # is what a payload without the field actually meant.
                endpoint=_as_str(item.get("endpoint")) or PRIMARY_ENDPOINT,
            )
        )
    return tracker


# --------------------------------------------------------------------- jobs
@dataclass
class EvaluateJob:
    """Inputs for one queued Evaluate run."""

    statement_text: str
    records: list[ExtractedDocument]
    request_id: str = ""
    # Record-source labels ("Upload", "VA.gov", …) carried so the worker's audit
    # event says where the pages came from, exactly like the in-process path.
    record_sources: list[str] = field(default_factory=list)
    # Witness metadata dict — consumed only for its structured ``aa_*``
    # Aid & Attendance intake answers (see app/aa_intake.py). Empty default
    # keeps every pre-existing payload byte-identical on the wire.
    witness: dict[str, str] = field(default_factory=dict)
    # W3C trace context (``traceparent``/``tracestate``) from the web pod's submit
    # span, so the worker's spans join that trace instead of starting a new one.
    # Empty whenever tracing is off — see app/tracing.py.
    trace_context: dict[str, str] = field(default_factory=dict)


@dataclass
class DraftJob:
    """Inputs for one queued Draft run."""

    records: list[ExtractedDocument]
    witness: dict[str, str]
    observations: str
    condition: str
    claim_type: str
    request_id: str = ""
    record_sources: list[str] = field(default_factory=list)
    trace_context: dict[str, str] = field(default_factory=dict)  # see EvaluateJob


@dataclass
class RunResult:
    """A finished run's result plus the usage it consumed."""

    kind: str
    result: EvaluationResult | DraftResult
    usage: UsageTracker
    request_id: str = ""


def _dump(payload: dict[str, Any]) -> str:
    try:
        encoded = json.dumps(payload)
    except (TypeError, ValueError) as exc:
        raise PayloadError(f"job payload is not JSON-serializable: {exc}") from exc
    size = len(encoded.encode("utf-8"))
    if size > config.JOB_QUEUE_MAX_PAYLOAD_BYTES:
        raise PayloadTooLargeError(
            f"job payload is {size:,} bytes, over the {config.JOB_QUEUE_MAX_PAYLOAD_BYTES:,}-byte "
            "queue limit (VA_LSE_JOB_QUEUE_MAX_PAYLOAD_BYTES). Split the record set."
        )
    return encoded


def documents_bundle(job: EvaluateJob | DraftJob) -> bytes:
    """The bytes a blob store holds for a job: just its extracted documents.

    Storing only the documents (not the whole payload) keeps content addressing
    useful: two users submitting the same record bundle reuse one blob, because
    the statement/observations sitting next to it never enter the hash.
    """
    return dumps_documents(
        {"version": PAYLOAD_VERSION, "documents": documents_to_json(job.records)}
    )


def _job_envelope(kind: str, job: EvaluateJob | DraftJob) -> dict[str, Any]:
    base: dict[str, Any] = {
        "version": PAYLOAD_VERSION,
        "kind": kind,
        "request_id": job.request_id,
        "record_sources": list(job.record_sources),
    }
    # Only present when a span is actually recording, so a tracing-off deployment
    # produces byte-identical payloads to one that predates tracing. Imported here
    # rather than at module scope to keep this module's stdlib-only import graph
    # (see the module docstring) — tracing is optional and may be absent.
    from .tracing import inject_trace_context

    trace_context = inject_trace_context()
    if trace_context:
        base["trace_context"] = trace_context
    if isinstance(job, EvaluateJob):
        base["statement_text"] = job.statement_text
        # Present only when the caller supplied intake answers, so pre-existing
        # payloads stay byte-identical.
        if job.witness:
            base["witness"] = dict(job.witness)
    else:
        base["observations"] = job.observations
        base["condition"] = job.condition
        base["claim_type"] = job.claim_type
        base["witness"] = dict(job.witness)
    return base


def encode_job(kind: str, job: EvaluateJob | DraftJob) -> str:
    """Encode a job's inputs with its documents inline (size-checked)."""
    base = _job_envelope(kind, job)
    base["documents"] = documents_to_json(job.records)
    return _dump(base)


def encode_job_with_blob(kind: str, job: EvaluateJob | DraftJob, ref: BlobRef) -> str:
    """Encode a job whose documents live in the blob store.

    The queue carries the reference and the run's small inputs; the blob carries
    the record text. See ``app/blob_store.py`` for why.
    """
    base = _job_envelope(kind, job)
    base["documents_ref"] = ref.to_json()
    return _dump(base)


def payload_needs_blob(kind: str, job: EvaluateJob | DraftJob) -> bool:
    """True when this job's inline payload would exceed the inline threshold."""
    try:
        encoded = encode_job(kind, job)
    except PayloadTooLargeError:
        return True
    return len(encoded.encode("utf-8")) > config.JOB_QUEUE_INLINE_MAX_BYTES


def decode_job(
    kind: str, raw: str, *, blob_store: BlobStore | None = None
) -> EvaluateJob | DraftJob:
    """Decode a queued job's inputs, raising :class:`PayloadError` if unusable.

    Documents come either inline or from ``blob_store`` via a reference. A
    reference with no store configured is a hard error rather than a fallback:
    silently running with zero records would produce a confident, empty report.
    """
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise PayloadError("job payload is not valid JSON") from exc
    if not isinstance(data, dict):
        raise PayloadError("job payload is not a JSON object")
    if _as_str(data.get("kind")) != kind:
        raise PayloadError(f"job payload kind mismatch: expected {kind!r}")

    records: list[ExtractedDocument] = []
    if data.get("documents_ref") is not None:
        records = _documents_from_ref(data.get("documents_ref"), blob_store)
    else:
        records = documents_from_json(data.get("documents"))
    if not records:
        raise PayloadError("job payload carries no extractable record text")
    request_id = _as_str(data.get("request_id"))
    sources = _str_items(data.get("record_sources"))
    if kind == KIND_EVALUATE:
        statement = _as_str(data.get("statement_text"))
        if not statement.strip():
            raise PayloadError("job payload carries no statement text")
        return EvaluateJob(
            statement_text=statement,
            records=records,
            request_id=request_id,
            record_sources=sources,
            witness=_str_map(data.get("witness")),
            trace_context=_str_map(data.get("trace_context")),
        )
    if kind == KIND_DRAFT:
        return DraftJob(
            records=records,
            witness=_str_map(data.get("witness")),
            observations=_as_str(data.get("observations")),
            condition=_as_str(data.get("condition")),
            claim_type=_as_str(data.get("claim_type")),
            request_id=request_id,
            record_sources=sources,
            trace_context=_str_map(data.get("trace_context")),
        )
    raise PayloadError(f"unknown job kind: {kind!r}")


def _documents_from_ref(raw: Any, blob_store: BlobStore | None) -> list[ExtractedDocument]:
    """Fetch and parse a job's documents from the blob store."""
    ref = BlobRef.from_json(raw)
    if ref is None:
        raise PayloadError("job payload has an unreadable documents reference")
    if blob_store is None:
        raise PayloadError(
            f"this job's records are in blob {ref.key} ({ref.backend}), but no blob store is "
            "configured on this worker. Set VA_LSE_BLOB_DIR or VA_LSE_BLOB_S3_BUCKET to the "
            "same value the web tier uses (see DEPLOYMENT.md → Pattern C)."
        )
    try:
        bundle = loads_documents(blob_store.get(ref))
    except BlobStoreError as exc:
        raise PayloadError(str(exc)) from exc
    return documents_from_json(bundle.get("documents"))


def encode_result(run: RunResult) -> str:
    """Encode a finished run for the queue (best-effort — never loses the run)."""
    body: dict[str, Any] = {
        "version": PAYLOAD_VERSION,
        "kind": run.kind,
        "request_id": run.request_id,
        "usage": usage_to_json(run.usage),
    }
    if isinstance(run.result, EvaluationResult):
        body["result"] = evaluation_to_json(run.result)
    elif isinstance(run.result, DraftResult):
        body["result"] = draft_to_json(run.result)
    else:  # pragma: no cover - defensive; only the two pipelines queue work
        raise PayloadError(f"unsupported result type: {type(run.result).__name__}")
    return _dump(body)


def decode_result(raw: str) -> RunResult:
    """Decode a worker's stored result back into pipeline dataclasses."""
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise PayloadError("job result is not valid JSON") from exc
    if not isinstance(data, dict):
        raise PayloadError("job result is not a JSON object")
    kind = _as_str(data.get("kind"))
    usage = usage_from_json(data.get("usage"))
    request_id = _as_str(data.get("request_id"))
    if kind == KIND_EVALUATE:
        return RunResult(
            kind=kind,
            result=evaluation_from_json(data.get("result")),
            usage=usage,
            request_id=request_id,
        )
    if kind == KIND_DRAFT:
        return RunResult(
            kind=kind,
            result=draft_from_json(data.get("result")),
            usage=usage,
            request_id=request_id,
        )
    raise PayloadError(f"unknown job result kind: {kind!r}")


def payload_page_count(job: EvaluateJob | DraftJob) -> int:
    """Total pages carried by a job payload (for logging / audit parity)."""
    return sum(len(doc.pages) for doc in job.records)
