"""Draft tab: record upload, claim/witness inputs, run orchestration, results.

Rendering + run orchestration only; the pipeline itself is
``app.draft.run_draft`` (unchanged). Audit/logging/error mapping stay with the
view so a failed run surfaces in the same browser session that started it.
"""
from __future__ import annotations

import time
import traceback
from typing import Any

import streamlit as st

from .. import audit as audit_log
from ..agiloop_telemetry import track_goal, track_impression, track_interaction
from ..documents import (
    DRAFT_INTERNAL_MAX_CHARS,
    MAX_OBSERVATIONS_CHARS,
)
from ..draft import DraftResult, grounding_markdown, run_draft
from ..job_payload import DraftJob
from ..logging_config import get_logger
from ..pdf_export import detect_unconfirmed_placeholders, generate_statement_pdf
from ..pipeline_guard import (
    PipelineTimeoutError,
    check_memory_before_run,
    run_with_timeout,
)
from ..profiler import RunProfiler, get_profiler
from ..run_log import run_log_event
from ..shutdown import enter_run, exit_run
from .follow_up import (
    append_follow_up_answers,
    draft_follow_up_questions,
    mark_follow_up_answers_consumed,
    render_follow_up_questions,
)

from . import job_runner
from .ops import render_failure_detail
from .shared import (
    audit_condition_for_slot,
    audit_record_meta,
    check_endpoint_gate,
    check_shutdown_gate,
    format_error_for_user,
    get_llm,
    new_run_request_id,
    progress_widgets,
    records_uploader,
    render_condition_selector_for_slot,
    render_endpoint_preflight_notice,
    render_usage_summary,
    record_watchdog_run,
    reference_suffix,
    report_failure,
)

logger = get_logger("app.views.draft")

# Feature: Final Statement PDF Export
PDF_EXPORT_FEATURE_ID = "0d76d70b-8dd6-4561-a874-f768d5929222"  # final-statement-pdf-export

CLAIM_TYPES = [
    "Service connection (new claim)",
    "Increased rating (worsening condition)",
    "PTSD stressor corroboration",
    "TDIU / individual unemployability",
    "Continuity of symptoms since service",
]

RELATIONSHIPS = [
    "Spouse",
    "Family member",
    "Friend",
    "Coworker / supervisor",
    "Fellow service member",
    "Other",
]


def render_draft_tab() -> None:
    """Render the Draft tab (inputs, run button, cached results)."""
    st.subheader("Step 1 — Upload the veteran's medical records")
    records = records_uploader("draft")
    if records:
        total_pages = sum(len(d.pages) for d in records)
        st.success(
            f"Loaded {len(records)} record file(s), {total_pages:,} page(s): "
            + ", ".join(d.filename for d in records)
        )
        if total_pages > 200:
            st.info(
                "Large record set: chunks are digested in parallel with duplicate pages "
                "skipped, but expect a longer run for a meticulous review."
            )

    render_condition_selector_for_slot("draft")

    st.subheader("Step 2 — Claim details")
    col1, col2 = st.columns(2)
    veteran_name = col1.text_input("Veteran's name", key="draft_vet_name")
    condition = col2.text_input(
        "Condition the statement supports (e.g., PTSD, lumbar strain, tinnitus)",
        key="draft_condition",
    )
    col3, col4 = st.columns(2)
    claim_type = col3.selectbox("Claim type", CLAIM_TYPES, key="draft_claim_type")
    relationship = col4.selectbox("Witness relationship", RELATIONSHIPS, key="draft_rel")

    st.subheader("Step 3 — Witness details")
    col5, col6, col7 = st.columns(3)
    witness_name = col5.text_input("Witness full name", key="draft_witness_name")
    known_since = col6.text_input("Known the veteran since / for", key="draft_known")
    contact_frequency = col7.text_input(
        "How often they see each other (opportunity to observe)", key="draft_freq"
    )
    witnessed_event = st.radio(
        "Did the witness personally see the in-service event happen?",
        ["No", "Yes", "Not applicable"],
        horizontal=True,
        key="draft_witnessed",
    )

    st.subheader("Step 4 — What has the witness observed?")
    observations = st.text_area(
        "Describe everything the witness has personally seen, heard, or experienced "
        "regarding the veteran's condition: symptoms, incidents, changes over time, "
        "impact on work, family and social life. Bullet points are fine — the app will "
        "turn them into a polished, factually grounded statement.",
        height=220,
        key="draft_observations",
    )
    if observations:
        _render_observations_length_guidance(observations)

    if job_runner.queue_mode_active():
        st.caption(
            f"⚙️ This run is processed by a background worker ({job_runner.queue_status_line()}). "
            "You can close this tab — the results will be waiting when you come back."
        )

    # A blocked preflight is kept on screen (with its waiver) from here, above the
    # button that would start the run — see app/views/shared.py.
    render_endpoint_preflight_notice("draft")

    run = st.button("✍️ Draft the statement", type="primary", key="draft_run")
    if run:
        # Mint the correlation id BEFORE validation so every rejection carries a
        # reference the run log can resolve (see app/run_log.py).
        rid = new_run_request_id()
        if not _validate_draft_inputs(records, observations, condition, rid):
            return
        _run_draft_flow(
            rid=rid,
            records=records,
            condition=condition,
            claim_type=claim_type,
            relationship=relationship,
            witness_name=witness_name,
            veteran_name=veteran_name,
            known_since=known_since,
            contact_frequency=contact_frequency,
            witnessed_event=witnessed_event,
            observations=observations,
        )

    # A queued run outlives this browser session, so re-attach to one started
    # earlier (a reload mid-digest would otherwise look like nothing happened).
    job_runner.resume_pending_job("draft", action_label="Drafting")

    cached_draft: Any = st.session_state.get("draft_result")
    if cached_draft is None:
        return
    draft_result: Any = cached_draft

    _render_draft_results(draft_result)


# ------------------------------------------------------------------ input UI
def _run_draft_queued(
    *,
    rid: str,
    records: list,
    condition: str,
    claim_type: str,
    relationship: str,
    witness_name: str,
    veteran_name: str,
    known_since: str,
    contact_frequency: str,
    witnessed_event: str,
    observations: str,
) -> None:
    """Submit the drafting run to a worker and wait for its result (Pattern C).

    The worker writes the audit start/ok/error pair, so this path emits none —
    one audit record per run regardless of where it executed.
    """
    if not check_shutdown_gate("draft"):
        run_log_event(
            "draft", "rejected", request_id=rid,
            error="app shutting down", reason="draining",
        )
        return
    config_error = job_runner.worker_config_error()
    if config_error:
        run_log_event(
            "draft", "rejected", request_id=rid,
            error=config_error, reason="worker_key_missing",
        )
        st.error(config_error)
        return
    _sources, _files, _pages = audit_record_meta("draft", records)
    outcome = job_runner.submit_job(
        slot="draft",
        job=DraftJob(
            records=records,
            witness={
                "name": witness_name.strip(),
                "relationship": relationship,
                "known_since": known_since.strip(),
                "contact_frequency": contact_frequency.strip(),
                "veteran_name": veteran_name.strip(),
                "witnessed_event": witnessed_event,
            },
            observations=observations.strip(),
            condition=condition.strip(),
            claim_type=claim_type,
            request_id=rid,
            record_sources=_sources,
        ),
        request_id=rid,
        condition=(condition.strip()[:120] if condition.strip() else None),
        sources=_sources,
        files=_files,
        pages=_pages,
        action_label="Drafting",
    )
    if outcome is not None and outcome.ok:
        st.success(f"Draft complete — reference `{rid}`.")


def _render_observations_length_guidance(observations: str) -> None:
    n = len(observations)
    st.caption(
        f"Observations length: {n:,} / {MAX_OBSERVATIONS_CHARS:,} characters "
        f"(recommended limit; hard prompt limit {DRAFT_INTERNAL_MAX_CHARS:,})."
    )
    if n > MAX_OBSERVATIONS_CHARS:
        over = n - MAX_OBSERVATIONS_CHARS
        will_truncate = max(0, n - DRAFT_INTERNAL_MAX_CHARS)
        if will_truncate:
            st.warning(
                f"⚠️ Observations are {n:,} characters — {over:,} over the "
                f"{MAX_OBSERVATIONS_CHARS:,} recommended limit. {will_truncate:,} "
                f"characters beyond the {DRAFT_INTERNAL_MAX_CHARS:,} internal limit "
                f"will be truncated and not grounded. Details at the end may be missed. "
                f"Consider shortening or splitting."
            )
        else:
            st.warning(
                f"⚠️ Observations are {n:,} characters — {over:,} over the "
                f"{MAX_OBSERVATIONS_CHARS:,} recommended limit. They will still be "
                f"grounded in full (internal limit {DRAFT_INTERNAL_MAX_CHARS:,}), but "
                f"very long inputs may reduce accuracy."
            )
        st.checkbox(
            f"I understand the observations are {over:,} characters over the limit and "
            "want to proceed anyway (any truncated portion will be noted in the results).",
            key="draft_confirm_oversize",
        )
    elif n > int(MAX_OBSERVATIONS_CHARS * 0.85):
        st.caption(
            f"ℹ️ Approaching the {MAX_OBSERVATIONS_CHARS:,} recommended limit "
            f"({MAX_OBSERVATIONS_CHARS - n:,} remaining before confirmation is required)."
        )


def _validate_draft_inputs(records: list, observations: str, condition: str, rid: str) -> bool:
    """Pre-run validation; shows the specific error and returns False when invalid.

    Every rejection is recorded in the persistent run log so the reference shown
    in future error messages is always correlatable, even for pre-pipeline
    failures that never reach the audit log.
    """
    if not records:
        msg = "Upload at least one medical record file first."
        run_log_event("draft", "rejected", request_id=rid, error=msg, reason="no_records")
        # Shown with the id the run log recorded, so a rejection the user quotes
        # is findable even though it never reached the audit log.
        st.error(f"{msg}{reference_suffix(rid)}")
        return False
    if not observations.strip() or not condition.strip():
        msg = "Enter the condition and the witness's observations."
        run_log_event(
            "draft", "rejected", request_id=rid, error=msg,
            reason="missing_observations" if not observations.strip() else "missing_condition",
        )
        st.error(f"{msg}{reference_suffix(rid)}")
        return False
    if len(observations) > MAX_OBSERVATIONS_CHARS and not st.session_state.get(
        "draft_confirm_oversize"
    ):
        over = len(observations) - MAX_OBSERVATIONS_CHARS
        will_truncate = max(0, len(observations) - DRAFT_INTERNAL_MAX_CHARS)
        msg = (
            f"Observations are {len(observations):,} characters — {over:,} over the "
            f"{MAX_OBSERVATIONS_CHARS:,} limit. Check the confirmation box above to proceed, "
            "or shorten/split the observations."
        )
        if will_truncate:
            msg += f" {will_truncate:,} characters would be truncated and not grounded."
        run_log_event(
            "draft", "rejected", request_id=rid, error=msg,
            reason="observations_oversize", observations_chars=len(observations),
        )
        st.error(f"{msg}{reference_suffix(rid)}")
        return False
    return True


# -------------------------------------------------------------- run pipeline
def _run_draft_flow(
    *,
    rid: str,
    records: list,
    condition: str,
    claim_type: str,
    relationship: str,
    witness_name: str,
    veteran_name: str,
    known_since: str,
    contact_frequency: str,
    witnessed_event: str,
    observations: str,
) -> None:
    """Run the pipeline with the pre-minted run id; persist the result."""
    # Before anything is spent: a configuration whose every call is rejected is
    # detectable in one request. Covers the queued path too — this is the only
    # entry point, and the worker never re-checks inside its own session-less run.
    if not check_endpoint_gate("draft", log_action="draft", request_id=rid):
        return
    if job_runner.queue_mode_active():
        _run_draft_queued(
            rid=rid,
            records=records,
            condition=condition,
            claim_type=claim_type,
            relationship=relationship,
            witness_name=witness_name,
            veteran_name=veteran_name,
            known_since=known_since,
            contact_frequency=contact_frequency,
            witnessed_event=witnessed_event,
            observations=observations,
        )
        return
    llm = get_llm()
    if llm is None:
        run_log_event("draft", "rejected", request_id=rid, error="LLM client unavailable", reason="llm_unavailable")
        return
    # Graceful shutdown gate — reject new work when draining (SIGTERM/SIGINT).
    if not check_shutdown_gate("draft"):
        run_log_event("draft", "rejected", request_id=rid, error="app shutting down", reason="draining")
        return
    if not enter_run():
        run_log_event("draft", "rejected", request_id=rid, error="app shutting down", reason="draining")
        st.error(
            "The app is shutting down — no new draft runs can start right now. "
            "Please try again in a moment."
        )
        return

    # Audit: start — metadata only, never observations/record text.
    _audit_sources_d, _audit_files_d, _audit_pages_d = audit_record_meta("draft", records)
    _audit_condition_d = (
        condition.strip()[:120]
        if condition and condition.strip()
        else audit_condition_for_slot("draft")
    )
    audit_log.audit_draft_start(
        request_id=rid,
        condition=_audit_condition_d or None,
        record_sources=_audit_sources_d or None,
        record_files=_audit_files_d,
        record_pages=_audit_pages_d,
    )

    logger.info(
        "draft run start observations_chars=%d pages=%d condition=%s",
        len(observations.strip()),
        sum(len(d.pages) for d in records),
        condition.strip()[:60] if condition.strip() else "-",
        extra={"request_id": rid, "phase": "draft", "status": "start"},
    )
    run_log_event(
        "draft", "start", request_id=rid, pages=sum(len(d.pages) for d in records),
        observations_chars=len(observations.strip()),
    )
    witness = {
        "name": witness_name.strip(),
        "relationship": relationship,
        "known_since": known_since.strip(),
        "contact_frequency": contact_frequency.strip(),
        "veteran_name": veteran_name.strip(),
        "witnessed_event": witnessed_event,
    }
    bar, update = progress_widgets(llm, request_id=rid)
    _profiler_run = (
        RunProfiler(action="draft", request_id=rid, run_start_mono=time.monotonic())
        if get_profiler()
        else None
    )
    t0 = time.perf_counter()
    try:
        check_memory_before_run()
        result = run_with_timeout(
            run_draft,
            llm,
            records,
            witness,
            append_follow_up_answers(observations.strip(), slot="draft"),
            condition.strip(),
            claim_type,
            progress=update,
        )
    except MemoryError as mem_exc:
        _handle_draft_abort(rid, "error", mem_exc, t0, _audit_condition_d, _audit_sources_d, _audit_files_d, _audit_pages_d)
        st.error(f"Draft aborted: {format_error_for_user(mem_exc, rid)}")
        render_failure_detail(rid)
        return
    except PipelineTimeoutError as timeout_exc:
        _handle_draft_abort(rid, "timeout", timeout_exc, t0, _audit_condition_d, _audit_sources_d, _audit_files_d, _audit_pages_d)
        st.error(f"Draft aborted: {format_error_for_user(timeout_exc, rid)}")
        render_failure_detail(rid)
        return
    except Exception as exc:  # noqa: BLE001
        _handle_draft_error(rid, exc, t0, _audit_condition_d, _audit_sources_d, _audit_files_d, _audit_pages_d)
        st.error(f"Drafting failed: {format_error_for_user(exc, rid)}")
        render_failure_detail(rid)
        return
    finally:
        exit_run()
    _finish_draft_run(rid, llm, result, t0, _profiler_run, _audit_condition_d, _audit_sources_d, _audit_files_d, _audit_pages_d)


def _handle_draft_abort(
    rid: str,
    kind: str,
    exc: BaseException,
    t0: float,
    condition: str | None,
    sources: list[str] | None,
    files: int,
    pages: int,
) -> None:
    """Log + audit a MemoryError / timeout abort (bar cleanup is the caller's)."""
    duration_ms = int((time.perf_counter() - t0) * 1000)
    run_log_event(
        "draft", kind, request_id=rid, duration_ms=duration_ms,
        error=str(exc), error_class=type(exc).__name__,
    )
    logger.error(
        "draft run %s duration_ms=%d error=%s",
        kind,
        duration_ms,
        str(exc),
        exc_info=exc,
        extra={
            "request_id": rid,
            "phase": "draft",
            "status": kind,
            "duration_ms": duration_ms,
            "error_class": type(exc).__name__,
        },
    )
    audit_log.audit_draft_error(
        request_id=rid,
        duration_ms=duration_ms,
        error=exc,
        condition=condition,
        record_sources=sources,
        record_files=files,
        record_pages=pages,
    )


def _handle_draft_error(
    rid: str,
    exc: Exception,
    t0: float,
    condition: str | None,
    sources: list[str] | None,
    files: int,
    pages: int,
) -> None:
    """Log + audit a generic drafting failure."""
    duration_ms = int((time.perf_counter() - t0) * 1000)
    run_log_event(
        "draft", "error", request_id=rid, duration_ms=duration_ms,
        error=f"{type(exc).__name__}: {exc}", error_class=type(exc).__name__,
        traceback=traceback.format_exc(limit=8),
    )
    logger.error(
        "draft run error duration_ms=%d error=%s",
        duration_ms,
        f"{type(exc).__name__}: {exc}",
        exc_info=exc,
        extra={
            "request_id": rid,
            "phase": "draft",
            "status": "error",
            "duration_ms": duration_ms,
            "error_class": type(exc).__name__,
        },
    )
    audit_log.audit_draft_error(
        request_id=rid,
        duration_ms=duration_ms,
        error=exc,
        condition=condition,
        record_sources=sources,
        record_files=files,
        record_pages=pages,
    )


def _finish_draft_run(
    rid: str,
    llm: Any,
    result: DraftResult,
    t0: float,
    profiler_run: RunProfiler | None,
    condition: str | None,
    sources: list[str] | None,
    files: int,
    pages: int,
) -> None:
    """Log/audit success and stash the result for the results renderer."""
    duration_ms = int((time.perf_counter() - t0) * 1000)
    total = llm.usage.totals()
    # Which endpoint(s) actually served this run (see evaluate_view).
    _draft_endpoints = llm.usage.endpoints_used()
    logger.info(
        "draft run done duration_ms=%d calls=%d tokens_in=%d tokens_out=%d",
        duration_ms,
        total.calls,
        total.prompt_tokens,
        total.completion_tokens,
        extra={
            "request_id": rid,
            "phase": "draft",
            "status": "ok",
            "duration_ms": duration_ms,
            "calls": total.calls,
            "prompt_tokens": total.prompt_tokens,
            "completion_tokens": total.completion_tokens,
        },
    )
    # Audit: ok — only small outcome classification, never draft text.
    try:
        _outcome_d: dict[str, Any] = {
            "draft_chars": len(
                getattr(result, "output_statement", "") or getattr(result, "draft", "") or ""
            ),
            "grounding_items": len(getattr(result, "grounding", {}) or {}),
        }
    except Exception:  # noqa: BLE001
        _outcome_d = {}
    audit_log.audit_draft_ok(
        request_id=rid,
        duration_ms=duration_ms,
        condition=condition,
        record_sources=sources,
        record_files=files,
        record_pages=pages,
        outcome=_outcome_d or None,
        llm_endpoints=_draft_endpoints,
    )
    run_log_event(
        "draft",
        "ok",
        request_id=rid,
        duration_ms=duration_ms,
        endpoints=",".join(_draft_endpoints),
        **_outcome_d,
    )
    # Profiler: emit per-run timing breakdown.
    if profiler_run is not None:
        profiler_run.run_end_mono = time.monotonic()
        profiler_run.emit()
        get_profiler().record_run(profiler_run)
    st.session_state.draft_result = result
    st.session_state.draft_usage = llm.usage
    st.session_state.draft_request_id = rid
    mark_follow_up_answers_consumed("draft")
    record_watchdog_run(llm.usage)


# ------------------------------------------------------------------ results UI
def _render_pdf_export(statement_text: str) -> None:
    """Render the "export final statement as PDF" download button.

    Fires the telemetry required for the Final Statement PDF Export feature
    (`.implement/FEATURE-BRIEF.md`): one `impression` per rendered panel, one
    `interaction` + `goal` per actual download click. No statement/condition
    text is ever attached to a telemetry payload — only booleans/labels.
    """
    impression_key = "pdf_export_impression_sent_draft"
    if not st.session_state.get(impression_key):
        try:
            track_impression(PDF_EXPORT_FEATURE_ID, entry_point="draft")
        except Exception:  # noqa: BLE001 - telemetry must never break the UI
            pass
        st.session_state[impression_key] = True

    condition = str(st.session_state.get("draft_condition", "") or "")
    witness_role = str(st.session_state.get("draft_rel", "") or "")
    try:
        pdf_bytes = generate_statement_pdf(statement_text, condition, witness_role)
    except Exception as exc:  # noqa: BLE001 - PDF generation is best-effort in the UI
        st.error(
            report_failure(
                f"Could not generate the PDF export: {exc}",
                phase="draft_pdf_export",
                exc=exc,
            )
        )
        return

    has_placeholders = detect_unconfirmed_placeholders(statement_text)
    clicked = st.download_button(
        "📄 Export final statement as PDF",
        data=pdf_bytes,
        file_name="VA_Statement.pdf",
        mime="application/pdf",
        key="pdf_export_button_draft",
    )
    if clicked:
        try:
            track_interaction(
                PDF_EXPORT_FEATURE_ID,
                action="pdf_export_click",
                has_unconfirmed_placeholders=has_placeholders,
            )
            track_goal(
                PDF_EXPORT_FEATURE_ID,
                "final statement exported as PDF",
                view="draft",
            )
        except Exception:  # noqa: BLE001 - telemetry must never break the UI
            pass


def _render_draft_results(draft_result: Any) -> None:
    render_usage_summary(st.session_state.get("draft_usage"))

    if getattr(draft_result, "truncation_warning", ""):
        st.warning(
            f"⚠️ {draft_result.truncation_warning} (input was {draft_result.input_chars:,} chars; "
            f"{draft_result.truncated_chars:,} truncated). Review the grounding section for details."
        )

    st.divider()
    st.subheader("📋 Draft Results")

    with st.expander("Grounding analysis — how the draft ties to the records", expanded=True):
        st.markdown(grounding_markdown(draft_result))

    render_follow_up_questions(
        slot="draft",
        source_id=str(st.session_state.get("draft_request_id", "") or ""),
        questions=draft_follow_up_questions(draft_result),
        empty_message=(
            "No follow-up questions are needed — the grounding analysis found no uncovered "
            "applicable checklist topics."
        ),
        next_run_label="draft",
    )

    if draft_result.review_issues:
        with st.expander("Self-review findings (fixed in the final version)"):
            for issue in draft_result.review_issues:
                st.write(f"- {issue}")

    st.markdown("### Final statement (editable)")
    st.caption(
        "Review every bracketed [Confirm: ...] placeholder and resolve it before signing. "
        "Submit on VA Form 21-10210 (one form per witness)."
    )
    edited = st.text_area(
        "Statement", value=draft_result.output_statement, height=460, key="draft_edited"
    )
    col_a, col_b = st.columns(2)
    col_a.download_button(
        "⬇️ Download statement (.txt)",
        data=edited.encode("utf-8"),
        file_name="lay_statement_draft.txt",
        mime="text/plain",
    )
    col_b.download_button(
        "⬇️ Download statement (.md)",
        data=edited.encode("utf-8"),
        file_name="lay_statement_draft.md",
        mime="text/markdown",
    )
    _render_pdf_export(edited)

    if draft_result.digest:
        with st.expander("Medical record digest used for grounding"):
            st.caption(
                f"{len(draft_result.digest.facts):,} facts extracted from "
                f"{draft_result.digest.pages_reviewed:,} pages "
                f"({draft_result.digest.chunks_reviewed} chunks, "
                f"{draft_result.digest.duplicates_skipped} duplicate page(s) skipped)"
            )
            st.write(draft_result.digest.summary)
            st.code(draft_result.digest.timeline_text()[:20000], language=None)

        # Timeline view
        st.divider()
        st.subheader("📅 Medical Record Timeline")
        st.caption(
            "View all medical events extracted from your records, sorted chronologically. "
            "Use filters to focus on specific time periods or event types."
        )

        # Add timeline tab
        timeline_tab, _ = st.columns([1, 3])
        with timeline_tab:
            if st.button("🗓️ Open Full Timeline", type="secondary"):
                # Store digest in session state for the timeline tab
                st.session_state.timeline_digest = draft_result.digest
                st.switch_page("/timeline")

        # Compact inline timeline preview
        from .timeline import render_timeline_in_results
        render_timeline_in_results(draft_result.digest)
