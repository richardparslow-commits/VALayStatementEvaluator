"""Evaluate tab: statement input, record upload, run orchestration, results.

Rendering + run orchestration only; the pipeline itself is
``app.evaluate.run_evaluation`` (unchanged). Audit/logging/error mapping stay
with the view so a failed run surfaces in the same browser session that
started it.
"""
from __future__ import annotations

import time
import traceback
from typing import Any

import pandas as pd
import streamlit as st

try:  # Chart library — guarded like the Redis queue backend (app/job_queue.py): a
    # slim image or a stale dev venv that lacks it still loads the Evaluate tab,
    # and the timeline panel degrades to its event list instead of taking the tab
    # down at import time.
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - the degradation path is covered by tests
    go = None

from .. import audit as audit_log
from .. import knowledge_currency as currency
from ..agiloop_telemetry import (
    track_feature_error,
    track_goal,
    track_impression,
    track_interaction,
)
from ..evaluate import (
    DIMENSION_LABELS,
    VERDICTS,
    build_evidence_dashboard,
    compute_score_band,
    coverage_lines,
    rubric_and_positive_sources,
    run_evaluation,
)
from ..exporter import export_facts, filter_facts
from ..job_payload import EvaluateJob
from ..config import Settings, load_settings
from ..logging_config import get_logger, get_request_id
from ..documents import (
    EVALUATE_INTERNAL_MAX_CHARS,
    MAX_STATEMENT_CHARS,
)
from ..medical_review import build_timeline_data
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
    evaluate_follow_up_questions,
    mark_follow_up_answers_consumed,
    render_follow_up_questions,
)

from . import job_runner
from .aa_form import collect_aa_answers, render_aa_intake_wizard
from .ops import render_failure_detail
from .shared import (
    FEATURE_ID,
    audit_condition_for_slot,
    audit_record_meta,
    check_endpoint_gate,
    check_shutdown_gate,
    check_upload_limits,
    ensure_request_id,
    extract_uploads,
    format_error_for_user,
    get_llm,
    new_run_request_id,
    progress_widgets,
    records_uploader,
    render_condition_selector_for_slot,
    render_endpoint_preflight_notice,
    render_record_search,
    render_usage_summary,
    record_watchdog_run,
    reference_suffix,
    report_failure,
)

logger = get_logger("app.views.evaluate")

# Feature: Final Statement PDF Export
PDF_EXPORT_FEATURE_ID = "0d76d70b-8dd6-4561-a874-f768d5929222"  # final-statement-pdf-export

# Feature: Evidence Strength Dashboard
EVIDENCE_DASHBOARD_FEATURE_ID = "b25a523d-b974-43e1-a554-374bbdebb01d"  # evidence-strength-dashboard

# Feature: Fact Citation Exporter
EXPORT_FACTS_FEATURE_ID = "051bb638-ac1c-40cf-95f5-164779b4382c"  # fact-citation-exporter

# Feature: Medical Event Timeline Visualization
TIMELINE_FEATURE_ID = "222efbdb-50be-4ff7-a384-1595d543c842"  # medical-event-timeline-visualization

_TIMELINE_CATEGORY_COLORS: dict[str, str] = {
    "diagnostic": "#2563eb",
    "treatment": "#16a34a",
    "other": "#9333ea",
}
_TIMELINE_FILTER_LABELS: dict[str, str] = {
    "All": "all",
    "Diagnostic only": "diagnostic",
    "Treatment only": "treatment",
}

# Feature: Statement Effectiveness Score & Improvement Recommendations
EFFECTIVENESS_SCORE_FEATURE_ID = "94104045-aa12-4018-95c6-e6912e659803"  # statement-effectiveness-score-improvement-recommendations

_SCORE_BAND_DISPLAY = {
    "green": ("🟢", "Strong statement"),
    "yellow": ("🟡", "Needs improvement"),
    "red": ("🔴", "Weak — act on recommendations below"),
}


def render_evaluate_tab() -> None:
    """Render the Evaluate tab (input, run button, cached results)."""
    st.subheader("Step 1 — Provide the lay statement")
    mode = st.radio(
        "Statement source", ["Upload file", "Paste text"], key="eval_mode", horizontal=True
    )
    statement_text = ""
    if mode == "Paste text":
        statement_text = st.text_area(
            "Paste the full lay/witness statement", height=260, key="eval_paste"
        )
    else:
        files = st.file_uploader(
            "Upload the statement (PDF, TXT, MD, DOCX)",
            type=["pdf", "txt", "md", "docx"],
            key="eval_statement_file",
        )
        if files is not None:
            accepted, rejections = check_upload_limits([files])
            for msg in rejections:
                st.warning(
                    report_failure(
                        msg,
                        phase="statement_upload_limits",
                        severity="warning",
                        once=True,
                    )
                )
            if rejections:
                docs = []
            else:
                docs = extract_uploads([files], "eval_statement")
            if docs:
                statement_text = docs[0].full_text

    if statement_text:
        _render_statement_length_guidance(statement_text)

    st.subheader("Step 2 — Provide the medical records")
    records = records_uploader("eval")
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
        render_record_search("eval", records)

    render_condition_selector_for_slot("eval")

    # Optional structured A&A intake — the same 16-question wizard the Draft
    # tab uses. Answers feed the recommendations phase: the model recommends
    # the statement cover care observations the witness has already attested
    # to, which is usually the highest-impact edit available.
    st.subheader("Aid & Attendance intake (optional)")
    render_aa_intake_wizard("eval")

    if job_runner.queue_mode_active():
        st.caption(
            f"⚙️ This run is processed by a background worker ({job_runner.queue_status_line()}). "
            "You can close this tab — the results will be waiting when you come back."
        )

    # A blocked preflight is kept on screen (with its waiver) from here, above the
    # button that would start the run — see app/views/shared.py.
    render_endpoint_preflight_notice("evaluation")

    run = st.button("🔍 Run exhaustive evaluation", type="primary", key="eval_run")
    if run:
        if not _validate_evaluate_inputs(statement_text, records):
            return
        _run_evaluation_flow(statement_text, records, collect_aa_answers("eval"))

    # A queued run outlives this browser session, so re-attach to one started
    # earlier (a reload mid-digest would otherwise look like nothing happened).
    job_runner.resume_pending_job("eval", action_label="Evaluation")
    # If the session state was lost entirely (browser restart, pod failover),
    # offer a recovery form so the user can paste their reference to recover.
    job_runner.render_recovery_form("eval", action_label="Evaluation")

    cached_eval: Any = st.session_state.get("eval_result")
    if cached_eval is None:
        return
    eval_result: Any = cached_eval

    _render_evaluation_results(eval_result)


# ------------------------------------------------------------------ input UI
def _render_statement_length_guidance(statement_text: str) -> None:
    n = len(statement_text)
    st.caption(
        f"Statement length: {n:,} / {MAX_STATEMENT_CHARS:,} characters "
        f"(recommended limit; hard prompt limit {EVALUATE_INTERNAL_MAX_CHARS:,})."
    )
    if n > MAX_STATEMENT_CHARS:
        over = n - MAX_STATEMENT_CHARS
        will_truncate = max(0, n - EVALUATE_INTERNAL_MAX_CHARS)
        if will_truncate:
            st.warning(
                f"⚠️ Statement is {n:,} characters — {over:,} over the {MAX_STATEMENT_CHARS:,} "
                f"recommended limit. {will_truncate:,} characters beyond the "
                f"{EVALUATE_INTERNAL_MAX_CHARS:,} internal prompt limit will be "
                f"truncated and not analyzed. Claims at the end (e.g., family impact, "
                f"caregiver necessity) may be missed. Consider splitting the statement "
                f"into smaller parts or shortening it."
            )
        else:
            st.warning(
                f"⚠️ Statement is {n:,} characters — {over:,} over the {MAX_STATEMENT_CHARS:,} "
                f"recommended limit. It will still be analyzed in full (internal limit "
                f"{EVALUATE_INTERNAL_MAX_CHARS:,}), but very long statements may reduce "
                f"model accuracy. Consider shortening for best results."
            )
        st.checkbox(
            f"I understand the statement is {over:,} characters over the limit and "
            "want to proceed anyway (any truncated portion will be noted in the report).",
            key="eval_confirm_oversize",
        )
    elif n > int(MAX_STATEMENT_CHARS * 0.85):
        st.caption(
            f"ℹ️ Approaching the {MAX_STATEMENT_CHARS:,} character recommended limit "
            f"({MAX_STATEMENT_CHARS - n:,} remaining before a confirmation is required)."
        )


def _validate_evaluate_inputs(statement_text: str, records: list) -> bool:
    """Pre-run validation; shows the specific error and returns False when invalid."""
    # Minted rather than defaulted to "-": each rejection below is written to the
    # run log under this id, and an id the user cannot see is not a reference.
    rid = ensure_request_id()
    if not statement_text.strip():
        msg = "Provide the lay statement first (upload or paste)."
        run_log_event("evaluate", "rejected", request_id=rid, error=msg, reason="no_statement")
        st.error(f"{msg}{reference_suffix(rid)}")
        return False
    if len(statement_text) > MAX_STATEMENT_CHARS and not st.session_state.get(
        "eval_confirm_oversize"
    ):
        over = len(statement_text) - MAX_STATEMENT_CHARS
        will_truncate = max(0, len(statement_text) - EVALUATE_INTERNAL_MAX_CHARS)
        msg = (
            f"Statement is {len(statement_text):,} characters — {over:,} over the "
            f"{MAX_STATEMENT_CHARS:,} limit. Check the confirmation box above to proceed, "
            "or split/shorten the statement."
        )
        if will_truncate:
            msg += f" {will_truncate:,} characters would be truncated and not analyzed."
        run_log_event(
            "evaluate", "rejected", request_id=rid, error=msg,
            reason="statement_oversize", statement_chars=len(statement_text),
        )
        st.error(f"{msg}{reference_suffix(rid)}")
        return False
    if not records:
        msg = "Upload at least one medical record file."
        run_log_event("evaluate", "rejected", request_id=rid, error=msg, reason="no_records")
        st.error(f"{msg}{reference_suffix(rid)}")
        return False
    return True


# -------------------------------------------------------------- run pipeline
def _run_evaluation_flow(statement_text: str, records: list, witness: dict[str, str] | None = None) -> None:
    """Mint a run id, gate shutdown, run the pipeline, persist the result.

    *witness* carries the structured A&A intake answers (``aa_*`` keys); it
    flows to the recommendations phase on both the in-process and queued
    paths. ``None``/``{}`` keeps every prompt byte-identical to the
    pre-intake pipeline.
    """
    # Before anything is spent: a configuration whose every call is rejected is
    # detectable in one request. This is the only entry point into the pipeline, so
    # the queued path is covered too, and the worker never re-checks inside its own
    # session-less run.
    if not check_endpoint_gate("evaluation", log_action="evaluate"):
        return
    if job_runner.queue_mode_active():
        _run_evaluation_queued(statement_text, records, witness or {})
        return
    rid = new_run_request_id()
    llm = get_llm()
    if llm is None:
        run_log_event("evaluate", "rejected", request_id=rid, error="LLM client unavailable", reason="llm_unavailable")
        return
    # Graceful shutdown gate — reject new work when draining (SIGTERM/SIGINT).
    if not check_shutdown_gate("evaluation"):
        run_log_event("evaluate", "rejected", request_id=rid, error="app shutting down", reason="draining")
        return
    if not enter_run():
        run_log_event("evaluate", "rejected", request_id=rid, error="app shutting down", reason="draining")
        st.error(
            "The app is shutting down — no new evaluation runs can start right now. "
            "Please try again in a moment."
        )
        return

    # Audit: start — metadata only, never statement/record text.
    _audit_sources, _audit_files, _audit_pages = audit_record_meta("eval", records)
    _audit_condition = audit_condition_for_slot("eval")
    audit_log.audit_evaluate_start(
        request_id=rid,
        condition=_audit_condition or None,
        record_sources=_audit_sources or None,
        record_files=_audit_files,
        record_pages=_audit_pages,
    )

    logger.info(
        "evaluate run start statement_chars=%d pages=%d",
        len(statement_text.strip()),
        sum(len(d.pages) for d in records),
        extra={"request_id": rid, "phase": "evaluate", "status": "start"},
    )
    run_log_event(
        "evaluate", "start", request_id=rid, pages=sum(len(d.pages) for d in records),
        statement_chars=len(statement_text.strip()),
    )
    bar, update = progress_widgets(llm, request_id=rid)
    _profiler_run = (
        RunProfiler(action="evaluate", request_id=rid, run_start_mono=time.monotonic())
        if get_profiler()
        else None
    )
    t0 = time.perf_counter()
    try:
        check_memory_before_run()
        result = run_with_timeout(
            run_evaluation,
            llm,
            append_follow_up_answers(statement_text.strip(), slot="eval"),
            records,
            progress=update,
            witness=witness or {},
        )
    except MemoryError as mem_exc:
        bar.empty()
        duration_ms = int((time.perf_counter() - t0) * 1000)
        run_log_event(
            "evaluate", "error", request_id=rid, duration_ms=duration_ms,
            error=f"{type(mem_exc).__name__}: {mem_exc}", error_class=type(mem_exc).__name__,
            traceback=traceback.format_exc(limit=8),
        )
        logger.error(
            "evaluate run error duration_ms=%d error=%s",
            duration_ms,
            f"{type(mem_exc).__name__}: {mem_exc}",
            exc_info=mem_exc,
            extra={
                "request_id": rid,
                "phase": "evaluate",
                "status": "error",
                "duration_ms": duration_ms,
                "error_class": type(mem_exc).__name__,
            },
        )
        st.error(f"Evaluation aborted: {format_error_for_user(mem_exc, rid)}")
        render_failure_detail(rid)
        return
    except PipelineTimeoutError as timeout_exc:
        bar.empty()
        duration_ms = int((time.perf_counter() - t0) * 1000)
        run_log_event(
            "evaluate", "timeout", request_id=rid, duration_ms=duration_ms,
            error=str(timeout_exc), error_class="PipelineTimeoutError",
        )
        logger.error(
            "evaluate run timeout duration_ms=%d error=%s",
            duration_ms,
            str(timeout_exc),
            extra={
                "request_id": rid,
                "phase": "evaluate",
                "status": "timeout",
                "duration_ms": duration_ms,
                "error_class": "PipelineTimeoutError",
            },
        )
        st.error(f"Evaluation aborted: {format_error_for_user(timeout_exc, rid)}")
        render_failure_detail(rid)
        return
    except Exception as exc:  # noqa: BLE001
        bar.empty()
        duration_ms = int((time.perf_counter() - t0) * 1000)
        run_log_event(
            "evaluate", "error", request_id=rid, duration_ms=duration_ms,
            error=f"{type(exc).__name__}: {exc}", error_class=type(exc).__name__,
            traceback=traceback.format_exc(limit=8),
        )
        logger.error(
            "evaluate run error duration_ms=%d error=%s",
            duration_ms,
            f"{type(exc).__name__}: {exc}",
            exc_info=exc,
            extra={
                "request_id": rid,
                "phase": "evaluate",
                "status": "error",
                "duration_ms": duration_ms,
                "error_class": type(exc).__name__,
            },
        )
        audit_log.audit_evaluate_error(
            request_id=rid,
            duration_ms=duration_ms,
            error=exc,
            condition=_audit_condition or None,
            record_sources=_audit_sources or None,
            record_files=_audit_files,
            record_pages=_audit_pages,
        )
        st.error(f"Evaluation failed: {format_error_for_user(exc, rid)}")
        render_failure_detail(rid)
        return
    except BaseException as ctrl:  # noqa: BLE001 - Streamlit control flow (see comment)
        # Last clause on purpose: ``RerunException``/``StopException`` derive from
        # ``BaseException``, not ``Exception``, so the handlers above never see
        # them — a widget interaction or session teardown during a run unwinds
        # the script run *mid-pipeline*, leaving a lone audit ``start`` event, no
        # results, and no traceable completion for the reference shown on screen.
        # Record the interruption, then re-raise so Streamlit still handles it.
        bar.empty()
        duration_ms = int((time.perf_counter() - t0) * 1000)
        run_log_event(
            "evaluate", "interrupted", request_id=rid, duration_ms=duration_ms,
            error=f"{type(ctrl).__name__}: script run torn down mid-pipeline",
            error_class=type(ctrl).__name__,
        )
        logger.warning(
            "evaluate run interrupted duration_ms=%d error=%s",
            duration_ms,
            type(ctrl).__name__,
            extra={
                "request_id": rid,
                "phase": "evaluate",
                "status": "interrupted",
                "duration_ms": duration_ms,
                "error_class": type(ctrl).__name__,
            },
        )
        audit_log.audit_evaluate_error(
            request_id=rid,
            duration_ms=duration_ms,
            error=ctrl,
            condition=_audit_condition or None,
            record_sources=_audit_sources or None,
            record_files=_audit_files,
            record_pages=_audit_pages,
        )
        raise
    finally:
        exit_run()
    duration_ms = int((time.perf_counter() - t0) * 1000)
    total = llm.usage.totals()
    # Which endpoint(s) actually served this run. Stamped on the audit record and
    # the run log, because a failover changes who wrote the output.
    _endpoints = llm.usage.endpoints_used()
    logger.info(
        "evaluate run done duration_ms=%d calls=%d tokens_in=%d tokens_out=%d",
        duration_ms,
        total.calls,
        total.prompt_tokens,
        total.completion_tokens,
        extra={
            "request_id": rid,
            "phase": "evaluate",
            "status": "ok",
            "duration_ms": duration_ms,
            "calls": total.calls,
            "prompt_tokens": total.prompt_tokens,
            "completion_tokens": total.completion_tokens,
        },
    )
    # Audit: ok — only classifications/counters, never statement text.
    try:
        _outcome: dict[str, Any] = {
            "claims": len(getattr(result, "claims", []) or []),
            "contradictions": int(getattr(result, "contradiction_count", 0) or 0),
            "overall_rating": str(getattr(result, "overall_rating", "") or ""),
        }
    except Exception:  # noqa: BLE001
        _outcome = {}
    audit_log.audit_evaluate_ok(
        request_id=rid,
        duration_ms=duration_ms,
        condition=_audit_condition or None,
        record_sources=_audit_sources or None,
        record_files=_audit_files,
        record_pages=_audit_pages,
        outcome=_outcome or None,
        llm_endpoints=_endpoints,
    )
    run_log_event(
        "evaluate",
        "ok",
        request_id=rid,
        duration_ms=duration_ms,
        endpoints=",".join(_endpoints),
        **_outcome,
    )
    if _is_empty_analysis(result):
        # Completed, but the model handed back nothing usable (e.g. ``{}`` for
        # every phase). Log it once here — the results panel re-renders on every
        # rerun and must not emit duplicate events.
        logger.warning(
            "evaluate run returned an empty analysis calls=%d claims=0 scores=0",
            total.calls,
            extra={
                "request_id": rid,
                "phase": "evaluate",
                "status": "empty",
                "duration_ms": duration_ms,
                "llm_calls": total.calls,
            },
        )
        run_log_event(
            "evaluate", "empty", request_id=rid, duration_ms=duration_ms,
            llm_calls=total.calls, claims=0, scores=0,
        )
    bar.empty()
    # Profiler: emit per-run timing breakdown.
    if _profiler_run is not None:
        _profiler_run.run_end_mono = time.monotonic()
        _profiler_run.emit()
        get_profiler().record_run(_profiler_run)
    st.session_state.eval_result = result
    st.session_state.eval_usage = llm.usage
    st.session_state.eval_request_id = rid
    mark_follow_up_answers_consumed("eval")
    record_watchdog_run(llm.usage)


# ------------------------------------------------------------------ results UI
def _run_evaluation_queued(
    statement_text: str, records: list, witness: dict[str, str] | None = None
) -> None:
    """Submit the run to a worker and wait for its result (Pattern C).

    No audit start is emitted here: the worker writes the start/ok/error pair so
    a run produces exactly one audit record whether it ran in-process or on a
    worker. The web pod only records that the work was handed off.
    """
    rid = new_run_request_id()
    if not check_shutdown_gate("evaluation"):
        run_log_event(
            "evaluate", "rejected", request_id=rid,
            error="app shutting down", reason="draining",
        )
        return
    config_error = job_runner.worker_config_error()
    if config_error:
        run_log_event(
            "evaluate", "rejected", request_id=rid,
            error=config_error, reason="worker_key_missing",
        )
        st.error(config_error)
        return
    _sources, _files, _pages = audit_record_meta("eval", records)
    _condition = audit_condition_for_slot("eval")
    outcome = job_runner.submit_job(
        slot="eval",
        job=EvaluateJob(
            statement_text=statement_text.strip(),
            records=records,
            request_id=rid,
            record_sources=_sources,
            witness=dict(witness or {}),
        ),
        request_id=rid,
        condition=_condition,
        sources=_sources,
        files=_files,
        pages=_pages,
        action_label="Evaluation",
    )
    if outcome is not None and outcome.ok:
        st.success(f"Evaluation complete — reference `{rid}`.")


def _result_reference() -> str:
    """Reference id of the run that produced the cached results, if known."""
    rid_raw: Any = st.session_state.get("eval_request_id", "")
    return rid_raw if isinstance(rid_raw, str) else ""


def _is_empty_analysis(result: Any) -> bool:
    """True when a completed run carried no usable analysis at all.

    An endpoint/model that accepts the request but answers every phase with
    empty JSON yields zero claims, zero verifications, and no rubric scores.
    Without this check the results panel renders a calm but meaningless
    "Not scored" report that looks exactly like a real verdict — and, because
    the panel is re-rendered from session state on every rerun, it also looks
    like a run that finished instantly.
    """
    return not (
        getattr(result, "claims", None)
        or getattr(result, "verifications", None)
        or getattr(result, "scores", None)
    )


def _render_pdf_export(statement_text: str, *, entry_point: str) -> None:
    """Render the "export final statement as PDF" download button.

    Fires the telemetry required for the Final Statement PDF Export feature
    (`.implement/FEATURE-BRIEF.md`): one `impression` per rendered panel, one
    `interaction` + `goal` per actual download click. No statement/condition
    text is ever attached to a telemetry payload — only booleans/labels.
    """
    impression_key = f"pdf_export_impression_sent_{entry_point}"
    if not st.session_state.get(impression_key):
        try:
            track_impression(PDF_EXPORT_FEATURE_ID, entry_point=entry_point)
        except Exception:  # noqa: BLE001 - telemetry must never break the UI
            pass
        st.session_state[impression_key] = True

    condition = audit_condition_for_slot(entry_point) or ""
    try:
        pdf_bytes = generate_statement_pdf(statement_text, condition, witness_role="")
    except Exception as exc:  # noqa: BLE001 - PDF generation is best-effort in the UI
        st.error(
            report_failure(
                f"Could not generate the PDF export: {exc}",
                phase="evaluate_pdf_export",
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
        key=f"pdf_export_button_{entry_point}",
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
                view=entry_point,
            )
        except Exception:  # noqa: BLE001 - telemetry must never break the UI
            pass


def _render_evidence_dashboard(eval_result: Any) -> None:
    """Render the Evidence Strength Dashboard below the verification table.

    Groups verified claims by inferred record type (Diagnosis, Medication,
    Symptom, Other) and shows verdict counts as a horizontal stacked bar
    chart (``st.bar_chart`` renders native hover tooltips with the exact
    per-segment count and, via the percentage caption below, the share of
    each verdict), plus a concise text summary of overall evidence
    strength. A dashboard build failure must never break the rest of the
    Evaluate results panel — it is caught, tracked, and skipped.
    """
    try:
        dashboard = build_evidence_dashboard(
            eval_result.verifications, eval_result.claims
        )
    except Exception as exc:  # noqa: BLE001 - dashboard is best-effort, never fatal
        logger.warning("evidence dashboard build failed error=%s", exc, exc_info=True)
        try:
            track_feature_error(EVIDENCE_DASHBOARD_FEATURE_ID, exc, phase="dashboard_build")
        except Exception:  # noqa: BLE001 - telemetry must never break the UI
            pass
        return

    if not dashboard:
        return

    impression_key = "evidence_dashboard_impression_sent"
    if not st.session_state.get(impression_key):
        try:
            track_impression(
                EVIDENCE_DASHBOARD_FEATURE_ID,
                entry_point="evaluate_report",
                recordTypeCount=len(dashboard),
            )
        except Exception:  # noqa: BLE001 - telemetry must never break the UI
            pass
        st.session_state[impression_key] = True

    with st.expander("📊 Evidence strength dashboard", expanded=True):
        st.caption(
            "Claims grouped by the type of medical record most likely to confirm them, "
            "with verdict counts from the verification step above. Hover a bar segment "
            "for the exact count and percentage of that record type."
        )
        chart_df = pd.DataFrame(dashboard).T.reindex(columns=list(VERDICTS)).fillna(0).astype(int)
        st.bar_chart(chart_df, horizontal=True)

        total_claims = sum(sum(counts.values()) for counts in dashboard.values())
        verdict_totals = {
            verdict: sum(counts.get(verdict, 0) for counts in dashboard.values())
            for verdict in VERDICTS
        }
        supported = verdict_totals["SUPPORTED"] + verdict_totals["PARTIALLY SUPPORTED"]
        supported_pct = (supported / total_claims * 100) if total_claims else 0.0
        contradicted_pct = (
            (verdict_totals["CONTRADICTED"] / total_claims * 100) if total_claims else 0.0
        )

        percentage_rows = [
            {
                "Record type": record_type,
                **{
                    verdict: f"{counts.get(verdict, 0)} ({counts.get(verdict, 0) / max(sum(counts.values()), 1) * 100:.0f}%)"
                    for verdict in VERDICTS
                },
            }
            for record_type, counts in sorted(dashboard.items())
        ]
        st.dataframe(percentage_rows, width="stretch", hide_index=True)

        weakest_type = min(
            dashboard.items(),
            key=lambda item: (
                item[1].get("SUPPORTED", 0) + item[1].get("PARTIALLY SUPPORTED", 0)
            )
            / max(sum(item[1].values()), 1),
        )[0]
        summary = (
            f"Of {total_claims} verified claim(s) across {len(dashboard)} record type(s), "
            f"{supported} ({supported_pct:.0f}%) are supported or partially supported by the "
            f"medical records"
            + (f", while {contradicted_pct:.0f}% are contradicted" if verdict_totals["CONTRADICTED"] else "")
            + f". **{weakest_type}** evidence is the weakest category — consider requesting "
            "or reviewing additional records in that area."
        )
        st.write(summary)

        try:
            track_interaction(
                EVIDENCE_DASHBOARD_FEATURE_ID,
                action="dashboard_viewed",
                claimCount=total_claims,
            )
            track_goal(
                EVIDENCE_DASHBOARD_FEATURE_ID,
                "evidence_strength_dashboard_rendered",
                supportedPct=round(supported_pct, 1),
            )
        except Exception:  # noqa: BLE001 - telemetry must never break the UI
            pass


# --------------------------------------------------------- fact citation export



def _rubric_and_positive_sources(eval_result: Any) -> tuple[set[str], set[str]]:
    """View-side wrapper over ``app.evaluate.rubric_and_positive_sources``.

    The link between a digest fact and a verification used to be a substring guess
    over free text; it is now an exact ``(document, page)`` join, computed in
    ``app.evaluate`` so the pipeline module owns the rule and this module only
    supplies the result object.
    """
    return rubric_and_positive_sources(
        getattr(eval_result, "digest", None),
        getattr(eval_result, "verifications", None) or [],
    )


def _render_fact_export_section(eval_result: Any) -> None:
    """Render the "Export Facts" panel: digest summary, filters, and downloads.

    Implements F7.S1 (Fact Citation Exporter, feature id
    `051bb638-ac1c-40cf-95f5-164779b4382c`) — placed immediately after the
    medical-digest summary so the export controls sit next to the data they
    export. Fires `impression` on first render, `interaction` on the Export
    Facts click and each individual download, `error` at the export boundary,
    and `goal` once files are generated. No fact content (descriptions,
    quotes, document names) is ever attached to a telemetry payload — only
    counts, format labels, and the two filter booleans.
    """
    digest = getattr(eval_result, "digest", None)
    if not digest or not digest.facts:
        return

    impression_key = "export_facts_impression_sent"
    if not st.session_state.get(impression_key):
        try:
            track_impression(
                EXPORT_FACTS_FEATURE_ID,
                entry_point="evaluate",
                fact_count=len(digest.facts),
            )
        except Exception:  # noqa: BLE001 - telemetry must never break the UI
            pass
        st.session_state[impression_key] = True

    with st.expander("📑 Medical record digest — export fact citations", expanded=False):
        st.caption(
            f"{len(digest.facts):,} facts extracted from {digest.pages_reviewed:,} pages "
            f"({digest.chunks_reviewed} chunk(s), {digest.duplicates_skipped} duplicate "
            "page(s) skipped)."
        )
        if digest.summary:
            st.write(digest.summary)

        only_rubric = st.checkbox(
            "Only show facts cited in rubric verification",
            key="export_facts_only_rubric",
        )
        only_positive = st.checkbox(
            "Only show facts supporting positive outcomes",
            key="export_facts_only_positive",
        )

        rid = get_request_id() or "-"
        if st.button("📤 Export Facts", key="export_facts_button"):
            try:
                track_interaction(
                    EXPORT_FACTS_FEATURE_ID,
                    action="export_click",
                    filter_rubric=only_rubric,
                    filter_positive=only_positive,
                )
            except Exception:  # noqa: BLE001 - telemetry must never break the UI
                pass

            cited_sources, positive_sources = _rubric_and_positive_sources(eval_result)
            generated: dict[str, bytes] = {}
            errors: list[str] = []
            for fmt in ("csv", "pdf", "md"):
                try:
                    generated[fmt] = export_facts(
                        digest,
                        fmt,
                        only_rubric_cited=only_rubric,
                        only_positive_outcomes=only_positive,
                        rubric_cited_sources=cited_sources,
                        positive_outcome_sources=positive_sources,
                    )
                except Exception as exc:  # noqa: BLE001 - one format failing must not block the rest
                    errors.append(f"{fmt}: {exc}")
                    logger.error(
                        "fact export generation failed format=%s error=%s",
                        fmt,
                        exc,
                        exc_info=True,
                        extra={
                            "request_id": rid,
                            "phase": "export_facts",
                            "status": "error",
                            "format": fmt,
                        },
                    )
            st.session_state["export_facts_files"] = generated
            filtered_count = len(
                filter_facts(
                    digest.facts,
                    only_rubric_cited=only_rubric,
                    only_positive_outcomes=only_positive,
                    rubric_cited_sources=cited_sources,
                    positive_outcome_sources=positive_sources,
                )
            )
            if errors:
                st.error("Some export formats could not be generated: " + "; ".join(errors))
            if generated:
                try:
                    track_goal(
                        EXPORT_FACTS_FEATURE_ID,
                        "facts exported",
                        fact_count=filtered_count,
                        filtered=only_rubric or only_positive,
                    )
                except Exception:  # noqa: BLE001 - telemetry must never break the UI
                    pass
                st.success(f"Generated export files for {filtered_count} fact row(s).")

        files_any: Any = st.session_state.get("export_facts_files")
        files: dict[str, bytes] = files_any if isinstance(files_any, dict) else {}
        if files:
            labels = {
                "csv": ("⬇️ Download CSV", "text/csv", "medical_digest_facts.csv"),
                "pdf": ("⬇️ Download PDF", "application/pdf", "medical_digest_facts.pdf"),
                "md": ("⬇️ Download Markdown", "text/markdown", "medical_digest_facts.md"),
            }
            columns = st.columns(3)
            for col, fmt in zip(columns, ("csv", "pdf", "md")):
                data = files.get(fmt)
                if data is None:
                    continue
                label, mime, filename = labels[fmt]
                clicked = col.download_button(
                    label,
                    data=data,
                    file_name=filename,
                    mime=mime,
                    key=f"export_facts_download_{fmt}",
                )
                if clicked:
                    try:
                        track_interaction(
                            EXPORT_FACTS_FEATURE_ID,
                            action="download",
                            format=fmt,
                            filter_rubric=only_rubric,
                            filter_positive=only_positive,
                        )
                    except Exception:  # noqa: BLE001 - telemetry must never break the UI
                        pass


# ------------------------------------------------------------- event timeline
def _build_timeline_figure(
    events: list[dict[str, Any]], gaps: list[dict[str, Any]]
) -> "go.Figure":
    """Build the vertical Plotly timeline figure for the given (already
    filtered) events. Dated events are plotted with time on the y-axis
    (earliest at top) and coarse type category on the x-axis; each marker
    carries the event's index (via ``customdata``) so a click can be mapped
    back to the full fact for the details panel. Gap periods are shaded as
    horizontal bands behind the markers.
    """
    if go is None:
        raise RuntimeError(
            "plotly is not installed, so the interactive timeline chart cannot be drawn"
        )
    fig = go.Figure()

    for gap in gaps:
        fig.add_hrect(
            y0=gap["start"],
            y1=gap["end"],
            fillcolor="rgba(220, 38, 38, 0.10)",
            line_width=0,
            annotation_text="gap in records",
            annotation_position="top left",
            annotation_font_size=10,
        )

    for category, color in _TIMELINE_CATEGORY_COLORS.items():
        subset = [(i, e) for i, e in enumerate(events) if e.get("category") == category and e.get("date_iso")]
        if not subset:
            continue
        fig.add_trace(
            go.Scatter(
                x=[category] * len(subset),
                y=[e["date_iso"] for _, e in subset],
                mode="markers",
                name=category.capitalize(),
                marker=dict(size=13, color=color, line=dict(width=1, color="white")),
                text=[f"{e.get('type', '')}: {e.get('description', '')[:100]}" for _, e in subset],
                customdata=[[i] for i, _ in subset],
                hovertemplate="%{y}<br>%{text}<extra>%{fullData.name}</extra>",
            )
        )

    fig.update_yaxes(title="Date", autorange="reversed", type="date")
    fig.update_xaxes(title="Event type", type="category")
    dated_count = sum(1 for e in events if e.get("date_iso"))
    fig.update_layout(
        height=max(360, min(1400, 70 * max(dated_count, 3))),
        showlegend=True,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def _render_timeline_event_details(event: dict[str, Any]) -> None:
    """Render the full fact text + source citation for a clicked timeline event."""
    with st.container(border=True):
        st.markdown(f"**{event.get('date_label', 'unknown')} — {event.get('type', '')}**")
        st.write(event.get("description", "") or "(no description)")
        if event.get("quote"):
            st.caption(f"“{event['quote']}”")
        st.caption(f"Source: {event.get('source', 'unknown')}")


def _render_medical_timeline(eval_result: Any, *, request_reference: str) -> None:
    """Render the Medical Event Timeline subsection (F7.S2, feature id
    `222efbdb-50be-4ff7-a384-1595d543c842`).

    Builds (and caches in ``st.session_state['timeline_data']``, keyed by the
    current run's reference so a stale digest never leaks across runs — F7.S1
    acceptance criterion "only facts from the current request_id are
    processed") a vertical Plotly timeline from the digest's facts, offers
    all/diagnostic/treatment filter buttons, and shows full fact text +
    source citation for a clicked event. A build/render failure here must
    never break the rest of the Evaluate results panel.
    """
    digest = getattr(eval_result, "digest", None)
    if not digest or not digest.facts:
        return

    cached_reference = st.session_state.get("timeline_request_id")
    if st.session_state.get("timeline_data") is None or cached_reference != request_reference:
        try:
            llm = get_llm()
        except Exception:  # noqa: BLE001 - LLM date-inference fallback is optional
            llm = None
        st.session_state["timeline_data"] = build_timeline_data(
            digest, llm, feature_id=TIMELINE_FEATURE_ID
        )
        st.session_state["timeline_request_id"] = request_reference

    timeline_data: dict[str, Any] = st.session_state.get("timeline_data") or {}
    events: list[dict[str, Any]] = timeline_data.get("events") or []
    if not events:
        return

    impression_key = f"timeline_impression_sent_{request_reference}"
    if not st.session_state.get(impression_key):
        try:
            track_impression(
                TIMELINE_FEATURE_ID,
                entry_point="evaluate",
                timeline_event_count=len(events),
            )
        except Exception:  # noqa: BLE001 - telemetry must never break the UI
            pass
        st.session_state[impression_key] = True

    with st.expander("🗓️ Medical event timeline", expanded=True):
        st.caption(
            f"{len(events)} event(s) — {timeline_data.get('dated_count', 0)} dated, "
            f"{timeline_data.get('undated_count', 0)} undated. "
            f"{timeline_data.get('gap_count', 0)} gap period(s) with no records detected."
        )

        filter_choice = st.radio(
            "Filter events",
            list(_TIMELINE_FILTER_LABELS),
            key="timeline_filter_choice",
            horizontal=True,
        )
        selected_filter = _TIMELINE_FILTER_LABELS[filter_choice]
        if st.session_state.get("timeline_filter_tracked") != selected_filter:
            st.session_state["timeline_filter_tracked"] = selected_filter
            try:
                track_interaction(TIMELINE_FEATURE_ID, action="filter", filter=selected_filter)
            except Exception:  # noqa: BLE001 - telemetry must never break the UI
                pass

        filtered = (
            events
            if selected_filter == "all"
            else [e for e in events if e.get("category") == selected_filter]
        )

        if not filtered:
            st.info("No events match this filter.")
            return

        undated_in_view = [e for e in filtered if not e.get("date_iso")]
        dated_in_view = [e for e in filtered if e.get("date_iso")]

        clicked_event: dict[str, Any] | None = None
        if dated_in_view:
            # The figure build sits *inside* the guard: an unavailable/broken chart
            # library must cost the chart, not the whole results panel.
            try:
                fig = _build_timeline_figure(filtered, timeline_data.get("gaps") or [])
                chart_state = st.plotly_chart(
                    fig,
                    width="stretch",
                    key=f"timeline_chart_{request_reference}_{selected_filter}",
                    on_select="rerun",
                    selection_mode=("points",),
                )
                points = getattr(getattr(chart_state, "selection", None), "points", None) or []
                if points:
                    customdata = points[0].get("customdata") or []
                    if customdata:
                        idx = int(customdata[0])
                        if 0 <= idx < len(filtered):
                            clicked_event = filtered[idx]
            except Exception as exc:  # noqa: BLE001 - chart build/render must never break the panel
                logger.warning("timeline chart unavailable error=%s", exc, exc_info=True)
                try:
                    track_feature_error(TIMELINE_FEATURE_ID, exc, phase="chart_render")
                except Exception:  # noqa: BLE001 - telemetry must never break the UI
                    pass
                st.info(
                    "The interactive timeline chart is unavailable — use the event "
                    "list below instead."
                )

        if undated_in_view:
            st.caption(f"{len(undated_in_view)} undated event(s) — select below to view details.")

        options = list(range(len(filtered)))
        selected_idx = st.selectbox(
            "Or choose an event from the list",
            options=options,
            format_func=lambda i: (
                f"{filtered[i].get('date_label', 'unknown')} — "
                f"{filtered[i].get('type', '')}: {filtered[i].get('description', '')[:80]}"
            ),
            key=f"timeline_event_select_{selected_filter}",
            index=None,
            placeholder="Choose an event…",
        )
        if selected_idx is not None:
            clicked_event = filtered[selected_idx]

        if clicked_event is not None:
            try:
                track_interaction(TIMELINE_FEATURE_ID, action="event_click", filter=selected_filter)
            except Exception:  # noqa: BLE001 - telemetry must never break the UI
                pass
            _render_timeline_event_details(clicked_event)


def _render_effectiveness_score(eval_result: Any) -> None:
    """Render the effectiveness score badge and ranked recommendations (F4.S2).

    Fires `impression` once per rendered run (`entryPoint` attribute) when
    the score becomes visible, and `interaction` on each recommendation
    button click (`recommendationIndex` + `action`). The `goal` event for the
    computed score itself is fired at the compute boundary in
    `app/evaluate.py::_score_and_recommend` — not here — since it must fire
    exactly once per computation, not once per render.
    """
    score = int(getattr(eval_result, "effectiveness_score", 0) or 0)
    band = getattr(eval_result, "score_band", "") or compute_score_band(score)
    rid = _result_reference() or "no-ref"

    impression_key = f"effectiveness_score_impression_sent_{rid}"
    if not st.session_state.get(impression_key):
        try:
            track_impression(EFFECTIVENESS_SCORE_FEATURE_ID, entry_point="evaluate_report_tab")
        except Exception:  # noqa: BLE001 - telemetry must never break the UI
            pass
        st.session_state[impression_key] = True

    st.subheader("🎯 Statement Effectiveness Score")
    emoji, label = _SCORE_BAND_DISPLAY.get(band, _SCORE_BAND_DISPLAY["red"])
    st.metric("Effectiveness score", f"{score}/100")
    banner = {"green": st.success, "yellow": st.warning, "red": st.error}.get(band, st.error)
    banner(f"{emoji} {label} ({band.upper()} band)")

    recommendations = getattr(eval_result, "recommendations", None) or []
    if not recommendations:
        return

    st.markdown("**Top improvement recommendations (ranked by estimated impact):**")
    claims = getattr(eval_result, "claims", None) or []
    claim_text = {c.get("id"): c.get("text", "") for c in claims}
    for index, rec in enumerate(recommendations, start=1):
        title = str(rec.get("title", ""))
        impact = str(rec.get("impact", ""))
        explanation = str(rec.get("explanation", ""))
        claim_id = rec.get("claim_id")
        st.markdown(f"**{index}. {title}** _{impact}_")
        st.caption(explanation)
        has_matching_claim = claim_id is not None and claim_id in claim_text
        action_label = (
            f"🔍 Jump to claim #{claim_id}" if has_matching_claim else "✏️ Apply to rewrite"
        )
        clicked = st.button(action_label, key=f"eval_rec_action_{rid}_{index}")
        if clicked:
            action = "jump_to_claim" if has_matching_claim else "trigger_rewrite"
            st.session_state["eval_recommendation_target"] = {
                "claim_id": claim_id,
                "recommendation_index": index,
                "action": action,
            }
            try:
                track_interaction(
                    EFFECTIVENESS_SCORE_FEATURE_ID,
                    recommendationIndex=index,
                    action=action,
                )
            except Exception:  # noqa: BLE001 - telemetry must never break the UI
                pass
            if has_matching_claim:
                st.info(f"📍 Claim #{claim_id}: {claim_text.get(claim_id, '')}")
            else:
                st.info(
                    "✏️ Marked for rewrite — see 'Suggested improvements — proposed rewrite' below."
                )


def _render_record_coverage(eval_result: Any) -> None:
    """Show what the record set contained, what was read, and how citations held up.

    Rendered only when there is something to say (files with known page counts, an
    unreadable-page count, chunks that yielded no facts, a citation check, or
    coverage-gap claims), so an ordinary small run is not buried in caveats. The
    panel opens automatically when something is missing, because a partial review
    that arrives silently is the failure this exists to prevent.
    """
    digest = getattr(eval_result, "digest", None)
    if digest is None:
        return
    check = getattr(digest, "citation_check", None) or {}
    gaps = getattr(eval_result, "evidence_gaps", None) or []
    if not (
        digest.pages_in_files
        or digest.unreadable_pages
        or digest.chunks_without_facts
        or check.get("checked")
        or gaps
    ):
        return

    with st.expander(
        "🧾 Record coverage & citation check",
        expanded=bool(digest.unreadable_pages or gaps),
    ):
        st.caption(
            "What the uploaded files contained, what the review actually read, and whether "
            "each citation's quote was found on the page it names."
        )
        for line in coverage_lines(digest):
            st.markdown(line)
        if digest.files:
            st.dataframe(digest.files, width="stretch", hide_index=True)
        if digest.duplicate_pages:
            shown = ", ".join(
                f"{row.get('document')} p.{row.get('page')} = {row.get('duplicate_of')}"
                for row in digest.duplicate_pages[:8]
            )
            more = " …" if len(digest.duplicate_pages) > 8 else ""
            st.caption(f"Pages skipped as duplicates: {shown}{more}")
        if gaps:
            st.warning(
                "These claims had no matching text in the uploaded records, so nothing could "
                "be checked against them. That is a record-coverage gap, not a contradiction:"
            )
            for gap in gaps:
                st.write(f"- {gap.get('claim', '')}")


def _case_topic_letters(eval_result: Any) -> list[str]:
    """The checklist topics this case concerns, as letters.

    The condition selector's per-slot selection is authoritative — it is what the run was
    actually scoped to. The fallback reads the result's own topic rows for a reloaded
    result or a session that no longer holds the selector state, so the flags do not
    silently disappear after a page reload.
    """
    from ..condition_selector import TOPIC_LABELS

    stored = st.session_state.get("preselected_topics_eval")
    if isinstance(stored, (list, tuple)):
        chosen = [str(letter).strip().upper() for letter in stored]
        known = [letter for letter in chosen if letter in TOPIC_LABELS]
        if known:
            return sorted(set(known))

    derived: set[str] = set()
    for row in getattr(eval_result, "topic_rows", []) or []:
        if not row.get("applicable"):
            continue
        first = str(row.get("topic", "")).strip()[:1].upper()
        if first in TOPIC_LABELS:
            derived.add(first)
    return sorted(derived)


def _framework_currency_ttl_days() -> int:
    """The freshness window for a currency verdict (sidebar settings win when present)."""
    settings = st.session_state.get("settings")
    if isinstance(settings, Settings):
        return settings.framework_currency_ttl_days
    return load_settings().framework_currency_ttl_days


def _render_framework_currency_flags(eval_result: Any) -> None:
    """Flag checklist topics whose committed text is known to be out of date.

    Read-only by design: this renders the verdict a paid check already stored (see
    ``app/knowledge_currency``) and never calls the API, so an evaluation never gains a
    surprise network call or a surprise bill. When no verdict applies, it says so in a
    caption rather than showing nothing — silence would read as "all current", which is
    the one thing an absent verdict does not mean.
    """
    letters = _case_topic_letters(eval_result)
    if not letters:
        return

    flag = currency.case_currency_flag(letters, ttl_days=_framework_currency_ttl_days())

    if flag.stale:
        st.warning(
            "⚠️ **These checklist topics have changed under current VA law.** This run's "
            "guidance on them rests on committed text that no longer matches the sources "
            "the app cites, so treat it as unreliable — verify the topics in the Research "
            "tab, and see app/knowledge/ before revising a statement from them:"
        )
        for verdict in flag.stale:
            suffix = f" — {verdict.authority}" if verdict.authority else ""
            st.markdown(f"- **{verdict.topic} — {verdict.label}:** {verdict.note}{suffix}")

    if flag.unconfirmed:
        st.info(
            "❓ **Could not be confirmed as current** (the check found conflicting sources "
            "or could not confirm): "
            + ", ".join(f"{v.topic} — {v.label}" for v in flag.unconfirmed)
        )

    if flag.verified:
        checked = flag.report.checked_at[:10] if flag.report is not None else ""
        st.caption(
            f"Checklist currency verified for this run's topics on {checked}. "
            "Verification covers the committed checklist and legal framework, not the "
            "statement's facts."
        )
    elif not flag.stale and not flag.unconfirmed:
        st.caption(
            "Checklist currency has not been verified for these topics, so nothing here "
            "confirms the app's committed framework is still current. "
            "See the Research tab → Framework currency."
        )


def _render_evaluation_results(eval_result: Any) -> None:
    render_usage_summary(st.session_state.get("eval_usage"))

    rid = _result_reference()
    if rid:
        # The panel is cached for the whole session, so say which run it came
        # from: otherwise any rerun makes an old report look like a fresh run.
        st.caption(
            f"Results for reference `{rid}` — the last completed run in this session. "
            "Each new run mints a new reference (see About → Recent run log)."
        )

    if _is_empty_analysis(eval_result):
        st.error(
            "This run returned no usable analysis: 0 claims extracted, no rubric "
            "scores, and no rewrite. The model or endpoint accepted the request but "
            "returned empty output."
            + (f" Reference: {rid}." if rid else "")
            + " Check the sidebar model names and base URL (About → Recent run log "
            "shows the LLM call count for this run), then re-run."
        )
        # A run that returned nothing usable is still a failed run: the reference
        # is in the message, so the lines it points at belong beside it too.
        if rid:
            render_failure_detail(rid)

    if getattr(eval_result, "truncation_warning", ""):
        st.warning(
            f"⚠️ {eval_result.truncation_warning} (input was {eval_result.input_chars:,} chars; "
            f"{eval_result.truncated_chars:,} truncated). Review the report header for details."
        )

    st.divider()
    _render_record_coverage(eval_result)

    st.divider()
    _render_effectiveness_score(eval_result)

    st.divider()
    st.subheader("📋 Evaluation Results")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall rating", eval_result.overall_rating)
    col2.metric("Claims verified", len(eval_result.verifications))
    col3.metric("Contradictions", eval_result.contradiction_count)
    _applicable = [t for t in eval_result.topic_rows if t.get("applicable")]
    _covered = [t for t in _applicable if t.get("coverage") == "covered"]
    col4.metric(
        "Topics covered", f"{len(_covered)}/{len(_applicable)}" if _applicable else "—"
    )

    with st.expander("Executive summary", expanded=True):
        st.write(eval_result.executive_summary)

    with st.expander("Claim-by-claim verification table", expanded=True):
        rows = []
        claim_text = {c["id"]: c.get("text", "") for c in eval_result.claims}
        for v in eval_result.verifications:
            rows.append(
                {
                    "Claim": claim_text.get(v.get("id"), ""),
                    "Verdict": v.get("verdict", ""),
                    "Record reference": v.get("record_reference", ""),
                    "Note": v.get("note", ""),
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)

    _render_evidence_dashboard(eval_result)

    with st.expander("Rubric scores", expanded=True):
        score_rows = [
            {
                "Dimension": DIMENSION_LABELS.get(k, k),
                "Score": eval_result.scores.get(k, 0),
                "Rationale": eval_result.rationales.get(k, ""),
            }
            for k in DIMENSION_LABELS
        ]
        st.dataframe(score_rows, width="stretch", hide_index=True)
        st.bar_chart(
            {DIMENSION_LABELS[k]: eval_result.scores.get(k, 0) for k in DIMENSION_LABELS},
            horizontal=True,
        )

    if eval_result.topic_rows:
        with st.expander(
            "🧭 Topic coverage — what the statement does and does not address", expanded=True
        ):
            if eval_result.topic_focus:
                st.write(f"**Claim focus:** {eval_result.topic_focus}")
            topic_table = [
                {
                    "Topic": t.get("topic", ""),
                    "Applicable": "Yes" if t.get("applicable") else "No",
                    "Coverage": t.get("coverage", ""),
                    "Evidence in statement": t.get("evidence", ""),
                    "How to strengthen": t.get("gap_note", ""),
                }
                for t in eval_result.topic_rows
            ]
            st.dataframe(topic_table, width="stretch", hide_index=True)
            # Before the gaps, not after: a topic whose committed text is out of date can
            # be *the reason* a gap was reported here, so the caveat has to arrive before
            # the reader acts on the gaps rather than as a footnote.
            _render_framework_currency_flags(eval_result)
            if eval_result.topic_critical_gaps:
                st.warning(
                    "**Critical gaps — the highest-impact topics this statement still misses:**"
                )
                for gap in eval_result.topic_critical_gaps:
                    st.write(f"- {gap}")
            if eval_result.topic_notes:
                st.caption(eval_result.topic_notes)

    render_follow_up_questions(
        slot="eval",
        source_id=_result_reference(),
        questions=evaluate_follow_up_questions(eval_result),
        empty_message=(
            "No follow-up questions are needed — every applicable checklist topic is already "
            "covered well enough for this run."
        ),
        next_run_label="evaluation",
    )

    with st.expander("Improvements & record facts to add", expanded=True):
        for imp in eval_result.improvements:
            st.markdown(f"**{imp.get('priority', '?')}. {imp.get('problem', '')}**")
            st.write(imp.get("suggestion", ""))
            if imp.get("example_rewrite"):
                st.caption(f"Example: “{imp.get('example_rewrite')}”")
        if eval_result.omitted_record_facts:
            st.markdown("**Facts from the records you could add (verify first):**")
            for fact_dict in eval_result.omitted_record_facts:
                st.write(
                    f"- {fact_dict.get('fact', '')} _(source: {fact_dict.get('source', '')})_"
                )

    if eval_result.revised_statement or eval_result.revision_changes:
        with st.expander("📝 Suggested improvements — proposed rewrite", expanded=True):
            if eval_result.revision_notes:
                st.info(eval_result.revision_notes)
            if eval_result.revision_changes:
                change_rows = [
                    {
                        "Category": c.get("category", ""),
                        "Original": c.get("original", "") or "(addition)",
                        "Suggested": c.get("revised", ""),
                        "Why": c.get("reason", ""),
                    }
                    for c in eval_result.revision_changes
                ]
                st.dataframe(change_rows, width="stretch", hide_index=True)
            if eval_result.added_facts_to_verify:
                st.markdown(
                    "**Record-sourced facts added — the witness must confirm each before signing:**"
                )
                for fact_str in eval_result.added_facts_to_verify:
                    st.write(f"- {fact_str}")
            st.markdown("#### Revised statement")
            st.caption(
                "Contradictions have been corrected to match the medical records. Resolve every "
                "[Confirm: ...] placeholder with the witness before signing."
            )
            revised = st.text_area(
                "Revised statement (editable)",
                value=eval_result.revised_statement,
                height=420,
                key="eval_revised_statement",
            )
            col_a, col_b = st.columns(2)
            col_a.download_button(
                "⬇️ Download revised statement (.txt)",
                data=revised.encode("utf-8"),
                file_name="lay_statement_revised.txt",
                mime="text/plain",
            )
            col_b.download_button(
                "⬇️ Download revised statement (.md)",
                data=revised.encode("utf-8"),
                file_name="lay_statement_revised.md",
                mime="text/markdown",
            )
            _render_pdf_export(revised, entry_point="evaluate")

    _render_fact_export_section(eval_result)
    _render_medical_timeline(eval_result, request_reference=_result_reference())

    with st.expander("Full markdown report"):
        st.markdown(eval_result.report_markdown)
    st.download_button(
        "⬇️ Download evaluation report (.md)",
        data=eval_result.report_markdown.encode("utf-8"),
        file_name="lay_statement_evaluation.md",
        mime="text/markdown",
    )
