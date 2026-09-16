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

import streamlit as st

from .. import audit as audit_log
from ..agiloop_telemetry import track_goal, track_impression, track_interaction
from ..evaluate import DIMENSION_LABELS, run_evaluation
from ..logging_config import get_logger, get_request_id
from ..documents import (
    EVALUATE_INTERNAL_MAX_CHARS,
    MAX_STATEMENT_CHARS,
)
from ..pdf_export import detect_unconfirmed_placeholders, generate_statement_pdf
from ..pipeline_guard import (
    PipelineTimeoutError,
    check_memory_before_run,
    run_with_timeout,
)
from ..profiler import RunProfiler, get_profiler
from ..run_log import run_log_event
from ..shutdown import enter_run, exit_run
from .shared import (
    FEATURE_ID,
    audit_condition_for_slot,
    audit_record_meta,
    check_shutdown_gate,
    check_upload_limits,
    extract_uploads,
    format_error_for_user,
    get_llm,
    new_run_request_id,
    progress_widgets,
    records_uploader,
    render_condition_selector_for_slot,
    render_usage_summary,
    record_watchdog_run,
)

logger = get_logger("app.views.evaluate")

# Feature: Final Statement PDF Export
PDF_EXPORT_FEATURE_ID = "0d76d70b-8dd6-4561-a874-f768d5929222"  # final-statement-pdf-export


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
                st.warning(msg)
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

    render_condition_selector_for_slot("eval")

    run = st.button("🔍 Run exhaustive evaluation", type="primary", key="eval_run")
    if run:
        if not _validate_evaluate_inputs(statement_text, records):
            return
        _run_evaluation_flow(statement_text, records)

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
    rid = get_request_id() or "-"
    if not statement_text.strip():
        msg = "Provide the lay statement first (upload or paste)."
        run_log_event("evaluate", "rejected", request_id=rid, error=msg, reason="no_statement")
        st.error(msg)
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
        st.error(msg)
        return False
    if not records:
        msg = "Upload at least one medical record file."
        run_log_event("evaluate", "rejected", request_id=rid, error=msg, reason="no_records")
        st.error(msg)
        return False
    return True


# -------------------------------------------------------------- run pipeline
def _run_evaluation_flow(statement_text: str, records: list) -> None:
    """Mint a run id, gate shutdown, run the pipeline, persist the result."""
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
            run_evaluation, llm, statement_text.strip(), records, progress=update
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
        st.error(f"Evaluation aborted: {mem_exc} (reference: {rid})")
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
        st.error(f"Evaluation aborted: {timeout_exc} (reference: {rid})")
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
    )
    run_log_event("evaluate", "ok", request_id=rid, duration_ms=duration_ms, **_outcome)
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
    record_watchdog_run(llm.usage)


# ------------------------------------------------------------------ results UI
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
        st.error(f"Could not generate the PDF export: {exc}")
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

    if getattr(eval_result, "truncation_warning", ""):
        st.warning(
            f"⚠️ {eval_result.truncation_warning} (input was {eval_result.input_chars:,} chars; "
            f"{eval_result.truncated_chars:,} truncated). Review the report header for details."
        )

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
            if eval_result.topic_critical_gaps:
                st.warning(
                    "**Critical gaps — the highest-impact topics this statement still misses:**"
                )
                for gap in eval_result.topic_critical_gaps:
                    st.write(f"- {gap}")
            if eval_result.topic_notes:
                st.caption(eval_result.topic_notes)

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

    with st.expander("Full markdown report"):
        st.markdown(eval_result.report_markdown)
    st.download_button(
        "⬇️ Download evaluation report (.md)",
        data=eval_result.report_markdown.encode("utf-8"),
        file_name="lay_statement_evaluation.md",
        mime="text/markdown",
    )
