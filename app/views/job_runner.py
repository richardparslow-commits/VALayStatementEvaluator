"""Queue-mode run orchestration for the Evaluate and Draft tabs.

This is the web-pod half of Pattern C. When a distributed job backend is
configured, a run is *submitted* rather than executed: the tab serializes its
inputs, enqueues them, and polls a small status key while a worker does the
digest. The pod serving the browser never holds the ~1.8 GB peak, so a heavy
record set stops pinning one pod, and closing the tab or losing a pod no longer
kills the run — the worker finishes it and any pod can render the result.

The in-process path in the two tab views is untouched: this module is only
entered when :func:`queue_mode_active` is true, so single-instance deployments
behave exactly as before.

Because a queued run outlives the browser session that started it, the job id is
kept in ``st.session_state`` and :func:`resume_pending_job` re-attaches to it on
the next rerun instead of leaving the user with a blank tab and a run they
cannot see.
"""
from __future__ import annotations

from .. import pilot
from ..queue_policy import require_synthetic_queue, synthetic_mode, synthetic_queue

import time
from dataclasses import dataclass
from typing import Any

import streamlit as st

from .. import config
from .. import tracing
from ..blob_store import BlobStoreError, get_blob_store
from ..error_report import report_failure
from ..job_payload import (
    KIND_DRAFT,
    KIND_EVALUATE,
    DraftJob,
    EvaluateJob,
    PayloadError,
    RunResult,
    decode_result,
    decode_job,
    documents_bundle,
    encode_job,
    encode_job_with_blob,
    payload_needs_blob,
)
from ..job_queue import (
    STATUS_DONE,
    JobBackend,
    JobRecord,
    JobQueueError,
    get_job_backend,
)
from ..logging_config import get_logger
from ..run_log import run_log_event
from .ops import render_failure_detail

logger = get_logger("app.views.job_runner")

# Session keys for the job a tab is waiting on, and the result keys each tab's
# results renderer reads (kept in sync with evaluate_view/draft_view).
_PENDING_KEY = "va_lse_pending_job_{slot}"
_REQUEST_ID_KEY = "va_lse_pending_request_{slot}"
_RESULT_KEYS: dict[str, tuple[str, str, str]] = {
    "eval": ("eval_result", "eval_usage", "eval_request_id"),
    "draft": ("draft_result", "draft_usage", "draft_request_id"),
}

# How long the browser waits before it stops blocking on a still-running job.
# The worker keeps going; the user gets a banner and a way back in rather than a
# spinner that never ends.
UI_WAIT_SLACK_SECONDS = 60.0


@dataclass
class QueueOutcome:
    """Result of a queued run: either a decoded result or a user-facing failure."""

    ok: bool
    run: RunResult | None = None
    error: str = ""
    error_class: str = ""
    request_id: str = ""
    # True when the job is still running on a worker and the UI gave up waiting.
    still_running: bool = False


def _pending_key(slot: str) -> str:
    return _PENDING_KEY.format(slot=slot)


def _pending_request_key(slot: str) -> str:
    return _REQUEST_ID_KEY.format(slot=slot)


def queue_mode_active() -> bool:
    """True when runs should be submitted to a worker instead of run in-process."""
    if not synthetic_mode() or not config.JOB_QUEUE_ENABLED:
        return False
    try:
        distributed = bool(get_job_backend().is_distributed)
        return synthetic_mode() and distributed
    except Exception as exc:  # noqa: BLE001 - fall back to in-process, never break the tab
        if not synthetic_mode():
            return False
        logger.warning(
            "job queue unavailable; running in-process: %s", f"{type(exc).__name__}: {exc}"
        )
        return False


@synthetic_queue
def worker_config_error() -> str:
    """Return why a worker could not authenticate, or "" when it is configured.

    A worker has no browser session, so a key typed into the sidebar cannot reach
    it. Checking here turns that into a clear message before a job is queued,
    rather than a job that fails on a worker moments later with a stack trace the
    user cannot act on.
    """
    try:
        settings = config.load_settings()
    except Exception as exc:  # noqa: BLE001
        return f"could not read worker configuration: {type(exc).__name__}: {exc}"
    if not settings.configured:
        return (
            "Queue mode is enabled, but the LLM API key is not configured in the "
            "environment. Worker processes cannot read a key typed into the sidebar — "
            "set OPENAI_API_KEY in the environment or the deployed secret "
            "(see DEPLOYMENT.md → Pattern C)."
        )
    return ""


@synthetic_queue
def _hydrate(slot: str, run: RunResult) -> None:
    """Store a decoded result under the keys the tab's results renderer reads."""
    result_key, usage_key, rid_key = _RESULT_KEYS[slot]
    try:
        from .usage import record_watchdog_run

        record_watchdog_run(run.usage)
    except Exception:  # noqa: BLE001 - the watchdog is advisory
        pass
    require_synthetic_queue()
    if slot == "eval":
        # Recovery cannot bind new questions to the current form inputs. Keep
        # earlier answers and their own binding, but block collection into it.
        st.session_state["eval_follow_up_questions_bound"] = False
    st.session_state[result_key] = run.result
    st.session_state[usage_key] = run.usage
    st.session_state[rid_key] = run.request_id or st.session_state.get(rid_key, "")


@synthetic_queue
def _fetch_outcome(backend: JobBackend, record: JobRecord, slot: str) -> QueueOutcome:
    """Turn a terminal job record into an outcome, decoding the result on success."""
    if not pilot.owns(record):
        return QueueOutcome(ok=False, error="Run unavailable for this session.", error_class="AccessDenied")
    if slot not in _RESULT_KEYS or record.kind != (KIND_EVALUATE if slot == "eval" else KIND_DRAFT):
        return QueueOutcome(ok=False, error="Queued result does not match this run.", error_class="PayloadError")
    request_id = record.request_id or ""
    if record.status != STATUS_DONE:
        return QueueOutcome(
            ok=False,
            error=record.error or "the worker failed to complete this run",
            error_class=record.error_class,
            request_id=request_id,
        )
    raw = backend.get_result(record.job_id)
    require_synthetic_queue()
    if not raw:
        return QueueOutcome(
            ok=False,
            error="the run finished but its result is no longer in the queue "
            "(it most likely expired) — please re-run",
            error_class="ResultExpired",
            request_id=request_id,
        )
    try:
        run = decode_result(raw)
        if run.kind != record.kind or (record.request_id and run.request_id != record.request_id):
            raise PayloadError("Queued result does not match this run.")
    except PayloadError as exc:
        return QueueOutcome(
            ok=False,
            error=str(exc),
            error_class="PayloadError",
            request_id=request_id,
        )
    _hydrate(slot, run)
    return QueueOutcome(ok=True, run=run, request_id=request_id)


@synthetic_queue
def _poll(
    backend: JobBackend,
    job_id: str,
    slot: str,
    *,
    wait_seconds: float,
) -> QueueOutcome:
    """Poll a job until it finishes, the wait budget expires, or the queue breaks."""
    bar = st.progress(0.0, text="Queued — waiting for a worker…")
    deadline = time.monotonic() + max(1.0, wait_seconds)
    last_progress = 0.0
    consecutive_errors = 0
    try:
        while True:
            try:
                record = backend.get(job_id)
                require_synthetic_queue()
                consecutive_errors = 0
            except pilot.PilotBlocked:
                raise
            except Exception as exc:  # noqa: BLE001 - a flaky queue must not lose the job
                require_synthetic_queue()
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    return QueueOutcome(
                        ok=False,
                        error=f"lost contact with the job queue ({type(exc).__name__}). "
                        "The run may still finish — reload this tab to check again.",
                        error_class="JobQueueError",
                        still_running=True,
                    )
                time.sleep(float(config.JOB_QUEUE_UI_POLL_SECONDS))
                continue
            if record is None:
                return QueueOutcome(
                    ok=False,
                    error="this run is no longer in the queue (it most likely expired) — "
                    "please re-run",
                    error_class="JobExpired",
                )
            if not pilot.owns(record):
                return QueueOutcome(ok=False, error="Run unavailable for this session.", error_class="AccessDenied")
            last_progress = max(last_progress, record.progress)
            bar.progress(
                min(max(last_progress, 0.0), 1.0),
                text=record.message or "Running on a worker…",
            )
            if record.is_terminal:
                break
            if time.monotonic() >= deadline:
                return QueueOutcome(
                    ok=False,
                    error="this run is still going on a worker. You can leave this page — "
                    "the results will be here when it finishes.",
                    request_id=record.request_id or "",
                    still_running=True,
                )
            time.sleep(max(0.25, float(config.JOB_QUEUE_UI_POLL_SECONDS)))
    finally:
        bar.empty()
    return _fetch_outcome(backend, record, slot)


@synthetic_queue
def _render_failure(outcome: QueueOutcome, action_label: str) -> None:
    """Show a queued run's failure (or still-running state) to the user."""
    if outcome.still_running:
        pilot.display(outcome.error, container=st, method="info")
        return
    reference = f" (reference: {outcome.request_id})" if outcome.request_id else ""
    pilot.display(f"{action_label} failed: {outcome.error}{reference}", container=st, method="error")
    # A queued run is executed by a worker, so its lines are not in this process's
    # buffer — the shared run log is what makes this one resolvable at all.
    if outcome.request_id:
        render_failure_detail(outcome.request_id)


@synthetic_queue
def _encode_payload(kind: str, job: EvaluateJob | DraftJob) -> str:
    """Encode a job, externalizing its documents when they are large.

    Small jobs travel inline in the queue; large ones put their extracted record
    text in the blob store and carry a reference, so tens of megabytes never sit
    in Redis, reducing allocation pressure and explicit capacity failures, and
    are not re-fetched on every status poll. See ``app/blob_store.py``.

    Blobs are **not** deleted when a job finishes: keys are content-addressed, so
    two jobs over the same record bundle share one blob and deleting it on the
    first completion would break the second. They expire on the blob store's own
    TTL sweep instead, which mirrors the queue's result TTL.
    """
    if not payload_needs_blob(kind, job):
        return encode_job(kind, job)
    ref = get_blob_store().put(documents_bundle(job))
    logger.info(
        "externalized job documents blob=%s bytes=%d",
        ref.key,
        ref.size,
        extra={"phase": kind, "status": "blob", "bytes": ref.size},
    )
    return encode_job_with_blob(kind, job, ref)


@synthetic_queue
def submit_job(
    *,
    slot: str,
    job: EvaluateJob | DraftJob | None,
    request_id: str,
    condition: str | None,
    sources: list[str],
    files: int,
    pages: int,
    action_label: str,
    wait_for_result: bool = True,
) -> QueueOutcome | None:
    """Enqueue a run and wait for it; return None without a confirmed outcome.

    None means "there is no confirmed outcome yet". A lost server reply may
    still have committed the submission; retain its exact inputs and reference
    in this session so a retry cannot start duplicate work.
    """
    backend = get_job_backend()
    submission_key = f"{slot}_queue_submission"
    pending = st.session_state.get(submission_key)
    if job is None:
        if not isinstance(pending, dict) or pending.get("kind") not in (KIND_EVALUATE, KIND_DRAFT):
            return None
        kind = pending["kind"]
    else:
        kind = KIND_EVALUATE if isinstance(job, EvaluateJob) else KIND_DRAFT

    # One run per tab at a time. The in-process pattern gets this for free — a
    # running pipeline owns the script run, so no further widget event is
    # processed until it returns. Queue mode releases the script run as soon as
    # the UI stops waiting, so the guard has to be explicit: otherwise a user who
    # gives up on a slow run and clicks again queues a second digest over the
    # same records, doubling the LLM spend for one report.
    existing = st.session_state.get(_pending_key(slot))
    if isinstance(existing, str) and existing:
        try:
            record_in_flight = backend.get(existing)
            require_synthetic_queue()
        except Exception:  # noqa: BLE001 - fall through and let the submit fail loudly
            require_synthetic_queue()
            record_in_flight = None
        checking_confirmed = isinstance(pending, dict) and pending.get("confirmed_job_id") == existing
        if record_in_flight is not None and not record_in_flight.is_terminal and not checking_confirmed:
            run_log_event(
                kind,
                "rejected",
                request_id=request_id,
                error=f"run {existing} already in progress",
                reason="already_running",
            )
            pilot.display(
                "A run is already in progress for these inputs — wait for it above, or "
                "reload this page. Only one run per tab is submitted at a time."
            , container=st, method="warning")
            return None

    # The submit span is what the worker's run span hangs off: it wraps encoding
    # (which injects the trace context into the payload) and the enqueue itself.
    with tracing.phase_span(
        "queue:submit", kind=kind, files=files, pages=pages, backend=backend.name
    ):
        owner = pilot.current_owner()
        try:
            if isinstance(pending, dict):
                if pending["owner_id"] != owner or pending["kind"] != kind:
                    raise PayloadError("The previous submission belongs to another session.")
                payload = pending["payload"]
                decode_job(kind, payload, blob_store=get_blob_store())
                request_id = pending["request_id"]
                files, pages = pending["files"], pending["pages"]
                condition, sources = pending["condition"], pending["sources"]
                action_label = pending["action_label"]
                pilot.display("Checking the earlier submission using its saved inputs. "
                              "Recent edits will apply only after you discard that attempt.",
                              container=st, method="info")
            else:
                if job is None:
                    raise PayloadError("No earlier submission is available to check.")
                payload = _encode_payload(kind, job)
                st.session_state[submission_key] = {
                    "payload": payload, "request_id": request_id, "owner_id": owner,
                    "kind": kind, "files": files, "pages": pages,
                    "condition": condition, "sources": sources, "action_label": action_label,
                }
        except (PayloadError, BlobStoreError) as exc:
            require_synthetic_queue()
            run_log_event(kind, "rejected", request_id=request_id, error=str(exc), reason="payload")
            pilot.display(
                report_failure(
                    f"{action_label} could not be queued: {exc}",
                    phase=f"{kind}_queue_payload",
                    exc=exc,
                    request_id=request_id,
                )
            , container=st, method="error")
            return None
        try:
            record = backend.enqueue(kind, payload, request_id=request_id, owner_id=owner)
            require_synthetic_queue()
        except Exception as exc:  # noqa: BLE001 - any enqueue failure must surface, not crash the tab
            require_synthetic_queue()
            run_log_event(
                kind, "error", request_id=request_id,
                error=f"{type(exc).__name__}: {exc}", reason="submission_unconfirmed",
            )
            pilot.display(
                report_failure(
                    f"{action_label} submission is unconfirmed: {type(exc).__name__}: {exc}. "
                    "It may already be queued. Retry to check the same submission before "
                    "starting another run.",
                    phase=f"{kind}_enqueue",
                    exc=exc,
                    request_id=request_id,
                )
            , container=st, method="error")
            return None

    if record.recovery_available:
        st.session_state.pop(submission_key, None)
    else:
        # Keep exact inputs for a safe reference-repair retry, including while
        # the confirmed job is still running. Never submit current edits here.
        st.session_state[submission_key]["confirmed_job_id"] = record.job_id
    st.session_state[_pending_key(slot)] = record.job_id
    st.session_state[_pending_request_key(slot)] = request_id
    if not record.recovery_available:
        pilot.display(
            "This run is confirmed, but its recovery reference could not be restored. "
            "Keep this tab open. Check the earlier submission again after queue capacity "
            "is available to restore recovery from another session.",
            container=st, method="warning",
        )
    # The producer committed recovery indexing with the job, before returning.
    # The worker writes the audit start/ok/error pair, so the web pod records
    # only that the work was handed off — one audit record per run either way.
    run_log_event(
        kind, "queued", request_id=request_id, pages=pages, job_id=record.job_id
    )
    if condition:
        pilot.display(f"Queued as `{record.job_id}` — a worker is picking this up.", container=st, method="caption")
    logger.info(
        "queued %s job job_id=%s pages=%d",
        kind,
        record.job_id,
        pages,
        extra={"request_id": request_id, "phase": kind, "status": "queued", "pages": pages},
    )
    if not wait_for_result and not record.is_terminal:
        if record.recovery_available:
            pilot.display(f"Submission confirmed — recovery reference `{record.request_id}` is available. "
                          "The worker is still processing this run.", container=st, method="info")
        return QueueOutcome(ok=False, still_running=True, request_id=record.request_id)
    wait_budget = float(config.PIPELINE_TIMEOUT_SECONDS) + UI_WAIT_SLACK_SECONDS
    outcome = (_poll(backend, record.job_id, slot, wait_seconds=wait_budget)
               if wait_for_result else _fetch_outcome(backend, record, slot))
    if outcome.ok or not outcome.still_running:
        # Terminal (done or failed): nothing left to resume.
        st.session_state.pop(_pending_key(slot), None)
    if not outcome.ok:
        _render_failure(outcome, action_label)
    return outcome


@synthetic_queue
def _render_uncertain_submission(slot: str) -> None:
    pending = st.session_state.get(f"{slot}_queue_submission")
    if not isinstance(pending, dict):
        return
    pilot.display(
        (f"An earlier run is confirmed, but reference recovery is unavailable "
         f"(reference `{pending['request_id']}`). " if pending.get("confirmed_job_id") else
         f"An earlier submission is unconfirmed (reference `{pending['request_id']}`). ") +
        "Checking it uses the inputs saved with that attempt. To use your recent edits, "
        "discard the earlier attempt first. It may still run and incur charges.",
        container=st, method="warning",
    )
    if st.button("Check earlier submission", key=f"check_submission_{slot}"):
        outcome = submit_job(
            slot=slot, job=None, request_id=pending["request_id"],
            condition=pending["condition"], sources=pending["sources"],
            files=pending["files"], pages=pending["pages"], action_label=pending["action_label"],
            wait_for_result=False,
        )
        if outcome is not None and outcome.ok:
            pilot.display(f"Run result available — reference `{outcome.request_id}`.", container=st, method="info")
        return
    discard_allowed = st.checkbox(
        "I understand the earlier job may still run; I want to start a separate run.",
        key=f"confirm_discard_submission_{slot}",
    )
    if st.button("Discard earlier submission", key=f"discard_submission_{slot}",
                 disabled=not discard_allowed) and discard_allowed:
        st.session_state.pop(f"{slot}_queue_submission", None)
        if st.session_state.get(_pending_key(slot)) == pending.get("confirmed_job_id"):
            st.session_state.pop(_pending_key(slot), None)
        if st.session_state.get(_pending_request_key(slot)) == pending["request_id"]:
            st.session_state.pop(_pending_request_key(slot), None)
        pilot.display("Earlier attempt discarded. Your next submission will use the current inputs.",
                      container=st, method="info")


def resume_pending_job(slot: str, *, action_label: str) -> None:
    """Re-attach to a run queued earlier in this session, if any.

    Called on every render of a tab. When queue mode is active a pending job is
    recovered from one of three locations, in order:

    1. The pending job_id stored in ``st.session_state`` (present across reruns
       within the same browser session).
    2. The request_id stored in ``st.session_state`` (survives the same way,
       and is backed by the recovery index in Redis so step 3 applies).
    3. A recovery token entered by the user into the recovery input field
       rendered below this call.

    Without steps 2 and 3, a queued run outlives the browser session but the UI
    cannot find it — so a user who reloads mid-run sees an empty tab and
    re-submits work that is already 20 minutes into the digest.
    """
    if not synthetic_mode() or not queue_mode_active():
        return
    _render_uncertain_submission(slot)
    job_id = st.session_state.get(_pending_key(slot))
    if not isinstance(job_id, str) or not job_id:
        # No direct job_id, but maybe we still have the request_id from this
        # session — try to look it up in the recovery index.
        request_id = st.session_state.get(_pending_request_key(slot))
        if isinstance(request_id, str) and request_id:
            job_id = _resolve_job_by_request_id(request_id)
            if job_id is not None:
                # Found it: re-hydrate the session so the normal path proceeds.
                st.session_state[_pending_key(slot)] = job_id
            else:
                # The reference maps to nothing (expired, or from another
                # deployment): drop it rather than looking it up again on every
                # rerun. render_recovery_form still offers manual entry.
                st.session_state.pop(_pending_request_key(slot), None)
    if not isinstance(job_id, str) or not job_id:
        return
    try:
        record = get_job_backend().get(job_id)
        require_synthetic_queue()
    except Exception:  # noqa: BLE001 - never break the tab over a status read
        require_synthetic_queue()
        return
    if record is None:
        st.session_state.pop(_pending_key(slot), None)
        st.session_state.pop(_pending_request_key(slot), None)
        return
    if not pilot.owns(record):
        st.session_state.pop(_pending_key(slot), None)
        st.session_state.pop(_pending_request_key(slot), None)
        pilot.display("Run unavailable for this session.", container=st, method="warning")
        return None
    if record.is_terminal:
        outcome = _fetch_outcome(get_job_backend(), record, slot)
        st.session_state.pop(_pending_key(slot), None)
        if not outcome.ok:
            _render_failure(outcome, action_label)
        return
    pilot.display(
        f"A run you started earlier is still in progress on a worker "
        f"({record.progress * 100:.0f}% — {record.message or 'running'}). "
        f"Reference: `{record.request_id or job_id}`."
    , container=st, method="info")
    if st.button("⏳ Wait for it to finish", key=f"resume_{slot}"):
        outcome = _poll(
            get_job_backend(),
            job_id,
            slot,
            wait_seconds=float(config.PIPELINE_TIMEOUT_SECONDS) + UI_WAIT_SLACK_SECONDS,
        )
        if outcome.ok or not outcome.still_running:
            st.session_state.pop(_pending_key(slot), None)
        if not outcome.ok:
            _render_failure(outcome, action_label)
        else:
            st.rerun()


@synthetic_queue
def _resolve_job_by_request_id(request_id: str) -> str | None:
    """Look up a job_id from the recovery index in the job backend."""
    try:
        return get_job_backend().lookup_by_request_id(request_id)
    except Exception:  # noqa: BLE001
        return None


@synthetic_queue
def recover_job_by_request_id(
    slot: str, request_id: str, *, action_label: str
) -> QueueOutcome | None:
    """Recover a queued job by its request_id when the session state is lost.

    Called from the recovery form (``render_recovery_form``) and by
    ``resume_pending_job`` when the pending-key is missing but the pending-request
    key is still present in session state. Performs the same lookup, poll, and
    hydrate steps that ``resume_pending_job`` does for a direct job_id.
    """
    if not queue_mode_active() or not request_id:
        return None
    job_id = _resolve_job_by_request_id(request_id)
    if job_id is None:
        pilot.display(
            f"No queued job found for reference `{request_id}` — it may have expired "
            f"(results live for {config.JOB_QUEUE_TTL_SECONDS // 3600}h) or the reference "
            f"may be from a different deployment."
        , container=st, method="warning")
        return None
    st.session_state[_pending_key(slot)] = job_id
    st.session_state[_pending_request_key(slot)] = request_id
    try:
        record = get_job_backend().get(job_id)
        require_synthetic_queue()
    except Exception:  # noqa: BLE001
        require_synthetic_queue()
        return None
    if record is None:
        st.session_state.pop(_pending_key(slot), None)
        return None
    if not pilot.owns(record):
        st.session_state.pop(_pending_key(slot), None)
        st.session_state.pop(_pending_request_key(slot), None)
        pilot.display("Run unavailable for this session.", container=st, method="warning")
        return None
    if record.is_terminal:
        outcome = _fetch_outcome(get_job_backend(), record, slot)
        st.session_state.pop(_pending_key(slot), None)
        if not outcome.ok:
            _render_failure(outcome, action_label)
        return outcome
    # Still running — re-poll.
    outcome = _poll(
        get_job_backend(),
        job_id,
        slot,
        wait_seconds=float(config.PIPELINE_TIMEOUT_SECONDS) + UI_WAIT_SLACK_SECONDS,
    )
    if outcome.ok or not outcome.still_running:
        st.session_state.pop(_pending_key(slot), None)
    if not outcome.ok:
        _render_failure(outcome, action_label)
    return outcome


def render_recovery_form(slot: str, *, action_label: str) -> None:
    """Render a text input that lets a user recover a previous run by its reference.

    Displayed when no cached result is present AND no pending job is attached to
    this session — exactly the case where a browser restart or pod failover lost
    the session state that normally routes the UI back to the backend.
    """
    if not synthetic_mode() or not queue_mode_active():
        return
    # Only show the recovery form when there is no result to display and no
    # pending job is already attached.
    result_key, _, _ = _RESULT_KEYS[slot]
    if st.session_state.get(result_key) is not None:
        return
    if st.session_state.get(_pending_key(slot)):
        return

    with st.expander("🔍 Recover a previous run", expanded=False):
        pilot.display(
            "Lost your browser tab or restarted? Paste the reference you saved "
            "(it looks like `req_abc123…`) to recover your completed run."
        , container=st, method="caption")
        ref = st.text_input(
            "Run reference (req_…)",
            key=f"recover_{slot}_ref",
            placeholder="req_…",
        )
        if st.button("Recover", key=f"recover_{slot}_btn") and ref:
            with st.spinner("Looking up your run …"):
                outcome = recover_job_by_request_id(slot, ref.strip(), action_label=action_label)
            if outcome is not None and outcome.ok:
                st.rerun()


def queue_status_line() -> str:
    """Short description of the active backend, for a tab caption.

    Deliberately does no I/O: this renders on every rerun of the tab, and the
    Upstash tier's ``depth()`` is two HTTP requests. Backlog depth belongs in
    ``GET /health`` (``job_queue.depth``), which monitoring polls on a schedule.
    """
    if not synthetic_mode():
        return "queued processing excluded"
    try:
        backend = get_job_backend()
    except Exception as exc:  # noqa: BLE001
        if not synthetic_mode():
            return "queued processing excluded"
        return f"job queue unavailable ({type(exc).__name__})"
    if not synthetic_mode():
        return "queued processing excluded"
    return f"handed to a worker via {backend.name}"
