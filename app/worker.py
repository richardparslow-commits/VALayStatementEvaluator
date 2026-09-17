"""Worker process that executes queued Evaluate/Draft runs.

Run it alongside the Streamlit pods (``DEPLOYMENT.md`` → Pattern C):

    python -m app.worker                 # serve until SIGTERM
    python -m app.worker --once          # drain one job and exit (smoke check)
    python -m app.worker --drain-stale   # re-queue abandoned jobs, then exit

The worker is the process that actually spends the CPU and memory: it claims a
job, rebuilds the extracted record text from the payload, runs
``run_evaluation`` / ``run_draft`` under the same memory and timeout guards the
in-process path uses, and writes the result back. The Streamlit pod that
submitted the job only polls a small status key — which is what removes the
"one user with a 2,000-page bundle pins one pod" hot spot.

Each job is registered with :mod:`app.shutdown` while it runs, so SIGTERM
finishes the current job before the orchestrator's SIGKILL instead of losing 35
minutes of work. Long jobs are protected from worker death by a lease: if a
worker is SIGKILLed, the job's lease expires and :func:`drain_stale` (or the next
worker's periodic sweep) hands it to a healthy worker.

Audit parity: the worker emits the same ``start``/``ok``/``error`` audit events
the in-process path does, keyed by the ``req_…`` reference the UI shows. The
submit path deliberately skips its own ``start`` event in queue mode so a run
produces exactly one audit record either way.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from . import audit as audit_log
from . import config
from . import shutdown
from . import tracing
from .blob_store import BlobStoreError, get_blob_store
from .draft import DraftResult
from .evaluate import EvaluationResult
from .job_payload import (
    KIND_DRAFT,
    KIND_EVALUATE,
    DraftJob,
    EvaluateJob,
    PayloadError,
    RunResult,
    decode_job,
    encode_result,
)
from .job_queue import KINDS, JobBackend, JobQueueError, JobRecord, get_job_backend
from .llm import LLMClient, LLMError
from .logging_config import configure_logging, get_logger, set_request_id
from .pipeline_guard import (
    PipelineTimeoutError,
    check_memory_before_run,
    run_with_timeout,
)
from .profiler import RunProfiler, get_profiler
from .run_log import run_log_event

logger = get_logger("app.worker")

ProgressCallback = Callable[[float, str], None]

# Progress writes are throttled so a 2,000-page run's per-chunk callbacks do not
# become hundreds of Redis round-trips per job. The final 100% update always
# goes through, so the UI never sticks below complete.
_PROGRESS_MIN_INTERVAL_SECONDS = 0.5
# How often an idle worker sweeps for jobs abandoned by a dead worker.
_STALE_SWEEP_INTERVAL_SECONDS = 60.0
# Cap on retained claim errors per worker, so a persistently broken backend does
# not grow this list without bound.
_MAX_RETAINED_ERRORS = 20


class WorkerConfigError(RuntimeError):
    """Raised when the worker cannot run at all (missing configuration)."""


@dataclass
class WorkerStats:
    """Counters for one worker's lifetime (logged on exit)."""

    claimed: int = 0
    completed: int = 0
    failed: int = 0
    requeued: int = 0
    errors: list[str] = field(default_factory=list)


def default_worker_id() -> str:
    """Stable-ish worker identity: env override, else hostname:pid."""
    override = config.JOB_QUEUE_WORKER_ID
    if override:
        return override
    return f"{socket.gethostname()}:{os.getpid()}"


# ------------------------------------------------------------------- execution
def _progress_callback(backend: JobBackend, record: JobRecord) -> ProgressCallback:
    """Return a throttled progress callback that persists to the queue."""
    last_sent = 0.0

    def update(fraction: float, message: str) -> None:
        nonlocal last_sent
        now = time.monotonic()
        if fraction < 1.0 and (now - last_sent) < _PROGRESS_MIN_INTERVAL_SECONDS:
            return
        last_sent = now
        try:
            backend.set_progress(record.job_id, fraction, message)
        except Exception as exc:  # noqa: BLE001 - progress must never kill a run
            logger.warning(
                "progress update failed job_id=%s error=%s",
                record.job_id,
                f"{type(exc).__name__}: {exc}",
                extra={"phase": "worker", "status": "warning"},
            )

    return update


def build_llm() -> LLMClient:
    """Build the worker's LLM client from environment/secrets configuration.

    Deliberately not the view-layer ``get_llm()``: a worker has no browser
    session, so a key typed into the sidebar cannot reach it. Queue mode
    therefore requires the key in the worker's environment or deployed secret —
    the submit path enforces the same rule up front and says so in the UI.
    """
    settings = config.load_settings()
    if not settings.configured:
        raise WorkerConfigError(
            "the LLM API key is not configured in this worker's environment. Queue "
            "mode cannot use a key typed into the sidebar — set OPENAI_API_KEY in the "
            "worker's environment or secret (see DEPLOYMENT.md → Pattern C)."
        )
    try:
        return LLMClient(settings)
    except LLMError as exc:
        raise WorkerConfigError(str(exc)) from exc


def _run_pipeline(
    kind: str,
    job: EvaluateJob | DraftJob,
    llm: LLMClient,
    progress: Any,
) -> RunResult:
    """Execute the queued pipeline and return its result plus usage."""
    from .draft import run_draft
    from .evaluate import run_evaluation

    if isinstance(job, EvaluateJob):
        evaluation = run_evaluation(llm, job.statement_text, job.records, progress=progress)
        return RunResult(
            kind=kind, result=evaluation, usage=llm.usage, request_id=job.request_id
        )
    if isinstance(job, DraftJob):
        draft = run_draft(
            llm,
            job.records,
            job.witness,
            job.observations,
            job.condition,
            job.claim_type,
            progress=progress,
        )
        return RunResult(kind=kind, result=draft, usage=llm.usage, request_id=job.request_id)
    raise PayloadError(f"unsupported job payload for kind {kind!r}")  # pragma: no cover


def _outcome_for(kind: str, result: EvaluationResult | DraftResult) -> dict[str, Any]:
    """Classification-only outcome for the audit/run log (never free text)."""
    try:
        if kind == KIND_EVALUATE and isinstance(result, EvaluationResult):
            return {
                "claims": len(result.claims),
                "contradictions": result.contradiction_count,
                "overall_rating": result.overall_rating,
            }
        if isinstance(result, DraftResult):
            return {
                "draft_chars": len(result.output_statement),
                "grounding_items": len(result.grounding or {}),
            }
    except Exception:  # noqa: BLE001 - reporting must never fail a good run
        pass
    return {}


def _audit_start(
    kind: str,
    *,
    request_id: str,
    condition: str | None,
    sources: list[str],
    files: int,
    pages: int,
) -> None:
    """Emit the audit ``start`` event for the kind being run (best-effort)."""
    kwargs: dict[str, Any] = {
        "request_id": request_id,
        "condition": condition,
        "record_sources": sources or None,
        "record_files": files,
        "record_pages": pages,
    }
    try:
        if kind == KIND_EVALUATE:
            audit_log.audit_evaluate_start(**kwargs)
        else:
            audit_log.audit_draft_start(**kwargs)
    except Exception:  # noqa: BLE001 - audit is best-effort
        pass


def _audit_ok(
    kind: str,
    *,
    request_id: str,
    duration_ms: int,
    condition: str | None,
    sources: list[str],
    files: int,
    pages: int,
    outcome: dict[str, Any],
    endpoints: list[str] | None = None,
) -> None:
    kwargs: dict[str, Any] = {
        "request_id": request_id,
        "duration_ms": duration_ms,
        "condition": condition,
        "record_sources": sources or None,
        "record_files": files,
        "record_pages": pages,
        "outcome": outcome or None,
        "llm_endpoints": endpoints or None,
    }
    try:
        if kind == KIND_EVALUATE:
            audit_log.audit_evaluate_ok(**kwargs)
        else:
            audit_log.audit_draft_ok(**kwargs)
    except Exception:  # noqa: BLE001 - audit is best-effort
        pass


def execute_job(
    record: JobRecord,
    payload: str,
    backend: JobBackend,
    *,
    llm: LLMClient | None = None,
) -> bool:
    """Claim-to-completion body for one job. Returns True on success.

    Kept separate from the claim loop so ``--once``, the test suite, and any
    future threaded pool share exactly one execution path.
    """
    kind = record.kind
    # The payload's request_id is the one the submit path showed the user, so it
    # wins over the queue record's copy; the record is the fallback for a payload
    # that never decodes.
    request_id = record.request_id or "-"
    set_request_id(request_id)
    progress = _progress_callback(backend, record)
    profiler_run = (
        RunProfiler(action=kind, request_id=request_id, run_start_mono=time.monotonic())
        if get_profiler()
        else None
    )
    t0 = time.perf_counter()
    audit_condition: str | None = None
    sources: list[str] = []
    files = 0
    pages = 0
    started = False
    try:
        # Also resolves a documents reference when the web pod externalized a large
        # record bundle to the blob store (see app/blob_store.py).
        job = decode_job(kind, payload, blob_store=get_blob_store())
        if job.request_id:
            request_id = job.request_id
            set_request_id(request_id)
        if isinstance(job, DraftJob):
            audit_condition = job.condition.strip()[:120] or None
        sources = list(job.record_sources)
        files = len(job.records)
        pages = sum(len(doc.pages) for doc in job.records)
        # Resolve the client before the audit start: a misconfigured worker
        # fails every job, and pairing an audit "start" with a config error that
        # never touched the pipeline would misrepresent the run.
        client = llm if llm is not None else build_llm()
        _audit_start(
            kind,
            request_id=request_id,
            condition=audit_condition,
            sources=sources,
            files=files,
            pages=pages,
        )
        started = True
        logger.info(
            "worker job start job_id=%s kind=%s pages=%d attempts=%d",
            record.job_id,
            kind,
            pages,
            record.attempts,
            extra={
                "request_id": request_id,
                "phase": kind,
                "status": "start",
                "pages": pages,
            },
        )
        run_log_event(kind, "start", request_id=request_id, pages=pages)
        check_memory_before_run()
        # Continue the web pod's trace: the payload carries its W3C trace context
        # (empty when tracing is off), so this run's spans join the submit span
        # instead of appearing as an unrelated trace from a different service.
        with tracing.attach_trace_context(job.trace_context):
            run = run_with_timeout(_run_pipeline, kind, job, client, progress)
    except (PayloadError, BlobStoreError, WorkerConfigError) as exc:
        _fail(
            backend, record, exc, type(exc).__name__,
            request_id=request_id, started=started, condition=audit_condition,
        )
        return False
    except MemoryError as exc:
        _fail(
            backend, record, exc, type(exc).__name__,
            request_id=request_id, started=started, condition=audit_condition,
        )
        return False
    except PipelineTimeoutError as exc:
        _fail(
            backend, record, exc, type(exc).__name__,
            request_id=request_id, started=started, condition=audit_condition,
        )
        return False
    except BaseException as exc:  # noqa: BLE001 - a worker survives any job failure
        _fail(
            backend, record, exc, type(exc).__name__,
            request_id=request_id, started=started, condition=audit_condition,
        )
        return False

    duration_ms = int((time.perf_counter() - t0) * 1000)
    try:
        backend.store_result(record.job_id, encode_result(run))
    except Exception as exc:  # noqa: BLE001 - an unstorable result is a failed job
        _fail(
            backend, record, exc, type(exc).__name__,
            request_id=request_id, started=started, condition=audit_condition,
        )
        return False
    try:
        backend.complete(record.job_id, message=f"completed in {duration_ms / 1000:.0f}s")
    except Exception as exc:  # noqa: BLE001 - result is stored; state write is best-effort
        logger.error(
            "could not mark job complete job_id=%s error=%s",
            record.job_id,
            f"{type(exc).__name__}: {exc}",
            extra={"phase": "worker", "status": "error"},
        )
    totals = run.usage.totals()
    # Preserved across the queue round trip (see job_payload), so the record of a
    # run served by the backup provider survives to the audit log and the run log.
    endpoints = run.usage.endpoints_used()
    outcome = _outcome_for(kind, run.result)
    logger.info(
        "worker job done job_id=%s kind=%s duration_ms=%d calls=%d",
        record.job_id,
        kind,
        duration_ms,
        totals.calls,
        extra={
            "request_id": request_id,
            "phase": kind,
            "status": "ok",
            "duration_ms": duration_ms,
            "calls": totals.calls,
        },
    )
    run_log_event(
        kind,
        "ok",
        request_id=request_id,
        duration_ms=duration_ms,
        endpoints=",".join(endpoints),
        **outcome,
    )
    _audit_ok(
        kind,
        request_id=request_id,
        duration_ms=duration_ms,
        condition=audit_condition,
        sources=sources,
        files=files,
        pages=pages,
        outcome=outcome,
        endpoints=endpoints,
    )
    if profiler_run is not None:
        profiler_run.run_end_mono = time.monotonic()
        profiler_run.emit()
        get_profiler().record_run(profiler_run)
    return True


def _fail(
    backend: JobBackend,
    record: JobRecord,
    exc: BaseException,
    error_class: str,
    *,
    request_id: str,
    started: bool,
    condition: str | None = None,
) -> None:
    """Record a job failure in the queue, logs, run log, and audit trail."""
    detail = f"{type(exc).__name__}: {exc}"
    logger.error(
        "worker job failed job_id=%s kind=%s error=%s",
        record.job_id,
        record.kind,
        detail,
        exc_info=exc,
        extra={
            "request_id": request_id,
            "phase": record.kind,
            "status": "error",
            "error_class": error_class,
        },
    )
    run_log_event(
        record.kind,
        "error",
        request_id=request_id,
        error=detail,
        error_class=error_class,
        traceback=traceback.format_exc(limit=8),
    )
    # Only a job that reached the pipeline gets an audit error: a payload that
    # never decoded emitted no audit "start" to pair with.
    if started:
        try:
            if record.kind == KIND_EVALUATE:
                audit_log.audit_evaluate_error(
                    request_id=request_id,
                    duration_ms=0,
                    error=exc,
                    condition=condition,
                )
            else:
                audit_log.audit_draft_error(
                    request_id=request_id,
                    duration_ms=0,
                    error=exc,
                    condition=condition,
                )
        except Exception:  # noqa: BLE001 - audit is best-effort
            pass
    try:
        backend.fail(record.job_id, error=detail, error_class=error_class)
    except Exception as write_exc:  # noqa: BLE001
        logger.error(
            "could not record job failure job_id=%s error=%s",
            record.job_id,
            f"{type(write_exc).__name__}: {write_exc}",
            extra={"phase": "worker", "status": "error"},
        )


# ---------------------------------------------------------------------- loop
def drain_stale(backend: JobBackend | None = None) -> int:
    """Re-queue jobs whose worker died. Returns the number re-queued."""
    active = backend if backend is not None else get_job_backend()
    try:
        count = active.requeue_stale()
    except Exception as exc:  # noqa: BLE001
        logger.warning("stale job sweep failed: %s", f"{type(exc).__name__}: {exc}")
        return 0
    if count:
        logger.warning(
            "re-queued %d job(s) whose worker lease expired",
            count,
            extra={"phase": "worker", "status": "requeued", "jobs": count},
        )
    return count


def run_worker(
    *,
    once: bool = False,
    backend: JobBackend | None = None,
    llm: LLMClient | None = None,
    worker_id: str | None = None,
    stop_event: threading.Event | None = None,
    max_jobs: int | None = None,
) -> WorkerStats:
    """Serve queued jobs until stopped. Returns lifetime counters.

    ``once`` makes one non-blocking claim attempt and returns, which is what a
    canary/smoke check wants. ``stop_event`` lets tests end the loop without
    signals; the real entrypoint trips it from SIGTERM/SIGINT via
    :mod:`app.shutdown`.
    """
    active = backend if backend is not None else get_job_backend()
    me = worker_id or default_worker_id()
    stopper = stop_event or threading.Event()
    stats = WorkerStats()
    last_sweep = time.monotonic()
    logger.info(
        "worker ready id=%s backend=%s distributed=%s",
        me,
        active.name,
        active.is_distributed,
        extra={"phase": "worker", "status": "ready"},
    )
    if not active.is_distributed:
        logger.warning(
            "worker is using a non-distributed backend (%s): jobs submitted by a "
            "Streamlit process in another interpreter will never be claimed. Set "
            "VA_LSE_REDIS_URL or VA_LSE_SHARED_CACHE_URL/_TOKEN for real Pattern C.",
            active.name,
            extra={"phase": "worker", "status": "warning"},
        )
    while not stopper.is_set():
        if not once and shutdown.is_shutting_down():
            logger.info("shutdown requested; worker stopping claim loop")
            break
        if (time.monotonic() - last_sweep) >= _STALE_SWEEP_INTERVAL_SECONDS:
            last_sweep = time.monotonic()
            stats.requeued += drain_stale(active)
        try:
            claimed = active.claim(KINDS, worker_id=me)
        except JobQueueError as exc:
            if len(stats.errors) < _MAX_RETAINED_ERRORS:
                stats.errors.append(str(exc))
            logger.error("claim failed: %s", exc, extra={"phase": "worker", "status": "error"})
            if once:
                return stats
            time.sleep(max(1.0, config.JOB_QUEUE_POLL_SECONDS))
            continue
        if claimed is None:
            if once:
                return stats
            continue
        record, payload = claimed
        stats.claimed += 1
        if not shutdown.enter_run():
            # Draining: hand the job straight back so another worker picks it up
            # instead of starting work this process cannot finish.
            logger.warning(
                "shutdown in progress; re-queueing claimed job job_id=%s", record.job_id
            )
            try:
                active.requeue(record.job_id, reason="re-queued: worker was draining")
            except Exception:  # noqa: BLE001
                pass
            stats.requeued += 1
            break
        try:
            ok = execute_job(record, payload, active, llm=llm)
        finally:
            shutdown.exit_run()
        if ok:
            stats.completed += 1
        else:
            stats.failed += 1
        if once or (max_jobs is not None and stats.claimed >= max_jobs):
            return stats
    return stats


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: ``python -m app.worker``."""
    parser = argparse.ArgumentParser(
        prog="python -m app.worker",
        description=(
            "Execute queued Evaluate/Draft runs from the distributed job queue "
            "(DEPLOYMENT.md → Pattern C)."
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="claim at most one job, then exit (non-blocking)",
    )
    parser.add_argument(
        "--drain-stale",
        action="store_true",
        help="re-queue abandoned jobs and exit without claiming new work",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="exit after this many claimed jobs (useful for canaries)",
    )
    args = parser.parse_args(argv)

    configure_logging()
    try:
        audit_log.configure_audit_logging()
    except Exception:  # noqa: BLE001 - audit is best-effort
        pass
    # No-op unless VA_LSE_TRACING=1 (and the OTel packages are installed).
    tracing.setup_tracing(role="worker")

    try:
        backend = get_job_backend()
    except Exception as exc:  # noqa: BLE001
        logger.error("worker could not build a job backend: %s", exc)
        return 2

    if args.drain_stale:
        count = drain_stale(backend)
        print(f"re-queued {count} stale job(s)")
        return 0

    if not args.once:
        # Reuse the app-wide drain: the in-flight job is registered through
        # shutdown.enter_run(), so SIGTERM waits for it rather than dropping it.
        shutdown.install_signal_handlers()
        _start_health_server()

    try:
        stats = run_worker(once=args.once, backend=backend, max_jobs=args.max_jobs)
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        logger.info("worker interrupted")
        return 130
    logger.info(
        "worker exiting claimed=%d completed=%d failed=%d requeued=%d",
        stats.claimed,
        stats.completed,
        stats.failed,
        stats.requeued,
        extra={"phase": "worker", "status": "stopped"},
    )
    for err in stats.errors[:5]:
        logger.warning("worker error: %s", err)
    # Flush buffered spans before the process exits; a BatchSpanProcessor holds
    # the most recent ones in memory, which is exactly the tail of a long run.
    tracing.shutdown_tracing()
    return 0


def _start_health_server() -> None:
    """Start the worker's liveness/readiness sidecar (best-effort)."""
    port = config.JOB_QUEUE_WORKER_HEALTH_PORT
    if port <= 0:
        return
    try:
        from .health import start_health_server

        start_health_server(port)
    except Exception as exc:  # noqa: BLE001 - health is best-effort
        logger.warning("worker health server unavailable: %s", exc)


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())
