"""Distributed job queue for Evaluate/Draft runs.

Why this exists
---------------
Streamlit's session is a live WebSocket bound to one server process, so session
affinity is mandatory no matter how session *data* is stored (see
``DEPLOYMENT.md`` → Pattern C). Affinity alone therefore decides where the
heaviest work in this app runs: ``review_medical_records`` executes inside the
script run, on the pod that owns the user's socket, peaking around 1.8 GB on a
2,000-page bundle (``PERFORMANCE.md``). One user with a large record set keeps
*their* pod busy while others idle — and a pod restart destroys 35 minutes of
work.

This module moves the work, not the session. The web pod serializes the run
(statement/observations + extracted record text), enqueues it, and polls a small
status key; a separate worker deployment claims it and runs the pipeline. The
pod that serves the browser stays thin, any pod can render any job's result, and
a killed worker's job is re-queued for the next one.

Backends
--------
``build_job_backend`` selects a transport from what is configured:

1. ``VA_LSE_REDIS_URL`` → redis-py against in-cluster Redis (no extra infra
   beyond the StatefulSet already in ``deploy/k8s/k8s-redis.yaml``).
2. ``VA_LSE_SHARED_CACHE_URL`` + ``_TOKEN`` → the Upstash REST tier the
   reference cache already uses, via the JSON command API. No new dependency.
3. Neither → :class:`InProcessJobBackend`, a single-process queue used by dev
   runs and the test suite so this code path is exercised without external
   services. It is **not** distributed and logs that at WARNING.

The queue is opt-in (``VA_LSE_JOB_QUEUE=1``): with it off, callers keep running
the pipeline in-process exactly as before.
"""
from __future__ import annotations

import base64
import contextlib
import json
import logging
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields
from typing import Any, Sequence

from . import config

logger = logging.getLogger("app.job_queue")

KIND_EVALUATE = "evaluate"
KIND_DRAFT = "draft"
KINDS: tuple[str, ...] = (KIND_EVALUATE, KIND_DRAFT)

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"
TERMINAL_STATUSES: tuple[str, ...] = (STATUS_DONE, STATUS_ERROR)

# A job is retried this many times before it is parked as failed. Guards against
# a payload that kills every worker (e.g. an OOM on a pathologically large
# bundle) being handed out forever.
MAX_ATTEMPTS = 3


class JobQueueError(RuntimeError):
    """Raised when the queue backend rejects an operation."""


class JobQueueUnavailable(JobQueueError):
    """Raised when a backend's dependency or configuration is missing."""


# --------------------------------------------------------------------- records
@dataclass
class JobRecord:
    """Everything the UI and worker need to know about one queued run.

    Deliberately metadata only: the statement/observations and record text live
    in the payload key, and the report lives in the result key, so listing or
    polling jobs never ships medical text.
    """

    job_id: str
    kind: str
    request_id: str = ""
    status: str = STATUS_QUEUED
    progress: float = 0.0
    message: str = ""
    worker_id: str = ""
    error: str = ""
    error_class: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    heartbeat_at: float = 0.0
    attempts: int = 0

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str | None) -> JobRecord | None:
        """Parse a record, tolerating absent/corrupt/legacy-shaped values."""
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {k: v for k, v in data.items() if k in known}
        try:
            record: JobRecord = cls(**kwargs)
        except TypeError:
            return None
        return record


def _meta_key(prefix: str, job_id: str) -> str:
    return f"{prefix}:job:{job_id}:meta"


def _payload_key(prefix: str, job_id: str) -> str:
    return f"{prefix}:job:{job_id}:payload"


def _result_key(prefix: str, job_id: str) -> str:
    return f"{prefix}:job:{job_id}:result"


def _queue_key(prefix: str, kind: str) -> str:
    return f"{prefix}:jobs:{kind}"


def _lease_key(prefix: str, kind: str) -> str:
    return f"{prefix}:jobs:{kind}:leases"


def new_job_id() -> str:
    """Return a short, log-safe job id (``job_…``)."""
    return f"job_{uuid.uuid4().hex[:16]}"


def _now() -> float:
    return time.time()


# -------------------------------------------------------------------- protocol
class JobBackend:
    """Storage contract shared by the queue backends.

    Not a ``typing.Protocol`` so the implementations can share the small amount
    of bookkeeping (record transitions) that must behave identically whether the
    transport is Redis, Upstash REST, or a dict.
    """

    name = "base"
    is_distributed = False
    # Whether ``depth()`` costs a network round trip. The in-process backend can
    # answer from a dict; every remote backend has to ask the transport, and for
    # Upstash that is two HTTP requests. ``health()`` uses this to decide whether
    # reading the backlog is safe on the liveness path.
    depth_is_remote = True

    def enqueue(self, kind: str, payload: str, *, request_id: str = "") -> JobRecord:
        raise NotImplementedError

    def claim(
        self, kinds: Sequence[str], *, worker_id: str
    ) -> tuple[JobRecord, str] | None:
        """Return ``(record, payload)`` for the next job, or None if idle."""
        raise NotImplementedError

    def set_progress(
        self, job_id: str, progress: float, message: str
    ) -> None:
        raise NotImplementedError

    def store_result(self, job_id: str, result: str) -> None:
        raise NotImplementedError

    def complete(self, job_id: str, *, message: str = "") -> None:
        raise NotImplementedError

    def fail(self, job_id: str, *, error: str, error_class: str = "") -> None:
        raise NotImplementedError

    def get(self, job_id: str) -> JobRecord | None:
        raise NotImplementedError

    def get_result(self, job_id: str) -> str | None:
        raise NotImplementedError

    def requeue(self, job_id: str, *, reason: str = "") -> bool:
        """Hand a claimed job back untouched, without burning an attempt.

        Used when a worker claims a job it then refuses to run (it is draining),
        so the work reaches a healthy worker immediately instead of waiting out
        the lease.
        """
        raise NotImplementedError

    def requeue_stale(self) -> int:
        """Re-queue jobs whose worker stopped heartbeating. Returns the count."""
        raise NotImplementedError

    def depth(self) -> int:
        raise NotImplementedError

    def ping(self) -> bool:
        raise NotImplementedError

    def health(self, probe: bool = False) -> dict[str, Any]:
        """Status for ``GET /health``, cheap by default.

        ``/health`` is a *liveness* probe: kubelet polls it every 10s and the
        endpoint is documented to answer in under 2s, which is why the shared
        cache is also read with ``probe=False``. Reading the backlog means
        ``LLEN`` per kind on Redis and two HTTP requests on Upstash, so on a slow
        or unreachable tier this would turn a queue problem into a failed
        liveness check — and Kubernetes would restart pods, discarding the very
        work the queue is holding. Depth is therefore read only when it is free
        (a local backend) or explicitly requested with ``probe=True``.

        When it is not read, ``depth`` is ``None`` rather than ``0``: Redis's
        ``depth()`` swallows connection errors and returns 0, so a default of 0
        would report a healthy empty queue at exactly the moment the truth is
        unknown. ``depth_source`` says which case this is.
        """
        payload: dict[str, Any] = {
            "backend": self.name,
            "is_distributed": self.is_distributed,
            "enabled": config.JOB_QUEUE_ENABLED,
        }
        if probe or not self.depth_is_remote:
            payload["depth"] = self.depth()
            payload["depth_source"] = "probed" if probe else "local"
        else:
            payload["depth"] = None
            payload["depth_source"] = "not_probed"
            payload["depth_note"] = (
                "omitted to keep /health free of network I/O; "
                "use GET /metrics?probe=1 or the sidebar's Check backlog button"
            )
        return payload


# ------------------------------------------------------------ in-process impl
class InProcessJobBackend(JobBackend):
    """Single-process queue: thread-safe dict + per-kind FIFO deques.

    Used for dev runs, ``--once`` worker smoke checks, and the test suite so the
    queue code path is real without external services. It cannot serve a worker
    in another process, so it pins ``is_distributed`` to False and says so in
    ``/health`` — a single-process "distributed" queue silently doing nothing is
    a worse failure than an honest one.
    """

    name = "inprocess"
    is_distributed = False
    depth_is_remote = False

    def __init__(self, *, prefix: str, ttl_seconds: int) -> None:
        self._prefix = prefix
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._records: dict[str, JobRecord] = {}
        self._payloads: dict[str, str] = {}
        self._results: dict[str, str] = {}
        self._queues: dict[str, deque[str]] = {kind: deque() for kind in KINDS}

    # -- producer -----------------------------------------------------------
    def enqueue(self, kind: str, payload: str, *, request_id: str = "") -> JobRecord:
        if kind not in KINDS:
            raise JobQueueError(f"unknown job kind: {kind}")
        record = JobRecord(
            job_id=new_job_id(),
            kind=kind,
            request_id=request_id,
            created_at=_now(),
            updated_at=_now(),
        )
        with self._cv:
            self._records[record.job_id] = record
            self._payloads[record.job_id] = payload
            self._queues.setdefault(kind, deque()).append(record.job_id)
            self._cv.notify_all()
        return record

    # -- consumer -----------------------------------------------------------
    def claim(
        self, kinds: Sequence[str], *, worker_id: str
    ) -> tuple[JobRecord, str] | None:
        deadline = time.monotonic() + float(config.JOB_QUEUE_CLAIM_TIMEOUT_SECONDS)
        with self._cv:
            while True:
                for kind in kinds:
                    queue = self._queues.get(kind)
                    if not queue:
                        continue
                    job_id = queue.popleft()
                    record = self._records.get(job_id)
                    payload = self._payloads.get(job_id)
                    # Only a *queued* job may be claimed. A duplicate queue entry
                    # (stale re-queue racing a live worker) must not start a
                    # second concurrent run of the same job.
                    if (
                        record is None
                        or payload is None
                        or record.status != STATUS_QUEUED
                    ):
                        continue
                    self._start(record, worker_id)
                    return record, payload
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(remaining)

    def _start(self, record: JobRecord, worker_id: str) -> None:
        record.status = STATUS_RUNNING
        record.worker_id = worker_id
        record.attempts += 1
        record.heartbeat_at = _now()
        record.updated_at = record.heartbeat_at
        record.message = "claimed by worker"

    def set_progress(self, job_id: str, progress: float, message: str) -> None:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return
            record.progress = min(max(progress, 0.0), 1.0)
            record.message = message
            record.heartbeat_at = _now()
            record.updated_at = record.heartbeat_at

    def store_result(self, job_id: str, result: str) -> None:
        with self._lock:
            self._results[job_id] = result

    def complete(self, job_id: str, *, message: str = "") -> None:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return
            record.status = STATUS_DONE
            record.progress = 1.0
            record.message = message or "completed"
            record.updated_at = _now()

    def fail(self, job_id: str, *, error: str, error_class: str = "") -> None:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return
            record.status = STATUS_ERROR
            record.error = error[:2000]
            record.error_class = error_class
            record.message = "failed"
            record.updated_at = _now()

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._records.get(job_id)

    def get_result(self, job_id: str) -> str | None:
        with self._lock:
            return self._results.get(job_id)

    def requeue(self, job_id: str, *, reason: str = "") -> bool:
        with self._cv:
            record = self._records.get(job_id)
            # Only a *running* job can be handed back; re-queueing an already
            # queued job would put it in the queue twice.
            if record is None or record.status != STATUS_RUNNING:
                return False
            record.status = STATUS_QUEUED
            record.attempts = max(0, record.attempts - 1)
            record.message = reason or "re-queued"
            record.updated_at = _now()
            self._queues.setdefault(record.kind, deque()).appendleft(job_id)
            self._cv.notify_all()
            return True

    def requeue_stale(self) -> int:
        cutoff = _now() - float(config.JOB_QUEUE_LEASE_SECONDS)
        requeued = 0
        with self._cv:
            for record in self._records.values():
                if record.status != STATUS_RUNNING or record.heartbeat_at > cutoff:
                    continue
                if record.attempts >= MAX_ATTEMPTS:
                    record.status = STATUS_ERROR
                    record.error = (
                        f"abandoned after {record.attempts} attempts "
                        "(worker lease expired each time)"
                    )
                    record.error_class = "JobAbandoned"
                    record.updated_at = _now()
                    continue
                record.status = STATUS_QUEUED
                record.message = "re-queued after worker lease expired"
                record.updated_at = _now()
                self._queues.setdefault(record.kind, deque()).appendleft(record.job_id)
                requeued += 1
            if requeued:
                self._cv.notify_all()
        return requeued

    def depth(self) -> int:
        with self._lock:
            return sum(len(q) for q in self._queues.values())

    def ping(self) -> bool:
        return True


# ------------------------------------------------------------- redis-py impl
class RedisJobBackend(JobBackend):
    """redis-py backend for in-cluster Redis (``VA_LSE_REDIS_URL``).

    Claiming polls ``RPOP`` rather than blocking on ``BRPOP`` so the same claim
    loop, lease bookkeeping, and timeout semantics apply to every backend —
    Upstash's REST API has no blocking pop, and one code path is easier to
    reason about than two.
    """

    name = "redis"
    is_distributed = True

    def __init__(
        self, url: str, *, prefix: str, ttl_seconds: int, timeout_seconds: float = 5.0
    ) -> None:
        try:
            import redis  # noqa: PLC0415 - optional dependency, imported lazily
        except ImportError as exc:  # pragma: no cover - depends on env
            raise JobQueueUnavailable(
                "redis package is not installed; pip install redis (or use the "
                "Upstash REST tier by unsetting VA_LSE_REDIS_URL)"
            ) from exc
        self._prefix = prefix
        self._ttl = ttl_seconds
        self._client: Any = redis.Redis.from_url(
            url,
            decode_responses=True,
            socket_timeout=timeout_seconds,
            socket_connect_timeout=timeout_seconds,
        )

    # -- helpers ------------------------------------------------------------
    @contextlib.contextmanager
    def _transport(self, operation: str) -> Iterator[None]:
        """Turn any redis-py failure into ``JobQueueError``.

        redis-py raises its own exception tree (``ConnectionError``, ``TimeoutError``,
        ``ResponseError``…). Letting those escape would kill the worker's claim loop
        with a traceback the moment Redis blinks, instead of the log-and-retry path
        ``run_worker`` already implements — and would sidestep the ``JobQueueError``
        handling every caller is written against.
        """
        try:
            yield
        except JobQueueError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize the whole redis.exceptions tree
            raise JobQueueError(
                f"redis {operation} failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _set_record(self, record: JobRecord) -> None:
        raw = record.to_json()
        with self._transport("set job record"):
            self._client.set(_meta_key(self._prefix, record.job_id), raw, ex=self._ttl)

    def _read_record(self, job_id: str) -> JobRecord | None:
        with self._transport("read job record"):
            raw = self._client.get(_meta_key(self._prefix, job_id))
        return JobRecord.from_json(raw if isinstance(raw, str) else None)

    def _touch_lease(self, record: JobRecord) -> None:
        with self._transport("update job lease"):
            self._client.zadd(
                _lease_key(self._prefix, record.kind), {record.job_id: record.heartbeat_at}
            )

    # -- producer -----------------------------------------------------------
    def enqueue(self, kind: str, payload: str, *, request_id: str = "") -> JobRecord:
        if kind not in KINDS:
            raise JobQueueError(f"unknown job kind: {kind}")
        record = JobRecord(
            job_id=new_job_id(),
            kind=kind,
            request_id=request_id,
            created_at=_now(),
            updated_at=_now(),
        )
        try:
            self._set_record(record)
            self._client.set(
                _payload_key(self._prefix, record.job_id), payload, ex=self._ttl
            )
            self._client.lpush(_queue_key(self._prefix, kind), record.job_id)
        except Exception as exc:  # noqa: BLE001 - surface as queue error
            raise JobQueueError(f"enqueue failed: {type(exc).__name__}: {exc}") from exc
        return record

    # -- consumer -----------------------------------------------------------
    def claim(
        self, kinds: Sequence[str], *, worker_id: str
    ) -> tuple[JobRecord, str] | None:
        deadline = time.monotonic() + float(config.JOB_QUEUE_CLAIM_TIMEOUT_SECONDS)
        poll = max(0.05, float(config.JOB_QUEUE_POLL_SECONDS))
        while True:
            for kind in kinds:
                with self._transport("claim"):
                    raw_id = self._client.rpop(_queue_key(self._prefix, kind))
                job_id = raw_id if isinstance(raw_id, str) else ""
                if not job_id:
                    continue
                record = self._read_record(job_id)
                with self._transport("read job payload"):
                    payload = self._client.get(_payload_key(self._prefix, job_id))
                # Only a queued job may be claimed, so a duplicate queue entry
                # cannot start a second concurrent run of the same job.
                if (
                    record is None
                    or not isinstance(payload, str)
                    or record.status != STATUS_QUEUED
                ):
                    continue
                record.status = STATUS_RUNNING
                record.worker_id = worker_id
                record.attempts += 1
                record.heartbeat_at = _now()
                record.updated_at = record.heartbeat_at
                record.message = "claimed by worker"
                self._set_record(record)
                self._touch_lease(record)
                return record, payload
            if time.monotonic() >= deadline:
                return None
            time.sleep(poll)

    def set_progress(self, job_id: str, progress: float, message: str) -> None:
        record = self._read_record(job_id)
        if record is None:
            return
        record.progress = min(max(progress, 0.0), 1.0)
        record.message = message
        record.heartbeat_at = _now()
        record.updated_at = record.heartbeat_at
        self._set_record(record)
        self._touch_lease(record)

    def store_result(self, job_id: str, result: str) -> None:
        with self._transport("store result"):
            self._client.set(_result_key(self._prefix, job_id), result, ex=self._ttl)

    def _finish(self, job_id: str, status: str, message: str, error: str = "", error_class: str = "") -> None:
        record = self._read_record(job_id)
        if record is None:
            return
        record.status = status
        record.message = message
        record.error = error[:2000]
        record.error_class = error_class
        record.updated_at = _now()
        if status == STATUS_DONE:
            record.progress = 1.0
        self._set_record(record)
        with self._transport("release job lease"):
            self._client.zrem(_lease_key(self._prefix, record.kind), job_id)

    def complete(self, job_id: str, *, message: str = "") -> None:
        self._finish(job_id, STATUS_DONE, message or "completed")

    def fail(self, job_id: str, *, error: str, error_class: str = "") -> None:
        self._finish(job_id, STATUS_ERROR, "failed", error, error_class)

    def get(self, job_id: str) -> JobRecord | None:
        try:
            return self._read_record(job_id)
        except Exception:  # noqa: BLE001 - polling must never break the UI
            return None

    def get_result(self, job_id: str) -> str | None:
        try:
            with self._transport("read result"):
                raw = self._client.get(_result_key(self._prefix, job_id))
        except Exception:  # noqa: BLE001
            return None
        return raw if isinstance(raw, str) else None

    def requeue(self, job_id: str, *, reason: str = "") -> bool:
        try:
            record = self._read_record(job_id)
            # Only a *running* job can be handed back; re-queueing an already
            # queued job would put it in the queue twice.
            if record is None or record.status != STATUS_RUNNING:
                return False
            record.status = STATUS_QUEUED
            record.attempts = max(0, record.attempts - 1)
            record.message = reason or "re-queued"
            record.updated_at = _now()
            self._set_record(record)
            with self._transport("requeue"):
                self._client.zrem(_lease_key(self._prefix, record.kind), job_id)
                self._client.lpush(_queue_key(self._prefix, record.kind), job_id)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "requeue failed job_id=%s error=%s", job_id, f"{type(exc).__name__}: {exc}"
            )
            return False

    def requeue_stale(self) -> int:
        cutoff = _now() - float(config.JOB_QUEUE_LEASE_SECONDS)
        requeued = 0
        for kind in KINDS:
            lease_key = _lease_key(self._prefix, kind)
            with self._transport("sweep stale jobs"):
                stale = self._client.zrangebyscore(lease_key, "-inf", f"({cutoff}")
            for raw_id in stale if isinstance(stale, list) else []:
                job_id = str(raw_id)
                record = self._read_record(job_id)
                with self._transport("clear expired lease"):
                    self._client.zrem(lease_key, job_id)
                if record is None or record.status != STATUS_RUNNING:
                    continue
                if record.attempts >= MAX_ATTEMPTS:
                    self._finish(
                        job_id,
                        STATUS_ERROR,
                        "failed",
                        f"abandoned after {record.attempts} attempts "
                        "(worker lease expired each time)",
                        "JobAbandoned",
                    )
                    continue
                record.status = STATUS_QUEUED
                record.message = "re-queued after worker lease expired"
                record.updated_at = _now()
                self._set_record(record)
                with self._transport("re-queue stale job"):
                    self._client.lpush(_queue_key(self._prefix, kind), job_id)
                requeued += 1
        return requeued

    def depth(self) -> int:
        try:
            return sum(
                int(self._client.llen(_queue_key(self._prefix, kind))) for kind in KINDS
            )
        except Exception:  # noqa: BLE001
            return 0

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:  # noqa: BLE001
            return False


# ------------------------------------------------------------ Upstash REST impl
class _UpstashRest:
    """Minimal Upstash REST client using the JSON command API.

    ``app.shared_cache``'s client speaks the legacy path-based API
    (``/get``, ``/set``), which cannot express list or sorted-set operations.
    This posts a JSON array (``["LPUSH", key, value]``) to the database root
    instead, which is the documented command form and needs no package.
    """

    def __init__(self, url: str, token: str, *, timeout_seconds: float) -> None:
        self._url = url.rstrip("/")
        self._token = token
        self._timeout = timeout_seconds

    def command(self, *args: str | int | float) -> Any:
        body = json.dumps(list(args)).encode("utf-8")
        auth = base64.b64encode(self._token.encode("utf-8")).decode("utf-8")
        request = urllib.request.Request(
            self._url,
            data=body,
            headers={
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                raw = response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise JobQueueError(f"upstash request failed: {type(exc).__name__}: {exc}") from exc
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise JobQueueError("upstash returned a non-JSON response") from exc
        if isinstance(data, dict):
            if data.get("error"):
                raise JobQueueError(str(data["error"]))
            return data.get("result")
        return data


class UpstashJobBackend(JobBackend):
    """Upstash Redis / Vercel KV backend over HTTP REST.

    Reuses the same credentials as the reference-data cache, so Pattern C can be
    adopted without deploying an in-cluster Redis. Claiming polls ``RPOP``
    because the REST API has no blocking pop.
    """

    name = "upstash_rest"
    is_distributed = True

    def __init__(
        self,
        url: str,
        token: str,
        *,
        prefix: str,
        ttl_seconds: int,
        timeout_seconds: float,
    ) -> None:
        self._prefix = prefix
        self._ttl = ttl_seconds
        self._rest = _UpstashRest(url, token, timeout_seconds=timeout_seconds)

    # -- helpers ------------------------------------------------------------
    def _set_record(self, record: JobRecord) -> None:
        self._rest.command(
            "SET", _meta_key(self._prefix, record.job_id), record.to_json(), "EX", self._ttl
        )

    def _read_record(self, job_id: str) -> JobRecord | None:
        raw = self._rest.command("GET", _meta_key(self._prefix, job_id))
        return JobRecord.from_json(raw if isinstance(raw, str) else None)

    # -- producer -----------------------------------------------------------
    def enqueue(self, kind: str, payload: str, *, request_id: str = "") -> JobRecord:
        if kind not in KINDS:
            raise JobQueueError(f"unknown job kind: {kind}")
        record = JobRecord(
            job_id=new_job_id(),
            kind=kind,
            request_id=request_id,
            created_at=_now(),
            updated_at=_now(),
        )
        try:
            self._set_record(record)
            self._rest.command(
                "SET", _payload_key(self._prefix, record.job_id), payload, "EX", self._ttl
            )
            self._rest.command("LPUSH", _queue_key(self._prefix, kind), record.job_id)
        except JobQueueError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise JobQueueError(f"enqueue failed: {type(exc).__name__}: {exc}") from exc
        return record

    # -- consumer -----------------------------------------------------------
    def claim(
        self, kinds: Sequence[str], *, worker_id: str
    ) -> tuple[JobRecord, str] | None:
        deadline = time.monotonic() + float(config.JOB_QUEUE_CLAIM_TIMEOUT_SECONDS)
        poll = max(0.25, float(config.JOB_QUEUE_POLL_SECONDS))
        while True:
            for kind in kinds:
                raw_id = self._rest.command("RPOP", _queue_key(self._prefix, kind))
                job_id = raw_id if isinstance(raw_id, str) else ""
                if not job_id:
                    continue
                record = self._read_record(job_id)
                payload_raw = self._rest.command("GET", _payload_key(self._prefix, job_id))
                if (
                    record is None
                    or not isinstance(payload_raw, str)
                    or record.status != STATUS_QUEUED
                ):
                    continue
                record.status = STATUS_RUNNING
                record.worker_id = worker_id
                record.attempts += 1
                record.heartbeat_at = _now()
                record.updated_at = record.heartbeat_at
                record.message = "claimed by worker"
                self._set_record(record)
                self._rest.command(
                    "ZADD",
                    _lease_key(self._prefix, kind),
                    record.heartbeat_at,
                    job_id,
                )
                return record, payload_raw
            if time.monotonic() >= deadline:
                return None
            time.sleep(poll)

    def set_progress(self, job_id: str, progress: float, message: str) -> None:
        record = self._read_record(job_id)
        if record is None:
            return
        record.progress = min(max(progress, 0.0), 1.0)
        record.message = message
        record.heartbeat_at = _now()
        record.updated_at = record.heartbeat_at
        self._set_record(record)
        self._rest.command(
            "ZADD", _lease_key(self._prefix, record.kind), record.heartbeat_at, job_id
        )

    def store_result(self, job_id: str, result: str) -> None:
        self._rest.command(
            "SET", _result_key(self._prefix, job_id), result, "EX", self._ttl
        )

    def _finish(
        self,
        job_id: str,
        status: str,
        message: str,
        error: str = "",
        error_class: str = "",
    ) -> None:
        record = self._read_record(job_id)
        if record is None:
            return
        record.status = status
        record.message = message
        record.error = error[:2000]
        record.error_class = error_class
        record.updated_at = _now()
        if status == STATUS_DONE:
            record.progress = 1.0
        self._set_record(record)
        self._rest.command("ZREM", _lease_key(self._prefix, record.kind), job_id)

    def complete(self, job_id: str, *, message: str = "") -> None:
        self._finish(job_id, STATUS_DONE, message or "completed")

    def fail(self, job_id: str, *, error: str, error_class: str = "") -> None:
        self._finish(job_id, STATUS_ERROR, "failed", error, error_class)

    def get(self, job_id: str) -> JobRecord | None:
        try:
            return self._read_record(job_id)
        except Exception:  # noqa: BLE001 - polling must never break the UI
            return None

    def get_result(self, job_id: str) -> str | None:
        try:
            raw = self._rest.command("GET", _result_key(self._prefix, job_id))
        except Exception:  # noqa: BLE001
            return None
        return raw if isinstance(raw, str) else None

    def requeue(self, job_id: str, *, reason: str = "") -> bool:
        try:
            record = self._read_record(job_id)
            if record is None or record.status != STATUS_RUNNING:
                return False
            record.status = STATUS_QUEUED
            record.attempts = max(0, record.attempts - 1)
            record.message = reason or "re-queued"
            record.updated_at = _now()
            self._set_record(record)
            self._rest.command("ZREM", _lease_key(self._prefix, record.kind), job_id)
            self._rest.command("LPUSH", _queue_key(self._prefix, record.kind), job_id)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "requeue failed job_id=%s error=%s", job_id, f"{type(exc).__name__}: {exc}"
            )
            return False

    def requeue_stale(self) -> int:
        cutoff = _now() - float(config.JOB_QUEUE_LEASE_SECONDS)
        requeued = 0
        for kind in KINDS:
            lease_key = _lease_key(self._prefix, kind)
            try:
                stale = self._rest.command("ZRANGEBYSCORE", lease_key, "-inf", f"({cutoff}")
            except JobQueueError:
                continue
            for raw_id in stale if isinstance(stale, list) else []:
                job_id = str(raw_id)
                self._rest.command("ZREM", lease_key, job_id)
                record = self._read_record(job_id)
                if record is None or record.status != STATUS_RUNNING:
                    continue
                if record.attempts >= MAX_ATTEMPTS:
                    self._finish(
                        job_id,
                        STATUS_ERROR,
                        "failed",
                        f"abandoned after {record.attempts} attempts "
                        "(worker lease expired each time)",
                        "JobAbandoned",
                    )
                    continue
                record.status = STATUS_QUEUED
                record.message = "re-queued after worker lease expired"
                record.updated_at = _now()
                self._set_record(record)
                self._rest.command("LPUSH", _queue_key(self._prefix, kind), job_id)
                requeued += 1
        return requeued

    def depth(self) -> int:
        total = 0
        for kind in KINDS:
            try:
                raw = self._rest.command("LLEN", _queue_key(self._prefix, kind))
                total += int(raw) if isinstance(raw, (int, float, str)) else 0
            except (JobQueueError, TypeError, ValueError):
                continue
        return total

    def ping(self) -> bool:
        try:
            self._rest.command("PING")
            return True
        except JobQueueError:
            return False


# -------------------------------------------------------------------- factory
_backend: JobBackend | None = None
_backend_lock = threading.Lock()


def build_job_backend() -> JobBackend:
    """Select a backend from configuration (see the module docstring)."""
    prefix = config.JOB_QUEUE_PREFIX
    ttl = config.JOB_QUEUE_TTL_SECONDS
    if not config.JOB_QUEUE_ENABLED:
        logger.info("job queue disabled; runs execute in-process (VA_LSE_JOB_QUEUE unset)")
        return InProcessJobBackend(prefix=prefix, ttl_seconds=ttl)
    if config.JOB_QUEUE_REDIS_URL:
        try:
            backend = RedisJobBackend(config.JOB_QUEUE_REDIS_URL, prefix=prefix, ttl_seconds=ttl)
        except JobQueueUnavailable as exc:
            logger.error("redis job backend unavailable: %s", exc)
        else:
            logger.info("job queue backend=redis url_configured=True")
            return backend
    if config.SHARED_CACHE_URL and config.SHARED_CACHE_TOKEN:
        logger.info("job queue backend=upstash_rest (shared cache credentials)")
        return UpstashJobBackend(
            config.SHARED_CACHE_URL,
            config.SHARED_CACHE_TOKEN,
            prefix=prefix,
            ttl_seconds=ttl,
            timeout_seconds=config.SHARED_CACHE_TIMEOUT_SECONDS,
        )
    logger.warning(
        "VA_LSE_JOB_QUEUE is enabled but no shared backend is configured "
        "(set VA_LSE_REDIS_URL or VA_LSE_SHARED_CACHE_URL/_TOKEN) — falling back to the "
        "in-process queue, which cannot serve a worker in another process"
    )
    return InProcessJobBackend(prefix=prefix, ttl_seconds=ttl)


def get_job_backend() -> JobBackend:
    """Return (and lazily create) the process-global job backend."""
    global _backend  # noqa: PLW0603
    if _backend is not None:
        return _backend
    with _backend_lock:
        if _backend is None:
            _backend = build_job_backend()
        return _backend


def reset_job_backend_for_tests() -> None:
    """Drop the cached backend so tests get a fresh one."""
    global _backend  # noqa: PLW0603
    with _backend_lock:
        _backend = None


def queue_is_distributed() -> bool:
    """True when jobs can be executed by a worker in another process."""
    try:
        return get_job_backend().is_distributed
    except Exception:  # noqa: BLE001 - never let this break a render
        return False
