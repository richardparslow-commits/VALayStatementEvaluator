"""Lightweight health sidecar for container orchestration.

Exposes ``GET /health`` (liveness — always 200 once the process is up),
``GET /ready`` (readiness — 200 only when the LLM endpoint is reachable and
the configured models are listed), and ``GET /metrics`` (Prometheus text
format, for alerting on trends rather than polling JSON). All three respond in
<2s, and none of them performs network I/O unless explicitly asked to with
``?probe=1`` — see :mod:`app.metrics` and
:meth:`app.job_queue.JobBackend.health` for why that constraint is load-bearing.

The server is stdlib-only (``http.server``) and runs on a daemon thread so it
never blocks Streamlit. Readiness is cached (TTL 30s) and the upstream
``GET {base_url}/models`` probe uses a 1.4s socket timeout so the handler can
always meet the 2s SLO even when the LLM gateway is slow.

Importing this module has no side effects; call :func:`start_health_server`
from the launcher (``run_app.py``) before Streamlit starts.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger("app.health")

# ------------------------------------------------------------------ constants
HEALTH_PATH = "/health"
READY_PATH = "/ready"
METRICS_PATH = "/metrics"

# Keep comfortably under the 2s SLO in the spec; leave headroom for JSON
# serialisation and TCP.
MODELS_PROBE_TIMEOUT_SECONDS = 1.4
READY_CACHE_TTL_SECONDS = 30.0

# Monotonic start time for uptime reporting.
_STARTED_MONO: float = time.monotonic()
_STARTED_WALL: float = time.time()

# Cached readiness: (checked_at_mono, ready, detail)
_ready_lock = threading.Lock()
_cached_ready_at: float = 0.0
_cached_ready: bool = False
_cached_detail: str = "not yet checked"

_server: ThreadingHTTPServer | None = None
_thread: threading.Thread | None = None


# --------------------------------------------------------------- probe helpers

def _health_port() -> int:
    raw = os.getenv("VA_LSE_HEALTH_PORT", "").strip()
    if raw:
        try:
            value = int(raw)
            if 1 <= value <= 65535:
                return value
        except ValueError:
            pass
    # Fallback: try config.HEALTH_PORT if config is importable, else 8001.
    try:
        from .config import HEALTH_PORT  # local import to avoid cycle at import time

        if isinstance(HEALTH_PORT, int) and 1 <= HEALTH_PORT <= 65535:
            return int(HEALTH_PORT)
    except Exception:  # noqa: BLE001
        pass
    return 8001


def _probe_endpoint_models(
    base_url: str,
    api_key: str,
    models: tuple[tuple[str, str], ...],
    *,
    timeout: float = MODELS_PROBE_TIMEOUT_SECONDS,
) -> tuple[bool, str]:
    """GET {base_url}/models and require every named model to be listed.

    Returns ``(ready, detail)``. Any network/auth/parse failure is "not ready"
    rather than an exception, and the detail never contains the key.
    """
    if not api_key:
        return False, "missing OPENAI_API_KEY"
    if not base_url:
        return False, "missing base_url"
    url = base_url.rstrip("/") + "/models"
    try:
        import urllib.error  # noqa: F401  # imported for side-effect type completeness
        import urllib.request

        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8")
        data: Any = json.loads(raw)
        rows = data.get("data", []) if isinstance(data, dict) else []
        ids: set[str] = set()
        for row in rows:
            if isinstance(row, dict):
                mid = row.get("id")
                if isinstance(mid, str) and mid.strip():
                    ids.add(mid.strip())
        # Every configured model must be listed for readiness.
        missing: list[str] = []
        for label, model in models:
            if model and model not in ids:
                missing.append(f"{label} `{model}` not listed at /models")
        if missing:
            return False, "; ".join(missing)
        return True, "ready"
    except Exception as exc:  # noqa: BLE001 - readiness is advisory, never raise
        return False, f"llm probe failed: {type(exc).__name__}: {exc}"


def _probe_llm_readiness() -> tuple[bool, str]:
    """Check the LLM endpoint(s) and configured models; always returns quickly.

    Ready means "this instance can serve a run", so a healthy *fallback* endpoint
    counts as ready: during an outage that has failed over cleanly the pod is
    serving users, and failing readiness would pull it out of the load balancer
    and page on-call for a primary problem that is already handled. The primary's
    outage is not hidden — it is in the detail here, in the ``llm_failover``
    block of ``/health``, and in ``va_lse_llm_failover_active``.

    The fallback is only probed when the primary fails, so the healthy path still
    costs one round trip (and stays inside the probe's 5s budget when it costs two).
    """
    # Import lazily so this module can be imported before config/logging setup.
    try:
        from .config import load_settings
    except Exception as exc:  # noqa: BLE001
        return False, f"config load failed: {exc}"

    try:
        settings = load_settings()
    except Exception as exc:  # noqa: BLE001
        return False, f"settings unavailable: {exc}"

    primary_ready, primary_detail = _probe_endpoint_models(
        (settings.base_url or "").strip(),
        (settings.api_key or "").strip(),
        (("Main model", settings.model_main), ("Fast model", settings.model_fast)),
    )
    if primary_ready:
        return True, "ready"
    if not settings.fallback_configured:
        return False, primary_detail

    fallback_ready, fallback_detail = _probe_endpoint_models(
        settings.fallback_base_url.strip(),
        settings.fallback_api_key_or_primary(),
        (
            ("Fallback main model", settings.fallback_model_main_or_primary()),
            ("Fallback fast model", settings.fallback_model_fast_or_primary()),
        ),
    )
    if fallback_ready:
        return True, f"primary endpoint unavailable ({primary_detail}); fallback endpoint ready"
    return False, f"primary: {primary_detail}; fallback: {fallback_detail}"


def _cached_readiness(*, force: bool = False) -> tuple[bool, str]:
    """Return cached readiness, refreshing if TTL expired or force=True."""
    global _cached_ready_at, _cached_ready, _cached_detail
    now = time.monotonic()
    with _ready_lock:
        if not force and (now - _cached_ready_at) < READY_CACHE_TTL_SECONDS and _cached_ready_at != 0.0:
            return _cached_ready, _cached_detail
    ready, detail = _probe_llm_readiness()
    with _ready_lock:
        _cached_ready_at = time.monotonic()
        _cached_ready = ready
        _cached_detail = detail
    return ready, detail


def cached_readiness_state() -> bool | None:
    """The last readiness verdict, or ``None`` if no check has been made yet.

    Deliberately never triggers a probe of its own: readiness costs a round trip to
    the LLM gateway, and ``/metrics`` is scraped every 15s and shares a port with
    the liveness probe, so it must not perform network I/O (see
    :mod:`app.metrics`). The orchestrator's readiness probe keeps this fresh; if
    nothing has probed yet there is no verdict to report, and ``None`` propagates
    to an absent series rather than an invented healthy one.
    """
    with _ready_lock:
        if _cached_ready_at == 0.0:
            return None
        return _cached_ready


def _health_payload(*, probe_cache: bool = False, probe_queue: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "ok",
        "service": "va-lay-statement-evaluator",
        "uptime_s": int(time.time() - _STARTED_WALL),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # Include shared cache status when available.
    try:
        from .shared_cache import get_cache as _gc
        cache = _gc()
        payload["cache"] = cache.health(probe=probe_cache)
    except Exception:  # noqa: BLE001
        payload["cache"] = {"backend": "unavailable"}
    # Include job-queue status: an operator needs to see, from the web pod, that
    # runs are being handed to a worker (and how deep that backlog is). The
    # backlog itself is only read for a local backend or when probe_queue is set,
    # because reading it on a remote tier is a network round trip on the liveness
    # path — see JobBackend.health.
    try:
        from .job_queue import get_job_backend as _gjb

        payload["job_queue"] = _gjb().health(probe=probe_queue)
    except Exception:  # noqa: BLE001
        payload["job_queue"] = {"backend": "unavailable"}
    # Include tracing status so "why are there no traces?" is answerable from the
    # same endpoint operators already poll. Reports configuration only — no probe
    # of the collector, because /health must not block on a network round trip.
    try:
        from .tracing import tracing_health as _th

        payload["tracing"] = _th()
    except Exception:  # noqa: BLE001
        payload["tracing"] = {"enabled": False, "active": False, "reason": "unavailable"}
    # Audit stream *and* its off-pod backup. Two separate blocks on purpose: the
    # audit log is the compliance artifact and the backup is what makes it survive
    # the pod, and either can be broken while the other looks fine.
    try:
        from .audit import audit_health as _ah

        payload["audit"] = _ah()
    except Exception:  # noqa: BLE001
        payload["audit"] = {"status": "unavailable"}
    try:
        from .audit_backup import audit_backup_health as _abh

        payload["audit_backup"] = _abh()
    except Exception:  # noqa: BLE001
        payload["audit_backup"] = {"status": "unavailable"}
    # Free space on the log volume. "No protection against disk-space exhaustion"
    # is answered by making it observable: below VA_LSE_DISK_MIN_FREE_BYTES this
    # reports below_floor, and audit.write_failures says whether it is already
    # costing us records. stdlib only, no I/O beyond a statvfs.
    try:
        from .audit_backup import disk_status as _ds

        payload["disk"] = _ds()
    except Exception:  # noqa: BLE001
        payload["disk"] = {"checked": False}
    # Whether a backup can be read back *with this configuration*. Config-only and
    # network-free on purpose: restoring is on demand, but "could we restore from
    # here?" is a question an operator should not have to shell into a pod to
    # answer, and it is the one thing that makes a green backup job meaningful.
    try:
        from .audit_restore import audit_integrity_health as _aih

        payload["restore"] = _aih()
    except Exception:  # noqa: BLE001
        payload["restore"] = {"restore_available": False, "error": "unavailable"}
    # Failover state. Network-free (config + the primary breaker), because /health
    # is polled every 10s by the kubelet and must never depend on a provider.
    try:
        from .llm import failover_status as _fs

        payload["llm_failover"] = _fs()
    except Exception:  # noqa: BLE001
        payload["llm_failover"] = {"configured": False, "active": False}
    return payload


def _ready_payload(ready: bool, detail: str) -> dict[str, Any]:
    base = _health_payload()
    base["ready"] = ready
    base["detail"] = detail
    # Explicit status field for load-balancer friendliness.
    base["status"] = "ready" if ready else "not_ready"
    return base


# ------------------------------------------------------------------ handler

class _HealthHandler(BaseHTTPRequestHandler):
    """Handles GET /health, GET /ready (and HEAD for probes that use it)."""

    def do_GET(self) -> None:  # noqa: N802
        self._handle(get_body=True)

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle(get_body=False)

    def _handle(self, *, get_body: bool) -> None:
        parsed = urlparse(self.path)
        path = (parsed.path or "/").rstrip("/") or "/"
        # ``?probe=1`` is the explicit opt-in to the one value that costs a
        # network round trip (queue depth on a remote backend). It is off by
        # default on every route so a scrape or a kubelet poll can never block on
        # a slow Redis tier.
        query = parse_qs(parsed.query)
        probe = (query.get("probe") or [""])[0].strip().lower() in ("1", "true", "yes")
        started = time.monotonic()
        try:
            if path == HEALTH_PATH:
                payload = _health_payload(probe_queue=probe)
                body = json.dumps(payload).encode("utf-8")
                self._send_json(200, body, get_body=get_body)
                logger.debug("health probe path=%s status=200", path)
                return
            if path == METRICS_PATH:
                from .metrics import CONTENT_TYPE as _metrics_ct
                from .metrics import render_prometheus as _render

                text = _render(_health_payload(probe_queue=probe))
                body = text.encode("utf-8")
                self._send(200, body, _metrics_ct, get_body=get_body)
                logger.debug("metrics scrape path=%s status=200 probe=%s", path, probe)
                return
            if path == READY_PATH:
                # During graceful shutdown the instance must fall out of the
                # load-balancer pool so no new work is routed to it.  Liveness
                # (/health) stays 200 — the process is still alive and draining.
                draining = False
                try:
                    from .shutdown import is_shutting_down, inflight_count

                    draining = is_shutting_down()
                except Exception:  # noqa: BLE001
                    pass
                if draining:
                    detail = f"draining — {inflight_count()} inflight run(s) finishing before SIGKILL"
                    payload = _ready_payload(False, detail)
                    body = json.dumps(payload).encode("utf-8")
                    self._send_json(503, body, get_body=get_body)
                    logger.warning(
                        "readiness probe draining inflight=%s status=503",
                        inflight_count(),
                        extra={"phase": "health", "status": "draining", "duration_ms": int((time.monotonic() - started) * 1000)},
                    )
                    return
                # Serve cached readiness but refresh on background if stale?
                # For correctness we refresh synchronously — still <2s because
                # the probe itself is bounded to 1.4s.
                ready, detail = _cached_readiness()
                payload = _ready_payload(ready, detail)
                body = json.dumps(payload).encode("utf-8")
                status = 200 if ready else 503
                self._send_json(status, body, get_body=get_body)
                elapsed_ms = int((time.monotonic() - started) * 1000)
                logger.info(
                    "readiness probe ready=%s status=%d duration_ms=%d detail=%s",
                    ready,
                    status,
                    elapsed_ms,
                    detail[:200],
                    extra={"phase": "health", "status": "ready" if ready else "not_ready", "duration_ms": elapsed_ms},
                )
                return
            # Unknown path
            body = json.dumps({"status": "not_found", "path": path}).encode("utf-8")
            self._send_json(404, body, get_body=get_body)
        except Exception as exc:  # noqa: BLE001 - handler must never crash
            logger.warning("health handler error path=%s error=%s", path, exc)
            try:
                body = json.dumps({"status": "error", "error": str(exc)[:300]}).encode("utf-8")
                self._send_json(500, body, get_body=get_body)
            except Exception:  # noqa: BLE001
                pass

    def _send_json(self, status: int, body: bytes, *, get_body: bool) -> None:
        self._send(status, body, "application/json; charset=utf-8", get_body=get_body)

    def _send(self, status: int, body: bytes, content_type: str, *, get_body: bool) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if get_body:
            self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Route through the app logger at debug level instead of stderr.
        logger.debug("health http: " + format, *args)


# --------------------------------------------------------------- lifecycle

def start_health_server(port: int | None = None, *, host: str = "0.0.0.0") -> ThreadingHTTPServer | None:
    """Start the health sidecar (idempotent). Returns the server or None on failure.

    Binds to ``host:port``. If the port is already in use the function logs a
    warning and returns None — the Streamlit app still starts (health is
    best-effort).
    """
    global _server, _thread
    if _server is not None:
        return _server

    chosen_port = int(port) if port is not None else _health_port()

    try:
        server = ThreadingHTTPServer((host, chosen_port), _HealthHandler)
        # Allow quick restart in tests / container restarts.
        server.daemon_threads = True
    except OSError as exc:
        logger.warning("health server could not bind %s:%d: %s", host, chosen_port, exc)
        return None

    thread = threading.Thread(
        target=server.serve_forever,
        name="va-lse-health",
        daemon=True,
    )
    thread.start()
    _server = server
    _thread = thread
    logger.info(
        "health server listening on %s:%d (GET /health, GET /ready, GET /metrics)",
        host,
        chosen_port,
    )
    return server


def stop_health_server() -> None:
    """Stop the health server if running (used in tests)."""
    global _server, _thread, _cached_ready_at
    srv = _server
    _server = None
    _thread = None
    _cached_ready_at = 0.0
    if srv is not None:
        try:
            srv.shutdown()
            srv.server_close()
        except Exception:  # noqa: BLE001
            pass
