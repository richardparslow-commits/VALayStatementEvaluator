"""Application settings.

Settings are loaded from environment variables (optionally via a .env file), falling
back to Streamlit secrets (``st.secrets``) for hosted deployments, and can be overridden
at runtime from the Streamlit sidebar.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"

# Load .env from the project root if present (never committed). override=True so
# the project's .env wins over any pre-existing environment values.
load_dotenv(PROJECT_ROOT / ".env", override=True)

# ---------------------------------------------------------------------------
# Secret resolution order.  A secret can arrive three ways, checked in order:
#
#   1. the process environment (orchestrator var, K8s/Docker secret, CI var),
#   2. the project .env (local development; git-ignored, so it never ships),
#   3. Streamlit secrets (``st.secrets`` → ``.streamlit/secrets.toml``).
#
# Streamlit Community Cloud is the case that matters: there is no .env in the
# deployed image, so the API key and endpoint can only come from the secrets
# manager.  Environment wins over secrets so a local .env always overrides a
# hosted secret store.  See DEPLOYMENT.md → Pattern D.
# ---------------------------------------------------------------------------
SECRET_FIELD_LABELS: dict[str, str] = {
    "OPENAI_API_KEY": "API key",
    "OPENAI_BASE_URL": "base URL",
    "LLM_MODEL_MAIN": "main model",
    "LLM_MODEL_FAST": "fast model",
    "FETCH_SANDBOX_API_KEY": "Fetch API key",
    "FETCH_SANDBOX_BASE_URL": "Fetch base URL",
    "OPENAI_BASE_URL_FALLBACK": "fallback base URL",
    "OPENAI_API_KEY_FALLBACK": "fallback API key",
    "LLM_MODEL_MAIN_FALLBACK": "fallback main model",
    "LLM_MODEL_FAST_FALLBACK": "fallback fast model",
    "FETCH_SANDBOX_RECORDS_PATH": "Fetch records path",
    "VA_LSE_SHARED_CACHE_URL": "shared cache URL",
    "VA_LSE_SHARED_CACHE_TOKEN": "shared cache token",
}

_SECRETS_CACHE: dict[str, Any] | None = None


def _read_streamlit_secrets() -> dict[str, Any]:
    """Return ``st.secrets`` as a flat string-keyed dict (empty when absent)."""
    try:
        import streamlit as st

        data = st.secrets.to_dict()
    except Exception:  # noqa: BLE001 - no secrets.toml is the normal local case
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): value for key, value in data.items()}


def streamlit_secrets() -> dict[str, Any]:
    """Read the Streamlit secrets manager once per process ({} when unset).

    Read explicitly rather than relying on Streamlit's implicit promotion of
    string secrets into ``os.environ``, which only happens the first time
    *anything* touches ``st.secrets`` — too fragile an ordering to depend on for
    the API key.  Streamlit is imported lazily and a missing secrets file is
    swallowed: local runs and the test suite simply have no secrets.
    """
    global _SECRETS_CACHE
    if _SECRETS_CACHE is None:
        _SECRETS_CACHE = _read_streamlit_secrets()
    return _SECRETS_CACHE


def _secret_value(name: str) -> str:
    """Return the Streamlit secrets value for ``name`` ("" when unset/non-string).

    Both the exact env-var name and its lowercase form are accepted, since
    ``secrets.toml`` is hand-written and either spelling is natural there.
    """
    secrets = streamlit_secrets()
    for key in (name, name.lower()):
        value = secrets.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _setting(
    name: str, default: str = "", from_secrets: set[str] | None = None
) -> str:
    """Resolve one string setting: environment → Streamlit secrets → default.

    Names recorded in ``from_secrets`` came from the secrets manager, so the
    sidebar can show where a pre-filled value came from instead of leaving the
    user to guess whether it is theirs, stale, or a code default.

    Streamlit promotes string secrets into ``os.environ`` the first time
    ``st.secrets`` is read, after which the two sources are indistinguishable by
    value alone.  A name whose environment value equals its secret value is
    therefore attributed to the secret: in a hosted deployment that env entry
    only exists because Streamlit put it there, and a local ``.env`` carrying the
    identical value is the same credential either way.
    """
    secret = _secret_value(name)
    env = os.getenv(name, "").strip()
    if env and env != secret:
        return env
    if secret:
        if from_secrets is not None:
            from_secrets.add(name)
        return secret
    return env or default


# Default endpoint tuned for the QwenCloud Individual Plan Lite subscription
# but the app speaks to ANY OpenAI-compatible POST {base_url}/chat/completions
# endpoint (see COMPATIBILITY.md / MIGRATION.md). Token Plan uses a dedicated
# sk-sp- API key that MUST be paired with this base URL (they do not work
# against the general MaaS gateway).
DEFAULT_BASE_URL = (
    "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
)
# Low-volume, high-value calls (claim extraction, verification, rubric scoring,
# topic audit, rewrite) use the strong reasoning model. Override via
# LLM_MODEL_MAIN for another provider (e.g. gpt-4-turbo — see COMPATIBILITY.md).
DEFAULT_MODEL_MAIN = "qwen3.7-max"
# The bulk digest/merge passes (one call per record chunk — by far the most
# calls) use the cheap model to preserve the Lite plan's limited credit quota.
# Override via LLM_MODEL_FAST (e.g. gpt-4o-mini for OpenAI).
DEFAULT_MODEL_FAST = "qwen3.7-flash"
DEFAULT_FETCH_SANDBOX_BASE_URL = "https://fetchsandbox.com"
DEFAULT_FETCH_SANDBOX_RECORDS_PATH = "/medical_records/{patient_id}"
DEFAULT_FETCH_SANDBOX_MAX_RESPONSE_BYTES = 100 * 1024 * 1024

# -------------------------------------------------------------- fallback endpoint
#
# OPTIONAL second LLM endpoint used when the primary has been unhealthy for
# FAILOVER_AFTER_SECONDS. Unset (the default) means the app behaves exactly as it
# did before this feature existed: one endpoint, no failover, no extra probe.
#
# A fallback is usually a *different provider* (the documented example is OpenAI
# as backup for QwenCloud), which in practice needs its own API key and its own
# model names — a key for one provider authenticates nothing at another. So all
# three are configurable, and each defaults to the primary's value:
#
#   OPENAI_BASE_URL_FALLBACK     the endpoint (setting this is what arms failover)
#   OPENAI_API_KEY_FALLBACK      its key        (default: the primary key)
#   LLM_MODEL_MAIN_FALLBACK      its heavy model (default: the primary's)
#   LLM_MODEL_FAST_FALLBACK      its bulk model  (default: the primary's)
#
# Configuring only OPENAI_BASE_URL_FALLBACK is supported and useful for a second
# gateway under the same account; it just cannot reach a different provider.
DEFAULT_FAILOVER_AFTER_SECONDS = 300

# Names for the two endpoints. Single definition on purpose: these strings are
# simultaneously the log/audit value, the metrics label, and the JSON field in a
# queued job's usage record, so a second spelling anywhere would split one
# endpoint into two identities.
PRIMARY_ENDPOINT = "primary"
FALLBACK_ENDPOINT = "fallback"


@dataclass
class Settings:
    """Runtime LLM settings (mutable; sidebar edits update the instance)."""

    api_key: str
    base_url: str
    model_main: str
    model_fast: str
    fetch_api_key: str
    fetch_base_url: str
    fetch_records_path: str
    # Optional failover endpoint. Empty base URL = failover disabled. Placed after
    # the required fields so the dataclass keeps its non-default-first ordering.
    fallback_base_url: str = ""
    fallback_api_key: str = ""
    fallback_model_main: str = ""
    fallback_model_fast: str = ""
    fetch_max_response_bytes: int = DEFAULT_FETCH_SANDBOX_MAX_RESPONSE_BYTES
    # Env-var names that were satisfied by Streamlit secrets rather than the
    # environment/.env — surfaced in the sidebar so a pre-filled value is not
    # mistaken for one the user typed.  See SECRET_FIELD_LABELS.
    from_secrets: frozenset[str] = frozenset()

    @property
    def configured(self) -> bool:
        return bool(self.api_key.strip())

    @property
    def fallback_configured(self) -> bool:
        """Whether a second endpoint is armed.

        Keyed on the base URL only: that is the one value that must be distinct
        for a second endpoint to exist at all. A key/model left unset inherits the
        primary's, so a same-account second gateway needs one variable, not four.
        """
        return bool(self.fallback_base_url.strip())

    def fallback_api_key_or_primary(self) -> str:
        return (self.fallback_api_key or self.api_key).strip()

    def fallback_model_main_or_primary(self) -> str:
        return (self.fallback_model_main or self.model_main).strip()

    def fallback_model_fast_or_primary(self) -> str:
        return (self.fallback_model_fast or self.model_fast).strip()

    @property
    def fetch_configured(self) -> bool:
        return bool(self.fetch_base_url.strip() and self.fetch_records_path.strip())


def load_settings() -> Settings:
    """Build settings from the environment, then Streamlit secrets, then defaults."""
    from_secrets: set[str] = set()
    return Settings(
        api_key=_setting("OPENAI_API_KEY", "", from_secrets),
        base_url=_setting("OPENAI_BASE_URL", DEFAULT_BASE_URL, from_secrets),
        model_main=_setting("LLM_MODEL_MAIN", DEFAULT_MODEL_MAIN, from_secrets),
        model_fast=_setting("LLM_MODEL_FAST", DEFAULT_MODEL_FAST, from_secrets),
        fetch_api_key=_setting("FETCH_SANDBOX_API_KEY", "", from_secrets),
        fetch_base_url=_setting(
            "FETCH_SANDBOX_BASE_URL", DEFAULT_FETCH_SANDBOX_BASE_URL, from_secrets
        ),
        fetch_records_path=_setting(
            "FETCH_SANDBOX_RECORDS_PATH", DEFAULT_FETCH_SANDBOX_RECORDS_PATH, from_secrets
        ),
        # Failover endpoint (optional). An empty base URL leaves failover disabled.
        fallback_base_url=_setting("OPENAI_BASE_URL_FALLBACK", "", from_secrets),
        fallback_api_key=_setting("OPENAI_API_KEY_FALLBACK", "", from_secrets),
        fallback_model_main=_setting("LLM_MODEL_MAIN_FALLBACK", "", from_secrets),
        fallback_model_fast=_setting("LLM_MODEL_FAST_FALLBACK", "", from_secrets),
        fetch_max_response_bytes=_positive_int_env(
            "FETCH_SANDBOX_MAX_RESPONSE_BYTES",
            DEFAULT_FETCH_SANDBOX_MAX_RESPONSE_BYTES,
        ),
        from_secrets=frozenset(from_secrets),
    )


def load_knowledge(name: str) -> str:
    """Read a knowledge-base markdown file by file name."""
    path = KNOWLEDGE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Knowledge file missing: {path}")
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Large-record-set handling. All overridable via environment variables so the
# app can be tuned per machine/API rate limits without code changes.
# ---------------------------------------------------------------------------
def _float_env(name: str, default: float | None) -> float | None:
    """Read an optional float env var; returns default when unset/invalid."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _positive_int_env(name: str, default: int) -> int:
    value = _int_env(name, default)
    return value if value > 0 else default


def _non_negative_int_env(name: str, default: int) -> int:
    """Like ``_positive_int_env``, but ``0`` is a valid setting. Never negative.

    For durations where 0 means "no wait" — using the positive variant there would
    turn a deliberate 0 into the default and silently reinstate the wait.
    """
    value = _int_env(name, default)
    return value if value >= 0 else default


def _flag_env(name: str, default: bool = False) -> bool:
    """Read a boolean env var (``1``/``true``/``yes``/``on``).

    Matches the truthiness rule already used inline for ``VA_LSE_TRACING`` and
    ``VA_LSE_JOB_QUEUE``; new flags should use this helper instead of repeating
    the tuple.
    """
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


# Sidecar health server (liveness + readiness + metrics) for container
# orchestration. Exposes GET /health (liveness), GET /ready (readiness) and
# GET /metrics (Prometheus) on this port on 0.0.0.0. Override with
# VA_LSE_HEALTH_PORT; disabled when set to 0.
HEALTH_PORT = _int_env("VA_LSE_HEALTH_PORT", 8001)

# Window for va_lse_session_count (app/metrics.py). Streamlit re-runs the script
# on every interaction, so a session is counted while it keeps interacting; a tab
# left open and untouched drops out after this many seconds. 5 minutes tracks
# "people using it right now" without flapping on a user who is reading a report;
# raise it if your users routinely pause for longer between clicks.
METRICS_SESSION_TTL_SECONDS = _positive_int_env("VA_LSE_METRICS_SESSION_TTL_SECONDS", 300)

# Hard cap on total pages across all uploaded record files.
MAX_RECORD_PAGES = _int_env("VA_LSE_MAX_RECORD_PAGES", 5000)

# Number of record chunks digested in parallel. Tuned down for the QwenCloud
# Individual Plan Lite, which allows ~1-2 concurrent agents; higher values just
# trigger rate limiting and burn the small credit quota faster. Raise via
# VA_LSE_RECORDS_CONCURRENCY if you move to a higher QwenCloud tier.
RECORDS_CONCURRENCY = max(1, _int_env("VA_LSE_RECORDS_CONCURRENCY", 2))

# Maximum facts kept in the digest after consolidation.
MAX_DIGEST_FACTS = _int_env("VA_LSE_MAX_DIGEST_FACTS", 1500)

# Characters per record chunk. Smaller chunks => more LLM calls but better
# recall on dense pages (nothing gets truncated mid-extraction).
DIGEST_CHUNK_CHARS = _int_env("VA_LSE_DIGEST_CHUNK_CHARS", 8000)

# DOCX unzip hardening: reject oversized internal members before decompression.
DOCX_MAX_INTERNAL_FILE_BYTES = _positive_int_env(
    "VA_LSE_DOCX_MAX_INTERNAL_FILE_BYTES",
    50 * 1024 * 1024,
)
DOCX_MAX_TOTAL_UNCOMPRESSED_BYTES = _positive_int_env(
    "VA_LSE_DOCX_MAX_TOTAL_UNCOMPRESSED_BYTES",
    200 * 1024 * 1024,
)
DOCX_MAX_INTERNAL_FILE_COUNT = _positive_int_env(
    "VA_LSE_DOCX_MAX_INTERNAL_FILE_COUNT",
    10_000,
)

# ---------------------------------------------------------------------------
# Upload size hardening — enforced both at the Streamlit server layer
# (.streamlit/config.toml maxUploadSize) and in Python (app/main.py) so the
# limit is visible even without the server config.
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = _positive_int_env("VA_LSE_MAX_UPLOAD_BYTES", 50 * 1024 * 1024)
MAX_TOTAL_UPLOAD_BYTES = _positive_int_env(
    "VA_LSE_MAX_TOTAL_UPLOAD_BYTES",
    200 * 1024 * 1024,
)

# ---------------------------------------------------------------------------
# Optional credit-burn gauge for the usage estimator. QwenCloud Token Plan does
# not publish a fixed credits-per-1M-token rate (and it changes), so we default
# to "unbounded": the estimator reports calls + estimated tokens per phase but
# only shows a credit figure once you set these to your plan's effective rates.
# Example: if the fast model burns ~800 credits per 1M tokens on your plan, set
# VA_LSE_CREDITS_PER_1M_FAST=800. The contributions are summed and compared to
# the quota shown in the UI.
CREDITS_PER_1M_MAIN = _float_env("VA_LSE_CREDITS_PER_1M_MAIN", None)
CREDITS_PER_1M_FAST = _float_env("VA_LSE_CREDITS_PER_1M_FAST", None)

# Weekly credit quota on the QwenCloud Individual Plan Lite (informational; used
# only for the usage gauge, never limits the run).
CREDIT_QUOTA = _float_env("VA_LSE_CREDIT_QUOTA", 2500.0)

# ---------------------------------------------------------------------------
# Circuit breaker + concurrency limiter for LLM calls (100-user protection).
# The breaker opens after N consecutive logical failures (a call that exhausts
# its retries counts as one) and stays open for RECOVERY_SECONDS before
# allowing a probe. The limiter caps simultaneous LLM calls and queues the
# rest up to MAX_DEPTH (see app/circuit_breaker.py and app/llm.py).
# ---------------------------------------------------------------------------
LLM_CB_FAILURE_THRESHOLD = _positive_int_env("VA_LSE_CB_FAILURE_THRESHOLD", 3)
LLM_CB_RECOVERY_SECONDS = _positive_int_env("VA_LSE_CB_RECOVERY_SECONDS", 60)
LLM_MAX_CONCURRENT = _positive_int_env("VA_LSE_MAX_CONCURRENT_LLM_CALLS", 20)
LLM_QUEUE_MAX_DEPTH = _positive_int_env("VA_LSE_LLM_QUEUE_MAX_DEPTH", 50)
LLM_QUEUE_TIMEOUT_SECONDS = _positive_int_env("VA_LSE_LLM_QUEUE_TIMEOUT_SECONDS", 30)

# ---------------------------------------------------------------------------
# Failover to a second LLM endpoint (optional — see app/llm.py).
#
# When the *primary* endpoint has been continuously unhealthy for this many
# seconds, calls are routed to OPENAI_BASE_URL_FALLBACK instead, and routed back
# as soon as the primary recovers. Still one endpoint unless a fallback base URL
# is configured, so this value does nothing on its own.
#
# Despite the name (kept for the documented contract) this is NOT an HTTP
# request timeout — the per-call HTTP timeout is LLM_CALL_TIMEOUT_SECONDS. It is
# the grace period before failover engages, so an operator can ride out a blip
# without switching providers. 5 minutes means "the primary has been failing long
# enough that waiting is worse than switching"; lower it to fail over sooner.
# ---------------------------------------------------------------------------
LLM_FAILOVER_AFTER_SECONDS = _non_negative_int_env(
    "LLM_ENDPOINT_FALLBACK_TIMEOUT_SECONDS", DEFAULT_FAILOVER_AFTER_SECONDS
)

# ---------------------------------------------------------------------------
# Audit logging — dedicated JSON stream for compliance/forensics (see app/audit.py).
# Separate from diagnostic app.log so it can be queried and retained under a
# different policy. Disabled only by setting dir to empty; otherwise writes to
# {AUDIT_LOG_DIR}/{AUDIT_LOG_FILE} with rotation.
# ---------------------------------------------------------------------------
AUDIT_LOG_DIR = os.getenv("VA_LSE_AUDIT_LOG_DIR", "").strip() or os.getenv("VA_LSE_LOG_DIR", "").strip() or "logs"
AUDIT_LOG_FILE = os.getenv("VA_LSE_AUDIT_LOG_FILE", "").strip() or "audit.log"
AUDIT_LOG_MAX_BYTES = _positive_int_env("VA_LSE_AUDIT_LOG_MAX_BYTES", 10 * 1024 * 1024)
AUDIT_LOG_BACKUPS = _positive_int_env("VA_LSE_AUDIT_LOG_BACKUPS", 10)

# ``error_message`` in an audit entry is arbitrary upstream exception text — it is
# NOT covered by the module's "counts and classifications only" contract, because
# a library can put anything in an exception message. The audit log's own
# docstring promised no free-text; this flag makes that promise enforceable for
# deployments that ship audit logs to third-party storage (see DEPLOYMENT.md →
# Audit log backup). ``error_class`` is always recorded and is always safe.
AUDIT_ERROR_MESSAGES = _flag_env("VA_LSE_AUDIT_ERROR_MESSAGES", True)

# ---------------------------------------------------------------------------
# Audit log retention + backup (see app/audit_backup.py, scripts/backup_audit_logs.py).
#
# Rotation bounds the *size* of the audit stream (AUDIT_LOG_MAX_BYTES x
# (AUDIT_LOG_BACKUPS + 1), ~110 MiB by default) but knows nothing about time, so
# it can and will delete a file that is younger than AUDIT_RETENTION_DAYS during
# a busy period. Local retention therefore also enforces an age ceiling, and the
# effective rule is whichever bites first. Backing rotated files up *before* that
# happens is the job of the backup pass.
#
# Backup is opt-in and off by default. AUDIT_BACKUP_DESTINATION must name a store
# that is genuinely off the pod. A filesystem destination that resolves to the
# audit log's own volume (or inside the log dir) is *not* refused — that would
# break local testing and Compose — but it is reported as off_pod: false and
# same_volume: true in /health, because it cannot survive the pod failure the
# backup exists to survive.
# ---------------------------------------------------------------------------
AUDIT_RETENTION_DAYS = _positive_int_env("VA_LSE_AUDIT_RETENTION_DAYS", 7)
AUDIT_BACKUP_DESTINATION = os.getenv("VA_LSE_AUDIT_BACKUP_DESTINATION", "").strip().lower()
AUDIT_BACKUP_DIR = os.getenv("VA_LSE_AUDIT_BACKUP_DIR", "").strip()
AUDIT_BACKUP_INTERVAL_HOURS = _float_env("VA_LSE_AUDIT_BACKUP_INTERVAL_HOURS", 6.0) or 6.0
AUDIT_BACKUP_CLOUD_RETENTION_DAYS = _positive_int_env("VA_LSE_AUDIT_BACKUP_CLOUD_RETENTION_DAYS", 90)
AUDIT_BACKUP_STATE_FILE = os.getenv("VA_LSE_AUDIT_BACKUP_STATE_FILE", "").strip() or ".audit-backup-state.json"
AUDIT_BACKUP_S3_BUCKET = os.getenv("VA_LSE_AUDIT_BACKUP_S3_BUCKET", "").strip()
AUDIT_BACKUP_S3_PREFIX = os.getenv("VA_LSE_AUDIT_BACKUP_S3_PREFIX", "").strip()
AUDIT_BACKUP_S3_ENDPOINT_URL = os.getenv("VA_LSE_AUDIT_BACKUP_S3_ENDPOINT_URL", "").strip()
AUDIT_BACKUP_GCS_BUCKET = os.getenv("VA_LSE_AUDIT_BACKUP_GCS_BUCKET", "").strip()
AUDIT_BACKUP_GCS_PREFIX = os.getenv("VA_LSE_AUDIT_BACKUP_GCS_PREFIX", "").strip()
AUDIT_BACKUP_AZURE_CONTAINER = os.getenv("VA_LSE_AUDIT_BACKUP_AZURE_CONTAINER", "").strip()
AUDIT_BACKUP_AZURE_PREFIX = os.getenv("VA_LSE_AUDIT_BACKUP_AZURE_PREFIX", "").strip()
AUDIT_BACKUP_AZURE_ACCOUNT_URL = os.getenv("VA_LSE_AUDIT_BACKUP_AZURE_ACCOUNT_URL", "").strip()
# Passed by the shipped CronJob so a deployed-but-unconfigured backup job fails
# visibly (``kubectl get jobs``) instead of exiting 0 forever.
AUDIT_BACKUP_REQUIRED = _flag_env("VA_LSE_AUDIT_BACKUP_REQUIRED", False)
# A pass that overruns the CronJob's activeDeadlineSeconds is SIGKILLed without
# running cleanup, so its lock file survives. Without takeover that one hung upload
# would disable every subsequent backup. Keep this above the job's
# activeDeadlineSeconds (900 s in the shipped manifest) with room to spare.
AUDIT_BACKUP_LOCK_STALE_SECONDS = _positive_int_env("VA_LSE_AUDIT_BACKUP_LOCK_STALE_SECONDS", 3600)

# ---------------------------------------------------------------------------
# Diagnostic run log (logs/runs.jsonl, see app/run_log.py).
#
# This stream was unbounded — plain append with no rotation — so it, not
# audit.log, was the file that could fill the volume. Rotation is by size for the
# same reason the audit log's is: the workload is bursty and a time-based bound
# cannot be enforced without a cron.
# ---------------------------------------------------------------------------
RUN_LOG_MAX_BYTES = _positive_int_env("VA_LSE_RUN_LOG_MAX_BYTES", 10 * 1024 * 1024)
RUN_LOG_BACKUPS = _positive_int_env("VA_LSE_RUN_LOG_BACKUPS", 5)

# Free-space floor for the log volume. Below this, ``/health`` reports the log
# storage as degraded so disk exhaustion is observable before writes start
# failing. Purely a signal — audit writes are never dropped to save space.
DISK_MIN_FREE_BYTES = _positive_int_env("VA_LSE_DISK_MIN_FREE_BYTES", 256 * 1024 * 1024)

# ---------------------------------------------------------------------------
# Graceful shutdown + per-LLM-call timeout (see app/shutdown.py + app/llm.py).
# On SIGTERM the app stops accepting new Evaluate/Draft runs, keeps
# /health 200 but flips /ready to 503 so the orchestrator drains traffic,
# and waits up to SHUTDOWN_GRACE_SECONDS for inflight runs to finish before
# the orchestrator's SIGKILL arrives. Each LLM call is bounded by
# LLM_CALL_TIMEOUT_SECONDS (5 min default) so a single hung call cannot block
# the drain forever; a timeout surfaces as a user-visible LLMError.
# ---------------------------------------------------------------------------
SHUTDOWN_GRACE_SECONDS = _positive_int_env("VA_LSE_SHUTDOWN_GRACE_SECONDS", 30)
LLM_CALL_TIMEOUT_SECONDS = _positive_int_env("VA_LSE_LLM_CALL_TIMEOUT_SECONDS", 300)

# ---------------------------------------------------------------------------
# Pipeline-level timeout + memory monitoring (see app/pipeline_guard.py).
# PIPELINE_TIMEOUT_SECONDS caps the total wall-clock time for an entire
# Evaluate or Draft run (including record digest, merge, and all LLM calls).
# If exceeded, the run is aborted with a user-visible error. MEMORY_WARN_MB
# is the RSS threshold (in MB) above which a warning is logged; below 200 MB
# the run is aborted with MemoryError. Both are advisory on platforms where
# RSS is unavailable.
# ---------------------------------------------------------------------------
PIPELINE_TIMEOUT_SECONDS = _positive_int_env("VA_LSE_PIPELINE_TIMEOUT_SECONDS", 1800)
MEMORY_WARN_MB = _positive_int_env("VA_LSE_MEMORY_WARN_MB", 500)

# ---------------------------------------------------------------------------
# Distributed cache for VA reference data (see app/shared_cache.py).
# When SHARED_CACHE_URL + SHARED_CACHE_TOKEN are set the app uses Upstash Redis
# (or any Upstash-compatible HTTP REST endpoint, including Vercel KV) as a
# shared cache across all Streamlit instances.  A local LRU (256 entries)
# always runs as a fallback so single-instance deployments need no Redis.
# ---------------------------------------------------------------------------
SHARED_CACHE_URL = _setting("VA_LSE_SHARED_CACHE_URL")
SHARED_CACHE_TOKEN = _setting("VA_LSE_SHARED_CACHE_TOKEN")
SHARED_CACHE_TIMEOUT_SECONDS = float(os.getenv("VA_LSE_SHARED_CACHE_TIMEOUT_SECONDS", "2"))
SHARED_CACHE_LOCAL_MAXSIZE = _positive_int_env("VA_LSE_SHARED_CACHE_LOCAL_MAXSIZE", 256)

# ---------------------------------------------------------------------------
# Distributed job queue + worker pool (see app/job_queue.py + app/worker.py).
#
# Moves the heavy Evaluate/Draft pipeline OFF the Streamlit request path onto
# dedicated worker processes, so a 500-page record set no longer pins the pod
# that happens to own the browser session's WebSocket.  This is what actually
# removes the Pattern A/B hot spot: session affinity is still required for the
# WebSocket itself, but the digest work is no longer placed by it.
#
# Opt-in.  With JOB_QUEUE_ENABLED off the app runs the pipeline in-process
# exactly as before — the right default for single-instance and dev use.
#
# Transport is auto-selected by what is configured, in order:
#   1. JOB_QUEUE_REDIS_URL          → redis-py, in-cluster Redis (Pattern C)
#   2. SHARED_CACHE_URL + _TOKEN    → Upstash REST tier (no extra dependency)
#   3. neither                      → in-process backend (single process only)
# See app/job_queue.py:build_job_backend for the resolution order.
# ---------------------------------------------------------------------------
JOB_QUEUE_ENABLED = os.getenv("VA_LSE_JOB_QUEUE", "").strip() in ("1", "true", "True", "yes")
JOB_QUEUE_REDIS_URL = os.getenv("VA_LSE_REDIS_URL", "").strip()
JOB_QUEUE_PREFIX = os.getenv("VA_LSE_JOB_QUEUE_PREFIX", "va_lse").strip() or "va_lse"
# Completed jobs (meta + payload + result) expire after this long. 24 h default:
# long enough for a user to come back to a finished report, short enough that
# abandoned record bundles do not sit in Redis indefinitely.
JOB_QUEUE_TTL_SECONDS = _positive_int_env("VA_LSE_JOB_QUEUE_TTL_SECONDS", 24 * 3600)
# A worker refreshes its lease on every progress update; a job whose lease has
# expired is considered abandoned and is re-queued for another worker (this is
# how a SIGKILLed worker's job survives without an orchestrator-specific
# reaper). Must exceed the longest gap between progress updates.
JOB_QUEUE_LEASE_SECONDS = _positive_int_env("VA_LSE_JOB_QUEUE_LEASE_SECONDS", 900)
JOB_QUEUE_CLAIM_TIMEOUT_SECONDS = _positive_int_env("VA_LSE_JOB_QUEUE_CLAIM_TIMEOUT_SECONDS", 5)
JOB_QUEUE_POLL_SECONDS = float(os.getenv("VA_LSE_JOB_QUEUE_POLL_SECONDS", "1.0"))
JOB_QUEUE_UI_POLL_SECONDS = float(os.getenv("VA_LSE_JOB_QUEUE_UI_POLL_SECONDS", "2.5"))
# Ceiling on the serialized job payload (statement/observations + extracted
# record text). Extraction happens on the web pod, so this is the size of the
# text the worker receives, not the size of the uploaded PDFs.
JOB_QUEUE_MAX_PAYLOAD_BYTES = _positive_int_env(
    "VA_LSE_JOB_QUEUE_MAX_PAYLOAD_BYTES", 32 * 1024 * 1024
)
# Above this size a job's extracted documents go to the blob store and the queue
# carries only a small reference (see app/blob_store.py). Small jobs stay inline —
# an extra round trip for a 40 KB job is pure overhead, and inlining keeps the
# zero-configuration path working.
JOB_QUEUE_INLINE_MAX_BYTES = _positive_int_env("VA_LSE_JOB_QUEUE_INLINE_MAX_BYTES", 256 * 1024)

# ---------------------------------------------------------------------------
# Blob store for job documents too large to inline (see app/blob_store.py).
#
# Keeping tens of megabytes of extracted record text out of Redis matters: the
# reference StatefulSet ships with a 256 MB maxmemory and an LRU policy that will
# evict a queued job, and Upstash bills per request and per byte.
#
# Mode: auto (filesystem when the queue is on, else none), filesystem, s3, none.
# The filesystem backend is stdlib-only but the directory MUST be shared between
# the web tier and every worker (K8s ReadWriteMany PVC / Compose named volume) —
# a per-pod emptyDir writes fine and fails on the worker.
# The s3 backend works with AWS S3 / Cloudflare R2 / MinIO / DO Spaces and needs
# boto3, which is NOT in requirements.txt (install requirements-s3.txt).
# ---------------------------------------------------------------------------
BLOB_STORE_MODE = os.getenv("VA_LSE_BLOB_STORE", "auto").strip().lower() or "auto"
BLOB_DIR = os.getenv("VA_LSE_BLOB_DIR", "blobs").strip() or "blobs"
BLOB_S3_BUCKET = os.getenv("VA_LSE_BLOB_S3_BUCKET", "").strip()
BLOB_S3_PREFIX = os.getenv("VA_LSE_BLOB_S3_PREFIX", "va-lse").strip()
BLOB_S3_ENDPOINT_URL = os.getenv("VA_LSE_BLOB_S3_ENDPOINT_URL", "").strip()
# Worker identity surfaced in logs, /health, and the job record.
JOB_QUEUE_WORKER_ID = os.getenv("VA_LSE_WORKER_ID", "").strip()
# Worker pool: how many jobs one worker process runs concurrently. Each job can
# peak ~1.8 GB on a 2,000-page bundle (PERFORMANCE.md), so raise the worker
# pod's memory limit alongside this.
JOB_QUEUE_WORKER_CONCURRENCY = _positive_int_env("VA_LSE_WORKER_CONCURRENCY", 1)
# Worker health sidecar port. Separate from HEALTH_PORT because a worker only
# ever runs one of these per pod; 0 disables it.
JOB_QUEUE_WORKER_HEALTH_PORT = _int_env("VA_LSE_WORKER_HEALTH_PORT", 8002)


# ---------------------------------------------------------------------------
# Production profiler (see app/profiler.py).  When VA_LSE_PROFILE_RUNS=1 the
# app emits per-phase timing breakdowns (p50/p95/p99) to the structured log
# after each Evaluate/Draft run.  Disabled by default to avoid overhead.
# ---------------------------------------------------------------------------
PROFILE_RUNS = os.getenv("VA_LSE_PROFILE_RUNS", "").strip() in ("1", "true", "True", "yes")

# ---------------------------------------------------------------------------
# Distributed tracing (OpenTelemetry) — see app/tracing.py and TRACING.md.
#
# Off by default, and a silent no-op when the OpenTelemetry packages are not
# installed, so the default install stays dependency-light and behaviour is
# unchanged. When on, one trace covers a run end to end: the web pod's submit
# span, every pipeline phase, and — in Pattern C — the worker that executes it.
#
# Export is OTLP, which Jaeger, Grafana Tempo, Datadog, New Relic and Honeycomb
# all ingest, so no vendor SDK is involved. Exporter settings use the standard
# OTEL_* variables (OTEL_EXPORTER_OTLP_ENDPOINT, ..._TRACES_ENDPOINT, _HEADERS,
# _COMPRESSION, OTEL_SERVICE_NAME, OTEL_RESOURCE_ATTRIBUTES, OTEL_SDK_DISABLED)
# so the OpenTelemetry docs apply verbatim; only sampling and span-detail knobs
# are app-specific.
# ---------------------------------------------------------------------------
TRACING_ENABLED = os.getenv("VA_LSE_TRACING", "").strip() in ("1", "true", "True", "yes")
TRACING_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "").strip() or "va-lay-statement-evaluator"
# Fraction of runs traced (0.0–1.0). A parent-based sampler keeps a sampled run's
# whole span tree, so a partial ratio drops whole traces rather than filling in
# the middle of one. Kept at 1.0 because enabling tracing at all is opt-in.
TRACING_SAMPLE_RATIO = min(
    1.0, max(0.0, _float_env("VA_LSE_TRACE_SAMPLE_RATIO", 1.0) or 0.0)
)
# One span per record-digest chunk. Off by default: a 5,000-page bundle is
# hundreds of chunks, and a span each would dwarf every other span in the trace
# (and your APM bill). The fan-out is still visible as a single `records:digest`
# span carrying chunks/concurrency; turn this on to see individual chunk times.
TRACING_CHUNK_SPANS = os.getenv("VA_LSE_TRACE_CHUNK_SPANS", "").strip() in ("1", "true", "True", "yes")
# One span per LLM call. Off by default for the same cardinality reason — the
# bulk-digest pass makes one call per chunk. Turn on to attribute latency to the
# endpoint rather than to a phase.
TRACING_LLM_CALLS = os.getenv("VA_LSE_TRACE_LLM_CALLS", "").strip() in ("1", "true", "True", "yes")
# Role label attached to every span's resource, so a trace shows which tier a
# span came from. "web" is the Streamlit process, "worker" is app.worker.
TRACING_ROLE = os.getenv("VA_LSE_TRACE_ROLE", "").strip() or "web"
