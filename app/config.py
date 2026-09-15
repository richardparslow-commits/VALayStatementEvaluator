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
    fetch_max_response_bytes: int = DEFAULT_FETCH_SANDBOX_MAX_RESPONSE_BYTES
    # Env-var names that were satisfied by Streamlit secrets rather than the
    # environment/.env — surfaced in the sidebar so a pre-filled value is not
    # mistaken for one the user typed.  See SECRET_FIELD_LABELS.
    from_secrets: frozenset[str] = frozenset()

    @property
    def configured(self) -> bool:
        return bool(self.api_key.strip())

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


# Sidecar health server (liveness + readiness) for container orchestration.
# Exposes GET /health (liveness) and GET /ready (readiness) on this port on
# 0.0.0.0. Override with VA_LSE_HEALTH_PORT; disabled when set to 0.
HEALTH_PORT = _int_env("VA_LSE_HEALTH_PORT", 8001)

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
# Audit logging — dedicated JSON stream for compliance/forensics (see app/audit.py).
# Separate from diagnostic app.log so it can be queried and retained under a
# different policy. Disabled only by setting dir to empty; otherwise writes to
# {AUDIT_LOG_DIR}/{AUDIT_LOG_FILE} with rotation.
# ---------------------------------------------------------------------------
AUDIT_LOG_DIR = os.getenv("VA_LSE_AUDIT_LOG_DIR", "").strip() or os.getenv("VA_LSE_LOG_DIR", "").strip() or "logs"
AUDIT_LOG_FILE = os.getenv("VA_LSE_AUDIT_LOG_FILE", "audit.log").strip() or "audit.log"
AUDIT_LOG_MAX_BYTES = _positive_int_env("VA_LSE_AUDIT_LOG_MAX_BYTES", 10 * 1024 * 1024)
AUDIT_LOG_BACKUPS = _positive_int_env("VA_LSE_AUDIT_LOG_BACKUPS", 10)

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
# Production profiler (see app/profiler.py).  When VA_LSE_PROFILE_RUNS=1 the
# app emits per-phase timing breakdowns (p50/p95/p99) to the structured log
# after each Evaluate/Draft run.  Disabled by default to avoid overhead.
# ---------------------------------------------------------------------------
PROFILE_RUNS = os.getenv("VA_LSE_PROFILE_RUNS", "").strip() in ("1", "true", "True", "yes")
