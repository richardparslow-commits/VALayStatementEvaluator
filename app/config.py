"""Application settings.

Settings are loaded from environment variables (optionally via a .env file) and can be
overridden at runtime from the Streamlit sidebar.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"

# Load .env from the project root if present (never committed). override=True so
# the project's .env wins over any pre-existing environment values.
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Default endpoint tuned for the QwenCloud Individual Plan Lite subscription.
# Token Plan uses a dedicated sk-sp- API key that MUST be paired with this
# base URL (Token Plan keys do not work against the general MaaS gateway).
DEFAULT_BASE_URL = (
    "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
)
# Low-volume, high-value calls (claim extraction, verification, rubric scoring,
# topic audit, rewrite) use the strong reasoning model.
DEFAULT_MODEL_MAIN = "qwen3.7-max"
# The bulk digest/merge passes (one call per record chunk — by far the most
# calls) use the cheap model to preserve the Lite plan's limited credit quota.
DEFAULT_MODEL_FAST = "qwen3.7-flash"
DEFAULT_FETCH_SANDBOX_BASE_URL = "https://fetchsandbox.com"
DEFAULT_FETCH_SANDBOX_RECORDS_PATH = "/medical_records/{patient_id}"


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

    @property
    def configured(self) -> bool:
        return bool(self.api_key.strip())

    @property
    def fetch_configured(self) -> bool:
        return bool(self.fetch_base_url.strip() and self.fetch_records_path.strip())


def load_settings() -> Settings:
    """Build settings from the environment with sensible defaults."""
    return Settings(
        api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL).strip()
        or DEFAULT_BASE_URL,
        model_main=os.getenv("LLM_MODEL_MAIN", DEFAULT_MODEL_MAIN).strip()
        or DEFAULT_MODEL_MAIN,
        model_fast=os.getenv("LLM_MODEL_FAST", DEFAULT_MODEL_FAST).strip()
        or DEFAULT_MODEL_FAST,
        fetch_api_key=os.getenv("FETCH_SANDBOX_API_KEY", "").strip(),
        fetch_base_url=os.getenv(
            "FETCH_SANDBOX_BASE_URL", DEFAULT_FETCH_SANDBOX_BASE_URL
        ).strip()
        or DEFAULT_FETCH_SANDBOX_BASE_URL,
        fetch_records_path=os.getenv(
            "FETCH_SANDBOX_RECORDS_PATH", DEFAULT_FETCH_SANDBOX_RECORDS_PATH
        ).strip()
        or DEFAULT_FETCH_SANDBOX_RECORDS_PATH,
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
