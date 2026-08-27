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

DEFAULT_BASE_URL = (
    "https://ws-tc9lxz8v26y5ywu1.us-east-1.maas.aliyuncs.com/compatible-mode/v1"
)
DEFAULT_MODEL_MAIN = "qwen3.7-max"
DEFAULT_MODEL_FAST = "qwen3.7-flash"


@dataclass
class Settings:
    """Runtime LLM settings (mutable; sidebar edits update the instance)."""

    api_key: str
    base_url: str
    model_main: str
    model_fast: str

    @property
    def configured(self) -> bool:
        return bool(self.api_key.strip())


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
def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


# Hard cap on total pages across all uploaded record files.
MAX_RECORD_PAGES = _int_env("VA_LSE_MAX_RECORD_PAGES", 5000)

# Number of record chunks digested in parallel. Keep modest to respect API
# rate limits; 6 is a good balance for ~2,000-page sets.
RECORDS_CONCURRENCY = max(1, _int_env("VA_LSE_RECORDS_CONCURRENCY", 6))

# Maximum facts kept in the digest after consolidation.
MAX_DIGEST_FACTS = _int_env("VA_LSE_MAX_DIGEST_FACTS", 1500)

# Characters per record chunk. Smaller chunks => more LLM calls but better
# recall on dense pages (nothing gets truncated mid-extraction).
DIGEST_CHUNK_CHARS = _int_env("VA_LSE_DIGEST_CHUNK_CHARS", 8000)
