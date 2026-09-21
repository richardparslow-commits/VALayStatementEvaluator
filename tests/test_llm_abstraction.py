"""Phase-1 LLM abstraction: the boundary is a tested invariant, not a convention.

Three guarantees, each with a reason to exist:

* **SDK isolation.** Only ``app/llm.py`` may import an LLM SDK; every component
  (draft, evaluate, medical_review, views, worker) goes through
  ``LLMClient.chat``/``chat_json`` on a settings-driven client. That is what
  makes the endpoint a *setting* — the future BAA-covered hosted tier is a
  configuration change, not a rewrite. Nothing stopped a view from
  ``import openai`` until now except discipline.
* **Protocol conformance.** ``LLMService`` is the typed surface components
  consume. The concrete client satisfies it, and so must any replacement
  backend (brokered HIPAA service, batch API, local runner).
* **Provider-neutral aliases.** ``LLM_PROVIDER_URL`` / ``LLM_API_KEY`` resolve
  exactly like their canonical ``OPENAI_*`` names, with the canonical name
  winning when both are set, so a hosted-tier platform team can provision
  generic names without the app caring.

Offline: nothing here calls the API.
"""

from __future__ import annotations

import os
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/ — log_isolation
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from log_isolation import isolate_app_logs  # noqa: E402

from app import config  # noqa: E402
from app.llm import LLMClient, LLMService  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Any OpenAI-compatible SDK import spells itself one of these two ways.
_SDK_IMPORT_RE = re.compile(r"^\s*(?:from openai import|import openai\b)", re.MULTILINE)


class _FakeSettings:
    """Shape-compatible Settings for offline client construction (house pattern)."""

    configured = True
    api_key = "test-key"
    base_url = "http://example.invalid"
    model_main = "test-model"
    model_fast = "test-fast"


class TestSdkIsolation(unittest.TestCase):
    """The SDK boundary is structural: one module owns it, tests enforce it."""

    def test_openai_sdk_imports_live_only_in_the_client_module(self) -> None:
        offenders: list[str] = []
        for path in sorted((PROJECT_ROOT / "app").rglob("*.py")):
            if path.name == "llm.py":
                continue
            if _SDK_IMPORT_RE.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path.relative_to(PROJECT_ROOT)))
        self.assertEqual(
            offenders,
            [],
            "components must not import the LLM SDK — route calls through "
            "app/llm.py's LLMClient (the LLMService boundary) so the endpoint "
            "stays a configuration change, not a rewrite",
        )

    def test_the_client_module_is_the_only_importer(self) -> None:
        """Positive control: the guard above must actually see the real import."""
        text = (PROJECT_ROOT / "app" / "llm.py").read_text(encoding="utf-8")
        self.assertTrue(_SDK_IMPORT_RE.search(text))


class MinimalBackend:
    """The smallest class a future backend needs — no SDK, no HTTP."""

    fast_model = "cheap-model"  # the bulk-call contract member

    def chat(self, system: str, user: str, *, model=None, temperature=0.2,
             max_tokens=8000, phase="general") -> str:
        return "ok"

    def chat_json(self, system: str, user: str, *, model=None, temperature=0.1,
                  max_tokens=8000, phase="general"):
        return {}


class TestLLMServiceProtocol(unittest.TestCase):
    def test_the_concrete_client_is_an_llmservice(self) -> None:
        self.assertIsInstance(LLMClient(_FakeSettings()), LLMService)

    def test_a_minimal_backend_satisfies_the_boundary(self) -> None:
        """A replacement backend needs only the two methods — that is the contract."""
        self.assertIsInstance(MinimalBackend(), LLMService)

    def test_a_class_missing_chat_json_does_not_satisfy_it(self) -> None:
        class Half:
            def chat(self, system, user, **_):
                return ""

        self.assertNotIsInstance(Half(), LLMService)


class TestProviderNeutralAliases(unittest.TestCase):
    """``LLM_PROVIDER_URL`` / ``LLM_API_KEY`` resolve like the canonical names."""

    def _load(self, **env: str) -> config.Settings:
        cleared = {
            "OPENAI_BASE_URL": "",
            "LLM_PROVIDER_URL": "",
            "OPENAI_API_KEY": "",
            "LLM_API_KEY": "",
            "PERPLEXITY_API_KEY": "",
            "LLM_MODEL_MAIN": "",
            "LLM_MODEL_FAST": "",
        }
        cleared.update(env)
        with patch.dict(os.environ, cleared, clear=False):
            return config.load_settings()

    def test_alias_url_is_used_when_the_canonical_is_unset(self) -> None:
        settings = self._load(LLM_PROVIDER_URL="https://baa.example/v1")
        self.assertEqual(settings.base_url, "https://baa.example/v1")

    def test_the_canonical_url_wins_over_the_alias(self) -> None:
        settings = self._load(
            OPENAI_BASE_URL="https://canonical.example/v1",
            LLM_PROVIDER_URL="https://alias.example/v1",
        )
        self.assertEqual(settings.base_url, "https://canonical.example/v1")

    def test_alias_key_is_used_when_the_canonical_is_unset(self) -> None:
        settings = self._load(LLM_API_KEY="k-alias")
        self.assertEqual(settings.api_key, "k-alias")
        self.assertTrue(settings.configured)

    def test_the_canonical_key_wins_over_the_alias(self) -> None:
        settings = self._load(OPENAI_API_KEY="k-canonical", LLM_API_KEY="k-alias")
        self.assertEqual(settings.api_key, "k-canonical")

    def test_research_key_aliasing_flows_through_the_alias(self) -> None:
        """A Perplexity key supplied only as ``LLM_API_KEY`` still serves the Research tab."""
        settings = self._load(LLM_API_KEY="pplx-through-alias")
        self.assertEqual(settings.base_url, config.DEFAULT_BASE_URL)
        self.assertTrue(settings.perplexity_configured)
        self.assertEqual(settings.perplexity_api_key, "pplx-through-alias")

    def test_alias_names_never_appear_in_secrets_provenance(self) -> None:
        """Provenance tracks the effective canonical name only (sidebar display)."""
        settings = self._load(
            LLM_PROVIDER_URL="https://baa.example/v1", LLM_API_KEY="k-alias"
        )
        self.assertNotIn("LLM_PROVIDER_URL", settings.from_secrets)
        self.assertNotIn("LLM_API_KEY", settings.from_secrets)


if __name__ == "__main__":
    unittest.main()
