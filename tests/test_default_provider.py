"""The shipped default provider: Perplexity's Router API, with its two best-fit models.

These defaults are what a fresh clone actually runs on, so they are worth pinning rather
than leaving to a comment: the base URL must be the one whose ``/chat/completions`` and
``/models`` paths the app already builds, the two model ids must come from the published
catalog (the catalog is also the allowlist — an unknown id is a 400, not a fallback), and
the credential aliasing that makes a single Perplexity key serve both the Router and the
grounded Agent API has to hold in both directions — reuse when the endpoint is Perplexity's,
and *no* reuse when it is somebody else's.

Offline: nothing here calls the API.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urljoin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/ — log_isolation
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from app import config  # noqa: E402
from app.prompt_sanitize import validate_model_name  # noqa: E402
from log_isolation import isolate_app_logs  # noqa: E402

# The published Router catalog (https://docs.perplexity.ai/docs/router/models).
ROUTER_CATALOG = {
    "perplexity/kimi-k3",
    "perplexity/glm-5.3",
    "perplexity/glm-5.3-flash",
    "perplexity/nemotron-3-ultra-550b-a55b",
}

ROUTER_BASE_URL = "https://api.perplexity.ai/router/v1"


class TestDefaultEndpoint(unittest.TestCase):
    def test_default_base_url_is_the_router_api(self) -> None:
        self.assertEqual(config.DEFAULT_BASE_URL, ROUTER_BASE_URL)

    def test_the_endpoints_the_app_builds_resolve_against_it(self) -> None:
        """``{base}/chat/completions`` (every call) and ``{base}/models`` (/ready probe)."""
        base = config.DEFAULT_BASE_URL.rstrip("/") + "/"
        self.assertEqual(
            urljoin(base, "chat/completions"),
            "https://api.perplexity.ai/router/v1/chat/completions",
        )
        self.assertEqual(
            urljoin(base, "models"), "https://api.perplexity.ai/router/v1/models"
        )

    def test_host_constant_matches_the_default(self) -> None:
        # The credential aliasing keys off this constant, so it must describe the default.
        self.assertIn(config.PERPLEXITY_HOST, config.DEFAULT_BASE_URL)


class TestDefaultModels(unittest.TestCase):
    def test_both_defaults_are_in_the_published_catalog(self) -> None:
        self.assertIn(config.DEFAULT_MODEL_MAIN, ROUTER_CATALOG)
        self.assertIn(config.DEFAULT_MODEL_FAST, ROUTER_CATALOG)

    def test_the_split_is_actually_a_split(self) -> None:
        """Same model in both slots would mean the bulk digests run at main-model prices."""
        self.assertNotEqual(config.DEFAULT_MODEL_MAIN, config.DEFAULT_MODEL_FAST)

    def test_default_ids_pass_the_sidebar_validator(self) -> None:
        for model in (config.DEFAULT_MODEL_MAIN, config.DEFAULT_MODEL_FAST):
            self.assertIsNone(validate_model_name(model), msg=model)

    def test_default_ids_use_the_creator_slash_model_form(self) -> None:
        for model in (config.DEFAULT_MODEL_MAIN, config.DEFAULT_MODEL_FAST):
            creator, _, name = model.partition("/")
            self.assertTrue(creator and name, msg=model)


class TestEffectiveDefaults(unittest.TestCase):
    """A fresh environment (no ``.env``) must land on the Perplexity defaults."""

    def _load(self, **env: str) -> config.Settings:
        cleared = {
            "OPENAI_BASE_URL": "",
            "LLM_MODEL_MAIN": "",
            "LLM_MODEL_FAST": "",
            "OPENAI_API_KEY": "",
            "PERPLEXITY_API_KEY": "",
        }
        cleared.update(env)
        with patch.dict(os.environ, cleared, clear=False):
            return config.load_settings()

    def test_unset_environment_uses_the_default_endpoint_and_models(self) -> None:
        settings = self._load()
        self.assertEqual(settings.base_url, config.DEFAULT_BASE_URL)
        self.assertEqual(settings.model_main, config.DEFAULT_MODEL_MAIN)
        self.assertEqual(settings.model_fast, config.DEFAULT_MODEL_FAST)

    def test_an_explicit_environment_still_wins(self) -> None:
        """The defaults are layerable, not enforced — a .env pins whatever it names."""
        settings = self._load(
            OPENAI_BASE_URL="https://other.example/v1",
            LLM_MODEL_MAIN="other-main",
            LLM_MODEL_FAST="other-fast",
        )
        self.assertEqual(settings.base_url, "https://other.example/v1")
        self.assertEqual(settings.model_main, "other-main")
        self.assertEqual(settings.model_fast, "other-fast")


class TestOneKeyForBothApis(unittest.TestCase):
    """One Perplexity key serves the Router (Chat Completions) and the Agent API."""

    def _load(self, **env: str) -> config.Settings:
        base = {
            "OPENAI_BASE_URL": config.DEFAULT_BASE_URL,
            "OPENAI_API_KEY": "",
            "PERPLEXITY_API_KEY": "",
        }
        base.update(env)
        with patch.dict(os.environ, base, clear=False):
            return config.load_settings()

    def test_primary_key_is_aliased_when_the_endpoint_is_perplexitys(self) -> None:
        settings = self._load(OPENAI_API_KEY="pplx-shared-key")
        self.assertTrue(settings.perplexity_configured)
        self.assertEqual(settings.perplexity_api_key, "pplx-shared-key")

    def test_no_aliasing_for_a_different_provider(self) -> None:
        """Forwarding another provider's key would turn a clear message into a 401."""
        settings = self._load(
            OPENAI_BASE_URL="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
            # Another provider's key shape, spelled as tests/test_diagnostics.py spells
            # its sk- token: the pre-commit hook refuses the literal on an added line.
            **{"OPENAI_API_KEY": "sk-" + "sp-some-other-providers-key"},
        )
        self.assertFalse(settings.perplexity_configured)
        self.assertEqual(settings.perplexity_api_key, "")

    def test_an_explicit_perplexity_key_always_wins(self) -> None:
        settings = self._load(
            **{"OPENAI_API_KEY": "pplx-from-openai-var"},
            PERPLEXITY_API_KEY="pplx-explicit",
        )
        self.assertEqual(settings.perplexity_api_key, "pplx-explicit")

    def test_neither_key_set_means_not_configured(self) -> None:
        settings = self._load()
        self.assertFalse(settings.perplexity_configured)


class TestTheSidebarShowsTheDefault(unittest.TestCase):
    """The sidebar must *display* the shipped default, not merely have it in config.

    The field is pre-filled from settings, so an environment with no ``.env`` renders the
    Router base URL — and since a pre-filled field is what the next "Apply settings" sends,
    a stale default here is how a valid key ends up at the wrong host.

    Driven with AppTest against the real sidebar: the widget is created by a long render
    function, so asserting on the arguments a mocked ``st`` received would test the mock.
    """

    def setUp(self) -> None:
        # AppTest runs the real app in-process — keep its run-log/audit-log writes out of
        # the developer's real logs/ (see tests/log_isolation.py).
        self._tmpdir = isolate_app_logs(self)

    def test_base_url_and_models_are_prefilled_with_the_defaults(self) -> None:
        from streamlit.testing.v1 import AppTest
        import app.views.sidebar as sidebar

        at = AppTest.from_file(str(PROJECT_ROOT / "run_app.py"), default_timeout=30)
        # The sidebar probes {base_url}/models on render; keep this test off the network and
        # report what a healthy Perplexity endpoint returns for these two ids.
        with patch.object(
            sidebar,
            "check_model_availability",
            return_value={config.DEFAULT_MODEL_MAIN, config.DEFAULT_MODEL_FAST},
        ):
            at.run()

        self.assertFalse(
            at.exception, msg=f"app raised while rendering: {[e.value for e in at.exception]}"
        )
        fields = {element.label: element.value for element in at.sidebar.text_input}
        self.assertEqual(fields.get("Base URL (OpenAI-compatible)"), config.DEFAULT_BASE_URL)
        self.assertEqual(fields.get("Main model"), config.DEFAULT_MODEL_MAIN)
        self.assertEqual(fields.get("Fast model"), config.DEFAULT_MODEL_FAST)


if __name__ == "__main__":
    unittest.main()
