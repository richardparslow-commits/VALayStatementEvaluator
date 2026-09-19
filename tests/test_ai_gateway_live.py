"""Live check of the Vercel AI Gateway path — opt-in, skipped without a key.

The app speaks OpenAI-compatible Chat Completions, and Vercel's **AI Gateway** serves
exactly that (`https://ai-gateway.vercel.sh/v1`) with model ids from its own catalog.
Nothing offline can prove a *credential*, and nothing offline can prove a catalog, so
this runs the app's own client against the real gateway:

    VA_LSE_TEST_AI_GATEWAY_KEY=vck_... python -m unittest tests.test_ai_gateway_live

It skips without that variable, so CI is unaffected, and it uses the app's own
`probe_models` / `LLMClient` rather than raw HTTP — the point is to exercise the code
path a deployment would use, not the gateway.

What it establishes: the gateway answers `GET /models` (which is the preflight's
probe), the ids `COMPATIBILITY.md` documents are in that catalog, and a small chat
plus a JSON-mode call complete and parse. What it deliberately does *not* assert is
entitlement: a gateway account's tier decides which models a key may call ("Free tier
users do not have access to this model"), so a restricted model is that account's
configuration rather than a compatibility failure. Those outcomes skip with the
provider's own message — set `VA_LSE_TEST_AI_GATEWAY_MODEL` to a model the account can
use to exercise the calls themselves.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app.config import load_settings  # noqa: E402
from app.llm import LLMClient, LLMError, probe_models  # noqa: E402

KEY_ENV = "VA_LSE_TEST_AI_GATEWAY_KEY"
MODEL_ENV = "VA_LSE_TEST_AI_GATEWAY_MODEL"
BASE_URL = "https://ai-gateway.vercel.sh/v1"
DEFAULT_MODEL = "openai/gpt-4.1-nano"

#: The gateway ids COMPATIBILITY.md names. Presence is stable; entitlement is not —
#: a free-tier account is refused `moonshotai/kimi-k3` with a 403, so this asserts
#: only that the catalog carries them.
DOCUMENTED_IDS = (
    "moonshotai/kimi-k3",
    "alibaba/qwen3.7-flash",
    "perplexity/sonar-pro",
    "openai/gpt-4.1-nano",
)


class AiGatewayTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.key = (os.getenv(KEY_ENV) or "").strip()
        if not self.key:
            self.skipTest(f"set {KEY_ENV} to check the real Vercel AI Gateway")
        self.model = (os.getenv(MODEL_ENV) or "").strip() or DEFAULT_MODEL
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": self.key,
                "OPENAI_BASE_URL": BASE_URL,
                "LLM_MODEL_MAIN": self.model,
                "LLM_MODEL_FAST": self.model,
            },
            clear=False,
        ):
            self.settings = load_settings()

    def _entitlement_skip(self, exc: BaseException) -> None:
        """Skip when the account may not use the model; fail on anything else.

        The gateway answers 403 ("Free tier users do not have access to this model")
        or 429 for a tier/spend limit, and both are the account's configuration. A
        transport failure, a 400 or an unparseable answer is a real incompatibility
        and must fail.
        """
        message = str(exc)
        restricted = any(
            marker in message for marker in ("403", "429", "Free tier", "no_providers_available")
        )
        if restricted:
            self.skipTest(f"this gateway key/account cannot call {self.model}: {message[:200]}")
        raise exc


class TestTheGatewayCatalog(AiGatewayTestCase):
    def test_the_preflights_probe_reads_the_gateway_catalog(self) -> None:
        result = probe_models(self.settings.base_url, self.settings.api_key)

        self.assertEqual(result.status, 200, result.error)
        self.assertTrue(result.ok)
        self.assertIn("openai/gpt-4.1-nano", result.models or set())

    def test_the_ids_compatibility_documents_are_listed(self) -> None:
        available = probe_models(self.settings.base_url, self.settings.api_key).models or set()

        missing = [model for model in DOCUMENTED_IDS if model not in available]
        self.assertEqual(missing, [], "COMPATIBILITY.md names ids the gateway no longer lists")


class TestTheAppsClientAgainstTheGateway(AiGatewayTestCase):
    def test_a_small_chat_completes(self) -> None:
        client = LLMClient(self.settings)
        try:
            text = client.chat(
                "You are terse.",
                "Reply with exactly: pong",
                model=self.model,
                max_tokens=16,
                temperature=0.0,
                phase="live_gateway",
            )
        except LLMError as exc:  # entitlement is not compatibility
            self._entitlement_skip(exc)
            return
        self.assertTrue(text.strip())

    def test_a_json_mode_call_parses(self) -> None:
        client = LLMClient(self.settings)
        try:
            data = client.chat_json(
                "You return JSON.",
                'Return {"ok": true, "n": 2}',
                model=self.model,
                max_tokens=64,
                temperature=0.0,
                phase="live_gateway",
            )
        except LLMError as exc:
            self._entitlement_skip(exc)
            return
        self.assertIsInstance(data, dict)
        self.assertTrue(data.get("ok"))


if __name__ == "__main__":
    unittest.main()
