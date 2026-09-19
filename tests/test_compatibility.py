"""Tests for check_model_availability and compat warning wiring.
No network in the success path — responses are stubbed.
"""
import sys
import unittest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app import config  # noqa: E402
from app.llm import check_model_availability, MODELS_ENDPOINT_TIMEOUT_SECONDS  # noqa: E402


class TestCheckModelAvailability(unittest.TestCase):
    def test_returns_ids_on_success(self):
        payload = json.dumps({"data": [{"id": "qwen3.7-max"}, {"id": "qwen3.7-flash"}]}).encode()
        fake_resp = MagicMock()
        fake_resp.read.return_value = payload
        fake_resp.__enter__ = lambda _: fake_resp
        fake_resp.__exit__ = lambda *_: False
        with patch("urllib.request.urlopen", return_value=fake_resp) as mock_urlopen:
            ids = check_model_availability("https://example.com/v1", "sk-test")
            self.assertEqual(ids, {"qwen3.7-max", "qwen3.7-flash"})
            # called with /models and Authorization header
            req = mock_urlopen.call_args[0][0]
            self.assertTrue(req.full_url.endswith("/models"))
            self.assertIn("Bearer sk-test", req.headers.get("Authorization", ""))

    def test_returns_none_without_api_key(self):
        self.assertIsNone(check_model_availability("https://example.com/v1", ""))
        self.assertIsNone(check_model_availability("https://example.com/v1", "   "))

    def test_returns_none_on_network_error(self):
        with patch("urllib.request.urlopen", side_effect=Exception("network down")):
            self.assertIsNone(check_model_availability("https://example.com/v1", "sk-x"))

    def test_returns_none_on_malformed_json(self):
        fake_resp = MagicMock()
        fake_resp.read.return_value = b"not json"
        fake_resp.__enter__ = lambda _: fake_resp
        fake_resp.__exit__ = lambda *_: False
        with patch("urllib.request.urlopen", return_value=fake_resp):
            self.assertIsNone(check_model_availability("https://example.com/v1", "sk-x"))

    def test_timeout_constant(self):
        self.assertGreaterEqual(MODELS_ENDPOINT_TIMEOUT_SECONDS, 5)

    def test_stripsTrailingSlash(self):
        payload = json.dumps({"data": []}).encode()
        fake_resp = MagicMock()
        fake_resp.read.return_value = payload
        fake_resp.__enter__ = lambda _: fake_resp
        fake_resp.__exit__ = lambda *_: False
        with patch("urllib.request.urlopen", return_value=fake_resp) as mock_urlopen:
            check_model_availability("https://example.com/v1/", "sk-x")
            self.assertEqual(mock_urlopen.call_args[0][0].full_url, "https://example.com/v1/models")


class TestCompatDocsPresent(unittest.TestCase):
    def test_files_exist(self):
        root = Path(__file__).resolve().parent.parent
        for name in ("COMPATIBILITY.md", "MIGRATION.md"):
            self.assertTrue((root / name).exists(), msg=f"{name} should exist")
            text = (root / name).read_text()
            self.assertIn("OPENAI_API_KEY" if name == "MIGRATION.md" else "Compatibility", text)


class TestEnvExampleDocumentsProviders(unittest.TestCase):
    def test_env_example_lists_providers(self):
        text = (Path(__file__).resolve().parent.parent / ".env.example").read_text()
        self.assertIn("COMPATIBILITY.md", text)
        self.assertIn("Perplexity", text)
        self.assertIn("OpenAI", text)
        self.assertIn("Ollama", text)

    def test_env_example_defaults_match_the_shipped_defaults(self):
        """The file a fresh clone copies must name the defaults app/config.py ships.

        This drifted once: the Router API became the default while .env.example
        still described QwenCloud as "(default)" with a Token Plan key and its
        models, so a new deployment landed on the old provider with no signal.
        """
        values: dict[str, str] = {}
        for line in (
            Path(__file__).resolve().parent.parent / ".env.example"
        ).read_text().splitlines():
            if line and not line.lstrip().startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()
        self.assertEqual(values.get("OPENAI_BASE_URL"), config.DEFAULT_BASE_URL)
        self.assertEqual(values.get("LLM_MODEL_MAIN"), config.DEFAULT_MODEL_MAIN)
        self.assertEqual(values.get("LLM_MODEL_FAST"), config.DEFAULT_MODEL_FAST)
        self.assertTrue(
            values.get("OPENAI_API_KEY", "").startswith("pplx-"),
            "the placeholder key should be a Perplexity key (pplx-...), got "
            + repr(values.get("OPENAI_API_KEY")),
        )


if __name__ == "__main__":
    unittest.main()
