"""Offline tests for scripts/sonar_watch.py — the .env rewrite and switch choice.

The watcher's whole value is the edit it makes to ``.env`` the hour sonar
passes, and that edit is one keystroke from a broken deployment: it must swap
exactly the documented keys, append what is missing, rewrite duplicate
assignments, and leave the API key line, comments, and lookalike keys
(``OPENAI_BASE_URL_FALLBACK``) untouched. ``set_env_keys`` is pure, so all of
that is pinned without a network or a real ``.env``. The switch choice pins
the documented preference: the Agent API switch (keeps OPENAI_BASE_URL) over
the chat-route switch.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)


def _load_watch():
    spec = importlib.util.spec_from_file_location(
        "sonar_watch", Path(__file__).resolve().parent.parent / "scripts" / "sonar_watch.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestSetEnvKeys(unittest.TestCase):
    ENV = (
        "# Environment / secrets\n"
        "OPENAI_API_KEY=pplx-test-fixture-not-a-real-key\n"
        "OPENAI_BASE_URL=https://api.perplexity.ai/v1\n"
        "OPENAI_BASE_URL_FALLBACK=https://fallback.example/v1\n"
        "LLM_MODEL_MAIN=perplexity/kimi-k3\n"
        "# LLM_MODEL_FAST=commented-out\n"
        "LLM_MODEL_FAST=perplexity/glm-5.3-flash\n"
    )

    def test_replaces_exactly_the_requested_keys(self) -> None:
        watch = _load_watch()
        out = watch.set_env_keys(
            self.ENV, {"LLM_MODEL_MAIN": "sonar-pro", "LLM_MODEL_FAST": "sonar"}
        )
        self.assertIn("LLM_MODEL_MAIN=sonar-pro\n", out)
        self.assertIn("LLM_MODEL_FAST=sonar\n", out)
        self.assertNotIn("kimi-k3", out)
        self.assertNotIn("glm-5.3-flash", out)

    def test_key_comments_and_lookalikes_survive(self) -> None:
        watch = _load_watch()
        out = watch.set_env_keys(
            self.ENV, {"OPENAI_BASE_URL": "https://api.perplexity.ai"}
        )
        # The key line is not a KEY= assignment and must never match.
        self.assertIn("OPENAI_API_KEY=pplx-test-fixture-not-a-real-key\n", out)
        # A commented assignment is a comment, not an assignment.
        self.assertIn("# LLM_MODEL_FAST=commented-out\n", out)
        # The lookalike key keeps its value: OPENAI_BASE_URL must not prefix-
        # match OPENAI_BASE_URL_FALLBACK.
        self.assertIn("OPENAI_BASE_URL_FALLBACK=https://fallback.example/v1\n", out)
        self.assertIn("OPENAI_BASE_URL=https://api.perplexity.ai\n", out)

    def test_missing_keys_are_appended(self) -> None:
        watch = _load_watch()
        out = watch.set_env_keys(self.ENV, {"VA_LSE_LLM_ENDPOINT_SCHEMA": "chat"})
        self.assertTrue(out.endswith("VA_LSE_LLM_ENDPOINT_SCHEMA=chat\n"))
        self.assertTrue(out.startswith("# Environment / secrets\n"))

    def test_duplicate_assignments_all_rewrite(self) -> None:
        # Which duplicate wins depends on the loader; a switch must not depend
        # on that, so every occurrence is rewritten to the same value.
        watch = _load_watch()
        text = "LLM_MODEL_MAIN=old-one\nLLM_MODEL_MAIN=old-two\n"
        out = watch.set_env_keys(text, {"LLM_MODEL_MAIN": "sonar-pro"})
        self.assertEqual(out, "LLM_MODEL_MAIN=sonar-pro\nLLM_MODEL_MAIN=sonar-pro\n")

    def test_idempotent_when_already_applied(self) -> None:
        watch = _load_watch()
        once = watch.set_env_keys(
            self.ENV, {"LLM_MODEL_MAIN": "sonar-pro", "LLM_MODEL_FAST": "sonar"}
        )
        twice = watch.set_env_keys(
            once, {"LLM_MODEL_MAIN": "sonar-pro", "LLM_MODEL_FAST": "sonar"}
        )
        self.assertEqual(once, twice)


class TestChooseSwitch(unittest.TestCase):
    def test_prefers_the_agent_api_switch(self) -> None:
        watch = _load_watch()
        self.assertEqual(watch.choose_switch(True, True), "responses")
        self.assertEqual(watch.choose_switch(True, False), "responses")

    def test_falls_back_to_the_chat_route(self) -> None:
        watch = _load_watch()
        self.assertEqual(watch.choose_switch(False, True), "chat")

    def test_none_when_neither_route_serves_sonar(self) -> None:
        watch = _load_watch()
        self.assertIsNone(watch.choose_switch(False, False))


class TestSwitchBlocks(unittest.TestCase):
    def test_responses_block_only_touches_the_model_lines(self) -> None:
        watch = _load_watch()
        keys = set(watch.SWITCHES["responses"])
        self.assertEqual(keys, {"LLM_MODEL_MAIN", "LLM_MODEL_FAST"})

    def test_chat_block_pins_schema_and_base_url(self) -> None:
        watch = _load_watch()
        block = watch.SWITCHES["chat"]
        self.assertEqual(block["OPENAI_BASE_URL"], watch.CHAT_BASE_URL)
        self.assertEqual(block["VA_LSE_LLM_ENDPOINT_SCHEMA"], "chat")
        # The pin is load-bearing: without it app/llm.py HEAD-probes the host
        # into the Responses schema and posts /responses where sonar is
        # rejected — the switch would look applied and serve nothing.
        self.assertNotIn("VA_LSE_LLM_ENDPOINT_SCHEMA", watch.SWITCHES["responses"])


class TestActiveBatchRun(unittest.TestCase):
    def test_live_pid_from_any_outputs_dir_is_found(self) -> None:
        watch = _load_watch()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "outputs" / "run-a").mkdir(parents=True)
            (root / "outputs" / "run-a" / "rerun.pid").write_text(str(os.getpid()))
            with patch.object(watch, "ROOT", root):
                self.assertEqual(watch.active_batch_run(), os.getpid())

    def test_stale_pid_is_ignored(self) -> None:
        watch = _load_watch()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "outputs" / "run-a").mkdir(parents=True)
            # A pid that cannot be a live process: recycled numbers are the
            # risk here, but a missing outputs tree (the common case when no
            # run has ever started) must not error either.
            (root / "outputs" / "run-a" / "rerun.pid").write_text("not-a-pid")
            with patch.object(watch, "ROOT", root):
                self.assertIsNone(watch.active_batch_run())


if __name__ == "__main__":
    unittest.main()
