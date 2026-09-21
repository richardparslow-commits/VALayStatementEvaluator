"""The hosted page-cap guardrail: one deployment-aware rejection everywhere.

The page cap already existed (``VA_LSE_MAX_RECORD_PAGES``, default 5,000) and
was enforced in three layers. What was wrong was the *remedy text*: every layer
told a hosted user to "raise VA_LSE_MAX_RECORD_PAGES" — an operator
environment variable a hosted user cannot touch. All three layers now share
``page_limit_message()``, which points a hosted user at the local tier (and a
local user at the setting they actually control). The message reads the
environment at call time, so flipping ``VA_LSE_ALLOW_LOCAL_PATHS`` changes the
text without a restart — and the tests flip it in-process.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app import config
from app.documents import page_limit_message


def _set_env(**values: str | None) -> None:
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


class TestPageLimitMessage(unittest.TestCase):
    def tearDown(self) -> None:
        _set_env(VA_LSE_ALLOW_LOCAL_PATHS=None)

    def test_hosted_default_points_at_the_local_tier(self) -> None:
        _set_env(VA_LSE_ALLOW_LOCAL_PATHS=None)
        msg = page_limit_message(700, 500)
        self.assertIn("700", msg)
        self.assertIn("500", msg)
        self.assertIn("hosted", msg.lower())
        self.assertIn("locally", msg.lower())
        # No operator env-var advice on the hosted path.
        self.assertNotIn("VA_LSE_MAX_RECORD_PAGES", msg)

    def test_limit_defaults_to_config_when_omitted(self) -> None:
        _set_env(VA_LSE_ALLOW_LOCAL_PATHS=None)
        msg = page_limit_message(9_999)
        self.assertIn(f"{config.MAX_RECORD_PAGES:,}", msg)

    def test_local_run_names_the_setting_the_operator_controls(self) -> None:
        _set_env(VA_LSE_ALLOW_LOCAL_PATHS="1")
        msg = page_limit_message(700, 500)
        self.assertIn("VA_LSE_MAX_RECORD_PAGES", msg)
        self.assertIn(".env", msg)
        self.assertNotIn("hosted", msg.lower())

    def test_message_rereads_env_each_call(self) -> None:
        _set_env(VA_LSE_ALLOW_LOCAL_PATHS=None)
        self.assertIn("hosted", page_limit_message(700, 500).lower())
        _set_env(VA_LSE_ALLOW_LOCAL_PATHS="1")
        self.assertIn("VA_LSE_MAX_RECORD_PAGES", page_limit_message(700, 500))
        _set_env(VA_LSE_ALLOW_LOCAL_PATHS=None)


class TestBackstopRaiseUsesSharedMessage(unittest.TestCase):
    """The run-time backstop raises ValueError carrying the shared message."""

    def test_backstop_message_is_shared_and_deployment_aware(self) -> None:
        from app.documents import extract_document
        from app.medical_review import review_medical_records
        from tests.test_core import FakeLLM

        _set_env(VA_LSE_ALLOW_LOCAL_PATHS=None)
        with mock.patch.object(config, "MAX_RECORD_PAGES", 1):
            with self.assertRaises(ValueError) as ctx:
                review_medical_records(FakeLLM(), [
                    extract_document("a.txt", b"EVT one"),
                    extract_document("b.txt", b"EVT two"),
                ])
        msg = str(ctx.exception)
        self.assertIn("hosted", msg.lower())
        self.assertIn("locally", msg.lower())
        # The old backstop text (operator-only remedy) is gone.
        self.assertNotIn("raise VA_LSE_MAX_RECORD_PAGES", msg)


if __name__ == "__main__":
    unittest.main()
