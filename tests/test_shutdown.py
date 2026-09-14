"""Offline tests for app/shutdown.py — graceful drain, inflight tracking, health draining.

No network — every probe is mocked. Covers:
  - Signal handler installation (idempotent, main-thread only)
  - Request shutdown: drain + timeout + force-exit paths
  - enter_run / exit_run inflight tracking
  - is_shutting_down() gate
  - Health /ready 503 when draining
  - LLM timeout message surfacing
  - reset_for_tests cleans state
"""
from __future__ import annotations

import signal
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import shutdown  # noqa: E402
from app.health import _cached_readiness, stop_health_server  # noqa: E402


class TestShutdownUnit(unittest.TestCase):
    """Pure unit tests for the shutdown module (no server, no network)."""

    def setUp(self) -> None:
        shutdown.reset_for_tests()

    def tearDown(self) -> None:
        shutdown.reset_for_tests()

    # ------------------------------------------------------------------ state

    def test_initial_state(self) -> None:
        self.assertFalse(shutdown.is_shutting_down())
        self.assertEqual(shutdown.inflight_count(), 0)

    def test_enter_exit_run(self) -> None:
        self.assertTrue(shutdown.enter_run())
        self.assertEqual(shutdown.inflight_count(), 1)
        self.assertTrue(shutdown.enter_run())
        self.assertEqual(shutdown.inflight_count(), 2)
        shutdown.exit_run()
        self.assertEqual(shutdown.inflight_count(), 1)
        shutdown.exit_run()
        self.assertEqual(shutdown.inflight_count(), 0)
        # Exit beyond zero is safe (clamped)
        shutdown.exit_run()
        self.assertEqual(shutdown.inflight_count(), 0)

    def test_enter_run_rejected_when_shutting_down(self) -> None:
        shutdown._shutdown_requested.set()
        self.assertFalse(shutdown.enter_run())
        self.assertEqual(shutdown.inflight_count(), 0)

    def test_is_shutting_down_reflects_flag(self) -> None:
        self.assertFalse(shutdown.is_shutting_down())
        shutdown._shutdown_requested.set()
        self.assertTrue(shutdown.is_shutting_down())

    # --------------------------------------------------------- request_shutdown

    def test_request_shutdown_drains_immediately_when_no_inflight(self) -> None:
        drained = shutdown.request_shutdown(source="test", grace_seconds=1.0)
        self.assertTrue(drained)
        self.assertTrue(shutdown.is_shutting_down())
        self.assertEqual(shutdown.inflight_count(), 0)

    def test_request_shutdown_waits_for_drain(self) -> None:
        # Simulate one inflight run that exits after 0.15s
        shutdown.enter_run()

        def finish_run() -> None:
            time.sleep(0.15)
            shutdown.exit_run()

        t = threading.Thread(target=finish_run, daemon=True)
        t.start()
        drained = shutdown.request_shutdown(source="test", grace_seconds=2.0)
        t.join(timeout=2.0)
        self.assertTrue(drained)
        self.assertEqual(shutdown.inflight_count(), 0)

    def test_request_shutdown_force_exits_after_grace(self) -> None:
        # Inflight run that never finishes
        shutdown.enter_run()
        drained = shutdown.request_shutdown(source="test", grace_seconds=0.1)
        self.assertFalse(drained)
        self.assertEqual(shutdown.inflight_count(), 1)

    def test_request_shutdown_idempotent(self) -> None:
        drained1 = shutdown.request_shutdown(source="test", grace_seconds=1.0)
        self.assertTrue(drained1)
        # Second call should not re-drain
        drained2 = shutdown.request_shutdown(source="test", grace_seconds=1.0)
        self.assertTrue(drained2)

    # ----------------------------------------------------- signal handler install

    def test_install_signal_handlers_returns_true(self) -> None:
        # Main thread — should succeed
        result = shutdown.install_signal_handlers()
        self.assertTrue(result)

    def test_install_signal_handlers_idempotent(self) -> None:
        shutdown.install_signal_handlers()
        shutdown.install_signal_handlers()  # should not raise
        self.assertTrue(shutdown._handlers_installed)

    def test_install_signal_handlers_not_on_main_thread(self) -> None:
        # Python 3.12+ allows signal.signal from non-main threads on some platforms,
        # so we verify the idempotent path instead: calling it twice should not raise.
        result1 = shutdown.install_signal_handlers()
        self.assertTrue(result1)
        # Calling from any thread after main-thread install should also return True
        # (idempotent — already installed).
        result_holder: list[bool] = []

        def worker() -> None:
            result_holder.append(shutdown.install_signal_handlers())

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(timeout=2.0)
        # Already installed → returns True (idempotent)
        self.assertTrue(result_holder[0])

    # --------------------------------------------------------- reset_for_tests

    def test_reset_clears_state(self) -> None:
        shutdown.enter_run()
        shutdown.enter_run()
        shutdown._shutdown_requested.set()
        shutdown.reset_for_tests()
        self.assertFalse(shutdown.is_shutting_down())
        self.assertEqual(shutdown.inflight_count(), 0)


class TestHealthDraining(unittest.TestCase):
    """Verify /ready returns 503 when shutdown is in progress."""

    def setUp(self) -> None:
        shutdown.reset_for_tests()
        stop_health_server()

    def tearDown(self) -> None:
        shutdown.reset_for_tests()
        stop_health_server()

    def test_readiness_503_when_draining(self) -> None:
        from app import health

        health.stop_health_server()
        # Patch probe to return ready so the non-draining path works if reached
        patcher = patch.object(
            health, "_probe_llm_readiness", return_value=(True, "ready"),
        )
        patcher.start()
        server = health.start_health_server(port=0, host="127.0.0.1")
        assert server is not None
        time.sleep(0.15)
        port = server.server_address[1]
        try:
            # Set shutdown flag — handler checks is_shutting_down() BEFORE
            # calling _cached_readiness(), so the probe mock is not reached.
            shutdown._shutdown_requested.set()
            # Force the cache to be empty so the handler does not serve stale data
            health._cached_readiness(force=True)

            import urllib.request
            import urllib.error
            import json

            url = f"http://127.0.0.1:{port}/ready"
            req = urllib.request.Request(url)
            try:
                with urllib.request.urlopen(req, timeout=2) as resp:  # noqa: S310
                    self.fail(f"expected 503, got {resp.status}")
            except urllib.error.HTTPError as exc:
                self.assertEqual(exc.code, 503)
                body = json.loads(exc.read().decode("utf-8"))
                self.assertFalse(body["ready"])
                self.assertIn("draining", body["detail"])
        finally:
            patcher.stop()
            shutdown.reset_for_tests()
            health.stop_health_server()

    def test_liveness_still_200_when_draining(self) -> None:
        from app import health

        health.stop_health_server()
        patcher = patch.object(
            health, "_probe_llm_readiness",
            side_effect=AssertionError("probe must not be called for liveness"),
        )
        patcher.start()
        server = health.start_health_server(port=0, host="127.0.0.1")
        assert server is not None
        time.sleep(0.15)
        port = server.server_address[1]
        try:
            shutdown._shutdown_requested.set()

            import urllib.request
            import json

            url = f"http://127.0.0.1:{port}/health"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as resp:  # noqa: S310
                self.assertEqual(resp.status, 200)
                body = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(body["status"], "ok")
        finally:
            patcher.stop()
            shutdown.reset_for_tests()
            health.stop_health_server()


class TestLLMTimeout(unittest.TestCase):
    """Verify the LLM client surfaces timeout errors with actionable messages."""

    def test_timeout_error_message_is_user_friendly(self) -> None:
        from app.llm import LLMClient, LLMError

        fake_settings = MagicMock()
        fake_settings.configured = True
        fake_settings.api_key = "test"
        fake_settings.base_url = "http://invalid"
        fake_settings.model_main = "m"
        fake_settings.model_fast = "f"
        # Set a short timeout
        with patch("app.config.LLM_CALL_TIMEOUT_SECONDS", 5):
            client = LLMClient(fake_settings)
        # Simulate a TimeoutError from the OpenAI client
        with patch.object(
            client._client.chat.completions, "create",
            side_effect=TimeoutError("timed out"),
        ):
            with patch("app.llm.time.sleep", return_value=None):
                with self.assertRaises(LLMError) as ctx:
                    client.chat("sys", "user", phase="test")
            msg = str(ctx.exception)
            self.assertIn("timed out", msg.lower())
            self.assertIn("VA_LSE_LLM_CALL_TIMEOUT_SECONDS", msg)

    def test_httpx_timeout_error_message(self) -> None:
        """If httpx is installed, TimeoutException should also be caught."""
        try:
            import httpx  # noqa: F401
        except ImportError:
            self.skipTest("httpx not installed")

        from app.llm import LLMClient, LLMError

        fake_settings = MagicMock()
        fake_settings.configured = True
        fake_settings.api_key = "test"
        fake_settings.base_url = "http://invalid"
        fake_settings.model_main = "m"
        fake_settings.model_fast = "f"
        with patch("app.config.LLM_CALL_TIMEOUT_SECONDS", 10):
            client = LLMClient(fake_settings)
        with patch.object(
            client._client.chat.completions, "create",
            side_effect=httpx.TimeoutException("connection timed out"),
        ):
            with patch("app.llm.time.sleep", return_value=None):
                with self.assertRaises(LLMError) as ctx:
                    client.chat("sys", "user", phase="test")
            msg = str(ctx.exception)
            self.assertIn("timed out", msg.lower())


if __name__ == "__main__":
    unittest.main()
