"""Offline tests for app/health.py sidecar — no network, no Streamlit.

Covers:
  GET /health (liveness, always 200), GET /ready (readiness, 200 vs 503),
  HEAD, 404, cached readiness, and pure helper contracts (payloads, port, probe).
"""
import json
import sys
import time
import unittest
import urllib.error
import urllib.request
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import health  # noqa: E402
from app.config import Settings  # noqa: E402


# ------------------------------------------------------------------ helpers
def _free_port_server(probe_return=None):
    """Start a health server on an ephemeral port; caller must stop it.

    If probe_return is not None, patch health._probe_llm_readiness to return it.
    Returns (server, port, patcher_or_None).
    """
    health.stop_health_server()
    patcher = None
    if probe_return is not None:
        patcher = patch.object(health, "_probe_llm_readiness", return_value=probe_return)
        patcher.start()
    server = health.start_health_server(port=0, host="127.0.0.1")
    assert server is not None, "health server failed to bind on ephemeral port"
    # Give the daemon thread a moment to start listening.
    time.sleep(0.15)
    port = server.server_address[1]
    return server, port, patcher


def _cleanup(patcher):
    if patcher is not None:
        patcher.stop()
    health.stop_health_server()
    # Allow socket to close; avoid Address already in use on next test.
    time.sleep(0.05)


def _get_json(url, timeout=2):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        body = resp.read().decode("utf-8")
        return resp.status, json.loads(body), dict(resp.headers)


def _head_request(host, port, path, timeout=2):
    conn = HTTPConnection(host, port, timeout=timeout)
    conn.request("HEAD", path)
    resp = conn.getresponse()
    status = resp.status
    headers = dict(resp.getheaders())
    body = resp.read()  # HEAD should be empty
    conn.close()
    return status, headers, body


# ------------------------------------------------------------------ pure helpers
class TestHealthPure(unittest.TestCase):
    def test_health_payload_shape(self):
        payload = health._health_payload()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["service"], "va-lay-statement-evaluator")
        self.assertIsInstance(payload["uptime_s"], int)
        self.assertGreaterEqual(payload["uptime_s"], 0)
        self.assertIn("T", payload["timestamp"])
        self.assertTrue(payload["timestamp"].endswith("Z"))

    def test_ready_payload_ready_true(self):
        payload = health._ready_payload(True, "ready")
        self.assertTrue(payload["ready"])
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["detail"], "ready")

    def test_ready_payload_not_ready(self):
        payload = health._ready_payload(False, "missing OPENAI_API_KEY")
        self.assertFalse(payload["ready"])
        self.assertEqual(payload["status"], "not_ready")
        # still carries uptime/service
        self.assertIn("uptime_s", payload)
        self.assertIn("service", payload)

    def test_health_port_env_overrides_config(self):
        with patch.dict("os.environ", {"VA_LSE_HEALTH_PORT": "9009"}, clear=False):
            self.assertEqual(health._health_port(), 9009)

    def test_health_port_ignores_invalid_env_and_falls_back(self):
        # invalid values fall back to config/default 8001
        for bad in ("", "0", "-1", "999999", "not-a-number"):
            with patch.dict("os.environ", {"VA_LSE_HEALTH_PORT": bad}, clear=False):
                port = health._health_port()
                self.assertIsInstance(port, int)
                self.assertGreaterEqual(port, 1)
                self.assertLessEqual(port, 65535)

    def test_probe_missing_api_key_not_ready(self):
        fake = Settings(
            api_key="",
            base_url="https://example.com/v1",
            model_main="m",
            model_fast="f",
            fetch_api_key="",
            fetch_base_url="https://fetchsandbox.com",
            fetch_records_path="/medical_records/{patient_id}",
        )
        with patch("app.config.load_settings", return_value=fake):
            ready, detail = health._probe_llm_readiness()
            self.assertFalse(ready)
            self.assertIn("missing OPENAI_API_KEY", detail)

    def test_probe_missing_base_url_not_ready(self):
        fake = Settings(
            api_key="sk-test",
            base_url="",
            model_main="m",
            model_fast="f",
            fetch_api_key="",
            fetch_base_url="https://fetchsandbox.com",
            fetch_records_path="/medical_records/{patient_id}",
        )
        with patch("app.config.load_settings", return_value=fake):
            ready, detail = health._probe_llm_readiness()
            self.assertFalse(ready)
            self.assertIn("missing base_url", detail)

    def test_probe_success_both_models_listed(self):
        fake = Settings(
            api_key="sk-test",
            base_url="https://example.com/v1",
            model_main="qwen3.7-max",
            model_fast="qwen3.7-flash",
            fetch_api_key="",
            fetch_base_url="https://fetchsandbox.com",
            fetch_records_path="/medical_records/{patient_id}",
        )
        payload = json.dumps({"data": [{"id": "qwen3.7-max"}, {"id": "qwen3.7-flash"}, {"id": "other"}]}).encode()
        fake_resp = MagicMock()
        fake_resp.read.return_value = payload
        fake_resp.__enter__ = lambda _: fake_resp
        fake_resp.__exit__ = lambda *_: False
        with patch("app.config.load_settings", return_value=fake):
            with patch("urllib.request.urlopen", return_value=fake_resp) as mock_open:
                ready, detail = health._probe_llm_readiness()
                self.assertTrue(ready)
                self.assertEqual(detail, "ready")
                req = mock_open.call_args[0][0]
                self.assertTrue(req.full_url.endswith("/models"))
                self.assertIn("Bearer sk-test", req.headers.get("Authorization", ""))

    def test_probe_missing_model_not_ready(self):
        fake = Settings(
            api_key="sk-test",
            base_url="https://example.com/v1",
            model_main="qwen3.7-max",
            model_fast="qwen3.7-flash",
            fetch_api_key="",
            fetch_base_url="https://fetchsandbox.com",
            fetch_records_path="/medical_records/{patient_id}",
        )
        payload = json.dumps({"data": [{"id": "qwen3.7-max"}]}).encode()
        fake_resp = MagicMock()
        fake_resp.read.return_value = payload
        fake_resp.__enter__ = lambda _: fake_resp
        fake_resp.__exit__ = lambda *_: False
        with patch("app.config.load_settings", return_value=fake):
            with patch("urllib.request.urlopen", return_value=fake_resp):
                ready, detail = health._probe_llm_readiness()
                self.assertFalse(ready)
                self.assertIn("Fast model", detail)
                self.assertIn("qwen3.7-flash", detail)

    def test_probe_network_error_not_ready(self):
        fake = Settings(
            api_key="sk-test",
            base_url="https://example.com/v1",
            model_main="m",
            model_fast="f",
            fetch_api_key="",
            fetch_base_url="https://fetchsandbox.com",
            fetch_records_path="/medical_records/{patient_id}",
        )
        with patch("app.config.load_settings", return_value=fake):
            with patch("urllib.request.urlopen", side_effect=Exception("network down")):
                ready, detail = health._probe_llm_readiness()
                self.assertFalse(ready)
                self.assertIn("llm probe failed", detail)

    def test_probe_malformed_json_not_ready(self):
        fake = Settings(
            api_key="sk-test",
            base_url="https://example.com/v1",
            model_main="m",
            model_fast="f",
            fetch_api_key="",
            fetch_base_url="https://fetchsandbox.com",
            fetch_records_path="/medical_records/{patient_id}",
        )
        fake_resp = MagicMock()
        fake_resp.read.return_value = b"not json at all"
        fake_resp.__enter__ = lambda _: fake_resp
        fake_resp.__exit__ = lambda *_: False
        with patch("app.config.load_settings", return_value=fake):
            with patch("urllib.request.urlopen", return_value=fake_resp):
                ready, detail = health._probe_llm_readiness()
                self.assertFalse(ready)

    def test_cached_readiness_uses_ttl(self):
        health.stop_health_server()
        call_count = 0

        def fake_probe():
            nonlocal call_count
            call_count += 1
            return True, "ready"

        with patch.object(health, "_probe_llm_readiness", side_effect=fake_probe):
            first_ready, _ = health._cached_readiness(force=True)
            self.assertTrue(first_ready)
            self.assertEqual(call_count, 1)
            # second call within TTL returns cached without calling probe again
            second_ready, _ = health._cached_readiness()
            self.assertTrue(second_ready)
            self.assertEqual(call_count, 1)
            # force bypasses cache
            third_ready, _ = health._cached_readiness(force=True)
            self.assertTrue(third_ready)
            self.assertEqual(call_count, 2)
        health.stop_health_server()


# ------------------------------------------------------------------ server integration (loopback, no external network)
class TestHealthServer(unittest.TestCase):
    def test_liveness_get_returns_200_json(self):
        server, port, patcher = _free_port_server(probe_return=(True, "ready"))
        try:
            status, payload, headers = _get_json(f"http://127.0.0.1:{port}/health")
            self.assertEqual(status, 200)
            self.assertEqual(payload["status"], "ok")
            self.assertIn("uptime_s", payload)
            self.assertEqual(payload["service"], "va-lay-statement-evaluator")
            self.assertEqual(headers.get("Content-Type"), "application/json; charset=utf-8")
            self.assertEqual(headers.get("Cache-Control"), "no-store")
        finally:
            _cleanup(patcher)

    def test_liveness_head_returns_200_no_body(self):
        server, port, patcher = _free_port_server()
        try:
            status, headers, body = _head_request("127.0.0.1", port, "/health")
            self.assertEqual(status, 200)
            self.assertEqual(body, b"")
            self.assertEqual(headers.get("Content-Type"), "application/json; charset=utf-8")
        finally:
            _cleanup(patcher)

    def test_readiness_when_ready_returns_200(self):
        server, port, patcher = _free_port_server(probe_return=(True, "ready"))
        try:
            status, payload, _ = _get_json(f"http://127.0.0.1:{port}/ready")
            self.assertEqual(status, 200)
            self.assertTrue(payload["ready"])
            self.assertEqual(payload["status"], "ready")
        finally:
            _cleanup(patcher)

    def test_readiness_when_not_ready_returns_503(self):
        server, port, patcher = _free_port_server(probe_return=(False, "missing OPENAI_API_KEY"))
        try:
            # readiness is cached per handler; force a fresh probe for this port
            health._cached_readiness(force=True)
            url = f"http://127.0.0.1:{port}/ready"
            req = urllib.request.Request(url)
            try:
                with urllib.request.urlopen(req, timeout=2) as resp:  # noqa: S310
                    self.fail(f"expected HTTPError 503, got {resp.status}")
            except urllib.error.HTTPError as exc:
                self.assertEqual(exc.code, 503)
                body = json.loads(exc.read().decode("utf-8"))
                self.assertFalse(body["ready"])
                self.assertEqual(body["status"], "not_ready")
                self.assertIn("OPENAI_API_KEY", body["detail"])
        finally:
            _cleanup(patcher)

    def test_readiness_cached_between_requests(self):
        # Within TTL, the second GET must not re-probe.
        call_count = 0

        def counting_probe():
            nonlocal call_count
            call_count += 1
            return True, "ready"

        health.stop_health_server()
        patcher = patch.object(health, "_probe_llm_readiness", side_effect=counting_probe)
        patcher.start()
        server = health.start_health_server(port=0, host="127.0.0.1")
        assert server is not None
        time.sleep(0.15)
        port = server.server_address[1]
        try:
            _get_json(f"http://127.0.0.1:{port}/ready")
            self.assertEqual(call_count, 1)
            _get_json(f"http://127.0.0.1:{port}/ready")
            # still 1 because of TTL cache
            self.assertEqual(call_count, 1)
        finally:
            _cleanup(patcher)

    def test_readiness_head_when_not_ready_returns_503_no_body(self):
        server, port, patcher = _free_port_server(probe_return=(False, "llm probe failed: TimeoutError: "))
        try:
            health._cached_readiness(force=True)
            status, headers, body = _head_request("127.0.0.1", port, "/ready")
            self.assertEqual(status, 503)
            self.assertEqual(body, b"")
            # HEAD still sends Content-Type/Length as GET would
            self.assertEqual(headers.get("Content-Type"), "application/json; charset=utf-8")
        finally:
            _cleanup(patcher)

    def test_unknown_path_returns_404(self):
        server, port, patcher = _free_port_server()
        try:
            url = f"http://127.0.0.1:{port}/unknown"
            req = urllib.request.Request(url)
            try:
                with urllib.request.urlopen(req, timeout=2) as resp:  # noqa: S310
                    self.fail(f"expected 404, got {resp.status}")
            except urllib.error.HTTPError as exc:
                self.assertEqual(exc.code, 404)
                body = json.loads(exc.read().decode("utf-8"))
                self.assertEqual(body["status"], "not_found")
        finally:
            _cleanup(patcher)

    def test_start_health_server_idempotent(self):
        health.stop_health_server()
        patcher = None
        try:
            s1 = health.start_health_server(port=0, host="127.0.0.1")
            assert s1 is not None
            time.sleep(0.05)
            s2 = health.start_health_server(port=0, host="127.0.0.1")
            self.assertIs(s1, s2)
        finally:
            _cleanup(patcher)

    def test_liveness_never_touches_llm_probe(self):
        # GET /health must not call _probe_llm_readiness at all.
        health.stop_health_server()
        patcher = patch.object(health, "_probe_llm_readiness", side_effect=AssertionError("probe must not be called for liveness"))
        patcher.start()
        server = health.start_health_server(port=0, host="127.0.0.1")
        assert server is not None
        time.sleep(0.15)
        port = server.server_address[1]
        try:
            status, _, _ = _get_json(f"http://127.0.0.1:{port}/health")
            self.assertEqual(status, 200)
        finally:
            _cleanup(patcher)


if __name__ == "__main__":
    unittest.main()
