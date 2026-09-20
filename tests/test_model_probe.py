"""Unit tests for `app.llm.probe_models` and its failure classification.

The probe feeds the only screen where a user can fix a bad key, base URL, or
model name in seconds, and it used to answer "unreachable, or the key was
rejected" for every possible failure. A 401, a 404, and a host that does not
resolve have three different fixes, so the probe now reports the status and the
response body that distinguish them. These tests pin each branch.
"""

from __future__ import annotations

import io
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app.llm import ModelProbe, check_model_availability, probe_models  # noqa: E402


def _http_error(status: int, body: str = "") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://example.test/v1/models",
        code=status,
        msg=f"HTTP {status}",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(body.encode("utf-8")),
    )


class _Resp(io.BytesIO):
    """Minimal urlopen response: a context manager that reads like a file."""

    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _ok(payload: bytes):
    return patch("urllib.request.urlopen", return_value=_Resp(payload))


class TestProbeModels(unittest.TestCase):
    def test_a_model_list_is_returned_with_status_200(self) -> None:
        payload = b'{"data": [{"id": "perplexity/kimi-k3"}, {"id": "perplexity/glm-5.3-flash"}]}'
        with _ok(payload):
            probe = probe_models("https://api.perplexity.ai/v1", "k")
        self.assertTrue(probe.ok)
        self.assertEqual(probe.status, 200)
        self.assertEqual(probe.models, {"perplexity/kimi-k3", "perplexity/glm-5.3-flash"})
        self.assertEqual(probe.error, "")

    def test_blank_ids_and_junk_rows_are_ignored(self) -> None:
        payload = b'{"data": [{"id": "  "}, {"id": "real-model"}, {"nope": 1}, "string"]}'
        with _ok(payload):
            probe = probe_models("https://example.test/v1", "k")
        self.assertEqual(probe.models, {"real-model"})

    def test_an_empty_catalog_fails_the_check_rather_than_passing_it(self) -> None:
        """A 200 with no models cannot confirm the configured models exist."""
        with _ok(b'{"data": []}'):
            probe = probe_models("https://example.test/v1", "k")
        self.assertFalse(probe.ok)
        self.assertEqual(probe.status, 200)
        self.assertIn("published no model ids", probe.error)

    def test_no_api_key_skips_the_request_entirely(self) -> None:
        with patch("urllib.request.urlopen") as urlopen:
            probe = probe_models("https://example.test/v1", "   ")
        urlopen.assert_not_called()
        self.assertFalse(probe.ok)
        self.assertIsNone(probe.status)
        self.assertIn("no API key", probe.error)

    def test_401_reports_the_status_and_the_response_body(self) -> None:
        """The body is where a provider says *why* it refused the key."""
        with patch("urllib.request.urlopen", side_effect=_http_error(401, "invalid api key")):
            probe = probe_models("https://api.perplexity.ai/v1", "k")
        self.assertFalse(probe.ok)
        self.assertEqual(probe.status, 401)
        self.assertIn("HTTP 401", probe.error)
        self.assertIn("invalid api key", probe.error)

    def test_403_is_kept_distinct_from_401(self) -> None:
        with patch("urllib.request.urlopen", side_effect=_http_error(403, "router access not enabled")):
            probe = probe_models("https://api.perplexity.ai/v1", "k")
        self.assertEqual(probe.status, 403)
        self.assertIn("router access not enabled", probe.error)

    def test_404_is_reported_as_a_404_not_as_an_auth_problem(self) -> None:
        with patch("urllib.request.urlopen", side_effect=_http_error(404, "Not Found")):
            probe = probe_models("https://example.test/v1/wrong-path", "k")
        self.assertEqual(probe.status, 404)

    def test_a_transport_failure_has_no_status_and_names_itself(self) -> None:
        err = urllib.error.URLError("name or service not known")
        with patch("urllib.request.urlopen", side_effect=err):
            probe = probe_models("https://nope.invalid/v1", "k")
        self.assertIsNone(probe.status)
        self.assertIn("URLError", probe.error)
        self.assertIn("name or service not known", probe.error)

    def test_a_non_json_body_does_not_raise(self) -> None:
        with _ok(b"<html>gateway</html>"):
            probe = probe_models("https://example.test/v1", "k")
        self.assertFalse(probe.ok)
        self.assertTrue(probe.error)

    def test_the_probe_never_raises(self) -> None:
        with patch("urllib.request.urlopen", side_effect=RuntimeError("boom")):
            probe = probe_models("https://example.test/v1", "k")
        self.assertFalse(probe.ok)
        self.assertIn("boom", probe.error)

    def test_the_compatibility_wrapper_still_answers_with_models_or_none(self) -> None:
        with _ok(b'{"data": [{"id": "m"}]}'):
            self.assertEqual(check_model_availability("https://example.test/v1", "k"), {"m"})
        with patch("urllib.request.urlopen", side_effect=_http_error(401)):
            self.assertIsNone(check_model_availability("https://example.test/v1", "k"))


class TestModelProbeShape(unittest.TestCase):
    def test_ok_follows_the_model_list_not_the_status(self) -> None:
        self.assertTrue(ModelProbe({"m"}, 200, "").ok)
        self.assertFalse(ModelProbe(None, 200, "no ids").ok)
        self.assertFalse(ModelProbe(None, None, "offline").ok)

    def test_it_is_a_tuple_so_it_cannot_drift_mid_flight(self) -> None:
        probe = ModelProbe({"m"}, 200, "")
        with self.assertRaises(AttributeError):
            probe.status = 500  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
