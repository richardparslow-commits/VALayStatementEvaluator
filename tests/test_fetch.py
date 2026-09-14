"""Offline unit tests for Fetch Sandbox integration."""
import base64
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.fetch_client import FetchClient, FetchSandboxError  # noqa: E402


class TestFetchClient(unittest.TestCase):
    def _settings(
        self,
        *,
        fetch_base_url: str = "https://demo.fetchsandbox.com",
        fetch_records_path: str = "/medical_records/{patient_id}",
        fetch_max_response_bytes: int = 100 * 1024 * 1024,
    ) -> Settings:
        return Settings(
            api_key="",
            base_url="https://llm.example",
            model_main="main-model",
            model_fast="fast-model",
            fetch_api_key="sandbox-token",
            fetch_base_url=fetch_base_url,
            fetch_records_path=fetch_records_path,
            fetch_max_response_bytes=fetch_max_response_bytes,
        )

    class _FakeResponse:
        def __init__(
            self,
            *,
            status: int = 200,
            reason: str = "OK",
            headers: dict[str, str] | None = None,
            chunks: list[bytes] | None = None,
        ) -> None:
            self.status = status
            self.reason = reason
            self._headers = headers or {}
            self._chunks = list(chunks or [])
            self.read_calls = 0

        def getheaders(self) -> list[tuple[str, str]]:
            return list(self._headers.items())

        def getheader(self, key: str, default: str | None = None) -> str | None:
            return self._headers.get(key, default)

        def read(self, _: int = -1) -> bytes:
            self.read_calls += 1
            if not self._chunks:
                return b""
            return self._chunks.pop(0)

    class _FakeConnection:
        def __init__(self, response: "TestFetchClient._FakeResponse") -> None:
            self._response = response
            self.closed = False
            self.request_calls: list[tuple[str, str, dict[str, str]]] = []

        def request(self, method: str, path: str, headers: dict[str, str]) -> None:
            self.request_calls.append((method, path, headers))

        def getresponse(self) -> "TestFetchClient._FakeResponse":
            return self._response

        def close(self) -> None:
            self.closed = True

    def test_requires_fetch_configuration(self):
        with self.assertRaises(FetchSandboxError):
            FetchClient(self._settings(fetch_base_url="", fetch_records_path=""))

    def test_builds_records_url_from_template(self):
        client = FetchClient(self._settings())
        self.assertEqual(
            client._build_records_url("patient 123"),
            "https://demo.fetchsandbox.com/medical_records/patient%20123",
        )

    def test_appends_query_param_when_path_has_no_placeholder(self):
        client = FetchClient(self._settings(fetch_records_path="/medical_records"))
        self.assertEqual(
            client._build_records_url("abc123"),
            "https://demo.fetchsandbox.com/medical_records?patient_id=abc123",
        )

    def test_rejects_blank_patient_id(self):
        client = FetchClient(self._settings())
        with self.assertRaises(FetchSandboxError):
            client.fetch_documents("   ")

    def test_rejects_non_fetchsandbox_base_url(self):
        client = FetchClient(self._settings(fetch_base_url="https://sandbox.example"))
        with self.assertRaises(FetchSandboxError):
            client._validated_url("/medical_records/test")

    def test_rejects_cross_host_download_urls(self):
        client = FetchClient(self._settings())
        with self.assertRaises(FetchSandboxError):
            client._validated_url("https://other.example/record.pdf")

    def test_normalizes_text_documents(self):
        client = FetchClient(self._settings())
        documents = client._normalize_payload(
            {"documents": [{"name": "visit-note", "text": "Veteran reports daily migraines."}]},
            "pt-1",
        )
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].filename, "visit-note.txt")
        self.assertIn("daily migraines", documents[0].full_text)

    def test_normalizes_base64_documents(self):
        client = FetchClient(self._settings())
        encoded = base64.b64encode(b"Medication changed to sertraline.").decode("ascii")
        documents = client._normalize_payload(
            {
                "documents": [
                    {
                        "name": "medication-update",
                        "content_type": "text/plain",
                        "base64": encoded,
                    }
                ]
            },
            "pt-2",
        )
        self.assertEqual(documents[0].filename, "medication-update.txt")
        self.assertIn("sertraline", documents[0].full_text)

    def test_normalizes_downloaded_documents(self):
        client = FetchClient(self._settings())
        with patch.object(
            client,
            "_download_document",
            return_value=(b"Low back pain worsened after lifting.", "text/plain", "record.txt"),
        ):
            documents = client._normalize_payload(
                {"documents": [{"download_url": "https://files.example/record.txt"}]},
                "pt-3",
            )
        self.assertEqual(documents[0].filename, "record.txt")
        self.assertIn("Low back pain", documents[0].full_text)

    def test_structured_payload_falls_back_to_json_document(self):
        client = FetchClient(self._settings())
        documents = client._normalize_payload(
            {
                "resourceType": "Bundle",
                "entry": [{"resource": {"resourceType": "Observation", "code": {"text": "PTSD"}}}],
            },
            "pt-4",
        )
        self.assertEqual(documents[0].filename, "pt-4_records.json")
        self.assertIn("Observation", documents[0].full_text)
        self.assertIn("PTSD", documents[0].full_text)

    def test_fetch_documents_rejects_empty_results(self):
        client = FetchClient(self._settings())
        with patch.object(client, "_request_json", return_value={"documents": []}):
            with self.assertRaises(FetchSandboxError):
                client.fetch_documents("pt-5")

    def test_http_get_rejects_oversized_content_length_before_read(self):
        client = FetchClient(self._settings(fetch_max_response_bytes=10))
        response = self._FakeResponse(
            headers={"Content-Length": "11"},
            chunks=[b"should-not-be-read"],
        )
        connection = self._FakeConnection(response)
        with patch("app.fetch_client.HTTPSConnection", return_value=connection):
            with self.assertRaises(FetchSandboxError):
                client._http_get("https://demo.fetchsandbox.com/records")
        self.assertEqual(response.read_calls, 0)
        self.assertTrue(connection.closed)

    def test_http_get_rejects_negative_content_length_before_read(self):
        client = FetchClient(self._settings(fetch_max_response_bytes=10))
        response = self._FakeResponse(
            headers={"Content-Length": "-1"},
            chunks=[b"should-not-be-read"],
        )
        connection = self._FakeConnection(response)
        with patch("app.fetch_client.HTTPSConnection", return_value=connection):
            with self.assertRaises(FetchSandboxError):
                client._http_get("https://demo.fetchsandbox.com/records")
        self.assertEqual(response.read_calls, 0)
        self.assertTrue(connection.closed)

    def test_http_get_rejects_oversized_body_during_chunked_read(self):
        client = FetchClient(self._settings(fetch_max_response_bytes=10))
        response = self._FakeResponse(
            headers={"Content-Length": "4"},
            chunks=[b"123456", b"78901"],
        )
        connection = self._FakeConnection(response)
        with patch("app.fetch_client.HTTPSConnection", return_value=connection):
            with self.assertRaises(FetchSandboxError):
                client._http_get("https://demo.fetchsandbox.com/records")
        self.assertTrue(connection.closed)

    def test_http_get_returns_normal_sized_response(self):
        client = FetchClient(self._settings(fetch_max_response_bytes=10))
        response = self._FakeResponse(
            headers={"Content-Type": "application/json"},
            chunks=[b"{", b"}"],
        )
        connection = self._FakeConnection(response)
        with patch("app.fetch_client.HTTPSConnection", return_value=connection):
            data, headers, reason = client._http_get("https://demo.fetchsandbox.com/records")
        self.assertEqual(data, b"{}")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(reason, "OK")
        self.assertTrue(connection.closed)


if __name__ == "__main__":
    unittest.main()
