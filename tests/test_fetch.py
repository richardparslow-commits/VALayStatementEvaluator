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
    ) -> Settings:
        return Settings(
            api_key="",
            base_url="https://llm.example",
            model_main="main-model",
            model_fast="fast-model",
            fetch_api_key="sandbox-token",
            fetch_base_url=fetch_base_url,
            fetch_records_path=fetch_records_path,
        )

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


if __name__ == "__main__":
    unittest.main()
