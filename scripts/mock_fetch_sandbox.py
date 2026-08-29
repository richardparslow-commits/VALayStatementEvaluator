"""A stdlib-only mock "Fetch Sandbox" endpoint for local testing.

The app's FetchClient only accepts base URLs on ``fetchsandbox.com`` (or a
subdomain) and keeps any ``download_url`` on that same host. So to test against
this local mock you must give localhost a fetchsandbox.com hostname:

    1. Add this line to /etc/hosts (requires sudo):
           127.0.0.1  local.fetchsandbox.com

    2. Run this server:
           .venv/bin/python scripts/mock_fetch_sandbox.py

    3. In the app sidebar, set:
           Fetch base URL:  http://local.fetchsandbox.com:8001
           Fetch records path: /medical_records/{patient_id}
           Fetch API key:    (optional; any value works, e.g. "local-key")

    4. In the "Fetch Sandbox" record source, enter any patient or record ID
       and click "Import medical records from Fetch Sandbox".

Run ``--help`` for options.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, unquote, urlparse


def build_records_payload(patient_id: str) -> dict:
    """Build a plausible JSON response shape for this patient."""
    text_note = (
        f"PATIENT {patient_id or '(unknown)'}\n"
        "Chief complaint: chronic left knee pain since service.\n"
        "Observed: antalgic gait, limping on the left during examination.\n"
        "Assessment: DJD of the left knee; prescribed a knee brace.\n"
    )
    header = (
        "VA C-FILE SAMPLE\n"
        "Volunteer veteran records for local Fetch Sandbox testing.\n\n"
    )
    return {
        "documents": [
            {
                "filename": "knee_clinic_2023.txt",
                "content_type": "text/plain",
                "text": header + text_note,
            },
            {
                "filename": "medication_list.txt",
                "content_type": "text/plain",
                "text": header + "Medications: meloxicam 15mg daily PRN for knee pain; KT tape as needed.\n",
            },
        ]
    }


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


class MockFetchHandler(BaseHTTPRequestHandler):
    server_version = "MockFetchSandbox/0.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        query = parse_qs(parsed.query)

        if path == "/health":
            self._respond(200, {"status": "ok"})
            return

        # Support both URL styles the app can emit:
        #   /medical_records/{patient_id}
        #   /medical_records?patient_id=...
        if path == "/medical_records" or path.startswith("/medical_records/"):
            patient_id = (
                unquote(path[len("/medical_records/"):])
                if path.startswith("/medical_records/")
                else (query.get("patient_id", [""])[0] or "")
            )
            self._respond(200, build_records_payload(patient_id))
            return

        self._respond(404, {"error": f"unknown path: {self.path}"})

    def _respond(self, status: int, payload: dict) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # The app sends both a Bearer token and X-API-Key. We don't validate
        # auth here (local test helper), but the app will still send them.
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: A003
        print(f"[mock-fetch] {self.address_string()} - {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), MockFetchHandler)
    print(
        f"[mock-fetch] serving on http://{args.host}:{args.port} "
        "(map local.fetchsandbox.com -> 127.0.0.1 in /etc/hosts and use "
        "http://local.fetchsandbox.com:8001 as the app's Fetch base URL)"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[mock-fetch] shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()