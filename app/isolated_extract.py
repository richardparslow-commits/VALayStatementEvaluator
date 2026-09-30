"""Bounded, killable document parser. No credentials are inherited by the child.

Process limits contain parser resource exhaustion; they are not an OS sandbox.
The pilot deployment must additionally use a non-root, restricted container.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .documents import ExtractedDocument, ExtractionError

MAX_OUTPUT_BYTES = 32 * 1024 * 1024


class IsolatedExtractor:
    def __init__(self, timeout: float = 60) -> None:
        self.timeout = timeout

    def extract(self, label: str, data: bytes) -> tuple[list[ExtractedDocument], list[str]]:
        from . import config
        from .job_payload import documents_from_json
        if len(data) > config.MAX_UPLOAD_BYTES:
            raise ExtractionError("Upload exceeds the parser input limit.")
        try:
            with tempfile.TemporaryDirectory(prefix="va-parser-") as scratch:
                root = Path(scratch)
                (root / "input").write_bytes(data)
                request = json.dumps({"label": label, "root": scratch})
                env = {
                    "PATH": os.defpath,
                    "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "VA_LSE_MAX_RECORD_PAGES": str(config.MAX_RECORD_PAGES),
                }
                process = subprocess.run(
                    [sys.executable, "-m", "app.isolated_extract"], input=request,
                    text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    cwd=scratch, env=env, timeout=self.timeout, check=False,
                )
                output = root / "output.json"
                if process.returncode or not output.exists() or output.stat().st_size > MAX_OUTPUT_BYTES:
                    raise ExtractionError("The isolated parser refused this file. No local fallback was used.")
                reply = json.loads(output.read_text(encoding="utf-8"))
                return documents_from_json(reply["documents"]), reply["skipped"]
        except subprocess.TimeoutExpired as exc:
            raise ExtractionError("The isolated parser reached its time limit and was stopped.") from exc
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ExtractionError("The isolated parser failed. No local fallback was used.") from exc


def _child() -> None:
    import resource
    # Imports happen before resource limits; file parsing happens after them.
    from .documents import InProcessExtractor
    from .job_payload import documents_to_json
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    if sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
    request: dict[str, Any] = json.loads(sys.stdin.read(65536))
    root = Path(request["root"])
    docs, skipped = InProcessExtractor().extract(request["label"], (root / "input").read_bytes())
    if sum(len(p.text) for d in docs for p in d.pages) > 20 * 1024 * 1024:
        raise ExtractionError("Extracted text exceeds the parser output limit.")
    encoded = json.dumps({"documents": documents_to_json(docs), "skipped": skipped}).encode()
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise ExtractionError("Serialized extraction exceeds the parser output limit.")
    (root / "output.json").write_bytes(encoded)


if __name__ == "__main__":
    _child()
