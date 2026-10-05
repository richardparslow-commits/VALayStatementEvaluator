"""PID 1 supervisor for a single document in the dedicated parser container."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from .parser_protocol import DEADLINE, MAX_OUTPUT, MAX_TEXT, MAX_PAGES, encode, read_request


def extraction_page_limit(request: dict[str, int]) -> int:
    maximum = int(os.environ.get("VA_LSE_PARSER_MAX_PAGES", "500"))
    if not 0 < maximum <= MAX_PAGES:
        raise ValueError("Invalid parser deployment page limit.")
    return min(maximum, request["page_limit"])


def child(root: Path) -> None:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT, MAX_OUTPUT))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))
    # Settings imports can only see the parser image and the deliberately empty environment.
    from .documents import DOCUMENT_SCHEMA_VERSION, InProcessExtractor
    from .parser_protocol import decode
    request = decode((root / "header").read_bytes())
    docs, skipped = InProcessExtractor().extract(request["label"], (root / "input").read_bytes())
    if sum(len(p.text) for d in docs for p in d.pages) > MAX_TEXT:
        raise ValueError("Parser text exceeds its limit.")
    documents = [{"filename": d.filename, "schema_version": DOCUMENT_SCHEMA_VERSION, "total_pages": d.total_pages,
                  "unreadable_pages": d.unreadable_pages, "pagination": d.pagination,
                  "coverage_known": d.coverage_known,
                  "source_sha256": d.source_sha256, "extraction_method": d.extraction_method,
                  "text_encoding": d.text_encoding,
                  "pages": [{"page": p.page, "kind": p.kind, "text": p.text,
                             "source_part": p.source_part, "source_start": p.source_start,
                             "source_end": p.source_end} for p in d.pages]}
                 for d in docs]
    result = encode({"version": 1, "nonce": request["nonce"], "sha256": request["sha256"],
                     "label": request["label"], "documents": documents, "skipped": skipped})
    if len(result) > MAX_OUTPUT - 1024:
        raise ValueError("Parser output exceeds its limit.")
    (root / "output").write_bytes(result)


def supervise() -> None:
    # The CLI launcher supplies input before the worker is started. The launcher
    # enforces its own transfer+execution deadline. This supervisor limits the
    # parsing child independently even when the launcher dies.
    # Explicit handler matters for PID 1, which ignores default termination signals.
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.alarm(DEADLINE + 10)
    request, data = read_request(sys.stdin.buffer)
    with tempfile.TemporaryDirectory(prefix="job-", dir="/tmp") as directory:
        root = Path(directory)
        (root / "header").write_bytes(encode(request))
        (root / "input").write_bytes(data)
        subprocess.run([sys.executable, "-m", "app.parser_worker", "--child", directory],
                       env={"PATH": os.defpath, "PYTHONPATH": "/app", "PYTHONDONTWRITEBYTECODE": "1",
                            "VA_LSE_MAX_RECORD_PAGES": str(extraction_page_limit(request))},
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=DEADLINE, check=True)
        output = root / "output"
        if output.is_symlink() or not output.is_file() or output.stat().st_size > MAX_OUTPUT - 1024:
            raise ValueError("Invalid parser output.")
        with output.open("rb") as stream:
            result = stream.read(MAX_OUTPUT + 1)
        if len(result) > MAX_OUTPUT - 1024:
            raise ValueError("Parser output exceeds its limit.")
        sys.stdout.buffer.write(result)
        sys.stdout.buffer.flush()
    # Exit of container PID 1 kills every descendant in its private PID namespace.


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child(Path(sys.argv[2]))
    else:
        supervise()
