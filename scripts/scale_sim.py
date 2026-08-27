"""Offline scale simulation: 2,000 synthetic pages through the review pipeline.

Uses the deterministic FakeLLM (no network) to verify orchestration overhead:
chunking, page dedup, parallel dispatch, hierarchical merge, targeted retrieval.

Run from project root: .venv/bin/python scripts/scale_sim.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from app.documents import DocumentPage, ExtractedDocument  # noqa: E402
from app.medical_review import review_medical_records  # noqa: E402
from test_core import FakeLLM  # noqa: E402


def main() -> None:
    pages: list[DocumentPage] = []
    for i in range(2000):
        body = f"EVT symptom assessment entry number {i} recorded during visit.\n\n"
        body += (
            "Clinical notes describe follow-up care, medication adjustment and "
            "provider observations relevant to the condition under review. "
        ) * 20
        if i % 10 == 7 and i >= 10:  # repeat an earlier page verbatim
            pages.append(pages[i - 10])
        else:
            pages.append(DocumentPage("big.pdf", i + 1, body))
    doc = ExtractedDocument(filename="big.pdf", pages=pages)

    llm = FakeLLM()
    start = time.time()
    digest = review_medical_records(llm, [doc])
    elapsed = time.time() - start

    print(f"pages_reviewed={digest.pages_reviewed:,}")
    print(f"duplicates_skipped={digest.duplicates_skipped:,}")
    print(f"chunks_reviewed={digest.chunks_reviewed:,}")
    print(f"digest_calls={llm.digest_calls:,} merge_calls={llm.merge_calls:,}")
    print(f"final_facts={len(digest.facts):,}")
    sample = digest.relevant_facts_text(
        "symptom assessment entry number 1500", max_facts=3
    )
    print("relevant-fact retrieval sample:")
    for line in sample.splitlines()[:4]:
        print("  " + line[:160])
    print(
        f"elapsed={elapsed:.2f}s with fake LLM; real runtime is API-latency-bound "
        "but fully parallelized across chunks."
    )
    assert digest.duplicates_skipped == 199, digest.duplicates_skipped
    assert "entry number 1500" in sample, "specific entry not retrieved"
    print("SCALE_SIM_OK")


if __name__ == "__main__":
    main()