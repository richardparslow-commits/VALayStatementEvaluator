"""End-to-end smoke test against the configured LLM endpoint.

Runs BOTH pathways over the fictional sample data in examples/ and prints a
condensed summary of every pipeline stage. Requires a valid .env.

Run: .venv/bin/python scripts/smoke_test.py [evaluate|draft|all]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_settings  # noqa: E402
from app.documents import extract_document  # noqa: E402
from app.draft import run_draft  # noqa: E402
from app.evaluate import run_evaluation  # noqa: E402
from app.llm import LLMClient  # noqa: E402

EXAMPLES = PROJECT_ROOT / "examples"


def progress(frac: float, msg: str) -> None:
    print(f"  [{frac:5.0%}] {msg}", flush=True)


def load_samples():
    statement = (EXAMPLES / "sample_lay_statement.txt").read_text(encoding="utf-8")
    records_doc = extract_document(
        "sample_medical_records.txt",
        (EXAMPLES / "sample_medical_records.txt").read_bytes(),
    )
    return statement, [records_doc]


def smoke_evaluate(llm: LLMClient) -> None:
    print("\n========== EVALUATE PATHWAY ==========")
    statement, records = load_samples()
    start = time.time()
    result = run_evaluation(llm, statement, records, progress=progress)
    print(f"  completed in {time.time() - start:.0f}s")
    print(f"  condition: {result.claimed_condition}")
    print(f"  writer role: {result.writer_role}")
    print(f"  claims extracted: {len(result.claims)}")
    print(f"  verifications: {len(result.verifications)}")
    verdict_counts: dict[str, int] = {}
    for v in result.verifications:
        verdict_counts[v.get("verdict", "?")] = verdict_counts.get(v.get("verdict", "?"), 0) + 1
    print(f"  verdict breakdown: {verdict_counts}")
    print(f"  scores: {result.scores}")
    print(f"  overall: {result.overall_rating}")
    print(f"  summary: {result.executive_summary[:300]}")
    print(f"  report length: {len(result.report_markdown)} chars")
    out = PROJECT_ROOT / "outputs"
    out.mkdir(exist_ok=True)
    (out / "smoke_evaluation_report.md").write_text(result.report_markdown, encoding="utf-8")
    print(f"  report written to outputs/smoke_evaluation_report.md")


def smoke_draft(llm: LLMClient) -> None:
    print("\n========== DRAFT PATHWAY ==========")
    _, records = load_samples()
    witness = {
        "name": "Mike Doe",
        "relationship": "Coworker / supervisor",
        "known_since": "2012",
        "contact_frequency": "Several times a week at work; occasional social events",
        "veteran_name": "John Q. Sample",
        "witnessed_event": "Yes",
    }
    observations = (
        "- I saw John hurt his back lifting a heavy pallet at Riverside Warehouse around "
        "2014; he cried out and had to sit down.\n"
        "- After that he wore a back brace some days and moved to light duty scanning.\n"
        "- He told me he was getting shots in his back from a doctor.\n"
        "- At a 2022 company picnic he had to stop setting up tables after about ten minutes.\n"
        "- He told me he wakes up from pain at night and his wife helps him with socks.\n"
        "- He quit the warehouse in 2019 because he could not lift anymore.\n"
        "- He stopped coming fishing with us."
    )
    start = time.time()
    result = run_draft(
        llm, records, witness, observations,
        condition="Chronic lumbar strain with radiculopathy (lower back condition)",
        claim_type="Service connection (new claim)",
        progress=progress,
    )
    print(f"  completed in {time.time() - start:.0f}s")
    grounding = result.grounding
    print(f"  supported observations: {len(grounding.get('supported_observations', []))}")
    print(f"  unverified observations: {len(grounding.get('unverified_observations', []))}")
    print(f"  conflicts: {len(grounding.get('conflicts', []))}")
    print(f"  strengthening questions: {len(grounding.get('strengthening_questions', []))}")
    print(f"  review issues fixed: {len(result.review_issues)}")
    print(f"  statement length: {len(result.output_statement)} chars")
    print("\n----- DRAFT PREVIEW (first 1200 chars) -----")
    print(result.output_statement[:1200])
    out = PROJECT_ROOT / "outputs"
    out.mkdir(exist_ok=True)
    (out / "smoke_draft_statement.txt").write_text(result.output_statement, encoding="utf-8")
    print("\n  statement written to outputs/smoke_draft_statement.txt")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    settings = load_settings()
    if not settings.configured:
        print("No API key configured — set .env first.")
        sys.exit(1)
    print(f"Endpoint: {settings.base_url}")
    print(f"Models: main={settings.model_main} fast={settings.model_fast}")
    llm = LLMClient(settings)
    if mode in ("evaluate", "all"):
        smoke_evaluate(llm)
    if mode in ("draft", "all"):
        smoke_draft(llm)
    print("\nSMOKE TEST DONE")


if __name__ == "__main__":
    main()
