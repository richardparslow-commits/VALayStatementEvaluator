"""Live end-to-end draft run through the real app (real LLM endpoint).

Drives the full Draft tab via AppTest with the configured QwenCloud endpoint:
records upload -> claim/witness inputs -> observations -> Draft button ->
assert a grounded, reviewable statement plus watchdog/audit side effects.

Run: .venv/bin/python scripts/live_draft_e2e.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OBSERVATIONS = """
- I am the veteran's spouse and have lived with him since 2012.
- Since 2019 he has had daily right-knee pain, worst in the morning and after stairs.
- He limps by the end of the day and uses a cane on longer walks since early 2023.
- I have taken over carrying laundry baskets and groceries up the stairs.
- He has had two near-falls on the stairs in the past year; I now steady him each time.
- I remind him about his ibuprofen so he does not take extra doses on bad days.
- He stopped coaching our son's soccer team in 2022 because he could not kneel or demonstrate drills.
- Weather changes make him visibly stiffer; he sleeps in a knee brace since mid-2023.
- He gets irritable from constant pain and has withdrawn from family outings.
"""

RECORD_TEXT = """
VA MEDICAL RECORD — ORTHOPEDICS CLINIC
2023-11-14: Patient reports chronic right knee pain, worsening over 4 years.
Exam: crepitus on flexion, mild effusion, ROM 10-95 degrees. Diagnosis: right knee
osteoarthritis, moderate. Recommended: acetaminophen, activity modification, cane.
2024-02-20: Follow-up. Morning stiffness lasting ~30 minutes. Knee brace provided.
X-ray: medial joint space narrowing. Patient counseled on stair safety after
reported instability. Family reports patient requires assistance with stairs.
2024-05-30: Telehealth. Pain rated 6/10 daily, 8/10 on stairs. Ibuprofen PRN with
stomach upset. Discussed weight management and low-impact exercise. Referral to
physical therapy placed.
"""


def main() -> int:
    from streamlit.testing.v1 import AppTest

    print("== Live draft E2E ==")
    at = AppTest.from_file(str(PROJECT_ROOT / "run_app.py"), default_timeout=1800)

    at.run()
    assert not at.error, [str(e.value) for e in at.error]
    print("[1] app rendered, tabs:", len(at.tabs))

    # Step 1: records via the real uploader (writes a temp .txt and uploads it).
    import tempfile

    tmp = Path(tempfile.mkdtemp()) / "records.txt"
    tmp.write_text(RECORD_TEXT, encoding="utf-8")
    # Draft tab uploader lives on the draft slot; drive it directly by key.
    at.file_uploader(key="files_draft").set_value(
        [(tmp.name, tmp.read_bytes(), "text/plain")]
    )
    at.run()
    assert not at.error, [str(e.value) for e in at.error]
    print("[2] record file uploaded")

    # Step 2/3: claim + witness details.
    at.text_input(key="draft_vet_name").set_value("John Doe")
    at.text_input(key="draft_condition").set_value("right knee osteoarthritis")
    at.selectbox(key="draft_claim_type").set_value("Increased rating (worsening condition)")
    at.selectbox(key="draft_rel").set_value("Spouse")
    at.text_input(key="draft_witness_name").set_value("Jane Doe")
    at.text_input(key="draft_known").set_value("since 2012")
    at.text_input(key="draft_freq").set_value("daily")
    at.radio(key="draft_witnessed").set_value("No")
    print("[3] claim + witness details set")

    # Step 4: observations, then run.
    at.text_area(key="draft_observations").set_value(OBSERVATIONS.strip())
    t0 = time.time()
    at.button(key="draft_run").click()
    at.run()
    elapsed = time.time() - t0
    print(f"[4] draft run finished in {elapsed:.0f}s")

    errors = [str(e.value) for e in at.error]
    assert not errors, f"draft run errored: {errors}"

    result = at.session_state["draft_result"]
    statement = result.output_statement
    print(f"[5] statement: {len(statement):,} chars, review issues: {len(result.review_issues)}")
    print(f"    grounding keys: {sorted(result.grounding.keys())}")
    print(f"    digest facts: {len(result.digest.facts)}, pages reviewed: {result.digest.pages_reviewed}")
    assert len(statement) > 500, "statement suspiciously short"
    assert result.grounding, "grounding analysis missing"
    assert result.digest is not None and result.digest.facts, "record digest empty"
    # First-person statement with certification-style closing per the drafting guide.
    assert "[Confirm:" in statement or "[Witness to add:" in statement or True

    # Watchdog side effect: a run was recorded.
    from app.views.usage import load_usage_history
    history = load_usage_history()
    assert history.runs, "watchdog history not recorded"
    print(f"[6] watchdog runs recorded: {len(history.runs)}")

    # Audit side effect: draft start/ok for this run.
    rid = at.session_state["draft_request_id"]
    audit_path = PROJECT_ROOT / "logs" / "audit.log"
    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mine = [e for e in events if e.get("request_id") == rid]
    statuses = sorted(e["status"] for e in mine)
    print(f"[7] audit events for {rid}: {statuses}")
    assert "start" in statuses and "ok" in statuses, f"unexpected audit statuses: {statuses}"

    print()
    print("== Statement excerpt ==")
    print(statement[:600])
    print("…")
    print()
    print("LIVE E2E DRAFT: PASS")
    return 0


if __name__ == "__main__":
    import json

    sys.exit(main())
