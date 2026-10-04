"""Prepare an unapproved, source-bound R15 queue design/acceptance worksheet offline."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.accuracy_benchmark import BenchmarkInvalid, digest
from scripts.legal_review import committed_text
from scripts.operations_review import EXTRA as OPERATIONS_INPUTS, prepare_packet as operations_packet

EXTRA = (*OPERATIONS_INPUTS, "scripts/queue_review.py", "deploy/PILOT_QUEUE_ACCEPTANCE.md",
         "app/queue_policy.py", "app/job_queue.py", "app/job_payload.py", "app/worker.py",
         "app/views/job_runner.py", "app/health.py", "app/blob_store.py", "app/pilot.py",
         "app/pilot_budget.py", "tests/test_pilot_queue.py", "tests/test_queue_review.py",
         "tests/test_job_queue.py", "tests/test_job_queue_atomic.py", "tests/test_job_submission.py",
         "tests/test_worker.py", "tests/test_job_runner.py", "DEPLOYMENT.md", "requirements.lock",
         "requirements-parser.lock", "run_app.py")

CHECKS = (
    ("Q01", "design", "Review a separately implemented queue profile; present release rejects every non-synthetic queue operation."),
    ("Q02", "prerequisites", "Accept current-release R07–R13 evidence plus expanded R08/R09 and durable case deletion R16 before queue admission."),
    ("Q03", "identity", "Prove authenticated tenant/case/revision binding at submission, status, recovery, cancellation and result access; references alone grant nothing."),
    ("Q04", "consent", "Test server-authoritative expiring/revocable case consent at claim, each provider attempt/retry and output; revoke while workers are delayed/disconnected."),
    ("Q05", "destinations", "Bind and recheck exact provider/model versions, notice, release and destination; deny altered/expired approvals before sending case text."),
    ("Q06", "storage", "Verify private TLS transport, least-privilege queue roles, protected credentials and denial of public/cross-role access; no fallback backend."),
    ("Q07", "capacity", "Measure accepted no-eviction capacity, payload/record/result bounds, TTLs and backpressure with realistic synthetic loads."),
    ("Q08", "persistence", "Crash/restart the actual Redis/host and verify accepted fsync, recovery RPO/RTO, expiry and lost/ambiguous submission behavior."),
    ("Q09", "integrity", "Reject substituted/truncated/missing-hash/version/kind/reference inputs before record retrieval and mismatched results before hydration."),
    ("Q10", "billing", "Prove shared durable start/active-run/attempt reservations across simultaneous web/worker hosts, restarts and partitions with no independent spending allowances."),
    ("Q11", "billing", "Accept tested provider worst-case price and independent cutoff; uncertain/retried paid attempts retain charge reservations before recovery."),
    ("Q12", "retries", "Exercise duplicate submissions, lost acknowledgments, provider timeout/unknown charge and supported provider idempotency without assuming exactly-once execution."),
    ("Q13", "fencing", "Kill/reclaim workers and race heartbeats/sweeps; expired owners cannot write progress/results/completion/failure or reclaim access."),
    ("Q14", "cancellation", "Cancel/revoke/delete during provider delays; subsequent attempts stop and late output is withheld; document already-sent request limits."),
    ("Q15", "lifecycle", "Verify per-case input/result/reference/lease/blob/version inventory and independent idle expiry/cleanup; missing inventory or failed storage closes admission."),
    ("Q16", "deletion", "Test case-scoped deletion, shared-object isolation, backup expiry and restore tombstones; every active/recovery path stays denied and another case remains intact."),
    ("Q17", "privacy", "Search every accepted log/trace/error/health/storage/backup sink for synthetic canaries; diagnostics and alerts disclose reviewed counts only."),
    ("Q18", "incident", "Named operators rehearse queue/worker stop, revocation, credential response, capacity/retention/billing alerts and acknowledgment on the actual private host."),
    ("Q19", "participants", "Verify keyboard/assistive-tool access and comprehension of background work, cancellation, expiry, reference access and uncertain provider charges."),
    ("Q20", "signoff", "Bind immutable source/images/config, observations and independent operator/security/privacy decisions; resolve failures before any activation."),
)


def prepare_packet(repo: Path) -> dict[str, Any]:
    # Reuse the byte-for-byte inventory of ALL tracked release inputs, including
    # binary assets, hidden-index changes and leaf/parent links. Import no clients.
    source = operations_packet(repo)["source"]
    for name in EXTRA:
        committed_text(repo, name, source)
    packet = {
        "schema_version": 1, "purpose": "pilot_queue_design_acceptance_preparation",
        "status": "unapproved", "pilot_admission": "not_authorized_by_this_tool",
        "queue_activation": "not_implemented_or_authorized_by_this_tool",
        "current_queue_scope": "synthetic_only", "source": source,
        "design_template": {
            "reviewed_revision": source["revision"], "source_tree": source["tree"],
            "separate_implementation_reference": "", "private_host_reference": "",
            "operator_id": "", "incident_owner_id": "", "security_reviewer_id": "",
            "privacy_reviewer_id": "", "authenticated_owner_case_authority_reference": "",
            "durable_consent_revocation_reference": "", "shared_budget_authority_reference": "",
            "provider_cutoff_and_price_acceptance_reference": "", "provider_idempotency_reference": "",
            "provider_base_url": "", "exact_model_versions": [], "notice_sha256": "",
            "effective_configuration_sha256": "", "least_privilege_roles_reference": "",
            "no_eviction_capacity_reference": "", "payload_result_limits_reference": "",
            "retention_cleanup_inventory_reference": "", "case_deletion_restore_reference": "",
            "queue_recovery_rpo_seconds": None, "queue_recovery_rto_seconds": None,
            "accepted_prerequisite_references": {key: "" for key in
                ("R07", "R08_expanded_queue_privacy", "R09_shared_budget", "R10", "R11", "R12", "R13", "R16")},
            "immutable_image_ids": {role: "" for role in ("web", "worker", "redis", "proxy", "cleanup")},
        },
        "checks": [{"id": key, "area": area, "required_observation": requirement,
                    "status": "not_run", "tester_id": "", "tested_at": "", "observations": "",
                    "evidence_reference": "", "retest_history": [], "inapplicability_rationale": "",
                    "reviewer_signature_reference": ""} for key, area, requirement in CHECKS],
        "decision_template": {"decision": "NO-GO", "approved_at": "", "expires_at": "",
                              "operator_signature_reference": "", "security_signature_reference": "",
                              "privacy_signature_reference": "", "unresolved_findings": []},
    }
    return {**packet, "packet_sha256": digest(packet)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        packet = prepare_packet(args.repo)
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(packet, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print("Unapproved queue worksheet prepared. Separate implementation and actual acceptance remain required; pilot is NO-GO.")
        return 0
    except (BenchmarkInvalid, KeyError, TypeError, ValueError, OSError, subprocess.CalledProcessError):
        print("Queue worksheet not written: source is incomplete or changed, or output already exists.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
