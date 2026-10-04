"""Prepare an unapproved source-bound R16 case-deletion design/acceptance worksheet offline."""
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

EXTRA = (*OPERATIONS_INPUTS, "scripts/deletion_review.py", "deploy/PILOT_DELETION_ACCEPTANCE.md",
         "app/storage_policy.py", "app/blob_store.py", "app/blob_cleanup.py", "app/pilot.py",
         "app/pilot_asgi.py", "app/text_exports.py", "app/job_queue.py", "app/queue_policy.py",
         "app/views/job_runner.py", "app/worker.py", "app/audit_backup.py", "app/audit_restore.py",
         "app/log_retention.py", "scripts/backup_audit_logs.py", "scripts/restore_audit_logs.py",
         "tests/test_pilot_storage.py", "tests/test_deletion_review.py", "tests/test_blob_store.py",
         "tests/test_blob_cleanup.py", "tests/test_blob_queue_retention.py", "DEPLOYMENT.md",
         "deploy/PILOT_QUEUE_ACCEPTANCE.md", "deploy/PILOT_EXPORT_ACCEPTANCE.md",
         "deploy/k8s/k8s-blobs.yaml", "deploy/k8s/k8s-blob-cleanup.yaml", "deploy/k8s/k8s-audit-backup.yaml",
         "requirements.lock", "requirements-parser.lock", "requirements-s3.txt", "requirements-backup.txt", "run_app.py")

CHECKS = (
    ("D01", "design", "Review separate authenticated durable case/deletion implementation; present release keeps non-synthetic durable blobs excluded."),
    ("D02", "prerequisites", "Accept exact-release R07–R13 and expanded R08 storage/privacy; evaluate optional R14/R15 only for separately enabled capabilities."),
    ("D03", "inventory", "Map every owner/case/revision input, derivative, result, reference, lease, export, temporary file, version, replica and retained copy to an authoritative inventory."),
    ("D04", "authorization", "Deny foreign/anonymous/stale delete, status and recovery requests even with known references; enforce roles, expiry and request authenticity."),
    ("D05", "privacy", "Verify case-scoped protected encryption/key authority, private storage roles, exact destinations and no unreviewed data-bearing diagnostic sinks."),
    ("D06", "revocation", "Commit durable access tombstone and revoke grants/leases before physical deletion; race retrieval, new writes, retries and delayed outputs."),
    ("D07", "shared_objects", "Delete case A without deleting authorized case B's shared data; race reference attach/detach and duplicate submissions against cleanup."),
    ("D08", "filesystem", "Actual shared volume demonstrates cross-process locks, namespace protection, atomic replacement and interruption/retry behavior; permission/link/lock failures remain pending."),
    ("D09", "s3", "Bind bucket/versioning/ownership/prefix and provider semantics; legacy deletion refuses Enabled/Suspended/unknown states or unsupported versioning APIs."),
    ("D10", "s3", "Future version-aware implementation inventories and deletes exact version IDs/markers without deleting adjacent keys or another case; reconcile every partial/denied/unknown result."),
    ("D11", "retained_copies", "Account for object locks, legal holds, replicas, noncurrent versions, multipart remnants and accepted delayed expiry; do not bypass retention locks."),
    ("D12", "retry", "Deletion is authenticated, durable and retryable; failed/unknown storage responses never mark completion, and lost acknowledgment/restart preserves pending state."),
    ("D13", "retention", "Independent idle and stopped-service retention runs within accepted limits; missing mount/inventory/schedule or pressure/permission failure alerts and closes admission."),
    ("D14", "backup", "Accept explicit backup/snapshot inventory, expiry deadlines and protected destruction authority; exclude unreviewed copies."),
    ("D15", "restore", "Restore old snapshots/backups into isolated synthetic fixtures; authoritative tombstones prevent deleted-case access, replay and delayed write resurrection."),
    ("D16", "active_paths", "Invented deleted case is inaccessible through every application-managed input/result/cache/blob/queue/export/recovery path within accepted deadlines."),
    ("D17", "providers", "Document actual provider/user-held copies and accepted deletion/retention request process; local deletion does not recall sent requests or saved files."),
    ("D18", "operations", "Named owners rehearse stopped/failed cleanup, pending/overdue deletion alerts, acknowledgment and count-only private incident evidence on the actual host."),
    ("D19", "participants", "Intended participants can request/check deletion accessibly and accurately explain session clearing, pending/complete/retained states and external-copy limits."),
    ("D20", "signoff", "Bind immutable release/images/config/storage policy and independent operator/security/privacy acceptance; unresolved inventory/access/lifecycle failures prohibit activation."),
)


def prepare_packet(repo: Path) -> dict[str, Any]:
    source = operations_packet(repo)["source"]
    for name in EXTRA:
        committed_text(repo, name, source)
    packet = {
        "schema_version": 1, "purpose": "pilot_case_deletion_design_acceptance_preparation",
        "status": "unapproved", "pilot_admission": "not_authorized_by_this_tool",
        "durable_storage_activation": "not_implemented_or_authorized_by_this_tool",
        "case_deletion_execution": "not_performed_by_this_tool",
        "current_durable_storage_scope": "synthetic_only", "source": source,
        "design_template": {
            "reviewed_revision": source["revision"], "source_tree": source["tree"],
            "separate_implementation_reference": "", "private_host_reference": "",
            "operator_id": "", "incident_owner_id": "", "security_reviewer_id": "", "privacy_reviewer_id": "",
            "owner_case_inventory_authority_reference": "", "durable_tombstone_revocation_reference": "",
            "encryption_key_authority_reference": "", "least_privilege_storage_roles_reference": "",
            "shared_reference_transaction_reference": "", "version_replica_multipart_inventory_reference": "",
            "retention_lock_and_hold_handling_reference": "", "retry_pending_state_reference": "",
            "idle_stopped_cleanup_schedule_reference": "", "backup_expiry_restore_reference": "",
            "provider_external_copy_procedure_reference": "", "participant_notice_sha256": "",
            "effective_configuration_sha256": "", "storage_destination_references": [],
            "access_revocation_deadline_seconds": None, "active_copy_deletion_deadline_seconds": None,
            "backup_copy_expiry_deadline_seconds": None,
            "accepted_prerequisite_references": {key: "" for key in
                ("R07", "R08_expanded_storage_privacy", "R09", "R10", "R11", "R12", "R13",
                 "R14_if_downloads_enabled", "R15_if_queues_enabled")},
            "immutable_image_ids": {role: "" for role in ("web", "storage", "cleanup", "restore")},
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
        print("Unapproved deletion worksheet prepared. Separate implementation and actual acceptance remain required; pilot is NO-GO.")
        return 0
    except (BenchmarkInvalid, KeyError, TypeError, ValueError, OSError, subprocess.CalledProcessError):
        print("Deletion worksheet not written: source is incomplete or changed, or output already exists.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
