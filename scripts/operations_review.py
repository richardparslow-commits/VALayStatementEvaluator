"""Prepare an unapproved, source-bound R12 operational acceptance worksheet offline."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.accuracy_benchmark import BenchmarkInvalid, digest, source_snapshot
from app.pilot import EVIDENCE_FIELDS
from scripts.legal_review import committed_text

EXTRA = (
    "scripts/operations_review.py", "scripts/legal_review.py", "scripts/pilot_status.py",
    "README.md", "PILOT.md", "Dockerfile", "docker-compose.pilot.yml", ".dockerignore",
    ".gitignore", ".streamlit/config.toml", "nginx/pilot.conf", "deploy/parser.Dockerfile",
    "deploy/parser-launcher.Dockerfile", "deploy/parser.apparmor",
    "deploy/pilot-approval.example.json", "deploy/PILOT_OPERATIONS_ACCEPTANCE.md",
    "deploy/PILOT_ACCEPTANCE.md", "deploy/PILOT_PRIVACY_ACCEPTANCE.md",
    "deploy/PILOT_BUDGET_ACCEPTANCE.md", "deploy/PILOT_ACCURACY_ACCEPTANCE.md",
    "deploy/PILOT_LEGAL_ACCEPTANCE.md", "deploy/monitoring/prometheus.yml",
    "deploy/monitoring/alertmanager.yml", "deploy/monitoring/alerts.yml", ".github/workflows/test.yml",
)

# These are required observations, never automatic results. Follow the complete
# procedure, including prerequisites, stop rules and evidence handling.
CHECKS = (
    ("O01", "release", "Verify clean source, locked dependencies, scanned images and effective immutable image IDs."),
    ("O02", "release", "Bind private origin, issuer, roles, exact model versions, effective config and eight accepted evidence references."),
    ("O03", "approval", "Verify protected file/mount/replacement authority and current full-SHA approval with at most 30 days validity."),
    ("O04", "approval", "Missing, expired, ambiguous, linked, writable or revision-mismatched approval blocks actions and provider work."),
    ("O05", "startup", "Cold start without browser traffic initializes cleanup, ledger lease and private health; unavailable dependencies close admission."),
    ("O06", "monitoring", "Check private app/parser health, readiness limitations and refusal of public diagnostic routes."),
    ("O07", "monitoring", "Independent monitor detects process/host failure, restart loops, resource pressure and stale/failed retention."),
    ("O08", "monitoring", "Approved private synthetic alert delivery and acknowledgment execute; missing recipients or monitor failure trigger escalation."),
    ("O09", "privacy", "Count-only diagnostics remain free of synthetic content canaries across every configured sink."),
    ("O10", "privacy", "Idle log sweeps and disconnected/upload cleanup run within reviewed limits without fresh application traffic."),
    ("O11", "privacy", "Stopped-volume and decommission deletion schedule is independently installed and observed, including historical copies."),
    ("O12", "failure", "Parser limits/timeouts/cleanup execute on the actual protected Linux host and exact image."),
    ("O13", "failure", "Approved synthetic provider delay/error exercises stop safely without retries past admission refusal or new destinations."),
    ("O14", "failure", "Cancellation, disconnect, resource pressure and shutdown discard delayed results; document already-sent provider limits."),
    ("O15", "recovery", "Stop-before-start restart/upgrade preserves quota totals, rejects concurrent web instances and avoids automatic recovery admission."),
    ("O16", "incident", "Remove an invitation during work; all open sessions refuse subsequent sensitive boundaries and delayed results."),
    ("O17", "incident", "Stop the entire pilot with the approved bounded procedure; confirm services stopped, restart disabled and no continuing jobs."),
    ("O18", "incident", "Rehearse protected credential revocation/rotation on approved synthetic credentials, including provider and OIDC handling."),
    ("O19", "incident", "Named owners rehearse private notification/escalation, count-only evidence preservation and provider-copy requests."),
    ("O20", "scope", "Document backup/restore exclusion; verify no case or operational-log backup/snapshot path is enabled outside separate approval."),
    ("O21", "recovery", "Recover the approved service without case restore, retain quota history and renew exact-release evidence before admission."),
    ("O22", "signoff", "Authenticate operator/security/privacy decisions; every applicable check has observations and accepted evidence with no unresolved failures."),
)


def prepare_packet(repo: Path) -> dict[str, Any]:
    source = source_snapshot(repo)
    for name in tuple(source["file_sha256"]):
        committed_text(repo, name, source)
    for name in EXTRA:
        committed_text(repo, name, source)
    packet = {
        "schema_version": 1, "purpose": "pilot_operational_acceptance_preparation",
        "status": "unapproved", "pilot_admission": "not_authorized_by_this_tool",
        "source": source,
        "release_template": {
            "reviewed_revision": source["revision"], "source_tree": source["tree"],
            "deployment_url": "", "access_method_reference": "", "platform_reference": "",
            "operator_id": "", "incident_owner_id": "", "backup_contact_id": "", "security_reviewer_id": "",
            "privacy_reviewer_id": "", "incident_procedure_reference": "",
            "alert_route_reference": "", "effective_configuration_sha256": "",
            "image_ids": {service: "" for service in ("web", "parser", "launcher", "proxy")},
            "image_scan_references": {service: "" for service in ("web", "parser", "launcher", "proxy")},
            "issuer": "", "invited_roles_reference": "", "provider_base_url": "",
            "exact_model_versions": [], "provider_account_region_reference": "",
            "quota_policy_sha256": "", "notice_sha256": "", "local_log_retention_days": None,
            "evidence_references": {field: "" for field in EVIDENCE_FIELDS},
            "backup_restore_scope": "excluded_pending_separate_acceptance",
        },
        "checks": [{"id": key, "area": area, "required_observation": requirement,
                    "status": "not_run", "tester_id": "", "tested_at": "",
                    "observations": "", "evidence_reference": "", "retest_history": [],
                    "inapplicability_rationale": "", "reviewer_signature_reference": ""}
                   for key, area, requirement in CHECKS],
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
        print("Unapproved operations worksheet prepared. Actual-host checks and signed acceptance remain required; pilot is NO-GO.")
        return 0
    except (BenchmarkInvalid, KeyError, TypeError, ValueError, OSError, subprocess.CalledProcessError):
        print("Operations worksheet not written: source is incomplete or changed, or output already exists.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
