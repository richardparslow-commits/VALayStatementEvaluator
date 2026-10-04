"""Prepare an unapproved, source-bound R13 accessibility worksheet offline."""
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
from scripts.operations_review import prepare_packet as operations_packet

EXTRA = (
    "app/accessibility.py", "app/views/factual_review.py", "app/views/about_view.py",
    "app/views/draft_view.py", "app/views/evaluate_view.py", "scripts/accessibility_review.py",
    "deploy/PILOT_ACCESSIBILITY_ACCEPTANCE.md",
)

# Observations of the actual release, participant tasks and assistive tools.
# Preparing this worksheet runs none of them and grants no admission authority.
CHECKS = (
    ("A01", "scope", "Bind exact clean release, images, configuration, private host and the participant browser/device/assistive-tool matrix."),
    ("A02", "keyboard", "Complete login, consent, uploads, refusal/errors, run/cancel, navigation, review, copy, clear and logout with keyboard only."),
    ("A03", "focus", "Observe visible focus, logical order, no traps and predictable focus after every relevant rerun, edit, error and cancellation."),
    ("A04", "assistive", "On actual intended screen readers, verify labels, headings, selected/disabled states and complete original passage reading."),
    ("A05", "assistive", "Observe spoken status/error/progress announcements without lost focus, repeated disruptive alerts or missing changes."),
    ("A06", "visual", "Check actual theme text/control/focus contrast, color-independent meaning, 200% text resize and 400% zoom/reflow."),
    ("A07", "mobile", "Complete permitted tasks at narrow viewport and intended mobile/touch settings without hidden controls or inaccessible sources."),
    ("A08", "errors", "Using invented inputs, recover from upload/parse limits, missing pages, incomplete results and blocked review without losing required warnings."),
    ("A09", "comprehension", "Participants explain NOT FOUND, apparent contradiction, support labels, uncertainty and partial results in their own words."),
    ("A10", "comprehension", "Participants resolve invented placeholders/changed facts, preserve attribution and explain exact-text review and edit invalidation."),
    ("A11", "comprehension", "Participants explain approved copying, disabled exports and the limits of clear/logout across originals, copied text and provider retention."),
    ("A12", "signoff", "Preserve failures and retests; independent reviewer and operator authenticate exact-release acceptance for the intended participant scope."),
)


def prepare_packet(repo: Path) -> dict[str, Any]:
    # R12's source binding verifies ALL tracked build/application bytes, including
    # hidden index changes, binary inputs and linked parents. Reuse only its
    # source map; no operational check or approval is carried into R13.
    source = operations_packet(repo)["source"]
    for name in EXTRA:
        committed_text(repo, name, source)
    packet = {
        "schema_version": 1, "purpose": "pilot_accessibility_acceptance_preparation",
        "status": "unapproved", "pilot_admission": "not_authorized_by_this_tool",
        "source": source,
        "release_template": {
            "reviewed_revision": source["revision"], "source_tree": source["tree"],
            "deployment_reference": "", "effective_configuration_sha256": "",
            "image_ids": {service: "" for service in ("web", "parser", "launcher", "proxy")},
            "operator_id": "", "accessibility_tester_id": "", "independent_reviewer_id": "",
            "comprehension_facilitator_id": "", "participant_scope_reference": "",
            "browser_device_assistive_tool_matrix": [], "synthetic_task_set_reference": "",
            "privacy_procedure_reference": "", "excluded_scope": ["file_exports"],
        },
        "checks": [{"id": key, "area": area, "required_observation": requirement,
                    "status": "not_run", "tester_id": "", "tested_at": "",
                    "matrix_row_references": [], "participant_alias_references": [],
                    "observations": "", "evidence_reference": "", "retest_history": [],
                    "inapplicability_rationale": "", "reviewer_signature_reference": ""}
                   for key, area, requirement in CHECKS],
        "decision_template": {"decision": "NO-GO", "approved_at": "", "expires_at": "",
                              "operator_signature_reference": "", "reviewer_signature_reference": "",
                              "accepted_participant_scope_reference": "", "unresolved_findings": []},
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
        print("Unapproved accessibility worksheet prepared. Actual participant and assistive-tool checks remain required; pilot is NO-GO.")
        return 0
    except (BenchmarkInvalid, KeyError, TypeError, ValueError, OSError, subprocess.CalledProcessError):
        print("Accessibility worksheet not written: source is incomplete or changed, or output already exists.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
