"""Freeze an offline, unapproved legal/evidence review packet. No admission authority."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.accuracy_benchmark import BenchmarkInvalid, digest, read_json, require, source_snapshot

KNOWLEDGE = tuple("app/knowledge/" + name for name in (
    "legal_framework.md", "evaluation_rubric.md", "drafting_guide.md", "topic_checklist.md"))
REGISTER = "review/legal-authority-register-v1.json"
SCENARIOS = "review/legal-scenarios-v1.json"
EXTRA = (REGISTER, SCENARIOS, "scripts/legal_review.py", "README.md", "PILOT.md",
         "deploy/PILOT_LEGAL_ACCEPTANCE.md", "scripts/batch_draft.py")
POLICIES = tuple("app/" + name for name in (
    "draft.py", "evaluate.py", "medical_review.py", "source_validation.py", "documents.py",
    "rubric_validation.py", "evaluation_topics.py", "condition_selector.py", "condition_topics.json",
    "aa_intake.py", "research_questions.py", "pdf_export.py", "exporter.py", "factual_integrity.py",
    "grounding_sources.py", "knowledge_currency.py", "pilot.py", "views/factual_review.py",
    "views/draft_view.py", "views/evaluate_view.py", "views/about_view.py", "views/shared.py"))
PRIMARY_HOSTS = {"www.ecfr.gov", "uscode.house.gov", "www.govinfo.gov", "www.va.gov",
                 "www.vba.va.gov", "www.benefits.va.gov", "www.uscourts.cavc.gov",
                 "uscourts.cavc.gov", "www.cafc.uscourts.gov"}


def committed_text(repo: Path, name: str, source: dict[str, Any]) -> str:
    """Read only fixed/tracked source paths; hidden working changes are refused."""
    require(name == Path(name).as_posix() and not Path(name).is_absolute()
            and ".." not in Path(name).parts, "Invalid source path.")
    path = repo / name
    require(path.is_file() and not any(repo.joinpath(*Path(name).parts[:i]).is_symlink()
                                     for i in range(1, len(Path(name).parts) + 1)),
            "Source must be a regular unlinked file.")
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:{name}"],
                                  stderr=subprocess.PIPE)
    require(path.read_bytes() == raw, "Working source differs from the committed review revision.")
    source["file_sha256"][name] = hashlib.sha256(raw).hexdigest()
    return raw.decode("utf-8")


def knowledge_units(name: str, content: str) -> list[dict[str, Any]]:
    """Partition every byte, including preambles, rubric dimensions and style rules."""
    lines = content.splitlines(keepends=True)
    starts = {0, *(i for i, line in enumerate(lines)
                   if re.match(r"^#{1,6} |^[1-9]\d*\. \*\*", line))}
    bounds = sorted(starts) + [len(lines)]
    units = []
    for start, end in zip(bounds, bounds[1:]):
        if end == start:
            continue
        excerpt = "".join(lines[start:end])
        units.append({"id": f"knowledge:{name}:{start + 1}", "kind": "knowledge",
                      "path": name, "start_line": start + 1, "end_line": end,
                      "text": excerpt, "sha256": hashlib.sha256(excerpt.encode()).hexdigest()})
    return units


def prompt_units(name: str, content: str) -> list[dict[str, Any]]:
    """Highlight prompt construction without importing or evaluating application code.

    This is a reading aid, not complete data-flow analysis. Full policy modules
    are also required review units, covering helpers and inline constructions.
    """
    lines = content.splitlines(keepends=True)
    units = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        syntax = ast.parse(content)
    for node in ast.walk(syntax):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [t.id for t in targets if isinstance(t, ast.Name)
                 and re.search(r"(?:^|_)(?:SYSTEM|USER|PROMPT)(?:_|$)", t.id.upper())]
        if not names:
            continue
        excerpt = "".join(lines[node.lineno - 1:node.end_lineno])
        units.append({"id": f"prompt:{name}:{node.lineno}", "kind": "prompt_construction",
                      "path": name, "start_line": node.lineno, "end_line": node.end_lineno,
                      "names": names, "text": excerpt,
                      "sha256": hashlib.sha256(excerpt.encode()).hexdigest()})
    return sorted(units, key=lambda unit: unit["start_line"])


def validate_seed(register: dict[str, Any], scenarios: dict[str, Any],
                  source: dict[str, Any]) -> None:
    for value in (register, scenarios):
        require(value.get("schema_version") == 1 and value.get("status") == "unreviewed_seed",
                "Only unreviewed version-one preparation seeds are supported.")
    authorities = register.get("authorities")
    controls = register.get("controls")
    require(isinstance(authorities, list) and bool(authorities)
            and isinstance(controls, list) and bool(controls), "Authority register is incomplete.")
    authority_ids: set[str] = set()
    for row in authorities:
        require(isinstance(row, dict), "Authority entry must be an object.")
        require(isinstance(row.get("id"), str) and bool(row["id"].strip())
                and row["id"] not in authority_ids, "Missing or duplicate authority ID.")
        authority_ids.add(row["id"])
        require(isinstance(row.get("url"), str), "Authority URL must be text.")
        url = urlsplit(row["url"])
        require(url.scheme == "https" and url.hostname in PRIMARY_HOSTS
                and not url.username and not url.password and not url.fragment,
                "Candidate authorities must use a primary public HTTPS source.")
        require(row.get("interpretation_status") == "unverified"
                and row.get("applicability_status") == "unverified",
                "Preparation cannot claim a legal interpretation or applicability approval.")
    control_ids: set[str] = set()
    covered: set[str] = set()
    for row in controls:
        require(isinstance(row, dict), "Control entry must be an object.")
        require(row.get("status") == "unverified", "Preparation controls must remain unverified.")
        require(isinstance(row.get("id"), str) and bool(row["id"].strip())
                and row["id"] not in control_ids, "Missing or duplicate control ID.")
        control_ids.add(row["id"])
        for field in ("authority_ids", "paths", "review_questions"):
            require(isinstance(row.get(field), list) and bool(row[field])
                    and all(isinstance(v, str) and bool(v.strip()) for v in row[field]),
                    "Control needs candidate authorities, source paths and review questions.")
        require(set(row["authority_ids"]).issubset(authority_ids), "Unknown candidate authority.")
        require(set(row["paths"]).issubset(source["file_sha256"]), "Control refers to unbound source.")
        covered.update(row["paths"])
    require(set(KNOWLEDGE).issubset(covered), "All four knowledge files require review coverage.")
    require(set(POLICIES).union({"README.md", "PILOT.md", "scripts/batch_draft.py"}).issubset(covered)
            and control_ids == {f"C{i:02}" for i in range(1, 15)},
            "Version-one controls and consequential policy files require complete review coverage.")
    cases = scenarios.get("cases")
    require(scenarios.get("synthetic") is True and isinstance(cases, list) and bool(cases),
            "Only invented review scenarios are supported.")
    case_ids: set[str] = set()
    exercised: set[str] = set()
    for case in cases:
        require(isinstance(case, dict), "Scenario must be an object.")
        require(case.get("status") == "not_run", "Preparation scenarios must remain not run.")
        require(isinstance(case.get("id"), str) and bool(case["id"].strip())
                and case["id"] not in case_ids, "Missing or duplicate scenario ID.")
        case_ids.add(case["id"])
        require(case.get("control_id") in control_ids, "Unknown scenario control.")
        for field in ("applicable_example", "inapplicable_example", "review_question"):
            require(isinstance(case.get(field), str) and bool(case[field].strip()),
                    "Scenario requires both applicability branches and a reviewer question.")
        exercised.add(case["control_id"])
    require(exercised == control_ids, "Every control requires an applicability scenario pair.")


def prepare_packet(repo: Path) -> dict[str, Any]:
    source = source_snapshot(repo)
    content = {name: committed_text(repo, name, source) for name in EXTRA}
    register = json.loads(content[REGISTER])
    scenarios = json.loads(content[SCENARIOS])
    # Apply the same strict duplicate-key/non-finite parser as benchmark evidence.
    require(register == read_json(repo / REGISTER) and scenarios == read_json(repo / SCENARIOS),
            "Seed bytes differ from the committed source.")
    validate_seed(register, scenarios, source)
    paths = sorted({path for control in register["controls"] for path in control["paths"]})
    units = []
    for name in paths:
        raw = committed_text(repo, name, source)
        if name in KNOWLEDGE:
            units.extend(knowledge_units(name, raw))
        else:
            units.append({"id": "file:" + name, "kind": "complete_policy_file", "path": name,
                          "start_line": 1, "end_line": len(raw.splitlines()), "text": raw,
                          "sha256": source["file_sha256"][name]})
    # Highlight constructions throughout all Python app modules, even if a new
    # module was not yet mapped into the seed. Reviewers must extend the map.
    for name in sorted(source["file_sha256"]):
        if name.startswith("app/") and name.endswith(".py"):
            units.extend(prompt_units(name, committed_text(repo, name, source)))
    require(len({unit["id"] for unit in units}) == len(units), "Duplicate review unit.")
    packet = {"schema_version": 1, "purpose": "independent_legal_evidence_review_preparation",
              "status": "unapproved", "pilot_admission": "not_authorized_by_this_tool",
              "source": source, "authority_register": register, "scenarios": scenarios,
              "review_units": units,
              "reviewer_template": {"reviewer_id": "", "qualification_reference": "",
                                    "independence_reference": "", "reviewed_at": "",
                                    "signature_reference": "", "unit_findings": [
                  {"unit_id": unit["id"], "unit_sha256": unit["sha256"],
                   "status": "unverified", "rule_findings": []} for unit in units],
                                    "scenario_findings": [
                  {"scenario_id": case["id"], "status": "not_run", "evidence_reference": "",
                   "rationale": ""} for case in scenarios["cases"]]}}
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
        print("Unapproved review packet prepared. Qualified signed review is still required; pilot remains NO-GO.")
        return 0
    except (BenchmarkInvalid, KeyError, TypeError, ValueError, SyntaxError, OSError,
            subprocess.CalledProcessError):
        # No arbitrary source/paths, operator information or tracebacks on errors.
        print("Review packet not written: source or seeds are incomplete, changed, or output already exists.",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
