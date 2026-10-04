"""Offline synthetic accuracy evidence packaging; never a model or admission gate.

This module does not import application settings, clients, or pipelines. Review
judgments and signature references are attestations, not authenticated signatures.
Hash binding detects changed artifacts; it cannot prove how outputs were obtained.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from types import SimpleNamespace
from urllib.parse import urlsplit

from .rubric_validation import rubric_is_complete

SCHEMA = 1
DIMENSIONS = {"extraction", "findings", "topic_applicability", "revision",
              "source_support", "critical_integrity", "coverage", "failure_handling"}
SCENARIOS = {"chronology", "negation", "laterality", "uncertain_onset", "attribution",
             "multiple_conditions", "silent_records", "missing_scan", "repeated_findings",
             "prompt_injection", "truncated_output", "numbers_and_diagnoses"}
LIMITS = {"unflagged_critical_changes": 0, "false_contradictions": 0,
          "missed_critical_facts": 0, "failed_checkpoints": 0}


class BenchmarkInvalid(ValueError):
    """Missing, inconsistent, or stale benchmark evidence."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BenchmarkInvalid(message)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    require(path.stat().st_size <= 32 * 1024 * 1024, "Evidence file exceeds 32 MiB.")
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key.")
            result[key] = value
        return result
    def constant(_: str) -> Any:
        raise BenchmarkInvalid("Non-finite JSON number.")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=constant)
    require(isinstance(value, dict), "Evidence must be a JSON object.")
    return dict(value)


def text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def timestamp(value: Any) -> datetime:
    require(isinstance(value, str), "Missing timestamp.")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None, "Timestamp must include a timezone.")
    return result.astimezone(timezone.utc)


def source_snapshot(repo: Path) -> dict[str, Any]:
    """Bind tracked source plus knowledge bytes without reading secrets or .env."""
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True,
                                       stderr=subprocess.PIPE).strip()
    require(not git("status", "--porcelain", "--untracked-files=all"),
            "Freeze a clean committed source revision first.")
    revision, tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
    names = git("ls-tree", "-r", "--name-only", "HEAD", "app", "requirements.lock").splitlines()
    fingerprints = {}
    for name in names:
        if name.startswith("app/") or name == "requirements.lock":
            path = repo / name
            require(path.is_file() and not path.is_symlink(), "Source file is missing or linked.")
            raw = path.read_bytes()
            committed = subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:{name}"],
                                                stderr=subprocess.PIPE)
            require(raw == committed, "Working source differs from the frozen Git objects.")
            fingerprints[name] = hashlib.sha256(raw).hexdigest()
    require(bool(fingerprints) and any(n.startswith("app/knowledge/") for n in fingerprints),
            "Application and knowledge snapshot required.")
    return {"revision": revision, "tree": tree, "file_sha256": fingerprints}


def document_specs(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    """Pure serialized document descriptors, with unreadable metadata, not blank pages."""
    records = inputs.get("records")
    require(isinstance(records, list) and bool(records), "Synthetic records must be a nonempty list.")
    assert isinstance(records, list)
    documents: dict[str, dict[str, Any]] = {}
    for unit in records:
        require(isinstance(unit, dict) and text(unit.get("label")) and isinstance(unit.get("text"), str),
                "Synthetic record unit must be an object with an address and text.")
        match = re.fullmatch(r"(.+) ([pb])\.([1-9]\d*)", unit["label"])
        require(match is not None, "Synthetic source label must name one page or block.")
        assert match is not None
        filename, letter, number = match.group(1), match.group(2), int(match.group(3))
        kind = "page" if letter == "p" else "block"
        unreadable = unit.get("unreadable", False)
        require(type(unreadable) is bool and (bool(unit["text"].strip()) != unreadable)
                and (not unreadable or kind == "page"), "Unreadable pages need explicit metadata and no extracted text.")
        doc = documents.setdefault(filename, {"filename": filename, "pages": [], "total_pages": 0,
                                              "unreadable_pages": [], "pagination": kind, "coverage_known": True})
        require(doc["pagination"] == kind, "One document cannot mix pages and text blocks.")
        doc["total_pages"] = max(doc["total_pages"], number)
        if unreadable:
            doc["unreadable_pages"].append(number)
        else:
            doc["pages"].append({"filename": filename, "page": number, "text": unit["text"], "kind": kind})
    require(all(doc["pages"] for doc in documents.values()),
            "Each benchmark document needs at least one readable page accepted by the app.")
    return list(documents.values())


def validate_corpus(corpus: dict[str, Any]) -> None:
    require(corpus.get("schema_version") == SCHEMA and corpus.get("synthetic") is True,
            "Only the synthetic corpus schema is supported.")
    cases = corpus.get("cases")
    require(isinstance(cases, list) and bool(cases), "Corpus has no cases.")
    assert isinstance(cases, list)
    ids: set[str] = set()
    tags: set[str] = set()
    dimensions: set[str] = set()
    for case in cases:
        require(isinstance(case, dict) and text(case.get("id")) and case["id"] not in ids,
                "Missing or duplicate case ID.")
        ids.add(case["id"])
        require(case.get("scenario") in SCENARIOS, "Unknown scenario.")
        tags.add(case["scenario"])
        require(case.get("origin") in {"actual_provider", "fault_injection"}, "Unknown run origin.")
        inputs = case["inputs"]
        require(isinstance(inputs, dict) and text(inputs.get("account"))
                and text(inputs.get("condition")) and isinstance(inputs.get("witness"), dict),
                "Missing synthetic inputs.")
        document_specs(inputs)
        units = {"account": inputs["account"]}
        for unit in inputs["records"]:
            require(isinstance(unit, dict) and text(unit.get("label"))
                    and unit["label"] not in units and isinstance(unit.get("text"), str),
                    "Invalid or duplicate source unit.")
            units[unit["label"]] = unit["text"]
        checks: set[str] = set()
        require(isinstance(case.get("checkpoints"), list) and bool(case["checkpoints"]),
                "Case has no expected checkpoints.")
        for check in case["checkpoints"]:
            require(isinstance(check, dict), "Expected checkpoint must be an object.")
            require(text(check.get("id")) and check["id"] not in checks
                    and check.get("dimension") in DIMENSIONS and text(check.get("expected")),
                    "Missing or duplicate expected checkpoint.")
            checks.add(check["id"])
            dimensions.add(check["dimension"])
            require(isinstance(check.get("spans"), list) and bool(check["spans"]),
                    "Expected finding must link an original span.")
            for span in check["spans"]:
                require(isinstance(span, dict), "Expected source span must be an object.")
                require(span.get("unit") in units and text(span.get("quote"))
                        and span["quote"] in units[span["unit"]], "Expected quote is absent from its unit.")
    require(tags == SCENARIOS and dimensions == DIMENSIONS, "Required scenario or review dimension is absent.")


def prepare(corpus: dict[str, Any], source: dict[str, Any], configuration: dict[str, Any],
            repetitions: int = 1) -> dict[str, Any]:
    validate_corpus(corpus)
    require(type(repetitions) is int and 1 <= repetitions <= 10, "Repetitions must be 1–10.")
    plan = {"schema_version": SCHEMA, "purpose": "synthetic_accuracy_review",
            "corpus": corpus, "corpus_sha256": digest(corpus), "source": source,
            "configuration": configuration, "thresholds": dict(LIMITS), "repetitions": repetitions}
    return {**plan, "plan_sha256": digest(plan)}


def validate_configuration(configuration: dict[str, Any]) -> dict[str, dict[str, Any]]:
    require(set(configuration) == {"schema_version", "provider", "account_reference", "region_reference",
                                  "approval_reference", "base_url", "tools", "fallbacks", "request_profiles"},
            "Only non-secret configuration fields are supported.")
    require(configuration.get("schema_version") == SCHEMA, "Unknown configuration schema.")
    for key in ("provider", "account_reference", "region_reference", "approval_reference"):
        require(text(configuration.get(key)), "Actual provider/account/region/approval references required.")
    url = urlsplit(configuration["base_url"])
    require(url.scheme == "https" and bool(url.hostname) and not url.username
            and not url.password and not url.query and not url.fragment, "Approved HTTPS endpoint required.")
    require(configuration.get("tools") == [] and configuration.get("fallbacks") == [],
            "Search/tools and untested fallback routes must be disabled.")
    profiles: dict[str, dict[str, Any]] = {}
    require(isinstance(configuration.get("request_profiles"), list), "Request profiles must be a list.")
    for profile in configuration["request_profiles"]:
        require(isinstance(profile, dict) and text(profile.get("id")) and profile["id"] not in profiles,
                "Invalid or duplicate request profile.")
        require(text(profile.get("model")) and text(profile.get("version_reference"))
                and isinstance(profile.get("parameters"), dict) and bool(profile["parameters"]),
                "Exact model/version and effective parameters required for each profile.")
        require(text(profile.get("phase")) and isinstance(profile.get("required_pathways"), list)
                and bool(profile["required_pathways"])
                and len(set(profile["required_pathways"])) == len(profile["required_pathways"])
                and set(profile["required_pathways"]).issubset({"evaluate", "draft"}),
                "Each request profile must name its phase and required pathways.")
        profiles[profile["id"]] = profile
    require(bool(profiles), "All effective request profiles must be recorded.")
    return profiles


def internal_complete(pathway: str, result: dict[str, Any]) -> bool:
    """Validate offline serialized completion independently of the outer label.

    Rubric uses the existing pure validator. Other checks stay here so assessment
    never imports UI, settings, telemetry or LLM clients. Full semantic support
    and the complete serialized payload remain independent review requirements.
    """
    if pathway == "evaluate":
        if (not rubric_is_complete(SimpleNamespace(**result))
                or result.get("topic_policy") != "complete_evaluation_topics_v1"
                or result.get("topic_status") != "complete"
                or result.get("verification_policy") != "uploaded_source_unit_v1"
                or not text(result.get("topic_focus"))
                or not isinstance(result.get("topic_notes"), str)
                or not isinstance(result.get("topic_critical_gaps"), list)):
            return False
        rows = result.get("topic_rows")
        if not isinstance(rows, list) or len(rows) != 15:
            return False
        labels = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"topic", "applicable", "coverage", "evidence", "gap_note"}:
                return False
            match = re.match(r"^([A-O])(?:\s*[.():\-–—]|\s|$)", str(row["topic"]))
            if (not match or match[1] in labels or type(row["applicable"]) is not bool
                    or row["coverage"] not in {"covered", "partial", "absent", "not applicable"}
                    or not isinstance(row["evidence"], str) or not isinstance(row["gap_note"], str)
                    or (row["coverage"] == "not applicable") == row["applicable"]):
                return False
            labels.add(match[1])
        claims, verdicts = result.get("claims"), result.get("verifications")
        if not isinstance(claims, list) or not claims or not isinstance(verdicts, list):
            return False
        ids = [c.get("id") for c in claims if isinstance(c, dict)]
        verified = [v.get("id") for v in verdicts if isinstance(v, dict)]
        if (len(ids) != len(claims) or len(verified) != len(verdicts)
                or any(type(i) is not int for i in ids + verified)
                or len(set(ids)) != len(ids) or len(ids) != len(verified) or set(ids) != set(verified)):
            return False
        return all(v.get("verdict") in {"SUPPORTED", "PARTIALLY SUPPORTED", "CONTRADICTED", "NOT FOUND"}
                   for v in verdicts)
    if (not text(result.get("draft"))
            or result.get("grounding_policy") != "retained_fact_full_quote_source_unit_v1"):
        return False
    grounding = result.get("grounding")
    if not isinstance(grounding, dict):
        return False
    for field in ("supported_observations", "unverified_observations", "conflicts",
                  "suggested_inclusions", "strengthening_questions", "topic_coverage"):
        if not isinstance(grounding.get(field), list):
            return False
    topics = grounding["topic_coverage"]
    labels = set()
    for row in topics:
        if not isinstance(row, dict) or not text(row.get("topic")):
            return False
        match = re.match(r"^([A-O])(?:\s*[.():\-–—]|\s|$)", row["topic"])
        if (not match or match[1] in labels or type(row.get("applicable")) is not bool
                or type(row.get("covered")) is not bool
                or (row["covered"] and not row["applicable"])
                or not isinstance(row.get("prompt_for_witness"), str)
                or bool(row["prompt_for_witness"].strip()) != (row["applicable"] and not row["covered"])):
            return False
        labels.add(match[1])
    return labels == set("ABCDEFGHIJKLMNO")


def check_pointer(pointer: Any, output: dict[str, Any]) -> None:
    require(isinstance(pointer, str) and pointer.startswith("/"), "Review must point into the captured output.")
    value: Any = output
    for part in pointer[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            require(part.isdecimal() and int(part) < len(value), "Review output pointer is absent.")
            value = value[int(part)]
        else:
            require(isinstance(value, dict) and part in value, "Review output pointer is absent.")
            value = value[part]


def rating(row: dict[str, Any]) -> tuple[bool, int, int, int]:
    require(type(row.get("passed")) is bool and text(row.get("rationale"))
            and text(row.get("output_pointer")), "Review needs an explicit judgment, rationale and output pointer.")
    counts = []
    for name in ("unflagged_critical_changes", "false_contradictions", "missed_critical_facts"):
        require(type(row.get(name)) is int and row[name] >= 0, "Review counts must be nonnegative integers.")
        counts.append(row[name])
    return row["passed"], counts[0], counts[1], counts[2]


def assess(plan: dict[str, Any], agreement: dict[str, Any], results: dict[str, Any],
           review: dict[str, Any], current_source: dict[str, Any]) -> dict[str, Any]:
    """Check complete, pre-agreed, independently adjudicated evidence bindings.

    REVIEW_READY means the submitted attestations meet the proposed thresholds.
    It never means signatures/provider provenance have been independently verified.
    """
    require(plan.get("schema_version") == SCHEMA and plan.get("purpose") == "synthetic_accuracy_review",
            "Unknown plan schema/purpose.")
    require(plan.get("plan_sha256") == digest({k: v for k, v in plan.items() if k != "plan_sha256"}),
            "Plan was changed after freezing.")
    require(plan["source"] == current_source, "Application/knowledge/source revision has changed.")
    validate_corpus(plan["corpus"])
    require(plan["corpus_sha256"] == digest(plan["corpus"]) and plan["thresholds"] == LIMITS
            and all(type(v) is int for v in plan["thresholds"].values()),
            "Corpus or proposed zero-error thresholds changed.")
    require(type(plan["repetitions"]) is int and 1 <= plan["repetitions"] <= 10, "Invalid repetition count.")
    profiles = validate_configuration(plan["configuration"])
    for bundle in (agreement, results, review):
        require(bundle.get("schema_version") == SCHEMA and bundle.get("plan_sha256") == plan["plan_sha256"],
                "Evidence belongs to another plan/schema.")
    for approval_field in ("operator_reference", "corpus_review_reference", "threshold_approval_reference",
                "provider_approval_reference", "budget_approval_reference"):
        require(text(agreement.get(approval_field)), "Missing pre-run review or approval reference.")
    reviewers = agreement["reviewers"]
    require(isinstance(reviewers, list) and len(reviewers) == 3, "Three distinct qualified reviewers required.")
    by_role: dict[str, str] = {}
    for person in reviewers:
        require(isinstance(person, dict), "Reviewer must be an object.")
        require(person.get("role") in {"evidence", "medical", "qa"} and text(person.get("id"))
                and person["id"] not in by_role.values() and person["role"] not in by_role
                and person.get("independent") is True and text(person.get("qualification_reference"))
                and text(person.get("signature_reference")), "Reviewer roles, independence or signature references incomplete.")
        by_role[person["role"]] = person["id"]
    agreement_hash = digest(agreement)
    require(results.get("agreement_sha256") == agreement_hash and review.get("agreement_sha256") == agreement_hash
            and review.get("results_sha256") == digest(results), "Results/review bindings are stale.")
    signed = timestamp(agreement["signed_at"])
    completed = timestamp(review["completed_at"])
    require(signed <= completed <= datetime.now(timezone.utc), "Invalid review timeline.")
    signers = review["signatures"]
    require(isinstance(signers, list) and len(signers) == 3
            and all(isinstance(s, dict) for s in signers)
            and {s["reviewer_id"] for s in signers} == set(by_role.values())
            and all(text(s.get("signature_reference")) for s in signers), "Final reviewer signatures incomplete.")
    cases = {c["id"]: c for c in plan["corpus"]["cases"]}
    expected = {(c, p, n) for c in cases for p in ("evaluate", "draft")
                for n in range(1, plan["repetitions"] + 1)}
    runs: dict[tuple[str, str, int], dict[str, Any]] = {}
    used_profiles: set[tuple[str, str]] = set()
    request_ids: set[str] = set()
    incomplete_runs = 0
    require(isinstance(results.get("runs"), list), "Runs must be a list.")
    for run in results["runs"]:
        require(isinstance(run, dict), "Run must be an object.")
        require(type(run.get("repetition")) is int, "Invalid repetition.")
        run_key = (run["case_id"], run["pathway"], run["repetition"])
        require(run_key in expected and run_key not in runs, "Missing, duplicate or extra run.")
        case = cases[run_key[0]]
        require(run.get("input_sha256") == digest(case["inputs"])
                and run.get("source_tree") == plan["source"]["tree"]
                and run.get("configuration_sha256") == digest(plan["configuration"])
                and run.get("origin") == case["origin"] and text(run.get("provider_evidence_reference")),
                "Run input/source/configuration/origin evidence is inconsistent.")
        require(signed <= timestamp(run["started_at"]) <= timestamp(run["finished_at"]) <= completed,
                "Collection must follow approval and precede final review.")
        output = run["output"]
        require(isinstance(output, dict) and output.get("status") in {"complete", "blocked", "partial", "error"}
                and isinstance(output.get("result"), dict) and bool(output["result"]), "Complete output or failure evidence required.")
        if case["origin"] == "actual_provider" and (output["status"] != "complete"
                                                  or not internal_complete(run["pathway"], output["result"])):
            incomplete_runs += 1
        if case["origin"] == "fault_injection":
            require(text(run.get("fault_injection_reference")), "Truncation injection evidence required.")
        require(isinstance(run["requests"], list) and bool(run["requests"]), "Actual provider attempt evidence missing.")
        for request in run["requests"]:
            require(isinstance(request, dict), "Provider request must be an object.")
            profile = profiles.get(request.get("profile_id"))
            require(profile is not None, "Untested request profile.")
            assert profile is not None
            require(request.get("returned_model") == profile["model"]
                    and request.get("version_reference") == profile["version_reference"]
                    and request.get("parameters") == profile["parameters"]
                    and request.get("phase") == profile["phase"]
                    and run["pathway"] in profile["required_pathways"], "Effective model/version/settings/phase changed.")
            require(text(request.get("request_id")) and request["request_id"] not in request_ids,
                    "Missing or reused provider request ID.")
            request_ids.add(request["request_id"])
            used_profiles.add((profile["id"], run["pathway"]))
            for field in ("system", "user", "response"):
                require(text(request.get(field)), "Full synthetic prompts and responses required.")
        runs[run_key] = run
    required_profiles = {(profile["id"], pathway) for profile in profiles.values()
                         for pathway in profile["required_pathways"]}
    require(set(runs) == expected and used_profiles == required_profiles,
            "Not all runs or required phase/model profiles were exercised in each pathway.")
    targets = {(c, p, n, check["id"]) for c, p, n in expected for check in cases[c]["checkpoints"]}
    ratings: dict[tuple[str, str, int, str], dict[str, tuple[bool, int, int, int]]] = {}
    require(isinstance(review.get("ratings"), list), "Ratings must be a list.")
    for row in review["ratings"]:
        require(isinstance(row, dict), "Rating must be an object.")
        require(type(row.get("repetition")) is int, "Invalid review repetition.")
        key = (row["case_id"], row["pathway"], row["repetition"], row["checkpoint_id"])
        require(key in targets and row.get("reviewer_id") in {by_role["evidence"], by_role["medical"]},
                "Unknown review target or reviewer.")
        votes = ratings.setdefault(key, {})
        require(row["reviewer_id"] not in votes, "Duplicate reviewer judgment.")
        check_pointer(row.get("output_pointer"), runs[key[:3]]["output"])
        votes[row["reviewer_id"]] = rating(row)
    require(set(ratings) == targets and all(len(v) == 2 for v in ratings.values()), "Independent checkpoint reviews are incomplete.")
    adjudications = {}
    require(isinstance(review.get("adjudications"), list), "Adjudications must be a list.")
    for row in review["adjudications"]:
        require(isinstance(row, dict), "Adjudication must be an object.")
        require(type(row.get("repetition")) is int, "Invalid adjudication repetition.")
        key = (row["case_id"], row["pathway"], row["repetition"], row["checkpoint_id"])
        require(key in targets and key not in adjudications and row.get("reviewer_id") == by_role["qa"]
                and text(row.get("signature_reference")), "Invalid or duplicate adjudication.")
        check_pointer(row.get("output_pointer"), runs[key[:3]]["output"])
        adjudications[key] = rating(row)
    disagreements = {key for key, votes in ratings.items() if len(set(votes.values())) != 1}
    require(set(adjudications) == disagreements, "Every disagreement needs a signed QA adjudication; preserve both original judgments.")
    counts = dict.fromkeys(LIMITS, 0)
    for key, votes in ratings.items():
        value = adjudications[key] if key in disagreements else next(iter(votes.values()))
        counts["failed_checkpoints"] += int(not value[0])
        for name, count in zip(list(LIMITS)[:3], value[1:]):
            counts[name] += count
    return {"schema_version": SCHEMA, "plan_sha256": plan["plan_sha256"],
            "results_sha256": digest(results), "review_sha256": digest(review),
            "disposition": "REVIEW_READY" if not incomplete_runs and all(counts[k] <= LIMITS[k] for k in LIMITS) else "NO_GO",
            "pilot_admission": "not_authorized_by_this_tool", "checkpoint_counts": counts,
            "runs": len(runs), "checkpoints": len(targets), "disagreements": len(disagreements),
            "incomplete_actual_runs": incomplete_runs,
            "limitations": "Attested judgments and signature/provenance references require independent verification. No statistical guarantee or live pilot authorization."}
