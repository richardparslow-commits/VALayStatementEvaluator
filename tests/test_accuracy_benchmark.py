"""Evidence validation with invented provider/reviewer attestations; no live calls."""
import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests import hermetic  # noqa: E402,F401  (isolated test configuration)

from app.accuracy_benchmark import (
    BenchmarkInvalid, assess, digest, prepare, read_json, source_snapshot, validate_corpus,
)

ROOT = Path(__file__).resolve().parent.parent


class AccuracyBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.corpus = read_json(ROOT / "benchmarks/synthetic-accuracy-v1.json")
        self.source = {"revision": "a" * 40, "tree": "b" * 40,
                       "file_sha256": {"app/evaluate.py": "c" * 64, "app/knowledge/example.md": "d" * 64}}
        self.config = {"schema_version": 1, "provider": "invented-test-provider",
                       "base_url": "https://synthetic.invalid/v1", "account_reference": "fixture-only",
                       "region_reference": "fixture-only", "approval_reference": "fixture-only",
                       "tools": [], "fallbacks": [], "request_profiles": [{"id": "fixture",
                       "model": "invented-model-v1", "version_reference": "fixture-v1",
                       "parameters": {"temperature": 0, "max_tokens": 100}}]}
        self.plan = prepare(self.corpus, self.source, self.config)
        people = [{"id": role, "role": role, "qualification_reference": "fixture-only",
                   "independent": True, "signature_reference": "invented-signature"}
                  for role in ("evidence", "medical", "qa")]
        self.agreement = {"schema_version": 1, "plan_sha256": self.plan["plan_sha256"],
                          "signed_at": "2020-01-01T00:00:00Z", "reviewers": people}
        for key in ("operator_reference", "corpus_review_reference", "threshold_approval_reference",
                    "provider_approval_reference", "budget_approval_reference"):
            self.agreement[key] = "invented-fixture-approval"
        self.results = {"schema_version": 1, "plan_sha256": self.plan["plan_sha256"],
                        "agreement_sha256": digest(self.agreement), "runs": []}
        self.review = {"schema_version": 1, "plan_sha256": self.plan["plan_sha256"],
                       "agreement_sha256": digest(self.agreement), "completed_at": "2020-01-04T00:00:00Z",
                       "signatures": [{"reviewer_id": r, "signature_reference": "invented-signature"}
                                      for r in ("evidence", "medical", "qa")],
                       "ratings": [], "adjudications": []}
        for case in self.corpus["cases"]:
            for pathway in ("evaluate", "draft"):
                identity = {"case_id": case["id"], "pathway": pathway, "repetition": 1}
                run = {**identity, "input_sha256": digest(case["inputs"]), "source_tree": self.source["tree"],
                       "configuration_sha256": digest(self.config), "origin": case["origin"],
                       "provider_evidence_reference": "invented-fixture-only",
                       "started_at": "2020-01-02T00:00:00Z", "finished_at": "2020-01-03T00:00:00Z",
                       "output": {"status": "complete", "result": {"text": "invented test output"}},
                       "requests": [{"profile_id": "fixture", "request_id": case["id"] + pathway,
                                     "returned_model": "invented-model-v1", "version_reference": "fixture-v1",
                                     "parameters": self.config["request_profiles"][0]["parameters"],
                                     "system": "invented fixture", "user": "invented fixture",
                                     "response": "invented fixture"}]}
                if case["origin"] == "fault_injection":
                    run["fault_injection_reference"] = "invented truncation evidence"
                self.results["runs"].append(run)
                for check in case["checkpoints"]:
                    for person in ("evidence", "medical"):
                        self.review["ratings"].append({**identity, "checkpoint_id": check["id"],
                            "reviewer_id": person, "passed": True, "unflagged_critical_changes": 0,
                            "false_contradictions": 0, "missed_critical_facts": 0,
                            "rationale": "invented reviewer judgment", "output_pointer": "/result/text"})
        self.bind_results()

    def bind_results(self):
        self.review["results_sha256"] = digest(self.results)

    def check(self):
        return assess(self.plan, self.agreement, self.results, self.review, self.source)

    def invalid(self):
        with self.assertRaises((BenchmarkInvalid, KeyError, TypeError, ValueError)):
            self.check()

    def test_complete_attestations_never_authorize_pilot(self):
        result = self.check()
        self.assertEqual(result["disposition"], "REVIEW_READY")
        self.assertEqual(result["pilot_admission"], "not_authorized_by_this_tool")
        self.assertEqual((result["runs"], result["checkpoints"]), (24, 148))
        self.assertNotIn("accuracy_validation", result)

    def test_corpus_has_every_original_span_and_scenario(self):
        validate_corpus(self.corpus)
        self.assertEqual(self.corpus["review_status"], "unreviewed_seed")
        self.corpus["cases"][0]["checkpoints"][0]["spans"][0]["quote"] = "absent quote"
        with self.assertRaises(BenchmarkInvalid):
            validate_corpus(self.corpus)

    def test_changed_plan_cannot_reuse_review(self):
        self.plan["corpus"]["cases"][0]["inputs"]["account"] += " changed"
        self.invalid()

    def test_changed_source_and_knowledge_block(self):
        self.source = copy.deepcopy(self.source)
        self.source["file_sha256"]["app/knowledge/example.md"] = "e" * 64
        self.invalid()

    def test_changed_output_invalidates_review(self):
        self.results["runs"][0]["output"]["result"]["text"] = "changed result"
        self.invalid()

    def test_missing_extra_and_duplicate_runs_block(self):
        for kind in ("missing", "extra", "duplicate"):
            with self.subTest(kind=kind):
                saved = copy.deepcopy(self.results)
                if kind == "missing": self.results["runs"].pop()
                else:
                    run = copy.deepcopy(self.results["runs"][0])
                    if kind == "extra": run["case_id"] = "not-in-corpus"
                    self.results["runs"].append(run)
                self.bind_results()
                self.invalid()
                self.results = saved

    def test_fake_origins_and_changed_input_block(self):
        for field, value in (("origin", "fake_model"), ("input_sha256", "x"),
                             ("configuration_sha256", "x"), ("source_tree", "x")):
            with self.subTest(field=field):
                run = self.results["runs"][0]
                old = run[field]
                run[field] = value
                self.bind_results()
                self.invalid()
                run[field] = old

    def test_profile_settings_version_and_model_must_match(self):
        request = self.results["runs"][0]["requests"][0]
        for field, value in (("parameters", {}), ("version_reference", "different-version"),
                             ("returned_model", "fallback-model"), ("profile_id", "unknown")):
            with self.subTest(field=field):
                old = request[field]
                request[field] = value
                self.bind_results()
                self.invalid()
                request[field] = old

    def test_unexercised_profile_blocks(self):
        new_config = copy.deepcopy(self.config)
        new_config["request_profiles"].append({**new_config["request_profiles"][0], "id": "untested"})
        self.plan = prepare(self.corpus, self.source, new_config)
        self.agreement["plan_sha256"] = self.plan["plan_sha256"]
        for bundle in (self.results, self.review):
            bundle["plan_sha256"] = self.plan["plan_sha256"]
            bundle["agreement_sha256"] = digest(self.agreement)
        for run in self.results["runs"]: run["configuration_sha256"] = digest(new_config)
        self.bind_results()
        self.invalid()

    def test_actual_attempts_full_prompts_and_unique_request_ids_required(self):
        run = self.results["runs"][0]
        saved = copy.deepcopy(run["requests"])
        for kind in ("empty", "missing_prompt", "reused_id"):
            with self.subTest(kind=kind):
                if kind == "empty": run["requests"] = []
                if kind == "missing_prompt": run["requests"][0]["system"] = ""
                if kind == "reused_id": run["requests"][0]["request_id"] = self.results["runs"][1]["requests"][0]["request_id"]
                self.bind_results()
                self.invalid()
                run["requests"] = copy.deepcopy(saved)

    def test_example_provider_configuration_is_unapproved(self):
        self.plan = prepare(self.corpus, self.source, read_json(ROOT / "benchmarks/accuracy-configuration.example.json"))
        self.invalid()

    def test_posthoc_or_future_reviews_block(self):
        for date in ("2019-01-01T00:00:00Z", "2999-01-01T00:00:00Z", "2020-01-01"):
            self.review["completed_at"] = date
            self.invalid()
        self.review["completed_at"] = "2020-01-04T00:00:00Z"
        self.results["runs"][0]["started_at"] = "2019-01-01T00:00:00Z"
        self.bind_results()
        self.invalid()

    def test_independent_review_coverage_and_signatures_required(self):
        for field in ("ratings", "signatures"):
            with self.subTest(field=field):
                old = self.review[field].pop()
                self.invalid()
                self.review[field].append(old)
        self.review["ratings"].append(copy.deepcopy(self.review["ratings"][0]))
        self.invalid()

    def test_invalid_pointer_or_non_boolean_judgment_blocks(self):
        for field, value in (("output_pointer", "/result/missing"), ("passed", "true"),
                             ("unflagged_critical_changes", True), ("false_contradictions", -1),
                             ("repetition", True)):
            with self.subTest(field=field):
                row = self.review["ratings"][0]
                old = row[field]
                row[field] = value
                self.invalid()
                row[field] = old

    def test_disagreement_preserved_and_signed_qa_adjudication_required(self):
        original = self.review["ratings"][0]
        original["passed"] = False
        self.invalid()
        resolved = {**original, "reviewer_id": "qa", "passed": True,
                    "signature_reference": "invented adjudication signature",
                    "rationale": "invented reason for resolving conflicting original judgments"}
        self.review["adjudications"] = [resolved]
        self.assertEqual(self.check()["disagreements"], 1)
        self.assertFalse(original["passed"])
        resolved["signature_reference"] = ""
        self.invalid()

    def test_each_error_metric_and_failed_checkpoint_produce_no_go(self):
        for metric in ("unflagged_critical_changes", "false_contradictions", "missed_critical_facts", "passed"):
            with self.subTest(metric=metric):
                for row in self.review["ratings"][:2]: row[metric] = False if metric == "passed" else 1
                result = self.check()
                self.assertEqual(result["disposition"], "NO_GO")
                for row in self.review["ratings"][:2]: row[metric] = True if metric == "passed" else 0

    def test_duplicate_json_keys_and_nonfinite_numbers_block(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            for value in ('{"id":1,"id":2}', '{"n":NaN}', '{"n":Infinity}', '[]'):
                path.write_text(value)
                with self.assertRaises(BenchmarkInvalid): read_json(path)

    def test_incomplete_actual_run_is_no_go_even_with_passing_attestations(self):
        for status in ("partial", "blocked", "error"):
            self.results["runs"][0]["output"]["status"] = status
            self.bind_results()
            result = self.check()
            self.assertEqual(result["disposition"], "NO_GO")
            self.assertEqual(result["incomplete_actual_runs"], 1)

    def test_evidence_storage_excluded_from_git_and_image(self):
        from tests.dockerfile import IgnoreFile
        ignore = IgnoreFile.for_repo()
        for path in ("accuracy-evidence/plan.json", "nested/accuracy-evidence/results.json"):
            self.assertTrue(ignore.ignored(path))
            process = subprocess.run(["git", "-C", str(ROOT), "check-ignore", path], capture_output=True)
            self.assertEqual(process.returncode, 0)

    def test_source_freeze_rejects_dirty_and_untracked_code(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            (repo / "app/knowledge").mkdir(parents=True)
            (repo / "app/example.py").write_text("pass\n")
            (repo / "app/knowledge/example.md").write_text("invented\n")
            subprocess.run(["git", "-C", str(repo), "add", "app"], check=True)
            subprocess.run(["git", "-C", str(repo), "-c", "user.name=Fixture", "-c",
                            "user.email=fixture@example.invalid", "commit", "-qm", "fixture"], check=True)
            snapshot = source_snapshot(repo)
            self.assertEqual(len(snapshot["file_sha256"]), 2)
            (repo / "app/untracked.py").write_text("pass\n")
            with self.assertRaises(BenchmarkInvalid): source_snapshot(repo)
            (repo / "app/untracked.py").unlink()
            # Git's assume-unchanged must not hide altered prompt/knowledge bytes.
            subprocess.run(["git", "-C", str(repo), "update-index", "--assume-unchanged", "app/example.py"], check=True)
            (repo / "app/example.py").write_text("raise Exception\n")
            with self.assertRaises(BenchmarkInvalid): source_snapshot(repo)


if __name__ == "__main__":
    unittest.main()
