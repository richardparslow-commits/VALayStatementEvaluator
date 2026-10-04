"""Evidence validation with invented provider/reviewer attestations; no live calls."""
import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests import hermetic  # noqa: E402,F401  (isolated test configuration)

from tests.rubric_fixtures import scored_result_fields
from tests.topic_fixtures import topic_result_fields
from tests.grounding_fixtures import complete_grounding

from app.accuracy_benchmark import (
    BenchmarkInvalid, assess, digest, document_specs, prepare, read_json, source_snapshot, validate_corpus,
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
                       "phase": "fixture-phase", "required_pathways": ["evaluate", "draft"],
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
                result = {"text": "invented test output"}
                if pathway == "evaluate":
                    result.update(scored_result_fields())
                    result.update(topic_result_fields())
                    result.update(verification_policy="uploaded_source_unit_v1",
                                  claims=[{"id":1,"text":case["inputs"]["account"]}],
                                  verifications=[{"id":1,"verdict":"NOT FOUND"}])
                else:
                    result.update(draft=case["inputs"]["account"],
                                  grounding_policy="retained_fact_full_quote_source_unit_v1",
                                  grounding=complete_grounding(case["inputs"]["account"]))
                run = {**identity, "input_sha256": digest(case["inputs"]), "source_tree": self.source["tree"],
                       "configuration_sha256": digest(self.config), "origin": case["origin"],
                       "provider_evidence_reference": "invented-fixture-only",
                       "started_at": "2020-01-02T00:00:00Z", "finished_at": "2020-01-03T00:00:00Z",
                       "output": {"status": "complete", "result": result},
                       "requests": [{"profile_id": "fixture", "phase":"fixture-phase", "request_id": case["id"] + pathway,
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

    def test_every_seed_case_passes_both_application_input_contracts(self):
        from app.documents import DocumentPage, ExtractedDocument
        from app.request_validation import validate_draft_request, validate_evaluation_request
        for case in self.corpus["cases"]:
            with self.subTest(case=case["id"]):
                inputs=case["inputs"]
                records=[]
                for descriptor in document_specs(inputs):
                    descriptor["pages"]=[DocumentPage(**page) for page in descriptor["pages"]]
                    records.append(ExtractedDocument(**descriptor))
                validate_evaluation_request(statement_text=inputs["account"],records=records,witness=inputs["witness"])
                validate_draft_request(observations=inputs["account"],condition=inputs["condition"],
                                       claim_type=inputs["claim_type"],witness=inputs["witness"],records=records)
                if case["scenario"]=="missing_scan":
                    self.assertEqual(len(records),1)
                    self.assertEqual((records[0].total_pages,len(records[0].pages),records[0].unreadable_pages),(2,1,[2]))

    def test_unreadable_unit_requires_explicit_metadata_and_readable_sibling(self):
        case=copy.deepcopy(next(c for c in self.corpus["cases"] if c["scenario"]=="missing_scan"))
        del case["inputs"]["records"][1]["unreadable"]
        with self.assertRaises(BenchmarkInvalid): document_specs(case["inputs"])
        case["inputs"]["records"][1].update(unreadable=True,label="standalone.pdf p.1")
        with self.assertRaises(BenchmarkInvalid): document_specs(case["inputs"])

    def test_missing_page_or_block_requires_explicit_coverage(self):
        for letter in ("p","b"):
            inputs={"records":[{"label":f"synthetic.txt {letter}.1","text":"first"},
                               {"label":f"synthetic.txt {letter}.3","text":"third"}]}
            with self.assertRaises(BenchmarkInvalid): document_specs(inputs)
        inputs["records"]=[{"label":"synthetic.pdf p.1","text":"first"},
                           {"label":"synthetic.pdf p.3","text":"third"},
                           {"label":"synthetic.pdf p.2","text":"","unreadable":True}]
        self.assertEqual(document_specs(inputs)[0]["unreadable_pages"],[2])
        inputs["records"].append(inputs["records"][0])
        with self.assertRaises(BenchmarkInvalid): document_specs(inputs)

    def test_non_object_evidence_rows_raise_sanitized_validation_error(self):
        for collection in ("reviewers","signatures","runs","requests","ratings","adjudications"):
            with self.subTest(collection=collection):
                agreement,results,review=copy.deepcopy((self.agreement,self.results,self.review))
                if collection=="reviewers": agreement["reviewers"][0]="malformed"
                elif collection=="requests": results["runs"][0]["requests"][0]="malformed"
                elif collection=="runs": results["runs"][0]="malformed"
                else: review[collection]=["malformed"] if collection=="adjudications" else ["malformed"]+review[collection][1:]
                results["agreement_sha256"]=review["agreement_sha256"]=digest(agreement)
                review["results_sha256"]=digest(results)
                with self.assertRaises(BenchmarkInvalid): assess(self.plan,agreement,results,review,self.source)

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
                             ("returned_model", "fallback-model"), ("profile_id", "unknown"), ("phase", "wrong-phase")):
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

    def test_internal_evaluation_and_drafting_incomplete_cannot_be_labeled_complete(self):
        eval_result=self.results["runs"][0]["output"]["result"]
        for field,value in (("scoring_status","incomplete"),("topic_status","incomplete"),
                            ("topic_rows",[]),("verifications",[]),("scores",{})):
            with self.subTest(field=field):
                old=eval_result[field]
                eval_result[field]=value
                self.bind_results()
                self.assertEqual(self.check()["disposition"],"NO_GO")
                eval_result[field]=old
        draft_result=self.results["runs"][1]["output"]["result"]
        for field,value in (("grounding",{}),("grounding_policy","legacy"),("draft","")):
            with self.subTest(field=field):
                old=draft_result[field]
                draft_result[field]=value
                self.bind_results()
                self.assertEqual(self.check()["disposition"],"NO_GO")
                draft_result[field]=old

    def test_complete_drafting_topic_labels_match_application_whitespace_rules(self):
        from app.draft import _normalize_grounding
        draft_result=self.results["runs"][1]["output"]["result"]
        for topic in draft_result["grounding"]["topic_coverage"]:
            topic["topic"]=" \t"+topic["topic"]+"  "
        _normalize_grounding(draft_result["grounding"],observations_present=True)
        self.bind_results()
        self.assertEqual(self.check()["disposition"],"REVIEW_READY")

    def test_profile_must_run_in_each_required_pathway(self):
        new_config=copy.deepcopy(self.config)
        new_config["request_profiles"].append({**new_config["request_profiles"][0], "id":"both-pathways"})
        self.plan=prepare(self.corpus,self.source,new_config)
        self.agreement["plan_sha256"]=self.plan["plan_sha256"]
        for bundle in (self.results,self.review):
            bundle["plan_sha256"]=self.plan["plan_sha256"]
            bundle["agreement_sha256"]=digest(self.agreement)
        for run in self.results["runs"]:
            run["configuration_sha256"]=digest(new_config)
            if run["pathway"]=="evaluate":
                request=copy.deepcopy(run["requests"][0])
                request.update(profile_id="both-pathways",request_id=request["request_id"]+"additional")
                run["requests"].append(request)
        self.bind_results()
        self.invalid()

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
            (repo / "app/condition_topics.json").write_text('{"fixture":true}\n')
            subprocess.run(["git", "-C", str(repo), "add", "app"], check=True)
            subprocess.run(["git", "-C", str(repo), "-c", "user.name=Fixture", "-c",
                            "user.email=fixture@example.invalid", "commit", "-qm", "fixture"], check=True)
            snapshot = source_snapshot(repo)
            self.assertEqual(len(snapshot["file_sha256"]), 3)
            (repo / "app/untracked.py").write_text("pass\n")
            with self.assertRaises(BenchmarkInvalid): source_snapshot(repo)
            (repo / "app/untracked.py").unlink()
            # Git's assume-unchanged must not hide altered prompt/knowledge bytes.
            subprocess.run(["git", "-C", str(repo), "update-index", "--assume-unchanged", "app/condition_topics.json"], check=True)
            (repo / "app/condition_topics.json").write_text('{"fixture":false}\n')
            with self.assertRaises(BenchmarkInvalid): source_snapshot(repo)


if __name__ == "__main__":
    unittest.main()
