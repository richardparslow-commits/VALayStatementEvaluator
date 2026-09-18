"""Offline tests for job payload (de)serialization (app/job_payload.py).

The queue ships JSON between processes, so every object the pipeline returns has
to survive a round trip. These tests pin the parts that would silently lose data:
the digest's facts (the evidence the report cites), the derived counters, and
tolerance for corrupt or forward-versioned values.
"""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app import config  # noqa: E402
from app.documents import document_from_text  # noqa: E402
from app.draft import DraftResult  # noqa: E402
from app.evaluate import DIMENSION_LABELS, EvaluationResult  # noqa: E402
from app.job_payload import (  # noqa: E402
    KIND_DRAFT,
    KIND_EVALUATE,
    DraftJob,
    EvaluateJob,
    PayloadError,
    PayloadTooLargeError,
    RunResult,
    decode_job,
    decode_result,
    digest_from_json,
    digest_to_json,
    documents_from_json,
    documents_to_json,
    draft_from_json,
    draft_to_json,
    encode_job,
    encode_result,
    evaluation_from_json,
    evaluation_to_json,
    payload_page_count,
    usage_from_json,
    usage_to_json,
)
from app.medical_review import MedicalDigest, MedicalFact  # noqa: E402
from app.usage import UsageTracker  # noqa: E402


def _docs():
    return [
        document_from_text("a.txt", "EVT one knee pain noted."),
        document_from_text("b.txt", "EVT two brace prescribed."),
    ]


def _digest() -> MedicalDigest:
    return MedicalDigest(
        facts=[
            MedicalFact("2020-01", "symptom", "Knee pain after lifting.", "a.txt p.1", "knee pain"),
            MedicalFact("2021-06", "treatment", "Prescribed brace.", "a.txt p.1", "brace"),
        ],
        conditions=["knee pain"],
        providers=["Dr. Smith (ortho)"],
        summary="Knee history.",
        pages_reviewed=2,
        chunks_reviewed=1,
        duplicates_skipped=3,
    )


def _evaluation() -> EvaluationResult:
    return EvaluationResult(
        claimed_condition="knee condition",
        writer_role="spouse",
        claims=[{"id": 1, "text": "Injured knee lifting pallet.", "type": "in_service_event"}],
        verifications=[
            {"id": 1, "verdict": "CONTRADICTED", "record_reference": "a.txt p.1", "note": "no"}
        ],
        scores={k: 6.0 for k in DIMENSION_LABELS},
        rationales={k: "Because " + k for k in DIMENSION_LABELS},
        improvements=[{"priority": 1, "problem": "Vague", "suggestion": "Add dates."}],
        omitted_record_facts=[{"fact": "Brace prescribed.", "source": "a.txt p.1"}],
        executive_summary="Adequate with gaps.",
        topic_focus="knee - increased rating",
        topic_rows=[{"topic": "A. Hazards", "applicable": True, "coverage": "partial"}],
        topic_critical_gaps=["A. Hazards needs detail"],
        topic_notes="Good base.",
        revision_notes="Aligned timeline.",
        revision_changes=[{"category": "specificity", "original": "hurt", "revised": "hurt daily"}],
        revised_statement="I confirm daily knee pain.",
        added_facts_to_verify=["Brace prescribed 2021-06"],
        digest=_digest(),
        report_markdown="# Report\n\nBody.",
        input_chars=120,
        truncated_chars=0,
        truncation_warning="",
    )


def _draft() -> DraftResult:
    return DraftResult(
        grounding={"supported_observations": [{"observation": "Daily knee pain."}]},
        draft="Draft body.",
        final_statement="Final body.",
        review_issues=["[Confirm: brace date]"],
        digest=_digest(),
        input_chars=80,
        truncated_chars=0,
        truncation_warning="",
    )


def _tracker() -> UsageTracker:
    tracker = UsageTracker()
    tracker.record(
        model="qwen3.7-flash",
        phase="records:digest",
        system="s",
        user="u",
        content="c",
        prompt_tokens=100,
        completion_tokens=20,
    )
    tracker.record(
        model="qwen3.7-max",
        phase="claims",
        system="s",
        user="u",
        content="c",
        prompt_tokens=50,
        completion_tokens=10,
    )
    return tracker


class TestDocumentsRoundtrip(unittest.TestCase):
    def test_documents_survive(self):
        restored = documents_from_json(documents_to_json(_docs()))
        self.assertEqual(len(restored), 2)
        self.assertEqual(restored[0].filename, "a.txt")
        self.assertEqual(restored[0].full_text, _docs()[0].full_text)

    def test_page_numbers_are_preserved(self):
        doc = document_from_text("a.txt", "line one\n\nline two")
        restored = documents_from_json(documents_to_json([doc]))
        self.assertEqual(restored[0].pages[0].page, 1)

    def test_empty_page_text_is_dropped_but_doc_survives(self):
        raw = [{"filename": "a.txt", "pages": [{"page": 1, "text": ""}, {"page": 2, "text": "real"}]}]
        restored = documents_from_json(raw)
        self.assertEqual(len(restored), 1)
        self.assertEqual(len(restored[0].pages), 1)
        self.assertEqual(restored[0].pages[0].page, 2)

    def test_unusable_entries_are_skipped_not_crashed(self):
        raw = [
            "not a dict",
            {"pages": [{"page": 1, "text": "no filename"}]},
            {"filename": "b.txt", "pages": []},
            {"filename": "c.txt", "pages": [{"page": 1, "text": "keep"}]},
        ]
        restored = documents_from_json(raw)
        self.assertEqual([d.filename for d in restored], ["c.txt"])

    def test_non_list_input_yields_empty(self):
        self.assertEqual(documents_from_json(None), [])
        self.assertEqual(documents_from_json("nope"), [])


class TestDigestRoundtrip(unittest.TestCase):
    def test_facts_and_counters_survive(self):
        restored = digest_from_json(digest_to_json(_digest()))
        self.assertEqual(len(restored.facts), 2)
        self.assertEqual(restored.facts[0].source, "a.txt p.1")
        self.assertEqual(restored.facts[1].quote, "brace")
        self.assertEqual(restored.conditions, ["knee pain"])
        self.assertEqual(restored.providers, ["Dr. Smith (ortho)"])
        self.assertEqual(restored.pages_reviewed, 2)
        self.assertEqual(restored.duplicates_skipped, 3)

    def test_none_roundtrips_to_none(self):
        self.assertIsNone(digest_to_json(None))
        self.assertIsNone(digest_from_json(None))

    def test_missing_fields_default_cleanly(self):
        restored = digest_from_json({"facts": [{"description": "only a description"}]})
        self.assertEqual(len(restored.facts), 1)
        self.assertEqual(restored.facts[0].date, "")
        self.assertEqual(restored.pages_reviewed, 0)

    def test_bad_fact_entries_are_skipped(self):
        restored = digest_from_json({"facts": ["nope", 42, {"description": "ok"}]})
        self.assertEqual(len(restored.facts), 1)


class TestResultRoundtrip(unittest.TestCase):
    def test_evaluation_survives_with_derived_values_intact(self):
        original = _evaluation()
        restored = evaluation_from_json(evaluation_to_json(original))
        self.assertEqual(restored.claims, original.claims)
        self.assertEqual(restored.verifications, original.verifications)
        self.assertEqual(restored.scores, original.scores)
        self.assertEqual(restored.revised_statement, original.revised_statement)
        self.assertEqual(restored.report_markdown, original.report_markdown)
        self.assertEqual(restored.added_facts_to_verify, original.added_facts_to_verify)
        self.assertEqual(restored.topic_critical_gaps, original.topic_critical_gaps)
        # Derived properties must recompute identically from restored data.
        self.assertEqual(restored.contradiction_count, 1)
        self.assertEqual(restored.overall_rating, original.overall_rating)

    def test_draft_survives(self):
        original = _draft()
        restored = draft_from_json(draft_to_json(original))
        self.assertEqual(restored.grounding, original.grounding)
        self.assertEqual(restored.final_statement, original.final_statement)
        self.assertEqual(restored.review_issues, original.review_issues)
        self.assertEqual(restored.output_statement, original.output_statement)

    def test_usage_survives(self):
        tracker = usage_from_json(usage_to_json(_tracker()))
        self.assertEqual(len(tracker.entries), 2)
        self.assertEqual(tracker.totals().calls, 2)
        self.assertEqual(tracker.totals().prompt_tokens, 150)
        # Per-role split drives the credit estimate, so it must survive too.
        self.assertEqual(tracker.per_role_tokens(), {"main": 60, "fast": 120})

    def test_usage_from_garbage_is_empty_not_broken(self):
        self.assertEqual(usage_from_json(None).totals().calls, 0)
        self.assertEqual(usage_from_json({"entries": "nope"}).totals().calls, 0)

    def test_int_scores_are_coerced_to_float(self):
        restored = evaluation_from_json({"scores": {"factual_accuracy": 7}})
        self.assertEqual(restored.scores, {"factual_accuracy": 7.0})


class TestJobEnvelope(unittest.TestCase):
    def test_evaluate_job_roundtrip(self):
        job = EvaluateJob(statement_text="Statement text.", records=_docs(), request_id="req_1")
        restored = decode_job(KIND_EVALUATE, encode_job(KIND_EVALUATE, job))
        self.assertEqual(restored.statement_text, "Statement text.")
        self.assertEqual(restored.request_id, "req_1")
        self.assertEqual(len(restored.records), 2)

    def test_draft_job_roundtrip(self):
        job = DraftJob(
            records=_docs(),
            witness={"name": "Jane Doe", "relationship": "Spouse"},
            observations="Bullet points.",
            condition="knee strain",
            claim_type="Service connection (new claim)",
            request_id="req_2",
        )
        restored = decode_job(KIND_DRAFT, encode_job(KIND_DRAFT, job))
        self.assertEqual(restored.witness["name"], "Jane Doe")
        self.assertEqual(restored.condition, "knee strain")
        self.assertEqual(restored.claim_type, "Service connection (new claim)")

    def test_record_sources_survive_for_audit(self):
        job = EvaluateJob(
            statement_text="s",
            records=_docs(),
            record_sources=["Upload", "VA.gov"],
        )
        restored = decode_job(KIND_EVALUATE, encode_job(KIND_EVALUATE, job))
        self.assertEqual(restored.record_sources, ["Upload", "VA.gov"])

    def test_page_count_helper(self):
        job = EvaluateJob(statement_text="s", records=_docs())
        self.assertEqual(payload_page_count(job), 2)

    def test_kind_mismatch_is_rejected(self):
        job = EvaluateJob(statement_text="s", records=_docs())
        encoded = encode_job(KIND_EVALUATE, job)
        with self.assertRaises(PayloadError):
            decode_job(KIND_DRAFT, encoded)

    def test_missing_statement_is_rejected(self):
        with self.assertRaises(PayloadError):
            decode_job(KIND_EVALUATE, json.dumps({"kind": KIND_EVALUATE, "documents": documents_to_json(_docs())}))

    def test_missing_records_is_rejected(self):
        with self.assertRaises(PayloadError):
            decode_job(KIND_EVALUATE, json.dumps({"kind": KIND_EVALUATE, "statement_text": "s"}))

    def test_corrupt_json_is_rejected_clearly(self):
        with self.assertRaises(PayloadError):
            decode_job(KIND_EVALUATE, "{not json")
        with self.assertRaises(PayloadError):
            decode_job(KIND_EVALUATE, "[1, 2, 3]")

    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(PayloadError):
            decode_job("verdict", json.dumps({}))

    def test_oversize_payload_is_refused_before_it_reaches_redis(self):
        job = EvaluateJob(statement_text="s" * 5000, records=_docs())
        with patch.object(config, "JOB_QUEUE_MAX_PAYLOAD_BYTES", 512):
            with self.assertRaises(PayloadTooLargeError):
                encode_job(KIND_EVALUATE, job)

    def test_unserializable_payload_is_a_payload_error(self):
        job = EvaluateJob(statement_text="s", records=_docs())
        job.records[0].pages[0].text = object()  # type: ignore[assignment]
        with self.assertRaises(PayloadError):
            encode_job(KIND_EVALUATE, job)


class TestResultEnvelope(unittest.TestCase):
    def test_evaluate_result_roundtrip(self):
        run = RunResult(
            kind=KIND_EVALUATE, result=_evaluation(), usage=_tracker(), request_id="req_1"
        )
        restored = decode_result(encode_result(run))
        self.assertEqual(restored.kind, KIND_EVALUATE)
        self.assertEqual(restored.request_id, "req_1")
        self.assertEqual(restored.result.contradiction_count, 1)
        self.assertEqual(restored.result.digest.pages_reviewed, 2)
        self.assertEqual(restored.usage.totals().calls, 2)

    def test_draft_result_roundtrip(self):
        run = RunResult(kind=KIND_DRAFT, result=_draft(), usage=_tracker())
        restored = decode_result(encode_result(run))
        self.assertEqual(restored.kind, KIND_DRAFT)
        self.assertEqual(restored.result.output_statement, "Final body.")
        self.assertEqual(restored.usage.totals().calls, 2)

    def test_unknown_result_kind_is_rejected(self):
        with self.assertRaises(PayloadError):
            decode_result(json.dumps({"kind": "verdict", "result": {}}))

    def test_corrupt_result_is_rejected(self):
        with self.assertRaises(PayloadError):
            decode_result("{oops")

    def test_payload_version_is_recorded(self):
        run = RunResult(kind=KIND_EVALUATE, result=_evaluation(), usage=_tracker())
        self.assertEqual(json.loads(encode_result(run))["version"], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
