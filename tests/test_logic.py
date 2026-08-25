"""Offline unit tests: relevance search and evaluation report logic."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.documents import extract_document  # noqa: E402
from app.evaluate import EvaluationResult, build_report  # noqa: E402
from app.medical_review import find_relevant_excerpts  # noqa: E402


class TestRelevanceSearch(unittest.TestCase):
    def test_finds_matching_paragraph(self):
        doc = extract_document(
            "records.txt",
            b"Patient reports low back pain after lifting a pallet at work.\n\n"
            b"Unrelated note about scheduling and parking availability.",
        )
        excerpts = find_relevant_excerpts([doc], "low back pain lifting pallet work")
        self.assertIn("low back pain", excerpts)
        self.assertNotIn("parking", excerpts)

    def test_empty_query_returns_empty(self):
        doc = extract_document("records.txt", b"Some medical text about back pain.")
        self.assertEqual(find_relevant_excerpts([doc], ""), "")


class TestEvaluationResult(unittest.TestCase):
    def _sample(self):
        return EvaluationResult(
            scores={"factual_accuracy": 9, "specificity_detail": 6, "lay_competence": 8,
                    "condition_connection": 7, "continuity_timeline": 5, "functional_impact": 6,
                    "credibility_consistency": 8, "form_completeness": 4},
            rationales={"factual_accuracy": "All checkable facts matched."},
            claims=[{"id": 1, "text": "Injured back lifting pallet in 2014."}],
            verifications=[
                {"id": 1, "verdict": "SUPPORTED", "record_reference": "records p.1 2014-09",
                 "note": "Matches clinic note."},
                {"id": 2, "verdict": "CONTRADICTED", "record_reference": "records p.2",
                 "note": "Date differs."},
            ],
            improvements=[{"priority": 1, "problem": "Vague pain description",
                           "suggestion": "Add frequency", "example_rewrite": "daily pain"}],
            omitted_record_facts=[{"fact": "MRI findings", "source": "chunk 1/1"}],
            executive_summary="Solid statement.",
        )

    def test_contradiction_count(self):
        self.assertEqual(self._sample().contradiction_count, 1)

    def test_overall_rating_labels(self):
        result = self._sample()
        self.assertIn(result.overall_rating,
                      {"Excellent", "Strong", "Adequate", "Needs Substantial Work"})

    def test_report_contains_key_sections(self):
        report = build_report(self._sample(), "statement body")
        for section in ["Evaluation Report", "Claim-by-Claim Verification", "Rubric Scores",
                        "Top Improvements", "CONTRADICTED", "not legal"]:
            self.assertIn(section, report, msg=f"missing: {section}")


if __name__ == "__main__":
    unittest.main()
