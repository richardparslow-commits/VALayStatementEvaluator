"""Offline unit tests: relevance search and evaluation report logic."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.config import load_knowledge  # noqa: E402
from app.documents import extract_document  # noqa: E402
from app.draft import DraftResult, grounding_markdown  # noqa: E402
from app.evaluate import EvaluationResult, build_report  # noqa: E402
from app.fetch_client import FetchClient  # noqa: E402
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

    def test_fetch_documents_work_with_existing_record_search(self):
        settings = Settings(
            api_key="",
            base_url="https://llm.example",
            model_main="main-model",
            model_fast="fast-model",
            fetch_api_key="sandbox-token",
            fetch_base_url="https://demo.fetchsandbox.com",
            fetch_records_path="/medical_records/{patient_id}",
        )
        client = FetchClient(settings)
        documents = client._normalize_payload(
            {
                "documents": [
                    {
                        "name": "records",
                        "text": "Veteran reports insomnia and panic attacks after deployment.",
                    }
                ]
            },
            "pt-6",
        )
        excerpts = find_relevant_excerpts(documents, "panic attacks insomnia")
        self.assertIn("panic attacks", excerpts)


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
            revision_notes="Corrected the date to match the STRs.",
            revision_changes=[
                {
                    "category": "contradiction_fix",
                    "original": "treated in 2009",
                    "revised": "treated in 2010 [Confirm: STRs show 2010]",
                    "reason": "Records show 2010 treatment.",
                }
            ],
            revised_statement="I was treated in 2010. [Confirm: exact date]",
            added_facts_to_verify=["2010 cortisone injection"],
            topic_focus="PTSD with need for regular assistance (spouse statement)",
            topic_rows=[
                {
                    "topic": "A. Hazards and Dangers",
                    "applicable": True,
                    "coverage": "partial",
                    "evidence": "He sometimes forgets the stove.",
                    "gap_note": "Describe a specific near-miss incident.",
                },
                {
                    "topic": "J. Physical Side Effects",
                    "applicable": False,
                    "coverage": "not applicable",
                    "evidence": "",
                    "gap_note": "",
                },
            ],
            topic_critical_gaps=["B. Caregiver Burden — not addressed"],
            topic_notes="Good base, but hazards need concrete incidents.",
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
                        "Top Improvements", "CONTRADICTED", "not legal",
                        "Suggested Improvements", "Proposed Rewrite", "Topic Coverage"]:
            self.assertIn(section, report, msg=f"missing: {section}")

    def test_report_renders_topic_coverage_rows(self):
        report = build_report(self._sample(), "statement body")
        self.assertIn("A. Hazards and Dangers", report)
        self.assertIn("partial", report)
        self.assertIn("He sometimes forgets the stove.", report)
        self.assertIn("Describe a specific near-miss incident.", report)
        self.assertIn("Critical gaps", report)
        self.assertIn("B. Caregiver Burden — not addressed", report)
        self.assertIn("PTSD with need for regular assistance", report)

    def test_report_topic_coverage_optional(self):
        result = self._sample()
        result.topic_rows = []
        result.topic_focus = ""
        report = build_report(result, "statement body")
        self.assertNotIn("Topic Coverage", report)

    def test_report_includes_revised_statement_and_confirm_flags(self):
        report = build_report(self._sample(), "statement body")
        self.assertIn("treated in 2010", report)
        self.assertIn("[Confirm: exact date]", report)
        self.assertIn("contradiction_fix", report)
        self.assertIn("2010 cortisone injection", report)


class TestGroundingMarkdown(unittest.TestCase):
    def test_topic_coverage_rendered(self):
        result = DraftResult(
            grounding={
                "supported_observations": [],
                "unverified_observations": [],
                "conflicts": [],
                "strengthening_questions": [],
                "topic_coverage": [
                    {"topic": "A. Hazards and Dangers", "applicable": True, "covered": True,
                     "prompt_for_witness": ""},
                    {"topic": "B. Caregiver Burden", "applicable": True, "covered": False,
                     "prompt_for_witness": "What happens if you are away for a day?"},
                    {"topic": "J. Physical Side Effects", "applicable": False, "covered": False,
                     "prompt_for_witness": ""},
                ],
            }
        )
        md = grounding_markdown(result)
        self.assertIn("Topic coverage", md)
        self.assertIn("A. Hazards and Dangers", md)
        self.assertIn("B. Caregiver Burden", md)
        self.assertIn("What happens if you are away for a day?", md)
        self.assertNotIn("J. Physical Side Effects", md)


class TestKnowledgeBase(unittest.TestCase):
    def test_all_knowledge_files_load(self):
        for name in ["legal_framework.md", "evaluation_rubric.md", "drafting_guide.md",
                     "topic_checklist.md"]:
            text = load_knowledge(name)
            self.assertGreater(len(text), 500, msg=f"{name} unexpectedly small")

    def test_topic_checklist_covers_required_topics(self):
        checklist = load_knowledge("topic_checklist.md")
        for topic in ["Hazards and Dangers", "Caregiver Burden", "Personal Care and Hygiene",
                      "Medication and Financial Management", "Household Safety",
                      "Routine Errands", "Symptom Progression", "Observable Behaviors",
                      "Family Dynamics", "Physical Side Effects",
                      "Functional Impairments from Medication", "Formatting and Certification",
                      "double-dosing", "stove", "before", "after"]:
            self.assertIn(topic, checklist, msg=f"checklist missing: {topic}")


if __name__ == "__main__":
    unittest.main()
