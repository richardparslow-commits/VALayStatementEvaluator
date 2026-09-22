"""Offline unit tests: relevance search and evaluation report logic."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app.config import Settings  # noqa: E402
from app.config import load_knowledge  # noqa: E402
from app.documents import extract_document  # noqa: E402
from app.draft import DraftResult, grounding_markdown  # noqa: E402
from app.evaluate import EvaluationResult, build_report  # noqa: E402
from app.fetch_client import FetchClient  # noqa: E402
from app.medical_review import find_relevant_excerpts, retrieve_evidence  # noqa: E402


class TestRelevanceSearch(unittest.TestCase):
    def test_finds_matching_paragraph(self):
        doc = extract_document(
            "records.txt",
            b"Patient reports low back pain after lifting a pallet at work.\n\n"
            b"Unrelated note about scheduling and parking availability.",
        )
        excerpts = find_relevant_excerpts([doc], "low back pain lifting pallet work")
        self.assertIn("low back pain", excerpts)
        # Retrieval now ranks instead of filtering: the whole record is offered to
        # the verifier so it always has context, and the matching paragraph must
        # come first. The relevance *signal* is what keeps the rest honest.
        self.assertLess(excerpts.index("low back pain"), excerpts.index("parking"))
        evidence = retrieve_evidence([doc], "low back pain lifting pallet work")
        self.assertFalse(evidence.weak)
        self.assertGreater(evidence.best_overlap, 0.0)

    def test_a_query_the_records_do_not_cover_is_flagged_weak(self):
        doc = extract_document(
            "records.txt", b"Patient reports low back pain after lifting a pallet.\n\n"
            b"Unrelated note about scheduling and parking availability.",
        )
        evidence = retrieve_evidence([doc], "torn rotator cuff impingement")
        self.assertTrue(evidence.weak)
        self.assertEqual(evidence.best_overlap, 0.0)
        # Context is still returned — the caller needs it to explain *why* the
        # claim is unverified, and must never receive an empty prompt here.
        self.assertTrue(evidence.text)

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

    def test_the_absence_rule_names_the_right_authority(self):
        """Barr is a lay-competence case; the absence rule is Horn / M21-1 / Buchanan.

        These two propositions sit next to each other in the framework, and swapping
        them would put a citation in a claim file that does not say what it is cited
        for.
        """
        framework = load_knowledge("legal_framework.md")
        self.assertIn("Buczynski v. Shinseki", framework)
        self.assertIn("Horn v. Shinseki", framework)
        self.assertIn("M21-1", framework)
        self.assertIn("Absence of evidence vs. negative evidence", framework)
        lines = framework.splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("- **Barr v. Nicholson"))
        bullet: list[str] = []
        for line in lines[start:]:
            if bullet and line.startswith("- "):
                break
            bullet.append(line)
        barr = " ".join(bullet)
        self.assertNotIn("negative evidence", barr.lower())
        self.assertIn("readily observable", barr)
        self.assertIn("medical in nature", barr)

    def test_the_framework_does_not_end_mid_sentence(self):
        """A truncated rule reads to the model as a weighing instruction.

        The absence bullet was split across the file: the sentence stopped at "is not",
        and its tail sat under "Weighing factors adjudicators apply". The rule the
        rubric cites must exist in one piece.
        """
        framework = load_knowledge("legal_framework.md")
        self.assertTrue(framework.rstrip().endswith("."))
        self.assertNotIn("\n\n  affirmative evidence against the claim", framework)
        # The combat-veteran sentence belongs to the absence rule, not to weighting.
        self.assertLess(
            framework.index("1154(b)"), framework.index("## Lay competence boundaries")
        )

    def test_the_rubric_requires_a_citation_for_a_contradiction(self):
        rubric = load_knowledge("evaluation_rubric.md")
        self.assertIn("CONTRADICTED requires affirmative contrary evidence", rubric)
        self.assertIn("If you cannot point to\n     the specific record text that conflicts", rubric)
        self.assertIn("Do not lower this dimension for unverifiable claims", rubric)
        self.assertIn("Never describe a NOT FOUND claim as unverified", rubric)

    def test_the_rubric_keeps_a_normal_static_exam_from_contradicting_a_symptom(self):
        rubric = load_knowledge("evaluation_rubric.md")
        self.assertIn("A normal finding does not contradict a symptom", rubric)
        self.assertIn("would normally be noted or reported", rubric)
        self.assertIn("Buczynski v. Shinseki", rubric)
        self.assertIn("DeLuca v. Brown", rubric)
        self.assertIn("4.59", rubric)

    def test_the_checklist_carries_musculoskeletal_function(self):
        """A physical claim needs the functional-loss facts, not a ROM number."""
        checklist = load_knowledge("topic_checklist.md")
        self.assertIn("Musculoskeletal function", checklist)
        self.assertIn("dedicated topics of its own", checklist)
        self.assertIn("never a measured range-of-motion number", checklist)
        self.assertIn("38 C.F.R. § 4.40", checklist)

    def test_an_inapplicable_physical_topic_is_not_a_deficiency(self):
        checklist = load_knowledge("topic_checklist.md")
        self.assertIn("an inapplicable topic is \"not applicable\", not \"absent\"", checklist)
        self.assertIn("A **physical or musculoskeletal** claim is carried by", checklist)

    def test_the_checklist_has_dedicated_musculoskeletal_topics(self):
        """Painful motion, repeated-use loss, and flare-ups are their own topics.

        DeLuca/Sharp made these the rating-deciding factors, so a woven-in sentence
        inside F is too easy for an audit model to mark "partial"; each needs a
        section the coverage audit must address by name.
        """
        checklist = load_knowledge("topic_checklist.md")
        self.assertIn("## M. Painful Motion (38 C.F.R. § 4.59; DeLuca v. Brown)", checklist)
        self.assertIn("## N. Functional Loss During Repeated Use", checklist)
        self.assertIn("## O. Flare-Ups: Frequency, Duration, Severity, and Cost", checklist)
        self.assertIn("Sharp v. Shulkin, 29 Vet. App. 26 (2017)", checklist)
        self.assertIn("FUNCTIONALLY UNABLE TO DO", checklist)
        self.assertIn("38 C.F.R. § 4.40", checklist)
        # F stays the everyday-function topic; movement detail lives in M/N/O.
        self.assertIn("dedicated topics of its own", checklist)
        # The carrier map in the header points at the new letters, not the old ones.
        self.assertIn("**M** (painful motion)", checklist)
        self.assertIn("**N** (repeated-use functional loss)", checklist)
        self.assertIn("**O** (flare-ups)", checklist)

    def test_the_rubric_scores_aggravation_and_benefit_of_the_doubt(self):
        rubric = load_knowledge("evaluation_rubric.md")
        self.assertIn("Aggravation claims are a second, equally valid timeline shape", rubric)
        self.assertIn("38 C.F.R. §§ 3.310", rubric)
        self.assertIn("baseline before service", rubric)
        self.assertIn("Never fault a statement\n   for the \"missing\" onset event", rubric)
        self.assertIn("benefit-of-the-doubt rule", rubric)
        self.assertIn("38 U.S.C. § 1154(b)", rubric)
        self.assertIn("§ 3.102", rubric)
        self.assertIn("immaterial discrepancies do not destroy credibility", rubric)
        self.assertIn("not calendrical precision", rubric)


if __name__ == "__main__":
    unittest.main()
