"""Complete extracted evidence must survive bounded model views and persistence."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)
from app import config
from app.documents import DocumentPage, ExtractedDocument, document_from_text
from app.draft import run_draft
from app.evaluate import run_evaluation
from app.exporter import export_facts_csv
from app.job_payload import digest_from_json, digest_to_json, draft_from_json, draft_to_json
from app.medical_review import MedicalDigest, MedicalFact, _merge_facts, review_medical_records


class _EvidenceLLM:
    def __init__(self, facts, *, lossy_merge=False):
        self._settings = SimpleNamespace(model_fast="fake-fast", model_main="fake-main")
        self.facts = facts
        self.lossy_merge = lossy_merge
        self.grounding_prompt = ""

    def chat_json(self, system, user, *, phase, **kwargs):
        if phase == "records:digest":
            return {"facts": [vars(f) for f in self.facts]}
        if phase == "records:merge":
            facts = json.loads(user.split("\n\n", 1)[1])
            return {"facts": facts[:1] if self.lossy_merge else facts}
        if phase == "grounding":
            self.grounding_prompt = user
            return {
                "supported_observations": [],
                "record_conflicts": [],
                "observations_needing_verification": [],
                "grounded_in_records": [],
                "unverifiable_observations": [],
                "follow_up_questions": [],
            }
        if phase == "claims":
            # Produce one claim per fact to keep verification balanced
            _FACT_TO_CLAIM_TYPE = {
                "in_service_event": "in_service_event",
                "diagnosis": "diagnosis_reference",
                "symptom": "symptom",
                "treatment": "treatment_reference",
                "medication": "treatment_reference",
                "hospitalization": "treatment_reference",
                "provider_visit": "treatment_reference",
                "functional_limitation": "functional_impact",
                "test_result": "diagnosis_reference",
            }
            return {
                "claimed_condition": "Test condition",
                "writer_role": "veteran",
                "claims": [
                    {
                        "id": i + 1,
                        "text": f.description,
                        "type": _FACT_TO_CLAIM_TYPE.get(f.type, "other"),
                    }
                    for i, f in enumerate(self.facts)
                ],
            }
        if phase == "verify":
            # Each fact's claim gets a SUPPORTED verdict
            import re
            claim_ids = set()
            for match in re.finditer(r'"id"\s*:\s*(\d+)', user):
                claim_ids.add(int(match.group(1)))
            return {
                "verifications": [
                    {
                        "id": cid,
                        "verdict": "SUPPORTED",
                        "record_reference": "clinic.pdf p.1",
                        "note": "Verified against records",
                    }
                    for cid in sorted(claim_ids)
                ]
            }
        if phase == "rubric":
            return {
                "scores": {
                    "clarity": 8, "specificity": 7, "consistency": 8,
                    "completeness": 6, "credibility": 7, "relevance": 8,
                    "timeliness": 7, "probative_value": 6,
                },
                "rationales": {k: "Adequate" for k in [
                    "clarity", "specificity", "consistency",
                    "completeness", "credibility", "relevance",
                    "timeliness", "probative_value",
                ]},
                "improvements": [],
            }
        if phase == "topic":
            return {
                "claim_focus": "Test focus",
                "topics": [],
                "critical_gaps": [],
                "notes": "",
            }
        if phase == "revision":
            return {
                "revision_notes": "",
                "changes": [],
                "revised_statement": user,
                "added_facts_to_verify": [],
            }
        if phase == "records:summary":
            return {}  # let chat() handle it
        if phase == "recommendations":
            return {"recommendations": []}
        return {}

    def chat(self, system, user, **kwargs):
        return "A bounded narrative summary or statement."


def _facts():
    facts = [
        MedicalFact("2020-01", "symptom", f"Routine assessment number {i}", "records.txt b.1")
        for i in range(1500)
    ]
    facts.append(MedicalFact(
        "2025-06", "diagnosis", "Occupational silicosis with progressive breathlessness",
        "late-record.txt b.1", "Occupational silicosis with progressive breathlessness",
        document="late-record.txt", page=1,
    ))
    return facts


class TestEvidencePreservation(unittest.TestCase):
    def test_changed_pages_reach_extraction_and_draft_grounding(self):
        class PageAwareLLM(_EvidenceLLM):
            def chat_json(self, system, user, *, phase, **kwargs):
                if phase == "records:digest":
                    return {"facts": [vars(f) for f in self.facts if f.quote in user]}
                return super().chat_json(system, user, phase=phase, **kwargs)

        template = " ".join(f"Routine clinical assessment {i}." for i in range(100))
        for first, second in [
            ("FEV1 percent predicted\n85", "FEV1 percent predicted\n45"),
            (template + "\nPatient denies chest pain.", template + "\nPatient reports chest pain."),
        ]:
            with self.subTest(first=first[-40:]):
                facts = [
                    MedicalFact("2020-01", "test", "Earlier assessment", "a.txt p.1", first, document="a.txt", page=1),
                    MedicalFact("2025-01", "test", "Later changed assessment", "b.txt p.1", second, document="b.txt", page=1),
                ]
                llm = PageAwareLLM(facts)
                docs = [
                    ExtractedDocument(name, [DocumentPage(name, 1, text)])
                    for name, text in [("a.txt", first), ("b.txt", second), ("copy.txt", first)]
                ]
                result = run_draft(
                    llm, docs, {"name": "Witness", "relationship": "Spouse"},
                    "Changed pulmonary function and chest pain", "Respiratory condition", "New claim",
                )
                self.assertEqual(result.digest.facts, facts)
                self.assertEqual(result.digest.pages_reviewed, 3)
                self.assertEqual(result.digest.duplicates_skipped, 1)
                self.assertEqual(result.digest.duplicate_pages, [
                    {"document": "copy.txt", "page": 1, "duplicate_of": "a.txt p.1"},
                ])
                self.assertIn("Later changed assessment", llm.grounding_prompt)
                self.assertIn("b.txt p.1", llm.grounding_prompt)

    def test_legacy_loss_warning_survives_reload_and_summary_merge(self):
        from app.evaluate import coverage_lines

        digest = MedicalDigest(facts=_facts()[:2], facts_dropped_by_cap=37)
        restored = digest_from_json(json.loads(json.dumps(digest_to_json(digest))))
        _merge_facts(_EvidenceLLM(restored.facts), restored)
        self.assertEqual(restored.facts_dropped_by_cap, 37)
        warning = " ".join(coverage_lines(restored))
        self.assertIn("37", warning)
        self.assertIn("saved result is incomplete", warning)

    def test_large_draft_retains_tail_evidence_even_when_merge_omits_it(self):
        for lossy in (False, True):
            with self.subTest(lossy_merge=lossy):
                facts = _facts()
                llm = _EvidenceLLM(facts, lossy_merge=lossy)
                docs = [document_from_text("records.txt", "Routine assessment history.")]
                with patch.object(config, "MAX_DIGEST_FACTS", 1500):
                    result = run_draft(
                        llm, docs, {"name": "Witness", "relationship": "Spouse"},
                        "Progressive breathlessness from occupational silicosis",
                        "Occupational silicosis", "Service connection (new claim)",
                    )
                self.assertEqual(len(result.digest.facts), 1501)
                self.assertEqual(result.digest.facts_dropped_by_cap, 0)
                self.assertEqual(result.digest.facts[-1], facts[-1])
                self.assertIn(facts[-1].description, llm.grounding_prompt)
                self.assertIn(facts[-1].source, llm.grounding_prompt)
                restored = draft_from_json(json.loads(json.dumps(draft_to_json(result))))
                self.assertEqual(restored.digest.facts, result.digest.facts)
                self.assertIn(facts[-1].description, restored.digest.relevant_facts_text("silicosis", max_facts=1))
                self.assertIn(facts[-1].description, export_facts_csv(restored.digest.facts, restored.digest).decode())

    def test_identical_descriptions_do_not_erase_distinct_quotes_or_sources(self):
        facts = [
            MedicalFact("2020", "symptom", "Breathlessness", "a.txt b.1", "while walking"),
            MedicalFact("2020", "symptom", "Breathlessness", "b.txt b.1", "at rest"),
            MedicalFact("2020", "symptom", "Breathlessness", "b.txt b.1", "during sleep"),
        ]
        digest = review_medical_records(
            _EvidenceLLM(facts + [facts[0]], lossy_merge=True),
            [document_from_text("a.txt", "Breathlessness while walking.")],
        )
        self.assertEqual(len(digest.facts), 3)
        self.assertEqual([f.quote for f in digest.facts], [f.quote for f in facts])
        self.assertEqual([f.source for f in digest.facts], [f.source for f in facts])

    def test_prompt_budgets_do_not_change_evidence_store(self):
        facts = _facts()
        digest = MedicalDigest(facts=facts.copy())
        with patch.object(config, "MAX_DIGEST_FACTS", 2):
            selected = json.loads(digest.as_json_text())
            self.assertEqual(len(selected["facts"]), 2)
            self.assertEqual(selected["total_facts"], 1501)
            self.assertTrue(selected["selection_limited"])
            prompt = digest.relevant_facts_text("silicosis", max_facts=150, budget_chars=500)
            self.assertIn("2 of 1501", prompt)
            self.assertIn(facts[-1].description, prompt)
            self.assertLessEqual(len(prompt), 500)
        self.assertEqual(digest.facts, facts)

    def test_evidence_source_preserved_independently_of_prompt_budgets(self):
        """Raw source pages are preserved in the result store, not just digest facts."""
        from app.evaluate import run_evaluation, _pages_to_source
        from app.job_payload import evaluation_from_json, evaluation_to_json

        # Build minimal records: two pages with known content
        records = [
            document_from_text("clinic.pdf", "Page one: back pain diagnosis."),
            document_from_text("imaging.pdf", "Page two: MRI shows L4-L5 herniation."),
        ]

        # Verify _pages_to_source extracts correctly
        source = _pages_to_source(records)
        self.assertEqual(len(source), 2)
        self.assertEqual(source[0]["filename"], "clinic.pdf")
        self.assertEqual(source[1]["filename"], "imaging.pdf")
        self.assertIn("back pain", source[0]["text"])
        self.assertIn("MRI", source[1]["text"])

        # Run evaluation: source must land in the result
        llm = _EvidenceLLM(_facts())
        statement = "I have chronic back pain from the herniated disc."
        result = run_evaluation(llm, statement, records)
        self.assertEqual(len(result.evidence_source), 2)
        self.assertEqual(result.evidence_source[0]["filename"], "clinic.pdf")
        self.assertIn("back pain", result.evidence_source[0]["text"])

        # Prompt budget changes must NOT affect evidence_source
        with patch.object(config, "MAX_DIGEST_FACTS", 2):
            # Re-run with a tight prompt budget
            result2 = run_evaluation(llm, statement, records)
            self.assertEqual(result2.evidence_source, result.evidence_source)
            self.assertIn("back pain", result2.evidence_source[0]["text"])

        # evidence_source must survive round-trip serialization
        serialized = evaluation_to_json(result)
        self.assertIn("evidence_source", serialized)
        self.assertEqual(len(serialized["evidence_source"]), 2)
        restored = evaluation_from_json(serialized)
        self.assertEqual(restored.evidence_source, result.evidence_source)
        self.assertEqual(restored.evidence_source[0]["filename"], "clinic.pdf")
        self.assertIn("back pain", restored.evidence_source[0]["text"])

    def test_draft_evidence_source_preserved(self):
        """Draft results also carry the raw source records."""
        from app.draft import _pages_to_source
        from app.job_payload import draft_from_json, draft_to_json

        records = [
            document_from_text("notes.txt", "Medical notes: wheezing on exertion."),
        ]
        source = _pages_to_source(records)
        self.assertEqual(len(source), 1)
        self.assertEqual(source[0]["filename"], "notes.txt")

        # Draft must preserve evidence_source
        llm = _EvidenceLLM(_facts())
        result = run_draft(
            llm, records, {"name": "Witness", "relationship": "Self"},
            "Wheezing when walking", "Asthma", "New claim",
        )
        self.assertEqual(result.evidence_source, source)
        self.assertEqual(len(result.evidence_source), 1)

        # Round-trip: draft serialization includes evidence_source
        serialized = draft_to_json(result)
        self.assertIn("evidence_source", serialized)
        restored = draft_from_json(serialized)
        self.assertEqual(restored.evidence_source, result.evidence_source)
        self.assertIn("wheezing", restored.evidence_source[0]["text"])

    def test_evidence_source_links_to_digest_facts(self):
        """The source pages contain the raw text from which digest facts were extracted."""
        from app.evaluate import run_evaluation

        # Facts that reference pages in the source records
        fact = MedicalFact(
            "2023-05", "diagnosis",
            "Diagnosis of lumbar strain",
            "clinic.pdf p.1", "lumbar strain",
            document="clinic.pdf", page=1,
        )
        llm = _EvidenceLLM([fact])
        records = [
            ExtractedDocument("clinic.pdf", [
                DocumentPage("clinic.pdf", 1, "Assessment: lumbar strain diagnosed.")
            ]),
        ]
        statement = "I have a lumbar strain from an injury."
        result = run_evaluation(llm, statement, records)

        # evidence_source carries the raw page text
        self.assertEqual(len(result.evidence_source), 1)
        raw_page = result.evidence_source[0]["text"]
        self.assertIn("lumbar strain", raw_page)

        # digest fact's document+page link back to evidence_source
        self.assertEqual(result.digest.facts[0].document, "clinic.pdf")
        self.assertEqual(result.digest.facts[0].page, 1)
        # The fact's quote is verifiable against the raw source text
        self.assertIn(result.digest.facts[0].quote, raw_page)


if __name__ == "__main__":
    unittest.main()
