"""Complete extracted evidence must survive bounded model views and persistence."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)
from app import config
from app.documents import document_from_text
from app.draft import run_draft
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


if __name__ == "__main__":
    unittest.main()
