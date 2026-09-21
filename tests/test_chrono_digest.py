"""Chronological evidence for the drafting pathway.

The drafted statement must read as a progression of the condition over time,
so the digest facts that feed the drafting prompts are presented earliest-first
(undated last), while selection stays relevance-ranked and the verification
pathway keeps best-evidence-first order.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session)

from app.documents import document_from_text  # noqa: E402
from app.draft import (  # noqa: E402
    DRAFT_SYSTEM_TEMPLATE,
    GROUNDING_PROMPT_MAX_CHARS,  # noqa: F401  (import guard: module still exports it)
    REVIEW_SYSTEM,
    run_draft,
)
from app.medical_review import MedicalDigest, MedicalFact  # noqa: E402
from app.config import MAX_DIGEST_FACTS  # noqa: E402


def _digest() -> MedicalDigest:
    # Deliberately jumbled: upload order is not chronological order, and the
    # knee fact (the query match) sits in the middle of it.
    return MedicalDigest(
        facts=[
            MedicalFact("2021-06", "treatment", "knee brace prescribed", "b.txt p.1"),
            MedicalFact("2019-03", "diagnosis", "tinnitus diagnosed", "a.txt p.1"),
            MedicalFact("unknown", "symptom", "undated sleep complaint", "c.txt p.1"),
            MedicalFact("2018-11", "in_service_event", "improvised explosive blast", "a.txt p.2"),
            MedicalFact("2020-01", "symptom", "knee pain while walking", "b.txt p.2"),
            MedicalFact("circa 2019", "symptom", "nightmares began", "a.txt p.3"),
        ],
        conditions=["knee pain"],
        providers=["Dr. Smith"],
        summary="Records show a knee and tinnitus history.",
        pages_reviewed=3,
        chunks_reviewed=3,
    )


class TestChronologicalRelevantFacts(unittest.TestCase):
    def test_sort_dates_presents_earliest_first_with_undated_last(self):
        text = _digest().relevant_facts_text("knee pain", sort_dates=True)
        lines = [line for line in text.splitlines() if line.startswith("[")]
        dates = [line.split("]")[0][1:] for line in lines]
        self.assertEqual(
            dates,
            ["2018-11", "2019-03", "circa 2019", "2020-01", "2021-06", "unknown"],
        )
        self.assertIn("chronological order", text.splitlines()[0])

    def test_selection_is_still_relevance_ranked(self):
        digest = MedicalDigest(
            facts=[
                MedicalFact("2000-01", "other", "unrelated dental cleaning", "x.txt p.1"),
                MedicalFact("2001-01", "symptom", "knee pain on stairs", "x.txt p.2"),
                MedicalFact("2002-01", "other", "routine eye exam", "x.txt p.3"),
                MedicalFact("2003-01", "symptom", "knee pain after walking", "x.txt p.4"),
            ]
        )
        text = digest.relevant_facts_text("knee pain", max_facts=2, sort_dates=True)
        self.assertIn("2 of 4", text)
        # Both matching facts are kept; the zero-overlap fillers are not.
        self.assertIn("knee pain on stairs", text)
        self.assertIn("knee pain after walking", text)
        self.assertNotIn("dental cleaning", text)
        self.assertNotIn("eye exam", text)
        # ...and the two kept facts are presented chronologically.
        body = [line for line in text.splitlines() if line.startswith("[")]
        self.assertLess(body[0].index("2001-01"), body[0].index("symptom"))
        self.assertTrue(body[0].startswith("[2001-01]"))
        self.assertTrue(body[1].startswith("[2003-01]"))

    def test_same_date_ties_break_toward_relevance_order(self):
        digest = MedicalDigest(
            facts=[
                MedicalFact("2020-01", "symptom", "knee", "x.txt p.1"),
                MedicalFact("2020-01", "symptom", "knee pain while walking", "x.txt p.2"),
            ]
        )
        text = digest.relevant_facts_text("knee pain while walking", sort_dates=True)
        body = [line for line in text.splitlines() if line.startswith("[")]
        # Same date, so the chronological key ties and relevance decides:
        # the stronger match is presented first.
        self.assertIn("knee pain while walking", body[0])

    def test_default_order_stays_best_match_first(self):
        text = _digest().relevant_facts_text("knee pain")
        body = [line for line in text.splitlines() if line.startswith("[")]
        self.assertIn("knee pain while walking", body[0])
        self.assertNotIn("chronological order", text.splitlines()[0])

    def test_budget_accounting_uses_the_longer_chronological_header(self):
        digest = MedicalDigest(
            facts=[MedicalFact(f"2020-01-{d:02d}", "other", f"fact {d} " + "x" * 60, "s p.1") for d in range(1, 10)]
        )
        text = digest.relevant_facts_text("fact", max_facts=9, budget_chars=420, sort_dates=True)
        self.assertLessEqual(len(text), 420)

    def test_limit_respects_the_configured_cap(self):
        digest = MedicalDigest(
            facts=[MedicalFact("2020-01", "other", f"fact {i}", "s p.1") for i in range(MAX_DIGEST_FACTS + 10)]
        )
        text = digest.relevant_facts_text("fact", max_facts=10_000, sort_dates=True)
        body = [line for line in text.splitlines() if line.startswith("[")]
        self.assertEqual(len(body), MAX_DIGEST_FACTS)


class TestChronologicalCondensedTimeline(unittest.TestCase):
    def test_small_digest_reads_chronologically(self):
        lines = _digest().condensed_timeline().splitlines()
        dates = [line.split("]")[0][1:] for line in lines]
        self.assertEqual(
            dates,
            ["2018-11", "2019-03", "circa 2019", "2020-01", "2021-06", "unknown"],
        )

    def test_stride_samples_across_time_not_document_order(self):
        # Upload order runs newest -> oldest; the stride must follow the
        # date-sorted order and still end on the earliest dated fact.
        facts = [MedicalFact(str(3000 - i), "other", f"fact-{i}", "src") for i in range(1000)]
        text = MedicalDigest(facts=facts).condensed_timeline(max_entries=100)
        lines = text.splitlines()
        dates = [int(line.split("]")[0][1:]) for line in lines]
        self.assertEqual(dates, sorted(dates))
        self.assertEqual(dates[0], 2001)
        self.assertEqual(dates[-1], 3000)

    def test_undated_facts_land_last_after_sorting(self):
        facts = [MedicalFact("unknown", "other", "undated", "s p.1")] + [
            MedicalFact(str(y), "other", f"y{y}", "s p.1") for y in range(2000, 2005)
        ]
        text = MedicalDigest(facts=facts).condensed_timeline()
        self.assertTrue(text.splitlines()[-1].startswith("[unknown]"))


# ------------------------------------------------------------------ draft wiring


class _FakeLLM:
    fast_model = "fake-fast"

    def __init__(self, overrides: dict | None = None):
        self._settings = MagicMock(model_fast="fake-fast", model_main="fake-main")
        self.overrides = overrides or {}
        self.calls: list[tuple[str, str, str]] = []  # (kind, phase, user)

    def chat_json(self, system, user, **kwargs):
        phase = kwargs.get("phase", "general")
        self.calls.append(("chat_json", phase, user))
        if phase in self.overrides:
            val = self.overrides[phase]
            return val(system, user, kwargs) if callable(val) else val
        if phase == "grounding":
            return {
                "supported_observations": [],
                "unverified_observations": [],
                "conflicts": [],
                "strengthening_questions": [],
                "suggested_inclusions": [],
                "topic_coverage": [],
            }
        if phase == "review":
            return {"issues_found": [], "improved_statement": "Improved statement long enough to satisfy the preservation checks with all required elements and certification. [Confirm: brace date]"}
        if phase == "records:digest":
            return {"facts": [], "conditions_mentioned": [], "providers_and_facilities": [], "notes": ""}
        if phase == "records:merge":
            return {"facts": []}
        return {}

    def chat(self, system, user, **kwargs):
        phase = kwargs.get("phase", "general")
        self.calls.append(("chat", phase, user))
        if phase == "records:summary":
            return "Summary."
        if phase == "draft":
            return "Draft statement. [Confirm: brace date]"
        return "Summary."


WITNESS = {"name": "Jane Doe", "relationship": "Spouse", "known_since": "2010",
           "contact_frequency": "daily", "veteran_name": "John Doe", "witnessed_event": "No"}


class TestDraftPromptChronology(unittest.TestCase):
    """The grounding digest the drafting model sees must be the chronological
    presentation, and the prompts must say what that order means."""

    def _run(self, llm, digest):
        with patch("app.draft.review_medical_records", return_value=digest), \
                patch("app.draft.load_knowledge", return_value="k"):
            return run_draft(llm, [document_from_text("a.txt", "note")], WITNESS,
                             "observed knee pain", "knee pain", "Increase")

    def test_grounding_receives_the_chronological_digest(self):
        llm = _FakeLLM()
        self._run(llm, _digest())
        grounding_user = next(user for kind, phase, user in llm.calls if phase == "grounding")
        section = grounding_user.split("MEDICAL RECORD DIGEST (JSON):", 1)[1]
        dates = [line.split("]")[0][1:] for line in section.splitlines() if line.startswith("[")]
        self.assertEqual(
            dates,
            ["2018-11", "2019-03", "circa 2019", "2020-01", "2021-06", "unknown"],
        )
        self.assertIn("chronological order", section)

    def test_the_digest_summary_prompt_reads_chronologically(self):
        # records:summary is a fast-model chat inside review_medical_records;
        # exercising it directly keeps the assertion independent of the digest
        # pipeline's concurrency.
        from app.medical_review import _summarize

        llm = _FakeLLM()
        _summarize(llm, _digest())
        summary_user = next(user for kind, phase, user in llm.calls if phase == "records:summary")
        dates = [line.split("]")[0][1:] for line in summary_user.splitlines() if line.startswith("[")]
        self.assertEqual(dates[0], "2018-11")
        self.assertEqual(dates[-1], "unknown")

    def test_the_drafting_instructions_demand_chronological_narrative(self):
        system = DRAFT_SYSTEM_TEMPLATE.format(guide="guide", checklist="checklist", credential_scope="")
        self.assertIn("chronological order", system)
        self.assertIn("earliest first", system)

    def test_the_review_pass_must_not_reorder_events(self):
        self.assertIn("chronological", REVIEW_SYSTEM)
        self.assertIn("do not reorder", REVIEW_SYSTEM)


if __name__ == "__main__":
    unittest.main()
