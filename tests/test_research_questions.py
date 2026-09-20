"""Offline tests for case-derived research questions (app/research_questions.py).

This module exists so the Research tab does not open onto a blank box, which means the
tests have to pin two different things: that the *questions* are worth asking, and that
deriving and attaching them never leaks the record set.

The second half is the load-bearing one. ``case_context_block`` is the only place in the
research path where anything derived from records reaches a prompt, so the tests here
assert by construction what it must **not** contain — fact descriptions, quotes, dates,
provider names, file names — rather than only checking what it does. A regression that
widened that block would otherwise be invisible: the answer would simply look better
grounded.

No Streamlit, no network, no API key: everything here is pure logic over a digest.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app.medical_review import MedicalDigest, MedicalFact  # noqa: E402
from app.prompt_sanitize import GUARD_NOTE  # noqa: E402
from app.research_questions import (  # noqa: E402
    CASE_CONTEXT_CHAR_BUDGET,
    KIND_CONDITION,
    KIND_EVIDENCE,
    KIND_OPINION,
    MAX_CONDITION_QUESTIONS,
    MAX_EVIDENCE_QUESTIONS,
    MAX_QUESTIONS,
    case_context_block,
    case_digest,
    derive_questions,
)


def _fact(description: str, *, kind: str = "symptom", quote: str = "", date: str = "2019-04-02",
          source: str = "VA Records.pdf — page 3") -> MedicalFact:
    return MedicalFact(
        date=date, type=kind, description=description, source=source, quote=quote
    )


def _digest(
    *descriptions: str,
    conditions: list[str] | None = None,
    providers: list[str] | None = None,
    summary: str = "",
    unreadable_pages: int = 0,
    kind: str = "symptom",
) -> MedicalDigest:
    """A digest with one fact per description and the given metadata."""
    return MedicalDigest(
        facts=[_fact(text, kind=kind) for text in descriptions],
        conditions=list(conditions or []),
        providers=list(providers or []),
        summary=summary,
        pages_reviewed=len(descriptions),
        unreadable_pages=unreadable_pages,
    )


class _Result:
    """A completed run object, i.e. anything carrying a ``digest`` attribute."""

    def __init__(self, digest: MedicalDigest) -> None:
        self.digest = digest


class TestDeriveQuestions(unittest.TestCase):
    def test_no_facts_yields_no_questions(self) -> None:
        self.assertEqual(derive_questions(MedicalDigest()), [])

    def test_conditions_ranked_by_fact_support(self) -> None:
        # Two facts about sleep apnea, one about tinnitus: the better-supported condition
        # must lead, whatever order the digest listed them in.
        digest = _digest(
            "diagnosed with sleep apnea",
            "sleep apnea symptoms worsen",
            "reports tinnitus in both ears",
            conditions=["tinnitus", "sleep apnea"],
        )
        questions = derive_questions(digest)
        first = questions[0]
        self.assertEqual(first.kind, KIND_CONDITION)
        self.assertEqual(first.condition, "sleep apnea")
        self.assertIn("sleep apnea", first.text)

    def test_condition_question_is_structured_and_names_its_condition(self) -> None:
        digest = _digest("sleep apnea noted", conditions=["sleep apnea"])
        condition_qs = [q for q in derive_questions(digest) if q.kind == KIND_CONDITION]
        self.assertEqual(len(condition_qs), 1)
        self.assertTrue(condition_qs[0].structured)
        self.assertEqual(condition_qs[0].condition, "sleep apnea")

    def test_condition_without_fact_support_still_asked_but_ranks_last(self) -> None:
        digest = _digest(
            "sleep apnea diagnosed at the VA clinic",
            conditions=["sleep apnea", "hearing loss"],
        )
        questions = derive_questions(digest)
        conditions = [q for q in questions if q.kind == KIND_CONDITION]
        self.assertEqual([q.condition for q in conditions], ["sleep apnea", "hearing loss"])
        self.assertIn("no fact text mentions it", conditions[1].rationale)

    def test_duplicate_condition_spellings_do_not_duplicate_questions(self) -> None:
        digest = _digest(
            "sleep apnea diagnosed",
            conditions=["Sleep Apnea", "sleep  apnea", "SLEEP APNEA"],
        )
        conditions = [q for q in derive_questions(digest) if q.kind == KIND_CONDITION]
        self.assertEqual(len(conditions), 1)

    def test_condition_matched_by_words_not_only_exact_phrase(self) -> None:
        # The digest paraphrases: the condition list and the fact text rarely word a
        # condition identically, so a phrase-only match would rank by accident.
        digest = _digest(
            "obstructive component of the veteran's sleep apnea documented",
            conditions=["obstructive sleep apnea"],
        )
        conditions = [q for q in derive_questions(digest) if q.kind == KIND_CONDITION]
        self.assertIn("1 record fact(s)", conditions[0].rationale)

    def test_very_short_condition_strings_are_ignored(self) -> None:
        digest = _digest("routine visit", conditions=["ab", "  ", "sleep apnea"])
        conditions = [q for q in derive_questions(digest) if q.kind == KIND_CONDITION]
        self.assertEqual([q.condition for q in conditions], ["sleep apnea"])

    def test_condition_question_count_is_capped(self) -> None:
        names = [f"condition number {i}" for i in range(MAX_CONDITION_QUESTIONS + 4)]
        digest = _digest("some record text", conditions=names)
        conditions = [q for q in derive_questions(digest) if q.kind == KIND_CONDITION]
        self.assertEqual(len(conditions), MAX_CONDITION_QUESTIONS)

    def test_gap_question_for_uncovered_element(self) -> None:
        # No fact maps to nexus or to a lay-observable behavior, so both gaps are asked.
        digest = _digest(
            "diagnosed with sleep apnea", conditions=["sleep apnea"]
        )
        gaps = {q.kind: q for q in derive_questions(digest)}
        self.assertIn(KIND_EVIDENCE, gaps)
        evidence = [q for q in derive_questions(digest) if q.kind == KIND_EVIDENCE]
        texts = " ".join(q.text for q in evidence)
        self.assertIn("nexus", texts)
        self.assertTrue(all(q.condition == "sleep apnea" for q in evidence))

    def test_covered_elements_do_not_produce_gap_questions(self) -> None:
        digest = _digest(
            "at least as likely as not caused by service", kind="other"
        )
        evidence = [q for q in derive_questions(digest) if q.kind == KIND_EVIDENCE]
        self.assertNotIn("nexus", " ".join(q.text for q in evidence))

    def test_gap_questions_are_capped_and_exclude_the_catch_all_element(self) -> None:
        digest = _digest("a record fact", conditions=["sleep apnea"])
        evidence = [q for q in derive_questions(digest) if q.kind == KIND_EVIDENCE]
        self.assertLessEqual(len(evidence), MAX_EVIDENCE_QUESTIONS)
        self.assertNotIn("other evidence", " ".join(q.text for q in evidence))

    def test_opinion_question_needs_more_than_one_provider(self) -> None:
        single = _digest("a fact", providers=["Dr. A"])
        several = _digest("a fact", providers=["Dr. A", "VA Medical Center"])
        self.assertNotIn(KIND_OPINION, [q.kind for q in derive_questions(single)])
        opinions = [q for q in derive_questions(several) if q.kind == KIND_OPINION]
        self.assertEqual(len(opinions), 1)
        self.assertIn("2 provider(s)", opinions[0].rationale)

    def test_total_question_count_is_capped(self) -> None:
        digest = _digest(
            *[f"fact {i}" for i in range(5)],
            conditions=[f"condition number {i}" for i in range(20)],
            providers=["Dr. A", "Dr. B"],
        )
        self.assertLessEqual(len(derive_questions(digest)), MAX_QUESTIONS)
        self.assertEqual(len(derive_questions(digest, limit=3)), 3)

    def test_limit_of_zero_yields_nothing(self) -> None:
        self.assertEqual(derive_questions(_digest("a fact"), limit=0), [])

    def test_every_question_is_askable(self) -> None:
        digest = _digest(
            "sleep apnea diagnosed",
            "tinnitus reported",
            conditions=["sleep apnea", "tinnitus"],
            providers=["Dr. A", "Dr. B"],
        )
        questions = derive_questions(digest)
        self.assertTrue(questions)
        for question in questions:
            self.assertTrue(question.text.strip().endswith("?"), question.text)
            self.assertTrue(question.rationale.strip(), question.text)

    def test_derivation_is_deterministic(self) -> None:
        digest = _digest(
            "sleep apnea diagnosed",
            "tinnitus reported",
            conditions=["tinnitus", "sleep apnea"],
            providers=["Dr. A", "Dr. B"],
        )
        self.assertEqual(derive_questions(digest), derive_questions(digest))


class TestCaseDigestLookup(unittest.TestCase):
    def test_reads_a_digest_off_a_result_object(self) -> None:
        digest = _digest("a fact")
        found, origin = case_digest({"eval_result": _Result(digest)})
        self.assertIs(found, digest)
        self.assertEqual(origin, "your last statement evaluation")

    def test_prefers_the_published_timeline_digest(self) -> None:
        # draft_view publishes timeline_digest for cross-tab reuse; honouring it first
        # keeps one digest authoritative instead of racing two.
        timeline = _digest("timeline fact")
        evaluated = _digest("evaluated fact")
        found, _ = case_digest(
            {"timeline_digest": timeline, "eval_result": _Result(evaluated)}
        )
        self.assertIs(found, timeline)

    def test_falls_back_to_the_draft_run(self) -> None:
        digest = _digest("a fact")
        found, origin = case_digest({"draft_result": _Result(digest)})
        self.assertIs(found, digest)
        self.assertEqual(origin, "your last draft run")

    def test_empty_session_yields_nothing(self) -> None:
        self.assertEqual(case_digest({}), (None, ""))

    def test_result_without_a_digest_is_ignored(self) -> None:
        self.assertEqual(case_digest({"eval_result": object()}), (None, ""))

    def test_digest_with_no_facts_is_not_a_case(self) -> None:
        self.assertEqual(case_digest({"eval_result": _Result(MedicalDigest())}), (None, ""))

    def test_wrong_typed_value_is_ignored_without_raising(self) -> None:
        digest = _digest("a fact")
        found, _ = case_digest(
            {"timeline_digest": "not a digest", "draft_result": _Result(digest)}
        )
        self.assertIs(found, digest)


class TestCaseContextBlock(unittest.TestCase):
    def test_carries_conditions_coverage_gaps_and_a_withheld_provider_count(self) -> None:
        digest = _digest(
            "diagnosed with sleep apnea",
            conditions=["sleep apnea"],
            providers=["Dr. Aloysius Pendergast", "VA Medical Center"],
            unreadable_pages=7,
        )
        block = case_context_block(digest)
        self.assertIn("sleep apnea", block)
        self.assertIn("no supporting fact yet", block)
        self.assertIn("2 (names withheld)", block)
        self.assertIn("7 page(s) could not be read", block)
        self.assertIn("severity and frequency of symptoms: 1", block)

    def test_never_carries_record_text_dates_names_or_files(self) -> None:
        digest = MedicalDigest(
            facts=[
                MedicalFact(
                    date="2019-04-02",
                    type="provider_visit",
                    description="Dr. Aloysius Pendergast noted the veteran cannot sleep",
                    source="Pendergast VA Records.pdf — page 3",
                    quote="patient states he sleeps two hours a night",
                )
            ],
            conditions=["sleep apnea"],
            providers=["Dr. Aloysius Pendergast"],
            summary="Summary text that must not travel",
        )
        block = case_context_block(digest)
        for leaked in (
            "Dr. Aloysius Pendergast",
            "Pendergast VA Records.pdf",
            "2019-04-02",
            "two hours a night",
            "cannot sleep",
            "Summary text",
        ):
            self.assertNotIn(leaked, block)

    def test_is_bounded_and_delimited_as_untrusted_data(self) -> None:
        digest = _digest(
            "a fact",
            conditions=[f"condition number {i}" for i in range(40)],
            providers=["Dr. A", "Dr. B"],
        )
        block = case_context_block(digest)
        self.assertIn(GUARD_NOTE, block)
        self.assertTrue(block.startswith("<<<\n"))
        self.assertIn(">>>", block)
        # The body is capped by the sanitizer, so the whole block is bounded by the budget
        # plus the delimiters and the guard note.
        self.assertLess(len(block), CASE_CONTEXT_CHAR_BUDGET + 1_000)

    def test_omits_omitted_condition_count_when_the_list_fits(self) -> None:
        digest = _digest("a fact", conditions=["sleep apnea", "tinnitus"])
        self.assertNotIn("omitted", case_context_block(digest))

    def test_notes_when_the_condition_list_is_cut_short(self) -> None:
        digest = _digest("a fact", conditions=[f"condition number {i}" for i in range(40)])
        self.assertIn("more, omitted", case_context_block(digest))

    def test_has_no_provider_line_when_no_providers_are_listed(self) -> None:
        self.assertNotIn("Providers", case_context_block(_digest("a fact")))


if __name__ == "__main__":
    unittest.main()
