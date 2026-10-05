"""A matching opening must never certify an invented ending or ambiguous page."""
from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests import hermetic  # noqa: F401 -- before app imports
from app import pilot
from app.documents import DocumentPage, ExtractedDocument
from app.evaluate import coverage_lines
from app.job_payload import digest_from_json, digest_to_json
from app.medical_review import (
    CITATION_MATCH_POLICY,
    MedicalDigest,
    MedicalFact,
    review_medical_records,
    verify_citations,
)
from app.views.evaluate_view import _render_record_coverage


OPENING = "Patient reports persistent knee pain after walking upstairs and carrying groceries"
SOURCE = OPENING + " for two years with no falls."


def document(text=SOURCE, *, filename="clinic.pdf", page=1):
    return ExtractedDocument(filename=filename, pages=[DocumentPage(filename, page, text)])


def fact(quote=SOURCE, *, filename="clinic.pdf", page=1):
    return MedicalFact(
        date="2024-01-01", type="symptom", description="Synthetic knee pain history",
        source=f"{filename} p.{page}", quote=quote, document=filename, page=page,
    )


class TestCompleteQuoteMatching(unittest.TestCase):
    def test_complete_long_quote_matches(self):
        stats = verify_citations([fact()], [document()])
        self.assertEqual(stats["match_policy"], CITATION_MATCH_POLICY)
        self.assertEqual((stats["total"], stats["verified"], stats["missing"], stats["skipped"]), (1, 1, 0, 0))

    def test_same_opening_with_invented_ending_is_not_verified(self):
        invented = OPENING + " for twenty years with frequent falls."
        stats = verify_citations([fact(invented)], [document()])
        self.assertEqual((stats["verified"], stats["missing"], stats["verified_ratio"]), (0, 1, 0.0))
        self.assertEqual(stats["examples"][0]["reason"], "quote_not_found")

    def test_changed_clinical_values_after_twelve_words_are_rejected(self):
        for original, altered in (
            ("reported on 2024-01-01.", "reported on 2024-01-02."),
            ("treated with 1.5 mg daily.", "treated with 15 mg daily."),
            ("occurring once per month.", "occurring once per day."),
            ("with no recent falls.", "with recent falls."),
        ):
            with self.subTest(original=original):
                stats = verify_citations([fact(OPENING + " " + altered)], [document(OPENING + " " + original)])
                self.assertEqual((stats["verified"], stats["missing"]), (0, 1))

    def test_punctuation_cannot_be_erased_to_create_a_match(self):
        stats = verify_citations([fact("Patient takes medication 15 mg daily")], [document("Patient takes medication 1.5 mg daily")])
        self.assertEqual(stats["missing"], 1)

    def test_long_authentic_ending_is_checked_without_a_length_cap(self):
        quote = " ".join(f"word{i}" for i in range(80))
        self.assertEqual(verify_citations([fact(quote)], [document(quote)])["verified"], 1)
        altered = quote.rsplit(" ", 1)[0] + " fabricated"
        self.assertEqual(verify_citations([fact(altered)], [document(quote)])["missing"], 1)

    def test_case_and_line_wrapping_are_allowed(self):
        stats = verify_citations([fact("PATIENT reports\npersistent   knee\tPAIN")], [document()])
        self.assertEqual(stats["verified"], 1)

    def test_unicode_letters_are_preserved(self):
        source = "Patient visited Café after treatment"
        self.assertEqual(verify_citations([fact(source.upper())], [document(source)])["verified"], 1)
        self.assertEqual(verify_citations([fact(source.replace("Café", "Cafe"))], [document(source)])["missing"], 1)

    def test_unicode_caseless_matching_in_both_directions(self):
        for source, quote in (
            ("Patient visited Straße after treatment", "PATIENT VISITED STRASSE AFTER TREATMENT"),
            ("PATIENT VISITED STRASSE AFTER TREATMENT", "Patient visited Straße after treatment"),
            ("Patient reports ΟΣ after treatment", "Patient reports ος after treatment"),
        ):
            with self.subTest(source=source):
                self.assertEqual(verify_citations([fact(quote)], [document(source)])["verified"], 1)

    def test_caseless_matching_does_not_erase_accents_or_digit_width(self):
        source = "Patient takes medication ５ mg daily"
        quote = "Patient takes medication 5 mg daily"
        self.assertEqual(verify_citations([fact(quote)], [document(source)])["missing"], 1)

    def test_short_numeric_or_hyphenated_quote_cannot_inflate_word_count(self):
        for quote in ("2024-01-02 5 mg", "2024-01-02 1.5 mg", "patient-reports-knee-pain"):
            with self.subTest(quote=quote):
                stats = verify_citations([fact(quote)], [document(quote)])
                self.assertEqual((stats["checked"], stats["verified"], stats["skipped"]), (0, 0, 1))
        # A dosage within four actual words remains checkable.
        quote = "Take 5 mg daily"
        self.assertEqual(verify_citations([fact(quote)], [document(quote)])["verified"], 1)

    def test_word_fragments_do_not_match(self):
        for quote, source in (
            ("Pain persists during walking", "NoPain persists during walking"),
            ("Pain persists during walk", "Pain persists during walking"),
        ):
            with self.subTest(quote=quote):
                self.assertEqual(verify_citations([fact(quote)], [document(source)])["missing"], 1)

    def test_partial_numbers_and_dates_at_quote_boundaries_do_not_match(self):
        for quote, source in (
            ("5 mg taken orally daily", "1.5 mg taken orally daily"),
            ("5 mg taken orally daily", "-5 mg taken orally daily"),
            (".5 mg taken orally daily", "1.5 mg taken orally daily"),
            ("Patient takes medication dose 1", "Patient takes medication dose 1.5 mg"),
            ("Patient takes medication dose 1.", "Patient takes medication dose 1.5 mg"),
            ("01-02 patient reported knee pain", "2024-01-02 patient reported knee pain"),
            ("Patient reported symptoms on 2024", "Patient reported symptoms on 2024-01-02"),
        ):
            with self.subTest(quote=quote):
                self.assertEqual(verify_citations([fact(quote)], [document(source)])["missing"], 1)
        # A number followed by sentence punctuation is a complete token.
        self.assertEqual(verify_citations([fact("Patient reports pain grade 3")], [document("Patient reports pain grade 3.")])["verified"], 1)

    def test_combining_accent_cannot_be_dropped_at_quote_boundary(self):
        self.assertEqual(verify_citations([fact("Patient visited local Cafe")], [document("Patient visited local Cafe\u0301")])["missing"], 1)
        self.assertEqual(verify_citations([fact("Patient visited local Cafe\u0301")], [document("Patient visited local Cafe\u0301")])["verified"], 1)

    def test_clinical_numeric_suffixes_cannot_be_omitted(self):
        for quote, source in (
            ("oxygen saturation was 95", "oxygen saturation was 95%"),
            ("reflex grade was recorded as 3", "reflex grade was recorded as 3+"),
            ("recorded body temperature was 37", "recorded body temperature was 37°C"),
            ("recorded body temperature was 37°", "recorded body temperature was 37°C"),
            ("reflex grade was recorded as 3+", "reflex grade was recorded as 3++"),
            ("% oxygen saturation was recorded", "95% oxygen saturation was recorded"),
            ("5 mg taken orally daily", "<5 mg taken orally daily"),
        ):
            with self.subTest(source=source):
                self.assertEqual(verify_citations([fact(quote)], [document(source)])["missing"], 1)
                self.assertEqual(verify_citations([fact(source)], [document(source)])["verified"], 1)

    def test_temperature_unit_or_uncertainty_sign_cannot_be_detached(self):
        for quote, source in (
            ("C patient was discharged", "37°C patient was discharged"),
            ("F patient was discharged", "98°F patient was discharged"),
            ("5 mg taken orally daily", "±5 mg taken orally daily"),
        ):
            with self.subTest(source=source):
                self.assertEqual(verify_citations([fact(quote)], [document(source)])["missing"], 1)
                self.assertEqual(verify_citations([fact(source)], [document(source)])["verified"], 1)

    def test_sentence_dashes_are_boundaries_but_numeric_ranges_are_not(self):
        quote = "he reports knee pain"
        for dash in ("–", "—"):
            with self.subTest(dash=dash):
                self.assertEqual(verify_citations([fact(quote)], [document("fell" + dash + quote)])["verified"], 1)
                self.assertEqual(verify_citations([fact(quote)], [document(quote + dash + "walks slowly")])["verified"], 1)
                self.assertEqual(verify_citations([fact("5 mg taken orally daily")], [document("3" + dash + "5 mg taken orally daily")])["missing"], 1)

    def test_hyphenated_terms_and_contractions_cannot_be_split(self):
        for quote, source in (
            ("weight bearing for six weeks", "non-weight bearing for six weeks"),
            ("weight bearing for six weeks", "non‐weight bearing for six weeks"),
            ("Patient was instructed to avoid weight", "Patient was instructed to avoid weight-bearing"),
            ("t take medication every day", "can't take medication every day"),
        ):
            with self.subTest(source=source):
                self.assertEqual(verify_citations([fact(quote)], [document(source)])["missing"], 1)
                self.assertEqual(verify_citations([fact(source)], [document(source)])["verified"], 1)

    def test_medical_prefixes_stay_attached_across_dash_variants(self):
        for dash in ("–", "—", "−"):
            for prefix, quote in (
                ("non", "weight bearing for six weeks"),
                ("anti", "inflammatory treatment was prescribed"),
                ("post", "operative pain persisted for weeks"),
                ("pre", "existing knee pain was documented"),
            ):
                with self.subTest(dash=dash, prefix=prefix):
                    source = prefix + dash + quote
                    self.assertEqual(verify_citations([fact(quote)], [document(source)])["missing"], 1)
                    self.assertEqual(verify_citations([fact(source)], [document(source)])["verified"], 1)
        # Do not detach the root at the other end of a quoted prefix either.
        self.assertEqual(verify_citations([fact("The patient was kept non")], [document("The patient was kept non–weight bearing")])["missing"], 1)

    def test_later_complete_occurrence_can_match_after_an_invalid_fragment(self):
        quote = "Pain persists during walking"
        source = "NoPain persists during walking; " + quote + "."
        self.assertEqual(verify_citations([fact(quote)], [document(source)])["verified"], 1)

    def test_fragments_joined_with_invented_ellipsis_do_not_match(self):
        quote = "Patient reports persistent knee pain ... with no falls."
        self.assertEqual(verify_citations([fact(quote)], [document()])["missing"], 1)
        # An ellipsis actually present in the source remains a valid literal excerpt.
        self.assertEqual(verify_citations([fact(quote)], [document(quote)])["verified"], 1)

    def test_quote_cannot_span_two_pages(self):
        docs = [ExtractedDocument(filename="clinic.pdf", pages=[
            DocumentPage("clinic.pdf", 1, "Patient reports persistent knee pain"),
            DocumentPage("clinic.pdf", 2, "with no falls."),
        ])]
        self.assertEqual(verify_citations([fact("Patient reports persistent knee pain with no falls.")], docs)["missing"], 1)

    def test_match_on_another_page_does_not_validate_the_cited_page(self):
        docs = [document("Unrelated synthetic treatment note"), document(SOURCE, page=2)]
        self.assertEqual(verify_citations([fact()], docs)["missing"], 1)

    def test_duplicate_source_addresses_are_ambiguous_in_either_order(self):
        original, different = document(), document("A different synthetic record with no knee symptoms")
        for docs in ([original, different], [different, original], [original, original]):
            with self.subTest(texts=[d.pages[0].text for d in docs]):
                stats = verify_citations([fact()], docs)
                self.assertEqual((stats["verified"], stats["missing"]), (0, 1))
                self.assertEqual(stats["examples"][0]["reason"], "source_ambiguous")

    def test_duplicate_pages_inside_one_document_are_ambiguous(self):
        doc = document()
        doc.pages.append(DocumentPage("clinic.pdf", 1, "Different synthetic page content"))
        self.assertEqual(verify_citations([fact()], [doc])["examples"][0]["reason"], "source_ambiguous")

    def test_same_text_at_different_addresses_is_not_ambiguous(self):
        docs = [document(), document(filename="copy.pdf")]
        self.assertEqual(verify_citations([fact()], docs)["verified"], 1)

    def test_unknown_short_and_unresolved_quotes_remain_visible_in_total(self):
        facts = [fact(), fact("knee"), fact(""), fact(filename="missing.pdf"), replace(fact(), document="", page=0)]
        stats = verify_citations(facts, [document()])
        self.assertEqual((stats["total"], stats["verified"], stats["checked"], stats["missing"], stats["skipped"]), (5, 1, 2, 1, 3))
        self.assertEqual(stats["verified_ratio"], 0.2)
        self.assertEqual(stats["examples"][0]["reason"], "source_not_found")
        lines = "\n".join(coverage_lines(MedicalDigest(facts=facts, citation_check=stats)))
        self.assertIn("1 of 5 fact(s) had a complete quote matched", lines)
        self.assertIn("3 too short or unresolved", lines)
        self.assertIn("source page unavailable", lines)

    def test_all_skipped_or_empty_checks_never_report_a_complete_ratio(self):
        skipped = verify_citations([fact("knee"), fact("")], [document()])
        self.assertEqual(skipped["verified_ratio"], 0.0)
        empty = verify_citations([], [document()])
        self.assertEqual((empty["total"], empty["verified"]), (0, 0))
        self.assertNotIn("verified_ratio", empty)


class TestCitationCoverage(unittest.TestCase):
    def test_prefix_check_from_saved_result_requires_rerun(self):
        for old_check in (
            {"checked": 1, "missing": 0, "skipped": 0, "verified_ratio": 1.0},
            {"match_policy": "unknown", "checked": 1, "verified": 1, "total": 1},
        ):
            with self.subTest(check=old_check):
                restored = digest_from_json(digest_to_json(MedicalDigest(facts=[fact()], citation_check=old_check)))
                lines = "\n".join(coverage_lines(restored))
                self.assertIn("Re-run the source records", lines)
                self.assertNotIn("1 of 1 fact(s) had a complete quote matched", lines)

    def test_current_policy_and_counts_survive_serialization(self):
        check = verify_citations([fact(), fact("knee")], [document()])
        restored = digest_from_json(digest_to_json(MedicalDigest(facts=[fact(), fact("knee")], citation_check=check)))
        self.assertEqual(restored.citation_check, check)
        self.assertIn("1 of 2 fact(s) had a complete quote matched", "\n".join(coverage_lines(restored)))

    def test_valid_quote_does_not_verify_generated_description_or_date(self):
        inaccurate = replace(fact(), date="1900-01-01", description="Model invents a conclusion")
        stats = verify_citations([inaccurate], [document()])
        self.assertEqual(stats["verified"], 1)
        lines = "\n".join(coverage_lines(MedicalDigest(facts=[inaccurate], citation_check=stats)))
        self.assertIn("does not verify the model's interpretation, dates or factual conclusions", lines)

    def test_unresolved_or_legacy_only_checks_open_coverage_panel(self):
        checks = [verify_citations([fact("knee")], [document()]), {"checked": 1, "missing": 0}]
        for check in checks:
            with self.subTest(check=check):
                result = SimpleNamespace(digest=MedicalDigest(citation_check=check), evidence_gaps=[])
                ui = MagicMock()
                with patch("app.views.evaluate_view.st", ui), patch("app.views.evaluate_view.pilot.display") as display:
                    _render_record_coverage(result)
                ui.expander.assert_called_once_with("🧾 Record coverage & citation check", expanded=True)
                messages = "\n".join(str(call.args[0]) for call in display.call_args_list)
                self.assertIn("does not establish that the model interpreted it correctly", messages)
                self.assertTrue("too short or unresolved" in messages or "Re-run the source records" in messages)


class _SyntheticLLM:
    fast_model = "synthetic-fast"

    def __init__(self, extracted):
        self._settings = SimpleNamespace(model_fast="synthetic-fast", model_main="synthetic-main")
        self.extracted = extracted
        self.phases = []

    def chat_json(self, system, user, *, phase, **kwargs):
        self.phases.append(phase)
        if phase in {"records:digest", "records:merge"}:
            return {"facts": [vars(f) for f in self.extracted]}
        raise AssertionError(f"Unexpected synthetic model phase: {phase}")

    def chat(self, system, user, *, phase, **kwargs):
        self.phases.append(phase)
        if phase != "records:summary":
            raise AssertionError(f"Unexpected synthetic model phase: {phase}")
        return "Synthetic summary"


class TestPilotCitationGate(unittest.TestCase):
    def test_incomplete_clinical_values_or_negating_prefixes_block_pilot(self):
        for quote, source in (
            ("oxygen saturation was 95", "oxygen saturation was 95%"),
            ("weight bearing for six weeks", "non-weight bearing for six weeks"),
            ("weight bearing for six weeks", "non–weight bearing for six weeks"),
            ("weight bearing for six weeks", "non—weight bearing for six weeks"),
            ("C patient was discharged", "37°C patient was discharged"),
            ("5 mg taken orally daily", "±5 mg taken orally daily"),
        ):
            with self.subTest(source=source):
                llm = _SyntheticLLM([fact(quote)])
                with patch("app.pilot.enabled", return_value=True):
                    with self.assertRaisesRegex(pilot.PilotBlocked, "unverified citations"):
                        review_medical_records(llm, [document(source)])
                self.assertNotIn("records:summary", llm.phases)

    def test_short_date_and_dose_quote_blocks_pilot_before_summary(self):
        quote = "2024-01-02 5 mg"
        llm = _SyntheticLLM([fact(quote)])
        with patch("app.pilot.enabled", return_value=True):
            with self.assertRaisesRegex(pilot.PilotBlocked, "unverified citations"):
                review_medical_records(llm, [document("Synthetic prescription note records " + quote)])
        self.assertNotIn("records:summary", llm.phases)

    def test_invented_tail_blocks_pilot_before_summary(self):
        llm = _SyntheticLLM([fact(OPENING + " for twenty years with frequent falls.")])
        with patch("app.pilot.enabled", return_value=True):
            with self.assertRaisesRegex(pilot.PilotBlocked, "unverified citations"):
                review_medical_records(llm, [document()])
        self.assertIn("records:digest", llm.phases)
        self.assertNotIn("records:summary", llm.phases)

    def test_ambiguous_source_blocks_pilot_even_after_page_deduplication(self):
        llm = _SyntheticLLM([fact()])
        with patch("app.pilot.enabled", return_value=True):
            with self.assertRaisesRegex(pilot.PilotBlocked, "unverified citations"):
                review_medical_records(llm, [document(), document()])
        self.assertNotIn("records:summary", llm.phases)

    def test_authentic_quote_with_unchanged_description_completes_pilot_record_review(self):
        original = fact()
        original.date = "unknown"
        original.description = SOURCE
        llm = _SyntheticLLM([original])
        with patch("app.pilot.enabled", return_value=True):
            digest = review_medical_records(llm, [document()])
        self.assertEqual(digest.citation_check["verified"], 1)
        self.assertEqual(digest.citation_check["match_policy"], CITATION_MATCH_POLICY)
        self.assertEqual(digest.summary, "Synthetic summary")

    def test_quote_alone_does_not_approve_changed_meaning_or_unsupported_date(self):
        llm = _SyntheticLLM([fact()])
        with patch("app.pilot.enabled", return_value=True):
            with self.assertRaises(pilot.PilotBlocked):
                review_medical_records(llm, [document()])
        self.assertNotIn("records:summary", llm.phases)


if __name__ == "__main__":
    unittest.main()
