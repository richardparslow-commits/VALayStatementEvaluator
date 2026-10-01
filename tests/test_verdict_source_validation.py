"""Synthetic regressions for uploaded-source addresses on evaluation verdicts."""
from contextlib import nullcontext
import json
import unittest
from unittest.mock import MagicMock, patch

from tests import hermetic  # noqa: F401
from tests.rubric_fixtures import scored_result_fields
from tests.topic_fixtures import topic_result_fields
from tests.test_evaluate import _FakeLLM, _fake_digest
from tests.test_views import _fake_streamlit, _patch_st
from app.documents import BLOCK, DocumentPage, ExtractedDocument, document_from_text
from app.evaluate import (
    EvaluationResult, LEGACY_REFERENCE_NOTICE, SOURCE_REFERENCE_NOTICE,
    SOURCE_REFERENCE_POLICY, VERIFICATION_MAX_ATTEMPTS, VerificationIncompleteError,
    _verify_claims, build_report, evaluation_report_markdown, run_evaluation,
)
from app.job_payload import evaluation_from_json, evaluation_to_json
from app.medical_review import MedicalDigest, MedicalFact


def _page(text="Patient reports knee pain.", *, filename="clinic.pdf", number=3, kind="page"):
    return ExtractedDocument(filename, [DocumentPage(filename, number, text, kind=kind)])


def _verdict(reference="clinic.pdf p.3", verdict="SUPPORTED", claim_id=1):
    return {"id": claim_id, "verdict": verdict, "record_reference": reference,
            "note": "Synthetic finding; original text requires human review."}


class TestVerdictSourceValidation(unittest.TestCase):
    def _verify(self, llm, records=None, claims=None, digest=None):
        return _verify_claims(
            llm, claims or [{"id": 1, "text": "Knee pain while walking."}],
            digest or MedicalDigest(facts=[MedicalFact("2020", "symptom", "Knee pain.", "clinic.pdf p.3")]),
            records if records is not None else [_page()], lambda *_: None,
        )

    def _assert_incomplete(self, reference, records=None, verdict="SUPPORTED", digest=None):
        llm = _FakeLLM(overrides={"verify": {"verifications": [_verdict(reference, verdict)]}})
        with self.assertRaises(VerificationIncompleteError) as caught:
            self._verify(llm, records=records, digest=digest)
        self.assertEqual(llm.calls.count(("chat_json", "verify")), VERIFICATION_MAX_ATTEMPTS)
        self.assertIn("valid source citations", str(caught.exception))
        return caught.exception

    def test_existing_page_accepts_all_evidence_verdicts(self):
        for verdict in ("SUPPORTED", "PARTIALLY SUPPORTED", "CONTRADICTED"):
            with self.subTest(verdict=verdict):
                llm = _FakeLLM(overrides={"verify": {"verifications": [_verdict(verdict=verdict)]}})
                rows, gaps = self._verify(llm)
                self.assertEqual(rows, [_verdict(verdict=verdict)])
                self.assertEqual(gaps, [])

    def test_single_page_aliases_are_canonicalized(self):
        for reference in ("[clinic.pdf — page 3]", "clinic.pdf - page 3",
                          "CLINIC.PDF\u00a0p.3", "clinic.pdf   —   page 3", "[clinic.pdf p.3]"):
            with self.subTest(reference=reference):
                llm = _FakeLLM(overrides={"verify": {"verifications": [_verdict(reference)]}})
                rows, _ = self._verify(llm)
                self.assertEqual(rows[0]["record_reference"], "clinic.pdf p.3")

    def test_text_blocks_keep_their_kind_and_uploaded_filename(self):
        llm = _FakeLLM(overrides={"verify": {"verifications": [_verdict("[Clinic Notes.txt — block 1]")]}})
        rows, _ = self._verify(llm, [_page("Knee pain.", filename="Clinic Notes.txt", number=1, kind=BLOCK)])
        self.assertEqual(rows[0]["record_reference"], "Clinic Notes.txt b.1")
        self._assert_incomplete("Clinic Notes.txt p.1", [_page("Knee pain.", filename="Clinic Notes.txt", number=1, kind=BLOCK)])
        self._assert_incomplete("clinic.pdf b.3")

    def test_invented_filename_or_page_is_rejected_for_every_evidence_verdict(self):
        for verdict in ("SUPPORTED", "PARTIALLY SUPPORTED", "CONTRADICTED"):
            for reference in ("absent.pdf p.999", "clinic.pdf p.999", "clinic.pdf p.1", "other.pdf p.3"):
                with self.subTest(verdict=verdict, reference=reference):
                    self._assert_incomplete(reference, verdict=verdict)

    def test_bare_filename_range_composite_and_explanation_are_not_single_citations(self):
        for reference in ("clinic.pdf", "clinic.pdf p.3-4", "clinic.pdf p.3–4",
                          "clinic.pdf p.3, clinic.pdf p.4", "clinic.pdf p.3; absent.pdf p.999",
                          "see clinic.pdf p.3", "clinic.pdf p.3 (2020 visit)",
                          "clinic.pdf p.3 2020-01-02", "[clinic.pdf p.3] and [clinic.pdf p.4]"):
            with self.subTest(reference=reference):
                self._assert_incomplete(reference)

    def test_blank_or_whitespace_evidence_citation_is_rejected(self):
        for reference in ("", " \n\t"):
            for verdict in ("SUPPORTED", "PARTIALLY SUPPORTED", "CONTRADICTED"):
                with self.subTest(reference=reference, verdict=verdict):
                    self._assert_incomplete(reference, verdict=verdict)

    def test_not_found_may_be_uncited_but_cannot_carry_a_fake_source(self):
        for reference in ("", "  ", "[clinic.pdf — page 3]"):
            with self.subTest(reference=reference):
                llm = _FakeLLM(overrides={"verify": {"verifications": [_verdict(reference, "NOT FOUND")]}})
                rows, _ = self._verify(llm)
                self.assertEqual(rows[0]["verdict"], "NOT FOUND")
                self.assertEqual(rows[0]["record_reference"], "clinic.pdf p.3" if reference.strip() else "")
        self._assert_incomplete("absent.pdf p.999", verdict="NOT FOUND")

    def test_no_uploaded_records_cannot_be_replaced_by_digest_sources(self):
        self._assert_incomplete("clinic.pdf p.3", records=[])

    def test_empty_or_explicitly_unreadable_source_is_rejected(self):
        for text in ("", " \n\t"):
            with self.subTest(text=text):
                self._assert_incomplete("clinic.pdf p.3", [_page(text)])
        doc = _page()
        doc.unreadable_pages = [3]
        self._assert_incomplete("clinic.pdf p.3", [doc])

    def test_duplicate_units_are_ambiguous_even_with_identical_text(self):
        for duplicate_text in ("Patient reports knee pain.", "Different record.", ""):
            with self.subTest(duplicate_text=duplicate_text):
                self._assert_incomplete("clinic.pdf p.3", [_page(), _page(duplicate_text)])
        doc = _page()
        doc.pages.append(DocumentPage("clinic.pdf", 3, "Another page with same address."))
        self._assert_incomplete("clinic.pdf p.3", [doc])

    def test_case_and_whitespace_alias_collisions_are_ambiguous(self):
        self._assert_incomplete("clinic.pdf p.3", [_page(), _page(filename="CLINIC.PDF")])
        self._assert_incomplete("clinic notes.pdf p.3", [
            _page(filename="clinic notes.pdf"), _page(filename="clinic  notes.pdf"),
        ])

    def test_unextracted_unreadable_duplicate_reserves_its_source_address(self):
        unreadable = ExtractedDocument("clinic.pdf", [], total_pages=3, unreadable_pages=[3])
        for records in ([_page(), unreadable], [unreadable, _page()]):
            with self.subTest(unreadable_first=records[0] is unreadable):
                self._assert_incomplete("clinic.pdf p.3", records)

    def test_mislabelled_source_cannot_be_used_under_either_filename(self):
        doc = _page()
        doc.pages[0].filename = "other.pdf"
        for reference in ("clinic.pdf p.3", "other.pdf p.3"):
            with self.subTest(reference=reference):
                self._assert_incomplete(reference, [doc])

    def test_invalid_source_kind_or_number_cannot_supply_a_citation(self):
        for number, kind in ((0, "page"), (-1, "page"), (True, "page"), (3, "unknown")):
            with self.subTest(number=number, kind=kind):
                self._assert_incomplete("clinic.pdf p.3", [_page(number=number, kind=kind)])

    def test_retry_replaces_the_entire_batch_with_corrected_citations(self):
        prompts = []
        def verify(system, user, kwargs):
            prompts.append(user)
            return {"verifications": [_verdict(claim_id=2), _verdict(
                "absent.pdf p.999" if len(prompts) == 1 else "[clinic.pdf — page 3]",
            )]}
        llm = _FakeLLM(overrides={"verify": verify})
        rows, _ = self._verify(llm, claims=[{"id": 1, "text": "Knee pain."}, {"id": 2, "text": "Pain while walking."}])
        self.assertEqual(len(prompts), 2)
        self.assertEqual([row["id"] for row in rows], [1, 2])
        self.assertEqual([row["record_reference"] for row in rows], ["clinic.pdf p.3"] * 2)
        self.assertIn("exactly one readable uploaded page", prompts[1])
        self.assertNotIn("absent.pdf p.999", prompts[1])

    def test_error_and_retry_logging_do_not_echo_private_model_citation(self):
        private = "PRIVATE-SYNTHETIC-ID-123.pdf p.999"
        with self.assertLogs("app.evaluate", level="WARNING") as logs:
            error = self._assert_incomplete(private)
        self.assertNotIn(private, str(error))
        self.assertNotIn(private, " ".join(logs.output))
        self.assertNotIn(private, str(error.__cause__))

    def test_later_invalid_batch_stops_public_pipeline_before_scoring_or_revision(self):
        for pilot_mode in (False, True):
            with self.subTest(pilot_mode=pilot_mode):
                def verify(system, user, kwargs):
                    claims = json.loads(user.split("CLAIMS TO VERIFY:\n<<<\n", 1)[1].split("\n>>>", 1)[0])
                    return {"verifications": [_verdict(
                        "a.txt p.1" if claim["id"] <= 8 else "absent.pdf p.999", claim_id=claim["id"],
                    ) for claim in claims]}
                llm = _FakeLLM(overrides={
                    "claims": {"claims": [{"id": i, "text": "Knee pain."} for i in range(1, 10)]},
                    "verify": verify,
                })
                with patch("app.pilot.enabled", return_value=pilot_mode), \
                        patch("app.pilot.action_budget", return_value=nullcontext()), \
                        patch("app.evaluate.review_medical_records", return_value=_fake_digest()), \
                        patch("app.evaluate.load_knowledge", return_value="Synthetic knowledge"):
                    with self.assertRaises(VerificationIncompleteError):
                        run_evaluation(llm, "Synthetic knee pain account.", [document_from_text("a.txt", "Knee pain.")])
                self.assertEqual(llm.calls.count(("chat_json", "verify")), 1 + VERIFICATION_MAX_ATTEMPTS)
                for phase in ("rubric", "topic", "revision", "recommendations"):
                    self.assertNotIn(("chat_json", phase), llm.calls)

    def test_successful_pipeline_records_policy_and_preserves_canonical_source(self):
        llm = _FakeLLM()
        with patch("app.evaluate.review_medical_records", return_value=_fake_digest()), \
                patch("app.evaluate.load_knowledge", return_value="Synthetic knowledge"):
            result = run_evaluation(llm, "Synthetic knee pain account.", [document_from_text("a.txt", "Knee pain.")])
        self.assertEqual(result.verification_policy, SOURCE_REFERENCE_POLICY)
        self.assertEqual(result.verifications[0]["record_reference"], "a.txt p.1")
        self.assertIn(SOURCE_REFERENCE_NOTICE, result.report_markdown)
        self.assertEqual(result.evidence_source[0]["kind"], "page")


class TestSavedVerdictSourcePolicy(unittest.TestCase):
    def test_current_policy_survives_json_round_trip(self):
        result = EvaluationResult(verification_policy=SOURCE_REFERENCE_POLICY)
        restored = evaluation_from_json(json.loads(json.dumps(evaluation_to_json(result))))
        self.assertEqual(restored.verification_policy, SOURCE_REFERENCE_POLICY)
        self.assertIn(SOURCE_REFERENCE_NOTICE, build_report(restored, "Synthetic statement."))
        self.assertNotIn(LEGACY_REFERENCE_NOTICE, build_report(restored, "Synthetic statement."))

    def test_legacy_and_unknown_policies_are_not_silently_relabelled(self):
        for raw in ({}, {"verification_policy": "future_policy"}):
            with self.subTest(raw=raw):
                result = evaluation_from_json(raw)
                self.assertEqual(result.verification_policy, raw.get("verification_policy", ""))
                self.assertIn(LEGACY_REFERENCE_NOTICE, build_report(result, "Synthetic statement."))
                self.assertNotIn(SOURCE_REFERENCE_NOTICE, build_report(result, "Synthetic statement."))

    def test_legacy_report_download_rebuilds_review_and_adds_warning_once(self):
        result = evaluation_from_json({"report_markdown": "# Historical report\nSynthetic finding."})
        report = evaluation_report_markdown(result)
        self.assertNotIn("# Historical report", report)
        self.assertIn("Scoring unavailable", report)
        self.assertEqual(report.count(LEGACY_REFERENCE_NOTICE), 1)
        self.assertEqual(result.report_markdown, "# Historical report\nSynthetic finding.")
        result.report_markdown = report
        self.assertEqual(evaluation_report_markdown(result), report)

    def test_current_report_download_preserves_original_report(self):
        result = EvaluationResult(**scored_result_fields(), **topic_result_fields(), verification_policy=SOURCE_REFERENCE_POLICY, report_markdown="# Synthetic report")
        self.assertEqual(evaluation_report_markdown(result), result.report_markdown)

    def test_ui_warns_on_legacy_results_and_the_download_contains_the_warning(self):
        import app.views.evaluate_view as view
        st_mock, _ = _fake_streamlit()
        st_mock.columns.return_value = tuple(MagicMock() for _ in range(4))
        result = EvaluationResult(claims=[{"id": 1, "text": "Synthetic account."}], report_markdown="# Historical report")
        with _patch_st(view, st_mock):
            view._render_evaluation_results(result)
        warnings = [str(call.args[0]) for call in st_mock.warning.call_args_list]
        self.assertIn(LEGACY_REFERENCE_NOTICE, warnings)
        downloads = [call.kwargs["data"] for call in st_mock.download_button.call_args_list
                     if call.kwargs.get("file_name") == "lay_statement_evaluation.md"]
        self.assertEqual(len(downloads), 1)
        self.assertIn(LEGACY_REFERENCE_NOTICE.encode(), downloads[0])

    def test_ui_caption_explains_the_limit_of_current_source_checks(self):
        import app.views.evaluate_view as view
        st_mock, _ = _fake_streamlit()
        st_mock.columns.return_value = tuple(MagicMock() for _ in range(4))
        result = EvaluationResult(verification_policy=SOURCE_REFERENCE_POLICY, claims=[{"id": 1, "text": "Synthetic account."}])
        with _patch_st(view, st_mock):
            view._render_evaluation_results(result)
        captions = [str(call.args[0]) for call in st_mock.caption.call_args_list]
        self.assertIn(SOURCE_REFERENCE_NOTICE, captions)
        warnings = [str(call.args[0]) for call in st_mock.warning.call_args_list]
        self.assertNotIn(LEGACY_REFERENCE_NOTICE, warnings)
