"""Synthetic factual-preservation regressions; no external model/service calls."""
from __future__ import annotations

import json
import unittest
from copy import deepcopy
from unittest.mock import MagicMock, patch

from tests import hermetic  # noqa: F401
from app.documents import document_from_text
from app.documents import ExtractedDocument
from app.draft import DraftResult, _pages_to_source, run_draft
from app.evaluate import EvaluationResult, _draft_revision, evaluation_report_markdown, run_evaluation
from app.factual_integrity import (FACTUAL_POLICY, MAX_SPANS, attach_review, build_context,
                                   compare, context_for_result, fingerprint, retained_inputs, review_markdown)
from app.job_payload import draft_from_json, draft_to_json, evaluation_from_json, evaluation_to_json
from app.medical_review import MedicalDigest, MedicalFact
from tests.test_views import _fake_streamlit
from tests.rubric_fixtures import scored_result_fields
from tests.topic_fixtures import topic_result_fields


ACCOUNT = "I observed knee pain around 2020."
QUOTE = "The patient denied right knee pain in June 2020."


def result(account=ACCOUNT, text=None):
    doc = document_from_text("records.txt", QUOTE)
    fact = MedicalFact("2020-06", "symptom", "Untrusted model description", "records.txt p.1", QUOTE,
                       document="records.txt", page=1)
    return DraftResult(draft=account if text is None else text, digest=MedicalDigest(facts=[fact]),
                       evidence_source=_pages_to_source([doc]),
                       factual_inputs=retained_inputs(account, {}, [doc]))


def reasons(review):
    return " ".join(review["issues"] + [issue for row in review["rows"] for issue in row["issues"]])


class TestFactualComparison(unittest.TestCase):
    def test_original_spans_are_immutable_with_exact_offsets(self):
        original = "  I saw pain.\n\nHe told me it started around 2020.  "
        r = result(original)
        first = context_for_result(r)
        second = context_for_result(r)
        self.assertEqual(first, second)
        for source in first["sources"]:
            if source["kind"] == "witness_account":
                self.assertEqual(original[source["start"]:source["end"]], source["text"])

    def test_record_ledger_uses_quotes_not_model_descriptions_or_inferred_dates(self):
        sources = context_for_result(result())["sources"]
        record = next(s for s in sources if s["kind"] == "record_quote")
        self.assertEqual(record["text"], QUOTE)
        self.assertEqual(record["quote_hash"], fingerprint(QUOTE))
        self.assertEqual(record["source_unit"], {"filename": "records.txt", "kind": "page", "number": 1})
        self.assertNotIn("Untrusted model description", json.dumps(sources))

    def test_large_multisentence_quote_is_not_duplicated_per_passage(self):
        r = result(); quote = "The patient reported knee pain. " * 300
        r.digest.facts[0].quote = quote
        r.evidence_source[0]["text"] = quote
        context = context_for_result(r)
        quotes = [s for s in context["sources"] if s["kind"] == "record_quote"]
        self.assertEqual(len(quotes), 300)
        self.assertTrue(all(s["quote_hash"] == fingerprint(quote) for s in quotes))
        self.assertLess(len(json.dumps(context)), len(quote) * 100)

    def test_missing_or_ambiguous_record_quotes_never_enter_ledger(self):
        for ambiguous in (False, True):
            r = result()
            if ambiguous:
                r.evidence_source.append({**r.evidence_source[0], "text": "Conflicting page text"})
            else:
                r.digest.facts[0].quote = "Fabricated quote"
            self.assertFalse(any(s["kind"] == "record_quote" for s in context_for_result(r)["sources"]))

    def test_unreadable_duplicate_address_remains_unusable_after_save(self):
        r = result()
        docs = [ExtractedDocument("records.txt", unreadable_pages=[1]), document_from_text("records.txt", QUOTE)]
        r.factual_inputs = retained_inputs(ACCOUNT, {}, docs)
        r.evidence_source = _pages_to_source(docs)
        for checked in (r, draft_from_json(draft_to_json(r))):
            self.assertFalse(any(s["kind"] == "record_quote" for s in context_for_result(checked)["sources"]))
        legacy = deepcopy(r); legacy.factual_inputs.pop("unreadable_units")
        self.assertIsNone(context_for_result(legacy))

    def test_unreadable_source_context_changes_invalidate_its_fingerprint(self):
        r = result()
        r.factual_inputs["unreadable_units"] = [{"filename": "scan.pdf", "kind": "page", "page": 1}]
        before = context_for_result(r)["hash"]
        r.factual_inputs["unreadable_units"][0]["page"] = 2
        self.assertNotEqual(before, context_for_result(r)["hash"])
        r.factual_inputs["unreadable_units"][0]["page"] = True
        self.assertIsNone(context_for_result(r))

    def test_unreadable_reservations_use_the_record_limit_not_the_span_limit(self):
        from app import config
        r = result()
        r.factual_inputs["unreadable_units"] = [{"filename": "scan.pdf", "kind": "page", "page": n}
                                               for n in range(1, 4002)]
        with patch.object(config, "MAX_RECORD_PAGES", 5000):
            context = context_for_result(r)
        self.assertIsNotNone(context)
        self.assertTrue(any(s["kind"] == "record_quote" for s in context["sources"]))
        self.assertEqual(compare(ACCOUNT, context)["status"], "review_required")

    def test_exact_passages_do_not_repeat_full_ledger_scans(self):
        from app import factual_integrity
        account = " ".join(f"I observed pain at event {n}." for n in range(400))
        context = context_for_result(result(account))
        calls = 0
        class CountedKey(str):
            def split(self, *args, **kwargs):
                nonlocal calls
                calls += 1
                return super().split(*args, **kwargs)
        original_key = factual_integrity._key
        with patch.object(factual_integrity, "_key", side_effect=lambda text: CountedKey(original_key(text))):
            review = compare(account, context)
        self.assertEqual(review["status"], "review_required")
        self.assertLess(calls, 3 * 400)

    def test_unchanged_uncertain_lay_account_is_eligible_only_for_human_review(self):
        r = result()
        review = compare(r.draft, context_for_result(r))
        self.assertEqual(review["status"], "review_required")
        self.assertFalse(review["issues"])
        self.assertTrue(review["rows"][0]["sources"])
        self.assertNotEqual(review["status"], "reviewed")

    def test_structured_witness_identity_and_opportunity_remain_usable(self):
        r = result(); r.factual_inputs["witness"] = {"name": "Alex", "known_since": "2010", "contact_frequency": "weekly"}
        context = context_for_result(r)
        for field, text in (("name", "My name is Alex."), ("known_since", "I have known him since 2010."),
                            ("contact_frequency", "I see him weekly.")):
            row = compare(text, context, require_account_coverage=False)["rows"][0]
            sid = next(s["id"] for s in context["sources"] if s.get("field") == field)
            linked = compare(text, context, {row["id"]: [sid]}, require_account_coverage=False)
            self.assertEqual(linked["status"], "review_required", reasons(linked))
        # A claimed condition is not firsthand evidence or a diagnosis.
        r.factual_inputs["witness"]["Claimed condition"] = "cancer"
        context = context_for_result(r)
        self.assertFalse(any(s.get("field") == "Claimed condition" for s in context["sources"]))
        linked = compare("I have cancer.", context, require_account_coverage=False)
        self.assertEqual(linked["status"], "blocked")

    def test_claim_labels_cannot_add_facts_to_a_witness_passage(self):
        r = result("I observed pain.")
        r.factual_inputs["witness"] = {"Claimed condition": "cancer", "Claim type": "service connected"}
        context = context_for_result(r)
        self.assertFalse(any(s.get("field") in r.factual_inputs["witness"] for s in context["sources"]))
        text = "I observed pain and cancer."
        row = compare(text, context)["rows"][0]
        witness = next(s["id"] for s in context["sources"] if s["kind"] == "witness_account")
        linked = compare(text, context, {row["id"]: [witness]})
        self.assertEqual(linked["status"], "blocked")
        self.assertIn("factual wording", reasons(linked))

    def test_known_since_and_original_observation_can_share_a_supported_sentence(self):
        r = result("I observed his knee pain.")
        r.factual_inputs["witness"] = {"known_since": "2010"}
        context = context_for_result(r)
        text = "I have known him since 2010 and observed his knee pain."
        row = compare(text, context)["rows"][0]
        selected = [s["id"] for s in context["sources"] if s["kind"] in {"witness_account", "witness_field"}]
        linked = compare(text, context, {row["id"]: selected})
        self.assertEqual(linked["status"], "review_required", reasons(linked))
        changed = text.replace("2010", "1995")
        row = compare(changed, context)["rows"][0]
        unsupported = compare(changed, context, {row["id"]: selected})
        self.assertEqual(unsupported["status"], "blocked")
        self.assertIn("dates/numbers", reasons(unsupported))

    def test_original_2020_to_1995_and_invented_frequency_are_specific(self):
        r = result(text="I observed knee pain in 1995 seven days per week.")
        review = compare(r.draft, context_for_result(r))
        self.assertEqual(review["status"], "blocked")
        self.assertIn("dates/numbers", reasons(review))
        self.assertIn("date precision", reasons(review))
        self.assertIn("factual wording", reasons(review))

    def test_critical_changes_cannot_be_waived_by_selecting_original_source(self):
        cases = (
            ("I observed pain in May 2020.", "I observed pain in June 2020.", "calendar dates"),
            ("I did not observe a fall.", "I observed a fall.", "negation"),
            ("He told me about knee pain.", "I observed knee pain.", "attribution"),
            ("I observed left knee pain.", "I observed right knee pain.", "laterality"),
            ("I think knee pain began around 2020.", "Knee pain began in 2020.", "uncertainty"),
            ("I observed daily knee pain.", "I observed weekly knee pain.", "frequency"),
            ("I observed knee pain.", "I was diagnosed with cancer.", "diagnosis/nexus"),
            ("I observed knee pain.", "Knee pain was caused by service.", "diagnosis/nexus"),
            ("I observed knee pain.", "I have cancer.", "factual wording"),
            ("I observed knee pain before service.", "I observed knee pain after service.", "chronology"),
            ("He reported knee pain.", "I reported knee pain.", "speaker"),
        )
        for original, changed, category in cases:
            r = result(original, changed); context = context_for_result(r)
            initial = compare(changed, context)
            sid = next(s["id"] for s in context["sources"] if s["kind"] == "witness_account")
            linked = compare(changed, context, {initial["rows"][0]["id"]: [sid]})
            with self.subTest(category=category):
                self.assertEqual(linked["status"], "blocked")
                self.assertIn(category, reasons(linked))

    def test_swapped_dates_and_quantities_keep_their_original_claim_associations(self):
        cases = (
            ("Pain began in 2020 and surgery occurred in 2021.", "Pain began in 2021 and surgery occurred in 2020."),
            ("Pain began in 2020 and swelling began in 2021.", "Pain began in 2021 and swelling began in 2020."),
            ("I saw two falls and three headaches.", "I saw three falls and two headaches."),
        )
        for original, changed in cases:
            with self.subTest(original=original):
                context = context_for_result(result(original))
                row = compare(changed, context)["rows"][0]
                sid = next(s["id"] for s in context["sources"] if s["kind"] == "witness_account")
                linked = compare(changed, context, {row["id"]: [sid]})
                self.assertEqual(linked["status"], "blocked")
                self.assertIn("number-to-claim associations", reasons(linked))
                self.assertEqual(compare(original, context)["status"], "review_required")

    def test_multi_sentence_witness_details_offer_independent_original_passages(self):
        r = result()
        first = "I observed falls in 2021."
        details = first + " I observed headaches in 2022."
        r.factual_inputs["witness"] = {"aa_daily_personal_care": details}
        context = context_for_result(r)
        fields = [s for s in context["sources"] if s["kind"] == "witness_field"]
        self.assertEqual(len(fields), 2)
        for source in fields:
            self.assertEqual(details[source["start"]:source["end"]], source["text"])
        self.assertEqual(compare(ACCOUNT + " " + first, context)["status"], "review_required")

    def test_record_negation_cannot_be_removed(self):
        r = result(); context = context_for_result(r)
        sid = next(s["id"] for s in context["sources"] if s["kind"] == "record_quote")
        text = "The medical record states the patient had right knee pain in June 2020."
        row = compare(text, context)["rows"][0]
        review = compare(text, context, {row["id"]: [sid]}, require_account_coverage=False)
        self.assertIn("negation", reasons(review))
        self.assertEqual(review["status"], "blocked")

    def test_record_passages_cannot_become_firsthand_accounts(self):
        r = result(); context = context_for_result(r)
        sid = next(s["id"] for s in context["sources"] if s["kind"] == "record_quote")
        review = compare(QUOTE, context, require_account_coverage=False)
        linked = compare(QUOTE, context, {review["rows"][0]["id"]: [sid]}, require_account_coverage=False)
        self.assertIn("firsthand", reasons(linked))

    def test_mixed_witness_and_record_sources_still_require_record_attribution(self):
        r = result("I observed knee pain.")
        quote = "Knee pain and swelling."
        r.digest.facts[0].quote = quote
        r.evidence_source[0]["text"] = quote
        context = context_for_result(r)
        selected = [s["id"] for s in context["sources"]]
        text = "I observed knee pain and knee swelling."
        row = compare(text, context)["rows"][0]
        linked = compare(text, context, {row["id"]: selected})
        self.assertEqual(linked["status"], "blocked")
        self.assertIn("firsthand", reasons(linked))

    def test_attributed_record_text_can_be_manually_linked(self):
        r = result(); context = context_for_result(r)
        sid = next(s["id"] for s in context["sources"] if s["kind"] == "record_quote")
        text = "The records state: " + QUOTE
        initial = compare(text, context, require_account_coverage=False)
        linked = compare(text, context, {initial["rows"][0]["id"]: [sid]}, require_account_coverage=False)
        self.assertEqual(linked["status"], "review_required")

    def test_omitted_original_account_blocks_approval(self):
        account = ACCOUNT + " I personally watched him fall."
        review = compare(ACCOUNT, context_for_result(result(account)))
        self.assertEqual(review["status"], "blocked")
        self.assertIn("Original witness passage 2", reasons(review))

    def test_repeated_original_passages_retain_independent_bindings(self):
        text = "I observed pain. I observed pain."
        review = compare(text, context_for_result(result(text)))
        self.assertEqual(review["status"], "review_required")
        self.assertNotEqual(review["rows"][0]["sources"], review["rows"][1]["sources"])

    def test_structural_headings_and_certification_do_not_hide_new_claims(self):
        context = context_for_result(result())
        for text in ("# Introduction", "I certify that this statement is true and correct to the best of my knowledge and belief.",
                     "# Introduction: cancer began in 1995"):
            self.assertEqual(compare(text, context)["status"], "blocked")

    def test_complete_original_statement_with_heading_and_certification_is_reviewable(self):
        account = ("# Introduction\n" + ACCOUNT + "\n"
                   "I certify that this statement is true and correct to the best of my knowledge and belief.")
        review = compare(account, context_for_result(result(account)))
        self.assertEqual(review["status"], "review_required", reasons(review))
        self.assertEqual(len(review["rows"]), 1)

    def test_placeholders_block_exact_text_approval(self):
        account = "I observed pain [Confirm: frequency]."
        review = compare(account, context_for_result(result(account)))
        self.assertIn("placeholder", reasons(review))

    def test_invalid_or_foreign_source_links_fail_closed(self):
        context = context_for_result(result())
        row = compare(ACCOUNT, context)["rows"][0]
        for ids in (["foreign"], 7, [row["sources"][0]] * 4):
            self.assertEqual(compare(ACCOUNT, context, {row["id"]: ids})["status"], "blocked")

    def test_full_source_context_change_changes_approval_binding(self):
        r = result(); previous = context_for_result(r)["hash"]
        r.evidence_source[0]["text"] += " A new unrelated source passage."
        self.assertNotEqual(previous, context_for_result(r)["hash"])

    def test_span_capacity_limit_is_explicit_not_silent(self):
        with patch("app.factual_integrity.MAX_SPANS", 1):
            context = context_for_result(result(ACCOUNT + " He reported a fall."))
        self.assertFalse(context["complete"])
        self.assertIn("comparison limit", reasons(compare(ACCOUNT, context)))

    def test_text_limit_and_malformed_original_inputs_fail_closed(self):
        for text in ("", " ", None, 7, [], object(), "x" * 120_001):
            self.assertEqual(compare(text, context_for_result(result()))["status"], "blocked")
        for inputs in ({}, {"policy": FACTUAL_POLICY, "account": 7, "witness": {}},
                       {"policy": FACTUAL_POLICY, "account": ACCOUNT, "witness": {"name": None}}):
            r = result(); r.factual_inputs = inputs
            self.assertIsNone(context_for_result(r))


class TestPipelineAndSavedReview(unittest.TestCase):
    def test_initial_draft_is_compared_with_original_not_the_generated_baseline(self):
        from tests.test_draft import _FakeLLM, _fake_digest, _doc
        changed = "I observed knee pain in 1995 seven days per week."
        llm = _FakeLLM({"draft": changed, "review": {"issues_found": [], "improved_statement": changed}})
        with patch("app.draft.review_medical_records", return_value=_fake_digest()), patch("app.draft.load_knowledge", return_value="guide"):
            r = run_draft(llm, [_doc()], {}, ACCOUNT, "Knee pain", "New claim")
        self.assertEqual(r.output_statement, changed)
        self.assertEqual(r.factual_inputs["account"], ACCOUNT)
        self.assertIn("dates/numbers", reasons(r.factual_review))
        self.assertEqual(r.factual_review["status"], "blocked")

    def test_failed_self_review_still_preserves_specific_factual_flags(self):
        from tests.test_draft import _FakeLLM, _fake_digest, _doc
        from app.llm import LLMError
        llm = _FakeLLM({"draft": "Knee pain began in 1995.", "review": LLMError("synthetic")})
        with patch("app.draft.review_medical_records", return_value=_fake_digest()), patch("app.draft.load_knowledge", return_value="guide"):
            r = run_draft(llm, [_doc()], {}, ACCOUNT, "Knee pain", "New claim")
        self.assertIn("dates/numbers", reasons(r.factual_review))
        self.assertTrue(r.review_issues)

    def test_evaluation_revision_and_proposed_edits_have_factual_comparisons(self):
        from tests.test_evaluate import _FakeLLM, _fake_digest, _doc
        changed = "I observed knee pain in 1995."
        llm = _FakeLLM({"revision": {"revised_statement": changed, "changes": [{"revised": changed}], "revision_notes": "", "added_facts_to_verify": []}})
        with patch("app.evaluate.review_medical_records", return_value=_fake_digest()):
            r = run_evaluation(llm, ACCOUNT, [_doc()])
        self.assertEqual(r.factual_inputs["account"], ACCOUNT)
        self.assertIn("dates/numbers", reasons(r.factual_review))
        self.assertIn("dates/numbers", reasons(r.factual_review["proposed_edits"][0]["comparison"]))
        self.assertNotIn("ledger", r.factual_review["proposed_edits"][0]["comparison"])
        self.assertIn("Factual comparison", evaluation_report_markdown(r))

    def test_malformed_revision_cannot_leave_stale_text_or_coerce_types(self):
        r = EvaluationResult(revised_statement="stale", **scored_result_fields(), **topic_result_fields())
        for raw in ([], {"revised_statement": 7}, {"revised_statement": "new", "changes": [{"revised": 7}]},
                    {"revised_statement": "new", "added_facts_to_verify": [7]}):
            llm = MagicMock(); llm.chat_json.return_value = raw
            _draft_revision(llm, r, ACCOUNT, lambda *a: None)
            self.assertEqual(r.revised_statement, "")
            self.assertFalse(r.revision_changes)

    def test_roundtrip_recomputes_comparison_and_preserves_originals(self):
        r = result(text="I observed knee pain in 1995.")
        payload = draft_to_json(r)
        payload["factual_review"] = {"policy": FACTUAL_POLICY, "status": "reviewed", "rows": [], "issues": []}
        restored = draft_from_json(json.loads(json.dumps(payload)))
        self.assertEqual(restored.factual_inputs, r.factual_inputs)
        self.assertEqual(restored.factual_review["status"], "blocked")
        self.assertIn("dates/numbers", reasons(restored.factual_review))

    def test_legacy_saved_draft_cannot_acquire_factual_approval(self):
        r = draft_from_json({"draft": ACCOUNT, "factual_review": {"status": "reviewed"}})
        self.assertEqual(r.factual_review["status"], "blocked")
        self.assertIn("Original inputs", reasons(r.factual_review))

    def test_cached_evaluation_rewrite_is_rebuilt_with_factual_warnings(self):
        r = EvaluationResult(revised_statement="I observed knee pain in 1995.",
                             report_markdown="APPROVED FINAL\n## Factual comparison — unreviewed output",
                             **scored_result_fields(), **topic_result_fields())
        report = evaluation_report_markdown(r)
        self.assertNotIn("APPROVED FINAL", report)
        self.assertIn("unreviewed", report)

    def test_cached_prose_without_structured_rewrite_cannot_bypass_review(self):
        r = EvaluationResult(report_markdown="APPROVED FINAL: onset 1995", **scored_result_fields(), **topic_result_fields())
        self.assertNotIn("APPROVED FINAL", evaluation_report_markdown(r))

    def test_rebuilding_display_and_saved_reports_does_not_repeat_goal_events(self):
        from app.evaluate import build_report
        r = EvaluationResult(report_citations=[{"source": "records.txt p.1", "excerpt": QUOTE}],
                             **scored_result_fields(), **topic_result_fields())
        with patch("app.evaluate.track_goal") as goal:
            build_report(r, ACCOUNT)
            goal.assert_called_once()
            goal.reset_mock()
            evaluation_report_markdown(r)
            evaluation_report_markdown(r, include_rewrite=False)
            evaluation_to_json(r)
            goal.assert_not_called()

    def test_itemized_only_edits_have_report_warnings(self):
        d = result()
        r = EvaluationResult(revision_changes=[{"revised": "Knee pain began in 1995."}],
                             factual_inputs=d.factual_inputs, digest=d.digest, evidence_source=d.evidence_source,
                             **scored_result_fields(), **topic_result_fields())
        report = evaluation_report_markdown(r)
        self.assertIn("Proposed edit 1", report)
        self.assertIn("dates/numbers", report)

    def test_unresolved_draft_disables_statement_download_and_pdf(self):
        from app.views import draft_view, factual_review
        r = result(text="I observed knee pain in 1995."); r.digest = None
        st, _ = _fake_streamlit(); st.columns.return_value = [MagicMock(), MagicMock()]
        st.text_area.return_value = r.draft
        st.multiselect.side_effect = lambda label, options, **kw: kw["default"]
        with patch.object(draft_view, "st", st), patch.object(factual_review, "st", st), \
                patch.object(draft_view, "render_follow_up_questions"), patch.object(draft_view, "render_usage_summary"), \
                patch.object(draft_view.pilot, "file_download") as download, \
                patch.object(draft_view, "_render_pdf_export") as pdf:
            draft_view._render_draft_results(r)
        self.assertEqual(download.call_count, 2)
        self.assertTrue(all(c.kwargs["disabled"] for c in download.call_args_list))
        pdf.assert_not_called()

    def test_evaluation_report_download_omits_unapproved_rewrite_and_edits(self):
        from app.views import evaluate_view, factual_review
        d = result(text="I observed knee pain in 1995.")
        r = EvaluationResult(revised_statement=d.draft, revision_changes=[{"revised": "NEW-EDIT-SENTINEL"}],
                             factual_inputs=d.factual_inputs, digest=d.digest, evidence_source=d.evidence_source,
                             **scored_result_fields(), **topic_result_fields())
        safe_report = evaluation_report_markdown(r, include_rewrite=False)
        self.assertNotIn(d.draft, safe_report)
        self.assertNotIn("NEW-EDIT-SENTINEL", safe_report)
        self.assertIn("omitted", safe_report)
        self.assertEqual(r.revised_statement, d.draft)
        st, _ = _fake_streamlit(); st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
        st.text_area.return_value = d.draft
        st.multiselect.side_effect = lambda label, options, **kw: kw["default"]
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch.object(evaluate_view, "st", st))
            stack.enter_context(patch.object(factual_review, "st", st))
            for name in ("render_usage_summary", "render_follow_up_questions", "_render_record_coverage",
                         "_render_effectiveness_score", "_render_evidence_dashboard", "_render_framework_currency_flags",
                         "_render_fact_export_section", "_render_medical_timeline"):
                stack.enter_context(patch.object(evaluate_view, name))
            stack.enter_context(patch.object(evaluate_view, "_result_reference", return_value=""))
            download = stack.enter_context(patch.object(evaluate_view.pilot, "file_download"))
            pdf = stack.enter_context(patch.object(evaluate_view, "_render_pdf_export"))
            evaluate_view._render_evaluation_results(r)
        statement_calls = [c for c in download.call_args_list if "revised statement" in c.args[0]]
        self.assertEqual(len(statement_calls), 2)
        self.assertTrue(all(c.kwargs["disabled"] for c in statement_calls))
        report_call = next(c for c in download.call_args_list if "evaluation report" in c.args[0])
        self.assertNotIn(d.draft.encode(), report_call.kwargs["data"])
        self.assertNotIn(b"NEW-EDIT-SENTINEL", report_call.kwargs["data"])
        pdf.assert_not_called()

    def test_saved_evaluation_factual_review_is_recomputed(self):
        d = result(text="I observed knee pain in 1995.")
        r = EvaluationResult(revised_statement=d.draft, factual_inputs=d.factual_inputs,
                             digest=d.digest, evidence_source=d.evidence_source,
                             **scored_result_fields(), **topic_result_fields())
        payload = evaluation_to_json(r); payload["factual_review"] = {"status": "reviewed"}
        restored = evaluation_from_json(payload)
        self.assertEqual(restored.factual_review["status"], "blocked")

    def test_saved_queue_results_omit_recomputable_comparison_ledgers(self):
        from app import config
        from app.job_payload import encode_result, decode_result, RunResult, KIND_DRAFT, KIND_EVALUATE
        from app.usage import UsageTracker
        d = result(); d.factual_review = {"redundant_private_ledger": "x" * 100_000}
        e = EvaluationResult(revised_statement=ACCOUNT, factual_inputs=d.factual_inputs,
                             digest=d.digest, evidence_source=d.evidence_source,
                             factual_review=d.factual_review, **scored_result_fields(), **topic_result_fields())
        for kind, candidate in ((KIND_DRAFT, d), (KIND_EVALUATE, e)):
            with self.subTest(kind=kind), patch.object(config, "JOB_QUEUE_MAX_PAYLOAD_BYTES", 50_000):
                encoded = encode_result(RunResult(kind=kind, result=candidate, usage=UsageTracker()))
                payload = json.loads(encoded)["result"]
                self.assertNotIn("factual_review", payload)
                self.assertNotIn("redundant_private_ledger", encoded)
                restored = decode_result(encoded).result
                self.assertEqual(restored.factual_review["status"], "review_required")
        self.assertNotIn(ACCOUNT, evaluation_to_json(e)["report_markdown"])

    def test_batch_current_and_legacy_final_have_explicit_unreviewed_comparisons(self):
        from tests.test_batch_draft import batch_draft
        r = result(text="Knee pain began in 1995.")
        final = {"statement": r.draft, "factual_inputs": r.factual_inputs, "factual_evidence_source": r.evidence_source,
                 "factual_digest": draft_to_json(r)["digest"], "factual_review": {"status": "reviewed"}}
        current = batch_draft.saved_factual_review(final)
        self.assertIn("dates/numbers", reasons(current))
        legacy = batch_draft.saved_factual_review({"statement": ACCOUNT})
        self.assertEqual(legacy["status"], "blocked")


class TestExactHumanReview(unittest.TestCase):
    def setUp(self):
        from app.views import factual_review
        self.view = factual_review
        self.st, self.session = _fake_streamlit()
        self.st.multiselect.side_effect = lambda label, options, **kw: kw["default"]
        self.st.checkbox.side_effect = lambda label, **kw: self.session.get(kw["key"], False)
        self.stack = __import__("contextlib").ExitStack()
        self.stack.enter_context(patch.object(self.view, "st", self.st))
        self.stack.enter_context(patch.object(self.view.pilot, "enabled", return_value=False))
        self.addCleanup(self.stack.close)

    def approve(self, r):
        self.assertFalse(self.view.render_factual_review(r, r.draft, slot="draft"))
        self.session[self.st.checkbox.call_args.kwargs["key"]] = True
        self.assertTrue(self.view.render_factual_review(r, r.draft, slot="draft"))

    def test_source_comparison_requires_explicit_human_approval(self):
        r = result()
        self.approve(r)
        receipt = self.session["factual_draft_receipt"]
        self.assertEqual(receipt["text_hash"], compare(r.draft, context_for_result(r))["text_hash"])
        self.assertNotIn("text", receipt)

    def test_unchanged_source_choices_reuse_the_same_comparison(self):
        with patch.object(self.view, "compare", wraps=compare) as checked:
            self.assertFalse(self.view.render_factual_review(result(), ACCOUNT, slot="draft"))
        self.assertEqual(checked.call_count, 1)

    def test_critical_factual_change_cannot_be_approved(self):
        r = result(text="I observed knee pain in 1995.")
        self.st.multiselect.side_effect = lambda label, options, **kw: [options[0]]
        self.st.checkbox.return_value = True
        self.assertFalse(self.view.render_factual_review(r, r.draft, slot="draft"))
        self.st.checkbox.assert_not_called()
        self.assertNotIn("factual_draft_receipt", self.session)

    def test_edit_invalidates_approval_including_edit_then_revert(self):
        r = result(); self.approve(r)
        r.draft = "I observed knee pain in 1995."
        self.assertFalse(self.view.render_factual_review(r, r.draft, slot="draft"))
        r.draft = ACCOUNT
        self.assertFalse(self.view.render_factual_review(r, r.draft, slot="draft"))
        self.assertNotIn("factual_draft_receipt", self.session)

    def test_source_context_change_invalidates_approval(self):
        r = result(); self.approve(r)
        r.evidence_source[0]["text"] += " A new record statement."
        self.assertFalse(self.view.render_factual_review(r, r.draft, slot="draft"))

    def test_source_selection_change_invalidates_approval(self):
        r = result(); self.approve(r)
        self.st.multiselect.side_effect = lambda *a, **kw: []
        self.assertFalse(self.view.render_factual_review(r, r.draft, slot="draft"))

    def test_owner_change_invalidates_approval(self):
        r = result()
        with patch.object(self.view.pilot, "enabled", return_value=True), patch.object(self.view.pilot, "current_owner", return_value="owner-a"):
            self.approve(r)
        with patch.object(self.view.pilot, "enabled", return_value=True), patch.object(self.view.pilot, "current_owner", return_value="owner-b"):
            self.assertFalse(self.view.render_factual_review(r, r.draft, slot="draft"))

    def test_legacy_or_foreign_inputs_cannot_use_old_receipt(self):
        r = result(); self.approve(r)
        r.factual_inputs = {}
        self.assertFalse(self.view.render_factual_review(r, r.draft, slot="draft"))
        self.assertNotIn("factual_draft_receipt", self.session)


class TestReviewInStreamlit(unittest.TestCase):
    def test_real_widgets_require_fresh_approval_after_edit_and_revert(self):
        from streamlit.testing.v1 import AppTest
        code = '''
import streamlit as st
from tests.test_factual_integrity import result, ACCOUNT
from app.views.factual_review import render_factual_review
r = result()
text = st.text_area("Candidate", value=ACCOUNT, key="candidate")
ready = render_factual_review(r, text, slot="draft")
st.write("READY" if ready else "BLOCKED")
'''
        at = AppTest.from_string(code, default_timeout=20).run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.checkbox), 1)
        at.checkbox[0].check().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.markdown[-1].value, "READY")
        at.text_area[0].set_value("I observed pain in 1995.").run()
        self.assertFalse(at.exception)
        self.assertEqual(at.markdown[-1].value, "BLOCKED")
        self.assertEqual(len(at.checkbox), 0)
        at.text_area[0].set_value(ACCOUNT).run()
        self.assertFalse(at.exception)
        self.assertFalse(at.checkbox[0].value)
        self.assertEqual(at.markdown[-1].value, "BLOCKED")

    def test_both_review_panels_can_render_without_colliding_widgets(self):
        from streamlit.testing.v1 import AppTest
        code = '''
from tests.test_factual_integrity import result, ACCOUNT
from app.views.factual_review import render_factual_review
render_factual_review(result(), ACCOUNT, slot="draft")
render_factual_review(result(), ACCOUNT, slot="eval")
'''
        at = AppTest.from_string(code, default_timeout=20).run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.checkbox), 2)
