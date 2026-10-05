"""Synthetic IA-06/07/08 regressions; no credentials, real records or paid calls."""
from __future__ import annotations

import copy
import json
import sys
import unittest
from unittest.mock import MagicMock, patch

from tests import hermetic  # noqa: F401
from app import pilot
from app.bounded_json import MAX_JSON_DEPTH, MAX_JSON_ITEMS, MAX_RESPONSE_BYTES, JSONContractError, decode_json
from app.documents import DocumentPage, ExtractedDocument, document_from_text
from app.draft import DraftResult, _pages_to_source, run_draft
from app.drafting_service import DraftingPayloadError
from app.evaluate import run_evaluation
from app.factual_integrity import compare, context_for_result, retained_inputs
from app.job_payload import draft_from_json, draft_to_json
from app.llm import LLMClient, LLMError, LLMParseError, LLMJSONContractError, _parse_json
from app.medical_review import MedicalDigest, MedicalFact, critical_fact_flags, review_medical_records, verify_citations
from app.provider_limits import validate_response_json
from app.request_validation import RequestValidationError, validate_records

QUOTE = "Pain was 2/10 at rest and 8/10 walking."
SWAP = "Pain was 8/10 at rest and 2/10 walking."


def fact(description=QUOTE):
    return MedicalFact("unknown", "symptom", description, "record.txt p.1", QUOTE, "record.txt", 1)


def client(*responses):
    llm = object.__new__(LLMClient)
    llm.chat = MagicMock(side_effect=responses)
    return llm


class StrictModelJSON(unittest.TestCase):
    def test_ambiguous_values_refused_in_inner_and_outer_documents(self):
        for text in ('{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '{"x":1e999}',
                     '{"x":-1e999}', '{"x":1,"x":2}', '{"x":1,"\\u0078":2}',
                     '{"nested":{"x":1,"x":2}}', '{"x":"\\ud800"}',
                     '{"\\udfff":1}', '{"x":"\ud800"}', '{"x":' + '9' * 5000 + '}'):
            with self.subTest(text=text[:60]):
                with self.assertRaises(LLMJSONContractError):
                    _parse_json(text)
                with self.assertRaises(pilot.PilotBlocked):
                    validate_response_json(text.encode("utf-8", errors="surrogatepass"))

    def test_valid_unicode_finite_values_arrays_and_escaped_punctuation(self):
        text = '{"quote":"Élodie β 😀 \\\"[{}]\\\" \\\\","dose":-1.25e2,"x":[true,null]}'
        self.assertEqual(_parse_json(text), json.loads(text))
        self.assertEqual(_parse_json('["\\ud83d\\ude00"]'), ["😀"])
        validate_response_json(text.encode())

    def test_limits_are_checked_before_json_graph_allocation(self):
        for raw in (b"[" * (MAX_JSON_DEPTH + 1) + b"0" + b"]" * (MAX_JSON_DEPTH + 1),
                    b"[" + b"0," * (MAX_JSON_ITEMS + 1) + b"0]",
                    b'{"x":"' + b"a" * MAX_RESPONSE_BYTES + b'"}'):
            with self.subTest(size=len(raw)), patch("app.bounded_json.json.loads") as loads:
                with self.assertRaises(JSONContractError):
                    decode_json(raw)
                loads.assert_not_called()
        nested = "[" * MAX_JSON_DEPTH + "0" + "]" * MAX_JSON_DEPTH
        self.assertIsInstance(_parse_json(nested), list)
        # 1 opening array + 19,999 commas = exactly 20,000 structural units.
        self.assertEqual(len(_parse_json("[" + "0," * (MAX_JSON_ITEMS - 1) + "0]")), MAX_JSON_ITEMS)

    def test_bytes_include_multibyte_text_and_fence_or_prose_wrappers(self):
        for text in ('{"x":"' + "界" * (MAX_RESPONSE_BYTES // 3) + '"}',
                     "```json\n{}\n```" + " " * MAX_RESPONSE_BYTES,
                     "p" * MAX_RESPONSE_BYTES + " {}"):
            with self.subTest(size=len(text)), self.assertRaises(LLMJSONContractError):
                _parse_json(text)
        body = '{"x":"' + "x" * (MAX_RESPONSE_BYTES - len('{"x":""}')) + '"}'
        self.assertEqual(len(body.encode()), MAX_RESPONSE_BYTES)
        self.assertEqual(len(_parse_json(body)["x"]), MAX_RESPONSE_BYTES - 8)

    def test_scalar_roots_refused_and_pilot_prose_cannot_be_salvaged(self):
        for text in ("null", "1", '"value"', "true"):
            with self.subTest(text=text), self.assertRaises(LLMJSONContractError):
                _parse_json(text)
        with patch.object(pilot, "enabled", return_value=True):
            with self.assertRaises(LLMParseError):
                _parse_json('Here is {"facts":[]}')
            self.assertEqual(_parse_json('```json\n{"facts":[]}\n```'), {"facts": []})
        with patch.object(pilot, "enabled", return_value=False):
            self.assertEqual(_parse_json('Here is {"facts":[]}'), {"facts": []})

    def test_invalid_object_cannot_be_replaced_by_its_valid_nested_array(self):
        for text in ('{"facts":[],"x":NaN}', '{"facts":[],bad}',
                     'prose {"facts":[],"x":1,"x":2}'):
            with self.subTest(text=text), self.assertRaises(LLMParseError):
                _parse_json(text)

    def test_hard_contract_failure_never_reasks_with_larger_budget(self):
        for text in ('{"x":NaN}', '{"x":1,"x":2}', "[" * 2000 + "0" + "]" * 2000,
                     '{"x":"' + "x" * MAX_RESPONSE_BYTES):
            llm = client(text, '{"ok":true}')
            with self.subTest(size=len(text)), self.assertRaises(LLMJSONContractError):
                llm.chat_json("sys", "user")
            self.assertEqual(llm.chat.call_count, 1)

    def test_repair_candidate_has_the_identical_strict_contract(self):
        llm = client('{"facts":', '{"x":1e999}')
        with self.assertRaises(LLMParseError):
            llm.chat_json("sys", "user")
        self.assertEqual(llm.chat.call_count, 2)
        with patch.object(pilot, "enabled", return_value=True):
            llm = client('{"facts":', '{"facts":[]}')
            self.assertEqual(llm.chat_json("sys", "user"), {"facts": []})
        self.assertEqual(llm.chat.call_count, 2)

    def test_integer_bound_does_not_depend_on_interpreter_configuration(self):
        previous = sys.get_int_max_str_digits()
        try:
            sys.set_int_max_str_digits(0)
            with self.assertRaises(LLMJSONContractError):
                _parse_json('{"x":' + '9' * 5000 + '}')
        finally:
            sys.set_int_max_str_digits(previous)

    def test_record_retry_round_does_not_retry_hard_json_contract_failure(self):
        llm = MagicMock()
        llm.chat_json.side_effect = LLMJSONContractError("Synthetic JSON limit refusal")
        with patch.object(pilot, "enabled", return_value=True), self.assertRaises(LLMError):
            review_medical_records(llm, [document_from_text("record.txt", QUOTE)])
        self.assertEqual(llm.chat_json.call_count, 1)
        llm.chat.assert_not_called()


class CriticalMeaningScreen(unittest.TestCase):
    def test_critical_feature_changes_are_flagged_despite_same_number_sets(self):
        cases = (
            (QUOTE, SWAP, "number-to-claim associations"),
            ("Dose A was 10 mg and dose B was 20 mg.", "Dose A was 20 mg and dose B was 10 mg.", "number-to-claim associations"),
            ("Pain preceded treatment in January.", "Pain preceded treatment in February.", "calendar dates"),
            ("Pain began before treatment.", "Pain began after treatment.", "chronology"),
            ("I reported pain.", "He reported pain.", "speaker"),
            ("Pain began around 2020.", "Pain began in 2020.", "date precision"),
            ("Pain occurred daily.", "Pain occurred weekly.", "frequency"),
            ("Patient denies pain.", "Patient reports pain.", "negation"),
            ("Pain was reported.", "Pain was observed.", "attribution"),
            ("Possible left knee pain.", "Left knee pain.", "uncertainty"),
            ("Left knee pain.", "Right knee pain.", "laterality"),
            ("Patient reported pain.", "Patient diagnosed with PTSD.", "diagnosis/nexus"),
        )
        for quote, description, field in cases:
            quote += " This is the synthetic clinical assessment."
            description += " This is the synthetic clinical assessment."
            f = MedicalFact("unknown", "symptom", description, "record.txt p.1", quote, "record.txt", 1)
            with self.subTest(field=field):
                checked = verify_citations([f], [document_from_text("record.txt", quote)])
                self.assertEqual(checked["quote_verified"], 1)
                self.assertEqual(checked["semantic_verified"], 0)
                self.assertIn(field, checked["critical_review_flags"][0]["fields"])
                self.assertEqual(f.meaning_status, "unreviewed")

    def test_same_text_has_no_flag_and_never_claims_semantic_approval(self):
        checked = verify_citations([fact()], [document_from_text("record.txt", QUOTE)])
        self.assertEqual(checked["critical_review_flags"], [])
        self.assertEqual(checked["semantic_review_required"], 1)

    def test_pilot_number_swap_stops_before_summary(self):
        llm = MagicMock()
        llm.chat_json.return_value = {"facts": [vars(fact(SWAP))]}
        with patch.object(pilot, "enabled", return_value=True), self.assertRaises(pilot.PilotBlocked):
            review_medical_records(llm, [document_from_text("record.txt", QUOTE)])
        llm.chat.assert_not_called()


class CompletePilotRecords(unittest.TestCase):
    def variants(self):
        doc = document_from_text("record.txt", QUOTE)
        for fields in ({"total_pages": 2, "unreadable_pages": [2]}, {"coverage_known": False},
                       {"coverage_known": 1}, {"total_pages": 0}, {"total_pages": True},
                       {"total_pages": 2}, {"pages": [DocumentPage("record.txt", 2, QUOTE)]},
                       {"pages": [DocumentPage("record.txt", 1, QUOTE, kind="block")]},
                       {"total_pages": 2, "pages": [DocumentPage("record.txt", 1, QUOTE)] * 2},
                       {"total_pages": 2, "pages": [DocumentPage("record.txt", 2, QUOTE), DocumentPage("record.txt", 1, QUOTE)]}):
            changed = copy.deepcopy(doc)
            for name, value in fields.items():
                setattr(changed, name, value)
            yield changed

    def test_incomplete_records_stop_evaluation_draft_and_direct_review(self):
        for doc in self.variants():
            llm = MagicMock()
            with self.subTest(doc=doc), patch.object(pilot, "enabled", return_value=True), \
                    patch.object(pilot, "action_budget") as budget:
                with self.assertRaises(RequestValidationError):
                    run_evaluation(llm, "I observed pain.", [doc])
                with self.assertRaises(DraftingPayloadError):
                    run_draft(llm, [doc], {}, "I observed pain.", "Knee", "New claim")
                with self.assertRaises(RequestValidationError):
                    review_medical_records(llm, [doc])
                budget.assert_not_called()
                llm.chat_json.assert_not_called()
                llm.chat.assert_not_called()

    def test_complete_page_and_block_records_remain_eligible(self):
        for text in (QUOTE, "Long clinical source. " * 10000):
            doc = document_from_text("record.txt", text)
            with self.subTest(kind=doc.pagination), patch.object(pilot, "enabled", return_value=True):
                self.assertEqual(validate_records([doc])[0], len(doc.pages))
        doc = document_from_text("record.txt", QUOTE)
        with patch.object(pilot, "enabled", return_value=True), self.assertRaises(RequestValidationError):
            validate_records([doc, doc])

    def test_nonpilot_partial_coverage_keeps_existing_analysis_policy(self):
        doc = document_from_text("record.txt", QUOTE)
        doc.total_pages = 2; doc.unreadable_pages = [2]
        with patch.object(pilot, "enabled", return_value=False):
            self.assertEqual(validate_records([doc])[0], 2)

    def result(self):
        doc = document_from_text("record.txt", QUOTE)
        return DraftResult(draft="I observed pain.", digest=MedicalDigest(facts=[fact()]),
                           factual_inputs=retained_inputs("I observed pain.", {}, [doc]),
                           evidence_source=_pages_to_source([doc]))

    def test_saved_and_delayed_results_require_current_complete_source_manifest(self):
        original = self.result()
        changes = (
            lambda r: r.factual_inputs.pop("source_metadata"),
            lambda r: r.factual_inputs["source_metadata"][0].pop("coverage_policy"),
            lambda r: r.factual_inputs["source_metadata"][0].update(coverage_known=False),
            lambda r: r.factual_inputs["source_metadata"][0].update(total_pages=2),
            lambda r: r.factual_inputs["source_metadata"][0].update(total_pages=True),
            lambda r: r.factual_inputs["source_metadata"][0].update(unreadable_pages=[2]),
            lambda r: r.evidence_source.append(copy.deepcopy(r.evidence_source[0])),
            lambda r: r.digest.facts[0].__setattr__("description", SWAP),
            lambda r: r.digest.__setattr__("facts_dropped_by_cap", 1),
        )
        with patch.object(pilot, "enabled", return_value=True):
            self.assertIsNotNone(context_for_result(original))
            self.assertIsNotNone(context_for_result(draft_from_json(draft_to_json(original))))
            for change in changes:
                r = copy.deepcopy(original); change(r)
                # A saved all-clear does not override a fresh source check.
                r.digest.citation_check = {"critical_review_flags": [], "missing": 0, "skipped": 0}
                with self.subTest(change=change):
                    self.assertIsNone(context_for_result(r))
                    restored = draft_from_json(draft_to_json(r))
                    self.assertEqual(restored.factual_review["status"], "blocked")
                    self.assertEqual(compare(restored.output_statement, context_for_result(restored))["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
