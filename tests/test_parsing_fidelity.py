"""Synthetic end-to-end fidelity and provider-boundary regressions; no paid calls."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tests import hermetic  # noqa: F401
from tests.ingestion_fixtures import docx_parts, package
from tests.test_controlled_pilot import approval
from tests.test_extractors import _pdf_bytes
from app import pilot
from app.documents import ChunkPlan, DocumentPage, ExtractedDocument, ExtractionError, extract_document, paragraph_index, search_records
from app.job_payload import document_to_json, document_from_json, digest_to_json, digest_from_json
from app.medical_review import MedicalDigest, MedicalFact, _dates_in_text, _regex_extract_date, _summarize, build_timeline_data, retrieve_evidence, summary_sample, verify_citations
from app.provider_limits import MAX_RESPONSE_BYTES, bounded_transport, check_request, check_response_model, validate_response_json
from app.request_validation import RequestValidationError, validate_evaluation_request
from app.text_fidelity import evidence_excerpt
from scripts import ocr_records, ocr_and_extract

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


class SourceCoverage(unittest.TestCase):
    def test_uploaded_statement_refused_before_provider_for_each_incomplete_state(self):
        from app.evaluate import run_evaluation
        records = [extract_document("record.txt", b"Synthetic medical record.")]
        good = extract_document("statement.txt", b"Synthetic witness account.")
        for change in ({"total_pages": 2, "unreadable_pages": [2]}, {"coverage_known": False}, {"pages": []}):
            doc = copy.deepcopy(good)
            for key, value in change.items():
                setattr(doc, key, value)
            llm = MagicMock()
            with self.subTest(change=change), self.assertRaises(RequestValidationError):
                run_evaluation(llm, good.full_text, records, statement_source=doc)
            llm.chat.assert_not_called()
            llm.chat_json.assert_not_called()

    def test_statement_text_and_provenance_fingerprints_change_together(self):
        from app.views.follow_up import evaluation_input_key
        doc = extract_document("statement.txt", b"Synthetic testimony.")
        record = extract_document("records.txt", b"Synthetic assessment.")
        validate_evaluation_request(statement_text=doc.full_text, records=[record], statement_source=doc)
        with self.assertRaises(RequestValidationError):
            validate_evaluation_request(statement_text="Different testimony.", records=[record], statement_source=doc)
        initial = evaluation_input_key(doc.full_text, [record], {}, doc)
        doc.source_sha256 = "f" * 64
        self.assertNotEqual(initial, evaluation_input_key(doc.full_text, [record], {}, doc))

    def test_unicode_decoding_identity_survives_serialization(self):
        raw = "Élodie denies β pain; 10 mg; −1.".encode("utf-16")
        doc = extract_document("record.txt", raw)
        saved = document_from_json(document_to_json(doc))
        self.assertEqual(saved.source_sha256, hashlib.sha256(raw).hexdigest())
        self.assertEqual(saved.text_encoding, "utf-16-bom")
        self.assertEqual(saved.full_text, doc.full_text)

    def test_hard_limit_rejects_before_models_and_preserves_tail_at_boundary(self):
        from app.evaluate import run_evaluation
        from app.prompt_sanitize import sanitize_for_prompt
        llm = MagicMock()
        record = extract_document("record.txt", b"Synthetic source.")
        with self.assertRaises(RequestValidationError):
            run_evaluation(llm, "x" * 80001, [record])
        llm.chat_json.assert_not_called()
        source = "```" * 26000 + " FINAL_DENIAL_CANARY"
        self.assertIn("FINAL_DENIAL_CANARY", sanitize_for_prompt(source, max_chars=None))
        self.assertNotIn("by prompt sanitizer", sanitize_for_prompt(source, max_chars=None))


class WordStories(unittest.TestCase):
    def test_tabs_breaks_tables_and_all_supported_stories_keep_source_parts(self):
        parts = docx_parts()
        parts["word/document.xml"] = (f'<w:document xmlns:w="{W}"><w:body><w:p><w:r><w:t>Patient denies</w:t><w:tab/><w:t>pain</w:t><w:br/><w:t>since 2019.</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>No PTSD.</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>').encode()
        for stem, root in (("header1", "hdr"), ("footer1", "ftr"), ("footnotes", "footnotes"), ("endnotes", "endnotes"), ("comments", "comments")):
            entry = f'<w:p><w:r><w:t>{stem} FINAL_STORY_CANARY.</w:t></w:r></w:p>'
            if root in ("footnotes", "endnotes", "comments"):
                child = root[:-1]
                entry = f'<w:{child} w:id="7" w:author="Synthetic reviewer" w:date="2024-03-05">{entry}</w:{child}>'
            parts[f"word/{stem}.xml"] = f'<w:{root} xmlns:w="{W}">{entry}</w:{root}>'.encode()
        doc = extract_document("statement.docx", package(parts))
        self.assertIn("denies\tpain\nsince 2019", doc.full_text)
        self.assertTrue(any("denies pain\nsince 2019" in chunk.text for chunk in ChunkPlan(doc.pages)))
        self.assertIn("No PTSD.", doc.full_text)
        self.assertEqual(doc.pagination, "block")
        self.assertEqual({p.source_part for p in doc.pages}, {n for n in parts if n.startswith("word/")})
        self.assertIn("author=Synthetic reviewer", doc.full_text)
        saved = document_from_json(document_to_json(doc))
        self.assertEqual([p.source_part for p in saved.pages], [p.source_part for p in doc.pages])

    def test_visual_revisions_fields_and_equations_cannot_claim_complete_coverage(self):
        for element in ('<w:drawing/>', '<w:ins/>', '<w:del/>', '<w:instrText>DATE</w:instrText>', '<w:sym w:char="F0B1"/>', '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"/>'):
            parts = docx_parts()
            parts["word/document.xml"] = parts["word/document.xml"].replace(b"</w:r>", element.encode() + b"</w:r>")
            with self.subTest(element=element), self.assertRaises(ExtractionError):
                extract_document("statement.docx", package(parts))


class RetrievalFidelity(unittest.TestCase):
    def test_short_negations_and_unicode_remain_indexed_and_found(self):
        for text, query in (("No PTSD.", "PTSD"), ("No SI.", "SI"), ("Élodie nie la douleur.", "Élodie"), ("没有疼痛。", "没有疼痛"), ("β = −1.", "β")):
            doc = extract_document("record.txt", text.encode())
            with self.subTest(text=text):
                self.assertEqual(paragraph_index(doc)[0].text, text)
                found = retrieve_evidence([doc], query)
                self.assertEqual(found.excerpts, 1)
                self.assertIn(text, found.text)
                self.assertTrue(search_records([doc], query))

    def test_query_at_end_is_returned_with_negation_and_attribution(self):
        source = "Routine earlier observation. " * 70 + "On 2024-03-05 the veteran denies PTSD, according to the clinician."
        doc = extract_document("record.txt", source.encode())
        found = retrieve_evidence([doc], "PTSD")
        self.assertIn("veteran denies PTSD, according to the clinician.", found.text)
        self.assertIn("2024-03-05", found.text)
        self.assertIn("Earlier text omitted", found.text)

    def test_unfit_sentence_is_disclosed_and_never_treated_as_strong_evidence(self):
        doc = extract_document("record.txt", ("unbroken " * 400 + "denies PTSD").encode())
        found = retrieve_evidence([doc], "PTSD", excerpt_chars=100)
        self.assertEqual(found.excerpts, 0)
        self.assertTrue(found.weak)
        self.assertIn("oversized sentences omitted", found.text)
        self.assertEqual(evidence_excerpt(doc.full_text, "PTSD", 100), "")


class DateFidelity(unittest.TestCase):
    def test_approximate_full_dates_keep_qualifier_and_never_create_exact_gaps(self):
        self.assertEqual(_regex_extract_date("approximately 2020-03-04"), ("2020-03-04", "approximate-day"))
        self.assertEqual(_regex_extract_date("around June 2021"), ("2021-06-01", "approximate-month"))
        anchors = [json.loads(x) for x in _dates_in_text("approximately 2020-03-04; around June 2021")]
        self.assertTrue(all(item["approximate"] for item in anchors))
        self.assertEqual(anchors[1]["normalized"], "2021-06")
        facts = [MedicalFact("approximately 2020-03-04", "symptom", "Synthetic", "a p.1"),
                 MedicalFact("2024-03-04", "symptom", "Synthetic", "a p.2")]
        self.assertEqual(build_timeline_data(MedicalDigest(facts=facts))["gaps"], [])

    def test_slash_dates_resolve_only_when_order_is_unambiguous(self):
        for raw, expected in (("04/17/2019", ("2019-04-17", "day")), ("17/04/2019", ("2019-04-17", "day")), ("04/04/2019", ("2019-04-04", "day")), ("01/02/2019", None), ("04/17/19", None), ("04/31/2019", None), ("2019-13", None), ("February 31, 2019", None), ("April 17, 2019", ("2019-04-17", "day"))):
            with self.subTest(raw=raw):
                self.assertEqual(_regex_extract_date(raw), expected)

    def test_anchor_retains_raw_precision_and_ambiguity_without_invented_day(self):
        anchors = [json.loads(x) for x in _dates_in_text("circa 2019; June 2021; 01/02/2024; 04/17/2024")]
        self.assertEqual(anchors[0], {"raw": "circa 2019", "normalized": "2019", "precision": "year", "approximate": True})
        self.assertEqual(anchors[1]["normalized"], "2021-06")
        self.assertIsNone(anchors[2]["normalized"])
        self.assertEqual(anchors[3]["normalized"], "2024-04-17")

    def test_ambiguous_printed_date_cannot_be_replaced_by_quote_or_optional_model(self):
        fact = MedicalFact("01/02/2019", "symptom", "Synthetic observation", "a p.1", "Earlier note 2020-03-04")
        llm = MagicMock()
        result = build_timeline_data(MedicalDigest(facts=[fact]), llm)
        self.assertEqual(result["dated_count"], 0)
        llm.chat_json.assert_not_called()


class SummaryAndMeaning(unittest.TestCase):
    def test_late_fact_survives_whole_fact_budget_and_manifest_round_trip(self):
        facts = [MedicalFact(str(1800 + i), "symptom", "Earlier observation " * 12, "record.txt p.1") for i in range(200)]
        facts.append(MedicalFact("2025-03-04", "symptom", "LATEST_DENIAL_CANARY: veteran denies symptoms.", "record.txt p.1"))
        digest = MedicalDigest(facts=facts)
        llm = MagicMock()
        llm.chat.return_value = "Synthetic summary"
        _summarize(llm, digest)
        prompt = llm.chat.call_args.args[1]
        self.assertIn("LATEST_DENIAL_CANARY", prompt)
        self.assertTrue(digest.summary_selection["latest_dated_included"])
        self.assertGreater(digest.summary_selection["omitted"], 0)
        self.assertEqual(digest_from_json(digest_to_json(digest)).summary_selection, digest.summary_selection)
        text, manifest = summary_sample(facts, 100)
        self.assertLessEqual(len(text), 100)
        self.assertEqual(manifest["total"], len(facts))
        self.assertIn("LATEST_DENIAL_CANARY", text)

    def test_real_quote_cannot_certify_changed_diagnosis_negation_or_date(self):
        text = "In 2021 the veteran denies PTSD. The clinician reported 10 mg."
        doc = extract_document("records.txt", text.encode())
        fact = MedicalFact("2022-01-01", "diagnosis", "Confirmed PTSD in 2022; veteran observed 20 mg.", "records.txt p.1", text, "records.txt", 1)
        checked = verify_citations([fact], [doc])
        self.assertEqual(checked["quote_verified"], 1)
        self.assertEqual(checked["semantic_verified"], 0)
        self.assertEqual(checked["semantic_review_required"], 1)
        fields = checked["critical_review_flags"][0]["fields"]
        self.assertIn("negation", fields)
        self.assertIn("attribution", fields)
        self.assertIn("dates/numbers", fields)
        self.assertEqual(fact.meaning_status, "unreviewed")


class ProviderLimits(unittest.TestCase):
    def test_returned_model_must_match_the_exact_approved_version(self):
        from types import SimpleNamespace
        data = approval()
        data["model_profiles"]["test-model"]["model_version"] = "immutable-test-version-1"
        with patch.dict(os.environ, {"VA_LSE_MODE": "controlled-pilot"}), patch.object(pilot, "load_approval", return_value=data):
            check_response_model("test-model", SimpleNamespace(model="immutable-test-version-1"))
            for model in (None, "immutable-test-version-2"):
                with self.subTest(model=model), self.assertRaises(pilot.PilotBlocked):
                    check_response_model("test-model", SimpleNamespace(model=model))

    def test_installed_sdk_uses_guarded_transport_for_both_endpoint_schemas(self):
        import httpx2
        import openai
        seen = []
        def respond(request):
            seen.append((str(request.url), request.headers["Accept-Encoding"], json.loads(request.content)))
            if request.url.path.endswith("/responses"):
                body = {"id": "synthetic", "object": "response", "created_at": 0, "model": "test-version", "output": [], "status": "completed"}
            else:
                body = {"id": "synthetic", "object": "chat.completion", "created": 0, "model": "test-version", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Synthetic denial."}, "finish_reason": "stop"}]}
            return httpx2.Response(200, stream=httpx2.ByteStream(json.dumps(body).encode()))
        transport = bounded_transport(10, httpx2.MockTransport(respond))
        with openai.OpenAI(api_key="SYNTHETIC_NOT_A_CREDENTIAL", base_url="https://provider.example.test/v1", max_retries=0,
                           http_client=openai.DefaultHttpxClient(transport=transport, trust_env=False)) as client:
            self.assertEqual(client.chat.completions.create(model="route", messages=[{"role": "user", "content": "Synthetic"}]).model, "test-version")
            self.assertEqual(client.responses.create(model="route", input="Synthetic").model, "test-version")
        self.assertEqual([item[1] for item in seen], ["identity", "identity"])

    def test_installed_sdk_preserves_refusal_cause_without_accepting_compressed_body(self):
        import httpx2
        import openai
        from app.llm import _normalize_provider_error
        transport = bounded_transport(10, httpx2.MockTransport(lambda req: httpx2.Response(200, headers={"Content-Encoding": "gzip"}, stream=httpx2.ByteStream(b"tiny"))))
        with openai.OpenAI(api_key="SYNTHETIC_NOT_A_CREDENTIAL", base_url="https://provider.example.test/v1", max_retries=0,
                           http_client=openai.DefaultHttpxClient(transport=transport, trust_env=False)) as client:
            try:
                client.chat.completions.create(model="route", messages=[{"role": "user", "content": "Synthetic"}])
                self.fail("SDK accepted compressed body")
            except openai.APIConnectionError as exc:
                with self.assertRaises(pilot.PilotBlocked):
                    _normalize_provider_error(exc)

    def test_complete_multibyte_request_output_and_profiles_are_enforced(self):
        data = approval()
        data["model_profiles"]["test-model"].update(context_window_tokens=4096, framing_token_reserve=128, max_output_tokens=1024)
        with patch.dict(os.environ, {"VA_LSE_MODE": "controlled-pilot"}), patch.object(pilot, "load_approval", return_value=data):
            check_request("test-model", {"messages": [{"role": "system", "content": "short"}]}, 512)
            for model, body, output in (("unknown-model", {}, 1), ("test-model", {"messages": [{"role": "system", "content": "界" * 1500}]}, 512), ("test-model", {}, 1025)):
                with self.subTest(model=model, output=output), self.assertRaises(pilot.PilotBlocked):
                    check_request(model, body, output)
            data.pop("model_profiles")
            with self.assertRaises(pilot.PilotBlocked):
                check_request("test-model", {}, 1)

    def test_stream_caps_count_actual_bytes_and_close_before_sdk_parses(self):
        import httpx2
        class Stream(httpx2.SyncByteStream):
            def __init__(self):
                self.reads = 0
                self.closed = False
            def __iter__(self):
                for _ in range(100):
                    self.reads += 1
                    yield b"x" * 65536
            def close(self):
                self.closed = True
        stream = Stream()
        with httpx2.Client(transport=bounded_transport(10, httpx2.MockTransport(lambda req: httpx2.Response(200, stream=stream)))) as client:
            with self.assertRaises(pilot.PilotBlocked):
                client.post("https://provider.example.test/v1/chat/completions")
        self.assertLessEqual(stream.reads, MAX_RESPONSE_BYTES // 65536 + 1)
        self.assertTrue(stream.closed)

    def test_compression_depth_duplicate_keys_invalid_unicode_and_item_bombs_refused(self):
        import httpx2
        with httpx2.Client(transport=bounded_transport(10, httpx2.MockTransport(lambda req: httpx2.Response(200, headers={"Content-Encoding": "gzip"}, stream=httpx2.ByteStream(b"tiny"))))) as client:
            with self.assertRaises(pilot.PilotBlocked):
                client.post("https://provider.example.test/v1/responses")
        for raw in (b'{"x":' + b"[" * 65 + b"0" + b"]" * 65 + b"}", b'{"x":1,"x":2}', b'{"x":"\xff"}', b'{"x":[' + b"0," * 20001 + b"0]}"):
            with self.subTest(length=len(raw)), self.assertRaises(pilot.PilotBlocked):
                validate_response_json(raw)
        validate_response_json('{"text":"Élodie denies pain."}'.encode())

    def test_sdk_wrapped_resource_refusal_never_becomes_retriable(self):
        from app.llm import _normalize_provider_error
        try:
            try:
                raise pilot.PilotBlocked("Synthetic bounded transport refusal")
            except pilot.PilotBlocked as inner:
                raise RuntimeError("SDK connection error") from inner
        except RuntimeError as wrapped:
            with self.assertRaises(pilot.PilotBlocked):
                _normalize_provider_error(wrapped)


class OCRFidelity(unittest.TestCase):
    def test_writer_refuses_121_lines_without_publishing_a_partial_transcript(self):
        pdf = MagicMock()
        with self.assertRaisesRegex(ValueError, "no text was dropped"):
            ocr_records._write_text_page(pdf, "\n".join(f"line {i}" for i in range(120)) + "\nFINAL_CANARY", 792, 612)
        pdf.drawString.assert_not_called()
        pdf.showPage.assert_not_called()

    def test_unsupported_glyphs_and_overwide_words_refused_before_drawing(self):
        for text in ("β = −1", "word" * 300):
            pdf = MagicMock()
            with self.subTest(text=text[:10]), self.assertRaises(ValueError):
                ocr_records._write_text_page(pdf, text, 792, 612)
            pdf.drawString.assert_not_called()

    def test_atomic_ocr_failure_keeps_original_and_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "source.pdf", Path(directory) / "output.pdf"
            source.write_bytes(_pdf_bytes([""]))
            original = source.read_bytes()
            destination.write_bytes(b"PREVIOUS_COMPLETE_OUTPUT")
            with patch.object(ocr_records, "_ocr_pages_parallel", return_value={1: "line\n" * 121}):
                with self.assertRaises(ValueError):
                    ocr_records.ocr_with_tesseract(source, destination, jobs=1)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(destination.read_bytes(), b"PREVIOUS_COMPLETE_OUTPUT")
            self.assertFalse(list(Path(directory).glob(".ocr-*")))


if __name__ == "__main__":
    unittest.main()
