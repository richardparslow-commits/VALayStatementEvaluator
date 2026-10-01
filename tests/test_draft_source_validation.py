"""Synthetic provenance checks; no provider calls or veteran information."""
from __future__ import annotations

import copy
import json
import unittest
from contextlib import nullcontext
from unittest.mock import patch

from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app.documents import BLOCK, PAGE, DocumentPage, ExtractedDocument
from app.draft import DraftResult, _citation_display, _normalize_grounding, grounding_markdown, run_draft
from app.drafting_service import DraftingError
from app.grounding_sources import (
    GROUNDING_SOURCE_POLICY, LEGACY_GROUNDING_SOURCE_NOTICE,
    grounding_catalog, validate_grounding_sources,
)
from app.job_payload import draft_from_json, draft_to_json
from app.llm import LLMParseError
from app.medical_review import MedicalDigest, MedicalFact
from app.source_validation import build_source_index
from tests.grounding_fixtures import complete_grounding
from tests.test_draft import WITNESS, _FakeLLM

QUOTE = "Patient reports knee pain every morning."


def evidence(kind=PAGE, filename="clinic.txt"):
    doc = ExtractedDocument(filename, [DocumentPage(filename, 1, QUOTE, kind)], pagination=kind)
    source = doc.pages[0].label
    digest = MedicalDigest(facts=[MedicalFact("2024-01", "symptom", QUOTE, source, QUOTE,
                                             document=filename, page=1)])
    return [doc], digest


def record_grounding(entry, section="supported_observations"):
    data = complete_grounding()
    field = {"supported_observations": "record_support", "conflicts": "record_fact",
             "suggested_inclusions": "fact"}[section]
    row = {key: copy.deepcopy(entry[key]) for key in ("fact_id", "source", "source_unit", "quote")}
    row[field] = entry["description"]
    if section != "suggested_inclusions":
        row["observation"] = "I think knee pain began around 2020; I am unsure."
    if section == "conflicts":
        row["resolution_note"] = "Ask the witness to clarify without replacing their account."
    data[section] = [row]
    return data


class TestGroundingSources(unittest.TestCase):
    def setUp(self):
        self.docs, self.digest = evidence()
        self.catalog, self.prompt = grounding_catalog(self.digest, self.docs, "knee pain")
        self.entry = next(iter(self.catalog.values()))

    def validate(self, data, docs=None, catalog=None):
        return validate_grounding_sources(_normalize_grounding(data),
            self.catalog if catalog is None else catalog,
            build_source_index(self.docs if docs is None else docs))

    def test_all_record_sections_bind_to_fact_and_typed_source(self):
        for section in ("supported_observations", "conflicts", "suggested_inclusions"):
            with self.subTest(section=section):
                data = record_grounding(self.entry, section)
                self.assertEqual(self.validate(data), data)

    def test_exact_alias_is_canonicalized_without_rewriting_witness(self):
        data = record_grounding(self.entry)
        row = data["supported_observations"][0]
        row["source"] = "[CLINIC.TXT — page 1]"
        row["quote"] = QUOTE.upper().replace(" ", "\n")
        result = self.validate(data)["supported_observations"][0]
        self.assertEqual(result["source"], "clinic.txt p.1")
        self.assertEqual(result["quote"], QUOTE)
        self.assertEqual(result["observation"], "I think knee pain began around 2020; I am unsure.")

    def test_invalid_file_page_kind_range_and_extra_prose_are_rejected(self):
        for source in ("absent.pdf p.999", "clinic.txt p.999", "clinic.txt b.1",
                       "clinic.txt p.1-p.2", "clinic.txt p.1 confirms the claim"):
            with self.subTest(source=source):
                data = record_grounding(self.entry)
                data["supported_observations"][0]["source"] = source
                with self.assertRaises(LLMParseError):
                    self.validate(data)

    def test_missing_or_fabricated_metadata_rejects_each_record_section(self):
        for section in ("supported_observations", "conflicts", "suggested_inclusions"):
            for field in ("fact_id", "source", "source_unit", "quote"):
                with self.subTest(section=section, field=field):
                    data = record_grounding(self.entry, section)
                    del data[section][0][field]
                    with self.assertRaises(LLMParseError):
                        self.validate(data)
        data = record_grounding(self.entry)
        data["supported_observations"][0]["fact_id"] = "fact-invented"
        with self.assertRaises(LLMParseError):
            self.validate(data)

    def test_typed_address_requires_exact_kind_filename_and_integer(self):
        for address in ({"filename": "clinic.txt", "kind": BLOCK, "number": 1},
                        {"filename": "other.txt", "kind": PAGE, "number": 1},
                        {"filename": "clinic.txt", "kind": PAGE, "number": True},
                        {"filename": "clinic.txt", "kind": PAGE, "number": "1"}):
            with self.subTest(address=address):
                data = record_grounding(self.entry)
                data["supported_observations"][0]["source_unit"] = address
                with self.assertRaises(LLMParseError):
                    self.validate(data)

    def test_altered_quote_and_invented_description_are_rejected(self):
        for field, value in (("quote", QUOTE.replace("morning", "night")),
                             ("quote", "Patient reports knee pain"),
                             ("record_support", "Pain is severe and constant.")):
            data = record_grounding(self.entry)
            data["supported_observations"][0][field] = value
            with self.subTest(field=field), self.assertRaises(LLMParseError):
                self.validate(data)

    def test_valid_fact_id_cannot_be_used_with_another_valid_source(self):
        docs = self.docs + [ExtractedDocument("other.txt", [DocumentPage("other.txt", 1, QUOTE)])]
        data = record_grounding(self.entry)
        data["supported_observations"][0]["source"] = "other.txt p.1"
        with self.assertRaises(LLMParseError):
            self.validate(data, docs)

    def test_duplicate_and_unreadable_sources_cannot_supply_catalog_facts(self):
        unreadable = ExtractedDocument("clinic.txt", [], total_pages=1, unreadable_pages=[1])
        for docs in (self.docs * 2, self.docs + [unreadable]):
            with self.subTest(docs=len(docs)):
                catalog, text = grounding_catalog(self.digest, docs, "knee")
                self.assertEqual(catalog, {})
                self.assertEqual(json.loads(text), [])
                with self.assertRaises(LLMParseError):
                    self.validate(record_grounding(self.entry), docs)

    def test_empty_and_mislabelled_units_are_rejected(self):
        for text, filename in (("", "clinic.txt"), (QUOTE, "wrong.txt")):
            docs = [ExtractedDocument("clinic.txt", [DocumentPage(filename, 1, text)])]
            with self.subTest(filename=filename):
                self.assertEqual(grounding_catalog(self.digest, docs, "")[0], {})

    def test_conflicting_fact_resolution_metadata_is_excluded(self):
        for field, value in (("document", "other.txt"), ("page", 999), ("source", "clinic.txt p.2-p.3")):
            digest = copy.deepcopy(self.digest)
            setattr(digest.facts[0], field, value)
            with self.subTest(field=field):
                self.assertEqual(grounding_catalog(digest, self.docs, "")[0], {})

    def test_text_blocks_are_supported_without_page_alias(self):
        docs, digest = evidence(BLOCK)
        catalog, _ = grounding_catalog(digest, docs, "")
        entry = next(iter(catalog.values()))
        self.assertEqual(entry["source_unit"]["kind"], BLOCK)
        data = record_grounding(entry)
        self.validate(data, docs, catalog)
        data["supported_observations"][0]["source"] = "clinic.txt p.1"
        with self.assertRaises(LLMParseError):
            self.validate(data, docs, catalog)

    def test_quote_tail_short_quotes_and_numeric_fragments_are_excluded(self):
        for quote, text in ((QUOTE + " This did not happen.", QUOTE),
                            ("knee pain", QUOTE),
                            ("5 mg orally every morning", "1.5 mg orally every morning"),
                            ("oxygen saturation measured at 95", "oxygen saturation measured at 95%")):
            docs, digest = evidence()
            docs[0].pages[0].text = text
            digest.facts[0].quote = quote
            with self.subTest(quote=quote):
                self.assertEqual(grounding_catalog(digest, docs, "")[0], {})

    def test_fact_ids_are_stable_and_change_with_snapshot(self):
        self.assertEqual(self.catalog, grounding_catalog(copy.deepcopy(self.digest), self.docs, "")[0])
        for field, value in (("date", "2025"), ("description", "Different extracted description")):
            digest = copy.deepcopy(self.digest)
            setattr(digest.facts[0], field, value)
            self.assertNotEqual(set(self.catalog), set(grounding_catalog(digest, self.docs, "")[0]))

    def test_catalog_budgets_drop_whole_entries_and_keep_digest(self):
        for limit in (0, 10, len(self.prompt) - 1):
            with self.subTest(limit=limit):
                catalog, text = grounding_catalog(self.digest, self.docs, "", budget_chars=limit)
                self.assertEqual(catalog, {})
                self.assertEqual(json.loads(text), [])
        self.assertEqual(len(self.digest.facts), 1)
        self.assertEqual(grounding_catalog(self.digest, self.docs, "", max_facts=0)[0], {})

    def test_prompt_escaping_cannot_bless_an_altered_excerpt(self):
        docs, digest = evidence()
        text = "Patient says ``` knee pain every morning."
        docs[0].pages[0].text = text
        digest.facts[0].quote = text
        self.assertEqual(grounding_catalog(digest, docs, "")[0], {})

    def test_budget_omitted_fact_is_not_referenceable(self):
        with self.assertRaises(LLMParseError):
            self.validate(record_grounding(self.entry), catalog={})

    def test_witness_only_uncertain_observations_need_no_record(self):
        data = complete_grounding("I think this began around 2020; I am unsure.")
        self.assertEqual(self.validate(data, [], {}), data)

    def test_errors_do_not_echo_private_model_fields(self):
        data = record_grounding(self.entry)
        data["supported_observations"][0]["source"] = "PRIVATE-FILENAME.pdf p.999"
        with self.assertRaises(LLMParseError) as ctx:
            self.validate(data)
        self.assertNotIn("PRIVATE", str(ctx.exception))

    def test_range_resolves_quote_on_later_unit_and_canonicalizes_address(self):
        docs, digest = evidence()
        docs[0].pages = [DocumentPage("clinic.txt", 1, "An unrelated earlier clinical assessment."),
                        DocumentPage("clinic.txt", 2, QUOTE)]
        digest.facts[0].source = "clinic.txt p.1-p.2"
        catalog, _ = grounding_catalog(digest, docs, "")
        entry = next(iter(catalog.values()))
        self.assertEqual(entry["source"], "clinic.txt p.2")
        self.assertEqual(entry["source_unit"]["number"], 2)
        self.validate(record_grounding(entry), docs, catalog)

    def test_range_does_not_search_outside_scope_or_other_file_or_kind(self):
        for source in ("clinic.txt p.2-p.3", "other.txt p.1-p.3", "clinic.txt b.1-b.3",
                       "clinic.txt p.3-p.1", "clinic.txt p.1-b.3"):
            digest = copy.deepcopy(self.digest)
            digest.facts[0].source = source
            with self.subTest(source=source):
                self.assertEqual(grounding_catalog(digest, self.docs, "")[0], {})

    def test_range_repeated_quote_ambiguous_or_unreadable_units_are_unresolved(self):
        for docs in ([ExtractedDocument("clinic.txt", [DocumentPage("clinic.txt", 1, QUOTE),
                                                       DocumentPage("clinic.txt", 2, QUOTE)])],
                     self.docs * 2,
                     self.docs + [ExtractedDocument("clinic.txt", [], unreadable_pages=[2])]):
            digest = copy.deepcopy(self.digest)
            digest.facts[0].source = "clinic.txt p.1-p.2"
            with self.subTest(docs=len(docs)):
                self.assertEqual(grounding_catalog(digest, docs, "")[0], {})

    def test_block_range_resolves_only_one_complete_quote(self):
        docs, digest = evidence(BLOCK)
        digest.facts[0].source = "[clinic.txt b.1-b.2]"
        catalog, _ = grounding_catalog(digest, docs, "")
        self.assertEqual(next(iter(catalog.values()))["source"], "clinic.txt b.1")

    def test_citation_markdown_escapes_images_links_html_urls_and_newlines(self):
        data = record_grounding(self.entry)
        payload = "![image](https://example.test/a) [link](https://example.test) <img src=x>\n# heading"
        for section in ("supported_observations", "conflicts", "suggested_inclusions"):
            row = record_grounding(self.entry, section)[section][0]
            row["source"] = payload
            row["quote"] = payload
            data[section] = [row]
        markdown = grounding_markdown(DraftResult(grounding=data, grounding_policy=GROUNDING_SOURCE_POLICY))
        self.assertNotIn("![image]", markdown)
        self.assertNotIn("[link](", markdown)
        self.assertIn("\\<img", markdown)
        self.assertNotIn("https://", markdown)
        self.assertNotIn("\n# heading", markdown)
        self.assertEqual(markdown.count(_citation_display(payload)), 6)

    def test_plain_text_pilot_report_keeps_original_citation_punctuation(self):
        data = record_grounding(self.entry)
        markdown = grounding_markdown(DraftResult(grounding=data, grounding_policy=GROUNDING_SOURCE_POLICY), literal=True)
        self.assertIn("clinic.txt p.1", markdown)
        self.assertIn(QUOTE, markdown)
        self.assertNotIn("clinic\\.txt", markdown)

    def test_policy_and_provenance_survive_saved_result_roundtrip(self):
        data = self.validate(record_grounding(self.entry))
        restored = draft_from_json(draft_to_json(DraftResult(grounding=data, grounding_policy=GROUNDING_SOURCE_POLICY)))
        self.assertEqual(restored.grounding_policy, GROUNDING_SOURCE_POLICY)
        self.assertEqual(restored.grounding, data)
        markdown = grounding_markdown(restored)
        self.assertIn(_citation_display(QUOTE), markdown)
        self.assertIn(_citation_display("clinic.txt p.1"), markdown)
        self.assertNotIn(LEGACY_GROUNDING_SOURCE_NOTICE, markdown)

    def test_legacy_and_unknown_policy_have_explicit_warning(self):
        for raw in ({"grounding": complete_grounding()},
                    {"grounding": complete_grounding(), "grounding_policy": "future-policy"}):
            self.assertIn(LEGACY_GROUNDING_SOURCE_NOTICE, grounding_markdown(draft_from_json(raw)))


class TestDraftingSourceGate(unittest.TestCase):
    def run_pipeline(self, responses, *, pilot=False):
        docs, digest = evidence()
        calls = iter(responses)
        llm = _FakeLLM({"grounding": lambda *_: copy.deepcopy(next(calls))})
        with patch("app.draft.review_medical_records", return_value=digest), \
             patch("app.draft.load_knowledge", return_value="Synthetic checklist"), \
             patch("app.pilot.enabled", return_value=pilot), \
             patch("app.pilot.action_budget", return_value=nullcontext()):
            result = run_draft(llm, docs, WITNESS, "Knee pain, perhaps since 2020.", "Knee", "New")
        return result, llm

    def test_invalid_then_valid_response_retries_before_generation(self):
        docs, digest = evidence()
        catalog, _ = grounding_catalog(digest, docs, "")
        good = record_grounding(next(iter(catalog.values())))
        bad = copy.deepcopy(good)
        bad["supported_observations"][0]["source"] = "absent.pdf p.999"
        result, llm = self.run_pipeline([bad, good])
        self.assertEqual(result.grounding_policy, GROUNDING_SOURCE_POLICY)
        self.assertEqual(llm.calls[:3], [("chat_json", "grounding"), ("chat_json", "grounding"), ("chat", "draft")])

    def test_bounded_exhaustion_stops_before_draft_and_review_in_both_modes(self):
        for pilot in (False, True):
            docs, digest = evidence()
            bad = complete_grounding()
            bad["supported_observations"] = [{"observation": "Pain", "record_support": "Invented support"}]
            llm = _FakeLLM({"grounding": bad})
            with self.subTest(pilot=pilot), \
                 patch("app.draft.review_medical_records", return_value=digest), \
                 patch("app.draft.load_knowledge", return_value="Synthetic checklist"), \
                 patch("app.pilot.enabled", return_value=pilot), \
                 patch("app.pilot.action_budget", return_value=nullcontext()), \
                 self.assertRaises(DraftingError):
                run_draft(llm, docs, WITNESS, "Pain", "Knee", "New")
            self.assertEqual(llm.calls, [("chat_json", "grounding")] * 3)

    def test_uncited_observation_can_complete_in_both_modes(self):
        for pilot in (False, True):
            with self.subTest(pilot=pilot):
                result, _ = self.run_pipeline([complete_grounding("Perhaps since 2020; unsure.")], pilot=pilot)
                self.assertIn("unsure", result.grounding["unverified_observations"][0]["observation"])
                self.assertTrue(result.draft)

    def test_batch_path_rejects_fake_reference_with_existing_bounded_retry(self):
        from tests.test_batch_draft import TestFinalPhaseSemantics
        bad = complete_grounding()
        bad["conflicts"] = [{"observation": "Pain", "record_fact": "Invented", "resolution_note": "Clarify"}]
        with self.assertRaises(LLMParseError):
            TestFinalPhaseSemantics()._run_final("ok", grounding_override=bad)

    def test_batch_reextracts_sources_and_stops_on_bad_then_accepts_valid_evidence(self):
        import re
        import tempfile
        from pathlib import Path
        from scripts import batch_draft
        from tests.test_batch_draft import _make_cfg

        cfg = _make_cfg(Path(tempfile.mkdtemp()))
        docs, digest = evidence(filename="Part1.pdf")
        # Match the configured input glob; extraction returns a named synthetic unit.
        (cfg.records_dir / "Part1.pdf").write_text("Synthetic fixture", encoding="utf-8")
        phases = []
        attempts = 0

        class FakeLLM:
            def chat_json(self, system, user, **kwargs):
                nonlocal attempts
                phases.append(kwargs.get("phase"))
                if kwargs.get("phase") == "review":
                    return {"issues_found": [], "improved_statement": ""}
                attempts += 1
                match = re.search(r"MEDICAL RECORD FACT CATALOG \(JSON\):\n<<<\n(.*?)\n>>>", user, re.DOTALL)
                entry = json.loads(match[1])[0]
                data = record_grounding(entry)
                if attempts == 1:
                    data["supported_observations"][0]["quote"] = "An invented supporting quote from records."
                return data

            def chat(self, system, user, **kwargs):
                phases.append(kwargs.get("phase"))
                return "Synthetic witness statement."

        with patch("app.pipeline_guard.run_with_timeout", side_effect=lambda fn, **kw: fn()), \
             patch("app.documents.records_from_local_path", return_value=(docs, [])) as extract, \
             patch("app.medical_review._merge_facts", side_effect=lambda llm, d, **kw: d.facts), \
             patch("app.medical_review._summarize", return_value="Synthetic summary"), \
             patch.object(batch_draft.time, "sleep"), patch.object(batch_draft, "_wait_for_breaker"):
            result = batch_draft.final_phase(FakeLLM(), cfg, {"batch_99": batch_draft.digest_to_state(digest)})
        extract.assert_called_once_with(str(cfg.records_dir / "Part1.pdf"))
        self.assertEqual(phases, ["grounding", "grounding", "draft", "review"])
        self.assertEqual(result["grounding_policy"], GROUNDING_SOURCE_POLICY)
        self.assertEqual(result["grounding_raw"]["supported_observations"][0]["quote"], QUOTE)

    def test_batch_excludes_failed_and_quarantined_inputs_with_new_or_legacy_state(self):
        import tempfile
        from pathlib import Path
        from scripts import batch_draft
        from tests.test_batch_draft import _make_cfg

        for legacy in (False, True):
            cfg = _make_cfg(Path(tempfile.mkdtemp()), parts_per_batch=1)
            for name in ("Part1.pdf", "Part2.pdf", "Part3.pdf"):
                (cfg.records_dir / name).write_text("Synthetic", encoding="utf-8")
            docs, digest = evidence(filename="Part1.pdf")
            good_state = batch_draft.digest_to_state(digest)
            excluded_state = batch_draft.digest_to_state(MedicalDigest())
            excluded_state["quarantined"] = ["Part3.pdf"]
            if not legacy:
                good_state["source_files"] = ["Part1.pdf"]
                excluded_state["source_files"] = []
            calls = []

            def extract(path):
                calls.append(Path(path).name)
                if Path(path).name != "Part1.pdf":
                    raise RuntimeError("Failed input must never be re-extracted")
                return docs, []

            llm = _FakeLLM({"grounding": complete_grounding()})
            with self.subTest(legacy=legacy), \
                 patch("app.pipeline_guard.run_with_timeout", side_effect=lambda fn, **kw: fn()), \
                 patch("app.documents.records_from_local_path", side_effect=extract), \
                 patch("app.medical_review._merge_facts", side_effect=lambda llm, d, **kw: d.facts), \
                 patch("app.medical_review._summarize", return_value="Synthetic summary"):
                result = batch_draft.final_phase(llm, cfg, {"batch_01": good_state,
                    "batch_02": {"error": "Extraction failed"}, "batch_03": excluded_state})
            self.assertEqual(calls, ["Part1.pdf"])
            self.assertEqual(result["grounding_policy"], GROUNDING_SOURCE_POLICY)

    def test_batch_grounding_sets_explicit_two_attempt_limit_without_sleep(self):
        from tests.test_batch_draft import batch_draft, TestFinalPhaseSemantics
        with patch.object(batch_draft, "_retry_phase", wraps=batch_draft._retry_phase) as retry:
            TestFinalPhaseSemantics()._run_final("ok")
        grounding_calls = [call for call in retry.call_args_list if call.args[1] == "grounding"]
        self.assertEqual(len(grounding_calls), 1)
        self.assertEqual(grounding_calls[0].kwargs["attempts"], 2)
        self.assertEqual(grounding_calls[0].kwargs["base_wait_s"], 0.0)

    def test_cached_batch_final_warns_for_missing_and_unknown_policy_without_redrafting(self):
        import tempfile
        from pathlib import Path
        from app.llm import ChatProbe
        from tests.test_batch_draft import batch_draft, _gate_settings, _retry_argv

        for policy in (None, "unknown-policy", GROUNDING_SOURCE_POLICY):
            with tempfile.TemporaryDirectory() as raw:
                directory = Path(raw)
                final = {"statement": "Original cached statement", "grounding_markdown": "Original analysis",
                         "facts_total": 1, "facts_pre_merge": 1, "review_issues": []}
                if policy is not None:
                    final["grounding_policy"] = policy
                argv = _retry_argv(directory, final_result=final)
                with patch("app.config.load_settings", return_value=_gate_settings()), \
                     patch("app.llm.probe_chat", return_value=ChatProbe(200, "ok")), \
                     patch.object(batch_draft, "final_phase") as generate:
                    self.assertEqual(batch_draft.main(argv), 0)
                generate.assert_not_called()
                markdown = (directory / "out" / "grounding.md").read_text()
                self.assertIn("Original analysis", markdown)
                self.assertEqual(LEGACY_GROUNDING_SOURCE_NOTICE in markdown, policy != GROUNDING_SOURCE_POLICY)

    def test_unreadable_retained_legacy_source_stops_before_generation(self):
        import re
        import tempfile
        from pathlib import Path
        from app.documents import ExtractionError
        from scripts import batch_draft
        from tests.test_batch_draft import _make_cfg
        cfg = _make_cfg(Path(tempfile.mkdtemp()))
        for name in ("Part1.pdf", "Part2.pdf"):
            (cfg.records_dir / name).write_text("Synthetic", encoding="utf-8")
        docs, digest = evidence(filename="Part1.pdf")

        digest.facts.append(MedicalFact("2024", "symptom", QUOTE, "Part2.pdf p.1", QUOTE, document="Part2.pdf", page=1))

        def extract(path):
            if Path(path).name == "Part2.pdf":
                raise ExtractionError("No extractable documents")
            return docs, []

        class FakeLLM:
            def chat_json(self, system, user, **kwargs):
                if kwargs.get("phase") == "review":
                    return {"issues_found": [], "improved_statement": ""}
                match = re.search(r"MEDICAL RECORD FACT CATALOG \(JSON\):\n<<<\n(.*?)\n>>>", user, re.DOTALL)
                return record_grounding(json.loads(match[1])[0])

            def chat(self, *args, **kwargs):
                return "Synthetic statement"

        llm = FakeLLM()
        with patch("app.pipeline_guard.run_with_timeout", side_effect=lambda fn, **kw: fn()), \
             patch("app.documents.records_from_local_path", side_effect=extract), \
             patch("app.medical_review._merge_facts", side_effect=lambda llm, d, **kw: d.facts), \
             patch("app.medical_review._summarize", return_value="Synthetic summary"), \
             patch.object(batch_draft, "_wait_for_breaker"), \
             patch.object(llm, "chat_json", wraps=llm.chat_json) as generate, \
             self.assertRaisesRegex(ValueError, "could not be re-extracted"):
            batch_draft.final_phase(llm, cfg, {"batch_01": batch_draft.digest_to_state(digest)})
        generate.assert_not_called()

    def test_new_batch_checkpoint_records_only_inputs_that_produced_documents(self):
        import tempfile
        from pathlib import Path
        from scripts import batch_draft
        from tests.test_batch_draft import _make_cfg
        cfg = _make_cfg(Path(tempfile.mkdtemp()))
        paths = [cfg.records_dir / name for name in ("Part1.pdf", "Part2.pdf")]
        for path in paths:
            path.write_text("Synthetic", encoding="utf-8")
        doc = ExtractedDocument("Part1.pdf", [DocumentPage("Part1.pdf", 1, QUOTE)])
        with patch("app.documents.records_from_local_path", return_value=([doc], ["Part2.pdf unavailable"])), \
             patch("app.medical_review.review_medical_records", return_value=MedicalDigest()), \
             patch("app.pipeline_guard.run_with_timeout", side_effect=lambda fn, *args, **kw: fn(*args)):
            state, _ = batch_draft.digest_group(object(), cfg, "batch_01", paths)
        self.assertEqual(state["source_files"], ["Part1.pdf"])

    def test_missing_legacy_source_stops_generation_instead_of_silently_losing_support(self):
        import tempfile
        from pathlib import Path
        from scripts import batch_draft
        from tests.test_batch_draft import _make_cfg
        cfg = _make_cfg(Path(tempfile.mkdtemp()))
        docs, digest = evidence(filename="Missing.pdf")
        llm = _FakeLLM({"grounding": complete_grounding()})
        with patch("app.pipeline_guard.run_with_timeout", side_effect=lambda fn, **kw: fn()), \
             patch("app.medical_review._merge_facts", side_effect=lambda llm, d, **kw: d.facts), \
             patch("app.medical_review._summarize", return_value="Synthetic summary"), \
             self.assertRaisesRegex(ValueError, "legacy source input is unavailable"):
            batch_draft.final_phase(llm, cfg, {"batch_99": batch_draft.digest_to_state(digest)})
        self.assertEqual(llm.calls, [])

    def test_unknown_legacy_fact_sources_produce_explicit_coverage_warning(self):
        from tests.test_batch_draft import TestFinalPhaseSemantics
        result = TestFinalPhaseSemantics()._run_final("ok", legacy_unknown_sources=True)
        self.assertEqual(result["legacy_source_facts_unresolved"], 1)
        self.assertIn("Legacy source coverage", result["grounding_markdown"])

    def test_batch_witness_only_result_records_new_policy(self):
        from tests.test_batch_draft import TestFinalPhaseSemantics
        result = TestFinalPhaseSemantics()._run_final("ok")
        self.assertEqual(result["grounding_policy"], GROUNDING_SOURCE_POLICY)
        self.assertNotIn(LEGACY_GROUNDING_SOURCE_NOTICE, result["grounding_markdown"])
