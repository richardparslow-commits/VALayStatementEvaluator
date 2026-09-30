"""Incomplete model analysis must not masquerade as a completed witness review."""
from copy import deepcopy
from contextlib import nullcontext
import unittest
from unittest.mock import patch

from tests import hermetic  # noqa: F401
from tests.grounding_fixtures import complete_grounding
from tests.test_draft import _FakeLLM, _fake_digest, _doc, WITNESS
from app.draft import DraftResult, _normalize_grounding, grounding_markdown, run_draft
from app.drafting_service import DraftingError
from app.job_payload import draft_from_json, draft_to_json
from app.llm import LLMParseError
from app.views.follow_up import draft_follow_up_questions


class TestGroundingValidation(unittest.TestCase):
    def test_empty_rows_are_rejected_in_every_analysis_section(self):
        for section in ("supported_observations", "unverified_observations", "conflicts",
                        "suggested_inclusions", "topic_coverage"):
            with self.subTest(section=section):
                data = complete_grounding()
                data[section] = [{}]
                with self.assertRaises(LLMParseError):
                    _normalize_grounding(data)

    def test_each_top_level_section_is_required(self):
        for section in complete_grounding():
            with self.subTest(section=section):
                data = complete_grounding()
                del data[section]
                with self.assertRaises(LLMParseError):
                    _normalize_grounding(data)

    def test_null_sections_cannot_become_completed_empty_checks(self):
        for section in complete_grounding():
            with self.subTest(section=section):
                data = complete_grounding()
                data[section] = None
                with self.assertRaises(LLMParseError):
                    _normalize_grounding(data)

    def test_all_observation_and_source_fields_are_required(self):
        rows = {
            "supported_observations": {"observation": "Pain observed.", "record_support": "clinic.txt b.1"},
            "unverified_observations": {"observation": "Pain observed.", "action": "Confirm with witness."},
            "conflicts": {"observation": "Pain observed.", "record_fact": "No pain reported.",
                          "resolution_note": "Ask the witness to clarify."},
            "suggested_inclusions": {"fact": "Brace prescribed.", "source": "clinic.txt b.1"},
        }
        for section, row in rows.items():
            for field in row:
                with self.subTest(section=section, field=field):
                    data = complete_grounding()
                    data[section] = [dict(row)]
                    del data[section][0][field]
                    with self.assertRaises(LLMParseError):
                        _normalize_grounding(data)

    def test_blank_or_nontext_content_is_rejected(self):
        for value in ("", " \n ", None, 0, False, [], {}):
            with self.subTest(value=value):
                data = complete_grounding()
                data["unverified_observations"][0]["observation"] = value
                with self.assertRaises(LLMParseError):
                    _normalize_grounding(data)

    def test_topic_flags_require_json_booleans(self):
        for field in ("applicable", "covered"):
            for value in ("false", "true", 0, 1, None, [], {}):
                with self.subTest(field=field, value=value):
                    data = complete_grounding()
                    data["topic_coverage"][0][field] = value
                    with self.assertRaises(LLMParseError):
                        _normalize_grounding(data)

    def test_missing_topic_flags_are_rejected(self):
        for field in ("applicable", "covered", "prompt_for_witness", "topic"):
            with self.subTest(field=field):
                data = complete_grounding()
                del data["topic_coverage"][0][field]
                with self.assertRaises(LLMParseError):
                    _normalize_grounding(data)

    def test_each_checklist_topic_is_required(self):
        for index in range(15):
            with self.subTest(index=index):
                data = complete_grounding()
                del data["topic_coverage"][index]
                with self.assertRaises(LLMParseError):
                    _normalize_grounding(data)

    def test_duplicate_topic_cannot_replace_a_missing_topic(self):
        data = complete_grounding()
        data["topic_coverage"][-1] = dict(data["topic_coverage"][0])
        with self.assertRaises(LLMParseError):
            _normalize_grounding(data)

    def test_unknown_or_ambiguous_topic_labels_are_rejected(self):
        for label in ("P. Extra", "Hazards", "ABC", "A1. Topic", "a. Topic", ""):
            with self.subTest(label=label):
                data = complete_grounding()
                data["topic_coverage"][0]["topic"] = label
                with self.assertRaises(LLMParseError):
                    _normalize_grounding(data)

    def test_inapplicable_topic_cannot_be_covered(self):
        data = complete_grounding()
        data["topic_coverage"][0]["covered"] = True
        with self.assertRaises(LLMParseError):
            _normalize_grounding(data)

    def test_applicable_uncovered_topic_requires_a_question(self):
        data = complete_grounding()
        data["topic_coverage"][0]["applicable"] = True
        with self.assertRaises(LLMParseError):
            _normalize_grounding(data)

    def test_question_cannot_contradict_covered_or_inapplicable_flags(self):
        for applicable, covered in ((True, True), (False, False)):
            with self.subTest(applicable=applicable, covered=covered):
                data = complete_grounding()
                data["topic_coverage"][0].update(applicable=applicable, covered=covered,
                                                  prompt_for_witness="Who helps with stairs?")
                with self.assertRaises(LLMParseError):
                    _normalize_grounding(data)

    def test_strengthening_questions_require_nonempty_strings(self):
        for value in ("", " \n", False, 0, None, {}, []):
            with self.subTest(value=value):
                data = complete_grounding()
                data["strengthening_questions"] = [value]
                with self.assertRaises(LLMParseError):
                    _normalize_grounding(data)

    def test_topics_are_ordered_without_modifying_the_response(self):
        data = complete_grounding("I think pain began around 2020.")
        data["topic_coverage"].reverse()
        original = deepcopy(data)
        result = _normalize_grounding(data, observations_present=True)
        self.assertEqual(data, original)
        self.assertEqual([row["topic"][0] for row in result["topic_coverage"]], list("ABCDEFGHIJKLMNO"))
        self.assertEqual(result["unverified_observations"], original["unverified_observations"])

    def test_record_suggestions_cannot_replace_witness_analysis(self):
        data = complete_grounding("")
        data["suggested_inclusions"] = [{"fact": "Brace prescribed.", "source": "clinic.txt b.1"}]
        with self.assertRaises(LLMParseError):
            _normalize_grounding(data, observations_present=True)

    def test_no_observations_may_have_empty_observation_sections(self):
        data = complete_grounding("")
        self.assertEqual(_normalize_grounding(data, observations_present=False), data)

    def test_invalid_analysis_stops_public_pipeline_before_draft_or_review(self):
        invalid = complete_grounding()
        invalid["supported_observations"] = [{}]
        for pilot_mode in (False, True):
            with self.subTest(pilot_mode=pilot_mode):
                llm = _FakeLLM(overrides={"grounding": invalid})
                with patch("app.pilot.enabled", return_value=pilot_mode), \
                        patch("app.pilot.action_budget", return_value=nullcontext()), \
                        patch("app.draft.review_medical_records", return_value=_fake_digest()), \
                        patch("app.draft.load_knowledge", return_value="Synthetic guide"):
                    with self.assertRaises(DraftingError) as error:
                        run_draft(llm, [_doc()], WITNESS, "Daily knee pain.", "knee pain", "Service connection")
                self.assertEqual(error.exception.error_kind, "parse_error")
                self.assertNotIn(("chat", "draft"), llm.calls)
                self.assertNotIn(("chat_json", "review"), llm.calls)

    def test_valid_analysis_keeps_unverified_account_and_follow_up(self):
        data = complete_grounding("I think the pain started around 2020.")
        data["topic_coverage"][0].update(applicable=True, covered=False,
                                          prompt_for_witness="What happens on stairs?")
        result = DraftResult(grounding=_normalize_grounding(data, observations_present=True))
        restored = draft_from_json(draft_to_json(result))
        self.assertEqual(restored.grounding, data)
        self.assertEqual(draft_follow_up_questions(restored), [{
            "topic": "A. Synthetic topic A", "question": "What happens on stairs?",
        }])
        self.assertIn("still legitimate lay evidence", grounding_markdown(restored))
        self.assertIn("I think the pain started around 2020.", grounding_markdown(restored))

    def test_complete_response_reaches_drafting_in_both_modes(self):
        for pilot_mode in (False, True):
            with self.subTest(pilot_mode=pilot_mode):
                data = complete_grounding("I think the pain started around 2020.")
                llm = _FakeLLM(overrides={"grounding": data})
                with patch("app.pilot.enabled", return_value=pilot_mode), \
                        patch("app.pilot.action_budget", return_value=nullcontext()), \
                        patch("app.draft.review_medical_records", return_value=_fake_digest()), \
                        patch("app.draft.load_knowledge", return_value="Synthetic guide"):
                    result = run_draft(llm, [_doc()], WITNESS, "I think the pain started around 2020.",
                                       "knee pain", "Service connection")
                self.assertEqual(result.grounding, data)
                self.assertIn(("chat", "draft"), llm.calls)
                self.assertTrue(result.draft)

    def test_legacy_saved_boolean_strings_do_not_claim_topic_coverage(self):
        old = DraftResult(grounding={"topic_coverage": [{
            "topic": "A. Synthetic private topic", "applicable": "false", "covered": "false",
            "prompt_for_witness": "Private question",
        }]})
        rendered = grounding_markdown(draft_from_json(draft_to_json(old)))
        self.assertIn("Re-run", rendered)
        self.assertNotIn("Covered by the witness", rendered)
        self.assertEqual(draft_follow_up_questions(old), [])


if __name__ == "__main__":
    unittest.main()
