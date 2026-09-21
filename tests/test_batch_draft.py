"""The promoted batched-draft script, exercised without any live LLM call.

``scripts/batch_draft.py`` is the operator tool for full C-files too large for
one run: it digests timeout-sized batches, persists resumable state, and drafts
once over the combined evidence. What the tests own here is the part that must
be right *before* anyone points it at 1,800 real pages:

* the batch planner (a bad batch split silently changes which records end up
  quarantined together);
* digest-state round-tripping, because state.json is the only thing between an
  interrupted run and hours of lost LLM spend;
* the bisect-or-quarantine policy, driven through the real ``digest_group``
  code with the pipeline stubbed — a file that fails alone must not quarantine
  its siblings (the measured failure: one dense chunk's 21k-char truncated
  JSON response);
* merge arithmetic and the final phase's failure semantics — a failed review
  call keeps the finished draft, exactly like ``run_draft``.

Everything runs on a stub LLM and staged temp files; no network.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "batch_draft", PROJECT_ROOT / "scripts" / "batch_draft.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass processing looks the module up in
    # sys.modules, and a dangling module would crash the decorator on 3.13.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


batch_draft = _load_script()


def _make_cfg(tmp: Path, **overrides):
    defaults = dict(
        records_dir=tmp / "records",
        out_dir=tmp / "out",
        condition="PTSD",
        claim_type="Initial claim - service connection",
        witness={"name": "[Witness Name]", "relationship": "Spouse"},
        observations="I have watched the veteran decline since 2022.",
    )
    defaults.update(overrides)
    cfg = batch_draft.BatchConfig(**defaults)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.records_dir.mkdir(parents=True, exist_ok=True)
    return cfg


class TestBatchPlanning(unittest.TestCase):
    def test_splits_evenly_and_keeps_a_remainder(self) -> None:
        files = [Path(f"p{i:02d}.pdf") for i in range(7)]
        batches = batch_draft.plan_batches(files, 3)
        self.assertEqual([len(b) for b in batches], [3, 3, 1])

    def test_exact_multiple_has_no_empty_tail(self) -> None:
        files = [Path(f"p{i}.pdf") for i in range(4)]
        self.assertEqual([len(b) for b in batch_draft.plan_batches(files, 2)], [2, 2])

    def test_empty_input_makes_no_batches(self) -> None:
        self.assertEqual(batch_draft.plan_batches([], 13), [])

    def test_per_batch_below_one_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            batch_draft.plan_batches([Path("a.pdf")], 0)


class TestDigestStateRoundTrip(unittest.TestCase):
    def test_round_trip_preserves_facts_and_counters(self) -> None:
        from app.medical_review import MedicalDigest, MedicalFact

        digest = MedicalDigest(
            facts=[
                MedicalFact(date="2023-04-01", type="medication",
                            description="Sertraline increased", source="p. 12",
                            document="part03.pdf", page=12),
                MedicalFact(date="circa 2023", type="symptom",
                            description="Nightmares most nights", source="p. 30"),
            ],
            conditions=["PTSD"], providers=["Dr. Reyes"],
            pages_reviewed=260, chunks_reviewed=8, duplicates_skipped=2,
            unreadable_pages=1, pages_in_files=260, chunks_without_facts=1,
        )
        state = batch_draft.digest_to_state(digest)
        back = batch_draft.digest_from_state(state)
        self.assertEqual(len(back.facts), 2)
        self.assertEqual(back.facts[0].description, "Sertraline increased")
        self.assertEqual(back.facts[0].document, "part03.pdf")
        self.assertEqual(back.facts[0].page, 12)
        self.assertEqual(back.facts[1].date, "circa 2023")
        self.assertEqual(back.pages_reviewed, 260)
        self.assertEqual(back.chunks_reviewed, 8)
        self.assertEqual(back.unreadable_pages, 1)
        self.assertEqual(back.pages_in_files, 260)
        self.assertEqual(back.chunks_without_facts, 1)
        self.assertEqual(back.conditions, ["PTSD"])

    def test_state_file_round_trips_through_json(self) -> None:
        from app.medical_review import MedicalDigest, MedicalFact

        digest = MedicalDigest(
            facts=[MedicalFact(date="2022-01-01", type="symptom",
                               description="Startle response", source="p. 5")],
            pages_in_files=10, pages_reviewed=10,
        )
        cfg = _make_cfg(Path(tempfile.mkdtemp()))
        state = {"batches": {"batch_01": batch_draft.digest_to_state(digest)},
                 "final": None}
        batch_draft.save_state(cfg, state)
        reloaded = batch_draft.load_state(cfg)
        back = batch_draft.digest_from_state(reloaded["batches"]["batch_01"])
        self.assertEqual(back.facts[0].description, "Startle response")

    def test_load_state_missing_file_starts_clean(self) -> None:
        cfg = _make_cfg(Path(tempfile.mkdtemp()))
        self.assertEqual(batch_draft.load_state(cfg), {"batches": {}, "final": None})

    def test_fresh_flag_ignores_existing_state(self) -> None:
        cfg = _make_cfg(Path(tempfile.mkdtemp()))
        batch_draft.save_state(cfg, {"batches": {"batch_01": {"facts": [1]}},
                                     "final": None})
        self.assertEqual(batch_draft.load_state(cfg, fresh=True),
                         {"batches": {}, "final": None})


class TestMergeStates(unittest.TestCase):
    def test_union_of_facts_and_counters(self) -> None:
        a = dict(batch_draft.EMPTY_DIGEST_STATE, facts=[{"description": "x"}],
                 conditions=["PTSD"], pages_reviewed=10, pages_in_files=12,
                 unreadable_pages=1, files=2, pages=11, duration_s=60.0,
                 quarantined=["bad.pdf"])
        b = dict(batch_draft.EMPTY_DIGEST_STATE, facts=[{"description": "y"}],
                 conditions=["PTSD", "MDD"], pages_reviewed=5, pages_in_files=8,
                 unreadable_pages=0, files=1, pages=7, duration_s=30.0,
                 quarantined=[])
        m = batch_draft.merge_states(a, b)
        self.assertEqual(len(m["facts"]), 2)
        self.assertEqual(m["conditions"], ["MDD", "PTSD"])
        self.assertEqual(m["pages_reviewed"], 15)
        self.assertEqual(m["unreadable_pages"], 1)
        self.assertEqual(m["files"], 3)
        self.assertEqual(m["duration_s"], 90.0)
        self.assertEqual(m["quarantined"], ["bad.pdf"])
        self.assertEqual(m["coverage_ratio"], round(1 - 1 / 20, 4))

    def test_merge_tolerates_missing_optional_keys(self) -> None:
        a = dict(batch_draft.EMPTY_DIGEST_STATE, facts=[])
        b = dict(batch_draft.EMPTY_DIGEST_STATE, facts=[{"description": "y"}])
        m = batch_draft.merge_states(a, b)
        self.assertEqual(len(m["facts"]), 1)
        self.assertEqual(m["coverage_ratio"], 1.0)


class TestBisectPolicy(unittest.TestCase):
    """One failing file must not quarantine its group."""

    def _stub_review(self, fail_names: set[str]):
        """Patch records_from_local_path + review_medical_records so the named
        file's group raises, everything else returns a minimal digest."""
        from app.medical_review import MedicalDigest, MedicalFact

        def fake_docs(path_str):
            path = Path(path_str)
            from app.documents import DocumentPage, ExtractedDocument

            docs = [
                ExtractedDocument(
                    filename=f.name,
                    pages=[DocumentPage(f.name, 1, f"{f.name} EVT note.")],
                )
                for f in sorted(path.glob("*.pdf"))
            ]
            return docs, []

        def fake_review(llm, docs, progress=None):
            names = {d.filename for d in docs}
            bad = sorted(names & fail_names)
            if bad:
                raise RuntimeError(f"digest overflow in {bad[0]}")
            return MedicalDigest(
                facts=[
                    MedicalFact(date="2023-01-01", type="symptom",
                                description=f"note from {d.filename}", source="p. 1")
                    for d in docs
                ],
                pages_reviewed=len(docs), pages_in_files=len(docs),
                chunks_reviewed=1,
            )

        return (
            mock.patch("app.documents.records_from_local_path", side_effect=fake_docs),
            mock.patch("app.medical_review.review_medical_records", side_effect=fake_review),
        )

    def _run_group(self, cfg, fail_names):
        p1, p2 = self._stub_review(fail_names)
        with p1, p2:
            return batch_draft.digest_group(
                object(),  # the llm is passed straight to the stubbed review
                cfg, "batch_01", sorted(cfg.records_dir.glob("*.pdf")),
            )

    def test_isolated_failure_quarantines_only_itself(self) -> None:
        cfg = _make_cfg(Path(tempfile.mkdtemp()))
        for name in ("good_a.pdf", "poison.pdf", "good_b.pdf"):
            (cfg.records_dir / name).write_text("EVT note.", encoding="utf-8")

        state, excluded = self._run_group(cfg, {"poison.pdf"})
        self.assertEqual(excluded, ["poison.pdf"])
        self.assertEqual(state["quarantined"], ["poison.pdf"])
        self.assertEqual(len(state["facts"]), 2)
        # files counts every staged file, including the quarantined one.
        self.assertEqual(state["files"], 3)

    def test_all_files_failing_quarantines_everything(self) -> None:
        cfg = _make_cfg(Path(tempfile.mkdtemp()))
        for name in ("a.pdf", "b.pdf"):
            (cfg.records_dir / name).write_text("EVT note.", encoding="utf-8")

        state, excluded = self._run_group(cfg, {"a.pdf", "b.pdf"})
        self.assertEqual(excluded, ["a.pdf", "b.pdf"])
        self.assertEqual(state["facts"], [])

    def test_no_failure_returns_full_group(self) -> None:
        cfg = _make_cfg(Path(tempfile.mkdtemp()))
        for name in ("a.pdf", "b.pdf"):
            (cfg.records_dir / name).write_text("EVT note.", encoding="utf-8")

        state, excluded = self._run_group(cfg, set())
        self.assertEqual(excluded, [])
        self.assertEqual(len(state["facts"]), 2)
        self.assertEqual(state["files"], 2)


class TestFinalPhaseSemantics(unittest.TestCase):
    """The final phase must treat a failed review like run_draft does."""

    DRAFT = "I certify the statements above are true and correct."
    # A reviewer may only ADD detail; run_draft's structural guard requires the
    # improved statement to clear the 200-char minimum-length check and to end
    # with the original's certification closing.
    IMPROVED = (
        "Additional observed detail about the veteran's daily functioning. "
        "He requires reminders for medication and supervision near the stove. "
        "Since 2022 his nightmares occur most nights and he sleeps in the spare room. "
    ) + DRAFT

    def _run_final(self, review_behavior: str) -> dict:
        from app.medical_review import MedicalDigest, MedicalFact

        cfg = _make_cfg(Path(tempfile.mkdtemp()))
        batch_state = batch_draft.digest_to_state(MedicalDigest(
            facts=[MedicalFact(date="2023-01-01", type="symptom",
                               description="Nightmares most nights", source="p. 1")],
            pages_in_files=10, pages_reviewed=10,
        ))

        draft_text = self.DRAFT
        improved_text = self.IMPROVED

        class FakeLLM:
            def __init__(self):
                self.chat_calls = 0
                self.chat_json_calls = 0

            def chat(self, system, user, **kw):
                self.chat_calls += 1
                return draft_text

            def chat_json(self, system, user, **kw):
                self.chat_json_calls += 1
                if kw.get("phase") == "review":
                    if review_behavior == "raise":
                        raise RuntimeError("provider 500")
                    if review_behavior == "ok":
                        return {"issues_found": ["add dates"],
                                "improved_statement": improved_text}
                    return {}
                # grounding (and any other JSON phase) returns a plain payload
                return {"supported_observations": []}

        llm = FakeLLM()
        with mock.patch("app.pipeline_guard.run_with_timeout",
                        side_effect=lambda fn, *a, **k: fn()), \
             mock.patch("app.medical_review._summarize", return_value="Summary."), \
             mock.patch("app.medical_review._merge_facts",
                        side_effect=lambda llm2, d, progress=None: d.facts):
            result = batch_draft.final_phase(llm, cfg, {"batch_01": batch_state})
        result["_calls"] = (llm.chat_calls, llm.chat_json_calls)  # grounding + review
        return result

    def test_review_failure_keeps_the_draft(self) -> None:
        result = self._run_final("raise")
        self.assertEqual(result["_calls"], (1, 2))  # grounding chat_json + failed review
        self.assertEqual(result["statement"], self.DRAFT)
        self.assertTrue(
            any("Self-review pass was skipped" in i for i in result["review_issues"]))
        self.assertEqual(result["facts_total"], 1)
        self.assertEqual(result["summary"], "Summary.")

    def test_review_adopted_when_structurally_sound(self) -> None:
        result = self._run_final("ok")
        self.assertEqual(result["statement"], self.IMPROVED)
        self.assertEqual(result["review_issues"], ["add dates"])

    def test_non_dict_review_preserves_the_draft(self) -> None:
        result = self._run_final("empty")
        self.assertEqual(result["statement"], self.DRAFT)
        self.assertTrue(
            any("Self-review not applied" in i for i in result["review_issues"]))


class TestConfigFromArgs(unittest.TestCase):
    def test_flags_only(self) -> None:
        args = batch_draft.build_parser().parse_args([
            "--records", "recs", "--condition", "PTSD",
            "--claim-type", "Initial claim", "--out", "o",
        ])
        cfg = batch_draft.config_from_args(args)
        self.assertEqual(cfg.condition, "PTSD")
        self.assertEqual(cfg.claim_type, "Initial claim")
        self.assertEqual(cfg.parts_per_batch, 13)
        self.assertTrue(cfg.run_final)
        self.assertEqual(cfg.state_path, Path("o") / "state.json")

    def test_witness_json_supplies_defaults_flags_override(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        wj = tmp / "w.json"
        wj.write_text(json.dumps({
            "witness": {"name": "Jane Doe", "relationship": "Spouse"},
            "condition": "MDD",
            "claim_type": "Increase",
            "observations": "She has not slept through the night since 2021.",
        }), encoding="utf-8")
        obs = tmp / "obs.txt"
        obs.write_text("Flag observations win.", encoding="utf-8")

        args = batch_draft.build_parser().parse_args([
            "--records", "recs", "--witness-json", str(wj),
            "--observations", str(obs),
            "--condition", "PTSD",  # overrides the JSON's "MDD"
        ])
        cfg = batch_draft.config_from_args(args)
        self.assertEqual(cfg.condition, "PTSD")
        self.assertEqual(cfg.claim_type, "Increase")  # from JSON, no flag
        self.assertEqual(cfg.witness["name"], "Jane Doe")
        self.assertEqual(cfg.observations, "Flag observations win.")

    def test_no_final_flag_sets_digest_only(self) -> None:
        args = batch_draft.build_parser().parse_args([
            "--records", "recs", "--condition", "PTSD", "--claim-type", "x",
            "--no-final",
        ])
        cfg = batch_draft.config_from_args(args)
        self.assertFalse(cfg.run_final)

    def test_fresh_flag_parses(self) -> None:
        args = batch_draft.build_parser().parse_args([
            "--records", "recs", "--condition", "PTSD", "--claim-type", "x",
            "--fresh",
        ])
        cfg = batch_draft.config_from_args(args)
        self.assertTrue(args.fresh)
        self.assertEqual(cfg.batch_timeout_s, 1740)  # default preserved


if __name__ == "__main__":
    unittest.main()
