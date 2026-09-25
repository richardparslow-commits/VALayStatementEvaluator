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
import time
import unittest
from pathlib import Path
from unittest import mock

from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app import config  # noqa: E402
from app.llm import ChatProbe  # noqa: E402

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


class TestEmptyStateFreshness(unittest.TestCase):
    """Empty/quarantined batches must never alias EMPTY_DIGEST_STATE's lists."""

    def test_fresh_states_do_not_alias_each_other_or_the_constant(self) -> None:
        a = batch_draft._fresh_empty_state()
        b = batch_draft._fresh_empty_state()
        a["facts"].append({"description": "x"})
        a["conditions"].append("PTSD")
        a["providers"].append("VA")
        self.assertEqual(b["facts"], [])
        self.assertEqual(b["conditions"], [])
        self.assertEqual(b["providers"], [])
        self.assertEqual(batch_draft.EMPTY_DIGEST_STATE["facts"], [])
        self.assertEqual(batch_draft.EMPTY_DIGEST_STATE["conditions"], [])
        self.assertEqual(batch_draft.EMPTY_DIGEST_STATE["providers"], [])

    def test_fresh_state_keeps_the_pinned_shape(self) -> None:
        self.assertEqual(batch_draft._fresh_empty_state(),
                         batch_draft.EMPTY_DIGEST_STATE)

    def test_empty_group_guard_returns_unshared_state(self) -> None:
        cfg = _make_cfg(Path(tempfile.mkdtemp()))
        state, excluded = batch_draft.digest_group(object(), cfg, "batch_01", [])
        self.assertEqual(excluded, [])
        state["facts"].append({"description": "x"})
        self.assertEqual(batch_draft.EMPTY_DIGEST_STATE["facts"], [])


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

    def test_quarantine_state_does_not_alias_the_constant(self) -> None:
        cfg = _make_cfg(Path(tempfile.mkdtemp()))
        (cfg.records_dir / "poison.pdf").write_text("EVT note.", encoding="utf-8")

        state, excluded = self._run_group(cfg, {"poison.pdf"})
        self.assertEqual(excluded, ["poison.pdf"])
        # Mutating a quarantined batch's state must not touch the constant or
        # any other batch's lists.
        state["conditions"].append("PTSD")
        state["facts"].append({"description": "x"})
        self.assertEqual(batch_draft.EMPTY_DIGEST_STATE["conditions"], [])
        self.assertEqual(batch_draft.EMPTY_DIGEST_STATE["facts"], [])

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

    def _run_final(self, review_behavior: str, *,
                   bad_grounding_once: bool = False) -> dict:
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
                self.grounding_tried = False

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
                if bad_grounding_once and not self.grounding_tried:
                    self.grounding_tried = True
                    # Parseable JSON of the wrong shape: chat_json succeeds,
                    # _normalize_grounding then rejects it — the 2026-09-23
                    # 23:21 failure mode (a bare array).
                    return ["not", "an", "object"]
                return {"supported_observations": []}

        llm = FakeLLM()
        # The final phase's own resilience (_retry_phase sleeps between outer
        # attempts and polls the breaker inside _wait_for_breaker) is covered
        # by its own tests; this helper drives prompt plumbing, so both are
        # stubbed to keep a failing review call from stalling the suite.
        with mock.patch("app.pipeline_guard.run_with_timeout",
                        side_effect=lambda fn, *a, **k: fn()), \
             mock.patch("app.medical_review._summarize", return_value="Summary."), \
             mock.patch("app.medical_review._merge_facts",
                        side_effect=lambda llm2, d, progress=None, **kw: d.facts), \
             mock.patch.object(batch_draft.time, "sleep"), \
             mock.patch.object(batch_draft, "_wait_for_breaker"):
            result = batch_draft.final_phase(llm, cfg, {"batch_01": batch_state})
        result["_calls"] = (llm.chat_calls, llm.chat_json_calls)  # grounding + review
        return result

    def test_review_failure_keeps_the_draft(self) -> None:
        result = self._run_final("raise")
        # grounding chat_json once, then the review's two outer attempts (the
        # polish call retries once before its graceful keep-the-draft fallback)
        self.assertEqual(result["_calls"], (1, 3))
        self.assertEqual(result["statement"], self.DRAFT)
        self.assertTrue(
            any("Self-review pass was skipped" in i for i in result["review_issues"]))
        self.assertEqual(result["facts_total"], 1)
        self.assertEqual(result["summary"], "Summary.")

    def test_review_adopted_when_structurally_sound(self) -> None:
        result = self._run_final("ok")
        self.assertEqual(result["statement"], self.IMPROVED)
        self.assertEqual(result["review_issues"], ["add dates"])

    def test_a_shape_invalid_grounding_is_retried_not_fatal(self) -> None:
        # The normalizer must run INSIDE _retry_phase: a parseable-but-wrong-
        # shape grounding answer has to re-roll like any other grounding
        # failure, not kill the final phase after the merge is banked.
        result = self._run_final("ok", bad_grounding_once=True)
        # grounding chat_json x2 (1 bad shape + 1 good) + review x1, draft x1
        self.assertEqual(result["_calls"], (1, 3))
        self.assertEqual(result["statement"], self.IMPROVED)

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


class TestResumableState(unittest.TestCase):
    """The sharded, atomic state format — and the legacy file it must still read."""

    def _two_batch_state(self) -> dict:
        from app.medical_review import MedicalDigest, MedicalFact

        digest = batch_draft.digest_to_state(MedicalDigest(
            facts=[MedicalFact(date="2023-01-01", type="symptom",
                               description="Nightmares most nights", source="p. 1")],
            pages_in_files=10, pages_reviewed=10,
        ))
        return {"batches": {"batch_01": digest,
                            "batch_02": {"error": "RuntimeError: digest overflow"}},
                "final": None}

    def test_save_writes_one_shard_per_batch_plus_index(self) -> None:
        cfg = _make_cfg(Path(tempfile.mkdtemp()))
        batch_draft.save_state(cfg, self._two_batch_state())
        shards = cfg.out_dir / "state"
        self.assertTrue((shards / "batch_01.json.gz").is_file())
        self.assertTrue((shards / "batch_02.json.gz").is_file())
        self.assertTrue((shards / "index.json").is_file())
        self.assertFalse((shards / "final.json.gz").exists())  # final is None
        index = json.loads((shards / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(index["batches"]["batch_01"]["facts"], 1)
        self.assertIn("digest overflow", index["batches"]["batch_02"]["error"])

    def test_load_reads_shards_without_needing_the_index(self) -> None:
        cfg = _make_cfg(Path(tempfile.mkdtemp()))
        state = self._two_batch_state()
        batch_draft.save_state(cfg, state)
        (cfg.out_dir / "state" / "index.json").unlink()  # index lost entirely
        reloaded = batch_draft.load_state(cfg)
        self.assertEqual(set(reloaded["batches"]), {"batch_01", "batch_02"})
        back = batch_draft.digest_from_state(reloaded["batches"]["batch_01"])
        self.assertEqual(back.facts[0].description, "Nightmares most nights")

    def test_a_corrupt_shard_reads_as_absent_instead_of_killing_the_resume(self) -> None:
        cfg = _make_cfg(Path(tempfile.mkdtemp()))
        batch_draft.save_state(cfg, self._two_batch_state())
        (cfg.out_dir / "state" / "batch_02.json.gz").write_bytes(b"truncated garbage")
        reloaded = batch_draft.load_state(cfg)
        self.assertEqual(set(reloaded["batches"]), {"batch_01"})

    def test_final_phase_shard_round_trips(self) -> None:
        cfg = _make_cfg(Path(tempfile.mkdtemp()))
        state = self._two_batch_state()
        state["final"] = {"statement": "I certify...", "facts_total": 1}
        batch_draft.save_state(cfg, state)
        reloaded = batch_draft.load_state(cfg)
        self.assertEqual(reloaded["final"]["statement"], "I certify...")

    def test_legacy_monolithic_state_json_still_loads(self) -> None:
        """Runs interrupted before the sharded format resume unchanged."""
        cfg = _make_cfg(Path(tempfile.mkdtemp()))
        legacy = {"batches": {"batch_01": {"facts": [{"description": "old run"}]}},
                  "final": None}
        cfg.state_path.write_text(json.dumps(legacy), encoding="utf-8")
        reloaded = batch_draft.load_state(cfg)
        self.assertEqual(reloaded["batches"]["batch_01"]["facts"][0]["description"],
                         "old run")

    def test_a_truncated_legacy_state_json_starts_clean(self) -> None:
        cfg = _make_cfg(Path(tempfile.mkdtemp()))
        cfg.state_path.write_text('{"batches": {"batch_01": {"fac', encoding="utf-8")
        self.assertEqual(batch_draft.load_state(cfg), {"batches": {}, "final": None})

    def test_saving_after_a_legacy_load_reshards_the_run(self) -> None:
        cfg = _make_cfg(Path(tempfile.mkdtemp()))
        cfg.state_path.write_text(
            json.dumps({"batches": {"batch_01": {"facts": [{"description": "old"}]}},
                       "final": None}),
            encoding="utf-8",
        )
        state = batch_draft.load_state(cfg)
        state["batches"]["batch_02"] = self._two_batch_state()["batches"]["batch_01"]
        batch_draft.save_state(cfg, state)
        shards = cfg.out_dir / "state"
        self.assertTrue((shards / "batch_01.json.gz").is_file())
        self.assertTrue((shards / "batch_02.json.gz").is_file())
        reloaded = batch_draft.load_state(cfg)
        self.assertEqual(set(reloaded["batches"]), {"batch_01", "batch_02"})


class TestFinalPhaseFailureRetry(unittest.TestCase):
    """A stored final-phase error is a resume point, not a verdict.

    2026-09-23: the final phase failed twice mid-merge during a provider
    degradation and each failure was persisted as an error shard — the resume
    then refused to re-draft ("final phase previously failed", exit 1) with
    six batches of paid digest work frozen behind it. An operator had to move
    the shard aside by hand. Re-running the final phase on every resume costs
    nothing when it succeeds and rediscovers the failure when it persists.
    """

    def test_a_stored_final_error_is_retried_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            final_result = {
                "statement": "I certify the foregoing is true.",
                "grounding_markdown": "# Grounding\n",
                "facts_total": 1, "facts_pre_merge": 2, "review_issues": [],
            }
            with (
                mock.patch("app.config.load_settings", return_value=_gate_settings()),
                mock.patch("app.llm.probe_chat",
                           return_value=ChatProbe(200, "ok")),
                mock.patch("batch_draft.final_phase", return_value=final_result),
            ):
                code = batch_draft.main(_retry_argv(tmp, final_error="CircuitBreakerOpenError: breaker OPEN"))

            self.assertEqual(code, 0)
            statement = (tmp / "out" / "statement.md").read_text(encoding="utf-8")
            self.assertIn("I certify the foregoing is true.", statement)
            self.assertTrue((tmp / "out" / "grounding.md").is_file())
            # The stored error must not survive as state: a resume after this
            # run sees a clean success, not another retry.
            reloaded = batch_draft.load_state(_make_cfg(tmp, out_dir=tmp / "out"))
            self.assertNotIn("error", reloaded["final"])

    def test_a_clean_final_shard_is_not_redone(self) -> None:
        # The retry path must not throw away a finished draft: a state with a
        # real final result skips straight to writing the outputs.
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            final_result = {
                "statement": "already drafted", "grounding_markdown": "g",
                "facts_total": 1, "facts_pre_merge": 2, "review_issues": [],
            }
            with (
                mock.patch("app.config.load_settings", return_value=_gate_settings()),
                mock.patch("app.llm.probe_chat",
                           return_value=ChatProbe(200, "ok")),
                mock.patch("batch_draft.final_phase") as final_phase,
            ):
                code = batch_draft.main(_retry_argv(tmp, final_result=final_result))

            self.assertEqual(code, 0)
            final_phase.assert_not_called()
            self.assertIn("already drafted", (tmp / "out" / "statement.md").read_text(encoding="utf-8"))


class TestWaitForBreaker(unittest.TestCase):
    """The single calls after the merge must wait out an OPEN breaker.

    2026-09-23, attempt 3: the merge's keep-raw-facts fallback absorbed the
    degradation (that part worked), but _summarize then started with the
    breaker still OPEN from the absorbed failures and died instantly on the
    entry check — the merge had no checkpoint, so the whole final phase's work
    was thrown away. Waiting out the breaker's own recovery clock turns that
    into a pause.
    """

    def _watched_breaker(self, **kwargs) -> object:
        from app.circuit_breaker import CircuitBreaker

        # A short-recovery breaker distinct from the process-wide "llm" one,
        # handed to _wait_for_breaker via the get_llm_breaker patch below.
        defaults = dict(failure_threshold=2, recovery_timeout=0.2, name="test-wait")
        defaults.update(kwargs)
        return CircuitBreaker(**defaults)

    def test_returns_immediately_when_the_breaker_is_closed(self) -> None:
        breaker = self._watched_breaker()
        with mock.patch("app.circuit_breaker.get_llm_breaker", return_value=breaker):
            started = time.monotonic()
            batch_draft._wait_for_breaker("unit test")
        self.assertLess(time.monotonic() - started, 1.0)

    def test_waits_out_an_open_breaker_until_a_call_is_admitted(self) -> None:
        breaker = self._watched_breaker()
        # Force the breaker OPEN; allow_request flips it HALF_OPEN once the
        # (short) recovery timeout elapses, which is what unblocks the wait.
        for _ in range(breaker.failure_threshold):
            breaker.record_failure(reason="unit test", retriable=True)
        self.assertEqual(breaker.state, "OPEN")
        original_sleep = time.sleep
        sleeps: list[float] = []

        def fast_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            original_sleep(min(seconds, 0.01))

        started = time.monotonic()
        with (
            mock.patch("app.circuit_breaker.get_llm_breaker", return_value=breaker),
            mock.patch("time.sleep", side_effect=fast_sleep),
        ):
            batch_draft._wait_for_breaker("unit test")
        self.assertTrue(sleeps, "wait must sleep between polls")
        self.assertLess(time.monotonic() - started, 10.0)
        self.assertIn(breaker.state, {"HALF_OPEN", "CLOSED"})


class TestRetryPhase(unittest.TestCase):
    """The post-merge single calls must survive minutes-long provider bursts.

    2026-09-23, attempt 4: summarize succeeded, then the grounding call died
    to one burst — its 3-attempt ladder spans seconds, the burst spanned
    minutes, and there is no fallback, so the whole final phase's work was
    discarded again.
    """

    def _run_without_breaker_wait(self) -> object:
        # _wait_for_breaker polls the process-wide breaker with real sleeps;
        # these tests own the retry loop, not the wait, and another test
        # module's failures can leave that breaker OPEN.
        return mock.patch.object(batch_draft, "_wait_for_breaker")

    def test_retries_through_transient_failures_and_returns_the_result(self) -> None:
        attempts: list[str] = []

        def flaky() -> str:
            attempts.append("hit")
            if len(attempts) < 3:
                raise RuntimeError("The Responses run did not complete (status: incomplete)")
            return "draft text"

        with (
            self._run_without_breaker_wait(),
            mock.patch.object(batch_draft.time, "sleep"),
        ):
            result = batch_draft._retry_phase(flaky, "unit test")
        self.assertEqual(result, "draft text")
        self.assertEqual(len(attempts), 3)

    def test_raises_after_the_last_attempt(self) -> None:
        def always_fails() -> None:
            raise RuntimeError("burst")

        with (
            self._run_without_breaker_wait(),
            mock.patch.object(batch_draft.time, "sleep"),
        ):
            with self.assertRaises(RuntimeError):
                batch_draft._retry_phase(always_fails, "unit test", attempts=3)

    def test_an_auth_error_is_never_retried(self) -> None:
        from app.llm import LLMAuthError

        calls: list[int] = []

        def refused() -> None:
            calls.append(1)
            raise LLMAuthError("401: invalid key")

        with (
            self._run_without_breaker_wait(),
            mock.patch.object(batch_draft.time, "sleep"),
        ):
            with self.assertRaises(LLMAuthError):
                batch_draft._retry_phase(refused, "unit test")
        self.assertEqual(len(calls), 1, "a credential verdict never heals")


def _gate_settings() -> "config.Settings":
    """A real Settings for main()-level tests: once the credential gate passes,
    main() constructs LLMClient from it, and the SDK validates its arguments —
    a MagicMock here would die inside OpenAI(**settings)."""
    return config.Settings(
        api_key="k", base_url="https://api.perplexity.ai/v1",
        model_main="perplexity/kimi-k3", model_fast="perplexity/glm-5.3-flash",
        fetch_api_key="", fetch_base_url="", fetch_records_path="",
    )


def _main_argv(tmp: Path) -> list[str]:
    """A minimal main() argument vector over one staged part, shared by the
    credential-gate and fatal-auth tests; --no-final keeps main() out of the
    final phase, which these tests do not own."""
    records = tmp / "records"
    records.mkdir(parents=True, exist_ok=True)
    (records / "Part1.pdf").write_text("EVT", encoding="utf-8")
    obs = tmp / "observations.txt"
    obs.write_text("I have watched the veteran decline since 2022.", encoding="utf-8")
    return ["--records", str(records), "--glob", "Part*.pdf",
            "--out", str(tmp / "out"),
            "--condition", "PTSD", "--claim-type", "Initial claim",
            "--observations", str(obs), "--no-final"]


def _retry_argv(tmp: Path, final_error: str = "",
                final_result: dict | None = None) -> list[str]:
    """A resume-argv whose out-dir already holds digest shards plus a final
    shard — the shape the 2026-09-23 run was left in when the final phase
    died mid-merge: hours of digest work saved, draft absent. Unlike
    ``_main_argv`` there is no --no-final: these tests drive the final phase.
    """
    records = tmp / "records"
    records.mkdir(parents=True, exist_ok=True)
    (records / "Part1.pdf").write_text("EVT", encoding="utf-8")
    obs = tmp / "observations.txt"
    obs.write_text("I have watched the veteran decline since 2022.", encoding="utf-8")
    out = tmp / "out"
    cfg = _make_cfg(tmp, out_dir=out)
    state: dict = {"batches": {"batch_01": dict(batch_draft.EMPTY_DIGEST_STATE,
                                                facts=[{"description": "night terrors"}])},
                   "final": None}
    if final_error:
        state["final"] = {"error": final_error}
    elif final_result is not None:
        state["final"] = final_result
    batch_draft.save_state(cfg, state)
    return ["--records", str(records), "--glob", "Part*.pdf",
            "--out", str(out),
            "--condition", "PTSD", "--claim-type", "Initial claim",
            "--observations", str(obs)]


class TestStartupCredentialGate(unittest.TestCase):
    """A refused credential check must stop the run before any work begins.

    The 2026-09-22 dead-key incident spent 17 minutes discovering a 401 one
    chunk at a time; the gate makes that failure cost one probe (~600 ms when
    refused — a 401 spends nothing) and one readable line.
    """

    def test_a_refused_credential_check_stops_the_run_before_any_batch(self) -> None:
        from app.llm import ChatProbe

        cfg_calls: list[list[str]] = []

        def refused_probe(base_url: str, api_key: str, model: str) -> ChatProbe:
            cfg_calls.append([base_url, model])
            return ChatProbe(401, "Invalid API key provided.")

        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            with (
                mock.patch("app.config.load_settings", return_value=_gate_settings()),
                mock.patch("app.llm.probe_chat", side_effect=refused_probe),
                mock.patch("batch_draft.digest_group") as digest,
            ):
                code = batch_draft.main(_main_argv(tmp))

        self.assertEqual(code, 1)
        self.assertEqual(len(cfg_calls), 1, "exactly one credential probe")
        digest.assert_not_called()

    def test_an_ambiguous_probe_lets_the_run_proceed_to_batch_planning(self) -> None:
        # The gate blocks only on deterministic refusals; a probe that could not
        # get a response (status None) must not refuse a working run.
        from app.llm import ChatProbe

        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            digest_calls: list[str] = []

            def fake_digest(llm, cfg, label, files, depth=0):  # noqa: ANN001
                digest_calls.append(label)
                # Real signature: (state, excluded-files) — main unpacks it.
                return dict(batch_draft.EMPTY_DIGEST_STATE), []

            with (
                mock.patch("app.config.load_settings", return_value=_gate_settings()),
                mock.patch("app.llm.probe_chat",
                           return_value=ChatProbe(None, "URLError: name or service not known")),
                mock.patch("batch_draft.digest_group", side_effect=fake_digest),
            ):
                code = batch_draft.main(_main_argv(tmp))

        self.assertEqual(code, 0, "digest-only run completes")
        self.assertEqual(digest_calls, ["batch_01"])


class TestFatalAuthHaltsTheRun(unittest.TestCase):
    """A credential refusal mid-run stops the pipeline; it never bisects.

    The dead-key incident bisected five levels deep and quarantined nine
    healthy files because the bisect policy had no way to know an auth refusal
    is not a property of any file. LLMAuthError now escapes the bisect entirely.
    """

    def test_digest_group_stops_the_run_on_an_auth_error(self) -> None:
        """Through the REAL digest_group: the review layer refuses credentials,
        digest_group converts that into SystemExit(2) — never a bisect, never a
        quarantine. Exactly one review call proves the point: a bisect would
        invoke it again on each half."""
        from app.llm import LLMAuthError

        cfg = _make_cfg(Path(tempfile.mkdtemp()))
        for name in ("good_a.pdf", "good_b.pdf", "good_c.pdf"):
            (cfg.records_dir / name).write_text("EVT note.", encoding="utf-8")

        review_calls = 0

        def refused(llm, docs, progress=None):  # noqa: ANN001
            nonlocal review_calls
            review_calls += 1
            raise LLMAuthError("Credentials refused by the LLM endpoint (HTTP 401)",
                               retriable=False, status_code=401)

        with (
            mock.patch("app.documents.records_from_local_path",
                       side_effect=lambda p: ([], [])),
            mock.patch("app.medical_review.review_medical_records", side_effect=refused),
        ):
            with self.assertRaises(SystemExit) as ctx:
                batch_draft.digest_group(
                    object(), cfg, "batch_01",
                    sorted(cfg.records_dir.glob("*.pdf")),
                )

        self.assertEqual(ctx.exception.code, 2)
        self.assertEqual(review_calls, 1, "one refusal, zero bisects")

    def test_main_exits_2_when_a_batch_hits_the_auth_wall(self) -> None:
        # Through the REAL digest_group (a mock would bypass the very
        # except-LLMAuthError branch under test): the review layer refuses
        # credentials, digest_group converts that into SystemExit(2), and
        # main() — whose per-batch handler catches only Exception — lets it
        # escape. In production `raise SystemExit(main())` turns that into
        # exit code 2; here we assert on the exception itself.
        from app.llm import ChatProbe, LLMAuthError

        def refused_review(llm, docs, progress=None):  # noqa: ANN001
            raise LLMAuthError("Credentials refused by the LLM endpoint (HTTP 401)",
                               retriable=False, status_code=401)

        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            records = tmp / "records"
            records.mkdir(parents=True, exist_ok=True)
            (records / "Part1.pdf").write_text("EVT", encoding="utf-8")
            with (
                mock.patch("app.config.load_settings", return_value=_gate_settings()),
                mock.patch("app.llm.probe_chat", return_value=ChatProbe(200, "", "OK")),
                mock.patch("app.documents.records_from_local_path",
                           side_effect=lambda p: ([], [])),
                mock.patch("app.medical_review.review_medical_records",
                           side_effect=refused_review),
            ):
                with self.assertRaises(SystemExit) as ctx:
                    batch_draft.main(_main_argv(tmp))

        self.assertEqual(ctx.exception.code, 2,
                         "SystemExit(2) — a stop-the-run signal, not a batch failure")


class TestMergeRoundCheckpointing(unittest.TestCase):
    """Merge-round checkpointing: a relaunch resumes the consolidation.

    The final phase's merge is the most expensive single stage (an hour of
    rounds at production scale), and every relaunch before checkpointing
    redid all of it. These tests pin the contract from both sides: the shard
    round-trips under its input fingerprint, a mismatched or corrupt shard
    reads as absent, a resume actually short-circuits round 1, and --fresh
    clears the checkpoint along with the rest of the state.
    """

    def _final_with_real_merge(self, cfg):
        """Drive final_phase with the real _merge_facts (shrinking fake LLM).

        Returns (result, llm, cfg, fingerprint) so tests can assert on the
        checkpoint the merge itself wrote. 200 distinct facts in, and every
        merge call keeps 3/4 of its batch: round 1 ends at 150 facts (> the
        120 single-call limit, < 200 — the merge continues, so the round IS
        checkpointed), round 2 ends at 112 (<= 120 — terminal, so it is not:
        a terminal list must never be resumed from, see the resume guard).
        """
        from app.medical_review import MedicalDigest, MedicalFact, _dedupe_facts

        batch_state = batch_draft.digest_to_state(MedicalDigest(
            facts=[MedicalFact(date="2023-01-01", type="symptom",
                               description=f"fact number {i}", source="p. 1")
                    for i in range(200)],
            pages_in_files=10, pages_reviewed=10,
        ))

        class FactPassingLLM:
            """Merge calls pass facts through unchanged, like test_core's fake."""

            fast_model = "fake-fast"

            def __init__(self):
                self.merge_calls = 0
                self.chat_calls = 0

            def chat(self, system, user, **kw):
                self.chat_calls += 1
                return ""

            def chat_json(self, system, user, **kw):
                if kw.get("phase") == "records:merge":
                    self.merge_calls += 1
                    facts = json.loads(user.split("\n\n", 1)[1])
                    return {"facts": facts[: max(1, (len(facts) * 3) // 4)]}
                return {"supported_observations": []}  # grounding / review

        # The exact fingerprint final_phase will derive: dedupe of all batch facts.
        all_facts = [MedicalFact(**f) for f in batch_state["facts"]]
        fingerprint = batch_draft._merge_fingerprint(_dedupe_facts(all_facts))
        llm = FactPassingLLM()
        with mock.patch("app.pipeline_guard.run_with_timeout",
                        side_effect=lambda fn, *a, **k: fn()), \
             mock.patch("app.medical_review._summarize", return_value="Summary."), \
             mock.patch.object(batch_draft.time, "sleep"), \
             mock.patch.object(batch_draft, "_wait_for_breaker"):
            result = batch_draft.final_phase(llm, cfg, {"batch_01": batch_state})
        return result, llm, cfg, fingerprint

    def test_round_checkpoint_round_trips_and_fingerprint_gates(self) -> None:
        from app.medical_review import MedicalFact

        facts = [
            MedicalFact(date="2023-01-01", type="symptom",
                        description="Nightmares most nights", source="p. 1"),
            MedicalFact(date="2023-02-01", type="medication",
                        description="Sertraline increased", source="p. 2",
                        document="part03.pdf", page=12),
            MedicalFact(date="2023-03-01", type="visit",
                        description="Missed appointment", source="p. 3"),
        ]
        cfg = _make_cfg(Path(tempfile.mkdtemp()))
        fingerprint = batch_draft._merge_fingerprint(facts)
        batch_draft.save_merge_checkpoint(cfg, fingerprint, 1, facts)

        loaded = batch_draft.load_merge_checkpoint(cfg, fingerprint)
        self.assertEqual(loaded, facts)  # dataclass equality: every field survives

        # A different input's fingerprint must not open this checkpoint: the
        # resume must never continue a consolidation whose input was different.
        other = [MedicalFact(date="2023-01-01", type="symptom",
                             description="different fact entirely", source="p. 1")]
        self.assertIsNone(
            batch_draft.load_merge_checkpoint(cfg, batch_draft._merge_fingerprint(other)))

    def test_load_missing_or_corrupt_checkpoint_is_none(self) -> None:
        cfg = _make_cfg(Path(tempfile.mkdtemp()))
        self.assertIsNone(batch_draft.load_merge_checkpoint(cfg, "whatever"))
        path = batch_draft._merge_ckpt_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not gzip at all")
        self.assertIsNone(batch_draft.load_merge_checkpoint(cfg, "whatever"))

    def test_final_phase_resumes_from_checkpoint_and_reuses_it(self) -> None:
        """A stored round-1 list short-circuits round 1 — including on a
        relaunch where nothing but the shard on disk carries the state."""
        from app.medical_review import MedicalFact

        result, llm, cfg, fingerprint = self._final_with_real_merge(
            _make_cfg(Path(tempfile.mkdtemp())))
        # Round 1: 5 batches of 48 -> 36 each = 150 (checkpointed, the merge
        # continues). Round 2: 4 batches -> 112 <= 120 -> terminal, no rewrite.
        self.assertEqual(llm.merge_calls, 9)
        self.assertEqual(result["facts_total"], 112)
        stored = batch_draft.load_merge_checkpoint(cfg, fingerprint)
        self.assertIsNotNone(stored)
        self.assertEqual(len(stored), 150)

        # Now the resume: pre-store a smaller round-1 output for the SAME
        # input fingerprint and re-run the final phase from scratch (new LLM,
        # nothing in memory — exactly a relaunch reading the same out-dir).
        resumed = [MedicalFact(date="2023-01-01", type="symptom",
                               description=f"resumed fact {i}", source="p. 1")
                   for i in range(150)]
        batch_draft.save_merge_checkpoint(cfg, fingerprint, 1, resumed)
        result2, llm2, cfg2, fp2 = self._final_with_real_merge(cfg)
        self.assertEqual(fp2, fingerprint)
        # 150 stored facts -> 4 batches; round 1 skipped entirely...
        self.assertEqual(llm2.merge_calls, 4)
        # ...and this run's round 1 is terminal (112 <= 120), so the stored
        # checkpoint was consumed, not rewritten.
        self.assertEqual(result2["facts_total"], 112)
        self.assertEqual(batch_draft.load_merge_checkpoint(cfg2, fp2), resumed)

    def test_fresh_clears_merge_checkpoint(self) -> None:
        from app.medical_review import MedicalDigest, MedicalFact

        cfg = _make_cfg(Path(tempfile.mkdtemp()))
        path = batch_draft._merge_ckpt_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stale checkpoint bytes")
        argv = _main_argv(cfg.records_dir.parent)
        argv[argv.index("--out") + 1] = str(cfg.out_dir)
        records = cfg.records_dir
        (records / "Part1.pdf").write_text("EVT", encoding="utf-8")
        with mock.patch("app.config.load_settings", return_value=_gate_settings()), \
             mock.patch("app.llm.probe_chat", return_value=ChatProbe(200, "", "OK")), \
             mock.patch("app.documents.records_from_local_path",
                        side_effect=lambda p: ([], [])), \
             mock.patch("app.medical_review.review_medical_records",
                        return_value=MedicalDigest(facts=[
                            MedicalFact(date="2023-01-01", type="symptom",
                                        description="Nightmares most nights",
                                        source="p. 1")],
                                pages_in_files=10, pages_reviewed=10)):
            batch_draft.main(argv + ["--fresh"])
        self.assertFalse(path.exists(), "--fresh must clear the merge checkpoint")


if __name__ == "__main__":
    unittest.main()
