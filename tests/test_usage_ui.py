"""UI-level tests for the usage/credit estimator.

Drives the real app with Streamlit's AppTest, stubbing the LLM so no network or
key is needed, and asserts that a run surfaces (a) a live caption line and
(b) the per-phase "Estimated API usage" expander.
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from app.usage import UsageTracker  # noqa: E402
from app import watchdog  # noqa: E402


class _FakeSettings:
    model_main = "fake-main"
    model_fast = "fake-fast"


class _FakeLLM:
    """Stub LLMClient that returns minimal valid pipeline results."""

    def __init__(self) -> None:
        self._settings = _FakeSettings()
        self.usage = UsageTracker()

    def _record(self, model, phase, system, user, content):
        self.usage.record(
            model=model, phase=phase, system=system, user=user,
            content=content, prompt_tokens=None, completion_tokens=None,
        )

    def chat_json(self, system, user, **kwargs):
        content = '{"facts": []}'
        self._record(kwargs.get("model") or self._settings.model_main,
                     kwargs.get("phase", "general"), system, user, content)
        # Route by phase so each stage returns something the pipeline accepts.
        phase = kwargs.get("phase", "general")
        if phase == "records:merge":
            return {"facts": []}
        if phase == "claims":
            return {"claimed_condition": "", "writer_role": "", "claims": []}
        if phase == "rubric":
            return {
                "scores": {}, "rationales": {}, "improvements": [],
                "omitted_record_facts": [], "executive_summary": "",
            }
        if phase == "topic":
            return {"claim_focus": "", "topics": [], "critical_gaps": [], "notes": ""}
        if phase == "revision":
            return {
                "revision_notes": "", "changes": [], "revised_statement": "",
                "added_facts_to_verify": [],
            }
        if phase == "review":
            return {"issues_found": [], "improved_statement": ""}
        return {"facts": []}

    def chat(self, system, user, **kwargs):
        content = "Summary of records."
        self._record(kwargs.get("model") or self._settings.model_main,
                     kwargs.get("phase", "general"), system, user, content)
        return content


@unittest.skipUnless((PROJECT_ROOT / ".venv").exists(), "requires the project venv")
class TestUsageUi(unittest.TestCase):
    def setUp(self) -> None:
        # Route the watchdog's persisted history to a temp file so test runs
        # don't pollute (or depend on) the real usage_history.json.
        import tempfile

        self._tmp_hist = tempfile.mkdtemp()
        self._old_env = os.environ.get("VA_LSE_WATCHDOG_PATH")
        os.environ["VA_LSE_WATCHDOG_PATH"] = str(Path(self._tmp_hist) / "hist.json")

    def tearDown(self) -> None:
        if self._old_env is None:
            os.environ.pop("VA_LSE_WATCHDOG_PATH", None)
        else:
            os.environ["VA_LSE_WATCHDOG_PATH"] = self._old_env

    def _app(self):
        from streamlit.testing.v1 import AppTest

        return AppTest.from_file(str(PROJECT_ROOT / "run_app.py"), default_timeout=30)

    def test_live_caption_and_usage_expander_render_after_run(self):
        import app.main as main

        fake = _FakeLLM()
        main._get_llm = lambda: fake
        at = self._app()
        at.run()

        # Step 1: paste a statement.
        at.radio(key="eval_mode").set_value("Paste text")
        at.run()
        at.text_area(key="eval_paste").set_value("I watched the veteran limp after duty.")
        at.run()

        # Step 2: upload one record file.
        at.file_uploader(key="files_eval").set_value(
            [("good.txt", b"Knee pain noted during visit.", "text/plain")]
        )
        at.run()

        # Run the evaluation.
        at.button(key="eval_run").click().run()

        # The per-phase usage expander should have rendered a totals caption.
        # (The live progress-caption line is transient — cleared by bar.empty()
        # at the end of the run — so we assert on the persistent expander.)
        expanders = at.expander
        self.assertTrue(
            any("Estimated API usage" in e.label for e in expanders),
            msg=f"usage expander not found: {[e.label for e in expanders]}",
        )
        inner = [c.value for e in expanders for c in e.caption]
        inner_text = " ".join(inner)
        self.assertIn("Total:", inner_text)
        self.assertIn("7 call(s)", inner_text)

        # Contents of the fake tracker should include the pipeline phases.
        phases = set(fake.usage.per_phase().keys())
        self.assertIn("records:digest", phases)
        self.assertIn("records:summary", phases)
        self.assertIn("claims", phases)
        self.assertGreater(fake.usage.totals().calls, 0)


    def test_explicit_env_rate_not_overwritten_by_watchdog(self):
        """A .env rate set for ONE model must not be clobbered by the watchdog
        blended rate; the watchdog only fills the missing model's slot."""
        import app.config as config
        import unittest.mock as mock

        # Seed watchdog history with a learned blended rate (e.g. 100 credits/1M).
        history = watchdog.UsageHistory()
        watchdog.record_run(history, prompt_tokens=1000, completion_tokens=0, calls=1)
        watchdog.record_calibration(history, credits=0.0, ts=1.0)
        watchdog.record_run(history, prompt_tokens=1000, completion_tokens=0, calls=1)
        watchdog.record_calibration(history, credits=0.1, ts=2.0)  # 100 credits/1M
        watchdog.save_history(history, str(Path(self._tmp_hist) / "hist.json"))

        # User explicitly set only the MAIN rate in .env; watchdog knows a rate.
        with mock.patch.object(config, "CREDITS_PER_1M_MAIN", 800.0), mock.patch.object(
            config, "CREDITS_PER_1M_FAST", None
        ):
            import app.main as main

            rates, label = main._effective_credit_rates()

        # MAIN keeps its explicit 800; only FAST borrows from the watchdog (100).
        self.assertEqual(rates[config.DEFAULT_MODEL_MAIN], 800.0)
        self.assertAlmostEqual(rates[config.DEFAULT_MODEL_FAST], 100.0, delta=1e-6)
        self.assertIn("watchdog", label)


if __name__ == "__main__":
    unittest.main()