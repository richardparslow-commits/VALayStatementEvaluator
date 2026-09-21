"""UI-level tests for the usage/credit estimator.

Drives the real app with Streamlit's AppTest, stubbing the LLM so no network or
key is needed, and asserts that a run surfaces (a) a live caption line and
(b) the per-phase "Estimated API usage" expander.

These run wherever the runtime dependencies are installed — including CI, which
pip-installs requirements.lock before `unittest discover`. There is deliberately
no skip gate here: a `PROJECT_ROOT/.venv` check used to sit on the helper class
below (where it did nothing at all), and gating AppTest tests on a venv path
would silently drop this coverage in CI. If Streamlit's AppTest harness is
unavailable the test must error loudly, not skip.
"""
import sys
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/ — log_isolation
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from app.usage import UsageTracker  # noqa: E402
from app import watchdog  # noqa: E402
from log_isolation import isolate_app_logs  # noqa: E402


class _FakeSettings:
    model_main = "fake-main"
    model_fast = "fake-fast"


class _FakeLLM:
    """Stub LLMClient that returns minimal valid pipeline results."""

    fast_model = "fake-fast"

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


class _FailoverLLM(_FakeLLM):
    """Same stub, recording every call against the backup endpoint."""

    def _record(self, model, phase, system, user, content):
        self.usage.record(
            model=model, phase=phase, system=system, user=user,
            content=content, prompt_tokens=None, completion_tokens=None,
            endpoint="fallback",
        )


class TestUsageUi(unittest.TestCase):
    def setUp(self) -> None:
        # Keep run-log, audit-log, and watchdog side effects out of the
        # developer's real logs/ and usage history: these AppTest runs would
        # otherwise be indistinguishable from real use (tests/log_isolation.py).
        self._tmpdir = isolate_app_logs(self)
        self._tmp_hist = self._tmpdir

    def _app(self):
        from streamlit.testing.v1 import AppTest

        return AppTest.from_file(str(PROJECT_ROOT / "run_app.py"), default_timeout=30)

    def test_live_caption_and_usage_expander_render_after_run(self):
        import app.views.evaluate_view as evaluate_view

        fake = _FakeLLM()
        evaluate_view.get_llm = lambda: fake  # type: ignore[assignment]
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
        # 7 calls for the pipeline phases (record digest + summary, claims,
        # verify, rubric, topic, revision) plus 1 for the effectiveness-score
        # recommendations pass. Pin the total so an accidental extra call still
        # fails here, and read the caption back rather than trusting it.
        calls = fake.usage.totals().calls
        self.assertEqual(
            calls,
            8,
            "pipeline LLM call count changed — update this expectation deliberately",
        )
        self.assertIn(f"{calls} call(s)", inner_text)

        # Contents of the fake tracker should include the pipeline phases.
        phases = set(fake.usage.per_phase().keys())
        self.assertIn("records:digest", phases)
        self.assertIn("records:summary", phases)
        self.assertIn("claims", phases)
        self.assertGreater(fake.usage.totals().calls, 0)


    def _run_evaluation(self, fake):
        """Drive one end-to-end Evaluate run in the real app with a stubbed LLM."""
        import app.views.evaluate_view as evaluate_view

        evaluate_view.get_llm = lambda: fake  # type: ignore[assignment]
        at = self._app()
        at.run()
        at.radio(key="eval_mode").set_value("Paste text")
        at.run()
        at.text_area(key="eval_paste").set_value("I watched the veteran limp after duty.")
        at.run()
        at.file_uploader(key="files_eval").set_value(
            [("good.txt", b"Knee pain noted during visit.", "text/plain")]
        )
        at.run()
        at.button(key="eval_run").click().run()
        return at

    def test_a_failed_over_run_tells_the_user_it_used_the_backup(self):
        """A different model wrote this document — the reader must not have to
        open a details panel, or read the audit log, to find that out."""
        at = self._run_evaluation(_FailoverLLM())
        messages = [w.value for w in at.warning]
        self.assertTrue(
            any("backup LLM endpoint" in m for m in messages),
            msg=f"no failover warning rendered: {messages}",
        )

    def test_a_normal_run_does_not_warn_about_failover(self):
        """Silence in the ordinary case is what makes the warning meaningful."""
        at = self._run_evaluation(_FakeLLM())
        messages = [w.value for w in at.warning]
        self.assertFalse(any("backup LLM endpoint" in m for m in messages), msg=str(messages))

    def test_watchdog_widget_renders_with_a_fit_and_no_env_rates(self):
        """Regression: once a watchdog fit exists the sidebar must render even
        with no `.env` rates set.

        The guard this replaced used `all(<bool>)` and raised TypeError on
        exactly this branch, which killed the whole app render. The assertion
        was lost when the AppTest suite was reorganised; it is restored here
        because nothing else exercises the widget's fit branch end to end (the
        unit tests below only cover `effective_credit_rates()` in isolation).
        """
        import app.config as config
        import unittest.mock as mock

        history = watchdog.UsageHistory()
        watchdog.record_run(history, prompt_tokens=1000, completion_tokens=0, calls=1)
        watchdog.record_calibration(history, credits=0.0, ts=1.0)
        watchdog.record_run(history, prompt_tokens=1000, completion_tokens=0, calls=1)
        watchdog.record_calibration(history, credits=0.1, ts=2.0)  # 100 credits/1M
        # Save where the app will read it: isolate_app_logs() points
        # VA_LSE_WATCHDOG_PATH at this test's temp dir.
        watchdog.save_history(history, watchdog.history_path())

        with mock.patch.object(config, "CREDITS_PER_1M_MAIN", None), mock.patch.object(
            config, "CREDITS_PER_1M_FAST", None
        ):
            at = self._app()
            at.run()

        self.assertEqual(
            list(at.exception), [], msg=f"app raised during render: {list(at.exception)}"
        )
        # The fit branch rendered the fallback guidance — the branch that used
        # to raise. Markdown lives inside the sidebar expander, so search both.
        rendered = [m.value for m in at.markdown] + [
            m.value for expander in at.expander for m in expander.markdown
        ]
        self.assertTrue(
            any("The estimator will now use" in value for value in rendered),
            msg=f"fallback guidance not found: {rendered}",
        )

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
        watchdog.save_history(history, str(Path(self._tmp_hist) / "usage_history.json"))

        # User explicitly set only the MAIN rate in .env; watchdog knows a rate.
        with mock.patch.object(config, "CREDITS_PER_1M_MAIN", 800.0), mock.patch.object(
            config, "CREDITS_PER_1M_FAST", None
        ):
            import app.views.shared as shared

            rates, label = shared.effective_credit_rates()

        # MAIN keeps its explicit 800; only FAST borrows from the watchdog (100).
        self.assertEqual(rates[config.DEFAULT_MODEL_MAIN], 800.0)
        self.assertAlmostEqual(rates[config.DEFAULT_MODEL_FAST], 100.0, delta=1e-6)
        self.assertIn("watchdog", label)


if __name__ == "__main__":
    unittest.main()