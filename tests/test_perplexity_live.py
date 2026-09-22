"""Live check of the Perplexity Agent API path — opt-in, skipped without a key.

`tests/test_perplexity_agent.py` drives the **real** ``perplexityai`` SDK with its
transport replaced by an in-process ``httpx.MockTransport``, so it pins this
repository's request shape and its extraction of an answer, its citations and its
structured output. What no fake can settle is the other side of the wire: whether a real
``pplx-`` key is accepted by ``https://api.perplexity.ai/v1``, whether the Agent API
actually runs the ``web_search`` tool for this request shape, whether a live response
still carries ``search_results`` in a shape ``_collect_citations`` reads, and whether the
configured preset resolves to a model the account may call. Those are the failures that
reach a user as "research is broken", so this runs the app's own ``research()`` — the
same call the Research tab makes — against the real API:

    VA_LSE_TEST_PERPLEXITY_KEY=pplx-... python -m unittest tests.test_perplexity_live

The key arrives under the ``VA_LSE_TEST_*`` prefix (the hermetic session's one sanctioned
way for a runner to opt a test into real setup) and is injected as the app's own
``PERPLEXITY_API_KEY``. It is never printed, logged, or put in a failure message — the
assertions below name response *fields*, not the credential. The module skips without the
variable, so local runs and ordinary CI are unaffected; the dispatch-only
``perplexity-live`` job in ``.github/workflows/test.yml`` supplies it.

What it establishes, at the cost of a few search-tool calls:

* a grounded answer comes back with prose **and at least one citation** — zero sources
  from a web-grounded call means the tool did not run, and an unsourced answer still
  reads as confident, which is the one failure this feature exists to prevent;
* a structured call (``condition_audit_schema``) returns parseable ``findings``;
* the framework-currency check the Evaluate tab reads produces a report with a verdict
  for every topic it was asked about, keyed to the committed checklist.

What it deliberately does *not* assert: the **content** of an answer, because the web
moves and a disagreement with this file would be a finding about the world rather than
about the integration. Nor does it assert entitlement — a key with no Agent API access,
or a spend/rate limit, is that account's configuration rather than a compatibility
failure, so those outcomes skip with the provider's own message instead of failing the
job.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app.circuit_breaker import reset_llm_breaker  # noqa: E402
from app.config import (  # noqa: E402
    PERPLEXITY_PRESETS,
    load_knowledge,
    load_settings,
)
from app.knowledge_currency import (  # noqa: E402
    STATUS_CHANGED,
    STATUS_CURRENT,
    STATUS_UNCONFIRMED,
    forget_report,
    normalize_topics,
    parse_topic_sections,
    verify_framework_currency,
)
from app.perplexity_agent import (  # noqa: E402
    BREAKER_NAME,
    GroundedAnswer,
    PerplexityConfigurationError,
    PerplexityUpstreamError,
    condition_audit_schema,
    configured_preset,
    research,
    sdk_installed,
)

KEY_ENV = "VA_LSE_TEST_PERPLEXITY_KEY"
PRESET_ENV = "VA_LSE_TEST_PERPLEXITY_PRESET"

#: The default the panel ships with, used when the runner sets no preset. It is the
#: cheapest one that still grounds, which is what a live check should spend.
DEFAULT_PRESET = "low"

#: A question whose *subject* is stable law, so the assertion is that a grounded answer
#: arrives at all — not that the web agrees with anything in this file.
QUESTION = (
    "Under 38 CFR 3.303, what must be shown to service-connect a chronic disease, "
    "and how does the presumption for a chronic disease during service work?"
)

#: How the API reports a credential the account cannot use, or a spend/rate limit. Both
#: are the account's configuration rather than a compatibility failure of this app.
_UNUSABLE_STATUSES = frozenset({401, 403, 429})
_UNUSABLE_MARKERS = (
    "unauthorized",
    "forbidden",
    "invalid api key",
    "rate limit",
    "quota",
    "insufficient",
    "no access",
    "not entitled",
    "billing",
)


class LivePerplexityTestCase(unittest.TestCase):
    """Real key, real Agent API, no fakes — and no credential in any output."""

    def setUp(self) -> None:
        self.key = (os.getenv(KEY_ENV) or "").strip()
        if not self.key:
            self.skipTest(f"set {KEY_ENV} to check the real Perplexity Agent API")
        # The SDK is in the core set, so CI has it; a venv without it is an install gap
        # (`pip install -r requirements-perplexity.txt`) rather than a defect.
        if not sdk_installed():
            self.skipTest(
                "the perplexity SDK is not installed: "
                "pip install -r requirements-perplexity.txt"
            )
        preset = (os.getenv(PRESET_ENV) or "").strip() or DEFAULT_PRESET
        with patch.dict(
            os.environ,
            {"PERPLEXITY_API_KEY": self.key, "PERPLEXITY_PRESET": preset},
            clear=False,
        ):
            self.settings = load_settings()
        # A live call must not inherit a breaker another test opened — an open breaker
        # would present as a refused research call, which is not what is under test.
        reset_llm_breaker(name=BREAKER_NAME)
        self.addCleanup(reset_llm_breaker, name=BREAKER_NAME)

    def _unusable_skip(self, exc: BaseException) -> None:
        """Skip when the account may not make this call; re-raise anything else.

        A transport failure, an unparseable answer, a 5xx or a refusal the provider
        words differently is a real incompatibility and must fail.
        """
        status = getattr(exc, "status_code", None)
        message = str(exc)
        lowered = message.lower()
        if status in _UNUSABLE_STATUSES or any(m in lowered for m in _UNUSABLE_MARKERS):
            self.skipTest(
                f"this Perplexity key/account cannot run research right now: {message[:200]}"
            )
        raise exc

    def _research(self, question: str, **kwargs: Any) -> GroundedAnswer:
        """The app's own research call, with configuration errors surfaced as failures.

        ``setUp`` already proved a key is present and the SDK is installed, so a
        ``PerplexityConfigurationError`` here is a finding about the settings this test
        built — never something to skip past.
        """
        try:
            return research(question, settings=self.settings, **kwargs)
        except PerplexityConfigurationError as exc:
            self.fail(f"research reported a configuration problem despite {KEY_ENV}: {exc}")
        except PerplexityUpstreamError as exc:
            self._unusable_skip(exc)
            raise  # unreachable: skipTest always raises


class TestTheAppsResearchPath(LivePerplexityTestCase):
    def test_a_grounded_answer_arrives_with_sources(self) -> None:
        answer = self._research(QUESTION)

        self.assertTrue(answer.text.strip(), "the Agent API returned no answer text")
        self.assertEqual(answer.preset, configured_preset(self.settings))
        self.assertIn(answer.preset, PERPLEXITY_PRESETS)
        self.assertTrue(answer.response_id, "no response id, so a run cannot be traced")
        # Web grounding is a *tool invocation*: an answer with no sources means the tool
        # never ran, and the answer still reads as confident without one.
        self.assertGreater(
            answer.citation_count, 0, "a grounded call came back with no citations"
        )
        for citation in answer.citations:
            self.assertTrue(citation.url.startswith("http"), citation.url)
            self.assertTrue(citation.label(), "a citation with neither title nor url")

    def test_a_structured_call_returns_parseable_findings(self) -> None:
        answer = self._research(
            "Audit what the record must show for a chronic disease under 38 CFR 3.303.",
            instructions="Answer only with JSON matching the supplied schema.",
            schema=condition_audit_schema(),
        )

        self.assertIsNotNone(answer.findings, "response_format did not produce JSON")
        self.assertIsInstance(answer.findings, dict)
        assert answer.findings is not None  # narrows for the assertions below
        self.assertIn("condition", answer.findings)
        self.assertIn("summary", answer.findings)


class TestTheFrameworkCurrencyCheck(LivePerplexityTestCase):
    """The Evaluate tab reads this report, so the call that produces it is checked live.

    One paid call, one topic, and the stored report is removed afterwards so a live run
    cannot leave a verdict in the shared cache that a later run would read as its own.
    """

    def test_one_topic_gets_a_verdict_keyed_to_the_checklist(self) -> None:
        sections = parse_topic_sections(load_knowledge("topic_checklist.md"))
        letters = normalize_topics([section.letter for section in sections])
        self.assertTrue(letters, "the committed checklist named no known topic")
        letter = letters[0]

        forget_report()
        self.addCleanup(forget_report)
        try:
            report = verify_framework_currency(settings=self.settings, topics=[letter])
        except PerplexityConfigurationError as exc:
            self.fail(f"the currency check was not usable despite {KEY_ENV}: {exc}")
        except PerplexityUpstreamError as exc:
            self._unusable_skip(exc)
            raise  # unreachable: skipTest always raises

        # A verdict for the topic it was asked about, from the committed framework —
        # an empty list here is the silent "unverified" the report exists to avoid.
        verdicts = report.for_topics([letter])
        self.assertEqual([verdict.topic for verdict in verdicts], [letter])
        self.assertIn(
            verdicts[0].status,
            (STATUS_CURRENT, STATUS_CHANGED, STATUS_UNCONFIRMED),
            "a verdict with a status the report cannot classify reads as unverified",
        )
        self.assertTrue(report.fingerprint, "the report is not keyed to the framework text")
        self.assertIn(report.preset, PERPLEXITY_PRESETS)
        self.assertTrue(report.checked_at, "the report has no timestamp to age against")


if __name__ == "__main__":
    unittest.main()
