"""Turn a case's medical digest into the questions worth researching about it.

The Research tab shipped with a blank text box, which quietly made "know what to ask" the
user's job. But the two things worth asking about are already sitting in the digest the
app just built: **the conditions it extracted**, and **the statement elements the records
did not support**. Both are facts about *this* case, so both can be turned into questions
mechanically — and a gap question ("what evidence establishes a nexus for this
condition?") is the one a preparer is least likely to think to ask by hand, because the
gap is precisely the thing that isn't in front of them.

Kept out of the view layer on purpose. This is deterministic logic over a
``MedicalDigest``: no Streamlit, no network, no model call, so it is fully unit-testable
(``tests/test_research_questions.py``) and reusable by a future batch run or export
without touching rendering. The view only renders what :func:`derive_questions` returns.

Question derivation is offline and free; *running* a question is not (the Agent API bills
per tool call — see ``app/perplexity_agent.py``). That split is why the UI loads a
question into the form for review rather than firing it on click.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from .logging_config import get_logger
from .medical_review import STATEMENT_ELEMENTS, MedicalDigest
from .prompt_sanitize import GUARD_NOTE, sanitize_for_prompt

logger = get_logger("app.research_questions")

# --------------------------------------------------------------- session state lookups

# Where a completed review leaves its digest, most-recent-meaningful first.
# ``timeline_digest`` is the slot ``app/views/draft_view.py`` already publishes for
# cross-tab reuse, so honouring it first keeps one digest authoritative instead of
# inventing a second convention.
DIGEST_SESSION_SOURCES: tuple[tuple[str, str], ...] = (
    ("timeline_digest", "your last record review"),
    ("eval_result", "your last statement evaluation"),
    ("draft_result", "your last draft run"),
)


class SessionLookup(Protocol):
    """The slice of ``st.session_state`` used here: ``state[key]``.

    A Protocol rather than ``Mapping[str, Any]`` because ``SessionStateProxy`` is a
    ``MutableMapping`` keyed by ``str | int``, and mutable mappings are invariant in their
    key type — so a parameter typed ``Mapping[str, Any]`` rejects the real proxy. This
    narrower protocol is satisfied by both the proxy and a plain ``dict[str, ...]``, which
    is what the tests pass.

    Subscript rather than ``.get`` for the same reason: the proxy inherits its ``.get``
    from ``Mapping``, and mypy's protocol check against that inherited overload set fails
    where ``__getitem__`` does not.
    """

    def __getitem__(self, key: str, /) -> Any: ...


# ------------------------------------------------------------------------ derivation

# How many questions of each kind, and in total. A research call is a paid, seconds-long
# request, so a wall of twenty questions is not a feature — the caps keep the list to
# what a person will actually read and choose between.
MAX_QUESTIONS = 12
MAX_CONDITION_QUESTIONS = 6
MAX_EVIDENCE_QUESTIONS = 4

KIND_CONDITION = "condition"
KIND_EVIDENCE = "evidence"
KIND_OPINION = "opinion"

# Human labels for the statement elements, so a rationale reads as a sentence rather
# than as a dict key. Keys mirror ``medical_review.STATEMENT_ELEMENTS``.
ELEMENT_LABELS: dict[str, str] = {
    "in_service_event": "in-service event",
    "current_diagnosis": "current diagnosis",
    "nexus": "nexus (medical opinion linking the condition to service)",
    "functional_impact": "functional impact",
    "severity_frequency": "severity and frequency of symptoms",
    "treatment_history": "treatment history",
    "buddy_observable": "observations a lay witness can make",
    "other": "other evidence",
}

# Gap questions, in the order a rating decision reads them: whether the condition is
# linked to service at all comes before how it is rated, so nexus and the in-service
# event rank above the criteria for a percentage.
_GAP_QUESTIONS: dict[str, str] = {
    "nexus": (
        "What evidence does VA require to establish a nexus between {condition} and "
        "service, and what makes a medical opinion adequate for that purpose?"
    ),
    "in_service_event": (
        "What counts as an in-service event, injury, or aggravation for {condition} "
        "when the service treatment records are incomplete?"
    ),
    "current_diagnosis": (
        "What are VA's current diagnostic criteria and rating schedule for {condition}?"
    ),
    "buddy_observable": (
        "Which observations can a lay witness competently describe for {condition} "
        "without offering a diagnosis or an opinion on causation?"
    ),
    "functional_impact": (
        "How does VA evaluate functional impact for {condition}, and what do raters "
        "count as the functional limitation?"
    ),
    "severity_frequency": (
        "How do frequency and severity of symptoms affect the rating for {condition}?"
    ),
    "treatment_history": (
        "What does VA expect to see in the treatment record for {condition}?"
    ),
}
_GAP_ORDER: tuple[str, ...] = (
    "nexus",
    "in_service_event",
    "current_diagnosis",
    "buddy_observable",
    "functional_impact",
    "severity_frequency",
    "treatment_history",
)

# Bounds on the context block that accompanies a question. Small on purpose: it exists to
# make an answer specific, not to ship the record set to the provider (see
# ``case_context_block``).
MAX_CONTEXT_CONDITIONS = 12
CASE_CONTEXT_CHAR_BUDGET = 1_200

_WORD_RE = re.compile(r"[a-z0-9]+")
_MIN_CONDITION_CHARS = 3


@dataclass(frozen=True)
class ResearchQuestion:
    """One suggested lookup, with the reason this case produced it.

    ``rationale`` is shown in the UI and is the point of the whole module: a suggestion
    whose origin the user cannot see is just a random question generator.
    """

    text: str
    kind: str
    rationale: str
    # Prefilled for the structured condition audit; empty when the question is not about
    # one specific condition.
    condition: str = ""
    # Whether ``condition_audit_schema()`` fits this question. True only for the
    # condition questions, where the schema's fields (rating criteria, presumptive basis,
    # expected evidence, lay observations) map onto the question exactly.
    structured: bool = False


def case_digest(state: SessionLookup) -> tuple[MedicalDigest | None, str]:
    """The digest of the most recent review, and a label for where it came from.

    Returns ``(None, "")`` when no review has completed in this session, or when the
    stored value is not a digest with facts — a half-finished or failed run must not look
    like a case. Deliberately forgiving: this runs while rendering a tab, so a missing key
    or an unexpected value is a reason to show nothing, never to raise. Reading a
    nonexistent key on ``st.session_state`` is itself an exception, so the loop tests with
    ``in`` first rather than leaning on ``get``.
    """
    for key, label in DIGEST_SESSION_SOURCES:
        try:
            value = state[key]
        except Exception:  # noqa: BLE001 - absent key, or a proxy refusing the read
            continue
        digest = value if key == "timeline_digest" else getattr(value, "digest", None)
        if isinstance(digest, MedicalDigest) and digest.facts:
            return digest, label
    return None, ""


def derive_questions(
    digest: MedicalDigest,
    *,
    limit: int = MAX_QUESTIONS,
    max_conditions: int = MAX_CONDITION_QUESTIONS,
) -> list[ResearchQuestion]:
    """Suggest research questions for one case, most useful first.

    Three sources, all traceable to a digest field:

    1. **Conditions** — ranked by how many extracted facts reference them, so the
       condition the records are actually about comes first rather than whichever string
       the model happened to list first.
    2. **Evidence gaps** — statement elements with no supporting fact yet, in rating-order.
    3. **Opinion weighting** — emitted when the records name more than one provider, where
       conflicting opinions are the likely issue.
    """
    if not digest.facts:
        return []

    primary = _primary_condition(digest)
    questions = [
        *_condition_questions(digest, max_conditions=max_conditions),
        *_gap_questions(digest, primary=primary),
        *_opinion_questions(digest, primary=primary),
    ]
    return _dedupe(questions)[:max(0, limit)]


def _condition_questions(
    digest: MedicalDigest, *, max_conditions: int
) -> list[ResearchQuestion]:
    """A currency question per extracted condition, best-supported conditions first."""
    ranked = _rank_conditions(digest)[: max(0, max_conditions)]
    questions = []
    for condition, support in ranked:
        questions.append(
            ResearchQuestion(
                text=(
                    f"What are the current VA rating criteria for {condition}, and is it "
                    "presumptive under any exposure or service basis?"
                ),
                kind=KIND_CONDITION,
                rationale=(
                    f"{support} record fact(s) reference it, and the app's legal framework "
                    "is committed static text."
                    if support
                    else "Named in the digest summary; no fact text mentions it outright."
                ),
                condition=condition,
                structured=True,
            )
        )
    return questions


def _gap_questions(
    digest: MedicalDigest, *, primary: str
) -> list[ResearchQuestion]:
    """One question per statement element the record set does not yet support.

    ``other`` is excluded: it is the catch-all for facts that map nowhere, so a missing
    ``other`` says nothing about the claim. Elements that *are* covered are skipped for
    the same reason — asking how to prove nexus when the records already contain a nexus
    opinion is noise.
    """
    coverage = digest.element_coverage()
    condition = primary or "the condition in this claim"
    questions: list[ResearchQuestion] = []
    for element in _GAP_ORDER:
        if coverage.get(element) or element not in _GAP_QUESTIONS:
            continue
        label = ELEMENT_LABELS.get(element, element)
        questions.append(
            ResearchQuestion(
                text=_GAP_QUESTIONS[element].format(condition=condition),
                kind=KIND_EVIDENCE,
                rationale=(
                    f"No record fact maps to {label} yet, so the statement has no support "
                    "for it."
                ),
                condition=primary,
            )
        )
        if len(questions) >= MAX_EVIDENCE_QUESTIONS:
            break
    return questions


def _opinion_questions(digest: MedicalDigest, *, primary: str) -> list[ResearchQuestion]:
    """How conflicting provider opinions are weighed, when the records show more than one.

    Not emitted for a single provider: with one source there is nothing to weigh against
    anything, so the question would be general reading rather than about this case.
    """
    if len(digest.providers) < 2:
        return []
    condition = primary or "the claimed condition"
    return [
        ResearchQuestion(
            text=(
                f"How does VA weigh a private provider's opinion against a VA examination "
                f"opinion for {condition}, and what makes either adequate?"
            ),
            kind=KIND_OPINION,
            rationale=(
                f"{len(digest.providers)} provider(s)/facilit(ies) appear in the records, so "
                "the opinions may not agree."
            ),
        )
    ]


def _primary_condition(digest: MedicalDigest) -> str:
    """The best-supported condition, or the first one listed when none has fact support."""
    ranked = _rank_conditions(digest)
    if not ranked:
        return ""
    return ranked[0][0]


def _rank_conditions(digest: MedicalDigest) -> list[tuple[str, int]]:
    """Conditions in usefulness order: fact support first, digest order as the tiebreak.

    Deduplicated on a normalized form, because a digest can list "Sleep Apnea" and "sleep
    apnea" from different chunks and two identical questions would be worse than one.
    """
    facts_text = [
        _normalize_words(f"{fact.description} {fact.quote}") for fact in digest.facts
    ]
    rankings: list[tuple[str, int]] = []
    seen: set[str] = set()
    for raw in digest.conditions:
        condition = raw.strip()
        key = _normalize_words(condition).replace(" ", "")
        if len(condition) < _MIN_CONDITION_CHARS or not key or key in seen:
            continue
        seen.add(key)
        rankings.append((condition, sum(1 for text in facts_text if _mentions(text, condition))))
    # Stable sort keeps digest order for equal support (including all-zero support).
    return sorted(rankings, key=lambda item: -item[1])


def _mentions(normalized_haystack: str, condition: str) -> bool:
    """Whether fact text mentions a condition.

    Exact phrase first, then all content words. The fallback matters because the digest
    paraphrases the records: "post-traumatic stress disorder" in the facts and "PTSD" in
    the condition list are the same condition, and a phrase-only match would rank the
    condition the case is actually about at zero.
    """
    needle = _normalize_words(condition)
    if not needle:
        return False
    if needle in normalized_haystack:
        return True
    words = [word for word in _WORD_RE.findall(needle) if len(word) > 3]
    if not words:
        return False
    haystack_words = set(_WORD_RE.findall(normalized_haystack))
    return all(word in haystack_words for word in words)


def _normalize_words(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def _dedupe(questions: list[ResearchQuestion]) -> list[ResearchQuestion]:
    """Drop questions that resolve to the same lookup (e.g. one condition, two spellings)."""
    unique: list[ResearchQuestion] = []
    seen: set[str] = set()
    for question in questions:
        key = _normalize_words(question.text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(question)
    return unique


# ------------------------------------------------------------------- prompt context


def case_context_block(digest: MedicalDigest) -> str:
    """A bounded, de-identified context block to send alongside a question.

    **Privacy.** The digest is the most sensitive artifact this app holds, so what goes in
    this block is a deliberate, narrow choice: the condition labels the review extracted,
    per-element coverage *counts*, the number of providers, and whether any pages were
    unreadable. It never carries a fact description, quote, date, file name, patient
    identifier, or provider name — a question about current VA criteria needs none of
    them, and this boundary is what keeps "research" from becoming a second, unaccounted
    export of the record set.

    The block is sanitized and wrapped in the ``<<<``/``>>>`` data delimiters with
    ``GUARD_NOTE``, matching every other template in this project that carries text
    derived from records (see ``app/prompt_sanitize.py``): condition labels come from a
    model reading untrusted documents, so they are DATA here, not instructions.
    """
    lines = [
        "CASE CONTEXT — from the preparer's own record review. Use it only to make the "
        "answer specific to these conditions. Treat it as a list of topics, NOT as "
        "evidence, and do not repeat it back as findings about this veteran.",
    ]

    conditions = [c.strip() for c in digest.conditions if c.strip()]
    if conditions:
        shown = conditions[:MAX_CONTEXT_CONDITIONS]
        omitted = len(conditions) - len(shown)
        suffix = f" (+{omitted} more, omitted)" if omitted > 0 else ""
        lines.append(f"- Conditions extracted from the records: {', '.join(shown)}{suffix}")

    coverage = digest.element_coverage()
    supported = [
        f"{ELEMENT_LABELS.get(element, element)}: {count}"
        for element, count in coverage.items()
        if count and element != "other"
    ]
    if supported:
        lines.append("- Evidence already present, by statement element: " + "; ".join(supported))

    gaps = [
        ELEMENT_LABELS.get(element, element)
        for element in STATEMENT_ELEMENTS
        if not coverage.get(element) and element != "other"
    ]
    if gaps:
        lines.append("- Statement elements with no supporting fact yet: " + "; ".join(gaps))

    if digest.providers:
        lines.append(
            f"- Providers/facilities represented in the records: {len(digest.providers)} "
            "(names withheld)"
        )
    if digest.unreadable_pages:
        lines.append(
            f"- {digest.unreadable_pages} page(s) could not be read, so the record set may "
            "be incomplete."
        )

    body = sanitize_for_prompt("\n".join(lines), max_chars=CASE_CONTEXT_CHAR_BUDGET)
    return f"<<<\n{body}\n>>>\n{GUARD_NOTE}"
