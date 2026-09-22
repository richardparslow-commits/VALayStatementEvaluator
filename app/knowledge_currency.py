"""Is the committed legal framework still current? A grounded, cached, per-topic verdict.

The app reasons from static committed markdown in ``app/knowledge/`` — the 12-topic
checklist, the legal framework, the rubric, the drafting guide. That text encodes VA law as
it stood when it was written, and VA law moves: presumptive conditions are added, the rating
schedule is amended, M21-1 procedure changes. Until now nothing in the app noticed, and the
failure mode is silent — a stale topic produces confident, wrong guidance at exactly the
point the user is least able to check it.

This module closes that loop with the grounded research path:

    committed text ──▶ parse topic sections ──▶ Perplexity Agent (per-topic verdict)
                                                          │
                                                          ▼
                                                  cached CurrencyReport
                                                          │
                             Evaluate tab ◀───────────────┘  read-only: no network, no spend

Three properties are deliberate and load-bearing:

1. **Verifying is explicit; flagging is automatic.** Only :func:`verify_framework_currency`
   costs money (one Agent API call, billed per tool invocation). The Evaluate tab calls
   :func:`case_currency_flag`, which only *reads* the cached report — a 20-minute
   evaluation must never acquire a surprise network call or a surprise bill.
2. **A verdict names the exact text it reviewed.** The report carries a fingerprint of the
   framework files. If those files change, the verdict expires on read no matter how recent
   it is; otherwise editing the checklist would inherit a "still current" stamp for text
   nobody reviewed.
3. **Absent, expired, and changed-text all read as unverified — never as current.** "We
   have not checked" and "we checked and it is fine" are different claims, and a flag that
   cannot tell them apart is worse than no flag at all.

The report is stored in ``app/shared_cache`` (tiered, process-local LRU plus optional shared
Redis), so one verification serves every session and every pod that shares the cache.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from .config import Settings, load_knowledge
from .logging_config import get_logger
from .perplexity_agent import (
    PerplexityConfigurationError,
    PerplexityParseError,
    framework_currency_schema,
    research,
)
from .prompt_sanitize import GUARD_NOTE, sanitize_for_prompt

logger = get_logger("app.knowledge_currency")

# The committed framework this verdict is about. Only the two files the *topics* live in
# are fingerprinted — the rubric and drafting guide are style/structure documents whose
# currency is not a legal question, so including them would invalidate a good verdict
# every time someone rewords a tip. The rubric does carry decision rules that cite
# authority (a contradiction needs a cited record entry; a normal static exam does not
# contradict a symptom), but those are rules about how to label evidence rather than
# statements of current law, and the authorities behind them are stated — and therefore
# fingerprinted — in the framework and checklist text above.
FRAMEWORK_FILES: tuple[str, ...] = ("topic_checklist.md", "legal_framework.md")

# Stable cache key: deliberately *not* keyed by fingerprint, because the interesting
# states are "never checked" and "checked, but the text changed since" — a keyed lookup
# could not tell those apart.
CACHE_KEY = "va:framework_currency:report"
# Eviction bound, not a freshness rule. Freshness is computed from ``checked_at`` against
# ``Settings.framework_currency_ttl_days`` so the policy is visible in one place; this only
# stops an abandoned key living in the shared cache forever.
CACHE_TTL_SECONDS = 180 * 24 * 3600

# Cap on the checklist excerpt sent for review. All twelve topics run to roughly 14k
# characters, so a full-framework check is normally under this; a truncated excerpt is
# marked by ``sanitize_for_prompt`` and the verdict is then explicitly about what fit.
MAX_EXCERPT_CHARS = 12_000

# Verdict statuses, mirroring the schema enum in ``perplexity_agent``.
STATUS_CURRENT = "current"
STATUS_CHANGED = "changed"
STATUS_UNCONFIRMED = "unclear"
_KNOWN_STATUSES = (STATUS_CURRENT, STATUS_CHANGED, STATUS_UNCONFIRMED)

# Freshness states, mirroring the four things a caller can be looking at.
STATE_NEVER_CHECKED = "never_checked"
STATE_FRAMEWORK_CHANGED = "framework_changed"
STATE_EXPIRED = "expired"
STATE_FRESH = "fresh"

_TOPIC_HEADING_RE = re.compile(r"^##\s+([A-Z])\.\s+(.+?)\s*$", re.MULTILINE)


# ------------------------------------------------------------------------ the frame


@dataclass(frozen=True)
class TopicSection:
    """One lettered topic of the checklist, as committed."""

    letter: str
    title: str
    text: str


def parse_topic_sections(checklist_text: str) -> list[TopicSection]:
    """Split the checklist into one section per lettered topic (``A``–``L``).

    Matches the file's own ``## A. Title`` headings, so unlettered sections ("Writer
    guidelines") are skipped rather than mislabelled — the app's topic vocabulary is the
    lettered set (see ``app/condition_selector.TOPIC_LABELS``), and inventing an entry for
    prose that has no letter would put a topic in front of the user that no mapping, no
    selector, and no evaluation row refers to.
    """
    matches = list(_TOPIC_HEADING_RE.finditer(checklist_text))
    sections: list[TopicSection] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(checklist_text)
        body = checklist_text[match.end() : end].strip()
        sections.append(
            TopicSection(letter=match.group(1), title=match.group(2).strip(), text=body)
        )
    return sections


def topic_labels() -> dict[str, str]:
    """The app's topic vocabulary (letter → label), read from the condition selector.

    Imported lazily: ``app/condition_selector.py`` imports Streamlit at module level, and
    this module is also read from the evaluation side, where importing an entire UI
    framework to get a dict of twelve labels would be a real cost.
    """
    from .condition_selector import TOPIC_LABELS

    return dict(TOPIC_LABELS)


def normalize_topics(topics: Iterable[str]) -> list[str]:
    """Upper-case, de-duplicate, and keep only letters this app actually has.

    Silently dropping unknown entries is right here: the letters can arrive from session
    state written by an older build of the selector, and asking the API about a topic that
    no longer exists would produce a verdict nothing can be matched against.
    """
    known = topic_labels()
    seen: list[str] = []
    for raw in topics:
        letter = str(raw).strip().upper()
        if letter in known and letter not in seen:
            seen.append(letter)
    return [letter for letter in sorted(seen)]


# --------------------------------------------------------------------- the report


@dataclass(frozen=True)
class TopicVerdict:
    """What the grounded check concluded about one topic's committed text."""

    topic: str
    status: str
    note: str = ""
    authority: str = ""

    @property
    def label(self) -> str:
        return topic_labels().get(self.topic, self.topic)

    @property
    def stale(self) -> bool:
        """Whether the committed text is known to be out of date."""
        return self.status == STATUS_CHANGED

    @property
    def unconfirmed(self) -> bool:
        """Whether the check could not establish that the text is current."""
        return self.status == STATUS_UNCONFIRMED


@dataclass(frozen=True)
class CurrencyReport:
    """A completed currency check: when, against which text, and what it found."""

    checked_at: str
    fingerprint: str
    verdicts: tuple[TopicVerdict, ...]
    preset: str = ""
    model: str = ""
    request_id: str = ""
    latency_ms: int = 0

    @property
    def topics(self) -> tuple[str, ...]:
        return tuple(verdict.topic for verdict in self.verdicts)

    @property
    def stale(self) -> list[TopicVerdict]:
        return [verdict for verdict in self.verdicts if verdict.stale]

    @property
    def unconfirmed(self) -> list[TopicVerdict]:
        return [verdict for verdict in self.verdicts if verdict.unconfirmed]

    def for_topics(self, topics: Iterable[str]) -> list[TopicVerdict]:
        """Only the verdicts for these letters, in the report's own order."""
        wanted = set(normalize_topics(topics))
        return [verdict for verdict in self.verdicts if verdict.topic in wanted]

    def age_days(self, now: datetime | None = None) -> float | None:
        """Days since the check ran, or ``None`` if ``checked_at`` is unreadable."""
        checked = _parse_timestamp(self.checked_at)
        if checked is None:
            return None
        return max(0.0, ((now or _utc_now()) - checked).total_seconds() / 86_400.0)

    def to_json(self) -> str:
        return json.dumps(
            {
                "checked_at": self.checked_at,
                "fingerprint": self.fingerprint,
                "preset": self.preset,
                "model": self.model,
                "request_id": self.request_id,
                "latency_ms": self.latency_ms,
                "verdicts": [
                    {
                        "topic": verdict.topic,
                        "status": verdict.status,
                        "note": verdict.note,
                        "authority": verdict.authority,
                    }
                    for verdict in self.verdicts
                ],
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> CurrencyReport | None:
        """Parse a cached report, or ``None`` when the stored value is unusable.

        Forgiving by design: this runs while rendering a tab, and a cache entry written by
        an older build (or a truncated one) is a reason to report "not verified", never to
        raise inside the Evaluate tab.
        """
        try:
            data: Any = json.loads(raw)
        except (TypeError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        checked_at = data.get("checked_at")
        fingerprint = data.get("fingerprint")
        rows = data.get("verdicts")
        if not isinstance(checked_at, str) or not isinstance(fingerprint, str):
            return None
        if not isinstance(rows, list):
            return None
        verdicts = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            topic = str(row.get("topic", "")).strip().upper()
            if not topic:
                continue
            verdicts.append(
                TopicVerdict(
                    topic=topic,
                    status=_coerce_status(row.get("status")),
                    note=str(row.get("note", "") or ""),
                    authority=str(row.get("authority", "") or ""),
                )
            )
        if not verdicts:
            return None
        return cls(
            checked_at=checked_at,
            fingerprint=fingerprint,
            verdicts=tuple(verdicts),
            preset=str(data.get("preset", "") or ""),
            model=str(data.get("model", "") or ""),
            request_id=str(data.get("request_id", "") or ""),
            latency_ms=_coerce_int(data.get("latency_ms")),
        )


# ----------------------------------------------------------------- fingerprinting


def framework_fingerprint() -> str:
    """A digest of the committed framework text a verdict applies to.

    Deliberately over content, not mtime: a checkout, a redeploy, or a touch would all
    change mtimes without changing a word, and each false "the framework changed, re-check"
    would be a paid call the user did not need. Returns ``"missing"`` when a file cannot be
    read, which can never equal a stored fingerprint — an unreadable framework must not
    validate a verdict.
    """
    digest = hashlib.sha256()
    for name in FRAMEWORK_FILES:
        try:
            content = load_knowledge(name)
        except OSError:
            logger.warning("framework file unreadable for fingerprinting: %s", name)
            return "missing"
        digest.update(name.encode("utf-8"))
        digest.update(content.encode("utf-8"))
    return digest.hexdigest()


# ------------------------------------------------------------------ the live check


def verify_framework_currency(
    *,
    settings: Settings,
    topics: Sequence[str],
    domains: Sequence[str] | None = None,
) -> CurrencyReport:
    """Run the grounded currency check for ``topics`` and store the report.

    One Agent API call for the whole selection: the checklist excerpts go in the
    instructions, the per-topic verdicts come back as structured output constrained to the
    letters being reviewed. Raises :class:`PerplexityConfigurationError` when there is
    nothing to check or the integration is unusable, and otherwise the same failure types
    as any other research call (see ``app/perplexity_agent.research``) — the caller reports
    them through the usual failure path.
    """
    letters = normalize_topics(topics)
    if not letters:
        raise PerplexityConfigurationError(
            "Select at least one checklist topic to check."
        )

    sections = {section.letter: section for section in _checklist_sections()}
    selected = [sections[letter] for letter in letters if letter in sections]
    if not selected:
        raise PerplexityConfigurationError(
            "None of the selected topics could be found in the committed checklist. "
            "app/knowledge/topic_checklist.md may be missing or renamed."
        )

    answer = research(
        _currency_question(selected),
        settings=settings,
        instructions=_currency_instructions(selected),
        # Primary sources by default: a currency verdict is only worth storing if it rests
        # on the regulation, statute, or handbook it names.
        domains=list(domains) if domains is not None else settings.perplexity_source_domains(),
        schema=framework_currency_schema([section.letter for section in selected]),
    )

    report = CurrencyReport(
        checked_at=_utc_now().isoformat(),
        fingerprint=framework_fingerprint(),
        verdicts=tuple(_parse_verdicts(answer.findings, expected=[s.letter for s in selected])),
        preset=answer.preset or "",
        model=answer.model or "",
        request_id=answer.response_id,
        latency_ms=answer.latency_ms,
    )
    store_report(report)
    return report


def _checklist_sections() -> list[TopicSection]:
    return parse_topic_sections(load_knowledge("topic_checklist.md"))


def _currency_question(sections: Sequence[TopicSection]) -> str:
    listed = "; ".join(f"{section.letter} ({section.title})" for section in sections)
    return (
        "Is this app's committed lay-statement topic checklist still accurate under "
        f"current VA law and procedure? Check these topics: {listed}."
    )


def _currency_instructions(sections: Sequence[TopicSection]) -> str:
    """The standing rules plus the committed text under review, as one prompt block."""
    excerpt = "\n\n".join(
        f"[{section.letter}] {section.title}\n{section.text}" for section in sections
    )
    body = sanitize_for_prompt(excerpt, max_chars=MAX_EXCERPT_CHARS)
    return (
        "You are auditing a veterans-claims reference document for CURRENCY, not for style "
        "or completeness. Between <<< and >>> is the committed checklist text for specific "
        "topics, each introduced by its bracketed letter.\n"
        "For every topic, judge from current primary sources whether the text is still "
        "accurate as written:\n"
        "- current: the substance still matches current law and VA procedure\n"
        "- changed: current law or procedure differs, or the text omits a change in a way "
        "that would now mislead a claim preparer\n"
        "- unclear: primary sources conflict, or you cannot confirm from primary sources\n"
        "Return exactly one verdict per topic, using the letter as given. In note, say "
        "concretely what changed, or why the text is still accurate. In authority, give the "
        "short title of the controlling source (for example '38 C.F.R. § 4.130'); never a "
        "URL. Do not invent a source you did not find.\n\n"
        f"<<<\n{body}\n>>>\n{GUARD_NOTE}"
    )


def _parse_verdicts(findings: dict[str, Any] | None, *, expected: Sequence[str]) -> list[TopicVerdict]:
    """Turn structured output into exactly one verdict per requested letter.

    Missing letters are *filled in* as unconfirmed rather than dropped. A model that
    answers eight of twelve topics must not produce a report in which the other four look
    fine by their absence — the whole value of this check is that it distinguishes
    "verified current" from "not established".
    """
    if findings is None:
        raise PerplexityParseError(
            "The currency check returned no structured findings, so nothing could be "
            "verified. Treat the committed framework as unverified."
        )
    rows = findings.get("verdicts")
    if not isinstance(rows, list):
        rows = []

    by_letter: dict[str, TopicVerdict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        letter = str(row.get("topic", "")).strip().upper()
        if letter in by_letter:
            continue
        by_letter[letter] = TopicVerdict(
            topic=letter,
            status=_coerce_status(row.get("status")),
            note=str(row.get("note", "") or "").strip(),
            authority=str(row.get("authority", "") or "").strip(),
        )

    ordered: list[TopicVerdict] = []
    for letter in expected:
        verdict = by_letter.get(letter)
        if verdict is None:
            verdict = TopicVerdict(
                topic=letter,
                status=STATUS_UNCONFIRMED,
                note="The check returned no verdict for this topic.",
            )
        ordered.append(verdict)
    return ordered


def _coerce_int(value: Any) -> int:
    """Read an int from cached JSON, defaulting to 0 for anything unexpected."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _coerce_status(value: Any) -> str:
    """Map anything unexpected onto ``unclear``.

    An unrecognized status is not a reason to fail the whole check, but it must not be
    silently read as ``current`` either — that is the one error this feature cannot afford.
    """
    text = str(value or "").strip().lower()
    if text in _KNOWN_STATUSES:
        return text
    if text in ("stale", "outdated", "superseded"):
        return STATUS_CHANGED
    return STATUS_UNCONFIRMED


# ------------------------------------------------------------------------ storage


def store_report(report: CurrencyReport, *, ttl_seconds: int = CACHE_TTL_SECONDS) -> None:
    """Best-effort cache write. A cache failure must not fail a completed check."""
    try:
        from .shared_cache import get_cache

        get_cache().set(CACHE_KEY, report.to_json(), ttl_seconds=ttl_seconds)
    except Exception:  # noqa: BLE001 - the report is still returned to the caller
        logger.warning("could not cache the framework currency report", exc_info=True)


def load_report() -> CurrencyReport | None:
    """The stored report, whatever text it was about, or ``None`` when there is none."""
    try:
        from .shared_cache import get_cache

        raw = get_cache().get(CACHE_KEY)
    except Exception:  # noqa: BLE001 - an unreachable cache reads as "not verified"
        logger.warning("could not read the framework currency report", exc_info=True)
        return None
    if not raw:
        return None
    return CurrencyReport.from_json(raw)


def forget_report() -> None:
    """Drop the stored report (used by tests, and by an operator who wants a re-check)."""
    try:
        from .shared_cache import get_cache

        get_cache().delete(CACHE_KEY)
    except Exception:  # noqa: BLE001 - best effort
        logger.warning("could not clear the framework currency report", exc_info=True)


# ------------------------------------------------------------------- reading side


def freshness(report: CurrencyReport | None, *, ttl_days: int, now: datetime | None = None) -> str:
    """Classify a report against the current framework text and freshness window."""
    if report is None:
        return STATE_NEVER_CHECKED
    if report.fingerprint != framework_fingerprint():
        return STATE_FRAMEWORK_CHANGED
    age = report.age_days(now)
    if age is None or age > ttl_days:
        return STATE_EXPIRED
    return STATE_FRESH


@dataclass(frozen=True)
class CurrencyFlag:
    """What the Evaluate tab needs to say about this case's topics.

    ``stale`` and ``unconfirmed`` are already narrowed to the case's own topics: a stale
    topic the statement never touches is not this user's problem, and noise is what makes
    people stop reading warnings.
    """

    state: str
    report: CurrencyReport | None = None
    age_days: float | None = None
    stale: tuple[TopicVerdict, ...] = ()
    unconfirmed: tuple[TopicVerdict, ...] = ()

    @property
    def verified(self) -> bool:
        """Whether a fresh verdict exists for the framework currently on disk."""
        return self.state == STATE_FRESH

    def covered(self) -> tuple[str, ...]:
        """The case's topics this report actually speaks to."""
        return tuple(verdict.topic for verdict in (*self.stale, *self.unconfirmed)) or (
            self.report.topics if self.report else ()
        )


def case_currency_flag(
    topics: Sequence[str],
    *,
    ttl_days: int,
    report: CurrencyReport | None = None,
    now: datetime | None = None,
) -> CurrencyFlag:
    """Summarize the stored verdict for one case's topics.

    Never calls the API. When nothing has been verified, the caller is expected to say so
    explicitly rather than showing nothing — silence would read as "all current", which is
    the one claim this function cannot make.

    ``report`` defaults to the stored one rather than to "nothing verified": every real
    caller wants the stored verdict, and a default of ``None`` would silently disable
    flagging wherever a caller forgot to load it (which is exactly the bug this signature
    exists to prevent). Tests inject a report to pin freshness and narrowing without a
    cache.
    """
    if report is None:
        report = load_report()
    state = freshness(report, ttl_days=ttl_days, now=now)
    age = report.age_days(now) if report is not None else None
    if report is None:
        return CurrencyFlag(state=state, report=None, age_days=None)
    relevant = report.for_topics(topics)
    return CurrencyFlag(
        state=state,
        report=report,
        age_days=age,
        stale=tuple(verdict for verdict in relevant if verdict.stale),
        unconfirmed=tuple(verdict for verdict in relevant if verdict.unconfirmed),
    )


def with_verdicts(report: CurrencyReport, verdicts: Sequence[TopicVerdict]) -> CurrencyReport:
    """A copy of ``report`` carrying only ``verdicts`` (kept for callers that subset one)."""
    return replace(report, verdicts=tuple(verdicts))


# --------------------------------------------------------------------------- time


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    # A naive timestamp can only have come from a hand-written entry; assume UTC rather
    # than raising, since the alternative is losing an otherwise usable report.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
