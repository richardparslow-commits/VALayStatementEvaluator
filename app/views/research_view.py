"""Research tab: web-grounded, cited answers about current VA law.

Why a separate tab and a separate provider rather than more prompting of the pipeline
model: ``app/knowledge/*.md`` is the legal framework the evaluator and drafter reason
from, and it is *static committed text*. VA law is not static — presumptive conditions,
rating criteria, and M21-1 procedure all change — so the app had no way to answer "is this
framework still current?" for a given condition. Answering that needs the live web and
citable sources, which is what the Perplexity Agent API provides (see
``app/perplexity_agent.py`` for why that is the Agent API and not the Router API).

The questions are not typed by hand. ``app/research_questions.py`` derives them from the
digest of the case the user last reviewed — the conditions the review extracted, and the
statement elements its facts do *not* cover — so the tab opens onto this case's real gaps
instead of a blank box. Deriving them is offline and free, but running one is a paid,
web-grounded request, so picking a derived question loads it into the form for review
rather than firing it on the click.

Everything here is metadata-only in the audit stream: preset, source and citation counts,
and the condition *classification* — never the question text or the answer. Terminal
failures carry a ``req_…`` reference and a "What happened?" expander, exactly like the
Evaluate and Draft tabs, because a research failure the user cannot quote is a failure
nobody can diagnose.
"""
from __future__ import annotations

import streamlit as st

from .. import audit as audit_log
from .. import knowledge_currency as currency
from ..config import PERPLEXITY_PRESETS, Settings, load_settings
from ..logging_config import get_logger
from ..perplexity_agent import (
    SDK_REQUIREMENTS_FILE,
    GroundedAnswer,
    PerplexityError,
    condition_audit_schema,
    configured_preset,
    research,
    unavailable_reason,
)
from ..prompt_sanitize import sanitize_for_prompt
from ..research_questions import (
    ResearchQuestion,
    case_context_block,
    case_digest,
    derive_questions,
)
from ..run_log import run_log_event
from .ops import render_failure_detail
from .shared import check_shutdown_gate, ensure_request_id, report_failure

logger = get_logger("app.views.research_view")

# Where the last answer lives. Session state (not a module global) so two browser
# sessions cannot read each other's research, matching how the other tabs hold results.
# Bound on the question text. The same reasoning as the statement/observations caps:
# a single field must not be able to flood the prompt, and the truncation is visibly
# marked by sanitize_for_prompt rather than silently dropped.
RESEARCH_QUESTION_MAX_CHARS = 2_000
ANSWER_KEY = "research_last_answer"
# Kept separate from the form widgets so the result survives the rerun a form submit
# triggers, without re-running the request.
ERROR_KEY = "research_last_error"

# Form widget keys, named as constants because ``_load_question`` writes them directly to
# seed the form with a derived question.
QUESTION_KEY = "research_question"
CONDITION_KEY = "research_condition"
STRUCTURED_KEY = "research_structured"
CASE_CONTEXT_KEY = "research_case_context"
# The derived question text most recently loaded, or "" when the user is writing their
# own. Doubles as the panel's "has the user already picked one?" flag and as the signal
# for whether a run's question was machine-derived.
LOADED_CASE_QUESTION_KEY = "research_loaded_case_question"
# The radio selection among this case's derived questions.
CASE_QUESTION_KEY = "research_case_question"
# Framework-currency panel: the topic selection, and the last failure text (these are the
# one place in the app that spends money on a currency check).
CURRENCY_TOPICS_KEY = "research_currency_topics"
CURRENCY_ERROR_KEY = "research_currency_error"


def render_research_tab() -> None:
    """Render the Research tab."""
    settings = _settings()
    st.subheader("🧭 Research current VA law")
    st.caption(
        "Ask a question that needs the live web — current rating criteria, presumptive "
        "status, a recent procedure change — and get an answer with the sources it came "
        "from. This is how you check whether the framework the evaluator reasons from is "
        "still current."
    )

    reason = unavailable_reason(settings)
    if reason is not None:
        st.info(reason)
        _render_scope_note()
        return

    # Derived from the digest of the case the user last reviewed, above the form on
    # purpose: choosing one of these has to seed the form's widgets through session
    # state, and Streamlit only accepts that before the widget is instantiated (see
    # ``_load_question``).
    digest, origin = case_digest(st.session_state)
    questions = derive_questions(digest) if digest is not None else []
    _render_case_context(origin=origin, questions=questions)

    _render_framework_currency(settings)

    _render_form(settings, has_case=bool(questions))
    _render_last_result()


def _settings() -> Settings:
    """The session's settings object, falling back to a fresh load.

    The sidebar owns ``st.session_state.settings`` and mutates it in place; this tab is
    renderable on its own (and in tests) without the sidebar having run first.
    """
    settings = st.session_state.get("settings")
    return settings if isinstance(settings, Settings) else load_settings()


def _render_case_context(*, origin: str, questions: list[ResearchQuestion]) -> None:
    """Offer the questions this case implies, derived from its own digest.

    Collapsed once the user has loaded one — at that point they are here to run it, not to
    browse — and expanded when they have not, because the derived list is the feature.
    """
    if not questions:
        return
    loaded = bool(st.session_state.get(LOADED_CASE_QUESTION_KEY))
    with st.expander(f"🗂️ Questions from this case ({len(questions)})", expanded=not loaded):
        st.caption(
            f"Derived from {origin} — offline, from the conditions and evidence gaps in "
            "the digest. Nothing from the records is sent to produce these."
        )
        picked = st.radio(
            "Pick one to load into the form",
            options=questions,
            format_func=lambda item: item.text,
            key=CASE_QUESTION_KEY,
        )
        st.caption(f"Why this one: {picked.rationale}")
        if st.button("Load into the form", key="research_load_case_question"):
            _load_question(picked)


def _render_framework_currency(settings: Settings) -> None:
    """Verify (and refresh) the grounded verdict on the committed checklist.

    This is the only place in the app that spends money on a currency check. The Evaluate
    tab reads the stored verdict and never calls out, so that an evaluation cannot
    acquire a hidden network call or a hidden bill (see ``app/knowledge_currency``).
    """
    report = currency.load_report()
    state = currency.freshness(report, ttl_days=settings.framework_currency_ttl_days)
    with st.expander(
        "📐 Framework currency — is the committed checklist still current?",
        expanded=state != currency.STATE_FRESH,
    ):
        st.caption(
            "The evaluator and drafter reason from committed markdown in app/knowledge/ — "
            "static text that encodes VA law as it stood when it was written. This compares "
            "that text against current primary sources and returns one verdict per topic, "
            "which is what lets the Evaluate tab flag a topic that has gone out of date."
        )
        st.caption(_currency_state_line(state, report, settings.framework_currency_ttl_days))

        labels = currency.topic_labels()
        options = list(labels)
        selected = st.multiselect(
            "Checklist topics to check",
            options=options,
            default=options,
            format_func=lambda letter: f"{letter} — {labels[letter]}",
            key=CURRENCY_TOPICS_KEY,
            help=(
                "One API call covers the whole selection. Narrow it to a few topics for a "
                "cheaper, faster check."
            ),
        )
        if st.button("Verify selected topics", key="research_verify_currency"):
            _run_currency_check(settings, selected)

        failure = st.session_state.get(CURRENCY_ERROR_KEY)
        if isinstance(failure, str) and failure:
            st.error(failure)
            render_failure_detail(failure.rsplit("reference: ", 1)[-1].rstrip(")"))

        if report is not None:
            _render_currency_report(report)


def _currency_state_line(
    state: str, report: currency.CurrencyReport | None, ttl_days: int
) -> str:
    """One honest sentence about where the stored verdict stands.

    The four states are kept distinct on purpose: "never checked", "checked but the text
    changed", "checked too long ago", and "verified" are different claims, and collapsing
    them would let an expired or inapplicable verdict read as reassurance.
    """
    if state == currency.STATE_FRAMEWORK_CHANGED and report is not None:
        return (
            "⚠️ The committed framework files have changed since this check ran, so the "
            "verdict below describes text that is no longer on disk. Verify again."
        )
    if state == currency.STATE_EXPIRED and report is not None:
        return (
            f"⌛ Last checked {_age_phrase(report.age_days())}, which is past the "
            f"{ttl_days}-day window. Treat the verdict below as unverified until re-checked."
        )
    if state == currency.STATE_FRESH and report is not None:
        return (
            f"✅ Verified {_age_phrase(report.age_days())} — {len(report.verdicts)} topic(s) "
            f"checked{_model_phrase(report)}."
        )
    return "⏳ Not verified yet: nothing has checked these topics against current VA law."


def _age_phrase(age_days: float | None) -> str:
    """Human wording for an age in days (a fraction of a day reads as "today")."""
    if age_days is None:
        return "at an unknown time"
    whole = int(age_days)
    if whole <= 0:
        return "today"
    return f"{whole} day{'s' if whole != 1 else ''} ago"


def _model_phrase(report: currency.CurrencyReport) -> str:
    """" (model, preset) when the call recorded them"""
    bits = [bit for bit in (report.model, report.preset) if bit]
    return f" ({', '.join(bits)})" if bits else ""


def _render_currency_report(report: currency.CurrencyReport) -> None:
    """The per-topic verdict table, then the topics that need action."""
    status_labels = {
        currency.STATUS_CURRENT: "✅ current",
        currency.STATUS_CHANGED: "⚠️ changed",
        currency.STATUS_UNCONFIRMED: "❓ unclear",
    }
    st.dataframe(
        [
            {
                "Topic": f"{verdict.topic} — {verdict.label}",
                "Status": status_labels.get(verdict.status, verdict.status),
                "What the check found": verdict.note,
                "Authority": verdict.authority,
            }
            for verdict in report.verdicts
        ],
        width="stretch",
        hide_index=True,
    )

    if report.stale:
        st.warning(
            "**These committed topics no longer match current law.** Guidance the app "
            "derives from them — evaluation rubrics, drafting advice, and the checklist "
            "itself — should be treated as unreliable until app/knowledge/ is updated:"
        )
        for verdict in report.stale:
            suffix = f" — {verdict.authority}" if verdict.authority else ""
            st.markdown(f"- **{verdict.topic} — {verdict.label}:** {verdict.note}{suffix}")
    if report.unconfirmed:
        st.info(
            "**Not established either way** (primary sources conflicted, or the check "
            "could not confirm): "
            + ", ".join(f"{v.topic} — {v.label}" for v in report.unconfirmed)
        )


def _run_currency_check(settings: Settings, topics: list[str]) -> None:
    """Run one paid currency check, with the tab's usual audit + attribution around it."""
    rid = ensure_request_id()
    st.session_state.pop(CURRENCY_ERROR_KEY, None)
    letters = currency.normalize_topics(topics)
    if not letters:
        st.warning("Select at least one checklist topic to check.")
        return
    if not check_shutdown_gate("research"):
        return

    audit_log.audit_event(
        "research", "start", request_id=rid,
        record_sources=["Perplexity"], llm_endpoints=["perplexity-agent"],
        outcome={"check": "framework_currency", "topics": len(letters)},
    )
    run_log_event("framework_currency", "start", request_id=rid, topics=len(letters))

    with st.spinner(f"Checking {len(letters)} checklist topic(s) against current VA law…"):
        try:
            report = currency.verify_framework_currency(settings=settings, topics=letters)
        except Exception as exc:  # noqa: BLE001 - every failure path gets a reference
            message = report_failure(
                f"Framework currency check failed: {exc}",
                phase="research",
                exc=exc,
                request_id=rid,
            )
            st.session_state[CURRENCY_ERROR_KEY] = message
            audit_log.audit_event(
                "research", "error", request_id=rid, error_class=type(exc).__name__,
                error_message=str(exc), llm_endpoints=["perplexity-agent"],
                outcome={"check": "framework_currency", "topics": len(letters)},
            )
            run_log_event(
                "framework_currency", "error", request_id=rid, topics=len(letters),
                error=str(exc), error_class=type(exc).__name__,
            )
            logger.warning(
                "framework currency check failed: %s",
                type(exc).__name__,
                extra={"request_id": rid, "phase": "research", "status": "error"},
            )
            return

    audit_log.audit_event(
        "research", "ok", request_id=rid, duration_ms=report.latency_ms,
        llm_endpoints=["perplexity-agent"],
        outcome={
            "check": "framework_currency",
            "topics": len(letters),
            "changed": len(report.stale),
            "unclear": len(report.unconfirmed),
        },
    )
    run_log_event(
        "framework_currency", "ok", request_id=rid, topics=len(letters),
        changed=len(report.stale), unclear=len(report.unconfirmed),
        duration_ms=report.latency_ms,
    )
    # Rerun so the panel above re-reads the stored report rather than working from the
    # snapshot taken before the call.
    st.rerun()


def _load_question(question: ResearchQuestion) -> None:
    """Seed the form's widgets with a derived question, then rerun.

    Streamlit rejects an assignment to a widget's session-state key *after* that widget
    exists in the current run, and this section renders above the form — so the values are
    written here and ``st.rerun()`` re-executes the script, where the form instantiates
    with them as its initial state. That ordering is why this section must stay above the
    form, and why the form's widgets are declared with ``key=`` and no ``value=`` (a
    widget with both would warn about the value being set twice).
    """
    st.session_state[QUESTION_KEY] = question.text
    st.session_state[CONDITION_KEY] = question.condition
    st.session_state[STRUCTURED_KEY] = question.structured
    st.session_state[LOADED_CASE_QUESTION_KEY] = question.text
    st.rerun()


def _render_form(settings: Settings, *, has_case: bool) -> None:
    """The research form.

    A ``st.form`` rather than bare widgets: Streamlit reruns on every keystroke, and a
    submit here spends real money (model tokens plus per-tool-call billing), so typing
    must not be able to trigger a request. The same reasoning drives the reference lookup
    in ``about_view.py``.
    """
    sources = settings.perplexity_source_domains()
    with st.form("research_form"):
        question = st.text_area(
            "Research question",
            key=QUESTION_KEY,
            height=110,
            placeholder=(
                "e.g. What are the current rating criteria and presumptive bases for "
                "sleep apnea secondary to PTSD?"
            ),
        )
        col_left, col_right = st.columns(2)
        with col_left:
            official_only = st.checkbox(
                "Official sources only",
                value=bool(sources),
                key="research_official_only",
                help=(
                    "Restrict web search to the primary sources this app's framework "
                    "cites"
                    + (f" ({', '.join(sources)})." if sources else ".")
                ),
            )
            # No ``value=`` here: ``_load_question`` sets this key from a derived
            # question, and Streamlit warns when a keyed widget also declares a default.
            structured = st.checkbox(
                "Return a structured condition audit",
                key=STRUCTURED_KEY,
                help="Adds a parseable findings object instead of prose alone.",
            )
            case_context = st.checkbox(
                "Include case context",
                value=True,
                key=CASE_CONTEXT_KEY,
                disabled=not has_case,
                help=(
                    "Sends the condition labels and evidence-gap counts from your last "
                    "record review — never record text, names, or dates — so the answer "
                    "targets this case."
                    if has_case
                    else (
                        "Available after a record review: run the Evaluate or Draft tab "
                        "first."
                    )
                ),
            )
        with col_right:
            preset = st.selectbox(
                "Research depth",
                options=list(PERPLEXITY_PRESETS),
                index=list(PERPLEXITY_PRESETS).index(configured_preset(settings)),
                key="research_preset",
                help=(
                    "Presets bundle model, tools and limits. 'low' is everyday research; "
                    "'high'/'xhigh' are slower and cost more."
                ),
            )
            condition = st.text_input(
                "Condition (for a structured audit)",
                key=CONDITION_KEY,
                placeholder="sleep apnea",
            )
        submitted = st.form_submit_button("Research", type="primary")

    if submitted:
        _run(settings, question=question, preset=preset, official_only=official_only,
             structured=structured, condition=condition, use_case_context=case_context)


def _run(
    settings: Settings,
    *,
    question: str,
    preset: str,
    official_only: bool,
    structured: bool,
    condition: str,
    use_case_context: bool,
) -> None:
    """Execute one research call, with audit + run-log + error attribution around it."""
    rid = ensure_request_id()
    st.session_state.pop(ANSWER_KEY, None)
    st.session_state.pop(ERROR_KEY, None)

    condition_label = condition.strip()[:120]
    if not question.strip():
        st.warning("Enter a research question first.")
        return
    if structured and not condition.strip():
        st.warning("A structured audit needs the condition it should cover.")
        return
    if not check_shutdown_gate("research"):
        return

    # Same guard as every other model-facing input: this text reaches a prompt.
    safe_question = sanitize_for_prompt(question, max_chars=RESEARCH_QUESTION_MAX_CHARS)
    domains = settings.perplexity_source_domains() if official_only else []
    schema = condition_audit_schema() if structured else None

    # Computed at submit time, not carried from the click, so the context always matches
    # the digest as it stands now — and it is bounded and de-identified by
    # ``case_context_block``. An edited derived question keeps its context; a question the
    # user typed by hand gets it too, which is the point of the checkbox.
    context = ""
    if use_case_context:
        digest, _ = case_digest(st.session_state)
        if digest is not None:
            context = case_context_block(digest)
    loaded_question = str(st.session_state.get(LOADED_CASE_QUESTION_KEY) or "").strip()
    from_case = bool(loaded_question) and question.strip() == loaded_question

    audit_log.audit_event(
        "research", "start", request_id=rid, condition=condition_label or None,
        record_sources=["Perplexity"] if official_only else ["Perplexity (open web)"],
        llm_endpoints=["perplexity-agent"],
    )
    run_log_event(
        "research", "start", request_id=rid, preset=preset,
        structured=structured, official_only=official_only,
        question_chars=len(question), from_case=from_case,
        case_context=bool(context),
    )

    with st.spinner(f"Researching with preset '{preset}'…"):
        try:
            answer = research(
                safe_question,
                settings=settings,
                instructions=_instructions(structured, context=context),
                domains=domains,
                schema=schema,
            )
        except Exception as exc:  # noqa: BLE001 - every failure path gets a reference
            _record_failure(exc, rid=rid, preset=preset, structured=structured)
            return

    st.session_state[ANSWER_KEY] = answer
    audit_log.audit_event(
        "research", "ok", request_id=rid, condition=condition_label or None,
        duration_ms=answer.latency_ms, llm_endpoints=["perplexity-agent"],
        outcome={
            "preset": answer.preset or preset,
            "model": answer.model or "",
            "citations": answer.citation_count,
            "structured": structured,
            "from_case": from_case,
            "case_context": bool(context),
        },
    )
    run_log_event(
        "research", "ok", request_id=rid, preset=answer.preset or preset,
        citations=answer.citation_count, structured=structured,
        duration_ms=answer.latency_ms, from_case=from_case,
        case_context=bool(context),
    )


def _instructions(structured: bool, *, context: str = "") -> str:
    """Standing rules for the run, split from the question as the docs direct.

    Deliberately states the app's lay-competence boundary: the panel exists to inform a
    witness, so an answer that reaches for a diagnosis or a causation claim would be
    advice this tool is not allowed to give.

    ``context`` is the already-sanitized, already-delimited case block from
    ``research_questions.case_context_block`` (delimiters plus ``GUARD_NOTE``), so it is
    appended verbatim rather than sanitized twice.
    """
    base = (
        "You are supporting a VA disability claim preparer. Prefer primary sources "
        "(VA, eCFR, the courts) and be explicit when current guidance is unclear or "
        "conflicts with older material. Describe functional impact and observable "
        "behaviour; do not assert diagnoses or causation."
    )
    if structured:
        # Reinforcing the schema in natural language is the documented tip for improving
        # adherence, and the schema itself never asks for links (see
        # perplexity_agent.condition_audit_schema) because the model must not be the
        # source of citations.
        base = (
            f"{base} Return the data as a JSON object matching the schema. Do not include "
            "links or URLs anywhere in the JSON."
        )
    return f"{base}\n\n{context}" if context else base


def _record_failure(exc: BaseException, *, rid: str, preset: str, structured: bool) -> None:
    """Log, attribute, and display a failed research call (never the key or prompt body)."""
    message = f"Research failed: {exc}"
    st.session_state[ERROR_KEY] = report_failure(
        message, phase="research", exc=exc, request_id=rid
    )
    audit_log.audit_event(
        "research", "error", request_id=rid, error_class=type(exc).__name__,
        error_message=str(exc), llm_endpoints=["perplexity-agent"],
    )
    run_log_event(
        "research", "error", request_id=rid, preset=preset, structured=structured,
        error=str(exc), error_class=type(exc).__name__,
    )
    logger.warning(
        "research run failed: %s",
        type(exc).__name__,
        extra={"request_id": rid, "phase": "research", "status": "error"},
    )


def _render_last_result() -> None:
    """Show the error or the answer from the most recent submit."""
    failure = st.session_state.get(ERROR_KEY)
    if isinstance(failure, str) and failure:
        st.error(failure)
        # The reference in that message resolves here without leaving the app.
        reference = failure.rsplit("reference: ", 1)[-1].rstrip(")")
        render_failure_detail(reference)
        return

    answer = st.session_state.get(ANSWER_KEY)
    if isinstance(answer, GroundedAnswer):
        _render_answer(answer)


def _render_answer(answer: GroundedAnswer) -> None:
    """Render prose, structured findings, and the sources — in that order."""
    st.markdown(answer.text)

    if answer.findings:
        _render_findings(answer.findings)

    if answer.citations:
        with st.expander(f"Sources ({answer.citation_count})", expanded=True):
            for index, citation in enumerate(answer.citations, start=1):
                date = f" — {citation.date}" if citation.date else ""
                st.markdown(f"{index}. [{citation.label()}]({citation.url}){date}")
                if citation.snippet:
                    st.caption(citation.snippet[:280])
    else:
        st.caption(
            "No sources were returned for this answer. Treat it as unverified: the "
            "grounding search may not have run."
        )

    meta_bits = [bit for bit in (answer.preset, answer.model) if bit]
    if answer.usage.get("total_tokens"):
        meta_bits.append(f"{answer.usage['total_tokens']:,} tokens")
    if meta_bits:
        st.caption(" · ".join(meta_bits))
    _render_scope_note()


def _render_findings(findings: dict[str, object]) -> None:
    """Render the structured condition audit as readable sections.

    Order matters: the scalar assessments first, then the list-shaped detail, then the
    checklist mapping — which is the part that answers "is the app's committed framework
    still right for this condition?".
    """
    with st.expander("Structured audit", expanded=True):
        for key, label in (
            ("condition", "Condition"),
            ("summary", "Summary"),
            ("rating_criteria", "Rating criteria"),
            ("presumptive", "Presumptive basis"),
        ):
            value = findings.get(key)
            if isinstance(value, str) and value.strip():
                st.markdown(f"**{label}:** {value}")

        for key, label in (
            ("evidence_expectations", "Evidence the rater will expect"),
            ("lay_observations", "Observations a lay witness can competently make"),
            ("caveats", "Caveats"),
        ):
            values = findings.get(key)
            if isinstance(values, list) and values:
                st.markdown(f"**{label}**")
                for item in values:
                    st.markdown(f"- {item}")

        rows = findings.get("framework_findings")
        if isinstance(rows, list) and rows:
            st.markdown("**Checklist currency**")
            st.dataframe(rows, width="stretch", hide_index=True)


def _render_scope_note() -> None:
    """The same boundary the rest of the app states, repeated where research happens."""
    st.info(
        "Research output is web-grounded material to read and verify, not legal advice and "
        "not a finding of fact. Check any source before relying on it, and never submit a "
        "statement whose facts the witness has not personally confirmed."
    )
