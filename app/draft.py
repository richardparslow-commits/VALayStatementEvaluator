"""Draft pathway: create a factually grounded lay/witness statement from
medical records plus the witness's own observations."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from typing import Any

import json
import logging
import re
import time

from .agiloop_telemetry import track_feature_error
from .config import load_knowledge
from .drafting_service import error_extra, map_drafting_exception, validate_drafting_request
from .documents import (
    DRAFT_INTERNAL_MAX_CHARS,
    ExtractedDocument,
    MAX_OBSERVATIONS_CHARS,
)
from .llm import LLMClient, LLMError, LLMParseError, LLMService
from . import tracing
from .logging_config import PhaseTimer, get_request_id
from .profiler import phase_timer
from .pipeline_guard import check_pipeline_cancelled
from .aa_intake import care_observation_block
from .prompt_sanitize import GUARD_NOTE, sanitize_digest_text, sanitize_for_prompt, validate_witness_field

logger = logging.getLogger("app.draft")
from .medical_review import MedicalDigest, ProgressCallback, review_medical_records

# Feature: Condition-Specific Templates
FEATURE_ID = "02f0935a-ee5e-4083-88a2-10e11753ccc9"  # condition-specific-templates

REVIEW_MAX_CHARS = 16_000
# Budget for the grounding analysis handed to the drafting model.
# The bound is enforced by shrinking the data, never by slicing the JSON.
GROUNDING_PROMPT_MAX_CHARS = 25_000

# ---------------------------------------------------------------------------
# Feature: Witness medical credentials (hybrid statements).
#
# A witness who holds a medical credential (e.g. an RN spouse) can write a
# stronger "hybrid" statement: clinical descriptions of personally observed
# symptoms, functional/ADL assessment, medication effects. How far the
# statement may go depends on the credential — under 38 CFR § 3.159 a
# physician can render diagnoses and nexus opinions that nursing credentials
# cannot. The UI collects the credentials (app/views/draft_view.py), the
# helpers below turn them into prompt material, and every drafting prompt
# carries the scope rule for the witness's own credential level so the
# statement gains weight without overreaching lay/medical competence.
# ---------------------------------------------------------------------------

WITNESS_CREDENTIAL_LEVELS = (
    "None (lay witness)",
    "Nurse / clinician (RN, LPN, LVN, CNA)",
    "Advanced clinician (NP, PA, PA-C)",
    "Physician (MD, DO)",
    "Other licensed medical professional",
)

_CREDENTIAL_LAY_LEVEL = WITNESS_CREDENTIAL_LEVELS[0]

# Short scope summary embedded in the credentials block shown to every prompt.
_CREDENTIAL_SCOPE_SUMMARY = {
    "Nurse / clinician (RN, LPN, LVN, CNA)": (
        "May clinically describe personally observed symptoms, functional/ADL "
        "impact, medication effects, and flare-ups. Must not diagnose, interpret "
        "imaging/labs, or offer a causation (nexus) opinion."
    ),
    "Advanced clinician (NP, PA, PA-C)": (
        "May clinically describe personally observed symptoms, functional/ADL "
        "impact, and give a professional assessment within scope of practice. "
        "A causation (nexus) opinion only if clearly attributed to the witness's "
        "own professional judgment."
    ),
    "Physician (MD, DO)": (
        "May additionally provide a formal diagnosis and a medical-nexus opinion "
        "stated to the VA's 'at least as likely as not' standard, attributed to "
        "the witness's own qualifications and review."
    ),
    "Other licensed medical professional": (
        "May describe observed symptoms and functional impact through that "
        "professional lens, within the witness's scope of practice. Must not "
        "render a diagnosis or nexus opinion beyond that scope."
    ),
}

# Instruction paragraphs for the drafting/review SYSTEM prompts, per level.
_CREDENTIAL_SCOPE_DIRECTIVE = {
    "Nurse / clinician (RN, LPN, LVN, CNA)": (
        "The witness is a credentialed nurse or clinical caregiver ({level}). "
        "The statement may use precise clinical vocabulary to describe what the "
        "witness directly observed: symptom presentation (e.g., apneic episodes, "
        "antalgic gait, muscle guarding), functional impairment and activities of "
        "daily living, medication effects, and objective descriptions of flare-ups "
        "(frequency, duration, visible signs). It must NOT render a formal "
        "diagnosis, interpret imaging or lab results, or offer a medical-nexus or "
        "causation opinion. Every clinical statement must be anchored in what the "
        "witness personally observed."
    ),
    "Advanced clinician (NP, PA, PA-C)": (
        "The witness is an advanced practice clinician ({level}). In addition to "
        "clinical descriptions of personally observed symptoms and functional "
        "impact, the statement may include the witness's professional assessment "
        "of the veteran's condition within their scope of practice. A medical-nexus "
        "(causation) opinion should appear only if it is clearly attributed to the "
        "witness's own professional judgment and qualifications. Every statement "
        "must stay anchored in personal observation."
    ),
    "Physician (MD, DO)": (
        "The witness is a licensed physician ({level}). In addition to clinical "
        "observation, the statement may include a formal diagnosis and a "
        "medical-nexus opinion stated to the VA's 'at least as likely as not' "
        "standard, clearly attributed to the witness's own qualifications and "
        "professional review of the veteran's records and presentation. It must "
        "remain grounded in what the witness personally observed and reviewed — "
        "never fabricate examination findings, imaging interpretations, or record "
        "citations."
    ),
    "Other licensed medical professional": (
        "The witness holds a licensed medical credential ({level}). The statement "
        "may describe observed symptoms and functional impact through that "
        "professional lens, within the witness's scope of practice. It must NOT "
        "render a formal diagnosis or a medical-nexus/causation opinion beyond "
        "that scope."
    ),
}


def _credential_level(witness: dict[str, str]) -> str:
    """The witness's chosen credential level, or '' for a lay witness.

    Defensive against saved/queued payloads: only a known level counts, so a
    stale or hand-edited value degrades to lay treatment instead of putting an
    unknown scope rule into the prompts.
    """
    level = witness.get("credential_level", "") or ""
    return level if level in _CREDENTIAL_SCOPE_DIRECTIVE else ""


def witness_credentials_block(witness: dict[str, str]) -> str:
    """Render the witness's professional credentials as a prompt section.

    Empty for a lay witness, so prompts for the existing purely-lay path stay
    unchanged. Every field passes through ``sanitize_for_prompt`` — a pasted
    delimiter sequence or injection directive must not break the enclosing
    ``<<<``/``>>>`` block any more than observations can.
    """
    level = _credential_level(witness)
    if not level:
        return ""
    lines = [
        "WITNESS PROFESSIONAL CREDENTIALS:",
        f"- Credential level: {sanitize_for_prompt(level, max_chars=200)}",
    ]
    specialties = str(witness.get("medical_specialties", "") or "").strip()
    if specialties:
        lines.append(f"- Medical specialties: {sanitize_for_prompt(specialties, max_chars=500)}")
    detail = str(witness.get("credentials_detail", "") or "").strip()
    if detail:
        lines.append(f"- Credentials & certifications: {sanitize_for_prompt(detail, max_chars=500)}")
    relevance = str(witness.get("credential_relevance", "") or "").strip()
    if relevance:
        lines.append(
            "- How the professional experience relates to the observations: "
            + sanitize_for_prompt(relevance, max_chars=500)
        )
    lines.append(f"- Scope of this statement: {_CREDENTIAL_SCOPE_SUMMARY[level]}")
    return "\n".join(lines)


def witness_scope_directive(witness: dict[str, str]) -> str:
    """The scope instruction for the drafting/review system prompts.

    Empty for a lay witness — the drafting guide's global lay-competence rules
    already govern that path unchanged.
    """
    level = _credential_level(witness)
    if not level:
        return ""
    return _CREDENTIAL_SCOPE_DIRECTIVE[level].format(level=level)

GROUNDING_SYSTEM = """You are a veterans-claims evidence specialist preparing to draft a \
lay/witness statement (VA Form 21-10210 style). You must ground every available fact in the \
medical record digest and the witness's own observations, and honestly flag anything that \
cannot be verified. Never invent or embellish. Distinguish what the witness personally \
observed from what the records show. You also audit the witness's observations against the \
topic checklist you are given: decide which topics are applicable to this claim, which the \
observations already cover, and craft a specific follow-up question for every applicable \
topic they do not yet cover."""

GROUNDING_USER = """The witness provided the observations below. Compare them with the medical \
record digest and produce a grounding analysis.

Return JSON:
{{
  "supported_observations": [
    {{ "observation": "witness observation", "record_support": "matching record fact + source" }}
  ],
  "unverified_observations": [
    {{ "observation": "witness observation not in records", "action": "keep as lay evidence but witness should double-check before signing" }}
  ],
  "conflicts": [
    {{ "observation": "...", "record_fact": "...", "resolution_note": "draft only what the witness can truthfully support" }}
  ],
  "strengthening_questions": [
    "6-10 targeted questions for the witness, prioritizing applicable checklist topics the observations do not yet cover (hazards, before/after baseline, family impact, medication management) plus record facts they may be able to confirm personally"
  ],
  "suggested_inclusions": [
    {{ "fact": "record fact worth including if witness confirms", "source": "..." }}
  ],
  "topic_coverage": [
    {{ "topic": "checklist topic label (A-O)", "applicable": true | false, "covered": true | false, "prompt_for_witness": "specific question to elicit this topic if not covered, else empty string" }}
  ]
}}
One topic_coverage entry per checklist topic (A through O), in checklist order. Never invent \
coverage: mark a topic covered only if the observations genuinely address it.

CLAIMED CONDITION: {condition}
CLAIM TYPE: {claim_type}
WITNESS ROLE/RELATIONSHIP: {relationship}
{credentials_block}
{care_block}
WITNESS OBSERVATIONS:
<<<
{observations}
>>>

MEDICAL RECORD DIGEST (JSON):
<<<
{digest}
>>>

TOPIC CHECKLIST:
<<<
{checklist}
>>>

{guard_note}"""

DRAFT_SYSTEM_TEMPLATE = """You are drafting a VA lay/witness statement for submission on \
VA Form 21-10210. Follow this drafting guide EXACTLY. Write only facts grounded in the \
provided grounding analysis and witness observations. Use bracketed placeholders like \
"[Confirm: ...]" anywhere the witness must verify a record-derived fact, and \
"[Witness to add: ...]" anywhere an applicable checklist topic is not yet supplied by the \
observations. Never fabricate dates, events, or details. Where applicable to this claim, \
translate symptoms into concrete observable dangers (e.g., memory loss -> double-dosing risk, \
stove left on) and make clear when the caregiver's help is necessary for safety, not merely \
convenient. Include a before/after comparison and symptom progression whenever the observations \
support them. The record evidence is presented in chronological order — within the guide's \
required sections, tell what was observed in that order (earliest first) so the statement \
reads as a faithful progression of the condition over time, not a list of isolated incidents.

When the witness holds a professional credential, apply the witness-specific scope rule \
in the CREDENTIAL SCOPE section instead of the blanket lay-evidence restrictions in the \
guide, exactly as far as the rule allows — no farther. When the witness is uncredentialed, \
follow the guide's lay-evidence rules unchanged.

{guide}

TOPIC CHECKLIST — organize the statement so that every applicable topic the observations \
support is covered (there is no page limit; be as detailed as the material allows):

{checklist}

{credential_scope}"""

DRAFT_USER = """Draft the lay/witness statement now.

WITNESS INFORMATION:
- Name: {witness_name}
- Relationship to veteran: {relationship}
- Known veteran since / for: {known_since}
- Opportunity to observe: {contact_frequency}
- Veteran's name: {veteran_name}
- Claimed condition: {condition}
- Claim type: {claim_type}
- Witnessed the in-service event personally: {witnessed_event}

{credentials_block}
{care_block}
WITNESS OBSERVATIONS:
<<<
{observations}
>>>

GROUNDING ANALYSIS (JSON):
<<<
{grounding}
>>>

RECORD SUMMARY:
<<<
{digest_summary}
>>>

{guard_note}

Output the statement ONLY (no meta commentary), in first person, following the guide's
structure including the certification closing."""

REVIEW_SYSTEM = """You are quality-checking a drafted VA lay statement against the drafting \
rubric and the topic checklist. Identify concrete fixes: vagueness, missing specifics, \
lay-competence violations, missing structure elements, ungrounded facts, or applicable topics \
from the checklist that the draft fails to cover. Then return the IMPROVED full statement. \
Where an applicable topic lacks any supplied material, insert a "[Witness to add: ...]" \
placeholder rather than inventing content. Preserve the draft's chronological narrative \
order (earliest events first) — do not reorder the events it recounts. When a witness scope \
rule is provided below, keep every credential-based assertion within that scope — do not \
expand claims into diagnoses, nexus opinions, or clinical interpretations the scope rule \
does not allow, and strip any that exceed it."""

REVIEW_USER = """Improve this draft statement. Preserve all bracketed placeholders \
(including every [Confirm: ...] and [Witness to add: ...]) and all grounded facts; \
do not add new facts. Keep all section headings. Copy the certification and the \
entire signature/contact closing verbatim. Return the full statement, never a \
summary or an excerpt. Return JSON:
{{
  "issues_found": ["issue 1", "..."],
  "improved_statement": "<the full improved statement text>"
}}

DRAFT:
<<<
{draft}
>>>

DRAFTING GUIDE FOR REFERENCE:
<<<
{guide}
>>>

TOPIC CHECKLIST FOR REFERENCE:
<<<
{checklist}
>>>

{credential_scope}

{guard_note}"""


@dataclass
class DraftResult:
    grounding: dict[str, Any] = field(default_factory=dict)
    draft: str = ""
    final_statement: str = ""
    review_issues: list[str] = field(default_factory=list)
    digest: MedicalDigest | None = None
    # Truncation audit for observations
    input_chars: int = 0
    truncated_chars: int = 0
    truncation_warning: str = ""
    # Complete source evidence preserved independently of prompt budgets.
    # Each dict holds {"filename": str, "kind": "page"|"block", "page": int,
    # "text": str} — the raw record pages that fed the digest extraction.
    evidence_source: list[dict] = field(default_factory=list)

    @property
    def output_statement(self) -> str:
        return self.final_statement or self.draft


def run_draft(
    llm: LLMService,
    records: list[ExtractedDocument],
    witness: dict[str, str],
    observations: str,
    condition: str,
    claim_type: str,
    progress: ProgressCallback | None = None,
) -> DraftResult:
    """Execute the full drafting pipeline."""
    rid = get_request_id() or "-"
    t0 = time.perf_counter()
    pages = sum(len(d.pages) for d in records)
    logger.info(
        "draft start pages=%d observations_chars=%d condition=%s",
        pages,
        len(observations),
        condition[:60] if condition else "-",
        extra={"request_id": rid, "phase": "draft_pipeline", "status": "start"},
    )
    # Root span for the run — see run_evaluation for the mirrored comment.
    with tracing.run_span("draft", files=len(records), pages=pages, chars=len(observations)):
        try:
            validate_drafting_request(
                observations=observations,
                condition=condition,
                claim_type=claim_type,
                witness=witness,
            )
            result = _run_draft(
                llm, records, witness, observations, condition, claim_type, progress
            )
            duration_ms = int((time.perf_counter() - t0) * 1000)
            logger.info(
                "draft done duration_ms=%d draft_chars=%d grounding_items=%d",
                duration_ms,
                len(result.output_statement),
                len(result.grounding) if isinstance(result.grounding, dict) else 0,
                extra={
                    "request_id": rid,
                    "phase": "draft_pipeline",
                    "status": "ok",
                    "duration_ms": duration_ms,
                },
            )
            return result
        except Exception as exc:  # noqa: BLE001 - feature-error boundary
            mapped = map_drafting_exception(exc, request_id=rid, phase="draft_pipeline")
            duration_ms = int((time.perf_counter() - t0) * 1000)
            logger.error(
                "draft error duration_ms=%d error=%s",
                duration_ms,
                mapped.diagnostics,
                exc_info=exc,
                extra={
                    "request_id": rid,
                    "phase": "draft_pipeline",
                    "status": "error",
                    "duration_ms": duration_ms,
                    "error_class": type(exc).__name__,
                    **error_extra(mapped),
                },
            )
            track_feature_error(FEATURE_ID, exc)
            raise mapped from exc


def _truncate_for_prompt(text: str, limit: int = DRAFT_INTERNAL_MAX_CHARS) -> tuple[str, int]:
    if len(text) <= limit:
        return text, 0
    return text[:limit], len(text) - limit


def _normalized_text(text: str) -> str:
    return " ".join(text.split())


def _section_headings(text: str, *, with_content: bool = False) -> Counter[str]:
    """Recognize explicit sections without trying to infer meaning from prose."""
    headings: Counter[str] = Counter()
    pending_heading = ""
    plain_labels = {
        "header", "introduction", "introduction & credentials of observation",
        "in-service event / onset", "observed symptoms and their progression",
        "observed symptoms", "functional impact", "continuity", "continuity statement",
        "closing", "certification", "closing & certification", "signature",
    }
    for line in text.splitlines():
        line = line.strip()
        label = re.sub(r"^(?:#{1,6}\s+|\d+[.)]\s+)", "", line)
        label = _normalized_text(label.strip("*_#: ")).casefold()
        if label and (
            line.startswith("#") or (line.startswith("**") and line.rstrip(":").endswith("**"))
            or (re.match(r"^\d+[.)]\s+", line) and len(label) <= 120)
            or label in plain_labels
        ):
            pending_heading = label
            if not with_content:
                headings[label] += 1
        elif with_content and pending_heading and line:
            headings[pending_heading] += 1
            pending_heading = ""
    return headings


def _review_rejection_reason(original: str, improved: Any) -> str:
    """Conservative structural checks, not a claim of semantic fact verification."""
    if not isinstance(improved, str) or not improved.strip():
        return "the reviewer did not return a statement"
    if len(improved.strip()) <= max(200, int(len(original) * 0.4)):
        return "the proposed statement was too short to adopt safely"
    original_placeholders = Counter(
        _normalized_text(value) for value in re.findall(r"\[[^\[\]]+\]", original)
    )
    improved_placeholders = Counter(
        _normalized_text(value) for value in re.findall(r"\[[^\[\]]+\]", improved)
    )
    if original_placeholders - improved_placeholders:
        return "the proposed statement removed or changed a confirmation or other placeholder"
    if _section_headings(original) - _section_headings(improved):
        return "the proposed statement removed or changed a section heading"
    if _section_headings(original, with_content=True) - _section_headings(improved, with_content=True):
        return "the proposed statement left a previously populated section empty"
    closing = re.search(
        r"\bI\s+(?:(?:hereby|solemnly)\s+)?(?:certify|declare|affirm|attest)\b",
        original, re.IGNORECASE,
    )
    if closing and not _normalized_text(improved).endswith(
        _normalized_text(original[closing.start():])
    ):
        return "the proposed statement did not preserve the certification and signature closing"
    return ""


def _pages_to_source(records: list[ExtractedDocument]) -> list[dict]:
    """Extract raw source pages into a compact, serialisable list.

    Each entry is ``{"filename": str, "kind": "page"|"block", "page": int,
    "text": str}``. This store is preserved in the result independently of
    prompt budgets — summaries and selected facts are derived views of it.
    """
    pages: list[dict] = []
    for doc in records:
        for page in doc.pages:
            pages.append({
                "filename": page.filename,
                "kind": page.kind,
                "page": page.page,
                "text": page.text,
            })
    return pages


_GROUNDING_OBJECT_LISTS = (
    "supported_observations",
    "unverified_observations",
    "conflicts",
    "suggested_inclusions",
    "topic_coverage",
)


def _normalize_grounding(raw: Any) -> dict[str, Any]:
    """Validate the grounding analysis before it feeds the draft, UI, or store.

    ``chat_json`` guarantees parseable JSON, not the object-of-lists-of-objects
    the grounding prompt asks for: a model can return an array, a bare string,
    or rows that are not objects. Every downstream reader (the draft prompt,
    ``grounding_markdown``, the job payload, follow-up questions) reads these
    entries with ``.get``, so a shape mismatch is rejected here as a parse
    failure — a statement cannot be grounded in an analysis that is not there —
    instead of surfacing later as an AttributeError once the digest work is done.
    """
    if not isinstance(raw, dict):
        raise LLMParseError("Grounding analysis is incomplete: expected a JSON object.")
    normalized: dict[str, Any] = dict(raw)
    for field_name in _GROUNDING_OBJECT_LISTS:
        value = normalized.get(field_name)
        if value is None:
            normalized[field_name] = []
            continue
        if not isinstance(value, list):
            raise LLMParseError(f"Grounding analysis is incomplete: {field_name} must be a list.")
        if any(not isinstance(item, dict) for item in value):
            raise LLMParseError(
                f"Grounding analysis is incomplete: each entry in {field_name} must be an object."
            )
    questions = normalized.get("strengthening_questions")
    if questions is None:
        normalized["strengthening_questions"] = []
    elif not isinstance(questions, list) or any(not isinstance(item, str) for item in questions):
        raise LLMParseError(
            "Grounding analysis is incomplete: strengthening_questions must be a list of strings."
        )
    return normalized


def _run_draft(
    llm: LLMService,
    records: list[ExtractedDocument],
    witness: dict[str, str],
    observations: str,
    condition: str,
    claim_type: str,
    progress: ProgressCallback | None,
) -> DraftResult:
    result = DraftResult()
    result.input_chars = len(observations)
    # Preserve complete source evidence — the raw record pages that
    # produced every digest fact. This store lives independently of
    # prompt budgets so saved results always carry full provenance.
    result.evidence_source = _pages_to_source(records)
    obs_for_prompt, removed = _truncate_for_prompt(observations, DRAFT_INTERNAL_MAX_CHARS)
    result.truncated_chars = removed
    if removed:
        result.truncation_warning = (
            f"Observations were {result.input_chars:,} characters — "
            f"{removed:,} characters beyond the {DRAFT_INTERNAL_MAX_CHARS:,} internal prompt limit "
            f"were truncated and not grounded. Details at the end may have been missed. "
            f"Shorten or split the observations and re-run."
        )
        if result.input_chars > MAX_OBSERVATIONS_CHARS:
            result.truncation_warning = (
                f"Observations were {result.input_chars:,} characters — "
                f"{result.input_chars - MAX_OBSERVATIONS_CHARS:,} over the {MAX_OBSERVATIONS_CHARS:,} "
                f"recommended limit. {removed:,} characters were truncated for the model prompts; "
                f"details at the end may have been missed. Shorten or split and re-run."
            )

    def report(frac: float, msg: str) -> None:
        check_pipeline_cancelled()
        if progress:
            progress(frac, msg)

    rid = get_request_id() or "-"
    with (
        tracing.phase_span("records:review", files=len(records)),
        PhaseTimer(logger, "records:review", request_id=rid, chunks=len(records)),
    ):
        with phase_timer("records:review"):
            report(0.02, "Step 1/4 — Exhaustive review of medical records…")
            try:
                result.digest = review_medical_records(
                    llm, records, progress=lambda f, m: progress((0.02 + f * 0.45), m) if progress else None
                )
            except Exception as exc:  # noqa: BLE001
                raise map_drafting_exception(exc, request_id=rid, phase="records:review") from exc

    with tracing.phase_span("grounding"), PhaseTimer(logger, "grounding", request_id=rid):
        with phase_timer("grounding"):
            report(0.5, "Step 2/4 — Grounding witness observations against the records and topic checklist…")
            grounding_query = f"{condition} {obs_for_prompt}"
            try:
                raw_grounding = llm.chat_json(
                    GROUNDING_SYSTEM,
                    GROUNDING_USER.format(
                        condition=sanitize_for_prompt(condition, max_chars=500),
                        claim_type=sanitize_for_prompt(claim_type, max_chars=500),
                        relationship=sanitize_for_prompt(witness.get("relationship", "not specified"), max_chars=500),
                        credentials_block=witness_credentials_block(witness),
                        care_block=care_observation_block(witness),
                        observations=sanitize_for_prompt(obs_for_prompt, max_chars=DRAFT_INTERNAL_MAX_CHARS),
                        # Chronological presentation: the drafting narrative
                        # follows the records' timeline, not keyword order.
                        digest=sanitize_digest_text(
                            result.digest.relevant_facts_text(grounding_query, max_facts=150, sort_dates=True),
                            max_chars=120_000,
                        ),
                        checklist=load_knowledge("topic_checklist.md"),
                        guard_note=GUARD_NOTE,
                    ),
                    phase="grounding",
                )
                result.grounding = _normalize_grounding(raw_grounding)
            except Exception as exc:  # noqa: BLE001
                raise map_drafting_exception(exc, request_id=rid, phase="grounding") from exc

    # Self-contained scope block: heading included only when a rule exists, so
    # a lay witness's prompts carry nothing extra at all.
    scope_block = witness_scope_directive(witness)
    if scope_block:
        scope_block = f"CREDENTIAL SCOPE:\n{scope_block}"

    with tracing.phase_span("draft"), PhaseTimer(logger, "draft", request_id=rid):
        with phase_timer("draft"):
            report(0.68, "Step 3/4 — Drafting the statement…")
            try:
                result.draft = llm.chat(
                    DRAFT_SYSTEM_TEMPLATE.format(
                        guide=load_knowledge("drafting_guide.md"),
                        checklist=load_knowledge("topic_checklist.md"),
                        credential_scope=scope_block,
                    ),
                    DRAFT_USER.format(
                        witness_name=sanitize_for_prompt(witness.get("name", "[Witness Name]"), max_chars=500),
                        relationship=sanitize_for_prompt(witness.get("relationship", "[relationship]"), max_chars=500),
                        known_since=sanitize_for_prompt(witness.get("known_since", "[how long known]"), max_chars=500),
                        contact_frequency=sanitize_for_prompt(witness.get("contact_frequency", "[frequency of contact]"), max_chars=500),
                        veteran_name=sanitize_for_prompt(witness.get("veteran_name", "[Veteran Name]"), max_chars=500),
                        condition=sanitize_for_prompt(condition, max_chars=500),
                        claim_type=sanitize_for_prompt(claim_type, max_chars=500),
                        witnessed_event=sanitize_for_prompt(witness.get("witnessed_event", "unknown"), max_chars=500),
                        credentials_block=witness_credentials_block(witness),
                        care_block=care_observation_block(witness),
                        observations=sanitize_for_prompt(obs_for_prompt, max_chars=DRAFT_INTERNAL_MAX_CHARS),
                        grounding=_grounding_for_prompt(result.grounding),
                        digest_summary=sanitize_digest_text(result.digest.summary or "(no summary)", max_chars=20_000),
                        guard_note=GUARD_NOTE,
                    ),
                    max_tokens=6000,
                    phase="draft",
                )
            except Exception as exc:  # noqa: BLE001
                raise map_drafting_exception(exc, request_id=rid, phase="draft") from exc

    with tracing.phase_span("review"), PhaseTimer(logger, "review", request_id=rid):
        with phase_timer("review"):
            report(0.85, "Step 4/4 — Self-review and improvement pass…")
            # Escaping can expand text (e.g. code fences). Sanitize without
            # truncation, then check the actual review input against its budget.
            review_draft = sanitize_for_prompt(result.draft, max_chars=2 * len(result.draft) + 1)
            if len(review_draft) > REVIEW_MAX_CHARS:
                result.review_issues = [
                    f"Self-review was skipped because the full statement exceeds the "
                    f"{REVIEW_MAX_CHARS:,}-character review limit. The complete original "
                    "draft is preserved; review it manually before signing."
                ]
                report(1.0, "Draft complete (full original preserved; self-review skipped).")
                return result
            try:
                review = llm.chat_json(
                    REVIEW_SYSTEM,
                    REVIEW_USER.format(
                        draft=review_draft,
                        guide=load_knowledge("drafting_guide.md")[:6000],
                        checklist=load_knowledge("topic_checklist.md")[:6000],
                        credential_scope=scope_block,
                        guard_note=GUARD_NOTE,
                    ),
                    phase="review",
                )
            except Exception as exc:  # noqa: BLE001
                # The review pass is cosmetic — grounding and the draft are
                # already complete. A filter/retry failure here must not
                # discard the finished draft, so keep it and note the miss.
                logger.warning(
                    "review pass unavailable — keeping unreviewed draft error=%s",
                    f"{type(exc).__name__}: {exc}",
                    extra={"request_id": rid, "phase": "review", "status": "error", "error_class": type(exc).__name__},
                )
                result.review_issues = [
                    "Self-review pass was skipped (the model call failed) — the statement below is the "
                    "unreviewed draft. Re-run to get the polished version."
                ]
                report(1.0, "Draft complete (self-review skipped — model call failed).")
                return result
    if not isinstance(review, dict):
        review = {}
    issues = review.get("issues_found", [])
    result.review_issues = [issue for issue in issues if isinstance(issue, str)] if isinstance(issues, list) else []
    improved = review.get("improved_statement", "")
    rejection = _review_rejection_reason(result.draft, improved)
    if rejection:
        result.review_issues.append(
            f"Self-review was not applied because {rejection}. The complete original "
            "draft is preserved; review it manually before signing."
        )
    else:
        result.final_statement = improved.strip()

    report(1.0, "Draft complete.")
    return result


_GROUNDING_TRIMMED_NOTE = (
    "Shortened to fit the drafting prompt budget; trailing rows and long fields were dropped."
)

# String caps tried in order; within each, the list cap is binary-searched so
# the analysis retained is as large as the budget allows.
_GROUNDING_STRING_CAPS = (2_000, 1_000, 400, 120, 60)


def _cap_json_value(value: Any, per_string: int, per_list: int) -> Any:
    """Rebuild a JSON value with long strings capped and lists shortened from the end."""
    if isinstance(value, str):
        if len(value) <= per_string:
            return value
        return value[:per_string] + "… [shortened for prompt budget]"
    if isinstance(value, list):
        return [_cap_json_value(item, per_string, per_list) for item in value[:per_list]]
    if isinstance(value, dict):
        return {key: _cap_json_value(item, per_string, per_list) for key, item in value.items()}
    return value


def _longest_list(value: Any) -> int:
    """Length of the longest list anywhere in a JSON value (0 when there is none)."""
    if isinstance(value, list):
        nested = max((_longest_list(item) for item in value), default=0)
        return max(len(value), nested)
    if isinstance(value, dict):
        return max((_longest_list(item) for item in value.values()), default=0)
    return 0


def _trimmed_grounding(grounding: dict[str, Any], per_string: int, per_list: int) -> dict[str, Any]:
    data: dict[str, Any] = _cap_json_value(grounding, per_string, per_list)
    if data != grounding:
        data = {**data, "_note": _GROUNDING_TRIMMED_NOTE}
    return data


def _grounding_text(data: dict[str, Any]) -> str:
    # ``sanitize_for_prompt`` enforces its bound by slicing, so give it a bound
    # that cannot truncate: delimiter escaping grows text at most 5/3 (a fence
    # is 3 chars, its replacement 5), so ``2 * len + 1`` always suffices. The
    # real budget is enforced on the sanitized result, as the review pass does.
    raw = json.dumps(data, indent=1)
    return sanitize_for_prompt(raw, max_chars=2 * len(raw) + 1)


def _fits_grounding_budget(grounding: dict[str, Any], per_string: int, per_list: int) -> bool:
    trimmed = _trimmed_grounding(grounding, per_string, per_list)
    return len(_grounding_text(trimmed)) <= GROUNDING_PROMPT_MAX_CHARS


def _largest_fitting_list_cap(grounding: dict[str, Any], per_string: int) -> int | None:
    """Largest per-list row cap that fits, or ``None`` when even one row does not."""
    if not _fits_grounding_budget(grounding, per_string, 1):
        return None
    lo, hi = 1, max(1, _longest_list(grounding))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _fits_grounding_budget(grounding, per_string, mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def _grounding_for_prompt(grounding: dict[str, Any]) -> str:
    """Render the grounding analysis for the drafting prompt as complete JSON.

    The prompt budget is met by shrinking the data — long strings capped,
    trailing rows shed, tighter string caps last — never by slicing the
    serialized document, which would hand the drafting model unparseable JSON
    mid-string. Within each string cap the row cap is binary-searched, so the
    model receives as much of the analysis as the budget allows. Output is
    always valid JSON, and carries a note whenever it was trimmed.
    """
    text = _grounding_text(grounding)
    if len(text) <= GROUNDING_PROMPT_MAX_CHARS:
        return text
    for per_string in _GROUNDING_STRING_CAPS:
        per_list = _largest_fitting_list_cap(grounding, per_string)
        if per_list is not None:
            return _grounding_text(_trimmed_grounding(grounding, per_string, per_list))
    # The tightest cap fits any analysis the normalizer accepts; this return
    # only guards against a future schema change making the last cap too big.
    return _grounding_text(_trimmed_grounding(grounding, _GROUNDING_STRING_CAPS[-1], 1))


def _grounding_rows(grounding: dict[str, Any], field_name: str) -> list[dict]:
    """Rows a renderer can read; misshapen saved results are skipped, not fatal."""
    value = grounding.get(field_name)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _grounding_strings(grounding: dict[str, Any], field_name: str) -> list[str]:
    value = grounding.get(field_name)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def grounding_markdown(result: DraftResult) -> str:
    """Render the grounding analysis as readable markdown for the UI."""
    lines: list[str] = []
    if result.truncation_warning:
        lines.append(f"> ⚠️ **Truncated observations:** {result.truncation_warning}")
        lines.append("")
    grounding = result.grounding if isinstance(result.grounding, dict) else {}
    supported = _grounding_rows(grounding, "supported_observations")
    if supported:
        lines.append("### ✅ Observations corroborated by the records")
        for item in supported:
            lines.append(f"- **{item.get('observation', '')}**")
            lines.append(f"  - Record support: {item.get('record_support', '')}")
        lines.append("")
    unverified = _grounding_rows(grounding, "unverified_observations")
    if unverified:
        lines.append("### ⚪ Observations not found in records (still legitimate lay evidence)")
        for item in unverified:
            lines.append(f"- {item.get('observation', '')} — _{item.get('action', '')}_")
        lines.append("")
    conflicts = _grounding_rows(grounding, "conflicts")
    if conflicts:
        lines.append("### ⚠️ Conflicts with the records — resolve before signing")
        for item in conflicts:
            lines.append(f"- Observation: {item.get('observation', '')}")
            lines.append(f"  - Records show: {item.get('record_fact', '')}")
            lines.append(f"  - Guidance: {item.get('resolution_note', '')}")
        lines.append("")
    topics = _grounding_rows(grounding, "topic_coverage")
    if topics:
        covered = [t for t in topics if t.get("applicable") and t.get("covered")]
        missing = [t for t in topics if t.get("applicable") and not t.get("covered")]
        lines.append("### 🧭 Topic coverage — what the observations do and do not address")
        if covered:
            lines.append("**Covered by the witness's observations:**")
            for t in covered:
                lines.append(f"- {t.get('topic', '')}")
        if missing:
            lines.append("")
            lines.append(
                "**Applicable topics still missing — the witness should answer these if true** "
                "(details below, and they become `[Witness to add: ...]` placeholders in the draft):"
            )
            for t in missing:
                prompt = t.get("prompt_for_witness") or "Describe what you have observed."
                lines.append(f"- **{t.get('topic', '')}** — {prompt}")
        lines.append("")
    questions = _grounding_strings(grounding, "strengthening_questions")
    if questions:
        lines.append("### ❓ Answer these to strengthen the statement (records suggest you may know)")
        for question in questions:
            lines.append(f"- {question}")
        lines.append("")
    return "\n".join(lines) or "_No grounding details produced._"
