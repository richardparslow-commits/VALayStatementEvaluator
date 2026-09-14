"""Draft pathway: create a factually grounded lay/witness statement from
medical records plus the witness's own observations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import logging
import time

from .agiloop_telemetry import track_feature_error
from .config import load_knowledge
from .documents import (
    DRAFT_INTERNAL_MAX_CHARS,
    ExtractedDocument,
    MAX_OBSERVATIONS_CHARS,
)
from .llm import LLMClient
from .logging_config import PhaseTimer, get_request_id

logger = logging.getLogger("app.draft")
from .medical_review import MedicalDigest, ProgressCallback, review_medical_records

# Feature: Condition-Specific Templates
FEATURE_ID = "02f0935a-ee5e-4083-88a2-10e11753ccc9"  # condition-specific-templates

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
    {{ "topic": "checklist topic label (A-L)", "applicable": true | false, "covered": true | false, "prompt_for_witness": "specific question to elicit this topic if not covered, else empty string" }}
  ]
}}
One topic_coverage entry per checklist topic (A through L), in checklist order. Never invent \
coverage: mark a topic covered only if the observations genuinely address it.

CLAIMED CONDITION: {condition}
CLAIM TYPE: {claim_type}
WITNESS ROLE/RELATIONSHIP: {relationship}
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
>>>"""

DRAFT_SYSTEM_TEMPLATE = """You are drafting a VA lay/witness statement for submission on \
VA Form 21-10210. Follow this drafting guide EXACTLY. Write only facts grounded in the \
provided grounding analysis and witness observations. Use bracketed placeholders like \
"[Confirm: ...]" anywhere the witness must verify a record-derived fact, and \
"[Witness to add: ...]" anywhere an applicable checklist topic is not yet supplied by the \
observations. Never fabricate dates, events, or details. Where applicable to this claim, \
translate symptoms into concrete observable dangers (e.g., memory loss -> double-dosing risk, \
stove left on) and make clear when the caregiver's help is necessary for safety, not merely \
convenient. Include a before/after comparison and symptom progression whenever the observations \
support them.

{guide}

TOPIC CHECKLIST — organize the statement so that every applicable topic the observations \
support is covered (there is no page limit; be as detailed as the material allows):

{checklist}"""

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

Output the statement ONLY (no meta commentary), in first person, following the guide's
structure including the certification closing."""

REVIEW_SYSTEM = """You are quality-checking a drafted VA lay statement against the drafting \
rubric and the topic checklist. Identify concrete fixes: vagueness, missing specifics, \
lay-competence violations, missing structure elements, ungrounded facts, or applicable topics \
from the checklist that the draft fails to cover. Then return the IMPROVED full statement. \
Where an applicable topic lacks any supplied material, insert a "[Witness to add: ...]" \
placeholder rather than inventing content."""

REVIEW_USER = """Improve this draft statement. Preserve all bracketed [Confirm: ...] \
placeholders and all grounded facts; do not add new facts. Return JSON:
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
>>>"""


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

    @property
    def output_statement(self) -> str:
        return self.final_statement or self.draft


def run_draft(
    llm: LLMClient,
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
    logger.info(
        "draft start pages=%d observations_chars=%d condition=%s",
        sum(len(d.pages) for d in records),
        len(observations),
        condition[:60] if condition else "-",
        extra={"request_id": rid, "phase": "draft_pipeline", "status": "start"},
    )
    try:
        result = _run_draft(llm, records, witness, observations, condition, claim_type, progress)
        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "draft done duration_ms=%d draft_chars=%d grounding_items=%d",
            duration_ms,
            len(result.output_statement),
            len(result.grounding) if isinstance(result.grounding, dict) else 0,
            extra={"request_id": rid, "phase": "draft_pipeline", "status": "ok", "duration_ms": duration_ms},
        )
        return result
    except Exception as exc:  # noqa: BLE001 - feature-error boundary
        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.error(
            "draft error duration_ms=%d error=%s",
            duration_ms,
            f"{type(exc).__name__}: {exc}",
            exc_info=exc,
            extra={"request_id": rid, "phase": "draft_pipeline", "status": "error", "duration_ms": duration_ms, "error_class": type(exc).__name__},
        )
        track_feature_error(FEATURE_ID, exc)
        raise


def _truncate_for_prompt(text: str, limit: int = DRAFT_INTERNAL_MAX_CHARS) -> tuple[str, int]:
    if len(text) <= limit:
        return text, 0
    return text[:limit], len(text) - limit


def _run_draft(
    llm: LLMClient,
    records: list[ExtractedDocument],
    witness: dict[str, str],
    observations: str,
    condition: str,
    claim_type: str,
    progress: ProgressCallback | None,
) -> DraftResult:
    result = DraftResult()
    result.input_chars = len(observations)
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
        if progress:
            progress(frac, msg)

    rid = get_request_id() or "-"
    with PhaseTimer(logger, "records:review", request_id=rid, chunks=len(records)):
        report(0.02, "Step 1/4 — Exhaustive review of medical records…")
        result.digest = review_medical_records(
            llm, records, progress=lambda f, m: progress((0.02 + f * 0.45), m) if progress else None
        )

    with PhaseTimer(logger, "grounding", request_id=rid):
        report(0.5, "Step 2/4 — Grounding witness observations against the records and topic checklist…")
        grounding_query = f"{condition} {obs_for_prompt}"
        result.grounding = llm.chat_json(
            GROUNDING_SYSTEM,
            GROUNDING_USER.format(
                condition=condition,
                claim_type=claim_type,
                relationship=witness.get("relationship", "not specified"),
                observations=obs_for_prompt,
                digest=result.digest.relevant_facts_text(grounding_query, max_facts=150),
                checklist=load_knowledge("topic_checklist.md"),
            ),
            phase="grounding",
        )

    with PhaseTimer(logger, "draft", request_id=rid):
        report(0.68, "Step 3/4 — Drafting the statement…")
        result.draft = llm.chat(
            DRAFT_SYSTEM_TEMPLATE.format(
                guide=load_knowledge("drafting_guide.md"),
                checklist=load_knowledge("topic_checklist.md"),
            ),
            DRAFT_USER.format(
                witness_name=witness.get("name", "[Witness Name]"),
                relationship=witness.get("relationship", "[relationship]"),
                known_since=witness.get("known_since", "[how long known]"),
                contact_frequency=witness.get("contact_frequency", "[frequency of contact]"),
                veteran_name=witness.get("veteran_name", "[Veteran Name]"),
                condition=condition,
                claim_type=claim_type,
                witnessed_event=witness.get("witnessed_event", "unknown"),
                observations=obs_for_prompt,
                grounding=_json_dumps(result.grounding),
                digest_summary=result.digest.summary or "(no summary)",
            ),
            max_tokens=6000,
            phase="draft",
        )

    with PhaseTimer(logger, "review", request_id=rid):
        report(0.85, "Step 4/4 — Self-review and improvement pass…")
        review = llm.chat_json(
            REVIEW_SYSTEM,
            REVIEW_USER.format(
                draft=result.draft[:16000],
                guide=load_knowledge("drafting_guide.md")[:6000],
                checklist=load_knowledge("topic_checklist.md")[:6000],
            ),
            phase="review",
        )
    result.review_issues = review.get("issues_found", [])
    improved = review.get("improved_statement", "")
    if improved and len(improved) > max(200, int(len(result.draft) * 0.4)):
        result.final_statement = improved.strip()

    report(1.0, "Draft complete.")
    return result


def _json_dumps(data: Any) -> str:
    import json

    try:
        return json.dumps(data, indent=1)[:20000]
    except (TypeError, ValueError):
        return str(data)[:20000]


def grounding_markdown(result: DraftResult) -> str:
    """Render the grounding analysis as readable markdown for the UI."""
    lines: list[str] = []
    if result.truncation_warning:
        lines.append(f"> ⚠️ **Truncated observations:** {result.truncation_warning}")
        lines.append("")
    supported = result.grounding.get("supported_observations", [])
    if supported:
        lines.append("### ✅ Observations corroborated by the records")
        for item in supported:
            lines.append(f"- **{item.get('observation', '')}**")
            lines.append(f"  - Record support: {item.get('record_support', '')}")
        lines.append("")
    unverified = result.grounding.get("unverified_observations", [])
    if unverified:
        lines.append("### ⚪ Observations not found in records (still legitimate lay evidence)")
        for item in unverified:
            lines.append(f"- {item.get('observation', '')} — _{item.get('action', '')}_")
        lines.append("")
    conflicts = result.grounding.get("conflicts", [])
    if conflicts:
        lines.append("### ⚠️ Conflicts with the records — resolve before signing")
        for item in conflicts:
            lines.append(f"- Observation: {item.get('observation', '')}")
            lines.append(f"  - Records show: {item.get('record_fact', '')}")
            lines.append(f"  - Guidance: {item.get('resolution_note', '')}")
        lines.append("")
    topics = result.grounding.get("topic_coverage", [])
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
    questions = result.grounding.get("strengthening_questions", [])
    if questions:
        lines.append("### ❓ Answer these to strengthen the statement (records suggest you may know)")
        for question in questions:
            lines.append(f"- {question}")
        lines.append("")
    return "\n".join(lines) or "_No grounding details produced._"
