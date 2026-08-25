"""Draft pathway: create a factually grounded lay/witness statement from
medical records plus the witness's own observations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import load_knowledge
from .documents import ExtractedDocument
from .llm import LLMClient
from .medical_review import MedicalDigest, ProgressCallback, review_medical_records

GROUNDING_SYSTEM = """You are a veterans-claims evidence specialist preparing to draft a \
lay/witness statement (VA Form 21-10210 style). You must ground every available fact in the \
medical record digest and the witness's own observations, and honestly flag anything that \
cannot be verified. Never invent or embellish. Distinguish what the witness personally \
observed from what the records show."""

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
    "5-8 targeted questions for the witness, based on record facts they may be able to confirm personally (treatments they witnessed, appointments they drove to, symptoms at home, work impact they saw)"
  ],
  "suggested_inclusions": [
    {{ "fact": "record fact worth including if witness confirms", "source": "..." }}
  ]
}}

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
>>>"""

DRAFT_SYSTEM_TEMPLATE = """You are drafting a VA lay/witness statement for submission on \
VA Form 21-10210. Follow this drafting guide EXACTLY. Write only facts grounded in the \
provided grounding analysis and witness observations. Use bracketed placeholders like \
"[Confirm: ...]" anywhere the witness must verify a record-derived fact. Never fabricate \
dates, events, or details.

{guide}"""

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
rubric. Identify concrete fixes: vagueness, missing specifics, lay-competence violations, \
missing structure elements, or ungrounded facts. Then return the IMPROVED full statement."""

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
>>>"""


@dataclass
class DraftResult:
    grounding: dict[str, Any] = field(default_factory=dict)
    draft: str = ""
    final_statement: str = ""
    review_issues: list[str] = field(default_factory=list)
    digest: MedicalDigest | None = None

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
    result = DraftResult()

    def report(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)

    report(0.02, "Step 1/4 — Exhaustive review of medical records…")
    result.digest = review_medical_records(
        llm, records, progress=lambda f, m: progress((0.02 + f * 0.45), m) if progress else None
    )

    report(0.5, "Step 2/4 — Grounding witness observations against the records…")
    result.grounding = llm.chat_json(
        GROUNDING_SYSTEM,
        GROUNDING_USER.format(
            condition=condition,
            claim_type=claim_type,
            relationship=witness.get("relationship", "not specified"),
            observations=observations[:20000],
            digest=result.digest.as_json_text()[:24000],
        ),
    )

    report(0.68, "Step 3/4 — Drafting the statement…")
    result.draft = llm.chat(
        DRAFT_SYSTEM_TEMPLATE.format(guide=load_knowledge("drafting_guide.md")),
        DRAFT_USER.format(
            witness_name=witness.get("name", "[Witness Name]"),
            relationship=witness.get("relationship", "[relationship]"),
            known_since=witness.get("known_since", "[how long known]"),
            contact_frequency=witness.get("contact_frequency", "[frequency of contact]"),
            veteran_name=witness.get("veteran_name", "[Veteran Name]"),
            condition=condition,
            claim_type=claim_type,
            witnessed_event=witness.get("witnessed_event", "unknown"),
            observations=observations[:20000],
            grounding=_json_dumps(result.grounding),
            digest_summary=result.digest.summary or "(no summary)",
        ),
        max_tokens=6000,
    )

    report(0.85, "Step 4/4 — Self-review and improvement pass…")
    review = llm.chat_json(
        REVIEW_SYSTEM,
        REVIEW_USER.format(
            draft=result.draft[:16000], guide=load_knowledge("drafting_guide.md")[:6000]
        ),
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
    questions = result.grounding.get("strengthening_questions", [])
    if questions:
        lines.append("### ❓ Answer these to strengthen the statement (records suggest you may know)")
        for question in questions:
            lines.append(f"- {question}")
        lines.append("")
    return "\n".join(lines) or "_No grounding details produced._"
