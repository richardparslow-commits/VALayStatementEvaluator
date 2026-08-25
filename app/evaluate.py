"""Evaluation pathway: check an existing lay statement against medical records
and score it against the VA lay-evidence rubric."""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import load_knowledge
from .documents import ExtractedDocument
from .llm import LLMClient, LLMError
from .medical_review import (
    MedicalDigest,
    ProgressCallback,
    find_relevant_excerpts,
    review_medical_records,
)

CLAIMS_SYSTEM = """You are a VA claims evidence analyst. Decompose a lay/witness statement \
into atomic factual assertions so each can be checked against medical records. Distinguish \
firsthand observations from hearsay and from medical/legal conclusions."""

CLAIMS_USER = """Decompose the following lay/witness statement into atomic factual claims.

Return JSON:
{{
  "claimed_condition": "the disability condition this statement appears to support",
  "writer_role": "veteran | spouse | family | friend | coworker | fellow servicemember | other",
  "claims": [
    {{
      "id": 1,
      "text": "the single factual assertion, quoted/paraphrased faithfully",
      "type": "in_service_event | onset | symptom | diagnosis_reference | treatment_reference | date_or_place | functional_impact | continuity | other"
    }}
  ]
}}
Rules: capture EVERY checkable assertion (events, dates, places, symptoms, treatments,
providers, facilities). Keep each claim to one assertion. Number ids sequentially.

STATEMENT:
<<<
{statement}
>>>"""

VERIFY_SYSTEM = """You are an evidence auditor for VA disability claims. You must verify each \
factual claim from a lay statement against (1) a structured digest of the veteran's medical \
records and (2) raw record excerpts. Be rigorous but fair:
- SUPPORTED: a record entry clearly supports the claim.
- PARTIALLY SUPPORTED: supported in substance but with a discrepancy (e.g., date off by a
  year, different facility name).
- CONTRADICTED: a record entry clearly conflicts with the claim.
- NOT FOUND: nothing in the records confirms or denies it. IMPORTANT: under Buchanan v.
  Nicholson and Barr v. Nicholson, absence from records is NOT negative evidence — many lay
  facts (home symptoms, undocumented events) will legitimately be NOT FOUND. Never treat
  NOT FOUND as an error; only CONTRADICTED findings are accuracy failures.
Cite the supporting/conflicting record fact (with its source label and date) whenever possible."""

VERIFY_USER = """Verify each claim below against the medical record digest and raw excerpts.

Return JSON:
{{
  "verifications": [
    {{
      "id": <claim id>,
      "verdict": "SUPPORTED | PARTIALLY SUPPORTED | CONTRADICTED | NOT FOUND",
      "record_reference": "source label + date of the supporting/conflicting record fact, or empty",
      "note": "one short sentence explaining the verdict"
    }}
  ]
}}

MEDICAL RECORD DIGEST (JSON):
<<<
{digest}
>>>

RAW RECORD EXCERPTS RELEVANT TO THESE CLAIMS:
<<<
{excerpts}
>>>

CLAIMS TO VERIFY:
<<<
{claims}
>>>"""

RUBRIC_SYSTEM_TEMPLATE = """You are a senior veterans-claims advocate grading a lay/witness \
statement. Apply this rubric strictly and specifically, quoting the statement where useful.

{rubric}

LEGAL FRAMEWORK REFERENCE:
{legal}"""

RUBRIC_USER = """Grade this statement on all 8 rubric dimensions.

Return JSON:
{{
  "scores": {{
    "factual_accuracy": <0-10>,
    "specificity_detail": <0-10>,
    "lay_competence": <0-10>,
    "condition_connection": <0-10>,
    "continuity_timeline": <0-10>,
    "functional_impact": <0-10>,
    "credibility_consistency": <0-10>,
    "form_completeness": <0-10>
  }},
  "rationales": {{ "<dimension_key>": "1-3 sentences tied to the actual text" }},
  "improvements": [
    {{ "priority": 1, "problem": "...", "suggestion": "...", "example_rewrite": "..." }}
  ],
  "omitted_record_facts": [
    {{ "fact": "record fact the statement could usefully add", "source": "..." }}
  ],
  "executive_summary": "3-5 sentence overall assessment"
}}
Include 4-6 improvements ordered by impact. Only list omitted_record_facts the witness could
plausibly confirm from personal knowledge. Do NOT begin the executive_summary with an
"Overall Rating:" label — the tool computes the overall rating deterministically from the
scores, so your summary should focus on strengths, weaknesses, and the single most important
fix.

STATEMENT UNDER REVIEW:
<<<
{statement}
>>>

CLAIM-VERIFICATION RESULTS:
<<<
{verifications}
>>>

MEDICAL RECORD SUMMARY:
<<<
{digest_summary}
>>>"""


REVISE_SYSTEM = """You are a senior veterans-claims advocate rewriting a lay/witness statement \
to make it as strong and factually safe as possible, guided by a completed evidence review. \
Hard rules:
- NEVER invent facts. Every new factual detail must come from the provided record digest or the \
original statement. Facts taken from the records that the witness should personally confirm \
before signing MUST be wrapped in [Confirm: ...] placeholders.
- Claims verified CONTRADICTED must be corrected to match the medical records (use the record \
fact noted for them). If the witness might genuinely remember it differently, correct to the \
record and append a [Confirm: ...] note.
- Claims PARTIALLY SUPPORTED: align the disputed detail (date, place, name) with the records.
- Claims NOT FOUND: KEEP them — absence from records is not negative evidence (Buchanan v. \
Nicholson; Barr v. Nicholson). You may sharpen their wording but must not delete firsthand \
observations merely because the records are silent.
- Keep the writer's voice, grammatical person, and relationship (a coworker writes as a coworker).
- Stay inside lay competence: observations, symptoms, events, and functional impact only. Reword \
medical or legal conclusions as observations or attributed statements ("he told me his doctor \
said...").
- Do not pad with filler; preserve supported facts essentially as written.
- Add any missing formal elements: witness identity/relationship, opportunity to observe, \
certification of truthfulness, signature/date block, and a note that the statement is submitted \
on VA Form 21-10210.
- Prefer concrete dates, frequencies, and specific incidents over vague language."""

REVISE_USER = """Rewrite this lay/witness statement so it corrects every problem identified by \
the review while keeping all legitimate lay observations.

Return JSON:
{{
  "revision_notes": "2-3 sentences explaining your revision strategy",
  "changes": [
    {{
      "category": "contradiction_fix | alignment | specificity | lay_competence | structure | record_addition",
      "original": "quoted or summarized original passage (empty string if pure addition)",
      "revised": "the replacement or added text",
      "reason": "why, tied to the verification result or rubric finding"
    }}
  ],
  "revised_statement": "the COMPLETE revised statement, ready for the witness to review, with [Confirm: ...] placeholders wherever they must check something before signing",
  "added_facts_to_verify": ["each record-sourced fact you added that the witness must confirm"]
}}
Include one changes entry per meaningful change, in statement order (typically 5-15 entries).

ORIGINAL STATEMENT:
<<<
{statement}
>>>

CLAIM-VERIFICATION RESULTS:
<<<
{verifications}
>>>

RUBRIC IMPROVEMENTS IDENTIFIED:
<<<
{improvements}
>>>

RECORD FACTS THE STATEMENT COULD ADD:
<<<
{omitted_facts}
>>>

MEDICAL RECORD SUMMARY:
<<<
{digest_summary}
>>>"""


@dataclass
class EvaluationResult:
    claimed_condition: str = ""
    writer_role: str = ""
    claims: list[dict] = field(default_factory=list)
    verifications: list[dict] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    rationales: dict[str, str] = field(default_factory=dict)
    improvements: list[dict] = field(default_factory=list)
    omitted_record_facts: list[dict] = field(default_factory=list)
    executive_summary: str = ""
    revision_notes: str = ""
    revision_changes: list[dict] = field(default_factory=list)
    revised_statement: str = ""
    added_facts_to_verify: list[str] = field(default_factory=list)
    digest: MedicalDigest | None = None
    report_markdown: str = ""

    @property
    def contradiction_count(self) -> int:
        return sum(
            1 for v in self.verifications if v.get("verdict") == "CONTRADICTED"
        )

    @property
    def overall_rating(self) -> str:
        if not self.scores:
            return "Not scored"
        weighted = dict(self.scores)
        weighted["factual_accuracy"] = weighted.get("factual_accuracy", 0) * 1.5
        avg = sum(weighted.values()) / (len(self.scores) + 0.5)
        if avg >= 8.5:
            return "Excellent"
        if avg >= 7.0:
            return "Strong"
        if avg >= 5.0:
            return "Adequate"
        return "Needs Substantial Work"


DIMENSION_LABELS = {
    "factual_accuracy": "Factual Accuracy vs. Records",
    "specificity_detail": "Specificity & Detail",
    "lay_competence": "Lay Competence Boundaries",
    "condition_connection": "Connection to Claimed Condition",
    "continuity_timeline": "Continuity & Timeline",
    "functional_impact": "Functional Impact",
    "credibility_consistency": "Credibility & Consistency",
    "form_completeness": "Form & Completeness",
}


def run_evaluation(
    llm: LLMClient,
    statement_text: str,
    records: list[ExtractedDocument],
    progress: ProgressCallback | None = None,
) -> EvaluationResult:
    """Execute the full evaluation pipeline."""
    result = EvaluationResult()

    def report(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)

    report(0.02, "Step 1/6 — Exhaustive review of medical records…")
    result.digest = review_medical_records(
        llm, records, progress=lambda f, m: progress((0.02 + f * 0.48), m) if progress else None
    )

    report(0.52, "Step 2/6 — Extracting factual claims from the statement…")
    claims_data = llm.chat_json(
        CLAIMS_SYSTEM,
        CLAIMS_USER.format(statement=statement_text[:40000]),
    )
    result.claimed_condition = claims_data.get("claimed_condition", "")
    result.writer_role = claims_data.get("writer_role", "")
    result.claims = claims_data.get("claims", [])

    report(0.60, "Step 3/6 — Verifying each claim against the records…")
    result.verifications = _verify_claims(llm, result.claims, result.digest, records, report)

    report(0.78, "Step 4/6 — Scoring against the lay-evidence rubric…")
    rubric_data = llm.chat_json(
        RUBRIC_SYSTEM_TEMPLATE.format(
            rubric=load_knowledge("evaluation_rubric.md"),
            legal=load_knowledge("legal_framework.md"),
        ),
        RUBRIC_USER.format(
            statement=statement_text[:30000],
            verifications=_verifications_text(result),
            digest_summary=result.digest.summary or "(no summary)",
        ),
    )
    result.scores = {k: float(v) for k, v in rubric_data.get("scores", {}).items()}
    result.rationales = rubric_data.get("rationales", {})
    result.improvements = rubric_data.get("improvements", [])
    result.omitted_record_facts = rubric_data.get("omitted_record_facts", [])
    result.executive_summary = rubric_data.get("executive_summary", "")

    report(0.86, "Step 5/6 — Drafting improvement suggestions and a revised statement…")
    _draft_revision(llm, result, statement_text, report)

    report(0.96, "Step 6/6 — Building the report…")
    result.report_markdown = build_report(result, statement_text)
    report(1.0, "Evaluation complete.")
    return result


def _draft_revision(
    llm: LLMClient,
    result: EvaluationResult,
    statement_text: str,
    report: ProgressCallback,
) -> None:
    """Generate the itemized improvement plan and a suggested rewrite.

    A failure here should not lose the completed evaluation, so errors are
    swallowed and the revision fields simply stay empty.
    """
    import json as _json

    try:
        revise_data = llm.chat_json(
            REVISE_SYSTEM,
            REVISE_USER.format(
                statement=statement_text[:30000],
                verifications=_verifications_text(result),
                improvements=_json.dumps(result.improvements, indent=1)[:6000] or "(none)",
                omitted_facts=_json.dumps(result.omitted_record_facts, indent=1)[:4000]
                or "(none)",
                digest_summary=(result.digest.summary or "(no summary)")[:12000],
            ),
            max_tokens=6000,
        )
    except LLMError:
        result.revision_notes = "Revision draft unavailable — the model call failed."
        return
    result.revision_notes = revise_data.get("revision_notes", "")
    result.revision_changes = revise_data.get("changes", [])
    result.revised_statement = revise_data.get("revised_statement", "")
    result.added_facts_to_verify = [
        str(f) for f in revise_data.get("added_facts_to_verify", []) if str(f).strip()
    ]
    report(0.94, "Improvement suggestions drafted.")


def _verify_claims(
    llm: LLMClient,
    claims: list[dict],
    digest: MedicalDigest,
    records: list[ExtractedDocument],
    report: ProgressCallback,
) -> list[dict]:
    """Verify claims in small batches so each prompt stays focused."""
    verdict_by_id: dict[int, dict] = {}
    batch_size = 8
    batches = [claims[i : i + batch_size] for i in range(0, len(claims), batch_size)]
    for index, batch in enumerate(batches, start=1):
        report(
            0.60 + 0.16 * index / max(len(batches), 1),
            f"Verifying claims — batch {index}/{len(batches)}…",
        )
        batch_query = " ".join(str(c.get("text", "")) for c in batch)
        excerpts = find_relevant_excerpts(records, batch_query, top_k=6)
        import json as _json

        data = llm.chat_json(
            VERIFY_SYSTEM,
            VERIFY_USER.format(
                digest=digest.as_json_text()[:24000],
                excerpts=excerpts[:8000] or "(no matching raw excerpts found)",
                claims=_json.dumps(batch, indent=1),
            ),
        )
        for item in data.get("verifications", []):
            try:
                verdict_by_id[int(item.get("id"))] = item
            except (TypeError, ValueError):
                continue
    return [verdict_by_id.get(c["id"], {"id": c["id"], "verdict": "NOT FOUND",
            "record_reference": "", "note": "Not returned by verifier."}) for c in claims]


def _verifications_text(result: EvaluationResult) -> str:
    lines = []
    claim_text = {c["id"]: c.get("text", "") for c in result.claims}
    for v in result.verifications:
        lines.append(
            f"- Claim {v.get('id')}: \"{claim_text.get(v.get('id'), '')}\" => "
            f"{v.get('verdict')} | ref: {v.get('record_reference', '')} | {v.get('note', '')}"
        )
    return "\n".join(lines) or "(no claims extracted)"


_VERDICT_EMOJI = {
    "SUPPORTED": "✅",
    "PARTIALLY SUPPORTED": "🟡",
    "CONTRADICTED": "❌",
    "NOT FOUND": "⚪",
}


def build_report(result: EvaluationResult, statement_text: str) -> str:
    """Render the full markdown evaluation report."""
    lines: list[str] = []
    lines.append("# Lay Statement Evaluation Report")
    lines.append("")
    lines.append(f"**Overall rating: {result.overall_rating}**")
    if result.claimed_condition:
        lines.append(f"**Appears to support claim for:** {result.claimed_condition}")
    if result.writer_role:
        lines.append(f"**Writer role:** {result.writer_role}")
    if result.digest:
        lines.append(
            f"**Records reviewed:** {result.digest.pages_reviewed} pages, "
            f"{len(result.digest.facts)} facts extracted"
        )
    lines.append("")

    lines.append("## Executive Summary")
    lines.append(result.executive_summary or "(none)")
    lines.append("")

    if result.contradiction_count:
        lines.append(f"## ⚠️ Critical: {result.contradiction_count} contradiction(s) with the medical records")
        lines.append("")
        for v in result.verifications:
            if v.get("verdict") == "CONTRADICTED":
                claim_text = next(
                    (c.get("text", "") for c in result.claims if c.get("id") == v.get("id")), ""
                )
                lines.append(f"- **Claim:** {claim_text}")
                lines.append(f"  - **Conflicting record:** {v.get('record_reference', 'n/a')}")
                lines.append(f"  - **Note:** {v.get('note', '')}")
        lines.append("")
        lines.append(
            "Contradictions materially damage credibility. Correct these statements to match "
            "the records, or obtain a written explanation if the records are wrong."
        )
        lines.append("")

    lines.append("## Claim-by-Claim Verification")
    lines.append("")
    lines.append("| # | Claim | Verdict | Record Reference | Note |")
    lines.append("|---|-------|---------|------------------|------|")
    claim_text = {c["id"]: c.get("text", "") for c in result.claims}
    for v in result.verifications:
        verdict = v.get("verdict", "NOT FOUND")
        emoji = _VERDICT_EMOJI.get(verdict, "⚪")
        text = claim_text.get(v.get("id"), "").replace("|", "/")[:120]
        ref = (v.get("record_reference") or "—").replace("|", "/")[:80]
        note = (v.get("note") or "").replace("|", "/")[:120]
        lines.append(f"| {v.get('id')} | {text} | {emoji} {verdict} | {ref} | {note} |")
    lines.append("")
    lines.append(
        "_⚪ NOT FOUND is not a failure — lay facts like home symptoms or undocumented events "
        "often legitimately do not appear in medical records (Buchanan v. Nicholson; Barr v. Nicholson)._"
    )
    lines.append("")

    lines.append("## Rubric Scores")
    lines.append("")
    lines.append("| Dimension | Score | Rationale |")
    lines.append("|-----------|-------|-----------|")
    for key, label in DIMENSION_LABELS.items():
        score = result.scores.get(key, 0)
        rationale = (result.rationales.get(key) or "").replace("|", "/")[:200]
        lines.append(f"| {label} | {score:.1f}/10 | {rationale} |")
    lines.append("")

    if result.improvements:
        lines.append("## Top Improvements (in priority order)")
        lines.append("")
        for imp in result.improvements:
            lines.append(f"**{imp.get('priority', '?')}. {imp.get('problem', '')}**")
            lines.append(f"   - Fix: {imp.get('suggestion', '')}")
            if imp.get("example_rewrite"):
                lines.append(f"   - Example: “{imp.get('example_rewrite')}”")
            lines.append("")

    if result.omitted_record_facts:
        lines.append("## Facts in the Records You Could Add (verify from personal knowledge first)")
        lines.append("")
        for fact in result.omitted_record_facts:
            lines.append(f"- {fact.get('fact', '')} _(source: {fact.get('source', 'records')})_")
        lines.append("")

    if result.revised_statement or result.revision_changes:
        lines.append("## Suggested Improvements — Proposed Rewrite")
        lines.append("")
        if result.revision_notes:
            lines.append(f"**Revision strategy:** {result.revision_notes}")
            lines.append("")
        if result.revision_changes:
            lines.append("| # | Category | Original | Suggested | Why |")
            lines.append("|---|----------|----------|-----------|-----|")
            for index, change in enumerate(result.revision_changes, start=1):
                original = str(change.get("original", "") or "(addition)").replace("|", "/")[:200]
                revised = str(change.get("revised", "")).replace("|", "/")[:300]
                reason = str(change.get("reason", "")).replace("|", "/")[:200]
                category = str(change.get("category", ""))[:24]
                lines.append(f"| {index} | {category} | {original} | {revised} | {reason} |")
            lines.append("")
        if result.added_facts_to_verify:
            lines.append("**Record-sourced facts added — the witness must confirm each before signing:**")
            lines.append("")
            for fact in result.added_facts_to_verify:
                lines.append(f"- {fact}")
            lines.append("")
        if result.revised_statement:
            lines.append("### Revised statement (resolve every `[Confirm: …]` before signing)")
            lines.append("")
            lines.append("```text")
            lines.append(result.revised_statement)
            lines.append("```")
            lines.append("")

    if result.digest and result.digest.summary:
        lines.append("## Medical Record Digest (what the review saw)")
        lines.append("")
        lines.append(result.digest.summary)
        lines.append("")

    lines.append("---")
    lines.append(
        "_This report was generated by an automated tool as a drafting aid. It is not legal, "
        "medical, or claims advice. Consult an accredited VSO, claims agent, or attorney "
        "(www.va.gov/ogc/apps/accreditation) before submitting evidence._"
    )
    return "\n".join(lines)
