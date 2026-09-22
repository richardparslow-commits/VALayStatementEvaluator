"""Evaluation pathway: check an existing lay statement against medical records
and score it against the VA lay-evidence rubric."""
from __future__ import annotations

from dataclasses import dataclass, field

import logging
import re
import time
from typing import Any

import streamlit as st

from .agiloop_telemetry import track_feature_error, track_goal
from .config import load_knowledge
from .documents import (
    EVALUATE_INTERNAL_MAX_CHARS,
    ExtractedDocument,
    MAX_STATEMENT_CHARS,
)
from .exporter import parse_source
from .llm import LLMClient, LLMError, LLMParseError, LLMService
from . import tracing
from .logging_config import PhaseTimer, get_request_id
from .profiler import phase_timer
from .pipeline_guard import check_pipeline_cancelled
from .prompt_sanitize import GUARD_NOTE, sanitize_digest_text, sanitize_for_prompt

logger = logging.getLogger("app.evaluate")
from .medical_review import (
    MedicalDigest,
    ProgressCallback,
    query_has_content_words,
    retrieve_evidence,
    review_medical_records,
)
from .aa_intake import care_gaps_text

# Feature: Condition-Specific Templates
FEATURE_ID = "02f0935a-ee5e-4083-88a2-10e11753ccc9"  # condition-specific-templates

# Feature: Medical Record Search & Citation Index
SEARCH_FEATURE_ID = "22bc7e10-dcda-431e-b3fb-4e8ff9b532cb"  # medical-record-search-citation-index

# Feature: Statement Effectiveness Score & Improvement Recommendations
EFFECTIVENESS_FEATURE_ID = "94104045-aa12-4018-95c6-e6912e659803"  # statement-effectiveness-score-improvement-recommendations

VERIFICATION_MAX_ATTEMPTS = 3
CLAIM_TYPES = frozenset({
    "in_service_event", "onset", "symptom", "diagnosis_reference",
    "treatment_reference", "date_or_place", "functional_impact", "continuity", "other",
})
# How the writer knows each fact. Evidentiary weight turns on this: the writer's own
# experience or a firsthand observation is competent lay evidence (38 U.S.C. § 1154(a);
# 38 C.F.R. § 3.159(a)(2)), a relay of what someone else said is evidence of the
# statement made rather than of the inner state, and a diagnosis, cause or rating
# assertion is outside lay competence (Jandreau v. Nicholson). Asked for in
# ``CLAIMS_USER``, carried into the verifier prompt with the claims themselves, and
# surfaced to the rubric and revision prompts by ``_verifications_text``.
CLAIM_BASES = frozenset({
    "experienced", "observed", "reported", "provider_statement", "conclusion",
})
# What a basis is recorded as when the model omits it or invents a value. The basis
# guides the reviewer and the downstream prompts; it is never a reason to fail an
# evaluation whose records have already been read, so anything unexpected degrades to
# this quietly instead of raising.
UNKNOWN_CLAIM_BASIS = "unspecified"


class VerificationIncompleteError(LLMParseError):
    """Verification could not produce a complete, unambiguous set of verdicts."""

CLAIMS_SYSTEM = """You are a VA claims evidence analyst. Decompose a lay/witness statement \
into atomic factual assertions so each can be checked against medical records, and record the \
BASIS OF KNOWLEDGE for each one — how the writer knows it. Evidentiary weight turns on that \
basis, so it must be captured exactly as written, never smoothed over:
- "experienced": the writer's own sensation, thought or capacity ("I felt it catch", "I still \
  cannot lift a gallon of milk", "I wake at 3 a.m."). Competent lay evidence.
- "observed": the writer personally saw, heard or smelled it ("I watched him stop after half a \
  block", "I heard him cry out"). Competent lay evidence.
- "reported": the writer relays what another person said about that person's own inner state \
  ("he told me his back was burning"). This is competent evidence THAT THE STATEMENT WAS \
  MADE; it is weaker proof of the inner state than the person's own account, and it never \
  becomes the writer's own observation.
- "provider_statement": the writer relays what a medical provider said or did ("the doctor told \
  him a nerve was pinched", "they gave him shots"). Evidence of what the provider said, not \
  proof of the clinical fact.
- "conclusion": the writer's own diagnosis, cause, prognosis or rating assertion ("his \
  arthritis is service-connected", "he meets the criteria for 100%"). Outside lay competence \
  (Jandreau v. Nicholson). Still capture it — the reviewer must see it — and do not reword it \
  into a symptom or into later "fact".
Rules that make the rest of the pipeline trustworthy:
- Preserve the writer's attribution in the claim text. Keep the words that show who perceived \
  it: "I saw", "he told me", "the doctor said". NEVER upgrade a relay into a firsthand fact \
  ("he told me his knee gave out" is not "his knee gave out") and never restate an \
  observation as a diagnosis.- One assertion that carries two bases is TWO claims — split it. "I saw him limp and he told \
me the pain was a 9" is one observation and one report, and they are weighed differently, so \
splitting is required; re-labeling a merged assertion with one basis loses the distinction.
- Stay in the writer's own vantage point. A veteran's "I" and a spouse's "he" are different \
  witnesses: quote the statement's point of view, do not swap or convert it.
- Do not decide whether a claim is credible or true here. This pass is decomposition only."""

CLAIMS_USER = """Decompose the following lay/witness statement into atomic factual claims.

Return JSON:
{{
  "claimed_condition": "the disability condition this statement appears to support",
  "writer_role": "veteran | spouse | family | friend | coworker | fellow servicemember | other",
  "claims": [
    {{
      "id": 1,
      "text": "the single factual assertion, quoted/paraphrased faithfully, keeping the writer's attribution words (I saw, he told me, the doctor said)",
      "type": "in_service_event | onset | symptom | diagnosis_reference | treatment_reference | date_or_place | functional_impact | continuity | other",
      "basis": "experienced | observed | reported | provider_statement | conclusion"
    }}
  ]
}}
Rules: capture EVERY checkable assertion (events, dates, places, symptoms, treatments,
providers, facilities). Keep each claim to one assertion, and set "basis" from how the writer
knows the fact — not from how strong it sounds. If one sentence holds both what the writer
observed and what someone reported, emit two claims, one per basis. Number ids sequentially.
Set a basis whenever the text shows one; if the sentence genuinely does not show how the writer
knows the fact, omit the field rather than guess — the tool records an omitted basis as unknown,
and a guessed basis would be weighed as if the witness had actually said it.

STATEMENT:
<<<
{statement}
>>>

{guard_note}"""

VERIFY_SYSTEM = """You are an evidence auditor for VA disability claims. You must verify each \
factual claim from a lay statement against (1) a structured digest of the veteran's medical \
records and (2) raw record excerpts. Be rigorous but fair:
- SUPPORTED: a record entry clearly supports the claim.
- PARTIALLY SUPPORTED: supported in substance but with a discrepancy (e.g., date off by a
  year, different facility name).
- CONTRADICTED: a record entry AFFIRMATIVELY states the opposite — a dated normal finding, \
  "denies pain", "no tenderness", "gait normal", the other side, an incompatible date or \
  facility. A contradiction is an affirmative finding, so it MUST name the conflicting record \
  entry in record_reference. If you cannot point to that specific record text, the verdict is \
  NOT FOUND, not CONTRADICTED; the tool downgrades an uncited contradiction to a record gap.
- NOT FOUND: nothing in the records confirms or denies it. This is the correct verdict for \
  silence, and it is not an accuracy failure. Absence of evidence on a question is not \
  substantive negative evidence against a claimant (Buczynski v. Shinseki; Horn v. Shinseki; \
  M21-1 V.ii.1.A), and the absence of contemporaneous records is not, standing alone, a \
  reason to reject lay evidence (Buchanan v. Nicholson). Many lay facts (home symptoms, \
  unrecorded events) will legitimately be NOT FOUND. Silence counts against a claim only \
  where the fact is one that would normally be noted or reported (Buczynski v. Shinseki) — \
  and even then the verdict is NOT FOUND, because a missing entry means the record set may \
  be incomplete, not that the claim is false. Never treat NOT FOUND as an error; only \
  CONTRADICTED findings are accuracy failures.
A normal finding does not contradict a symptom: a static exam recording normal range of \
motion, a normal gait, or "no acute distress" never measured painful motion, functional \
loss, or flare-ups, so it does not conflict with a claim about them (38 C.F.R. §§ 4.40, 4.45, \
4.59; DeLuca v. Brown). Mark those claims NOT FOUND.
Each claim arrives with a "basis" field — how the writer knows it — and the verdict must \
respect it:
- experienced / observed: a competent lay account of a symptom, an event, or functional loss \
  (38 U.S.C. § 1154(a); 38 C.F.R. § 3.159(a)(2)). Records are commonly silent on these, and \
  silence is NOT FOUND. A record entry that restates what the veteran reported supports that \
  the report was made, not the clinical fact, so say so in the note rather than treating it as \
  corroboration of the medicine.
- reported: verify the fact that it was said, not a third person's inner state. A record entry \
  documenting the veteran's own complaint SUPPORTS that he said it, even where the examiner's \
  own findings were normal. Only a dated, specific entry about the same symptom can CONTRADICT \
  it.
- provider_statement: SUPPORTED only where a record entry documents the provider saying or \
  doing that; the clinical fact behind it is not thereby established, so record that in the \
  note. Where the record shows the same conversation differently, that is a discrepancy in what \
  was said, so use PARTIALLY SUPPORTED — not CONTRADICTED.
- conclusion: judge only the checkable part against the record (does the record name that \
  condition? does it record that event?). An inference a layperson is not competent to draw is \
  NOT FOUND, with a note that competence rather than accuracy is the issue, because the rubric \
  grades competence separately. Never mark it CONTRADICTED merely because a lay witness could \
  not assert it, and never mark it SUPPORTED merely because the record repeats the writer's own \
  words back to them.
Cite the supporting or conflicting record fact, with its source label and date, on every \
verdict: a named record entry for SUPPORTED, PARTIALLY SUPPORTED and CONTRADICTED; empty is \
correct only for NOT FOUND.
Do not describe a NOT FOUND claim as unsupported, unverified, inaccurate or inconsistent, and \
never write that the records "show no" or "contain no record of" the fact — write that the \
provided records do not address it."""

VERIFY_USER = """Verify each claim below against the medical record digest and raw excerpts.
Return exactly one verdict for EVERY submitted claim id, with no duplicates or extra ids.
Use NOT FOUND only when the supplied evidence does not confirm or deny the claim, never
as a substitute for skipping verification.

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
>>>

{guard_note}"""

RUBRIC_SYSTEM_TEMPLATE = """You are a senior veterans-claims advocate grading a lay/witness \
statement. Apply this rubric strictly and specifically, quoting the statement where useful.

THREE WAYS THIS RUBRIC IS MISAPPLIED — avoid all three:
1. Treating absence as evidence. A claim the records do not address is NOT FOUND: it is not a \
discrepancy and it does not lower factual accuracy. Only an affirmative contradiction does \
(Buczynski v. Shinseki; Horn v. Shinseki; M21-1 V.ii.1.A; Buchanan v. Nicholson). A claim the \
records merely fail to corroborate was weighed as evidence, not left unsupported.
2. Reading a static examination as rebuttal. A normal range of motion, a normal gait or "no \
acute distress" never measured painful motion, functional loss or flare-ups, so it cannot \
contradict a statement about them (38 C.F.R. §§ 4.40, 4.45, 4.59; DeLuca v. Brown; Sharp v. \
Shulkin). Where the record holds no such measurement, the statement's functional detail is \
unimpeached, and the missing measurement is a development question — not a credibility defect \
in the witness.
3. Diluting competent lay evidence. A witness describing what they personally felt, saw or \
could not do is giving competent evidence (38 U.S.C. § 1154(a); 38 C.F.R. § 3.159(a)(2); \
Jandreau v. Nicholson). Do not withhold credit because no medical opinion joins it, and do not \
reward a diagnosis, a cause, a prognosis or a rating the witness is not competent to give. \
Penalize both errors, and supplied the reword for each.
Each claim carries a "basis" field (experienced, observed, reported, provider_statement, \
conclusion). Use it when judging lay competence and credibility: a firsthand account is \
competent evidence even when the records are silent, a relay of what someone else said shows \
the statement was made rather than proving the inner state, and a provider's reported words \
are evidence of what was said — worth noting, and worth asking that the underlying treatment \
record be obtained.
Score each dimension on the statement as written, tie every rationale to its actual text, and \
treat the claim-verification results as findings to grade rather than verdicts to re-derive.

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
>>>

{guard_note}"""


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
- Claims NOT FOUND: KEEP them — absence from records is not negative evidence (Horn v. \
Shinseki; M21-1 V.ii.1.A; Buchanan v. Nicholson). You may sharpen their wording but must not \
delete firsthand observations merely because the records are silent, and you must not describe \
them as unverified, inaccurate or unsupported.
- LAY ATTRIBUTION — mandatory on every NOT FOUND claim. Because the records say nothing about \
it, the wording is the only thing the reader has, so every sentence carrying a NOT FOUND claim \
must show HOW THE WITNESS KNOWS IT. State the basis of knowledge in the witness's own voice:

  * veteran writing about himself — "I felt", "I noticed", "I still cannot", "it wakes me";
  * a witness describing the veteran — "I saw", "I watched", "I was there when", "I heard";
  * something the veteran told the witness — keep it a relay: "he told me", "she described".

Attribution is the basis of knowledge, NOT doubt. Never hedge a competent observation \
("I believe I may have had trouble sleeping" -> "I woke three times a night and could not fall \
back asleep"), and never attach "unverified", "allegedly", "claimed", "reportedly" or \
"supposedly" to the witness's own account.
- NO UPGRADE TO DIAGNOSIS on a NOT FOUND claim. Never convert an unverified symptom into a \
diagnosis, a cause, a prognosis or a rating, and never imply the records support it: do not \
write "was diagnosed with", "due to", "caused by", "service-connected", "% disabled", "the \
records show", "it is documented", or any condition name the witness did not use. A symptom \
stays a symptom — "pain that wakes me at 3 a.m." — described in terms of what the witness \
perceived and could not do.
- If a NOT FOUND claim is a medical or legal conclusion the writer is not competent to give, \
reword it to the observation behind it (Jandreau v. Nicholson: lay competence reaches what the \
witness personally observed, and conditions that are simple and readily observable) and put the \
unverified clinical part in a [Confirm: ...] placeholder instead of asserting it.
- When the writer is relaying another person's inner state, keep the relay: a witness cannot \
testify to what someone else felt or thought (38 C.F.R. § 3.159(a)(2)), so "he told me the pain \
was a 9" stays as he said it and never becomes "he was in severe pain".
- Keep the writer's voice, grammatical person, and relationship (a coworker writes as a coworker).
- Stay inside lay competence: observations, symptoms, events, and functional impact only. Reword \
medical or legal conclusions as observations or attributed statements ("he told me his doctor \
said...").
- Do not pad with filler; preserve supported facts essentially as written.
- Add any missing formal elements: witness identity/relationship, opportunity to observe, \
certification of truthfulness, signature/date block, and a note that the statement is submitted \
on VA Form 21-10210.
- Prefer concrete dates, frequencies, and specific incidents over vague language.
- TOPIC COVERAGE: where the topic analysis says an applicable topic is absent or partial, the \
rewrite may insert a bracketed prompt such as [Add if applicable: describe ...] so the witness \
can supply it if true. NEVER invent the content of a topic the witness never described.
- Translate symptoms into observable, dangerous consequences where grounded (e.g., memory loss \
-> double-dosing risk), and where applicable make explicit that the caregiver's help is \
necessary for safety, not merely convenient."""

REVISE_USER = """Rewrite this lay/witness statement so it corrects every problem identified by \
the review while keeping all legitimate lay observations.

Return JSON:
{{
  "revision_notes": "2-3 sentences explaining your revision strategy",
  "changes": [
    {{
      "category": "contradiction_fix | alignment | specificity | lay_competence | lay_attribution | structure | record_addition",
      "original": "quoted or summarized original passage (empty string if pure addition)",
      "revised": "the replacement or added text",
      "reason": "why, tied to the verification result or rubric finding"
    }}
  ],
  "revised_statement": "the COMPLETE revised statement, ready for the witness to review, with [Confirm: ...] placeholders wherever they must check something before signing",
  "added_facts_to_verify": ["each record-sourced fact you added that the witness must confirm"]
}}
Include one changes entry per meaningful change, in statement order (typically 5-15 entries).
Return only the structure above — no extra fields, no commentary — and keep every entry short.
lay_attribution is a required entry, not an optional one: for EVERY claim the verification
results mark NOT FOUND, add one changes entry with category lay_attribution whose `revised` field
is the sentence that now carries the claim and whose `reason` quotes the words that state the
witness's basis of knowledge ("I watched him stop", "he told me the pain was a 9"). Those entries
are how the witness checks, before signing, that an unverified claim reads as their own account
rather than as a diagnosis — and if the original sentence already stated the basis, say so in the
reason and change nothing about it.

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
>>>

TOPIC COVERAGE ANALYSIS:
<<<
{topic_analysis}
>>>

{guard_note}"""


TOPIC_SYSTEM_TEMPLATE = """You are a senior veterans-claims advocate auditing a lay/witness \
statement for TOPIC COVERAGE against this checklist:

{checklist}

LEGAL FRAMEWORK REFERENCE:
{legal}

Rules:
- First identify the statement's claim focus, then decide which checklist topics are APPLICABLE \
to it. Physical-condition claims rarely need medication-mismanagement or self-harm topics; \
mental-health / caregiver-necessity / aid-and-attendance claims usually need most of them. \
Never force inapplicable topics onto a claim: an inapplicable topic is "not applicable", not a \
gap, and a physical claim is never deficient for lacking hazard or self-harm detail.
- For each applicable topic, judge coverage from the STATEMENT TEXT itself: "covered" (concrete \
examples present), "partial" (mentioned vaguely or without specifics), "absent" (not addressed).
- Judge coverage by OBSERVABLE DETAIL, not by medical language. For a musculoskeletal or \
physical claim, "covered" requires the function: what the pain stops him from doing and for how \
long, pain on movement rather than only at rest, what a flare-up costs him and how often it \
occurs, and what has changed over time (38 C.F.R. §§ 4.40, 4.45, 4.59; DeLuca v. Brown; Sharp v. \
Shulkin). A statement that reports only a static limitation is "partial" at best, even when it \
names the condition — and a missing range-of-motion number is not the gap.
- Never ask the witness for what a layperson is not competent to give: no diagnosis, no cause, \
no prognosis, no rating or percentage, no measured range-of-motion values (Jandreau v. \
Nicholson; 38 C.F.R. § 3.159(a)(2)). Ask for what the witness personally observed, or for what \
the veteran could not do.
- Silence in the medical records is not a gap in the statement (Buczynski v. Shinseki; Horn v. \
Shinseki): judge the statement on what it says, and never fault it because the records do not \
mention the topic.
- Quote short evidence from the statement where present.
- Each gap_note must be a concrete, implementable suggestion for what the witness should \
describe or add if it is true, phrased as something the witness can supply in their own voice \
(describe how far he could walk before stopping) rather than as a fact to be established.
- critical_gaps: the missing/weak applicable topics whose absence most weakens THIS claim (max 5).
- NEVER propose adding facts the witness may not know; gap notes prompt recollection or flag \
the topic for the witness to address if true."""

TOPIC_USER = """Analyze topic coverage for this statement.

Return JSON:
{{
  "claim_focus": "one-line description of what this statement supports (condition + claim angle)",
  "topics": [
    {{
      "topic": "checklist topic label (A-O)",
      "applicable": true | false,
      "coverage": "covered | partial | absent | not applicable",
      "evidence": "short quote/paraphrase from the statement, or empty string",
      "gap_note": "how to cover or strengthen it, or empty string"
    }}
  ],
  "critical_gaps": ["up to 5 highest-impact missing or weak applicable topics and why they matter"],
  "notes": "1-2 sentence overall coverage assessment"
}}
Return one topics entry per checklist topic (A through O), in checklist order.

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
>>>

{guard_note}"""


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
    topic_focus: str = ""
    topic_rows: list[dict] = field(default_factory=list)
    topic_critical_gaps: list[str] = field(default_factory=list)
    topic_notes: str = ""
    revision_notes: str = ""
    revision_changes: list[dict] = field(default_factory=list)
    revised_statement: str = ""
    added_facts_to_verify: list[str] = field(default_factory=list)
    digest: MedicalDigest | None = None
    report_markdown: str = ""
    # Truncation audit — set when the input exceeds EVALUATE_INTERNAL_MAX_CHARS
    input_chars: int = 0
    truncated_chars: int = 0
    truncation_warning: str = ""
    # Statement Effectiveness Score & Improvement Recommendations (F4)
    effectiveness_score: int = 0
    recommendations: list[dict] = field(default_factory=list)
    # Claims whose batch retrieved no matching record text. These are coverage
    # gaps, not findings: the statement is silent *and* the records supplied
    # nothing to check it against, which is a different thing to tell a veteran.
    evidence_gaps: list[dict] = field(default_factory=list)
    # Complete source evidence preserved independently of prompt budgets.
    # Each dict holds {"filename": str, "kind": "page"|"block", "page": int,
    # "text": str} — the raw record pages that fed the digest extraction and
    # verification. Summaries and selected facts in this result are derived
    # views of this store; prompt budget limits never truncate it.
    evidence_source: list[dict] = field(default_factory=list)

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

    @property
    def score_band(self) -> str:
        """Color band for ``effectiveness_score`` (green >75, yellow 50-75, red <50)."""
        return compute_score_band(self.effectiveness_score)


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
    llm: LLMService,
    statement_text: str,
    records: list[ExtractedDocument],
    progress: ProgressCallback | None = None,
    witness: dict[str, str] | None = None,
) -> EvaluationResult:
    """Execute the full evaluation pipeline.

    *witness* is the optional witness-metadata dict (the same one the draft
    pathway carries). Only its ``aa_*`` keys are read — the structured Aid &
    Attendance intake answers — and they feed the recommendations phase's
    care-coverage gaps. ``None`` keeps every prompt byte-identical to the
    pre-intake pipeline.
    """
    rid = get_request_id() or "-"
    t0 = time.perf_counter()
    pages = sum(len(d.pages) for d in records)
    logger.info(
        "evaluate start pages=%d statement_chars=%d",
        pages,
        len(statement_text),
        extra={"request_id": rid, "phase": "evaluate", "status": "start"},
    )
    # Root span for the run (no-op unless tracing is enabled). Phase spans nest
    # under it, and in Pattern C the worker continues this same trace.
    with tracing.run_span(
        "evaluate", files=len(records), pages=pages, chars=len(statement_text)
    ):
        try:
            result = _run_evaluation(llm, statement_text, records, progress, witness or {})
            duration_ms = int((time.perf_counter() - t0) * 1000)
            logger.info(
                "evaluate done duration_ms=%d claims=%d verifications=%d contradictions=%d",
                duration_ms,
                len(result.claims),
                len(result.verifications),
                result.contradiction_count,
                extra={
                    "request_id": rid,
                    "phase": "evaluate",
                    "status": "ok",
                    "duration_ms": duration_ms,
                },
            )
            return result
        except Exception as exc:  # noqa: BLE001 - feature-error boundary
            duration_ms = int((time.perf_counter() - t0) * 1000)
            logger.error(
                "evaluate error duration_ms=%d error=%s",
                duration_ms,
                f"{type(exc).__name__}: {exc}",
                exc_info=exc,
                extra={
                    "request_id": rid,
                    "phase": "evaluate",
                    "status": "error",
                    "duration_ms": duration_ms,
                    "error_class": type(exc).__name__,
                },
            )
            track_feature_error(FEATURE_ID, exc)
            raise


def _truncate_for_prompt(text: str, limit: int = EVALUATE_INTERNAL_MAX_CHARS) -> tuple[str, int]:
    """Return (possibly truncated text, chars_removed) bounded at *limit*."""
    if len(text) <= limit:
        return text, 0
    return text[:limit], len(text) - limit


def _normalize_claim_id(value: Any) -> int:
    """Accept JSON integers and decimal strings, never bools or rounded floats."""
    if type(value) is int and value > 0:
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        try:
            parsed = int(value.strip())
        except ValueError:
            pass
        else:
            if parsed > 0:
                return parsed
    raise LLMParseError("Claim ids must be positive integers or decimal integer strings.")


def _normalize_claim_basis(value: Any) -> str:
    """The claim's basis of knowledge, or ``UNKNOWN_CLAIM_BASIS`` if unrecognised.

    Deliberately lenient: the basis sharpens how a claim is weighed downstream, but a
    missing or invented label must never invalidate an evaluation whose records have
    already been read, so anything unexpected degrades to a neutral value instead of
    raising the way a malformed id, text or type does.
    """
    if isinstance(value, str) and value.strip().lower() in CLAIM_BASES:
        return value.strip().lower()
    return UNKNOWN_CLAIM_BASIS


def _normalize_claims(data: Any) -> list[dict]:
    if not isinstance(data, list):
        raise LLMParseError("Claim extraction is incomplete: expected a claims list.")
    normalized = []
    seen: set[int] = set()
    for item in data:
        if not isinstance(item, dict):
            raise LLMParseError("Claim extraction is incomplete: a claim is not an object.")
        claim_id = _normalize_claim_id(item.get("id"))
        if claim_id in seen:
            raise LLMParseError("Claim extraction is incomplete: duplicate claim ids.")
        text = item.get("text")
        claim_type = item.get("type", "other")
        if not isinstance(text, str) or not text.strip():
            raise LLMParseError("Claim extraction is incomplete: a claim has no valid text.")
        if not isinstance(claim_type, str) or claim_type.strip().lower() not in CLAIM_TYPES:
            raise LLMParseError("Claim extraction is incomplete: invalid claim type.")
        normalized.append({
            "id": claim_id,
            "text": text.strip(),
            "type": claim_type.strip().lower(),
            "basis": _normalize_claim_basis(item.get("basis")),
        })
        seen.add(claim_id)
    return normalized


def _normalize_verifications(data: Any, expected_ids: set[int]) -> list[dict]:
    if not isinstance(data, dict) or not isinstance(data.get("verifications"), list):
        raise LLMParseError("Expected an object containing a verifications list.")
    normalized = []
    seen: set[int] = set()
    for item in data["verifications"]:
        if not isinstance(item, dict):
            raise LLMParseError("Each verification must be an object.")
        claim_id = _normalize_claim_id(item.get("id"))
        if claim_id not in expected_ids:
            raise LLMParseError("Verifier returned an id not submitted in this batch.")
        if claim_id in seen:
            raise LLMParseError("Verifier returned duplicate verdicts for a claim.")
        verdict = item.get("verdict")
        if not isinstance(verdict, str) or verdict.strip().upper() not in VERDICTS:
            raise LLMParseError("Verifier returned an invalid verdict label.")
        reference, note = item.get("record_reference"), item.get("note")
        if not isinstance(reference, str) or not isinstance(note, str):
            raise LLMParseError("Verifier reference and note must both be strings.")
        normalized.append({
            "id": claim_id, "verdict": verdict.strip().upper(),
            "record_reference": reference.strip(), "note": note.strip(),
        })
        seen.add(claim_id)
    if seen != expected_ids:
        raise LLMParseError(f"Verifier omitted {len(expected_ids - seen)} submitted claim(s).")
    return normalized


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


def _run_evaluation(
    llm: LLMService,
    statement_text: str,
    records: list[ExtractedDocument],
    progress: ProgressCallback | None,
    witness: dict[str, str],
) -> EvaluationResult:
    result = EvaluationResult()
    result.input_chars = len(statement_text)
    # Preserve complete source evidence — the raw record pages that
    # produced every digest fact and verification. This store lives
    # independently of prompt budgets so saved results always carry
    # full provenance.
    result.evidence_source = _pages_to_source(records)
    # Hard bound for LLM prompts (80k) — the 60k soft limit is enforced in the UI
    # with a warning + confirmation. Direct callers bypassing the UI still get
    # bounded prompts and an auditable warning in the report.
    prompt_statement, removed = _truncate_for_prompt(statement_text, EVALUATE_INTERNAL_MAX_CHARS)
    result.truncated_chars = removed
    if removed:
        result.truncation_warning = (
            f"Statement was {result.input_chars:,} characters — "
            f"{removed:,} characters beyond the {EVALUATE_INTERNAL_MAX_CHARS:,} internal prompt limit "
            f"were truncated and not analyzed. Claims at the end of the statement may have been missed. "
            f"Split the statement or shorten it and re-run for complete coverage."
        )
        # Layer a soft-limit note when the input also exceeded the 60k UI gate
        if result.input_chars > MAX_STATEMENT_CHARS:
            result.truncation_warning = (
                f"Statement was {result.input_chars:,} characters — "
                f"{result.input_chars - MAX_STATEMENT_CHARS:,} characters over the {MAX_STATEMENT_CHARS:,} "
                f"recommended limit. {removed:,} characters were truncated for the model prompts; "
                f"claims at the end (e.g., family impact, caregiver necessity) may have been missed. "
                f"Split the statement into smaller parts or shorten it and re-run."
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
            report(0.02, "Step 1/7 — Exhaustive review of medical records…")
            result.digest = review_medical_records(
                llm, records, progress=lambda f, m: progress((0.02 + f * 0.48), m) if progress else None
            )

    with tracing.phase_span("claims"), PhaseTimer(logger, "claims", request_id=rid):
        with phase_timer("claims"):
            report(0.52, "Step 2/7 — Extracting factual claims from the statement…")
            claims_data = llm.chat_json(
                CLAIMS_SYSTEM,
                CLAIMS_USER.format(
                    statement=sanitize_for_prompt(prompt_statement, max_chars=EVALUATE_INTERNAL_MAX_CHARS),
                    guard_note=GUARD_NOTE,
                ),
                phase="claims",
            )
            if not isinstance(claims_data, dict):
                raise LLMParseError("Claim extraction is incomplete: expected a JSON object.")
            result.claims = _normalize_claims(claims_data.get("claims"))
            for key in ("claimed_condition", "writer_role"):
                if not isinstance(claims_data.get(key, ""), str):
                    raise LLMParseError("Claim extraction returned invalid condition or writer metadata.")
            result.claimed_condition = claims_data.get("claimed_condition", "").strip()
            result.writer_role = claims_data.get("writer_role", "").strip()
            logger.info(
                "claims extracted count=%d condition=%s role=%s",
                len(result.claims), result.claimed_condition[:80] if result.claimed_condition else "-",
                result.writer_role or "-",
                extra={"request_id": rid, "phase": "claims", "status": "ok"},
            )

    with (
        tracing.phase_span("verify", claims=len(result.claims)),
        PhaseTimer(logger, "verify", request_id=rid, claims=len(result.claims)),
    ):
        with phase_timer("verify"):
            report(0.60, "Step 3/7 — Verifying each claim against the records…")
            assert result.digest is not None  # set by records:review above
            result.verifications, result.evidence_gaps = _verify_claims(
                llm, result.claims, result.digest, records, report
            )

    with tracing.phase_span("rubric"), PhaseTimer(logger, "rubric", request_id=rid):
        with phase_timer("rubric"):
            report(0.78, "Step 4/7 — Scoring against the lay-evidence rubric…")
            rubric_data = llm.chat_json(
                RUBRIC_SYSTEM_TEMPLATE.format(
                    rubric=load_knowledge("evaluation_rubric.md"),
                    legal=load_knowledge("legal_framework.md"),
                ),
                RUBRIC_USER.format(
                    statement=sanitize_for_prompt(prompt_statement, max_chars=EVALUATE_INTERNAL_MAX_CHARS),
                    verifications=sanitize_for_prompt(_verifications_text(result), max_chars=20_000),
                    digest_summary=sanitize_digest_text((result.digest.summary if result.digest else "") or "(no summary)", max_chars=20_000),
                    guard_note=GUARD_NOTE,
                ),
                phase="rubric",
            )
            result.scores = {k: float(v) for k, v in rubric_data.get("scores", {}).items()}
            result.rationales = rubric_data.get("rationales", {})
            result.improvements = rubric_data.get("improvements", [])
            result.omitted_record_facts = rubric_data.get("omitted_record_facts", [])
            result.executive_summary = rubric_data.get("executive_summary", "")

    with tracing.phase_span("topic"), PhaseTimer(logger, "topic", request_id=rid):
        with phase_timer("topic"):
            report(0.79, "Step 5/7 — Auditing topic coverage (hazards, care, family, progression)…")
            _analyze_topics(llm, result, statement_text, report)

    with tracing.phase_span("revision"), PhaseTimer(logger, "revision", request_id=rid):
        with phase_timer("revision"):
            report(0.86, "Step 6/8 — Drafting improvement suggestions and a revised statement…")
            _draft_revision(llm, result, statement_text, report)

    # The score/recommendations pass is LLM-backed like every other phase, so it
    # gets the same phase span treatment as its neighbours.
    with tracing.phase_span("score"), PhaseTimer(logger, "score", request_id=rid):
        with phase_timer("score"):
            report(0.92, "Step 7/8 — Computing effectiveness score and recommendations…")
            _score_and_recommend(llm, result, report, witness)

    with tracing.phase_span("report"), PhaseTimer(logger, "report", request_id=rid):
        with phase_timer("report"):
            report(0.96, "Step 8/8 — Building the report…")
            result.report_markdown = build_report(
                result, statement_text, citations=_citation_index_snapshot()
            )
    report(1.0, "Evaluation complete.")
    return result


def _citation_index_snapshot() -> list[dict[str, str]]:
    """Best-effort read of ``citation_index`` from session state.

    Guarded because this pipeline can run outside an active Streamlit script
    context (e.g. offline tests, CLI usage) where ``st.session_state`` raises
    instead of returning a default.
    """
    try:
        citations = st.session_state.get("citation_index", [])
    except Exception:  # noqa: BLE001 - session state may be unavailable
        return []
    return citations if isinstance(citations, list) else []


def _analyze_topics(
    llm: LLMService,
    result: EvaluationResult,
    statement_text: str,
    report: ProgressCallback,
) -> None:
    """Audit the statement against the topic checklist (A–O).

    Like the revision step, a failure here must not discard the completed
    evaluation, so errors are swallowed and the fields stay empty.
    """
    truncated_statement, _ = _truncate_for_prompt(statement_text, EVALUATE_INTERNAL_MAX_CHARS)
    try:
        topic_data = llm.chat_json(
            TOPIC_SYSTEM_TEMPLATE.format(
                checklist=load_knowledge("topic_checklist.md"),
                legal=load_knowledge("legal_framework.md"),
            ),
            TOPIC_USER.format(
                statement=sanitize_for_prompt(truncated_statement, max_chars=EVALUATE_INTERNAL_MAX_CHARS),
                verifications=sanitize_for_prompt(_verifications_text(result), max_chars=20_000),
                digest_summary=sanitize_digest_text(((result.digest.summary if result.digest else "") or "(no summary)")[:12000], max_chars=20_000),
                guard_note=GUARD_NOTE,
            ),
            phase="topic",
        )
    except LLMError as exc:
        logger.warning(
            "topic coverage unavailable error=%s",
            f"{type(exc).__name__}: {exc}",
            extra={"request_id": get_request_id() or "-", "phase": "topic", "status": "error", "error_class": type(exc).__name__},
        )
        result.topic_notes = "Topic coverage analysis unavailable — the model call failed."
        return
    result.topic_focus = topic_data.get("claim_focus", "")
    result.topic_rows = topic_data.get("topics", [])
    result.topic_critical_gaps = [str(g) for g in topic_data.get("critical_gaps", []) if str(g).strip()]
    result.topic_notes = topic_data.get("notes", "")
    report(0.85, "Topic coverage audited.")


def _draft_revision(
    llm: LLMService,
    result: EvaluationResult,
    statement_text: str,
    report: ProgressCallback,
) -> None:
    """Generate the itemized improvement plan and a suggested rewrite.

    A failure here should not lose the completed evaluation, so errors are
    swallowed and the revision fields simply stay empty.
    """
    import json as _json

    if result.topic_rows:
        topic_analysis = _json.dumps(
            {
                "claim_focus": result.topic_focus,
                "topics": result.topic_rows,
                "critical_gaps": result.topic_critical_gaps,
            },
            indent=1,
        )[:20000]
    else:
        topic_analysis = "(no topic coverage analysis available)"

    truncated_statement, _ = _truncate_for_prompt(statement_text, EVALUATE_INTERNAL_MAX_CHARS)
    try:
        revise_data = llm.chat_json(
            REVISE_SYSTEM,
            REVISE_USER.format(
                statement=sanitize_for_prompt(truncated_statement, max_chars=EVALUATE_INTERNAL_MAX_CHARS),
                verifications=sanitize_for_prompt(_verifications_text(result), max_chars=20_000),
                improvements=sanitize_for_prompt(_json.dumps(result.improvements, indent=1)[:6000] or "(none)", max_chars=10_000),
                omitted_facts=sanitize_for_prompt(_json.dumps(result.omitted_record_facts, indent=1)[:4000] or "(none)", max_chars=10_000),
                digest_summary=sanitize_digest_text(((result.digest.summary if result.digest else "") or "(no summary)")[:12000], max_chars=20_000),
                topic_analysis=sanitize_for_prompt(topic_analysis, max_chars=25_000),
                guard_note=GUARD_NOTE,
            ),
            max_tokens=6000,
            phase="revision",
        )
    except LLMError as exc:
        logger.warning(
            "revision draft unavailable error=%s",
            f"{type(exc).__name__}: {exc}",
            extra={"request_id": get_request_id() or "-", "phase": "revision", "status": "error", "error_class": type(exc).__name__},
        )
        result.revision_notes = "Revision draft unavailable — the model call failed."
        return
    result.revision_notes = revise_data.get("revision_notes", "")
    result.revision_changes = revise_data.get("changes", [])
    result.revised_statement = revise_data.get("revised_statement", "")
    result.added_facts_to_verify = [
        str(f) for f in revise_data.get("added_facts_to_verify", []) if str(f).strip()
    ]
    report(0.94, "Improvement suggestions drafted.")


def _contradiction_downgrade_reason(verification: dict, evidence_absent: bool) -> str:
    """Why a CONTRADICTED verdict cannot stand, or ``""`` when it can.

    A contradiction is an affirmative finding about the record, so it has to point at the
    record text that makes it — the same requirement the rubric, the legal framework and
    ``VERIFY_SYSTEM`` state for the model itself (absence of evidence is not substantive
    negative evidence: Horn v. Shinseki, 25 Vet. App. 231, 239 n.7 (2012); M21-1, Part V,
    Subpart ii, Ch. 1, § A). Two cases leave a CONTRADICTED verdict with no evidence:

    * nothing in this record set matches the claim (``evidence_absent``); or
    * the verifier named no conflicting record entry, so there is nothing to check.

    The second case is the one that survives the first test, and that is why it is checked
    separately: a batch can hold plenty of overlapping record text about a *different* fact
    and still receive a bare CONTRADICTED verdict for a claim the records never address.
    Accepting that verdict would penalise the claim's score, print a conflict in the report,
    and instruct the reviser to rewrite the claim to match records it never cited.
    """
    if str(verification.get("verdict", "")).upper() != "CONTRADICTED":
        return ""
    if evidence_absent:
        return (
            "no matching record text was retrieved for this claim, so it is a "
            "record-coverage gap rather than a contradiction."
        )
    if not str(verification.get("record_reference", "")).strip():
        return (
            "no conflicting record entry was cited, and a contradiction must point at the "
            "record text that makes it, so it is a record-coverage gap."
        )
    return ""


def _verify_claims(
    llm: LLMService,
    claims: list[dict],
    digest: MedicalDigest,
    records: list[ExtractedDocument],
    report: ProgressCallback,
) -> tuple[list[dict], list[dict]]:
    """Verify claims in small batches so each prompt stays focused.

    Returns ``(verifications, evidence_gaps)``. ``evidence_gaps`` lists the claims a
    CONTRADICTED verdict could not be substantiated for — either the batch found no
    raw record text at all, or the verifier cited no conflicting record entry. Both
    are downgraded to NOT FOUND, because "the records disagree" and "nothing in the
    records addresses this" are different findings and only one of them is true.
    Reporting a coverage gap as a contradiction would put a false statement in front
    of a veteran who is about to sign it (see
    :func:`_contradiction_downgrade_reason`).

    Each batch must pass schema, identity, uniqueness and coverage validation.
    Invalid batches are retried in full; exhaustion raises an explicit incomplete
    verification error before downstream scoring instead of inventing verdicts.
    """
    claims = _normalize_claims(claims)
    verdict_by_id: dict[int, dict] = {}
    evidence_gaps: list[dict] = []
    batch_size = 8
    batches = [claims[i : i + batch_size] for i in range(0, len(claims), batch_size)]
    claim_text_by_id = {c["id"]: str(c.get("text", "")) for c in claims}
    for index, batch in enumerate(batches, start=1):
        report(
            0.60 + 0.16 * index / max(len(batches), 1),
            f"Verifying claims — batch {index}/{len(batches)}…",
        )
        batch_query = " ".join(str(c.get("text", "")) for c in batch)
        evidence = retrieve_evidence(records, batch_query, top_k=8)
        # A contradiction needs something to contradict. Downgrade only when the
        # claims are judgeable (they carry content words), the raw records hold
        # nothing that overlaps them, AND the digested facts hold nothing either —
        # otherwise a claim the digest documents in different words would be
        # rewritten as a record gap, hiding a genuine conflict.
        overlap = digest.keyword_overlap(batch_query)
        evidence_absent = (
            query_has_content_words(batch_query) and evidence.weak and overlap == 0.0
        )
        gap_note = (
            "(No record text in this record set matches these claims, and the analyzed "
            "record summary contains nothing about them. Where the records simply do "
            "not address a claim, answer NOT FOUND — do not answer CONTRADICTED.)"
            if evidence_absent
            else ""
        )
        import json as _json

        prompt = VERIFY_USER.format(
            digest=sanitize_digest_text(digest.relevant_facts_text(batch_query, max_facts=150), max_chars=120_000),
            excerpts=sanitize_digest_text(evidence.text[:16000] or "(no matching raw excerpts found)", max_chars=20_000),
            claims=sanitize_for_prompt(_json.dumps(batch, indent=1), max_chars=20_000),
            guard_note=(GUARD_NOTE + "\n\n" + gap_note) if gap_note else GUARD_NOTE,
        )
        retry_note = ""
        for attempt in range(1, VERIFICATION_MAX_ATTEMPTS + 1):
            check_pipeline_cancelled()
            try:
                data = llm.chat_json(VERIFY_SYSTEM, prompt + retry_note, phase="verify")
                verified = _normalize_verifications(data, {c["id"] for c in batch})
                break
            except LLMParseError as exc:
                if attempt == VERIFICATION_MAX_ATTEMPTS:
                    raise VerificationIncompleteError(
                        f"Verification incomplete: batch {index}/{len(batches)} did not return "
                        f"exactly one valid verdict per claim after {attempt} attempts. "
                        "These claims remain unverified, not NOT FOUND. Evaluation stopped "
                        "before scoring or rewriting; please re-run."
                    ) from exc
                # Do not quote a malformed model response (which can contain
                # medical text) in logs or the correction instruction.
                logger.warning(
                    "verification batch %d/%d invalid; retrying attempt %d/%d",
                    index, len(batches), attempt + 1, VERIFICATION_MAX_ATTEMPTS,
                    extra={"phase": "verify", "status": "retry"},
                )
                retry_note = (
                    "\n\nThe previous response was incomplete or invalid. Return a fresh "
                    "complete verifications list with exactly one valid verdict per "
                    "submitted id, and no other ids. Include string record_reference "
                    "and note fields for every verdict."
                )
        for item in verified:
            claim_id = item["id"]
            downgrade_reason = _contradiction_downgrade_reason(item, evidence_absent)
            if downgrade_reason:
                item = dict(item)
                item["verdict"] = "NOT FOUND"
                item["note"] = (
                    f"{str(item.get('note', '')).strip()} "
                    f"[Downgraded from CONTRADICTED: {downgrade_reason}]"
                ).strip()
                evidence_gaps.append(
                    {
                        "id": claim_id,
                        "claim": claim_text_by_id.get(claim_id, ""),
                        "reason": downgrade_reason,
                    }
                )
            verdict_by_id[claim_id] = item
    return (
        [verdict_by_id[c["id"]] for c in claims],
        evidence_gaps,
    )


# Feature: Evidence Strength Dashboard (F3.S1)
VERDICTS: tuple[str, ...] = ("SUPPORTED", "PARTIALLY SUPPORTED", "CONTRADICTED", "NOT FOUND")

DEFAULT_RECORD_TYPE = "Other"

# Simple keyword/regex inference of the "record type" a claim most likely
# relates to. Order matters — the first matching pattern wins, so more
# specific categories (diagnosis, medication) are checked before the
# broader "symptom" bucket.
_RECORD_TYPE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "Diagnosis",
        re.compile(
            r"\b(diagnos\w*|scan|x-ray|mri|ct\s*scan|biopsy|lab\s*(test|result)s?|"
            r"bloodwork|imaging)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Medication",
        re.compile(
            r"\b(medication\w*|prescri\w*|dosage|dose|milligram\w*|\bmg\b|pill\w*|"
            r"tablet\w*|\bdrug\w*|refill\w*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Symptom",
        re.compile(
            r"\b(pain|ache\w*|symptom\w*|nause\w*|dizz\w*|fatigue\w*|numbness|"
            r"anxiety|depress\w*|insomnia|tremor\w*|swelling|headache\w*)\b",
            re.IGNORECASE,
        ),
    ),
)


def _infer_record_type(claim_text: str) -> str:
    """Infer the medical record type a claim most plausibly relates to.

    Uses simple keyword regexes over the claim text (per the story's
    implementation notes) — diagnostic/imaging language, medication
    language, and symptom language are checked in that order; anything
    that matches none of them falls back to :data:`DEFAULT_RECORD_TYPE`.
    """
    for record_type, pattern in _RECORD_TYPE_PATTERNS:
        if pattern.search(claim_text):
            return record_type
    return DEFAULT_RECORD_TYPE


def build_evidence_dashboard(
    verifications: list[dict[str, Any]],
    claims: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Aggregate claim verdicts by inferred record type for the dashboard.

    Groups every verified claim under an inferred record type (Diagnosis,
    Medication, Symptom, or Other — see :func:`_infer_record_type`) and
    tallies verdict counts within each group. Every record type present is
    given a full ``{verdict: count}`` mapping across all four verdicts
    (zero-filled where absent) so chart-building code never has to guard
    against missing keys.

    Returns an empty dict when there are no verifications (e.g. a run that
    produced zero extracted claims) — callers must treat that as "nothing
    to render" rather than an error.
    """
    claim_text_by_id: dict[Any, str] = {c.get("id"): str(c.get("text", "")) for c in claims}
    dashboard: dict[str, dict[str, int]] = {}
    for verification in verifications:
        claim_text = claim_text_by_id.get(verification.get("id"), "")
        record_type = _infer_record_type(claim_text)
        verdict = str(verification.get("verdict") or "NOT FOUND")
        if verdict not in VERDICTS:
            verdict = "NOT FOUND"
        counts = dashboard.setdefault(record_type, {v: 0 for v in VERDICTS})
        counts[verdict] += 1
    return dashboard


def _verifications_text(result: EvaluationResult) -> str:
    """One line per verdict, carrying the claim's basis of knowledge when it is known.

    The basis travels with the finding because the rubric, the topic audit and the
    reviser all have to know whether the writer experienced, saw, was told about, or
    concluded the fact. Without it, the claim text alone cannot show whether a silent
    record leaves competent lay evidence intact or leaves a relay unproven (see
    ``CLAIM_BASES``); a claim whose basis is unknown simply omits the field.
    """
    lines = []
    claim_text = {c["id"]: c.get("text", "") for c in result.claims}
    claim_basis = {c["id"]: c.get("basis", UNKNOWN_CLAIM_BASIS) for c in result.claims}
    for v in result.verifications:
        line = (
            f"- Claim {v.get('id')}: \"{claim_text.get(v.get('id'), '')}\" => "
            f"{v.get('verdict')} | ref: {v.get('record_reference', '')} | {v.get('note', '')}"
        )
        basis = claim_basis.get(v.get("id"), UNKNOWN_CLAIM_BASIS)
        if basis != UNKNOWN_CLAIM_BASIS:
            line += f" | basis: {basis}"
        lines.append(line)
    return "\n".join(lines) or "(no claims extracted)"


# ------------------------------------------------------ effectiveness score
# Weights are exact per .implement/functional-spec.md: rubric dimension
# scores 40%, claim verdict distribution 30%, evidence density (citations per
# claim) 20%, statement length/complexity 10%.
_RUBRIC_WEIGHT = 0.40
_VERDICT_WEIGHT = 0.30
_DENSITY_WEIGHT = 0.20
_LENGTH_WEIGHT = 0.10

_VERDICT_POINTS = {
    "SUPPORTED": 1.0,
    "PARTIALLY SUPPORTED": 0.5,
    "NOT FOUND": 0.0,
    "CONTRADICTED": -1.0,
}

_LENGTH_FLOOR_CHARS = 300
_LENGTH_CEILING_CHARS = 40_000


def compute_score_band(score: int) -> str:
    """Color band for an effectiveness score: green >75, yellow 50-75, red <50."""
    if score > 75:
        return "green"
    if score >= 50:
        return "yellow"
    return "red"


def _rubric_component(result: "EvaluationResult") -> float:
    """0-100 rubric component: mean of the 8 dimension scores (each 0-10)."""
    if not result.scores:
        return 0.0
    values = list(result.scores.values())
    avg = sum(values) / len(values)
    return max(0.0, min(100.0, avg * 10.0))


def _verdict_component(result: "EvaluationResult") -> float:
    """0-100 claim-verdict-distribution component.

    SUPPORTED claims boost the score and CONTRADICTED claims penalize it;
    NOT FOUND is neutral — absence from records is not negative evidence
    (Horn v. Shinseki; M21-1 V.ii.1.A). With zero verifications this
    is the neutral midpoint (50) rather than dividing by zero.
    """
    if not result.verifications:
        return 50.0
    points = [_VERDICT_POINTS.get(str(v.get("verdict", "")), 0.0) for v in result.verifications]
    avg = sum(points) / len(points)  # in [-1, 1]
    return max(0.0, min(100.0, (avg + 1.0) * 50.0))


def _evidence_density_component(result: "EvaluationResult") -> float:
    """0-100 evidence-density component: share of claims with a cited record reference.

    With zero claims there is no evidence to cite, so this is defined as 0
    rather than dividing by zero — ``generate_improvement_recommendations``
    surfaces that gap explicitly.
    """
    if not result.claims:
        return 0.0
    cited = sum(1 for v in result.verifications if str(v.get("record_reference", "")).strip())
    return max(0.0, min(100.0, (cited / len(result.claims)) * 100.0))


def _length_complexity_component(result: "EvaluationResult") -> float:
    """0-100 length/complexity component derived from the raw statement length.

    Very short statements rarely carry enough detail to be persuasive, so the
    component ramps up from 0 below the floor; very long statements are still
    fully credited here (truncation is handled/warned about elsewhere).
    """
    chars = result.input_chars
    if chars <= 0:
        return 0.0
    if chars < _LENGTH_FLOOR_CHARS:
        return max(0.0, (chars / _LENGTH_FLOOR_CHARS) * 60.0)
    if chars > _LENGTH_CEILING_CHARS:
        return 70.0
    return 100.0


def compute_effectiveness_score(result: "EvaluationResult") -> int:
    """Weighted 0-100 effectiveness score for a completed evaluation.

    Weights (exact, per functional spec): rubric dimension scores 40%, claim
    verdict distribution 30%, evidence density 20%, statement length/
    complexity 10%. Always returns an integer in [0, 100], including when
    *result* carries zero claims (verdict/density components fall back to
    defined neutrals instead of raising ``ZeroDivisionError``).
    """
    weighted = (
        _rubric_component(result) * _RUBRIC_WEIGHT
        + _verdict_component(result) * _VERDICT_WEIGHT
        + _evidence_density_component(result) * _DENSITY_WEIGHT
        + _length_complexity_component(result) * _LENGTH_WEIGHT
    )
    return int(round(max(0.0, min(100.0, weighted))))


RECOMMENDATIONS_SYSTEM = """You are a veterans-claims advocate advising a witness how to raise \
the effectiveness of their lay statement before submission. Recommend the highest-impact, most \
concrete edits — grounded strictly in the evaluation results provided. NEVER invent facts."""

RECOMMENDATIONS_USER = """Based on the completed evaluation below, propose 3 to 5 ranked \
improvement recommendations, ordered by expected score impact (highest first).

Return JSON:
{{
  "recommendations": [
    {{
      "title": "short, specific action (e.g., 'Add supporting evidence for onset claim')",
      "impact": "estimated score impact, e.g. '+8 points'",
      "explanation": "1-2 sentences explaining why this matters",
      "claim_id": <id of the related claim if this recommendation targets one specific claim, else null>
    }}
  ]
}}
Return between 3 and 5 recommendations. If the statement has few or no factual claims, \
recommend adding verifiable, specific factual assertions the witness can support from personal \
knowledge — do not fabricate any.

EFFECTIVENESS SCORE: {score}/100

CLAIM VERIFICATION SUMMARY:
<<<
{verifications}
>>>

RUBRIC IMPROVEMENTS IDENTIFIED:
<<<
{improvements}
>>>

TOPIC COVERAGE GAPS:
<<<
{topic_gaps}
>>>

AID & ATTENDANCE INTAKE COVERAGE:
<<<
{care_gaps}
>>>

{guard_note}"""


def _fallback_recommendations(result: "EvaluationResult", minimum: int) -> list[dict]:
    """Deterministic, data-driven recommendations derived from already-computed fields.

    Used when the LLM call fails or returns fewer than the required minimum,
    so ``generate_improvement_recommendations`` always satisfies the "3-5
    items" acceptance criterion — including the zero-claims case, where the
    recommendations must reflect missing evidence rather than invent content.
    """
    candidates: list[dict] = []
    if not result.claims:
        candidates.append(
            {
                "title": "Add specific, verifiable factual claims",
                "impact": "+15 points",
                "explanation": (
                    "No checkable factual assertions were found in the statement. Add concrete "
                    "dates, events, symptoms, or treatments the witness personally observed."
                ),
                "claim_id": None,
            }
        )
        candidates.append(
            {
                "title": "Cite supporting medical record evidence",
                "impact": "+10 points",
                "explanation": (
                    "There are no claims yet to cite evidence for — once factual claims are "
                    "added, cite the record source (filename + page) for each where available."
                ),
                "claim_id": None,
            }
        )
    if result.contradiction_count:
        candidates.append(
            {
                "title": "Resolve contradictions with the medical records",
                "impact": "+10 points",
                "explanation": (
                    f"{result.contradiction_count} claim(s) conflict with the medical records — "
                    "correct them or add a written explanation before submitting."
                ),
                "claim_id": None,
            }
        )
    for gap in result.topic_critical_gaps:
        candidates.append(
            {
                "title": f"Address gap: {gap}"[:160],
                "impact": "+5 points",
                "explanation": "This checklist topic is applicable but weakly covered in the statement.",
                "claim_id": None,
            }
        )
    for imp in result.improvements:
        candidates.append(
            {
                "title": (str(imp.get("problem", "")) or "Improve statement detail")[:160],
                "impact": "+5 points",
                "explanation": str(imp.get("suggestion", "")) or "Add more specific, record-grounded detail.",
                "claim_id": None,
            }
        )
    generic_fillers = [
        {
            "title": "Improve overall specificity and detail",
            "impact": "+3 points",
            "explanation": "Add concrete dates, frequencies, and specific incidents where possible.",
            "claim_id": None,
        },
        {
            "title": "Strengthen functional-impact descriptions",
            "impact": "+3 points",
            "explanation": "Describe a specific daily activity the condition limits, not just the diagnosis.",
            "claim_id": None,
        },
        {
            "title": "Add clearer before/after timeline detail",
            "impact": "+3 points",
            "explanation": "Contrast functioning before and after onset so continuity is unmistakable.",
            "claim_id": None,
        },
    ]
    for filler in generic_fillers:
        if len(candidates) >= minimum:
            break
        candidates.append(filler)
    return candidates[:5]


def _parse_recommendation_items(data: Any) -> list[dict]:
    """Extract and normalize recommendation entries from a parsed LLM response."""
    raw = data.get("recommendations", []) if isinstance(data, dict) else []
    items: list[dict] = []
    seen_titles: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title", "")).strip()
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        claim_id = entry.get("claim_id")
        items.append(
            {
                "title": title[:160],
                "impact": str(entry.get("impact", "")).strip()[:40] or "+0 points",
                "explanation": str(entry.get("explanation", "")).strip()[:400],
                "claim_id": claim_id if isinstance(claim_id, int) else None,
            }
        )
    return items


def _pad_recommendations(items: list[dict], result: "EvaluationResult", minimum: int) -> list[dict]:
    """Top up *items* to *minimum* using data-driven fallbacks, de-duplicated by title."""
    seen = {it["title"] for it in items}
    for fallback in _fallback_recommendations(result, minimum):
        if len(items) >= minimum:
            break
        if fallback["title"] in seen:
            continue
        seen.add(fallback["title"])
        items.append(fallback)
    return items


def generate_improvement_recommendations(
    result: "EvaluationResult", llm: LLMService, witness: dict[str, str] | None = None,
) -> list[dict]:
    """Generate 3-5 ranked improvement recommendations via exactly one LLM call.

    Each item has ``title``, ``impact``, and ``explanation`` (plus an
    optional ``claim_id`` the UI can use to jump to the matching claim). Reuses
    the already-computed verifications/rubric/topic fields on *result* — no
    additional record or statement text is sent to the model. Pads with
    data-driven (never invented) fallback recommendations if the model
    returns fewer than 3 items, so this always returns 3-5 items, including
    when *result* carries zero claims.
    """
    import json as _json

    topic_gaps = "\n".join(f"- {g}" for g in result.topic_critical_gaps) or "(none identified)"
    improvements_text = (
        _json.dumps(result.improvements[:6], indent=1) if result.improvements else "(none)"
    )
    # Structured A&A intake answers the statement does not yet address. When
    # the caller supplied none, the section says so explicitly — the slot in
    # RECOMMENDATIONS_USER is a fixed part of the prompt; what varies is its
    # content.
    care_gaps = care_gaps_text(witness or {}) or "(no structured intake answers were provided)"
    data = llm.chat_json(
        RECOMMENDATIONS_SYSTEM,
        RECOMMENDATIONS_USER.format(
            score=compute_effectiveness_score(result),
            verifications=sanitize_for_prompt(_verifications_text(result), max_chars=8_000),
            improvements=sanitize_for_prompt(improvements_text, max_chars=4_000),
            topic_gaps=sanitize_for_prompt(topic_gaps, max_chars=2_000),
            care_gaps=care_gaps,
            guard_note=GUARD_NOTE,
        ),
        phase="recommendations",
    )
    items = _parse_recommendation_items(data)
    if len(items) < 3:
        items = _pad_recommendations(items, result, 3)
    return items[:5]


def _score_and_recommend(
    llm: LLMService, result: "EvaluationResult", report: ProgressCallback,
    witness: dict[str, str] | None = None,
) -> None:
    """Compute the effectiveness score and improvement recommendations.

    Telemetry call sites for the Statement Effectiveness Score & Improvement
    Recommendations feature (feature id EFFECTIVENESS_FEATURE_ID) live here,
    at the compute/generate boundary: a ``goal`` event on completion carrying
    scoreValue/scoreBand/recommendationCount, and a ``feature.error`` event
    if recommendation generation fails (score computation itself never
    raises). A failure here must not discard the completed evaluation.
    """
    result.effectiveness_score = compute_effectiveness_score(result)
    try:
        result.recommendations = generate_improvement_recommendations(result, llm, witness)
    except Exception as exc:  # noqa: BLE001 - feature-error boundary
        logger.warning(
            "recommendation generation unavailable error=%s",
            f"{type(exc).__name__}: {exc}",
            extra={
                "request_id": get_request_id() or "-",
                "phase": "recommendations",
                "status": "error",
                "error_class": type(exc).__name__,
            },
        )
        try:
            track_feature_error(EFFECTIVENESS_FEATURE_ID, exc, phase="recommendations")
        except Exception:  # noqa: BLE001 - telemetry must never break the pipeline
            pass
        result.recommendations = _fallback_recommendations(result, 3)[:5]

    try:
        track_goal(
            EFFECTIVENESS_FEATURE_ID,
            "effectiveness score computed",
            scoreValue=result.effectiveness_score,
            scoreBand=compute_score_band(result.effectiveness_score),
            recommendationCount=len(result.recommendations),
        )
    except Exception:  # noqa: BLE001 - telemetry must never break the pipeline
        pass
    report(0.965, "Effectiveness score and recommendations ready.")


_VERDICT_EMOJI = {
    "SUPPORTED": "✅",
    "PARTIALLY SUPPORTED": "🟡",
    "CONTRADICTED": "❌",
    "NOT FOUND": "⚪",
}


# Verdicts in which the record set backs the statement rather than contradicting it.
_SUPPORTIVE_VERDICTS = ("SUPPORTED", "PARTIALLY SUPPORTED")


def rubric_and_positive_sources(
    digest: MedicalDigest | None,
    verifications: list[dict] | None,
) -> tuple[set[str], set[str]]:
    """Cross-reference digest facts against the verification results.

    ``MedicalFact`` now carries the ``document``/``page`` it was read from, and a
    verification's ``record_reference`` names its source too, so the link the fact
    export filters need can be an **exact join** instead of a substring guess: both
    sides are parsed to ``(document, page)`` and matched. The original
    case-insensitive substring test is kept as a union (not a fallback), because a
    free-form reference such as ``"records.pdf p.7, 2020-01-01"`` carries real
    information that a strict parse would throw away.

    Returns ``(cited, positive)``: sets of ``fact.source`` strings that appear in
    any verification, and in a verification whose verdict is supportive (the record
    corroborates the statement rather than contradicting it).
    """
    cited: set[str] = set()
    positive: set[str] = set()
    facts = getattr(digest, "facts", None) or []
    if not facts or not verifications:
        return cited, positive

    def _page_key(pages: str) -> str:
        match = re.match(r"\d+", pages or "")
        return match.group(0) if match else ""

    exact: dict[tuple[str, str], set[str]] = {}
    loose: list[tuple[str, str]] = []
    for verification in verifications:
        reference = str(verification.get("record_reference", "") or "").strip()
        if not reference:
            continue
        verdict = str(verification.get("verdict", "") or "")
        document, pages = parse_source(reference)
        page = _page_key(pages)
        if document and page:
            exact.setdefault((document.strip().casefold(), page), set()).add(verdict)
        loose.append((reference.casefold(), verdict))

    for fact in facts:
        source = (fact.source or "").strip()
        if not source:
            continue
        verdicts: set[str] = set()
        document, page = fact.document, str(fact.page or "")
        if not (document and page):
            document, pages = parse_source(source)
            page = _page_key(pages)
        if document and page:
            verdicts |= exact.get((document.strip().casefold(), page), set())
        source_lower = source.casefold()
        verdicts |= {
            verdict for reference, verdict in loose if source_lower in reference
        }
        if verdicts:
            cited.add(fact.source)
            if verdicts & set(_SUPPORTIVE_VERDICTS):
                positive.add(fact.source)
    return cited, positive


def coverage_lines(digest: MedicalDigest) -> list[str]:
    """Markdown lines describing what was read, what was not, and how citations held.

    Public because the report and the results panel must not be able to disagree
    about coverage: a partial review is exactly the thing a user needs to notice,
    and it should read the same wherever they look.
    """
    lines: list[str] = []
    if digest.pages_in_files and digest.unreadable_pages:
        missing_pct = round((1.0 - digest.coverage_ratio) * 100)
        lines.append(
            f"> ⚠️ **Partial coverage:** {digest.unreadable_pages:,} of "
            f"{digest.pages_in_files:,} source pages ({missing_pct}%) had no extractable "
            "text (image-only scans) and were not analyzed. OCR those pages and re-run "
            "for a complete review."
        )
    elif digest.pages_in_files:
        lines.append(f"**Coverage:** all {digest.pages_in_files:,} source page(s) read.")
    if digest.chunks_without_facts:
        lines.append(
            f"- {digest.chunks_without_facts} of {digest.chunks_reviewed} analyzed chunks "
            "contained no extractable facts."
        )
    if digest.facts_dropped_by_cap:
        lines.append(
            f"> ⚠️ **Digest capped:** {digest.facts_dropped_by_cap:,} consolidated fact(s) "
            "were dropped by a previous review's storage cap "
            "(VA_LSE_MAX_DIGEST_FACTS). This saved result is incomplete. "
            "Re-run the source records to regenerate it with full evidence retention."
        )
    if digest.corroborated_pages:
        names = ", ".join(
            f"{row.get('document')} p.{row.get('page')}" for row in digest.corroborated_pages[:5]
        )
        more = " …" if len(digest.corroborated_pages) > 5 else ""
        lines.append(
            f"- {len(digest.corroborated_pages):,} page(s) appear in more than one source "
            f"(corroborated): {names}{more}"
        )
    check = digest.citation_check or {}
    if check.get("checked"):
        lines.append(
            f"- Citations self-checked: {check['checked']:,} quote(s) matched against the "
            f"page they cite; {check.get('missing', 0):,} not found."
        )
        for example in (check.get("examples") or [])[:3]:
            lines.append(
                f"  - ⚠️ quote not found on {example.get('document')} "
                f"p.{example.get('page')}: “{str(example.get('quote', ''))[:120]}”"
            )
    return lines


def build_report(
    result: EvaluationResult,
    statement_text: str,
    citations: list[dict[str, str]] | None = None,
) -> str:
    """Render the full markdown evaluation report.

    ``citations`` (F2.S2) is the ``citation_index`` collected via the medical
    record search widget (excerpt + source per entry); when non-empty, a
    "Sources" section listing every citation is appended to the report.
    """
    lines: list[str] = []
    lines.append("# Lay Statement Evaluation Report")
    lines.append("")
    if result.truncation_warning:
        lines.append(f"> ⚠️ **Truncated input:** {result.truncation_warning}")
        lines.append("")
    lines.append(f"**Overall rating: {result.overall_rating}**")
    lines.append(
        f"**Effectiveness score: {result.effectiveness_score}/100 "
        f"({result.score_band.upper()})**"
    )
    claimed: str = result.claimed_condition  # narrow type for mypy
    if claimed:
        lines.append(f"**Appears to support claim for:** {claimed}")
    if result.writer_role:
        lines.append(f"**Writer role:** {result.writer_role}")
    if result.digest:
        lines.append(
            f"**Records reviewed:** {result.digest.pages_reviewed:,} pages in "
            f"{result.digest.chunks_reviewed} chunks "
            f"({result.digest.duplicates_skipped} duplicate page(s) skipped), "
            f"{len(result.digest.facts):,} facts extracted"
        )
        lines.extend(coverage_lines(result.digest))
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

    if result.evidence_gaps:
        lines.append("## Record Coverage Gaps (not contradictions)")
        lines.append("")
        lines.append(
            "These claims had no matching text anywhere in the uploaded records, so the "
            "verification had nothing to check them against. That is a gap in the record "
            "set — or in what could be read from it — not a statement that the records "
            "disagree. If the records exist, add them (or OCR the unreadable pages) and "
            "re-run before relying on this section."
        )
        lines.append("")
        for evidence_gap in result.evidence_gaps:
            lines.append(f"- {evidence_gap.get('claim', '')}")
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
        "often legitimately do not appear in medical records, and the absence of evidence is "
        "not negative evidence (Horn v. Shinseki; M21-1 V.ii.1.A; Buchanan v. Nicholson). Where "
        "the records hold no measurement of painful motion, functional loss or flare-ups, a "
        "normal examination does not contradict them either (38 C.F.R. §§ 4.40, 4.45, 4.59; "
        "DeLuca v. Brown)._"
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

    if result.recommendations:
        lines.append("## Improvement Recommendations (ranked by estimated impact)")
        lines.append("")
        for index, rec in enumerate(result.recommendations, start=1):
            lines.append(f"**{index}. {rec.get('title', '')}** ({rec.get('impact', '')})")
            lines.append(f"   - {rec.get('explanation', '')}")
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

    if result.topic_rows:
        lines.append("## Topic Coverage — What the Statement Does and Does Not Address")
        lines.append("")
        if result.topic_focus:
            lines.append(f"**Claim focus:** {result.topic_focus}")
            lines.append("")
        lines.append("| Topic | Applicable | Coverage | Evidence in statement | How to strengthen |")
        lines.append("| --- | --- | --- | --- | --- |")
        for t in result.topic_rows:
            lines.append(
                f"| {t.get('topic', '')} | {'Yes' if t.get('applicable') else 'No'} "
                f"| {t.get('coverage', '')} | {t.get('evidence', '')} | {t.get('gap_note', '')} |"
            )
        lines.append("")
        if result.topic_critical_gaps:
            lines.append("**Critical gaps — the highest-impact topics to address:**")
            for gap in result.topic_critical_gaps:
                lines.append(f"- {gap}")
            lines.append("")
        if result.topic_notes:
            lines.append(f"_{result.topic_notes}_")
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
            for fact_str in result.added_facts_to_verify:
                lines.append(f"- {fact_str}")
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

    if citations:
        lines.append("## Sources")
        lines.append("")
        for citation in citations:
            source = str(citation.get("source", "")).strip() or "(unknown source)"
            excerpt = str(citation.get("excerpt", "")).strip()
            lines.append(f"- **{source}**: {excerpt}")
        lines.append("")
        try:
            track_goal(
                SEARCH_FEATURE_ID,
                "report_sources_appended",
                citation_count=len(citations),
            )
        except Exception:  # noqa: BLE001 - telemetry must never break report generation
            pass

    lines.append("---")
    lines.append(
        "_This report was generated by an automated tool as a drafting aid. It is not legal, "
        "medical, or claims advice. Consult an accredited VSO, claims agent, or attorney "
        "(www.va.gov/ogc/apps/accreditation) before submitting evidence._"
    )
    return "\n".join(lines)
