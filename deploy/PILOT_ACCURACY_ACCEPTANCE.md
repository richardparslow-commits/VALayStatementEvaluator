# R10 — Synthetic factual accuracy acceptance

**R10 remains open. Real-information admission remains NO-GO.** The seed corpus
and offline tool prepare evidence; they do not demonstrate actual model quality.
No provider/model or reviewer is approved by this document. The operator has
suggested zero tolerated unflagged critical changes, without recording the other
approvals. Do not fill `accuracy_validation` with a CI result, draft plan, matching
quote, successful tool exit or fabricated fixture result.

## Proposed corpus and thresholds

`benchmarks/synthetic-accuracy-v1.json` contains 12 entirely invented cases and
74 proposed checkpoints. Each case runs through both evaluation and drafting:
24 actions and 148 checkpoints for one repetition. Two reviewers independently
judge every checkpoint, producing 296 judgments. Repetitions are agreed before
collection and increase the matrix; none are silently discarded.

The scenarios cover chronology, negation, side, uncertain onset, attribution,
multiple conditions, silent records, an unreadable scan, duplicate findings,
document prompt injection, deliberately truncated output, and numbers/tentative
diagnoses. Checkpoints refer to exact original account/source spans. Review
extraction, findings, A–O topic applicability, both initial and revised/final text,
source support, critical integrity, missing-page coverage and failure handling.
Inspect summaries, score rationales, grounding and other derived fields too.
The truncation case is explicitly a fault-injection check, separate from claims
about ordinary model performance. Its actual provider attempts and the injected
cut/retry evidence must both be preserved.

The seed is **unreviewed**, not a reference standard. An independent VA/evidence
reviewer and medical-record reviewer must correct its expectations and ambiguity
before agreeing to it. The generic topic checks deliberately avoid inventing
legal determinations. R11 legal/authority review is a separate gate. A reviewer
may need additional cases, explicit claim-level expected verdicts or A–O rows;
add them before freezing and recollect the entire agreed matrix after changes.
The v1 scenario/dimension taxonomy is fixed to the listed categories. Additional
cases/checkpoints can use those categories; a new category/dimension requires a
reviewed assessor source change, a new frozen plan and fresh acceptance evidence.
These small cases do not establish large-document, condition-specific or
population-wide accuracy. Include additional scale, domain and held-out cases
before claiming those capabilities.

Proposed acceptance is zero failed checkpoints, false contradictions, missed
critical facts, and unflagged critical alterations in text treated as reviewed or
final. Inspect dates, numbers, side, negation, attribution, uncertainty and
diagnoses in context. A warning elsewhere does not necessarily prevent a user
from treating altered final text as reviewed: inspect actual comparison/review
status and export behavior. A visibly unresolved alteration must still be judged
against the applicable checkpoint; it is not automatically correct.

The tool reports sums of **adjudicated checkpoint counts**, not deduplicated error
events, statistical rates or a guarantee of no future errors. With proposed zero
thresholds, any positive count fails. Broader metrics and thresholds must be
defined by the reviewers before extending/changing this policy and tool.

## Independent people and non-secret configuration

Appoint three distinct people, independent of implementation/seed authors:

- Evidence reviewer: a VA-accredited representative/claims agent/attorney or an
  independently qualified VA evidence specialist; document qualifications.
- Medical reviewer: a clinician or qualified medical-record reviewer experienced
  in dates, attribution, diagnostic uncertainty and clinical documentation.
- QA reviewer: coordinates completeness, preserves disagreements and signs the
  final adjudication of each disputed checkpoint with a reason.

Verify actual qualifications, independence and signatures outside this tool.
Agree the corrected corpus, matrix, thresholds, immutable application revision,
provider privacy terms, spending envelope and request profiles **before** calls.
Collect only these invented inputs. Do not use real veteran records or paste
credentials, authentication claims, raw headers or real names into evidence.

Use the exact intended pilot provider/account/region, all intended main/fast
model versions, and effective parameters for every phase and retry. Do not infer
approval from defaults or switch models after failures. If a provider offers only
a mutable alias, document returned identity and a dated provider version/change
reference and the operator's explicit limitation acceptance; retest on changes.
An alias string alone does not prove an immutable backend version. Tools/search
and fallbacks stay disabled in this proposed benchmark profile. Any future
fallback needs its own approved matrix before being enabled in the pilot.

`benchmarks/accuracy-configuration.example.json` is intentionally incomplete and
cannot pass assessment. Fill its non-secret fields in a private local copy:
`provider`, `base_url`, `account_reference`, `region_reference`,
`approval_reference`, and `request_profiles`. Each profile has `id`, `model`,
`version_reference`, and a nonempty `parameters` object containing the complete
effective request settings (including output cap, temperature/reasoning where
used and tier). Each profile also requires a `phase` name and nonempty distinct
`required_pathways` list (`evaluate`, `draft`, or both). Different phase settings
need different profiles. Every required profile/pathway pair must actually be
exercised; requests must name the matching phase. Keep `tools`
and `fallbacks` empty. Never include API keys or environment/secret dumps.

The existing $250 pilot proposal is not new approval for a benchmark's costs.
Approve a separate bounded test envelope or explicitly allocate it within that
ceiling; include retries and output/reasoning charges. All approved actions must
fit the actual account cutoff and per-attempt bound. A 24-action matrix can
require many provider attempts; do not assume it fits from the number of cases.
There is no paid collection command in this tool and no live CI benchmark.

## Freeze the plan offline

Use a clean checkout of the exact committed revision to be evaluated. Store
local artifacts under `accuracy-evidence/` (excluded from Git and Docker), with
operator-approved permissions, retention and deletion. The source snapshot
records revision/tree and SHA-256 of every tracked file under `app/` (including
runtime JSON data), plus runtime lock bytes. Full modules bind prompt templates and pipeline code without
loading settings, clients, `.env` or secrets. Tracked bytes must match Git objects;
untracked nonignored code or changed knowledge prevents freezing.

```sh
mkdir -m 700 accuracy-evidence
python scripts/accuracy_benchmark.py prepare \
  --corpus benchmarks/synthetic-accuracy-v1.json \
  --configuration accuracy-evidence/configuration.json \
  --repetitions 1 --out accuracy-evidence/plan.json
```

The command only freezes a **draft plan**, even with blank example configuration.
It makes no approvals or provider calls. Corrected corpus/configuration,
repetitions or source changes produce a new plan and need new agreements/results.
Preserve original plans and failed runs; outputs are created exclusively and never
overwritten. Hashes use canonical JSON (sorted keys, ASCII, compact separators,
finite numbers); `app.accuracy_benchmark.digest` is the shared implementation.

## Collect and retain actual application evidence

A trusted operator/QA collector must run the frozen application's
`run_evaluation` and `run_draft` entry points using the approved actual LLM client,
with approved transport, privacy and spending safeguards. Supply `inputs.account`
as statement/observations and the exact witness, condition and claim type. Preserve
each labeled record as its typed page/block; the empty scan stays unreadable.
Use the full dataclass result (`dataclasses.asdict`), including digest, provenance,
factual comparison/review, initial/final text, topics, scores and incomplete status.
On failure preserve the partial result and safe failure evidence; never omit or
replace a failed action with only its successful retry. Inspect actual UI/export
behavior when deciding whether anything could be treated as reviewed/final.

Capture **every** provider attempt/retry with its actual effective request profile,
returned model/version reference, unique provider request ID and complete synthetic
system/user prompts and response. Independently verify raw provider/account
evidence and attempted work against the collection log. Do not use cached/fake
outputs or pass reference expectations to the model. A separate synthetic fault
collector must document the truncation mechanism and recovery. This repository
does not supply or run that paid collector: selection/configuration/approval and
actual collection remain required R10 work.

The offline assessor accepts this evidence contract:

| Artifact | Required fields |
|---|---|
| All | `schema_version: 1`, `plan_sha256` |
| Agreement | `signed_at` (timezone timestamp); nonempty `operator_reference`, `corpus_review_reference`, `threshold_approval_reference`, `provider_approval_reference`, `budget_approval_reference`; `reviewers` with exactly three distinct IDs/roles `evidence`, `medical`, `qa`, each `independent: true`, `qualification_reference`, `signature_reference` |
| Results | `agreement_sha256`; `runs` with exactly one entry for each case × pathway (`evaluate`/`draft`) × repetition (1-based) |
| Each run | `case_id`, `pathway`, `repetition`, `input_sha256` of the exact case `inputs`, frozen `source_tree`, `configuration_sha256`, `origin` matching the case, `started_at`, `finished_at`, `provider_evidence_reference`, full `output`, and `requests`; injected case also needs `fault_injection_reference` |
| Output | `status` (`complete`, `blocked`, `partial`, `error`), nonempty `result` object containing the complete app result/partial failure evidence |
| Each request | `profile_id`, matching `phase`, unique `request_id`, `returned_model`, `version_reference`, effective `parameters`, full nonempty `system`, `user`, `response` text |
| Review | `agreement_sha256`, `results_sha256`, `completed_at`; `signatures` for all three reviewer IDs, each with `signature_reference`; independent `ratings` and `adjudications` |
| Each rating | `case_id`, `pathway`, integer `repetition`, `checkpoint_id`, `reviewer_id`; boolean `passed`; integer `unflagged_critical_changes`, `false_contradictions`, `missed_critical_facts` (all ≥0); `rationale`; `output_pointer` resolving into that run's output |
| Adjudication | Same rating fields, QA `reviewer_id` and `signature_reference`; exactly one for each disputed judgment/count, retaining both original ratings |

The agreement's signed time must precede every collection; final review must
follow all actions and cannot be future-dated. Both evidence and medical reviewers
judge every checkpoint independently, including negative findings and coverage
gaps. An output pointer is a JSON pointer such as `/result/revised_statement` or
`/result/digest/facts/0`; it locates evidence, not semantic proof. Record multiple
locations and detailed error occurrences in the rationale/external review record.
Each model/phase profile must be exercised in every pathway that its pre-agreed
`required_pathways` lists. Conditional phases may need extra cases to exercise
them; do not remove an intended phase's coverage requirement after collection.

```sh
python scripts/accuracy_benchmark.py assess \
  --plan accuracy-evidence/plan.json \
  --agreement accuracy-evidence/agreement.json \
  --results accuracy-evidence/results.json \
  --review accuracy-evidence/review.json \
  --out accuracy-evidence/assessment.json
```

Malformed, duplicate, incomplete, stale or mismatched evidence produces no
assessment artifact and exit 2. A complete failing package produces `NO_GO` and
exit 2. Submitted attestations within the proposed thresholds produce
`REVIEW_READY` and exit 0. An incomplete ordinary actual-provider action always
produces `NO_GO`, even if submitted judgments claim success. The assessor also
checks the serialized application's rubric/topic policies, completion statuses,
score fields, topic rows and claim/verdict coverage for evaluation, and current
grounding policy, topic completeness and generated text for drafting. An outer
`complete` label cannot override those missing/incomplete fields. These are
structural checks; semantic support still requires reviewers. Every assessment says
`pilot_admission: not_authorized_by_this_tool`. Hash binding cannot authenticate
signatures, prove an actual provider call, prove corpus-only inputs or establish
semantic correctness. A fabricated package can satisfy a local schema; the
independent operator must inspect provenance and signed judgments.

## Close R10 only on actual evidence

Preserve and verify the immutable plan/agreement, all actual runs/attempts,
independent ratings, original disagreements, signed adjudications and observed
failures. Fix failures, freeze the revised source and rerun the complete agreed
matrix plus regression cases; never relabel a failed older run as passing.
The operator then signs a revision/model/configuration-specific acceptance
decision and references that externally verified package in `accuracy_validation`.
The existing admission manifest stores an attestation reference; it does not
authenticate this benchmark or those signatures automatically.

Real pilot admission also requires R07 actual deployment, R08 provider/privacy,
R09 actual price/cutoff and the other applicable gates. CI and unit tests exercise
only evidence validation, with explicitly invented provider/reviewer fixtures.
They establish no actual-model accuracy. The synthetic corpus, reviewer review,
paid collection and final R10 acceptance are still **NOT RUN / NOT APPROVED**.
