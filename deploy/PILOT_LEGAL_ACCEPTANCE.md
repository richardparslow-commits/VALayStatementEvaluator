# R11: independent legal and evidence acceptance

Status: **review preparation only; no qualified review has been completed**.
The candidate register, scenario questions and offline packet do not authorize
real-information admission. Keep the controlled pilot **NO-GO** until all
applicable release gates, including this review, have actual accepted evidence.

## Appoint the reviewer

The operator appoints an independent veterans-law reviewer or VA-accredited
representative with experience in the claim types actually offered. Record a
non-secret reviewer ID, qualification reference, verification date, independence
and conflicts statement, and agreed scope. Verify accreditation using the
[VA OGC search](https://www.va.gov/ogc/apps/accreditation/index.asp), or preserve
the qualified legal reviewer's relevant licensing and veterans-law experience.
No person has been appointed by the seed, and no message is sent by the tool.
Another model or an automated code-review service cannot provide this sign-off.

## Freeze the exact material

Start from a clean committed checkout of the proposed release. Store reviewer
work in protected storage outside the repository and build context. The command
uses only committed source and local files; it has no provider calls, research,
telemetry, application settings or client initialization:

```sh
python scripts/legal_review.py --out /protected/legal-evidence/packet.json
```

The parent directory must already exist. Exit 0 means an **unapproved packet was
prepared**, not accepted. Exit 2 means the checkout/seed is invalid or changed,
or the output already exists; no existing artifact is overwritten. This tool
has no assess, approve, sign, deploy or live-model command. It does not parse
completed reviewer findings or authenticate signatures.

The packet contains:

- Exact commit/tree and SHA-256 hashes for every tracked application file and
  `requirements.lock`, including condition-mapping JSON and all prompt helpers.
  Additional hashes bind the register, scenario seed, packet tool, batch draft
  script, this procedure, README and controlled-pilot guide.
- Every byte of the four knowledge files partitioned into numbered review
  units, including opening instructions, dimensions, style rules and examples.
- Complete consequential policy modules/screens named by the register, plus
  prompt-construction highlights found by static syntax inspection throughout
  the application. Highlights are reading aids; the whole mapped file must be
  reviewed. Dynamic construction, inline strings and helper effects are not
  proven complete by that inspection.
- A versioned candidate authority register and 14 entirely invented pairs of
  applicable/inapplicable scenarios. These are **unreviewed questions**, not
  approved expected legal outcomes or executed tests of model behavior.
- A blank reviewer ledger. Every unit is `unverified`, every scenario is
  `not_run`, and all reviewer/signature fields are empty.

The canonical packet hash covers all fields except `packet_sha256`, using the
same sorted, ASCII JSON SHA-256 convention as the accuracy preparation tool.
Preserve the original packet. Put findings in a separate signed evidence record
bound to its hash and the source tree. No preparation output fills `legal_review`
or changes the application's approval manifest. Nonempty manifest references
alone cannot establish that a real independent review occurred.

## Review every consequential rule

The [candidate register](../review/legal-authority-register-v1.json) supplies
43 source entries and 14 review controls. It is a starting map, **not a complete
or accepted citation table**. Court/manual lookup links explicitly remain
`lookup_only`; they do not stand in for exact primary opinions or pinpoints.
Retrieval observations are separate from interpretation and applicability;
all interpretation/applicability fields remain `unverified`.

The reviewer must inspect every packet unit and expand the map for omitted
rules, authorities, inline prompts or screens. For each consequential sentence,
example, scoring rule or condition mapping, record the unit/hash and exact span,
rule interpretation, claim posture and scope, applicable/inapplicable conditions,
primary source and pinpoint, effective date, protected source snapshot/hash,
later-treatment check, reviewer/date, disposition, rationale and signed reference.
Record editorial/scoring choices as such; do not invent legal authority for them.
Distinguish statutes/regulations and precedential holdings from internal manual
guidance, form instructions and the application's conservative drafting choices.

Use `accepted`, `changed`, `unclear`, `not_applicable` or `unverified` for each
finding. A justified exclusion needs a signed scope rationale. No unreviewed or
unresolved applicable rule may silently inherit another topic's approval. Review
the full original statement/evidence context when deciding applicability; an
uploaded-record subset is not the VA claims file.

Priority questions identified during engineering preparation:

| Control | Open question and required resolution |
|---|---|
| C01 | Reconcile the rubric's prohibition on trained-witness diagnoses with credential-sensitive drafting. Verify firsthand, reported-diagnosis and professional competence limits; avoid blanket exclusion of competent evidence. |
| C02 | Verify special chronic-continuity and combat routes, including conditions, incurrence and nexus limits; correct any conflation of 1154(a) and 1154(b). |
| C03 | Separate conservative record-gap labels from adjudicative weighing. Verify foundation/context for silence or normal findings and retain incomplete-source limitations. |
| C04 | Separate preservice aggravation, soundness and secondary aggravation. Resolve any suggestion that a lay baseline universally replaces required medical evidence. |
| C05 | Explain/justify the 1.5 accuracy weight and effectiveness components (40/30/20/10). Ensure writing scores and credibility examples do not predict an award or substitute for the benefit-of-doubt standard. |
| C06–C07 | Verify condition/rating/posture limits of functional-loss and examination-adequacy statements. Avoid categorical legal conclusions from incomplete records. |
| C08–C10 | Check aid-and-attendance/housebound/competency distinctions, PTSD/TDIU routes, forced topics and medication attribution. Do not force an irrelevant hazard or causation narrative. |
| C11 | Resolve official form metadata, certification/declaration wording and submission instructions. Verify claims about preferred forms, per-condition forms and addenda. |
| C12–C14 | Check factual preservation, witness approval, rhetoric, professional assistance and source freshness. Witness approval and a cached currency result do not establish legal sufficiency. |

For C04, the retrieved texts treat
[preservice aggravation](https://www.ecfr.gov/current/title-38/section-3.306)
and [secondary aggravation](https://www.ecfr.gov/current/title-38/chapter-I/part-3/subpart-A/subject-group-ECFR39056aee4e9ff13/section-3.310)
separately; the latter includes a medical-evidence baseline provision. The
rubric combines related routes in one paragraph. This is a **review question**,
not a completed determination of current case law, burdens or a veteran's claim.

For C01, [3.159(a)](https://www.ecfr.gov/current/title-38/section-3.159)
distinguishes competent medical from competent lay evidence. The exact scope
of the app's credential and prohibition rules still needs qualified review.

On 2026-10-03, the [VA form information page](https://www.va.gov/forms/21-10210/)
reported June 2021, while the [downloadable official PDF](https://www.vba.va.gov/pubs/forms/VBA-21-10210-ARE.pdf)
identified July 2024, superseding June 2021. The discrepancy is recorded for
resolution; it is not an approved form or submission workflow. The preliminary
1111 source page was under maintenance. Several opinions/manual provisions
remain lookup-only. Recheck all primary authorities at actual sign-off.

## Exercise and accept the scope

The [scenario pairs](../review/legal-scenarios-v1.json) contain invented examples
for every control. Before execution, the qualified reviewer approves or amends
their expected legal treatment, adds missing condition/posture branches and
specifies acceptable output language. Then exercise the exact proposed software
and approved model/configuration using synthetic inputs. Capture complete
outputs, observed applicability, source references, defects and signed findings
for both branches. No scenario has been run by this preparation tool. R10's
actual-provider benchmark and approvals remain separate requirements; reuse
accepted evidence only when source/configuration/scope are demonstrably equal.

Engineering regressions prove packet integrity and source coverage, not legal
correctness or model behavior. Record scenario outcomes as PASS, FAIL, BLOCKED
or NOT RUN. Keep critical legal misstatements, unsupported outcome promises,
lost competent evidence, forced inapplicable rules and incorrect submission
claims unresolved until corrected and re-tested. The reviewer specifies and
signs acceptance thresholds before testing; this procedure supplies no approval.

For acceptance, the operator independently verifies the reviewer's qualification,
independence and authentic signature; checks all units, added rules, primary
sources and both scenario branches are accounted for; confirms every required
finding is resolved; and preserves a complete signed record identifying exact
commit/tree, file hashes, packet hash, scope, review date and expiration/review
triggers. There is no automated ledger validator in this change. The operator
and reviewer must verify completeness; a signed summary alone is insufficient.

Keep affected guidance visibly unverified and pilot admission closed while any
applicable rule is unresolved. The existing controlled-pilot guide disclaims
legal sufficiency and witness truth certification. Change product guidance only
through a reviewed patch with regression evidence, then freeze a new packet and
obtain revision-specific acceptance. Knowledge, prompts/helpers, scoring,
condition mappings, consequential screens, authority changes or scope changes
invalidate affected evidence. Merge commits with identical source trees still
need an explicit signed revision binding; do not silently transfer sign-off.

Only after genuine acceptance may the operator reference the protected signed
packet in the manifest's `legal_review` field for the approved release. Verify
R07–R10/R12 and all other applicable gates separately. Generated formatted PDFs
remain supporting drafts unless the official form, completion, signature and
submission path have separately been verified. PDF downloads remain disabled
in the controlled real-information pilot.
