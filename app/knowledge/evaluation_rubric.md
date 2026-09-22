# Evaluation Rubric for VA Lay / Witness Statements

Score the statement on the 8 dimensions below, each 0–10. For every dimension, give:
(a) the numeric score, (b) 1–3 sentence rationale tied to the actual text, and (c) specific,
actionable improvements (quote the weak passage and propose a rewrite where useful).

## Dimension definitions

1. **Factual Accuracy vs. Medical Records** (weight: highest)
   Compare every factual assertion (dates, diagnoses, treatments, providers, facilities,
   injuries, test references) against the provided medical records. Mark each claim as
   SUPPORTED, CONTRADICTED, PARTIALLY SUPPORTED, or NOT FOUND IN RECORDS. Contradictions are
   critical findings — they can destroy credibility and must be highlighted prominently — but
   a contradiction must be **evidenced by record text you can cite**. Never manufacture one
   out of silence.

   **Verdict discipline — apply these exactly.** A mislabelled claim is worse than an
   unlabelled one, because the label is what the witness signs off on and what the claim is
   prepared around.

   - **CONTRADICTED requires affirmative contrary evidence**: a record entry that positively
     states the opposite — a dated finding, a normal test result, "denies knee pain", "no
     tenderness", "gait normal", the other side, an incompatible date or facility. Name that
     entry, with its source label and date, in `record_reference`. **If you cannot point to
     the specific record text that conflicts, the verdict is NOT FOUND.** The tool enforces
     this: a CONTRADICTED verdict with an empty reference is downgraded to a record gap.
   - **NOT FOUND IN RECORDS is the correct verdict for silence** — the provided records do
     not address the claim at all — and it is **not** an accuracy failure. It is the normal
     verdict for home symptoms, unrecorded symptoms, and events nobody wrote down. Absence
     of evidence on a question cannot be treated as substantive negative evidence against a
     claimant (Horn v. Shinseki, 25 Vet. App. 231, 239 n.7 (2012); M21-1, Part V,
     Subpart ii, Ch. 1, § A). The absence of contemporaneous records is not, standing alone,
     a reason to reject lay evidence (Buchanan v. Nicholson, 451 F.3d 1331 (Fed. Cir. 2006)).
   - **When is a gap itself meaningful?** Only when the fact is one that would ordinarily be
     documented — surgery, hospitalization, a fracture, an imaging study, a prescribed
     medication. Silence in the record cannot be treated as negative evidence unless it is
     the sort of condition or symptom that would normally be noted or reported
     (Buczynski v. Shinseki, 24 Vet. App. 221, 224 (2011)). Generalizable symptoms (pain,
     sleep, mood, home function) are routinely unrecorded, so their absence proves nothing.
     Even where documentation would be expected, the verdict stays NOT FOUND and the gap
     becomes a **development item** ("obtain the surgical report / service treatment
     records") — never a contradiction.
   - **A normal finding does not contradict a symptom.** A static examination recording
     normal range of motion, a normal gait, or "no acute distress" does not conflict with a
     claim of pain-limited use, functional loss, or intermittent flare-ups: it simply never
     measured them. Mark such a claim NOT FOUND ("record does not address functional loss"),
     never CONTRADICTED — see 38 C.F.R. §§ 4.40, 4.45, 4.59 and DeLuca v. Brown.
   - **Never describe a NOT FOUND claim as unverified, inaccurate, unsupported, or
     inconsistent** in a rationale, a `note`, or the verification table. Write "the provided
     records do not address this" — never "the records show no …", "there is no record of
     …", "the file is silent, therefore …", or "fails to appear in the records". That
     phrasing is carried into the revision and the report, where it reads as an affirmative
     finding the records never made.
   - **Do not lower this dimension for unverifiable claims.** Score accuracy on SUPPORTED
     versus CONTRADICTED only. How much of the statement is *checkable* from this record set
     is reported separately, as coverage gaps and evidence density; a statement whose claims
     are mostly NOT FOUND can still be entirely accurate.

   Score: 9–10 = every checkable fact supported, none contradicted; 7–8 = all facts either
   supported or not addressed by the records, with at most a trivial discrepancy (an
   unverifiable claim is not a discrepancy); 4–6 = one material contradiction; 0–3 = multiple
   contradictions, including any assertion the records affirmatively refute. A contradiction
   is the only thing that lowers this dimension.

2. **Specificity & Detail**
   Dates or time frames, places, names/units, frequency and duration of symptoms, concrete
   examples. Penalize vague language ("he's always been different", "often", "a lot") unless
   accompanied by concrete instances. Reward the pattern: observation + when + where + effect.

3. **Lay Competence Boundaries**
   Apply Jandreau v. Nicholson, 492 F.3d 1372 (Fed. Cir. 2007): competency depends on the
   disability claimed, and a lay witness is competent to describe what they personally
   observed — including the onset and continuity of symptoms — and to identify conditions
   that are simple and readily observable by a layperson. **This cuts both ways:**
   - **Penalize** medical conclusions the writer is not competent to give: formal diagnoses
     of non-obvious conditions, medical etiology ("the arthritis was caused by his
     service"), interpretation of imaging, labs or test results, prognoses, and legal or
     rating conclusions ("he meets the criteria for 100%", "this should be rated 70%"). In
     every case supply the reword as observation or attributed statement ("I watched him
     stop after half a block"; "his doctor told him …").
   - **Penalize equally the opposite error**: discounting, hedging, or burying a competent
     lay observation because it lacks medical support. A symptom description ("his knee gave
     out and he caught the counter") is competent evidence — 38 U.S.C. § 1154(a);
     38 C.F.R. § 3.159(a)(2); Washington v. Nicholson — and must be credited as such.
   - **Sequence is not etiology.** "It began when he fell from the truck and it has hurt ever
     since" is competent lay evidence of incurrence and of continuity of symptoms
     (38 C.F.R. § 3.303(b); Davidson v. Shinseki); do not flag it as a medical opinion. What
     needs a medical opinion is the *cause* of a condition whose cause is not apparent to a
     layperson.
   - A witness with medical training (nurse, medic, corpsman) may describe observations with
     more precision — mechanism, observable signs, functional measurements in ordinary
     language — but must still not diagnose the veteran inside the statement.
   - Statements attributed to a provider ("his doctor said …") are competent evidence of
     what the provider said, not proof of the clinical fact; note them for the record and
     ask whether the underlying treatment record can be obtained.

4. **Connection to Claimed Condition**
   Is the statement explicitly tied to the condition(s) claimed? Does every paragraph advance
   that condition's claim (service connection / increase / TDIU / stressor)? Penalize generic
   narrative that could apply to any claim.

5. **Continuity & Timeline**
   Does it establish onset/in-service event → continuity of symptoms since service → present-day
   severity? Identify gaps in the timeline and, where records show treatment gaps, check whether
   the statement honestly explains them (gaps are not fatal if explained).
   **Aggravation claims are a second, equally valid timeline shape.** Not every condition begins
   in service: where the statement says a pre-existing condition was made worse — "it ached
   sometimes before; after the carrying and climbing it never stopped" — the required arc is
   baseline before service → the service that increased it → a permanent increase since
   (38 U.S.C. § 1111; 38 C.F.R. §§ 3.310, 3.303(d)). Lay evidence of the baseline, the
   aggravation, and the continuity is competent evidence under 38 U.S.C. § 1154(a) and
   38 C.F.R. § 3.159(a)(2) and is often decisive. Score such a
   statement on that arc, not against a direct-onset template; a missing pre-service medical
   record is NOT FOUND territory, and a competent baseline description ("he limped only after
   long hikes before; now he limps to the mailbox") substitutes for one. Never fault a statement
   for the "missing" onset event that aggravation claims do not have.

6. **Functional Impact**
   Concrete description of impact on: work/occupational function, activities of daily living,
   and social/family relationships. The more concrete and quantified (frequency, duration,
   specific incidents), the higher the score.
   For any physical or musculoskeletal condition the statement must describe **functional
   loss, not measurements**: pain on movement and what it stops him from doing; loss of
   motion that appears with repeated use ("after ten minutes of walking he has to sit");
   **flare-ups** — how often, how long, how severe, what triggers them (cold, damp, overuse,
   standing), and what he cannot do during one; weakness, giving way, numbness, disturbed
   sleep, and difficulty with weight-bearing or locomotion (38 C.F.R. §§ 4.40, 4.45, 4.59;
   DeLuca v. Brown, 8 Vet. App. 202 (1995); Sharp v. Shulkin, 29 Vet. App. 26 (2017)). Where
   pain causes functional loss it is rated as if that same loss had another cause (Mitchell
   v. Shinseki), so "it only hurts" is incomplete reporting, not evidence of minimal impact.
   A statement reporting only a static limitation ("I can't bend my knee past 90 degrees")
   with no pain, flare or function misses the criteria that decide these ratings. The witness
   describes the function; never assert a measured range-of-motion value or a percentage
   rating.
   Separately, flag as a **development gap — not a statement defect** — any record examination
   that measured the joint only at rest without addressing painful motion, functional loss,
   or flare-ups: that examination is inadequate for rating purposes, and the remedy is to
   request one that addresses those factors, not to score the witness down for the gap.

7. **Credibility & Consistency**
   Internal consistency (no contradictions within the statement), plausibility, tone (factual,
   not melodramatic or accusatory), absence of exaggeration, honest acknowledgment of limits of
   knowledge, and consistency with the medical record digest.
   **Basis of knowledge is a credibility factor.** What the witness personally experienced or
   observed carries more weight than what they learned secondhand (38 U.S.C. § 1154(a);
   38 C.F.R. § 3.159(a)(2)): a relayed account is credited as evidence of what was said, not
   as proof of the inner state, and a statement that attributes every claim to the veteran's
   own say-so leaves little the witness has actually seen.   Penalize hedging of the witness's
   own observation; do not penalize the absence of a medical opinion beside it.
   **Benefit of the doubt — immaterial discrepancies do not destroy credibility.** Under the
   benefit-of-the-doubt rule (38 U.S.C. § 1154(b); 38 C.F.R.
   § 3.102), the claimant prevails where the evidence is approximately balanced, and a reasonable
   doubt is resolved in the claimant's favor. A minor, immaterial timeline slip — a date wrong
   by a few weeks, an incident placed in "the spring of 2010" when it was 2011, a medication name
   remembered imperfectly a decade later — is expected of lay memory after years and must not
   lower this dimension when the substance is consistent: the event, its character, its effect.
   What still matters is material consistency — did it happen, was it service-related, has it
   continued, is one account contradicted by record evidence — not calendrical precision.

8. **Form & Completeness**
   Present: witness identity and relationship to veteran; how/when witness knows the veteran and
   opportunity to observe; the claimed issue addressed; certification of truthfulness;
   signature/date block. Note VA Form 21-10210 is the preferred form (one per witness).

## Output requirements

- Also produce an **overall rating**: Excellent (≥8.5 avg) / Strong (7–8.4) / Adequate (5–6.9)
  / Needs Substantial Work (<5), weighted toward Factual Accuracy.
- Produce a **claim-by-claim verification table**: claim text | claim type | verdict |
  supporting/conflicting record reference (with page or date if available) | note. Every
  CONTRADICTED row must name the conflicting record entry in its reference cell; a row with
  an empty reference is a coverage gap, whatever the verdict said. Never restate NOT FOUND as
  a finding that the records refute the claim, and never list NOT FOUND claims as critical
  findings or contradictions.
- List the **top 5 prioritized improvements**, each concrete and implementable.
- If records were provided, list **facts in the medical records the statement omits** that would
  strengthen it (the writer may confirm them from personal knowledge before adding).
- Include a **"not legal advice"** footer.
