# Original-account comparison and human review

`original_account_source_comparison_v1` applies to generated initial drafts,
evaluation rewrites and proposed edits. It does not treat a generated first
draft or a second model's opinion as a factual baseline. The original witness
account and supplied witness/claim details are retained in `factual_inputs`.
Complete uploaded source evidence remains separate from prompt budgets.

The comparison builds immutable source IDs over original text spans and typed
metadata. Witness spans retain exact character offsets. Record passages enter
the ledger only through the existing retained-fact/full-quote/source-unit
validator. The source quotation, not a model's description or inferred date,
is evidence. A quotation can be split into comparison passages while retaining
its complete quote fingerprint, fact ID and typed source address; the complete
quote remains in the retained digest. It is not duplicated per sentence or
proposed edit. Ambiguous, missing or
fabricated quotes cannot become supporting passages. Original unreadable source
addresses are retained and reconstructed too, so another upload with the same
filename/page cannot make an ambiguous quotation selectable. Legacy inputs
missing these reservations require a re-run.

Every non-structural output sentence has offsets, a text fingerprint, selected
source IDs, candidate passages and unresolved comparison items. Exact original
matches receive a source link; lexical candidates alone never receive one.
Other sentences need explicit human source selection. Prose-valued witness
fields are split into passages with original offsets and field identity.
Recognized headings/certification are excluded from factual coverage requirements.
Source selection does not waive changed numbers, calendar dates, negation, approximate dates,
uncertainty, laterality, frequency, chronology, speaker attribution, diagnosis/nexus terms or factual
wording. Numeric occurrences also retain nearby wording within clause boundaries and occurrence counts,
so swapping existing years/quantities between claims is flagged. These lexical
associations can flag legitimate rephrasing and are not semantic proof. Original
witness passages must be preserved or linked to supported sentences. Record statements retain attribution to the records rather than
becoming firsthand observations, including when combined with witness sources.
Unresolved placeholders block approval.
Structured name, relationship, known-since and observation-frequency fields permit
narrow identity/opportunity phrasing; a claimed condition does not supply a
firsthand account or diagnosis. Claim condition/type labels stay in the context
fingerprint but cannot be selected as supporting evidence. These wrappers do not
permit new factual details.

The lexical screen is deliberately conservative. It can flag legitimate
paraphrases, abbreviations or template wording. Correct the output against the
original account, or update the original inputs with the witness and re-run.
Do not use an approval checkbox to waive new facts. No parser or lexical screen
establishes semantic entailment or truth; the actual-model factual benchmark,
qualified evidence review and legal acceptance gates remain separate.

The editable draft and rewrite forms show original passages and a source picker
for each sentence. All unresolved comparison items remain visible. A review
checkbox appears only after the comparison items are resolved. Its session-only
receipt binds the exact edited text, source context, source selections, workflow
and current pilot owner. Any change, including editing and reverting the text,
invalidates the prior receipt. Source and witness review must be performed again.
Receipts contain opaque fingerprints, not private passages, and are not durable
case approval or signatures. Pilot download exclusion still applies.

Saved payloads retain the original inputs but cannot grant approval by storing a
`reviewed` flag. Comparisons are rebuilt from original account/metadata, retained
facts and complete source pages. Durable app result payloads omit recomputable
comparison ledgers; evaluation payloads also omit rewrites from cached report
prose while retaining the structured rewrite for fresh review. Report rebuilds
do not emit duplicate source-appendix goal events. Missing or malformed legacy provenance stays
unreviewed and requires a re-run with original inputs. Cached rewrite reports are
rebuilt so historical final/approved claims cannot bypass current annotations.
Evaluation report downloads omit generated rewrites and proposed edits; the
dedicated statement exports require review of the exact edited text.

The batch path uses the same original-input comparison. Current and resumed
outputs are explicitly unreviewed and produce `factual_review.md`. Legacy finals
without original context remain unreviewed. Queues and durable batch storage
remain outside the controlled real-information pilot.

Comparison is bounded to 120,000 output characters, 4,000 source spans and 4,000
output spans. Excess spans explicitly block approval instead of silently
dropping evidence. The record catalogue uses a 4,000,000-character presentation
budget and omits unusable or excess quotation entries; omitted record quotations
cannot be selected as support. Full uploaded sources remain available separately.
Exact passages use an indexed lookup; substring checks use token intersections,
and an unchanged source selection reuses the first comparison. Displayed report
excerpts are shortened explicitly; comparison uses the complete
retained passage. No new model calls or external services are introduced.

`tests/test_factual_integrity.py` covers the original 2020→1995/invented-frequency
case, uncertainty and date precision, negation, calendar dates, laterality,
attribution, novel diagnoses/nexus wording, record quotation admission, initial
generation, revisions/proposed edits, saved/legacy output, batch output, exact
human approval and real Streamlit widget resets. All fixtures are synthetic.
