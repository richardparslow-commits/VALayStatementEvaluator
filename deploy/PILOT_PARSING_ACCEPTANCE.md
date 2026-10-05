# Parsing, processing and understanding acceptance — OPEN

These code controls address audit A06–A14. They do not establish clinical
accuracy, OCR recognition accuracy, complete PDF visual extraction, or approval
to process real veteran information. R10 and actual-host acceptance remain open.

| Finding | Implemented control | Required operator evidence |
| --- | --- | --- |
| A06 encoding | Strict Unicode decoding; original byte SHA-256, decoding method and source metadata survive parser transport and saved results. Unicode retrieval retains short terms and negation. | Reviewed source-to-text comparisons, including non-English testimony and symbols. |
| A07 statement coverage | Uploaded statements retain their extracted-document contract through validation and evaluation. Unknown/missing coverage and source/text mismatches refuse before provider work. Source metadata changes invalidate session review bindings. | Verify statement upload, pasted text, follow-up answers and unavailable-page behavior on the approved host. |
| A08 Word stories | Main body, tables, headers, footers, footnotes, endnotes and comments become numbered blocks with package-part provenance. Tabs/breaks separate words; note IDs and comment attribution are retained. Visual content, tracked changes, field-generated text, equations and unsupported symbols are refused. | Compare supported stories against Word rendering. Convert unsupported documents to a complete reviewed readable source; do not waive extraction omissions. Blocks are not Word page numbers. |
| A09 OCR output | Fallback transcript writer refuses overflow, overwide words and unsupported glyphs before drawing. Generated text must round-trip through extraction. Both OCR backends stage privately, check source page count and publish atomically; failures preserve previous output and original input. | OCR engine accuracy and image/text alignment benchmark. The fallback remains a text transcript, not a source-image-preserving OCR layer; the original is required for review. External OCR remains excluded from controlled-pilot execution. |
| A10 retrieval | Nonempty short passages remain indexed. Unicode words are preserved. Excerpts select complete nearby sentences around a match; oversized sentences are disclosed and cannot yield strong evidence. | Condition/language-specific retrieval recall and attribution review; lexical matching does not establish absence of a condition. |
| A11 dates | Unambiguous MDY/DMY slash dates resolve; ambiguous order, two-digit years and invalid dates stay unresolved. Prompt anchors contain raw spelling and precision; approximate full dates keep their qualifier and cannot create exact gap durations. Pilot timelines never infer missing dates with an optional model. | Review date-order policy and source precision. Timeline sort anchors are not asserted event dates. |
| A12 model/resource bounds | Accepted account text is never prefix-truncated. Pilot requests include all final fields, roles, escaping, re-asks and output allocation in the reviewed bound. Returned model version must match. Response limits apply before SDK object parsing. | Exact model versions, context/output limits and tested upper-bound contract for every route, including a provider's hidden framing and multilingual/escape-heavy requests. |
| A13 summary | Whole authoritative facts are selected within a budget, prioritizing date endpoints. A saved coverage manifest identifies selected fact IDs and counts omissions. Model consolidation and the second prefix cut no longer feed summaries. | Review earliest/latest and contradictory evidence, incomplete summary disclosures, and independently judge summary meaning. |
| A14 meaning | Literal quote matches are explicitly separate from semantic acceptance. Every fact is unreviewed. A conservative comparison flags all critical feature categories, including number associations, chronology, speaker, frequency, date precision, negation, attribution, uncertainty, laterality and diagnosis/factual wording; any flag blocks pilot statement generation. Exact-output source comparison/review remains required. | Independent actual-model benchmark and source-bound human review. Lexical warnings can have false positives/negatives and cannot certify clinical entailment. |

## Model profiles — no defaults or approvals supplied

Each `models` route needs a matching entry in `model_profiles`:

```json
{
  "APPROVED_ROUTE": {
    "model_version": "EXACT_RETURNED_IMMUTABLE_MODEL_VERSION",
    "context_window_tokens": null,
    "max_output_tokens": null,
    "input_tokens_per_utf8_byte": null,
    "framing_token_reserve": null,
    "token_bound_evidence": ""
  }
}
```

The null/empty template deliberately closes admission. The request counter uses
the byte length of the complete ASCII-escaped JSON serialization (including
syntax), multiplied by the approved input-token upper bound per UTF-8 byte,
plus a framing reserve. It reserves the entire requested output allowance.
This is a conservative **operator-tested upper bound**, not an exact tokenizer
or a universal claim that one byte always bounds one token. The operator must
verify it against the actual provider/model tokenizer, hidden framing, reasoning
allocation and all used endpoint schemas; otherwise that model cannot be used.
The provider/model version returned in every response must match the profile.
Changing profiles changes consent and requires renewed revision-specific review.

Pilot responses request identity encoding and refuse compressed responses before
HTTP decompression. Actual bodies, including errors, are capped at 2 MiB;
JSON depth is capped at 64 and structural items at 20,000 before SDK parsing.
The per-attempt wall deadline supplements network timeouts and the existing
private-client watchdog. A wrapped SDK transport refusal propagates without a
retry. This requires the installed `openai`/`httpx2` stack; synthetic SDK tests
and the locked runtime image must pass after dependency changes.

## Acceptance record

Record the immutable source/tree and parser-image digest, host configuration,
model/profile references, synthetic fixtures and complete outputs, independent
reviewers, observed failures and signed decision. Preserve original inputs and
source hashes in the approved review workflow; no real inputs in repository,
logs, test fixtures or benchmark artifacts. Follow `PILOT_ACCURACY_ACCEPTANCE.md`
and the resource, ingestion, privacy and operations acceptance documents.

**Actual extraction/model benchmark: NOT RUN. Model profiles: NOT APPROVED.
Actual-host acceptance: OPEN. Real-information admission: NO-GO.**

## IA-06/IA-07/IA-08 follow-up safeguards

Code remediation covers ingestion audit IA-06, IA-07 and IA-08. **Independent
accuracy and actual-host acceptance remain OPEN.** Use synthetic records only.

### Enforced contracts

| Boundary | Behavior |
| --- | --- |
| Provider envelope and inner completion JSON | Shared decoder checks at most 2 MiB of actual UTF-8, depth 64 and 20,000 structural units before building the JSON graph. Structural units count container openings, commas and colons outside quoted strings; they are not an exact object count. Envelope roots must be objects; completion roots must be objects or arrays, with the existing caller-specific schema checks still required. |
| Ambiguous JSON values | Duplicate keys (including escaped equivalents), NaN/Infinity, floating overflow to infinity, integers above 4,300 digits or the interpreter’s stricter limit, malformed UTF-8 and decoded lone surrogates in keys/values are refused. Valid Unicode scalars and escaped surrogate pairs are retained. Errors contain no response text. |
| Completion recovery | A single layer of Markdown fences is tolerated. Pilot mode refuses prose/braces rescue. Non-pilot legacy prose rescue only applies to a prose-led document, with the same decoder on every candidate. A hard value/resource violation cannot trigger the JSON helper's re-ask or its larger truncation budget. Syntax/truncation may receive the existing single bounded re-ask; the replacement has identical limits and normal provider/budget checks. |
| Critical clinical drift | Compare every existing critical feature category, including number-to-claim associations, calendar dates, chronology, speaker, approximate-date qualifiers and frequency. Added factual wording and unsupported fact-date numbers remain flagged. Any flag blocks pilot record analysis before its narrative summary. Complete quotation matching does not certify clinical interpretation. |
| Pilot input coverage | Before quota/model work, records must declare known coverage and a positive explicit source-unit count, contain every sequential page/block exactly once, have matching document/kind addresses and readable text, and contain no unreadable units. Duplicate document identities are refused. Direct record-review calls apply the same input contract. |
| Saved/delayed output review | Retained source metadata includes versioned coverage, counts and unreadable addresses. Pilot review reconstructs the complete record set and rechecks actual citations/critical features rather than trusting saved all-clear flags. Legacy, partial or inconsistent manifests and formerly capped digests cannot receive source/witness review approval; re-run with complete readable inputs. Non-pilot partial-record analysis retains its warning policy. |

### Synthetic acceptance

Run `tests.test_parsing_understanding` alongside parsing-fidelity, citation,
request-validation, factual-integrity and JSON-retry regressions. Confirm:

- Deep/item/byte overflow is refused before `json.loads` graph allocation,
  including multibyte content and oversized wrappers. Valid boundary cases pass.
- The authentic quote “Pain was 2/10 at rest and 8/10 walking.” cannot accompany
  a description with the two scores swapped and reach pilot summary generation.
- Date, dose, speaker, chronology, negation, attribution, laterality, uncertainty
  and frequency changes are flagged. Unchanged descriptions remain unreviewed
  for clinical meaning, even when their quotes match.
- One readable page plus an unreadable second page, unknown/zero source counts,
  duplicate/missing/reordered/wrong-kind addresses and legacy result metadata
  cannot bypass fresh or recovered pilot admission. Refusal happens before
  quota reservation and provider work; the original inputs remain available.

### Remaining review and release gates

Complete [synthetic accuracy acceptance](PILOT_ACCURACY_ACCEPTANCE.md) with the
exact approved provider/model versions and independent qualified reviewers.
These code tests are not that benchmark, an OCR verification, or semantic
entailment proof. Lexical comparison is deliberately conservative and can flag
legitimate paraphrases; reconcile the output with the source instead of waiving
new facts. Identical lexical feature sets can still conceal other meaning errors.

Complete readable coverage means all declared source units yielded text; it
does not establish correct columns, tables, OCR, spelling or interpretation.
Image-only/unreadable material requires a reviewed readable copy and a fresh run.
Nonclinical/no-fact chunks and ranked retrieval subsets still need uncertainty
review; absence from a summary or retrieval subset does not prove absence from
the medical record. Complete [actual-host acceptance](PILOT_ACCEPTANCE.md) before
processing real veteran information. No provider/reviewer approval is created by
these changes.

Python decoder defaults and hooks were checked against the
[Python 3.13 JSON documentation](https://docs.python.org/3.13/library/json.html).
`parse_constant` covers named non-finite constants; `parse_float` is also required
to refuse overflowing exponent notation. Source extraction remains subject to
the [pypdf extraction limitations](https://pypdf.readthedocs.io/en/6.18.1/user/extract-text.html).
