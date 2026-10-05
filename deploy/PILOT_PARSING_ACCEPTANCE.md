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
| A14 meaning | Literal quote matches are explicitly separate from semantic acceptance. Every fact is unreviewed. A conservative comparison flags dates/numbers, negation, attribution, uncertainty, laterality and diagnosis/factual wording; any flag blocks pilot statement generation. Exact-output source comparison/review remains required. | Independent actual-model benchmark and source-bound human review. Lexical warnings can have false positives/negatives and cannot certify clinical entailment. |

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
