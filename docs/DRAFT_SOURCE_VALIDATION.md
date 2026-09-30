# Drafting record provenance

Drafting validates record-based grounding before generating a statement. This applies to `run_draft` (including worker jobs) and the offline batch final phase. Evaluation uses the same uploaded-source address index.

## What passes

Each supported observation, record conflict and suggested inclusion must copy a retained fact's immutable `fact_id`, description, complete quote, source label and typed `source_unit` (`filename`, `kind`, integer `number`) from the supplied catalog. The address must identify one readable, unambiguous uploaded page or text block. The entire quote must match that unit under the existing `full_quote_page_v1` whitespace, Unicode case and word/numeric boundary rules. The fact's resolved document/page metadata, when present, must agree with the address. Accepted fields are replaced with the catalog's original values.

A fact ID hashes the canonical typed address and complete retained fact snapshot. A changed snapshot gets a different ID. The catalog selects up to 150 relevant resolvable facts within 90,000 characters and presents them chronologically, dropping whole entries. It never truncates retained digest facts or quotes. Facts whose prompt escaping would alter their text are excluded from the copyable catalog. An omitted, unresolvable or short-quoted fact cannot be used as validated record support. A range citation must be resolved to an actual single source unit upstream before it can enter this catalog.

Witness-only observations need no record citation. The prompt preserves uncertain firsthand accounts as unverified lay evidence; absence from the bounded catalog is not proof that an observation is false.

## Failure and saved results

Invalid responses receive at most three whole grounding attempts in the app. The batch final phase validates inside its existing two-attempt wrapper. Exhaustion stops before statement generation or self-review. Validation errors and retry logs contain no model-supplied private fields. Cancellation is checked before each app attempt.

The batch path re-extracts source files matching its configured glob, excludes quarantined files, and checks current source units rather than trusting digest checkpoints as source text. Changed or unavailable source material cannot establish record support just because a saved digest carries its old label or quote. Re-extraction failures propagate; an empty catalog permits witness-only rows but no record-based rows.

Successful results persist `grounding_policy=retained_fact_full_quote_source_unit_v1`; source labels and complete quotes are rendered for human review. Older or unknown policies receive an explicit re-run warning rather than being silently relabelled.

## Limits

Matching a quote does not prove that the digest description, date or interpretation is accurate, that the witness observation is corroborated, or that a proposed conflict is clinically meaningful. This gate does not validate every factual assertion in the final generated statement or certify a pilot deployment. Human review and the remaining factual-preservation work are still required.

All regression fixtures for this change are synthetic. No live provider calls or real veteran records are required.
