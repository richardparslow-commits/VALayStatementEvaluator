# Complete rubric scoring and partial evaluations

The scoring boundary uses `complete_rubric_v1`. A result is scored only when its
policy and completion status are recognized and its complete rubric response
passes validation. This verifies response structure and numeric bounds; it does
not prove the assessment, cited interpretation, or legal guidance is correct.

The model response must contain exactly these five fields:

- `scores`: exactly the eight documented rubric dimensions, each a finite JSON
  number in the inclusive range 0–10. Booleans, numeric strings, missing or extra
  dimensions, NaN, infinity, and values outside the range are rejected.
- `rationales`: exactly the same eight dimensions, each with a nonempty string.
- `improvements`: a list of objects with `priority` (positive JSON integer),
  `problem` and `suggestion` (nonempty strings), and `example_rewrite` (string;
  may be empty). Extra or missing row fields are rejected.
- `omitted_record_facts`: a list of objects containing exactly nonempty string
  `fact` and `source` fields. This checks shape, not source validity or witness
  knowledge; those still require review.
- `executive_summary`: a nonempty string.

Empty improvement and omitted-fact lists are valid. No score is clamped or
substituted to make a malformed response appear complete.

The pipeline makes at most three rubric-phase attempts, within the existing
provider retry, cancellation, timeout, and pilot budget controls. Correction
instructions and scoring-boundary logs never quote the rejected output or its
exception text. Cancellation and pipeline timeout continue to stop the run.

Only a fully valid response is committed to the result. If all attempts fail,
`scoring_status` is `incomplete`: completed record review, extracted claims,
verified claims, evidence gaps, source evidence, and any available topic review
are retained. The rating is `Not scored`; effectiveness is `None` with an
`unavailable` band. Scoring recommendations and the proposed rewrite are skipped.
The progress message, screen, report, saved payload, and run-log classification
identify a partial evaluation instead of showing default zero scores or a
colored effectiveness badge. A finished queue job can contain this explicitly
partial result; queue completion means the job returned its retained result.

Saved scoring data is validated before coercion or filtering. The policy and
status survive the queue round trip, and effectiveness is recomputed from valid
restored inputs. Missing, unknown, incomplete, or malformed saved scoring data
cannot acquire a rating. Scoring output and dependent rewrites are withheld.
Historical cached markdown is rebuilt from the retained structured review data
for these results, so old grades and proposed rewrites cannot bypass the gate.
Historical narrative or report-only additions are not carried into the rebuilt
export; retain the original separately if needed, and re-run before using scoring.

Synthetic regressions cover numeric and schema rejection, boundary acceptance,
retry recovery and exhaustion, retained partial review, cancellation/timeouts,
static errors, saved-result and queue round trips, cached-report rejection, and
UI suppression. They use fake model responses and synthetic records. No real
model behavior, legal accuracy, or real-information deployment readiness is
established by these checks.
