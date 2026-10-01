# Complete topic coverage and partial evaluations

Evaluation topic analysis uses `complete_evaluation_topics_v1`. A recognized policy
and `complete` status are required, together with a valid complete response. The
check validates structure and internal consistency; it does not prove that the
model chose the correct applicability, found all meaningful gaps, or accurately
interpreted the statement or legal framework.

The response contains exactly `claim_focus` (nonempty string), `topics`,
`critical_gaps`, and `notes` (string, which may be empty). `topics` is a JSON list
with exactly one row for each checklist topic A–O. Rows contain exactly:

- `topic`: the uppercase letter, or that letter and its documented checklist
  heading. Existing explanatory parenthetical heading suffixes are accepted.
  Unknown and duplicate topics are rejected. Valid rows are ordered A–O and use
  the shared checklist label catalog for display.
- `applicable`: a JSON boolean. Strings such as `"false"`, numbers, and null are
  rejected before rendering, filtering, or saved-result coercion.
- `coverage`: `covered`, `partial`, `absent`, or `not applicable`. Inapplicable
  topics must use `not applicable`; applicable topics cannot use it.
- `evidence`: a string, nonempty for covered/partial topics and empty for
  absent/inapplicable topics. This is a structural check; it does not verify the
  evidence against the statement.
- `gap_note`: a nonempty, non-placeholder string for partial/absent applicable
  topics, empty for covered/inapplicable topics. Structural length and placeholder
  checks cannot establish the suggestion's clinical or legal quality.

`critical_gaps` is a list of up to five nonempty strings. Each starts with a
distinct A–O topic letter referring to a partial/absent applicable row. An empty
list is allowed; it does not override the gap notes in those rows.

The topic phase makes at most three response attempts within existing provider
retry, timeout, cancellation, and pilot budget controls. Rejected output and
exception text are not echoed into correction prompts or topic-boundary logs.
Permanent provider authentication/configuration errors stop immediately through
the existing error path. Cancellation and pipeline timeout propagate.

Only a fully validated response is committed. Exhaustion leaves `topic_status`
`incomplete`, with empty topic rows and a static explanation. Independent record
review, claims, verification, evidence gaps, source appendix, and valid rubric
scores remain available. Rubric-based effectiveness is retained independently;
it is not a topic coverage score. Topic coverage counts/table, derived framework
flags, follow-up conclusions, topic-dependent recommendations, and the proposed
rewrite are withheld. The final progress, direct/queued completion message, run
log and worker classification report a partial evaluation. Pilot diagnostic
metadata admits only the closed set of topic status labels, excluding arbitrary
caller strings. A finished queue job may contain a partial evaluation.

Saved raw topic fields are validated before coercion/filtering can hide malformed
rows. Missing/unknown policies, incomplete statuses, and malformed data cannot
become complete. Unvalidated topic data and dependent outputs are withheld on
serialization and restoration. Historical cached reports are rebuilt from
retained structured findings and citations whenever rubric or topic validation
is incomplete. A re-run is needed to restore trusted topic coverage. Accepted
follow-up answers stay pending when a direct run remains partial, allowing a
subsequent complete evaluation to consume them. Saved answers and their clear
control remain accessible when coverage fails or all generated questions have
already been handled. Changing the run reference does not discard pending
answers for the same inputs; explicit consumption or clearing removes them.
Direct reuse requires a session-only SHA-256 binding over the exact statement,
record content and metadata, and witness inputs. Different or unrecognized inputs
do not receive pending answers. A returned result for different inputs clears
prior pending/applied answers and skipped questions before binding its new
questions. Legacy/unbound state is not automatically reused. Recovered queued
results invalidate direct answer bindings because they can represent earlier
submission inputs. The binding is not logged or persisted to a provider.

The follow-up panel displays validated gaps. An empty filtered question list
never claims every applicable topic is covered: questions may have already been
saved or skipped, while the underlying analysis still contains gaps. Saved or
in-memory invalid results cannot bypass the table/count/follow-up/rewrite gates.

Regression checks use synthetic records and fake model responses. They cover
boolean strings/numbers, duplicate/missing/unknown labels, inconsistent fields,
gap references, current checklist headings, bounded retry recovery/exhaustion,
privacy of rejected output, permanent errors, cancellation/timeouts, partial
review retention, cached exports, queue round trips, UI suppression, and worker
classification. No real-information admission or live deployment readiness is
established by these checks.
