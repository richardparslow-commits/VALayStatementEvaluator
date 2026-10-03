# R08 — Provider, participant notice and retention acceptance

**NOT RUN / NO-GO.** Repository tests establish software behavior only. A notice
string, evidence reference, or `store=false` request is not provider or privacy
approval. Complete this procedure for the actual account and host with synthetic
canaries before real-information admission. R07, R09–R13 remain separate gates.

## Review the actual service configuration

Record the named operator/controller and privacy/security reviewer; private host,
region and storage; identity provider; exact analysis service, account/product,
base URL, approved models and regions; and every destination receiving data.
Inventory the provider, identity logs, host/container logs, browser caches and
copies, support tools, monitoring, crash dumps, snapshots and backups. Include
subprocessors and the operator's support/incident contacts. Do not infer account
terms from a generic provider homepage or SDK setting.

Obtain dated evidence for that actual account: training/data use; request,
response, abuse-monitoring and support logging; configured storage; each
retention/deletion deadline and exception; geographic routing; subprocessors;
contractual permissions; incidents and deletion requests. Review the exact
participant notice against those findings. Never claim that clearing a case
removes a request already sent to the provider or a copy made outside the app.

## Software policy and its limits

The protected approval JSON must contain the exact reviewed plain-text
`participant_notice` (80–16,000 characters) and integer
`local_log_retention_days` (1–30). Set `VA_LSE_PILOT_LOG_RETENTION_DAYS` to the
same integer. There is no real-data default. Obtain independent acceptance for
this maximum; a stricter policy can require a smaller value or excluding logs.
The deliberately incomplete example is not a usable approval.

R09 adds a separate private control volume containing a random pilot ID,
HMAC-derived participant identifiers, run/attempt timestamps, counters and
reported token counts. These are pseudonymous metadata, not anonymous data.
Approve their use and deletion deadline in the participant notice and retention
review; see the [budget procedure](PILOT_BUDGET_ACCEPTANCE.md). Case clearing
preserves quota controls. Exclude the control volume from snapshots/backups;
never delete or restore it during an active pilot to reset limits.

The pilot Compose entrypoint (`python -m app.pilot_server`) initializes cleanup
and the private health server before Streamlit starts listening, without waiting
for a browser session. An unusable policy/sink stops process startup. Other pilot
hosts must use this entrypoint or prove equivalent server-start initialization.

All three local count-only streams (`app.log`, `audit.log`, `runs.jsonl`) use size
rotation plus age cleanup under each writer's lock. The first event's timestamp
sets the file's expiry; continued writes and changed filesystem modification
times do not renew it. Whole files can disappear early, including newer events
in that file. Numbered rotations outside the configured count are also swept.
Unknown, malformed or future-dated legacy files have no provable age and are
removed. Startup cleanup, cleanup before writes and a daemon pass every 60
seconds enforce this limit **while the process is running**. Scheduling/resource
pressure can delay a pass; health reports a stale sweep after 120 seconds and
subsequent admission closes. A failed cleanup or file write closes subsequent
admission; previously lost audit entries cannot be recovered by that refusal.
Do not operate an independent sweeper against a running writer.

**Stopped volumes need their own accepted deletion procedure/schedule.** An
in-process thread cannot remove a stopped container's volume, host snapshot or
backup. The host operator must document and test how these copies expire during
outage/decommissioning and how restart/restore avoids resurrecting them. The app
sweeps again on startup before admission, but this does not establish the offline
deletion deadline. Logical unlink is not certified secure erasure of underlying
blocks; disk encryption and host lifecycle are separate review evidence.

Retained rotations are also restricted to owner-only mode, with file-type/owner
validation; audit write failures preserve both the audit counter and retention
failure signal.

The pilot Compose profile disables container log persistence for web, launcher
and proxy. Local count logs remain on the private volume. Audit backup,
verification/download and restore entry points refuse pilot execution, including
explicit destination overrides. Do not enable general backup/restore examples,
log drains or snapshots for this pilot without separate destination, contents and
retention approval. Historical backups created before this change are unaffected. Operators can run
`python scripts/pilot_status.py` from the private host checkout to inspect only
fixed service states, exit/restart counts, OOM and health classifications. It reads
no raw logs, environment or health-check output, and never echoes CLI errors.
The status tool exits successfully only when web/parser health is `healthy` and
all three processes are running; nginx has no health check, so its process state
does not establish TLS/routing acceptance. Pending/unavailable health is refusal.
Use the private app health endpoint for count-only audit/retention status and the
private parser health protocol for readiness. These prepared diagnostics do not
replace R12 actual-host monitoring/incident acceptance; deeper investigation
requires a separately approved synthetic reproduction, not enabling record logs.
No durable case blob, queue or optional external service is admitted.

Before case controls appear, an invited participant reads the exact notice as
literal text and explicitly consents. Consent is kept only in that browser
session, bound to the verified owner, exact notice, provider/models, log policy
and privacy/provider/retention evidence references. Changes clear the working
case at the next screen admission and require consent again. Work start, each
provider request/response and cached/delayed result access check current consent.
Worker threads receive the accepted notice snapshot; changing the notice during
work refuses its delayed result. A transmitted request cannot be recalled.
Clearing/sign-out revokes the shared session consent grant before removing
uploads, so already-copied worker contexts refuse subsequent attempts and delayed
responses. A provider attempt already begun cannot be recalled. Clearing/sign-out
clears consent too. Other open sessions and provider copies
have the separately documented R07 limits. The 60-second disconnected-session
setting is checked; actual release on the actual host still needs observation.

## Participant notice worksheet

Write a complete, understandable plain-text notice, approved by the named privacy
owner, and place that exact text in the operator-owned approval. Resolve every
item below with actual values; do not paste this unfinished worksheet as consent.

- Who operates this pilot and how to contact the privacy/incident owner.
- Why the participant's statements, witness observations and extracted record
  text are sent, which service/account/models receive them, and where processing
  and provider retention occur.
- Actual provider training/data-use rules, retained copies, deadlines, deletion
  exceptions and how to request deletion. Explain any support/abuse-monitoring
  retention and onward subprocessors.
- What the app retains in session memory and parser temporary containers; the
  actual clear, sign-out, disconnect and process-restart behavior and limitations.
- The approved local count-log maximum, possible earlier rotation/whole-file
  deletion, 60-second sweep interval and accepted stopped-volume procedure.
  Explain historical/snapshot copies if present; current app backups are disabled.
- Participant-controlled originals, browser storage, copies to approved systems,
  who can access these, and their separate deletion duties.
- Limits of AI output, required source/witness review, voluntary participation,
  authority to submit third-party records, withdrawal and consequences of
  requests already sent. Legal/claim guidance is separately reviewed under R11.

## Actual-host acceptance record

Bind evidence to source revision, image IDs, account/product/region, exact notice
hash and policy. Retain only synthetic fixture identifiers/counts in the report.

| Check | Required observation/evidence |
|---|---|
| P01 | Named operator/reviewer and complete destination/copy inventory. Current actual-account terms and configuration screenshots/references. |
| P02 | Independent review establishes training use, retention/deletion exceptions, subprocessors, regions, support/incident handling; notice accurately reflects each. |
| P03 | Missing notice or age policy, policy mismatch, disabled/missing local sink, bad permissions, stale cleanup, unapproved backup and a disconnect setting above 60 seconds close admission. |
| P04 | Invited sign-in shows full literal notice before statement/upload/action controls. No acceptance prevents upload/provider work; consent permits the released controls. |
| P05 | Change notice, provider/model or privacy/retention reference; existing case is cleared, cached results/provider calls are refused and fresh consent is required. Test during a delayed run, during request preparation, and after clearing/sign-out in another execution context too. |
| P06 | Synthetic personal/token/filename canaries cover upload, parse success/refusal, provider success/error/timeout, cancellation and clearing. Inspect all three logs, STDERR/platform sinks and effective network destinations; no content canary or unapproved destination is present. |
| P07 | Clear/sign-out removes working session values and registered uploads; new identity cannot inherit case or consent. Document provider/original/other-tab limits truthfully. |
| P08 | Disconnect beyond 60 seconds, reconnect and restart; observe actual session and upload-manager release. Record transient in-flight request/worker limitations. |
| P09 | Start the server without opening a browser and verify cleanup starts. Exercise low-volume active files, continued writes, size rotations, idle expiry, restart, malformed legacy files and failed deletion. Health signals failure/staleness; no expired file is read/admitted at restart. Review clock synchronisation. |
| P10 | Verify effective container logging driver, absent backup destinations/drains and denied override/restore, private retained rotations, preserved audit loss counters and restricted service-state diagnostics. Test accepted stopped-volume/snapshot/decommission expiry and restore policy. |
| P11 | Privacy owner accepts the exact notice, actual provider terms and separate policies for every copy, deletion deadline/exception and incident process. |

Copy and complete this evidence record. All fields deliberately remain empty.

```json
{
  "status": "NOT RUN / NO-GO",
  "reviewed_revision": "",
  "host_and_images": "",
  "operator_and_privacy_owner": "",
  "provider_account_product_models_regions": "",
  "destination_and_copy_inventory": "",
  "actual_provider_terms_and_settings_evidence": "",
  "participant_notice_sha256": "",
  "local_log_retention_days": null,
  "stopped_volume_snapshot_deletion_evidence": "",
  "synthetic_lifecycle_and_canary_evidence": "",
  "checks_P01_through_P11": "",
  "deletion_exceptions_and_incident_contacts": "",
  "independent_privacy_acceptance": "",
  "provider_terms": "",
  "privacy_review": "",
  "retention_policy": ""
}
```
