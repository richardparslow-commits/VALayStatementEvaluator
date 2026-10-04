# R12 — actual-host operations and release acceptance

**Status: NOT RUN / NO-GO.** This procedure and the offline worksheet are
preparation. They do not establish installed monitoring, completed incident
drills, authenticated signatures or approval to process real veteran information.
General authorization to implement fixes is not operational acceptance evidence.

## Prerequisites and responsible owners

The operator records the private staging origin/access method, platform, named
operator, incident-response owner and backup contact, security/privacy reviewers,
private alert destination and approved incident procedure in protected storage.
Do not place credentials, invitation subjects, real records, raw environment
values or private incident evidence in GitHub. Use invented participants and
records; prohibit real case material during these exercises.

R07 host/isolation, R08 account/privacy, R09 spending, R10 factual quality and R11
legal acceptance remain separate prerequisites for real-information admission.
Any exercise involving the actual provider, account changes, credential rotation
or notification delivery requires explicit operator approval for those actions
and their recipients/costs. An offline worksheet never authorizes them. If the
necessary approval is missing, record **BLOCKED**, not PASS. A local fixture can
test the code path but cannot demonstrate the actual provider or private host.

The current controlled pilot excludes queues, durable case blobs, downloads,
remote OCR/research/fetch, external telemetry, case backups and operational-log
backups/restores. Do not enable the general worker, backup or monitoring examples
to satisfy this checklist. New storage/destinations require separate review.

## Freeze the exact release before exercising the host

1. Start from a clean, committed checkout. Record the full 40-character lowercase
   Git commit, tree, lockfile hash and build provenance. Build the web, parser and
   trusted launcher from that checkout with the exact reviewed revision; do not
   infer provenance from a manually supplied build label.
2. Scan every deployment image, including the proxy, with the operator-approved
   scanner. Record scanner/database versions, timestamps, image IDs/digests,
   findings, fixes and authenticated disposition of unresolved findings. CI
   builds and a tag alone do not establish actual-host vulnerability acceptance.
3. Prepare a protected release override/configuration using the reviewed immutable
   images. `docker-compose.pilot.yml` builds web/launcher and supplies a mutable
   proxy example tag; it is not itself an immutable image release. Record each
   effective web/parser/launcher/proxy image ID and registry digest, where used.
   Prevent rebuilding or pulling changed tags during launch/recovery. Preserve
   effective configuration hashes privately; redacted references go in evidence.
4. Bind the actual origin, issuer, invited roles, exact approved provider model
   versions, account/region, notice, retention and quota-policy hashes. The
   runtime manifest has **nine** required references: `provider_terms`,
   `retention_policy`, `privacy_review`, `legal_review`, `accuracy_validation`,
   `deployment_validation`, `incident_response`, `spending_controls`, `ingestion_security`.
5. Verify the approval is an absolute-path regular file owned by root or the
   runtime user, with no group/world write permission. Use an operator-controlled
   read-only mount and protect host parent directories and replacement authority
   from participants and the application. Runtime checks reject leaf symlinks,
   hard links, nonregular files, oversized/changed reads, duplicate JSON keys, nonfinite
   numbers, ambiguous identity/model lists, invalid timestamps and short release
   labels. These checks do not authenticate an attestation or secure the host's
   parent directories by themselves.
6. Approval must reference the exact release commit and valid timezone-aware
   timestamps with a maximum 30-day lifetime. Verify every referenced acceptance
   is genuine, current, complete and authorized. Preserve the intentionally
   expired/incomplete example; never populate it with placeholder acceptance.
   Establish operator-controlled renewal and change/revocation procedures.

## Prepare the offline worksheet

After committing the proposed release, from that clean checkout:

```sh
mkdir -p operations-evidence
python scripts/operations_review.py --out operations-evidence/draft.json
```

The tool compares and hashes every tracked release file, including binary assets,
entrypoints, parser lockfiles and scripts copied by Docker, and requires the fixed
deployment/monitoring/acceptance files. It refuses dirty, missing, linked or changed
source, including hidden working changes. It reads no operator environment,
approval file, credentials or records, calls no provider and performs no host
probe. Existing outputs are preserved rather than overwritten.

All 22 checks start **NOT RUN**, all owners/evidence/signatures are blank, and
the decision is **NO-GO**. Its hash binds the draft only. It does not validate
completed worksheets, authenticate signatures, produce an admission manifest
or certify a deployment. Keep the original draft/hash; record completed
observations, failures and later retests as separate protected evidence, hash
the completed packet and authenticate the reviewers through the approved channel.
The operator must manually verify completeness and authenticity before release.

## Required actual-host observations

| ID | Exercise and acceptance evidence |
|---|---|
| O01 | Verify clean source/provenance, locked dependencies, all image scans and actual immutable image identities match the tested release. |
| O02 | Reconcile the actual origin/access, issuer/roles, account/model versions, effective configuration and all nine accepted evidence references. |
| O03 | Inspect file ownership/mode, read-only mount, protected host parents and update authority; verify exact SHA and current approval of at most 30 days. |
| O04 | With synthetic approval copies, exercise missing/expired/mismatched revision, duplicate fields, short SHA, invalid timestamp, symlink and writable file. Admission closes before controls/provider work; restoring a valid file alone does not constitute release approval. |
| O05 | Cold start with no browser connected. Demonstrate log writers/idle cleanup, exclusive ledger lease and private health initialization. Missing approval/retention/parser dependencies prevent admission. |
| O06 | Inspect private app/parser channels and public route denial. Liveness, dependency readiness and admission authorization are separate observations; record their limitations below. |
| O07 | Stop/crash a synthetic service, induce an approved bounded resource/disk test and stale/failed cleanup condition. An independent private monitor detects process/host death, restart loops, pressure and retention failure within operator-approved limits. |
| O08 | Send an explicitly approved synthetic test alert containing only fixed service labels/counts/times to the approved private route. Record delivery and named acknowledgment timestamps, escalation, recovery notification and monitor-heartbeat failure detection. Empty recipients or an unobserved route is FAIL. |
| O09 | Exercise synthetic errors and inspect every actual diagnostic sink for canaries. Preserve fixed labels/counts without records, filenames, tokens, raw health bodies or arbitrary exception messages. Apply the R08 procedure. |
| O10 | Observe the running process without new UI traffic: log cleanup at startup and on the 60-second idle sweep; stale retention is detected after 120 seconds. Verify actual disconnected/upload lifecycle bounds through R07/R08 observations rather than asserting deletion from a UI reset. |
| O11 | Install and observe the separately managed stopped-volume/decommission schedule. Record deletion deadlines, access control, proof of execution without a running app and treatment of pre-existing snapshots/backups/provider copies. The app's idle sweep cannot clean a stopped host. |
| O12 | Execute all actual Linux parser boundary methods on the exact image/host: limits, deadline, mount/network refusal and container cleanup. Skips are incomplete, not PASS. |
| O13 | Use approved synthetic fixtures for provider timeout/error/redirect cases. No retries follow admission refusal, no optional destination activates and no delayed result bypasses ownership/approval. Actual-provider integration remains BLOCKED without approved account/model/cost evidence. |
| O14 | Exercise cancel, disconnect, resource pressure and bounded shutdown during a synthetic run. Measure response discard/cleanup and service survival; record the provider/browser copies already outside local control. |
| O15 | Stop before starting the replacement. Verify at most one web process, exclusive ledger lease, preserved quota reservations/totals and no overlapping admission. Do not reset the ledger/pilot ID to make a restart pass. |
| O16 | Remove a synthetic invitation during work, including other open tabs. At the next sensitive boundary subsequent work/results are refused; already displayed content cannot be recalled. |
| O17 | Drill the full incident stop below. Confirm all pilot services and any owned parser jobs are stopped, restart/redeploy paths held and new network/provider attempts absent after termination. |
| O18 | Only with separate action approval, rehearse revocation/rotation using synthetic test credentials for provider and OIDC client/cookie handling. Prove old credentials no longer work without logging them. Otherwise BLOCKED. |
| O19 | Named incident owners rehearse approved private notification/escalation, count-only evidence preservation, provider-copy handling and recipient acknowledgment. Record limitations and decision authority. |
| O20 | Verify case/log backups, snapshots, restore jobs and extra destinations are excluded from the effective pilot configuration. Do not restore case records or copy the quota database to a remote sink. Document historical-copy handling separately. |
| O21 | Recover the approved service without a case restore. Keep the durable control ledger, restore protected configuration through the approved operator method, check health/cleanup/alerts and require current exact-release evidence before renewed admission. |
| O22 | Authenticate operator, security and privacy acceptance and any signed narrowly justified inapplicability. All applicable checks have measured observations and accepted evidence; resolve every failure/blocked prerequisite before GO. |

## Monitoring boundaries and idle operation

Choose the private monitor/access method with the actual platform operator. The
app health interface binds to loopback inside the web container. The generic
Prometheus service-DNS scrape cannot reach that interface as currently configured,
and the generic Alertmanager receivers contain no notification integration.
Neither template proves delivered alerts. Do not publish diagnostic ports or
re-enable excluded services to make the examples work.

The existing `python scripts/pilot_status.py` reports fixed service state/counts
privately; it does not inspect credentials/log bodies, test TLS/provider access,
schedule monitoring or deliver alerts. Use an approved protected collector in
the necessary host/network boundary; restrict retained/exported observations to
an explicit count-only schema and test its failure behavior independently.

`/health` reports process liveness even when retention has failed: check the
structured retention active/failed/stale fields and actual cleanup timestamps.
HTTP 200 alone is insufficient. `/ready` can contact the configured provider's
model-list endpoint and is not proof of authenticated users, current manifest,
network policy, permitted costs or output quality. Do not make provider probes
without the separate approvals. Public `/health`, `/ready`, `/metrics` remain
blocked through the pilot proxy. A monitor in the same failed process cannot
detect total process/host death; verify an independent heartbeat/escalation path.

Record operator-approved detection/acknowledgment deadlines before O07/O08.
Missing values, a dashboard without delivery, stale observations or unavailable
monitoring keep the operational gate open. Do not collect raw diagnostic bodies
or private deployment metadata in a public service.

## Incident stop and recovery drill

The incident owner first holds any automatic redeploy/start path and closes
private ingress. Stop the full pilot using the reviewed effective configuration
and an operator-approved bounded stop timeout, with forced termination on expiry.
The application SIGTERM drain closes new pipeline admission but its daemon
waiter is not itself a process kill. Merely removing approval/invitations blocks
subsequent sensitive boundaries; it does not cancel HTTP already sent.

Confirm proxy, web and launcher are stopped, the private ingress is closed and
the daemon has no remaining parser jobs belonging to this pilot. Identify those
jobs through protected launcher/daemon evidence and terminate only the known
pilot jobs if required by the approved host procedure. Do not broadly prune
containers, remove volumes or reset the quota ledger as an incident shortcut.
Verify the operator can keep the deployment stopped across host restart, and
measure that no new provider work is initiated by this stopped release.

Preserve only approved count-only evidence under its retention limit. Follow
the named procedure for privately notifying approved recipients, revoking or
rotating credentials and contacting the provider about its copies; local
shutdown cannot recall provider requests, copied browser text or originals.
Such messages/credential changes are actions requiring their separate approval.

Recovery is stop-before-start using the same verified images, protected current
configuration and preserved quota ledger. Re-test health, retention and alert
delivery. A changed revision, image, origin, issuer, role, model, notice, policy
or destination requires the corresponding renewed review. Expired approval must
be renewed through actual reviewers; do not extend its dates to bypass the gate.

## Signed release decision

Store tester aliases, UTC windows, actual observed values, evidence hashes,
failed observations and retests for O01–O22. Record **PASS / FAIL / BLOCKED /
NOT RUN**. A reviewer may mark a narrowly inapplicable subcheck with a signed
rationale; required controls and prerequisites cannot disappear through a label.
Backup/restore exclusion is a required scope observation, not an invitation to
enable backup. No completed host observations or authentic signatures exist in
this repository's examples.

Only after accepted actual results and all other pilot gates are satisfied may
authorized reviewers sign the exact revision/configuration/evidence decision
and the operator reference it in `deployment_validation` and `incident_response`.
Verify both are current and genuine; nine nonblank strings alone are insufficient.
R12 preparation does not close R13 participant accessibility/comprehension or
authorize public expansion.
