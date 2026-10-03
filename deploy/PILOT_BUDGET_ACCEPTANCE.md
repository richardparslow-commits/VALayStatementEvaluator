# R09 — Durable quotas and actual spending acceptance

**NOT RUN / NO-GO.** Offline regressions prove the implemented reservation and
restart behavior. They do not prove actual model charges, the provider account's
cutoff, host storage durability or ingress/WebSocket protection. R07/R08 and
R10–R13 remain independent acceptance gates. Use synthetic canaries only.

## Reviewed policy and spending boundary

The operator approved a $250 total pilot envelope, two starts per participant per
rolling 24 hours, 50 attempts per run and a **provisional** $1 maximum charge per
attempt. The protected example records those numbers but is deliberately expired,
has an invalid pilot UUID and has no spending evidence. Do not make it admissible
by inventing review references. Document the named alert owner and stop operator.

`quota_policy` requires exactly these fields. All numeric fields are integers;
one micro-USD is $0.000001. The runtime requires a private existing ledger whose
policy matches these values exactly, including the pilot ID and expiration.

| Field | Proposed value | Meaning |
|---|---|---|
| `pilot_id` | New reviewed UUID | Stable for the whole budget period and approval renewals |
| `expires_at` | Reviewed deadline | At most 30 days; approval cannot outlive it |
| `participant_daily_starts` | 2 | Rolling 24-hour starts, in addition to two per rolling hour |
| `run_attempts` | 50 | Includes failures, network uncertainty and every application retry |
| `run_prompt_chars` | 2000000 | Across the entire run and every client/worker |
| `attempt_output_tokens` | 8192 | Maximum requested output for either API schema |
| `attempt_charge_microusd` | 1000000 | Conservative full charge per actual attempt; provisional |
| `pilot_total_microusd` | 250000000 | Global envelope, including all outstanding reservations |
| `pilot_total_attempts` | 250 | Independent global attempt envelope |

Before processing, a run reserves all 50 attempts at $1 each ($50). If less than
$50 or 50 attempt slots remain, admission refuses before paid work even if the
operator expects a cheap run. Every SDK attempt is durably marked before the
request. All clients and copied worker contexts use the same reservation.
No reserved run means no pilot provider request. SDK retries remain disabled;
paid credential-test probes and raw POST probes are disabled in this mode.
application retries debit separately. Normal exit releases only unused slots;
every attempted slot keeps its full $1 charge. A crash, failed ledger, or unknown
in-flight outcome retains the entire reservation. This can close the pilot early.

Returned input/output usage is recorded once for reconciliation evidence. Missing,
invalid and uncertain usage stays unknown; no tokenizer guess, optimistic price,
failed-call refund or usage-based discount replenishes the budget. These counters
bound real charges **only if** the operator proves that every permitted attempt
fits the $1 ceiling. They exclude hosting/identity expenses and other account
users. Use a dedicated project/credential and an independent enforced account or
gateway cutoff for the approved $250 exposure, including delayed metering and
in-flight requests. A dashboard alert alone does not establish an enforced cutoff.

Review every approved model, tier, region and surcharge, worst permitted input,
requested/reasoning output and request fees, retry/error charging and concurrent
in-flight charges. Prove the per-attempt upper bound with actual pricing/contract
and a bounded synthetic acceptance test. If $1 is insufficient, reduce the request
envelope or obtain a revised budget and cutoff before real use. Do not switch to
another provider/model for cost without its privacy/accuracy acceptance.

## Private control store and initial provisioning

The pilot Compose profile mounts `pilot-control` only into the web application at
`/app/pilot-control`; no parser, ingress or backup service receives it. The image
creates that directory with mode 0700 for the non-root runtime user. The database
and separate process-lock file have mode 0600. Use a local filesystem with proven
SQLite durable writes and OS locking; network filesystems and multiple hosts are
outside this implementation. Review the host/volume permissions and encryption.

After all policy/account/privacy approvals are ready and **before starting the
web application**, initialize a new ledger once using the reviewed image and
protected approval/environment files:

```sh
docker compose -f docker-compose.pilot.yml run --rm --no-deps streamlit-web python -m app.pilot_budget init
```

The initializer uses the configured `VA_LSE_PILOT_BUDGET_FILE` and creates files
exclusively. It refuses existing files. Normal startup never provisions a missing
ledger: it acquires the single-process lease before listening, checks the exact
policy, and recovers abandoned reservations conservatively. A missing, corrupt,
replaced, public, linked or unusable control file closes admission. A changed
policy or backward wall-clock step closes it too. Operators must maintain a
trusted clock; deliberate clock manipulation and privileged storage rollback
require host/gateway controls.

The app detects missing or replaced files and refuses mismatched policies. An
in-place rollback by a privileged administrator is not intrinsically detectable
from that rolled-back store; suspension and the independent gateway/account
ceiling must cover that risk. Never count the local ledger as a rollback-resistant
cross-host spending authority.

Do not remove the volume, use Compose `down -v`, initialize a new UUID/ledger,
restore an older copy or change budget files to recover credits in an active
pilot. Approval/build revisions must keep the same pilot ID, deadline and policy.
Loss or uncertainty requires suspension and reconciliation with actual account
usage by the operator. There is no automatic reset/replenishment command.

The store contains an HMAC-derived participant key (random local salt), random
run IDs, timestamps, numeric counters and token usage. It has no names, subject
claims, provider credentials, statement/record text, filenames, model responses
or errors. This is pseudonymous control metadata and still needs R08 approval.
Finished/abandoned run and attempt rows older than 24 hours prune on ledger access;
global counters, salt and policy remain for the whole budget period (at most
30 days). SQLite secure-delete clears deleted table payloads; that does not erase
host snapshots or establish physical erasure. Idle/stopped volumes are not erased
by this app. The privacy owner must
approve and schedule complete volume deletion after suspension and reconciliation,
including host snapshots, crash copies and physical-erasure limitations.
Clearing/signing out of a case must retain quotas. Control backups are excluded.

## Actual-host/account acceptance

| Check | Required result |
|---|---|
| B01 Policy and price | Actual account/model/region terms prove the attempt ceiling for the entire allowed request envelope; operator approves limits, period and alert/stop owner |
| B02 Independent cutoff | Bounded synthetic test reaches the configured account/gateway limit and further billable requests are refused, with metering lag/in-flight exposure accounted for |
| B03 Restarts | Two starts, a process/container/host restart and a third same-owner start cannot reset rolling limits or global budget; crash keeps reservations |
| B04 Tabs/workers/retries | Two tabs, concurrent participants, multiple clients, worker threads and all retries consume the same run/global envelope; failed or unknown calls retain charges |
| B05 Storage boundary | Only the web runtime mounts the private durable volume; second process refuses; missing/corrupt/replaced/rolled-back store causes suspension; no automatic creation/reset |
| B06 Active ingress | Upload limits and repeated actions on an already upgraded WebSocket refuse before provider work; handshake rate limiting alone is insufficient |
| B07 Operations | Named owner receives actual account alerts, suspends admission and reconciles remaining exposure; no volume reset or unreviewed provider change |
| B08 Privacy/deletion | Notice and retention approve pseudonymous control metadata, excluded backups and full stopped-volume deletion schedule; execute the synthetic deletion check |

Preserve non-secret evidence: immutable source/image revisions; host and volume
configuration; actual provider/project/model/region and dated price/cutoff
references; approved policy and notices; synthetic observed results; timestamp,
reviewer and disposition for each check. Do not put credentials or real records
in the repository or evidence bundle.

```json
{
  "source_revision": "REPLACE_WITH_REVIEWED_IMMUTABLE_REVISION",
  "image_identity": "",
  "policy_reference": "",
  "provider_price_and_attempt_bound": "",
  "actual_cutoff_test_reference": "",
  "metadata_privacy_and_deletion_reference": "",
  "alert_owner": "",
  "stop_operator": "",
  "checks": {"B01":"NOT RUN","B02":"NOT RUN","B03":"NOT RUN","B04":"NOT RUN",
             "B05":"NOT RUN","B06":"NOT RUN","B07":"NOT RUN","B08":"NOT RUN"},
  "independent_reviewer": "",
  "decision": "NO-GO"
}
```

Only after actual acceptance may `spending_controls` point to the signed review.
A nonempty reference alone is an operator attestation, not automated proof.
