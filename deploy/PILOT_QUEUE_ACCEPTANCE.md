# R15 — Future queued processing: design and acceptance

**Status: UNAPPROVED. Every actual-host check below is NOT RUN. Queued real-information processing remains excluded.**

The current release rejects queue construction/use, cached backends, queue serialization,
worker entrypoints and result hydration whenever `VA_LSE_MODE` is not `synthetic`.
The controlled profile keeps `VA_LSE_JOB_QUEUE=0`. No flag, worksheet, reference or
approval-file entry can enable real-information queueing. A separately implemented
and reviewed queue profile is required before positive real-host acceptance tests.
Use invented records during design and testing. Never place participant records,
credentials or reusable browser/OIDC tokens in this worksheet or public evidence.

## Required design before activation

Accept R07–R13 for the exact future release, expand R08 privacy/provider/storage
review and R09 spending acceptance to include all workers and persistent stores,
and complete R16 durable case deletion. The current session consent event and
single-process SQLite budget ledger cannot authorize a distributed worker that
outlives the browser or allocate independent spending allowances to each process.
Do not remove the exclusion guards as a substitute for these implementations.

Define an authenticated server-authoritative tenant, case, revision, job and owner
relationship. Submission, status, recovery, cancellation and result access must
recheck it. A guessed job/request reference grants no access. Store bounded,
expiring case-specific work grants backed by current consent/revocation authority,
not copied browser credentials. At claim, every provider attempt/retry, delayed
output and later retrieval, recheck owner/case access, grant expiry/revocation,
current release, approved provider destination, exact model version and notice.
A disconnected browser does not imply continuing consent. Define and review its
explicit lifetime and participant-facing behavior.

Use a private TLS queue with separate least-privilege producer, claimant, result,
health and retention roles and protected credentials. Deny public/cross-role
access and development fallback. Fix no-eviction behavior and measure realistic
synthetic capacity, payload/result/key bounds, TTLs, backpressure and independent
cleanup. Bind accepted persistence/fsync settings, crash behavior and measurable
recovery point/time objectives to the actual Redis instance, host and immutable
images. In-memory fake transport tests do not establish persistence guarantees.

Use one shared durable budget authority for participant starts, active ownership,
attempt reservations and uncertain charges across every web/worker process,
restart and partition. The proposed limits are $250 total, two starts per participant
per rolling 24 hours, 50 attempts per run and a provisional $1 maximum charge per
attempt ($50 run reservation). Provider price/cutoff acceptance remains R09 work;
the provisional amount is not a proven price ceiling. Failed, retried, timed-out
and ambiguously charged attempts keep their full reservation. Recovering or
replaying work must acquire/reserve every possible extra paid attempt before it
starts. Test provider idempotency only where the actual provider supports it.
Neither queue idempotency nor lease fencing guarantees exactly-once paid work.

Fencing tokens reject stale progress/result/completion/failure writes. They cannot
recall an already-sent provider request or stop a partitioned worker from spending
without a shared fail-closed authority. Exercise lease expiry, heartbeat races,
worker death/restarts, concurrent recovery, delayed responses, cancellation,
revocation and shutdown. Persist uncertainty instead of blindly resubmitting.

Maintain a per-case inventory of inputs, results, recovery references, leases,
shared blobs, object versions, replicas and backups. Implement authenticated,
retryable deletion, independent idle retention and failure escalation; deny deleted
case access before and after restore, including delayed worker writes. A second
case sharing an object must retain its authorized data. Provider and participant
copies have separate accepted handling. No backup restore or blob storage becomes
approved through this worksheet.

Submission SHA-256/kind/version/request checks run before following document
references. Result kind/request checks run before screen hydration. These detect
mismatches; an attacker who controls the entire datastore can alter metadata and
hashes together. These checks do not authenticate compromised storage or replace
case ownership, authorized worker roles, durable grants or output review. Jobs
without a recorded digest are refused: re-submit original invented inputs for
synthetic testing, never automatically migrate historical real information.

Mode checks run before operations and after delayed replies, including callbacks.
They discard late responses and prevent subsequent sensitive steps. They do not
atomically roll back previously issued storage writes, rendered bytes or provider
requests. Keep mode immutable for a process and stop every producer/worker before
changing the deployment profile; verify stop/revocation on the actual host.

## Actual-host observations

Every row starts **NOT RUN**. Record tester identity, immutable revision/images,
effective configuration hash, timestamps, observations, private evidence reference,
failures/retests and independent reviewer signature. Preparation tests and CI
passes never mark these rows accepted. An unimplemented requirement is a failure,
not inapplicability.

| Check | Required observation |
| --- | --- |
| Q01 | Separate queue profile implements this design; present pilot still refuses queueing. |
| Q02 | Exact-release R07–R13 and expanded R08/R09 plus R16 accepted. |
| Q03 | Tenant/case/revision isolation at submit/status/recovery/cancel/results; known references cannot cross accounts. |
| Q04 | Durable consent expiry/revocation works during delayed or disconnected worker activity. |
| Q05 | Altered/expired model, provider, notice and release approvals refuse every subsequent attempt. |
| Q06 | Private TLS, role permissions, secret protection and public/cross-role denial; no fallback. |
| Q07 | No-eviction capacity, maximum payload/results, TTLs and pressure behavior measured. |
| Q08 | Actual Redis/host crash, fsync, restart and ambiguous/lost submission meet accepted RPO/RTO. |
| Q09 | Swapped/truncated/unsupported inputs and mismatched results refuse before record retrieval/hydration. |
| Q10 | Concurrent workers, restarts and partitions cannot duplicate start/active/budget allowances. |
| Q11 | Actual worst-case provider price/cutoff and conservative uncertain-charge reservations accepted. |
| Q12 | Duplicate/lost acknowledgments, unknown provider charges and supported idempotency exercised. |
| Q13 | Reclaimed workers and racing sweepers cannot write using an old lease token. |
| Q14 | Cancel/revoke/delete during provider delays stops further attempts and withholds late output. |
| Q15 | Complete inventory, TTL and independent idle cleanup verified; incomplete/failed cleanup closes admission. |
| Q16 | Case deletion and restore stay denied across every path; another authorized shared-object case stays intact. |
| Q17 | Invented content canaries absent from every accepted diagnostic/alert/log/trace/backup sink. |
| Q18 | Named owners rehearse stop/revocation/credentials and capacity/retention/billing alert acknowledgment. |
| Q19 | Intended participants understand background consent, cancellation, expiry, references and uncertain charges with accessible controls. |
| Q20 | Immutable release/config and independent operator/security/privacy signoff resolve every finding. |

Stop testing/admission on cross-case access, stale permission, unreserved paid work,
missing inventory, unbounded retention, unapproved destinations or exposure.
Exercise approved synthetic incident procedures only; do not send messages or
change provider/account settings without explicit authorization.

## Offline preparation

From a clean committed checkout, prepare a new private file:

```sh
python scripts/queue_review.py --repo . --out /approved/private/queue-evidence/r15-draft.json
```

The output path must already have a protected parent directory. The tool refuses
an existing output and checks every tracked source byte, including hidden index
changes, links and binary release assets. It hashes a source-bound **unapproved**
worksheet with twenty `not_run` observations, blank authority/evidence/signatures
and a NO-GO decision. It performs no host, provider, alert or acceptance actions;
it cannot validate approvals or enable queues. Keep private evidence in an ignored
`queue-evidence/` directory excluded from both Git and container builds. A later
source/config/model/host/lifecycle change requires fresh review and applicable
retests. Retain old evidence for the accepted private audit period; do not overwrite
it or copy a signature into a new revision's worksheet.
