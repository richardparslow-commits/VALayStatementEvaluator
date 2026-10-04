# R16 — Future durable case deletion: design and acceptance

**UNAPPROVED. All twenty actual-host checks are NOT RUN. Durable real-information storage remains excluded.**

The present pilot keeps working cases in the active session. Clearing a case revokes
session consent/exports and releases registered uploads and working state; it does
not certify physical memory erasure, delete durable cases or recall provider/user
copies. Durable blob factories, cached stores, direct filesystem/S3 operations and
cleanup refuse every non-synthetic mode, including unknown/blank modes. No flag,
worksheet or approval reference enables durable real-information case storage.
A separately implemented owner-aware case/deletion service and actual acceptance
are required. Use invented cases and private disposable test stores throughout
preparation; do not connect this tool to real records or existing cloud buckets.

## Implement before enabling

Accept current-release R07–R13 and expand R08 privacy/storage review. Optional R14
exports or R15 queues require their own capability acceptance and affected deletion
checks; neither becomes enabled here. A deletion service can be designed/tested
while queues remain excluded, avoiding an assumed circular activation prerequisite.

Use server-authoritative authenticated tenant/case/revision ownership and a complete
case inventory. Prefer case-scoped encrypted objects with protected key authority.
Inventory original uploads, extracted/derived text, results, temporary files, caches,
queued inputs/results/references/leases, prepared exports, object versions, replicas,
backups and restore sources. Content-addressed keys are not credentials, owners or
case IDs. If data is shared, use ownership-aware transactional references and test
concurrent attach/detach/deletion; case A's deletion cannot erase case B's authorized
copy. Do not infer authorization from a guessed key, request/job reference or hash.

Authenticate and authorize deletion/status requests, validate expiry/role/request
authenticity, and commit a durable access tombstone before physical cleanup. Revoke
case work grants, consent, leases and export/cache access, and enforce the tombstone
at retrieval, every new write, provider attempt, retry, late output and recovery.
Preserve deletion intent and progress across crashes. Stop-before-start/restart,
lost acknowledgment and partial storage failure must resume safely; an unknown
result stays pending. A recreated object or denied verification is not success.

Use private least-privilege storage, bounded operations and an accepted namespace
and destination inventory. Verify POSIX locking on the actual shared volume,
namespace link protections, concurrent replacement and interruption handling.
The current filesystem `delete` raises on failure; missing-file retries are harmless.
Its raw blob removal is not an authenticated case deletion transaction and does
not implement ownership-aware reference counting or durable tombstones.

The legacy S3 `delete` now queries bucket versioning, refuses Enabled/Suspended/
unknown or unqueryable states before deletion, and only attempts synthetic objects
in a verified never-versioned bucket. It rejects delete-marker/version-bearing
responses, verifies object absence and rechecks versioning. Failed, recreated or
unverified deletion raises. The required API permissions now include bucket
versioning inspection and object head/deletion. S3-compatible services that cannot
supply accepted versioning semantics remain unsupported for this deletion path.
On AWS, head verification needs object read permission and bucket listing authority
to distinguish absence (404) from denied access (403); see
[AWS head-object permissions](https://docs.aws.amazon.com/AmazonS3/latest/API/API_HeadObject.html).
These observations are not an atomic guarantee against administrative changes or
new writes after verification; pin configuration and coordinate all writers.

For a future version-aware implementation, enumerate exact object keys and version
IDs (including null IDs), noncurrent versions and delete markers with bounded,
complete pagination. Avoid prefix-only deletion of neighboring keys. Account for
replicas, multipart remnants, retention locks/legal holds and lifecycle rules;
do not bypass retention protection or treat delayed expiry as immediate erasure.
Use authoritative case/version mapping before destroying any object. Reconcile
partial/denied/ambiguous responses and accepted deadlines for retained copies.
A simple versionless S3 delete can insert a marker while older data remains;
see [AWS version deletion](https://docs.aws.amazon.com/AmazonS3/latest/userguide/DeletingObjectVersions.html)
and [bucket versioning states](https://docs.aws.amazon.com/AmazonS3/latest/API/API_GetBucketVersioning.html).

Install and observe independent idle/stopped-service retention with named owners,
accepted deadlines and private count-only failure/overdue alerts. A configuration
file is not an installed schedule. Missing inventory/mount/lock, storage pressure,
permission errors and incomplete passes must stay visible and close admission.
Keep backup/snapshot inventory, expiry/destruction authority and external provider
handling explicit. Restore old copies only into isolated invented-data fixtures;
current tombstones must deny deleted-case reads, replay and late writes after
restore. Do not restore permission from a historical snapshot. Physical erasure,
key destruction and storage/provider-held copies require their own measured and
reviewed guarantees. Local session clearing, TTL or a successful object call alone
cannot establish them.

Mode guards run before operations and after delayed replies, before subsequent
reads/cleanup steps. They do not recall already-issued writes/deletes, rendered
bytes or provider requests. Keep mode immutable in a running process and stop all
writers/workers/cleaners before changing the profile. Do not set synthetic mode to
operate on real information. Quarantine historical real-data stores through the
accepted incident procedure; destruction needs a separately reviewed inventory,
authorized operator and appropriate offline tooling. This release performs no
historical purge, cloud changes, lock bypass, provider request or notification.

## Actual-host observations

Every row begins **NOT RUN**. Record immutable source/images/config/storage policy,
tester identity, timestamps, observations, private evidence, failures/retests and
independent signature. An unimplemented requirement is a failure, not inapplicability.
Only separately excluded optional capabilities may have documented scope rationale.

| Check | Required observation |
| --- | --- |
| D01 | Separate authenticated case/deletion implementation exists; current pilot refuses durable blobs. |
| D02 | Exact-release R07–R13, expanded R08 and any separately enabled R14/R15 accepted. |
| D03 | Every owner/case/revision object, derivative, reference, version and retained copy is inventoried. |
| D04 | Known references cannot authorize anonymous, foreign, stale or inappropriate deletion/status/recovery. |
| D05 | Protected case encryption/keys, private roles/destinations and approved diagnostics verified. |
| D06 | Tombstone/revocation precedes physical deletion and prevents racing retrieval, new writes and late output. |
| D07 | Case A deletion preserves authorized case B data under concurrent reference changes. |
| D08 | Actual shared-volume locks, namespace protection, interruption/retry and failure visibility verified. |
| D09 | Bucket ownership/prefix/versioning semantics accepted; legacy versioned/suspended/unverified deletes refuse. |
| D10 | Exact version/marker inventory and deletion are bounded, complete and safe for adjacent/shared keys. |
| D11 | Replicas, multipart remnants, noncurrent versions and retained/locked copies handled within accepted policy. |
| D12 | Durable pending/complete states survive crash, denial, lost acknowledgment and retry without false completion. |
| D13 | Installed idle/stopped-service cleanup meets deadlines; failures/overdue work alert and close admission. |
| D14 | Backup/snapshot inventory, expiry and protected destruction authority accepted. |
| D15 | Restore cannot resurrect deleted-case access, replay or delayed writes; authoritative tombstones survive. |
| D16 | Deleted invented case inaccessible through every active application path within policy deadlines. |
| D17 | Actual provider/user-held copy process and already-sent/saved-copy limits documented. |
| D18 | Named owners rehearse failed cleanup/deletion incidents and private alert acknowledgment. |
| D19 | Intended participants can access deletion controls and understand session, pending, complete and retained states. |
| D20 | Exact-release operator/security/privacy signoff resolves every access/inventory/lifecycle failure. |

Stop admission/testing on cross-case access, lost deletion intent, unidentified
copies, false completion, overdue unhandled retention, restore resurrection or an
unreviewed destination. Rehearse approved synthetic incidents only; do not send
messages, change cloud settings or delete external data through this preparation.

## Offline worksheet

From a clean committed checkout, write a new file under a pre-existing protected
parent directory:

```sh
python scripts/deletion_review.py --repo . --out /approved/private/deletion-evidence/r16-draft.json
```

The tool checks every tracked byte, including binary assets, hidden index changes
and linked paths. It freezes an **unapproved** exact-source worksheet with twenty
`not_run` observations, blank authority/evidence/signatures and a NO-GO decision.
It refuses an existing output and performs no deletion, host/provider/alert action,
approval validation or activation. Private `deletion-evidence/` folders are excluded
from Git and container builds at root and nested paths. Source/config/storage policy,
key authority, lifecycle or host changes require fresh review and applicable tests.
Preserve old private evidence for its accepted retention period; never carry a
signature forward as though a changed release had already passed.
