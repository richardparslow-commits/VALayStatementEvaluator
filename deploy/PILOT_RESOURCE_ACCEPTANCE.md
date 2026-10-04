# Resource admission and overload acceptance

Status: **actual-host acceptance OPEN**. These controls address ingestion audit
A01–A04 in code. They do not approve a real-information release or replace the
revision-specific deployment, privacy, spending, accuracy and incident reviews.
Use invented records for every exercise below and retain only non-secret results.

## Supported boundary

Start the controlled pilot through `python -m app.pilot_server`, which runs
`app/pilot_asgi.py`. The reviewed signed-cookie/upload adapter requires the locked
Streamlit **1.63.0** and a root HTTPS deployment URL. Other versions fail startup;
prefixed upload paths are refused. Screen admission refuses a runtime that has
not served HTTP through the upload middleware. A plain Streamlit launch cannot
open the pilot uploader.

The ingress must be private, terminate TLS, overwrite forwarded headers and
prevent access to the app port. The middleware verifies the signed OIDC cookie,
current invitation/token expiry, exact configured origin and a server-owned
session/consent binding before calling `receive()`. The URL's session identifier
and a caller-supplied identity header do not grant access. Consent and identity
are checked during body receipt and before a successful upload is committed.
DELETE checks ownership; Streamlit still applies its XSRF checks to both methods.

## Enforced defaults

These are conservative implementation ceilings for this single-process profile.
The operator must accept them and their measured working headroom on the actual
release. Configuration may lower the byte caps; it cannot raise pilot ceilings.

| Resource | Ceiling / behavior |
| --- | --- |
| File body | 50 MiB per file, plus at most 1 MiB total multipart framing |
| Participant retained raw uploads | 200 MiB and 32 files across sessions/workflow slots |
| Process retained raw uploads | 512 MiB including pending worst-case reservations |
| Simultaneous body transfers | Two process-wide; one per participant; excess refused immediately |
| Participant upload PUT attempts | 32 per rolling minute; 128 per rolling hour, including owned attempts refused for size/concurrency/capacity |
| Rate history | Bounded to 20 owner buckets; clearing a case does not reset history; process restart resets this in-memory upload history |
| Registered sessions | 20 process-wide |
| Retained parsed source text | 20 Mi characters per participant; 80 Mi characters process-wide |
| Parsed extraction identities | 32 per participant; at most 640 process-wide, including zero-length identities |
| Body receipt/handler deadline | 60 seconds elapsed, independent of incoming trickles |
| Upload memory admission | At least 384 MiB available for the first transfer, 768 MiB for a second pending transfer |
| Analysis work lease | At most one per participant and two process-wide, retained through completion of the primary future and tracked child futures |
| Analysis memory admission | At least 200 MiB per currently reserved work slot, up to 400 MiB for a second slot |
| Queued digest/merge work | At most twice the configured worker count, including running futures |
| Process-wide paragraph index cache | Four million characters; a single oversized index is not retained |
| Pilot shared source caches | Paragraph indexes and matching-token source arguments bypass the process-wide caches; synthetic mode keeps its caches |
| Local OCR tool subprocess | 120 seconds per tool; 4 MiB combined stdout/stderr; owned POSIX process group terminated and parent reaped |

The existing durable analysis ledger still imposes **one active analysis** and its
reviewed spending/daily-start limits. The work lease is an additional guard for
threads that survive cancellation; it does not increase that analysis allowance.
The ingress retains its existing IP request/connection limits.

Reservations pessimistically charge a complete maximum-size file before receipt,
including a replacement's old bytes. A small new file may therefore be refused
when less than a full file allowance remains. Raw file bookkeeping is released
after manager removal or confirmed absence. Dead/revoked/inactive sessions are
swept on subsequent guarded requests or admission; this is not a background
deletion SLA. Clear case removes the session's raw uploads. Pending transfers
keep their reservation until cleanup finishes.

Parsed-text reservations remain charged until the case's consent object is no
longer retained, including by running workers. Removing/replacing a selected file
does not refund this envelope immediately; clear the case to select a new set.
No original source text is truncated to fit a resource limit. A pilot batch with
any rejected/skipped selected file cannot proceed as an accepted subset.
Pilot matching bypasses the shared paragraph/token caches so cleared cases do not
accumulate original source strings there. Recomputing matching adds CPU work;
include it in the maximum-case host measurement. This does not establish deletion
of all other app, operating-system, provider or recipient copies.

## What is buffered, and what is incremental

Streamlit still buffers the admitted multipart request and retains uploaded file
bytes. Parsing still retains the admitted source pages. This is bounded admission,
not a stream from the browser directly to the model. ChunkPlan counts/replays the
retained pages and materializes only current chunk windows/page working data and
the bounded submitted work window. Chunk count, order, overlap and source markers
are preserved. ZIP metadata is validated first; member bodies are then expanded
and parsed individually with actual-byte quotas, rather than retaining every raw
expanded body at once. Compatibility list helpers remain for existing callers.

Linux headroom reads the stricter available host and finite cgroup v2/v1 figure
at the supported container paths. Unknown headroom refuses pilot admission. This
is a point-in-time admission heuristic; heap overhead, parser replies, transport
buffers and other processes still require measurement under the 3 GiB web limit.
The cgroup limit remains the hard boundary and can still trigger OOM if actual
working memory exceeds the measured envelope. Kernel documentation distinguishes
current usage from the hard memory limit: [cgroup v2 memory controls](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html#memory).

The proxy disables request buffering and forwards using HTTP/1.1. Its possible
upload temporary directory is explicitly placed on a 128 MiB tmpfs, and the proxy
has CPU/memory/PID limits. Do not assert that disk-spooling is impossible on an
unverified host: confirm the effective configuration, mounts and chunked-request
behavior. See [NGINX request buffering](https://nginx.org/en/docs/http/ngx_http_proxy_module.html#proxy_request_buffering).

## Timeout containment and its remaining limit

A pipeline timeout cancels queued work and blocks further phases/retries. The
participant cannot start another upload or analysis while that run's work lease
is held. Python does not forcibly stop already running thread futures when
cancelling/shutting down an executor; see [Executor.shutdown](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor.shutdown).

Each pilot provider attempt owns a separate SDK/HTTP pool. Its watchdog is always
enabled, is clipped to the remaining pipeline deadline, and closes only that
attempt's pool. Normal/exceptional returns close the private client. Socket rescue
uses transport internals and is best-effort. If a call does not exit, capacity stays
closed; the operator must investigate/restart under the approved incident process.
A guaranteed hard cutoff for arbitrary blocked Python/transport code still
requires process isolation. This release does not claim that guarantee.

Local OCR now bounds its external commands and queued page tasks, but remains
excluded from the controlled pilot. Its PDF inspection, raster dimensions,
document fidelity, tool versions and separate sandbox acceptance remain open.
The existing isolated pilot parser retains its container CPU/memory/process,
input/output and supervisor deadlines without an in-process fallback.

## Required actual-host evidence

Record the reviewed commit/image IDs, host/cgroup layout, proxy configuration,
operator, date, observed peaks/status codes, cleanup times and evidence location.
Keep all checks below `not_run` until performed against private staging.

1. Verify an invited, consented session can upload and delete; missing/expired/
   revoked cookies, another participant's session, missing XSRF and foreign origins
   are refused. Demonstrate unauthorized traffic reaches no body parser and leaves
   no file. Withdraw consent during transfer and before commit and verify removal.
2. Exercise overlapping uploads across tabs/participants, byte/count/text quotas,
   replacement uploads, upload rate exhaustion and case clearing. Check process
   peak RSS and `memory.current`, active transfers and file counts. Confirm the
   earlier record set cannot run when a selected file was refused.
3. Use declared-length and chunked slow bodies, malformed multipart and bodies
   beyond the actual-byte cap. Verify the elapsed deadline, connection behavior,
   released reservations, proxy temp mount and absence of retained content.
4. Under a lower cgroup limit, leave host memory plentiful and verify refusal
   before new body receipt/work. Exercise unavailable counters and prove fail-closed
   behavior. Check the actual cgroup path/parent constraints, not only fixtures.
5. Run the maximum approved record set, repeat upload/remove/clear cycles, and
   demonstrate bounded RSS, queued tasks, source cache entries and archive working
   data. Verify first/middle/final evidence and retry coverage against invented truth.
6. Stall/trickle the approved provider using a synthetic private test endpoint.
   Verify the attempt's watchdog cannot close another pool, new work stays refused
   until the original call/children finish, and the incident procedure restores
   service when a transport cannot be interrupted. Measure the actual timer cutoff.
7. Kill/disconnect the browser, cancel/clear a case, interrupt a parse and restart
   the stack. Verify file/reservation cleanup, bounded leftovers, durable analysis
   quotas and non-sensitive logs. In-memory upload rate counters reset on restart;
   record why private ingress and the independently durable controls are adequate.

Local unit/ASGI fixtures establish code behavior only. No actual-host acceptance,
provider cutoff, privacy deletion completeness or model accuracy approval is
recorded by this document. Other ingestion audit findings remain separate work.
