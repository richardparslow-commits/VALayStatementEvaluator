# R14: authenticated statement TXT export acceptance

Status: **UNAPPROVED / ACTUAL-HOST CHECKS NOT RUN**. The service is prepared,
disabled by default, and cannot admit real information until R07–R13 and this
capability's independent acceptance are complete. Use invented canaries only
during acceptance. A successful repository test or merge is not activation approval.

## Fixed capability and limits

Only the exact edited draft or evaluation rewrite that completed source/witness
review and the explicit export confirmation may be prepared. A separate native
button retains the UTF-8 bytes; a native link downloads them from
`/pilot-exports/<opaque handle>`. There is no automatic creation on reruns.
PDF, Markdown, reports, digest/timeline/CSV/DOCX and every legacy Streamlit media
download remain disabled. Unresolved confirmation placeholders are refused.

Each GET verifies the current protected approval, invited identity, issuer,
signed cookie and token expiry, owning participant, live session/review lease,
unchanged full approval and current consent. A handle alone gives no access.
Anonymous, foreign-owner, expired and revoked identities receive the same fixed
404 refusal. This does not implement remote identity-provider logout notification:
an otherwise valid signed identity lasts until its expiry or invitation removal.
Use the app's **Clear case and sign out** to revoke this case before logout.
Revoking an invitation or stopping admission must be exercised on the actual host.

Artifacts are held in process memory only: 1 MiB per file, 8 MiB and 64 files
per process, five-minute monotonic expiry without renewal by downloads/reruns,
and an independent one-second idle sweep. A new explicit preparation after expiry
requires the still-current review and consent. Edits to reviewed text or source
selections, review withdrawal, a new run, and clearing/sign-out revoke the lease.
When the owning session is disposed, the weak lease is released and the next
sweep drops its bytes. The approved disconnected-session TTL remains at most
60 seconds. Restart and ASGI shutdown drop all artifacts; an inactive cleanup
worker refuses delivery. Reference release is not a promise of memory zeroization.

Delivery checks run when the HTTP response is sent. They cannot recall bytes
already authorized in an in-flight response, files already saved, browser-held
copies or external/provider records. Explain these limits in the notice and
participant comprehension checks; clear-case cannot promise external erasure.

Responses use `text/plain; charset=utf-8`, a fixed `reviewed_statement.txt`
attachment filename, no identifiers in headers, no compression, no MIME
sniffing, no referrer, sandbox CSP and private/no-store caching. HEAD, Range,
query-bearing handles, cross-site origins and malformed handles do not deliver
case bytes. No redirects, public object store, disk artifact or bearer fallback
is used. Opaque URLs still must not be logged or shared as acceptance evidence.

## Required operator configuration and evidence

Use the single-process controlled launcher `python -m app.pilot_server`, the
locked Streamlit **1.63.0**, the reviewed root HTTPS origin and its private
ingress. `server.baseUrlPath` must be empty. The custom route uses the supported
[Streamlit st.App API](https://docs.streamlit.io/1.63.0/develop/api-reference/server/st.app).
Its bounded cookie adapter is intentionally pinned to this version; changing
Streamlit requires protocol review, route tests and new exact-release acceptance.
The [Streamlit authentication guidance](https://docs.streamlit.io/develop/concepts/connections/authentication)
describes its identity-cookie behavior; download authorization additionally
checks the retained token claims on every request.

The app port must remain private. The reviewed TLS proxy overwrites forwarded
scheme/host headers, disables access URL logs, and must not cache, compress or
buffer case responses. Never trust a public client-provided forwarded scheme.
The supplied Nginx profile disables access logging and proxy buffering; the
pinned Streamlit CLI disables Uvicorn access logging. Verify any load balancer,
CDN, APM, request tracing, error collection or alternate runner independently.
No secrets, cookies, handles, case text or private approval observations belong
in Git, image layers, analytics, shared reports or provider requests.

Keep private observations under `export-evidence/` outside deployment source.
That folder is excluded from Git and Docker at any depth. Record the exact
40-character source SHA and tree, immutable web/parser image digests, locked
runtime, actual private origin, proxy/configuration hashes, named operator,
independent security/privacy/accessibility reviewers, UTC test times and expiry,
all checks below, remediation/retest references and authenticated sign-off.
Store signed original evidence privately; the approval's evidence reference is
only a locator and is not itself cryptographic verification.

Keep `VA_LSE_PILOT_TEXT_EXPORTS=0` and `text_export_policy.enabled=false` while
preparing evidence. After independent acceptance of the exact release, the
operator may enable the environment flag and the protected approval's optional
policy, with **exactly** these keys and scope:

```json
"text_export_policy": {
  "enabled": true,
  "formats": ["txt"],
  "evidence_reference": "private reference to separately accepted exact-release evidence"
}
```

The example approval stays incomplete and expired. The existing full approval
and current R07–R13 references remain mandatory. Update the participant notice
for retention, destination, revocation and saved-copy limits; the optional
policy participates in consent binding, so participants must consent again.
Repeat R13 testing for both native buttons, keyboard/focus, screen reader,
expired/refused state and teach-back. Existing download-disabled acceptance
does not transfer to this feature. Roll back by disabling the flag, stopping
admission and restarting the single process to discard outstanding artifacts.

## Actual-host checks

Every check is currently **NOT RUN**. Retain observations privately, with counts
and pass/fail labels in shared summaries. Test the reviewed host, actual OIDC
cookie, private proxy and supported participant browsers; synthetic signed
cookies in unit tests do not prove these boundaries.

| ID | Exercise and required observation |
|---|---|
| X01 | Default flag off, absent/disabled/incomplete policy, unsupported runtime and alternate launcher: no TXT creation or delivery; every legacy media format remains unavailable. |
| X02 | Actual invited A reviews a canary draft and rewrite, explicitly prepares and downloads: bytes exactly match each reviewed UTF-8 text, fixed attachment/type, no active HTML or external asset request. |
| X03 | Anonymous and invited B request A's known handle, including from a fresh browser: same fixed refusal, zero case bytes. Same-owner access remains permitted only while A's owning review session is live. |
| X04 | Expired tokens, uninvited/removed subjects, wrong issuer, stale notice, withdrawn consent or changed approval: known-handle delivery refused on the next request; admission and other delayed actions remain closed. |
| X05 | Edit and revert the reviewed statement; change supporting passages/context; uncheck either review; start a new run: previous handle refused and a new export requires current exact-text review. |
| X06 | Clear case and clear/sign-out: old handles refused, consent revoked, working state/uploads released. Cookies removed in the sign-out browser; exercise other tabs and document provider/session logout limits. |
| X07 | Close/disconnect the browser without further traffic: session disposal within the accepted TTL releases the lease, and the next one-second sweep releases bytes. Reconnecting cannot restore a disposed case. |
| X08 | Idle five-minute expiry, downloads at the boundary, repeated preparations/reruns and system wall-clock changes: monotonic expiry, no extension of retained artifact lifetime, fixed refusal after expiry. |
| X09 | Unicode/oversized/unreviewed/placeholder text and 1 MiB, 8 MiB, 64-file limits: bounded explicit refusal, no hidden disk spill; expired/revoked items free capacity. |
| X10 | Actual large/chunked signed cookies, duplicate cookies, malformed/compressed inputs, future/invalid claims and version mismatch: bounded parsing and fixed refusal without crash, secret exposure or body logging. |
| X11 | Host/origin/forwarded-scheme/cross-site, query, path traversal, HEAD, Range, replay and alternate ports: no case delivery outside the accepted root/private TLS boundary; app port unreachable externally. |
| X12 | Browser/network/cache inspection: exact private/no-store/nosniff/no-referrer/sandbox/type/filename headers, no public cache, ETag, redirect or compression; approved destination only. |
| X13 | Review all actual proxy/runner/CDN/APM/audit/error/swap/crash layers using canaries: no cookie, handle, text or participant identifiers in retained observations; account for memory/crash retention separately under R08. |
| X14 | Concurrent GET, edit/clear/revoke, cleanup failure, shutdown/restart and rollback: authorization rechecked at delivery, inactive/shutting service refuses, restart loses artifacts. Record the already-authorized in-flight limitation. |
| X15 | Keyboard, screen reader, focus and representative participant teach-back: distinguish review from preparation/download, understand expiry/refusal, approved destinations and copies the app cannot erase. Complete affected R13 checks. |

Any failure, missing reviewer, unverified signature, stale release/configuration
or unanswered lifecycle/privacy question keeps R14 **OPEN** and the capability
disabled. Acceptance never approves PDFs, queues, durable storage or optional
integrations; those remain separate releases under R15–R17.
