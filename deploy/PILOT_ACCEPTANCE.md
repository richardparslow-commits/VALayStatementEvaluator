# R07: acceptance on the actual private pilot

Status of this document: **test procedure, not completed deployment evidence**.
Keep real-information admission **NO-GO** until every applicable check passes on
the actual host and the named reviewers accept the evidence. Local tests and
GitHub runners cannot substitute for that host. No deployment is created by this
procedure. Do not invent evidence references to make the application start.

Use synthetic accounts, records and witness text throughout. Record only
non-secret configuration, opaque test references, numeric counts and outcomes.
Keep cookies, tokens, credentials, actual OIDC subjects, records and unredacted
browser/network captures out of the repository and issue trackers.

## Prerequisites and binding

The operator supplies the private staging URL, access method, real OIDC issuer,
named host/security owner and authorized test identities: participants A and B,
operator O, and uninvited U. Use independent browser profiles for each identity,
plus an anonymous profile. A and B must not share a browser cookie store.
Account configuration, paid provider calls and deployments require their own
authorization; this checklist does not grant it.

Before authenticated checks, satisfy the application manifest's prerequisite
reviews with truthful, current evidence. Its `deployment_validation` reference
can identify the existing reviewed host evidence packet being extended here;
its presence is not a completed R07 verdict. If prerequisites are missing, keep
admission closed, collect only host/proxy evidence that is independently
accessible, and mark the dependent tests **BLOCKED**.

The separate [ingestion acceptance](PILOT_INGESTION_ACCEPTANCE.md) now requires
an `ingestion_security` review reference for the passive-file policy and residual
malware decision. Old manifests without that reference keep admission closed.

Bind the packet to these actual values before testing:

| Field | Required observation |
|---|---|
| Deployment | Private HTTPS origin, hosting platform and UTC test window |
| Source | Full reviewed commit and source tree SHA; clean build checkout |
| Images | Immutable IDs/digests of web, parser, launcher and proxy; running containers use those images |
| Runtime | Linux host, Docker version, AppArmor/default seccomp, non-root web/parser, container limits and private mounts |
| Identity | Exact issuer and redirect URI; token lifetime at most one hour; confidential client; MFA enforcement policy |
| Network | Private ingress policy, host/container routing and enforced egress policy; approved provider and identity destinations |
| Reviewer | Named operator/security reviewer; no secrets or raw subject IDs in this packet |

Read the running container metadata, not just Compose text. Record one web
instance and a stop-before-start upgrade procedure. An image label or environment
variable is an assertion; also compare the running image digest with the
reviewed build artifact. A changed image, issuer, proxy, network policy or host
invalidates the affected results and requires re-test.

## Checks and expected results

For each row, record **PASS**, **FAIL**, **BLOCKED** or **NOT RUN**, UTC time, tester
alias, observed result and a protected evidence reference. Unexpected success is
a failure. Only the security reviewer can accept a documented inapplicability;
absence of a service must itself be demonstrated. Start each case with unique
synthetic canaries for A and B so crossed content is recognizable.

| ID | Exercise | Required result |
|---|---|---|
| D01 | Connect from approved private access and a denied external/client network; try host ports 8501 and 8001 directly. | Only approved private ingress reaches the proxy. Direct app and diagnostics ports are inaccessible from participant/untrusted networks. |
| D02 | Validate certificate/hostname/chain and HTTPS-only ingress; connect WebSocket from the approved origin, then an unrelated origin; send an upload without a valid XSRF token. | Valid TLS and same-origin WebSocket work. Unapproved origin and missing/invalid XSRF uploads fail. No insecure fallback or HTTP credential exchange. |
| D03 | Visit `/health`, `/ready` and `/metrics` through the proxy; check path/trailing-slash variants and private sidecar binding. | No global diagnostics or sidecar content reaches a participant. Private operator checks remain possible. |
| D04 | Visit anonymously, authenticate as U, try incorrect issuer/redirect and replay invalid, expired or tampered authentication material using a sanctioned test client. | No uploads, case actions, records or diagnostics are accessible. OIDC state/nonce/signature/audience checks work in the actual integration. Never save authentication material in this report. |
| D05 | Fresh sign-in for A/B/O, including bypass/recovery paths and existing identity-provider sessions. | Actual policy requires MFA for every allowed path. Subjects use exact verified issuer/subject identities; changing displayed name/email cannot change authorization. |
| D06 | A uploads canary A and B uploads canary B; open known/guessed request references, try recovery/status/cancel/result paths, switch profiles and inspect any generated media URLs. | Neither participant can access or operate on the other's case. Queue/recovery and file downloads stay disabled. No case media/download URL is created. |
| D07 | A accesses global diagnostics and provider/configuration widgets; O accesses allowed count-only diagnostics. | Only O can see global operational diagnostics. Neither role can redirect managed credentials or view keys/tokens. Operator role does not grant another participant's case. |
| D08 | Revoke A in the protected manifest during a delayed request and again after the final call but before run completion; remove/expire the manifest. | Subsequent requests stop. Delayed responses/completed runs are refused, not stored as results. The next application interaction clears working case state and stops rendering. Already-sent requests cannot be recalled. |
| D09 | Let the real short-lived token expire with a tab open, during a delayed request and before saving a result; retry with the persistent identity cookie. | Expiry blocks sensitive actions/results despite the cookie. No retries or new provider calls follow an admission refusal; quota slots are released. |
| D10 | Open multiple A tabs, clear/sign out in one, switch identity, disconnect beyond 60 seconds, reconnect and restart the process. | Clearing/sign-out release that session's values and registered uploads; identity changes do not inherit a case. Disconnected sessions expire as configured. Demonstrate the documented limits below for other tabs and provider copies. |
| D11 | From the actual web network namespace, test approved provider/OIDC HTTPS connectivity and connection attempts to a harmless operator-owned unapproved destination, private/link-local metadata, IPv4 and IPv6 targets. Inspect host firewall/routing policy. | Approved required destinations work. Unapproved outbound traffic is denied by the network, independently of application checks. No arbitrary shell inside the app or contact with third-party targets is needed; use an operator-owned diagnostic process with the same policy. |
| D12 | Complete `PARSER_ENGINE_ACCEPTANCE.md`; run all current `tests.test_parser_container_live` methods with the exact parser image/revision and dedicated private mutual-TLS engine; inspect host/mount/resource metadata and cleanup. | No skips. Parser daemon is on a separate reviewed host/VM with a different engine ID and no application assets. Web/launcher/parser have no application Docker socket; launcher has only dedicated-engine credentials. Parser has no network/application mounts. TLS/identity/protection failures close admission. |
| D13 | Test 50 MB file, 200 MB aggregate and 500 page boundaries, oversized/hostile synthetic inputs, one active run, third start within an hour, ingress throttling and resource pressure. | Actual deployed limits refuse excess before processing. Parser remains bounded/cleaned; health checks work. Counters resetting on restart remain an R09 gate; do not infer durable quotas from this test. |
| D14 | Exercise permitted provider timeout/redirect/error paths and search every configured log/telemetry/backup sink for synthetic canaries; render literal malicious Markdown/HTML and inspect browser requests. | No content canary enters diagnostics or unapproved destinations. Redirects/proxies and optional tools remain blocked; record/model images do not cause browser egress. Account spending/provider-policy acceptance remains separate. |
| D15 | Inspect replica counts and perform the documented stop-before-start upgrade/restart while observing containers and ingress. | At most one web instance admits actions throughout. Image changes require new revision-specific evidence; there is no rolling overlap using process-local quotas. |

For provider-delay/error exercises, prefer an already approved synthetic fixture.
Fixture results do not establish the real provider's integration. Repeat the
applicable checks with the exact approved destination only when the account,
privacy and spending approvals permit it; otherwise leave them **BLOCKED**.

Streamlit sign-out affects the current session; other already-open sessions can
remain signed in. The application rechecks token expiry/invitations on reruns,
before/after provider calls and before accepting a run/result. It does not
continuously introspect identity-provider revocation or remotely erase content
already displayed in a browser. For removal across tabs, remove the invitation;
for immediate incident suspension, stop the application. Confirm this behavior
with the [Streamlit authentication documentation](https://docs.streamlit.io/develop/concepts/connections/authentication)
and the actual issuer. If instantaneous global logout is required, the pilot
must remain closed until that additional control exists.

## Evidence packet and release decision

Use the following structure in operator-controlled storage. This blank template
is deliberately **NOT RUN** and is not an approval. Do not copy it into the
approval manifest as if acceptance were complete.

```text
R07 deployment evidence
Deployment / platform / UTC test window:
Reviewed source commit / tree:
Actual web / parser / launcher / proxy image IDs:
Host and effective policy evidence references:
OIDC and MFA policy evidence references:
Check ID | status | tester alias | UTC | actual observation | evidence reference
D01 ... D15: NOT RUN
Failures / blocked tests / required follow-up:
Operator / security reviewer / decision timestamp:
Decision: NO-GO
```

Keep original failed observations when re-testing; append the correction and new
result. After all applicable checks pass, the security reviewer records a
revision-specific acceptance decision and the manifest references that final
packet. R07 acceptance alone does not close provider/privacy (R08), durable
spending (R09), factual quality (R10), legal (R11), operations (R12) or participant
accessibility/comprehension (R13) gates.
