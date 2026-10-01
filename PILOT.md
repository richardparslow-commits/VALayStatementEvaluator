# Controlled pilot with real veteran information

This change prepares a **single-instance, invite-only pilot**. It does not authorize
a deployment to accept real information by itself. The application defaults to
`VA_LSE_MODE=synthetic`; use synthetic records in that mode.

The controlled pilot blocks access until an operator supplies current,
revision-specific review evidence. The example approval file is deliberately
expired and incomplete. Do not replace evidence references with a general
"approved" statement or fabricate a completed review.

## Released pilot scope

- At most ten invited OIDC subjects, with a separately listed operator role.
- One application instance; one active analysis; at most two starts per participant
  per hour. These process-local counters reset on restart. The operator must also
  enforce an account spending limit and ingress limits.
- Uploads only: at most 500 source pages per case, 50 MB per file, 200 MB total.
  Each parse runs in a fresh non-root Linux container with no network access or
  application mounts. A 60-second child deadline (70-second supervisor deadline),
  30 CPU seconds per process, 32 MB output, 1 GB memory/no swap, 32 processes and 128 MB
  temporary filesystem bound each parse. There is no web-process fallback.
- One operator-configured HTTPS analysis provider and approved model list. The
  pilot HTTP client refuses redirects and environment proxies. Each run is
  limited to 200 actual provider attempts, two million submitted prompt
  characters, and 8,192 requested output tokens per attempt. Character limits
  are not a guarantee of dollar cost.
- Cases in the current Streamlit session. Queueing, durable case blobs, shared
  remote cache, external telemetry, tracing, remote fetching, research tools,
  local folder access, and external OCR runners are blocked in this profile.
- Record text and model reports render as literal text, preventing embedded
  Markdown images or HTML assets from contacting outside services.
  Undated timeline entries stay undated; optional model date inference is disabled.
- File downloads are disabled: Streamlit media URLs do not provide the owner
  authorization this release requires. Participants can copy reviewed text from
  their authorized session to an approved destination. Re-enabling file exports
  requires an authenticated, owner-authorized download service.

The private parser launcher accepts only a file label, size, hash, random request
identifier, bounded extraction page limit and bounded bytes. It fixes the image, command, user, network, mounts
and resource limits. Only the launcher has Docker daemon access; neither the web
application nor the parser receives the daemon socket. The launcher must be
treated as a trusted host administrator because Docker access is powerful. It
receives no provider/OIDC secrets, approvals, logs or case storage. It serves one
parse at a time over a private Unix socket and keeps no case files or content
logs. Health checks remain responsive during parsing. Another parse receives a
prompt busy refusal before file bytes are accepted, rather than queuing past its
deadline. Parser temporary files disappear with the per-file container, which
also has daemon-managed automatic removal if its launcher disappears.

The parser installs only the hash-pinned `pypdf` and `python-dotenv` versions from
[requirements-parser.lock](requirements-parser.lock), copied from the application
lock. The launcher relays bounded output without building a document object
graph. Both sides limit JSON nesting/structure before decoding. The web client
and worker apply the lower of the request and launcher page caps before PDF text
extraction (at most 500 in pilot; at most 5,000 for synthetic isolated use). The
pilot Compose launcher fixes `VA_LSE_PARSER_MAX_PAGES=500`. A separate synthetic
launcher can set this to 5,000 to match its application's configured limit.
ZIP member names are citation labels only, never host paths. A
refused or empty file produces a per-file warning and other files continue;
there is no fallback or implicit acceptance of the refused file.

The mandatory [AppArmor policy](deploy/parser.apparmor) denies all sockets,
capabilities, mounts and process tracing in addition to Docker's default seccomp
policy. Host filesystem/process/network namespaces exclude application secrets
and other cases. Unsupported hosts, missing images/profile, parser failures,
timeouts and malformed or mismatched replies fail closed. This is a container
boundary, not a guarantee against a host-kernel or Docker vulnerability. Linux CI
uses synthetic fixtures; repeat the acceptance probes on the actual reviewed
host before admitting real information. Mac/Windows desktop execution is not a
supported controlled-pilot environment.

## Operator evidence and sign-in setup

1. Choose the private hosting environment, domain, named operator and participants.
   Require MFA at the identity provider. Restrict ingress to approved participants
   or their private network; the app's sign-in does not replace network protection
   against upload or WebSocket resource exhaustion.
2. Review the actual provider agreement, training/data-use terms, logging,
   retention/deletion, subprocessors, region and incident handling for the account
   and models being used. `store=false` is requested for generated responses;
   it does not guarantee that a provider retains no copies. Record the agreed
   retention policy and the participant privacy notice.
3. Obtain independent privacy, security, and accredited legal/evidence review.
   Evaluate synthetic cases with known dates, negation, uncertainty, witness
   attribution, missing scans, conflicting evidence, and false citations. Record
   failures and the acceptance criteria. A matching quote proves quote presence,
   not that a model interpreted it correctly. Every generated factual sentence
   still requires source and witness review.
4. Build and test the exact immutable revision. Set `VA_LSE_BUILD_SHA` to that
   revision and record the same value as `reviewed_revision` in an operator-owned
   copy of [deploy/pilot-approval.example.json](deploy/pilot-approval.example.json).
   Use absolute paths, protect this file from participants, and supply actual
   evidence references for every required field. Approval lasts at most 30 days.
   Removing a subject or expiring approval blocks the next rerun/provider call.
5. Register a confidential OIDC client with the exact redirect URI
   `https://YOUR-PILOT-DOMAIN/oauth2callback`. Configure identity tokens with an
   `iat` and `exp` lifetime of at most one hour. Invitations use the verified
   `(issuer, subject)` identity, not a typed name or email address. Follow the
   [Streamlit authentication configuration](https://docs.streamlit.io/develop/concepts/connections/authentication).

   Mount an operator-owned `.streamlit/secrets.toml` containing:

   ```toml
   [auth]
   redirect_uri = "https://YOUR-PILOT-DOMAIN/oauth2callback"
   cookie_secret = "REPLACE_WITH_A_RANDOM_SECRET_OF_AT_LEAST_32_CHARACTERS"
   client_id = "YOUR_CONFIDENTIAL_OIDC_CLIENT_ID"
   client_secret = "YOUR_OIDC_CLIENT_SECRET"
   server_metadata_url = "https://YOUR-ISSUER/.well-known/openid-configuration"
   ```

   The approval's issuer must match the token issuer exactly. Do not expose
   identity/access tokens through the UI. Streamlit's persistent identity cookie
   does not extend the application token-expiry rule. Audience, signature and
   nonce validation are performed by the configured Streamlit/Authlib integration.
6. Create a protected environment file with the provider key, exact approved base
   URL and main/fast model names. Do not put keys in browser widgets or the
   repository. Set up TLS certificate/key files, the approval file, and OIDC
   secrets before starting [docker-compose.pilot.yml](docker-compose.pilot.yml).

   Required Compose variables:

   | Variable | Operator-supplied value |
   |---|---|
   | `VA_LSE_BUILD_SHA` | Reviewed immutable revision |
   | `VA_LSE_PARSER_IMAGE` | Local parser image ID (`sha256:` plus 64 hex digits), built from that revision |
   | `VA_LSE_DOCKER_GID` | Linux Docker socket group ID; added only to the trusted launcher |
   | `VA_LSE_PILOT_ENV_FILE` | Protected provider environment file |
   | `VA_LSE_PILOT_APPROVAL_HOST_FILE` | Absolute path to actual approval JSON |
   | `VA_LSE_OIDC_SECRETS_FILE` | Absolute path to OIDC secrets |
   | `VA_LSE_BIND_IP` | Private ingress bind address |
   | `VA_LSE_TLS_CERT`, `VA_LSE_TLS_KEY` | TLS certificate and private-key files |

   ```sh
   # On the reviewed Linux Docker host with AppArmor and default seccomp enabled:
   sudo apparmor_parser -r deploy/parser.apparmor
   docker build -f deploy/parser.Dockerfile --build-arg VA_LSE_BUILD_SHA="$VA_LSE_BUILD_SHA" -t va-lse-parser:reviewed .
   export VA_LSE_PARSER_IMAGE="$(docker image inspect va-lse-parser:reviewed --format '{{.Id}}')"
   export VA_LSE_DOCKER_GID="$(stat -c '%g' /var/run/docker.sock)"
   VA_LSE_TEST_PARSER_IMAGE="$VA_LSE_PARSER_IMAGE" VA_LSE_TEST_PARSER_REVISION="$VA_LSE_BUILD_SHA" python -m unittest tests.test_parser_container_live -v
   docker compose -f docker-compose.pilot.yml config --quiet
   docker compose -f docker-compose.pilot.yml up --build -d
   ```

   Never scale this profile. Do not use the general Compose/Kubernetes worker
   examples for real pilot records. Pin deployment image digests after the build
   and scanner checks; restrict outgoing traffic to reviewed destinations and
   deny private/link-local services. Ensure the TLS domain, OIDC registration,
   approval manifest and actual endpoint all agree.

## Deployment acceptance before the first real case

Use synthetic records for this acceptance exercise. Verify an anonymous visitor
cannot reach case controls, an uninvited subject is refused, and an expired token
cannot start or continue provider work. Verify revocation, sign-out, role separation
and identity changes release the previous session's case. Test TLS and WebSockets,
upload caps, parser timeout/resource limits, real provider error paths, logs and
egress restrictions. Verify that no case download URL is created and that
`/health`, `/ready`, and `/metrics` are unavailable through the pilot proxy.

The local test suite exercises the code paths with synthetic identities and
providers. It **does not prove live OIDC integration, TLS configuration, provider
terms, container limits, backup restoration, or output accuracy**. Record those
deployment results in `deployment_validation` and `accuracy_validation`; admission
must remain closed until the responsible reviewers accept them.

## Case lifecycle and incident procedure

“Clear this case” removes session values and registered uploaded files. Sign-out
also clears working state. Python memory release is not cryptographic erasure;
in-flight requests may already have reached the provider. Browser content,
copied text, originals and provider copies follow their own retention policies.
Set and verify a short disconnected-session timeout, prohibit shared browser
profiles, and encrypt the host and operational log volume. Back up only the
count-only pilot operational logs using the reviewed retention policy.

Pilot logs keep fixed lifecycle labels, opaque generated references and numeric
counts. They omit filenames, conditions, witness text, arbitrary extras,
exception messages and stacks. Global diagnostic views require an operator role.
Operators should use a reference and reproduce problems with synthetic data;
do not request real record text through issue trackers or diagnostic messages.

To suspend admission, expire/remove the approval file or remove invitations;
stop the application if an immediate stop is required. Revoke compromised
provider/OIDC secrets, preserve count-only incident evidence, and follow the
named incident-response procedure. Check provider copies separately. Restart
only after the reviewed revision, invitations, evidence and secrets are current.

## Audit remediation map

| Audit issue | Treatment in this release |
|---|---|
| F01: managed keys / editable destinations | Keys omitted from widgets; managed settings immutable to session inputs; approved HTTPS destinations and redirect/proxy refusal |
| F02: result and diagnostic isolation | Owner metadata checked before status/results; legacy unowned results refused; operator-only diagnostics; queue disabled in pilot |
| F03: ZIP cache loses members | Cache stores and replays all documents and skipped members |
| F04: coverage/citation metadata lost | Versioned serialization preserves total pages, unreadable pages and citation units; legacy coverage marked unknown |
| F05: repeated clinical evidence removed | Raw repeated source lines preserved |
| F06: error/content leakage | Count-only pilot file, console, audit and diagnostic sinks; external telemetry/tracing blocked |
| F07: factual rewrite/grounding/citations | Grounding requires every analysis section, meaningful typed rows, JSON boolean coverage flags and all checklist topics A–O; omitted witness analysis or malformed responses stop drafting; older incomplete results remain inspectable with a re-run warning. Automatic self-review cannot change witness text; complete contiguous quotes must match an unambiguous cited page, including their endings and punctuation; missing/short/unresolved citations block pilot generation; older prefix checks require a re-run; human verification of descriptions, dates and interpretations still required |
| F08: parser exhaustion/fallback | Dedicated container, no application mounts, no sockets, AppArmor/default seccomp, bounded resources and replies, no pilot fallback |
| F09: misleading retention/deletion | Session clearing includes registered uploads; accurate consent; queues/blobs excluded; provider retention requires review |
| F10: deployment wiring | Runtime is default non-root image; backup scripts included; health checks use installed Python; service DNS and monitoring targets corrected |
| F11: hosted access/action controls | Verified invited OIDC identities, operator role, TLS/private pilot proxy, quotas, no media exports; monitoring ports local and anonymous dashboards disabled |
| F12: dates | Shared calendar parser handles full and named-month dates; invalid full dates do not become partial dates; precise gaps require full dates |
| F13: queue reliability | Atomic, input-bound submission/recovery indexing, bounded admission and AOF/fsync settings; synthetic crash/pressure/retry tests; actual hosted persistence and ownership still need worker-release acceptance; queue disabled in pilot |
| F14: timeline PDF markup | Dynamic paragraph content escaped; PDF failures handled; file exports disabled in pilot |
| F15: legal/currency claims | Corrected general doubt/continuity/functional-loss language; all knowledge files fingerprinted; topic status cannot certify full legal currency |

Deferred features are release-gated, not declared fixed or approved for real data.
This restricted pilot must not be represented as a public launch, claims decision
system, compliance certification, or replacement for accredited human assistance.
