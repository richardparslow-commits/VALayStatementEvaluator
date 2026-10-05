# Dedicated parser engine boundary (IA-03)

Status: **actual-host acceptance OPEN**. The production launcher no longer mounts
the application Docker socket or joins its daemon group. It uses private mutual
TLS to a **different engine on a dedicated Linux host or VM**. No real information
may enter until the named security owner accepts this boundary for the exact
revision/images and the existing `ingestion_security` approval reference records
that review. CI cannot supply that independent approval.

## Required deployment

1. Provision a dedicated Linux parser host/VM with AppArmor, built-in seccomp and
   effective CPU/memory/PID limits. Do not run the application, providers, OIDC,
   case/log/control storage, general workloads or their credentials there. No
   shared application filesystem, host socket, hypervisor management credential,
   container registry write key or general-purpose Docker endpoint belongs there.
   Restrict the VM's administrative identity and disk/snapshot/log retention.
2. Configure a dedicated Docker daemon with `tlsverify=true`, server TLS files,
   and the daemon label `va-lse-purpose=parser-only`. Bind its TLS listener to a
   reviewed literal RFC1918 IPv4 address on port 2376. Do not expose plaintext
   port 2375. Keep any VM-local administrative socket private to that VM.
3. Use a CA reserved for this parser engine. Issue a server certificate containing
   that private IP in its SAN and the serverAuth usage, and a client certificate
   with clientAuth usage. Keep the CA private key off the launcher. Certificate
   expiry/rotation/revocation is an operator control; TLS does not provide a
   least-privilege Docker authorization policy by itself.
4. Install `deploy/parser.apparmor` on the parser VM. Build/import only the reviewed
   parser image from `deploy/parser.Dockerfile`. Pin its `sha256:` image ID and
   `VA_LSE_BUILD_SHA`. Read the parser daemon's `docker info --format '{{.ID}}'`
   through the verified TLS client, and separately read the application daemon's
   ID. They must differ. Never point the endpoint at the application host, even
   with another port or relabeled daemon.
5. On the application host, create a private directory containing only `ca.pem`,
   `cert.pem`, and `key.pem`. The launcher runs as UID 65534; its key must be owned
   by that UID, owner-readable and inaccessible to group/others (for example
   0400). The mounted directory must be owned by that UID with no group/other
   permissions (for example 0700). All files must be bounded regular files without symlinks or group/other
   write permission. Mount that directory read-only only into the launcher at
   `/run/parser-engine/tls`. The web/parser containers receive no TLS credentials.
6. Set `VA_LSE_PARSER_ENGINE_ENDPOINT=tcp://PRIVATE_IP:2376`,
   `VA_LSE_PARSER_ENGINE_ID`, `VA_LSE_APPLICATION_ENGINE_ID`, and
   `VA_LSE_PARSER_ENGINE_TLS_DIRECTORY` for Compose. The web receives only the
   non-secret parser engine ID. Remove the obsolete `VA_LSE_DOCKER_GID` setting.
7. Enforce the launcher network's egress allowlist outside Compose: only the
   reviewed parser address:2376, without access to the application daemon,
   provider/identity/storage endpoints, private management services or the
   Internet. On the VM allow TLS ingress only from the reviewed launcher path.
   Deny general egress from the parser VM; review any time-limited build/import
   exception separately. Containers continue to run with `--network=none`.

`ParserEngine` refuses local sockets, SSH/contexts, DNS hosts, public/loopback/
link-local addresses, alternative ports, identical/missing IDs and unsafe key
files. Docker verifies TLS and client authentication. Readiness and every launch
recheck engine ID/purpose and the existing Linux security/image/revision checks.
Health and output envelopes bind the expected engine identity; legacy envelopes
are refused. Changing the endpoint does not authorize another engine identity.

## Actual-host synthetic checks

Record revision/tree, application/parser/launcher/proxy image IDs, both daemon
IDs, private endpoint, certificate fingerprints/expiry, firewall revision,
reviewer and UTC window. Keep private keys and case content out of evidence.

| Check | Required result |
| --- | --- |
| Host inventory and storage/mount review | Parser host/VM is actually distinct; no application/other-case secrets, volumes, workloads or management access. Different IDs/labels alone do not prove separation. |
| Effective TLS path | Valid client succeeds; missing client certificate, wrong CA/server, wrong IP SAN, expired cert or disabled server authentication refuses before provider work. |
| Wrong/shared daemon ID, missing purpose, old launcher envelope | Admission closed; no upload falls back to local parsing. |
| Credential exposure | Launcher key is read-only/private; web and parser cannot read it; no app daemon socket or group is available to launcher. |
| Destination policy | Launcher can reach only its reviewed TLS engine. No application Docker API, management/provider services or Internet access. |
| Existing Linux parser probes | All current methods in `tests.test_parser_container_live` execute without skips, using synthetic test configuration pointed at the reviewed engine. Preserve filesystem, AppArmor/seccomp, no-network, memory/CPU/PID/wall limits and cleanup. |
| Stop/restart/rotation/revocation | Unreachable engine, invalidated client credentials and wrong engine ID close admission with fixed errors. Recover only with matching reviewed identity and refreshed approval as required. |

For host probes, use the `VA_LSE_TEST_PARSER_ENGINE_*` variables matching the
production endpoint, ID and TLS directory, plus `VA_LSE_TEST_APPLICATION_ENGINE_ID`,
`VA_LSE_TEST_PARSER_IMAGE` and `VA_LSE_TEST_PARSER_REVISION`. The launcher end-to-end
probe also needs `VA_LSE_TEST_PARSER_LAUNCHER_TLS_DIRECTORY` and the locally built
synthetic `va-lse-parser-launcher:ci` image. It deliberately tests the application
daemon only to start a synthetic launcher with TLS credentials; none are mounted
into the parser. Do not use the CI setup script to provision a production engine.

## Residual authority and verified limits

The launcher client still has administrative authority on the dedicated parser
VM. A launcher/daemon compromise can damage that VM and transient records being
processed there. Mutual TLS authenticates the engine/client, rather than limiting
Docker commands. This change reduces cross-application blast radius through the
required separate host, private routing and absence of other assets. It does not
demonstrate a parser escape, supply an antivirus/CDR service, attest host inventory,
or eliminate hypervisor/kernel risk. The existing passive-file/malware decision,
privacy and actual-release acceptance gates continue to apply.

CI's `scripts/parser_engine_ci.py` starts a second daemon on a disposable runner with its own data/exec roots,
engine ID and one-day synthetic TLS credentials, then imports only the parser
image. It runs the actual launcher over TLS without an application socket and
tests certificate refusals and the original parser protections. **Both CI
daemons share the runner kernel/filesystem**; this is transport/daemon/protection
evidence and cannot prove production host/VM separation.

## Primary documentation cross-check

- [Docker socket protection](https://docs.docker.com/engine/security/protect-access/):
  mutual TLS requires server/client certificates; possession of client keys
  conveys daemon/host authority. The separate host requirement remains necessary.
- [Docker rootless limitations](https://docs.docker.com/engine/security/rootless/troubleshoot/):
  AppArmor is unsupported in rootless mode. This change preserves AppArmor and
  uses a dedicated VM instead of substituting rootless with weaker protections.
- [Docker AppArmor](https://docs.docker.com/engine/security/apparmor/): verify the
  effective profile inside the actual parser container, not just a config flag.
