# Persistent disk attachment diagnostic

Observed October 23, 2026 Central (October 24 UTC). No veteran records or credentials are included in this report.

## Confirmed results

- Control-plane authentication succeeds with the current API key in `aws-us-east-1`; listing disks returns three disks.
- The knowledge and results disks previously passed independent serverless read/write checks; the knowledge read-only smoke rejected writes.
- Both disks currently report zero connected clients.
- A persistent sandbox starts successfully with `mounts=[]`.
- Attaching only the knowledge disk at its root, read-only, fails before the sandbox starts.
- Attaching only the results disk at its root, read-write, also fails before startup.
- Pre-creating `/mnt/kb` and `/mnt/results` on the sandbox root filesystem, stopping, and retrying the knowledge-only attachment does not resolve the failure.
- The previously observed release-suffix disk-ID error is avoided by omitting `subdirectory`, but attachment still fails. PR #101 supplies that root-mount correction.

## Resources

Region: `aws-us-east-1`.

| Resource | ID |
| --- | --- |
| Knowledge disk | dsk-0000000000113f9e |
| Results disk | dsk-0000000000113fb6 |
| Diagnostic sandbox | 01a16b44-addb-7953-a4ee-81cd3ead9d97 |
| Original app sandbox | 01a16b36-645f-77e6-8e44-1321316d6239 |

Diagnostic sandbox: `python:3.13-slim`, 1 vCPU, 2 GiB, one-hour session TTL. The original app sandbox was recovered without attached disks and paused. The diagnostic sandbox remains failed after the last attachment attempt and has no running application or veteran records.

Guest client reports `v0.10.0-1792692895`, build `2026-10-22 18:14:55 UTC (f6b72cb9)`, protocol Version2. Python SDK is 0.18.0.

## Minimal reproduction

Use an existing account API key through the process environment; do not paste it into source or logs.

```python
import archil

with archil.Archil(region="aws-us-east-1", timeout=180) as client:
    box = client.sandboxes.get("01a16b44-addb-7953-a4ee-81cd3ead9d97")
    box.start(mounts=[archil.SandboxMountSpec(
        disk="dsk-0000000000113f9e", path="/mnt/kb", read_only=True,
    )])
```

The API accepts startup, then the SDK raises `SandboxStartError`. At `2026-10-24 01:48:19.099065Z`, the reported failure is:

```
mount /#2267685e/#74fec0c3 (dsk-0000000000113f9e):
INFO [archil_client::auth::mount line 1667]
[MOUNT-TIMING] refresh_endpoint_no_retry total: 90ms
(connect=46ms, sts_sign=0ms, write=0ms, read=44ms)
```

The results-only attempt at `2026-10-24 01:47:52.424695Z` reported the same timing-only failure for `dsk-0000000000113fb6`.

## Unresolved question

The exit reason exposes an INFO timing line rather than the underlying mount error. Vendor-side startup/mount logs are needed to identify the failure. A platform-managed mount authentication, client startup, or other mount-path fault remains possible; these tests do not distinguish them. No forced delegation recovery, disk-user removal, firewall weakening, or resource deletion was attempted.
