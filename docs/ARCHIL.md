# Archil synthetic workspace

Resources created in the personal account richard.parslow@g.austincc.edu,
AWS `aws-us-east-1`:

| Resource | ID |
| --- | --- |
| valay-eval-kb | dsk-0000000000113f9e |
| valay-eval-runs | dsk-0000000000113fb6 |
| valaystatementevaluator | 01a16b36-645f-77e6-8e44-1321316d6239 |

The private Python 3.13 sandbox has 1 vCPU, 2 GiB, and a one-hour session TTL.
No public port is created by this setup. These resources alone do not authorize
real-veteran processing or prove HIPAA compliance, encryption policy, retention,
or provider agreements. Existing pilot and synthetic-storage controls still apply.
The sandbox root is persistent: `/tmp`, process memory snapshots, extracted text,
and stdout must not be treated as ephemeral secure deletion.

## Finish with an operator credential

Install `requirements-archil.txt` in an operator environment. Set `ARCHIL_API_KEY`
through a secret manager (never commit it or put it in a command/URL). The key
is account-wide and must not be placed in the evaluation sandbox.

```
python scripts/archil_setup.py plan
python scripts/archil_setup.py upload-kb
python scripts/archil_setup.py smoke
```

Upload is restricted to the four checked-in knowledge markdown files and the
explicitly synthetic benchmark JSON. It does not scan `reference_docs`, statements,
records, local outputs or logs. Releases use a content hash plus manifest and
refuse an inconsistent existing object. This is a public-reference seed, not a
complete or newly validated CFR/M21-1/BVA corpus.

Stop the sandbox in the console before attaching disks:

```
python scripts/archil_setup.py mount
```

This mounts the knowledge disk root at `/mnt/kb` read-only and the outputs disk at
`/mnt/results` read-write. Verify the current plan's release exists first.
For the UI, launch from `/opt/valay` with:

```
VA_LSE_MODE=synthetic VA_LSE_KNOWLEDGE_DIR=/mnt/kb/releases/b51574cdbc2dbbabdcf550e3970b12ea72f260df5746af63c56bb11253a7aff2/app/knowledge \
  python -m streamlit run run_app.py --server.address=0.0.0.0 --server.port=8080
```

Do not disable Streamlit XSRF. Keep ports private until application authentication
and access controls have been reviewed. Starting Streamlit alone does not provide
an external URL or supervised restart. Model credentials must be configured
separately before any real evaluation can succeed; never send PHI for smoke tests.
Keep prompt/response bodies out of logs and keep existing storage restrictions.

## CI and execution limits

The manual `Archil reference smoke (manual)` workflow uses the
`archil-synthetic` environment's `ARCHIL_API_KEY` secret. It only checks the
reference release and read-only enforcement. It makes no model calls and does
not claim accuracy or benchmark regression results. Run existing accuracy
prepare/assess tooling with independent evidence for scoring comparisons.

Serverless `exec` is limited to five minutes, while this app permits 30-minute
pipelines. Full evaluations therefore need a persistent sandbox/worker or
separately terminable job process, not an unmodified serverless exec wrapper.

Per the current Archil documentation, disk branch deletion is unsupported and
credentials cover all branches. Automatic per-PR disk branches are deliberately
not enabled; they cannot deliver automatic cleanup or tenant isolation. Use
read-only reference releases and separate disposable job storage until supported
cleanup and reviewed access controls are available. No merge-back API is assumed.

References checked 2026-10-23:
- https://docs.archil.com/sdks/python
- https://docs.archil.com/compute/serverless-execution
- https://docs.archil.com/compute/sandboxes/introduction
- https://docs.archil.com/concepts/branches-and-checkpoints

## Verified setup state (2026-10-23 Central)

The named API key was created by the account owner and used outside the evaluation
sandbox. The public reference release was uploaded and verified; serverless
`readonly-smoke-ok` confirms the knowledge mount rejects writes. A separate
serverless results-disk run wrote `synthetic/mount-smoke.txt` successfully.

**Persistent attachment is blocked by Archil startup errors.** Both starting the
existing sandbox with attached disks and creating a new mounted sandbox failed
while attaching `valay-eval-runs`. A root-only start of the original sandbox
succeeds. Do not assume `/mnt/kb` or `/mnt/results` is available in that sandbox.
The failed diagnostic sandbox is `01a16b44-addb-7953-a4ee-81cd3ead9d97`
(`valay-evaluator-mounted`); retain it for vendor diagnosis. It contains no
veteran records. No failed resources were deleted.

The original sandbox was updated to integration commit
`8ac8178833288ae85a586778f57befdb4189543a`; the three targeted tests pass there.
A private native Archil service uses this command (the working directory is
required so Streamlit reads the committed hardening configuration):

```
archil services create valay-ui --dir /opt/valay --tcp-port 8080 \
  --env VA_LSE_MODE=synthetic -- python -m streamlit run run_app.py \
  --server.address=0.0.0.0 --server.port=8080
```

This service uses bundled rules until persistent mount attachment is repaired.
No model key or GitHub Actions secret was configured. The manual smoke workflow
still needs `ARCHIL_API_KEY` in its environment secret before it can run in CI.

## Root mount correction

A console retry returned `Invalid disk ID format` with the release subdirectory
appended to the knowledge disk ID. Persistent mounts now omit `subdirectory`.
The disk root is read-only; the app selects the content-addressed release using
`VA_LSE_KNOWLEDGE_DIR`, printed by the mount command. Use that printed value if
reference contents change. The serverless subdirectory smoke passed independently
and retains its existing configuration. Live persistent attachment of this
correction remains unverified pending an authenticated operator retry.
