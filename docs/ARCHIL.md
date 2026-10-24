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

This mounts a pinned release at `/mnt/kb` read-only and the outputs disk at
`/mnt/results` read-write. Verify the current plan's release exists first.
For the UI, launch from `/opt/valay` with:

```
VA_LSE_MODE=synthetic VA_LSE_KNOWLEDGE_DIR=/mnt/kb/app/knowledge \
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
