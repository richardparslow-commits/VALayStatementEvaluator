# Security — secrets, keys, and safe deployment

This document describes how to handle secrets (API keys, tokens, and other
credentials) for this project and how to prevent leaking them.

> Commit code and configuration. Never commit secrets.

## 1. What counts as a secret

All of the following are secrets and must not be committed to git:

- `OPENAI_API_KEY` / `LLM_MODEL_*` keys (`sk-sp-...`) and any QwenCloud /
  OpenAI-compatible provider key.
- `FETCH_SANDBOX_API_KEY`, `AGILOOP_INSPECT_API_KEY`, `VA_GOV_API_BASE_URL`
  tokens, and any bearer tokens or session tokens that appear in logs or
  `st.session_state`.
- Any credential copied from a cloud console or vault.

The app never writes secrets to disk beyond the local `.env` you create; see
**Storage** below. Treat anything that looks like `sk-*` as a secret.

## 2. Storage — local development

| Rule | Detail |
|---|---|
| **Source of truth** | `cp .env.example .env` then edit `.env` locally (never committed). |
| **Local overrides** | Use **`.env.local`** (or **`.env.<name>.local`**, e.g. `.env.dev.local`) for machine-specific testing. Both patterns are in `.gitignore`. |
| **Never create** | `.env` with real keys inside a cloned repo that you then `git add`. If an IDE scaffolds `.env.local`, it is still git-ignored — but double-check before `git add -A`. |
| **Precedence** | `app/config.py` loads `PROJECT_ROOT/.env` with `override=True`. That means `.env` wins over shell env vars. Set env vars in your shell/CI only when you *intend* them to carry the secret (e.g. in GitHub Actions). For local dev, prefer `.env`. |

Example:

```bash
cp .env.example .env          # one-time per clone
# edit .env, set OPENAI_API_KEY, leave AGILOOP_* unset if you don't use telemetry
streamlit run run_app.py
```

If you need a one-off override without editing `.env`:

```bash
cp .env .env.local
echo 'LLM_MODEL_MAIN=qwen3.7-max-override' >> .env.local
# .env.local is git-ignored; remove it when done
```

Never do:

```bash
git add .env
echo "OPENAI_API_KEY=sk-sp-..." >> README.md   # or any tracked file
```

## 3. Rotation

Rotate API keys periodically — at minimum:

- On contributor offboarding or credential compromise.
- Every 90 days for long-lived provider keys.
- Immediately if a secret was ever committed (even if reverted).

Steps (QwenCloud Token Plan):

1. Create a new key in the provider console (new `sk-sp-...`).
2. Replace `OPENAI_API_KEY` in your local `.env` (and any vault / GH Secret below) — keep the old key active until all consumers are updated.
3. Verify: `streamlit run run_app.py` → run `examples/` sample; and the offline suite `python -m unittest discover -s tests -v` (no key needed) still passes.
4. Revoke the old key in the provider console.
5. If the old key was ever committed, follow **Incident recovery** below — revoking alone is not enough (history still holds it until rewritten or purged).

## 4. CI / CD — GitHub Actions (GitHub Secrets)

The pipeline never commits secrets. The live `smoke` job in `.github/workflows/test.yml`
is `workflow_dispatch`-only and gated on secrets:

- Required: `OPENAI_API_KEY`
- Optional overrides: `OPENAI_BASE_URL`, `LLM_MODEL_MAIN`, `LLM_MODEL_FAST`

Configure at **Settings → Secrets and variables → Actions → New repository secret**
(not as a plaintext `env:` literal in the workflow file). The workflow references them
as `${{ secrets.OPENAI_API_KEY }}` etc. — never put a real key in `.github/workflows/*.yml`.

Offline tests and `scripts/scale_sim.py` run on every `push`/`pull_request` to `main`
without any secrets.

## 5. Production deployments (cloud apps)

Local filesystem imports are disabled by default. Keep `VA_LSE_ALLOW_LOCAL_PATHS`
unset or `0` on hosted/shared deployments and use file uploads. Request headers
(including Host and forwarded headers) and browser URLs never grant filesystem
access. Setting `VA_LSE_ALLOW_LOCAL_PATHS=1` grants every user of that app access
to supported files readable by the server process; it is not per-user authorization.
Enable it only for trusted single-user use with Streamlit bound to `127.0.0.1`,
without a public reverse proxy or tunnel. See README for the local launch command.

For any hosted / multi-user deployment (Agiloop, Streamlit Cloud, Docker, etc.):

- **Use a managed secret store**, never a checked-in `.env`. Examples:
  - AWS Secrets Manager / Parameter Store
  - Azure Key Vault (Container Apps / App Service → Key Vault reference)
  - GCP Secret Manager
  - Streamlit Community Cloud **Settings → Secrets** (writes `.streamlit/secrets.toml` inside the
    deployment; the only channel available there, since `.env` does not ship). Values are read in
    the order environment → `.env` → secrets → code default, so the platform store is a fallback,
    never an override, and the sidebar captions which fields came from it (`DEPLOYMENT.md` → Pattern D)
  - Agiloop / platform-managed environment secrets (preferred when available)
- Inject secrets as **runtime environment variables** or **mounted secret files** — not baked into the image.
- Scope roles narrowly (least privilege) and enable rotation/autorotation.
- Ensure app logs do not emit secret values — this app logs phase/timing counts and never prompt bodies, statement text, observations, or API keys. Treat any logged provider key (the `sk-` prefix) as an incident.
- **One deliberate exception to "no record content":** an upload *filename* appears in the log when that file fails to extract, because "which file failed" is usually the whole question — and a filename is a string the user themself chose. A file named after a person is therefore a name in the log. Everything the app shows from a log line passes through `app/diagnostics.py:redact_secrets()` first (provider keys, `Bearer`/`Basic` credentials, and `key=value` secrets), so what reaches a screen is scrubbed even when the line on disk is not.
- **Log content is never served over HTTP.** `app/diagnostics.py` resolves a `req_…` reference to its lines inside the app's own session (About → **Look up a reference**). It is deliberately not a route on the health sidecar: that port binds `0.0.0.0` without authentication, and a lookup endpoint there would publish log content to anything able to reach its port.

`.agiloop/deploy.json` (if present) should not contain literal keys; set secrets
in the Agiloop environment and reference them.

## 6. Preventing accidental commits

### Pre-commit hook (recommended)

Copy `scripts/hooks/pre-commit` to `.git/hooks/pre-commit` (or wire it via
[`pre-commit`](https://pre-commit.com/)):

```bash
cp scripts/hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

That copy goes stale the next time the hook changes, so it is worth installing as a
link or a configured hooks path instead — either one keeps the checks below at the
version in this repository:

```bash
ln -sfn ../../scripts/hooks/pre-commit .git/hooks/pre-commit   # a link to the file
git config core.hooksPath scripts/hooks                        # or point git at the directory
```

The hook rejects any staged path matching:

- `.env` (any `.env` file, including `.env`, `.env.example` is allowed)
- `.env.local`, `.env.*.local`
- `.streamlit/secrets.toml`
- `*.pem`, `*.key`

and scans staged diffs for lines that look like `KEY=sk-...` or `OPENAI_API_KEY=`.

It also refuses staged test modules that are not wired into the hermetic test
harness (`tests/harness_imports.py`) — either importing no harness at all or
importing it after an `app` import, which would let that module read whatever
configuration this machine happens to have. It is the same rule the suite scans
with, read from the staged copy, and it has already caught a real mistake: a new
`tests/test_job_queue_*.py` reached `main` unwired and failed only in CI, on the
merge result. It needs `python3` on `PATH` (the project's `.venv` is preferred);
when no interpreter exists the commit is **blocked**, not waved through.

To install as a `pre-commit` framework hook instead, add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: no-secrets
        name: Block .env and sk-* secrets
        entry: bash scripts/hooks/pre-commit
        # Also enforces the test-harness import (tests/harness_imports.py).
        language: system
        pass_filenames: false
```

### Scanning the repo

- `git log -p --all -S 'sk-sp-'` — search history for the Token Plan prefix.
- `git diff --cached` — always review staged changes before `git push`.
- GitHub **secret scanning** (enable at Settings → Code security → Secret scanning)
  will alert on pushed `sk-*` / generic API keys and can be configured to block pushes.

## 7. Incident recovery — a secret was committed

If any secret appears in a commit (even on a private branch) assume it is compromised:

1. **Rotate immediately** — create a new key and revoke the old one at the provider.
2. **Purge from git history** (optional but recommended; history still leaks otherwise):
   ```bash
   # Rewrite history to remove the file (requires force-push; coordinate with team)
   git filter-repo --path .env --invert-paths   # or: BFG Repo-Cleaner
   git push --force --all && git push --force --tags
   ```
   Alternatively, use `git filter-branch` (deprecated) or open a new repo from a clean snapshot.
3. **Notify** the rotation and scope of exposure to collaborators.

## 8. Auditing `.gitignore`

The following entries are covered and tested by `tests/test_security_gitignore.py`:

```
.env
.env.local
.env.*.local   # via *.env.local — covers .env.dev.local, .env.prod.local, etc.
.streamlit/secrets.toml
```

If you add a new secret-bearing file pattern, add it to `.gitignore` and to
the test helper in `tests/test_security_gitignore.py`.

### The same audit for `.dockerignore`

`.gitignore` keeps secrets out of git; [`.dockerignore`](.dockerignore) keeps
them out of **images**, which matters more in one respect: the build context is
uploaded to a builder, and the `sandbox` target's image can be pushed to a
registry and snapshotted (see [DEPLOYMENT.md §6](DEPLOYMENT.md#6-dockerfile)).
Neither file implies the other — docker does not read git state — so a file can
be perfectly safe to commit and still have no business inside an image.

`tests/test_dockerignore.py` asserts both directions:

* **nothing sensitive can reach a stage**: `.env` and its variants, key material
  (`.pem`, `.key`), `.streamlit/secrets.toml`, the audit trail (`logs/`), the blob
  store (`blobs/`), exported reports (`outputs/`) and `usage_history.json` —
  checked against the Dockerfile's own `COPY` lines *and* against a simulated
  `COPY . .`, so the filter rather than today's instructions is what protects the
  image;
* **the filter is not so broad that a stage loses what it needs**, which is why
  `.env.example` survives the `.env.*` pattern through an explicit re-include —
  it is a template, not a credential, and the sandbox image copies it on purpose.

One thing to know before editing it: `.dockerignore` patterns are anchored at the
context root, unlike `.gitignore` patterns, so `*.pem` matches only a top-level
file while `**/*.pem` matches any depth. The first version of this file got that
wrong in four places and the test caught every one.

## 9. Reporting a vulnerability

Do not open a public issue for security-sensitive findings. See the
`## Security notes` section in `README.md` for the private channel.
