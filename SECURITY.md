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
This is enforced at runtime: with the flag set and the server not bound to a
loopback address (including the default all-interfaces bind), local imports stay
disabled and the app tells the operator how to restart correctly. The bind check
derives the address from the process's own launch argv (`--server.address`)
backed by the committed `.streamlit/config.toml` — app code deliberately does
not read Streamlit's runtime configuration (see `app/local_paths.py`).

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

Install it as a link, or point git at the hooks directory — not as a copy (wiring
it via [`pre-commit`](https://pre-commit.com/) is also supported, see below):

```bash
ln -sfn ../../scripts/hooks/pre-commit .git/hooks/pre-commit   # a link to the file
git config core.hooksPath scripts/hooks                        # or point git at the directory
```

A copy goes stale the next time the hook changes, and in a repository with
worktrees `.git/hooks` lives in the main checkout — so the hook every worktree runs
can belong to another branch entirely. The hook therefore compares the bytes being
run with the checkout's own `scripts/hooks/pre-commit` and **refuses the commit**
when they differ, naming both files and the `core.hooksPath` install that cannot
drift. Only a hook from before that check can still run stale.

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

The same treatment covers a staged **app** module that reads a Streamlit config
option (`tests/streamlit_option_reads.py`): an option's value comes from the
machine and the directory the process started in, so a decision made from one — the
hardening check answering "is XSRF on *here*?" rather than "does this deployment
ship it?" — is ambient. The suite scans for it too; the hook catches it at the
commit, when the rule is still cheap to satisfy.

To install as a `pre-commit` framework hook instead, add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: no-secrets
        name: Block .env and sk-* secrets
        entry: bash scripts/hooks/pre-commit
        # Also enforces the test-harness import (tests/harness_imports.py) and
        # refuses Streamlit config reads in app/ (tests/streamlit_option_reads.py).
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

## 9. Local browser automation (CDP attach) and derived PHI at rest

The VA.gov downloader (`scripts/va_records_download.py`) and the Streamlit app touch
complementary but different data at rest. The boundary matters:

**CDP attach — how the local automation boundary works.** When you use
`python scripts/va_records_download.py --cdp http://127.0.0.1:9222`, Playwright talks to a
Chrome you launched yourself with `--remote-debugging-port=9222 --user-data-dir=...`. The
protocol is plain HTTP + WebSocket on that port. Anyone or anything that can reach the port
can run arbitrary JavaScript in the browser (including reading the page). Practical guidance:

- Attach only to `127.0.0.1` — the loopback bind is the entire security boundary. Never
  `--remote-debugging-port` on `0.0.0.0` or a non-loopback interface, and never expose the
  port through SSH tunnels to shared machines, reverse proxies, or port-forwarding configs.
- **Use a dedicated `--user-data-dir`** — Chrome requires this (it refuses remote debugging
  on the default profile since Chrome 136), and it is also the right containment: the
  debugging port plus a *daily-driver* profile is the worst case (cookies and sessions for
  every site you use are one JavaScript evaluation away). A dedicated profile keeps the blast
  radius to the VA.gov session you opened it for. The script never closes an attached
  browser — quit Chrome yourself when done; closing it kills the debug port.
- **Start the port only for the download session** and close Chrome after — a lingering
  listener is standing attack surface for anything else running on your machine.
- **A local process does not need the port to read your session.** Malware running as your
  user can read the profile directory directly. CDP raises convenience here, not the local
  threat model. The port matters when you run the downloader on a machine you share with
  other software you do not fully trust.
- **On failure the script writes signed-in-page evidence** — screenshot + page HTML + URL to
  `va_gov_download_artifacts/` (default). That HTML is a live signed-in VA.gov page and can
  contain medical data. Treat the directory as PHI: keep it local, share contents only with
  explicit intent, and delete it after debugging. The Streamlit app itself never uses CDP
  and never reads or writes browser profiles; its VA.gov source is a sandbox client only.

**Profile-cookie TTL — the profile at `~/.va_lse_va_gov_profile`.** This is the one place
VA.gov session cookies persist on disk (the app writes no sessions anywhere). Understand its
lifetime rather than trusting it:

- Cookies are stored unencrypted in `Cookies` (SQLite) inside the profile, so the profile is
  only as strong as the filesystem underneath it — put it on FileVault-encrypted disk.
- VA.gov session and SSO cookies carry their own expiry (session cookies die with the
  browser; persistent ID.me/VA SSO cookies are typically hours to weeks by issuer policy).
- Treat the profile as **hot for up to 30 days** regardless — if the disk or a backup of it
  leaks, assume the session inside is still usable. Manage lifetime explicitly:
  delete the profile to sign out (`rm -rf ~/.va_lse_va_gov_profile`, or
  `rm -rf ~/.va_lse_debug_chrome` for the CDP-attach profile); pass `--no-persist` for a
  one-shot throwaway session (full MFA each time); and re-authenticate deliberately when a
  session older than a few weeks is reused.
- Do not back the profile up, sync it (iCloud/Dropbox), or copy it between machines.

**Derived PHI at rest — what this repo's tooling writes to local disk.** All of it is
unencrypted; keep the whole checkout on encrypted disk and out of synced folders.
`logs/` and `outputs/` are already git-ignored:

| Path | Content | Notes |
|---|---|---|
| `logs/runs.jsonl` (+ `logs/audit.log`) | Per-run audit trail: request ids, phase timings, counts, scrubbed error text | No traceback is ever written. `app/audit.py` keeps the app's own error classes to a curated PHI-free reason and scrubs every remaining free-text field (SSN forms, digit runs, emails, URLs, paths, record-file names, phone numbers, cue-shaped names, control and invisible characters) — including the caller-supplied `condition`, `source` labels and `outcome` values. `VA_LSE_AUDIT_ERROR_MESSAGES=0` drops the free-text field entirely. Scrubbing happens only at write: a future field added without it is not protected by the old scrubber |
| `outputs/batch-draft/state/batch_NN.json.gz` | **Resumable digest state: compressed per-batch extracted facts — derived PHI** | Written atomically (fsync + rename). Batch-fact shards hold the extracted medical evidence itself |
| `outputs/batch-draft/staging/` | Extracted record text staged per batch group | Derived PHI; remove with the run output when done |
| `outputs/batch-draft/…final` artifacts | The drafted statement, merge summaries, and full fact lists | Derived PHI |
| `va_gov_download_artifacts/` | Signed-in-page screenshots/HTML from downloader failures | **Actual** signed-in PHI, not derived; now git-ignored — it was not before |
| `~/.va_lse_va_gov_profile` | VA.gov session cookies | See profile-cookie TTL above |
| `usage_history.json` (repo root; `VA_LSE_WATCHDOG_PATH` overrides) | Token/cost counters, no record content | Aggregates only |

Retention guidance: delete `outputs/batch-draft/` when a run's statement has been exported;
treat `logs/runs.jsonl` as the long-lived record (scrubbed by design) and rotate or delete
`logs/` on your own schedule; never commit, sync, or back up any of these paths.

## 10. The prompt data boundary (injection from records)

The record set, the statement, the observations and the witness fields are **untrusted input
to a model**, and everything the model reads is ultimately read by a human as a legal work
product. `app/prompt_sanitize.py` is the mechanical half of that boundary and `GUARD_NOTE` is
the behavioral half — the module says so out loud, because a sanitizer that claims to *solve*
injection is worse than none.

**What is enforced mechanically.** Untrusted text is placed inside `<<<`/`>>>` blocks; every
run of two or more angle brackets is escaped (runs, not triples, so two fields cannot re-form
a delimiter when a template concatenates them), fenced-code runs are broken, angle-bracket
lookalikes (fullwidth `＜`, CJK `⟨`) are translated to ASCII first, invisible characters
(zero-widths, bidi overrides, Unicode tag characters) are dropped, and chat-template role
tokens and line-leading role labels (`<|im_start|>`, `<system>`, `[INST]`, `System:`) are
neutralized so record text cannot speak in the system role. Escaping is chosen to touch only
characters that the citation check ignores — `verify_citations` compares quotes back against
the page by alphanumeric token, so hardening the boundary cannot make a real citation
unverifiable.

**What is not, and cannot be.** No instruction is filtered: a record may legitimately discuss
instructions, and blocking phrases would corrupt analyses rather than protect them. The
answers are: the guard note lives in the **system** message of every prompt that carries
untrusted or record-derived text (including the merge, summary and date-inference prompts,
which are reached through the digest model's own JSON — the second-order path); every fact
carries a page citation that is checked back against the record; and the output is reviewed
by the person who uploaded the records. The realistic harm model for this app is therefore
**integrity** (a crafted page steering an analysis), not exfiltration: a user's model output
is shown to that user.

## 10. Reporting a vulnerability

Do not open a public issue for security-sensitive findings. See the
`## Security notes` section in `README.md` for the private channel.
