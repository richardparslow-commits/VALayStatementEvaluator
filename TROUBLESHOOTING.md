# Troubleshooting: LLM Circuit Breaker Open Error

## Error Message

The message names the failure that *caused* the run to fail, which is not always
the failure the breaker reported. A rejected request looks like this:

```
Drafting failed: records:review: LLMError: Record review failed: LLMUpstreamError: LLM
provider rejected the request — check model, endpoint, and payload settings.
(AuthenticationError: Invalid API key; status=401). The endpoint rejected the request
(HTTP 401), which retrying cannot fix: check the base URL, API key and model names in
the sidebar (use Test connection), then re-run. Chunks affected: 328 of 328 (chunk 1,
chunk 2, chunk 3, ...). (reference: req_xxxxx)
```

The message is ordered so the part that decides what you do comes first. The cause
leads, its fix follows, and the affected chunks are summarised last — an earlier
version listed every index up front, which on a large bundle pushed the real error
past the end of the visible box.

When the provider was failing for a reason of its own, the cause is quoted instead,
and the advice asks you to wait rather than to change anything:

```
...LLMUpstreamError: Transient LLM provider error — retry may succeed.
(status=503). The endpoint returned HTTP 503. Re-run; if it persists, split the
record set into smaller files. Chunks affected: 12 of 328 (chunk 4, chunk 19, ...).
(reference: req_xxxxx)
```

The failure itself carries a **What happened?** expander holding the lines for that
reference — the run's own log entries and traceback — so start there rather than
searching the logs by hand.

## What Happened

The **circuit breaker** (a protective mechanism) detected that the LLM endpoint was
failing and **opened** to stop sending requests to a degraded service. This is
**working as designed** — it protects both your application and the LLM provider from
aggressive retries during an outage.

### Sequence of Events

1. Your record review started processing N chunks of medical records
2. All chunk digest calls to the LLM failed (after exhausting retries)
3. After **3 consecutive logical failures** (default: `VA_LSE_CB_FAILURE_THRESHOLD=3`),
   the circuit breaker opened
4. Once open, all subsequent LLM calls fail **fast** (<50ms, no network calls) with
   a clear error message
5. The breaker will automatically try again after **60 seconds** (default:
   `VA_LSE_CB_RECOVERY_SECONDS`)

## Immediate Actions

### 1. Act on the Cause the Message Names

Read the quoted error before doing anything else. The breaker opening is usually a
*consequence*: it counts consecutive failures and then refuses calls fast, so when a
request is rejected every chunk after the first three reports the breaker instead of
the rejection. Two kinds of cause behave differently:

| Cause in the message | What it means | What to do |
|---|---|---|
| `status=401` / `403`, "rejected the request" | The endpoint refused the request — key, model id, or endpoint. No number of retries changes this, so the client now spends exactly one attempt on it, never fails over to a backup endpoint (which shares the credential's fate), and the batch runner **stops the run** (exit code 2) instead of bisecting and quarantining healthy files | Fix the settings or the credential (regenerate the key, check the credit balance), then re-run the same command: completed batches resume from state. Waiting will not help |
| `status=400` naming a model, against Perplexity's Router API | The model is not in the account's catalog. Router API is in **private preview** and the published catalog *is* the allowlist, so a correct-looking `perplexity/…` id can still be refused | Request Router access (api@perplexity.ai), or point the sidebar fields at a provider the key already serves. See `COMPATIBILITY.md` |
| Want Sonar models (`sonar-pro`, `sonar`) | Route-split as of 2026-09-24 (live-probed): the Agent API (`/v1/responses`, this app's route) still rejects them (400 "not supported"), but the classic `https://api.perplexity.ai/chat/completions` route **serves both** (200). `/v1/chat/completions` does not exist (404), so the chat route needs the base URL without `/v1` plus the schema pin | Agent-API switch (when it passes): `.venv/bin/python scripts/sonar_probe.py` exit `0` → set `LLM_MODEL_MAIN=sonar-pro` / `LLM_MODEL_FAST=sonar` in `.env` (one-line switch, no code change). Chat-route switch (available now): set `OPENAI_BASE_URL=https://api.perplexity.ai`, `VA_LSE_LLM_ENDPOINT_SCHEMA=chat`, and the two model lines. Or run `.venv/bin/python scripts/sonar_watch.py --launch`: it re-probes both routes hourly and applies whichever switch passes (backing up to `.env.bak-sonar-*` first, never mid-batch-run) and notifies |
| `status=429` | Rate-limited, and the call's own retries were already spent | Wait a little, lower `VA_LSE_RECORDS_CONCURRENCY`, or prevent recurrence by pacing requests: set `VA_LSE_LLM_MIN_INTERVAL_SECONDS` (e.g. `2` = one call every 2 s) or `VA_LSE_LLM_MAX_RPM` (e.g. `30`) so calls are spaced rather than retried. When the 429 carries `Retry-After` the client already obeys it exactly (capped at `VA_LSE_LLM_RETRY_AFTER_MAX_SECONDS`) instead of the fixed ladder — look for `llm retry-after honored` in the log |
| `status=5xx`, timeouts, connection errors | A provider-side problem | Wait for the recovery timeout and re-run |
| `llm stall watchdog force-closed … pool` in the log | A call sat silent far past its HTTP deadline — a connection that died without closing (with a keep-alive socket, no read timeout ever fires, so without the watchdog the call hangs indefinitely) | Nothing to do: the watchdog force-closed the pool and the call retried. If it fires often, check network stability; `VA_LSE_LLM_STALL_WATCHDOG_MULTIPLIER` (default `2`) scales the budget |
| Nothing quoted — `fail_fast` only | The endpoint never answered during this run, and the breaker was already open when the first chunk ran | Wait for the recovery timeout, then re-run |

Waiting a minute and re-running is correct for the transient causes and useless for
the deterministic ones, so let the message decide rather than trying it first.

### 2. Check LLM Endpoint Health

Verify your LLM endpoint is actually reachable and working:

**Using the app's built-in test:**
1. Open the Streamlit app sidebar
2. Enter your API key and base URL (if not already set)
3. Click **"Test connection"** — it runs the same preflight a run does (the model listing
   plus a real call) and shows the verdict it produces

**Manual check:**
```bash
# Test if the endpoint is reachable
curl -s https://your-endpoint.example.com/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY" | head -c 500
```

### 3. Verify API Key and Base URL Match

A common cause: the API key and base URL come from **different providers/accounts**.

| Provider | Base URL | Key Prefix |
|----------|----------|------------|
| Perplexity Agent API (default) | `https://api.perplexity.ai/v1` | `pplx-` |
| QwenCloud Token Plan | `https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1` | `sk-sp-` |
| OpenAI | `https://api.openai.com/v1` | `sk-proj-` |
| Local Ollama | `http://localhost:11434/v1` | any non-empty |

If your key and base URL don't match, **all calls will fail** and the breaker will
open.

## Configuring a Fallback Endpoint

To automatically fail over to a backup LLM provider during outages, configure a
fallback endpoint. This is the **recommended solution** for production use.

### Step 1: Choose a Backup Provider

You need a **different** LLM provider as backup. Common combinations:

- **Primary:** Perplexity Agent API → **Fallback:** OpenAI or QwenCloud
- **Primary:** QwenCloud → **Fallback:** OpenAI
- **Primary:** OpenAI → **Fallback:** QwenCloud or another OpenAI-compatible endpoint
- **Primary:** Any → **Fallback:** Local Ollama (for offline/dev)

### Step 2: Add Fallback Configuration

Edit your `.env` file (or Streamlit secrets for hosted deployments):

```bash
# .env example - OpenAI as backup for the default Perplexity Router primary
OPENAI_BASE_URL=https://api.perplexity.ai/v1
OPENAI_API_KEY=pplx-your-key
LLM_MODEL_MAIN=perplexity/kimi-k3
LLM_MODEL_FAST=perplexity/glm-5.3-flash

# Fallback to OpenAI
OPENAI_BASE_URL_FALLBACK=https://api.openai.com/v1
OPENAI_API_KEY_FALLBACK=sk-proj-your-openai-key
LLM_MODEL_MAIN_FALLBACK=gpt-4-turbo
LLM_MODEL_FAST_FALLBACK=gpt-4o-mini

# How long to wait before failover engages (default: 300s = 5 min)
# Set to 0 to fail over immediately
LLM_ENDPOINT_FALLBACK_TIMEOUT_SECONDS=300
```

### Step 3: Restart the App

The fallback is read at startup. Restart Streamlit after changing `.env`:

```bash
# Local development
streamlit run run_app.py

# Or if using the runner
python run_app.py
```

### How Failover Works

| Stage | Behavior |
|-------|----------|
| Primary healthy | All calls go to primary |
| Primary fails briefly (< grace period) | Calls still go to primary (don't flip on blips) |
| Primary fails for grace period | Calls route to fallback automatically |
| Primary recovers | Next successful call on primary switches back automatically |

During failover, runs are stamped with both endpoints in the audit record and run log.

**See:** `DEPLOYMENT.md → LLM endpoint failover (optional)`

## Reducing the Impact: Split Large Record Sets

Large record bundles (500+ pages) trigger many parallel LLM calls. If the endpoint
has issues, this amplifies the problem. Splitting records reduces the blast radius.

### Option A: Pre-split Files (Recommended for Recurring Issues)

Use the provided script to split large PDFs before uploading:

```bash
# Split a large PDF into 100-page chunks
python scripts/split_records.py large_records.pdf --max-pages 100 --output-dir ./split_records

# Split a text file
python scripts/split_records.py records.txt --max-lines 5000 --output-dir ./split_records
```

Then upload the split files in smaller batches (2-3 files at a time).

### Option B: Upload in Batches

Instead of uploading all files at once:
1. Upload 2-3 files
2. Run evaluate/draft
3. Repeat with the next batch

### Option C: Adjust Chunk Size

Smaller chunks = more LLM calls but less data per call. If you're hitting timeouts
rather than outages, larger chunks might help:

```bash
# In .env - increase chunk size (default: 8000 chars)
VA_LSE_DIGEST_CHUNK_CHARS=16000
```

**Warning:** Larger chunks may truncate mid-extraction on dense pages. Test with
your actual records.

## Tuning the Circuit Breaker

If the default behavior doesn't match your needs, you can tune it:

```bash
# .env - circuit breaker tuning

# Open after fewer failures (faster protection, but more sensitive to blips)
VA_LSE_CB_FAILURE_THRESHOLD=2

# Open after more failures (less sensitive, but waits longer)
VA_LSE_CB_FAILURE_THRESHOLD=5

# Recovery timeout: how long before trying again (default: 60s)
VA_LSE_CB_RECOVERY_SECONDS=30   # Try again sooner
VA_LSE_CB_RECOVERY_SECONDS=120  # Wait longer on flaky endpoints
```

**Defaults are sane for most deployments.** Only tune if you understand the tradeoffs.

## Checking the Current State

### Via the App UI

The sidebar shows real-time failover status:
1. Open the sidebar
2. Expand **"🔀 LLM endpoint failover"**
3. See: breaker state, unhealthy duration, when failover will engage

### Via Health Endpoint

If the sidecar health server is running (default: port 8001):

```bash
# Liveness + readiness + failover status
curl http://localhost:8001/health | jq .

# Prometheus metrics (includes breaker state)
curl http://localhost:8001/metrics | grep llm
```

If that `curl` works on the host but the same path is refused or `404`s through a
published URL (a Vercel Sandbox, a forwarded dev port), nothing is broken: the
sidecar is bound to loopback. `VA_LSE_HEALTH_HOST=127.0.0.1` — the default in the
`sandbox` image target — deliberately keeps the unauthenticated routes
(`/health`, `/ready`, `/metrics`) off any interface the internet can reach. Run
the command inside the box, or set the variable to `0.0.0.0` *and* accept that
you are publishing them.

Key metrics:
- `va_lse_circuit_breaker_state` — CLOSED/OPEN/HALF_OPEN
- `va_lse_llm_failover_active` — 1 when serving from fallback
- `va_lse_llm_primary_unhealthy_seconds` — how long primary has been down

## The run did not start

A message ending in **Run not started** above the run button is not a failure: it is the
preflight refusing to spend a run on a configuration that cannot work. Nothing was sent to the
model and no credits were used — the check itself is a model listing plus one short chat call,
which is what a run's own first call costs. If that configuration was checked moments ago —
by **Test connection** or an earlier run attempt — its verdict is reused for a few minutes
instead of probing again; when that reused verdict is what refused the run, the notice
says how old the check is. The run log (`logs/runs.jsonl`) records every attempt —
`accepted` when the run starts, `rejected` when the check refuses it — with `endpoint_check`
naming whether the verdict was fresh or reused, so `grep req_… logs/runs.jsonl` answers
"was the endpoint actually probed?" even for a run that failed later.

| Message | What it means | What to do |
|---|---|---|
| The endpoint rejected this API key (HTTP 401/403) | The key and the base URL are not from the same provider account — or, on Perplexity's Router API, the account does not have Router access (private preview) | Fix the key/base URL pair, click **Test connection** to confirm, then re-run. **Run not started** disappears on its own once the settings change |
| The endpoint does not offer `model-x` (or several) | The configured model id is not in the endpoint's published catalog, so every call would be rejected | Correct `LLM_MODEL_MAIN`/`LLM_MODEL_FAST` (or the sidebar fields) against the provider's list; `COMPATIBILITY.md` has the per-provider ids |
| The endpoint refused a real call to `model-x` (HTTP 403) | The endpoint publishes the id and then refuses an actual call — an entitlement or preview gate rather than a missing model. The provider's own message is in the fix line | Read that message: Perplexity's Router API says "currently in limited preview" and is granted per account (`api@perplexity.ai`). Until it is, run on an endpoint that serves the account — their standard `sonar` API, or another provider. Re-running unchanged reproduces it exactly |
| The endpoint refused a real call to `model-x` (HTTP 404) | The base URL serves no completions path, so every call in the run would 404 — a wrong base URL (or a provider whose chat route is not `/chat/completions`) | Check `OPENAI_BASE_URL` against the provider's docs. A `404` on `/models` alone does **not** block — only a real call that fails does |

What does **not** block: an unreachable host, a `404` on `/models` (some OpenAI-compatible
servers serve completions without listing models), a `429` (the key worked; a limit is the
account's business and the run decides for itself), and provider-side `5xx`. Those are reported
and the run proceeds, because an inconclusive check must not refuse to start a working run.

If the check is wrong about your endpoint — a gateway that answers `/models` differently than it
serves completions, or a catalog that lags what is actually served — open **The check can be
wrong — start the run anyway** under the message and tick the waiver. It applies to that exact
endpoint, key and model set; changing any of them asks you again. Every decision is logged at
`phase=endpoint_preflight`, and a refusal is a `rejected` run-log event, so it resolves through
the same **What happened?** panel as any other failure.

## Common Causes and Fixes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| All chunks fail immediately | API key/base URL mismatch | Verify key matches endpoint provider |
| The run will not start at all ("Run not started") | The preflight proved the configuration cannot work | See **The run did not start** above |
| Fails after some successful calls | Rate limiting | Reduce `VA_LSE_RECORDS_CONCURRENCY`, or set `VA_LSE_LLM_MIN_INTERVAL_SECONDS` / `VA_LSE_LLM_MAX_RPM` to space calls and prevent the 429s instead of absorbing them |
| Intermittent failures | Provider outage or network issue | Configure fallback endpoint |
| Fails only on large files | Timeout or payload too large | Split records or increase `VA_LSE_LLM_CALL_TIMEOUT_SECONDS` |
| **Apply settings** rejects a model name ("ends with '.'") | A model id copied out of a sentence or list kept its punctuation ("… the cheapest is `perplexity/glm-5.3-flash`.") | Delete the trailing `.` — copy ids from the provider's own model list, never from running prose. The same check refuses the run before any call is made. If the field only *looks* like it ends in punctuation, it is clipped, not mistyped — the applied ids are printed in full beneath the model boxes |
| Fails with "content filter" message | Moderation filter on output | Retry (filter is stochastic); reword graphic details |
| GitHub commit shows a red **Preview/Production deployment failure** from `vercel[bot]` | Vercel's Git integration auto-detected this repo as a Python **serverless** app and demanded a WSGI `app`/`handler` export, which a Streamlit app can never have | See **Vercel Preview deployments fail** below |

## Vercel Preview deployments fail

`vercel[bot]` created one failed Preview deployment per push on September 18–19
(seven in total, including one Production twin), each reporting:

> Found app/main.py but it does not export a top-level "app", "application",
> or "handler" variable.

**Why it cannot be fixed in code.** Vercel's Python framework preset builds a
*serverless* app — short-lived request handlers exporting a WSGI/ASGI callable
from a single module. This application is a long-running **Streamlit server**
(`streamlit run app/main.py`) with background job queues; it is architecturally
incapable of that export, and `app/main.py` deliberately has no `app =` symbol.
No commit can satisfy both models at once.

**What was done (September 20, 2026).** All seven failed deployments were
deleted from the Vercel project, their GitHub deployment records marked
`inactive`, and the project's framework preset set to *Other* so the detector
cannot fire again. The GitHub→Vercel connection had already gone quiet after
September 19, and the project never had a successful deployment.

**If Vercel previews start failing again**, the Git integration has been
re-connected: disconnect `VALayStatementEvaluator` from the Vercel project in
the Vercel dashboard (**Project Settings → Git → Disconnect**), or delete the
project. This repo's supported hosting paths are Streamlit Community Cloud (the
deployed app), Docker Compose, and Kubernetes — see `DEPLOYMENT.md` §2–§4.
Railway deploys succeed through its own integration and are unaffected.

## VA.gov downloader and CDP attach

`scripts/va_records_download.py` problems, including the Chrome remote-debugging
(`--cdp`) route. Security context for both: [SECURITY.md §9](SECURITY.md#9-local-browser-automation-cdp-attach-and-derived-phi-at-rest).

| Symptom | Likely cause | Fix |
|---|---|---|
| Script never detects sign-in; keeps waiting | You signed in, but the wizard URL was not reached (closed the tab, or VA.gov redirected elsewhere) | Leave the window on `va.gov` after ID.me; the script watches for `/my-health` and `/my-va` paths. Restart the script and sign in again |
| `--cdp` attach fails with connection refused | Chrome is not listening on the port — it was started without the flag, already exited, or the debug port closed when you quit it | Start Chrome with `--remote-debugging-port=9222 --user-data-dir="$HOME/.va_lse_debug_chrome"` *before* running the script; the port exists only while that Chrome runs |
| Chrome refuses `--remote-debugging-port` on your normal profile | Chrome 136+ rejects remote debugging against the default profile | Use a dedicated `--user-data-dir` (also the safer setup — the debug port can run JavaScript in whatever profile it is attached to) |
| A wizard step fails and the script names it | VA.gov changed its markup | Update the selector candidates in the script (`ALL_TIME_OPTIONS`, `SELECT_ALL_RECORDS_OPTIONS`, …) using the screenshot + page HTML it wrote to `va_gov_download_artifacts/`. Treat those artifacts as PHI — they are signed-in pages |
| Downloaded PDF has no extractable text (the app reports image-only pages) | VA.gov exports scans for some record types | OCR locally first: `python scripts/ocr_records.py <file>` — see README *Scanned pages and OCR* |
| Signed-in session unexpectedly expired | The profile directory was deleted, or the browser was closed in a way that dropped session cookies | Sign in again (full ID.me + SMS); use `--no-persist` if you want throwaway sessions by design. The session profile `~/.va_lse_va_gov_profile` holds cookies — delete it whenever you want to force sign-out |

## When to Contact Support

- The endpoint is reachable but returns unexpected errors
- You've verified key/URL match but calls still fail
- You need help choosing a fallback provider
- The circuit breaker keeps opening despite a healthy-looking endpoint

## Quick Reference: Environment Variables

```bash
# Primary endpoint (required)
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://...

# Fallback endpoint (optional but recommended)
OPENAI_BASE_URL_FALLBACK=https://...
OPENAI_API_KEY_FALLBACK=sk-...
LLM_MODEL_MAIN_FALLBACK=...
LLM_MODEL_FAST_FALLBACK=...
LLM_ENDPOINT_FALLBACK_TIMEOUT_SECONDS=300

# Circuit breaker (optional tuning)
VA_LSE_CB_FAILURE_THRESHOLD=3
VA_LSE_CB_RECOVERY_SECONDS=60

# Record processing (optional tuning)
VA_LSE_RECORDS_CONCURRENCY=2
VA_LSE_DIGEST_CHUNK_CHARS=8000
VA_LSE_LLM_CALL_TIMEOUT_SECONDS=300
```
