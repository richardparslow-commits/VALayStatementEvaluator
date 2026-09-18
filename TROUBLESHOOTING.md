# Troubleshooting: LLM Circuit Breaker Open Error

## Error Message

```
Drafting failed: records:review: LLMError: Record review failed: could not digest
chunk 1, chunk 2, ... chunk N after a retry (Circuit breaker 'llm' is OPEN — LLM
endpoint temporarily unavailable. Failing fast to protect the endpoint (retry in 60s).
After 3 consecutive failures the breaker opened for 60s.). Re-run the review; if it
persists, split the record set into smaller files. (reference: req_xxxxx)
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

### 1. Wait and Retry (Simplest)

The circuit breaker auto-recovers after the recovery timeout (default 60s). Wait a
minute and re-run your review.

```bash
# No action needed - just wait ~60 seconds and re-run
```

### 2. Check LLM Endpoint Health

Verify your LLM endpoint is actually reachable and working:

**Using the app's built-in test:**
1. Open the Streamlit app sidebar
2. Enter your API key and base URL (if not already set)
3. Click **"Test connection"** — this checks if the endpoint responds

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

- **Primary:** QwenCloud → **Fallback:** OpenAI
- **Primary:** OpenAI → **Fallback:** QwenCloud or another OpenAI-compatible endpoint
- **Primary:** Any → **Fallback:** Local Ollama (for offline/dev)

### Step 2: Add Fallback Configuration

Edit your `.env` file (or Streamlit secrets for hosted deployments):

```bash
# .env example - OpenAI as backup for QwenCloud primary
OPENAI_BASE_URL=https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=sk-sp-your-primary-key

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

Key metrics:
- `va_lse_circuit_breaker_state` — CLOSED/OPEN/HALF_OPEN
- `va_lse_llm_failover_active` — 1 when serving from fallback
- `va_lse_llm_primary_unhealthy_seconds` — how long primary has been down

## Common Causes and Fixes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| All chunks fail immediately | API key/base URL mismatch | Verify key matches endpoint provider |
| Fails after some successful calls | Rate limiting | Reduce `VA_LSE_RECORDS_CONCURRENCY` or wait |
| Intermittent failures | Provider outage or network issue | Configure fallback endpoint |
| Fails only on large files | Timeout or payload too large | Split records or increase `VA_LSE_LLM_CALL_TIMEOUT_SECONDS` |
| Fails with "content filter" message | Moderation filter on output | Retry (filter is stochastic); reword graphic details |

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
