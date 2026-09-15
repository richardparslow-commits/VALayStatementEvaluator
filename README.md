# 🎖️ VA Lay Statement Evaluator

A Streamlit application that performs **exhaustive medical-record review** to:

1. **Evaluate** an existing lay/witness statement (VA Form 21-10210 style) — extracting every
   factual claim, verifying each claim against the medical records, scoring the statement on an
   8-dimension rubric drawn from VA lay-evidence law, auditing it against the **topic
   checklist** (hazards and dangers, caregiver necessity, personal care, medication and
   financial management, household safety, errands and driving, before/after progression,
   observable behaviors, family impact, medication side effects), and then suggesting how to
   improve it: a prioritized improvement plan plus a proposed rewrite with record-grounded
   corrections; and
2. **Draft** a new, factually grounded statement from a witness's own observations — checking
   the observations against the topic checklist, asking follow-up questions for applicable
   topics not yet covered, and flagging anything that conflicts with or cannot be verified in
   the records.

> ⚠️ **Disclaimer.** This tool is an educational and drafting aid. It is **not** legal, medical,
> or claims advice. No output should be submitted to the VA without the witness personally
> verifying every fact. For accredited help see
> [www.va.gov/ogc/apps/accreditation](https://www.va.gov/ogc/apps/accreditation).

## Legal foundation baked into the tool

- Competent lay evidence: 38 C.F.R. § 3.159(a)(2); *Jandreau v. Nicholson*, 492 F.3d 1372 (Fed. Cir. 2007)
- Benefit of the doubt: 38 U.S.C. § 5107(b); *Gilbert v. Derwinski*
- Combat presumption & duty to consider all lay evidence: 38 U.S.C. § 1154(a); *Buchanan v. Nicholson*, 451 F.3d 1331
- Absence from records ≠ negative evidence: *Barr v. Nicholson*; *Buczynski v. Shinseki*
- Nexus lay competence limits: *Layno v. Brown*; *Kahana v. Shinseki*; *Davidson v. Shinseki*

Key design rule: a claim absent from the records is reported as **NOT FOUND**, never as
contradicted; only an explicit record conflict is **CONTRADICTED**.

## Architecture

```
run_app.py                Streamlit launcher
app/
  main.py                 UI: Evaluate / Draft / About tabs
  config.py               Settings (.env), knowledge-file loader
  fetch_client.py         Fetch Sandbox GET client -> normalized record documents
  va_gov_client.py        VA.gov auth/fetch/merge client (real HTTPS or in-memory mock)
  telemetry.py            Agiloop Inspect telemetry helper (feature-id-neutral)
  llm.py                  OpenAI-compatible client (retry, JSON parsing, circuit breaker & concurrency limiter)
  circuit_breaker.py      Stdlib circuit breaker (fail-fast after N failures) + in-memory concurrency limiter & bounded queue
  shutdown.py             Graceful SIGTERM/SIGINT shutdown: inflight drain, /ready 503, per-call timeout
  documents.py            TXT/MD/DOCX/PDF extraction, page-aware chunking
  medical_review.py       Exhaustive chunked record review -> fact digest
  evaluate.py             Claim extraction -> verification -> rubric scoring -> topic coverage
                          audit -> improvement suggestions & proposed rewrite -> report
  draft.py                Grounding + topic coverage -> draft -> self-review pipeline
  condition_selector.py   Claimed-condition selector: body-system radio buttons + searchable
                          condition dropdown -> auto pre-selects relevant topics from the
                          12-topic checklist; A&A/SMC-L toggle forces topics B, C, E, J
  condition_topics.json   Body-system -> condition -> topic-checklist mapping (34 conditions)
  agiloop_telemetry.py    Agiloop Inspect telemetry client (impression/interaction/error/goal)
  knowledge/              legal_framework.md, evaluation_rubric.md, drafting_guide.md,
                          topic_checklist.md
scripts/
  extract_pdfs.py         Build reference_docs/extracted/*.txt from source PDFs
  smoke_test.py           End-to-end pipeline test against the live LLM endpoint
  scale_sim.py            Offline 2,000-page pipeline simulation (no API calls)
tests/                    Offline unit tests (no API key required)
examples/                 Fictional sample statement + sample medical records
Dockerfile                Production container image (non-root, hash-pinned deps)
docker-compose.yml        Multi-instance: 3 Streamlit replicas + nginx (Pattern A)
nginx/                    Reverse proxy config with session affinity
deploy/k8s/               Kubernetes manifests (Deployment, Service, Ingress, HPA, Redis)
DEPLOYMENT.md             Multi-instance deployment guide (Docker Compose, K8s, session persistence)
```

Long documents are processed in overlapping, page-labelled chunks so reviews are exhaustive
regardless of record length. See **Large record sets** below for how very large files scale.

> 📐 **Design rationale & trade-offs.** The short map above covers *what*. For *why* —
> why Streamlit, why hierarchical fact merging over a single mega-call, why chunking at
> paragraph boundaries, why concurrency is capped at 2 and model split matters, and how the
> evaluate/draft data flows work — see **[`ARCHITECTURE.md`](ARCHITECTURE.md)** (ADRs, data-flow
> diagrams, scale-engine deep dive, and "constraints you must not break").

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --require-hashes -r requirements.lock   # exact, hash-verified versions
cp .env.example .env     # then put your API key in .env (never commit .env — see SECURITY.md)
```

> 🔒 **Secrets safety.** Never commit `.env` or any key to git — see
> [`SECURITY.md`](SECURITY.md) for storage, rotation, CI/CD (GitHub Secrets),
> production secret stores (AWS Secrets Manager / Azure Key Vault), and the
> pre-commit hook (`scripts/hooks/pre-commit`) that blocks accidental commits.

Install from **`requirements.lock`** for development, CI, and production so every environment
runs the identical tested dependency set. `requirements.txt` stays the human-edited input
(minimum versions); see **Dependency locking** below.

> This app is tested with QwenCloud Token Plan but works with **any
> OpenAI-compatible API** (OpenAI, Azure OpenAI via proxy, local Ollama with an
> OpenAI-compat shim). See [`COMPATIBILITY.md`](COMPATIBILITY.md) for supported
> endpoints and models and [`MIGRATION.md`](MIGRATION.md) for switching providers.
> At launch the sidebar warns if your configured models are not listed at
> `GET {base_url}/models` (non-blocking; network failures are ignored).

### Dependency locking

`requirements.txt` lists direct dependencies with minimum versions; `requirements.lock` is the
`pip-compile`-generated lockfile pinning **every** direct and transitive package to an exact
version with SHA-256 hashes, so local dev, CI, and deploys resolve identically. Every install
path uses it (`pip install --require-hashes -r requirements.lock` in `.github/workflows/test.yml`
and `.agiloop/deploy.json`).

To upgrade dependencies:

```bash
pip install pip-tools                 # provides pip-compile
# 1. edit requirements.txt (add a package or raise a minimum version)
# 2. re-resolve the full graph and rewrite the lockfile
pip-compile --generate-hashes requirements.txt -o requirements.lock
# 3. review the diff, run the offline tests, then commit BOTH files together
python -m unittest discover -s tests -v
git add requirements.txt requirements.lock && git diff --cached
```

Use `pip-compile --upgrade` (optionally with `-P <package>`) for a deliberate bulk or
single-package bump; pip-compile pins hashes for every platform wheel, so the same lockfile
installs on macOS and Linux CI.

### Environment variables (`.env`)

| Variable | Meaning | Default |
|---|---|---|
| `OPENAI_API_KEY` | QwenCloud Token Plan API key (starts `sk-sp-`) | (required) |
| `OPENAI_BASE_URL` | OpenAI-compatible base URL (Token Plan) | `https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1` |
| `LLM_MODEL_MAIN` | Low-volume heavy model (analysis/scoring/drafting) | `qwen3.7-max` |
| `LLM_MODEL_FAST` | Cheap model for the bulk digest/merge passes | `qwen3.7-flash` |
| `VA_LSE_MAX_RECORD_PAGES` | Max total pages across uploaded record files | `5000` |
| `VA_LSE_RECORDS_CONCURRENCY` | Parallel chunk-digest workers | `2` (Lite plan fits 1–2 concurrent agents) |
| `VA_LSE_MAX_DIGEST_FACTS` | Max facts kept in the consolidated digest | `1500` |
| `VA_LSE_DIGEST_CHUNK_CHARS` | Characters per record chunk | `8000` |
| `VA_LSE_DOCX_MAX_INTERNAL_FILE_BYTES` | Max uncompressed bytes allowed for a single DOCX internal file | `52428800` |
| `VA_LSE_DOCX_MAX_TOTAL_UNCOMPRESSED_BYTES` | Max total uncompressed bytes allowed across all DOCX internal files | `209715200` |
| `VA_LSE_DOCX_MAX_INTERNAL_FILE_COUNT` | Max number of internal files allowed in a DOCX archive | `10000` |
| `FETCH_SANDBOX_API_KEY` | Optional Fetch Sandbox API key | empty |
| `FETCH_SANDBOX_BASE_URL` | Fetch Sandbox base URL (`fetchsandbox.com` or subdomain) | `https://fetchsandbox.com` |
| `FETCH_SANDBOX_RECORDS_PATH` | GET path for the records endpoint | `/medical_records/{patient_id}` |
| `FETCH_SANDBOX_MAX_RESPONSE_BYTES` | Max bytes accepted from a Fetch Sandbox HTTP response | `104857600` |
| `VA_LSE_ALLOW_LOCAL_PATHS` | Force-enable the local folder/file record source (`1`) even when the local-run check can't detect localhost | (auto) |
| `VA_LSE_CREDITS_PER_1M_MAIN` | Approx credits per 1M tokens for the main model (enables the credit-burn gauge) | (unset — gauge shows tokens/calls only) |
| `VA_LSE_CREDITS_PER_1M_FAST` | Approx credits per 1M tokens for the fast model (enables the credit-burn gauge) | (unset — gauge shows tokens/calls only) |
| `VA_LSE_CREDIT_QUOTA` | Your plan's weekly credit quota, used to render %-of-quota burn | `2500` |
| `VA_GOV_API_BASE_URL` | HTTPS base URL for the VA.gov record-retrieval API. Leave unset to run VA.gov auth/fetch in mock mode. | empty (mock mode) |
| `FRONTEND_URL` | App origin for CORS allowlisting. Unused today (single-origin Streamlit app); documented for deploy-harness forward compatibility. | empty |
| `AGILOOP_INSPECT_API_KEY` | Server-side Agiloop Inspect telemetry API key. Leave unset (with `AGILOOP_PROJECT_ID`) to run telemetry in mock/no-op mode. | empty (mock mode) |
| `AGILOOP_INSPECT_URL` | Agiloop Inspect telemetry endpoint base URL | `https://inspect.api.agiloop.app` |
| `AGILOOP_PROJECT_ID` | Agiloop project id for telemetry event routing | empty (mock mode) |
| `VA_LSE_LOG_LEVEL` | Structured log level (`DEBUG`/`INFO`/`WARNING`…) | `INFO` |
| `VA_LSE_LOG_JSON` | `1` → JSON lines for ELK/CloudWatch/Datadog; `0` → plain text | `0` |
| `VA_LSE_LOG_DIR` | Directory for rotating `app.log` (also always logs to stdout) | empty (stdout only) |
| `VA_LSE_LOG_FILE` | Filename inside `VA_LSE_LOG_DIR` | `app.log` |
| `VA_LSE_LOG_MAX_BYTES` | Rotate size per log file (bytes) | `10485760` (10 MiB) |
| `VA_LSE_LOG_BACKUPS` | Rotated files kept | `5` |
| `VA_LSE_CB_FAILURE_THRESHOLD` | Consecutive LLM failures before circuit breaker opens (fail-fast while open) | `3` |
| `VA_LSE_CB_RECOVERY_SECONDS` | Cooldown while breaker is open before allowing a probe | `60` |
| `VA_LSE_MAX_CONCURRENT_LLM_CALLS` | Max simultaneous LLM calls (extra callers queue) | `20` |
| `VA_LSE_LLM_QUEUE_MAX_DEPTH` | Max queued callers waiting for a concurrency slot | `50` |
| `VA_LSE_LLM_QUEUE_TIMEOUT_SECONDS` | Seconds a queued caller waits before `QueueFullError` | `30` |
| `VA_LSE_HEALTH_PORT` | Sidecar health server port (`0` disables `GET /health` & `GET /ready`) | `8001` |
| `VA_LSE_AUDIT_LOG_DIR` | Directory for the separate `audit.log` JSON stream (audit trail, distinct from `VA_LSE_LOG_DIR`) | `logs` (or `VA_LSE_LOG_DIR` when set) |
| `VA_LSE_AUDIT_LOG_FILE` | Filename inside `VA_LSE_AUDIT_LOG_DIR` | `audit.log` |
| `VA_LSE_AUDIT_LOG_MAX_BYTES` | Rotate size per audit log file (bytes) | `10485760` (10 MiB) |
| `VA_LSE_AUDIT_LOG_BACKUPS` | Rotated audit files kept | `10` |
| `VA_LSE_SHUTDOWN_GRACE_SECONDS` | Max seconds to wait for inflight runs on SIGTERM before orchestrator SIGKILL | `30` |
| `VA_LSE_LLM_CALL_TIMEOUT_SECONDS` | Per-LLM-call timeout (prevents hung calls from blocking graceful shutdown) | `300` (5 min) |
| `VA_LSE_PIPELINE_TIMEOUT_SECONDS` | Total wall-clock timeout for an entire Evaluate/Draft run (including digest + merge) | `1800` (30 min) |
| `VA_LSE_MEMORY_WARN_MB` | RSS memory warning threshold (MB); run aborted below 200 MB | `500` |

Leave the `VA_GOV_API_BASE_URL` group or the `AGILOOP_INSPECT_*` group fully unset to run
those integrations in mock mode. Partially configuring an integration (e.g. setting
`AGILOOP_INSPECT_API_KEY` without `AGILOOP_PROJECT_ID`) does not fail startup for this app —
telemetry simply logs that combination as mock and drops events, since telemetry must never
block the app.

All settings can also be overridden live in the app sidebar. Model availability depends on your
provider: the app checks `GET {base_url}/models` at startup and warns if `LLM_MODEL_MAIN` or
`LLM_MODEL_FAST` is not listed (see [`COMPATIBILITY.md`](COMPATIBILITY.md)).

### Telemetry (Agiloop Inspect)

The app ships with usage telemetry (impressions, interactions, errors, goals) for the
claimed-condition selector, sent via `app/agiloop_telemetry.py`. Because Streamlit runs
entirely server-side, this module sending events directly to Inspect *is* the same-origin
proxy pattern — the rendered page never holds or transmits the API key. Telemetry is fully
optional: leave `AGILOOP_INSPECT_API_KEY` and `AGILOOP_PROJECT_ID` unset to run in mock mode
(events are logged at debug level and dropped; the app works identically either way).

### Estimating API usage & credit burn

Every run shows a live usage line in the progress caption and, after completion, an expandable
"Estimated API usage" table in the results with **calls, input, and output tokens per phase**
(record digest/merge/summary, claims, verification, rubric, topic audit, revision — or grounding/
draft/review on the Draft tab). Token counts are estimates based on prompt length and model
output, using the provider's reported usage when the endpoint supplies it.

Because QwenCloud Token Plan doesn't publish a fixed credits-per-1M-token rate, you can provide
it two ways:

1. **Set explicit rates** — `VA_LSE_CREDITS_PER_1M_MAIN` and `VA_LSE_CREDITS_PER_1M_FAST` to
   your plan's effective rates (explicit values always win), plus `VA_LSE_CREDIT_QUOTA` if your
   quota differs from the 2,500-credit Lite default.
2. **Let the watchdog learn it** (default). Every finished run persists its per-role token totals
   (main vs fast model) to a git-ignored `usage_history.json`. In the sidebar's **Usage watchdog**
   panel, enter the plan's cumulative "credits used" reading from the QwenCloud console each time
   after a run. With readings separated by new runs, the app fits a **separate credits-per-1M rate
   per model** by least squares over the calibration intervals (falling back to one blended rate
   when the data can't separate them), and uses those rates for the credit-burn estimate.

The estimator is purely informational — it never limits or throttles a run.

Fetch Sandbox settings can also be overridden in the sidebar. Because Fetch Sandbox mirrors
your own OpenAPI spec, you must point `FETCH_SANDBOX_RECORDS_PATH` at the GET endpoint your
sandbox exposes for record retrieval.

Fetch Sandbox HTTP responses are read in chunks and capped by
`FETCH_SANDBOX_MAX_RESPONSE_BYTES` (default 100 MB). If a response exceeds this
limit, the import fails with `FetchSandboxError`.

## Run

```bash
streamlit run run_app.py
```

### Production build & start

Matches `.agiloop/deploy.json` (single `streamlit-web` service):

```bash
# Build
pip install --require-hashes -r requirements.lock

# Start (binds to the platform-provided $PORT)
streamlit run run_app.py --server.port $PORT --server.address 0.0.0.0
```

> 🚀 **Scaling to multiple instances?** For multi-instance deployment behind a load
> balancer, Docker Compose with nginx, Kubernetes with session affinity, or Redis-backed
> session persistence — see **[`DEPLOYMENT.md`](DEPLOYMENT.md)** (Dockerfile, compose,
> k8s manifests, session tradeoff analysis).

### Evaluate a statement

1. Upload or paste the lay statement.
2. Choose a medical-record source:
   - **Upload files** (PDF/TXT/MD/DOCX, multiple files OK),
   - **Fetch Sandbox** (enter a patient or record ID and import from your sandbox endpoint),
   - **VA.gov** (secure per-session login + explicit consent, then automatic fetch of all
     available records — see **VA.gov record source** below), or
   - **Local folder / file** (local runs only — read records straight from a path on this
     machine, e.g. `~/Desktop/ClaimRecords`; hidden when the app is served remotely).
   - Security hardening: DOCX uploads with oversized uncompressed internal ZIP contents are
     rejected to prevent decompression-bomb memory exhaustion.
3. Pick the **claimed condition**: choose a body system (radio buttons), then search and
   select one or more conditions from the filtered dropdown. The app automatically
   pre-selects the relevant 12-topic-checklist topics (union across all selected
   conditions); toggle **Aid & Attendance / SMC-L** to force topics B, C, E, and J as
   mandatory. Adjust the pre-selection freely, then click **Proceed**.
4. Click **Run exhaustive evaluation** — watch chunked record review, claim verification,
   rubric scoring, improvement drafting, and report generation progress.
5. Review the verdict table (✅ supported / 🟡 partial / ❌ contradicted / ⚪ not found),
   scores, and the prioritized improvement plan.
6. Review the **proposed rewrite**: a change-by-change table (original → suggested → why),
   the revised statement with `[Confirm: ...]` placeholders, and downloads for both the
   report and the revised statement.

### Draft a statement

1. Choose the veteran's medical-record source: upload files, import from Fetch Sandbox, or
   read from a local folder/file path (local runs only).
2. Pick the **claimed condition** (body system + searchable dropdown) the same way as in
   Evaluate mode; adjust the pre-selected topics or toggle **Aid & Attendance / SMC-L**, then
   click **Proceed**.
3. Enter witness details and bulleted firsthand observations.
4. Click **Draft the statement** — the app grounds every observation in the records, flags
   conflicts, suggests strengthening questions, drafts the statement, and self-reviews it.
5. Resolve every bracketed `[Confirm: ...]` placeholder with the witness before signing.
   Submit on VA Form 21-10210 (one form per witness).

## Large record sets (1 to ~5,000 pages)

The reviewer is built for full VA claim files, including bundles of 1,000–2,000+ pages:

- **Parallel digestion** — record chunks are extracted by `VA_LSE_RECORDS_CONCURRENCY`
  workers at once instead of serially, so a ~1,900-page file that would take hours
  sequentially completes in tens of minutes. Progress is reported per chunk.
- **Duplicate-page skipping** — pages repeated within or across files (very common in
  VA bundles) are hash-detected and skipped, with the count shown in the results.
- **Transient-failure tolerance** — a chunk that fails (e.g. rate limit) is retried
  once; the run only aborts if it still fails, and the failing chunks are named.
- **Hierarchical fact merging** — thousands of extracted facts are consolidated in
  parallel batches (never one oversized call), deduplicated, and capped at
  `VA_LSE_MAX_DIGEST_FACTS`.
- **No evidence lost to truncation** — claim verification and draft grounding do not
  read only the head of the digest. Each claim batch retrieves the digest facts most
  relevant to it (IDF-weighted term matching) plus matching raw-record excerpts, so
  evidence buried on page 1,700 is found just like evidence on page 2.
- **Full-timeline summaries** — the narrative record summary samples facts evenly across
  the whole timeline instead of only the earliest documents.

Tuning: raise `VA_LSE_RECORDS_CONCURRENCY` if your endpoint allows more parallel
requests; lower `VA_LSE_DIGEST_CHUNK_CHARS` for extra recall on very dense pages (at the
cost of more LLM calls). `scripts/scale_sim.py` runs an offline 2,000-page simulation of
the pipeline (no API calls) to verify orchestration at scale.

## Tests

```bash
python -m unittest discover -s tests -v        # offline unit tests (incl. health probes)
python -m mypy app                             # strict type check (see pyproject.toml)
python scripts/smoke_test.py all               # live end-to-end (needs valid .env)
```

### Type checking (mypy — strict)

`pyproject.toml` enables strict `mypy` for `app/` (`disallow_untyped_defs`, `warn_return_any`,
`no_implicit_optional`, etc.; tests/scripts are relaxed). CI runs `mypy app` as a blocking gate
on every push/PR. Run locally with `mypy app` (install `pip install -r requirements-dev.txt` once).
All public helpers in `app/main.py`, `app/fetch_client.py`, `app/evaluate.py` carry precise types
(`ExtractedDocument`/`EvaluationResult`/`DraftResult`/`Settings` etc.) instead of `Any`.

> GitHub Actions installs from the hash-pinned `requirements.lock` (see **Dependency locking**),
> then runs the offline tests and scale simulation automatically on every push to `main` (and on
> pull requests). The live smoke test is triggered **manually** from the
> Actions tab and only runs when an `OPENAI_API_KEY` secret is configured; the optional
> `OPENAI_BASE_URL`, `LLM_MODEL_MAIN`, and `LLM_MODEL_FAST` secrets override the endpoint and
> models in that job if set (see `.env.example`).

## QwenCloud Individual Plan Lite tuning

These defaults are tuned for a single user on the QwenCloud Individual Plan Lite subscription
($8/month, **2,500 Credits per rolling 7-day window**, 1–2 concurrent agents):

- **Base URL / key are paired** — the `sk-sp-` Token Plan key only works with the Token Plan
  base URL; they never work against the general MaaS gateway.
- **Model split is the biggest credit saver.** The bulk record-digest and merge passes (one
  call per chunk — by far the most calls, especially on large files) run on the cheap
  `qwen3.7-flash`. Only the low-volume, high-value steps — claim extraction, verification,
  rubric scoring, topic audit, and the rewrite — use the strong `qwen3.7-max`.
- **Concurrency is capped at 2** to match the plan's 1–2 agent limit; higher parallelism just
  triggers rate limiting.
- **Watch the window quota.** A single exhaustive run over a very large record set (hundreds
  to thousands of pages) can consume much of the 2,500-credit quota. Run the `examples/`
  sample first to gauge burn, and consider the Credit Pack add-on for heavy use.

## Fetch Sandbox contract

This integration assumes the sandbox exposes a **GET** endpoint that returns JSON. The app
supports these response shapes:

- `{ "documents": [...] }`, `{ "records": [...] }`, `{ "files": [...] }`, or `{ "items": [...] }`
- a top-level JSON array of document items
- a single structured JSON object, which the app will import as one JSON-backed document

Each document item may provide one of:

- `text` / `content` / `body` / `markdown`
- `base64` / `data_base64` / `file_base64` / `content_base64`
- `download_url` / `url` / `file_url` / `href`

Optional metadata fields:

- `filename` / `name` / `title`
- `content_type` / `mime_type` / `media_type`

When an API key is provided, the app sends both a bearer-token auth header and an `X-API-Key`
header to maximize compatibility with different sandbox auth setups.

For safety, the Fetch base URL must point to `fetchsandbox.com` (or one of its subdomains),
and imported document URLs must resolve to that same host.

### Testing locally with the mock sandbox

A stdlib-only mock server is included at `scripts/mock_fetch_sandbox.py` for local testing. Because
the app only accepts `*.fetchsandbox.com` hosts, map a subdomain to localhost in `/etc/hosts`, then
run the mock and point the app at it:

```sh
# 1. /etc/hosts: 127.0.0.1  local.fetchsandbox.com
# 2. run the mock:
.venv/bin/python scripts/mock_fetch_sandbox.py
# 3. app sidebar: base URL http://local.fetchsandbox.com:8001, records path /medical_records/{patient_id}
```

The mock responds to both URL styles the app emits (`/medical_records/{patient_id}` and
`/medical_records?patient_id=...`) with a two-document JSON payload — see the script's docstring
for full instructions.

## VA.gov record source

Selecting **VA.gov** as the record source (Evaluate or Draft) opens a secure, per-session login
form with an explicit consent checkbox. After consenting and signing in, the app automatically
fetches all available VA.gov records for that session, merges them with any other sources
already loaded in the same workflow this session, and shows a **merged records summary** (source
label + file + page count per row) that requires explicit confirmation before the merged set is
used for evaluation or drafting.

- **Mock mode (default):** leave `VA_GOV_API_BASE_URL` unset — `authenticate_va_gov` and
  `fetch_va_records` return a deterministic in-memory mock session and two mock records, so the
  full login → fetch → merge → confirm flow works with zero VA.gov env vars configured.
- **Real mode:** set `VA_GOV_API_BASE_URL` (must be `https://`) to call a real VA.gov-compatible
  authenticated record-retrieval API. Requests retry up to 3 times with exponential backoff.
- **Partial/connection-error handling:** if VA.gov returns fewer records than expected or the
  connection drops, the app shows the retrieved-vs-expected counts, a **Retry** button, and a
  **Continue with available records** option — VA.gov failures never block using the other
  record sources.
- **Privacy:** VA.gov records are treated identically to every other source for extraction,
  chunking, duplicate detection, and page labeling. Fetched records and the VA.gov session token
  live only in `st.session_state` for the current browser session; credentials are **never**
  written to disk, `.env`, or logs.## Health checks (container orchestration)

A stdlib-only sidecar (`app/health.py`, started from `run_app.py` before Streamlit) exposes two
orchestrator-friendly probes on `0.0.0.0:$VA_LSE_HEALTH_PORT` — no extra dependencies:

| Endpoint | Meaning | Status | Latency |
|---|---|---|---|
| `GET /health` | **Liveness** — the process is up | `200` with `{status:"ok", service, uptime_s}` | < 50 ms |
| `GET /ready` | **Readiness** — LLM endpoint + configured models are reachable (`GET {base_url}/models`) | `200` when ready, `503` when not (JSON always includes `ready` + `detail`) | < 2 s (probe timeout 1.4 s, cached 30 s) |
| `HEAD /health`, `HEAD /ready` | Same as GET but no body — for probes that use HEAD | same | same |
| any other path | | `404` | — |

```bash
# Local quick check
curl -s http://localhost:8001/health | python -m json.tool
curl -s -w "%{http_code}\n" http://localhost:8001/ready
# Change or disable the sidecar
VA_LSE_HEALTH_PORT=9001 streamlit run run_app.py   # different port
VA_LSE_HEALTH_PORT=0 streamlit run run_app.py      # disable sidecar entirely
```

- **Liveness** never touches the LLM gateway — it is `200` as soon as the Python process starts.
- **Readiness** calls `GET {base_url}/models` with a 1.4 s timeout and is cached for 30 s so the
  handler always meets the < 2 s SLO even when the LLM gateway is slow. Missing `OPENAI_API_KEY`,
  missing `OPENAI_BASE_URL`, or any network/auth/parse failure → `503` with a short `detail` (no
  key leaked). Callers should treat `503` as "not ready, keep out of the load-balancer pool."
- **Kubernetes / Agiloop / Docker Swarm** — point `livenessProbe` at `httpGet: path:/health port:8001`
  and `readinessProbe` at `httpGet: path:/ready port:8001` (adjust `port` if you override
  `VA_LSE_HEALTH_PORT`). Example hints for a `Deployment` are in the docstring of `app/health.py`.
- The sidecar binds with `ThreadingHTTPServer` (stdlib) on a **daemon thread** and is idempotent —
  if the port is already in use it logs a warning and the Streamlit app still starts (health is
  best-effort). Tests (`tests/test_health.py`) cover `/health`, `/ready` (cached/mocked), `HEAD`,
  and `404` without touching the network.

## Resilience (circuit breaker & concurrency limiting)

`app/llm.py` (via `app/circuit_breaker.py`) wraps every LLM call with two guards so 100 concurrent users cannot turn a brief endpoint degradation into a prolonged outage:

| Guard | What it does | Defaults | Tuning |
|---|---|---|---|
| **Circuit breaker** | Counts *logical* LLM failures (a call that exhausts its 3 retries is one). After `VA_LSE_CB_FAILURE_THRESHOLD` consecutive failures it **opens**: every new `chat` fails fast with `CircuitBreakerOpenError` in <50 ms (no network, no retries), protecting the endpoint. After `VA_LSE_CB_RECOVERY_SECONDS` it enters `HALF_OPEN` and lets one probe through — success closes it, failure re-opens it. | `threshold=3`, `recovery=60s` | Lower the threshold for faster fail-fast; raise `recovery` on flaky gateways |
| **Concurrency limiter** | Global semaphore (`VA_LSE_MAX_CONCURRENT_LLM_CALLS`, default `20`) caps simultaneous LLM calls across all threads/users. Extras queue; up to `VA_LSE_LLM_QUEUE_MAX_DEPTH=50` are queued and block up to `VA_LSE_LLM_QUEUE_TIMEOUT_SECONDS=30s`. Beyond either limit the call is rejected with `QueueFullError` (no retry). | `concurrent=20`, `queue=50`, `timeout=30s` | Raise `MAX_CONCURRENT` on higher-tier endpoints; raise `MAX_DEPTH` on bursty multi-user hosts |

* Queue + breaker interact correctly: the breaker is checked **before** queuing (immediate fail-fast when open) and **again** after queuing (in case it opened while waiting). Queue-full or breaker rejections are **not** counted as endpoint failures. All breaker state changes (`CLOSED → OPEN`, `OPEN → HALF_OPEN`, `HALF_OPEN → CLOSED/OPEN`) log at `WARNING` with `phase=circuit_breaker`; queue-full/timeout log at `WARNING` with `phase=concurrency` — wire these to your alerting. `CircuitBreakerOpenError`/`QueueFullError` are re-exported from `app/llm.py` so callers can distinguish them from `LLMError`. Tests in `tests/test_circuit_breaker.py` cover the full state machine, the fail-fast <50 ms SLO, and the limiter queue off offline (no network).

## Audit logging (Evaluate & Draft)

Every Evaluate and Draft run emits **two** audit entries to a separate JSON
stream (`{VA_LSE_AUDIT_LOG_DIR}/audit.log`, default `logs/audit.log`) so
forensics and retention tooling can query it without scraping the diagnostic
`app.log`. The stream is independent: it has its own rotating file logger
(`audit.log`, 10 MiB, 10 backups by default) and a distinct `audit` logger
name, so you can route or retain it differently.

**Entry fields (no PII):** `timestamp`, `action` (`evaluate`|`draft`),
`status` (`start`|`ok`|`error`), `request_id` (`req_…` for correlation with
diagnostic logs), `user_session_id` (`sess_…`, stable per browser session),
`condition` (claimed-condition label only, truncated), `record_sources` (source
labels e.g. `Upload`/`Fetch Sandbox`/`VA.gov`/`Local folder / file`), `record_files`,
`record_pages`, `duration_ms`, and a small `outcome` classification (`claims`/`contradictions`/`overall_rating`
for Evaluate; `draft_chars`/`grounding_items` for Draft). Error entries add `error_class` +
user-facing `error_message`. **Never logged:** statement / observations / record
text, veteran/witness names, or file content.

```bash
# Tail the audit log (JSON lines, jq-friendly)
cat logs/audit.log | python -m json.tool   # or: tail -f logs/audit.log
# Filter by action
cat logs/audit.log | python -c "import json,sys; [print(l) for l in sys.stdin if json.loads(l).get('action')=='evaluate']"
# Env overrides (see .env.example)
VA_LSE_AUDIT_LOG_DIR=/var/log/va-lse VA_LSE_AUDIT_LOG_FILE=audit.log \
VA_LSE_AUDIT_LOG_MAX_BYTES=10485760 VA_LSE_AUDIT_LOG_BACKUPS=10 \
streamlit run run_app.py
```

The audit logger is best-effort and never blocks a run; a failure to open the
audit file falls back to stdout (the `audit` logger on `sys.stdout`) so
deployments without a writable log directory still emit the stream.

## Telemetry (Agiloop Inspect)

The app reports usage telemetry (impressions, interactions, errors) to Agiloop Inspect via
`app/telemetry.py`. Because this is a single-origin Streamlit app with no browser JS bundle, the
Inspect API key never leaves the Python process.

- **Mock mode (default):** leave `AGILOOP_INSPECT_API_KEY` and `AGILOOP_PROJECT_ID` unset —
events are logged at debug level and dropped instead of sent. Nothing about the app's behavior
changes; telemetry is always best-effort and never blocks the UI.
- **Real mode:** set both `AGILOOP_INSPECT_API_KEY` and `AGILOOP_PROJECT_ID` (and optionally
`AGILOOP_INSPECT_URL` to point at a non-default Inspect deployment) to send real events.
- `app/telemetry.py` is feature-id-neutral shared infrastructure: it never hardcodes a feature
id. Each feature's call sites (e.g. `app/va_gov_client.py`, `app/main.py`) supply their own
`featureId` explicitly.

## Graceful shutdown (SIGTERM/SIGINT)

Kubernetes (and similar orchestrators) send SIGTERM to a pod before SIGKILL.
Without a handler the process dies mid-request, aborting in-flight LLM calls
and losing the user's work.  This app handles SIGTERM and SIGINT to drain
in-flight runs cleanly:

1. **Signal handlers** are installed in `run_app.py` on the main thread before
   Streamlit starts (`app/shutdown.py`).  On SIGTERM/SIGINT a daemon thread
   sets a process-wide flag and waits up to `VA_LSE_SHUTDOWN_GRACE_SECONDS`
   (default 30 s) for inflight Evaluate/Draft runs to finish.
2. **`/ready` flips to 503** immediately so the orchestrator stops routing
   new traffic to this instance (`app/health.py` checks `is_shutting_down()`
   before the LLM probe).  Liveness (`/health`) stays 200 — the process is
   still alive and draining.
3. **New runs are rejected** with a user-visible warning in both the Evaluate
   and Draft tabs once shutdown begins (`app/main.py` checks `enter_run()`).
4. **Per-LLM-call timeout** (`VA_LSE_LLM_CALL_TIMEOUT_SECONDS`, default 300 s
   / 5 min) prevents a single hung call from blocking the drain forever.  A
   timeout surfaces as `LLMError` with a clear, actionable message.
5. If the grace window expires with runs still in flight, a WARNING is logged
   and the orchestrator's SIGKILL forces exit.

```bash
# Tune the drain window (default 30 s — increase for very large record sets)
VA_LSE_SHUTDOWN_GRACE_SECONDS=60 streamlit run run_app.py
# Tune the per-call timeout (default 300 s / 5 min)
VA_LSE_LLM_CALL_TIMEOUT_SECONDS=300 streamlit run run_app.py
```

Test locally with `kill -TERM <pid>` or Ctrl+C; inspect `logs/app.log` for
`phase=shutdown` WARNING lines showing `draining` → `drained` or `force_exit`.
See `ARCHITECTURE.md → Cross-cutting → Graceful shutdown` for the full design.

## Pipeline timeout and memory monitoring

Long-running Evaluate/Draft runs are protected by two backstops configured
via `app/pipeline_guard.py`:

| Guard | Default | What happens |
|---|---|---|
| **Pipeline timeout** | `VA_LSE_PIPELINE_TIMEOUT_SECONDS=1800` (30 min) | Run is aborted with `PipelineTimeoutError`; user sees a clear error message with elapsed time and the configured limit |
| **Memory pre-check** | `VA_LSE_MEMORY_WARN_MB=500` MB RSS | Warning logged if RSS exceeds threshold; run aborted with `MemoryError` if RSS < 200 MB |
| **Memory checkpoints** | After chunk dedup and merge | RSS logged at INFO (or WARNING if above threshold) so operators can see memory growth in structured logs |

The timeout wraps `run_evaluation` and `run_draft` in a worker thread via
`concurrent.futures.ThreadPoolExecutor`; when the deadline expires the caller
receives a `PipelineTimeoutError` (not `signal.alarm`, which only works on
the main thread).  Memory checks use `/proc/self/status` (Linux) or
`resource.getrusage` (macOS) and gracefully degrade on unsupported platforms.

```bash
# Tune for very large record sets (e.g. 5,000 pages)
VA_LSE_PIPELINE_TIMEOUT_SECONDS=3600 streamlit run run_app.py  # 60 min
# Raise memory warning for high-RAM servers
VA_LSE_MEMORY_WARN_MB=1024 streamlit run run_app.py
```

Both guards are also wired into `DEPLOYMENT.md` scaling guidance: with 100
users the per-pod timeout and memory limits prevent one user's runaway
process from destabilizing shared infrastructure.

## Distributed cache (VA reference data)

For multi-instance deployments, condition topics and other VA reference data
are cached in a shared backend (Upstash Redis / Vercel KV) so every Streamlit
pod serves the same data without redundant upstream fetches.  A process-local
LRU (256 entries) always runs as a fallback so single-instance deployments
need no Redis.

| Config | Default | Purpose |
|---|---|---|
| `VA_LSE_SHARED_CACHE_URL` | *(empty = local LRU only)* | Upstash REST URL (or Vercel KV endpoint) |
| `VA_LSE_SHARED_CACHE_TOKEN` | *(empty = disabled)* | Authentication token for the REST API |
| `VA_LSE_SHARED_CACHE_TIMEOUT_SECONDS` | `2` | HTTP timeout per cache call |
| `VA_LSE_SHARED_CACHE_LOCAL_MAXSIZE` | `256` | Local LRU fallback entries |

**Setup (Upstash free tier):**
1. Create a Redis database at [console.upstash.com](https://console.upstash.com/)
2. Copy the **REST URL** and **token** (HTTP API — not the `redis://` URL)
3. Set the two env vars above
4. Restart — `GET /health` reports cache backend and hit rate

```bash
# Quick verify
curl -s http://localhost:8001/health | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('cache',{}))"
# Should show {backend: 'upstash_redis', is_shared: true, reachable: true, ...}
```

**How it works:** Reads check shared first, then local LRU.  Writes populate
both tiers.  If the shared backend is unreachable the local LRU absorbs
traffic transparently — the app never hangs on cache I/O.  Hit rates are
reported in the health endpoint and structured logs.

## Structured logging (diagnostics & performance traces)

Every Evaluate/Draft run mints a correlation id (`req_…`) stored in `st.session_state` and a
`ContextVar` so parallel record-digest/merge workers carry it. The id is attached to every log
record (`request_id`), appended to user-facing errors as `reference: req_…` for post-mortem
correlation, and never carries PII — logs emit only phases, timings, counts, and classifications
(prompt/response bodies, statements, observations, and record text are excluded).

- **`app/logging_config.py`** — `configure_logging()` (idempotent), `JsonFormatter` (one JSON line
  per record for ELK/CloudWatch/Datadog) and `PlainFormatter` fallback, `PhaseTimer` context
  manager, and the `ContextVar` helpers. The `app` parent logger fans out to all `app.*`
  children, so existing modules need no individual setup.
- **`app/llm.py`** — every LLM call logs `phase`, `model`, `attempt`/`retries`, `duration_ms`,
  `prompt_tokens`/`completion_tokens` (or `"est"`), and `sys_chars`/`user_chars`/`out_chars`
  once; retries log at WARNING, final failures at ERROR with stack trace.
- **`app/medical_review.py`** — `records:review` start/done with `pages`/`chunks`/`facts`, per-
  worker `records:digest` with chunk label + duration, plus `records:merge`/`records:summary`
  via `PhaseTimer` so slow paths are traceable.
- **`app/evaluate.py` / `app/draft.py`** — full pipeline spans: each phase (`claims`, `verify`,
  `rubric`, `topic`, `revision`, `report` / `grounding`, `draft`, `review`) wrapped in
  `PhaseTimer`; the outer `run_evaluation`/`run_draft` wrapper logs pipeline start/done with
  `duration_ms` and routes the error via the feature-id-neutral error boundary.
- **`app/main.py`** — baseline `request_id` per session, per-run `req_…` minted on the action
  button, `evaluate`/`draft` start/done, `llm_config` validation, st.error/warning paths
  enriched with `reference: req_…`, and a root `app` error boundary.

Configure via env (see `.env.example`): `VA_LSE_LOG_LEVEL`, `VA_LSE_LOG_JSON` (JSON vs plain),
`VA_LSE_LOG_DIR`/`VA_LSE_LOG_FILE` (rotating file + stdout), `VA_LSE_LOG_MAX_BYTES` and
`VA_LSE_LOG_BACKUPS`. With `VA_LSE_LOG_DIR` unset the app still logs to stdout so platform drains
(`docker logs`, Agiloop build harness) stay useful; setting it adds a `RotatingFileHandler`.

## Production hardening (Streamlit)

`.streamlit/config.toml` is committed so shared deployments cannot silently run with
Streamlit's permissive defaults. It pins:

- `server.enableXsrfProtection = true` (XSRF on form posts / uploads)
- `client.toolbarMode = "minimal"` (no fork/deploy buttons in hosted mode)
- `server.headless = true`
- `logger.level = "info"`
- `server.maxUploadSize = 50` (per-file cap; matched by `VA_LSE_MAX_UPLOAD_BYTES` in `app/config.py`)

At startup `app/main.py` emits `⚠️ Streamlit security hardening is not fully active`
if the file is missing or those keys are absent (non-blocking warning).

Upload limits are also enforced in Python (`app/main.py:_check_upload_limits`) so the
tight 50 MB per-file / 200 MB batch caps (`VA_LSE_MAX_UPLOAD_BYTES` /
`VA_LSE_MAX_TOTAL_UPLOAD_BYTES`, overridable via env) produce a clear in-UI message
even when `maxUploadSize` is not active; total size is capped across the batch
(largest files dropped first until it fits).

Streamlit cannot set arbitrary HTTP response headers from `config.toml`. For
production, front the app with a reverse proxy (nginx / CloudFlare / Agiloop)
that adds:

- `Content-Security-Policy` (tight `default-src 'self'` with Streamlit-allowed inline styles/scripts)
- `X-Frame-Options: SAMEORIGIN` (clickjacking guard)
- `Strict-Transport-Security` (HSTS) and `X-Content-Type-Options: nosniff`
- Rate limiting per IP and upload throttling

`.streamlit/secrets.toml` is git-ignored — never store deploy keys there as a
checked-in file.

For the orchestrator probes, see **Health checks (container orchestration)** above —
front the same `GET /health` / `GET /ready` sidecar on `VA_LSE_HEALTH_PORT` (default `8001`)
with your proxy if the stream is TLS-terminated there.

## Compatibility & migration

- **Tested endpoints & models:** QwenCloud Token Plan (`qwen3.7-max`/`flash`), OpenAI (`gpt-4-turbo`/`gpt-4o-mini`), and any OpenAI-compatible proxy (Ollama via shim) — see [`COMPATIBILITY.md`](COMPATIBILITY.md) for minimum versions, model tables, and breaking-change history.
- **Switching providers:** see [`MIGRATION.md`](MIGRATION.md) (QwenCloud ↔ OpenAI ↔ local). No code change needed — update `.env`.
- **Deployed app outdated?** Check the startup warning: `GET {base_url}/models` is queried; missing `LLM_MODEL_*` values produce a non-blocking sidebar warning linking to `COMPATIBILITY.md`.

## Security notes

- **Streamlit hardening:** `.streamlit/config.toml` (committed) sets XSRF, toolbar, and `maxUploadSize`; missed config triggers a startup warning (see Production hardening above).
- **Secrets are never committed.** `.env`, `.env.local`, `.env.*.local`, and `.streamlit/secrets.toml` are git-ignored (see `SECURITY.md`). Rotate keys after any leak.
- **Pre-commit guard.** `scripts/hooks/pre-commit` rejects staged `.env` files, `*.pem`/`*.key`, and key assignments (`OPENAI_API_KEY=`, `sk-*`). Install with `cp scripts/hooks/pre-commit .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit`.
- Medical records stay local: they are only sent to the configured LLM endpoint. VA.gov credentials and session tokens are never written to disk, `.env`, or logs.
- See [`SECURITY.md`](SECURITY.md) for full secrets management guidance (local `.env.local` overrides, CI/CD with GitHub Secrets, managed secret stores in production).
