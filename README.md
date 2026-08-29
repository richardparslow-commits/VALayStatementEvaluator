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
  llm.py                  OpenAI-compatible client (retry, JSON parsing)
  documents.py            TXT/MD/DOCX/PDF extraction, page-aware chunking
  medical_review.py       Exhaustive chunked record review -> fact digest
  evaluate.py             Claim extraction -> verification -> rubric scoring -> topic coverage
                          audit -> improvement suggestions & proposed rewrite -> report
  draft.py                Grounding + topic coverage -> draft -> self-review pipeline
  knowledge/              legal_framework.md, evaluation_rubric.md, drafting_guide.md,
                          topic_checklist.md
scripts/
  extract_pdfs.py         Build reference_docs/extracted/*.txt from source PDFs
  smoke_test.py           End-to-end pipeline test against the live LLM endpoint
  scale_sim.py            Offline 2,000-page pipeline simulation (no API calls)
tests/                    Offline unit tests (no API key required)
examples/                 Fictional sample statement + sample medical records
```

Long documents are processed in overlapping, page-labelled chunks so reviews are exhaustive
regardless of record length. See **Large record sets** below for how very large files scale.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # then put your API key in .env (never commit .env)
```

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
| `FETCH_SANDBOX_API_KEY` | Optional Fetch Sandbox API key | empty |
| `FETCH_SANDBOX_BASE_URL` | Fetch Sandbox base URL (`fetchsandbox.com` or subdomain) | `https://fetchsandbox.com` |
| `FETCH_SANDBOX_RECORDS_PATH` | GET path for the records endpoint | `/medical_records/{patient_id}` |
| `VA_LSE_ALLOW_LOCAL_PATHS` | Force-enable the local folder/file record source (`1`) even when the local-run check can't detect localhost | (auto) |
| `VA_LSE_CREDITS_PER_1M_MAIN` | Approx credits per 1M tokens for the main model (enables the credit-burn gauge) | (unset — gauge shows tokens/calls only) |
| `VA_LSE_CREDITS_PER_1M_FAST` | Approx credits per 1M tokens for the fast model (enables the credit-burn gauge) | (unset — gauge shows tokens/calls only) |
| `VA_LSE_CREDIT_QUOTA` | Your plan's weekly credit quota, used to render %-of-quota burn | `2500` |

All settings can also be overridden live in the app sidebar. Model availability depends on your
gateway workspace; check `GET {base_url}/models`.

### Estimating API usage & credit burn

Every run shows a live usage line in the progress caption and, after completion, an expandable
"Estimated API usage" table in the results with **calls, input, and output tokens per phase**
(record digest/merge/summary, claims, verification, rubric, topic audit, revision — or grounding/
draft/review on the Draft tab). Token counts are estimates based on prompt length and model
output, using the provider's reported usage when the endpoint supplies it.

Because QwenCloud Token Plan doesn't publish a fixed credits-per-1M-token rate, the gauge only
shows a **credit burn vs. your weekly quota** after you set `VA_LSE_CREDITS_PER_1M_MAIN` and
`VA_LSE_CREDITS_PER_1M_FAST` to your plan's effective rates (and `VA_LSE_CREDIT_QUOTA` if your
quota differs from the 2,500-credit Lite default). Users on higher tiers can override the same
knobs; the estimator never limits a run, it only reports.

Fetch Sandbox settings can also be overridden in the sidebar. Because Fetch Sandbox mirrors
your own OpenAPI spec, you must point `FETCH_SANDBOX_RECORDS_PATH` at the GET endpoint your
sandbox exposes for record retrieval.

## Run

```bash
streamlit run run_app.py
```

### Evaluate a statement

1. Upload or paste the lay statement.
2. Choose a medical-record source:
   - **Upload files** (PDF/TXT/MD/DOCX, multiple files OK),
   - **Fetch Sandbox** (enter a patient or record ID and import from your sandbox endpoint), or
   - **Local folder / file** (local runs only — read records straight from a path on this
     machine, e.g. `~/Desktop/ClaimRecords`; hidden when the app is served remotely).
3. Click **Run exhaustive evaluation** — watch chunked record review, claim verification,
   rubric scoring, improvement drafting, and report generation progress.
4. Review the verdict table (✅ supported / 🟡 partial / ❌ contradicted / ⚪ not found),
   scores, and the prioritized improvement plan.
5. Review the **proposed rewrite**: a change-by-change table (original → suggested → why),
   the revised statement with `[Confirm: ...]` placeholders, and downloads for both the
   report and the revised statement.

### Draft a statement

1. Choose the veteran's medical-record source: upload files, import from Fetch Sandbox, or
   read from a local folder/file path (local runs only).
2. Enter witness details and bulleted firsthand observations.
3. Click **Draft the statement** — the app grounds every observation in the records, flags
   conflicts, suggests strengthening questions, drafts the statement, and self-reviews it.
4. Resolve every bracketed `[Confirm: ...]` placeholder with the witness before signing.
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
python -m unittest discover -s tests -v        # offline unit tests
python scripts/smoke_test.py all               # live end-to-end (needs valid .env)
```

> GitHub Actions runs the offline tests and scale simulation automatically on every push to
> `main` (and on pull requests). The live smoke test is triggered **manually** from the
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

## Security notes

- `.env`, `.venv/`, and `outputs/` are git-ignored.
- Medical records stay local: they are only sent to the configured LLM endpoint.
