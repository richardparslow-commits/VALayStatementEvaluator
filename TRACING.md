# Distributed tracing (OpenTelemetry)

Structured logs (`README.md → Structured logging`) tell you *what* happened on
one pod, with a `request_id` to stitch the lines together. The profiler
(`PERFORMANCE.md`) tells you the phase breakdown of a run, but only as log text
you have to parse yourself, and only per process.

Tracing answers the two questions neither can: **which phase was slow for this
particular run** (span structure, with start/end times rather than totals), and
**what the whole run looked like when it crossed a process boundary** — in
Pattern C the digest runs on a worker pod, and a log-derived view of a run stops
at the queue.

Tracing is **off by default** and every code path is a no-op when the
OpenTelemetry packages are missing, so nothing in the default install changed.

## Enabling it

```bash
pip install -r requirements-otel.txt

export VA_LSE_TRACING=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318   # your collector

streamlit run run_app.py          # web tier
python -m app.worker              # worker tier (Pattern C) — same env
```

Both tiers must have `VA_LSE_TRACING=1`: the web pod opens the trace and the
worker continues it. A worker without the flag starts a fresh trace per job.

Verify:

```bash
curl -s localhost:8001/health | python -m json.tool | grep -A 12 '"tracing"'
curl -s localhost:8002/health | python -m json.tool | grep -A 12 '"tracing"'
```

`active: true` means the SDK started and spans are being exported. An empty
`reason` means nothing is wrong; anything else says exactly why tracing is off
(packages missing, flag unset, `OTEL_SDK_DISABLED`, or a setup failure).

## Configuring a backend

The exporter speaks **OTLP over HTTP**, so any OTLP-capable backend works with
no code change and no vendor SDK. Only the standard OpenTelemetry variables are
involved, which means each vendor's own OpenTelemetry documentation applies
verbatim — the table below is the short version.

| Backend | `OTEL_EXPORTER_OTLP_ENDPOINT` | Extra |
|---|---|---|
| **Jaeger all-in-one** (self-hosted) | `http://jaeger:4318` | nothing — Jaeger ingests OTLP natively |
| **Grafana Tempo / OpenTelemetry Collector** | `http://otel-collector:4318` | whatever the collector pipeline needs |
| **Datadog** | your Datadog Agent's OTLP intake (e.g. `http://datadog-agent:4318`) | the Agent must have OTLP intake enabled; point it at the Datadog site with your API key |
| **New Relic** | `https://otlp.nr-data.net` | `OTEL_EXPORTER_OTLP_HEADERS=api-key=<INGEST_LICENSE_KEY>` |
| **Honeycomb** | `https://api.honeycomb.io` | `OTEL_EXPORTER_OTLP_HEADERS=x-honeycomb-team=<API_KEY>` |

```bash
# Local smoke test: Jaeger with a UI on :16686
docker run --rm -p 16686:16686 -p 4318:4318 jaegertracing/all-in-one:latest
VA_LSE_TRACING=1 OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
  VA_LSE_TRACE_CHUNK_SPANS=0 streamlit run run_app.py
# run one Evaluate, then open http://localhost:16686 and search service
# va-lay-statement-evaluator
```

Kubernetes: set the variables in the same secret the app already reads
(`va-lse-env`) and add the collector's Service name as the endpoint, e.g.
`OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector.observability:4318`. No
manifest change is needed — nothing in the app requires sidecars.

`OTEL_TRACES_EXPORTER` is honoured for two special values:

| Value | Effect |
|---|---|
| `otlp` (default) | OTLP over HTTP |
| `console` | spans printed to stderr — the fastest way to confirm instrumentation before wiring a backend |
| `none` | spans are created and dropped (measures instrumentation overhead) |

## What gets traced

| Span | Where | Notes |
|---|---|---|
| `run:evaluate` / `run:draft` | `app/evaluate.py`, `app/draft.py` | root span for the run: `files`, `pages`, `chars` |
| `queue:submit` | `app/views/job_runner.py` | web pod only (Pattern C): `kind`, `files`, `pages`, `backend` |
| `records:review`, `records:merge`, `records:summary` | `app/medical_review.py` | digest phases; `facts`/`files` where known |
| `records:digest` | `app/medical_review.py` | the parallel fan-out: `chunks`, `concurrency`, `pages` (plus `retry=true` on the retry round) |
| `records:digest.chunk` | `app/medical_review.py` | **opt-in** (`VA_LSE_TRACE_CHUNK_SPANS=1`): one per chunk, `chunk=<index>` |
| `claims`, `verify`, `rubric`, `topic`, `revision`, `score`, `report` | `app/evaluate.py` | one per Evaluate phase |
| `grounding`, `draft`, `review` | `app/draft.py` | one per Draft phase |
| `llm:<phase>` | `app/llm.py` | **opt-in** (`VA_LSE_TRACE_LLM_CALLS=1`): one per provider call, `model`, `attempt` |

Span names are deliberately the same strings that appear as `phase=` in the
structured log, so a span and its log lines share a vocabulary (`claims` is
`claims` in both), and every span carries `request.id` when a run has one — that
is the join key back to the logs.

In Pattern C the trace reads:

```
queue:submit (web pod)
└── run:evaluate (worker pod)
    ├── records:review
    │   └── records:digest
    ├── claims
    ├── verify
    ├── rubric
    └── …
```

Two deliberate omissions:

- **No span per Streamlit script run.** Streamlit re-executes the script on
  every widget event, so a per-run span would bury the trace list under
  millisecond spans and there is no request object to attach one to. The spans
  that matter start when a pipeline starts.
- **No span per record chunk or per LLM call by default.** A 5,000-page bundle is
  hundreds of chunks with one LLM call each; enabling either turns one trace into
  thousands of spans and, on a metered APM, into a bill. The fan-out is visible
  as a single `records:digest` span carrying `chunks`/`concurrency`, which is
  enough to attribute the phase to endpoint latency rather than to the pipeline.

## Sampling

| Setting | Effect |
|---|---|
| `VA_LSE_TRACE_SAMPLE_RATIO=1.0` (default) | every run is traced |
| `VA_LSE_TRACE_SAMPLE_RATIO=0.1` | ~1 in 10 runs, plus **all** runs inherited from a sampled parent |
| `OTEL_TRACES_SAMPLER=parentbased_traceidratio` + `OTEL_TRACES_SAMPLER_ARG=0.1` | the standard OTel spellings (`always_on`, `always_off`, `traceidratio`, `parentbased_*`) are recognised and take precedence |

Sampling is parent-based, so a sampled run keeps its whole span tree: you never
get the middle of a trace without its root. The ratio is applied per process, so
in Pattern C the worker inherits the web pod's decision (the propagated context
carries the sampled flag) — a run is traced or it is not.

Runs are long (minutes) and few, so the ratio is about cost and storage rather
than noise. A single 2,000-page Evaluate produces roughly a dozen spans with the
defaults.

## PII and data egress

This app handles medical records and veterans' statements, so tracing follows the
same rule as logging: **no free text leaves in a span**.

- Only phase names, counts, sizes, byte lengths, model names, job ids, error
  classes, and the run's `request_id` (a random hex id) are attached.
- Span attribute names that look like free text are dropped before the SDK sees
  them — `statement`, `statement_text`, `observations`, `prompt`, `content`,
  `witness`, `records`, and anything ending in `_text`. That guard exists so a
  future call site cannot leak by accident, and there is a test for it
  (`tests/test_tracing.py → TestPiiScreening`).
- A failing phase records the exception object (message plus stack) as a span
  event, which is the same text the app already writes to its own error logs and
  renders to the user; the attribute set stays to the exception class.

That said, **enabling tracing sends data to a third party.** Span metadata about
a claim — phase timings, page counts, model names, and the fact that a run
happened at all — is metadata you may not want a US-hosted SaaS to hold if your
deployment is PHI-sensitive. For those deployments, run a self-hosted collector
(Jaeger, Grafana Tempo, or an OpenTelemetry Collector writing to your own
storage) and keep `OTEL_EXPORTER_OTLP_ENDPOINT` inside your network. Choosing a
vendor is a compliance decision, not a technical one.

## Overhead and shutdown

- Span creation is a no-op when tracing is off, so the default path pays nothing.
- With tracing on, the cost is a span per phase plus one batched OTLP POST per
  batch interval; the exporter runs on its own thread and never blocks a run.
- Spans are buffered and flushed on graceful shutdown (`SIGTERM`), after the
  in-flight run drains — the trace of the run that just finished is exported
  rather than lost with the process. `python -m app.worker --once` flushes on
  exit too.
- A collector that is down or misconfigured does **not** fail runs: the exporter
  logs a warning and drops batches. `GET /health → tracing` still reports
  `active: true` in that case, because it reports configuration rather than
  reachability (the health probe must never block on a network round trip).

## Troubleshooting

| Symptom | Check |
|---|---|
| No spans in the backend | `curl :8001/health \| grep -A12 tracing` — look at `active` and `reason` |
| `reason: opentelemetry packages not importable` | `pip install -r requirements-otel.txt` |
| `reason: disabled (set VA_LSE_TRACING=1…)` | the flag is unset in **this** process (Pattern C: the worker needs it too) |
| Web spans arrive, worker spans do not | the worker has no `VA_LSE_TRACING=1`, or it has a different `OTEL_EXPORTER_OTLP_ENDPOINT` |
| Traces from web and worker look like two separate traces | the payload's trace context was dropped — check that both tiers run the same app version (the envelope key is additive) |
| Traces stop at `queue:submit` | the job failed before the worker started it; check `VA_LSE_JOB_QUEUE` on the web tier and the worker's logs |
| Spans missing for the last run before a restart | the flush happens after the drain — check `VA_LSE_SHUTDOWN_GRACE_SECONDS` gave the run time to finish |
