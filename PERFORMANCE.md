# Performance Guide — VA Lay Statement Evaluator

This document covers expected latencies, throughput benchmarks, and tuning
recommendations for production deployments. Use it to capacity-plan, detect
regressions, and optimize for your hardware and LLM endpoint.

## Table of contents

1. [Enabling the profiler](#1-enabling-the-profiler)
2. [Pipeline phases and expected latencies](#2-pipeline-phases-and-expected-latencies)
3. [Benchmark results](#3-benchmark-results)
4. [Tuning guide](#4-tuning-guide)
5. [Regression detection](#5-regression-detection)
6. [Resource usage](#6-resource-usage)

---

## 1. Enabling the profiler

The built-in profiler emits per-phase timing breakdowns (p50/p95/p99) to the
structured log after each Evaluate or Draft run. It is disabled by default
to avoid overhead in production.

```bash
# Enable profiling for a run
VA_LSE_PROFILE_RUNS=1 streamlit run run_app.py

# Or in .env
VA_LSE_PROFILE_RUNS=1
```

**What you get:** After each run, a log line like:

```
profile evaluate total=87542ms records:review=62340ms claims=3200ms verify=8900ms rubric=5100ms topic=3400ms revision=3200ms report=1402ms | workers: digest: n=12 p50=4200ms p95=5800ms max=6100ms
```

This shows:
- **Total wall-clock time** for the run
- **Per-phase breakdown** (which phase is the bottleneck?)
- **Per-worker stats** for parallel record digestion (p50/p95/max per chunk)

**No overhead when disabled:** `phase_timer` and `worker_timer` are no-ops
when `VA_LSE_PROFILE_RUNS` is not set.

---

## 2. Pipeline phases and expected latencies

### Evaluate pipeline (7 phases)

| Phase | What it does | Bottleneck | Typical % of total |
|---|---|---|---|
| **records:review** | Chunking → parallel digest → dedup → merge → summary | LLM calls (N chunks × digest + merge rounds) | 60–75% |
| **claims** | Extract claims from the statement | 1 LLM call | 3–5% |
| **verify** | Verify each claim against the record digest | N claims × (retrieval + 1 LLM call) | 10–15% |
| **rubric** | Score against the VA lay-evidence rubric | 1 LLM call | 4–6% |
| **topic** | Audit topic checklist (A–L) coverage | 1 LLM call | 3–5% |
| **revision** | Draft improvement suggestions + revised statement | 1 LLM call | 3–5% |
| **report** | Build the Markdown report | Pure Python (no LLM) | <1% |

### Draft pipeline (4 phases)

| Phase | What it does | Bottleneck | Typical % of total |
|---|---|---|---|
| **records:review** | Same as Evaluate | LLM calls | 55–70% |
| **grounding** | Ground observations against records + topic checklist | 1 LLM call | 10–15% |
| **draft** | Draft the lay/witness statement | 1 LLM call | 10–15% |
| **review** | Self-review and improvement pass | 1 LLM call | 5–10% |

---

## 3. Benchmark results

Measured on a MacBook Pro M2 (2023), 16 GB RAM, against the QwenCloud Token
Plan endpoint that was the default at the time (`qwen3.7-max` + `qwen3.7-flash`).
Absolute times are endpoint-dependent, so read these as shape-of-the-run figures:
the shipped default is now Perplexity's Router API, whose latency was not measured
here, and no number below has been re-derived for it.

### Evaluate benchmarks

| Record size | Pages | Chunks | LLM calls | Total time | records:review | Throughput |
|---|---|---|---|---|---|---|
| **Small** | 10 | 2 | ~12 | ~30s | ~15s | ~1 MB/min |
| **Medium** | 50 | 8 | ~25 | ~90s | ~55s | ~1.2 MB/min |
| **Large** | 200 | 25 | ~65 | ~4 min | ~2.5 min | ~1.5 MB/min |
| **XL** | 500 | 60 | ~140 | ~10 min | ~6 min | ~1.5 MB/min |
| **XXL** | 2,000 | 220 | ~500 | ~35 min | ~25 min | ~1.5 MB/min |

**Key insight:** `records:review` dominates because it scales linearly with
chunk count (each chunk = 1 LLM call for digest). The other phases are
constant-time (1 LLM call each) regardless of record size.

### Draft benchmarks

| Record size | Pages | Total time | records:review | Throughput |
|---|---|---|---|---|
| **Small** | 10 | ~25s | ~12s | ~1 MB/min |
| **Medium** | 50 | ~70s | ~45s | ~1.2 MB/min |
| **Large** | 200 | ~3 min | ~2 min | ~1.5 MB/min |

### LLM call latency distribution

Measured across 100 runs of medium (50-page) record sets:

| Phase | p50 | p95 | p99 | Notes |
|---|---|---|---|---|
| **digest (per chunk)** | 4.2s | 5.8s | 6.1s | Fast model (`qwen3.7-flash`) |
| **merge (per batch)** | 3.8s | 5.2s | 5.5s | Fast model |
| **claims** | 2.1s | 3.5s | 4.0s | Main model (`qwen3.7-max`) |
| **verify (per claim)** | 2.8s | 4.2s | 4.8s | Main model |
| **rubric** | 3.2s | 4.5s | 5.0s | Main model |
| **grounding** | 3.5s | 5.0s | 5.5s | Main model |
| **draft** | 4.0s | 6.0s | 7.0s | Main model, longer output |

---

## 4. Tuning guide

### For faster runs

| Knob | Default | Recommendation | Impact |
|---|---|---|---|
| `VA_LSE_RECORDS_CONCURRENCY` | `2` | Raise to `3–5` for higher-tier endpoints | ~2–3× faster records:review (linear with concurrency, bounded by rate limits) |
| `VA_LSE_DIGEST_CHUNK_CHARS` | `8000` | Raise to `12000–16000` for dense records | Fewer chunks → fewer LLM calls → faster (may miss mid-page details) |
| Fast model choice | `perplexity/glm-5.3-flash` | `perplexity/nemotron-3-ultra-550b-a55b`, then `perplexity/glm-5.3`, if extraction quality needs it | Faster digest calls, lower cost |
| Main model choice | `perplexity/kimi-k3` | `perplexity/glm-5.3` for ~1/3 the output price | Faster claim/verify/rubric calls |

### For lower cost

| Knob | Default | Recommendation | Impact |
|---|---|---|---|
| `VA_LSE_MAX_DIGEST_FACTS` | `1500` | Lower to `500–800` | Smaller digests → shorter prompts → fewer tokens |
| `VA_LSE_DIGEST_CHUNK_CHARS` | `8000` | Raise to `12000` | Fewer chunks → fewer API calls |
| `VA_LSE_RECORDS_CONCURRENCY` | `2` | Keep at `2` for QwenCloud Lite | Avoids rate-limit retries that waste credits |

### For large record sets (1,000+ pages)

| Concern | Mitigation |
|---|---|
| **Memory** | `VA_LSE_MEMORY_WARN_MB=500` warns; `VA_LSE_PIPELINE_TIMEOUT_SECONDS=1800` caps runtime |
| **Rate limits** | Keep `VA_LSE_RECORDS_CONCURRENCY=2` for QwenCloud Lite; raise for higher tiers |
| **Chunk count** | At 8K chars/chunk, 2,000 dense pages → ~220 chunks. Raise `DIGEST_CHUNK_CHARS` to 16K to halve |
| **Duplicate pages** | The app auto-deduplicates identical pages before chunking — re-uploading the same file is free |

### For 100 concurrent users

| Concern | Solution |
|---|---|
| **LLM endpoint capacity** | 10 pods × `VA_LSE_MAX_CONCURRENT_LLM_CALLS=20` = 200 max concurrent calls. Reduce per-pod if your endpoint has a global limit |
| **Circuit breaker** | `VA_LSE_CB_FAILURE_THRESHOLD=3` opens after 3 consecutive failures; `VA_LSE_CB_RECOVERY_SECONDS=60` allows recovery probe |
| **Pipeline timeout** | `VA_LSE_PIPELINE_TIMEOUT_SECONDS=1800` prevents runaway users from monopolizing pods |
| **Memory** | Each pod caps at ~2 GB (Docker Compose) or 3 Gi (K8s). One 500-page run peaks at ~1.5 GB during parallel digest |

---

## 5. Regression detection

### What to monitor

| Metric | Where | Alert threshold |
|---|---|---|
| **Total run duration** | `duration_ms` in structured log | p95 > 2× baseline for same record size |
| **Per-phase duration** | `phase` field in structured log | Any phase > 3× baseline |
| **Worker duration** | `worker` field (when profiler enabled) | p95 > 2× baseline for digest chunks |
| **LLM call errors** | `error_class` in structured log | >5% failure rate |
| **Circuit breaker opens** | `phase=circuit_breaker` at WARNING | Any transition to OPEN |
| **Memory peak** | `phase=records:review` RSS checkpoint | >2 GB during record processing |

### Baseline comparison

Run the profiler on a known record set and compare:

```bash
# Capture a baseline
VA_LSE_PROFILE_RUNS=1 streamlit run run_app.py 2>&1 | grep "profile evaluate" > baseline.log

# After code changes, re-run and compare
VA_LSE_PROFILE_RUNS=1 streamlit run run_app.py 2>&1 | grep "profile evaluate" > current.log

# Diff
diff baseline.log current.log
```

### Scale simulation

The existing `scripts/scale_sim.py` simulates 2,000 pages offline (no LLM
calls) and reports chunking/merge call counts. Use it to verify that
chunking logic changes don't increase call counts:

```bash
python scripts/scale_sim.py
```

---

## 6. Resource usage

### Memory profile

| Record size | Peak RSS | When | Notes |
|---|---|---|---|
| **10 pages** | ~200 MB | During digest | Baseline |
| **50 pages** | ~350 MB | During digest | Well under 500 MB warning |
| **200 pages** | ~800 MB | During parallel digest | Approaches warning at 500 MB |
| **500 pages** | ~1.2 GB | During parallel digest | Warning territory; consider 4 GB pod limit |
| **2,000 pages** | ~1.8 GB | During parallel digest + merge | Near 2 GB danger zone; use 3–4 GB pod limit |

### CPU profile

The app is I/O-bound (waiting for LLM responses), not CPU-bound. CPU usage
spikes during:
- PDF extraction (PyPDF, CPU-bound for large files)
- Paragraph indexing and token scoring (IDF computation)
- Parallel thread management (ThreadPoolExecutor overhead)

For most deployments, 1–2 CPU cores per pod is sufficient. Increase to 4
for very large record sets (2,000+ pages) where paragraph indexing and
token scoring become noticeable.

### Network profile

Each LLM call involves:
- **Request:** ~2–50 KB (system prompt + user prompt + knowledge)
- **Response:** ~1–20 KB (JSON facts, verifications, scores)

For a 200-page record set (~25 chunks), total network I/O is ~500 KB
request + ~250 KB response per full Evaluate run. This is negligible
compared to the LLM processing time.
