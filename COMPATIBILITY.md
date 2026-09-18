# Compatibility — LLM endpoints, models, and versions

This file records what the app has been tested against, what it works with,
and what breaks when you change providers or models. Keep it updated when
defaults in `app/config.py` change or when providers deprecate APIs.

> **Rule of thumb:** This app works with **any OpenAI-compatible**
> `POST {base_url}/chat/completions` endpoint. The Perplexity Router values are
> the shipped defaults; the QwenCloud Token Plan values were the previous default
> and still work. See `MIGRATION.md` for how to switch.

## Current defaults (`app/config.py`)

| Setting | Default value | Purpose |
|---------|---------------|---------|
| `DEFAULT_BASE_URL` | `https://api.perplexity.ai/router/v1` | Perplexity **Router API** — OpenAI Chat Completions schema, drop-in via base URL + key. In private preview (request access from api@perplexity.ai); the catalog is also an allowlist, so an unlisted model id is a 400 |
| `DEFAULT_MODEL_MAIN` | `perplexity/kimi-k3` | Low-volume heavy calls: claim extraction, verification, rubric, topic, rewrite — the strongest model in the Router catalog ($3 / $15 per 1M) |
| `DEFAULT_MODEL_FAST` | `perplexity/glm-5.3-flash` | High-volume bulk calls: record-chunk digest and fact-merge — cheapest in the catalog ($0.15 / $0.50; cache reads $0.03) |
| `DEFAULT_FETCH_SANDBOX_BASE_URL` | `https://fetchsandbox.com` | Fetch Sandbox host allowlist — change only for your own sandbox subdomain |

Override any of the above via `OPENAI_API_KEY` / `OPENAI_BASE_URL` /
`LLM_MODEL_MAIN` / `LLM_MODEL_FAST` in `.env` or the sidebar.

## Tested endpoints

| Provider / setup | Base URL pattern | Auth | Status | Min version / notes |
|------------------|------------------|------|--------|---------------------|
| **Perplexity Router API** | `https://api.perplexity.ai/router/v1` | Perplexity key (`pplx-...`) | **Primary / recommended (default)** | OpenAI Chat Completions schema; also serves the Anthropic Messages schema at `/router/v1/messages`. **Private preview** — request access from api@perplexity.ai. The same key serves the Agent API on the Research tab (and is read automatically from `OPENAI_API_KEY` when this is the base URL) |
| **QwenCloud Token Plan (MaaS)** | `https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1` | `sk-sp-...` (Token Plan key) | Tested (previous default) | OpenAI API v1 compatible; key+URL must be paired (Token Plan keys fail against the general gateway) |
| **OpenAI** | `https://api.openai.com/v1` | `sk-proj-...` or `sk-...` | Tested | `openai` Python SDK `>=1.0`; works with `gpt-4o`, `gpt-4-turbo`, `gpt-4o-mini`, etc. |
| **Azure OpenAI** | `https://{resource}.openai.azure.com/openai/deployments/{deployment}/` | Azure key / Entra | Compatible when fronted with an OpenAI-compat proxy or `base_url` pointing at the proxy | Requires API version header on the proxy |
| **Local Ollama (OpenAI-compat)** | `http://localhost:11434/v1` (Ollama proxy) | none / stub | Compatible for dev | Requires an OpenAI-compat shim (e.g. `openai` proxy mode); not benchmarked for quality |

### What any provider must support

- `POST {base_url}/chat/completions` with `model`, `messages`, `temperature`, `max_tokens`
- `GET {base_url}/models` (optional; used for the startup availability warning)
- JSON-mode helper appends `Respond with ONLY valid JSON` to the system prompt — the model must obey JSON output (all modern instruction models do)
- Non-streaming responses with `response.choices[0].message.content` and optional `response.usage.prompt_tokens / completion_tokens`

If your provider deviates, add a compatibility shim (proxy) rather than forking the app.

## Tested models per endpoint

| Endpoint | Main (heavy) | Fast (bulk) | Notes |
|----------|--------------|-------------|-------|
| Perplexity Router API | `perplexity/kimi-k3` (default) | `perplexity/glm-5.3-flash` (default) | Full catalog: `perplexity/kimi-k3`, `perplexity/glm-5.3`, `perplexity/glm-5.3-flash`, `perplexity/nemotron-3-ultra-550b-a55b`. Step the fast model up the ladder (`nemotron-3-ultra-550b-a55b`, then `glm-5.3`) if extraction quality needs it |
| QwenCloud Token Plan | `qwen3.7-max` | `qwen3.7-flash` | Tuned for Individual Plan Lite (2500 credits / 7 days, 1–2 concurrent agents, `qwen3.7-flash` saves credit) |
| OpenAI | `gpt-4-turbo`, `gpt-4o` | `gpt-4o-mini` | Any reasoning-capable model works for main; use a cheaper model for fast |
| Ollama | provider-dependent | provider-dependent | Quality not evaluated; prefer at least a 7B instruction model |

The split (`model_main` vs `model_fast`) is the single largest credit/cost saver
on large record sets (1000+ pages = hundreds of chunk-digest calls). Do not point
both at the expensive model unless you accept the cost.

## Breaking changes by app version

| App version / commit | Change | Impact |
|----------------------|--------|--------|
| `main` @ 2026-09-18 (`DEFAULT_BASE_URL` → Router API, `model_{main,fast}` → `perplexity/*`) | Defaults moved from QwenCloud Token Plan to Perplexity's Router API, with the fast/main split mapped onto that catalog | **An existing `.env` is unaffected** — environment and sidebar values override every default, so a deployment that names its endpoint keeps running exactly as before. Only a fresh clone with no `.env` changes behaviour. When the base URL is Perplexity's, `OPENAI_API_KEY` is reused as the Agent API key, so one key serves both the chat endpoint and the Research tab |
| `main` @ 2026-09-14 (`DEFAULT_BASE_URL` → Token Plan, `model_{main,fast}` → `qwen3.7-*`) | Defaults moved from generic OpenAI to QwenCloud Token Plan | Existing `.env` pointing at OpenAI continues to work (env overrides defaults). New clones without `.env` defaulted to QwenCloud — set `OPENAI_BASE_URL` explicitly if you mean OpenAI. |
| Prompt injection hardening (`app/prompt_sanitize.py`) | User/record text is escaped before prompt interpolation; `>>>/<<</``` sanitized; guard note added | No API break; model output may be marginally different (safer). |
| `EVALUATE_INTERNAL_MAX_CHARS` / `DRAFT_INTERNAL_MAX_CHARS` hardened to `80_000` | Long statements/observations above 80K are truncated with a warning banner | Bypass callers that previously sent >80K now see truncation; split inputs. |

Add a new row here whenever defaults, prompt templates, or API contracts change.

## Running two endpoints at once (failover)

The app can hold **two** endpoint configurations at the same time: the primary
(`OPENAI_BASE_URL`) and an optional fallback (`OPENAI_BASE_URL_FALLBACK`). They are
usually *different providers*, which is why the fallback has its own key and model
names:

| Role | Primary | Fallback |
|------|---------|----------|
| Endpoint | `OPENAI_BASE_URL` | `OPENAI_BASE_URL_FALLBACK` |
| Credentials | `OPENAI_API_KEY` | `OPENAI_API_KEY_FALLBACK` (default: the primary's) |
| Heavy calls | `LLM_MODEL_MAIN` | `LLM_MODEL_MAIN_FALLBACK` (default: the primary's) |
| Bulk calls | `LLM_MODEL_FAST` | `LLM_MODEL_FAST_FALLBACK` (default: the primary's) |

**A key is provider-specific, and so are model names.** The documented example —
OpenAI as the backup for a QwenCloud primary — needs all four fallback values;
`OPENAI_BASE_URL_FALLBACK` alone can only reach a second gateway that accepts the
*same* key and serves the *same* model names (a second region, or a proxy in front
of the same account).

Both models of the fallback are checked against its `/models` list by the
readiness probe, exactly as the primary's are, and it is only probed when the
primary fails — so a healthy deployment still costs one round trip.

Things worth knowing before you point the two at different vendors:

- **The backup may write differently.** These are legal work products. Output from
gpt-4-turbo is not byte-identical to output from qwen3.7-max, and the prompt
tuning in this repo (including the QwenCloud moderation-nudge handling) targets the
configured model. A run served by the fallback is stamped `llm_endpoints` in the
audit record and the run log so the distinction is recoverable afterwards.
- **Model names are not interchangeable across providers.** The two *roles*
  (heavy/bulk) are mapped onto the fallback's names — see
  `LLMClient._resolve_model`. A caller that names some other model gets that name
  passed through unchanged, because guessing an equivalent on another vendor would
  be guesswork.
- **Costs differ.** The fallback bills at its own rates, and
  `VA_LSE_CREDITS_PER_1M_*` describes the *primary*. A long failover window is a
  real (if small) budget event; `va_lse_llm_failover_active` tells you it is
  happening.
- **Rate limits differ.** The concurrency limiter
  (`VA_LSE_MAX_CONCURRENT_LLM_CALLS`) is shared by both endpoints, so it must fit
  whichever of the two has the tighter limit.

## Detecting an outdated or unsupported configuration

On every app launch the sidebar runs a best-effort **model availability check**:

- `GET {base_url}/models` with the configured API key
- If `model_main` or `model_fast` is missing from the provider's model list, the UI shows a non-blocking warning linking to this file and `MIGRATION.md`
- Network/permission failures are ignored — the app always remains usable (warnings only).

See `app/llm.py` (`check_model_availability`) and `app/main.py:_sidebar_settings`.

## What to do when QwenCloud (or any provider) changes their API

1. Confirm the new `base_url` and required key prefix with the provider.
2. Update `.env` (`OPENAI_BASE_URL`, key) or use the sidebar **Apply settings** — no code change needed for a pure endpoint move.
3. If the provider changes the models, update `LLM_MODEL_MAIN` / `LLM_MODEL_FAST` in `.env`. Check this file and `MIGRATION.md` for per-provider suggestions.
4. If the provider changes request/response shape (rare for OpenAI-compat), file an issue and note it here under Breaking changes.
