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
| `DEFAULT_BASE_URL` | `https://api.perplexity.ai/v1` | Perplexity **Agent API** — OpenAI Responses schema at `{base_url}/responses` (the SDK alias of `/v1/agent`), served by `app/llm.py` automatically for Perplexity hosts; any Chat-Completions endpoint is likewise served its own schema. Multi-provider catalog (OpenAI, Anthropic, Google, xAI) under one key |
| `DEFAULT_MODEL_MAIN` | `perplexity/kimi-k3` | Low-volume heavy calls: claim extraction, verification, rubric, topic, rewrite — the strongest Perplexity-routed model in the Agent API catalog ($3 / $15 per 1M) |
| `DEFAULT_MODEL_FAST` | `perplexity/glm-5.3-flash` | High-volume bulk calls: record-chunk digest and fact-merge — cheapest in the catalog ($0.15 / $0.50; cache reads $0.03) |
| `DEFAULT_FETCH_SANDBOX_BASE_URL` | `https://fetchsandbox.com` | Fetch Sandbox host allowlist — change only for your own sandbox subdomain |

Override any of the above via `OPENAI_API_KEY` / `OPENAI_BASE_URL` /
`LLM_MODEL_MAIN` / `LLM_MODEL_FAST` in `.env` or the sidebar.

## Tested endpoints

| Provider / setup | Base URL pattern | Auth | Status | Min version / notes |
|------------------|------------------|------|--------|---------------------|
| **Perplexity Agent API** | `https://api.perplexity.ai/v1` | Perplexity key (`pplx-...`) | **Primary / recommended (default)** | OpenAI **Responses** schema at `{base_url}/responses` (`/v1/agent` is the same route; the SDK alias is what a `.../v1` base URL hits) — `app/llm.py` sends Responses automatically for Perplexity hosts and Chat Completions everywhere else, so an endpoint swap never needs a code change. Replaces the deprecated Sonar API. Multi-provider catalog (`openai/gpt-5.5`, `anthropic/claude-sonnet-4-6`, `google/gemini-3.5-flash`, `xai/grok-4.5`, ...) at first-party pricing, listed live at `GET {base_url}/models`. Runs are sent with `store: false`, so no retrievable copy of a run accumulates on the provider. The same key serves the Research tab (and is read automatically from `OPENAI_API_KEY` when this is the base URL). The Research tab's own call path is covered live, opt-in, by `tests/test_perplexity_live.py` (`VA_LSE_TEST_PERPLEXITY_KEY`) |
| **Perplexity Router API** (retired default) | `https://api.perplexity.ai/router/v1` | Perplexity key (`pplx-...`) | Not recommended — private preview | Chat Completions schema; calls are refused with `403 The Router API is currently in limited preview` unless the account was granted preview access. Still reachable: point `OPENAI_BASE_URL` at it and the app sends Chat Completions (a non-Perplexity hostname, e.g. a proxy, is required only because Perplexity hosts select Responses). The sidebar shows a warning banner when the base URL, or either model id, matches a retired provider (Router or Vercel AI Gateway) so a doomed configuration is named before a run starts. The finding follows the configuration into the audit trail — run-gate decisions in `logs/runs.jsonl` carry `retired_endpoint` (kind) and `retired_provider` (label) fields whenever it applies — and the run button warns when a run proceeds despite it. The banner also carries the repair: **Apply Perplexity defaults** sets the default base URL and model ids in one click, keeping a `pplx-`-shaped key and clearing anything else (session-only — an `.env` on disk is unchanged) |
| **Vercel AI Gateway** | `https://ai-gateway.vercel.sh/v1` | AI Gateway key (`vck_...`) | Tested (`tests/test_ai_gateway_live.py`, opt-in; see *Vercel credentials are not interchangeable*) | OpenAI Chat Completions schema, so nothing in `app/llm.py` changes. Model ids come from the gateway's **own catalog** (`owner/model`): the Router ids are not in it. One gateway key can serve the whole pipeline, but the Research tab still needs `PERPLEXITY_API_KEY` — it uses the Agent API, which the gateway does not serve |
| **QwenCloud Token Plan (MaaS)** | `https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1` | `sk-sp-...` (Token Plan key) | Tested (previous default) | OpenAI API v1 compatible; key+URL must be paired (Token Plan keys fail against the general gateway) |
| **OpenAI** | `https://api.openai.com/v1` | `sk-proj-...` or `sk-...` | Tested | `openai` Python SDK `>=1.0`; works with `gpt-4o`, `gpt-4-turbo`, `gpt-4o-mini`, etc. |
| **Azure OpenAI** | `https://{resource}.openai.azure.com/openai/deployments/{deployment}/` | Azure key / Entra | Compatible when fronted with an OpenAI-compat proxy or `base_url` pointing at the proxy | Requires API version header on the proxy |
| **Local Ollama (OpenAI-compat)** | `http://localhost:11434/v1` (Ollama proxy) | none / stub | Compatible for dev | Requires an OpenAI-compat shim (e.g. `openai` proxy mode); not benchmarked for quality |

### What any provider must support

- `POST {base_url}/chat/completions` with `model`, `messages`, `temperature`, `max_tokens`
- `GET {base_url}/models` (optional; used for the startup availability warning)
- JSON-mode helper appends `Respond with ONLY valid JSON` to the system prompt — the model must obey JSON output (all modern instruction models do)
- Non-streaming responses with `response.choices[0].message.content` and optional `response.usage.prompt_tokens / completion_tokens`
- Structured responses that are not truncated: a provider that stops generating below the requested `max_tokens` reports an **incomplete** run or hands back an unparseable document (measured on the configured Perplexity endpoint). Keep JSON schemas compact — a field that restates text already in the response is what pushes a long answer over that edge.

If your provider deviates, add a compatibility shim (proxy) rather than forking the app.

### Vercel credentials are not interchangeable

Three different things carry Vercel's name, and only the first one is an LLM key. Mixing
them up produces a 401/403 that looks like a provider outage, which is why the preflight
(`app/preflight.py`) names the mix-up and `scripts/vercel_sandbox_runner.py` refuses the
wrong one by name:

| Credential | Where it comes from | What it authenticates |
|---|---|---|
| **AI Gateway API key** (`vck_...`) | Vercel project → AI Gateway → API Keys | This app's LLM calls: `OPENAI_API_KEY` with `OPENAI_BASE_URL=https://ai-gateway.vercel.sh/v1`. Never a sandbox |
| **Access token** | Vercel Account Settings → Tokens (scoped to the team) | Vercel **Sandbox** (`VERCEL_TOKEN`, and the OCR runner in `DEPLOYMENT.md` §6). No LLM endpoint accepts it |
| **OIDC token** (`VERCEL_OIDC_TOKEN`) | Provisioned automatically inside a Vercel Function | Vercel **Sandbox** as well — Vercel's recommendation, because nothing long-lived has to be stored |

## Tested models per endpoint

| Endpoint | Main (heavy) | Fast (bulk) | Notes |
|----------|--------------|-------------|-------|
| Perplexity Router API | `perplexity/kimi-k3` (default) | `perplexity/glm-5.3-flash` (default) | Full catalog: `perplexity/kimi-k3`, `perplexity/glm-5.3`, `perplexity/glm-5.3-flash`, `perplexity/nemotron-3-ultra-550b-a55b`. Step the fast model up the ladder (`nemotron-3-ultra-550b-a55b`, then `glm-5.3`) if extraction quality needs it |
| Vercel AI Gateway | `moonshotai/kimi-k3`, `openai/gpt-4.1-nano` | `alibaba/qwen3.7-flash` | Measured against the live gateway: `GET /v1/models` lists 372 ids, and a chat plus a JSON-mode call complete through `app/llm.py`. Catalog ids are `owner/model`, so the Router ids are **not** listed. The account's tier decides what a key may call — a free-tier key is refused `moonshotai/kimi-k3` with `403 Free tier users do not have access to this model`, which is entitlement rather than incompatibility; check `GET /models` (or **Test connection**) before a run. A free-tier key is also **rate-limited per model**: measured, `openai/gpt-4.1-nano` answered four calls and then returned `429 Free tier requests on this model are rate-limited` through all three of the app's retries in under a second, so a run of hundreds of calls needs paid credits on that key — the wiring is fine, the tier is not |
| QwenCloud Token Plan | `qwen3.7-max` | `qwen3.7-flash` | Tuned for Individual Plan Lite (2500 credits / 7 days, 1–2 concurrent agents, `qwen3.7-flash` saves credit) |
| OpenAI | `gpt-4-turbo`, `gpt-4o` | `gpt-4o-mini` | Any reasoning-capable model works for main; use a cheaper model for fast |
| Ollama | provider-dependent | provider-dependent | Quality not evaluated; prefer at least a 7B instruction model |

The split (`model_main` vs `model_fast`) is the single largest credit/cost saver
on large record sets (1000+ pages = hundreds of chunk-digest calls). Do not point
both at the expensive model unless you accept the cost.

## Breaking changes by app version

| App version / commit | Change | Impact |
|----------------------|--------|--------|
| Lay-attribution rewrite rules and per-claim `basis` (this commit) | `CLAIMS_SYSTEM`/`CLAIMS_USER` now record *how the writer knows* each fact (`basis`: `experienced \| observed \| reported \| provider_statement \| conclusion`), `VERIFY_SYSTEM`/`RUBRIC_SYSTEM_TEMPLATE`/`TOPIC_SYSTEM_TEMPLATE` are tightened on the same authorities, and `REVISE_SYSTEM` requires every NOT FOUND claim to be rewritten with its basis of knowledge and never promoted into a diagnosis, cause or rating | Claims gain an optional `basis` key: saved results and job payloads from earlier versions load unchanged (an absent basis is recorded `unspecified` and is not rendered), and exports are unaffected. The revision response grows by one short `lay_attribution` entry per NOT FOUND claim; do not add a field that restates the revised text — that truncated the JSON at the endpoint's effective output ceiling (measured: an 8,911-character response, cut mid-string) |
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

The sidebar's **Test connection** button runs the same **preflight** the run buttons do — the
`GET {base_url}/models` listing *and* one real call per configured model — and shows the verdict
it produces, because it is the screen where the user can still fix it: the HTTP status and the
provider's response body distinguish a rejected key (`401`/`403`) from a wrong path in the base
URL (`404`) from a host that does not answer, and a listing that answers while every completion
is refused is caught by the real call instead of being reported as healthy. The verdict is
remembered for the run gate, so the first run after a check — and the run after that — reuses
it instead of repeating the same two requests. The run log keeps the receipt: each attempt
writes one line (`accepted`, or `rejected` when the check refuses) naming `endpoint_check` as
`fresh` or `reused` with the reused check's age.

See `app/preflight.py` (`check_endpoint`, the shared policy), `app/llm.py` (`probe_models` and
`probe_chat`) and `app/views/sidebar.py:render_sidebar_settings`.

### The same check, with teeth, at the moment a run starts

The launch check is advisory; the **preflight** is not. Pressing a run button probes twice and
refuses to start when the configuration cannot work: a configured model missing from the
endpoint's catalog, a key the endpoint rejects (`401`/`403`), or — the case a listing cannot
see — a **short chat call** the endpoint refuses (`401`/`403`, or a `404` that says the base URL
serves no completions path at all). Each means every call in the run would be rejected, and the
probes cost two requests instead of the first minutes of the bundle's chunks — or none when
that configuration was just checked (by **Test connection** or a recent run attempt), whose
verdict is reused for a few minutes — and a run that reuses a check says so where it
starts, with the age of the check behind it. The same provenance lands in the run log:
`accepted`/`rejected` events carry `endpoint_check` and the reused check's age.

The second probe is why this section exists in a document about compatibility. A listing is not
a promise: Perplexity's Router API publishes its ids and then answers every completion with
`403 The Router API is currently in limited preview`, so the listing alone reports a healthy
endpoint. That is an entitlement gate, not a missing model — which is precisely why it has to be
observed by calling, and why the verdict carries the provider's own message.

Nothing inconclusive blocks: an unreachable host, a `404` on `/models` (some
OpenAI-compatible servers serve completions without listing models), a `429`, and provider-side
`5xx` are reported and the run proceeds.
A block carries a per-configuration waiver for endpoints whose `/models` answer does not
describe what they serve — see `README.md → Before a run starts` and
`TROUBLESHOOTING.md → The run did not start`.

See `app/preflight.py` (policy, socket-free to test) and `app/views/shared.py`
(`check_endpoint_gate`, `render_endpoint_preflight_notice`).

## What to do when QwenCloud (or any provider) changes their API

1. Confirm the new `base_url` and required key prefix with the provider.
2. Update `.env` (`OPENAI_BASE_URL`, key) or use the sidebar **Apply settings** — no code change needed for a pure endpoint move.
3. If the provider changes the models, update `LLM_MODEL_MAIN` / `LLM_MODEL_FAST` in `.env`. Check this file and `MIGRATION.md` for per-provider suggestions.
4. If the provider changes request/response shape (rare for OpenAI-compat), file an issue and note it here under Breaking changes.
