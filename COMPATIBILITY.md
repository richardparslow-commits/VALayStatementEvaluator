# Compatibility — LLM endpoints, models, and versions

This file records what the app has been tested against, what it works with,
and what breaks when you change providers or models. Keep it updated when
defaults in `app/config.py` change or when providers deprecate APIs.

> **Rule of thumb:** This app works with **any OpenAI-compatible**
> `POST {base_url}/chat/completions` endpoint. The QwenCloud Token Plan values
> are just the tuned defaults. See `MIGRATION.md` for how to switch.

## Current defaults (`app/config.py`)

| Setting | Default value | Purpose |
|---------|---------------|---------|
| `DEFAULT_BASE_URL` | `https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1` | QwenCloud Token Plan (MaaS) — paired with `sk-sp-` keys; does NOT work against the general MaaS gateway |
| `DEFAULT_MODEL_MAIN` | `qwen3.7-max` | Low-volume heavy calls: claim extraction, verification, rubric, topic, rewrite |
| `DEFAULT_MODEL_FAST` | `qwen3.7-flash` | High-volume bulk calls: record-chunk digest and fact-merge |
| `DEFAULT_FETCH_SANDBOX_BASE_URL` | `https://fetchsandbox.com` | Fetch Sandbox host allowlist — change only for your own sandbox subdomain |

Override any of the above via `OPENAI_API_KEY` / `OPENAI_BASE_URL` /
`LLM_MODEL_MAIN` / `LLM_MODEL_FAST` in `.env` or the sidebar.

## Tested endpoints

| Provider / setup | Base URL pattern | Auth | Status | Min version / notes |
|------------------|------------------|------|--------|---------------------|
| **QwenCloud Token Plan (MaaS)** | `https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1` | `sk-sp-...` (Token Plan key) | **Primary / recommended** | OpenAI API v1 compatible; key+URL must be paired (Token Plan keys fail against the general gateway) |
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
| QwenCloud Token Plan | `qwen3.7-max` (default) | `qwen3.7-flash` (default) | Tuned for Individual Plan Lite (2500 credits / 7 days, 1–2 concurrent agents, `qwen3.7-flash` saves credit) |
| OpenAI | `gpt-4-turbo`, `gpt-4o` | `gpt-4o-mini` | Any reasoning-capable model works for main; use a cheaper model for fast |
| Ollama | provider-dependent | provider-dependent | Quality not evaluated; prefer at least a 7B instruction model |

The split (`model_main` vs `model_fast`) is the single largest credit/cost saver
on large record sets (1000+ pages = hundreds of chunk-digest calls). Do not point
both at the expensive model unless you accept the cost.

## Breaking changes by app version

| App version / commit | Change | Impact |
|----------------------|--------|--------|
| `main` @ 2026-09-14 (`DEFAULT_BASE_URL` → Token Plan, `model_{main,fast}` → `qwen3.7-*`) | Defaults moved from generic OpenAI to QwenCloud Token Plan | Existing `.env` pointing at OpenAI continues to work (env overrides defaults). New clones without `.env` now default to QwenCloud — set `OPENAI_BASE_URL` explicitly if you mean OpenAI. |
| Prompt injection hardening (`app/prompt_sanitize.py`) | User/record text is escaped before prompt interpolation; `>>>/<<</``` sanitized; guard note added | No API break; model output may be marginally different (safer). |
| `EVALUATE_INTERNAL_MAX_CHARS` / `DRAFT_INTERNAL_MAX_CHARS` hardened to `80_000` | Long statements/observations above 80K are truncated with a warning banner | Bypass callers that previously sent >80K now see truncation; split inputs. |

Add a new row here whenever defaults, prompt templates, or API contracts change.

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
