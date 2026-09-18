# Migration guide — switching LLM providers or endpoints

This guide covers the only file you usually need to touch: `.env`.
No code change is required for endpoint or model migrations.

## Quick reference

| Move | `.env` change | Notes |
|------|---------------|-------|
| Anything → **Perplexity Router API** (current default) | `OPENAI_BASE_URL=https://api.perplexity.ai/router/v1` + `LLM_MODEL_MAIN=perplexity/kimi-k3` / `LLM_MODEL_FAST=perplexity/glm-5.3-flash` | OpenAI Chat Completions schema, so only the base URL and key change. **Private preview** — request access from api@perplexity.ai first; the catalog is also the allowlist, so an unlisted model id returns a 400. The same key is reused as `PERPLEXITY_API_KEY` automatically, which turns on the Research tab |
| Perplexity Router → **QwenCloud Token Plan** | `OPENAI_BASE_URL=https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1` + `sk-sp-...` key + `LLM_MODEL_MAIN=qwen3.7-max` / `LLM_MODEL_FAST=qwen3.7-flash` | Key and URL must be paired (Token Plan keys fail against the general MaaS gateway, and vice versa). Set `PERPLEXITY_API_KEY` separately if you still want the Research tab |
| QwenCloud Token Plan → **OpenAI** | `OPENAI_BASE_URL=https://api.openai.com/v1` + OpenAI key + `LLM_MODEL_MAIN=gpt-4-turbo` / `LLM_MODEL_FAST=gpt-4o-mini` | Also works via `OPENAI_BASE_URL=https://api.openai.azure.com`-style Azure front-doors through a proxy |
| OpenAI → **QwenCloud Token Plan** | `OPENAI_BASE_URL=https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1` + `sk-sp-...` key + `LLM_MODEL_MAIN=qwen3.7-max` / `LLM_MODEL_FAST=qwen3.7-flash` | Key and URL must be paired; Token Plan keys fail against the general MaaS gateway (and vice versa) |
| Any provider → **local Ollama** (OpenAI-compat shim) | `OPENAI_BASE_URL=http://localhost:11434/v1` (or your proxy) + `LLM_MODEL_MAIN`/`LLM_MODEL_FAST` to locally-served model IDs | Requires an OpenAI-compat proxy in front of Ollama (e.g. Ollama's `/v1` shim). Quality not benchmarked — prefer a ≥7B instruction model. No auth key needed |
| **Only change the models** (same provider) | `LLM_MODEL_MAIN=...` / `LLM_MODEL_FAST=...` | Use a cheaper model for `FAST` on large record sets (1000+ pages) to save cost/credit |

> All names use the app's existing env vars (`OPENAI_*` / `LLM_MODEL_*`) even
> when the provider is not OpenAI — they are simply the OpenAI-compat knobs.
> The sidebar **LLM Settings → Apply settings** writes the same values.
>
> **Defaults vs. your configuration:** the shipped defaults are Perplexity's Router
> API (see `COMPATIBILITY.md → Current defaults`), but any value in `.env`, Streamlit
> secrets, or the sidebar overrides them — so an existing deployment that names its
> endpoint and models is untouched by a defaults change.

## Recipe — QwenCloud Token Plan → OpenAI

The app's defaults are tuned for QwenCloud Individual Plan Lite. To switch a
development or production deployment to OpenAI:

1. Create or edit `.env` at the project root (copy from `.env.example` if needed):

   ```ini
   OPENAI_API_KEY=sk-proj-your-openai-key-here
   OPENAI_BASE_URL=https://api.openai.com/v1
   LLM_MODEL_MAIN=gpt-4-turbo
   LLM_MODEL_FAST=gpt-4o-mini
   ```

   Anything not set falls back to defaults in `app/config.py` — setting only
   `OPENAI_API_KEY` and `OPENAI_BASE_URL` is enough to start; adjust the
   models when Qwen-named ones error.

2. Reload the app (`streamlit run run_app.py`, or rerun in Streamlit Cloud /
   Agiloop). The sidebar shows the new base URL and models; Apply settings is
   not needed when you changed `.env` on disk.

3. Trigger the **model availability warning** check (see below) — if either
   `LLM_MODEL_*` was left as `qwen3.7-*`, the sidebar shows:

   > `Model 'qwen3.7-max' not found at this endpoint. Check COMPATIBILITY.md ...`

   Replace with an OpenAI model that `GET {base_url}/models` lists. Suggested
   pairs: `gpt-4-turbo / gpt-4o-mini` or `gpt-4o / gpt-4o-mini`.

4. Optional: tune `VA_LSE_*` limits unchanged — they are provider-agnostic.

For CI, set the same four values as repository secrets (`OPENAI_API_KEY`,
`OPENAI_BASE_URL`, `LLM_MODEL_MAIN`, `LLM_MODEL_FAST`) per `SECURITY.md`; the
live smoke job (`scripts/smoke_test.py`) picks them up automatically.

## Recipe — OpenAI → QwenCloud

Reverse of above. The key detail is pairing:

- QwenCloud **Token Plan** keys start `sk-sp-` and *only* work with
  `https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1`.
- General MaaS gateway keys do not work against the Token Plan URL.

```ini
OPENAI_API_KEY=sk-sp-your-qwencloud-token-plan-key
OPENAI_BASE_URL=https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
LLM_MODEL_MAIN=qwen3.7-max
LLM_MODEL_FAST=qwen3.7-flash
```

Keep the credit tuning in mind: `qwen3.7-flash` for bulk digest + low
concurrency (`VA_LSE_RECORDS_CONCURRENCY=2`) preserves the 2,500 credit /
7-day Lite window. See `COMPATIBILITY.md` and README → *QwenCloud Individual
Plan Lite tuning*.

## Recipe — switching to a local model (Ollama)

Use an OpenAI-compatible proxy. Example for an in-repo proxy listening on
`11434`:

```ini
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
LLM_MODEL_MAIN=llama3.1:8b
LLM_MODEL_FAST=llama3.1:8b
```

The key can be any non-empty placeholder. Quality and JSON-mode reliability
vary by local model — the app appends `Respond with ONLY valid JSON` and
retries, but smaller/weaker models may still fail `app/llm.py:chat_json`.

## Model availability warning (GET {base_url}/models)

On startup `app/main.py` calls `app/llm.py:check_model_availability(base_url, api_key)`:

- On success it fetches `GET {base_url}/models` and checks that
  `LLM_MODEL_MAIN` and `LLM_MODEL_FAST` appear in the returned `data[].id` list.
- Missing models produce a non-blocking `st.warning` per model, linking to
  `COMPATIBILITY.md` and `MIGRATION.md`.
- Network failures, 401s, or permission errors are **ignored** — the app remains
  usable and the warning is simply not shown. Heavy per-request checking is not
  done; only the sidebar's cached check runs.

If you see `Model '…' not found`, run:

```bash
curl -s -H "Authorization: Bearer $OPENAI_API_KEY" "$OPENAI_BASE_URL/models" | python -m json.tool | grep '"id"'
```

and copy an `id` into `LLM_MODEL_MAIN` / `LLM_MODEL_FAST`.

## When you must change code

You do not for almost all migrations. Change code only when:

- The provider's response shape is not OpenAI-compatible (not needed for OpenAI, QwenCloud, Ollama via shim).
- You want to change the defaults for all fresh clones — edit `app/config.py` (`DEFAULT_BASE_URL` / `DEFAULT_MODEL_*`) and update `COMPATIBILITY.md`'s defaults table.
