"""Probe whether Perplexity's Sonar models are live on the Agent API yet.

The 2026-09-23 audit (session evidence, live probes against this account's
key) established the routing truth this script re-checks on demand:

- The account's working surface is the Agent API: ``OPENAI_BASE_URL`` points
  at ``https://api.perplexity.ai/v1`` and every call goes to ``/responses``.
  ``/chat/completions`` does not exist there (live probe: 404), and on the
  classic host it answers 403 with "Sonar is now the Agent API. Use
  /v1/responses instead" — i.e. the standard is moving TO the Agent API, not
  away from it.
- ``sonar`` and ``sonar-pro`` were rejected on the Agent API that day with
  ``400 validation failed: model "…" is not supported``. When Perplexity
  finishes migrating Sonar onto the Agent API (their own 403 says they are),
  that rejection flips to 200 — and the app switch is one .env edit:
  ``LLM_MODEL_MAIN=sonar-pro`` / ``LLM_MODEL_FAST=sonar``. No code changes:
  the client is model-agnostic and the schema routing already serves
  ``/responses`` for this host.

Run it whenever you want to know:

    .venv/bin/python scripts/sonar_probe.py            # human-readable
    .venv/bin/python scripts/sonar_probe.py --json     # machine-readable

Exit codes: 0 = at least one probed model is ACCEPTED (switch is possible),
1 = probed models all rejected (not yet), 2 = inconclusive (rate limited,
auth refused, or endpoint unreachable). Accepted probes cost at most
``max_output_tokens`` (16) output tokens per model; refused models cost
nothing. The API key is never printed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_settings  # noqa: E402

#: Perplexity's documented max_tokens floor; smaller probes are refused with
#: a confusing "max_tokens must be at least 16" instead of a model verdict.
_MIN_OUTPUT_TOKENS = 16

_OK_INPUT = "Reply with exactly: OK"


def _classify(status: int | None, body: str) -> tuple[str, str]:
    """Map a probe response onto ``(verdict, note)``.

    Verdicts: ``accepted`` (model served), ``rejected`` (model not in the
    catalog yet), ``forbidden`` (valid key, no entitlement), ``auth`` (key
    refused — regenerate/credits first; probe result is meaningless),
    ``rate_limited`` (try again later), ``unreachable`` (no HTTP answer).

    Classification always sees the FULL body — a marker phrase sitting past
    the display truncation must still classify correctly (a probe that read
    only the first 200 characters would report a real catalog rejection as
    ``inconclusive``). Truncation is for the note text only.

    A 404 splits on the body: a JSON object with an ``error`` key is the
    endpoint's structured API error (several providers reject unknown models
    with 404-shaped bodies), so it reads as ``rejected`` — the conservative
    direction ("not yet", never "switch now"). A bare or non-JSON 404 is a
    missing route or a proxy page, which says nothing about the model, and
    stays ``unreachable``.
    """
    if status == 200:
        return "accepted", "model served by the endpoint"
    if status is None:
        return "unreachable", "no HTTP response (network/timeout)"
    lowered = body.lower()
    if status == 400 and "not supported" in lowered:
        return "rejected", "model not in the endpoint's catalog yet"
    if status == 404:
        try:
            structured_error = "error" in json.loads(body)
        except ValueError:
            structured_error = False
        if structured_error:
            return "rejected", "endpoint answered 404 with a structured error — model not served"
        return "unreachable", f"route missing at this base URL ({status})"
    if status == 403:
        return "forbidden", "valid key without entitlement to this model"
    if status in (401,):
        return "auth", "key refused — check the key and credit balance first"
    if status == 429 or "rate limit" in lowered:
        return "rate_limited", "throttled (the batch run may own the budget) — retry later"
    return "inconclusive", f"HTTP {status}: {body[:120]}"


def _post(base_url: str, api_key: str, model: str, timeout: float) -> tuple[int | None, str]:
    """One tiny ``/responses`` call; returns ``(status or None, full body)``.

    The body is returned whole because classification reads it (see
    :func:`_classify`); nothing renders it unbounded — the note truncates.
    """
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/responses",
        data=json.dumps(
            {
                "model": model,
                "input": _OK_INPUT,
                "max_output_tokens": _MIN_OUTPUT_TOKENS,
                "store": False,
            }
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - operator-configured endpoint
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        exc.close()
        return exc.code, body
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return None, repr(exc)


def _request_timeout(requested: int | None) -> float:
    """CLI value, else the app's call-timeout setting, clamped for a probe."""
    if requested and requested > 0:
        return min(float(requested), 120.0)
    from app import config as _cfg

    try:
        return min(max(1.0, float(getattr(_cfg, "LLM_CALL_TIMEOUT_SECONDS", 30))), 120.0)
    except (TypeError, ValueError):
        return 30.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check whether Sonar models are accepted on the Perplexity Agent API yet.",
    )
    parser.add_argument(
        "--models",
        default="sonar-pro,sonar",
        help="Comma-separated model ids to probe (default: sonar-pro,sonar)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        metavar="S",
        help="Per-request timeout in seconds (default: the app's LLM call timeout, clamped to 120)",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON report instead of text")
    args = parser.parse_args(argv)

    settings = load_settings()
    if not settings.api_key:
        print("No API key configured — set OPENAI_API_KEY (or add it to .env).", file=sys.stderr)
        return 2

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    timeout = _request_timeout(args.timeout)

    results: list[dict[str, Any]] = []
    for model in models:
        t0 = time.perf_counter()
        status, body = _post(settings.base_url, settings.api_key, model, timeout)
        verdict, note = _classify(status, body)
        results.append(
            {
                "model": model,
                "status": status,
                "verdict": verdict,
                "note": note,
                "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            }
        )

    if args.json:
        print(json.dumps({"base_url": settings.base_url, "results": results}, indent=2))
    else:
        print(f"endpoint: {settings.base_url}  route: /responses  timeout: {timeout:.0f}s")
        for r in results:
            print(
                f"  {r['model']:<12} HTTP {str(r['status']):>4}  "
                f"{r['verdict']:<12} {r['note']}  ({r['elapsed_ms']} ms)"
            )

    verdicts = {r["verdict"] for r in results}
    if "accepted" in verdicts:
        if not args.json:
            print(
                "\nSonar is live on the Agent API — the switch is now a .env edit:\n"
                "  LLM_MODEL_MAIN=sonar-pro\n  LLM_MODEL_FAST=sonar\n"
                "(no code changes needed; keep OPENAI_BASE_URL as-is)"
            )
        return 0
    if verdicts <= {"rejected"}:
        if not args.json:
            print("\nSonar is not on the Agent API yet — keep the current models.")
        return 1
    if not args.json:
        print("\nInconclusive — resolve the note above and re-run.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
