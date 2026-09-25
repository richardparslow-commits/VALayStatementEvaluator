"""Detached availability prober for Perplexity's Sonar models.

Re-checks (hourly) whether ``sonar-pro`` / ``sonar`` can serve this app yet,
and the hour they can, rewrites ``.env`` to use them and fires a desktop
notification. Exit-after-success: one switch per watcher.

Two gates are probed, because live evidence (2026-09-24) split the answer by
route rather than by account:

* **responses** — the Agent API (``POST {OPENAI_BASE_URL}/responses``, this
  app's current route) accepting both models. This is the documented
  no-code-change switch: rewrite ``LLM_MODEL_MAIN=sonar-pro`` /
  ``LLM_MODEL_FAST=sonar`` and keep the base URL (see
  ``scripts/sonar_probe.py`` for the audit trail).
* **chat** — the classic Chat Completions route
  (``https://api.perplexity.ai/chat/completions``) accepting both. That route
  served sonar live on 2026-09-24 while the Agent API still rejected the
  models and ``/v1/chat/completions`` did not exist (404). The app builds its
  chat URL as ``{base_url}/chat/completions``, so reaching the classic route
  means rewriting ``OPENAI_BASE_URL`` to the bare host *and* pinning the wire
  schema with ``VA_LSE_LLM_ENDPOINT_SCHEMA=chat`` (app/llm.py's override knob;
  without the pin its HEAD probe resolves the host to the Responses schema).

``responses`` is preferred when both gates pass — it keeps the endpoint this
app is exercised against.

Safety rails:

* Every rewrite is backed up (``.env.bak-sonar-<stamp>``) first, and the edit
  is line-scoped to ``KEY=`` assignments — the API key line is never matched,
  comments and lookalike keys (``OPENAI_BASE_URL_FALLBACK``) survive intact.
* While a batch run is active (any live ``outputs/*/rerun.pid``) or a defer
  file exists, a passing probe notifies but does not rewrite: models switch
  between pipelines, never under one.

Usage::

    .venv/bin/python scripts/sonar_watch.py --launch     # detached, hourly
    .venv/bin/python scripts/sonar_watch.py --once       # one cycle, no detach

Exit codes: 0 = switched (or already switched / ``--once`` passes), 1 = still
unavailable (or the watcher's deadline passed), 2 = deferred or inconclusive.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, NamedTuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

ENV_FILE = ROOT / ".env"
LOG = ROOT / "logs" / "sonar_watch.log"
PIDFILE = ROOT / "logs" / "sonar_watch.pid"

MAIN_MODEL = "sonar-pro"
FAST_MODEL = "sonar"

#: Classic Chat Completions route. The ``/v1`` mirror of it does not exist
#: (404, live probe 2026-09-24), and the app appends ``/chat/completions`` to
#: ``OPENAI_BASE_URL`` (app/llm.py ``_endpoint_request_url``), so the chat
#: switch points the base URL at the bare host.
CHAT_URL = "https://api.perplexity.ai/chat/completions"
CHAT_BASE_URL = "https://api.perplexity.ai"

#: The two candidate .env rewrites. Order here is display order; the
#: preference order lives in :func:`choose_switch`.
SWITCHES: dict[str, dict[str, str]] = {
    "responses": {
        "LLM_MODEL_MAIN": MAIN_MODEL,
        "LLM_MODEL_FAST": FAST_MODEL,
    },
    "chat": {
        "OPENAI_BASE_URL": CHAT_BASE_URL,
        "VA_LSE_LLM_ENDPOINT_SCHEMA": "chat",
        "LLM_MODEL_MAIN": MAIN_MODEL,
        "LLM_MODEL_FAST": FAST_MODEL,
    },
}

PROBE_PROMPT = "Reply with exactly: OK"
#: Perplexity's documented max_tokens floor is 16 (see scripts/sonar_probe.py).
PROBE_MAX_TOKENS = 16
PROBE_TIMEOUT_S = 45.0


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def notify(title: str, message: str) -> None:
    """Best-effort macOS desktop notification; never raises.

    Fixed message strings only (no provider body, no .env content), so nothing
    response- or config-derived reaches a shell parser. Silent no-op on
    non-macOS hosts or when osascript is absent.
    """
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{message}" with title "{title}" sound name "Glass"'],
            check=False, capture_output=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        pass


# --------------------------------------------------------------------------- #
# The .env rewrite (pure — the safety of the switch lives here)
# --------------------------------------------------------------------------- #

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


def set_env_keys(text: str, updates: dict[str, str]) -> str:
    """Return *text* with each ``KEY=`` assignment in *updates* rewritten.

    Every occurrence of a key is rewritten (duplicate assignments would leave a
    stale value behind — which one wins depends on the loader, and a switch
    must not depend on that). Keys with no existing line are appended at the
    end. Comments, blank lines, and lookalike keys (``OPENAI_BASE_URL`` does
    not touch ``OPENAI_BASE_URL_FALLBACK``) pass through unchanged, so the
    file's comments and the API key survive the edit.
    """
    out: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines(keepends=True):
        m = _ENV_LINE.match(line)
        key = m.group(1) if m else None
        if key in updates:
            out.append(f"{key}={updates[key]}\n")
            seen.add(key)
        else:
            out.append(line)
    for key, value in updates.items():
        if key not in seen:
            out.append(f"{key}={value}\n")
    return "".join(out)


def choose_switch(responses_ok: bool, chat_ok: bool) -> str | None:
    """Which switch to apply given gate outcomes, or None when neither passes.

    ``responses`` wins ties: it is the documented switch and keeps
    ``OPENAI_BASE_URL`` — the endpoint this app is exercised against daily.
    """
    if responses_ok:
        return "responses"
    if chat_ok:
        return "chat"
    return None


def already_switched() -> bool:
    """True when .env already carries the model switch (either route)."""
    try:
        text = ENV_FILE.read_text(encoding="utf-8")
    except OSError:
        return False
    models = {"LLM_MODEL_MAIN": MAIN_MODEL, "LLM_MODEL_FAST": FAST_MODEL}
    return set_env_keys(text, models) == text


def apply_switch(name: str) -> str:
    """Back up .env, apply SWITCHES[*name*], and return the backup's name."""
    updates = SWITCHES[name]
    text = ENV_FILE.read_text(encoding="utf-8")
    backup = ENV_FILE.with_name(f".env.bak-sonar-{time.strftime('%Y%m%d-%H%M%S')}")
    backup.write_text(text, encoding="utf-8")
    ENV_FILE.write_text(set_env_keys(text, updates), encoding="utf-8")
    applied = ", ".join(f"{k}={v}" for k, v in updates.items())
    log(f"switch '{name}' applied ({applied}); backup at {backup.name}")
    return backup.name


# --------------------------------------------------------------------------- #
# Probing
# --------------------------------------------------------------------------- #

class GateResult(NamedTuple):
    ok: bool
    #: True when every probe verdict was definitive (accepted or rejected), so
    #: "still unavailable" can be said out loud instead of "inconclusive".
    certain: bool
    detail: str


def _post_chat(model: str, api_key: str) -> tuple[int | None, str]:
    """One tiny classic Chat Completions call; returns ``(status or None, body)``.

    Same contract as ``sonar_probe._post``: the body travels whole because the
    classification reads it; nothing renders it unbounded.
    """
    req = urllib.request.Request(
        CHAT_URL,
        data=json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": PROBE_PROMPT}],
                "max_tokens": PROBE_MAX_TOKENS,
            }
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_S) as resp:  # noqa: S310 - operator-configured endpoint
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        exc.close()
        return exc.code, body
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return None, repr(exc)


def probe_gate(gate: str, api_key: str) -> GateResult:
    """Probe both models on *gate* (``"responses"`` or ``"chat"``).

    Both must be ``accepted`` for the gate to pass: the switch serves
    sonar-pro as main and sonar as fast, so a half-available catalog is not a
    switch. Verdicts reuse ``sonar_probe._classify`` — one table for "is this
    model served", whoever reads it.
    """
    import sonar_probe  # same directory (sys.path above); stdlib-only

    ok = True
    certain = True
    details: list[str] = []
    for model in (MAIN_MODEL, FAST_MODEL):
        if gate == "responses":
            from app.config import load_settings

            status, body = sonar_probe._post(
                load_settings().base_url, api_key, model, PROBE_TIMEOUT_S
            )
        else:
            status, body = _post_chat(model, api_key)
        verdict, _note = sonar_probe._classify(status, body)
        ok = ok and verdict == "accepted"
        certain = certain and verdict in ("accepted", "rejected")
        details.append(f"{model}={verdict}")
    return GateResult(ok, certain, ", ".join(details))


def active_batch_run() -> int | None:
    """Pid of a live batch run (any ``outputs/*/rerun.pid``), or None.

    The switch must not land under a running pipeline: a resume would load the
    new models mid-project and the output would silently mix model families.
    """
    for pidfile in sorted((ROOT / "outputs").glob("*/rerun.pid")):
        try:
            pid = int(pidfile.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            return pid
        else:
            return pid
    return None


# --------------------------------------------------------------------------- #
# Cycles and CLI
# --------------------------------------------------------------------------- #

#: One probe cycle's outcome. ``deferred`` is a pass with the rewrite held back.
CYCLE_RESULTS = ("switched", "deferred", "not-yet", "inconclusive")


def cycle(defer_file: Path | None) -> str:
    """Run one probe cycle and act on it. See :data:`CYCLE_RESULTS`."""
    from app.config import load_settings

    api_key = load_settings().api_key
    if not api_key:
        log("no API key configured — inconclusive")
        return "inconclusive"

    results = {}
    for gate in SWITCHES:
        res = probe_gate(gate, api_key)
        results[gate] = res
        log(f"gate {gate}: {'PASS' if res.ok else 'fail'} — {res.detail}")

    name = choose_switch(results["responses"].ok, results["chat"].ok)
    if name is None:
        certain = all(r.certain for r in results.values())
        return "not-yet" if certain else "inconclusive"

    pid = active_batch_run()
    if pid is not None or (defer_file is not None and defer_file.exists()):
        why = f"batch run active (pid {pid})" if pid is not None else f"defer file {defer_file}"
        log(f"sonar passes via '{name}' but the switch is deferred: {why}")
        return "deferred"

    backup = apply_switch(name)
    notify(
        "Sonar is live — .env switched",
        f"sonar-pro/sonar now serve the '{name}' route; backup {backup}",
    )
    return "switched"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe for Sonar availability and switch .env the hour it passes.",
    )
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit")
    parser.add_argument(
        "--interval", type=int, default=3600, metavar="S",
        help="seconds between cycles (default: 3600 — one probe per hour)",
    )
    parser.add_argument(
        "--max-hours", type=float, default=72.0, metavar="H",
        help="give up after this many hours of 'not yet' (default: 72)",
    )
    parser.add_argument(
        "--defer-file", type=Path, default=None, metavar="PATH",
        help="while this file exists, a passing probe notifies but does not rewrite .env",
    )
    parser.add_argument(
        "--launch", action="store_true",
        help="detach into its own session and keep probing in the background",
    )
    args = parser.parse_args(argv)

    if args.launch:
        child = [str(Path(sys.executable)), str(Path(__file__).resolve()),
                 "--interval", str(args.interval), "--max-hours", str(args.max_hours)]
        if args.defer_file is not None:
            child += ["--defer-file", str(args.defer_file)]
        proc = subprocess.Popen(  # noqa: S603 - fixed argv, own interpreter
            child,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True, cwd=str(ROOT),
        )
        PIDFILE.parent.mkdir(parents=True, exist_ok=True)
        PIDFILE.write_text(str(proc.pid))
        print("detached sonar watcher pid:", proc.pid)
        return 0

    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()))

    if already_switched():
        log("sonar already configured in .env — nothing to do")
        print("sonar already configured — nothing to do")
        return 0

    exit_codes = {"switched": 0, "not-yet": 1, "inconclusive": 2, "deferred": 2}
    if args.once:
        result = cycle(args.defer_file)
        print(f"cycle result: {result}")
        return exit_codes[result]

    deadline = time.time() + args.max_hours * 3600.0
    deferred_notified = False
    log(f"sonar watcher started (pid {os.getpid()}), probing every {args.interval}s")
    while time.time() < deadline:
        result = cycle(args.defer_file)
        if result == "switched":
            return 0
        if result == "deferred" and not deferred_notified:
            notify(
                "Sonar switch deferred",
                "sonar is live but a batch run is active — switching when it finishes",
            )
            deferred_notified = True
        time.sleep(args.interval)
    log(f"gave up after {args.max_hours}h — sonar still unavailable on both routes")
    notify("Sonar watcher gave up", "sonar still unavailable — keep the current models")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
