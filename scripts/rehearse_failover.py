#!/usr/bin/env python
"""Observe and verify LLM endpoint failover on a running deployment.

Answers the two questions an operator has during (or after) a provider outage:
**did traffic move to the backup, and did it come back?** It reads the health
sidecar's probe port only — ``/health`` and ``/metrics`` — so it needs no
credentials, sends no LLM traffic, and cannot disturb a run in progress. It never
induces an outage; it observes one, or a rehearsal you caused yourself.

    # one snapshot, human readable
    python scripts/rehearse_failover.py

    # watch an outage live, printing each stage change as it happens
    python scripts/rehearse_failover.py --watch --interval 5

    # scripted rehearsal: assert the stage, and fail the command if it is wrong
    python scripts/rehearse_failover.py --expect-idle     # before the outage
    python scripts/rehearse_failover.py --expect-active   # during it

    # inside a pod, or against a remote probe port
    python scripts/rehearse_failover.py --url http://va-lse-web.va-lse.svc:8001

Rehearsal procedure (see DEPLOYMENT.md -> LLM endpoint failover):

1. ``--expect-idle`` — confirms the backup is armed and unused.
2. Cause the outage (point ``OPENAI_BASE_URL`` at an unroutable host, or block
   egress to it) and ``--watch``. Expect: the primary breaker to open, the
   unhealthy clock to grow, then ``active`` to flip to 1 after
   ``LLM_ENDPOINT_FALLBACK_TIMEOUT_SECONDS``.
3. ``--expect-active`` — confirms traffic is being served by the backup, and that
   the run stamps say so.
4. Restore the primary and ``--watch`` again. Expect ``active`` to return to 0 and
   the unhealthy clock to reset **without** a restart.
5. Undo step 2, then ``--expect-idle`` once more.

Exit codes:

* ``0`` — the snapshot is consistent and matches whatever ``--expect-*`` asked for
* ``1`` — the deployment could not be read, or the snapshot is self-contradictory
* ``2`` — readable and consistent, but not the stage that was expected. Also
  argparse's standard code for a bad flag, so ``--expect-nonsense`` fails loudly
  rather than being silently ignored.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_URL = "http://localhost:8001"


class SnapshotError(RuntimeError):
    """The deployment's health surface could not be read."""


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        # /ready legitimately returns 503; /health does not. Surface the code.
        raise SnapshotError(f"{url} returned HTTP {exc.code}") from exc
    except Exception as exc:  # noqa: BLE001 - reported, not raised to a traceback
        raise SnapshotError(f"{url} unreachable: {type(exc).__name__}: {exc}") from exc
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise SnapshotError(f"{url} did not return JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SnapshotError(f"{url} returned {type(data).__name__}, expected an object")
    return data


def _get_text(url: str, timeout: float) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return str(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - reported, not raised to a traceback
        raise SnapshotError(f"{url} unreachable: {type(exc).__name__}: {exc}") from exc


def _metric(body: str, name: str) -> float | None:
    """Value of an unlabelled metric, or None when the series is absent.

    Absent is *not* zero here: the exposition omits a value it could not read, and
    the checks below must be able to tell "not reported" from "reported as 0".
    """
    for line in body.splitlines():
        if line.startswith(name + " "):
            _, _, value = line.rpartition(" ")
            try:
                return float(value)
            except ValueError:
                return None
    return None


def snapshot(base_url: str, *, timeout: float = 5.0) -> dict[str, Any]:
    """Read one consistent-ish snapshot of the failover state."""
    root = base_url.rstrip("/")
    health = _get_json(f"{root}/health", timeout)
    metrics = _get_text(f"{root}/metrics", timeout)

    failover = health.get("llm_failover")
    if not isinstance(failover, Mapping):
        # An older pod without this feature. Say so rather than inventing zeros.
        failover = {}

    breaker = _breaker_state(metrics)
    return {
        "health_status": str(health.get("status") or "unknown"),
        "configured": bool(failover.get("configured", False)),
        "active": bool(failover.get("active", False)),
        "after_seconds": failover.get("after_seconds"),
        "primary_unhealthy_seconds": failover.get("primary_unhealthy_seconds"),
        "feature_present": bool(failover),
        "metrics": {
            "failover_enabled": _metric(metrics, "va_lse_llm_failover_enabled"),
            "failover_active": _metric(metrics, "va_lse_llm_failover_active"),
            "failover_after_seconds": _metric(metrics, "va_lse_llm_failover_after_seconds"),
            "primary_unhealthy_seconds": _metric(
                metrics, "va_lse_llm_primary_unhealthy_seconds"
            ),
            "primary_breaker_open": _metric(
                metrics, 'va_lse_circuit_breaker_open{breaker="llm"}'
            ),
            "primary_breaker_state": breaker,
        },
    }


def _breaker_state(metrics: str) -> str:
    """The primary breaker's state name, from the enum gauge."""
    for line in metrics.splitlines():
        if line.startswith('va_lse_circuit_breaker_state{breaker="llm"} '):
            _, _, value = line.rpartition(" ")
            try:
                codes = {"0": "CLOSED", "1": "HALF_OPEN", "2": "OPEN"}
                return codes.get(value.strip(), "UNKNOWN")
            except Exception:  # noqa: BLE001 - display only
                return "UNKNOWN"
    return "ABSENT"


def inconsistencies(snap: Mapping[str, Any]) -> list[str]:
    """Ways this snapshot contradicts itself.

    The point of a check like this: a monitoring stack that reports something
    impossible is worse than one that reports nothing, because it is trusted. Each
    entry names the two values that disagree, so the message is actionable without
    reading this source.
    """
    problems: list[str] = []
    metrics = snap.get("metrics") or {}
    enabled = metrics.get("failover_enabled")
    active_metric = metrics.get("failover_active")
    active = bool(snap.get("active"))
    unhealthy = snap.get("primary_unhealthy_seconds")
    threshold = snap.get("after_seconds")

    if enabled == 0 and active:
        problems.append(
            "failover is active while va_lse_llm_failover_enabled is 0 — "
            "no backup endpoint is configured, so nothing can be serving these calls"
        )
    if enabled is not None and active_metric is not None and (active != bool(active_metric)):
        problems.append(
            f"/health says active={active} but va_lse_llm_failover_active is "
            f"{active_metric:g} — the two surfaces disagree"
        )
    if (
        isinstance(unhealthy, (int, float))
        and isinstance(threshold, (int, float))
        and threshold >= 0
        and unhealthy >= threshold
        and not active
    ):
        problems.append(
            f"the primary has been unhealthy for {unhealthy:.0f}s, past the "
            f"{threshold:.0f}s failover threshold, but active is false — traffic "
            "should already have moved to the backup"
        )
    if active and metrics.get("primary_breaker_state") in ("CLOSED", "ABSENT"):
        problems.append(
            "failover is active while the primary breaker is "
            f"{metrics.get('primary_breaker_state')} — either the primary recovered "
            "without the clock resetting, or the breaker is not the one being used"
        )
    return problems


def describe(snap: Mapping[str, Any]) -> str:
    """One human-readable block for a snapshot."""
    metrics = snap.get("metrics") or {}
    lines = [f"  health status  : {snap.get('health_status')}"]
    if not snap.get("feature_present"):
        lines.append(
            "  failover       : NOT REPORTED by this pod (no `llm_failover` block — "
            "an older build, or a sidecar that predates the feature)"
        )
        return "\n".join(lines)

    if not snap.get("configured"):
        state = "not configured — this deployment runs on one endpoint"
    elif snap.get("active"):
        state = "ACTIVE — calls are being served by the backup endpoint"
    elif isinstance(snap.get("primary_unhealthy_seconds"), (int, float)) and (
        snap["primary_unhealthy_seconds"] or 0
    ) > 0:
        remaining = max(0.0, float(snap.get("after_seconds") or 0) - float(snap["primary_unhealthy_seconds"]))
        state = f"armed, primary failing — failover in ~{remaining:.0f}s"
    else:
        state = "armed and unused — the primary is healthy"

    lines += [
        f"  failover       : {state}",
        f"  primary breaker: {metrics.get('primary_breaker_state')}",
        f"  unhealthy for  : {_seconds(snap.get('primary_unhealthy_seconds'))}",
        f"  failover after : {_seconds(snap.get('after_seconds'))}",
    ]
    return "\n".join(lines)


def _seconds(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.0f}s"
    return "healthy" if value is None else "unknown"


def _stage(snap: Mapping[str, Any]) -> str:
    """A short stage name, used to print changes only while watching."""
    if not snap.get("configured"):
        return "single-endpoint"
    if snap.get("active"):
        return "serving-from-backup"
    unhealthy = snap.get("primary_unhealthy_seconds")
    if isinstance(unhealthy, (int, float)) and unhealthy > 0:
        return "primary-failing"
    return "primary-healthy"


def _verdict(
    snap: Mapping[str, Any],
    *,
    expect_active: bool,
    expect_idle: bool,
    quiet: bool = False,
) -> int:
    """Exit code for a snapshot, with the explanation printed unless quiet.

    ``quiet`` exists so ``--json`` can emit *only* JSON on stdout: prose after a
    JSON document makes the output unparseable for whatever consumes it, which
    defeats the point of the flag.
    """

    def say(message: str) -> None:
        if not quiet:
            print(message)

    problems = inconsistencies(snap)
    if problems:
        say("\nINCONSISTENT SNAPSHOT:")
        for problem in problems:
            say(f"  - {problem}")
        return 1
    if not snap.get("configured"):
        say(
            "\nOK: one endpoint, no failover configured. (Set "
            "OPENAI_BASE_URL_FALLBACK to arm one.)"
        )
        return 0
    if expect_active and not snap.get("active"):
        say("\nEXPECTED ACTIVE, but traffic is still on the primary.")
        return 2
    if expect_idle and snap.get("active"):
        say("\nEXPECTED IDLE, but traffic is being served by the backup.")
        return 2
    if snap.get("active"):
        say(
            "\nOK: consistent, and being served by the backup endpoint. Runs during "
            "this window are stamped with both endpoints (llm_endpoints)."
        )
    else:
        say("\nOK: consistent, and on the primary endpoint.")
    return 0


def _watch(base_url: str, *, interval: float, timeout: float) -> int:
    """Follow the state, printing each stage change. Ctrl-C to stop."""
    print(f"Watching {base_url} every {interval:g}s — Ctrl-C to stop.\n")
    last_stage: str | None = None
    last_problem_count = -1
    while True:
        try:
            snap = snapshot(base_url, timeout=timeout)
            stage = _stage(snap)
            problems = inconsistencies(snap)
            if stage != last_stage or len(problems) != last_problem_count:
                stamp = time.strftime("%H:%M:%S")
                print(f"[{stamp}] stage={stage}")
                print(describe(snap))
                for problem in problems:
                    print(f"  !! {problem}")
                last_stage = stage
                last_problem_count = len(problems)
        except SnapshotError as exc:
            print(f"[{time.strftime('%H:%M:%S')}] {exc}")
            last_stage = "unreachable"
        time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rehearse_failover.py",
        description=(
            "Read LLM endpoint failover state from a running deployment's probe port. "
            "Read-only: sends no LLM traffic and cannot disturb a run."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", default=DEFAULT_URL, help=f"base URL (default {DEFAULT_URL})")
    parser.add_argument("--timeout", type=float, default=5.0, help="per-request timeout (s)")
    parser.add_argument("--json", action="store_true", help="emit the snapshot as JSON")
    parser.add_argument("--watch", action="store_true", help="follow the state until interrupted")
    parser.add_argument("--interval", type=float, default=5.0, help="watch poll interval (s)")
    expectation = parser.add_mutually_exclusive_group()
    expectation.add_argument(
        "--expect-active", action="store_true", help="exit 2 unless traffic is on the backup"
    )
    expectation.add_argument(
        "--expect-idle", action="store_true", help="exit 2 if traffic is on the backup"
    )
    args = parser.parse_args(argv)

    if args.watch:
        if args.expect_active or args.expect_idle:
            print("--watch and --expect-* are different modes; use one.", file=sys.stderr)
            return 2
        try:
            return _watch(args.url, interval=args.interval, timeout=args.timeout)
        except KeyboardInterrupt:
            print("\nStopped.")
            return 0

    try:
        snap = snapshot(args.url, timeout=args.timeout)
    except SnapshotError as exc:
        print(f"Could not read {args.url}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        # JSON only on stdout, so this stays pipeable into jq or a check script.
        print(json.dumps(snap, indent=2, sort_keys=True))
    else:
        print(f"Failover state for {args.url}")
        print(describe(snap))
    return _verdict(
        snap,
        expect_active=bool(args.expect_active),
        expect_idle=bool(args.expect_idle),
        quiet=bool(args.json),
    )


if __name__ == "__main__":
    sys.exit(main())
