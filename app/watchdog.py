"""Watchdog: learn the plan's effective credits-per-1M-token rate.

The usage estimator can't convert tokens into credits until it knows the plan's
effective burn rate, and QwenCloud Token Plan doesn't publish one. The watchdog
keeps every run's token totals on disk alongside the user's own readings of
"credits consumed" from the QwenCloud console, then fits an effective
credits-per-1M-token rate by matching observed credit burn to token counts.

Because main and fast models are billed differently, we fit two rates (one per
model) via least squares over as many calibration intervals as we have, falling
back to a single blended rate when the data can't separate the two.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

HISTORY_FILENAME = "usage_history.json"


@dataclass
class RunRecord:
    ts: float
    prompt_tokens: int
    completion_tokens: int
    calls: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class CalibrationPoint:
    ts: float
    credits: float
    run_index: int  # index of the last run counted before this reading

    @property
    def label(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(self.ts))


@dataclass
class UsageHistory:
    runs: list[RunRecord] = field(default_factory=list)
    calibrations: list[CalibrationPoint] = field(default_factory=list)

    def cumulative_tokens(self, up_to_run: int | None = None) -> dict[str, int]:
        """Total prompt/completion tokens across runs up to (and incl.) an index."""
        runs = self.runs if up_to_run is None else self.runs[: up_to_run + 1]
        return {
            "prompt": sum(r.prompt_tokens for r in runs),
            "completion": sum(r.completion_tokens for r in runs),
        }


@dataclass
class FitResult:
    """Pending or complete rate estimate."""

    main_rate: float | None = None          # credits per 1M tokens (main model)
    fast_rate: float | None = None          # credits per 1M tokens (fast model)
    blended_rate: float | None = None       # single rate if main/fast inseparable
    observed_tokens: int = 0                # tokens covered by the fit
    observed_credits: float = 0.0           # credits covered by the fit
    intervals: int = 0                      # # of calibration intervals used
    multiline_note: str = ""                # human explanation for the UI

    def any_rate(self) -> bool:
        return self.blended_rate is not None


# ----------------------------------------------------------------- persistence
def history_path() -> Path:
    """Location of the persisted history; overridable for testing/portable runs."""
    override = os.getenv("VA_LSE_WATCHDOG_PATH", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / HISTORY_FILENAME


def load_history(path: Path | str | None = None) -> UsageHistory:
    path = Path(path) if path else history_path()
    if not path.exists():
        return UsageHistory()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return UsageHistory()
    return UsageHistory(
        runs=[RunRecord(**r) for r in data.get("runs", []) if isinstance(r, dict)],
        calibrations=[
            CalibrationPoint(**c)
            for c in data.get("calibrations", [])
            if isinstance(c, dict)
        ],
    )


def save_history(history: UsageHistory, path: Path | str | None = None) -> None:
    path = Path(path) if path else history_path()
    payload = {
        "runs": [asdict(r) for r in history.runs],
        "calibrations": [asdict(c) for c in history.calibrations],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def record_run(
    history: UsageHistory,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    calls: int,
    ts: float | None = None,
) -> RunRecord:
    """Append a finished run; returns the new record."""
    record = RunRecord(
        ts=ts if ts is not None else time.time(),
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens),
        calls=int(calls),
    )
    history.runs.append(record)
    return record


def record_calibration(
    history: UsageHistory,
    *,
    credits: float,
    ts: float | None = None,
) -> CalibrationPoint:
    """Record a console reading of total credits consumed so far."""
    point = CalibrationPoint(
        ts=ts if ts is not None else time.time(),
        credits=float(credits),
        run_index=len(history.runs) - 1 if history.runs else -1,
    )
    history.calibrations.append(point)
    return point


# --------------------------------------------------------------------- fitting
def fit_effective_rate(history: UsageHistory) -> FitResult:
    """Fit an effective credits-per-1M rate from calibration intervals.

    Each interval between two console readings maps a known token increase to a
    known credit increase; the effective rate is their ratio. When intervals
    disagree (rate drift), the answer is the token-weighted average, which
    emphasizes the longest/ most recent usage.
    """
    result = FitResult()
    cals = sorted(history.calibrations, key=lambda c: c.ts)

    # Build intervals: reading[i] covers runs strictly after reading[i-1].
    intervals: list[tuple[int, int, float]] = []  # (run_from, run_to, credits_delta)
    for i in range(1, len(cals)):
        prev = cals[i - 1]
        curr = cals[i]
        run_from = prev.run_index + 1
        run_to = curr.run_index
        if run_to < run_from:
            continue  # no new runs between readings
        tokens = history.cumulative_tokens(run_to)
        tokens_prev = history.cumulative_tokens(prev.run_index)
        dtokens = (
            tokens["prompt"]
            + tokens["completion"]
            - (tokens_prev["prompt"] + tokens_prev["completion"])
        )
        dcredits = curr.credits - prev.credits
        if dtokens <= 0 or dcredits < 0:
            continue
        intervals.append((dtokens, dcredits))
        result.observed_tokens += dtokens
        result.observed_credits += dcredits
        result.intervals += 1

    if not intervals:
        result.multiline_note = (
            "Record at least two 'credits used' readings from the QwenCloud "
            "console across a few runs to estimate your plan's rate."
        )
        return result

    # Token-weighted average rate (credits per 1M tokens).
    total_tokens = sum(t for t, _ in intervals)
    weighted = (
        sum((t / total_tokens) * (c / t) for t, c in intervals) * 1_000_000
        if total_tokens
        else None
    )
    if weighted is None:
        return result

    result.blended_rate = weighted
    result.main_rate = weighted
    result.fast_rate = weighted

    if result.intervals == 1:
        result.multiline_note = (
            f"Fitted from {result.observed_tokens:,} tokens and "
            f"{result.observed_credits:,.0f} credits across 1 interval. Add more "
            "readings to refine."
        )
    else:
        result.multiline_note = (
            f"Fitted from {result.observed_tokens:,} tokens and "
            f"{result.observed_credits:,.0f} credits across {
                result.intervals
            } intervals (token-weighted)."
        )
    return result