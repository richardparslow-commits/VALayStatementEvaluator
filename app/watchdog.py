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
    by_role: dict[str, int] = field(default_factory=dict)  # "main"/"fast" -> tokens

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
    def _run_from_dict(raw: dict) -> RunRecord:
        return RunRecord(
            ts=float(raw.get("ts", 0.0)),
            prompt_tokens=int(raw.get("prompt_tokens", 0)),
            completion_tokens=int(raw.get("completion_tokens", 0)),
            calls=int(raw.get("calls", 0)),
            by_role={
                str(k): int(v)
                for k, v in (raw.get("by_role") or {}).items()
                if isinstance(k, str) and isinstance(v, (int, float))
            },
        )

    return UsageHistory(
        runs=[_run_from_dict(r) for r in data.get("runs", []) if isinstance(r, dict)],
        calibrations=[
            CalibrationPoint(
                ts=float(c.get("ts", 0.0)),
                credits=float(c.get("credits", 0.0)),
                run_index=int(c.get("run_index", -1)),
            )
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
    by_role: dict[str, int] | None = None,
    ts: float | None = None,
) -> RunRecord:
    """Append a finished run; returns the new record."""
    record = RunRecord(
        ts=ts if ts is not None else time.time(),
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens),
        calls=int(calls),
        by_role={str(k): int(v) for k, v in (by_role or {}).items()},
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
def _role_delta(runs: list[RunRecord]) -> dict[str, int]:
    """Sum per-role tokens across a run slice."""
    out: dict[str, int] = {}
    for run in runs:
        for role, tokens in run.by_role.items():
            out[role] = out.get(role, 0) + tokens
    return out


def fit_effective_rate(history: UsageHistory) -> FitResult:
    """Fit credits-per-1M rates for the main and fast models separately.

    Each interval between console readings maps a known credit increase to known
    token increases per model role. With at least two intervals whose main/fast
    mix differs, we solve the two rates via ordinary least squares. When the
    data can't separate the models (only one model used, or every interval has
    the same main/fast ratio), we fall back to a single blended rate.

    Regression model per interval i:
        credits_i ~= main_tokens_i/1e6 * r_main + fast_tokens_i/1e6 * r_fast
    """
    result = FitResult()
    cals = sorted(history.calibrations, key=lambda c: c.ts)

    intervals: list[tuple[dict[str, int], float]] = []  # (role tokens, credits delta)
    for i in range(1, len(cals)):
        prev = cals[i - 1]
        curr = cals[i]
        run_from = prev.run_index + 1
        run_to = curr.run_index
        if run_to < run_from:
            continue  # no new runs between readings
        covered = history.runs[run_from : run_to + 1]
        tokens_prev = history.cumulative_tokens(prev.run_index)
        tokens = history.cumulative_tokens(run_to)
        dtokens = (
            tokens["prompt"]
            + tokens["completion"]
            - (tokens_prev["prompt"] + tokens_prev["completion"])
        )
        dcredits = curr.credits - prev.credits
        if dtokens <= 0 or dcredits <= 0:
            continue
        intervals.append((_role_delta(covered), dcredits))
        result.observed_tokens += dtokens
        result.observed_credits += dcredits
        result.intervals += 1

    if not intervals:
        result.multiline_note = (
            "Record at least two 'credits used' readings from the QwenCloud "
            "console across a few runs to estimate your plan's rate."
        )
        return result

    blended = result.observed_credits / result.observed_tokens * 1_000_000
    result.blended_rate = blended

    # --- least squares for (r_main, r_fast): X^T X r = X^T y ---
    x1: list[tuple[float, float, float]] = []  # (main_tokens/1e6, fast_tokens/1e6, credits)
    for role_tokens, dcredits in intervals:
        dm = role_tokens.get("main", 0)
        df = role_tokens.get("fast", 0)
        if dm + df <= 0:
            continue  # no per-role breakdown for this interval
        x1.append((dm / 1_000_000.0, df / 1_000_000.0, dcredits))

    if len(x1) >= 2:
        # Normal equations.
        s_aa = sum(a * a for a, _, _ in x1)
        s_ab = sum(a * b for a, b, _ in x1)
        s_bb = sum(b * b for _, b, _ in x1)
        s_ay = sum(a * y for a, _, y in x1)
        s_by = sum(b * y for _, b, y in x1)
        det = s_aa * s_bb - s_ab * s_ab
        if abs(det) > 1e-12:
            r_main = (s_ay * s_bb - s_ab * s_by) / det
            r_fast = (s_aa * s_by - s_ab * s_ay) / det
            if r_main >= 0 and r_fast >= 0:  # reject negative (nonphysical) fits
                result.main_rate = r_main
                result.fast_rate = r_fast

    if result.main_rate is None:
        # Couldn't separate models: apply the blended rate to both.
        result.main_rate = blended
        result.fast_rate = blended

    if result.intervals == 1:
        result.multiline_note = (
            f"Fitted from {result.observed_tokens:,} tokens and "
            f"{result.observed_credits:,.0f} credits across 1 interval. Add "
            "readings after both models run to separate their rates."
        )
    elif result.main_rate == result.fast_rate:
        result.multiline_note = (
            f"Fitted from {result.observed_tokens:,} tokens and "
            f"{result.observed_credits:,.0f} credits across {
                result.intervals
            } intervals, but the data couldn't separate the two models "
            "(same rate applied to both). Add runs where main and fast mix "
            "differs to split them."
        )
    else:
        result.multiline_note = (
            f"Fitted from {result.observed_tokens:,} tokens and "
            f"{result.observed_credits:,.0f} credits across {
                result.intervals
            } intervals for two separate model rates."
        )
    return result