"""Estimated LLM API usage tracking.

Each LLM call is recorded with the model it used and the pipeline phase that
made it. Token counts are *estimates*: when the provider reports usage metadata
we trust those figures, otherwise we fall back to a characters/4 heuristic.
Credit burn (QwenCloud Token Plan credits, etc.) is converted only when a
per-model rate is configured — Token Plan does not publish fixed rates, so we
never guess one by default.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars/token, never zero for non-empty input."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


@dataclass
class UsageEntry:
    model: str
    phase: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class PhaseStats:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    models: Counter = field(default_factory=Counter)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class UsageTracker:
    """Collect per-call usage across a single pipeline run."""

    def __init__(self) -> None:
        self.entries: list[UsageEntry] = []

    def record(
        self,
        *,
        model: str,
        phase: str,
        system: str,
        user: str,
        content: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        """Record one completed call. Prefer real metadata when available."""
        self.entries.append(
            UsageEntry(
                model=model,
                phase=phase,
                prompt_tokens=(
                    prompt_tokens
                    if prompt_tokens is not None
                    else estimate_tokens(system) + estimate_tokens(user)
                ),
                completion_tokens=(
                    completion_tokens
                    if completion_tokens is not None
                    else estimate_tokens(content)
                ),
            )
        )

    # ------------------------------------------------------------ aggregation
    def per_phase(self) -> OrderedDict[str, PhaseStats]:
        stats: OrderedDict[str, PhaseStats] = OrderedDict()
        for entry in self.entries:
            s = stats.setdefault(entry.phase, PhaseStats())
            s.calls += 1
            s.prompt_tokens += entry.prompt_tokens
            s.completion_tokens += entry.completion_tokens
            s.models[entry.model] += 1
        return stats

    def totals(self) -> PhaseStats:
        total = PhaseStats()
        for entry in self.entries:
            total.calls += 1
            total.prompt_tokens += entry.prompt_tokens
            total.completion_tokens += entry.completion_tokens
            total.models[entry.model] += 1
        return total

    # ---------------------------------------------------------- credit gauge
    def credit_estimate(self, credits_per_1m: dict[str, float | None]) -> float | None:
        """Estimate credit burn given per-model credits-per-1M-token rates.

        Only the two configured models map to a rate by default; any phase run
        on another model (or with no rate configured) yields ``None`` so we
        don't report a false guarantee.
        """
        known = {m: r for m, r in credits_per_1m.items() if r is not None}
        if not known:
            return None
        tokens_by_model: Counter[str] = Counter()
        for entry in self.entries:
            tokens_by_model[entry.model] += entry.prompt_tokens + entry.completion_tokens
        credits = 0.0
        for model, tokens in tokens_by_model.items():
            rate = known.get(model)
            if rate is None:
                return None  # unrated model => whole estimate unknown
            credits += tokens / 1_000_000.0 * rate
        return credits

    # ------------------------------------------------------------- reporting
    def live_line(self) -> str:
        """Short single-line status shown in the progress caption."""
        total = self.totals()
        if not total.calls:
            return ""
        return (
            f" · est {total.prompt_tokens:,} in / {total.completion_tokens:,} out "
            f"tokens ({total.calls} call{'s' if total.calls != 1 else ''})"
        )

    def summary(self) -> dict[str, Any]:
        """Structured totals for post-run display."""
        total = self.totals()
        return {
            "calls": total.calls,
            "prompt_tokens": total.prompt_tokens,
            "completion_tokens": total.completion_tokens,
            "total_tokens": total.total_tokens,
        }