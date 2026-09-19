"""Usage view helpers: watchdog persistence, credit rates, usage summary.

Split out of ``app/views/shared.py``; the estimator itself is
``app.usage`` and the learning logic is ``app.watchdog``.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from .. import config
from .. import watchdog
from ..logging_config import get_logger
from ..usage import UsageTracker

logger = get_logger("app.views.usage")


def load_usage_history() -> watchdog.UsageHistory:
    """Load cached usage history; never fails (empty on first run)."""
    try:
        return watchdog.load_history()
    except Exception:  # noqa: BLE001 - keep the app usable on any I/O error
        return watchdog.UsageHistory()


def save_usage_history(history: watchdog.UsageHistory) -> None:
    """Persist usage history; swallow I/O errors so the app never breaks on them."""
    try:
        watchdog.save_history(history)
    except Exception:  # noqa: BLE001
        pass


def record_watchdog_run(usage: Any) -> None:
    """Append a finished run's token totals to the persisted history.

    Per-role token totals (main vs fast) are carried along so the watchdog can
    fit separate credit rates for the two models.
    """
    total = usage.totals()
    if not total.calls:
        return
    history = load_usage_history()
    watchdog.record_run(
        history,
        prompt_tokens=total.prompt_tokens,
        completion_tokens=total.completion_tokens,
        calls=total.calls,
        by_role=usage.per_role_tokens(),
    )
    save_usage_history(history)


def effective_credit_rates() -> tuple[dict[str, float | None], str]:
    """Resolve credits-per-1M rates for the estimator.

    Prefers explicitly-configured env/per-model rates; otherwise falls back to the
    effective blended rate learned by the watchdog (if it has enough data).
    Returns (rates_by_model, source_label).
    """
    rates = {
        config.DEFAULT_MODEL_MAIN: config.CREDITS_PER_1M_MAIN,
        config.DEFAULT_MODEL_FAST: config.CREDITS_PER_1M_FAST,
    }
    if all(value is not None for value in rates.values()):
        return rates, "configured in .env"
    fit = watchdog.fit_effective_rate(load_usage_history())
    if fit.any_rate():  # pragma: no branch - guarded
        label = "estimated by the usage watchdog"
        # Fill only the rates missing from .env; never overwrite an explicit
        # value the user configured for one model.
        if rates[config.DEFAULT_MODEL_MAIN] is None and fit.main_rate is not None:
            rates[config.DEFAULT_MODEL_MAIN] = fit.main_rate
        if rates[config.DEFAULT_MODEL_FAST] is None and fit.fast_rate is not None:
            rates[config.DEFAULT_MODEL_FAST] = fit.fast_rate
        return rates, label
    return rates, "entry"


def render_usage_summary(usage: Any) -> None:
    """Render an estimated per-phase usage breakdown after a successful run."""
    if usage is None:
        return
    total = usage.totals()
    if not total.calls:
        return

    # Shown outside (and above) the collapsed expander on purpose: a failover means
    # a different model wrote part of this document, and for a document that goes
    # into a claim file that is something the reader should not have to open a
    # details panel to discover. The audit record is the durable record of it.
    if getattr(usage, "used_fallback", False):
        st.warning(
            "⚠️ Part of this run was served by the **backup LLM endpoint**: the primary "
            "endpoint was unavailable. The result is complete and reviewed the same way, "
            "but its wording may differ from a normal run. This run's audit record "
            "(`llm_endpoints`) and the run log name the endpoint(s) that produced it."
        )

    with st.expander("⚙️ Estimated API usage (tokens / calls)", expanded=False):
        rows = []
        for phase, stats in usage.per_phase().items():
            models = ", ".join(f"{m}×{c}" for m, c in stats.models.items())
            rows.append(
                {
                    "Phase": phase,
                    "Calls": stats.calls,
                    "Est. input tokens": stats.prompt_tokens,
                    "Est. output tokens": stats.completion_tokens,
                    "Models (calls)": models,
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)
        st.caption(
            f"**Total:** {total.calls} call(s) · "
            f"{total.prompt_tokens:,} input / {total.completion_tokens:,} output tokens "
            f"({total.total_tokens:,} total). Token counts are estimates based on prompt "
            "length and model output; they use the provider's reported usage when available."
        )

        rates, rate_source = effective_credit_rates()
        credits = usage.credit_estimate(rates)
        quota = config.CREDIT_QUOTA
        if credits is not None:
            if quota:
                pct = credits / quota * 100
                st.caption(
                    f"**Estimated credit burn:** {credits:,.0f} / {quota:,.0f} "
                    f"({pct:.1f}% of the weekly quota, {rate_source})"
                )
            else:
                st.caption(
                    f"**Estimated credit burn:** {credits:,.0f} units ({rate_source}). "
                    "Set `VA_LSE_CREDIT_QUOTA` to render this as a percentage of your "
                    "plan's weekly allowance."
                )
        else:
            st.caption(
                "Set `VA_LSE_CREDITS_PER_1M_MAIN` / `VA_LSE_CREDITS_PER_1M_FAST` in your "
                ".env — or add console readings in the sidebar's **Usage watchdog** "
                "panel so the app can estimate your rate — to see credit burn here."
            )


__all__ = [
    "UsageTracker",
    "effective_credit_rates",
    "load_usage_history",
    "record_watchdog_run",
    "render_usage_summary",
    "save_usage_history",
]
