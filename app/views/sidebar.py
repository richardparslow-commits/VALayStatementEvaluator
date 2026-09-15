"""Sidebar view: LLM settings, Fetch Sandbox settings, usage-watchdog widget.

Extracted from ``app/main.py`` so the entry point stays a thin router. The
settings object itself lives in ``st.session_state.settings`` and is mutated
in place exactly as before — only the rendering moved here.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from .. import config
from .. import watchdog
from ..config import DEFAULT_BASE_URL, load_settings
from ..llm import check_model_availability
from ..logging_config import get_logger, get_request_id
from ..prompt_sanitize import validate_api_key, validate_model_name
from .usage import load_usage_history, save_usage_history

logger = get_logger("app.views.sidebar")


def render_sidebar_settings() -> None:
    """Render the LLM/Fetch settings sidebar, mutating session settings in place."""
    if "settings" not in st.session_state:
        st.session_state.settings = load_settings()
    settings = st.session_state.settings

    with st.sidebar:
        st.title("⚙️ LLM Settings")
        st.session_state.api_key_input = st.text_input(
            "API key",
            value=settings.api_key,
            type="password",
            help="Stored only in this browser session and used for LLM calls.",
        )
        st.session_state.base_url_input = st.text_input(
            "Base URL (OpenAI-compatible)", value=settings.base_url or DEFAULT_BASE_URL
        )
        col1, col2 = st.columns(2)
        st.session_state.model_main_input = col1.text_input(
            "Main model", value=settings.model_main, help="Analysis, scoring, drafting"
        )
        st.session_state.model_fast_input = col2.text_input(
            "Fast model",
            value=settings.model_fast,
            help="Bulk record digests — use a cheap model here to preserve "
            "QwenCloud Lite quota.",
        )
        st.divider()
        st.subheader("Fetch Sandbox")
        st.session_state.fetch_api_key_input = st.text_input(
            "Fetch API key",
            value=settings.fetch_api_key,
            type="password",
            help="Optional if the sandbox runs in relaxed auth mode.",
        )
        st.session_state.fetch_base_url_input = st.text_input(
            "Fetch base URL", value=settings.fetch_base_url
        )
        st.session_state.fetch_records_path_input = st.text_input(
            "Fetch records path",
            value=settings.fetch_records_path,
            help="GET path for records. Use {patient_id} where the selected ID belongs.",
        )
        if st.button("Apply settings"):
            api_key_val = st.session_state.api_key_input.strip()
            fetch_key_val = st.session_state.fetch_api_key_input.strip()
            model_main_val = st.session_state.model_main_input.strip()
            model_fast_val = st.session_state.model_fast_input.strip()
            errors: list[str] = []
            for label, val, validator in (
                ("API key", api_key_val, validate_api_key),
                ("Fetch API key", fetch_key_val, validate_api_key),
                ("Main model", model_main_val, validate_model_name),
                ("Fast model", model_fast_val, validate_model_name),
            ):
                msg = validator(val)
                if msg:
                    errors.append(f"{label}: {msg}")
            if errors:
                for msg in errors:
                    st.error(msg)
            else:
                settings.api_key = api_key_val
                settings.base_url = st.session_state.base_url_input.strip() or DEFAULT_BASE_URL
                settings.model_main = model_main_val
                settings.model_fast = model_fast_val
                settings.fetch_api_key = fetch_key_val
                settings.fetch_base_url = st.session_state.fetch_base_url_input.strip()
                settings.fetch_records_path = st.session_state.fetch_records_path_input.strip()
                st.rerun()

        _compat_model_warning(settings)

        st.divider()
        _credit_calibration_widget()
        st.divider()
        st.caption(
            "⚠️ Uploaded documents are sent to the configured LLM endpoint for analysis. "
            "Review privacy before uploading sensitive records."
        )
        st.caption(
            "This tool is an aid for drafting and reviewing lay statements. It is not "
            "legal, medical, or claims advice."
        )


def _compat_model_warning(settings: Any) -> None:
    """Warn if the configured models are not listed at GET {base_url}/models.

    Advisory only: network/permission failures are silently ignored and the
    warning is cached per-session so the endpoint is not hit on every rerun.
    """
    sig = f"{settings.base_url}|{settings.model_main}|{settings.model_fast}"
    if st.session_state.get("_compat_checked_sig") == sig:
        for msg in st.session_state.get("_compat_warnings", []):
            st.warning(msg)
        return
    try:
        available = check_model_availability(settings.base_url, settings.api_key)
    except Exception:  # noqa: BLE001 - never break the UI on a compat check
        available = None
    warnings: list[str] = []
    if available is not None:
        for label, model in (
            ("Main model", settings.model_main),
            ("Fast model", settings.model_fast),
        ):
            if model and model not in available:
                warnings.append(
                    f"⚠️ {label} `{model}` not found at `{settings.base_url.rstrip('/')}/models`. "
                    "The provider may have deprecated it — check `COMPATIBILITY.md` and `MIGRATION.md`."
                )
    st.session_state["_compat_checked_sig"] = sig
    st.session_state["_compat_warnings"] = warnings
    for msg in warnings:
        logger.warning(
            "model availability warning: %s",
            msg,
            extra={
                "request_id": get_request_id() or "-",
                "phase": "compat",
                "status": "warning",
            },
        )
        st.warning(msg)


def _credit_calibration_widget() -> None:
    """Sidebar: record console readings and surface the learned rate."""
    with st.expander("🎚️ Usage watchdog (credit rate)"):
        history = load_usage_history()
        fit = watchdog.fit_effective_rate(history)

        last_credits = st.session_state.get("watchdog_last_credits", "")
        credits = st.text_input(
            "Total credits used (from QwenCloud console)",
            value=last_credits,
            key="watchdog_credits_input",
        )
        captured = False
        if st.button("Record this reading"):
            try:
                parsed = float(credits)
                if parsed < 0:
                    raise ValueError
                watchdog.record_calibration(history, credits=parsed)
                save_usage_history(history)
                st.session_state["watchdog_last_credits"] = credits
                captured = True
            except (TypeError, ValueError):
                st.warning("Enter a non-negative number for credits used.")

        n_runs = len(history.runs)
        if captured:
            st.success(
                f"Reading saved ({n_runs} run(s) recorded). Repeat after more runs to refine."
            )

        st.caption(
            f"Runs tracked: {n_runs} · calibrations: {len(history.calibrations)}"
        )
        if fit.any_rate():
            separate = fit.main_rate != fit.fast_rate
            if separate:
                st.markdown(
                    f"**Learned rates:** main ≈{fit.main_rate:,.0f} · fast "
                    f"≈{fit.fast_rate:,.0f} credits/1M tokens. {fit.multiline_note}"
                )
            else:
                st.markdown(
                    f"**Learned effective rate:** ≈{fit.blended_rate:,.0f} credits / 1M "
                    f"tokens {fit.multiline_note}"
                )
            enabled = config.CREDITS_PER_1M_MAIN is None and config.CREDITS_PER_1M_FAST is None
            if enabled:
                rate_desc = (
                    f"main **{fit.main_rate:,.0f}** / fast **{fit.fast_rate:,.0f} credits/1M**"
                    if separate
                    else f"**{fit.blended_rate:,.0f} credits/1M**"
                )
                st.markdown(
                    f"The estimator will now use {rate_desc} as a fallback for the "
                    "credit estimate until you set explicit rates in `.env`."
                )
        else:
            st.markdown(
                "Add the total credits your plan reports each time after a run. Once you've "
                "recorded at least two readings separated by new runs, the app fits your "
                "effective credits-per-1M rate and starts estimating credit burn."
            )
            st.caption(
                "Tip: post each eval/draft run's totals (shown here) and your console's "
                "cumulative credits to converge in a few runs."
            )
