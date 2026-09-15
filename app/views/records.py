"""Records view helpers: source widget (Upload / Fetch / VA.gov / Local).

Split out of ``app/views/shared.py`` so record-source handling is
independently navigable. Extraction itself stays in ``app.documents``; this
module is the Streamlit skin over it plus session-state bookkeeping.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from .. import config
from .. import telemetry
from .. import va_gov_client
from ..documents import ExtractionError, records_from_local_path
from ..fetch_client import FetchClient, FetchSandboxError
from ..logging_config import get_logger
from .uploads import check_upload_limits, extract_uploads

logger = get_logger("app.views.records")


def is_local_run() -> bool:
    """True when the app is served on the same machine as the browser.

    Used to gate reading records directly from the local filesystem: that is
    only safe for a local Streamlit run, never for a public deployment (where
    it would let anyone read server files). Fall back to an explicit opt-in
    env var for unusual local setups.
    """
    import os
    from urllib.parse import urlparse

    if os.getenv("VA_LSE_ALLOW_LOCAL_PATHS", "").strip() == "1":
        return True
    try:
        context = st.context
        host = (context.headers.get("Host") or "").split(":")[0].lower()
        if host in ("localhost", "127.0.0.1", "::1"):
            return True
        url_host = urlparse(context.url or "").hostname or ""
        return url_host in ("localhost", "127.0.0.1", "::1")
    except Exception:  # noqa: BLE001 - context may be unavailable in tests
        return False


def _track_selector_impression(slot: str) -> None:
    """Fire one impression event per Streamlit session per workflow slot."""
    flag_key = f"va_gov_selector_impression_{slot}"
    if st.session_state.get(flag_key):
        return
    st.session_state[flag_key] = True
    telemetry.track_impression(
        va_gov_client.FEATURE_ID,
        entry_point=slot,
        selector_rendered=True,
        workflow=slot,
    )


def remember_source_records(slot: str, label: str, docs: list) -> None:
    """Track the most recent successful load per source for this slot/session.

    Used to merge VA.gov records with whatever other sources the user has
    already loaded in this workflow slot this session (FR4/FR6), without
    changing the existing single-select source radio into a multi-select.
    """
    if not docs:
        return
    store = st.session_state.setdefault(f"source_records_{slot}", {})
    store[label] = docs


def records_uploader(slot: str) -> list:
    """Render the record-source widget for a slot and return loaded documents."""
    _track_selector_impression(slot)
    sources = ["Upload files", "Fetch Sandbox", "VA.gov"]
    if is_local_run():
        sources.append("Local folder / file")
    source = st.radio(
        "Medical record source",
        sources,
        horizontal=True,
        key=f"records_source_{slot}",
    )
    if source == "Fetch Sandbox":
        return _fetch_records(slot)
    if source == "VA.gov":
        return _va_gov_records(slot)
    if source == "Local folder / file":
        return _local_records(slot)

    files = st.file_uploader(
        "Upload medical records (PDF, TXT, MD, DOCX — multiple allowed)",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        key=f"files_{slot}",
    )
    # Enforce upload size caps before extraction so the tight 50 MB default
    # surfaces as a clear message rather than a downstream failure.
    if files:
        accepted, rejections = check_upload_limits(files)
        for msg in rejections:
            st.warning(msg)
        files = accepted if rejections else files
    documents = extract_uploads(files, slot)
    total_pages = sum(len(d.pages) for d in documents)
    if documents and total_pages > config.MAX_RECORD_PAGES:
        st.error(
            f"Record set is {total_pages:,} pages, which exceeds the configured limit of "
            f"{config.MAX_RECORD_PAGES:,} pages. Remove some files or raise "
            "VA_LSE_MAX_RECORD_PAGES."
        )
        return []
    remember_source_records(slot, "Upload", documents)
    return documents


def _local_records(slot: str) -> list:
    """Load record files straight from a path on the local machine."""
    st.caption(
        "Reads supported record files (.pdf/.txt/.md/.docx) directly from this "
        "machine's filesystem. Only available when the app runs locally."
    )
    path = st.text_input(
        "Folder or file path",
        key=f"local_path_{slot}",
        placeholder="e.g. ~/Desktop/ClaimRecords or ~/Desktop/records.pdf",
    )
    import_key = f"local_records_{slot}"
    skipped_key = f"local_records_skipped_{slot}"
    if st.button("Load records from path", key=f"local_load_{slot}"):
        st.session_state.pop(import_key, None)
        st.session_state.pop(skipped_key, None)
        try:
            records, skipped = records_from_local_path(path)
        except ExtractionError as exc:
            st.warning(str(exc))
        else:
            st.session_state[import_key] = records
            if skipped:
                st.session_state[skipped_key] = skipped
    for message in st.session_state.get(skipped_key, []):
        st.warning(message)
    cached_any: Any = st.session_state.get(import_key, [])
    cached_records: list[Any] = cached_any if isinstance(cached_any, list) else []
    remember_source_records(slot, "Local folder / file", cached_records)
    return cached_records


def _fetch_records(slot: str) -> list[Any]:
    settings = st.session_state.settings
    patient_id = st.text_input(
        "Patient or record ID",
        key=f"fetch_patient_id_{slot}",
        help="Used for the {patient_id} placeholder in the Fetch records path.",
    )
    st.caption(
        f"GET {settings.fetch_base_url.rstrip('/')}{settings.fetch_records_path}"
    )
    import_key = f"fetch_records_{slot}"
    if st.button("Import medical records from Fetch Sandbox", key=f"fetch_import_{slot}"):
        st.session_state.pop(import_key, None)
        try:
            records = FetchClient(settings).fetch_documents(patient_id)
        except FetchSandboxError as exc:
            st.warning(str(exc))
        else:
            st.session_state[import_key] = records
    records_any2: Any = st.session_state.get(import_key, [])
    records2: list[Any] = records_any2 if isinstance(records_any2, list) else []
    remember_source_records(slot, "Fetch Sandbox", records2)
    return records2


def _va_gov_records(slot: str) -> list[Any]:
    """VA.gov source: per-session secure login + consent, then automatic fetch.

    Failure to authenticate or fetch never blocks the other record sources —
    switching the radio back to Upload/Fetch Sandbox/Local always works
    regardless of VA.gov state.
    """
    st.caption(
        "⚠️ Fetched documents are sent to the configured LLM endpoint for analysis. "
        "Review privacy before fetching sensitive records — the same warning shown for "
        "every other record source applies equally here."
    )
    configured = va_gov_client.va_gov_configured()
    st.caption(
        ("🔌 **Records API mode** — talking to `VA_GOV_API_BASE_URL`. This is a "
         "sandbox/simulator path: real VA.gov access is ID.me + SMS-MFA protected and "
         "exposes no patient-facing records API.")
        if configured
        else ("🧪 **Sandbox mode** — `VA_GOV_API_BASE_URL` is unset, so this fetches "
              "deterministic mock records. Real VA.gov cannot be reached this way "
              "(ID.me + SMS MFA, no patient-facing records API), so a configured base "
              "URL here is always a sandbox or simulator — use sandbox credentials, "
              "never your real VA.gov password.")
    )
    st.caption(
        "For **your own** records, download them from VA.gov instead: "
        "`scripts/va_records_download.py` automates the download wizard locally "
        "(you sign in and enter the SMS code yourself), or follow "
        "**My Health → Medical records → Download** by hand and upload the PDF as an "
        "*Upload files* source. See `README.md → VA.gov record source`."
    )
    st.caption(
        "Signs in for this session only, then automatically fetches the available "
        "records. Credentials are never stored to disk or `.env`; records stay local "
        "after fetch."
    )
    consent = st.checkbox(
        "I consent to fetching my medical records for this session only. "
        "Records are not stored beyond this session and my credentials are never saved.",
        key=f"va_gov_consent_{slot}",
    )
    authed = bool(st.session_state.get(f"va_gov_authed_{slot}"))
    with st.expander("🔒 VA.gov secure login", expanded=not authed):
        st.caption(
            "Sandbox/simulator credentials — not your VA.gov password."
        )
        username = st.text_input("Sandbox username", key=f"va_gov_user_{slot}")
        password = st.text_input(
            "Sandbox password", type="password", key=f"va_gov_pass_{slot}"
        )
        login_clicked = st.button(
            "Log in and fetch VA.gov records",
            key=f"va_gov_login_{slot}",
        )

    if login_clicked:
        telemetry.track_interaction(
            va_gov_client.FEATURE_ID,
            {
                "source_selected": "va_gov",
                "login_modal_opened": True,
                "consent_given": consent,
            },
        )
        if not consent:
            st.warning("Consent is required before VA.gov records can be fetched.")
        else:
            try:
                session = va_gov_client.authenticate_va_gov(username, password)
            except va_gov_client.VaGovError as exc:
                st.error(str(exc))
            else:
                st.session_state[f"va_gov_session_{slot}"] = session
                st.session_state[f"va_gov_authed_{slot}"] = True
                st.session_state[f"va_gov_fetch_{slot}"] = va_gov_client.fetch_va_records(session)
                st.session_state.pop(f"va_gov_accept_partial_{slot}", None)

    result = st.session_state.get(f"va_gov_fetch_{slot}")
    if result is None:
        return []

    if result.error_message:
        st.error(
            f"VA.gov fetch problem: {result.error_message} "
            f"(retrieved {result.retrieved} of {result.expected} expected record(s))."
        )
        col_retry, col_continue = st.columns(2)
        if col_retry.button("Retry VA.gov fetch", key=f"va_gov_retry_{slot}"):
            session_any: Any = st.session_state.get(f"va_gov_session_{slot}")
            if session_any is not None:
                st.session_state[f"va_gov_fetch_{slot}"] = va_gov_client.fetch_va_records(session_any)
                st.rerun()
        if col_continue.button(
            "Continue with available VA.gov records", key=f"va_gov_continue_{slot}"
        ):
            st.session_state[f"va_gov_accept_partial_{slot}"] = True

    if not result.documents:
        return []
    if result.error_message and not st.session_state.get(f"va_gov_accept_partial_{slot}"):
        return []

    remember_source_records(slot, "VA.gov", result.documents)
    other_sources = dict(st.session_state.get(f"source_records_{slot}", {}))
    merged = va_gov_client.merge_records(other_sources)

    st.subheader("Merged records summary")
    st.dataframe(
        [
            {"Source": row.source, "File": row.filename, "Pages": row.pages}
            for row in merged.summary
        ],
        width="stretch",
        hide_index=True,
    )
    confirmed = st.checkbox(
        "I confirm this merged record set is correct and want to proceed.",
        key=f"va_gov_confirm_{slot}",
    )
    telemetry.track_interaction(
        va_gov_client.FEATURE_ID,
        {
            "records_fetched": result.retrieved,
            "records_expected": result.expected,
            "sources_merged": merged.sources_merged,
            "confirmed": confirmed,
        },
    )
    if not confirmed:
        return []
    return merged.documents


__all__ = [
    "is_local_run",
    "records_uploader",
    "remember_source_records",
]
