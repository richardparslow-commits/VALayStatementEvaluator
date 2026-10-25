"""Records view helpers: source widget (Upload / Fetch / VA.gov / Local).

Split out of ``app/views/shared.py`` so record-source handling is
independently navigable. Extraction itself stays in ``app.documents``; this
module is the Streamlit skin over it plus session-state bookkeeping.
"""
from __future__ import annotations

from .. import pilot

import re
from typing import Any

import streamlit as st

from .. import config
from .. import local_paths
from .. import telemetry
from .. import va_gov_client
from .. import va_gov_export
from ..agiloop_telemetry import track_feature_error, track_impression, track_interaction
from ..documents import (
    ExtractionError,
    SearchResult,
    export_citation_index,
    page_limit_message,
    records_from_local_path,
    search_records,
)
from ..error_report import report_failure
from ..fetch_client import FetchClient, FetchSandboxError
from .uploads import check_upload_limits, extract_uploads, render_record_volume_warning

# Feature: Medical Record Search & Citation Index
SEARCH_FEATURE_ID = "22bc7e10-dcda-431e-b3fb-4e8ff9b532cb"  # medical-record-search-citation-index
CITATION_INDEX_KEY = "citation_index"


def is_local_run() -> bool:
    """Whether the server explicitly permits local record imports.

    Enable this only for a trusted, single-user installation bound to loopback:
    the flag alone is not enough. The bind check reads the OS socket table
    via ``app/local_paths.py`` — the bind address of the socket identified as
    the Streamlit UI (Streamlit-marker probe over loopback; the health
    sidecar's own wide bind is that subsystem's contract, not the gate's
    concern) — with the committed ``.streamlit/config.toml`` as fallback. An
    address absent from both means the UI bind cannot be proven loopback, so
    local imports stay disabled, warning once with the restart command that
    fixes it.
    """
    import os

    if pilot.enabled() or os.getenv("VA_LSE_ALLOW_LOCAL_PATHS", "").strip() != "1":
        return False
    address = local_paths.server_bind_address()
    if local_paths.is_loopback(address):
        return True
    _warn_loopback_required(address)
    return False


def _warn_loopback_required(address: Any) -> None:
    """Explain, once per session, why the local-import flag is not taking effect."""
    if st.session_state.get("local_paths_loopback_warned"):
        return
    st.session_state["local_paths_loopback_warned"] = True
    shown = (
        "unset, so Streamlit is listening on every interface"
        if address in (None, "")
        else f"set to {address!r}"
    )
    pilot.display(
        "VA_LSE_ALLOW_LOCAL_PATHS=1 is set, but the server is not bound to the "
        f"loopback interface (server.address is {shown}). Records read from local "
        "paths could then be requested by any machine that can reach this server, "
        "so local imports stay disabled. Restart bound to loopback: "
        "`streamlit run run_app.py --server.address 127.0.0.1`"
    , container=st, method="warning")


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


def _forget_source_records(slot: str, label: str) -> None:
    """Remove cached source documents for ``label`` when cleared or invalid."""
    store = st.session_state.get(f"source_records_{slot}")
    if isinstance(store, dict):
        store.pop(label, None)


def _reconcile_search_and_citations(slot: str, documents: list[Any]) -> None:
    """Prune stale search results and citation_index entries when records change."""
    from ..exporter import build_record_excerpt_matcher, filter_grounded_citations

    matcher = build_record_excerpt_matcher(documents or [])
    if matcher is None:
        # Unit-test MagicMock placeholder documents: do not filter by content.
        return

    results_key = f"search_results_{slot}"
    results_any: Any = st.session_state.get(results_key)
    if isinstance(results_any, list):
        kept_results = []
        for r in results_any:
            fname = getattr(r, "filename", None)
            if not isinstance(fname, str) or not fname:
                if documents:
                    kept_results.append(r)
                continue
            r_page = getattr(r, "page", None)
            r_excerpt = str(getattr(r, "excerpt", "") or "")
            if matcher(fname, r_page if isinstance(r_page, int) else None, r_excerpt):
                kept_results.append(r)
        if len(kept_results) != len(results_any):
            st.session_state[results_key] = kept_results

    if slot == "eval" or not st.session_state.get("source_records_eval"):
        index_any: Any = st.session_state.get(CITATION_INDEX_KEY)
        if isinstance(index_any, list) and index_any:
            valid_dicts = [item for item in index_any if isinstance(item, dict)]
            kept_citations = filter_grounded_citations(valid_dicts, documents or [])
            if len(kept_citations) != len(index_any):
                st.session_state[CITATION_INDEX_KEY] = kept_citations
                eval_res = st.session_state.get("eval_result")
                if eval_res is not None and isinstance(getattr(eval_res, "report_citations", None), list):
                    eval_res.report_citations = list(kept_citations)


def records_uploader(slot: str) -> list:
    """Render the record-source widget for a slot and return loaded documents."""
    _track_selector_impression(slot)
    sources = ["Upload files"] if pilot.enabled() else ["Upload files", "Fetch Sandbox", "VA.gov"]
    if is_local_run():
        sources.append("Local folder / file")
    source = st.radio(
        "Medical record source",
        sources,
        horizontal=True,
        key=f"records_source_{slot}",
    )
    if pilot.enabled() and source != "Upload files":
        pilot.display("Only uploads are enabled for this pilot.", container=st, method="error")
        return []
    if pilot.enabled():
        consent = st.checkbox("I have authority to use these records and consent to sending "
                              "their text and my statement to the approved analysis provider. "
                              "I understand its retention policy and the pilot privacy notice.",
                              key=f"pilot_consent_{slot}")
        if not consent:
            return []
    if source == "Fetch Sandbox":
        return _fetch_records(slot)
    if source == "VA.gov":
        return _va_gov_records(slot)
    if source == "Local folder / file":
        return _local_records(slot)

    files = st.file_uploader(
        "Upload medical records (PDF, TXT, MD, DOCX, or a ZIP of them — multiple allowed)",
        type=["pdf", "txt", "md", "docx", "zip"],
        accept_multiple_files=True,
        key=f"files_{slot}",
        help="Upload one file at a time. A pilot case can retain up to 32 uploaded files and record versions. "
             "Use passive documents and UTF-8 text. PDF/Word files with scripts, attachments, macros, or external links "
             "are refused; keep the original and upload a passive copy."
             if pilot.enabled() else None,
    )
    # Enforce upload size caps before extraction so the tight 50 MB default
    # surfaces as a clear message rather than a downstream failure.
    if files:
        accepted, rejections = check_upload_limits(files)
        for msg in rejections:
            pilot.display(
                report_failure(
                    msg, phase="upload_limits", severity="warning", once=True
                )
            , container=st, method="warning")
        if pilot.enabled() and rejections:
            pilot.display("Resolve every selected file before starting this case.", container=st, method="error")
            return []
        files = accepted if rejections else files
    documents = extract_uploads(files, slot)
    # Pages the files contain, not pages that yielded text: a scan-only bundle has
    # very few readable pages and must not look like a small record set.
    total_pages = sum(d.source_page_count for d in documents)
    if documents and total_pages > config.MAX_RECORD_PAGES:
        pilot.display(page_limit_message(total_pages, config.MAX_RECORD_PAGES), container=st, method="error")
        _forget_source_records(slot, "Upload")
        _label_detected_va_gov_exports(slot, [])
        _reconcile_search_and_citations(slot, [])
        return []
    render_record_volume_warning(documents, slot=slot)
    if documents:
        remember_source_records(slot, "Upload", documents)
    else:
        _forget_source_records(slot, "Upload")
    _label_detected_va_gov_exports(slot, documents)
    _reconcile_search_and_citations(slot, documents)
    return documents


def _label_detected_va_gov_exports(slot: str, documents: list[Any]) -> None:
    """Record uploaded VA.gov exports under the VA.gov source, and say so.

    Downloading the report from VA.gov and uploading the PDF is the supported way
    to get a real record set, so recognising it keeps the merged records summary
    honest about where those pages came from.
    """
    exports, names = va_gov_export.split_va_gov_exports(documents)
    st.session_state[f"_uploaded_va_gov_exports_{slot}"] = list(exports)
    if not exports:
        if not st.session_state.get(f"va_gov_fetch_{slot}"):
            st.session_state.get(f"source_records_{slot}", {}).pop("VA.gov", None)
        return
    remember_source_records(slot, "VA.gov", exports)
    pilot.display(
        f"🔎 Detected a VA.gov medical-records export ({', '.join(names)}) — labelled as "
        "the **VA.gov** source so the merged records summary shows where it came from."
    , container=st, method="caption")


def _safe_doc_pages(documents: list[Any]) -> int:
    total = 0
    for d in documents:
        spc = getattr(d, "source_page_count", None)
        if type(spc) is int:
            total += spc
        else:
            pages = getattr(d, "pages", None)
            if isinstance(pages, (list, tuple)):
                total += len(pages)
    return total


def _local_records(slot: str) -> list:
    """Load record files straight from a path on the local machine."""
    # Enforce the capability at the handler, including stale source selections.
    if not is_local_run():
        pilot.display("Local file access is disabled on this server. Upload records instead.", container=st, method="error")
        return []
    pilot.display(
        "Reads supported record files (.pdf/.txt/.md/.docx/.zip) directly from "
        "this machine's filesystem. Available only when "
        "VA_LSE_ALLOW_LOCAL_PATHS=1 is set, for trusted single-user use."
    , container=st, method="caption")
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
            pilot.display(
                report_failure(
                    str(exc), phase="local_records_load", exc=exc, severity="warning"
                )
            , container=st, method="warning")
        else:
            st.session_state[import_key] = records
            if skipped:
                st.session_state[skipped_key] = skipped
    # Replayed from session state on every rerun; ``once`` keeps a file that keeps
    # failing from rewriting the same log line on each interaction.
    for message in st.session_state.get(skipped_key, []):
        pilot.display(
            report_failure(
                message, phase="local_records_extract", severity="warning", once=True
            )
        , container=st, method="warning")
    cached_any: Any = st.session_state.get(import_key, [])
    cached_records: list[Any] = cached_any if isinstance(cached_any, list) else []
    total_pages = _safe_doc_pages(cached_records)
    if cached_records and total_pages > config.MAX_RECORD_PAGES:
        pilot.display(page_limit_message(total_pages, config.MAX_RECORD_PAGES), container=st, method="error")
        _forget_source_records(slot, "Local folder / file")
        _reconcile_search_and_citations(slot, [])
        return []
    render_record_volume_warning(cached_records, slot=slot)
    if cached_records:
        remember_source_records(slot, "Local folder / file", cached_records)
    else:
        _forget_source_records(slot, "Local folder / file")
    _reconcile_search_and_citations(slot, cached_records)
    return cached_records


def _fetch_records(slot: str) -> list[Any]:
    if pilot.enabled():
        raise pilot.PilotBlocked("Remote fetching is outside this pilot profile.")
    settings = st.session_state.settings
    patient_id = st.text_input(
        "Patient or record ID",
        key=f"fetch_patient_id_{slot}",
        help="Used for the {patient_id} placeholder in the Fetch records path.",
    )
    pilot.display(
        f"GET {settings.fetch_base_url.rstrip('/')}{settings.fetch_records_path}"
    , container=st, method="caption")
    import_key = f"fetch_records_{slot}"
    if st.button("Import medical records from Fetch Sandbox", key=f"fetch_import_{slot}"):
        st.session_state.pop(import_key, None)
        try:
            records = FetchClient(settings).fetch_documents(patient_id)
        except FetchSandboxError as exc:
            pilot.display(
                report_failure(
                    str(exc), phase="fetch_records_import", exc=exc, severity="warning"
                )
            , container=st, method="warning")
        else:
            st.session_state[import_key] = records
    records_any2: Any = st.session_state.get(import_key, [])
    records2: list[Any] = records_any2 if isinstance(records_any2, list) else []
    total_pages = _safe_doc_pages(records2)
    if records2 and total_pages > config.MAX_RECORD_PAGES:
        pilot.display(page_limit_message(total_pages, config.MAX_RECORD_PAGES), container=st, method="error")
        _forget_source_records(slot, "Fetch Sandbox")
        _reconcile_search_and_citations(slot, [])
        return []
    render_record_volume_warning(records2, slot=slot)
    if records2:
        remember_source_records(slot, "Fetch Sandbox", records2)
    else:
        _forget_source_records(slot, "Fetch Sandbox")
    _reconcile_search_and_citations(slot, records2)
    return records2


def _va_gov_records(slot: str) -> list[Any]:
    """VA.gov source: per-session secure login + consent, then automatic fetch.

    Failure to authenticate or fetch never blocks the other record sources —
    switching the radio back to Upload/Fetch Sandbox/Local always works
    regardless of VA.gov state.
    """
    pilot.display(
        "⚠️ Fetched documents are sent to the configured LLM endpoint for analysis. "
        "Review privacy before fetching sensitive records — the same warning shown for "
        "every other record source applies equally here."
    , container=st, method="caption")
    if pilot.enabled():
        raise pilot.PilotBlocked("Simulator login is outside this pilot profile.")
    configured = va_gov_client.va_gov_configured()
    pilot.display(
        ("🔌 **Records API mode** — talking to `VA_GOV_API_BASE_URL`. This is a "
         "sandbox/simulator path: real VA.gov access is ID.me + SMS-MFA protected and "
         "exposes no patient-facing records API.")
        if configured
        else ("🧪 **Sandbox mode** — `VA_GOV_API_BASE_URL` is unset, so this fetches "
              "deterministic mock records. Real VA.gov cannot be reached this way "
              "(ID.me + SMS MFA, no patient-facing records API), so a configured base "
              "URL here is always a sandbox or simulator — use sandbox credentials, "
              "never your real VA.gov password.")
    , container=st, method="caption")
    pilot.display(
        "For **your own** records, download them from VA.gov instead: "
        "`scripts/va_records_download.py` automates the download wizard locally "
        "(you sign in and enter the SMS code yourself), or follow "
        "**My Health → Medical records → Download** by hand and upload the PDF as an "
        "*Upload files* source. See `README.md → VA.gov record source`."
    , container=st, method="caption")
    pilot.display(
        "Signs in for this session only, then automatically fetches the available "
        "records. Credentials are never stored to disk or `.env`; records stay local "
        "after fetch."
    , container=st, method="caption")
    consent = st.checkbox(
        "I consent to fetching these simulator records for analysis. Queued runs and "
        "blob storage may retain copies until expiry; the analysis provider has its own retention policy.",
        key=f"va_gov_consent_{slot}",
    )
    user_key = f"va_gov_user_{slot}"
    user_backup_key = f"_va_gov_user_saved_{slot}"
    if user_key not in st.session_state and user_backup_key in st.session_state:
        st.session_state[user_key] = st.session_state[user_backup_key]
    authed = bool(st.session_state.get(f"va_gov_authed_{slot}"))
    with st.expander("🔒 VA.gov secure login", expanded=not authed):
        pilot.display(
            "Sandbox/simulator credentials — not your VA.gov password."
        , container=st, method="caption")
        username = st.text_input("Sandbox username", key=user_key)
        st.session_state[user_backup_key] = username
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
        st.session_state.pop(f"va_gov_fetch_{slot}", None)
        st.session_state.pop(f"va_gov_session_{slot}", None)
        st.session_state.pop(f"va_gov_authed_{slot}", None)
        st.session_state.pop(f"va_gov_accept_partial_{slot}", None)
        st.session_state.pop(f"va_gov_confirm_{slot}", None)
        uploaded_exports = list(st.session_state.get(f"_uploaded_va_gov_exports_{slot}", []) or [])
        if uploaded_exports:
            remember_source_records(slot, "VA.gov", uploaded_exports)
        else:
            st.session_state.get(f"source_records_{slot}", {}).pop("VA.gov", None)
        if not consent:
            pilot.display("Consent is required before VA.gov records can be fetched.", container=st, method="warning")
        else:
            try:
                session = va_gov_client.authenticate_va_gov(username, password)
            except va_gov_client.VaGovError as exc:
                pilot.display(report_failure(str(exc), phase="va_gov_auth", exc=exc), container=st, method="error")
            else:
                st.session_state[f"va_gov_session_{slot}"] = session
                st.session_state[f"va_gov_authed_{slot}"] = True
                st.session_state[f"va_gov_fetch_{slot}"] = va_gov_client.fetch_va_records(session)

    result = st.session_state.get(f"va_gov_fetch_{slot}")
    if result is None:
        return []

    has_fetch_problem = bool(result.partial or result.error_message)
    if has_fetch_problem:
        err_text = result.error_message or (
            f"Incomplete VA.gov record set: retrieved {result.retrieved} of {result.expected} expected record(s)."
        )
        if result.expected > 0:
            msg = f"VA.gov fetch problem: {err_text} (retrieved {result.retrieved} of {result.expected} expected record(s))."
        else:
            msg = f"VA.gov fetch problem: {err_text}"
        pilot.display(msg, container=st, method="error")
        col_retry, col_continue = st.columns(2)
        if col_retry.button("Retry VA.gov fetch", key=f"va_gov_retry_{slot}"):
            session_any: Any = st.session_state.get(f"va_gov_session_{slot}")
            if session_any is not None:
                st.session_state[f"va_gov_fetch_{slot}"] = va_gov_client.fetch_va_records(session_any)
                st.session_state.pop(f"va_gov_accept_partial_{slot}", None)
                st.session_state.pop(f"va_gov_confirm_{slot}", None)
                st.rerun()
        if result.documents and col_continue.button(
            "Continue with available VA.gov records", key=f"va_gov_continue_{slot}"
        ):
            st.session_state[f"va_gov_accept_partial_{slot}"] = True

    if not result.documents:
        return []
    if has_fetch_problem and not st.session_state.get(f"va_gov_accept_partial_{slot}"):
        return []

    uploaded_exports = list(st.session_state.get(f"_uploaded_va_gov_exports_{slot}", []) or [])
    combined_va_docs = list(uploaded_exports) + list(result.documents)
    remember_source_records(slot, "VA.gov", combined_va_docs)
    other_sources = dict(st.session_state.get(f"source_records_{slot}", {}))
    merged = va_gov_client.merge_records(other_sources)

    st.subheader("Merged records summary")
    pilot.dataframe(st,
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
    total_pages = _safe_doc_pages(merged.documents)
    if merged.documents and total_pages > config.MAX_RECORD_PAGES:
        pilot.display(page_limit_message(total_pages, config.MAX_RECORD_PAGES), container=st, method="error")
        return []
    render_record_volume_warning(merged.documents, slot=slot)
    if not confirmed:
        return []
    return merged.documents


# ------------------------------------------------------- record search + citations
def render_record_search(slot: str, documents: list[Any]) -> None:
    """Search widget over loaded medical records with date/provider filters.

    Renders once at least one record is loaded for the slot (F2.S1): a query
    box plus optional date-range/provider filters returns TF-IDF-ranked
    passages with highlighted matches and a source label. Clicking "Export"
    on a result adds its excerpt + source to ``citation_index`` in
    ``session_state`` (read by ``build_report``/CSV-JSON export in F2.S2).
    """
    _reconcile_search_and_citations(slot, documents)
    if not documents:
        return

    impression_key = f"search_impression_sent_{slot}"
    if not st.session_state.get(impression_key):
        try:
            track_impression(SEARCH_FEATURE_ID, entry_point=slot, records_loaded=len(documents))
        except Exception:  # noqa: BLE001 - telemetry must never break the UI
            pass
        st.session_state[impression_key] = True

    with st.expander("🔎 Search medical records", expanded=False):
        query = st.text_input("Search query", key=f"search_query_{slot}")
        col_from, col_to, col_provider = st.columns(3)
        date_from = col_from.date_input("From date", value=None, key=f"search_date_from_{slot}")
        date_to = col_to.date_input("To date", value=None, key=f"search_date_to_{slot}")
        provider = col_provider.text_input("Provider (optional)", key=f"search_provider_{slot}")

        results_key = f"search_results_{slot}"
        search_clicked = st.button("Search", key=f"search_submit_{slot}")
        if search_clicked:
            if not query.strip():
                st.session_state[results_key] = []
            else:
                try:
                    results = search_records(
                        documents,
                        query,
                        date_from=date_from or None,
                        date_to=date_to or None,
                        provider=provider or None,
                    )
                except Exception as exc:  # noqa: BLE001 - search must never crash the UI
                    try:
                        track_feature_error(SEARCH_FEATURE_ID, exc, stage="search")
                    except Exception:  # noqa: BLE001
                        pass
                    pilot.display(
                        report_failure(f"Search failed: {exc}", phase="record_search", exc=exc)
                    , container=st, method="error")
                    results = []
                st.session_state[results_key] = results
                try:
                    track_interaction(
                        SEARCH_FEATURE_ID,
                        action="search",
                        has_date_filter=bool(date_from or date_to),
                        has_provider_filter=bool(provider),
                        result_count=len(results),
                    )
                except Exception:  # noqa: BLE001 - telemetry must never break the UI
                    pass

        _reconcile_search_and_citations(slot, documents)
        results_any: Any = st.session_state.get(results_key, [])
        results_list: list[SearchResult] = results_any if isinstance(results_any, list) else []
        if not results_list:
            if query.strip():
                pilot.display("No matching passages found.", container=st, method="caption")
        else:
            _render_search_results(slot, results_list)

        _render_citation_index_export(slot)


def _sanitize_search_excerpt_for_display(excerpt: str) -> str:
    # Neutralize Markdown images ![alt](url) so record text cannot trigger browser fetches
    return re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"[image: \1]", excerpt)


def _render_search_results(slot: str, results: list[SearchResult]) -> None:
    for i, result in enumerate(results):
        pilot.display(f"**{result.label}**", container=st, method="markdown")
        pilot.display(_sanitize_search_excerpt_for_display(result.excerpt), container=st, method="markdown")
        if st.button("➕ Export excerpt", key=f"search_export_{slot}_{i}"):
            citation = {"excerpt": result.excerpt, "source": result.label}
            index_any: Any = st.session_state.setdefault(CITATION_INDEX_KEY, [])
            index: list[dict[str, str]] = index_any if isinstance(index_any, list) else []
            if citation not in index:
                index.append(citation)
            st.session_state[CITATION_INDEX_KEY] = index
            try:
                track_interaction(
                    SEARCH_FEATURE_ID,
                    action="export_excerpt",
                    source=result.label,
                    page=result.page,
                )
            except Exception:  # noqa: BLE001 - telemetry must never break the UI
                pass
            pilot.display("Added to citation index.", container=st, method="success")
        st.divider()


def _render_citation_index_export(slot: str) -> None:
    """CSV/JSON download buttons for the accumulated ``citation_index`` (F2.S2)."""
    index_any: Any = st.session_state.get(CITATION_INDEX_KEY, [])
    citations: list[dict[str, str]] = index_any if isinstance(index_any, list) else []
    if not citations:
        return

    pilot.display(f"📌 Citation index: {len(citations)} excerpt(s) collected this session.", container=st, method="caption")
    col_csv, col_json = st.columns(2)
    for fmt, col, mime in (("csv", col_csv, "text/csv"), ("json", col_json, "application/json")):
        try:
            data = export_citation_index(citations, fmt)
        except ValueError as exc:  # noqa: BLE001 - defensive; fmt is hard-coded above
            try:
                track_feature_error(SEARCH_FEATURE_ID, exc, stage="citation_export")
            except Exception:  # noqa: BLE001 - telemetry must never break the UI
                pass
            continue
        clicked = pilot.file_download(
            f"⬇️ Export citations (.{fmt})",
            data=data,
            file_name=f"citation_index.{fmt}",
            mime=mime,
            key=f"citation_export_{fmt}_{slot}",
         container=col)
        if clicked:
            try:
                track_interaction(
                    SEARCH_FEATURE_ID,
                    action="export_citation_index",
                    format=fmt,
                    citation_count=len(citations),
                )
            except Exception:  # noqa: BLE001 - telemetry must never break the UI
                pass


__all__ = [
    "is_local_run",
    "records_uploader",
    "remember_source_records",
    "render_record_search",
]
