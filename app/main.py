"""VA Lay Statement Evaluator — Streamlit application entry point.

Run with:
    streamlit run app/main.py
"""
from __future__ import annotations

import streamlit as st

from . import config
from .config import DEFAULT_BASE_URL, load_settings
from .documents import (
    ExtractionError,
    MAX_STATEMENT_CHARS,
    extract_document,
    extract_uploaded_documents,
    records_from_local_path,
)
from .draft import grounding_markdown, run_draft
from .evaluate import DIMENSION_LABELS, run_evaluation
from .fetch_client import FetchClient, FetchSandboxError
from .llm import LLMClient, LLMError

st.set_page_config(
    page_title="VA Lay Statement Evaluator",
    page_icon="🎖️",
    layout="wide",
)

CLAIM_TYPES = [
    "Service connection (new claim)",
    "Increased rating (worsening condition)",
    "PTSD stressor corroboration",
    "TDIU / individual unemployability",
    "Continuity of symptoms since service",
]

RELATIONSHIPS = [
    "Spouse",
    "Family member",
    "Friend",
    "Coworker / supervisor",
    "Fellow service member",
    "Other",
]


# ------------------------------------------------------------------- settings
def _sidebar_settings() -> None:
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
            "Fast model", value=settings.model_fast, help="Bulk record digests"
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
            settings.api_key = st.session_state.api_key_input.strip()
            settings.base_url = st.session_state.base_url_input.strip() or DEFAULT_BASE_URL
            settings.model_main = st.session_state.model_main_input.strip()
            settings.model_fast = st.session_state.model_fast_input.strip()
            settings.fetch_api_key = st.session_state.fetch_api_key_input.strip()
            settings.fetch_base_url = st.session_state.fetch_base_url_input.strip()
            settings.fetch_records_path = st.session_state.fetch_records_path_input.strip()
            st.rerun()

        st.divider()
        st.caption(
            "⚠️ Uploaded documents are sent to the configured LLM endpoint for analysis. "
            "Review privacy before uploading sensitive records."
        )
        st.caption(
            "This tool is an aid for drafting and reviewing lay statements. It is not "
            "legal, medical, or claims advice."
        )


def _get_llm() -> LLMClient | None:
    settings = st.session_state.settings
    settings.api_key = st.session_state.get("api_key_input", settings.api_key).strip()
    if not settings.configured:
        st.error("Enter your LLM API key in the sidebar before running.")
        return None
    try:
        return LLMClient(settings)
    except LLMError as exc:
        st.error(str(exc))
        return None


# ------------------------------------------------------------- shared uploads
def _extract_uploads(files, slot: str) -> list:
    """Extract text from uploaded files; cache results per file identity.

    Files that fail extraction (e.g. image-only PDFs) are reported as
    per-file warnings that persist in session state, so unreadable uploads
    never silently disappear.
    """
    documents = []
    to_extract = []
    for uploaded in files:
        cache_key = f"{slot}:{uploaded.name}:{uploaded.size}"
        if cache_key in st.session_state:
            documents.append(st.session_state[cache_key])
        else:
            to_extract.append(uploaded)

    new_docs, skipped = extract_uploaded_documents(to_extract)
    for doc in new_docs:
        # Cache each successful extraction by its (slot, name, size) identity.
        for uploaded in to_extract:
            if uploaded.name == doc.filename:
                st.session_state[f"{slot}:{uploaded.name}:{uploaded.size}"] = doc
                break
        documents.append(doc)
    # The uploader re-delivers files on every rerun, so warnings are recomputed
    # fresh each run: they persist while a bad file is still uploaded and clear
    # as soon as it is removed or replaced.
    for message in skipped:
        st.warning(message)
    return documents


def _is_local_run() -> bool:
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


def _records_uploader(slot: str) -> list:
    sources = ["Upload files", "Fetch Sandbox"]
    if _is_local_run():
        sources.append("Local folder / file")
    source = st.radio(
        "Medical record source",
        sources,
        horizontal=True,
        key=f"records_source_{slot}",
    )
    if source == "Fetch Sandbox":
        return _fetch_records(slot)
    if source == "Local folder / file":
        return _local_records(slot)

    files = st.file_uploader(
        "Upload medical records (PDF, TXT, MD, DOCX — multiple allowed)",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        key=f"files_{slot}",
    )
    documents = _extract_uploads(files, slot)
    total_pages = sum(len(d.pages) for d in documents)
    if documents and total_pages > config.MAX_RECORD_PAGES:
        st.error(
            f"Record set is {total_pages:,} pages, which exceeds the configured limit of "
            f"{config.MAX_RECORD_PAGES:,} pages. Remove some files or raise "
            "VA_LSE_MAX_RECORD_PAGES."
        )
        return []
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
    return st.session_state.get(import_key, [])


def _fetch_records(slot: str) -> list:
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
    return st.session_state.get(import_key, [])


def _progress_widgets():
    bar = st.progress(0.0, text="Starting…")

    def update(frac: float, msg: str) -> None:
        bar.progress(min(max(frac, 0.0), 1.0), text=msg)

    return bar, update



# --------------------------------------------------------------- evaluate tab
def evaluate_tab() -> None:
    st.subheader("Step 1 — Provide the lay statement")
    mode = st.radio(
        "Statement source", ["Upload file", "Paste text"], key="eval_mode", horizontal=True
    )
    statement_text = ""
    if mode == "Paste text":
        statement_text = st.text_area(
            "Paste the full lay/witness statement", height=260, key="eval_paste"
        )
    else:
        files = st.file_uploader(
            "Upload the statement (PDF, TXT, MD, DOCX)",
            type=["pdf", "txt", "md", "docx"],
            key="eval_statement_file",
        )
        if files is not None:
            docs = _extract_uploads([files], "eval_statement")
            if docs:
                statement_text = docs[0].full_text

    st.subheader("Step 2 — Provide the medical records")
    records = _records_uploader("eval")
    if records:
        total_pages = sum(len(d.pages) for d in records)
        st.success(
            f"Loaded {len(records)} record file(s), {total_pages:,} page(s): "
            + ", ".join(d.filename for d in records)
        )
        if total_pages > 200:
            st.info(
                "Large record set: chunks are digested in parallel with duplicate pages "
                "skipped, but expect a longer run for a meticulous review."
            )

    run = st.button("🔍 Run exhaustive evaluation", type="primary", key="eval_run")
    if run:
        if not statement_text.strip():
            st.error("Provide the lay statement first (upload or paste).")
            return
        if len(statement_text) > MAX_STATEMENT_CHARS:
            st.error(f"Statement is too long (max {MAX_STATEMENT_CHARS} characters).")
            return
        if not records:
            st.error("Upload at least one medical record file.")
            return
        llm = _get_llm()
        if llm is None:
            return

        bar, update = _progress_widgets()
        try:
            result = run_evaluation(llm, statement_text.strip(), records, progress=update)
        except Exception as exc:  # noqa: BLE001
            bar.empty()
            st.error(f"Evaluation failed: {exc}")
            return
        bar.empty()
        st.session_state.eval_result = result

    result = st.session_state.get("eval_result")
    if result is None:
        return

    st.divider()
    st.subheader("📋 Evaluation Results")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall rating", result.overall_rating)
    col2.metric("Claims verified", len(result.verifications))
    col3.metric("Contradictions", result.contradiction_count)
    _applicable = [t for t in result.topic_rows if t.get("applicable")]
    _covered = [t for t in _applicable if t.get("coverage") == "covered"]
    col4.metric("Topics covered", f"{len(_covered)}/{len(_applicable)}" if _applicable else "—")

    with st.expander("Executive summary", expanded=True):
        st.write(result.executive_summary)

    with st.expander("Claim-by-claim verification table", expanded=True):
        rows = []
        claim_text = {c["id"]: c.get("text", "") for c in result.claims}
        for v in result.verifications:
            rows.append(
                {
                    "Claim": claim_text.get(v.get("id"), ""),
                    "Verdict": v.get("verdict", ""),
                    "Record reference": v.get("record_reference", ""),
                    "Note": v.get("note", ""),
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("Rubric scores", expanded=True):
        score_rows = [
            {
                "Dimension": DIMENSION_LABELS.get(k, k),
                "Score": result.scores.get(k, 0),
                "Rationale": result.rationales.get(k, ""),
            }
            for k in DIMENSION_LABELS
        ]
        st.dataframe(score_rows, use_container_width=True, hide_index=True)
        st.bar_chart(
            {DIMENSION_LABELS[k]: result.scores.get(k, 0) for k in DIMENSION_LABELS},
            horizontal=True,
        )

    if result.topic_rows:
        with st.expander(
            "🧭 Topic coverage — what the statement does and does not address", expanded=True
        ):
            if result.topic_focus:
                st.write(f"**Claim focus:** {result.topic_focus}")
            topic_table = [
                {
                    "Topic": t.get("topic", ""),
                    "Applicable": "Yes" if t.get("applicable") else "No",
                    "Coverage": t.get("coverage", ""),
                    "Evidence in statement": t.get("evidence", ""),
                    "How to strengthen": t.get("gap_note", ""),
                }
                for t in result.topic_rows
            ]
            st.dataframe(topic_table, use_container_width=True, hide_index=True)
            if result.topic_critical_gaps:
                st.warning(
                    "**Critical gaps — the highest-impact topics this statement still misses:**"
                )
                for gap in result.topic_critical_gaps:
                    st.write(f"- {gap}")
            if result.topic_notes:
                st.caption(result.topic_notes)

    with st.expander("Improvements & record facts to add", expanded=True):
        for imp in result.improvements:
            st.markdown(f"**{imp.get('priority', '?')}. {imp.get('problem', '')}**")
            st.write(imp.get("suggestion", ""))
            if imp.get("example_rewrite"):
                st.caption(f"Example: “{imp.get('example_rewrite')}”")
        if result.omitted_record_facts:
            st.markdown("**Facts from the records you could add (verify first):**")
            for fact in result.omitted_record_facts:
                st.write(f"- {fact.get('fact', '')} _(source: {fact.get('source', '')})_")

    if result.revised_statement or result.revision_changes:
        with st.expander("📝 Suggested improvements — proposed rewrite", expanded=True):
            if result.revision_notes:
                st.info(result.revision_notes)
            if result.revision_changes:
                change_rows = [
                    {
                        "Category": c.get("category", ""),
                        "Original": c.get("original", "") or "(addition)",
                        "Suggested": c.get("revised", ""),
                        "Why": c.get("reason", ""),
                    }
                    for c in result.revision_changes
                ]
                st.dataframe(change_rows, use_container_width=True, hide_index=True)
            if result.added_facts_to_verify:
                st.markdown(
                    "**Record-sourced facts added — the witness must confirm each before signing:**"
                )
                for fact in result.added_facts_to_verify:
                    st.write(f"- {fact}")
            st.markdown("#### Revised statement")
            st.caption(
                "Contradictions have been corrected to match the medical records. Resolve every "
                "[Confirm: ...] placeholder with the witness before signing."
            )
            revised = st.text_area(
                "Revised statement (editable)",
                value=result.revised_statement,
                height=420,
                key="eval_revised_statement",
            )
            col_a, col_b = st.columns(2)
            col_a.download_button(
                "⬇️ Download revised statement (.txt)",
                data=revised.encode("utf-8"),
                file_name="lay_statement_revised.txt",
                mime="text/plain",
            )
            col_b.download_button(
                "⬇️ Download revised statement (.md)",
                data=revised.encode("utf-8"),
                file_name="lay_statement_revised.md",
                mime="text/markdown",
            )

    with st.expander("Full markdown report"):
        st.markdown(result.report_markdown)
    st.download_button(
        "⬇️ Download evaluation report (.md)",
        data=result.report_markdown.encode("utf-8"),
        file_name="lay_statement_evaluation.md",
        mime="text/markdown",
    )


# ----------------------------------------------------------------- draft tab
def draft_tab() -> None:
    st.subheader("Step 1 — Upload the veteran's medical records")
    records = _records_uploader("draft")
    if records:
        total_pages = sum(len(d.pages) for d in records)
        st.success(
            f"Loaded {len(records)} record file(s), {total_pages:,} page(s): "
            + ", ".join(d.filename for d in records)
        )
        if total_pages > 200:
            st.info(
                "Large record set: chunks are digested in parallel with duplicate pages "
                "skipped, but expect a longer run for a meticulous review."
            )

    st.subheader("Step 2 — Claim details")
    col1, col2 = st.columns(2)
    veteran_name = col1.text_input("Veteran's name", key="draft_vet_name")
    condition = col2.text_input(
        "Condition the statement supports (e.g., PTSD, lumbar strain, tinnitus)",
        key="draft_condition",
    )
    col3, col4 = st.columns(2)
    claim_type = col3.selectbox("Claim type", CLAIM_TYPES, key="draft_claim_type")
    relationship = col4.selectbox("Witness relationship", RELATIONSHIPS, key="draft_rel")

    st.subheader("Step 3 — Witness details")
    col5, col6, col7 = st.columns(3)
    witness_name = col5.text_input("Witness full name", key="draft_witness_name")
    known_since = col6.text_input("Known the veteran since / for", key="draft_known")
    contact_frequency = col7.text_input(
        "How often they see each other (opportunity to observe)", key="draft_freq"
    )
    witnessed_event = st.radio(
        "Did the witness personally see the in-service event happen?",
        ["No", "Yes", "Not applicable"],
        horizontal=True,
        key="draft_witnessed",
    )

    st.subheader("Step 4 — What has the witness observed?")
    observations = st.text_area(
        "Describe everything the witness has personally seen, heard, or experienced "
        "regarding the veteran's condition: symptoms, incidents, changes over time, "
        "impact on work, family and social life. Bullet points are fine — the app will "
        "turn them into a polished, factually grounded statement.",
        height=220,
        key="draft_observations",
    )

    run = st.button("✍️ Draft the statement", type="primary", key="draft_run")
    if run:
        if not records:
            st.error("Upload at least one medical record file first.")
            return
        if not observations.strip() or not condition.strip():
            st.error("Enter the condition and the witness's observations.")
            return
        llm = _get_llm()
        if llm is None:
            return

        witness = {
            "name": witness_name.strip(),
            "relationship": relationship,
            "known_since": known_since.strip(),
            "contact_frequency": contact_frequency.strip(),
            "veteran_name": veteran_name.strip(),
            "witnessed_event": witnessed_event,
        }
        bar, update = _progress_widgets()
        try:
            result = run_draft(
                llm, records, witness, observations.strip(), condition.strip(),
                claim_type, progress=update,
            )
        except Exception as exc:  # noqa: BLE001
            bar.empty()
            st.error(f"Drafting failed: {exc}")
            return
        bar.empty()
        st.session_state.draft_result = result

    result = st.session_state.get("draft_result")
    if result is None:
        return

    st.divider()
    st.subheader("📋 Draft Results")

    with st.expander("Grounding analysis — how the draft ties to the records", expanded=True):
        st.markdown(grounding_markdown(result))

    if result.review_issues:
        with st.expander("Self-review findings (fixed in the final version)"):
            for issue in result.review_issues:
                st.write(f"- {issue}")

    st.markdown("### Final statement (editable)")
    st.caption(
        "Review every bracketed [Confirm: ...] placeholder and resolve it before signing. "
        "Submit on VA Form 21-10210 (one form per witness)."
    )
    edited = st.text_area(
        "Statement", value=result.output_statement, height=460, key="draft_edited"
    )
    col_a, col_b = st.columns(2)
    col_a.download_button(
        "⬇️ Download statement (.txt)",
        data=edited.encode("utf-8"),
        file_name="lay_statement_draft.txt",
        mime="text/plain",
    )
    col_b.download_button(
        "⬇️ Download statement (.md)",
        data=edited.encode("utf-8"),
        file_name="lay_statement_draft.md",
        mime="text/markdown",
    )

    if result.digest:
        with st.expander("Medical record digest used for grounding"):
            st.caption(
                f"{len(result.digest.facts):,} facts extracted from "
                f"{result.digest.pages_reviewed:,} pages "
                f"({result.digest.chunks_reviewed} chunks, "
                f"{result.digest.duplicates_skipped} duplicate page(s) skipped)"
            )
            st.write(result.digest.summary)
            st.code(result.digest.timeline_text()[:20000], language=None)


# ----------------------------------------------------------------- about tab
def about_tab() -> None:
    from .config import load_knowledge

    st.subheader("What this tool does")
    st.markdown(
        """
**Pathway 1 — Evaluate:** Upload an already-written lay/witness statement plus the veteran's
medical records. The app conducts an exhaustive review of the records, extracts every factual
claim in the statement, verifies each claim against the records (supported / contradicted /
partially supported / not found), scores the statement on an 8-dimension rubric drawn from VA
lay-evidence law, and audits it against the topic checklist (hazards and dangers, caregiver
necessity, personal care, medication and financial management, household safety, errands and
driving, before/after progression, observable behaviors, family impact, medication side
effects). It then suggests how to improve it: a prioritized improvement plan plus a proposed
rewrite with corrections grounded in the records and confirmation placeholders.

**Pathway 2 — Draft:** Upload the veteran's medical records and answer questions about what the
witness has personally observed. The app grounds the statement in the records, checks the
observations against the topic checklist and asks follow-up questions for applicable topics the
witness has not yet covered, flags anything that conflicts or cannot be verified, and drafts a
first-person statement in VA Form 21-10210 style that stays strictly within lay-competence
boundaries.

Both pathways review every page of every uploaded document — records are processed in chunks
so very long files are handled exhaustively. Large record sets (hundreds to thousands of pages,
up to a configurable cap of ~5,000 pages) are supported: chunks are digested in parallel,
duplicate pages are skipped automatically, and verification always searches the full digest for
evidence relevant to each claim rather than reading only the first pages.
"""
    )
    st.subheader("Legal foundation")
    with st.expander("Legal framework distilled into this tool"):
        st.markdown(load_knowledge("legal_framework.md"))
    with st.expander("Evaluation rubric"):
        st.markdown(load_knowledge("evaluation_rubric.md"))
    with st.expander("Drafting guide"):
        st.markdown(load_knowledge("drafting_guide.md"))
    with st.expander("Topic checklist"):
        st.markdown(load_knowledge("topic_checklist.md"))
    st.info(
        "This tool is an educational and drafting aid. It is not legal, medical, or claims "
        "advice, and no output should be submitted without the witness personally verifying "
        "every fact. For accredited help: www.va.gov/ogc/apps/accreditation"
    )


# --------------------------------------------------------------------- layout
def main() -> None:
    _sidebar_settings()
    st.title("🎖️ VA Lay Statement Evaluator")
    st.caption(
        "Exhaustive medical-record review to verify existing lay statements — and to draft "
        "factually correct new ones. Grounded in 38 U.S.C. § 1154(a), § 5107(b) and the "
        "Jandreau/Buchanan/Caluza line of cases."
    )

    tab_eval, tab_draft, tab_about = st.tabs(
        ["🔍 Evaluate a statement", "✍️ Draft a statement", "📖 About / Guide"]
    )
    with tab_eval:
        evaluate_tab()
    with tab_draft:
        draft_tab()
    with tab_about:
        about_tab()


if __name__ == "__main__":
    main()
