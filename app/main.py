"""VA Lay Statement Evaluator — Streamlit application entry point.

Run with:
    streamlit run app/main.py
"""
from __future__ import annotations

import streamlit as st

from .config import DEFAULT_BASE_URL, load_settings
from .documents import ExtractionError, extract_document, MAX_STATEMENT_CHARS
from .draft import grounding_markdown, run_draft
from .evaluate import DIMENSION_LABELS, run_evaluation
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
        if st.button("Apply settings"):
            settings.api_key = st.session_state.api_key_input.strip()
            settings.base_url = st.session_state.base_url_input.strip() or DEFAULT_BASE_URL
            settings.model_main = st.session_state.model_main_input.strip()
            settings.model_fast = st.session_state.model_fast_input.strip()
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
    """Extract text from uploaded files; cache results per file identity."""
    documents = []
    for uploaded in files:
        cache_key = f"{slot}:{uploaded.name}:{uploaded.size}"
        if cache_key in st.session_state:
            documents.append(st.session_state[cache_key])
            continue
        try:
            doc = extract_document(uploaded.name, uploaded.getvalue())
        except ExtractionError as exc:
            st.warning(str(exc))
            continue
        st.session_state[cache_key] = doc
        documents.append(doc)
    return documents


def _records_uploader(slot: str) -> list:
    files = st.file_uploader(
        "Upload medical records (PDF, TXT, MD, DOCX — multiple allowed)",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        key=f"files_{slot}",
    )
    return _extract_uploads(files, slot)


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
            f"Loaded {len(records)} record file(s), {total_pages} page(s): "
            + ", ".join(d.filename for d in records)
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

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall rating", result.overall_rating)
    col2.metric("Claims verified", len(result.verifications))
    col3.metric("Contradictions", result.contradiction_count)

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
            f"Loaded {len(records)} record file(s), {total_pages} page(s): "
            + ", ".join(d.filename for d in records)
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
            st.write(result.digest.summary)
            st.code(result.digest.timeline_text()[:12000], language=None)


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
lay-evidence law, and then suggests how to improve it: a prioritized improvement plan plus a
proposed rewrite with corrections grounded in the records and confirmation placeholders.

**Pathway 2 — Draft:** Upload the veteran's medical records and answer questions about what the
witness has personally observed. The app grounds the statement in the records, flags anything
that conflicts or cannot be verified, suggests strengthening questions, and drafts a
first-person statement in VA Form 21-10210 style that stays strictly within lay-competence
boundaries.

Both pathways review every page of every uploaded document — records are processed in chunks
so very long files are handled exhaustively.
"""
    )
    st.subheader("Legal foundation")
    with st.expander("Legal framework distilled into this tool"):
        st.markdown(load_knowledge("legal_framework.md"))
    with st.expander("Evaluation rubric"):
        st.markdown(load_knowledge("evaluation_rubric.md"))
    with st.expander("Drafting guide"):
        st.markdown(load_knowledge("drafting_guide.md"))
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
