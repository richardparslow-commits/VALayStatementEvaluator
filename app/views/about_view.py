"""About/Guide tab: static explainer + knowledge-base expanders."""
from __future__ import annotations

from .. import pilot

import streamlit as st

from ..build_info import build_sha, build_source
from ..config import load_knowledge
from ..diagnostics import CAPTURE_LIMIT, MAX_LINES, lookup
from .ops import render_run_log_tail


def _render_build_identity() -> None:
    """Show which commit this deployment runs, so staleness is visible.

    A stale deployment hides a merged feature — the question "where is the
    credentials section?" was really "is my deployment current?" three times
    in one week. The answer belongs where the user is looking, with an honest
    "unknown" (a source upload or an image built without the build arg) rather
    than a blank that reads as absence of the feature.
    """
    sha = build_sha()
    source = build_source()
    if sha:
        origin = {"environment": "from the image/deployment", "git": "from the git checkout"}.get(
            source, ""
        )
        pilot.display(f"Running build: `{sha}` {origin}", container=st, method="caption")
    else:
        pilot.display(
            "Running build: **unknown** — this deployment cannot see its own commit "
            "(source upload, or an image built without `VA_LSE_BUILD_SHA`). If a "
            "recently announced feature appears missing, redeploy from `main` first."
        , container=st, method="caption")


def render_about_tab() -> None:
    """Render the About / Guide tab (static content, no session state)."""
    _render_build_identity()
    if pilot.enabled():
        st.subheader("Controlled pilot guide")
        pilot.display("Only invited users can upload records and request analysis. Keep the source "
                 "records and witness account open while reviewing every AI statement. "
                 "The app blocks unknown page coverage and unverified record citations.", container=st, method="write")
        pilot.display("Cases remain in the current session. Clear case removes working data and "
                 "registered uploads; it cannot erase provider copies or text copied elsewhere. "
                 "Provider retention follows the reviewed pilot notice.", container=st, method="write")
        pilot.display("File downloads, remote fetching, research tools and worker queues are disabled. "
                 "Copy reviewed text only to an approved destination. The tool does not certify "
                 "legal sufficiency or the truth of a witness statement.", container=st, method="write")
        render_run_log_tail()
        return
    st.subheader("What this tool does")
    pilot.display(
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
    , container=st, method="markdown")
    st.subheader("Legal foundation")
    with st.expander("Legal framework distilled into this tool"):
        pilot.display(load_knowledge("legal_framework.md"), container=st, method="markdown")
    with st.expander("Evaluation rubric"):
        pilot.display(load_knowledge("evaluation_rubric.md"), container=st, method="markdown")
    with st.expander("Drafting guide"):
        pilot.display(load_knowledge("drafting_guide.md"), container=st, method="markdown")
    with st.expander("Topic checklist"):
        pilot.display(load_knowledge("topic_checklist.md"), container=st, method="markdown")
    pilot.display(
        "This tool is an educational and drafting aid. It is not legal, medical, or claims "
        "advice, and no output should be submitted without the witness personally verifying "
        "every fact. For accredited help: www.va.gov/ogc/apps/accreditation"
    , container=st, method="info")

    st.subheader("If something goes wrong — error references")
    pilot.display(
        """
When a run fails, the error message includes a **reference** such as
``(reference: req_4f8a2b1c9d0e)``. That id identifies your exact run so the
specific cause can be found in the logs — no run fails without a trace.

**Where the logs live** (relative to the app's folder):

| File | What it records |
|---|---|
| ``logs/runs.jsonl`` | One JSON line per run event — the gate's `accepted` (with `endpoint_check` fresh or reused), `start`, `ok`, `error`, `timeout`, or pre-run `rejected` — keyed by the reference id, with the error text and a stack-trace digest. This is the first place to look. |
| ``logs/audit.log`` | Audit trail of run outcomes (metadata only: which record sources, page counts, durations, outcome classification). |
| ``logs/unhandled_errors.log`` | Fallback record for errors outside the run flow (e.g. while rendering results). |
| ``logs/app.log`` | Structured application log (when file logging is enabled via ``VA_LSE_LOG_DIR``). |

**Reading a reference:** search any of these files for the id, e.g.
``grep req_4f8a2b1c9d0e logs/runs.jsonl`` — or use **Look up a reference**
below, which does the search for you and needs no shell access to the server.
A failed run also carries its own **What happened?** expander, so the lines for
that particular failure are usually one click from the error itself.
Each event line includes a UTC timestamp, the action (``draft``/``evaluate``/
``app``), the status, and — for failures — the exact error and a short traceback.
The logs never contain statement text, observations, or medical-record content;
only sizes, counts, and classifications. Uploaded **filenames** do appear in
messages, because "which file failed" is usually the whole question — they are
stored in the log exactly as the file was named. Secret-shaped strings are
replaced with ``[redacted]`` before anything is shown here.
"""
    , container=st, method="markdown")

    with st.expander("🔎 Look up a reference", expanded=False):
        _render_reference_lookup()

    with st.expander("Recent run log (live)", expanded=False):
        render_run_log_tail()


def _render_reference_lookup() -> None:
    """Resolve a ``req_…`` reference to its lines, from inside the app.

    A form rather than a bare text input: Streamlit reruns on every keystroke, and
    the lookup reads the run log, so unsubmitted typing must not do I/O. This is
    also why the panel is here instead of on the health sidecar — that port binds
    ``0.0.0.0`` with no authentication, and log content should not be published to
    it. The user's own session is the authorization.
    """
    if not pilot.operator_allowed():
        return
    pilot.display(
        "Paste a reference from an error message — the whole message works — to see "
        "what this app recorded for that run. Nothing is sent anywhere."
    , container=st, method="caption")
    with st.form("diagnostics_lookup", clear_on_submit=False):
        query = st.text_input(
            "Reference",
            key="diagnostics_reference_query",
            placeholder="req_4f8a2b1c9d0e",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Look up", key="diagnostics_lookup_submit")
    if not submitted or not query.strip():
        return

    detail = lookup(query)
    if not detail.valid:
        pilot.display(detail.problem, container=st, method="warning")
        return

    pilot.display(f"**Reference `{detail.reference}`**", container=st, method="markdown")

    if detail.events:
        pilot.display(
            f"**Run-log events** ({len(detail.events)} of {detail.scanned} examined) — "
            "written by whichever process ran the job, including workers:"
        , container=st, method="markdown")
        st.dataframe(detail.events, width="stretch", hide_index=True)
    else:
        pilot.display(
            f"No run-log event for `{detail.reference}` in the last "
            f"{detail.scanned} event(s)."
        , container=st, method="caption")

    if detail.lines:
        pilot.display(
            f"**Log lines** ({len(detail.lines)} from this process's last "
            f"{detail.buffered} record(s)):"
        , container=st, method="markdown")
        st.code("\n".join(detail.lines), language=None)
    elif detail.note:
        pilot.display(detail.note, container=st, method="info")

    if detail.truncated:
        pilot.display(
            f"Showing at most {MAX_LINES} lines and {CAPTURE_LIMIT} buffered records; "
            "older detail is on the server's log files."
        , container=st, method="caption")
