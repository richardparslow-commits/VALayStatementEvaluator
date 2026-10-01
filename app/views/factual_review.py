"""Source/witness review of exact edited text; no durable case approval claims."""
from __future__ import annotations

from typing import Any

import streamlit as st

from .. import pilot
from ..factual_integrity import compare, context_for_result, fingerprint, review_notice


def render_factual_review(result: Any, text: str, *, slot: str) -> bool:
    context = context_for_result(result)
    initial = compare(text, context)
    pilot.display(review_notice(initial), container=st, method="info")
    sources = {s["id"]: s for s in context["sources"]} if context else {}
    prefix = f"factual_{slot}_" + fingerprint([initial["text_hash"], initial["context_hash"]])
    links: dict[str, list[str]] = {}
    with st.expander("Compare each sentence with the original account and sources", expanded=initial["status"] == "blocked"):
        if sources:
            kinds = {"witness_account": "Witness account", "witness_field": "Witness detail", "record_quote": "Record quotation"}
            st.dataframe([{"Source": s["label"], "Source type": kinds[s["kind"]], "Original passage": s["text"]}
                          for s in sources.values()], hide_index=True, width="stretch")
        for index, row in enumerate(initial["rows"], 1):
            pilot.display(f"Sentence {index}", container=st, method="caption")
            # Plain text prevents uploaded/model markup from becoming controls.
            st.text(row["text"])
            selected = st.multiselect(
                f"Supporting original passages for sentence {index}", list(sources),
                default=row["sources"], max_selections=3,
                format_func=lambda sid: sources[sid]["label"] + " — " + sources[sid]["text"][:120],
                key=prefix + "_" + row["id"],
            )
            links[row["id"]] = selected if isinstance(selected, list) else []
        checked = (initial if all(links[row["id"]] == row["sources"] for row in initial["rows"])
                   else compare(text, context, links))
        for row in checked["rows"]:
            for issue in row["issues"]:
                pilot.display(f"Characters {row['start']}–{row['end']}: {issue}", container=st, method="warning")
        for issue in checked["issues"]:
            pilot.display(issue, container=st, method="warning")
        pilot.display("Select only passages that support the sentence's meaning. Correct changed facts, "
                      "restore uncertainty and attribution, and preserve every original witness passage. "
                      "For new facts, update the original account with the witness and re-run. "
                      "Source matches alone do not prove truth.", container=st, method="caption")

    owner = pilot.current_owner() if pilot.enabled() else "local-session"
    scope = fingerprint([slot, checked["text_hash"], checked["context_hash"], links, owner])
    state_key = f"factual_{slot}_approval_scope"
    old = st.session_state.get(state_key)
    if old != scope:
        if isinstance(old, str):
            st.session_state.pop("factual_approve_" + old, None)
        st.session_state.pop(f"factual_{slot}_receipt", None)
        st.session_state[state_key] = scope
    if checked["status"] != "review_required":
        st.session_state.pop("factual_approve_" + scope, None)
        st.session_state.pop(f"factual_{slot}_receipt", None)
        pilot.display("This text has unresolved factual review items and cannot be marked reviewed. "
                      "Edit it or restore the original inputs before proceeding.", container=st, method="warning")
        return False
    accepted = st.checkbox("I compared every sentence with its selected original passages and confirmed "
                           "the meaning, dates, numbers, uncertainty and firsthand attribution with the witness. "
                           "I reviewed this exact edited text.", key="factual_approve_" + scope)
    if accepted is not True:
        st.session_state.pop(f"factual_{slot}_receipt", None)
        return False
    st.session_state[f"factual_{slot}_receipt"] = {"scope": scope, "text_hash": checked["text_hash"],
                                                 "context_hash": checked["context_hash"]}
    pilot.display("Source and witness review recorded for this exact text in this session. "
                  "Any text, source selection, source-context or user change requires a new review.",
                  container=st, method="success")
    return True
