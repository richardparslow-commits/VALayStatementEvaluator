"""Medical Record Timeline View.

Renders a visual, sortable, filterable timeline of medical events extracted from
records. Includes gap detection, expandable event details, and PDF export.
"""
from __future__ import annotations

from .. import pilot

from typing import Any

import streamlit as st

from .. import medical_review
from ..documents import ExtractedDocument
from .shared import render_usage_summary


def render_timeline_tab(digest: medical_review.MedicalDigest) -> None:
    """Render the medical record timeline tab with filters and event details.

    Args:
        digest: The MedicalDigest containing extracted facts from record review.
    """
    st.subheader("📅 Medical Record Timeline")

    if not digest.facts:
        pilot.display("No medical events were extracted from the records. Try re-running the review.", container=st, method="info")
        return

    events = medical_review.build_timeline_events(digest)
    type_counts = medical_review.get_timeline_type_counts(events)
    providers = medical_review.get_timeline_providers(events)

    # Timeline stats bar
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total events", len(events))
    col2.metric("Date range", _format_date_range(events))
    col3.metric("Event types", len(type_counts))
    col4.metric("Providers", len(providers))

    st.divider()

    # Filter controls in an expander
    with st.expander("🔍 Filter & Sort Options", expanded=True):
        # Date range filter
        st.subheader("Date Range")
        date_cols = st.columns(2)
        with date_cols[0]:
            date_from = st.date_input(
                "From date",
                value=None,
                key="timeline_date_from",
                help="Show events on or after this date",
            )
        with date_cols[1]:
            date_to = st.date_input(
                "To date",
                value=None,
                key="timeline_date_to",
                help="Show events on or before this date",
            )

        # Gap detection toggle
        show_gaps = st.checkbox(
            "🔎 Detect gap periods",
            value=False,
            key="timeline_show_gaps",
            help="Highlight periods with no recorded medical events (potential missing records)",
        )

        # Fact type filter (multi-select)
        st.subheader("Event Type")
        all_types = sorted(type_counts.keys())
        selected_types = st.multiselect(
            "Filter by event type",
            options=all_types,
            default=all_types,
            key="timeline_fact_types",
            help="Select one or more event types to display",
        )
        if not selected_types:
            selected_types = all_types  # Show all if none selected

        # Provider filter
        st.subheader("Provider / Facility")
        selected_providers = st.multiselect(
            "Filter by provider",
            options=providers,
            default=providers,
            key="timeline_providers",
            help="Select providers/facilities to include",
        )

        # Sort order
        sort_order = st.radio(
            "Sort order",
            options=["Chronological (oldest first)", "Chronological (newest first)"],
            key="timeline_sort_order",
            horizontal=True,
        )

        apply_filters = st.button("Apply filters", type="primary")
        clear_filters = st.button("Clear all filters")

        if clear_filters:
            st.session_state.timeline_date_from = None
            st.session_state.timeline_date_to = None
            st.session_state.timeline_show_gaps = False
            st.session_state.timeline_fact_types = all_types
            st.session_state.timeline_providers = providers
            st.session_state.timeline_sort_order = "Chronological (oldest first)"
            # Re-run to clear the view
            st.rerun()

    # Apply filters
    date_from_str = _date_to_string(date_from) if date_from else None
    date_to_str = _date_to_string(date_to) if date_to else None

    filtered_events = medical_review.filter_timeline_events(
        events,
        date_from=date_from_str,
        date_to=date_to_str,
        fact_types=set(selected_types),
        providers=set(selected_providers) if selected_providers else None,
    )

    # Sort events
    reverse_sort = sort_order == "Chronological (newest first)"
    filtered_events.sort(
        key=lambda e: (e.date_sortable, e.index),
        reverse=reverse_sort,
    )

    # Display gap warnings if enabled
    if show_gaps and len(filtered_events) >= 2:
        gaps = medical_review.detect_timeline_gaps(filtered_events, min_gap_months=6)
        if gaps:
            pilot.display(f"⚠️ **{len(gaps)} gap period(s)** detected with no recorded events:", container=st, method="warning")
            for gap in gaps:
                pilot.display(f"• {gap.duration_months} months between {gap.start_date} and {gap.end_date}", container=st, method="caption")
                pilot.display(f"  {gap.note}", container=st, method="caption")

    # Timeline visualization
    if not filtered_events:
        pilot.display("No events match the current filters.", container=st, method="info")
        return

    # Render timeline as an interactive list
    st.divider()
    pilot.display(f"Showing {len(filtered_events)} of {len(events)} events", container=st, method="caption")

    # Color coding by event type
    type_colors = {
        "diagnosis": "#3b82f6",  # Blue
        "symptom": "#ef4444",  # Red
        "treatment": "#10b981",  # Green
        "medication": "#8b5cf6",  # Purple
        "test_result": "#f59e0b",  # Amber
        "hospitalization": "#ec4899",  # Pink
        "provider_visit": "#06b6d4",  # Cyan
        "in_service_event": "#f97316",  # Orange
        "functional_limitation": "#6366f1",  # Indigo
        "observable_behavior": "#14b8a6",  # Teal
        "administrative": "#6b7280",  # Gray
        "other": "#9ca3af",  # Light gray
    }

    for event in filtered_events:
        color = type_colors.get(event.type, "#6b7280")
        date_display = event.date if event.date and event.date != "unknown" else "Date unknown"

        # Event card
        with st.expander(
            pilot.text_label(f"📌 **{date_display}** — {event.type.replace('_', ' ').title()}"),
            expanded=False,
        ):
            # Full description
            pilot.display(f"**{event.full_description}**", container=st, method="markdown")

            # Quote if present
            if event.quote:
                pilot.display(f"> \\\"{event.quote}\\\"", container=st, method="markdown")

            # Source citation
            pilot.display(f"📄 Source: {event.source}", container=st, method="caption")

            # Type badge with color
            pilot.display(
                f'<span style="background-color: {color}; color: white; padding: 2px 8px; '
                f'border-radius: 4px; font-size: 0.8em;">{event.type.replace("_", " ").title()}</span>',
                unsafe_allow_html=True,
             container=st, method="markdown")

    # Export options
    st.divider()
    st.subheader("Export Timeline")

    col_pdf, col_md = st.columns(2)

    with col_pdf:
        if st.button("🖨️ Print Timeline (PDF)", type="primary"):
            _export_timeline_pdf(digest, filtered_events)

    with col_md:
        if st.button("📄 Export as Markdown"):
            _copy_timeline_markdown(filtered_events)


def _format_date_range(events: list[medical_review.TimelineEvent]) -> str:
    """Format the date range of events for display."""
    if not events:
        return "—"

    dates = [e.date for e in events if e.date and e.date != "unknown"]
    if not dates:
        return "No dates"

    return f"{dates[0]} to {dates[-1]}"


def _date_to_string(date_obj: Any) -> str:
    """Convert a date object to YYYY-MM-DD string (empty string when absent)."""
    if date_obj is None:
        return ""
    return str(date_obj.strftime("%Y-%m-%d"))


def _export_timeline_pdf(digest: medical_review.MedicalDigest, events: list[medical_review.TimelineEvent]) -> None:
    """Generate and offer download of a timeline PDF."""
    try:
        from ..pdf_export import generate_timeline_pdf
    except ImportError:
        pilot.display("PDF export is not available. Please ensure reportlab is installed.", container=st, method="error")
        return

    try:
        pdf_bytes = generate_timeline_pdf(events)
    except Exception:
        pilot.display("Timeline export failed. No PDF was created. Please retry or contact the operator.", container=st, method="error")
        return
    pilot.file_download(
        "⬇️ Download Timeline PDF",
        data=pdf_bytes,
        file_name="medical_record_timeline.pdf",
        mime="application/pdf",
        key="timeline_pdf_download",
     container=st)


def _copy_timeline_markdown(events: list[medical_review.TimelineEvent]) -> None:
    """Copy timeline as markdown to clipboard and offer download."""
    markdown = medical_review.render_timeline_markdown(events)

    # Offer download
    pilot.file_download(
        "⬇️ Download as .md",
        data=markdown.encode("utf-8"),
        file_name="medical_record_timeline.md",
        mime="text/markdown",
        key="timeline_md_download",
     container=st)

    # Also copy to clipboard if possible
    try:
        import pyperclip
        pyperclip.copy(markdown)
        pilot.display("Timeline markdown copied to clipboard!", container=st, method="success")
    except ImportError:
        pilot.display("Markdown exported. Install pyperclip for clipboard support.", container=st, method="info")


def render_timeline_in_results(digest: medical_review.MedicalDigest) -> None:
    """Render a compact timeline view within results panels.

    This is a lighter version suitable for inclusion in draft/evaluate results,
    showing key events without the full filtering UI.
    """
    if not digest.facts:
        return

    events = medical_review.build_timeline_events(digest)

    # Show first 20 events in a compact view
    display_events = events[:20]

    pilot.display("### Key Medical Events", container=st, method="markdown")
    pilot.display(f"Showing first 20 of {len(events)} events. View full timeline for all events.", container=st, method="caption")

    # Compact timeline as a table
    rows = []
    for event in display_events:
        rows.append({
            "Date": event.date if event.date else "Unknown",
            "Type": event.type.replace("_", " ").title(),
            "Description": event.description,
            "Source": event.source,
        })

    st.dataframe(rows, width="stretch", hide_index=True)

    if len(events) > 20:
        pilot.display(f"...and {len(events) - 20} more events. Use the Timeline tab for the full view.", container=st, method="caption")


__all__ = [
    "render_timeline_tab",
    "render_timeline_in_results",
]
