"""Explicit preparation of an exact-reviewed TXT download, behind pilot acceptance."""
from __future__ import annotations

import streamlit as st

from .. import pilot
from ..factual_integrity import fingerprint
from ..text_exports import STORE, ExportLease, ExportUnavailable, invalidate, policy_binding


def render_text_export(text: str, *, slot: str, confirmed: bool) -> None:
    if not pilot.enabled():
        return
    if not confirmed:
        invalidate(slot)
        return
    try:
        approval = pilot.load_approval()
        binding = policy_binding(approval)
    except ExportUnavailable:
        invalidate(slot)
        return
    if not STORE.active:
        raise pilot.PilotBlocked("The private export service is unavailable. Use the reviewed pilot launcher.")
    pilot.require_session_access()
    owner = pilot.current_owner()
    consent = pilot.require_consent(owner)
    receipt = st.session_state.get(f"factual_{slot}_receipt")
    if not isinstance(receipt, dict) or receipt.get("text_hash") != fingerprint(text):
        invalidate(slot)
        return
    lease_key = "_text_export_lease_" + slot
    handle_key = "_text_export_handle_" + slot
    lease = st.session_state.get(lease_key)
    if (not isinstance(lease, ExportLease) or lease.revoked.is_set() or lease.owner != owner
            or lease.scope != receipt["scope"] or lease.text_hash != receipt["text_hash"]
            or lease.approval_binding != binding or lease.consent is not consent):
        invalidate(slot)
        lease = ExportLease(owner, slot, receipt["scope"], receipt["text_hash"], binding, consent)
        st.session_state[lease_key] = lease
    st.caption("Reviewed .txt statements only. This private download expires within five minutes; "
               "clear or edit revokes it. A saved copy cannot be erased by this app.")
    if st.button("Prepare reviewed text download", key="prepare_text_export_" + slot):
        try:
            # Recheck after the participant's click and before retaining any bytes.
            pilot.require_session_access()
            if policy_binding(pilot.load_approval()) != binding:
                raise ExportUnavailable("Export approval changed; repeat the review.")
            st.session_state[handle_key] = STORE.create(lease, text)
        except ExportUnavailable as exc:
            invalidate(slot)
            pilot.display(str(exc), container=st, method="warning")
            return
    handle = st.session_state.get(handle_key)
    if isinstance(handle, str):
        try:
            STORE.read(handle, owner, approval)
        except ExportUnavailable:
            st.session_state.pop(handle_key, None)
            pilot.display("The private download expired or was revoked. Prepare it again after review.",
                          container=st, method="info")
            return
        st.link_button("Download reviewed statement (.txt)",
                       pilot.https_url(approval["deployment_url"]).rstrip("/") + "/pilot-exports/" + handle)
