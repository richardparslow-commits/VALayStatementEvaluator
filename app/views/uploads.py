"""Upload helpers: size-limit gate + extraction caching with skip warnings.

Split out of ``app/views/shared.py``; the actual parsing lives in
``app.documents``.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from .. import config
from ..documents import extract_uploaded_documents
from ..logging_config import get_logger

logger = get_logger("app.views.uploads")


def check_upload_limits(files: Any) -> tuple[list[Any], list[str]]:
    """Split uploaded files into accepted vs rejected by VA_LSE_MAX_UPLOAD_BYTES.

    Returns (accepted_files, rejection_messages). Accepted files also pass a
    total-batch cap (VA_LSE_MAX_TOTAL_UPLOAD_BYTES) — the largest files are
    dropped first until the batch fits, with one message per dropped file.
    """
    if not files:
        return [], []

    # Each Streamlit UploadedFile exposes .name and .size (bytes). Fall back to
    # len(getvalue()) for test fakes that only expose getvalue().
    def _size(f: Any) -> int:
        try:
            return int(getattr(f, "size", None) or len(f.getvalue()))
        except Exception:  # noqa: BLE001
            return 0

    per_file_limit = config.MAX_UPLOAD_BYTES
    total_limit = config.MAX_TOTAL_UPLOAD_BYTES
    rejected_msgs: list[str] = []
    # Per-file check
    accepted: list = []
    for f in files:
        sz = _size(f)
        if sz > per_file_limit:
            rejected_msgs.append(
                f"✖️ {f.name}: {sz // 1_048_576} MB exceeds the per-file limit "
                f"({per_file_limit // 1_048_576} MB). Reduce or split this file."
            )
        else:
            accepted.append(f)
    # Batch total check — drop excess largest-first so the user's first files tend to survive.
    total = sum(_size(f) for f in accepted)
    if total > total_limit and accepted:
        accepted.sort(key=_size)  # smallest first; we keep small ones
        kept: list = []
        running = 0
        for f in accepted:
            if running + _size(f) <= total_limit:
                kept.append(f)
                running += _size(f)
            else:
                rejected_msgs.append(
                    f"✖️ {f.name}: batch total would exceed {total_limit // 1_048_576} MB — file skipped. "
                    "Remove some files or raise VA_LSE_MAX_TOTAL_UPLOAD_BYTES."
                )
        accepted = kept
    return accepted, rejected_msgs


def extract_uploads(files: Any, slot: str) -> list[Any]:
    """Extract text from uploaded files; cache results per file identity.

    Files that fail extraction (e.g. image-only PDFs) are reported as
    per-file warnings plus a loaded-vs-skipped summary, recomputed fresh each
    run so unreadable uploads never silently disappear and never linger once
    the bad file is removed or replaced.
    """
    documents = []
    to_extract = []
    for uploaded in files:
        cache_key = f"{slot}:{uploaded.name}:{uploaded.size}"
        if cache_key in st.session_state:
            documents.append(st.session_state[cache_key])
        else:
            to_extract.append(uploaded)

    new_docs, skipped = extract_uploaded_documents(to_extract) if to_extract else ([], [])
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
    _render_skip_summary(files, documents, skipped)
    for message in skipped:
        st.warning(message)
    return documents


def _render_skip_summary(files: Any, documents: list[Any], skipped: list[str]) -> None:
    """Show a loaded-vs-skipped summary under an uploader when any file failed."""
    if not files or not skipped:
        return
    total = len(files)
    loaded = len(documents)
    st.caption(
        f"Loaded {loaded} of {total} file(s) — {len(skipped)} skipped (listed below)."
    )
