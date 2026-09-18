"""Upload helpers: size-limit gate + extraction caching with skip warnings.

Split out of ``app/views/shared.py``; the actual parsing lives in
``app.documents``.
"""
from __future__ import annotations

import hashlib
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


def _upload_cache_key(slot: str, uploaded: Any) -> str | None:
    """Cache key for one upload: slot, name, size **and content hash**.

    Name and size alone are not identity. A user who fixes a file locally and
    re-uploads a corrected copy with the same name and byte length would otherwise
    keep getting the old text back from ``session_state`` for the rest of the
    session — the failure looks like the app ignoring their correction.
    Unreadable bytes have no trustworthy cache identity.
    """
    try:
        digest = hashlib.sha256(uploaded.getvalue()).hexdigest()
    except Exception:  # noqa: BLE001 - never fall back to name/size identity
        return None
    return f"{slot}:{uploaded.name}:{getattr(uploaded, 'size', 0)}:{digest}"


def _prune_upload_cache(slot: str, live_keys: set[str]) -> None:
    """Drop cached extractions for files that are no longer uploaded."""
    try:
        state: Any = st.session_state
        stale = [
            key
            for key in list(state.keys())
            if isinstance(key, str) and key.startswith(f"{slot}:") and key not in live_keys
        ]
        for key in stale:
            del state[key]
    except Exception:  # noqa: BLE001 - session state is best-effort here
        return


def extract_uploads(files: Any, slot: str) -> list[Any]:
    """Extract text from uploaded files; cache results per file identity.

    Files that fail extraction (e.g. image-only PDFs) are reported as
    per-file warnings plus a loaded-vs-skipped summary, recomputed fresh each
    run so unreadable uploads never silently disappear and never linger once
    the bad file is removed or replaced.
    """
    documents = []
    skipped: list[str] = []
    live_keys: set[str] = set()
    for uploaded in files:
        cache_key = _upload_cache_key(slot, uploaded)
        if cache_key is None:
            skipped.append(f"✖️ {uploaded.name}: could not read uploaded file.")
            continue
        live_keys.add(cache_key)
        if cache_key in st.session_state:
            documents.append(st.session_state[cache_key])
            continue

        # Extract individually so duplicate names and skipped files cannot shift
        # the association between an upload's content key and its document.
        new_docs, file_skipped = extract_uploaded_documents([uploaded])
        skipped.extend(file_skipped)
        if new_docs:
            st.session_state[cache_key] = new_docs[0]
            documents.extend(new_docs)
    _prune_upload_cache(slot, live_keys)
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


def render_record_volume_warning(documents: list[Any], *, slot: str) -> None:
    """Warn about a record set big enough to be slow, partial, or over the cap.

    The uploader already says how many pages loaded; what it does not say is that
    the digest is capped (``MAX_DIGEST_FACTS``) and that a very large set will take
    a long time and may report fewer facts than the records contain. Better to say
    so before the run than to have the report quietly be thinner than the records.
    """
    if not documents:
        return
    text_pages = sum(len(getattr(doc, "pages", []) or []) for doc in documents)
    try:
        total_pages = sum(
            int(getattr(doc, "source_page_count", len(getattr(doc, "pages", []) or [])))
            for doc in documents
        )
    except Exception:  # noqa: BLE001 - defensive: any doc shape
        total_pages = text_pages
    if total_pages > config.MAX_RECORD_PAGES:
        st.error(
            f"⚠️ This record set is {total_pages:,} pages, over the configured limit of "
            f"{config.MAX_RECORD_PAGES:,}. The run will be refused — split the files or "
            "raise VA_LSE_MAX_RECORD_PAGES."
        )
    elif total_pages > config.RECORD_SIZE_WARN_PAGES:
        st.warning(
            f"⚠️ Large record set ({total_pages:,} pages): the review will take a long time "
            f"and the digest is capped at {config.MAX_DIGEST_FACTS:,} facts, so a record set "
            f"this size may not be represented in full. Consider splitting it by date range "
            "so each run can cover its pages completely."
        )
    logger.debug(
        "record volume slot=%s files=%d pages=%d text_pages=%d",
        slot,
        len(documents),
        total_pages,
        text_pages,
    )
