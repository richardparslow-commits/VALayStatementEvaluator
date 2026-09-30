"""Exact, collision-safe addresses shared by evaluation and drafting."""
from .documents import BLOCK, PAGE, DocumentPage, ExtractedDocument


def source_reference_key(reference: str) -> str:
    return " ".join(reference.split()).casefold()


def source_reference_aliases(filename: str, kind: str, number: int) -> set[str]:
    prefix = "p." if kind == PAGE else "b."
    labels = (
        f"{filename} {prefix}{number}",
        f"{filename} — {kind} {number}",
        f"{filename} - {kind} {number}",
    )
    return {
        source_reference_key(alias)
        for label in labels for alias in (label, f"[{label}]")
    }


def build_source_index(records: list[ExtractedDocument]) -> dict[str, DocumentPage | None]:
    """Index exact single-unit aliases; collisions are deliberately unusable.

    An uploaded filename is not necessarily unique. Even duplicate pages with
    identical text remain ambiguous: we cannot tell which upload was cited.
    Unreadable units reserve their aliases so a duplicate readable unit cannot
    disguise the ambiguity. No model-generated digest labels are trusted here.
    """
    index: dict[str, DocumentPage | None] = {}
    for document in records:
        # Image-only pages are normally absent from extracted units. Reserve
        # these addresses too, including collisions with another upload.
        if document.pagination in (PAGE, BLOCK):
            for number in document.unreadable_pages:
                if type(number) is int and number > 0:
                    for alias in source_reference_aliases(document.filename, document.pagination, number):
                        index[alias] = None
        for unit in document.pages:
            if unit.kind not in (PAGE, BLOCK) or type(unit.page) is not int or unit.page < 1:
                continue
            readable = (
                unit.filename == document.filename
                and bool(unit.text.strip())
                and unit.page not in document.unreadable_pages
            )
            canonical = unit if readable else None
            # Reserve both names on a malformed/mislabelled extracted unit.
            for filename in {document.filename, unit.filename}:
                for alias in source_reference_aliases(filename, unit.kind, unit.page):
                    index[alias] = None if alias in index else canonical
    return index
