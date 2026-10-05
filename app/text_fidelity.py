"""Unicode retrieval and whole-sentence excerpts; never prefix-cut testimony."""
from __future__ import annotations

import re
import unicodedata


def words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFC", text).casefold(), re.UNICODE)


def evidence_excerpt(text: str, query: str, limit: int) -> str:
    """Return contiguous complete sentences around the best literal query hit.

    Empty means a relevant sentence cannot fit, not absence of evidence. Callers
    must disclose that omission. Sentence boundaries are deterministic, not a
    claim of linguistic or clinical understanding.
    """
    if limit < 1:
        return ""
    if len(text) <= limit:
        return text
    units = [m for m in re.finditer(r".+?(?:[.!?](?=\s|$)|\n+|$)", text, re.S) if m.group().strip()]
    if not units:
        return ""
    tokens = set(words(query))
    index = max(range(len(units)), key=lambda i: len(tokens & set(words(units[i].group()))))
    start, end = units[index].span()
    if end - start > limit:
        return ""
    # Keep nearby date/attribution context when it fits without cutting a unit.
    left = right = index
    while left > 0 or right < len(units) - 1:
        if left > 0 and end - units[left - 1].start() <= limit:
            left -= 1
            start = units[left].start()
        elif right < len(units) - 1 and units[right + 1].end() - start <= limit:
            right += 1
            end = units[right].end()
        else:
            break
    excerpt = text[start:end].strip()
    return ("[Earlier text omitted]\n" if start else "") + excerpt + ("\n[Later text omitted]" if end < len(text) else "")
