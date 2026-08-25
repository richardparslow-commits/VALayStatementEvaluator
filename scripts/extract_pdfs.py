"""Extract text from reference PDFs into the project's reference_docs/extracted folder."""
import re
import sys
from pathlib import Path

from pypdf import PdfReader

DESKTOP = Path("/Users/richardparslow/Desktop")
OUT = Path(__file__).resolve().parent.parent / "reference_docs" / "extracted"

FILES = [
    "Lay Evidence in Veterans Affairs Disability Claims.pdf",
    "The Role and Legal Framework of Witness Lay Statements in Veterans Affairs Disability Adjudication.pdf",
    "Writing a Strong Statement in Support of Claim for VA Benefits _ CCK Law.pdf",
    "Witness statement.pdf",
    "Wife VA Buddy Letter EXAMPLE to Support Your VA Disability Claim.pdf",
    "VA Disability Letter Example_ 6 Tips for Strong Claims - Homefront Group.pdf",
    "VA Buddy Letter Guide_ How to Write Effective Lay Statements for Disability Claims _ LinkedIn.pdf",
    "Lay Statement Examples for VA Claims 2026, Sample Personal and Buddy Statements _ VA Claims US.pdf",
]


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[:80]


def extract(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            text = f"[page {i + 1} extraction error: {exc}]"
        parts.append(f"\n\n----- Page {i + 1} -----\n{text}")
    return "".join(parts)


def main() -> None:
    extra = [Path(a) if Path(a).is_absolute() else DESKTOP / a for a in sys.argv[1:]]
    OUT.mkdir(parents=True, exist_ok=True)
    for src in [DESKTOP / f for f in FILES] + extra:
        if not src.exists():
            print(f"MISSING: {src}")
            continue
        text = extract(src)
        dest = OUT / f"{slug(src.stem)}.txt"
        dest.write_text(text, encoding="utf-8")
        print(f"{src.name}: {len(text)} chars -> {dest.name}")


if __name__ == "__main__":
    main()
