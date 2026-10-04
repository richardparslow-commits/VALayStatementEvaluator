"""Small synthetic OPC packages with actual required content types/relationships."""
from __future__ import annotations

import io
import zipfile
from xml.sax.saxutils import escape

CONTENT_TYPES = b'''<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>'''
RELATIONSHIPS = b'''<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''


def docx_parts(text: str = "Synthetic knee pain noted. No diagnosis established.") -> dict[str, bytes]:
    return {"[Content_Types].xml": CONTENT_TYPES, "_rels/.rels": RELATIONSHIPS,
            "word/document.xml": ('<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                                  '<w:body><w:p><w:r><w:t>' + escape(text) +
                                  '</w:t></w:r></w:p></w:body></w:document>').encode()}


def package(parts: dict[str, bytes], compression: int = zipfile.ZIP_STORED) -> bytes:
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", compression=compression) as archive:
        for name, body in parts.items():
            archive.writestr(name, body)
    return result.getvalue()
