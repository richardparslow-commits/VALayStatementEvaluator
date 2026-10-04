"""Narrow passive-document policy, enforced even in the secret-free parser child.

These checks are format validation, not antivirus or content disarm/reconstruction.
Never open, execute, render, or fetch relationships from an uploaded document.
"""
from __future__ import annotations

import codecs
import posixpath
import re
import stat
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from typing import Any, Callable, Iterable
from urllib.parse import unquote, urlsplit



class IngestionRefused(ValueError):
    """Fixed policy messages contain neither document text nor parser diagnostics."""


def validate_label(label: str, *, limit: int = 1024) -> None:
    if (not isinstance(label, str) or not 0 < len(label) <= limit
            or "\\" in label or any(c in label for c in "[]:")
            or any(unicodedata.category(c) in {"Cc", "Cf", "Cs"} for c in label)
            or any(part in {"", ".", ".."} or part != part.strip() for part in label.split("/"))):
        raise IngestionRefused("Unsafe or ambiguous file label. Rename the file before uploading.")


_BINARY_PREFIXES = (b"%PDF-", b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08",
                    b"\x89PNG", b"\xff\xd8\xff", b"GIF87a", b"GIF89a",
                    b"\xd0\xcf\x11\xe0", b"\x7fELF", b"\x1f\x8b",
                    b"7z\xbc\xaf\x27\x1c", b"Rar!", b"SQLite format 3\x00")


def decode_text(data: bytes) -> str:
    # Longest BOM first: UTF-32 LE shares its first two bytes with UTF-16 LE.
    encoding = "utf-8"
    for bom, candidate in ((codecs.BOM_UTF32_LE, "utf-32"), (codecs.BOM_UTF32_BE, "utf-32"),
                           (codecs.BOM_UTF16_LE, "utf-16"), (codecs.BOM_UTF16_BE, "utf-16"),
                           (codecs.BOM_UTF8, "utf-8-sig")):
        if data.startswith(bom):
            encoding = candidate
            break
    if data.startswith(_BINARY_PREFIXES):
        raise IngestionRefused("Binary content cannot be uploaded as TXT or MD.")
    try:
        text = data.decode(encoding, errors="strict")
    except UnicodeError as exc:
        raise IngestionRefused("Text encoding is unsupported or damaged. Save as UTF-8 and upload again.") from exc
    if (text.lstrip("\ufeff \t\r\n")[:32].encode("utf-8").startswith(_BINARY_PREFIXES)
            or any((unicodedata.category(c) in {"Cc", "Cs"} and c not in "\t\r\n\f") for c in text)):
        raise IngestionRefused("Text contains binary content or unsupported control characters.")
    return text


def validate_signature(suffix: str, data: bytes) -> None:
    if suffix == ".pdf" and not re.match(rb"%PDF-[12]\.[0-9](?:\r|\n|\s)", data[:12]):
        raise IngestionRefused("The file does not have a supported PDF header.")
    if suffix in {".docx", ".zip"} and not data.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        raise IngestionRefused("The file does not have a ZIP package header.")


def validate_zip_entries(archive: zipfile.ZipFile, *, package: bool = False) -> None:
    names: set[str] = set()
    for info in archive.infolist():
        name = info.filename.rstrip("/") if info.is_dir() else info.filename
        # ZipInfo silently truncates at NUL; examine orig_filename as well.
        if info.orig_filename != info.filename:
            raise IngestionRefused("Archive member name contains a null byte.")
        if package and name == "[Content_Types].xml":
            pass  # The one OPC-required name containing citation delimiters.
        else:
            validate_label(name, limit=1024)
        normalized = unicodedata.normalize("NFC", name).casefold()
        if normalized in names:
            raise IngestionRefused("Archive contains duplicate or ambiguous member names.")
        names.add(normalized)
        mode = info.external_attr >> 16
        if stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise IngestionRefused("Archive links and special files are not supported.")
        if info.flag_bits & 1:
            raise IngestionRefused("Encrypted archive members are not supported.")
        if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
            raise IngestionRefused("Archive compression method is not supported.")


XML_MAX_NODES = 200_000
XML_MAX_DEPTH = 128


class _PassiveTreeBuilder(ET.TreeBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.nodes = self.depth = 0

    def doctype(self, name: str, pubid: str | None, system: str | None) -> None:
        raise IngestionRefused("XML document types and entities are not supported.")

    def pi(self, target: str, text: str | None = None) -> ET.Element:
        raise IngestionRefused("XML processing instructions are not supported.")

    def start(self, tag: str, attrs: dict[str, str]) -> ET.Element:
        self.nodes += 1
        self.depth += 1
        if self.nodes > XML_MAX_NODES or self.depth > XML_MAX_DEPTH:
            raise IngestionRefused("XML structure exceeds the processing limit.")
        return super().start(tag, attrs)

    def end(self, tag: str) -> ET.Element:
        self.depth -= 1
        return super().end(tag)


def passive_xml(data: bytes) -> ET.Element:
    try:
        return ET.fromstring(data, parser=ET.XMLParser(target=_PassiveTreeBuilder()))
    except ET.ParseError as exc:
        raise IngestionRefused("DOCX contains malformed or unsupported XML.") from exc


_WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_OFFICE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
_MAIN_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
_XML_TYPES = {"application/xml", "text/xml", "application/vnd.openxmlformats-package.relationships+xml",
              "application/vnd.openxmlformats-package.core-properties+xml",
              "application/vnd.ms-word.stylesWithEffects+xml", _MAIN_TYPE}
_XML_TYPES.update("application/vnd.openxmlformats-officedocument." + suffix + "+xml" for suffix in (
    "extended-properties", "custom-properties", "theme", "customXmlProperties",
    *("wordprocessingml." + part for part in ("styles", "numbering", "settings", "webSettings",
      "fontTable", "header", "footer", "footnotes", "endnotes", "comments"))))
_IMAGE_PREFIXES = {"image/png": (b"\x89PNG\r\n\x1a\n",), "image/jpeg": (b"\xff\xd8\xff",),
                   "image/gif": (b"GIF87a", b"GIF89a"), "image/bmp": (b"BM",),
                   "image/tiff": (b"II*\x00", b"MM\x00*")}
_REL_TYPES = {_OFFICE_REL + suffix for suffix in (
    "officeDocument", "styles", "numbering", "settings", "webSettings", "fontTable", "theme",
    "header", "footer", "footnotes", "endnotes", "comments", "image", "hyperlink", "customXml",
    "customXmlProps", "extended-properties", "custom-properties")}
_REL_TYPES.add(_REL_NS + "/metadata/core-properties")
_REL_TYPES.add(_REL_NS + "/metadata/thumbnail")
_REL_TYPES.add("http://schemas.microsoft.com/office/2007/relationships/stylesWithEffects")


def _relationship_target(source: str, target: str) -> str:
    decoded = unquote(target)
    uri = urlsplit(decoded)
    if uri.scheme or uri.netloc or uri.query or uri.fragment or "\\" in decoded:
        raise IngestionRefused("DOCX external or ambiguous relationships are not supported.")
    path = posixpath.normpath(posixpath.join(posixpath.dirname(source), decoded))
    if decoded.startswith("/"):
        path = posixpath.normpath(decoded.lstrip("/"))
    validate_label(path)
    return path


def validate_docx(archive: zipfile.ZipFile, read: Callable[[str], bytes]) -> ET.Element:
    """Read each bounded part once; retain only main XML, not all package bodies."""
    validate_zip_entries(archive, package=True)
    names = {i.filename for i in archive.infolist() if not i.is_dir()}
    if not {"[Content_Types].xml", "_rels/.rels", "word/document.xml"} <= names:
        raise IngestionRefused("DOCX is missing required package structure.")
    types = passive_xml(read("[Content_Types].xml"))
    if types.tag != f"{{{_CT_NS}}}Types":
        raise IngestionRefused("DOCX content type structure is invalid.")
    defaults: dict[str, str] = {}
    overrides: dict[str, str] = {}
    for node in types:
        kind = node.get("ContentType", "")
        if kind not in _XML_TYPES and kind not in _IMAGE_PREFIXES:
            raise IngestionRefused("DOCX macros, embedded objects, or unsupported part types are refused.")
        if node.tag == f"{{{_CT_NS}}}Default":
            extension = node.get("Extension", "").lower()
            if not extension or extension in defaults:
                raise IngestionRefused("DOCX content types are ambiguous.")
            defaults[extension] = kind
        elif node.tag == f"{{{_CT_NS}}}Override":
            part = node.get("PartName", "")
            if not part.startswith("/") or part[1:] in overrides or part[1:] not in names:
                raise IngestionRefused("DOCX content types are ambiguous.")
            overrides[part[1:]] = kind
        else:
            raise IngestionRefused("DOCX content type structure is invalid.")
    main: ET.Element | None = None
    has_main_relationship = False
    for name in sorted(names - {"[Content_Types].xml"}):
        kind = overrides.get(name, defaults.get(name.rsplit(".", 1)[-1].lower(), ""))
        if not kind or (name == "word/document.xml" and kind != _MAIN_TYPE):
            raise IngestionRefused("DOCX part has a missing or unsupported content type.")
        body = read(name)
        if kind in _IMAGE_PREFIXES:
            if not body.startswith(_IMAGE_PREFIXES[kind]):
                raise IngestionRefused("DOCX image content does not match its declared type.")
            continue
        root = passive_xml(body)
        if name.endswith(".rels"):
            if kind != "application/vnd.openxmlformats-package.relationships+xml" or root.tag != f"{{{_REL_NS}}}Relationships":
                raise IngestionRefused("DOCX relationship structure is invalid.")
            if name == "_rels/.rels":
                source = ""
            elif "/_rels/" in name:
                parent, tail = name.rsplit("/_rels/", 1)
                source = parent + "/" + tail[:-5]
                if source not in names:
                    raise IngestionRefused("DOCX relationship source is missing.")
            else:
                raise IngestionRefused("DOCX relationship location is invalid.")
            ids: set[str] = set()
            for rel in root:
                identifier, reltype = rel.get("Id", ""), rel.get("Type", "")
                if (rel.tag != f"{{{_REL_NS}}}Relationship" or not identifier or identifier in ids
                        or reltype not in _REL_TYPES or rel.get("TargetMode", "Internal") != "Internal"):
                    raise IngestionRefused("DOCX active, external, or ambiguous relationships are refused.")
                ids.add(identifier)
                target = _relationship_target(source, rel.get("Target", ""))
                if target not in names:
                    raise IngestionRefused("DOCX relationship target is missing.")
                if not source and reltype == _OFFICE_REL + "officeDocument":
                    if target != "word/document.xml" or has_main_relationship:
                        raise IngestionRefused("DOCX main document relationship is ambiguous.")
                    has_main_relationship = True
        instructions: list[str] = []
        for node in root.iter():
            if node.tag in {f"{{{_WORD_NS}}}{n}" for n in ("object", "control", "altChunk", "attachedTemplate")}:
                raise IngestionRefused("DOCX active or embedded content is refused.")
            if node.tag == f"{{{_WORD_NS}}}instrText":
                instructions.append(node.text or "")
            if node.tag == f"{{{_WORD_NS}}}fldSimple":
                instructions.append(node.get(f"{{{_WORD_NS}}}instr", ""))
        if re.search(r"\b(?:DDEAUTO|DDE|INCLUDETEXT|INCLUDEPICTURE|LINK|MACROBUTTON|EMBED)\b", "".join(instructions), re.I):
            raise IngestionRefused("DOCX active field instructions are refused.")
        if name == "word/document.xml":
            if root.tag != f"{{{_WORD_NS}}}document":
                raise IngestionRefused("DOCX main document XML is invalid.")
            main = root
    if main is None or not has_main_relationship:
        raise IngestionRefused("DOCX main document relationship is missing.")
    return main


PDF_MAX_OBJECTS = 100_000
PDF_MAX_DEPTH = 128
_PDF_ACTIVE_KEYS = {"/OpenAction", "/AA", "/A", "/JS", "/JavaScript", "/EmbeddedFiles", "/EF",
                    "/XFA", "/RichMedia", "/RichMediaContent", "/RichMediaSettings", "/3DD",
                    "/Movie", "/Sound", "/Collection"}


def validate_pdf(reader: Any) -> None:
    """Traverse reachable object dictionaries without inflating content streams."""
    from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject, NameObject
    pending = [(reader.trailer, 0)]
    seen: set[tuple[Any, ...]] = set()
    count = 0
    while pending:
        value, depth = pending.pop()
        count += 1
        if count > PDF_MAX_OBJECTS or depth > PDF_MAX_DEPTH:
            raise IngestionRefused("PDF structure exceeds the processing limit.")
        if isinstance(value, IndirectObject):
            identity: tuple[Any, ...] = ("ref", value.idnum, value.generation)
            if identity in seen:
                continue
            seen.add(identity)
            pending.append((value.get_object(), depth))
        elif isinstance(value, (DictionaryObject, ArrayObject)):
            identity = ("object", id(value))
            if identity in seen:
                continue
            seen.add(identity)
            if isinstance(value, DictionaryObject):
                kind, subtype = value.get("/Type"), value.get("/Subtype")
                if isinstance(kind, IndirectObject):
                    kind = kind.get_object()
                if isinstance(subtype, IndirectObject):
                    subtype = subtype.get_object()
                if any(t is not None and not isinstance(t, NameObject) for t in (kind, subtype)):
                    raise IngestionRefused("PDF object type is invalid or unsupported.")
                if (_PDF_ACTIVE_KEYS.intersection(value)
                        or kind in {"/Action", "/EmbeddedFile", "/Filespec"}
                        or subtype in {"/RichMedia", "/Movie", "/Sound", "/3D"}):
                    raise IngestionRefused("PDF actions, attachments, and active media are refused. Upload a passive copy.")
                children: Iterable[Any] = value.values()
            else:
                children = value
            # Bound the work list as well as visited nodes, before copying a wide graph.
            for child in children:
                if len(pending) + count >= PDF_MAX_OBJECTS:
                    raise IngestionRefused("PDF structure exceeds the processing limit.")
                pending.append((child, depth + 1))
