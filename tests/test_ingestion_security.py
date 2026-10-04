"""Functional hostile-format fixtures. No real records or network/provider calls."""
from __future__ import annotations

import codecs
import hashlib
import io
import os
import stat
import struct
import unittest
import warnings
import zipfile
from unittest.mock import patch

from tests import hermetic  # noqa: F401
from tests.ingestion_fixtures import CONTENT_TYPES, RELATIONSHIPS, docx_parts, package
from app import config, ingestion_policy as policy, parser_protocol as wire
from app.documents import ExtractionError, InProcessExtractor, extract_document, extract_uploaded_documents
from tests.test_extractors import _pdf_bytes


class TextAndLabelSecurity(unittest.TestCase):
    def test_spoofed_mime_type_cannot_bypass_uploaded_file_parser(self):
        class File:
            name = "renamed.txt"
            type = "text/plain"
            size = 16

            def getvalue(self):
                return b"\x89PNG\r\n\x1a\n12345678"

        documents, skipped = extract_uploaded_documents([File()])
        self.assertFalse(documents)
        self.assertEqual(len(skipped), 1)
        self.assertIn("Binary content", skipped[0])

    def test_binary_signatures_cannot_be_renamed_as_text(self):
        for prefix in policy._BINARY_PREFIXES:
            with self.subTest(prefix=prefix), self.assertRaises(ExtractionError):
                extract_document("note.txt", prefix + b"Some plausible clinical text.")

    def test_unicode_boms_preserve_critical_evidence(self):
        text = "Élodie did NOT receive 10 mg on 2024-02-03. β = −1; 37°C."
        encodings = ((codecs.BOM_UTF8, "utf-8"), (codecs.BOM_UTF16_LE, "utf-16-le"),
                     (codecs.BOM_UTF16_BE, "utf-16-be"), (codecs.BOM_UTF32_LE, "utf-32-le"),
                     (codecs.BOM_UTF32_BE, "utf-32-be"))
        for bom, encoding in encodings:
            with self.subTest(encoding=encoding):
                self.assertEqual(extract_document("note.txt", bom + text.encode(encoding)).full_text, text)

    def test_controls_invalid_encoding_and_malformed_bom_are_refused(self):
        for data in (b"A\x00B", b"A\x1bB", b"A\x7fB", b"A\xc2\x85B", b"A\xffB", b"MZ\x00\x03\x00",
                     b"N\x00o\x00", codecs.BOM_UTF16_LE + b"X", codecs.BOM_UTF8 + b"\xff",
                     codecs.BOM_UTF16_LE + "%PDF-1.7\n".encode("utf-16-le")):
            with self.subTest(data=data), self.assertRaises(ExtractionError):
                extract_document("note.md", data)

    def test_safe_nested_unicode_labels_are_preserved(self):
        doc = extract_document("2024/Élodie’s note.txt", b"No medication was prescribed.")
        self.assertEqual(doc.filename, "2024/Élodie’s note.txt")
        self.assertEqual(extract_document("note.txt", b"BM normal. No bleeding.").full_text, "BM normal. No bleeding.")

    def test_unsafe_labels_are_rejected_at_document_and_protocol_boundary(self):
        for name in ("../note.txt", "/note.txt", "C:/note.txt", "folder\\note.txt", "a//b.txt",
                     "a/./b.txt", "[citation].txt", "note\x00.txt", "bad\n.txt", "bad\u202e.txt",
                     "a/ note.txt", " note.txt", "a/../note.txt", "x\ud800.txt"):
            with self.subTest(name=name):
                with self.assertRaises(ExtractionError):
                    extract_document(name, b"Synthetic note.")
                with self.assertRaises(wire.ParserRefused):
                    wire.validate_request({"version": 1, "label": name, "size": 1,
                        "sha256": "a" * 64, "nonce": "b" * 32, "page_limit": 500})

    def test_extension_and_structural_signatures_must_agree(self):
        for name, body in (("note.pdf", package({"one.txt": b"x"})), ("note.docx", b"%PDF-1.7\n"),
                           ("note.zip", b"Plain text"), ("note.pdf", b"%PDF-1.7\ncorrupt")):
            with self.subTest(name=name), self.assertRaises(ExtractionError):
                InProcessExtractor().extract(name, body) if name.endswith(".zip") else extract_document(name, body)


class ArchiveSecurity(unittest.TestCase):
    def test_safe_archive_preserves_distinct_nested_citations(self):
        docs, skips = InProcessExtractor().extract("records.zip", package({
            "2023/note.txt": b"First synthetic observation.", "2024/note.txt": b"Second synthetic observation."}))
        self.assertFalse(skips)
        self.assertEqual([d.filename for d in docs], ["records/2023/note.txt", "records/2024/note.txt"])

    def test_unsafe_member_refuses_whole_archive_before_any_body_is_opened(self):
        for name in ("../bad.txt", "/absolute.txt", "C:/bad.txt", "a\\b.txt", "a//b.txt", "a/[b].txt"):
            data = package({"good.txt": b"Synthetic note.", name: b"Unsafe note."})
            with self.subTest(name=name), patch.object(zipfile.ZipFile, "open") as opened:
                with self.assertRaises(ExtractionError):
                    InProcessExtractor().extract("records.zip", data)
                opened.assert_not_called()

    def test_duplicate_unicode_and_case_collisions_are_refused(self):
        for pair in (("same.txt", "same.txt"), ("Note.txt", "note.txt"), ("é.txt", "e\u0301.txt")):
            output = io.BytesIO()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(output, "w") as archive:
                    for name in pair:
                        archive.writestr(name, b"Synthetic observation.")
            with self.subTest(pair=pair), self.assertRaises(ExtractionError):
                InProcessExtractor().extract("records.zip", output.getvalue())

    def test_null_truncation_links_and_unsupported_compression_are_refused(self):
        info = zipfile.ZipInfo("link.txt")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr(info, b"../../target")
        null_name = package({"noteX.txt": b"Synthetic note."}).replace(b"noteX.txt", b"note\x00.txt")
        for data in (output.getvalue(), null_name, package({"note.txt": b"Synthetic note."}, zipfile.ZIP_BZIP2)):
            with self.subTest(kind=data[:30]), self.assertRaises(ExtractionError):
                InProcessExtractor().extract("records.zip", data)

    def test_encrypted_flag_is_refused_before_reading(self):
        data = bytearray(package({"note.txt": b"Synthetic note."}))
        for signature, offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
            start = data.index(signature) + offset
            flags = struct.unpack_from("<H", data, start)[0]
            struct.pack_into("<H", data, start, flags | 1)
        with patch.object(zipfile.ZipFile, "open") as opened, self.assertRaises(ExtractionError):
            InProcessExtractor().extract("records.zip", bytes(data))
        opened.assert_not_called()

    def test_existing_expansion_and_ratio_quotas_remain_effective(self):
        data = package({"note.txt": b"A" * 10000}, zipfile.ZIP_DEFLATED)
        with patch.object(config, "ZIP_MAX_COMPRESSION_RATIO", 2):
            docs, skipped = InProcessExtractor().extract("records.zip", data)
        self.assertFalse(docs)
        self.assertIn("compression ratio", skipped[0])
        data = package({"one.txt": b"A" * 100, "two.txt": b"B" * 100})
        with patch.object(config, "ZIP_MAX_TOTAL_UNCOMPRESSED_BYTES", 150), self.assertRaises(ExtractionError):
            InProcessExtractor().extract("records.zip", data)


class DocxSecurity(unittest.TestCase):
    def test_valid_passive_package_has_readable_text(self):
        self.assertIn("No diagnosis established.", extract_document("note.docx", package(docx_parts())).full_text)

    def test_passive_styles_and_thumbnail_from_word_packages_are_supported(self):
        parts = docx_parts()
        parts["word/stylesWithEffects.xml"] = b'<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>'
        parts["docProps/thumbnail.jpeg"] = b'\xff\xd8\xff\xe0Synthetic inert image header'
        parts["[Content_Types].xml"] = CONTENT_TYPES.replace(b'</Types>',
            b'<Default Extension="jpeg" ContentType="image/jpeg"/>'
            b'<Override PartName="/word/stylesWithEffects.xml" ContentType="application/vnd.ms-word.stylesWithEffects+xml"/></Types>')
        parts["_rels/.rels"] = RELATIONSHIPS.replace(b'</Relationships>',
            b'<Relationship Id="thumb" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail" Target="docProps/thumbnail.jpeg"/></Relationships>')
        parts["word/_rels/document.xml.rels"] = (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="effects" Type="http://schemas.microsoft.com/office/2007/relationships/stylesWithEffects" '
            'Target="stylesWithEffects.xml"/></Relationships>').encode()
        self.assertIn("No diagnosis established.", extract_document("note.docx", package(parts)).full_text)

    def test_duplicate_content_type_or_main_relationship_is_refused(self):
        for member, original, element in (
                ("[Content_Types].xml", CONTENT_TYPES, b'<Default Extension="xml" ContentType="application/xml"/>'),
                ("_rels/.rels", RELATIONSHIPS, b'<Relationship Id="second" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>')):
            parts = docx_parts()
            tail = b'</Types>' if member.startswith('[') else b'</Relationships>'
            parts[member] = original.replace(tail, element + tail)
            with self.subTest(member=member), self.assertRaises(ExtractionError):
                extract_document("note.docx", package(parts))

    def test_missing_structure_and_wrong_main_content_type_are_refused(self):
        for member in ("[Content_Types].xml", "_rels/.rels", "word/document.xml"):
            parts = docx_parts(); parts.pop(member)
            with self.subTest(member=member), self.assertRaises(ExtractionError):
                extract_document("note.docx", package(parts))
        parts = docx_parts()
        parts["[Content_Types].xml"] = CONTENT_TYPES.replace(b"document.main+xml", b"template.main+xml")
        with self.assertRaises(ExtractionError):
            extract_document("note.docx", package(parts))

    def test_macro_and_embedded_parts_are_refused(self):
        for name, kind in (("word/vbaProject.bin", "application/vnd.ms-office.vbaProject"),
                           ("word/embeddings/object.bin", "application/vnd.openxmlformats-officedocument.oleObject"),
                           ("word/media/image.svg", "image/svg+xml")):
            parts = docx_parts()
            parts[name] = b"Synthetic inert payload"
            parts["[Content_Types].xml"] = CONTENT_TYPES.replace(b"</Types>",
                f'<Override PartName="/{name}" ContentType="{kind}"/></Types>'.encode())
            with self.subTest(name=name), self.assertRaises(ExtractionError):
                extract_document("note.docx", package(parts))

    def test_external_active_missing_and_escaping_relationships_are_refused(self):
        prefix = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
        for reltype, target, mode in (("hyperlink", "https://example.invalid", "External"),
                                     ("attachedTemplate", "settings.xml", "Internal"),
                                     ("styles", "../../../outside.xml", "Internal"),
                                     ("styles", "missing.xml", "Internal"),
                                     ("styles", "%2e%2e/%2e%2e/outside.xml", "Internal")):
            parts = docx_parts()
            parts["word/_rels/document.xml.rels"] = (
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                f'<Relationship Id="x" Type="{prefix}{reltype}" Target="{target}" TargetMode="{mode}"/>'
                '</Relationships>').encode()
            with self.subTest(target=target, reltype=reltype), self.assertRaises(ExtractionError):
                extract_document("note.docx", package(parts))

    def test_internal_relative_relationships_can_resolve_inside_package(self):
        parts = docx_parts(); parts["customXml/item1.xml"] = b"<root>Passive metadata</root>"
        parts["word/_rels/document.xml.rels"] = (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="x" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/customXml" '
            'Target="../customXml/item1.xml"/></Relationships>').encode()
        self.assertIn("Synthetic knee", extract_document("note.docx", package(parts)).full_text)

    def test_entity_declarations_and_processing_instructions_in_any_xml_part_are_refused(self):
        for payload in (b'<!DOCTYPE root [<!ENTITY x "expanded">]><root>&x;</root>',
                        b'<!DOCTYPE root SYSTEM "file:///etc/passwd"><root/>',
                        b'<?fetch uri="https://example.invalid"?><root/>',
                        '<!DOCTYPE root [<!ENTITY x "expanded">]><root>&x;</root>'.encode("utf-16")):
            parts = docx_parts(); parts["customXml/item1.xml"] = payload
            with self.subTest(payload=payload[:24]), self.assertRaises(ExtractionError):
                extract_document("note.docx", package(parts))

    def test_malformed_main_xml_is_a_named_refusal(self):
        parts = docx_parts(); parts["word/document.xml"] = b"<unclosed>"
        with self.assertRaisesRegex(ExtractionError, "malformed"):
            extract_document("note.docx", package(parts))

    def test_xml_node_and_depth_limits(self):
        for payload, setting in ((b"<r><x/><y/></r>", "XML_MAX_NODES"), (b"<r><x><y/></x></r>", "XML_MAX_DEPTH")):
            parts = docx_parts(); parts["customXml/item1.xml"] = payload
            with patch.object(policy, setting, 2), self.subTest(setting=setting), self.assertRaises(ExtractionError):
                extract_document("note.docx", package(parts))

    def test_active_tags_and_split_field_instructions_are_refused(self):
        for xml in ('<w:object/>', '<w:altChunk/>', '<w:control/>',
                    '<w:r><w:instrText>DD</w:instrText></w:r><w:r><w:instrText>EAUTO payload</w:instrText></w:r>'):
            parts = docx_parts()
            parts["word/document.xml"] = parts["word/document.xml"].replace(b"</w:body>", xml.encode() + b"</w:body>")
            with self.subTest(xml=xml), self.assertRaises(ExtractionError):
                extract_document("note.docx", package(parts))

    def test_mislabeled_executable_image_is_refused(self):
        parts = docx_parts(); parts["word/media/image.png"] = b"MZsynthetic nonexecuting payload"
        parts["[Content_Types].xml"] = CONTENT_TYPES.replace(b"</Types>", b'<Default Extension="png" ContentType="image/png"/></Types>')
        with self.assertRaisesRegex(ExtractionError, "image content"):
            extract_document("note.docx", package(parts))


class PdfSecurity(unittest.TestCase):
    def writer(self):
        from pypdf import PdfReader, PdfWriter
        return PdfWriter(clone_from=PdfReader(io.BytesIO(_pdf_bytes(["Synthetic evidence. No medication prescribed." ]))))

    def serialized(self, writer):
        output = io.BytesIO(); writer.write(output); return output.getvalue()

    def test_passive_pdf_remains_readable(self):
        self.assertIn("No medication prescribed", extract_document("note.pdf", self.serialized(self.writer())).full_text)

    def test_javascript_attachment_and_encryption_are_refused(self):
        for mutation in (lambda w: w.add_js('app.alert("synthetic");'),
                         lambda w: w.add_attachment("note.txt", b"Synthetic attachment."),
                         lambda w: w.encrypt("synthetic-password")):
            writer = self.writer(); mutation(writer)
            with self.subTest(mutation=mutation), self.assertRaises(ExtractionError):
                extract_document("note.pdf", self.serialized(writer))

    def test_nested_annotation_actions_and_media_are_refused_before_text_extraction(self):
        from pypdf.generic import DictionaryObject, NameObject, ArrayObject, TextStringObject
        for key in ("/A", "/AA", "/OpenAction", "/XFA", "/RichMediaContent", "/3DD"):
            writer = self.writer()
            annotation = DictionaryObject({NameObject(key): TextStringObject("synthetic")})
            writer.pages[0][NameObject("/Annots")] = ArrayObject([writer._add_object(annotation)])
            with self.subTest(key=key), patch('app.documents._page_text') as parsed:
                with self.assertRaises(ExtractionError):
                    extract_document("note.pdf", self.serialized(writer))
                parsed.assert_not_called()

    def test_cyclic_passive_graph_does_not_loop_or_inflate_streams(self):
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
        writer = self.writer()
        node = DictionaryObject(); ref = writer._add_object(node); node[NameObject("/Back")] = ref
        writer._root_object[NameObject("/SyntheticCycle")] = ref
        # The policy traverses dictionaries only; PDF text extraction happens later.
        with patch.object(DecodedStreamObject, "get_data", side_effect=AssertionError("inflated")):
            from pypdf import PdfReader
            policy.validate_pdf(PdfReader(io.BytesIO(self.serialized(writer)), strict=True))

    def test_pdf_graph_budget_is_enforced(self):
        with patch.object(policy, "PDF_MAX_OBJECTS", 5), self.assertRaises(ExtractionError):
            extract_document("note.pdf", self.serialized(self.writer()))

    def test_indirect_active_type_and_non_name_type_cannot_bypass_checks(self):
        from pypdf.generic import DictionaryObject, NameObject
        for value in (NameObject('/Filespec'), DictionaryObject({NameObject('/Synthetic'): NameObject('/Value')})):
            writer = self.writer()
            holder = DictionaryObject({NameObject('/Type'): writer._add_object(value)})
            writer._root_object[NameObject('/SyntheticNestedType')] = writer._add_object(holder)
            with self.subTest(value_type=type(value).__name__), self.assertRaisesRegex(ExtractionError, 'refused|object type'):
                extract_document("note.pdf", self.serialized(writer))

    def test_no_mode_setting_can_disable_the_policy(self):
        for mode in ("controlled-pilot", "synthetic", ""):
            with patch.dict(os.environ, {"VA_LSE_MODE": mode}):
                writer = self.writer(); writer.add_js("synthetic")
                with self.assertRaises(ExtractionError):
                    extract_document("note.pdf", self.serialized(writer))
