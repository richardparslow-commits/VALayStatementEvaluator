"""Ingest quality: what the app can read out of uploaded files, and what it says.

Every defect covered here shares one shape — the pipeline produced a plausible
answer while having silently lost something: pages that never became text, facts
cited to a chunk instead of a page, a corrected re-upload served from a stale
cache. None of them raise, which is exactly why they need tests.
"""
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.documents import (  # noqa: E402
    BLOCK,
    PAGE,
    DocumentPage,
    ExtractionError,
    ExtractedDocument,
    chunk_page_labelled_text,
    clean_text,
    document_from_text,
    paragraph_index,
    strip_running_headers,
)
from app.medical_review import (  # noqa: E402
    MedicalDigest,
    MedicalFact,
    _dates_in_text,
    _dedupe_pages,
    _fact_from_raw,
    _llm_infer_undated,
    _parse_citation,
    _page_shingles,
    _shingle_similarity,
    statement_element_for,
    verify_citations,
)
from app.va_gov_export import section_map  # noqa: E402


def _pdf_bytes(pages: list[str]) -> bytes:
    """A real PDF with the given text, one page per entry ("" = blank page).

    Each line is drawn as its own text object, the way a real document is laid
    out: drawing a multi-line string in one call makes the extractor report it as
    a single line, which would make these tests agree with a fiction rather than
    with the line structure the extraction code actually has to work on.
    """
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    for text in pages:
        y = 760
        for line in (text.split("\n") if text else []):
            pdf.drawString(72, y, line)
            y -= 14
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


class TestTextDecoding(unittest.TestCase):
    def test_cp1252_text_is_read_not_replaced(self) -> None:
        """A Windows-encoded export must not turn into mojibake in quotes."""
        from app.documents import extract_document

        # 0x92 is a right single quote in cp1252 and invalid UTF-8.
        raw = b"Patient\x92s knee pain began in service."
        doc = extract_document("notes.txt", raw)
        self.assertIn("Patient’s knee pain", doc.full_text)

    def test_utf8_still_wins(self) -> None:
        from app.documents import extract_document

        doc = extract_document("notes.md", "Dyspnée noted — accented.".encode("utf-8"))
        self.assertIn("Dyspnée", doc.full_text)


class TestParagraphGranularity(unittest.TestCase):
    def test_a_page_with_no_blank_lines_is_not_one_retrieval_unit(self) -> None:
        lines = "\n".join(f"line {i} about knee pain and limited walking" for i in range(200))
        doc = ExtractedDocument(
            filename="scan.pdf", pages=[DocumentPage("scan.pdf", 1, lines)]
        )
        paragraphs = paragraph_index(doc)
        self.assertGreater(len(paragraphs), 1)
        self.assertTrue(all(p.label == "scan.pdf p.1" for p in paragraphs))

    def test_short_blocks_are_left_whole(self) -> None:
        doc = ExtractedDocument(
            filename="note.pdf",
            pages=[DocumentPage("note.pdf", 1, "A single short clinical note about pain.")],
        )
        self.assertEqual(len(paragraph_index(doc)), 1)


class TestPdfExtraction(unittest.TestCase):
    def test_blank_pages_are_reported_not_dropped(self) -> None:
        from app.documents import _extract_pdf

        data = _pdf_bytes(["Patient reports knee pain since 2014.", "", "Pain 7/10 today."])
        doc = _extract_pdf("records.pdf", data)
        self.assertEqual(doc.total_pages, 3)
        self.assertEqual([p.page for p in doc.pages], [1, 3])
        self.assertEqual(doc.unreadable_pages, [2])
        self.assertEqual(doc.unreadable_count, 1)
        self.assertEqual(doc.source_page_count, 3)

    def test_a_password_protected_pdf_names_the_problem(self) -> None:
        """pypdf raises FileNotDecryptedError on page access; that must not escape."""
        from pypdf import PdfWriter

        from app.documents import _extract_pdf

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.encrypt("secret")
        buffer = io.BytesIO()
        writer.write(buffer)

        with self.assertRaises(ExtractionError) as raised:
            _extract_pdf("locked.pdf", buffer.getvalue())
        self.assertIn("password", str(raised.exception))

    def test_a_non_pdf_named_pdf_is_an_extraction_error(self) -> None:
        from app.documents import _extract_pdf

        with self.assertRaises(ExtractionError):
            _extract_pdf("fake.pdf", b"this is not a pdf at all")

    def test_running_headers_and_footers_are_stripped(self) -> None:
        from app.documents import _extract_pdf

        page = "VA.gov | My HealtheVet — medical records\nKnee pain noted on exam {n}.\nPage {n} of 5"
        data = _pdf_bytes([page.format(n=i) for i in range(1, 6)])
        doc = _extract_pdf("va.pdf", data)
        self.assertNotIn("VA.gov | My HealtheVet", doc.full_text)
        for index in range(1, 6):
            self.assertIn(f"Knee pain noted on exam {index}.", doc.full_text)

    def test_stripping_needs_enough_pages_to_be_meaningful(self) -> None:
        pages = [
            DocumentPage("short.pdf", i, "Repeated line\nReal content here.") for i in (1, 2)
        ]
        stripped = strip_running_headers(pages)
        self.assertEqual([p.text for p in stripped], [p.text for p in pages])

    def test_a_date_line_is_never_stripped_as_boilerplate(self) -> None:
        # A date that repeats on every page (same-day encounter stamp) is content:
        # removing it would delete the encounter date from the record.
        pages = [
            DocumentPage("day.pdf", i, f"2020-03-14\nNote body {i}.")
            for i in range(1, 6)
        ]
        stripped = strip_running_headers(pages)
        self.assertTrue(all("2020-03-14" in p.text for p in stripped))


class TestBlockAddressedText(unittest.TestCase):
    def test_long_text_is_addressable(self) -> None:
        text = clean_text("\n\n".join(f"Paragraph {i} about chronic pain." for i in range(200)))
        doc = document_from_text("notes.docx", text)
        self.assertEqual(doc.pagination, BLOCK)
        self.assertGreater(len(doc.pages), 1)
        self.assertEqual(doc.pages[0].kind, BLOCK)
        self.assertTrue(doc.pages[0].label.endswith("b.1"))
        self.assertIn("[notes.docx — block 1]", doc.page_labelled_text())

    def test_short_text_stays_a_single_page(self) -> None:
        doc = document_from_text("short.txt", "One clinical sentence about knee pain.")
        self.assertEqual(doc.pagination, PAGE)
        self.assertEqual(len(doc.pages), 1)
        self.assertEqual(doc.pages[0].label, "short.txt p.1")


class TestChunkCitations(unittest.TestCase):
    def _doc(self, filename: str, pages: int) -> ExtractedDocument:
        return ExtractedDocument(
            filename=filename,
            pages=[
                DocumentPage(filename, number, f"Page {number} body. " + ("pain noted " * 40))
                for number in range(1, pages + 1)
            ],
            total_pages=pages,
        )

    def test_chunk_knows_which_pages_it_covers(self) -> None:
        doc = self._doc("clinic.pdf", 4)
        chunks = chunk_page_labelled_text(doc.page_labelled_text(), max_chars=1_200)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(chunk.pages, f"{chunk.label} lost its page markers")
            self.assertTrue(chunk.source_hint.startswith("clinic.pdf p."))

    def test_a_span_is_compressed_into_a_range(self) -> None:
        doc = self._doc("labs.pdf", 6)
        chunks = chunk_page_labelled_text(doc.page_labelled_text(), max_chars=20_000)
        self.assertIn("labs.pdf p.1-p.6", chunks[0].source_hint)

    def test_a_chunk_spanning_two_files_names_both(self) -> None:
        first = self._doc("a.pdf", 2)
        second = self._doc("b.pdf", 2)
        text = first.page_labelled_text() + "\n\n" + second.page_labelled_text()
        chunks = chunk_page_labelled_text(text, max_chars=100_000)
        self.assertIn("a.pdf", chunks[0].source_hint)
        self.assertIn("b.pdf", chunks[0].source_hint)

    def test_text_without_markers_falls_back_to_the_chunk_label(self) -> None:
        chunks = chunk_page_labelled_text("plain text with no page markers at all")
        self.assertEqual(chunks[0].source_hint, "chunk 1/1")


class TestDuplicatePages(unittest.TestCase):
    def _page(self, number: int, header: str) -> DocumentPage:
        body = " ".join(
            f"sentence {i} about chronic knee pain and limited walking distance"
            for i in range(30)
        )
        return DocumentPage("bundle.pdf", number, f"{header}\n{body}")

    def test_a_reprinted_page_is_matched_by_content_not_bytes(self) -> None:
        """Same page, different running header: exact hashing alone would miss it."""
        doc = ExtractedDocument(
            filename="bundle.pdf",
            pages=[
                self._page(1, "VA record page 1"),
                DocumentPage("bundle.pdf", 2, "Entirely different note about tinnitus."),
                self._page(3, "VA record page 3"),
            ],
            total_pages=3,
            unreadable_pages=[9],
        )
        unique, duplicates = _dedupe_pages([doc])
        self.assertEqual([p.page for p in unique[0].pages], [1, 2])
        self.assertEqual(
            duplicates,
            [{"document": "bundle.pdf", "page": 3, "duplicate_of": "bundle.pdf p.1"}],
        )
        # Coverage metadata survives deduplication, or the report loses it.
        self.assertEqual(unique[0].total_pages, 3)
        self.assertEqual(unique[0].unreadable_pages, [9])

    def test_distinct_pages_are_kept(self) -> None:
        doc = ExtractedDocument(
            filename="bundle.pdf",
            pages=[
                DocumentPage("bundle.pdf", 1, "Knee pain after lifting a pallet."),
                DocumentPage("bundle.pdf", 2, "Tinnitus diagnosed in service."),
            ],
            total_pages=2,
        )
        unique, duplicates = _dedupe_pages([doc])
        self.assertEqual(len(unique[0].pages), 2)
        self.assertEqual(duplicates, [])

    def test_shingle_similarity_ignores_length_differences(self) -> None:
        phrase = "knee pain noted on exam "
        short = _page_shingles(phrase * 8)
        long = _page_shingles(phrase * 500)
        self.assertGreater(_shingle_similarity(short, long), 0.9)
        self.assertEqual(_shingle_similarity(short, frozenset()), 0.0)
        self.assertEqual(_shingle_similarity(frozenset(), long), 0.0)


class TestFactCitations(unittest.TestCase):
    def test_citation_parsing_handles_the_new_shapes(self) -> None:
        self.assertEqual(_parse_citation("clinic.pdf p.7"), ("clinic.pdf", 7))
        self.assertEqual(_parse_citation("clinic.pdf p.3-p.9"), ("clinic.pdf", 3))
        self.assertEqual(_parse_citation("notes.docx b.2"), ("notes.docx", 2))
        self.assertEqual(_parse_citation("[clinic.pdf — page 4]"), ("clinic.pdf", 4))
        # Unresolvable free text yields no page rather than a guessed one.
        self.assertEqual(_parse_citation("chunk 2/9"), ("", 0))
        self.assertEqual(_parse_citation("a.pdf p.3; b.pdf p.1"), ("", 0))

    def test_the_models_own_page_wins_over_the_chunk_span(self) -> None:
        fact = _fact_from_raw(
            {"description": "Knee pain.", "source": "clinic.pdf p.7"},
            "clinic.pdf p.3-p.9",
            document="clinic.pdf",
            page=3,
            section="Problem list",
        )
        assert fact is not None
        self.assertEqual((fact.document, fact.page), ("clinic.pdf", 7))

    def test_an_unresolved_source_falls_back_to_the_chunks_pages(self) -> None:
        fact = _fact_from_raw(
            {"description": "Knee pain.", "source": "chunk 2/9"},
            "clinic.pdf p.3-p.9",
            document="clinic.pdf",
            page=3,
            section="Problem list",
        )
        assert fact is not None
        self.assertEqual((fact.document, fact.page), ("clinic.pdf", 3))
        self.assertEqual(fact.section, "Problem list")

    def test_a_fact_without_a_description_is_dropped(self) -> None:
        self.assertIsNone(_fact_from_raw({"source": "a.pdf p.1"}, "a.pdf p.1"))


class TestCitationSelfCheck(unittest.TestCase):
    def _doc(self) -> ExtractedDocument:
        return ExtractedDocument(
            filename="clinic.pdf",
            pages=[
                DocumentPage(
                    "clinic.pdf",
                    1,
                    "Patient reports knee pain when walking upstairs since 2014.",
                ),
                DocumentPage("clinic.pdf", 2, "Tinnitus noted; audiology referral placed."),
            ],
            total_pages=2,
        )

    def _fact(self, page: int, quote: str) -> MedicalFact:
        return MedicalFact(
            "2020-01", "symptom", "Knee pain.", "clinic.pdf p.1", quote, document="clinic.pdf",
            page=page,
        )

    def test_a_quote_found_on_its_page_is_verified(self) -> None:
        report = verify_citations(
            [self._fact(1, "knee pain when walking upstairs")], [self._doc()]
        )
        self.assertEqual((report["checked"], report["missing"]), (1, 0))
        self.assertEqual(report["verified_ratio"], 1.0)

    def test_a_quote_not_on_its_page_is_flagged_with_an_example(self) -> None:
        report = verify_citations([self._fact(2, "knee pain when walking upstairs")], [self._doc()])
        self.assertEqual(report["checked"], 1)
        self.assertEqual(report["missing"], 1)
        self.assertEqual(report["examples"][0]["page"], 2)

    def test_uncheckable_facts_are_counted_as_skipped(self) -> None:
        facts = [
            self._fact(1, "knee"),  # too short to be meaningful
            MedicalFact("2020-01", "other", "No page.", "chunk 1/2", "walking upstairs since 2014"),
            MedicalFact(
                "2020-01", "other", "No quote.", "clinic.pdf p.1", "", document="clinic.pdf", page=1
            ),
        ]
        report = verify_citations(facts, [self._doc()])
        self.assertEqual(report["checked"], 0)
        self.assertEqual(report["skipped"], 3)


class TestCoverageReporting(unittest.TestCase):
    def test_coverage_ratio_counts_unreadable_pages(self) -> None:
        digest = MedicalDigest(pages_in_files=100, unreadable_pages=25)
        self.assertAlmostEqual(digest.coverage_ratio, 0.75)

    def test_coverage_ratio_is_total_when_page_counts_are_unknown(self) -> None:
        self.assertEqual(MedicalDigest().coverage_ratio, 1.0)

    def test_keyword_overlap_separates_wording_from_silence(self) -> None:
        digest = MedicalDigest(
            facts=[MedicalFact("2020-01", "symptom", "Left knee pain on stairs.", "a.pdf p.1")]
        )
        self.assertGreater(digest.keyword_overlap("knee pain on stairs"), 0.0)
        self.assertEqual(digest.keyword_overlap("torn rotator cuff"), 0.0)

    def test_element_coverage_maps_facts_onto_statement_elements(self) -> None:
        digest = MedicalDigest(
            facts=[
                MedicalFact("2020-01", "diagnosis", "Lumbar strain.", "a.pdf p.1"),
                MedicalFact("2020-01", "in_service_event", "Injury on active duty.", "a.pdf p.2"),
                MedicalFact("2020-01", "other", "Unclear.", "a.pdf p.3"),
            ]
        )
        coverage = digest.element_coverage()
        self.assertEqual(coverage["current_diagnosis"], 1)
        self.assertEqual(coverage["in_service_event"], 1)
        self.assertEqual(coverage["buddy_observable"], 0)
        self.assertEqual(set(coverage), set(digest.element_coverage()))


class TestStatementElements(unittest.TestCase):
    def test_a_medical_opinion_is_nexus_evidence_whatever_its_type(self) -> None:
        fact = MedicalFact(
            "2019-01",
            "diagnosis",
            "Lumbar strain.",
            "a.pdf p.1",
            "at least as likely as not caused by service",
        )
        self.assertEqual(statement_element_for(fact), "nexus")

    def test_types_map_to_their_element(self) -> None:
        cases = {
            "diagnosis": "current_diagnosis",
            "symptom": "severity_frequency",
            "medication": "treatment_history",
            "functional_limitation": "functional_impact",
            "observable_behavior": "buddy_observable",
            "in_service_event": "in_service_event",
        }
        for fact_type, element in cases.items():
            with self.subTest(fact_type=fact_type):
                self.assertEqual(
                    statement_element_for(
                        MedicalFact("2020-01", fact_type, "Description.", "a.pdf p.1")
                    ),
                    element,
                )

    def test_an_unknown_type_is_other(self) -> None:
        self.assertEqual(
            statement_element_for(MedicalFact("2020-01", "", "Description.", "a.pdf p.1")),
            "other",
        )


class TestVAGovSections(unittest.TestCase):
    def _doc(self, pages: list[str]) -> ExtractedDocument:
        return ExtractedDocument(
            filename="va_records.pdf",
            pages=[
                DocumentPage("va_records.pdf", index, text)
                for index, text in enumerate(pages, start=1)
            ],
            total_pages=len(pages),
        )

    def test_headings_are_mapped_and_carried_forward(self) -> None:
        doc = self._doc(
            [
                "Download your medical records\nPatient information\nName: Test",
                "Problem list\nChronic knee pain.",
                "Knee pain continues.",
                "Medications\nIbuprofen 400mg",
            ]
        )
        mapping = section_map(doc)
        self.assertEqual(mapping[1], "Patient information")
        self.assertEqual(mapping[2], "Problem list")
        self.assertEqual(mapping[3], "Problem list")  # carried forward
        self.assertEqual(mapping[4], "Medications")

    def test_a_page_mentioning_a_heading_mid_sentence_is_not_a_heading(self) -> None:
        doc = self._doc(["The provider discussed your medications and allergies today."])
        self.assertEqual(section_map(doc), {})

    def test_pages_before_any_heading_map_to_nothing(self) -> None:
        doc = self._doc(["Random administrative cover page.", "Vitals\nBP 120/80"])
        mapping = section_map(doc)
        self.assertNotIn(1, mapping)
        self.assertEqual(mapping[2], "Vitals")


class TestDateHygiene(unittest.TestCase):
    def test_page_dates_are_extracted_without_month_prefixes(self) -> None:
        found = _dates_in_text("Visit 2019-04-17. Labs 2020-05 and Jan 2021.")
        self.assertEqual(found, ["2019-04-17", "2020-05-01", "2021-01-01"])

    def test_text_without_dates_yields_nothing(self) -> None:
        self.assertEqual(_dates_in_text("No dates on this page at all."), [])


class _StubLLM:
    """Minimal LLMClient surface used by ``_llm_infer_undated``."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self._settings = type("_S", (), {"model_fast": "fast-model"})()

    def chat_json(self, system: str, user: str, **kwargs: object) -> object:
        self.calls.append({"user": user, **kwargs})
        if not self.responses:
            raise RuntimeError("no scripted response")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class TestUndatedInference(unittest.TestCase):
    def _facts(self, count: int) -> list[MedicalFact]:
        return [
            MedicalFact("unknown", "symptom", f"description {i}", "a.pdf p.1")
            for i in range(count)
        ]

    def test_batches_are_bounded(self) -> None:
        with patch("app.medical_review.config.UNDATED_FACT_BATCH_SIZE", 4):
            llm = _StubLLM(
                [
                    {"dates": [{"index": i, "date": "2019"} for i in range(4)]},
                    {"dates": [{"index": i, "date": "2020"} for i in range(4)]},
                ]
            )
            inferred = _llm_infer_undated(llm, self._facts(8))
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(inferred[0][0], "2019-01-01")
        self.assertEqual(inferred[7][0], "2020-01-01")

    def test_a_truncated_batch_does_not_lose_the_other_rows(self) -> None:
        with patch("app.medical_review.config.UNDATED_FACT_BATCH_SIZE", 4):
            llm = _StubLLM(
                [
                    {"dates": [{"index": 0, "date": "2019"}, {"index": "bad", "date": "2020"},
                               "not-a-row", {"index": 9, "date": "2021"}]},
                ]
            )
            inferred = _llm_infer_undated(llm, self._facts(4))
        self.assertEqual(inferred[0][0], "2019-01-01")
        self.assertNotIn(9, inferred)

    def test_a_failed_call_is_retried_once(self) -> None:
        with patch("app.medical_review.config.UNDATED_FACT_BATCH_SIZE", 4):
            llm = _StubLLM(
                [
                    RuntimeError("network down"),
                    {"dates": [{"index": 0, "date": "2018"}]},
                ]
            )
            inferred = _llm_infer_undated(llm, self._facts(4))
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(inferred[0][0], "2018-01-01")

    def test_a_permanently_failing_batch_is_dropped_not_raised(self) -> None:
        with patch("app.medical_review.config.UNDATED_FACT_BATCH_SIZE", 2):
            llm = _StubLLM([RuntimeError("down"), RuntimeError("down"), {"dates": []}])
            inferred = _llm_infer_undated(llm, self._facts(4))
        self.assertEqual(inferred, {})

    def test_no_facts_means_no_call(self) -> None:
        llm = _StubLLM([])
        self.assertEqual(_llm_infer_undated(llm, []), {})
        self.assertEqual(llm.calls, [])


if __name__ == "__main__":
    unittest.main()
