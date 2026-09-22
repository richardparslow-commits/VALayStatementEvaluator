"""Offline unit tests for TF-IDF medical-record search (F2.S1).

Run from project root: .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app.documents import (  # noqa: E402
    BLOCK,
    DocumentPage,
    ExtractedDocument,
    Paragraph,
    _PARAGRAPH_CACHE,
    _PARAGRAPH_CACHE_MAX_CHARS,
    _PARAGRAPH_CACHE_MAX_ENTRIES,
    _cached_paragraph_chars,
    build_inverted_index,
    export_citation_index,
    paragraph_index,
    search_records,
)
from app.medical_review import retrieve_evidence  # noqa: E402


def _doc(filename: str, pages: list[str]) -> ExtractedDocument:
    return ExtractedDocument(
        filename=filename,
        pages=[DocumentPage(filename, i, text) for i, text in enumerate(pages, start=1)],
    )


class TestBuildInvertedIndex(unittest.TestCase):
    def test_indexes_term_frequency_per_paragraph(self) -> None:
        paragraphs = [
            Paragraph("a.pdf p.1", "asthma asthma inhaler"),
            Paragraph("a.pdf p.2", "unrelated back pain note"),
        ]
        index = build_inverted_index(paragraphs)
        self.assertEqual(index["asthma"], {0: 2})
        self.assertEqual(index["inhaler"], {0: 1})
        self.assertNotIn("asthma", index.get("pain", {}))

    def test_empty_paragraphs_yields_empty_index(self) -> None:
        self.assertEqual(build_inverted_index([]), {})


class TestParagraphCacheIdentity(unittest.TestCase):
    def setUp(self) -> None:
        self.cache_patch = patch.dict(_PARAGRAPH_CACHE, {}, clear=True)
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)

    @staticmethod
    def _patient(name: str, condition: str) -> ExtractedDocument:
        return _doc(
            "records.txt",
            [f"Patient {name}: {condition}. Follow-up clinical observations and treatment plan."],
        )

    def test_equal_length_replacement_uses_new_text(self) -> None:
        original = self._patient("A", "asthma")
        replacement = self._patient("B", "injury")
        self.assertEqual(original.char_count, replacement.char_count)
        first = paragraph_index(original)
        second = paragraph_index(replacement)
        self.assertIn("Patient A", first[0].text)
        self.assertIn("Patient B", second[0].text)
        self.assertNotIn("Patient A", second[0].text)

    def test_search_keeps_duplicate_filenames_distinct(self) -> None:
        original = self._patient("A", "asthma")
        replacement = self._patient("B", "injury")
        results = search_records([original, replacement], "injury")
        self.assertEqual(len(results), 1)
        self.assertIn("Patient B", results[0].excerpt)
        self.assertEqual(search_records([replacement], "asthma"), [])

    def test_independent_sessions_retrieve_only_their_own_evidence(self) -> None:
        first_session = [self._patient("A", "asthma")]
        second_session = [self._patient("B", "injury")]
        first = retrieve_evidence(first_session, "asthma")
        second = retrieve_evidence(second_session, "injury")
        self.assertIn("Patient A", first.text)
        self.assertIn("Patient B", second.text)
        self.assertNotIn("Patient A", second.text)
        self.assertGreater(second.best_overlap, 0)

    def test_mutated_document_is_reindexed(self) -> None:
        doc = self._patient("A", "asthma")
        first = paragraph_index(doc)
        doc.pages[0].text = self._patient("B", "injury").pages[0].text
        second = paragraph_index(doc)
        self.assertIn("Patient A", first[0].text)
        self.assertIn("Patient B", second[0].text)

    def test_minimum_length_is_part_of_cache_identity(self) -> None:
        doc = _doc("records.txt", ["Short note.\n\n" + "Long clinical observation. " * 3])
        self.assertEqual(len(paragraph_index(doc, min_chars=40)), 1)
        self.assertEqual(len(paragraph_index(doc, min_chars=1)), 2)
        self.assertEqual(paragraph_index(doc, min_chars=1000), [])
        self.assertEqual(len(paragraph_index(doc, min_chars=40)), 1)

    def test_citation_labels_are_part_of_cache_identity(self) -> None:
        doc = self._patient("A", "asthma")
        first = paragraph_index(doc)
        doc.pages[0].filename = "renamed.txt"
        doc.pages[0].page = 7
        doc.pages[0].kind = BLOCK
        second = paragraph_index(doc)
        self.assertEqual(first[0].label, "records.txt p.1")
        self.assertEqual(second[0].label, "renamed.txt b.7")

    def test_page_order_and_boundaries_are_part_of_cache_identity(self) -> None:
        doc = _doc("records.txt", ["a" * 50, "b" * 50])
        first = paragraph_index(doc)
        doc.pages.reverse()
        reversed_index = paragraph_index(doc)
        self.assertEqual([p.label for p in reversed_index], ["records.txt p.2", "records.txt p.1"])
        self.assertEqual(first[0].text, "a" * 50)
        self.assertEqual(reversed_index[0].text, "b" * 50)
        repartitioned = _doc("records.txt", ["a" * 49, "a" + "b" * 50])
        self.assertEqual(repartitioned.char_count, doc.char_count)
        self.assertEqual(paragraph_index(repartitioned)[1].text, "a" + "b" * 50)

    def test_split_limit_is_part_of_cache_identity(self) -> None:
        doc = _doc("records.txt", ["a" * 50 + "\n" + "b" * 50])
        with patch("app.documents.PARAGRAPH_MAX_CHARS", 200):
            self.assertEqual(len(paragraph_index(doc)), 1)
        with patch("app.documents.PARAGRAPH_MAX_CHARS", 60):
            self.assertEqual(len(paragraph_index(doc)), 2)

    def test_identical_contents_and_labels_still_reuse_cache(self) -> None:
        first = paragraph_index(self._patient("A", "asthma"))
        second = paragraph_index(self._patient("A", "asthma"))
        self.assertIs(first, second)
        self.assertEqual(len(_PARAGRAPH_CACHE), 1)

    def test_concurrent_sessions_with_matching_sizes_do_not_share_text(self) -> None:
        def retrieve(patient: int) -> str:
            return retrieve_evidence(
                [self._patient(f"{patient:03d}", "asthma")], "asthma"
            ).text

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(retrieve, range(100)))
        for patient, result in enumerate(results):
            self.assertIn(f"Patient {patient:03d}:", result)
        self.assertLessEqual(len(_PARAGRAPH_CACHE), _PARAGRAPH_CACHE_MAX_ENTRIES)

    def test_cache_is_bounded_by_paragraph_text_not_only_by_entries(self) -> None:
        """One 5,000-page document indexes to ~12 MB of paragraph strings, so an
        entry-count-only cap of 64 retains hundreds of MB in a long-lived server
        process. The cache must evict once either bound is exceeded.

        The byte bound is patched down here so the mechanism is what is tested
        (the production bound needs megabytes of paragraph text to reach).
        """
        with patch("app.documents._PARAGRAPH_CACHE_MAX_CHARS", 900):
            for patient in range(12):
                paragraph_index(self._patient(f"{patient:03d}", "asthma " * 40))
        self.assertGreater(_cached_paragraph_chars(), 0)
        self.assertLessEqual(_cached_paragraph_chars(), 900)
        self.assertLess(len(_PARAGRAPH_CACHE), 12)
        # The newest entry survives eviction: a repeat of it is still a hit.
        newest = paragraph_index(self._patient("011", "asthma " * 40))
        self.assertIs(newest, paragraph_index(self._patient("011", "asthma " * 40)))


class TestSearchRecords(unittest.TestCase):
    def test_ranks_more_relevant_paragraph_first(self) -> None:
        documents = [
            _doc(
                "clinic.pdf",
                [
                    "Patient reports chronic asthma symptoms worsening with cold air exposure. "
                    "Asthma flare noted during examination.",
                    "Routine dental cleaning performed, no complaints, follow up in six months.",
                ],
            )
        ]
        results = search_records(documents, "asthma symptoms")
        self.assertTrue(results)
        self.assertIn("asthma", results[0].excerpt.lower())
        self.assertEqual(results[0].filename, "clinic.pdf")
        self.assertEqual(results[0].page, 1)

    def test_highlights_query_terms_in_excerpt(self) -> None:
        documents = [_doc("f.pdf", ["The veteran reports chronic knee pain after the incident."])]
        results = search_records(documents, "knee pain")
        self.assertTrue(results)
        self.assertIn("**knee**", results[0].excerpt)
        self.assertIn("**pain**", results[0].excerpt)

    def test_empty_query_returns_no_results(self) -> None:
        documents = [_doc("f.pdf", ["Some clinical note text long enough to be a paragraph."])]
        self.assertEqual(search_records(documents, ""), [])
        self.assertEqual(search_records(documents, "   "), [])

    def test_no_matching_documents_returns_empty(self) -> None:
        documents = [_doc("f.pdf", ["Completely unrelated clinical note about dermatology visit."])]
        self.assertEqual(search_records(documents, "asthma inhaler"), [])

    def test_provider_filter_excludes_non_matching_paragraphs(self) -> None:
        documents = [
            _doc(
                "f.pdf",
                [
                    "Seen by Dr. Smith for chronic back pain evaluation and treatment plan.",
                    "Seen by Dr. Jones for chronic back pain follow-up visit today.",
                ],
            )
        ]
        results = search_records(documents, "back pain", provider="Dr. Smith")
        self.assertEqual(len(results), 1)
        self.assertIn("Smith", results[0].excerpt)

    def test_date_range_filter_excludes_out_of_range_paragraphs(self) -> None:
        documents = [
            _doc(
                "f.pdf",
                [
                    "Visit on 2023-01-15 for chronic migraine treatment and medication review.",
                    "Visit on 2024-06-01 for chronic migraine follow-up and medication review.",
                ],
            )
        ]
        results = search_records(
            documents,
            "migraine",
            date_from=date(2024, 1, 1),
            date_to=date(2024, 12, 31),
        )
        self.assertEqual(len(results), 1)
        self.assertIn("2024-06-01", results[0].excerpt)

    def test_date_filter_excludes_paragraphs_without_any_date(self) -> None:
        documents = [_doc("f.pdf", ["Chronic migraine follow-up, medication review, no date noted here."])]
        results = search_records(
            documents, "migraine", date_from=date(2024, 1, 1), date_to=date(2024, 12, 31)
        )
        self.assertEqual(results, [])

    def test_limit_caps_result_count(self) -> None:
        pages = [
            f"Encounter number {i} regarding chronic asthma symptoms and inhaler use plan."
            for i in range(30)
        ]
        documents = [_doc("f.pdf", pages)]
        results = search_records(documents, "asthma", limit=5)
        self.assertEqual(len(results), 5)


class TestExportCitationIndex(unittest.TestCase):
    _CITATIONS = [
        {"excerpt": "chronic knee pain noted", "source": "clinic.pdf p.2"},
        {"excerpt": "asthma flare-up, prescribed inhaler", "source": "hospital.pdf p.5"},
    ]

    def test_json_export_round_trips(self) -> None:
        import json

        data = export_citation_index(self._CITATIONS, "json")
        self.assertEqual(json.loads(data.decode("utf-8")), self._CITATIONS)

    def test_csv_export_has_excerpt_and_source_columns(self) -> None:
        import csv
        import io

        data = export_citation_index(self._CITATIONS, "csv")
        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
        self.assertEqual(rows[0]["excerpt"], "chronic knee pain noted")
        self.assertEqual(rows[0]["source"], "clinic.pdf p.2")
        self.assertEqual(rows[1]["source"], "hospital.pdf p.5")

    def test_format_is_case_insensitive(self) -> None:
        self.assertEqual(
            export_citation_index(self._CITATIONS, "CSV"),
            export_citation_index(self._CITATIONS, "csv"),
        )

    def test_unsupported_format_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            export_citation_index(self._CITATIONS, "xml")

    def test_empty_citation_index_exports_empty_list_or_header_only(self) -> None:
        import json

        self.assertEqual(json.loads(export_citation_index([], "json").decode("utf-8")), [])
        csv_bytes = export_citation_index([], "csv").decode("utf-8")
        self.assertIn("excerpt", csv_bytes)
        self.assertIn("source", csv_bytes)

    def test_csv_export_neutralizes_formula_injection(self) -> None:
        """A malicious record excerpt/source starting with =, +, -, or @ must
        not be written as a live spreadsheet formula (CSV injection guard)."""
        import csv
        import io

        dangerous = [
            {"excerpt": "=cmd|' /C calc'!A1", "source": "@SUM(1+1)*cmd|'/C calc'"},
            {"excerpt": "+1+1", "source": "-2+3"},
        ]
        data = export_citation_index(dangerous, "csv")
        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
        for row in rows:
            self.assertFalse(row["excerpt"].startswith(("=", "+", "-", "@")))
            self.assertFalse(row["source"].startswith(("=", "+", "-", "@")))
            # original content is preserved (minus the leading quote guard)
        self.assertTrue(rows[0]["excerpt"].startswith("'="))
        self.assertTrue(rows[0]["source"].startswith("'@"))
        self.assertTrue(rows[1]["excerpt"].startswith("'+"))
        self.assertTrue(rows[1]["source"].startswith("'-"))


if __name__ == "__main__":
    unittest.main()
