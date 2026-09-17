"""Offline unit tests for TF-IDF medical-record search (F2.S1).

Run from project root: .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.documents import (  # noqa: E402
    DocumentPage,
    ExtractedDocument,
    Paragraph,
    build_inverted_index,
    export_citation_index,
    search_records,
)


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
