"""Unit tests for `app.exporter` (Fact Citation Exporter, F7.S1).

Covers: filter logic, CSV/markdown/PDF output shape, the summary row on
every format, CSV-formula-injection guarding, and best-effort source
parsing — all against mock `MedicalFact`/`MedicalDigest` instances (no LLM
or Streamlit dependency).

Run from project root: python -m unittest discover -s tests -v
"""
from __future__ import annotations

import csv
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.medical_review import MedicalDigest, MedicalFact  # noqa: E402
from app import exporter  # noqa: E402


def _fact(**overrides) -> MedicalFact:
    base = dict(
        date="2020-01-01",
        type="diagnosis",
        description="Diagnosed with lumbar strain",
        source="records.pdf — page 3",
        quote="patient presents with lumbar strain",
    )
    base.update(overrides)
    return MedicalFact(**base)


class TestParseSource(unittest.TestCase):
    def test_em_dash_page_format(self) -> None:
        self.assertEqual(exporter.parse_source("records.pdf — page 3"), ("records.pdf", "3"))

    def test_pages_plural_range(self) -> None:
        doc, pages = exporter.parse_source("records.pdf — pages 3-4")
        self.assertEqual(doc, "records.pdf")
        self.assertIn("3", pages)

    def test_p_dot_format(self) -> None:
        self.assertEqual(exporter.parse_source("records.pdf p.12"), ("records.pdf", "12"))

    def test_unparseable_source_falls_back_to_raw_string(self) -> None:
        self.assertEqual(exporter.parse_source("chunk 2/9"), ("chunk 2/9", ""))

    def test_empty_source(self) -> None:
        self.assertEqual(exporter.parse_source(""), ("", ""))


class TestFilterFacts(unittest.TestCase):
    def test_no_filters_returns_all_facts(self) -> None:
        facts = [_fact(source="a"), _fact(source="b")]
        self.assertEqual(exporter.filter_facts(facts), facts)

    def test_rubric_filter_keeps_only_cited_sources(self) -> None:
        facts = [_fact(source="a"), _fact(source="b")]
        result = exporter.filter_facts(
            facts, only_rubric_cited=True, rubric_cited_sources={"a"}
        )
        self.assertEqual([f.source for f in result], ["a"])

    def test_rubric_filter_without_set_yields_empty(self) -> None:
        facts = [_fact(source="a")]
        result = exporter.filter_facts(facts, only_rubric_cited=True)
        self.assertEqual(result, [])

    def test_positive_outcome_filter(self) -> None:
        facts = [_fact(source="a"), _fact(source="b")]
        result = exporter.filter_facts(
            facts, only_positive_outcomes=True, positive_outcome_sources={"b"}
        )
        self.assertEqual([f.source for f in result], ["b"])

    def test_both_filters_combine_as_intersection(self) -> None:
        facts = [_fact(source="a"), _fact(source="b"), _fact(source="c")]
        result = exporter.filter_facts(
            facts,
            only_rubric_cited=True,
            only_positive_outcomes=True,
            rubric_cited_sources={"a", "b"},
            positive_outcome_sources={"b", "c"},
        )
        self.assertEqual([f.source for f in result], ["b"])


class TestExportFactsCsv(unittest.TestCase):
    def test_header_matches_required_columns(self) -> None:
        digest = MedicalDigest(facts=[_fact()], conditions=["PTSD"])
        data = exporter.export_facts_csv(digest.facts, digest)
        reader = csv.reader(io.StringIO(data.decode("utf-8")))
        header = next(reader)
        self.assertEqual(
            tuple(header),
            ("Date", "Type", "Fact Description", "Source Quote", "Page(s)", "Document Name"),
        )

    def test_summary_row_present_with_totals_and_conditions(self) -> None:
        facts = [
            _fact(date="2019-05-01", source="a.pdf — page 1"),
            _fact(date="2021-02-01", source="b.pdf — page 2"),
        ]
        digest = MedicalDigest(facts=facts, conditions=["PTSD", "Lumbar strain"])
        data = exporter.export_facts_csv(facts, digest)
        rows = list(csv.reader(io.StringIO(data.decode("utf-8"))))
        summary_row = rows[1]
        self.assertEqual(summary_row[0], "Summary")
        self.assertIn("Total facts: 2", summary_row[2])
        self.assertIn("2019", summary_row[2])
        self.assertIn("2021", summary_row[2])
        self.assertIn("PTSD", summary_row[2])
        self.assertIn("Lumbar strain", summary_row[2])

    def test_fact_rows_include_parsed_page_and_document_name(self) -> None:
        digest = MedicalDigest(facts=[_fact(source="records.pdf — page 7")])
        data = exporter.export_facts_csv(digest.facts, digest)
        rows = list(csv.reader(io.StringIO(data.decode("utf-8"))))
        fact_row = rows[2]
        self.assertEqual(fact_row[4], "7")
        self.assertEqual(fact_row[5], "records.pdf")

    def test_csv_formula_injection_is_neutralized(self) -> None:
        digest = MedicalDigest(
            facts=[_fact(description="=cmd|'/c calc'!A1", quote="+1+1")]
        )
        data = exporter.export_facts_csv(digest.facts, digest).decode("utf-8")
        self.assertNotIn("\n=cmd", data)
        self.assertIn("'=cmd", data)
        self.assertIn("'+1+1", data)

    def test_no_facts_still_yields_header_and_summary(self) -> None:
        digest = MedicalDigest(facts=[], conditions=[])
        data = exporter.export_facts_csv([], digest)
        rows = list(csv.reader(io.StringIO(data.decode("utf-8"))))
        self.assertEqual(len(rows), 2)  # header + summary, no fact rows
        self.assertIn("Total facts: 0", rows[1][2])
        self.assertIn("Unknown", rows[1][2])


class TestExportFactsMarkdown(unittest.TestCase):
    def test_contains_summary_and_table(self) -> None:
        digest = MedicalDigest(facts=[_fact()], conditions=["PTSD"])
        text = exporter.export_facts_markdown(digest.facts, digest).decode("utf-8")
        self.assertIn("## Summary", text)
        self.assertIn("Total facts:", text)
        self.assertIn("| Date | Type | Fact Description | Source Quote | Page(s) | Document Name |", text)
        self.assertIn("records.pdf", text)


class TestExportFactsPdf(unittest.TestCase):
    def test_returns_valid_pdf_bytes(self) -> None:
        digest = MedicalDigest(facts=[_fact()], conditions=["PTSD"])
        data = exporter.export_facts_pdf(digest.facts, digest)
        self.assertTrue(data.startswith(b"%PDF"))

    def test_unicode_punctuation_does_not_raise(self) -> None:
        digest = MedicalDigest(
            facts=[_fact(description="Patient stated “I can't sleep” — chronic pain")]
        )
        data = exporter.export_facts_pdf(digest.facts, digest)
        self.assertTrue(data.startswith(b"%PDF"))


class TestExportFactsEntryPoint(unittest.TestCase):
    def test_unsupported_format_raises_value_error(self) -> None:
        digest = MedicalDigest(facts=[_fact()])
        with self.assertRaises(ValueError):
            exporter.export_facts(digest, "xlsx")

    def test_applies_filters_before_formatting(self) -> None:
        facts = [_fact(source="a"), _fact(source="b")]
        digest = MedicalDigest(facts=facts)
        data = exporter.export_facts(
            digest, "csv", only_rubric_cited=True, rubric_cited_sources={"a"}
        )
        rows = list(csv.reader(io.StringIO(data.decode("utf-8"))))
        self.assertEqual(len(rows), 3)  # header + summary + 1 fact row
        self.assertIn("Total facts: 1", rows[1][2])

    def test_generation_error_is_tracked_and_reraised(self) -> None:
        digest = MedicalDigest(facts=[_fact()])
        with patch.dict(
            exporter._EXPORTERS, {"csv": lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))}
        ), patch("app.exporter.track_feature_error") as mock_track:
            with self.assertRaises(RuntimeError):
                exporter.export_facts(digest, "csv")
            mock_track.assert_called_once()
            args = mock_track.call_args[0]
            self.assertEqual(args[0], exporter.FEATURE_ID)
            self.assertIsInstance(args[1], RuntimeError)


if __name__ == "__main__":
    unittest.main()
