"""Synthetic R13 rendering and exact-review regressions; no provider calls."""
from __future__ import annotations

import subprocess
import sys
import unittest
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import MagicMock, patch

from tests import hermetic  # noqa: F401
from app import pilot
from app.accessibility import READING_GUIDE, render_reading_guide, status_html, table_html


class ParsedStatus(HTMLParser):
    def __init__(self, value):
        super().__init__(convert_charrefs=True)
        self.tags, self.text = [], []
        self.feed(value)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)


class PilotAccessibilityTests(unittest.TestCase):
    def test_hostile_messages_remain_exact_text_inside_fixed_status_markup(self):
        payloads = (
            '</div><img src="https://outside.invalid/CANARY" onerror="alert(1)">',
            '<svg onload="alert(1)"><script>CANARY</script></svg>',
            '<a href="javascript:alert(1)">fake approval</a><form action="https://outside.invalid">',
            '<style>body{display:none}</style>&lt;img src=x&gt;\n"\'&',
            '![CANARY](https://outside.invalid/CANARY) **not approved**',
        )
        for method in ("error", "warning", "info", "success"):
            for value in payloads:
                with self.subTest(method=method, value=value):
                    parsed = ParsedStatus(status_html(method, value))
                    self.assertEqual([tag for tag, _ in parsed.tags], ["div", "strong"])
                    self.assertTrue("".join(parsed.text).endswith(value))
                    attrs = parsed.tags[0][1]
                    self.assertEqual(attrs["role"], "alert" if method in ("error", "warning") else "status")
                    self.assertEqual(attrs["aria-live"], "assertive" if method in ("error", "warning") else "polite")
                    self.assertEqual(attrs["aria-atomic"], "true")
                    self.assertFalse(any(key.startswith("on") or key in ("src", "href") for key in attrs))

    def test_pilot_status_ignores_unsafe_caller_options_but_synthetic_delegates(self):
        container = MagicMock()
        with patch.object(pilot, "enabled", return_value=True):
            pilot.display("<img src=x>", container=container, method="warning",
                          unsafe_allow_html=True, unsafe_allow_javascript=True)
        container.html.assert_called_once_with(status_html("warning", "<img src=x>"))
        container.warning.assert_not_called()
        with patch.object(pilot, "enabled", return_value=False):
            pilot.display("synthetic", container=container, method="warning", icon="⚠️")
        container.warning.assert_called_once_with("synthetic", icon="⚠️")

    def test_help_is_pilot_only_and_has_no_review_or_admission_authority(self):
        container = MagicMock()
        with patch.object(pilot, "enabled", return_value=False):
            render_reading_guide(container)
        container.expander.assert_not_called()
        with patch.object(pilot, "enabled", return_value=True):
            render_reading_guide(container, expanded=True)
        container.expander.assert_called_once_with("How to read these results", expanded=True)
        self.assertEqual(container.subheader.call_count, len(READING_GUIDE))
        self.assertEqual(container.text.call_count, len(READING_GUIDE))
        container.checkbox.assert_not_called()

    def test_table_headers_and_cells_are_literal_native_elements_without_downloads(self):
        rows = [{'<img src="https://outside.invalid/HEADER">': '<svg onload="alert(1)">CELL</svg>',
                 "Status": "NOT FOUND"}, {"Status": "![image](https://outside.invalid/CELL)"}]
        parsed = ParsedStatus(table_html(rows))
        self.assertEqual(sum(tag == "th" for tag, _ in parsed.tags), 2)
        self.assertEqual(sum(tag == "td" for tag, _ in parsed.tags), 4)
        self.assertTrue(all(attrs["scope"] == "col" for tag, attrs in parsed.tags if tag == "th"))
        self.assertFalse({tag for tag, _ in parsed.tags} & {"img", "svg", "a", "script", "button"})
        self.assertIn(rows[0]["Status"], parsed.text)
        self.assertIn(rows[0]['<img src="https://outside.invalid/HEADER">'], parsed.text)
        container = MagicMock()
        with patch.object(pilot, "enabled", return_value=True):
            pilot.dataframe(container, rows, unsafe_allow_javascript=True)
        container.html.assert_called_once_with(table_html(rows))
        container.dataframe.assert_not_called()
        with patch.object(pilot, "enabled", return_value=False):
            pilot.dataframe(container, rows, hide_index=True)
        container.dataframe.assert_called_once_with(rows, hide_index=True)

    def test_all_application_dataframes_use_the_pilot_table_boundary(self):
        import ast
        root = Path(__file__).resolve().parent.parent / "app"
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "dataframe":
                    self.assertTrue(path.name == "pilot.py" or
                                    (isinstance(node.func.value, ast.Name) and node.func.value.id == "pilot"), path)

    def test_module_import_has_no_ui_client_or_settings_side_effects(self):
        code = ('import sys; import app.accessibility; '
                'assert not {"streamlit", "app.config", "app.llm", "app.telemetry", "openai"}.intersection(sys.modules)')
        run = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parent.parent,
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_full_passage_selection_is_literal_complete_and_does_not_grant_review(self):
        from streamlit.testing.v1 import AppTest
        code = '''
from unittest.mock import patch
import streamlit as st
from app import pilot
from app.accessibility import render_reading_guide
from app.views.factual_review import render_factual_review
from tests.test_factual_integrity import pilot_result as result
long_passage = "Invented observation " + "long context " * 500 + '<img src="https://outside.invalid/CANARY">'
r = result(long_passage)
with patch.object(pilot, "enabled", return_value=True), patch.object(pilot, "current_owner", return_value="fixture-owner"):
    render_factual_review(r, r.draft, slot="draft")
    render_reading_guide(st)
st.write("RECEIPT" if "factual_draft_receipt" in st.session_state else "UNREVIEWED")
'''
        at = AppTest.from_string(code, default_timeout=20).run()
        self.assertFalse(at.exception)
        original = "Invented observation " + "long context " * 500 + '<img src="https://outside.invalid/CANARY">'
        self.assertIn(original, [item.value for item in at.text])
        self.assertEqual(at.markdown[-1].value, "UNREVIEWED")
        supports = [item.value for item in at.multiselect]
        at.selectbox[0].select(at.selectbox[0].options[-1]).run()
        self.assertFalse(at.exception)
        self.assertIn("The patient denied right knee pain in June 2020.", [item.value for item in at.text])
        self.assertEqual([item.value for item in at.multiselect], supports)
        self.assertFalse(at.checkbox[0].value)
        at.checkbox[0].check().run()
        self.assertEqual(at.markdown[-1].value, "RECEIPT")
        at.selectbox[0].select(at.selectbox[0].options[0]).run()
        self.assertEqual(at.markdown[-1].value, "RECEIPT")
        self.assertIn(original, [item.value for item in at.text])


if __name__ == "__main__":
    unittest.main()
