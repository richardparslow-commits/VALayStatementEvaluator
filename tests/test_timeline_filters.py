"""Regression test for the timeline's "Clear all filters" control.

Streamlit refuses to let a widget's session-state key change after that widget
has been created in the same run. The clear control therefore has to reset the
keys in an ``on_click`` callback, which runs before the next render. This test
locks that in by making any in-render write to a ``timeline_*`` key raise, as
Streamlit does.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app.medical_review import MedicalDigest, MedicalFact  # noqa: E402
from app.views import timeline  # noqa: E402


class _WidgetSession(dict):
    """Session state that rejects writes to widget keys once locked."""

    locked = False

    def __getattr__(self, name: str):  # noqa: ANN202 - dict passthrough
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value) -> None:  # noqa: ANN001
        self[name] = value

    def __setitem__(self, key, value) -> None:  # noqa: ANN001
        if self.locked and isinstance(key, str) and key.startswith("timeline_"):
            raise RuntimeError(
                f"StreamlitAPIException: st.session_state.{key} cannot be "
                "modified after the widget with that key is instantiated."
            )
        super().__setitem__(key, value)


class TestClearTimelineFilters(unittest.TestCase):
    def _digest(self) -> MedicalDigest:
        return MedicalDigest(
            facts=[
                MedicalFact(
                    "2020-01-01",
                    "diagnosis",
                    "Diagnosed with a knee condition.",
                    "clinic.pdf p.1",
                )
            ],
            pages_reviewed=1,
            chunks_reviewed=1,
        )

    def _streamlit(self, session: _WidgetSession) -> MagicMock:
        st_mock = MagicMock()
        st_mock.session_state = session
        st_mock.columns.side_effect = lambda spec, **kwargs: [
            MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
        ]
        st_mock.date_input.return_value = None
        st_mock.checkbox.return_value = False
        st_mock.multiselect.return_value = ["diagnosis"]
        st_mock.radio.return_value = "Chronological (oldest first)"
        # The branch the old code took on every click of Clear all filters.
        st_mock.button.side_effect = lambda label, **kwargs: label == "Clear all filters"
        return st_mock

    def test_clear_runs_in_a_callback_not_in_the_rendered_branch(self) -> None:
        session = _WidgetSession()
        st_mock = self._streamlit(session)

        with patch.object(timeline, "st", st_mock):
            session.locked = True
            timeline.render_timeline_tab(self._digest())

            clear = next(
                call
                for call in st_mock.button.call_args_list
                if call.args and call.args[0] == "Clear all filters"
            )
            self.assertIs(clear.kwargs.get("on_click"), timeline._clear_timeline_filters)

            session.locked = False
            session["timeline_date_from"] = "2021-01-01"
            session["timeline_date_to"] = "2021-12-31"
            session["timeline_show_gaps"] = True
            session["timeline_fact_types"] = ["symptom"]
            session["timeline_providers"] = ["Elsewhere"]
            session["timeline_sort_order"] = "Chronological (newest first)"

            clear.kwargs["on_click"](*clear.kwargs["args"])

        self.assertIsNone(session["timeline_date_from"])
        self.assertIsNone(session["timeline_date_to"])
        self.assertIs(session["timeline_show_gaps"], False)
        self.assertEqual(session["timeline_fact_types"], list(clear.kwargs["args"][0]))
        self.assertEqual(session["timeline_providers"], list(clear.kwargs["args"][1]))
        self.assertEqual(
            session["timeline_sort_order"], "Chronological (oldest first)"
        )


if __name__ == "__main__":
    unittest.main()
