"""Unit tests for UsageTracker's read/write consistency contract.

``UsageTracker`` promises every reader "a consistent point-in-time view" (see
its class docstring): workers ``record()`` concurrently while the UI and the
job serialiser read totals. These tests pin the two structural guarantees:

* ``summary()`` aggregates from a SINGLE snapshot, so one report can never mix
  point-in-time views (a ``fallback_calls``/``endpoints`` that disagrees with
  ``calls``);
* the deserialisation path writes through the locked bulk ``add_entries()``,
  so ``record()`` is not the only guarded writer by convention alone.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app.config import FALLBACK_ENDPOINT, PRIMARY_ENDPOINT  # noqa: E402
from app.job_payload import usage_from_json, usage_to_json  # noqa: E402
from app.usage import UsageEntry, UsageTracker  # noqa: E402


def _record(tracker: UsageTracker, endpoint: str = PRIMARY_ENDPOINT) -> None:
    tracker.record(
        model="m", phase="grounding", system="s", user="u", content="c",
        prompt_tokens=10, completion_tokens=5, endpoint=endpoint,
    )


class TestSummarySingleSnapshot(unittest.TestCase):
    def test_summary_takes_exactly_one_snapshot(self) -> None:
        tracker = UsageTracker()
        _record(tracker)
        real = UsageTracker._snapshot
        with patch.object(
            UsageTracker, "_snapshot", autospec=True, side_effect=real
        ) as snap:
            report = tracker.summary()
        self.assertEqual(snap.call_count, 1, "one report == one point-in-time view")
        self.assertEqual(report["calls"], 1)
        self.assertEqual(report["prompt_tokens"], 10)
        self.assertEqual(report["completion_tokens"], 5)
        self.assertEqual(report["total_tokens"], 15)
        self.assertEqual(report["endpoints"], [PRIMARY_ENDPOINT])
        self.assertEqual(report["fallback_calls"], 0)

    def test_a_worker_recording_mid_report_cannot_split_it(self) -> None:
        """A record() landing after the report's snapshot must not leak into it.

        Regression: summary() used to take three separate snapshots (totals(),
        endpoints_used(), and an inline one for fallback_calls); a concurrent
        writer between them produced reports like calls=1 while endpoints
        named two — internally impossible for any single point in time.
        """
        tracker = UsageTracker()
        _record(tracker)
        real = UsageTracker._snapshot
        state = {"raced": False}

        def racing_snapshot(self):
            entries = real(self)
            if not state["raced"]:
                state["raced"] = True
                _record(tracker, endpoint=FALLBACK_ENDPOINT)
            return entries

        with patch.object(
            UsageTracker, "_snapshot", autospec=True, side_effect=racing_snapshot
        ):
            report = tracker.summary()
        self.assertTrue(state["raced"], "the racing writer never fired")
        # The fallback call landed after the single snapshot: no field sees it.
        self.assertEqual(report["calls"], 1)
        self.assertEqual(report["fallback_calls"], 0)
        self.assertEqual(report["endpoints"], [PRIMARY_ENDPOINT])
        # ...and the tracker itself does contain it for the *next* report.
        self.assertEqual(tracker.summary()["fallback_calls"], 1)
        self.assertEqual(tracker.summary()["calls"], 2)


class TestLockedBulkLoad(unittest.TestCase):
    def test_usage_from_json_writes_through_the_locked_bulk_path(self) -> None:
        tracker = UsageTracker()
        _record(tracker)
        _record(tracker, endpoint=FALLBACK_ENDPOINT)
        payload = usage_to_json(tracker)
        seen: dict[str, int] = {}
        real_add = UsageTracker.add_entries

        def spy(self, entries):
            seen["count"] = len(entries)
            return real_add(self, entries)

        with patch.object(UsageTracker, "add_entries", spy):
            restored = usage_from_json(payload)
        self.assertEqual(
            seen.get("count"), 2, "usage_from_json must bulk-load via add_entries"
        )
        self.assertEqual(len(restored.entries_snapshot()), 2)
        self.assertTrue(restored.used_fallback)

    def test_add_entries_extends_under_the_lock(self) -> None:
        tracker = UsageTracker()
        entries = [UsageEntry(model="m", phase="p", prompt_tokens=1, completion_tokens=2)]
        with patch.object(tracker, "_lock") as lock_mock:
            tracker.add_entries(entries)
        lock_mock.__enter__.assert_called_once()
        self.assertEqual(len(tracker.entries), 1)
        self.assertEqual(tracker.totals().total_tokens, 3)
