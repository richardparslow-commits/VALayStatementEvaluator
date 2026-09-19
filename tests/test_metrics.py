"""Tests for Prometheus metrics (app/metrics.py) and the /metrics route.

Two things are load-bearing here:

* the renderer is **pure** — a scrape must never perform I/O, because /metrics
  shares a port with the liveness probe and a scrape every 15s cannot be allowed
  to depend on a slow Redis tier;
* a value that could not be read is **omitted rather than zeroed**, so a dashboard
  cannot show a healthy disk or an empty backlog at the moment the truth is
  unknown. That distinction is the difference between a graph that is wrong and a
  graph that is missing, and only one of those gets noticed.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app import (  # noqa: E402
    circuit_breaker,
    config,
    health,
    job_queue,
    logging_config,
    metrics,
)
from app.llm import LLMClient  # noqa: E402


class _FakeSettings:
    """Minimal settings stand-in so LLMClient can be built without a network."""

    configured = True
    api_key = "test-key"
    base_url = "http://example.invalid"
    model_main = "test-model"
    model_fast = "test-fast"


def _histogram_buckets(text: str, metric: str, **labels: str) -> dict[str, str]:
    """``{le: value}`` for the one label set matching ``labels``."""
    out: dict[str, str] = {}
    prefix = f"{metric}_bucket{{"
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        head, _, value = line.rpartition(" ")
        inner = head[len(prefix) : -1]
        parsed = {
            key: raw[1:-1]
            for key, _, raw in (part.partition("=") for part in inner.split(","))
        }
        if all(parsed.get(key) == expected for key, expected in labels.items()):
            out[parsed["le"]] = value
    return out


def _by_key(samples) -> dict:  # noqa: ANN001 - test helper
    """Rebuild ``samples()`` as ``{(label, ...): value}`` so labels are hashable."""
    return {tuple(sorted(labels.items())): value for labels, value in samples}


def _render(**sections) -> str:  # noqa: ANN003 - test payload builder
    """Render a metric body from just the sections under test."""
    base: dict = {"uptime_s": 42}
    base.update(sections)
    return metrics.render_prometheus(base)


def _samples(text: str) -> dict[str, str]:
    """Flat ``{metric name or name{labels}: value}`` view of an exposition body."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        out[name] = value
    return out


class TestExpositionFormat(unittest.TestCase):
    def test_types_and_help_are_declared_once_per_family(self) -> None:
        text = _render(
            audit_backup={"status": "ok", "configured": True, "destination": "s3"},
            job_queue={"backend": "redis", "enabled": True, "is_distributed": True, "depth": 2},
        )
        self.assertEqual(text.count("# TYPE va_lse_audit_backup_state "), 1)
        self.assertEqual(text.count("# HELP va_lse_audit_backup_state "), 1)
        for line in text.splitlines():
            if line.startswith("# TYPE"):
                name = line.split()[2]
                self.assertEqual(
                    text.count(f"# TYPE {name} "), 1, f"{name} declared twice"
                )

    def test_every_line_is_a_comment_or_a_named_sample(self) -> None:
        text = _render(
            cache={"backend": "local_lru"},
            tracing={"enabled": True, "active": False},
            audit={"configured": True, "write_failures": 3},
        )
        for line in text.splitlines():
            self.assertTrue(line.startswith("#") or " " in line, line)
            if not line.startswith("#"):
                name, _, value = line.rpartition(" ")
                self.assertRegex(name, r"^[a-zA-Z_:][a-zA-Z0-9_:]*(\{.*\})?$")
                self.assertNotEqual(value, "")

    def test_body_ends_with_a_newline(self) -> None:
        self.assertTrue(_render(audit={"configured": True}).endswith("\n"))

    def test_up_is_always_present(self) -> None:
        self.assertEqual(_samples(_render())["va_lse_up"], "1")

    def test_metric_names_documented_match_what_is_emitted(self) -> None:
        """The exported name list is what DEPLOYMENT.md documents — keep it honest."""
        text = _render(
            cache={"backend": "local_lru"},
            tracing={"enabled": True, "active": True},
            audit={"configured": True, "write_failures": 0},
            audit_backup={
                "status": "ok",
                "configured": True,
                "off_pod": True,
                "last_success_utc": "2026-09-16T12:00:00+00:00",
                "age_seconds": 60,
                "pending_bytes": 10,
                "uploaded_objects": 1,
                "uploaded_bytes": 2,
                "runs": 3,
                "interval_hours": 6,
                "local_retention_days": 7,
                "cloud_retention_days": 90,
                "destination": "s3",
            },
            disk={
                "checked": True,
                "free_bytes": 1,
                "total_bytes": 2,
                "used_percent": 50.0,
                "below_floor": False,
                "min_free_bytes": 3,
            },
            job_queue={
                "backend": "redis",
                "enabled": True,
                "is_distributed": True,
                "depth": 0,
            },
            ready=True,
        )
        emitted = {line.split()[2] for line in text.splitlines() if line.startswith("# TYPE")}
        self.assertEqual(emitted, set(metrics.metric_names()))


class TestValueHandling(unittest.TestCase):
    def test_booleans_render_as_one_and_zero(self) -> None:
        text = _render(tracing={"enabled": True, "active": False})
        samples = _samples(text)
        self.assertEqual(samples["va_lse_tracing_enabled"], "1")
        self.assertEqual(samples["va_lse_tracing_active"], "0")

    def test_a_zero_depth_is_emitted_but_a_missing_one_is_not(self) -> None:
        """0 waiting jobs is information; an unread backlog is not the same thing."""
        with_zero = _samples(_render(job_queue={"backend": "redis", "depth": 0}))
        self.assertEqual(with_zero["va_lse_job_queue_depth"], "0")
        without = _samples(_render(job_queue={"backend": "redis", "depth": None}))
        self.assertNotIn("va_lse_job_queue_depth", without)

    def test_unknown_values_are_omitted_not_zeroed(self) -> None:
        text = _render(audit_backup={"status": "never_ran", "configured": True})
        samples = _samples(text)
        self.assertNotIn("va_lse_audit_backup_last_success_timestamp_seconds", samples)
        self.assertNotIn("va_lse_audit_backup_pending_bytes", samples)

    def test_unchecked_disk_emits_no_disk_series(self) -> None:
        text = _render(disk={"checked": False})
        self.assertNotIn("va_lse_disk_free_bytes", text)
        # ...but the family is not half-declared either.
        self.assertNotIn("va_lse_disk_below_floor", text)

    def test_iso_timestamps_become_unix_seconds(self) -> None:
        for raw in ("2026-09-16T12:00:00+00:00", "2026-09-16T12:00:00Z"):
            samples = _samples(
                _render(audit_backup={"status": "ok", "last_success_utc": raw})
            )
            self.assertEqual(
                float(samples["va_lse_audit_backup_last_success_timestamp_seconds"]),
                1789560000.0,
                raw,
            )

    def test_an_unparseable_timestamp_is_omitted(self) -> None:
        samples = _samples(_render(audit_backup={"status": "ok", "last_success_utc": "soon"}))
        self.assertNotIn("va_lse_audit_backup_last_success_timestamp_seconds", samples)

    def test_hours_are_converted_to_the_base_unit(self) -> None:
        samples = _samples(_render(audit_backup={"status": "ok", "interval_hours": 6}))
        self.assertEqual(samples["va_lse_audit_backup_interval_seconds"], "21600.0")


class TestBackupStateEnum(unittest.TestCase):
    def test_every_status_maps_to_its_documented_value(self) -> None:
        for status, expected in metrics.BACKUP_STATE_VALUES.items():
            samples = _samples(_render(audit_backup={"status": status}))
            self.assertEqual(
                samples["va_lse_audit_backup_state"], str(expected), status
            )

    def test_an_unknown_status_is_not_reported_as_healthy(self) -> None:
        """A status this version does not know must not masquerade as 0/ok."""
        samples = _samples(_render(audit_backup={"status": "who knows"}))
        self.assertEqual(samples["va_lse_audit_backup_state"], "5")  # unavailable

    def test_missing_status_is_not_reported_as_healthy(self) -> None:
        samples = _samples(_render(audit_backup={"configured": True}))
        self.assertEqual(samples["va_lse_audit_backup_state"], "5")


class TestLabelEscaping(unittest.TestCase):
    def test_quotes_backslashes_and_newlines_are_escaped(self) -> None:
        text = _render(job_queue={"backend": 'a"b\\c\nd'})
        line = next(l for l in text.splitlines() if l.startswith("va_lse_job_queue_info{"))
        # Input characters a " b \ c <newline> d become a \" b \\ c \n d.
        self.assertIn(r'a\"b\\c\nd', line)
        self.assertEqual(line.count("\n"), 0)

    def test_labels_are_sorted_so_output_is_deterministic(self) -> None:
        first = _render(audit_backup={"status": "ok", "destination": "s3"})
        second = _render(audit_backup={"status": "ok", "destination": "s3"})
        self.assertEqual(first, second)

    def test_help_text_newlines_do_not_break_the_parse(self) -> None:
        text = metrics.render_prometheus({"uptime_s": 1})
        self.assertTrue(all(
            line.startswith("#") or " " in line for line in text.splitlines()
        ))


class TestNoSideEffects(unittest.TestCase):
    def test_rendering_never_touches_collaborators(self) -> None:
        """Pure function: no config reads, no destination clients, no network."""
        with patch("app.audit_backup.build_destination") as builder:
            with patch("app.job_queue.get_job_backend") as backend:
                _render(
                    audit_backup={"status": "ok", "configured": True},
                    job_queue={"backend": "redis", "depth": 1},
                )
        builder.assert_not_called()
        backend.assert_not_called()


# ------------------------------------------------------------------ HTTP route


def _serve():
    """Start the health sidecar on an ephemeral port (caller must stop it)."""
    health.stop_health_server()
    server = health.start_health_server(port=0, host="127.0.0.1")
    assert server is not None, "health server failed to bind"
    time.sleep(0.15)
    return server.server_address[1]


def _get(url: str) -> tuple[int, str, str]:
    with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310
        return resp.status, resp.read().decode("utf-8"), resp.headers.get("Content-Type", "")


class _CountingBackend(job_queue.JobBackend):
    """A remote-looking backend that records how often its backlog is read.

    Subclasses the real :class:`JobBackend` on purpose: the behaviour under test is
    ``JobBackend.health``'s decision about whether reading the backlog is safe, so a
    hand-rolled double that only implements ``depth`` would silently fall back to
    "unavailable" instead of exercising it.
    """

    name = "redis"
    is_distributed = True
    depth_is_remote = True

    def __init__(self) -> None:
        self.depth_reads = 0

    def depth(self) -> int:
        self.depth_reads += 1
        return 4

    def ping(self) -> bool:
        return True


class TestMetricsRoute(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = _CountingBackend()
        self._patch = patch("app.job_queue.get_job_backend", return_value=self.backend)
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        health.stop_health_server()
        time.sleep(0.05)

    def test_metrics_returns_prometheus_content_type(self) -> None:
        port = _serve()
        status, body, content_type = _get(f"http://127.0.0.1:{port}/metrics")
        self.assertEqual(status, 200)
        self.assertIn("text/plain", content_type)
        self.assertIn("version=0.0.4", content_type)
        self.assertIn("va_lse_up 1", body)

    def test_metrics_does_not_probe_the_queue_by_default(self) -> None:
        """A scrape every 15s must not become a network round trip."""
        port = _serve()
        _get(f"http://127.0.0.1:{port}/metrics")
        self.assertEqual(self.backend.depth_reads, 0)

    def test_metrics_probe_flag_reads_the_queue_when_asked(self) -> None:
        port = _serve()
        _, body, _ = _get(f"http://127.0.0.1:{port}/metrics?probe=1")
        self.assertEqual(self.backend.depth_reads, 1)
        self.assertIn("va_lse_job_queue_depth 4", body)

    def test_health_does_not_probe_the_queue_by_default(self) -> None:
        port = _serve()
        _, body, _ = _get(f"http://127.0.0.1:{port}/health")
        self.assertEqual(self.backend.depth_reads, 0)
        queue = json.loads(body)["job_queue"]
        self.assertIsNone(queue["depth"])
        self.assertEqual(queue["depth_source"], "not_probed")
        self.assertIn("depth_note", queue)

    def test_health_probe_flag_reads_the_queue_when_asked(self) -> None:
        port = _serve()
        _, body, _ = _get(f"http://127.0.0.1:{port}/health?probe=1")
        self.assertEqual(self.backend.depth_reads, 1)
        self.assertEqual(json.loads(body)["job_queue"]["depth"], 4)

    def test_head_on_metrics_sends_no_body(self) -> None:
        import http.client

        port = _serve()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("HEAD", "/metrics")
        resp = conn.getresponse()
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.read(), b"")
        conn.close()

    def test_metrics_reflects_audit_backup_state(self) -> None:
        with patch(
            "app.audit_backup.audit_backup_health",
            return_value={"status": "stale", "configured": True, "destination": "s3"},
        ):
            port = _serve()
            _, body, _ = _get(f"http://127.0.0.1:{port}/metrics")
        samples = _samples(body)
        self.assertEqual(samples["va_lse_audit_backup_state"], "3")  # stale
        self.assertIn('va_lse_audit_backup_info{destination="s3",status="stale"}', body)


# --------------------------------------------------------- in-process registry


class _RegistryCase(unittest.TestCase):
    """Resets the module-level registry so tests cannot leak series into each other."""

    def setUp(self) -> None:
        metrics.reset_for_tests()

    def tearDown(self) -> None:
        metrics.reset_for_tests()

    @staticmethod
    def _render_registry() -> str:
        return metrics.render_prometheus({"uptime_s": 1})


class TestCounterSemantics(_RegistryCase):
    def test_counts_are_kept_per_label_combination(self) -> None:
        counter = metrics.Counter(name="t_total", help_text="t")
        counter.inc(kind="a")
        counter.inc(kind="a")
        counter.inc(kind="b")
        self.assertEqual(
            _by_key(counter.samples()), {(("kind", "a"),): 2.0, (("kind", "b"),): 1.0}
        )

    def test_a_zero_value_is_still_a_present_series(self) -> None:
        """An observed zero is information; an absent series is not."""
        counter = metrics.Counter(name="t_total", help_text="t")
        counter.inc()
        self.assertIn("t_total 1", self._render_counter(counter))

    @staticmethod
    def _render_counter(counter: metrics.Counter) -> str:
        writer = metrics._Writer()
        writer.counter(counter)
        return writer.text()

    def test_label_order_does_not_create_a_second_series(self) -> None:
        counter = metrics.Counter(name="t_total", help_text="t")
        counter.inc(a="1", b="2")
        counter.inc(b="2", a="1")
        self.assertEqual(len(counter.samples()), 1)
        self.assertEqual(counter.samples()[0][1], 2.0)

    def test_a_stringification_failure_is_swallowed(self) -> None:
        """Instrumentation is called from the hot path and must never raise."""

        class _Hostile:
            def __str__(self) -> str:
                raise RuntimeError("boom")

        counter = metrics.Counter(name="t_total", help_text="t")
        counter.inc(kind=_Hostile())  # must not raise
        self.assertEqual(counter.samples(), [])

    def test_the_cap_admits_existing_combinations_and_refuses_new_ones(self) -> None:
        """A runaway label must not grow memory without bound."""
        counter = metrics.Counter(name="t_total", help_text="t")
        for index in range(metrics.MAX_SERIES_PER_METRIC):
            counter.inc(kind=str(index))
        self.assertEqual(len(counter.samples()), metrics.MAX_SERIES_PER_METRIC)

        counter.inc(kind="one-too-many")
        self.assertEqual(len(counter.samples()), metrics.MAX_SERIES_PER_METRIC)

        # ...but a label already admitted keeps counting, so the cap cannot make a
        # busy series go silently flat.
        counter.inc(kind="0")
        self.assertEqual(_by_key(counter.samples())[(("kind", "0"),)], 2.0)

    def test_the_drop_is_reported_with_the_offending_family(self) -> None:
        counter = metrics.Counter(name="t_total", help_text="t")
        for index in range(metrics.MAX_SERIES_PER_METRIC):
            counter.inc(kind=str(index))
        counter.inc(kind="one-too-many")
        dropped = _by_key(metrics._series_dropped.samples())
        self.assertEqual(dropped[(("metric", "t_total"),)], 1.0)
        self.assertIn("va_lse_metrics_series_dropped_total", self._render_registry())

    def test_the_drop_counter_cannot_recurse_into_itself(self) -> None:
        """Fill the drop counter itself, then drop something: no recursion."""
        full_store = {(("metric", str(i)),): 1.0 for i in range(metrics.MAX_SERIES_PER_METRIC)}
        admitted = metrics._admit(metrics._DROPPED_NAME, (("metric", "new"),), full_store)
        self.assertFalse(admitted)


class TestHistogramMath(_RegistryCase):
    def _record(self, *values: float) -> metrics.Histogram:
        # Integer bounds, matching the real ladders: the `le` label is rendered
        # verbatim, so a float ladder would expose le="10.0" to PromQL.
        family = metrics.Histogram(
            name="t_ms", help_text="t", buckets=(10, 20, 30)
        )
        for value in values:
            family.observe(value, phase="p")
        writer = metrics._Writer()
        writer.histogram(family)
        self.text = writer.text()
        return family

    def test_buckets_are_cumulative_and_end_at_inf(self) -> None:
        self._record(5, 15, 25, 999)
        buckets = _histogram_buckets(self.text, "t_ms", phase="p")
        self.assertEqual(buckets["10"], "1.0")  # <= 10
        self.assertEqual(buckets["20"], "2.0")  # <= 20 (cumulative)
        self.assertEqual(buckets["30"], "3.0")
        self.assertEqual(buckets["+Inf"], "4.0")  # everything, including 999

    def test_the_bucket_label_is_an_integer_not_a_float(self) -> None:
        """PromQL matches `le="10"`; a rendered `10.0` would never match."""
        self._record(5)
        buckets = _histogram_buckets(self.text, "t_ms", phase="p")
        self.assertEqual(sorted(buckets, key=lambda le: (le == "+Inf", float(le))), ["10", "20", "30", "+Inf"])

    def test_count_and_sum_are_recorded(self) -> None:
        self._record(5, 15)
        samples = _samples(self.text)
        self.assertEqual(samples['t_ms_count{phase="p"}'], "2.0")
        self.assertEqual(samples['t_ms_sum{phase="p"}'], "20.0")
        self.assertEqual(_samples(self.text)['t_ms_bucket{phase="p",le="+Inf"}'], "2.0")

    def test_the_inf_bucket_is_never_below_the_explicit_ones(self) -> None:
        self._record(1, 1000)
        buckets = _histogram_buckets(self.text, "t_ms", phase="p")
        explicit = max(float(v) for le, v in buckets.items() if le != "+Inf")
        self.assertGreaterEqual(float(buckets["+Inf"]), explicit)

    def test_a_value_above_every_bucket_lands_only_in_inf(self) -> None:
        self._record(1000)
        buckets = _histogram_buckets(self.text, "t_ms", phase="p")
        self.assertEqual(buckets["10"], "0.0")
        self.assertEqual(buckets["+Inf"], "1.0")

    def test_negative_durations_are_clamped_rather_than_dropped(self) -> None:
        """A backwards clock must not produce a negative bucket or a lost sample."""
        self._record(-5)
        samples = _samples(self.text)
        self.assertEqual(samples['t_ms_count{phase="p"}'], "1.0")
        self.assertEqual(samples['t_ms_sum{phase="p"}'], "0.0")
        self.assertEqual(_histogram_buckets(self.text, "t_ms", phase="p")["10"], "1.0")

    def test_a_failed_observation_is_swallowed(self) -> None:
        class _Hostile:
            def __str__(self) -> str:
                raise RuntimeError("boom")

        family = metrics.Histogram(name="t_ms", help_text="t")
        family.observe(5, phase=_Hostile())  # must not raise
        self.assertEqual(family.observed, 0)

    def test_an_unobserved_histogram_is_absent(self) -> None:
        family = metrics.Histogram(name="t_ms", help_text="t")
        writer = metrics._Writer()
        writer.histogram(family)
        self.assertEqual(writer.text(), "")

    def test_zero_when_empty_emits_a_bindable_series(self) -> None:
        """An alert on rate(...) needs the family to exist before the first call."""
        family = metrics.Histogram(name="t_ms", help_text="t", buckets=(10,))
        writer = metrics._Writer()
        writer.histogram(family, zero_when_empty=True)
        text = writer.text()
        self.assertIn('t_ms_bucket{le="10"} 0', text)
        self.assertIn('t_ms_bucket{le="+Inf"} 0', text)
        self.assertIn("t_ms_count 0", text)
        self.assertIn("t_ms_sum 0", text)


class TestRegistryThreadSafety(_RegistryCase):
    def _hammer(self, work) -> None:  # noqa: ANN001 - test helper
        threads = [threading.Thread(target=work) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def test_concurrent_increments_are_not_lost(self) -> None:
        """The registry is written from Streamlit's session threads and workers."""
        counter = metrics.Counter(name="t_total", help_text="t")
        self._hammer(lambda: [counter.inc(kind="a") for _ in range(500)])
        self.assertEqual(_by_key(counter.samples())[(("kind", "a"),)], 8 * 500)

    def test_count_and_sum_stay_consistent_under_concurrency(self) -> None:
        family = metrics.Histogram(name="t_ms", help_text="t", buckets=(10,))
        self._hammer(lambda: [family.observe(2.5, phase="p") for _ in range(200)])
        self.assertEqual(family.observed, 8 * 200)
        writer = metrics._Writer()
        writer.histogram(family)
        samples = _samples(writer.text())
        self.assertEqual(samples['t_ms_count{phase="p"}'], "1600.0")
        self.assertEqual(samples['t_ms_sum{phase="p"}'], "4000.0")


class TestSessionTracker(_RegistryCase):
    def test_a_touched_session_is_counted(self) -> None:
        metrics.sessions.touch("sid-1")
        self.assertEqual(metrics.session_count(), 1)

    def test_an_untouched_tracker_counts_nothing(self) -> None:
        self.assertEqual(metrics.session_count(), 0)

    def test_an_idle_session_ages_out_of_the_window(self) -> None:
        metrics.sessions.touch("sid-1")
        now = time.monotonic() + metrics.sessions.ttl_seconds + 1
        self.assertEqual(metrics.session_count(now=now), 0)

    def test_a_recent_touch_keeps_a_session_in_the_window(self) -> None:
        metrics.sessions.touch("sid-1")
        metrics.sessions.touch("sid-1")
        later = time.monotonic() + metrics.sessions.ttl_seconds - 1
        self.assertEqual(metrics.session_count(now=later), 1)

    def test_counting_prunes_expired_sessions_rather_than_amassing_them(self) -> None:
        """Otherwise a long-lived process leaks one dict entry per browser session."""
        for index in range(20):
            metrics.sessions.touch(f"sid-{index}")
        now = time.monotonic() + metrics.sessions.ttl_seconds + 1
        self.assertEqual(metrics.session_count(now=now), 0)
        self.assertEqual(metrics.session_count(now=now), 0)

    def test_an_empty_id_is_ignored(self) -> None:
        metrics.sessions.touch("")
        self.assertEqual(metrics.session_count(), 0)

    def test_a_broken_id_does_not_raise(self) -> None:
        class _Hostile:
            def __bool__(self) -> bool:
                raise RuntimeError("boom")

        metrics.sessions.touch(_Hostile())  # type: ignore[arg-type]  # must not raise

    def test_capacity_evicts_the_stalest_session_not_the_newest(self) -> None:
        """Refusing the new session would under-report exactly when traffic spikes."""
        with patch("app.metrics.MAX_TRACKED_SESSIONS", 2):
            metrics.sessions.touch("oldest")
            metrics.sessions.touch("middle")
            metrics.sessions.touch("newest")
            tracked = set(metrics._sessions)  # type: ignore[attr-defined]
        self.assertEqual(metrics.session_count(), 2)
        self.assertNotIn("oldest", tracked)
        self.assertEqual(tracked, {"middle", "newest"})

    def test_the_ttl_comes_from_the_configured_setting(self) -> None:
        self.assertEqual(
            metrics.sessions.ttl_seconds, float(config.METRICS_SESSION_TTL_SECONDS)
        )
        self.assertGreater(metrics.sessions.ttl_seconds, 0)

    def test_the_count_is_exposed_as_a_gauge_with_its_window(self) -> None:
        metrics.sessions.touch("sid-1")
        text = self._render_registry()
        self.assertIn("va_lse_session_count 1", text)
        self.assertIn(
            f"va_lse_session_ttl_seconds {int(metrics.sessions.ttl_seconds)}", text
        )


class TestRecordingHooks(_RegistryCase):
    def test_a_logical_call_records_duration_and_count_together(self) -> None:
        """These two must never drift: the rate and the latency of the same event."""
        metrics.observe_llm_call("draft", "ok", 1234)
        metrics.observe_llm_call("draft", "error", 10)
        self.assertEqual(metrics.llm_call_duration_ms.observed, 2)
        samples = _by_key(metrics.llm_calls_total.samples())
        self.assertEqual(samples[(("outcome", "ok"), ("phase", "draft"))], 1.0)
        self.assertEqual(samples[(("outcome", "error"), ("phase", "draft"))], 1.0)
        text = self._render_registry()
        self.assertEqual(
            _samples(text)['va_lse_llm_calls_total{outcome="ok",phase="draft"}'], "1.0"
        )

    def test_attempts_and_errors_are_separable_from_calls(self) -> None:
        metrics.observe_llm_attempt("draft", "retry")
        metrics.observe_llm_attempt("draft", "error")
        metrics.observe_llm_error("retry")
        self.assertEqual(
            len(metrics.llm_attempts_total.samples()), 2, "retry and error are distinct"
        )
        self.assertEqual(
            _by_key(metrics.llm_errors_total.samples()), {(("category", "retry"),): 1.0}
        )

    def test_a_missing_phase_falls_back_to_a_named_bucket(self) -> None:
        """An empty label value would be a second, invisible series for one phase."""
        metrics.observe_llm_call("", "ok", 5)
        metrics.observe_phase("", "ok", 5)
        metrics.observe_llm_attempt("", "ok")
        self.assertEqual(
            _by_key(metrics.llm_calls_total.samples()),
            {(("outcome", "ok"), ("phase", "general")): 1.0},
        )
        metrics.observe_llm_error("")
        self.assertEqual(
            _by_key(metrics.llm_errors_total.samples()), {(("category", "unknown"),): 1.0}
        )

    def test_a_phase_records_its_outcome(self) -> None:
        metrics.observe_phase("records:review", "error", 9000)
        text = self._render_registry()
        self.assertEqual(
            _samples(text)['va_lse_phase_duration_ms_count{outcome="error",phase="records:review"}'],
            "1.0",
        )

    def test_a_breaker_transition_records_both_ends(self) -> None:
        metrics.observe_breaker_transition("llm", "CLOSED", "OPEN")
        metrics.observe_breaker_rejection("llm")
        samples = _by_key(metrics.breaker_transitions_total.samples())
        self.assertEqual(
            samples[(("breaker", "llm"), ("from_state", "CLOSED"), ("to_state", "OPEN"))],
            1.0,
        )
        self.assertEqual(
            _by_key(metrics.breaker_rejections_total.samples()), {(("breaker", "llm"),): 1.0}
        )

    def test_reset_clears_every_family_including_sessions(self) -> None:
        metrics.observe_llm_call("draft", "ok", 1)
        metrics.observe_phase("topic", "ok", 1)
        metrics.sessions.touch("sid-1")
        metrics.reset_for_tests()
        self.assertEqual(metrics.llm_call_duration_ms.observed, 0)
        self.assertEqual(metrics.llm_calls_total.samples(), [])
        self.assertEqual(metrics.phase_duration_ms.samples(), [])
        self.assertEqual(metrics.session_count(), 0)


class TestPhaseTimerWiring(_RegistryCase):
    def setUp(self) -> None:
        super().setUp()
        self.logger = logging.getLogger("test.metrics.phase")
        self.logger.setLevel(logging.CRITICAL + 1)  # keep phase logs out of test output

    def test_a_completed_phase_records_a_duration(self) -> None:
        with logging_config.PhaseTimer(self.logger, "records:review"):
            pass
        self.assertEqual(metrics.phase_duration_ms.observed, 1)
        labels = metrics.phase_duration_ms.samples()[0][0]
        self.assertEqual(labels, {"outcome": "ok", "phase": "records:review"})

    def test_a_failed_phase_records_the_error_outcome_and_re_raises(self) -> None:
        with self.assertRaises(ValueError):
            with logging_config.PhaseTimer(self.logger, "claims"):
                raise ValueError("bad claim")
        self.assertEqual(
            metrics.phase_duration_ms.samples()[0][0],
            {"outcome": "error", "phase": "claims"},
        )

    def test_the_recorded_duration_is_a_plausible_millisecond_delta(self) -> None:
        with logging_config.PhaseTimer(self.logger, "rubric"):
            time.sleep(0.02)
        entry = metrics.phase_duration_ms.samples()[0][1]
        self.assertEqual(entry[-2], 1.0)
        self.assertGreaterEqual(entry[-1], 15.0)
        self.assertLess(entry[-1], 5000.0)

    def test_metrics_recording_cannot_break_a_phase(self) -> None:
        """The logging layer sits under everything: recording must be best-effort."""
        with patch("app.metrics.observe_phase", side_effect=RuntimeError("boom")):
            with logging_config.PhaseTimer(self.logger, "topic"):
                pass


class TestLlmWiring(_RegistryCase):
    def setUp(self) -> None:
        super().setUp()
        circuit_breaker.reset_all_for_tests()

    @staticmethod
    def _client_returning(text: str = "ok response") -> LLMClient:
        client = LLMClient(_FakeSettings())  # type: ignore[arg-type]
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content=text))]
        response.usage = None
        client._client.chat.completions.create = MagicMock(  # type: ignore[attr-defined]
            return_value=response
        )
        return client

    def test_a_successful_call_records_one_call_and_one_attempt(self) -> None:
        client = self._client_returning("drafted")
        self.assertEqual(client.chat("sys", "usr", phase="draft"), "drafted")
        self.assertEqual(
            _by_key(metrics.llm_calls_total.samples()),
            {(("outcome", "ok"), ("phase", "draft")): 1.0},
        )
        self.assertEqual(metrics.llm_call_duration_ms.observed, 1)
        # The successful attempt counts too, so attempts/calls is 1.0 with no retries.
        self.assertEqual(
            _by_key(metrics.llm_attempts_total.samples()),
            {(("outcome", "ok"), ("phase", "draft")): 1.0},
        )

    def test_attempts_never_fall_below_calls(self) -> None:
        """`attempts / calls` is the retry-overhead alert; it must be >= 1."""
        client = self._client_returning()
        client.chat("s", "u", phase="topic")
        attempts = sum(value for _, value in metrics.llm_attempts_total.samples())
        calls = sum(value for _, value in metrics.llm_calls_total.samples())
        self.assertGreaterEqual(attempts, calls)

    def test_a_client_error_is_recorded_as_one_failed_call(self) -> None:
        client = LLMClient(_FakeSettings())  # type: ignore[arg-type]
        error = Exception("Error code: 401 - invalid api key")
        error.status_code = 401  # type: ignore[attr-defined]
        error.response = MagicMock(status_code=401)  # type: ignore[attr-defined]
        client._client.chat.completions.create = MagicMock(  # type: ignore[attr-defined]
            side_effect=error
        )
        with self.assertRaises(Exception):
            client.chat("s", "u", phase="draft")
        self.assertEqual(
            _by_key(metrics.llm_calls_total.samples()),
            {(("outcome", "error"), ("phase", "draft")): 1.0},
        )
        self.assertEqual(
            _by_key(metrics.llm_errors_total.samples()), {(("category", "client"),): 1.0}
        )

    def test_a_failed_call_records_no_success_attempt(self) -> None:
        client = LLMClient(_FakeSettings())  # type: ignore[arg-type]
        error = Exception("Error code: 401 - invalid api key")
        error.status_code = 401  # type: ignore[attr-defined]
        error.response = MagicMock(status_code=401)  # type: ignore[attr-defined]
        client._client.chat.completions.create = MagicMock(  # type: ignore[attr-defined]
            side_effect=error
        )
        with self.assertRaises(Exception):
            client.chat("s", "u", phase="draft")
        outcomes = {labels["outcome"] for labels, _ in metrics.llm_attempts_total.samples()}
        self.assertEqual(outcomes, {"error"})

    def test_an_open_breaker_counts_rejections_not_attempts(self) -> None:
        """A fail-fast rejection is not an endpoint failure and must not be counted as one."""
        circuit_breaker.reset_all_for_tests(breaker_threshold=1)
        breaker = circuit_breaker.get_llm_breaker()
        breaker.record_failure()
        self.assertEqual(breaker.state, "OPEN")
        client = self._client_returning()
        with self.assertRaises(circuit_breaker.CircuitBreakerOpenError):
            client.chat("s", "u", phase="draft")
        # Proven: the endpoint was never touched.
        self.assertEqual(client._client.chat.completions.create.call_count, 0)  # type: ignore[attr-defined]
        self.assertEqual(len(metrics.breaker_rejections_total.samples()), 1)
        self.assertEqual(metrics.llm_attempts_total.samples(), [])
        self.assertEqual(len(metrics.breaker_transitions_total.samples()), 1)

    def test_the_first_failure_does_not_open_the_breaker_on_a_healthy_default(self) -> None:
        """Resetting the singletons is what makes the rejection test above meaningful."""
        client = LLMClient(_FakeSettings())  # type: ignore[arg-type]
        error = Exception("boom")
        error.status_code = 401  # type: ignore[attr-defined]
        error.response = MagicMock(status_code=401)  # type: ignore[attr-defined]
        client._client.chat.completions.create = MagicMock(  # type: ignore[attr-defined]
            side_effect=error
        )
        with self.assertRaises(Exception):
            client.chat("s", "u", phase="draft")
        self.assertEqual(circuit_breaker.get_llm_breaker().state, "CLOSED")
        self.assertEqual(circuit_breaker.get_llm_breaker().failure_count, 1)


class TestRuntimeSeriesInScrape(unittest.TestCase):
    """The runtime gauges an operator pages on must actually appear in a scrape."""

    def setUp(self) -> None:
        metrics.reset_for_tests()
        circuit_breaker.reset_all_for_tests()

    def tearDown(self) -> None:
        metrics.reset_for_tests()
        circuit_breaker.reset_all_for_tests()

    def test_breaker_and_limiter_state_are_reported(self) -> None:
        text = metrics.render_prometheus({"uptime_s": 1})
        samples = _samples(text)
        self.assertEqual(samples['va_lse_circuit_breaker_state{breaker="llm"}'], "0")
        self.assertEqual(samples['va_lse_circuit_breaker_open{breaker="llm"}'], "0")
        self.assertEqual(
            samples['va_lse_circuit_breaker_consecutive_failures{breaker="llm"}'], "0"
        )
        self.assertIn("va_lse_llm_active_calls", samples)
        self.assertIn("va_lse_llm_max_concurrent_calls", samples)
        self.assertIn("va_lse_active_requests", samples)

    def test_an_open_breaker_is_visible_as_a_boolean_gauge(self) -> None:
        circuit_breaker.reset_all_for_tests(breaker_threshold=1)
        circuit_breaker.get_llm_breaker().record_failure()
        samples = _samples(metrics.render_prometheus({"uptime_s": 1}))
        self.assertEqual(samples['va_lse_circuit_breaker_state{breaker="llm"}'], "2")
        self.assertEqual(samples['va_lse_circuit_breaker_open{breaker="llm"}'], "1")

    def test_readiness_is_absent_until_something_has_checked(self) -> None:
        """/metrics must not probe, and must not invent a verdict it never checked."""
        with patch("app.health.cached_readiness_state", return_value=None):
            self.assertNotIn("va_lse_ready", metrics.render_prometheus({"uptime_s": 1}))

    def test_the_cached_readiness_verdict_is_reported(self) -> None:
        with patch("app.health.cached_readiness_state", return_value=False):
            samples = _samples(metrics.render_prometheus({"uptime_s": 1}))
        self.assertEqual(samples["va_lse_ready"], "0")
        with patch("app.health.cached_readiness_state", return_value=True):
            samples = _samples(metrics.render_prometheus({"uptime_s": 1}))
        self.assertEqual(samples["va_lse_ready"], "1")

    def test_a_failing_source_does_not_blank_the_other_series(self) -> None:
        """The metrics that survive an outage are the ones an operator needs."""
        with patch("app.shutdown.inflight_count", side_effect=RuntimeError("boom")):
            samples = _samples(metrics.render_prometheus({"uptime_s": 1}))
        self.assertNotIn("va_lse_active_requests", samples)
        self.assertEqual(samples["va_lse_up"], "1")
        self.assertIn('va_lse_circuit_breaker_state{breaker="llm"}', samples)

    def test_a_missing_circuit_breaker_module_does_not_blank_the_exposition(self) -> None:
        with patch.dict(sys.modules, {"app.circuit_breaker": None}):
            samples = _samples(metrics.render_prometheus({"uptime_s": 1}))
        self.assertNotIn("va_lse_circuit_breaker_state", samples)
        self.assertEqual(samples["va_lse_up"], "1")

    def test_the_runtime_gauges_are_local_reads(self) -> None:
        """A scrape shares a port with the liveness probe and must not touch the network."""
        backend = _CountingBackend()
        with patch("app.job_queue.get_job_backend", return_value=backend):
            metrics.render_prometheus({"uptime_s": 1})
        self.assertEqual(backend.depth_reads, 0)

    def test_every_registry_family_is_in_the_documented_name_list(self) -> None:
        """The registry families must be listed, or the dashboard/alert drift check
        (which validates queries against metric_names()) silently ignores them."""
        registry = {
            metrics.phase_duration_ms.name,
            metrics.llm_call_duration_ms.name,
            metrics.llm_calls_total.name,
            metrics.llm_attempts_total.name,
            metrics.llm_errors_total.name,
            metrics.breaker_rejections_total.name,
            metrics.breaker_transitions_total.name,
            metrics._series_dropped.name,
        }
        self.assertEqual(registry - set(metrics.metric_names()), set(), "not documented")


if __name__ == "__main__":
    unittest.main()
