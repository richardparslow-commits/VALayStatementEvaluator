"""The operator's failover rehearsal tool, driven over a real socket.

``scripts/rehearse_failover.py`` is what someone runs *during* a provider outage,
so the two things that matter most are tested the hardest:

* **its exit codes**, because a rehearsal step is only meaningful if it fails
  loudly when the deployment is not in the stage that was asserted; and
* **its output being parseable**, because ``--json`` is the hook a check script
  or a CI stage uses. A defect found here while writing this file: ``--json``
  printed the JSON document *and then* the human verdict text on the same
  stream, so nothing downstream could parse it. Prose is now suppressed on that
  path while the exit code still reflects ``--expect-*``.

The server below is a loopback HTTP server rather than a patched ``snapshot``,
so the request path, the 503 handling, and the metric parsing are all exercised
the way the real deployment exercises them.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "rehearse_failover", PROJECT_ROOT / "scripts" / "rehearse_failover.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rehearse = _load_cli()

METRIC_NAMES = (
    "va_lse_llm_failover_enabled",
    "va_lse_llm_failover_active",
    "va_lse_llm_failover_after_seconds",
    "va_lse_llm_primary_unhealthy_seconds",
    'va_lse_circuit_breaker_state{breaker="llm"}',
    'va_lse_circuit_breaker_open{breaker="llm"}',
)


class _ProbeHandler(BaseHTTPRequestHandler):
    """Serves whatever the test staged, and nothing else."""

    protocol_version = "HTTP/1.1"
    health: dict = {}
    metrics: str = ""
    status_code = 200

    def do_GET(self) -> None:  # noqa: N802
        body = (
            json.dumps(self.health).encode()
            if self.path.startswith("/health")
            else self.metrics.encode()
        )
        self.send_response(self.status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # keep test output clean
        return


def metrics_text(
    *,
    enabled: float | None = 1.0,
    active: float | None = 0.0,
    after: float | None = 300.0,
    unhealthy: float | None = None,
    breaker_state: str = "CLOSED",
    breaker_open: float = 0.0,
) -> str:
    """A minimal but faithful slice of the real exposition."""
    codes = {"CLOSED": "0", "HALF_OPEN": "1", "OPEN": "2"}
    lines = ["# HELP va_lse_llm_failover_enabled whether a backup endpoint is armed"]
    for name, value in (
        ("va_lse_llm_failover_enabled", enabled),
        ("va_lse_llm_failover_active", active),
        ("va_lse_llm_failover_after_seconds", after),
        ("va_lse_llm_primary_unhealthy_seconds", unhealthy),
    ):
        if value is not None:
            lines.append(f"{name} {value}")
    lines.append(f'va_lse_circuit_breaker_state{{breaker="llm"}} {codes[breaker_state]}')
    lines.append(f'va_lse_circuit_breaker_open{{breaker="llm"}} {breaker_open}')
    return "\n".join(lines) + "\n"


class _ProbeCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _ProbeHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self) -> None:
        self.url = f"http://127.0.0.1:{self.port}"
        _ProbeHandler.status_code = 200
        self.stage_health(configured=True, active=False, unhealthy=None, after=300)
        _ProbeHandler.metrics = metrics_text()

    def stage_health(
        self,
        *,
        configured: bool,
        active: bool,
        unhealthy: float | None,
        after: float | None,
        status: str = "ok",
    ) -> None:
        _ProbeHandler.health = {
            "status": status,
            "llm_failover": {
                "configured": configured,
                "active": active,
                "after_seconds": after,
                "primary_unhealthy_seconds": unhealthy,
            },
        }

    def run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = rehearse.main(["--url", self.url, *argv])
        return code, out.getvalue(), err.getvalue()


class TestSnapshot(_ProbeCase):
    def test_reads_health_and_metrics_into_one_snapshot(self) -> None:
        snap = rehearse.snapshot(self.url)
        self.assertTrue(snap["feature_present"])
        self.assertTrue(snap["configured"])
        self.assertFalse(snap["active"])
        self.assertEqual(snap["metrics"]["failover_after_seconds"], 300.0)
        self.assertEqual(snap["metrics"]["primary_breaker_state"], "CLOSED")

    def test_an_older_pod_is_reported_as_such_not_as_zeros(self) -> None:
        """A missing block must not read as "configured=False, active=False".

        Those are real values meaning "one endpoint, healthy primary". A pod that
        predates the feature has no opinion, and conflating the two would let a
        rehearsal pass against a build that cannot fail over at all.
        """
        _ProbeHandler.health = {"status": "ok"}
        snap = rehearse.snapshot(self.url)
        self.assertFalse(snap["feature_present"])
        self.assertIn("NOT REPORTED", rehearse.describe(snap))

    def test_an_absent_series_is_none_not_zero(self) -> None:
        """The exposition omits what it could not read; 0 is a real value."""
        _ProbeHandler.metrics = metrics_text(unhealthy=None)
        snap = rehearse.snapshot(self.url)
        self.assertIsNone(snap["metrics"]["primary_unhealthy_seconds"])
        self.assertIsNone(snap["primary_unhealthy_seconds"])

    def test_a_labelled_series_does_not_satisfy_an_unlabelled_lookup(self) -> None:
        """Otherwise a per-breaker series would be read as the global one."""
        self.assertIsNone(
            rehearse._metric(
                'va_lse_llm_failover_enabled{breaker="llm"} 1',
                "va_lse_llm_failover_enabled",
            )
        )

    def test_an_unreachable_deployment_is_an_error_not_a_crash(self) -> None:
        code, _, err = self.run_cli(["--url", "http://127.0.0.1:1"])
        self.assertEqual(code, 1)
        self.assertIn("Could not read", err)

    def test_the_primary_breaker_state_comes_from_the_enum_gauge(self) -> None:
        _ProbeHandler.metrics = metrics_text(breaker_state="OPEN", breaker_open=1.0)
        snap = rehearse.snapshot(self.url)
        self.assertEqual(snap["metrics"]["primary_breaker_state"], "OPEN")
        self.assertEqual(snap["metrics"]["primary_breaker_open"], 1.0)

    def test_a_breaker_absent_from_the_exposition_says_absent(self) -> None:
        _ProbeHandler.metrics = "va_lse_llm_failover_enabled 1\n"
        self.assertEqual(rehearse.snapshot(self.url)["metrics"]["primary_breaker_state"], "ABSENT")


class TestJsonContract(_ProbeCase):
    """`--json` exists to be consumed, so its stdout must be exactly one document."""

    def test_json_output_is_parseable_and_contains_no_prose(self) -> None:
        code, out, _ = self.run_cli(["--json"])
        self.assertEqual(code, 0)
        snap = json.loads(out)
        self.assertTrue(snap["configured"])
        self.assertNotIn("OK:", out)
        self.assertNotIn("INCONSISTENT", out)

    def test_json_still_carries_the_expectation_verdict(self) -> None:
        """Suppressing prose must not suppress the check the flag was used with."""
        code, out, _ = self.run_cli(["--json", "--expect-active"])
        self.assertEqual(code, 2)
        json.loads(out)  # still parseable

    def test_json_reports_an_inconsistency_through_its_exit_code(self) -> None:
        _ProbeHandler.metrics = metrics_text(enabled=0.0, active=1.0)
        self.stage_health(configured=True, active=True, unhealthy=400, after=300)
        code, out, _ = self.run_cli(["--json"])
        self.assertEqual(code, 1)
        json.loads(out)


class TestExitCodes(_ProbeCase):
    def test_expect_idle_passes_when_the_primary_is_healthy(self) -> None:
        code, out, _ = self.run_cli(["--expect-idle"])
        self.assertEqual(code, 0, out)
        self.assertIn("on the primary endpoint", out)

    def test_expect_idle_fails_when_traffic_is_on_the_backup(self) -> None:
        self.stage_health(configured=True, active=True, unhealthy=600, after=300)
        _ProbeHandler.metrics = metrics_text(
            active=1.0, unhealthy=600.0, breaker_state="OPEN", breaker_open=1.0
        )
        code, out, _ = self.run_cli(["--expect-idle"])
        self.assertEqual(code, 2, out)
        self.assertIn("EXPECTED IDLE", out)

    def test_expect_active_fails_while_traffic_is_still_on_the_primary(self) -> None:
        code, out, _ = self.run_cli(["--expect-active"])
        self.assertEqual(code, 2)
        self.assertIn("EXPECTED ACTIVE", out)

    def test_expect_active_passes_once_the_backup_is_serving(self) -> None:
        self.stage_health(configured=True, active=True, unhealthy=600, after=300)
        _ProbeHandler.metrics = metrics_text(
            active=1.0, unhealthy=600.0, breaker_state="OPEN", breaker_open=1.0
        )
        code, out, _ = self.run_cli(["--expect-active"])
        self.assertEqual(code, 0, out)
        self.assertIn("llm_endpoints", out)

    def test_a_single_endpoint_deployment_is_a_valid_idle_state(self) -> None:
        """No fallback armed is a supported configuration, not a failed rehearsal."""
        self.stage_health(configured=False, active=False, unhealthy=None, after=None)
        code, out, _ = self.run_cli(["--expect-idle"])
        self.assertEqual(code, 0, out)
        self.assertIn("no failover configured", out)

    def test_watch_and_expect_are_rejected_together(self) -> None:
        code, _, err = self.run_cli(["--watch", "--expect-active"])
        self.assertEqual(code, 2)
        self.assertIn("different modes", err)

    def test_an_unknown_flag_fails_loudly(self) -> None:
        """argparse's exit 2 is why 2 means "not the stage expected" and not "crash"."""
        with self.assertRaises(SystemExit) as ctx:
            rehearse.main(["--expect-nonsense"])
        self.assertEqual(ctx.exception.code, 2)


class TestInconsistencyDetection(_ProbeCase):
    """A monitoring surface that reports something impossible is worse than silence."""

    def test_active_with_failover_disabled_names_both_values(self) -> None:
        _ProbeHandler.metrics = metrics_text(enabled=0.0, active=1.0)
        self.stage_health(configured=True, active=True, unhealthy=600, after=300)
        problems = rehearse.inconsistencies(rehearse.snapshot(self.url))
        self.assertTrue(problems)
        self.assertTrue(any("failover_enabled is 0" in p for p in problems))

    def test_health_and_metrics_disagreeing_is_caught(self) -> None:
        self.stage_health(configured=True, active=True, unhealthy=600, after=300)
        _ProbeHandler.metrics = metrics_text(
            active=0.0, unhealthy=600.0, breaker_state="OPEN", breaker_open=1.0
        )
        problems = rehearse.inconsistencies(rehearse.snapshot(self.url))
        self.assertTrue(any("the two surfaces disagree" in p for p in problems))

    def test_past_the_threshold_but_still_on_the_primary_is_caught(self) -> None:
        """The exact dead-code failure mode the grace period could have had."""
        self.stage_health(configured=True, active=False, unhealthy=361, after=300)
        _ProbeHandler.metrics = metrics_text(
            active=0.0, unhealthy=361.0, breaker_state="OPEN", breaker_open=1.0
        )
        problems = rehearse.inconsistencies(rehearse.snapshot(self.url))
        self.assertTrue(any("traffic should already have moved" in p for p in problems))

    def test_inside_the_grace_period_is_not_an_inconsistency(self) -> None:
        """Failing fast while the primary is retried is the documented contract."""
        self.stage_health(configured=True, active=False, unhealthy=30, after=300)
        _ProbeHandler.metrics = metrics_text(
            active=0.0, unhealthy=30.0, breaker_state="OPEN", breaker_open=1.0
        )
        self.assertEqual(rehearse.inconsistencies(rehearse.snapshot(self.url)), [])

    def test_active_while_the_primary_breaker_is_closed_is_caught(self) -> None:
        self.stage_health(configured=True, active=True, unhealthy=600, after=300)
        _ProbeHandler.metrics = metrics_text(
            active=1.0, unhealthy=600.0, breaker_state="CLOSED"
        )
        problems = rehearse.inconsistencies(rehearse.snapshot(self.url))
        self.assertTrue(any("breaker is CLOSED" in p for p in problems))

    def test_a_healthy_armed_deployment_has_no_problems(self) -> None:
        self.assertEqual(rehearse.inconsistencies(rehearse.snapshot(self.url)), [])


class TestDescribe(_ProbeCase):
    def test_the_countdown_is_shown_while_the_primary_is_failing(self) -> None:
        self.stage_health(configured=True, active=False, unhealthy=120, after=300)
        _ProbeHandler.metrics = metrics_text(active=0.0, unhealthy=120.0)
        text = rehearse.describe(rehearse.snapshot(self.url))
        self.assertIn("failover in ~180s", text)

    def test_a_healthy_primary_reads_as_armed_and_unused(self) -> None:
        text = rehearse.describe(rehearse.snapshot(self.url))
        self.assertIn("armed and unused", text)
        self.assertIn("healthy", text)


class TestAgainstTheRealExposition(unittest.TestCase):
    """The script's expectations must match what the app actually serves.

    ``_metric`` returns ``None`` for a name that is not in the exposition, which
    is the correct behaviour but a silent one: a renamed or re-labelled series
    would make the rehearsal tool report "not reported" instead of failing. These
    two checks close that gap without a running server.
    """

    def test_every_name_the_script_reads_is_exported_by_metrics(self) -> None:
        from app import metrics

        known = set(metrics.metric_names())
        for name in METRIC_NAMES:
            base = name.split("{", 1)[0]
            with self.subTest(name=name):
                self.assertIn(base, known)

    def test_the_failover_block_is_present_in_a_real_health_payload(self) -> None:
        from app import health

        payload = health._health_payload()
        self.assertIn("llm_failover", payload)
        block = payload["llm_failover"]
        self.assertIn("configured", block)
        self.assertIn("active", block)
        # The script renders both of these; a missing key would print "unknown".
        self.assertIn("after_seconds", block)
        self.assertIn("primary_unhealthy_seconds", block)

    def test_the_real_exposition_parses_cleanly(self) -> None:
        from app import health, metrics

        text = metrics.render_prometheus(health._health_payload())
        self.assertEqual(rehearse._metric(text, "va_lse_llm_failover_enabled"), 0.0)
        self.assertIn(rehearse._breaker_state(text), {"CLOSED", "HALF_OPEN", "OPEN"})


if __name__ == "__main__":
    unittest.main()
