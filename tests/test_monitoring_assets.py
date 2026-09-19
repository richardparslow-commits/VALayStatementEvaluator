"""Tests for the shipped monitoring assets (deploy/monitoring/*).

The failure these exist to prevent is drift: an alert rule or a dashboard panel
that references a metric the app does not emit. Nothing errors — Prometheus simply
records no data for that expression, the panel renders empty, and the alert never
fires. A monitoring stack that silently watches nothing is worse than no stack,
because it is trusted.

So the central assertion here is that every ``va_lse_*`` name appearing in an
alert expression, an alert annotation, or a dashboard target is a real metric that
:func:`app.metrics.metric_names` knows about.
"""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app import metrics  # noqa: E402

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is a test dependency
    yaml = None  # type: ignore[assignment]

DEPLOY = Path(__file__).resolve().parent.parent / "deploy" / "monitoring"

# Metric name as it appears in PromQL or prose. Lowercase with underscores, which
# is why `va-lse-web` (a job name) and `VA_LSE_REDIS_URL` (an env var) do not match.
_MENTION = re.compile(r"\bva_lse_[a-z0-9_]+")

# Histograms are exposed as <name>_bucket / <name>_sum / <name>_count, and a
# histogram *aggregation* query references those children, not the base name.
_HISTOGRAM_SUFFIXES = ("_bucket", "_sum", "_count")


def _base_name(name: str) -> str:
    for suffix in _HISTOGRAM_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _known() -> set[str]:
    return set(metrics.metric_names())


def _resolves(name: str, known: set[str]) -> bool:
    """Whether a name mentioned in an asset is a metric the app emits.

    The stripped form is tried *in addition to* the literal name, not instead of
    it, because the suffix rule is ambiguous: a histogram's children are
    ``<name>_bucket``/``_sum``/``_count``, but a counter may legitimately be called
    ``va_lse_session_count`` in its own right. Checking both is what keeps this
    helper from reporting a real metric as missing.
    """
    return name in known or _base_name(name) in known


def _load_yaml(path: Path):
    assert yaml is not None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _walk_strings(node) -> list[str]:  # noqa: ANN001 - recursive over unknown shapes
    """Every string anywhere in a parsed YAML/JSON document."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        out: list[str] = []
        for key, value in node.items():
            out.append(str(key))
            out.extend(_walk_strings(value))
        return out
    if isinstance(node, list):
        out = []
        for item in node:
            out.extend(_walk_strings(item))
        return out
    return []


@unittest.skipIf(yaml is None, "PyYAML not installed")
class TestAlertRules(unittest.TestCase):
    def setUp(self) -> None:
        self.path = DEPLOY / "alerts.yml"
        self.doc = _load_yaml(self.path)
        self.rules = [rule for group in self.doc["groups"] for rule in group["rules"]]

    def test_file_exists_and_parses(self) -> None:
        self.assertTrue(self.path.exists())
        self.assertTrue(self.rules)

    def test_every_referenced_metric_is_one_the_app_emits(self) -> None:
        known = _known()
        unknown: list[tuple[str, str]] = []
        for rule in self.rules:
            for text in _walk_strings(rule):
                for mention in _MENTION.findall(text):
                    if not _resolves(mention, known):
                        unknown.append((rule["alert"], mention))
        self.assertEqual(unknown, [], f"alerts reference metrics that do not exist: {unknown}")

    def test_every_rule_routes_somewhere(self) -> None:
        """AlertManager routing keys off `severity`, so a rule without one is a
        rule that reaches the default receiver and may never page anyone."""
        for rule in self.rules:
            self.assertIn("severity", rule.get("labels", {}), rule["alert"])
            self.assertIn(
                rule["labels"]["severity"], {"critical", "warning"}, rule["alert"]
            )

    def test_every_rule_explains_itself_and_has_a_hold_time(self) -> None:
        for rule in self.rules:
            annotations = rule.get("annotations", {})
            self.assertTrue(annotations.get("summary"), rule["alert"])
            self.assertTrue(annotations.get("description"), rule["alert"])
            # `for` may be absent in principle; here every rule needs one, because
            # each of these conditions is normal for a few seconds at a time.
            self.assertIn("for", rule, f"{rule['alert']} has no hold time")

    def test_alert_names_are_unique(self) -> None:
        names = [rule["alert"] for rule in self.rules]
        self.assertEqual(len(names), len(set(names)), "duplicate alert names")

    def test_the_specs_required_alerts_are_present(self) -> None:
        """The two the request called out by name."""
        by_name = {rule["alert"]: rule for rule in self.rules}
        self.assertIn("VaElseCircuitBreakerOpen", by_name)
        self.assertIn("VaElseReadinessFailing", by_name)
        # Circuit breaker open for *more than a minute*, not instantly.
        self.assertEqual(by_name["VaElseCircuitBreakerOpen"]["for"], "1m")
        # Readiness is an HTTP status, so it is probed rather than scraped.
        self.assertIn("probe_success", by_name["VaElseReadinessFailing"]["expr"])

    def test_every_group_has_an_interval(self) -> None:
        for group in self.doc["groups"]:
            self.assertIn("interval", group, group["name"])


@unittest.skipIf(yaml is None, "PyYAML not installed")
class TestDashboard(unittest.TestCase):
    def setUp(self) -> None:
        self.path = DEPLOY / "grafana-dashboard.json"
        self.doc = json.loads(self.path.read_text(encoding="utf-8"))
        self.panels = self.doc["panels"]

    def test_is_valid_json_with_a_uid_and_panels(self) -> None:
        self.assertTrue(self.doc["uid"])
        self.assertTrue(self.panels)

    def test_every_referenced_metric_is_one_the_app_emits(self) -> None:
        known = _known()
        unknown: list[tuple[str, str]] = []
        for panel in self.panels:
            for target in panel.get("targets", []):
                expr = target.get("expr", "")
                for mention in _MENTION.findall(expr):
                    if not _resolves(mention, known):
                        unknown.append((panel["title"], mention))
        self.assertEqual(unknown, [], f"panels reference metrics that do not exist: {unknown}")

    def test_every_panel_has_a_title_query_and_datasource(self) -> None:
        for panel in self.panels:
            self.assertTrue(panel.get("title"), panel.get("id"))
            self.assertTrue(panel.get("targets"), panel["title"])
            for target in panel["targets"]:
                self.assertTrue(target.get("expr"), panel["title"])
            self.assertTrue(panel.get("datasource", {}).get("uid"), panel["title"])

    def test_panel_ids_are_unique(self) -> None:
        ids = [panel["id"] for panel in self.panels]
        self.assertEqual(len(ids), len(set(ids)), "duplicate panel ids")

    def test_panels_use_the_provisioned_datasource_uid(self) -> None:
        """The datasource file pins uid=prometheus; a panel using another uid
        renders 'datasource not found' rather than data."""
        datasource = _load_yaml(
            DEPLOY / "grafana-provisioning" / "datasources" / "prometheus.yml"
        )
        uid = datasource["datasources"][0]["uid"]
        for panel in self.panels:
            self.assertEqual(panel["datasource"]["uid"], uid, panel["title"])

    def test_the_specs_required_panels_exist(self) -> None:
        """Latency percentiles, breaker state, and endpoint availability."""
        titles = " | ".join(panel["title"].lower() for panel in self.panels)
        self.assertIn("p50", titles)
        self.assertIn("p95", titles)
        self.assertIn("p99", titles)
        self.assertIn("circuit breaker", titles)
        self.assertIn("readiness", titles)

    def test_percentile_panels_use_histogram_quantile(self) -> None:
        percentile_panels = [p for p in self.panels if "p95" in p["title"].lower()]
        self.assertTrue(percentile_panels)
        for panel in percentile_panels:
            exprs = " ".join(t["expr"] for t in panel["targets"])
            self.assertIn("histogram_quantile", exprs, panel["title"])
            # Quantiles must be summed by `le` or the result is meaningless.
            self.assertIn("by (le", exprs, panel["title"])

    def test_the_phase_variable_is_used_by_at_least_one_panel(self) -> None:
        variables = {v["name"] for v in self.doc["templating"]["list"]}
        self.assertIn("phase", variables)
        exprs = " ".join(
            target["expr"] for panel in self.panels for target in panel.get("targets", [])
        )
        self.assertIn("$phase", exprs)


@unittest.skipIf(yaml is None, "PyYAML not installed")
class TestScrapeConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.doc = _load_yaml(DEPLOY / "prometheus.yml")
        self.jobs = {job["job_name"]: job for job in self.doc["scrape_configs"]}

    def test_scrapes_both_tiers_on_their_health_ports(self) -> None:
        web = self.jobs["va-lse-web"]["static_configs"][0]["targets"]
        worker = self.jobs["va-lse-worker"]["static_configs"][0]["targets"]
        self.assertIn("web:8001", web)
        self.assertIn("worker:8002", worker)

    def test_scrape_never_forces_the_expensive_queue_probe(self) -> None:
        """`?probe=1` costs a network round trip; a 15s scrape must not pay it."""
        for job in self.doc["scrape_configs"]:
            self.assertNotIn("probe=1", job.get("metrics_path", ""))

    def test_rule_files_are_wired(self) -> None:
        self.assertIn("/etc/prometheus/alerts.yml", self.doc["rule_files"])

    def test_alertmanager_is_configured(self) -> None:
        targets = self.doc["alerting"]["alertmanagers"][0]["static_configs"][0]["targets"]
        self.assertIn("alertmanager:9093", targets)

    def test_readiness_is_probed_through_blackbox(self) -> None:
        job = self.jobs["va-lse-readiness"]
        replacements = {r.get("target_label"): r.get("replacement") for r in job["relabel_configs"]}
        self.assertEqual(replacements["__address__"], "blackbox-exporter:9115")
        self.assertIn("/ready", job["static_configs"][0]["targets"][0])


@unittest.skipIf(yaml is None, "PyYAML not installed")
class TestComposeMonitoringProfile(unittest.TestCase):
    def setUp(self) -> None:
        self.doc = _load_yaml(Path(__file__).resolve().parent.parent / "docker-compose.yml")
        self.services = self.doc["services"]

    def test_monitoring_profile_defines_the_stack(self) -> None:
        profile = {
            name
            for name, svc in self.services.items()
            if "monitoring" in (svc.get("profiles") or [])
        }
        self.assertEqual(
            profile, {"prometheus", "alertmanager", "grafana", "blackbox-exporter"}
        )

    def test_the_app_itself_is_untouched_by_the_profile(self) -> None:
        """Monitoring must be purely additive: enabling it must not change how the
        app runs, or the stack becomes a deployment risk rather than a safety net."""
        self.assertIn("streamlit-web", self.services)
        for name in ("streamlit-web", "audit-backup"):
            self.assertTrue(
                not self.services[name].get("profiles"),
                f"{name} must not be gated behind a profile",
            )

    def test_every_mounted_config_exists_on_disk(self) -> None:
        root = Path(__file__).resolve().parent.parent
        for name, svc in self.services.items():
            if "monitoring" not in (svc.get("profiles") or []):
                continue
            for mount in svc.get("volumes", []):
                source = str(mount).split(":")[0]
                if not source.startswith("./"):
                    continue
                self.assertTrue(
                    (root / source).exists(), f"{name} mounts missing path {source}"
                )

    def test_named_volumes_are_declared(self) -> None:
        declared = set(self.doc["volumes"])
        for name, svc in self.services.items():
            for mount in svc.get("volumes", []):
                source = str(mount).split(":")[0]
                if source.startswith("./") or "/" in source:
                    continue
                self.assertIn(source, declared, f"{name} mounts undeclared volume {source}")


if __name__ == "__main__":
    unittest.main()
