"""Tests for check_model_availability and compat warning wiring.
No network in the success path — responses are stubbed.
"""
import sys
import unittest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from app import config  # noqa: E402
from app.llm import check_model_availability, MODELS_ENDPOINT_TIMEOUT_SECONDS  # noqa: E402


class TestCheckModelAvailability(unittest.TestCase):
    def test_returns_ids_on_success(self):
        payload = json.dumps({"data": [{"id": "qwen3.7-max"}, {"id": "qwen3.7-flash"}]}).encode()
        fake_resp = MagicMock()
        fake_resp.read.return_value = payload
        fake_resp.__enter__ = lambda _: fake_resp
        fake_resp.__exit__ = lambda *_: False
        with patch("urllib.request.urlopen", return_value=fake_resp) as mock_urlopen:
            ids = check_model_availability("https://example.com/v1", "sk-test")
            self.assertEqual(ids, {"qwen3.7-max", "qwen3.7-flash"})
            # called with /models and Authorization header
            req = mock_urlopen.call_args[0][0]
            self.assertTrue(req.full_url.endswith("/models"))
            self.assertIn("Bearer sk-test", req.headers.get("Authorization", ""))

    def test_returns_none_without_api_key(self):
        self.assertIsNone(check_model_availability("https://example.com/v1", ""))
        self.assertIsNone(check_model_availability("https://example.com/v1", "   "))

    def test_returns_none_on_network_error(self):
        with patch("urllib.request.urlopen", side_effect=Exception("network down")):
            self.assertIsNone(check_model_availability("https://example.com/v1", "sk-x"))

    def test_returns_none_on_malformed_json(self):
        fake_resp = MagicMock()
        fake_resp.read.return_value = b"not json"
        fake_resp.__enter__ = lambda _: fake_resp
        fake_resp.__exit__ = lambda *_: False
        with patch("urllib.request.urlopen", return_value=fake_resp):
            self.assertIsNone(check_model_availability("https://example.com/v1", "sk-x"))

    def test_timeout_constant(self):
        self.assertGreaterEqual(MODELS_ENDPOINT_TIMEOUT_SECONDS, 5)

    def test_stripsTrailingSlash(self):
        payload = json.dumps({"data": []}).encode()
        fake_resp = MagicMock()
        fake_resp.read.return_value = payload
        fake_resp.__enter__ = lambda _: fake_resp
        fake_resp.__exit__ = lambda *_: False
        with patch("urllib.request.urlopen", return_value=fake_resp) as mock_urlopen:
            check_model_availability("https://example.com/v1/", "sk-x")
            self.assertEqual(mock_urlopen.call_args[0][0].full_url, "https://example.com/v1/models")


class TestCompatDocsPresent(unittest.TestCase):
    def test_files_exist(self):
        root = Path(__file__).resolve().parent.parent
        for name in ("COMPATIBILITY.md", "MIGRATION.md"):
            self.assertTrue((root / name).exists(), msg=f"{name} should exist")
            text = (root / name).read_text()
            self.assertIn("OPENAI_API_KEY" if name == "MIGRATION.md" else "Compatibility", text)


class TestEnvExampleDocumentsProviders(unittest.TestCase):
    def test_env_example_lists_providers(self):
        text = (Path(__file__).resolve().parent.parent / ".env.example").read_text()
        self.assertIn("COMPATIBILITY.md", text)
        self.assertIn("Perplexity", text)
        self.assertIn("OpenAI", text)
        self.assertIn("Ollama", text)

    def test_env_example_defaults_match_the_shipped_defaults(self):
        """The file a fresh clone copies must name the defaults app/config.py ships.

        This drifted once: the Router API became the default while .env.example
        still described QwenCloud as "(default)" with a Token Plan key and its
        models, so a new deployment landed on the old provider with no signal.
        """
        values: dict[str, str] = {}
        for line in (
            Path(__file__).resolve().parent.parent / ".env.example"
        ).read_text().splitlines():
            if line and not line.lstrip().startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()
        self.assertEqual(values.get("OPENAI_BASE_URL"), config.DEFAULT_BASE_URL)
        self.assertEqual(values.get("LLM_MODEL_MAIN"), config.DEFAULT_MODEL_MAIN)
        self.assertEqual(values.get("LLM_MODEL_FAST"), config.DEFAULT_MODEL_FAST)
        self.assertTrue(
            values.get("OPENAI_API_KEY", "").startswith("pplx-"),
            "the placeholder key should be a Perplexity key (pplx-...), got "
            + repr(values.get("OPENAI_API_KEY")),
        )


class TestReadmeEnvTableMatchesDefaults(unittest.TestCase):
    """The README environment table states defaults in prose; keep them true.

    Nothing else can catch a default moving in ``app/config.py`` while its README
    row keeps describing the old value - the drift ``.env.example`` shipped once
    (see ``TestEnvExampleDocumentsProviders``). Only rows whose Default column
    quotes a config value directly are listed; rows that describe derived
    behavior ("(empty = off)", "(required)") are prose and stay out.
    """

    #: README environment variable -> the config attribute its Default cell must quote.
    ROWS: dict[str, str] = {
        "OPENAI_BASE_URL": "DEFAULT_BASE_URL",
        "LLM_MODEL_MAIN": "DEFAULT_MODEL_MAIN",
        "LLM_MODEL_FAST": "DEFAULT_MODEL_FAST",
        "LLM_ENDPOINT_FALLBACK_TIMEOUT_SECONDS": "DEFAULT_FAILOVER_AFTER_SECONDS",
        "VA_LSE_MAX_RECORD_PAGES": "MAX_RECORD_PAGES",
        "VA_LSE_RECORDS_CONCURRENCY": "RECORDS_CONCURRENCY",
        "VA_LSE_PIPELINE_TIMEOUT_SECONDS": "PIPELINE_TIMEOUT_SECONDS",
        "VA_LSE_MAX_CONCURRENT_LLM_CALLS": "LLM_MAX_CONCURRENT",
        "VA_LSE_BLOB_STORE": "BLOB_STORE_MODE",
        "VA_LSE_HEALTH_PORT": "HEALTH_PORT",
        "VA_LSE_EXTRACTOR_TIMEOUT_SECONDS": "EXTRACTOR_TIMEOUT_SECONDS",
        "VA_LSE_MAX_DIGEST_FACTS": "MAX_DIGEST_FACTS",
        "VA_LSE_DIGEST_CHUNK_CHARS": "DIGEST_CHUNK_CHARS",
        "VA_LSE_LLM_CALL_TIMEOUT_SECONDS": "LLM_CALL_TIMEOUT_SECONDS",
        "VA_LSE_AUDIT_RETENTION_DAYS": "AUDIT_RETENTION_DAYS",
    }

    def _readme_defaults(self) -> dict[str, str]:
        """The table's rows as ``env name -> Default cell``; multi-name rows are skipped."""
        text = (Path(__file__).resolve().parent.parent / "README.md").read_text()
        rows: dict[str, str] = {}
        for line in text.splitlines():
            if not line.startswith("| `"):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 3:
                continue
            name = cells[0].strip("`")
            if "`" in name:  # a row naming two variables has no single default
                continue
            rows[name] = cells[2]
        return rows

    def test_readme_defaults_match_config(self) -> None:
        rows = self._readme_defaults()
        for env_name, attr in sorted(self.ROWS.items()):
            with self.subTest(env=env_name):
                self.assertIn(env_name, rows, f"{env_name} is missing from the README table")
                expected = getattr(config, attr)
                self.assertIn(
                    f"`{expected}`",
                    rows[env_name],
                    f"README's default for {env_name} is {rows[env_name]!r}, "
                    f"but config.{attr} ships {expected!r}",
                )


def _literal(value: object) -> str:
    """A config default as the documents write it: ``6.0`` renders as ``6``."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


class TestDeploymentAndCompatibilityTablesMatchDefaults(unittest.TestCase):
    """The other two documents that state defaults, guarded like the README's table.

    ``DEPLOYMENT.md``'s §9 table is the operator's checklist and
    ``COMPATIBILITY.md``'s "Current defaults" table is the provider reference;
    both quote values that live in ``app/config.py``, and neither was checked
    (the README and ``.env.example`` are, since ``1b79d2a`` / ``f64c8bd``). Rows
    whose Default is prose - "(empty = off)", "hostname:pid", "(stdout only)" -
    stay out: they have no single config value to compare.
    """

    #: DEPLOYMENT.md §9 environment variable -> config attribute its Default cell must quote.
    DEPLOYMENT_ROWS: dict[str, str] = {
        "OPENAI_BASE_URL": "DEFAULT_BASE_URL",
        "LLM_MODEL_MAIN": "DEFAULT_MODEL_MAIN",
        "LLM_MODEL_FAST": "DEFAULT_MODEL_FAST",
        "VA_LSE_RECORDS_CONCURRENCY": "RECORDS_CONCURRENCY",
        "VA_LSE_MAX_CONCURRENT_LLM_CALLS": "LLM_MAX_CONCURRENT",
        "VA_LSE_HEALTH_PORT": "HEALTH_PORT",
        "VA_LSE_SHUTDOWN_GRACE_SECONDS": "SHUTDOWN_GRACE_SECONDS",
        "VA_LSE_LLM_CALL_TIMEOUT_SECONDS": "LLM_CALL_TIMEOUT_SECONDS",
        "VA_LSE_AUDIT_LOG_MAX_BYTES": "AUDIT_LOG_MAX_BYTES",
        "VA_LSE_AUDIT_LOG_BACKUPS": "AUDIT_LOG_BACKUPS",
        "VA_LSE_AUDIT_RETENTION_DAYS": "AUDIT_RETENTION_DAYS",
        "VA_LSE_AUDIT_BACKUP_INTERVAL_HOURS": "AUDIT_BACKUP_INTERVAL_HOURS",
        "VA_LSE_RUN_LOG_MAX_BYTES": "RUN_LOG_MAX_BYTES",
        "VA_LSE_RUN_LOG_BACKUPS": "RUN_LOG_BACKUPS",
        "VA_LSE_DISK_MIN_FREE_BYTES": "DISK_MIN_FREE_BYTES",
        "VA_LSE_JOB_QUEUE_TTL_SECONDS": "JOB_QUEUE_TTL_SECONDS",
        "VA_LSE_JOB_QUEUE_LEASE_SECONDS": "JOB_QUEUE_LEASE_SECONDS",
        "VA_LSE_WORKER_CONCURRENCY": "JOB_QUEUE_WORKER_CONCURRENCY",
        "VA_LSE_WORKER_HEALTH_PORT": "JOB_QUEUE_WORKER_HEALTH_PORT",
        "VA_LSE_JOB_QUEUE_INLINE_MAX_BYTES": "JOB_QUEUE_INLINE_MAX_BYTES",
        "VA_LSE_BLOB_STORE": "BLOB_STORE_MODE",
        "VA_LSE_BLOB_DIR": "BLOB_DIR",
        "VA_LSE_EXTRACTOR": "EXTRACTOR_MODE",
        "VA_LSE_EXTRACTOR_TIMEOUT_SECONDS": "EXTRACTOR_TIMEOUT_SECONDS",
    }

    #: COMPATIBILITY.md's "Current defaults" table names the config constants directly.
    COMPATIBILITY_ROWS: tuple[str, ...] = (
        "DEFAULT_BASE_URL",
        "DEFAULT_MODEL_MAIN",
        "DEFAULT_MODEL_FAST",
        "DEFAULT_FETCH_SANDBOX_BASE_URL",
    )

    def _table(self, doc: str, header: tuple[str, str]) -> list[list[str]]:
        """The rows under one table header, as stripped cells.

        *header* is ``(first cell, a cell naming the Default column)``: both are
        matched so a table that merely repeats a variable name elsewhere in the
        document cannot be picked up, and only the contiguous rows under the
        header are returned.
        """
        lines = (Path(__file__).resolve().parent.parent / doc).read_text().splitlines()
        for index, line in enumerate(lines):
            if not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if cells and cells[0] == header[0] and header[1] in cells[1:]:
                rows: list[list[str]] = []
                for row_line in lines[index + 2 :]:  # skip the header and its separator
                    if not row_line.startswith("|"):
                        break
                    rows.append([c.strip() for c in row_line.strip().strip("|").split("|")])
                return rows
        self.fail(f"{doc}: no {header[0]} table with a {header[1]!r} column")

    def test_deployment_default_column_matches_config(self) -> None:
        rows = {row[0]: row for row in self._table("DEPLOYMENT.md", ("Variable", "Default"))}
        for env_name, attr in sorted(self.DEPLOYMENT_ROWS.items()):
            with self.subTest(env=env_name):
                row = rows.get(f"`{env_name}`")
                self.assertIsNotNone(row, f"{env_name} is missing from DEPLOYMENT.md's table")
                expected = _literal(getattr(config, attr))
                self.assertIn(
                    f"`{expected}`",
                    row[2],
                    f"DEPLOYMENT.md's default for {env_name} is {row[2]!r}, "
                    f"but config.{attr} ships {expected!r}",
                )

    def test_compatibility_current_defaults_match_config(self) -> None:
        rows = {
            row[0]: row
            for row in self._table("COMPATIBILITY.md", ("Setting", "Default value"))
        }
        for name in self.COMPATIBILITY_ROWS:
            with self.subTest(setting=name):
                row = rows.get(f"`{name}`")
                self.assertIsNotNone(row, f"{name} is missing from COMPATIBILITY.md's table")
                expected = _literal(getattr(config, name))
                self.assertIn(
                    f"`{expected}`",
                    row[1],
                    f"COMPATIBILITY.md states {name} = {row[1]!r}, but config ships {expected!r}",
                )

    def test_compatibility_marks_the_shipped_defaults_as_default(self) -> None:
        """The rows carrying the shipped values must be the ones marked "(default)".

        Derived from config rather than naming a provider, so if the default ever
        moves to another endpoint the marker has to move with it.
        """
        lines = (Path(__file__).resolve().parent.parent / "COMPATIBILITY.md").read_text().splitlines()
        rows = [
            [cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in lines
            if line.startswith("|") and line.count("|") >= 3
        ]
        endpoint_rows = [row for row in rows if f"`{config.DEFAULT_BASE_URL}`" in " ".join(row)]
        self.assertTrue(
            endpoint_rows,
            f"{config.DEFAULT_BASE_URL} appears in no COMPATIBILITY.md table",
        )
        self.assertTrue(
            any("(default)" in " ".join(row) for row in endpoint_rows),
            "no COMPATIBILITY.md row marks the shipped default endpoint as (default)",
        )
        for model in (config.DEFAULT_MODEL_MAIN, config.DEFAULT_MODEL_FAST):
            with self.subTest(model=model):
                self.assertTrue(
                    any(f"`{model}` (default)" in " ".join(row) for row in rows),
                    f"no COMPATIBILITY.md table row marks {model} as the shipped default",
                )


if __name__ == "__main__":
    unittest.main()
