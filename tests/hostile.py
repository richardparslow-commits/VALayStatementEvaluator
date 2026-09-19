"""The hostile configuration the suite must ignore — defined once, used twice.

Two things have to agree on what "hostile" means: the canaries in
``tests/test_hermetic.py``, which prove the harness resists a hostile
*subprocess* environment, and the CI job that runs the whole suite that way. Two
copies of this list would drift, and a drifted list fails **open** — the job would
keep passing while testing less than it claims. So the fixtures live here and both
callers import them.

Reproduce the CI job locally:

    python -m tests.hostile

That writes a hostile ``.streamlit/secrets.toml`` (git-ignored, and removed again
on the way out so it cannot mislead a later ``streamlit run``), checks its own
fixtures are *live* — a child that skips the harness must see every one of them,
or this would be a green run that tests nothing — and then runs the suite with
the environment applied, streaming output and passing the exit code through.

Why the environment and the secrets file, and not a hostile ``.env`` as well:
``app/config.py`` is the only caller of ``load_dotenv``, and ``tests/hermetic.py``
replaces that function outright, so whether a ``.env`` exists cannot reach the
suite either way — a fixture there would exercise nothing. The other routes are
live ones: an exported variable is readable by any test that asks, Streamlit's
secrets file *writes into* ``os.environ`` as it is parsed (which is why it needs a
file rather than a variable), and Streamlit's machine-scoped
``~/.streamlit/config.toml`` is parsed on the first config read and decides a long
tail of runtime behaviour — how much of an error is rendered, what is logged.

That last fixture goes in ``~`` rather than the working directory on purpose: the
repository's own ``.streamlit/config.toml`` is a tracked file and part of the
deployment, so overwriting it would test the wrong thing (and dirty the tree).

The scale simulation is deliberately not part of this run: ``scripts/`` is an
operator tool, not a test, and reading configuration as configuration is correct
behaviour there.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SECRETS_FILE = PROJECT_ROOT / ".streamlit" / "secrets.toml"
BACKUP_SUFFIX = ".hostile-runner-backup"

#: ``~/.streamlit`` — where Streamlit looks for the machine-scoped config file.
def global_config_file(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".streamlit" / "config.toml"

#: Every configurable knob at a non-default value. Each one changes behaviour the
#: suite asserts a *default* for; with the harness removed this set produces 13
#: failures and 11 errors across eight modules. Values are deliberately blunt
#: (1, 0.1, 16 bytes) rather than plausible — the point is to be unmistakable.
HOSTILE_ENVIRONMENT: dict[str, str] = {
    # limits and sizes
    "VA_LSE_MAX_RECORD_PAGES": "5",
    "VA_LSE_RECORD_SIZE_WARN_PAGES": "1",
    "VA_LSE_MAX_DIGEST_FACTS": "3",
    "VA_LSE_MAX_UPLOAD_BYTES": "1024",
    "VA_LSE_MAX_TOTAL_UPLOAD_BYTES": "2048",
    "VA_LSE_DIGEST_CHUNK_CHARS": "200",
    "VA_LSE_DOCUMENT_BLOCK_CHARS": "100",
    "VA_LSE_PARAGRAPH_MAX_CHARS": "50",
    "VA_LSE_UNDATED_FACT_BATCH_SIZE": "2",
    "VA_LSE_DISK_MIN_FREE_BYTES": "1000000000000000000",
    "VA_LSE_CREDIT_QUOTA": "7",
    "VA_LSE_CREDITS_PER_1M_MAIN": "999",
    "VA_LSE_CREDITS_PER_1M_FAST": "999",
    # extraction quality knobs, all at values that should change what a test sees
    "VA_LSE_DUPLICATE_PAGE_SIMILARITY": "0.1",
    "VA_LSE_EVIDENCE_WEAK_OVERLAP": "0.9",
    "VA_LSE_PDF_LAYOUT_EXTRACTION": "0",
    "VA_LSE_RUNNING_LINE_RATIO": "0.1",
    "VA_LSE_RUNNING_LINE_MIN_PAGES": "1",
    "VA_LSE_ZIP_MAX_MEMBERS": "1",
    "VA_LSE_ZIP_MAX_MEMBER_BYTES": "10",
    "VA_LSE_ZIP_MAX_TOTAL_UNCOMPRESSED_BYTES": "20",
    "VA_LSE_ZIP_MAX_COMPRESSION_RATIO": "1",
    # concurrency, breaker and timeouts
    "VA_LSE_RECORDS_CONCURRENCY": "7",
    "VA_LSE_CB_FAILURE_THRESHOLD": "99",
    "VA_LSE_CB_RECOVERY_SECONDS": "1",
    "VA_LSE_MAX_CONCURRENT_LLM_CALLS": "1",
    "VA_LSE_LLM_QUEUE_MAX_DEPTH": "1",
    "VA_LSE_LLM_QUEUE_TIMEOUT_SECONDS": "1",
    "VA_LSE_PIPELINE_TIMEOUT_SECONDS": "1",
    "VA_LSE_MEMORY_WARN_MB": "1",
    "VA_LSE_SHUTDOWN_GRACE_SECONDS": "1",
    "VA_LSE_LLM_CALL_TIMEOUT_SECONDS": "1",
    "VA_LSE_METRICS_SESSION_TTL_SECONDS": "1",
    "LLM_ENDPOINT_FALLBACK_TIMEOUT_SECONDS": "1",
    # log and audit streams, pointed outside the checkout so a hostile run cannot
    # be mistaken for evidence
    "VA_LSE_AUDIT_LOG_DIR": "/tmp/va-lse-hostile-audit",
    "VA_LSE_AUDIT_LOG_FILE": "hostile-audit.log",
    "VA_LSE_AUDIT_LOG_MAX_BYTES": "4096",
    "VA_LSE_AUDIT_LOG_BACKUPS": "1",
    "VA_LSE_AUDIT_RETENTION_DAYS": "1",
    "VA_LSE_AUDIT_ERROR_MESSAGES": "0",
    "VA_LSE_AUDIT_BACKUP_DESTINATION": "s3",
    "VA_LSE_AUDIT_BACKUP_S3_BUCKET": "hostile-bucket",
    "VA_LSE_RUN_LOG_DIR": "/tmp/va-lse-hostile-runs",
    "VA_LSE_RUN_LOG_MAX_BYTES": "4096",
    "VA_LSE_RUN_LOG_BACKUPS": "1",
    "VA_LSE_LOG_DIR": "/tmp/va-lse-hostile-log",
    "VA_LSE_LOG_FILE": "hostile-app.log",
    "VA_LSE_LOG_MAX_BYTES": "4096",
    "VA_LSE_LOG_BACKUPS": "1",
    "VA_LSE_LOG_LEVEL": "WARNING",
    "VA_LSE_WATCHDOG_PATH": "/tmp/va-lse-hostile-watchdog.json",
    # queue, blobs, tracing, and the advisory switches
    "VA_LSE_JOB_QUEUE": "1",
    "VA_LSE_REDIS_URL": "",
    "VA_LSE_JOB_QUEUE_PREFIX": "hostile",
    "VA_LSE_JOB_QUEUE_TTL_SECONDS": "60",
    "VA_LSE_JOB_QUEUE_LEASE_SECONDS": "1",
    "VA_LSE_JOB_QUEUE_POLL_SECONDS": "0.01",
    "VA_LSE_JOB_QUEUE_UI_POLL_SECONDS": "0.01",
    "VA_LSE_BLOB_STORE": "none",
    "VA_LSE_BLOB_DIR": "/tmp/va-lse-hostile-blobs",
    "VA_LSE_WORKER_ID": "hostile-worker",
    "VA_LSE_WORKER_CONCURRENCY": "3",
    "VA_LSE_WORKER_HEALTH_PORT": "0",
    "VA_LSE_HEALTH_PORT": "0",
    # The sandbox image pins this so a published port cannot reach the sidecar;
    # setting it here too means the whole suite is exercised with that bind
    # address rather than only discovering it in a sandbox.
    "VA_LSE_HEALTH_HOST": "127.0.0.1",
    "VA_LSE_PROFILE_RUNS": "1",
    "VA_LSE_TRACING": "1",
    "VA_LSE_TRACE_CHUNK_SPANS": "1",
    "VA_LSE_TRACE_LLM_CALLS": "1",
    "VA_LSE_TRACE_SAMPLE_RATIO": "0.5",
    "VA_LSE_TRACE_ROLE": "hostile",
    "OTEL_SERVICE_NAME": "hostile-service",
    # credentials and endpoints, as a developer's shell might carry them
    "OPENAI_API_KEY": "hostile-ambient-key",
    "OPENAI_BASE_URL": "https://hostile.invalid/v1",
    "OPENAI_API_KEY_FALLBACK": "hostile-fallback-key",
    "OPENAI_BASE_URL_FALLBACK": "https://hostile-fallback.invalid/v1",
    "LLM_MODEL_MAIN": "hostile-main",
    "LLM_MODEL_FAST": "hostile-fast",
    "LLM_MODEL_MAIN_FALLBACK": "hostile-main-fallback",
    "LLM_MODEL_FAST_FALLBACK": "hostile-fast-fallback",
    "FETCH_SANDBOX_API_KEY": "hostile-fetch-key",
    "FETCH_SANDBOX_BASE_URL": "https://hostile-fetch.invalid",
    "FETCH_SANDBOX_RECORDS_PATH": "/hostile/{patient_id}",
    "FETCH_SANDBOX_MAX_RESPONSE_BYTES": "64",
    "VA_GOV_API_BASE_URL": "https://hostile.invalid",
    "VA_GOV_MAX_RESPONSE_BYTES": "16",
    "VA_LSE_ALLOW_LOCAL_PATHS": "0",
    "AZURE_STORAGE_CONNECTION_STRING": "hostile-azure-connection-string",
}

#: A secrets file a developer could plausibly have: credentials and a model
#: override. Values are not credential-shaped on purpose — this file is committed
#: and the repo's pre-commit hook rejects provider-shaped strings.
HOSTILE_SECRETS: dict[str, str] = {
    "OPENAI_API_KEY": "hostile-secret-from-the-fixture",
    "OPENAI_BASE_URL": "https://hostile.invalid/v1",
    "LLM_MODEL_MAIN": "hostile-model-from-the-fixture",
    "LLM_MODEL_FAST": "hostile-fast-model-from-the-fixture",
}


#: A machine-scoped Streamlit config a developer could plausibly have, and one
#: ``tests/hermetic.py`` must make invisible. Every key in the *signature* below is
#: deliberately one the repository's own ``.streamlit/config.toml`` does **not**
#: pin — otherwise the committed file would win over this one and the fixture would
#: prove nothing. ``logger.level`` is included in the file but not the signature for
#: exactly that reason.
HOSTILE_CONFIG: dict[str, dict[str, object]] = {
    "server": {"enableStaticServing": True},
    "client": {"showSidebarNavigation": False, "showErrorDetails": "none"},
    "logger": {"level": "debug"},
}

#: What a child that skips the harness must report, and a protected one must not.
#: These are the unpinned keys above, so they can only have come from this file.
#:
#: ``server.port`` is deliberately **not** one of them, although it is the most
#: obvious knob to reach for. Streamlit marks an install as ``developmentMode``
#: whenever the package is not under a ``site-packages`` directory, and 1.64's
#: ``_check_conflicts`` then *raises* on a config that sets ``server.port`` — so a
#: fixture setting it would fail for a developer whose Streamlit is a source
#: checkout (dev mode is also what makes ``logger.level`` default to ``debug``),
#: for a reason that has nothing to do with hermeticity. These three keys carry no
#: such validation.
HOSTILE_CONFIG_SIGNATURE: dict[str, object] = {
    "server.enableStaticServing": True,
    "client.showSidebarNavigation": False,
    "client.showErrorDetails": "none",
}


def config_toml_text() -> str:
    """The fixture as TOML, built from the mapping so the two cannot drift."""
    blocks = [
        "\n".join(
            [f"[{section}]"]
            + [f"{key} = {json.dumps(value)}" for key, value in options.items()]
        )
        for section, options in HOSTILE_CONFIG.items()
    ]
    return "\n".join(blocks) + "\n"


def secrets_toml_text() -> str:
    """The fixture as TOML: one ``key = "value"`` line per entry.

    Built from the mapping rather than written out, so this file contains no
    literal credential assignment for the pre-commit hook to reject.
    """
    return "".join(
        f"{name} = {json.dumps(value)}\n" for name, value in HOSTILE_SECRETS.items()
    )


def write_hostile_secrets_file(directory: Path = PROJECT_ROOT) -> Path:
    """Write the fixture under *directory*; return the path written."""
    path = directory / ".streamlit" / "secrets.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(secrets_toml_text(), encoding="utf-8")
    return path


def write_hostile_config_file(directory: Path) -> Path:
    """Write the config fixture under *directory*/.streamlit; return the path.

    Pass ``~`` for the machine-scoped file Streamlit reads, or the working
    directory for the per-project one — the same fixture covers both, which is
    what makes the two tests in ``tests/test_hermetic.py`` check the same thing.
    """
    path = directory / ".streamlit" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config_toml_text(), encoding="utf-8")
    return path


def hostile_child_environment() -> dict[str, str]:
    """The current environment plus every hostile knob."""
    return {**os.environ, **HOSTILE_ENVIRONMENT}


class _installed_file:
    """Put fixture *text* at *path*, and give back whatever was there before.

    Both machines in this file hold something a developer would be sorry to lose
    — ``.streamlit/secrets.toml`` is a credential store, and ``~/.streamlit/
    config.toml`` governs every Streamlit app they run — so overwriting either one
    (the obvious implementation) is not acceptable. This moves it aside for the
    run and restores it afterwards; the fixture itself is removed on the way out,
    because leaving it behind would silently change the app the next time someone
    ran ``streamlit run run_app.py``.
    """

    def __init__(self, path: Path, text: str) -> None:
        self._path = path
        self._text = text
        self._backup: Path | None = None

    def __enter__(self) -> Path:
        if self._path.exists():
            self._backup = self._path.with_name(self._path.name + BACKUP_SUFFIX)
            if self._backup.exists():  # pragma: no cover - a stale run
                raise SystemExit(
                    f"{self._backup} already exists — a previous hostile run did not "
                    "finish cleanly. Restore it by hand, then run again."
                )
            self._path.replace(self._backup)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(self._text, encoding="utf-8")
        return self._path

    def __exit__(self, *_exc: object) -> None:
        self._path.unlink(missing_ok=True)
        if self._backup is not None:
            self._backup.replace(self._path)


class installed_secrets_file(_installed_file):
    """The hostile ``.streamlit/secrets.toml``, in the project directory."""

    def __init__(self, directory: Path = PROJECT_ROOT) -> None:
        super().__init__(
            directory / ".streamlit" / "secrets.toml", secrets_toml_text()
        )


class installed_global_config(_installed_file):
    """The hostile machine-scoped ``~/.streamlit/config.toml``.

    In ``~`` rather than the working directory because that is the file the
    repository does not ship: its own ``.streamlit/config.toml`` is tracked and
    part of the deployment, so the ambient one is the one worth simulating.
    """

    def __init__(self, home: Path | None = None) -> None:
        super().__init__(global_config_file(home), config_toml_text())


def require_effective_fixtures() -> str:
    """Prove the fixtures reach a child that does *not* use the harness.

    Returns a one-line summary for the log. Raises SystemExit when the child sees
    nothing: a hostile run whose inputs are not live is a green job that tests
    nothing, which is the one outcome worse than a red one.
    """
    program = (
        "import json, os, sys\n"
        # Read the environment *first*. Importing Streamlit parses the secrets file
        # and promotes its values into os.environ, overwriting the hostile
        # variables — so reading them afterwards makes it look as though they never
        # arrived. (That overwrite is itself the reason the secrets route needs a
        # file fixture: a file can outrank the shell.)
        "raw = {name: os.environ.get(name) for name in sys.argv[1:]}\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        "import streamlit\n"
        "try:\n"
        "    parsed = sorted(streamlit.secrets)\n"
        "except Exception as exc:\n"
        "    parsed = [type(exc).__name__]\n"
        "promoted = sorted(n for n in raw if os.environ.get(n) != raw[n])\n"
        "import streamlit.config as streamlit_config\n"
        "streamlit_config.get_config_options()\n"
        "options = {\n"
        f"    name: streamlit_config.get_option(name)\n"
        f"    for name in {sorted(HOSTILE_CONFIG_SIGNATURE)!r}\n"
        "}\n"
        "print(json.dumps({\n"
        "    'env': raw,\n"
        "    'secrets': parsed,\n"
        "    'promoted': promoted,\n"
        "    'options': options,\n"
        "}))\n"
    )
    process = subprocess.run(
        [sys.executable, "-c", program, *sorted(HOSTILE_ENVIRONMENT)],
        cwd=str(PROJECT_ROOT),
        env=hostile_child_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise SystemExit(
            "could not inspect the fixtures with a child process:\n"
            f"{process.stdout}\n{process.stderr}"
        )
    report = json.loads(process.stdout.strip().splitlines()[-1])

    missing = [
        name
        for name, value in HOSTILE_ENVIRONMENT.items()
        if report["env"].get(name) != value
    ]
    if missing:
        raise SystemExit(
            "these hostile variables did not reach the child, so the run below "
            "would not test them: " + ", ".join(sorted(missing))
        )
    if sorted(HOSTILE_SECRETS) != report["secrets"]:
        raise SystemExit(
            "the hostile secrets file was not readable by a child that skips the "
            f"harness (it saw {report['secrets']}), so the run below would not "
            "test the secrets route"
        )
    wrong = {
        name: report["options"].get(name)
        for name, value in HOSTILE_CONFIG_SIGNATURE.items()
        if report["options"].get(name) != value
    }
    if wrong:
        raise SystemExit(
            "the hostile config file in ~/.streamlit did not decide these options "
            f"in a child that skips the harness: {wrong!r} — the run below would "
            "not test the config route"
        )
    promoted = report["promoted"]
    detail = (
        f"; parsing the file overrode {len(promoted)} of those variables in the "
        "child, so the file is authoritative"
        if promoted
        else ""
    )
    return (
        f"{len(HOSTILE_ENVIRONMENT)} variables, {len(HOSTILE_SECRETS)} secret "
        f"names and {len(HOSTILE_CONFIG_SIGNATURE)} config options are live "
        f"without the harness{detail}"
    )


def run_suite(argv: list[str]) -> int:
    """Run the offline suite with the hostile environment applied."""
    return subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", *argv],
        cwd=str(PROJECT_ROOT),
        env=hostile_child_environment(),
        check=False,
    ).returncode


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    # flush=True throughout: redirected to a pipe — which is what CI does — Python
    # block-buffers this process, while the child writes straight through, so
    # without it the header explaining the run appears after the run.
    print(f"hostile environment: {len(HOSTILE_ENVIRONMENT)} variables", flush=True)
    with installed_secrets_file() as secrets_path, installed_global_config() as config_path:
        print(
            "hostile secrets file: "
            f"{secrets_path.relative_to(PROJECT_ROOT)} "
            f"({', '.join(sorted(HOSTILE_SECRETS))})",
            flush=True,
        )
        print(f"hostile config file: {config_path}", flush=True)
        print(f"fixtures verified: {require_effective_fixtures()}", flush=True)
        print("running the offline suite under them…", flush=True)
        return run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
