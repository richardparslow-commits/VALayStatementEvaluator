"""The suite's hermeticity is a property, so it gets a test.

``tests/hermetic.py`` empties the ambient configuration at import — that is what
stops a developer's ``.env`` or an exported ``VA_LSE_*`` variable from deciding
what the suite sees. This file is what keeps that true, and it has four jobs:

* it pins the *derivation* (the scan that discovers which names the app reads),
  because the scan is the thing that must not silently shrink;
* it checks the environment really is empty of those names in the process, that a
  real ``.env`` file cannot get back in, and that a ``secrets.toml`` on this
  machine is invisible — including the values Streamlit would otherwise promote
  into ``os.environ`` while parsing it;
* it scans the test tree for a module that forgot the harness, or placed it after
  its app import — the one ordering mistake that quietly defeats the whole idea;
* it re-runs the tests that provably depend on ambient configuration in a
  subprocess with a hostile environment. That last one is the guard that can
  fail: delete the scrub and it goes red on any machine, not just on one that
  happens to have the variable exported; and
* it checks the workflow still runs the whole suite that way, because a gate that
  can be deleted without a test noticing is not a gate.

Measured before the harness existed, by running the suite under the same hostile
environment this file replays: 13 failures and 11 errors across eight modules,
every one of them for a reason unrelated to the code under test.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))
from tests import devlayout, harness_imports, hermetic, hostile  # noqa: E402

#: Names the app reads, spread across the modules that read them. A derivation
#: that stops seeing these has stopped seeing whole files.
REQUIRED_NAMES = frozenset(
    {
        # app/config.py — module-level constants, the frozen-at-import case
        "VA_LSE_PIPELINE_TIMEOUT_SECONDS",
        "VA_LSE_MEMORY_WARN_MB",
        "VA_LSE_AUDIT_LOG_BACKUPS",
        "VA_LSE_ZIP_MAX_MEMBERS",
        "VA_LSE_TRACING",
        "VA_LSE_REDIS_URL",
        "OPENAI_API_KEY",
        "LLM_MODEL_MAIN",
        "FETCH_SANDBOX_BASE_URL",
        # read outside config.py, so a config-only scan would miss them
        "VA_LSE_ALLOW_LOCAL_PATHS",
        "VA_LSE_RUN_LOG_DISABLED",
        "VA_LSE_LOG_LEVEL",
        "VA_LSE_WATCHDOG_PATH",
        "VA_GOV_API_BASE_URL",
        "AZURE_STORAGE_CONNECTION_STRING",
    }
)

#: The floor on the derivation. The app reads >120 names; a regex that stopped
#: matching would drop this to single digits, and the only symptom would be tests
#: quietly reading the machine again — silence, which is why it is asserted.
MINIMUM_DISCOVERED_NAMES = 100

#: The hostile fixtures live in one place — ``tests/hostile.py`` — because the CI
#: job runs the *whole* suite under them. Two copies of that list would drift, and
#: a drifted copy fails open: the job would keep passing while testing less than it
#: claims. Everything below reads them from there.

#: Real tests that read ambient configuration and are *supposed* to be reading
#: code defaults. Each failed under ``hostile.HOSTILE_ENVIRONMENT`` before the
#: harness existed.
CANARY_TESTS = (
    "tests.test_pipeline_guard.TestConfigWiring",
    "tests.test_audit_backup.TestRotatedFiles",
    "tests.test_llm_failover.TestFallbackConfiguration",
    "tests.test_ingest_quality.TestArchiveUploads",
)

#: Ways a module could ask Streamlit for an option's value. Compiled once so the
#: scan and its self-tests cannot disagree about what is being matched.
OPTION_READ_PATTERNS = tuple(re.compile(pattern) for pattern in (
    r"\bget_option\s*\(",
    r"\bset_option\s*\(",
    r"\bst\.config\b",
    r"\bstreamlit\.config\b",
))


def option_read_offence(name: str | Path, source: str) -> list[str]:
    """Lines where *source* reads a Streamlit config option, as ``name:line``."""
    return [
        f"{name}:{number}: {line.strip()}"
        for number, line in enumerate(source.splitlines(), start=1)
        if any(pattern.search(line) for pattern in OPTION_READ_PATTERNS)
    ]


class TestTheScanFindsTheConfigSurface(unittest.TestCase):
    """The derivation is the load-bearing part, so it is pinned, not assumed."""

    def test_every_name_the_app_reads_is_discovered(self) -> None:
        discovered = hermetic.configurable_env_names()
        self.assertEqual(
            sorted(REQUIRED_NAMES - discovered),
            [],
            "these names are read by the app but the scan does not see them, so an "
            "ambient value would survive into the suite",
        )

    def test_the_scan_does_not_silently_shrink(self) -> None:
        discovered = hermetic.configurable_env_names()
        self.assertGreaterEqual(
            len(discovered),
            MINIMUM_DISCOVERED_NAMES,
            f"only {len(discovered)} names discovered — the scan has stopped seeing "
            "part of the app, and nothing else would notice",
        )

    def test_a_runners_opt_in_is_not_stripped(self) -> None:
        """VA_LSE_TEST_* is how a runner opts a test into real ambient setup."""
        self.assertNotIn("VA_LSE_TEST_REDIS_URL", hermetic.protected_names())
        self.assertNotIn("VA_LSE_TEST_REDIS_URL", hermetic.configurable_env_names())

    def test_process_essential_names_are_never_stripped(self) -> None:
        protected = hermetic.protected_names()
        self.assertEqual(protected & hermetic.NEVER_STRIP, frozenset())
        for name in ("PATH", "HOME"):
            with self.subTest(name=name):
                self.assertNotIn(name, protected)


class TestTheAmbientConfigurationIsGone(unittest.TestCase):
    def test_no_protected_name_survives_in_this_process(self) -> None:
        present = sorted(name for name in hermetic.protected_names() if name in os.environ)
        self.assertEqual(
            present,
            [],
            "these are readable by any test that runs now, so the suite is still "
            "reading the machine it runs on",
        )

    def test_no_env_file_key_survives_in_this_process(self) -> None:
        present = sorted(key for key in hermetic.env_file_keys() if key in os.environ)
        self.assertEqual(present, [], f"the project .env reached the suite: {present}")

    def test_the_dotenv_loader_is_neutralised(self) -> None:
        import dotenv

        self.assertTrue(hermetic.DOTENV_NEUTRALIZED)
        self.assertIs(dotenv.load_dotenv, hermetic._ignore_dotenv)

    def test_a_real_env_file_cannot_seed_the_process(self) -> None:
        """Load a genuine .env and prove it lands nowhere."""
        import dotenv

        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("VA_LSE_PROBE_FROM_ENV_FILE=1\n", encoding="utf-8")
            loaded = dotenv.load_dotenv(env_file)
        self.assertFalse(loaded, "load_dotenv must report that it loaded nothing")
        self.assertNotIn("VA_LSE_PROBE_FROM_ENV_FILE", os.environ)

    def test_stripping_removes_a_hostile_value(self) -> None:
        """The scrub itself, driven directly rather than through import order."""
        with patch.dict(os.environ, {"VA_LSE_MEMORY_WARN_MB": "1"}, clear=False):
            removed = hermetic.strip_ambient_config()
            self.assertIn("VA_LSE_MEMORY_WARN_MB", removed)
            self.assertNotIn("VA_LSE_MEMORY_WARN_MB", os.environ)

    def test_the_code_default_is_what_config_holds(self) -> None:
        from app import config

        self.assertEqual(config.MEMORY_WARN_MB, 500)
        self.assertEqual(config.PIPELINE_TIMEOUT_SECONDS, 1800)
        self.assertFalse(config.TRACING_ENABLED)


class TestTheSecretsManagerCannotReachTheSuite(unittest.TestCase):
    """The third ambient source: a developer's ``.streamlit/secrets.toml``.

    Not merely readable — Streamlit **promotes** every string secret into
    ``os.environ`` as it parses the file, so a secrets file can repopulate a name
    after the harness has already swept it. These checks cover the promotion and
    not only the read, which is why one of them watches the environment after
    forcing the manager to be consulted.
    """

    def test_streamlit_is_told_there_is_no_secrets_file(self) -> None:
        import streamlit.config as streamlit_config

        self.assertTrue(hermetic.STREAMLIT_SECRETS_NEUTRALIZED)
        self.assertEqual(
            streamlit_config.get_option(hermetic.SECRETS_FILES_OPTION),
            [],
            "Streamlit is still looking for a secrets.toml, so a developer who has "
            "one runs a different app from CI",
        )

    def test_the_app_resolves_no_secret(self) -> None:
        from app import config

        with patch.object(config, "_SECRETS_CACHE", None):
            self.assertEqual(config.streamlit_secrets(), {})

    def test_reading_the_secrets_manager_promotes_nothing_into_the_environment(
        self,
    ) -> None:
        """The subtle half: Streamlit writes string secrets into ``os.environ``."""
        from app import config

        with patch.object(config, "_SECRETS_CACHE", None):
            config.streamlit_secrets()
        promoted = sorted(
            name for name in hermetic.protected_names() if name in os.environ
        )
        self.assertEqual(
            promoted,
            [],
            "reading the secrets manager put these back into the environment after "
            "the harness had removed them, so a later test can still see them",
        )

    def test_a_secrets_file_in_the_working_directory_is_invisible(self) -> None:
        """The guard that can fail: a real file, in the place Streamlit looks.

        Run against a child that does *not* import the harness first, so the
        fixture is proved hostile rather than assumed to be; then against one that
        does.
        """
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            # Exactly the fixture CI installs, written by the same function.
            hostile.write_hostile_secrets_file(project)
            # One of the fixture's names: if it stops being credential-shaped, this
            # control fails and says so rather than letting the guard pass quietly.
            probe_key = "OPENAI_API_KEY"

            control = self._run_probe(project, protect=False)
            self.assertIn(
                probe_key,
                control.stdout,
                "the fixture is not actually hostile, so this test could not fail:"
                f"\n{control.stdout}\n{control.stderr}",
            )

            protected = self._run_probe(project, protect=True)
            self.assertEqual(
                protected.returncode,
                0,
                "a secrets.toml in the working directory reached the suite:\n"
                f"--- stdout ---\n{protected.stdout[-2000:]}\n"
                f"--- stderr ---\n{protected.stderr[-2000:]}",
            )
            self.assertIn("no-secrets-file", protected.stdout)

    @staticmethod
    def _run_probe(project: Path, *, protect: bool) -> subprocess.CompletedProcess[str]:
        """Read Streamlit's secrets in a child whose working directory holds one.

        The working directory is what makes the fixture findable: Streamlit's
        default candidates include ``<cwd>/.streamlit/secrets.toml``.
        """
        harness = "from tests import hermetic\n" if protect else ""
        program = (
            "import os, sys\n"
            f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
            f"sys.path.insert(0, {str(TESTS_DIR)!r})\n"
            + harness
            + "import streamlit\n"
            "try:\n"
            "    data = streamlit.secrets.to_dict()\n"
            "except Exception as exc:\n"
            "    print('no-secrets-file:', type(exc).__name__)\n"
            "    sys.exit(0)\n"
            "print('secrets-visible:', sorted(data))\n"
            "print('promoted:', sorted(k for k in data if k in os.environ))\n"
            "sys.exit(1 if data else 0)\n"
        )
        return subprocess.run(
            [sys.executable, "-c", program],
            cwd=str(project),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )


class TestTheConfigFilesCannotReachTheSuite(unittest.TestCase):
    """The fourth ambient source: Streamlit's *configuration* files.

    ``.streamlit/config.toml`` is read from ``~`` and from the **working
    directory**, so the file that decides the app's runtime options is on the
    machine, not in the repository. Measured before the fix, with the harness as
    it stood: a hostile ``~/.streamlit/config.toml`` set ``server.port`` to 9999
    and ``client.showErrorDetails`` to ``"none"`` in this very process, while the
    repository's committed hardening stayed invisible — and starting the tests
    from another directory replaced the committed file entirely
    (``server.enableXsrfProtection`` came back ``manually_set=False``).

    No test currently asserts an option value, so nothing was *failing*: the suite
    was simply reading a different configuration than the deployed one, which is
    the kind of difference that shows up later as a result nobody can explain.
    The fix pins the session to the repository's file. Both checks below are
    control-backed — a child that skips the harness must actually see the hostile
    values, or the guard could not fail.
    """

    #: Reported by the probe: what the fixture sets, plus two keys the repository's
    #: committed file pins (which must survive).
    OPTIONS = tuple(hostile.HOSTILE_CONFIG_SIGNATURE) + (
        "server.enableXsrfProtection",
        "server.maxUploadSize",
    )

    def test_the_repository_config_file_is_the_one_that_applies(self) -> None:
        import streamlit.config as streamlit_config

        self.assertTrue(hermetic.STREAMLIT_CONFIG_NEUTRALIZED)
        self.assertTrue(
            hermetic.REPO_CONFIG_FILE.exists(),
            f"{hermetic.REPO_CONFIG_FILE} is the session's only config source, so "
            "without it the suite silently runs Streamlit's bare defaults",
        )
        self.assertEqual(
            streamlit_config.get_config_files(hermetic.CONFIG_FILE_NAME),
            [str(hermetic.REPO_CONFIG_FILE)],
            "the session must read the repository's config file and no other",
        )
        # Read from a file (rather than merely defaulted to) is what proves the
        # committed file is in play: XSRF protection defaults to True anyway.
        self.assertTrue(streamlit_config.is_manually_set("server.enableXsrfProtection"))
        self.assertEqual(streamlit_config.get_option("server.maxUploadSize"), 50)

    def test_a_machine_config_file_cannot_reach_the_session(self) -> None:
        """A hostile ``~/.streamlit/config.toml``, against a control."""
        with tempfile.TemporaryDirectory() as home:
            hostile_file = hostile.write_hostile_config_file(Path(home))
            self.assertTrue(hostile_file.exists())

            control = self._probe(home=Path(home), protect=False)
            self.assertEqual(
                {
                    name: control["values"][name]
                    for name in hostile.HOSTILE_CONFIG_SIGNATURE
                },
                dict(hostile.HOSTILE_CONFIG_SIGNATURE),
                "the fixture is not actually hostile, so this test could not fail",
            )

            protected = self._probe(home=Path(home), protect=True)
            self.assertEqual(
                sorted(
                    name
                    for name in hostile.HOSTILE_CONFIG_SIGNATURE
                    if protected["manually_set"][name]
                ),
                [],
                "a config.toml in ~ decided this session's options, so the suite "
                f"runs a different app from the deployed one: {protected}",
            )
            self.assert_pinned_repo_config_applies(protected)

    def test_a_config_file_in_the_working_directory_cannot_reach_the_session(
        self,
    ) -> None:
        """The same fixture under ``$CWD``, which is the other place it is read.

        This is the determinism half: the suite must not depend on the directory
        it was launched from.
        """
        with tempfile.TemporaryDirectory() as cwd:
            hostile.write_hostile_config_file(Path(cwd))

            control = self._probe(cwd=Path(cwd), protect=False)
            self.assertEqual(
                {
                    name: control["values"][name]
                    for name in hostile.HOSTILE_CONFIG_SIGNATURE
                },
                dict(hostile.HOSTILE_CONFIG_SIGNATURE),
                "the fixture is not actually hostile, so this test could not fail",
            )

            protected = self._probe(cwd=Path(cwd), protect=True)
            self.assertEqual(
                sorted(
                    name
                    for name in hostile.HOSTILE_CONFIG_SIGNATURE
                    if protected["manually_set"][name]
                ),
                [],
                "a config.toml in the working directory decided this session's "
                f"options: {protected}",
            )
            self.assert_pinned_repo_config_applies(protected)

    def test_a_sensitive_options_environment_variable_is_stripped(self) -> None:
        """The other route into the option table, and the only live one.

        ``STREAMLIT_*`` variables are honoured for exactly the options Streamlit
        marks ``sensitive``, so the derivation reads Streamlit's own table rather
        than a hand-written pair that would go stale on upgrade.
        """
        names = set(hermetic.SENSITIVE_ENV_NAMES)
        self.assertTrue(names, "the derivation found no sensitive options at all")
        self.assertLess(
            len(names),
            10,
            "environment variables are read for a handful of options, not many — a "
            "number this size suggests the derivation is reading the wrong field",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertTrue(name.startswith("STREAMLIT_"))
        with patch.dict(os.environ, {name: "hostile" for name in names}, clear=False):
            removed = hermetic.strip_ambient_config()
        self.assertEqual(sorted(names - set(removed)), [])
        self.assertEqual(sorted(name for name in names if name in os.environ), [])

    def test_the_committed_config_sets_no_port(self) -> None:
        """A port here is fatal on a dev-mode install, and this file is the one
        source the session is pinned to.

        Streamlit raises at config-parse time when a port is set and it considers
        itself a development install — its own test is whether the package sits
        under a ``site-packages`` directory, so a source checkout, an editable
        install, or a vendored one qualifies — and nothing warns first. The session
        reads this file, so a port added here would break the suite on such a machine
        before a single test ran, and the app would fail to start for the same reason.
        Reported upstream as streamlit/streamlit#17031.
        """
        data = tomllib.loads(hermetic.REPO_CONFIG_FILE.read_text(encoding="utf-8"))
        self.assertNotIn(
            "port",
            data.get("server", {}),
            "a port in the committed config makes the app fail to start on any install "
            "Streamlit treats as a development one (no `site-packages` in the package "
            "path) — the port belongs to the container's --server.port instead",
        )
        self.assertNotIn(
            "serverPort",
            data.get("browser", {}),
            "`browser.serverPort` fails the same way as `server.port` on a dev-mode "
            "install",
        )

    def test_the_committed_config_file_still_carries_the_hardening(self) -> None:
        """The session's one config source has to be worth pinning.

        ``app/main.py`` warns (non-blockingly) when these keys go missing; this is
        the same contract asserted here, because the harness now *depends* on that
        file being the deployed configuration.
        """
        text = hermetic.REPO_CONFIG_FILE.read_text(encoding="utf-8")
        for key in ("enableXsrfProtection", "toolbarMode", "maxUploadSize"):
            with self.subTest(key=key):
                self.assertIn(key, text)

    def assert_pinned_repo_config_applies(self, report: dict) -> None:
        """The other half: dropping the machine's file must not drop the repo's.

        A harness that neutralised *too* much would leave the session on Streamlit's
        bare defaults, which is a machine CI never runs — and this is what tells the
        two apart.
        """
        self.assertTrue(
            report["manually_set"]["server.enableXsrfProtection"],
            "the repository's committed config is not in play, so the session is "
            "running Streamlit's bare defaults rather than the deployed settings",
        )
        self.assertEqual(report["values"]["server.maxUploadSize"], 50)

    @staticmethod
    def _probe(
        *, cwd: Path | None = None, home: Path | None = None, protect: bool
    ) -> dict[str, object]:
        """Read this file's options and config files in a child process.

        ``manually_set`` is reported as well as the values: it says whether a *file*
        decided the option, which is the actual question — asserting a hardcoded
        default would also pass if Streamlit changed the default and the config file
        went missing.
        """
        harness = "from tests import hermetic\n" if protect else ""
        names = list(TestTheConfigFilesCannotReachTheSuite.OPTIONS)
        program = (
            "import json, sys\n"
            f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
            + harness
            + "import streamlit.config as streamlit_config\n"
            "streamlit_config.get_config_options()\n"
            f"names = {names!r}\n"
            "print(json.dumps({\n"
            "    'values': {n: streamlit_config.get_option(n) for n in names},\n"
            "    'manually_set': {n: streamlit_config.is_manually_set(n) for n in names},\n"
            "    'config_files': streamlit_config.get_config_files('config.toml'),\n"
            "}))\n"
        )
        environment = dict(os.environ)
        if home is not None:
            environment["HOME"] = str(home)
        process = subprocess.run(
            [sys.executable, "-c", program],
            cwd=str(cwd or PROJECT_ROOT),
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if process.returncode != 0:
            raise AssertionError(
                f"probe failed:\n{process.stdout}\n{process.stderr}"
            )
        return json.loads(process.stdout.strip().splitlines()[-1])


class TestStreamlitStillIgnoresNonSensitiveEnvVars(unittest.TestCase):
    """The measurement the harness's *scope* rests on, pinned so it cannot rot.

    ``strip_ambient_config`` removes the names Streamlit honours for a config
    option, and that is adequate only because the environment is consulted for
    ``sensitive`` options alone — 2 of 343 on Streamlit 1.63.0. Streamlit's own
    documentation has always claimed the wider behaviour (``STREAMLIT_SERVER_PORT``
    is its example, and ``STREAMLIT_CLIENT_SHOW_ERROR_DETAILS`` its worked
    equivalence), and upstream fixing that would be a fix that breaks this suite
    quietly: every option would become env-overridable while the harness stripped
    two names, which is the ambient dependence it exists to prevent, arriving
    through an upgrade.

    So this asserts the measured behaviour rather than the documentation. It runs
    in a child process — the variables have to be set before the config is parsed
    — and each expectation is compared against the *same* child run without them,
    so a repository config that pins these keys cannot make it cry wolf. The
    sensitive option is the control: it must pick up its variable, which proves
    the probe can see a live environment route at all.
    """

    #: A sensitive option's variable, used as the live control.
    LIVE_SECRET = "from-the-environment"

    #: (option, the variable the documentation says overrides it, a value that
    #: would be unmistakable in the result).
    DOCUMENTED_BUT_INERT = (
        ("server.port", "STREAMLIT_SERVER_PORT", "9999"),
        ("client.showErrorDetails", "STREAMLIT_CLIENT_SHOW_ERROR_DETAILS", "false"),
    )

    def test_a_documented_env_var_does_not_override_a_non_sensitive_option(
        self,
    ) -> None:
        baseline = self._probe(os.environ.copy())
        with_vars = self._probe(
            {
                **os.environ,
                "STREAMLIT_SERVER_COOKIE_SECRET": self.LIVE_SECRET,
                **{name: value for _, name, value in self.DOCUMENTED_BUT_INERT},
            }
        )

        self.assertEqual(
            with_vars["server.cookieSecret"],
            self.LIVE_SECRET,
            "no environment variable reached the config parse in this child, so "
            "this test could not observe anything and would pass vacuously",
        )
        for option, env_name, _ in self.DOCUMENTED_BUT_INERT:
            with self.subTest(option=option, env=env_name):
                self.assertEqual(
                    with_vars[option],
                    baseline[option],
                    f"{env_name} now overrides {option}. If upstream fixed the "
                    "environment route for non-sensitive options, every option is "
                    "env-overridable and tests/hermetic.py strips only "
                    f"{sorted(hermetic.SENSITIVE_ENV_NAMES)} — so the harness would "
                    "under-protect. Derive the stripped set from the live route "
                    "(see tests/hostile.py for how to check it the same way).",
                )

    @staticmethod
    def _probe(environment: dict[str, str]) -> dict[str, object]:
        program = (
            "import json, sys\n"
            f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
            # Deliberately no harness: this pins what Streamlit does, and the
            # harness is what strips the variable under test.
            "import streamlit.config as streamlit_config\n"
            "streamlit_config.get_config_options()\n"
            "print(json.dumps({\n"
            "    'server.port': streamlit_config.get_option('server.port'),\n"
            "    'client.showErrorDetails': "
            "streamlit_config.get_option('client.showErrorDetails'),\n"
            "    'server.cookieSecret': "
            "streamlit_config.get_option('server.cookieSecret'),\n"
            "}))\n"
        )
        process = subprocess.run(
            [sys.executable, "-c", program],
            cwd=str(PROJECT_ROOT),
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if process.returncode != 0:
            raise AssertionError(f"probe failed:\n{process.stdout}\n{process.stderr}")
        return json.loads(process.stdout.strip().splitlines()[-1])


class TestADevelopmentModeInstallCannotReachTheSession(unittest.TestCase):
    """The fifth source is not a setting at all: it is how Streamlit was installed.

    Streamlit derives ``global.developmentMode`` from its own package path
    (``"site-packages" not in __file__``), so a source checkout, an editable
    install, and a vendored ``--target`` install all report true. It is a fork in
    behaviour — ``logger.level`` and ``logger.messageFormat`` default differently —
    and with a port set in any config file it makes parsing raise. AppTest is where
    that becomes invisible: measured on 1.64.0, the clear message appears only on
    stderr inside a ``ScriptRunner.scriptThread`` traceback, while ``at.run()``
    raises ``KeyError: 'st.session_state has no key
    "$$STREAMLIT_INTERNAL_KEY_SCRIPT_RUN_WITHOUT_ERRORS"'`` or a bare ``AppTest
    script run timed out``, depending on timing. Since this suite's view tests are
    almost all AppTest-driven, a contributor with an editable Streamlit and a port in
    their config would see a dozen view tests fail with nothing to go on.

    Both children claim to be a development install by setting the option's value on
    the template before anything parses — which is exactly what makes a real
    ``--target`` install fail, verified against one. The control proves the fixture is
    hostile; the protected child proves the session is not. Pinning the config files
    would already stop the port from arriving, so what this pins is the *outcome*: the
    environment the suite runs in is the deployment's, whichever of the two guards is
    doing the work.
    """

    APP = 'import streamlit as st\n\nst.write("hello")\n'
    PORT_CONFIG = '[server]\nport = 9999\n'
    FAILURE = "server.port does not work when global.developmentMode is true"

    def test_the_control_child_really_fails(self) -> None:
        process = self._run_child(protect=False)
        self.assertIn(
            self.FAILURE,
            process.stdout + process.stderr,
            "a development-mode install with a port configured no longer fails, so "
            "the guard below testifies to nothing:\n"
            f"--- stdout ---\n{process.stdout}\n--- stderr ---\n{process.stderr[-2000:]}",
        )
        self.assertNotIn("APPTEST=ok", process.stdout)
        # The symptom a contributor actually sees, recorded so the guard's reason is
        # legible rather than folklore.
        self.assertIn("APPTEST=", process.stdout)

    def test_the_session_runs_as_the_deployment_does(self) -> None:
        process = self._run_child(protect=True)
        self.assertIn(
            "APPTEST=ok",
            process.stdout,
            "under a development-mode install the AppTest path failed inside the "
            "test session:\n"
            f"--- stdout ---\n{process.stdout}\n--- stderr ---\n{process.stderr[-2000:]}",
        )
        self.assertIn("exceptions=0", process.stdout)
        self.assertIn(
            "DEVMODE=False",
            process.stdout,
            "the session is still reporting the install layout rather than the "
            "deployment's, so a dev-mode default can still reach a test",
        )

    def _run_child(self, *, protect: bool) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / ".streamlit").mkdir()
            (project / ".streamlit" / "config.toml").write_text(
                self.PORT_CONFIG, encoding="utf-8"
            )
            (project / "app.py").write_text(self.APP, encoding="utf-8")
            # Order matters twice over. The mode is claimed *before* any parse (the
            # default function would have set it the same way on such an install),
            # and nothing reads an option before AppTest — because a failed parse
            # leaves the cache behind, so an earlier read would make the control
            # child succeed and this test vacuous.
            program = (
                "import sys\n"
                f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
                "import streamlit.config as config\n"
                f"config._config_options_template[{hermetic.DEVELOPMENT_MODE_OPTION!r}]"
                ".set_value(True, 'child')\n"
                + ("from tests import hermetic\n" if protect else "")
                + "try:\n"
                "    from streamlit.testing.v1 import AppTest\n"
                "    at = AppTest.from_file('app.py', default_timeout=15)\n"
                "    at.run()\n"
                "    print('APPTEST=ok exceptions=%d markdown=%d'\n"
                "          % (len(at.exception), len(at.markdown)))\n"
                "except Exception as exc:\n"
                "    print('APPTEST=%s: %s' % (type(exc).__name__, str(exc)[:90]))\n"
                "print('DEVMODE=%r' % config.get_option('global.developmentMode'))\n"
            )
            return subprocess.run(
                [sys.executable, "-c", program],
                cwd=str(project),
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )


class TestTheDevelopmentLayoutRunnerRefusesAVacuousRun(unittest.TestCase):
    """The gate for the gate: a runner that cannot tell it is in the wrong install.

    ``tests/devlayout.py`` is what the ``dev-layout`` CI job runs, and it is the one
    place the development-mode pin meets a *real* non-``site-packages`` install — the
    guard above simulates the layout, because a test process cannot reinstall
    Streamlit. That makes its precondition the load-bearing part: if it ran the suite
    anyway, the job would report green while exercising none of what it exists for,
    the same shape as a hostile run whose fixtures were not live.

    The decisions are therefore pinned here against reports rather than against
    whatever Streamlit happens to be installed for whoever runs this file (the two
    real cases — a development layout and the deployment's — are the CI jobs', one
    each). The last test runs the real probes, because a child program that never
    imports the harness would compare one process against itself.
    """

    INSTALL = {"version": "1.63.0", "file": "/tmp/dev-layout/streamlit/__init__.py"}

    def _report(self, dev_mode: object) -> dict:
        return {**self.INSTALL, "dev_mode": dev_mode}

    def test_a_development_layout_passes_the_first_check(self) -> None:
        summary = devlayout.assert_development_layout(self._report(True))
        self.assertIn(str(self.INSTALL["file"]), summary)

    def test_a_normal_install_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            devlayout.assert_development_layout(self._report(False))
        message = str(caught.exception)
        self.assertIn("not a development layout", message)
        self.assertIn(
            "PYTHONPATH",
            message,
            "the refusal has to say how to get an install that qualifies, or it is "
            "just a red build",
        )

    def test_the_pin_must_hold(self) -> None:
        devlayout.assert_pin_holds(self._report(False))
        with self.assertRaises(SystemExit) as caught:
            devlayout.assert_pin_holds(self._report(True))
        self.assertIn("did not pin", str(caught.exception))

    def test_the_two_probes_differ_only_by_the_harness(self) -> None:
        """Otherwise the contrast could be two different Streamlists."""
        without = devlayout.child_program(protect=False)
        with_harness = devlayout.child_program(protect=True)
        self.assertNotEqual(without, with_harness)
        self.assertNotIn("hermetic", without)
        for line in without.splitlines():
            self.assertIn(
                line,
                with_harness,
                "the protected probe must be the unprotected one plus the harness "
                "import, so the difference between them is the pin and nothing else",
            )

    def test_the_real_probes_measure_this_suite_s_install(self) -> None:
        import streamlit

        unprotected = devlayout.layout_report(protect=False)
        protected = devlayout.layout_report(protect=True)
        self.assertEqual(
            unprotected["file"],
            streamlit.__file__,
            "the probe measured a different Streamlit than the one this suite imports",
        )
        # Raises when the two children are not the same install.
        devlayout.same_install(unprotected, protected)
        self.assertFalse(
            protected["dev_mode"],
            "the pin did not hold with the harness imported, in the same install "
            f"that reports {unprotected['dev_mode']!r} without it",
        )


class TestTheAppReadsNoStreamlitConfigOption(unittest.TestCase):
    """Why the file above can be pinned at all: nothing in ``app/`` asks for an
    option.

    The app's hardening check reads the committed file as *text* on purpose —
    Streamlit exposes no way to ask which source a value came from, so asking for
    the value would answer a different question ("is XSRF on in this process?",
    which a machine's own config file can answer wrongly) instead of the one that
    matters ("does the deployment ship the hardening?"). This keeps that
    distinction from being lost by a future refactor that reaches for
    ``get_option`` — which would make the app's security posture depend on the
    environment it happens to start in.
    """

    def test_no_app_module_reads_a_streamlit_option(self) -> None:
        offenders: list[str] = []
        for path in sorted((PROJECT_ROOT / "app").rglob("*.py")):
            found = option_read_offence(
                path.relative_to(PROJECT_ROOT), path.read_text(encoding="utf-8")
            )
            offenders.extend(found)
        self.assertEqual(
            offenders,
            [],
            "the app must not read Streamlit config options: an option's value "
            "depends on the machine and the working directory, so a decision made "
            "from one is ambient. Read the committed file instead (see "
            "app/main.py:_check_streamlit_config_hardening).",
        )

    def test_the_scan_flags_a_planted_read(self) -> None:
        """A guard that cannot fail is decoration."""
        source = "import streamlit as st\n\n\ndef f():\n    return st.get_option('server.port')\n"
        self.assertTrue(option_read_offence("planted.py", source))

    def test_the_scan_accepts_reading_the_file(self) -> None:
        """The committed file read as text is the *intended* shape."""
        source = (
            "from pathlib import Path\n"
            "cfg = Path(__file__).parent.parent / '.streamlit' / 'config.toml'\n"
            "text = cfg.read_text(encoding='utf-8')\n"
        )
        self.assertEqual(option_read_offence("fine.py", source), [])


class TestEveryTestModuleImportsTheHarness(unittest.TestCase):
    """The rule itself lives in ``tests/harness_imports.py``.

    It has two callers — this scan and ``scripts/hooks/pre-commit`` — and a second
    copy would drift, so both read it from there. That module's own tests are
    pinned here and its behaviour under the hook in
    ``tests/test_security_gitignore.py``.
    """

    def test_no_test_module_is_left_unprotected(self) -> None:
        offenders = harness_imports.offenders(
            harness_imports.test_module_sources(TESTS_DIR)
        )
        self.assertEqual(offenders, [], harness_imports.REMEDY)

    def test_the_scan_flags_a_module_that_forgets(self) -> None:
        """A guard that cannot fail is decoration, so the scan is self-tested."""
        source = "import unittest\n\n\nclass T(unittest.TestCase):\n    pass\n"
        offence = harness_imports.harness_import_offence("forgot.py", source)
        self.assertIsNotNone(offence)
        self.assertIn("does not import", str(offence))

    def test_the_scan_flags_a_module_that_imports_the_app_first(self) -> None:
        source = (
            "import unittest\n"
            "from app import config\n"
            "from tests import hermetic  # noqa: E402,F401\n"
        )
        offence = harness_imports.harness_import_offence("too_late.py", source)
        self.assertIsNotNone(offence)
        self.assertIn("lands after", str(offence))

    def test_the_scan_accepts_a_module_that_does_it_right(self) -> None:
        source = (
            "import unittest\n"
            "from tests import hermetic  # noqa: E402,F401\n"
            "from app import config\n"
        )
        self.assertIsNone(
            harness_imports.harness_import_offence("fine.py", source)
        )

    def test_the_stdlib_set_is_exact_where_the_interpreter_has_one(self) -> None:
        """The rule's discriminator has to be the stdlib, not an approximation.

        ``sys.stdlib_module_names`` arrived in 3.10 and is what this suite runs
        with; anything else is a fallback for the interpreter the *commit hook*
        may find (see below), never the path under test.
        """
        if not hasattr(sys, "stdlib_module_names"):
            self.skipTest("this interpreter predates sys.stdlib_module_names")
        self.assertEqual(
            harness_imports.stdlib_names(), set(sys.stdlib_module_names)
        )

    def test_the_fallback_covers_builtins_not_only_the_stdlib_directory(self) -> None:
        """A measured false positive, pinned.

        On Python 3.9 — which the hook picks up when a checkout has no ``.venv``
        and ``python3`` is the Command Line Tools one — ``sys`` has no file in the
        stdlib directory, because it is compiled into the interpreter. A fallback
        built from that directory alone therefore called ``import sys`` foreign and
        refused a *correctly wired* module. Both other sources are asserted here.
        """
        names = harness_imports._interpreter_stdlib_names()
        for builtin in ("sys", "builtins"):
            self.assertIn(builtin, names, "compiled in, so no file anywhere")
        for module in ("unittest", "pathlib", "ast", "json"):
            self.assertIn(module, names, "a file or package in the stdlib directory")
        self.assertNotIn("app", names)
        self.assertNotIn("tests", names)
        self.assertNotIn("site-packages", names)

    def test_the_fallback_answers_for_this_project_not_for_the_interpreter(
        self,
    ) -> None:
        """The second measured false positive, and the worse one.

        ``tomllib`` arrived in 3.11. A fallback that reported only what the running
        interpreter has declared a *correctly wired* module unwired — on the commit
        that added this very rule, under the 3.9 the hook found with no ``.venv`` —
        and refused it. The hook asks whether the import is stdlib for this code,
        whose floor is 3.12, so the names added since 3.8 have to be there.
        """
        with patch.object(sys, "stdlib_module_names", None):
            names = harness_imports.stdlib_names()
        self.assertIn("sys", names)
        self.assertIn("unittest", names)
        self.assertIn("tomllib", names)
        self.assertNotIn("app", names)

    def test_the_newer_stdlib_names_are_a_subset_of_this_interpreter(self) -> None:
        """So a wrong entry fails here rather than misjudging somebody's commit."""
        if not hasattr(sys, "stdlib_module_names"):
            self.skipTest("this interpreter predates sys.stdlib_module_names")
        unknown = harness_imports.STDLIB_BEYOND_3_8 - set(sys.stdlib_module_names)
        self.assertEqual(
            unknown,
            set(),
            "these entries are not stdlib in the interpreter the suite runs on, so "
            "the fallback would treat a real module as foreign and refuse a "
            "correctly wired commit",
        )



class TestAHostileEnvironmentCannotReachTheSuite(unittest.TestCase):
    """The end-to-end guard: replay the hostile environment in a fresh process.

    This is the check that fails on *any* machine if the harness is removed or its
    action is lost, not just on one that happens to have a variable exported. It
    uses the same fixtures as the CI job that runs the whole suite this way, so the
    canary and that job cannot drift apart.
    """

    def test_the_env_sensitive_tests_still_pass_under_a_hostile_environment(
        self,
    ) -> None:
        environment = hostile.hostile_child_environment()
        process = subprocess.run(
            [sys.executable, "-m", "unittest", *CANARY_TESTS, "-v"],
            cwd=str(PROJECT_ROOT),
            env=environment,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if process.returncode != 0:
            self.fail(
                "the suite read the environment it was run in: "
                f"`{' '.join(hostile.HOSTILE_ENVIRONMENT)}`\n"
                f"--- stdout ---\n{process.stdout[-4000:]}\n"
                f"--- stderr ---\n{process.stderr[-4000:]}"
            )


class TestTheCIGateIsStillWired(unittest.TestCase):
    """A gate that can be deleted without a test noticing is not a gate.

    The workflow's hermetic job is what stops ambient dependence returning through
    some future test, and the dev-layout job is what keeps the development-mode pin
    meeting a real install instead of only the simulation in the guard above. Nothing
    else in the suite reads the workflow file, so without this a deleted or renamed
    job would leave CI green with the property silently unguarded — and the
    repository has been bitten by an invalid workflow before, which fails every run
    in zero seconds with no jobs created at all.
    """

    WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "test.yml"
    JOB = "hermetic"
    RUNNER = "python -m tests.hostile"
    LATEST_JOB = "streamlit-latest"
    LATEST_RUNNER = "unittest discover -s tests"
    DEV_LAYOUT_JOB = "dev-layout"
    DEV_LAYOUT_RUNNER = "python -m tests.devlayout"

    def _job_commands(self, name: str) -> str:
        """Every ``run:`` line of a job, joined — the thing the guards assert on."""
        job = self._workflow()["jobs"].get(name)
        self.assertIsNotNone(job, f"there is no '{name}' job")
        return "\n".join(str(step.get("run", "")) for step in job["steps"])

    def _triggers(self) -> dict:
        """The workflow's ``on:`` block.

        Spelled out because PyYAML resolves a bare ``on`` to the boolean ``True``
        (YAML 1.1), so looking up ``"on"`` returns nothing and every assertion
        below would pass while checking an empty dict — a guard that cannot fail.
        """
        workflow = self._workflow()
        return workflow.get("on") or workflow.get(True) or {}

    def _workflow(self) -> dict:
        try:
            import yaml
        except ImportError:  # pragma: no cover - a declared test prerequisite
            raise unittest.SkipTest("PyYAML is not installed")
        return yaml.safe_load(self.WORKFLOW.read_text(encoding="utf-8"))

    def test_the_workflow_is_a_valid_file(self) -> None:
        self.assertIn(
            "jobs",
            self._workflow(),
            "the workflow does not parse, which fails every run with no jobs",
        )

    def test_the_hermetic_job_runs_the_shared_fixtures(self) -> None:
        job = self._workflow()["jobs"].get(self.JOB)
        self.assertIsNotNone(
            job,
            f"there is no '{self.JOB}' job, so nothing runs the suite under the "
            "hostile fixtures and ambient dependence can land again unnoticed",
        )
        commands = "\n".join(str(step.get("run", "")) for step in job["steps"])
        self.assertIn(
            self.RUNNER,
            commands,
            "the hermetic job no longer runs the shared fixtures, so it is not "
            "testing the same thing the canaries do",
        )

    def test_the_job_gates(self) -> None:
        """A job that cannot fail is reporting, not gating."""
        job = self._workflow()["jobs"][self.JOB]
        self.assertNotIn("continue-on-error", job)
        self.assertNotIn("if", job)

    def test_the_trigger_lookup_finds_the_triggers(self) -> None:
        self.assertTrue(
            self._triggers(),
            "the trigger block came back empty, so this file is asserting nothing "
            "about when the jobs run",
        )

    def test_the_newest_streamlit_job_is_scheduled(self) -> None:
        """The pinned lock is what makes this job necessary.

        ``requirements.lock`` fixes Streamlit to one version, so a *release* can
        change option resolution — or rename a private hook the harness reaches
        into — while every other check stays green. Nothing but a schedule notices
        that, and it must not run per-PR: the news arrives weekly, and a fresh
        dependency resolution on every push is not news about this diff.
        """
        self.assertIn(
            "schedule",
            self._triggers(),
            "nothing runs this workflow on a schedule, so no job in it can notice a "
            "Streamlit release",
        )
        job = self._workflow()["jobs"].get(self.LATEST_JOB)
        self.assertIsNotNone(
            job,
            f"there is no '{self.LATEST_JOB}' job, so a Streamlit release that "
            "changes option resolution fails nothing: the suite stays green while "
            "tests/hermetic.py quietly stops neutralising",
        )
        self.assertIn(
            "schedule",
            str(job.get("if", "")),
            "this job must be limited to schedule/dispatch — otherwise a fresh "
            "dependency resolution runs on every push, which is not about this "
            "repository's diff",
        )

    def test_the_development_layout_job_installs_the_layout_and_runs_the_runner(
        self,
    ) -> None:
        commands = self._job_commands(self.DEV_LAYOUT_JOB)
        self.assertIn(
            "--target",
            commands,
            "the job no longer installs Streamlit outside site-packages, so the "
            "suite would run as the deployment does and exercise none of the layout "
            "the in-suite guard only simulates",
        )
        self.assertIn(
            "--no-deps",
            commands,
            "without it the target install drags Streamlit's dependencies into the "
            "relocated copy, so the job would test a different dependency set than "
            "the lock pins and a failure would have two explanations",
        )
        self.assertIn(
            "streamlit==",
            commands,
            "this job moves the install *layout*; the version must stay the one "
            "requirements.lock pins, or a red run cannot be attributed either"
            " — the newest release is the streamlit-latest job's variable",
        )
        self.assertIn(
            self.DEV_LAYOUT_RUNNER,
            commands,
            "the job must run the suite through tests/devlayout.py, which refuses to "
            "continue unless the layout is real and the pin holds in it",
        )

    def test_the_development_layout_job_gates(self) -> None:
        """It is the pin's only real-install exercise, so it must not be optional."""
        job = self._workflow()["jobs"][self.DEV_LAYOUT_JOB]
        self.assertNotIn("continue-on-error", job)
        self.assertNotIn("if", job)

    def test_the_newest_streamlit_job_upgrades_streamlit_and_runs_the_suite(
        self,
    ) -> None:
        job = self._workflow()["jobs"][self.LATEST_JOB]
        runs = [str(step.get("run", "")) for step in job["steps"]]
        self.assertTrue(
            [run for run in runs if "streamlit" in run and "--upgrade" in run],
            "the job never upgrades Streamlit, so it would test the locked version "
            "twice and could never see a release",
        )
        self.assertTrue(
            [run for run in runs if self.LATEST_RUNNER in run],
            f"the job must run the offline suite ({self.LATEST_RUNNER}); the suite's "
            "hermetic guards are the sensors for option resolution",
        )

    def test_the_newest_streamlit_job_is_allowed_to_fail(self) -> None:
        """It surfaces a change rather than gating the diff — but it must surface.

        ``continue-on-error`` would leave the run green while the assertions
        failed, which is the outcome this repository has already been bitten by; the
        thing that keeps it from blocking anyone is the schedule, not a swallow.
        """
        job = self._workflow()["jobs"][self.LATEST_JOB]
        self.assertNotIn("continue-on-error", job)


if __name__ == "__main__":
    unittest.main()
