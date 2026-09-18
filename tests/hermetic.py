"""Hermetic test session — the suite must not read the machine it runs on.

A test that passes on one laptop and fails in CI (or the reverse) is usually not
a flaky test: it is a test that read ambient configuration. Five ambient sources
reach this suite, and each reaches the whole *process* rather than one test:

* ``app/config.py`` calls ``load_dotenv(PROJECT_ROOT / ".env", override=True)``
  at import, so a developer's uncommitted ``.env`` is copied into ``os.environ``
  and stays there for every test that follows. A local ``.env`` naming
  ``OPENAI_API_KEY``/``LLM_MODEL_MAIN``/…  therefore changes what the suite sees,
  and CI — which has no ``.env`` — sees the opposite.
* ``app/config.py`` reads ~110 names into **module-level constants** at import
  time. An exported ``VA_LSE_*`` variable is frozen into the app under test, and
  no ``patch.dict(os.environ, ...)`` written inside a test can undo it.
  ``clear=False`` — the pattern this suite uses most — leaves every other ambient
  variable in place anyway.
* ``app/config.py`` falls back to Streamlit's secrets manager, and Streamlit's
  manager is a *file on this machine* — ``.streamlit/secrets.toml`` in the
  project or in ``~``. Worse than a read: as Streamlit parses that file it
  **promotes every string secret into ``os.environ``**, so the value lands in the
  environment *after* any sweep of it, and stays for every later test.
* Streamlit also derives ``global.developmentMode`` from how it was **installed** —
  true whenever its package is not under a ``site-packages`` directory — and that is
  not a label but a fork in behaviour: ``logger.level`` and ``logger.messageFormat``
  default differently, and a config that sets a port makes parsing *raise*, which
  reaches AppTest as a dead runner thread rather than as that message.
* Streamlit reads a *configuration* file as well — ``.streamlit/config.toml`` in
  the project or in ``~`` — and a long tail of runtime behaviour comes out of it.
  Nothing in ``app/`` reads an option today: ``app/main.py``'s hardening check
  reads the committed file as *text*, deliberately, because Streamlit offers no
  API to ask which file a value came from. Streamlit itself does read options
  though (``client.showErrorDetails`` decides how much of an error is rendered,
  ``logger.level`` how much is logged), and the *machine's* file shadows the
  repository's for any key the repository does not pin — so it is ambient input
  either way, and the suite pins it below.

Measured, not assumed: run under a hostile environment (every configurable knob
at a non-default value) the suite produces 13 failures and 11 errors across eight
modules, every one of them for the wrong reason. ``test_pipeline_guard`` asserts
the 1800 s pipeline timeout while ``VA_LSE_PIPELINE_TIMEOUT_SECONDS=1`` is
exported; ``test_tracing``'s span-tree tests break because ``VA_LSE_TRACING=1``
starts a real OTLP exporter at a non-existent endpoint; ``test_audit_backup``'s
rotation test reads ``VA_LSE_AUDIT_LOG_BACKUPS``; and so on. Measured with a
hostile ``secrets.toml`` in the working directory: ``load_settings()`` returned
the file's ``OPENAI_API_KEY``, reported both its names as secret-sourced, and
left ``LLM_MODEL_MAIN`` sitting in ``os.environ``.

The repair is one place rather than twenty-four. This module empties the ambient
configuration at import, before any test module imports the app — except for what it
*pins* rather than empties, because the deployment has a definite value there and
the machine merely has *a* value: the session reads the repository's own
``.streamlit/config.toml`` and nothing else, and runs with
``global.developmentMode`` false, which is what "installed under site-packages"
means. Importing it is a single line at the top of a test module, and the ordering
is load bearing:

    import hermetic  # noqa: F401  (hermetic test session — see tests/hermetic.py)

It must come **before** the module's first ``app`` import, because those
constants are read once, at ``app/config.py`` import, and the first import in the
session decides what they hold. ``tests/test_hermetic.py`` scans for exactly that
mistake.

Two deliberate exceptions, both narrow:

* ``VA_LSE_TEST_*`` names survive. Setting one is the existing convention for a
  *runner* opting a test into real ambient setup — ``test_job_queue_redis_live``
  reads ``VA_LSE_TEST_REDIS_URL`` and skips itself when it is unset, and the CI
  job that exercises it sets exactly that name.
* a short list of process-essential names (``PATH``, ``HOME``, …) is never
  touched, so a test that repoints ``HOME`` to expand ``~`` still works.

A test that *needs* a configuration value still gets it: patch the ``config``
attribute directly (``patch.object(config, "AUDIT_LOG_BACKUPS", 3)``), or set the
variable inside the test body with ``patch.dict``, which runs long after this
scrub. What no test can do any more is *accidentally* read the machine's value.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
ENV_FILE = PROJECT_ROOT / ".env"

#: A runner opts a test into real ambient setup by setting a name with this
#: prefix (see the module docstring). Never stripped.
KEEP_PREFIXES = ("VA_LSE_TEST_",)

#: Process-essential names the operating system or interpreter owns. If the app
#: ever reads one of these, the scrub must not be what takes it away.
NEVER_STRIP = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "TEMP",
        "TMP",
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
    }
)

# Every name the app reads from the environment goes through a helper whose name
# ends in ``_env``/``_setting``, or straight through ``os.environ``:
#
#   os.getenv("X")   _int_env("X")   _positive_int_env("X")   _setting("X")
#   os.environ["X"]  os.environ.get("X")
#
# One pattern covers the surface, and the surface is *derived from the source*
# rather than listed here by hand, so a knob added tomorrow is covered the day it
# appears — no second place to remember.
_ENV_READ = re.compile(
    r"""(?:\w*env\s*\(|\w*setting\s*\(|environ\s*\.\s*get\s*\(|environ\s*\[)"""
    r"""\s*["']([A-Za-z_][A-Za-z0-9_]*)["']"""
)

# An environment name, as conventionally spelled.
_NAME = re.compile(r"[A-Z][A-Z0-9_]*\Z")


def _app_sources() -> list[Path]:
    return sorted(APP_DIR.rglob("*.py"))


def configurable_env_names() -> frozenset[str]:
    """Names the app reads from the environment, scanned out of its source."""
    found: set[str] = set()
    for path in _app_sources():
        for name in _ENV_READ.findall(path.read_text(encoding="utf-8")):
            if _NAME.match(name):
                found.add(name)
    return frozenset(found)


def env_file_keys() -> frozenset[str]:
    """Keys named by the project ``.env`` — keys only, values are never read."""
    if not ENV_FILE.exists():
        return frozenset()
    keys: set[str] = set()
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if _NAME.match(key):
            keys.add(key)
    return frozenset(keys)


def protected_names() -> frozenset[str]:
    """Every ambient name this module takes out of the test process."""
    names = set(configurable_env_names()) | set(env_file_keys())
    return frozenset(
        name
        for name in names
        if name not in NEVER_STRIP and not name.startswith(KEEP_PREFIXES)
    )


def _ignore_dotenv(*_args: object, **_kwargs: object) -> bool:
    """Stand-in for ``dotenv.load_dotenv``: this process has no ``.env``."""
    return False


def neutralize_dotenv() -> bool:
    """Make ``load_dotenv`` a no-op for the rest of the process.

    ``app/config.py`` does ``from dotenv import load_dotenv`` when it is
    imported, so replacing the attribute on the ``dotenv`` module is enough —
    provided this runs first, which the import ordering required of every test
    module guarantees. Returns False when ``dotenv`` is not importable (it is a
    hard dependency of the app, so this is defensive only).
    """
    try:
        import dotenv
    except ImportError:  # pragma: no cover - a hard dependency of the app
        return False
    dotenv.load_dotenv = _ignore_dotenv  # type: ignore[assignment]
    return True


#: The one place Streamlit's secrets manager learns where to look. Its value is
#: read when the file is parsed, not when it is configured, so overriding it is
#: enough to make a ``secrets.toml`` on this machine invisible.
SECRETS_FILES_OPTION = "secrets.files"


def neutralize_streamlit_secrets() -> bool:
    """Make Streamlit behave as though no ``secrets.toml`` exists on this machine.

    ``app/config.py`` resolves a credential from the environment, then from
    ``.env``, then from the secrets manager — which is Streamlit's file, in the
    project or in ``~``. A developer who uses one therefore runs a different app
    from CI, which has none.

    Rebinding ``st.secrets`` would not be enough: the same singleton is reachable
    as ``streamlit.runtime.secrets.secrets_singleton`` and as
    ``streamlit._secrets_singleton``, and Streamlit's own runtime touches it. The
    option below is the narrowest hook that covers every route, because it is the
    only source of the candidate paths. Emptying it makes ``Secrets._parse()``
    take exactly the path a machine with no secrets file takes — no values, no
    promotion into ``os.environ``, no file watchers — which is what CI does, so
    the suite now tests the deployed condition rather than the developer's.

    ``STREAMLIT_SECRETS_FILES`` does *not* work for this (measured: the option
    still reported both default paths), which is why this patches the accessor.
    Returns False when Streamlit is not importable — it is a hard dependency of
    the app, so that is defensive only.
    """
    try:
        import streamlit.config as streamlit_config
    except ImportError:  # pragma: no cover - a hard dependency of the app
        return False
    original_get_option = streamlit_config.get_option

    def get_option(option: str, *args: object, **kwargs: object) -> object:
        if option == SECRETS_FILES_OPTION:
            return []
        return original_get_option(option, *args, **kwargs)  # type: ignore[arg-type]

    streamlit_config.get_option = get_option  # type: ignore[assignment]
    return True


#: The config file Streamlit looks for, and the only one this session may read:
#: the repository's own copy, resolved from the repository rather than from the
#: working directory. ``.streamlit/config.toml`` is committed *as* the deployment
#: contract (``DEPLOYMENT.md`` copies it into the image), so the suite should run
#: it — what it should not run is the file in ``~``, which exists on one machine
#: and not on the next, or a copy that happens to sit in whatever directory the
#: tests were started from.
CONFIG_FILE_NAME = "config.toml"
REPO_CONFIG_FILE = PROJECT_ROOT / ".streamlit" / CONFIG_FILE_NAME


def sensitive_option_env_names() -> frozenset[str]:
    """The environment variables Streamlit itself honours for a config option.

    Derived from Streamlit's option table rather than written out, because the
    two names below are the *only* route by which a variable can change a config
    option in this version. Measured on Streamlit 1.63.0: ``env_var`` is read in
    exactly one place — ``_update_config_with_sensitive_env_var``, which skips
    every option not marked ``sensitive`` — so of 343 options, exactly two
    (``server.cookieSecret``, ``mapbox.token``) consult the environment. The
    documented ``STREAMLIT_SERVER_PORT``-style overrides do nothing for the rest,
    which is worth knowing before trusting one. Reading the table does not parse
    any file, so this cannot pull a value in while sweeping for one.
    """
    try:
        import streamlit.config as streamlit_config
    except ImportError:  # pragma: no cover - a hard dependency of the app
        return frozenset()
    template = getattr(streamlit_config, "_config_options_template", None) or {}
    return frozenset(
        option.env_var
        for option in template.values()
        if option.sensitive and option.env_var
    )


SENSITIVE_ENV_NAMES = sensitive_option_env_names()


def neutralize_streamlit_config_files() -> bool:
    """Make the session read the repository's ``.streamlit/config.toml`` only.

    Streamlit resolves configuration from a global file in ``~``, then a
    per-project file under the *working directory*, then script-level and finally
    flags. Replacing the resolver's file list — the single place the parse gets
    its candidates from — pins the session to the repository's committed file, so
    the two machine-scoped routes cannot reach it. That is deliberately not the
    same as emptying the list: an empty list would drop the suite to Streamlit's
    bare defaults, which is a machine CI never runs, whereas the committed file
    *is* what production runs.

    It also removes the working directory from the picture, which matters because
    the tests can be started from anywhere: today the project-level file is
    ``$CWD/.streamlit/config.toml``, so the same suite reads different
    configuration depending on where it was launched.

    Returns False when Streamlit is not importable — it is a hard dependency of
    the app, so that is defensive only.
    """
    try:
        import streamlit.config as streamlit_config
    except ImportError:  # pragma: no cover - a hard dependency of the app
        return False
    original_get_config_files = streamlit_config.get_config_files

    def get_config_files(file_name: str, *args: object, **kwargs: object) -> list[str]:
        if file_name == CONFIG_FILE_NAME:
            return [str(REPO_CONFIG_FILE)] if REPO_CONFIG_FILE.exists() else []
        return original_get_config_files(file_name, *args, **kwargs)  # type: ignore[arg-type]

    streamlit_config.get_config_files = get_config_files  # type: ignore[assignment]
    return True


#: The option that says how Streamlit was *installed* rather than what anyone
#: configured — see ``neutralize_streamlit_development_mode``.
DEVELOPMENT_MODE_OPTION = "global.developmentMode"

#: Where a value this module installs came from, for ``get_where_defined``.
DEFINED_BY_HARNESS = "tests/hermetic.py"


def neutralize_streamlit_development_mode() -> bool:
    """Run as the deployed app does, not as whoever installed Streamlit did.

    ``global.developmentMode`` is not a preference anyone sets: Streamlit derives it
    from the install *layout* (``"site-packages" not in __file__``), so it is true
    for a source checkout, an editable install, and a vendored or ``--target`` one.
    It is a fork in behaviour rather than a label — ``logger.level`` and
    ``logger.messageFormat`` default differently under it — and, with a port set in
    any config file, it makes parsing raise, which reaches AppTest not as that
    message but as a dead runner thread: measured on 1.64.0, either a ``KeyError``
    about ``$$STREAMLIT_INTERNAL_KEY_SCRIPT_RUN_WITHOUT_ERRORS`` or a bare
    ``AppTest script run timed out`` depending on timing. A contributor with an
    editable Streamlit would therefore see the view tests fail with nothing to go
    on, while CI stayed green.

    The config pinning above already prevents the *port* from arriving from this
    machine, so this is the second, independent guard: it makes the suite's
    environment the deployment's (Streamlit installed under ``site-packages``)
    instead of the developer's, which is the same reasoning as pinning
    ``secrets.files``. The value is written to the option *template*, because
    parsing copies that — no parse has happened at import, and the harness is
    imported before anything that would parse.

    Returns False when Streamlit is not importable, or when the option is gone.
    """
    try:
        import streamlit.config as streamlit_config
    except ImportError:  # pragma: no cover - a hard dependency of the app
        return False
    template = getattr(streamlit_config, "_config_options_template", None)
    if not template or DEVELOPMENT_MODE_OPTION not in template:
        return False  # pragma: no cover - the option has existed for years
    # Both, in case something parsed config before this ran.
    for options in (template, streamlit_config._config_options):
        if options is None:
            continue
        options[DEVELOPMENT_MODE_OPTION].set_value(False, DEFINED_BY_HARNESS)
    return True


def strip_ambient_config() -> tuple[str, ...]:
    """Remove the protected names from ``os.environ``; return those removed.

    The ``.env`` keys are stripped as well as the names the app reads, because
    that is the only way to undo a load that already happened: if any test module
    reached ``app.config`` before this one, ``load_dotenv`` has already copied
    the file into the environment, and blocking *future* loads cannot take it
    back out.

    ``SENSITIVE_ENV_NAMES`` is included because config files are not the only way
    into Streamlit's option table: those two variables are read from the
    environment at parse time, and an exported ``STREAMLIT_SERVER_COOKIE_SECRET``
    on the machine under test is ambient input like any other.
    """
    removed: list[str] = []
    for name in sorted(protected_names() | SENSITIVE_ENV_NAMES):
        if name in os.environ:
            del os.environ[name]
            removed.append(name)
    return tuple(removed)


# Applied on import: from here on the process cannot see the machine's
# configuration, so every test module imported after this one — and with
# ``unittest discover`` that is all of them, before a single test runs — sees the
# app's code defaults and the repository's config file, and nothing else.
DOTENV_NEUTRALIZED = neutralize_dotenv()
STREAMLIT_SECRETS_NEUTRALIZED = neutralize_streamlit_secrets()
STREAMLIT_CONFIG_NEUTRALIZED = neutralize_streamlit_config_files()
STREAMLIT_DEVELOPMENT_MODE_NEUTRALIZED = neutralize_streamlit_development_mode()
REMOVED_AMBIENT_NAMES = strip_ambient_config()
