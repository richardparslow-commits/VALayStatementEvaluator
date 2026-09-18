"""The install *layout* is not a setting: run the suite as a developer's install sees it.

Streamlit derives ``global.developmentMode`` from its own package path — true
whenever ``site-packages`` is not in it — so a source checkout, an editable
install, a vendored copy and a ``pip install --target`` all report true. It is a
fork in behaviour rather than a label: ``logger.level`` and ``logger.messageFormat``
default differently, and with a port in any config file it makes parsing raise,
which under AppTest arrives as a dead runner thread rather than as the message.
That is why ``tests/hermetic.py`` pins it false — production is "installed under
``site-packages``", so the session should behave as the deployment does rather than
as whoever ran it did.

The pin has a sensor in the suite
(``tests/test_hermetic.py::TestADevelopmentModeInstallCannotReachTheSession``), but
the sensor **simulates** the layout: it writes the option's value on the template
before anything parses, because a test process cannot reinstall Streamlit. A
simulation that has drifted from what it simulates passes while testing nothing —
the failure mode this repository guards against everywhere else — so this runner is
the other half: CI installs Streamlit where ``site-packages`` is not in its path and
runs the whole suite *there*, so the pin meets real material. Its first check is the
drift sensor: if Streamlit ever stops calling such an install a development one, or
derives the mode some other way, this fails and says so instead of the in-suite
simulation quietly asserting about a layout that no longer exists.

Three checks, ordered so a failure is attributable:

1. a child that skips the harness must report ``developmentMode`` **true**, or the
   install is not a development layout and everything below is a green run that
   tested nothing;
2. the same child *with* the harness must report it **false** — the pin working on
   real material rather than on the simulation;
3. the offline suite must pass in that install.

Both children therefore have to resolve the same Streamlit, which is checked too:
otherwise the contrast in (2) would be about two different packages.

Reproduce the CI job locally by pointing ``PYTHONPATH`` at a target install::

    pip install --target /tmp/streamlit-dev-layout --no-deps --upgrade streamlit
    PYTHONPATH=/tmp/streamlit-dev-layout python -m tests.devlayout

or run it in an editable or vendored Streamlit, where check 1 already holds and
there is nothing to install::

    python -m tests.devlayout

Extra arguments go to ``unittest discover``, exactly as in ``tests/hostile.py``.

The install is deliberately the version ``requirements.lock`` pins (the CI job
copies that version into the target): the variable here is the layout, so a failure
has one explanation. The newest release is ``streamlit-latest``'s variable, and the
suite with hostile *configuration* is ``tests.hostile``'s — three different
conditions, three jobs, so a red run names a cause.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def child_program(*, protect: bool) -> str:
    """The probe a child runs: what does *this* install report?

    The harness import lives in the child rather than being arranged from here,
    because the ordering the pin depends on — before anything parses — has to be the
    child's own. The two variants differ by that one line and nothing else, so the
    comparison in ``main`` is about the harness rather than about a different
    Streamlit.
    """
    return (
        "import json\n"
        "import sys\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        "import streamlit\n"
        "import streamlit.config as config\n"
        + (
            "from tests import hermetic  # noqa: F401  (the pin under test)\n"
            if protect
            else ""
        )
        + "print(json.dumps({\n"
        "    'version': streamlit.__version__,\n"
        "    'file': streamlit.__file__,\n"
        "    'dev_mode': config.get_option('global.developmentMode'),\n"
        "}))\n"
    )


def layout_report(*, protect: bool) -> dict[str, object]:
    """Run the probe in a child process; return what it reported."""
    process = subprocess.run(
        [sys.executable, "-c", child_program(protect=protect)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if process.returncode != 0:
        raise SystemExit(
            "could not inspect the install layout with a child process:\n"
            f"--- stdout ---\n{process.stdout}\n--- stderr ---\n{process.stderr[-4000:]}"
        )
    report: dict[str, object] = json.loads(process.stdout.strip().splitlines()[-1])
    return report


def describe(report: dict[str, object]) -> str:
    """One line naming the install a report came from, for the log and messages."""
    return f"streamlit {report['version']} at {report['file']}"


def assert_development_layout(report: dict[str, object]) -> str:
    """Check 1: the layout is live, or this run would test nothing.

    Nothing else here means anything if it is not: the suite would pass in a normal
    install and the job would report green while exercising none of what it exists
    for. So it fails rather than warns, and the message says how to get a layout
    that qualifies.
    """
    if report["dev_mode"] is not True:
        raise SystemExit(
            "this install is not a development layout, so the run below would test "
            f"nothing: {describe(report)} reports global.developmentMode="
            f"{report['dev_mode']!r}. Streamlit calls an install a development one "
            "when its package path has no site-packages (or dist-packages, or "
            "__pypackages__) in it. Put such a copy on PYTHONPATH — see the "
            "`dev-layout` job in .github/workflows/test.yml — or run this from an "
            "editable checkout of Streamlit."
        )
    return f"{describe(report)} is a development layout"


def assert_pin_holds(report: dict[str, object]) -> str:
    """Check 2: ``tests/hermetic.py`` overrides it, on real material.

    This is the claim the in-suite guard makes about a *simulated* layout; here it
    is made about an actual one, in the same install the check above measured.
    """
    if report["dev_mode"] is not False:
        raise SystemExit(
            "tests/hermetic.py did not pin global.developmentMode to false in a "
            f"development-layout install: {describe(report)} reports "
            f"{report['dev_mode']!r}. The session is running with the developer's "
            "environment, which is the condition the pin exists to prevent."
        )
    return "the harness pins global.developmentMode to false in the same install"


def same_install(unprotected: dict[str, object], protected: dict[str, object]) -> None:
    """Both children must be looking at the same Streamlit, or (2) proves nothing."""
    if (unprotected["file"], unprotected["version"]) != (
        protected["file"],
        protected["version"],
    ):
        raise SystemExit(
            "the two probe children resolved different Streamlit installs, so the "
            "pin below would not be measured against the layout above:\n"
            f"  without the harness: {describe(unprotected)}\n"
            f"  with the harness:    {describe(protected)}"
        )


def run_suite(argv: list[str]) -> int:
    """Run the offline suite in this install, streaming output and the exit code."""
    return subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", *argv],
        cwd=str(PROJECT_ROOT),
        check=False,
    ).returncode


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    unprotected = layout_report(protect=False)
    protected = layout_report(protect=True)
    same_install(unprotected, protected)
    # flush=True throughout: redirected to a pipe — which is what CI does — Python
    # block-buffers this process while the child writes straight through, so without
    # it the header explaining the run appears after the run (tests/hostile.py
    # documents the same trap).
    print(f"layout: {assert_development_layout(unprotected)}", flush=True)
    print(f"pin: {assert_pin_holds(protected)}", flush=True)
    print("running the offline suite as a development-layout install…", flush=True)
    return run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
