"""The sandbox image target, checked without a container engine.

No CI job builds an image, so before this module a Dockerfile edit was the one
change in this repository that *nothing* verified: drop a COPY and the suite
fails only inside a sandbox, with an error that looks nothing like its cause;
drop the ``USER root`` and an agent gets a workspace it cannot write to; drop
``git`` and the git-ground-truth tests error instead of passing. So the file is
parsed here and its contract asserted. The parser itself, and the .dockerignore
matcher, live in ``tests/dockerfile.py`` — shared with ``test_dockerignore.py``,
because two copies would drift and drift here fails open.

The contract has two halves, and the second is what keeps the first from being
"just ship everything":

* the **sandbox** image must be able to run the app *and* the offline suite —
  the interpreter, the hash-pinned lock, the dev extras, and every file the
  suite reads off the filesystem (derived from the working tree, so a new root
  page or workflow file fails this test rather than a build);
* the **runtime** image — the one that gets deployed — must not quietly grow
  any of that. It stays non-root, stays free of the dev extras, and stays
  without ``tests/`` and ``scripts/``, which is a decision its own comments
  make on purpose.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from tests import dockerfile  # noqa: E402

PROJECT_ROOT = dockerfile.PROJECT_ROOT
RUNTIME_STAGE = "runtime"
SANDBOX_STAGE = "sandbox"


class SandboxImageTestCase(unittest.TestCase):
    """Shared parsing, so every assertion below reads the same stages."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = dockerfile.DOCKERFILE.read_text(encoding="utf-8")
        cls.stages = dockerfile.stages(cls.text)
        for name in (RUNTIME_STAGE, SANDBOX_STAGE):
            assert name in cls.stages, f"Dockerfile has no `{name}` stage"

    @property
    def runtime(self) -> list[str]:
        return self.stages[RUNTIME_STAGE]

    @property
    def sandbox(self) -> list[str]:
        return self.stages[SANDBOX_STAGE]

    def carries(self, path: str) -> bool:
        """Path present in the final sandbox *image* — its own COPYs or inherited."""
        return dockerfile.carries(self.sandbox, path) or dockerfile.carries(self.runtime, path)

    def last_user(self, stage: list[str]) -> str | None:
        users = [i.split()[1] for i in stage if i.startswith("USER ")]
        return users[-1] if users else None


class TestSandboxStageIsAnAgentWorkspace(SandboxImageTestCase):
    def test_the_stage_is_built_on_the_deployment_image(self) -> None:
        """So a sandbox runs the interpreter and packages a deployment runs."""
        self.assertIn("FROM runtime AS sandbox", self.text)
        self.assertEqual(
            [i for i in self.sandbox if i.startswith("FROM ")], [],
            "the sandbox stage must not start from a second base image",
        )

    def test_it_runs_as_root(self) -> None:
        """A workspace that cannot be written to is not a workspace."""
        self.assertEqual(self.last_user(self.sandbox), "root")

    def test_its_pip_installs_are_additive_to_the_lock(self) -> None:
        """Every install here adds to the lock, and none of them move it.

        The lock is installed by the inherited stage *before* anything below runs,
        which is what makes the extras additive instead of an unpinned upgrade: a
        `--require-hashes` line here would re-pin the runtime's set, and an
        `--upgrade` would change the packages CI proved. A second install is
        expected — the OCR engine is not a test tool, so it gets its own line (and
        its own cache layer) rather than riding along with ``requirements-dev.txt``.
        """
        installs = [i for i in self.sandbox if i.startswith("RUN pip install")]
        self.assertTrue(installs, "this stage installs nothing at all")
        joined = " ".join(installs)
        self.assertIn("requirements-dev.txt", joined)
        self.assertNotIn("--require-hashes", joined)
        self.assertNotIn("--upgrade", joined)
        self.assertTrue(
            any("requirements.lock" in i and "--require-hashes" in i for i in self.runtime)
        )

    def test_it_carries_git_because_the_suite_asks_git_for_ground_truth(self) -> None:
        apt = " ".join(i for i in self.sandbox if i.startswith("RUN apt-get"))
        self.assertIn("git", apt)

    def test_it_initialises_a_git_repository(self) -> None:
        """`git check-ignore` runs with cwd at the project root, so a checkout
        without a repository errors rather than answering."""
        git_runs = [i for i in self.sandbox if i.startswith("RUN git ")]
        self.assertEqual(len(git_runs), 1)
        self.assertIn("git init", git_runs[0])
        self.assertIn("commit", git_runs[0])

    def test_the_health_sidecar_is_kept_off_a_published_port(self) -> None:
        """Sandbox ports are public URLs and /health, /ready and /metrics have no
        authentication, so the image that gets published binds loopback."""
        envs = [i for i in self.sandbox if i.startswith("ENV VA_LSE_HEALTH_HOST=")]
        self.assertEqual(len(envs), 1, "the sandbox stage must pin VA_LSE_HEALTH_HOST")
        address = envs[0].split("=", 1)[1].strip()
        self.assertIn(
            address,
            {"127.0.0.1", "::1", "localhost"},
            f"the sandbox sidecar binds {address!r}, which a published port can reach",
        )

    def test_its_command_does_not_set_a_port(self) -> None:
        """A *set* port is fatal under a development-layout Streamlit install
        ("server.port does not work when global.developmentMode is true"), which
        is what `pip install -e` or a vendored Streamlit in this box would be.
        8501 is already the default, so omitting it costs nothing."""
        commands = [i for i in self.sandbox if i.startswith(("CMD ", "ENTRYPOINT "))]
        self.assertEqual(len(commands), 1)
        self.assertIn("run_app.py", commands[0])
        self.assertNotIn("--server.port", commands[0])


class TestSandboxCarriesWhatTheSuiteReads(SandboxImageTestCase):
    #: Files and directories the offline suite resolves by path. Each one is a
    #: test that fails inside a sandbox without it.
    REQUIRED = (
        # the app itself, inherited from the runtime stage
        "app",
        "run_app.py",
        ".streamlit/config.toml",
        "requirements.lock",
        # this stage's reason to exist
        "tests",
        "scripts",
        "requirements-dev.txt",
        # tests/test_hermetic.py reads the workflow that schedules the fixtures
        ".github/workflows",
        # tests/test_monitoring_assets.py parses these, and asserts the paths the
        # compose file mounts exist on disk
        "deploy",
        "nginx",
        "docker-compose.yml",
        # mypy's settings, so `mypy app` in the box means what it means in CI
        "pyproject.toml",
        # tests/test_security_gitignore.py asks git about these
        ".gitignore",
        ".env.example",
        # these two are what the guards read and what the docs link to, so the
        # contract can be re-checked from inside the sandbox as well
        "Dockerfile",
        ".dockerignore",
        # sample inputs for a manual run
        "examples",
        # the OCR floor the sandbox stage installs by hand is this file's, so the
        # box can be compared against the local instructions without a network
        "requirements-local.txt",
    )

    def test_every_required_path_is_in_the_image(self) -> None:
        missing = [path for path in self.REQUIRED if not self.carries(path)]
        self.assertEqual(
            missing,
            [],
            "the sandbox image would not carry: " + ", ".join(missing),
        )

    def test_every_run_can_read_the_files_it_names(self) -> None:
        """A RUN cannot read what a later COPY brings in.

        `pip install -r requirements-dev.txt` named its file while nothing in the
        stage ever copied it in, and `test_every_required_path_is_in_the_image`
        passed anyway: a question about mentions is satisfied by a RUN line that
        mentions it. The build failed at that step the first time the stage was
        built — which is what the sandbox-image CI job is for — so this asks the
        ordering question instead.
        """
        missing = dockerfile.files_read_before_being_carried(self.sandbox, self.runtime)
        self.assertEqual(
            missing,
            [],
            "a RUN reads these before anything copies them in: " + ", ".join(missing),
        )

    def test_the_required_paths_still_exist(self) -> None:
        """A renamed directory must fail here, not silently stop being checked."""
        missing = [p for p in self.REQUIRED if not (PROJECT_ROOT / p).exists()]
        self.assertEqual(missing, [], "REQUIRED names something that is gone: " + ", ".join(missing))

    def test_every_root_page_is_copied(self) -> None:
        """tests/test_docs_structure.py resolves links and anchors across these,
        and names every script on one of them — so a new page that never reaches
        the image breaks the suite inside the sandbox."""
        pages = sorted(p.name for p in PROJECT_ROOT.glob("*.md"))
        self.assertTrue(pages, "no root pages found — the path must be wrong")
        missing = [page for page in pages if not dockerfile.carries(self.sandbox, page)]
        self.assertEqual(
            missing,
            [],
            "these pages are not copied into the sandbox image: " + ", ".join(missing),
        )

    def test_every_workflow_file_is_copied(self) -> None:
        workflows = sorted(
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in (PROJECT_ROOT / ".github" / "workflows").glob("*")
        )
        self.assertTrue(workflows, "no workflow files found — the path must be wrong")
        missing = [w for w in workflows if not dockerfile.carries(self.sandbox, w)]
        self.assertEqual(missing, [], "workflows left out of the image: " + ", ".join(missing))

    def test_no_copy_source_is_stale(self) -> None:
        """Every source must still match something on disk — Docker will not fail
        a build for a glob that matches nothing, so the omission would be silent."""
        stale = [
            source
            for source in dockerfile.copy_sources(self.sandbox)
            if not dockerfile.expand(source)
        ]
        self.assertEqual(stale, [], "these COPY sources match nothing on disk: " + ", ".join(stale))


class TestTheSandboxCanReadAScan(SandboxImageTestCase):
    """OCR is the reason this stage has a job the app cannot do.

    The app has no OCR dependency and never shells out, so an image-only page is
    counted and reported, never read (``scripts/ocr_records.py``). The sandbox is
    the machine that can read it, which makes the tooling and the entrypoint part
    of the image's contract rather than a convenience — and the *deployment* image
    must not gain either (see ``TestTheDeploymentImageIsUnchanged``).
    """

    BINARIES = ("tesseract-ocr", "tesseract-ocr-eng", "poppler-utils", "ghostscript", "qpdf")

    def test_the_ocr_binaries_are_installed(self) -> None:
        apt = " ".join(i for i in self.sandbox if i.startswith("RUN apt-get"))
        missing = [tool for tool in self.BINARIES if tool not in apt]
        self.assertEqual(
            missing,
            [],
            "scripts/ocr_records.py calls these: " + ", ".join(missing),
        )

    def test_ocrmypdf_is_installed_and_the_preferred_backend_survives(self) -> None:
        """The fallback (poppler + tesseract) works without it, but ocrmypdf is the
        one that keeps the original pages, so its absence must be a choice."""
        installs = " ".join(i for i in self.sandbox if i.startswith("RUN pip install"))
        self.assertIn("ocrmypdf", installs)

    def test_the_entrypoint_is_carried(self) -> None:
        self.assertTrue(
            self.carries("scripts/ocr_and_extract.py"),
            "the box installs OCR tooling but carries nothing that uses it",
        )

    def test_the_entrypoint_exists_and_names_that_file(self) -> None:
        """A renamed entrypoint must fail here, not inside a sandbox."""
        self.assertTrue((PROJECT_ROOT / "scripts" / "ocr_and_extract.py").is_file())


class TestTheDeploymentImageIsUnchanged(SandboxImageTestCase):
    def test_it_stays_non_root(self) -> None:
        self.assertEqual(self.last_user(self.runtime), "nobody")

    def test_it_does_not_install_the_dev_extras(self) -> None:
        self.assertFalse(
            any("requirements-dev" in i for i in self.runtime),
            "the deployed image carries no test tooling — that is a deliberate choice",
        )

    def test_it_does_not_grow_an_ocr_engine(self) -> None:
        """The deployment must not carry a PDF renderer and an OCR engine: it has
        no code path that would call them, and the app's own docs say OCR happens
        before the upload. The sandbox stage is where they belong."""
        apt = " ".join(i for i in self.runtime if i.startswith("RUN apt-get"))
        for tool in ("tesseract", "poppler", "ghostscript", "qpdf"):
            self.assertNotIn(tool, apt, f"the deployed image now installs {tool}")
        installs = " ".join(i for i in self.runtime if i.startswith("RUN pip install"))
        self.assertNotIn("ocrmypdf", installs)

    def test_it_does_not_carry_the_tests_or_scripts(self) -> None:
        for path in ("tests", "scripts"):
            with self.subTest(path=path):
                self.assertFalse(
                    dockerfile.carries(self.runtime, path),
                    f"the deployment image now carries {path}: say so in its comments, "
                    "or take it out",
                )

    def test_its_command_and_ports_are_the_deployment_contract(self) -> None:
        command = [i for i in self.runtime if i.startswith("CMD ")][0]
        self.assertIn("--server.port=8501", command)
        self.assertIn("--server.address=0.0.0.0", command)
        exposes = [i for i in self.runtime if i.startswith("EXPOSE ")][0]
        self.assertIn("8501", exposes)
        self.assertIn("8001", exposes)
