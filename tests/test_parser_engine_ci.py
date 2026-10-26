"""Unit tests for the synthetic parser daemon's CI startup probe.

The real probe starts a second Docker daemon with mutual TLS on a disposable CI
runner. These tests cover the identity wait it performs: a daemon that is still
coming up must be polled, not mistaken for the application's daemon, and a
genuine identity mismatch has to name both engines instead of failing on a bare
assertion.
"""
import contextlib
import io
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

from scripts import parser_engine_ci  # noqa: E402

ENV = {"DOCKER_HOST": "tcp://10.73.0.1:2376"}


class _Clock:
    """Monotonic clock whose sleep() advances it, so the wait loop terminates."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _DockerInfo:
    """Fake ``docker info``: replays replies, repeating the last one."""

    def __init__(self, *replies) -> None:  # noqa: ANN002
        self._replies = list(replies)
        self.calls = []

    def __call__(self, args, **kwargs):  # noqa: ANN001, ANN003
        self.calls.append((list(args), kwargs))
        reply = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        return reply


def _answered(engine_id: bytes):
    return subprocess.CompletedProcess(["docker", "info"], 0, stdout=engine_id, stderr=b"")


def _refused(stderr: bytes = b"Cannot connect to the Docker daemon"):
    return subprocess.CompletedProcess(["docker", "info"], 1, stdout=b"", stderr=stderr)


class TestWaitForDistinctEngine(unittest.TestCase):
    def _wait(self, replies, *, application_id="application-engine", timeout=4.0):  # noqa: ANN001
        clock = _Clock()
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            engine_id = parser_engine_ci.wait_for_distinct_engine(
                ENV,
                application_id,
                timeout=timeout,
                clock=clock,
                sleep=clock.sleep,
                run=replies,
            )
        return engine_id, replies.calls, captured.getvalue()

    def test_accepts_the_dedicated_engine(self) -> None:
        engine_id, calls, _ = self._wait(_DockerInfo(_answered(b"parser-engine\n")))
        self.assertEqual(engine_id, "parser-engine")
        self.assertEqual(len(calls), 1)
        # The probe must ask the dedicated endpoint, not the runner's daemon.
        self.assertEqual(calls[0][1]["env"], ENV)

    def test_retries_while_the_reply_names_the_application_engine(self) -> None:
        replies = _DockerInfo(
            _answered(b"application-engine\n"),
            _answered(b"application-engine\n"),
            _answered(b"parser-engine\n"),
        )
        engine_id, calls, output = self._wait(replies)
        self.assertEqual(engine_id, "parser-engine")
        self.assertEqual(len(calls), 3)
        self.assertIn("not distinct yet", output)
        self.assertEqual(output.count("not distinct yet"), 1)

    def test_retries_while_the_daemon_is_unreachable(self) -> None:
        replies = _DockerInfo(_refused(), _refused(), _answered(b"parser-engine\n"))
        engine_id, calls, _ = self._wait(replies)
        self.assertEqual(engine_id, "parser-engine")
        self.assertEqual(len(calls), 3)

    def test_empty_engine_id_is_not_accepted(self) -> None:
        replies = _DockerInfo(_answered(b"\n"), _answered(b"parser-engine\n"))
        engine_id, calls, _ = self._wait(replies)
        self.assertEqual(engine_id, "parser-engine")
        self.assertEqual(len(calls), 2)

    def test_missing_engine_identity_is_reported_with_both_ids(self) -> None:
        replies = _DockerInfo(_answered(b"application-engine\n"))
        with self.assertRaises(RuntimeError) as raised:
            self._wait(replies, timeout=2.0)
        message = str(raised.exception)
        self.assertIn("did not become a distinct engine", message)
        self.assertIn(ENV["DOCKER_HOST"], message)
        self.assertIn("application-engine", message)
        self.assertIn("attempt", message)
        # It must exhaust the window rather than fail on the first reply.
        self.assertGreater(len(replies.calls), 1)

    def test_unreachable_daemon_reports_the_transport_error(self) -> None:
        replies = _DockerInfo(_refused())
        with self.assertRaises(RuntimeError) as raised:
            self._wait(replies, timeout=1.0)
        self.assertIn("Cannot connect to the Docker daemon", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
