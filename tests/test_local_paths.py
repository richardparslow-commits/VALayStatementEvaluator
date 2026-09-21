"""The loopback-derivation module behind the VA_LSE_ALLOW_LOCAL_PATHS gate.

The gate's decision must come from the process's own argv backed by the
committed ``.streamlit/config.toml`` — never from Streamlit's ambient
configuration (``tests/test_hermetic.py`` forbids that read in app code).
These tests pin the derivation: flag forms, flag-beats-config precedence,
and the fail-closed reading of a missing/invalid config.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests import hermetic  # noqa: F401  (hermetic test session; see tests/hermetic.py)

from app import local_paths


class TestAddressFromArgv(unittest.TestCase):
    def test_space_form_is_read(self) -> None:
        argv = ["streamlit", "run", "run_app.py", "--server.address", "127.0.0.1"]
        self.assertEqual(local_paths._address_from_argv(argv), "127.0.0.1")

    def test_equals_form_is_read(self) -> None:
        argv = ["streamlit", "run", "run_app.py", "--server.address=127.0.0.1"]
        self.assertEqual(local_paths._address_from_argv(argv), "127.0.0.1")

    def test_underscore_form_is_read(self) -> None:
        argv = ["streamlit", "run", "run_app.py", "--server-address", "::1"]
        self.assertEqual(local_paths._address_from_argv(argv), "::1")

    def test_absent_flag_is_none(self) -> None:
        self.assertIsNone(local_paths._address_from_argv(["streamlit", "run", "run_app.py"]))

    def test_flag_without_a_value_is_none(self) -> None:
        argv = ["streamlit", "run", "run_app.py", "--server.port", "8501", "--server.address"]
        self.assertIsNone(local_paths._address_from_argv(argv))

    def test_first_flag_wins(self) -> None:
        argv = ["streamlit", "run", "--server.address", "127.0.0.1", "--server.address", "0.0.0.0"]
        self.assertEqual(local_paths._address_from_argv(argv), "127.0.0.1")


class TestAddressFromConfig(unittest.TestCase):
    def _read(self, text: str) -> str | None:
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "config.toml"
            path.write_text(text, encoding="utf-8")
            return local_paths._address_from_config(path)

    def test_server_address_is_read(self) -> None:
        self.assertEqual(self._read('[server]\naddress = "127.0.0.1"\n'), "127.0.0.1")

    def test_missing_address_is_none(self) -> None:
        self.assertIsNone(self._read("[server]\nheadless = true\n"))

    def test_empty_address_is_none(self) -> None:
        self.assertIsNone(self._read('[server]\naddress = ""\n'))

    def test_invalid_toml_is_none(self) -> None:
        self.assertIsNone(self._read("not [valid toml"))

    def test_missing_file_is_none(self) -> None:
        self.assertIsNone(local_paths._address_from_config(Path("/nonexistent/config.toml")))


class TestServerBindAddressPrecedence(unittest.TestCase):
    def test_flag_beats_config_even_when_flag_is_not_loopback(self) -> None:
        """Launch flags override the config file, matching Streamlit's precedence.

        Deliberately a non-loopback flag: a launch that explicitly binds wide
        must not be masked by a loopback value left in the committed config.
        """
        with mock.patch.object(
            local_paths, "_address_from_argv", return_value="0.0.0.0"
        ), mock.patch.object(
            local_paths, "_address_from_config", return_value="127.0.0.1"
        ):
            self.assertEqual(local_paths.server_bind_address(), "0.0.0.0")

    def test_config_backs_up_absent_flag(self) -> None:
        with mock.patch.object(
            local_paths, "_address_from_argv", return_value=None
        ), mock.patch.object(
            local_paths, "_address_from_config", return_value="localhost"
        ):
            self.assertEqual(local_paths.server_bind_address(), "localhost")

    def test_absent_everywhere_is_none(self) -> None:
        with mock.patch.object(
            local_paths, "_address_from_argv", return_value=None
        ), mock.patch.object(
            local_paths, "_address_from_config", return_value=None
        ):
            self.assertIsNone(local_paths.server_bind_address())

    def test_committed_repo_config_currently_sets_no_address(self) -> None:
        """Pin reality: the shipped config does not set an address, so loopback
        launches must pass the flag (the documented recipes do). If someone
        adds ``[server] address`` to the committed file, this test documents
        that the config fallback becomes the decision for flagless launches.
        """
        address = local_paths._address_from_config()
        self.assertIn(address, (None, "127.0.0.1", "localhost", "::1"))


class TestIsLoopback(unittest.TestCase):
    def test_loopback_binds_pass(self) -> None:
        for address in ("127.0.0.1", "localhost", "::1", "::1%lo0"):
            with self.subTest(address=address):
                self.assertTrue(local_paths.is_loopback(address))

    def test_wide_and_missing_binds_fail_closed(self) -> None:
        for address in (None, "", "0.0.0.0", "::", "example.com", "127.0.0.2"):
            with self.subTest(address=address):
                self.assertFalse(local_paths.is_loopback(address))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
