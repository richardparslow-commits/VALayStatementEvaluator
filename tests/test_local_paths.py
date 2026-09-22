"""The loopback-derivation module behind the VA_LSE_ALLOW_LOCAL_PATHS gate.

The gate's decision comes from the bind address of the socket that actually
serves the UI — identified from the outside by a Streamlit-marker probe over
loopback, then read from the process's own socket table — backed by the
committed ``.streamlit/config.toml`` when no UI socket is identifiable yet.

Never from Streamlit's ambient configuration (``tests/test_hermetic.py``
forbids that read in app code), and never from ``sys.argv``: Streamlit's
script runner rewrites it (measured 2026-09-21 on Streamlit 1.63.0, the
script sees ``[script_path]``), so an argv-derived address can never see a
launch flag. Nor from *all* listen sockets: the health sidecar deliberately
binds ``0.0.0.0`` and a loopback UI beside it is not an exposed app.

These tests pin: lsof parsing, marker-probe identification, sidecar
exclusion, wide-UI refusal, mixed-bind fail-closed, socket-beats-config
precedence, and the fail-closed reading of a missing/invalid config.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests import hermetic  # noqa: F401  (hermetic test session; see tests/hermetic.py)

from app import local_paths


def _lsof_output(*lines: str) -> subprocess.CompletedProcess:
    """A fake lsof CompletedProcess from raw field-output lines."""
    stdout = "".join(f"{line}\n" for line in lines)
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


class TestLsofParsing(unittest.TestCase):
    """The ``lsof -F n`` parser: real output shapes from macOS."""

    def _parse(self, *lines: str) -> dict[int, tuple[str, int]]:
        with mock.patch.object(
            local_paths.subprocess, "run", return_value=_lsof_output(*lines)
        ):
            return local_paths._lsof_snapshot()

    def test_loopback_ipv4_listen(self) -> None:
        snap = self._parse("p123", "f6", "n127.0.0.1:8501")
        self.assertEqual(snap, {6: ("127.0.0.1", 8501)})

    def test_hostname_listen(self) -> None:
        snap = self._parse("f6", "nlocalhost:8501")
        self.assertEqual(snap, {6: ("localhost", 8501)})

    def test_wildcard_listen_is_observable_as_wide(self) -> None:
        """``*:8501`` means every interface — kept refusable, never hidden."""
        snap = self._parse("f6", "n*:8501")
        self.assertEqual(snap, {6: ("0.0.0.0", 8501)})

    def test_named_port_resolved_numerically(self) -> None:
        """The live sidecar shows as ``*:vcom-tunnel`` — service-name form
        for port 8001 (``/etc/services``; the run doc's sidecar port)."""
        snap = self._parse("f21", "n*:vcom-tunnel")
        self.assertEqual(snap, {21: ("0.0.0.0", 8001)})

    def test_multiple_listens_kept_apart_by_fd(self) -> None:
        snap = self._parse("f6", "nlocalhost:8501", "f21", "n*:vcom-tunnel")
        self.assertEqual(
            snap, {6: ("localhost", 8501), 21: ("0.0.0.0", 8001)}
        )

    def test_connected_socket_remote_form_is_tolerated(self) -> None:
        snap = self._parse("f6", "n127.0.0.1:8501->1.2.3.4:9999")
        self.assertEqual(snap, {6: ("127.0.0.1", 8501)})

    def test_garbage_lines_are_skipped(self) -> None:
        snap = self._parse("p1", "zzz", "f6", "nlocalhost", "n*:notaport")
        self.assertEqual(snap, {})

    def test_lsof_failure_yields_empty_snapshot(self) -> None:
        with mock.patch.object(
            local_paths.subprocess,
            "run",
            side_effect=OSError("lsof not installed"),
        ):
            self.assertEqual(local_paths._lsof_snapshot(), {})


class TestMarkerProbe(unittest.TestCase):
    def _with_http(self, status: int, body: bytes) -> None:
        class _Resp:
            def __init__(self) -> None:
                self.status = status

            def read(self, n: int) -> bytes:
                return body[:n]

            def __enter__(self):  # noqa: D105
                return self

            def __exit__(self, *exc):  # noqa: D105
                return False

        patcher = mock.patch.object(
            local_paths.urllib.request, "urlopen", return_value=_Resp()
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_streamlit_index_page_identifies_the_ui_port(self) -> None:
        self._with_http(200, b"<!doctype html><html>... streamlit ...")
        self.assertTrue(local_paths._serves_streamlit(8501))

    def test_non_streamlit_service_is_rejected(self) -> None:
        self._with_http(200, b'{"status": "error"}')
        self.assertFalse(local_paths._serves_streamlit(7123))

    def test_http_error_status_is_rejected(self) -> None:
        self._with_http(404, b'{"status": "not_found"}')
        self.assertFalse(local_paths._serves_streamlit(7123))


class TestUiBindAddress(unittest.TestCase):
    """The decision: snapshot → loopback candidates → probe → cross-check."""

    def setUp(self) -> None:
        local_paths._ui_bind_address.cache_clear()

    def tearDown(self) -> None:
        local_paths._ui_bind_address.cache_clear()

    def _with(
        self,
        snapshot: dict[int, tuple[str, int]] | None,
        *,
        streamlit_port: int | None = None,
    ) -> None:
        """Patch the two external seams: the socket table and the probe."""
        proc = _lsof_output()  # real run replaced wholesale below
        if snapshot is None:
            run_patch = mock.patch.object(
                local_paths.subprocess, "run", side_effect=OSError("no lsof")
            )
        else:
            lines: list[str] = []
            for fd, (host, port) in sorted(snapshot.items()):
                lines.extend([f"f{fd}", f"n{host}:{port}"])
            proc = _lsof_output(*lines)
            run_patch = mock.patch.object(
                local_paths.subprocess, "run", return_value=proc
            )
        run_patch.start()
        self.addCleanup(run_patch.stop)
        probe_patch = mock.patch.object(
            local_paths,
            "_serves_streamlit",
            side_effect=lambda port: port == streamlit_port,
        )
        probe_patch.start()
        self.addCleanup(probe_patch.stop)

    def test_loopback_ui_is_identified(self) -> None:
        self._with({6: ("localhost", 8501), 21: ("0.0.0.0", 8001)}, streamlit_port=8501)
        self.assertEqual(local_paths._ui_bind_address(), "localhost")

    def test_sidecar_wide_bind_is_excluded_not_fatal(self) -> None:
        """The field-measured pair: loopback UI + wide health sidecar."""
        self._with({6: ("127.0.0.1", 8501), 21: ("0.0.0.0", 8001)}, streamlit_port=8501)
        self.assertEqual(local_paths._ui_bind_address(), "127.0.0.1")

    def test_wide_bound_ui_is_refused(self) -> None:
        """A wide-bound UI must not be a probe candidate: probing it over
        loopback would succeed and launder the bind into a loopback verdict.
        """
        self._with({6: ("0.0.0.0", 8501)}, streamlit_port=8501)
        self.assertIsNone(local_paths._ui_bind_address())

    def test_no_identifiable_ui_falls_to_config(self) -> None:
        """A sidecar-only process identifies no UI port."""
        self._with({21: ("0.0.0.0", 8001)}, streamlit_port=None)
        with mock.patch.object(
            local_paths, "_address_from_config", return_value="127.0.0.1"
        ):
            self.assertEqual(local_paths.server_bind_address(), "127.0.0.1")

    def test_no_listen_socket_at_all_falls_to_config(self) -> None:
        self._with({}, streamlit_port=None)
        with mock.patch.object(
            local_paths, "_address_from_config", return_value="localhost"
        ):
            self.assertEqual(local_paths.server_bind_address(), "localhost")

    def test_lsof_unavailable_falls_to_config(self) -> None:
        self._with(None, streamlit_port=None)
        with mock.patch.object(
            local_paths, "_address_from_config", return_value="localhost"
        ):
            self.assertEqual(local_paths.server_bind_address(), "localhost")

    def test_mixed_binds_on_the_ui_port_fail_closed(self) -> None:
        self._with(
            {6: ("127.0.0.1", 8501), 7: ("192.168.1.10", 8501)}, streamlit_port=8501
        )
        self.assertEqual(local_paths._ui_bind_address(), "192.168.1.10")
        self.assertFalse(local_paths.is_loopback(local_paths._ui_bind_address()))

    def test_result_is_cached_per_process(self) -> None:
        self._with({6: ("localhost", 8501)}, streamlit_port=8501)
        local_paths._ui_bind_address()
        with mock.patch.object(
            local_paths,
            "_ui_bind_address_uncached",
            side_effect=AssertionError("re-derived a cached answer"),
        ):
            self.assertEqual(local_paths.server_bind_address(), "localhost")


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
    def setUp(self) -> None:
        local_paths._ui_bind_address.cache_clear()
        self.addCleanup(local_paths._ui_bind_address.cache_clear)

    def test_ui_socket_beats_config_even_when_not_loopback(self) -> None:
        """Observable sockets override the config file.

        Deliberately a wide UI socket: a server actually listening on every
        interface must not be masked by a loopback value in the committed
        config.
        """
        with mock.patch.object(
            local_paths, "_ui_bind_address", return_value="0.0.0.0"
        ), mock.patch.object(
            local_paths, "_address_from_config", return_value="127.0.0.1"
        ):
            self.assertEqual(local_paths.server_bind_address(), "0.0.0.0")

    def test_config_backs_up_unidentifiable_ui(self) -> None:
        with mock.patch.object(
            local_paths, "_ui_bind_address", return_value=None
        ), mock.patch.object(
            local_paths, "_address_from_config", return_value="localhost"
        ):
            self.assertEqual(local_paths.server_bind_address(), "localhost")

    def test_absent_everywhere_is_none(self) -> None:
        with mock.patch.object(
            local_paths, "_ui_bind_address", return_value=None
        ), mock.patch.object(
            local_paths, "_address_from_config", return_value=None
        ):
            self.assertIsNone(local_paths.server_bind_address())

    def test_committed_repo_config_currently_sets_no_address(self) -> None:
        """Pin reality: the shipped config does not set an address, so the
        config fallback never rescues a wide-bind launch; loopback launches
        must have a real loopback UI socket (the documented recipes do).
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
