"""Loopback binding — the gate on ``VA_LSE_ALLOW_LOCAL_PATHS``.

Why the socket table, not Streamlit
-----------------------------------
``VA_LSE_ALLOW_LOCAL_PATHS=1`` lets the app read arbitrary local files. On a
server bound to a non-loopback interface that flag would let anyone who can
reach the server read files from its disk through the records widget, so the
flag only counts when the server is provably bound to loopback.

Two earlier derivations failed in the field, both times refusing the
*positive* path on a legitimate loopback launch:

1. Launch argv — on the belief (never checked on the right code path) that
   Streamlit's ScriptRunner passes the launch argv through. Measured
   2026-09-21 on Streamlit 1.63.0: the script thread sees
   ``sys.argv == [script_path]``, so a flag-derived address is always
   ``None``.
2. All listen sockets — the process's full socket table includes the health
   sidecar (``app/health.py``), which deliberately binds ``0.0.0.0`` (its
   contract: Kubernetes health probes come over the network, not loopback).
   A loopback Streamlit plus a wide sidecar is *not* an exposed app, but a
   naive "any wide listen fails the gate" rule refused exactly that pair —
   measured live as ``localhost:8501`` + ``*:vcom-tunnel`` (7123).

The app must not read Streamlit's ambient configuration either
(``tests/test_hermetic.py``, the Streamlit-option rule), so the decision now
comes from the socket that actually serves the UI, identified from the
outside: each loopback TCP port this process listens on gets one
Streamlit-marker HTTP probe (HTTP 200 + "streamlit" in the body — the
standard Streamlit index page). The identified port's bind address, read
from the same ``lsof`` snapshot, is the gate's ground truth; a wide-bound
UI still carries a ``*`` address there and fails. That is the same
identification a Docker healthcheck performs on every container, so it
adds no new trust assumption.

Ports with no streamlit marker (the sidecar, debuggers, future auxiliaries)
are excluded from the decision — their binds are their own subsystems'
contracts, documented in their modules. No identifiable UI socket (startup
race, exotic platform) falls back to the committed ``.streamlit/config.toml``
``[server] address``, and an address absent from both means fail-closed:
local imports stay disabled, with a warning naming the restart command.
"""

from __future__ import annotations

import functools
import subprocess
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

_LOOPBACK_BINDS = frozenset({"127.0.0.1", "localhost", "::1", "::1%lo0"})

_CONFIG_PATH = Path(__file__).resolve().parents[2] / ".streamlit" / "config.toml"

#: Size of the marker probe's read. The standard Streamlit index page embeds
#: the product name well within a kilobyte; more is wasted work.
_MARKER_READ_BYTES = 4096

#: One HTTP probe per candidate loopback port, bounded so a hung port costs
#: the render a fraction of a second, not a script timeout.
_MARKER_TIMEOUT_SECONDS = 2.0


def is_loopback(address: str | None) -> bool:
    """Whether a bind address keeps the server on the loopback interface.

    ``None``/empty — Streamlit's default, every interface — is not loopback.
    """
    return address in _LOOPBACK_BINDS


@functools.lru_cache(maxsize=1)
def _ui_bind_address() -> str | None:
    """The bind address of the socket that actually serves the UI.

    ``None`` means "could not determine" — no identifiable UI socket (startup
    race, lsof unavailable, no loopback listener). The caller falls back to
    the committed config in that case, keeping the gate fail-closed.

    Cached for the process lifetime: the gate decides per render, and a
    running server does not rebind its UI port mid-life.
    """
    return _ui_bind_address_uncached()


def _ui_bind_address_uncached() -> str | None:
    """The uncached decision: snapshot sockets, probe, pick, cross-check."""
    snapshot = _lsof_snapshot()
    if not snapshot:
        return None
    # Candidates: loopback-only listens. A wide-bound UI must be refused, so
    # it must NOT be a probe candidate: probing ``*:8501`` over 127.0.0.1
    # would succeed and wrongly launder a wide bind into a loopback verdict.
    loopback_ports = sorted(
        {port for addr, port in snapshot.values() if addr in _LOOPBACK_BINDS}
    )
    ui_port = next((port for port in loopback_ports if _serves_streamlit(port)), None)
    if ui_port is None:
        return None
    addrs = {addr for addr, port in snapshot.values() if port == ui_port}
    if len(addrs) == 1:
        return next(iter(addrs))
    # Same port on several addresses (e.g. a v6-dual listener): fail-closed
    # with a truthful value — mixed binds on the UI port are not local-only.
    non_loopback = sorted(a for a in addrs if a not in _LOOPBACK_BINDS)
    return non_loopback[0] if non_loopback else "localhost"


def _lsof_snapshot() -> dict[int, tuple[str, int]]:
    """This process's TCP listen sockets: ``{fd: (host, port)}``.

    Parsed from ``lsof -a -p <pid> -iTCP -sTCP:LISTEN -F n`` field output —
    ``f<fd>`` lines followed by ``n<host>:<port>`` (``->`` remote forms are
    connected sockets; LISTEN has no remote, but parse past it anyway).
    """
    import os

    try:
        proc = subprocess.run(
            ["lsof", "-a", "-p", str(os.getpid()), "-iTCP", "-sTCP:LISTEN", "-F", "n"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    snapshot: dict[int, tuple[str, int]] = {}
    fd = -1
    for line in proc.stdout.splitlines():
        if line.startswith("f"):
            fd = int(line[1:])
            continue
        if not line.startswith("n") or fd < 0:
            continue
        spec = line[1:].split("->", 1)[0]
        host, sep, port_raw = spec.rpartition(":")
        if not sep:
            continue
        if port_raw.isdigit():
            port = int(port_raw)
        else:
            # Service-name form (``*:vcom-tunnel`` — lsof resolves the port
            # when it can, names it otherwise); resolve back via the
            # services database, skipping names it does not know.
            try:
                import socket

                port = socket.getservbyname(port_raw, "tcp")
            except OSError:
                continue
        if host == "*":
            # lsof's mark for "every interface" — kept observable as 0.0.0.0
            # so a wide bind is refusable, never invisible.
            host = "0.0.0.0"
        snapshot[fd] = (host, port)
    return snapshot


def _serves_streamlit(port: int) -> bool:
    """Whether ``127.0.0.1:<port>`` serves the standard Streamlit index page.

    HTTP 200 with "streamlit" in the first bytes of the body. Deliberately
    over loopback: a UI port wide-bound on every interface is still
    reachable at 127.0.0.1, so identification succeeds; the *bind* is then
    read from the socket table, where the wide bind is visible, and fails.
    """
    url = f"http://127.0.0.1:{port}/"
    try:
        with urllib.request.urlopen(url, timeout=_MARKER_TIMEOUT_SECONDS) as resp:
            if resp.status != 200:
                return False
            body = resp.read(_MARKER_READ_BYTES)
    except (urllib.error.URLError, OSError, ValueError):
        return False
    return b"streamlit" in body.lower()


def _address_from_config(path: Path | None = None) -> str | None:
    """``[server] address`` from the committed config, read by absolute path.

    Read by absolute path (not Streamlit's cwd resolution) so a developer
    running from another directory still gets this repo's committed value.
    """
    try:
        with (path or _CONFIG_PATH).open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    address = data.get("server", {}).get("address")
    return str(address) if address not in (None, "") else None


def server_bind_address() -> str | None:
    """The address the UI server listens on, or None for all interfaces.

    The UI socket's bind address is the ground truth; the committed config
    file backs it up when no UI socket is identifiable yet. ``None`` means
    "cannot prove loopback", which the gate treats as wide-bind.
    """
    sockets = _ui_bind_address()
    if sockets is not None:
        return sockets
    return _address_from_config()
