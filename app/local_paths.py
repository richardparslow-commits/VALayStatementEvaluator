"""Loopback binding — the gate on ``VA_LSE_ALLOW_LOCAL_PATHS``.

Why argv, not Streamlit
-----------------------
``VA_LSE_ALLOW_LOCAL_PATHS=1`` lets the app read arbitrary local files. On a
server bound to a non-loopback interface that flag would let anyone who can
reach the server read files from its disk through the records widget, so the
flag only counts when the server is provably bound to loopback.

The app must not decide from Streamlit's option API (``get_option``) for
this decision: app code reading Streamlit's ambient configuration is forbidden
by the suite (``tests/test_hermetic.py``, the Streamlit-option rule). Reading the committed
``.streamlit/config.toml`` directly is the prescribed escape hatch — but a
``--server.address`` flag passed at launch beats that file, and flags are not
visible in it. Streamlit's ScriptRunner never rewrites ``sys.argv`` (verified
against streamlit 1.63/1.64 sources: the script sees the true launch argv), so
argv is the authoritative source: the documented launch recipes all pass
``--server.address 127.0.0.1`` as a flag. The committed config file backs that
up for launches that set it there, and an absent value means Streamlit binds
every interface — refused, with a restart command.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

_LOOPBACK_BINDS = frozenset({"127.0.0.1", "localhost", "::1", "::1%lo0"})

_FLAG_FORMS = ("--server.address", "--server-address")

_CONFIG_PATH = Path(__file__).resolve().parents[2] / ".streamlit" / "config.toml"


def is_loopback(address: str | None) -> bool:
    """Whether a bind address keeps the server on the loopback interface.

    ``None``/empty — Streamlit's default, every interface — is not loopback.
    """
    return address in _LOOPBACK_BINDS


def _address_from_argv(argv: list[str] | None = None) -> str | None:
    """The ``--server.address`` flag value from the process's own argv."""
    argv = sys.argv if argv is None else argv
    for form in _FLAG_FORMS:
        prefix = f"{form}="
        for arg in argv:
            if arg == form and argv.index(arg) + 1 < len(argv):
                return argv[argv.index(arg) + 1]
            if arg.startswith(prefix):
                return arg[len(prefix) :]
    return None


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
    """The address this server was launched to bind, or None for all interfaces.

    Launch flags beat the committed config file, matching Streamlit's own
    precedence. Missing everywhere means Streamlit listens on every interface.
    """
    return _address_from_argv() or _address_from_config()
