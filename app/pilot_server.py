"""Pilot process entrypoint: start privacy cleanup before any browser connects."""
from __future__ import annotations

import os
import sys
from typing import Sequence


def initialize() -> None:
    from . import pilot
    from .log_retention import retention_days
    if not pilot.enabled():
        raise pilot.PilotBlocked("This server launcher requires controlled-pilot mode.")
    # No OIDC/user context or provider request is needed to clean local logs.
    # Full account/revision/identity acceptance remains at screen admission.
    pilot.validate_log_policy({"local_log_retention_days": retention_days()})
    from .shutdown import install_signal_handlers
    install_signal_handlers()
    if os.getenv("VA_LSE_HEALTH_PORT", "").strip() != "0":
        from .health import start_health_server
        if start_health_server() is None:
            raise pilot.PilotBlocked("The pilot health server could not start.")


def main(argv: Sequence[str] | None = None) -> None:
    initialize()  # Deliberately fail startup if retention cannot be enforced.
    from streamlit.web.cli import main as streamlit_main
    sys.argv = ["streamlit", "run", "run_app.py", *(sys.argv[1:] if argv is None else argv)]
    streamlit_main()


if __name__ == "__main__":
    main()
