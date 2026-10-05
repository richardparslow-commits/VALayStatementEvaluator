"""Pilot process entrypoint: start privacy cleanup before any browser connects."""
from __future__ import annotations

import os
import sys
from typing import Sequence


def initialize() -> None:
    from .privacy_logging import install
    install()  # Before approval, retention, SDK or Streamlit startup logging.
    from . import pilot
    from .log_retention import retention_days
    if not pilot.enabled():
        raise pilot.PilotBlocked("This server launcher requires controlled-pilot mode.")
    from .upload_temp import validate as validate_upload_temp
    validate_upload_temp()
    from .private_uploads import install as install_private_uploads
    install_private_uploads()
    from importlib.metadata import version
    if version("streamlit") != "1.63.0":
        raise pilot.PilotBlocked("Upload admission requires the reviewed Streamlit release.")
    from urllib.parse import urlsplit
    approval = pilot.load_approval()
    if urlsplit(pilot.https_url(approval["deployment_url"])).path not in ("", "/"):
        raise pilot.PilotBlocked("Pilot upload admission requires a root deployment URL.")
    # No OIDC/user context or provider request is needed to clean local logs.
    # Full account/revision/identity acceptance remains at screen admission.
    pilot.validate_log_policy({"local_log_retention_days": retention_days()})
    from .pilot_budget import get_ledger
    get_ledger(approval)  # Hold the single-process lock before listening.
    if os.getenv("VA_LSE_PILOT_TEXT_EXPORTS", "0") == "1":
        from importlib.metadata import version
        from .text_exports import STORE, ExportUnavailable, policy_binding
        try:
            policy_binding(pilot.load_approval())
            if version("streamlit") != "1.63.0":
                raise ExportUnavailable("Unsupported signed-cookie protocol.")
        except ExportUnavailable as exc:
            raise pilot.PilotBlocked("Text exports require separately accepted exact-release evidence.") from exc
        STORE.start()
    from .shutdown import install_signal_handlers
    install_signal_handlers()
    if os.getenv("VA_LSE_HEALTH_PORT", "").strip() != "0":
        from .health import start_health_server
        if start_health_server() is None:
            raise pilot.PilotBlocked("The pilot health server could not start.")


def main(argv: Sequence[str] | None = None) -> None:
    initialize()  # Deliberately fail startup if retention cannot be enforced.
    from streamlit.web.cli import main as streamlit_main
    sys.argv = ["streamlit", "run", "app/pilot_asgi.py", *(sys.argv[1:] if argv is None else argv)]
    streamlit_main()


if __name__ == "__main__":
    main()
