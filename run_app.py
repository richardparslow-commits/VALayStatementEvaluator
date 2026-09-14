"""Launcher for the Streamlit app.

Run from the project root with:
    streamlit run run_app.py

The health sidecar (GET /health, GET /ready) is started before Streamlit
takes over so liveness/readiness probes work even during Streamlit startup.
Set VA_LSE_HEALTH_PORT=0 to disable the sidecar.

Graceful shutdown handlers (SIGTERM/SIGINT) are installed in the main thread
so Kubernetes pod termination (SIGTERM → grace → SIGKILL) allows in-flight
Evaluate/Draft runs to finish cleanly.  See app/shutdown.py.
"""
from __future__ import annotations

import os


def _maybe_start_health_server() -> None:
    raw = os.getenv("VA_LSE_HEALTH_PORT", "").strip()
    if raw == "0":
        return
    try:
        from app.health import start_health_server

        start_health_server()
    except Exception:  # noqa: BLE001 - health is best-effort, never block the app
        pass


_maybe_start_health_server()

# Install SIGTERM/SIGINT handlers on the main thread *before* Streamlit
# takes over its event loop, so the signals are delivered promptly on
# container stop.  See app/shutdown.py for the drain logic.
try:
    from app.shutdown import install_signal_handlers

    install_signal_handlers()
except Exception:  # noqa: BLE001 - shutdown is best-effort, never block the app
    pass

from app.main import main  # noqa: E402  # health sidecar must start first

main()
