"""Supported Streamlit ASGI entrypoint; pilot_server initializes before listening."""
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncIterator

import streamlit as st

from app.export_routes import routes
from app.text_exports import STORE


@asynccontextmanager
async def lifespan(app: st.App) -> AsyncIterator[None]:
    try:
        yield
    finally:
        STORE.close()


app = st.App(Path(__file__).resolve().parent.parent / "run_app.py", routes=routes(), lifespan=lifespan)
