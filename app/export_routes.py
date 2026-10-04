"""Per-request authenticated TXT route; no media URLs, redirects or bearer fallback."""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from . import pilot
from .export_cookie import claims
from .text_exports import STORE, policy_binding

HEADERS = {"Cache-Control": "private, no-store, max-age=0", "Pragma": "no-cache", "Expires": "0",
           "Vary": "Cookie", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
           "Content-Security-Policy": "default-src 'none'; sandbox", "Content-Encoding": "identity"}


def _content(request: Request) -> bytes:
    from .shutdown import is_shutting_down
    if not STORE.active or is_shutting_down() or request.method != "GET" or request.url.query or request.headers.get("range"):
        raise ValueError
    handle = request.path_params["handle"]
    if not re.fullmatch(r"[0-9a-f]{64}", handle):
        raise ValueError
    approval = pilot.load_approval()
    policy_binding(approval)
    origin = pilot.https_url(approval["deployment_url"])
    parsed = urlsplit(origin)
    origin = f"https://{parsed.netloc}"
    if (request.headers.get("host") != parsed.netloc
            or (request.url.scheme != "https" and request.headers.get("x-forwarded-proto") != "https")
            or request.headers.get("sec-fetch-site", "same-origin") not in ("same-origin", "none")
            or request.headers.get("origin", origin) != origin):
        raise ValueError
    import streamlit as st
    auth = st.secrets["auth"]
    if (auth["redirect_uri"] != origin + "/oauth2callback"
            or auth["server_metadata_url"] != pilot.https_url(approval["issuer"]) + "/.well-known/openid-configuration"):
        raise ValueError
    identity = claims(request.headers.getlist("cookie"), auth["cookie_secret"], origin)
    owner = pilot.authorized_identity(identity, approval)
    return STORE.read(handle, owner, approval)


class _Download(Response):
    def __init__(self, request: Request):
        super().__init__(b"", headers=HEADERS)
        self.request = request

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Authorize at delivery, not just while building an earlier response.
        # Fixed refusal never logs exception/cookie/handle/body details.
        try:
            content = _content(self.request)
        except Exception:
            response = Response("Download unavailable.", status_code=404, media_type="text/plain", headers=HEADERS)
        else:
            response = Response(content, media_type="text/plain; charset=utf-8",
                                headers={**HEADERS, "Content-Disposition": 'attachment; filename="reviewed_statement.txt"'})
        await response(scope, receive, send)


async def download(request: Request) -> Response:
    return _Download(request)


def routes() -> list[Any]:
    return [Route("/pilot-exports/{handle:path}", download, methods=["GET", "HEAD"])]
