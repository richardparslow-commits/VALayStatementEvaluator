"""Maintained pilot PUT adapter for the reviewed Streamlit/Starlette releases.

Streamlit retains DELETE/OPTIONS and its XSRF/CORS protocol. The PUT handler
owns every parsed spool, including refusal and cancellation paths. Never install
this adapter for an unreviewed dependency release.
"""
from __future__ import annotations

from contextlib import aclosing
from importlib.metadata import version
import threading
from typing import Any, Iterable

from starlette.datastructures import FormData, UploadFile
from starlette.exceptions import HTTPException
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import BaseRoute, Route
from starlette.types import Message

from . import pilot

OVERHEAD = 1024 * 1024
MAX_FIELDS = 4
MAX_FIELD_BYTES = 4096
_CLEANUP_FAILED = threading.Event()


class OwnedMultipartParser(MultiPartParser):
    """Require the terminal boundary; finalize alone does not prove it exists."""
    completed = False

    def on_end(self) -> None:
        self.completed = True
        super().on_end()


def verify_runtime() -> None:
    check_cleanup()
    if version("streamlit") != "1.63.0" or version("starlette") != "1.6.0":
        raise pilot.PilotBlocked("Private uploads require the reviewed runtime releases.")


def check_cleanup() -> None:
    if _CLEANUP_FAILED.is_set():
        raise pilot.PilotBlocked("Upload cleanup failed. The operator must stop and recover the pilot.")


def close_form(form: FormData) -> None:
    """Synchronous close cannot be interrupted by another task cancellation.

    Starlette spools are local, bounded files. Visit every multi-item even when
    one close fails; report only a fixed error. Parsing failures are cleaned by
    the reviewed Starlette parser's BaseException handler.
    """
    close_spools(value.file for _, value in form.multi_items() if isinstance(value, UploadFile))


def close_spools(files: Iterable[Any]) -> None:
    failed = False
    for spool in files:
        try:
            spool.close()
        except Exception:
            failed = True
    if failed:
        _CLEANUP_FAILED.set()
        raise pilot.PilotBlocked("Upload temporary-file cleanup failed.")


def create_upload_routes(runtime: Any, upload_mgr: Any, base_url: str | None) -> list[BaseRoute]:
    verify_runtime()
    from . import config
    from streamlit.runtime.uploaded_file_manager import UploadedFileRec
    from streamlit.web.server.starlette import starlette_routes as upstream
    routes = upstream.create_upload_routes(runtime, upload_mgr, base_url)

    async def put(request: Request) -> Response:
        if _CLEANUP_FAILED.is_set():
            raise HTTPException(503, "Upload cleanup unavailable")
        if upstream.is_xsrf_enabled() and not upstream.validate_xsrf_token(
                request.headers.get("X-Xsrftoken"), request.cookies.get(upstream.XSRF_COOKIE_NAME)):
            raise HTTPException(403, "XSRF token missing or invalid")
        session, file_id = request.path_params["session_id"], request.path_params["file_id"]
        if not runtime.is_active_session(session):
            raise HTTPException(400, "Invalid upload session")
        maximum = min(config.MAX_UPLOAD_BYTES, 50 * 1024 * 1024)
        length = request.headers.get("content-length")
        if length is not None:
            try:
                declared = int(length)
            except ValueError:
                raise HTTPException(400, "Invalid Content-Length") from None
            if declared < 0:
                raise HTTPException(400, "Invalid Content-Length")
            if declared > maximum:
                raise HTTPException(413, "File too large")
        received = 0
        form: FormData | None = None
        parser: MultiPartParser | None = None
        if not request.headers.get("content-type", "").lower().startswith("multipart/form-data;"):
            raise HTTPException(400, "A multipart file is required")
        try:
            async with aclosing(request.stream()) as chunks:
                async def receive() -> Message:
                    nonlocal received
                    try:
                        chunk = await anext(chunks)
                    except StopAsyncIteration:
                        return {"type": "http.request", "body": b"", "more_body": False}
                    except RuntimeError:
                        raise HTTPException(400, "Upload body unavailable") from None
                    received += len(chunk)
                    if received > maximum + OVERHEAD:
                        raise HTTPException(413, "File too large")
                    return {"type": "http.request", "body": chunk, "more_body": True}
                limited = Request(request.scope, receive)
                owned = OwnedMultipartParser(limited.headers, limited.stream(), max_files=1,
                                              max_fields=MAX_FIELDS, max_part_size=MAX_FIELD_BYTES)
                parser = owned
                try:
                    form = await parser.parse()
                except MultiPartException:
                    raise HTTPException(400, "Invalid multipart upload") from None
                if not owned.completed:
                    raise HTTPException(400, "Incomplete multipart upload")
            uploads = [value for _, value in form.multi_items() if isinstance(value, UploadFile)]
            if len(uploads) != 1:
                raise HTTPException(400, "Expected exactly one file")
            upload = uploads[0]
            data = await upload.read(maximum + 1)
            if len(data) > maximum:
                raise HTTPException(413, "File too large")
            # Close BEFORE inserting bytes in the manager or sending success.
        finally:
            if parser is not None:
                # The reviewed parser tracks ALL allocations, even a truncated
                # unfinished part that never enters FormData. Closing only form
                # entries would miss that successful-but-incomplete parse path.
                close_spools(parser._files_to_close_on_error)
        upload_mgr.add_file(session_id=session, file=UploadedFileRec(
            file_id=file_id, name=upload.filename or "",
            type=upload.content_type or "application/octet-stream", data=data))
        response = Response(status_code=204)
        # Use the reviewed original OPTIONS endpoint for the same CORS headers.
        options = next(route for route in routes if isinstance(route, Route) and "OPTIONS" in (route.methods or set()))
        headers = await options.endpoint(request)
        response.headers.update(headers.headers)
        response.headers["Cache-Control"] = "no-store"
        return response

    return [Route(route.path, put, methods=["PUT"]) if isinstance(route, Route)
            and "PUT" in (route.methods or set()) else route for route in routes]


def install() -> None:
    verify_runtime()
    from streamlit.web.server.starlette import starlette_app
    starlette_app.create_upload_routes = create_upload_routes
