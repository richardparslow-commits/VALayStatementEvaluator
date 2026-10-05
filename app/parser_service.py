"""Single-flight launcher for a dedicated parser VM over private mutual TLS.

It has no application daemon socket, application credentials or case mounts.
Its engine credentials still authorize the parser VM: isolate and review that
host. The parser gets stdin bytes, a fresh tmpfs, and no bind mounts.
"""
from __future__ import annotations

import os
import json
import re
import selectors
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor
from .parser_engine import ParserEngine

from .parser_protocol import (DEADLINE, MAX_HEADER, MAX_INPUT, MAX_OUTPUT, MAX_PAGES, SOCKET_PATH,
                              ParserRefused, bind_input, decode, encode, frame,
                              recv_exact, recv_frame, validate_request, validate_json_structure)

PROFILE = "va-lse-parser"


class DockerParser:
    def __init__(self, image: str, revision: str, docker: str = "/usr/local/bin/docker", max_pages: int = 500,
                 *, engine: ParserEngine | None = None) -> None:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image) or not revision:
            raise ParserRefused("An immutable reviewed parser image is required.")
        self.image, self.revision, self.docker = image, revision, docker
        if type(max_pages) is not int or not 0 < max_pages <= MAX_PAGES:
            raise ParserRefused("Invalid parser deployment page limit.")
        self.max_pages = max_pages
        self.parse_lock = threading.Lock()
        self.engine = engine or ParserEngine.from_environment()
        self.env = self.engine.environment()

    def _command(self, args: list[str]) -> Any:
        self.engine.verify_credentials()
        result = subprocess.run([self.docker, *args], env=self.env, capture_output=True,
                                timeout=10, check=True)
        if len(result.stdout) > 65536:
            raise ParserRefused("Invalid parser runtime metadata.")
        # Daemon metadata is trusted and separately bounded to 64 KB.
        return json.loads(result.stdout)

    def ready(self) -> dict[str, Any]:
        info = self._command(["info", "--format", "{{json .}}"])
        if (not isinstance(info, dict) or info.get("ID") != self.engine.identity
                or "va-lse-purpose=parser-only" not in info.get("Labels", [])):
            raise ParserRefused("The dedicated parser engine does not match the reviewed identity.")
        security = info.get("SecurityOptions", [])
        if (info.get("OSType") != "linux" or not any("apparmor" in x for x in security)
                or not any("seccomp" in x and "builtin" in x for x in security)):
            raise ParserRefused("Linux AppArmor and default seccomp are required.")
        images = self._command(["image", "inspect", self.image])
        if (not isinstance(images, list) or len(images) != 1 or images[0].get("Id") != self.image
                or f"VA_LSE_BUILD_SHA={self.revision}" not in images[0].get("Config", {}).get("Env", [])):
            raise ParserRefused("The parser image does not match the reviewed revision.")
        return {"ready": True, "image": self.image, "revision": self.revision, "max_pages": self.max_pages,
                "engine_id": self.engine.identity}

    def command(self, name: str) -> list[str]:
        return [self.docker, "run", "--rm", "--pull=never", "--name", name, "--interactive",
                "--network=none", "--ipc=none", "--read-only", "--user=65534:65534",
                "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
                f"--security-opt=apparmor={PROFILE}", "--memory=1g", "--memory-swap=1g",
                "--cpus=1", "--pids-limit=32", "--ulimit=cpu=30:30", "--ulimit=core=0:0", "--ulimit=nofile=64:64",
                # PID 1 stages up to 50 MB input; the parsing child lowers this to 32 MB.
                f"--ulimit=fsize={MAX_INPUT}:{MAX_INPUT}", "--log-driver=none",
                "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=128m,mode=1777",
                "--workdir=/tmp", "--env=PYTHONPATH=/app", f"--env=VA_LSE_PARSER_MAX_PAGES={self.max_pages}",
                "--entrypoint=/usr/local/bin/python", self.image,
                "-m", "app.parser_worker"]

    def run(self, request: dict[str, Any], data: bytes) -> bytes:
        validate_request(request)
        bind_input(request, data)
        self.ready()  # Recheck identity before every launch, including readiness probes.
        name = "va-parser-" + uuid.uuid4().hex
        process = subprocess.Popen(self.command(name), env=self.env, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   start_new_session=True, bufsize=0)
        output = bytearray()
        payload = frame(encode(request)) + data
        deadline = time.monotonic() + DEADLINE + 10
        try:
            assert process.stdin is not None and process.stdout is not None
            with selectors.DefaultSelector() as selector:
                os.set_blocking(process.stdin.fileno(), False)
                os.set_blocking(process.stdout.fileno(), False)
                selector.register(process.stdin, selectors.EVENT_WRITE)
                selector.register(process.stdout, selectors.EVENT_READ)
                offset = 0
                while selector.get_map():
                    if time.monotonic() >= deadline:
                        raise ParserRefused("Parser deadline exceeded.")
                    for key, events in selector.select(0.1):
                        if events & selectors.EVENT_WRITE:
                            try:
                                offset += os.write(key.fd, payload[offset:offset + 65536])
                            except BrokenPipeError as exc:
                                raise ParserRefused("Parser input was refused.") from exc
                            if offset == len(payload):
                                selector.unregister(key.fileobj)
                                process.stdin.close()
                        else:
                            part = os.read(key.fd, 65536)
                            if not part:
                                selector.unregister(key.fileobj)
                            output.extend(part)
                            if len(output) > MAX_OUTPUT:
                                raise ParserRefused("Parser output exceeds its limit.")
            if process.wait(timeout=max(0.1, deadline - time.monotonic())) != 0:
                raise ParserRefused("Parser refused the document.")
            # The Docker-authorized launcher never builds an object graph from
            # document output. Bound lexical structure and pass opaque bytes in
            # an image-bound envelope; the web client validates the full schema.
            validate_json_structure(bytes(output))
            encoded = (b'{"image":' + encode(self.image) + b',"engine_id":' + encode(self.engine.identity)
                       + b',"response":' + bytes(output) + b'}')
            if len(encoded) > MAX_OUTPUT:
                raise ParserRefused("Parser output exceeds its limit.")
            return encoded
        finally:
            try:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
            finally:
                try:
                    for pipe in (process.stdin, process.stdout):
                        if pipe is not None:
                            pipe.close()
                finally:
                    # Removal must still run after kill, reap or pipe-close failure.
                    subprocess.run([self.docker, "rm", "--force", name], env=self.env,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   # Auto-remove may have already removed an exited container.
                                   timeout=10, check=False)


def handle(connection: socket.socket, runner: DockerParser) -> None:
    connection.settimeout(10)
    locked = False
    try:
        header = decode(recv_frame(connection, MAX_HEADER, time.monotonic() + 10))
        if header == {"operation": "health"}:
            reply = encode(runner.ready())
        else:
            request = validate_request(header)
            locked = runner.parse_lock.acquire(blocking=False)
            if not locked:
                connection.sendall(frame(encode({"busy": True})))
                return
            # Explicit admission before bytes: a busy caller does not upload or queue.
            connection.sendall(frame(encode({"accepted": True})))
            # A monotonic overall upload deadline also bounds slow partial senders.
            deadline = time.monotonic() + 15
            data = bytearray()
            while len(data) < request["size"]:
                connection.settimeout(max(0.001, deadline - time.monotonic()))
                data.extend(recv_exact(connection, min(65536, request["size"] - len(data)), deadline))
                if time.monotonic() >= deadline:
                    raise ParserRefused("Parser upload deadline exceeded.")
            bind_input(request, bytes(data))
            reply = runner.run(request, bytes(data))
        connection.settimeout(10)
        connection.sendall(frame(reply))
    except (OSError, ValueError, subprocess.SubprocessError, AssertionError, TypeError, KeyError, RecursionError):
        # Deliberately exclude labels, parser stderr and exception text from logs/replies.
        try:
            connection.sendall(frame(encode({"error": "Parser refused this file; no fallback was used."})))
        except OSError:
            pass
    finally:
        if locked:
            runner.parse_lock.release()


def _handle_owned(connection: socket.socket, runner: DockerParser, slots: threading.Semaphore) -> None:
    try:
        with connection:
            handle(connection, runner)
    finally:
        slots.release()


def serve() -> None:
    runner = DockerParser(os.environ.get("VA_LSE_PARSER_IMAGE", ""), os.environ.get("VA_LSE_BUILD_SHA", ""),
                          max_pages=int(os.environ.get("VA_LSE_PARSER_MAX_PAGES", "500")))
    runner.ready()
    import hashlib
    data = b"Synthetic parser readiness fixture."
    runner.run({"version": 1, "label": "readiness.txt", "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(), "nonce": uuid.uuid4().hex, "page_limit": runner.max_pages}, data)
    path = Path(SOCKET_PATH)
    path.unlink(missing_ok=True)
    slots = threading.Semaphore(4)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener, ThreadPoolExecutor(max_workers=4) as executor:
        listener.bind(str(path))
        path.chmod(0o600)
        listener.listen(4)
        while True:
            slots.acquire()
            try:
                connection, _ = listener.accept()
                executor.submit(_handle_owned, connection, runner, slots)
            except BaseException:
                slots.release()
                raise


if __name__ == "__main__":
    serve()
