"""Bounded local tool execution with owned POSIX descendants and output caps."""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from typing import BinaryIO


def run_bounded(command: list[str], *, timeout: float = 120, max_output: int = 4 * 1024 * 1024) -> subprocess.CompletedProcess[str]:
    if os.name != "posix" or timeout <= 0 or max_output <= 0:
        raise RuntimeError("Bounded tool execution requires POSIX process groups and positive limits.")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    lock = threading.Lock()
    overflow = threading.Event()
    buffers = [bytearray(), bytearray()]
    total = 0
    deadline = time.monotonic() + timeout

    def kill_group() -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def collect(stream: BinaryIO, index: int) -> None:
        nonlocal total
        while True:
            part = stream.read(65536)
            if not part:
                break
            with lock:
                if total + len(part) > max_output:
                    overflow.set()
                    kill_group()
                    return
                total += len(part)
                buffers[index].extend(part)

    assert process.stdout is not None and process.stderr is not None
    readers = [threading.Thread(target=collect, args=(stream, index), daemon=True)
               for index, stream in enumerate((process.stdout, process.stderr))]
    try:
        for thread in readers:
            thread.start()
        process.wait(timeout=max(0.001, deadline - time.monotonic()))
        for thread in readers:
            thread.join(max(0, deadline - time.monotonic()))
        if overflow.is_set():
            raise RuntimeError("Tool output exceeded the processing limit.")
        if any(thread.is_alive() for thread in readers):
            raise subprocess.TimeoutExpired(command, timeout)
        return subprocess.CompletedProcess(command, process.returncode,
                                           buffers[0].decode("utf-8", errors="strict"),
                                           buffers[1].decode("utf-8", errors="strict"))
    finally:
        # Also terminate descendants after a parent exits while keeping pipes open.
        kill_group()
        process.wait()
        for thread in readers:
            if thread.ident is not None:
                thread.join(2)
        process.stdout.close()
        process.stderr.close()
