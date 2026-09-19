#!/usr/bin/env python3
"""A fake ``sandbox`` CLI, so tests/test_vercel_sandbox_runner.py needs no Vercel.

The Vercel Sandbox CLI is not installed in CI and there is no account to point it
at, so ``scripts/vercel_sandbox_runner.py`` invokes *this* instead. It implements
the four subcommands the runner uses — ``create``, ``exec``, ``copy``, ``remove``
— against a directory standing in for the microVMs, and it does the honest thing
at the one step that matters: an ``exec`` of the entrypoint **runs the real
``scripts/ocr_and_extract.py``** over the file the runner really copied in (with
the remote paths rewritten to their local stand-ins). Everything on this side of
the CLI boundary is then exercised for real — the manifest, the label-to-path
mapping, the exec argv, the copy-back, the JSON on stdout.

Environment (all read by the fake, never by the runner):

    FAKE_SANDBOX_BOX       directory standing in for the boxes        (required)
    FAKE_SANDBOX_JOURNAL   JSONL of every call, in order              (required)
    FAKE_SANDBOX_PROJECT   checkout the exec step runs scripts/ from  (default: cwd)
    FAKE_SANDBOX_FAIL      step to fail, exit 7 (entrypoint: exit 2):
                           create | mkdir | copy-in | entrypoint | copy-out | remove
    FAKE_SANDBOX_HOLD      step to block in until <box>/release exists, so a test
                           can signal the runner while that step is in flight
    FAKE_SANDBOX_REPORT    corrupt the report written by copy-out so the runner's
                           own validation can be tested: garbage | no-documents

Exit codes: the step's, always — this is a fake of a CLI, not of a failure mode.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BOX = Path(os.environ["FAKE_SANDBOX_BOX"])
JOURNAL = Path(os.environ["FAKE_SANDBOX_JOURNAL"])
PROJECT = Path(os.environ.get("FAKE_SANDBOX_PROJECT", Path.cwd()))
FAIL_AT = os.environ.get("FAKE_SANDBOX_FAIL", "")
HOLD_AT = os.environ.get("FAKE_SANDBOX_HOLD", "")
CORRUPT_REPORT = os.environ.get("FAKE_SANDBOX_REPORT", "")

REMOTE_ROOT = "/work"
APP_ROOT = "/app/"
HOLD_SECONDS = 30.0


def note(entry: dict) -> None:
    with JOURNAL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def option(argv: list[str], flag: str) -> str | None:
    for index, arg in enumerate(argv):
        if arg == flag and index + 1 < len(argv):
            return argv[index + 1]
    return None


def split_remote(text: str) -> tuple[str | None, str]:
    """``"va-lse-ocr-abc:/work/x"`` → ``("va-lse-ocr-abc", "/work/x")``."""
    head, separator, tail = text.partition(":")
    if separator and tail.startswith("/"):
        return head, tail
    return None, text


def local_for(name: str, remote: str) -> Path:
    """A remote path as its stand-in inside the box directory."""
    if not remote.startswith(REMOTE_ROOT):
        raise SystemExit(f"fake sandbox: remote path outside {REMOTE_ROOT}: {remote}")
    return BOX / name / Path(remote).relative_to("/")


def hold(step: str) -> bool:
    """Block in *step* until the test releases us; False = we were cancelled.

    The marker file is how a test knows the step is in flight — the step's journal
    entry is only written when it finishes, and the runner has already captured
    our stderr.
    """
    if HOLD_AT != step:
        return True
    (BOX / "holding").write_text(step, encoding="utf-8")
    print(f"fake sandbox: holding in {step}", file=sys.stderr, flush=True)
    deadline = time.monotonic() + HOLD_SECONDS
    release = BOX / "release"
    while time.monotonic() < deadline:
        if release.exists():
            return True
        time.sleep(0.05)
    return False


def maybe_fail(step: str, *, entrypoint: bool = False) -> int | None:
    """The injected failure, as an exit code — or None to carry on."""
    if FAIL_AT != step:
        return None
    if entrypoint:
        message = (
            "✖ Scans are present and no OCR tooling is installed, so those pages "
            "have no text to extract."
        )
        print(f"✖ fake sandbox: {step} failed on purpose", file=sys.stderr)
        print(message, file=sys.stderr)
        return 2
    print(f"✖ fake sandbox: {step} failed on purpose", file=sys.stderr)
    return 7


def rewrite(name: str, command: list[str]) -> list[str]:
    """The box's command line, with its paths pointed at the box directory."""
    rewritten: list[str] = []
    for index, arg in enumerate(command):
        if arg == REMOTE_ROOT or arg.startswith(f"{REMOTE_ROOT}/"):
            rewritten.append(str(local_for(name, arg)))
        elif arg.startswith(APP_ROOT):
            rewritten.append(str(PROJECT / arg[len(APP_ROOT) :]))
        elif index == 0 and arg in ("python", "python3"):
            rewritten.append(sys.executable)
        else:
            rewritten.append(arg)
    return rewritten


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else ""
    rest = argv[2:]
    # Every call is journalled whatever happens to it, so a test can assert the
    # order of steps (and that a failing step still cleaned up after itself).
    entry: dict = {"command": command, "argv": rest}

    if command == "create":
        name = option(rest, "--name") or "unnamed"
        entry["name"] = name
        failure = maybe_fail("create")
        if failure is not None:
            note({**entry, "exit": failure})
            return failure
        if not hold("create"):
            note({**entry, "exit": 130})
            return 130
        (BOX / name).mkdir(parents=True, exist_ok=True)
        note({**entry, "exit": 0})
        return 0

    if command == "remove":
        name = rest[0] if rest else "unnamed"
        entry["name"] = name
        failure = maybe_fail("remove")
        if failure is not None:
            note({**entry, "exit": failure})
            return failure
        shutil.rmtree(BOX / name, ignore_errors=True)
        note({**entry, "exit": 0})
        return 0

    if command == "copy":
        positionals: list[str] = []
        index = 0
        while index < len(rest):
            arg = rest[index]
            if arg in ("--token", "--project", "--scope"):
                index += 2  # the option's value is not a path
                continue
            if not arg.startswith("-"):
                positionals.append(arg)
            index += 1
        source, destination = positionals[0], positionals[1]
        source_name, source_remote = split_remote(source)
        destination_name, destination_remote = split_remote(destination)
        if source_name is not None:
            step, name, remote, local = "copy-out", source_name, source_remote, Path(destination)
        else:
            step, name, remote, local = "copy-in", destination_name or "", destination_remote, Path(source)
        entry.update({"step": step, "name": name, "remote": remote, "local": local.as_posix()})
        failure = maybe_fail(step)
        if failure is not None:
            note({**entry, "exit": failure})
            return failure
        if not hold(step):
            note({**entry, "exit": 130})
            return 130
        target = local_for(name, remote)
        if step == "copy-out":
            if CORRUPT_REPORT == "garbage":
                local.write_text("not json at all", encoding="utf-8")
                note({**entry, "exit": 0})
                return 0
            if CORRUPT_REPORT == "no-documents":
                local.write_text(json.dumps({"version": 1, "bundle": []}), encoding="utf-8")
                note({**entry, "exit": 0})
                return 0
            if not target.is_file():
                print(f"fake sandbox: no such file in the box: {remote}", file=sys.stderr)
                note({**entry, "exit": 1})
                return 1
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target, local)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, target)
        note({**entry, "exit": 0})
        return 0

    if command == "exec":
        command_argv: list[str] = []
        name = ""
        values = ("--token", "--project", "--scope", "--workdir", "--env")
        index = 0
        while index < len(rest):
            arg = rest[index]
            if arg == "--":
                command_argv = rest[index + 1 :]
                break
            if arg in values:
                index += 2  # the option's value is not the sandbox's name
                continue
            if not arg.startswith("-") and not name:
                name = arg
            index += 1
        if not command_argv:
            print("fake sandbox: exec needs a command after --", file=sys.stderr)
            note({**entry, "exit": 2})
            return 2
        if command_argv[0] == "mkdir":
            entry.update({"step": "mkdir", "name": name, "directory": command_argv[-1]})
            failure = maybe_fail("mkdir")
            if failure is not None:
                note({**entry, "exit": failure})
                return failure
            if not hold("mkdir"):
                note({**entry, "exit": 130})
                return 130
            local_for(name, command_argv[-1]).mkdir(parents=True, exist_ok=True)
            note({**entry, "exit": 0})
            return 0
        entry.update({"step": "entrypoint", "name": name, "command_argv": command_argv})
        failure = maybe_fail("entrypoint", entrypoint=True)
        if failure is not None:
            note({**entry, "exit": failure})
            return failure
        if not hold("entrypoint"):
            note({**entry, "exit": 130})
            return 130
        proc = subprocess.run(
            rewrite(name, command_argv), cwd=PROJECT, capture_output=True, text=True, check=False
        )
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        note({**entry, "exit": proc.returncode})
        return proc.returncode

    print(f"fake sandbox: unknown subcommand {command!r}", file=sys.stderr)
    note({**entry, "exit": 127})
    return 127


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
