"""Provision public references and mount the private synthetic Archil workspace.

No raw statements, medical records, model keys or prompt logs are uploaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE = ("legal_framework.md", "evaluation_rubric.md", "drafting_guide.md", "topic_checklist.md")
KB_DISK = "dsk-0000000000113f9e"
RUNS_DISK = "dsk-0000000000113fb6"
SANDBOX = "01a16b36-645f-77e6-8e44-1321316d6239"


def reference_release(root: Path = ROOT) -> tuple[str, dict[str, bytes]]:
    """Allowlisted public rules + explicitly synthetic benchmark, content addressed."""
    paths = [*(f"app/knowledge/{name}" for name in KNOWLEDGE),
             "benchmarks/synthetic-accuracy-v1.json"]
    files = {}
    for name in paths:
        path = root / name
        if path.is_symlink():
            raise ValueError("Reference symlinks are not accepted")
        files[name] = path.read_bytes()
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    encoded = json.dumps(manifest, sort_keys=True).encode()
    release = hashlib.sha256(encoded).hexdigest()
    files["manifest.json"] = encoded
    return release, files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "upload-kb", "mount", "smoke"))
    args = parser.parse_args()
    release, files = reference_release()
    if args.action == "plan":
        print(json.dumps({"release": release, "files": list(files), "kb_disk": KB_DISK,
                          "runs_disk": RUNS_DISK, "sandbox": SANDBOX}, indent=2))
        return 0
    if os.getenv("VA_LSE_MODE", "synthetic") != "synthetic":
        parser.error("Archil setup is limited to synthetic mode")
    if not os.getenv("ARCHIL_API_KEY"):
        parser.error("Set ARCHIL_API_KEY through your secret manager")
    import archil
    with archil.Archil(region="aws-us-east-1", timeout=300) as client:
        if args.action == "upload-kb":
            disk = client.disks.get(KB_DISK)
            for name, data in files.items():
                key = f"releases/{release}/{name}"
                if disk.object_exists(key):
                    if disk.get_object(key) != data:
                        raise ValueError("Existing release differs; refusing overwrite")
                else:
                    disk.put_object(key, data)
            print("Public reference release verified: " + release)
        elif args.action == "mount":
            disk = client.disks.get(KB_DISK)
            if disk.get_object(f"releases/{release}/manifest.json") != files["manifest.json"]:
                raise ValueError("Reference release is missing or inconsistent")
            box = client.sandboxes.get(SANDBOX)
            # Existing mounts may only change while stopped. This is explicit operator work.
            if box.status not in {"stopped", "exited"}:
                parser.error("Stop the sandbox in the console before changing mounts")
            box.start(mounts=[
                archil.SandboxMountSpec(disk=KB_DISK, path="/mnt/kb", read_only=True),
                archil.SandboxMountSpec(disk=RUNS_DISK, path="/mnt/results"),
            ])
            print("Private sandbox started with pinned read-only knowledge and separate results")
        else:
            # No LLM calls, no user files, no main-disk writes. Tests mount semantics.
            result = client.exec(disks={"kb": archil.ExecMountSpec(KB_DISK, read_only=True,
                                      subdirectory=f"releases/{release}")},
                                 command="test -r /mnt/archil/kb/manifest.json && "
                                         "test -r /mnt/archil/kb/app/knowledge/legal_framework.md && "
                                         "if touch /mnt/archil/kb/.write-probe 2>/dev/null; "
                                         "then exit 1; else echo readonly-smoke-ok; fi")
            if result.exit_code != 0:
                raise RuntimeError("Archil mounted-reference smoke failed")
            print(result.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
