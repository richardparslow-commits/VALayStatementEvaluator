"""Prepare or assess synthetic review artifacts offline. Never calls a provider."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.accuracy_benchmark import BenchmarkInvalid, assess, prepare, read_json, source_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("prepare", help="Freeze a draft plan; no approvals or calls")
    freeze.add_argument("--corpus", type=Path, required=True)
    freeze.add_argument("--configuration", type=Path, required=True)
    freeze.add_argument("--repetitions", type=int, default=1)
    freeze.add_argument("--out", type=Path, required=True)
    check = commands.add_parser("assess", help="Validate independently reviewed artifacts")
    for field in ("plan", "agreement", "results", "review", "out"):
        check.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args()
    try:
        source = source_snapshot(args.repo)
        if args.command == "prepare":
            artifact = prepare(read_json(args.corpus), source, read_json(args.configuration), args.repetitions)
        else:
            artifact = assess(read_json(args.plan), read_json(args.agreement), read_json(args.results),
                              read_json(args.review), source)
        # Exclusive creation prevents an accidental overwrite of frozen evidence.
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(artifact, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print("Draft plan frozen; independent approval and actual runs are still required."
              if args.command == "prepare" else artifact["disposition"] + "; pilot admission is not authorized by this tool.")
        return 0 if args.command == "prepare" or artifact["disposition"] == "REVIEW_READY" else 2
    except (BenchmarkInvalid, KeyError, TypeError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        # Do not echo source text, arbitrary paths, configuration or operator data.
        print("Benchmark evidence incomplete or inconsistent; no acceptance artifact written."
              if not isinstance(exc, BenchmarkInvalid) else str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
