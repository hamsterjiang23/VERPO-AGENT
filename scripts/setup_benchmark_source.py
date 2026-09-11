#!/usr/bin/env python3
"""Fetch a locked benchmark reference; never install its veRL into this runtime."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from verpo_agent.benchmarks.sources import verify_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    path = Path(args.directory).resolve()
    lock = json.loads((ROOT / "benchmark_sources.lock.json").read_text())["agentopsd"]
    if not path.exists():
        path.mkdir(parents=True)
        subprocess.run(["git", "init", str(path)], check=True)
        subprocess.run(
            ["git", "-C", str(path), "remote", "add", "origin", lock["url"]], check=True
        )
        subprocess.run(
            ["git", "-C", str(path), "fetch", "--depth", "1", "origin", lock["commit"]],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(path), "checkout", "--detach", lock["commit"]], check=True
        )
    verify_source(path)
    print(
        json.dumps(
            {
                "status": "passed",
                "commit": lock["commit"],
                "scope": "benchmark_source_only",
            }
        )
    )


if __name__ == "__main__":
    main()
