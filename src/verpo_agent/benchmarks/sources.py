"""Verify the external environment engine without importing its trainer."""

import json
from pathlib import Path
import subprocess
from verpo_agent.provenance import ROOT, digest


def verify_source(directory):
    directory = Path(directory)
    lock = json.loads((ROOT / "benchmark_sources.lock.json").read_text())["agentopsd"]
    revision = subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(directory), "diff", "HEAD", "--name-only"], text=True
    )
    if revision != lock["commit"] or dirty:
        raise ValueError(
            "Environment engine must use the clean pinned AgentOPSD snapshot"
        )
    for name, expected in lock["source_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Benchmark engine source mismatch: {name}")
    return revision
