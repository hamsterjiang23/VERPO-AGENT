"""Source/runtime verification performed by the driver and distributed workers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def source_identity():
    spec = importlib.util.spec_from_file_location(
        "agent_preflight", ROOT / "scripts/preflight.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.inspect_workspace(ROOT)
    if report["status"] != "passed":
        raise ValueError("Dependency preflight failed: " + json.dumps(report))
    dep = ROOT / "third_party/PGR-Probe"
    for package, expected in (
        ("verl", dep / "verl/verl"),
        ("risk_aware_opsd", dep / "risk_aware_opsd"),
    ):
        found = importlib.util.find_spec(package)
        locations = list(found.submodule_search_locations or []) if found else []
        if not locations or not all(Path(p).resolve() == expected for p in locations):
            raise ValueError(
                f"{package} must resolve to the pinned source; use scripts/run_with_upstream.py"
            )
    files = (
        list((ROOT / "src/verpo_agent").rglob("*.py"))
        + list((ROOT / "scripts").glob("*.py"))
        + [
            ROOT / "pyproject.toml",
            ROOT / "upstream.lock.json",
            ROOT / "benchmark_sources.lock.json",
        ]
        + list((ROOT / "configs/benchmarks").glob("*.json"))
    )
    manifest = {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}
    return {
        "agent_commit": module.git(ROOT, "rev-parse", "HEAD"),
        "upstream_commit": json.loads((ROOT / "upstream.lock.json").read_text())[
            "commit"
        ],
        "agent_source_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        "files": manifest,
    }


def verify_runtime(expected):
    distributions = {"transfer_queue": "TransferQueue"}
    actual = {
        name: importlib.metadata.version(distributions.get(name, name))
        for name in expected
    }
    if actual != expected:
        raise ValueError(f"Runtime mismatch: expected={expected}, actual={actual}")
    return {"python": sys.version, "packages": actual}
