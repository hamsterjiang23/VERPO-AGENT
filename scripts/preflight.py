#!/usr/bin/env python3
"""Network-free provenance checks. This does not validate or launch training."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise ValueError("git command failed: " + " ".join(args))
    return result.stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("manifest path escapes its root")
    return target


def inspect_workspace(root: Path = ROOT) -> dict:
    checks = []

    def check(name: str, condition: bool) -> None:
        checks.append({"name": name, "passed": bool(condition)})

    try:
        check("python_3_10_or_newer", sys.version_info >= (3, 10))
        lock = json.loads((root / "upstream.lock.json").read_text())
        dep = inside(root, lock["path"])
        check("dependency_initialized", (dep / ".git").exists())
        if not checks[-1]["passed"]:
            raise ValueError("Initialize third_party/PGR-Probe with git submodule update --init")
        check("dependency_is_own_worktree", Path(git(dep, "rev-parse", "--show-toplevel")).resolve() == dep)
        check("dependency_commit_matches_lock", git(dep, "rev-parse", "HEAD") == lock["commit"])
        gitlink = git(root, "ls-files", "--stage", "--", lock["path"]).split()
        check("gitlink_matches_lock", len(gitlink) >= 3 and gitlink[:3] == ["160000", lock["commit"], "0"])
        url = git(root, "config", "-f", ".gitmodules", "--get", f"submodule.{lock['path']}.url")
        check("submodule_url_matches_lock", url == lock["url"])
        check("dependency_has_no_tracked_edits", not git(dep, "status", "--porcelain", "--untracked-files=no"))
        for relative, expected in lock["source_sha256"].items():
            file = inside(dep, relative)
            check("source:" + relative, file.is_file() and sha256(file) == expected)

        design = json.loads((root / "configs/observation_replay_design.json").read_text())
        check("design_is_not_runnable", design["status"] == "design_only_not_runnable" and design["training_enabled"] is False)
        for key in ("step_gate_enabled", "benefit_predictor_enabled", "fec_projection_enabled"):
            check(key + "_is_false", design[key] is False)
        check("replay_is_full_trajectory", design["replay"]["unit"] == "full_trajectory")
        provenance = json.loads((root / "docs/research/provenance.json").read_text())
        check("research_snapshot_commit", provenance["upstream_commit"] == lock["commit"])
        for entry in provenance["files"]:
            source = inside(dep, entry["source"])
            exported = inside(root, entry["destination"])
            check("snapshot_source:" + entry["source"], source.is_file() and sha256(source) == entry["source_sha256"])
            check("snapshot_export:" + entry["destination"], exported.is_file() and sha256(exported) == entry["export_sha256"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        checks.append({"name": "preflight_input_error", "passed": False, "detail": str(error)})
    return {
        "status": "passed" if checks and all(c["passed"] for c in checks) else "failed",
        "scope": "dependency_and_design_only",
        "training_ready": False,
        "checks": checks,
    }


if __name__ == "__main__":
    report = inspect_workspace()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)
