#!/usr/bin/env python3
"""Run an explicit command against checked, pinned source paths."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from preflight import ROOT, inspect_workspace


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] != "--" or len(args) == 1:
        print("Usage: python scripts/run_with_upstream.py -- COMMAND [ARGS...]", file=sys.stderr)
        return 2
    report = inspect_workspace()
    if report["status"] != "passed":
        print(json.dumps(report, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    command = args[1:]
    if command[0] in ("python", "python3"):
        command[0] = sys.executable
    env = os.environ.copy()
    dep = ROOT / "third_party/PGR-Probe"
    paths = [str(ROOT / "src"), str(dep / "verl"), str(dep)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    try:
        return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode
    except OSError as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
