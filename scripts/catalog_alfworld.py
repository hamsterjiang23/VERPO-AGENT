#!/usr/bin/env python3
"""Inventory real ALFWorld games into selection inputs; no expert answers exported."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from verpo_agent.benchmarks.protocol import ALF_TASKS
from verpo_agent.provenance import atomic_json, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, out = Path(args.data_root).resolve(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "selection.json").exists():
        raise ValueError("Choose a new catalog directory")
    spec = {
        "benchmark": "alfworld",
        "provenance": {
            "dataset_revision": "ALFWorld json_2.1.1",
            "selection_rule": "all discovered games with explicit source hash; valid_seen development / valid_unseen test",
        },
        "splits": {},
    }
    for split, native in (
        ("train", "train"),
        ("validation", "valid_seen"),
        ("test", "valid_unseen"),
    ):
        rows = []
        for trajectory in sorted(
            (root / "json_2.1.1" / native).rglob("traj_data.json")
        ):
            game = trajectory.parent / "game.tw-pddl"
            metadata = json.loads(trajectory.read_text())
            if not game.is_file() or metadata.get("task_type") not in ALF_TASKS:
                continue
            relative = str(game.relative_to(root))
            rows.append(
                {
                    "task_id": "alfworld:" + relative,
                    "gamefile": relative,
                    "game_sha256": digest(game),
                    "subset": metadata["task_type"],
                }
            )
        if not rows:
            raise ValueError(f"No official ALFWorld games found in {native}")
        path = out / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        spec["splits"][split] = [{"path": path.name, "sha256": digest(path)}]
    atomic_json(out / "selection.json", spec)
    print(out / "selection.json")


if __name__ == "__main__":
    main()
