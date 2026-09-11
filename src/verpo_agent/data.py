"""Inspectable seeded datasets with content hashes and isolated splits."""

from __future__ import annotations

import json
from pathlib import Path

from .environment import generate_tasks, initial_prompt
from .provenance import atomic_json, digest


def write_dataset(directory, seed, counts):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        raise ValueError("Dataset manifest already exists; choose a new directory")
    manifest = {
        "schema_version": 1,
        "generator": "lookup_calculate_v1",
        "seed": seed,
        "splits": {},
    }
    for split, rows in generate_tasks(seed, counts).items():
        path = directory / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        manifest["splits"][split] = {
            "path": path.name,
            "count": len(rows),
            "sha256": digest(path),
        }
    atomic_json(directory / "manifest.json", manifest)
    return directory / "manifest.json"


def read_dataset(manifest_path):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if (
        manifest["schema_version"] != 1
        or manifest["generator"] != "lookup_calculate_v1"
        or set(manifest["splits"]) != {"train", "validation", "test"}
    ):
        raise ValueError("Invalid dataset contract")
    datasets, ids, instances = {}, set(), set()
    for split, info in manifest["splits"].items():
        source = (path.parent / info["path"]).resolve()
        if (
            not source.is_relative_to(path.parent.resolve())
            or digest(source) != info["sha256"]
        ):
            raise ValueError("Dataset path or checksum mismatch")
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        if len(rows) != info["count"] or not rows:
            raise ValueError("Dataset count mismatch")
        for row in rows:
            instance = json.dumps([row["records"], row["expression"]], sort_keys=True)
            if row["task_id"] in ids or instance in instances:
                raise ValueError("Duplicate task or instance across dataset splits")
            ids.add(row["task_id"])
            instances.add(instance)
        datasets[split] = rows
    return datasets


def native_dataset_files(manifest_path, directory):
    """Materialize veRL JSON rows; private environment state stays in extra_info."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, rows in read_dataset(manifest_path).items():
        path = directory / f"{split}.jsonl"
        native = [
            {
                "data_source": "agent_tools_v1",
                "agent_name": "agent_verpo_tools",
                "prompt": [{"role": "user", "content": initial_prompt(row)}],
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {"task": row, "index": i},
            }
            for i, row in enumerate(rows)
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in native))
        paths[split] = str(path)
    return paths
