"""Explicit, hashed episode selections; no random synthetic benchmark replacement."""

from __future__ import annotations
import json
from pathlib import Path
from .protocol import BENCHMARKS, QA_DATASETS, ALF_TASKS
from verpo_agent.provenance import atomic_json, digest


def require_provenance(value):
    if (
        not isinstance(value, dict)
        or not value
        or any(v is None or v == "" for v in value.values())
    ):
        raise ValueError(
            "Provenance must contain actual values, not template placeholders"
        )


def validate_episode(row, benchmark, split):
    if (
        row.get("benchmark") != benchmark
        or row.get("split") != split
        or not row.get("task_id")
    ):
        raise ValueError("Episode benchmark/split/identity mismatch")
    if benchmark == "alfworld":
        if (
            row["subset"] not in ALF_TASKS
            or not row.get("gamefile")
            or not row.get("game_sha256")
        ):
            raise ValueError(
                "ALFWorld requires a real game file, hash and task category"
            )
        expected = {
            "train": "train",
            "validation": "valid_seen",
            "test": "valid_unseen",
        }[split]
        if expected not in Path(row["gamefile"]).parts:
            raise ValueError("ALFWorld official split mismatch")
        return row["gamefile"]
    if benchmark == "webshop":
        goal = row["goal_index"]
        if (
            type(goal) is not int
            or goal < 0
            or (split == "train" and goal < 500)
            or (split != "train" and goal >= 500)
        ):
            raise ValueError(
                "WebShop uses train goal indices >=500 and evaluation <500"
            )
        return str(goal)
    if row["subset"] not in QA_DATASETS or (
        split == "train" and row["subset"] not in {"nq", "hotpotqa"}
    ):
        raise ValueError("Search-QA training is restricted to NQ/HotpotQA")
    if not isinstance(row.get("question"), str) or not row["question"].strip():
        raise ValueError("Question is missing")
    if (
        not isinstance(row.get("answers"), list)
        or not row["answers"]
        or not all(isinstance(a, str) and a.strip() for a in row["answers"])
    ):
        raise ValueError("Private nonempty answer aliases are required")
    return " ".join(row["question"].lower().split())


def read_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    require_provenance(manifest.get("provenance"))
    if (
        manifest.get("schema_version") != 2
        or manifest.get("generator") != "benchmark_episodes_v1"
        or manifest.get("benchmark") not in BENCHMARKS
    ):
        raise ValueError("Invalid benchmark episode manifest")
    if set(manifest["splits"]) != {"train", "validation", "test"} or not manifest.get(
        "provenance"
    ):
        raise ValueError("Explicit split selection and provenance are required")
    rows, ids, instances = {}, set(), set()
    for split, info in manifest["splits"].items():
        source = (path.parent / info["path"]).resolve()
        if (
            not source.is_relative_to(path.parent.resolve())
            or digest(source) != info["sha256"]
        ):
            raise ValueError("Benchmark file path/hash mismatch")
        rows[split] = [json.loads(line) for line in source.read_text().splitlines()]
        if len(rows[split]) != info["count"] or not rows[split]:
            raise ValueError("Empty or inconsistent benchmark split")
        for row in rows[split]:
            instance = validate_episode(row, manifest["benchmark"], split)
            if row["task_id"] in ids or instance in instances:
                raise ValueError("Duplicate benchmark instance across splits")
            ids.add(row["task_id"])
            instances.add(instance)
    return rows


def prepare(selection_path, directory):
    """Selection JSON lists local JSONL/parquet files and their declared provenance."""
    selection_path, directory = Path(selection_path).resolve(), Path(directory)
    spec = json.loads(selection_path.read_text())
    require_provenance(spec.get("provenance"))
    benchmark = spec["benchmark"]
    if benchmark not in BENCHMARKS or not spec.get("provenance"):
        raise ValueError("Benchmark/provenance must be explicit")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        raise ValueError("Choose a new benchmark manifest directory")
    manifest = {
        "schema_version": 2,
        "generator": "benchmark_episodes_v1",
        "benchmark": benchmark,
        "provenance": {
            **spec["provenance"],
            "selection_sha256": digest(selection_path),
        },
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        output, source_hashes = [], []
        for item in spec["splits"][split]:
            source = (selection_path.parent / item["path"]).resolve()
            if digest(source) != item["sha256"]:
                raise ValueError("Input selection checksum mismatch")
            source_hashes.append(
                {"sha256": item["sha256"], "subset": item.get("subset")}
            )
            if source.suffix == ".parquet":
                import pyarrow.parquet as pq

                records = pq.read_table(source).to_pylist()
            else:
                records = [json.loads(line) for line in source.read_text().splitlines()]
            selected = item.get("indices", list(range(len(records))))
            if not isinstance(selected, list) or any(
                type(i) is not int or not 0 <= i < len(records) for i in selected
            ):
                raise ValueError("Selection indices must be explicit valid row indices")
            for index in selected:
                record = records[index]
                if benchmark == "search_qa":
                    targets = record.get("answers", record.get("golden_answers"))
                    if targets is None:
                        targets = record["reward_model"]["ground_truth"]
                    if isinstance(targets, dict):
                        targets = targets["target"]
                    if isinstance(targets, str):
                        targets = [targets]
                    record = {
                        "question": record.get(
                            "question", record.get("extra_info", {}).get("question")
                        ),
                        "answers": targets,
                        "subset": item["subset"],
                    }
                record = {
                    **record,
                    "task_id": record.get(
                        "task_id", f"{benchmark}:{split}:{len(output)}"
                    ),
                    "benchmark": benchmark,
                    "split": split,
                }
                validate_episode(record, benchmark, split)
                output.append(record)
        target = directory / f"{split}.jsonl"
        target.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output)
        )
        manifest["splits"][split] = {
            "path": target.name,
            "count": len(output),
            "sha256": digest(target),
            "sources": source_hashes,
        }
    # Publish only a validated manifest; failed selections do not become runnable.
    pending = directory / "manifest.pending.json"
    atomic_json(pending, manifest)
    read_manifest(pending)
    pending.replace(directory / "manifest.json")
    return directory / "manifest.json"
