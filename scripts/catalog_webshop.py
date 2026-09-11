#!/usr/bin/env python3
"""Select explicit real WebShop goal IDs from its loaded catalog."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from verpo_agent.benchmarks.backends import make_factory
from verpo_agent.benchmarks.sources import verify_source
from verpo_agent.provenance import atomic_json, digest


def main():
    import numpy as np

    parser = argparse.ArgumentParser()
    for key in ("source-root", "products-file", "attributes-file", "output"):
        parser.add_argument("--" + key, required=True)
    for key in ("seed", "validation-count", "test-count"):
        parser.add_argument("--" + key, required=True, type=int)
    args = parser.parse_args()
    if (
        min(args.validation_count, args.test_count) <= 0
        or args.validation_count + args.test_count > 500
    ):
        raise ValueError(
            "Choose disjoint validation/test counts within the 500 evaluation goals"
        )
    source = Path(args.source_root).resolve()
    verify_source(source)
    cfg = {
        "source_root": str(source),
        "products_file": str(Path(args.products_file).resolve()),
        "attributes_file": str(Path(args.attributes_file).resolve()),
    }
    episode = make_factory("webshop", cfg)({"goal_index": 0}, args.seed)
    try:
        episode.reset()
        count = len(episode.env.server.goals)
    finally:
        episode.close()
    if count <= 500:
        raise ValueError(
            "Full benchmark catalog must contain training goals beyond index 499"
        )
    rng = np.random.RandomState(args.seed)
    validation = rng.choice(
        range(500), size=args.validation_count, replace=False
    ).tolist()
    test = rng.choice(
        sorted(set(range(500)) - set(validation)), size=args.test_count, replace=False
    ).tolist()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "selection.json").exists():
        raise ValueError("Choose a new catalog directory")
    spec = {
        "benchmark": "webshop",
        "provenance": {
            "dataset_revision": digest(args.products_file),
            "attributes_sha256": digest(args.attributes_file),
            "selection_rule": "RandomState choice without replacement; held-out test excludes validation",
            "seed": args.seed,
        },
        "splits": {},
    }
    for split, indices in (
        ("train", range(500, count)),
        ("validation", validation),
        ("test", test),
    ):
        path = out / f"{split}.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {"task_id": f"webshop:{i}", "goal_index": i, "subset": "webshop"}
                )
                + "\n"
                for i in indices
            )
        )
        spec["splits"][split] = [{"path": path.name, "sha256": digest(path)}]
    atomic_json(out / "selection.json", spec)
    print(out / "selection.json")


if __name__ == "__main__":
    main()
