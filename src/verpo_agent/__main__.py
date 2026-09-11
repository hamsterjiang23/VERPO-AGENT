from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="verpo-agent")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-config", "train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
    generate = commands.add_parser("generate-data")
    generate.add_argument("--output", required=True)
    for key in ("seed", "train", "validation", "test"):
        generate.add_argument("--" + key, type=int, required=True)
    audit = commands.add_parser("replay-audit")
    audit.add_argument("--trajectory", required=True)
    audit.add_argument(
        "--tokenizer", required=True, help="Existing local tokenizer directory"
    )
    audit.add_argument("--max-length", type=int, required=True)
    audit.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "generate-data":
        from .data import write_dataset

        print(
            write_dataset(
                args.output,
                args.seed,
                {key: getattr(args, key) for key in ("train", "validation", "test")},
            )
        )
    elif args.command == "replay-audit":
        from transformers import AutoTokenizer
        from .replay import build_replay
        from .trajectory import Trajectory
        from .provenance import atomic_json

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, local_files_only=True, trust_remote_code=False
        )
        trajectory = Trajectory.from_dict(json.loads(Path(args.trajectory).read_text()))
        atomic_json(
            args.output, build_replay(trajectory, tokenizer, args.max_length).to_dict()
        )
        print(
            json.dumps(
                {"status": "passed", "action_tokens": len(trajectory.action_targets)}
            )
        )
    else:
        from .config import load_config
        from .data import read_dataset

        config = load_config(args.config)
        if args.command == "validate-config":
            from .provenance import source_identity

            data = read_dataset(config["dataset_manifest"])
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "scope": "configuration_and_sources",
                        "gpu_validated": False,
                        "source": source_identity(),
                        "splits": {s: len(r) for s, r in data.items()},
                    },
                    indent=2,
                )
            )
        else:
            from .native import run

            run(config, evaluate=args.command == "evaluate")


if __name__ == "__main__":
    main()
