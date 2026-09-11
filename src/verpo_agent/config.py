"""Independent explicit experiment contract; no inherited experiment defaults."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

REQUIRED = {
    "schema_version",
    "objective",
    "model",
    "runtime",
    "dataset_manifest",
    "output_dir",
    "seed",
    "train_steps",
    "train_batch_size",
    "group_size",
    "learning_rate",
    "clip_ratio",
    "ema_decay",
    "lambda_ref",
    "lambda_feedback",
    "max_prompt_tokens",
    "max_response_tokens",
    "max_replay_tokens",
    "max_action_tokens",
    "max_turns",
    "timeout_seconds",
    "validation_interval",
    "checkpoint_interval",
}


def validate_config(value):
    c = copy.deepcopy(value)
    required = REQUIRED | ({"environment"} if c.get("schema_version") == 2 else set())
    if set(c) != required:
        raise ValueError(
            f"Config keys missing={sorted(required - set(c))}, unknown={sorted(set(c) - required)}"
        )
    if c["schema_version"] not in (1, 2) or c["objective"] not in {
        "residual_verpo",
        "feedback_opd",
    }:
        raise ValueError("Unsupported schema or objective")
    if set(c["model"]) != {"path", "revision"} or not all(
        isinstance(v, str) and v for v in c["model"].values()
    ):
        raise ValueError("Model path and immutable revision are required")
    if len(c["model"]["revision"]) != 40 or any(
        x not in "0123456789abcdef" for x in c["model"]["revision"]
    ):
        raise ValueError("Model revision must be a full immutable 40-character commit")
    integers = (
        "train_steps",
        "train_batch_size",
        "group_size",
        "max_prompt_tokens",
        "max_response_tokens",
        "max_replay_tokens",
        "max_action_tokens",
        "max_turns",
        "validation_interval",
        "checkpoint_interval",
    )
    if (
        any(type(c[k]) is not int or c[k] <= 0 for k in integers)
        or type(c["seed"]) is not int
    ):
        raise ValueError("Counts must be positive integers; seed must be an integer")
    if c["group_size"] < 2 or c["max_action_tokens"] > c["max_response_tokens"]:
        raise ValueError("Invalid GRPO group size or action budget")
    if c["max_replay_tokens"] <= c["max_prompt_tokens"] + c["max_response_tokens"]:
        raise ValueError(
            "Reserve an explicit additional context budget for Teacher feedback"
        )
    for key in (
        "learning_rate",
        "clip_ratio",
        "ema_decay",
        "lambda_ref",
        "lambda_feedback",
        "timeout_seconds",
    ):
        if type(c[key]) not in (int, float) or not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"{key} must be explicitly positive and finite")
    if c["ema_decay"] >= 1 or c["clip_ratio"] >= 1:
        raise ValueError("EMA decay and PPO clip ratio must be below 1")
    r = c["runtime"]
    runtime_keys = {
        "nnodes",
        "gpus_per_node",
        "tensor_parallel_size",
        "micro_batch_size_per_gpu",
        "vllm_memory_utilization",
        "package_versions",
    }
    if set(r) != runtime_keys:
        raise ValueError(f"Runtime must contain exactly {sorted(runtime_keys)}")
    for key in runtime_keys - {"vllm_memory_utilization", "package_versions"}:
        if type(r[key]) is not int or r[key] <= 0:
            raise ValueError(f"runtime.{key} must be a positive integer")
    if not 0 < r["vllm_memory_utilization"] <= 0.60:
        raise ValueError("Colocated rollout utilization must be in (0, 0.60]")
    world = r["nnodes"] * r["gpus_per_node"]
    if (
        world % r["tensor_parallel_size"]
        or (c["train_batch_size"] * c["group_size"]) % world
    ):
        raise ValueError(
            "GPU count must divide rollout group batch and support tensor parallelism"
        )
    if (c["train_batch_size"] * c["group_size"] // world) % r[
        "micro_batch_size_per_gpu"
    ]:
        raise ValueError("Per-rank batch must divide evenly into explicit microbatches")
    versions = r["package_versions"]
    if not {
        "torch",
        "transformers",
        "ray",
        "vllm",
        "tensordict",
        "transfer_queue",
    } <= set(versions):
        raise ValueError(
            "Runtime must pin torch, transformers, ray, vllm, tensordict and transfer_queue"
        )
    if any(
        not isinstance(v, str) or not v.strip() or v in {"latest", "*"}
        for v in versions.values()
    ):
        raise ValueError("Runtime package versions must be explicit")
    for key in ("dataset_manifest", "output_dir"):
        if not isinstance(c[key], str) or not c[key]:
            raise ValueError(f"{key} is required")
    if c["schema_version"] == 2:
        from .benchmarks.protocol import BENCHMARKS
        from urllib.parse import urlsplit

        e = c["environment"]
        if (
            set(e) != {"benchmark", "endpoint", "identity", "evaluation_temperature"}
            or e["benchmark"] not in BENCHMARKS
        ):
            raise ValueError("Invalid benchmark environment configuration")
        if (
            not isinstance(e["endpoint"], str)
            or urlsplit(e["endpoint"]).scheme not in {"http", "https"}
            or not urlsplit(e["endpoint"]).hostname
        ):
            raise ValueError("Configure the running benchmark service endpoint")
        if (
            not isinstance(e["identity"], str)
            or len(e["identity"]) != 64
            or any(x not in "0123456789abcdef" for x in e["identity"])
        ):
            raise ValueError("Pin the environment service identity from /health")
        if (
            c["max_turns"] != BENCHMARKS[e["benchmark"]]
            or c["max_action_tokens"] != 512
        ):
            raise ValueError(
                "Paper profiles require 512 tokens per turn and horizons 50/15/4"
            )
        if (
            type(e["evaluation_temperature"]) not in (int, float)
            or not 0 <= e["evaluation_temperature"] <= 1
        ):
            raise ValueError("Explicit evaluation temperature in [0,1] is required")
    return c


def load_config(path):
    path = Path(path).resolve()
    config = validate_config(json.loads(path.read_text()))
    for key in ("dataset_manifest", "output_dir"):
        config[key] = str((path.parent / config[key]).resolve())
    return config
