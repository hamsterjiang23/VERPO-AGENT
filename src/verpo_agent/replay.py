"""Whole-trajectory causal replay with explicit action predictor positions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json

from .trajectory import Trajectory


@dataclass
class ReplayBranch:
    input_ids: list[int]
    predictor_indices: list[int]
    target_ids: list[int]
    attention_mask: list[int] | None = None
    position_ids: list[int] | None = None

    def __post_init__(self):
        if self.attention_mask is None:
            self.attention_mask = [1] * len(self.input_ids)
        if self.position_ids is None:
            self.position_ids = list(range(len(self.input_ids)))

    def validate(self):
        if self.attention_mask != [1] * len(
            self.input_ids
        ) or self.position_ids != list(range(len(self.input_ids))):
            raise ValueError(
                "Unpadded replay requires full causal attention and contiguous positions"
            )
        if len(self.predictor_indices) != len(self.target_ids):
            raise ValueError("Predictor/target count mismatch")
        if self.predictor_indices != sorted(set(self.predictor_indices)):
            raise ValueError("Predictor indices must be unique and ordered")
        for index, target in zip(self.predictor_indices, self.target_ids):
            if (
                not 0 <= index < len(self.input_ids) - 1
                or self.input_ids[index + 1] != target
            ):
                raise ValueError(
                    "Action predictor is not aligned with its original target"
                )
        return self


@dataclass
class ReplayBatch:
    task_id: str
    student: ReplayBranch
    reference: ReplayBranch
    base: ReplayBranch
    evidence: ReplayBranch
    feedback_text: str

    def validate(self):
        for branch in (self.student, self.reference, self.base, self.evidence):
            branch.validate()
            if branch.target_ids != self.student.target_ids:
                raise ValueError("Teacher/Student action targets differ")
        return self

    def to_dict(self):
        self.validate()
        return asdict(self)


def feedback_text(trajectory):
    # Whitelist environment fields. Never copy task records, answers, raw model
    # actions, arbitrary metadata, or dataset ground_truth into the prefix.
    observations = [
        {
            k: e[k]
            for k in ("turn", "kind", "tool", "key", "result", "error_type", "message")
            if k in e
        }
        for e in trajectory.events
        if e["kind"] != "final"
    ]
    feedback = {
        "observations": observations,
        "success": bool(trajectory.reward),
        "termination": trajectory.termination,
    }
    return (
        "Training-only retrospective environment feedback:\n"
        + json.dumps(feedback, ensure_ascii=False, sort_keys=True)
        + "\nOriginal trajectory:\n"
    )


def build_replay(trajectory: Trajectory, tokenizer, max_length: int):
    trajectory.validate()
    if not trajectory.action_targets:
        raise ValueError("Trajectory has no sampled action tokens")
    original = trajectory.input_ids
    feedback = feedback_text(trajectory)
    prefix = list(tokenizer.encode(feedback, add_special_tokens=False))
    # Preserve the original initial BOS and every original token verbatim.
    insertion = int(
        bool(original) and getattr(tokenizer, "bos_token_id", None) == original[0]
    )
    evidence = original[:insertion] + prefix + original[insertion:]
    if max(len(original), len(evidence)) > max_length:
        raise ValueError(
            f"Replay exceeds max_length={max_length}; no action truncation is permitted"
        )
    targets, predictors = trajectory.action_targets, trajectory.predictor_indices
    branch = lambda: ReplayBranch(list(original), list(predictors), list(targets))
    evidence_predictors = [i + len(prefix) for i in predictors]
    return ReplayBatch(
        trajectory.task_id,
        branch(),
        branch(),
        branch(),
        ReplayBranch(evidence, evidence_predictors, list(targets)),
        feedback,
    ).validate()


def pack_replays(replays, device="cpu"):
    """Row-aligned jagged fields survive TensorDict indexing and microbatching."""
    import torch
    from tensordict import TensorDict

    values = {}
    for name in ("student", "reference", "base", "evidence"):
        for field in (
            "input_ids",
            "predictor_indices",
            "target_ids",
            "attention_mask",
            "position_ids",
        ):
            rows = [
                torch.tensor(
                    getattr(getattr(r, name), field), dtype=torch.long, device=device
                )
                for r in replays
            ]
            values[f"agent_{name}_{field}"] = torch.nested.as_nested_tensor(
                rows, layout=torch.jagged
            )
    return TensorDict(values, batch_size=[len(replays)])
