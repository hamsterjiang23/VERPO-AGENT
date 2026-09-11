"""Lossless, JSON-serializable policy/environment transcript."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math


@dataclass
class Trajectory:
    task_id: str
    group_id: str
    policy_version: str
    prompt_ids: list[int]
    response_ids: list[int] = field(default_factory=list)
    action_mask: list[int] = field(default_factory=list)
    behavior_logprobs: list[float] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    reward: float = 0.0
    termination: str = "running"
    schema_version: int = 1

    def append(self, ids, *, policy: bool, logprobs=None):
        ids = list(ids)
        if policy and (logprobs is None or len(logprobs) != len(ids)):
            raise ValueError(
                "Every sampled token requires its behavior log probability"
            )
        self.response_ids.extend(ids)
        self.action_mask.extend([int(policy)] * len(ids))
        self.behavior_logprobs.extend(list(logprobs) if policy else [0.0] * len(ids))

    def validate(self, *, complete=True):
        if (
            self.schema_version != 1
            or not self.task_id
            or not self.group_id
            or not self.policy_version
        ):
            raise ValueError("Missing trajectory identity or unsupported schema")
        if not self.prompt_ids or any(
            type(t) is not int or t < 0 for t in self.prompt_ids + self.response_ids
        ):
            raise ValueError(
                "Token IDs must be nonnegative integers with a nonempty prompt"
            )
        if (
            not len(self.response_ids)
            == len(self.action_mask)
            == len(self.behavior_logprobs)
        ):
            raise ValueError("Trajectory token/mask/logprob lengths differ")
        if any(type(m) is not int or m not in (0, 1) for m in self.action_mask):
            raise ValueError("Action mask must be binary")
        if any(
            not math.isfinite(x) or (m and x > 1e-5)
            for x, m in zip(self.behavior_logprobs, self.action_mask)
        ):
            raise ValueError("Invalid behavior log probability")
        if self.reward not in (0.0, 1.0):
            raise ValueError("Expected binary environment outcome")
        if complete and self.termination == "running":
            raise ValueError("Incomplete trajectory")
        return self

    def to_dict(self):
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value).validate()

    @property
    def input_ids(self):
        return self.prompt_ids + self.response_ids

    @property
    def action_targets(self):
        return [t for t, m in zip(self.response_ids, self.action_mask) if m]

    @property
    def predictor_indices(self):
        return [
            len(self.prompt_ids) + i - 1 for i, m in enumerate(self.action_mask) if m
        ]
