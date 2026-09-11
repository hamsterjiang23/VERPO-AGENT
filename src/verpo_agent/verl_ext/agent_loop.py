"""Real vLLM multi-turn sampling with deterministic local tools."""

from __future__ import annotations

from pathlib import Path
import json
from uuid import uuid4
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)

from verpo_agent.provenance import atomic_json, source_identity, verify_runtime
from verpo_agent.rollout import collect


class AgentToolsLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs):
        from omegaconf import OmegaConf

        c = OmegaConf.to_container(
            self.config.actor_rollout_ref.agent.experiment, resolve=True
        )
        identity = source_identity()
        if identity != OmegaConf.to_container(
            self.config.actor_rollout_ref.agent.source_identity, resolve=True
        ):
            raise ValueError("AgentLoop source differs from driver")
        verify_runtime(c["runtime"]["package_versions"])
        task = kwargs["extra_info"]["task"]
        sample_id = uuid4().hex

        async def generate(ids, budget):
            if len(ids) > c["max_prompt_tokens"] + c["max_response_tokens"]:
                raise ValueError("Student context exceeds rollout budget")
            params = dict(sampling_params)
            params.update(max_tokens=budget, logprobs=True)
            return await self.server_manager.generate(
                request_id=uuid4().hex, prompt_ids=ids, sampling_params=params
            )

        trajectory = await collect(
            task,
            self.tokenizer,
            generate,
            group_id=str(kwargs["uid"]),
            policy_version=str(kwargs["global_steps"]),
            max_turns=c["max_turns"],
            max_response_tokens=c["max_response_tokens"],
            max_action_tokens=c["max_action_tokens"],
            timeout_seconds=c["timeout_seconds"],
        )
        record = trajectory.to_dict()
        atomic_json(
            Path(c["output_dir"]) / "raw_rollouts" / f"{sample_id}.json", record
        )
        if trajectory.termination == "backend_error":
            raise RuntimeError(
                f"Generation backend failed; partial trajectory preserved as {sample_id}"
            )
        if not trajectory.action_targets:
            raise ValueError(
                f"Empty policy trajectory preserved as {sample_id}; cannot train on it"
            )
        if len(trajectory.prompt_ids) > c["max_prompt_tokens"]:
            raise ValueError("Initial Student prompt exceeds configured budget")
        step = int(kwargs["global_steps"])
        return AgentLoopOutput(
            prompt_ids=trajectory.prompt_ids,
            response_ids=trajectory.response_ids,
            response_mask=trajectory.action_mask,
            response_logprobs=trajectory.behavior_logprobs,
            reward_score=trajectory.reward,
            num_turns=len(trajectory.events),
            metrics=AgentLoopMetrics(),
            extra_fields={
                "agent_trajectory": record,
                "min_global_steps": step,
                "max_global_steps": step,
                "reward_extra_info": {
                    "acc": trajectory.reward,
                    "trajectory_json": json.dumps(record),
                },
            },
        )
