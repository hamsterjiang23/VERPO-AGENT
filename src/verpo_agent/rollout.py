"""Backend-neutral async rollout preserving exact generated tokens."""

from __future__ import annotations

import asyncio
import json

from .environment import ToolEnvironment, initial_prompt
from .trajectory import Trajectory


async def collect(
    task,
    tokenizer,
    generate,
    *,
    group_id,
    policy_version,
    max_turns,
    max_response_tokens,
    max_action_tokens,
    timeout_seconds,
):
    prompt = initial_prompt(task)
    ids = list(tokenizer.encode(prompt, add_special_tokens=True))
    trajectory = Trajectory(
        task["task_id"],
        group_id,
        str(policy_version),
        ids,
        messages=[{"role": "user", "content": prompt}],
    )
    environment = ToolEnvironment(task)
    for turn in range(max_turns):
        remaining = max_response_tokens - len(trajectory.response_ids)
        if remaining <= 0:
            trajectory.termination = "length"
            break
        budget = min(max_action_tokens, remaining)
        try:
            output = await asyncio.wait_for(
                generate(trajectory.input_ids, budget), timeout_seconds
            )
        except asyncio.TimeoutError:
            trajectory.events.append(
                {"kind": "error", "error_type": "TimeoutError", "turn": turn}
            )
            trajectory.termination = "timeout"
            break
        except Exception as error:
            trajectory.events.append(
                {
                    "kind": "error",
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "turn": turn,
                }
            )
            trajectory.termination = "backend_error"
            break
        token_ids, logprobs = list(output.token_ids), output.log_probs
        if len(token_ids) > budget:
            raise ValueError("Generation backend exceeded token budget")
        trajectory.append(token_ids, policy=True, logprobs=logprobs)
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        trajectory.messages.append(
            {
                "role": "assistant",
                "content": text,
                "turn": turn,
                "stop_reason": getattr(output, "stop_reason", None),
                "generation_budget": budget,
                "returned_tokens": len(token_ids),
            }
        )
        if not token_ids:
            trajectory.termination = "empty_generation"
            break
        event = environment.step(text)
        event["turn"] = turn
        event["raw_action"] = text
        trajectory.events.append(event)
        if environment.done:
            trajectory.reward = environment.reward
            trajectory.termination = "final"
            break
        observation = (
            "\nObservation: "
            + json.dumps(event, ensure_ascii=False, sort_keys=True)
            + "\nAction: "
        )
        observation_ids = list(tokenizer.encode(observation, add_special_tokens=False))
        if len(trajectory.response_ids) + len(observation_ids) >= max_response_tokens:
            trajectory.termination = "length"
            break
        trajectory.messages.append(
            {"role": "tool", "content": observation, "turn": turn}
        )
        trajectory.append(observation_ids, policy=False)
    else:
        trajectory.termination = "turn_limit"
    return trajectory.validate()
