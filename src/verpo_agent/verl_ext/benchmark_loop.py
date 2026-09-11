"""Benchmark adapter around veRL's actual ToolAgentLoop state machine."""

from __future__ import annotations
import asyncio
import json
from pathlib import Path
from uuid import uuid4
from omegaconf import OmegaConf

from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop, AgentState
from verl.experimental.agent_loop.tool_parser import ToolParser, FunctionCall
from verpo_agent.benchmarks.protocol import prompt
from verpo_agent.provenance import atomic_json, source_identity, verify_runtime
from verpo_agent.trajectory import Trajectory


@ToolParser.register("benchmark_actions")
class BenchmarkActionParser(ToolParser):
    async def extract_tool_calls(self, responses_ids, tools=None):
        # Format validation belongs to the real environment, including invalid calls.
        text = self.tokenizer.decode(responses_ids, skip_special_tokens=True)
        return text, [
            FunctionCall(
                name="benchmark_environment", arguments=json.dumps({"action": text})
            )
        ]


class BenchmarkAgentLoop(ToolAgentLoop):
    async def run(self, sampling_params, **kwargs):
        self.experiment = OmegaConf.to_container(
            self.config.actor_rollout_ref.agent.experiment, resolve=True
        )
        if source_identity() != OmegaConf.to_container(
            self.config.actor_rollout_ref.agent.source_identity, resolve=True
        ):
            raise ValueError("Benchmark AgentLoop source identity mismatch")
        verify_runtime(self.experiment["runtime"]["package_versions"])
        self._data, self._termination = None, "running"
        tool = self.tools["benchmark_environment"]
        row = kwargs["extra_info"]["task"]
        # Same task and seed for every member of a GRPO group; distinct sessions.
        session, initial = await tool.create(
            create_kwargs={"task_id": row["task_id"], "seed": self.experiment["seed"]}
        )
        self._tool, self._session = tool, session
        self._initial_messages = prompt(row["benchmark"], initial.text)
        available = tool.sessions[session]["initial"].get("available_actions")
        if available:
            self._initial_messages[-1]["content"] += (
                "\nAvailable actions: " + json.dumps(available)
            )
        kwargs["raw_prompt"] = self._initial_messages
        failure = None
        try:
            # Native PENDING -> GENERATING -> PROCESSING_TOOLS transitions and output.
            output = await super().run(sampling_params, **kwargs)
            if len(output.response_ids) != len(self._data.response_mask):
                raise ValueError("Native output truncated generated action tokens")
            if any(
                output.extra_fields.get(k) != int(kwargs["global_steps"])
                for k in ("min_global_steps", "max_global_steps")
            ):
                raise ValueError("Benchmark sampler used an unexpected policy version")
        except BaseException as error:
            failure = error
            self._termination = (
                "timeout"
                if isinstance(error, (TimeoutError, asyncio.TimeoutError))
                else "backend_error"
            )
        finally:
            try:
                state = tool.sessions[session]
                last = state["last"] or {"reward": 0.0, "score": 0.0}
                if self._data is not None:
                    data = self._data
                    length = len(data.response_mask)
                    split = len(data.prompt_ids) - length
                    trajectory = Trajectory(
                        row["task_id"],
                        str(kwargs["uid"]),
                        str(kwargs["global_steps"]),
                        list(data.prompt_ids[:split]),
                        list(data.prompt_ids[split:]),
                        list(data.response_mask),
                        list(data.response_logprobs),
                        list(data.messages),
                        list(state["events"]),
                        float(last["reward"]),
                        self._termination,
                        metadata={
                            "benchmark": row["benchmark"],
                            "subset": row.get("subset", "webshop"),
                            "split": row["split"],
                            "score": last["score"],
                            "environment_identity": tool.config["identity"],
                            "session_id": session,
                            "sampling_params": dict(sampling_params),
                        },
                    )
                    if failure:
                        trajectory.events.append(
                            {
                                "kind": "error",
                                "error_type": type(failure).__name__,
                                "message": str(failure),
                            }
                        )
                    from dataclasses import asdict

                    record = asdict(trajectory)
                    try:
                        trajectory.validate()
                    except Exception as validation_error:
                        failure = failure or validation_error
                        record["validation_error"] = str(validation_error)
                    atomic_json(
                        Path(self.experiment["output_dir"])
                        / "raw_rollouts"
                        / f"{uuid4().hex}.json",
                        record,
                    )
            finally:
                try:
                    await tool.release(session)
                except Exception:
                    if failure is None:
                        raise
        if failure is not None:
            raise failure
        if not trajectory.action_targets:
            raise ValueError("No policy tokens in benchmark rollout")
        output.reward_score = trajectory.reward
        output.extra_fields.update(
            agent_trajectory=record,
            min_global_steps=int(kwargs["global_steps"]),
            max_global_steps=int(kwargs["global_steps"]),
            reward_extra_info={
                "acc": trajectory.reward,
                "trajectory_json": json.dumps(record),
            },
        )
        return output

    async def _handle_pending_state(self, data, sampling_params):
        self._data = data
        # Benchmark XML actions are intentional, not native JSON function-call prompts.
        data._active_tool_schemas = []
        state = await super()._handle_pending_state(data, sampling_params)
        if len(data.prompt_ids) > self.experiment["max_prompt_tokens"]:
            raise ValueError("Benchmark initial observation exceeds prompt budget")
        return state

    async def _handle_generating_state(
        self, data, sampling_params, ignore_termination=False
    ):
        remaining = self.response_length - len(data.response_mask)
        if remaining <= 0:
            self._termination = "length"
            return AgentState.TERMINATED
        params = {
            **sampling_params,
            "max_tokens": min(remaining, self.experiment["max_action_tokens"]),
            "logprobs": True,
        }
        # Let the last legal action reach the environment before enforcing the horizon.
        old_assistant, old_user = self.max_assistant_turns, self.max_user_turns
        self.max_assistant_turns = self.max_user_turns = None
        try:
            state = await asyncio.wait_for(
                super()._handle_generating_state(data, params, ignore_termination=True),
                self.experiment["timeout_seconds"],
            )
        finally:
            self.max_assistant_turns, self.max_user_turns = old_assistant, old_user
        if len(data.response_mask) > self.response_length or len(
            data.response_mask
        ) != len(data.response_logprobs):
            raise ValueError("Sampler exceeded budget or omitted behavior logprobs")
        data.messages.append(
            {
                "role": "assistant",
                "content": self.tokenizer.decode(
                    data.response_ids, skip_special_tokens=True
                ),
                "token_ids": list(data.response_ids),
            }
        )
        if not data.response_ids:
            self._termination = "empty_generation"
            return AgentState.TERMINATED
        return state

    async def _call_tool(self, tool_call, tools_kwargs, data):
        # Upstream default lifecycle creates/releases per call. Benchmarks need
        # one persistent episode spanning all calls; keep the BaseTool contract.
        return await self._tool.execute(
            self._session, json.loads(tool_call.arguments), agent_data=data
        )

    async def _handle_processing_tools_state(self, data):
        before = len(data.response_mask)
        state = await super()._handle_processing_tools_state(data)
        last = self._tool.sessions[self._session]["last"]
        if last["done"]:
            self._termination = last["termination"]
            return AgentState.TERMINATED
        if state == AgentState.TERMINATED or len(data.response_mask) == before:
            self._termination = "length"
            return AgentState.TERMINATED
        if data.assistant_turns >= self.experiment["max_turns"]:
            self._termination = "turn_limit"
            return AgentState.TERMINATED
        return state
