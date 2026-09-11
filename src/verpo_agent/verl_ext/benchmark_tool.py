"""veRL BaseTool with one persistent benchmark session per sampled trajectory."""

from __future__ import annotations
import asyncio
import json
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse, OpenAIFunctionToolSchema
from verpo_agent.benchmarks.backends import request_json


class BenchmarkEnvironmentTool(BaseTool):
    def __init__(self, config, tool_schema=None):
        if tool_schema is None:
            tool_schema = OpenAIFunctionToolSchema.model_validate(
                {
                    "type": "function",
                    "function": {
                        "name": "benchmark_environment",
                        "description": "Execute one benchmark action in the current episode.",
                        "parameters": {
                            "type": "object",
                            "properties": {"action": {"type": "string"}},
                            "required": ["action"],
                        },
                    },
                }
            )
        super().__init__(config, tool_schema)
        self.sessions = {}

    async def request(self, route, payload=None, method=None):
        return await asyncio.to_thread(
            request_json,
            self.config["endpoint"].rstrip("/") + route,
            payload,
            method=method,
            timeout=self.config["timeout_seconds"],
        )

    async def create(self, instance_id=None, **kwargs):
        health = await self.request("/health")
        if any(
            health[k] != self.config[k]
            for k in ("identity", "benchmark", "manifest_sha256")
        ):
            raise ValueError(
                "Environment service identity, benchmark or episodes mismatch"
            )
        result = await self.request("/sessions", kwargs["create_kwargs"])
        key = result["session_id"]
        if result["identity"] != self.config["identity"]:
            await self.request(f"/sessions/{key}", method="DELETE")
            raise ValueError("Environment changed during session creation")
        self.sessions[key] = {"turn": 0, "initial": result, "events": [], "last": None}
        return key, ToolResponse(text=result["observation"])

    async def execute(self, instance_id, parameters, **kwargs):
        state = self.sessions[instance_id]
        result = await self.request(
            f"/sessions/{instance_id}/step",
            {"action": parameters["action"], "turn": state["turn"]},
        )
        observation = result["observation"]
        if result["available_actions"] is not None:
            observation += "\nAvailable actions: " + json.dumps(
                result["available_actions"]
            )
        event = {
            "kind": "tool" if result["valid_action"] else "error",
            "tool": self.config["benchmark"],
            "turn": state["turn"],
            "raw_action": parameters["action"],
            "result": observation,
            "available_actions": result["available_actions"],
            "score": result["score"],
            "done": result["done"],
            "details": result["details"],
        }
        if not result["valid_action"]:
            event["message"] = "Invalid benchmark action format"
        state["events"].append(event)
        state["turn"] += 1
        state["last"] = result
        return ToolResponse(text=observation), result["reward"], result

    async def calc_reward(self, instance_id, **kwargs):
        result = self.sessions[instance_id]["last"]
        return result["reward"] if result else 0.0

    async def release(self, instance_id, **kwargs):
        try:
            await self.request(f"/sessions/{instance_id}", method="DELETE")
        finally:
            self.sessions.pop(instance_id, None)
