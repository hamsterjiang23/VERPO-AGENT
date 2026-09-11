"""Metrics derived from complete saved Student trajectories."""

from __future__ import annotations


def summarize(trajectories):
    if not trajectories:
        raise ValueError("Evaluation must contain trajectories")
    calls = sum(sum(e["kind"] != "final" for e in t.events) for t in trajectories)
    errors = sum(sum(e["kind"] == "error" for e in t.events) for t in trajectories)
    return {
        "trajectories": len(trajectories),
        "success_rate": sum(t.reward for t in trajectories) / len(trajectories),
        "tool_calls": calls,
        "mean_tool_calls": calls / len(trajectories),
        "invalid_call_rate": errors / calls if calls else 0.0,
        "truncation_rate": sum(
            t.termination in {"length", "turn_limit"} for t in trajectories
        )
        / len(trajectories),
        "timeouts": sum(t.termination == "timeout" for t in trajectories),
        "student_privileged_prefix": False,
    }
