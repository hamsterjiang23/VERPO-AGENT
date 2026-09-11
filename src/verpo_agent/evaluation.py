"""Metrics derived from complete saved Student trajectories."""

from __future__ import annotations


def summarize(trajectories):
    if not trajectories:
        raise ValueError("Evaluation must contain trajectories")
    calls = sum(sum(e["kind"] != "final" for e in t.events) for t in trajectories)
    errors = sum(sum(e["kind"] == "error" for e in t.events) for t in trajectories)
    result = {
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
    groups = {}
    for t in trajectories:
        if t.metadata.get("benchmark"):
            key = t.metadata["benchmark"] + "/" + t.metadata["subset"]
            groups.setdefault(key, []).append(t)
    for key, group in groups.items():
        result[key + "/count"] = len(group)
        result[key + "/success_rate"] = sum(t.reward for t in group) / len(group)
    webshop = [t for t in trajectories if t.metadata.get("benchmark") == "webshop"]
    if webshop:
        result["webshop/score"] = (
            100 * sum(t.metadata["score"] for t in webshop) / len(webshop)
        )
        result["webshop/success_percent"] = (
            100 * sum(t.reward for t in webshop) / len(webshop)
        )
    qa = [t for t in trajectories if t.metadata.get("benchmark") == "search_qa"]
    if qa:
        result["search_qa/micro_accuracy"] = sum(t.reward for t in qa) / len(qa)
        for label, in_domain in (("id", True), ("ood", False)):
            part = [
                t
                for t in qa
                if (t.metadata["subset"] in {"nq", "hotpotqa"}) == in_domain
            ]
            result[f"search_qa/{label}_count"] = len(part)
            if part:
                result[f"search_qa/{label}_micro_accuracy"] = sum(
                    t.reward for t in part
                ) / len(part)
        subsets = {t.metadata["subset"] for t in qa}
        result["search_qa/macro_accuracy"] = sum(
            result[f"search_qa/{s}/success_rate"] for s in subsets
        ) / len(subsets)
        result["search_qa/subsets_evaluated"] = len(subsets)
    return result
