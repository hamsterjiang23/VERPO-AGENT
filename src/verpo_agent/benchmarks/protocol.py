"""AgentOPSD benchmark action and metric conventions (not its training method)."""

from __future__ import annotations

import re
import string

BENCHMARKS = {"alfworld": 50, "webshop": 15, "search_qa": 4}
QA_DATASETS = {"nq", "triviaqa", "popqa", "hotpotqa", "2wiki", "musique", "bamboogle"}
ALF_TASKS = {
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
}


def project_action(text, benchmark):
    """Keep raw sampled tokens elsewhere; only project text for environment execution.

    Matches the official action projection, including validity flags. The flag is
    diagnostic and does not add the old invalid-action reward penalty.
    """
    if benchmark == "search_qa":
        trimmed = text
        for tag in ("search", "answer"):
            if f"</{tag}>" in text:
                trimmed = text.split(f"</{tag}>", 1)[0] + f"</{tag}>"
                break
        counts = [
            len(re.findall(f"<{tag}>", text, re.I)) for tag in ("search", "answer")
        ]
        for tag in ("search", "answer"):
            match = re.search(f"<{tag}>(.*?)</{tag}>", trimmed, re.I | re.S)
            if match:
                return tag, match[1].strip(), sum(counts) == 1
        return "invalid", "", False
    if benchmark not in BENCHMARKS:
        raise ValueError("Unknown benchmark")
    match = re.search(r"<action>(.*?)</action>", text.lower(), re.S)
    action = (
        match[1].strip()
        if match
        else text.lower()[-(30 if benchmark == "alfworld" else 20) :]
    )
    valid = bool(
        match
        and "<think>" in text
        and "</think>" in text
        and not re.search(r"[\u4e00-\u9fff]", text)
    )
    return "action", action, valid


def normalize_answer(text):
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def exact_match(answer, targets):
    return float(any(normalize_answer(answer) == normalize_answer(t) for t in targets))


def prompt(benchmark, observation):
    # Project-owned wording: same action syntax, no privileged skills or answers.
    if benchmark == "alfworld":
        instruction = "Complete the household goal using the available actions. Think in <think>...</think>, then output one command in <action>...</action>."
    elif benchmark == "webshop":
        instruction = "Find and purchase the product satisfying the shopping request. Think in <think>...</think>, then output one search[query] or click[label] command in <action>...</action>."
    else:
        instruction = "Answer the question using retrieved evidence. You may reason in <think>...</think>. Each turn issue one <search>query</search> or finish with <answer>answer</answer>."
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": observation},
    ]
