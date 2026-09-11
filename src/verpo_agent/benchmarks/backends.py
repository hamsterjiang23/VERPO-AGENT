"""Real benchmark APIs. Import benchmark engines only in the environment service."""

from __future__ import annotations
import importlib
import json
from pathlib import Path
import sys
from uuid import uuid4
from urllib.request import Request, urlopen

from .protocol import BENCHMARKS, project_action, exact_match
from verpo_agent.provenance import digest


def request_json(url, payload=None, *, method=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


class SearchEpisode:
    def __init__(self, row, config, seed):
        self.row, self.config = row, config
        self.turns, self.done = 0, False

    def reset(self):
        return self.row["question"], {}

    def step(self, text):
        self.turns += 1
        kind, value, valid = project_action(text, "search_qa")
        self.done = kind == "answer" or self.turns >= BENCHMARKS["search_qa"]
        reward = exact_match(value, self.row["answers"]) if kind == "answer" else 0.0
        observation, details = "", {}
        if not self.done and kind == "search":
            # Official AgentOPSD SearchToolGroup sends a one-query list as `query`.
            result = request_json(
                self.config["search_url"],
                {"query": [value], "topk": self.config["topk"], "return_scores": True},
                timeout=self.config["timeout_seconds"],
            )
            documents = result["result"]
            if len(documents) != 1 or not isinstance(documents[0], list):
                raise ValueError(
                    "Retriever does not implement the registered Search-R1 response schema"
                )
            docs = documents[0]
            content = "".join(
                f"Doc {i + 1}: {doc['document']['contents'].strip()}\n"
                for i, doc in enumerate(docs)
            )
            observation = (
                "\n<information>"
                + json.dumps({"result": content or "No search results found."})
                + "</information>\n"
            )
            details = {"retrieval": result, "query": value}
        elif not self.done:
            observation = (
                "Invalid action. Use one <search>...</search> or <answer>...</answer>."
            )
        return (
            observation,
            reward,
            self.done,
            {
                "valid_action": valid,
                "task_score": reward,
                "time_limit": self.turns >= 4 and kind != "answer",
                **details,
            },
        )

    def close(self):
        pass


class AlfworldEpisode:
    def __init__(self, row, config, seed):
        import yaml

        module = importlib.import_module("alfworld.agents.environment.alfred_tw_env")
        root = Path(config["data_root"]).resolve()
        game = (root / row["gamefile"]).resolve()
        if not game.is_relative_to(root) or digest(game) != row["game_sha256"]:
            raise ValueError("ALFWorld game provenance mismatch")
        settings = yaml.safe_load(Path(config["alfworld_config"]).read_text())
        settings["dataset"].update(
            data_path=str(root / "json_2.1.1/train"),
            eval_id_data_path=str(root / "json_2.1.1/valid_seen"),
            eval_ood_data_path=str(root / "json_2.1.1/valid_unseen"),
        )
        settings["env"]["domain_randomization"] = False
        settings["dagger"]["training"]["max_nb_steps_per_episode"] = 50
        settings["general"]["training_method"] = "dagger"
        split = {
            "train": "train",
            "validation": "eval_in_distribution",
            "test": "eval_out_of_distribution",
        }[row["split"]]
        base = module.AlfredTWEnv(settings, train_eval=split)
        if str(game) not in base.game_files:
            raise ValueError(
                "Selected game is not a solvable task in the official split"
            )
        base.game_files, base.use_expert = [str(game)], False
        self.env = base.init_env(batch_size=1)
        self.env.seed(seed)
        self.available = []
        self.turns = 0

    def reset(self):
        observations, info = self.env.reset()
        self.available = info["admissible_commands"][0]
        return observations[0], {"available_actions": self.available}

    def step(self, text):
        _, action, valid = project_action(text, "alfworld")
        self.turns += 1
        observations, _, dones, info = self.env.step([action])
        was_admissible = action in self.available
        self.available = info["admissible_commands"][0]
        won = float(bool(info["won"][0]))
        return (
            observations[0],
            won,
            bool(dones[0]),
            {
                "valid_action": valid,
                "admissible_action": was_admissible,
                "available_actions": self.available,
                "task_score": won,
                "time_limit": self.turns >= 50 and not won,
            },
        )

    def close(self):
        self.env.close()


class WebshopEpisode:
    def __init__(self, row, config, seed, shared_server=None):
        from web_agent_site.envs import WebAgentTextEnv

        self.row = row
        self.env = WebAgentTextEnv(
            observation_mode="text",
            file_path=config["products_file"],
            attr_path=config["attributes_file"],
            num_products=None,
            seed=seed,
            server=shared_server,
            session_prefix=uuid4().hex + "_",
        )

    def reset(self):
        observation, _ = self.env.reset(session=self.row["goal_index"])
        return observation, {"available_actions": self.env.get_available_actions()}

    def step(self, text):
        _, action, valid = project_action(text, "webshop")
        observation, score, done, _ = self.env.step(action)
        score = float(score)
        return (
            observation,
            float(done and score == 1.0),
            bool(done),
            {
                "valid_action": valid,
                "task_score": score,
                "available_actions": self.env.get_available_actions(),
            },
        )

    def close(self):
        # Shared catalog/search index persists, per-trajectory browser/session does not.
        self.env.server.user_sessions.pop(self.env.session, None)
        self.env.close()


def make_factory(benchmark, config):
    if benchmark == "search_qa":
        return lambda row, seed: SearchEpisode(row, config, seed)
    package_root = Path(config["source_root"]) / "agent_system/environments/env_package"
    path = package_root / ("alfworld" if benchmark == "alfworld" else "webshop/webshop")
    sys.path.insert(0, str(path.resolve()))
    name = "alfworld" if benchmark == "alfworld" else "web_agent_site"
    module = importlib.import_module(name)
    if not Path(module.__file__).resolve().is_relative_to(path.resolve()):
        raise ValueError(f"{name} must resolve to the pinned benchmark implementation")
    if benchmark == "alfworld":
        return lambda row, seed: AlfworldEpisode(row, config, seed)
    if benchmark == "webshop":
        shared = []

        def create(row, seed):
            episode = WebshopEpisode(row, config, seed, shared[0] if shared else None)
            if not shared:
                shared.append(episode.env.server)
            return episode

        return create
    raise ValueError("Unknown environment backend")
