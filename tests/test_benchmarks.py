from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest

from verpo_agent.benchmarks.backends import SearchEpisode
from verpo_agent.benchmarks.data import prepare, read_manifest
from verpo_agent.benchmarks.protocol import project_action, exact_match
from verpo_agent.benchmarks.service import SessionService, create_app, verify_service
from verpo_agent.provenance import atomic_json, digest, source_identity
from helpers import config_fixture


@contextmanager
def retriever():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            result = {
                "result": [
                    [
                        {
                            "document": {"contents": "Paris is the capital of France."},
                            "score": 1.0,
                        }
                    ]
                ]
            }
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/retrieve", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@contextmanager
def environment_http(service):
    import uvicorn

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app(service), log_level="error", lifespan="off")
    )
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        raise RuntimeError("Environment test HTTP server did not start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()


def make_dataset(directory):
    spec = {
        "benchmark": "search_qa",
        "provenance": {
            "dataset_revision": "test-fixture",
            "selection_rule": "explicit three questions",
        },
        "splits": {},
    }
    for split, subset in (
        ("train", "nq"),
        ("validation", "hotpotqa"),
        ("test", "bamboogle"),
    ):
        path = directory / f"input_{split}.jsonl"
        path.write_text(
            json.dumps(
                {"question": f"Capital of France? {split}", "answers": ["Paris"]}
            )
            + "\n"
        )
        spec["splits"][split] = [
            {"path": path.name, "sha256": digest(path), "subset": subset}
        ]
    atomic_json(directory / "selection.json", spec)
    return prepare(directory / "selection.json", directory / "data")


def service_fixture(directory, search_url):
    manifest = make_dataset(directory)
    rows = read_manifest(manifest)
    config = {
        "manifest": str(manifest),
        "benchmark": "search_qa",
        "output_dir": str(directory / "service"),
        "max_sessions": 8,
        "session_ttl_seconds": 30,
    }
    backend = {"search_url": search_url, "topk": 3, "timeout_seconds": 5}
    return SessionService(
        config, rows, "b" * 64, lambda row, seed: SearchEpisode(row, backend, seed)
    ), manifest


def local_tokenizer(directory):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast, GPT2Config

    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
    vocab.update(
        {c: i + 3 for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    )
    core = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>"
    )
    tokenizer.chat_template = "{% for m in messages %}{{ bos_token + m['role'] + ': ' + m['content'] + eos_token }}{% endfor %}{% if add_generation_prompt %}{{ bos_token + 'assistant: ' }}{% endif %}"
    directory.mkdir()
    tokenizer.save_pretrained(directory)
    GPT2Config(
        vocab_size=len(vocab),
        n_positions=8192,
        n_embd=8,
        n_layer=1,
        n_head=1,
        architectures=["GPT2LMHeadModel"],
    ).save_pretrained(directory)
    return tokenizer


class BenchmarkTests(unittest.TestCase):
    def test_projection_and_exact_match(self):
        self.assertEqual(
            project_action(
                "<think>go</think><action>TAKE APPLE 1</action>", "alfworld"
            ),
            ("action", "take apple 1", True),
        )
        self.assertFalse(
            project_action("<action>click[Buy Now]</action>", "webshop")[2]
        )
        self.assertEqual(
            project_action("<search> a </search><answer>b</answer>", "search_qa"),
            ("search", "a", False),
        )
        self.assertEqual(exact_match("The PARIS!", ["Paris"]), 1)
        self.assertEqual(exact_match("Paris, France", ["Paris"]), 0)

    def test_manifest_split_and_hash_enforcement(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_dataset(Path(tmp))
            rows = read_manifest(path)
            self.assertEqual(rows["train"][0]["subset"], "nq")
            self.assertEqual(rows["test"][0]["answers"], ["Paris"])
            (path.parent / "test.jsonl").write_text("{}\n")
            with self.assertRaises(ValueError):
                read_manifest(path)

    def test_real_search_http_and_session_isolation(self):
        with tempfile.TemporaryDirectory() as tmp, retriever() as (url, requests):
            service, _ = service_fixture(Path(tmp), url)
            a = service.create("search_qa:train:0", 0)["session_id"]
            b = service.create("search_qa:train:0", 0)["session_id"]
            result = service.step(a, "<search>France capital</search>", 0)
            self.assertIn("<information>", result["observation"])
            self.assertEqual(requests[0]["query"], ["France capital"])
            self.assertEqual(service.sessions[b]["turn"], 0)
            with self.assertRaises(ValueError):
                service.step(a, "<answer>Paris</answer>", 0)
            self.assertEqual(service.step(a, "<answer>Paris</answer>", 1)["reward"], 1)
            self.assertEqual(service.step(b, "<answer>Rome</answer>", 0)["reward"], 0)
            service.release(a)
            service.release(b)
            self.assertFalse(service.sessions)

    def test_service_provenance_and_horizon(self):
        with tempfile.TemporaryDirectory() as tmp, retriever() as (url, _):
            directory = Path(tmp)
            service, _ = service_fixture(directory, url)
            resources = directory / "resources.json"
            asset = directory / "retriever.json"
            asset.write_text('{"fixture": true}')
            atomic_json(
                resources,
                {
                    "provenance": {"revision": "fixture"},
                    "files": {asset.name: digest(asset)},
                },
            )
            config = {
                **service.config,
                "resource_manifest": str(resources),
                "package_versions": {"fastapi": importlib.metadata.version("fastapi")},
                "backend": {
                    "search_url": url,
                    "topk": 3,
                    "timeout_seconds": 5,
                    "retriever_provenance": {
                        "model_revision": "fixture",
                        "corpus_revision": "fixture",
                        "index_sha256": digest(asset),
                    },
                },
            }
            rows, identity = verify_service(config)
            self.assertEqual(len(identity), 64)
            self.assertEqual(len(rows["train"]), 1)
            config["backend"]["retriever_provenance"] = None
            with self.assertRaises(ValueError):
                verify_service(config)
            key = service.create("search_qa:train:0", 1)["session_id"]
            for turn in range(4):
                result = service.step(key, "invalid action", turn)
            self.assertTrue(result["done"])
            self.assertEqual(result["termination"], "turn_limit")
            self.assertEqual(result["reward"], 0)
            with self.assertRaises(ValueError):
                service.step(key, "<answer>Paris</answer>", 4)
            service.release(key)

    def test_native_manager_tool_loop_queue_and_replay(self):
        import ray
        import torch
        import transfer_queue as tq
        from omegaconf import OmegaConf
        from verl.utils import tensordict_utils as tu
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ
        from verpo_agent.native import project_native
        from verpo_agent.verl_ext.sampler import AgentReplayBuffer
        from verpo_agent.verl_ext.trainer import AgentTrainer
        from verpo_agent.replay import build_replay, pack_replays
        from verpo_agent.verl_ext.loss import validate_microbatch

        with (
            tempfile.TemporaryDirectory(prefix="abm-", dir="/tmp") as tmp,
            retriever() as (url, requests),
        ):
            directory = Path(tmp)
            service, manifest = service_fixture(directory, url)
            tokenizer = local_tokenizer(directory / "model")

            # Only the LLM backend is scripted. Manager, Ray loop workers, state
            # transitions, BaseTool, HTTP environment, TQ and sampler are real.
            class ScriptedClient:
                async def generate(
                    self, request_id, prompt_ids, sampling_params, **kwargs
                ):
                    from verl.workers.rollout.replica import TokenOutput

                    text = tokenizer.decode(prompt_ids)
                    reply = (
                        "<answer>Paris</answer>"
                        if "<information>" in text
                        else "<search>France capital</search>"
                    )
                    ids = tokenizer.encode(reply, add_special_tokens=False) + [
                        tokenizer.eos_token_id
                    ]
                    return TokenOutput(
                        token_ids=ids,
                        log_probs=[-1.0] * len(ids),
                        extra_fields={"min_global_steps": 1, "max_global_steps": 1},
                    )

            with environment_http(service) as endpoint:
                c = config_fixture()
                c.update(
                    schema_version=2,
                    output_dir=tmp,
                    dataset_manifest=str(manifest),
                    max_turns=4,
                    max_action_tokens=512,
                    max_prompt_tokens=2048,
                    max_response_tokens=2048,
                    max_replay_tokens=8192,
                    environment={
                        "benchmark": "search_qa",
                        "endpoint": endpoint,
                        "identity": service.identity,
                        "evaluation_temperature": 0.4,
                    },
                )
                native = project_native(
                    c,
                    directory / "model",
                    dict(train="train", validation="val", test="test"),
                    source_identity(),
                )
                # Explicit CPU transport test boundary: no vLLM package or CUDA claim.
                native.actor_rollout_ref.agent.experiment.runtime.package_versions = {
                    k: importlib.metadata.version(k)
                    for k in ("torch", "transformers", "ray", "tensordict")
                }
                native.actor_rollout_ref.rollout.agent.num_workers = 1
                native.actor_rollout_ref.model.override_config.attn_implementation = (
                    "eager"
                )
                OmegaConf.save(
                    OmegaConf.create(
                        [
                            {
                                "name": "agent_verpo_benchmark",
                                "_target_": "verpo_agent.verl_ext.benchmark_loop.BenchmarkAgentLoop",
                            }
                        ]
                    ),
                    directory / "agent_loop.yaml",
                )
                OmegaConf.save(
                    OmegaConf.create(
                        {
                            "tools": [
                                {
                                    "class_name": "verpo_agent.verl_ext.benchmark_tool.BenchmarkEnvironmentTool",
                                    "config": {
                                        **c["environment"],
                                        "manifest_sha256": digest(manifest),
                                        "timeout_seconds": 5,
                                        "type": "native",
                                    },
                                }
                            ]
                        }
                    ),
                    directory / "benchmark_tool.yaml",
                )
                ray.init(
                    num_cpus=4,
                    include_dashboard=False,
                    _temp_dir=str(directory / "ray"),
                    logging_level="ERROR",
                )
                manager = None
                try:
                    tq.init(
                        OmegaConf.create(
                            {
                                "backend": {
                                    "storage_backend": "SimpleStorage",
                                    "SimpleStorage": {
                                        "num_data_storage_units": 1,
                                        "total_storage_size": 16,
                                    },
                                }
                            }
                        )
                    )
                    manager = AgentLoopManagerTQ.create(native, ScriptedClient())
                    rows = read_manifest(manifest)
                    data = tu.get_tensordict(
                        {
                            "uid": ["benchmarkgroup"],
                            "index": [0],
                            "raw_prompt": [[{"role": "user", "content": "benchmark"}]],
                            "extra_info": [{"task": rows["train"][0]}],
                            "data_source": ["nq"],
                            "reward_model": [{"ground_truth": ""}],
                        },
                    )
                    data.batch_size = [1]
                    tu.assign_non_tensor(data, global_steps=1, validate=False)
                    tq.kv_batch_put(
                        keys=["benchmarkgroup"],
                        partition_id="train",
                        tags=[
                            {"is_prompt": True, "status": "pending", "global_steps": 1}
                        ],
                    )
                    manager.generate_sequences(data)
                    sampler = AgentReplayBuffer(
                        "sync",
                        native.trainer,
                        1,
                        "wait",
                        OmegaConf.create({}),
                        poll_interval=0.05,
                    )
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        sampler._sync_metadata_from_transfer_queue()
                        if sampler.finished_keys["train"]:
                            break
                        time.sleep(0.05)
                    self.assertTrue(
                        sampler.finished_keys["train"],
                        "Native AgentLoop did not finish",
                    )
                    batch, _ = sampler.sample(1, "train", 1)
                    trajectories = AgentTrainer._read_trajectories(batch)
                    self.assertEqual(len(trajectories), 2)
                    self.assertTrue(all(t.reward == 1.0 for t in trajectories))
                    self.assertEqual(len(requests), 2)
                    self.assertFalse(service.sessions)
                    self.assertEqual(
                        len({t.metadata["session_id"] for t in trajectories}), 2
                    )
                    for t in trajectories:
                        self.assertEqual(len(t.events), 2)
                        self.assertNotIn(
                            "Paris", t.messages[0]["content"] + t.messages[1]["content"]
                        )
                        self.assertIn(tokenizer.eos_token_id, t.action_targets)
                    packed = pack_replays(
                        [build_replay(t, tokenizer, 8192) for t in trajectories]
                    )
                    for key, rows in {
                        "input_ids": [t.input_ids for t in trajectories],
                        "responses": [t.response_ids for t in trajectories],
                        "response_mask": [t.action_mask for t in trajectories],
                    }.items():
                        packed[key] = torch.nested.as_nested_tensor(
                            [torch.tensor(r) for r in rows], layout=torch.jagged
                        )
                    validate_microbatch(packed)
                    if os.environ.get("VERPO_ACCEPTANCE_ARTIFACTS"):
                        target = (
                            Path(os.environ["VERPO_ACCEPTANCE_ARTIFACTS"])
                            / "native_benchmark"
                        )
                        atomic_json(
                            target / "trajectories.json",
                            [t.to_dict() for t in trajectories],
                        )
                        atomic_json(
                            target / "replays.json",
                            [
                                build_replay(t, tokenizer, 8192).to_dict()
                                for t in trajectories
                            ],
                        )
                        atomic_json(target / "retrieval_requests.json", requests)
                finally:
                    if manager:
                        for worker in manager.agent_loop_workers:
                            ray.kill(worker)
                    tq.close()
                    ray.shutdown()


if __name__ == "__main__":
    unittest.main()
