import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[1]


class ByteTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return ([1] if add_special_tokens else []) + [b + 3 for b in text.encode()]

    def decode(self, ids, skip_special_tokens=True):
        return bytes(i - 3 for i in ids if i >= 3).decode()


class TinyCausalModel(torch.nn.Module):
    """Small random causal network for real CPU optimizer/EMA tests."""

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(259, 8)
        self.context = torch.nn.Linear(8, 8)
        self.head = torch.nn.Linear(8, 259)

    def forward(self, input_ids, **kwargs):
        embedded = self.embedding(input_ids)
        positions = torch.arange(1, input_ids.shape[1] + 1, device=input_ids.device)[
            None, :, None
        ]
        # Uniform causal attention retains long-prefix information, unlike a
        # tiny random GRU whose prefix signal underflows after hundreds of tokens.
        history = embedded.cumsum(dim=1) / positions
        hidden = torch.tanh(embedded + self.context(history))
        return SimpleNamespace(logits=self.head(hidden))


def teacher_class():
    # Import this exact pinned source without importing unrelated distillation
    # managers (which require the GPU runtime). The same class is used in workers.
    path = (
        ROOT
        / "third_party/PGR-Probe/verl/verl/trainer/distillation/snapshot_teacher.py"
    )
    spec = importlib.util.spec_from_file_location("pinned_snapshot_teacher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ActorSideTeacher


def config_fixture():
    return {
        "schema_version": 1,
        "objective": "residual_verpo",
        "model": {"path": "test/model", "revision": "a" * 40},
        "runtime": {
            "nnodes": 1,
            "gpus_per_node": 2,
            "tensor_parallel_size": 1,
            "micro_batch_size_per_gpu": 1,
            "vllm_memory_utilization": 0.5,
            "package_versions": {
                k: "1.0.0"
                for k in (
                    "torch",
                    "transformers",
                    "ray",
                    "vllm",
                    "tensordict",
                    "transfer_queue",
                )
            },
        },
        "dataset_manifest": "manifest.json",
        "output_dir": "outputs/test",
        "seed": 123,
        "train_steps": 2,
        "train_batch_size": 2,
        "group_size": 2,
        "learning_rate": 0.001,
        "clip_ratio": 0.2,
        "ema_decay": 0.9,
        "lambda_ref": 1.0,
        "lambda_feedback": 1.0,
        "max_prompt_tokens": 1024,
        "max_response_tokens": 1024,
        "max_replay_tokens": 4096,
        "max_action_tokens": 256,
        "max_turns": 6,
        "timeout_seconds": 10,
        "validation_interval": 1,
        "checkpoint_interval": 1,
    }
