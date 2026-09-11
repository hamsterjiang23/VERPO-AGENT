"""Synchronous veRL trainer with Agent-owned replay and experiment semantics."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from uuid import uuid4
import numpy as np
import ray
import torch
from tensordict import TensorDict
import transfer_queue as tq

from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

from verpo_agent.evaluation import summarize
from verpo_agent.provenance import atomic_json, digest
from verpo_agent.replay import build_replay, pack_replays
from verpo_agent.trajectory import Trajectory
from .workers import AgentActorRolloutRefWorker


def metadata_rows(value):
    """TQ/TensorDict may expose ndarray, list, NonTensorStack or LinkedList."""
    return value.tolist() if hasattr(value, "tolist") else list(value)


class AgentTrainer(PPOTrainerSync):
    def _init_resource_pool_mgr(self):
        super()._init_resource_pool_mgr()
        if self.use_critic:
            raise ValueError("Agent GRPO must not instantiate a critic")
        self.role_worker_mapping[Role.ActorRolloutRef] = ray.remote(
            AgentActorRolloutRefWorker
        )
        self.role_worker_mapping.pop(Role.ActorRollout, None)
        self.mapping.pop(Role.ActorRollout, None)
        self.mapping[Role.ActorRolloutRef] = "global_pool"
        self.use_reference_policy = True

    def _compute_reward_colocate(self, batch, metrics=None):
        # Environment has already scored each trajectory. No legacy reward manager.
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["rm_scores"],
        )
        if "rm_scores" not in data:
            raise ValueError("Environment outcome is missing")
        return batch

    def _compute_ref_log_prob(self, batch, metrics):
        # The independent reference is consumed as full logits by AgentLoss.
        return batch

    def _compute_advantage(self, batch, metrics):
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["uid", "rm_scores", "response_mask"],
        )
        groups = metadata_rows(data["uid"])
        expected = int(self.config.actor_rollout_ref.agent.experiment.group_size)
        if set(Counter(groups).values()) != {expected}:
            raise ValueError(
                "Incomplete GRPO groups; do not silently normalize a partial group"
            )
        masks, scores = (
            data["response_mask"].to_padded_tensor(0),
            data["rm_scores"].to_padded_tensor(0.0),
        )
        advantages, returns = compute_grpo_outcome_advantage(
            scores, masks, np.asarray(groups), norm_adv_by_std_in_grpo=True
        )
        lengths = [len(row) for row in data["response_mask"].unbind()]
        fields = {}
        for name, values in (
            ("advantages", advantages),
            ("returns", returns),
            ("token_level_scores", scores),
            ("token_level_rewards", scores),
        ):
            fields[name] = torch.nested.as_nested_tensor(
                [values[i, :length] for i, length in enumerate(lengths)],
                layout=torch.jagged,
            )
        tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=TensorDict(fields, batch_size=[len(groups)]),
        )
        return batch

    def _update_actor(self, batch, metrics):
        fields = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["extra_fields", "responses", "response_mask", "input_ids"],
        )
        records = metadata_rows(fields["extra_fields"])
        trajectories = [
            Trajectory.from_dict(row["agent_trajectory"]) for row in records
        ]
        cfg = self.config.actor_rollout_ref.agent.experiment
        directory = Path(cfg.output_dir) / "replay" / f"step_{self.global_steps}"
        directory.mkdir(parents=True, exist_ok=True)
        try:
            replays = [
                build_replay(t, self.tokenizer, cfg.max_replay_tokens)
                for t in trajectories
            ]
            for i, trajectory in enumerate(trajectories):
                if (
                    fields["input_ids"][i].tolist() != trajectory.input_ids
                    or fields["responses"][i].tolist() != trajectory.response_ids
                    or fields["response_mask"][i].tolist() != trajectory.action_mask
                ):
                    raise ValueError(
                        "TransferQueue altered trajectory tokens or policy mask"
                    )
                if trajectory.policy_version != str(self.global_steps):
                    raise ValueError(
                        "Stale rollout policy version in synchronous Agent batch"
                    )
            packed = pack_replays(replays)
            tq.kv_batch_put(
                keys=batch.keys, partition_id=batch.partition_id, fields=packed
            )
        except Exception as error:
            atomic_json(
                directory / "error.json",
                {
                    "error": str(error),
                    "trajectories": [t.to_dict() for t in trajectories],
                },
            )
            raise
        atomic_json(
            directory / "batch.json",
            {"keys": list(batch.keys), "replays": [r.to_dict() for r in replays]},
        )
        from verl.utils.metric import reduce_metrics

        # A KVBatchMeta created before replay/advantage insertion may carry an
        # obsolete field selection. The worker bridge must fetch the complete row.
        batch.fields = None

        batch.extra_info.update(
            {
                "actor_loss_use_full_logits": True,
                "verpo_use_full_logits": False,
                "distillation_use_topk": False,
                "calculate_entropy": True,
                "global_batch_size": cfg.train_batch_size * cfg.group_size,
                "mini_batch_size": cfg.train_batch_size * cfg.group_size,
                "epochs": 1,
                "seed": cfg.seed,
                "dataloader_kwargs": {"shuffle": False},
                "temperature": 1.0,
                "global_steps": self.global_steps,
            }
        )
        output = self.actor_rollout_wg.update_actor(batch)
        metrics.update(
            {"actor/" + k: v for k, v in reduce_metrics(output["metrics"]).items()}
        )
        return batch

    @staticmethod
    def _read_trajectories(batch):
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["extra_fields"],
        )
        return [
            Trajectory.from_dict(row["agent_trajectory"])
            for row in metadata_rows(data["extra_fields"])
        ]

    def _save_trajectories(self, directory, trajectories):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{self.global_steps}.jsonl").write_text(
            "".join(json.dumps(t.to_dict()) + "\n" for t in trajectories)
        )
        atomic_json(
            directory / f"metrics_{self.global_steps}.json", summarize(trajectories)
        )

    def _log_rollout_data(self, batch, timing_raw, rollout_data_dir):
        self._save_trajectories(rollout_data_dir, self._read_trajectories(batch))

    def _validate(self):
        from verl.utils import tensordict_utils as tu

        trajectories = []
        for batch_dict in self.val_dataloader:
            count = len(batch_dict["raw_prompt"])
            uids = [uuid4().hex for _ in range(count)]
            batch_dict["uid"] = np.array(uids, dtype=object)
            data = tu.get_tensordict(batch_dict)
            tu.assign_non_tensor(data, global_steps=self.global_steps, validate=True)
            tq.kv_batch_put(
                keys=uids,
                partition_id="val",
                tags=[
                    {
                        "is_prompt": True,
                        "status": "pending",
                        "global_steps": self.global_steps,
                    }
                    for _ in uids
                ],
            )
            self.agent_loop_manager.generate_sequences(data)
            batch, _ = self.replay_buffer.sample(
                global_steps=self.global_steps, partition_id="val", batch_size=count
            )
            if len(batch) != count:
                raise ValueError("Incomplete Student evaluation batch")
            trajectories.extend(self._read_trajectories(batch))
            tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)
        if len({t.task_id for t in trajectories}) != len(trajectories):
            raise ValueError("Duplicated task in Student evaluation")
        self._save_trajectories(self.config.trainer.validation_data_dir, trajectories)
        return {"val/" + key: value for key, value in summarize(trajectories).items()}

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        trajectories = self._read_trajectories(batch)
        metrics.update(
            {"training/" + key: value for key, value in summarize(trajectories).items()}
        )
        metrics.update({"training/global_step": global_steps, "training/epoch": epoch})
        metrics.update(
            {
                "timing/" + key: value
                for key, value in timing_raw.items()
                if isinstance(value, (int, float))
            }
        )
        atomic_json(
            Path(self.config.actor_rollout_ref.agent.experiment.output_dir)
            / "training_metrics"
            / f"{global_steps}.json",
            metrics,
        )

    def _save_checkpoint(self):
        root = Path(self.config.trainer.default_local_dir)
        path = root / f"global_step_{self.global_steps}"
        self.actor_rollout_wg.save_checkpoint(
            str(path / "actor"), None, self.global_steps, max_ckpt_to_keep=None
        )
        torch.save(self.train_dataloader.state_dict(), path / "data.pt")
        world = self.config.trainer.nnodes * self.config.trainer.n_gpus_per_node
        for rank in range(world):
            for name in (
                f"agent_contract_rank_{rank}.json",
                f"verpo_actor_teacher_world_size_{world}_rank_{rank}.pt",
            ):
                if not (path / "actor" / name).is_file():
                    raise ValueError(f"Incomplete checkpoint: {name}")
        atomic_json(
            path / "complete.json",
            {
                "step": self.global_steps,
                "files": {
                    str(p.relative_to(path)): digest(p)
                    for p in path.rglob("*")
                    if p.is_file() and p.name != "complete.json"
                },
            },
        )
        pointer = root / "latest_checkpointed_iteration.txt"
        temporary = pointer.with_suffix(".tmp")
        temporary.write_text(str(self.global_steps))
        temporary.replace(pointer)

    def _load_checkpoint(self):
        root = Path(self.config.trainer.default_local_dir)
        pointer = root / "latest_checkpointed_iteration.txt"
        if pointer.exists():
            path = root / f"global_step_{int(pointer.read_text())}"
            manifest = json.loads((path / "complete.json").read_text())
            if "data.pt" not in manifest["files"]:
                raise ValueError("Checkpoint has no data progress")
            for name, expected in manifest["files"].items():
                if digest(path / name) != expected:
                    raise ValueError(f"Incomplete or corrupted checkpoint: {name}")
        super()._load_checkpoint()

    @staticmethod
    def _write_generations(
        inputs,
        outputs,
        gts,
        scores,
        reward_extra_infos_dict,
        dump_path,
        global_steps,
        **kwargs,
    ):
        payloads = reward_extra_infos_dict.get("trajectory_json", [])
        if len(payloads) != len(inputs):
            raise ValueError("Evaluation must retain every original trajectory")
        trajectories = [Trajectory.from_dict(json.loads(text)) for text in payloads]
        directory = Path(dump_path)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{global_steps}.jsonl").write_text(
            "".join(json.dumps(t.to_dict()) + "\n" for t in trajectories)
        )
        atomic_json(directory / f"metrics_{global_steps}.json", summarize(trajectories))
