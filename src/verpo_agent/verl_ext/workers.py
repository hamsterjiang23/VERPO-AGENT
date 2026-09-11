"""Worker-local extension; leaves upstream registries and source implementations intact."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import torch

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id, get_device_name
from verl.workers.engine import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker
from verl.trainer.distillation.snapshot_teacher import ActorSideTeacher

from verpo_agent.provenance import atomic_json, source_identity, verify_runtime
from .loss import AgentLoss


@EngineRegistry.register(
    model_type="agent_language_model", backend="fsdp2", device="cuda"
)
class AgentFSDPEngine(FSDPEngineWithLMHead):
    def __init__(self, model_config, **kwargs):
        # Separate registry key, same HF language-model architecture.
        model_config.model_type = "language_model"
        super().__init__(model_config=model_config, **kwargs)

    def forward_step(self, micro_batch, loss_function, forward_only):
        if forward_only or not isinstance(loss_function, AgentLoss):
            return super().forward_step(micro_batch, loss_function, forward_only)
        micro_batch = micro_batch.to(get_device_id())
        dtype = getattr(self, "_autocast_dtype", torch.bfloat16)
        ctx = (
            nullcontext()
            if dtype == torch.float32
            else torch.autocast(get_device_name(), dtype=dtype)
        )
        try:
            with ctx:
                loss_function.prepare(micro_batch, self.get_data_parallel_group())
            return super().forward_step(micro_batch, loss_function, forward_only)
        finally:
            loss_function.clear()


class AgentTrainingWorker(TrainingWorker):
    def __init__(self, config):
        super().__init__(replace(config, model_type="agent_language_model"))


class AgentActorRolloutRefWorker(ActorRolloutRefWorker):
    actor_worker_cls = AgentTrainingWorker

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from omegaconf import OmegaConf

        self.agent_config = OmegaConf.to_container(
            self.config.agent.experiment, resolve=True
        )
        self.agent_identity = source_identity()
        expected = OmegaConf.to_container(
            self.config.agent.source_identity, resolve=True
        )
        if self.agent_identity != expected:
            raise ValueError("Worker source identity differs from driver")
        verify_runtime(self.agent_config["runtime"]["package_versions"])
        super().init_model()
        if (
            self.actor is None
            or self.ref is None
            or self.ref.engine.optimizer is not None
        ):
            raise ValueError(
                "Agent worker requires Student and independent forward-only reference"
            )
        for module in self.actor.engine.module.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        actor_config = replace(
            omega_conf_to_dataclass(self.config.actor), use_kl_loss=False
        )
        teacher = ActorSideTeacher(
            self.actor.engine, mode="ema", ema_decay=self.agent_config["ema_decay"]
        )
        self.actor._verpo_actor_teacher = (
            teacher  # Native successful-update and checkpoint hooks.
        )
        pad_token_id = self.actor.model_config.hf_config.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.actor.model_config.hf_config.eos_token_id
        if isinstance(pad_token_id, list):
            pad_token_id = pad_token_id[0]
        self.loss_fn = AgentLoss(
            actor_config, self.agent_config, self.ref.engine, teacher, pad_token_id or 0
        )
        self.actor.set_loss_fn(self.loss_fn)

    def _checkpoint_contract(self):
        return {
            "source": self.agent_identity,
            "experiment": self.agent_config,
            "world_size": torch.distributed.get_world_size(),
            "reference": self.agent_config["model"],
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(
        self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None
    ):
        super().save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)
        rank = torch.distributed.get_rank()
        atomic_json(
            Path(local_path) / f"agent_contract_rank_{rank}.json",
            self._checkpoint_contract(),
        )
        torch.distributed.barrier()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        rank = torch.distributed.get_rank()
        contract = json.loads(
            (Path(local_path) / f"agent_contract_rank_{rank}.json").read_text()
        )
        if contract != self._checkpoint_contract():
            raise ValueError(
                "Checkpoint source, reference, runtime, experiment or layout mismatch"
            )
        super().load_checkpoint(local_path, hdfs_path, del_local_after_load)
