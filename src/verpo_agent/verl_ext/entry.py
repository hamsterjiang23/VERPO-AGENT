"""Concrete veRL TaskRunner; no old experimental launcher is invoked."""

from __future__ import annotations

import ray
from omegaconf import OmegaConf
from verl.trainer.main_ppo import run_ppo

from verpo_agent.provenance import source_identity, verify_runtime


class AgentTaskRunner:
    def init_agent_loop_manager(self):
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ

        self.agent_loop_manager = AgentLoopManagerTQ.create(
            config=self.config,
            llm_client=self.trainer.get_llm_client(),
            teacher_client=None,
            reward_loop_worker_handles=self.trainer.get_reward_handles(),
        )

    def run(self, config):
        import transfer_queue as tq
        from .trainer import AgentTrainer

        if source_identity() != OmegaConf.to_container(
            config.actor_rollout_ref.agent.source_identity, resolve=True
        ):
            raise ValueError("TaskRunner source differs from driver")
        verify_runtime(
            OmegaConf.to_container(
                config.actor_rollout_ref.agent.experiment.runtime.package_versions
            )
        )
        self.config = config
        tq.init(config.transfer_queue)
        try:
            self.trainer = AgentTrainer(config)
            self.trainer.init()
            self.init_agent_loop_manager()
            self.trainer.fit(self.agent_loop_manager)
        finally:
            tq.close()


def launch(native):
    run_ppo(native, ray.remote(num_cpus=1)(AgentTaskRunner))
