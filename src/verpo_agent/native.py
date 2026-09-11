"""Project explicit Agent settings onto a pinned veRL infrastructure config."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .config import validate_config
from .provenance import ROOT, atomic_json, digest, source_identity, verify_runtime


def project_native(c, model_path, dataset_files, identity, *, evaluate=False):
    from omegaconf import OmegaConf

    c = validate_config(c)
    cfg = OmegaConf.load(
        ROOT
        / "third_party/PGR-Probe/verl/verl/trainer/config/_generated_ppo_trainer.yaml"
    )
    r, output = c["runtime"], Path(c["output_dir"])
    values = {
        "actor_rollout_ref.agent": {"experiment": c, "source_identity": identity},
        "actor_rollout_ref.model.path": str(model_path),
        "actor_rollout_ref.model.use_remove_padding": False,
        "actor_rollout_ref.model.use_fused_kernels": False,
        "actor_rollout_ref.model.trust_remote_code": False,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.model.lora_rank": 0,
        "actor_rollout_ref.model.lora.rank": 0,
        "actor_rollout_ref.actor.strategy": "fsdp2",
        "actor_rollout_ref.actor.verpo": {
            "_target_": "verl.workers.config.actor.VerpoZPDConfig",
            "enabled": False,
        },
        "actor_rollout_ref.actor.paper_baseline": {
            "_target_": "verl.workers.config.actor.PaperBaselineConfig",
            "enabled": False,
        },
        "actor_rollout_ref.actor.loss_mode": "ppo",
        "actor_rollout_ref.actor.policy_loss.loss_mode": "vanilla",
        "actor_rollout_ref.actor.use_kl_loss": True,  # Instantiate independent ref; AgentLoss disables native KL.
        "actor_rollout_ref.actor.kl_loss_coef": 0.0,
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.calculate_entropy": True,
        "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
        "actor_rollout_ref.actor.loss_scale_factor": None,
        "actor_rollout_ref.actor.ppo_mini_batch_size": c["train_batch_size"],
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.shuffle": False,
        "actor_rollout_ref.actor.use_dynamic_bsz": False,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": r[
            "micro_batch_size_per_gpu"
        ],
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": c["max_replay_tokens"]
        * r["micro_batch_size_per_gpu"],
        "actor_rollout_ref.actor.clip_ratio": c["clip_ratio"],
        "actor_rollout_ref.actor.clip_ratio_low": c["clip_ratio"],
        "actor_rollout_ref.actor.clip_ratio_high": c["clip_ratio"],
        "actor_rollout_ref.actor.optim.lr": c["learning_rate"],
        "actor_rollout_ref.actor.optim.weight_decay": 0.0,
        "actor_rollout_ref.actor.optim.betas": [0.9, 0.999],
        "actor_rollout_ref.actor.optim.lr_warmup_steps": 0,
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio": 0.0,
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
        "actor_rollout_ref.actor.optim.total_training_steps": c["train_steps"],
        "actor_rollout_ref.actor.checkpoint.async_save": False,
        "actor_rollout_ref.ref.strategy": "fsdp2",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": r[
            "micro_batch_size_per_gpu"
        ],
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz": False,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.mode": "async",
        "actor_rollout_ref.rollout.n": c["group_size"],
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.top_k": -1,
        "actor_rollout_ref.rollout.seed": c["seed"],
        "actor_rollout_ref.rollout.calculate_log_probs": True,
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": False,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": r[
            "micro_batch_size_per_gpu"
        ],
        "actor_rollout_ref.rollout.prompt_length": c["max_prompt_tokens"],
        "actor_rollout_ref.rollout.response_length": c["max_response_tokens"],
        "actor_rollout_ref.rollout.max_model_len": c["max_replay_tokens"],
        "actor_rollout_ref.rollout.gpu_memory_utilization": r[
            "vllm_memory_utilization"
        ],
        "actor_rollout_ref.rollout.tensor_model_parallel_size": r[
            "tensor_parallel_size"
        ],
        "actor_rollout_ref.rollout.multi_turn.enable": True,
        "actor_rollout_ref.rollout.agent.default_agent_loop": "agent_verpo_tools",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path": str(
            output / "agent_loop.yaml"
        ),
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 0.0,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": False,
        "actor_rollout_ref.rollout.custom": {},
        "algorithm.adv_estimator": "grpo",
        "algorithm.use_kl_in_reward": False,
        "algorithm.norm_adv_by_std_in_grpo": True,
        "algorithm.filter_groups.enable": False,
        "algorithm.rollout_correction.bypass_mode": True,
        "algorithm.rollout_correction.rollout_is": None,
        "algorithm.rollout_correction.rollout_rs": None,
        "distillation.enabled": False,
        "data.train_files": [dataset_files["train"]],
        "data.val_files": [dataset_files["test" if evaluate else "validation"]],
        "data.train_batch_size": c["train_batch_size"],
        "data.max_prompt_length": c["max_prompt_tokens"],
        "data.max_response_length": c["max_response_tokens"],
        "data.filter_overlong_prompts": False,
        "data.shuffle": True,
        "data.seed": c["seed"],
        "trainer.use_v1": True,
        "trainer.v1.trainer_mode": "sync",
        "trainer.v1.sampler.max_off_policy_threshold": 0,
        "trainer.v1.sampler.custom_sampler.path": str(
            ROOT / "src/verpo_agent/verl_ext/sampler.py"
        ),
        "trainer.v1.sampler.custom_sampler.name": "AgentReplayBuffer",
        "trainer.critic_warmup": 0,
        "trainer.nnodes": r["nnodes"],
        "trainer.n_gpus_per_node": r["gpus_per_node"],
        "trainer.total_training_steps": c["train_steps"],
        "trainer.total_epochs": c["train_steps"],
        "trainer.test_freq": c["validation_interval"],
        "trainer.save_freq": c["checkpoint_interval"],
        "trainer.val_before_train": True,
        "trainer.val_only": evaluate,
        "trainer.logger": ["console"],
        "trainer.project_name": "agent-verpo",
        "trainer.experiment_name": c["objective"],
        "trainer.default_local_dir": str(output / "checkpoints"),
        "trainer.default_hdfs_dir": None,
        "trainer.rollout_data_dir": str(output / "rollouts"),
        "trainer.validation_data_dir": str(
            output / ("evaluation" if evaluate else "validation")
        ),
        "trainer.resume_mode": "auto",
        "trainer.del_local_ckpt_after_load": False,
        "reward.reward_model.enable": False,
        "reward.reward_model.rollout.name": "vllm",
        "transfer_queue.enable": True,
    }
    for part in ("actor", "ref"):
        for key, value in {
            "strategy": "fsdp2",
            "ulysses_sequence_parallel_size": 1,
            "param_offload": False,
            "optimizer_offload": False,
            "offload_policy": False,
            "seed": c["seed"],
            "use_torch_compile": False,
        }.items():
            values[f"actor_rollout_ref.{part}.fsdp_config.{key}"] = value
    for key, value in values.items():
        OmegaConf.update(cfg, key, value, merge=False, force_add=True)
    return cfg


def run(c, *, evaluate=False):
    import torch
    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf
    from .data import native_dataset_files, read_dataset

    c = validate_config(c)
    identity = source_identity()
    runtime = verify_runtime(c["runtime"]["package_versions"])
    read_dataset(c["dataset_manifest"])
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Distributed train/evaluate requires the declared CUDA runtime; no training ran"
        )
    if torch.cuda.device_count() < c["runtime"]["gpus_per_node"]:
        raise ValueError("Visible GPU count is below the configured per-node count")
    if c["runtime"]["nnodes"] > 1 and not os.environ.get("RAY_ADDRESS"):
        raise ValueError(
            "Multi-node training requires RAY_ADDRESS and identical source/runtime on every node"
        )
    output = Path(c["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": identity,
        "runtime": runtime,
        "config": c,
        "dataset_sha256": digest(c["dataset_manifest"]),
    }
    run_manifest = output / "run_manifest.json"
    if run_manifest.exists() and json.loads(run_manifest.read_text()) != manifest:
        raise ValueError(
            "Output directory belongs to a different experiment, source or runtime"
        )
    atomic_json(run_manifest, manifest)
    model = snapshot_download(
        repo_id=c["model"]["path"], revision=c["model"]["revision"]
    )
    from transformers import AutoConfig, AutoTokenizer

    model_config = AutoConfig.from_pretrained(model, trust_remote_code=False)
    if c["max_replay_tokens"] > model_config.max_position_embeddings:
        raise ValueError("Replay context exceeds model positional capacity")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    files = native_dataset_files(c["dataset_manifest"], output / "native_data")
    loop_config = [
        {
            "name": "agent_verpo_tools",
            "_target_": "verpo_agent.verl_ext.agent_loop.AgentToolsLoop",
        }
    ]
    OmegaConf.save(OmegaConf.create(loop_config), output / "agent_loop.yaml")
    native = project_native(c, model, files, identity, evaluate=evaluate)
    if os.environ.get("RAY_ADDRESS"):
        OmegaConf.update(
            native,
            "ray_kwargs.ray_init.address",
            os.environ["RAY_ADDRESS"],
            force_add=True,
        )
    # Every node must deploy the same source tree and environment; inherited
    # PYTHONPATH alone is not treated as worker verification.
    OmegaConf.update(
        native,
        "ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH",
        os.environ.get("PYTHONPATH", ""),
        force_add=True,
    )
    OmegaConf.save(
        native,
        output / ("native_evaluation.yaml" if evaluate else "native_training.yaml"),
        resolve=True,
    )
    from .verl_ext.entry import launch

    launch(native)
