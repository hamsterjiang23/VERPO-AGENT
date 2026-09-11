from contextlib import nullcontext
from dataclasses import replace
import tempfile
from types import SimpleNamespace
import unittest

import torch

from helpers import ByteTokenizer, TinyCausalModel, config_fixture, teacher_class
from test_core import example
from verpo_agent.native import project_native
from verpo_agent.replay import build_replay, pack_replays
from verpo_agent.verl_ext.loss import AgentLoss


class NativeContractTests(unittest.TestCase):
    def test_native_entry_imports_and_worker_selection(self):
        from verpo_agent.verl_ext.entry import AgentTaskRunner
        from verpo_agent.verl_ext.trainer import AgentTrainer
        from verpo_agent.verl_ext.workers import (
            AgentActorRolloutRefWorker,
            AgentTrainingWorker,
            AgentFSDPEngine,
        )
        from verpo_agent.verl_ext.agent_loop import AgentToolsLoop
        from verpo_agent.verl_ext.sampler import AgentReplayBuffer

        self.assertIs(AgentActorRolloutRefWorker.actor_worker_cls, AgentTrainingWorker)
        self.assertTrue(callable(AgentTaskRunner.run))
        self.assertTrue(callable(AgentTrainer._update_actor))
        self.assertTrue(callable(AgentFSDPEngine.forward_step))
        self.assertTrue(callable(AgentToolsLoop.run))
        self.assertTrue(callable(AgentReplayBuffer.sample))

    def test_worker_accepts_frozen_native_config(self):
        from unittest.mock import patch
        from verl.workers.config import TrainingWorkerConfig
        from verl.workers.engine_workers import TrainingWorker
        from verpo_agent.verl_ext.workers import AgentTrainingWorker

        original = TrainingWorkerConfig(model_type="language_model")
        with patch.object(TrainingWorker, "__init__", return_value=None) as parent:
            AgentTrainingWorker(original)
        self.assertEqual(original.model_type, "language_model")
        self.assertEqual(parent.call_args.args[0].model_type, "agent_language_model")

    def test_native_projection_disables_legacy_experiments(self):
        from omegaconf import OmegaConf
        from verl.utils.config import omega_conf_to_dataclass

        config = project_native(
            config_fixture(),
            "/model",
            dict(train="train", validation="val", test="test"),
            {},
        )
        resolved = OmegaConf.to_container(config, resolve=True)
        actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
        self.assertFalse(actor.verpo.enabled)
        self.assertFalse(actor.paper_baseline.enabled)
        self.assertEqual(actor.loss_agg_mode, "token-mean")
        self.assertEqual(actor.strategy, "fsdp2")
        self.assertEqual(config.actor_rollout_ref.agent.experiment.ema_decay, 0.9)
        self.assertTrue(config.algorithm.rollout_correction.bypass_mode)
        self.assertFalse(config.algorithm.filter_groups.enable)
        self.assertEqual(resolved["data"]["val_files"], ["val"])
        self.assertNotIn("~/", str(resolved["data"]["train_files"]))

    def test_actual_verl_loss_interface_and_teacher_order(self):
        from verl.utils import tensordict_utils as tu
        from verl.utils.config import omega_conf_to_dataclass

        torch.manual_seed(31)
        experiment = config_fixture()
        native = project_native(
            experiment, "/model", dict(train="train", validation="val", test="test"), {}
        )
        actor_config = replace(
            omega_conf_to_dataclass(native.actor_rollout_ref.actor), use_kl_loss=False
        )
        for objective in ("residual_verpo", "feedback_opd"):
            experiment["objective"] = objective
            model = TinyCausalModel()
            reference = TinyCausalModel()
            engine = SimpleNamespace(module=model)
            teacher = teacher_class()(engine, mode="ema", ema_decay=0.9)
            ref_engine = SimpleNamespace(module=reference, eval_mode=nullcontext)
            trajectories = [example("a"), example("a little longer")]
            replays = [build_replay(t, ByteTokenizer(), 4096) for t in trajectories]
            data = pack_replays(replays)
            columns = {
                "input_ids": [t.input_ids for t in trajectories],
                "responses": [t.response_ids for t in trajectories],
                "prompts": [t.prompt_ids for t in trajectories],
                "response_mask": [t.action_mask for t in trajectories],
                "loss_mask": [t.action_mask for t in trajectories],
                "old_log_probs": [t.behavior_logprobs for t in trajectories],
                "advantages": [[0.0] * len(t.response_ids) for t in trajectories],
            }
            for key, rows in columns.items():
                data[key] = torch.nested.as_nested_tensor(
                    [torch.tensor(row) for row in rows], layout=torch.jagged
                )
            total_actions = sum(sum(t.action_mask) for t in trajectories)
            tu.assign_non_tensor(
                data,
                dp_size=1,
                batch_num_tokens=total_actions,
                global_batch_size=2,
                max_response_len=max(len(t.response_ids) for t in trajectories),
            )
            loss = AgentLoss(actor_config, experiment, ref_engine, teacher, 0)
            loss.prepare(data)
            state = {k: v.clone() for k, v in model.state_dict().items()}
            # Padded causal forward, flattened just as the pinned no-rmpad engine does.
            padded = torch.nn.utils.rnn.pad_sequence(
                list(data["input_ids"].unbind()), batch_first=True
            )
            logits = model(padded).logits
            flat = torch.cat(
                [
                    logits[i, : len(row)]
                    for i, row in enumerate(data["input_ids"].unbind())
                ]
            )
            result = loss(student_logits=flat.unsqueeze(0), data=data)
            labels = data["input_ids"].values().roll(-1)
            lp = flat.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
            outputs = {
                "log_probs": torch.nested.nested_tensor_from_jagged(
                    lp, data["input_ids"].offsets()
                )
            }
            for name, values in result.items():
                outputs[name] = torch.nested.nested_tensor_from_jagged(
                    values.squeeze(0), data["input_ids"].offsets()
                )
            total, metrics = loss(model_output=outputs, data=data)
            total.backward()
            self.assertTrue(torch.isfinite(total))
            self.assertGreater(
                sum(p.grad.abs().sum().item() for p in model.parameters()), 0
            )
            self.assertTrue(all(p.grad is None for p in reference.parameters()))
            self.assertIn("agent/feedback_loss", metrics)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, state[key])
            loss.clear()
            self.assertIsNone(loss.cache)

    def test_real_transfer_queue_roundtrip(self):
        import ray
        import transfer_queue as tq
        from omegaconf import OmegaConf
        from verl.utils.tensordict_utils import chunk_tensordict

        replays = [
            build_replay(example("first"), ByteTokenizer(), 4096),
            build_replay(example("different length"), ByteTokenizer(), 4096),
        ]
        fields = pack_replays(replays)
        # This starts only a local CPU Ray cluster, not a trainer or model server.
        with tempfile.TemporaryDirectory(prefix="atq-", dir="/tmp") as directory:
            ray.init(
                num_cpus=4,
                include_dashboard=False,
                _temp_dir=directory,
                logging_level="ERROR",
            )
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
                tq.kv_batch_put(
                    keys=["row0", "row1"], partition_id="agent_test", fields=fields
                )
                restored = tq.kv_batch_get(
                    keys=["row1", "row0"],
                    partition_id="agent_test",
                    select_fields=list(fields.keys()),
                )
                micros = chunk_tensordict(restored, 2)
                for micro, expected in zip(micros, reversed(replays)):
                    for branch in ("student", "reference", "base", "evidence"):
                        for field in (
                            "input_ids",
                            "predictor_indices",
                            "target_ids",
                            "attention_mask",
                            "position_ids",
                        ):
                            self.assertEqual(
                                micro[f"agent_{branch}_{field}"][0].tolist(),
                                getattr(getattr(expected, branch), field),
                            )
                self._check_trainer_queue_steps(tq, directory)
            finally:
                tq.close()
                ray.shutdown()

    def _check_trainer_queue_steps(self, tq, directory):
        from verl.utils import tensordict_utils as tu
        from verpo_agent.verl_ext.trainer import AgentTrainer
        from verpo_agent.verl_ext.loss import validate_microbatch

        trajectories = [example("good"), example("wrong")]
        trajectories[0].reward = 1.0
        fields = tu.get_tensordict(
            {},
            non_tensor_dict={
                "uid": ["same_group", "same_group"],
                "extra_fields": [
                    {"agent_trajectory": t.to_dict()} for t in trajectories
                ],
            },
        )
        fields.batch_size = [2]
        for name, rows in {
            "prompts": [t.prompt_ids for t in trajectories],
            "responses": [t.response_ids for t in trajectories],
            "input_ids": [t.input_ids for t in trajectories],
            "response_mask": [t.action_mask for t in trajectories],
            "rollout_log_probs": [t.behavior_logprobs for t in trajectories],
            "rm_scores": [
                [0.0] * (len(t.response_ids) - 1) + [t.reward] for t in trajectories
            ],
        }.items():
            fields[name] = torch.nested.as_nested_tensor(
                [torch.tensor(row) for row in rows], layout=torch.jagged
            )
        batch = tq.kv_batch_put(
            keys=["train0", "train1"], partition_id="native_train", fields=fields
        )
        # Real methods on a CPU harness; only the GPU worker RPC is replaced.
        trainer = object.__new__(AgentTrainer)
        config = config_fixture()
        config.update(output_dir=directory, train_batch_size=1)
        trainer.config = project_native(
            config, "/model", dict(train="train", validation="val", test="test"), {}
        )
        trainer.tokenizer = ByteTokenizer()
        trainer.global_steps = 1
        calls = []

        def update(meta):
            self.assertTrue(meta.extra_info["actor_loss_use_full_logits"])
            from verl.utils.transferqueue_utils import (
                kv_batch_meta2batch_meta,
                _meta_to_realdata,
            )

            received = _meta_to_realdata(kv_batch_meta2batch_meta(meta))
            validate_microbatch(received)
            self.assertEqual(
                received["old_log_probs"][0].tolist(), trajectories[0].behavior_logprobs
            )
            self.assertGreater(received["advantages"][0].max(), 0)
            self.assertLess(received["advantages"][1].min(), 0)
            calls.append(meta)
            return {"metrics": {}}

        trainer.actor_rollout_wg = SimpleNamespace(update_actor=update)
        batch = trainer._compute_old_log_prob(batch, {})
        batch = trainer._compute_advantage(batch, {})
        trainer._update_actor(batch, {})
        self.assertEqual(len(calls), 1)
        from pathlib import Path
        import json

        trainer._log_rollout_data(batch, {}, Path(directory) / "rollouts")
        recorded = [
            json.loads(line)
            for line in (Path(directory) / "rollouts/1.jsonl").read_text().splitlines()
        ]
        self.assertEqual(recorded, [t.to_dict() for t in trajectories])
        metrics = {}
        trainer._compute_metrics(batch, metrics, {"update": 0.1}, 1, 0)
        self.assertEqual(metrics["training/success_rate"], 0.5)

        # Evaluation schedules the same real queue rows, never adds privileged input.
        trajectories[1].task_id = "second_task"
        evaluation_rows = tu.get_tensordict(
            {},
            non_tensor_dict={
                "extra_fields": [
                    {"agent_trajectory": t.to_dict()} for t in trajectories
                ]
            },
        )
        evaluation_rows.batch_size = [2]
        evaluation_batches = []

        def generate(data):
            self.assertTrue(tu.get_non_tensor_data(data, "validate", default=False))
            self.assertEqual(len(data), 2)
            evaluation_batches.append(
                tq.kv_batch_put(
                    keys=["val0", "val1"], partition_id="val", fields=evaluation_rows
                )
            )

        trainer.agent_loop_manager = SimpleNamespace(generate_sequences=generate)
        trainer.replay_buffer = SimpleNamespace(
            sample=lambda **kwargs: (
                evaluation_batches[-1],
                {},
            )
        )
        trainer.val_dataloader = [
            {"raw_prompt": [[{"role": "user", "content": "query"}]] * 2}
        ]
        self.assertEqual(trainer._validate()["val/success_rate"], 0.5)
        self.assertTrue((Path(directory) / "validation/1.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
