import asyncio
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from verpo_agent.config import validate_config
from verpo_agent.data import read_dataset, write_dataset
from verpo_agent.environment import ToolEnvironment, calculate, generate_tasks
from verpo_agent.evaluation import summarize
from verpo_agent.objectives import global_token_mean, grpo_advantages, token_losses
from verpo_agent.replay import build_replay, pack_replays
from verpo_agent.rollout import collect
from verpo_agent.trajectory import Trajectory
from verpo_agent.verl_ext.loss import gather_branch, validate_microbatch
from helpers import ByteTokenizer, TinyCausalModel, config_fixture, teacher_class


def example(text="x"):
    tokenizer = ByteTokenizer()
    t = Trajectory("task", "group", "1", tokenizer.encode("prompt", True))
    t.append(
        tokenizer.encode(text) + [2], policy=True, logprobs=[-1.0] * (len(text) + 1)
    )
    t.append(tokenizer.encode(" observation "), policy=False)
    t.append(tokenizer.encode("next") + [2], policy=True, logprobs=[-1.0] * 5)
    t.events = [
        {"kind": "tool", "tool": "lookup", "result": 7},
        {"kind": "error", "message": "missing key"},
    ]
    t.termination = "final"
    return t


class TrajectoryTests(unittest.TestCase):
    def test_roundtrip_alignment_and_no_privilege_leak(self):
        t = example()
        restored = Trajectory.from_dict(json.loads(json.dumps(t.to_dict())))
        self.assertEqual(restored, t)
        replay = build_replay(t, ByteTokenizer(), 4096)
        self.assertEqual(replay.student.input_ids, t.input_ids)
        self.assertEqual(replay.reference.input_ids, replay.base.input_ids)
        self.assertEqual(replay.evidence.target_ids, t.action_targets)
        self.assertIn(2, replay.evidence.target_ids)
        self.assertNotIn(
            "Training-only", ByteTokenizer().decode(replay.student.input_ids)
        )
        self.assertIn(
            "Training-only", ByteTokenizer().decode(replay.evidence.input_ids)
        )
        for branch in (replay.student, replay.reference, replay.base, replay.evidence):
            self.assertEqual(
                [branch.input_ids[i + 1] for i in branch.predictor_indices],
                t.action_targets,
            )
        with self.assertRaises(ValueError):
            build_replay(t, ByteTokenizer(), 8)
        replay.evidence.predictor_indices[0] -= 1
        with self.assertRaises(ValueError):
            replay.validate()

    def test_jagged_reorder_serialization_microbatch(self):
        trajectories = [example("a"), example("a longer sample"), example("third")]
        replays = [build_replay(t, ByteTokenizer(), 4096) for t in trajectories]
        packed = pack_replays(replays)
        for key, rows in {
            "input_ids": [t.input_ids for t in trajectories],
            "responses": [t.response_ids for t in trajectories],
            "response_mask": [t.action_mask for t in trajectories],
        }.items():
            packed[key] = torch.nested.as_nested_tensor(
                [torch.tensor(r) for r in rows], layout=torch.jagged
            )
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "transport.pt"
            torch.save(packed, path)
            restored = torch.load(path, weights_only=False)
        from verl.utils.tensordict_utils import (
            index_select_tensor_dict,
            chunk_tensordict,
        )

        reordered = index_select_tensor_dict(restored, [2, 0, 1])
        for micro in chunk_tensordict(reordered, 3):
            validate_microbatch(micro)
        self.assertEqual(
            reordered["agent_student_input_ids"][0].tolist(), trajectories[2].input_ids
        )
        reordered["agent_student_predictor_indices"][0][0] += 1
        with self.assertRaises(ValueError):
            validate_microbatch(reordered)

    def test_invalid_masks_logprobs_empty(self):
        for change in (
            lambda t: t.action_mask.pop(),
            lambda t: t.behavior_logprobs.__setitem__(0, float("nan")),
        ):
            t = example()
            change(t)
            with self.assertRaises(ValueError):
                t.validate()
        t = example()
        t.action_mask = [0] * len(t.action_mask)
        with self.assertRaises(ValueError):
            build_replay(t, ByteTokenizer(), 4096)


class EnvironmentTests(unittest.TestCase):
    def test_safe_calculation_and_errors(self):
        self.assertEqual(calculate("(12 + 3) * 2"), "30")
        for expression in ("__import__('os')", "2 ** 20", "1 / 0", "x", "1e99"):
            with self.assertRaises((ValueError, ArithmeticError)):
                calculate(expression)
        env = ToolEnvironment(
            generate_tasks(1, dict(train=1, validation=1, test=1))["train"][0]
        )
        self.assertEqual(env.step('{"tool":"lookup","key":"absent"}')["kind"], "error")
        self.assertEqual(env.step("invalid")["kind"], "error")

    def test_multiturn_rollout_preserves_observation_and_actual_tokens(self):
        task = {
            "task_id": "unit",
            "question": "Look up a and return a * 2",
            "records": {"a": 7},
            "expression": "{a} * 2",
        }
        tok = ByteTokenizer()
        replies = iter(
            [
                '{"tool":"lookup","key":"a"}',
                '{"tool":"calculate","expression":"7 * 2"}',
                '{"answer":"14"}',
            ]
        )

        async def generate(ids, budget):
            response = tok.encode(next(replies)) + [2]
            return SimpleNamespace(token_ids=response, log_probs=[-2.0] * len(response))

        t = asyncio.run(
            collect(
                task,
                tok,
                generate,
                group_id="g",
                policy_version="1",
                max_turns=4,
                max_response_tokens=2048,
                max_action_tokens=128,
                timeout_seconds=1,
            )
        )
        self.assertEqual(t.reward, 1.0)
        self.assertEqual(len(t.events), 3)
        self.assertIn(0, t.action_mask)
        self.assertEqual(summarize([t])["success_rate"], 1.0)
        self.assertEqual(
            build_replay(t, tok, 4096).student.target_ids, t.action_targets
        )

    def test_timeout_and_length_are_recorded(self):
        task = generate_tasks(1, dict(train=1, validation=1, test=1))["train"][0]

        async def slow(ids, budget):
            await asyncio.sleep(1)

        t = asyncio.run(
            collect(
                task,
                ByteTokenizer(),
                slow,
                group_id="g",
                policy_version="1",
                max_turns=1,
                max_response_tokens=10,
                max_action_tokens=2,
                timeout_seconds=0.001,
            )
        )
        self.assertEqual(t.termination, "timeout")

        async def invalid(ids, budget):
            return SimpleNamespace(token_ids=[123] * budget, log_probs=[-1.0] * budget)

        t = asyncio.run(
            collect(
                task,
                ByteTokenizer(),
                invalid,
                group_id="g",
                policy_version="1",
                max_turns=2,
                max_response_tokens=2,
                max_action_tokens=2,
                timeout_seconds=1,
            )
        )
        self.assertEqual(t.termination, "length")

    def test_dataset_hashes_splits_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_dataset(directory, 19, dict(train=4, validation=2, test=2))
            data = read_dataset(manifest)
            self.assertEqual(sum(map(len, data.values())), 8)
            with (Path(directory) / "train.jsonl").open("a") as f:
                f.write("{}\n")
            with self.assertRaises(ValueError):
                read_dataset(manifest)
        validate_config(config_fixture())
        for field, value in (
            ("group_size", 1),
            ("ema_decay", 1),
            ("lambda_ref", None),
            ("train_steps", True),
        ):
            c = config_fixture()
            c[field] = value
            with self.assertRaises(ValueError):
                validate_config(c)


class LossTests(unittest.TestCase):
    def test_exact_formula_and_gradient_both_arms(self):
        torch.manual_seed(7)
        for objective in ("residual_verpo", "feedback_opd"):
            logits = [torch.randn(5, 11, requires_grad=True) for _ in range(4)]
            p, ref, base, evidence = logits
            r, f = token_losses(*logits, objective)
            lp = p.log_softmax(-1)
            qr, qb, qe = [x.detach().softmax(-1) for x in logits[1:]]
            expected_r = (qr * (ref.detach().log_softmax(-1) - lp)).sum(-1)
            expected_f = (
                (-(qe - qb) * lp).sum(-1)
                if objective == "residual_verpo"
                else (qe * (evidence.detach().log_softmax(-1) - lp)).sum(-1)
            )
            torch.testing.assert_close(r, expected_r, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(f, expected_f, atol=1e-6, rtol=1e-5)
            actual = torch.autograd.grad((r + f).sum(), p, retain_graph=True)[0]
            expected = torch.autograd.grad((expected_r + expected_f).sum(), p)[0]
            torch.testing.assert_close(actual, expected)
            (r + f).sum().backward()
            self.assertTrue(all(x.grad is None for x in logits[1:]))

    def test_zero_residual_and_nonpolicy_positions(self):
        full = torch.randn(17, 9, requires_grad=True)
        ids = torch.tensor([1, 5, 9])
        q = torch.randn(3, 9)
        r, f = token_losses(full[ids], q, q, q, "residual_verpo")
        torch.testing.assert_close(f, torch.zeros_like(f))
        (r + f).sum().backward()
        self.assertEqual(
            full.grad[[0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]].abs().sum(), 0
        )
        torch.testing.assert_close(
            grpo_advantages([1, 1, 0, 0], ["a", "a", "b", "b"]), torch.zeros(4)
        )
        self.assertGreater(full.grad[ids].abs().sum(), 0)

    def test_real_optimizer_ema_exception_and_resume(self):
        torch.manual_seed(12)
        model = TinyCausalModel()
        reference = copy.deepcopy(model)
        teacher = teacher_class()(
            SimpleNamespace(module=model), mode="ema", ema_decay=0.9
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        replay = build_replay(example(), ByteTokenizer(), 4096)

        def gather(module, branch):
            return gather_branch(
                module,
                [torch.tensor(branch.input_ids)],
                [torch.tensor(branch.predictor_indices)],
                pad_token_id=0,
            )

        before = {k: v.clone() for k, v in model.state_dict().items()}
        for objective in ("residual_verpo", "feedback_opd"):
            with teacher.forward_context():
                base, evidence = (
                    gather(model, replay.base),
                    gather(model, replay.evidence),
                )
            qref = gather(reference, replay.reference)
            ids = torch.tensor([replay.student.input_ids])
            p = model(ids).logits[0, replay.student.predictor_indices]
            r, f = token_losses(p, qref, base, evidence, objective)
            # Constant-reward GRPO group has zero policy loss, still updates via feedback.
            loss = global_token_mean((r + f).sum(), len(replay.student.target_ids))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            teacher.after_optimizer_step(update_applied=True)
        self.assertTrue(
            any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
        )
        self.assertEqual(teacher.optimizer_update_count, 2)
        current = {k: v.clone() for k, v in model.state_dict().items()}
        with self.assertRaises(RuntimeError):
            with teacher.forward_context():
                raise RuntimeError("injected Teacher forward failure")
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, current[k])
        teacher.after_optimizer_step(update_applied=False)
        self.assertEqual(teacher.optimizer_update_count, 2)
        with tempfile.TemporaryDirectory() as directory:
            teacher.save(directory)
            saved = teacher.fingerprint()
            other = teacher_class()(
                SimpleNamespace(module=model), mode="ema", ema_decay=0.9
            )
            other.load(directory)
            self.assertEqual(other.fingerprint(), saved)
            self.assertEqual(other.optimizer_update_count, 2)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng": torch.get_rng_state(),
                },
                Path(directory) / "cpu.pt",
            )
            payload = torch.load(Path(directory) / "cpu.pt", weights_only=False)
            resumed = TinyCausalModel()
            resumed.load_state_dict(payload["model"])
            resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01)
            resumed_optimizer.load_state_dict(payload["optimizer"])
            for m, o in ((model, optimizer), (resumed, resumed_optimizer)):
                torch.set_rng_state(payload["rng"])
                o.zero_grad()
                m(torch.tensor([[1, 3, 4]])).logits.square().mean().backward()
                o.step()
            for x, y in zip(model.parameters(), resumed.parameters()):
                torch.testing.assert_close(x, y)


if __name__ == "__main__":
    unittest.main()
