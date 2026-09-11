#!/usr/bin/env python3
"""Run CPU acceptance and retain inspectable random-model update artifacts."""

from __future__ import annotations

import asyncio
import argparse
import os
import copy
import importlib.metadata
import io
from pathlib import Path
import platform
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def cpu_evidence(directory):
    import torch
    from types import SimpleNamespace
    from helpers import ByteTokenizer, TinyCausalModel, teacher_class
    from verpo_agent.objectives import token_losses
    from verpo_agent.provenance import atomic_json
    from verpo_agent.replay import build_replay
    from verpo_agent.rollout import collect
    from verpo_agent.verl_ext.loss import gather_branch

    tok = ByteTokenizer()
    task = {
        "task_id": "cpu-evidence",
        "question": "Look up a, then calculate a * 2.",
        "records": {"a": 7},
        "expression": "{a} * 2",
    }
    actions = iter(
        [
            '{"tool":"lookup","key":"a"}',
            '{"tool":"calculate","expression":"7 * 2"}',
            '{"answer":"14"}',
        ]
    )

    async def generate(ids, budget):
        tokens = tok.encode(next(actions)) + [tok.eos_token_id]
        return SimpleNamespace(token_ids=tokens, log_probs=[-1.0] * len(tokens))

    trajectory = asyncio.run(
        collect(
            task,
            tok,
            generate,
            group_id="cpu",
            policy_version="random-init-7",
            max_turns=3,
            max_response_tokens=2048,
            max_action_tokens=128,
            timeout_seconds=2,
        )
    )
    replay = build_replay(trajectory, tok, 4096)
    atomic_json(directory / "trajectory.json", trajectory.to_dict())
    atomic_json(directory / "replay.json", replay.to_dict())
    metrics = {}
    for objective in ("residual_verpo", "feedback_opd"):
        torch.manual_seed(7)
        student = TinyCausalModel()
        reference = copy.deepcopy(student)
        teacher = teacher_class()(
            SimpleNamespace(module=student), mode="ema", ema_decay=0.9
        )
        optimizer = torch.optim.AdamW(student.parameters(), lr=0.001)
        before = {k: v.clone() for k, v in student.state_dict().items()}

        def score(model, branch):
            return gather_branch(
                model,
                [torch.tensor(branch.input_ids)],
                [torch.tensor(branch.predictor_indices)],
                pad_token_id=0,
            )

        qref = score(reference, replay.reference)
        with teacher.forward_context():
            q0, qe = score(student, replay.base), score(student, replay.evidence)
        p = student(torch.tensor([replay.student.input_ids])).logits[
            0, replay.student.predictor_indices
        ]
        ref_loss, feedback_loss = token_losses(p, qref, q0, qe, objective)
        loss = (ref_loss + feedback_loss).mean()
        optimizer.zero_grad()
        loss.backward()
        gradient_norm = float(
            torch.sqrt(sum(p.grad.square().sum() for p in student.parameters()))
        )
        optimizer.step()
        teacher.after_optimizer_step(update_applied=True)
        delta = float(
            torch.sqrt(
                sum(
                    (v - before[k]).square().sum()
                    for k, v in student.state_dict().items()
                )
            )
        )
        if not delta > 1e-6 or not gradient_norm > 1e-6:
            raise AssertionError("CPU objective did not actually update parameters")
        path = directory / objective
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "student": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(),
            },
            path / "cpu_checkpoint.pt",
        )
        teacher.save(path)
        metrics[objective] = {
            "reference_loss": ref_loss.mean().item(),
            "feedback_loss": feedback_loss.mean().item(),
            "loss": loss.item(),
            "gradient_norm": gradient_norm,
            "parameter_delta_l2": delta,
            "ema_updates": teacher.optimizer_update_count,
            "action_tokens": len(trajectory.action_targets),
        }
    atomic_json(directory / "metrics.json", metrics)
    return metrics


def main():
    from verpo_agent.provenance import atomic_json, source_identity, digest

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/core_validation")
    parser.add_argument(
        "--report", default="reports/core_implementation/validation.json"
    )
    args = parser.parse_args()
    directory = ROOT / args.output
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["VERPO_ACCEPTANCE_ARTIFACTS"] = str(directory.resolve())

    class RecordingResult(unittest.TextTestResult):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.passed = []

        def addSuccess(self, test):
            self.passed.append(test.id())
            super().addSuccess(test)

    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(
        stream=stream, verbosity=2, resultclass=RecordingResult
    ).run(suite)
    (directory / "tests.log").write_text(stream.getvalue())
    print(stream.getvalue())
    evidence = cpu_evidence(directory) if result.wasSuccessful() else None
    packages = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "numpy",
            "transformers",
            "ray",
            "tensordict",
            "TransferQueue",
        )
    }
    report = {
        "status": "passed" if result.wasSuccessful() else "failed",
        "tests_run": result.testsRun,
        "passed_tests": result.passed,
        "test_source_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in sorted((ROOT / "tests").glob("*.py"))
        },
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "source": source_identity(),
        "runtime": {
            "python": platform.python_version(),
            "system": platform.system(),
            "packages": packages,
        },
        "cpu_evidence": evidence,
        "artifact_directory": args.output,
        "scope": "CPU kernels, native interfaces, real Ray/TransferQueue, Gloo, and random-model optimizer updates",
        "rollout_evidence": "scripted deterministic tools; not model-generated agent performance",
        "gpu_fsdp2_nccl_validated": False,
        "formal_training_started": False,
        "task_performance_claimed": False,
    }
    atomic_json(ROOT / args.report, report)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
