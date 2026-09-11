"""Actual Gloo collectives with unequal trajectory lengths and microbatches."""

from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verpo_agent.objectives import global_token_mean, token_losses


def gloo_worker(rank, rendezvous, directory):
    dist.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=2
    )
    try:
        for objective in ("residual_verpo", "feedback_opd"):
            torch.manual_seed(4)
            student = torch.randn(7, requires_grad=True)
            refs, bases, evidences = [torch.randn(7, 7) for _ in range(3)]
            selection = [0, 1] if rank == 0 else [2, 3, 4, 5, 6]
            count = torch.tensor(len(selection))
            dist.all_reduce(count)
            scalar = 0.0
            for chunk in (selection[:1], selection[1:]):
                ref, feedback = token_losses(
                    student.expand(len(chunk), -1),
                    refs[chunk],
                    bases[chunk],
                    evidences[chunk],
                    objective,
                )
                loss = global_token_mean(
                    (ref + feedback).sum(), count.item(), dp_size=2
                )
                loss.backward()
                scalar += loss.detach()
            dist.all_reduce(student.grad)
            student.grad /= 2
            dist.all_reduce(scalar)
            scalar /= 2
            oracle = student.detach().clone().requires_grad_(True)
            r, f = token_losses(oracle.expand(7, -1), refs, bases, evidences, objective)
            expected = (r + f).mean()
            expected.backward()
            torch.testing.assert_close(student.grad, oracle.grad, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(scalar, expected.detach(), atol=1e-6, rtol=1e-5)
        (Path(directory) / f"rank_{rank}.ok").write_text("both objectives passed")
    finally:
        dist.destroy_process_group()


class DistributedTests(unittest.TestCase):
    def test_two_rank_gloo_matches_single_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                gloo_worker,
                args=(str(Path(directory) / "rendezvous"), directory),
                nprocs=2,
                join=True,
            )
            self.assertEqual(len(list(Path(directory).glob("rank_*.ok"))), 2)


if __name__ == "__main__":
    unittest.main()
