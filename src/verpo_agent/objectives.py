"""Exact action-level VERPO and OPD. Shared kernels come from the pinned source."""

from __future__ import annotations

import torch


def token_losses(student, reference, base, evidence, objective):
    from risk_aware_opsd.verpo_zpd import compute_fixed_teacher_token_losses

    if student.ndim != 2 or any(
        x.shape != student.shape for x in (reference, base, evidence)
    ):
        raise ValueError("Expected matching [action_tokens, vocab] logits")
    if not all(torch.isfinite(x).all() for x in (student, reference, base, evidence)):
        raise ValueError("Non-finite active logits")
    # Upstream expects [batch, sequence, vocabulary].
    ref_loss, correction, _ = compute_fixed_teacher_token_losses(
        student.unsqueeze(0),
        reference.detach().unsqueeze(0),
        base.detach().unsqueeze(0),
        evidence.detach().unsqueeze(0),
        torch.ones_like(student[:, 0]).unsqueeze(0),
        temperature=1.0,
        vocab_mode="full",
        return_interpolated_probs=False,
    )
    if objective == "residual_verpo":
        feedback = correction.squeeze(0)
    elif objective == "feedback_opd":
        log_q = evidence.detach().float().log_softmax(-1)
        feedback = (log_q.exp() * (log_q - student.float().log_softmax(-1))).sum(-1)
    else:
        raise ValueError("Unknown Agent objective")
    return ref_loss.squeeze(0), feedback


def global_token_mean(local_sum, global_action_count, dp_size=1):
    """Scale for DDP/FSDP gradient averaging; sum over microbatches, never mean."""
    if global_action_count <= 0 or dp_size <= 0:
        raise ValueError("Global action count and data-parallel size must be positive")
    return local_sum * dp_size / global_action_count


def grpo_advantages(rewards, group_ids):
    """CPU oracle for veRL's sample-std GRPO (including constant groups)."""
    if len(rewards) != len(group_ids):
        raise ValueError("Reward/group count mismatch")
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    if not torch.isfinite(rewards).all():
        raise ValueError("Non-finite rewards")
    result = torch.zeros_like(rewards)
    for group in dict.fromkeys(group_ids):
        indices = [i for i, x in enumerate(group_ids) if x == group]
        if len(indices) < 2:
            raise ValueError("GRPO requires at least two trajectories per group")
        values = rewards[indices]
        result[indices] = (values - values.mean()) / (values.std() + 1e-6)
    return result
