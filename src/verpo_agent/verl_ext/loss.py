"""veRL's two-phase logits processor/loss interface, with pre-forward Teachers."""

from __future__ import annotations

from contextlib import contextmanager
import torch

from verpo_agent.objectives import global_token_mean, token_losses


@contextmanager
def evaluation_mode(module):
    modes = [(m, m.training) for m in module.modules()]
    module.eval()
    try:
        yield
    finally:
        for child, training in modes:
            child.training = training


def gather_branch(module, rows, indices, *, pad_token_id):
    """A single padded forward per branch and rank; retain only action logits."""
    from torch.nn.utils.rnn import pad_sequence

    lengths = [row.numel() for row in rows]
    ids = pad_sequence(rows, batch_first=True, padding_value=pad_token_id)
    position = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
    mask = position < torch.tensor(lengths, device=ids.device).unsqueeze(-1)
    with evaluation_mode(module), torch.no_grad():
        output = module(
            input_ids=ids,
            attention_mask=mask.long(),
            position_ids=position.masked_fill(~mask, 0),
            use_cache=False,
        )
        return torch.cat(
            [output.logits[i, predictors] for i, predictors in enumerate(indices)]
        ).detach()


def validate_microbatch(data):
    rows = list(data["input_ids"].unbind())
    for i, original in enumerate(rows):
        student = data["agent_student_input_ids"][i]
        if not torch.equal(student, original):
            raise ValueError("Replay Student tokens differ from the actual rollout")
        expected = data["agent_student_target_ids"][i]
        mask = data["response_mask"][i].bool()
        response = data["responses"][i]
        if not torch.equal(response[mask], expected):
            raise ValueError("Response mask differs from replay targets")
        predicted = mask.nonzero().flatten() + len(original) - len(response) - 1
        if not torch.equal(predicted, data["agent_student_predictor_indices"][i]):
            raise ValueError("Student predictor mapping differs from response mask")
        for branch in ("student", "reference", "base", "evidence"):
            ids = data[f"agent_{branch}_input_ids"][i]
            attention = data[f"agent_{branch}_attention_mask"][i]
            positions = data[f"agent_{branch}_position_ids"][i]
            if not torch.equal(attention, torch.ones_like(ids)) or not torch.equal(
                positions, torch.arange(len(ids), device=ids.device)
            ):
                raise ValueError("Replay attention or position metadata mismatch")
            indices = data[f"agent_{branch}_predictor_indices"][i]
            targets = data[f"agent_{branch}_target_ids"][i]
            if indices.numel() != expected.numel() or not torch.equal(
                targets, expected
            ):
                raise ValueError("Teacher action target identity mismatch")
            if (
                (indices < 0).any()
                or (indices + 1 >= len(ids)).any()
                or not torch.equal(ids[indices + 1], targets)
            ):
                raise ValueError("Invalid Teacher predictor mapping")
    return rows


class AgentLoss:
    def __init__(
        self, actor_config, experiment, reference_engine, teacher, pad_token_id
    ):
        self.actor_config = actor_config
        self.experiment = experiment
        self.reference_engine = reference_engine
        self.teacher = teacher
        self.pad_token_id = pad_token_id
        self.cache = None

    def prepare(self, data, dp_group=None):
        # Every rank validates before any rank enters a sharded model forward.
        error = None
        try:
            validate_microbatch(data)
        except (ValueError, IndexError, RuntimeError) as exc:
            error = str(exc)
        flag = torch.tensor(int(error is not None), device=data["input_ids"].device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                flag, op=torch.distributed.ReduceOp.MAX, group=dp_group
            )
        if flag.item():
            from pathlib import Path
            from verpo_agent.provenance import atomic_json

            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
            directory = Path(self.experiment["output_dir"]) / "replay_errors"
            atomic_json(directory / f"rank_{rank}.json", {"error": error, "rank": rank})
            torch.save(data.cpu(), directory / f"rank_{rank}.pt")
            raise ValueError(
                error or "Replay validation failed on another data-parallel rank"
            )

        def forward(engine, branch):
            return gather_branch(
                engine.module,
                list(data[f"agent_{branch}_input_ids"].unbind()),
                list(data[f"agent_{branch}_predictor_indices"].unbind()),
                pad_token_id=self.pad_token_id,
            )

        with self.reference_engine.eval_mode():
            reference = forward(self.reference_engine, "reference")
        # This context must exit BEFORE the Student graph is constructed.
        with self.teacher.forward_context():
            base = forward(self.teacher.engine, "base")
            evidence = forward(self.teacher.engine, "evidence")
        self.cache = reference, base, evidence

    def clear(self):
        self.cache = None

    def __call__(
        self, model_output=None, data=None, dp_group=None, student_logits=None, **kwargs
    ):
        from verl.utils import tensordict_utils as tu
        from verl.utils.metric import AggregationType, Metric
        from verl.workers.utils.losses import ppo_loss

        if student_logits is not None:
            if self.cache is None:
                raise RuntimeError(
                    "Teacher logits must be prepared before Student forward"
                )
            offsets, indices = 0, []
            for row, predictors in zip(
                data["input_ids"].unbind(),
                data["agent_student_predictor_indices"].unbind(),
            ):
                indices.append(predictors + offsets)
                offsets += len(row)
            indices = torch.cat(indices)
            flat = student_logits.reshape(-1, student_logits.shape[-1])
            if flat.shape[0] != offsets:
                raise ValueError(
                    "Student logits must be flattened over true sequence lengths"
                )
            reference, feedback = token_losses(
                flat[indices], *self.cache, self.experiment["objective"]
            )

            def scatter(values):
                return (
                    flat.new_zeros(flat.shape[0], dtype=torch.float32)
                    .scatter(0, indices, values)
                    .unsqueeze(0)
                )

            return {
                "agent_reference_loss": scatter(reference),
                "agent_feedback_loss": scatter(feedback),
            }
        pg, metrics = ppo_loss(
            self.actor_config, model_output=model_output, data=data, dp_group=dp_group
        )
        count = tu.get_non_tensor_data(data, "batch_num_tokens", default=0)
        dp_size = tu.get_non_tensor_data(data, "dp_size", default=0)

        def reduce(name):
            value = model_output[name]
            values = value.values() if value.is_nested else value
            return global_token_mean(values.sum(), count, dp_size)

        reference, feedback = (
            reduce("agent_reference_loss"),
            reduce("agent_feedback_loss"),
        )
        metrics["agent/reference_loss"] = Metric(
            value=reference.detach(), aggregation=AggregationType.SUM
        )
        metrics["agent/feedback_loss"] = Metric(
            value=feedback.detach(), aggregation=AggregationType.SUM
        )
        total = (
            pg
            + self.experiment["lambda_ref"] * reference
            + self.experiment["lambda_feedback"] * feedback
        )
        if not torch.isfinite(total):
            raise ValueError("Non-finite Agent objective")
        return total, metrics
