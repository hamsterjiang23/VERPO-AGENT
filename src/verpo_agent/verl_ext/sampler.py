"""Surface failed async Agent tasks instead of returning partial GRPO groups."""

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer


class AgentReplayBuffer(ReplayBuffer):
    def _sync_metadata_from_transfer_queue(self):
        super()._sync_metadata_from_transfer_queue()
        failures = {
            split: list(keys) for split, keys in self.failure_keys.items() if keys
        }
        if failures:
            raise RuntimeError(
                f"Agent generation failed; preserved raw records must be inspected: {failures}"
            )
