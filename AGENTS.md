# Agent OPSD repository instructions

## Scope and current baseline

- This repository owns Agent environments, multi-turn rollouts, global-feedback-prefixed full-trajectory Teacher replay, independent experiment configs, evaluation and reports.
- Current baseline: no step gate, no benefit-prediction network, no FEC projection. Action masks exclude non-policy targets and are not step gates.
- Teacher sees a training-only observation prefix; Student retains its original causal history. Use aligned predictor positions and action-only losses. Full-trajectory replay is the default; step-local replay is a separate future ablation.
- Fixed token weighting is the initial attribution control. Reusing the original token controller is a separate ablation, not a learned benefit network.
- Future-derived feedback is retrospective supervision; it does not inherit predictable-direction return guarantees from the archived derivation.

## Dependencies and ownership

- `third_party/PGR-Probe` is a pinned Git submodule. Never advance it automatically to main, edit it silently, or commit dirty dependency trees.
- `upstream.lock.json`, the gitlink and the checked-out dependency must agree. Update them together only for an intentional dependency upgrade.
- Use the vendored `verl` and the upstream `risk_aware_opsd` via the explicit runner. A PyPI veRL package is not an equivalent replacement for this modified version.
- Shared fixes belong upstream; Agent adapters and protocols belong in this repository. Do not inherit the old SDPO/RLCSD dataset, reward, coefficient or launcher settings without an explicit experimental decision.
- Read the dependency's own instructions before changing its source. Normal dependency use does not authorize changes to the dependency.

## Implementation and validation

- Current delivery includes deterministic tool environments, exact-token trajectories, full-trajectory feedback replay, residual VERPO/OPD objectives, and independent veRL TaskRunner/Trainer/worker/AgentLoop adapters. CPU math, real optimizer/EMA updates, native interfaces, Gloo and Ray/TransferQueue are verified; GPU/FSDP2/NCCL end-to-end training is not yet verified.
- `configs/observation_replay_design.json` remains a historical non-runnable design contract. The new `configs/agent_tools.template.json` also requires explicit experiment values before launch. The first environment is deterministic lookup/calculation, with GRPO + residual VERPO or feedback OPD, an independent frozen reference and EMA Teacher. A production model, dataset slice, runtime and hardware remain unregistered.
- Native Agent training uses full-vocabulary forward KL, fixed token weight 1, no group/step gate, no FEC, no extra KL shaping, and action-only global-token loss normalization. Do not route it through old experiment launchers.
- Teacher replay must finish and Student parameters must be restored before creating the Student autograd graph. Preserve the upstream successful-update EMA hook and checkpoint state. Use veRL's jagged TensorDict row-selection helpers, not unsupported ordinary advanced indexing.
- Validate changes with `python scripts/run_with_upstream.py -- python scripts/verify_core.py` after installing `.[cpu]`. This performs only CPU acceptance and scripted random-model replay checks, not a formal training experiment.
- Formal training is not authorized by repository creation. Future training requires a concrete user-requested experiment with preflight on the target hardware.
- Do not add placeholder training commands that return success without training.
- Verify provenance with `python scripts/preflight.py`; run `python -m unittest discover -s tests -v` for changes to the dependency checks.
- Keep raw rollouts, prompts, tool returns, masks, errors and evaluation outputs inspectable. Record source commits, runtime, decoding, Teacher schedule and selection rules for every real run.
- Store trials and large data outside Git (`outputs/`, `runs/`, `datasets/`, `checkpoints/`). Keep reports and manifests in `reports/`. Never commit credentials or personal machine paths.
