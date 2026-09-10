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

- Current delivery is an installable package scaffold, version checks, dependency runner and research docs. There is no implemented Agent trainer or environment adapter yet.
- `configs/observation_replay_design.json` is a non-runnable design contract. Do not describe it as a launchable protocol. A model, environment, dataset manifest, runtime and evaluation protocol remain undecided.
- Formal training is not authorized by repository creation. Future training requires a concrete user-requested experiment with preflight on the target hardware.
- Do not add placeholder training commands that return success without training.
- Verify provenance with `python scripts/preflight.py`; run `python -m unittest discover -s tests -v` for changes to the dependency checks.
- Keep raw rollouts, prompts, tool returns, masks, errors and evaluation outputs inspectable. Record source commits, runtime, decoding, Teacher schedule and selection rules for every real run.
- Store trials and large data outside Git (`outputs/`, `runs/`, `datasets/`, `checkpoints/`). Keep reports and manifests in `reports/`. Never commit credentials or personal machine paths.
