# ICLR experiment uplift plan

## What the current evidence establishes

The local project already contains controlled normal-form, tabular, small neural-policy, and finite-trajectory experiments.
These experiments test the mathematical mechanism, but the neural tasks use small custom Markov games and are not sufficient as the only empirical evidence for an ICLR submission.

The HPC4 archive also contains earlier PPO-RARL runs on Hopper, Walker2d, and HalfCheetah.
Those archived results are provenance, not final ICLR evidence.
In the corrected one-million-step summary, QP+G improves the matched noG endpoint on three of five HalfCheetah seeds, is statistically unresolved on Hopper, and underperforms on Walker2d.
The ICLR paper therefore must not claim universal superiority from these runs or selectively report only favorable environments.

## Dedicated ICLR protocol

### Layer A: controlled geometry

- Retain a compact rotation-ratio sweep.
- Report field energy, curvature activation, and the relative curvature contribution.
- Add an estimation-noise sweep to test whether the controller reduces curvature use when the Jacobian-vector estimate becomes unreliable.

### Layer B: exact Markov-game diagnostics

- Use a small collection of tabular games only for exact Nash-gap and best-response measurements.
- Sweep state count, action count, transition stochasticity, and rotation ratio instead of treating hand-designed game names as the primary evidence.
- Report paired results over generated games drawn from a documented, fixed distribution.

### Layer C: neural robust-control benchmarks

- Use Hopper, Walker2d, and HalfCheetah with a learned adversarial force policy.
- Use identical protagonist and adversary networks, critic architecture, rollout batches, entropy terms, training budgets, and seeds for every optimizer.
- Change only the simultaneous minimax policy-update rule.
- Compare field-only PPO/GDA, EGM, a proximal approximation, fixed field-curvature coupling, the noG ablation, and the proposed adaptive two-direction update.
- Evaluate clean return, return against the co-trained adversary, return against a separately fine-tuned best-response adversary, and late-training stability.

### Layer D: scale and robustness

- Sweep adversary force budget and policy-network width.
- Evaluate held-out dynamics perturbations and adversary budgets.
- Report curvature activation, model-prediction error, accepted-step fraction, and wall-clock overhead to explain when the method helps.

## Practical safeguard to test

The archived Walker2d result shows that an approximate local model can select harmful curvature even though the exact QP dominates the field-only candidate at the model level.
The new implementation should therefore evaluate the QP+G and noG candidates on a held-out trajectory batch and accept curvature only when the measured composite merit improves by more than a prespecified tolerance.
This is an algorithmic safeguard, not a tuning rule based on final returns.
Its tolerance and validation-batch size must be selected on separate screening seeds and frozen before final evaluation.

## Statistical protocol

- Use at least ten final seeds per environment if compute permits.
- Freeze the optimizer-specific hyperparameters using disjoint screening seeds.
- Report paired confidence intervals from matched seeds and predeclare the primary metric.
- Treat fine-tuned-adversary return as the primary robust-performance metric and approximate exploitability as a secondary metric.
- Report all prespecified environments, including neutral or negative results.

## HPC4 layout

- Active development and runs: `/scratch/jzhuangag/rarl_iclr2027`.
- Durable completed artifacts: `/project/vincentlau/jzhuangag/rarl_iclr2027`.
- Local canonical editable source: `E:/HKUST-study/vin/claude记录/RARL-Codex-Project/iclr/experiments`.
- Existing remote archives remain read-only provenance and will not be overwritten.

## Go/no-go criteria for the ICLR claim

Proceed with the strong geometry-adaptive-learning claim only if the frozen protocol shows both of the following:

1. Curvature use increases with measured rotation and decreases with unreliable curvature estimates.
2. The safeguarded method improves or statistically matches the field-only update on the primary metric across the prespecified benchmark suite while providing a clear gain on at least two rotation-rich tasks.

If these criteria fail, narrow the paper to the theoretical optimizer and controlled diagnostics rather than presenting an unsupported deep-RL performance claim.
