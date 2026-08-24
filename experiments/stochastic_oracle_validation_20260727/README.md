# Finite-trajectory oracle validation

This directory implements the Section VI-D validation with sampled
trajectories. It tests whether the population QP+G/noG separation remains
visible when both field and curvature directions are estimated from finite
batches.

The protagonist and adversary use separate `4-8-3` tanh--softmax policies.
Finite-horizon trajectories provide a per-decision likelihood-ratio objective.
At the behavior parameters, repeated differentiation is the DiCE estimator;
at nearby QP stencil points, prefix importance ratios preserve the derivative
of the policy-induced trajectory distribution. The curvature query is the
same-batch Hessian--vector action

```text
G_hat = D F_hat(z; batch) F_hat(z; batch).
```

QP+G and noG reuse each trajectory batch across their coefficient stencils
(common random numbers). Training does not call a best-response solver;
population hard best responses and field norms are checkpoint-only evaluators.

## Frozen protocol

- Environment: `CyclicControl`.
- Methods: QP+G, noG, and EGM.
- Simultaneous updates, with no warm-up.
- QP caps and EGM learning rate: `0.03`.
- Horizon: 16.
- Transition batches: 128, 512, and 2048.
- Screening: seeds 3000--3004 for 40 updates.
- Formal evaluation: disjoint seeds 4100--4109 for 60 updates.
- Checkpoint interval: 10 updates.

The reported batch-2048 table compares QP+G with noG on worst-case return,
exploitability, and population field norm. All paired 95% intervals are
two-sided Student-t intervals on matched seeds and are oriented so that a
positive value favors QP+G. The exact source rows and recomputed statistics are
recorded in `output/data/vi_d_finite_trajectory_table.json`.

## Commands

From the repository root:

```powershell
python experiments/stochastic_oracle_validation_20260727/sentinels_dice.py
python experiments/stochastic_oracle_validation_20260727/stochastic_dice_policy.py --phase smoke
python experiments/stochastic_oracle_validation_20260727/stochastic_dice_policy.py --phase screen
python experiments/stochastic_oracle_validation_20260727/stochastic_dice_policy.py --phase formal
python experiments/stochastic_oracle_validation_20260727/audit_formal.py experiments/stochastic_oracle_validation_20260727/results/formal-CyclicControl-dice-20260727-124224
python experiments/stochastic_oracle_validation_20260727/make_vi_d_table.py
```

The full formal run is the most expensive command. For a fast integrity check,
use the frozen result directory and rerun only `make_vi_d_table.py`.

The formal batch-2048 comparison gives a paired worst-case-return improvement
of 0.3422 with 95% interval [0.1230, 0.5614], lowers mean exploitability from
0.9318 to 0.4287, and lowers the population field norm from 0.3769 to 0.1694.
