# Stochastic-oracle validation

This directory adds a trajectory-sampled validation cell without modifying the
frozen VI-A/VI-B/VI-C population experiments or their artifacts.

## Scientific question

Does the population QP+G/noG separation survive a finite-trajectory stochastic
oracle, and does increasing the transition batch size reduce the observed
stationarity floor?

The training oracle is stochastic.  Protagonist and adversary policies are
separate `4-8-3` tanh--softmax networks.  Finite-horizon trajectories provide a
per-decision likelihood-ratio objective.  At the behavior parameters, repeated
differentiation is the DiCE estimator; at nearby QP stencil points, explicit
prefix importance ratios retain the policy-induced trajectory-distribution
derivative.  The curvature oracle is the same-batch autograd
Hessian--vector action,

`G_hat = D F_hat(z; batch) F_hat(z; batch)`.

QP+G/noG coefficient stencils reuse the same trajectory batch (common random
numbers).  Training never calls a best-response solver.  Exact population hard
best responses are used only at saved evaluation checkpoints.

## Frozen protocol

- Environment: `CyclicControl`.
- Methods: `QP+G`, `noG`, `EGM`.
- Simultaneous protagonist/adversary updates; no warm-up.
- Coefficient caps / EGM learning rate: `0.03`.
- Horizon: `16`.
- Transition batches: `128`, `512`, `2048`.
- Screening seeds: `3000--3004`; 40 updates.
- Formal seeds: `4100--4109`; 60 updates.  These seeds are disjoint from all
  development, numerical-sentinel, and screening queries.
- Checkpoint interval: 10 updates.
- Primary comparison: final hard-BR return, QP+G minus noG, batch 2048.
- Mechanism diagnostic: final population field norm and held-out stochastic
  field norm as functions of batch size.
- Small batches are not required to be positive; a larger residual floor is an
  expected stochastic-theory outcome.

Screening and formal results are written to separate timestamped directories
under `results/`.  Existing files are never overwritten.

## Commands

```powershell
python sentinels_dice.py
python stochastic_dice_policy.py --phase smoke
python stochastic_dice_policy.py --phase screen
python stochastic_dice_policy.py --phase formal
```

The script writes `protocol.json`, `curves.csv`, `diagnostics.csv`,
`summary.json`, and a vector PDF figure.

`stochastic_actor_critic.py` and its early smoke outputs are retained only as
an auditable rejected prototype.  Its fixed-batch finite-difference curvature
omits the trajectory-distribution derivative and is not used as evidence.

## Completed evidence

- Numerical sentinels:
  `results/sentinels-dice-20260727-121445/`.
- Screening:
  `results/screen-CyclicControl-dice-20260727-123840/`.
- Formal untouched seeds:
  `results/formal-CyclicControl-dice-20260727-124224/`.
- Formal integrity and paired-statistics audit:
  `results/formal-CyclicControl-dice-20260727-124224/formal_audit.json`.

The prespecified batch-2048 comparison confirms QP+G over noG: paired hard-BR
gain `0.3422`, 95% interval `[0.1230, 0.5614]`, `9/10` wins, exact one-sided
sign-test `p=0.010742`, and lower mean hard exploitability (`0.4287` versus
`0.9318`).  Batch 128 is a negative curvature-variance boundary and batch 512
is inconclusive.  See the project-root
`STOCHASTIC_ORACLE_VALIDATION_HANDOFF_20260727.md` for the full audit and
manuscript integration record.
