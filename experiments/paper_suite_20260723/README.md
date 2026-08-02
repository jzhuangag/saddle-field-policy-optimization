# Final paper experiment suite

This directory is isolated from the recovered and diagnostic experiments.  It
contains the final three-level experimental suite used by the revised paper:

1. `linear_geometry.py`: an exactly solvable normal linear saddle field that
   verifies the skew threshold and the two-direction quadratic model.
2. `markov_game_suite.py --mode tabular`: exact finite-state zero-sum Markov
   games with tabular softmax policies.
3. `markov_game_suite.py --mode neural`: the same exact Markov-game evaluator
   with separate state-conditioned neural policies for both players.

At every QP/noG stencil point, the entropy-regularized best responses are
recomputed to the recorded Bellman tolerance.  The hard evaluation best
responses are separate unregularized dynamic-programming solves.  Thus the
performance component is the actual regularized policy-space Nash gap up to a
reported numerical Bellman residual; it is not a frozen-response surrogate.

All runs write timestamped raw CSV/JSON artifacts below `results/`.  The paper
figures are copied only by `assemble_paper_results.py`, which also creates a
manifest containing the selected raw artifact paths and SHA-256 hashes.

Additional reviewer-grade checks:

- `theory_sentinels.py` checks the symmetric/skew identities, cone and box QP
  formulas, entropy performance bridge, centered-softplus properties, and the
  normalized pure-rotation curvature margin.
- `audit_saved_bridge.py` solves the unregularized Shapley equations and audits
  the performance bridge at every saved final checkpoint; it also recomputes
  exact sign tests and Holm adjustments.
- `tune_fixed_baselines.py` selects one global learning rate per fixed baseline
  on neural-game seeds 1000--1004 from the grid
  `{0.001, 0.003, 0.01, 0.03}`, then evaluates the selected rates only on final
  seeds 40--49.  QP+G/noG keep coefficient caps 0.03.
