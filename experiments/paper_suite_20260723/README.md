# Population-oracle experiment suite

This directory contains the three population-oracle experiment layers used in
the journal paper:

1. `linear_geometry.py` evaluates the exactly solvable normal linear saddle
   field used in Section VI-A.
2. `markov_game_suite.py --mode tabular` evaluates exact finite-state games
   with tabular softmax policies.
3. `markov_game_suite.py --mode neural` evaluates four finite-state games with
   separate `4-8-3` tanh--softmax policies: `CyclicControl`,
   `FrequencyHopping`, `RoutingInterdiction`, and `SecurityPatrol`.

`journal_games.py` is the self-contained source for those four game instances
and the neural policy. At every QP/noG stencil point, the entropy-regularized
best responses are recomputed to the recorded Bellman tolerance. Unregularized
hard best responses are separate checkpoint-only evaluators. Thus the
performance component is the regularized policy-space Nash gap, up to the
reported Bellman residual, rather than a frozen-response surrogate.

## Frozen journal artifacts

The paper figures are regenerated from the following immutable result sets:

- `results/linear-geometry-20260823-234632/`
- `results/tabular-exact-gap-20260723-113009/`
- `results/neural-journal-four-20260824/`

Run `python reproduce.py figures` from the repository root to rebuild all paper
figures, summary tables, and the SHA-256 manifest from these artifacts.

The fixed baselines use one learning rate per method, selected on seeds
1000--1004 from `{0.001, 0.003, 0.01, 0.03}` and evaluated on disjoint seeds
40--49. The selected rates are `0.03` for GDA, EGM, and PPM-3, and `0.001` for
Adam-GDA. QP+G and noG retain coefficient caps of `0.03`.

## Full experiment commands

From the repository root:

```powershell
python experiments/paper_suite_20260723/linear_geometry.py
python experiments/paper_suite_20260723/markov_game_suite.py --mode tabular
python experiments/paper_suite_20260723/markov_game_suite.py --mode neural
python experiments/paper_suite_20260723/theory_sentinels.py
python experiments/paper_suite_20260723/audit_saved_bridge.py
```

Each experiment creates a timestamped directory under `results/`; it never
overwrites the frozen journal evidence. `tune_fixed_baselines.py` reruns the
complete independent learning-rate screen and is substantially more expensive
than regenerating the paper from frozen data. After a separate QP+G/noG run and
the baseline screen, `merge_neural_journal.py --controller-dir DIR
--tuning-summary TUNING_DIR/summary.json` validates and combines the six
methods into a new four-environment journal dataset. The root command
`python reproduce.py full` executes this complete chain automatically.

`theory_sentinels.py` checks the symmetric/skew identities, cone and box-QP
formulas, entropy performance bridge, centered-softplus properties, and the
pure-rotation curvature margin. `audit_saved_bridge.py` solves the
unregularized Shapley equations at every saved final checkpoint and recomputes
the paired sign tests and Holm adjustment.
