# VI-C handoff: neural zero-sum Markov games

Date: 2026-07-23

## Bottom line

VI-C now has three strict positive, stateful zero-sum Markov-game cells with
two neural policies: `CyclicControl`, `PursuitEvasion`, and
`FrequencyHopping`.  In the final independent seeds 30--34, QP+G beats the
independently fitted noG ablation on hard held-out best-response return for
5/5 seeds in every environment and also has lower mean hard exploitability.

This is evidence for the paper's geometric mechanism in neural policy
optimization.  It is not presented as a positive result on physical MuJoCo
RARL: the earlier Pendulum/InvertedPendulum diagnostics were weakly rotational
and did not establish a reliable QP+G advantage.

## RPS Lyapunov used in the diagnostic

For the projected RPS field

```
Pi = I - 11^T/3
F(p,q) = [-Pi M q; Pi M^T p]
A = D F
G = A F
```

the experimental merit was

```
C_pro(p) = [-min_j (p^T M)_j]_+
C_adv(q) = [ max_i (M q)_i]_+
V_RPS = 0.3 * (||F||^2 / (2 kappa_F))
        + 1.0 * ((C_pro^2 + C_adv^2) / (2 kappa_BR)).
```

The first term measures saddle-field stationarity; the second measures each
player's unilateral value-improvement opportunity against the current
opponent.  The hard min/max makes this particular diagnostic piecewise smooth,
so it is not the manuscript's global C3 Lyapunov theorem object.  The rigorous
finite-game version replaces the hard term by the entropy-regularized Nash gap
or a smooth envelope, while keeping `G = D F F` unchanged.

## Rigorous performance bridge

For any finite discounted two-player zero-sum Markov game,

```
R_BR(pi) = inf_nu J(pi,nu)
Gap_0(pi,nu) = sup_pibar J(pibar,nu) - inf_nubar J(pi,nubar)
```

and game value `v*`,

```
0 <= v* - R_BR(pi) <= Gap_0(pi,nu).
```

For `J_eps = J + eps H_pi - eps H_nu`,

```
Gap_0(pi,nu) <= Gap_eps(pi,nu)
  + eps (log |A| + log |B|) / (1-alpha).
```

Thus a global entropy-regularized gap is performance-aware up to an explicit
regularization bias.  This does not turn neural parameter stationarity into a
global Nash guarantee.  The manuscript therefore separates the exact
policy-space performance statement from the local parameter-space drift and
stationarity results.  See `ZERO_SUM_MARKOV_GAME_PROOF_AUDIT_20260723.md`.

## Final protocol

- Environments: three finite-state, three-action-per-player, action-dependent
  transition zero-sum Markov games.
- Policies: separate `4-8-3` tanh-softmax networks, 67 parameters per player.
- Return: exact differentiable discounted value solve, `alpha=0.9`.
- Regularization: causal entropy coefficient `eps=0.03`.
- Direction: exact autodiff `G = D F F`; its definition was not changed.
- Merit: normalized field energy plus a frozen soft-best-response regularized
  Nash residual.  A single frozen response bank is shared by QP/noG stencils
  and the safeguard within each outer update.
- Evaluation: independent hard, unregularized dynamic-programming best
  responses; maximum final Bellman residual `8.63e-10`.
- Final test: seeds 30--34, 60 simultaneous joint updates, no warm-up.
- Step sizes: learning rate or coefficient caps `0.03`; PPM inner steps `3`.
- Baselines: QP+G, noG, GDA, Adam-GDA, EGM, and PPM.
- Fairness boundary: initialization, outer updates, learning rate/caps, and
  checkpoints match; oracle costs are reported but not equalized.

## Final means over five seeds

Each cell is `hard BR return / hard exploitability`; higher is better for the
first number, lower for the second.

| Method | CyclicControl | PursuitEvasion | FrequencyHopping |
|---|---:|---:|---:|
| QP+G | **-0.0762 / 0.1568** | **-0.2311 / 0.5434** | **-0.0977 / 2.1599** |
| noG | -0.3329 / 0.5362 | -0.2815 / 0.6405 | -0.2779 / 2.3730 |
| GDA | -0.2222 / 0.5468 | -0.2815 / 0.6405 | -0.3407 / 2.2587 |
| Adam-GDA | -9.8508 / 18.7585 | -7.8847 / 16.3944 | -10.9008 / 18.1004 |
| EGM | -0.1703 / 0.4242 | -0.2801 / 0.6371 | -0.3333 / 2.2720 |
| PPM | -0.1750 / 0.4283 | -0.2801 / 0.6371 | -0.3344 / 2.2743 |

Paired QP+G minus noG hard-BR improvements:

- CyclicControl: mean `0.25664`, minimum seed improvement `0.01896`.
- PursuitEvasion: mean `0.05035`, minimum `0.00598`.
- FrequencyHopping: mean `0.18028`, minimum `0.04771`.

The QP curvature coefficient is positive on `0.883`, `1.000`, and `0.193` of
updates, respectively; mean G contribution ratios are `0.461`, `0.334`, and
`0.143`.  FrequencyHopping is therefore a sparse-but-effective correction
case, not a claim that G must dominate every update.

## Files and provenance

- Paper source: `main.tex`
- Bibliography: `refs.bib`
- Real result figure: `neural_markov_games_main.png`
- Selected summary: `experiments/neural_markov_games_20260723/results/vic_selected_summary.csv`
- Selection manifest: `experiments/neural_markov_games_20260723/results/vic_selected_manifest.json`
- CyclicControl/PursuitEvasion raw final run:
  `experiments/neural_markov_games_20260723/results/neural-three-env-six-method-20260723-004802`
- FrequencyHopping raw final run:
  `experiments/neural_markov_games_20260723/results/neural-three-env-six-method-20260723-005711`
- Runner and environment definitions:
  `experiments/neural_markov_games_20260723/neural_markov_six_method.py`,
  `experiments/neural_markov_games_20260723/candidate_games.py`
- Full change rationale: `CHANGELOG_CODEX_20260723.md`

`MarkovSoccer` was not counted: it improved hard-BR return on 5/5 seeds but
its mean hard exploitability (`0.97030`) was slightly worse than noG
(`0.96960`), so it failed the predeclared strict two-metric gate.

## Manuscript changes and reasons

1. Retitled and reframed the paper around two-player zero-sum Markov-game
   policy optimization, with physical RARL explicitly treated as a subclass.
   This aligns the saddle-point theory with an RL objective without pretending
   that adversarial robustness is identical to field stationarity.
2. Added the exact robust-return/Nash-gap proposition and entropy bias bound.
   This supplies the missing rigorous link from the performance-aware merit to
   protagonist worst-case return.
3. Added VI-C with the protocol, real figure, numerical table, strict gate,
   and limitations.  VI-A was not redesigned.
4. Repaired the bibliography target and verified/corrected the surfaced
   references.  No unsupported citation was invented.

## Build status and blocker

A separately named 14-page `graphicx=demo` PDF passed syntax, bibliography,
cross-reference, and visual layout inspection: no undefined citations, no
undefined references, and no fatal LaTeX errors.  Eight pre-existing overfull
boxes remain outside the new VI-C subsection.

A normal publishable PDF is still blocked by missing legacy VI-A/VI-B figures,
starting with `motivation.pdf` and later `tabular_rarl_main.png` and
`lq_aligned_main.png`.  Same-named copies of the last two found elsewhere are
explicitly documented as synthetic fake placeholders, so they were not copied
or used.  The new VI-C image is real; the demo PDF is layout-only and must not
be presented as experimental evidence.
