# Final paper and experiment handoff

Date: 2026-07-23

This handoff supersedes HANDOFF_20260723_VIC.md and the earlier five-seed
screening conclusions. The publishable target is output/pdf/RARL_final.pdf;
its source is main.tex, with bibliography refs.bib.

## Frozen scope and paper claim

The paper is framed as **Lyapunov-drift adaptive two-direction policy
optimization for two-player zero-sum Markov games**. Robust adversarial
reinforcement learning is an important subclass, not the only claimed
application. This scope preserves the RL objective, the unchanged curvature
direction G = D F F, and the two independently selected step sizes, while
avoiding the false claim that local parameter stationarity alone guarantees
physical-control robustness.

The experimental sections are:

1. **VI-A -- controlled normal-form geometry.** An exactly solvable normal
   linear saddle field isolates the skew threshold and validates the
   quadratic drift identity.
2. **VI-B -- exact-gap tabular Markov games.** RPS, CyclicControl, and
   FrequencyHopping remove neural approximation while retaining sequential
   Markov structure and exact dynamic-programming best responses.
3. **VI-C -- stateful Markov games with two neural policies.** Two separate
   4-8-3 tanh-softmax networks are jointly and simultaneously updated.
   CyclicControl, FrequencyHopping, and RoutingInterdiction are the strict
   10/10-seed cells; SecurityPatrol is an additional statistically confirmed
   9/10 cell. PursuitEvasion is retained as a nonconfirmed boundary case.

## Superseding result decision

The earlier 12-panel image used five seeds and counted PursuitEvasion as
positive. The final exact-gap, independent ten-seed run changes that
conclusion:

| Environment | QP+G minus noG hard-BR gain (95% CI) | wins | one-sided exact sign p | decision |
|---|---:|---:|---:|---|
| CyclicControl | 0.2056 [0.0768, 0.3345] | 10/10 | 0.00098 | strict positive |
| FrequencyHopping | 0.3205 [0.1509, 0.4900] | 10/10 | 0.00098 | strict positive |
| RoutingInterdiction | 0.1184 [0.0420, 0.1947] | 10/10 | 0.00098 | strict positive |
| SecurityPatrol | 0.1428 [0.0494, 0.2362] | 9/10 | 0.01074 | confirmed additional |
| PursuitEvasion | 0.0309 [-0.0071, 0.0690] | 7/10 | 0.17188 | not confirmed |

Every confirmed cell also has lower mean hard exploitability. The paper does
not call this criterion preregistered: it is a transparent final confirmation
rule requiring a positive paired 95% Student-t lower endpoint, one-sided exact
sign-test p < 0.05, and lower mean hard exploitability.

## Frozen protocol

- VI-B seeds: 200--209; 100 simultaneous joint updates.
- VI-C final seeds: 40--49, disjoint from screening seeds; 60 simultaneous
  joint updates.
- All baseline learning rates or QP caps: 0.03.
- Baselines: GDA, Adam-GDA, EGM, PPM-3, noG, and QP+G.
- PPM has three inner fixed-point iterations.
- No method uses protagonist warm-up or alternating player updates.
- EGM/PPM/GDA/Adam-GDA update both players; Adam-GDA means the simultaneous
  GDA game field passed through Adam's coordinate-wise moment preconditioner.
- Oracle cost is disclosed rather than falsely matched: EGM uses two field
  queries, PPM-3 three, and QP+G additionally uses one Hessian-vector product
  and a two-dimensional merit stencil.
- Soft entropy best responses are recomputed at every merit-stencil point.
  Hard, unregularized best responses are separate evaluation quantities.
- Maximum recorded soft-BR Bellman residual is below 9.00e-11; maximum
  hard-BR residual is below 9.01e-13.

## Theory audit and corrections

The final manuscript includes complete appendix proofs and makes the following
material corrections:

1. The exact dominance identity is
   Delta* - DeltaF = (b e + c d)^2/(2 c delta) + [-e]_+^2/(2c).
   Strict improvement therefore occurs iff the reduced gradient is nonzero
   or e < 0; in the usual field-descent regime e >= 0, it is equivalent to
   a nonzero reduced gradient.
2. The finite-game performance bridge is
   0 <= v* - R_BR(pi) <= Gap_0(pi,nu) <= Gap_eps(pi,nu)
   + eps(log|A|+log|B|)/(1-alpha).
3. Under finite state/action spaces, positive C3 policies, and positive
   entropy regularization, the soft Bellman fixed points and regularized gap
   are C3; this is justified by contraction, Jacobian invertibility, and the
   implicit-function theorem.
4. The composite Lyapunov theory is explicitly local in neural parameter
   space. It does not identify field-norm convergence with global robust
   performance; the policy-space gap supplies the stated performance bridge.
5. The exact box-constrained QP fallback enumerates a feasible interior point
   and the four clipped edge minimizers. Dominance is guaranteed only over
   comparator pairs lying in the same box.
6. The pure-rotation statement is now consistent with the assumptions:
   the field-only theorem requires positive field dissipation, whereas the
   curvature-improved theorem allows zero field dissipation and uses the
   reduced-gradient margin.

Numerical sentinels in
experiments/paper_suite_20260723/theory_sentinels.py verify 1000
decomposition identities, 1000 corrected dominance identities, the box-QP
solution against SciPy, and 500 random matrix-game performance bridges. They
all pass, but are diagnostics rather than substitutes for the proofs.

## Frozen raw artifacts

- VI-A:
  experiments/paper_suite_20260723/results/linear-geometry-20260723-103816
- VI-B:
  experiments/paper_suite_20260723/results/tabular-exact-gap-20260723-113009
- VI-C:
  experiments/paper_suite_20260723/results/neural-exact-gap-20260723-113325
- Theory sentinels:
  experiments/paper_suite_20260723/results/theory-sentinels-20260723-111906
- Checksums and decisions:
  output/data/final_experiment_manifest.json
- Compact tables:
  output/data/vi_b_tabular_summary.csv and
  output/data/vi_c_neural_summary.csv

## Build and verification

From the project root, run:

    latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=RARL_final main.tex

The final build has 16 pages, no undefined citations, no undefined references,
and no fatal LaTeX errors. All pages were rendered with Poppler and visually
inspected. The remaining small overfull-box warnings are at most 7.1 pt and
do not create visible text collisions; the previously visible VI-A and
appendix collisions were removed.

Final PDF SHA-256:
2A6F8BAC75463FC408CB93B1B5820D9729EF1EBAF97414C38EACC528EF1FB7F9.
It should be recomputed whenever a figure or the manuscript source changes.

## Analytic mechanism figure

The former local-source motivation.pdf was audited but not copied into the
paper. Its script mixed an illustrative coefficient choice with a noisy
trajectory simulation, so it did not match the final exact-geometry evidence
standard. It was replaced by output/pdf/fig_motivation_geometry.pdf, generated
by experiments/paper_suite_20260723/motivation_geometry.py. Its three panels
are closed-form: pure-rotation field-energy geometry, an exact normal-field
drift QP, and the exact rotation-ratio alignment threshold. The generator
asserts positive definiteness, a positive interior pair, and strict model
dominance over the pure-field, pure-curvature, and fixed-coupling
restrictions. The caption explicitly labels the figure as an analytic
mechanism illustration rather than an empirical benchmark.

All four manuscript figures were regenerated with embedded TrueType fonts.
The final paper contains no Type-3 fonts.

## Honest limitations

- The confirmed positive environments are finite zero-sum Markov games,
  including communications/control-motivated tasks, rather than standard
  MuJoCo locomotion benchmarks.
- Earlier Gymnasium Pendulum and MuJoCo InvertedPendulum disturbance wrappers
  did not yield reproducible usable skew or a reliable QP+G advantage.
- PowerControlJamming was nearly potential-like and selected zero curvature
  steps; MarkovSoccer improved hard-BR return in screening but worsened
  exploitability. Neither is counted as positive.
- The empirical evidence supports the proposed geometric mechanism under
  usable performance-aware reduced gradients; it does not claim universal
  superiority in adversarial RL.
