# Major-revision handoff — 2026-07-24

## Status

The review-level revision is complete.  The canonical manuscript is
`main.tex`; the stable rendered paper is `output/pdf/RARL_final.pdf`.
`RECOVERY_HANDOFF.md` and the recovered Claude conversations remain provenance
records, not current instructions or current experimental evidence.

## Project relocation — 2026-07-26

The complete canonical project was moved from the former `C:`-drive Codex
output directory to:

`E:\HKUST-study\vin\claude记录\RARL-Codex-Project`

The old project directory no longer exists.  Current experiment manifests,
summary JSON files, and build logs were rewritten to the new root and validated;
historical paths embedded in recovered conversations and provenance attachments
were intentionally left unchanged.  The project-level `AGENTS.md` now requires
all project sources, experiments, logs, scratch files, rendered pages, and final
artifacts to remain under this E-drive project root.

The paper has not been rewritten from scratch.  The original two-direction
idea and the definition

\[
G(z)=\nabla F(z)F(z)
\]

are unchanged.  The revision repairs the assumptions and rate statement,
replaces recovery placeholders with auditable experiment artifacts, and narrows
claims where the available theory or evidence is local.

## Paper scope and experiment organization

Current title:

> Lyapunov-Drift Adaptive Two-Direction Policy Optimization for Two-Player
> Zero-Sum Markov Games

The three experiment subsections are:

1. **VI-A, controlled normal-form geometry.**  An exactly solvable normal
   saddle field checks the symmetric/antisymmetric decomposition, the
   `|sigma|>|mu|` field-energy threshold, the pure-rotation mechanism, and the
   exact QP drift identity.  The mechanism figure is retained and explicitly
   labelled as a closed-form illustration rather than an empirical benchmark.
2. **VI-B, exact-gap tabular Markov games.**  RPS, CyclicControl, and
   FrequencyHopping use exact finite-state value and best-response solves.
3. **VI-C, stateful Markov games with two neural policies.**  CyclicControl,
   FrequencyHopping, RoutingInterdiction, SecurityPatrol, and PursuitEvasion
   use separate protagonist and adversary neural networks and simultaneous
   joint updates.

The original Gymnasium Pendulum and MuJoCo InvertedPendulum disturbance
wrappers are kept as negative evidence in the manuscript: they did not produce
reproducible usable skew or a QP+G advantage.  The paper therefore does not
claim that every adversarial continuous-control task is favorable.

## Theory repairs

### Performance bridge and policy space

- Added bounded rewards and made all policy-space extrema explicit over
  stationary randomized Markov policies.
- Proved, for finite action sets,

  \[
  0\le v^\star-R_{\rm BR}(\pi)
  \le {\rm Gap}_0(\pi,\nu)
  \le {\rm Gap}_\varepsilon(\pi,\nu)
  +\frac{\varepsilon(\log|\mathcal A|+\log|\mathcal B|)}{1-\alpha}.
  \]

- Added a smoothness proposition for finite entropy-regularized games using
  contraction of the soft Bellman operator and the implicit-function theorem.
- Kept the distinction between a global policy-space gap and local neural
  parameter-space stationarity throughout the abstract, theorem discussion,
  experiments, and conclusion.

### Composite Lyapunov zero set

- Defined local improvement sets as nonempty compact neighborhoods containing
  the current parameters, so the local gaps exist and are nonnegative.
- Replaced the ordinary softplus by the centered softplus

  \[
  \widetilde\sigma_\epsilon(u)
  =\epsilon\log\!\frac{1+\exp(u/\epsilon)}2,\qquad u\ge0,
  \]

  which is nonnegative and satisfies
  `tilde_sigma_epsilon(0)=0`.  This removes the positive offset that made a
  Lyapunov contraction inequality impossible at the target.
- Stated sufficient differentiability conditions: unique strict interior
  optimizers, nonsingular second-order matrices, and smooth feasible-set
  parameterizations.  Hard projection/acceptance is covered only on a fixed
  active set or after smooth replacement.
- Made the zero-set implication one-directional: `V(z)=0` implies `F(z)=0`,
  but a first-order stationary point may retain a positive local gap.

### Local convergence assumptions

- The admissible set is open with compact closure contained in the smooth
  domain.  Iterate containment is an explicit local hypothesis and may be
  enforced by the stated trust-region/backtracking safeguard.
- Replaced the incorrect suggestion that bounded gap gradients yield an
  `O(||F||^2)` compatibility bound.  The paper now identifies this as a genuine
  local error-bound/compatibility condition requiring the adverse directional
  components to vanish at least linearly with `F`.
- Local geometric contraction is invoked only when the composite merit
  vanishes on the target component and a composite PL inequality holds.

### Curvature-rate theorem

- Removed the scale-incompatible raw curvature margin.
- The rate theorem now assumes a feasible normalized-dominance margin on the
  *actual additional predicted decrease*:

  \[
  \mathfrak D_{G,k}
  =\widehat\Delta_k^\star-\widehat\Delta_k^F
  =
  \frac{(\hat b_k\hat e_k+\hat c_k\hat d_k)^2}
       {2\hat c_k\hat\delta_k}
  +\frac{[-\hat e_k]_+^2}{2\hat c_k}
  \ge C_G\Phi(z_k).
  \]

- The theorem explicitly requires exact coefficients and acceptance of the
  feasible unconstrained-cone minimizer without clipping or backtracking.
- Under this condition, the effective dissipation denominator changes from
  `bar_beta*mu_Phi` to `bar_beta*mu_Phi+C_G`.
- For the canonical unit-frequency pure rotation with field-energy merit,
  the paper proves `D_G=Phi/2`, so `C_G=1/2` at every nonstationary radius.
- Rotation, non-collinearity, and skew dominance are now described as
  diagnostics, not as sufficient conditions for a performance-aware gain.
  Approximate-coefficient probability claims require a separate concentration
  result and strict feasibility slack.

### Proof completion

- Expanded the PPM and EGM `O(s^3)` derivations with a Banach fixed-point
  argument and explicit local remainder bounds.
- Rechecked the symmetric/skew identities, closed-form cone solution, exact
  box fallback, dominance-gap identity, pathwise drift, conditional drift,
  ergodic bound, curvature-improved bound, contraction recursion, and
  coefficient perturbation proof.

## Experiment and statistical repairs

- Every QP/noG performance stencil recomputes both entropy-regularized best
  responses.  The performance component is no longer a frozen-response
  surrogate.
- Hard BR return and hard exploitability use separate unregularized dynamic
  programming.
- The saved checkpoints are audited against independently solved
  unregularized Shapley values.
- Final confirmation is QP+G versus noG using paired seeds.  Across the five
  neural cells, confirmation requires:
  1. a positive lower endpoint of the descriptive paired 95% Student-t
     interval;
  2. a one-sided exact sign test with Holm-adjusted `p<0.05`;
  3. lower mean hard exploitability.
  Because Holm significance is necessary, the compound rule cannot add a
  rejection and retains familywise type-I error control at 0.05.
- Four neural cells are confirmed: CyclicControl, FrequencyHopping,
  RoutingInterdiction, and SecurityPatrol.  PursuitEvasion remains
  nonconfirmed (`7/10` BR wins) even though its mean QP+G performance is better.
- Tabular Holm-adjusted sign-test values are `0.0029297`, `0.0029297`, and
  `0.0107422`.
- Neural Holm-adjusted values are `0.0048828` for each of the three strict
  `10/10` cells, `0.0214844` for SecurityPatrol, and `0.171875` for
  PursuitEvasion.

## Independent fixed-baseline tuning

Script: `experiments/paper_suite_20260723/tune_fixed_baselines.py`

- Learning-rate grid: `{0.001, 0.003, 0.01, 0.03}`.
- Tuning seeds: `1000--1004`.
- Final reporting seeds: `40--49`, never used for selection.
- One global learning rate is selected per method over all five environments.
- Score: mean final hard-BR gain normalized by initial hard exploitability,
  with lower normalized final exploitability and then smaller learning rate as
  tie breakers.
- All updates remain simultaneous and use no warm-up.
- Selected rates:
  - GDA: `0.03`
  - Adam-GDA: `0.001`
  - EGM: `0.03`
  - PPM-3: `0.03`
- QP+G and noG retain coefficient caps `0.03`.

The independently tuned final table is now Table III.  QP+G retains a better
mean hard-BR/exploitability pair than every tuned fixed baseline in all five
reported environments.  This is a descriptive sensitivity result; the four
formal confirmations remain the paired QP+G/noG decisions above.

Artifact:

`experiments/paper_suite_20260723/results/tuned-baselines-20260724-013019/summary.json`

The run took 6306.47 seconds and records all tuning and final source
directories.

## Machine-checkable audits

### Theory sentinels

Artifact:

`experiments/paper_suite_20260723/results/theory-sentinels-20260724-013613/summary.json`

Status: `PASS`.

- maximum symmetric/skew decomposition error:
  `5.684341886080802e-14`
- maximum corrected dominance-identity error:
  `4.85722573273506e-17`
- maximum box-QP objective error versus SciPy:
  `1.4675760606763788e-15`
- maximum pure-rotation normalized-margin error from `1/2`:
  `2.220446049250313e-16`
- maximum centered-softplus property violation:
  `3.469446951953614e-15`
- maximum performance-bridge violation: `0`

### Saved-checkpoint performance bridge

Artifact:

`experiments/paper_suite_20260723/results/saved-bridge-audit-20260724-013425/summary.json`

Status: `PASS`.

- tabular rows: `3780`; minimum slack `0.3695118667376809`;
  maximum violation `0`
- neural rows: `3900`; minimum slack `0.38594131744340465`;
  maximum violation `0`
- maximum recomputed Shapley residual:
  `8.446576771348191e-13`

## Bibliography

- Removed the unverified recovered entry
  `YuanTsangLau2026StochasticMinimax`.
- Verified the cited 2025 TSP step-size paper and added DOI
  `10.1109/TSP.2025.3592678`.
- Retained verified primary entries for the 2025 ICASSP minimax paper and the
  adaptive Polyak step-size paper.
- Final compilation has no undefined citation or reference.

## Code changes

- `experiments/paper_suite_20260723/markov_game_suite.py`
  - added exact unregularized Shapley solves and saved bridge metrics;
  - added method subsets and fixed learning-rate CLI options;
  - added Holm-adjusted decisions;
  - corrected the reported Shapley residual to evaluate the Bellman operator at
    the final returned value;
  - avoids unnecessary HVP/G computation in noG without changing its iterates.
- `theory_sentinels.py`: algebraic and numerical theory checks.
- `audit_saved_bridge.py`: independent saved-trajectory bridge and statistics
  audit.
- `tune_fixed_baselines.py`: disjoint-seed global learning-rate sensitivity.
- `experiments/paper_suite_20260723/README.md`: current reproduction map.

## Final build and visual QA

Build command:

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Final checks:

- 17 pages, US letter
- no undefined references or citations
- no LaTeX/package errors
- no overfull boxes
- no Type 3 fonts
- all 17 pages rendered at 130 dpi and visually inspected
- figures, tables, equations, algorithm, appendices, and references have no
  clipping or overlap

Stable PDF:

`output/pdf/RARL_final.pdf`

- size: `835180` bytes
- SHA-256:
  `B41D6E1E6F82460CA2909AF828448DE11C125C6721B943156ABB45E2D72A2C0D`

Selected source hashes:

- `main.tex`:
  `D396172AA371F27FE2E9DBF584D5BA5D028E56EA15E6384349E941282DE12E17`
- `refs.bib`:
  `F660F6DC429B1536CCE11F23D2E0803A637E71E00C61799BC94539718EB2C1AC`
- `markov_game_suite.py`:
  `F09E554EE4520ED8F5354E3701FDCEB986DD5909019BA82B962C01A6D582AC68`

## Remaining claim boundaries

These are limitations, not unfinished repairs:

1. The neural experiments use population gradients and exact finite-state
   value/BR solves.  They are two-neural-policy RL/Markov-game experiments but
   not trajectory-sampled actor-critic experiments.
2. Methods are matched by initialization and outer-update count, not by oracle
   cost.
3. The neural parameter-space convergence theorem is local and conditional; it
   does not certify a global Nash policy without additional landscape
   assumptions.
4. The curvature-improved rate requires the normalized-dominance margin and
   exact accepted cone step.  Skew magnitude alone is not sufficient.
5. A standard continuous-control Gymnasium/MuJoCo positive cell remains useful
   future work, but its absence is now reported rather than concealed.
