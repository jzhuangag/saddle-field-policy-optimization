# TSP reviewer-insight audit

Date: 2026-07-27

This note audits the two pasted reviewer-style assessments against the
manuscript, implementation, saved results, and current IEEE Signal Processing
Society guidance.  It records changes made to `main.tex` and separates
mathematical corrections from reviewer preferences and unexecuted new
experiments.

## Overall judgment

The second assessment is substantially right about several theory--implementation
closure issues, but its numerical recommendation ("Borderline Reject") is a
subjective review score rather than a mathematical fact.  The core
symmetric--antisymmetric identity, two-coordinate QP, drift decomposition, and
performance bridge remain valid.  The paper needed a major revision, not a
restart.

The claim that a realistic communications application is required for TSP is
not supported by the journal's stated scope.  TSP covers novel theory,
algorithms, performance analysis, and learning from signals.  The SPS Unified
EDICS explicitly includes machine learning, reinforcement learning, game
theory for signal processing, and optimization methods for signal processing;
the MLSP Technical Committee also identifies online/adaptive nonlinear signal
processing and data-driven learning as central topics.  The paper is therefore
positioned as adaptive online learning/optimization for stochastic
two-player policy fields.  It does not add a token communications experiment
or relabel an unrelated toy problem as a communications contribution.

Official sources checked:

- https://signalprocessingsociety.org/publications-resources/ieee-transactions-signal-processing
- https://signalprocessingsociety.org/publications-resources/information-authors
- https://signalprocessingsociety.org/publications-resources/unified-edics
- https://signalprocessingsociety.org/community-involvement/machine-learning-signal-processing

## Correct criticisms incorporated into the manuscript

### Positioning and claims

- The title is now `Lyapunov-Drift-Based Adaptive Field--Curvature Policy
  Optimization for Two-Player Zero-Sum Markov Games`.
- The abstract was reduced from approximately 287 words to 214 words, within
  the official 150--250-word range.
- The contribution is the online coefficient controller over the span of
  `-F` and `+G`, not the invention of `G = nabla F F`.
- The introduction now distinguishes `J F`, `J^T F`, and
  symmetric/Hamiltonian corrections, and describes the paper's contribution
  relative to proximal, extragradient, and game-geometry methods.
- Performance claims are explicitly a policy-space interpretation and
  robust-return deficiency bound.  Local parameter-space stationarity is not
  upgraded to global neural-policy Nash convergence.

### Smoothness and second-order expansions

- The finite-game smoothness statement now gives the soft Bellman operator
  explicitly and requires locally `C^3` policy maps with probabilities bounded
  away from zero.
- The inaccurate infinite-dimensional Bellman wording was removed.
- The auxiliary fitted block is described as stationary for the fitting loss,
  rather than at an unrestricted optimum.
- The bilinear example now uses a sign convention consistent with the
  max--min field.
- PPM and EGM remainders now state their dependence on local bounds for
  `F`, `nabla F`, and `nabla^2 F`, and the PPM statement is local.

### Geometry

- The normal-plane condition is stated using an orthonormal invariant plane.
- The former overgeneralization that a "potential-like" field makes `F` an
  eigenvector was removed.
- The pure-rotation boundary solution is called a feasible unconstrained
  critical point, not an interior point.
- The single-column motivation figure remains a complete two-panel analytic
  mechanism figure; its merit, parameters, initial point, and absence of caps
  are stated in the caption.

### Stochastic assumptions and safeguards

- The local third-derivative envelope is now uniform over all feasible trial
  pairs and current-oracle realizations, with the filtration/measurability
  condition stated.
- Positive-definite inflation is an explicit implementable rule:
  `chat = max(c, lambda_pd)` and
  `ahat = max(a, b^2/chat + lambda_pd)`.
- The text no longer claims that backtracking can verify an unknown local
  invariance assumption.
- The local dissipation condition is named as such; it is not presented as an
  automatic global composite PL property.

### Curvature-rate theorem

- The main curvature theorem now uses the implemented coefficient boxes,
  approximate coefficients, step-solver residual, safeguard residual, and
  accepted-step residual.
- The gain is defined as the box-QP predicted decrease minus the box-constrained
  pure-field decrease.
- A cap-valid sufficient condition is proved by evaluating the box-QP at the
  pure-field box minimizer plus a feasible curvature coordinate.  It covers
  active field or curvature caps and yields an explicit positive constant
  from a normalized reduced-gradient lower bound.
- The uncapped problem additionally has the sharper Schur-complement reduced
  coordinate
  `r_G = d_hat + (b_hat/c_hat)e_hat` and
  `h_G = a_hat - b_hat^2/c_hat`.
- The theorem retains the normalized box-gain margin as an explicit condition.
  It does not claim that skew dominance alone implies performance improvement
  for every composite Lyapunov merit.
- Coefficient approximation errors are propagated into the rate residual.

### Implementation and experiment disclosure

- Exact automatic differentiation is stated to involve third derivatives of
  the underlying game objective when the merit contains field energy.
- The directional-secant implementation now gives the full nine-point central
  stencil, including the mixed coefficient.
- The manuscript reports the probe rule, positive-definite floor, nine merit
  evaluations, and that both soft best responses are recomputed at every
  stencil point.
- The first occurrence of hard-BR return is formally defined as unregularized
  robust best-response return.
- The outer-update comparison is explicitly not an oracle-cost-matched
  comparison.
- The prespecified neural family size, Holm family, activation, contribution,
  and pooled rotation statistics are defined.
- The redundant constant feature in RoutingInterdiction and SecurityPatrol is
  disclosed rather than silently changed, because changing it would invalidate
  the frozen results without a rerun.
- Appendix proof references no longer render as awkward `A-A`/`A-0` links.
- The conclusion was compressed.

## Partly correct comments not treated as automatic manuscript changes

- Additional OGDA, consensus, Hamiltonian, and symplectic baselines would
  broaden the empirical comparison, but they are not mathematical corrections
  and are not mandatory as a set.  No unrun baseline is mentioned as evidence.
- A full QP-component factorial ablation could be useful, but the matched noG
  ablation already isolates the paper's claimed extra coordinate.  Adding
  further components requires a prespecified rerun.
- Per-environment tuning and wall-clock/oracle-budget curves are stronger
  protocols than the current global-rate sensitivity check.  The manuscript
  now states the exact comparison scope instead of implying cost efficiency.
- An algorithm float is a presentation choice.  The current compact
  implementation statement, closed-form active-set rule, stencil formulas,
  safeguards, and update equations specify the algorithm without a second
  redundant float.
- The neural table remains compact because paired uncertainty intervals and
  exact tests are reported immediately in the text.
- Thirty-one directly used references satisfy the requested minimum; reference
  count alone is not a quality criterion.

## Genuine remaining evidence gaps

These items require new experiments and were not fabricated or inferred:

1. The stochastic-oracle theory is still validated mainly by population
   value-system experiments.  A trajectory-sampled actor--critic or
   finite-rollout experiment would test oracle bias, variance, and residual
   floors directly.
2. The current figures match outer-update count, not wall-clock time, field
   calls, best-response solves, or equivalent oracle budget.
3. The controller uses a global regularized policy-space gap in the reported
   finite games.  A scalable local-surrogate ablation would be needed before
   claiming large-scale model-free practicality.
4. The four confirmed neural cells are selected under a five-cell
   prespecified family, but the rejected cell is intentionally absent from the
   paper under the user's page/scope decision.  The family size and Holm
   correction nevertheless include it.

The manuscript therefore claims mechanism identification, local conditional
theory, and population-game evidence.  It does not claim scalable
trajectory-based superiority or oracle-cost efficiency.

## Page-limit finding

The official SPS author instructions impose 13 double-column pages for an
initial Regular Paper or a resubmission treated as new, including appendices,
proofs, and references.  A revised manuscript may use up to 16 pages.  The
current combined manuscript is 14 pages because the project instruction is to
keep proofs, environment definitions, and references in the same file.

Consequently:

- the current 14-page file is suitable as the complete master or a formal
  revision;
- it is not formally compliant as an initial TSP submission;
- if the next action is an initial submission, one additional page must be
  removed without splitting the paper or shrinking below the IEEE template.

## Verified deliverable

- Source: `main.tex`
- PDF: `output/pdf/RARL_TSP_14page.pdf`
- Stable copy: `output/pdf/RARL_final.pdf`
- Pages: 14
- Abstract: 214 words
- References: 31
- SHA-256:
  `8A91D518A3151D3FC4871CA00D8A20CF292DCC1FCB164B07B0AECF74CA14FCF1`
- Build: no undefined citations/references, LaTeX/package warnings, overfull
  boxes, or Type-3 fonts.
- Visual QA: all 14 pages rendered; the two-panel motivation figure, three
  experiment figures, single-column table, proofs, environments, and
  references are readable and unclipped.
- New experiments in this audit: none.
