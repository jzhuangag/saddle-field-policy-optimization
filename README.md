# RARL / Zero-Sum Markov-Game Codex Project

This is the canonical project recovered from the Claude conversation archives,
the RARL project memory, and the local CMRAC materials.  Recovery records are
retained for provenance; superseded manuscript-source and manuscript-PDF
copies have been removed.

## Canonical entry points

- `main.tex`: current IEEE TSP manuscript source.
- `refs.bib`: checked bibliography used by the manuscript.
- `RARL_final.pdf`: the single current 13-page IEEE TSP manuscript PDF,
  including all proofs, environment
  definitions, 30 references, the finite-trajectory validation, and the
  three-panel vector motivation figure.
- `TSP_FINAL_SUBMISSION_AUDIT_20260729.md`: current theorem, reporting,
  bibliography, and release-quality gate.
- `output/audit/citation_integrity_20260802_rollback/`: current citation passport, live
  DOI metadata check, authoritative-record evidence, and IEEE bibliography
  audit.
- `FINAL_IEEE_ARS_AUDIT_20260727.md`: preceding IEEE bibliography and
  academic-integrity audit, retained for provenance.
- `SUBMISSION_METADATA_TO_CONFIRM_20260727.md`: proposed CRediT roles and
  release-policy wording that require author confirmation before insertion.
- `STOCHASTIC_ORACLE_VALIDATION_HANDOFF_20260727.md`: stochastic-oracle design,
  formal statistics, manuscript changes, and QA.
- `TSP_14PAGE_REVIEW_HANDOFF_20260726.md`: current page-limit, theory-audit,
  experiment-scope, and build record.
- `TSP_REVIEWER_INSIGHT_AUDIT_20260727.md`: adjudication of the two pasted
  reviewer opinions, the resulting theorem/implementation revisions, official
  TSP-scope check, and remaining evidence gaps.
- `MAJOR_REVISION_HANDOFF_20260724.md`: historical theory, experiment, and
  verification record preceding the combined 13-page build.
- `experiments/paper_suite_20260723/`: final three-level experiment suite,
  theory sentinels, performance-bridge audit, and independent baseline tuning.
- Recovery conversations, external source PDFs, superseded drafts, and local
  build artifacts remain outside version control.

## Current paper scope

The manuscript is titled:

> Saddle-Field Policy Optimization for Zero-Sum Markov Games: Adaptive
> Lyapunov-Drift Control with Finite-Time Stationarity Guarantees

It studies simultaneous maximizing/minimizing-player saddle fields; robust RL
is one specialization.  The curvature direction remains
\(G=\nabla F\,F\).  A local quadratic model of a composite Lyapunov drift
selects independent nonnegative coefficients for \(-F\) and \(+G\).

## Experiment organization

1. VI-A: an exactly solvable normal-form saddle field for the
   symmetric/skew decomposition, rotation threshold, and QP drift identity.
2. VI-B: three exact-gap tabular zero-sum Markov games.
3. VI-C: four stateful Markov games in which the two players use separate
   neural networks.
4. VI-D: finite-trajectory CyclicControl validation on untouched seeds.

All manuscript figures are generated from saved CSV/JSON results.  Synthetic
recovery placeholders are not used.  Hard best-response return, hard
exploitability, entropy-regularized gap, and field norm are evaluated
separately.  An independent Shapley-value solve audits the performance bridge
on every saved checkpoint.  The analytical motivation figure is generated
from closed-form linear-field updates by
`experiments/paper_suite_20260723/motivation_geometry.py`; its three axes are
equal-size squares and are not empirical results.

## Important theory boundaries

- Skew dominance exactly characterizes when \(+G\) is a field-energy descent
  direction.  It is not, by itself, sufficient for a performance gain under an
  arbitrary composite Lyapunov function.
- The curvature-rate result covers the implemented box-QP, approximate
  coefficients, solver/safeguard errors, and an expected normalized box-gain
  margin with an explicit residual floor.  A cap-valid pathwise comparator
  certificate supplies a zero-residual sufficient condition; a sharper
  Schur-complement formula describes the uncapped reduced curvature
  coordinate.
- Neural/continuous parameter-space guarantees are local first-order
  stationarity results; they do not claim automatic recovery of a global Nash
  policy.
- Local geometric contraction is invoked only when the composite merit is
  zero-compatible with the target and satisfies the stated composite
  Polyak--Lojasiewicz condition.

## Reproducibility

See `experiments/paper_suite_20260723/README.md` and
`experiments/stochastic_oracle_validation_20260727/README.md`.  Formal runs use
independent, timestamped result directories and retain seeds, learning rates,
Bellman/Shapley residuals, raw curves, and statistical decisions.  Screening
seeds must not be reported as final confirmation seeds, and existing artifacts
must not be overwritten.
