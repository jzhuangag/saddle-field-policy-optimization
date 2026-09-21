## Material Passport

- Origin skill: experiment-agent
- Origin mode: run + validate
- Verification status: ANALYZED
- Dataset: four-environment finite-trajectory attack-strength sweep
- Protocol: 4 environments x 7 attack strengths x 5 paired seeds x 2 methods
- Training oracle: finite-trajectory DiCE, 512 trajectories/update, horizon 16
- Training Lyapunov weight: lambda_P = 0
- Evaluation: exact population metrics at checkpoints only

## Integrity checks

- Complete curve grid: 1,960/1,960 rows.
- Complete diagnostic grid: 16,800/16,800 rows.
- Nonfinite values: none.
- Maximum paired initial-metric difference: 0.
- Maximum hard-best-response residual: 8.9995e-13.
- Maximum soft-best-response residual: 8.9991e-11.
- Maximum game-value residual: 8.9817e-13.
- All five formal seeds (6100--6104) are retained in every environment-strength cell.
- The 512-trajectory batch was selected using a disjoint pilot and was not retuned on the formal seeds.

## Statistical findings

- QP+G has a positive paired mean worst-case-return gain in 28/28 environment-strength cells.
- QP+G has a positive paired mean exploitability reduction in 28/28 cells.
- With five seeds, the paired 95% t interval excludes zero in 4/28 return cells and 9/28 exploitability cells.
- Therefore, the supported descriptive claim is that QP+G improves the mean of both metrics in every evaluated cell. The data do not support claiming a statistically significant improvement in every cell.

## Eleven-item fallacy scan

1. Simpson's paradox: checked; no aggregate/subgroup sign reversal because all 28 cell means have the same favorable direction.
2. Ecological fallacy: not applicable; claims concern algorithm-level runs, matching the unit of analysis.
3. Berkson's paradox: not applicable; no outcome-based filtering of seeds or cells.
4. Collider bias: not applicable; no covariate adjustment is used.
5. Base-rate neglect: not applicable; no diagnostic-classification probabilities are reported.
6. Regression to the mean: checked; paired fixed seeds are used and runs were not selected for extreme initial outcomes.
7. Survivorship bias: checked; all formal seeds and all planned cells are included.
8. Look-elsewhere effect: checked; all 28 planned cells are reported. Because many intervals are examined without multiplicity correction, they are treated descriptively rather than as a family of significance tests.
9. Garden of forking paths: caution; no formal preregistration artifact is available, but batch selection used disjoint pilot seeds and the formal batch, seed grid, lambda_P, and metrics were frozen before the four-environment run.
10. Correlation versus causation: not applicable; this is a controlled algorithmic intervention in simulation.
11. Reverse causality: not applicable; the comparison is experimentally assigned by the update rule.

Coverage: 11/11 checked.

## Reproducibility status

The stochastic training runs completed successfully and the deterministic merge was structurally validated. This report is ANALYZED rather than VERIFIED because an independent repeat of the full stochastic sweep has not been performed.
