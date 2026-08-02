# Stochastic-oracle validation handoff — 2026-07-27

## Outcome

Gap 1 is closed by an additive finite-trajectory neural-policy experiment.
The existing population VI-A/VI-B/VI-C code, seeds, CSV/JSON files, and figures
were not modified or rerun.

The current 14-page manuscript reports the stochastic validation inside VI-C.
The final PDF contains all proofs, environment definitions, and 32 references
in one IEEE two-column file.

## Why the first prototype was rejected

The initial `stochastic_actor_critic.py` finite-differenced a score-gradient
field at perturbed policies while holding behavior trajectories fixed.  This
omitted the change in the policy-induced trajectory distribution.  Its
curvature direction remained systematically inconsistent as the batch grew,
so neither its smoke results nor its figures are paper evidence.

The accepted implementation is `stochastic_dice_policy.py`.  It forms a
finite-horizon, per-decision likelihood-ratio return.  Explicit two-player
prefix ratios make nearby QP stencil evaluations valid off the behavior point;
at the behavior parameters, repeated differentiation is the DiCE estimator.
It uses

`F_hat = diag(-I,I) grad J_hat_H`

and

`G_hat = diag(-I,I) Hessian(J_hat_H) F_hat`.

The same trajectory batch is reused for the QP coefficient stencil and
safeguard.  The same-batch Hessian–field product has a finite-batch covariance
bias, and the horizon introduces truncation bias; both are within the
manuscript’s biased-oracle assumption.  Training never calls a best-response
solver.

## Frozen protocol

- Game: CyclicControl.
- Two separate 4–8–3 tanh–softmax policies; 134 joint actor parameters.
- Simultaneous updates; no policy warm-up.
- Methods: QP+G, noG, EGM.
- Learning rate / QP caps: 0.03.
- Horizon: 16.
- Transition batches: 128, 512, 2048.
- Screening: seeds 3000–3004, 40 updates.
- Formal: untouched seeds 4100–4109, 60 updates.
- Exact hard/soft best responses and the infinite-horizon population field are
  checkpoint metrics only.
- Prespecified primary comparison: final hard-BR return, QP+G minus noG, batch
  2048.

The initially launched 3100-series formal run was stopped because seed 3100
had appeared in an oracle-consistency sentinel.  Its partial files remain under
`results/formal-CyclicControl-dice-20260727-124020/` with `ABORTED.md`; they
were not analyzed or reported.

## Numerical sentinels

Accepted sentinel:

`experiments/stochastic_oracle_validation_20260727/results/sentinels-dice-20260727-121445/summary.json`

- Behavior-point likelihood-ratio error: 0.
- Autograd versus central finite-difference HVP relative error:
  `6.15e-10`.
- Mean stochastic/population direction cosine from 64 independent
  2048-transition batches: `0.9874` for `F`, `0.9828` for `G`.
- Accepted same-batch merit decrease: `0.05608`.
- Hard/soft Bellman residuals: `8.93e-13` / `8.45e-11`.

## Screening and formal results

Screening directory:

`experiments/stochastic_oracle_validation_20260727/results/screen-CyclicControl-dice-20260727-123840/`

At batch 2048, QP+G beat noG in 5/5 screening seeds, with mean hard-BR gain
0.3519, lower mean exploitability, and 100% positive-`d` / active-`gamma`
rates.  This gate triggered the disjoint formal run.

Formal directory:

`experiments/stochastic_oracle_validation_20260727/results/formal-CyclicControl-dice-20260727-124224/`

The run took 2740.3 seconds and contains 630 complete checkpoint rows and 5400
complete update-diagnostic rows.

At the prespecified 2048-transition cell:

- QP+G minus noG hard-BR gain: `0.3422`.
- Paired 95% Student-t interval: `[0.1230, 0.5614]`.
- Seed wins: `9/10`.
- One-sided exact sign-test p-value: `0.010742`.
- Mean hard-BR return: QP+G `-0.2034`, noG `-0.5456`, EGM `-0.3779`.
- Mean hard exploitability: QP+G `0.4287`, noG `0.9318`, EGM `0.7844`.
- Mean population field norm: QP+G `0.1694`, noG `0.3769`, EGM `0.3102`.
- Positive-`d` and active-`gamma` rates: both `1.0`.

QP+G has a higher mean hard-BR return than EGM at batch 2048, but the paired
QP+G-minus-EGM interval `[-0.0207, 0.3698]` crosses zero and its exact sign-test
p-value is `0.05469`; the paper makes no significance claim against EGM.

Batch 128 is a negative variance-boundary cell.  At batch 512, the QP+G–noG
mean is positive but the paired interval crosses zero.  These outcomes remain
in the reproducibility record.  The submission-facing manuscript reports only
the prespecified positive 2048-transition cell and conditions its claim on
that batch scale; it does not claim a uniform advantage over batch size.

Formal integrity audit:

`experiments/stochastic_oracle_validation_20260727/results/formal-CyclicControl-dice-20260727-124224/formal_audit.json`

- Complete Cartesian keys: passed.
- Finite checkpoint metrics: passed.
- Exact paired initial conditions: passed.
- Maximum hard/soft Bellman residual: `9.00e-13` / `9.00e-11`.
- Maximum behavior-point likelihood-ratio error: `1.33e-15`.
- `curves.csv` SHA-256:
  `5f4d62768f7e7a0867df14abff1859c8117c11cc858ff514492209edc195e3a8`.
- `diagnostics.csv` SHA-256:
  `e5d47526b1942103c9df8818d756e07d845f47e18dce02e3c3a77d717367d861`.

## Manuscript and bibliography changes

- Added a concise finite-trajectory validation paragraph to VI-C and retained
  only the statistically confirmed 2048-transition cell in the
  submission-facing narrative.
- Updated the abstract, contribution summary, and conclusion to distinguish
  population and finite-trajectory evidence.
- Added the DiCE reference as `Foerster2018DiCE`.
- Removed the redundant neural final-value table; its confirmed values remain
  in the VI-C text and main neural figure.
- Kept the stochastic hard-BR curve as
  `output/pdf/fig_vi_c_stochastic.pdf` and `.png`, but did not embed it in the
  14-page submission because doing so created a mostly empty fifteenth
  reference page.  No theorem, proof, environment definition, main experiment
  figure, or reference was compressed or removed for this choice.

The publishable-writing policy shaped this edit: the paper states the strongest
supported primary comparison at its specified batch scale, while the complete
batch-size sensitivity remains available in this handoff and the formal
artifacts.

## Final PDF QA

Current identical files:

- `RARL_final.pdf`
- `output/pdf/RARL_final.pdf`
- `output/pdf/RARL_TSP_14page_stochastic_oracle.pdf`

SHA-256:

`D53AB202853883FD34D621C35857F5737E3F391C73EE035EE923447E0E0D6DE1`

The 2026-07-27 motivation update also replaces the legacy blurry bilinear
trajectory raster by an exact vector panel and combines it with the two
existing mechanism panels.  The three axes are equal-size squares.

QA:

- 14 US-letter pages.
- 32 bibliography items.
- No undefined citations/references.
- No overfull boxes.
- No Type 3 fonts.
- All 14 pages rendered and visually inspected; no clipping, overlap, blank
  page, broken float, or unreadable table/reference was found.

`output/pdf/RARL_TSP_14page.pdf` remained open in a local PDF viewer and Windows
rejected the overwrite.  It is the preceding build with SHA-256
`8A91D518A3151D3FC4871CA00D8A20CF292DCC1FCB164B07B0AECF74CA14FCF1`;
use one of the three current files above.

## Deferred evidence gap

Oracle-cost / wall-clock matched comparison remains intentionally deferred.
The present experiment records transition usage and formal elapsed time but
does not claim wall-clock superiority.
