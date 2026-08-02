# IEEE TSP combined 14-page handoff

Date: 2026-07-26

## Canonical deliverable

- Source: `main.tex`
- Submission PDF: `output/pdf/RARL_TSP_14page.pdf`
- Equivalent stable copy: `output/pdf/RARL_final.pdf`
- Rendered QA pages: `tmp/pdfs/combined-14page-final/`
- Reviewer-audit QA pages:
  `tmp/pdfs/reviewer-insight-audit-20260726/`
- Reviewer-opinion adjudication:
  `TSP_REVIEWER_INSIGHT_AUDIT_20260727.md`

The submission PDF is one IEEE two-column document of exactly 14 pages.
It contains the abstract, main theory, experiments, conclusion, all retained
proofs, finite-game environment definitions, and references.  There is no
separate supplement entry point or submission artifact.

The earlier split-build and 13-page PDFs were moved reversibly to
`historical/superseded-split-build/` and
`historical/superseded-13page-build/`; they are not submission files.

## Layout changes

1. The motivation figure is a readable single-column, two-panel mechanism
   figure.  The first panel isolates the pure-rotation fact: `-F` is tangent
   to field-energy level sets while `+G` points inward.  The second panel
   displays the exact two-coordinate drift QP and its strict interior optimum
   relative to the coordinate rays and fixed `(s,s^2)` coupling.
2. The neural results table is single-column and reports the exact QP+G/noG
   hard-BR-return/exploitability pairs needed for the confirmation claim.
   Complete baseline trajectories remain in the adjacent figure.
3. Proofs are grouped under one appendix section with paragraph-level proof
   headings.  Algebraically elementary derivations were compressed, but the
   proof-critical identities, inequalities, conditional expectations,
   telescoping argument, KKT argument, and zero-set conditions remain.
4. The algorithm float was replaced by a concise implementation paragraph
   because the update, QP, exact boundary solver, and backtracking rule are
   already formally specified.
5. Repeated narrative and defensive qualification remain removed.  The
   bibliography has 31 directly used references spanning robust RL/MDPs,
   zero-sum Markov games, saddle algorithms and game geometry, adaptive
   scaling, policy-gradient/actor--critic estimation, the PL condition, and
   the RPS multi-agent benchmark.
6. Only one numerical table is retained: the single-column QP+G/noG neural
   confirmation table.  Complete baseline trajectories remain in figures.

## Mathematical scope retained

- The performance bridge remains a global finite-game policy-space statement.
- Neural convergence remains local, conditional, and parameter-space.
- Skew dominance characterizes field-energy descent of `+G`, not unconditional
  performance gain.
- Strict QP advantage is tied to the curvature reduced gradient.
- The curvature-improved rate now covers the implemented box caps,
  approximate coefficient errors, and accepted-step/safeguard residuals.  It
  requires the stated normalized box-gain margin, for which the manuscript
  gives a cap-valid comparator certificate and a sharper uncapped
  Schur-complement diagnostic.
- Local contraction requires a zero-compatible composite merit.
- Approximate secant coefficients enter through the explicit coefficient
  residual and are not silently identified with the exact-AD theorem.

## Verification

- Page count: 14.
- SHA-256:
  `8A91D518A3151D3FC4871CA00D8A20CF292DCC1FCB164B07B0AECF74CA14FCF1`.
- `main.log`: no undefined references/citations, multiply-defined labels,
  overfull boxes, LaTeX warnings, package warnings, or PDF destination warnings.
- Fonts: embedded Type 1/TrueType; no Type 3 fonts.
- All 14 pages were rendered at 110 dpi and visually inspected.  The
  two-panel single-column motivation figure and neural table are readable; no text,
  equation, figure, table, appendix, or reference is clipped or overlapping.
- PursuitEvasion remains absent from the manuscript.
- No new experiment or synthetic result was introduced in the reviewer-audit
  pass.

## Submission-stage warning

The official SPS instructions limit an initial Regular Paper (and a rejected
paper resubmitted as new) to 13 double-column pages, including appendices,
proofs, and references; formal revisions may use up to 16.  This 14-page
combined file is therefore a complete master/revision artifact.  If it is to
be uploaded as an initial TSP submission, it still requires a one-page
compression pass without splitting the manuscript.

## Experiment audit retained

The publication-scope bridge audit remains:

- neural: 3120 saved rows, minimum slack `0.38594131744340465`, zero violations;
- tabular: 3780 saved rows, minimum slack `0.3695118667376809`, zero violations.

Artifact:
`experiments/paper_suite_20260723/results/saved-bridge-audit-20260726-093652/summary.json`.
