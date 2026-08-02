# IEEE TSP final-submission audit — 2026-07-29

## Decision

The current manuscript is ready for an initial IEEE Transactions on Signal
Processing submission from the theory, evidence, bibliography, and file-format
perspectives audited here.
No claim-blocking mathematical error, missing proof dependency, unsupported
experimental statement, unresolved citation, or PDF-format defect was found.

This decision is an internal technical gate, not a prediction of editorial
acceptance.

## Theory gate

The audit checked the policy-space performance bridge, the smooth
entropy-regularized performance component, the field/curvature template
expansions, the symmetric--antisymmetric field-energy identity, the
skew-dominance theorem, the composite Lyapunov construction, the closed-form
cone and box QP solutions, the reduced-gradient dominance identity, the
pathwise drift and residual bounds, the biased-oracle conditional drift, the
ergodic residual-floor theorem, the boxed curvature-rate theorem, and the
local contraction corollary.

The assumptions now close every derivative and admissibility requirement used
in the proofs.
In particular:

- all iterates and every directional-stencil point lie in the analyzed
  differentiability/trust region;
- the skew diagnostic explicitly requires Jacobian--vector and
  transposed-Jacobian--vector products;
- the spectral shift uses a fixed \(\lambda_{\rm pd}>0\), so the positive
  definiteness claim is explicit;
- the main theorem uses a conditional expected normalized box-gain margin and
  carries coefficient, comparator, margin, finite-difference, and safeguard
  residuals into its stated floor;
- the performance bridge is global in policy space, while the stochastic
  stationarity result is local in parameter space;
- the per-state Shannon entropy in the smoothness proof is distinguished from
  the discounted causal-entropy functional used in the performance bridge;
- the optional Lyapunov residual remains generic, without claiming an
  unformalized augmented critic field.
- the modeled stochastic field contains exactly the two policy blocks
  \((\theta,\psi)\); the introduction makes no claim of analyzed
  value-estimation or auxiliary-block dynamics;
- the Euclidean field-energy theorem is explicitly separated from the
  composite controller merit; skew dominance is not asserted to characterize
  composite descent, which uses its own directional coefficient and box gain;
- the composite quantity is introduced as a Lyapunov merit and becomes a local
  stochastic Lyapunov function under the stated dissipation condition.

No additional theorem is required to support the claims made in the initial
submission.

## Publication-writing gate

The manuscript follows the supervisor gate recorded in `AGENTS.md` and in the
personal `academic-research-suite` and `publishable-academic-writing` skills.
Assumptions are enumerated, the culminating stochastic curvature guarantee is
named as the main theorem, formal results have concise role explanations, and
transitions connect the geometry, drift model, convergence theory,
implementation, and experiments.

First-use expansions cover the manuscript acronyms, including RL, GDA, PPM,
EGM, GAN, ODE, QP, BR, PL, RPS, KKT, and DiCE.
The source contains no `\qquad`.

The abstract contains 199 words and no display equation.
The paper has exactly five keywords.

The final title is `Saddle-Field Policy Optimization for Zero-Sum Markov
Games: Adaptive Lyapunov-Drift Control with Finite-Time Stationarity
Guarantees`.
It names the policy-optimization setting, adaptive Lyapunov-drift mechanism,
and finite-time stationarity target without implying global Nash convergence.

The publication prose uses direct, active statements and positive scope
definitions.
It contains no failed-experiment narrative, pre-emptive rebuttal, generic
limitations section, or missing-comparison disclaimer.

Formal definitions and theorems use maximizing/minimizing player terminology.
The manuscript does not use protagonist/adversary terminology; the
introduction describes robust RL through maximizing task and minimizing
disturbance policies.

The publication-facing source contains no omitted neural-game cell and makes no
neural-population Holm-family claim.
It contains no screened-out trajectory result, wall-clock/oracle-cost
discussion, disclosure section, project chronology, or failed-experiment
narrative.

## Experimental gate

Section VI is organized by scientific question:

1. controlled normal-form geometry verifies the exact skew threshold and the
   predicted/realized QP identity;
2. three tabular Markov games test population-gap performance without function
   approximation;
3. four neural-policy population games test the same mechanism in a
   134-dimensional neural field;
4. finite-trajectory CyclicControl validates the stochastic field and
   curvature oracles on sampled transitions.

The first and fourth layers use the field-energy function, whereas the
tabular and neural population layers use the composite Lyapunov function.
All statistics in the paper are backed by the frozen project results.
The tabular cells show positive worst-case-return gains and lower mean
exploitability.
The four neural-population cells have positive paired intervals, and the
finite-trajectory cell has a worst-case-return gain of \(0.3422\), a paired 95% interval
\([0.1230,0.5614]\), and \(9/10\) wins.
The neural-population evidence is stated through paired gains, 95% intervals,
lower mean exploitability, and seed-win counts.  The Holm correction retained
in the paper applies only to the explicitly listed three-game tabular family;
the finite-trajectory cell is a separate prespecified primary comparison.
The manuscript figures show only the selected claim-bearing cells and retain
the full mechanism figure as three equal-sized vector panels.

## Citation and IEEE bibliography gate

`main.tex` cites exactly 30 keys and `refs.bib` contains exactly those 30
entries.
There are no missing keys, unused entries, duplicate keys, duplicate titles,
duplicate identifiers, empty required fields, or definite IEEE BibTeX style
errors.

The complete authoritative metadata evidence is stored in
`output/audit/citation_integrity_20260729/`.
After the wording-only 2026-07-30 revision, the unchanged 30-entry
bibliography was refreshed under
`output/audit/citation_integrity_20260730/` with a new citation passport,
IEEE style audit, and live DOI audit:

- 11 DOI-bearing entries passed live Crossref field-level comparison;
- 19 non-DOI entries were refreshed against authoritative publisher,
  proceedings, journal, arXiv, JSTOR, or bibliographic records;
- the DiCE proceedings entry was refreshed again on 2026-07-30 against the
  official PMLR record and arXiv:1802.05098 after protecting its canonical
  capitalization in BibTeX;
- the combined gate resolves all 30 entries with no material title, author,
  year, venue, volume/issue, page, or identifier conflict;
- the IEEE bibliography audit status is `pass`.

After the 2026-07-31 language refinement, the same unchanged bibliography was
refreshed again under `output/audit/citation_integrity_20260731/`.
The IEEE audit again reports 30 cited keys, 30 entries, and no missing, unused,
duplicate, or definitely malformed entry; all 11 DOI records again pass live
Crossref metadata comparison.

The 2026-08-01 release refresh is stored under
`output/audit/citation_integrity_20260801/`.
The current 30-entry bibliography retains the previously verified 19
authoritative non-DOI records; its 11 DOI records pass a new live Crossref
metadata comparison, and the IEEE style audit remains `pass`.

After the scope, rate, and quantitative-evidence revision, the unchanged
bibliography was refreshed once more under
`output/audit/citation_integrity_20260801_revised/`.  All 30 records resolve,
the 11 DOI records pass a fresh live Crossref comparison, and the IEEE audit
reports no missing, unused, duplicate, or malformed entry.

After the final prose and notation integration, the unchanged bibliography was
refreshed under `output/audit/citation_integrity_20260801_final_prose/`.
All 30 entries resolve: 11 DOI records pass a fresh live Crossref comparison,
19 entries pass the preserved authoritative-record gate, and the IEEE style
audit reports no missing, unused, duplicate, malformed, or judgment-call item.

After the Cesaro, performance-bridge, and clarity revision, the unchanged
bibliography was refreshed under
`output/audit/citation_integrity_20260801_clarity_bridge/`.
All 30 entries resolve: 11 DOI records pass a fresh live Crossref comparison,
19 entries pass the authoritative-record gate, and the IEEE style audit again
reports no missing, unused, duplicate, malformed, or judgment-call item.

The academic-suite multi-index helper was also invoked with a no-cache request,
but its configured endpoints timed out before emitting a record set.
The final gate therefore uses the complete live DOI and authoritative-record
paths above rather than a partial multi-index output.

## PDF and release gate

- IEEEtran, 10-point, US Letter, double column;
- 13 pages including Appendix A, Appendix B, and all references;
- the experiments section begins on page 9, after the theory and
  implementation development;
- zero undefined citations or references;
- zero overfull boxes and zero LaTeX errors;
- three visually benign underfull notices;
- every font embedded and zero Type-3 fonts;
- all 13 pages rendered and visually inspected with no clipping, overlap, or
  layout regression;
- vector motivation and experiment figures have no clipping or overlap.

Canonical release artifacts after the 2026-08-01 clarity revision:

- `main.tex` SHA-256:
  `98F5AFE9E302C548440A7C3BD4BD2612EABBE581305FA498E295DB4FF6C0D282`;
- `refs.bib` SHA-256:
  `75E29A4DB355A4D79D94CEC59620D55C0E770FE402FD772A9F77FFAB06BC4A66`;
- `RARL_final.pdf` SHA-256:
  `E50F9239BDD5D790F8DB0F706CC086E939762FFEA38D8C99E528F0818BF62112`.

`main.pdf` and `RARL_final.pdf` are byte-identical.

## Final Introduction and release refresh (2026-08-01)

The final Introduction defines every problem-specific symbol before use. It
defines the saddle field and Jacobian before the ODE and game-decomposition
discussion, explains that field energy is zero exactly on the first-order
stationary set, and then introduces the curvature direction. The exact
GDA/PPM/EGM updates appear once, before the design questions. The scalar step
size `s`, the extrapolation step size `gamma_t`, and the final step size
`eta_t` are all introduced before the corresponding step-size pairs.

The contribution statement now consists of three concise bullets followed by
a short paper-organization paragraph. PPM is treated as an implicit proximal
method, while EGM, Mirror-Prox, and policy extragradient are described as
extragradient-type methods. The abstract contains 199 words, the keyword list
contains five terms, and the manuscript cites all and only the 30 entries in
`refs.bib`.

The current release passes the final gates:

- 13 IEEEtran double-column pages, including appendices and references;
- no LaTeX errors, warnings, undefined references/citations, overfull boxes,
  or underfull boxes;
- no full-form narrative `Equation` references;
- no `PursuitEvasion` or defensive cost-comparison prose;
- final visual checks of the title/abstract, Introduction, mechanism figure,
  appendices, and references show no clipping, overlap, or layout regression;
- citation integrity and IEEE style status `pass` in
  `output/audit/citation_integrity_20260801_intro_restructure/` (11 live DOI
  matches, 19 authoritative non-DOI records, zero unresolved entries).

Canonical hashes:

- `main.tex`: `34A243F231B13342C2E836AE16802BDBB748ECA123F3205215EC5FBCFEB9CF52`;
- `refs.bib`: `75E29A4DB355A4D79D94CEC59620D55C0E770FE402FD772A9F77FFAB06BC4A66`;
- `main.pdf` and `RARL_final.pdf`:
  `6B27EF5F81A1D346CF64D7129069B4B8B9964855C82A214D7DDBF204E4B12B84`.

## Superseded introduction-flow release gate (2026-08-02)

This gate describes a rejected intermediate revision and is retained only for
provenance. The rollback gate below defines the canonical submission files.

The Introduction now defines the saddle field before field energy and derives
the curvature direction from the local PPM/EGM expansion before posing the
adaptive step-size problem. This order removes the previous abrupt appearance
of field energy and the curvature direction. All problem-specific symbols are
defined before use, and the Introduction contains no technical-section forward
reference to Proposition 3.

The complete release passes the following checks:

- 13 IEEEtran double-column pages, including appendices and 30 references;
- 200-word abstract and exactly five index terms;
- no LaTeX errors or warnings, undefined citations/references, or overfull
  boxes; two visually benign float-related underfull-vbox notices on page 11;
- Figs. 2--4 occur before the Conclusion and in the order used by the
  experiment narrative;
- all matrices appear on separate displayed lines;
- no narrative `so`, full-form `Equation` reference, `PursuitEvasion`, or
  defensive cost-comparison language;
- all fonts embedded and no Type-3 fonts;
- fresh citation-integrity and IEEE-style status `pass` in
  `output/audit/citation_integrity_20260802_flow_revision/` (11 live DOI
  matches, 19 authoritative records, zero unresolved entries).

Canonical hashes:

- `main.tex`:
  `CFB8544BC339D8E3CA5EFE4A54490C30B10349498A3D58A49AC9C3EEB976E41C`;
- `refs.bib`:
  `75E29A4DB355A4D79D94CEC59620D55C0E770FE402FD772A9F77FFAB06BC4A66`;
- `main.pdf` and `RARL_final.pdf`:
  `254223A75CE8D8CCAC7AE1C5D233561500DBEA2D64E393C665DA22F6FBBE5A55`.

## Canonical rollback release gate (2026-08-02)

The manuscript restores the 2026-08-01 Introduction structure and all other
pre-flow-revision prose, captions, figure sizes, and float placement. The only
retained edits remove informal narrative wording and place matrices on
separate displayed lines.

- 13 IEEEtran double-column pages, including appendices and 30 references;
- no LaTeX errors or warnings, undefined citations/references, underfull boxes,
  or overfull boxes;
- all inspected matrices render as separate displays without clipping;
- no narrative `so`, full-form `Equation` reference, `PursuitEvasion`, or
  defensive cost-comparison language;
- citation-integrity and IEEE-style status `pass` in
  `output/audit/citation_integrity_20260802_rollback/`.

Canonical hashes:

- `main.tex`:
  `4F5977140F353DDACE3CFE331326EECB63CBAB1D906EFD7E1EBFFE1538BDC116`;
- `refs.bib`:
  `75E29A4DB355A4D79D94CEC59620D55C0E770FE402FD772A9F77FFAB06BC4A66`;
- `main.pdf` and `RARL_final.pdf`:
  `73DA723EE2D6E98550BC67A97EAEB2D232FA1E0C844AF6FB03D4EC9A708B50E7`.
