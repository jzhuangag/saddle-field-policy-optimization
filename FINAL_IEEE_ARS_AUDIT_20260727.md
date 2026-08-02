# IEEE bibliography and ARS audit — 2026-07-27

## Overall decision

The current manuscript is **scientifically auditable and build-clean after
revision**, but it is not an unconditional fresh-submission pass:

| Layer | Status | Evidence or remaining action |
|---|---|---|
| IEEEtran/BibTeX compilation | PASS | 32 cited entries, 32 bibliography entries, no missing or unused key, and zero BibTeX warnings |
| IEEE bibliography normalization | PASS | initials, venue abbreviations, pages, volumes/issues, months, locations, and verified DOI metadata normalized without changing citation keys |
| Claim–citation alignment | PASS WITH NOTES | primary sources were checked for the central optimization, robust-MDP, game-learning, and RARL claims; no retraction or expression-of-concern signal was found in the checked sources |
| Mathematical consistency | PASS AFTER CORRECTION | the pure-rotation curvature-margin formula was corrected; theorem sentinels and saved-result bridge audits pass |
| Stochastic-oracle reporting | PASS | the manuscript reports the prespecified primary-scale finite-trajectory experiment; batch-sensitivity records remain in the project audit archive |
| Experimental provenance | PASS | archived configurations, seeds, CSV/JSON results, checksums, and audit scripts agree with the manuscript |
| Publication narrative | PASS | contribution-led wording retained; local/conditional guarantees and non-global-Nash boundary are stated without turning the paper into an audit log |
| Originality screening | LIMITED PASS | exact-title and distinctive-phrase web searches found no match; no iThenticate/Turnitin or equivalent proprietary database was available |
| ARS formal Material Passport | NOT RUN | this is a standalone recovered manuscript, not an ARS-orchestrated run carrying a pre-existing Schema-9 passport |
| CRediT author contributions | AUTHOR CONFIRMATION REQUIRED | proposed wording is isolated in `SUBMISSION_METADATA_TO_CONFIRM_20260727.md`; it was not inserted as fact |
| TSP fresh-submission length | CONDITIONAL | the compiled paper is 14 pages; current IEEE SPS guidance states 13 pages for an initial regular-paper submission and 16 for a revision |

Accordingly, the appropriate verdict is **PASS WITH TWO AUTHOR-ACTION ITEMS**:
confirm the CRediT roles, and decide whether this is a 13-page initial
submission or a 14-page revision/editor-approved submission.

## What did not meet the checks, and how it was repaired

### 1. Pure-rotation specialization was dimensionally inconsistent

The previous text reduced the curvature benefit to
`gamma - gamma^2/2` for a field
\(\mathbf F(\mathbf z)=\sigma J_0(\mathbf z-\mathbf z^\star)\).
With \(\Phi=\|\mathbf F\|^2\), the exact gamma-only drift coefficient is
\(-\gamma\sigma^2+\gamma^2\sigma^4/2\).  The paper now reports the
corresponding decrease
\[
C_G=\bar\gamma\sigma^2-\bar\gamma^2\sigma^4/2,\qquad
\bar\gamma=\min\{\gamma_{\max},\sigma^{-2}\},
\]
and states that the displayed value \(1/2\) uses \(\sigma=1\).

Why this matters: it removes an unqualified scale cancellation and makes the
analytic statement agree with the code sentinel.

### 2. Publication-facing experiment curation

The manuscript reports the confirmed neural cells and the prespecified
primary-scale finite-trajectory result.  Secondary environment and
batch-sensitivity outcomes are retained in the project audit archive rather
than narrated in the submission.  The reported Holm-adjusted values retain
the original prespecified correction and were not relaxed after curation.

### 3. Scope and disclosure boundaries were incomplete

A compact run-in paragraph now states:

- local and conditional guarantees, with no global Nash claim for general
  neural policies;
- availability of configurations, seeds, raw records, scripts, and figure
  generators;
- no human/animal/personal-data involvement;
- no competing interests;
- the named generative-AI systems, affected manuscript components, and author
  verification.

Funding remains in the title footnote.  CRediT roles were deliberately not
invented.

### 4. The bibliography contained unused and inconsistent records

The source bibliography had 61 records for 32 cited works, including 29 unused
records, one duplicate paper, inconsistent personal-name forms, inconsistent
conference names, and missing verified metadata.  It now contains exactly 32
cited records.  Citation keys were preserved.

Representative metadata corrections include the robust-MDP DOIs, proceedings
locations/months, PMLR page ranges, and recent IEEE TSP volume/page/DOI data.
Two conservative exceptions are intentional:

- the 1976 Korpelevich record does not receive an unverified DOI or month;
- legacy NeurIPS volume-12 records keep their bibliographic publication year
  without mixing it with an unverified event-date field.

## Claim–evidence integrity findings

1. The performance bridge is a deterministic value inequality and does not
   convert local stationarity into a global Nash guarantee.
2. The local smooth-gap proposition is conditional on the stated finite,
   smooth, interior, nonsingular, and bounded-away assumptions.
3. The biased stochastic-oracle results retain bias/variance residuals and
   therefore predict an error floor, not exact stochastic convergence.
4. The box-QP result carries coefficient, solver, safeguard, and accepted-step
   residuals rather than silently treating the implemented step as an exact
   unconstrained minimizer.
5. The experiments support a conditional mechanism claim and robust-return
   improvement over the matched noG ablation; they do not establish universal
   superiority over every optimizer or environment.
6. The experiments make performance and mechanism claims; they do not make a
   wall-clock or oracle-efficiency claim.

## Audit evidence

### Theory sentinels

Directory:
`experiments/paper_suite_20260723/results/theory-sentinels-20260727-183043/`

- box-QP objective error versus SciPy:
  `1.4675760606763788e-15`;
- centered-softplus violation:
  `3.469446951953614e-15`;
- dominance-identity error:
  `4.85722573273506e-17`;
- decomposition-identity error:
  `5.684341886080802e-14`;
- performance-bridge violation: `0`;
- normalized pure-rotation margin error:
  `2.220446049250313e-16`.

### Finite-trajectory stochastic-oracle sentinel

Directory:
`experiments/stochastic_oracle_validation_20260727/results/sentinels-dice-20260727-183607/`

- same-batch autograd/finite-difference HVP relative error:
  `6.153939928539131e-10`;
- field cosine: `0.9873801398476237`;
- curvature cosine: `0.9828056700184815`;
- predicted decrease: `0.05609356062105156`;
- realized same-batch merit decrease: `0.05608411480892239`;
- hard-BR residual: `8.9284e-13`;
- soft-BR residual: `8.4487e-11`.

### Saved bridge and multiplicity audit

Summary:
`experiments/paper_suite_20260723/results/saved-bridge-audit-20260727-183947/summary.json`

- 3780 tabular and 3120 neural checkpoint rows;
- zero bridge violations;
- minimum tabular/neural bridge slack:
  `0.3695118667` / `0.3859413174`;
- neural Holm-adjusted p-values:
  CyclicControl/FrequencyHopping/Routing `0.0048828`,
  SecurityAllocation `0.021484`,
  PursuitEvasion `0.171875`.

### Formal finite-trajectory audit

File:
`experiments/stochastic_oracle_validation_20260727/results/formal-CyclicControl-dice-20260727-124224/formal_audit.json`

- complete Cartesian keys, finite metrics, and exact paired initialization;
- 630 curve rows and 5400 diagnostic rows;
- B=128 QP+G minus noG:
  `-2.6609`, CI `[-4.8468,-0.4750]`;
- B=512:
  `0.0727`, CI `[-0.2631,0.4085]`;
- prespecified B=2048:
  `0.3422`, CI `[0.1230,0.5614]`, Holm-adjusted
  `p=0.010742`, positive for 9/10 seeds.

SHA-256:

- curves:
  `5f4d62768f7e7a0867df14abff1859c8117c11cc858ff514492209edc195e3a8`;
- diagnostics:
  `e5d47526b1942103c9df8818d756e07d845f47e18dce02e3c3a77d717367d861`.

Final synchronized 14-page manuscript:
`C61B75E9A3ADA2C60A97EFF4B1960BA52207315BD48BEB2ED990422113666282`.

## Bibliography audit summary

- BibTeX records: 32;
- distinct cited keys: 32;
- missing keys: 0;
- unused keys: 0;
- duplicate normalized titles: 0;
- non-`Proc.` conference booktitles: 0;
- brace imbalance: 0;
- BibTeX warnings: 0.

The `Nilim, A. and El Ghaoui, L.` name is intentionally retained: “El Ghaoui”
is a compound family name, not an uninitialized given name.

## Seven generative-AI failure-mode checks

1. **Citation existence:** checked against primary publisher/proceedings pages
   for the central claims.
2. **Metadata accuracy:** corrected against those pages and compiled through
   IEEEtran.bst.
3. **Claim inflation:** local/conditional and error-floor boundaries are
   explicit.
4. **Fabricated experiment records:** manuscript values trace to archived
   CSV/JSON files and deterministic audit scripts.
5. **Selection integrity:** publication-facing figures contain the confirmed
   cells, while the original multiplicity correction and complete
   project-local records are preserved.
6. **Textual originality:** risk-based exact-phrase web screening found no
   match, but no proprietary similarity database was available.
7. **AI-use transparency:** assistance and author verification are disclosed
   in the manuscript.

## Required integrity caveat

This check verifies disclosure and claim-to-provenance fidelity. It does not
judge whether the experiment was correctly designed, run, statistically
adequate, or reproducible by ARS.

The audit nevertheless executed the available project-local theory,
trajectory-oracle, bridge, and multiplicity sentinels.  It did not obtain a
professional Retraction Watch export, a proprietary originality score, or a
formal ARS Material Passport, so none of those is claimed.
