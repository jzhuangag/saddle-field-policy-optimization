# Reproducibility guide

This guide separates fast reconstruction from frozen data and complete
experiment reruns. The fast path reproduces the manuscript artifacts without
retraining. Full reruns write new timestamped raw-result directories without
overwriting the frozen raw evidence; their final assembly intentionally
refreshes the derived files under `output/pdf/` and `output/data/`.

## 1. Environment

The reference CPU environment is:

```text
Python      3.8.12
NumPy       1.20.3
SciPy       1.7.3
Matplotlib  3.5.1
PyTorch     1.11.0+cpu
```

Create and populate an isolated Python environment:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the manuscript build, install a TeX distribution providing `IEEEtran`,
BibTeX, and `latexmk`. Run all commands from the repository root.

## 2. Fast reconstruction from frozen data

### Verify release integrity

```bash
python reproduce.py verify
```

The verifier reads `output/data/final_experiment_manifest.json`, checks every
listed CSV/JSON source, and requires the neural population data to contain
exactly `CyclicControl`, `FrequencyHopping`, `RoutingInterdiction`, and
`SecurityPatrol`. For cross-platform stability, text bytes are normalized from
CRLF or CR to LF before both SHA-256 and byte-count comparison.

### Regenerate figures and Table I statistics

```bash
python reproduce.py figures
```

This mode runs, in order:

```bash
python experiments/paper_suite_20260723/motivation_geometry_abd_wide.py
python experiments/paper_suite_20260723/assemble_paper_results.py
python experiments/stochastic_oracle_validation_20260727/make_vi_d_table.py
```

The vector manuscript figures are written to `output/pdf/`. Population
summaries and the refreshed manifest are written to `output/data/`. The
finite-trajectory table audit is written to
`output/data/vi_d_finite_trajectory_table.json`, and its three data-driven
LaTeX rows are written to `output/data/vi_d_table_rows.tex` for direct
inclusion by `main.tex`.

### Compile the manuscript

```bash
python reproduce.py paper
```

The compiled file is `output/build/reproduce/main.pdf`. To reconstruct all
frozen-data artifacts and then compile the paper, run:

```bash
python reproduce.py all
```

## 3. Frozen release protocols

### Linear geometry

- Field: `F(z) = (mu I + sigma J) z`.
- Methods: `QP+G`, `noG`, `GDA`, `Adam-GDA`, `EGM`, and `PPM-3`.
- Updates: 120.
- Field and curvature caps: 0.03.
- Frozen directory:
  `experiments/paper_suite_20260723/results/linear-geometry-20260823-234632/`.

### Tabular population-oracle games

- Environments: `RPS`, `CyclicControl`, and `FrequencyHopping`.
- Seeds: 200--209.
- Updates: 100.
- Methods: `QP+G`, `noG`, `GDA`, `Adam-GDA`, `EGM`, and `PPM-3`.
- Frozen directory:
  `experiments/paper_suite_20260723/results/tabular-exact-gap-20260723-113009/`.

### Neural population-oracle games

- Environments: `CyclicControl`, `FrequencyHopping`, `RoutingInterdiction`,
  and `SecurityPatrol`.
- Two separate 4--8--3 tanh--softmax policies.
- Seeds: 40--49.
- Updates: 60.
- Controller caps: 0.03.
- Fixed-baseline learning rates: GDA 0.03, Adam-GDA 0.001, EGM 0.03,
  and PPM-3 0.03.
- Frozen directory:
  `experiments/paper_suite_20260723/results/neural-journal-four-20260824/`.

The fixed-baseline rates were selected once across all four environments on
seeds 1000--1004 from `{0.001, 0.003, 0.01, 0.03}`. The reporting seeds are
disjoint from the tuning seeds.

### Finite-trajectory validation

- Environment: `CyclicControl`.
- Two separate 4--8--3 tanh--softmax policies.
- Formal seeds: 4100--4109.
- Updates: 60.
- Horizon: 16.
- Transition batches: 128, 512, and 2048.
- Methods: `QP+G`, `noG`, and `EGM`.
- Frozen directory:
  `experiments/stochastic_oracle_validation_20260727/results/formal-CyclicControl-dice-20260727-124224/`.

## 4. Complete experiment reruns

The complete experiment-to-paper workflow is automated by:

```bash
python reproduce.py full
```

This command reruns every reported experiment, performs the independent
fixed-baseline screen, merges the controller and selected-baseline results,
reruns both formal audits, rebuilds the figures and Table I from the newly
created directories, and compiles the manuscript. It is substantially more
expensive than `reproduce.py all`, which uses frozen data. Every experiment
stage creates a new timestamped result directory and prints its path.

The component commands below expose the same workflow for selective reruns.

### Analytical geometry and numerical sentinels

```bash
python experiments/paper_suite_20260723/linear_geometry.py
python experiments/paper_suite_20260723/theory_sentinels.py
```

The motivation graphic is deterministic and can be regenerated separately:

```bash
python experiments/paper_suite_20260723/motivation_geometry_abd_wide.py
```

### Tabular population-oracle suite

```bash
python experiments/paper_suite_20260723/markov_game_suite.py --mode tabular --environments RPS CyclicControl FrequencyHopping --seeds 10 --seed-start 200 --steps 100
```

### Neural population-oracle controller and ablation

```bash
python experiments/paper_suite_20260723/markov_game_suite.py --mode neural --environments CyclicControl FrequencyHopping RoutingInterdiction SecurityPatrol --seeds 10 --seed-start 40 --steps 60 --fixed-lr 0.03 --methods QP+G noG
```

### Neural fixed-baseline tuning and evaluation

```bash
python experiments/paper_suite_20260723/tune_fixed_baselines.py
```

This program evaluates the four-point learning-rate grid on seeds 1000--1004,
selects one global rate per fixed method, and evaluates the selected rates on
seeds 40--49. Its output `summary.json` records all selected rates, tuning
scores, and timestamped source directories.

Merge a newly generated controller directory and tuning result into the
six-method journal dataset:

```bash
python experiments/paper_suite_20260723/merge_neural_journal.py --controller-dir PATH_TO_CONTROLLER_RESULT --tuning-summary PATH_TO_TUNING_RESULT/summary.json
```

The merger validates the exact four-environment, ten-seed, 60-update protocol
before writing a new `neural-journal-four-*` directory. To build population
figures from selected rerun directories rather than the frozen defaults, run:

```bash
python experiments/paper_suite_20260723/assemble_paper_results.py --linear-dir PATH_TO_LINEAR_RESULT --tabular-dir PATH_TO_TABULAR_RESULT --neural-dir PATH_TO_MERGED_NEURAL_RESULT
```

### Performance-bridge audit on frozen checkpoints

```bash
python experiments/paper_suite_20260723/audit_saved_bridge.py
```

### Formal finite-trajectory run and audit

```bash
python experiments/stochastic_oracle_validation_20260727/sentinels_dice.py
python experiments/stochastic_oracle_validation_20260727/stochastic_dice_policy.py --phase formal
```

The formal runner prints an `OUTPUT=` path. Pass that directory to the audit:

```bash
python experiments/stochastic_oracle_validation_20260727/audit_formal.py PATH_PRINTED_AFTER_OUTPUT
```

Then reconstruct the table and its LaTeX rows from that same run:

```bash
python experiments/stochastic_oracle_validation_20260727/make_vi_d_table.py --source-dir PATH_PRINTED_AFTER_OUTPUT
```

The formal run is the most computationally intensive component. It uses only
CPU-compatible PyTorch operations in the reference configuration.

## 5. Output interpretation

Population runs produce:

- `curves.csv`: checkpoint metrics for every method and seed;
- `diagnostics.csv`: QP geometry, step sizes, and predicted/realized decrease;
- `summary.csv`: final mean and standard-error summaries;
- `summary.json`: protocol, numerical tolerances, decisions, and elapsed time;
- vector PDF and PNG previews for the run.

The finite-trajectory run additionally writes `protocol.json` and its formal
audit records paired matched-seed statistics. Random seeds, update counts,
policy sizes, step-size caps, Bellman tolerances, and evaluation definitions
are stored with the results.

Small last-digit differences in rendered raster previews can arise from local
font availability. The statistical quantities are reconstructed from the
frozen CSV/JSON files, while the manuscript consumes vector PDF figures.
