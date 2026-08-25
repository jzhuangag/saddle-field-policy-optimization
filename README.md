# Saddle-Field Policy Optimization for Zero-Sum Markov Games

This repository contains the manuscript, implementation, frozen experiment
artifacts, and plotting pipeline for:

> **Saddle-Field Policy Optimization for Zero-Sum Markov Games: Adaptive
> Lyapunov-Drift Control with Finite-Time Stationarity Guarantees**

The release is scoped to this journal manuscript and its reproducibility
artifacts.

The method combines the saddle-field direction $-F$ with the
Jacobian--vector curvature direction $G=\nabla F\,F$. A two-variable box
quadratic program selects their step sizes independently from a local
second-order Lyapunov-drift model.

## Artifact contents

- [`main.tex`](main.tex), [`refs.bib`](refs.bib), and
  [`RARL_final.pdf`](RARL_final.pdf): manuscript source, bibliography, and
  rendered paper.
- [`experiments/paper_suite_20260723/`](experiments/paper_suite_20260723/):
  analytical geometry, tabular and neural population-oracle experiments,
  theory sentinels, baseline tuning, performance-bridge audit, and paper
  figure assembly.
- [`experiments/stochastic_oracle_validation_20260727/`](experiments/stochastic_oracle_validation_20260727/):
  finite-trajectory DiCE experiment, formal statistical audit, and Table I
  reconstruction.
- [`output/data/`](output/data/): frozen-artifact manifest and manuscript
  summary tables.
- [`output/pdf/`](output/pdf/): vector figures consumed by the LaTeX source.
- [`docs/verification/`](docs/verification/): machine-readable citation and
  IEEE bibliography-style release gates.
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md): protocols and exact full-rerun
  commands.

The neural population suite contains the four environments reported in the
paper: `CyclicControl`, `FrequencyHopping`, `RoutingInterdiction`, and
`SecurityPatrol`.

## Reference environment

The released artifacts were produced with:

| Component | Version |
|---|---:|
| Python | 3.8.12 |
| NumPy | 1.20.3 |
| SciPy | 1.7.3 |
| Matplotlib | 3.5.1 |
| PyTorch | 1.11.0+cpu |

Install the Python dependencies in an isolated environment:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Activate `.venv` using the command appropriate for your shell before running
the commands below. A TeX distribution containing `IEEEtran`, BibTeX, and
`latexmk` is additionally required to rebuild the paper.

## Frozen-data reproduction

The root entry point provides four fast frozen-data modes and one complete
experiment-to-paper mode:

```bash
python reproduce.py verify
python reproduce.py figures
python reproduce.py paper
python reproduce.py all
python reproduce.py full
```

| Mode | Action |
|---|---|
| `verify` | Check every manifest source using canonical LF-normalized SHA-256 and verify the formal environment sets. |
| `figures` | Verify the frozen data, regenerate all manuscript figures, and reconstruct the finite-trajectory table statistics. |
| `paper` | Verify the frozen data and compile `main.tex` with `latexmk`. |
| `all` | Run verification, figure/table reconstruction, and paper compilation in sequence. |
| `full` | Rerun every reported experiment, tune and merge the neural baselines, rebuild all figures and Table I, audit the new outputs, and compile the paper. |

The paper build is written to `output/build/reproduce/main.pdf`; it does not
overwrite the released `RARL_final.pdf`. The `full` mode is substantially more
expensive than the frozen-data modes; it writes new timestamped result
directories and passes them explicitly through the assembly pipeline.

## Experiment map

| Paper component | Primary program | Frozen source |
|---|---|---|
| Rotational linear geometry | `linear_geometry.py` | `linear-geometry-20260823-234632/` |
| Tabular population games | `markov_game_suite.py --mode tabular` | `tabular-exact-gap-20260723-113009/` |
| Neural population games | `markov_game_suite.py --mode neural`, `tune_fixed_baselines.py`, and `merge_neural_journal.py` | `neural-journal-four-20260824/` |
| Finite-trajectory validation | `stochastic_dice_policy.py --phase formal` | `formal-CyclicControl-dice-20260727-124224/` |

All experiment programs create timestamped result directories and leave the
frozen release artifacts unchanged. See
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for seeds, update counts, exact
commands, output files, and audit steps.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff).
