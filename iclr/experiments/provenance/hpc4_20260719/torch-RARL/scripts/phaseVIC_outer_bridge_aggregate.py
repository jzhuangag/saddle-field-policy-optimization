from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phaseVIC_baseline_gate_aggregate import plateau, training_bins


MODES = ("none", "diagnostic", "gda")
LABELS = {"none": "No outer correction", "diagnostic": "Diagnostic only", "gda": "GDA outer correction"}
COLORS = {"none": "#555555", "diagnostic": "#1f77b4", "gda": "#d62728"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--modes", default=",".join(MODES))
    return parser.parse_args()


def mean_sem(frame: pd.DataFrame, x: str, y: str):
    grouped = frame.groupby(x)[y]
    mean = grouped.mean()
    sem = grouped.sem().fillna(0.0)
    return mean.index.to_numpy(dtype=float), mean.to_numpy(dtype=float), sem.to_numpy(dtype=float)


def main():
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    modes = tuple(value.strip() for value in args.modes.split(",") if value.strip())
    unknown = set(modes) - set(MODES)
    if unknown:
        raise ValueError(f"unknown modes: {sorted(unknown)}")
    args.output.mkdir(parents=True, exist_ok=True)
    training_frames, clean_frames, robust_frames = [], [], []
    plateau_rows, mechanism_rows = [], []
    for mode in modes:
        for seed in seeds:
            run = args.root / mode / f"seed_{seed}"
            training = pd.read_csv(run / "short_training_return_all_methods.csv")
            clean = pd.read_csv(run / "short_clean_eval_all_methods.csv")
            robust = pd.read_csv(run / "short_adv_eval_force_all_methods.csv")
            training["rolling"] = training["episode_return"].rolling(20, min_periods=1).mean()
            for frame in (training, clean, robust):
                frame["mode"] = mode
                frame["seed"] = seed
            training_frames.append(training)
            clean_frames.append(clean)
            robust_frames.append(robust)
            binned = training_bins(training, clean["timesteps"].to_numpy(dtype=float))
            for metric, values in (
                ("training_binned", binned["mean_reward"].to_numpy()),
                ("clean", clean["mean_reward"].to_numpy()),
                ("robust", robust["mean_reward"].to_numpy()),
            ):
                row = {"mode": mode, "seed": seed, "metric": metric}
                row.update(plateau(values))
                plateau_rows.append(row)
            if mode != "none":
                candidates = list(run.glob("runs/adam/saved_models/**/analysis/composite_outer_diagnostics.csv"))
                if len(candidates) != 1:
                    raise RuntimeError(f"expected one diagnostics file under {run}, found {candidates}")
                diagnostics = pd.read_csv(candidates[0])
                late = diagnostics.tail(20)
                mechanism_rows.append(
                    {
                        "mode": mode,
                        "seed": seed,
                        "rows": len(diagnostics),
                        "applied_update_fraction": float(diagnostics["applied_update"].mean()),
                        "late_corr_Q_MC": float(late["corr_Q_MC"].mean()),
                        "late_WF_over_SF": float(late["WF_over_SF"].mean()),
                        "late_cross_to_same_ratio": float(late["cross_to_same_ratio"].mean()),
                    }
                )

    training_all = pd.concat(training_frames, ignore_index=True)
    clean_all = pd.concat(clean_frames, ignore_index=True)
    robust_all = pd.concat(robust_frames, ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    for mode in modes:
        for axis, frame, x_col, y_col in (
            (axes[0], training_all[training_all["mode"] == mode], "cumulative_timesteps", "rolling"),
            (axes[1], clean_all[clean_all["mode"] == mode], "timesteps", "mean_reward"),
            (axes[2], robust_all[robust_all["mode"] == mode], "timesteps", "mean_reward"),
        ):
            x, mean, sem = mean_sem(frame, x_col, y_col)
            axis.plot(x / 1000.0, mean, color=COLORS[mode], label=LABELS[mode])
            axis.fill_between(x / 1000.0, mean - sem, mean + sem, color=COLORS[mode], alpha=0.16)
    titles = (
        "Protagonist training return (20-episode rolling mean)",
        "Frozen clean evaluation (raw checkpoints)",
        "Co-trained robust evaluation (raw checkpoints)",
    )
    for axis, title in zip(axes, titles):
        axis.set_title(title)
        axis.set_xlabel("Protagonist environment steps (thousands)")
        axis.set_ylabel("Native task return")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(args.output / "outer_bridge_convergence.png", dpi=180)
    fig.savefig(args.output / "outer_bridge_convergence.pdf")
    plt.close(fig)
    pd.DataFrame(plateau_rows).to_csv(args.output / "outer_bridge_plateau.csv", index=False)
    pd.DataFrame(mechanism_rows).to_csv(args.output / "outer_bridge_mechanism.csv", index=False)


if __name__ == "__main__":
    main()
