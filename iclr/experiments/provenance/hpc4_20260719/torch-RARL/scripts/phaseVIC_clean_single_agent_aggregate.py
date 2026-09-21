from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ("gda", "nog", "qpg")
LABELS = {"gda": "SGD/GDA", "nog": "noG", "qpg": "QP+G"}
COLORS = {"gda": "#4D4D4D", "nog": "#2878B5", "qpg": "#C53D32"}


def plateau(values: pd.Series) -> dict[str, float | bool]:
    y = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if y.size < 5:
        return {"cv": float("inf"), "min_over_median": float("-inf"), "relative_range": float("inf"),
                "relative_trend": float("inf"), "stable": False}
    mean = float(np.mean(y))
    median = float(np.median(y))
    scale = max(abs(mean), 1e-12)
    cv = float(np.std(y, ddof=1) / scale)
    min_over_median = float(np.min(y) / max(abs(median), 1e-12))
    relative_range = float((np.max(y) - np.min(y)) / scale)
    relative_trend = float(abs(np.polyfit(np.arange(y.size), y, 1)[0]) * (y.size - 1) / scale)
    stable = bool(cv <= 0.15 and min_over_median >= 0.80 and relative_range <= 0.40 and relative_trend <= 0.20)
    return {"cv": cv, "min_over_median": min_over_median, "relative_range": relative_range,
            "relative_trend": relative_trend, "stable": stable}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    curves, updates = [], []
    for seed in range(5):
        for method in METHODS:
            c = pd.read_csv(args.root / f"seed_{seed}" / method / "convergence.csv")
            c["seed"], c["method"] = seed, method
            curves.append(c)
            u = pd.read_csv(args.root / f"seed_{seed}" / method / "updates.csv")
            u["seed"], u["method"] = seed, method
            updates.append(u)
    all_curves = pd.concat(curves, ignore_index=True)
    all_updates = pd.concat(updates, ignore_index=True)
    all_curves.to_csv(args.output / "all_convergence.csv", index=False)
    all_updates.to_csv(args.output / "all_updates.csv", index=False)

    mean = all_curves.groupby(["method", "protagonist_env_steps"], as_index=False).agg(
        train_mean=("train_episode_return", "mean"), train_sem=("train_episode_return", "sem"),
        clean_mean=("clean_return", "mean"), clean_sem=("clean_return", "sem"),
        skew_mean=("skew_symmetry_error", "mean"), skew_max=("skew_symmetry_error", "max"),
        corr_mean=("corr_Q_MC", "mean"),
    )
    mean.to_csv(args.output / "mean_convergence.csv", index=False)

    qpg = all_curves.loc[all_curves.method == "qpg"]
    nog = all_curves.loc[all_curves.method == "nog"]
    paired = qpg.merge(nog, on=["seed", "protagonist_env_steps"], suffixes=("_qpg", "_nog"), validate="one_to_one")
    paired["clean_gain"] = paired.clean_return_qpg - paired.clean_return_nog
    paired_mean = paired.groupby("protagonist_env_steps", as_index=False).clean_gain.agg(["mean", "sem"]).reset_index()
    paired.to_csv(args.output / "paired_clean_gains.csv", index=False)

    late_plateau = {}
    for method in METHODS:
        d = mean.loc[mean.method == method].sort_values("protagonist_env_steps").tail(5)
        late_plateau[method] = {"training": plateau(d.train_mean), "clean": plateau(d.clean_mean)}
    qpg_updates = all_updates.loc[all_updates.method == "qpg"]
    summary = {
        "env": "Ant-v4",
        "reward": "native Gymnasium reward",
        "force_identically_zero": bool((all_curves.force_norm == 0.0).all()),
        "optimization_variables": ["protagonist_actor"],
        "joint_training_seeds": 5,
        "shared_pretrained_checkpoint": True,
        "late_plateau": late_plateau,
        "maximum_numerical_skew_error": float(all_curves.skew_symmetry_error.max()),
        "qpg_gamma_active_fraction": float((pd.to_numeric(qpg_updates.gamma_active, errors="coerce") > 0.0).mean()),
        "qpg_mean_G_contribution_norm": float(pd.to_numeric(qpg_updates.G_contribution_norm, errors="coerce").mean()),
        "predicted_inclusion_pass_fraction": float(pd.to_numeric(qpg_updates.predicted_inclusion_pass, errors="coerce").mean()),
        "realized_nog_safeguard_pass_fraction": float(pd.to_numeric(qpg_updates.realized_qp_le_nog, errors="coerce").mean()),
        "final_paired_clean_gain_mean": float(paired.sort_values("protagonist_env_steps").groupby("seed").tail(1).clean_gain.mean()),
    }
    summary["all_curves_converged"] = bool(
        all(channel["stable"] for method in late_plateau.values() for channel in method.values())
    )
    summary["single_agent_invariant_gate"] = bool(
        summary["force_identically_zero"]
        and summary["maximum_numerical_skew_error"] <= 1e-4
        and summary["predicted_inclusion_pass_fraction"] == 1.0
        and summary["realized_nog_safeguard_pass_fraction"] == 1.0
    )
    summary["decision"] = "PASS" if summary["all_curves_converged"] and summary["single_agent_invariant_gate"] else "FAIL"
    (args.output / "clean_single_agent_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for method in METHODS:
        d = mean.loc[mean.method == method].sort_values("protagonist_env_steps")
        x = d.protagonist_env_steps.to_numpy(float) / 1000.0
        for ax, ycol, secol in ((axes[0, 0], "train_mean", "train_sem"), (axes[0, 1], "clean_mean", "clean_sem")):
            y = d[ycol].to_numpy(float)
            se = d[secol].fillna(0.0).to_numpy(float)
            ax.plot(x, y, color=COLORS[method], label=LABELS[method], linewidth=2)
            ax.fill_between(x, y - se, y + se, color=COLORS[method], alpha=0.12)
        axes[1, 1].plot(x, d.skew_mean, color=COLORS[method], label=LABELS[method], linewidth=2)
    px = paired_mean.protagonist_env_steps.to_numpy(float) / 1000.0
    py = paired_mean["mean"].to_numpy(float)
    pse = paired_mean["sem"].fillna(0.0).to_numpy(float)
    axes[1, 0].axhline(0.0, color="black", linewidth=1)
    axes[1, 0].plot(px, py, color=COLORS["qpg"], linewidth=2)
    axes[1, 0].fill_between(px, py - pse, py + pse, color=COLORS["qpg"], alpha=0.15)
    axes[0, 0].set_title("Clean training episode return")
    axes[0, 1].set_title("Frozen clean evaluation")
    axes[1, 0].set_title("Paired clean gain (QP+G - noG)")
    axes[1, 1].set_title("Numerical antisymmetric-Jacobian error")
    for ax in axes.flat:
        ax.set_xlabel("Protagonist environment steps (thousands)")
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Undiscounted native task return")
    axes[0, 1].set_ylabel("Undiscounted native task return")
    axes[1, 0].set_ylabel("Return difference")
    axes[1, 1].set_ylabel("Relative symmetry error")
    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[1, 1].legend()
    fig.suptitle("Ant-v4 true single-agent optimizer sanity check")
    fig.savefig(args.output / "clean_single_agent_convergence.png", dpi=220)
    fig.savefig(args.output / "clean_single_agent_convergence.pdf")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
