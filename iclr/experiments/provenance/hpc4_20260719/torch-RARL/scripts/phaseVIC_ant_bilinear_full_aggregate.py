from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ("gda", "egm", "ppm", "nog", "qpg")
COLORS = {"gda": "#4D4D4D", "egm": "#E69F00", "ppm": "#009E73", "nog": "#2878B5", "qpg": "#C53D32"}
LABELS = {"gda": "GDA", "egm": "EGM", "ppm": "PPM", "nog": "noG", "qpg": "QP+G"}


def trapezoid(frame: pd.DataFrame, column: str) -> float:
    ordered = frame.sort_values("protagonist_env_steps")
    return float(np.trapezoid(ordered[column].to_numpy(float), ordered["protagonist_env_steps"].to_numpy(float)))


def bootstrap_ci(values: np.ndarray, seed: int = 20260720) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(values, size=values.size, replace=True).mean() for _ in range(20000)])
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def plateau_metrics(values: pd.Series) -> dict[str, float | bool]:
    y = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if y.size < 5:
        return {"cv": float("inf"), "min_over_median": float("-inf"), "relative_range": float("inf"),
                "relative_trend": float("inf"), "stable": False}
    mean = float(np.mean(y))
    median = float(np.median(y))
    cv = float(np.std(y, ddof=1) / max(abs(mean), 1e-12)) if y.size > 1 else 0.0
    min_over_median = float(np.min(y) / max(abs(median), 1e-12))
    relative_range = float((np.max(y) - np.min(y)) / max(abs(mean), 1e-12))
    relative_trend = float(abs(np.polyfit(np.arange(y.size), y, 1)[0]) * (y.size - 1) / max(abs(mean), 1e-12))
    stable = bool(cv <= 0.15 and min_over_median >= 0.80 and relative_range <= 0.40 and relative_trend <= 0.20)
    return {"cv": cv, "min_over_median": min_over_median, "relative_range": relative_range,
            "relative_trend": relative_trend, "stable": stable}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    curves: list[pd.DataFrame] = []
    updates: list[pd.DataFrame] = []
    for seed in range(5):
        for method in METHODS:
            c = pd.read_csv(args.root / f"seed_{seed}" / method / "convergence.csv")
            c["seed"] = seed
            c["method"] = method
            curves.append(c)
            u = pd.read_csv(args.root / f"seed_{seed}" / method / "updates.csv")
            u["seed"] = seed
            u["method"] = method
            updates.append(u)
    all_curves = pd.concat(curves, ignore_index=True)
    all_updates = pd.concat(updates, ignore_index=True)
    all_curves.to_csv(args.output / "all_convergence.csv", index=False)
    all_updates.to_csv(args.output / "all_updates.csv", index=False)

    grouped = (
        all_curves.groupby(["method", "protagonist_env_steps"], as_index=False)
        .agg(
            train_mean=("train_episode_return", "mean"),
            train_sem=("train_episode_return", "sem"),
            robust_mean=("own_adversary_robust_return", "mean"),
            robust_sem=("own_adversary_robust_return", "sem"),
            clean_mean=("clean_return", "mean"),
            clean_sem=("clean_return", "sem"),
        )
    )
    grouped.to_csv(args.output / "mean_convergence.csv", index=False)

    qpg = all_curves.loc[all_curves.method == "qpg"]
    nog = all_curves.loc[all_curves.method == "nog"]
    paired = qpg.merge(nog, on=["seed", "protagonist_env_steps"], suffixes=("_qpg", "_nog"), validate="one_to_one")
    paired["robust_gain"] = paired["own_adversary_robust_return_qpg"] - paired["own_adversary_robust_return_nog"]
    paired["clean_gain"] = paired["clean_return_qpg"] - paired["clean_return_nog"]
    paired.to_csv(args.output / "paired_checkpoint_gains.csv", index=False)

    endpoint_rows: list[dict] = []
    auc_rows: list[dict] = []
    for seed in range(5):
        for method in METHODS:
            run = all_curves.loc[(all_curves.seed == seed) & (all_curves.method == method)].sort_values("protagonist_env_steps")
            final = run.iloc[-1]
            endpoint_rows.append({
                "seed": seed,
                "method": method,
                "clean_return": float(final.clean_return),
                "own_robust_return": float(final.own_adversary_robust_return),
            })
            auc_rows.append({
                "seed": seed,
                "method": method,
                "clean_auc": trapezoid(run, "clean_return"),
                "robust_auc": trapezoid(run, "own_adversary_robust_return"),
            })
    endpoints = pd.DataFrame(endpoint_rows)
    aucs = pd.DataFrame(auc_rows)
    endpoints.to_csv(args.output / "endpoints.csv", index=False)
    aucs.to_csv(args.output / "aucs.csv", index=False)

    ep_pair = endpoints.loc[endpoints.method == "qpg"].merge(
        endpoints.loc[endpoints.method == "nog"], on="seed", suffixes=("_qpg", "_nog"), validate="one_to_one"
    )
    auc_pair = aucs.loc[aucs.method == "qpg"].merge(
        aucs.loc[aucs.method == "nog"], on="seed", suffixes=("_qpg", "_nog"), validate="one_to_one"
    )
    robust_ep_gain = (ep_pair.own_robust_return_qpg - ep_pair.own_robust_return_nog).to_numpy(float)
    clean_ep_gain = (ep_pair.clean_return_qpg - ep_pair.clean_return_nog).to_numpy(float)
    robust_auc_gain = (auc_pair.robust_auc_qpg - auc_pair.robust_auc_nog).to_numpy(float)
    clean_auc_gain = (auc_pair.clean_auc_qpg - auc_pair.clean_auc_nog).to_numpy(float)
    late = paired.sort_values("protagonist_env_steps").groupby("seed", as_index=False).tail(5)
    qpg_updates = all_updates.loc[all_updates.method == "qpg"]
    late_plateau: dict[str, dict[str, dict[str, float | bool]]] = {}
    for method in METHODS:
        d = grouped.loc[grouped.method == method].sort_values("protagonist_env_steps").tail(5)
        late_plateau[method] = {
            "training": plateau_metrics(d.train_mean),
            "own_robust": plateau_metrics(d.robust_mean),
            "clean": plateau_metrics(d.clean_mean),
        }
    summary = {
        "env": "Ant-v4",
        "joint_training_seeds": 5,
        "shared_clean_pretraining_checkpoint": True,
        "robust_endpoint_mean_gain": float(robust_ep_gain.mean()),
        "robust_endpoint_bootstrap_95ci": bootstrap_ci(robust_ep_gain),
        "robust_endpoint_wins": int(np.sum(robust_ep_gain > 0.0)),
        "robust_auc_mean_gain": float(robust_auc_gain.mean()),
        "robust_auc_bootstrap_95ci": bootstrap_ci(robust_auc_gain, 20260721),
        "robust_auc_wins": int(np.sum(robust_auc_gain > 0.0)),
        "late_positive_checkpoint_fraction": float(np.mean(late.robust_gain > 0.0)),
        "clean_endpoint_mean_gain": float(clean_ep_gain.mean()),
        "clean_endpoint_bootstrap_95ci": bootstrap_ci(clean_ep_gain, 20260722),
        "clean_endpoint_wins": int(np.sum(clean_ep_gain > 0.0)),
        "clean_auc_mean_gain": float(clean_auc_gain.mean()),
        "clean_auc_wins": int(np.sum(clean_auc_gain > 0.0)),
        "gamma_active_fraction": float(np.mean(pd.to_numeric(qpg_updates.gamma_active, errors="coerce") > 0.0)),
        "curvature_reliability_fraction": float(pd.to_numeric(qpg_updates.curvature_reliability_gate, errors="coerce").mean()),
        "predicted_inclusion_pass_fraction": float(pd.to_numeric(qpg_updates.predicted_inclusion_pass, errors="coerce").mean()),
        "realized_nog_safeguard_pass_fraction": float(pd.to_numeric(qpg_updates.realized_qp_le_nog, errors="coerce").mean()),
        "late_plateau": late_plateau,
    }
    summary["all_method_convergence_gate"] = bool(
        all(channel["stable"] for method in late_plateau.values() for channel in method.values())
    )
    summary["paired_performance_gate"] = bool(
        summary["robust_endpoint_mean_gain"] > 0.0
        and summary["robust_endpoint_wins"] >= 3
        and summary["robust_auc_mean_gain"] > 0.0
        and summary["robust_auc_wins"] >= 3
        and summary["late_positive_checkpoint_fraction"] >= 0.80
        and summary["clean_endpoint_mean_gain"] >= -0.10 * float(endpoints.loc[endpoints.method == "nog", "clean_return"].mean())
        and summary["all_method_convergence_gate"]
    )
    summary["mechanism_gate"] = bool(
        summary["gamma_active_fraction"] >= 0.10
        and summary["curvature_reliability_fraction"] >= 0.60
        and summary["predicted_inclusion_pass_fraction"] == 1.0
        and summary["realized_nog_safeguard_pass_fraction"] == 1.0
    )
    summary["decision_before_common_bank"] = "POSITIVE" if summary["paired_performance_gate"] and summary["mechanism_gate"] else "NOT_POSITIVE"
    (args.output / "full_suite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for method in METHODS:
        d = grouped.loc[grouped.method == method].sort_values("protagonist_env_steps")
        x = d.protagonist_env_steps.to_numpy(float) / 1000.0
        for ax, mean_col, sem_col in (
            (axes[0, 0], "train_mean", "train_sem"),
            (axes[0, 1], "robust_mean", "robust_sem"),
            (axes[1, 0], "clean_mean", "clean_sem"),
        ):
            mean = d[mean_col].to_numpy(float)
            sem = d[sem_col].fillna(0.0).to_numpy(float)
            ax.plot(x, mean, color=COLORS[method], label=LABELS[method], linewidth=2)
            ax.fill_between(x, mean - sem, mean + sem, color=COLORS[method], alpha=0.12)
    gain = paired.groupby("protagonist_env_steps").robust_gain.agg(["mean", "sem"]).reset_index()
    x = gain.protagonist_env_steps.to_numpy(float) / 1000.0
    mean = gain["mean"].to_numpy(float)
    sem = gain["sem"].fillna(0.0).to_numpy(float)
    axes[1, 1].axhline(0.0, color="black", linewidth=1)
    axes[1, 1].plot(x, mean, color=COLORS["qpg"], linewidth=2)
    axes[1, 1].fill_between(x, mean - sem, mean + sem, color=COLORS["qpg"], alpha=0.18)
    axes[0, 0].set_title("Training episode return")
    axes[0, 1].set_title("Frozen co-trained-pair robust evaluation")
    axes[1, 0].set_title("Frozen clean evaluation")
    axes[1, 1].set_title("Paired robust gain (QP+G - noG)")
    for ax in axes.flat:
        ax.set_xlabel("Protagonist environment steps (thousands)")
        ax.set_ylabel("Undiscounted native task return")
        ax.grid(alpha=0.2)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.legend(ncol=2, fontsize=9)
    fig.suptitle("Ant-v4 native-reward exact joint actor RARL: five joint-training seeds")
    fig.savefig(args.output / "ant_bilinear_full_convergence.png", dpi=220)
    fig.savefig(args.output / "ant_bilinear_full_convergence.pdf")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
