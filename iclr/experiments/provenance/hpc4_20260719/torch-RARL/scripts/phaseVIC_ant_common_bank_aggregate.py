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


def bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(values, size=values.size, replace=True).mean() for _ in range(20000)])
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def auc(frame: pd.DataFrame, column: str) -> float:
    d = frame.sort_values("protagonist_env_steps")
    return float(np.trapezoid(d[column].to_numpy(float), d.protagonist_env_steps.to_numpy(float)))


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
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--full-aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frames = [pd.read_csv(p) for p in sorted(args.bank_root.glob("*_seed_*.csv"))]
    if len(frames) != 25:
        raise RuntimeError(f"Expected 25 common-bank worker CSVs, found {len(frames)}")
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(args.output / "common_bank_raw.csv", index=False)
    per_seed = (
        raw.groupby(["protagonist_method", "protagonist_seed", "protagonist_env_steps"], as_index=False)
        .robust_return.mean()
        .rename(columns={"robust_return": "common_bank_return"})
    )
    per_seed.to_csv(args.output / "common_bank_per_seed.csv", index=False)
    mean = (
        per_seed.groupby(["protagonist_method", "protagonist_env_steps"], as_index=False)
        .common_bank_return.agg(["mean", "sem"])
        .reset_index()
    )
    mean.to_csv(args.output / "common_bank_mean_convergence.csv", index=False)

    qpg = per_seed.loc[per_seed.protagonist_method == "qpg"]
    nog = per_seed.loc[per_seed.protagonist_method == "nog"]
    paired = qpg.merge(nog, on=["protagonist_seed", "protagonist_env_steps"], suffixes=("_qpg", "_nog"), validate="one_to_one")
    paired["gain"] = paired.common_bank_return_qpg - paired.common_bank_return_nog
    paired.to_csv(args.output / "common_bank_paired_gains.csv", index=False)
    endpoints = paired.sort_values("protagonist_env_steps").groupby("protagonist_seed", as_index=False).tail(1)
    endpoint_gain = endpoints.gain.to_numpy(float)
    auc_gain = []
    for seed in range(5):
        d = paired.loc[paired.protagonist_seed == seed]
        auc_gain.append(auc(d.rename(columns={"common_bank_return_qpg": "value"}), "value") - auc(d.rename(columns={"common_bank_return_nog": "value"}), "value"))
    auc_gain = np.asarray(auc_gain, dtype=float)
    late = paired.sort_values("protagonist_env_steps").groupby("protagonist_seed", as_index=False).tail(5)
    bank_summary = {
        "attacker_bank": "five final noG adversaries, seeds 0-4",
        "episodes_per_pairing": int(raw.episodes.iloc[0]),
        "endpoint_mean_gain": float(endpoint_gain.mean()),
        "endpoint_bootstrap_95ci": bootstrap_ci(endpoint_gain, 20260723),
        "endpoint_wins": int(np.sum(endpoint_gain > 0.0)),
        "auc_mean_gain": float(auc_gain.mean()),
        "auc_bootstrap_95ci": bootstrap_ci(auc_gain, 20260724),
        "auc_wins": int(np.sum(auc_gain > 0.0)),
        "late_positive_checkpoint_fraction": float(np.mean(late.gain > 0.0)),
    }
    bank_summary["late_plateau"] = {
        method: plateau_metrics(
            mean.loc[mean.protagonist_method == method].sort_values("protagonist_env_steps").tail(5)["mean"]
        )
        for method in METHODS
    }
    bank_summary["all_method_convergence_gate"] = bool(
        all(item["stable"] for item in bank_summary["late_plateau"].values())
    )
    bank_summary["common_bank_gate"] = bool(
        bank_summary["endpoint_mean_gain"] > 0.0
        and bank_summary["endpoint_wins"] >= 3
        and bank_summary["auc_mean_gain"] > 0.0
        and bank_summary["auc_wins"] >= 3
        and bank_summary["late_positive_checkpoint_fraction"] >= 0.80
        and bank_summary["all_method_convergence_gate"]
    )
    full_summary = json.loads((args.full_aggregate / "full_suite_summary.json").read_text(encoding="utf-8"))
    combined = {
        "full_suite": full_summary,
        "common_bank": bank_summary,
        "final_decision": "POSITIVE" if full_summary["decision_before_common_bank"] == "POSITIVE" and bank_summary["common_bank_gate"] else "NOT_POSITIVE",
    }
    (args.output / "combined_decision.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")

    full_mean = pd.read_csv(args.full_aggregate / "mean_convergence.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for method in METHODS:
        d = full_mean.loc[full_mean.method == method].sort_values("protagonist_env_steps")
        x = d.protagonist_env_steps.to_numpy(float) / 1000.0
        for ax, mean_col, sem_col in (
            (axes[0, 0], "train_mean", "train_sem"),
            (axes[0, 1], "robust_mean", "robust_sem"),
            (axes[1, 0], "clean_mean", "clean_sem"),
        ):
            y = d[mean_col].to_numpy(float)
            se = d[sem_col].fillna(0.0).to_numpy(float)
            ax.plot(x, y, color=COLORS[method], label=LABELS[method], linewidth=2)
            ax.fill_between(x, y - se, y + se, color=COLORS[method], alpha=0.12)
        b = mean.loc[mean.protagonist_method == method].sort_values("protagonist_env_steps")
        bx = b.protagonist_env_steps.to_numpy(float) / 1000.0
        by = b["mean"].to_numpy(float)
        bse = b["sem"].fillna(0.0).to_numpy(float)
        axes[1, 1].plot(bx, by, color=COLORS[method], label=LABELS[method], linewidth=2)
        axes[1, 1].fill_between(bx, by - bse, by + bse, color=COLORS[method], alpha=0.12)
    axes[0, 0].set_title("Training episode return")
    axes[0, 1].set_title("Frozen co-trained-pair robust evaluation")
    axes[1, 0].set_title("Frozen clean evaluation")
    axes[1, 1].set_title("Frozen common noG-adversary-bank evaluation")
    for ax in axes.flat:
        ax.set_xlabel("Protagonist environment steps (thousands)")
        ax.set_ylabel("Undiscounted native task return")
        ax.grid(alpha=0.2)
        ax.legend(ncol=2, fontsize=9)
    fig.suptitle("Ant-v4 native-reward exact joint actor RARL: five joint-training seeds")
    fig.savefig(args.output / "ant_bilinear_full_common_bank_convergence.png", dpi=220)
    fig.savefig(args.output / "ant_bilinear_full_common_bank_convergence.pdf")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
