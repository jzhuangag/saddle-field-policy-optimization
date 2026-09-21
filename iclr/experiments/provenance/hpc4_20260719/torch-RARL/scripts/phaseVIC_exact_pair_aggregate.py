from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Aggregate an exact noG/QP+G paired gate")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def plateau(values: np.ndarray) -> dict[str, float | bool]:
    values = np.asarray(values, dtype=float)[-5:]
    median = float(np.median(values))
    mean = float(np.mean(values))
    scale = max(abs(median), 1.0)
    slope = float(np.polyfit(np.arange(values.size), values, 1)[0])
    stats = {
        "mean": mean,
        "cv": float(np.std(values) / max(abs(mean), 1.0)),
        "min_over_median": float(np.min(values) / median) if median > 0 else float("nan"),
        "relative_range": float((np.max(values) - np.min(values)) / scale),
        "normalized_trend": float(abs(slope) * 4.0 / scale),
    }
    stats["pass"] = bool(
        stats["cv"] <= 0.15
        and stats["min_over_median"] >= 0.8
        and stats["relative_range"] <= 0.4
        and stats["normalized_trend"] <= 0.2
    )
    return stats


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    curves = {method: pd.read_csv(args.root / method / "convergence.csv") for method in ("nog", "qpg")}
    updates = {method: pd.read_csv(args.root / method / "updates.csv") for method in ("nog", "qpg")}
    labels = {"nog": "noG", "qpg": "QP+G"}
    colors = {"nog": "#2878B5", "qpg": "#C43C39"}

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)
    metrics = (
        ("train_episode_return", "Training episode return"),
        ("own_adversary_robust_return", "Co-trained robust evaluation"),
        ("clean_return", "Clean evaluation"),
    )
    for axis, (column, title) in zip(axes.flat[:3], metrics):
        for method, frame in curves.items():
            axis.plot(frame["protagonist_env_steps"] / 1000.0, frame[column], label=labels[method], color=colors[method])
        axis.set_title(title)
        axis.set_xlabel("Protagonist environment steps (thousands)")
        axis.set_ylabel("Native task return")
        axis.grid(alpha=0.25)
    merged = curves["qpg"][["protagonist_env_steps", "own_adversary_robust_return"]].merge(
        curves["nog"][["protagonist_env_steps", "own_adversary_robust_return"]],
        on="protagonist_env_steps", suffixes=("_qpg", "_nog")
    )
    gain = merged["own_adversary_robust_return_qpg"] - merged["own_adversary_robust_return_nog"]
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].plot(merged["protagonist_env_steps"] / 1000.0, gain, color=colors["qpg"])
    axes[1, 1].set_title("Paired robust gain (QP+G - noG)")
    axes[1, 1].set_xlabel("Protagonist environment steps (thousands)")
    axes[1, 1].set_ylabel("Return difference")
    axes[1, 1].grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.savefig(args.output / "exact_pair_convergence.png", dpi=180)
    fig.savefig(args.output / "exact_pair_convergence.pdf")
    plt.close(fig)

    summary: dict[str, object] = {}
    for method in ("nog", "qpg"):
        frame = curves[method]
        upd = updates[method]
        summary[method] = {
            "training_plateau": plateau(frame["train_episode_return"].to_numpy()),
            "clean_plateau": plateau(frame["clean_return"].to_numpy()),
            "robust_plateau": plateau(frame["own_adversary_robust_return"].to_numpy()),
            "clean_auc": float(np.trapezoid(frame["clean_return"], frame["protagonist_env_steps"])),
            "robust_auc": float(np.trapezoid(frame["own_adversary_robust_return"], frame["protagonist_env_steps"])),
            "late_corr_Q_MC": float(frame["corr_Q_MC"].tail(5).mean()),
            "late_WF_over_SF": float(frame["WF_over_SF"].tail(5).mean()),
            "late_cross_to_same": float(frame["cross_to_same_ratio"].tail(5).mean()),
            "predicted_inclusion_fraction": float(upd["predicted_inclusion_pass"].mean()),
            "realized_safeguard_fraction": float(upd["realized_qp_le_nog"].mean()),
            "gamma_active_fraction": float(upd["gamma_active"].mean()),
            "fallback_fraction": float(upd["fallback"].mean()),
            "mean_G_over_update": float(upd["G_over_update_norm"].mean()),
        }
    summary["paired"] = {
        "late_clean_gain": float(curves["qpg"]["clean_return"].tail(5).mean() - curves["nog"]["clean_return"].tail(5).mean()),
        "late_robust_gain": float(gain.tail(5).mean()),
        "robust_gain_auc": float(np.trapezoid(gain, merged["protagonist_env_steps"])),
        "late_positive_fraction": float((gain.tail(5) > 0.0).mean()),
    }
    (args.output / "exact_pair_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
