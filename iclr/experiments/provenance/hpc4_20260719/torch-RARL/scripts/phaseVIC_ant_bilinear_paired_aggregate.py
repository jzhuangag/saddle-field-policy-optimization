from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"nog": "#2878B5", "qpg": "#C53D32"}
LABELS = {"nog": "noG", "qpg": "QP+G"}


def auc(frame: pd.DataFrame, column: str) -> float:
    ordered = frame.sort_values("protagonist_env_steps")
    return float(np.trapz(ordered[column].to_numpy(float), ordered["protagonist_env_steps"].to_numpy(float)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    convergence: dict[str, pd.DataFrame] = {}
    updates: dict[str, pd.DataFrame] = {}
    for method in ("nog", "qpg"):
        convergence[method] = pd.read_csv(args.root / method / "convergence.csv").sort_values("protagonist_env_steps")
        updates[method] = pd.read_csv(args.root / method / "updates.csv")
    merged = convergence["qpg"].merge(
        convergence["nog"], on="protagonist_env_steps", suffixes=("_qpg", "_nog"), validate="one_to_one"
    )
    merged["paired_robust_gain"] = merged["own_adversary_robust_return_qpg"] - merged["own_adversary_robust_return_nog"]
    merged["paired_clean_gain"] = merged["clean_return_qpg"] - merged["clean_return_nog"]
    merged.to_csv(args.output / "paired_checkpoint_gains.csv", index=False)

    qpg_updates = updates["qpg"]
    late = merged.tail(5)
    qpg_final = convergence["qpg"].iloc[-1]
    nog_final = convergence["nog"].iloc[-1]
    robust_auc_gain = auc(convergence["qpg"], "own_adversary_robust_return") - auc(convergence["nog"], "own_adversary_robust_return")
    clean_auc_gain = auc(convergence["qpg"], "clean_return") - auc(convergence["nog"], "clean_return")
    summary = {
        "env": "Ant-v4",
        "reward": "native Gymnasium undiscounted task return",
        "force_max": 0.5,
        "critic": "state-conditioned linear/bilinear game critic, own-action quadratic scale 0",
        "qpg_final_robust": float(qpg_final["own_adversary_robust_return"]),
        "nog_final_robust": float(nog_final["own_adversary_robust_return"]),
        "final_robust_gain": float(late.iloc[-1]["paired_robust_gain"]),
        "robust_auc_gain": robust_auc_gain,
        "late_positive_fraction": float(np.mean(late["paired_robust_gain"] > 0.0)),
        "qpg_final_clean": float(qpg_final["clean_return"]),
        "nog_final_clean": float(nog_final["clean_return"]),
        "final_clean_gain": float(late.iloc[-1]["paired_clean_gain"]),
        "clean_auc_gain": clean_auc_gain,
        "clean_noninferior_10pct": bool(float(qpg_final["clean_return"]) >= 0.9 * float(nog_final["clean_return"])),
        "gamma_active_fraction": float(np.mean(pd.to_numeric(qpg_updates["gamma_active"], errors="coerce") > 0.0)),
        "curvature_reliability_fraction": float(pd.to_numeric(qpg_updates["curvature_reliability_gate"], errors="coerce").mean()),
        "predicted_inclusion_pass_fraction": float(pd.to_numeric(qpg_updates["predicted_inclusion_pass"], errors="coerce").mean()),
        "realized_nog_safeguard_pass_fraction": float(pd.to_numeric(qpg_updates["realized_qp_le_nog"], errors="coerce").mean()),
    }
    summary["performance_gate"] = bool(
        summary["final_robust_gain"] > 0.0
        and summary["robust_auc_gain"] > 0.0
        and summary["late_positive_fraction"] >= 0.60
        and summary["clean_noninferior_10pct"]
    )
    summary["mechanism_gate"] = bool(
        summary["gamma_active_fraction"] >= 0.10
        and summary["curvature_reliability_fraction"] >= 0.60
        and summary["predicted_inclusion_pass_fraction"] == 1.0
        and summary["realized_nog_safeguard_pass_fraction"] == 1.0
    )
    summary["decision"] = "EXPAND_TO_PAIRED_SEEDS" if summary["performance_gate"] and summary["mechanism_gate"] else "DO_NOT_EXPAND"
    (args.output / "paired_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for method in ("nog", "qpg"):
        c = convergence[method]
        x = c["protagonist_env_steps"] / 1000.0
        axes[0, 0].plot(x, c["train_episode_return"], label=LABELS[method], color=COLORS[method], linewidth=2)
        axes[0, 1].plot(x, c["own_adversary_robust_return"], label=LABELS[method], color=COLORS[method], linewidth=2)
        axes[1, 0].plot(x, c["clean_return"], label=LABELS[method], color=COLORS[method], linewidth=2)
    x = merged["protagonist_env_steps"] / 1000.0
    axes[1, 1].axhline(0.0, color="black", linewidth=1)
    axes[1, 1].plot(x, merged["paired_robust_gain"], color=COLORS["qpg"], linewidth=2)
    axes[1, 1].scatter([x.iloc[-1]], [merged["paired_robust_gain"].iloc[-1]], color=COLORS["qpg"], zorder=3)
    axes[0, 0].set_title("Training episode return")
    axes[0, 1].set_title("Frozen co-trained-pair robust evaluation")
    axes[1, 0].set_title("Frozen clean evaluation")
    axes[1, 1].set_title("Paired robust gain (QP+G - noG)")
    for ax in axes.flat:
        ax.set_xlabel("Protagonist environment steps (thousands)")
        ax.set_ylabel("Undiscounted native task return")
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[1, 0].legend()
    fig.suptitle("Ant-v4 native-reward exact joint actor RARL gate")
    fig.savefig(args.output / "ant_bilinear_paired_convergence.png", dpi=220)
    fig.savefig(args.output / "ant_bilinear_paired_convergence.pdf")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
