from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ["sgd", "egm", "ppm", "proposed_noG", "proposed_qp"]
LABELS = {"sgd": "GDA", "egm": "EGM", "ppm": "PPM", "proposed_noG": "noG", "proposed_qp": "QP+G"}
COLORS = {"sgd": "#4D4D4D", "egm": "#E69F00", "ppm": "#009E73", "proposed_noG": "#2878B5", "proposed_qp": "#C43C39"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Aggregate standard RARL evaluation")
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--convergence-csv", required=True)
    parser.add_argument("--training-csv", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def mean_sem_curve(axis, frame: pd.DataFrame, evaluation: str) -> None:
    rows = frame[frame["evaluation"] == evaluation]
    for method in METHODS:
        method_rows = rows[rows["method"] == method]
        stats = method_rows.groupby("timesteps")["return_ma"].agg(["mean", "sem"]).reset_index()
        x = stats["timesteps"].to_numpy(float) / 1000.0
        mean = stats["mean"].to_numpy(float)
        sem = stats["sem"].fillna(0).to_numpy(float)
        axis.plot(x, mean, color=COLORS[method], label=LABELS[method], linewidth=2)
        axis.fill_between(x, mean - sem, mean + sem, color=COLORS[method], alpha=0.1)
    axis.grid(alpha=0.2)
    axis.set_xlabel("Protagonist environment steps (thousands)")


def main() -> None:
    args = parse_args()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(args.eval_root).glob("seed_*.csv"))
    if len(paths) != 25:
        raise FileNotFoundError(f"Expected 25 standard-evaluation files, found {len(paths)}")
    standard = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    convergence = pd.read_csv(args.convergence_csv)
    training = pd.read_csv(args.training_csv)
    standard.to_csv(output / "standard_rarl_all_episodes.csv", index=False)

    common = standard[standard["evaluation"] == "common_attacker_bank"]
    common_seed = common.groupby(["target_method", "target_seed"], as_index=False).agg(
        common_bank_mean=("return", "mean"),
        common_bank_std=("return", "std"),
    )
    common_summary = common_seed.groupby("target_method", as_index=False).agg(
        mean=("common_bank_mean", "mean"), sem=("common_bank_mean", "sem")
    )
    common_seed.to_csv(output / "common_attacker_bank_per_seed.csv", index=False)
    common_summary.to_csv(output / "common_attacker_bank_summary.csv", index=False)
    cross_play = common.groupby(["target_method", "attacker_method"], as_index=False)["return"].mean()
    cross_play.to_csv(output / "cross_play_method_matrix.csv", index=False)

    common_curve = standard[standard["evaluation"] == "common_attacker_convergence"].groupby(
        ["target_method", "target_seed", "checkpoint_steps"], as_index=False
    )["return"].mean()
    common_curve.to_csv(output / "common_attacker_convergence.csv", index=False)

    dynamics = standard[standard["evaluation"] == "dynamics_sweep"]
    dynamics_seed = dynamics.groupby(
        ["target_method", "target_seed", "parameter", "multiplier"], as_index=False
    )["return"].mean()
    dynamics_seed.to_csv(output / "dynamics_sweep_per_seed.csv", index=False)

    # Main paper figure: one training channel and three frozen evaluation channels.
    main_fig, main_axes = plt.subplots(2, 2, figsize=(12.8, 8.1))
    for method in METHODS:
        rows = training[training["method"] == method]
        stats = rows.groupby("timesteps")["return_ma"].agg(["mean", "sem"]).reset_index()
        xx = stats["timesteps"].to_numpy(float) / 1000.0
        yy = stats["mean"].to_numpy(float)
        ee = stats["sem"].fillna(0).to_numpy(float)
        main_axes[0, 0].plot(xx, yy, color=COLORS[method], label=LABELS[method], linewidth=2)
        main_axes[0, 0].fill_between(xx, yy - ee, yy + ee, color=COLORS[method], alpha=0.1)
    main_axes[0, 0].set_title("Protagonist training return")
    main_axes[0, 0].set_ylabel("Episode return")
    main_axes[0, 0].legend(frameon=False, ncol=2)
    main_axes[0, 0].grid(alpha=0.2)

    mean_sem_curve(main_axes[0, 1], convergence, "robust")
    main_axes[0, 1].set_title("Co-trained learned-adversary evaluation")
    mean_sem_curve(main_axes[1, 0], convergence, "clean")
    main_axes[1, 0].set_title("Clean evaluation")
    main_axes[1, 0].set_ylabel("Average episodic return")

    for method in METHODS:
        rows = common_curve[common_curve["target_method"] == method]
        stats = rows.groupby("checkpoint_steps")["return"].agg(["mean", "sem"]).reset_index()
        xx = stats["checkpoint_steps"].to_numpy(float) / 1000.0
        yy = stats["mean"].to_numpy(float)
        ee = stats["sem"].fillna(0).to_numpy(float)
        main_axes[1, 1].plot(xx, yy, color=COLORS[method], label=LABELS[method], linewidth=2)
        main_axes[1, 1].fill_between(xx, yy - ee, yy + ee, color=COLORS[method], alpha=0.1)
    main_axes[1, 1].set_title("Fixed common-adversary bank evaluation")
    main_axes[1, 1].grid(alpha=0.2)

    for axis in main_axes.flat:
        axis.set_xlabel("Protagonist environment steps (thousands)")
    main_fig.suptitle("HalfCheetah-v4 synchronized-lagged PPO-RARL", fontsize=14)
    main_fig.tight_layout(rect=[0, 0, 1, 0.97])
    main_fig.savefig(output / "paper_main_four_panel_convergence.png", dpi=240, bbox_inches="tight")
    main_fig.savefig(output / "paper_main_four_panel_convergence.pdf", bbox_inches="tight")
    plt.close(main_fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.4))
    mean_sem_curve(axes[0, 0], convergence, "robust")
    axes[0, 0].set_title("Frozen learned-adversary convergence")
    axes[0, 0].set_ylabel("Return")
    axes[0, 0].legend(frameon=False, ncol=2)
    mean_sem_curve(axes[0, 1], convergence, "clean")
    axes[0, 1].set_title("Frozen clean convergence")

    x = np.arange(len(METHODS))
    ordered = common_summary.set_index("target_method").loc[METHODS]
    axes[0, 2].errorbar(x, ordered["mean"], yerr=ordered["sem"], fmt="o", capsize=4, color="#222222")
    axes[0, 2].set_xticks(x, [LABELS[m] for m in METHODS])
    axes[0, 2].set_title("Fixed common-adversary bank")
    axes[0, 2].set_ylabel("Mean episodic return")
    axes[0, 2].grid(axis="y", alpha=0.2)

    for axis, parameter, title in (
        (axes[1, 0], "body_mass", "Body-mass generalization"),
        (axes[1, 1], "geom_friction", "Contact-friction generalization"),
    ):
        rows = dynamics_seed[dynamics_seed["parameter"] == parameter]
        for method in METHODS:
            stats = rows[rows["target_method"] == method].groupby("multiplier")["return"].agg(["mean", "sem"]).reset_index()
            xx = stats["multiplier"].to_numpy(float)
            yy = stats["mean"].to_numpy(float)
            ee = stats["sem"].fillna(0).to_numpy(float)
            axis.plot(xx, yy, marker="o", color=COLORS[method], label=LABELS[method])
            axis.fill_between(xx, yy - ee, yy + ee, color=COLORS[method], alpha=0.1)
        axis.axvline(1.0, color="#555555", linewidth=1, linestyle="--")
        axis.set_title(title)
        axis.set_xlabel("Dynamics multiplier")
        axis.set_ylabel("Clean episodic return")
        axis.grid(alpha=0.2)

    robust = convergence[convergence["evaluation"] == "robust"]
    paired = robust[robust["method"].isin(["proposed_noG", "proposed_qp"])].pivot(
        index=["seed", "timesteps"], columns="method", values="return_ma"
    ).reset_index()
    paired["gain"] = paired["proposed_qp"] - paired["proposed_noG"]
    stats = paired.groupby("timesteps")["gain"].agg(["mean", "sem"]).reset_index()
    xx = stats["timesteps"].to_numpy(float) / 1000.0
    yy = stats["mean"].to_numpy(float)
    ee = stats["sem"].fillna(0).to_numpy(float)
    axes[1, 2].axhline(0, color="#222222", linewidth=1)
    axes[1, 2].plot(xx, yy, color=COLORS["proposed_qp"], linewidth=2)
    axes[1, 2].fill_between(xx, yy - ee, yy + ee, color=COLORS["proposed_qp"], alpha=0.15)
    axes[1, 2].set_title("Paired robust convergence gain")
    axes[1, 2].set_xlabel("Protagonist environment steps (thousands)")
    axes[1, 2].set_ylabel("QP+G minus noG return")
    axes[1, 2].grid(alpha=0.2)

    fig.suptitle("HalfCheetah-v4 synchronized-lagged PPO-RARL: convergence and standard robustness", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output / "standard_rarl_convergence_and_robustness.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "standard_rarl_convergence_and_robustness.pdf", bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "common_attacker_bank": "all 25 final adversaries; every target protagonist faces the same bank",
        "dynamics": "all non-world body masses or all geom friction coefficients multiplied uniformly",
        "episodes_per_attacker_or_multiplier": 20,
        "evaluation_learning": False,
        "policy_actions": "deterministic",
    }
    (output / "standard_rarl_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
