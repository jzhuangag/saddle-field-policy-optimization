from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCHEDULES = ["alternating", "synchronized", "synchronized_warmup"]
SCHEDULE_LABELS = {
    "alternating": "Alternating 5:1 (no warm-up)",
    "synchronized": "Synchronized-lagged 1:1 (no warm-up)",
    "synchronized_warmup": "Synchronized-lagged 1:1 (50k clean warm-up)",
}
METHODS = ["proposed_noG", "proposed_qp"]
METHOD_LABELS = {"proposed_noG": "noG", "proposed_qp": "QP+G"}
COLORS = {"proposed_noG": "#2878B5", "proposed_qp": "#C43C39"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Aggregate the VI-C update-schedule gate")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--eval-window", type=int, default=3)
    return parser.parse_args()


def read_gate(root: Path, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_frames = []
    train_frames = []
    missing = []
    for schedule in SCHEDULES:
        for seed in seeds:
            for method in METHODS:
                task = root / schedule / f"seed_{seed}" / method
                for evaluation, filename in (
                    ("robust", "short_adv_eval_force_all_methods.csv"),
                    ("clean", "short_clean_eval_all_methods.csv"),
                ):
                    path = task / filename
                    if not path.exists():
                        missing.append(str(path))
                        continue
                    frame = pd.read_csv(path)
                    frame["schedule"] = schedule
                    frame["seed"] = seed
                    frame["evaluation"] = evaluation
                    eval_frames.append(frame)
                path = task / "short_training_return_all_methods.csv"
                if not path.exists():
                    missing.append(str(path))
                    continue
                frame = pd.read_csv(path)
                frame["schedule"] = schedule
                frame["seed"] = seed
                train_frames.append(frame)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} artifacts; first: {missing[0]}")
    return pd.concat(eval_frames, ignore_index=True), pd.concat(train_frames, ignore_index=True)


def smooth_evaluations(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    frame = frame.sort_values(["schedule", "evaluation", "method", "seed", "timesteps"]).copy()
    frame["return_ma"] = frame.groupby(
        ["schedule", "evaluation", "method", "seed"], sort=False
    )["mean_reward"].transform(lambda x: x.rolling(window, min_periods=1).mean())
    return frame


def smooth_training(frame: pd.DataFrame, bin_width: int = 10_000) -> pd.DataFrame:
    frame = frame.copy()
    frame["timesteps"] = (np.floor(frame["cumulative_timesteps"] / bin_width) * bin_width).astype(int)
    frame = frame.groupby(
        ["schedule", "method", "seed", "timesteps"], as_index=False
    ).agg(mean_reward=("episode_return", "mean"))
    frame["return_ma"] = frame.groupby(
        ["schedule", "method", "seed"], sort=False
    )["mean_reward"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    return frame


def mean_sem(axis, frame: pd.DataFrame, *, legend: bool = False) -> None:
    for method in METHODS:
        rows = frame[frame["method"] == method]
        stats = rows.groupby("timesteps")["return_ma"].agg(["mean", "sem"]).reset_index()
        x = stats["timesteps"].to_numpy(dtype=float) / 1000.0
        mean = stats["mean"].to_numpy(dtype=float)
        sem = stats["sem"].fillna(0.0).to_numpy(dtype=float)
        axis.plot(x, mean, color=COLORS[method], linewidth=2, label=METHOD_LABELS[method])
        axis.fill_between(x, mean - sem, mean + sem, color=COLORS[method], alpha=0.13, linewidth=0)
    axis.grid(alpha=0.2)
    if legend:
        axis.legend(frameon=False)


def plot_evaluation(eval_frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14.2, 11.2), sharex="col")
    for row, schedule in enumerate(SCHEDULES):
        rows = eval_frame[eval_frame["schedule"] == schedule]
        robust = rows[rows["evaluation"] == "robust"]
        clean = rows[rows["evaluation"] == "clean"]
        mean_sem(axes[row, 0], robust, legend=row == 0)
        mean_sem(axes[row, 1], clean)

        paired = robust.pivot(index=["seed", "timesteps"], columns="method", values="return_ma").reset_index()
        paired["gain"] = paired["proposed_qp"] - paired["proposed_noG"]
        stats = paired.groupby("timesteps")["gain"].agg(["mean", "sem"]).reset_index()
        x = stats["timesteps"].to_numpy(dtype=float) / 1000.0
        mean = stats["mean"].to_numpy(dtype=float)
        sem = stats["sem"].fillna(0.0).to_numpy(dtype=float)
        axes[row, 2].axhline(0, color="#333333", linewidth=1)
        axes[row, 2].plot(x, mean, color=COLORS["proposed_qp"], linewidth=2)
        axes[row, 2].fill_between(x, mean - sem, mean + sem, color=COLORS["proposed_qp"], alpha=0.15)
        axes[row, 2].annotate(f"final {mean[-1]:+.1f}", (x[-1], mean[-1]), xytext=(-55, 10), textcoords="offset points")
        axes[row, 2].grid(alpha=0.2)
        axes[row, 0].set_ylabel(f"{SCHEDULE_LABELS[schedule]}\nreturn")

    for col, title in enumerate(("Frozen-pair learned-adversary eval", "Frozen clean eval", "Paired robust gain (QP+G - noG)")):
        axes[0, col].set_title(title)
        axes[-1, col].set_xlabel("Protagonist environment steps (thousands)")
    fig.suptitle("HalfCheetah-v4 PPO-RARL update-schedule gate", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output / "joint_gate_evaluation_convergence.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "joint_gate_evaluation_convergence.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_training(train_frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), sharey=True)
    for axis, schedule in zip(axes, SCHEDULES):
        mean_sem(axis, train_frame[train_frame["schedule"] == schedule], legend=schedule == SCHEDULES[0])
        axis.set_title(SCHEDULE_LABELS[schedule])
        axis.set_xlabel("Total collection environment steps (thousands)")
    axes[0].set_ylabel("3-bin moving-average episode return")
    fig.suptitle("Training collection convergence (both player phases)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output / "joint_gate_training_convergence.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "joint_gate_training_convergence.pdf", bbox_inches="tight")
    plt.close(fig)


def summarize(eval_frame: pd.DataFrame, window: int) -> list[dict[str, object]]:
    out = []
    for schedule in SCHEDULES:
        robust = eval_frame[
            (eval_frame["schedule"] == schedule) & (eval_frame["evaluation"] == "robust")
        ].pivot(index=["seed", "timesteps"], columns="method", values="mean_reward").reset_index()
        per_seed = robust.groupby("seed").apply(
            lambda x: float((x["proposed_qp"] - x["proposed_noG"]).tail(window).mean()),
            include_groups=False,
        )
        out.append({
            "schedule": schedule,
            "label": SCHEDULE_LABELS[schedule],
            "paired_robust_gain_mean": float(per_seed.mean()),
            "paired_robust_gain_std": float(per_seed.std(ddof=1)),
            "wins": int((per_seed > 0).sum()),
            "seeds": int(per_seed.size),
            "per_seed_gains": {str(k): float(v) for k, v in per_seed.items()},
        })
    return out


def main() -> None:
    args = parse_args()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    evaluations, training = read_gate(Path(args.input_root), args.seeds)
    evaluations = smooth_evaluations(evaluations, args.eval_window)
    training = smooth_training(training)
    evaluations.to_csv(output / "joint_gate_checkpoint_evaluations.csv", index=False)
    training.to_csv(output / "joint_gate_training_curves.csv", index=False)
    (output / "joint_gate_summary.json").write_text(
        json.dumps(summarize(evaluations, args.eval_window), indent=2), encoding="utf-8"
    )
    plot_evaluation(evaluations, output)
    plot_training(training, output)


if __name__ == "__main__":
    main()
