from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Aggregate baseline-only standard RARL gates")
    parser.add_argument("--run", action="append", required=True, help="LABEL=RUN_DIR")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-window", type=int, default=20)
    return parser.parse_args()


def plateau(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)[-5:]
    median = float(np.median(values))
    mean = float(np.mean(values))
    scale = max(abs(median), 1.0)
    x = np.arange(values.size, dtype=np.float64)
    slope = float(np.polyfit(x, values, 1)[0]) if values.size > 1 else 0.0
    cv = float(np.std(values) / max(abs(mean), 1.0))
    min_over_median = float(np.min(values) / median) if median > 0.0 else float("nan")
    relative_range = float((np.max(values) - np.min(values)) / scale)
    normalized_trend = float(abs(slope) * max(values.size - 1, 1) / scale)
    passed = int(
        cv <= 0.15
        and (not np.isfinite(min_over_median) or min_over_median >= 0.8)
        and relative_range <= 0.4
        and normalized_trend <= 0.2
    )
    return {
        "late5_mean": mean,
        "late5_cv": cv,
        "late5_min_over_median": min_over_median,
        "late5_relative_range": relative_range,
        "late5_normalized_trend": normalized_trend,
        "plateau_pass": passed,
    }


def training_bins(frame: pd.DataFrame, eval_steps: np.ndarray) -> pd.DataFrame:
    steps = frame["cumulative_timesteps"].to_numpy(dtype=np.float64)
    returns = frame["episode_return"].to_numpy(dtype=np.float64)
    edges = np.concatenate(([0.0], np.asarray(eval_steps, dtype=np.float64)))
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (steps > lo) & (steps <= hi)
        if np.any(mask):
            rows.append({"timesteps": hi, "mean_reward": float(np.mean(returns[mask]))})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs: list[tuple[str, Path]] = []
    for spec in args.run:
        label, raw_path = spec.split("=", 1)
        runs.append((label, Path(raw_path)))

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    rows: list[dict[str, float | int | str]] = []
    for label, run_dir in runs:
        training = pd.read_csv(run_dir / "short_training_return_all_methods.csv")
        clean = pd.read_csv(run_dir / "short_clean_eval_all_methods.csv")
        robust = pd.read_csv(run_dir / "short_adv_eval_force_all_methods.csv")
        eval_steps = clean["timesteps"].to_numpy(dtype=np.float64)
        binned = training_bins(training, eval_steps)

        rolling = training["episode_return"].rolling(args.training_window, min_periods=1).mean()
        axes[0].plot(training["cumulative_timesteps"] / 1000.0, rolling, label=label)
        axes[1].plot(clean["timesteps"] / 1000.0, clean["mean_reward"], marker="o", ms=3, label=label)
        axes[2].plot(robust["timesteps"] / 1000.0, robust["mean_reward"], marker="o", ms=3, label=label)

        clean_late = float(clean["mean_reward"].tail(5).mean())
        robust_late = float(robust["mean_reward"].tail(5).mean())
        degradation = (clean_late - robust_late) / max(abs(clean_late), 1.0)
        for metric, values in (
            ("training_binned", binned["mean_reward"].to_numpy()),
            ("clean", clean["mean_reward"].to_numpy()),
            ("robust", robust["mean_reward"].to_numpy()),
        ):
            row: dict[str, float | int | str] = {"label": label, "metric": metric}
            row.update(plateau(values))
            row["clean_minus_robust_fraction"] = degradation
            rows.append(row)

    titles = (
        f"Protagonist training return ({args.training_window}-episode rolling mean)",
        "Frozen clean evaluation (raw checkpoints)",
        "Co-trained robust evaluation (raw checkpoints)",
    )
    for axis, title in zip(axes, titles):
        axis.set_title(title)
        axis.set_xlabel("Protagonist environment steps (thousands)")
        axis.set_ylabel("Native task return")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(args.output / "baseline_gate_convergence.png", dpi=180)
    fig.savefig(args.output / "baseline_gate_convergence.pdf")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(args.output / "baseline_gate_plateau.csv", index=False)


if __name__ == "__main__":
    main()
