from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ("sgd", "egm", "ppm")
LABELS = {"sgd": "SGD/GDA", "egm": "EGM", "ppm": "PPM (3 inner)"}
COLORS = {"sgd": "#4D4D4D", "egm": "#E69F00", "ppm": "#009E73"}


def plateau(values: pd.Series) -> dict[str, float | bool]:
    y = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if y.size < 5:
        return {"cv": float("inf"), "range": float("inf"), "trend": float("inf"), "stable": False}
    scale = max(abs(float(np.mean(y))), 1.0)
    cv = float(np.std(y, ddof=1) / scale)
    relative_range = float((np.max(y) - np.min(y)) / scale)
    trend = float(abs(np.polyfit(np.arange(y.size), y, 1)[0]) * (y.size - 1) / scale)
    return {"cv": cv, "range": relative_range, "trend": trend,
            "stable": bool(cv <= 0.15 and relative_range <= 0.40 and trend <= 0.20)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--min-rise", type=float, required=True)
    parser.add_argument("--max-late-drop", type=float, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    trains, cleans, robusts, summaries = [], [], [], []
    for method in METHODS:
        task = args.root / method
        t = pd.read_csv(task / "short_training_return_all_methods.csv")
        c = pd.read_csv(task / "short_clean_eval_all_methods.csv")
        r = pd.read_csv(task / "short_adv_eval_force_all_methods.csv")
        s = pd.read_csv(task / "stage7e_short_summary.csv")
        for frame in (t, c, r, s):
            frame["method"] = method
        trains.append(t); cleans.append(c); robusts.append(r); summaries.append(s)
    train = pd.concat(trains, ignore_index=True)
    clean = pd.concat(cleans, ignore_index=True)
    robust = pd.concat(robusts, ignore_index=True)
    summary_frame = pd.concat(summaries, ignore_index=True)
    train.to_csv(args.output / "training.csv", index=False)
    clean.to_csv(args.output / "clean_evaluation.csv", index=False)
    robust.to_csv(args.output / "robust_evaluation.csv", index=False)

    audit = {}
    for method in METHODS:
        c = clean.loc[clean.method == method].sort_values("timesteps")
        r = robust.loc[robust.method == method].sort_values("timesteps")
        c_early, c_late = float(c.head(3).mean_reward.mean()), float(c.tail(5).mean_reward.mean())
        r_early, r_late = float(r.head(3).mean_reward.mean()), float(r.tail(5).mean_reward.mean())
        c_best, r_best = float(c.mean_reward.max()), float(r.mean_reward.max())
        c_plateau, r_plateau = plateau(c.tail(5).mean_reward), plateau(r.tail(5).mean_reward)
        method_pass = bool(
            c_late - c_early >= args.min_rise and r_late - r_early >= args.min_rise
            and c_late >= c_best - args.max_late_drop and r_late >= r_best - args.max_late_drop
            and c_plateau["stable"] and r_plateau["stable"]
        )
        audit[method] = {
            "clean_early": c_early, "clean_late": c_late, "clean_rise": c_late - c_early,
            "robust_early": r_early, "robust_late": r_late, "robust_rise": r_late - r_early,
            "clean_plateau": c_plateau, "robust_plateau": r_plateau, "pass": method_pass,
        }
    protocol_ok = bool(
        (summary_frame.N_mu == 5).all() and (summary_frame.N_nu == 1).all()
        and (summary_frame.protagonist_optimizer == summary_frame.method).all()
        and (summary_frame.adversary_optimizer == summary_frame.method).all()
        and (summary_frame.crash_flag == 0).all() and (summary_frame.nan_flag == 0).all()
    )
    decision = {"env": args.env, "protocol_gate": protocol_ok, "methods": audit,
                "decision": "PASS" if protocol_ok and all(v["pass"] for v in audit.values()) else "FAIL"}
    (args.output / "baseline_gate_summary.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for method in METHODS:
        t = train.loc[train.method == method].sort_values("cumulative_timesteps")
        rolling = t.episode_return.rolling(50, min_periods=10).mean()
        axes[0].plot(t.cumulative_timesteps, t.episode_return, color=COLORS[method], alpha=0.08)
        axes[0].plot(t.cumulative_timesteps, rolling, color=COLORS[method], label=LABELS[method], linewidth=2)
        for ax, frame in ((axes[1], clean), (axes[2], robust)):
            d = frame.loc[frame.method == method].sort_values("timesteps")
            x, y, sd = d.timesteps.to_numpy(float), d.mean_reward.to_numpy(float), d.std_reward.to_numpy(float)
            ax.plot(x, y, color=COLORS[method], label=LABELS[method], linewidth=2)
            ax.fill_between(x, y - sd, y + sd, color=COLORS[method], alpha=0.10)
    axes[0].set_title("Protagonist-phase training return")
    axes[1].set_title("Frozen clean evaluation")
    axes[2].set_title("Frozen learned-adversary evaluation")
    for ax in axes:
        ax.set_xlabel("Protagonist environment steps")
        ax.set_ylabel("Undiscounted episode return")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle(f"{args.env}: 5:1 alternating PPO-RARL baselines")
    fig.savefig(args.output / "baseline_convergence.png", dpi=220)
    fig.savefig(args.output / "baseline_convergence.pdf")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
