from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ["sgd", "egm", "ppm", "proposed_noG", "proposed_qp"]
LABELS = {
    "sgd": "GDA",
    "egm": "EGM",
    "ppm": "PPM",
    "proposed_noG": "noG",
    "proposed_qp": "QP+G",
}
COLORS = {
    "sgd": "#4D4D4D",
    "egm": "#E69F00",
    "ppm": "#009E73",
    "proposed_noG": "#2878B5",
    "proposed_qp": "#C43C39",
}
EVAL_FILES = {
    "robust": "short_adv_eval_force_all_methods.csv",
    "clean": "short_clean_eval_all_methods.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Aggregate the full multi-environment VI-C PPO-RARL suite")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--envs", nargs="+", default=["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--training-bin", type=int, default=10_000)
    parser.add_argument("--eval-window", type=int, default=5)
    parser.add_argument(
        "--ppm-input-root",
        default=None,
        help="Optional alternate suite root containing corrected PPM runs.",
    )
    return parser.parse_args()


def task_root(root: Path, env_id: str, seed: int, method: str) -> Path:
    return root / env_id / f"seed_{seed}" / method


def read_protagonist_training_curve(task: Path, env_id: str, method: str) -> pd.DataFrame:
    candidates = sorted(task.glob(f"runs/{method}/saved_models/rarl-ppo/{env_id}/{env_id}_*/analysis/protagonist_episode_returns.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one phase-pure protagonist return file, found {len(candidates)}")
    frame = pd.read_csv(candidates[0]).rename(columns={"protagonist_timesteps": "cumulative_timesteps"})
    frame["method"] = method
    return frame[["method", "cumulative_timesteps", "episode_return"]]


def bootstrap_ci(values: np.ndarray, seed: int, draws: int = 20_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, values.size), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def read_suite(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(args.input_root)
    eval_frames = []
    training_frames = []
    summary_frames = []
    missing = []
    for env_id in args.envs:
        for seed in args.seeds:
            for method in args.methods:
                method_root = Path(args.ppm_input_root) if method == "ppm" and args.ppm_input_root else root
                run = task_root(method_root, env_id, seed, method)
                summary_path = run / "stage7e_short_summary.csv"
                paths = [summary_path] + [run / filename for filename in EVAL_FILES.values()]
                absent = [str(path) for path in paths if not path.exists()]
                if absent:
                    missing.extend(absent)
                    continue
                summary = pd.read_csv(summary_path)
                summary["env_id"] = env_id
                summary["seed"] = seed
                summary_frames.append(summary)
                training = read_protagonist_training_curve(run, env_id, method)
                training["env_id"] = env_id
                training["seed"] = seed
                training_frames.append(training)
                for evaluation, filename in EVAL_FILES.items():
                    frame = pd.read_csv(run / filename)
                    frame["env_id"] = env_id
                    frame["seed"] = seed
                    frame["evaluation"] = evaluation
                    eval_frames.append(frame)
    if missing:
        preview = "\n".join(missing[:12])
        raise FileNotFoundError(f"Missing {len(missing)} suite artifacts. First paths:\n{preview}")
    return (
        pd.concat(summary_frames, ignore_index=True),
        pd.concat(training_frames, ignore_index=True),
        pd.concat(eval_frames, ignore_index=True),
    )


def prepare_eval_curves(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    frame = frame.sort_values(["env_id", "evaluation", "seed", "method", "timesteps"]).reset_index(drop=True)
    frame["return_ma"] = frame.groupby(
        ["env_id", "evaluation", "seed", "method"], sort=False
    )["mean_reward"].transform(lambda values: values.rolling(window, min_periods=1).mean())
    return frame


def prepare_training_curves(frame: pd.DataFrame, bin_width: int) -> pd.DataFrame:
    frame = frame.copy()
    frame["timesteps"] = (
        np.floor(frame["cumulative_timesteps"].to_numpy(dtype=float) / bin_width) * bin_width
    ).astype(int)
    binned = frame.groupby(["env_id", "seed", "method", "timesteps"], as_index=False).agg(
        mean_reward=("episode_return", "mean"),
        episodes=("episode_return", "size"),
    )
    binned = binned.sort_values(["env_id", "seed", "method", "timesteps"]).reset_index(drop=True)
    binned["return_ma"] = binned.groupby(["env_id", "seed", "method"], sort=False)[
        "mean_reward"
    ].transform(lambda values: values.rolling(3, min_periods=1).mean())
    return binned


def plot_mean_sem(axis, frame: pd.DataFrame, methods: list[str]) -> None:
    for method in methods:
        rows = frame[frame["method"] == method]
        stats = rows.groupby("timesteps")["return_ma"].agg(["mean", "sem"]).reset_index()
        if stats.empty:
            continue
        x = stats["timesteps"].to_numpy(dtype=float) / 1000.0
        mean = stats["mean"].to_numpy(dtype=float)
        sem = stats["sem"].fillna(0.0).to_numpy(dtype=float)
        axis.plot(x, mean, color=COLORS[method], linewidth=2.0, label=LABELS[method])
        axis.fill_between(x, mean - sem, mean + sem, color=COLORS[method], alpha=0.10, linewidth=0)
    axis.grid(alpha=0.20, linewidth=0.7)
    axis.set_xlabel("Protagonist environment steps (thousands)")


def endpoint_rows(env_id: str, eval_curves: pd.DataFrame, methods: list[str], window: int) -> list[dict[str, object]]:
    rows = []
    for evaluation in EVAL_FILES:
        subset = eval_curves[(eval_curves["env_id"] == env_id) & (eval_curves["evaluation"] == evaluation)]
        for method in methods:
            method_rows = subset[subset["method"] == method]
            per_seed = method_rows.groupby("seed").apply(
                lambda values: float(values.sort_values("timesteps")["mean_reward"].tail(window).mean()),
                include_groups=False,
            )
            auc_seed = method_rows.groupby("seed").apply(
                lambda values: float(np.trapezoid(
                    values.sort_values("timesteps")["mean_reward"],
                    values.sort_values("timesteps")["timesteps"],
                )),
                include_groups=False,
            )
            mean_curve = method_rows.groupby("timesteps", as_index=False)["return_ma"].mean().sort_values("timesteps")
            late = mean_curve.tail(window)
            slope = float(np.polyfit(late["timesteps"] / 10_000.0, late["return_ma"], 1)[0]) if len(late) >= 2 else np.nan
            rows.append(
                {
                    "env_id": env_id,
                    "evaluation": evaluation,
                    "method": method,
                    "method_label": LABELS[method],
                    "seeds": int(per_seed.size),
                    "last_window_mean": float(per_seed.mean()),
                    "last_window_std": float(per_seed.std(ddof=1)),
                    "auc_mean": float(auc_seed.mean()),
                    "late_slope_return_per_10k": slope,
                }
            )
    return rows


def plot_environment(
    env_id: str,
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    methods: list[str],
    output_root: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.1))
    train = training[training["env_id"] == env_id]
    robust = evaluation[(evaluation["env_id"] == env_id) & (evaluation["evaluation"] == "robust")]
    clean = evaluation[(evaluation["env_id"] == env_id) & (evaluation["evaluation"] == "clean")]

    plot_mean_sem(axes[0, 0], train, methods)
    axes[0, 0].set_title("Protagonist training return")
    axes[0, 0].set_xlabel("Protagonist environment steps (thousands)")
    axes[0, 0].set_ylabel("3-bin moving-average episode return")
    axes[0, 0].legend(frameon=False, ncol=2)

    plot_mean_sem(axes[0, 1], robust, methods)
    axes[0, 1].set_title("Learned-adversary robust evaluation")
    axes[0, 1].set_ylabel("5-checkpoint moving-average return")

    plot_mean_sem(axes[1, 0], clean, methods)
    axes[1, 0].set_title("Clean evaluation")
    axes[1, 0].set_ylabel("5-checkpoint moving-average return")

    paired = robust[robust["method"].isin(["proposed_noG", "proposed_qp"])].pivot(
        index=["seed", "timesteps"], columns="method", values="return_ma"
    ).reset_index()
    paired["gain"] = paired["proposed_qp"] - paired["proposed_noG"]
    stats = paired.groupby("timesteps")["gain"].agg(["mean", "sem"]).reset_index()
    x = stats["timesteps"].to_numpy(dtype=float) / 1000.0
    mean = stats["mean"].to_numpy(dtype=float)
    sem = stats["sem"].fillna(0.0).to_numpy(dtype=float)
    axes[1, 1].axhline(0.0, color="#222222", linewidth=1.0)
    axes[1, 1].plot(x, mean, color=COLORS["proposed_qp"], linewidth=2.1)
    axes[1, 1].fill_between(x, mean - sem, mean + sem, color=COLORS["proposed_qp"], alpha=0.15, linewidth=0)
    axes[1, 1].scatter([x[-1]], [mean[-1]], color=COLORS["proposed_qp"], s=30, zorder=3)
    axes[1, 1].annotate(
        f"final {mean[-1]:+.2f}", xy=(x[-1], mean[-1]), xytext=(-48, 12),
        textcoords="offset points", fontsize=9, color=COLORS["proposed_qp"]
    )
    axes[1, 1].set_title("Paired robust gain")
    axes[1, 1].set_xlabel("Protagonist environment steps (thousands)")
    axes[1, 1].set_ylabel("QP+G minus noG return")
    axes[1, 1].grid(alpha=0.20, linewidth=0.7)

    fig.suptitle(f"{env_id}: PPO-RARL full optimizer comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    slug = env_id.replace("-", "_").lower()
    fig.savefig(output_root / f"{slug}_full_convergence.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_root / f"{slug}_full_convergence.pdf", bbox_inches="tight")
    plt.close(fig)


def paired_ablation(
    env_id: str, evaluation: pd.DataFrame, window: int
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    endpoint_values: dict[str, np.ndarray] = {}
    auc_values: dict[str, np.ndarray] = {}
    method_endpoints: dict[str, dict[str, list[float]]] = {}
    robust_late_positive_fraction = 0.0
    for eval_type in EVAL_FILES:
        subset = evaluation[
            (evaluation["env_id"] == env_id)
            & (evaluation["evaluation"] == eval_type)
            & (evaluation["method"].isin(["proposed_noG", "proposed_qp"]))
        ]
        pivot = subset.pivot(
            index=["seed", "timesteps"], columns="method", values="mean_reward"
        ).reset_index().sort_values(["seed", "timesteps"])
        gains = []
        auc_gains = []
        qp_endpoints = []
        nog_endpoints = []
        for seed, seed_rows in pivot.groupby("seed"):
            qp_endpoint = float(seed_rows["proposed_qp"].tail(window).mean())
            nog_endpoint = float(seed_rows["proposed_noG"].tail(window).mean())
            gain = qp_endpoint - nog_endpoint
            auc_gain = float(np.trapz(
                seed_rows["proposed_qp"] - seed_rows["proposed_noG"],
                seed_rows["timesteps"],
            ))
            gains.append(gain)
            auc_gains.append(auc_gain)
            qp_endpoints.append(qp_endpoint)
            nog_endpoints.append(nog_endpoint)
            rows.append(
                {
                    "env_id": env_id,
                    "evaluation": eval_type,
                    "seed": int(seed),
                    "qp_endpoint": qp_endpoint,
                    "nog_endpoint": nog_endpoint,
                    "endpoint_gain": gain,
                    "auc_gain": auc_gain,
                }
            )
        endpoint_values[eval_type] = np.asarray(gains, dtype=np.float64)
        auc_values[eval_type] = np.asarray(auc_gains, dtype=np.float64)
        method_endpoints[eval_type] = {"qp": qp_endpoints, "nog": nog_endpoints}
        if eval_type == "robust":
            mean_checkpoint_gain = pivot.groupby("timesteps").apply(
                lambda values: float((values["proposed_qp"] - values["proposed_noG"]).mean()),
                include_groups=False,
            )
            robust_late_positive_fraction = float((mean_checkpoint_gain.tail(window) > 0.0).mean())

    robust = endpoint_values["robust"]
    clean = endpoint_values["clean"]
    robust_auc = auc_values["robust"]
    clean_qp = np.asarray(method_endpoints["clean"]["qp"], dtype=np.float64)
    clean_nog = np.asarray(method_endpoints["clean"]["nog"], dtype=np.float64)
    clean_noninferior = bool(clean_qp.mean() >= 0.9 * clean_nog.mean())
    robust_ci = bootstrap_ci(robust, seed=20260719)
    clean_ci = bootstrap_ci(clean, seed=20260720)
    gate = bool(
        (robust > 0.0).sum() >= 3
        and robust.mean() > 0.0
        and robust_auc.mean() > 0.0
        and robust_late_positive_fraction >= 0.8
        and clean_noninferior
    )
    statistically_resolved = bool(robust_ci[0] > 0.0)
    if gate and statistically_resolved:
        decision = "POSITIVE_STATISTICALLY_RESOLVED"
    elif gate:
        decision = "POSITIVE_PERSISTENT_ENDPOINT_UNRESOLVED"
    else:
        decision = "NOT_POSITIVE"
    summary = {
        "env_id": env_id,
        "decision": decision,
        "robust_endpoint_wins": int((robust > 0.0).sum()),
        "robust_endpoint_mean_gain": float(robust.mean()),
        "robust_endpoint_bootstrap_95ci": list(robust_ci),
        "robust_auc_wins": int((robust_auc > 0.0).sum()),
        "robust_auc_mean_gain": float(robust_auc.mean()),
        "robust_late_positive_checkpoint_fraction": robust_late_positive_fraction,
        "clean_endpoint_wins": int((clean > 0.0).sum()),
        "clean_endpoint_mean_gain": float(clean.mean()),
        "clean_endpoint_bootstrap_95ci": list(clean_ci),
        "clean_noninferior_10pct": clean_noninferior,
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries, training_raw, eval_raw = read_suite(args)
    evaluation = prepare_eval_curves(eval_raw, args.eval_window)
    training = prepare_training_curves(training_raw, args.training_bin)

    summaries.to_csv(output_root / "phaseVIC_full_run_summaries.csv", index=False)
    evaluation.to_csv(output_root / "phaseVIC_full_checkpoint_evaluations.csv", index=False)
    training.to_csv(output_root / "phaseVIC_full_training_curves.csv", index=False)

    endpoints = []
    for env_id in args.envs:
        plot_environment(env_id, training, evaluation, args.methods, output_root)
        endpoints.extend(endpoint_rows(env_id, evaluation, args.methods, args.eval_window))
    endpoint_frame = pd.DataFrame(endpoints)
    endpoint_frame.to_csv(output_root / "phaseVIC_full_endpoints.csv", index=False)

    paired_rows = []
    decisions = []
    for env_id in args.envs:
        env_rows, env_decision = paired_ablation(env_id, evaluation, args.eval_window)
        paired_rows.extend(env_rows)
        decisions.append(env_decision)
    pd.DataFrame(paired_rows).to_csv(output_root / "phaseVIC_full_paired_qp_vs_nog.csv", index=False)
    (output_root / "phaseVIC_full_decisions.json").write_text(
        json.dumps(decisions, indent=2), encoding="utf-8"
    )

    metadata = {
        "environments": args.envs,
        "seeds": args.seeds,
        "methods": args.methods,
        "method_labels": {method: LABELS[method] for method in args.methods},
        "training_bin_steps": args.training_bin,
        "evaluation_smoothing_checkpoints": args.eval_window,
        "reward_modified": False,
        "adversary_channel": "two-dimensional external force on torso",
        "training_curve_scope": "native-return episodes starting and ending within protagonist collection phases; cross-phase episodes excluded",
    }
    (output_root / "phaseVIC_full_suite_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
