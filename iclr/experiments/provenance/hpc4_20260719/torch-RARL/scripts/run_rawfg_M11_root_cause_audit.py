from __future__ import annotations

import argparse
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from models.RARL import RARL
from scripts.full_policy_followup_common import (
    SavedRun,
    load_rarl_for_eval,
    set_rarl_eval_mode,
)
from scripts.full_policy_optimizer_probe import (
    collect_probe_batches,
    compute_loss_and_grads,
    named_parameters,
)
from scripts.run_proposed_qp_new_v2_rawfg_audit import build_control_manager, classify_block
from scripts.run_rawfg_M8_online import build_saved_run, find_latest_run_dir, make_collage, read_method_frames


METHODS = [
    "adam",
    "sgd",
    "egm",
    "ppm",
    "proposed_noG_rawFG_eta1_cap003",
    "proposed_qp_rawFG_eta1_cap003",
]


def frame_to_text(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    return frame.to_string(index=False)


@dataclass(frozen=True)
class MethodRun:
    method: str
    run_root: pathlib.Path
    analysis_dir: pathlib.Path
    latest_run_dir: pathlib.Path
    saved_run: SavedRun
    n_steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("M11 train-eval mismatch and Lyapunov norm audit")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-eval-episodes", type=int, default=50)
    parser.add_argument("--num-probes", type=int, default=2)
    parser.add_argument("--reference-method", type=str, default="sgd")
    return parser.parse_args()


def ensure_columns(frame: pd.DataFrame, required: Sequence[str], context: str) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns for {context}: {missing}")


def load_method_runs(output_root: pathlib.Path) -> Dict[str, MethodRun]:
    runs_dir = output_root / "runs_seed0"
    results: Dict[str, MethodRun] = {}
    for method in METHODS:
        run_root = runs_dir / method
        analysis_dir = run_root / "analysis"
        latest_run_dir = find_latest_run_dir(run_root / "saved_models", "HalfCheetah-v4")
        saved_run = build_saved_run(latest_run_dir, method)
        n_steps = int(saved_run.args_data.get("n_steps", 2048))
        results[method] = MethodRun(
            method=method,
            run_root=run_root,
            analysis_dir=analysis_dir,
            latest_run_dir=latest_run_dir,
            saved_run=saved_run,
            n_steps=n_steps,
        )
    return results


def safe_corr(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=np.float64)
    y_arr = np.asarray(list(y), dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 2:
        return float("nan")
    x_sel = x_arr[mask]
    y_sel = y_arr[mask]
    if np.std(x_sel) < 1e-12 or np.std(y_sel) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x_sel, y_sel)[0, 1])


def last5_mean(series: pd.Series) -> float:
    return float(series.tail(min(5, len(series))).mean()) if len(series) else float("nan")


def align_training_eval(training: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    left = eval_df[["outer_iteration", "mean_reward", "timestep"]].sort_values("outer_iteration").copy()
    right = training[["outer_iteration", "episode_return", "timestep"]].sort_values("outer_iteration").copy()
    return pd.merge_asof(left, right, on="outer_iteration", direction="backward", suffixes=("_eval", "_train"))


def make_train_eval_alignment(method_runs: Dict[str, MethodRun], output_root: pathlib.Path) -> tuple[pd.DataFrame, List[str]]:
    rows: List[Dict[str, object]] = []
    missing_notes: List[str] = []
    scatter_rows: List[Dict[str, object]] = []

    for method, method_run in method_runs.items():
        frames = read_method_frames(method, method_run.latest_run_dir, method_run.analysis_dir)
        training = frames["training"].copy()
        clean = frames["clean"].copy()
        adv = frames["adv"].copy()
        ensure_columns(training, ["outer_iteration", "episode_return", "timestep"], f"{method} training")
        ensure_columns(clean, ["outer_iteration", "mean_reward", "timestep"], f"{method} clean eval")
        ensure_columns(adv, ["outer_iteration", "mean_reward", "timestep"], f"{method} control-adv eval")

        training_aligned_clean = align_training_eval(training, clean)
        training_aligned_adv = align_training_eval(training, adv)
        clean_adv = pd.merge_asof(
            clean[["outer_iteration", "mean_reward", "timestep"]].sort_values("outer_iteration"),
            adv[["outer_iteration", "mean_reward", "timestep"]].sort_values("outer_iteration"),
            on="outer_iteration",
            direction="nearest",
            suffixes=("_clean", "_adv"),
        )

        corr_train_clean = safe_corr(training_aligned_clean["episode_return"], training_aligned_clean["mean_reward"])
        corr_train_adv = safe_corr(training_aligned_adv["episode_return"], training_aligned_adv["mean_reward"])
        corr_clean_adv = safe_corr(clean_adv["mean_reward_clean"], clean_adv["mean_reward_adv"])

        training_last5 = last5_mean(training["episode_return"])
        clean_last5 = last5_mean(clean["mean_reward"])
        adv_last5 = last5_mean(adv["mean_reward"])

        summary = pd.read_csv(method_run.analysis_dir / "run_summary.csv").iloc[0]
        rows.append(
            {
                "method": method,
                "total_iterations": int(summary["total_iterations"]),
                "num_training_episodes": int(summary["num_training_episodes"]),
                "num_clean_eval_points": int(summary["num_clean_eval_points"]),
                "num_adv_eval_points": int(summary["num_adv_eval_points"]),
                "corr_training_vs_clean": corr_train_clean,
                "corr_training_vs_control_adv": corr_train_adv,
                "corr_clean_vs_control_adv": corr_clean_adv,
                "training_final_last5": training_last5,
                "clean_eval_final_last5": clean_last5,
                "control_adv_eval_final_last5": adv_last5,
                "gap_training_minus_clean": training_last5 - clean_last5,
                "gap_training_minus_control_adv": training_last5 - adv_last5,
            }
        )

        scatter_rows.extend(
            [
                {"method": method, "pair": "training_vs_clean", "x": x, "y": y}
                for x, y in zip(training_aligned_clean["episode_return"], training_aligned_clean["mean_reward"])
            ]
        )
        scatter_rows.extend(
            [
                {"method": method, "pair": "training_vs_control_adv", "x": x, "y": y}
                for x, y in zip(training_aligned_adv["episode_return"], training_aligned_adv["mean_reward"])
            ]
        )
        scatter_rows.extend(
            [
                {"method": method, "pair": "clean_vs_control_adv", "x": x, "y": y}
                for x, y in zip(clean_adv["mean_reward_clean"], clean_adv["mean_reward_adv"])
            ]
        )

    alignment_df = pd.DataFrame(rows)
    alignment_df.to_csv(output_root / "M11_train_eval_alignment.csv", index=False)

    scatter_df = pd.DataFrame(scatter_rows)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pair_specs = [
        ("training_vs_clean", "Training return", "Clean eval return"),
        ("training_vs_control_adv", "Training return", "Control-adv eval return"),
        ("clean_vs_control_adv", "Clean eval return", "Control-adv eval return"),
    ]
    for ax, (pair, xlabel, ylabel) in zip(axes, pair_specs):
        sub = scatter_df[scatter_df["pair"] == pair]
        for method, group in sub.groupby("method"):
            ax.scatter(group["x"], group["y"], s=18, alpha=0.75, label=method)
        ax.set_title(pair.replace("_", " "))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_root / "plots" / "M11_train_vs_eval_alignment.png", dpi=200)
    plt.close(fig)

    proposed_rows = alignment_df[alignment_df["method"].str.contains("proposed_")]
    baseline_rows = alignment_df[~alignment_df["method"].str.contains("proposed_")]
    lines = [
        "# M11 Train vs Eval Alignment Report",
        "",
        f"- Methods successfully read: {', '.join(alignment_df['method'].tolist())}",
        "",
        "## Key answers",
        f"- Proposed good only in training return? {'Yes' if proposed_rows['gap_training_minus_clean'].mean() > baseline_rows['gap_training_minus_clean'].mean() else 'Not clearly'}",
        f"- Training return predicts clean eval for baselines? mean corr = {baseline_rows['corr_training_vs_clean'].mean():.3f}",
        f"- Training return predicts clean eval for proposed? mean corr = {proposed_rows['corr_training_vs_clean'].mean():.3f}",
        f"- Proposed larger train-eval gap than EGM/PPM/SGD? {'Yes' if proposed_rows['gap_training_minus_clean'].mean() > alignment_df[alignment_df['method'].isin(['sgd','egm','ppm'])]['gap_training_minus_clean'].mean() else 'No'}",
        "",
        "## Per-method summary",
        frame_to_text(alignment_df),
    ]
    (output_root / "M11_train_eval_alignment_report.md").write_text("\n".join(lines), encoding="utf-8")
    return alignment_df, missing_notes


def collect_reference_probes(method_runs: Dict[str, MethodRun], reference_method: str, device: str, num_probes: int):
    reference_run = method_runs[reference_method]
    model, vec_env = load_rarl_for_eval(reference_run.saved_run, adv_impact="control", adv_strength=1.0, device=device)
    try:
        probes = collect_probe_batches(model, "protagonist", num_probes)
    finally:
        vec_env.close()
    return probes


def tensor_norm_from_names(grads: Dict[str, object], names: Sequence[str]) -> float:
    if not names:
        return 0.0
    pieces = [grads[name].detach().reshape(-1) for name in names]
    if not pieces:
        return 0.0
    vec = pieces[0] if len(pieces) == 1 else __import__("torch").cat(pieces)
    return float(__import__("torch").norm(vec).item())


def compute_field_norm_rows(
    method_runs: Dict[str, MethodRun],
    output_root: pathlib.Path,
    device: str,
    num_probes: int,
    reference_method: str,
) -> pd.DataFrame:
    probes = collect_reference_probes(method_runs, reference_method, device, num_probes)
    rows: List[Dict[str, object]] = []
    for method, method_run in method_runs.items():
        model, vec_env = load_rarl_for_eval(method_run.saved_run, adv_impact="control", adv_strength=1.0, device=device)
        try:
            algo_cls = model.protagonist.__class__
            checkpoint_df = pd.read_csv(method_run.analysis_dir / "checkpoint_inventory.csv").sort_values("timesteps")
            for checkpoint in checkpoint_df.itertuples(index=False):
                checkpoint_path = method_run.latest_run_dir / checkpoint.checkpoint_file
                algo = algo_cls.load(str(checkpoint_path), env=model.protagonist.env, device=device)
                named = named_parameters(algo.policy)
                param_names = [name for name, _ in named]
                actor_names = [name for name in param_names if classify_block(name) == "actor"]
                logstd_names = [name for name in param_names if classify_block(name) == "logstd"]
                critic_names = [name for name in param_names if classify_block(name) == "critic"]
                full_policy_names = list(param_names)
                actor_game_names = actor_names + logstd_names
                per_scope_rows: Dict[str, List[Dict[str, object]]] = {"full_policy": [], "actor_game": []}
                for probe in probes:
                    eval_info = compute_loss_and_grads(
                        algo,
                        probe.rollout_data,
                        max_grad_norm=float("inf"),
                        vf_coef=float(algo.vf_coef),
                        ent_coef=float(algo.ent_coef),
                    )
                    actor_norm = tensor_norm_from_names(eval_info["grads"], actor_names)
                    logstd_norm = tensor_norm_from_names(eval_info["grads"], logstd_names)
                    critic_norm = tensor_norm_from_names(eval_info["grads"], critic_names)
                    full_norm = tensor_norm_from_names(eval_info["grads"], full_policy_names)
                    actor_game_norm = tensor_norm_from_names(eval_info["grads"], actor_game_names)
                    common = {
                        "method": method,
                        "timestep": int(checkpoint.timesteps),
                        "outer_iteration": float(checkpoint.timesteps) / float(method_run.n_steps),
                        "checkpoint_path": str(checkpoint_path),
                        "probe_id": probe.probe_idx,
                        "actor_F_norm": actor_norm,
                        "logstd_F_norm": logstd_norm,
                        "critic_F_norm": critic_norm,
                        "approx_kl_on_probe": float(eval_info["approx_kl"]),
                        "clip_fraction_on_probe": float(eval_info["clip_fraction"]),
                    }
                    per_scope_rows["full_policy"].append(
                        {
                            **common,
                            "scope": "full_policy",
                            "F_norm": full_norm,
                            "V": 0.5 * (full_norm ** 2),
                        }
                    )
                    per_scope_rows["actor_game"].append(
                        {
                            **common,
                            "scope": "actor_game",
                            "F_norm": actor_game_norm,
                            "V": 0.5 * (actor_game_norm ** 2),
                        }
                    )
                for scope_rows in per_scope_rows.values():
                    rows.extend(scope_rows)
        finally:
            vec_env.close()

    long_df = pd.DataFrame(rows)
    grouped = (
        long_df.groupby(["method", "timestep", "outer_iteration", "checkpoint_path", "scope"], as_index=False)[
            ["F_norm", "V", "actor_F_norm", "logstd_F_norm", "critic_F_norm", "approx_kl_on_probe", "clip_fraction_on_probe"]
        ]
        .mean()
    )
    grouped.to_csv(output_root / "M11_lyapunov_field_norms.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    full_df = grouped[grouped["scope"] == "full_policy"]
    for method, group in full_df.groupby("method"):
        group = group.sort_values("outer_iteration")
        axes[0].plot(group["outer_iteration"], group["V"], label=method)
    axes[0].set_title("V(z)=0.5||F||^2 vs outer iteration")
    axes[0].set_xlabel("Outer iteration")
    axes[0].set_ylabel("V")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)

    proposed_sub = full_df[full_df["method"].isin(["proposed_noG_rawFG_eta1_cap003", "proposed_qp_rawFG_eta1_cap003"])]
    for method, group in proposed_sub.groupby("method"):
        group = group.sort_values("outer_iteration")
        axes[1].plot(group["outer_iteration"], group["actor_F_norm"], label=f"{method} actor")
        axes[1].plot(group["outer_iteration"], group["logstd_F_norm"], linestyle=":", label=f"{method} logstd")
        axes[1].plot(group["outer_iteration"], group["critic_F_norm"], linestyle="--", label=f"{method} critic")
    axes[1].set_title("Proposed F norms by block")
    axes[1].set_xlabel("Outer iteration")
    axes[1].set_ylabel("F block norm")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_root / "plots" / "M11_V_field_norm_vs_outer_iteration.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for method, group in full_df.groupby("method"):
        group = group.sort_values("outer_iteration")
        ax.plot(group["outer_iteration"], group["actor_F_norm"], label=f"{method} actor")
        ax.plot(group["outer_iteration"], group["logstd_F_norm"], linestyle=":", label=f"{method} logstd")
        ax.plot(group["outer_iteration"], group["critic_F_norm"], linestyle="--", label=f"{method} critic")
    ax.set_title("F norm by block vs outer iteration (full-policy probe)")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Norm")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(output_root / "plots" / "M11_F_norm_by_block_vs_outer_iteration.png", dpi=200)
    plt.close(fig)

    summary_rows = []
    for method, method_run in method_runs.items():
        summary = pd.read_csv(method_run.analysis_dir / "run_summary.csv").iloc[0]
        summary_rows.append(
            {
                "method": method,
                "clean_last5": float(summary["last5_clean_mean"]),
                "control_adv_last5": float(summary["last5_adversarial_mean"]),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    clean_final = summary_df[["method", "clean_last5"]]
    adv_final = summary_df[["method", "control_adv_last5"]]
    compare_df = (
        full_df.groupby("method")["V"].mean().rename("mean_V").reset_index().merge(clean_final, on="method").merge(adv_final, on="method")
    )
    corr_v_clean = safe_corr(compare_df["mean_V"], compare_df["clean_last5"])
    corr_v_adv = safe_corr(compare_df["mean_V"], compare_df["control_adv_last5"])
    lines = [
        "# M11 Lyapunov Field Norm Report",
        "",
        f"- Probe protocol: protagonist role, fixed minibatches collected once from `{reference_method}` final checkpoint under control-RARL.",
        "- Scope coverage: full_policy and actor_game.",
        "- Offline field norm probe uses `max_grad_norm = inf` for every method to avoid turning the Lyapunov norm into a proxy for each method's training-time clip threshold.",
        "",
        "## Key answers",
        f"- Does proposed_qp have smaller V than EGM/PPM/SGD during training? {'Yes' if compare_df.set_index('method').loc['proposed_qp_rawFG_eta1_cap003','mean_V'] < compare_df[compare_df['method'].isin(['sgd','egm','ppm'])]['mean_V'].mean() else 'No'}",
        f"- Does lower V correlate with higher clean eval? corr(V, clean) = {corr_v_clean:.3f}",
        f"- Does lower V correlate with higher control-adv eval? corr(V, control_adv) = {corr_v_adv:.3f}",
        f"- If proposed has low V but poor eval: {'Yes' if compare_df.set_index('method').loc['proposed_qp_rawFG_eta1_cap003','mean_V'] < compare_df[compare_df['method'].isin(['sgd','egm','ppm'])]['mean_V'].mean() and compare_df.set_index('method').loc['proposed_qp_rawFG_eta1_cap003','clean_last5'] < compare_df[compare_df['method'].isin(['sgd','egm','ppm'])]['clean_last5'].mean() else 'No'}",
        "",
        "## Mean V vs eval",
        frame_to_text(compare_df),
    ]
    (output_root / "M11_lyapunov_field_norm_report.md").write_text("\n".join(lines), encoding="utf-8")
    return grouped


def evaluate_protocol(
    saved_run: SavedRun,
    *,
    method: str,
    adv_impact: str,
    adv_strength: float,
    operating_mode: str | None,
    deterministic: bool,
    n_eval_episodes: int,
    device: str,
    protocol_name: str,
) -> Dict[str, object]:
    try:
        model, vec_env = load_rarl_for_eval(saved_run, adv_impact=adv_impact, adv_strength=adv_strength, device=device)
        try:
            set_rarl_eval_mode(model, vec_env, operating_mode=operating_mode, adv_strength=adv_strength)
            obs = vec_env.reset()
            episode_rewards: List[float] = []
            episode_lengths: List[int] = []
            adv_norms: List[float] = []
            perturb_norms: List[float] = []
            clip_fractions: List[float] = []
            ep_reward = 0.0
            ep_len = 0
            ep_adv: List[float] = []
            ep_perturb: List[float] = []
            ep_clip: List[float] = []
            while len(episode_rewards) < n_eval_episodes:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, rewards, dones, infos = vec_env.step(action)
                info = infos[0]
                ep_reward += float(rewards[0])
                ep_len += 1
                ep_adv.append(float(info.get("adversary_action_norm_post_clip", info.get("adversary_action_norm_pre_clip", 0.0))))
                ep_perturb.append(
                    float(
                        info.get(
                            "applied_control_perturbation_norm",
                            info.get("applied_force_norm", info.get("applied_disturbance_norm", 0.0)),
                        )
                    )
                )
                ep_clip.append(float(info.get("action_clip_fraction", 0.0)))
                if bool(dones[0]):
                    episode_rewards.append(ep_reward)
                    episode_lengths.append(ep_len)
                    adv_norms.append(float(np.mean(ep_adv)) if ep_adv else 0.0)
                    perturb_norms.append(float(np.mean(ep_perturb)) if ep_perturb else 0.0)
                    clip_fractions.append(float(np.mean(ep_clip)) if ep_clip else 0.0)
                    obs = vec_env.reset()
                    ep_reward = 0.0
                    ep_len = 0
                    ep_adv = []
                    ep_perturb = []
                    ep_clip = []
            return {
                "method": method,
                "protocol": protocol_name,
                "adv_impact": adv_impact,
                "adv_strength": adv_strength,
                "deterministic": deterministic,
                "status": "ok",
                "mean_return": float(np.mean(episode_rewards)),
                "std_return": float(np.std(episode_rewards)),
                "adversary_action_norm": float(np.mean(adv_norms)),
                "applied_perturbation_norm": float(np.mean(perturb_norms)),
                "clip_fraction": float(np.mean(clip_fractions)),
                "episode_length": float(np.mean(episode_lengths)),
                "error": "",
            }
        finally:
            vec_env.close()
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {
            "method": method,
            "protocol": protocol_name,
            "adv_impact": adv_impact,
            "adv_strength": adv_strength,
            "deterministic": deterministic,
            "status": "error",
            "mean_return": float("nan"),
            "std_return": float("nan"),
            "adversary_action_norm": float("nan"),
            "applied_perturbation_norm": float("nan"),
            "clip_fraction": float("nan"),
            "episode_length": float("nan"),
            "error": str(exc),
        }


def run_eval_protocol_consistency(method_runs: Dict[str, MethodRun], output_root: pathlib.Path, device: str, n_eval_episodes: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for method, method_run in method_runs.items():
        saved_run = method_run.saved_run
        rows.append(
            evaluate_protocol(
                saved_run,
                method=method,
                adv_impact="control",
                adv_strength=0.0,
                operating_mode=None,
                deterministic=True,
                n_eval_episodes=n_eval_episodes,
                device=device,
                protocol_name="clean_deterministic",
            )
        )
        rows.append(
            evaluate_protocol(
                saved_run,
                method=method,
                adv_impact="control",
                adv_strength=0.0,
                operating_mode=None,
                deterministic=False,
                n_eval_episodes=n_eval_episodes,
                device=device,
                protocol_name="clean_stochastic",
            )
        )
        rows.append(
            evaluate_protocol(
                saved_run,
                method=method,
                adv_impact="control",
                adv_strength=1.0,
                operating_mode="protagonist",
                deterministic=True,
                n_eval_episodes=n_eval_episodes,
                device=device,
                protocol_name="control_adv_deterministic",
            )
        )
        rows.append(
            evaluate_protocol(
                saved_run,
                method=method,
                adv_impact="control",
                adv_strength=1.0,
                operating_mode="protagonist",
                deterministic=False,
                n_eval_episodes=n_eval_episodes,
                device=device,
                protocol_name="control_adv_stochastic",
            )
        )
        rows.append(
            evaluate_protocol(
                saved_run,
                method=method,
                adv_impact="force",
                adv_strength=1.0,
                operating_mode="protagonist",
                deterministic=True,
                n_eval_episodes=n_eval_episodes,
                device=device,
                protocol_name="force_adv_deterministic",
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_root / "M11_eval_protocol_consistency.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for method, group in df[df["protocol"].isin(["clean_deterministic", "clean_stochastic"])].groupby("method"):
        axes[0].plot(group["protocol"], group["mean_return"], marker="o", label=method)
    axes[0].set_title("Clean deterministic vs stochastic")
    axes[0].set_ylabel("Mean return")
    axes[0].grid(alpha=0.3)

    for method, group in df[df["protocol"].isin(["control_adv_deterministic", "control_adv_stochastic"])].groupby("method"):
        axes[1].plot(group["protocol"], group["mean_return"], marker="o", label=method)
    axes[1].set_title("Control-adv deterministic vs stochastic")
    axes[1].set_ylabel("Mean return")
    axes[1].grid(alpha=0.3)

    for method, group in df[df["protocol"] == "control_adv_deterministic"].groupby("method"):
        axes[2].bar(method, float(group["applied_perturbation_norm"].mean()))
    axes[2].set_title("Control-adv perturbation norm")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_root / "plots" / "M11_eval_protocol_consistency.png", dpi=200)
    plt.close(fig)

    qprop = df[df["method"] == "proposed_qp_rawFG_eta1_cap003"].set_index("protocol")
    lines = [
        "# M11 Eval Protocol Consistency Report",
        "",
        f"- Does proposed only fail under deterministic eval? {'Yes' if qprop.loc['clean_stochastic','mean_return'] > qprop.loc['clean_deterministic','mean_return'] and qprop.loc['control_adv_stochastic','mean_return'] > qprop.loc['control_adv_deterministic','mean_return'] else 'No'}",
        f"- Does proposed only fail under control-adv eval? {'Yes' if qprop.loc['clean_deterministic','mean_return'] > qprop.loc['control_adv_deterministic','mean_return'] else 'No'}",
        "- Does adversary perturbation differ strongly across methods? See per-method control-adv perturbation norms below.",
        "- Is control-adv eval pairing fair across methods? The same protocol was rerun for every final checkpoint; any residual difference comes from learned adversary behavior, not eval code.",
        "",
        frame_to_text(df),
    ]
    (output_root / "M11_eval_protocol_consistency_report.md").write_text("\n".join(lines), encoding="utf-8")
    return df


def run_robustness_sanity(method_runs: Dict[str, MethodRun], output_root: pathlib.Path, device: str, n_eval_episodes: int) -> pd.DataFrame:
    strengths = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    rows: List[Dict[str, object]] = []
    for method, method_run in method_runs.items():
        for strength in strengths:
            rows.append(
                evaluate_protocol(
                    method_run.saved_run,
                    method=method,
                    adv_impact="control",
                    adv_strength=strength,
                    operating_mode="protagonist",
                    deterministic=True,
                    n_eval_episodes=n_eval_episodes,
                    device=device,
                    protocol_name=f"control_strength_{strength:g}",
                )
            )
    df = pd.DataFrame(rows)
    df["strength"] = df["adv_strength"]
    df.to_csv(output_root / "M11_robustness_sweep_sanity.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ok_df = df[df["status"] == "ok"]
    for method, group in ok_df.groupby("method"):
        group = group.sort_values("strength")
        ax.errorbar(group["strength"], group["mean_return"], yerr=group["std_return"], marker="o", label=method)
    ax.set_title("Control robustness sweep sanity")
    ax.set_xlabel("Adversary strength")
    ax.set_ylabel("Mean return")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_root / "plots" / "M11_robustness_sweep_sanity.png", dpi=200)
    plt.close(fig)

    lines = ["# M11 Robustness Sweep Sanity Report", ""]
    for method, group in ok_df.groupby("method"):
        group = group.sort_values("strength")
        returns = group["mean_return"].to_numpy()
        monotonic = bool(np.all(np.diff(returns) <= 1e-6))
        lines.append(f"- `{method}` monotonic harmful under control sweep: {monotonic}")
    lines.append("")
    lines.append(frame_to_text(df))
    (output_root / "M11_robustness_sweep_sanity_report.md").write_text("\n".join(lines), encoding="utf-8")
    return df


def write_root_cause_report(
    output_root: pathlib.Path,
    alignment_df: pd.DataFrame,
    lyap_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
) -> None:
    full_df = lyap_df[lyap_df["scope"] == "full_policy"]
    mean_v = full_df.groupby("method")["V"].mean()
    mean_critic_share = (
        full_df.assign(total_blocks=lambda df: df["actor_F_norm"] + df["logstd_F_norm"] + df["critic_F_norm"])
        .assign(critic_share=lambda df: np.where(df["total_blocks"] > 1e-12, df["critic_F_norm"] / df["total_blocks"], np.nan))
        .groupby("method")["critic_share"]
        .mean()
    )
    q_alignment = alignment_df.set_index("method")
    protocol_pivot = consistency_df.pivot_table(index="method", columns="protocol", values="mean_return", aggfunc="mean")
    robust_pivot = robustness_df[robustness_df["status"] == "ok"].pivot_table(index="method", columns="strength", values="mean_return", aggfunc="mean")

    qp_method = "proposed_qp_rawFG_eta1_cap003"
    baseline_subset = ["sgd", "egm", "ppm"]
    baseline_clean_mean = q_alignment.loc[baseline_subset, "clean_eval_final_last5"].mean()
    baseline_adv_mean = q_alignment.loc[baseline_subset, "control_adv_eval_final_last5"].mean()

    labels: List[str] = []
    if q_alignment.loc[qp_method, "gap_training_minus_clean"] > q_alignment.loc[baseline_subset, "gap_training_minus_clean"].mean():
        labels.append("A. training/eval metric mismatch")
    if mean_v.loc[qp_method] < mean_v.loc[baseline_subset].mean() and q_alignment.loc[qp_method, "clean_eval_final_last5"] < baseline_clean_mean:
        labels.append("B. Lyapunov V-return objective mismatch")
    if mean_v.loc[qp_method] >= mean_v.loc[baseline_subset].mean():
        labels.append("C. proposed does not actually minimize V online")
    if protocol_pivot.loc[qp_method, "control_adv_deterministic"] + 1e-6 < protocol_pivot.loc[qp_method, "clean_deterministic"]:
        perturb_qp = consistency_df[(consistency_df["method"] == qp_method) & (consistency_df["protocol"] == "control_adv_deterministic")]["applied_perturbation_norm"].mean()
        perturb_baseline = consistency_df[
            (consistency_df["method"].isin(baseline_subset)) & (consistency_df["protocol"] == "control_adv_deterministic")
        ]["applied_perturbation_norm"].mean()
        if math.isfinite(perturb_qp) and math.isfinite(perturb_baseline) and perturb_qp > 1.25 * perturb_baseline:
            labels.append("D. control adversary eval pairing unfair")
    if protocol_pivot.loc[qp_method, "clean_stochastic"] > protocol_pivot.loc[qp_method, "clean_deterministic"] + 5.0 or protocol_pivot.loc[qp_method, "control_adv_stochastic"] > protocol_pivot.loc[qp_method, "control_adv_deterministic"] + 5.0:
        labels.append("E. deterministic/stochastic eval mismatch")
    m10_diag_path = output_root / "rawFG_M10_eta1_cap003_qp_diagnostics.csv"
    if m10_diag_path.exists():
        m10_diag = pd.read_csv(m10_diag_path)
        qp_cap_active = float(m10_diag[m10_diag["method"] == qp_method]["cap_active"].mean()) if "cap_active" in m10_diag.columns else float("nan")
        if math.isfinite(qp_cap_active) and qp_cap_active > 0.3:
            labels.append("F. cap too tight or too often active")
    if mean_critic_share.loc[qp_method] > 0.6:
        labels.append("G. critic/log_std blocks dominate field norm")
    if q_alignment.loc[qp_method, "clean_eval_final_last5"] < baseline_clean_mean and q_alignment.loc[qp_method, "control_adv_eval_final_last5"] < baseline_adv_mean:
        labels.append("H. EGM/PPM still better aligned with PPO return")
    if not labels:
        labels.append("No single dominant failure mode isolated")

    lines = [
        "# M11 Root Cause Report",
        "",
        "## Main classification",
        *[f"- {label}" for label in labels],
        "",
        "## Supporting evidence",
        f"- Proposed train-clean gap: {q_alignment.loc[qp_method, 'gap_training_minus_clean']:.3f}",
        f"- Proposed train-adv gap: {q_alignment.loc[qp_method, 'gap_training_minus_control_adv']:.3f}",
        f"- Proposed mean V (full_policy probe): {mean_v.loc[qp_method]:.6e}",
        f"- Baseline mean V (sgd/egm/ppm average): {mean_v.loc[baseline_subset].mean():.6e}",
        f"- Proposed final clean last5: {q_alignment.loc[qp_method, 'clean_eval_final_last5']:.3f}",
        f"- Proposed final control-adv last5: {q_alignment.loc[qp_method, 'control_adv_eval_final_last5']:.3f}",
        f"- Baseline final clean last5 mean (sgd/egm/ppm): {baseline_clean_mean:.3f}",
        f"- Baseline final control-adv last5 mean (sgd/egm/ppm): {baseline_adv_mean:.3f}",
        f"- Proposed critic share of field norm: {mean_critic_share.loc[qp_method]:.3f}",
        "",
        "## Protocol observations",
        f"- Proposed clean deterministic vs stochastic: {protocol_pivot.loc[qp_method, 'clean_deterministic']:.3f} vs {protocol_pivot.loc[qp_method, 'clean_stochastic']:.3f}",
        f"- Proposed control-adv deterministic vs stochastic: {protocol_pivot.loc[qp_method, 'control_adv_deterministic']:.3f} vs {protocol_pivot.loc[qp_method, 'control_adv_stochastic']:.3f}",
        "",
        "## Control robustness sanity (final checkpoints)",
    ]
    for method in METHODS:
        if method in robust_pivot.index:
            returns = robust_pivot.loc[method].sort_index()
            monotonic = bool(np.all(np.diff(returns.to_numpy()) <= 1e-6))
            lines.append(f"- {method}: monotonic_harmful={monotonic}, returns={returns.to_dict()}")
    (output_root / "M11_root_cause_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    (output_root / "plots").mkdir(parents=True, exist_ok=True)
    method_runs = load_method_runs(output_root)

    alignment_df, _ = make_train_eval_alignment(method_runs, output_root)
    lyap_df = compute_field_norm_rows(method_runs, output_root, args.device, args.num_probes, args.reference_method)
    consistency_df = run_eval_protocol_consistency(method_runs, output_root, args.device, args.n_eval_episodes)
    robustness_df = run_robustness_sanity(method_runs, output_root, args.device, args.n_eval_episodes)
    write_root_cause_report(output_root, alignment_df, lyap_df, consistency_df, robustness_df)


if __name__ == "__main__":
    main()
