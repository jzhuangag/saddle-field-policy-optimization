from __future__ import annotations

import argparse
import math
import pathlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
from stable_baselines3.common.utils import obs_as_tensor

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.full_policy_followup_common import SavedRun, load_rarl_for_eval, set_rarl_eval_mode
from scripts.run_rawfg_M8_online import build_saved_run, find_latest_run_dir


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
    parser = argparse.ArgumentParser("M12 policy distribution and action statistics audit")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-eval-episodes", type=int, default=100)
    parser.add_argument("--n-probe-states", type=int, default=1000)
    parser.add_argument("--n-probe-samples", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=20260602)
    parser.add_argument("--reference-method", type=str, default="sgd")
    return parser.parse_args()


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


def compact_array(values: np.ndarray | Sequence[float]) -> str:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return ";".join(f"{x:.6g}" for x in arr)


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


def entropy_from_std(std: np.ndarray) -> float:
    std = np.asarray(std, dtype=np.float64)
    return float(np.sum(0.5 * np.log(2.0 * np.pi * np.e * np.maximum(std, 1e-12) ** 2)))


@contextmanager
def temporary_log_std_scale(policy, scale: float):
    original = getattr(policy, "log_std", None)
    if original is None:
        yield
        return
    backup = original.data.detach().clone()
    try:
        if scale <= 0.0:
            original.data.fill_(-20.0)
        else:
            original.data.copy_(backup + math.log(scale))
        yield
    finally:
        original.data.copy_(backup)


def get_policy_distribution(policy, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs_tensor = obs_as_tensor(obs, policy.device)
    with th.no_grad():
        dist = policy.get_distribution(obs_tensor)
        base_dist = getattr(dist, "distribution", dist)
        mean = base_dist.mean.detach().cpu().numpy()
        std = base_dist.stddev.detach().cpu().numpy()
        deterministic_action = dist.get_actions(deterministic=True).detach().cpu().numpy()
        sampled_action = dist.get_actions(deterministic=False).detach().cpu().numpy()
    return mean, std, deterministic_action, sampled_action


def eval_clean_action_rows(
    method_run: MethodRun,
    *,
    deterministic: bool,
    n_eval_episodes: int,
    eval_seed: int,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model, vec_env = load_rarl_for_eval(method_run.saved_run, adv_impact="control", adv_strength=0.0, device=device)
    set_rarl_eval_mode(model, vec_env, operating_mode=None, adv_strength=0.0)
    policy = model.protagonist.policy
    action_low = np.asarray(vec_env.action_space.low, dtype=np.float64)
    action_high = np.asarray(vec_env.action_space.high, dtype=np.float64)
    timestep_rows: List[Dict[str, object]] = []
    episode_rows: List[Dict[str, object]] = []
    try:
        for episode_id in range(n_eval_episodes):
            vec_env.seed(eval_seed + episode_id)
            obs = vec_env.reset()
            episode_buffer: List[Dict[str, object]] = []
            episode_reward = 0.0
            episode_step = 0
            while True:
                obs_arr = np.asarray(obs, dtype=np.float64)
                raw_obs = vec_env.unnormalize_obs(obs_arr.copy()) if hasattr(vec_env, "unnormalize_obs") else obs_arr.copy()
                mean, std, action_det, action_sample = get_policy_distribution(policy, obs_arr)
                log_std = getattr(policy, "log_std", None)
                log_std_arr = (
                    log_std.detach().cpu().numpy().reshape(1, -1).repeat(mean.shape[0], axis=0)
                    if log_std is not None
                    else np.full_like(mean, np.nan)
                )
                mean_0 = mean[0]
                std_0 = std[0]
                det_0 = action_det[0]
                sample_0 = action_sample[0]
                exec_raw = det_0 if deterministic else sample_0
                exec_action = np.clip(exec_raw, action_low, action_high)
                clip_fraction = float(np.mean(np.abs(exec_raw - exec_action) > 1e-12))
                obs, rewards, dones, infos = vec_env.step(exec_action.reshape(1, -1))
                reward = float(rewards[0])
                episode_reward += reward
                episode_step += 1
                episode_buffer.append(
                    {
                        "method": method_run.method,
                        "eval_mode": "deterministic" if deterministic else "stochastic",
                        "episode_id": episode_id,
                        "timestep": episode_step,
                        "observation_norm": float(np.linalg.norm(raw_obs[0])),
                        "action_mean": compact_array(mean_0),
                        "action_sample": compact_array(sample_0),
                        "action_std": compact_array(std_0),
                        "log_std": compact_array(log_std_arr[0]),
                        "action_mean_norm": float(np.linalg.norm(mean_0)),
                        "action_sample_norm": float(np.linalg.norm(sample_0)),
                        "action_std_norm": float(np.linalg.norm(std_0)),
                        "action_clip_fraction": clip_fraction,
                        "reward": reward,
                    }
                )
                if bool(dones[0]):
                    for row in episode_buffer:
                        row["episode_return"] = episode_reward
                    timestep_rows.extend(episode_buffer)
                    episode_rows.append(
                        {
                            "method": method_run.method,
                            "eval_mode": "deterministic" if deterministic else "stochastic",
                            "episode_id": episode_id,
                            "episode_return": episode_reward,
                            "episode_length": episode_step,
                            "mean_action_mean_norm": float(np.mean([row["action_mean_norm"] for row in episode_buffer])),
                            "mean_action_sample_norm": float(np.mean([row["action_sample_norm"] for row in episode_buffer])),
                            "mean_action_std_norm": float(np.mean([row["action_std_norm"] for row in episode_buffer])),
                            "mean_action_clip_fraction": float(np.mean([row["action_clip_fraction"] for row in episode_buffer])),
                        }
                    )
                    break
    finally:
        vec_env.close()
    return pd.DataFrame(timestep_rows), pd.DataFrame(episode_rows)


def run_clean_action_distribution_audit(
    method_runs: Dict[str, MethodRun],
    output_root: pathlib.Path,
    *,
    device: str,
    n_eval_episodes: int,
    eval_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[pd.DataFrame] = []
    episode_frames: List[pd.DataFrame] = []
    for method_run in method_runs.values():
        for deterministic in [True, False]:
            timestep_df, episode_df = eval_clean_action_rows(
                method_run,
                deterministic=deterministic,
                n_eval_episodes=n_eval_episodes,
                eval_seed=eval_seed,
                device=device,
            )
            rows.append(timestep_df)
            episode_frames.append(episode_df)
    long_df = pd.concat(rows, ignore_index=True)
    episode_df = pd.concat(episode_frames, ignore_index=True)
    long_df.to_csv(output_root / "M12_clean_action_distribution_audit.csv", index=False)

    summary = (
        long_df.groupby(["method", "eval_mode"], as_index=False)
        .agg(
            return_mean=("episode_return", "mean"),
            return_std=("episode_return", "std"),
            action_mean_norm=("action_mean_norm", "mean"),
            action_sample_norm=("action_sample_norm", "mean"),
            action_std_norm=("action_std_norm", "mean"),
            action_clip_fraction=("action_clip_fraction", "mean"),
        )
    )
    log_std_stats = (
        long_df.assign(log_std_mean=long_df["log_std"].str.split(";").apply(lambda xs: np.mean([float(x) for x in xs if x != ""])))
        .groupby(["method", "eval_mode"], as_index=False)["log_std_mean"]
        .mean()
    )
    summary = summary.merge(log_std_stats, on=["method", "eval_mode"], how="left")

    corr_rows = []
    successful_rows = []
    for (method, eval_mode), group in episode_df.groupby(["method", "eval_mode"]):
        corr_rows.append(
            {
                "method": method,
                "eval_mode": eval_mode,
                "corr_action_std_norm_vs_return": safe_corr(group["mean_action_std_norm"], group["episode_return"]),
            }
        )
        if eval_mode == "stochastic":
            top_cut = group["episode_return"].quantile(0.75)
            success_ids = set(group[group["episode_return"] >= top_cut]["episode_id"].tolist())
            success_steps = long_df[
                (long_df["method"] == method)
                & (long_df["eval_mode"] == "stochastic")
                & (long_df["episode_id"].isin(success_ids))
            ].copy()
            if not success_steps.empty:
                distances = []
                for _, row in success_steps.iterrows():
                    mean = np.asarray([float(x) for x in str(row["action_mean"]).split(";") if x != ""], dtype=np.float64)
                    sample = np.asarray([float(x) for x in str(row["action_sample"]).split(";") if x != ""], dtype=np.float64)
                    distances.append(float(np.linalg.norm(mean - sample)))
                successful_rows.append(
                    {
                        "method": method,
                        "deterministic_action_distance_to_successful_stochastic_sample": float(np.mean(distances)) if distances else float("nan")
                    }
                )
    corr_df = pd.DataFrame(corr_rows)
    success_df = pd.DataFrame(successful_rows)
    summary = summary.merge(corr_df, on=["method", "eval_mode"], how="left")
    summary = summary.merge(success_df, on="method", how="left")

    fig, ax = plt.subplots(figsize=(12, 6))
    for mode, group in summary.groupby("eval_mode"):
        x = np.arange(len(group))
        ax.bar(
            x + (-0.18 if mode == "deterministic" else 0.18),
            group["return_mean"],
            width=0.35,
            label=mode,
            yerr=group["return_std"].fillna(0.0),
        )
    ax.set_xticks(np.arange(len(summary["method"].unique())))
    ax.set_xticklabels(summary["method"].unique(), rotation=45, ha="right")
    ax.set_ylabel("Clean return")
    ax.set_title("Clean deterministic vs stochastic returns")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_root / "plots" / "M12_clean_det_vs_stoch_returns.png", dpi=200)
    plt.close(fig)

    time_agg = (
        long_df.groupby(["method", "eval_mode", "timestep"], as_index=False)[
            ["action_mean_norm", "action_std_norm", "action_clip_fraction"]
        ]
        .mean()
    )
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    for method, group in time_agg[time_agg["eval_mode"] == "deterministic"].groupby("method"):
        axes[0].plot(group["timestep"], group["action_mean_norm"], label=method)
    axes[0].set_title("Deterministic clean action mean norm vs time")
    axes[0].set_ylabel("Action mean norm")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    for method, group in time_agg[time_agg["eval_mode"] == "stochastic"].groupby("method"):
        axes[1].plot(group["timestep"], group["action_mean_norm"], label=method)
    axes[1].set_title("Stochastic clean action mean norm vs time")
    axes[1].set_xlabel("Episode timestep")
    axes[1].set_ylabel("Action mean norm")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_root / "plots" / "M12_action_mean_norm_vs_time.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    for method, group in time_agg[time_agg["eval_mode"] == "deterministic"].groupby("method"):
        axes[0].plot(group["timestep"], group["action_std_norm"], label=method)
    axes[0].set_title("Deterministic clean action std norm vs time")
    axes[0].set_ylabel("Action std norm")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    for method, group in time_agg[time_agg["eval_mode"] == "stochastic"].groupby("method"):
        axes[1].plot(group["timestep"], group["action_std_norm"], label=method)
    axes[1].set_title("Stochastic clean action std norm vs time")
    axes[1].set_xlabel("Episode timestep")
    axes[1].set_ylabel("Action std norm")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_root / "plots" / "M12_action_std_norm_vs_time.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    log_plot = summary.drop_duplicates(["method", "eval_mode"])[["method", "eval_mode", "log_std_mean"]]
    for idx, mode in enumerate(["deterministic", "stochastic"]):
        group = log_plot[log_plot["eval_mode"] == mode].sort_values("method")
        ax.bar(np.arange(len(group)) + (-0.18 if mode == "deterministic" else 0.18), group["log_std_mean"], width=0.35, label=mode)
    ax.set_xticks(np.arange(len(log_plot["method"].unique())))
    ax.set_xticklabels(sorted(log_plot["method"].unique()), rotation=45, ha="right")
    ax.set_ylabel("Mean log_std")
    ax.set_title("Policy log_std by method")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_root / "plots" / "M12_log_std_by_method.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    clip_plot = summary.sort_values(["method", "eval_mode"])
    for idx, mode in enumerate(["deterministic", "stochastic"]):
        group = clip_plot[clip_plot["eval_mode"] == mode]
        ax.bar(np.arange(len(group)) + (-0.18 if mode == "deterministic" else 0.18), group["action_clip_fraction"], width=0.35, label=mode)
    ax.set_xticks(np.arange(len(clip_plot["method"].unique())))
    ax.set_xticklabels(sorted(clip_plot["method"].unique()), rotation=45, ha="right")
    ax.set_ylabel("Clip fraction")
    ax.set_title("Clean action clip fraction by method")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_root / "plots" / "M12_action_clip_fraction_by_method.png", dpi=200)
    plt.close(fig)

    proposed = summary[summary["method"] == "proposed_qp_rawFG_eta1_cap003"].set_index("eval_mode")
    baseline = summary[summary["method"].isin(["sgd", "egm", "ppm"])].groupby("eval_mode").mean(numeric_only=True)
    lines = [
        "# M12 Clean Action Distribution Report",
        "",
        f"- Methods successfully read: {', '.join(METHODS)}",
        f"- n_eval_episodes per method/mode: {n_eval_episodes}",
        "",
        "## Key answers",
        f"- Does proposed need stochasticity to perform well? {'Yes' if proposed.loc['stochastic', 'return_mean'] > proposed.loc['deterministic', 'return_mean'] else 'No'}",
        f"- Is deterministic mean action bad for proposed? {'Yes' if proposed.loc['deterministic', 'return_mean'] < baseline.loc['deterministic', 'return_mean'] and proposed.loc['stochastic', 'return_mean'] > proposed.loc['deterministic', 'return_mean'] else 'Not clearly'}",
        f"- Is proposed log_std larger than baselines? {'Yes' if proposed['log_std_mean'].mean() > baseline['log_std_mean'].mean() else 'No'}",
        "",
        "## Per-method summary",
        frame_to_text(summary),
    ]
    (output_root / "M12_clean_action_distribution_report.md").write_text("\n".join(lines), encoding="utf-8")
    return long_df, summary


def collect_probe_states(
    method_run: MethodRun,
    *,
    n_probe_states: int,
    eval_seed: int,
    device: str,
) -> np.ndarray:
    model, vec_env = load_rarl_for_eval(method_run.saved_run, adv_impact="control", adv_strength=0.0, device=device)
    set_rarl_eval_mode(model, vec_env, operating_mode=None, adv_strength=0.0)
    states: List[np.ndarray] = []
    try:
        episode_id = 0
        while len(states) < n_probe_states:
            vec_env.seed(eval_seed + episode_id)
            obs = vec_env.reset()
            done = False
            while not done and len(states) < n_probe_states:
                raw_obs = vec_env.unnormalize_obs(np.asarray(obs).copy()) if hasattr(vec_env, "unnormalize_obs") else np.asarray(obs).copy()
                states.append(np.asarray(raw_obs[0], dtype=np.float64))
                action, _ = model.predict(obs, deterministic=True)
                obs, _, dones, _ = vec_env.step(action)
                done = bool(dones[0])
            episode_id += 1
    finally:
        vec_env.close()
    return np.asarray(states[:n_probe_states], dtype=np.float64)


def probe_policy_distribution(
    method_run: MethodRun,
    raw_probe_states: np.ndarray,
    *,
    n_samples: int,
    device: str,
) -> pd.DataFrame:
    model, vec_env = load_rarl_for_eval(method_run.saved_run, adv_impact="control", adv_strength=0.0, device=device)
    set_rarl_eval_mode(model, vec_env, operating_mode=None, adv_strength=0.0)
    policy = model.protagonist.policy
    action_low = np.asarray(vec_env.action_space.low, dtype=np.float64)
    action_high = np.asarray(vec_env.action_space.high, dtype=np.float64)
    try:
        norm_obs = vec_env.normalize_obs(raw_probe_states.copy()) if hasattr(vec_env, "normalize_obs") else raw_probe_states.copy()
        obs_tensor = obs_as_tensor(norm_obs, policy.device)
        with th.no_grad():
            dist = policy.get_distribution(obs_tensor)
            base_dist = getattr(dist, "distribution", dist)
            mean = base_dist.mean.detach().cpu().numpy()
            std = base_dist.stddev.detach().cpu().numpy()
            sample_list = [dist.get_actions(deterministic=False).detach().cpu().numpy() for _ in range(n_samples)]
        samples = np.stack(sample_list, axis=1)
        rows = []
        for idx in range(raw_probe_states.shape[0]):
            mean_i = mean[idx]
            std_i = std[idx]
            samples_i = samples[idx]
            sample_mean = np.mean(samples_i, axis=0)
            clipped_samples = np.clip(samples_i, action_low, action_high)
            saturation_fraction = float(np.mean(np.abs(samples_i - clipped_samples) > 1e-12))
            rows.append(
                {
                    "method": method_run.method,
                    "probe_id": idx,
                    "action_mean": compact_array(mean_i),
                    "action_std": compact_array(std_i),
                    "action_mean_norm": float(np.linalg.norm(mean_i)),
                    "action_std_norm": float(np.linalg.norm(std_i)),
                    "sample_mean_norm": float(np.linalg.norm(sample_mean)),
                    "policy_entropy": entropy_from_std(std_i),
                    "covariance_trace": float(np.sum(std_i ** 2)),
                    "action_saturation_fraction": saturation_fraction,
                    "sample_return_proxy": float("nan"),
                    "tanh_pre_squash_stat": float("nan"),
                }
            )
        return pd.DataFrame(rows)
    finally:
        vec_env.close()


def run_probe_policy_distribution_audit(
    method_runs: Dict[str, MethodRun],
    output_root: pathlib.Path,
    *,
    n_probe_states: int,
    n_samples: int,
    eval_seed: int,
    reference_method: str,
    device: str,
) -> pd.DataFrame:
    raw_probe_states = collect_probe_states(
        method_runs[reference_method],
        n_probe_states=n_probe_states,
        eval_seed=eval_seed,
        device=device,
    )
    frames = [
        probe_policy_distribution(method_run, raw_probe_states, n_samples=n_samples, device=device)
        for method_run in method_runs.values()
    ]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(output_root / "M12_probe_policy_distribution.csv", index=False)

    summary = (
        df.groupby("method", as_index=False)[
            ["action_mean_norm", "action_std_norm", "sample_mean_norm", "policy_entropy", "covariance_trace", "action_saturation_fraction"]
        ]
        .mean()
    )
    qp_means = df[df["method"] == "proposed_qp_rawFG_eta1_cap003"].copy()
    compare_rows = []
    qp_mean_vectors = np.stack(
        [np.asarray([float(x) for x in v.split(";") if x != ""], dtype=np.float64) for v in qp_means["action_mean"]]
    )
    qp_std_vectors = np.stack(
        [np.asarray([float(x) for x in v.split(";") if x != ""], dtype=np.float64) for v in qp_means["action_std"]]
    )
    for method in ["sgd", "egm", "ppm", "proposed_noG_rawFG_eta1_cap003"]:
        comp = df[df["method"] == method].copy()
        comp_mean_vectors = np.stack(
            [np.asarray([float(x) for x in v.split(";") if x != ""], dtype=np.float64) for v in comp["action_mean"]]
        )
        comp_std_vectors = np.stack(
            [np.asarray([float(x) for x in v.split(";") if x != ""], dtype=np.float64) for v in comp["action_std"]]
        )
        compare_rows.append(
            {
                "compare_to": method,
                "mean_action_distance": float(np.mean(np.linalg.norm(qp_mean_vectors - comp_mean_vectors, axis=1))),
                "std_action_distance": float(np.mean(np.linalg.norm(qp_std_vectors - comp_std_vectors, axis=1))),
            }
        )
    compare_df = pd.DataFrame(compare_rows)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for method, group in df.groupby("method"):
        axes[0].hist(group["action_mean_norm"], bins=30, alpha=0.4, density=True, label=method)
        axes[1].hist(group["action_std_norm"], bins=30, alpha=0.4, density=True, label=method)
    axes[0].set_title("Probe action mean norm distribution")
    axes[0].set_xlabel("Action mean norm")
    axes[1].set_title("Probe action std norm distribution")
    axes[1].set_xlabel("Action std norm")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_root / "plots" / "M12_probe_action_mean_distribution.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(summary["method"], summary["policy_entropy"])
    ax.set_title("Policy entropy vs method")
    ax.set_ylabel("Entropy")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_root / "plots" / "M12_policy_entropy_vs_method.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(summary["method"], summary["action_std_norm"])
    ax.set_title("Probe action std norm by method")
    ax.set_ylabel("Action std norm")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_root / "plots" / "M12_probe_action_std_distribution.png", dpi=200)
    plt.close(fig)

    lines = [
        "# M12 Probe Policy Distribution Report",
        "",
        f"- Probe states collected from `{reference_method}` clean deterministic rollout: {n_probe_states}",
        f"- Samples per state: {n_samples}",
        "",
        "## Summary by method",
        frame_to_text(summary),
        "",
        "## proposed_qp distance to comparison methods",
        frame_to_text(compare_df),
    ]
    (output_root / "M12_probe_policy_distribution_report.md").write_text("\n".join(lines), encoding="utf-8")
    return df


def evaluate_logstd_scale(
    method_run: MethodRun,
    *,
    scales: Sequence[float],
    n_eval_episodes: int,
    eval_seed: int,
    device: str,
) -> pd.DataFrame:
    model, vec_env = load_rarl_for_eval(method_run.saved_run, adv_impact="control", adv_strength=0.0, device=device)
    set_rarl_eval_mode(model, vec_env, operating_mode=None, adv_strength=0.0)
    policy = model.protagonist.policy
    action_low = np.asarray(vec_env.action_space.low, dtype=np.float64)
    action_high = np.asarray(vec_env.action_space.high, dtype=np.float64)
    rows: List[Dict[str, object]] = []
    try:
        for scale in scales:
            with temporary_log_std_scale(policy, scale):
                episode_returns = []
                action_std_norms = []
                for episode_id in range(n_eval_episodes):
                    vec_env.seed(eval_seed + episode_id)
                    obs = vec_env.reset()
                    episode_reward = 0.0
                    step_std_norms = []
                    while True:
                        obs_arr = np.asarray(obs, dtype=np.float64)
                        mean, std, det_action, sample_action = get_policy_distribution(policy, obs_arr)
                        if scale == 0.0:
                            exec_raw = det_action[0]
                        else:
                            exec_raw = sample_action[0]
                        exec_action = np.clip(exec_raw, action_low, action_high)
                        step_std_norms.append(float(np.linalg.norm(std[0])))
                        obs, rewards, dones, _ = vec_env.step(exec_action.reshape(1, -1))
                        episode_reward += float(rewards[0])
                        if bool(dones[0]):
                            episode_returns.append(episode_reward)
                            action_std_norms.append(float(np.mean(step_std_norms)) if step_std_norms else 0.0)
                            break
                rows.append(
                    {
                        "method": method_run.method,
                        "log_std_scale": scale,
                        "mean_return": float(np.mean(episode_returns)),
                        "std_return": float(np.std(episode_returns)),
                        "mean_action_std_norm": float(np.mean(action_std_norms)),
                    }
                )
    finally:
        vec_env.close()
    return pd.DataFrame(rows)


def run_logstd_scale_audit(
    method_runs: Dict[str, MethodRun],
    output_root: pathlib.Path,
    *,
    n_eval_episodes: int,
    eval_seed: int,
    device: str,
) -> pd.DataFrame:
    scales = [0.0, 0.25, 0.5, 1.0, 1.5]
    df = evaluate_logstd_scale(
        method_runs["proposed_qp_rawFG_eta1_cap003"],
        scales=scales,
        n_eval_episodes=n_eval_episodes,
        eval_seed=eval_seed,
        device=device,
    )
    df.to_csv(output_root / "M12_logstd_scale_eval.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(df["log_std_scale"], df["mean_return"], yerr=df["std_return"], marker="o")
    ax.set_title("proposed_qp clean eval under log_std scaling")
    ax.set_xlabel("log_std scale")
    ax.set_ylabel("Clean return")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_root / "plots" / "M12_logstd_scale_clean_eval.png", dpi=200)
    plt.close(fig)

    best_scale = float(df.loc[df["mean_return"].idxmax(), "log_std_scale"])
    lines = [
        "# M12 LogStd Scale Eval Report",
        "",
        f"- Best clean return scale: {best_scale}",
        f"- Does proposed need stochasticity to perform well? {'Yes' if df.loc[df['log_std_scale'] == 1.0, 'mean_return'].iloc[0] > df.loc[df['log_std_scale'] == 0.0, 'mean_return'].iloc[0] else 'No'}",
        f"- Does reducing log_std improve proposed clean eval? {'Yes' if df.loc[df['log_std_scale'] == 0.5, 'mean_return'].iloc[0] > df.loc[df['log_std_scale'] == 1.0, 'mean_return'].iloc[0] else 'No'}",
        "",
        frame_to_text(df),
    ]
    (output_root / "M12_logstd_scale_eval_report.md").write_text("\n".join(lines), encoding="utf-8")
    return df


def run_v_policy_join(
    output_root: pathlib.Path,
    action_summary: pd.DataFrame,
    probe_df: pd.DataFrame,
) -> pd.DataFrame:
    v_df = pd.read_csv(output_root / "M11_lyapunov_field_norms.csv")
    full_df = v_df[v_df["scope"] == "full_policy"].copy()
    final_v = (
        full_df.sort_values(["method", "outer_iteration"])
        .groupby("method", as_index=False)
        .tail(1)[["method", "V", "actor_F_norm", "logstd_F_norm", "critic_F_norm"]]
        .rename(columns={"V": "final_V"})
    )
    clean_det = action_summary[action_summary["eval_mode"] == "deterministic"].copy()
    clean_stoch = action_summary[action_summary["eval_mode"] == "stochastic"].copy()
    probe_summary = (
        probe_df.groupby("method", as_index=False)[["action_mean_norm", "action_std_norm", "policy_entropy", "action_saturation_fraction"]]
        .mean()
        .rename(
            columns={
                "action_mean_norm": "probe_action_mean_norm",
                "action_std_norm": "probe_action_std_norm",
                "policy_entropy": "probe_policy_entropy",
                "action_saturation_fraction": "probe_action_clip_fraction",
            }
        )
    )
    joined = (
        final_v.merge(
            clean_det[["method", "return_mean", "log_std_mean", "action_mean_norm", "action_std_norm", "action_clip_fraction"]].rename(
                columns={
                    "return_mean": "clean_deterministic_return",
                    "log_std_mean": "clean_deterministic_log_std_mean",
                    "action_mean_norm": "clean_deterministic_action_mean_norm",
                    "action_std_norm": "clean_deterministic_action_std_norm",
                    "action_clip_fraction": "clean_deterministic_clip_fraction",
                }
            ),
            on="method",
            how="left",
        )
        .merge(
            clean_stoch[["method", "return_mean", "log_std_mean", "action_mean_norm", "action_std_norm", "action_clip_fraction"]].rename(
                columns={
                    "return_mean": "clean_stochastic_return",
                    "log_std_mean": "clean_stochastic_log_std_mean",
                    "action_mean_norm": "clean_stochastic_action_mean_norm",
                    "action_std_norm": "clean_stochastic_action_std_norm",
                    "action_clip_fraction": "clean_stochastic_clip_fraction",
                }
            ),
            on="method",
            how="left",
        )
        .merge(probe_summary, on="method", how="left")
    )
    joined.to_csv(output_root / "M12_V_policy_distribution_join.csv", index=False)

    corr_v_stoch = safe_corr(joined["final_V"], joined["clean_stochastic_return"])
    corr_v_det = safe_corr(joined["final_V"], joined["clean_deterministic_return"])
    proposed = joined[joined["method"] == "proposed_qp_rawFG_eta1_cap003"].iloc[0]
    baseline = joined[joined["method"].isin(["sgd", "egm", "ppm"])].mean(numeric_only=True)
    lines = [
        "# M12 V vs Policy Distribution Report",
        "",
        f"- Does lower V correlate with higher stochastic return? corr = {corr_v_stoch:.3f}",
        f"- Does lower V correlate with lower deterministic return? corr = {corr_v_det:.3f}",
        f"- Is proposed low V mainly associated with higher policy variance? {'Yes' if proposed['probe_action_std_norm'] > baseline['probe_action_std_norm'] else 'No'}",
        "",
        frame_to_text(joined),
    ]
    (output_root / "M12_V_policy_distribution_report.md").write_text("\n".join(lines), encoding="utf-8")
    return joined


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    (output_root / "plots").mkdir(parents=True, exist_ok=True)
    method_runs = load_method_runs(output_root)

    clean_long_df, clean_summary = run_clean_action_distribution_audit(
        method_runs,
        output_root,
        device=args.device,
        n_eval_episodes=args.n_eval_episodes,
        eval_seed=args.eval_seed,
    )
    probe_df = run_probe_policy_distribution_audit(
        method_runs,
        output_root,
        n_probe_states=args.n_probe_states,
        n_samples=args.n_probe_samples,
        eval_seed=args.eval_seed,
        reference_method=args.reference_method,
        device=args.device,
    )
    run_logstd_scale_audit(
        method_runs,
        output_root,
        n_eval_episodes=args.n_eval_episodes,
        eval_seed=args.eval_seed,
        device=args.device,
    )
    run_v_policy_join(output_root, clean_summary, probe_df)


if __name__ == "__main__":
    main()
