from __future__ import annotations

import argparse
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
from gymnasium.wrappers.common import TimeLimit

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.run_rawfg_M12_policy_distribution_audit import (
    METHODS,
    compact_array,
    entropy_from_std,
    frame_to_text,
    get_policy_distribution,
    load_method_runs,
    safe_corr,
)
from scripts.full_policy_followup_common import load_rarl_for_eval, set_rarl_eval_mode


@dataclass
class PolicyContext:
    method: str
    model: object
    vec_env: object
    policy: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("M13 action mean quality / sampled action advantage audit")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-probe-states", type=int, default=1000)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--best-of-k", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=20260603)
    parser.add_argument("--reference-method", type=str, default="sgd")
    return parser.parse_args()


def open_policy_contexts(method_runs: Dict[str, object], device: str) -> Dict[str, PolicyContext]:
    contexts: Dict[str, PolicyContext] = {}
    for method, method_run in method_runs.items():
        model, vec_env = load_rarl_for_eval(method_run.saved_run, adv_impact="control", adv_strength=0.0, device=device)
        set_rarl_eval_mode(model, vec_env, operating_mode=None, adv_strength=0.0)
        contexts[method] = PolicyContext(
            method=method,
            model=model,
            vec_env=vec_env,
            policy=model.protagonist.policy,
        )
    return contexts


def close_policy_contexts(contexts: Dict[str, PolicyContext]) -> None:
    for ctx in contexts.values():
        ctx.vec_env.close()


def make_raw_env(env_id: str) -> gym.Env:
    return gym.make(env_id)


def unwrap_time_limit_and_base(env: gym.Env) -> tuple[TimeLimit | None, gym.Env]:
    time_limit = None
    current = env
    visited = set()
    while True:
        if id(current) in visited:
            break
        visited.add(id(current))
        if isinstance(current, TimeLimit):
            time_limit = current
        if not hasattr(current, "env"):
            break
        current = current.env
    return time_limit, current


def capture_env_state(env: gym.Env) -> Dict[str, object]:
    time_limit, base = unwrap_time_limit_and_base(env)
    state = {
        "qpos": np.asarray(base.data.qpos, dtype=np.float64).copy(),
        "qvel": np.asarray(base.data.qvel, dtype=np.float64).copy(),
        "elapsed_steps": int(getattr(time_limit, "_elapsed_steps", 0)) if time_limit is not None else 0,
    }
    return state


def restore_env_state(env: gym.Env, state: Dict[str, object]) -> np.ndarray:
    time_limit, base = unwrap_time_limit_and_base(env)
    base.set_state(np.asarray(state["qpos"], dtype=np.float64), np.asarray(state["qvel"], dtype=np.float64))
    if time_limit is not None:
        time_limit._elapsed_steps = int(state["elapsed_steps"])
    return np.asarray(base._get_obs(), dtype=np.float64)


def one_step_reward_proxy(env: gym.Env, state: Dict[str, object], action: np.ndarray) -> tuple[float, float, bool]:
    restore_env_state(env, state)
    _, reward, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float64))
    next_obs = restore_env_state(env, state)
    return float(reward), float(np.linalg.norm(next_obs)), bool(terminated or truncated)


def distribution_from_raw_obs(ctx: PolicyContext, raw_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norm_obs = ctx.vec_env.normalize_obs(np.asarray(raw_obs, dtype=np.float64)[None, :])
    mean, std, _, _ = get_policy_distribution(ctx.policy, norm_obs)
    return mean[0], std[0]


def sample_actions_from_raw_obs(ctx: PolicyContext, raw_obs: np.ndarray, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    norm_obs = ctx.vec_env.normalize_obs(np.asarray(raw_obs, dtype=np.float64)[None, :])
    mean, std, _, _ = get_policy_distribution(ctx.policy, norm_obs)
    obs_tensor = th.as_tensor(norm_obs, dtype=th.float32, device=ctx.policy.device)
    with th.no_grad():
        dist = ctx.policy.get_distribution(obs_tensor)
        samples = [dist.get_actions(deterministic=False).detach().cpu().numpy()[0] for _ in range(n_samples)]
    return mean[0], np.asarray(samples, dtype=np.float64)


def collect_probe_snapshots(
    reference_ctx: PolicyContext,
    *,
    n_probe_states: int,
    eval_seed: int,
) -> List[Dict[str, object]]:
    env_id = reference_ctx.vec_env.venv.envs[0].spec.id
    raw_env = make_raw_env(env_id)
    snapshots: List[Dict[str, object]] = []
    try:
        episode_id = 0
        while len(snapshots) < n_probe_states:
            obs, _ = raw_env.reset(seed=eval_seed + episode_id)
            done = False
            while not done and len(snapshots) < n_probe_states:
                state = capture_env_state(raw_env)
                mean, _ = distribution_from_raw_obs(reference_ctx, obs)
                snapshots.append(
                    {
                        "state_id": len(snapshots),
                        "raw_obs": np.asarray(obs, dtype=np.float64).copy(),
                        "state": state,
                        "reference_action_mean": mean.copy(),
                    }
                )
                obs, _, terminated, truncated, _ = raw_env.step(mean)
                done = bool(terminated or truncated)
            episode_id += 1
    finally:
        raw_env.close()
    return snapshots


def run_action_reward_proxy_audit(
    method_runs: Dict[str, object],
    output_root: pathlib.Path,
    *,
    device: str,
    n_probe_states: int,
    n_samples: int,
    eval_seed: int,
    reference_method: str,
) -> pd.DataFrame:
    contexts = open_policy_contexts(method_runs, device)
    proxy_env = make_raw_env(method_runs[reference_method].saved_run.args_data["env"])
    action_low = np.asarray(proxy_env.action_space.low, dtype=np.float64)
    action_high = np.asarray(proxy_env.action_space.high, dtype=np.float64)
    try:
        proxy_env.reset(seed=eval_seed)
        snapshots = collect_probe_snapshots(contexts[reference_method], n_probe_states=n_probe_states, eval_seed=eval_seed)
        rows: List[Dict[str, object]] = []
        baseline_methods = ["sgd", "egm", "ppm"]
        rng = np.random.default_rng(eval_seed)
        for method in METHODS:
            ctx = contexts[method]
            for snapshot in snapshots:
                raw_obs = snapshot["raw_obs"]
                state = snapshot["state"]
                mean_action, sampled_actions = sample_actions_from_raw_obs(ctx, raw_obs, n_samples)
                mean_action = np.clip(mean_action, action_low, action_high)
                sample_rewards: List[float] = []
                sample_actions_clipped: List[np.ndarray] = []
                for sample_idx, sample_action in enumerate(sampled_actions):
                    clipped = np.clip(sample_action, action_low, action_high)
                    reward, next_state_norm, done_flag = one_step_reward_proxy(proxy_env, state, clipped)
                    sample_rewards.append(reward)
                    sample_actions_clipped.append(clipped)
                    rows.append(
                        {
                            "method": method,
                            "state_id": int(snapshot["state_id"]),
                            "candidate_type": f"sample_{sample_idx:02d}",
                            "action_norm": float(np.linalg.norm(clipped)),
                            "distance_to_method_mean": float(np.linalg.norm(clipped - mean_action)),
                            "immediate_reward": reward,
                            "next_state_norm": next_state_norm,
                            "done_flag": int(done_flag),
                        }
                    )
                reward_mean, next_state_norm_mean, done_mean = one_step_reward_proxy(proxy_env, state, mean_action)
                rows.append(
                    {
                        "method": method,
                        "state_id": int(snapshot["state_id"]),
                        "candidate_type": "method_mean",
                        "action_norm": float(np.linalg.norm(mean_action)),
                        "distance_to_method_mean": 0.0,
                        "immediate_reward": reward_mean,
                        "next_state_norm": next_state_norm_mean,
                        "done_flag": int(done_mean),
                    }
                )
                best_idx = int(np.argmax(sample_rewards))
                best_action = sample_actions_clipped[best_idx]
                rows.append(
                    {
                        "method": method,
                        "state_id": int(snapshot["state_id"]),
                        "candidate_type": "best_sample",
                        "action_norm": float(np.linalg.norm(best_action)),
                        "distance_to_method_mean": float(np.linalg.norm(best_action - mean_action)),
                        "immediate_reward": float(sample_rewards[best_idx]),
                        "next_state_norm": float("nan"),
                        "done_flag": 0,
                    }
                )
                for baseline_method in baseline_methods:
                    baseline_mean, _ = distribution_from_raw_obs(contexts[baseline_method], raw_obs)
                    baseline_mean = np.clip(baseline_mean, action_low, action_high)
                    reward_base, next_state_norm_base, done_base = one_step_reward_proxy(proxy_env, state, baseline_mean)
                    rows.append(
                        {
                            "method": method,
                            "state_id": int(snapshot["state_id"]),
                            "candidate_type": f"{baseline_method}_mean",
                            "action_norm": float(np.linalg.norm(baseline_mean)),
                            "distance_to_method_mean": float(np.linalg.norm(baseline_mean - mean_action)),
                            "immediate_reward": reward_base,
                            "next_state_norm": next_state_norm_base,
                            "done_flag": int(done_base),
                        }
                    )
                _, std_action = distribution_from_raw_obs(ctx, raw_obs)
                for perturb_idx in range(4):
                    noise = rng.normal(size=mean_action.shape) * std_action * 0.5
                    perturbed = np.clip(mean_action + noise, action_low, action_high)
                    reward_p, next_state_norm_p, done_p = one_step_reward_proxy(proxy_env, state, perturbed)
                    rows.append(
                        {
                            "method": method,
                            "state_id": int(snapshot["state_id"]),
                            "candidate_type": f"local_perturb_{perturb_idx}",
                            "action_norm": float(np.linalg.norm(perturbed)),
                            "distance_to_method_mean": float(np.linalg.norm(perturbed - mean_action)),
                            "immediate_reward": reward_p,
                            "next_state_norm": next_state_norm_p,
                            "done_flag": int(done_p),
                        }
                    )
        df = pd.DataFrame(rows)
        df.to_csv(output_root / "M13_action_reward_proxy.csv", index=False)

        report_rows = []
        proposed_qp = df[df["method"] == "proposed_qp_rawFG_eta1_cap003"]
        for candidate in ["method_mean", "best_sample", "sgd_mean", "egm_mean", "ppm_mean"]:
            sub = proposed_qp[proposed_qp["candidate_type"] == candidate]
            report_rows.append({"candidate_type": candidate, "mean_immediate_reward": float(sub["immediate_reward"].mean())})
        report_df = pd.DataFrame(report_rows)

        fig, ax = plt.subplots(figsize=(12, 6))
        plot_df = proposed_qp[proposed_qp["candidate_type"].isin(["method_mean", "best_sample", "sgd_mean", "egm_mean", "ppm_mean"])]
        agg = plot_df.groupby("candidate_type", as_index=False)["immediate_reward"].mean()
        ax.bar(agg["candidate_type"], agg["immediate_reward"])
        ax.set_title("proposed_qp one-step reward proxy by candidate")
        ax.set_ylabel("Immediate reward proxy")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_root / "plots" / "M13_action_reward_proxy_by_candidate.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        gap_df = proposed_qp[proposed_qp["candidate_type"].isin(["method_mean", "best_sample"])]
        gap_pivot = gap_df.pivot(index="state_id", columns="candidate_type", values="immediate_reward").dropna()
        gap = gap_pivot["best_sample"] - gap_pivot["method_mean"]
        ax.hist(gap, bins=40)
        ax.set_title("proposed_qp best-sample minus mean reward proxy")
        ax.set_xlabel("Reward gap")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_root / "plots" / "M13_mean_vs_best_sample_reward_gap.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        compare_df = proposed_qp[proposed_qp["candidate_type"].isin(["method_mean", "egm_mean"])].pivot(
            index="state_id", columns="candidate_type", values="immediate_reward"
        ).dropna()
        ax.scatter(compare_df["method_mean"], compare_df["egm_mean"], s=8, alpha=0.5)
        min_v = float(min(compare_df.min()))
        max_v = float(max(compare_df.max()))
        ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="black")
        ax.set_title("proposed_qp mean vs EGM mean one-step reward proxy")
        ax.set_xlabel("proposed_qp mean reward proxy")
        ax.set_ylabel("EGM mean reward proxy")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_root / "plots" / "M13_proposed_vs_egm_action_reward_proxy.png", dpi=200)
        plt.close(fig)

        best_sample_better_frac = float(np.mean(gap > 0.0)) if len(gap) else float("nan")
        lines = [
            "# M13 Action Reward Proxy Report",
            "",
            f"- Probe states: {n_probe_states}",
            f"- Samples per state: {n_samples}",
            "",
            f"1. Is proposed mean action worse than its sampled actions? {'Yes' if report_df.set_index('candidate_type').loc['best_sample','mean_immediate_reward'] > report_df.set_index('candidate_type').loc['method_mean','mean_immediate_reward'] else 'No'}",
            f"2. Is proposed best sampled action better than proposed mean action? {'Yes' if best_sample_better_frac > 0.5 else 'No'} (fraction of states: {best_sample_better_frac:.3f})",
            f"3. Is EGM/PPM mean action better than proposed mean action on same states? {'Yes' if report_df.set_index('candidate_type').loc['egm_mean','mean_immediate_reward'] > report_df.set_index('candidate_type').loc['method_mean','mean_immediate_reward'] or report_df.set_index('candidate_type').loc['ppm_mean','mean_immediate_reward'] > report_df.set_index('candidate_type').loc['method_mean','mean_immediate_reward'] else 'No'}",
            f"4. Does proposed distribution contain good actions but mean is misplaced? {'Yes' if best_sample_better_frac > 0.5 and report_df.set_index('candidate_type').loc['best_sample','mean_immediate_reward'] > report_df.set_index('candidate_type').loc['method_mean','mean_immediate_reward'] else 'No'}",
            f"5. Is the stochastic gain explainable by action samples with better reward proxy? {'Yes' if best_sample_better_frac > 0.5 else 'No'}",
            "",
            frame_to_text(report_df),
        ]
        (output_root / "M13_action_reward_proxy_report.md").write_text("\n".join(lines), encoding="utf-8")
        return df
    finally:
        proxy_env.close()
        close_policy_contexts(contexts)


def run_action_substitution_eval(
    method_runs: Dict[str, object],
    output_root: pathlib.Path,
    *,
    device: str,
    n_eval_episodes: int,
    eval_seed: int,
    best_of_k: int,
) -> pd.DataFrame:
    contexts = open_policy_contexts(method_runs, device)
    env_id = method_runs["proposed_qp_rawFG_eta1_cap003"].saved_run.args_data["env"]
    action_low = np.asarray(gym.make(env_id).action_space.low, dtype=np.float64)
    action_high = np.asarray(gym.make(env_id).action_space.high, dtype=np.float64)
    proxy_env = make_raw_env(env_id)
    variants = [
        "normal_deterministic",
        "normal_stochastic",
        "mean_plus_noise",
        "best_of_k",
        "egm_mean_substitution",
        "proposed_mean_with_egm_std",
    ]
    rows: List[Dict[str, object]] = []
    try:
        proxy_env.reset(seed=eval_seed)
        proposed_ctx = contexts["proposed_qp_rawFG_eta1_cap003"]
        egm_ctx = contexts["egm"]
        rng = np.random.default_rng(eval_seed)
        for variant in variants:
            raw_env = make_raw_env(env_id)
            try:
                for episode_id in range(n_eval_episodes):
                    raw_obs, _ = raw_env.reset(seed=eval_seed + episode_id)
                    done = False
                    episode_return = 0.0
                    action_norms = []
                    action_std_norms = []
                    distances = []
                    reward_proxies = []
                    while not done:
                        state = capture_env_state(raw_env)
                        prop_mean, prop_std = distribution_from_raw_obs(proposed_ctx, raw_obs)
                        egm_mean, egm_std = distribution_from_raw_obs(egm_ctx, raw_obs)
                        if variant == "normal_deterministic":
                            action = prop_mean
                            distance = 0.0
                            std_used = prop_std
                        elif variant == "normal_stochastic":
                            _, prop_samples = sample_actions_from_raw_obs(proposed_ctx, raw_obs, 1)
                            action = prop_samples[0]
                            distance = float(np.linalg.norm(action - prop_mean))
                            std_used = prop_std
                        elif variant == "mean_plus_noise":
                            action = prop_mean + rng.normal(size=prop_mean.shape) * prop_std
                            distance = float(np.linalg.norm(action - prop_mean))
                            std_used = prop_std
                        elif variant == "best_of_k":
                            _, prop_samples = sample_actions_from_raw_obs(proposed_ctx, raw_obs, best_of_k)
                            candidate_rewards = []
                            clipped_candidates = []
                            for sample in prop_samples:
                                clipped = np.clip(sample, action_low, action_high)
                                reward_proxy, _, _ = one_step_reward_proxy(proxy_env, state, clipped)
                                candidate_rewards.append(reward_proxy)
                                clipped_candidates.append(clipped)
                            best_idx = int(np.argmax(candidate_rewards))
                            action = clipped_candidates[best_idx]
                            distance = float(np.linalg.norm(action - prop_mean))
                            std_used = prop_std
                        elif variant == "egm_mean_substitution":
                            action = egm_mean
                            distance = float(np.linalg.norm(action - prop_mean))
                            std_used = egm_std
                        elif variant == "proposed_mean_with_egm_std":
                            action = prop_mean + rng.normal(size=prop_mean.shape) * egm_std
                            distance = float(np.linalg.norm(action - prop_mean))
                            std_used = egm_std
                        else:
                            raise ValueError(variant)
                        action = np.clip(action, action_low, action_high)
                        reward_proxy, _, _ = one_step_reward_proxy(proxy_env, state, action)
                        raw_obs, reward, terminated, truncated, _ = raw_env.step(action)
                        done = bool(terminated or truncated)
                        episode_return += float(reward)
                        action_norms.append(float(np.linalg.norm(action)))
                        action_std_norms.append(float(np.linalg.norm(std_used)))
                        distances.append(distance)
                        reward_proxies.append(reward_proxy)
                    rows.append(
                        {
                            "method": "proposed_qp_rawFG_eta1_cap003",
                            "variant": variant,
                            "episode_id": episode_id,
                            "episode_return": episode_return,
                            "action_norm": float(np.mean(action_norms)),
                            "action_std_norm": float(np.mean(action_std_norms)),
                            "selected_action_distance_from_mean": float(np.mean(distances)),
                            "one_step_reward_proxy": float(np.mean(reward_proxies)),
                        }
                    )
            finally:
                raw_env.close()
        df = pd.DataFrame(rows)
        df.to_csv(output_root / "M13_action_substitution_eval.csv", index=False)

        fig, ax = plt.subplots(figsize=(12, 6))
        agg = df.groupby("variant", as_index=False)["episode_return"].mean()
        ax.bar(agg["variant"], agg["episode_return"])
        ax.set_title("Proposed action substitution returns")
        ax.set_ylabel("Episode return")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_root / "plots" / "M13_action_substitution_returns.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 6))
        dist_agg = df.groupby("variant", as_index=False)["selected_action_distance_from_mean"].mean()
        ax.bar(dist_agg["variant"], dist_agg["selected_action_distance_from_mean"])
        ax.set_title("Selected action distance from proposed mean")
        ax.set_ylabel("Distance")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_root / "plots" / "M13_selected_action_distance.png", dpi=200)
        plt.close(fig)

        mean_returns = agg.set_index("variant")["episode_return"]
        lines = [
            "# M13 Action Substitution Eval Report",
            "",
            f"1. Does best-of-K greatly improve proposed? {'Yes' if mean_returns['best_of_k'] > mean_returns['normal_deterministic'] + 20.0 else 'No'}",
            f"2. Does adding noise to proposed mean recover stochastic performance? {'Yes' if mean_returns['mean_plus_noise'] > mean_returns['normal_deterministic'] else 'No'}",
            f"3. Does EGM mean action outperform proposed mean on same states? {'Yes' if mean_returns['egm_mean_substitution'] > mean_returns['normal_deterministic'] else 'No'}",
            f"4. Is proposed failure mainly a mean-action issue rather than distribution issue? {'Yes' if mean_returns['best_of_k'] > mean_returns['normal_deterministic'] and mean_returns['normal_stochastic'] > mean_returns['normal_deterministic'] else 'No'}",
            "",
            frame_to_text(df.groupby('variant', as_index=False)[['episode_return','action_norm','action_std_norm','selected_action_distance_from_mean','one_step_reward_proxy']].mean()),
        ]
        (output_root / "M13_action_substitution_eval_report.md").write_text("\n".join(lines), encoding="utf-8")
        return df
    finally:
        proxy_env.close()
        close_policy_contexts(contexts)


def run_block_responsibility_audit(output_root: pathlib.Path) -> pd.DataFrame:
    lyap = pd.read_csv(output_root / "M11_lyapunov_field_norms.csv")
    diag = pd.read_csv(output_root / "rawFG_M10_eta1_cap003_qp_diagnostics.csv")
    methods = ["proposed_noG_rawFG_eta1_cap003", "proposed_qp_rawFG_eta1_cap003"]
    full = lyap[(lyap["scope"] == "full_policy") & (lyap["method"].isin(methods))].copy()
    full = full.sort_values(["method", "outer_iteration"])
    rows = []
    for method, group in full.groupby("method"):
        group = group.sort_values("outer_iteration").copy()
        group["actor_V_proxy"] = 0.5 * group["actor_F_norm"] ** 2
        group["logstd_V_proxy"] = 0.5 * group["logstd_F_norm"] ** 2
        group["critic_V_proxy"] = 0.5 * group["critic_F_norm"] ** 2
        group["actor_V_decrease"] = group["actor_V_proxy"].shift(1) - group["actor_V_proxy"]
        group["logstd_V_decrease"] = group["logstd_V_proxy"].shift(1) - group["logstd_V_proxy"]
        group["critic_V_decrease"] = group["critic_V_proxy"].shift(1) - group["critic_V_proxy"]
        denom = (
            group[["actor_V_decrease", "logstd_V_decrease", "critic_V_decrease"]]
            .clip(lower=0.0)
            .sum(axis=1)
            .replace(0.0, np.nan)
        )
        group["actor_V_decrease_fraction"] = group["actor_V_decrease"] / denom
        group["logstd_V_decrease_fraction"] = group["logstd_V_decrease"] / denom
        group["critic_V_decrease_fraction"] = group["critic_V_decrease"] / denom
        diag_sub = (
            diag[diag["method"] == method]
            .groupby("outer_iteration", as_index=False)[["actor_update_norm", "logstd_update_norm", "critic_update_norm", "actual_V_change"]]
            .mean()
        )
        merged = pd.merge_asof(
            group.sort_values("outer_iteration"),
            diag_sub.sort_values("outer_iteration"),
            on="outer_iteration",
            direction="backward",
            tolerance=1.0,
        )
        rows.append(merged)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(output_root / "M13_block_responsibility.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for method, group in out.groupby("method"):
        axes[0].plot(group["outer_iteration"], group["actor_V_decrease_fraction"], label=f"{method} actor")
        axes[0].plot(group["outer_iteration"], group["logstd_V_decrease_fraction"], linestyle=":", label=f"{method} logstd")
        axes[0].plot(group["outer_iteration"], group["critic_V_decrease_fraction"], linestyle="--", label=f"{method} critic")
    axes[0].set_title("Block responsibility for V decrease")
    axes[0].set_xlabel("Outer iteration")
    axes[0].set_ylabel("Fraction")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    for method, group in out.groupby("method"):
        axes[1].plot(group["outer_iteration"], group["actor_update_norm"], label=f"{method} actor")
        axes[1].plot(group["outer_iteration"], group["logstd_update_norm"], linestyle=":", label=f"{method} logstd")
        axes[1].plot(group["outer_iteration"], group["critic_update_norm"], linestyle="--", label=f"{method} critic")
    axes[1].set_title("Block update norms")
    axes[1].set_xlabel("Outer iteration")
    axes[1].set_ylabel("Update norm")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_root / "plots" / "M13_block_responsibility.png", dpi=200)
    plt.close(fig)

    summary = (
        out.groupby("method", as_index=False)[
            [
                "actor_V_decrease_fraction",
                "logstd_V_decrease_fraction",
                "critic_V_decrease_fraction",
                "actor_update_norm",
                "logstd_update_norm",
                "critic_update_norm",
                "actual_V_change",
            ]
        ]
        .mean()
    )
    qp = summary[summary["method"] == "proposed_qp_rawFG_eta1_cap003"].iloc[0]
    lines = [
        "# M13 Block Responsibility Report",
        "",
        f"1. Is QP mostly reducing critic/log_std field norm instead of actor mean field norm? {'Yes' if qp['critic_V_decrease_fraction'] > qp['actor_V_decrease_fraction'] else 'No'}",
        f"2. Is actor mean update too weak? {'Yes' if qp['actor_update_norm'] < qp['critic_update_norm'] else 'No'}",
        "3. Would actor_mean-only QP be a plausible next experiment? Yes" if qp["critic_V_decrease_fraction"] > qp["actor_V_decrease_fraction"] else "3. Would actor_mean-only QP be a plausible next experiment? Possibly, but not strongly indicated",
        "",
        frame_to_text(summary),
    ]
    (output_root / "M13_block_responsibility_report.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def write_root_cause_report(
    output_root: pathlib.Path,
    reward_proxy_df: pd.DataFrame,
    substitution_df: pd.DataFrame,
    block_df: pd.DataFrame,
) -> None:
    proposed = reward_proxy_df[reward_proxy_df["method"] == "proposed_qp_rawFG_eta1_cap003"]
    pivot = proposed[proposed["candidate_type"].isin(["method_mean", "best_sample", "egm_mean", "ppm_mean"])].pivot(
        index="state_id", columns="candidate_type", values="immediate_reward"
    ).dropna()
    mean_reward = float(pivot["method_mean"].mean())
    best_reward = float(pivot["best_sample"].mean())
    egm_reward = float(pivot["egm_mean"].mean())
    ppm_reward = float(pivot["ppm_mean"].mean())
    sub_mean = substitution_df.groupby("variant", as_index=False)["episode_return"].mean().set_index("variant")["episode_return"]
    block_summary = (
        block_df.groupby("method", as_index=False)[["actor_V_decrease_fraction", "critic_V_decrease_fraction", "logstd_V_decrease_fraction"]]
        .mean()
    )
    qp_block = block_summary[block_summary["method"] == "proposed_qp_rawFG_eta1_cap003"].iloc[0]

    labels: List[str] = []
    if best_reward > mean_reward:
        labels.append("A. proposed mean action is bad")
        labels.append("B. proposed distribution contains good sampled actions")
    if qp_block["critic_V_decrease_fraction"] > qp_block["actor_V_decrease_fraction"]:
        labels.append("C. QP reduces V mostly outside actor mean block")
    if egm_reward > mean_reward or ppm_reward > mean_reward:
        labels.append("D. EGM/PPM mean actions are better aligned with reward")
    if sub_mean["normal_stochastic"] > sub_mean["normal_deterministic"]:
        labels.append("E. stochastic deployment would make proposed look better")
    if sub_mean["egm_mean_substitution"] > sub_mean["normal_deterministic"]:
        labels.append("F. deterministic deployment needs actor-mean-specific objective")

    lines = [
        "# M13 Action Mean Root Cause Report",
        "",
        "## Classification",
        *[f"- {label}" for label in labels],
        "",
        "## Key evidence",
        f"- proposed_qp mean one-step reward proxy: {mean_reward:.6f}",
        f"- proposed_qp best-sample one-step reward proxy: {best_reward:.6f}",
        f"- EGM mean one-step reward proxy on same states: {egm_reward:.6f}",
        f"- PPM mean one-step reward proxy on same states: {ppm_reward:.6f}",
        f"- substitution returns, deterministic vs stochastic: {sub_mean['normal_deterministic']:.3f} vs {sub_mean['normal_stochastic']:.3f}",
        f"- substitution return with EGM mean: {sub_mean['egm_mean_substitution']:.3f}",
        f"- block responsibility actor vs critic: {qp_block['actor_V_decrease_fraction']:.3f} vs {qp_block['critic_V_decrease_fraction']:.3f}",
    ]
    (output_root / "M13_action_mean_root_cause_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    (output_root / "plots").mkdir(parents=True, exist_ok=True)
    method_runs = load_method_runs(output_root)
    reward_proxy_df = run_action_reward_proxy_audit(
        method_runs,
        output_root,
        device=args.device,
        n_probe_states=args.n_probe_states,
        n_samples=args.n_samples,
        eval_seed=args.eval_seed,
        reference_method=args.reference_method,
    )
    substitution_df = run_action_substitution_eval(
        method_runs,
        output_root,
        device=args.device,
        n_eval_episodes=args.n_episodes,
        eval_seed=args.eval_seed,
        best_of_k=args.best_of_k,
    )
    block_df = run_block_responsibility_audit(output_root)
    write_root_cause_report(output_root, reward_proxy_df, substitution_df, block_df)


if __name__ == "__main__":
    main()
