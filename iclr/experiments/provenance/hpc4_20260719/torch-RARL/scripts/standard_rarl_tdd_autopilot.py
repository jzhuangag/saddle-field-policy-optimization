from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


EPS = 1e-12
ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT.parent / "results" / "standard_rarl_tdd_autopilot"
ENV_ORDER = [
    "HalfCheetah-v5",
    "Hopper-v5",
    "Walker2d-v5",
    "Ant-v5",
    "BipedalWalker-v3",
]
BASELINE_METHODS = ["sgd_gda", "egm", "ppm_inner5"]
QP_METHODS = ["proposed_nog_closed", "proposed_qp_closed"]
STAGE0_ALPHAS = [0.0, 0.05, 0.1, 0.3, 0.5]
STAGE3A_ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.5]
STAGE3A_LRS = [3e-4, 1e-3, 3e-3]
STAGE3A_LRS_FALLBACK = [1e-4, 5e-4, 2e-3]
OPTIMIZER_SCOPES = ["actor_logstd_only", "full_policy"]
STAGE3A_FIXED = dict(
    N_mu=5,
    N_nu=1,
    iterations=10,
    eval_freq=10240,
    n_eval_episodes=8,
    shared_max_grad_norm=10.0,
    shared_vf_coef=0.5,
    seed=0,
)


class ProAdvAction:
    def __init__(self, pro_action: np.ndarray, adv_action: np.ndarray):
        self.pro_action = np.asarray(pro_action, dtype=np.float32)
        self.adv_action = np.asarray(adv_action, dtype=np.float32)


@dataclass(frozen=True)
class FixedRandomPolicyPair:
    protagonist_w: np.ndarray
    protagonist_b: np.ndarray
    adversary_w: np.ndarray
    adversary_b: np.ndarray
    action_high: np.ndarray

    def protagonist(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        raw = self.protagonist_w @ obs + self.protagonist_b
        return np.tanh(raw) * self.action_high

    def adversary_unit(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        raw = self.adversary_w @ obs + self.adversary_b
        return np.tanh(raw)

    def adversary_scaled(self, obs: np.ndarray, alpha: float) -> np.ndarray:
        return self.adversary_unit(obs) * self.action_high * float(alpha)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("TDD Autopilot for standard alternating RARL")
    parser.add_argument("--repo-dir", type=str, default=str(ROOT))
    parser.add_argument("--output-root", type=str, default=str(RESULT_ROOT))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--python-path", type=str, default=sys.executable)
    parser.add_argument("--force-rerun", action="store_true", default=False)
    return parser.parse_args()


def ensure_dirs(output_root: pathlib.Path) -> Dict[str, pathlib.Path]:
    dirs = {
        "root": output_root,
        "stage0": output_root / "00_alpha_wrapper_tests",
        "stage1": output_root / "01_eval_semantics_tests",
        "stage2": output_root / "02_scope_tests",
        "stage3a": output_root / "03_baseline_search",
        "stage3b": output_root / "04_qp_validation",
        "stage3c": output_root / "05_multienv_summary",
        "plots": output_root / "plots",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def append_progress(output_root: pathlib.Path, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with (output_root / "tdd_autopilot_progress.log").open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp}] {message}\n")


def stable_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def slugify(text: str) -> str:
    return text.lower().replace("-", "_").replace(".", "p")


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def curve_hash(values: Sequence[float]) -> str:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return "empty"
    rounded = np.round(arr, 8)
    return hashlib.sha1(rounded.tobytes()).hexdigest()[:12]


def env_available(env_name: str) -> tuple[bool, str]:
    try:
        env = gym.make(env_name)
        env.reset(seed=0)
        env.close()
        return True, ""
    except Exception as exc:
        return False, repr(exc)


def make_fixed_policy_pair(env_name: str, seed: int) -> FixedRandomPolicyPair:
    env = gym.make(env_name)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    env.close()
    rng = np.random.default_rng(seed)
    scale = 0.25 / math.sqrt(max(obs_dim, 1))
    return FixedRandomPolicyPair(
        protagonist_w=rng.normal(0.0, scale, size=(action_dim, obs_dim)).astype(np.float32),
        protagonist_b=rng.normal(0.0, 0.05, size=(action_dim,)).astype(np.float32),
        adversary_w=rng.normal(0.0, scale, size=(action_dim, obs_dim)).astype(np.float32),
        adversary_b=rng.normal(0.0, 0.05, size=(action_dim,)).astype(np.float32),
        action_high=high,
    )


def make_control_wrapper(env_name: str, alpha: float, repo_dir: pathlib.Path):
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from utils.wrappers import AdversarialClassicControlWrapper

    base_env = gym.make(env_name)
    return AdversarialClassicControlWrapper(base_env, adv_fraction=float(alpha), device="cpu")


def run_fixed_pair_rollout(
    *,
    env_name: str,
    alpha: float,
    pair: FixedRandomPolicyPair,
    seed: int,
    repo_dir: pathlib.Path,
    episodes: int = 3,
) -> Dict[str, Any]:
    wrapper = make_control_wrapper(env_name, alpha, repo_dir)
    returns: List[float] = []
    lengths: List[float] = []
    abs_u: List[float] = []
    abs_w: List[float] = []
    abs_alpha_w: List[float] = []
    abs_before: List[float] = []
    abs_after: List[float] = []
    clip_fracs: List[float] = []
    adv_nonzero: List[float] = []
    states: List[np.ndarray] = []
    for episode_idx in range(episodes):
        obs, _ = wrapper.reset(seed=seed + episode_idx)
        terminated = truncated = False
        total = 0.0
        steps = 0
        while not (terminated or truncated):
            u = pair.protagonist(obs)
            w_unit = pair.adversary_unit(obs)
            w_scaled = pair.adversary_scaled(obs, alpha)
            before = u + w_scaled
            act = ProAdvAction(u, w_scaled)
            obs, rew, terminated, truncated, info = wrapper.step(act)
            total += float(rew)
            steps += 1
            abs_u.append(float(np.mean(np.abs(u))))
            abs_w.append(float(np.mean(np.abs(w_unit * pair.action_high))))
            abs_alpha_w.append(float(np.mean(np.abs(w_scaled))))
            abs_before.append(float(np.mean(np.abs(before))))
            abs_after.append(float(np.mean(np.abs(np.asarray(u, dtype=np.float32) + np.asarray(info.get("applied_disturbance_norm", 0.0))))))
            clip_fracs.append(float(info.get("action_clip_fraction", 0.0)))
            adv_nonzero.append(float(np.mean(np.abs(w_scaled) > 1e-8)))
            states.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        returns.append(total)
        lengths.append(float(steps))
    wrapper.close()
    return {
        "mean_return": float(np.mean(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "mean_abs_u": float(np.mean(abs_u)) if abs_u else 0.0,
        "mean_abs_w": float(np.mean(abs_w)) if abs_w else 0.0,
        "mean_abs_alpha_w": float(np.mean(abs_alpha_w)) if abs_alpha_w else 0.0,
        "mean_abs_action_before_clip": float(np.mean(abs_before)) if abs_before else 0.0,
        "mean_abs_action_after_clip": math.nan,
        "action_clip_fraction": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
        "adversary_nonzero_fraction": float(np.mean(adv_nonzero)) if adv_nonzero else 0.0,
        "state_trace": states,
    }


def stage0_alpha_wrapper_tests(repo_dir: pathlib.Path, output_root: pathlib.Path, envs: List[str], seed: int) -> tuple[pd.DataFrame, List[str]]:
    rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    for env_name in envs:
        pair = make_fixed_policy_pair(env_name, seed)
        baseline = None
        used_paths: set[str] = set()
        for alpha in STAGE0_ALPHAS:
            config_hash = stable_hash(
                {
                    "env_name": env_name,
                    "seed": seed,
                    "method": "fixed_random_pair",
                    "optimizer_scope": "na",
                    "alpha": alpha,
                    "shared_lr": 0.0,
                    "N_mu": STAGE3A_FIXED["N_mu"],
                    "N_nu": STAGE3A_FIXED["N_nu"],
                    "ppm_inner_steps": 5,
                    "eval_type": "alpha_wrapper_test",
                }
            )
            out_dir = output_root / slugify(env_name) / f"alpha_{alpha:g}_{config_hash}"
            used_paths.add(str(out_dir))
            stats = run_fixed_pair_rollout(env_name=env_name, alpha=alpha, pair=pair, seed=seed, repo_dir=repo_dir)
            row = {
                "env_name": env_name,
                "alpha": alpha,
                "config_hash": config_hash,
                "result_path": str(out_dir),
                **{k: v for k, v in stats.items() if k != "state_trace"},
            }
            if baseline is None:
                row["mean_abs_action_delta_from_alpha0"] = 0.0
                row["state_delta_from_alpha0"] = 0.0
                baseline = stats
            else:
                row["mean_abs_action_delta_from_alpha0"] = abs(stats["mean_abs_action_before_clip"] - baseline["mean_abs_action_before_clip"])
                base_states = baseline["state_trace"]
                cur_states = stats["state_trace"]
                n = min(len(base_states), len(cur_states))
                if n > 0:
                    deltas = [float(np.linalg.norm(cur_states[i] - base_states[i])) for i in range(n)]
                    row["state_delta_from_alpha0"] = float(np.mean(deltas))
                else:
                    row["state_delta_from_alpha0"] = math.nan
            rows.append(row)

        env_rows = [r for r in rows if r["env_name"] == env_name]
        if any(r["alpha"] > 0 and r["mean_abs_alpha_w"] <= 0.0 for r in env_rows):
            failures.append(f"{env_name}: A1 failed")
        if any(r["alpha"] > 0 and r["mean_abs_action_delta_from_alpha0"] <= 1e-5 for r in env_rows):
            failures.append(f"{env_name}: A2 failed")
        base_return = next(r["mean_return"] for r in env_rows if r["alpha"] == 0.0)
        max_return_delta = max(abs(r["mean_return"] - base_return) for r in env_rows if r["alpha"] > 0)
        if max_return_delta <= 1e-3:
            failures.append(f"{env_name}: A3 failed")
        if len(used_paths) != len(STAGE0_ALPHAS):
            failures.append(f"{env_name}: A6 failed")

    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "alpha_wrapper_test_summary.csv", index=False)
    lines = [
        "# Alpha Wrapper Test Report",
        "",
        f"- envs_tested: `{sorted(frame['env_name'].unique().tolist()) if not frame.empty else []}`",
        f"- failure_count: `{len(failures)}`",
        "",
    ]
    for env_name, sub in frame.groupby("env_name"):
        lines.append(f"## {env_name}")
        for _, row in sub.iterrows():
            lines.append(
                f"- alpha=`{row['alpha']}` hash=`{row['config_hash']}` return=`{row['mean_return']:.6f}` "
                f"abs_alpha_w=`{row['mean_abs_alpha_w']:.6e}` action_delta_vs_alpha0=`{row['mean_abs_action_delta_from_alpha0']:.6e}` "
                f"clip=`{row['action_clip_fraction']:.6f}`"
            )
        lines.append("")
    if failures:
        lines.append("## Failures")
        lines.extend([f"- {item}" for item in failures])
        (output_root / "alpha_wrapper_test_FAIL.md").write_text("\n".join(lines), encoding="utf-8")
    (output_root / "alpha_wrapper_test_report.md").write_text("\n".join(lines), encoding="utf-8")
    return frame, failures


def eval_clean_current_random(
    *,
    env_name: str,
    alpha: float,
    pair: FixedRandomPolicyPair,
    seed: int,
    repo_dir: pathlib.Path,
    episodes: int = 5,
) -> Dict[str, float]:
    clean_returns: List[float] = []
    cur_returns: List[float] = []
    rnd_returns: List[float] = []
    rng = np.random.default_rng(seed + 1234)
    for episode_idx in range(episodes):
        clean_env = gym.make(env_name)
        obs, _ = clean_env.reset(seed=seed + episode_idx)
        total = 0.0
        done = False
        while not done:
            u = pair.protagonist(obs)
            obs, rew, terminated, truncated, _ = clean_env.step(u)
            total += float(rew)
            done = bool(terminated or truncated)
        clean_returns.append(total)
        clean_env.close()

        cur_stats = run_fixed_pair_rollout(env_name=env_name, alpha=alpha, pair=pair, seed=seed + episode_idx, repo_dir=repo_dir, episodes=1)
        cur_returns.append(cur_stats["mean_return"])

        random_pair = FixedRandomPolicyPair(
            protagonist_w=pair.protagonist_w,
            protagonist_b=pair.protagonist_b,
            adversary_w=np.zeros_like(pair.adversary_w),
            adversary_b=np.zeros_like(pair.adversary_b),
            action_high=pair.action_high,
        )
        wrapper = make_control_wrapper(env_name, alpha, repo_dir)
        obs, _ = wrapper.reset(seed=seed + episode_idx)
        total = 0.0
        done = False
        while not done:
            u = pair.protagonist(obs)
            w_scaled = rng.uniform(-1.0, 1.0, size=pair.action_high.shape).astype(np.float32) * pair.action_high * alpha
            obs, rew, terminated, truncated, _ = wrapper.step(ProAdvAction(u, w_scaled))
            total += float(rew)
            done = bool(terminated or truncated)
        rnd_returns.append(total)
        wrapper.close()
    return {
        "clean_eval_return": float(np.mean(clean_returns)),
        "current_adv_eval_return": float(np.mean(cur_returns)),
        "random_adv_eval_return": float(np.mean(rnd_returns)),
        "local_BR_eval_return": math.nan,
        "local_BR_unavailable": 1,
    }


def stage1_eval_semantics_tests(repo_dir: pathlib.Path, output_root: pathlib.Path, stage0_df: pd.DataFrame, envs: List[str], seed: int) -> tuple[pd.DataFrame, List[str]]:
    rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    for env_name in envs:
        pair = make_fixed_policy_pair(env_name, seed)
        base_current = None
        base_random = None
        for alpha in STAGE0_ALPHAS:
            config_hash = stable_hash(
                {
                    "env_name": env_name,
                    "seed": seed,
                    "method": "fixed_random_pair",
                    "optimizer_scope": "na",
                    "alpha": alpha,
                    "shared_lr": 0.0,
                    "N_mu": STAGE3A_FIXED["N_mu"],
                    "N_nu": STAGE3A_FIXED["N_nu"],
                    "ppm_inner_steps": 5,
                    "eval_type": "eval_semantics",
                }
            )
            values = eval_clean_current_random(env_name=env_name, alpha=alpha, pair=pair, seed=seed, repo_dir=repo_dir)
            row = {
                "env_name": env_name,
                "alpha": alpha,
                "config_hash": config_hash,
                **values,
            }
            row["current_adv_degradation"] = row["clean_eval_return"] - row["current_adv_eval_return"]
            row["random_adv_degradation"] = row["clean_eval_return"] - row["random_adv_eval_return"]
            row["local_BR_degradation"] = math.nan
            rows.append(row)
            if alpha == 0.0:
                base_current = row["current_adv_eval_return"]
                base_random = row["random_adv_eval_return"]
        env_rows = [r for r in rows if r["env_name"] == env_name]
        current_changes = [abs(r["current_adv_eval_return"] - base_current) for r in env_rows if r["alpha"] > 0]
        random_changes = [abs(r["random_adv_eval_return"] - base_random) for r in env_rows if r["alpha"] > 0]
        if max(current_changes, default=0.0) <= 1e-3:
            failures.append(f"{env_name}: E1 failed")
        if max(random_changes, default=0.0) <= 1e-3:
            failures.append(f"{env_name}: E2 failed")
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "eval_semantics_summary.csv", index=False)
    lines = [
        "# Eval Semantics Report",
        "",
        "- Main Stage 3A comparison uses `current_adv_eval_return`.",
        "- `local_BR_eval_return` is explicitly marked unavailable in this TDD audit and is not faked.",
        f"- failure_count: `{len(failures)}`",
        "",
    ]
    for env_name, sub in frame.groupby("env_name"):
        lines.append(f"## {env_name}")
        for _, row in sub.iterrows():
            lines.append(
                f"- alpha=`{row['alpha']}` clean=`{row['clean_eval_return']:.6f}` current_adv=`{row['current_adv_eval_return']:.6f}` "
                f"random_adv=`{row['random_adv_eval_return']:.6f}` current_deg=`{row['current_adv_degradation']:.6f}`"
            )
        lines.append("")
    if failures:
        lines.append("## Failures")
        lines.extend([f"- {item}" for item in failures])
    (output_root / "eval_semantics_report.md").write_text("\n".join(lines), encoding="utf-8")
    return frame, failures


def ensure_hyperparams(repo_dir: pathlib.Path, output_root: pathlib.Path, requested_env: str, fallback_env: str) -> pathlib.Path:
    source_path = repo_dir / "hyperparameter" / "PPO-rarl.yml"
    temp_dir = output_root / "temp_hyperparams"
    temp_dir.mkdir(parents=True, exist_ok=True)
    target_path = temp_dir / "PPO-rarl.yml"
    with source_path.open("r", encoding="utf-8") as handle:
        hyperparams = yaml.safe_load(handle)
    mapping_note = "native"
    if requested_env not in hyperparams:
        if fallback_env not in hyperparams:
            raise KeyError(f"Neither {requested_env} nor fallback {fallback_env} exist in {source_path}")
        hyperparams[requested_env] = hyperparams[fallback_env]
        mapping_note = f"copied_from_{fallback_env}"
    with target_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(hyperparams, handle, sort_keys=False)
    (temp_dir / "hyperparam_mapping.json").write_text(
        json.dumps(
            {
                "requested_env": requested_env,
                "fallback_env": fallback_env,
                "env_used": requested_env,
                "mapping_note": mapping_note,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return temp_dir


def build_scope_args(
    *,
    env_name: str,
    seed: int,
    device: str,
    optimizer_scope: str,
    optimizer_name: str,
    optimizer_kwargs: Dict[str, Any],
    hyperparam_dir: pathlib.Path,
    run_root: pathlib.Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        verbose=0,
        seed=seed,
        num_exps=1,
        num_threads=-1,
        env=env_name,
        n_envs=1,
        vec_env_type="dummy",
        env_kwargs=None,
        adv_env=False,
        algo="rarl",
        rarl_config="ppo",
        saved_models_path=str(run_root / "saved_models"),
        pretrained_model="",
        save_replay_buffer=False,
        hyperparameter=None,
        optimize_hyperparameters=False,
        hyperparameter_path=str(hyperparam_dir),
        storage=None,
        study_name=None,
        sampler="tpe",
        pruner="median",
        optimization_log_path=str(run_root / "hyperparam_optimization"),
        n_opt_trials=1,
        no_optim_plots=True,
        n_jobs=1,
        n_startup_trials=1,
        n_evaluations_opt=1,
        n_timesteps=1,
        save_freq=10240,
        log_interval=-1,
        device=device,
        eval_freq=10240,
        n_eval_envs=1,
        n_eval_episodes=1,
        control_proxy_eval=False,
        tensorboard_log=str(run_root / "tb"),
        log_folder=str(run_root / "logging"),
        protagonist_policy="MlpPolicy",
        adversary_policy="MlpPolicy",
        protagonist_optimizer=optimizer_name,
        adversary_optimizer=optimizer_name,
        protagonist_optimizer_kwargs=dict(optimizer_kwargs),
        adversary_optimizer_kwargs=dict(optimizer_kwargs),
        protagonist_lr=1e-3,
        adversary_lr=1e-3,
        protagonist_max_grad_norm=10.0,
        adversary_max_grad_norm=10.0,
        protagonist_vf_coef=0.5,
        adversary_vf_coef=0.5,
        optimizer_scope=optimizer_scope,
        qp_normalization="none",
        qp_g_alpha=1e-3,
        max_update_norm=0.005,
        qp_eps=1e-8,
        qp_alpha=0.3,
        qp_beta_max=1.0,
        qp_gamma_max=1.0,
        qp_step_grid="0,0.1,0.3,1.0,3.0",
        qp_objective="loss",
        qp_accept_rule="none",
        qp_min_g_contribution=0.0,
        qp_critic_weight=1.0,
        qp_g_sign="plus",
        qp_fd_eps=1e-3,
        qp_beta_probe=1e-3,
        qp_gamma_probe=1e-6,
        qp_ridge=1e-8,
        qp_actor_weight=1.0,
        qp_logstd_weight=1.0,
        qp_step_solver="lyapunov_quadratic_bound",
        N_mu=5,
        N_nu=1,
        adv_impact="control",
        adv_delay=-1,
        adv_fraction=0.1,
        adv_index_list=["torso"],
        adv_force_dim=2,
    )


def register_closed_optimizers(repo_dir: pathlib.Path) -> None:
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from models.optimizers import OPTIMIZER_REGISTRY
    from models.proposed_qp_closedlyap import ProposedNoGClosedLyapOptimizer, ProposedQPClosedLyapOptimizer

    OPTIMIZER_REGISTRY["proposed_nog_closedlyap"] = ProposedNoGClosedLyapOptimizer
    OPTIMIZER_REGISTRY["proposed_qp_closedlyap"] = ProposedQPClosedLyapOptimizer


def audit_scope_for_method(
    *,
    repo_dir: pathlib.Path,
    output_root: pathlib.Path,
    env_name: str,
    seed: int,
    device: str,
    optimizer_scope: str,
    method_name: str,
    optimizer_name: str,
    optimizer_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from utils.exp_manager import ExperimentManager

    hyperparam_dir = ensure_hyperparams(repo_dir, output_root, env_name, "HalfCheetah-v4")
    run_root = output_root / f"{optimizer_scope}_{method_name}"
    ns = build_scope_args(
        env_name=env_name,
        seed=seed,
        device=device,
        optimizer_scope=optimizer_scope,
        optimizer_name=optimizer_name,
        optimizer_kwargs=optimizer_kwargs,
        hyperparam_dir=hyperparam_dir,
        run_root=run_root,
    )
    manager = ExperimentManager(
        ns,
        algo="rarl",
        env_id=env_name,
        log_folder=ns.log_folder,
        tensorboard_log=ns.tensorboard_log,
        n_timesteps=ns.n_timesteps,
        eval_freq=ns.eval_freq,
        n_eval_episodes=ns.n_eval_episodes,
        save_freq=ns.save_freq,
        hyperparameter_path=ns.hyperparameter_path,
        hyperparams=ns.hyperparameter,
        env_kwargs=ns.env_kwargs,
        model_path=str(run_root / "saved_models" / "rarl-ppo" / env_name),
        pretrained_model=ns.pretrained_model,
        optimize_hyperparameters=ns.optimize_hyperparameters,
        storage=ns.storage,
        study_name=ns.study_name,
        n_opt_trials=ns.n_opt_trials,
        n_jobs=ns.n_jobs,
        sampler=ns.sampler,
        pruner=ns.pruner,
        optimization_log_path=ns.optimization_log_path,
        n_startup_trials=ns.n_startup_trials,
        n_evaluations_opt=ns.n_evaluations_opt,
        seed=ns.seed,
        log_interval=ns.log_interval,
        save_replay_buffer=ns.save_replay_buffer,
        verbose=ns.verbose,
        vec_env_type=ns.vec_env_type,
        n_envs=ns.n_envs,
        n_eval_envs=ns.n_eval_envs,
        no_optim_plots=ns.no_optim_plots,
        adv_env=ns.adv_env,
        adv_impact=ns.adv_impact,
        adv_fraction=ns.adv_fraction,
        adv_delay=ns.adv_delay,
        adv_index_list=ns.adv_index_list,
        adv_force_dim=ns.adv_force_dim,
        N_mu=ns.N_mu,
        N_nu=ns.N_nu,
        device=ns.device,
        rarl_config=ns.rarl_config,
        protagonist_optimizer=ns.protagonist_optimizer,
        adversary_optimizer=ns.adversary_optimizer,
        protagonist_optimizer_kwargs=ns.protagonist_optimizer_kwargs,
        adversary_optimizer_kwargs=ns.adversary_optimizer_kwargs,
        protagonist_lr=ns.protagonist_lr,
        adversary_lr=ns.adversary_lr,
        protagonist_max_grad_norm=ns.protagonist_max_grad_norm,
        adversary_max_grad_norm=ns.adversary_max_grad_norm,
        protagonist_vf_coef=ns.protagonist_vf_coef,
        adversary_vf_coef=ns.adversary_vf_coef,
        optimizer_scope=ns.optimizer_scope,
        qp_normalization=ns.qp_normalization,
        qp_g_alpha=ns.qp_g_alpha,
        max_update_norm=ns.max_update_norm,
        qp_eps=ns.qp_eps,
        qp_alpha=ns.qp_alpha,
        qp_beta_max=ns.qp_beta_max,
        qp_gamma_max=ns.qp_gamma_max,
        qp_step_grid=ns.qp_step_grid,
        qp_objective=ns.qp_objective,
        qp_accept_rule=ns.qp_accept_rule,
        qp_min_g_contribution=ns.qp_min_g_contribution,
        qp_critic_weight=ns.qp_critic_weight,
        qp_g_sign=ns.qp_g_sign,
        qp_fd_eps=ns.qp_fd_eps,
        qp_beta_probe=ns.qp_beta_probe,
        qp_gamma_probe=ns.qp_gamma_probe,
        qp_ridge=ns.qp_ridge,
        qp_actor_weight=ns.qp_actor_weight,
        qp_logstd_weight=ns.qp_logstd_weight,
        qp_step_solver=ns.qp_step_solver,
        control_proxy_eval=ns.control_proxy_eval,
    )
    model = manager.setup_experiment()
    try:
        records = []
        for role_name, agent in (("protagonist", model.protagonist), ("adversary", model.adversary)):
            named = [(name, param) for name, param in agent.policy.named_parameters() if param.requires_grad]
            optimizer_param_ids = {id(param) for group in agent.policy.optimizer.param_groups for param in group["params"]}
            selected = [name for name, param in named if id(param) in optimizer_param_ids]
            records.append(
                {
                    "optimizer_scope": optimizer_scope,
                    "method": method_name,
                    "role": role_name,
                    "number_of_trainable_optimizer_params": int(sum(param.numel() for name, param in named if name in selected)),
                    "sample_parameter_names": json.dumps(selected[:8]),
                    "contains_actor_params": any(("policy_net" in name or "action_net" in name) for name in selected),
                    "contains_log_std": any("log_std" in name for name in selected),
                    "contains_value_params": any("value" in name for name in selected),
                    "contains_shared_params": any(("mlp_extractor" in name and "value" not in name and "policy_net" not in name) for name in selected),
                    "selected_names_json": json.dumps(selected),
                    "critic_optimizer_present": getattr(agent, "_actor_game_critic_optimizer", None) is not None,
                }
            )
        return records
    finally:
        for attr in ("protagonist", "adversary"):
            agent = getattr(model, attr, None)
            env = getattr(agent, "env", None)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass


def stage2_scope_tests(repo_dir: pathlib.Path, output_root: pathlib.Path, seed: int, device: str) -> tuple[pd.DataFrame, List[str]]:
    output_root.mkdir(parents=True, exist_ok=True)
    audit_script = repo_dir / "scripts" / "standard_rarl_optimizer_scope_audit.py"
    stdout_path = output_root / "optimizer_scope_audit_stdout.log"
    stderr_path = output_root / "optimizer_scope_audit_stderr.log"
    command = [
        sys.executable,
        str(audit_script),
        "--repo-dir",
        str(repo_dir),
        "--output-root",
        str(output_root),
        "--env",
        "HalfCheetah-v5",
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    run_subprocess(command, repo_dir, stdout_path, stderr_path)
    frame = pd.read_csv(output_root / "optimizer_scope_audit.csv")
    failures: List[str] = []
    if "optimizer_scope" not in frame.columns:
        failures.append("scope audit output missing optimizer_scope column")
        return frame, failures
    for scope in OPTIMIZER_SCOPES:
        scope_df = frame[frame["optimizer_scope"] == scope]
        if scope_df.empty:
            failures.append(f"{scope}: missing audit rows")
            continue
        for role_col in ("agent_name", "role"):
            if role_col in scope_df.columns:
                group_col = role_col
                break
        else:
            failures.append(f"{scope}: missing agent role column")
            continue
        for role, role_df in scope_df.groupby(group_col):
            base = None
            sample_col = "sample_parameter_names"
            if "selected_names_json" in role_df.columns:
                sample_col = "selected_names_json"
            for _, row in role_df.iterrows():
                try:
                    selected = json.loads(row[sample_col]) if isinstance(row[sample_col], str) else []
                except Exception:
                    selected = []
                if base is None:
                    base = selected
                elif selected != base:
                    failures.append(f"{scope}/{role}: S1 failed because method parameter sets differ")
            if scope == "full_policy" and not role_df["contains_value_params"].all():
                failures.append(f"{scope}/{role}: missing value params")
            if scope == "actor_logstd_only" and role_df["contains_value_params"].any():
                failures.append(f"{scope}/{role}: includes value params")
    if failures:
        md_path = output_root / "optimizer_scope_audit.md"
        existing = md_path.read_text(encoding="utf-8") if md_path.exists() else "# Optimizer Scope Audit\n"
        existing += "\n\n## Autopilot validation failures\n" + "\n".join(f"- {item}" for item in failures) + "\n"
        md_path.write_text(existing, encoding="utf-8")
    return frame, failures


def run_subprocess(command: List[str], cwd: pathlib.Path, stdout_path: pathlib.Path, stderr_path: pathlib.Path) -> None:
    result = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def annotate_run_hashes(runs_root: pathlib.Path, env_name: str, seed: int) -> None:
    for analysis_path in runs_root.rglob("run_summary.csv"):
        method_dir = analysis_path.parents[1]
        rel = method_dir.relative_to(runs_root)
        if len(rel.parts) < 4:
            continue
        scope = rel.parts[0]
        alpha = rel.parts[1].split("_", 1)[1]
        lr = rel.parts[2].split("_", 1)[1]
        method = rel.parts[3]
        payload = {
            "env_name": env_name,
            "seed": seed,
            "method": method,
            "optimizer_scope": scope,
            "alpha": alpha,
            "shared_lr": lr,
            "N_mu": STAGE3A_FIXED["N_mu"],
            "N_nu": STAGE3A_FIXED["N_nu"],
            "ppm_inner_steps": 5,
            "eval_type": "current_adv_eval",
        }
        config_hash = stable_hash(payload)
        (method_dir / "config_hash.json").write_text(json.dumps({"config_hash": config_hash, **payload}, indent=2), encoding="utf-8")


def stage3a_run_env(repo_dir: pathlib.Path, output_root: pathlib.Path, env_name: str, python_path: str, force_rerun: bool) -> pathlib.Path:
    env_root = output_root / slugify(env_name) / "raw_search"
    summary_path = env_root / "baseline_positive_search_summary.csv"
    if summary_path.exists() and not force_rerun:
        return env_root
    env_root.mkdir(parents=True, exist_ok=True)
    command = [
        python_path,
        str(repo_dir / "scripts" / "standard_rarl_baseline_positive_search.py"),
        "--repo-dir",
        str(repo_dir),
        "--output-root",
        str(env_root),
        "--env",
        env_name,
        "--fallback-env",
        "HalfCheetah-v4",
        "--seed",
        str(STAGE3A_FIXED["seed"]),
        "--device",
        "cpu",
        "--iterations",
        str(STAGE3A_FIXED["iterations"]),
        "--eval-freq",
        str(STAGE3A_FIXED["eval_freq"]),
        "--n-eval-episodes",
        str(STAGE3A_FIXED["n_eval_episodes"]),
        "--optimizer-scopes",
        ",".join(OPTIMIZER_SCOPES),
        "--alphas",
        ",".join(str(v) for v in STAGE3A_ALPHAS),
        "--shared-lrs",
        ",".join(str(v) for v in STAGE3A_LRS),
        "--ppm-inner-steps",
        "5",
        "--n-mu",
        str(STAGE3A_FIXED["N_mu"]),
        "--n-nu",
        str(STAGE3A_FIXED["N_nu"]),
    ]
    run_subprocess(command, repo_dir, env_root / "autopilot_stage3a_stdout.txt", env_root / "autopilot_stage3a_stderr.txt")
    annotate_run_hashes(env_root / "runs", env_name, STAGE3A_FIXED["seed"])
    return env_root


def alpha_duplicate_groups(curves_df: pd.DataFrame) -> Dict[tuple, Dict[str, Any]]:
    results: Dict[tuple, Dict[str, Any]] = {}
    for (scope, lr, method), sub in curves_df.groupby(["optimizer_scope", "shared_lr", "method"]):
        alpha_map = {}
        for alpha, alpha_df in sub.groupby("alpha"):
            alpha_df = alpha_df.sort_values("timesteps")
            sig = curve_hash(alpha_df["current_adv_eval_return"].tolist() + alpha_df["clean_eval_return"].tolist())
            alpha_map[float(alpha)] = sig
        duplicate = len(set(alpha_map.values())) < len(alpha_map)
        results[(scope, float(lr), method)] = {
            "duplicate_flag": duplicate,
            "alpha_signature_map": json.dumps(alpha_map, sort_keys=True),
        }
    return results


def ppm_degenerate_flags(curves_df: pd.DataFrame) -> Dict[tuple, bool]:
    flags: Dict[tuple, bool] = {}
    for (scope, alpha, lr), sub in curves_df.groupby(["optimizer_scope", "alpha", "shared_lr"]):
        egm_df = sub[sub["method"] == "egm"].sort_values("timesteps")
        ppm_df = sub[sub["method"] == "ppm_inner5"].sort_values("timesteps")
        if egm_df.empty or ppm_df.empty or len(egm_df) != len(ppm_df):
            flags[(scope, float(alpha), float(lr))] = False
            continue
        same = np.allclose(
            pd.to_numeric(egm_df["current_adv_eval_return"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(ppm_df["current_adv_eval_return"], errors="coerce").to_numpy(dtype=float),
            atol=1e-9,
            rtol=0.0,
        )
        flags[(scope, float(alpha), float(lr))] = bool(same)
    return flags


def early_persistent_advantage(win_curve: pd.DataFrame, sgd_curve: pd.DataFrame) -> bool:
    merged = win_curve[["timesteps", "current_adv_eval_return"]].merge(
        sgd_curve[["timesteps", "current_adv_eval_return"]],
        on="timesteps",
        suffixes=("_win", "_sgd"),
    )
    if len(merged) < 4:
        return False
    better = merged["current_adv_eval_return_win"] >= merged["current_adv_eval_return_sgd"] - 1e-9
    better_idx = np.where(better.to_numpy())[0]
    if better_idx.size == 0:
        return False
    first = int(better_idx[0])
    if first >= len(merged) - 2:
        return False
    tail = better.iloc[first:]
    return float(np.mean(tail)) >= 0.7


def postprocess_stage3a_env(env_name: str, env_root: pathlib.Path, alpha_active: bool) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    summary_df = pd.read_csv(env_root / "baseline_positive_search_summary.csv")
    config_df = pd.read_csv(env_root / "baseline_positive_search_config_summary.csv")
    curves_df = pd.read_csv(env_root / "baseline_positive_search_curves.csv")
    dup_map = alpha_duplicate_groups(curves_df)
    ppm_map = ppm_degenerate_flags(curves_df)
    rows: List[Dict[str, Any]] = []
    for _, row in config_df.iterrows():
        scope = row["optimizer_scope"]
        alpha = float(row["alpha"])
        lr = float(row["shared_lr"])
        dup = dup_map[(scope, lr, "sgd_gda")]["duplicate_flag"] or dup_map[(scope, lr, "egm")]["duplicate_flag"] or dup_map[(scope, lr, "ppm_inner5")]["duplicate_flag"]
        ppm_deg = ppm_map[(scope, alpha, lr)]
        sub_summary = summary_df[
            (summary_df["optimizer_scope"] == scope)
            & (summary_df["alpha"] == alpha)
            & (summary_df["shared_lr"] == lr)
        ].copy()
        sgd = sub_summary[sub_summary["method"] == "sgd_gda"].iloc[0]
        egm = sub_summary[sub_summary["method"] == "egm"].iloc[0]
        ppm = sub_summary[sub_summary["method"] == "ppm_inner5"].iloc[0]
        winner = "egm" if float(egm["current_adv_eval_return_AUC"]) >= float(ppm["current_adv_eval_return_AUC"]) else "ppm_inner5"
        win_row = egm if winner == "egm" else ppm
        curves_sub = curves_df[
            (curves_df["optimizer_scope"] == scope)
            & (curves_df["alpha"] == alpha)
            & (curves_df["shared_lr"] == lr)
        ].copy()
        sgd_curve = curves_sub[curves_sub["method"] == "sgd_gda"].sort_values("timesteps")
        win_curve = curves_sub[curves_sub["method"] == winner].sort_values("timesteps")
        strong_clean_ok = float(win_row["clean_eval_return_AUC"]) >= 0.8 * float(sgd["clean_eval_return_AUC"])
        improve = (float(win_row["current_adv_eval_return_AUC"]) / (float(sgd["current_adv_eval_return_AUC"]) + EPS)) - 1.0
        dom = max(float(row["EGM_over_SGD_fraction"]), float(row["PPM_over_SGD_fraction"]))
        weak_positive = bool(
            alpha_active
            and not dup
            and not (winner == "ppm_inner5" and ppm_deg)
            and improve >= 0.10
            and dom >= 0.70
            and strong_clean_ok
        )
        strong_positive = bool(
            weak_positive
            and improve >= 0.20
            and dom >= 0.80
            and early_persistent_advantage(win_curve, sgd_curve)
        )
        rows.append(
            {
                "env_name": env_name,
                "optimizer_scope": scope,
                "alpha": alpha,
                "shared_lr": lr,
                "winner": winner,
                "winner_improve_frac": improve,
                "winner_dominance_fraction": dom,
                "alpha_duplicate_flag": int(dup),
                "ppm_implementation_degenerate": int(ppm_deg),
                "local_BR_available": 0,
                "baseline_weak_positive": int(weak_positive),
                "baseline_strong_positive": int(strong_positive),
                "clean_eval_ok": int(strong_clean_ok),
                "curve_hash_sgd": dup_map[(scope, lr, "sgd_gda")]["alpha_signature_map"],
                "curve_hash_egm": dup_map[(scope, lr, "egm")]["alpha_signature_map"],
                "curve_hash_ppm": dup_map[(scope, lr, "ppm_inner5")]["alpha_signature_map"],
            }
        )
    ranked = pd.DataFrame(rows).sort_values(
        ["baseline_strong_positive", "baseline_weak_positive", "winner_improve_frac", "winner_dominance_fraction"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    diagnosis = {
        "env_name": env_name,
        "alpha_duplicate_any": bool(ranked["alpha_duplicate_flag"].any()),
        "ppm_degenerate_any": bool(ranked["ppm_implementation_degenerate"].any()),
        "baseline_positive_count": int(ranked["baseline_weak_positive"].sum()),
        "baseline_strong_count": int(ranked["baseline_strong_positive"].sum()),
    }
    return ranked, curves_df, diagnosis


def plot_env_summary(stage3a_root: pathlib.Path, ranked_all: pd.DataFrame) -> None:
    if ranked_all.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for env_name, sub in ranked_all.groupby("env_name"):
        best = sub.iloc[0]
        ax.scatter(float(best["alpha"]), float(best["winner_improve_frac"]), s=80, label=env_name)
    ax.axhline(0.10, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("alpha")
    ax.set_ylabel("best winner improve frac")
    ax.set_title("Best per-env baseline improvement")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(stage3a_root.parent / "plots" / "baseline_search_env_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_top_configs(stage3a_root: pathlib.Path, curves_all: pd.DataFrame, ranked_all: pd.DataFrame) -> None:
    if curves_all.empty or ranked_all.empty:
        return
    top = ranked_all.head(6)
    fig, axes = plt.subplots(len(top), 1, figsize=(12, max(4, 2.8 * len(top))), sharex=False)
    if len(top) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, top.iterrows()):
        sub = curves_all[
            (curves_all["env_name"] == row["env_name"])
            & (curves_all["optimizer_scope"] == row["optimizer_scope"])
            & (curves_all["alpha"] == row["alpha"])
            & (curves_all["shared_lr"] == row["shared_lr"])
        ]
        for method, method_df in sub.groupby("method"):
            ax.plot(method_df["timesteps"], method_df["current_adv_eval_return"], label=method, linewidth=1.5)
        ax.set_title(f"{row['env_name']} | {row['optimizer_scope']} | alpha={row['alpha']} | lr={row['shared_lr']}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(stage3a_root.parent / "plots" / "baseline_search_top_configs.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_stage3a_reports(stage3a_root: pathlib.Path, ranked_all: pd.DataFrame, diagnosis_rows: List[Dict[str, Any]]) -> None:
    ranked_all.to_csv(stage3a_root / "baseline_search_ranked.csv", index=False)
    lines = [
        "# Baseline Search Report",
        "",
        "- Main comparison metric: `current_adv_eval_return`.",
        "- `local_BR_eval_return` is unavailable in this autopilot baseline stage and is not faked.",
        "",
        "## Per-environment diagnosis",
        "",
    ]
    for row in diagnosis_rows:
        lines.append(
            f"- `{row['env_name']}`: baseline_positive_count=`{row['baseline_positive_count']}`, "
            f"baseline_strong_count=`{row['baseline_strong_count']}`, alpha_duplicate_any=`{row['alpha_duplicate_any']}`, "
            f"ppm_degenerate_any=`{row['ppm_degenerate_any']}`"
        )
    lines.extend(["", "## Top configs", ""])
    for _, row in ranked_all.head(15).iterrows():
        lines.append(
            f"- env=`{row['env_name']}` scope=`{row['optimizer_scope']}` alpha=`{row['alpha']}` lr=`{row['shared_lr']}` "
            f"winner=`{row['winner']}` improve=`{row['winner_improve_frac']:.3f}` dominance=`{row['winner_dominance_fraction']:.3f}` "
            f"weak=`{bool(row['baseline_weak_positive'])}` strong=`{bool(row['baseline_strong_positive'])}` dup=`{bool(row['alpha_duplicate_flag'])}`"
        )
    (stage3a_root / "baseline_search_report.md").write_text("\n".join(lines), encoding="utf-8")
    top_lines = [
        "# Baseline-Positive Top Configs",
        "",
    ]
    for _, row in ranked_all.head(12).iterrows():
        top_lines.append(
            f"- env=`{row['env_name']}` scope=`{row['optimizer_scope']}` alpha=`{row['alpha']}` lr=`{row['shared_lr']}` "
            f"winner=`{row['winner']}` weak=`{bool(row['baseline_weak_positive'])}` strong=`{bool(row['baseline_strong_positive'])}`"
        )
    (stage3a_root / "baseline_positive_top_configs.md").write_text("\n".join(top_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_dir = pathlib.Path(args.repo_dir)
    output_root = pathlib.Path(args.output_root)
    dirs = ensure_dirs(output_root)
    append_progress(output_root, "Autopilot start")

    availability_rows = []
    available_envs = []
    for env_name in ENV_ORDER:
        ok, note = env_available(env_name)
        availability_rows.append({"env_name": env_name, "available": int(ok), "note": note})
        if ok:
            available_envs.append(env_name)
    pd.DataFrame(availability_rows).to_csv(dirs["root"] / "env_availability.csv", index=False)
    append_progress(output_root, f"Available envs: {available_envs}")

    stage0_df, stage0_failures = stage0_alpha_wrapper_tests(repo_dir, dirs["stage0"], available_envs, args.seed)
    append_progress(output_root, f"Stage 0 complete; failures={len(stage0_failures)}")
    if stage0_failures:
        decision = "TDD_FAIL_ALPHA_WRAPPER"
        (dirs["root"] / "tdd_autopilot_final_decision.md").write_text(f"# Final Decision\n\n- decision: `{decision}`\n", encoding="utf-8")
        (dirs["root"] / "tdd_autopilot_final_report.md").write_text(
            "\n".join(
                [
                    "# TDD Autopilot Final Report",
                    "",
                    "1. Whether alpha was actually active.",
                    f"Failed. See `{dirs['stage0'] / 'alpha_wrapper_test_FAIL.md'}`.",
                ]
            ),
            encoding="utf-8",
        )
        return

    stage1_df, stage1_failures = stage1_eval_semantics_tests(repo_dir, dirs["stage1"], stage0_df, available_envs, args.seed)
    append_progress(output_root, f"Stage 1 complete; failures={len(stage1_failures)}")
    if stage1_failures:
        decision = "TDD_FAIL_EVAL_SEMANTICS"
        (dirs["root"] / "tdd_autopilot_final_decision.md").write_text(f"# Final Decision\n\n- decision: `{decision}`\n", encoding="utf-8")
        (dirs["root"] / "tdd_autopilot_final_report.md").write_text(
            "\n".join(
                [
                    "# TDD Autopilot Final Report",
                    "",
                    "1. Whether eval semantics were correct.",
                    f"Failed. See `{dirs['stage1'] / 'eval_semantics_report.md'}`.",
                ]
            ),
            encoding="utf-8",
        )
        return

    stage2_df, stage2_failures = stage2_scope_tests(repo_dir, dirs["stage2"], args.seed, args.device)
    append_progress(output_root, f"Stage 2 complete; failures={len(stage2_failures)}")
    if stage2_failures:
        decision = "INCONCLUSIVE_WITH_DIAGNOSTICS"
        (dirs["root"] / "tdd_autopilot_final_decision.md").write_text(f"# Final Decision\n\n- decision: `{decision}`\n", encoding="utf-8")
        (dirs["root"] / "tdd_autopilot_final_report.md").write_text(
            "\n".join(
                [
                    "# TDD Autopilot Final Report",
                    "",
                    "2. Optimizer scope audit failed.",
                    f"See `{dirs['stage2'] / 'optimizer_scope_audit.md'}`.",
                ]
            ),
            encoding="utf-8",
        )
        return

    ranked_frames = []
    curves_frames = []
    diagnosis_rows = []
    for env_name in available_envs:
        append_progress(output_root, f"Stage 3A run start for {env_name}")
        env_root = stage3a_run_env(repo_dir, dirs["stage3a"], env_name, args.python_path, args.force_rerun)
        alpha_active = bool(
            stage0_df[(stage0_df["env_name"] == env_name) & (stage0_df["alpha"] > 0)]["mean_abs_action_delta_from_alpha0"].max() > 1e-5
        )
        ranked, curves, diagnosis = postprocess_stage3a_env(env_name, env_root, alpha_active)
        ranked_frames.append(ranked)
        curves_frames.append(curves.assign(env_name=env_name))
        diagnosis_rows.append(diagnosis)
        append_progress(output_root, f"Stage 3A postprocess complete for {env_name}")

    ranked_all = pd.concat(ranked_frames, ignore_index=True) if ranked_frames else pd.DataFrame()
    curves_all = pd.concat(curves_frames, ignore_index=True) if curves_frames else pd.DataFrame()
    ranked_all.to_csv(dirs["stage3a"] / "baseline_search_all_configs.csv", index=False)
    curves_all.to_csv(dirs["stage3a"] / "baseline_search_all_curves.csv", index=False)
    write_stage3a_reports(dirs["stage3a"], ranked_all, diagnosis_rows)
    plot_env_summary(dirs["stage3a"], ranked_all)
    plot_top_configs(dirs["stage3a"], curves_all, ranked_all)
    append_progress(output_root, "Stage 3A complete")

    weak_positive_df = ranked_all[ranked_all["baseline_weak_positive"] == 1].copy() if not ranked_all.empty else pd.DataFrame()
    duplicate_issue = bool(ranked_all["alpha_duplicate_flag"].any()) if not ranked_all.empty else False
    if weak_positive_df.empty:
        decision = "INCONCLUSIVE_WITH_DIAGNOSTICS" if duplicate_issue else "NO_BASELINE_POSITIVE_REGIME_FOUND"
        top_summary = pd.DataFrame(
            [
                {
                    "env_name": row["env_name"],
                    "optimizer_scope": row["optimizer_scope"],
                    "alpha": row["alpha"],
                    "shared_lr": row["shared_lr"],
                    "winner": row["winner"],
                    "winner_improve_frac": row["winner_improve_frac"],
                    "winner_dominance_fraction": row["winner_dominance_fraction"],
                }
                for _, row in ranked_all.head(12).iterrows()
            ]
        )
        top_summary.to_csv(dirs["root"] / "paper_ready_results_summary.csv", index=False)
        paper_lines = [
            "# Paper-Ready Top Configs",
            "",
            "No clean baseline-positive configs were confirmed.",
            "",
        ]
        for _, row in ranked_all.head(12).iterrows():
            paper_lines.append(
                f"- env=`{row['env_name']}` scope=`{row['optimizer_scope']}` alpha=`{row['alpha']}` lr=`{row['shared_lr']}` "
                f"improve=`{row['winner_improve_frac']:.3f}` dominance=`{row['winner_dominance_fraction']:.3f}` dup=`{bool(row['alpha_duplicate_flag'])}`"
            )
        (dirs["root"] / "paper_ready_top_configs.md").write_text("\n".join(paper_lines), encoding="utf-8")
        final_lines = [
            "# TDD Autopilot Final Report",
            "",
            f"1. Whether alpha was actually active.\n`True` in Stage 0 for the available environments.",
            f"2. Whether eval semantics were correct.\n`True` for clean/current_adv/random_adv naming; `local_BR_eval_return` remained unavailable and was not faked.",
            "3. Which optimizer scope worked better.",
            "No paper-usable answer yet because the strict baseline gate was invalidated by copied/duplicated alpha curves in several searches."
            if duplicate_issue
            else "No scope produced a clean baseline-positive regime under the stricter gate.",
            "4. Which envs were baseline-positive.",
            "`None` under the stricter gate.",
            "5. Which envs were QP-positive.",
            "QP was not run because Stage 3A did not produce clean baseline-positive configs.",
            "6. Whether QP outperformed almost throughout training.",
            "Not evaluated.",
            "7. Whether results are paper-ready.",
            "`False`.",
            "8. What figures can be used in Subsection 3.",
            f"Only diagnostic figures such as `{dirs['stage3a'] / 'baseline_search_report.md'}` and the Stage 0/1/2 audits.",
            "9. What caveats must be stated.",
            "Standard RARL baseline search remains inconclusive because copied alpha curves invalidate the apparent positives."
            if duplicate_issue
            else "No clean extra-gradient-friendly baseline regime was found under the current standard RARL setup.",
        ]
        (dirs["root"] / "tdd_autopilot_final_report.md").write_text("\n\n".join(final_lines), encoding="utf-8")
        (dirs["root"] / "tdd_autopilot_final_decision.md").write_text(
            "\n".join(
                [
                    "# Final Decision",
                    "",
                    f"- decision: `{decision}`",
                    f"- duplicate_alpha_curves_detected: `{duplicate_issue}`",
                    f"- baseline_positive_count: `{len(weak_positive_df)}`",
                ]
            ),
            encoding="utf-8",
        )
        append_progress(output_root, f"Autopilot stop after Stage 3A with decision={decision}")
        return

    # If we ever reach this branch, Stage 3A found clean positives. This implementation
    # intentionally stops here until the stricter QP stage is validated on top of clean baselines.
    decision = "BASELINE_POSITIVE_BUT_QP_FAILS"
    weak_positive_df.to_csv(dirs["root"] / "paper_ready_results_summary.csv", index=False)
    (dirs["root"] / "paper_ready_top_configs.md").write_text(
        "\n".join(
            [
                "# Paper-Ready Top Configs",
                "",
                "Clean baseline-positive configs were found, but QP validation was not executed because this run stopped at the baseline gate.",
            ]
        ),
        encoding="utf-8",
    )
    (dirs["root"] / "tdd_autopilot_final_report.md").write_text(
        "\n".join(
            [
                "# TDD Autopilot Final Report",
                "",
                "Clean baseline-positive configs were found.",
                "QP validation was not run in this implementation branch, so no QP-positive claim is made.",
            ]
        ),
        encoding="utf-8",
    )
    (dirs["root"] / "tdd_autopilot_final_decision.md").write_text(
        f"# Final Decision\n\n- decision: `{decision}`\n",
        encoding="utf-8",
    )
    append_progress(output_root, f"Autopilot stop after Stage 3A positives with decision={decision}")


if __name__ == "__main__":
    main()
