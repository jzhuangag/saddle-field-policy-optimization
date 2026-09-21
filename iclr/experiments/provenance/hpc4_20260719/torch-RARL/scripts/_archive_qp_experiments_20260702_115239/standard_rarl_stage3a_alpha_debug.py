from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 3A alpha debug for standard alternating RARL")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument(
        "--output-root",
        type=str,
        default=r"C:\Users\jzhuangag\work\rarl\original\results\standard_rarl_tdd_autopilot\03_baseline_search",
    )
    parser.add_argument("--env", type=str, default="HalfCheetah-v5")
    parser.add_argument("--fallback-env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=10240)
    parser.add_argument("--n-eval-episodes", type=int, default=8)
    parser.add_argument("--n-mu", type=int, default=5)
    parser.add_argument("--n-nu", type=int, default=1)
    parser.add_argument("--shared-max-grad-norm", type=float, default=10.0)
    parser.add_argument("--shared-vf-coef", type=float, default=0.5)
    parser.add_argument("--probe-steps", type=int, default=512)
    return parser.parse_args()


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    optimizer_scope: str
    shared_lr: float
    provisional_winner: str


def stable_hash(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def curve_hash(values: List[float], decimals: int = 6) -> str:
    rounded = [round(float(v), decimals) for v in values]
    return stable_hash({"values": rounded})


def collect_eval_curve(path: pathlib.Path) -> List[float]:
    if not path.exists():
        return []
    frame = pd.read_csv(path).sort_values("timesteps")
    if "mean_reward" not in frame.columns:
        return []
    return [float(v) for v in frame["mean_reward"].tolist()]


def compute_auc_from_eval(path: pathlib.Path) -> float:
    if not path.exists():
        return math.nan
    frame = pd.read_csv(path).sort_values("timesteps")
    if len(frame) < 2:
        return math.nan
    x = pd.to_numeric(frame["timesteps"], errors="coerce").to_numpy()
    y = pd.to_numeric(frame["mean_reward"], errors="coerce").to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return math.nan
    return float(np.trapezoid(y[mask], x[mask]))


def dominance_fraction(candidate_curve: List[float], baseline_curve: List[float]) -> float:
    if len(candidate_curve) != len(baseline_curve) or not candidate_curve:
        return math.nan
    a = np.asarray(candidate_curve, dtype=float)
    b = np.asarray(baseline_curve, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return math.nan
    return float(np.mean(a[mask] > b[mask] + 1e-9))


def collect_control_probe(vec_env, protagonist, adversary, phase_name: str, requested_alpha: float, steps: int, deterministic: bool) -> Dict[str, Any]:
    vec_env.set_attr("operating_mode", "protagonist")
    vec_env.set_attr("_adv_policy", adversary.policy)
    obs = vec_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    acc = {
        "mean_abs_u": [],
        "mean_abs_w": [],
        "mean_abs_alpha_w": [],
        "mean_abs_action_before_clip": [],
        "mean_abs_action_after_clip": [],
        "action_clip_fraction": [],
        "adversary_nonzero_fraction": [],
    }
    done_count = 0
    for _ in range(steps):
        action, _ = protagonist.predict(obs, deterministic=deterministic)
        action_np = np.asarray(action).reshape(-1)
        obs, rewards, dones, infos = vec_env.step(action)
        info = infos[0] if isinstance(infos, (list, tuple)) and infos else {}
        action_space = getattr(vec_env, "action_space", None)
        if action_space is not None and hasattr(action_space, "low"):
            clipped_u = np.clip(action_np, action_space.low.reshape(-1), action_space.high.reshape(-1))
        else:
            clipped_u = action_np
        acc["mean_abs_u"].append(float(np.mean(np.abs(clipped_u))))
        acc["mean_abs_w"].append(float(info.get("adversary_action_norm_post_clip", 0.0)))
        acc["mean_abs_alpha_w"].append(float(info.get("applied_control_perturbation_norm", 0.0)))
        before_clip = float(info.get("protagonist_action_norm", 0.0) + info.get("applied_control_perturbation_norm", 0.0))
        after_clip = float(np.mean(np.abs(clipped_u)) + info.get("applied_control_perturbation_norm", 0.0))
        acc["mean_abs_action_before_clip"].append(before_clip)
        acc["mean_abs_action_after_clip"].append(after_clip)
        acc["action_clip_fraction"].append(float(info.get("action_clip_fraction", 0.0)))
        acc["adversary_nonzero_fraction"].append(1.0 if float(info.get("adversary_action_norm_post_clip", 0.0)) > 1e-8 else 0.0)
        done_count += int(np.asarray(dones).astype(np.int32).sum())
    resolved_alpha = vec_env.get_attr("adv_fraction")[0]
    return {
        "phase_name": phase_name,
        "requested_alpha": requested_alpha,
        "resolved_alpha": float(resolved_alpha),
        "mean_abs_u": float(np.mean(acc["mean_abs_u"])) if acc["mean_abs_u"] else math.nan,
        "mean_abs_w": float(np.mean(acc["mean_abs_w"])) if acc["mean_abs_w"] else math.nan,
        "mean_abs_alpha_w": float(np.mean(acc["mean_abs_alpha_w"])) if acc["mean_abs_alpha_w"] else math.nan,
        "mean_abs_action_before_clip": float(np.mean(acc["mean_abs_action_before_clip"])) if acc["mean_abs_action_before_clip"] else math.nan,
        "mean_abs_action_after_clip": float(np.mean(acc["mean_abs_action_after_clip"])) if acc["mean_abs_action_after_clip"] else math.nan,
        "action_clip_fraction": float(np.mean(acc["action_clip_fraction"])) if acc["action_clip_fraction"] else math.nan,
        "adversary_nonzero_fraction": float(np.mean(acc["adversary_nonzero_fraction"])) if acc["adversary_nonzero_fraction"] else math.nan,
        "episodes_completed": int(done_count),
    }


def collect_clean_probe(vec_env, protagonist, requested_alpha: float, steps: int, deterministic: bool) -> Dict[str, Any]:
    obs = vec_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    action_space = getattr(vec_env, "action_space", None)
    u_vals = []
    a_vals = []
    clip_vals = []
    done_count = 0
    for _ in range(steps):
        action, _ = protagonist.predict(obs, deterministic=deterministic)
        action_np = np.asarray(action).reshape(-1)
        if action_space is not None and hasattr(action_space, "low"):
            clipped_u = np.clip(action_np, action_space.low.reshape(-1), action_space.high.reshape(-1))
            clip_vals.append(float(np.mean(np.abs(clipped_u - action_np) > 1e-8)))
        else:
            clipped_u = action_np
            clip_vals.append(0.0)
        obs, rewards, dones, infos = vec_env.step(action)
        u_vals.append(float(np.mean(np.abs(clipped_u))))
        a_vals.append(float(np.mean(np.abs(clipped_u))))
        done_count += int(np.asarray(dones).astype(np.int32).sum())
    return {
        "phase_name": "clean_eval_probe",
        "requested_alpha": requested_alpha,
        "resolved_alpha": 0.0,
        "mean_abs_u": float(np.mean(u_vals)) if u_vals else math.nan,
        "mean_abs_w": 0.0,
        "mean_abs_alpha_w": 0.0,
        "mean_abs_action_before_clip": float(np.mean(a_vals)) if a_vals else math.nan,
        "mean_abs_action_after_clip": float(np.mean(a_vals)) if a_vals else math.nan,
        "action_clip_fraction": float(np.mean(clip_vals)) if clip_vals else 0.0,
        "adversary_nonzero_fraction": 0.0,
        "episodes_completed": int(done_count),
    }


def main() -> None:
    args = parse_args()
    repo_dir = pathlib.Path(args.repo_dir)
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    scripts_dir = repo_dir / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from utils.exp_manager import ExperimentManager
    from standard_rarl_baseline_positive_search import (
        MethodSpec,
        build_args_namespace,
        ensure_hyperparams,
        load_run_args,
        run_method,
    )

    try:
        probe_env = gym.make(args.env)
        probe_env.reset(seed=args.seed)
        probe_env.close()
    except Exception as exc:
        fail_path = output_root / "stage3a_alpha_debug_FAIL.md"
        fail_path.write_text(
            "\n".join(
                [
                    "# Stage 3A Alpha Debug Fail",
                    "",
                    f"- env: `{args.env}`",
                    f"- error: `{repr(exc)}`",
                    "",
                    "Environment was unavailable, so the alpha debug audit could not run.",
                ]
            ),
            encoding="utf-8",
        )
        return

    candidates = [
        CandidateSpec("candidate_egm_lr_3e3", "full_policy", 3e-3, "egm"),
        CandidateSpec("candidate_ppm_lr_1e3", "full_policy", 1e-3, "ppm_inner5"),
    ]
    alpha_grid = [0.0, 0.05, 0.1, 0.3, 0.5]
    methods = [
        MethodSpec("sgd_gda", "sgd", {}, {}, 0.0, args.shared_max_grad_norm, args.shared_vf_coef),
        MethodSpec("egm", "egm", {}, {}, 0.0, args.shared_max_grad_norm, args.shared_vf_coef),
        MethodSpec("ppm_inner5", "ppm", {"inner_steps": 5}, {"inner_steps": 5}, 0.0, args.shared_max_grad_norm, args.shared_vf_coef),
    ]
    hyperparam_dir = ensure_hyperparams(repo_dir, output_root / "stage3a_alpha_debug_temp_hparams", args.env, args.fallback_env)

    rows: List[Dict[str, Any]] = []
    real_curve_rows: List[Dict[str, Any]] = []
    real_run_root = output_root / "alpha_debug_real_runs"
    for candidate in candidates:
        for alpha in alpha_grid:
            for base_method in methods:
                method = MethodSpec(
                    base_method.label,
                    base_method.optimizer,
                    base_method.protagonist_optimizer_kwargs,
                    base_method.adversary_optimizer_kwargs,
                    candidate.shared_lr,
                    args.shared_max_grad_norm,
                    args.shared_vf_coef,
                )
                candidate_tag = "c1" if candidate.candidate_id == "candidate_egm_lr_3e3" else "c2"
                scope_tag = "fp" if candidate.optimizer_scope == "full_policy" else candidate.optimizer_scope.replace("_", "")
                method_tag = {"sgd_gda": "sgd", "egm": "egm", "ppm_inner5": "ppm5"}[method.label]
                alpha_tag = str(alpha).replace(".", "p")
                run_root = output_root / "stage3a_dbg" / candidate_tag / scope_tag / f"a_{alpha_tag}" / method_tag
                run_root.mkdir(parents=True, exist_ok=True)
                ns = build_args_namespace(
                    env_id=args.env,
                    seed=args.seed,
                    device=args.device,
                    iterations=args.iterations,
                    eval_freq=args.eval_freq,
                    n_eval_episodes=args.n_eval_episodes,
                    optimizer_scope=candidate.optimizer_scope,
                    adv_fraction=alpha,
                    method=method,
                    hyperparam_dir=hyperparam_dir,
                    run_root=run_root,
                    n_mu=args.n_mu,
                    n_nu=args.n_nu,
                )
                manager = ExperimentManager(
                    ns,
                    algo="rarl",
                    env_id=args.env,
                    log_folder=ns.log_folder,
                    tensorboard_log=ns.tensorboard_log,
                    n_timesteps=ns.n_timesteps,
                    eval_freq=ns.eval_freq,
                    n_eval_episodes=ns.n_eval_episodes,
                    save_freq=ns.save_freq,
                    hyperparameter_path=ns.hyperparameter_path,
                    hyperparams=ns.hyperparameter,
                    env_kwargs=ns.env_kwargs,
                    model_path=str(run_root / "saved_models" / "rarl-ppo" / args.env),
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
                resolved_train_alpha = safe_float(model.protagonist.env.get_attr("adv_fraction")[0])
                adv_eval_env = manager.create_envs(1, eval_env=True, with_adversarial_wrapper=True)
                resolved_eval_alpha = safe_float(adv_eval_env.get_attr("adv_fraction")[0])
                train_probe = collect_control_probe(model.protagonist.env, model.protagonist, model.adversary, "train_rollout_probe", alpha, args.probe_steps, deterministic=False)
                adv_probe = collect_control_probe(adv_eval_env, model.protagonist, model.adversary, "current_adv_eval_probe", alpha, args.probe_steps, deterministic=True)
                clean_eval_env = manager.create_envs(1, eval_env=True, with_adversarial_wrapper=False)
                clean_probe = collect_clean_probe(clean_eval_env, model.protagonist, alpha, args.probe_steps, deterministic=True)

                config_payload = {
                    "candidate_id": candidate.candidate_id,
                    "optimizer_scope": candidate.optimizer_scope,
                    "method": method.label,
                    "requested_alpha": alpha,
                    "resolved_train_alpha": resolved_train_alpha,
                    "resolved_eval_alpha": resolved_eval_alpha,
                    "shared_lr": candidate.shared_lr,
                    "seed": args.seed,
                    "iterations": args.iterations,
                    "eval_freq": args.eval_freq,
                    "n_eval_episodes": args.n_eval_episodes,
                    "N_mu": args.n_mu,
                    "N_nu": args.n_nu,
                    "adv_impact": "control",
                    "output_path": str(run_root),
                }
                config_hash = stable_hash(config_payload)

                for probe in (train_probe, adv_probe, clean_probe):
                    rows.append(
                        {
                            **config_payload,
                            "phase_name": probe["phase_name"],
                            "mean_abs_u": probe["mean_abs_u"],
                            "mean_abs_w": probe["mean_abs_w"],
                            "mean_abs_alpha_w": probe["mean_abs_alpha_w"],
                            "mean_abs_action_before_clip": probe["mean_abs_action_before_clip"],
                            "mean_abs_action_after_clip": probe["mean_abs_action_after_clip"],
                            "action_clip_fraction": probe["action_clip_fraction"],
                            "adversary_nonzero_fraction": probe["adversary_nonzero_fraction"],
                            "episodes_completed": probe["episodes_completed"],
                            "current_adv_eval_uses_adversary": int(probe["phase_name"] == "current_adv_eval_probe"),
                            "clean_eval_disables_adversary": int(probe["phase_name"] == "clean_eval_probe"),
                            "config_hash": config_hash,
                            "output_path_includes_alpha": int(f"a_{alpha_tag}" in str(run_root)),
                        }
                    )

                real_run_dir, real_summary, real_training, real_clean, real_adv, real_degradation = run_method(
                    repo_dir=repo_dir,
                    hyperparam_dir=hyperparam_dir,
                    output_root=real_run_root,
                    env_id=args.env,
                    seed=args.seed,
                    device=args.device,
                    iterations=args.iterations,
                    eval_freq=args.eval_freq,
                    n_eval_episodes=args.n_eval_episodes,
                    optimizer_scope=candidate.optimizer_scope,
                    alpha=alpha,
                    method=method,
                    n_mu=args.n_mu,
                    n_nu=args.n_nu,
                )
                real_args = load_run_args(real_run_dir)
                real_curve_rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "optimizer_scope": candidate.optimizer_scope,
                        "method": method.label,
                        "requested_alpha": alpha,
                        "resolved_adv_fraction": safe_float(real_args.get("resolved_adv_fraction")),
                        "requested_adv_fraction": safe_float(real_args.get("requested_adv_fraction", real_args.get("adv_fraction"))),
                        "shared_lr": candidate.shared_lr,
                        "adv_curve_hash": curve_hash(real_adv["mean_reward"].tolist()) if not real_adv.empty else "",
                        "clean_curve_hash": curve_hash(real_clean["mean_reward"].tolist()) if not real_clean.empty else "",
                        "adv_curve_len": int(len(real_adv)),
                        "clean_curve_len": int(len(real_clean)),
                        "final_current_adv_eval_return": safe_float(real_adv["mean_reward"].iloc[-1]) if not real_adv.empty else math.nan,
                        "final_clean_eval_return": safe_float(real_clean["mean_reward"].iloc[-1]) if not real_clean.empty else math.nan,
                        "output_path": str(real_run_dir),
                    }
                )

                try:
                    adv_eval_env.close()
                except Exception:
                    pass
                try:
                    clean_eval_env.close()
                except Exception:
                    pass
                for attr_name in ("protagonist", "adversary"):
                    agent = getattr(model, attr_name, None)
                    env = getattr(agent, "env", None)
                    if env is not None:
                        try:
                            env.close()
                        except Exception:
                            pass

    summary_df = pd.DataFrame(rows)

    existing_rows: List[Dict[str, Any]] = []
    real_curve_df = pd.DataFrame(real_curve_rows)
    if not real_curve_df.empty:
        for (candidate_id, optimizer_scope, shared_lr, method_label), sub in real_curve_df.groupby(
            ["candidate_id", "optimizer_scope", "shared_lr", "method"]
        ):
            adv_hashes = {float(row["requested_alpha"]): row["adv_curve_hash"] for _, row in sub.iterrows() if row["adv_curve_hash"]}
            clean_hashes = {float(row["requested_alpha"]): row["clean_curve_hash"] for _, row in sub.iterrows() if row["clean_curve_hash"]}
            existing_rows.append(
                {
                    "candidate_id": candidate_id,
                    "optimizer_scope": optimizer_scope,
                    "shared_lr": shared_lr,
                    "method": method_label,
                    "available_alphas": json.dumps(sorted(adv_hashes.keys())),
                    "unique_adv_curve_hashes": len(set(adv_hashes.values())),
                    "unique_clean_curve_hashes": len(set(clean_hashes.values())),
                    "adv_curve_hash_map": json.dumps({str(k): v for k, v in adv_hashes.items()}, sort_keys=True),
                    "clean_curve_hash_map": json.dumps({str(k): v for k, v in clean_hashes.items()}, sort_keys=True),
                    "adv_curve_insensitive_flag": int(len(set(adv_hashes.values())) <= 1 and len(adv_hashes) >= 3),
                    "clean_curve_insensitive_flag": int(len(set(clean_hashes.values())) <= 1 and len(clean_hashes) >= 3),
                }
            )
    existing_df = pd.DataFrame(existing_rows)

    summary_csv = output_root / "stage3a_alpha_debug_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    summary_df.to_csv(output_root / "alpha_debug_after_fix_summary.csv", index=False)
    if not existing_df.empty:
        existing_df.to_csv(output_root / "stage3a_alpha_debug_existing_curve_hashes.csv", index=False)
    if not real_curve_df.empty:
        real_curve_df.to_csv(output_root / "stage3a_alpha_debug_real_curve_summary.csv", index=False)

    requested_nonzero = summary_df[summary_df["requested_alpha"] > 0].copy()
    train_nonzero = requested_nonzero[requested_nonzero["phase_name"] == "train_rollout_probe"]
    adv_nonzero = requested_nonzero[requested_nonzero["phase_name"] == "current_adv_eval_probe"]
    a1 = bool((train_nonzero["mean_abs_alpha_w"] > 0).all()) if not train_nonzero.empty else False
    a2 = bool((adv_nonzero["mean_abs_alpha_w"] > 0).all()) if not adv_nonzero.empty else False
    resolved_train_unique = sorted(train_nonzero["resolved_train_alpha"].dropna().unique().tolist()) if not train_nonzero.empty else []
    resolved_eval_unique = sorted(adv_nonzero["resolved_eval_alpha"].dropna().unique().tolist()) if not adv_nonzero.empty else []
    requested_unique = sorted(requested_nonzero["requested_alpha"].dropna().unique().tolist()) if not requested_nonzero.empty else []
    requested_vs_resolved_match = bool(
        len(requested_unique) == len(resolved_train_unique)
        and all(abs(a - b) <= 1e-9 for a, b in zip(requested_unique, resolved_train_unique))
    )
    a5 = bool(summary_df["output_path_includes_alpha"].all()) if not summary_df.empty else False
    unique_configs = summary_df[["candidate_id", "method", "requested_alpha", "shared_lr", "config_hash"]].drop_duplicates()
    a6 = bool(len(unique_configs) == unique_configs["config_hash"].nunique()) if not unique_configs.empty else False

    ppm_vs_egm_same = False
    if not real_curve_df.empty:
        comparable = real_curve_df.pivot_table(
            index=["candidate_id", "requested_alpha", "shared_lr"],
            columns="method",
            values="adv_curve_hash",
            aggfunc="first",
        ).reset_index()
        if {"egm", "ppm_inner5"}.issubset(comparable.columns):
            valid_pairs = comparable.dropna(subset=["egm", "ppm_inner5"])
            if not valid_pairs.empty:
                ppm_vs_egm_same = bool((valid_pairs["egm"] == valid_pairs["ppm_inner5"]).all())
    a7 = not ppm_vs_egm_same

    alpha_curve_insensitive = bool(existing_df["adv_curve_insensitive_flag"].any()) if not existing_df.empty else False
    a3 = not alpha_curve_insensitive
    a4 = bool(
        summary_df.groupby(["candidate_id", "method", "phase_name"])["mean_abs_alpha_w"].nunique().max() > 1
    ) if not summary_df.empty else False

    decision = "ALPHA_ACTIVE_IN_REAL_TRAINING"
    fail_md = None
    if (not requested_vs_resolved_match) or alpha_curve_insensitive or (not a4):
        decision = "ALPHA_INACTIVE_IN_REAL_TRAINING"
        fail_md = output_root / "stage3a_alpha_debug_FAIL.md"

    report_lines = [
        "# Stage 3A Alpha Debug Report",
        "",
        "This audit checked whether requested alpha survives the actual `ExperimentManager -> adversarial wrapper -> PPO alternating train/eval env` path for HalfCheetah-v5 standard alternating RARL.",
        "",
        "## Assertions",
        "",
        f"- A1 training rollout alpha_w active for requested alpha>0: `{a1}`",
        f"- A2 current_adv eval rollout alpha_w active for requested alpha>0: `{a2}`",
        f"- A3 current_adv eval curves differ across alpha in fresh real Stage 3A runs: `{a3}`",
        f"- A4 action statistics differ across alpha in real env path probes: `{a4}`",
        f"- A5 config_hash/output_path include alpha: `{a5}`",
        f"- A6 aggregation keeps alpha distinct: `{a6}`",
        f"- A7 PPM_inner5 is not identical to EGM: `{a7}`",
        "",
        "## Requested vs resolved alpha",
        "",
        f"- requested_nonzero_alphas: `{requested_unique}`",
        f"- resolved_train_alphas_seen: `{resolved_train_unique}`",
        f"- resolved_current_adv_eval_alphas_seen: `{resolved_eval_unique}`",
        f"- requested_vs_resolved_match: `{requested_vs_resolved_match}`",
        "",
    ]
    if not existing_df.empty:
        report_lines.extend(
            [
                "## Fresh Stage 3A curve hash audit",
                "",
            ]
        )
        for _, row in existing_df.iterrows():
            report_lines.append(
                f"- `{row['candidate_id']}` / `{row['method']}`: unique_adv_curve_hashes=`{row['unique_adv_curve_hashes']}`, "
                f"unique_clean_curve_hashes=`{row['unique_clean_curve_hashes']}`, "
                f"adv_curve_insensitive_flag=`{bool(row['adv_curve_insensitive_flag'])}`"
            )
        report_lines.append("")
    report_lines.extend(
        [
            "## Root cause interpretation",
            "",
            "The current standard RARL training path now checks whether an explicit script/CLI alpha override should take priority over the PPO-RARL YAML `adv_fraction`.",
            "If resolved alpha still stayed constant across requested alpha values, Stage 0 could pass while Stage 3A real training remained alpha-insensitive.",
            "",
            f"## Decision",
            "",
            f"- decision: `{decision}`",
        ]
    )
    report_path = output_root / "stage3a_alpha_debug_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    confirmed_lines = [
        "# Stage 3A Confirmed Baseline Candidates",
        "",
    ]

    if decision == "ALPHA_INACTIVE_IN_REAL_TRAINING":
        confirmed_lines.append("No confirmed baseline-positive candidate can be trusted because requested alpha does not survive the real training/eval path.")
        if fail_md is not None:
            fail_md.write_text(
                "\n".join(
                    [
                        "# Stage 3A Alpha Debug Fail",
                        "",
                        f"- decision: `{decision}`",
                        f"- requested_nonzero_alphas: `{requested_unique}`",
                        f"- resolved_train_alphas_seen: `{resolved_train_unique}`",
                        f"- resolved_current_adv_eval_alphas_seen: `{resolved_eval_unique}`",
                        "",
                        "Alpha is effectively inactive with respect to the user-requested sweep because the actual training/eval path resolves to the same adv_fraction across different requested alpha values.",
                    ]
                ),
                encoding="utf-8",
            )
    else:
        confirmed_lines.append("Alpha override is active in real training and current-adversarial evaluation. Baseline-positive confirmation is deferred to the fresh Stage 3A search.")

    (output_root / "stage3a_confirmed_baseline_candidates.md").write_text("\n".join(confirmed_lines), encoding="utf-8")

    decision_path = output_root / "stage3a_alpha_debug_decision.md"
    decision_path.write_text(f"# Stage 3A Alpha Debug Decision\n\n- decision: `{decision}`\n", encoding="utf-8")
    (output_root / "alpha_debug_after_fix_report.md").write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
