from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import pathlib
import shutil
import sys
import traceback
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class ConfigKey:
    optimizer_scope: str
    alpha: float
    shared_lr: float

    @property
    def slug(self) -> str:
        scope = {"full_policy": "fp", "actor_logstd_only": "als"}.get(self.optimizer_scope, self.optimizer_scope)
        alpha = str(self.alpha).replace(".", "p")
        lr = f"{self.shared_lr:g}".replace(".", "p")
        return f"{scope}_a{alpha}_lr{lr}"


@dataclass(frozen=True)
class MethodSpec:
    label: str
    optimizer: str
    protagonist_optimizer_kwargs: Dict[str, object]
    adversary_optimizer_kwargs: Dict[str, object]
    lr: float
    max_grad_norm: float
    vf_coef: float
    reuse_only: bool = False


def load_module(module_name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Minimal Lyapunov repair minisearch for standard RARL")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument(
        "--output-root",
        type=str,
        default=r"C:\Users\jzhuangag\work\rarl\original\results\standard_rarl_lyapunov_repair_minisearch",
    )
    parser.add_argument("--env", type=str, default="HalfCheetah-v5")
    parser.add_argument("--fallback-env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--stage1-iterations", type=int, default=8)
    parser.add_argument("--stage1-eval-freq", type=int, default=10240)
    parser.add_argument("--stage1-eval-episodes", type=int, default=6)
    parser.add_argument("--stage2-iterations", type=int, default=10)
    parser.add_argument("--stage2-eval-freq", type=int, default=10240)
    parser.add_argument("--stage2-eval-episodes", type=int, default=8)
    parser.add_argument("--n-mu", type=int, default=5)
    parser.add_argument("--n-nu", type=int, default=1)
    parser.add_argument("--shared-max-grad-norm", type=float, default=10.0)
    parser.add_argument("--shared-vf-coef", type=float, default=0.5)
    parser.add_argument("--fd-eps", type=float, default=1e-3)
    parser.add_argument("--max-update-norm", type=float, default=0.005)
    parser.add_argument("--fallback-tolerance", type=float, default=0.0)
    return parser.parse_args()


def log_progress(path: pathlib.Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def safe_float(value, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def auc_from_curve(frame: pd.DataFrame, x_col: str, y_col: str) -> float:
    if frame.empty or x_col not in frame.columns or y_col not in frame.columns:
        return math.nan
    sub = frame[[x_col, y_col]].dropna()
    if len(sub) < 2:
        return math.nan
    return float(np.trapz(sub[y_col].to_numpy(dtype=float), sub[x_col].to_numpy(dtype=float)))


def dominance_fraction(a: Sequence[float], b: Sequence[float], higher_better: bool = True) -> float:
    av = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(dtype=float)
    bv = pd.to_numeric(pd.Series(b), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(av) & np.isfinite(bv)
    if mask.sum() == 0:
        return math.nan
    if higher_better:
        return float(np.mean(av[mask] > bv[mask] + 1e-9))
    return float(np.mean(av[mask] < bv[mask] - 1e-9))


def ensure_dir(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def register_minisearch_optimizers(repo_dir: pathlib.Path) -> None:
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from models.optimizers import OPTIMIZER_REGISTRY
    from models.proposed_qp_closedlyap import ProposedQPClosedLyapOptimizer
    from models.proposed_qp_closedlyap_merit import (
        ProposedNoGClosedActualPPOMeritOptimizer,
        ProposedQPClosedActualPPOMeritOptimizer,
        ProposedQPClosedTrustRegionMeritOptimizer,
    )

    OPTIMIZER_REGISTRY["proposed_qp_closedlyap"] = ProposedQPClosedLyapOptimizer
    OPTIMIZER_REGISTRY["proposed_nog_closed_actualppo"] = ProposedNoGClosedActualPPOMeritOptimizer
    OPTIMIZER_REGISTRY["proposed_qp_closed_actualppo"] = ProposedQPClosedActualPPOMeritOptimizer
    OPTIMIZER_REGISTRY["proposed_qp_closed_trustregion"] = ProposedQPClosedTrustRegionMeritOptimizer


def stage1_method_specs(shared_lr: float, max_grad_norm: float, vf_coef: float) -> List[MethodSpec]:
    return [
        MethodSpec("sgd_gda", "sgd", {}, {}, shared_lr, max_grad_norm, vf_coef),
        MethodSpec("egm", "egm", {}, {}, shared_lr, max_grad_norm, vf_coef),
    ]


def stage2_method_specs(
    *,
    cfg: ConfigKey,
    shared_max_grad_norm: float,
    shared_vf_coef: float,
    fd_eps: float,
    max_update_norm: float,
    fallback_tolerance: float,
    diagnostics_root: pathlib.Path,
) -> List[MethodSpec]:
    def diag_kwargs(method_label: str, role: str, cost_mode: str, lambda_F: float, lambda_R: float, extra: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        payload = {
            "optimizer_scope": cfg.optimizer_scope,
            "lambda_F": lambda_F,
            "lambda_R": lambda_R,
            "vf_coef": shared_vf_coef,
            "fd_eps": fd_eps,
            "beta_probe": cfg.shared_lr,
            "gamma_probe": max(cfg.shared_lr * cfg.shared_lr, 1e-6),
            "ridge": 1e-8,
            "beta_max": 3.0 * cfg.shared_lr,
            "gamma_max": 3.0 * max(cfg.shared_lr * cfg.shared_lr, 1e-6),
            "max_update_norm": max_update_norm,
            "qp_eps": 1e-8,
            "allow_fallback_to_egm": True,
            "fallback_tolerance": fallback_tolerance,
            "cost_mode": cost_mode,
            "role": role,
            "diagnostics_csv_path": str(diagnostics_root / f"{role}_{method_label}.csv"),
        }
        if extra:
            payload.update(extra)
        return payload

    methods = [
        MethodSpec("sgd_gda", "sgd", {}, {}, cfg.shared_lr, shared_max_grad_norm, shared_vf_coef),
        MethodSpec("egm", "egm", {}, {}, cfg.shared_lr, shared_max_grad_norm, shared_vf_coef),
        MethodSpec(
            "proposed_nog_closed",
            "proposed_nog_closed_actualppo",
            diag_kwargs("proposed_nog_closed", "protagonist", "actual_ppo_scope_matched", 0.001, 1.0),
            diag_kwargs("proposed_nog_closed", "adversary", "actual_ppo_scope_matched", 0.001, 1.0),
            cfg.shared_lr,
            shared_max_grad_norm,
            shared_vf_coef,
        ),
        MethodSpec(
            "proposed_qp_closed_current_merit_lF0p01_lR1",
            "proposed_qp_closedlyap",
            diag_kwargs("proposed_qp_closed_current_merit_lF0p01_lR1", "protagonist", "unclipped_actor_surrogate_cost", 0.01, 1.0),
            diag_kwargs("proposed_qp_closed_current_merit_lF0p01_lR1", "adversary", "unclipped_actor_surrogate_cost", 0.01, 1.0),
            cfg.shared_lr,
            shared_max_grad_norm,
            shared_vf_coef,
        ),
        MethodSpec(
            "proposed_qp_closed_actual_ppo_merit_lF0p001_lR1",
            "proposed_qp_closed_actualppo",
            diag_kwargs("proposed_qp_closed_actual_ppo_merit_lF0p001_lR1", "protagonist", "actual_ppo_scope_matched", 0.001, 1.0),
            diag_kwargs("proposed_qp_closed_actual_ppo_merit_lF0p001_lR1", "adversary", "actual_ppo_scope_matched", 0.001, 1.0),
            cfg.shared_lr,
            shared_max_grad_norm,
            shared_vf_coef,
        ),
        MethodSpec(
            "proposed_qp_closed_trustreg_lF0p01_lR0p3_k0p3_c0p1",
            "proposed_qp_closed_trustregion",
            diag_kwargs(
                "proposed_qp_closed_trustreg_lF0p01_lR0p3_k0p3_c0p1",
                "protagonist",
                "trust_region_policy_kl_clip",
                0.01,
                0.3,
                {"lambda_KL": 0.3, "lambda_CF": 0.1, "target_kl": 0.03},
            ),
            diag_kwargs(
                "proposed_qp_closed_trustreg_lF0p01_lR0p3_k0p3_c0p1",
                "adversary",
                "trust_region_policy_kl_clip",
                0.01,
                0.3,
                {"lambda_KL": 0.3, "lambda_CF": 0.1, "target_kl": 0.03},
            ),
            cfg.shared_lr,
            shared_max_grad_norm,
            shared_vf_coef,
        ),
    ]
    return methods


def build_ns(
    *,
    env_id: str,
    seed: int,
    device: str,
    iterations: int,
    eval_freq: int,
    n_eval_episodes: int,
    optimizer_scope: str,
    adv_fraction: float,
    method: MethodSpec,
    hyperparam_dir: pathlib.Path,
    run_root: pathlib.Path,
    n_mu: int,
    n_nu: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        verbose=1,
        seed=seed,
        num_exps=1,
        num_threads=-1,
        env=env_id,
        n_envs=1,
        vec_env_type="dummy",
        env_kwargs=None,
        adv_env=False,
        algo="rarl",
        rarl_config="ppo",
        saved_models_path=str(run_root / "sm"),
        pretrained_model="",
        save_replay_buffer=False,
        hyperparameter=None,
        optimize_hyperparameters=False,
        hyperparameter_path=str(hyperparam_dir),
        storage=None,
        study_name=None,
        sampler="tpe",
        pruner="median",
        optimization_log_path=str(run_root / "opt"),
        n_opt_trials=10,
        no_optim_plots=False,
        n_jobs=1,
        n_startup_trials=10,
        n_evaluations_opt=20,
        n_timesteps=iterations,
        save_freq=eval_freq,
        log_interval=-1,
        device=device,
        eval_freq=eval_freq,
        n_eval_envs=1,
        n_eval_episodes=n_eval_episodes,
        control_proxy_eval=False,
        tensorboard_log=str(run_root / "tb"),
        log_folder=str(run_root / "log"),
        protagonist_policy="MlpPolicy",
        adversary_policy="MlpPolicy",
        protagonist_optimizer=method.optimizer,
        adversary_optimizer=method.optimizer,
        protagonist_optimizer_kwargs=method.protagonist_optimizer_kwargs,
        adversary_optimizer_kwargs=method.adversary_optimizer_kwargs,
        protagonist_lr=method.lr,
        adversary_lr=method.lr,
        protagonist_max_grad_norm=method.max_grad_norm,
        adversary_max_grad_norm=method.max_grad_norm,
        protagonist_vf_coef=method.vf_coef,
        adversary_vf_coef=method.vf_coef,
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
        qp_beta_probe=method.lr,
        qp_gamma_probe=max(method.lr * method.lr, 1e-6),
        qp_ridge=1e-8,
        qp_actor_weight=1.0,
        qp_logstd_weight=1.0,
        qp_step_solver="lyapunov_quadratic_bound",
        N_mu=n_mu,
        N_nu=n_nu,
        adv_impact="control",
        adv_delay=-1,
        adv_fraction=adv_fraction,
        adv_fraction_override=True,
        requested_alpha=adv_fraction,
        requested_adv_fraction=adv_fraction,
        adv_index_list=["torso"],
        adv_force_dim=2,
    )


def load_existing_stage3a_data(existing_root: pathlib.Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ranked = pd.read_csv(existing_root / "baseline_search_ranked_RECOMPUTED.csv")
    curves = pd.read_csv(existing_root / "baseline_positive_search_curves.csv")
    return ranked, curves


def stage0_scan(existing_ranked: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    crit = (
        (existing_ranked["requested_alpha_matches_resolved_flag"] == 1)
        & (existing_ranked["alpha_curve_sensitive_flag_all_methods"] == 1)
        & (existing_ranked["EGM_over_SGD_fraction"] >= 0.60)
        & (existing_ranked["egm_current_adv_eval_return_AUC"] >= existing_ranked["sgd_current_adv_eval_return_AUC"])
        & (existing_ranked["EGM_clean_auc_ratio_vs_SGD"] >= 0.75)
        & (existing_ranked["egm_not_final_only_flag"] == 1)
    )
    cols = [
        "optimizer_scope",
        "alpha",
        "shared_lr",
        "egm_current_adv_eval_return_AUC",
        "sgd_current_adv_eval_return_AUC",
        "EGM_over_SGD_fraction",
        "EGM_clean_auc_ratio_vs_SGD",
        "egm_not_final_only_flag",
        "requested_alpha_matches_resolved_flag",
        "alpha_curve_sensitive_flag_all_methods",
    ]
    selected = existing_ranked.loc[crit, cols].copy()
    selected["existing_candidate_flag"] = 1
    selected = selected.sort_values(["EGM_over_SGD_fraction", "EGM_clean_auc_ratio_vs_SGD"], ascending=False).reset_index(drop=True)
    decision = "EXISTING_EGM_CANDIDATES_FOUND" if len(selected) > 0 else "NO_EXISTING_EGM_CANDIDATES"
    return selected, decision


def reuse_stage3a_curves_for_config(curves_df: pd.DataFrame, cfg: ConfigKey, method_label: str) -> Optional[pd.DataFrame]:
    sub = curves_df[
        (curves_df["optimizer_scope"] == cfg.optimizer_scope)
        & (pd.to_numeric(curves_df["alpha"], errors="coerce") == cfg.alpha)
        & (pd.to_numeric(curves_df["shared_lr"], errors="coerce") == cfg.shared_lr)
        & (curves_df["method"] == method_label)
    ].copy()
    if sub.empty:
        return None
    return sub.sort_values("timesteps").reset_index(drop=True)


def stage1_reuse_summary(existing_ranked: pd.DataFrame, cfg: ConfigKey, method_label: str) -> Optional[Dict[str, object]]:
    sub = existing_ranked[
        (existing_ranked["optimizer_scope"] == cfg.optimizer_scope)
        & (pd.to_numeric(existing_ranked["alpha"], errors="coerce") == cfg.alpha)
        & (pd.to_numeric(existing_ranked["shared_lr"], errors="coerce") == cfg.shared_lr)
    ]
    if sub.empty:
        return None
    row = sub.iloc[0].to_dict()
    prefix = "sgd" if method_label == "sgd_gda" else "egm"
    return {
        "optimizer_scope": cfg.optimizer_scope,
        "alpha": cfg.alpha,
        "shared_lr": cfg.shared_lr,
        "method": method_label,
        "curve_sane_flag": int(row[f"{prefix}_curve_sane_flag"]),
        "requested_alpha_matches_resolved_flag": int(row["requested_alpha_matches_resolved_flag"]),
        "alpha_curve_sensitive_flag_all_methods": int(row["alpha_curve_sensitive_flag_all_methods"]),
        "source": "stage3a_reuse",
    }


def run_with_exp_manager(
    *,
    repo_dir: pathlib.Path,
    env_id: str,
    hyperparam_dir: pathlib.Path,
    run_root: pathlib.Path,
    ns: SimpleNamespace,
    method_label: str,
) -> pathlib.Path:
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from utils.exp_manager import ExperimentManager

    stdout_path = run_root / "stdout.txt"
    stderr_path = run_root / "stderr.txt"
    ensure_dir(run_root)
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        try:
            with contextlib.redirect_stdout(stdout_handle), contextlib.redirect_stderr(stderr_handle):
                manager = ExperimentManager(
                    ns,
                    algo="rarl",
                    env_id=env_id,
                    log_folder=ns.log_folder,
                    tensorboard_log=ns.tensorboard_log,
                    n_timesteps=ns.n_timesteps,
                    eval_freq=ns.eval_freq,
                    n_eval_episodes=ns.n_eval_episodes,
                    save_freq=ns.save_freq,
                    hyperparameter_path=ns.hyperparameter_path,
                    hyperparams=ns.hyperparameter,
                    env_kwargs=ns.env_kwargs,
                    model_path=str(run_root / "sm" / "rarl-ppo" / env_id),
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
                if model is None:
                    raise RuntimeError("Hyperparameter optimization mode is unsupported for minisearch.")
                manager.learn(model)
                manager.save_trained_model(model)
        except Exception:
            stderr_handle.write("\n" + traceback.format_exc())
            raise

    env_root = run_root / "sm" / "rarl-ppo" / env_id
    run_dirs = [path for path in env_root.iterdir() if path.is_dir() and path.name.startswith(f"{env_id}_")]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories produced for {method_label} in {run_root}")
    return max(run_dirs, key=lambda path: int(path.name.rsplit("_", 1)[-1]))


def try_latest_run_dir(run_root: pathlib.Path, env_id: str) -> Optional[pathlib.Path]:
    env_root = run_root / "sm" / "rarl-ppo" / env_id
    if not env_root.exists():
        return None
    run_dirs = [path for path in env_root.iterdir() if path.is_dir() and path.name.startswith(f"{env_id}_")]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda path: int(path.name.rsplit("_", 1)[-1]))


def analyze_run(base_module, repo_dir: pathlib.Path, run_dir: pathlib.Path, analysis_dir: pathlib.Path, method_label: str) -> Tuple[Dict[str, object], pd.DataFrame]:
    ensure_dir(analysis_dir)
    if not (analysis_dir / "run_summary.csv").exists():
        base_module.run_analysis(repo_dir, run_dir, analysis_dir, method_label)
    summary = pd.read_csv(analysis_dir / "run_summary.csv").iloc[0].to_dict()
    training = base_module.load_frame(analysis_dir / "training_episode_returns.csv", method_label)
    clean = base_module.load_frame(analysis_dir / "clean_eval_returns.csv", method_label)
    adv = base_module.load_frame(analysis_dir / "adversarial_eval_returns.csv", method_label)
    degradation = pd.DataFrame(
        {
            "timesteps": clean["timesteps"],
            "train_return": np.nan,
            "clean_eval_return": clean["mean_reward"],
            "current_adv_eval_return": adv["mean_reward"],
            "current_adv_degradation": clean["mean_reward"] - adv["mean_reward"],
            "method": method_label,
        }
    )
    metric_frame = base_module.aggregate_training_metrics(run_dir)
    curve = base_module.build_method_curve(
        method_label=method_label,
        shared_lr=safe_float(summary.get("shared_lr", math.nan)),
        training=training,
        clean=clean,
        adv=adv,
        degradation=degradation,
        metric_frame=metric_frame,
    )
    return summary, curve


def load_analyzed_curve(base_module, run_dir: pathlib.Path, analysis_dir: pathlib.Path, method_label: str) -> Tuple[Dict[str, object], pd.DataFrame]:
    summary = pd.read_csv(analysis_dir / "run_summary.csv").iloc[0].to_dict()
    training = base_module.load_frame(analysis_dir / "training_episode_returns.csv", method_label)
    clean = base_module.load_frame(analysis_dir / "clean_eval_returns.csv", method_label)
    adv = base_module.load_frame(analysis_dir / "adversarial_eval_returns.csv", method_label)
    degradation = pd.DataFrame(
        {
            "timesteps": clean["timesteps"],
            "train_return": np.nan,
            "clean_eval_return": clean["mean_reward"],
            "current_adv_eval_return": adv["mean_reward"],
            "current_adv_degradation": clean["mean_reward"] - adv["mean_reward"],
            "method": method_label,
        }
    )
    metric_frame = base_module.aggregate_training_metrics(run_dir)
    curve = base_module.build_method_curve(
        method_label=method_label,
        shared_lr=safe_float(summary.get("shared_lr", math.nan)),
        training=training,
        clean=clean,
        adv=adv,
        degradation=degradation,
        metric_frame=metric_frame,
    )
    return summary, curve


def run_stage1_candidate(
    *,
    base_module,
    repo_dir: pathlib.Path,
    env_id: str,
    hyperparam_dir: pathlib.Path,
    stage1_root: pathlib.Path,
    cfg: ConfigKey,
    method: MethodSpec,
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    run_root = stage1_root / cfg.slug / method.label
    analysis_dir = run_root / "analysis"
    existing_run_dir = try_latest_run_dir(run_root, env_id)
    if existing_run_dir is not None and (analysis_dir / "run_summary.csv").exists():
        summary, curve = analyze_run(base_module, repo_dir, existing_run_dir, analysis_dir, method.label)
        row = dict(summary)
        row.update(
            {
                "optimizer_scope": cfg.optimizer_scope,
                "alpha": cfg.alpha,
                "shared_lr": cfg.shared_lr,
                "method": method.label,
                "curve_sane_flag": int(base_module.curve_sane(summary, curve)),
                "clean_eval_return_AUC": auc_from_curve(curve, "timesteps", "clean_eval_return"),
                "current_adv_eval_return_AUC": auc_from_curve(curve, "timesteps", "current_adv_eval_return"),
                "source": "stage1_reuse_same_root",
            }
        )
        return row, curve
    ns = build_ns(
        env_id=env_id,
        seed=args.seed,
        device=args.device,
        iterations=args.stage1_iterations,
        eval_freq=args.stage1_eval_freq,
        n_eval_episodes=args.stage1_eval_episodes,
        optimizer_scope=cfg.optimizer_scope,
        adv_fraction=cfg.alpha,
        method=method,
        hyperparam_dir=hyperparam_dir,
        run_root=run_root,
        n_mu=args.n_mu,
        n_nu=args.n_nu,
    )
    run_dir = run_with_exp_manager(repo_dir=repo_dir, env_id=env_id, hyperparam_dir=hyperparam_dir, run_root=run_root, ns=ns, method_label=method.label)
    summary, curve = analyze_run(base_module, repo_dir, run_dir, analysis_dir, method.label)
    row = dict(summary)
    row.update(
        {
            "optimizer_scope": cfg.optimizer_scope,
            "alpha": cfg.alpha,
            "shared_lr": cfg.shared_lr,
            "method": method.label,
            "curve_sane_flag": int(base_module.curve_sane(summary, curve)),
            "clean_eval_return_AUC": auc_from_curve(curve, "timesteps", "clean_eval_return"),
            "current_adv_eval_return_AUC": auc_from_curve(curve, "timesteps", "current_adv_eval_return"),
            "source": "stage1_new_run",
        }
    )
    return row, curve


def select_stage2_configs(stage0_candidates: pd.DataFrame, stage1_ranked: pd.DataFrame) -> List[ConfigKey]:
    chosen: List[ConfigKey] = []
    seen = set()
    for frame in [stage0_candidates, stage1_ranked]:
        for _, row in frame.iterrows():
            key = ConfigKey(str(row["optimizer_scope"]), float(row["alpha"]), float(row["shared_lr"]))
            if key.slug in seen:
                continue
            chosen.append(key)
            seen.add(key.slug)
            if len(chosen) >= 4:
                return chosen
    return chosen


def curve_sane_simple(curve: pd.DataFrame) -> bool:
    if curve.empty:
        return False
    series = pd.to_numeric(curve["current_adv_eval_return"], errors="coerce")
    return bool(series.notna().all())


def rank_stage1_configs(summary_rows: List[Dict[str, object]], curves_map: Dict[Tuple[str, float, float, str], pd.DataFrame]) -> pd.DataFrame:
    df = pd.DataFrame(summary_rows)
    ranked_rows: List[Dict[str, object]] = []
    for (scope, alpha, lr), sub in df.groupby(["optimizer_scope", "alpha", "shared_lr"]):
        per_method = {row["method"]: row for _, row in sub.iterrows()}
        if set(per_method.keys()) != {"sgd_gda", "egm"}:
            continue
        sgd = per_method["sgd_gda"]
        egm = per_method["egm"]
        sgd_curve = curves_map[(scope, alpha, lr, "sgd_gda")]
        egm_curve = curves_map[(scope, alpha, lr, "egm")]
        common = sgd_curve[["timesteps", "current_adv_eval_return"]].rename(columns={"current_adv_eval_return": "sgd"})
        common = common.merge(egm_curve[["timesteps", "current_adv_eval_return"]].rename(columns={"current_adv_eval_return": "egm"}), on="timesteps", how="inner")
        common_clean = sgd_curve[["timesteps", "clean_eval_return"]].rename(columns={"clean_eval_return": "sgd"})
        common_clean = common_clean.merge(egm_curve[["timesteps", "clean_eval_return"]].rename(columns={"clean_eval_return": "egm"}), on="timesteps", how="inner")
        dom = dominance_fraction(common["egm"], common["sgd"], higher_better=True)
        sgd_auc = auc_from_curve(sgd_curve, "timesteps", "current_adv_eval_return")
        egm_auc = auc_from_curve(egm_curve, "timesteps", "current_adv_eval_return")
        improve = (egm_auc / (sgd_auc + EPS)) - 1.0 if finite(sgd_auc) and finite(egm_auc) else math.nan
        sgd_clean_auc = auc_from_curve(sgd_curve, "timesteps", "clean_eval_return")
        egm_clean_auc = auc_from_curve(egm_curve, "timesteps", "clean_eval_return")
        clean_ratio = egm_clean_auc / (sgd_clean_auc + EPS) if finite(egm_clean_auc) and finite(sgd_clean_auc) else math.nan
        egm_beats_mask = pd.to_numeric(common["egm"], errors="coerce") > pd.to_numeric(common["sgd"], errors="coerce") + 1e-9
        not_final_only = int(bool(len(egm_beats_mask) >= 2 and egm_beats_mask.iloc[:-1].any()))
        alpha_match = int(sgd.get("requested_alpha_matches_resolved_flag", 1)) if "requested_alpha_matches_resolved_flag" in sgd else 1
        alpha_sensitive = int(sgd.get("alpha_curve_sensitive_flag_all_methods", 1)) if "alpha_curve_sensitive_flag_all_methods" in sgd else 1
        weak = bool(alpha_match == 1 and alpha_sensitive == 1 and finite(improve) and improve >= 0.05 and finite(dom) and dom >= 0.60 and finite(clean_ratio) and clean_ratio >= 0.75 and not_final_only == 1 and int(sgd["curve_sane_flag"]) == 1 and int(egm["curve_sane_flag"]) == 1)
        strong = bool(alpha_match == 1 and alpha_sensitive == 1 and finite(improve) and improve >= 0.10 and finite(dom) and dom >= 0.70 and finite(clean_ratio) and clean_ratio >= 0.80 and not_final_only == 1 and int(sgd["curve_sane_flag"]) == 1 and int(egm["curve_sane_flag"]) == 1)
        ranked_rows.append(
            {
                "optimizer_scope": scope,
                "alpha": alpha,
                "shared_lr": lr,
                "sgd_current_adv_eval_return_AUC": sgd_auc,
                "egm_current_adv_eval_return_AUC": egm_auc,
                "EGM_over_SGD_fraction": dom,
                "EGM_clean_auc_ratio_vs_SGD": clean_ratio,
                "egm_improvement_frac": improve,
                "egm_not_final_only_flag": not_final_only,
                "alpha_match": alpha_match,
                "alpha_sensitive": alpha_sensitive,
                "egm_weak_positive": int(weak),
                "egm_strong_positive": int(strong),
            }
        )
    if not ranked_rows:
        return pd.DataFrame()
    frame = pd.DataFrame(ranked_rows).sort_values(["egm_strong_positive", "egm_weak_positive", "egm_improvement_frac", "EGM_over_SGD_fraction"], ascending=[False, False, False, False]).reset_index(drop=True)
    return frame


def run_stage2_method(
    *,
    base_module,
    repo_dir: pathlib.Path,
    env_id: str,
    hyperparam_dir: pathlib.Path,
    stage2_root: pathlib.Path,
    cfg: ConfigKey,
    method: MethodSpec,
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], pd.DataFrame, pathlib.Path]:
    run_root = stage2_root / cfg.slug / method.label
    analysis_dir = run_root / "analysis"
    existing_run_dir = try_latest_run_dir(run_root, env_id)
    if existing_run_dir is not None and (analysis_dir / "run_summary.csv").exists():
        summary, curve = analyze_run(base_module, repo_dir, existing_run_dir, analysis_dir, method.label)
        row = dict(summary)
        row.update(
            {
                "optimizer_scope": cfg.optimizer_scope,
                "alpha": cfg.alpha,
                "shared_lr": cfg.shared_lr,
                "method": method.label,
                "curve_sane_flag": int(curve_sane_simple(curve)),
                "train_return_AUC": auc_from_curve(curve, "timesteps", "train_return"),
                "clean_eval_return_AUC": auc_from_curve(curve, "timesteps", "clean_eval_return"),
                "current_adv_eval_return_AUC": auc_from_curve(curve, "timesteps", "current_adv_eval_return"),
                "current_adv_degradation_AUC": auc_from_curve(curve, "timesteps", "current_adv_degradation"),
                "field_norm_AUC": auc_from_curve(curve, "timesteps", "field_norm"),
                "surrogate_lyapunov_AUC": auc_from_curve(curve, "timesteps", "surrogate_lyapunov_value"),
            }
        )
        return row, curve, run_root
    ns = build_ns(
        env_id=env_id,
        seed=args.seed,
        device=args.device,
        iterations=args.stage2_iterations,
        eval_freq=args.stage2_eval_freq,
        n_eval_episodes=args.stage2_eval_episodes,
        optimizer_scope=cfg.optimizer_scope,
        adv_fraction=cfg.alpha,
        method=method,
        hyperparam_dir=hyperparam_dir,
        run_root=run_root,
        n_mu=args.n_mu,
        n_nu=args.n_nu,
    )
    register_minisearch_optimizers(repo_dir)
    run_dir = run_with_exp_manager(repo_dir=repo_dir, env_id=env_id, hyperparam_dir=hyperparam_dir, run_root=run_root, ns=ns, method_label=method.label)
    summary, curve = analyze_run(base_module, repo_dir, run_dir, analysis_dir, method.label)
    row = dict(summary)
    row.update(
        {
            "optimizer_scope": cfg.optimizer_scope,
            "alpha": cfg.alpha,
            "shared_lr": cfg.shared_lr,
            "method": method.label,
            "curve_sane_flag": int(curve_sane_simple(curve)),
            "train_return_AUC": auc_from_curve(curve, "timesteps", "train_return"),
            "clean_eval_return_AUC": auc_from_curve(curve, "timesteps", "clean_eval_return"),
            "current_adv_eval_return_AUC": auc_from_curve(curve, "timesteps", "current_adv_eval_return"),
            "current_adv_degradation_AUC": auc_from_curve(curve, "timesteps", "current_adv_degradation"),
            "field_norm_AUC": auc_from_curve(curve, "timesteps", "field_norm"),
            "surrogate_lyapunov_AUC": auc_from_curve(curve, "timesteps", "surrogate_lyapunov_value"),
        }
    )
    return row, curve, run_root


def reuse_stage2_baseline_from_existing(
    *,
    base_module,
    existing_ranked: pd.DataFrame,
    existing_curves: pd.DataFrame,
    stage1_root: pathlib.Path,
    cfg: ConfigKey,
    method_label: str,
) -> Optional[Tuple[Dict[str, object], pd.DataFrame]]:
    if cfg.shared_lr != 1e-4:
        reuse_summary = stage1_reuse_summary(existing_ranked, cfg, method_label)
        reuse_curve = reuse_stage3a_curves_for_config(existing_curves, cfg, method_label)
        if reuse_summary is not None and reuse_curve is not None:
            row = dict(reuse_summary)
            row.update(
                {
                    "train_return_AUC": auc_from_curve(reuse_curve, "timesteps", "train_return"),
                    "clean_eval_return_AUC": auc_from_curve(reuse_curve, "timesteps", "clean_eval_return"),
                    "current_adv_eval_return_AUC": auc_from_curve(reuse_curve, "timesteps", "current_adv_eval_return"),
                    "current_adv_degradation_AUC": auc_from_curve(reuse_curve, "timesteps", "current_adv_degradation"),
                    "field_norm_AUC": auc_from_curve(reuse_curve, "timesteps", "field_norm"),
                    "surrogate_lyapunov_AUC": auc_from_curve(reuse_curve, "timesteps", "surrogate_lyapunov_value"),
                }
            )
            return row, reuse_curve
        return None
    analysis_dir = stage1_root / cfg.slug / method_label / "analysis"
    run_dir = try_latest_run_dir(stage1_root / cfg.slug / method_label, "HalfCheetah-v5")
    if run_dir is None or not (analysis_dir / "run_summary.csv").exists():
        return None
    summary, curve = load_analyzed_curve(base_module, run_dir=run_dir, analysis_dir=analysis_dir, method_label=method_label)
    row = dict(summary)
    row.update(
        {
            "optimizer_scope": cfg.optimizer_scope,
            "alpha": cfg.alpha,
            "shared_lr": cfg.shared_lr,
            "method": method_label,
            "curve_sane_flag": int(curve_sane_simple(curve)),
            "train_return_AUC": auc_from_curve(curve, "timesteps", "train_return"),
            "clean_eval_return_AUC": auc_from_curve(curve, "timesteps", "clean_eval_return"),
            "current_adv_eval_return_AUC": auc_from_curve(curve, "timesteps", "current_adv_eval_return"),
            "current_adv_degradation_AUC": auc_from_curve(curve, "timesteps", "current_adv_degradation"),
            "field_norm_AUC": auc_from_curve(curve, "timesteps", "field_norm"),
            "surrogate_lyapunov_AUC": auc_from_curve(curve, "timesteps", "surrogate_lyapunov_value"),
            "source": "stage1_reuse",
        }
    )
    return row, curve


def summarise_diag_pair(run_root: pathlib.Path, method_label: str) -> Dict[str, float]:
    protagonist_path = run_root / f"protagonist_{method_label}.csv"
    adversary_path = run_root / f"adversary_{method_label}.csv"
    rows = []
    for path in [protagonist_path, adversary_path]:
        if path.exists():
            frame = pd.read_csv(path)
            if not frame.empty:
                g_ratio = pd.to_numeric(frame.get("G_contribution_norm", pd.Series(np.nan, index=frame.index)), errors="coerce") / (
                    pd.to_numeric(frame.get("update_norm_post_cap", pd.Series(np.nan, index=frame.index)), errors="coerce").abs() + EPS
                )
                qp_better = pd.to_numeric(frame.get("actual_drift", pd.Series(np.nan, index=frame.index)), errors="coerce") < (
                    pd.to_numeric(frame.get("egm_actual_drift", pd.Series(np.nan, index=frame.index)), errors="coerce") - 1e-12
                )
                rows.append(
                    {
                        "fallback_to_egm_frac": float(pd.to_numeric(frame.get("fallback_to_egm", pd.Series(np.nan, index=frame.index)), errors="coerce").mean()),
                        "gamma_active_frac": float(pd.to_numeric(frame.get("gamma_active_frac", pd.Series(np.nan, index=frame.index)), errors="coerce").mean()),
                        "G_contribution_ratio": float(pd.to_numeric(g_ratio, errors="coerce").replace([np.inf, -np.inf], np.nan).mean()),
                        "QP_better_than_EGM_drift_fraction": float(pd.to_numeric(qp_better.astype(float), errors="coerce").mean()),
                        "mean_actual_QP_drift": float(pd.to_numeric(frame.get("actual_drift", pd.Series(np.nan, index=frame.index)), errors="coerce").mean()),
                        "mean_actual_EGM_drift": float(pd.to_numeric(frame.get("egm_actual_drift", pd.Series(np.nan, index=frame.index)), errors="coerce").mean()),
                        "cos_F_G": float(pd.to_numeric(frame.get("cos_F_G", pd.Series(np.nan, index=frame.index)), errors="coerce").mean()),
                        "non_collinearity": float(pd.to_numeric(frame.get("non_collinearity", pd.Series(np.nan, index=frame.index)), errors="coerce").mean()),
                    }
                )
    if not rows:
        return {}
    keys = rows[0].keys()
    return {f"avg_{key}": float(np.nanmean([row[key] for row in rows])) for key in keys}


def plot_top_curves(curves: Dict[str, pd.DataFrame], out_path: pathlib.Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    metrics = [
        ("train_return", "Train Return"),
        ("clean_eval_return", "Clean Eval Return"),
        ("current_adv_eval_return", "Current-Adv Eval Return"),
        ("current_adv_degradation", "Current-Adv Degradation"),
    ]
    for ax, (col, name) in zip(axes.flatten(), metrics):
        for method, frame in curves.items():
            if col in frame.columns:
                ax.plot(frame["timesteps"], frame[col], label=method, linewidth=1.4)
        ax.set_title(name)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_geometry(rank_df: pd.DataFrame, out_path: pathlib.Path) -> None:
    qp_rows = rank_df[rank_df["method"].str.contains("proposed_qp_closed", na=False)].copy()
    if qp_rows.empty:
        return
    labels = [f"{row.optimizer_scope}\na={row.alpha}\nlr={row.shared_lr}\n{row.method}" for row in qp_rows.itertuples()]
    x = np.arange(len(qp_rows))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].bar(x - 0.2, qp_rows["avg_gamma_active_frac"], width=0.2, label="gamma active")
    axes[0].bar(x, qp_rows["avg_G_contribution_ratio"], width=0.2, label="G ratio")
    axes[0].bar(x + 0.2, qp_rows["avg_QP_better_than_EGM_drift_fraction"], width=0.2, label="QP>EGM drift")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Useful Geometry Signals")
    axes[1].bar(x - 0.2, qp_rows["avg_fallback_to_egm_frac"], width=0.2, label="fallback")
    axes[1].bar(x, qp_rows["avg_cos_F_G"], width=0.2, label="cos(F,G)")
    axes[1].bar(x + 0.2, qp_rows["avg_non_collinearity"], width=0.2, label="non-collinearity")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_title("Fallback / Geometry")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_dir = pathlib.Path(args.repo_dir)
    output_root = pathlib.Path(args.output_root)
    stage0_root = ensure_dir(output_root / "00_existing_egm_candidate_scan")
    stage1_root = ensure_dir(output_root / "01_small_egm_baseline_search")
    stage2_root = ensure_dir(output_root / "02_lyapunov_variant_qp")
    plots_root = ensure_dir(output_root / "plots")
    progress_log = output_root / "lyapunov_repair_progress.log"

    base_module = load_module("s3a_base_mod", repo_dir / "scripts" / "standard_rarl_baseline_positive_search.py")
    one_seed_module = load_module("s3b_one_seed_mod", repo_dir / "scripts" / "standard_rarl_one_seed_closed_qp_check.py")
    register_minisearch_optimizers(repo_dir)
    hyperparam_dir = one_seed_module.ensure_hyperparams(repo_dir, output_root, args.env, args.fallback_env)

    existing_root = repo_dir.parent / "results" / "standard_rarl_tdd_autopilot" / "03_baseline_search_after_alpha_fix"
    existing_ranked, existing_curves = load_existing_stage3a_data(existing_root)

    log_progress(progress_log, "Stage 0: scanning existing after-alpha-fix EGM candidates")
    stage0_candidates, stage0_decision = stage0_scan(existing_ranked)
    stage0_candidates.to_csv(stage0_root / "existing_egm_candidate_scan.csv", index=False)
    (stage0_root / "existing_egm_candidate_scan.md").write_text(
        "\n".join(
            [
                f"Decision: `{stage0_decision}`",
                "",
                f"Found `{len(stage0_candidates)}` existing EGM trial candidate(s) using the prompt filters.",
            ]
            + [
                f"- scope=`{row.optimizer_scope}`, alpha=`{row.alpha}`, lr=`{row.shared_lr}`: "
                f"EGM/SGD AUC=`{row.egm_current_adv_eval_return_AUC / (row.sgd_current_adv_eval_return_AUC + EPS):.3f}`, "
                f"dominance=`{row.EGM_over_SGD_fraction:.3f}`, clean-ratio=`{row.EGM_clean_auc_ratio_vs_SGD:.3f}`"
                for row in stage0_candidates.itertuples()
            ]
        ),
        encoding="utf-8",
    )

    stage1_summary_rows: List[Dict[str, object]] = []
    stage1_curves: Dict[Tuple[str, float, float, str], pd.DataFrame] = {}
    if len(stage0_candidates) < 2:
        log_progress(progress_log, "Stage 1: running small EGM baseline search (reuse existing after-alpha-fix runs when possible)")
        reuse_configs = [
            ConfigKey(scope, alpha, lr)
            for scope in ["actor_logstd_only", "full_policy"]
            for alpha in [0.05, 0.1, 0.3]
            for lr in [3e-4, 1e-3, 3e-3]
        ]
        new_probe_configs = [
            ConfigKey("actor_logstd_only", 0.05, 1e-4),
            ConfigKey("full_policy", 0.05, 1e-4),
        ]
        for cfg in reuse_configs:
            for method in stage1_method_specs(cfg.shared_lr, args.shared_max_grad_norm, args.shared_vf_coef):
                reuse_summary = stage1_reuse_summary(existing_ranked, cfg, method.label)
                reuse_curve = reuse_stage3a_curves_for_config(existing_curves, cfg, method.label)
                if reuse_summary is not None and reuse_curve is not None:
                    stage1_summary_rows.append(reuse_summary)
                    stage1_curves[(cfg.optimizer_scope, cfg.alpha, cfg.shared_lr, method.label)] = reuse_curve
        for cfg in new_probe_configs:
            for method in stage1_method_specs(cfg.shared_lr, args.shared_max_grad_norm, args.shared_vf_coef):
                log_progress(progress_log, f"Stage 1 run: scope={cfg.optimizer_scope} alpha={cfg.alpha} lr={cfg.shared_lr} method={method.label}")
                row, curve = run_stage1_candidate(
                    base_module=base_module,
                    repo_dir=repo_dir,
                    env_id=args.env,
                    hyperparam_dir=hyperparam_dir,
                    stage1_root=stage1_root,
                    cfg=cfg,
                    method=method,
                    args=args,
                )
                row["requested_alpha_matches_resolved_flag"] = 1
                row["alpha_curve_sensitive_flag_all_methods"] = 1
                stage1_summary_rows.append(row)
                stage1_curves[(cfg.optimizer_scope, cfg.alpha, cfg.shared_lr, method.label)] = curve
        stage1_ranked = rank_stage1_configs(stage1_summary_rows, stage1_curves)
        stage1_ranked.to_csv(stage1_root / "small_egm_search_ranked.csv", index=False)
        pd.DataFrame(stage1_summary_rows).to_csv(stage1_root / "small_egm_search_all.csv", index=False)
        strong_count = int((stage1_ranked["egm_strong_positive"] == 1).sum()) if not stage1_ranked.empty else 0
        weak_count = int((stage1_ranked["egm_weak_positive"] == 1).sum()) if not stage1_ranked.empty else 0
        decision = "EGM_STRONG_POSITIVE_FOUND" if strong_count > 0 else ("EGM_WEAK_POSITIVE_FOUND" if weak_count > 0 else "NO_EGM_POSITIVE_BEST_NEAR_MISSES_ONLY")
        lines = [f"Decision: `{decision}`", "", f"Evaluated `{len(stage1_ranked)}` `(scope, alpha, lr)` configs."]
        lines.append("- existing after-alpha-fix Stage 3A overlaps were reused directly")
        lines.append("- only two new `1e-4` probe configs were trained in this minisearch")
        for row in stage1_ranked.head(8).itertuples():
            lines.append(
                f"- scope=`{row.optimizer_scope}`, alpha=`{row.alpha}`, lr=`{row.shared_lr}`: "
                f"improve=`{row.egm_improvement_frac:.3f}`, dom=`{row.EGM_over_SGD_fraction:.3f}`, "
                f"clean-ratio=`{row.EGM_clean_auc_ratio_vs_SGD:.3f}`, weak=`{row.egm_weak_positive}`, strong=`{row.egm_strong_positive}`"
            )
        (stage1_root / "small_egm_search_report.md").write_text("\n".join(lines), encoding="utf-8")
        top_rows = stage1_ranked[(stage1_ranked["egm_weak_positive"] == 1) | (stage1_ranked["egm_strong_positive"] == 1)]
        if top_rows.empty:
            top_rows = stage1_ranked.head(3)
        (stage1_root / "small_egm_top_configs.md").write_text(
            "\n".join(
                ["Top Stage 1 configs:", ""]
                + [
                    f"- scope=`{row.optimizer_scope}`, alpha=`{row.alpha}`, lr=`{row.shared_lr}`: "
                    f"EGM/SGD AUC improve=`{row.egm_improvement_frac:.3f}`, dominance=`{row.EGM_over_SGD_fraction:.3f}`"
                    for row in top_rows.itertuples()
                ]
            ),
            encoding="utf-8",
        )
    else:
        stage1_ranked = pd.DataFrame()

    if stage1_ranked is not None and not stage1_ranked.empty:
        positives = stage1_ranked[(stage1_ranked["egm_strong_positive"] == 1) | (stage1_ranked["egm_weak_positive"] == 1)].copy()
        if positives.empty:
            positives = stage1_ranked.head(2).copy()
    else:
        positives = pd.DataFrame()

    stage2_configs = select_stage2_configs(stage0_candidates, positives)
    stage2_configs = stage2_configs[:1] if len(stage2_configs) > 1 else stage2_configs
    log_progress(progress_log, f"Stage 2 selected configs: {[cfg.slug for cfg in stage2_configs]}")

    all_stage2_rows: List[Dict[str, object]] = []
    all_stage2_curves: Dict[Tuple[str, str], pd.DataFrame] = {}
    for cfg in stage2_configs:
        log_progress(progress_log, f"Stage 2 config start: {cfg.slug}")
        diagnostics_root = ensure_dir(stage2_root / cfg.slug / "diagnostics")
        for method in stage2_method_specs(
            cfg=cfg,
            shared_max_grad_norm=args.shared_max_grad_norm,
            shared_vf_coef=args.shared_vf_coef,
            fd_eps=args.fd_eps,
            max_update_norm=args.max_update_norm,
            fallback_tolerance=args.fallback_tolerance,
            diagnostics_root=diagnostics_root,
        ):
            if method.label in {"sgd_gda", "egm"}:
                reused = reuse_stage2_baseline_from_existing(
                    base_module=base_module,
                    existing_ranked=existing_ranked,
                    existing_curves=existing_curves,
                    stage1_root=stage1_root,
                    cfg=cfg,
                    method_label=method.label,
                )
                if reused is not None:
                    row, curve = reused
                    run_root = stage2_root / cfg.slug / method.label
                    log_progress(progress_log, f"Stage 2 reuse: {cfg.slug} :: {method.label}")
                else:
                    log_progress(progress_log, f"Stage 2 run: {cfg.slug} :: {method.label}")
                    row, curve, run_root = run_stage2_method(
                        base_module=base_module,
                        repo_dir=repo_dir,
                        env_id=args.env,
                        hyperparam_dir=hyperparam_dir,
                        stage2_root=stage2_root,
                        cfg=cfg,
                        method=method,
                        args=args,
                    )
            else:
                log_progress(progress_log, f"Stage 2 run: {cfg.slug} :: {method.label}")
                row, curve, run_root = run_stage2_method(
                    base_module=base_module,
                    repo_dir=repo_dir,
                    env_id=args.env,
                    hyperparam_dir=hyperparam_dir,
                    stage2_root=stage2_root,
                    cfg=cfg,
                    method=method,
                    args=args,
                )
            if "proposed_qp_closed" in method.label or "proposed_nog_closed" in method.label:
                row.update(summarise_diag_pair(diagnostics_root, method.label))
            all_stage2_rows.append(row)
            all_stage2_curves[(cfg.slug, method.label)] = curve

    stage2_df = pd.DataFrame(all_stage2_rows)
    if stage2_df.empty:
        raise RuntimeError("Stage 2 produced no runs; cannot write minisearch report.")
    stage2_df.to_csv(stage2_root / "lyapunov_variant_all_runs.csv", index=False)

    rank_rows: List[Dict[str, object]] = []
    for cfg in stage2_configs:
        sub = stage2_df[(stage2_df["optimizer_scope"] == cfg.optimizer_scope) & (pd.to_numeric(stage2_df["alpha"], errors="coerce") == cfg.alpha) & (pd.to_numeric(stage2_df["shared_lr"], errors="coerce") == cfg.shared_lr)].copy()
        if sub.empty:
            continue
        sgd = sub[sub["method"] == "sgd_gda"].iloc[0].to_dict()
        egm = sub[sub["method"] == "egm"].iloc[0].to_dict()
        nog = sub[sub["method"] == "proposed_nog_closed"].iloc[0].to_dict()
        for _, row in sub[sub["method"].str.contains("proposed_qp_closed", na=False)].iterrows():
            row = row.to_dict()
            qp_curve = all_stage2_curves[(cfg.slug, row["method"])]
            sgd_curve = all_stage2_curves[(cfg.slug, "sgd_gda")]
            egm_curve = all_stage2_curves[(cfg.slug, "egm")]
            nog_curve = all_stage2_curves[(cfg.slug, "proposed_nog_closed")]
            merged = sgd_curve[["timesteps", "current_adv_eval_return"]].rename(columns={"current_adv_eval_return": "sgd"})
            for name, frame in [("egm", egm_curve), ("nog", nog_curve), ("qp", qp_curve)]:
                merged = merged.merge(frame[["timesteps", "current_adv_eval_return"]].rename(columns={"current_adv_eval_return": name}), on="timesteps", how="inner")
            qp_dom_vs_sgd = dominance_fraction(merged["qp"], merged["sgd"], higher_better=True)
            qp_dom_vs_egm = dominance_fraction(merged["qp"], merged["egm"], higher_better=True)
            qp_dom_vs_nog = dominance_fraction(merged["qp"], merged["nog"], higher_better=True)
            qp_auc = safe_float(row["current_adv_eval_return_AUC"])
            sgd_auc = safe_float(sgd["current_adv_eval_return_AUC"])
            egm_auc = safe_float(egm["current_adv_eval_return_AUC"])
            nog_auc = safe_float(nog["current_adv_eval_return_AUC"])
            improve_vs_sgd = qp_auc / (sgd_auc + EPS) - 1.0
            improve_vs_egm = qp_auc / (egm_auc + EPS) - 1.0
            improve_vs_nog = qp_auc / (nog_auc + EPS) - 1.0
            clean_ratio_vs_egm = safe_float(row["clean_eval_return_AUC"]) / (safe_float(egm["clean_eval_return_AUC"]) + EPS)
            qp_beats_mask = pd.to_numeric(merged["qp"], errors="coerce") > pd.concat(
                [pd.to_numeric(merged["sgd"], errors="coerce"), pd.to_numeric(merged["egm"], errors="coerce"), pd.to_numeric(merged["nog"], errors="coerce")],
                axis=1,
            ).max(axis=1) + 1e-9
            not_final_only = int(bool(len(qp_beats_mask) >= 2 and qp_beats_mask.iloc[:-1].any()))
            useful_geometry = bool(
                safe_float(row.get("avg_gamma_active_frac", math.nan)) >= 0.30
                and safe_float(row.get("avg_G_contribution_ratio", math.nan)) >= 0.10
                and safe_float(row.get("avg_non_collinearity", math.nan)) >= 0.10
                and safe_float(row.get("avg_QP_better_than_EGM_drift_fraction", math.nan)) >= 0.60
            )
            weak_positive = bool(
                improve_vs_sgd >= 0.05
                and improve_vs_egm >= 0.05
                and improve_vs_nog >= 0.05
                and qp_dom_vs_sgd >= 0.70
                and qp_dom_vs_egm >= 0.70
                and qp_dom_vs_nog >= 0.70
                and clean_ratio_vs_egm >= 0.80
                and not_final_only == 1
                and useful_geometry
                and safe_float(row.get("avg_fallback_to_egm_frac", math.nan)) < 0.35
            )
            strong_positive = bool(
                improve_vs_sgd >= 0.10
                and improve_vs_egm >= 0.10
                and improve_vs_nog >= 0.10
                and qp_dom_vs_sgd >= 0.80
                and qp_dom_vs_egm >= 0.80
                and qp_dom_vs_nog >= 0.80
                and clean_ratio_vs_egm >= 0.85
                and not_final_only == 1
                and safe_float(row.get("avg_gamma_active_frac", math.nan)) >= 0.50
                and safe_float(row.get("avg_G_contribution_ratio", math.nan)) >= 0.20
                and safe_float(row.get("avg_QP_better_than_EGM_drift_fraction", math.nan)) >= 0.70
                and safe_float(row.get("avg_fallback_to_egm_frac", math.nan)) < 0.25
            )
            rank_rows.append(
                {
                    "optimizer_scope": cfg.optimizer_scope,
                    "alpha": cfg.alpha,
                    "shared_lr": cfg.shared_lr,
                    "method": row["method"],
                    "current_adv_eval_return_AUC": qp_auc,
                    "clean_eval_return_AUC": safe_float(row["clean_eval_return_AUC"]),
                    "improve_vs_sgd": improve_vs_sgd,
                    "improve_vs_egm": improve_vs_egm,
                    "improve_vs_nog": improve_vs_nog,
                    "dom_vs_sgd": qp_dom_vs_sgd,
                    "dom_vs_egm": qp_dom_vs_egm,
                    "dom_vs_nog": qp_dom_vs_nog,
                    "clean_ratio_vs_egm": clean_ratio_vs_egm,
                    "not_final_only": not_final_only,
                    "useful_geometry_flag": int(useful_geometry),
                    "qp_weak_positive": int(weak_positive),
                    "qp_strong_positive": int(strong_positive),
                    "avg_fallback_to_egm_frac": safe_float(row.get("avg_fallback_to_egm_frac", math.nan)),
                    "avg_gamma_active_frac": safe_float(row.get("avg_gamma_active_frac", math.nan)),
                    "avg_G_contribution_ratio": safe_float(row.get("avg_G_contribution_ratio", math.nan)),
                    "avg_QP_better_than_EGM_drift_fraction": safe_float(row.get("avg_QP_better_than_EGM_drift_fraction", math.nan)),
                    "avg_cos_F_G": safe_float(row.get("avg_cos_F_G", math.nan)),
                    "avg_non_collinearity": safe_float(row.get("avg_non_collinearity", math.nan)),
                }
            )

    rank_df = pd.DataFrame(rank_rows).sort_values(["qp_strong_positive", "qp_weak_positive", "improve_vs_egm", "dom_vs_egm"], ascending=[False, False, False, False]).reset_index(drop=True)
    rank_df.to_csv(stage2_root / "lyapunov_variant_ranked.csv", index=False)
    rank_df.to_csv(output_root / "lyapunov_variant_ranked.csv", index=False)

    if not rank_df.empty:
        best = rank_df.iloc[0]
        best_cfg_slug = ConfigKey(str(best["optimizer_scope"]), float(best["alpha"]), float(best["shared_lr"])).slug
        config_curves = {
            method: frame
            for (slug, method), frame in all_stage2_curves.items()
            if slug == best_cfg_slug and method in {"sgd_gda", "egm", "proposed_nog_closed", best["method"]}
        }
        plot_top_curves(config_curves, plots_root / "lyapunov_variant_top_curves.png", f"Top config: {best_cfg_slug}")
        plot_geometry(rank_df, plots_root / "lyapunov_variant_geometry.png")
        qp_vs_egm = {
            "egm": all_stage2_curves[(best_cfg_slug, "egm")],
            "qp": all_stage2_curves[(best_cfg_slug, best["method"])],
        }
        plot_top_curves(qp_vs_egm, plots_root / "lyapunov_variant_qp_vs_egm.png", f"QP vs EGM: {best_cfg_slug}")

    strong_exists = bool((rank_df["qp_strong_positive"] == 1).any()) if not rank_df.empty else False
    weak_exists = bool((rank_df["qp_weak_positive"] == 1).any()) if not rank_df.empty else False
    if strong_exists:
        final_decision = "LYA_REPAIR_QP_STRONG_POSITIVE"
    elif weak_exists:
        final_decision = "LYA_REPAIR_QP_WEAK_POSITIVE"
    elif not stage2_configs:
        final_decision = "LYA_REPAIR_FAIL_NO_EGM_REGIME"
    else:
        best_drift = safe_float(rank_df["avg_QP_better_than_EGM_drift_fraction"].max(), math.nan) if not rank_df.empty else math.nan
        final_decision = "LYA_REPAIR_LOCAL_DRIFT_ONLY" if finite(best_drift) and best_drift >= 0.60 else "LYA_REPAIR_FAIL_QP_NOT_BETTER_THAN_EGM"

    report_lines = [
        f"Decision: `{final_decision}`",
        "",
        "**Stage 0**",
        f"- existing EGM candidate decision: `{stage0_decision}`",
        f"- candidates found: `{len(stage0_candidates)}`",
        "",
        "**Stage 1**",
        f"- searched new EGM regime only because Stage 0 found fewer than two candidates",
        f"- selected Stage 2 configs: `{', '.join(cfg.slug for cfg in stage2_configs) if stage2_configs else 'none'}`",
        "",
        "**Merit Variants**",
        "- `proposed_nog_closed` uses a scope-matched actual PPO merit: `total_loss` for `full_policy`, and `total_loss - vf_coef * value_loss` for `actor_logstd_only`.",
        "- `proposed_qp_closed_current_merit_lF0p01_lR1` keeps the existing raw field + surrogate-style control merit.",
        "- `proposed_qp_closed_actual_ppo_merit_lF0p001_lR1` uses the same scope-matched actual PPO merit with stronger return weighting.",
        "- `proposed_qp_closed_trustreg_lF0p01_lR0p3_k0p3_c0p1` uses clipped-policy loss plus KL and clip-fraction penalties on the same frozen batch.",
        "- The lambda grid was intentionally reduced to one representative setting per merit family to keep this a true minisearch.",
        "",
        "**Top QP Variants**",
    ]
    if rank_df.empty:
        report_lines.append("- no Stage 2 QP variants completed")
    else:
        for row in rank_df.head(8).itertuples():
            report_lines.append(
                f"- scope=`{row.optimizer_scope}`, alpha=`{row.alpha}`, lr=`{row.shared_lr}`, method=`{row.method}`: "
                f"vs EGM improve=`{row.improve_vs_egm:.3f}`, dom=`{row.dom_vs_egm:.3f}`, "
                f"fallback=`{row.avg_fallback_to_egm_frac:.3f}`, gamma-active=`{row.avg_gamma_active_frac:.3f}`, "
                f"G-ratio=`{row.avg_G_contribution_ratio:.3f}`, drift-win=`{row.avg_QP_better_than_EGM_drift_fraction:.3f}`, "
                f"weak=`{row.qp_weak_positive}`, strong=`{row.qp_strong_positive}`"
            )
    report_lines.extend(
        [
            "",
            "**Pass/Fail framing**",
            "- PPM-inner5 was not used as the pass/fail target in this minisearch.",
            "- The target chain was `QP > EGM > SGD/noG` under equal-budget one-step methods only.",
        ]
    )
    (stage2_root / "lyapunov_variant_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    (output_root / "lyapunov_variant_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    top_lines = [f"Decision: `{final_decision}`", "", "Top configs:"]
    if rank_df.empty:
        top_lines.append("- none")
    else:
        for row in rank_df.head(5).itertuples():
            top_lines.append(
                f"- `{row.method}` on scope=`{row.optimizer_scope}`, alpha=`{row.alpha}`, lr=`{row.shared_lr}`: "
                f"vs-EGM AUC improve=`{row.improve_vs_egm:.3f}`, dominance=`{row.dom_vs_egm:.3f}`, "
                f"fallback=`{row.avg_fallback_to_egm_frac:.3f}`"
            )
    (stage2_root / "lyapunov_variant_top_configs.md").write_text("\n".join(top_lines), encoding="utf-8")
    (output_root / "lyapunov_variant_top_configs.md").write_text("\n".join(top_lines), encoding="utf-8")
    (output_root / "lyapunov_repair_final_decision.md").write_text(final_decision + "\n", encoding="utf-8")
    (output_root / "lyapunov_repair_final_report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
