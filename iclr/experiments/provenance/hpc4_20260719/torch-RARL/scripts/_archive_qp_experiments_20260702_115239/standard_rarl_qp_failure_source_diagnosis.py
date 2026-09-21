from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import pathlib
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


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


def load_module(module_name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ensure_dir(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_float(value, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


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


def safe_corr(a: Sequence[float], b: Sequence[float]) -> float:
    av = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(dtype=float)
    bv = pd.to_numeric(pd.Series(b), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(av) & np.isfinite(bv)
    if mask.sum() < 2:
        return math.nan
    if np.nanstd(av[mask]) <= 1e-12 or np.nanstd(bv[mask]) <= 1e-12:
        return math.nan
    return float(np.corrcoef(av[mask], bv[mask])[0, 1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Standard RARL QP failure-source diagnosis")
    repo_default = pathlib.Path(__file__).resolve().parents[1]
    parser.add_argument("--repo-dir", type=str, default=str(repo_default))
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(repo_default.parent / "results" / "standard_rarl_qp_failure_source_diagnosis"),
    )
    parser.add_argument("--env", type=str, default="HalfCheetah-v5")
    parser.add_argument("--fallback-env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def register_optimizers(repo_dir: pathlib.Path) -> None:
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from models.optimizers import OPTIMIZER_REGISTRY
    from models.proposed_qp_closedlyap_merit import (
        ProposedNoGClosedTrustRegionMeritOptimizer,
        ProposedQPClosedTrustRegionMeritOptimizer,
    )

    OPTIMIZER_REGISTRY["proposed_nog_closed_trustregion"] = ProposedNoGClosedTrustRegionMeritOptimizer
    OPTIMIZER_REGISTRY["proposed_qp_closed_trustregion"] = ProposedQPClosedTrustRegionMeritOptimizer


def make_diag_kwargs(method_label: str, role: str, cfg: ConfigKey, diagnostics_root: pathlib.Path) -> Dict[str, object]:
    return {
        "optimizer_scope": cfg.optimizer_scope,
        "lambda_F": 0.01,
        "lambda_R": 0.3,
        "lambda_KL": 0.3,
        "lambda_CF": 0.1,
        "target_kl": 0.03,
        "vf_coef": 0.5,
        "fd_eps": 1e-3,
        "beta_probe": cfg.shared_lr,
        "gamma_probe": max(cfg.shared_lr * cfg.shared_lr, 1e-6),
        "ridge": 1e-8,
        "beta_max": 3.0 * cfg.shared_lr,
        "gamma_max": 3.0 * max(cfg.shared_lr * cfg.shared_lr, 1e-6),
        "max_update_norm": 0.005,
        "qp_eps": 1e-8,
        "allow_fallback_to_egm": True,
        "fallback_tolerance": 0.0,
        "cost_mode": "trust_region_policy_kl_clip",
        "role": role,
        "diagnostics_csv_path": str(diagnostics_root / f"{role}_{method_label}.csv"),
    }


def alternating_methods(cfg: ConfigKey, diagnostics_root: pathlib.Path) -> List[MethodSpec]:
    return [
        MethodSpec("sgd_gda", "sgd", {}, {}, cfg.shared_lr, 10.0, 0.5),
        MethodSpec("egm", "egm", {}, {}, cfg.shared_lr, 10.0, 0.5),
        MethodSpec(
            "proposed_nog_closed_trustreg",
            "proposed_nog_closed_trustregion",
            make_diag_kwargs("nog_trustreg", "protagonist", cfg, diagnostics_root),
            make_diag_kwargs("nog_trustreg", "adversary", cfg, diagnostics_root),
            cfg.shared_lr,
            10.0,
            0.5,
        ),
        MethodSpec(
            "proposed_qp_closed_trustreg",
            "proposed_qp_closed_trustregion",
            make_diag_kwargs("qp_trustreg", "protagonist", cfg, diagnostics_root),
            make_diag_kwargs("qp_trustreg", "adversary", cfg, diagnostics_root),
            cfg.shared_lr,
            10.0,
            0.5,
        ),
    ]


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


def run_with_exp_manager(
    *,
    repo_dir: pathlib.Path,
    env_id: str,
    run_root: pathlib.Path,
    ns: SimpleNamespace,
) -> pathlib.Path:
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from utils.exp_manager import ExperimentManager

    ensure_dir(run_root)
    stdout_path = run_root / "stdout.txt"
    stderr_path = run_root / "stderr.txt"
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
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
                raise RuntimeError("Unexpected hyperparameter-optimization branch")
            manager.learn(model)
            manager.save_trained_model(model)

    env_root = run_root / "sm" / "rarl-ppo" / env_id
    run_dirs = [path for path in env_root.iterdir() if path.is_dir() and path.name.startswith(f"{env_id}_")]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {env_root}")
    return max(run_dirs, key=lambda path: int(path.name.rsplit("_", 1)[-1]))


def read_diag_pair(run_root: pathlib.Path, method_stem: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for role in ["protagonist", "adversary"]:
        path = run_root / "diagnostics" / f"{role}_{method_stem}.csv"
        if path.exists():
            frame = pd.read_csv(path)
            if not frame.empty:
                frame["diag_role"] = role
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def aggregate_qp_diag(diag_df: pd.DataFrame) -> Dict[str, float | str]:
    if diag_df.empty:
        return {}
    out: Dict[str, float | str] = {}
    numeric_cols = [
        "field_norm",
        "G_norm",
        "cos_F_G",
        "non_collinearity",
        "beta",
        "gamma",
        "gamma_active",
        "G_contribution_ratio",
        "QP_better_than_noG",
        "QP_better_than_EGM",
        "noG_better_than_QP",
        "actual_drift_noG",
        "actual_drift_QP",
        "actual_drift_EGM",
        "predicted_drift_noG",
        "predicted_drift_QP",
        "V_before",
        "V_after_noG",
        "V_after_QP",
        "V_after_EGM",
        "G_plus_improves",
        "G_minus_improves",
    ]
    for col in numeric_cols:
        if col in diag_df.columns:
            out[f"avg_{col}"] = float(pd.to_numeric(diag_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).mean())
    if {"gamma", "field_norm", "G_norm", "beta"}.issubset(diag_df.columns):
        gamma = pd.to_numeric(diag_df["gamma"], errors="coerce").fillna(0.0).abs()
        field = pd.to_numeric(diag_df["field_norm"], errors="coerce").fillna(0.0).abs()
        gnorm = pd.to_numeric(diag_df["G_norm"], errors="coerce").fillna(0.0).abs()
        beta = pd.to_numeric(diag_df["beta"], errors="coerce").fillna(0.0).abs()
        ratio = (gamma * gnorm) / (gamma * gnorm + beta * field + EPS)
        out["gamma_active_frac"] = float((gamma > 1e-8).mean())
        out["G_contribution_ratio"] = float(ratio.mean())
    else:
        out["gamma_active_frac"] = math.nan
        out["G_contribution_ratio"] = math.nan

    if "G_sign_preference" in diag_df.columns:
        prefs = diag_df["G_sign_preference"].fillna("none").astype(str)
        out["G_sign_preference_mode"] = prefs.mode().iloc[0] if not prefs.mode().empty else "none"
    return out


def parse_alignment(diag_df: pd.DataFrame, curve: pd.DataFrame) -> Dict[str, float]:
    if diag_df.empty or curve.empty or "current_adv_eval_return" not in curve.columns:
        return {
            "corr_actual_drift_QP_next_current_adv_change": math.nan,
            "corr_actual_drift_noG_next_current_adv_change": math.nan,
            "corr_QP_advantage_vs_return_advantage": math.nan,
        }
    eval_curve = curve[["timesteps", "current_adv_eval_return", "clean_eval_return", "train_return"]].copy()
    eval_curve["next_current_adv_return_change"] = pd.to_numeric(eval_curve["current_adv_eval_return"], errors="coerce").shift(-1) - pd.to_numeric(eval_curve["current_adv_eval_return"], errors="coerce")
    eval_curve["next_clean_eval_return_change"] = pd.to_numeric(eval_curve["clean_eval_return"], errors="coerce").shift(-1) - pd.to_numeric(eval_curve["clean_eval_return"], errors="coerce")
    eval_curve["next_train_return_change"] = pd.to_numeric(eval_curve["train_return"], errors="coerce").shift(-1) - pd.to_numeric(eval_curve["train_return"], errors="coerce")
    deltas = eval_curve.dropna(subset=["next_current_adv_return_change"]).reset_index(drop=True)
    if deltas.empty:
        return {
            "corr_actual_drift_QP_next_current_adv_change": math.nan,
            "corr_actual_drift_noG_next_current_adv_change": math.nan,
            "corr_QP_advantage_vs_return_advantage": math.nan,
        }

    step_df = diag_df.copy().reset_index(drop=True)
    step_df["actual_drift_QP"] = pd.to_numeric(step_df.get("actual_drift_QP", np.nan), errors="coerce")
    step_df["actual_drift_noG"] = pd.to_numeric(step_df.get("actual_drift_noG", np.nan), errors="coerce")
    step_df["predicted_drift_QP"] = pd.to_numeric(step_df.get("predicted_drift_QP", np.nan), errors="coerce")
    n_steps = len(step_df)
    n_eval = len(deltas)
    bucket = np.minimum((np.arange(n_steps) * n_eval) // max(n_steps, 1), n_eval - 1)
    step_df["eval_bucket"] = bucket
    agg = (
        step_df.groupby("eval_bucket", as_index=False)[["actual_drift_QP", "actual_drift_noG", "predicted_drift_QP"]]
        .mean()
        .merge(deltas.reset_index().rename(columns={"index": "eval_bucket"}), on="eval_bucket", how="left")
    )
    qp_advantage = agg["actual_drift_noG"] - agg["actual_drift_QP"]
    return_advantage = agg["next_current_adv_return_change"]
    return {
        "corr_actual_drift_QP_next_current_adv_change": safe_corr(agg["actual_drift_QP"], agg["next_current_adv_return_change"]),
        "corr_actual_drift_noG_next_current_adv_change": safe_corr(agg["actual_drift_noG"], agg["next_current_adv_return_change"]),
        "corr_QP_advantage_vs_return_advantage": safe_corr(qp_advantage, return_advantage),
    }


def setup_manager_only(repo_dir: pathlib.Path, env_id: str, ns: SimpleNamespace):
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from utils.exp_manager import ExperimentManager

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
        model_path=str(pathlib.Path(ns.saved_models_path) / "rarl-ppo" / env_id),
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
        raise RuntimeError("Unexpected hyperparameter-optimization branch")
    return manager, model


def run_joint_smoke(
    *,
    repo_dir: pathlib.Path,
    hyperparam_dir: pathlib.Path,
    output_root: pathlib.Path,
    env_id: str,
    fallback_env_id: str,
    device: str,
    seed: int,
) -> pd.DataFrame:
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from models.optimizers import apply_state_delta, clone_named_state, restore_named_state
    from models.proposed_qp_closedlyap_merit import (
        ProposedNoGClosedTrustRegionMeritOptimizer,
        ProposedQPClosedTrustRegionMeritOptimizer,
    )
    from utils.callbacks import SetupAdvTrainingCallback, SetupProTrainingCallback

    cfg = ConfigKey("actor_logstd_only", 0.3, 1e-3)
    dummy_method = MethodSpec("sgd_gda", "sgd", {}, {}, cfg.shared_lr, 10.0, 0.5)
    run_root = ensure_dir(output_root / "joint_smoke")
    ns = build_ns(
        env_id=env_id,
        seed=seed,
        device=device,
        iterations=6,
        eval_freq=10240,
        n_eval_episodes=5,
        optimizer_scope=cfg.optimizer_scope,
        adv_fraction=cfg.alpha,
        method=dummy_method,
        hyperparam_dir=hyperparam_dir,
        run_root=run_root,
        n_mu=5,
        n_nu=1,
    )
    smoke_env_id = env_id
    try:
        _, rarl_model = setup_manager_only(repo_dir, smoke_env_id, ns)
    except Exception:
        if fallback_env_id == env_id:
            raise
        smoke_env_id = fallback_env_id
        ns.env = smoke_env_id
        _, rarl_model = setup_manager_only(repo_dir, smoke_env_id, ns)
    probe_module = load_module("qp_fail_probe_mod", repo_dir / "scripts" / "full_policy_optimizer_probe.py")

    def collect_first_batch(role: str):
        if role == "protagonist":
            algo = rarl_model.protagonist
            callback = [SetupProTrainingCallback(rarl_model.adversary.policy)]
        else:
            algo = rarl_model.adversary
            callback = [SetupAdvTrainingCallback(rarl_model.protagonist.policy)]
        original_train = algo.train
        algo.train = lambda: None
        try:
            algo.learn(
                algo.n_steps * algo.env.num_envs,
                    callback=callback,
                    log_interval=1,
                    reset_num_timesteps=True,
                )
        finally:
            algo.train = original_train
        batch = next(iter(algo.rollout_buffer.get(algo.batch_size)))
        actions = batch.actions.long().flatten() if algo.action_space.__class__.__name__ == "Discrete" else batch.actions
        clip_range = algo.clip_range(algo._current_progress_remaining)
        clip_range_vf = None if algo.clip_range_vf is None else algo.clip_range_vf(algo._current_progress_remaining)
        eval_closure = algo._build_eval_closure(batch, actions, clip_range, clip_range_vf)
        named_params = [(name, param) for name, param in algo.policy.named_parameters() if param.requires_grad]
        return algo, batch, eval_closure, named_params

    pro_algo, _, pro_eval_closure, pro_named = collect_first_batch("protagonist")
    adv_algo, _, adv_eval_closure, adv_named = collect_first_batch("adversary")

    def make_role_tools(algo, named_params, role: str):
        noG = ProposedNoGClosedTrustRegionMeritOptimizer(
            [param for _, param in named_params],
            lr=cfg.shared_lr,
            optimizer_scope=cfg.optimizer_scope,
            lambda_F=0.01,
            lambda_R=0.3,
            lambda_KL=0.3,
            lambda_CF=0.1,
            vf_coef=0.5,
            fd_eps=1e-3,
            beta_probe=cfg.shared_lr,
            gamma_probe=max(cfg.shared_lr * cfg.shared_lr, 1e-6),
            max_update_norm=0.005,
            beta_max=3.0 * cfg.shared_lr,
            gamma_max=3.0 * max(cfg.shared_lr * cfg.shared_lr, 1e-6),
            allow_fallback_to_egm=False,
            role=role,
        )
        qp = ProposedQPClosedTrustRegionMeritOptimizer(
            [param for _, param in named_params],
            lr=cfg.shared_lr,
            optimizer_scope=cfg.optimizer_scope,
            lambda_F=0.01,
            lambda_R=0.3,
            lambda_KL=0.3,
            lambda_CF=0.1,
            vf_coef=0.5,
            fd_eps=1e-3,
            beta_probe=cfg.shared_lr,
            gamma_probe=max(cfg.shared_lr * cfg.shared_lr, 1e-6),
            max_update_norm=0.005,
            beta_max=3.0 * cfg.shared_lr,
            gamma_max=3.0 * max(cfg.shared_lr * cfg.shared_lr, 1e-6),
            allow_fallback_to_egm=False,
            role=role,
        )
        selected_names = noG._selected_names(named_params)

        def eval_state(optimizer, eval_closure, theta_state):
            return optimizer._evaluate_state(
                eval_closure=eval_closure,
                theta_state=theta_state,
                selected_names=selected_names,
                candidate_merit_evaluator=None,
            )

        return noG, qp, selected_names, eval_state

    pro_noG, pro_qp, pro_selected, pro_eval_state = make_role_tools(pro_algo, pro_named, "protagonist")
    adv_noG, adv_qp, adv_selected, adv_eval_state = make_role_tools(adv_algo, adv_named, "adversary")

    init_pro = clone_named_state(pro_named)
    init_adv = clone_named_state(adv_named)
    methods = ["sgd", "egm", "nog", "qp"]
    state_bank = {
        method: {"pro": {k: v.clone() for k, v in init_pro.items()}, "adv": {k: v.clone() for k, v in init_adv.items()}}
        for method in methods
    }
    rows: List[Dict[str, object]] = []

    def role_step(method: str, theta_old, optimizer_noG, optimizer_qp, selected_names, eval_closure, eval_state_fn, named_params):
        base_eval = eval_state_fn(optimizer_noG, eval_closure, theta_old)
        if method == "sgd":
            theta_new = {name: tensor.clone() for name, tensor in theta_old.items()}
            for name in selected_names:
                theta_new[name] = theta_old[name] - cfg.shared_lr * base_eval["grads_selected"][name]
            return theta_new, base_eval, {"beta": cfg.shared_lr, "gamma": 0.0}
        if method == "egm":
            theta_egm, _ = optimizer_noG._egm_state(
                theta_old=theta_old,
                f_old=base_eval["grads_selected"],
                selected_names=selected_names,
                eval_closure=eval_closure,
                eta=cfg.shared_lr,
            )
            return theta_egm, base_eval, {"beta": cfg.shared_lr, "gamma": 0.0}
        if method == "nog":
            step_out = optimizer_noG._step_impl(
                theta_old=theta_old,
                selected_names=selected_names,
                base_eval=base_eval,
                eval_state=lambda state: eval_state_fn(optimizer_noG, eval_closure, state),
                eval_closure=eval_closure,
                eta=cfg.shared_lr,
            )
            return step_out["theta_candidate"], base_eval, step_out
        step_out = optimizer_qp._step_impl(
            theta_old=theta_old,
            selected_names=selected_names,
            base_eval=base_eval,
            eval_state=lambda state: eval_state_fn(optimizer_qp, eval_closure, state),
            eval_closure=eval_closure,
            eta=cfg.shared_lr,
        )
        return step_out["theta_candidate"], base_eval, step_out

    for iter_idx in range(6):
        for method in methods:
            restore_named_state(pro_named, state_bank[method]["pro"])
            restore_named_state(adv_named, state_bank[method]["adv"])
            pro_new, pro_base, pro_step = role_step(method, state_bank[method]["pro"], pro_noG, pro_qp, pro_selected, pro_eval_closure, pro_eval_state, pro_named)
            adv_new, adv_base, adv_step = role_step(method, state_bank[method]["adv"], adv_noG, adv_qp, adv_selected, adv_eval_closure, adv_eval_state, adv_named)
            pro_after = pro_eval_state(pro_noG if method != "qp" else pro_qp, pro_eval_closure, pro_new)
            adv_after = adv_eval_state(adv_noG if method != "qp" else adv_qp, adv_eval_closure, adv_new)
            v_before = float(pro_base["V"]) + float(adv_base["V"])
            v_after = float(pro_after["V"]) + float(adv_after["V"])
            rows.append(
                {
                    "iteration": iter_idx,
                    "method": method,
                    "V_before": v_before,
                    "V_after": v_after,
                    "actual_drift": v_after - v_before,
                    "pro_field_norm": safe_float(pro_base.get("field_term", math.nan)),
                    "adv_field_norm": safe_float(adv_base.get("field_term", math.nan)),
                    "pro_beta": safe_float(pro_step.get("beta", math.nan)),
                    "adv_beta": safe_float(adv_step.get("beta", math.nan)),
                    "pro_gamma": safe_float(pro_step.get("gamma", math.nan)),
                    "adv_gamma": safe_float(adv_step.get("gamma", math.nan)),
                }
            )
            state_bank[method]["pro"] = pro_new
            state_bank[method]["adv"] = adv_new

    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "joint_smoke_summary.csv", index=False)
    return frame


def plot_curves(curve_map: Dict[str, pd.DataFrame], output_path: pathlib.Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [
        ("train_return", "Train Return"),
        ("clean_eval_return", "Clean Eval Return"),
        ("current_adv_eval_return", "Current-Adv Eval Return"),
        ("current_adv_degradation", "Current-Adv Degradation"),
    ]
    for ax, (col, label) in zip(axes.flatten(), metrics):
        for method, frame in curve_map.items():
            if col in frame.columns:
                ax.plot(frame["timesteps"], frame[col], label=method, linewidth=1.5)
        ax.set_title(label)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_joint_smoke(frame: pd.DataFrame, output_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for method, group in frame.groupby("method"):
        ax.plot(group["iteration"], group["actual_drift"], marker="o", label=method)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_title("Joint Frozen-Batch Smoke: Actual Drift")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Actual Drift")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_dir = pathlib.Path(args.repo_dir).resolve()
    output_root = ensure_dir(pathlib.Path(args.output_root).resolve())
    plots_root = ensure_dir(output_root / "plots")
    hyperparam_dir = repo_dir / "hyperparameter"

    register_optimizers(repo_dir)
    base_module = load_module("qp_fail_base_mod", repo_dir / "scripts" / "standard_rarl_baseline_positive_search.py")
    helper_module = load_module("qp_fail_helper_mod", repo_dir / "scripts" / "standard_rarl_lyapunov_repair_minisearch.py")

    main_cfg = ConfigKey("full_policy", 0.05, 3e-4)
    alt_root = ensure_dir(output_root / "alternating_main")
    diagnostics_root = ensure_dir(alt_root / "diagnostics")
    curve_map: Dict[str, pd.DataFrame] = {}
    alt_rows: List[Dict[str, object]] = []
    resolved_env_ids: List[str] = []

    for method in alternating_methods(main_cfg, diagnostics_root):
        run_root = ensure_dir(alt_root / method.label)
        analysis_dir = ensure_dir(run_root / "analysis")
        if (analysis_dir / "run_summary.csv").exists():
            env_id_for_analysis = args.env if (run_root / "sm" / "rarl-ppo" / args.env).exists() else args.fallback_env
            run_dir = helper_module.try_latest_run_dir(run_root, env_id_for_analysis)
            if run_dir is None:
                raise FileNotFoundError(f"Could not find saved run directory under {run_root}")
            summary, curve = helper_module.load_analyzed_curve(
                base_module,
                run_dir,
                analysis_dir,
                method.label,
            )
        else:
            ns = build_ns(
                env_id=args.env,
                seed=args.seed,
                device=args.device,
                iterations=4,
                eval_freq=10240,
                n_eval_episodes=5,
                optimizer_scope=main_cfg.optimizer_scope,
                adv_fraction=main_cfg.alpha,
                method=method,
                hyperparam_dir=hyperparam_dir,
                run_root=run_root,
                n_mu=5,
                n_nu=1,
            )
            try:
                run_dir = run_with_exp_manager(repo_dir=repo_dir, env_id=args.env, run_root=run_root, ns=ns)
            except Exception:
                if args.env != args.fallback_env:
                    ns.env = args.fallback_env
                    run_dir = run_with_exp_manager(repo_dir=repo_dir, env_id=args.fallback_env, run_root=run_root, ns=ns)
                else:
                    raise
            summary, curve = helper_module.analyze_run(base_module, repo_dir, run_dir, analysis_dir, method.label)
        curve_map[method.label] = curve
        if "env_id" in summary:
            resolved_env_ids.append(str(summary["env_id"]))
        row = {
            "method": method.label,
            "optimizer_scope": main_cfg.optimizer_scope,
            "alpha": main_cfg.alpha,
            "shared_lr": main_cfg.shared_lr,
            "train_return_AUC": auc_from_curve(curve, "timesteps", "train_return"),
            "clean_eval_return_AUC": auc_from_curve(curve, "timesteps", "clean_eval_return"),
            "current_adv_eval_return_AUC": auc_from_curve(curve, "timesteps", "current_adv_eval_return"),
            "current_adv_degradation_AUC": auc_from_curve(curve, "timesteps", "current_adv_degradation"),
            "field_norm_AUC": auc_from_curve(curve, "timesteps", "field_norm"),
            "surrogate_lyapunov_AUC": auc_from_curve(curve, "timesteps", "surrogate_lyapunov_value"),
        }
        alt_rows.append(row)

    plot_curves(curve_map, plots_root / "alternating_main_curves.png", f"Alternating diagnosis: {main_cfg.slug}")
    resolved_env = resolved_env_ids[0] if resolved_env_ids else args.env

    qp_diag = read_diag_pair(alt_root, "qp_trustreg")
    nog_diag = read_diag_pair(alt_root, "nog_trustreg")
    qp_diag.to_csv(output_root / "qp_step_level_diagnostics.csv", index=False)
    if not nog_diag.empty:
        nog_diag.to_csv(output_root / "nog_step_level_diagnostics.csv", index=False)
    qp_local = aggregate_qp_diag(qp_diag)
    qp_alignment = parse_alignment(qp_diag, curve_map.get("proposed_qp_closed_trustreg", pd.DataFrame()))

    joint_frame = run_joint_smoke(
        repo_dir=repo_dir,
        hyperparam_dir=hyperparam_dir,
        output_root=output_root,
        env_id=args.env,
        fallback_env_id=args.fallback_env,
        device=args.device,
        seed=args.seed,
    )
    plot_joint_smoke(joint_frame, plots_root / "joint_smoke_drift.png")

    alt_df = pd.DataFrame(alt_rows)
    alt_df.to_csv(output_root / "alternating_summary.csv", index=False)

    current_adv_auc = {
        row["method"]: safe_float(row["current_adv_eval_return_AUC"], math.nan) for row in alt_rows
    }
    qp_auc = current_adv_auc.get("proposed_qp_closed_trustreg", math.nan)
    nog_auc = current_adv_auc.get("proposed_nog_closed_trustreg", math.nan)
    egm_auc = current_adv_auc.get("egm", math.nan)
    sgd_auc = current_adv_auc.get("sgd_gda", math.nan)
    qp_curve = curve_map["proposed_qp_closed_trustreg"]
    egm_curve = curve_map["egm"]
    nog_curve = curve_map["proposed_nog_closed_trustreg"]
    sgd_curve = curve_map["sgd_gda"]
    common = qp_curve[["timesteps", "current_adv_eval_return"]].rename(columns={"current_adv_eval_return": "qp"})
    common = common.merge(egm_curve[["timesteps", "current_adv_eval_return"]].rename(columns={"current_adv_eval_return": "egm"}), on="timesteps", how="inner")
    common = common.merge(nog_curve[["timesteps", "current_adv_eval_return"]].rename(columns={"current_adv_eval_return": "nog"}), on="timesteps", how="inner")
    common = common.merge(sgd_curve[["timesteps", "current_adv_eval_return"]].rename(columns={"current_adv_eval_return": "sgd"}), on="timesteps", how="inner")

    qp_vs_noG_fraction = safe_float(qp_local.get("avg_QP_better_than_noG", math.nan), math.nan)
    qp_vs_egm_fraction = safe_float(qp_local.get("avg_QP_better_than_EGM", math.nan), math.nan)
    gamma_active_frac = safe_float(qp_local.get("gamma_active_frac", math.nan), math.nan)
    g_ratio = safe_float(qp_local.get("G_contribution_ratio", math.nan), math.nan)
    g_plus = safe_float(qp_local.get("avg_G_plus_improves", math.nan), math.nan)
    g_minus = safe_float(qp_local.get("avg_G_minus_improves", math.nan), math.nan)
    corr_qp = safe_float(qp_alignment.get("corr_actual_drift_QP_next_current_adv_change", math.nan), math.nan)
    corr_nog = safe_float(qp_alignment.get("corr_actual_drift_noG_next_current_adv_change", math.nan), math.nan)
    corr_adv = safe_float(qp_alignment.get("corr_QP_advantage_vs_return_advantage", math.nan), math.nan)

    joint_pivot = joint_frame.pivot_table(index="iteration", columns="method", values="actual_drift", aggfunc="mean")
    joint_qp_beats = float(
        ((pd.to_numeric(joint_pivot.get("qp"), errors="coerce") < pd.concat(
            [
                pd.to_numeric(joint_pivot.get("sgd"), errors="coerce"),
                pd.to_numeric(joint_pivot.get("egm"), errors="coerce"),
                pd.to_numeric(joint_pivot.get("nog"), errors="coerce"),
            ],
            axis=1,
        ).min(axis=1) - 1e-12)).mean()
    ) if not joint_pivot.empty else math.nan

    if not finite(qp_vs_noG_fraction) or qp_vs_noG_fraction < 0.5 or not finite(gamma_active_frac) or gamma_active_frac < 0.2 or not finite(g_ratio) or g_ratio < 0.05:
        final_decision = "GEOMETRY_FAIL_NO_USEFUL_G"
    elif finite(g_minus) and finite(g_plus) and g_minus > g_plus + 0.15:
        final_decision = "POSSIBLE_G_SIGN_OR_SCALING_ISSUE"
    elif finite(qp_vs_noG_fraction) and qp_vs_noG_fraction >= 0.6 and finite(corr_adv) and corr_adv <= 0.05:
        final_decision = "LYAPUNOV_RETURN_MISMATCH"
    elif finite(joint_qp_beats) and joint_qp_beats >= 0.6 and (not finite(qp_vs_egm_fraction) or qp_vs_egm_fraction < 0.5):
        final_decision = "PROTOCOL_FAIL_ALTERNATING_ONLY"
    elif finite(joint_qp_beats) and joint_qp_beats < 0.5 and finite(qp_vs_noG_fraction) and qp_vs_noG_fraction < 0.6:
        final_decision = "ENV_OR_LYAPUNOV_FAIL"
    elif finite(qp_vs_noG_fraction) and qp_vs_noG_fraction >= 0.6 and finite(corr_adv) and corr_adv > 0.05:
        final_decision = "QP_MECHANISM_PRESENT_AND_RETURN_ALIGNED"
    else:
        final_decision = "QP_MECHANISM_PRESENT_BUT_WEAK"

    summary_row = {
        "env": resolved_env,
        "optimizer_scope": main_cfg.optimizer_scope,
        "alpha": main_cfg.alpha,
        "shared_lr": main_cfg.shared_lr,
        "qp_current_adv_eval_return_AUC": qp_auc,
        "nog_current_adv_eval_return_AUC": nog_auc,
        "egm_current_adv_eval_return_AUC": egm_auc,
        "sgd_current_adv_eval_return_AUC": sgd_auc,
        "qp_better_than_noG_fraction": qp_vs_noG_fraction,
        "qp_better_than_EGM_fraction": qp_vs_egm_fraction,
        "gamma_active_frac": gamma_active_frac,
        "G_contribution_ratio": g_ratio,
        "avg_cos_F_G": safe_float(qp_local.get("avg_cos_F_G", math.nan), math.nan),
        "avg_non_collinearity": safe_float(qp_local.get("avg_non_collinearity", math.nan), math.nan),
        "avg_field_norm": safe_float(qp_local.get("avg_field_norm", math.nan), math.nan),
        "avg_G_norm": safe_float(qp_local.get("avg_G_norm", math.nan), math.nan),
        "avg_V_before": safe_float(qp_local.get("avg_V_before", math.nan), math.nan),
        "avg_V_after_noG": safe_float(qp_local.get("avg_V_after_noG", math.nan), math.nan),
        "avg_V_after_QP": safe_float(qp_local.get("avg_V_after_QP", math.nan), math.nan),
        "avg_V_after_EGM": safe_float(qp_local.get("avg_V_after_EGM", math.nan), math.nan),
        "corr_actual_drift_QP_next_current_adv_change": corr_qp,
        "corr_actual_drift_noG_next_current_adv_change": corr_nog,
        "corr_QP_advantage_vs_return_advantage": corr_adv,
        "avg_G_plus_improves": g_plus,
        "avg_G_minus_improves": g_minus,
        "joint_qp_best_fraction": joint_qp_beats,
        "final_decision": final_decision,
    }
    pd.DataFrame([summary_row]).to_csv(output_root / "failure_source_summary.csv", index=False)

    report_lines = [
        "# QP Failure Source Report",
        "",
        f"- requested env: `{args.env}`",
        f"- resolved env used in runs: `{resolved_env}`",
        f"- main alternating config: `scope={main_cfg.optimizer_scope}, alpha={main_cfg.alpha}, lr={main_cfg.shared_lr}, seed={args.seed}, N_mu=5, N_nu=1, iterations=4`",
        "- reward/wrapper remained standard RARL only: original env reward, adversary reward = negative protagonist reward, additive `a_env = clip(u + alpha*w)`.",
        "",
        "## Part 1: Step-Level Local Geometry Test",
        "",
        f"- `QP_better_than_noG_fraction`: `{qp_vs_noG_fraction:.3f}`",
        f"- `QP_better_than_EGM_fraction`: `{qp_vs_egm_fraction:.3f}`",
        f"- `gamma_active_frac`: `{gamma_active_frac:.3f}`",
        f"- `G_contribution_ratio`: `{g_ratio:.3f}`",
        f"- `avg_cos_F_G`: `{safe_float(qp_local.get('avg_cos_F_G', math.nan), math.nan):.3f}`",
        f"- `avg_non_collinearity`: `{safe_float(qp_local.get('avg_non_collinearity', math.nan), math.nan):.3f}`",
        "",
        "## Part 2: G Sign Test",
        "",
        f"- `avg_G_plus_improves`: `{g_plus:.3f}`",
        f"- `avg_G_minus_improves`: `{g_minus:.3f}`",
        f"- `G_sign_preference_mode`: `{qp_local.get('G_sign_preference_mode', 'none')}`",
        "",
        "## Part 3: Lyapunov-Return Alignment",
        "",
        f"- `corr(actual_drift_QP, next_current_adv_return_change)`: `{corr_qp:.3f}`",
        f"- `corr(actual_drift_noG, next_current_adv_return_change)`: `{corr_nog:.3f}`",
        f"- `corr(QP local V advantage over noG, return advantage)`: `{corr_adv:.3f}`",
        "",
        "## Part 4: Tiny Joint Frozen-Batch Smoke",
        "",
        "- config: `scope=actor_logstd_only, alpha=0.3, joint_lr=1e-3, iterations=6`",
        f"- `joint_qp_best_fraction`: `{joint_qp_beats:.3f}`",
        "",
        "## A/B/C Diagnosis",
        "",
        f"- A usable-G check: `{'pass' if finite(qp_vs_noG_fraction) and qp_vs_noG_fraction >= 0.6 and finite(gamma_active_frac) and gamma_active_frac >= 0.3 and finite(g_ratio) and g_ratio >= 0.1 else 'fail_or_weak'}`",
        f"- B Lyapunov/return alignment: `{'mismatch' if finite(corr_adv) and corr_adv <= 0.05 else 'not_clearly_mismatched'}`",
        f"- C alternating-protocol breakage: `{'possible' if finite(joint_qp_beats) and joint_qp_beats >= 0.6 and (not finite(qp_vs_egm_fraction) or qp_vs_egm_fraction < 0.5) else 'not_supported'}`",
        "",
        f"Final decision: `{final_decision}`",
    ]
    (output_root / "failure_source_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (output_root / "failure_source_decision.md").write_text(final_decision + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
