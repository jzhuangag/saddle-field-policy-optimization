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
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class ConfirmedConfig:
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
class LambdaPair:
    lambda_F: float
    lambda_R: float

    @property
    def slug(self) -> str:
        lF = f"{self.lambda_F:g}".replace(".", "p")
        lR = f"{self.lambda_R:g}".replace(".", "p")
        return f"lF_{lF}_lR_{lR}"


@dataclass(frozen=True)
class MethodSpec:
    label: str
    optimizer: str
    protagonist_optimizer_kwargs: Dict[str, object]
    adversary_optimizer_kwargs: Dict[str, object]
    lr: float
    max_grad_norm: float
    vf_coef: float


CONFIRMED_CONFIGS = [
    ConfirmedConfig("full_policy", 0.05, 0.003),
    ConfirmedConfig("full_policy", 0.1, 0.003),
    ConfirmedConfig("actor_logstd_only", 0.3, 0.001),
    ConfirmedConfig("actor_logstd_only", 0.5, 0.001),
]

LAMBDA_PAIRS = [
    LambdaPair(0.01, 1.0),
    LambdaPair(0.001, 1.0),
    LambdaPair(0.01, 0.3),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 3B QP validation on confirmed HalfCheetah standard-RARL configs")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument(
        "--output-root",
        type=str,
        default=r"C:\Users\jhuangag\work\rarl\original\results\standard_rarl_tdd_autopilot\04_qp_validation_after_alpha_fix",
    )
    parser.add_argument("--env", type=str, default="HalfCheetah-v5")
    parser.add_argument("--fallback-env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--eval-freq", type=int, default=10240)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--ppm-inner-steps", type=int, default=5)
    parser.add_argument("--shared-max-grad-norm", type=float, default=10.0)
    parser.add_argument("--shared-vf-coef", type=float, default=0.5)
    parser.add_argument("--fd-eps", type=float, default=1e-3)
    parser.add_argument("--max-update-norm", type=float, default=0.005)
    parser.add_argument("--fallback-tolerance", type=float, default=0.0)
    return parser.parse_args()


def load_module(module_name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_args_namespace(
    *,
    args: argparse.Namespace,
    cfg: ConfirmedConfig,
    lambda_pair: LambdaPair,
    method: MethodSpec,
    hyperparam_dir: pathlib.Path,
    run_root: pathlib.Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        verbose=1,
        seed=args.seed,
        num_exps=1,
        num_threads=-1,
        env=args.env,
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
        n_opt_trials=10,
        no_optim_plots=False,
        n_jobs=1,
        n_startup_trials=10,
        n_evaluations_opt=20,
        n_timesteps=args.iterations,
        save_freq=args.eval_freq,
        log_interval=-1,
        device=args.device,
        eval_freq=args.eval_freq,
        n_eval_envs=1,
        n_eval_episodes=args.n_eval_episodes,
        control_proxy_eval=False,
        tensorboard_log=str(run_root / "tb"),
        log_folder=str(run_root / "logging"),
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
        optimizer_scope=cfg.optimizer_scope,
        qp_normalization="none",
        qp_g_alpha=1e-3,
        max_update_norm=args.max_update_norm,
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
        qp_fd_eps=args.fd_eps,
        qp_beta_probe=method.lr,
        qp_gamma_probe=max(method.lr * method.lr, 1e-6),
        qp_ridge=1e-8,
        qp_actor_weight=1.0,
        qp_logstd_weight=1.0,
        qp_step_solver="lyapunov_quadratic_bound",
        N_mu=args.N_mu,
        N_nu=args.N_nu,
        adv_impact="control",
        adv_delay=-1,
        adv_fraction=cfg.alpha,
        adv_fraction_override=True,
        requested_alpha=cfg.alpha,
        requested_adv_fraction=cfg.alpha,
        adv_index_list=["torso"],
        adv_force_dim=2,
        lambda_F=lambda_pair.lambda_F,
        lambda_R=lambda_pair.lambda_R,
    )


def build_method_specs(
    *,
    optimizer_scope: str,
    shared_lr: float,
    max_grad_norm: float,
    vf_coef: float,
    ppm_inner_steps: int,
    lambda_pair: LambdaPair,
    fd_eps: float,
    max_update_norm: float,
    fallback_tolerance: float,
    diagnostics_root: pathlib.Path,
) -> List[MethodSpec]:
    beta_probe = shared_lr
    gamma_probe = max(shared_lr * shared_lr, 1e-6)
    beta_max = 3.0 * shared_lr
    gamma_max = 3.0 * gamma_probe

    def closed_kwargs(method_label: str, role: str, allow_fallback: bool) -> Dict[str, object]:
        return {
            "optimizer_scope": optimizer_scope,
            "lambda_F": lambda_pair.lambda_F,
            "lambda_R": lambda_pair.lambda_R,
            "actor_weight": 1.0,
            "logstd_weight": 1.0,
            "critic_weight": 1.0,
            "fd_eps": fd_eps,
            "beta_probe": beta_probe,
            "gamma_probe": gamma_probe,
            "ridge": 1e-8,
            "beta_max": beta_max,
            "gamma_max": gamma_max,
            "max_update_norm": max_update_norm,
            "qp_eps": 1e-8,
            "allow_fallback_to_egm": allow_fallback,
            "fallback_tolerance": fallback_tolerance,
            "cost_mode": "unclipped_actor_surrogate_cost",
            "short_return_horizon": 16,
            "short_return_episodes": 1,
            "short_return_seed_offset": 0,
            "role": role,
            "diagnostics_csv_path": str(diagnostics_root / f"{role}_{method_label}_diagnostics.csv"),
        }

    return [
        MethodSpec("sgd_gda", "sgd", {}, {}, shared_lr, max_grad_norm, vf_coef),
        MethodSpec("egm", "egm", {}, {}, shared_lr, max_grad_norm, vf_coef),
        MethodSpec("ppm_inner5", "ppm", {"inner_steps": ppm_inner_steps}, {"inner_steps": ppm_inner_steps}, shared_lr, max_grad_norm, vf_coef),
        MethodSpec(
            "proposed_nog_closed",
            "proposed_nog_closedlyap",
            closed_kwargs("proposed_nog_closed", "protagonist", False),
            closed_kwargs("proposed_nog_closed", "adversary", False),
            shared_lr,
            max_grad_norm,
            vf_coef,
        ),
        MethodSpec(
            "proposed_qp_closed",
            "proposed_qp_closedlyap",
            closed_kwargs("proposed_qp_closed", "protagonist", True),
            closed_kwargs("proposed_qp_closed", "adversary", True),
            shared_lr,
            max_grad_norm,
            vf_coef,
        ),
    ]


def load_run_args(one_seed_module, run_dir: pathlib.Path) -> Dict[str, object]:
    return one_seed_module.load_run_args(run_dir)


def optimizer_kwargs_match(actual: Dict[str, object], expected: Dict[str, object]) -> bool:
    if not expected:
        return True
    if not isinstance(actual, dict):
        return False
    for key, value in expected.items():
        if key not in actual:
            return False
        actual_value = actual[key]
        if isinstance(value, float):
            if not math.isclose(float(actual_value), float(value), rel_tol=0.0, abs_tol=1e-12):
                return False
        else:
            if actual_value != value:
                return False
    return True


def existing_run_matches_request(
    one_seed_module,
    run_dir: pathlib.Path,
    *,
    ns: SimpleNamespace,
    method: MethodSpec,
) -> bool:
    run_args = load_run_args(one_seed_module, run_dir)
    if not run_args:
        return False
    expected = {
        "seed": ns.seed,
        "env": ns.env,
        "n_timesteps": ns.n_timesteps,
        "protagonist_optimizer": method.optimizer,
        "adversary_optimizer": method.optimizer,
        "optimizer_scope": ns.optimizer_scope,
        "N_mu": ns.N_mu,
        "N_nu": ns.N_nu,
        "adv_impact": ns.adv_impact,
        "adv_fraction": ns.adv_fraction,
        "requested_adv_fraction": ns.requested_adv_fraction,
        "lambda_F": ns.lambda_F,
        "lambda_R": ns.lambda_R,
    }
    for key, value in expected.items():
        if key not in run_args:
            return False
        actual = run_args.get(key)
        if isinstance(value, float):
            if not math.isclose(float(actual), float(value), rel_tol=0.0, abs_tol=1e-12):
                return False
        else:
            if actual != value:
                return False
    if not math.isclose(float(run_args.get("protagonist_lr", float("nan"))), float(method.lr), rel_tol=0.0, abs_tol=1e-12):
        return False
    if not math.isclose(float(run_args.get("adversary_lr", float("nan"))), float(method.lr), rel_tol=0.0, abs_tol=1e-12):
        return False
    if not optimizer_kwargs_match(run_args.get("protagonist_optimizer_kwargs", {}), method.protagonist_optimizer_kwargs):
        return False
    if not optimizer_kwargs_match(run_args.get("adversary_optimizer_kwargs", {}), method.adversary_optimizer_kwargs):
        return False
    return True


def find_latest_run_dir(saved_models_dir: pathlib.Path, env_id: str) -> pathlib.Path:
    env_root = saved_models_dir / "rarl-ppo" / env_id
    run_dirs = [path for path in env_root.iterdir() if path.is_dir() and path.name.startswith(f"{env_id}_")]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories under {env_root}")
    return max(run_dirs, key=lambda path: int(path.name.rsplit("_", 1)[-1]))


def try_find_latest_run_dir(saved_models_dir: pathlib.Path, env_id: str) -> pathlib.Path | None:
    env_root = saved_models_dir / "rarl-ppo" / env_id
    if not env_root.exists():
        return None
    run_dirs = [path for path in env_root.iterdir() if path.is_dir() and path.name.startswith(f"{env_id}_")]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda path: int(path.name.rsplit("_", 1)[-1]))


def run_method(
    *,
    one_seed_module,
    args: argparse.Namespace,
    cfg: ConfirmedConfig,
    lambda_pair: LambdaPair,
    repo_dir: pathlib.Path,
    hyperparam_dir: pathlib.Path,
    group_root: pathlib.Path,
    method: MethodSpec,
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from utils.exp_manager import ExperimentManager

    run_root = group_root / "runs_seed0" / method.label
    analysis_dir = run_root / "analysis"
    stdout_path = run_root / "stdout.txt"
    stderr_path = run_root / "stderr.txt"
    run_root.mkdir(parents=True, exist_ok=True)

    ns = build_args_namespace(
        args=args,
        cfg=cfg,
        lambda_pair=lambda_pair,
        method=method,
        hyperparam_dir=hyperparam_dir,
        run_root=run_root,
    )

    existing_run_dir = try_find_latest_run_dir(run_root / "saved_models", args.env)
    analysis_exists = (analysis_dir / "run_summary.csv").exists()
    existing_run_matches = existing_run_dir is not None and existing_run_matches_request(one_seed_module, existing_run_dir, ns=ns, method=method)

    if analysis_exists and existing_run_matches:
        summary = pd.read_csv(analysis_dir / "run_summary.csv").iloc[0].to_dict()
        training = one_seed_module.load_frame(analysis_dir / "training_episode_returns.csv", method.label)
        clean = one_seed_module.load_frame(analysis_dir / "clean_eval_returns.csv", method.label)
        adv = one_seed_module.load_frame(analysis_dir / "adversarial_eval_returns.csv", method.label)
    elif existing_run_matches:
        analysis_dir.mkdir(parents=True, exist_ok=True)
        if stdout_path.exists():
            shutil.copy2(stdout_path, analysis_dir / "stdout.txt")
        if stderr_path.exists():
            shutil.copy2(stderr_path, analysis_dir / "stderr.txt")
        one_seed_module.run_analysis(repo_dir, existing_run_dir, analysis_dir, method.label)
        summary = pd.read_csv(analysis_dir / "run_summary.csv").iloc[0].to_dict()
        training = one_seed_module.load_frame(analysis_dir / "training_episode_returns.csv", method.label)
        clean = one_seed_module.load_frame(analysis_dir / "clean_eval_returns.csv", method.label)
        adv = one_seed_module.load_frame(analysis_dir / "adversarial_eval_returns.csv", method.label)
    else:
        one_seed_module.register_closed_optimizers(repo_dir)
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
            try:
                with contextlib.redirect_stdout(stdout_handle), contextlib.redirect_stderr(stderr_handle):
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
                    if model is None:
                        raise RuntimeError("Hyperparameter optimization mode is unsupported for this benchmark.")
                    manager.learn(model)
                    manager.save_trained_model(model)
            except Exception:
                stderr_handle.write("\n" + traceback.format_exc())
                raise

        run_dir = find_latest_run_dir(run_root / "saved_models", args.env)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stdout_path, analysis_dir / "stdout.txt")
        shutil.copy2(stderr_path, analysis_dir / "stderr.txt")
        one_seed_module.run_analysis(repo_dir, run_dir, analysis_dir, method.label)
        summary = pd.read_csv(analysis_dir / "run_summary.csv").iloc[0].to_dict()
        training = one_seed_module.load_frame(analysis_dir / "training_episode_returns.csv", method.label)
        clean = one_seed_module.load_frame(analysis_dir / "clean_eval_returns.csv", method.label)
        adv = one_seed_module.load_frame(analysis_dir / "adversarial_eval_returns.csv", method.label)

    degradation = pd.DataFrame(
        {
            "timesteps": clean["timesteps"],
            "clean_mean_reward": clean["mean_reward"],
            "robust_mean_reward": adv["mean_reward"],
            "robust_degradation": clean["mean_reward"] - adv["mean_reward"],
            "method": method.label,
        }
    )
    return summary, training, clean, adv, degradation


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


def dominance_fraction(a: pd.Series, b: pd.Series, higher_better: bool = True) -> float:
    av = pd.to_numeric(a, errors="coerce").to_numpy()
    bv = pd.to_numeric(b, errors="coerce").to_numpy()
    mask = np.isfinite(av) & np.isfinite(bv)
    if mask.sum() == 0:
        return math.nan
    if higher_better:
        return float(np.mean(av[mask] > bv[mask] + 1e-9))
    return float(np.mean(av[mask] < bv[mask] - 1e-9))


def common_curve_metric(curves: Dict[str, pd.DataFrame], column: str) -> pd.DataFrame:
    common = None
    for method, frame in curves.items():
        sub = frame[["timesteps", column]].rename(columns={column: method})
        common = sub if common is None else common.merge(sub, on="timesteps", how="inner")
    return common if common is not None else pd.DataFrame()


def summarise_diag(diag_path: pathlib.Path) -> Dict[str, float]:
    if not diag_path.exists():
        return {}
    frame = pd.read_csv(diag_path)
    if frame.empty:
        return {}
    g_contrib_ratio = pd.to_numeric(frame.get("G_contribution_norm", pd.Series(np.nan, index=frame.index)), errors="coerce") / (
        pd.to_numeric(frame.get("update_norm_post_cap", pd.Series(np.nan, index=frame.index)), errors="coerce").abs() + EPS
    )
    qp_better = pd.to_numeric(frame.get("actual_drift", pd.Series(np.nan, index=frame.index)), errors="coerce") < (
        pd.to_numeric(frame.get("egm_actual_drift", pd.Series(np.nan, index=frame.index)), errors="coerce") - 1e-12
    )
    return {
        "fallback_to_egm_frac": float(pd.to_numeric(frame.get("fallback_to_egm", pd.Series(np.nan, index=frame.index)), errors="coerce").mean()),
        "gamma_active_frac": float(pd.to_numeric(frame.get("gamma_active_frac", pd.Series(np.nan, index=frame.index)), errors="coerce").mean()),
        "G_contribution_ratio": float(pd.to_numeric(g_contrib_ratio, errors="coerce").replace([np.inf, -np.inf], np.nan).mean()),
        "QP_better_than_EGM_drift_fraction": float(pd.to_numeric(qp_better.astype(float), errors="coerce").mean()),
        "mean_actual_QP_drift": float(pd.to_numeric(frame.get("actual_drift", pd.Series(np.nan, index=frame.index)), errors="coerce").mean()),
        "mean_actual_EGM_drift": float(pd.to_numeric(frame.get("egm_actual_drift", pd.Series(np.nan, index=frame.index)), errors="coerce").mean()),
        "cos_F_G": float(pd.to_numeric(frame.get("cos_F_G", pd.Series(np.nan, index=frame.index)), errors="coerce").mean()),
        "non_collinearity": float(pd.to_numeric(frame.get("non_collinearity", pd.Series(np.nan, index=frame.index)), errors="coerce").mean()),
        "field_norm": float(pd.to_numeric(frame.get("field_norm", pd.Series(np.nan, index=frame.index)), errors="coerce").mean()),
    }


def save_config_grid_plot(config_curves: Dict[str, pd.DataFrame], output_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    flat = axes.flatten()
    for ax, (title, frame) in zip(flat, config_curves.items()):
        for method, group in frame.groupby("method"):
            ax.plot(group["timesteps"], group["mean_reward"], label=method, linewidth=1.3)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in flat[len(config_curves):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_fallback_plot(rank_df: pd.DataFrame, output_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    qp_rows = rank_df.copy()
    x = np.arange(len(qp_rows))
    labels = [f"{row.optimizer_scope}\na={row.alpha}\nlr={row.shared_lr}\n{row.lambda_slug}" for row in qp_rows.itertuples()]
    axes[0].bar(x - 0.15, qp_rows["qp_avg_fallback_to_egm_frac"], width=0.3, label="avg fallback")
    axes[0].bar(x + 0.15, qp_rows["qp_avg_gamma_active_frac"], width=0.3, label="avg gamma active")
    axes[0].set_title("QP Fallback / Gamma Activity")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].bar(x - 0.15, qp_rows["qp_avg_actual_drift"], width=0.3, label="QP drift")
    axes[1].bar(x + 0.15, qp_rows["qp_avg_egm_drift"], width=0.3, label="EGM drift")
    axes[1].set_title("Actual Drift vs EGM Drift")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.N_mu = 5
    args.N_nu = 1
    repo_dir = pathlib.Path(args.repo_dir)
    output_root = pathlib.Path(args.output_root)
    plots_dir = output_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    one_seed = load_module("standard_rarl_one_seed_closed_qp_check_mod", repo_dir / "scripts" / "standard_rarl_one_seed_closed_qp_check.py")
    one_seed.register_closed_optimizers(repo_dir)
    hyperparam_dir = one_seed.ensure_hyperparams(repo_dir, output_root, args.env, args.fallback_env)

    all_method_rows: List[Dict[str, object]] = []
    ranked_rows: List[Dict[str, object]] = []
    top_curve = None
    top_title = ""
    top_score = -math.inf
    per_config_best_adv: Dict[str, pd.DataFrame] = {}

    for cfg in CONFIRMED_CONFIGS:
        cfg_root = output_root / cfg.slug
        baseline_root = cfg_root / "baseline"
        baseline_methods = build_method_specs(
            optimizer_scope=cfg.optimizer_scope,
            shared_lr=cfg.shared_lr,
            max_grad_norm=args.shared_max_grad_norm,
            vf_coef=args.shared_vf_coef,
            ppm_inner_steps=args.ppm_inner_steps,
            lambda_pair=LAMBDA_PAIRS[0],
            fd_eps=args.fd_eps,
            max_update_norm=args.max_update_norm,
            fallback_tolerance=args.fallback_tolerance,
            diagnostics_root=baseline_root / "runs_seed0",
        )[:3]
        baseline_results = {}
        for method in baseline_methods:
            summary, training, clean, adv, degradation = run_method(
                one_seed_module=one_seed,
                args=args,
                cfg=cfg,
                lambda_pair=LAMBDA_PAIRS[0],
                repo_dir=repo_dir,
                hyperparam_dir=hyperparam_dir,
                group_root=baseline_root,
                method=method,
            )
            baseline_results[method.label] = (summary, training, clean, adv, degradation)

        for lambda_pair in LAMBDA_PAIRS:
            lam_root = cfg_root / lambda_pair.slug
            diagnostics_root = lam_root / "runs_seed0"
            proposed_methods = build_method_specs(
                optimizer_scope=cfg.optimizer_scope,
                shared_lr=cfg.shared_lr,
                max_grad_norm=args.shared_max_grad_norm,
                vf_coef=args.shared_vf_coef,
                ppm_inner_steps=args.ppm_inner_steps,
                lambda_pair=lambda_pair,
                fd_eps=args.fd_eps,
                max_update_norm=args.max_update_norm,
                fallback_tolerance=args.fallback_tolerance,
                diagnostics_root=diagnostics_root,
            )[3:]

            method_bundle: Dict[str, Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
            method_bundle.update(baseline_results)
            for method in proposed_methods:
                summary, training, clean, adv, degradation = run_method(
                    one_seed_module=one_seed,
                    args=args,
                    cfg=cfg,
                    lambda_pair=lambda_pair,
                    repo_dir=repo_dir,
                    hyperparam_dir=hyperparam_dir,
                    group_root=lam_root,
                    method=method,
                )
                method_bundle[method.label] = (summary, training, clean, adv, degradation)

            adv_curves = {}
            clean_curves = {}
            degradation_curves = {}
            metric_rows = {}
            for method_label, (summary, training, clean, adv, degradation) in method_bundle.items():
                row = dict(summary)
                row.update(
                    {
                        "config_slug": cfg.slug,
                        "optimizer_scope": cfg.optimizer_scope,
                        "alpha": cfg.alpha,
                        "shared_lr": cfg.shared_lr,
                        "lambda_F": lambda_pair.lambda_F,
                        "lambda_R": lambda_pair.lambda_R,
                        "lambda_slug": lambda_pair.slug,
                        "method": method_label,
                        "curve_sane_flag": int(one_seed.is_curve_sane(row)),
                        "train_return_AUC": one_seed.auc_from_curve(training, "cumulative_timesteps", "episode_return"),
                        "clean_eval_return_AUC": one_seed.auc_from_curve(clean, "timesteps", "mean_reward"),
                        "current_adv_eval_return_AUC": one_seed.auc_from_curve(adv, "timesteps", "mean_reward"),
                        "current_adv_degradation_AUC": one_seed.auc_from_curve(degradation, "timesteps", "robust_degradation"),
                        "final_train_return": safe_float(training["episode_return"].dropna().iloc[-1]) if not training.empty and training["episode_return"].notna().any() else math.nan,
                        "final_clean_eval_return": safe_float(clean["mean_reward"].iloc[-1]) if not clean.empty else math.nan,
                        "final_current_adv_eval_return": safe_float(adv["mean_reward"].iloc[-1]) if not adv.empty else math.nan,
                        "final_current_adv_degradation": safe_float(degradation["robust_degradation"].iloc[-1]) if not degradation.empty else math.nan,
                    }
                )
                if method_label.startswith("proposed_"):
                    run_root = (lam_root / "runs_seed0" / method_label)
                    prot_diag = summarise_diag(run_root / f"protagonist_{method_label}_diagnostics.csv")
                    adv_diag = summarise_diag(run_root / f"adversary_{method_label}_diagnostics.csv")
                    for prefix, diag in (("protagonist", prot_diag), ("adversary", adv_diag)):
                        for key, value in diag.items():
                            row[f"{prefix}_{key}"] = value
                    if prot_diag or adv_diag:
                        row["avg_fallback_to_egm_frac"] = float(np.nanmean([prot_diag.get("fallback_to_egm_frac", np.nan), adv_diag.get("fallback_to_egm_frac", np.nan)]))
                        row["avg_gamma_active_frac"] = float(np.nanmean([prot_diag.get("gamma_active_frac", np.nan), adv_diag.get("gamma_active_frac", np.nan)]))
                        row["avg_G_contribution_ratio"] = float(np.nanmean([prot_diag.get("G_contribution_ratio", np.nan), adv_diag.get("G_contribution_ratio", np.nan)]))
                        row["avg_QP_better_than_EGM_drift_fraction"] = float(np.nanmean([prot_diag.get("QP_better_than_EGM_drift_fraction", np.nan), adv_diag.get("QP_better_than_EGM_drift_fraction", np.nan)]))
                        row["avg_actual_QP_drift"] = float(np.nanmean([prot_diag.get("mean_actual_QP_drift", np.nan), adv_diag.get("mean_actual_QP_drift", np.nan)]))
                        row["avg_actual_EGM_drift"] = float(np.nanmean([prot_diag.get("mean_actual_EGM_drift", np.nan), adv_diag.get("mean_actual_EGM_drift", np.nan)]))
                        row["avg_cos_F_G"] = float(np.nanmean([prot_diag.get("cos_F_G", np.nan), adv_diag.get("cos_F_G", np.nan)]))
                        row["avg_non_collinearity"] = float(np.nanmean([prot_diag.get("non_collinearity", np.nan), adv_diag.get("non_collinearity", np.nan)]))
                metric_rows[method_label] = row
                all_method_rows.append(row)
                adv_curves[method_label] = adv.assign(method=method_label)
                clean_curves[method_label] = clean.assign(method=method_label)
                degradation_curves[method_label] = degradation.assign(method=method_label)

            common_adv = common_curve_metric(adv_curves, "mean_reward")
            qp_row = metric_rows["proposed_qp_closed"]
            nog_row = metric_rows["proposed_nog_closed"]
            base_rows = {k: metric_rows[k] for k in ["sgd_gda", "egm", "ppm_inner5", "proposed_nog_closed"]}
            qp_auc = safe_float(qp_row["current_adv_eval_return_AUC"])
            compare_improvements = {}
            compare_dominance = {}
            for baseline_label, baseline_row in base_rows.items():
                base_auc = safe_float(baseline_row["current_adv_eval_return_AUC"])
                compare_improvements[baseline_label] = (qp_auc / (base_auc + EPS)) - 1.0 if finite(qp_auc) and finite(base_auc) else math.nan
                if not common_adv.empty:
                    compare_dominance[baseline_label] = dominance_fraction(common_adv["proposed_qp_closed"], common_adv[baseline_label], True)
                else:
                    compare_dominance[baseline_label] = math.nan

            baseline_clean_best = max(safe_float(base_rows[label]["clean_eval_return_AUC"]) for label in ["sgd_gda", "egm", "ppm_inner5"])
            qp_clean_auc = safe_float(qp_row["clean_eval_return_AUC"])
            qp_final_adv = safe_float(qp_row["final_current_adv_eval_return"])
            best_final_adv = max(safe_float(metric_rows[label]["final_current_adv_eval_return"]) for label in metric_rows.keys())
            nonfinal_best_baseline = None
            qp_not_final_only = 0
            if not common_adv.empty:
                baseline_cols = ["sgd_gda", "egm", "ppm_inner5", "proposed_nog_closed"]
                baseline_best = common_adv[baseline_cols].max(axis=1)
                qp_beats_best = pd.to_numeric(common_adv["proposed_qp_closed"], errors="coerce") > pd.to_numeric(baseline_best, errors="coerce") + 1e-9
                qp_not_final_only = int(bool(len(qp_beats_best) >= 2 and qp_beats_best.iloc[:-1].any()))
                nonfinal_best_baseline = qp_beats_best

            weak_positive = bool(
                all(finite(compare_improvements[label]) and compare_improvements[label] >= 0.05 for label in base_rows)
                and all(finite(compare_dominance[label]) and compare_dominance[label] >= 0.70 for label in base_rows)
                and qp_final_adv >= best_final_adv - 1e-9
                and finite(qp_clean_auc)
                and finite(baseline_clean_best)
                and qp_clean_auc >= 0.75 * baseline_clean_best
                and finite(qp_row.get("avg_fallback_to_egm_frac"))
                and float(qp_row.get("avg_fallback_to_egm_frac")) < 0.35
                and qp_not_final_only == 1
                and int(qp_row.get("curve_sane_flag", 0)) == 1
            )
            strong_positive = bool(
                all(finite(compare_improvements[label]) and compare_improvements[label] >= 0.10 for label in base_rows)
                and all(finite(compare_dominance[label]) and compare_dominance[label] >= 0.80 for label in base_rows)
                and qp_final_adv >= best_final_adv - 1e-9
                and finite(qp_clean_auc)
                and finite(baseline_clean_best)
                and qp_clean_auc >= 0.85 * baseline_clean_best
                and finite(qp_row.get("avg_fallback_to_egm_frac"))
                and float(qp_row.get("avg_fallback_to_egm_frac")) < 0.25
                and finite(qp_row.get("avg_QP_better_than_EGM_drift_fraction"))
                and float(qp_row.get("avg_QP_better_than_EGM_drift_fraction")) > 0.65
                and qp_not_final_only == 1
                and int(qp_row.get("curve_sane_flag", 0)) == 1
            )

            ranked_row = {
                "config_slug": cfg.slug,
                "optimizer_scope": cfg.optimizer_scope,
                "alpha": cfg.alpha,
                "shared_lr": cfg.shared_lr,
                "lambda_F": lambda_pair.lambda_F,
                "lambda_R": lambda_pair.lambda_R,
                "lambda_slug": lambda_pair.slug,
                "qp_vs_sgd_auc_improvement": compare_improvements["sgd_gda"],
                "qp_vs_egm_auc_improvement": compare_improvements["egm"],
                "qp_vs_ppm_auc_improvement": compare_improvements["ppm_inner5"],
                "qp_vs_nog_auc_improvement": compare_improvements["proposed_nog_closed"],
                "qp_dom_over_sgd": compare_dominance["sgd_gda"],
                "qp_dom_over_egm": compare_dominance["egm"],
                "qp_dom_over_ppm": compare_dominance["ppm_inner5"],
                "qp_dom_over_nog": compare_dominance["proposed_nog_closed"],
                "qp_final_current_adv_eval_return": qp_final_adv,
                "best_final_current_adv_eval_return": best_final_adv,
                "qp_clean_eval_return_AUC": qp_clean_auc,
                "best_baseline_clean_eval_return_AUC": baseline_clean_best,
                "qp_avg_fallback_to_egm_frac": qp_row.get("avg_fallback_to_egm_frac", math.nan),
                "qp_avg_gamma_active_frac": qp_row.get("avg_gamma_active_frac", math.nan),
                "qp_avg_G_contribution_ratio": qp_row.get("avg_G_contribution_ratio", math.nan),
                "qp_avg_QP_better_than_EGM_drift_fraction": qp_row.get("avg_QP_better_than_EGM_drift_fraction", math.nan),
                "qp_avg_actual_drift": qp_row.get("avg_actual_QP_drift", math.nan),
                "qp_avg_egm_drift": qp_row.get("avg_actual_EGM_drift", math.nan),
                "qp_avg_cos_F_G": qp_row.get("avg_cos_F_G", math.nan),
                "qp_avg_non_collinearity": qp_row.get("avg_non_collinearity", math.nan),
                "qp_not_final_only_flag": qp_not_final_only,
                "all_curve_sane_flag": int(all(int(metric_rows[m]["curve_sane_flag"]) == 1 for m in metric_rows)),
                "qp_weak_positive_flag": int(weak_positive),
                "qp_strong_positive_flag": int(strong_positive),
            }
            ranked_rows.append(ranked_row)

            score = (2.0 if strong_positive else 1.0 if weak_positive else 0.0) + safe_float(compare_improvements["ppm_inner5"], -1.0) + 0.1 * safe_float(compare_dominance["ppm_inner5"], 0.0)
            if score > top_score:
                top_score = score
                top_curve = {
                    "training": pd.concat([method_bundle[m][1].assign(method=m) for m in method_bundle], ignore_index=True),
                    "clean": pd.concat([method_bundle[m][2].assign(method=m) for m in method_bundle], ignore_index=True),
                    "adv": pd.concat([method_bundle[m][3].assign(method=m) for m in method_bundle], ignore_index=True),
                    "degradation": pd.concat([method_bundle[m][4].assign(method=m) for m in method_bundle], ignore_index=True),
                }
                top_title = f"{cfg.slug} / {lambda_pair.slug}"

            best_for_cfg = per_config_best_adv.get(cfg.slug)
            candidate_adv = pd.concat([method_bundle[m][3].assign(method=m) for m in method_bundle], ignore_index=True)
            if best_for_cfg is None or score > best_for_cfg.attrs.get("score", -math.inf):
                candidate_adv.attrs["score"] = score
                per_config_best_adv[cfg.slug] = candidate_adv

    all_df = pd.DataFrame(all_method_rows).sort_values(["config_slug", "lambda_F", "lambda_R", "method"]).reset_index(drop=True)
    ranked_df = pd.DataFrame(ranked_rows).sort_values(
        ["qp_strong_positive_flag", "qp_weak_positive_flag", "qp_vs_ppm_auc_improvement", "qp_dom_over_ppm"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    all_df.to_csv(output_root / "qp_validation_all_configs.csv", index=False)
    ranked_df.to_csv(output_root / "qp_validation_ranked.csv", index=False)

    strong_any = bool(ranked_df["qp_strong_positive_flag"].eq(1).any()) if not ranked_df.empty else False
    weak_any = bool(ranked_df["qp_weak_positive_flag"].eq(1).any()) if not ranked_df.empty else False
    all_sane_any = bool(ranked_df["all_curve_sane_flag"].eq(1).all()) if not ranked_df.empty else False
    if strong_any:
        final_decision = "QP_STRONG_POSITIVE"
    elif weak_any:
        final_decision = "QP_WEAK_POSITIVE"
    elif not all_sane_any:
        final_decision = "QP_INCONCLUSIVE"
    else:
        final_decision = "BASELINE_CONFIRMED_BUT_QP_FAILS"

    top_lines = [
        "# Stage 3B QP-Positive Top Configs",
        "",
        "Ranking is by strong/weak positive flags first, then QP vs PPM improvement and dominance.",
        "",
    ]
    for _, row in ranked_df.head(12).iterrows():
        top_lines.append(
            f"- cfg=`{row['config_slug']}`, lambda=`{row['lambda_slug']}`:"
            f" weak=`{bool(row['qp_weak_positive_flag'])}`, strong=`{bool(row['qp_strong_positive_flag'])}`,"
            f" qp_vs_ppm_auc_improvement=`{row['qp_vs_ppm_auc_improvement']:.3f}`,"
            f" qp_dom_over_ppm=`{row['qp_dom_over_ppm']:.3f}`,"
            f" fallback=`{row['qp_avg_fallback_to_egm_frac']:.3f}`,"
            f" qp_better_than_egm_drift_frac=`{row['qp_avg_QP_better_than_EGM_drift_fraction']:.3f}`"
        )
    (output_root / "qp_positive_top_configs.md").write_text("\n".join(top_lines), encoding="utf-8")

    report_lines = [
        "# Stage 3B QP Validation After Alpha Fix",
        "",
        "- environment: `HalfCheetah-v5`",
        "- seed: `0`",
        "- alternating schedule: `N_mu=5`, `N_nu=1`",
        "- disturbance wrapper: `a_env = clip(u + alpha * w)`",
        "- reward shaping added: `False`",
        "- confirmed Stage 3A configs only: `True`",
        "- methods: `sgd_gda`, `egm`, `ppm_inner5`, `proposed_nog_closed`, `proposed_qp_closed`",
        "- baselines reused only within this Stage 3B root and only across lambda pairs for the same confirmed config",
        "",
        "## Final Decision",
        "",
        f"`{final_decision}`",
        "",
        "## Top Rows",
        "",
    ]
    for _, row in ranked_df.head(8).iterrows():
        report_lines.append(
            f"- cfg=`{row['config_slug']}`, lambda=`{row['lambda_slug']}`:"
            f" weak=`{bool(row['qp_weak_positive_flag'])}`, strong=`{bool(row['qp_strong_positive_flag'])}`,"
            f" qp_vs_ppm_auc_improvement=`{row['qp_vs_ppm_auc_improvement']:.3f}`,"
            f" qp_dom_over_ppm=`{row['qp_dom_over_ppm']:.3f}`,"
            f" clean_ratio=`{row['qp_clean_eval_return_AUC'] / (row['best_baseline_clean_eval_return_AUC'] + EPS):.3f}`,"
            f" fallback=`{row['qp_avg_fallback_to_egm_frac']:.3f}`"
        )
    (output_root / "qp_validation_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    if top_curve is not None:
        one_seed.save_big_figure(top_curve["training"], top_curve["clean"], top_curve["adv"], top_curve["degradation"], plots_dir / "qp_validation_top_config_all_curves.png")

    if per_config_best_adv:
        save_config_grid_plot(per_config_best_adv, plots_dir / "qp_validation_each_confirmed_config.png")

    if not ranked_df.empty:
        save_fallback_plot(ranked_df, plots_dir / "qp_validation_fallback_and_drift.png")


if __name__ == "__main__":
    main()
