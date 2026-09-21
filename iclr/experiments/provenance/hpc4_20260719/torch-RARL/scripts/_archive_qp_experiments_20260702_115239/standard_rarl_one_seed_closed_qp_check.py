from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class MethodSpec:
    label: str
    optimizer: str
    protagonist_optimizer_kwargs: Dict[str, object]
    adversary_optimizer_kwargs: Dict[str, object]
    lr: float
    max_grad_norm: float
    vf_coef: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("One-seed standard RARL closed-QP check on HalfCheetah")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v5")
    parser.add_argument("--fallback-env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=2500)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--shared-lr", type=float, default=1e-3)
    parser.add_argument("--shared-max-grad-norm", type=float, default=10.0)
    parser.add_argument("--shared-vf-coef", type=float, default=0.5)
    parser.add_argument("--ppm-inner-steps", type=int, default=2)
    parser.add_argument("--optimizer-scope", type=str, default="full_policy", choices=["full_policy", "actor_game", "actor_logstd_only"])
    parser.add_argument("--adv-impact", type=str, default="control", choices=["control", "force"])
    parser.add_argument("--lambda-F", type=float, default=0.01)
    parser.add_argument("--lambda-R", type=float, default=1.0)
    parser.add_argument("--beta-max-mult", type=float, default=3.0)
    parser.add_argument("--gamma-max-mult", type=float, default=3.0)
    parser.add_argument("--fd-eps", type=float, default=1e-3)
    parser.add_argument("--max-update-norm", type=float, default=0.005)
    parser.add_argument("--fallback-tolerance", type=float, default=0.0)
    return parser.parse_args()


def register_closed_optimizers(repo_dir: pathlib.Path) -> None:
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from models.optimizers import OPTIMIZER_REGISTRY
    from models.proposed_qp_closedlyap import ProposedNoGClosedLyapOptimizer, ProposedQPClosedLyapOptimizer

    OPTIMIZER_REGISTRY["proposed_nog_closedlyap"] = ProposedNoGClosedLyapOptimizer
    OPTIMIZER_REGISTRY["proposed_qp_closedlyap"] = ProposedQPClosedLyapOptimizer


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

    metadata = {
        "requested_env": requested_env,
        "fallback_env": fallback_env,
        "env_used": requested_env,
        "mapping_note": mapping_note,
        "source_yaml": str(source_path),
        "target_yaml": str(target_path),
    }
    (temp_dir / "hyperparam_mapping.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return temp_dir


def kwargs_tokens(kwargs: Dict[str, object]) -> List[str]:
    tokens: List[str] = []
    for key, value in sorted(kwargs.items()):
        rendered = repr(value) if isinstance(value, str) else value
        tokens.append(f"{key}:{rendered}")
    return tokens


def build_args_namespace(
    *,
    method: MethodSpec,
    args: argparse.Namespace,
    repo_dir: pathlib.Path,
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
        optimizer_scope=args.optimizer_scope,
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
        N_mu=-1,
        N_nu=-1,
        adv_impact=args.adv_impact,
        adv_delay=-1,
        adv_fraction=1.0,
        adv_fraction_override=True,
        requested_alpha=1.0,
        adv_index_list=["torso"],
        adv_force_dim=2,
    )


def build_method_specs(args: argparse.Namespace, output_root: pathlib.Path) -> List[MethodSpec]:
    beta_probe = args.shared_lr
    gamma_probe = max(args.shared_lr * args.shared_lr, 1e-6)
    beta_max = args.beta_max_mult * args.shared_lr
    gamma_max = args.gamma_max_mult * gamma_probe

    def closed_kwargs(method_label: str, role: str, allow_fallback: bool) -> Dict[str, object]:
        return {
            "optimizer_scope": args.optimizer_scope,
            "lambda_F": args.lambda_F,
            "lambda_R": args.lambda_R,
            "actor_weight": 1.0,
            "logstd_weight": 1.0,
            "critic_weight": 1.0,
            "fd_eps": args.fd_eps,
            "beta_probe": beta_probe,
            "gamma_probe": gamma_probe,
            "ridge": 1e-8,
            "beta_max": beta_max,
            "gamma_max": gamma_max,
            "max_update_norm": args.max_update_norm,
            "qp_eps": 1e-8,
            "allow_fallback_to_egm": allow_fallback,
            "fallback_tolerance": args.fallback_tolerance,
            # Keep a performance-like term in V without invoking the very slow short-horizon
            # rollout merit evaluator on every PPO minibatch step.
            "cost_mode": "unclipped_actor_surrogate_cost",
            "short_return_horizon": 16,
            "short_return_episodes": 1,
            "short_return_seed_offset": 0,
            "role": role,
            "diagnostics_csv_path": str(output_root / "runs_seed0" / method_label / f"{role}_{method_label}_diagnostics.csv"),
        }

    return [
        MethodSpec(
            label="sgd_gda",
            optimizer="sgd",
            protagonist_optimizer_kwargs={},
            adversary_optimizer_kwargs={},
            lr=args.shared_lr,
            max_grad_norm=args.shared_max_grad_norm,
            vf_coef=args.shared_vf_coef,
        ),
        MethodSpec(
            label="egm",
            optimizer="egm",
            protagonist_optimizer_kwargs={},
            adversary_optimizer_kwargs={},
            lr=args.shared_lr,
            max_grad_norm=args.shared_max_grad_norm,
            vf_coef=args.shared_vf_coef,
        ),
        MethodSpec(
            label="ppm",
            optimizer="ppm",
            protagonist_optimizer_kwargs={"inner_steps": args.ppm_inner_steps},
            adversary_optimizer_kwargs={"inner_steps": args.ppm_inner_steps},
            lr=args.shared_lr,
            max_grad_norm=args.shared_max_grad_norm,
            vf_coef=args.shared_vf_coef,
        ),
        MethodSpec(
            label="proposed_nog_closed",
            optimizer="proposed_nog_closedlyap",
            protagonist_optimizer_kwargs=closed_kwargs("proposed_nog_closed", "protagonist", False),
            adversary_optimizer_kwargs=closed_kwargs("proposed_nog_closed", "adversary", False),
            lr=args.shared_lr,
            max_grad_norm=args.shared_max_grad_norm,
            vf_coef=args.shared_vf_coef,
        ),
        MethodSpec(
            label="proposed_qp_closed",
            optimizer="proposed_qp_closedlyap",
            protagonist_optimizer_kwargs=closed_kwargs("proposed_qp_closed", "protagonist", True),
            adversary_optimizer_kwargs=closed_kwargs("proposed_qp_closed", "adversary", True),
            lr=args.shared_lr,
            max_grad_norm=args.shared_max_grad_norm,
            vf_coef=args.shared_vf_coef,
        ),
    ]


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


def load_run_args(run_dir: pathlib.Path) -> Dict[str, object]:
    args_path = run_dir / "args.yml"
    if not args_path.exists():
        return {}
    text = args_path.read_text(encoding="utf-8")
    for loader in (yaml.safe_load, yaml.full_load, yaml.unsafe_load):
        try:
            data = loader(text) or {}
            if isinstance(data, dict):
                return dict(data)
        except Exception:
            continue
    return {}


def existing_run_matches_request(run_dir: pathlib.Path, args: argparse.Namespace, method: MethodSpec) -> bool:
    run_args = load_run_args(run_dir)
    if not run_args:
        return False
    expected = {
        "seed": args.seed,
        "env": args.env,
        "n_timesteps": args.iterations,
        "protagonist_optimizer": method.optimizer,
        "adversary_optimizer": method.optimizer,
        "optimizer_scope": args.optimizer_scope,
    }
    for key, value in expected.items():
        if run_args.get(key) != value:
            return False
    if float(run_args.get("protagonist_lr", float("nan"))) != float(method.lr):
        return False
    if float(run_args.get("adversary_lr", float("nan"))) != float(method.lr):
        return False
    return True


def load_frame(path: pathlib.Path, method: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["method"] = method
    return frame


def auc_from_curve(frame: pd.DataFrame, x_col: str, y_col: str) -> float:
    if len(frame) < 2:
        return float("nan")
    return float(np.trapezoid(frame[y_col].to_numpy(), frame[x_col].to_numpy()))


def is_curve_sane(summary_row: Dict[str, object]) -> bool:
    if int(summary_row.get("crash_flag", 1)) != 0 or int(summary_row.get("nan_flag", 1)) != 0:
        return False
    for key in ["final_training_return", "best_training_return", "final_clean_return", "final_adversarial_return"]:
        value = summary_row.get(key, np.nan)
        if not np.isfinite(value):
            return False
    return True


def run_analysis(repo_dir: pathlib.Path, run_dir: pathlib.Path, analysis_dir: pathlib.Path, method_label: str) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = analysis_dir / "stdout.txt"
    stderr_path = analysis_dir / "stderr.txt"
    if not stdout_path.exists():
        stdout_path.write_text(
            "direct ExperimentManager run; stdout was not captured by subprocess for this analysis.\n",
            encoding="utf-8",
        )
    if not stderr_path.exists():
        stderr_path.write_text("", encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(repo_dir / "scripts" / "analyze_rarl_run.py"),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(analysis_dir),
            "--method",
            method_label,
        ],
        cwd=str(repo_dir),
        check=True,
        capture_output=True,
        text=True,
    )


def run_method(
    *,
    args: argparse.Namespace,
    repo_dir: pathlib.Path,
    hyperparam_dir: pathlib.Path,
    output_root: pathlib.Path,
    method: MethodSpec,
):
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from utils.exp_manager import ExperimentManager

    run_root = output_root / "runs_seed0" / method.label
    analysis_dir = run_root / "analysis"
    stdout_path = run_root / "stdout.txt"
    stderr_path = run_root / "stderr.txt"
    run_root.mkdir(parents=True, exist_ok=True)

    existing_run_dir = try_find_latest_run_dir(run_root / "saved_models", args.env)
    analysis_exists = (analysis_dir / "run_summary.csv").exists()
    existing_run_matches = existing_run_dir is not None and existing_run_matches_request(existing_run_dir, args, method)

    if analysis_exists and existing_run_matches:
        summary = pd.read_csv(analysis_dir / "run_summary.csv").iloc[0].to_dict()
        training = load_frame(analysis_dir / "training_episode_returns.csv", method.label)
        clean = load_frame(analysis_dir / "clean_eval_returns.csv", method.label)
        adv = load_frame(analysis_dir / "adversarial_eval_returns.csv", method.label)
    elif existing_run_matches:
        analysis_dir.mkdir(parents=True, exist_ok=True)
        if stdout_path.exists():
            shutil.copy2(stdout_path, analysis_dir / "stdout.txt")
        if stderr_path.exists():
            shutil.copy2(stderr_path, analysis_dir / "stderr.txt")
        run_analysis(repo_dir, existing_run_dir, analysis_dir, method.label)
        summary = pd.read_csv(analysis_dir / "run_summary.csv").iloc[0].to_dict()
        training = load_frame(analysis_dir / "training_episode_returns.csv", method.label)
        clean = load_frame(analysis_dir / "clean_eval_returns.csv", method.label)
        adv = load_frame(analysis_dir / "adversarial_eval_returns.csv", method.label)
    else:
        ns = build_args_namespace(
            method=method,
            args=args,
            repo_dir=repo_dir,
            hyperparam_dir=hyperparam_dir,
            run_root=run_root,
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
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
            try:
                with contextlib.redirect_stdout(stdout_handle), contextlib.redirect_stderr(stderr_handle):
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
        run_analysis(repo_dir, run_dir, analysis_dir, method.label)
        summary = pd.read_csv(analysis_dir / "run_summary.csv").iloc[0].to_dict()
        training = load_frame(analysis_dir / "training_episode_returns.csv", method.label)
        clean = load_frame(analysis_dir / "clean_eval_returns.csv", method.label)
        adv = load_frame(analysis_dir / "adversarial_eval_returns.csv", method.label)

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


def save_line_plot(frame: pd.DataFrame, x_col: str, y_col: str, ylabel: str, title: str, output_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in frame.groupby("method"):
        ax.plot(group[x_col], group[y_col], label=method, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_big_figure(training_df: pd.DataFrame, clean_df: pd.DataFrame, adv_df: pd.DataFrame, degradation_df: pd.DataFrame, output_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for method, group in training_df.groupby("method"):
        axes[0, 0].plot(group["cumulative_timesteps"], group["episode_return"], label=method)
    axes[0, 0].set_title("Train Return")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    for method, group in clean_df.groupby("method"):
        axes[0, 1].plot(group["timesteps"], group["mean_reward"], label=method)
    axes[0, 1].set_title("Clean Eval Return")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    for method, group in adv_df.groupby("method"):
        axes[1, 0].plot(group["timesteps"], group["mean_reward"], label=method)
    axes[1, 0].set_title("Robust Eval Return")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    for method, group in degradation_df.groupby("method"):
        axes[1, 1].plot(group["timesteps"], group["robust_degradation"], label=method)
    axes[1, 1].set_title("Robust Degradation")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_geometry_plots(output_root: pathlib.Path, method_specs: List[MethodSpec]) -> None:
    plot_specs = [
        ("field_norm", "Field Norm", "geometry_field_norm.png"),
        ("cos_F_G", "cos(F, G)", "geometry_cos_F_G.png"),
        ("non_collinearity", "Non-collinearity", "geometry_non_collinearity.png"),
        ("actual_drift", "Actual Lyapunov Drift", "geometry_actual_drift.png"),
    ]
    frames: List[pd.DataFrame] = []
    for method in method_specs:
        if "proposed_" not in method.label:
            continue
        for role in ("protagonist", "adversary"):
            diag_path = output_root / "runs_seed0" / method.label / f"{role}_{method.label}_diagnostics.csv"
            if diag_path.exists():
                frame = pd.read_csv(diag_path)
                frame["method"] = method.label
                frame["role"] = role
                if "step" not in frame.columns:
                    frame["step"] = np.arange(len(frame))
                frames.append(frame)
    if not frames:
        return

    geometry_df = pd.concat(frames, ignore_index=True)
    geometry_df.to_csv(output_root / "geometry_curves.csv", index=False)

    for column, ylabel, filename in plot_specs:
        if column not in geometry_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        for (method, role), group in geometry_df.groupby(["method", "role"]):
            series = pd.to_numeric(group[column], errors="coerce")
            if not np.isfinite(series.to_numpy(dtype=float)).any():
                continue
            ax.plot(group["step"], series, label=f"{method}:{role}", linewidth=1.5)
        ax.set_title(ylabel)
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_root / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    available_specs = [spec for spec in plot_specs if spec[0] in geometry_df.columns]
    if not available_specs:
        return
    rows = 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    flat_axes = axes.flatten()
    for ax, (column, ylabel, _) in zip(flat_axes, available_specs):
        for (method, role), group in geometry_df.groupby(["method", "role"]):
            series = pd.to_numeric(group[column], errors="coerce")
            if not np.isfinite(series.to_numpy(dtype=float)).any():
                continue
            ax.plot(group["step"], series, label=f"{method}:{role}", linewidth=1.3)
        ax.set_title(ylabel)
        ax.set_xlabel("Optimizer Step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in flat_axes[len(available_specs):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_root / "geometry_all_plots_big.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_dir = pathlib.Path(args.repo_dir)
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    register_closed_optimizers(repo_dir)
    hyperparam_dir = ensure_hyperparams(repo_dir, output_root, args.env, args.fallback_env)
    method_specs = build_method_specs(args, output_root)

    summary_rows: List[Dict[str, object]] = []
    training_frames: List[pd.DataFrame] = []
    clean_frames: List[pd.DataFrame] = []
    adv_frames: List[pd.DataFrame] = []
    degradation_frames: List[pd.DataFrame] = []
    commands: Dict[str, Dict[str, object]] = {}

    for method in method_specs:
        summary, training, clean, adv, degradation = run_method(
            args=args,
            repo_dir=repo_dir,
            hyperparam_dir=hyperparam_dir,
            output_root=output_root,
            method=method,
        )
        row = dict(summary)
        row["method"] = method.label
        row["mapped_optimizer"] = method.optimizer
        row["method_lr"] = method.lr
        row["method_max_grad_norm"] = method.max_grad_norm
        row["method_vf_coef"] = method.vf_coef
        row["curve_sane_flag"] = int(is_curve_sane(row))
        row["auc_train"] = auc_from_curve(training, "cumulative_timesteps", "episode_return")
        row["auc_eval_clean"] = auc_from_curve(clean, "timesteps", "mean_reward")
        row["auc_eval_robust"] = auc_from_curve(adv, "timesteps", "mean_reward")
        row["auc_robust_degradation"] = auc_from_curve(degradation, "timesteps", "robust_degradation")
        if "proposed_" in method.label:
            run_root = output_root / "runs_seed0" / method.label
            prot_diag_path = run_root / f"protagonist_{method.label}_diagnostics.csv"
            adv_diag_path = run_root / f"adversary_{method.label}_diagnostics.csv"
            if prot_diag_path.exists():
                prot_diag = pd.read_csv(prot_diag_path)
                row["protagonist_fallback_to_egm_frac"] = float(prot_diag["fallback_to_egm"].mean()) if "fallback_to_egm" in prot_diag else float("nan")
                row["protagonist_actual_drift_mean"] = float(prot_diag["actual_drift"].mean()) if "actual_drift" in prot_diag else float("nan")
            if adv_diag_path.exists():
                adv_diag = pd.read_csv(adv_diag_path)
                row["adversary_fallback_to_egm_frac"] = float(adv_diag["fallback_to_egm"].mean()) if "fallback_to_egm" in adv_diag else float("nan")
                row["adversary_actual_drift_mean"] = float(adv_diag["actual_drift"].mean()) if "actual_drift" in adv_diag else float("nan")
        summary_rows.append(row)
        training_frames.append(training)
        clean_frames.append(clean)
        adv_frames.append(adv)
        degradation_frames.append(degradation)
        commands[method.label] = {
            "optimizer": method.optimizer,
            "lr": method.lr,
            "max_grad_norm": method.max_grad_norm,
            "vf_coef": method.vf_coef,
            "protagonist_optimizer_kwargs": method.protagonist_optimizer_kwargs,
            "adversary_optimizer_kwargs": method.adversary_optimizer_kwargs,
        }

    summary_df = pd.DataFrame(summary_rows).sort_values("method").reset_index(drop=True)
    training_df = pd.concat(training_frames, ignore_index=True)
    clean_df = pd.concat(clean_frames, ignore_index=True)
    adv_df = pd.concat(adv_frames, ignore_index=True)
    degradation_df = pd.concat(degradation_frames, ignore_index=True)

    summary_df.to_csv(output_root / "summary.csv", index=False)
    training_df.to_csv(output_root / "train_curves.csv", index=False)
    clean_df.to_csv(output_root / "eval_clean_curves.csv", index=False)
    adv_df.to_csv(output_root / "eval_robust_curves.csv", index=False)
    degradation_df.to_csv(output_root / "robust_degradation_curves.csv", index=False)
    (output_root / "commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")

    save_line_plot(training_df, "cumulative_timesteps", "episode_return", "Episode Return", "Training Return", output_root / "train_return.png")
    save_line_plot(clean_df, "timesteps", "mean_reward", "Mean Reward", "Clean Eval Return", output_root / "eval_clean_return.png")
    save_line_plot(adv_df, "timesteps", "mean_reward", "Mean Reward", "Robust Eval Return", output_root / "eval_robust_return.png")
    save_line_plot(degradation_df, "timesteps", "robust_degradation", "Clean - Robust", "Robust Degradation", output_root / "robust_degradation.png")
    save_big_figure(training_df, clean_df, adv_df, degradation_df, output_root / "all_plots_big.png")
    save_geometry_plots(output_root, method_specs)

    robust_final = summary_df[["method", "final_adversarial_return"]].sort_values("final_adversarial_return", ascending=False)
    sgd_robust = float(summary_df.loc[summary_df["method"] == "sgd_gda", "final_adversarial_return"].iloc[0])
    egm_robust = float(summary_df.loc[summary_df["method"] == "egm", "final_adversarial_return"].iloc[0])
    ppm_robust = float(summary_df.loc[summary_df["method"] == "ppm", "final_adversarial_return"].iloc[0])
    nog_robust = float(summary_df.loc[summary_df["method"] == "proposed_nog_closed", "final_adversarial_return"].iloc[0])
    qp_robust = float(summary_df.loc[summary_df["method"] == "proposed_qp_closed", "final_adversarial_return"].iloc[0])

    all_sane = bool(summary_df["curve_sane_flag"].all())
    egm_beats_sgd = bool(egm_robust > sgd_robust)
    ppm_beats_sgd = bool(ppm_robust > sgd_robust)
    qp_beats_all = bool(qp_robust > max(sgd_robust, egm_robust, ppm_robust, nog_robust))

    if not all_sane:
        decision = "BASELINES_NEED_TUNING"
    elif qp_beats_all:
        decision = "PROMISING_QP_STANDARD_RARL_ONE_SEED"
    else:
        decision = "QP_NOT_PROMISING_ONE_SEED"

    lines = [
        "# One-Seed Standard RARL Closed-QP Check",
        "",
        "This is a one-seed screening run only. Do not over-claim from it.",
        "",
        f"- Environment: `{args.env}`",
        f"- Seed: `{args.seed}`",
        f"- Adversary impact: `{args.adv_impact}`",
        f"- RARL regime: `alternating`, using env hyperparameters copied from `{args.fallback_env}` if needed",
        f"- Optimizer scope for all methods: `{args.optimizer_scope}`",
        f"- Baselines: `sgd`, `egm`, `ppm` on the native PPO loss with shared `lr/max_grad_norm/vf_coef`",
        f"- Proposed mapping: `proposed_nog_closedlyap`, `proposed_qp_closedlyap`",
        f"- Closed Lyapunov design: `V = lambda_F * field_term + lambda_R * return_term`, with `lambda_F={args.lambda_F}` and `lambda_R={args.lambda_R}`",
        "- Return term priority: use the PPO frozen-batch unclipped actor surrogate as the performance-like term, instead of the much slower short-horizon rollout merit.",
        "",
        "## Questions",
        "",
        f"1. Did all baselines run without crashing? `{bool((summary_df['crash_flag'] == 0).all() and (summary_df['nan_flag'] == 0).all())}`",
        f"2. Are baseline curves sane / convergent / non-degenerate? `{all_sane}`",
        f"3. Does EGM or PPM outperform SGD/GDA? `EGM>{egm_beats_sgd}`, `PPM>{ppm_beats_sgd}`",
        f"4. Does closed QP outperform noG / EGM / PPM / SGD on robust eval return? `{qp_beats_all}`",
        f"5. Is the result promising enough to justify multi-seed? `{decision == 'PROMISING_QP_STANDARD_RARL_ONE_SEED'}`",
        f"6. Any obvious instability or config mismatch? `Review summary.csv plus proposed diagnostics for fallback/drift caveats.`",
        "",
        "## Final robust ranking",
        "",
    ]
    for _, robust_row in robust_final.iterrows():
        lines.append(f"- `{robust_row['method']}` final robust eval return = `{robust_row['final_adversarial_return']:.6f}`")
    lines.extend(
        [
            "",
            "## Final decision",
            "",
            f"`{decision}`",
        ]
    )
    (output_root / "one_seed_decision.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
