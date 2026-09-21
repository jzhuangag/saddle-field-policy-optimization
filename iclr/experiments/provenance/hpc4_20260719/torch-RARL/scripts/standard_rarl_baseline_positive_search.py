from __future__ import annotations

import argparse
import contextlib
import json
import math
import pathlib
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


EPS = 1e-12


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
    parser = argparse.ArgumentParser("Stage A baseline-positive search for standard alternating RARL")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument(
        "--output-root",
        type=str,
        default=r"C:\Users\jhuangag\work\rarl\original\results\standard_rarl_baseline_positive_search",
    )
    parser.add_argument("--env", type=str, default="HalfCheetah-v5")
    parser.add_argument("--fallback-env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--eval-freq", type=int, default=10240)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--shared-max-grad-norm", type=float, default=10.0)
    parser.add_argument("--shared-vf-coef", type=float, default=0.5)
    parser.add_argument("--optimizer-scopes", type=str, default="full_policy,actor_logstd_only")
    parser.add_argument("--alphas", type=str, default="0.05,0.1,0.2,0.3,0.5")
    parser.add_argument("--shared-lrs", type=str, default="3e-4,1e-3,3e-3")
    parser.add_argument("--ppm-inner-steps", type=int, default=5)
    parser.add_argument("--n-mu", type=int, default=5)
    parser.add_argument("--n-nu", type=int, default=1)
    return parser.parse_args()


def parse_csv_list(raw: str, cast) -> List:
    return [cast(token.strip()) for token in raw.split(",") if token.strip()]


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


def curve_hash(values: Sequence[float], decimals: int = 6) -> str:
    rounded = []
    for value in values:
        try:
            rounded.append(round(float(value), decimals))
        except Exception:
            rounded.append("nan")
    payload = json.dumps({"values": rounded}, sort_keys=True, separators=(",", ":"))
    return __import__("hashlib").sha256(payload.encode("utf-8")).hexdigest()[:16]


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


def compact_scope(scope: str) -> str:
    return {"full_policy": "fp", "actor_logstd_only": "als", "actor_game": "ag"}.get(scope, scope.replace("_", ""))


def compact_alpha(alpha: float) -> str:
    return f"a_{str(alpha).replace('.', 'p')}"


def compact_lr(lr: float) -> str:
    return f"lr_{str(lr).replace('.', 'p')}"


def compact_method(method_label: str) -> str:
    return {"sgd_gda": "sgd", "egm": "egm", "ppm_inner5": "ppm5"}.get(method_label, method_label)


def method_specs(shared_lr: float, max_grad_norm: float, vf_coef: float, ppm_inner_steps: int) -> List[MethodSpec]:
    return [
        MethodSpec(
            label="sgd_gda",
            optimizer="sgd",
            protagonist_optimizer_kwargs={},
            adversary_optimizer_kwargs={},
            lr=shared_lr,
            max_grad_norm=max_grad_norm,
            vf_coef=vf_coef,
        ),
        MethodSpec(
            label="egm",
            optimizer="egm",
            protagonist_optimizer_kwargs={},
            adversary_optimizer_kwargs={},
            lr=shared_lr,
            max_grad_norm=max_grad_norm,
            vf_coef=vf_coef,
        ),
        MethodSpec(
            label="ppm_inner5",
            optimizer="ppm",
            protagonist_optimizer_kwargs={"inner_steps": ppm_inner_steps},
            adversary_optimizer_kwargs={"inner_steps": ppm_inner_steps},
            lr=shared_lr,
            max_grad_norm=max_grad_norm,
            vf_coef=vf_coef,
        ),
    ]


def build_args_namespace(
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
        adv_index_list=["torso"],
        adv_force_dim=2,
    )


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


def existing_run_matches_request(
    run_dir: pathlib.Path,
    *,
    env_id: str,
    seed: int,
    iterations: int,
    optimizer_scope: str,
    alpha: float,
    method: MethodSpec,
    n_mu: int,
    n_nu: int,
) -> bool:
    run_args = load_run_args(run_dir)
    if not run_args:
        return False
    expected = {
        "seed": seed,
        "env": env_id,
        "n_timesteps": iterations,
        "protagonist_optimizer": method.optimizer,
        "adversary_optimizer": method.optimizer,
        "optimizer_scope": optimizer_scope,
        "adv_impact": "control",
        "N_mu": n_mu,
        "N_nu": n_nu,
    }
    for key, value in expected.items():
        if run_args.get(key) != value:
            return False
    if safe_float(run_args.get("adv_fraction")) != float(alpha):
        return False
    if safe_float(run_args.get("protagonist_lr")) != float(method.lr):
        return False
    if safe_float(run_args.get("adversary_lr")) != float(method.lr):
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


def load_frame(path: pathlib.Path, method: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["method"] = method
    return frame


def run_method(
    *,
    repo_dir: pathlib.Path,
    hyperparam_dir: pathlib.Path,
    output_root: pathlib.Path,
    env_id: str,
    seed: int,
    device: str,
    iterations: int,
    eval_freq: int,
    n_eval_episodes: int,
    optimizer_scope: str,
    alpha: float,
    method: MethodSpec,
    n_mu: int,
    n_nu: int,
):
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from utils.exp_manager import ExperimentManager

    run_root = output_root / "r" / compact_scope(optimizer_scope) / compact_alpha(alpha) / compact_lr(method.lr) / compact_method(method.label)
    analysis_dir = run_root / "an"
    stdout_path = run_root / "stdout.txt"
    stderr_path = run_root / "stderr.txt"
    run_root.mkdir(parents=True, exist_ok=True)

    existing_run_dir = try_find_latest_run_dir(run_root / "sm", env_id)
    analysis_exists = (analysis_dir / "run_summary.csv").exists()
    existing_run_matches = existing_run_dir is not None and existing_run_matches_request(
        existing_run_dir,
        env_id=env_id,
        seed=seed,
        iterations=iterations,
        optimizer_scope=optimizer_scope,
        alpha=alpha,
        method=method,
        n_mu=n_mu,
        n_nu=n_nu,
    )

    if analysis_exists and existing_run_matches:
        run_dir = existing_run_dir
    else:
        if not existing_run_matches:
            ns = build_args_namespace(
                env_id=env_id,
                seed=seed,
                device=device,
                iterations=iterations,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                optimizer_scope=optimizer_scope,
                adv_fraction=alpha,
                method=method,
                hyperparam_dir=hyperparam_dir,
                run_root=run_root,
                n_mu=n_mu,
                n_nu=n_nu,
            )
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
            run_dir = find_latest_run_dir(run_root / "sm", env_id)
        else:
            run_dir = existing_run_dir

        analysis_dir.mkdir(parents=True, exist_ok=True)
        if stdout_path.exists():
            shutil.copy2(stdout_path, analysis_dir / "stdout.txt")
        if stderr_path.exists():
            shutil.copy2(stderr_path, analysis_dir / "stderr.txt")
        run_analysis(repo_dir, run_dir, analysis_dir, method.label)

    summary = pd.read_csv(analysis_dir / "run_summary.csv").iloc[0].to_dict()
    run_args = load_run_args(run_dir)
    summary["requested_adv_fraction"] = safe_float(run_args.get("requested_adv_fraction", run_args.get("adv_fraction")))
    summary["resolved_adv_fraction"] = safe_float(run_args.get("resolved_adv_fraction", run_args.get("adv_fraction")))
    summary["adv_fraction_override"] = int(bool(run_args.get("adv_fraction_override", False)))
    training = load_frame(analysis_dir / "training_episode_returns.csv", method.label)
    clean = load_frame(analysis_dir / "clean_eval_returns.csv", method.label)
    adv = load_frame(analysis_dir / "adversarial_eval_returns.csv", method.label)
    degradation = pd.DataFrame(
        {
            "timesteps": clean["timesteps"],
            "train_return": np.nan,
            "clean_eval_return": clean["mean_reward"],
            "current_adv_eval_return": adv["mean_reward"],
            "current_adv_degradation": clean["mean_reward"] - adv["mean_reward"],
            "local_BR_eval_return": np.nan,
            "local_BR_degradation": np.nan,
            "method": method.label,
        }
    )
    return run_dir, summary, training, clean, adv, degradation


def aggregate_training_metrics(run_dir: pathlib.Path) -> pd.DataFrame:
    role_frames = []
    for role in ("protagonist", "adversary"):
        path = run_dir / "analysis" / f"{role}_training_metrics.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["role"] = role
        frame["update_norm_total"] = np.sqrt(
            frame["actor_update_norm"].fillna(0.0) ** 2
            + frame["logstd_update_norm"].fillna(0.0) ** 2
            + frame["critic_update_norm"].fillna(0.0) ** 2
        )
        rename = {
            "update_norm_total": f"{role}_update_norm_total",
            "actor_update_norm": f"{role}_actor_update_norm",
            "logstd_update_norm": f"{role}_logstd_update_norm",
            "critic_update_norm": f"{role}_critic_update_norm",
            "approx_kl": f"{role}_approx_kl",
            "clip_fraction": f"{role}_clip_fraction",
            "value_loss": f"{role}_value_loss",
            "policy_gradient_loss": f"{role}_policy_gradient_loss",
            "loss": f"{role}_loss",
        }
        cols = ["num_timesteps"] + list(rename.keys())
        keep = frame[cols].rename(columns=rename).sort_values("num_timesteps")
        role_frames.append(keep)
    if not role_frames:
        return pd.DataFrame()
    merged = role_frames[0]
    for frame in role_frames[1:]:
        merged = pd.merge_ordered(merged, frame, on="num_timesteps", how="outer")
    merged = merged.sort_values("num_timesteps").ffill()
    return merged


def build_method_curve(
    *,
    method_label: str,
    shared_lr: float,
    training: pd.DataFrame,
    clean: pd.DataFrame,
    adv: pd.DataFrame,
    degradation: pd.DataFrame,
    metric_frame: pd.DataFrame,
) -> pd.DataFrame:
    curve = degradation.copy()
    curve["train_return"] = np.nan
    if not training.empty:
        train_points = training[["cumulative_timesteps", "episode_return"]].rename(
            columns={"cumulative_timesteps": "timesteps", "episode_return": "train_return"}
        )
        curve = pd.merge_asof(
            curve.sort_values("timesteps"),
            train_points.sort_values("timesteps"),
            on="timesteps",
            direction="backward",
        )
        curve["train_return"] = curve.pop("train_return_y").combine_first(curve.pop("train_return_x"))
    else:
        curve["train_return"] = np.nan

    curve["method"] = method_label

    if metric_frame.empty:
        curve["field_norm"] = np.nan
        curve["surrogate_lyapunov_value"] = np.nan
        curve["actual_surrogate_drift"] = np.nan
        return curve

    merged = pd.merge_asof(
        curve.sort_values("timesteps"),
        metric_frame.sort_values("num_timesteps"),
        left_on="timesteps",
        right_on="num_timesteps",
        direction="backward",
    )
    protagonist_total = merged.get("protagonist_update_norm_total", pd.Series(np.nan, index=merged.index)).fillna(0.0)
    adversary_total = merged.get("adversary_update_norm_total", pd.Series(np.nan, index=merged.index)).fillna(0.0)
    field_norm_proxy = np.sqrt(protagonist_total**2 + adversary_total**2) / max(shared_lr, EPS)
    field_energy = 0.5 * (field_norm_proxy**2)
    degradation_term = merged["current_adv_degradation"].fillna(0.0)
    field0 = abs(field_energy.iloc[0]) + 1.0
    deg0 = abs(degradation_term.iloc[0]) + 1.0
    surrogate = (field_energy / field0) + (degradation_term / deg0)
    merged["field_norm"] = field_norm_proxy
    merged["surrogate_lyapunov_value"] = surrogate
    merged["actual_surrogate_drift"] = merged["surrogate_lyapunov_value"].diff().fillna(0.0)
    return merged.drop(columns=["num_timesteps"], errors="ignore")


def auc_from_curve(frame: pd.DataFrame, x_col: str, y_col: str) -> float:
    if len(frame) < 2:
        return math.nan
    y = pd.to_numeric(frame[y_col], errors="coerce")
    x = pd.to_numeric(frame[x_col], errors="coerce")
    mask = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())
    if mask.sum() < 2:
        return math.nan
    return float(np.trapezoid(y.to_numpy()[mask], x.to_numpy()[mask]))


def dominance_fraction(a: pd.Series, b: pd.Series, higher_better: bool = True) -> float:
    av = pd.to_numeric(a, errors="coerce").to_numpy()
    bv = pd.to_numeric(b, errors="coerce").to_numpy()
    mask = np.isfinite(av) & np.isfinite(bv)
    if mask.sum() == 0:
        return math.nan
    if higher_better:
        return float(np.mean(av[mask] > bv[mask] + 1e-9))
    return float(np.mean(av[mask] < bv[mask] - 1e-9))


def curve_sane(summary_row: Dict[str, object], curve: pd.DataFrame) -> bool:
    if int(summary_row.get("crash_flag", 0)) != 0 or int(summary_row.get("nan_flag", 0)) != 0:
        return False
    required = ["clean_eval_return", "current_adv_eval_return", "current_adv_degradation"]
    for col in required:
        values = pd.to_numeric(curve[col], errors="coerce")
        if not np.isfinite(values.to_numpy()).any():
            return False
    return True


def save_line_plot(frame: pd.DataFrame, x_col: str, y_col: str, ylabel: str, title: str, output_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in frame.groupby("method"):
        values = pd.to_numeric(group[y_col], errors="coerce")
        if not np.isfinite(values.to_numpy()).any():
            continue
        ax.plot(group[x_col], values, label=method, linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_big_figure(frame: pd.DataFrame, output_path: pathlib.Path, subtitle: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    specs = [
        ("train_return", "Train Return"),
        ("clean_eval_return", "Clean Eval Return"),
        ("current_adv_eval_return", "Current-Adversarial Eval Return"),
        ("current_adv_degradation", "Current-Adversarial Degradation"),
        ("field_norm", "Field Norm Proxy"),
        ("surrogate_lyapunov_value", "Surrogate Lyapunov Proxy"),
    ]
    for ax, (column, title) in zip(axes.flatten(), specs):
        for method, group in frame.groupby("method"):
            series = pd.to_numeric(group[column], errors="coerce")
            if not np.isfinite(series.to_numpy()).any():
                continue
            ax.plot(group["timesteps"], series, label=method, linewidth=1.5)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(subtitle)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def config_slug(scope: str, alpha: float, lr: float) -> str:
    return f"{scope}__alpha_{alpha:g}__lr_{lr:g}".replace(".", "p")


def main() -> None:
    args = parse_args()
    repo_dir = pathlib.Path(args.repo_dir)
    output_root = pathlib.Path(args.output_root)
    plots_dir = output_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    env_notes = {"requested_env": args.env, "actual_env": args.env, "env_available": False, "env_error": ""}
    try:
        probe_env = gym.make(args.env)
        probe_env.reset(seed=args.seed)
        probe_env.close()
        env_notes["env_available"] = True
    except Exception as exc:
        env_notes["env_error"] = repr(exc)
        (output_root / "baseline_positive_search_report.md").write_text(
            "\n".join(
                [
                    "# Stage A Baseline-Positive Search",
                    "",
                    f"- requested_env: `{args.env}`",
                    f"- env_available: `False`",
                    f"- env_error: `{repr(exc)}`",
                    "",
                    "Stage A stopped before training because the requested environment could not be constructed in the current runtime.",
                ]
            ),
            encoding="utf-8",
        )
        return

    subprocess.run(
        [
            sys.executable,
            str(repo_dir / "scripts" / "standard_rarl_optimizer_scope_audit.py"),
            "--repo-dir",
            str(repo_dir),
            "--output-root",
            str(output_root),
            "--env",
            args.env,
            "--fallback-env",
            args.fallback_env,
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--shared-lr",
            "1e-3",
            "--shared-max-grad-norm",
            str(args.shared_max_grad_norm),
            "--shared-vf-coef",
            str(args.shared_vf_coef),
        ],
        cwd=str(repo_dir),
        check=True,
    )

    optimizer_scopes = parse_csv_list(args.optimizer_scopes, str)
    alphas = parse_csv_list(args.alphas, float)
    shared_lrs = parse_csv_list(args.shared_lrs, float)
    hyperparam_dir = ensure_hyperparams(repo_dir, output_root, args.env, args.fallback_env)

    summary_rows: List[Dict[str, object]] = []
    curve_frames: List[pd.DataFrame] = []
    config_rows: List[Dict[str, object]] = []
    best_config_key = None
    best_config_score = -math.inf
    best_config_curve = None

    for optimizer_scope in optimizer_scopes:
        for alpha in alphas:
            for shared_lr in shared_lrs:
                per_method: Dict[str, Dict[str, object]] = {}
                per_curve: Dict[str, pd.DataFrame] = {}
                methods = method_specs(shared_lr, args.shared_max_grad_norm, args.shared_vf_coef, args.ppm_inner_steps)
                for method in methods:
                    run_dir, summary, training, clean, adv, degradation = run_method(
                        repo_dir=repo_dir,
                        hyperparam_dir=hyperparam_dir,
                        output_root=output_root,
                        env_id=args.env,
                        seed=args.seed,
                        device=args.device,
                        iterations=args.iterations,
                        eval_freq=args.eval_freq,
                        n_eval_episodes=args.n_eval_episodes,
                        optimizer_scope=optimizer_scope,
                        alpha=alpha,
                        method=method,
                        n_mu=args.n_mu,
                        n_nu=args.n_nu,
                    )
                    metrics_frame = aggregate_training_metrics(run_dir)
                    curve = build_method_curve(
                        method_label=method.label,
                        shared_lr=shared_lr,
                        training=training,
                        clean=clean,
                        adv=adv,
                        degradation=degradation,
                        metric_frame=metrics_frame,
                    )
                    per_curve[method.label] = curve
                    row = dict(summary)
                    row.update(
                        {
                            "optimizer_scope": optimizer_scope,
                            "alpha": alpha,
                            "shared_lr": shared_lr,
                            "method": method.label,
                            "mapped_optimizer": method.optimizer,
                            "ppm_inner_steps": args.ppm_inner_steps if method.label == "ppm_inner5" else math.nan,
                            "curve_sane_flag": int(curve_sane(row, curve)),
                            "train_return_AUC": auc_from_curve(curve, "timesteps", "train_return"),
                            "clean_eval_return_AUC": auc_from_curve(curve, "timesteps", "clean_eval_return"),
                            "current_adv_eval_return_AUC": auc_from_curve(curve, "timesteps", "current_adv_eval_return"),
                            "local_BR_eval_return_AUC": math.nan,
                            "current_adv_degradation_AUC": auc_from_curve(curve, "timesteps", "current_adv_degradation"),
                            "local_BR_degradation_AUC": math.nan,
                            "field_norm_AUC": auc_from_curve(curve, "timesteps", "field_norm"),
                            "surrogate_lyapunov_AUC": auc_from_curve(curve, "timesteps", "surrogate_lyapunov_value"),
                            "final_train_return": safe_float(curve["train_return"].dropna().iloc[-1]) if curve["train_return"].notna().any() else math.nan,
                            "final_clean_eval_return": safe_float(curve["clean_eval_return"].iloc[-1]),
                            "final_current_adv_eval_return": safe_float(curve["current_adv_eval_return"].iloc[-1]),
                            "final_local_BR_eval_return": math.nan,
                            "final_current_adv_degradation": safe_float(curve["current_adv_degradation"].iloc[-1]),
                            "final_local_BR_degradation": math.nan,
                            "requested_alpha_matches_resolved_flag": int(
                                finite(row.get("requested_adv_fraction"))
                                and finite(row.get("resolved_adv_fraction"))
                                and abs(float(row["requested_adv_fraction"]) - float(row["resolved_adv_fraction"])) <= 1e-9
                            ),
                            "current_adv_curve_hash": curve_hash(curve["current_adv_eval_return"].tolist()),
                            "clean_eval_curve_hash": curve_hash(curve["clean_eval_return"].tolist()),
                        }
                    )
                    per_method[method.label] = row
                    curve_frames.append(curve.assign(optimizer_scope=optimizer_scope, alpha=alpha, shared_lr=shared_lr))
                    summary_rows.append(row)

                sgd = per_method["sgd_gda"]
                egm = per_method["egm"]
                ppm = per_method["ppm_inner5"]
                sgd_curve = per_curve["sgd_gda"].sort_values("timesteps").reset_index(drop=True)
                egm_curve = per_curve["egm"].sort_values("timesteps").reset_index(drop=True)
                ppm_curve = per_curve["ppm_inner5"].sort_values("timesteps").reset_index(drop=True)
                common = sgd_curve[["timesteps", "current_adv_eval_return"]].rename(columns={"current_adv_eval_return": "sgd_current_adv"})
                common = common.merge(
                    egm_curve[["timesteps", "current_adv_eval_return"]].rename(columns={"current_adv_eval_return": "egm_current_adv"}),
                    on="timesteps",
                    how="inner",
                )
                common = common.merge(
                    ppm_curve[["timesteps", "current_adv_eval_return"]].rename(columns={"current_adv_eval_return": "ppm_current_adv"}),
                    on="timesteps",
                    how="inner",
                )
                egm_beats_mask = (pd.to_numeric(common["egm_current_adv"], errors="coerce") > pd.to_numeric(common["sgd_current_adv"], errors="coerce") + 1e-9)
                ppm_beats_mask = (pd.to_numeric(common["ppm_current_adv"], errors="coerce") > pd.to_numeric(common["sgd_current_adv"], errors="coerce") + 1e-9)
                egm_dom = dominance_fraction(common["egm_current_adv"], common["sgd_current_adv"], higher_better=True)
                ppm_dom = dominance_fraction(common["ppm_current_adv"], common["sgd_current_adv"], higher_better=True)
                sgd_auc = safe_float(sgd["current_adv_eval_return_AUC"])
                egm_auc = safe_float(egm["current_adv_eval_return_AUC"])
                ppm_auc = safe_float(ppm["current_adv_eval_return_AUC"])
                egm_improve = (egm_auc / (sgd_auc + EPS)) - 1.0 if finite(sgd_auc) and finite(egm_auc) else math.nan
                ppm_improve = (ppm_auc / (sgd_auc + EPS)) - 1.0 if finite(sgd_auc) and finite(ppm_auc) else math.nan
                egm_clean_ratio = (safe_float(egm["clean_eval_return_AUC"]) / (safe_float(sgd["clean_eval_return_AUC"]) + EPS)) if finite(egm["clean_eval_return_AUC"]) and finite(sgd["clean_eval_return_AUC"]) else math.nan
                ppm_clean_ratio = (safe_float(ppm["clean_eval_return_AUC"]) / (safe_float(sgd["clean_eval_return_AUC"]) + EPS)) if finite(ppm["clean_eval_return_AUC"]) and finite(sgd["clean_eval_return_AUC"]) else math.nan
                candidate_improve = max(
                    egm_improve if finite(egm_improve) else -math.inf,
                    ppm_improve if finite(ppm_improve) else -math.inf,
                )
                candidate_dom = max(
                    egm_dom if finite(egm_dom) else -math.inf,
                    ppm_dom if finite(ppm_dom) else -math.inf,
                )
                all_curve_sane = int(sgd["curve_sane_flag"] == 1 and egm["curve_sane_flag"] == 1 and ppm["curve_sane_flag"] == 1)
                ppm_identical_to_egm = int(ppm["current_adv_curve_hash"] == egm["current_adv_curve_hash"])
                alpha_match_all = int(
                    sgd["requested_alpha_matches_resolved_flag"] == 1
                    and egm["requested_alpha_matches_resolved_flag"] == 1
                    and ppm["requested_alpha_matches_resolved_flag"] == 1
                )
                egm_not_final_only = int(bool(len(egm_beats_mask) >= 2 and egm_beats_mask.iloc[:-1].any()))
                ppm_not_final_only = int(bool(len(ppm_beats_mask) >= 2 and ppm_beats_mask.iloc[:-1].any()))
                positive = bool(
                    candidate_improve >= 0.10
                    and candidate_dom >= 0.70
                    and all_curve_sane == 1
                    and alpha_match_all == 1
                    and ppm_identical_to_egm == 0
                    and (
                        (finite(egm_improve) and egm_improve >= ppm_improve and finite(egm_clean_ratio) and egm_clean_ratio >= 0.80 and egm_not_final_only == 1)
                        or (finite(ppm_improve) and ppm_improve > egm_improve and finite(ppm_clean_ratio) and ppm_clean_ratio >= 0.80 and ppm_not_final_only == 1)
                    )
                )
                strong = bool(
                    candidate_improve >= 0.20
                    and candidate_dom >= 0.80
                    and all_curve_sane == 1
                    and alpha_match_all == 1
                    and ppm_identical_to_egm == 0
                    and (
                        (finite(egm_improve) and egm_improve >= ppm_improve and finite(egm_clean_ratio) and egm_clean_ratio >= 0.80 and egm_not_final_only == 1)
                        or (finite(ppm_improve) and ppm_improve > egm_improve and finite(ppm_clean_ratio) and ppm_clean_ratio >= 0.80 and ppm_not_final_only == 1)
                    )
                )
                winner = "egm" if (finite(egm_improve) and egm_improve >= ppm_improve) else "ppm_inner5"
                config_row = {
                    "optimizer_scope": optimizer_scope,
                    "alpha": alpha,
                    "shared_lr": shared_lr,
                    "sgd_curve_sane_flag": sgd["curve_sane_flag"],
                    "egm_curve_sane_flag": egm["curve_sane_flag"],
                    "ppm_curve_sane_flag": ppm["curve_sane_flag"],
                    "sgd_current_adv_eval_return_AUC": sgd_auc,
                    "egm_current_adv_eval_return_AUC": egm_auc,
                    "ppm_current_adv_eval_return_AUC": ppm_auc,
                    "sgd_local_BR_eval_return_AUC": math.nan,
                    "egm_local_BR_eval_return_AUC": math.nan,
                    "ppm_local_BR_eval_return_AUC": math.nan,
                    "EGM_over_SGD_fraction": egm_dom,
                    "PPM_over_SGD_fraction": ppm_dom,
                    "EGM_clean_auc_ratio_vs_SGD": egm_clean_ratio,
                    "PPM_clean_auc_ratio_vs_SGD": ppm_clean_ratio,
                    "best_improvement_frac": candidate_improve,
                    "best_dominance_fraction": candidate_dom,
                    "winner": winner,
                    "all_curve_sane_flag": all_curve_sane,
                    "requested_alpha_matches_resolved_flag": alpha_match_all,
                    "ppm_identical_to_egm_flag": ppm_identical_to_egm,
                    "egm_not_final_only_flag": egm_not_final_only,
                    "ppm_not_final_only_flag": ppm_not_final_only,
                    "baseline_positive_flag": int(positive),
                    "strong_pass_flag": int(strong),
                }
                config_rows.append(config_row)
                config_key = config_slug(optimizer_scope, alpha, shared_lr)
                if finite(candidate_improve):
                    score = candidate_improve + 0.1 * (candidate_dom if finite(candidate_dom) else 0.0)
                    if score > best_config_score:
                        best_config_score = score
                        best_config_key = config_key
                        best_config_curve = pd.concat(
                            [
                                per_curve["sgd_gda"].assign(method="sgd_gda"),
                                per_curve["egm"].assign(method="egm"),
                                per_curve["ppm_inner5"].assign(method="ppm_inner5"),
                            ],
                            ignore_index=True,
                        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["optimizer_scope", "alpha", "shared_lr", "method"]).reset_index(drop=True)
    curves_df = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    config_df = pd.DataFrame(config_rows).sort_values(
        ["strong_pass_flag", "baseline_positive_flag", "best_improvement_frac", "best_dominance_fraction"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    alpha_hash_audit = pd.DataFrame()
    if not summary_df.empty:
        alpha_hash_audit = (
            summary_df.groupby(["optimizer_scope", "shared_lr", "method"])
            .agg(
                unique_adv_curve_hashes=("current_adv_curve_hash", "nunique"),
                unique_clean_curve_hashes=("clean_eval_curve_hash", "nunique"),
            )
            .reset_index()
        )
        alpha_hash_audit["alpha_curve_sensitive_flag"] = (
            (alpha_hash_audit["unique_adv_curve_hashes"] > 1) & (alpha_hash_audit["unique_clean_curve_hashes"] > 1)
        ).astype(int)
        alpha_scope_lr = (
            alpha_hash_audit.groupby(["optimizer_scope", "shared_lr"])["alpha_curve_sensitive_flag"]
            .min()
            .reset_index()
            .rename(columns={"alpha_curve_sensitive_flag": "alpha_curve_sensitive_flag_all_methods"})
        )
        config_df = config_df.merge(alpha_scope_lr, on=["optimizer_scope", "shared_lr"], how="left")
        config_df["alpha_curve_sensitive_flag_all_methods"] = config_df["alpha_curve_sensitive_flag_all_methods"].fillna(0).astype(int)
        config_df["baseline_positive_flag"] = (
            (config_df["baseline_positive_flag"] == 1)
            & (config_df["alpha_curve_sensitive_flag_all_methods"] == 1)
        ).astype(int)
        config_df["strong_pass_flag"] = (
            (config_df["strong_pass_flag"] == 1)
            & (config_df["alpha_curve_sensitive_flag_all_methods"] == 1)
        ).astype(int)

    summary_df.to_csv(output_root / "baseline_positive_search_summary.csv", index=False)
    curves_df.to_csv(output_root / "baseline_positive_search_curves.csv", index=False)
    config_df.to_csv(output_root / "baseline_positive_search_config_summary.csv", index=False)
    if not alpha_hash_audit.empty:
        alpha_hash_audit.to_csv(output_root / "baseline_positive_search_alpha_hash_audit.csv", index=False)
    config_df = config_df.sort_values(
        ["strong_pass_flag", "baseline_positive_flag", "best_improvement_frac", "best_dominance_fraction"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    top_lines = [
        "# Stage A Baseline-Positive Top Configs",
        "",
        "Ranking is by strict pass flags first, then best improvement fraction and dominance fraction.",
        "",
    ]
    for _, row in config_df.head(12).iterrows():
        top_lines.append(
            f"- scope=`{row['optimizer_scope']}`, alpha=`{row['alpha']}`, lr=`{row['shared_lr']}`:"
            f" positive=`{bool(row['baseline_positive_flag'])}`, strong=`{bool(row['strong_pass_flag'])}`,"
            f" winner=`{row['winner']}`, best_improvement_frac=`{row['best_improvement_frac']:.3f}`,"
            f" best_dominance_fraction=`{row['best_dominance_fraction']:.3f}`"
        )
    (output_root / "baseline_positive_top_configs.md").write_text("\n".join(top_lines), encoding="utf-8")
    summary_df.to_csv(output_root / "baseline_search_all_configs.csv", index=False)
    config_df.to_csv(output_root / "baseline_search_ranked.csv", index=False)

    positive_count = int(config_df["baseline_positive_flag"].sum()) if not config_df.empty else 0
    strong_count = int(config_df["strong_pass_flag"].sum()) if not config_df.empty else 0
    winner_counts = config_df["winner"].value_counts().to_dict() if not config_df.empty else {}
    confirmed_egm = bool(((config_df["winner"] == "egm") & (config_df["baseline_positive_flag"] == 1)).any()) if not config_df.empty else False
    confirmed_ppm = bool(((config_df["winner"] == "ppm_inner5") & (config_df["baseline_positive_flag"] == 1)).any()) if not config_df.empty else False
    if confirmed_egm and confirmed_ppm:
        final_decision = "CONFIRMED_BASELINE_BOTH"
    elif confirmed_egm:
        final_decision = "CONFIRMED_BASELINE_EGM"
    elif confirmed_ppm:
        final_decision = "CONFIRMED_BASELINE_PPM"
    else:
        final_decision = "ALPHA_FIXED_BUT_NO_BASELINE_POSITIVE"
    report_lines = [
        "# Stage A Baseline-Positive Search",
        "",
        f"- env: `{args.env}`",
        f"- seed: `{args.seed}`",
        f"- alternating schedule: `N_mu={args.n_mu}`, `N_nu={args.n_nu}`",
        f"- disturbance wrapper: `a_env = clip(u + alpha * w)`",
        f"- reward: `original env reward` for protagonist, adversary gets the negated reward via the standard RARL wrapper",
        f"- reward shaping added: `False`",
        f"- methods: `sgd_gda`, `egm`, `ppm_inner5`",
        f"- methods_not_run: `proposed_nog_closed`, `proposed_qp_closed`",
        f"- PPM inner_steps: `{args.ppm_inner_steps}`",
        "",
        "## Metric Notes",
        "",
        "- `current_adv_eval_return` is the primary robust metric used for pass/fail in Stage A.",
        "- `local_BR_eval_return` is marked unavailable in this Stage A search because the vanilla PPO/RARL baseline pipeline does not expose a cheap local best-response evaluator.",
        "- `field_norm` is a proxy built from protagonist/adversary PPO update blocks recorded in `analysis/*_training_metrics.csv`, rescaled by the shared lr.",
        "- `surrogate_lyapunov_value` is a Stage-A proxy: normalized field-energy proxy plus normalized current-adversarial degradation.",
        "",
        "## Search Outcome",
        "",
        "- old Stage 3A results were invalidated and not reused",
        f"- all reused runs are under: `{output_root}`",
        "- reused runs are accepted only when `requested_alpha == resolved_alpha`",
        f"- configs_evaluated: `{len(config_df)}`",
        f"- baseline_positive_count: `{positive_count}`",
        f"- strong_pass_count: `{strong_count}`",
        f"- winner_counts: `{winner_counts}`",
        f"- best_config_key: `{best_config_key}`",
        f"- final_decision: `{final_decision}`",
        "",
        "## Best Configs",
        "",
    ]
    for _, row in config_df.head(10).iterrows():
        report_lines.append(
            f"- scope=`{row['optimizer_scope']}`, alpha=`{row['alpha']}`, lr=`{row['shared_lr']}`:"
            f" positive=`{bool(row['baseline_positive_flag'])}`, strong=`{bool(row['strong_pass_flag'])}`,"
            f" winner=`{row['winner']}`, EGM_over_SGD_fraction=`{row['EGM_over_SGD_fraction']:.3f}`,"
            f" PPM_over_SGD_fraction=`{row['PPM_over_SGD_fraction']:.3f}`, best_improvement_frac=`{row['best_improvement_frac']:.3f}`,"
            f" alpha_match=`{row['requested_alpha_matches_resolved_flag']}`,"
            f" egm_not_final_only=`{row['egm_not_final_only_flag']}`, ppm_not_final_only=`{row['ppm_not_final_only_flag']}`"
        )
    report_lines.extend(
        [
            "",
            "## Decision Labels",
            "",
            "- `CONFIRMED_BASELINE_EGM`",
            "- `CONFIRMED_BASELINE_PPM`",
            "- `CONFIRMED_BASELINE_BOTH`",
            "- `ALPHA_FIXED_BUT_NO_BASELINE_POSITIVE`",
            "",
            "## Stop Condition",
            "",
            "Stage A only. Proposed QP/noG were not implemented or run here.",
        ]
    )
    (output_root / "baseline_positive_search_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    (output_root / "baseline_search_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    (output_root / "baseline_search_after_alpha_fix_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    (output_root / "baseline_search_after_alpha_fix_decision.md").write_text(
        f"# Stage 3A Baseline Search After Alpha Fix Decision\n\n- decision: `{final_decision}`\n",
        encoding="utf-8",
    )

    if best_config_curve is not None and not best_config_curve.empty:
        save_line_plot(best_config_curve, "timesteps", "train_return", "Return", "Best Config Train Return", plots_dir / "stageA_train_return.png")
        save_line_plot(best_config_curve, "timesteps", "clean_eval_return", "Return", "Best Config Clean Eval Return", plots_dir / "stageA_clean_eval_return.png")
        save_line_plot(best_config_curve, "timesteps", "current_adv_eval_return", "Return", "Best Config Current-Adversarial Eval Return", plots_dir / "stageA_current_adv_eval_return.png")
        save_line_plot(best_config_curve, "timesteps", "current_adv_degradation", "Clean - Current Adv", "Best Config Current-Adversarial Degradation", plots_dir / "stageA_degradation.png")
        save_line_plot(best_config_curve, "timesteps", "field_norm", "Field Norm Proxy", "Best Config Field Norm Proxy", plots_dir / "stageA_field_norm.png")
        save_line_plot(best_config_curve, "timesteps", "surrogate_lyapunov_value", "Proxy V", "Best Config Surrogate Lyapunov Proxy", plots_dir / "stageA_surrogate_lyapunov.png")
        save_big_figure(best_config_curve, plots_dir / "stageA_all_plots_big.png", f"Best Stage A Config: {best_config_key}")
        save_big_figure(best_config_curve, plots_dir / "baseline_search_top_configs.png", f"Best Stage A Config: {best_config_key}")
        save_big_figure(best_config_curve, plots_dir / "stageA_after_alpha_fix_all_plots_big.png", f"Best Stage A Config: {best_config_key}")


if __name__ == "__main__":
    main()
