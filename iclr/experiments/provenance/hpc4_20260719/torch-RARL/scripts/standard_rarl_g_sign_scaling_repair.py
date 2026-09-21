from __future__ import annotations

import argparse
import contextlib
import importlib.util
import math
import pathlib
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def auc_from_curve(frame: pd.DataFrame, x_col: str, y_col: str) -> float:
    if frame.empty or x_col not in frame.columns or y_col not in frame.columns:
        return math.nan
    sub = frame[[x_col, y_col]].dropna()
    if len(sub) < 2:
        return math.nan
    return float(np.trapezoid(sub[y_col].to_numpy(dtype=float), sub[x_col].to_numpy(dtype=float)))


def dominance_fraction(a: Sequence[float], b: Sequence[float], higher_better: bool = True) -> float:
    av = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(dtype=float)
    bv = pd.to_numeric(pd.Series(b), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(av) & np.isfinite(bv)
    if mask.sum() == 0:
        return math.nan
    if higher_better:
        return float(np.mean(av[mask] > bv[mask] + 1e-9))
    return float(np.mean(av[mask] < bv[mask] - 1e-9))


def finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def finite_max(*values: float) -> float:
    finite_values = [float(v) for v in values if finite(v)]
    return max(finite_values) if finite_values else math.nan


def parse_args() -> argparse.Namespace:
    repo_default = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser("Standard RARL G sign/scaling repair")
    parser.add_argument("--repo-dir", type=str, default=str(repo_default))
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(repo_default.parent / "results" / "standard_rarl_g_sign_scaling_repair"),
    )
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def register_optimizers(repo_dir: pathlib.Path) -> None:
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from models.optimizers import OPTIMIZER_REGISTRY
    from models.proposed_qp_closedlyap_merit import (
        ProposedNoGClosedTrustRegionMeritOptimizer,
        ProposedQPClosedTrustRegionMeritOptimizer,
        ProposedQPClosedMinusGTrustRegionMeritOptimizer,
        ProposedQPClosedSignSelectTrustRegionMeritOptimizer,
        ProposedQPNogSafeSignSelectTrustRegionMeritOptimizer,
    )

    OPTIMIZER_REGISTRY["proposed_nog_closed_trustregion"] = ProposedNoGClosedTrustRegionMeritOptimizer
    OPTIMIZER_REGISTRY["proposed_qp_plusg_closed_trustregion"] = ProposedQPClosedTrustRegionMeritOptimizer
    OPTIMIZER_REGISTRY["proposed_qp_minusg_closed_trustregion"] = ProposedQPClosedMinusGTrustRegionMeritOptimizer
    OPTIMIZER_REGISTRY["proposed_qp_signselect_closed_trustregion"] = ProposedQPClosedSignSelectTrustRegionMeritOptimizer
    OPTIMIZER_REGISTRY["proposed_qp_nogsafe_signselect_closed_trustregion"] = ProposedQPNogSafeSignSelectTrustRegionMeritOptimizer


def make_diag_kwargs(method_stem: str, role: str, diagnostics_root: pathlib.Path) -> Dict[str, object]:
    return {
        "optimizer_scope": "full_policy",
        "lambda_F": 0.01,
        "lambda_R": 0.3,
        "lambda_KL": 0.3,
        "lambda_CF": 0.1,
        "target_kl": 0.03,
        "vf_coef": 0.5,
        "fd_eps": 1e-3,
        "beta_probe": 3e-4,
        "gamma_probe": 1e-6,
        "ridge": 1e-8,
        "beta_max": 9e-4,
        "gamma_max": 3e-6,
        "max_update_norm": 0.005,
        "qp_eps": 1e-8,
        "allow_fallback_to_egm": False,
        "cost_mode": "trust_region_policy_kl_clip",
        "role": role,
        "diagnostics_csv_path": str(diagnostics_root / f"{role}_{method_stem}.csv"),
    }


def method_specs(diagnostics_root: pathlib.Path) -> List[MethodSpec]:
    return [
        MethodSpec("sgd_gda", "sgd", {}, {}, 3e-4, 10.0, 0.5),
        MethodSpec("egm", "egm", {}, {}, 3e-4, 10.0, 0.5),
        MethodSpec(
            "proposed_nog_closed_trustreg",
            "proposed_nog_closed_trustregion",
            make_diag_kwargs("nog_trustreg", "protagonist", diagnostics_root),
            make_diag_kwargs("nog_trustreg", "adversary", diagnostics_root),
            3e-4,
            10.0,
            0.5,
        ),
        MethodSpec(
            "proposed_qp_plusG_trustreg",
            "proposed_qp_plusg_closed_trustregion",
            make_diag_kwargs("plusg_trustreg", "protagonist", diagnostics_root),
            make_diag_kwargs("plusg_trustreg", "adversary", diagnostics_root),
            3e-4,
            10.0,
            0.5,
        ),
        MethodSpec(
            "proposed_qp_minusG_trustreg",
            "proposed_qp_minusg_closed_trustregion",
            make_diag_kwargs("minusg_trustreg", "protagonist", diagnostics_root),
            make_diag_kwargs("minusg_trustreg", "adversary", diagnostics_root),
            3e-4,
            10.0,
            0.5,
        ),
        MethodSpec(
            "proposed_qp_sign_select_trustreg",
            "proposed_qp_signselect_closed_trustregion",
            make_diag_kwargs("signselect_trustreg", "protagonist", diagnostics_root),
            make_diag_kwargs("signselect_trustreg", "adversary", diagnostics_root),
            3e-4,
            10.0,
            0.5,
        ),
        MethodSpec(
            "proposed_qp_nog_safe_sign_select_trustreg",
            "proposed_qp_nogsafe_signselect_closed_trustregion",
            make_diag_kwargs("nogsafe_trustreg", "protagonist", diagnostics_root),
            make_diag_kwargs("nogsafe_trustreg", "adversary", diagnostics_root),
            3e-4,
            10.0,
            0.5,
        ),
    ]


def build_ns(method: MethodSpec, run_root: pathlib.Path, repo_dir: pathlib.Path, env: str, seed: int, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        verbose=1,
        seed=seed,
        num_exps=1,
        num_threads=-1,
        env=env,
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
        hyperparameter_path=str(repo_dir / "hyperparameter"),
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
        n_timesteps=6,
        save_freq=10240,
        log_interval=-1,
        device=device,
        eval_freq=10240,
        n_eval_envs=1,
        n_eval_episodes=5,
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
        optimizer_scope="full_policy",
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
        N_mu=5,
        N_nu=1,
        adv_impact="control",
        adv_delay=-1,
        adv_fraction=0.05,
        adv_fraction_override=True,
        requested_alpha=0.05,
        requested_adv_fraction=0.05,
        adv_index_list=["torso"],
        adv_force_dim=2,
    )


def run_with_exp_manager(repo_dir: pathlib.Path, env_id: str, run_root: pathlib.Path, ns: SimpleNamespace) -> pathlib.Path:
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


def read_diag_pair(diagnostics_root: pathlib.Path, method_stem: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for role in ["protagonist", "adversary"]:
        path = diagnostics_root / f"{role}_{method_stem}.csv"
        if path.exists():
            frame = pd.read_csv(path)
            if not frame.empty:
                frame["diag_role"] = role
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_diag(diag_df: pd.DataFrame) -> Dict[str, object]:
    if diag_df.empty:
        return {}
    out: Dict[str, object] = {}
    for col in [
        "V_before",
        "V_after_noG",
        "V_after_plusG_QP",
        "V_after_minusG_QP",
        "actual_drift_noG",
        "actual_drift_plusG_QP",
        "actual_drift_minusG_QP",
        "plusG_better_than_noG",
        "minusG_better_than_noG",
        "sign_select_better_than_noG",
        "nog_safe_qp_better_than_noG",
        "beta_plus",
        "gamma_plus",
        "beta_minus",
        "gamma_minus",
        "gamma_active_plus",
        "gamma_active_minus",
        "G_contribution_ratio_plus",
        "G_contribution_ratio_minus",
        "cos_F_G",
        "non_collinearity",
    ]:
        if col in diag_df.columns:
            out[f"avg_{col}"] = float(pd.to_numeric(diag_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).mean())
    if "chosen_step" in diag_df.columns:
        chosen = diag_df["chosen_step"].fillna("unknown").astype(str)
        out["chosen_plus_frac"] = float((chosen == "plusG").mean())
        out["chosen_minus_frac"] = float((chosen == "minusG").mean())
        out["chosen_noG_frac"] = float((chosen == "noG").mean())
    return out


def plot_curves(curves: Dict[str, pd.DataFrame], output_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    metrics = [
        ("train_return", "Train Return"),
        ("clean_eval_return", "Clean Eval Return"),
        ("current_adv_eval_return", "Current-Adv Eval Return"),
        ("current_adv_degradation", "Current-Adv Degradation"),
    ]
    for ax, (col, title) in zip(axes.flatten(), metrics):
        for method, frame in curves.items():
            if col in frame.columns:
                ax.plot(frame["timesteps"], frame[col], label=method, linewidth=1.4)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_step_diagnostics(summary_df: pd.DataFrame, output_path: pathlib.Path) -> None:
    methods = [
        "proposed_qp_plusG_trustreg",
        "proposed_qp_minusG_trustreg",
        "proposed_qp_sign_select_trustreg",
        "proposed_qp_nog_safe_sign_select_trustreg",
    ]
    sub = summary_df[summary_df["method"].isin(methods)].copy()
    if sub.empty:
        return
    x = np.arange(len(sub))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].bar(x - 0.15, sub["better_than_nog_fraction"], width=0.3, label="beats noG fraction")
    axes[0].bar(x + 0.15, sub["dominance_over_nog"], width=0.3, label="dominance over noG")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sub["method"], rotation=35, ha="right", fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[0].set_title("Step/curve advantage over noG")

    axes[1].bar(x - 0.2, sub["avg_gamma_active"], width=0.2, label="gamma active")
    axes[1].bar(x, sub["avg_g_ratio"], width=0.2, label="G ratio")
    axes[1].bar(x + 0.2, sub["avg_non_collinearity"], width=0.2, label="non-collinearity")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sub["method"], rotation=35, ha="right", fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    axes[1].set_title("Geometry usage diagnostics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_dir = pathlib.Path(args.repo_dir).resolve()
    output_root = ensure_dir(pathlib.Path(args.output_root).resolve())
    plots_root = ensure_dir(output_root / "plots")

    register_optimizers(repo_dir)
    base_module = load_module("gsign_base_mod", repo_dir / "scripts" / "standard_rarl_baseline_positive_search.py")
    helper_module = load_module("gsign_helper_mod", repo_dir / "scripts" / "standard_rarl_lyapunov_repair_minisearch.py")
    diagnostics_root = ensure_dir(output_root / "diagnostics")

    rows: List[Dict[str, object]] = []
    curves: Dict[str, pd.DataFrame] = {}

    method_to_stem = {
        "proposed_nog_closed_trustreg": "nog_trustreg",
        "proposed_qp_plusG_trustreg": "plusg_trustreg",
        "proposed_qp_minusG_trustreg": "minusg_trustreg",
        "proposed_qp_sign_select_trustreg": "signselect_trustreg",
        "proposed_qp_nog_safe_sign_select_trustreg": "nogsafe_trustreg",
    }

    for method in method_specs(diagnostics_root):
        run_root = ensure_dir(output_root / method.label)
        ns = build_ns(method, run_root, repo_dir, args.env, args.seed, args.device)
        run_dir = run_with_exp_manager(repo_dir, args.env, run_root, ns)
        analysis_dir = ensure_dir(run_root / "analysis")
        summary, curve = helper_module.analyze_run(base_module, repo_dir, run_dir, analysis_dir, method.label)
        curves[method.label] = curve
        row = {
            "method": method.label,
            "train_return_AUC": auc_from_curve(curve, "timesteps", "train_return"),
            "clean_eval_return_AUC": auc_from_curve(curve, "timesteps", "clean_eval_return"),
            "current_adv_eval_return_AUC": auc_from_curve(curve, "timesteps", "current_adv_eval_return"),
            "current_adv_degradation_AUC": auc_from_curve(curve, "timesteps", "current_adv_degradation"),
            "final_current_adv_eval_return": safe_float(pd.to_numeric(curve["current_adv_eval_return"], errors="coerce").dropna().iloc[-1] if "current_adv_eval_return" in curve.columns and not curve.empty else math.nan),
        }
        if method.label in method_to_stem:
            diag = read_diag_pair(diagnostics_root, method_to_stem[method.label])
            diag_summary = summarize_diag(diag)
            row["better_than_nog_fraction"] = safe_float(
                diag_summary.get(
                    "avg_plusG_better_than_noG"
                    if method.label == "proposed_qp_plusG_trustreg"
                    else "avg_minusG_better_than_noG"
                    if method.label == "proposed_qp_minusG_trustreg"
                    else "avg_sign_select_better_than_noG"
                    if method.label == "proposed_qp_sign_select_trustreg"
                    else "avg_nog_safe_qp_better_than_noG",
                    math.nan,
                ),
                math.nan,
            )
            row["avg_gamma_active"] = safe_float(
                diag_summary.get(
                    "avg_gamma_active_plus"
                    if method.label == "proposed_qp_plusG_trustreg"
                    else "avg_gamma_active_minus"
                    if method.label == "proposed_qp_minusG_trustreg"
                    else finite_max(
                        safe_float(diag_summary.get("avg_gamma_active_plus", math.nan), math.nan),
                        safe_float(diag_summary.get("avg_gamma_active_minus", math.nan), math.nan),
                    ),
                    math.nan,
                ),
                math.nan,
            )
            row["avg_g_ratio"] = safe_float(
                diag_summary.get(
                    "avg_G_contribution_ratio_plus"
                    if method.label == "proposed_qp_plusG_trustreg"
                    else "avg_G_contribution_ratio_minus"
                    if method.label == "proposed_qp_minusG_trustreg"
                    else finite_max(
                        safe_float(diag_summary.get("avg_G_contribution_ratio_plus", math.nan), math.nan),
                        safe_float(diag_summary.get("avg_G_contribution_ratio_minus", math.nan), math.nan),
                    ),
                    math.nan,
                ),
                math.nan,
            )
            row["avg_non_collinearity"] = safe_float(diag_summary.get("avg_non_collinearity", math.nan), math.nan)
            row["chosen_plus_frac"] = safe_float(diag_summary.get("chosen_plus_frac", math.nan), math.nan)
            row["chosen_minus_frac"] = safe_float(diag_summary.get("chosen_minus_frac", math.nan), math.nan)
            row["chosen_noG_frac"] = safe_float(diag_summary.get("chosen_noG_frac", math.nan), math.nan)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    nog_curve = curves["proposed_nog_closed_trustreg"][["timesteps", "current_adv_eval_return"]].rename(columns={"current_adv_eval_return": "nog"})
    egm_curve = curves["egm"][["timesteps", "current_adv_eval_return"]].rename(columns={"current_adv_eval_return": "egm"})

    for method in [
        "proposed_qp_plusG_trustreg",
        "proposed_qp_minusG_trustreg",
        "proposed_qp_sign_select_trustreg",
        "proposed_qp_nog_safe_sign_select_trustreg",
    ]:
        curve = curves[method][["timesteps", "current_adv_eval_return", "clean_eval_return"]].rename(
            columns={"current_adv_eval_return": "candidate", "clean_eval_return": "candidate_clean"}
        )
        merged = nog_curve.merge(curve, on="timesteps", how="inner").merge(egm_curve, on="timesteps", how="inner")
        summary_df.loc[summary_df["method"] == method, "dominance_over_nog"] = dominance_fraction(merged["candidate"], merged["nog"], higher_better=True)
        summary_df.loc[summary_df["method"] == method, "dominance_over_egm"] = dominance_fraction(merged["candidate"], merged["egm"], higher_better=True)
        summary_df.loc[summary_df["method"] == method, "improve_vs_nog_auc"] = safe_float(
            summary_df.loc[summary_df["method"] == method, "current_adv_eval_return_AUC"].iloc[0]
        ) / (safe_float(summary_df.loc[summary_df["method"] == "proposed_nog_closed_trustreg", "current_adv_eval_return_AUC"].iloc[0]) + EPS) - 1.0
        summary_df.loc[summary_df["method"] == method, "improve_vs_egm_auc"] = safe_float(
            summary_df.loc[summary_df["method"] == method, "current_adv_eval_return_AUC"].iloc[0]
        ) / (safe_float(summary_df.loc[summary_df["method"] == "egm", "current_adv_eval_return_AUC"].iloc[0]) + EPS) - 1.0

    summary_df.to_csv(output_root / "g_sign_scaling_summary.csv", index=False)

    plus_row = summary_df[summary_df["method"] == "proposed_qp_plusG_trustreg"].iloc[0]
    minus_row = summary_df[summary_df["method"] == "proposed_qp_minusG_trustreg"].iloc[0]
    sign_row = summary_df[summary_df["method"] == "proposed_qp_sign_select_trustreg"].iloc[0]
    nogsafe_row = summary_df[summary_df["method"] == "proposed_qp_nog_safe_sign_select_trustreg"].iloc[0]

    if (
        safe_float(minus_row.get("improve_vs_nog_auc", math.nan), math.nan) >= 0.05
        and safe_float(minus_row.get("dominance_over_nog", math.nan), math.nan) >= 0.6
        and safe_float(minus_row.get("current_adv_eval_return_AUC", math.nan), math.nan)
        > safe_float(plus_row.get("current_adv_eval_return_AUC", math.nan), math.nan) * 1.05
    ):
        final_decision = "G_SIGN_BUG_LIKELY"
    elif (
        safe_float(sign_row.get("better_than_nog_fraction", math.nan), math.nan) >= 0.6
        and safe_float(sign_row.get("improve_vs_nog_auc", math.nan), math.nan) < 0.05
    ):
        final_decision = "G_SCALING_OR_ACCEPTANCE_ISSUE"
    elif (
        safe_float(sign_row.get("better_than_nog_fraction", math.nan), math.nan) < 0.5
        and safe_float(plus_row.get("better_than_nog_fraction", math.nan), math.nan) < 0.5
        and safe_float(minus_row.get("better_than_nog_fraction", math.nan), math.nan) < 0.5
    ):
        final_decision = "NO_USABLE_G_IN_THIS_REGIME"
    elif safe_float(nogsafe_row.get("chosen_noG_frac", math.nan), math.nan) >= 0.65:
        final_decision = "NOG_SAFE_QP_REDUCES_TO_NOG"
    elif any(
        (
            safe_float(row.get("improve_vs_nog_auc", math.nan), math.nan) >= 0.05
            and safe_float(row.get("improve_vs_egm_auc", math.nan), math.nan) >= 0.05
            and safe_float(row.get("dominance_over_nog", math.nan), math.nan) >= 0.6
            and safe_float(row.get("dominance_over_egm", math.nan), math.nan) >= 0.6
        )
        for _, row in summary_df[summary_df["method"].isin(["proposed_qp_sign_select_trustreg", "proposed_qp_nog_safe_sign_select_trustreg"])].iterrows()
    ):
        final_decision = "G_SIGN_REPAIR_QP_POSITIVE"
    else:
        final_decision = "NO_USABLE_G_IN_THIS_REGIME"

    plot_curves(curves, plots_root / "g_sign_scaling_curves.png")
    plot_step_diagnostics(summary_df, plots_root / "g_sign_scaling_step_diagnostics.png")

    report_lines = [
        "# G Sign / Scaling Repair Report",
        "",
        "- env: `HalfCheetah-v4`",
        "- scope: `full_policy`",
        "- alpha: `0.05`",
        "- shared_lr: `0.0003`",
        "- standard RARL unchanged: original reward, adversary reward = negative protagonist reward, additive `a_env = clip(u + alpha*w)`, no shaping.",
        "",
        "## Summary",
        "",
        f"- plusG better-than-noG fraction: `{safe_float(plus_row.get('better_than_nog_fraction', math.nan), math.nan):.3f}`",
        f"- minusG better-than-noG fraction: `{safe_float(minus_row.get('better_than_nog_fraction', math.nan), math.nan):.3f}`",
        f"- sign-select better-than-noG fraction: `{safe_float(sign_row.get('better_than_nog_fraction', math.nan), math.nan):.3f}`",
        f"- noG-safe chooses QP fraction: `{1.0 - safe_float(nogsafe_row.get('chosen_noG_frac', math.nan), math.nan):.3f}`",
        f"- noG-safe chooses noG fraction: `{safe_float(nogsafe_row.get('chosen_noG_frac', math.nan), math.nan):.3f}`",
        f"- sign-select plus-vs-minus preference: `plus={safe_float(sign_row.get('chosen_plus_frac', math.nan), math.nan):.3f}`, `minus={safe_float(sign_row.get('chosen_minus_frac', math.nan), math.nan):.3f}`",
        "",
        "## Return AUC",
        "",
        f"- noG current-adv AUC: `{safe_float(summary_df.loc[summary_df['method']=='proposed_nog_closed_trustreg','current_adv_eval_return_AUC'].iloc[0], math.nan):.3f}`",
        f"- plusG current-adv AUC: `{safe_float(plus_row.get('current_adv_eval_return_AUC', math.nan), math.nan):.3f}`",
        f"- minusG current-adv AUC: `{safe_float(minus_row.get('current_adv_eval_return_AUC', math.nan), math.nan):.3f}`",
        f"- sign-select current-adv AUC: `{safe_float(sign_row.get('current_adv_eval_return_AUC', math.nan), math.nan):.3f}`",
        f"- noG-safe current-adv AUC: `{safe_float(nogsafe_row.get('current_adv_eval_return_AUC', math.nan), math.nan):.3f}`",
        "",
        f"Final decision: `{final_decision}`",
    ]
    (output_root / "g_sign_scaling_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (output_root / "g_sign_scaling_decision.md").write_text(final_decision + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
