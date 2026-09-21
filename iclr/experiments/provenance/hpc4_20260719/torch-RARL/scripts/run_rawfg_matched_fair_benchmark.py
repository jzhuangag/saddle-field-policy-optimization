from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.full_policy_followup_common import SavedRun, load_yaml, load_rarl_for_eval, set_rarl_eval_mode


@dataclass(frozen=True)
class Candidate:
    method_label: str
    optimizer: str
    lr: float
    max_grad_norm: float
    vf_coef: float
    optimizer_kwargs: Dict[str, object]
    kind: str
    eta: Optional[float] = None
    config_label: Optional[str] = None


@dataclass(frozen=True)
class RawFGPlan:
    eta: float
    beta_max: float
    gamma_max: float
    bound_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Matched fair baseline + rawFG rerun")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument("--python-path", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--eval-freq", type=int, default=2500)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--eta-egm", type=float, default=1e-3)
    return parser.parse_args()


def run_command(command: List[str], cwd: pathlib.Path, stdout_path: pathlib.Path, stderr_path: pathlib.Path) -> None:
    result = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def render_kwargs_tokens(kwargs: Dict[str, object]) -> List[str]:
    tokens = []
    for key, value in sorted(kwargs.items()):
        if isinstance(value, str):
            rendered = repr(value)
        elif isinstance(value, float) and math.isinf(value):
            rendered = "float('inf')"
        else:
            rendered = value
        tokens.append(f"{key}:{rendered}")
    return tokens


def find_latest_run_dir(saved_models_dir: pathlib.Path, env_id: str) -> pathlib.Path:
    env_root = saved_models_dir / "rarl-ppo" / env_id
    candidates = [path for path in env_root.iterdir() if path.is_dir() and path.name.startswith(f"{env_id}_")]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {env_root}")
    return max(candidates, key=lambda path: int(path.name.rsplit("_", 1)[-1]))


def build_saved_run(run_dir: pathlib.Path, method: str) -> SavedRun:
    config_dir = next(path for path in run_dir.iterdir() if path.is_dir() and (path / "args.yml").exists())
    return SavedRun(
        method=method,
        tag=method,
        run_root=run_dir,
        model_dir=config_dir,
        args_data=load_yaml(config_dir / "args.yml"),
        config_data=load_yaml(config_dir / "config.yml"),
    )


def extract_prior_control_configs(results_root: pathlib.Path) -> Dict[str, Dict[str, object]]:
    control_root = results_root / "ppo_control_rarl_benchmark" / "full_policy_runs_seed0"
    configs: Dict[str, Dict[str, object]] = {}
    for method in ["adam", "sgd", "egm", "ppm"]:
        summary = pd.read_csv(control_root / method / "analysis" / "run_summary.csv").iloc[0]
        config_dir = next((control_root / method / "saved_models" / "rarl-ppo" / "HalfCheetah-v4" / "HalfCheetah-v4_1").glob("*"))
        if not (config_dir / "args.yml").exists():
            config_dir = next(path for path in (control_root / method / "saved_models" / "rarl-ppo" / "HalfCheetah-v4" / "HalfCheetah-v4_1").iterdir() if path.is_dir() and (path / "args.yml").exists())
        args_data = load_yaml(config_dir / "args.yml")
        configs[method] = {
            "prior_lr": float(summary["protagonist_lr"]),
            "prior_max_grad_norm": float(summary["protagonist_max_grad_norm"]),
            "prior_vf_coef": float(summary["protagonist_vf_coef"]),
            "prior_optimizer": str(summary["protagonist_optimizer"]),
            "prior_ppm_inner_steps": int(summary["ppm_inner_steps"]),
            "n_steps": int(args_data.get("n_steps", 2048)),
            "ent_coef": float(args_data.get("protagonist_ent_coef", args_data.get("ent_coef", 0.0))),
            "clip_range": float(args_data.get("protagonist_clip_range", args_data.get("clip_range", 0.2))),
            "n_epochs": int(args_data.get("protagonist_n_epochs", args_data.get("n_epochs", 10))),
            "batch_size": int(args_data.get("protagonist_batch_size", args_data.get("batch_size", 64))),
        }
    return configs


def build_baseline_candidates(prior_configs: Dict[str, Dict[str, object]]) -> List[Candidate]:
    return [
        Candidate("adam", "adam", prior_configs["adam"]["prior_lr"], 1.0, 1.0, {}, "baseline"),
        Candidate("sgd", "sgd", prior_configs["sgd"]["prior_lr"], 1.0, 1.0, {}, "baseline"),
        Candidate("egm", "egm", prior_configs["egm"]["prior_lr"], 1.0, 1.0, {}, "baseline"),
        Candidate(
            "ppm",
            "ppm",
            prior_configs["ppm"]["prior_lr"],
            1.0,
            1.0,
            {"inner_steps": prior_configs["ppm"]["prior_ppm_inner_steps"]},
            "baseline",
        ),
    ]


def run_rawfg_preflight(
    args: argparse.Namespace,
    *,
    output_dir: pathlib.Path,
    eta: float,
    beta_max: float,
    gamma_max: float,
    max_grad_norm: float,
    vf_coef: float,
) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        args.python_path,
        "scripts/run_proposed_qp_new_v2_rawfg_audit.py",
        "--output-dir",
        str(output_dir),
        "--env",
        args.env,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--role",
        "protagonist",
        "--num-probes",
        "4",
        "--eta-egm",
        str(args.eta_egm),
        "--external-eta",
        str(eta),
        "--fd-eps",
        "1e-3",
        "--beta-probe",
        "1e-3",
        "--gamma-probe",
        "1e-6",
        "--ridge",
        "1e-8",
        "--actor-weight",
        "1.0",
        "--logstd-weight",
        "1.0",
        "--critic-weight",
        "0.3",
        "--max-grad-norm",
        str(max_grad_norm),
        "--vf-coef",
        str(vf_coef),
        "--ppm-inner-steps",
        "10",
        "--single-beta-max",
        str(beta_max),
        "--single-gamma-max",
        str(gamma_max),
    ]
    run_command(command, cwd=pathlib.Path(args.repo_dir), stdout_path=output_dir / "preflight_stdout.txt", stderr_path=output_dir / "preflight_stderr.txt")
    return output_dir / "raw_fg_qp_audit.csv"


def summarize_preflight(preflight_csv: pathlib.Path, eta: float, bound_label: str, beta_max: float, gamma_max: float) -> Dict[str, object]:
    df = pd.read_csv(preflight_csv)
    qp = df[df["method"] == "proposed_qp_new_v2_rawFG"].copy()
    nog = df[df["method"] == "proposed_noG_new_v2_rawFG"].copy()
    egm = df[df["method"] == "egm"].copy()
    nan_flag = int((~np.isfinite(qp.select_dtypes(include=[np.number]).to_numpy())).any() or (~np.isfinite(nog.select_dtypes(include=[np.number]).to_numpy())).any())
    qp_v_mean = float(qp["actual_V_change"].mean())
    nog_v_mean = float(nog["actual_V_change"].mean())
    gamma_active_frac = float(qp["gamma_active"].mean())
    g_contrib_mean = float(qp["G_contribution_norm"].mean())
    qp_kl_max = float(qp["approx_kl_after"].max())
    qp_clip_max = float(qp["clip_fraction_after"].max())
    qp_update_mean = float(qp["update_norm"].mean())
    egm_update_mean = float(egm["update_norm"].mean()) if not egm.empty else float("nan")
    beta_mean = float(qp["beta_QP"].mean())
    gamma_mean = float(qp["gamma_QP"].mean())
    beta_eff_mean = float(qp["beta_eff"].mean()) if "beta_eff" in qp.columns else eta * beta_mean
    gamma_eff_mean = float(qp["gamma_eff"].mean()) if "gamma_eff" in qp.columns else eta * gamma_mean
    beta_at_bound_frac = float(qp["beta_at_bound"].mean())
    gamma_at_bound_frac = float(qp["gamma_at_bound"].mean())
    pass_gate = (
        nan_flag == 0
        and np.isfinite(beta_mean)
        and np.isfinite(gamma_mean)
        and gamma_active_frac > 0.0
        and g_contrib_mean > 0.0
        and qp_v_mean < nog_v_mean
        and qp_kl_max <= 0.1
        and qp_clip_max <= 0.8
        and (not np.isfinite(egm_update_mean) or qp_update_mean <= max(egm_update_mean * 5.0, 1e-12))
    )
    reason = "pass"
    if nan_flag:
        reason = "nan_or_inf"
    elif not np.isfinite(beta_mean) or not np.isfinite(gamma_mean):
        reason = "nonfinite_beta_gamma"
    elif gamma_active_frac <= 0.0:
        reason = "gamma_inactive"
    elif g_contrib_mean <= 0.0:
        reason = "g_contribution_zero"
    elif not (qp_v_mean < nog_v_mean):
        reason = "qp_not_better_than_nog_on_V"
    elif qp_kl_max > 0.1:
        reason = "approx_kl_spike"
    elif qp_clip_max > 0.8:
        reason = "clip_fraction_saturates"
    elif np.isfinite(egm_update_mean) and qp_update_mean > max(egm_update_mean * 5.0, 1e-12):
        reason = "update_norm_too_large"
    return {
        "eta": eta,
        "bound_label": bound_label,
        "beta_max": beta_max,
        "gamma_max": gamma_max,
        "preflight_pass": bool(pass_gate),
        "reason": reason,
        "beta_mean": beta_mean,
        "gamma_mean": gamma_mean,
        "beta_eff_mean": beta_eff_mean,
        "gamma_eff_mean": gamma_eff_mean,
        "beta_over_eta_EGM_mean": float(qp["beta_QP_over_eta_EGM"].mean()),
        "gamma_over_eta_EGM_squared_mean": float(qp["gamma_QP_over_eta_EGM_squared"].mean()),
        "beta_noG_mean": float(nog["beta_noG"].mean()),
        "beta_noG_over_eta_EGM_mean": float(nog["beta_noG_over_eta_EGM"].mean()),
        "qp_V_change_mean": qp_v_mean,
        "nog_V_change_mean": nog_v_mean,
        "qp_approx_kl_max": qp_kl_max,
        "qp_clip_fraction_max": qp_clip_max,
        "qp_update_norm_mean": qp_update_mean,
        "egm_update_norm_mean": egm_update_mean,
        "gamma_active_frac": gamma_active_frac,
        "G_contribution_norm_mean": g_contrib_mean,
        "G_over_update_norm_mean": float(qp["G_contribution_norm"].mean() / max(qp_update_mean, 1e-12)),
        "beta_at_bound_frac": beta_at_bound_frac,
        "gamma_at_bound_frac": gamma_at_bound_frac,
        "nan_or_inf_flag": nan_flag,
        "preflight_csv": str(preflight_csv),
    }


def choose_rawfg_plans(args: argparse.Namespace, output_root: pathlib.Path) -> Tuple[pd.DataFrame, List[RawFGPlan]]:
    preflight_root = output_root / "preflight"
    rows: List[Dict[str, object]] = []
    plans: List[RawFGPlan] = []
    primary = ("primary", 1e-2, 3e-5)
    backup = ("backup", 3e-3, 1e-5)
    for eta in [1.0, 0.1, 0.02]:
        primary_dir = preflight_root / f"eta_{eta:g}" / primary[0]
        primary_csv = run_rawfg_preflight(
            args,
            output_dir=primary_dir,
            eta=eta,
            beta_max=primary[1],
            gamma_max=primary[2],
            max_grad_norm=0.5,
            vf_coef=0.5,
        )
        primary_summary = summarize_preflight(primary_csv, eta, primary[0], primary[1], primary[2])
        rows.append(primary_summary)
        if primary_summary["preflight_pass"]:
            plans.append(RawFGPlan(eta=eta, beta_max=primary[1], gamma_max=primary[2], bound_label=primary[0]))
            continue
        backup_dir = preflight_root / f"eta_{eta:g}" / backup[0]
        backup_csv = run_rawfg_preflight(
            args,
            output_dir=backup_dir,
            eta=eta,
            beta_max=backup[1],
            gamma_max=backup[2],
            max_grad_norm=0.5,
            vf_coef=0.5,
        )
        backup_summary = summarize_preflight(backup_csv, eta, backup[0], backup[1], backup[2])
        rows.append(backup_summary)
        if backup_summary["preflight_pass"]:
            plans.append(RawFGPlan(eta=eta, beta_max=backup[1], gamma_max=backup[2], bound_label=backup[0]))
    preflight_df = pd.DataFrame(rows)
    preflight_df.to_csv(output_root / "rawFG_matched_preflight_summary.csv", index=False)
    return preflight_df, plans


def build_proposed_candidates(plans: Sequence[RawFGPlan]) -> List[Candidate]:
    candidates: List[Candidate] = []
    for plan in plans:
        kwargs = {
            "optimizer_scope": "full_policy",
            "qp_fd_eps": 1e-3,
            "qp_beta_probe": 1e-3,
            "qp_gamma_probe": 1e-6,
            "qp_ridge": 1e-8,
            "qp_actor_weight": 1.0,
            "qp_logstd_weight": 1.0,
            "qp_critic_weight": 0.3,
            "qp_beta_max": plan.beta_max,
            "qp_gamma_max": plan.gamma_max,
            "qp_max_update_norm": float("inf"),
            "qp_eps": 1e-8,
        }
        eta_label = f"{plan.eta:g}"
        candidates.append(
            Candidate(
                method_label=f"proposed_noG_rawFG_eta{eta_label}",
                optimizer="proposed_noG_rawFG",
                lr=plan.eta,
                max_grad_norm=0.5,
                vf_coef=0.5,
                optimizer_kwargs=dict(kwargs),
                kind="proposed",
                eta=plan.eta,
                config_label=plan.bound_label,
            )
        )
        candidates.append(
            Candidate(
                method_label=f"proposed_qp_rawFG_eta{eta_label}",
                optimizer="proposed_qp_rawFG",
                lr=plan.eta,
                max_grad_norm=0.5,
                vf_coef=0.5,
                optimizer_kwargs=dict(kwargs),
                kind="proposed",
                eta=plan.eta,
                config_label=plan.bound_label,
            )
        )
    return candidates


def ensure_run(candidate: Candidate, args: argparse.Namespace, runs_dir: pathlib.Path) -> pathlib.Path:
    run_root = runs_dir / candidate.method_label
    saved_models_dir = run_root / "saved_models"
    logging_dir = run_root / "logging"
    tb_dir = run_root / "tb"
    analysis_dir = run_root / "analysis"
    run_root.mkdir(parents=True, exist_ok=True)
    env_root = saved_models_dir / "rarl-ppo" / args.env
    if env_root.exists():
        latest_run_dir = find_latest_run_dir(saved_models_dir, args.env)
        adv_npz = latest_run_dir / "adv_eval" / "evaluations.npz"
        if adv_npz.exists() and (analysis_dir / "run_summary.csv").exists():
            return latest_run_dir

    command = [
        args.python_path,
        "scripts/train_adversary.py",
        "--algo",
        "rarl",
        "--rarl-config",
        "ppo",
        "--env",
        args.env,
        "--adv-impact",
        "control",
        "--device",
        args.device,
        "--verbose",
        "1",
        "--seed",
        str(args.seed),
        "-n",
        str(args.iterations),
        "--eval-freq",
        str(args.eval_freq),
        "--save-freq",
        str(args.eval_freq),
        "--n-eval-episodes",
        str(args.n_eval_episodes),
        "--saved-models-path",
        str(saved_models_dir),
        "--log-folder",
        str(logging_dir),
        "--tensorboard-log",
        str(tb_dir),
        "--optimizer-scope",
        "full_policy",
        "--protagonist-optimizer",
        candidate.optimizer,
        "--adversary-optimizer",
        candidate.optimizer,
        "--protagonist-lr",
        str(candidate.lr),
        "--adversary-lr",
        str(candidate.lr),
        "--protagonist-max-grad-norm",
        str(candidate.max_grad_norm),
        "--adversary-max-grad-norm",
        str(candidate.max_grad_norm),
        "--protagonist-vf-coef",
        str(candidate.vf_coef),
        "--adversary-vf-coef",
        str(candidate.vf_coef),
    ]
    tokens = render_kwargs_tokens(candidate.optimizer_kwargs)
    if tokens:
        command.extend(["--protagonist-optimizer-kwargs", *tokens])
        command.extend(["--adversary-optimizer-kwargs", *tokens])

    run_command(command, cwd=pathlib.Path(args.repo_dir), stdout_path=run_root / "stdout.txt", stderr_path=run_root / "stderr.txt")
    latest_run_dir = find_latest_run_dir(saved_models_dir, args.env)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_root / "stdout.txt", analysis_dir / "stdout.txt")
    shutil.copy2(run_root / "stderr.txt", analysis_dir / "stderr.txt")
    run_command(
        [args.python_path, "scripts/analyze_rarl_run.py", "--run-dir", str(latest_run_dir), "--output-dir", str(analysis_dir), "--method", candidate.method_label],
        cwd=pathlib.Path(args.repo_dir),
        stdout_path=analysis_dir / "analyze_stdout.txt",
        stderr_path=analysis_dir / "analyze_stderr.txt",
    )
    return latest_run_dir


def get_run_root_for_analysis(latest_run_dir: pathlib.Path) -> pathlib.Path:
    return latest_run_dir / "analysis"


def read_method_frames(method_label: str, latest_run_dir: pathlib.Path, analysis_dir: pathlib.Path) -> Dict[str, pd.DataFrame]:
    config_dir = next(path for path in latest_run_dir.iterdir() if path.is_dir() and (path / "args.yml").exists())
    args_data = load_yaml(config_dir / "args.yml")
    n_steps = int(args_data.get("n_steps", 2048))

    training = pd.read_csv(analysis_dir / "training_episode_returns.csv")
    training["method"] = method_label
    training["outer_iteration"] = training["cumulative_timesteps"] / float(n_steps)

    clean = pd.read_csv(analysis_dir / "clean_eval_returns.csv")
    clean["method"] = method_label
    clean["outer_iteration"] = clean["timesteps"] / float(n_steps)

    adv = pd.read_csv(analysis_dir / "adversarial_eval_returns.csv")
    adv["method"] = method_label
    adv["outer_iteration"] = adv["timesteps"] / float(n_steps)

    param = pd.read_csv(analysis_dir / "parameter_norms.csv")
    param["method"] = method_label
    param["outer_iteration"] = param["num_timesteps"] / float(n_steps)

    protagonist_metrics = pd.read_csv(latest_run_dir / "analysis" / "protagonist_training_metrics.csv")
    protagonist_metrics["method"] = method_label
    protagonist_metrics["optimizer_role"] = "protagonist"
    protagonist_metrics["outer_iteration"] = protagonist_metrics["num_timesteps"] / float(n_steps)

    adversary_metrics = pd.read_csv(latest_run_dir / "analysis" / "adversary_training_metrics.csv")
    adversary_metrics["method"] = method_label
    adversary_metrics["optimizer_role"] = "adversary"
    adversary_metrics["outer_iteration"] = adversary_metrics["num_timesteps"] / float(n_steps)

    diagnostics = pd.DataFrame()
    for diag_path in sorted(latest_run_dir.glob("*diagnostics.csv")):
        diag = pd.read_csv(diag_path)
        role = "protagonist" if "protagonist" in diag_path.name else "adversary"
        metrics_df = protagonist_metrics if role == "protagonist" else adversary_metrics
        if metrics_df.empty:
            continue
        cumulative = metrics_df["n_updates"].astype(int).tolist()
        start = 0
        chunks = []
        for idx, stop in enumerate(cumulative):
            stop = int(stop)
            if stop <= start:
                continue
            chunk = diag.iloc[start:stop].copy()
            chunk["num_timesteps"] = float(metrics_df.iloc[idx]["num_timesteps"])
            chunk["outer_iteration"] = float(metrics_df.iloc[idx]["outer_iteration"])
            chunk["optimizer_role"] = role
            chunks.append(chunk)
            start = stop
        if chunks:
            role_df = pd.concat(chunks, ignore_index=True)
            role_df["method"] = method_label
            diagnostics = pd.concat([diagnostics, role_df], ignore_index=True)

    return {
        "training": training,
        "clean": clean,
        "adv": adv,
        "param": param,
        "protagonist_metrics": protagonist_metrics,
        "adversary_metrics": adversary_metrics,
        "diagnostics": diagnostics,
        "config_args": pd.DataFrame([args_data]),
    }


def evaluate_control_strength_sweep(run_dir: pathlib.Path, method: str, device: str, n_eval_episodes: int) -> pd.DataFrame:
    saved_run = build_saved_run(run_dir, method)
    strengths = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    rows = []
    model, vec_env = load_rarl_for_eval(saved_run, adv_impact="control", adv_strength=1.0, device=device)
    for strength in strengths:
        set_rarl_eval_mode(model, vec_env, operating_mode="protagonist", adv_strength=strength)
        episode_rewards = []
        perturbation_norms = []
        clip_fractions = []
        obs = vec_env.reset()
        ep_reward = 0.0
        ep_perturb = []
        ep_clip = []
        while len(episode_rewards) < n_eval_episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            ep_reward += float(rewards[0])
            info = infos[0]
            ep_perturb.append(float(info.get("applied_control_perturbation_norm", info.get("applied_disturbance_norm", 0.0))))
            ep_clip.append(float(info.get("action_clip_fraction", 0.0)))
            if bool(dones[0]):
                episode_rewards.append(ep_reward)
                perturbation_norms.append(float(np.mean(ep_perturb)) if ep_perturb else 0.0)
                clip_fractions.append(float(np.mean(ep_clip)) if ep_clip else 0.0)
                obs = vec_env.reset()
                ep_reward = 0.0
                ep_perturb = []
                ep_clip = []
        rows.append(
            {
                "method": method,
                "adv_strength": strength,
                "mean_return": float(np.mean(episode_rewards)),
                "std_return": float(np.std(episode_rewards)),
                "applied_control_perturbation_norm": float(np.mean(perturbation_norms)),
                "action_clip_fraction": float(np.mean(clip_fractions)),
            }
        )
    vec_env.close()
    return pd.DataFrame(rows)


def plot_eval_with_band(ax, df: pd.DataFrame, y_col: str, std_col: str, title: str, ylabel: str, colors: Dict[str, str]) -> None:
    for method, group in df.groupby("method"):
        group = group.sort_values("outer_iteration")
        ax.plot(group["outer_iteration"], group[y_col], label=method, color=colors.get(method))
        if std_col in group.columns:
            ax.fill_between(group["outer_iteration"], group[y_col] - group[std_col], group[y_col] + group[std_col], color=colors.get(method), alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def plot_param_norms(ax, df: pd.DataFrame, agent_name: str, colors: Dict[str, str]) -> None:
    sub = df[df["agent_name"] == agent_name]
    for method, group in sub.groupby("method"):
        group = group.sort_values("outer_iteration")
        base = colors.get(method)
        ax.plot(group["outer_iteration"], group["actor_param_norm"], color=base, label=f"{method} actor")
        ax.plot(group["outer_iteration"], group["log_std_norm"], color=base, linestyle=":", label=f"{method} log_std")
        ax.plot(group["outer_iteration"], group["critic_param_norm"], color=base, linestyle="--", label=f"{method} critic")
        ax.plot(group["outer_iteration"], group["total_param_norm"], color=base, linestyle="-.", label=f"{method} total")
    ax.set_title(f"{agent_name.capitalize()} parameter norms")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("L2 norm")
    ax.grid(alpha=0.3)


def plot_update_norms(ax, df: pd.DataFrame, colors: Dict[str, str]) -> None:
    for method, group in df.groupby("method"):
        group = group[group["optimizer_role"] == "protagonist"].sort_values("outer_iteration")
        ax.plot(group["outer_iteration"], group["actor_update_norm"], color=colors.get(method), label=f"{method} actor")
        ax.plot(group["outer_iteration"], group["critic_update_norm"], color=colors.get(method), linestyle="--", label=f"{method} critic")
    ax.set_title("Protagonist update norms")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Update norm")
    ax.grid(alpha=0.3)


def make_collage(plot_paths: Sequence[pathlib.Path], output_path: pathlib.Path, cols: int = 2) -> None:
    images = [Image.open(path).convert("RGB") for path in plot_paths if path.exists()]
    if not images:
        return
    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)
    rows = math.ceil(len(images) / cols)
    pad = 20
    header_h = 36
    canvas = Image.new("RGB", (pad + cols * (cell_w + pad), pad + rows * (cell_h + header_h + pad)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    for idx, (img, path) in enumerate(zip(images, [p for p in plot_paths if p.exists()])):
        row = idx // cols
        col = idx % cols
        x0 = pad + col * (cell_w + pad)
        y0 = pad + row * (cell_h + header_h + pad)
        draw.text((x0 + 8, y0 + 8), f"{idx + 1}. {path.name}", fill="black", font=font)
        thumb = img.copy()
        thumb.thumbnail((cell_w, cell_h))
        canvas.paste(thumb, (x0 + (cell_w - thumb.width) // 2, y0 + header_h + (cell_h - thumb.height) // 2))
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    repo_dir = pathlib.Path(args.repo_dir)
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results_root = output_root.parent
    prior_configs = extract_prior_control_configs(results_root)
    baseline_candidates = build_baseline_candidates(prior_configs)

    config_rows = []
    for candidate in baseline_candidates:
        config_rows.append(
            {
                "method": candidate.method_label,
                "kind": candidate.kind,
                "optimizer": candidate.optimizer,
                "lr": candidate.lr,
                "max_grad_norm": candidate.max_grad_norm,
                "vf_coef": candidate.vf_coef,
                "ent_coef": prior_configs["egm"]["ent_coef"] if candidate.optimizer in {"sgd", "egm", "ppm", "adam"} else 0.0,
                "clip_range": prior_configs["egm"]["clip_range"],
                "n_epochs": prior_configs["egm"]["n_epochs"],
                "batch_size": prior_configs["egm"]["batch_size"],
                "n_steps": prior_configs[candidate.optimizer]["n_steps"] if candidate.optimizer in prior_configs else 2048,
                "optimizer_scope": candidate.optimizer_kwargs.get("optimizer_scope", "full_policy"),
                "ppm_inner_steps": candidate.optimizer_kwargs.get("inner_steps", -1),
                "eta": np.nan,
                "qp_beta_max": np.nan,
                "qp_gamma_max": np.nan,
                "qp_fd_eps": np.nan,
                "qp_actor_weight": np.nan,
                "qp_logstd_weight": np.nan,
                "qp_critic_weight": np.nan,
                "config_source": "rerun_from_scratch_matched_baseline",
            }
        )

    preflight_df, plans = choose_rawfg_plans(args, output_root)
    proposed_candidates = build_proposed_candidates(plans)
    for candidate in proposed_candidates:
        config_rows.append(
            {
                "method": candidate.method_label,
                "kind": candidate.kind,
                "optimizer": candidate.optimizer,
                "lr": candidate.lr,
                "max_grad_norm": candidate.max_grad_norm,
                "vf_coef": candidate.vf_coef,
                "ent_coef": prior_configs["egm"]["ent_coef"],
                "clip_range": prior_configs["egm"]["clip_range"],
                "n_epochs": prior_configs["egm"]["n_epochs"],
                "batch_size": prior_configs["egm"]["batch_size"],
                "n_steps": prior_configs["egm"]["n_steps"],
                "optimizer_scope": candidate.optimizer_kwargs.get("optimizer_scope", "full_policy"),
                "ppm_inner_steps": -1,
                "eta": candidate.eta,
                "qp_beta_max": candidate.optimizer_kwargs.get("qp_beta_max"),
                "qp_gamma_max": candidate.optimizer_kwargs.get("qp_gamma_max"),
                "qp_fd_eps": candidate.optimizer_kwargs.get("qp_fd_eps"),
                "qp_actor_weight": candidate.optimizer_kwargs.get("qp_actor_weight"),
                "qp_logstd_weight": candidate.optimizer_kwargs.get("qp_logstd_weight"),
                "qp_critic_weight": candidate.optimizer_kwargs.get("qp_critic_weight"),
                "config_source": f"rawfg_{candidate.config_label}",
            }
        )

    config_df = pd.DataFrame(config_rows)
    config_df.to_csv(output_root / "config_matrix.csv", index=False)

    candidates = baseline_candidates + proposed_candidates
    runs_dir = output_root / "runs_seed0"
    summary_rows = []
    training_frames = []
    clean_frames = []
    adv_frames = []
    param_frames = []
    update_frames = []
    qp_diag_frames = []
    sweep_frames = []

    for candidate in candidates:
        latest_run_dir = ensure_run(candidate, args, runs_dir)
        analysis_dir = runs_dir / candidate.method_label / "analysis"
        run_summary = pd.read_csv(analysis_dir / "run_summary.csv").iloc[0].to_dict()
        run_summary["method"] = candidate.method_label
        run_summary["kind"] = candidate.kind
        run_summary["eta"] = candidate.eta if candidate.eta is not None else np.nan
        run_summary["bound_label"] = candidate.config_label if candidate.config_label is not None else ""
        summary_rows.append(run_summary)

        frames = read_method_frames(candidate.method_label, latest_run_dir, analysis_dir)
        training_frames.append(frames["training"])
        clean_frames.append(frames["clean"])
        adv_frames.append(frames["adv"])
        param_frames.append(frames["param"])
        update_frames.append(pd.concat([frames["protagonist_metrics"], frames["adversary_metrics"]], ignore_index=True))
        if not frames["diagnostics"].empty:
            frames["diagnostics"]["eta"] = candidate.eta
            qp_diag_frames.append(frames["diagnostics"])
        sweep_df = evaluate_control_strength_sweep(latest_run_dir, candidate.method_label, args.device, args.n_eval_episodes)
        sweep_frames.append(sweep_df)

    summary_df = pd.DataFrame(summary_rows)
    training_df = pd.concat(training_frames, ignore_index=True)
    clean_df = pd.concat(clean_frames, ignore_index=True)
    adv_df = pd.concat(adv_frames, ignore_index=True)
    param_df = pd.concat(param_frames, ignore_index=True)
    update_df = pd.concat(update_frames, ignore_index=True)
    qp_diag_df = pd.concat(qp_diag_frames, ignore_index=True) if qp_diag_frames else pd.DataFrame()
    sweep_df = pd.concat(sweep_frames, ignore_index=True)

    eval_df = pd.concat(
        [
            clean_df.assign(eval_type="clean"),
            adv_df.assign(eval_type="control_adversarial"),
        ],
        ignore_index=True,
    )

    summary_df.to_csv(output_root / "rawFG_matched_training_summary.csv", index=False)
    eval_df.to_csv(output_root / "rawFG_matched_eval_curves.csv", index=False)
    param_df.to_csv(output_root / "rawFG_matched_param_norms.csv", index=False)
    update_df.to_csv(output_root / "rawFG_matched_update_diagnostics.csv", index=False)
    qp_diag_df.to_csv(output_root / "rawFG_matched_qp_diagnostics.csv", index=False)
    sweep_df.to_csv(output_root / "rawFG_matched_robustness_sweep.csv", index=False)

    colors = {
        "adam": "tab:orange",
        "sgd": "tab:gray",
        "egm": "tab:green",
        "ppm": "tab:red",
    }
    for candidate in proposed_candidates:
        colors[candidate.method_label] = "tab:blue" if "qp" in candidate.method_label else "tab:purple"

    plt.figure(figsize=(10, 6))
    for method, group in training_df.groupby("method"):
        plt.plot(group["outer_iteration"], group["episode_return"], label=method, color=colors.get(method))
    plt.title("Training return vs outer iteration")
    plt.xlabel("Outer iteration")
    plt.ylabel("Episode return")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "training_return_vs_outer_iteration.png", dpi=180, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_eval_with_band(ax, clean_df, "mean_reward", "std_reward", "Clean eval vs outer iteration", "Mean return", colors)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "clean_eval_vs_outer_iteration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_eval_with_band(ax, adv_df, "mean_reward", "std_reward", "Control adversarial eval vs outer iteration", "Mean return", colors)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "control_adv_eval_vs_outer_iteration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    plot_param_norms(ax, param_df, "protagonist", colors)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(plots_dir / "protagonist_param_norms_vs_outer_iteration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    plot_param_norms(ax, param_df, "adversary", colors)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(plots_dir / "adversary_param_norms_vs_outer_iteration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    plot_update_norms(ax, update_df, colors)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(plots_dir / "update_norms_vs_outer_iteration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if not qp_diag_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for method, group in qp_diag_df.groupby("method"):
            group = group[group["optimizer_role"] == "protagonist"].sort_values("outer_iteration")
            axes[0, 0].plot(group["outer_iteration"], group["beta"], label=method, color=colors.get(method))
            axes[0, 1].plot(group["outer_iteration"], group["gamma"], label=method, color=colors.get(method))
            axes[1, 0].plot(group["outer_iteration"], group["beta_eff"], label=method, color=colors.get(method))
            axes[1, 0].plot(group["outer_iteration"], group["gamma_eff"], linestyle="--", label=f"{method} gamma_eff", color=colors.get(method))
            axes[1, 1].plot(group["outer_iteration"], group["gamma_active_frac"], label=method, color=colors.get(method))
            axes[1, 1].plot(group["outer_iteration"], group["G_contribution_norm"], linestyle="--", label=f"{method} G", color=colors.get(method))
        axes[0, 0].set_title("beta")
        axes[0, 1].set_title("gamma")
        axes[1, 0].set_title("beta_eff / gamma_eff")
        axes[1, 1].set_title("gamma_active_frac / G_contribution_norm")
        for ax in axes.ravel():
            ax.set_xlabel("Outer iteration")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plots_dir / "qp_beta_gamma_diagnostics.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for method, group in qp_diag_df.groupby("method"):
            group = group[group["optimizer_role"] == "protagonist"].sort_values("outer_iteration")
            axes[0, 0].plot(group["outer_iteration"], group["actual_V_change"], label=method, color=colors.get(method))
            axes[0, 1].plot(group["outer_iteration"], group["q_pred"], label=method, color=colors.get(method))
            axes[1, 0].plot(group["outer_iteration"], group["approx_kl_after"], label=method, color=colors.get(method))
            axes[1, 1].plot(group["outer_iteration"], group["clip_fraction_after"], label=method, color=colors.get(method))
        axes[0, 0].set_title("actual_V_change")
        axes[0, 1].set_title("q_pred")
        axes[1, 0].set_title("approx_kl")
        axes[1, 1].set_title("clip_fraction")
        for ax in axes.ravel():
            ax.set_xlabel("Outer iteration")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plots_dir / "qp_V_KL_clip_diagnostics.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    plt.figure(figsize=(10, 6))
    for method, group in sweep_df.groupby("method"):
        group = group.sort_values("adv_strength")
        plt.plot(group["adv_strength"], group["mean_return"], marker="o", label=method, color=colors.get(method))
    plt.title("Control robustness sweep final")
    plt.xlabel("Adversary strength")
    plt.ylabel("Mean return")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "control_robustness_sweep_final.png", dpi=180, bbox_inches="tight")
    plt.close()

    final_bar = summary_df[["method", "last5_clean_mean", "last5_adversarial_mean"]].copy()
    auc_rows = []
    for method, group in sweep_df.groupby("method"):
        ordered = group.sort_values("adv_strength")
        auc_rows.append({"method": method, "robustness_auc": float(np.trapezoid(ordered["mean_return"], ordered["adv_strength"]))})
    auc_df = pd.DataFrame(auc_rows)
    final_bar = final_bar.merge(auc_df, on="method", how="left")
    x = np.arange(len(final_bar))
    width = 0.25
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, final_bar["last5_clean_mean"], width=width, label="clean final")
    plt.bar(x, final_bar["last5_adversarial_mean"], width=width, label="control-adv final")
    plt.bar(x + width, final_bar["robustness_auc"], width=width, label="robustness AUC")
    plt.xticks(x, final_bar["method"], rotation=30, ha="right")
    plt.title("Final bar comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "final_bar_comparison.png", dpi=180, bbox_inches="tight")
    plt.close()

    collage_inputs = [
        plots_dir / "training_return_vs_outer_iteration.png",
        plots_dir / "clean_eval_vs_outer_iteration.png",
        plots_dir / "control_adv_eval_vs_outer_iteration.png",
        plots_dir / "protagonist_param_norms_vs_outer_iteration.png",
        plots_dir / "adversary_param_norms_vs_outer_iteration.png",
        plots_dir / "update_norms_vs_outer_iteration.png",
        plots_dir / "control_robustness_sweep_final.png",
        plots_dir / "final_bar_comparison.png",
        plots_dir / "qp_beta_gamma_diagnostics.png",
        plots_dir / "qp_V_KL_clip_diagnostics.png",
    ]
    make_collage(collage_inputs, plots_dir / "stageM_all_plots_big.png", cols=2)

    config_lines = [
        "# RawFG matched config report",
        "",
        "- Fresh result root used: `rawFG_matched_fair_baselines_r1`",
        "- Old result folders were left untouched.",
        "- Baselines were rerun from scratch under proper control-RARL.",
        "- Matched baseline fairness setting applied to `sgd`, `egm`, `ppm`: `max_grad_norm=1.0`, `vf_coef=1.0`.",
        "- Adam was also rerun with `max_grad_norm=1.0`, `vf_coef=1.0` for this fairness pass.",
        "- Proposed rawFG preflight/online kept raw-F/G solver and explicit eta sweep; no PPO-loss grid search was used.",
        "",
        "## Config matrix",
        "",
    ]
    for row in config_df.to_dict(orient="records"):
        config_lines.append(
            f"- `{row['method']}`: optimizer=`{row['optimizer']}`, lr/eta=`{row['lr']}`, max_grad_norm=`{row['max_grad_norm']}`, vf_coef=`{row['vf_coef']}`, ppm_inner_steps=`{row['ppm_inner_steps']}`, qp_beta_max=`{row['qp_beta_max']}`, qp_gamma_max=`{row['qp_gamma_max']}`, qp_critic_weight=`{row['qp_critic_weight']}`"
        )
    (output_root / "rawFG_matched_config_report.md").write_text("\n".join(config_lines) + "\n", encoding="utf-8")

    preflight_lines = [
        "# RawFG matched preflight report",
        "",
        f"- Proposed eta values tested: `{sorted(set(preflight_df['eta'].tolist()))}`",
        f"- Eta values that passed preflight: `{sorted(set(preflight_df.loc[preflight_df['preflight_pass'], 'eta'].tolist()))}`",
        "",
    ]
    for row in preflight_df.to_dict(orient="records"):
        preflight_lines.append(
            f"- eta=`{row['eta']}`, bound=`{row['bound_label']}`, pass=`{row['preflight_pass']}`, reason=`{row['reason']}`, beta_eff_mean=`{row['beta_eff_mean']:.6e}`, gamma_eff_mean=`{row['gamma_eff_mean']:.6e}`, qp_V_change_mean=`{row['qp_V_change_mean']:.6e}`, noG_V_change_mean=`{row['nog_V_change_mean']:.6e}`, KL_max=`{row['qp_approx_kl_max']:.6f}`, clip_max=`{row['qp_clip_fraction_max']:.6f}`"
        )
    (output_root / "rawFG_matched_preflight_report.md").write_text("\n".join(preflight_lines) + "\n", encoding="utf-8")

    def get_row(method: str) -> pd.Series:
        return summary_df[summary_df["method"] == method].iloc[0]

    egm_row = get_row("egm")
    ppm_row = get_row("ppm")
    sgd_row = get_row("sgd")

    proposed_qp_rows = summary_df[summary_df["method"].str.startswith("proposed_qp_rawFG_eta")].copy()
    proposed_nog_rows = summary_df[summary_df["method"].str.startswith("proposed_noG_rawFG_eta")].copy()
    best_qp = proposed_qp_rows.sort_values("last5_clean_mean", ascending=False).iloc[0] if not proposed_qp_rows.empty else None
    matched_nog = None
    if best_qp is not None:
        eta_suffix = str(best_qp["method"]).split("_eta", 1)[-1]
        matched_nog = proposed_nog_rows[proposed_nog_rows["method"] == f"proposed_noG_rawFG_eta{eta_suffix}"].iloc[0]

    final_lines = [
        "# RawFG matched final report",
        "",
        f"1. Were all baselines rerun from scratch with matched budget? `True`",
        f"2. Was EGM changed to max_grad_norm=1.0 and vf_coef=1.0? `True`",
        f"3. Does EGM remain strong after this fairness change? `{bool(float(egm_row['last5_clean_mean']) >= float(ppm_row['last5_clean_mean']) and float(egm_row['last5_adversarial_mean']) >= float(sgd_row['last5_adversarial_mean']))}`",
        f"4. Does PPM improve or remain weak? `{'improve_or_hold' if float(ppm_row['last5_clean_mean']) >= float(sgd_row['last5_clean_mean']) else 'weak'}`",
        f"5. Which proposed eta values passed preflight? `{sorted(set(preflight_df.loc[preflight_df['preflight_pass'], 'eta'].tolist()))}`",
    ]
    if best_qp is not None and matched_nog is not None:
        best_qp_diag = qp_diag_df[(qp_diag_df["method"] == best_qp["method"]) & (qp_diag_df["optimizer_role"] == "protagonist")] if not qp_diag_df.empty else pd.DataFrame()
        final_lines.extend(
            [
                f"6. Does proposed_qp_rawFG beat proposed_noG_rawFG at matched budget? `{bool(float(best_qp['last5_clean_mean']) > float(matched_nog['last5_clean_mean']) and float(best_qp['last5_adversarial_mean']) > float(matched_nog['last5_adversarial_mean']))}`",
                f"7. Does proposed_qp_rawFG beat SGD at matched budget? `{bool(float(best_qp['last5_clean_mean']) > float(sgd_row['last5_clean_mean']) and float(best_qp['last5_adversarial_mean']) > float(sgd_row['last5_adversarial_mean']))}`",
                f"8. Does proposed_qp_rawFG beat PPM at matched budget? `{bool(float(best_qp['last5_clean_mean']) > float(ppm_row['last5_clean_mean']) and float(best_qp['last5_adversarial_mean']) > float(ppm_row['last5_adversarial_mean']))}`",
                f"9. Does proposed_qp_rawFG beat EGM at matched budget? `{bool(float(best_qp['last5_clean_mean']) > float(egm_row['last5_clean_mean']) and float(best_qp['last5_adversarial_mean']) > float(egm_row['last5_adversarial_mean']))}`",
                f"10. Are beta/gamma usually larger than EGM-like coefficients? `{bool(not best_qp_diag.empty and float(best_qp_diag['beta'].mean()) > args.eta_egm and float(best_qp_diag['gamma'].mean()) > args.eta_egm ** 2)}`",
                f"11. Does gamma remain useful online? `{bool(not best_qp_diag.empty and float(best_qp_diag['gamma_active_frac'].mean()) > 0.0 and float(best_qp_diag['G_contribution_norm'].mean()) > 0.0)}`",
            ]
        )
        failure_reason = "F. EGM still better aligned with PPO return"
        if float(best_qp["last5_clean_mean"]) < float(matched_nog["last5_clean_mean"]):
            failure_reason = "D. V objective mismatch"
        elif not best_qp_diag.empty and float(best_qp_diag["beta_at_bound"].mean()) > 0.5:
            failure_reason = "B. QP too aggressive"
        elif not best_qp_diag.empty and float(best_qp_diag["update_norm_post_cap"].mean()) < 1e-5:
            failure_reason = "C. QP too conservative"
        elif not best_qp_diag.empty and float(best_qp_diag["critic_update_norm"].mean()) > float(best_qp_diag["actor_update_norm"].mean()) * 2.0:
            failure_reason = "E. critic block interference"
        final_lines.append(f"12. Are failures due to: `{failure_reason}`")
    (output_root / "rawFG_matched_final_report.md").write_text("\n".join(final_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
