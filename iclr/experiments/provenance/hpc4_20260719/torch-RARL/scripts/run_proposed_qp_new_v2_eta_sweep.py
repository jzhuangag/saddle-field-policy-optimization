from __future__ import annotations

import argparse
import math
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Candidate:
    method: str
    optimizer: str
    lr: float
    max_grad_norm: float
    vf_coef: float
    optimizer_kwargs: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage P7 eta sweep for proposed_qp_new_v2 / proposed_noG_new_v2")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument("--python-path", type=str, required=True)
    parser.add_argument("--baseline-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=2500)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--eta-list", type=str, default="1e-3,1e-2,1e-1,1")
    parser.add_argument("--beta-max", type=float, default=0.3)
    parser.add_argument("--gamma-max", type=float, default=0.3)
    parser.add_argument("--horizon-timesteps", type=int, default=10000)
    return parser.parse_args()


def parse_eta_list(raw: str) -> List[float]:
    etas = []
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        etas.append(float(token))
    if not etas:
        raise ValueError("eta list is empty")
    return etas


def eta_label(eta: float) -> str:
    text = f"{eta:.0e}" if eta != 1.0 else "1"
    return text.replace("+", "").replace(".", "p").replace("-", "m")


def render_kwargs_tokens(kwargs: Dict[str, object]) -> List[str]:
    tokens: List[str] = []
    for key, value in sorted(kwargs.items()):
        if isinstance(value, float) and not math.isfinite(value):
            continue
        rendered = repr(value) if isinstance(value, str) else value
        tokens.append(f"{key}:{rendered}")
    return tokens


def run_command(command: List[str], cwd: pathlib.Path, stdout_path: pathlib.Path, stderr_path: pathlib.Path) -> None:
    result = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def find_latest_run_dir(saved_models_dir: pathlib.Path, env_id: str) -> pathlib.Path:
    env_root = saved_models_dir / "rarl-ppo" / env_id
    candidates = [path for path in env_root.iterdir() if path.is_dir() and path.name.startswith(f"{env_id}_")]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {env_root}")
    return max(candidates, key=lambda path: int(path.name.rsplit("_", 1)[-1]))


def load_curve_frame(path: pathlib.Path, method: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["method"] = method
    return frame


def summarize_short_horizon(
    *,
    method: str,
    training_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    adv_df: pd.DataFrame,
    horizon_timesteps: int,
) -> Dict[str, object]:
    training_cut = training_df.loc[training_df["cumulative_timesteps"] <= horizon_timesteps].copy()
    clean_cut = clean_df.loc[clean_df["timesteps"] <= horizon_timesteps].copy()
    adv_cut = adv_df.loc[adv_df["timesteps"] <= horizon_timesteps].copy()

    def safe_last(frame: pd.DataFrame, column: str) -> float:
        if frame.empty:
            return float("nan")
        return float(frame.iloc[-1][column])

    def safe_last_mean(frame: pd.DataFrame, column: str, window: int = 5) -> float:
        if frame.empty:
            return float("nan")
        return float(frame[column].tail(window).mean())

    return {
        "method": method,
        "horizon_timesteps": horizon_timesteps,
        "final_training_return": safe_last(training_cut, "episode_return"),
        "last5_training_mean": safe_last_mean(training_cut, "episode_return"),
        "final_clean_return": safe_last(clean_cut, "mean_reward"),
        "last5_clean_mean": safe_last_mean(clean_cut, "mean_reward"),
        "final_adversarial_return": safe_last(adv_cut, "mean_reward"),
        "last5_adversarial_mean": safe_last_mean(adv_cut, "mean_reward"),
        "num_training_points": int(len(training_cut)),
        "num_clean_points": int(len(clean_cut)),
        "num_adv_points": int(len(adv_cut)),
    }


def load_frozen_baselines(baseline_root: pathlib.Path, horizon_timesteps: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for method in ["adam", "sgd", "egm", "ppm"]:
        analysis_dir = baseline_root / method / "analysis"
        training_df = pd.read_csv(analysis_dir / "training_episode_returns.csv")
        clean_df = pd.read_csv(analysis_dir / "clean_eval_returns.csv")
        adv_df = pd.read_csv(analysis_dir / "adversarial_eval_returns.csv")
        row = summarize_short_horizon(
            method=method,
            training_df=training_df,
            clean_df=clean_df,
            adv_df=adv_df,
            horizon_timesteps=horizon_timesteps,
        )
        row.update(
            {
                "eta": np.nan,
                "preflight_pass": True,
                "online_ran": False,
                "optimizer": method,
                "notes": "frozen_baseline_reference",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run_preflight(
    *,
    args: argparse.Namespace,
    output_dir: pathlib.Path,
    eta: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        args.python_path,
        "scripts/run_proposed_qp_new_v2_preflight.py",
        "--output-dir",
        str(output_dir),
        "--env",
        args.env,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--proposed-lr",
        str(eta),
        "--proposed-beta-max",
        str(args.beta_max),
        "--proposed-gamma-max",
        str(args.gamma_max),
    ]
    run_command(command, cwd=pathlib.Path(args.repo_dir), stdout_path=output_dir / "preflight_stdout.txt", stderr_path=output_dir / "preflight_stderr.txt")


def evaluate_preflight_gate(preflight_csv: pathlib.Path, method: str) -> Dict[str, object]:
    frame = pd.read_csv(preflight_csv)
    method_frame = frame.loc[frame["method"] == method].copy()
    if method_frame.empty:
        return {"pass": False, "reason": f"missing_method:{method}"}

    numeric_checks = [
        "beta",
        "eta",
        "beta_eff",
        "update_norm_post_cap",
        "actual_V_change",
        "approx_kl_after",
        "clip_fraction_after",
    ]
    if method == "proposed_qp_new_v2":
        numeric_checks.extend(["gamma", "gamma_eff", "G_contribution_norm", "G_over_update_norm"])
    for column in numeric_checks:
        if not np.isfinite(method_frame[column].astype(float)).all():
            return {"pass": False, "reason": f"nonfinite:{column}"}

    if float(method_frame["update_norm_post_cap"].max()) > 0.3:
        return {"pass": False, "reason": "update_norm_explodes"}
    if float(method_frame["approx_kl_after"].abs().max()) > 0.1:
        return {"pass": False, "reason": "approx_kl_spike"}
    if float(method_frame["clip_fraction_after"].max()) > 0.8:
        return {"pass": False, "reason": "clip_fraction_saturates"}
    if float(method_frame["beta_eff"].abs().max()) > 0.03:
        return {"pass": False, "reason": "beta_eff_too_large"}
    gamma_active_column = "gamma_active"
    if gamma_active_column not in method_frame.columns and "gamma_active_frac" in method_frame.columns:
        gamma_active_column = "gamma_active_frac"

    if method == "proposed_qp_new_v2":
        if float(method_frame["gamma_eff"].abs().max()) > 0.03:
            return {"pass": False, "reason": "gamma_eff_too_large"}
        if float(method_frame[gamma_active_column].mean()) <= 0.0:
            return {"pass": False, "reason": "gamma_inactive"}
        if float(method_frame["G_contribution_norm"].mean()) <= 1e-12:
            return {"pass": False, "reason": "g_contribution_tiny"}

    return {
        "pass": True,
        "reason": "pass",
        "beta_mean": float(method_frame["beta"].mean()),
        "gamma_mean": float(method_frame["gamma"].mean()) if "gamma" in method_frame.columns else 0.0,
        "beta_eff_mean": float(method_frame["beta_eff"].mean()),
        "gamma_eff_mean": float(method_frame["gamma_eff"].mean()) if "gamma_eff" in method_frame.columns else 0.0,
        "update_norm_post_cap_mean": float(method_frame["update_norm_post_cap"].mean()),
        "actual_V_change_mean": float(method_frame["actual_V_change"].mean()),
        "approx_kl_after_mean": float(method_frame["approx_kl_after"].mean()),
        "clip_fraction_after_mean": float(method_frame["clip_fraction_after"].mean()),
        "gamma_active_frac": float(method_frame[gamma_active_column].mean()) if gamma_active_column in method_frame.columns else 0.0,
        "G_contribution_norm_mean": float(method_frame["G_contribution_norm"].mean()) if "G_contribution_norm" in method_frame.columns else 0.0,
        "G_over_update_norm_mean": float(method_frame["G_over_update_norm"].mean()) if "G_over_update_norm" in method_frame.columns else 0.0,
        "beta_bound_frac": float(method_frame["beta_at_bound"].mean()) if "beta_at_bound" in method_frame.columns else 0.0,
        "gamma_bound_frac": float(method_frame["gamma_at_bound"].mean()) if "gamma_at_bound" in method_frame.columns else 0.0,
    }


def build_proposed_candidates(eta: float, beta_max: float, gamma_max: float) -> List[Candidate]:
    common = {
        "optimizer_scope": "full_policy",
        "qp_normalization": "block",
        "qp_fd_eps": 1e-3,
        "qp_beta_probe": 1e-3,
        "qp_gamma_probe": 1e-3,
        "qp_ridge": 1e-8,
        "qp_actor_weight": 1.0,
        "qp_logstd_weight": 1.0,
        "qp_critic_weight": 0.3,
        "qp_beta_max": beta_max,
        "qp_gamma_max": gamma_max,
        "qp_max_update_norm": float("inf"),
        "qp_eps": 1e-8,
        "qp_step_solver": "lyapunov_quadratic_bound",
    }
    return [
        Candidate(
            method="proposed_noG_new_v2",
            optimizer="proposed_noG_new_v2",
            lr=eta,
            max_grad_norm=10.0,
            vf_coef=0.5,
            optimizer_kwargs=dict(common),
        ),
        Candidate(
            method="proposed_qp_new_v2",
            optimizer="proposed_qp_new_v2",
            lr=eta,
            max_grad_norm=10.0,
            vf_coef=0.5,
            optimizer_kwargs=dict(common),
        ),
    ]


def ensure_new_run(candidate: Candidate, args: argparse.Namespace, run_root: pathlib.Path) -> pathlib.Path:
    saved_models_dir = run_root / "saved_models"
    logging_dir = run_root / "logging"
    tb_dir = run_root / "tb"
    analysis_dir = run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    env_root = saved_models_dir / "rarl-ppo" / args.env
    if env_root.exists():
        latest_run_dir = find_latest_run_dir(saved_models_dir, args.env)
        if (analysis_dir / "run_summary.csv").exists():
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
    shutil.copy2(run_root / "stdout.txt", analysis_dir / "stdout.txt")
    shutil.copy2(run_root / "stderr.txt", analysis_dir / "stderr.txt")
    run_command(
        [
            args.python_path,
            "scripts/analyze_rarl_run.py",
            "--run-dir",
            str(latest_run_dir),
            "--output-dir",
            str(analysis_dir),
            "--method",
            candidate.method,
        ],
        cwd=pathlib.Path(args.repo_dir),
        stdout_path=analysis_dir / "analyze_stdout.txt",
        stderr_path=analysis_dir / "analyze_stderr.txt",
    )
    return latest_run_dir


def collect_run_diagnostics(run_dir: pathlib.Path, method: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for diag_path in sorted(run_dir.glob("*diagnostics.csv")):
        frame = pd.read_csv(diag_path)
        frame["method"] = method
        frame["diagnostic_file"] = diag_path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_diagnostics(diagnostics_df: pd.DataFrame) -> Dict[str, object]:
    if diagnostics_df.empty:
        return {
            "beta_mean": np.nan,
            "gamma_mean": np.nan,
            "beta_eff_mean": np.nan,
            "gamma_eff_mean": np.nan,
            "update_norm_post_cap_mean": np.nan,
            "gamma_active_frac": np.nan,
            "G_contribution_norm_mean": np.nan,
            "G_over_update_norm_mean": np.nan,
            "approx_kl_after_mean": np.nan,
            "clip_fraction_after_mean": np.nan,
        }
    gamma_active_column = "gamma_active"
    if gamma_active_column not in diagnostics_df.columns and "gamma_active_frac" in diagnostics_df.columns:
        gamma_active_column = "gamma_active_frac"
    result = {
        "beta_mean": float(diagnostics_df["beta"].mean()) if "beta" in diagnostics_df.columns else np.nan,
        "gamma_mean": float(diagnostics_df["gamma"].mean()) if "gamma" in diagnostics_df.columns else np.nan,
        "beta_eff_mean": float(diagnostics_df["beta_eff"].mean()) if "beta_eff" in diagnostics_df.columns else np.nan,
        "gamma_eff_mean": float(diagnostics_df["gamma_eff"].mean()) if "gamma_eff" in diagnostics_df.columns else np.nan,
        "update_norm_post_cap_mean": float(diagnostics_df["update_norm_post_cap"].mean()) if "update_norm_post_cap" in diagnostics_df.columns else np.nan,
        "gamma_active_frac": float(diagnostics_df[gamma_active_column].mean()) if gamma_active_column in diagnostics_df.columns else np.nan,
        "G_contribution_norm_mean": float(diagnostics_df["G_contribution_norm"].mean()) if "G_contribution_norm" in diagnostics_df.columns else np.nan,
        "G_over_update_norm_mean": float(diagnostics_df["G_over_update_norm"].mean()) if "G_over_update_norm" in diagnostics_df.columns else np.nan,
        "approx_kl_after_mean": float(diagnostics_df["approx_kl_after"].mean()) if "approx_kl_after" in diagnostics_df.columns else np.nan,
        "clip_fraction_after_mean": float(diagnostics_df["clip_fraction_after"].mean()) if "clip_fraction_after" in diagnostics_df.columns else np.nan,
    }
    return result


def plot_eta_sweep(summary_df: pd.DataFrame, baseline_df: pd.DataFrame, plots_dir: pathlib.Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    proposed_df = summary_df.loc[summary_df["method"].isin(["proposed_noG_new_v2", "proposed_qp_new_v2"])].copy()
    proposed_df = proposed_df.sort_values(["method", "eta"])
    baseline_colors = {"adam": "tab:orange", "sgd": "tab:gray", "egm": "tab:green", "ppm": "tab:red"}
    method_colors = {"proposed_noG_new_v2": "tab:purple", "proposed_qp_new_v2": "tab:blue"}

    def draw_metric(path_name: str, y_column: str, title: str, y_label: str, baseline_column: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        for method, group in proposed_df.groupby("method"):
            ax.plot(group["eta"], group[y_column], marker="o", linewidth=2, label=method, color=method_colors.get(method))
        for _, row in baseline_df.iterrows():
            ax.axhline(float(row[baseline_column]), linestyle="--", alpha=0.5, color=baseline_colors.get(str(row["method"])), label=f"{row['method']} frozen")
        ax.set_xscale("log")
        ax.set_xlabel("eta")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        dedup: Dict[str, object] = {}
        for handle, label in zip(handles, labels):
            dedup.setdefault(label, handle)
        ax.legend(dedup.values(), dedup.keys(), fontsize=9)
        fig.tight_layout()
        fig.savefig(plots_dir / path_name, dpi=180, bbox_inches="tight")
        plt.close(fig)

    draw_metric(
        "stageP7_clean_eval_eta_sweep.png",
        "last5_clean_mean",
        "Stage P7 Clean Eval vs eta",
        "last5 clean eval mean",
        "last5_clean_mean",
    )
    draw_metric(
        "stageP7_adv_eval_eta_sweep.png",
        "last5_adversarial_mean",
        "Stage P7 Adversarial Eval vs eta",
        "last5 adversarial eval mean",
        "last5_adversarial_mean",
    )
    draw_metric(
        "stageP7_training_return_eta_sweep.png",
        "last5_training_mean",
        "Stage P7 Training Return vs eta",
        "last5 training return",
        "last5_training_mean",
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for method, group in proposed_df.groupby("method"):
        axes[0].plot(group["eta"], group["beta_eff_mean"], marker="o", label=f"{method} beta_eff", color=method_colors.get(method))
        axes[0].plot(group["eta"], group["gamma_eff_mean"], marker="s", linestyle="--", label=f"{method} gamma_eff", color=method_colors.get(method))
        axes[1].plot(group["eta"], group["beta_mean"], marker="o", label=f"{method} beta", color=method_colors.get(method))
        axes[1].plot(group["eta"], group["gamma_mean"], marker="s", linestyle="--", label=f"{method} gamma", color=method_colors.get(method))
    axes[0].set_title("Effective step multipliers")
    axes[1].set_title("Dimensionless beta / gamma")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("eta")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "stageP7_beta_gamma_eff_eta_sweep.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    qp_df = proposed_df.loc[proposed_df["method"] == "proposed_qp_new_v2"].copy()
    if not qp_df.empty:
        axes[0].plot(qp_df["eta"], qp_df["gamma_active_frac"], marker="o", color="tab:blue")
        axes[1].plot(qp_df["eta"], qp_df["G_over_update_norm_mean"], marker="o", color="tab:blue")
        axes[2].plot(qp_df["eta"], qp_df["approx_kl_after_mean"], marker="o", color="tab:blue", label="approx_kl")
        axes[2].plot(qp_df["eta"], qp_df["clip_fraction_after_mean"], marker="s", linestyle="--", color="tab:red", label="clip_fraction")
    axes[0].set_title("QP gamma active fraction")
    axes[1].set_title("QP G / update norm")
    axes[2].set_title("QP KL / clip diagnostics")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("eta")
        ax.grid(alpha=0.3)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "stageP7_qp_diagnostics_eta_sweep.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    final_rows = baseline_df[["method", "last5_clean_mean", "last5_adversarial_mean"]].copy()
    proposed_bar = proposed_df[["method", "eta", "last5_clean_mean", "last5_adversarial_mean"]].copy()
    proposed_bar["method"] = proposed_bar.apply(lambda row: f"{row['method']}@{row['eta']:.0e}", axis=1)
    final_rows = pd.concat([final_rows, proposed_bar[["method", "last5_clean_mean", "last5_adversarial_mean"]]], ignore_index=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(final_rows))
    width = 0.35
    ax.bar(x - width / 2, final_rows["last5_clean_mean"], width=width, label="clean")
    ax.bar(x + width / 2, final_rows["last5_adversarial_mean"], width=width, label="adversarial")
    ax.set_xticks(x)
    ax.set_xticklabels(final_rows["method"], rotation=30, ha="right")
    ax.set_title("Stage P7 final comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stageP7_final_bar.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_dir = pathlib.Path(args.repo_dir)
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    baseline_root = pathlib.Path(args.baseline_root)
    eta_values = parse_eta_list(args.eta_list)

    baseline_df = load_frozen_baselines(baseline_root, args.horizon_timesteps)
    summary_rows: List[Dict[str, object]] = [row for row in baseline_df.to_dict(orient="records")]
    diagnostics_frames: List[pd.DataFrame] = []

    preflight_root = output_root / "preflight_eta_runs"
    online_root = output_root / "stageP7_runs_seed0"
    preflight_root.mkdir(parents=True, exist_ok=True)
    online_root.mkdir(parents=True, exist_ok=True)

    for eta in eta_values:
        label = eta_label(eta)
        preflight_dir = preflight_root / f"eta_{label}"
        run_preflight(args=args, output_dir=preflight_dir, eta=eta)
        preflight_csv = preflight_dir / "stageP5_fixed_minibatch_audit.csv"

        candidates = build_proposed_candidates(eta, args.beta_max, args.gamma_max)
        for candidate in candidates:
            gate = evaluate_preflight_gate(preflight_csv, candidate.method)
            row: Dict[str, object] = {
                "method": candidate.method,
                "optimizer": candidate.optimizer,
                "eta": eta,
                "preflight_pass": bool(gate["pass"]),
                "online_ran": False,
                "notes": str(gate["reason"]),
                "horizon_timesteps": args.horizon_timesteps,
                "beta_mean": gate.get("beta_mean", np.nan),
                "gamma_mean": gate.get("gamma_mean", np.nan),
                "beta_eff_mean": gate.get("beta_eff_mean", np.nan),
                "gamma_eff_mean": gate.get("gamma_eff_mean", np.nan),
                "update_norm_post_cap_mean": gate.get("update_norm_post_cap_mean", np.nan),
                "actual_V_change_mean": gate.get("actual_V_change_mean", np.nan),
                "approx_kl_after_mean": gate.get("approx_kl_after_mean", np.nan),
                "clip_fraction_after_mean": gate.get("clip_fraction_after_mean", np.nan),
                "gamma_active_frac": gate.get("gamma_active_frac", np.nan),
                "G_contribution_norm_mean": gate.get("G_contribution_norm_mean", np.nan),
                "G_over_update_norm_mean": gate.get("G_over_update_norm_mean", np.nan),
                "beta_bound_frac": gate.get("beta_bound_frac", np.nan),
                "gamma_bound_frac": gate.get("gamma_bound_frac", np.nan),
            }
            if not gate["pass"]:
                summary_rows.append(row)
                continue

            run_root = online_root / f"{candidate.method}_eta_{label}"
            latest_run_dir = ensure_new_run(candidate, args, run_root)
            analysis_dir = run_root / "analysis"
            training_df = load_curve_frame(analysis_dir / "training_episode_returns.csv", candidate.method)
            clean_df = load_curve_frame(analysis_dir / "clean_eval_returns.csv", candidate.method)
            adv_df = load_curve_frame(analysis_dir / "adversarial_eval_returns.csv", candidate.method)
            short_metrics = summarize_short_horizon(
                method=candidate.method,
                training_df=training_df,
                clean_df=clean_df,
                adv_df=adv_df,
                horizon_timesteps=args.horizon_timesteps,
            )
            diag_df = collect_run_diagnostics(latest_run_dir, candidate.method)
            if not diag_df.empty:
                diag_df["eta"] = eta
                diagnostics_frames.append(diag_df)
            diag_summary = summarize_diagnostics(diag_df)
            row.update(short_metrics)
            row.update(diag_summary)
            row["online_ran"] = True
            row["notes"] = "online_completed"
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    diagnostics_df = pd.concat(diagnostics_frames, ignore_index=True) if diagnostics_frames else pd.DataFrame()

    summary_df.to_csv(output_root / "stageP7_online_eta_sweep_summary.csv", index=False)
    diagnostics_df.to_csv(output_root / "stageP7_online_eta_sweep_diagnostics.csv", index=False)

    plot_eta_sweep(summary_df, baseline_df, plots_dir)

    report_lines = [
        "# Stage P7 Online eta Sweep Report",
        "",
        "- Protocol: `proper control-RARL`, `full_policy`, `seed=0`, short online only.",
        f"- Online budget per proposed config: `{args.iterations}` outer iteration(s), horizon reference `{args.horizon_timesteps}` timesteps.",
        "- Frozen baselines were read from the existing control benchmark; they were not retrained or retuned here.",
        "",
    ]

    proposed_qp_df = summary_df.loc[summary_df["method"] == "proposed_qp_new_v2"].copy()
    proposed_nog_df = summary_df.loc[summary_df["method"] == "proposed_noG_new_v2"].copy()
    online_qp_df = proposed_qp_df.loc[proposed_qp_df["online_ran"] == True].copy()
    online_nog_df = proposed_nog_df.loc[proposed_nog_df["online_ran"] == True].copy()

    egm_ref = float(baseline_df.loc[baseline_df["method"] == "egm", "last5_clean_mean"].iloc[0])
    ppm_ref = float(baseline_df.loc[baseline_df["method"] == "ppm", "last5_clean_mean"].iloc[0])
    sgd_ref = float(baseline_df.loc[baseline_df["method"] == "sgd", "last5_clean_mean"].iloc[0])

    is_eta_conservative = False
    eta_1e3 = proposed_qp_df.loc[np.isclose(proposed_qp_df["eta"].astype(float), 1e-3)]
    if not eta_1e3.empty:
        is_eta_conservative = bool(
            float(eta_1e3["update_norm_post_cap_mean"].fillna(0.0).mean())
            < float(proposed_qp_df["update_norm_post_cap_mean"].fillna(0.0).max()) * 0.2
        )

    egm_preflight = pd.read_csv(preflight_root / f"eta_{eta_label(eta_values[0])}" / "stageP5_fixed_minibatch_audit.csv")
    egm_update_norm = float(egm_preflight.loc[egm_preflight["method"] == "egm", "update_norm_post_cap"].mean())
    ppm_update_norm = float(egm_preflight.loc[egm_preflight["method"] == "ppm", "update_norm_post_cap"].mean())
    qp_online = proposed_qp_df.loc[proposed_qp_df["preflight_pass"] == True].copy()
    if not qp_online.empty:
        qp_online["distance_to_egm_ppm"] = (qp_online["update_norm_post_cap_mean"] - (0.5 * (egm_update_norm + ppm_update_norm))).abs()
        closest_eta = float(qp_online.sort_values("distance_to_egm_ppm").iloc[0]["eta"])
    else:
        closest_eta = float("nan")

    report_lines.append(f"- Is current eta=1e-3 too conservative online? `{is_eta_conservative}`")
    report_lines.append(f"- Which eta gives update_norm closest to EGM/PPM preflight scale? `{closest_eta}`")

    if not online_qp_df.empty and not online_nog_df.empty:
        merged = online_qp_df.merge(online_nog_df, on="eta", suffixes=("_qp", "_nog"))
        merged["qp_beats_nog"] = (
            (merged["last5_clean_mean_qp"] > merged["last5_clean_mean_nog"])
            & (merged["last5_adversarial_mean_qp"] > merged["last5_adversarial_mean_nog"])
        )
        report_lines.append(f"- proposed_qp_new_v2 beats proposed_noG_new_v2 at any eta: `{bool(merged['qp_beats_nog'].any())}`")
    else:
        report_lines.append("- proposed_qp_new_v2 beats proposed_noG_new_v2 at any eta: `False`")

    if not online_qp_df.empty:
        best_qp = online_qp_df.sort_values("last5_clean_mean", ascending=False).iloc[0]
        report_lines.extend(
            [
                f"- Best QP eta by clean eval: `{best_qp['eta']}`",
                f"- proposed_qp_new_v2 beats EGM at best eta: `{bool(best_qp['last5_clean_mean'] > egm_ref)}`",
                f"- proposed_qp_new_v2 beats PPM at best eta: `{bool(best_qp['last5_clean_mean'] > ppm_ref)}`",
                f"- proposed_qp_new_v2 beats SGD at best eta: `{bool(best_qp['last5_clean_mean'] > sgd_ref)}`",
                f"- Best QP clean / adv means: `{best_qp['last5_clean_mean']:.6f}` / `{best_qp['last5_adversarial_mean']:.6f}`",
            ]
        )
        likely_reason = []
        if float(best_qp["update_norm_post_cap_mean"]) < 5 * egm_update_norm:
            likely_reason.append("update_too_small")
        if float(best_qp["gamma_active_frac"]) <= 0.0 or float(best_qp["G_contribution_norm_mean"]) <= 1e-12:
            likely_reason.append("gamma_too_weak")
        if float(best_qp["approx_kl_after_mean"]) > 0.05:
            likely_reason.append("eta_too_aggressive")
        if not likely_reason:
            likely_reason.append("EGM_still_best_aligned_with_PPO_return")
        report_lines.append(f"- If QP still loses to EGM, current most likely reason: `{','.join(likely_reason)}`")
    else:
        report_lines.append("- No QP eta passed preflight strongly enough to produce an online run.")

    (output_root / "stageP7_online_eta_sweep_report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
