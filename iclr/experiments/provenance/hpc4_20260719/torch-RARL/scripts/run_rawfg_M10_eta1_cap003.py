from __future__ import annotations

import argparse
import math
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.run_rawfg_M8_online import (
    evaluate_control_strength_sweep,
    find_latest_run_dir,
    make_collage,
    read_method_frames,
)


@dataclass(frozen=True)
class Candidate:
    method_label: str
    optimizer: str
    lr: float
    max_grad_norm: float
    vf_coef: float
    optimizer_kwargs: Dict[str, object]
    eta_ext: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage M10 eta=1 capped rawFG online")
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
    tokens: List[str] = []
    for key, value in sorted(kwargs.items()):
        if isinstance(value, str):
            rendered = repr(value)
        elif isinstance(value, float) and math.isinf(value):
            rendered = "float('inf')"
        else:
            rendered = value
        tokens.append(f"{key}:{rendered}")
    return tokens


def summarize_preflight(preflight_csv: pathlib.Path, eta_egm: float, update_cap: float) -> pd.DataFrame:
    df = pd.read_csv(preflight_csv)
    rows: List[Dict[str, object]] = []
    tol = 1e-5
    for method in ["proposed_noG_new_v2_rawFG", "proposed_qp_new_v2_rawFG"]:
        sub = df[df["method"] == method].copy()
        if sub.empty:
            continue
        is_qp = method.endswith("qp_new_v2_rawFG")
        core_numeric = ["beta_eff", "update_norm_post_cap", "V_before", "V_after", "actual_V_change", "q_pred", "approx_kl_after", "clip_fraction_after"]
        if is_qp:
            core_numeric.extend(["beta_QP", "gamma_QP", "gamma_eff", "gamma_active_frac", "G_contribution_norm"])
        else:
            core_numeric.extend(["beta_noG"])
        core_nan = False
        for col in core_numeric:
            if col not in sub.columns:
                core_nan = True
                break
            vals = pd.to_numeric(sub[col], errors="coerce")
            if not vals.map(np.isfinite).all():
                core_nan = True
                break
        reasons: List[str] = []
        if core_nan:
            reasons.append("core_nan_or_inf")
        if is_qp:
            no_g = df[df["method"] == "proposed_noG_new_v2_rawFG"].copy()
            if float(sub["actual_V_change"].mean()) >= float(no_g["actual_V_change"].mean()):
                reasons.append("actual_V_change_QP_not_better_than_noG")
            if float(sub["gamma_active_frac"].mean()) <= 0.0:
                reasons.append("gamma_inactive_for_QP")
            if float(sub["G_contribution_norm"].mean()) <= 0.0:
                reasons.append("G_contribution_zero_for_QP")
        if float(sub["approx_kl_after"].max()) > 0.1:
            reasons.append("approx_kl_spike")
        if float(sub["clip_fraction_after"].max()) > 0.8:
            reasons.append("clip_fraction_saturates")
        if float(sub["update_norm_post_cap"].max()) > update_cap + tol:
            reasons.append("update_norm_post_cap_exceeds_cap")

        beta_col = "beta_QP" if is_qp else "beta_noG"
        gamma_col = "gamma_QP" if is_qp else None
        row = {
            "method": "proposed_qp_rawFG_eta1_cap003" if is_qp else "proposed_noG_rawFG_eta1_cap003",
            "preflight_pass": len(reasons) == 0,
            "failure_reason": "pass" if not reasons else "|".join(reasons),
            "eta_ext": 1.0,
            "beta_mean": float(sub[beta_col].mean()),
            "gamma_mean": float(sub[gamma_col].mean()) if is_qp else 0.0,
            "beta_eff_mean": float(sub["beta_eff"].mean()),
            "gamma_eff_mean": float(sub["gamma_eff"].mean()) if is_qp else 0.0,
            "beta_eff_over_eta_egm_mean": float(sub["beta_eff"].mean() / eta_egm),
            "gamma_eff_over_eta_egm_squared_mean": float(sub["gamma_eff"].mean() / (eta_egm ** 2)) if is_qp else 0.0,
            "update_norm_pre_cap_mean": float(sub["update_norm_pre_cap"].mean()),
            "update_norm_post_cap_mean": float(sub["update_norm_post_cap"].mean()),
            "cap_active_frac": float(sub["cap_active"].mean()) if "cap_active" in sub.columns else np.nan,
            "actual_V_change_mean": float(sub["actual_V_change"].mean()),
            "q_pred_mean": float(sub["q_pred"].mean()),
            "approx_kl_max": float(sub["approx_kl_after"].max()),
            "clip_fraction_max": float(sub["clip_fraction_after"].max()),
            "gamma_active_frac_mean": float(sub["gamma_active_frac"].mean()) if is_qp else 0.0,
            "G_contribution_norm_mean": float(sub["G_contribution_norm"].mean()) if is_qp else 0.0,
            "G_over_update_norm_mean": float(sub["G_over_update_norm"].mean()) if is_qp else 0.0,
            "beta_at_bound_frac": float(sub["beta_at_bound"].mean()),
            "gamma_at_bound_frac": float(sub["gamma_at_bound"].mean()) if is_qp else 0.0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def run_preflight(args: argparse.Namespace, output_root: pathlib.Path) -> pd.DataFrame:
    preflight_dir = output_root / "preflight" / "M10_eta1_cap003"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    command = [
        args.python_path,
        "scripts/run_proposed_qp_new_v2_rawfg_audit.py",
        "--output-dir",
        str(preflight_dir),
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
        "1.0",
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
        "0.5",
        "--vf-coef",
        "0.5",
        "--single-beta-max",
        "1e-2",
        "--single-gamma-max",
        "3e-5",
        "--update-cap",
        "0.003",
    ]
    run_command(command, pathlib.Path(args.repo_dir), preflight_dir / "stdout.txt", preflight_dir / "stderr.txt")
    summary = summarize_preflight(preflight_dir / "raw_fg_qp_audit.csv", args.eta_egm, update_cap=0.003)
    summary.to_csv(output_root / "rawFG_M10_eta1_cap003_preflight.csv", index=False)

    lines = [
        "# RawFG M10 eta=1 cap=0.003 preflight report",
        "",
        "- Exact preflight rerun from current rawFG audit chain.",
        "- Control-RARL probe only. No online training in this step.",
        "",
    ]
    for _, row in summary.iterrows():
        lines.extend(
            [
                f"- `{row['method']}`",
                f"  preflight_pass={bool(row['preflight_pass'])}, failure_reason={row['failure_reason']}",
                f"  beta_mean={row['beta_mean']:.6e}, gamma_mean={row['gamma_mean']:.6e}",
                f"  beta_eff_mean={row['beta_eff_mean']:.6e}, gamma_eff_mean={row['gamma_eff_mean']:.6e}",
                f"  beta_eff/eta_EGM={row['beta_eff_over_eta_egm_mean']:.6f}, gamma_eff/eta_EGM^2={row['gamma_eff_over_eta_egm_squared_mean']:.6f}",
                f"  update_norm_pre_cap_mean={row['update_norm_pre_cap_mean']:.6e}, update_norm_post_cap_mean={row['update_norm_post_cap_mean']:.6e}, cap_active_frac={row['cap_active_frac']:.3f}",
                f"  actual_V_change_mean={row['actual_V_change_mean']:.6e}, q_pred_mean={row['q_pred_mean']:.6e}",
                f"  approx_kl_max={row['approx_kl_max']:.6e}, clip_fraction_max={row['clip_fraction_max']:.6e}",
                f"  gamma_active_frac_mean={row['gamma_active_frac_mean']:.6f}, G_contribution_norm_mean={row['G_contribution_norm_mean']:.6e}",
                "",
            ]
        )
    (output_root / "rawFG_M10_eta1_cap003_preflight_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def ensure_existing_baseline(method: str, args: argparse.Namespace, runs_dir: pathlib.Path) -> pathlib.Path:
    run_root = runs_dir / method
    analysis_dir = run_root / "analysis"
    saved_models_dir = run_root / "saved_models"
    if not (analysis_dir / "run_summary.csv").exists():
        raise FileNotFoundError(f"Missing baseline analysis for {method}: {analysis_dir / 'run_summary.csv'}")
    return find_latest_run_dir(saved_models_dir, args.env)


def ensure_proposed_run(candidate: Candidate, args: argparse.Namespace, runs_dir: pathlib.Path) -> pathlib.Path:
    run_root = runs_dir / candidate.method_label
    saved_models_dir = run_root / "saved_models"
    analysis_dir = run_root / "analysis"
    env_root = saved_models_dir / "rarl-ppo" / args.env
    if env_root.exists() and (analysis_dir / "run_summary.csv").exists():
        latest = find_latest_run_dir(saved_models_dir, args.env)
        if (latest / "adv_eval" / "evaluations.npz").exists():
            return latest

    logging_dir = run_root / "logging"
    tb_dir = run_root / "tb"
    run_root.mkdir(parents=True, exist_ok=True)
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
    run_command(command, pathlib.Path(args.repo_dir), run_root / "stdout.txt", run_root / "stderr.txt")
    latest_run_dir = find_latest_run_dir(saved_models_dir, args.env)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_root / "stdout.txt", analysis_dir / "stdout.txt")
    shutil.copy2(run_root / "stderr.txt", analysis_dir / "stderr.txt")
    run_command(
        [args.python_path, "scripts/analyze_rarl_run.py", "--run-dir", str(latest_run_dir), "--output-dir", str(analysis_dir), "--method", candidate.method_label],
        pathlib.Path(args.repo_dir),
        analysis_dir / "analyze_stdout.txt",
        analysis_dir / "analyze_stderr.txt",
    )
    return latest_run_dir


def plot_eval_with_band(ax, df: pd.DataFrame, title: str, ylabel: str, colors: Dict[str, str]) -> None:
    for method, group in df.groupby("method"):
        group = group.sort_values("outer_iteration")
        ax.plot(group["outer_iteration"], group["mean_reward"], label=method, color=colors.get(method))
        ax.fill_between(group["outer_iteration"], group["mean_reward"] - group["std_reward"], group["mean_reward"] + group["std_reward"], color=colors.get(method), alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def plot_training(ax, df: pd.DataFrame, colors: Dict[str, str]) -> None:
    for method, group in df.groupby("method"):
        group = group.sort_values("outer_iteration")
        ax.plot(group["outer_iteration"], group["episode_return"], label=method, color=colors.get(method))
    ax.set_title("Training return vs outer iteration")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Episode return")
    ax.grid(alpha=0.3)


def plot_qp_beta_gamma(ax_beta, ax_gamma, df: pd.DataFrame, colors: Dict[str, str]) -> None:
    for method, group in df.groupby("method"):
        group = group[group["optimizer_role"] == "protagonist"].sort_values("outer_iteration")
        ax_beta.plot(group["outer_iteration"], group["beta"], label=f"{method} beta", color=colors.get(method))
        ax_beta.plot(group["outer_iteration"], group["beta_eff"], linestyle="--", label=f"{method} beta_eff", color=colors.get(method))
        ax_gamma.plot(group["outer_iteration"], group["gamma"], label=f"{method} gamma", color=colors.get(method))
        ax_gamma.plot(group["outer_iteration"], group["gamma_eff"], linestyle="--", label=f"{method} gamma_eff", color=colors.get(method))
    ax_beta.set_title("beta / beta_eff")
    ax_gamma.set_title("gamma / gamma_eff")
    for ax in (ax_beta, ax_gamma):
        ax.set_xlabel("Outer iteration")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)


def plot_qp_cap(ax1, ax2, df: pd.DataFrame, colors: Dict[str, str]) -> None:
    for method, group in df.groupby("method"):
        group = group[group["optimizer_role"] == "protagonist"].sort_values("outer_iteration")
        ax1.plot(group["outer_iteration"], group["update_norm_pre_cap"], label=f"{method} pre", color=colors.get(method))
        ax1.plot(group["outer_iteration"], group["update_norm_post_cap"], linestyle="--", label=f"{method} post", color=colors.get(method))
        ax2.plot(group["outer_iteration"], group["cap_active"], label=f"{method} cap_active", color=colors.get(method))
    ax1.set_title("Update norm pre/post cap")
    ax2.set_title("Cap active")
    for ax in (ax1, ax2):
        ax.set_xlabel("Outer iteration")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)


def plot_qp_v_kl_clip(axs: Sequence, df: pd.DataFrame, colors: Dict[str, str]) -> None:
    for method, group in df.groupby("method"):
        group = group[group["optimizer_role"] == "protagonist"].sort_values("outer_iteration")
        axs[0].plot(group["outer_iteration"], group["actual_V_change"], label=method, color=colors.get(method))
        axs[1].plot(group["outer_iteration"], group["q_pred"], label=method, color=colors.get(method))
        axs[2].plot(group["outer_iteration"], group["approx_kl"], label=method, color=colors.get(method))
        axs[3].plot(group["outer_iteration"], group["clip_fraction"], label=method, color=colors.get(method))
    titles = ["actual_V_change", "q_pred", "approx_kl", "clip_fraction"]
    for ax, title in zip(axs, titles):
        ax.set_title(title)
        ax.set_xlabel("Outer iteration")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    runs_dir = output_root / "runs_seed0"
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    preflight_df = run_preflight(args, output_root)
    qp_pre = preflight_df[preflight_df["method"] == "proposed_qp_rawFG_eta1_cap003"]
    nog_pre = preflight_df[preflight_df["method"] == "proposed_noG_rawFG_eta1_cap003"]
    if qp_pre.empty or not bool(qp_pre.iloc[0]["preflight_pass"]):
        raise RuntimeError(f"M10 stopped: QP preflight failed: {qp_pre.to_dict('records')}")

    common_kwargs = {
        "optimizer_scope": "full_policy",
        "qp_fd_eps": 1e-3,
        "qp_beta_probe": 1e-3,
        "qp_gamma_probe": 1e-6,
        "qp_ridge": 1e-8,
        "qp_actor_weight": 1.0,
        "qp_logstd_weight": 1.0,
        "qp_critic_weight": 0.3,
        "qp_beta_max": 1e-2,
        "qp_gamma_max": 3e-5,
        "qp_max_update_norm": 0.003,
        "qp_eps": 1e-8,
    }
    candidates = [
        Candidate("proposed_noG_rawFG_eta1_cap003", "proposed_noG_rawFG", 1.0, 0.5, 0.5, dict(common_kwargs), 1.0),
        Candidate("proposed_qp_rawFG_eta1_cap003", "proposed_qp_rawFG", 1.0, 0.5, 0.5, dict(common_kwargs), 1.0),
    ]
    proposed_run_dirs: Dict[str, pathlib.Path] = {}
    for candidate in candidates:
        if candidate.method_label == "proposed_noG_rawFG_eta1_cap003" and (not nog_pre.empty) and (not bool(nog_pre.iloc[0]["preflight_pass"])):
            continue
        proposed_run_dirs[candidate.method_label] = ensure_proposed_run(candidate, args, runs_dir)

    baseline_methods = ["adam", "sgd", "egm", "ppm"]
    method_order = baseline_methods + [c.method_label for c in candidates if c.method_label in proposed_run_dirs]

    summary_rows = []
    training_frames = []
    clean_frames = []
    adv_frames = []
    qp_diag_frames = []

    for method_label in method_order:
        latest_run_dir = ensure_existing_baseline(method_label, args, runs_dir) if method_label in baseline_methods else proposed_run_dirs[method_label]
        frames = read_method_frames(method_label, latest_run_dir, runs_dir / method_label / "analysis")
        training_frames.append(frames["training"])
        clean_frames.append(frames["clean"])
        adv_frames.append(frames["adv"])
        if not frames["diagnostics"].empty:
            diag = frames["diagnostics"].copy()
            diag["eta_ext"] = 1.0 if "eta1_cap003" in method_label else np.nan
            diag["beta_eff_over_eta_egm"] = pd.to_numeric(diag["beta_eff"], errors="coerce") / args.eta_egm if "beta_eff" in diag.columns else np.nan
            diag["gamma_eff_over_eta_egm_squared"] = pd.to_numeric(diag["gamma_eff"], errors="coerce") / (args.eta_egm ** 2) if "gamma_eff" in diag.columns else np.nan
            qp_diag_frames.append(diag)
        summary = pd.read_csv(runs_dir / method_label / "analysis" / "run_summary.csv").iloc[0].to_dict()
        summary["method"] = method_label
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    training_df = pd.concat(training_frames, ignore_index=True)
    clean_df = pd.concat(clean_frames, ignore_index=True)
    adv_df = pd.concat(adv_frames, ignore_index=True)
    qp_diag_df = pd.concat(qp_diag_frames, ignore_index=True) if qp_diag_frames else pd.DataFrame()

    eval_df = pd.concat([clean_df.assign(eval_type="clean"), adv_df.assign(eval_type="control_adversarial")], ignore_index=True)

    baseline_sweep = pd.read_csv(output_root / "rawFG_matched_robustness_sweep.csv")
    baseline_sweep = baseline_sweep[baseline_sweep["method"].isin(baseline_methods)].copy()
    proposed_sweeps = [evaluate_control_strength_sweep(run_dir, method, args.device, args.n_eval_episodes) for method, run_dir in proposed_run_dirs.items()]
    sweep_df = pd.concat([baseline_sweep] + proposed_sweeps, ignore_index=True) if proposed_sweeps else baseline_sweep

    summary_df.to_csv(output_root / "rawFG_M10_eta1_cap003_summary.csv", index=False)
    eval_df.to_csv(output_root / "rawFG_M10_eta1_cap003_eval_curves.csv", index=False)
    qp_diag_df.to_csv(output_root / "rawFG_M10_eta1_cap003_qp_diagnostics.csv", index=False)

    colors = {
        "adam": "tab:orange",
        "sgd": "tab:gray",
        "egm": "tab:green",
        "ppm": "tab:red",
        "proposed_noG_rawFG_eta1_cap003": "tab:purple",
        "proposed_qp_rawFG_eta1_cap003": "tab:blue",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_training(ax, training_df, colors)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "M10_training_return_vs_outer_iteration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_eval_with_band(ax, clean_df, "Clean eval vs outer iteration", "Mean return", colors)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "M10_clean_eval_vs_outer_iteration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_eval_with_band(ax, adv_df, "Control adversarial eval vs outer iteration", "Mean return", colors)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "M10_control_adv_eval_vs_outer_iteration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    proposed_only = qp_diag_df[qp_diag_df["method"].str.contains("proposed_")].copy() if not qp_diag_df.empty else pd.DataFrame()
    if not proposed_only.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plot_qp_beta_gamma(axes[0], axes[1], proposed_only, colors)
        fig.tight_layout()
        fig.savefig(plots_dir / "M10_qp_beta_gamma.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plot_qp_cap(axes[0], axes[1], proposed_only, colors)
        fig.tight_layout()
        fig.savefig(plots_dir / "M10_qp_update_cap_diagnostics.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        plot_qp_v_kl_clip(list(axes.ravel()), proposed_only, colors)
        fig.tight_layout()
        fig.savefig(plots_dir / "M10_qp_V_KL_clip.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in sweep_df.groupby("method"):
        group = group.sort_values("adv_strength")
        ax.errorbar(group["adv_strength"], group["mean_return"], yerr=group["std_return"], marker="o", label=method, color=colors.get(method))
    ax.set_title("Control robustness sweep final")
    ax.set_xlabel("Adversary strength")
    ax.set_ylabel("Mean return")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "M10_control_robustness_sweep_final.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    bar_rows = []
    for method in method_order:
        row = summary_df[summary_df["method"] == method].iloc[0]
        method_sweep = sweep_df[sweep_df["method"] == method].sort_values("adv_strength")
        auc = float(np.trapz(method_sweep["mean_return"], method_sweep["adv_strength"])) if not method_sweep.empty else np.nan
        bar_rows.extend(
            [
                {"method": method, "metric": "clean_last5", "value": float(row.get("last5_clean_mean", np.nan))},
                {"method": method, "metric": "control_adv_last5", "value": float(row.get("last5_adversarial_mean", np.nan))},
                {"method": method, "metric": "robustness_auc", "value": auc},
            ]
        )
    bar_df = pd.DataFrame(bar_rows)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric, title in zip(axes, ["clean_last5", "control_adv_last5", "robustness_auc"], ["Final clean", "Final control-adv", "Robustness AUC"]):
        sub = bar_df[bar_df["metric"] == metric]
        ax.bar(sub["method"], sub["value"], color=[colors.get(m) for m in sub["method"]])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "M10_final_bar_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    collage_paths = [
        plots_dir / "M10_training_return_vs_outer_iteration.png",
        plots_dir / "M10_clean_eval_vs_outer_iteration.png",
        plots_dir / "M10_control_adv_eval_vs_outer_iteration.png",
        plots_dir / "M10_qp_beta_gamma.png",
        plots_dir / "M10_qp_update_cap_diagnostics.png",
        plots_dir / "M10_qp_V_KL_clip.png",
        plots_dir / "M10_control_robustness_sweep_final.png",
        plots_dir / "M10_final_bar_comparison.png",
    ]
    make_collage(collage_paths, plots_dir / "M10_all_plots_big.png", cols=2)

    def last5(method: str, col: str) -> float:
        sub = summary_df[summary_df["method"] == method]
        if sub.empty:
            return float("nan")
        return float(sub.iloc[0].get(col, np.nan))

    qp_method = "proposed_qp_rawFG_eta1_cap003"
    nog_method = "proposed_noG_rawFG_eta1_cap003"
    qp_present = qp_method in summary_df["method"].values
    nog_present = nog_method in summary_df["method"].values
    qp_diag_sub = proposed_only[proposed_only["method"] == qp_method].copy() if not proposed_only.empty else pd.DataFrame()
    eta1_better_than_nog_clean = qp_present and nog_present and last5(qp_method, "last5_clean_mean") > last5(nog_method, "last5_clean_mean")
    eta1_better_than_nog_adv = qp_present and nog_present and last5(qp_method, "last5_adversarial_mean") > last5(nog_method, "last5_adversarial_mean")
    qp_beats_sgd_clean = qp_present and last5(qp_method, "last5_clean_mean") > last5("sgd", "last5_clean_mean")
    qp_beats_ppm_clean = qp_present and last5(qp_method, "last5_clean_mean") > last5("ppm", "last5_clean_mean")
    qp_beats_egm_clean = qp_present and last5(qp_method, "last5_clean_mean") > last5("egm", "last5_clean_mean")
    qp_beats_adv_baselines = qp_present and last5(qp_method, "last5_adversarial_mean") > max(last5("sgd", "last5_adversarial_mean"), last5("ppm", "last5_adversarial_mean"), last5("egm", "last5_adversarial_mean"))

    lines = [
        "# RawFG M10 eta=1 cap=0.003 final report",
        "",
        f"1. Did eta=1 cap=0.003 pass preflight? `{bool(qp_present)}`",
        f"2. Did QP beat noG? `clean={eta1_better_than_nog_clean}, control_adv={eta1_better_than_nog_adv}`",
        f"3. Did QP beat SGD on clean? `{qp_beats_sgd_clean}`",
        f"4. Did QP beat PPM on clean? `{qp_beats_ppm_clean}`",
        f"5. Did QP beat EGM on clean? `{qp_beats_egm_clean}`",
        f"6. Did QP beat SGD/PPM/EGM on control adversarial eval? `{qp_beats_adv_baselines}`",
    ]
    if not qp_diag_sub.empty:
        lines.extend(
            [
                f"7. Was cap active frequently? `{float(qp_diag_sub['cap_active'].mean()) > 0.25}`",
                f"8. Were beta_eff/gamma_eff EGM-scale or larger? `{bool(float(qp_diag_sub['beta_eff_over_eta_egm'].mean()) >= 1.0 and float(qp_diag_sub['gamma_eff_over_eta_egm_squared'].mean()) >= 1.0)}`",
                f"9. Did gamma remain active online? `{bool(float(qp_diag_sub['gamma_active_frac'].mean()) > 0.0)}`",
                f"10. Did QP remain KL/clip healthy? `{bool(float(qp_diag_sub['approx_kl'].max()) <= 0.1 and float(qp_diag_sub['clip_fraction'].max()) <= 0.8)}`",
            ]
        )
    else:
        lines.extend(
            [
                "7. Was cap active frequently? `False`",
                "8. Were beta_eff/gamma_eff EGM-scale or larger? `False`",
                "9. Did gamma remain active online? `False`",
                "10. Did QP remain KL/clip healthy? `False`",
            ]
        )

    failure_label = "F. EGM/PPM still better aligned with PPO return"
    if qp_present and not qp_beats_sgd_clean and not qp_beats_ppm_clean and not qp_beats_egm_clean and not qp_diag_sub.empty:
        cap_active_frac = float(qp_diag_sub["cap_active"].mean())
        if cap_active_frac > 0.5:
            failure_label = "A. cap too tight"
    lines.append(f"11. If QP still loses, classification: `{failure_label}`")

    (output_root / "rawFG_M10_eta1_cap003_final_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
