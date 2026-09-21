from __future__ import annotations

import argparse
import math
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.full_policy_followup_common import SavedRun, load_yaml, load_rarl_for_eval, set_rarl_eval_mode


@dataclass(frozen=True)
class RawFGConfig:
    label: str
    beta_max: float
    gamma_max: float


@dataclass(frozen=True)
class Candidate:
    method: str
    optimizer: str
    lr: float
    max_grad_norm: float
    vf_coef: float
    optimizer_kwargs: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage R6 raw F/G short online benchmark")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument("--python-path", type=str, required=True)
    parser.add_argument("--baseline-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--short-iterations", type=int, default=1)
    parser.add_argument("--long-iterations", type=int, default=5)
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


def evaluate_control_strength_sweep(run_dir: pathlib.Path, method: str, device: str, n_eval_episodes: int) -> pd.DataFrame:
    saved_run = build_saved_run(run_dir, method)
    strengths = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    rows = []
    model, vec_env = load_rarl_for_eval(saved_run, adv_impact="control", adv_strength=1.0, device=device)
    for strength in strengths:
        set_rarl_eval_mode(model, vec_env, operating_mode="protagonist", adv_strength=strength)
        episode_rewards = []
        obs = vec_env.reset()
        ep_reward = 0.0
        while len(episode_rewards) < n_eval_episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            del infos
            ep_reward += float(rewards[0])
            if bool(dones[0]):
                episode_rewards.append(ep_reward)
                obs = vec_env.reset()
                ep_reward = 0.0
        rows.append(
            {
                "method": method,
                "adv_strength": strength,
                "mean_return": float(np.mean(episode_rewards)),
                "std_return": float(np.std(episode_rewards)),
            }
        )
    vec_env.close()
    return pd.DataFrame(rows)


def load_analysis_frame(path: pathlib.Path, method: str, config_label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["method"] = method
    frame["config_label"] = config_label
    return frame


def load_baseline_frames(baseline_root: pathlib.Path) -> Dict[str, Dict[str, object]]:
    data: Dict[str, Dict[str, object]] = {}
    for method in ["adam", "sgd", "egm", "ppm"]:
        analysis_dir = baseline_root / method / "analysis"
        data[method] = {
            "summary": pd.read_csv(analysis_dir / "run_summary.csv"),
            "training": pd.read_csv(analysis_dir / "training_episode_returns.csv"),
            "clean": pd.read_csv(analysis_dir / "clean_eval_returns.csv"),
            "adv": pd.read_csv(analysis_dir / "adversarial_eval_returns.csv"),
            "run_dir": find_latest_run_dir(baseline_root / method / "saved_models", "HalfCheetah-v4"),
        }
    return data


def run_rawfg_preflight(args: argparse.Namespace, config: RawFGConfig, preflight_dir: pathlib.Path) -> pathlib.Path:
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
        "--ppm-inner-steps",
        "10",
        "--single-beta-max",
        str(config.beta_max),
        "--single-gamma-max",
        str(config.gamma_max),
    ]
    run_command(command, cwd=pathlib.Path(args.repo_dir), stdout_path=preflight_dir / "preflight_stdout.txt", stderr_path=preflight_dir / "preflight_stderr.txt")
    return preflight_dir / "raw_fg_qp_audit.csv"


def evaluate_preflight_gate(preflight_csv: pathlib.Path) -> Dict[str, object]:
    df = pd.read_csv(preflight_csv)
    qp = df[df["method"] == "proposed_qp_new_v2_rawFG"].copy()
    nog = df[df["method"] == "proposed_noG_new_v2_rawFG"].copy()
    if qp.empty or nog.empty:
        return {"pass": False, "reason": "missing_qp_or_nog"}
    qp_v = float(qp["actual_V_change"].mean())
    nog_v = float(nog["actual_V_change"].mean())
    gamma_active_frac = float(qp["gamma_active"].mean())
    g_contrib = float(qp["G_contribution_norm"].mean())
    kl_max = float(qp["approx_kl_after"].max())
    clip_max = float(qp["clip_fraction_after"].max())
    update_norm_mean = float(qp["update_norm"].mean())
    egm_norm = float(df[df["method"] == "egm"]["update_norm"].mean())
    if not np.isfinite(qp[["beta_QP", "gamma_QP"]].to_numpy(dtype=float)).all():
        return {"pass": False, "reason": "nonfinite_beta_gamma"}
    if gamma_active_frac <= 0.0:
        return {"pass": False, "reason": "gamma_inactive"}
    if g_contrib <= 0.0:
        return {"pass": False, "reason": "g_contribution_zero"}
    if not (qp_v < nog_v):
        return {"pass": False, "reason": "qp_not_better_than_nog_on_V"}
    if kl_max > 0.1:
        return {"pass": False, "reason": "approx_kl_spike"}
    if clip_max > 0.8:
        return {"pass": False, "reason": "clip_fraction_saturates"}
    if update_norm_mean > max(egm_norm * 5.0, 1e-12):
        return {"pass": False, "reason": "update_norm_too_large"}
    return {
        "pass": True,
        "reason": "pass",
        "qp_v_mean": qp_v,
        "nog_v_mean": nog_v,
        "gamma_active_frac": gamma_active_frac,
        "g_contribution_norm_mean": g_contrib,
        "kl_max": kl_max,
        "clip_max": clip_max,
        "update_norm_mean": update_norm_mean,
        "beta_qp_mean": float(qp["beta_QP"].mean()),
        "gamma_qp_mean": float(qp["gamma_QP"].mean()),
        "beta_over_eta_mean": float(qp["beta_QP_over_eta_EGM"].mean()),
        "gamma_over_eta2_mean": float(qp["gamma_QP_over_eta_EGM_squared"].mean()),
        "beta_nog_mean": float(nog["beta_noG"].mean()),
        "beta_nog_over_eta_mean": float(nog["beta_noG_over_eta_EGM"].mean()),
    }


def build_candidates(config: RawFGConfig) -> List[Candidate]:
    common = {
        "optimizer_scope": "full_policy",
        "qp_fd_eps": 1e-3,
        "qp_beta_probe": 1e-3,
        "qp_gamma_probe": 1e-6,
        "qp_ridge": 1e-8,
        "qp_actor_weight": 1.0,
        "qp_logstd_weight": 1.0,
        "qp_critic_weight": 0.3,
        "qp_beta_max": config.beta_max,
        "qp_gamma_max": config.gamma_max,
        "qp_max_update_norm": float("inf"),
        "qp_eps": 1e-8,
    }
    return [
        Candidate("proposed_noG_rawFG", "proposed_noG_rawFG", 1.0, 0.5, 0.5, dict(common)),
        Candidate("proposed_qp_rawFG", "proposed_qp_rawFG", 1.0, 0.5, 0.5, dict(common)),
    ]


def ensure_new_run(candidate: Candidate, args: argparse.Namespace, run_root: pathlib.Path, iterations: int) -> pathlib.Path:
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
        str(iterations),
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
        [args.python_path, "scripts/analyze_rarl_run.py", "--run-dir", str(latest_run_dir), "--output-dir", str(analysis_dir), "--method", candidate.method],
        cwd=pathlib.Path(args.repo_dir),
        stdout_path=analysis_dir / "analyze_stdout.txt",
        stderr_path=analysis_dir / "analyze_stderr.txt",
    )
    return latest_run_dir


def load_analysis_frame(path: pathlib.Path, method: str, config_label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["method"] = method
    frame["config_label"] = config_label
    return frame


def load_run_summary(analysis_dir: pathlib.Path, method: str, config_label: str, iterations: int) -> Dict[str, object]:
    row = pd.read_csv(analysis_dir / "run_summary.csv").iloc[0].to_dict()
    row["method"] = method
    row["config_label"] = config_label
    row["iterations_run"] = iterations
    return row


def collect_diag_frame(run_dir: pathlib.Path, method: str, config_label: str, eta_egm: float) -> pd.DataFrame:
    frames = []
    for path in sorted(run_dir.glob("*diagnostics.csv")):
        df = pd.read_csv(path)
        df["method"] = method
        df["config_label"] = config_label
        df["beta_over_eta_EGM"] = df["beta"] / eta_egm if "beta" in df.columns else np.nan
        df["gamma_over_eta_EGM_squared"] = df["gamma"] / (eta_egm ** 2) if "gamma" in df.columns else np.nan
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_short_horizon(training_df: pd.DataFrame, clean_df: pd.DataFrame, adv_df: pd.DataFrame, horizon_timesteps: int) -> Dict[str, float]:
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
        "final_training_return_short": safe_last(training_cut, "episode_return"),
        "last5_training_mean_short": safe_last_mean(training_cut, "episode_return"),
        "final_clean_return_short": safe_last(clean_cut, "mean_reward"),
        "last5_clean_mean_short": safe_last_mean(clean_cut, "mean_reward"),
        "final_adversarial_return_short": safe_last(adv_cut, "mean_reward"),
        "last5_adversarial_mean_short": safe_last_mean(adv_cut, "mean_reward"),
    }


def plot_all(
    *,
    training_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    adv_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    diag_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    plots_dir: pathlib.Path,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "adam": "tab:orange",
        "sgd": "tab:gray",
        "egm": "tab:green",
        "ppm": "tab:red",
        "proposed_noG_rawFG": "tab:purple",
        "proposed_qp_rawFG": "tab:blue",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in training_df.groupby("method"):
        ax.plot(group["cumulative_timesteps"], group["episode_return"], label=method, color=colors.get(method))
    ax.set_title("Stage R6 Training Return")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode return")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stageR6_training_return.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in clean_df.groupby("method"):
        ax.plot(group["timesteps"], group["mean_reward"], label=method, color=colors.get(method))
        ax.fill_between(group["timesteps"], group["mean_reward"] - group["std_reward"], group["mean_reward"] + group["std_reward"], color=colors.get(method), alpha=0.15)
    ax.set_title("Stage R6 Clean Eval")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stageR6_clean_eval.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in adv_df.groupby("method"):
        ax.plot(group["timesteps"], group["mean_reward"], label=method, color=colors.get(method))
        ax.fill_between(group["timesteps"], group["mean_reward"] - group["std_reward"], group["mean_reward"] + group["std_reward"], color=colors.get(method), alpha=0.15)
    ax.set_title("Stage R6 Control Adversarial Eval")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stageR6_control_adv_eval.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in sweep_df.groupby("method"):
        ax.plot(group["adv_strength"], group["mean_return"], marker="o", label=method, color=colors.get(method))
    ax.set_title("Stage R6 Control Robustness Sweep")
    ax.set_xlabel("Adversary strength")
    ax.set_ylabel("Mean return")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stageR6_robustness_sweep.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    proposed_diag = diag_df[diag_df["method"].isin(["proposed_noG_rawFG", "proposed_qp_rawFG"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for method, group in proposed_diag.groupby("method"):
        axes[0].plot(group.index, group["beta_over_eta_EGM"], label=f"{method} beta/eta", color=colors.get(method))
        axes[0].plot(group.index, group["gamma_over_eta_EGM_squared"], linestyle="--", label=f"{method} gamma/eta^2", color=colors.get(method))
        axes[1].plot(group.index, group["gamma_active_frac"], label=f"{method} gamma_active", color=colors.get(method))
        axes[1].plot(group.index, group["G_contribution_norm"], linestyle="--", label=f"{method} G_contrib", color=colors.get(method))
    axes[0].set_title("RawFG beta / gamma diagnostics")
    axes[1].set_title("RawFG gamma activity")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlabel("Diagnostic row")
    fig.tight_layout()
    fig.savefig(plots_dir / "stageR6_beta_gamma.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for method, group in proposed_diag.groupby("method"):
        axes[0].plot(group.index, group["actual_V_change"], label=method, color=colors.get(method))
        axes[1].plot(group.index, group["update_norm_post_cap"], label=method, color=colors.get(method))
    axes[0].set_title("RawFG actual V change")
    axes[1].set_title("RawFG update norm")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_xlabel("Diagnostic row")
    fig.tight_layout()
    fig.savefig(plots_dir / "stageR6_V_change.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in proposed_diag.groupby("method"):
        ax.plot(group.index, group["actor_update_norm"], label=f"{method} actor", color=colors.get(method))
        ax.plot(group.index, group["critic_update_norm"], linestyle="--", label=f"{method} critic", color=colors.get(method))
    ax.set_title("Stage R6 update norms by block")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "stageR6_update_norms.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    final_rows = summary_df[["method", "last5_clean_mean", "last5_adversarial_mean"]].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(final_rows))
    width = 0.35
    ax.bar(x - width / 2, final_rows["last5_clean_mean"], width=width, label="clean")
    ax.bar(x + width / 2, final_rows["last5_adversarial_mean"], width=width, label="control-adv")
    ax.set_xticks(x)
    ax.set_xticklabels(final_rows["method"], rotation=25, ha="right")
    ax.set_title("Stage R6 Final last5 means")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stageR6_final_bar.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_collage(plots_dir: pathlib.Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    files = [
        "stageR6_training_return.png",
        "stageR6_clean_eval.png",
        "stageR6_control_adv_eval.png",
        "stageR6_robustness_sweep.png",
        "stageR6_beta_gamma.png",
        "stageR6_V_change.png",
        "stageR6_update_norms.png",
        "stageR6_final_bar.png",
    ]
    images = [(name, Image.open(plots_dir / name).convert("RGB")) for name in files]
    cell_w, cell_h, header_h, pad = 1600, 950, 70, 24
    cols = 2
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_w + (cols + 1) * pad, rows * (cell_h + header_h) + (rows + 1) * pad), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, (name, img) in enumerate(images):
        row = idx // cols
        col = idx % cols
        x0 = pad + col * cell_w
        y0 = pad + row * (cell_h + header_h)
        draw.text((x0 + 8, y0 + 8), f"{idx+1}. {name}", fill="black", font=font)
        inner = img.copy()
        inner.thumbnail((cell_w, cell_h))
        ix = x0 + (cell_w - inner.width) // 2
        iy = y0 + header_h + (cell_h - inner.height) // 2
        canvas.paste(inner, (ix, iy))
        draw.rectangle([x0, y0 + header_h, x0 + cell_w - 1, y0 + header_h + cell_h - 1], outline="lightgray", width=2)
    canvas.save(plots_dir / "stageR6_all_plots.png")


def main() -> None:
    args = parse_args()
    repo_dir = pathlib.Path(args.repo_dir)
    baseline_root = pathlib.Path(args.baseline_root)
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    baseline_data = load_baseline_frames(baseline_root)
    sweep_root = baseline_root.parent
    baseline_sweep = pd.read_csv(sweep_root / "control_robustness_sweep.csv")

    configs = [
        RawFGConfig("A", 1e-2, 3e-5),
        RawFGConfig("B", 3e-3, 1e-5),
    ]

    all_rows: List[Dict[str, object]] = []
    all_training = []
    all_clean = []
    all_adv = []
    all_sweep = [baseline_sweep.copy()]
    all_diag = []

    final_choice: RawFGConfig | None = None
    final_choice_extended = False

    for config in configs:
        preflight_dir = output_root / "preflight_runs" / config.label
        preflight_csv = run_rawfg_preflight(args, config, preflight_dir)
        gate = evaluate_preflight_gate(preflight_csv)

        all_rows.append(
            {
                "method": f"rawfg_preflight_{config.label}",
                "config_label": config.label,
                "beta_max": config.beta_max,
                "gamma_max": config.gamma_max,
                "preflight_pass": bool(gate["pass"]),
                "notes": str(gate["reason"]),
                "beta_qp_mean": gate.get("beta_qp_mean", np.nan),
                "gamma_qp_mean": gate.get("gamma_qp_mean", np.nan),
                "beta_over_eta_mean": gate.get("beta_over_eta_mean", np.nan),
                "gamma_over_eta2_mean": gate.get("gamma_over_eta2_mean", np.nan),
                "qp_v_mean": gate.get("qp_v_mean", np.nan),
                "nog_v_mean": gate.get("nog_v_mean", np.nan),
                "gamma_active_frac": gate.get("gamma_active_frac", np.nan),
                "G_contribution_norm_mean": gate.get("g_contribution_norm_mean", np.nan),
                "approx_kl_QP_max": gate.get("kl_max", np.nan),
                "clip_fraction_QP_max": gate.get("clip_max", np.nan),
                "update_norm_QP_mean": gate.get("update_norm_mean", np.nan),
            }
        )

        if not gate["pass"]:
            if config.label == "A":
                continue
            break

        run_root = output_root / "stageR6_runs_seed0" / config.label
        for candidate in build_candidates(config):
            method_root = run_root / candidate.method / "short"
            latest_run_dir = ensure_new_run(candidate, args, method_root, args.short_iterations)
            analysis_dir = method_root / "analysis"
            summary = load_run_summary(analysis_dir, candidate.method, config.label, args.short_iterations)
            summary["beta_max"] = config.beta_max
            summary["gamma_max"] = config.gamma_max
            summary["preflight_pass"] = True
            summary["phase"] = "short"
            all_rows.append(summary)
            all_training.append(load_analysis_frame(analysis_dir / "training_episode_returns.csv", candidate.method, config.label))
            all_clean.append(load_analysis_frame(analysis_dir / "clean_eval_returns.csv", candidate.method, config.label))
            all_adv.append(load_analysis_frame(analysis_dir / "adversarial_eval_returns.csv", candidate.method, config.label))
            sweep = evaluate_control_strength_sweep(latest_run_dir, candidate.method, args.device, args.n_eval_episodes)
            sweep["config_label"] = config.label
            all_sweep.append(sweep)
            diag = collect_diag_frame(latest_run_dir, candidate.method, config.label, args.eta_egm)
            if not diag.empty:
                all_diag.append(diag)

        short_qp = next(row for row in all_rows if row.get("method") == "proposed_qp_rawFG" and row.get("config_label") == config.label and row.get("phase") == "short")
        short_nog = next(row for row in all_rows if row.get("method") == "proposed_noG_rawFG" and row.get("config_label") == config.label and row.get("phase") == "short")
        qp_better_than_nog = (
            float(short_qp["last5_clean_mean"]) > float(short_nog["last5_clean_mean"])
            and float(short_qp["last5_adversarial_mean"]) > float(short_nog["last5_adversarial_mean"])
            and int(short_qp.get("crash_flag", 0)) == 0
            and int(short_qp.get("nan_flag", 0)) == 0
        )
        if qp_better_than_nog and config.label == "A":
            final_choice = config
            final_choice_extended = True
            for candidate in build_candidates(config):
                method_root = run_root / candidate.method / "long"
                latest_run_dir = ensure_new_run(candidate, args, method_root, args.long_iterations)
                analysis_dir = method_root / "analysis"
                summary = load_run_summary(analysis_dir, candidate.method, config.label, args.long_iterations)
                summary["beta_max"] = config.beta_max
                summary["gamma_max"] = config.gamma_max
                summary["preflight_pass"] = True
                summary["phase"] = "long"
                all_rows.append(summary)
                all_training.append(load_analysis_frame(analysis_dir / "training_episode_returns.csv", candidate.method, config.label))
                all_clean.append(load_analysis_frame(analysis_dir / "clean_eval_returns.csv", candidate.method, config.label))
                all_adv.append(load_analysis_frame(analysis_dir / "adversarial_eval_returns.csv", candidate.method, config.label))
                sweep = evaluate_control_strength_sweep(latest_run_dir, candidate.method, args.device, args.n_eval_episodes)
                sweep["config_label"] = config.label
                all_sweep.append(sweep)
                diag = collect_diag_frame(latest_run_dir, candidate.method, config.label, args.eta_egm)
                if not diag.empty:
                    all_diag.append(diag)
            break
        if config.label == "A" and not qp_better_than_nog:
            continue
        if config.label == "B":
            final_choice = config
            break

    # Build final comparison using frozen baselines + chosen config short or long
    final_rows = []
    for method, data in baseline_data.items():
        row = data["summary"].iloc[0].to_dict()
        row["method"] = method
        row["config_label"] = "frozen_baseline"
        row["phase"] = "frozen"
        final_rows.append(row)
        all_training.append(load_analysis_frame(baseline_root / method / "analysis" / "training_episode_returns.csv", method, "frozen_baseline"))
        all_clean.append(load_analysis_frame(baseline_root / method / "analysis" / "clean_eval_returns.csv", method, "frozen_baseline"))
        all_adv.append(load_analysis_frame(baseline_root / method / "analysis" / "adversarial_eval_returns.csv", method, "frozen_baseline"))

    rows_df = pd.DataFrame(all_rows)
    if final_choice is not None:
        phase_to_take = "long" if final_choice_extended else "short"
        for method in ["proposed_noG_rawFG", "proposed_qp_rawFG"]:
            match = rows_df[(rows_df["method"] == method) & (rows_df["config_label"] == final_choice.label) & (rows_df["phase"] == phase_to_take)]
            if not match.empty:
                final_rows.append(match.iloc[-1].to_dict())

    summary_df = pd.DataFrame(final_rows)
    training_df = pd.concat(all_training, ignore_index=True) if all_training else pd.DataFrame()
    clean_df = pd.concat(all_clean, ignore_index=True) if all_clean else pd.DataFrame()
    adv_df = pd.concat(all_adv, ignore_index=True) if all_adv else pd.DataFrame()
    sweep_df = pd.concat(all_sweep, ignore_index=True) if all_sweep else pd.DataFrame()
    diag_df = pd.concat(all_diag, ignore_index=True) if all_diag else pd.DataFrame()

    rows_df.to_csv(output_root / "stageR6_online_summary.csv", index=False)
    diag_df.to_csv(output_root / "stageR6_online_diagnostics.csv", index=False)
    training_df.to_csv(output_root / "stageR6_training_curves.csv", index=False)
    clean_df.to_csv(output_root / "stageR6_clean_eval_curves.csv", index=False)
    adv_df.to_csv(output_root / "stageR6_control_adv_eval_curves.csv", index=False)
    sweep_df.to_csv(output_root / "stageR6_robustness_sweep.csv", index=False)

    plot_all(
        training_df=training_df,
        clean_df=clean_df,
        adv_df=adv_df,
        sweep_df=sweep_df,
        diag_df=diag_df,
        summary_df=summary_df,
        plots_dir=plots_dir,
    )
    make_collage(plots_dir)

    report_lines = [
        "# Stage R6 Raw F/G Short Online Report",
        "",
        "- Protocol: `proper control-RARL`, `full_policy`, `seed=0`.",
        "- Baselines were reused from frozen control-RARL runs; they were not retrained or retuned here.",
        f"- Config A preflight pass: `{bool(rows_df[(rows_df['method'] == 'rawfg_preflight_A')]['preflight_pass'].iloc[0]) if not rows_df[(rows_df['method'] == 'rawfg_preflight_A')].empty else False}`",
    ]
    if not rows_df[(rows_df["method"] == "rawfg_preflight_B")].empty:
        report_lines.append(f"- Config B preflight pass: `{bool(rows_df[(rows_df['method'] == 'rawfg_preflight_B')]['preflight_pass'].iloc[0])}`")

    qp_final = summary_df[summary_df["method"] == "proposed_qp_rawFG"]
    nog_final = summary_df[summary_df["method"] == "proposed_noG_rawFG"]
    egm_final = summary_df[summary_df["method"] == "egm"]
    ppm_final = summary_df[summary_df["method"] == "ppm"]
    sgd_final = summary_df[summary_df["method"] == "sgd"]
    if not qp_final.empty and not nog_final.empty:
        qrow = qp_final.iloc[-1]
        nrow = nog_final.iloc[-1]
        erow = egm_final.iloc[-1]
        prow = ppm_final.iloc[-1]
        srow = sgd_final.iloc[-1]
        report_lines.extend(
            [
                f"- 1. Does proposed_qp_rawFG beat proposed_noG_rawFG? `{bool(float(qrow['last5_clean_mean']) > float(nrow['last5_clean_mean']) and float(qrow['last5_adversarial_mean']) > float(nrow['last5_adversarial_mean']))}`",
                f"- 2. Does proposed_qp_rawFG beat EGM on clean eval? `{bool(float(qrow['last5_clean_mean']) > float(erow['last5_clean_mean']))}`",
                f"- 3. Does proposed_qp_rawFG beat EGM on control adversarial eval? `{bool(float(qrow['last5_adversarial_mean']) > float(erow['last5_adversarial_mean']))}`",
                f"- 4. Does proposed_qp_rawFG beat PPM? `{bool(float(qrow['last5_clean_mean']) > float(prow['last5_clean_mean']) and float(qrow['last5_adversarial_mean']) > float(prow['last5_adversarial_mean']))}`",
                f"- 5. Does proposed_qp_rawFG clearly beat SGD? `{bool(float(qrow['last5_clean_mean']) > float(srow['last5_clean_mean']) and float(qrow['last5_adversarial_mean']) > float(srow['last5_adversarial_mean']))}`",
            ]
        )
        qdiag = diag_df[diag_df["method"] == "proposed_qp_rawFG"]
        if not qdiag.empty:
            report_lines.extend(
                [
                    f"- 6. Are beta and gamma usually above EGM-like coefficients? `{bool(float(qdiag['beta_over_eta_EGM'].mean()) > 1.0 and float(qdiag['gamma_over_eta_EGM_squared'].mean()) > 1.0)}`",
                    f"- 7. Is gamma actually useful online? `{bool(float(qdiag['gamma_active_frac'].mean()) > 0.0 and float(qdiag['G_contribution_norm'].mean()) > 0.0)}`",
                ]
            )
        failure_reasons = []
        if float(qrow["last5_clean_mean"]) <= float(erow["last5_clean_mean"]) or float(qrow["last5_adversarial_mean"]) <= float(erow["last5_adversarial_mean"]):
            if not qdiag.empty and float(qdiag["actual_V_change"].mean()) < 0.0:
                failure_reasons.append("A. V objective mismatch")
            if not qdiag.empty and (float(qdiag["beta_at_bound"].mean()) > 0.0 or float(qdiag["gamma_at_bound"].mean()) > 0.0):
                failure_reasons.append("B. beta/gamma too aggressive")
            if not qdiag.empty and float(qdiag["update_norm_post_cap"].mean()) < 1e-5:
                failure_reasons.append("C. beta/gamma too conservative")
            if not qdiag.empty and float(qdiag["critic_update_norm"].mean()) > float(qdiag["actor_update_norm"].mean()) * 2.0:
                failure_reasons.append("D. critic block dominates")
            if not qdiag.empty and float(qdiag["actual_V_change"].mean()) < 0.0 and float(qrow["last5_clean_mean"]) < float(nrow["last5_clean_mean"]):
                failure_reasons.append("E. G helps V but hurts return")
            if not failure_reasons:
                failure_reasons.append("F. EGM still better aligned with PPO return")
        report_lines.append(f"- 8. Failure classification if QP does not win: `{', '.join(failure_reasons) if failure_reasons else 'none'}`")
    (output_root / "stageR6_online_report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
