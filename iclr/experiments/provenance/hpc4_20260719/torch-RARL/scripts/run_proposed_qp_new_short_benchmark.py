from __future__ import annotations

import argparse
import json
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
class Candidate:
    method: str
    optimizer: str
    lr: float
    max_grad_norm: float
    vf_coef: float
    optimizer_kwargs: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 6 short benchmark for proposed_qp_new")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument("--python-path", type=str, required=True)
    parser.add_argument("--baseline-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--eval-freq", type=int, default=2500)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    return parser.parse_args()


def render_kwargs_tokens(kwargs: Dict[str, object]) -> List[str]:
    tokens = []
    for key, value in sorted(kwargs.items()):
        if isinstance(value, float) and not np.isfinite(value):
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
    strengths = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    rows = []
    model, vec_env = load_rarl_for_eval(saved_run, adv_impact="control", adv_strength=1.0, device=device)
    for strength in strengths:
        set_rarl_eval_mode(model, vec_env, operating_mode="protagonist", adv_strength=strength)
        episode_rewards = []
        protagonist_action_norms = []
        adv_pre_norms = []
        adv_post_norms = []
        perturbation_norms = []
        clip_fractions = []
        obs = vec_env.reset()
        ep_reward = 0.0
        ep_pro_norm = []
        ep_adv_pre = []
        ep_adv_post = []
        ep_perturb = []
        ep_clip = []
        while len(episode_rewards) < n_eval_episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            ep_reward += float(rewards[0])
            info = infos[0]
            ep_pro_norm.append(float(info.get("protagonist_action_norm", 0.0)))
            ep_adv_pre.append(float(info.get("adversary_action_norm_pre_clip", 0.0)))
            ep_adv_post.append(float(info.get("adversary_action_norm_post_clip", 0.0)))
            ep_perturb.append(float(info.get("applied_control_perturbation_norm", 0.0)))
            ep_clip.append(float(info.get("action_clip_fraction", 0.0)))
            if bool(dones[0]):
                episode_rewards.append(ep_reward)
                protagonist_action_norms.append(float(np.mean(ep_pro_norm)) if ep_pro_norm else 0.0)
                adv_pre_norms.append(float(np.mean(ep_adv_pre)) if ep_adv_pre else 0.0)
                adv_post_norms.append(float(np.mean(ep_adv_post)) if ep_adv_post else 0.0)
                perturbation_norms.append(float(np.mean(ep_perturb)) if ep_perturb else 0.0)
                clip_fractions.append(float(np.mean(ep_clip)) if ep_clip else 0.0)
                obs = vec_env.reset()
                ep_reward = 0.0
                ep_pro_norm = []
                ep_adv_pre = []
                ep_adv_post = []
                ep_perturb = []
                ep_clip = []
        rows.append(
            {
                "method": method,
                "adv_strength": strength,
                "mean_return": float(np.mean(episode_rewards)),
                "std_return": float(np.std(episode_rewards)),
                "protagonist_action_norm": float(np.mean(protagonist_action_norms)),
                "adversary_action_norm_pre_clip": float(np.mean(adv_pre_norms)),
                "adversary_action_norm_post_clip": float(np.mean(adv_post_norms)),
                "applied_control_perturbation_norm": float(np.mean(perturbation_norms)),
                "action_clip_fraction": float(np.mean(clip_fractions)),
            }
        )
    vec_env.close()
    return pd.DataFrame(rows)


def load_analysis_frame(analysis_dir: pathlib.Path, filename: str, method: str) -> pd.DataFrame:
    frame = pd.read_csv(analysis_dir / filename)
    frame["method"] = method
    return frame


def load_baseline_frames(baseline_root: pathlib.Path, method: str) -> Dict[str, pd.DataFrame]:
    analysis_dir = baseline_root / method / "analysis"
    return {
        "summary": pd.read_csv(analysis_dir / "run_summary.csv"),
        "training": load_analysis_frame(analysis_dir, "training_episode_returns.csv", method),
        "clean": load_analysis_frame(analysis_dir, "clean_eval_returns.csv", method),
        "adv": load_analysis_frame(analysis_dir, "adversarial_eval_returns.csv", method),
        "run_dir": find_latest_run_dir(baseline_root / method / "saved_models", "HalfCheetah-v4"),
    }


def ensure_new_run(candidate: Candidate, args: argparse.Namespace, output_root: pathlib.Path) -> pathlib.Path:
    run_root = output_root / candidate.method
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

    sign_cfg = json.loads((output_root.parent / "stage5_selected_qp_sign.json").read_text(encoding="utf-8"))
    qp_sign = sign_cfg["chosen_sign"]
    optimizer_kwargs = dict(candidate.optimizer_kwargs)
    if candidate.method == "proposed_qp_new":
        optimizer_kwargs["qp_g_sign"] = qp_sign

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
    tokens = render_kwargs_tokens(optimizer_kwargs)
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


def save_plots(
    *,
    training_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    adv_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    plots_dir: pathlib.Path,
) -> None:
    colors = {
        "adam": "tab:orange",
        "sgd": "tab:gray",
        "egm": "tab:green",
        "ppm": "tab:red",
        "proposed_noG_new": "tab:purple",
        "proposed_qp_new": "tab:blue",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in training_df.groupby("method"):
        ax.plot(group["cumulative_timesteps"], group["episode_return"], label=method, color=colors.get(method))
    ax.set_title("Stage 6 Training Return")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode return")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stage6_training_return.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in clean_df.groupby("method"):
        ax.plot(group["timesteps"], group["mean_reward"], label=method, color=colors.get(method))
        ax.fill_between(group["timesteps"], group["mean_reward"] - group["std_reward"], group["mean_reward"] + group["std_reward"], color=colors.get(method), alpha=0.15)
    ax.set_title("Stage 6 Clean Eval")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stage6_clean_eval.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in adv_df.groupby("method"):
        ax.plot(group["timesteps"], group["mean_reward"], label=method, color=colors.get(method))
        ax.fill_between(group["timesteps"], group["mean_reward"] - group["std_reward"], group["mean_reward"] + group["std_reward"], color=colors.get(method), alpha=0.15)
    ax.set_title("Stage 6 Control Adversarial Eval")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stage6_adv_eval.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in sweep_df.groupby("method"):
        ax.plot(group["adv_strength"], group["mean_return"], marker="o", label=method, color=colors.get(method))
    ax.set_title("Stage 6 Control Robustness Sweep")
    ax.set_xlabel("Adversary strength")
    ax.set_ylabel("Mean return")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stage6_robustness_sweep.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if not diagnostics_df.empty:
        for method, group in diagnostics_df.groupby("method"):
            axes[0].plot(group.index, group["gamma_active_frac"], marker="o", label=method, color=colors.get(method))
            axes[1].plot(group.index, group["G_contribution_norm"], marker="o", label=method, color=colors.get(method))
    axes[0].set_title("Gamma active fraction")
    axes[1].set_title("G contribution norm")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_xlabel("Diagnostic row")
    fig.tight_layout()
    fig.savefig(plots_dir / "stage6_qp_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(summary_df))
    width = 0.35
    ax.bar(x - width / 2, summary_df["last5_clean_mean"], width=width, label="clean")
    ax.bar(x + width / 2, summary_df["last5_adversarial_mean"], width=width, label="control-adv")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["method"], rotation=25, ha="right")
    ax.set_title("Stage 6 Final last5 means")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stage6_final_bar.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_dir = pathlib.Path(args.repo_dir)
    baseline_root = pathlib.Path(args.baseline_root)
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    baseline_methods = ["adam", "sgd", "egm", "ppm"]
    rows = []
    training_frames = []
    clean_frames = []
    adv_frames = []
    sweep_frames = []

    baseline_run_dirs: Dict[str, pathlib.Path] = {}
    for method in baseline_methods:
        data = load_baseline_frames(baseline_root, method)
        rows.append(data["summary"].iloc[0].to_dict())
        training_frames.append(data["training"])
        clean_frames.append(data["clean"])
        adv_frames.append(data["adv"])
        baseline_run_dirs[method] = data["run_dir"]

    new_candidates = [
        Candidate(
            "proposed_noG_new",
            "proposed_noG_new",
            1e-3,
            10.0,
            0.5,
            {
                "optimizer_scope": "full_policy",
                "qp_normalization": "block",
                "qp_alpha": 0.3,
                "qp_beta_max": 1.0,
                "qp_gamma_max": 1.0,
                "qp_step_grid": "0,0.1,0.3,1.0,3.0",
                "qp_objective": "loss",
                "qp_accept_rule": "none",
                "qp_min_g_contribution": 0.0,
                "qp_critic_weight": 1.0,
                "qp_g_sign": "minus",
                "qp_max_update_norm": 1.0,
            },
        ),
        Candidate(
            "proposed_qp_new",
            "proposed_qp_new",
            1e-3,
            10.0,
            0.5,
            {
                "optimizer_scope": "full_policy",
                "qp_normalization": "block",
                "qp_alpha": 0.3,
                "qp_beta_max": 1.0,
                "qp_gamma_max": 1.0,
                "qp_step_grid": "0,0.1,0.3,1.0,3.0",
                "qp_objective": "loss",
                "qp_accept_rule": "none",
                "qp_min_g_contribution": 0.0,
                "qp_critic_weight": 1.0,
                "qp_g_sign": "minus",
                "qp_max_update_norm": 1.0,
            },
        ),
    ]

    diagnostics_frames = []
    run_dirs = dict(baseline_run_dirs)
    for candidate in new_candidates:
        run_dir = ensure_new_run(candidate, args, output_root / "stage6_runs_seed0")
        run_dirs[candidate.method] = run_dir
        analysis_dir = output_root / "stage6_runs_seed0" / candidate.method / "analysis"
        rows.append(pd.read_csv(analysis_dir / "run_summary.csv").iloc[0].to_dict())
        training_frames.append(load_analysis_frame(analysis_dir, "training_episode_returns.csv", candidate.method))
        clean_frames.append(load_analysis_frame(analysis_dir, "clean_eval_returns.csv", candidate.method))
        adv_frames.append(load_analysis_frame(analysis_dir, "adversarial_eval_returns.csv", candidate.method))
        for diag_name in [f"protagonist_{candidate.optimizer}_diagnostics.csv", f"adversary_{candidate.optimizer}_diagnostics.csv"]:
            diag_path = run_dir / diag_name
            if diag_path.exists():
                diag_df = pd.read_csv(diag_path)
                diag_df["method"] = candidate.method
                diagnostics_frames.append(diag_df)

    for method, run_dir in run_dirs.items():
        sweep_frames.append(evaluate_control_strength_sweep(run_dir, method, args.device, args.n_eval_episodes))

    summary_df = pd.DataFrame(rows)
    training_df = pd.concat(training_frames, ignore_index=True)
    clean_df = pd.concat(clean_frames, ignore_index=True)
    adv_df = pd.concat(adv_frames, ignore_index=True)
    sweep_df = pd.concat(sweep_frames, ignore_index=True)
    diagnostics_df = pd.concat(diagnostics_frames, ignore_index=True) if diagnostics_frames else pd.DataFrame()

    summary_df.to_csv(output_root / "stage6_short_summary.csv", index=False)
    training_df.to_csv(output_root / "stage6_training_curves.csv", index=False)
    clean_df.to_csv(output_root / "stage6_clean_eval_curves.csv", index=False)
    adv_df.to_csv(output_root / "stage6_adv_eval_curves.csv", index=False)
    sweep_df.to_csv(output_root / "stage6_robustness_sweep.csv", index=False)
    diagnostics_df.to_csv(output_root / "stage6_qp_diagnostics.csv", index=False)

    save_plots(
        training_df=training_df,
        clean_df=clean_df,
        adv_df=adv_df,
        sweep_df=sweep_df,
        diagnostics_df=diagnostics_df,
        summary_df=summary_df,
        plots_dir=plots_dir,
    )

    report_lines = [
        "# Stage 6 Short Benchmark Report",
        "",
        "- Protocol: `proper control-RARL`, `full_policy`, `seed=0`",
        "- Frozen baselines reused from Stage 9 control benchmark",
        "- New methods trained in this stage: `proposed_noG_new`, `proposed_qp_new`",
        "",
    ]
    row_map = {row["method"]: row for row in rows}
    report_lines.extend(
        [
            f"- proposed_qp_new > proposed_noG_new: `{row_map['proposed_qp_new']['last5_clean_mean'] > row_map['proposed_noG_new']['last5_clean_mean'] and row_map['proposed_qp_new']['last5_adversarial_mean'] > row_map['proposed_noG_new']['last5_adversarial_mean']}`",
            f"- proposed_qp_new > EGM: `{row_map['proposed_qp_new']['last5_clean_mean'] > row_map['egm']['last5_clean_mean'] and row_map['proposed_qp_new']['last5_adversarial_mean'] > row_map['egm']['last5_adversarial_mean']}`",
            f"- proposed_qp_new > PPM: `{row_map['proposed_qp_new']['last5_clean_mean'] > row_map['ppm']['last5_clean_mean'] and row_map['proposed_qp_new']['last5_adversarial_mean'] > row_map['ppm']['last5_adversarial_mean']}`",
            f"- proposed_qp_new > SGD: `{row_map['proposed_qp_new']['last5_clean_mean'] > row_map['sgd']['last5_clean_mean'] and row_map['proposed_qp_new']['last5_adversarial_mean'] > row_map['sgd']['last5_adversarial_mean']}`",
        ]
    )
    (output_root / "stage6_short_report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
