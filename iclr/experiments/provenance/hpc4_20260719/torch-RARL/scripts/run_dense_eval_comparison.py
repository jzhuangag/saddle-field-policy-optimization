from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.full_policy_followup_common import load_stage5_best_runs, rolling_mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 6B dense deterministic eval comparison")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument("--python-path", type=str, required=True)
    parser.add_argument("--stage5-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--eval-freq", type=int, default=2500)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    return parser.parse_args()


def run_command(command: List[str], cwd: pathlib.Path, stdout_path: pathlib.Path, stderr_path: pathlib.Path) -> None:
    result = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def find_run_dir(saved_models_dir: pathlib.Path, env_id: str) -> pathlib.Path:
    env_root = saved_models_dir / "rarl-ppo" / env_id
    return next(path for path in env_root.iterdir() if path.is_dir() and path.name.startswith(f"{env_id}_"))


def run_method(args: argparse.Namespace, method: str, spec: Dict) -> pathlib.Path:
    run_root = pathlib.Path(args.output_root) / "dense_runs" / method
    saved_models_dir = run_root / "saved_models"
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
        "force",
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
        "--protagonist-optimizer",
        spec["optimizer"],
        "--adversary-optimizer",
        spec["optimizer"],
        "--protagonist-lr",
        str(spec["lr"]),
        "--adversary-lr",
        str(spec["lr"]),
        "--protagonist-max-grad-norm",
        str(spec["max_grad_norm"]),
        "--adversary-max-grad-norm",
        str(spec["max_grad_norm"]),
        "--protagonist-vf-coef",
        str(spec["vf_coef"]),
        "--adversary-vf-coef",
        str(spec["vf_coef"]),
    ]
    for key, value in sorted(spec.get("optimizer_kwargs", {}).items()):
        command.extend(["--protagonist-optimizer-kwargs", f"{key}:{value}", "--adversary-optimizer-kwargs", f"{key}:{value}"])

    run_command(command, cwd=pathlib.Path(args.repo_dir), stdout_path=run_root / "stdout.txt", stderr_path=run_root / "stderr.txt")
    run_dir = find_run_dir(saved_models_dir, args.env)
    analysis_dir = run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_root / "stdout.txt", analysis_dir / "stdout.txt")
    shutil.copy2(run_root / "stderr.txt", analysis_dir / "stderr.txt")
    run_command(
        [args.python_path, "scripts/analyze_rarl_run.py", "--run-dir", str(run_dir), "--output-dir", str(analysis_dir), "--method", method],
        cwd=pathlib.Path(args.repo_dir),
        stdout_path=analysis_dir / "analyze_stdout.txt",
        stderr_path=analysis_dir / "analyze_stderr.txt",
    )
    return analysis_dir


def plot_dense_eval(output_root: pathlib.Path, summaries: pd.DataFrame, analysis_dirs: Dict[str, pathlib.Path]) -> None:
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    clean_frames = []
    adv_frames = []
    training_frames = []
    for method, analysis_dir in analysis_dirs.items():
        clean_df = pd.read_csv(analysis_dir / "clean_eval_returns.csv")
        adv_df = pd.read_csv(analysis_dir / "adversarial_eval_returns.csv")
        train_df = pd.read_csv(analysis_dir / "training_episode_returns.csv")
        clean_df["rolling_mean_reward"] = rolling_mean(clean_df["mean_reward"], window=3)
        adv_df["rolling_mean_reward"] = rolling_mean(adv_df["mean_reward"], window=3)
        train_df["rolling_episode_return"] = rolling_mean(train_df["episode_return"], window=15)
        clean_frames.append(clean_df)
        adv_frames.append(adv_df)
        training_frames.append(train_df)
    clean_all = pd.concat(clean_frames, ignore_index=True)
    adv_all = pd.concat(adv_frames, ignore_index=True)
    train_all = pd.concat(training_frames, ignore_index=True)
    clean_all.to_csv(output_root / "dense_clean_eval_all_methods.csv", index=False)
    adv_all.to_csv(output_root / "dense_adv_eval_all_methods.csv", index=False)
    train_all.to_csv(output_root / "dense_training_all_methods.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in clean_all.groupby("method"):
        ax.plot(group["timesteps"], group["mean_reward"], alpha=0.25, linewidth=1.0)
        ax.plot(group["timesteps"], group["rolling_mean_reward"], linewidth=2.0, label=method)
        ax.fill_between(group["timesteps"], group["mean_reward"] - group["std_reward"], group["mean_reward"] + group["std_reward"], alpha=0.12)
    ax.set_title("Dense Clean Eval")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "dense_clean_eval_all_methods.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in adv_all.groupby("method"):
        ax.plot(group["timesteps"], group["mean_reward"], alpha=0.25, linewidth=1.0)
        ax.plot(group["timesteps"], group["rolling_mean_reward"], linewidth=2.0, label=method)
        ax.fill_between(group["timesteps"], group["mean_reward"] - group["std_reward"], group["mean_reward"] + group["std_reward"], alpha=0.12)
    ax.set_title("Dense Adversarial Eval")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "dense_adv_eval_all_methods.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in train_all.groupby("method"):
        ax.plot(group["cumulative_timesteps"], group["episode_return"], alpha=0.2, linewidth=0.8)
        ax.plot(group["cumulative_timesteps"], group["rolling_episode_return"], linewidth=2.0, label=method)
    ax.set_title("Dense Training Return")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode return")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "dense_training_return_all_methods.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage5_runs = load_stage5_best_runs(pathlib.Path(args.stage5_root))
    final_configs = json.loads((pathlib.Path(args.stage5_root) / "final_full_policy_optimizer_configs.json").read_text(encoding="utf-8"))

    analysis_dirs: Dict[str, pathlib.Path] = {}
    rows = []
    for method in sorted(stage5_runs.keys()):
        spec = final_configs[method]
        analysis_dir = run_method(args, method, spec)
        analysis_dirs[method] = analysis_dir
        row = pd.read_csv(analysis_dir / "run_summary.csv").iloc[0].to_dict()
        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
    summary_df.to_csv(output_root / "dense_eval_summary.csv", index=False)
    plot_dense_eval(output_root, summary_df, analysis_dirs)


if __name__ == "__main__":
    main()
