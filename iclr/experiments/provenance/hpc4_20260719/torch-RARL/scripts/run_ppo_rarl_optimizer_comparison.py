import argparse
import itertools
import json
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class Candidate:
    method: str
    optimizer: str
    lr: float
    optimizer_kwargs: Dict
    tag: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run PPO-RARL optimizer comparison on HalfCheetah-v4")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument("--python-path", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--screen-iters", type=int, default=1)
    parser.add_argument("--final-iters", type=int, default=5)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--n-eval-episodes", type=int, default=1)
    return parser.parse_args()


def build_candidates() -> List[Candidate]:
    candidates: List[Candidate] = []
    grids = {
        "adam": {"optimizer": "adam", "lrs": [2.0633e-05, 5.0e-05], "optimizer_kwargs_list": [{}]},
        "sgd": {"optimizer": "sgd", "lrs": [1.0e-04, 3.0e-04, 1.0e-03], "optimizer_kwargs_list": [{}, {"momentum": 0.9}]},
        "egm": {"optimizer": "egm", "lrs": [2.0633e-05, 5.0e-05, 1.0e-04], "optimizer_kwargs_list": [{}]},
        "ppm": {"optimizer": "ppm", "lrs": [2.0633e-05, 5.0e-05, 1.0e-04], "optimizer_kwargs_list": [{"inner_steps": 5}, {"inner_steps": 10}]},
    }
    for method, spec in grids.items():
        for lr, optimizer_kwargs in itertools.product(spec["lrs"], spec["optimizer_kwargs_list"]):
            tag = f"lr_{lr:.1e}".replace("+", "")
            if optimizer_kwargs:
                extras = "_".join(f"{key}_{value}" for key, value in sorted(optimizer_kwargs.items()))
                tag = f"{tag}_{extras}"
            candidates.append(Candidate(method=method, optimizer=spec["optimizer"], lr=lr, optimizer_kwargs=optimizer_kwargs, tag=tag))
    return candidates


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
    run_dirs = sorted(path for path in env_root.iterdir() if path.is_dir() and path.name.startswith(f"{env_id}_"))
    if len(run_dirs) != 1:
        raise RuntimeError(f"Expected one run dir under {env_root}, found {run_dirs}")
    return run_dirs[0]


def candidate_score(summary_path: pathlib.Path) -> float:
    summary = pd.read_csv(summary_path).iloc[0]
    return float(summary["last5_clean_mean"]) + float(summary["last5_adversarial_mean"])


def run_candidate(
    args: argparse.Namespace,
    candidate: Candidate,
    phase: str,
    iterations: int,
) -> pathlib.Path:
    run_root = pathlib.Path(args.output_root) / phase / candidate.method / candidate.tag
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
        "--protagonist-optimizer",
        candidate.optimizer,
        "--adversary-optimizer",
        candidate.optimizer,
        "--protagonist-lr",
        str(candidate.lr),
        "--adversary-lr",
        str(candidate.lr),
    ]
    for key, value in sorted(candidate.optimizer_kwargs.items()):
        command.extend(["--protagonist-optimizer-kwargs", f"{key}:{value}", "--adversary-optimizer-kwargs", f"{key}:{value}"])

    run_command(command, cwd=pathlib.Path(args.repo_dir), stdout_path=run_root / "stdout.txt", stderr_path=run_root / "stderr.txt")

    run_dir = find_run_dir(saved_models_dir, args.env)
    analysis_dir = run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_root / "stdout.txt", analysis_dir / "stdout.txt")
    shutil.copy2(run_root / "stderr.txt", analysis_dir / "stderr.txt")
    analyze_cmd = [
        args.python_path,
        "scripts/analyze_rarl_run.py",
        "--run-dir",
        str(run_dir),
        "--output-dir",
        str(analysis_dir),
        "--method",
        candidate.method,
    ]
    run_command(analyze_cmd, cwd=pathlib.Path(args.repo_dir), stdout_path=analysis_dir / "analyze_stdout.txt", stderr_path=analysis_dir / "analyze_stderr.txt")
    return analysis_dir


def aggregate_final_results(final_analysis_dirs: List[pathlib.Path], output_root: pathlib.Path) -> pd.DataFrame:
    frames = [pd.read_csv(path / "run_summary.csv") for path in final_analysis_dirs]
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(output_root / "optimizer_comparison_summary.csv", index=False)
    return summary


def plot_aggregate(final_analysis_dirs: List[pathlib.Path], output_root: pathlib.Path) -> None:
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    training_frames = [pd.read_csv(path / "training_episode_returns.csv") for path in final_analysis_dirs]
    clean_frames = [pd.read_csv(path / "clean_eval_returns.csv") for path in final_analysis_dirs]
    adv_frames = [pd.read_csv(path / "adversarial_eval_returns.csv") for path in final_analysis_dirs]
    norm_frames = [pd.read_csv(path / "parameter_norms.csv") for path in final_analysis_dirs]

    training_df = pd.concat(training_frames, ignore_index=True)
    clean_df = pd.concat(clean_frames, ignore_index=True)
    adv_df = pd.concat(adv_frames, ignore_index=True)
    norm_df = pd.concat(norm_frames, ignore_index=True)

    training_df.to_csv(output_root / "training_episode_returns_all_methods.csv", index=False)
    clean_df.to_csv(output_root / "clean_eval_returns_all_methods.csv", index=False)
    adv_df.to_csv(output_root / "adversarial_eval_returns_all_methods.csv", index=False)
    norm_df.to_csv(output_root / "parameter_norms_all_methods.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for method, group in training_df.groupby("method"):
        axes[0, 0].plot(group["cumulative_timesteps"], group["episode_return"], label=method, linewidth=1.1)
    axes[0, 0].set_title("Training Return by Optimizer")
    axes[0, 0].set_xlabel("Timesteps")
    axes[0, 0].set_ylabel("Episode Return")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    pro_norms = norm_df[norm_df["agent_name"] == "protagonist"]
    for method, group in pro_norms.groupby("method"):
        axes[0, 1].plot(group["num_timesteps"], group["total_param_norm"], label=method, linewidth=1.1)
    axes[0, 1].set_title("Protagonist Total Param Norm by Optimizer")
    axes[0, 1].set_xlabel("Timesteps")
    axes[0, 1].set_ylabel("Total Param Norm")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    for method, group in clean_df.groupby("method"):
        axes[1, 0].plot(group["timesteps"], group["mean_reward"], label=method, linewidth=1.2)
    axes[1, 0].set_title("Clean Eval Reward by Optimizer")
    axes[1, 0].set_xlabel("Timesteps")
    axes[1, 0].set_ylabel("Mean Reward")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    for method, group in adv_df.groupby("method"):
        axes[1, 1].plot(group["timesteps"], group["mean_reward"], label=method, linewidth=1.2)
    axes[1, 1].set_title("Adversarial Eval Reward by Optimizer")
    axes[1, 1].set_xlabel("Timesteps")
    axes[1, 1].set_ylabel("Mean Reward")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(plots_dir / "optimizer_comparison_4panel.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(output_root: pathlib.Path, summary_df: pd.DataFrame, final_config_map: Dict[str, Dict]) -> None:
    lines = [
        "# PPO-RARL Optimizer Comparison",
        "",
        "## Final configs",
        "",
        "```json",
        json.dumps(final_config_map, indent=2),
        "```",
        "",
        "## Final results",
        "",
    ]
    for _, row in summary_df.sort_values("method").iterrows():
        lines.extend(
            [
                f"- `{row['method']}` / `{row['optimizer_label']}`",
                f"  - protagonist_lr: `{row['protagonist_lr']}`",
                f"  - adversary_lr: `{row['adversary_lr']}`",
                f"  - final_clean_return: `{row['final_clean_return']:.6f}`",
                f"  - final_adversarial_return: `{row['final_adversarial_return']:.6f}`",
                f"  - best_clean_return: `{row['best_clean_return']:.6f}`",
                f"  - best_adversarial_return: `{row['best_adversarial_return']:.6f}`",
                f"  - crash_flag / nan_flag: `{int(row['crash_flag'])}` / `{int(row['nan_flag'])}`",
            ]
        )
    (output_root / "optimizer_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    candidates = build_candidates()
    best_candidates: Dict[str, Candidate] = {}

    for method in sorted({candidate.method for candidate in candidates}):
        method_candidates = [candidate for candidate in candidates if candidate.method == method]
        best_score = None
        best_candidate = None
        for candidate in method_candidates:
            analysis_dir = run_candidate(args, candidate, phase="screen", iterations=args.screen_iters)
            score = candidate_score(analysis_dir / "run_summary.csv")
            if best_score is None or score > best_score:
                best_score = score
                best_candidate = candidate
        best_candidates[method] = best_candidate

    final_analysis_dirs = []
    final_config_map = {}
    for method, candidate in best_candidates.items():
        analysis_dir = run_candidate(args, candidate, phase="final", iterations=args.final_iters)
        final_analysis_dirs.append(analysis_dir)
        final_config_map[method] = {
            "optimizer": candidate.optimizer,
            "lr": candidate.lr,
            "optimizer_kwargs": candidate.optimizer_kwargs,
            "tag": candidate.tag,
        }

    (output_root / "final_optimizer_configs.json").write_text(json.dumps(final_config_map, indent=2), encoding="utf-8")
    summary_df = aggregate_final_results(final_analysis_dirs, output_root)
    plot_aggregate(final_analysis_dirs, output_root)
    write_report(output_root, summary_df, final_config_map)


if __name__ == "__main__":
    main()
