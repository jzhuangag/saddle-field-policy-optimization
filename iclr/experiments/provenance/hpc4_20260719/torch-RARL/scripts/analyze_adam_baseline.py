import argparse
import csv
import pathlib
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Analyze compatible Adam baseline outputs")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the saved_models/.../<run_id> directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for processed csv/report/plots")
    return parser.parse_args()


def parse_monitor_csv(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=1)
    df = df.rename(columns={"r": "episode_return", "l": "episode_length", "t": "wall_time_seconds"})
    df["episode_idx"] = np.arange(1, len(df) + 1)
    df["cumulative_timesteps"] = df["episode_length"].cumsum()
    return df


def parse_eval_npz(path: pathlib.Path, label: str) -> pd.DataFrame:
    data = np.load(path)
    rewards = data["results"]
    df = pd.DataFrame(
        {
            "eval_type": label,
            "timesteps": data["timesteps"],
            "mean_reward": rewards.mean(axis=1),
            "std_reward": rewards.std(axis=1),
            "min_reward": rewards.min(axis=1),
            "max_reward": rewards.max(axis=1),
            "mean_ep_length": data["ep_lengths"].mean(axis=1),
        }
    )
    return df


def combine_param_norms(paths: List[pathlib.Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame(
            columns=[
                "agent_name",
                "rollout_index",
                "num_timesteps",
                "actor_param_norm",
                "critic_param_norm",
                "log_std_norm",
                "total_param_norm",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def extract_learning_rate(config_string: str) -> float:
    match = re.search(r"learning_rate=([0-9eE.+-]+)", config_string)
    if match is None:
        return float("nan")
    return float(match.group(1))


def count_checkpoints(run_dir: pathlib.Path) -> pd.DataFrame:
    rows = []
    for checkpoint in sorted(run_dir.glob("rl_model_*_steps.zip")):
        match = re.search(r"rl_model_(\d+)_steps\.zip", checkpoint.name)
        timesteps = int(match.group(1)) if match else -1
        rows.append({"checkpoint_file": checkpoint.name, "timesteps": timesteps, "bytes": checkpoint.stat().st_size})
    return pd.DataFrame(rows)


def read_text_auto(path: pathlib.Path) -> str:
    for encoding in ("utf-8", "utf-16", "utf-16-le", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def contains_nan(*frames: pd.DataFrame) -> int:
    for frame in frames:
        if frame.isna().any().any():
            return 1
    return 0


def write_summary(
    output_dir: pathlib.Path,
    args_data: Dict,
    config_data: Dict,
    training_df: pd.DataFrame,
    clean_eval_df: pd.DataFrame,
    adv_eval_df: pd.DataFrame,
    checkpoints_df: pd.DataFrame,
    stdout_text: str,
    stderr_text: str,
) -> pd.DataFrame:
    protagonist_lr = extract_learning_rate(config_data["protagonist_kwargs"])
    adversary_lr = extract_learning_rate(config_data["adversary_kwargs"])
    effective_n_mu = int(config_data["N_mu"]) if int(args_data["N_mu"]) < 0 else int(args_data["N_mu"])
    effective_n_nu = int(config_data["N_nu"]) if int(args_data["N_nu"]) < 0 else int(args_data["N_nu"])
    crash_flag = int("Traceback" in stdout_text or "Traceback" in stderr_text)
    nan_flag = contains_nan(training_df, clean_eval_df, adv_eval_df)
    if re.search(r"\bnan\b", stdout_text, flags=re.IGNORECASE) or re.search(r"\bnan\b", stderr_text, flags=re.IGNORECASE):
        nan_flag = 1

    summary = pd.DataFrame(
        [
            {
                "env_id": args_data["env"],
                "seed": args_data["seed"],
                "total_iterations": args_data["n_timesteps"],
                "N_mu": effective_n_mu,
                "N_nu": effective_n_nu,
                "adv_delay": args_data["adv_delay"],
                "adv_impact": args_data["adv_impact"],
                "protagonist_lr": protagonist_lr,
                "adversary_lr": adversary_lr,
                "final_training_return": float(training_df["episode_return"].iloc[-1]),
                "best_training_return": float(training_df["episode_return"].max()),
                "final_clean_return": float(clean_eval_df["mean_reward"].iloc[-1]),
                "best_clean_return": float(clean_eval_df["mean_reward"].max()),
                "final_adversarial_return": float(adv_eval_df["mean_reward"].iloc[-1]),
                "best_adversarial_return": float(adv_eval_df["mean_reward"].max()),
                "num_training_episodes": int(len(training_df)),
                "num_clean_eval_points": int(len(clean_eval_df)),
                "num_adv_eval_points": int(len(adv_eval_df)),
                "num_checkpoints": int(len(checkpoints_df)),
                "crash_flag": crash_flag,
                "nan_flag": nan_flag,
            }
        ]
    )
    summary_path = output_dir / "adam_baseline_summary.csv"
    summary.to_csv(summary_path, index=False)
    return summary


def make_plot(
    output_dir: pathlib.Path,
    training_df: pd.DataFrame,
    clean_eval_df: pd.DataFrame,
    adv_eval_df: pd.DataFrame,
    param_df: pd.DataFrame,
) -> pathlib.Path:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(training_df["cumulative_timesteps"], training_df["episode_return"], color="tab:blue", linewidth=1.0)
    axes[0, 0].set_title("Training Episode Return")
    axes[0, 0].set_xlabel("Timesteps")
    axes[0, 0].set_ylabel("Episode Return")
    axes[0, 0].grid(alpha=0.3)

    for agent_name, color_prefix in [("protagonist", "tab"), ("adversary", "dark")]:
        agent_df = param_df[param_df["agent_name"] == agent_name]
        if agent_df.empty:
            continue
        if agent_name == "protagonist":
            actor_color, critic_color, logstd_color = "tab:orange", "tab:green", "tab:red"
        else:
            actor_color, critic_color, logstd_color = "saddlebrown", "slateblue", "deeppink"
        axes[0, 1].plot(agent_df["num_timesteps"], agent_df["actor_param_norm"], label=f"{agent_name}_actor", linewidth=1.0, color=actor_color)
        axes[0, 1].plot(agent_df["num_timesteps"], agent_df["critic_param_norm"], label=f"{agent_name}_critic", linewidth=1.0, color=critic_color)
        axes[0, 1].plot(agent_df["num_timesteps"], agent_df["log_std_norm"], label=f"{agent_name}_log_std", linewidth=1.0, color=logstd_color)
    axes[0, 1].set_title("Parameter Norms During Training")
    axes[0, 1].set_xlabel("Timesteps")
    axes[0, 1].set_ylabel("L2 Norm")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(clean_eval_df["timesteps"], clean_eval_df["mean_reward"], color="tab:green", linewidth=1.5)
    axes[1, 0].fill_between(
        clean_eval_df["timesteps"],
        clean_eval_df["mean_reward"] - clean_eval_df["std_reward"],
        clean_eval_df["mean_reward"] + clean_eval_df["std_reward"],
        color="tab:green",
        alpha=0.2,
    )
    axes[1, 0].set_title("Clean Eval Reward by Checkpoint")
    axes[1, 0].set_xlabel("Timesteps")
    axes[1, 0].set_ylabel("Mean Eval Reward")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(adv_eval_df["timesteps"], adv_eval_df["mean_reward"], color="tab:red", linewidth=1.5)
    axes[1, 1].fill_between(
        adv_eval_df["timesteps"],
        adv_eval_df["mean_reward"] - adv_eval_df["std_reward"],
        adv_eval_df["mean_reward"] + adv_eval_df["std_reward"],
        color="tab:red",
        alpha=0.2,
    )
    axes[1, 1].set_title("Adversarial Eval Reward by Checkpoint")
    axes[1, 1].set_xlabel("Timesteps")
    axes[1, 1].set_ylabel("Mean Eval Reward")
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    plot_path = plots_dir / "adam_baseline_4panel.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def write_report(
    output_dir: pathlib.Path,
    summary_df: pd.DataFrame,
    plot_path: pathlib.Path,
    run_dir: pathlib.Path,
) -> None:
    row = summary_df.iloc[0]
    report = f"""# Adam Baseline Report

- Run directory: `{run_dir}`
- Environment: `{row['env_id']}`
- Seed: `{int(row['seed'])}`
- Total iterations: `{int(row['total_iterations'])}`
- N_mu / N_nu: `{int(row['N_mu'])}` / `{int(row['N_nu'])}`
- Adversarial impact: `{row['adv_impact']}`
- Protagonist learning rate: `{row['protagonist_lr']}`
- Adversary learning rate: `{row['adversary_lr']}`

## Outcome

- Final training return: `{row['final_training_return']:.6f}`
- Best training return: `{row['best_training_return']:.6f}`
- Final clean eval mean reward: `{row['final_clean_return']:.6f}`
- Best clean eval mean reward: `{row['best_clean_return']:.6f}`
- Final adversarial eval mean reward: `{row['final_adversarial_return']:.6f}`
- Best adversarial eval mean reward: `{row['best_adversarial_return']:.6f}`
- Number of training episodes logged: `{int(row['num_training_episodes'])}`
- Number of clean eval checkpoints: `{int(row['num_clean_eval_points'])}`
- Number of adversarial eval checkpoints: `{int(row['num_adv_eval_points'])}`
- Number of saved checkpoints: `{int(row['num_checkpoints'])}`
- Crash flag: `{int(row['crash_flag'])}`
- NaN flag: `{int(row['nan_flag'])}`

## Plot

- 4-panel overview: [{plot_path.name}]({plot_path.as_posix()})
"""
    (output_dir / "adam_baseline_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = pathlib.Path(args.run_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dir = run_dir / "HalfCheetah-v4"
    args_data = yaml.load((config_dir / "args.yml").read_text(encoding="utf-8"), Loader=yaml.UnsafeLoader)
    config_data = yaml.load((config_dir / "config.yml").read_text(encoding="utf-8"), Loader=yaml.UnsafeLoader)

    stdout_text = read_text_auto(output_dir / "baseline_stdout.txt")
    stderr_text = read_text_auto(output_dir / "baseline_stderr.txt")

    training_df = parse_monitor_csv(run_dir / "0.monitor.csv")
    clean_eval_df = parse_eval_npz(run_dir / "evaluations.npz", "clean")
    adv_eval_df = parse_eval_npz(run_dir / "adv_eval" / "evaluations.npz", "adversarial")
    param_df = combine_param_norms(
        [
            run_dir / "analysis" / "protagonist_param_norms.csv",
            run_dir / "analysis" / "adversary_param_norms.csv",
        ]
    )
    checkpoints_df = count_checkpoints(run_dir)

    training_df.to_csv(output_dir / "training_episode_returns.csv", index=False)
    clean_eval_df.to_csv(output_dir / "clean_eval_returns.csv", index=False)
    adv_eval_df.to_csv(output_dir / "adversarial_eval_returns.csv", index=False)
    param_df.to_csv(output_dir / "parameter_norms.csv", index=False)
    checkpoints_df.to_csv(output_dir / "checkpoint_inventory.csv", index=False)

    summary_df = write_summary(
        output_dir,
        args_data=args_data,
        config_data=config_data,
        training_df=training_df,
        clean_eval_df=clean_eval_df,
        adv_eval_df=adv_eval_df,
        checkpoints_df=checkpoints_df,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
    )
    plot_path = make_plot(output_dir, training_df, clean_eval_df, adv_eval_df, param_df)
    write_report(output_dir, summary_df, plot_path, run_dir)


if __name__ == "__main__":
    main()
