import argparse
import json
import pathlib
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Analyze one RARL run with optimizer metadata")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the saved_models/.../<run_id> directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for processed csv/report/plots")
    parser.add_argument("--method", type=str, default=None, help="Optional method label override")
    return parser.parse_args()


def parse_monitor_csv(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=1)
    df = df.rename(columns={"r": "episode_return", "l": "episode_length", "t": "wall_time_seconds"})
    df["episode_idx"] = np.arange(1, len(df) + 1)
    df["cumulative_timesteps"] = df["episode_length"].cumsum()
    return df


def parse_training_returns(run_dir: pathlib.Path) -> pd.DataFrame:
    protagonist_path = run_dir / "analysis" / "protagonist_episode_returns.csv"
    if protagonist_path.exists():
        df = pd.read_csv(protagonist_path).rename(
            columns={"protagonist_timesteps": "cumulative_timesteps"}
        )
        required = {"cumulative_timesteps", "episode_return", "episode_length"}
        missing = required.difference(df.columns)
        if missing:
            raise RuntimeError(f"Missing protagonist-return columns {sorted(missing)} in {protagonist_path}")
        df = df.sort_values("cumulative_timesteps").reset_index(drop=True)
        df["episode_idx"] = np.arange(1, len(df) + 1)
        df["training_log_source"] = "protagonist_phase_callback"
        return df

    df = parse_monitor_csv(run_dir / "0.monitor.csv")
    df["training_log_source"] = "mixed_phase_monitor_fallback"
    return df


def parse_eval_npz(path: pathlib.Path, label: str) -> pd.DataFrame:
    data = np.load(path)
    rewards = data["results"]
    return pd.DataFrame(
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


def parse_optional_eval_npz(path: pathlib.Path, label: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "eval_type",
                "timesteps",
                "mean_reward",
                "std_reward",
                "min_reward",
                "max_reward",
                "mean_ep_length",
            ]
        )
    return parse_eval_npz(path, label)


def combine_param_norms(paths: List[pathlib.Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths if path.exists()]
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


def extract_learning_rate(config_value) -> float:
    if isinstance(config_value, dict):
        if "learning_rate" in config_value:
            return float(config_value["learning_rate"])
        return float("nan")
    config_string = str(config_value)
    match = re.search(r"learning_rate=([0-9eE.+-]+)", config_string)
    if match is None:
        return float("nan")
    return float(match.group(1))


def extract_numeric(config_value, key: str) -> float:
    if isinstance(config_value, dict):
        if key in config_value:
            return float(config_value[key])
        return float("nan")
    config_string = str(config_value)
    match = re.search(rf"{re.escape(key)}=([0-9eE.+-]+)", config_string)
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
    return int(any(frame.isna().any().any() for frame in frames))


def pick_field(primary, fallback):
    return primary if primary not in (None, "", "None") else fallback


def infer_metadata(args_data: Dict, config_data: Dict, method_override: str | None) -> Dict:
    protagonist_optimizer = pick_field(args_data.get("protagonist_optimizer"), config_data.get("protagonist_optimizer", "adam"))
    adversary_optimizer = pick_field(args_data.get("adversary_optimizer"), config_data.get("adversary_optimizer", "adam"))
    protagonist_lr = pick_field(args_data.get("protagonist_lr"), extract_learning_rate(config_data.get("protagonist_kwargs")))
    adversary_lr = pick_field(args_data.get("adversary_lr"), extract_learning_rate(config_data.get("adversary_kwargs")))
    protagonist_optimizer_kwargs = pick_field(args_data.get("protagonist_optimizer_kwargs"), config_data.get("protagonist_optimizer_kwargs", {}))
    adversary_optimizer_kwargs = pick_field(args_data.get("adversary_optimizer_kwargs"), config_data.get("adversary_optimizer_kwargs", {}))

    optimizer_label = protagonist_optimizer if protagonist_optimizer == adversary_optimizer else f"{protagonist_optimizer}+{adversary_optimizer}"
    method = method_override or optimizer_label
    ppm_inner_steps = None
    if protagonist_optimizer == "ppm":
        kwargs = protagonist_optimizer_kwargs if isinstance(protagonist_optimizer_kwargs, dict) else {}
        ppm_inner_steps = kwargs.get("inner_steps")

    return {
        "method": method,
        "optimizer_label": optimizer_label,
        "protagonist_optimizer": protagonist_optimizer,
        "adversary_optimizer": adversary_optimizer,
        "protagonist_lr": float(protagonist_lr),
        "adversary_lr": float(adversary_lr),
        "protagonist_optimizer_kwargs": json.dumps(protagonist_optimizer_kwargs, sort_keys=True),
        "adversary_optimizer_kwargs": json.dumps(adversary_optimizer_kwargs, sort_keys=True),
        "ppm_inner_steps": -1 if ppm_inner_steps is None else ppm_inner_steps,
    }


def add_metadata_columns(df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
    out = df.copy()
    for key, value in metadata.items():
        out[key] = value
    return out


def rolling_tail_mean(values: pd.Series, tail: int = 5) -> float:
    if len(values) == 0:
        return float("nan")
    return float(values.tail(min(tail, len(values))).mean())


def auc_trapezoid(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2:
        return float("nan")
    return float(np.trapezoid(y.to_numpy(), x.to_numpy()))


def parse_metric_series(text: str, metric_name: str) -> List[float]:
    pattern = re.compile(rf"\|\s+{re.escape(metric_name)}\s+\|\s+([-+0-9.eE]+)\s+\|")
    return [float(match.group(1)) for match in pattern.finditer(text)]


def write_summary(
    output_dir: pathlib.Path,
    args_data: Dict,
    config_data: Dict,
    metadata: Dict,
    training_df: pd.DataFrame,
    clean_eval_df: pd.DataFrame,
    adv_eval_df: pd.DataFrame,
    control_eval_df: pd.DataFrame,
    checkpoints_df: pd.DataFrame,
    stdout_text: str,
    stderr_text: str,
) -> pd.DataFrame:
    effective_n_mu = int(config_data["N_mu"]) if int(args_data["N_mu"]) < 0 else int(args_data["N_mu"])
    effective_n_nu = int(config_data["N_nu"]) if int(args_data["N_nu"]) < 0 else int(args_data["N_nu"])
    crash_flag = int("Traceback" in stdout_text or "Traceback" in stderr_text)
    nan_flag = contains_nan(training_df, clean_eval_df, adv_eval_df)
    if re.search(r"\bnan\b", stdout_text, flags=re.IGNORECASE) or re.search(r"\bnan\b", stderr_text, flags=re.IGNORECASE):
        nan_flag = 1

    protagonist_kwargs = config_data.get("protagonist_kwargs")
    adversary_kwargs = config_data.get("adversary_kwargs")
    protagonist_max_grad_norm = pick_field(args_data.get("protagonist_max_grad_norm"), extract_numeric(protagonist_kwargs, "max_grad_norm"))
    adversary_max_grad_norm = pick_field(args_data.get("adversary_max_grad_norm"), extract_numeric(adversary_kwargs, "max_grad_norm"))
    protagonist_vf_coef = pick_field(args_data.get("protagonist_vf_coef"), extract_numeric(protagonist_kwargs, "vf_coef"))
    adversary_vf_coef = pick_field(args_data.get("adversary_vf_coef"), extract_numeric(adversary_kwargs, "vf_coef"))

    metric_means = {}
    for metric_name, column_name in {
        "approx_kl": "approx_kl_mean",
        "clip_fraction": "clip_fraction_mean",
        "explained_variance": "explained_variance_mean",
        "value_loss": "value_loss_mean",
        "policy_gradient_loss": "policy_gradient_loss_mean",
        "entropy_loss": "entropy_loss_mean",
        "loss": "loss_mean",
    }.items():
        values = parse_metric_series(stdout_text, metric_name)
        metric_means[column_name] = float(np.mean(values)) if values else float("nan")

    summary = pd.DataFrame(
        [
            {
                "method": metadata["method"],
                "optimizer_label": metadata["optimizer_label"],
                "protagonist_optimizer": metadata["protagonist_optimizer"],
                "adversary_optimizer": metadata["adversary_optimizer"],
                "ppm_inner_steps": metadata["ppm_inner_steps"],
                "env_id": args_data["env"],
                "seed": args_data["seed"],
                "total_iterations": args_data["n_timesteps"],
                "N_mu": effective_n_mu,
                "N_nu": effective_n_nu,
                "adv_delay": args_data["adv_delay"],
                "adv_impact": args_data["adv_impact"],
                "protagonist_lr": metadata["protagonist_lr"],
                "adversary_lr": metadata["adversary_lr"],
                "protagonist_max_grad_norm": protagonist_max_grad_norm,
                "adversary_max_grad_norm": adversary_max_grad_norm,
                "protagonist_vf_coef": protagonist_vf_coef,
                "adversary_vf_coef": adversary_vf_coef,
                "final_training_return": float(training_df["episode_return"].iloc[-1]),
                "best_training_return": float(training_df["episode_return"].max()),
                "last5_training_mean": rolling_tail_mean(training_df["episode_return"]),
                "auc_training": auc_trapezoid(training_df["cumulative_timesteps"], training_df["episode_return"]),
                "final_clean_return": float(clean_eval_df["mean_reward"].iloc[-1]),
                "best_clean_return": float(clean_eval_df["mean_reward"].max()),
                "last5_clean_mean": rolling_tail_mean(clean_eval_df["mean_reward"]),
                "auc_clean": auc_trapezoid(clean_eval_df["timesteps"], clean_eval_df["mean_reward"]),
                "final_adversarial_return": float(adv_eval_df["mean_reward"].iloc[-1]),
                "best_adversarial_return": float(adv_eval_df["mean_reward"].max()),
                "last5_adversarial_mean": rolling_tail_mean(adv_eval_df["mean_reward"]),
                "auc_adversarial": auc_trapezoid(adv_eval_df["timesteps"], adv_eval_df["mean_reward"]),
                "final_control_proxy_return": float(control_eval_df["mean_reward"].iloc[-1]) if not control_eval_df.empty else float("nan"),
                "best_control_proxy_return": float(control_eval_df["mean_reward"].max()) if not control_eval_df.empty else float("nan"),
                "last5_control_proxy_mean": rolling_tail_mean(control_eval_df["mean_reward"]) if not control_eval_df.empty else float("nan"),
                "auc_control_proxy": auc_trapezoid(control_eval_df["timesteps"], control_eval_df["mean_reward"]) if len(control_eval_df) >= 2 else float("nan"),
                "num_training_episodes": int(len(training_df)),
                "num_clean_eval_points": int(len(clean_eval_df)),
                "num_adv_eval_points": int(len(adv_eval_df)),
                "num_control_proxy_eval_points": int(len(control_eval_df)),
                "num_checkpoints": int(len(checkpoints_df)),
                "crash_flag": crash_flag,
                "nan_flag": nan_flag,
                **metric_means,
            }
        ]
    )
    summary.to_csv(output_dir / "run_summary.csv", index=False)
    return summary


def make_plot(
    output_dir: pathlib.Path,
    metadata: Dict,
    training_df: pd.DataFrame,
    clean_eval_df: pd.DataFrame,
    adv_eval_df: pd.DataFrame,
    param_df: pd.DataFrame,
    control_eval_df: pd.DataFrame,
) -> pathlib.Path:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    title_suffix = f"{metadata['method']} ({metadata['optimizer_label']})"

    axes[0, 0].plot(training_df["cumulative_timesteps"], training_df["episode_return"], color="tab:blue", linewidth=1.0)
    axes[0, 0].set_title(f"Training Episode Return: {title_suffix}")
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
    axes[0, 1].set_title(f"Parameter Norms: {title_suffix}")
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
    axes[1, 0].set_title(f"Clean Eval Reward: {title_suffix}")
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
    axes[1, 1].set_title(f"Adversarial Eval Reward: {title_suffix}")
    axes[1, 1].set_xlabel("Timesteps")
    axes[1, 1].set_ylabel("Mean Eval Reward")
    axes[1, 1].grid(alpha=0.3)
    if not control_eval_df.empty:
        axes[1, 1].plot(
            control_eval_df["timesteps"],
            control_eval_df["mean_reward"],
            color="tab:purple",
            linewidth=1.2,
            linestyle="--",
            label="control-proxy",
        )
        axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    plot_path = plots_dir / f"{metadata['method']}_4panel.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def write_report(output_dir: pathlib.Path, summary_df: pd.DataFrame, plot_path: pathlib.Path, run_dir: pathlib.Path) -> None:
    row = summary_df.iloc[0]
    report = f"""# RARL Run Report

- Run directory: `{run_dir}`
- Method: `{row['method']}`
- Optimizer label: `{row['optimizer_label']}`
- Environment: `{row['env_id']}`
- Seed: `{int(row['seed'])}`
- Total iterations: `{int(row['total_iterations'])}`
- N_mu / N_nu: `{int(row['N_mu'])}` / `{int(row['N_nu'])}`
- Adversarial impact: `{row['adv_impact']}`
- Protagonist optimizer: `{row['protagonist_optimizer']}` @ `{row['protagonist_lr']}`
- Adversary optimizer: `{row['adversary_optimizer']}` @ `{row['adversary_lr']}`

## Outcome

- Final training return: `{row['final_training_return']:.6f}`
- Best training return: `{row['best_training_return']:.6f}`
- Final clean eval mean reward: `{row['final_clean_return']:.6f}`
- Best clean eval mean reward: `{row['best_clean_return']:.6f}`
- Final adversarial eval mean reward: `{row['final_adversarial_return']:.6f}`
- Best adversarial eval mean reward: `{row['best_adversarial_return']:.6f}`
- Final control-proxy eval mean reward: `{row['final_control_proxy_return']:.6f}`
- Best control-proxy eval mean reward: `{row['best_control_proxy_return']:.6f}`
- Crash flag: `{int(row['crash_flag'])}`
- NaN flag: `{int(row['nan_flag'])}`

## Plot

- 4-panel overview: [{plot_path.name}]({plot_path.as_posix()})
"""
    (output_dir / "run_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = pathlib.Path(args.run_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dir_candidates = [path for path in run_dir.iterdir() if path.is_dir() and (path / "args.yml").exists()]
    if len(config_dir_candidates) != 1:
        raise RuntimeError(f"Expected exactly one config directory under {run_dir}, found {config_dir_candidates}")
    config_dir = config_dir_candidates[0]

    args_data = yaml.load((config_dir / "args.yml").read_text(encoding="utf-8"), Loader=yaml.UnsafeLoader)
    config_data = yaml.load((config_dir / "config.yml").read_text(encoding="utf-8"), Loader=yaml.UnsafeLoader)
    metadata = infer_metadata(args_data, config_data, args.method)

    stdout_text = read_text_auto(output_dir / "stdout.txt")
    stderr_text = read_text_auto(output_dir / "stderr.txt")

    training_df = parse_training_returns(run_dir)
    clean_eval_df = parse_eval_npz(run_dir / "evaluations.npz", "clean")
    adv_eval_df = parse_eval_npz(run_dir / "adv_eval" / "evaluations.npz", "adversarial")
    control_eval_df = parse_optional_eval_npz(run_dir / "control_proxy_eval" / "evaluations.npz", "control-proxy")
    param_df = combine_param_norms(
        [
            run_dir / "analysis" / "protagonist_param_norms.csv",
            run_dir / "analysis" / "adversary_param_norms.csv",
        ]
    )
    checkpoints_df = count_checkpoints(run_dir)

    training_df = add_metadata_columns(training_df, metadata)
    clean_eval_df = add_metadata_columns(clean_eval_df, metadata)
    adv_eval_df = add_metadata_columns(adv_eval_df, metadata)
    control_eval_df = add_metadata_columns(control_eval_df, metadata)
    param_df = add_metadata_columns(param_df, metadata)
    checkpoints_df = add_metadata_columns(checkpoints_df, metadata)

    training_df.to_csv(output_dir / "training_episode_returns.csv", index=False)
    clean_eval_df.to_csv(output_dir / "clean_eval_returns.csv", index=False)
    adv_eval_df.to_csv(output_dir / "adversarial_eval_returns.csv", index=False)
    control_eval_df.to_csv(output_dir / "control_proxy_eval_returns.csv", index=False)
    param_df.to_csv(output_dir / "parameter_norms.csv", index=False)
    checkpoints_df.to_csv(output_dir / "checkpoint_inventory.csv", index=False)

    summary_df = write_summary(
        output_dir,
        args_data=args_data,
        config_data=config_data,
        metadata=metadata,
        training_df=training_df,
        clean_eval_df=clean_eval_df,
        adv_eval_df=adv_eval_df,
        control_eval_df=control_eval_df,
        checkpoints_df=checkpoints_df,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
    )
    plot_path = make_plot(output_dir, metadata, training_df, clean_eval_df, adv_eval_df, param_df, control_eval_df)
    write_report(output_dir, summary_df, plot_path, run_dir)


if __name__ == "__main__":
    main()
