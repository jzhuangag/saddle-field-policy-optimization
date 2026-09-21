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
import yaml


@dataclass(frozen=True)
class MethodSpec:
    label: str
    optimizer: str
    optimizer_kwargs: Dict[str, object]
    lr: float | None = None
    max_grad_norm: float | None = None
    vf_coef: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("One-seed standard RARL QP check on HalfCheetah")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument("--python-path", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v5")
    parser.add_argument("--fallback-env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=2500)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--shared-lr", type=float, default=1e-3)
    parser.add_argument("--shared-max-grad-norm", type=float, default=10.0)
    parser.add_argument("--shared-vf-coef", type=float, default=0.5)
    parser.add_argument("--ppm-inner-steps", type=int, default=2)
    parser.add_argument("--adv-impact", type=str, default="control", choices=["control", "force"])
    parser.add_argument("--optimizer-scope", type=str, default="full_policy", choices=["full_policy", "actor_game"])
    parser.add_argument("--qp-normalization", type=str, default="global", choices=["none", "global", "block"])
    parser.add_argument("--qp-g-alpha", type=float, default=0.3)
    return parser.parse_args()


def run_command(command: List[str], cwd: pathlib.Path, stdout_path: pathlib.Path, stderr_path: pathlib.Path) -> None:
    result = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"COMMAND: {' '.join(command)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


def ensure_hyperparams(repo_dir: pathlib.Path, output_root: pathlib.Path, requested_env: str, fallback_env: str) -> pathlib.Path:
    source_path = repo_dir / "hyperparameter" / "PPO-rarl.yml"
    temp_dir = output_root / "temp_hyperparams"
    temp_dir.mkdir(parents=True, exist_ok=True)
    target_path = temp_dir / "PPO-rarl.yml"

    with source_path.open("r", encoding="utf-8") as handle:
        hyperparams = yaml.safe_load(handle)

    env_used = requested_env
    mapping_note = "native"
    if requested_env not in hyperparams:
        if fallback_env not in hyperparams:
            raise KeyError(f"Neither {requested_env} nor fallback {fallback_env} exist in {source_path}")
        hyperparams[requested_env] = hyperparams[fallback_env]
        env_used = requested_env
        mapping_note = f"copied_from_{fallback_env}"

    with target_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(hyperparams, handle, sort_keys=False)

    metadata = {
        "requested_env": requested_env,
        "fallback_env": fallback_env,
        "env_used": env_used,
        "mapping_note": mapping_note,
        "source_yaml": str(source_path),
        "target_yaml": str(target_path),
    }
    (temp_dir / "hyperparam_mapping.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return temp_dir


def build_method_specs(args: argparse.Namespace) -> List[MethodSpec]:
    return [
        MethodSpec(label="sgd_gda", optimizer="sgd", optimizer_kwargs={}),
        MethodSpec(label="egm", optimizer="egm", optimizer_kwargs={}),
        MethodSpec(label="ppm", optimizer="ppm", optimizer_kwargs={"inner_steps": args.ppm_inner_steps}),
        MethodSpec(
            label="qp_proposed",
            optimizer="proposed_qp_perfLyap",
            optimizer_kwargs={
                "perflyap_scope": "full_policy_actor_weighted",
                "lambda_N": 0.0,
                "lambda_P": 1.0,
                "lambda_critic": 1.0,
                "logstd_weight": 0.0,
                "qp_fd_eps": 0.001,
                "qp_beta_probe": 0.001,
                "qp_gamma_probe": 1.0e-06,
                "qp_ridge": 1.0e-08,
                "qp_rho": 1.0e-08,
                "qp_beta_max": 0.03,
                "qp_gamma_max": 3.0e-05,
                "qp_max_update_norm": 0.005,
                "qp_eps": 1.0e-08,
                "eta_egm_reference": 0.001,
                "g_sign_mode": "plus",
                "use_scale_normalization": False,
                "cost_mode": "mixed_clean_unclipped_actor_surrogate_cost",
                "selector_mode": "safe_fixed_minusg",
                "direction_mode": "egm_minus_JF_F",
                "fixed_beta_raw": 0.012,
                "fixed_gamma_raw": 2.4e-05,
                "selector_beta_grid": "0,0.009,0.012,0.015",
                "selector_gamma_grid": "0,1.2e-05,2.4e-05,3e-05",
                "ls_fit_variant": "LS_all_grid",
                "short_return_horizon": 16,
                "short_return_episodes": 1,
                "short_return_seed_offset": 0,
            },
            lr=1.0,
            max_grad_norm=1.0,
            vf_coef=1.0,
        ),
    ]


def kwargs_tokens(kwargs: Dict[str, object]) -> List[str]:
    tokens: List[str] = []
    for key, value in sorted(kwargs.items()):
        rendered = repr(value) if isinstance(value, str) else value
        tokens.append(f"{key}:{rendered}")
    return tokens


def find_latest_run_dir(saved_models_dir: pathlib.Path, env_id: str) -> pathlib.Path:
    env_root = saved_models_dir / "rarl-ppo" / env_id
    run_dirs = [path for path in env_root.iterdir() if path.is_dir() and path.name.startswith(f"{env_id}_")]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories under {env_root}")
    return max(run_dirs, key=lambda path: int(path.name.rsplit("_", 1)[-1]))


def load_frame(path: pathlib.Path, method: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["method"] = method
    return frame


def run_method(
    *,
    args: argparse.Namespace,
    repo_dir: pathlib.Path,
    hyperparam_dir: pathlib.Path,
    output_root: pathlib.Path,
    env_id: str,
    method: MethodSpec,
) -> Dict[str, object]:
    run_root = output_root / method.label
    analysis_dir = run_root / "analysis"
    saved_models_dir = run_root / "saved_models"
    logging_dir = run_root / "logging"
    tb_dir = run_root / "tb"
    run_root.mkdir(parents=True, exist_ok=True)

    method_lr = args.shared_lr if method.lr is None else float(method.lr)
    method_max_grad_norm = args.shared_max_grad_norm if method.max_grad_norm is None else float(method.max_grad_norm)
    method_vf_coef = args.shared_vf_coef if method.vf_coef is None else float(method.vf_coef)

    command = [
        args.python_path,
        "scripts/train_adversary.py",
        "--algo",
        "rarl",
        "--rarl-config",
        "ppo",
        "--env",
        env_id,
        "--adv-impact",
        args.adv_impact,
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
        "--hyperparameter-path",
        str(hyperparam_dir),
        "--optimizer-scope",
        args.optimizer_scope,
        "--protagonist-optimizer",
        method.optimizer,
        "--adversary-optimizer",
        method.optimizer,
        "--protagonist-lr",
        str(method_lr),
        "--adversary-lr",
        str(method_lr),
        "--protagonist-max-grad-norm",
        str(method_max_grad_norm),
        "--adversary-max-grad-norm",
        str(method_max_grad_norm),
        "--protagonist-vf-coef",
        str(method_vf_coef),
        "--adversary-vf-coef",
        str(method_vf_coef),
    ]
    opt_tokens = kwargs_tokens(method.optimizer_kwargs)
    if opt_tokens:
        command.extend(["--protagonist-optimizer-kwargs", *opt_tokens])
        command.extend(["--adversary-optimizer-kwargs", *opt_tokens])

    if (analysis_dir / "run_summary.csv").exists():
        summary = pd.read_csv(analysis_dir / "run_summary.csv").iloc[0].to_dict()
        training = load_frame(analysis_dir / "training_episode_returns.csv", method.label)
        clean = load_frame(analysis_dir / "clean_eval_returns.csv", method.label)
        adv = load_frame(analysis_dir / "adversarial_eval_returns.csv", method.label)
        degradation = pd.DataFrame(
            {
                "timesteps": clean["timesteps"],
                "clean_mean_reward": clean["mean_reward"],
                "robust_mean_reward": adv["mean_reward"],
                "robust_degradation": clean["mean_reward"] - adv["mean_reward"],
                "method": method.label,
            }
        )
        return {
            "command": command,
            "summary": summary,
            "training": training,
            "clean": clean,
            "adv": adv,
            "degradation": degradation,
            "run_dir": find_latest_run_dir(saved_models_dir, env_id),
            "analysis_dir": analysis_dir,
        }

    run_command(command, repo_dir, run_root / "stdout.txt", run_root / "stderr.txt")
    run_dir = find_latest_run_dir(saved_models_dir, env_id)

    analysis_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_root / "stdout.txt", analysis_dir / "stdout.txt")
    shutil.copy2(run_root / "stderr.txt", analysis_dir / "stderr.txt")
    run_command(
        [
            args.python_path,
            "scripts/analyze_rarl_run.py",
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(analysis_dir),
            "--method",
            method.label,
        ],
        repo_dir,
        analysis_dir / "analyze_stdout.txt",
        analysis_dir / "analyze_stderr.txt",
    )

    summary = pd.read_csv(analysis_dir / "run_summary.csv").iloc[0].to_dict()
    training = load_frame(analysis_dir / "training_episode_returns.csv", method.label)
    clean = load_frame(analysis_dir / "clean_eval_returns.csv", method.label)
    adv = load_frame(analysis_dir / "adversarial_eval_returns.csv", method.label)

    if "mean_reward" in clean and "mean_reward" in adv:
        degradation = pd.DataFrame(
            {
                "timesteps": clean["timesteps"],
                "clean_mean_reward": clean["mean_reward"],
                "robust_mean_reward": adv["mean_reward"],
                "robust_degradation": clean["mean_reward"] - adv["mean_reward"],
                "method": method.label,
            }
        )
    else:
        degradation = pd.DataFrame(columns=["timesteps", "clean_mean_reward", "robust_mean_reward", "robust_degradation", "method"])

    return {
        "command": command,
        "summary": summary,
        "training": training,
        "clean": clean,
        "adv": adv,
        "degradation": degradation,
        "run_dir": run_dir,
        "analysis_dir": analysis_dir,
    }


def is_curve_sane(summary_row: Dict[str, object]) -> bool:
    if int(summary_row.get("crash_flag", 1)) != 0 or int(summary_row.get("nan_flag", 1)) != 0:
        return False
    required = [
        "final_training_return",
        "best_training_return",
        "final_clean_return",
        "final_adversarial_return",
    ]
    for key in required:
        value = summary_row.get(key, np.nan)
        if not np.isfinite(value):
            return False
    return True


def auc_from_curve(frame: pd.DataFrame, x_col: str, y_col: str) -> float:
    if len(frame) < 2:
        return float("nan")
    return float(np.trapezoid(frame[y_col].to_numpy(), frame[x_col].to_numpy()))


def save_line_plot(frame: pd.DataFrame, x_col: str, y_col: str, ylabel: str, title: str, output_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in frame.groupby("method"):
        ax.plot(group[x_col], group[y_col], label=method, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_big_figure(training_df: pd.DataFrame, clean_df: pd.DataFrame, adv_df: pd.DataFrame, degradation_df: pd.DataFrame, output_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for method, group in training_df.groupby("method"):
        axes[0, 0].plot(group["cumulative_timesteps"], group["episode_return"], label=method)
    axes[0, 0].set_title("Train Return")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    for method, group in clean_df.groupby("method"):
        axes[0, 1].plot(group["timesteps"], group["mean_reward"], label=method)
    axes[0, 1].set_title("Clean Eval Return")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    for method, group in adv_df.groupby("method"):
        axes[1, 0].plot(group["timesteps"], group["mean_reward"], label=method)
    axes[1, 0].set_title("Robust Eval Return")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    for method, group in degradation_df.groupby("method"):
        axes[1, 1].plot(group["timesteps"], group["robust_degradation"], label=method)
    axes[1, 1].set_title("Robust Degradation")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_dir = pathlib.Path(args.repo_dir)
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir = output_root
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    hyperparam_dir = ensure_hyperparams(repo_dir, output_root, args.env, args.fallback_env)
    method_specs = build_method_specs(args)

    all_commands: Dict[str, List[str]] = {}
    rows: List[Dict[str, object]] = []
    training_frames: List[pd.DataFrame] = []
    clean_frames: List[pd.DataFrame] = []
    adv_frames: List[pd.DataFrame] = []
    degradation_frames: List[pd.DataFrame] = []

    for method in method_specs:
        result = run_method(
            args=args,
            repo_dir=repo_dir,
            hyperparam_dir=hyperparam_dir,
            output_root=output_root / "runs_seed0",
            env_id=args.env,
            method=method,
        )
        summary = dict(result["summary"])
        summary["method"] = method.label
        summary["mapped_optimizer"] = method.optimizer
        summary["shared_lr"] = args.shared_lr
        summary["shared_max_grad_norm"] = args.shared_max_grad_norm
        summary["shared_vf_coef"] = args.shared_vf_coef
        summary["method_lr"] = method.lr if method.lr is not None else args.shared_lr
        summary["method_max_grad_norm"] = method.max_grad_norm if method.max_grad_norm is not None else args.shared_max_grad_norm
        summary["method_vf_coef"] = method.vf_coef if method.vf_coef is not None else args.shared_vf_coef
        summary["curve_sane_flag"] = int(is_curve_sane(summary))
        summary["auc_train"] = auc_from_curve(result["training"], "cumulative_timesteps", "episode_return")
        summary["auc_eval_clean"] = auc_from_curve(result["clean"], "timesteps", "mean_reward")
        summary["auc_eval_robust"] = auc_from_curve(result["adv"], "timesteps", "mean_reward")
        summary["auc_robust_degradation"] = auc_from_curve(result["degradation"], "timesteps", "robust_degradation")
        rows.append(summary)
        training_frames.append(result["training"])
        clean_frames.append(result["clean"])
        adv_frames.append(result["adv"])
        degradation_frames.append(result["degradation"])
        all_commands[method.label] = result["command"]

        for name in ["stdout.txt", "stderr.txt"]:
            source = (output_root / "runs_seed0" / method.label / name)
            if source.exists():
                shutil.copy2(source, logs_dir / f"{method.label}_{name}")

    summary_df = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
    training_df = pd.concat(training_frames, ignore_index=True)
    clean_df = pd.concat(clean_frames, ignore_index=True)
    adv_df = pd.concat(adv_frames, ignore_index=True)
    degradation_df = pd.concat(degradation_frames, ignore_index=True)

    summary_df.to_csv(output_root / "summary.csv", index=False)
    training_df.to_csv(output_root / "train_curves.csv", index=False)
    clean_df.to_csv(output_root / "eval_clean_curves.csv", index=False)
    adv_df.to_csv(output_root / "eval_robust_curves.csv", index=False)
    degradation_df.to_csv(output_root / "robust_degradation_curves.csv", index=False)
    (output_root / "commands.json").write_text(json.dumps(all_commands, indent=2), encoding="utf-8")

    save_line_plot(training_df, "cumulative_timesteps", "episode_return", "Episode Return", "Training Return", output_root / "train_return.png")
    save_line_plot(clean_df, "timesteps", "mean_reward", "Mean Reward", "Clean Eval Return", output_root / "eval_clean_return.png")
    save_line_plot(adv_df, "timesteps", "mean_reward", "Mean Reward", "Robust Eval Return", output_root / "eval_robust_return.png")
    save_line_plot(degradation_df, "timesteps", "robust_degradation", "Clean - Robust", "Robust Degradation", output_root / "robust_degradation.png")
    save_big_figure(training_df, clean_df, adv_df, degradation_df, output_root / "all_plots_big.png")

    all_sane = bool(summary_df["curve_sane_flag"].all()) if not summary_df.empty else False
    robust_final = summary_df[["method", "final_adversarial_return"]].sort_values("final_adversarial_return", ascending=False)
    best_robust_method = robust_final.iloc[0]["method"] if not robust_final.empty else "unknown"
    sgd_robust = float(summary_df.loc[summary_df["method"] == "sgd_gda", "final_adversarial_return"].iloc[0]) if "sgd_gda" in set(summary_df["method"]) else float("nan")
    egm_robust = float(summary_df.loc[summary_df["method"] == "egm", "final_adversarial_return"].iloc[0]) if "egm" in set(summary_df["method"]) else float("nan")
    ppm_robust = float(summary_df.loc[summary_df["method"] == "ppm", "final_adversarial_return"].iloc[0]) if "ppm" in set(summary_df["method"]) else float("nan")
    qp_robust = float(summary_df.loc[summary_df["method"] == "qp_proposed", "final_adversarial_return"].iloc[0]) if "qp_proposed" in set(summary_df["method"]) else float("nan")

    egm_beats_sgd = bool(np.isfinite(egm_robust) and np.isfinite(sgd_robust) and egm_robust > sgd_robust)
    ppm_beats_sgd = bool(np.isfinite(ppm_robust) and np.isfinite(sgd_robust) and ppm_robust > sgd_robust)
    qp_beats_all = bool(
        np.isfinite(qp_robust)
        and np.isfinite(sgd_robust)
        and np.isfinite(egm_robust)
        and np.isfinite(ppm_robust)
        and qp_robust > max(sgd_robust, egm_robust, ppm_robust)
    )

    if not all_sane:
        decision = "BASELINES_NEED_TUNING"
    elif qp_beats_all:
        decision = "PROMISING_QP_STANDARD_RARL_ONE_SEED"
    else:
        decision = "QP_NOT_PROMISING_ONE_SEED"

    report_lines = [
        "# One-Seed Standard RARL QP Check",
        "",
        "This is a one-seed screening run only. Do not over-claim from it.",
        "",
        f"- Requested environment: `{args.env}`",
        f"- Actual environment used: `{args.env}`",
        f"- Hyperparameter mapping: `{json.loads((hyperparam_dir / 'hyperparam_mapping.json').read_text(encoding='utf-8'))['mapping_note']}`",
        f"- Seed: `{args.seed}`",
        f"- Adversary impact: `{args.adv_impact}`",
        f"- Shared lr: `{args.shared_lr}`",
        f"- Shared max_grad_norm: `{args.shared_max_grad_norm}`",
        f"- Shared vf_coef: `{args.shared_vf_coef}`",
        f"- PPM inner steps: `{args.ppm_inner_steps}`",
        f"- Optimizer scope: `{args.optimizer_scope}`",
        f"- QP mapping: `qp_proposed -> proposed_qp_perfLyap (safe_minusG-style native repo variant)`",
        f"- SGD/GDA mapping: `sgd_gda -> sgd`",
        "- Fairness caveat: baselines share `lr/max_grad_norm/vf_coef`, while the compatible QP mapping uses its native perfLyap step-scaling settings because legacy `proposed_qp` is not interface-compatible with the current PPO loop.",
        "",
        "## Questions",
        "",
        f"1. Did all baselines run without crashing? `{bool((summary_df['crash_flag'] == 0).all() and (summary_df['nan_flag'] == 0).all())}`",
        f"2. Are baseline curves sane / convergent / non-degenerate? `{all_sane}`",
        f"3. Does EGM or PPM outperform SGD/GDA? `EGM>{egm_beats_sgd}`, `PPM>{ppm_beats_sgd}`",
        f"4. Does QP/proposed outperform EGM/PPM/SGD on robust eval return? `{qp_beats_all}`",
        f"5. Is the result promising enough to justify multi-seed? `{decision == 'PROMISING_QP_STANDARD_RARL_ONE_SEED'}`",
        f"6. Any obvious instability, unfairness, or config mismatch? `{'No obvious crash/nan mismatch, but the QP mapping uses native perfLyap step scaling and is therefore not a perfectly apples-to-apples optimizer swap.' if all_sane else 'Yes: at least one method produced broken/non-sane curves under matched config.'}`",
        "",
        "## Final robust ranking",
        "",
    ]
    for _, row in robust_final.iterrows():
        report_lines.append(
            f"- `{row['method']}` final robust eval return = `{row['final_adversarial_return']:.6f}`"
        )
    report_lines.extend(
        [
            "",
            "## Exact commands",
            "",
            "```json",
            json.dumps(all_commands, indent=2),
            "```",
            "",
            f"## Final decision",
            "",
            f"`{decision}`",
        ]
    )
    (output_root / "one_seed_decision.md").write_text("\n".join(report_lines), encoding="utf-8")
    (output_root / "run_metadata.json").write_text(
        json.dumps(
            {
                "decision": decision,
                "output_root": str(output_root),
                "seed": args.seed,
                "job_id": None,
                "env": args.env,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
