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

from scripts.full_policy_optimizer_probe import (
    build_manager,
    clone_state,
    collect_probe_batches,
    compute_loss_and_grads,
    cosine_similarity,
    egm_update,
    flatten_named_tensors,
    named_parameters,
    ppm_update,
    restore_state,
    tensor_norm,
    vector_from_state_diff,
)


@dataclass(frozen=True)
class Candidate:
    label: str
    optimizer: str
    lr: float
    max_grad_norm: float
    vf_coef: float
    inner_steps: int = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run matched-config EGM/PPM diagnostic")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument("--python-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iterations", type=int, default=1, help="1 outer iteration is ~10k timesteps in the current RARL setup")
    parser.add_argument("--eval-freq", type=int, default=2500)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--num-probes", type=int, default=4)
    return parser.parse_args()


def render_kwargs_tokens(kwargs: Dict[str, object]) -> List[str]:
    tokens = []
    for key, value in sorted(kwargs.items()):
        tokens.append(f"{key}:{value!r}" if isinstance(value, str) else f"{key}:{value}")
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


def ensure_run(candidate: Candidate, args: argparse.Namespace, runs_root: pathlib.Path) -> pathlib.Path:
    run_root = runs_root / candidate.label
    analysis_dir = run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    saved_models_dir = run_root / "saved_models"
    if (analysis_dir / "run_summary.csv").exists():
        return find_latest_run_dir(saved_models_dir, args.env)

    kwargs = {}
    if candidate.inner_steps > 0:
        kwargs["inner_steps"] = candidate.inner_steps

    command = [
        args.python_path,
        "scripts/train_adversary.py",
        "--algo", "rarl",
        "--rarl-config", "ppo",
        "--env", args.env,
        "--adv-impact", "control",
        "--device", args.device,
        "--verbose", "1",
        "--seed", str(args.seed),
        "-n", str(args.iterations),
        "--eval-freq", str(args.eval_freq),
        "--save-freq", str(args.eval_freq),
        "--n-eval-episodes", str(args.n_eval_episodes),
        "--saved-models-path", str(saved_models_dir),
        "--log-folder", str(run_root / "logging"),
        "--tensorboard-log", str(run_root / "tb"),
        "--optimizer-scope", "full_policy",
        "--protagonist-optimizer", candidate.optimizer,
        "--adversary-optimizer", candidate.optimizer,
        "--protagonist-lr", str(candidate.lr),
        "--adversary-lr", str(candidate.lr),
        "--protagonist-max-grad-norm", str(candidate.max_grad_norm),
        "--adversary-max-grad-norm", str(candidate.max_grad_norm),
        "--protagonist-vf-coef", str(candidate.vf_coef),
        "--adversary-vf-coef", str(candidate.vf_coef),
    ]
    if kwargs:
        tokens = render_kwargs_tokens(kwargs)
        command.extend(["--protagonist-optimizer-kwargs", *tokens])
        command.extend(["--adversary-optimizer-kwargs", *tokens])

    run_command(command, cwd=pathlib.Path(args.repo_dir), stdout_path=run_root / "stdout.txt", stderr_path=run_root / "stderr.txt")
    latest_run_dir = find_latest_run_dir(saved_models_dir, args.env)
    shutil.copy2(run_root / "stdout.txt", analysis_dir / "stdout.txt")
    shutil.copy2(run_root / "stderr.txt", analysis_dir / "stderr.txt")
    run_command(
        [args.python_path, "scripts/analyze_rarl_run.py", "--run-dir", str(latest_run_dir), "--output-dir", str(analysis_dir), "--method", candidate.label],
        cwd=pathlib.Path(args.repo_dir),
        stdout_path=analysis_dir / "analyze_stdout.txt",
        stderr_path=analysis_dir / "analyze_stderr.txt",
    )
    return latest_run_dir


def run_fixed_minibatch_geometry(args: argparse.Namespace, output_dir: pathlib.Path) -> pd.DataFrame:
    manager = build_manager(argparse.Namespace(env=args.env, seed=args.seed, device=args.device))
    rarl_model = manager.setup_experiment()
    algo = rarl_model.protagonist
    probes = collect_probe_batches(rarl_model, "protagonist", args.num_probes)
    named_params = named_parameters(algo.policy)

    configs = [
        ("egm_matched", {"lr": 1e-3, "max_grad_norm": 0.5, "vf_coef": 0.5, "inner_steps": -1}),
        ("ppm_inner2_matched", {"lr": 1e-3, "max_grad_norm": 0.5, "vf_coef": 0.5, "inner_steps": 2}),
        ("ppm_inner5_matched", {"lr": 1e-3, "max_grad_norm": 0.5, "vf_coef": 0.5, "inner_steps": 5}),
        ("ppm_inner10_matched", {"lr": 1e-3, "max_grad_norm": 0.5, "vf_coef": 0.5, "inner_steps": 10}),
        ("ppm_inner20_matched", {"lr": 1e-3, "max_grad_norm": 0.5, "vf_coef": 0.5, "inner_steps": 20}),
    ]
    rows = []
    for probe in probes:
        theta_old = clone_state(named_params)
        old_eval, theta_half, half_eval, theta_egm = egm_update(
            algo, probe.rollout_data, theta_old, 1e-3, max_grad_norm=0.5, vf_coef=0.5, ent_coef=float(algo.ent_coef)
        )
        egm_update_vec = flatten_named_tensors(vector_from_state_diff(theta_egm, theta_old))
        rows.append(
            {
                "probe_idx": probe.probe_idx,
                "config": "egm_matched",
                "update_norm": tensor_norm(egm_update_vec),
                "grad_clip_active_frac": float(
                    tensor_norm(flatten_named_tensors(old_eval["grads"])) < tensor_norm(flatten_named_tensors(compute_loss_and_grads(algo, probe.rollout_data, max_grad_norm=float('inf'), vf_coef=0.5, ent_coef=float(algo.ent_coef))["grads"]))
                ),
                "approx_kl": float(old_eval["approx_kl"]),
                "clip_fraction": float(old_eval["clip_fraction"]),
                "ppm_inner_residual_last": np.nan,
                "ppm_fixed_point_residual": np.nan,
                "cosine_vs_egm": 1.0,
            }
        )
        for label, setting in configs[1:]:
            restore_state(named_params, theta_old)
            ppm_eval, theta_ppm, _, _, _, _, _, ppm_inner_residuals = ppm_update(
                algo,
                probe.rollout_data,
                theta_old,
                setting["lr"],
                inner_steps=setting["inner_steps"],
                max_grad_norm=setting["max_grad_norm"],
                vf_coef=setting["vf_coef"],
                ent_coef=float(algo.ent_coef),
            )
            update_ppm_vec = flatten_named_tensors(vector_from_state_diff(theta_ppm, theta_old))
            rows.append(
                {
                    "probe_idx": probe.probe_idx,
                    "config": label,
                    "update_norm": tensor_norm(update_ppm_vec),
                    "grad_clip_active_frac": float(
                        tensor_norm(flatten_named_tensors(ppm_eval["grads"])) < tensor_norm(flatten_named_tensors(compute_loss_and_grads(algo, probe.rollout_data, max_grad_norm=float('inf'), vf_coef=0.5, ent_coef=float(algo.ent_coef))["grads"]))
                    ),
                    "approx_kl": float(ppm_eval["approx_kl"]),
                    "clip_fraction": float(ppm_eval["clip_fraction"]),
                    "ppm_inner_residual_last": float(ppm_inner_residuals[-1]) if ppm_inner_residuals else np.nan,
                    "ppm_fixed_point_residual": float(ppm_inner_residuals[-1]) if ppm_inner_residuals else np.nan,
                    "cosine_vs_egm": cosine_similarity(update_ppm_vec, egm_update_vec),
                }
            )
        restore_state(named_params, theta_old)
    geometry_df = pd.DataFrame(rows)
    geometry_df.to_csv(output_dir / "egm_ppm_update_geometry.csv", index=False)
    return geometry_df


def plot_outputs(summary_df: pd.DataFrame, clean_df: pd.DataFrame, adv_df: pd.DataFrame, geometry_df: pd.DataFrame, plots_dir: pathlib.Path) -> None:
    colors = {
        "egm_stage10_ref": "tab:green",
        "ppm_stage10_ref": "tab:red",
        "egm_matched": "tab:olive",
        "ppm_inner2_matched": "tab:purple",
        "ppm_inner5_matched": "tab:blue",
        "ppm_inner10_matched": "tab:orange",
        "ppm_inner20_matched": "tab:brown",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, group in clean_df.groupby("method"):
        axes[0].plot(group["timesteps"], group["mean_reward"], label=label, color=colors.get(label))
    for label, group in adv_df.groupby("method"):
        axes[1].plot(group["timesteps"], group["mean_reward"], label=label, color=colors.get(label))
    axes[0].set_title("EGM/PPM matched config clean eval")
    axes[1].set_title("EGM/PPM matched config control-adv eval")
    for ax in axes:
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Mean reward")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "egm_ppm_matched_config.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    ppm_rows = geometry_df[geometry_df["config"].str.startswith("ppm_")]
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, group in ppm_rows.groupby("config"):
        ax.plot(group["probe_idx"], group["ppm_inner_residual_last"], marker="o", label=label, color=colors.get(label))
    ax.set_title("PPM inner residuals (matched config)")
    ax.set_xlabel("Probe")
    ax.set_ylabel("Residual")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "ppm_inner_residuals.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, group in geometry_df.groupby("config"):
        ax.plot(group["probe_idx"], group["cosine_vs_egm"], marker="o", label=label, color=colors.get(label))
    ax.set_title("EGM/PPM update geometry")
    ax.set_xlabel("Probe")
    ax.set_ylabel("Cosine vs EGM")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "egm_ppm_update_geometry.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_root = output_dir / "runs"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        Candidate("egm_stage10_ref", "egm", 1e-3, 10.0, 0.5),
        Candidate("ppm_stage10_ref", "ppm", 1e-3, 1.0, 1.0, inner_steps=10),
        Candidate("egm_matched", "egm", 1e-3, 0.5, 0.5),
        Candidate("ppm_inner2_matched", "ppm", 1e-3, 0.5, 0.5, inner_steps=2),
        Candidate("ppm_inner5_matched", "ppm", 1e-3, 0.5, 0.5, inner_steps=5),
        Candidate("ppm_inner10_matched", "ppm", 1e-3, 0.5, 0.5, inner_steps=10),
        Candidate("ppm_inner20_matched", "ppm", 1e-3, 0.5, 0.5, inner_steps=20),
    ]

    summary_rows = []
    clean_frames = []
    adv_frames = []
    for candidate in candidates:
        run_dir = ensure_run(candidate, args, runs_root)
        analysis_dir = runs_root / candidate.label / "analysis"
        summary = pd.read_csv(analysis_dir / "run_summary.csv").iloc[0].to_dict()
        summary["label"] = candidate.label
        summary["configured_max_grad_norm"] = candidate.max_grad_norm
        summary["configured_vf_coef"] = candidate.vf_coef
        summary["configured_inner_steps"] = candidate.inner_steps
        summary_rows.append(summary)
        clean = pd.read_csv(analysis_dir / "clean_eval_returns.csv")
        clean["method"] = candidate.label
        clean_frames.append(clean)
        adv = pd.read_csv(analysis_dir / "adversarial_eval_returns.csv")
        adv["method"] = candidate.label
        adv_frames.append(adv)

    summary_df = pd.DataFrame(summary_rows)
    clean_df = pd.concat(clean_frames, ignore_index=True)
    adv_df = pd.concat(adv_frames, ignore_index=True)
    geometry_df = run_fixed_minibatch_geometry(args, output_dir)

    summary_df.to_csv(output_dir / "egm_ppm_matched_config_summary.csv", index=False)
    clean_df.to_csv(output_dir / "egm_ppm_matched_clean_eval_curves.csv", index=False)
    adv_df.to_csv(output_dir / "egm_ppm_matched_adv_eval_curves.csv", index=False)
    plot_outputs(summary_df, clean_df, adv_df, geometry_df, plots_dir)

    report_lines = [
        "# EGM/PPM Matched-Config Report",
        "",
        "- Protocol: `proper control-RARL`, `seed=0`, `1 outer iteration ~= 10k timesteps`",
        "- Purpose: isolate whether PPM weakness came from `max_grad_norm` / `vf_coef` mismatch versus the optimizer itself.",
        "",
    ]
    by_label = {row["label"]: row for row in summary_rows}
    report_lines.extend(
        [
            f"- Does PPM still lose after fair max_grad_norm/vf_coef matching? `{by_label['ppm_inner10_matched']['last5_clean_mean'] < by_label['egm_matched']['last5_clean_mean'] and by_label['ppm_inner10_matched']['last5_adversarial_mean'] < by_label['egm_matched']['last5_adversarial_mean']}`",
            f"- PPM inner=2 behaves like EGM in update geometry? `{float(geometry_df[geometry_df['config'] == 'ppm_inner2_matched']['cosine_vs_egm'].mean()) > 0.999}`",
            f"- PPM inner residual decreases to a small last-step scale for inner=20? `{float(geometry_df[geometry_df['config'] == 'ppm_inner20_matched']['ppm_inner_residual_last'].mean()) < float(geometry_df[geometry_df['config'] == 'ppm_inner5_matched']['ppm_inner_residual_last'].mean())}`",
        ]
    )
    (output_dir / "egm_ppm_matched_config_report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
