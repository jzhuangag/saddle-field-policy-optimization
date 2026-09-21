from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.full_policy_followup_common import rolling_mean
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
    tensor_norm,
    vector_from_state_diff,
)


@dataclass(frozen=True)
class Candidate:
    method: str
    optimizer: str
    lr: float
    max_grad_norm: float
    vf_coef: float
    optimizer_kwargs: Dict[str, float]
    tag: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 6C PPM vs EGM distinction")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument("--python-path", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--screen-iters", type=int, default=1)
    parser.add_argument("--final-iters", type=int, default=5)
    parser.add_argument("--eval-freq", type=int, default=2500)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--num-probes", type=int, default=3)
    return parser.parse_args()


def make_tag(method: str, lr: float, max_grad_norm: float, vf_coef: float, optimizer_kwargs: Dict[str, float]) -> str:
    clip_text = "inf" if not math.isfinite(max_grad_norm) else f"{max_grad_norm:g}"
    parts = [f"lr_{lr:.1e}".replace("+", ""), f"clip_{clip_text}", f"vf_{vf_coef:g}"]
    for key, value in sorted(optimizer_kwargs.items()):
        parts.append(f"{key}_{value}")
    return f"{method}__" + "__".join(parts)


def build_candidates() -> List[Candidate]:
    candidates: List[Candidate] = []
    for lr in [3.0e-04, 1.0e-03, 3.0e-03]:
        for max_grad_norm in [1.0, 5.0, 10.0]:
            for vf_coef in [0.1, 0.5]:
                candidates.append(Candidate("egm", "egm", lr, max_grad_norm, vf_coef, {}, make_tag("egm", lr, max_grad_norm, vf_coef, {})))
    for lr in [1.0e-04, 3.0e-04, 1.0e-03, 3.0e-03]:
        for max_grad_norm in [1.0, 5.0, 10.0]:
            for vf_coef in [0.1, 0.5]:
                for inner_steps in [3, 5, 10, 20]:
                    kwargs = {"inner_steps": inner_steps}
                    candidates.append(Candidate("ppm", "ppm", lr, max_grad_norm, vf_coef, kwargs, make_tag("ppm", lr, max_grad_norm, vf_coef, kwargs)))
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
    return next(path for path in env_root.iterdir() if path.is_dir() and path.name.startswith(f"{env_id}_"))


def candidate_score(summary_path: pathlib.Path) -> float:
    row = pd.read_csv(summary_path).iloc[0]
    if int(row["crash_flag"]) or int(row["nan_flag"]):
        return -1e18
    return float(row["last5_clean_mean"]) + float(row["last5_adversarial_mean"]) + 0.1 * float(row["auc_clean"]) + 0.1 * float(row["auc_adversarial"])


def run_candidate(args: argparse.Namespace, candidate: Candidate, phase: str, iterations: int) -> pathlib.Path:
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
        "--protagonist-max-grad-norm",
        str(candidate.max_grad_norm),
        "--adversary-max-grad-norm",
        str(candidate.max_grad_norm),
        "--protagonist-vf-coef",
        str(candidate.vf_coef),
        "--adversary-vf-coef",
        str(candidate.vf_coef),
    ]
    for key, value in sorted(candidate.optimizer_kwargs.items()):
        command.extend(["--protagonist-optimizer-kwargs", f"{key}:{value}", "--adversary-optimizer-kwargs", f"{key}:{value}"])

    run_command(command, cwd=pathlib.Path(args.repo_dir), stdout_path=run_root / "stdout.txt", stderr_path=run_root / "stderr.txt")
    run_dir = find_run_dir(saved_models_dir, args.env)
    analysis_dir = run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_root / "stdout.txt", analysis_dir / "stdout.txt")
    shutil.copy2(run_root / "stderr.txt", analysis_dir / "stderr.txt")
    run_command(
        [args.python_path, "scripts/analyze_rarl_run.py", "--run-dir", str(run_dir), "--output-dir", str(analysis_dir), "--method", candidate.method],
        cwd=pathlib.Path(args.repo_dir),
        stdout_path=analysis_dir / "analyze_stdout.txt",
        stderr_path=analysis_dir / "analyze_stderr.txt",
    )
    return analysis_dir


def probe_pair(args: argparse.Namespace, egm_candidate: Candidate, ppm_candidate: Candidate) -> pd.DataFrame:
    rows = []
    for role in ["protagonist", "adversary"]:
        probe_args = SimpleNamespace(output_dir="", env=args.env, seed=args.seed, device=args.device, role=role, num_probes=args.num_probes)
        manager = build_manager(probe_args)
        rarl_model = manager.setup_experiment()
        algo = rarl_model.protagonist if role == "protagonist" else rarl_model.adversary
        probes = collect_probe_batches(rarl_model, role, args.num_probes)
        default_ent_coef = float(algo.ent_coef)
        for probe in probes:
            named_params = named_parameters(algo.policy)
            theta_old = clone_state(named_params)
            egm_old_eval, _, _, theta_egm = egm_update(
                algo,
                probe.rollout_data,
                theta_old,
                egm_candidate.lr,
                max_grad_norm=egm_candidate.max_grad_norm,
                vf_coef=egm_candidate.vf_coef,
                ent_coef=default_ent_coef,
            )
            ppm_eval, theta_ppm, ppm_losses, _, _, _, _, ppm_inner_residuals = ppm_update(
                algo,
                probe.rollout_data,
                theta_old,
                ppm_candidate.lr,
                inner_steps=int(ppm_candidate.optimizer_kwargs["inner_steps"]),
                max_grad_norm=ppm_candidate.max_grad_norm,
                vf_coef=ppm_candidate.vf_coef,
                ent_coef=default_ent_coef,
            )
            update_egm = flatten_named_tensors(vector_from_state_diff(theta_egm, theta_old))
            update_ppm = flatten_named_tensors(vector_from_state_diff(theta_ppm, theta_old))
            old_grad = flatten_named_tensors(egm_old_eval["grads"])
            field_movement_ratio = tensor_norm(update_egm) / max(tensor_norm(old_grad), 1e-12)
            rows.append(
                {
                    "role": role,
                    "probe_idx": probe.probe_idx,
                    "egm_lr": egm_candidate.lr,
                    "egm_max_grad_norm": egm_candidate.max_grad_norm,
                    "egm_vf_coef": egm_candidate.vf_coef,
                    "ppm_lr": ppm_candidate.lr,
                    "ppm_max_grad_norm": ppm_candidate.max_grad_norm,
                    "ppm_vf_coef": ppm_candidate.vf_coef,
                    "ppm_inner_steps": ppm_candidate.optimizer_kwargs["inner_steps"],
                    "update_cosine_ppm_vs_egm": cosine_similarity(update_ppm, update_egm),
                    "update_norm_ratio_ppm_vs_egm": tensor_norm(update_ppm) / max(tensor_norm(update_egm), 1e-12),
                    "field_movement_ratio": field_movement_ratio,
                    "ppm_inner_residuals": json.dumps(ppm_inner_residuals),
                    "ppm_inner_losses": json.dumps(ppm_losses),
                }
            )
    return pd.DataFrame(rows)


def plot_final_outputs(output_root: pathlib.Path, egm_analysis_dir: pathlib.Path, ppm_analysis_dir: pathlib.Path, probe_df: pd.DataFrame) -> None:
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    egm_clean = pd.read_csv(egm_analysis_dir / "clean_eval_returns.csv")
    ppm_clean = pd.read_csv(ppm_analysis_dir / "clean_eval_returns.csv")
    egm_adv = pd.read_csv(egm_analysis_dir / "adversarial_eval_returns.csv")
    ppm_adv = pd.read_csv(ppm_analysis_dir / "adversarial_eval_returns.csv")

    for frame in [egm_clean, ppm_clean]:
        frame["rolling_mean_reward"] = rolling_mean(frame["mean_reward"], window=3)
    for frame in [egm_adv, ppm_adv]:
        frame["rolling_mean_reward"] = rolling_mean(frame["mean_reward"], window=3)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, frame in [("egm", egm_clean), ("ppm", ppm_clean)]:
        ax.plot(frame["timesteps"], frame["mean_reward"], alpha=0.25, linewidth=1.0)
        ax.plot(frame["timesteps"], frame["rolling_mean_reward"], linewidth=2.0, label=label)
        ax.fill_between(frame["timesteps"], frame["mean_reward"] - frame["std_reward"], frame["mean_reward"] + frame["std_reward"], alpha=0.12)
    ax.set_title("PPM vs EGM Dense Clean Eval")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "ppm_vs_egm_dense_clean_eval.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, frame in [("egm", egm_adv), ("ppm", ppm_adv)]:
        ax.plot(frame["timesteps"], frame["mean_reward"], alpha=0.25, linewidth=1.0)
        ax.plot(frame["timesteps"], frame["rolling_mean_reward"], linewidth=2.0, label=label)
        ax.fill_between(frame["timesteps"], frame["mean_reward"] - frame["std_reward"], frame["mean_reward"] + frame["std_reward"], alpha=0.12)
    ax.set_title("PPM vs EGM Dense Adversarial Eval")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "ppm_vs_egm_dense_adv_eval.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    residual_rows = []
    for _, row in probe_df.iterrows():
        residuals = json.loads(row["ppm_inner_residuals"])
        for idx, value in enumerate(residuals, start=1):
            residual_rows.append({"role": row["role"], "probe_idx": row["probe_idx"], "inner_step": idx, "residual": value})
    residual_df = pd.DataFrame(residual_rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    for (role, probe_idx), group in residual_df.groupby(["role", "probe_idx"]):
        ax.plot(group["inner_step"], group["residual"], marker="o", alpha=0.7, label=f"{role}_{probe_idx}")
    ax.set_title("PPM Inner Residuals")
    ax.set_xlabel("Inner step")
    ax.set_ylabel("||z_m - z_(m-1)||")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "ppm_inner_residuals.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    similarity_df = probe_df.groupby("role").mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(similarity_df["role"], similarity_df["update_cosine_ppm_vs_egm"])
    axes[0].set_title("PPM vs EGM Update Cosine")
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(similarity_df["role"], similarity_df["update_norm_ratio_ppm_vs_egm"])
    axes[1].set_title("PPM vs EGM Update Norm Ratio")
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plots_dir / "ppm_egm_update_similarity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    candidates = build_candidates()
    screen_analysis_dirs: Dict[str, pathlib.Path] = {}
    best_by_method: Dict[str, tuple[float, Candidate]] = {}
    for candidate in candidates:
        analysis_dir = run_candidate(args, candidate, phase="screen", iterations=args.screen_iters)
        screen_analysis_dirs[candidate.tag] = analysis_dir
        score = candidate_score(analysis_dir / "run_summary.csv")
        current = best_by_method.get(candidate.method)
        if current is None or score > current[0]:
            best_by_method[candidate.method] = (score, candidate)

    egm_candidate = best_by_method["egm"][1]
    ppm_candidate = best_by_method["ppm"][1]

    egm_final_dir = run_candidate(args, egm_candidate, phase="final", iterations=args.final_iters)
    ppm_final_dir = run_candidate(args, ppm_candidate, phase="final", iterations=args.final_iters)

    egm_summary = pd.read_csv(egm_final_dir / "run_summary.csv").iloc[0].to_dict()
    ppm_summary = pd.read_csv(ppm_final_dir / "run_summary.csv").iloc[0].to_dict()
    summary_df = pd.DataFrame([egm_summary, ppm_summary]).sort_values("method").reset_index(drop=True)
    summary_df.to_csv(output_root / "ppm_egm_distinction_summary.csv", index=False)

    probe_df = probe_pair(args, egm_candidate, ppm_candidate)
    probe_df.to_csv(output_root / "ppm_egm_probe_metrics.csv", index=False)
    plot_final_outputs(output_root, egm_final_dir, ppm_final_dir, probe_df)

    cosine_mean = float(probe_df["update_cosine_ppm_vs_egm"].mean())
    norm_ratio_mean = float(probe_df["update_norm_ratio_ppm_vs_egm"].mean())
    if cosine_mean < 0.999 or abs(norm_ratio_mean - 1.0) > 0.02:
        distinction_verdict = "PPM(inner_steps>2) is measurably distinct from EGM on the fixed-minibatch probes."
    else:
        distinction_verdict = "PPM(inner_steps>2) remains nearly identical to EGM on the fixed-minibatch probes."

    lines = [
        "# PPM vs EGM Distinction",
        "",
        "## Selected candidates",
        "",
        "```json",
        json.dumps(
            {
                "egm_final": {
                    "lr": egm_candidate.lr,
                    "max_grad_norm": egm_candidate.max_grad_norm,
                    "vf_coef": egm_candidate.vf_coef,
                },
                "ppm_final": {
                    "lr": ppm_candidate.lr,
                    "max_grad_norm": ppm_candidate.max_grad_norm,
                    "vf_coef": ppm_candidate.vf_coef,
                    "inner_steps": ppm_candidate.optimizer_kwargs["inner_steps"],
                },
            },
            indent=2,
        ),
        "```",
        "",
        f"- Verdict: {distinction_verdict}",
        f"- Mean update cosine ppm vs egm: `{cosine_mean:.6f}`",
        f"- Mean update norm ratio ppm vs egm: `{norm_ratio_mean:.6f}`",
        "",
        "## Final eval rows",
        "",
    ]
    for _, row in summary_df.iterrows():
        lines.extend(
            [
                f"- `{row['method']}`",
                f"  - last5_clean_mean: `{row['last5_clean_mean']:.6f}`",
                f"  - last5_adversarial_mean: `{row['last5_adversarial_mean']:.6f}`",
                f"  - approx_kl_mean: `{row['approx_kl_mean']:.6f}`",
                f"  - clip_fraction_mean: `{row['clip_fraction_mean']:.6f}`",
            ]
        )
    (output_root / "ppm_egm_distinction_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
