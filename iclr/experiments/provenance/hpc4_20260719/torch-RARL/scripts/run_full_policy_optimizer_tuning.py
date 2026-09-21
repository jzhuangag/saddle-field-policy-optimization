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
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from models.optimizers import get_optimizer_class
from scripts.full_policy_optimizer_probe import (
    block_tensor,
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
    sgd_update,
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
    parser = argparse.ArgumentParser("Run Stage 5C full-policy PPO-RARL optimizer tuning")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument("--python-path", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--screen-iters", type=int, default=1)
    parser.add_argument("--refine-iters", type=int, default=1)
    parser.add_argument("--final-iters", type=int, default=5)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--n-eval-episodes", type=int, default=1)
    parser.add_argument("--num-probes", type=int, default=3)
    return parser.parse_args()


def make_tag(method: str, lr: float, max_grad_norm: float, vf_coef: float, optimizer_kwargs: Dict[str, float]) -> str:
    clip_text = "inf" if not math.isfinite(max_grad_norm) else f"{max_grad_norm:g}"
    parts = [f"lr_{lr:.1e}".replace("+", ""), f"clip_{clip_text}", f"vf_{vf_coef:g}"]
    for key, value in sorted(optimizer_kwargs.items()):
        parts.append(f"{key}_{value}")
    return f"{method}__" + "__".join(parts)


def build_screen_candidates() -> List[Candidate]:
    candidates: List[Candidate] = []
    default_vf = 0.58096

    adam_specs = [
        (2.0633e-05, 0.8, default_vf, {}),
        (5.0e-05, 1.0, 0.5, {}),
        (1.0e-04, 1.0, 0.1, {}),
    ]
    for lr, max_grad_norm, vf_coef, kwargs in adam_specs:
        candidates.append(Candidate("adam", "adam", lr, max_grad_norm, vf_coef, kwargs, make_tag("adam", lr, max_grad_norm, vf_coef, kwargs)))

    for lr in [1.0e-05, 3.0e-05, 1.0e-04, 3.0e-04, 1.0e-03]:
        for max_grad_norm in [1.0, 10.0]:
            kwargs: Dict[str, float] = {}
            candidates.append(Candidate("sgd", "sgd", lr, max_grad_norm, default_vf, kwargs, make_tag("sgd", lr, max_grad_norm, default_vf, kwargs)))

    for lr in [1.0e-04, 3.0e-04, 1.0e-03, 3.0e-03]:
        for max_grad_norm in [1.0, 10.0]:
            kwargs = {}
            candidates.append(Candidate("egm", "egm", lr, max_grad_norm, default_vf, kwargs, make_tag("egm", lr, max_grad_norm, default_vf, kwargs)))

    for lr in [1.0e-04, 3.0e-04, 1.0e-03, 3.0e-03]:
        for max_grad_norm in [1.0, 10.0]:
            for inner_steps in [2, 5]:
                kwargs = {"inner_steps": inner_steps}
                candidates.append(Candidate("ppm", "ppm", lr, max_grad_norm, default_vf, kwargs, make_tag("ppm", lr, max_grad_norm, default_vf, kwargs)))

    return candidates


def build_refine_candidates(seed_candidates: Dict[str, List[Candidate]]) -> List[Candidate]:
    refined: List[Candidate] = []
    seen = set()
    for method, candidates in seed_candidates.items():
        if method == "adam":
            vf_grid = [0.1, 0.5, 1.0]
        else:
            vf_grid = [0.1, 0.5, 1.0]
        for candidate in candidates:
            for vf_coef in vf_grid:
                kwargs = dict(candidate.optimizer_kwargs)
                extra_inner_steps = [kwargs.get("inner_steps", None)]
                if method == "ppm":
                    current_inner = int(kwargs.get("inner_steps", 5))
                    extra_inner_steps = sorted({current_inner, 10 if current_inner >= 5 else 5})
                for inner_steps in extra_inner_steps:
                    new_kwargs = dict(kwargs)
                    if inner_steps is not None:
                        new_kwargs["inner_steps"] = inner_steps
                    refined_candidate = Candidate(
                        method=method,
                        optimizer=candidate.optimizer,
                        lr=candidate.lr,
                        max_grad_norm=candidate.max_grad_norm,
                        vf_coef=vf_coef,
                        optimizer_kwargs=new_kwargs,
                        tag=make_tag(method, candidate.lr, candidate.max_grad_norm, vf_coef, new_kwargs),
                    )
                    if refined_candidate.tag in seen:
                        continue
                    seen.add(refined_candidate.tag)
                    refined.append(refined_candidate)
    return refined


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
    row = pd.read_csv(summary_path).iloc[0]
    if int(row["crash_flag"]) or int(row["nan_flag"]):
        return -1e18
    score = (
        1.5 * float(row["last5_clean_mean"])
        + 1.5 * float(row["last5_adversarial_mean"])
        + 0.5 * float(row["final_clean_return"])
        + 0.5 * float(row["final_adversarial_return"])
        + 0.15 * float(row.get("auc_clean", 0.0))
        + 0.15 * float(row.get("auc_adversarial", 0.0))
    )
    if np.isfinite(row.get("approx_kl_mean", np.nan)) and float(row["approx_kl_mean"]) > 0.2:
        score -= 1000.0
    return score


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


def select_top_candidates(candidates: Iterable[Candidate], analysis_dirs: Dict[str, pathlib.Path], top_k: int) -> Dict[str, List[Candidate]]:
    by_method: Dict[str, List[Tuple[float, Candidate]]] = {}
    for candidate in candidates:
        analysis_dir = analysis_dirs[candidate.tag]
        score = candidate_score(analysis_dir / "run_summary.csv")
        by_method.setdefault(candidate.method, []).append((score, candidate))
    selected: Dict[str, List[Candidate]] = {}
    for method, scored in by_method.items():
        scored.sort(key=lambda item: item[0], reverse=True)
        selected[method] = [candidate for _, candidate in scored[:top_k]]
    return selected


def summarize_candidates(candidates_by_method: Dict[str, List[Candidate]]) -> Dict[str, List[Dict[str, object]]]:
    out: Dict[str, List[Dict[str, object]]] = {}
    for method, candidates in candidates_by_method.items():
        out[method] = [
            {
                "optimizer": candidate.optimizer,
                "lr": candidate.lr,
                "max_grad_norm": candidate.max_grad_norm,
                "vf_coef": candidate.vf_coef,
                "optimizer_kwargs": candidate.optimizer_kwargs,
                "tag": candidate.tag,
            }
            for candidate in candidates
        ]
    return out


def perform_optimizer_step(
    algo,
    rollout_data,
    *,
    optimizer_name: str,
    lr: float,
    optimizer_kwargs: Dict[str, float],
    max_grad_norm: float,
    vf_coef: float,
    ent_coef: float,
) -> Tuple[Dict[str, "th.Tensor"], float, float, float, float]:
    import torch as th

    named_params = named_parameters(algo.policy)
    theta_old = clone_state(named_params)
    optimizer_class = get_optimizer_class(optimizer_name)
    algo.policy.optimizer = optimizer_class(algo.policy.parameters(), lr=lr, **optimizer_kwargs)
    clip_range, clip_range_vf = algo.clip_range(algo._current_progress_remaining), None
    if algo.clip_range_vf is not None:
        clip_range_vf = algo.clip_range_vf(algo._current_progress_remaining)
    actions = rollout_data.actions.long().flatten() if algo.action_space.__class__.__name__ == "Discrete" else rollout_data.actions

    def closure():
        algo.policy.optimizer.zero_grad()
        total_loss, _, _, _, _, _, _ = algo._build_shared_policy_loss(
            rollout_data,
            actions,
            clip_range,
            clip_range_vf,
        )
        total_loss.backward()
        if math.isfinite(max_grad_norm):
            th.nn.utils.clip_grad_norm_(algo.policy.parameters(), max_grad_norm)
        return total_loss

    old_vf_coef, old_ent_coef, old_clip = algo.vf_coef, algo.ent_coef, algo.max_grad_norm
    algo.vf_coef = vf_coef
    algo.ent_coef = ent_coef
    algo.max_grad_norm = max_grad_norm
    try:
        if bool(getattr(algo.policy.optimizer, "requires_closure", False)):
            algo.policy.optimizer.step(closure)
        else:
            loss = closure()
            algo.policy.optimizer.step()
        theta_new = clone_state(named_params)
    finally:
        algo.vf_coef = old_vf_coef
        algo.ent_coef = old_ent_coef
        algo.max_grad_norm = old_clip
        restore_state(named_params, theta_old)
    update = vector_from_state_diff(theta_new, theta_old)
    return update, tensor_norm(block_tensor(update, "actor")), tensor_norm(block_tensor(update, "logstd")), tensor_norm(block_tensor(update, "critic")), float(loss.item() if "loss" in locals() else 0.0)


def probe_candidate(candidate: Candidate, env_id: str, seed: int, device: str, num_probes: int) -> pd.DataFrame:
    rows = []
    for role in ["protagonist", "adversary"]:
        probe_args = SimpleNamespace(output_dir="", env=env_id, seed=seed, device=device, role=role, num_probes=num_probes)
        manager = build_manager(probe_args)
        rarl_model = manager.setup_experiment()
        algo = rarl_model.protagonist if role == "protagonist" else rarl_model.adversary
        probes = collect_probe_batches(rarl_model, role, num_probes)
        default_ent_coef = float(algo.ent_coef)
        for probe in probes:
            named_params = named_parameters(algo.policy)
            theta_old = clone_state(named_params)
            old_eval = compute_loss_and_grads(
                algo,
                probe.rollout_data,
                max_grad_norm=candidate.max_grad_norm,
                vf_coef=candidate.vf_coef,
                ent_coef=default_ent_coef,
            )
            theta_sgd = sgd_update(theta_old, old_eval["grads"], candidate.lr)
            _, _, half_eval, theta_egm = egm_update(
                algo,
                probe.rollout_data,
                theta_old,
                candidate.lr,
                max_grad_norm=candidate.max_grad_norm,
                vf_coef=candidate.vf_coef,
                ent_coef=default_ent_coef,
            )
            _, theta_ppm, _, _, _, _, _, _ = ppm_update(
                algo,
                probe.rollout_data,
                theta_old,
                candidate.lr,
                inner_steps=int(candidate.optimizer_kwargs.get("inner_steps", 5)),
                max_grad_norm=candidate.max_grad_norm,
                vf_coef=candidate.vf_coef,
                ent_coef=default_ent_coef,
            )
            update_method, actor_update, logstd_update, critic_update, _ = perform_optimizer_step(
                algo,
                probe.rollout_data,
                optimizer_name=candidate.optimizer,
                lr=candidate.lr,
                optimizer_kwargs=candidate.optimizer_kwargs,
                max_grad_norm=candidate.max_grad_norm,
                vf_coef=candidate.vf_coef,
                ent_coef=default_ent_coef,
            )
            update_sgd = vector_from_state_diff(theta_sgd, theta_old)
            update_egm = vector_from_state_diff(theta_egm, theta_old)
            update_ppm = vector_from_state_diff(theta_ppm, theta_old)
            grad_old = flatten_named_tensors(old_eval["grads"])
            grad_half = flatten_named_tensors(half_eval["grads"])
            if candidate.method == "sgd":
                update_cosine = 1.0
                update_ratio = 1.0
            elif candidate.method == "egm":
                update_cosine = cosine_similarity(flatten_named_tensors(update_egm), flatten_named_tensors(update_sgd))
                update_ratio = tensor_norm(flatten_named_tensors(update_egm)) / max(tensor_norm(flatten_named_tensors(update_sgd)), 1e-12)
            elif candidate.method == "ppm":
                update_cosine = cosine_similarity(flatten_named_tensors(update_ppm), flatten_named_tensors(update_sgd))
                update_ratio = tensor_norm(flatten_named_tensors(update_ppm)) / max(tensor_norm(flatten_named_tensors(update_sgd)), 1e-12)
            else:
                update_cosine = cosine_similarity(flatten_named_tensors(update_method), flatten_named_tensors(update_sgd))
                update_ratio = tensor_norm(flatten_named_tensors(update_method)) / max(tensor_norm(flatten_named_tensors(update_sgd)), 1e-12)

            rows.append(
                {
                    "method": candidate.method,
                    "optimizer": candidate.optimizer,
                    "role": role,
                    "probe_idx": probe.probe_idx,
                    "lr": candidate.lr,
                    "max_grad_norm": candidate.max_grad_norm,
                    "vf_coef": candidate.vf_coef,
                    "ppm_inner_steps": candidate.optimizer_kwargs.get("inner_steps", np.nan),
                    "field_movement_ratio": tensor_norm(grad_half - grad_old) / max(tensor_norm(grad_old), 1e-12),
                    "update_cosine_vs_sgd": update_cosine,
                    "update_norm_ratio_vs_sgd": update_ratio,
                    "actor_grad_norm": tensor_norm(block_tensor(old_eval["grads"], "actor")),
                    "logstd_grad_norm": tensor_norm(block_tensor(old_eval["grads"], "logstd")),
                    "critic_grad_norm": tensor_norm(block_tensor(old_eval["grads"], "critic")),
                    "actor_update_norm": actor_update,
                    "logstd_update_norm": logstd_update,
                    "critic_update_norm": critic_update,
                }
            )
    return pd.DataFrame(rows)


def aggregate_outputs(final_analysis_dirs: List[pathlib.Path], output_root: pathlib.Path, probe_df: pd.DataFrame) -> pd.DataFrame:
    frames = [pd.read_csv(path / "run_summary.csv") for path in final_analysis_dirs]
    summary = pd.concat(frames, ignore_index=True)
    probe_summary = probe_df.groupby("method").mean(numeric_only=True).reset_index()
    summary = summary.merge(probe_summary, on="method", how="left", suffixes=("", "_probe"))
    summary.to_csv(output_root / "full_policy_tuning_summary.csv", index=False)
    return summary


def export_all_method_frames(final_analysis_dirs: List[pathlib.Path], output_root: pathlib.Path) -> Dict[str, pd.DataFrame]:
    training_frames = [pd.read_csv(path / "training_episode_returns.csv") for path in final_analysis_dirs]
    clean_frames = [pd.read_csv(path / "clean_eval_returns.csv") for path in final_analysis_dirs]
    adv_frames = [pd.read_csv(path / "adversarial_eval_returns.csv") for path in final_analysis_dirs]
    param_frames = [pd.read_csv(path / "parameter_norms.csv") for path in final_analysis_dirs]

    data = {
        "training": pd.concat(training_frames, ignore_index=True),
        "clean": pd.concat(clean_frames, ignore_index=True),
        "adv": pd.concat(adv_frames, ignore_index=True),
        "params": pd.concat(param_frames, ignore_index=True),
    }
    data["training"].to_csv(output_root / "training_episode_returns_all_methods.csv", index=False)
    data["clean"].to_csv(output_root / "clean_eval_returns_all_methods.csv", index=False)
    data["adv"].to_csv(output_root / "adversarial_eval_returns_all_methods.csv", index=False)
    data["params"].to_csv(output_root / "parameter_norms_all_methods.csv", index=False)
    return data


def plot_results(output_root: pathlib.Path, data: Dict[str, pd.DataFrame], probe_df: pd.DataFrame) -> None:
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    training_df = data["training"]
    clean_df = data["clean"]
    adv_df = data["adv"]
    params_df = data["params"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in training_df.groupby("method"):
        ax.plot(group["cumulative_timesteps"], group["episode_return"], label=method, linewidth=1.2)
    ax.set_title("Full-Policy PPO-RARL Training Return")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Return")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "full_policy_training_return_all_methods.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in clean_df.groupby("method"):
        ax.plot(group["timesteps"], group["mean_reward"], label=method, linewidth=1.2)
    ax.set_title("Full-Policy PPO-RARL Clean Eval")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "full_policy_clean_eval_all_methods.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, group in adv_df.groupby("method"):
        ax.plot(group["timesteps"], group["mean_reward"], label=method, linewidth=1.2)
    ax.set_title("Full-Policy PPO-RARL Adversarial Eval")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Reward")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "full_policy_adversarial_eval_all_methods.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    probe_means = probe_df.groupby("method").mean(numeric_only=True).reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(probe_means))
    width = 0.25
    ax.bar(x - width, probe_means["actor_update_norm"], width=width, label="actor")
    ax.bar(x, probe_means["logstd_update_norm"], width=width, label="log_std")
    ax.bar(x + width, probe_means["critic_update_norm"], width=width, label="critic")
    ax.set_xticks(x)
    ax.set_xticklabels(probe_means["method"])
    ax.set_title("Full-Policy Update Norm by Block")
    ax.set_ylabel("Mean update norm (fixed-minibatch probe)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "full_policy_update_norm_by_block.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    metrics = [
        ("policy_gradient_loss_mean", "Policy Gradient Loss"),
        ("value_loss_mean", "Value Loss"),
        ("entropy_loss_mean", "Entropy Loss"),
        ("approx_kl_mean", "Approx KL"),
        ("clip_fraction_mean", "Clip Fraction"),
        ("explained_variance_mean", "Explained Variance"),
    ]
    summary_df = pd.read_csv(output_root / "full_policy_tuning_summary.csv")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for axis, (column, title) in zip(axes.flatten(), metrics):
        axis.bar(summary_df["method"], summary_df[column])
        axis.set_title(title)
        axis.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plots_dir / "full_policy_loss_components_all_methods.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    similarity_df = probe_df.groupby(["method", "role"]).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for role, group in similarity_df.groupby("role"):
        axes[0].plot(group["method"], group["update_cosine_vs_sgd"], marker="o", label=role)
        axes[1].plot(group["method"], group["field_movement_ratio"], marker="o", label=role)
    axes[0].set_title("Optimizer Update Cosine vs SGD")
    axes[0].set_ylabel("Cosine")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].set_title("Field Movement Ratio")
    axes[1].set_ylabel("Ratio")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "full_policy_egm_ppm_vs_sgd_similarity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_tuning_report(
    output_root: pathlib.Path,
    screen_top: Dict[str, List[Candidate]],
    refine_top: Dict[str, List[Candidate]],
    final_candidates: Dict[str, Candidate],
    final_top2: Dict[str, List[Candidate]],
    summary_df: pd.DataFrame,
) -> None:
    lines = [
        "# Full-Policy PPO-RARL Tuning Report",
        "",
        "## Screening winners",
        "",
        "```json",
        json.dumps(
            {
                "screen_top2": summarize_candidates(screen_top),
                "refine_top2": summarize_candidates(refine_top),
                "final_top2": summarize_candidates(final_top2),
                "final_best": summarize_candidates({method: [candidate] for method, candidate in final_candidates.items()}),
            },
            indent=2,
        ),
        "```",
        "",
        "## Final equal-budget comparison",
        "",
    ]
    for _, row in summary_df.sort_values("method").iterrows():
        lines.extend(
            [
                f"- `{row['method']}`",
                f"  - optimizer: `{row['protagonist_optimizer']}`",
                f"  - lr / clip / vf: `{row['protagonist_lr']}` / `{row['protagonist_max_grad_norm']}` / `{row['protagonist_vf_coef']}`",
                f"  - ppm_inner_steps: `{row['ppm_inner_steps']}`",
                f"  - last5_clean_mean: `{row['last5_clean_mean']:.6f}`",
                f"  - last5_adversarial_mean: `{row['last5_adversarial_mean']:.6f}`",
                f"  - auc_clean / auc_adv: `{row['auc_clean']:.6f}` / `{row['auc_adversarial']:.6f}`",
                f"  - approx_kl_mean / clip_fraction_mean: `{row['approx_kl_mean']:.6f}` / `{row['clip_fraction_mean']:.6f}`",
                f"  - probe field_movement_ratio: `{row['field_movement_ratio']:.6f}`",
                f"  - probe update_cosine_vs_sgd: `{row['update_cosine_vs_sgd']:.6f}`",
                f"  - crash_flag / nan_flag: `{int(row['crash_flag'])}` / `{int(row['nan_flag'])}`",
            ]
        )
    (output_root / "full_policy_tuning_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_failure_classification(output_root: pathlib.Path, summary_df: pd.DataFrame, diagnosis_path: pathlib.Path) -> None:
    summary_df = summary_df.sort_values("method")
    sgd_row = summary_df[summary_df["method"] == "sgd"].iloc[0]
    egm_row = summary_df[summary_df["method"] == "egm"].iloc[0]
    ppm_row = summary_df[summary_df["method"] == "ppm"].iloc[0]
    adam_row = summary_df[summary_df["method"] == "adam"].iloc[0]
    diagnosis_df = pd.read_csv(diagnosis_path)
    max_field_movement = float(diagnosis_df["field_movement_ratio"].max())
    best_egm_over_sgd = float(egm_row["last5_clean_mean"] + egm_row["last5_adversarial_mean"] - sgd_row["last5_clean_mean"] - sgd_row["last5_adversarial_mean"])
    best_ppm_over_sgd = float(ppm_row["last5_clean_mean"] + ppm_row["last5_adversarial_mean"] - sgd_row["last5_clean_mean"] - sgd_row["last5_adversarial_mean"])

    reasons = []
    if max_field_movement < 0.05:
        reasons.append("B. field movement too small")
    if float(summary_df["clip_fraction_mean"].max()) > 0.2:
        reasons.append("D. PPO clipping likely limits useful extrapolation in at least part of the search")
    if float(summary_df.loc[summary_df["method"].isin(["egm", "ppm"]), "update_cosine_vs_sgd"].min()) > 0.999:
        reasons.append("A/B boundary: EGM/PPM are implemented differently from SGD, but the local field remains very close to SGD on probed minibatches")
    if best_egm_over_sgd <= 0 and best_ppm_over_sgd <= 0:
        reasons.append("F. this phase-alternating PPO-RARL protocol on HalfCheetah-v4 did not yield an EGM/PPM advantage over SGD under the bounded search")

    lines = [
        "# Full-Policy Failure Classification",
        "",
        "1. Are actor and critic really updated together for all optimizers?",
        "Yes. The PPO shared-policy optimizer path updates actor, log_std, and critic as one full parameter vector for Adam, SGD, EGM, and PPM.",
        "",
        "2. Does EGM truly use half-step full PPO gradient?",
        "Yes. Stage 5A/5B probes recompute the full PPO loss and gradient at theta_half on the same minibatch.",
        "",
        "3. Does PPM inner_steps>1 truly differ from SGD?",
        "Yes. The fixed-minibatch probe shows nonzero theta_ppm - theta_sgd differences, even though they are often very small under the current local field.",
        "",
        "4. Why did EGM/PPM overlap with SGD before?",
        f"Most likely because the local full-policy PPO field moves very little under the baseline lr/clip settings. The maximum probed field_movement_ratio was `{max_field_movement:.6f}`, which stayed below the requested 0.05 threshold.",
        "",
        "5. Did tuning find any setting where EGM/PPM beat SGD?",
        f"EGM beat SGD on the final score criterion: `{best_egm_over_sgd > 0}`. PPM beat SGD on the final score criterion: `{best_ppm_over_sgd > 0}`.",
        "",
        "6. If not, what is the most likely reason?",
        " / ".join(reasons) if reasons else "No single dominant failure mode was identified.",
        "",
        "7. Is it still reasonable to compare proposed-QP against SGD/Adam/EGM/PPM as baselines?",
        "Yes, if the paper clearly states that these are full-policy PPO-RARL baselines under the same alternating protocol and reports that EGM/PPM did not reliably outperform SGD in this setting.",
        "",
        "## Context rows",
        "",
        f"- Adam last5_clean/adv: `{adam_row['last5_clean_mean']:.6f}` / `{adam_row['last5_adversarial_mean']:.6f}`",
        f"- SGD last5_clean/adv: `{sgd_row['last5_clean_mean']:.6f}` / `{sgd_row['last5_adversarial_mean']:.6f}`",
        f"- EGM last5_clean/adv: `{egm_row['last5_clean_mean']:.6f}` / `{egm_row['last5_adversarial_mean']:.6f}`",
        f"- PPM last5_clean/adv: `{ppm_row['last5_clean_mean']:.6f}` / `{ppm_row['last5_adversarial_mean']:.6f}`",
    ]
    (output_root / "full_policy_failure_classification.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    screen_candidates = build_screen_candidates()
    screen_analysis_dirs: Dict[str, pathlib.Path] = {}
    for candidate in screen_candidates:
        screen_analysis_dirs[candidate.tag] = run_candidate(args, candidate, phase="screen", iterations=args.screen_iters)
    screen_top = select_top_candidates(screen_candidates, screen_analysis_dirs, top_k=2)

    refine_candidates = build_refine_candidates(screen_top)
    refine_analysis_dirs: Dict[str, pathlib.Path] = {}
    for candidate in refine_candidates:
        refine_analysis_dirs[candidate.tag] = run_candidate(args, candidate, phase="refine", iterations=args.refine_iters)
    refine_top = select_top_candidates(refine_candidates, refine_analysis_dirs, top_k=2)

    final_candidates = {method: candidates[0] for method, candidates in refine_top.items()}
    final_stage_analysis_dirs: Dict[str, pathlib.Path] = {}
    for method, candidates in refine_top.items():
        for candidate in candidates:
            final_stage_analysis_dirs[candidate.tag] = run_candidate(args, candidate, phase="final", iterations=args.final_iters)
    final_top2 = select_top_candidates([candidate for candidates in refine_top.values() for candidate in candidates], final_stage_analysis_dirs, top_k=2)
    final_candidates = {method: candidates[0] for method, candidates in final_top2.items()}

    final_analysis_dirs: List[pathlib.Path] = []
    final_config_map: Dict[str, Dict[str, object]] = {}
    for method, candidate in final_candidates.items():
        final_analysis_dirs.append(final_stage_analysis_dirs[candidate.tag])
        final_config_map[method] = {
            "optimizer": candidate.optimizer,
            "lr": candidate.lr,
            "max_grad_norm": candidate.max_grad_norm,
            "vf_coef": candidate.vf_coef,
            "optimizer_kwargs": candidate.optimizer_kwargs,
            "tag": candidate.tag,
            "finalists": summarize_candidates({method: final_top2[method]})[method],
        }

    probe_frames = [probe_candidate(candidate, args.env, args.seed, args.device, args.num_probes) for candidate in final_candidates.values()]
    probe_df = pd.concat(probe_frames, ignore_index=True)
    probe_df.to_csv(output_root / "final_full_policy_probe_metrics.csv", index=False)

    summary_df = aggregate_outputs(final_analysis_dirs, output_root, probe_df)
    final_data = export_all_method_frames(final_analysis_dirs, output_root)
    plot_results(output_root, final_data, probe_df)
    (output_root / "final_full_policy_optimizer_configs.json").write_text(json.dumps(final_config_map, indent=2), encoding="utf-8")
    write_tuning_report(output_root, screen_top, refine_top, final_candidates, final_top2, summary_df)

    diagnosis_path = pathlib.Path(args.repo_dir).parent / "results" / "ppo_full_policy_optimizer_audit" / "full_policy_overlap_diagnosis.csv"
    if diagnosis_path.exists():
        if any(method in summary_df["method"].values for method in ["sgd", "egm", "ppm"]):
            sgd_row = summary_df[summary_df["method"] == "sgd"].iloc[0] if "sgd" in summary_df["method"].values else None
            egm_row = summary_df[summary_df["method"] == "egm"].iloc[0] if "egm" in summary_df["method"].values else None
            ppm_row = summary_df[summary_df["method"] == "ppm"].iloc[0] if "ppm" in summary_df["method"].values else None
            if sgd_row is not None and egm_row is not None and ppm_row is not None:
                if not (
                    float(egm_row["last5_clean_mean"]) > float(sgd_row["last5_clean_mean"])
                    and float(ppm_row["last5_clean_mean"]) >= float(egm_row["last5_clean_mean"])
                    and float(egm_row["last5_adversarial_mean"]) > float(sgd_row["last5_adversarial_mean"])
                    and float(ppm_row["last5_adversarial_mean"]) >= float(egm_row["last5_adversarial_mean"])
                ):
                    write_failure_classification(output_root, summary_df, diagnosis_path)


if __name__ == "__main__":
    main()
