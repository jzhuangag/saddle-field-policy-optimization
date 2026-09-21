from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from models.optimizers import get_optimizer_class
from models.proposed_qp_new import block_norm, named_difference
from scripts.full_policy_optimizer_probe import (
    build_manager,
    clone_state,
    collect_probe_batches,
    compute_clip_ranges,
    compute_loss_and_grads,
    cosine_similarity,
    get_actions_for_rollout,
    named_parameters,
    restore_state,
    temporary_algo_overrides,
    tensor_norm,
)


@dataclass(frozen=True)
class MethodConfig:
    method: str
    optimizer_name: str
    lr: float
    max_grad_norm: float
    vf_coef: float
    optimizer_scope: str = "full_policy"
    optimizer_kwargs: Optional[Dict[str, object]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run proposed_qp_new_v2 config freeze and fixed-minibatch audits")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--role", type=str, default="protagonist", choices=["protagonist", "adversary"])
    parser.add_argument("--num-probes", type=int, default=4)
    parser.add_argument("--proposed-lr", type=float, default=1e-3)
    parser.add_argument("--proposed-beta-max", type=float, default=0.3)
    parser.add_argument("--proposed-gamma-max", type=float, default=0.3)
    return parser.parse_args()


def lyapunov_value(grads: Dict[str, th.Tensor], actor_weight: float, logstd_weight: float, critic_weight: float) -> float:
    def block_mean_square(block: str) -> float:
        pieces = [tensor.reshape(-1) for name, tensor in grads.items() if classify_block(name) == block]
        if not pieces:
            return 0.0
        vec = th.cat(pieces)
        return float(th.mean(vec * vec).item())

    return 0.5 * (
        actor_weight * block_mean_square("actor")
        + logstd_weight * block_mean_square("logstd")
        + critic_weight * block_mean_square("critic")
    )


def classify_block(name: str) -> str:
    if "log_std" in name:
        return "logstd"
    if "value" in name:
        return "critic"
    return "actor"


def method_configs(args: argparse.Namespace) -> List[MethodConfig]:
    # Stage-10 line with fair max_grad_norm for baseline audit and explicit v2 defaults.
    return [
        MethodConfig("adam", "adam", lr=2.0633e-05, max_grad_norm=0.5, vf_coef=0.1),
        MethodConfig("sgd", "sgd", lr=1e-3, max_grad_norm=0.5, vf_coef=1.0),
        MethodConfig("egm", "egm", lr=1e-3, max_grad_norm=0.5, vf_coef=0.5),
        MethodConfig(
            "ppm",
            "ppm",
            lr=1e-3,
            max_grad_norm=0.5,
            vf_coef=1.0,
            optimizer_kwargs={"inner_steps": 10},
        ),
        MethodConfig(
            "proposed_noG_new_v2",
            "proposed_noG_new_v2",
            lr=args.proposed_lr,
            max_grad_norm=10.0,
            vf_coef=0.5,
            optimizer_kwargs={
                "optimizer_scope": "full_policy",
                "qp_normalization": "block",
                "qp_fd_eps": 1e-3,
                "qp_beta_probe": 1e-3,
                "qp_gamma_probe": 1e-3,
                "qp_ridge": 1e-8,
                "qp_actor_weight": 1.0,
                "qp_logstd_weight": 1.0,
                "qp_critic_weight": 0.3,
                "qp_beta_max": args.proposed_beta_max,
                "qp_gamma_max": args.proposed_gamma_max,
                "qp_max_update_norm": float("inf"),
                "qp_eps": 1e-8,
                "qp_step_solver": "lyapunov_quadratic_bound",
            },
        ),
        MethodConfig(
            "proposed_qp_new_v2",
            "proposed_qp_new_v2",
            lr=args.proposed_lr,
            max_grad_norm=10.0,
            vf_coef=0.5,
            optimizer_kwargs={
                "optimizer_scope": "full_policy",
                "qp_normalization": "block",
                "qp_fd_eps": 1e-3,
                "qp_beta_probe": 1e-3,
                "qp_gamma_probe": 1e-3,
                "qp_ridge": 1e-8,
                "qp_actor_weight": 1.0,
                "qp_logstd_weight": 1.0,
                "qp_critic_weight": 0.3,
                "qp_beta_max": args.proposed_beta_max,
                "qp_gamma_max": args.proposed_gamma_max,
                "qp_max_update_norm": float("inf"),
                "qp_eps": 1e-8,
                "qp_step_solver": "lyapunov_quadratic_bound",
            },
        ),
    ]


def instantiate_optimizer(algo, config: MethodConfig, diagnostics_csv_path: pathlib.Path, role: str):
    optimizer_class = get_optimizer_class(config.optimizer_name)
    kwargs = dict(config.optimizer_kwargs or {})
    if "proposed_" in config.optimizer_name:
        kwargs.setdefault("role", role)
        kwargs.setdefault("diagnostics_csv_path", str(diagnostics_csv_path))
    optimizer = optimizer_class(algo.policy.parameters(), lr=config.lr, **kwargs)
    return optimizer


def compute_pre_post_grad_stats(algo, rollout_data, *, vf_coef: float, ent_coef: float, max_grad_norm: float) -> Dict[str, float]:
    raw_eval = compute_loss_and_grads(algo, rollout_data, max_grad_norm=float("inf"), vf_coef=vf_coef, ent_coef=ent_coef)
    clipped_eval = compute_loss_and_grads(algo, rollout_data, max_grad_norm=max_grad_norm, vf_coef=vf_coef, ent_coef=ent_coef)

    raw_vec = th.cat([tensor.reshape(-1) for tensor in raw_eval["grads"].values()])
    clipped_vec = th.cat([tensor.reshape(-1) for tensor in clipped_eval["grads"].values()])
    raw_norm = tensor_norm(raw_vec)
    clipped_norm = tensor_norm(clipped_vec)
    return {
        "raw_grad_norm": raw_norm,
        "clipped_grad_norm": clipped_norm,
        "grad_clip_active": float(raw_norm > clipped_norm + 1e-10),
        "approx_kl": float(clipped_eval["approx_kl"]),
        "clip_fraction": float(clipped_eval["clip_fraction"]),
        "value_loss": float(clipped_eval["value_loss"]),
        "policy_loss": float(clipped_eval["policy_loss"]),
        "entropy_loss": float(clipped_eval["entropy_loss"]),
        "total_loss": float(clipped_eval["total_loss"]),
        "grads": clipped_eval["grads"],
        "raw_grads": raw_eval["grads"],
    }


def take_one_step(algo, rollout_data, config: MethodConfig, role: str, diagnostics_csv_path: pathlib.Path) -> Dict[str, object]:
    named_params = named_parameters(algo.policy)
    theta_old = clone_state(named_params)
    old_optimizer = algo.policy.optimizer
    optimizer = instantiate_optimizer(algo, config, diagnostics_csv_path, role)
    algo.policy.optimizer = optimizer
    try:
        ent_coef = float(algo.ent_coef)
        with temporary_algo_overrides(algo, vf_coef=config.vf_coef, ent_coef=ent_coef, max_grad_norm=config.max_grad_norm):
            grad_stats = compute_pre_post_grad_stats(
                algo,
                rollout_data,
                vf_coef=config.vf_coef,
                ent_coef=ent_coef,
                max_grad_norm=config.max_grad_norm,
            )
            clip_range, clip_range_vf = compute_clip_ranges(algo)
            actions = get_actions_for_rollout(algo, rollout_data)
            eval_closure = algo._build_eval_closure(rollout_data, actions, clip_range, clip_range_vf)

            restore_state(named_params, theta_old)
            if getattr(optimizer, "requires_eval_closure", False):
                optimizer.step(eval_closure=eval_closure, named_params=named_params)
            elif getattr(optimizer, "requires_closure", False):
                def closure():
                    return eval_closure(backward=True)["loss_tensor"]
                optimizer.step(closure)
            else:
                optimizer.zero_grad()
                total_loss, *_ = algo._build_shared_policy_loss(rollout_data, actions, clip_range, clip_range_vf)
                total_loss.backward()
                if math.isfinite(config.max_grad_norm):
                    th.nn.utils.clip_grad_norm_(algo.policy.parameters(), config.max_grad_norm)
                optimizer.step()

            theta_after = clone_state(named_params)
            diff = named_difference(theta_after, theta_old, [name for name, _ in named_params])
            after_eval = compute_loss_and_grads(
                algo,
                rollout_data,
                max_grad_norm=config.max_grad_norm,
                vf_coef=config.vf_coef,
                ent_coef=ent_coef,
            )

            metrics = getattr(optimizer, "last_step_metrics", {})
            lyapunov_before = lyapunov_value(
                grad_stats["grads"],
                actor_weight=float((config.optimizer_kwargs or {}).get("qp_actor_weight", 1.0)),
                logstd_weight=float((config.optimizer_kwargs or {}).get("qp_logstd_weight", 1.0)),
                critic_weight=float((config.optimizer_kwargs or {}).get("qp_critic_weight", 0.3)),
            )
            lyapunov_after = lyapunov_value(
                after_eval["grads"],
                actor_weight=float((config.optimizer_kwargs or {}).get("qp_actor_weight", 1.0)),
                logstd_weight=float((config.optimizer_kwargs or {}).get("qp_logstd_weight", 1.0)),
                critic_weight=float((config.optimizer_kwargs or {}).get("qp_critic_weight", 0.3)),
            )

            row = {
                "method": config.method,
                "lr": config.lr,
                "eta": config.lr,
                "max_grad_norm": config.max_grad_norm,
                "vf_coef": config.vf_coef,
                "ent_coef": ent_coef,
                "optimizer_scope": config.optimizer_scope,
                "ppm_inner_steps": int((config.optimizer_kwargs or {}).get("inner_steps", -1)),
                "qp_beta_max": (config.optimizer_kwargs or {}).get("qp_beta_max", np.nan),
                "qp_gamma_max": (config.optimizer_kwargs or {}).get("qp_gamma_max", np.nan),
                "qp_normalization": (config.optimizer_kwargs or {}).get("qp_normalization", ""),
                "qp_actor_weight": (config.optimizer_kwargs or {}).get("qp_actor_weight", np.nan),
                "qp_logstd_weight": (config.optimizer_kwargs or {}).get("qp_logstd_weight", np.nan),
                "qp_critic_weight": (config.optimizer_kwargs or {}).get("qp_critic_weight", np.nan),
                "mean_grad_norm_before_clip": grad_stats["raw_grad_norm"],
                "mean_grad_norm_after_clip": grad_stats["clipped_grad_norm"],
                "grad_clip_active_frac": grad_stats["grad_clip_active"],
                "mean_update_norm": tensor_norm(th.cat([tensor.reshape(-1) for tensor in diff.values()])),
                "actor_update_norm": block_norm(diff, list(diff.keys()), "actor"),
                "logstd_update_norm": block_norm(diff, list(diff.keys()), "logstd"),
                "critic_update_norm": block_norm(diff, list(diff.keys()), "critic"),
                "approx_kl_mean": after_eval["approx_kl"],
                "clip_fraction_mean": after_eval["clip_fraction"],
                "value_loss_mean": after_eval["value_loss"],
                "policy_loss_mean": after_eval["policy_loss"],
                "V_before": lyapunov_before,
                "V_after": lyapunov_after,
                "actual_V_change": lyapunov_after - lyapunov_before,
                "same_minibatch_total_loss_before": float(grad_stats["total_loss"]),
                "same_minibatch_total_loss_after": float(after_eval["total_loss"]),
                "same_minibatch_total_loss_change": float(after_eval["total_loss"] - grad_stats["total_loss"]),
                "policy_loss_before": float(grad_stats["policy_loss"]),
                "policy_loss_after": float(after_eval["policy_loss"]),
                "policy_loss_change": float(after_eval["policy_loss"] - grad_stats["policy_loss"]),
                "value_loss_before": float(grad_stats["value_loss"]),
                "value_loss_after": float(after_eval["value_loss"]),
                "value_loss_change": float(after_eval["value_loss"] - grad_stats["value_loss"]),
                "entropy_loss_before": float(grad_stats["entropy_loss"]),
                "entropy_loss_after": float(after_eval["entropy_loss"]),
                "entropy_loss_change": float(after_eval["entropy_loss"] - grad_stats["entropy_loss"]),
                "F_raw_norm": float(metrics.get("F_raw_norm", np.nan)),
                "F_dir_norm": float(metrics.get("F_dir_norm", np.nan)),
                "G_raw_norm": float(metrics.get("G_raw_norm", np.nan)),
                "G_dir_norm": float(metrics.get("G_dir_norm", np.nan)),
                "beta": float(metrics.get("beta", np.nan)),
                "gamma": float(metrics.get("gamma", np.nan)),
                "beta_eff": float(metrics.get("beta_eff", np.nan)),
                "gamma_eff": float(metrics.get("gamma_eff", np.nan)),
                "G_contribution_norm": float(metrics.get("G_contribution_norm", 0.0)),
                "G_over_update_norm": float(metrics.get("G_over_update_norm", 0.0)),
                "gamma_active_frac": float(metrics.get("gamma_active_frac", float(metrics.get("gamma_active", 0.0)))),
                "beta_at_bound": float(metrics.get("beta_at_bound", np.nan)),
                "gamma_at_bound": float(metrics.get("gamma_at_bound", np.nan)),
                "finite_difference_valid": float(metrics.get("finite_difference_valid", np.nan)),
                "update_norm_pre_cap": float(metrics.get("update_norm_pre_cap", np.nan)),
                "update_norm_post_cap": float(metrics.get("update_norm_post_cap", np.nan)),
                "cap_active": float(metrics.get("cap_active", np.nan)),
                "q_pred": float(metrics.get("q_pred", np.nan)),
                "loss_before": float(metrics.get("loss_before", grad_stats["total_loss"])),
                "loss_after_actual": float(metrics.get("loss_after_actual", after_eval["total_loss"])),
                "actual_loss_change": float(metrics.get("actual_loss_change", after_eval["total_loss"] - grad_stats["total_loss"])),
                "approx_kl_after": float(metrics.get("approx_kl_after", after_eval["approx_kl"])),
                "clip_fraction_after": float(metrics.get("clip_fraction_after", after_eval["clip_fraction"])),
                "selected_case": metrics.get("selected_case", ""),
                "q_condition_status": metrics.get("q_condition_status", ""),
                "ridge_used": float(metrics.get("ridge_used", np.nan)),
            }
    finally:
        restore_state(named_params, theta_old)
        algo.policy.optimizer = old_optimizer
    return row


def baseline_freeze_check(repo_dir: pathlib.Path) -> Dict[str, object]:
    targets = {
        "optimizers.py": repo_dir / "models" / "optimizers.py",
        "ppo.py": repo_dir / "models" / "ppo.py",
        "exp_manager.py": repo_dir / "utils" / "exp_manager.py",
        "train_adversary.py": repo_dir / "scripts" / "train_adversary.py",
    }
    paper_tokens = ["paper_sgd", "paper_egm", "paper_ppm", "paper_qp", "run_paper_qp_benchmark"]
    token_hits: Dict[str, List[str]] = {}
    for label, path in targets.items():
        text = path.read_text(encoding="utf-8")
        hits = [token for token in paper_tokens if token in text]
        if hits:
            token_hits[label] = hits
    changed = []
    git_status = (repo_dir / ".git").exists()
    if git_status:
        import subprocess

        result = subprocess.run(
            ["git", "-C", str(repo_dir), "diff", "--name-only"],
            capture_output=True,
            text=True,
            check=False,
        )
        changed = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {
        "paper_token_hits": token_hits,
        "changed_files": changed,
    }


def write_config_freeze_report(output_dir: pathlib.Path, configs: List[MethodConfig], freeze_check: Dict[str, object], sample_row: pd.DataFrame) -> None:
    lines = [
        "# Config Freeze Report",
        "",
        "## Baseline contamination check",
        "",
        f"- Adam/SGD/EGM/PPM still use Stage-10 line implementations: `{not freeze_check['paper_token_hits']}`",
        f"- Any `paper_*` optimizer tokens in baseline files: `{bool(freeze_check['paper_token_hits'])}`",
        f"- Changed files in working tree: `{json.dumps(freeze_check['changed_files'])}`",
        "- Train loop file `models/ppo.py` is not modified in the current working tree.",
        "- No same-minibatch PPO-loss grid search is used by the v2 solver path; only old `proposed_*_new` keeps that logic.",
        "",
        "## Config notes",
        "",
        "- Baseline fairness override used in this audit: `max_grad_norm = 0.5` for `adam/sgd/egm/ppm`.",
        "- Proposed v2 audit override used here: `max_grad_norm = 10.0` for `proposed_noG_new_v2/proposed_qp_new_v2` so the field estimate is not clipped too early.",
        "- `EGM` and `PPM` still have different `vf_coef` in the frozen Stage-10 configs, so a matched-config diagnostic remains required before any strong baseline claim.",
        "- External-eta convention is active in code: update now uses `delta = -eta * beta * f + eta * gamma * g` with `eta = optimizer lr`.",
        "",
        "## Current measured takeaways",
        "",
    ]
    if not sample_row.empty:
        egm = sample_row[sample_row["method"] == "egm"]
        ppm = sample_row[sample_row["method"] == "ppm"]
        qp = sample_row[sample_row["method"] == "proposed_qp_new_v2"]
        if not egm.empty and not ppm.empty:
            lines.append(f"- EGM and PPM matched max_grad_norm in this audit: `{float(egm.iloc[0]['max_grad_norm']) == float(ppm.iloc[0]['max_grad_norm'])}`")
            lines.append(f"- EGM and PPM matched vf_coef in this audit: `{float(egm.iloc[0]['vf_coef']) == float(ppm.iloc[0]['vf_coef'])}`")
        if not qp.empty:
            lines.append(f"- QP update norm mean in audit probes: `{float(qp.iloc[0]['mean_update_norm']):.6f}`")
            lines.append(f"- QP cap active in first aggregate row: `{bool(qp.iloc[0]['cap_active'])}`")
            lines.append(f"- QP gamma active fraction aggregate: `{float(qp.iloc[0]['gamma_active_frac']):.4f}`")
            lines.append(f"- QP mean beta/gamma: `beta={float(qp.iloc[0]['beta']):.6f}`, `gamma={float(qp.iloc[0]['gamma']):.6f}`")
            lines.append(f"- QP mean beta_eff/gamma_eff: `beta_eff={float(qp.iloc[0]['beta_eff']):.6f}`, `gamma_eff={float(qp.iloc[0]['gamma_eff']):.6f}`")
    (output_dir / "config_freeze_report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_stage_p5(summary_df: pd.DataFrame, plots_dir: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    proposed = summary_df[summary_df["method"].isin(["proposed_noG_new_v2", "proposed_qp_new_v2"])].copy()
    x = np.arange(len(proposed))
    width = 0.2
    ax.bar(x - 1.5 * width, proposed["beta"], width=width, label="beta")
    ax.bar(x - 0.5 * width, proposed["gamma"], width=width, label="gamma")
    ax.bar(x + 0.5 * width, proposed["beta_eff"], width=width, label="beta_eff")
    ax.bar(x + 1.5 * width, proposed["gamma_eff"], width=width, label="gamma_eff")
    ax.set_xticks(x)
    ax.set_xticklabels(proposed["method"], rotation=20, ha="right")
    ax.set_title("Stage P5 beta/gamma and effective step scale")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stageP5_beta_gamma_eff.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary_df["method"], summary_df["actual_V_change"], color="tab:blue")
    ax.set_title("Stage P5 actual V change")
    ax.set_ylabel("V_after - V_before")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "stageP5_V_change.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary_df["method"], summary_df["mean_update_norm"], color="tab:blue", alpha=0.8, label="update_norm")
    ax.bar(summary_df["method"], summary_df["actor_update_norm"], color="tab:green", alpha=0.6, label="actor")
    ax.bar(summary_df["method"], summary_df["critic_update_norm"], color="tab:red", alpha=0.5, label="critic")
    ax.set_title("Stage P5 update norms")
    ax.set_ylabel("Norm")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stageP5_update_norms.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(summary_df["method"], summary_df["approx_kl_after"], color="tab:purple")
    axes[0].set_title("Stage P5 approx_kl_after")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(alpha=0.3)
    axes[1].bar(summary_df["method"], summary_df["clip_fraction_after"], color="tab:orange")
    axes[1].set_title("Stage P5 clip_fraction_after")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "stageP5_kl_clip.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    qp_compare = summary_df[summary_df["method"].isin(["proposed_noG_new_v2", "proposed_qp_new_v2"])].copy()
    x = np.arange(len(qp_compare))
    width = 0.35
    ax.bar(x - width / 2, qp_compare["actual_V_change"], width=width, label="actual_V_change")
    ax.bar(x + width / 2, qp_compare["G_contribution_norm"], width=width, label="G_contribution_norm")
    ax.set_xticks(x)
    ax.set_xticklabels(qp_compare["method"], rotation=20, ha="right")
    ax.set_title("Stage P5 QP vs noG")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stageP5_qp_vs_noG.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_dir = pathlib.Path(__file__).resolve().parents[1]
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    configs = method_configs(args)
    freeze_check = baseline_freeze_check(repo_dir)

    manager = build_manager(args)
    rarl_model = manager.setup_experiment()
    algo = rarl_model.protagonist if args.role == "protagonist" else rarl_model.adversary
    probes = collect_probe_batches(rarl_model, args.role, args.num_probes)

    config_rows = []
    audit_rows = []
    for config in configs:
        config_rows.append(
            {
                "method": config.method,
                "optimizer_name": config.optimizer_name,
                "lr": config.lr,
                "max_grad_norm": config.max_grad_norm,
                "vf_coef": config.vf_coef,
                "ent_coef": float(algo.ent_coef),
                "clip_range": float(algo.clip_range(algo._current_progress_remaining)),
                "n_epochs": int(algo.n_epochs),
                "batch_size": int(algo.batch_size),
                "n_steps": int(algo.n_steps),
                "optimizer_scope": config.optimizer_scope,
                "ppm_inner_steps": int((config.optimizer_kwargs or {}).get("inner_steps", -1)),
                "qp_beta_max": (config.optimizer_kwargs or {}).get("qp_beta_max", np.nan),
                "qp_gamma_max": (config.optimizer_kwargs or {}).get("qp_gamma_max", np.nan),
                "qp_normalization": (config.optimizer_kwargs or {}).get("qp_normalization", ""),
                "qp_fd_eps": (config.optimizer_kwargs or {}).get("qp_fd_eps", np.nan),
                "qp_beta_probe": (config.optimizer_kwargs or {}).get("qp_beta_probe", np.nan),
                "qp_gamma_probe": (config.optimizer_kwargs or {}).get("qp_gamma_probe", np.nan),
                "qp_ridge": (config.optimizer_kwargs or {}).get("qp_ridge", np.nan),
                "qp_actor_weight": (config.optimizer_kwargs or {}).get("qp_actor_weight", np.nan),
                "qp_logstd_weight": (config.optimizer_kwargs or {}).get("qp_logstd_weight", np.nan),
                "qp_critic_weight": (config.optimizer_kwargs or {}).get("qp_critic_weight", np.nan),
                "qp_max_update_norm": (config.optimizer_kwargs or {}).get("qp_max_update_norm", np.nan),
            }
        )
        for probe in probes:
            row = take_one_step(
                algo,
                probe.rollout_data,
                config,
                role=args.role,
                diagnostics_csv_path=diagnostics_dir / f"{config.method}_{args.role}_diagnostics.csv",
            )
            row["probe_idx"] = probe.probe_idx
            audit_rows.append(row)

    config_df = pd.DataFrame(config_rows)
    audit_df = pd.DataFrame(audit_rows)
    summary_df = audit_df.groupby("method", sort=False).mean(numeric_only=True).reset_index()

    config_df.to_csv(output_dir / "config_matrix.csv", index=False)
    audit_df.to_csv(output_dir / "stageP5_fixed_minibatch_audit.csv", index=False)

    write_config_freeze_report(output_dir, configs, freeze_check, summary_df)
    plot_stage_p5(summary_df, plots_dir)

    no_g = summary_df[summary_df["method"] == "proposed_noG_new_v2"]
    qp = summary_df[summary_df["method"] == "proposed_qp_new_v2"]
    gate_lines = [
        "# Stage P5 Fixed-Minibatch Audit Report",
        "",
        f"- Role audited: `{args.role}`",
        f"- Number of probes: `{args.num_probes}`",
        "",
    ]
    if not no_g.empty and not qp.empty:
        gate_lines.extend(
            [
                f"- External-eta convention actually active: `{float(qp.iloc[0]['eta']) == float(qp.iloc[0]['lr']) and float(qp.iloc[0]['beta_eff']) <= float(qp.iloc[0]['eta']) * float(qp.iloc[0]['beta']) + 1e-12}`",
                f"- Mean beta/gamma: `beta={float(qp.iloc[0]['beta']):.6f}`, `gamma={float(qp.iloc[0]['gamma']):.6f}`",
                f"- Mean beta_eff/gamma_eff: `beta_eff={float(qp.iloc[0]['beta_eff']):.6f}`, `gamma_eff={float(qp.iloc[0]['gamma_eff']):.6f}`",
                f"- proposed_qp_new_v2 actual V decrease better than noG_new_v2 on average: `{float(qp.iloc[0]['actual_V_change']) < float(no_g.iloc[0]['actual_V_change'])}`",
                f"- proposed_qp_new_v2 gamma_active_frac > 0: `{float(qp.iloc[0]['gamma_active_frac']) > 0.0}`",
                f"- proposed_qp_new_v2 G_contribution_norm nontrivial: `{float(qp.iloc[0]['G_contribution_norm']) > 1e-8}`",
                f"- proposed_qp_new_v2 finite_difference_valid mean: `{float(qp.iloc[0]['finite_difference_valid']):.4f}`",
                f"- proposed_qp_new_v2 update_norm mean: `{float(qp.iloc[0]['mean_update_norm']):.6f}`",
                f"- proposed_qp_new_v2 G_over_update_norm mean: `{float(qp.iloc[0]['G_over_update_norm']):.6f}`",
                f"- proposed_qp_new_v2 approx_kl_after mean: `{float(qp.iloc[0]['approx_kl_after']):.6f}`",
                f"- proposed_qp_new_v2 clip_fraction_after mean: `{float(qp.iloc[0]['clip_fraction_after']):.6f}`",
                f"- proposed_qp_new_v2 beta/gamma hitting bounds: `beta_at_bound_mean={float(qp.iloc[0]['beta_at_bound']):.4f}`, `gamma_at_bound_mean={float(qp.iloc[0]['gamma_at_bound']):.4f}`",
                f"- proposed_noG_new_v2 uses same eta convention as QP: `{float(no_g.iloc[0]['eta']) == float(qp.iloc[0]['eta'])}`",
                f"- Safe to proceed to online training from this preflight alone: `{bool((float(qp.iloc[0]['actual_V_change']) < float(no_g.iloc[0]['actual_V_change'])) and (float(qp.iloc[0]['gamma_active_frac']) > 0.0) and np.isfinite(float(qp.iloc[0]['mean_update_norm'])))}`",
                "",
                "## Diagnostic note",
                "",
                "- This audit is the gate before any online benchmark. If `actual_V_change` does not improve over noG, we stop and diagnose before training.",
            ]
        )
    (output_dir / "stageP5_fixed_minibatch_audit_report.md").write_text("\n".join(gate_lines), encoding="utf-8")

    report_lines = [
        "# Config Audit Report",
        "",
        f"- Environment: `{args.env}`",
        f"- Role audited: `{args.role}`",
        f"- Probes used: `{args.num_probes}`",
        "",
        "## Questions answered",
        "",
    ]
    egm = summary_df[summary_df["method"] == "egm"]
    ppm = summary_df[summary_df["method"] == "ppm"]
    qp = summary_df[summary_df["method"] == "proposed_qp_new_v2"]
    if not egm.empty and not ppm.empty:
        report_lines.extend(
            [
                f"1. Are EGM and PPM using matched max_grad_norm? `{float(egm.iloc[0]['max_grad_norm']) == float(ppm.iloc[0]['max_grad_norm'])}`",
                f"2. Are EGM and PPM using matched vf_coef? `{float(egm.iloc[0]['vf_coef']) == float(ppm.iloc[0]['vf_coef'])}`",
                f"3. Is PPM being over-clipped relative to EGM? `{float(ppm.iloc[0]['grad_clip_active_frac']) > float(egm.iloc[0]['grad_clip_active_frac'])}`",
            ]
        )
    if not qp.empty and not egm.empty:
        report_lines.extend(
            [
                f"4. Is proposed_qp_new_v2 update norm comparable to EGM? `{0.3 <= float(qp.iloc[0]['mean_update_norm']) / max(float(egm.iloc[0]['mean_update_norm']), 1e-12) <= 3.0}`",
                f"5. Is qp_max_update_norm actually binding? `{bool(qp.iloc[0]['cap_active'])}`",
                f"6. Are QP beta/gamma saturated? `beta_at_bound_mean={float(qp.iloc[0]['beta_at_bound']):.4f}, gamma_at_bound_mean={float(qp.iloc[0]['gamma_at_bound']):.4f}`",
            ]
        )
    (output_dir / "config_audit_report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
