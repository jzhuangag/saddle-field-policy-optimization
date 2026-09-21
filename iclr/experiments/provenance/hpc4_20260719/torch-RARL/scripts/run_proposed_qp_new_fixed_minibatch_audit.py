from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from models.optimizers import get_optimizer_class
from scripts.full_policy_optimizer_probe import (
    build_manager,
    block_tensor,
    clone_state,
    collect_probe_batches,
    compute_clip_ranges,
    compute_loss_and_grads,
    cosine_similarity,
    egm_update,
    flatten_named_tensors,
    named_parameters,
    ppm_update,
    restore_state,
    sgd_update,
    temporary_algo_overrides,
    tensor_norm,
    vector_from_state_diff,
)


@dataclass(frozen=True)
class MethodConfig:
    method: str
    optimizer: str
    lr: float
    max_grad_norm: float
    vf_coef: float
    optimizer_kwargs: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 5 fixed-minibatch audit for proposed_qp_new")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-probes-per-role", type=int, default=2)
    return parser.parse_args()


def method_configs() -> List[MethodConfig]:
    common_qp = {
        "optimizer_scope": "full_policy",
        "qp_normalization": "block",
        "qp_alpha": 0.3,
        "qp_beta_max": 1.0,
        "qp_gamma_max": 1.0,
        "qp_max_update_norm": 1.0,
        "qp_step_grid": "0,0.1,0.3,1.0,3.0",
        "qp_objective": "loss",
        "qp_accept_rule": "none",
        "qp_min_g_contribution": 0.0,
        "qp_critic_weight": 1.0,
        "qp_eps": 1e-8,
    }
    return [
        MethodConfig("sgd", "sgd", 1e-3, 1.0, 1.0, {}),
        MethodConfig("egm", "egm", 1e-3, 10.0, 0.5, {}),
        MethodConfig("ppm", "ppm", 1e-3, 1.0, 1.0, {"inner_steps": 10}),
        MethodConfig("proposed_noG_new", "proposed_noG_new", 1e-3, 10.0, 0.5, dict(common_qp)),
        MethodConfig("proposed_qp_new_plus", "proposed_qp_new", 1e-3, 10.0, 0.5, dict(common_qp, qp_g_sign="plus")),
        MethodConfig("proposed_qp_new_minus", "proposed_qp_new", 1e-3, 10.0, 0.5, dict(common_qp, qp_g_sign="minus")),
    ]


def build_eval_closure(algo, rollout_data):
    clip_range, clip_range_vf = compute_clip_ranges(algo)
    actions = rollout_data.actions.long().flatten() if algo.action_space.__class__.__name__ == "Discrete" else rollout_data.actions
    return algo._build_eval_closure(rollout_data, actions, clip_range, clip_range_vf)


def evaluate_method(algo, rollout_data, cfg: MethodConfig, role: str) -> Dict[str, object]:
    named_params = algo._named_policy_parameters()
    theta_old = clone_state(named_params)
    with temporary_algo_overrides(algo, vf_coef=cfg.vf_coef, ent_coef=float(algo.ent_coef), max_grad_norm=cfg.max_grad_norm):
        old_eval = compute_loss_and_grads(
            algo,
            rollout_data,
            max_grad_norm=cfg.max_grad_norm,
            vf_coef=cfg.vf_coef,
            ent_coef=float(algo.ent_coef),
        )

        if cfg.optimizer == "sgd":
            theta_new = sgd_update(theta_old, old_eval["grads"], cfg.lr)
            restore_state(named_params, theta_new)
            new_eval = compute_loss_and_grads(
                algo,
                rollout_data,
                max_grad_norm=cfg.max_grad_norm,
                vf_coef=cfg.vf_coef,
                ent_coef=float(algo.ent_coef),
            )
            metrics = {
                "F_norm_raw": tensor_norm(flatten_named_tensors(old_eval["grads"])),
                "F_norm_used": tensor_norm(flatten_named_tensors(old_eval["grads"])),
                "G_norm_raw": 0.0,
                "G_norm_used": 0.0,
                "cosine_F_G": float("nan"),
                "beta": cfg.lr,
                "gamma": 0.0,
                "beta_selected": cfg.lr,
                "gamma_selected": 0.0,
                "selected_candidate_rank": 1,
                "best_noG_candidate_loss": float(new_eval["total_loss"]),
                "best_qp_candidate_loss": float(new_eval["total_loss"]),
                "G_contribution_norm": 0.0,
                "G_over_update_norm": 0.0,
                "gamma_active_frac": 0.0,
                "zero_update_flag": 0.0,
                "g_sign_used": "none",
                "same_minibatch_loss_before": float(old_eval["total_loss"]),
                "same_minibatch_loss_after": float(new_eval["total_loss"]),
                "same_minibatch_loss_change": float(new_eval["total_loss"]) - float(old_eval["total_loss"]),
                "policy_loss_change": float(new_eval["policy_loss"]) - float(old_eval["policy_loss"]),
                "value_loss_change": float(new_eval["value_loss"]) - float(old_eval["value_loss"]),
                "entropy_change": float(new_eval["entropy_loss"]) - float(old_eval["entropy_loss"]),
                "approx_kl_change": float(new_eval["approx_kl"]) - float(old_eval["approx_kl"]),
                "clip_fraction_change": float(new_eval["clip_fraction"]) - float(old_eval["clip_fraction"]),
            }
        elif cfg.optimizer == "egm":
            _, _, _, theta_new = egm_update(
                algo,
                rollout_data,
                theta_old,
                cfg.lr,
                max_grad_norm=cfg.max_grad_norm,
                vf_coef=cfg.vf_coef,
                ent_coef=float(algo.ent_coef),
            )
            restore_state(named_params, theta_new)
            new_eval = compute_loss_and_grads(
                algo,
                rollout_data,
                max_grad_norm=cfg.max_grad_norm,
                vf_coef=cfg.vf_coef,
                ent_coef=float(algo.ent_coef),
            )
            metrics = {
                "F_norm_raw": tensor_norm(flatten_named_tensors(old_eval["grads"])),
                "F_norm_used": tensor_norm(flatten_named_tensors(old_eval["grads"])),
                "G_norm_raw": 0.0,
                "G_norm_used": 0.0,
                "cosine_F_G": float("nan"),
                "beta": cfg.lr,
                "gamma": 0.0,
                "beta_selected": cfg.lr,
                "gamma_selected": 0.0,
                "selected_candidate_rank": 1,
                "best_noG_candidate_loss": float(new_eval["total_loss"]),
                "best_qp_candidate_loss": float(new_eval["total_loss"]),
                "G_contribution_norm": 0.0,
                "G_over_update_norm": 0.0,
                "gamma_active_frac": 0.0,
                "zero_update_flag": 0.0,
                "g_sign_used": "none",
                "same_minibatch_loss_before": float(old_eval["total_loss"]),
                "same_minibatch_loss_after": float(new_eval["total_loss"]),
                "same_minibatch_loss_change": float(new_eval["total_loss"]) - float(old_eval["total_loss"]),
                "policy_loss_change": float(new_eval["policy_loss"]) - float(old_eval["policy_loss"]),
                "value_loss_change": float(new_eval["value_loss"]) - float(old_eval["value_loss"]),
                "entropy_change": float(new_eval["entropy_loss"]) - float(old_eval["entropy_loss"]),
                "approx_kl_change": float(new_eval["approx_kl"]) - float(old_eval["approx_kl"]),
                "clip_fraction_change": float(new_eval["clip_fraction"]) - float(old_eval["clip_fraction"]),
            }
        elif cfg.optimizer == "ppm":
            _, theta_new, *_ = ppm_update(
                algo,
                rollout_data,
                theta_old,
                cfg.lr,
                inner_steps=int(cfg.optimizer_kwargs["inner_steps"]),
                max_grad_norm=cfg.max_grad_norm,
                vf_coef=cfg.vf_coef,
                ent_coef=float(algo.ent_coef),
            )
            restore_state(named_params, theta_new)
            new_eval = compute_loss_and_grads(
                algo,
                rollout_data,
                max_grad_norm=cfg.max_grad_norm,
                vf_coef=cfg.vf_coef,
                ent_coef=float(algo.ent_coef),
            )
            metrics = {
                "F_norm_raw": tensor_norm(flatten_named_tensors(old_eval["grads"])),
                "F_norm_used": tensor_norm(flatten_named_tensors(old_eval["grads"])),
                "G_norm_raw": 0.0,
                "G_norm_used": 0.0,
                "cosine_F_G": float("nan"),
                "beta": cfg.lr,
                "gamma": 0.0,
                "beta_selected": cfg.lr,
                "gamma_selected": 0.0,
                "selected_candidate_rank": 1,
                "best_noG_candidate_loss": float(new_eval["total_loss"]),
                "best_qp_candidate_loss": float(new_eval["total_loss"]),
                "G_contribution_norm": 0.0,
                "G_over_update_norm": 0.0,
                "gamma_active_frac": 0.0,
                "zero_update_flag": 0.0,
                "g_sign_used": "none",
                "same_minibatch_loss_before": float(old_eval["total_loss"]),
                "same_minibatch_loss_after": float(new_eval["total_loss"]),
                "same_minibatch_loss_change": float(new_eval["total_loss"]) - float(old_eval["total_loss"]),
                "policy_loss_change": float(new_eval["policy_loss"]) - float(old_eval["policy_loss"]),
                "value_loss_change": float(new_eval["value_loss"]) - float(old_eval["value_loss"]),
                "entropy_change": float(new_eval["entropy_loss"]) - float(old_eval["entropy_loss"]),
                "approx_kl_change": float(new_eval["approx_kl"]) - float(old_eval["approx_kl"]),
                "clip_fraction_change": float(new_eval["clip_fraction"]) - float(old_eval["clip_fraction"]),
            }
        else:
            eval_closure = build_eval_closure(algo, rollout_data)
            optimizer_class = get_optimizer_class(cfg.optimizer)
            optimizer = optimizer_class(
                [param for _, param in named_params],
                lr=cfg.lr,
                diagnostics_csv_path=None,
                role=role,
                **cfg.optimizer_kwargs,
            )
            optimizer.step(eval_closure=eval_closure, named_params=named_params)
            theta_new = clone_state(named_params)
            new_eval = compute_loss_and_grads(
                algo,
                rollout_data,
                max_grad_norm=cfg.max_grad_norm,
                vf_coef=cfg.vf_coef,
                ent_coef=float(algo.ent_coef),
            )
            metrics = dict(optimizer.last_step_metrics)

    diff = vector_from_state_diff(theta_new, theta_old)
    update_vec = flatten_named_tensors(diff)
    row = {
        "method": cfg.method,
        "active_role": role,
        "scope": "full_policy",
        "F_norm_raw": float(metrics["F_norm_raw"]),
        "F_norm_used": float(metrics["F_norm_used"]),
        "G_norm_raw": float(metrics["G_norm_raw"]),
        "G_norm_used": float(metrics["G_norm_used"]),
        "cosine_F_G": float(metrics["cosine_F_G"]) if pd.notna(metrics["cosine_F_G"]) else np.nan,
        "beta": float(metrics["beta"]),
        "gamma": float(metrics["gamma"]),
        "beta_selected": float(metrics["beta_selected"]),
        "gamma_selected": float(metrics["gamma_selected"]),
        "update_norm": tensor_norm(update_vec),
        "actor_update_norm": tensor_norm(block_tensor(diff, "actor")),
        "logstd_update_norm": tensor_norm(block_tensor(diff, "logstd")),
        "critic_update_norm": tensor_norm(block_tensor(diff, "critic")),
        "same_minibatch_loss_before": float(metrics["same_minibatch_loss_before"]),
        "same_minibatch_loss_after": float(metrics["same_minibatch_loss_after"]),
        "same_minibatch_loss_change": float(metrics["same_minibatch_loss_change"]),
        "policy_loss_change": float(metrics["policy_loss_change"]),
        "value_loss_change": float(metrics["value_loss_change"]),
        "entropy_change": float(metrics["entropy_change"]),
        "approx_kl_change": float(metrics["approx_kl_change"]),
        "clip_fraction_change": float(metrics["clip_fraction_change"]),
        "G_contribution_norm": float(metrics["G_contribution_norm"]),
        "G_over_update_norm": float(metrics["G_over_update_norm"]),
        "noG_candidate_loss": float(metrics["best_noG_candidate_loss"]),
        "QP_candidate_loss": float(metrics["best_qp_candidate_loss"]),
        "selected_candidate_rank": int(metrics["selected_candidate_rank"]),
        "gamma_active_frac": float(metrics["gamma_active_frac"]),
        "zero_update_flag": float(metrics["zero_update_flag"]),
        "g_sign_used": metrics["g_sign_used"],
        "optimizer_name": cfg.optimizer,
    }
    restore_state(named_params, theta_old)
    return row


def main() -> None:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manager = build_manager(args)
    manager.adv_impact = "control"
    rarl_model = manager.setup_experiment()
    rows: List[Dict[str, object]] = []

    for role in ("protagonist", "adversary"):
        algo = rarl_model.protagonist if role == "protagonist" else rarl_model.adversary
        probes = collect_probe_batches(rarl_model, role, args.num_probes_per_role)
        for probe in probes:
            for cfg in method_configs():
                row = evaluate_method(algo, probe.rollout_data, cfg, role)
                row["probe_idx"] = probe.probe_idx
                rows.append(row)

    audit_df = pd.DataFrame(rows)
    csv_path = output_dir / "stage5_fixed_minibatch_audit.csv"
    audit_df.to_csv(csv_path, index=False)

    plus_df = audit_df[audit_df["method"] == "proposed_qp_new_plus"]
    minus_df = audit_df[audit_df["method"] == "proposed_qp_new_minus"]
    plus_score = float(plus_df["same_minibatch_loss_change"].mean())
    minus_score = float(minus_df["same_minibatch_loss_change"].mean())
    chosen_sign = "plus" if plus_score <= minus_score else "minus"
    (output_dir / "stage5_selected_qp_sign.json").write_text(json.dumps({"chosen_sign": chosen_sign}, indent=2), encoding="utf-8")

    main_qp_df = plus_df if chosen_sign == "plus" else minus_df
    nog_df = audit_df[audit_df["method"] == "proposed_noG_new"]
    pass_gate = (
        (main_qp_df["gamma_active_frac"].mean() > 0.0)
        and (main_qp_df["G_contribution_norm"].mean() > 1e-8)
        and (main_qp_df["same_minibatch_loss_change"].mean() <= nog_df["same_minibatch_loss_change"].mean() + 1e-8)
        and (main_qp_df["zero_update_flag"].mean() < 1.0)
    )

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    summary = audit_df.groupby("method").mean(numeric_only=True).reset_index()
    axes[0, 0].bar(summary["method"], summary["same_minibatch_loss_change"])
    axes[0, 0].set_title("Mean same-minibatch loss change")
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 1].bar(summary["method"], summary["G_contribution_norm"])
    axes[0, 1].set_title("Mean G contribution norm")
    axes[0, 1].tick_params(axis="x", rotation=20)
    axes[1, 0].bar(summary["method"], summary["update_norm"])
    axes[1, 0].set_title("Mean update norm")
    axes[1, 0].tick_params(axis="x", rotation=20)
    qp_diag = audit_df[audit_df["method"].isin(["proposed_qp_new_plus", "proposed_qp_new_minus", "proposed_noG_new"])]
    for method, group in qp_diag.groupby("method"):
        axes[1, 1].plot(group.index, group["gamma_active_frac"], marker="o", label=method)
    axes[1, 1].set_title("Gamma active fraction per probe")
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stage5_fixed_minibatch_audit.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    report_lines = [
        "# Stage 5 Fixed-Minibatch Audit Report",
        "",
        f"- Environment: `{args.env}`",
        f"- Device: `{args.device}`",
        f"- Probes per role: `{args.num_probes_per_role}`",
        f"- Chosen fixed G sign for Stage 6: `{chosen_sign}`",
        "",
        "## Gate summary",
        "",
        f"- proposed_qp_new differs from proposed_noG_new: `{bool(not np.isclose(main_qp_df['G_contribution_norm'].mean(), 0.0))}`",
        f"- gamma_active_frac mean > 0: `{bool(main_qp_df['gamma_active_frac'].mean() > 0.0)}`",
        f"- mean G_contribution_norm > 0: `{bool(main_qp_df['G_contribution_norm'].mean() > 1e-8)}`",
        f"- proposed_qp_new same-minibatch loss no worse than noG mean: `{bool(main_qp_df['same_minibatch_loss_change'].mean() <= nog_df['same_minibatch_loss_change'].mean() + 1e-8)}`",
        f"- zero_update mean < 1: `{bool(main_qp_df['zero_update_flag'].mean() < 1.0)}`",
        "",
        f"Stage 5 gate: `{'PASS' if pass_gate else 'FAIL'}`",
        "",
        "## Failure hints if gate fails",
        "",
        "- G sign wrong?",
        "- G too small?",
        "- beta/gamma grid too conservative?",
        "- update norm cap too tight?",
        "- critic block dominates?",
        "- loss objective mismatch?",
    ]
    (output_dir / "stage5_fixed_minibatch_audit_report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
