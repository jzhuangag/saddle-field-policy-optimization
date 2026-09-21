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

from models.optimizers import block_norm, classify_parameter_block, clone_named_state, flatten_named_tensors, get_optimizer_class, named_difference, tensor_norm
from scripts.full_policy_optimizer_probe import (
    build_manager,
    collect_probe_batches,
    compute_clip_ranges,
    cosine_similarity,
    named_parameters,
)


@dataclass(frozen=True)
class Candidate:
    label: str
    optimizer: str
    lr: float
    max_grad_norm: float
    vf_coef: float
    optimizer_kwargs: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 10A proposed_qp vs EGM gap audit")
    parser.add_argument("--stage5-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-probes", type=int, default=3)
    parser.add_argument("--scope", type=str, default="full_policy", choices=["full_policy", "actor_game"])
    return parser.parse_args()


def build_control_manager(args: argparse.Namespace):
    manager = build_manager(args)
    manager.adv_impact = "control"
    manager.optimizer_scope = args.scope
    return manager


def load_stage5_configs(stage5_root: pathlib.Path) -> Dict[str, Dict[str, object]]:
    return json.loads((stage5_root / "final_full_policy_optimizer_configs.json").read_text(encoding="utf-8"))


def make_eval_closure(algo, rollout_data, *, max_grad_norm: float, vf_coef: float, ent_coef: float):
    clip_range, clip_range_vf = compute_clip_ranges(algo)
    named_params = named_parameters(algo.policy)
    actions = rollout_data.actions.long().flatten() if algo.action_space.__class__.__name__ == "Discrete" else rollout_data.actions

    def eval_closure(*, theta_override=None, backward: bool = True):
        if theta_override is not None:
            for name, param in named_params:
                param.data.copy_(theta_override[name])
        algo.policy.optimizer.zero_grad()
        critic_optimizer = getattr(algo, "_actor_game_critic_optimizer", None)
        if critic_optimizer is not None:
            critic_optimizer.zero_grad()
        old_vf_coef, old_ent_coef, old_max_grad_norm = algo.vf_coef, algo.ent_coef, algo.max_grad_norm
        algo.vf_coef = vf_coef
        algo.ent_coef = ent_coef
        algo.max_grad_norm = max_grad_norm
        try:
            total_loss, policy_loss, _, value_loss, entropy_loss, clip_fraction, approx_kl = algo._build_shared_policy_loss(
                rollout_data,
                actions,
                clip_range,
                clip_range_vf,
            )
            if backward:
                total_loss.backward()
                if np.isfinite(max_grad_norm):
                    th.nn.utils.clip_grad_norm_(algo.policy.parameters(), max_grad_norm)
            grads = {
                name: (param.grad.detach().clone() if param.grad is not None else th.zeros_like(param.data))
                for name, param in named_params
            }
            return {
                "loss_tensor": total_loss.detach(),
                "total_loss": float(total_loss.item()),
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy_loss": float(entropy_loss.item()),
                "clip_fraction": float(clip_fraction),
                "approx_kl": float(approx_kl),
                "grads": grads,
            }
        finally:
            algo.vf_coef = old_vf_coef
            algo.ent_coef = old_ent_coef
            algo.max_grad_norm = old_max_grad_norm

    return eval_closure


def instantiate_optimizer(algo, candidate: Candidate):
    optimizer_class = get_optimizer_class(candidate.optimizer)
    if candidate.optimizer_kwargs.get("optimizer_scope") == "actor_game" and getattr(algo, "_actor_game_critic_optimizer", None) is None:
        algo._configure_actor_game_optimizers()
        named_params = algo._actor_named_parameters()
        params = [p for _, p in named_params]
    else:
        params = algo.policy.parameters()
    optimizer = optimizer_class(params, lr=candidate.lr, **candidate.optimizer_kwargs)
    algo.policy.optimizer = optimizer
    return optimizer


def run_single_step(algo, rollout_data, candidate: Candidate) -> Dict[str, object]:
    named_params = named_parameters(algo.policy)
    theta_old = clone_named_state(named_params)
    ent_coef = float(algo.ent_coef)
    optimizer = instantiate_optimizer(algo, candidate)
    eval_closure = make_eval_closure(
        algo,
        rollout_data,
        max_grad_norm=candidate.max_grad_norm,
        vf_coef=candidate.vf_coef,
        ent_coef=ent_coef,
    )
    old_eval = eval_closure(theta_override=theta_old, backward=True)

    if bool(getattr(optimizer, "requires_eval_closure", False)):
        optimizer.step(eval_closure=eval_closure, named_params=named_params)
    elif bool(getattr(optimizer, "requires_closure", False)):
        def closure():
            return eval_closure(theta_override=None, backward=True)["loss_tensor"]
        optimizer.step(closure)
    else:
        optimizer.zero_grad()
        eval_closure(theta_override=None, backward=True)
        optimizer.step()

    theta_after_actor = clone_named_state(named_params)
    critic_update_norm_separate = 0.0
    if candidate.optimizer_kwargs.get("optimizer_scope") == "actor_game":
        critic_update_norm_separate = float(algo._apply_actor_game_critic_step(eval_closure))
    theta_new = clone_named_state(named_params)
    new_eval = eval_closure(theta_override=theta_new, backward=False)
    diff = named_difference(theta_new, theta_old)
    selected_names = list(diff.keys())
    update_vec = flatten_named_tensors(diff, selected_names)
    metrics = dict(getattr(optimizer, "last_step_metrics", {}))
    metrics.update(
        {
            "label": candidate.label,
            "optimizer": candidate.optimizer,
            "lr": candidate.lr,
            "max_grad_norm": candidate.max_grad_norm,
            "vf_coef": candidate.vf_coef,
            "same_minibatch_total_loss_change": float(new_eval["total_loss"] - old_eval["total_loss"]),
            "policy_loss_change": float(new_eval["policy_loss"] - old_eval["policy_loss"]),
            "value_loss_change": float(new_eval["value_loss"] - old_eval["value_loss"]),
            "entropy_change": float(new_eval["entropy_loss"] - old_eval["entropy_loss"]),
            "actual_loss_decrease": float(old_eval["total_loss"] - new_eval["total_loss"]),
            "update_norm": tensor_norm(update_vec),
            "actor_update_norm": block_norm(diff, selected_names, "actor"),
            "logstd_update_norm": block_norm(diff, selected_names, "logstd"),
            "critic_update_norm": block_norm(diff, selected_names, "critic"),
            "critic_update_norm_separate_adam": critic_update_norm_separate,
            "update_map": diff,
            "F_map": {name: old_eval["grads"][name].detach().clone() for name in old_eval["grads"]},
        }
    )
    for name, param in named_params:
        param.data.copy_(theta_old[name])
    return metrics


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    plots_dir = output_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    configs = load_stage5_configs(pathlib.Path(args.stage5_root))
    egm_cfg = configs["egm"]
    no_g_cfg = configs["proposed_noG"] if "proposed_noG" in configs else egm_cfg

    candidates = [
        Candidate("egm", "egm", float(egm_cfg["lr"]), float(egm_cfg["max_grad_norm"]), float(egm_cfg["vf_coef"]), {}),
        Candidate("proposed_noG", "proposed_noG", float(no_g_cfg.get("lr", egm_cfg["lr"])), float(no_g_cfg.get("max_grad_norm", egm_cfg["max_grad_norm"])), float(no_g_cfg.get("vf_coef", egm_cfg["vf_coef"])), {"optimizer_scope": args.scope, "qp_normalization": "global", "qp_g_alpha": 0.3}),
        Candidate("proposed_qp_block", "proposed_qp", float(egm_cfg["lr"]), float(egm_cfg["max_grad_norm"]), float(egm_cfg["vf_coef"]), {"optimizer_scope": args.scope, "qp_normalization": "block", "qp_g_alpha": 0.3}),
        Candidate("proposed_qp_global", "proposed_qp", float(egm_cfg["lr"]), float(egm_cfg["max_grad_norm"]), float(egm_cfg["vf_coef"]), {"optimizer_scope": args.scope, "qp_normalization": "global", "qp_g_alpha": 0.3}),
        Candidate("proposed_qp_none", "proposed_qp", float(egm_cfg["lr"]), float(egm_cfg["max_grad_norm"]), float(egm_cfg["vf_coef"]), {"optimizer_scope": args.scope, "qp_normalization": "none", "qp_g_alpha": 0.3}),
    ]

    manager = build_control_manager(args)
    rarl_model = manager.setup_experiment()
    rows: List[Dict[str, object]] = []

    for role in ["protagonist", "adversary"]:
        algo = rarl_model.protagonist if role == "protagonist" else rarl_model.adversary
        probes = collect_probe_batches(rarl_model, role, args.num_probes)
        for probe in probes:
            probe_results: Dict[str, Dict[str, object]] = {}
            for candidate in candidates:
                result = run_single_step(algo, probe.rollout_data, candidate)
                result["role"] = role
                result["probe_idx"] = probe.probe_idx
                probe_results[candidate.label] = result

            egm_update = flatten_named_tensors(probe_results["egm"]["update_map"])
            noG_update = flatten_named_tensors(probe_results["proposed_noG"]["update_map"])
            for label, result in probe_results.items():
                update_vec = flatten_named_tensors(result["update_map"])
                row = {k: v for k, v in result.items() if k not in {"update_map", "F_map"}}
                row["scope"] = args.scope
                row["cosine_qp_update_egm_update"] = cosine_similarity(update_vec, egm_update)
                row["cosine_noG_update_egm_update"] = cosine_similarity(noG_update, egm_update)
                row["cosine_qp_update_noG_update"] = cosine_similarity(update_vec, noG_update)
                f_vec = flatten_named_tensors(result["F_map"])
                row["actor_F_norm"] = block_norm(result["F_map"], list(result["F_map"].keys()), "actor")
                row["logstd_F_norm"] = block_norm(result["F_map"], list(result["F_map"].keys()), "logstd")
                row["critic_F_norm"] = block_norm(result["F_map"], list(result["F_map"].keys()), "critic")
                rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = output_root / "stage10a_gap_audit.csv"
    df.to_csv(out_csv, index=False)

    gamma_problem = bool((df.get("gamma_active_frac", pd.Series([0.0])).fillna(0.0) == 0.0).all())
    zero_problem = bool((df.get("zero_update_flag", pd.Series([0.0])).fillna(0.0) > 0.5).all())
    tiny_g_problem = bool((df.get("G_contribution_norm", pd.Series([0.0])).fillna(0.0) < 1e-8).all())

    summary = (
        df.groupby("label")
        .agg(
            same_minibatch_total_loss_change_mean=("same_minibatch_total_loss_change", "mean"),
            actual_loss_decrease_mean=("actual_loss_decrease", "mean"),
            update_norm_mean=("update_norm", "mean"),
            actor_update_norm_mean=("actor_update_norm", "mean"),
            critic_update_norm_mean=("critic_update_norm", "mean"),
            cosine_qp_update_egm_update_mean=("cosine_qp_update_egm_update", "mean"),
            cosine_qp_update_noG_update_mean=("cosine_qp_update_noG_update", "mean"),
            gamma_active_frac_mean=("gamma_active_frac", "mean"),
            G_contribution_norm_mean=("G_contribution_norm", "mean"),
            beta_mean=("beta", "mean"),
            gamma_mean=("gamma", "mean"),
        )
        .reset_index()
    )

    lines = [
        f"# Stage 10A QP-vs-EGM gap audit ({args.scope})",
        "",
        f"- probes per role: `{args.num_probes}`",
        f"- stop condition gamma_active all zero: `{gamma_problem}`",
        f"- stop condition updates mostly zero: `{zero_problem}`",
        f"- stop condition G contribution always tiny: `{tiny_g_problem}`",
        "",
        "## Aggregate means",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- `{row['label']}`: loss_change_mean=`{row['same_minibatch_total_loss_change_mean']:.6f}`, "
            f"actual_loss_decrease_mean=`{row['actual_loss_decrease_mean']:.6f}`, "
            f"update_norm_mean=`{row['update_norm_mean']:.6f}`, "
            f"actor_update_norm_mean=`{row['actor_update_norm_mean']:.6f}`, "
            f"critic_update_norm_mean=`{row['critic_update_norm_mean']:.6f}`, "
            f"gamma_active_frac_mean=`{row.get('gamma_active_frac_mean', float('nan')):.6f}`, "
            f"G_contribution_norm_mean=`{row.get('G_contribution_norm_mean', float('nan')):.6f}`"
        )

    lines.extend(
        [
            "",
            "## Key questions",
            "",
        ]
    )
    qp_global = summary[summary["label"] == "proposed_qp_global"]
    noG = summary[summary["label"] == "proposed_noG"]
    egm = summary[summary["label"] == "egm"]
    if not qp_global.empty and not noG.empty and not egm.empty:
        qp_row = qp_global.iloc[0]
        nog_row = noG.iloc[0]
        egm_row = egm.iloc[0]
        lines.append(f"- Is QP update close to EGM? cosine mean(global vs egm)=`{qp_row['cosine_qp_update_egm_update_mean']:.6f}`")
        lines.append(f"- Is QP nearly identical to noG? cosine mean(global vs noG)=`{qp_row['cosine_qp_update_noG_update_mean']:.6f}`")
        lines.append(f"- Does QP reduce same-minibatch loss more than EGM? `{qp_row['same_minibatch_total_loss_change_mean'] < egm_row['same_minibatch_total_loss_change_mean']}`")
        lines.append(f"- Does QP differ mainly in critic block? compare actor `{qp_row['actor_update_norm_mean']:.6f}` vs critic `{qp_row['critic_update_norm_mean']:.6f}`")
        lines.append(f"- Is gamma active but tiny? `{qp_row['gamma_active_frac_mean'] > 0 and qp_row['G_contribution_norm_mean'] < 1e-4}`")
        lines.append(f"- Is noG already nearly the same as QP? `{abs(qp_row['same_minibatch_total_loss_change_mean'] - nog_row['same_minibatch_total_loss_change_mean']) < 1e-4}`")

    report_path = output_root / "stage10a_gap_audit_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_df = summary.copy()
    axes[0].bar(plot_df["label"], plot_df["actual_loss_decrease_mean"])
    axes[0].set_title("Actual same-minibatch loss decrease")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(plot_df["label"], plot_df["cosine_qp_update_egm_update_mean"].fillna(np.nan))
    axes[1].set_title("Cosine vs EGM update")
    axes[1].tick_params(axis="x", rotation=25)
    axes[2].bar(plot_df["label"], plot_df["G_contribution_norm_mean"].fillna(0.0))
    axes[2].set_title("G contribution norm")
    axes[2].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(plots_dir / "stage10a_qp_vs_egm_update_geometry.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if gamma_problem or zero_problem or tiny_g_problem:
        raise SystemExit("Stage 10A stop condition triggered; see report.")


if __name__ == "__main__":
    main()
