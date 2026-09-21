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
    clone_state,
    collect_probe_batches,
    compute_clip_ranges,
    cosine_similarity,
    named_parameters,
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
    optimizer_kwargs: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 7D fixed-minibatch proposed-QP audit")
    parser.add_argument("--stage5-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-probes", type=int, default=3)
    parser.add_argument("--qp-g-alpha", type=float, default=1e-3)
    parser.add_argument("--qp-eps", type=float, default=1e-8)
    parser.add_argument("--max-update-norm", type=float, default=float("inf"))
    return parser.parse_args()


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


def run_single_step(algo, rollout_data, candidate: Candidate) -> Dict[str, object]:
    named_params = named_parameters(algo.policy)
    theta_old = clone_state(named_params)
    default_ent_coef = float(algo.ent_coef)
    optimizer_class = get_optimizer_class(candidate.optimizer)
    optimizer = optimizer_class(algo.policy.parameters(), lr=candidate.lr, **candidate.optimizer_kwargs)
    algo.policy.optimizer = optimizer
    eval_closure = make_eval_closure(
        algo,
        rollout_data,
        max_grad_norm=candidate.max_grad_norm,
        vf_coef=candidate.vf_coef,
        ent_coef=default_ent_coef,
    )
    old_eval = eval_closure(theta_override=theta_old, backward=True)

    if bool(getattr(optimizer, "requires_eval_closure", False)):
        step_loss = optimizer.step(eval_closure=eval_closure, named_params=named_params)
    elif bool(getattr(optimizer, "requires_closure", False)):
        def closure():
            closure_info = eval_closure(theta_override=None, backward=True)
            return closure_info["loss_tensor"]

        step_loss = optimizer.step(closure)
    else:
        optimizer.zero_grad()
        closure_info = eval_closure(theta_override=None, backward=True)
        optimizer.step()
        step_loss = closure_info["loss_tensor"]

    theta_new = clone_state(named_params)
    new_eval = eval_closure(theta_override=theta_new, backward=False)
    restore_theta = theta_old
    for name, param in named_params:
        param.data.copy_(restore_theta[name])

    update = vector_from_state_diff(theta_new, theta_old)
    update_vec = th.cat([tensor.reshape(-1) for tensor in update.values()]) if update else th.zeros(0)
    metrics = dict(getattr(optimizer, "last_step_metrics", {}))
    metrics.update(
        {
            "label": candidate.label,
            "optimizer": candidate.optimizer,
            "lr": candidate.lr,
            "max_grad_norm": candidate.max_grad_norm,
            "vf_coef": candidate.vf_coef,
            "optimizer_state_size": len(optimizer.state),
            "actual_same_minibatch_total_loss_change": float(new_eval["total_loss"] - old_eval["total_loss"]),
            "actual_policy_loss_change": float(new_eval["policy_loss"] - old_eval["policy_loss"]),
            "actual_value_loss_change": float(new_eval["value_loss"] - old_eval["value_loss"]),
            "actual_entropy_change": float(new_eval["entropy_loss"] - old_eval["entropy_loss"]),
            "update_norm": tensor_norm(update_vec),
            "step_loss": float(step_loss.item()) if step_loss is not None else float("nan"),
            "update_map": update,
        }
    )
    return metrics


def health_score(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in df.groupby("label"):
        gamma_active_mean = float(group.get("gamma_active_frac", pd.Series([0.0] * len(group))).mean())
        zero_update_mean = float(group.get("zero_update_flag", pd.Series([0.0] * len(group))).mean())
        loss_change_mean = float(group["actual_same_minibatch_total_loss_change"].mean())
        nonfinite = int((~np.isfinite(group["update_norm"])).any())
        score = (
            5.0 * gamma_active_mean
            - 3.0 * max(loss_change_mean, 0.0)
            - 2.0 * zero_update_mean
            - 1000.0 * nonfinite
        )
        rows.append(
            {
                "label": label,
                "health_score": score,
                "gamma_active_mean": gamma_active_mean,
                "zero_update_mean": zero_update_mean,
                "loss_change_mean": loss_change_mean,
                "nonfinite_flag": nonfinite,
            }
        )
    return pd.DataFrame(rows).sort_values("health_score", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    plots_dir = output_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    stage5_configs = load_stage5_configs(pathlib.Path(args.stage5_root))
    egm_cfg = stage5_configs["egm"]
    sgd_cfg = stage5_configs["sgd"]
    ppm_cfg = stage5_configs["ppm"]

    proposed_qp_screen = [
        Candidate(
            label=f"proposed_qp_{normalization}",
            optimizer="proposed_qp",
            lr=float(egm_cfg["lr"]),
            max_grad_norm=float(egm_cfg["max_grad_norm"]),
            vf_coef=float(egm_cfg["vf_coef"]),
            optimizer_kwargs={
                "optimizer_scope": "full_policy",
                "qp_normalization": normalization,
                "qp_g_alpha": args.qp_g_alpha,
                "max_update_norm": args.max_update_norm,
                "qp_eps": args.qp_eps,
            },
        )
        for normalization in ["none", "global", "block"]
    ]

    manager = build_manager(args)
    rarl_model = manager.setup_experiment()
    all_rows: List[Dict[str, object]] = []
    screen_rows: List[Dict[str, object]] = []

    for role in ["protagonist", "adversary"]:
        algo = rarl_model.protagonist if role == "protagonist" else rarl_model.adversary
        probes = collect_probe_batches(rarl_model, role, args.num_probes)
        for probe in probes:
            updates_by_label: Dict[str, th.Tensor] = {}
            row_cache: Dict[str, Dict[str, object]] = {}

            for candidate in proposed_qp_screen:
                result = run_single_step(algo, probe.rollout_data, candidate)
                result["role"] = role
                result["probe_idx"] = probe.probe_idx
                result["qp_normalization"] = candidate.optimizer_kwargs["qp_normalization"]
                row_cache[candidate.label] = result
                updates_by_label[candidate.label] = th.cat([tensor.reshape(-1) for tensor in result["update_map"].values()]) if result["update_map"] else th.zeros(0)
                screen_rows.append({key: value for key, value in result.items() if key != "update_map"})

    screen_df = pd.DataFrame(screen_rows)
    score_df = health_score(screen_df)
    healthiest_label = str(score_df.iloc[0]["label"])
    healthiest_normalization = healthiest_label.replace("proposed_qp_", "")
    (output_root / "stage7d_selected_qp_normalization.json").write_text(
        json.dumps({"selected_label": healthiest_label, "qp_normalization": healthiest_normalization}, indent=2),
        encoding="utf-8",
    )

    final_candidates = [
        Candidate(
            label=healthiest_label,
            optimizer="proposed_qp",
            lr=float(egm_cfg["lr"]),
            max_grad_norm=float(egm_cfg["max_grad_norm"]),
            vf_coef=float(egm_cfg["vf_coef"]),
            optimizer_kwargs={
                "optimizer_scope": "full_policy",
                "qp_normalization": healthiest_normalization,
                "qp_g_alpha": args.qp_g_alpha,
                "max_update_norm": args.max_update_norm,
                "qp_eps": args.qp_eps,
            },
        ),
        Candidate(
            label="proposed_noG",
            optimizer="proposed_noG",
            lr=float(egm_cfg["lr"]),
            max_grad_norm=float(egm_cfg["max_grad_norm"]),
            vf_coef=float(egm_cfg["vf_coef"]),
            optimizer_kwargs={
                "optimizer_scope": "full_policy",
                "qp_normalization": healthiest_normalization,
                "qp_g_alpha": args.qp_g_alpha,
                "max_update_norm": args.max_update_norm,
                "qp_eps": args.qp_eps,
            },
        ),
        Candidate(
            label="sgd",
            optimizer="sgd",
            lr=float(sgd_cfg["lr"]),
            max_grad_norm=float(sgd_cfg["max_grad_norm"]),
            vf_coef=float(sgd_cfg["vf_coef"]),
            optimizer_kwargs=dict(sgd_cfg.get("optimizer_kwargs", {})),
        ),
        Candidate(
            label="egm",
            optimizer="egm",
            lr=float(egm_cfg["lr"]),
            max_grad_norm=float(egm_cfg["max_grad_norm"]),
            vf_coef=float(egm_cfg["vf_coef"]),
            optimizer_kwargs=dict(egm_cfg.get("optimizer_kwargs", {})),
        ),
        Candidate(
            label="ppm",
            optimizer="ppm",
            lr=float(ppm_cfg["lr"]),
            max_grad_norm=float(ppm_cfg["max_grad_norm"]),
            vf_coef=float(ppm_cfg["vf_coef"]),
            optimizer_kwargs=dict(ppm_cfg.get("optimizer_kwargs", {})),
        ),
    ]

    for role in ["protagonist", "adversary"]:
        algo = rarl_model.protagonist if role == "protagonist" else rarl_model.adversary
        probes = collect_probe_batches(rarl_model, role, args.num_probes)
        for probe in probes:
            updates_by_label: Dict[str, th.Tensor] = {}
            row_cache: Dict[str, Dict[str, object]] = {}
            for candidate in final_candidates:
                result = run_single_step(algo, probe.rollout_data, candidate)
                result["role"] = role
                result["probe_idx"] = probe.probe_idx
                result["qp_normalization"] = candidate.optimizer_kwargs.get("qp_normalization", "")
                row_cache[candidate.label] = result
                updates_by_label[candidate.label] = th.cat([tensor.reshape(-1) for tensor in result["update_map"].values()]) if result["update_map"] else th.zeros(0)

            sgd_vec = updates_by_label["sgd"]
            egm_vec = updates_by_label["egm"]
            ppm_vec = updates_by_label["ppm"]
            qp_vec = updates_by_label[healthiest_label]
            for label, result in row_cache.items():
                result["cosine_proposed_qp_vs_sgd"] = cosine_similarity(qp_vec, sgd_vec)
                result["cosine_proposed_qp_vs_egm"] = cosine_similarity(qp_vec, egm_vec)
                result["cosine_proposed_qp_vs_ppm"] = cosine_similarity(qp_vec, ppm_vec)
                result["gamma_active_frac"] = float(result.get("gamma_active", 0))
                result["zero_update_frac"] = float(result.get("zero_update_flag", 0))
                result["boundary_frac"] = float(result.get("boundary_solution_flag", 0))
                all_rows.append({key: value for key, value in result.items() if key != "update_map"})

    audit_df = pd.DataFrame(all_rows)
    audit_df.to_csv(output_root / "stage7d_qp_audit.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    grouped = audit_df.groupby("label").mean(numeric_only=True).reset_index()
    axes[0].bar(grouped["label"], grouped["actual_same_minibatch_total_loss_change"])
    axes[0].set_title("Same-minibatch Total Loss Change")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(grouped["label"], grouped["update_norm"])
    axes[1].set_title("Update Norm")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(alpha=0.3, axis="y")
    axes[2].bar(grouped["label"], grouped.get("gamma_active_frac", pd.Series(np.zeros(len(grouped)))))
    axes[2].set_title("Gamma Active Fraction")
    axes[2].tick_params(axis="x", rotation=35)
    axes[2].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plots_dir / "proposed_qp_fixed_minibatch_audit.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    qp_rows = audit_df[audit_df["label"] == healthiest_label]
    gate_pass = (
        np.isfinite(qp_rows["update_norm"]).all()
        and (qp_rows["update_norm"] > 0).any()
        and float(qp_rows["gamma_active_frac"].mean()) > 0.0
        and float(qp_rows["actual_same_minibatch_total_loss_change"].mean()) <= 0.0
        and int((qp_rows["optimizer_state_size"] > 0).any()) == 0
    )

    report_lines = [
        "# Stage 7D Proposed-QP Fixed-Minibatch Audit",
        "",
        f"- Selected qp_normalization: `{healthiest_normalization}`",
        f"- Stage5 baseline root: `{args.stage5_root}`",
        f"- Gate pass: `{gate_pass}`",
        "",
        "## Proposed-QP normalization health screen",
        "",
        "```json",
        score_df.to_json(orient="records", indent=2),
        "```",
        "",
        "## Final comparison means",
        "",
    ]
    for _, row in grouped.sort_values("label").iterrows():
        report_lines.extend(
            [
                f"- `{row['label']}`",
                f"  - actual_same_minibatch_total_loss_change: `{row['actual_same_minibatch_total_loss_change']:.6e}`",
                f"  - update_norm: `{row['update_norm']:.6e}`",
                f"  - gamma_active_frac: `{row.get('gamma_active_frac', 0.0):.6f}`",
                f"  - zero_update_frac: `{row.get('zero_update_frac', 0.0):.6f}`",
                f"  - boundary_frac: `{row.get('boundary_frac', 0.0):.6f}`",
            ]
        )
    report_lines.extend(
        [
            "",
            "## G convention",
            "",
            "- `F = grad(full PPO loss)`",
            "- `update = - beta * F + gamma * G` after optional normalization of the raw directions",
            "- `G = (F(z) - F(z - alpha * F(z))) / alpha`",
            "",
            "## Notes",
            "",
            "- `control-proxy` is not used in this fixed-minibatch audit; this stage only checks optimizer health on the force-trained PPO-RARL loss closure.",
            "- No accept/reject rollback, forced gamma activation, or hidden cap is used.",
        ]
    )
    (output_root / "stage7d_qp_audit_report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
