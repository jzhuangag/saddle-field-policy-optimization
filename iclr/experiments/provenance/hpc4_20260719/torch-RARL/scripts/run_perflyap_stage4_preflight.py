from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from models.proposed_qp_perflyap import ProposedNoGPerfLyapOptimizer, ProposedQPPerfLyapOptimizer
from scripts.full_policy_optimizer_probe import clone_state, collect_probe_batches, named_parameters, restore_state
from scripts.run_proposed_qp_new_v2_rawfg_audit import build_control_manager


RESULT_METHODS = {
    "nog": "proposed_noG_perfLyap",
    "qp": "proposed_qp_perfLyap",
}


@dataclass(frozen=True)
class ConfigSpec:
    config_id: int
    scope: str
    lambda_N: float
    lambda_P: float
    lambda_critic: float
    logstd_weight: float
    beta_max: float
    gamma_max: float
    update_cap: float
    fd_eps: float
    beta_probe: float
    gamma_probe: float
    ridge: float
    rho: float

    @property
    def label(self) -> str:
        return (
            f"{self.scope}"
            f"_n{self.lambda_N:g}"
            f"_p{self.lambda_P:g}"
            f"_lc{self.lambda_critic:g}"
            f"_ls{self.logstd_weight:g}"
            f"_b{self.beta_max:g}"
            f"_g{self.gamma_max:g}"
            f"_cap{self.update_cap:g}"
            f"_fd{self.fd_eps:g}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 4 preflight for performance-aligned RARL Lyapunov QP")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-probes-per-role", type=int, default=4)
    parser.add_argument("--max-configs", type=int, default=80)
    parser.add_argument("--eta-egm", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--vf-coef", type=float, default=1.0)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    return parser.parse_args()


def build_configs(max_configs: int) -> List[ConfigSpec]:
    configs: List[ConfigSpec] = []
    config_id = 0

    def add_scope(scope: str, lambda_ns, lambda_ps, beta_maxs, gamma_maxs, update_caps, lambda_critic=0.0, logstd_weight=0.0):
        nonlocal config_id
        for lambda_N, lambda_P, beta_max, gamma_max, update_cap in itertools.product(
            lambda_ns, lambda_ps, beta_maxs, gamma_maxs, update_caps
        ):
            if len(configs) >= max_configs:
                return
            configs.append(
                ConfigSpec(
                    config_id=config_id,
                    scope=scope,
                    lambda_N=lambda_N,
                    lambda_P=lambda_P,
                    lambda_critic=lambda_critic,
                    logstd_weight=logstd_weight,
                    beta_max=beta_max,
                    gamma_max=gamma_max,
                    update_cap=update_cap,
                    fd_eps=1e-3,
                    beta_probe=1e-3,
                    gamma_probe=1e-6,
                    ridge=1e-8,
                    rho=1e-8,
                )
            )
            config_id += 1

    add_scope(
        "actor_mean_only",
        lambda_ns=[0.03, 0.1, 0.3],
        lambda_ps=[1.0, 3.0, 10.0],
        beta_maxs=[1e-2, 3e-2],
        gamma_maxs=[3e-5, 1e-4],
        update_caps=[0.003, 0.005],
        lambda_critic=0.0,
        logstd_weight=0.0,
    )
    if len(configs) < max_configs:
        add_scope(
            "actor_game",
            lambda_ns=[0.03, 0.1, 0.3],
            lambda_ps=[1.0, 3.0, 10.0],
            beta_maxs=[1e-2, 3e-2],
            gamma_maxs=[3e-5, 1e-4],
            update_caps=[0.003, 0.005],
            lambda_critic=0.0,
            logstd_weight=0.0,
        )
    return configs


def role_algo(model, role: str):
    return model.protagonist if role == "protagonist" else model.adversary


def build_model(args: argparse.Namespace):
    manager = build_control_manager(args)
    model = manager.setup_experiment()
    return model


def build_eval_closure(algo, rollout_data):
    clip_range = algo.clip_range(algo._current_progress_remaining)
    clip_range_vf = None
    if algo.clip_range_vf is not None:
        clip_range_vf = algo.clip_range_vf(algo._current_progress_remaining)
    actions = rollout_data.actions.long().flatten() if algo.action_space.__class__.__name__ == "Discrete" else rollout_data.actions
    return algo._build_eval_closure(rollout_data, actions, clip_range, clip_range_vf)


def instantiate_optimizer(which: str, named_params, config: ConfigSpec, args: argparse.Namespace, diagnostics_csv_path: pathlib.Path, role: str):
    optimizer_cls = ProposedQPPerfLyapOptimizer if which == "qp" else ProposedNoGPerfLyapOptimizer
    kwargs = dict(
        lr=args.lr,
        perflyap_scope=config.scope,
        lambda_N=config.lambda_N,
        lambda_P=config.lambda_P,
        lambda_critic=config.lambda_critic,
        logstd_weight=config.logstd_weight,
        qp_fd_eps=config.fd_eps,
        qp_beta_probe=config.beta_probe,
        qp_gamma_probe=config.gamma_probe,
        qp_ridge=config.ridge,
        qp_rho=config.rho,
        qp_beta_max=config.beta_max,
        qp_gamma_max=config.gamma_max,
        qp_max_update_norm=config.update_cap,
        qp_eps=1e-8,
        eta_egm_reference=args.eta_egm,
        diagnostics_csv_path=str(diagnostics_csv_path),
        role=role,
    )
    return optimizer_cls([param for _, param in named_params], **kwargs)


def run_single_probe(
    *,
    algo,
    role: str,
    train_probe,
    val_probe,
    config: ConfigSpec,
    which: str,
    args: argparse.Namespace,
    diagnostics_csv_path: pathlib.Path,
) -> Dict[str, object]:
    named_params = named_parameters(algo.policy)
    theta_old = clone_state(named_params)
    optimizer = instantiate_optimizer(which, named_params, config, args, diagnostics_csv_path, role)
    train_eval = build_eval_closure(algo, train_probe.rollout_data)
    val_eval = build_eval_closure(algo, val_probe.rollout_data)
    loss_tensor = optimizer.step(eval_closure=train_eval, named_params=named_params)
    del loss_tensor
    metrics = dict(optimizer.last_step_metrics)
    theta_new = clone_state(named_params)

    restore_state(named_params, theta_old)
    val_before = val_eval(theta_override=theta_old, backward=True, grad_scope_names=None)
    val_after = val_eval(theta_override=theta_new, backward=True, grad_scope_names=None)
    restore_state(named_params, theta_old)

    metrics.update(
        {
            "method": RESULT_METHODS[which],
            "role": role,
            "config_id": config.config_id,
            "config_label": config.label,
            "train_probe_id": int(train_probe.probe_idx),
            "val_probe_id": int(val_probe.probe_idx),
            "val_total_loss_before": float(val_before["total_loss"]),
            "val_total_loss_after": float(val_after["total_loss"]),
            "val_loss_change": float(val_after["total_loss"] - val_before["total_loss"]),
            "val_policy_loss_before": float(val_before["policy_loss"]),
            "val_policy_loss_after": float(val_after["policy_loss"]),
            "val_policy_loss_change": float(val_after["policy_loss"] - val_before["policy_loss"]),
            "val_value_loss_before": float(val_before["value_loss"]),
            "val_value_loss_after": float(val_after["value_loss"]),
            "val_value_loss_change": float(val_after["value_loss"] - val_before["value_loss"]),
            "val_entropy_loss_before": float(val_before["entropy_loss"]),
            "val_entropy_loss_after": float(val_after["entropy_loss"]),
            "val_entropy_loss_change": float(val_after["entropy_loss"] - val_before["entropy_loss"]),
            "val_clip_fraction_after": float(val_after["clip_fraction"]),
            "val_approx_kl_after": float(val_after["approx_kl"]),
            "update_norm_post_cap": float(metrics.get("update_norm_post_cap", float("nan"))),
        }
    )
    return metrics


def pair_probes(probes: Sequence[object]) -> List[Tuple[object, object]]:
    out = []
    for idx in range(0, len(probes) - 1, 2):
        out.append((probes[idx], probes[idx + 1]))
    return out


def tolerance_for_c(no_g_c: float) -> float:
    return max(1e-4, 0.1 * abs(no_g_c))


def summarize_config(config: ConfigSpec, rows: pd.DataFrame) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "config_id": config.config_id,
        "config_label": config.label,
        "scope": config.scope,
        "lambda_N": config.lambda_N,
        "lambda_P": config.lambda_P,
        "lambda_critic": config.lambda_critic,
        "logstd_weight": config.logstd_weight,
        "beta_max": config.beta_max,
        "gamma_max": config.gamma_max,
        "update_cap": config.update_cap,
        "fd_eps": config.fd_eps,
    }
    for method in RESULT_METHODS.values():
        sub = rows[rows["method"] == method]
        prefix = "qp" if method.endswith("qp_perfLyap") else "nog"
        summary[f"{prefix}_rows"] = int(len(sub))
        if sub.empty:
            continue
        for col in [
            "actual_V_change",
            "actual_C_change",
            "approx_kl",
            "clip_fraction",
            "gamma_active_frac",
            "G_contribution_norm",
            "beta",
            "gamma",
            "update_norm_post_cap",
            "actor_fraction_of_update",
            "actor_fraction_of_V_decrease",
            "critic_fraction_of_V_decrease",
            "val_loss_change",
        ]:
            if col in sub.columns:
                summary[f"{prefix}_{col}_mean"] = float(pd.to_numeric(sub[col], errors="coerce").mean())
                summary[f"{prefix}_{col}_max"] = float(pd.to_numeric(sub[col], errors="coerce").max())
        core_cols = [
            c
            for c in [
                "actual_V_change",
                "actual_C_change",
                "approx_kl",
                "clip_fraction",
                "update_norm_post_cap",
            ]
            if c in sub.columns
        ]
        core_array = sub[core_cols].astype(float).to_numpy() if core_cols else np.zeros((0, 0), dtype=float)
        summary[f"{prefix}_core_finite"] = bool(np.isfinite(core_array).all())
    qp_v = float(summary.get("qp_actual_V_change_mean", float("inf")))
    nog_v = float(summary.get("nog_actual_V_change_mean", float("inf")))
    qp_c = float(summary.get("qp_actual_C_change_mean", float("inf")))
    nog_c = float(summary.get("nog_actual_C_change_mean", float("inf")))
    qp_kl = float(summary.get("qp_approx_kl_max", float("inf")))
    qp_clip = float(summary.get("qp_clip_fraction_max", float("inf")))
    qp_gamma_active = float(summary.get("qp_gamma_active_frac_mean", 0.0))
    qp_g_norm = float(summary.get("qp_G_contribution_norm_mean", 0.0))
    qp_actor_update = float(summary.get("qp_actor_fraction_of_update_mean", 0.0))
    qp_actor_v = float(summary.get("qp_actor_fraction_of_V_decrease_mean", 0.0))
    qp_critic_v = float(summary.get("qp_critic_fraction_of_V_decrease_mean", 0.0))
    qp_update = float(summary.get("qp_update_norm_post_cap_max", float("inf")))
    qp_core_finite = bool(summary.get("qp_core_finite", False))
    nog_core_finite = bool(summary.get("nog_core_finite", False))
    summary["pass_core"] = bool(qp_core_finite and nog_core_finite)
    summary["pass_qp_beats_nog_on_V"] = bool(qp_v < nog_v)
    summary["pass_qp_not_worse_on_C"] = bool(qp_c <= nog_c + tolerance_for_c(nog_c))
    summary["pass_kl_clip"] = bool(qp_kl <= 0.1 and qp_clip <= 0.8)
    summary["pass_gamma_active"] = bool(qp_gamma_active > 0.0 and qp_g_norm > 0.0)
    if config.scope in {"actor_mean_only", "actor_game", "actor_mean_plus_adv_actor_mean"}:
        summary["pass_actor_focus"] = bool(qp_actor_update >= 0.5 and qp_actor_v >= 0.5 and qp_critic_v <= 0.5)
    else:
        summary["pass_actor_focus"] = True
    summary["pass_update_health"] = bool(np.isfinite(qp_update))
    summary["pass_validation"] = bool(float(summary.get("qp_val_loss_change_mean", 0.0)) <= 0.05)
    summary["stage4_pass"] = bool(
        summary["pass_core"]
        and summary["pass_qp_beats_nog_on_V"]
        and summary["pass_qp_not_worse_on_C"]
        and summary["pass_kl_clip"]
        and summary["pass_gamma_active"]
        and summary["pass_actor_focus"]
        and summary["pass_update_health"]
        and summary["pass_validation"]
    )
    return summary


def plot_stage4(summary_df: pd.DataFrame, output_root: pathlib.Path) -> None:
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    top = summary_df.sort_values(["stage4_pass", "qp_actual_V_change_mean"], ascending=[False, True]).head(20).copy()
    x = np.arange(len(top))
    labels = [label[:24] for label in top["config_label"]]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - 0.18, top["nog_actual_V_change_mean"], width=0.36, label="noG V change")
    ax.bar(x + 0.18, top["qp_actual_V_change_mean"], width=0.36, label="QP V change")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right")
    ax.set_ylabel("Actual V change")
    ax.set_title("Stage 4 V change by config")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stage4_V_change_by_config.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - 0.18, top["nog_actual_C_change_mean"], width=0.36, label="noG C change")
    ax.bar(x + 0.18, top["qp_actual_C_change_mean"], width=0.36, label="QP C change")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right")
    ax.set_ylabel("Actual C change")
    ax.set_title("Stage 4 C change by config")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stage4_C_change_by_config.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, top["qp_beta_mean"], marker="o", label="QP beta")
    ax.plot(x, top["qp_gamma_mean"], marker="o", label="QP gamma")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right")
    ax.set_title("Stage 4 beta/gamma")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stage4_beta_gamma.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, top["qp_approx_kl_mean"], marker="o", label="QP approx_kl")
    ax.plot(x, top["qp_clip_fraction_mean"], marker="o", label="QP clip_fraction")
    ax.plot(x, top["qp_update_norm_post_cap_mean"], marker="o", label="QP update_norm")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right")
    ax.set_title("Stage 4 KL / clip / update health")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stage4_KL_clip_update.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(summary_df["nog_actual_V_change_mean"], summary_df["nog_actual_C_change_mean"], label="noG", alpha=0.7)
    ax.scatter(summary_df["qp_actual_V_change_mean"], summary_df["qp_actual_C_change_mean"], label="QP", alpha=0.7)
    ax.set_xlabel("Actual V change")
    ax.set_ylabel("Actual C change")
    ax.set_title("Stage 4 QP vs noG frontier")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stage4_QP_vs_noG_frontier.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, top["qp_actor_fraction_of_update_mean"], marker="o", label="actor_fraction_of_update")
    ax.plot(x, top["qp_actor_fraction_of_V_decrease_mean"], marker="o", label="actor_fraction_of_V_decrease")
    ax.plot(x, top["qp_critic_fraction_of_V_decrease_mean"], marker="o", label="critic_fraction_of_V_decrease")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right")
    ax.set_title("Stage 4 actor block contribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stage4_actor_block_contribution.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_root / "stage4_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    configs = build_configs(args.max_configs)
    model = build_model(args)
    try:
        role_pairs: Dict[str, List[Tuple[object, object]]] = {}
        for role in ["protagonist", "adversary"]:
            probes = collect_probe_batches(model, role, args.num_probes_per_role)
            role_pairs[role] = pair_probes(probes)

        all_rows: List[Dict[str, object]] = []
        for config in configs:
            for role, pairs in role_pairs.items():
                algo = role_algo(model, role)
                for train_probe, val_probe in pairs:
                    for which in ["nog", "qp"]:
                        diag_csv = diagnostics_dir / f"{config.label}_{role}_{which}.csv"
                        row = run_single_probe(
                            algo=algo,
                            role=role,
                            train_probe=train_probe,
                            val_probe=val_probe,
                            config=config,
                            which=which,
                            args=args,
                            diagnostics_csv_path=diag_csv,
                        )
                        all_rows.append(row)
        all_df = pd.DataFrame(all_rows)
        all_df.to_csv(output_root / "stage4_validation_generalization.csv", index=False)

        summary_rows = [summarize_config(config, all_df[all_df["config_id"] == config.config_id].copy()) for config in configs]
        summary_df = pd.DataFrame(summary_rows).sort_values(["stage4_pass", "qp_actual_V_change_mean"], ascending=[False, True])
        summary_df.to_csv(output_root / "stage4_preflight_summary.csv", index=False)

        top_pass = summary_df[summary_df["stage4_pass"]].head(6).copy()
        report_lines = [
            "# Stage 4 Preflight Report",
            "",
            f"- Evaluated configs: `{len(configs)}`",
            f"- Passing configs: `{int(summary_df['stage4_pass'].sum())}`",
            f"- Top-6 kept configs: `{len(top_pass)}`",
            "",
            "## Gate Summary",
            f"- pass_core: `{int(summary_df['pass_core'].sum())}` configs",
            f"- pass_qp_beats_nog_on_V: `{int(summary_df['pass_qp_beats_nog_on_V'].sum())}` configs",
            f"- pass_qp_not_worse_on_C: `{int(summary_df['pass_qp_not_worse_on_C'].sum())}` configs",
            f"- pass_kl_clip: `{int(summary_df['pass_kl_clip'].sum())}` configs",
            f"- pass_gamma_active: `{int(summary_df['pass_gamma_active'].sum())}` configs",
            f"- pass_actor_focus: `{int(summary_df['pass_actor_focus'].sum())}` configs",
            f"- pass_validation: `{int(summary_df['pass_validation'].sum())}` configs",
            "",
            "## Selected Configs",
        ]
        if top_pass.empty:
            report_lines.append("- No config passed Stage 4 gate.")
        else:
            for _, row in top_pass.iterrows():
                report_lines.extend(
                    [
                        f"### {row['config_label']}",
                        f"- scope: `{row['scope']}`",
                        f"- QP V change mean: `{row['qp_actual_V_change_mean']:.6e}` vs noG `{row['nog_actual_V_change_mean']:.6e}`",
                        f"- QP C change mean: `{row['qp_actual_C_change_mean']:.6e}` vs noG `{row['nog_actual_C_change_mean']:.6e}`",
                        f"- QP KL/clip mean: `{row['qp_approx_kl_mean']:.6e}` / `{row['qp_clip_fraction_mean']:.6e}`",
                        f"- gamma_active_frac_mean: `{row['qp_gamma_active_frac_mean']:.6f}`",
                        f"- actor_fraction_of_update_mean: `{row['qp_actor_fraction_of_update_mean']:.6f}`",
                        f"- actor_fraction_of_V_decrease_mean: `{row['qp_actor_fraction_of_V_decrease_mean']:.6f}`",
                        f"- critic_fraction_of_V_decrease_mean: `{row['qp_critic_fraction_of_V_decrease_mean']:.6f}`",
                    ]
                )
        (output_root / "stage4_preflight_report.md").write_text("\n".join(report_lines), encoding="utf-8")
        plot_stage4(summary_df, output_root)
    finally:
        del model


if __name__ == "__main__":
    main()
