from __future__ import annotations

import argparse
import math
import pathlib
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from models.proposed_qp_perflyap import ProposedNoGPerfLyapOptimizer, ProposedQPPerfLyapOptimizer
from scripts.full_policy_optimizer_probe import clone_state, collect_probe_batches, named_parameters, restore_state
from scripts.run_perflyap_stage4_preflight import ConfigSpec, build_configs, build_eval_closure, build_model, pair_probes, role_algo


RESULT_METHODS = {
    "nog": "proposed_noG_perfLyap",
    "qp": "proposed_qp_perfLyap",
}
G_SIGN_MODES = ["plus", "minus", "auto_predicted", "auto_actual_preflight"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 4.6 repaired preflight for performance-aligned RARL Lyapunov QP")
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
    parser.add_argument("--dense-fallback-points", type=int, default=101)
    return parser.parse_args()


def instantiate_optimizer(which: str, named_params, config: ConfigSpec, args: argparse.Namespace, diagnostics_csv_path: pathlib.Path, role: str, g_sign_mode: str):
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
        g_sign_mode=g_sign_mode,
        qp_dense_fallback_points=args.dense_fallback_points if which == "qp" else 0,
        diagnostics_csv_path=str(diagnostics_csv_path),
        role=role,
    )
    return optimizer_cls([param for _, param in named_params], **kwargs)


def tolerance_for_c(no_g_c: float) -> float:
    return max(1e-4, 0.1 * abs(no_g_c))


def tolerance_for_v(no_g_v: float) -> float:
    return max(1e-4, 0.05 * abs(no_g_v))


def run_single_probe(
    *,
    algo,
    role: str,
    train_probe,
    val_probe,
    config: ConfigSpec,
    which: str,
    g_sign_mode: str,
    args: argparse.Namespace,
    diagnostics_csv_path: pathlib.Path,
) -> Dict[str, object]:
    named_params = named_parameters(algo.policy)
    theta_old = clone_state(named_params)
    optimizer = instantiate_optimizer(which, named_params, config, args, diagnostics_csv_path, role, g_sign_mode)
    train_eval = build_eval_closure(algo, train_probe.rollout_data)
    val_eval = build_eval_closure(algo, val_probe.rollout_data)
    optimizer.step(eval_closure=train_eval, named_params=named_params)
    metrics = dict(optimizer.last_step_metrics)
    theta_new = clone_state(named_params)

    selected_names = optimizer._selected_names(named_params)
    norm_scale = float(metrics.get("norm_scale", 1.0))
    perf_scale = float(metrics.get("perf_scale", 1.0))

    restore_state(named_params, theta_old)
    val_before = optimizer._evaluate_state(
        eval_closure=val_eval,
        theta_state=theta_old,
        selected_names=selected_names,
        backward=True,
        norm_scale=norm_scale,
        perf_scale=perf_scale,
    )
    val_after = optimizer._evaluate_state(
        eval_closure=val_eval,
        theta_state=theta_new,
        selected_names=selected_names,
        backward=True,
        norm_scale=norm_scale,
        perf_scale=perf_scale,
    )
    restore_state(named_params, theta_old)

    metrics.update(
        {
            "method": RESULT_METHODS[which],
            "g_sign_mode": g_sign_mode,
            "role": role,
            "config_id": config.config_id,
            "config_label": config.label,
            "train_probe_id": int(train_probe.probe_idx),
            "val_probe_id": int(val_probe.probe_idx),
            "val_V_before": float(val_before["V_merit"]),
            "val_V_after": float(val_after["V_merit"]),
            "val_actual_V_change": float(val_after["V_merit"] - val_before["V_merit"]),
            "val_C_before": float(val_before["C"]),
            "val_C_after": float(val_after["C"]),
            "val_actual_C_change": float(val_after["C"] - val_before["C"]),
            "val_total_loss_before": float(val_before["total_loss"]),
            "val_total_loss_after": float(val_after["total_loss"]),
            "val_loss_change": float(val_after["total_loss"] - val_before["total_loss"]),
        }
    )
    return metrics


def summarize_config(config: ConfigSpec, g_sign_mode: str, rows: pd.DataFrame) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "config_id": config.config_id,
        "config_label": config.label,
        "scope": config.scope,
        "g_sign_mode": g_sign_mode,
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
        numeric_cols = [
            "actual_V_change",
            "actual_C_change",
            "approx_kl",
            "clip_fraction",
            "gamma_active_frac",
            "G_contribution_norm",
            "beta",
            "gamma",
            "beta_raw",
            "gamma_raw",
            "beta_eff",
            "gamma_eff",
            "q_pred",
            "q_pred_pre_cap",
            "q_pred_post_cap",
            "update_norm_pre_cap",
            "update_norm_post_cap",
            "actor_fraction_of_update",
            "actor_fraction_of_V_decrease",
            "critic_fraction_of_V_decrease",
            "val_actual_V_change",
            "val_actual_C_change",
            "no_g_reference_q_pred_post_cap",
        ]
        for col in numeric_cols:
            if col in sub.columns:
                summary[f"{prefix}_{col}_mean"] = float(sub[col].mean())
                summary[f"{prefix}_{col}_max"] = float(sub[col].max())
        summary[f"{prefix}_core_finite"] = bool(np.isfinite(sub[["actual_V_change", "actual_C_change", "approx_kl", "clip_fraction", "update_norm_post_cap"]].to_numpy(dtype=float)).all())
        if "fallback_reason" in sub.columns:
            summary[f"{prefix}_fallback_count"] = int((sub["fallback_reason"].fillna("") != "").sum())
        if "dense_fallback_used" in sub.columns:
            summary[f"{prefix}_dense_fallback_count"] = int(sub["dense_fallback_used"].fillna(0).astype(int).sum())
    qp_q = float(summary.get("qp_q_pred_post_cap_mean", float("inf")))
    nog_q_reference = float(summary.get("qp_no_g_reference_q_pred_post_cap_mean", float("inf")))
    qp_v = float(summary.get("qp_actual_V_change_mean", float("inf")))
    nog_v = float(summary.get("nog_actual_V_change_mean", float("inf")))
    qp_c = float(summary.get("qp_actual_C_change_mean", float("inf")))
    nog_c = float(summary.get("nog_actual_C_change_mean", float("inf")))
    qp_kl = float(summary.get("qp_approx_kl_max", float("inf")))
    qp_clip = float(summary.get("qp_clip_fraction_max", float("inf")))
    qp_gamma_active = float(summary.get("qp_gamma_active_frac_mean", 0.0))
    qp_g_norm = float(summary.get("qp_G_contribution_norm_mean", 0.0))
    qp_update = float(summary.get("qp_update_norm_post_cap_max", float("inf")))
    qp_actor_update = float(summary.get("qp_actor_fraction_of_update_mean", 0.0))
    qp_actor_v = float(summary.get("qp_actor_fraction_of_V_decrease_mean", 0.0))
    qp_critic_v = float(summary.get("qp_critic_fraction_of_V_decrease_mean", 1.0))
    qp_val_v = float(summary.get("qp_val_actual_V_change_mean", float("inf")))
    nog_val_v = float(summary.get("nog_val_actual_V_change_mean", float("inf")))
    qp_val_c = float(summary.get("qp_val_actual_C_change_mean", float("inf")))
    nog_val_c = float(summary.get("nog_val_actual_C_change_mean", float("inf")))

    summary["pass_core"] = bool(summary.get("qp_core_finite", False) and summary.get("nog_core_finite", False))
    summary["pass_predicted_invariant"] = bool(qp_q <= nog_q_reference + 1e-10)
    summary["pass_composite_actual"] = bool(qp_v <= nog_v + tolerance_for_v(nog_v))
    summary["pass_c_not_catastrophic"] = bool(qp_c <= nog_c + tolerance_for_c(nog_c))
    summary["pass_kl_clip"] = bool(qp_kl <= 0.1 and qp_clip <= 0.8)
    summary["pass_gamma_active"] = bool(qp_gamma_active > 0.0 and qp_g_norm > 0.0)
    if config.scope in {"actor_mean_only", "actor_game", "actor_mean_plus_adv_actor_mean"}:
        summary["pass_actor_focus"] = bool(qp_actor_update >= 0.5 and qp_actor_v >= 0.5 and qp_critic_v <= 0.5)
    else:
        summary["pass_actor_focus"] = True
    summary["pass_update_health"] = bool(np.isfinite(qp_update) and qp_update <= config.update_cap + 1e-8)
    summary["pass_validation"] = bool(
        qp_val_v <= nog_val_v + tolerance_for_v(nog_val_v)
        and qp_val_c <= nog_val_c + tolerance_for_c(nog_val_c)
    )
    summary["stage46_pass"] = bool(
        summary["pass_core"]
        and summary["pass_predicted_invariant"]
        and summary["pass_composite_actual"]
        and summary["pass_c_not_catastrophic"]
        and summary["pass_kl_clip"]
        and summary["pass_gamma_active"]
        and summary["pass_actor_focus"]
        and summary["pass_update_health"]
        and summary["pass_validation"]
    )
    return summary


def plot_stage46(summary_df: pd.DataFrame, output_root: pathlib.Path) -> None:
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    top = summary_df.sort_values(["stage46_pass", "qp_actual_V_change_mean"], ascending=[False, True]).head(24).copy()
    labels = [f"{row['scope']}|{row['g_sign_mode']}"[:28] for _, row in top.iterrows()]
    x = np.arange(len(top))

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - 0.18, top["nog_actual_V_change_mean"], width=0.36, label="noG actual merit")
    ax.bar(x + 0.18, top["qp_actual_V_change_mean"], width=0.36, label="QP actual merit")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right")
    ax.set_title("Stage 4.6: QP vs noG predicted/actual composite merit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stage46_qp_vs_nog_predicted_actual.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    grouped = summary_df.groupby("g_sign_mode")[["qp_actual_V_change_mean", "nog_actual_V_change_mean"]].mean()
    grouped.plot(kind="bar", ax=ax)
    ax.set_title("Stage 4.6: G sign mode comparison")
    ax.set_ylabel("Mean actual composite merit change")
    fig.tight_layout()
    fig.savefig(plots_dir / "stage46_g_sign_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(summary_df["qp_update_norm_pre_cap_mean"], summary_df["qp_q_pred_pre_cap_mean"], label="pre-cap", alpha=0.7)
    ax.scatter(summary_df["qp_update_norm_post_cap_mean"], summary_df["qp_q_pred_post_cap_mean"], label="post-cap", alpha=0.7)
    ax.set_title("Stage 4.6: Cap consistency")
    ax.set_xlabel("Update norm")
    ax.set_ylabel("Predicted composite merit drift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stage46_cap_consistency.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = np.where(summary_df["stage46_pass"], "tab:green", "tab:red")
    ax.scatter(summary_df["qp_actual_V_change_mean"], summary_df["qp_actual_C_change_mean"], c=colors, alpha=0.7)
    ax.set_title("Stage 4.6: Preflight frontier")
    ax.set_xlabel("QP actual composite merit change")
    ax.set_ylabel("QP actual C change")
    fig.tight_layout()
    fig.savefig(plots_dir / "stage46_preflight_frontier.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    top_actor = summary_df.sort_values(["stage46_pass", "qp_actor_fraction_of_V_decrease_mean"], ascending=[False, False]).head(24)
    x = np.arange(len(top_actor))
    ax.bar(x - 0.18, top_actor["qp_actor_fraction_of_update_mean"], width=0.36, label="actor update share")
    ax.bar(x + 0.18, top_actor["qp_actor_fraction_of_V_decrease_mean"], width=0.36, label="actor V-decrease share")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['scope']}|{row['g_sign_mode']}"[:28] for _, row in top_actor.iterrows()], rotation=75, ha="right")
    ax.set_title("Stage 4.6: Actor block contribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stage46_actor_block_contribution.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_root / "stage46_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    configs = build_configs(args.max_configs)
    model = build_model(args)
    rows: List[Dict[str, object]] = []
    try:
        for role in ("protagonist", "adversary"):
            algo = role_algo(model, role)
            probes = collect_probe_batches(model, role, max(args.num_probes_per_role, 4))
            probe_pairs = pair_probes(probes)
            for config in configs:
                for g_sign_mode in G_SIGN_MODES:
                    for train_probe, val_probe in probe_pairs:
                        for which in ("nog", "qp"):
                            diagnostics_csv = diagnostics_dir / f"{role}_{which}_{g_sign_mode}_{config.config_id}.csv"
                            rows.append(
                                run_single_probe(
                                    algo=algo,
                                    role=role,
                                    train_probe=train_probe,
                                    val_probe=val_probe,
                                    config=config,
                                    which=which,
                                    g_sign_mode=g_sign_mode,
                                    args=args,
                                    diagnostics_csv_path=diagnostics_csv,
                                )
                            )
    finally:
        if hasattr(model, "env") and model.env is not None:
            model.env.close()

    all_df = pd.DataFrame(rows)
    all_df.to_csv(output_root / "stage46_validation_generalization.csv", index=False)

    summary_rows = []
    for config in configs:
        for g_sign_mode in G_SIGN_MODES:
            sub = all_df[(all_df["config_id"] == config.config_id) & (all_df["g_sign_mode"] == g_sign_mode)].copy()
            summary_rows.append(summarize_config(config, g_sign_mode, sub))
    summary_df = pd.DataFrame(summary_rows).sort_values(["stage46_pass", "qp_actual_V_change_mean"], ascending=[False, True])
    summary_df.to_csv(output_root / "stage46_preflight_summary.csv", index=False)

    solver_summary = (
        summary_df.groupby("g_sign_mode")
        .agg(
            configs=("config_id", "count"),
            pass_predicted_invariant=("pass_predicted_invariant", "sum"),
            pass_composite_actual=("pass_composite_actual", "sum"),
            pass_configs=("stage46_pass", "sum"),
            qp_fallbacks=("qp_fallback_count", "sum"),
            qp_dense_fallbacks=("qp_dense_fallback_count", "sum"),
        )
        .reset_index()
    )
    solver_summary.to_csv(output_root / "stage46_solver_repair_summary.csv", index=False)
    solver_report_lines = [
        "# Stage 4.6 Solver Repair Report",
        "",
        f"- Evaluated sign modes: `{', '.join(G_SIGN_MODES)}`",
        f"- Total summary rows: `{len(summary_df)}`",
        "",
    ]
    for _, row in solver_summary.iterrows():
        solver_report_lines.extend(
            [
                f"## {row['g_sign_mode']}",
                f"- Config rows: `{int(row['configs'])}`",
                f"- Predicted invariant passes: `{int(row['pass_predicted_invariant'])}`",
                f"- Actual composite-merit passes: `{int(row['pass_composite_actual'])}`",
                f"- Stage 4.6 passes: `{int(row['pass_configs'])}`",
                f"- QP fallback-to-noG count: `{int(row['qp_fallbacks'])}`",
                f"- Dense fallback count: `{int(row['qp_dense_fallbacks'])}`",
                "",
            ]
        )
    (output_root / "stage46_solver_repair_report.md").write_text("\n".join(solver_report_lines), encoding="utf-8")

    plot_stage46(summary_df, output_root)

    top_pass = summary_df[summary_df["stage46_pass"]].head(6).copy()
    report_lines = [
        "# Stage 4.6 Preflight Report",
        "",
        f"- Evaluated configs: `{len(configs)}`",
        f"- Evaluated sign modes: `{', '.join(G_SIGN_MODES)}`",
        f"- Summary rows: `{len(summary_df)}`",
        f"- Passing configs: `{int(summary_df['stage46_pass'].sum())}`",
        "",
        "## Gate Summary",
        f"- predicted invariant failures: `{int((~summary_df['pass_predicted_invariant']).sum())}`",
        f"- composite-merit actual failures: `{int((~summary_df['pass_composite_actual']).sum())}`",
        f"- KL/clip failures: `{int((~summary_df['pass_kl_clip']).sum())}`",
        f"- gamma-active failures: `{int((~summary_df['pass_gamma_active']).sum())}`",
        f"- actor-focus failures: `{int((~summary_df['pass_actor_focus']).sum())}`",
        "",
        "## Solver Repair Summary",
    ]
    for _, row in solver_summary.iterrows():
        report_lines.append(
            f"- `{row['g_sign_mode']}`: pass=`{int(row['pass_configs'])}`, invariant_ok=`{int(row['pass_predicted_invariant'])}/{int(row['configs'])}`, dense_fallbacks=`{int(row['qp_dense_fallbacks'])}`"
        )
    report_lines.extend(["", "## Top Passing Configs"])
    if top_pass.empty:
        report_lines.append("- No config passed Stage 4.6 gate.")
    else:
        for _, row in top_pass.iterrows():
            report_lines.append(
                f"- `{row['config_label']}` / `{row['g_sign_mode']}`: qp_actual_V_change_mean=`{row['qp_actual_V_change_mean']:.6e}`, qp_actual_C_change_mean=`{row['qp_actual_C_change_mean']:.6e}`"
            )
    (output_root / "stage46_preflight_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    pass_count = int(summary_df["stage46_pass"].sum())
    best_mode_row = solver_summary.sort_values(["pass_configs", "pass_composite_actual", "pass_predicted_invariant"], ascending=[False, False, False]).iloc[0]
    root_lines = [
        "# Stage 4.6 Root Cause After Repair",
        "",
        f"1. Was noG feasible-subset invariant repaired? `{int((~summary_df['pass_predicted_invariant']).sum()) == 0}`",
        f"2. Did plusG or minusG win? Best aggregate mode: `{best_mode_row['g_sign_mode']}`",
        f"3. Did auto_predicted match auto_actual_preflight? Compare `stage46_solver_repair_summary.csv` pass counts.",
        f"4. Was cap consistency repaired? QP fallback count: `{int(summary_df['qp_fallback_count'].sum())}`; dense fallback count: `{int(summary_df['qp_dense_fallback_count'].sum())}`",
        f"5. How many configs now pass preflight? `{pass_count}`",
        "6. Which top configs should enter Stage 5 online?",
    ]
    if top_pass.empty:
        root_lines.append("- None yet.")
    else:
        for _, row in top_pass.iterrows():
            root_lines.append(f"- `{row['config_label']}` / `{row['g_sign_mode']}`")
    if pass_count == 0:
        root_lines.extend(
            [
                "7. If still zero configs pass, is the failure due to G direction, merit design, or actor scope?",
                "- Remaining failure is most likely a merit-design / actual-drift mismatch if minus/auto modes repair invariants but still fail actual composite merit.",
            ]
        )
    else:
        root_lines.extend(
            [
                "7. If still zero configs pass, is the failure due to G direction, merit design, or actor scope?",
                "- Not applicable: at least one config passed.",
            ]
        )
    (output_root / "stage46_root_cause_after_repair.md").write_text("\n".join(root_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
