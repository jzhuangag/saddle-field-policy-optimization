from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from models.proposed_qp_new import apply_state_delta, flatten_named_tensors, named_difference, tensor_norm
from models.proposed_qp_perflyap import block_norm
from scripts.full_policy_optimizer_probe import clone_state, collect_probe_batches, named_parameters, restore_state
from scripts.run_perflyap_stage48_perf_audit import (
    DIR_MODES,
    Stage48Config,
    build_configs,
    build_eval_context,
    build_extended_eval_closure,
    build_model,
    evaluate_candidate,
    instantiate_helper,
    pair_probes,
    role_algo,
    solve_direction_mode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 4.9 extrapolation-vs-linear audit for perfLyap")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--stage48-summary", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-probes-per-role", type=int, default=4)
    parser.add_argument("--num-configs", type=int, default=12)
    parser.add_argument("--eta-egm", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--vf-coef", type=float, default=1.0)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    return parser.parse_args()


def tolerance_for_c(no_g_c: float) -> float:
    return max(1e-4, 0.1 * abs(no_g_c))


def tolerance_for_v(no_g_v: float) -> float:
    return max(1e-4, 0.05 * abs(no_g_v))


def select_top_configs(stage48_summary_path: pathlib.Path, all_configs: List[Stage48Config], num_configs: int) -> List[Stage48Config]:
    df = pd.read_csv(stage48_summary_path)
    config_df = (
        df[df["direction_mode"].isin(["egm_plus", "egm_minus"])][
            [
                "config_id",
                "config_label",
                "scope",
                "cost_mode",
                "lambda_N",
                "beta_max",
                "gamma_max",
                "update_cap",
                "qp_actual_V_change_mean",
                "qp_actual_C_change_mean",
                "qp_gamma_active_frac_mean",
            ]
        ]
        .sort_values(
            ["qp_actual_V_change_mean", "qp_actual_C_change_mean", "qp_gamma_active_frac_mean"],
            ascending=[True, True, False],
        )
        .drop_duplicates("config_id")
        .head(num_configs)
    )
    config_map = {cfg.config_id: cfg for cfg in all_configs}
    return [config_map[int(config_id)] for config_id in config_df["config_id"].tolist() if int(config_id) in config_map]


def apply_cap(update_map: Dict[str, torch.Tensor], selected_names: Sequence[str], cap: float, eps: float):
    update_vec = flatten_named_tensors(update_map, selected_names)
    norm_pre = tensor_norm(update_vec)
    cap_scale = 1.0
    cap_active = 0
    if np.isfinite(cap) and cap > 0.0 and norm_pre > cap:
        cap_scale = cap / max(norm_pre, eps)
        cap_active = 1
    update_map_capped = {name: tensor * cap_scale for name, tensor in update_map.items()}
    norm_post = tensor_norm(flatten_named_tensors(update_map_capped, selected_names))
    return update_map_capped, float(norm_pre), float(norm_post), float(cap_scale), int(cap_active)


def evaluate_update_map(
    *,
    helper,
    named_params,
    theta_old,
    selected_names,
    train_eval,
    val_eval,
    base_eval,
    norm_scale: float,
    perf_scale: float,
    update_map: Dict[str, torch.Tensor],
    config: Stage48Config,
    role: str,
    train_probe_id: int,
    val_probe_id: int,
    candidate_name: str,
    source_direction: str,
    beta_raw: float,
    gamma_raw: float,
    beta_eff: float,
    gamma_eff: float,
    alpha_shift: float,
    shift_sign: str,
    extra: Dict[str, object] | None = None,
) -> Dict[str, object]:
    theta_new = apply_state_delta(theta_old, update_map)
    new_eval = helper._evaluate_state(
        eval_closure=train_eval,
        theta_state=theta_new,
        selected_names=selected_names,
        backward=True,
        norm_scale=norm_scale,
        perf_scale=perf_scale,
    )
    diff = named_difference(theta_new, theta_old, selected_names)
    actor_update_norm = block_norm(diff, selected_names, "actor")
    logstd_update_norm = block_norm(diff, selected_names, "logstd")
    critic_update_norm = block_norm(diff, selected_names, "critic")
    total_update_norm = max(tensor_norm(flatten_named_tensors(update_map, selected_names)), helper.qp_eps)

    val_before = helper._evaluate_state(
        eval_closure=val_eval,
        theta_state=theta_old,
        selected_names=selected_names,
        backward=True,
        norm_scale=norm_scale,
        perf_scale=perf_scale,
    )
    val_after = helper._evaluate_state(
        eval_closure=val_eval,
        theta_state=theta_new,
        selected_names=selected_names,
        backward=True,
        norm_scale=norm_scale,
        perf_scale=perf_scale,
    )
    restore_state(named_params, theta_old)

    row = {
        "config_id": config.config_id,
        "config_label": config.label,
        "scope": config.scope,
        "cost_mode": config.cost_mode,
        "lambda_N": config.lambda_N,
        "lambda_P": config.lambda_P,
        "beta_max": config.beta_max,
        "gamma_max": config.gamma_max,
        "update_cap": config.update_cap,
        "role": role,
        "train_probe_id": train_probe_id,
        "val_probe_id": val_probe_id,
        "candidate_name": candidate_name,
        "source_direction": source_direction,
        "beta_raw": float(beta_raw),
        "gamma_raw": float(gamma_raw),
        "beta_eff": float(beta_eff),
        "gamma_eff": float(gamma_eff),
        "alpha_shift": float(alpha_shift),
        "shift_sign": shift_sign,
        "V_before": float(base_eval["V_merit"]),
        "V_after": float(new_eval["V_merit"]),
        "actual_V_change": float(new_eval["V_merit"] - base_eval["V_merit"]),
        "C_before": float(base_eval["C"]),
        "C_after": float(new_eval["C"]),
        "actual_C_change": float(new_eval["C"] - base_eval["C"]),
        "norm_term_before": float(base_eval["norm_term"]),
        "norm_term_after": float(new_eval["norm_term"]),
        "perf_term_before": float(base_eval["perf_term"]),
        "perf_term_after": float(new_eval["perf_term"]),
        "approx_kl": float(new_eval["approx_kl"]),
        "clip_fraction": float(new_eval["clip_fraction"]),
        "update_norm_post_cap": total_update_norm,
        "actor_update_norm": actor_update_norm,
        "logstd_update_norm": logstd_update_norm,
        "critic_update_norm": critic_update_norm,
        "actor_fraction_of_update": actor_update_norm / total_update_norm,
        "logstd_fraction_of_update": logstd_update_norm / total_update_norm,
        "critic_fraction_of_update": critic_update_norm / total_update_norm,
        "val_actual_V_change": float(val_after["V_merit"] - val_before["V_merit"]),
        "val_actual_C_change": float(val_after["C"] - val_before["C"]),
    }
    if extra:
        row.update(extra)
    return row


def extrapolated_update(
    *,
    helper,
    named_params,
    theta_old,
    selected_names,
    train_eval,
    base_eval,
    norm_scale: float,
    perf_scale: float,
    beta_raw: float,
    gamma_raw: float,
    shift_sign: str,
):
    if beta_raw <= helper.qp_eps:
        zero_map = {name: torch.zeros_like(base_eval["grads_selected"][name]) for name in selected_names}
        return zero_map, 0.0, 0.0, 0.0, 0.0, "degenerate_beta"

    alpha = gamma_raw / max(beta_raw, helper.qp_eps)
    sign = -1.0 if shift_sign == "minus_inside" else 1.0
    theta_shift = {name: tensor.clone() for name, tensor in theta_old.items()}
    for name in selected_names:
        theta_shift[name] = theta_old[name] + sign * alpha * base_eval["grads_selected"][name]
    shifted_eval = helper._evaluate_state(
        eval_closure=train_eval,
        theta_state=theta_shift,
        selected_names=selected_names,
        backward=True,
        norm_scale=norm_scale,
        perf_scale=perf_scale,
    )
    f_shift = {name: shifted_eval["grads_selected"][name].detach().clone() for name in selected_names}
    update_map_pre = {name: -beta_raw * f_shift[name] for name in selected_names}
    update_map, norm_pre, norm_post, cap_scale, cap_active = apply_cap(update_map_pre, selected_names, helper.qp_max_update_norm, helper.qp_eps)
    beta_eff = beta_raw * cap_scale
    gamma_eff = gamma_raw * cap_scale
    return update_map, norm_pre, norm_post, beta_eff, gamma_eff, cap_active


def run_probe_for_config(*, algo, role: str, config: Stage48Config, train_probe, val_probe, args: argparse.Namespace, diagnostics_csv_path: pathlib.Path) -> List[Dict[str, object]]:
    named_params = named_parameters(algo.policy)
    helper = instantiate_helper(named_params, config, args, diagnostics_csv_path, role, disable_g=False)
    train_ctx = build_eval_context(algo, train_probe.rollout_data)
    val_ctx = build_eval_context(algo, val_probe.rollout_data)
    train_eval = build_extended_eval_closure(train_ctx, config.cost_mode)
    val_eval = build_extended_eval_closure(val_ctx, config.cost_mode)

    theta_old = clone_state(named_params)
    selected_names = helper._selected_names(named_params)
    base_eval_unscaled = helper._evaluate_state(eval_closure=train_eval, theta_state=theta_old, selected_names=selected_names, backward=True)
    norm_scale, perf_scale = helper._update_scales(float(base_eval_unscaled["norm_term"]), float(base_eval_unscaled["perf_term"]))
    base_eval = helper._evaluate_state(
        eval_closure=train_eval,
        theta_state=theta_old,
        selected_names=selected_names,
        backward=True,
        norm_scale=norm_scale,
        perf_scale=perf_scale,
    )
    f_raw_map = {name: base_eval["grads_selected"][name].detach().clone() for name in selected_names}
    g_plus_map, _ = helper._compute_g_raw(
        theta_old=theta_old,
        eval_closure=train_eval,
        selected_names=selected_names,
        f_raw_map=f_raw_map,
        norm_scale=norm_scale,
        perf_scale=perf_scale,
    )
    g_minus_map = {name: -g_plus_map[name] for name in selected_names}

    _, no_g_solution, plus_solution, no_g_metrics, plus_metrics = solve_direction_mode(
        helper,
        theta_old,
        train_eval,
        named_params,
        selected_names,
        base_eval,
        norm_scale,
        perf_scale,
        f_raw_map,
        "egm_plus",
        g_plus_map,
        args.eta_egm,
    )
    _, _, minus_solution, _, minus_metrics = solve_direction_mode(
        helper,
        theta_old,
        train_eval,
        named_params,
        selected_names,
        base_eval,
        norm_scale,
        perf_scale,
        f_raw_map,
        "egm_minus",
        g_minus_map,
        args.eta_egm,
    )

    rows: List[Dict[str, object]] = []

    rows.append(
        evaluate_update_map(
            helper=helper,
            named_params=named_params,
            theta_old=theta_old,
            selected_names=selected_names,
            train_eval=train_eval,
            val_eval=val_eval,
            base_eval=base_eval,
            norm_scale=norm_scale,
            perf_scale=perf_scale,
            update_map=no_g_solution["update_map"],
            config=config,
            role=role,
            train_probe_id=int(train_probe.probe_idx),
            val_probe_id=int(val_probe.probe_idx),
            candidate_name="noG",
            source_direction="noG",
            beta_raw=float(no_g_solution["beta_raw"]),
            gamma_raw=0.0,
            beta_eff=float(no_g_solution["beta_eff"]),
            gamma_eff=0.0,
            alpha_shift=0.0,
            shift_sign="none",
            extra={
                "update_norm_pre_cap": float(no_g_solution["update_norm_pre_cap"]),
                "cap_scale": float(no_g_solution["cap_scale"]),
                "cap_active": int(no_g_solution["cap_active"]),
                "q_pred_post_cap": float(no_g_solution["q_pred_post_cap"]),
                "solver_case": str(no_g_solution["selected_case"]),
            },
        )
    )

    for candidate_name, solution, metrics in [
        ("linear_plus", plus_solution, plus_metrics),
        ("linear_minus", minus_solution, minus_metrics),
    ]:
        rows.append(
            evaluate_update_map(
                helper=helper,
                named_params=named_params,
                theta_old=theta_old,
                selected_names=selected_names,
                train_eval=train_eval,
                val_eval=val_eval,
                base_eval=base_eval,
                norm_scale=norm_scale,
                perf_scale=perf_scale,
                update_map=solution["update_map"],
                config=config,
                role=role,
                train_probe_id=int(train_probe.probe_idx),
                val_probe_id=int(val_probe.probe_idx),
                candidate_name=candidate_name,
                source_direction=candidate_name,
                beta_raw=float(solution["beta_raw"]),
                gamma_raw=float(solution["gamma_raw"]),
                beta_eff=float(solution["beta_eff"]),
                gamma_eff=float(solution["gamma_eff"]),
                alpha_shift=float(solution["gamma_raw"]) / max(float(solution["beta_raw"]), helper.qp_eps) if float(solution["beta_raw"]) > helper.qp_eps else 0.0,
                shift_sign="none",
                extra={
                    "update_norm_pre_cap": float(solution["update_norm_pre_cap"]),
                    "cap_scale": float(solution["cap_scale"]),
                    "cap_active": int(solution["cap_active"]),
                    "q_pred_post_cap": float(solution["q_pred_post_cap"]),
                    "solver_case": str(solution["selected_case"]),
                    "predicted_invariant_pass": bool(metrics["pass_predicted_invariant"]),
                },
            )
        )

    extrap_specs = [
        ("extrap_forward_from_plus", plus_solution, "minus_inside"),
        ("extrap_backward_from_minus", minus_solution, "plus_inside"),
    ]
    for candidate_name, source_solution, shift_sign in extrap_specs:
        update_map, norm_pre, norm_post, beta_eff, gamma_eff, cap_active = extrapolated_update(
            helper=helper,
            named_params=named_params,
            theta_old=theta_old,
            selected_names=selected_names,
            train_eval=train_eval,
            base_eval=base_eval,
            norm_scale=norm_scale,
            perf_scale=perf_scale,
            beta_raw=float(source_solution["beta_raw"]),
            gamma_raw=float(source_solution["gamma_raw"]),
            shift_sign=shift_sign,
        )
        rows.append(
            evaluate_update_map(
                helper=helper,
                named_params=named_params,
                theta_old=theta_old,
                selected_names=selected_names,
                train_eval=train_eval,
                val_eval=val_eval,
                base_eval=base_eval,
                norm_scale=norm_scale,
                perf_scale=perf_scale,
                update_map=update_map,
                config=config,
                role=role,
                train_probe_id=int(train_probe.probe_idx),
                val_probe_id=int(val_probe.probe_idx),
                candidate_name=candidate_name,
                source_direction="linear_plus" if "plus" in candidate_name else "linear_minus",
                beta_raw=float(source_solution["beta_raw"]),
                gamma_raw=float(source_solution["gamma_raw"]),
                beta_eff=float(beta_eff),
                gamma_eff=float(gamma_eff),
                alpha_shift=float(source_solution["gamma_raw"]) / max(float(source_solution["beta_raw"]), helper.qp_eps) if float(source_solution["beta_raw"]) > helper.qp_eps else 0.0,
                shift_sign=shift_sign,
                extra={
                    "update_norm_pre_cap": float(norm_pre),
                    "cap_scale": float(beta_eff / max(float(source_solution["beta_raw"]), helper.qp_eps)) if float(source_solution["beta_raw"]) > helper.qp_eps else 0.0,
                    "cap_active": int(cap_active),
                    "q_pred_post_cap": float("nan"),
                    "solver_case": "extrapolated_field",
                },
            )
        )

    restore_state(named_params, theta_old)
    return rows


def plot_results(detail_df: pd.DataFrame, output_root: pathlib.Path) -> None:
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    order = ["noG", "linear_plus", "linear_minus", "extrap_forward_from_plus", "extrap_backward_from_minus"]

    fig, ax = plt.subplots(figsize=(11, 6))
    data = [detail_df.loc[detail_df["candidate_name"] == name, "actual_V_change"].dropna() for name in order]
    ax.boxplot(data, tick_labels=order)
    ax.set_title("Stage 4.9 Actual Composite Merit Change")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(plots_dir / "stage49_extrapolation_vs_linear_merit.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    data = [detail_df.loc[detail_df["candidate_name"] == name, "actual_C_change"].dropna() for name in order]
    ax.boxplot(data, tick_labels=order)
    ax.set_title("Stage 4.9 Actual Cost Change")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(plots_dir / "stage49_extrapolation_vs_linear_cost.png", dpi=180)
    plt.close(fig)

    agg = detail_df.groupby("candidate_name")[["actual_V_change", "actual_C_change", "approx_kl", "clip_fraction"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(agg["actual_V_change"], agg["actual_C_change"], s=80)
    for _, row in agg.iterrows():
        ax.annotate(row["candidate_name"], (row["actual_V_change"], row["actual_C_change"]))
    ax.set_xlabel("Mean actual composite merit change")
    ax.set_ylabel("Mean actual cost change")
    ax.set_title("Stage 4.9 Frontier")
    fig.tight_layout()
    fig.savefig(plots_dir / "stage49_extrapolation_frontier.png", dpi=180)
    plt.close(fig)


def write_report(detail_df: pd.DataFrame, summary_df: pd.DataFrame, output_root: pathlib.Path, selected_configs: List[Stage48Config]) -> None:
    lines = [
        "# Stage 4.9 Extrapolation Audit Report",
        "",
        f"- Selected config count: `{len(selected_configs)}`",
        f"- Detail rows: `{len(detail_df)}`",
        "",
        "## Mean by Candidate",
        "",
        summary_df.to_csv(index=False),
    ]
    (output_root / "stage49_extrapolation_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_root / "stage49_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    all_configs = build_configs(72)
    selected_configs = select_top_configs(pathlib.Path(args.stage48_summary), all_configs, args.num_configs)
    model = build_model(args)
    all_rows: List[Dict[str, object]] = []
    for role in ("protagonist", "adversary"):
        algo = role_algo(model, role)
        probes = collect_probe_batches(model, role, max(args.num_probes_per_role, 4))
        for config in selected_configs:
            diagnostics_csv_path = diagnostics_dir / f"{config.label}_{role}.csv"
            for train_probe, val_probe in pair_probes(probes):
                rows = run_probe_for_config(
                    algo=algo,
                    role=role,
                    config=config,
                    train_probe=train_probe,
                    val_probe=val_probe,
                    args=args,
                    diagnostics_csv_path=diagnostics_csv_path,
                )
                all_rows.extend(rows)

    detail_df = pd.DataFrame(all_rows)
    detail_df.to_csv(output_root / "stage49_extrapolation_audit.csv", index=False)

    summary_df = (
        detail_df.groupby("candidate_name")[
            [
                "actual_V_change",
                "actual_C_change",
                "approx_kl",
                "clip_fraction",
                "update_norm_post_cap",
                "val_actual_V_change",
                "val_actual_C_change",
                "actor_fraction_of_update",
            ]
        ]
        .mean()
        .reset_index()
    )
    summary_df.to_csv(output_root / "stage49_extrapolation_summary.csv", index=False)
    plot_results(detail_df, output_root)
    write_report(detail_df, summary_df, output_root, selected_configs)


if __name__ == "__main__":
    main()
