from __future__ import annotations

import argparse
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.full_policy_followup_common import load_rarl_for_eval, set_rarl_eval_mode
from scripts.full_policy_optimizer_probe import clone_state, collect_probe_batches, compute_loss_and_grads, named_parameters, restore_state, tensor_norm
from scripts.run_proposed_qp_new_v2_rawfg_audit import (
    apply_two_direction_delta,
    apply_update_cap,
    build_control_manager,
    classify_block,
    evaluate_state,
    finite_difference_g_raw,
    flatten_named_tensors,
    lyapunov_value,
)
from scripts.run_rawfg_M12_policy_distribution_audit import frame_to_text, load_method_runs
from scripts.run_rawfg_M13_action_mean_audit import (
    collect_probe_snapshots,
    distribution_from_raw_obs,
    make_raw_env,
    one_step_reward_proxy,
    open_policy_contexts,
    close_policy_contexts,
)


TARGET_METHOD = "proposed_qp_rawFG_eta1_cap003"
COMPARE_METHODS = ["sgd", "egm", "ppm"]
LAMBDA_VALUES = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


@dataclass(frozen=True)
class SolverSpec:
    rule: str
    lambda_L: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("M14 mixed Lyapunov / PPO-loss objective audit")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--role", type=str, default="protagonist", choices=["protagonist"])
    parser.add_argument("--num-probes", type=int, default=4)
    parser.add_argument("--fd-eps", type=float, default=1e-3)
    parser.add_argument("--beta-probe", type=float, default=1e-3)
    parser.add_argument("--gamma-probe", type=float, default=1e-6)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--vf-coef", type=float, default=1.0)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--actor-weight", type=float, default=1.0)
    parser.add_argument("--logstd-weight", type=float, default=1.0)
    parser.add_argument("--critic-weight", type=float, default=0.3)
    parser.add_argument("--beta-max", type=float, default=1e-2)
    parser.add_argument("--gamma-max", type=float, default=3e-5)
    parser.add_argument("--eta-ext", type=float, default=1.0)
    parser.add_argument("--update-cap", type=float, default=0.003)
    parser.add_argument("--grid-points", type=int, default=50)
    parser.add_argument("--n-clean-probe-states", type=int, default=1000)
    parser.add_argument("--n-clean-probe-samples", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=20260604)
    return parser.parse_args()


def scalar_corr(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=np.float64)
    y_arr = np.asarray(list(y), dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 2:
        return float("nan")
    x_sel = x_arr[mask]
    y_sel = y_arr[mask]
    if np.std(x_sel) < 1e-12 or np.std(y_sel) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x_sel, y_sel)[0, 1])


def load_target_rarl(saved_run: object, device: str):
    model, vec_env = load_rarl_for_eval(saved_run, adv_impact="control", adv_strength=1.0, device=device)
    set_rarl_eval_mode(model, vec_env, operating_mode="protagonist", adv_strength=1.0)
    return model, vec_env


def block_norms(grads: Dict[str, th.Tensor], selected_names: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for block in ["actor", "logstd", "critic"]:
        names = [name for name in selected_names if classify_block(name) == block]
        if names:
            out[f"{block}_F_norm"] = tensor_norm(flatten_named_tensors(grads, names))
        else:
            out[f"{block}_F_norm"] = 0.0
    return out


def evaluate_objective_state(
    *,
    algo,
    rollout_data,
    named_params,
    theta_state: Dict[str, th.Tensor],
    max_grad_norm: float,
    vf_coef: float,
    ent_coef: float,
    actor_weight: float,
    logstd_weight: float,
    critic_weight: float,
    selected_names: Sequence[str],
) -> Dict[str, object]:
    eval_info = evaluate_state(
        algo=algo,
        rollout_data=rollout_data,
        named_params=named_params,
        theta_state=theta_state,
        max_grad_norm=max_grad_norm,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        actor_weight=actor_weight,
        logstd_weight=logstd_weight,
        critic_weight=critic_weight,
        selected_names=selected_names,
    )
    eval_info["L"] = float(eval_info["total_loss"])
    return eval_info


def estimate_quadratic_scalar(
    *,
    value_fn,
    beta_probe: float,
    gamma_probe: float,
    center_value: float,
) -> Dict[str, float]:
    db = beta_probe
    dg = gamma_probe
    vp_plus = float(value_fn(db, 0.0))
    vp_minus = float(value_fn(-db, 0.0))
    vr_plus = float(value_fn(0.0, dg))
    vr_minus = float(value_fn(0.0, -dg))
    vpp = float(value_fn(db, dg))
    vpm = float(value_fn(db, -dg))
    vmp = float(value_fn(-db, dg))
    vmm = float(value_fn(-db, -dg))
    a = (vp_plus - vp_minus) / (2.0 * db)
    c = (vp_plus - 2.0 * center_value + vp_minus) / (db * db)
    b = (vr_plus - vr_minus) / (2.0 * dg)
    k = (vr_plus - 2.0 * center_value + vr_minus) / (dg * dg)
    h = (vpp - vpm - vmp + vmm) / (4.0 * db * dg)
    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "h": float(h),
        "k": float(k),
        "vp_plus": vp_plus,
        "vp_minus": vp_minus,
        "vr_plus": vr_plus,
        "vr_minus": vr_minus,
        "vpp": vpp,
        "vpm": vpm,
        "vmp": vmp,
        "vmm": vmm,
    }


def q_value(beta: float, gamma: float, coeffs: Dict[str, float]) -> float:
    return (
        coeffs["a"] * beta
        + coeffs["b"] * gamma
        + 0.5 * coeffs["c"] * beta * beta
        + coeffs["h"] * beta * gamma
        + 0.5 * coeffs["k"] * gamma * gamma
    )


def dense_grid_candidates(beta_max: float, gamma_max: float, grid_points: int) -> pd.DataFrame:
    beta_grid = np.linspace(0.0, beta_max, grid_points)
    gamma_grid = np.linspace(0.0, gamma_max, grid_points)
    rows = [{"beta": float(beta), "gamma": float(gamma)} for beta in beta_grid for gamma in gamma_grid]
    return pd.DataFrame(rows)


def select_candidate(
    grid_df: pd.DataFrame,
    qf_coeffs: Dict[str, float],
    ql_coeffs: Dict[str, float],
    solver: SolverSpec,
) -> Dict[str, object]:
    work = grid_df.copy()
    work["q_F_pred"] = work.apply(lambda r: q_value(float(r["beta"]), float(r["gamma"]), qf_coeffs), axis=1)
    work["q_L_pred"] = work.apply(lambda r: q_value(float(r["beta"]), float(r["gamma"]), ql_coeffs), axis=1)
    if solver.rule == "pure_lyapunov":
        work["objective"] = work["q_F_pred"]
    elif solver.rule == "mixed_weighted":
        assert solver.lambda_L is not None
        work["objective"] = work["q_F_pred"] + float(solver.lambda_L) * work["q_L_pred"]
    elif solver.rule == "loss_constrained_lyapunov":
        work = work[work["q_L_pred"] <= 0.0].copy()
        work["objective"] = work["q_F_pred"]
    elif solver.rule == "lyapunov_constrained_loss":
        work = work[work["q_F_pred"] <= 0.0].copy()
        work["objective"] = work["q_L_pred"]
    else:
        raise ValueError(solver.rule)

    if work.empty:
        return {
            "beta": 0.0,
            "gamma": 0.0,
            "q_F_pred": 0.0,
            "q_L_pred": 0.0,
            "selected_case": "empty_feasible_fallback",
        }
    best = work.sort_values(["objective", "beta", "gamma"]).iloc[0]
    return {
        "beta": float(best["beta"]),
        "gamma": float(best["gamma"]),
        "q_F_pred": float(best["q_F_pred"]),
        "q_L_pred": float(best["q_L_pred"]),
        "selected_case": solver.rule if solver.lambda_L is None else f"{solver.rule}_lambda{solver.lambda_L:g}",
    }


def build_solver_specs() -> List[SolverSpec]:
    specs = [SolverSpec(rule="pure_lyapunov", lambda_L=None)]
    specs.extend(SolverSpec(rule="mixed_weighted", lambda_L=value) for value in LAMBDA_VALUES)
    specs.append(SolverSpec(rule="loss_constrained_lyapunov", lambda_L=None))
    specs.append(SolverSpec(rule="lyapunov_constrained_loss", lambda_L=None))
    return specs


def build_probe_pairs(probes: Sequence[object]) -> List[tuple[object, object]]:
    pairs = []
    for idx in range(0, len(probes) - 1, 2):
        pairs.append((probes[idx], probes[idx + 1]))
    return pairs


def compute_coefficients_and_candidates(args: argparse.Namespace, output_root: pathlib.Path):
    method_runs = load_method_runs(output_root)
    target_run = method_runs[TARGET_METHOD]
    model, vec_env = load_target_rarl(target_run.saved_run, args.device)
    try:
        probes = collect_probe_batches(model, args.role, args.num_probes)
        probe_pairs = build_probe_pairs(probes)
        algo = model.protagonist
        named_params = named_parameters(algo.policy)
        selected_names = [name for name, _ in named_params]
        theta_old = clone_state(named_params)
        grid_df = dense_grid_candidates(args.beta_max, args.gamma_max, args.grid_points)
        solver_specs = build_solver_specs()

        coeff_rows: List[Dict[str, object]] = []
        compare_rows: List[Dict[str, object]] = []
        validation_rows: List[Dict[str, object]] = []
        candidate_cache: Dict[tuple[int, str, float | None], Dict[str, object]] = {}

        for pair_id, (train_probe, val_probe) in enumerate(probe_pairs, start=1):
            base_train = evaluate_objective_state(
                algo=algo,
                rollout_data=train_probe.rollout_data,
                named_params=named_params,
                theta_state=theta_old,
                max_grad_norm=args.max_grad_norm,
                vf_coef=args.vf_coef,
                ent_coef=args.ent_coef,
                actor_weight=args.actor_weight,
                logstd_weight=args.logstd_weight,
                critic_weight=args.critic_weight,
                selected_names=selected_names,
            )
            F_raw = {name: base_train["grads"][name].detach().clone() for name in selected_names}
            g_raw, g_valid, plus_eval = finite_difference_g_raw(
                algo=algo,
                rollout_data=train_probe.rollout_data,
                named_params=named_params,
                theta_old=theta_old,
                F_raw=F_raw,
                fd_eps=args.fd_eps,
                max_grad_norm=args.max_grad_norm,
                vf_coef=args.vf_coef,
                ent_coef=args.ent_coef,
                actor_weight=args.actor_weight,
                logstd_weight=args.logstd_weight,
                critic_weight=args.critic_weight,
                selected_names=selected_names,
            )
            g_vec = flatten_named_tensors(g_raw, selected_names)
            F_vec = flatten_named_tensors(F_raw, selected_names)
            p_map = {name: -F_raw[name] for name in selected_names}
            r_map = {name: g_raw[name] for name in selected_names}

            def eval_at(beta_scale: float, gamma_scale: float) -> Dict[str, object]:
                theta_tmp = {name: tensor.clone() for name, tensor in theta_old.items()}
                for name in selected_names:
                    theta_tmp[name] = theta_old[name] + beta_scale * p_map[name] + gamma_scale * r_map[name]
                return evaluate_objective_state(
                    algo=algo,
                    rollout_data=train_probe.rollout_data,
                    named_params=named_params,
                    theta_state=theta_tmp,
                    max_grad_norm=args.max_grad_norm,
                    vf_coef=args.vf_coef,
                    ent_coef=args.ent_coef,
                    actor_weight=args.actor_weight,
                    logstd_weight=args.logstd_weight,
                    critic_weight=args.critic_weight,
                    selected_names=selected_names,
                )

            qf_coeffs = estimate_quadratic_scalar(
                value_fn=lambda b, g: eval_at(b, g)["V"],
                beta_probe=args.beta_probe,
                gamma_probe=args.gamma_probe,
                center_value=float(base_train["V"]),
            )
            ql_coeffs = estimate_quadratic_scalar(
                value_fn=lambda b, g: eval_at(b, g)["L"],
                beta_probe=args.beta_probe,
                gamma_probe=args.gamma_probe,
                center_value=float(base_train["L"]),
            )
            coeff_rows.append(
                {
                    "pair_id": pair_id,
                    "train_probe_id": int(train_probe.probe_idx),
                    "val_probe_id": int(val_probe.probe_idx),
                    "checkpoint_method": TARGET_METHOD,
                    "F_norm": float(tensor_norm(F_vec)),
                    "G_norm": float(tensor_norm(g_vec)),
                    "finite_difference_valid": int(g_valid),
                    "fd_eps": float(args.fd_eps),
                    "V0": float(base_train["V"]),
                    "L0": float(base_train["L"]),
                    "a_F": qf_coeffs["a"],
                    "b_F": qf_coeffs["b"],
                    "c_F": qf_coeffs["c"],
                    "h_F": qf_coeffs["h"],
                    "k_F": qf_coeffs["k"],
                    "a_L": ql_coeffs["a"],
                    "b_L": ql_coeffs["b"],
                    "c_L": ql_coeffs["c"],
                    "h_L": ql_coeffs["h"],
                    "k_L": ql_coeffs["k"],
                    "base_policy_loss": float(base_train["policy_loss"]),
                    "base_value_loss": float(base_train["value_loss"]),
                    "base_entropy_loss": float(base_train["entropy_loss"]),
                    **block_norms(base_train["grads"], selected_names),
                }
            )

            for solver in solver_specs:
                selected = select_candidate(grid_df, qf_coeffs, ql_coeffs, solver)
                theta_candidate = apply_two_direction_delta(
                    theta_old=theta_old,
                    f_map=F_raw,
                    g_map=g_raw,
                    beta=float(selected["beta"]),
                    gamma=float(selected["gamma"]),
                    selected_names=selected_names,
                    eta=args.eta_ext,
                )
                theta_capped, update_norm_pre, update_norm_post, cap_active = apply_update_cap(
                    theta_old,
                    theta_candidate,
                    selected_names,
                    args.update_cap,
                )
                train_after = evaluate_objective_state(
                    algo=algo,
                    rollout_data=train_probe.rollout_data,
                    named_params=named_params,
                    theta_state=theta_capped,
                    max_grad_norm=args.max_grad_norm,
                    vf_coef=args.vf_coef,
                    ent_coef=args.ent_coef,
                    actor_weight=args.actor_weight,
                    logstd_weight=args.logstd_weight,
                    critic_weight=args.critic_weight,
                    selected_names=selected_names,
                )
                val_base = evaluate_objective_state(
                    algo=algo,
                    rollout_data=val_probe.rollout_data,
                    named_params=named_params,
                    theta_state=theta_old,
                    max_grad_norm=args.max_grad_norm,
                    vf_coef=args.vf_coef,
                    ent_coef=args.ent_coef,
                    actor_weight=args.actor_weight,
                    logstd_weight=args.logstd_weight,
                    critic_weight=args.critic_weight,
                    selected_names=selected_names,
                )
                val_after = evaluate_objective_state(
                    algo=algo,
                    rollout_data=val_probe.rollout_data,
                    named_params=named_params,
                    theta_state=theta_capped,
                    max_grad_norm=args.max_grad_norm,
                    vf_coef=args.vf_coef,
                    ent_coef=args.ent_coef,
                    actor_weight=args.actor_weight,
                    logstd_weight=args.logstd_weight,
                    critic_weight=args.critic_weight,
                    selected_names=selected_names,
                )
                rule_label = solver.rule if solver.lambda_L is None else f"{solver.rule}_lambda{solver.lambda_L:g}"
                row_common = {
                    "pair_id": pair_id,
                    "train_probe_id": int(train_probe.probe_idx),
                    "val_probe_id": int(val_probe.probe_idx),
                    "rule": solver.rule,
                    "lambda_L": float(solver.lambda_L) if solver.lambda_L is not None else np.nan,
                    "rule_label": rule_label,
                    "beta": float(selected["beta"]),
                    "gamma": float(selected["gamma"]),
                    "update_norm_pre_cap": float(update_norm_pre),
                    "update_norm_post_cap": float(update_norm_post),
                    "cap_active": int(cap_active),
                    "q_F_pred": float(selected["q_F_pred"]),
                    "q_L_pred": float(selected["q_L_pred"]),
                    "actual_V_change": float(train_after["V"] - base_train["V"]),
                    "actual_PPO_loss_change": float(train_after["L"] - base_train["L"]),
                    "approx_kl": float(train_after["approx_kl"]),
                    "clip_fraction": float(train_after["clip_fraction"]),
                    "gamma_active_frac": float(float(selected["gamma"]) > 1e-12),
                    "G_contribution_norm": float(float(selected["gamma"]) * tensor_norm(g_vec)),
                    "selected_case": selected["selected_case"],
                }
                compare_rows.append(row_common)
                validation_rows.append(
                    {
                        **row_common,
                        "train_V_change": float(train_after["V"] - base_train["V"]),
                        "train_loss_change": float(train_after["L"] - base_train["L"]),
                        "val_V_change": float(val_after["V"] - val_base["V"]),
                        "val_loss_change": float(val_after["L"] - val_base["L"]),
                        "train_val_gap_V": float((train_after["V"] - base_train["V"]) - (val_after["V"] - val_base["V"])),
                        "train_val_gap_loss": float((train_after["L"] - base_train["L"]) - (val_after["L"] - val_base["L"])),
                    }
                )
                candidate_cache[(pair_id, solver.rule, solver.lambda_L)] = {
                    "theta_capped": {name: tensor.clone() for name, tensor in theta_capped.items()},
                    "selected": dict(row_common),
                }
            restore_state(named_params, theta_old)

        coeff_df = pd.DataFrame(coeff_rows)
        compare_df = pd.DataFrame(compare_rows)
        val_df = pd.DataFrame(validation_rows)
        coeff_df.to_csv(output_root / "M14_quadratic_coefficients.csv", index=False)
        compare_df.to_csv(output_root / "M14_candidate_solver_comparison.csv", index=False)
        val_df.to_csv(output_root / "M14_validation_generalization.csv", index=False)
        return method_runs, target_run, coeff_df, compare_df, val_df, candidate_cache
    finally:
        vec_env.close()


def write_reports_and_plots(
    output_root: pathlib.Path,
    coeff_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> None:
    lines = [
        "# M14 Quadratic Coefficients Report",
        "",
        "- Target checkpoint: proposed_qp_rawFG_eta1_cap003 final protagonist policy.",
        "- Probes: fixed control-RARL protagonist minibatches collected without training updates.",
        "",
        frame_to_text(coeff_df),
    ]
    (output_root / "M14_quadratic_coefficients_report.md").write_text("\n".join(lines), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    trade = compare_df.copy()
    for rule, group in trade.groupby("rule_label"):
        axes[0].scatter(group["q_F_pred"], group["q_L_pred"], label=rule, s=24)
    axes[0].set_title("Predicted q_F vs q_L")
    axes[0].set_xlabel("q_F_pred")
    axes[0].set_ylabel("q_L_pred")
    axes[0].grid(alpha=0.3)

    beta_plot = compare_df.copy()
    beta_plot["x_label"] = beta_plot["rule_label"]
    axes[1].scatter(beta_plot["x_label"], beta_plot["beta"], label="beta")
    axes[1].scatter(beta_plot["x_label"], beta_plot["gamma"] * 1e3, label="gamma x1e3")
    axes[1].set_title("Selected beta / gamma by rule")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    for rule, group in compare_df.groupby("rule_label"):
        axes[2].scatter(group["actual_V_change"], group["actual_PPO_loss_change"], label=rule, s=24)
    axes[2].set_title("Actual V vs PPO loss change")
    axes[2].set_xlabel("actual_V_change")
    axes[2].set_ylabel("actual_PPO_loss_change")
    axes[2].grid(alpha=0.3)
    axes[0].legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(output_root / "plots" / "M14_qF_vs_qL_tradeoff.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    agg = compare_df.groupby("rule_label", as_index=False)[["beta", "gamma"]].mean()
    axes[0].bar(agg["rule_label"], agg["beta"])
    axes[0].set_title("Mean beta by rule")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(alpha=0.3)
    axes[1].bar(agg["rule_label"], agg["gamma"])
    axes[1].set_title("Mean gamma by rule")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_root / "plots" / "M14_beta_gamma_by_rule.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    agg2 = compare_df.groupby("rule_label", as_index=False)[["actual_V_change", "actual_PPO_loss_change"]].mean()
    axes[0].bar(agg2["rule_label"], agg2["actual_V_change"])
    axes[0].set_title("Mean actual V change")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(alpha=0.3)
    axes[1].bar(agg2["rule_label"], agg2["actual_PPO_loss_change"])
    axes[1].set_title("Mean actual PPO loss change")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_root / "plots" / "M14_actual_V_and_loss_change.png", dpi=200)
    plt.close(fig)

    best_mixed = compare_df[compare_df["rule"] == "mixed_weighted"].groupby("lambda_L", as_index=False)[["actual_V_change", "actual_PPO_loss_change"]].mean()
    lines = [
        "# M14 Candidate Solver Comparison Report",
        "",
        f"- Does pure Lyapunov reduce V but hurt PPO loss? {'Yes' if compare_df[compare_df['rule']=='pure_lyapunov']['actual_V_change'].mean() < 0 and compare_df[compare_df['rule']=='pure_lyapunov']['actual_PPO_loss_change'].mean() > 0 else 'No'}",
        "",
        "## Per-rule averages",
        frame_to_text(compare_df.groupby("rule_label", as_index=False)[['beta','gamma','actual_V_change','actual_PPO_loss_change','approx_kl','clip_fraction']].mean()),
        "",
        "## Mixed lambdas",
        frame_to_text(best_mixed),
    ]
    (output_root / "M14_candidate_solver_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")

    mixed_candidates = val_df[val_df["rule"] == "mixed_weighted"].groupby("lambda_L", as_index=False)[["val_V_change", "val_loss_change"]].mean()
    if not mixed_candidates.empty:
        mixed_candidates["tradeoff_score"] = mixed_candidates["val_V_change"] + mixed_candidates["val_loss_change"]
        best_lambda = float(mixed_candidates.sort_values(["val_loss_change", "val_V_change"]).iloc[0]["lambda_L"])
    else:
        best_lambda = float("nan")
    lines = [
        "# M14 Validation Generalization Report",
        "",
        f"1. Does pure Lyapunov reduce V but hurt PPO loss? {'Yes' if val_df[val_df['rule']=='pure_lyapunov']['train_V_change'].mean() < 0 and val_df[val_df['rule']=='pure_lyapunov']['train_loss_change'].mean() > 0 else 'No'}",
        f"2. Does mixed objective reduce V while also improving PPO loss? {'Yes' if any((mixed_candidates['val_V_change'] < 0) & (mixed_candidates['val_loss_change'] < 0)) else 'No'}",
        f"3. Which lambda_L gives best V/loss tradeoff? {best_lambda}",
        f"4. Does constrained objective avoid the M10 failure mode? {'Yes' if val_df[val_df['rule'].isin(['loss_constrained_lyapunov','lyapunov_constrained_loss'])]['approx_kl'].mean() <= 0.1 else 'No'}",
        f"5. Does candidate generalize from train minibatch to validation minibatch? {'Yes' if abs(val_df['train_val_gap_loss']).mean() < 0.1 else 'Partially/No'}",
        "",
        frame_to_text(val_df.groupby('rule_label', as_index=False)[['train_V_change','train_loss_change','val_V_change','val_loss_change','train_val_gap_V','train_val_gap_loss']].mean()),
    ]
    (output_root / "M14_validation_generalization_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_deterministic_action_proxy(
    args: argparse.Namespace,
    output_root: pathlib.Path,
    method_runs: Dict[str, object],
    candidate_cache: Dict[tuple[int, str, float | None], Dict[str, object]],
) -> pd.DataFrame:
    contexts = open_policy_contexts(method_runs, args.device)
    raw_env = make_raw_env(method_runs["sgd"].saved_run.args_data["env"])
    action_low = np.asarray(raw_env.action_space.low, dtype=np.float64)
    action_high = np.asarray(raw_env.action_space.high, dtype=np.float64)
    try:
        raw_env.reset(seed=args.eval_seed)
        snapshots = collect_probe_snapshots(contexts["sgd"], n_probe_states=args.n_clean_probe_states, eval_seed=args.eval_seed)
        proposed_ctx = contexts[TARGET_METHOD]
        proposed_policy = proposed_ctx.policy
        named_params = named_parameters(proposed_policy)
        selected_names = [name for name, _ in named_params]
        theta_original = clone_state(named_params)

        candidate_specs = []
        for rule, lam in [("pure_lyapunov", None), ("loss_constrained_lyapunov", None), ("lyapunov_constrained_loss", None)]:
            candidate_specs.append((1, rule, lam))
        for lam in LAMBDA_VALUES:
            candidate_specs.append((1, "mixed_weighted", lam))

        rows: List[Dict[str, object]] = []
        baseline_means = {m: [] for m in COMPARE_METHODS}
        for snapshot in snapshots:
            raw_obs = snapshot["raw_obs"]
            for m in COMPARE_METHODS:
                mean_m, _ = distribution_from_raw_obs(contexts[m], raw_obs)
                baseline_means[m].append(np.clip(mean_m, action_low, action_high))

        for pair_id, rule, lam in candidate_specs:
            cache = candidate_cache.get((pair_id, rule, lam))
            if cache is None:
                continue
            theta_candidate = cache["theta_capped"]
            restore_state(named_params, theta_candidate)
            for snapshot in snapshots:
                raw_obs = snapshot["raw_obs"]
                state = snapshot["state"]
                mean, std = distribution_from_raw_obs(proposed_ctx, raw_obs)
                mean = np.clip(mean, action_low, action_high)
                reward_det, _, _ = one_step_reward_proxy(raw_env, state, mean)
                # stochastic performance proxy: sample a small set and average one-step reward
                _, samples = distribution_from_raw_obs(proposed_ctx, raw_obs), None
                mean_base, sampled_actions = None, None
                mean_base, sampled_actions = distribution_from_raw_obs(proposed_ctx, raw_obs)[0], None
                _, sampled_actions = None, None
                mean_dummy, sampled_actions = None, None
                # sample through helper to preserve exact policy distribution
                from scripts.run_rawfg_M13_action_mean_audit import sample_actions_from_raw_obs as _sample_actions_from_raw_obs
                _, sampled_actions = _sample_actions_from_raw_obs(proposed_ctx, raw_obs, args.n_clean_probe_samples)
                sample_rewards = []
                for sample in sampled_actions:
                    clipped = np.clip(sample, action_low, action_high)
                    reward_s, _, _ = one_step_reward_proxy(raw_env, state, clipped)
                    sample_rewards.append(reward_s)
                reward_stoch = float(np.mean(sample_rewards)) if sample_rewards else float("nan")
                egm_mean, _ = distribution_from_raw_obs(contexts["egm"], raw_obs)
                ppm_mean, _ = distribution_from_raw_obs(contexts["ppm"], raw_obs)
                rows.append(
                    {
                        "pair_id": pair_id,
                        "rule": rule,
                        "lambda_L": float(lam) if lam is not None else np.nan,
                        "rule_label": rule if lam is None else f"{rule}_lambda{lam:g}",
                        "state_id": int(snapshot["state_id"]),
                        "deterministic_action_mean": float(np.linalg.norm(mean)),
                        "deterministic_reward_proxy": reward_det,
                        "stochastic_reward_proxy": reward_stoch,
                        "distance_to_egm_mean": float(np.linalg.norm(mean - np.clip(egm_mean, action_low, action_high))),
                        "distance_to_ppm_mean": float(np.linalg.norm(mean - np.clip(ppm_mean, action_low, action_high))),
                        "action_std_norm": float(np.linalg.norm(std)),
                    }
                )
            restore_state(named_params, theta_original)
        df = pd.DataFrame(rows)
        df.to_csv(output_root / "M14_deterministic_action_proxy.csv", index=False)

        summary = df.groupby("rule_label", as_index=False)[["deterministic_reward_proxy", "stochastic_reward_proxy", "distance_to_egm_mean", "distance_to_ppm_mean", "action_std_norm"]].mean()
        pure_v = summary[summary["rule_label"] == "pure_lyapunov"].iloc[0]
        mixed = summary[summary["rule_label"].str.startswith("mixed_weighted")].sort_values(["deterministic_reward_proxy"], ascending=False).iloc[0]
        lines = [
            "# M14 Deterministic Action Proxy Report",
            "",
            f"1. Does mixed objective improve deterministic action proxy more than pure V? {'Yes' if mixed['deterministic_reward_proxy'] > pure_v['deterministic_reward_proxy'] else 'No'}",
            f"2. Does it move proposed mean action closer to EGM/PPM mean action? {'Yes' if mixed['distance_to_egm_mean'] < pure_v['distance_to_egm_mean'] or mixed['distance_to_ppm_mean'] < pure_v['distance_to_ppm_mean'] else 'No'}",
            f"3. Does it preserve stochastic performance proxy? {'Yes' if mixed['stochastic_reward_proxy'] >= pure_v['stochastic_reward_proxy'] - 1e-6 else 'No'}",
            "",
            frame_to_text(summary),
        ]
        (output_root / "M14_deterministic_action_proxy_report.md").write_text("\n".join(lines), encoding="utf-8")
        return df
    finally:
        raw_env.close()
        close_policy_contexts(contexts)


def write_root_cause_report(output_root: pathlib.Path, compare_df: pd.DataFrame, val_df: pd.DataFrame, proxy_df: pd.DataFrame) -> None:
    summary = compare_df.groupby("rule_label", as_index=False)[["actual_V_change", "actual_PPO_loss_change", "approx_kl", "clip_fraction"]].mean()
    pure = summary[summary["rule_label"] == "pure_lyapunov"].iloc[0]
    mixed_summary = summary[summary["rule_label"].str.startswith("mixed_weighted")].copy()
    proxy_summary = proxy_df.groupby("rule_label", as_index=False)[["deterministic_reward_proxy", "stochastic_reward_proxy", "distance_to_egm_mean", "distance_to_ppm_mean"]].mean()
    pure_proxy = proxy_summary[proxy_summary["rule_label"] == "pure_lyapunov"].iloc[0]
    merged = mixed_summary.merge(proxy_summary, on="rule_label", how="left")
    best_mixed = merged.sort_values(["deterministic_reward_proxy", "actual_PPO_loss_change"], ascending=[False, True]).iloc[0] if not merged.empty else None
    constrained = summary[summary["rule_label"].isin(["loss_constrained_lyapunov", "lyapunov_constrained_loss"])]

    recommendation = "A. keep pure V"
    if best_mixed is not None and best_mixed["actual_V_change"] <= pure["actual_V_change"] + 1e-6 and best_mixed["actual_PPO_loss_change"] < pure["actual_PPO_loss_change"]:
        recommendation = "B. use mixed objective q_F + lambda q_L"
    elif not constrained.empty and constrained["actual_PPO_loss_change"].min() < pure["actual_PPO_loss_change"]:
        recommendation = "C. use loss-constrained Lyapunov"
    elif not constrained.empty and constrained["actual_V_change"].min() < pure["actual_V_change"]:
        recommendation = "D. use Lyapunov-constrained loss"
    if best_mixed is not None and (best_mixed["distance_to_egm_mean"] > pure_proxy["distance_to_egm_mean"]):
        recommendation = "E. abandon full_policy QP and try actor_mean-only QP"
    if best_mixed is not None and best_mixed["deterministic_reward_proxy"] > pure_proxy["deterministic_reward_proxy"] and best_mixed["distance_to_egm_mean"] < pure_proxy["distance_to_egm_mean"]:
        recommendation = "F. include deterministic-action proxy in objective"

    lines = [
        "# M14 Mixed Lyapunov Loss Root Cause Report",
        "",
        f"- Recommendation: **{recommendation}**",
        "",
        "## Key answers",
        f"- Pure Lyapunov mean actual_V_change = {pure['actual_V_change']:.6g}",
        f"- Pure Lyapunov mean actual_PPO_loss_change = {pure['actual_PPO_loss_change']:.6g}",
    ]
    if best_mixed is not None:
        lines.extend(
            [
                f"- Best mixed rule label = {best_mixed['rule_label']}",
                f"- Best mixed deterministic reward proxy = {best_mixed['deterministic_reward_proxy']:.6g}",
                f"- Best mixed actual_V_change = {best_mixed['actual_V_change']:.6g}",
                f"- Best mixed actual_PPO_loss_change = {best_mixed['actual_PPO_loss_change']:.6g}",
            ]
        )
    lines.extend(
        [
            "",
            "## Compare summary",
            frame_to_text(summary),
            "",
            "## Validation summary",
            frame_to_text(val_df.groupby('rule_label', as_index=False)[['train_V_change','train_loss_change','val_V_change','val_loss_change']].mean()),
            "",
            "## Deterministic proxy summary",
            frame_to_text(proxy_summary),
        ]
    )
    (output_root / "M14_mixed_lyapunov_loss_root_cause_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    (output_root / "plots").mkdir(parents=True, exist_ok=True)
    method_runs, target_run, coeff_df, compare_df, val_df, candidate_cache = compute_coefficients_and_candidates(args, output_root)
    write_reports_and_plots(output_root, coeff_df, compare_df, val_df)
    proxy_df = run_deterministic_action_proxy(args, output_root, method_runs, candidate_cache)
    write_root_cause_report(output_root, compare_df, val_df, proxy_df)


if __name__ == "__main__":
    main()
