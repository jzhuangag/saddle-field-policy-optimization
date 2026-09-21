from __future__ import annotations

import argparse
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from models.proposed_qp_perflyap import ProposedNoGPerfLyapOptimizer, ProposedQPPerfLyapOptimizer
from scripts.full_policy_optimizer_probe import clone_state, collect_probe_batches, named_parameters, restore_state, tensor_norm
from scripts.run_perflyap_stage4_preflight import build_configs, build_model, build_eval_closure, role_algo


EPS = 1e-12


@dataclass(frozen=True)
class CandidateEval:
    candidate_type: str
    beta: float
    gamma: float
    q_pred_total_merit: float
    q_pred_norm_term: float
    q_pred_cost_term: float
    actual_total_merit_change: float
    actual_norm_term_change: float
    actual_cost_change: float
    update_norm_pre_cap: float
    update_norm_post_cap: float
    cap_active: int
    cap_scale: float
    approx_kl: float
    clip_fraction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 4.5 perfLyap invariant/sign/drift audit")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-probes-per-role", type=int, default=4)
    parser.add_argument("--eta-egm", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--vf-coef", type=float, default=1.0)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--max-configs", type=int, default=80)
    parser.add_argument("--gamma-sweep-topk", type=int, default=6)
    return parser.parse_args()


def classify_perf_block(name: str) -> str:
    if "log_std" in name:
        return "logstd"
    if "value" in name:
        return "critic"
    return "actor"


def flatten_named_tensors(named_map: Dict[str, th.Tensor], selected_names: Sequence[str]) -> th.Tensor:
    pieces = [named_map[name].reshape(-1) for name in selected_names]
    if not pieces:
        return th.zeros(0)
    return th.cat(pieces)


def scope_block_weights(scope: str) -> Dict[str, float]:
    if scope in {"actor_mean_only", "actor_game", "actor_mean_plus_adv_actor_mean"}:
        return {"actor": 1.0, "logstd": 0.0, "critic": 0.0}
    if scope == "critic_downweighted":
        return {"actor": 1.0, "logstd": 0.1, "critic": 0.01}
    if scope == "logstd_excluded":
        return {"actor": 1.0, "logstd": 0.0, "critic": 0.05}
    return {"actor": 1.0, "logstd": 0.1, "critic": 0.05}


def selected_names_for_scope(scope: str, named_params) -> List[str]:
    if scope in {"actor_mean_only", "actor_game", "actor_mean_plus_adv_actor_mean"}:
        return [name for name, _ in named_params if classify_perf_block(name) == "actor"]
    if scope == "logstd_excluded":
        return [name for name, _ in named_params if classify_perf_block(name) != "logstd"]
    return [name for name, _ in named_params]


def block_mean_square(grads_selected: Dict[str, th.Tensor], selected_names: Sequence[str], block: str) -> float:
    names = [name for name in selected_names if classify_perf_block(name) == block]
    if not names:
        return 0.0
    vec = flatten_named_tensors(grads_selected, names)
    return float(th.mean(vec * vec).item()) if vec.numel() else 0.0


def evaluate_perf_state(
    *,
    algo,
    eval_closure,
    theta_state: Dict[str, th.Tensor],
    selected_names: Sequence[str],
    scope: str,
    lambda_N: float,
    lambda_P: float,
    lambda_critic: float,
    logstd_weight: float,
    norm_scale: float | None,
    perf_scale: float | None,
) -> Dict[str, object]:
    raw = eval_closure(theta_override=theta_state, backward=True, grad_scope_names=list(selected_names))
    grads_selected = {name: raw["grads"][name].detach().clone() for name in selected_names}
    weights = scope_block_weights(scope)
    actor_norm_term = 0.5 * weights["actor"] * block_mean_square(grads_selected, selected_names, "actor")
    logstd_norm_term = 0.5 * weights["logstd"] * block_mean_square(grads_selected, selected_names, "logstd")
    critic_norm_term = 0.5 * weights["critic"] * block_mean_square(grads_selected, selected_names, "critic")
    norm_term = actor_norm_term + logstd_norm_term + critic_norm_term
    policy_component = float(raw["policy_loss"])
    critic_component = float(lambda_critic * raw["value_loss"])
    logstd_component = float(logstd_weight * raw["entropy_loss"])
    cost_term = policy_component + critic_component + logstd_component
    info = dict(raw)
    info["grads_selected"] = grads_selected
    info["actor_norm_term"] = actor_norm_term
    info["logstd_norm_term"] = logstd_norm_term
    info["critic_norm_term"] = critic_norm_term
    info["norm_term"] = norm_term
    info["policy_component"] = policy_component
    info["critic_component"] = critic_component
    info["logstd_component"] = logstd_component
    info["cost_term"] = cost_term
    if norm_scale is not None and perf_scale is not None:
        info["total_merit"] = lambda_N * norm_term / max(norm_scale, EPS) + lambda_P * cost_term / max(perf_scale, EPS)
    return info


def compute_scales(base_eval: Dict[str, object]) -> Tuple[float, float]:
    return max(abs(float(base_eval["norm_term"])), EPS), max(abs(float(base_eval["cost_term"])), EPS)


def finite_difference_g(
    *,
    theta_old: Dict[str, th.Tensor],
    F_raw: Dict[str, th.Tensor],
    selected_names: Sequence[str],
    fd_eps: float,
    variant: str,
    eval_state_fn,
) -> Tuple[Dict[str, th.Tensor], Dict[str, object], Dict[str, object] | None]:
    if variant == "forward_plus":
        theta_plus = {name: tensor.clone() for name, tensor in theta_old.items()}
        for name in selected_names:
            theta_plus[name] = theta_old[name] + fd_eps * F_raw[name]
        plus_eval = eval_state_fn(theta_plus)
        g_map = {name: (plus_eval["grads_selected"][name] - F_raw[name]) / fd_eps for name in selected_names}
        return g_map, plus_eval, None
    if variant == "forward_minus_to_egm":
        theta_minus = {name: tensor.clone() for name, tensor in theta_old.items()}
        for name in selected_names:
            theta_minus[name] = theta_old[name] - fd_eps * F_raw[name]
        minus_eval = eval_state_fn(theta_minus)
        g_map = {name: (F_raw[name] - minus_eval["grads_selected"][name]) / fd_eps for name in selected_names}
        return g_map, minus_eval, None
    if variant == "central":
        theta_plus = {name: tensor.clone() for name, tensor in theta_old.items()}
        theta_minus = {name: tensor.clone() for name, tensor in theta_old.items()}
        for name in selected_names:
            theta_plus[name] = theta_old[name] + fd_eps * F_raw[name]
            theta_minus[name] = theta_old[name] - fd_eps * F_raw[name]
        plus_eval = eval_state_fn(theta_plus)
        minus_eval = eval_state_fn(theta_minus)
        g_map = {
            name: (plus_eval["grads_selected"][name] - minus_eval["grads_selected"][name]) / (2.0 * fd_eps)
            for name in selected_names
        }
        return g_map, plus_eval, minus_eval
    raise ValueError(variant)


def q_value(beta: float, gamma: float, coeffs: Dict[str, float]) -> float:
    return (
        coeffs["l_beta"] * beta
        + coeffs["l_gamma"] * gamma
        + 0.5 * coeffs["H_bb"] * beta * beta
        + coeffs["H_bg"] * beta * gamma
        + 0.5 * coeffs["H_gg"] * gamma * gamma
    )


def estimate_quadratic(
    *,
    theta_old: Dict[str, th.Tensor],
    d_beta: Dict[str, th.Tensor],
    d_gamma: Dict[str, th.Tensor],
    center_value: float,
    beta_probe: float,
    gamma_probe: float,
    eval_scalar_fn,
) -> Dict[str, float]:
    db = beta_probe
    dg = gamma_probe

    def value_at(beta_scale: float, gamma_scale: float) -> float:
        theta_tmp = {name: tensor.clone() for name, tensor in theta_old.items()}
        for name in d_beta.keys():
            theta_tmp[name] = theta_old[name] + beta_scale * d_beta[name] + gamma_scale * d_gamma[name]
        return float(eval_scalar_fn(theta_tmp))

    vp_plus = value_at(db, 0.0)
    vp_minus = value_at(-db, 0.0)
    vr_plus = value_at(0.0, dg)
    vr_minus = value_at(0.0, -dg)
    vpp = value_at(db, dg)
    vpm = value_at(db, -dg)
    vmp = value_at(-db, dg)
    vmm = value_at(-db, -dg)
    return {
        "l_beta": (vp_plus - vp_minus) / (2.0 * db),
        "H_bb": (vp_plus - 2.0 * center_value + vp_minus) / (db * db),
        "l_gamma": (vr_plus - vr_minus) / (2.0 * dg),
        "H_gg": (vr_plus - 2.0 * center_value + vr_minus) / (dg * dg),
        "H_bg": (vpp - vpm - vmp + vmm) / (4.0 * db * dg),
    }


def apply_candidate(
    *,
    theta_old: Dict[str, th.Tensor],
    selected_names: Sequence[str],
    F_raw: Dict[str, th.Tensor],
    G_raw: Dict[str, th.Tensor],
    beta: float,
    gamma: float,
    update_cap: float,
    eval_state_fn,
    total_coeffs: Dict[str, float],
    norm_coeffs: Dict[str, float],
    cost_coeffs: Dict[str, float],
    candidate_type: str,
) -> CandidateEval:
    theta_new = {name: tensor.clone() for name, tensor in theta_old.items()}
    delta_map = {}
    for name in selected_names:
        delta = -beta * F_raw[name] + gamma * G_raw[name]
        delta_map[name] = delta
        theta_new[name] = theta_old[name] + delta
    delta_vec = flatten_named_tensors(delta_map, selected_names)
    update_norm_pre = tensor_norm(delta_vec)
    cap_active = 0
    cap_scale = 1.0
    if math.isfinite(update_cap) and update_cap > 0.0 and update_norm_pre > update_cap:
        cap_scale = update_cap / max(update_norm_pre, EPS)
        for name in selected_names:
            theta_new[name] = theta_old[name] + cap_scale * delta_map[name]
        cap_active = 1
    update_norm_post = tensor_norm(flatten_named_tensors({name: theta_new[name] - theta_old[name] for name in selected_names}, selected_names))
    after_eval = eval_state_fn(theta_new)
    beta_cap = beta * cap_scale
    gamma_cap = gamma * cap_scale
    return CandidateEval(
        candidate_type=candidate_type,
        beta=beta,
        gamma=gamma,
        q_pred_total_merit=q_value(beta, gamma, total_coeffs),
        q_pred_norm_term=q_value(beta, gamma, norm_coeffs),
        q_pred_cost_term=q_value(beta, gamma, cost_coeffs),
        actual_total_merit_change=float(after_eval["total_merit_change"]),
        actual_norm_term_change=float(after_eval["norm_term_change"]),
        actual_cost_change=float(after_eval["cost_term_change"]),
        update_norm_pre_cap=update_norm_pre,
        update_norm_post_cap=update_norm_post,
        cap_active=cap_active,
        cap_scale=cap_scale,
        approx_kl=float(after_eval["approx_kl"]),
        clip_fraction=float(after_eval["clip_fraction"]),
    )


def relative_error(a: th.Tensor, b: th.Tensor) -> float:
    return tensor_norm(a - b) / max(tensor_norm(b), EPS)


def pair_probes(probes: Sequence[object]) -> List[Tuple[object, object]]:
    out = []
    for idx in range(0, len(probes) - 1, 2):
        out.append((probes[idx], probes[idx + 1]))
    return out


def instantiate_optimizer(which: str, named_params, config, args, diagnostics_csv_path: pathlib.Path, role: str):
    optimizer_cls = ProposedQPPerfLyapOptimizer if which == "qp" else ProposedNoGPerfLyapOptimizer
    return optimizer_cls(
        [param for _, param in named_params],
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


def tol(value: float) -> float:
    return max(1e-8, 1e-4 * max(1.0, abs(value)))


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    configs = build_configs(args.max_configs)
    stage4_df = pd.read_csv(output_root / "stage4_preflight_summary.csv")
    top_failed_ids = (
        stage4_df.sort_values(["qp_actual_V_change_mean", "qp_actual_C_change_mean"]).head(args.gamma_sweep_topk)["config_id"].tolist()
    )
    config_by_id = {cfg.config_id: cfg for cfg in configs}
    model = build_model(args)

    invariant_rows: List[Dict[str, object]] = []
    sign_rows: List[Dict[str, object]] = []
    gsign_rows: List[Dict[str, object]] = []
    gamma_rows: List[Dict[str, object]] = []
    cap_rows: List[Dict[str, object]] = []

    try:
        role_pairs = {}
        for role in ["protagonist", "adversary"]:
            role_pairs[role] = pair_probes(collect_probe_batches(model, role, args.num_probes_per_role))

        for config in configs:
            for role, pairs in role_pairs.items():
                algo = role_algo(model, role)
                named_params = named_parameters(algo.policy)
                selected_names = selected_names_for_scope(config.scope, named_params)
                if not selected_names:
                    continue
                for train_probe, val_probe in pairs:
                    theta_old = clone_state(named_params)
                    train_eval_closure = build_eval_closure(algo, train_probe.rollout_data)
                    val_eval_closure = build_eval_closure(algo, val_probe.rollout_data)

                    base_eval_unscaled = evaluate_perf_state(
                        algo=algo,
                        eval_closure=train_eval_closure,
                        theta_state=theta_old,
                        selected_names=selected_names,
                        scope=config.scope,
                        lambda_N=config.lambda_N,
                        lambda_P=config.lambda_P,
                        lambda_critic=config.lambda_critic,
                        logstd_weight=config.logstd_weight,
                        norm_scale=None,
                        perf_scale=None,
                    )
                    norm_scale, perf_scale = compute_scales(base_eval_unscaled)

                    def train_eval_state(theta_state):
                        info = evaluate_perf_state(
                            algo=algo,
                            eval_closure=train_eval_closure,
                            theta_state=theta_state,
                            selected_names=selected_names,
                            scope=config.scope,
                            lambda_N=config.lambda_N,
                            lambda_P=config.lambda_P,
                            lambda_critic=config.lambda_critic,
                            logstd_weight=config.logstd_weight,
                            norm_scale=norm_scale,
                            perf_scale=perf_scale,
                        )
                        info["total_merit_change"] = float(info["total_merit"] - base_eval["total_merit"])
                        info["norm_term_change"] = float(info["norm_term"] - base_eval["norm_term"])
                        info["cost_term_change"] = float(info["cost_term"] - base_eval["cost_term"])
                        return info

                    base_eval = evaluate_perf_state(
                        algo=algo,
                        eval_closure=train_eval_closure,
                        theta_state=theta_old,
                        selected_names=selected_names,
                        scope=config.scope,
                        lambda_N=config.lambda_N,
                        lambda_P=config.lambda_P,
                        lambda_critic=config.lambda_critic,
                        logstd_weight=config.logstd_weight,
                        norm_scale=norm_scale,
                        perf_scale=perf_scale,
                    )

                    F_raw = {name: base_eval["grads_selected"][name].detach().clone() for name in selected_names}
                    F_vec = flatten_named_tensors(F_raw, selected_names)
                    G_plus, plus_eval, _ = finite_difference_g(
                        theta_old=theta_old,
                        F_raw=F_raw,
                        selected_names=selected_names,
                        fd_eps=config.fd_eps,
                        variant="forward_plus",
                        eval_state_fn=train_eval_state,
                    )
                    G_egm, minus_eval, _ = finite_difference_g(
                        theta_old=theta_old,
                        F_raw=F_raw,
                        selected_names=selected_names,
                        fd_eps=config.fd_eps,
                        variant="forward_minus_to_egm",
                        eval_state_fn=train_eval_state,
                    )
                    G_central, central_plus_eval, central_minus_eval = finite_difference_g(
                        theta_old=theta_old,
                        F_raw=F_raw,
                        selected_names=selected_names,
                        fd_eps=config.fd_eps,
                        variant="central",
                        eval_state_fn=train_eval_state,
                    )
                    current_G = G_plus
                    G_vec = flatten_named_tensors(current_G, selected_names)

                    d_beta = {name: -F_raw[name] for name in selected_names}
                    d_gamma = {name: current_G[name] for name in selected_names}
                    total_coeffs = estimate_quadratic(
                        theta_old=theta_old,
                        d_beta=d_beta,
                        d_gamma=d_gamma,
                        center_value=float(base_eval["total_merit"]),
                        beta_probe=config.beta_probe,
                        gamma_probe=config.gamma_probe,
                        eval_scalar_fn=lambda theta_state: train_eval_state(theta_state)["total_merit"],
                    )
                    norm_coeffs = estimate_quadratic(
                        theta_old=theta_old,
                        d_beta=d_beta,
                        d_gamma=d_gamma,
                        center_value=float(base_eval["norm_term"]),
                        beta_probe=config.beta_probe,
                        gamma_probe=config.gamma_probe,
                        eval_scalar_fn=lambda theta_state: train_eval_state(theta_state)["norm_term"],
                    )
                    cost_coeffs = estimate_quadratic(
                        theta_old=theta_old,
                        d_beta=d_beta,
                        d_gamma=d_gamma,
                        center_value=float(base_eval["cost_term"]),
                        beta_probe=config.beta_probe,
                        gamma_probe=config.gamma_probe,
                        eval_scalar_fn=lambda theta_state: train_eval_state(theta_state)["cost_term"],
                    )

                    # selected noG/QP from current optimizer logic
                    diag_dir = output_root / "stage45_tmp"
                    diag_dir.mkdir(parents=True, exist_ok=True)
                    noG_opt = instantiate_optimizer("nog", named_params, config, args, diag_dir / "nog.csv", role)
                    qp_opt = instantiate_optimizer("qp", named_params, config, args, diag_dir / "qp.csv", role)
                    noG_opt.step(eval_closure=train_eval_closure, named_params=named_params)
                    noG_metrics = dict(noG_opt.last_step_metrics)
                    restore_state(named_params, theta_old)
                    qp_opt.step(eval_closure=train_eval_closure, named_params=named_params)
                    qp_metrics = dict(qp_opt.last_step_metrics)
                    restore_state(named_params, theta_old)

                    beta_nog = float(noG_metrics["beta"])
                    beta_qp = float(qp_metrics["beta"])
                    gamma_qp = float(qp_metrics["gamma"])

                    # forced noG from QP total coeffs
                    H_bb = float(total_coeffs["H_bb"])
                    if H_bb > EPS and math.isfinite(H_bb):
                        beta_forced_nog = min(max(-float(total_coeffs["l_beta"]) / H_bb, 0.0), config.beta_max)
                    else:
                        q_zero = q_value(0.0, 0.0, total_coeffs)
                        q_bmax = q_value(config.beta_max, 0.0, total_coeffs)
                        beta_forced_nog = 0.0 if q_zero <= q_bmax else config.beta_max

                    # dense grid argmin on predicted total merit
                    beta_grid = np.linspace(0.0, config.beta_max, 101)
                    gamma_grid = np.linspace(0.0, config.gamma_max, 101)
                    bg_beta, bg_gamma = np.meshgrid(beta_grid, gamma_grid, indexing="ij")
                    q_grid = (
                        total_coeffs["l_beta"] * bg_beta
                        + total_coeffs["l_gamma"] * bg_gamma
                        + 0.5 * total_coeffs["H_bb"] * bg_beta * bg_beta
                        + total_coeffs["H_bg"] * bg_beta * bg_gamma
                        + 0.5 * total_coeffs["H_gg"] * bg_gamma * bg_gamma
                    )
                    best_idx = np.unravel_index(np.argmin(q_grid), q_grid.shape)
                    beta_best = float(beta_grid[best_idx[0]])
                    gamma_best = float(gamma_grid[best_idx[1]])

                    candidates = [
                        ("zero", 0.0, 0.0),
                        ("noG_selected", beta_nog, 0.0),
                        ("QP_selected", beta_qp, gamma_qp),
                        ("QP_forced_noG", beta_forced_nog, 0.0),
                        ("EGM_like", args.eta_egm, args.eta_egm * args.eta_egm),
                        ("best_dense_grid", beta_best, gamma_best),
                    ]
                    candidate_records = {}
                    for candidate_type, beta_val, gamma_val in candidates:
                        cand = apply_candidate(
                            theta_old=theta_old,
                            selected_names=selected_names,
                            F_raw=F_raw,
                            G_raw=current_G,
                            beta=beta_val,
                            gamma=gamma_val,
                            update_cap=config.update_cap,
                            eval_state_fn=train_eval_state,
                            total_coeffs=total_coeffs,
                            norm_coeffs=norm_coeffs,
                            cost_coeffs=cost_coeffs,
                            candidate_type=candidate_type,
                        )
                        candidate_records[candidate_type] = cand
                        invariant_rows.append(
                            {
                                "config_id": config.config_id,
                                "config_label": config.label,
                                "scope": config.scope,
                                "role": role,
                                "train_probe_id": int(train_probe.probe_idx),
                                "val_probe_id": int(val_probe.probe_idx),
                                "candidate_type": cand.candidate_type,
                                "q_pred_total_merit": cand.q_pred_total_merit,
                                "q_pred_norm_term": cand.q_pred_norm_term,
                                "q_pred_cost_term": cand.q_pred_cost_term,
                                "actual_total_merit_change": cand.actual_total_merit_change,
                                "actual_norm_term_change": cand.actual_norm_term_change,
                                "actual_cost_change": cand.actual_cost_change,
                                "beta": cand.beta,
                                "gamma": cand.gamma,
                                "update_norm_pre_cap": cand.update_norm_pre_cap,
                                "update_norm_post_cap": cand.update_norm_post_cap,
                                "cap_active": cand.cap_active,
                                "approx_kl": cand.approx_kl,
                                "clip_fraction": cand.clip_fraction,
                            }
                        )

                    noG_cand = candidate_records["noG_selected"]
                    qp_cand = candidate_records["QP_selected"]
                    best_grid_cand = candidate_records["best_dense_grid"]
                    invariant_rows[-1]["_dummy"] = 0  # keep list non-empty for later typing

                    # 4.5B active cost sign sanity
                    for eps in [1e-5, 1e-4, 1e-3]:
                        for sign_name, scale in [("minus_F", -eps), ("plus_F", eps)]:
                            theta_step = {name: tensor.clone() for name, tensor in theta_old.items()}
                            for name in selected_names:
                                theta_step[name] = theta_old[name] + scale * F_raw[name]
                            step_eval = train_eval_state(theta_step)
                            sign_rows.append(
                                {
                                    "config_id": config.config_id,
                                    "config_label": config.label,
                                    "scope": config.scope,
                                    "role": role,
                                    "train_probe_id": int(train_probe.probe_idx),
                                    "eps": eps,
                                    "step_type": sign_name,
                                    "C_before": float(base_eval["cost_term"]),
                                    "C_after": float(step_eval["cost_term"]),
                                    "norm_before": float(base_eval["norm_term"]),
                                    "norm_after": float(step_eval["norm_term"]),
                                    "total_merit_before": float(base_eval["total_merit"]),
                                    "total_merit_after": float(step_eval["total_merit"]),
                                    "C_change": float(step_eval["cost_term"] - base_eval["cost_term"]),
                                    "norm_change": float(step_eval["norm_term"] - base_eval["norm_term"]),
                                    "total_merit_change": float(step_eval["total_merit"] - base_eval["total_merit"]),
                                }
                            )

                    # EGM geometry
                    theta_half = {name: tensor.clone() for name, tensor in theta_old.items()}
                    for name in selected_names:
                        theta_half[name] = theta_old[name] - args.eta_egm * F_raw[name]
                    half_eval = train_eval_state(theta_half)
                    delta_egm_actual = flatten_named_tensors(
                        {name: -args.eta_egm * half_eval["grads_selected"][name] for name in selected_names},
                        selected_names,
                    )
                    delta_egm_exp = flatten_named_tensors(
                        {name: -args.eta_egm * F_raw[name] + (args.eta_egm ** 2) * current_G[name] for name in selected_names},
                        selected_names,
                    )

                    for g_variant_name, g_map in [
                        ("forward_plus", G_plus),
                        ("forward_minus_to_egm", G_egm),
                        ("central", G_central),
                    ]:
                        g_variant_vec = flatten_named_tensors(g_map, selected_names)
                        for sign_label, sign_mul in [("plusG", 1.0), ("minusG", -1.0)]:
                            cand = apply_candidate(
                                theta_old=theta_old,
                                selected_names=selected_names,
                                F_raw=F_raw,
                                G_raw={name: sign_mul * g_map[name] for name in selected_names},
                                beta=beta_qp,
                                gamma=gamma_qp,
                                update_cap=config.update_cap,
                                eval_state_fn=train_eval_state,
                                total_coeffs=total_coeffs,
                                norm_coeffs=norm_coeffs,
                                cost_coeffs=cost_coeffs,
                                candidate_type=f"{g_variant_name}_{sign_label}",
                            )
                            delta_qp = flatten_named_tensors(
                                {name: -beta_qp * F_raw[name] + sign_mul * gamma_qp * g_map[name] for name in selected_names},
                                selected_names,
                            )
                            gsign_rows.append(
                                {
                                    "config_id": config.config_id,
                                    "config_label": config.label,
                                    "scope": config.scope,
                                    "role": role,
                                    "train_probe_id": int(train_probe.probe_idx),
                                    "g_variant": g_variant_name,
                                    "update_sign": sign_label,
                                    "q_pred_total_merit": cand.q_pred_total_merit,
                                    "actual_total_merit_change": cand.actual_total_merit_change,
                                    "actual_norm_term_change": cand.actual_norm_term_change,
                                    "actual_cost_change": cand.actual_cost_change,
                                    "approx_kl": cand.approx_kl,
                                    "clip_fraction": cand.clip_fraction,
                                    "cosine_g_variant_to_current": float(th.dot(g_variant_vec, G_vec) / max((th.norm(g_variant_vec) * th.norm(G_vec)).item(), EPS)) if g_variant_vec.numel() and G_vec.numel() else 0.0,
                                    "cosine_delta_to_egm_actual": float(th.dot(delta_qp, delta_egm_actual) / max((th.norm(delta_qp) * th.norm(delta_egm_actual)).item(), EPS)) if delta_qp.numel() and delta_egm_actual.numel() else 0.0,
                                    "relative_error_to_egm_actual": relative_error(delta_qp, delta_egm_actual),
                                    "cosine_delta_to_egm_expansion": float(th.dot(delta_qp, delta_egm_exp) / max((th.norm(delta_qp) * th.norm(delta_egm_exp)).item(), EPS)) if delta_qp.numel() and delta_egm_exp.numel() else 0.0,
                                }
                            )

                    # Cap consistency on QP selected
                    cap_rows.append(
                        {
                            "config_id": config.config_id,
                            "config_label": config.label,
                            "scope": config.scope,
                            "role": role,
                            "train_probe_id": int(train_probe.probe_idx),
                            "pre_cap_update_norm": qp_cand.update_norm_pre_cap,
                            "post_cap_update_norm": qp_cand.update_norm_post_cap,
                            "cap_scale": qp_cand.cap_scale,
                            "cap_active": qp_cand.cap_active,
                            "q_pred_pre_cap": qp_cand.q_pred_total_merit,
                            "q_pred_post_cap_estimate": q_value(beta_qp * qp_cand.cap_scale, gamma_qp * qp_cand.cap_scale, total_coeffs),
                            "actual_change_post_cap": qp_cand.actual_total_merit_change,
                        }
                    )

                    # gamma sweep for top failed configs
                    if config.config_id in top_failed_ids:
                        for beta_ref_name, beta_fixed in [("beta_noG", beta_nog), ("beta_QP", beta_qp), ("eta_EGM", args.eta_egm)]:
                            for gamma_val in np.linspace(-config.gamma_max, config.gamma_max, 101):
                                cand = apply_candidate(
                                    theta_old=theta_old,
                                    selected_names=selected_names,
                                    F_raw=F_raw,
                                    G_raw=current_G,
                                    beta=float(beta_fixed),
                                    gamma=float(gamma_val),
                                    update_cap=config.update_cap,
                                    eval_state_fn=train_eval_state,
                                    total_coeffs=total_coeffs,
                                    norm_coeffs=norm_coeffs,
                                    cost_coeffs=cost_coeffs,
                                    candidate_type="gamma_sweep",
                                )
                                gamma_rows.append(
                                    {
                                        "config_id": config.config_id,
                                        "config_label": config.label,
                                        "scope": config.scope,
                                        "role": role,
                                        "train_probe_id": int(train_probe.probe_idx),
                                        "beta_reference": beta_ref_name,
                                        "beta_fixed": float(beta_fixed),
                                        "gamma": float(gamma_val),
                                        "actual_total_merit_change": cand.actual_total_merit_change,
                                        "actual_norm_term_change": cand.actual_norm_term_change,
                                        "actual_cost_change": cand.actual_cost_change,
                                        "approx_kl": cand.approx_kl,
                                        "clip_fraction": cand.clip_fraction,
                                        "update_norm_post_cap": cand.update_norm_post_cap,
                                    }
                                )

                    restore_state(named_params, theta_old)

        inv_df = pd.DataFrame(invariant_rows)
        if "_dummy" in inv_df.columns:
            inv_df = inv_df.drop(columns=["_dummy"])
        inv_df.to_csv(output_root / "stage45_invariant_audit.csv", index=False)

        sign_df = pd.DataFrame(sign_rows)
        sign_df.to_csv(output_root / "stage45_active_cost_sign_audit.csv", index=False)

        g_df = pd.DataFrame(gsign_rows)
        g_df.to_csv(output_root / "stage45_G_sign_audit.csv", index=False)

        gamma_df = pd.DataFrame(gamma_rows)
        gamma_df.to_csv(output_root / "stage45_gamma_sweep.csv", index=False)

        cap_df = pd.DataFrame(cap_rows)
        cap_df.to_csv(output_root / "stage45_cap_consistency.csv", index=False)

        # Reports
        noG_df = inv_df[inv_df["candidate_type"] == "noG_selected"].copy()
        qp_df = inv_df[inv_df["candidate_type"] == "QP_selected"].copy()
        grid_df = inv_df[inv_df["candidate_type"] == "best_dense_grid"].copy()
        merge_pred = qp_df.merge(
            noG_df[["config_id", "role", "train_probe_id", "q_pred_total_merit", "actual_total_merit_change"]],
            on=["config_id", "role", "train_probe_id"],
            suffixes=("_qp", "_nog"),
        ).merge(
            grid_df[["config_id", "role", "train_probe_id", "q_pred_total_merit"]],
            on=["config_id", "role", "train_probe_id"],
            suffixes=("", "_grid"),
        )
        merge_pred["predicted_invariant_fail"] = merge_pred["q_pred_total_merit_qp"] > merge_pred["q_pred_total_merit_nog"] + merge_pred["q_pred_total_merit_nog"].abs().map(tol)
        merge_pred["dense_grid_predicted_fail"] = merge_pred["q_pred_total_merit"] > merge_pred["q_pred_total_merit_nog"] + merge_pred["q_pred_total_merit_nog"].abs().map(tol)
        merge_pred["actual_fail"] = merge_pred["actual_total_merit_change_qp"] > merge_pred["actual_total_merit_change_nog"] + merge_pred["actual_total_merit_change_nog"].abs().map(tol)

        inv_lines = [
            "# Stage 4.5A Invariant Audit Report",
            "",
            f"- Rows audited: `{len(inv_df)}`",
            f"- Predicted invariant failures (QP worse than noG on q_pred_total_merit): `{int(merge_pred['predicted_invariant_fail'].sum())}/{len(merge_pred)}`",
            f"- Dense-grid predicted failures vs noG: `{int(merge_pred['dense_grid_predicted_fail'].sum())}/{len(merge_pred)}`",
            f"- Actual composite-merit failures (QP worse than noG): `{int(merge_pred['actual_fail'].sum())}/{len(merge_pred)}`",
            "",
            "Interpretation:",
            "- If predicted invariant fails, the solver/candidate setup is wrong because noG should be a feasible subset of QP.",
            "- If predicted passes but actual fails, the drift model / cap / finite-difference approximation is the problem.",
        ]
        (output_root / "stage45_invariant_audit_report.md").write_text("\n".join(inv_lines), encoding="utf-8")

        sign_pivot = sign_df.pivot_table(index=["config_id", "role", "train_probe_id", "eps"], columns="step_type", values="C_change", aggfunc="mean").reset_index()
        sign_pivot["plus_better_than_minus"] = sign_pivot["plus_F"] < sign_pivot["minus_F"]
        sign_lines = [
            "# Stage 4.5B Active Cost Sign Report",
            "",
            f"- Cases where `+F` changed active cost more favorably than `-F`: `{int(sign_pivot['plus_better_than_minus'].sum())}/{len(sign_pivot)}`",
            "- If this count is large, the sign of `F` or the sign of the active cost `C_b` is likely wrong.",
        ]
        (output_root / "stage45_active_cost_sign_report.md").write_text("\n".join(sign_lines), encoding="utf-8")

        g_group = g_df.groupby(["g_variant", "update_sign"]).agg(
            q_pred_total_merit_mean=("q_pred_total_merit", "mean"),
            actual_total_merit_change_mean=("actual_total_merit_change", "mean"),
            actual_cost_change_mean=("actual_cost_change", "mean"),
            cosine_delta_to_egm_actual_mean=("cosine_delta_to_egm_actual", "mean"),
            relative_error_to_egm_actual_mean=("relative_error_to_egm_actual", "mean"),
        ).reset_index()
        g_lines = [
            "# Stage 4.5C G Sign Report",
            "",
            f"- Current code G is based on forward_plus finite difference: `(F(z + eps F) - F(z)) / eps`, i.e. `+J_F F` under the chosen sign convention.",
            f"- Best actual total-merit mean among G/sign variants: `{g_group.sort_values('actual_total_merit_change_mean').iloc[0]['g_variant']}_{g_group.sort_values('actual_total_merit_change_mean').iloc[0]['update_sign']}`",
            f"- Worst actual total-merit mean among G/sign variants: `{g_group.sort_values('actual_total_merit_change_mean').iloc[-1]['g_variant']}_{g_group.sort_values('actual_total_merit_change_mean').iloc[-1]['update_sign']}`",
            "",
            "Questions answered by aggregate evidence:",
            f"- Does `+gamma G` help or hurt? compare means in `stage45_G_sign_audit.csv`.",
            f"- Does `-gamma G` help more than `+gamma G`? compare `update_sign=minusG` vs `plusG`.",
            f"- Is QP closer to EGM actual update or opposite sign? see `cosine_delta_to_egm_actual_mean` and `relative_error_to_egm_actual_mean`.",
        ]
        (output_root / "stage45_G_sign_report.md").write_text("\n".join(g_lines), encoding="utf-8")

        gamma_best = gamma_df.loc[gamma_df.groupby(["config_id", "role", "beta_reference"])["actual_total_merit_change"].idxmin()].copy()
        gamma_best["best_gamma_sign"] = np.sign(gamma_best["gamma"])
        gamma_lines = [
            "# Stage 4.5D Gamma Sweep Report",
            "",
            f"- Swept top failed configs: `{sorted(top_failed_ids)}`",
            f"- Best gamma < 0 count: `{int((gamma_best['gamma'] < -EPS).sum())}`",
            f"- Best gamma ~= 0 count: `{int((gamma_best['gamma'].abs() <= 1e-9).sum())}`",
            f"- Best gamma > 0 count: `{int((gamma_best['gamma'] > EPS).sum())}`",
            "",
            "Interpretation:",
            "- best gamma < 0 suggests the current G sign is wrong for actual merit.",
            "- best gamma = 0 suggests G is not useful under the current merit/scope.",
            "- predicted best gamma > 0 but actual best gamma <= 0 suggests the drift fit is wrong.",
        ]
        (output_root / "stage45_gamma_sweep_report.md").write_text("\n".join(gamma_lines), encoding="utf-8")

        cap_lines = [
            "# Stage 4.5E Cap Consistency Report",
            "",
            f"- Capped QP candidates: `{int(cap_df['cap_active'].sum())}/{len(cap_df)}`",
            f"- Mean pre-cap update norm: `{cap_df['pre_cap_update_norm'].mean():.6e}`",
            f"- Mean post-cap update norm: `{cap_df['post_cap_update_norm'].mean():.6e}`",
            f"- Mean q_pred_pre_cap: `{cap_df['q_pred_pre_cap'].mean():.6e}`",
            f"- Mean q_pred_post_cap_estimate: `{cap_df['q_pred_post_cap_estimate'].mean():.6e}`",
            f"- Mean actual_change_post_cap: `{cap_df['actual_change_post_cap'].mean():.6e}`",
        ]
        (output_root / "stage45_cap_consistency_report.md").write_text("\n".join(cap_lines), encoding="utf-8")

        # Root cause classification
        root_causes: List[str] = []
        if int(merge_pred["predicted_invariant_fail"].sum()) > 0 or int(merge_pred["dense_grid_predicted_fail"].sum()) > 0:
            root_causes.append("C. QP solver does not include noG feasible candidate")
        if int(sign_pivot["plus_better_than_minus"].sum()) > len(sign_pivot) * 0.25:
            root_causes.append("B. active cost sign wrong")
        minus_group = g_group[g_group["update_sign"] == "minusG"]["actual_total_merit_change_mean"].mean() if not g_group[g_group["update_sign"] == "minusG"].empty else np.nan
        plus_group = g_group[g_group["update_sign"] == "plusG"]["actual_total_merit_change_mean"].mean() if not g_group[g_group["update_sign"] == "plusG"].empty else np.nan
        if np.isfinite(minus_group) and np.isfinite(plus_group) and minus_group < plus_group:
            root_causes.append("A. G sign wrong")
        if int(merge_pred["predicted_invariant_fail"].sum()) == 0 and int(merge_pred["actual_fail"].sum()) > 0:
            root_causes.append("D. predicted q beats noG but actual drift does not, so finite-difference/drift model wrong")
        if int(cap_df["cap_active"].sum()) > 0 and abs(cap_df["q_pred_post_cap_estimate"].mean() - cap_df["actual_change_post_cap"].mean()) > abs(cap_df["q_pred_pre_cap"].mean() - cap_df["actual_change_post_cap"].mean()):
            root_causes.append("E. update cap invalidates q prediction")
        if int((gamma_best["gamma"].abs() <= 1e-9).sum()) > len(gamma_best) * 0.5:
            root_causes.append("F. gamma direction genuinely harmful for current merit")
        if not root_causes:
            root_causes.append("I. performance term scaling wrong")

        root_lines = [
            "# Stage 4.5 Root Cause Report",
            "",
            "## Main Findings",
            *(f"- {cause}" for cause in root_causes),
            "",
            "## Evidence",
            f"- Predicted invariant failures: `{int(merge_pred['predicted_invariant_fail'].sum())}/{len(merge_pred)}`",
            f"- Dense-grid invariant failures: `{int(merge_pred['dense_grid_predicted_fail'].sum())}/{len(merge_pred)}`",
            f"- Actual QP-vs-noG failures: `{int(merge_pred['actual_fail'].sum())}/{len(merge_pred)}`",
            f"- `+F` better than `-F` count in active cost sign audit: `{int(sign_pivot['plus_better_than_minus'].sum())}/{len(sign_pivot)}`",
            f"- mean actual merit for `+gamma G`: `{plus_group:.6e}`",
            f"- mean actual merit for `-gamma G`: `{minus_group:.6e}`",
            f"- capped QP candidates: `{int(cap_df['cap_active'].sum())}/{len(cap_df)}`",
            "",
            "No online training was started after this audit.",
        ]
        (output_root / "stage45_root_cause_report.md").write_text("\n".join(root_lines), encoding="utf-8")

        # Plots
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(merge_pred["q_pred_total_merit_nog"], merge_pred["actual_total_merit_change_nog"], label="noG", alpha=0.7)
        ax.scatter(merge_pred["q_pred_total_merit_qp"], merge_pred["actual_total_merit_change_qp"], label="QP", alpha=0.7)
        ax.set_xlabel("Predicted total merit change")
        ax.set_ylabel("Actual total merit change")
        ax.set_title("QP vs noG predicted vs actual")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "stage45_qp_vs_nog_predicted_actual.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(grid_df["q_pred_total_merit"], grid_df["actual_total_merit_change"], label="best_dense_grid", alpha=0.7)
        ax.scatter(noG_df["q_pred_total_merit"], noG_df["actual_total_merit_change"], label="noG_selected", alpha=0.7)
        ax.set_xlabel("Predicted total merit change")
        ax.set_ylabel("Actual total merit change")
        ax.set_title("Dense-grid frontier vs noG")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "stage45_dense_grid_frontier.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        for (g_variant, update_sign), sub in g_df.groupby(["g_variant", "update_sign"]):
            ax.scatter(sub["cosine_g_variant_to_current"], sub["actual_total_merit_change"], label=f"{g_variant}_{update_sign}", alpha=0.5)
        ax.set_xlabel("cosine(G_variant, G_current_code)")
        ax.set_ylabel("Actual total merit change")
        ax.set_title("G sign comparison")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / "stage45_G_sign_comparison.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(g_df["cosine_delta_to_egm_actual"], g_df["relative_error_to_egm_actual"], alpha=0.6)
        ax.set_xlabel("cosine(delta_QP, delta_EGM_actual)")
        ax.set_ylabel("relative error to EGM actual")
        ax.set_title("QP vs EGM geometry")
        fig.tight_layout()
        fig.savefig(plots_dir / "stage45_QP_vs_EGM_geometry.png", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        for beta_ref_name, sub in gamma_df.groupby("beta_reference"):
            sub_mean = sub.groupby("gamma").agg(
                total_merit=("actual_total_merit_change", "mean"),
                norm_term=("actual_norm_term_change", "mean"),
                cost=("actual_cost_change", "mean"),
            ).reset_index()
            axes[0].plot(sub_mean["gamma"], sub_mean["total_merit"], label=beta_ref_name)
            axes[1].plot(sub_mean["gamma"], sub_mean["norm_term"], label=beta_ref_name)
            axes[2].plot(sub_mean["gamma"], sub_mean["cost"], label=beta_ref_name)
        axes[0].set_ylabel("total merit change")
        axes[1].set_ylabel("norm term change")
        axes[2].set_ylabel("cost change")
        axes[2].set_xlabel("gamma")
        axes[0].legend()
        axes[0].set_title("Gamma sweep actual curves")
        fig.tight_layout()
        fig.savefig(plots_dir / "stage45_gamma_sweep_actual.png", dpi=180)
        plt.close(fig)
    finally:
        del model


if __name__ == "__main__":
    main()
