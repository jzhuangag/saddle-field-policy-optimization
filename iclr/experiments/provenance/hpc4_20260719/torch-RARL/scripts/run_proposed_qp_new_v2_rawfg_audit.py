from __future__ import annotations

import argparse
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from models.proposed_qp_new import block_norm
from scripts.full_policy_optimizer_probe import (
    clone_state,
    collect_probe_batches,
    compute_loss_and_grads,
    named_parameters,
    restore_state,
    temporary_algo_overrides,
    tensor_norm,
)
from utils.exp_manager import ExperimentManager


NamedTensorMap = Dict[str, th.Tensor]


@dataclass(frozen=True)
class BoundConfig:
    label: str
    beta_max: float
    gamma_max: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Raw F/G Lyapunov-QP diagnostic audit")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--role", type=str, default="protagonist", choices=["protagonist", "adversary"])
    parser.add_argument("--num-probes", type=int, default=4)
    parser.add_argument("--eta-egm", type=float, default=1e-3)
    parser.add_argument("--external-eta", type=float, default=1.0)
    parser.add_argument("--fd-eps", type=float, default=1e-3)
    parser.add_argument("--beta-probe", type=float, default=1e-3)
    parser.add_argument("--gamma-probe", type=float, default=1e-6)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--actor-weight", type=float, default=1.0)
    parser.add_argument("--logstd-weight", type=float, default=1.0)
    parser.add_argument("--critic-weight", type=float, default=0.3)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ppm-inner-steps", type=int, default=10)
    parser.add_argument("--single-beta-max", type=float, default=None)
    parser.add_argument("--single-gamma-max", type=float, default=None)
    parser.add_argument("--update-cap", type=float, default=float("inf"))
    return parser.parse_args()


def build_control_manager(args: argparse.Namespace) -> ExperimentManager:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    results_root = repo_root.parent / "results" / "_rawfg_m9_probe_tmp"
    return ExperimentManager(
        args=argparse.Namespace(),
        algo="rarl",
        rarl_config="ppo",
        env_id=args.env,
        log_folder=str(results_root / "logging"),
        tensorboard_log=str(results_root / "tb"),
        n_timesteps=1,
        eval_freq=-1,
        n_eval_episodes=1,
        save_freq=-1,
        hyperparameter_path=str(repo_root / "hyperparameter"),
        hyperparams=None,
        env_kwargs=None,
        model_path=str(results_root / "saved_models"),
        pretrained_model="",
        optimize_hyperparameters=False,
        storage=None,
        study_name=None,
        n_opt_trials=1,
        n_jobs=1,
        sampler="tpe",
        pruner="median",
        optimization_log_path=str(results_root / "opt"),
        n_startup_trials=0,
        n_evaluations_opt=1,
        seed=args.seed,
        log_interval=-1,
        save_replay_buffer=False,
        verbose=0,
        vec_env_type="dummy",
        n_envs=1,
        n_eval_envs=1,
        no_optim_plots=True,
        adv_env=False,
        adv_impact="control",
        adv_fraction=2.5,
        adv_delay=-1,
        adv_index_list=["torso"],
        adv_force_dim=2,
        N_mu=-1,
        N_nu=-1,
        device=args.device,
        protagonist_optimizer="adam",
        adversary_optimizer="adam",
    )


def classify_block(name: str) -> str:
    if "log_std" in name:
        return "logstd"
    if "value" in name:
        return "critic"
    return "actor"


def flatten_named_tensors(named_map: NamedTensorMap, selected_names: Sequence[str]) -> th.Tensor:
    pieces = [named_map[name].reshape(-1) for name in selected_names]
    if not pieces:
        return th.zeros(0)
    return th.cat(pieces)


def state_vector(theta_state: NamedTensorMap, selected_names: Sequence[str]) -> th.Tensor:
    return flatten_named_tensors(theta_state, selected_names)


def named_add_scaled(theta_old: NamedTensorMap, direction: NamedTensorMap, scale: float, selected_names: Sequence[str]) -> NamedTensorMap:
    theta_new = {name: tensor.clone() for name, tensor in theta_old.items()}
    for name in selected_names:
        theta_new[name] = theta_old[name] + scale * direction[name]
    return theta_new


def apply_two_direction_delta(
    theta_old: NamedTensorMap,
    f_map: NamedTensorMap,
    g_map: NamedTensorMap,
    beta: float,
    gamma: float,
    selected_names: Sequence[str],
    eta: float = 1.0,
) -> NamedTensorMap:
    theta_new = {name: tensor.clone() for name, tensor in theta_old.items()}
    for name in selected_names:
        theta_new[name] = theta_old[name] + eta * (-beta * f_map[name] + gamma * g_map[name])
    return theta_new


def apply_update_cap(
    theta_old: NamedTensorMap,
    theta_new: NamedTensorMap,
    selected_names: Sequence[str],
    cap: float,
) -> tuple[NamedTensorMap, float, float, int]:
    delta_map = {name: theta_new[name] - theta_old[name] for name in selected_names}
    delta_vec = flatten_named_tensors(delta_map, selected_names)
    update_norm_pre = tensor_norm(delta_vec)
    if (not math.isfinite(cap)) or cap <= 0.0 or update_norm_pre <= cap:
        return theta_new, update_norm_pre, update_norm_pre, 0
    scale = cap / max(update_norm_pre, 1e-12)
    theta_capped = {name: tensor.clone() for name, tensor in theta_old.items()}
    for name in selected_names:
        theta_capped[name] = theta_old[name] + scale * delta_map[name]
    update_norm_post = tensor_norm(flatten_named_tensors({name: theta_capped[name] - theta_old[name] for name in selected_names}, selected_names))
    return theta_capped, update_norm_pre, update_norm_post, 1


def safe_div(numerator: float, denominator: float, eps: float = 1e-12) -> Tuple[float, bool]:
    denom_small = abs(denominator) < eps
    denom = denominator if not denom_small else (eps if denominator >= 0 else -eps)
    return float(numerator / denom), bool(denom_small)


def cosine_similarity(a: th.Tensor, b: th.Tensor, eps: float = 1e-12) -> Tuple[float, bool]:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0, True
    denom = float((th.norm(a) * th.norm(b)).item())
    if not math.isfinite(denom) or abs(denom) < eps:
        return 0.0, True
    return float(th.dot(a, b).item() / denom), False


def relative_error(a: th.Tensor, b: th.Tensor, eps: float = 1e-12) -> Tuple[float, bool]:
    denom = max(tensor_norm(b), eps)
    return tensor_norm(a - b) / denom, bool(tensor_norm(b) < eps)


def lyapunov_value(grads: NamedTensorMap, *, actor_weight: float, logstd_weight: float, critic_weight: float, selected_names: Sequence[str]) -> float:
    def block_mean_square(block: str) -> float:
        pieces = [grads[name].reshape(-1) for name in selected_names if classify_block(name) == block]
        if not pieces:
            return 0.0
        vec = th.cat(pieces)
        return float(th.mean(vec * vec).item())

    return 0.5 * (
        actor_weight * block_mean_square("actor")
        + logstd_weight * block_mean_square("logstd")
        + critic_weight * block_mean_square("critic")
    )


def evaluate_state(
    *,
    algo,
    rollout_data,
    named_params,
    theta_state: NamedTensorMap,
    max_grad_norm: float,
    vf_coef: float,
    ent_coef: float,
    actor_weight: float,
    logstd_weight: float,
    critic_weight: float,
    selected_names: Sequence[str],
) -> Dict[str, object]:
    restore_state(named_params, theta_state)
    eval_info = compute_loss_and_grads(
        algo,
        rollout_data,
        max_grad_norm=max_grad_norm,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
    )
    v_value = lyapunov_value(
        eval_info["grads"],
        actor_weight=actor_weight,
        logstd_weight=logstd_weight,
        critic_weight=critic_weight,
        selected_names=selected_names,
    )
    eval_info["V"] = v_value
    return eval_info


def finite_difference_g_raw(
    *,
    algo,
    rollout_data,
    named_params,
    theta_old: NamedTensorMap,
    F_raw: NamedTensorMap,
    fd_eps: float,
    max_grad_norm: float,
    vf_coef: float,
    ent_coef: float,
    actor_weight: float,
    logstd_weight: float,
    critic_weight: float,
    selected_names: Sequence[str],
) -> Tuple[NamedTensorMap, bool, Dict[str, object]]:
    theta_plus = named_add_scaled(theta_old, F_raw, fd_eps, selected_names)
    plus_eval = evaluate_state(
        algo=algo,
        rollout_data=rollout_data,
        named_params=named_params,
        theta_state=theta_plus,
        max_grad_norm=max_grad_norm,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        actor_weight=actor_weight,
        logstd_weight=logstd_weight,
        critic_weight=critic_weight,
        selected_names=selected_names,
    )
    restore_state(named_params, theta_old)
    g_raw = {name: (plus_eval["grads"][name] - F_raw[name]) / fd_eps for name in selected_names}
    g_vec = flatten_named_tensors(g_raw, selected_names)
    valid = bool(th.isfinite(g_vec).all().item()) and tensor_norm(g_vec) > 1e-12
    return g_raw, valid, plus_eval


def estimate_quadratic_coefficients_raw(
    *,
    algo,
    rollout_data,
    named_params,
    theta_old: NamedTensorMap,
    p_map: NamedTensorMap,
    r_map: NamedTensorMap,
    v0: float,
    beta_probe: float,
    gamma_probe: float,
    ridge: float,
    max_grad_norm: float,
    vf_coef: float,
    ent_coef: float,
    actor_weight: float,
    logstd_weight: float,
    critic_weight: float,
    selected_names: Sequence[str],
) -> Dict[str, float | str]:
    def V_at(beta_scale: float, gamma_scale: float) -> float:
        theta_tmp = {name: tensor.clone() for name, tensor in theta_old.items()}
        for name in selected_names:
            theta_tmp[name] = theta_old[name] + beta_scale * p_map[name] + gamma_scale * r_map[name]
        eval_info = evaluate_state(
            algo=algo,
            rollout_data=rollout_data,
            named_params=named_params,
            theta_state=theta_tmp,
            max_grad_norm=max_grad_norm,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            actor_weight=actor_weight,
            logstd_weight=logstd_weight,
            critic_weight=critic_weight,
            selected_names=selected_names,
        )
        return float(eval_info["V"])

    db = beta_probe
    dg = gamma_probe
    vp_plus = V_at(db, 0.0)
    vp_minus = V_at(-db, 0.0)
    vr_plus = V_at(0.0, dg)
    vr_minus = V_at(0.0, -dg)
    vpp = V_at(db, dg)
    vpm = V_at(db, -dg)
    vmp = V_at(-db, dg)
    vmm = V_at(-db, -dg)

    a = (vp_plus - vp_minus) / (2.0 * db)
    c = (vp_plus - 2.0 * v0 + vp_minus) / (db * db)
    b = (vr_plus - vr_minus) / (2.0 * dg)
    k = (vr_plus - 2.0 * v0 + vr_minus) / (dg * dg)
    h = (vpp - vpm - vmp + vmm) / (4.0 * db * dg)

    ridge_used = 0.0
    det = c * k - h * h
    condition = "pd"
    if (not math.isfinite(c)) or (not math.isfinite(k)) or (not math.isfinite(h)):
        condition = "invalid"
    elif c <= 0.0 or k <= 0.0 or det <= 0.0:
        c += ridge
        k += ridge
        ridge_used = ridge
        det = c * k - h * h
        condition = "ridge_pd" if c > 0.0 and k > 0.0 and det > 0.0 else "ridge_failed"
    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "h": float(h),
        "k": float(k),
        "H_det": float(det),
        "ridge_used": float(ridge_used),
        "q_condition_status": condition,
        "db": float(db),
        "dg": float(dg),
    }


def q_value(beta: float, gamma: float, a: float, b: float, c: float, h: float, k: float) -> float:
    return a * beta + b * gamma + 0.5 * c * beta * beta + h * beta * gamma + 0.5 * k * gamma * gamma


def solve_no_g(a: float, c: float, beta_max: float, eps: float) -> Dict[str, float | str]:
    beta_star = 0.0
    if c > eps and math.isfinite(c):
        beta_star = -a / c
        candidates = [("interior" if 0.0 <= beta_star <= beta_max else "beta_bound", min(max(beta_star, 0.0), beta_max))]
    else:
        candidates = [("zero", 0.0), ("beta_max", beta_max)]
    best = ("zero", 0.0, float("inf"))
    for label, beta in candidates:
        val = q_value(beta, 0.0, a, 0.0, c if math.isfinite(c) else 1.0, 0.0, 1.0)
        if math.isfinite(val) and val < best[2]:
            best = (label, beta, val)
    if not math.isfinite(best[2]):
        best = ("zero_fallback", 0.0, 0.0)
    return {
        "beta_star_unclipped": float(beta_star),
        "gamma_star_unclipped": 0.0,
        "beta": float(best[1]),
        "gamma": 0.0,
        "selected_case": best[0],
        "beta_at_bound": int(abs(best[1] - beta_max) <= eps and beta_max > 0.0),
        "gamma_at_bound": 0,
        "gamma_active": 0,
        "q_pred": float(best[2]),
    }


def solve_two_direction_qp(a: float, b: float, c: float, h: float, k: float, beta_max: float, gamma_max: float, det: float, condition: str, eps: float) -> Dict[str, float | str]:
    candidates: List[Tuple[str, float, float]] = [
        ("corner_00", 0.0, 0.0),
        ("corner_b0", beta_max, 0.0),
        ("corner_0g", 0.0, gamma_max),
        ("corner_bg", beta_max, gamma_max),
    ]
    beta_star = 0.0
    gamma_star = 0.0
    interior_valid = False
    if condition in {"pd", "ridge_pd"} and det > eps:
        beta_star = (-a * k + h * b) / det
        gamma_star = (-c * b + h * a) / det
        if math.isfinite(beta_star) and math.isfinite(gamma_star) and 0.0 <= beta_star <= beta_max and 0.0 <= gamma_star <= gamma_max:
            candidates.append(("interior", beta_star, gamma_star))
            interior_valid = True
    if c > eps:
        candidates.append(("gamma0", min(max(-a / c, 0.0), beta_max), 0.0))
        candidates.append(("gamma_max", min(max(-(a + h * gamma_max) / c, 0.0), beta_max), gamma_max))
    if k > eps:
        candidates.append(("beta0", 0.0, min(max(-b / k, 0.0), gamma_max)))
        candidates.append(("beta_max", beta_max, min(max(-(b + h * beta_max) / k, 0.0), gamma_max)))

    best_label = "corner_00"
    best_beta = 0.0
    best_gamma = 0.0
    best_q = float("inf")
    for label, beta, gamma in candidates:
        q_pred = q_value(beta, gamma, a, b, c, h, k)
        if math.isfinite(q_pred) and q_pred < best_q:
            best_label = label
            best_beta = beta
            best_gamma = gamma
            best_q = q_pred
    if not math.isfinite(best_q):
        best_label = "finite_fallback_corner_00"
        best_beta = 0.0
        best_gamma = 0.0
        best_q = 0.0
    return {
        "beta_star_unclipped": float(beta_star),
        "gamma_star_unclipped": float(gamma_star),
        "beta": float(best_beta),
        "gamma": float(best_gamma),
        "selected_case": best_label,
        "beta_at_bound": int(abs(best_beta - beta_max) <= eps and beta_max > 0.0),
        "gamma_at_bound": int(abs(best_gamma - gamma_max) <= eps and gamma_max > 0.0),
        "gamma_active": int(best_gamma > eps),
        "interior_solution_valid": int(interior_valid),
        "q_pred": float(best_q),
    }


def build_bound_configs(eta_egm: float, single_beta_max: float | None = None, single_gamma_max: float | None = None) -> List[BoundConfig]:
    if single_beta_max is not None and single_gamma_max is not None:
        return [BoundConfig(f"target_b{single_beta_max:.0e}_g{single_gamma_max:.0e}", float(single_beta_max), float(single_gamma_max))]
    beta_values = [eta_egm, 3.0 * eta_egm, 10.0 * eta_egm, 30.0 * eta_egm]
    gamma_values = [eta_egm ** 2, 3.0 * eta_egm ** 2, 10.0 * eta_egm ** 2, 30.0 * eta_egm ** 2, 100.0 * eta_egm ** 2]
    configs = [
        BoundConfig(f"b{beta:.0e}_g{gamma:.0e}", beta, gamma)
        for beta in beta_values
        for gamma in gamma_values
    ]
    configs.append(BoundConfig("wide_diag", 1e-1, 1e-3))
    return configs


def main() -> None:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    manager = build_control_manager(args)
    rarl_model = manager.setup_experiment()
    algo = rarl_model.protagonist if args.role == "protagonist" else rarl_model.adversary
    probes = collect_probe_batches(rarl_model, args.role, args.num_probes)
    named_params = named_parameters(algo.policy)
    selected_names = [name for name, _ in named_params]
    ent_coef = float(algo.ent_coef)
    bound_configs = build_bound_configs(args.eta_egm, args.single_beta_max, args.single_gamma_max)

    rows: List[Dict[str, object]] = []

    for probe in probes:
        theta_old = clone_state(named_params)
        with temporary_algo_overrides(algo, vf_coef=args.vf_coef, ent_coef=ent_coef, max_grad_norm=args.max_grad_norm):
            base_eval = evaluate_state(
                algo=algo,
                rollout_data=probe.rollout_data,
                named_params=named_params,
                theta_state=theta_old,
                max_grad_norm=args.max_grad_norm,
                vf_coef=args.vf_coef,
                ent_coef=ent_coef,
                actor_weight=args.actor_weight,
                logstd_weight=args.logstd_weight,
                critic_weight=args.critic_weight,
                selected_names=selected_names,
            )
            F_raw: NamedTensorMap = {name: base_eval["grads"][name].clone() for name in selected_names}
            F_vec = flatten_named_tensors(F_raw, selected_names)
            F_norm = tensor_norm(F_vec)

            G_raw, fd_valid, plus_eval = finite_difference_g_raw(
                algo=algo,
                rollout_data=probe.rollout_data,
                named_params=named_params,
                theta_old=theta_old,
                F_raw=F_raw,
                fd_eps=args.fd_eps,
                max_grad_norm=args.max_grad_norm,
                vf_coef=args.vf_coef,
                ent_coef=ent_coef,
                actor_weight=args.actor_weight,
                logstd_weight=args.logstd_weight,
                critic_weight=args.critic_weight,
                selected_names=selected_names,
            )
            G_vec = flatten_named_tensors(G_raw, selected_names)
            G_norm = tensor_norm(G_vec)

            theta_half = apply_two_direction_delta(theta_old, F_raw, {name: th.zeros_like(F_raw[name]) for name in selected_names}, args.eta_egm, 0.0, selected_names)
            half_eval = evaluate_state(
                algo=algo,
                rollout_data=probe.rollout_data,
                named_params=named_params,
                theta_state=theta_half,
                max_grad_norm=args.max_grad_norm,
                vf_coef=args.vf_coef,
                ent_coef=ent_coef,
                actor_weight=args.actor_weight,
                logstd_weight=args.logstd_weight,
                critic_weight=args.critic_weight,
                selected_names=selected_names,
            )
            F_half = {name: half_eval["grads"][name].clone() for name in selected_names}
            delta_egm_map = {name: -args.eta_egm * F_half[name] for name in selected_names}
            theta_egm = {name: theta_old[name] + delta_egm_map[name] for name in selected_names}
            egm_after = evaluate_state(
                algo=algo,
                rollout_data=probe.rollout_data,
                named_params=named_params,
                theta_state=theta_egm,
                max_grad_norm=args.max_grad_norm,
                vf_coef=args.vf_coef,
                ent_coef=ent_coef,
                actor_weight=args.actor_weight,
                logstd_weight=args.logstd_weight,
                critic_weight=args.critic_weight,
                selected_names=selected_names,
            )

            delta_egm_exp_map = {name: (-args.eta_egm * F_raw[name]) + (args.eta_egm ** 2) * G_raw[name] for name in selected_names}
            theta_egm_exp = {name: theta_old[name] + delta_egm_exp_map[name] for name in selected_names}
            egm_exp_after = evaluate_state(
                algo=algo,
                rollout_data=probe.rollout_data,
                named_params=named_params,
                theta_state=theta_egm_exp,
                max_grad_norm=args.max_grad_norm,
                vf_coef=args.vf_coef,
                ent_coef=ent_coef,
                actor_weight=args.actor_weight,
                logstd_weight=args.logstd_weight,
                critic_weight=args.critic_weight,
                selected_names=selected_names,
            )

            theta_sgd = {name: theta_old[name] - args.eta_egm * F_raw[name] for name in selected_names}
            sgd_after = evaluate_state(
                algo=algo,
                rollout_data=probe.rollout_data,
                named_params=named_params,
                theta_state=theta_sgd,
                max_grad_norm=args.max_grad_norm,
                vf_coef=args.vf_coef,
                ent_coef=ent_coef,
                actor_weight=args.actor_weight,
                logstd_weight=args.logstd_weight,
                critic_weight=args.critic_weight,
                selected_names=selected_names,
            )

            theta_tmp = {name: tensor.clone() for name, tensor in theta_old.items()}
            ppm_inner_residual = []
            ppm_fp_residual = []
            for _ in range(args.ppm_inner_steps):
                tmp_eval = evaluate_state(
                    algo=algo,
                    rollout_data=probe.rollout_data,
                    named_params=named_params,
                    theta_state=theta_tmp,
                    max_grad_norm=args.max_grad_norm,
                    vf_coef=args.vf_coef,
                    ent_coef=ent_coef,
                    actor_weight=args.actor_weight,
                    logstd_weight=args.logstd_weight,
                    critic_weight=args.critic_weight,
                    selected_names=selected_names,
                )
                theta_next = {name: theta_old[name] - args.eta_egm * tmp_eval["grads"][name] for name in selected_names}
                ppm_inner_residual.append(tensor_norm(state_vector(theta_next, selected_names) - state_vector(theta_tmp, selected_names)))
                ppm_fp_residual.append(tensor_norm(flatten_named_tensors(tmp_eval["grads"], selected_names)))
                theta_tmp = theta_next
            ppm_after = evaluate_state(
                algo=algo,
                rollout_data=probe.rollout_data,
                named_params=named_params,
                theta_state=theta_tmp,
                max_grad_norm=args.max_grad_norm,
                vf_coef=args.vf_coef,
                ent_coef=ent_coef,
                actor_weight=args.actor_weight,
                logstd_weight=args.logstd_weight,
                critic_weight=args.critic_weight,
                selected_names=selected_names,
            )

            delta_egm_vec = flatten_named_tensors(delta_egm_map, selected_names)
            delta_egm_exp_vec = flatten_named_tensors(delta_egm_exp_map, selected_names)
            delta_sgd_vec = state_vector(theta_sgd, selected_names) - state_vector(theta_old, selected_names)
            delta_ppm_vec = state_vector(theta_tmp, selected_names) - state_vector(theta_old, selected_names)

            # Baseline rows once per probe
            baseline_entries = [
                ("sgd", delta_sgd_vec, sgd_after),
                ("egm", delta_egm_vec, egm_after),
                ("egm_expansion", delta_egm_exp_vec, egm_exp_after),
                ("ppm", delta_ppm_vec, ppm_after),
            ]
            for method, delta_vec, after_eval in baseline_entries:
                delta_map = {
                    name: (theta_sgd[name] - theta_old[name]) if method == "sgd" else
                          (theta_egm[name] - theta_old[name]) if method == "egm" else
                          (theta_egm_exp[name] - theta_old[name]) if method == "egm_expansion" else
                          (theta_tmp[name] - theta_old[name])
                    for name in selected_names
                }
                rows.append(
                    {
                        "probe_idx": probe.probe_idx,
                        "config_label": "baseline",
                        "method": method,
                        "eta_EGM": args.eta_egm,
                        "eta_EGM_squared": args.eta_egm ** 2,
                        "beta_max": np.nan,
                        "gamma_max": np.nan,
                        "fd_eps": args.fd_eps,
                        "beta_probe": args.beta_probe,
                        "gamma_probe": args.gamma_probe,
                        "beta_QP": np.nan,
                        "gamma_QP": np.nan,
                        "beta_eff": np.nan,
                        "gamma_eff": np.nan,
                        "beta_QP_over_eta_EGM": np.nan,
                        "gamma_QP_over_eta_EGM_squared": np.nan,
                        "beta_noG": np.nan,
                        "beta_noG_over_eta_EGM": np.nan,
                        "cosine_delta_QP_vs_EGM": np.nan,
                        "cosine_delta_QP_vs_EGM_denominator_too_small": np.nan,
                        "relative_error_delta_QP_vs_EGM": np.nan,
                        "relative_error_delta_QP_vs_EGM_denominator_too_small": np.nan,
                        "cosine_delta_QP_vs_EGM_expansion": np.nan,
                        "cosine_delta_QP_vs_EGM_expansion_denominator_too_small": np.nan,
                        "relative_error_delta_QP_vs_EGM_expansion": np.nan,
                        "relative_error_delta_QP_vs_EGM_expansion_denominator_too_small": np.nan,
                        "V_before": float(base_eval["V"]),
                        "V_after": float(after_eval["V"]),
                        "actual_V_change": float(after_eval["V"] - base_eval["V"]),
                        "PPO_loss_before": float(base_eval["total_loss"]),
                        "PPO_loss_after": float(after_eval["total_loss"]),
                        "PPO_loss_change": float(after_eval["total_loss"] - base_eval["total_loss"]),
                        "update_norm": tensor_norm(delta_vec),
                        "update_norm_pre_cap": tensor_norm(delta_vec),
                        "update_norm_post_cap": tensor_norm(delta_vec),
                        "cap_active": 0,
                        "actor_update_norm": block_norm(delta_map, selected_names, "actor"),
                        "logstd_update_norm": block_norm(delta_map, selected_names, "logstd"),
                        "critic_update_norm": block_norm(delta_map, selected_names, "critic"),
                        "approx_kl_after": float(after_eval["approx_kl"]),
                        "clip_fraction_after": float(after_eval["clip_fraction"]),
                        "F_raw_norm": F_norm,
                        "G_raw_norm": G_norm,
                        "cosine_F_G": cosine_similarity(F_vec, G_vec)[0] if fd_valid else np.nan,
                        "cosine_F_G_denominator_too_small": cosine_similarity(F_vec, G_vec)[1] if fd_valid else np.nan,
                        "finite_difference_valid": int(fd_valid),
                        "selected_case": "",
                        "q_condition_status": "",
                        "ridge_used": 0.0,
                        "beta_at_bound": 0,
                        "gamma_at_bound": 0,
                        "gamma_active": 0,
                        "gamma_active_frac": 0.0,
                        "G_contribution_norm": 0.0,
                        "G_over_update_norm": 0.0,
                        "G_over_update_norm_denominator_too_small": 0,
                        "update_norm_comparable_to_egm": float(tensor_norm(delta_vec) / max(tensor_norm(delta_egm_vec), 1e-12)),
                        "ppm_inner_residual_mean": float(np.mean(ppm_inner_residual)) if ppm_inner_residual else np.nan,
                        "ppm_fixed_point_residual_mean": float(np.mean(ppm_fp_residual)) if ppm_fp_residual else np.nan,
                    }
                )

            if not fd_valid:
                continue

            p_map = {name: -F_raw[name] for name in selected_names}
            r_map = {name: G_raw[name] for name in selected_names}
            coeffs = estimate_quadratic_coefficients_raw(
                algo=algo,
                rollout_data=probe.rollout_data,
                named_params=named_params,
                theta_old=theta_old,
                p_map=p_map,
                r_map=r_map,
                v0=float(base_eval["V"]),
                beta_probe=args.beta_probe,
                gamma_probe=args.gamma_probe,
                ridge=args.ridge,
                max_grad_norm=args.max_grad_norm,
                vf_coef=args.vf_coef,
                ent_coef=ent_coef,
                actor_weight=args.actor_weight,
                logstd_weight=args.logstd_weight,
                critic_weight=args.critic_weight,
                selected_names=selected_names,
            )

            for cfg in bound_configs:
                noG_sol = solve_no_g(float(coeffs["a"]), float(coeffs["c"]), cfg.beta_max, 1e-12)
                qp_sol = solve_two_direction_qp(
                    float(coeffs["a"]),
                    float(coeffs["b"]),
                    float(coeffs["c"]),
                    float(coeffs["h"]),
                    float(coeffs["k"]),
                    cfg.beta_max,
                    cfg.gamma_max,
                    float(coeffs["H_det"]),
                    str(coeffs["q_condition_status"]),
                    1e-12,
                )

                theta_nog_uncapped = apply_two_direction_delta(
                    theta_old, F_raw, G_raw, float(noG_sol["beta"]), 0.0, selected_names, eta=args.external_eta
                )
                theta_nog, nog_update_norm_pre, nog_update_norm_post, nog_cap_active = apply_update_cap(
                    theta_old, theta_nog_uncapped, selected_names, args.update_cap
                )
                nog_after = evaluate_state(
                    algo=algo,
                    rollout_data=probe.rollout_data,
                    named_params=named_params,
                    theta_state=theta_nog,
                    max_grad_norm=args.max_grad_norm,
                    vf_coef=args.vf_coef,
                    ent_coef=ent_coef,
                    actor_weight=args.actor_weight,
                    logstd_weight=args.logstd_weight,
                    critic_weight=args.critic_weight,
                    selected_names=selected_names,
                )
                theta_qp_uncapped = apply_two_direction_delta(
                    theta_old,
                    F_raw,
                    G_raw,
                    float(qp_sol["beta"]),
                    float(qp_sol["gamma"]),
                    selected_names,
                    eta=args.external_eta,
                )
                theta_qp, qp_update_norm_pre, qp_update_norm_post, qp_cap_active = apply_update_cap(
                    theta_old, theta_qp_uncapped, selected_names, args.update_cap
                )
                qp_after = evaluate_state(
                    algo=algo,
                    rollout_data=probe.rollout_data,
                    named_params=named_params,
                    theta_state=theta_qp,
                    max_grad_norm=args.max_grad_norm,
                    vf_coef=args.vf_coef,
                    ent_coef=ent_coef,
                    actor_weight=args.actor_weight,
                    logstd_weight=args.logstd_weight,
                    critic_weight=args.critic_weight,
                    selected_names=selected_names,
                )

                for method, solution, after_eval, theta_new in [
                    ("proposed_noG_new_v2_rawFG", noG_sol, nog_after, theta_nog),
                    ("proposed_qp_new_v2_rawFG", qp_sol, qp_after, theta_qp),
                ]:
                    delta_map = {name: theta_new[name] - theta_old[name] for name in selected_names}
                    delta_vec = flatten_named_tensors(delta_map, selected_names)
                    update_norm_pre = qp_update_norm_pre if method.endswith("qp_new_v2_rawFG") else nog_update_norm_pre
                    update_norm_post = qp_update_norm_post if method.endswith("qp_new_v2_rawFG") else nog_update_norm_post
                    cap_active = qp_cap_active if method.endswith("qp_new_v2_rawFG") else nog_cap_active
                    cosine_qp_egm, cosine_qp_egm_small = cosine_similarity(delta_vec, delta_egm_vec)
                    relerr_qp_egm, relerr_qp_egm_small = relative_error(delta_vec, delta_egm_vec)
                    cosine_qp_egmexp, cosine_qp_egmexp_small = cosine_similarity(delta_vec, delta_egm_exp_vec)
                    relerr_qp_egmexp, relerr_qp_egmexp_small = relative_error(delta_vec, delta_egm_exp_vec)
                    g_contrib = abs(args.external_eta * float(solution["gamma"])) * G_norm
                    update_norm = tensor_norm(delta_vec)
                    g_over_update, g_over_update_small = safe_div(g_contrib, update_norm, eps=1e-12)
                    beta_qp_ratio, beta_qp_ratio_small = safe_div(float(qp_sol["beta"]), args.eta_egm, eps=1e-12)
                    gamma_qp_ratio, gamma_qp_ratio_small = safe_div(float(qp_sol["gamma"]), args.eta_egm ** 2, eps=1e-12)
                    beta_nog_ratio, beta_nog_ratio_small = safe_div(float(noG_sol["beta"]), args.eta_egm, eps=1e-12)
                    rows.append(
                        {
                            "probe_idx": probe.probe_idx,
                            "config_label": cfg.label,
                            "method": method,
                            "eta_EGM": args.eta_egm,
                            "eta_EGM_squared": args.eta_egm ** 2,
                            "beta_max": cfg.beta_max,
                            "gamma_max": cfg.gamma_max,
                            "fd_eps": args.fd_eps,
                            "beta_probe": args.beta_probe,
                            "gamma_probe": args.gamma_probe,
                            "beta_QP": float(qp_sol["beta"]) if method.endswith("qp_new_v2_rawFG") else np.nan,
                            "gamma_QP": float(qp_sol["gamma"]) if method.endswith("qp_new_v2_rawFG") else np.nan,
                            "beta_eff": args.external_eta * float(qp_sol["beta"]) if method.endswith("qp_new_v2_rawFG") else args.external_eta * float(noG_sol["beta"]),
                            "gamma_eff": args.external_eta * float(qp_sol["gamma"]) if method.endswith("qp_new_v2_rawFG") else 0.0,
                            "beta_QP_over_eta_EGM": beta_qp_ratio if method.endswith("qp_new_v2_rawFG") else np.nan,
                            "beta_QP_over_eta_EGM_denominator_too_small": int(beta_qp_ratio_small) if method.endswith("qp_new_v2_rawFG") else np.nan,
                            "gamma_QP_over_eta_EGM_squared": gamma_qp_ratio if method.endswith("qp_new_v2_rawFG") else np.nan,
                            "gamma_QP_over_eta_EGM_squared_denominator_too_small": int(gamma_qp_ratio_small) if method.endswith("qp_new_v2_rawFG") else np.nan,
                            "beta_noG": float(noG_sol["beta"]),
                            "beta_noG_over_eta_EGM": beta_nog_ratio,
                            "beta_noG_over_eta_EGM_denominator_too_small": int(beta_nog_ratio_small),
                            "cosine_delta_QP_vs_EGM": cosine_qp_egm if method.endswith("qp_new_v2_rawFG") else np.nan,
                            "cosine_delta_QP_vs_EGM_denominator_too_small": int(cosine_qp_egm_small) if method.endswith("qp_new_v2_rawFG") else np.nan,
                            "relative_error_delta_QP_vs_EGM": relerr_qp_egm if method.endswith("qp_new_v2_rawFG") else np.nan,
                            "relative_error_delta_QP_vs_EGM_denominator_too_small": int(relerr_qp_egm_small) if method.endswith("qp_new_v2_rawFG") else np.nan,
                            "cosine_delta_QP_vs_EGM_expansion": cosine_qp_egmexp if method.endswith("qp_new_v2_rawFG") else np.nan,
                            "cosine_delta_QP_vs_EGM_expansion_denominator_too_small": int(cosine_qp_egmexp_small) if method.endswith("qp_new_v2_rawFG") else np.nan,
                            "relative_error_delta_QP_vs_EGM_expansion": relerr_qp_egmexp if method.endswith("qp_new_v2_rawFG") else np.nan,
                            "relative_error_delta_QP_vs_EGM_expansion_denominator_too_small": int(relerr_qp_egmexp_small) if method.endswith("qp_new_v2_rawFG") else np.nan,
                            "V_before": float(base_eval["V"]),
                            "V_after": float(after_eval["V"]),
                            "actual_V_change": float(after_eval["V"] - base_eval["V"]),
                            "PPO_loss_before": float(base_eval["total_loss"]),
                            "PPO_loss_after": float(after_eval["total_loss"]),
                            "PPO_loss_change": float(after_eval["total_loss"] - base_eval["total_loss"]),
                            "update_norm": update_norm,
                            "update_norm_pre_cap": update_norm_pre,
                            "update_norm_post_cap": update_norm_post,
                            "cap_active": cap_active,
                            "actor_update_norm": block_norm(delta_map, selected_names, "actor"),
                            "logstd_update_norm": block_norm(delta_map, selected_names, "logstd"),
                            "critic_update_norm": block_norm(delta_map, selected_names, "critic"),
                            "approx_kl_after": float(after_eval["approx_kl"]),
                            "clip_fraction_after": float(after_eval["clip_fraction"]),
                            "F_raw_norm": F_norm,
                            "G_raw_norm": G_norm,
                            "cosine_F_G": cosine_similarity(F_vec, G_vec)[0],
                            "cosine_F_G_denominator_too_small": int(cosine_similarity(F_vec, G_vec)[1]),
                            "finite_difference_valid": int(fd_valid),
                            "selected_case": str(solution["selected_case"]),
                            "q_condition_status": str(coeffs["q_condition_status"]),
                            "ridge_used": float(coeffs["ridge_used"]),
                            "beta_at_bound": int(solution["beta_at_bound"]),
                            "gamma_at_bound": int(solution["gamma_at_bound"]),
                            "gamma_active": int(solution["gamma_active"]),
                            "gamma_active_frac": float(solution["gamma_active"]),
                            "G_contribution_norm": g_contrib,
                            "G_over_update_norm": g_over_update,
                            "G_over_update_norm_denominator_too_small": int(g_over_update_small),
                            "update_norm_comparable_to_egm": update_norm / max(tensor_norm(delta_egm_vec), 1e-12),
                            "ppm_inner_residual_mean": float(np.mean(ppm_inner_residual)) if ppm_inner_residual else np.nan,
                            "ppm_fixed_point_residual_mean": float(np.mean(ppm_fp_residual)) if ppm_fp_residual else np.nan,
                            "a": float(coeffs["a"]),
                            "b": float(coeffs["b"]),
                            "c": float(coeffs["c"]),
                            "h": float(coeffs["h"]),
                            "k": float(coeffs["k"]),
                            "H_det": float(coeffs["H_det"]),
                            "q_pred": float(solution["q_pred"]),
                        }
                    )

        restore_state(named_params, theta_old)

    audit_df = pd.DataFrame(rows)
    audit_path = output_dir / "raw_fg_qp_audit.csv"
    audit_df.to_csv(audit_path, index=False)

    # Aggregate report views
    qp_df = audit_df[audit_df["method"] == "proposed_qp_new_v2_rawFG"].copy()
    nog_df = audit_df[audit_df["method"] == "proposed_noG_new_v2_rawFG"].copy()
    baseline_df = audit_df[audit_df["config_label"] == "baseline"].copy()

    agg_qp = qp_df.groupby("config_label").agg(
        beta_QP_mean=("beta_QP", "mean"),
        gamma_QP_mean=("gamma_QP", "mean"),
        beta_over_eta_mean=("beta_QP_over_eta_EGM", "mean"),
        gamma_over_eta2_mean=("gamma_QP_over_eta_EGM_squared", "mean"),
        V_change_QP_mean=("actual_V_change", "mean"),
        PPO_loss_change_QP_mean=("PPO_loss_change", "mean"),
        approx_kl_QP_max=("approx_kl_after", "max"),
        clip_fraction_QP_max=("clip_fraction_after", "max"),
        update_norm_QP_mean=("update_norm", "mean"),
        cosine_QP_vs_EGM_mean=("cosine_delta_QP_vs_EGM", "mean"),
        relerr_QP_vs_EGM_mean=("relative_error_delta_QP_vs_EGM", "mean"),
        cosine_QP_vs_EGMexp_mean=("cosine_delta_QP_vs_EGM_expansion", "mean"),
        relerr_QP_vs_EGMexp_mean=("relative_error_delta_QP_vs_EGM_expansion", "mean"),
        gamma_active_frac=("gamma_active", "mean"),
        G_contribution_norm_mean=("G_contribution_norm", "mean"),
        beta_at_bound_frac=("beta_at_bound", "mean"),
        gamma_at_bound_frac=("gamma_at_bound", "mean"),
    ).reset_index()
    agg_nog = nog_df.groupby("config_label").agg(
        beta_noG_mean=("beta_noG", "mean"),
        beta_noG_over_eta_mean=("beta_noG_over_eta_EGM", "mean"),
        V_change_noG_mean=("actual_V_change", "mean"),
        PPO_loss_change_noG_mean=("PPO_loss_change", "mean"),
    ).reset_index()
    agg = agg_qp.merge(agg_nog, on="config_label", how="left")

    egm_v = float(baseline_df.loc[baseline_df["method"] == "egm", "actual_V_change"].mean())
    egmexp_v = float(baseline_df.loc[baseline_df["method"] == "egm_expansion", "actual_V_change"].mean())
    ppm_v = float(baseline_df.loc[baseline_df["method"] == "ppm", "actual_V_change"].mean())
    sgd_v = float(baseline_df.loc[baseline_df["method"] == "sgd", "actual_V_change"].mean())

    agg["pass_gate"] = (
        np.isfinite(agg["beta_QP_mean"])
        & np.isfinite(agg["gamma_QP_mean"])
        & (agg["gamma_active_frac"] > 0.0)
        & (agg["G_contribution_norm_mean"] > 0.0)
        & (agg["V_change_QP_mean"] < agg["V_change_noG_mean"])
        & (agg["V_change_QP_mean"] <= max(egm_v, egmexp_v) + 1e-6)
        & (agg["approx_kl_QP_max"] <= 0.1)
        & (agg["clip_fraction_QP_max"] <= 0.8)
        & (agg["update_norm_QP_mean"] <= max(float(baseline_df.loc[baseline_df["method"] == "egm", "update_norm"].mean()) * 5.0, 1e-12))
    )

    best_row = agg.sort_values(["pass_gate", "V_change_QP_mean"], ascending=[False, True]).iloc[0]

    report_lines = [
        "# Raw F/G QP Audit Report",
        "",
        "- Diagnostic only. No online training was run.",
        f"- Active role: `{args.role}`",
        f"- Shared field config for this audit: `eta_EGM={args.eta_egm}`, `max_grad_norm={args.max_grad_norm}`, `vf_coef={args.vf_coef}`, `ent_coef={ent_coef}`",
        f"- Raw G construction: `G = (F(z + eps_fd * F) - F(z)) / eps_fd`, with `eps_fd={args.fd_eps}`",
        "",
        "## Direct answers",
        "",
        f"1. In raw F/G mode, does beta_QP exceed eta_EGM? `{bool((agg['beta_QP_mean'] > args.eta_egm).any())}`",
        f"2. Does gamma_QP exceed eta_EGM^2? `{bool((agg['gamma_QP_mean'] > args.eta_egm ** 2).any())}`",
        f"3. Is QP closer to EGM actual update or EGM expansion? `{'EGM expansion' if float(best_row['relerr_QP_vs_EGMexp_mean']) < float(best_row['relerr_QP_vs_EGM_mean']) else 'EGM actual'}`",
        f"4. Does QP decrease V more than noG? `{bool((agg['V_change_QP_mean'] < agg['V_change_noG_mean']).any())}`",
        f"5. Does QP decrease V more than EGM? `{bool((agg['V_change_QP_mean'] < egm_v).any())}`",
        f"6. Does raw F/G mode look safer or better aligned than normalized mode? `Mixed; use gate summary below.`",
        f"7. Is it worth running short online? `{bool(agg['pass_gate'].any())}`",
        "",
        "## Baseline references",
        "",
        f"- V_change_EGM mean: `{egm_v:.6e}`",
        f"- V_change_EGM_expansion mean: `{egmexp_v:.6e}`",
        f"- V_change_PPM mean: `{ppm_v:.6e}`",
        f"- V_change_SGD mean: `{sgd_v:.6e}`",
        "",
        "## Best raw-F/G config by gate then V decrease",
        "",
        f"- config_label: `{best_row['config_label']}`",
        f"- beta_QP mean: `{best_row['beta_QP_mean']:.6e}`",
        f"- gamma_QP mean: `{best_row['gamma_QP_mean']:.6e}`",
        f"- beta_QP / eta_EGM: `{best_row['beta_over_eta_mean']:.6f}`",
        f"- gamma_QP / eta_EGM^2: `{best_row['gamma_over_eta2_mean']:.6f}`",
        f"- V_change_QP mean: `{best_row['V_change_QP_mean']:.6e}`",
        f"- V_change_noG mean: `{best_row['V_change_noG_mean']:.6e}`",
        f"- approx_kl_QP max: `{best_row['approx_kl_QP_max']:.6f}`",
        f"- clip_fraction_QP max: `{best_row['clip_fraction_QP_max']:.6f}`",
        f"- gamma_active_frac: `{best_row['gamma_active_frac']:.6f}`",
        f"- pass_gate: `{bool(best_row['pass_gate'])}`",
    ]
    (output_dir / "raw_fg_qp_audit_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(agg["config_label"], agg["beta_over_eta_mean"], marker="o", label="beta / eta_EGM")
    axes[0].axhline(1.0, linestyle="--", color="tab:gray", label="EGM-like beta")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_title("beta_QP relative to eta_EGM")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(agg["config_label"], agg["gamma_over_eta2_mean"], marker="o", color="tab:blue", label="gamma / eta_EGM^2")
    axes[1].axhline(1.0, linestyle="--", color="tab:gray", label="EGM-like gamma")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_title("gamma_QP relative to eta_EGM^2")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "raw_fg_beta_gamma_vs_egm.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(agg["config_label"], agg["cosine_QP_vs_EGM_mean"], marker="o", label="cos(QP, EGM)")
    axes[0].plot(agg["config_label"], agg["cosine_QP_vs_EGMexp_mean"], marker="s", label="cos(QP, EGM expansion)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_title("Update cosine geometry")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(agg["config_label"], agg["relerr_QP_vs_EGM_mean"], marker="o", label="relerr vs EGM")
    axes[1].plot(agg["config_label"], agg["relerr_QP_vs_EGMexp_mean"], marker="s", label="relerr vs EGM expansion")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_title("Relative error geometry")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "raw_fg_update_geometry.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(agg["config_label"], agg["V_change_QP_mean"], marker="o", label="QP")
    ax.plot(agg["config_label"], agg["V_change_noG_mean"], marker="s", label="noG")
    ax.axhline(egm_v, linestyle="--", color="tab:green", label="EGM")
    ax.axhline(egmexp_v, linestyle=":", color="tab:blue", label="EGM expansion")
    ax.axhline(ppm_v, linestyle="--", color="tab:red", label="PPM")
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("V change comparison")
    ax.set_ylabel("V_after - V_before")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "raw_fg_V_change.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(agg["config_label"], agg["approx_kl_QP_max"], marker="o", label="approx_kl_QP_max")
    axes[0].axhline(0.1, linestyle="--", color="tab:red", label="gate")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_title("QP KL after update")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(agg["config_label"], agg["clip_fraction_QP_max"], marker="o", label="clip_fraction_QP_max")
    axes[1].axhline(0.8, linestyle="--", color="tab:red", label="gate")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_title("QP clip fraction after update")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "raw_fg_kl_clip.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
