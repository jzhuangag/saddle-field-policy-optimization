from __future__ import annotations

import argparse
import itertools
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from models.proposed_qp_new import apply_state_delta, flatten_named_tensors, named_difference, tensor_norm
from models.proposed_qp_perflyap import (
    ProposedNoGPerfLyapOptimizer,
    ProposedQPPerfLyapOptimizer,
    block_names,
    block_norm,
    classify_perf_block,
)
from scripts.full_policy_optimizer_probe import clone_state, collect_probe_batches, named_parameters, restore_state
from scripts.run_perflyap_stage4_preflight import build_model, pair_probes, role_algo


RESULT_METHODS = {
    "nog": "proposed_noG_perfLyap",
    "qp": "proposed_qp_perfLyap",
}
DIR_MODES = ["noG", "egm_plus", "egm_minus", "performance_grad", "merit_grad"]
COST_MODES = ["actor_surrogate_cost", "unclipped_actor_surrogate_cost"]


@dataclass(frozen=True)
class Stage48Config:
    config_id: int
    scope: str
    lambda_N: float
    lambda_P: float
    cost_mode: str
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
            f"_c{self.cost_mode.replace('_cost', '')}"
            f"_b{self.beta_max:g}"
            f"_g{self.gamma_max:g}"
            f"_cap{self.update_cap:g}"
            f"_fd{self.fd_eps:g}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 4.8 performance-only / small-norm merit audit")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-probes-per-role", type=int, default=4)
    parser.add_argument("--max-configs", type=int, default=72)
    parser.add_argument("--eta-egm", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--vf-coef", type=float, default=1.0)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    return parser.parse_args()


def build_configs(max_configs: int) -> List[Stage48Config]:
    configs: List[Stage48Config] = []
    config_id = 0
    for scope, lambda_N, cost_mode, beta_max in itertools.product(
        ["actor_mean_only", "actor_game", "actor_mean_heavy"],
        [0.0, 1e-4, 1e-3, 1e-2, 0.03, 0.1],
        COST_MODES,
        [1e-2, 3e-2],
    ):
        if len(configs) >= max_configs:
            break
        configs.append(
            Stage48Config(
                config_id=config_id,
                scope=scope,
                lambda_N=float(lambda_N),
                lambda_P=1.0,
                cost_mode=cost_mode,
                beta_max=float(beta_max),
                gamma_max=3e-5,
                update_cap=0.005,
                fd_eps=1e-3,
                beta_probe=1e-3,
                gamma_probe=1e-6,
                ridge=1e-8,
                rho=1e-8,
            )
        )
        config_id += 1
    return configs


def build_eval_context(algo, rollout_data) -> Dict[str, object]:
    clip_range = algo.clip_range(algo._current_progress_remaining)
    clip_range_vf = None
    if algo.clip_range_vf is not None:
        clip_range_vf = algo.clip_range_vf(algo._current_progress_remaining)
    actions = rollout_data.actions.long().flatten() if algo.action_space.__class__.__name__ == "Discrete" else rollout_data.actions
    named_params = algo._named_policy_parameters()
    return {
        "algo": algo,
        "rollout_data": rollout_data,
        "clip_range": float(clip_range),
        "clip_range_vf": clip_range_vf,
        "actions": actions,
        "named_params": named_params,
    }


def _loss_tensors(ctx: Dict[str, object], theta_override=None):
    algo = ctx["algo"]
    rollout_data = ctx["rollout_data"]
    clip_range = ctx["clip_range"]
    clip_range_vf = ctx["clip_range_vf"]
    actions = ctx["actions"]
    named_params = ctx["named_params"]

    if theta_override is not None:
        restore_state(named_params, theta_override)

    values, log_prob, entropy = algo.policy.evaluate_actions(rollout_data.observations, actions)
    values = values.flatten()
    advantages = rollout_data.advantages
    if algo.normalize_advantage and len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    ratio = torch.exp(log_prob - rollout_data.old_log_prob)
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
    unclipped_actor_surrogate_cost = -policy_loss_1.mean()

    if clip_range_vf is None:
        values_pred = values
    else:
        values_pred = rollout_data.old_values + torch.clamp(values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
    value_loss = F.mse_loss(rollout_data.returns, values_pred)
    if entropy is None:
        entropy_loss = -torch.mean(-log_prob)
    else:
        entropy_loss = -torch.mean(entropy)
    total_loss = policy_loss + algo.ent_coef * entropy_loss + algo.vf_coef * value_loss
    clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float())
    with torch.no_grad():
        log_ratio = log_prob - rollout_data.old_log_prob
        approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)
    return {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "unclipped_actor_surrogate_cost": unclipped_actor_surrogate_cost,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "clip_fraction": clip_fraction,
        "approx_kl": approx_kl,
    }


def build_extended_eval_closure(ctx: Dict[str, object], cost_mode: str):
    named_params = ctx["named_params"]
    algo = ctx["algo"]

    def eval_closure(*, theta_override=None, backward: bool = True, grad_scope_names: List[str] | None = None) -> Dict[str, object]:
        if theta_override is not None:
            restore_state(named_params, theta_override)
        algo.policy.optimizer.zero_grad()
        critic_optimizer = getattr(algo, "_actor_game_critic_optimizer", None)
        if critic_optimizer is not None:
            critic_optimizer.zero_grad()

        tensors = _loss_tensors(ctx, theta_override=None)
        perf_cost = tensors["policy_loss"] if cost_mode == "actor_surrogate_cost" else tensors["unclipped_actor_surrogate_cost"]
        if backward:
            tensors["total_loss"].backward()
            if np.isfinite(algo.max_grad_norm):
                if grad_scope_names is None:
                    grad_params = list(algo.policy.parameters())
                else:
                    scope_name_set = set(grad_scope_names)
                    grad_params = [param for name, param in named_params if name in scope_name_set]
                torch.nn.utils.clip_grad_norm_(grad_params, algo.max_grad_norm)
        grads = {
            name: (param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param.data))
            for name, param in named_params
        }
        return {
            "loss_tensor": tensors["total_loss"].detach(),
            "total_loss": float(tensors["total_loss"].item()),
            "policy_loss": float(perf_cost.item()),
            "policy_loss_clipped": float(tensors["policy_loss"].item()),
            "policy_loss_unclipped": float(tensors["unclipped_actor_surrogate_cost"].item()),
            "value_loss": float(tensors["value_loss"].item()),
            "entropy_loss": float(tensors["entropy_loss"].item()),
            "clip_fraction": float(tensors["clip_fraction"].item()),
            "approx_kl": float(tensors["approx_kl"].item()),
            "grads": grads,
        }

    return eval_closure


def _selected_param_list(named_params, selected_names: Sequence[str]) -> List[Tuple[str, torch.nn.Parameter]]:
    selected = set(selected_names)
    return [(name, param) for name, param in named_params if name in selected]


def _clip_named_grad_map(grad_map: Dict[str, torch.Tensor], selected_names: Sequence[str], max_grad_norm: float, eps: float) -> Dict[str, torch.Tensor]:
    out = {name: grad_map[name].clone() for name in selected_names}
    if not np.isfinite(max_grad_norm):
        return out
    vec = flatten_named_tensors(out, selected_names)
    norm = tensor_norm(vec)
    if norm > max_grad_norm and norm > eps:
        scale = max_grad_norm / norm
        for name in selected_names:
            out[name] = out[name] * scale
    return out


def grad_for_cost(ctx: Dict[str, object], theta_state, selected_names: Sequence[str], cost_mode: str, max_grad_norm: float, eps: float) -> Dict[str, torch.Tensor]:
    named_params = ctx["named_params"]
    restore_state(named_params, theta_state)
    for _, param in named_params:
        if param.grad is not None:
            param.grad.zero_()
    tensors = _loss_tensors(ctx, theta_override=None)
    cost_tensor = tensors["policy_loss"] if cost_mode == "actor_surrogate_cost" else tensors["unclipped_actor_surrogate_cost"]
    cost_tensor.backward()
    grad_map = {name: (param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param.data)) for name, param in named_params if name in selected_names}
    return _clip_named_grad_map(grad_map, selected_names, max_grad_norm, eps)


def grad_for_merit(helper, ctx: Dict[str, object], theta_state, selected_names: Sequence[str], cost_mode: str, norm_scale: float, perf_scale: float) -> Dict[str, torch.Tensor]:
    named_params = ctx["named_params"]
    selected_params = _selected_param_list(named_params, selected_names)
    restore_state(named_params, theta_state)
    for _, param in named_params:
        if param.grad is not None:
            param.grad.zero_()
    tensors = _loss_tensors(ctx, theta_override=None)
    total_loss = tensors["total_loss"]
    perf_cost = tensors["policy_loss"] if cost_mode == "actor_surrogate_cost" else tensors["unclipped_actor_surrogate_cost"]
    grads_graph = torch.autograd.grad(total_loss, [param for _, param in selected_params], create_graph=True, retain_graph=True, allow_unused=True)
    grad_map_graph = {
        name: (grad if grad is not None else torch.zeros_like(param))
        for (name, param), grad in zip(selected_params, grads_graph)
    }
    block_weights = helper._scope_block_weights()

    def block_mean_square(block: str) -> torch.Tensor:
        names = [name for name in selected_names if classify_perf_block(name) == block]
        if not names:
            return torch.zeros((), dtype=total_loss.dtype, device=total_loss.device)
        accum = torch.zeros((), dtype=total_loss.dtype, device=total_loss.device)
        for name in names:
            g = grad_map_graph[name]
            accum = accum + torch.mean(g.pow(2))
        return accum / max(len(names), 1)

    norm_term = (
        0.5 * block_weights["actor"] * block_mean_square("actor")
        + 0.5 * block_weights["logstd"] * block_mean_square("logstd")
        + 0.5 * block_weights["critic"] * block_mean_square("critic")
    )
    merit_tensor = helper.lambda_N * norm_term / max(norm_scale, helper.qp_eps) + helper.lambda_P * perf_cost / max(perf_scale, helper.qp_eps)
    merit_grads = torch.autograd.grad(merit_tensor, [param for _, param in selected_params], allow_unused=True)
    grad_map = {
        name: (grad.detach().clone() if grad is not None else torch.zeros_like(param))
        for (name, param), grad in zip(selected_params, merit_grads)
    }
    return _clip_named_grad_map(grad_map, selected_names, ctx["algo"].max_grad_norm, helper.qp_eps)


def zero_direction(selected_names: Sequence[str], reference_map: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: torch.zeros_like(reference_map[name]) for name in selected_names}


def instantiate_helper(named_params, config: Stage48Config, args: argparse.Namespace, diagnostics_csv_path: pathlib.Path, role: str, disable_g: bool) -> _PerformanceAlignedLyapunovQPBase:
    optimizer_cls = ProposedNoGPerfLyapOptimizer if disable_g else ProposedQPPerfLyapOptimizer
    return optimizer_cls(
        [param for _, param in named_params],
        lr=args.lr,
        perflyap_scope=config.scope,
        lambda_N=config.lambda_N,
        lambda_P=config.lambda_P,
        lambda_critic=0.0,
        logstd_weight=0.0,
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
        disable_g=disable_g,
    )


def evaluate_candidate(helper, eval_closure, named_params, theta_old, selected_names, base_eval, norm_scale, perf_scale, solution, direction_mode: str, eta_egm: float) -> Dict[str, object]:
    theta_new = apply_state_delta(theta_old, solution["update_map"])
    new_eval = helper._evaluate_state(
        eval_closure=eval_closure,
        theta_state=theta_new,
        selected_names=selected_names,
        backward=True,
        norm_scale=norm_scale,
        perf_scale=perf_scale,
    )
    restore_state(named_params, theta_old)
    diff = named_difference(theta_new, theta_old, selected_names)
    actor_update_norm = block_norm(diff, selected_names, "actor")
    logstd_update_norm = block_norm(diff, selected_names, "logstd")
    critic_update_norm = block_norm(diff, selected_names, "critic")
    total_v_decrease = max(float(base_eval["V_merit"]) - float(new_eval["V_merit"]), helper.qp_eps)
    actor_norm_decrease = helper.lambda_N * (float(base_eval["actor_norm_term"]) - float(new_eval["actor_norm_term"])) / max(norm_scale, helper.qp_eps)
    logstd_norm_decrease = helper.lambda_N * (float(base_eval["logstd_norm_term"]) - float(new_eval["logstd_norm_term"])) / max(norm_scale, helper.qp_eps)
    critic_norm_decrease = helper.lambda_N * (float(base_eval["critic_norm_term"]) - float(new_eval["critic_norm_term"])) / max(norm_scale, helper.qp_eps)
    actor_perf_decrease = helper.lambda_P * (float(base_eval["policy_component"]) - float(new_eval["policy_component"])) / max(perf_scale, helper.qp_eps)
    logstd_perf_decrease = helper.lambda_P * (float(base_eval["logstd_component"]) - float(new_eval["logstd_component"])) / max(perf_scale, helper.qp_eps)
    critic_perf_decrease = helper.lambda_P * (float(base_eval["critic_component"]) - float(new_eval["critic_component"])) / max(perf_scale, helper.qp_eps)
    actor_v_decrease = actor_norm_decrease + actor_perf_decrease
    logstd_v_decrease = logstd_norm_decrease + logstd_perf_decrease
    critic_v_decrease = critic_norm_decrease + critic_perf_decrease
    total_update_norm = max(float(solution["update_norm_post_cap"]), helper.qp_eps)
    gamma_eff = abs(float(solution.get("gamma_eff", 0.0)))
    return {
        "direction_mode": direction_mode,
        "beta": float(solution["beta"]),
        "gamma": float(solution["gamma"]),
        "beta_raw": float(solution["beta_raw"]),
        "gamma_raw": float(solution["gamma_raw"]),
        "beta_eff": float(solution["beta_eff"]),
        "gamma_eff": gamma_eff,
        "beta_over_eta_egm": float(solution["beta_eff"]) / eta_egm,
        "gamma_over_eta_egm_squared": gamma_eff / max(eta_egm * eta_egm, helper.qp_eps),
        "q_pred_pre_cap": float(solution["q_pred_pre_cap"]),
        "q_pred_post_cap": float(solution["q_pred_post_cap"]),
        "update_norm_pre_cap": float(solution["update_norm_pre_cap"]),
        "update_norm_post_cap": float(solution["update_norm_post_cap"]),
        "cap_scale": float(solution["cap_scale"]),
        "cap_active": int(solution["cap_active"]),
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
        "actor_update_norm": actor_update_norm,
        "logstd_update_norm": logstd_update_norm,
        "critic_update_norm": critic_update_norm,
        "actor_fraction_of_update": actor_update_norm / total_update_norm,
        "logstd_fraction_of_update": logstd_update_norm / total_update_norm,
        "critic_fraction_of_update": critic_update_norm / total_update_norm,
        "actor_fraction_of_V_decrease": actor_v_decrease / total_v_decrease,
        "logstd_fraction_of_V_decrease": logstd_v_decrease / total_v_decrease,
        "critic_fraction_of_V_decrease": critic_v_decrease / total_v_decrease,
        "approx_kl": float(new_eval["approx_kl"]),
        "clip_fraction": float(new_eval["clip_fraction"]),
        "gamma_active_frac": float(gamma_eff > helper.qp_eps),
        "G_contribution_norm": gamma_eff * tensor_norm(flatten_named_tensors(solution["direction_map"], selected_names)),
        "beta_at_bound": int(abs(float(solution["beta"]) - helper.qp_beta_max) <= helper.qp_eps and helper.qp_beta_max > 0.0),
        "gamma_at_bound": int(abs(float(solution["gamma"]) - helper.qp_gamma_max) <= helper.qp_eps and helper.qp_gamma_max > 0.0),
        "selected_case": str(solution.get("selected_case", "")),
        "fallback_reason": str(solution.get("fallback_reason", "")),
        "dense_fallback_used": int(solution.get("dense_fallback_used", 0)),
        "direction_norm": tensor_norm(flatten_named_tensors(solution["direction_map"], selected_names)),
    }


def tolerance_for_c(no_g_c: float) -> float:
    return max(1e-4, 0.1 * abs(no_g_c))


def tolerance_for_v(no_g_v: float) -> float:
    return max(1e-4, 0.05 * abs(no_g_v))


def solve_direction_mode(helper, theta_old, eval_closure, named_params, selected_names, base_eval, norm_scale, perf_scale, f_raw_map, direction_mode: str, direction_map: Dict[str, torch.Tensor], eta_egm: float):
    coeffs = helper._estimate_quadratic_coefficients(
        theta_old=theta_old,
        eval_closure=eval_closure,
        selected_names=selected_names,
        f_raw_map=f_raw_map,
        g_raw_map=direction_map,
        v0=float(base_eval["V_merit"]),
        norm_scale=norm_scale,
        perf_scale=perf_scale,
    )
    no_g_solution = helper._solve_no_g(
        coeffs,
        eta=1.0,
        f_raw_map=f_raw_map,
        g_raw_map=direction_map,
        selected_names=selected_names,
        compare_with_raw_coeffs=True,
    )
    no_g_solution["direction_map"] = zero_direction(selected_names, f_raw_map)
    qp_solution = helper._solve_two_direction_qp(
        coeffs,
        eta=1.0,
        f_raw_map=f_raw_map,
        g_raw_map=direction_map,
        selected_names=selected_names,
        g_sign="plus",
        no_g_solution=no_g_solution,
    )
    qp_solution["direction_map"] = direction_map
    no_g_metrics = evaluate_candidate(helper, eval_closure, named_params, theta_old, selected_names, base_eval, norm_scale, perf_scale, no_g_solution, direction_mode, eta_egm)
    qp_metrics = evaluate_candidate(helper, eval_closure, named_params, theta_old, selected_names, base_eval, norm_scale, perf_scale, qp_solution, direction_mode, eta_egm)
    qp_metrics["no_g_reference_q_pred_post_cap"] = float(no_g_solution["q_pred_post_cap"])
    qp_metrics["pass_predicted_invariant"] = bool(float(qp_solution["q_pred_post_cap"]) <= float(no_g_solution["q_pred_post_cap"]) + 1e-10)
    return coeffs, no_g_solution, qp_solution, no_g_metrics, qp_metrics


def run_single_probe(*, algo, role: str, train_probe, val_probe, config: Stage48Config, args: argparse.Namespace, diagnostics_csv_path: pathlib.Path) -> List[Dict[str, object]]:
    named_params = named_parameters(algo.policy)
    helper = instantiate_helper(named_params, config, args, diagnostics_csv_path, role, disable_g=False)
    train_eval = build_extended_eval_closure(build_eval_context(algo, train_probe.rollout_data), config.cost_mode)
    val_eval = build_extended_eval_closure(build_eval_context(algo, val_probe.rollout_data), config.cost_mode)
    theta_old = clone_state(named_params)
    selected_names = helper._selected_names(named_params)
    base_eval_unscaled = helper._evaluate_state(
        eval_closure=train_eval,
        theta_state=theta_old,
        selected_names=selected_names,
        backward=True,
    )
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
    g_plus_map, finite_difference_valid = helper._compute_g_raw(
        theta_old=theta_old,
        eval_closure=train_eval,
        selected_names=selected_names,
        f_raw_map=f_raw_map,
        norm_scale=norm_scale,
        perf_scale=perf_scale,
    )
    perf_grad_map = {
        name: -tensor
        for name, tensor in grad_for_cost(
            build_eval_context(algo, train_probe.rollout_data),
            theta_old,
            selected_names,
            config.cost_mode,
            args.max_grad_norm,
            helper.qp_eps,
        ).items()
    }
    merit_grad_map = {
        name: -tensor
        for name, tensor in grad_for_merit(
            helper,
            build_eval_context(algo, train_probe.rollout_data),
            theta_old,
            selected_names,
            config.cost_mode,
            norm_scale,
            perf_scale,
        ).items()
    }
    direction_maps = {
        "egm_plus": g_plus_map,
        "egm_minus": {name: -g_plus_map[name] for name in selected_names},
        "performance_grad": perf_grad_map,
        "merit_grad": merit_grad_map,
    }

    rows: List[Dict[str, object]] = []
    for direction_mode in DIR_MODES:
        if direction_mode == "noG":
            direction_map = zero_direction(selected_names, f_raw_map)
            coeffs = helper._estimate_quadratic_coefficients(
                theta_old=theta_old,
                eval_closure=train_eval,
                selected_names=selected_names,
                f_raw_map=f_raw_map,
                g_raw_map=direction_map,
                v0=float(base_eval["V_merit"]),
                norm_scale=norm_scale,
                perf_scale=perf_scale,
            )
            no_g_solution = helper._solve_no_g(
                coeffs,
                eta=1.0,
                f_raw_map=f_raw_map,
                g_raw_map=direction_map,
                selected_names=selected_names,
                compare_with_raw_coeffs=True,
            )
            no_g_solution["direction_map"] = direction_map
            metrics = evaluate_candidate(helper, train_eval, named_params, theta_old, selected_names, base_eval, norm_scale, perf_scale, no_g_solution, "noG", args.eta_egm)
            metrics["method"] = RESULT_METHODS["nog"]
            metrics["finite_difference_valid"] = int(finite_difference_valid)
            metrics["role"] = role
            metrics["config_id"] = config.config_id
            metrics["config_label"] = config.label
            metrics["scope"] = config.scope
            metrics["cost_mode"] = config.cost_mode
            metrics["lambda_N"] = config.lambda_N
            metrics["lambda_P"] = config.lambda_P
            metrics["train_probe_id"] = int(train_probe.probe_idx)
            metrics["val_probe_id"] = int(val_probe.probe_idx)
            val_before = helper._evaluate_state(eval_closure=val_eval, theta_state=theta_old, selected_names=selected_names, backward=True, norm_scale=norm_scale, perf_scale=perf_scale)
            theta_new = apply_state_delta(theta_old, no_g_solution["update_map"])
            val_after = helper._evaluate_state(eval_closure=val_eval, theta_state=theta_new, selected_names=selected_names, backward=True, norm_scale=norm_scale, perf_scale=perf_scale)
            restore_state(named_params, theta_old)
            metrics["val_actual_V_change"] = float(val_after["V_merit"] - val_before["V_merit"])
            metrics["val_actual_C_change"] = float(val_after["C"] - val_before["C"])
            rows.append(metrics)
            continue

        direction_map = direction_maps[direction_mode]
        coeffs, no_g_solution, qp_solution, no_g_metrics, qp_metrics = solve_direction_mode(
            helper,
            theta_old,
            train_eval,
            named_params,
            selected_names,
            base_eval,
            norm_scale,
            perf_scale,
            f_raw_map,
            direction_mode,
            direction_map,
            args.eta_egm,
        )
        for method_key, metrics in [("nog", no_g_metrics), ("qp", qp_metrics)]:
            metrics["method"] = RESULT_METHODS[method_key]
            metrics["finite_difference_valid"] = int(finite_difference_valid)
            metrics["role"] = role
            metrics["config_id"] = config.config_id
            metrics["config_label"] = config.label
            metrics["scope"] = config.scope
            metrics["cost_mode"] = config.cost_mode
            metrics["lambda_N"] = config.lambda_N
            metrics["lambda_P"] = config.lambda_P
            metrics["train_probe_id"] = int(train_probe.probe_idx)
            metrics["val_probe_id"] = int(val_probe.probe_idx)
            metrics["l_beta"] = float(coeffs["l_beta"])
            metrics["l_gamma"] = float(coeffs["l_gamma"])
            metrics["H_bb"] = float(coeffs["H_bb"])
            metrics["H_bg"] = float(coeffs["H_bg"])
            metrics["H_gg"] = float(coeffs["H_gg"])
            metrics["eig_min"] = float(coeffs["eig_min"])
            metrics["ridge_added"] = float(coeffs["ridge_added"])
            val_before = helper._evaluate_state(eval_closure=val_eval, theta_state=theta_old, selected_names=selected_names, backward=True, norm_scale=norm_scale, perf_scale=perf_scale)
            theta_new = apply_state_delta(theta_old, no_g_solution["update_map"] if method_key == "nog" else qp_solution["update_map"])
            val_after = helper._evaluate_state(eval_closure=val_eval, theta_state=theta_new, selected_names=selected_names, backward=True, norm_scale=norm_scale, perf_scale=perf_scale)
            restore_state(named_params, theta_old)
            metrics["val_actual_V_change"] = float(val_after["V_merit"] - val_before["V_merit"])
            metrics["val_actual_C_change"] = float(val_after["C"] - val_before["C"])
            rows.append(metrics)
    restore_state(named_params, theta_old)
    return rows


def summarize_config_direction(config: Stage48Config, direction_mode: str, rows: pd.DataFrame) -> Dict[str, object]:
    summary = {
        "config_id": config.config_id,
        "config_label": config.label,
        "scope": config.scope,
        "cost_mode": config.cost_mode,
        "lambda_N": config.lambda_N,
        "lambda_P": config.lambda_P,
        "beta_max": config.beta_max,
        "gamma_max": config.gamma_max,
        "update_cap": config.update_cap,
        "direction_mode": direction_mode,
    }
    nog = rows[rows["method"] == RESULT_METHODS["nog"]]
    qp = rows[rows["method"] == RESULT_METHODS["qp"]]
    for prefix, sub in [("nog", nog), ("qp", qp)]:
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
            "beta_eff",
            "gamma_eff",
            "update_norm_post_cap",
            "actor_fraction_of_update",
            "actor_fraction_of_V_decrease",
            "val_actual_V_change",
            "val_actual_C_change",
            "q_pred_post_cap",
            "no_g_reference_q_pred_post_cap",
        ]:
            if col in sub.columns:
                summary[f"{prefix}_{col}_mean"] = float(pd.to_numeric(sub[col], errors="coerce").mean())
                summary[f"{prefix}_{col}_max"] = float(pd.to_numeric(sub[col], errors="coerce").max())
        core_cols = [c for c in ["actual_V_change", "actual_C_change", "approx_kl", "clip_fraction", "update_norm_post_cap"] if c in sub.columns]
        summary[f"{prefix}_core_finite"] = bool(np.isfinite(sub[core_cols].astype(float).to_numpy()).all())
    nog_v = float(summary.get("nog_actual_V_change_mean", float("inf")))
    qp_v = float(summary.get("qp_actual_V_change_mean", float("inf")))
    nog_c = float(summary.get("nog_actual_C_change_mean", float("inf")))
    qp_c = float(summary.get("qp_actual_C_change_mean", float("inf")))
    nog_val_v = float(summary.get("nog_val_actual_V_change_mean", float("inf")))
    qp_val_v = float(summary.get("qp_val_actual_V_change_mean", float("inf")))
    nog_val_c = float(summary.get("nog_val_actual_C_change_mean", float("inf")))
    qp_val_c = float(summary.get("qp_val_actual_C_change_mean", float("inf")))
    summary["pass_core"] = bool(summary.get("nog_core_finite", False) and summary.get("qp_core_finite", False))
    summary["pass_predicted_invariant"] = bool(
        float(summary.get("qp_q_pred_post_cap_mean", float("inf")))
        <= float(summary.get("qp_no_g_reference_q_pred_post_cap_mean", float("inf"))) + 1e-10
    )
    summary["pass_composite_actual"] = bool(qp_v <= nog_v + tolerance_for_v(nog_v))
    summary["pass_c_not_catastrophic"] = bool(qp_c <= nog_c + tolerance_for_c(nog_c))
    summary["pass_kl_clip"] = bool(float(summary.get("qp_approx_kl_max", float("inf"))) <= 0.1 and float(summary.get("qp_clip_fraction_max", float("inf"))) <= 0.8)
    summary["pass_gamma_active"] = bool(direction_mode == "noG" or (float(summary.get("qp_gamma_active_frac_mean", 0.0)) > 0.0 and float(summary.get("qp_G_contribution_norm_mean", 0.0)) > 0.0))
    summary["pass_actor_focus"] = bool(float(summary.get("qp_actor_fraction_of_update_mean", 0.0)) >= 0.5)
    summary["pass_validation"] = bool(qp_val_v <= nog_val_v + tolerance_for_v(nog_val_v) and qp_val_c <= nog_val_c + tolerance_for_c(nog_val_c))
    summary["stage48_pass"] = bool(
        summary["pass_core"]
        and summary["pass_predicted_invariant"]
        and summary["pass_composite_actual"]
        and summary["pass_c_not_catastrophic"]
        and summary["pass_kl_clip"]
        and summary["pass_gamma_active"]
        and summary["pass_actor_focus"]
        and summary["pass_validation"]
    )
    return summary


def plot_stage48(summary_df: pd.DataFrame, output_root: pathlib.Path) -> None:
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    top = summary_df.sort_values(["stage48_pass", "qp_actual_C_change_mean", "qp_actual_V_change_mean"], ascending=[False, True, True]).head(24).copy()
    labels = [f"{row['scope']}|{row['direction_mode']}|n={row['lambda_N']:g}"[:34] for _, row in top.iterrows()]
    x = np.arange(len(top))

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - 0.18, top["nog_actual_V_change_mean"], width=0.36, label="noG actual merit")
    ax.bar(x + 0.18, top["qp_actual_V_change_mean"], width=0.36, label="QP actual merit")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right")
    ax.set_ylabel("Actual merit change")
    ax.set_title("Stage 4.8 Merit Change by Config")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stage48_merit_change_by_config.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - 0.18, top["nog_actual_C_change_mean"], width=0.36, label="noG actual cost change")
    ax.bar(x + 0.18, top["qp_actual_C_change_mean"], width=0.36, label="QP actual cost change")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right")
    ax.set_ylabel("Actual cost change")
    ax.set_title("Stage 4.8 Cost Change by Config")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stage48_C_change_by_config.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for direction_mode, group in summary_df.groupby("direction_mode"):
        ax.scatter(group["qp_actual_V_change_mean"], group["qp_actual_C_change_mean"], s=40, label=direction_mode, alpha=0.8)
    ax.set_xlabel("QP actual merit change")
    ax.set_ylabel("QP actual cost change")
    ax.set_title("Stage 4.8 Direction Mode Frontier")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stage48_direction_mode_frontier.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].boxplot([summary_df.loc[summary_df["direction_mode"] == dm, "qp_approx_kl_max"].dropna() for dm in DIR_MODES], labels=DIR_MODES)
    axes[0].set_title("QP approx_kl")
    axes[1].boxplot([summary_df.loc[summary_df["direction_mode"] == dm, "qp_clip_fraction_max"].dropna() for dm in DIR_MODES], labels=DIR_MODES)
    axes[1].set_title("QP clip_fraction")
    axes[2].boxplot([summary_df.loc[summary_df["direction_mode"] == dm, "qp_update_norm_post_cap_max"].dropna() for dm in DIR_MODES], labels=DIR_MODES)
    axes[2].set_title("QP update_norm_post_cap")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(plots_dir / "stage48_KL_clip_update.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    grouped = summary_df.groupby("direction_mode")[["qp_actor_fraction_of_update_mean", "qp_actor_fraction_of_V_decrease_mean"]].mean().reset_index()
    x = np.arange(len(grouped))
    ax.bar(x - 0.18, grouped["qp_actor_fraction_of_update_mean"], width=0.36, label="actor_fraction_of_update")
    ax.bar(x + 0.18, grouped["qp_actor_fraction_of_V_decrease_mean"], width=0.36, label="actor_fraction_of_V_decrease")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["direction_mode"], rotation=45, ha="right")
    ax.set_ylim(bottom=min(0.0, float(grouped[["qp_actor_fraction_of_update_mean", "qp_actor_fraction_of_V_decrease_mean"]].min().min()) - 0.1))
    ax.set_title("Stage 4.8 Actor Block Contribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stage48_actor_block_contribution.png", dpi=180)
    plt.close(fig)


def write_reports(summary_df: pd.DataFrame, output_root: pathlib.Path, config_count: int) -> None:
    direction_counts = summary_df.groupby("direction_mode")["stage48_pass"].sum().to_dict()
    best = summary_df.sort_values(["stage48_pass", "qp_actual_C_change_mean", "qp_actual_V_change_mean"], ascending=[False, True, True]).head(12)
    report_lines = [
        "# Stage 4.8 Preflight Report",
        "",
        f"- Config count: `{config_count}`",
        f"- Summary rows (config x direction): `{len(summary_df)}`",
        f"- Overall passing configs: `{int(summary_df['stage48_pass'].sum())}`",
        "",
        "## Direction Pass Counts",
        "",
    ]
    for direction_mode in DIR_MODES:
        report_lines.append(f"- `{direction_mode}`: `{int(direction_counts.get(direction_mode, 0))}` pass rows")
    report_lines.extend(["", "## Best Rows", ""])
    best_cols = [
        "config_label",
        "direction_mode",
        "scope",
        "cost_mode",
        "lambda_N",
        "qp_actual_V_change_mean",
        "nog_actual_V_change_mean",
        "qp_actual_C_change_mean",
        "nog_actual_C_change_mean",
        "qp_approx_kl_max",
        "qp_clip_fraction_max",
        "qp_actor_fraction_of_update_mean",
        "stage48_pass",
    ]
    report_lines.append(best[best_cols].to_csv(index=False))
    (output_root / "stage48_preflight_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    pass_rows = summary_df[summary_df["stage48_pass"]]
    root_lines = [
        "# Stage 4.8 Root Cause Report",
        "",
        f"- Passing rows: `{len(pass_rows)}`",
        f"- Direction modes tested: `{', '.join(DIR_MODES)}`",
        f"- Cost modes tested: `{', '.join(COST_MODES)}`",
        "",
        "## Main Readout",
        "",
    ]
    if pass_rows.empty:
        root_lines.extend(
            [
                "- No direction mode passed preflight yet.",
                "- This points to a merit-design conflict rather than a remaining solver bug.",
                f"- Best direction by mean actual cost change: `{summary_df.sort_values('qp_actual_C_change_mean').iloc[0]['direction_mode']}`",
                f"- Best direction by mean actual composite merit: `{summary_df.sort_values('qp_actual_V_change_mean').iloc[0]['direction_mode']}`",
            ]
        )
    else:
        top = pass_rows.sort_values(["qp_actual_C_change_mean", "qp_actual_V_change_mean"]).iloc[0]
        root_lines.extend(
            [
                f"- Best passing direction: `{top['direction_mode']}`",
                f"- Best passing config: `{top['config_label']}`",
                f"- Scope: `{top['scope']}`; cost mode: `{top['cost_mode']}`; lambda_N: `{top['lambda_N']}`",
            ]
        )
    (output_root / "stage48_root_cause_report.md").write_text("\n".join(root_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_root / "stage48_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    configs = build_configs(args.max_configs)
    model = build_model(args)
    all_rows: List[Dict[str, object]] = []
    for role in ("protagonist", "adversary"):
        algo = role_algo(model, role)
        probes = collect_probe_batches(model, role, max(args.num_probes_per_role, 4))
        for config in configs:
            diagnostics_csv_path = diagnostics_dir / f"{config.label}_{role}.csv"
            for train_probe, val_probe in pair_probes(probes):
                probe_rows = run_single_probe(
                    algo=algo,
                    role=role,
                    train_probe=train_probe,
                    val_probe=val_probe,
                    config=config,
                    args=args,
                    diagnostics_csv_path=diagnostics_csv_path,
                )
                all_rows.extend(probe_rows)

    detail_df = pd.DataFrame(all_rows)
    detail_df.to_csv(output_root / "stage48_direction_mode_comparison.csv", index=False)

    summary_rows: List[Dict[str, object]] = []
    val_rows: List[Dict[str, object]] = []
    for config in configs:
        for direction_mode in DIR_MODES:
            rows = detail_df[(detail_df["config_id"] == config.config_id) & (detail_df["direction_mode"] == direction_mode)]
            if rows.empty:
                continue
            summary = summarize_config_direction(config, direction_mode, rows)
            summary_rows.append(summary)
            for _, row in rows.iterrows():
                val_rows.append(
                    {
                        "config_id": config.config_id,
                        "config_label": config.label,
                        "direction_mode": direction_mode,
                        "scope": config.scope,
                        "cost_mode": config.cost_mode,
                        "lambda_N": config.lambda_N,
                        "method": row["method"],
                        "role": row["role"],
                        "train_probe_id": row["train_probe_id"],
                        "val_probe_id": row["val_probe_id"],
                        "train_actual_V_change": row["actual_V_change"],
                        "train_actual_C_change": row["actual_C_change"],
                        "val_actual_V_change": row["val_actual_V_change"],
                        "val_actual_C_change": row["val_actual_C_change"],
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / "stage48_preflight_summary.csv", index=False)
    pd.DataFrame(val_rows).to_csv(output_root / "stage48_validation_generalization.csv", index=False)
    plot_stage48(summary_df, output_root)
    write_reports(summary_df, output_root, len(configs))


if __name__ == "__main__":
    main()
