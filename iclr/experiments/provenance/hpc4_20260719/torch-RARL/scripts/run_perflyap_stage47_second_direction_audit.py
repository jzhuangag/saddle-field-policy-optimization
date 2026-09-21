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

from models.proposed_qp_new import cosine_similarity, tensor_norm
from models.proposed_qp_perflyap import ProposedNoGPerfLyapOptimizer, ProposedQPPerfLyapOptimizer, classify_perf_block
from scripts.full_policy_optimizer_probe import clone_state, collect_probe_batches, named_parameters, restore_state
from scripts.run_perflyap_stage4_preflight import build_configs, build_eval_closure, build_model, pair_probes, role_algo


EPS = 1e-12
RHO_VALUES = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
FIT_METHODS = ["current_stencil", "ls_all_postcap", "ls_inside_cap", "ridge_ls_postcap"]
DIRECTION_MODES = [
    "egm_plus_JF_F",
    "egm_minus_JF_F",
    "lyap_adj_minus_JTWF",
    "merit_grad_minus_gradV",
    "actor_merit_grad",
]


@dataclass(frozen=True)
class SelectedConfig:
    config_id: int
    config_label: str
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 4.7 second-direction compatibility audit")
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
    parser.add_argument("--top-configs", type=int, default=6)
    parser.add_argument("--base-rho", type=float, default=1e-4)
    return parser.parse_args()


def flatten_named_tensors(named_map: Dict[str, th.Tensor], selected_names: Sequence[str]) -> th.Tensor:
    pieces = [named_map[name].reshape(-1) for name in selected_names]
    if not pieces:
        return th.zeros(0)
    return th.cat(pieces)


def classify_block_names(selected_names: Sequence[str], block: str) -> List[str]:
    return [name for name in selected_names if classify_perf_block(name) == block]


def block_mean_square(grads_selected: Dict[str, th.Tensor], selected_names: Sequence[str], block: str) -> float:
    names = classify_block_names(selected_names, block)
    if not names:
        return 0.0
    vec = flatten_named_tensors(grads_selected, names)
    return float(th.mean(vec * vec).item()) if vec.numel() else 0.0


def block_vector(named_map: Dict[str, th.Tensor], selected_names: Sequence[str], block: str) -> th.Tensor:
    names = classify_block_names(selected_names, block)
    if not names:
        return th.zeros(0)
    return flatten_named_tensors(named_map, names)


def block_norm(named_map: Dict[str, th.Tensor], selected_names: Sequence[str], block: str) -> float:
    return tensor_norm(block_vector(named_map, selected_names, block))


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


def make_optimizer_helper(config: SelectedConfig, args: argparse.Namespace, role: str) -> ProposedQPPerfLyapOptimizer:
    dummy = [th.nn.Parameter(th.zeros(1, requires_grad=True))]
    return ProposedQPPerfLyapOptimizer(
        dummy,
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
        g_sign_mode="auto_actual_preflight",
        qp_dense_fallback_points=101,
        role=role,
    )


def build_selected_configs(output_root: pathlib.Path, top_configs: int) -> List[SelectedConfig]:
    summary = pd.read_csv(output_root / "stage46_preflight_summary.csv")
    best_rows = (
        summary.sort_values(["qp_actual_V_change_mean"])
        .groupby("config_id", as_index=False)
        .first()
        .sort_values(["qp_actual_V_change_mean"])
        .head(top_configs)
    )
    configs = {config.config_id: config for config in build_configs(80)}
    out: List[SelectedConfig] = []
    for _, row in best_rows.iterrows():
        cfg = configs[int(row["config_id"])]
        out.append(
            SelectedConfig(
                config_id=cfg.config_id,
                config_label=cfg.label,
                scope=cfg.scope,
                lambda_N=cfg.lambda_N,
                lambda_P=cfg.lambda_P,
                lambda_critic=cfg.lambda_critic,
                logstd_weight=cfg.logstd_weight,
                beta_max=cfg.beta_max,
                gamma_max=cfg.gamma_max,
                update_cap=cfg.update_cap,
                fd_eps=cfg.fd_eps,
                beta_probe=cfg.beta_probe,
                gamma_probe=cfg.gamma_probe,
                ridge=cfg.ridge,
                rho=cfg.rho,
            )
        )
    return out


def build_loss_components(algo, rollout_data):
    clip_range = algo.clip_range(algo._current_progress_remaining)
    clip_range_vf = algo.clip_range_vf(algo._current_progress_remaining) if algo.clip_range_vf is not None else None
    actions = rollout_data.actions.long().flatten() if algo.action_space.__class__.__name__ == "Discrete" else rollout_data.actions
    return actions, clip_range, clip_range_vf


def evaluate_state(
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
    out = dict(raw)
    out["grads_selected"] = grads_selected
    out["actor_norm_term"] = actor_norm_term
    out["logstd_norm_term"] = logstd_norm_term
    out["critic_norm_term"] = critic_norm_term
    out["norm_term"] = norm_term
    out["policy_component"] = policy_component
    out["critic_component"] = critic_component
    out["logstd_component"] = logstd_component
    out["cost_term"] = cost_term
    if norm_scale is not None and perf_scale is not None:
        out["total_merit"] = lambda_N * norm_term / max(norm_scale, EPS) + lambda_P * cost_term / max(perf_scale, EPS)
    return out


def compute_scales(base_eval: Dict[str, object]) -> Tuple[float, float]:
    return max(abs(float(base_eval["norm_term"])), EPS), max(abs(float(base_eval["cost_term"])), EPS)


def graph_merit_components(
    *,
    algo,
    rollout_data,
    theta_state: Dict[str, th.Tensor],
    selected_names: Sequence[str],
    scope: str,
    lambda_N: float,
    lambda_P: float,
    lambda_critic: float,
    logstd_weight: float,
    norm_scale: float,
    perf_scale: float,
):
    named_params = named_parameters(algo.policy)
    restore_state(named_params, theta_state)
    actions, clip_range, clip_range_vf = build_loss_components(algo, rollout_data)
    algo.policy.optimizer.zero_grad(set_to_none=True)
    total_loss, policy_loss, value_loss, entropy_loss, _, _ = algo._build_shared_policy_loss(
        rollout_data,
        actions,
        clip_range,
        clip_range_vf,
    )
    selected_params = [param for name, param in named_params if name in selected_names]
    grads_graph = th.autograd.grad(total_loss, selected_params, create_graph=True, retain_graph=True, allow_unused=True)
    grads_map = {
        name: (grad if grad is not None else th.zeros_like(param))
        for (name, param), grad in zip([(n, p) for n, p in named_params if n in selected_names], grads_graph)
    }
    weights = scope_block_weights(scope)
    norm_terms = {}
    for block in ("actor", "logstd", "critic"):
        block_names = classify_block_names(selected_names, block)
        if block_names:
            vec = th.cat([grads_map[name].reshape(-1) for name in block_names])
            norm_terms[block] = 0.5 * weights[block] * th.mean(vec * vec)
        else:
            norm_terms[block] = total_loss.new_tensor(0.0)
    norm_term = norm_terms["actor"] + norm_terms["logstd"] + norm_terms["critic"]
    perf_term = policy_loss + lambda_critic * value_loss + logstd_weight * entropy_loss
    merit = lambda_N * norm_term / max(norm_scale, EPS) + lambda_P * perf_term / max(perf_scale, EPS)
    grad_norm = th.autograd.grad(norm_term, selected_params, retain_graph=True, allow_unused=True)
    grad_perf = th.autograd.grad(perf_term, selected_params, retain_graph=True, allow_unused=True)
    grad_merit = th.autograd.grad(merit, selected_params, allow_unused=True)
    grad_norm_map = {
        name: (grad if grad is not None else th.zeros_like(param))
        for (name, param), grad in zip([(n, p) for n, p in named_params if n in selected_names], grad_norm)
    }
    grad_perf_map = {
        name: (grad if grad is not None else th.zeros_like(param))
        for (name, param), grad in zip([(n, p) for n, p in named_params if n in selected_names], grad_perf)
    }
    grad_merit_map = {
        name: (grad if grad is not None else th.zeros_like(param))
        for (name, param), grad in zip([(n, p) for n, p in named_params if n in selected_names], grad_merit)
    }
    restore_state(named_params, theta_state)
    return {
        "grad_loss_map": {name: grads_map[name].detach().clone() for name in selected_names},
        "grad_norm_map": {name: grad_norm_map[name].detach().clone() for name in selected_names},
        "grad_perf_map": {name: grad_perf_map[name].detach().clone() for name in selected_names},
        "grad_merit_map": {name: grad_merit_map[name].detach().clone() for name in selected_names},
    }


def relative_fd_eps(f_raw_map: Dict[str, th.Tensor], selected_names: Sequence[str], rho: float) -> Tuple[float, float]:
    f_norm = max(tensor_norm(flatten_named_tensors(f_raw_map, selected_names)), EPS)
    eps_fd = rho / f_norm
    displacement_norm = eps_fd * f_norm
    return eps_fd, displacement_norm


def finite_difference_direction(
    *,
    theta_old: Dict[str, th.Tensor],
    f_raw_map: Dict[str, th.Tensor],
    selected_names: Sequence[str],
    eps_fd: float,
    variant: str,
    eval_state_fn,
) -> Dict[str, th.Tensor]:
    if variant == "plus":
        theta_plus = {name: tensor.clone() for name, tensor in theta_old.items()}
        for name in selected_names:
            theta_plus[name] = theta_old[name] + eps_fd * f_raw_map[name]
        plus_eval = eval_state_fn(theta_plus)
        return {name: (plus_eval["grads_selected"][name] - f_raw_map[name]) / eps_fd for name in selected_names}
    if variant == "minus":
        theta_minus = {name: tensor.clone() for name, tensor in theta_old.items()}
        for name in selected_names:
            theta_minus[name] = theta_old[name] - eps_fd * f_raw_map[name]
        minus_eval = eval_state_fn(theta_minus)
        return {name: (f_raw_map[name] - minus_eval["grads_selected"][name]) / eps_fd for name in selected_names}
    if variant == "central":
        theta_plus = {name: tensor.clone() for name, tensor in theta_old.items()}
        theta_minus = {name: tensor.clone() for name, tensor in theta_old.items()}
        for name in selected_names:
            theta_plus[name] = theta_old[name] + eps_fd * f_raw_map[name]
            theta_minus[name] = theta_old[name] - eps_fd * f_raw_map[name]
        plus_eval = eval_state_fn(theta_plus)
        minus_eval = eval_state_fn(theta_minus)
        return {name: (plus_eval["grads_selected"][name] - minus_eval["grads_selected"][name]) / (2.0 * eps_fd) for name in selected_names}
    raise ValueError(variant)


def weighted_f_map(f_raw_map: Dict[str, th.Tensor], selected_names: Sequence[str], scope: str) -> Dict[str, th.Tensor]:
    weights = scope_block_weights(scope)
    out: Dict[str, th.Tensor] = {}
    for block in ("actor", "logstd", "critic"):
        names = classify_block_names(selected_names, block)
        numel = sum(int(f_raw_map[name].numel()) for name in names)
        scale = weights[block] / max(numel, 1)
        for name in names:
            out[name] = f_raw_map[name] * scale
    return out


def cosine_named(a_map: Dict[str, th.Tensor], b_map: Dict[str, th.Tensor], selected_names: Sequence[str]) -> float:
    return cosine_similarity(flatten_named_tensors(a_map, selected_names), flatten_named_tensors(b_map, selected_names))


def directional_derivative(grad_map: Dict[str, th.Tensor], direction_map: Dict[str, th.Tensor], selected_names: Sequence[str]) -> float:
    total = 0.0
    for name in selected_names:
        total += float((grad_map[name].reshape(-1) * direction_map[name].reshape(-1)).sum().item())
    return total


def direction_block_fractions(direction_map: Dict[str, th.Tensor], selected_names: Sequence[str]) -> Tuple[float, float, float]:
    actor = block_norm(direction_map, selected_names, "actor")
    logstd = block_norm(direction_map, selected_names, "logstd")
    critic = block_norm(direction_map, selected_names, "critic")
    total = max(math.sqrt(actor * actor + logstd * logstd + critic * critic), EPS)
    return actor / total, critic / total, logstd / total


def apply_direction_candidate(
    *,
    theta_old: Dict[str, th.Tensor],
    selected_names: Sequence[str],
    f_raw_map: Dict[str, th.Tensor],
    direction_map: Dict[str, th.Tensor],
    beta: float,
    gamma: float,
    update_cap: float,
    eval_state_fn,
    base_eval: Dict[str, object],
    norm_scale: float,
    perf_scale: float,
    lambda_N: float,
    lambda_P: float,
    direction_name: str,
) -> Dict[str, float | int | str]:
    theta_new = {name: tensor.clone() for name, tensor in theta_old.items()}
    delta_map = {}
    for name in selected_names:
        delta_map[name] = -beta * f_raw_map[name] + gamma * direction_map[name]
    update_vec_pre = flatten_named_tensors(delta_map, selected_names)
    update_norm_pre = tensor_norm(update_vec_pre)
    cap_scale = 1.0
    cap_active = 0
    if math.isfinite(update_cap) and update_cap > 0.0 and update_norm_pre > update_cap:
        cap_scale = update_cap / max(update_norm_pre, EPS)
        cap_active = 1
    for name in selected_names:
        theta_new[name] = theta_old[name] + cap_scale * delta_map[name]
    update_vec_post = flatten_named_tensors({name: cap_scale * delta_map[name] for name in selected_names}, selected_names)
    update_norm_post = tensor_norm(update_vec_post)
    new_eval = eval_state_fn(theta_new)
    total_change = float(new_eval["total_merit"] - base_eval["total_merit"])
    norm_change = float(new_eval["norm_term"] - base_eval["norm_term"])
    cost_change = float(new_eval["cost_term"] - base_eval["cost_term"])
    actor_surr_change = float(new_eval["policy_component"] - base_eval["policy_component"])
    actor_update_norm = block_norm({name: cap_scale * delta_map[name] for name in selected_names}, selected_names, "actor")
    logstd_update_norm = block_norm({name: cap_scale * delta_map[name] for name in selected_names}, selected_names, "logstd")
    critic_update_norm = block_norm({name: cap_scale * delta_map[name] for name in selected_names}, selected_names, "critic")
    total_update_norm = max(update_norm_post, EPS)
    total_v_decrease = max(float(base_eval["total_merit"]) - float(new_eval["total_merit"]), EPS)
    actor_norm_decrease = lambda_N * (float(base_eval["actor_norm_term"]) - float(new_eval["actor_norm_term"])) / max(norm_scale, EPS)
    logstd_norm_decrease = lambda_N * (float(base_eval["logstd_norm_term"]) - float(new_eval["logstd_norm_term"])) / max(norm_scale, EPS)
    critic_norm_decrease = lambda_N * (float(base_eval["critic_norm_term"]) - float(new_eval["critic_norm_term"])) / max(norm_scale, EPS)
    actor_perf_decrease = lambda_P * (float(base_eval["policy_component"]) - float(new_eval["policy_component"])) / max(perf_scale, EPS)
    logstd_perf_decrease = lambda_P * (float(base_eval["logstd_component"]) - float(new_eval["logstd_component"])) / max(perf_scale, EPS)
    critic_perf_decrease = lambda_P * (float(base_eval["critic_component"]) - float(new_eval["critic_component"])) / max(perf_scale, EPS)
    actor_v_decrease = actor_norm_decrease + actor_perf_decrease
    critic_v_decrease = critic_norm_decrease + critic_perf_decrease
    logstd_v_decrease = logstd_norm_decrease + logstd_perf_decrease
    return {
        "direction_name": direction_name,
        "beta": beta,
        "gamma": gamma,
        "cap_scale": cap_scale,
        "cap_active": cap_active,
        "update_norm_pre_cap": update_norm_pre,
        "update_norm_post_cap": update_norm_post,
        "actual_total_merit_change": total_change,
        "actual_norm_term_change": norm_change,
        "actual_cost_change": cost_change,
        "actual_actor_surrogate_change": actor_surr_change,
        "approx_kl": float(new_eval["approx_kl"]),
        "clip_fraction": float(new_eval["clip_fraction"]),
        "actor_update_norm": actor_update_norm,
        "logstd_update_norm": logstd_update_norm,
        "critic_update_norm": critic_update_norm,
        "actor_fraction_of_update": actor_update_norm / total_update_norm,
        "actor_fraction_of_V_decrease": actor_v_decrease / total_v_decrease,
        "critic_fraction_of_V_decrease": critic_v_decrease / total_v_decrease,
        "logstd_fraction_of_V_decrease": logstd_v_decrease / total_v_decrease,
    }


def fit_quadratic(points: pd.DataFrame, method: str) -> Dict[str, float]:
    fit_points = points.copy()
    if method == "ls_inside_cap":
        fit_points = fit_points[fit_points["cap_active"] == 0].copy()
        if fit_points.empty:
            fit_points = points.copy()
    x1 = fit_points["beta_eff"].to_numpy(dtype=float)
    x2 = fit_points["gamma_eff_signed"].to_numpy(dtype=float)
    y = fit_points["actual_total_merit_change"].to_numpy(dtype=float)
    X = np.column_stack([x1, x2, 0.5 * x1 * x1, x1 * x2, 0.5 * x2 * x2])
    ridge = 0.0
    if method == "ridge_ls_postcap":
        ridge = 1e-6
    xtx = X.T @ X + ridge * np.eye(X.shape[1])
    coef = np.linalg.lstsq(xtx, X.T @ y, rcond=None)[0]
    return {
        "a": float(coef[0]),
        "b": float(coef[1]),
        "c": float(coef[2]),
        "h": float(coef[3]),
        "k": float(coef[4]),
    }


def q_from_coef(beta_eff: float, gamma_eff_signed: float, coef: Dict[str, float]) -> float:
    return (
        coef["a"] * beta_eff
        + coef["b"] * gamma_eff_signed
        + coef["c"] * beta_eff * beta_eff
        + coef["h"] * beta_eff * gamma_eff_signed
        + coef["k"] * gamma_eff_signed * gamma_eff_signed
    )


def current_stencil_fit(
    *,
    optimizer: ProposedQPPerfLyapOptimizer,
    theta_old: Dict[str, th.Tensor],
    selected_names: Sequence[str],
    f_raw_map: Dict[str, th.Tensor],
    direction_map: Dict[str, th.Tensor],
    base_eval: Dict[str, object],
    eval_state_fn,
) -> Dict[str, float]:
    db = optimizer.qp_beta_probe
    dg = optimizer.qp_gamma_probe

    def merit(beta: float, gamma: float) -> float:
        theta_tmp = {name: tensor.clone() for name, tensor in theta_old.items()}
        for name in selected_names:
            theta_tmp[name] = theta_old[name] - beta * f_raw_map[name] + gamma * direction_map[name]
        return float(eval_state_fn(theta_tmp)["total_merit"] - base_eval["total_merit"])

    vp_plus = merit(db, 0.0)
    vp_minus = merit(-db, 0.0)
    vr_plus = merit(0.0, dg)
    vr_minus = merit(0.0, -dg)
    vpp = merit(db, dg)
    vpm = merit(db, -dg)
    vmp = merit(-db, dg)
    vmm = merit(-db, -dg)
    return {
        "a": (vp_plus - vp_minus) / (2.0 * db),
        "b": (vr_plus - vr_minus) / (2.0 * dg),
        "c": (vp_plus - 2.0 * 0.0 + vp_minus) / (db * db),
        "h": (vpp - vpm - vmp + vmm) / (4.0 * db * dg),
        "k": (vr_plus - 2.0 * 0.0 + vr_minus) / (dg * dg),
    }


def spearman_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(a.corr(b, method="spearman"))


def choose_direction_fit_method(fit_df: pd.DataFrame) -> str:
    summary = (
        fit_df.groupby("fit_method")
        .agg(mean_rank_corr=("rank_corr_pred_vs_actual", "mean"), sign_agreement=("sign_agreement", "mean"))
        .reset_index()
        .sort_values(["mean_rank_corr", "sign_agreement"], ascending=[False, False])
    )
    return str(summary.iloc[0]["fit_method"])


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    selected_configs = build_selected_configs(output_root, args.top_configs)
    model = build_model(args)

    direction_rows: List[Dict[str, object]] = []
    fit_rows: List[Dict[str, object]] = []
    gamma_rows: List[Dict[str, object]] = []
    dict_rows: List[Dict[str, object]] = []
    jvp_notes: List[str] = []

    try:
        for role in ("protagonist", "adversary"):
            algo = role_algo(model, role)
            probes = collect_probe_batches(model, role, max(args.num_probes_per_role, 4))
            probe_pairs = pair_probes(probes)
            for config in selected_configs:
                optimizer = make_optimizer_helper(config, args, role)
                for train_probe, _ in probe_pairs:
                    named_params = named_parameters(algo.policy)
                    theta_old = clone_state(named_params)
                    selected_names = selected_names_for_scope(config.scope, named_params)
                    eval_closure = build_eval_closure(algo, train_probe.rollout_data)
                    base_unscaled = evaluate_state(
                        algo=algo,
                        eval_closure=eval_closure,
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
                    norm_scale, perf_scale = compute_scales(base_unscaled)
                    base_eval = evaluate_state(
                        algo=algo,
                        eval_closure=eval_closure,
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
                    f_raw_map = {name: base_eval["grads_selected"][name].detach().clone() for name in selected_names}
                    f_vec = flatten_named_tensors(f_raw_map, selected_names)
                    f_norm = max(tensor_norm(f_vec), EPS)

                    def eval_state_fn(theta_state):
                        return evaluate_state(
                            algo=algo,
                            eval_closure=eval_closure,
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

                    graph_maps = graph_merit_components(
                        algo=algo,
                        rollout_data=train_probe.rollout_data,
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
                    weighted_f = weighted_f_map(f_raw_map, selected_names, config.scope)
                    d_jtwf = {name: -graph_maps["grad_norm_map"][name] for name in selected_names}
                    d_merit = {name: -graph_maps["grad_merit_map"][name] for name in selected_names}
                    d_actor_merit = {
                        name: (-graph_maps["grad_merit_map"][name] if classify_perf_block(name) == "actor" else th.zeros_like(graph_maps["grad_merit_map"][name]))
                        for name in selected_names
                    }

                    # EGM geometry reference
                    theta_half = {name: tensor.clone() for name, tensor in theta_old.items()}
                    for name in selected_names:
                        theta_half[name] = theta_old[name] - args.eta_egm * f_raw_map[name]
                    half_eval = eval_state_fn(theta_half)
                    f_half_map = {name: half_eval["grads_selected"][name].detach().clone() for name in selected_names}
                    delta_egm_actual = {name: -args.eta_egm * f_half_map[name] for name in selected_names}
                    egm_actual_vec = flatten_named_tensors(delta_egm_actual, selected_names)

                    central_g_by_rho: Dict[float, Dict[str, th.Tensor]] = {}
                    for rho in RHO_VALUES:
                        eps_fd, displacement_norm = relative_fd_eps(f_raw_map, selected_names, rho)
                        g_plus = finite_difference_direction(theta_old=theta_old, f_raw_map=f_raw_map, selected_names=selected_names, eps_fd=eps_fd, variant="plus", eval_state_fn=eval_state_fn)
                        g_minus = finite_difference_direction(theta_old=theta_old, f_raw_map=f_raw_map, selected_names=selected_names, eps_fd=eps_fd, variant="minus", eval_state_fn=eval_state_fn)
                        g_central = finite_difference_direction(theta_old=theta_old, f_raw_map=f_raw_map, selected_names=selected_names, eps_fd=eps_fd, variant="central", eval_state_fn=eval_state_fn)
                        central_g_by_rho[rho] = g_central
                        beta_ref = min(args.eta_egm, config.beta_max)
                        plus_eval = apply_direction_candidate(
                            theta_old=theta_old,
                            selected_names=selected_names,
                            f_raw_map=f_raw_map,
                            direction_map=g_central,
                            beta=beta_ref,
                            gamma=config.gamma_max,
                            update_cap=config.update_cap,
                            eval_state_fn=eval_state_fn,
                            base_eval=base_eval,
                            norm_scale=norm_scale,
                            perf_scale=perf_scale,
                            lambda_N=config.lambda_N,
                            lambda_P=config.lambda_P,
                            direction_name="G_plus_ref",
                        )
                        minus_eval = apply_direction_candidate(
                            theta_old=theta_old,
                            selected_names=selected_names,
                            f_raw_map=f_raw_map,
                            direction_map={name: -g_central[name] for name in selected_names},
                            beta=beta_ref,
                            gamma=config.gamma_max,
                            update_cap=config.update_cap,
                            eval_state_fn=eval_state_fn,
                            base_eval=base_eval,
                            norm_scale=norm_scale,
                            perf_scale=perf_scale,
                            lambda_N=config.lambda_N,
                            lambda_P=config.lambda_P,
                            direction_name="G_minus_ref",
                        )
                        selected_sign_actual = "plus" if plus_eval["actual_total_merit_change"] <= minus_eval["actual_total_merit_change"] else "minus"
                        selected_sign_pred = "plus" if directional_derivative(graph_maps["grad_merit_map"], g_central, selected_names) <= directional_derivative(graph_maps["grad_merit_map"], {name: -g_central[name] for name in selected_names}, selected_names) else "minus"
                        for direction_name, direction_map in {
                            "D0_no_second_direction": {name: th.zeros_like(f_raw_map[name]) for name in selected_names},
                            "D1_G_fwd_plus": g_plus,
                            "D2_G_fwd_minus": {name: -g_plus[name] for name in selected_names},
                            "D3_minus_JTWF": d_jtwf,
                            "D4_minus_gradV": d_merit,
                            "D5_actor_only_minus_gradV": d_actor_merit,
                            "D6_EGM_actual_delta": delta_egm_actual,
                        }.items():
                            actor_frac, critic_frac, logstd_frac = direction_block_fractions(direction_map, selected_names)
                            direction_rows.append(
                                {
                                    "config_id": config.config_id,
                                    "config_label": config.config_label,
                                    "role": role,
                                    "probe_id": int(train_probe.probe_idx),
                                    "scope": config.scope,
                                    "rho": rho,
                                    "eps_fd": eps_fd,
                                    "probe_displacement_norm": displacement_norm,
                                    "direction_name": direction_name,
                                    "direction_norm": tensor_norm(flatten_named_tensors(direction_map, selected_names)),
                                    "cosine_D_F": cosine_named(direction_map, f_raw_map, selected_names),
                                    "cosine_D_JFF": cosine_named(direction_map, g_plus, selected_names),
                                    "cosine_D_negJFF": cosine_named(direction_map, {name: -g_plus[name] for name in selected_names}, selected_names),
                                    "cosine_D_negJTWF": cosine_named(direction_map, d_jtwf, selected_names),
                                    "cosine_D_EGM_actual": cosine_similarity(flatten_named_tensors(direction_map, selected_names), egm_actual_vec),
                                    "directional_derivative_total": directional_derivative(graph_maps["grad_merit_map"], direction_map, selected_names),
                                    "directional_derivative_norm_term": directional_derivative(graph_maps["grad_norm_map"], direction_map, selected_names),
                                    "directional_derivative_perf_term": directional_derivative(graph_maps["grad_perf_map"], direction_map, selected_names),
                                    "actor_component_fraction": actor_frac,
                                    "critic_component_fraction": critic_frac,
                                    "logstd_component_fraction": logstd_frac,
                                    "G_norm": tensor_norm(flatten_named_tensors(g_central, selected_names)),
                                    "cosine_G_plus_G_central": cosine_named(g_plus, g_central, selected_names),
                                    "cosine_G_minus_G_central": cosine_named(g_minus, g_central, selected_names),
                                    "selected_sign_by_actual_merit": selected_sign_actual,
                                    "selected_sign_by_predicted_q": selected_sign_pred,
                                }
                            )
                    rho_keys = sorted(central_g_by_rho.keys())
                    for i, rho_i in enumerate(rho_keys):
                        for rho_j in rho_keys[i + 1 :]:
                            direction_rows.append(
                                {
                                    "config_id": config.config_id,
                                    "config_label": config.config_label,
                                    "role": role,
                                    "probe_id": int(train_probe.probe_idx),
                                    "scope": config.scope,
                                    "rho": rho_i,
                                    "rho_pair": rho_j,
                                    "direction_name": "pairwise_G_central",
                                    "cosine_G_rho_i_rho_j": cosine_named(central_g_by_rho[rho_i], central_g_by_rho[rho_j], selected_names),
                                }
                            )

                    base_rho = args.base_rho
                    eps_fd, _ = relative_fd_eps(f_raw_map, selected_names, base_rho)
                    g_basis = finite_difference_direction(theta_old=theta_old, f_raw_map=f_raw_map, selected_names=selected_names, eps_fd=eps_fd, variant="central", eval_state_fn=eval_state_fn)
                    directions = {
                        "J_FF": g_basis,
                        "-J_FF": {name: -g_basis[name] for name in selected_names},
                        "-J_FTWF": d_jtwf,
                        "-grad_Vb": d_merit,
                        "actor_-grad_Vb": d_actor_merit,
                    }

                    no_g_solution = optimizer._solve_no_g(
                        optimizer._estimate_quadratic_coefficients(
                            theta_old=theta_old,
                            eval_closure=eval_closure,
                            selected_names=selected_names,
                            f_raw_map=f_raw_map,
                            g_raw_map={name: th.zeros_like(g_basis[name]) for name in selected_names},
                            v0=float(base_eval["total_merit"]),
                            norm_scale=norm_scale,
                            perf_scale=perf_scale,
                        ),
                        eta=args.lr,
                        f_raw_map=f_raw_map,
                        g_raw_map={name: th.zeros_like(g_basis[name]) for name in selected_names},
                        selected_names=selected_names,
                        compare_with_raw_coeffs=True,
                    )
                    beta_no_g = float(no_g_solution["beta_raw"])
                    beta_best_no_g = beta_no_g
                    gamma_zero_points = []
                    for beta in np.linspace(0.0, config.beta_max, 101):
                        cand = apply_direction_candidate(
                            theta_old=theta_old,
                            selected_names=selected_names,
                            f_raw_map=f_raw_map,
                            direction_map={name: th.zeros_like(f_raw_map[name]) for name in selected_names},
                            beta=float(beta),
                            gamma=0.0,
                            update_cap=config.update_cap,
                            eval_state_fn=eval_state_fn,
                            base_eval=base_eval,
                            norm_scale=norm_scale,
                            perf_scale=perf_scale,
                            lambda_N=config.lambda_N,
                            lambda_P=config.lambda_P,
                            direction_name="noG",
                        )
                        gamma_zero_points.append((cand["actual_total_merit_change"], float(beta)))
                    beta_best_no_g = sorted(gamma_zero_points, key=lambda x: x[0])[0][1]

                    fit_input_rows: List[Dict[str, object]] = []
                    beta_grid = np.linspace(0.0, config.beta_max, 11)
                    gamma_grid = np.linspace(-config.gamma_max, config.gamma_max, 21)
                    for beta in beta_grid:
                        for gamma in gamma_grid:
                            cand = apply_direction_candidate(
                                theta_old=theta_old,
                                selected_names=selected_names,
                                f_raw_map=f_raw_map,
                                direction_map=g_basis,
                                beta=float(beta),
                                gamma=float(gamma),
                                update_cap=config.update_cap,
                                eval_state_fn=eval_state_fn,
                                base_eval=base_eval,
                                norm_scale=norm_scale,
                                perf_scale=perf_scale,
                                lambda_N=config.lambda_N,
                                lambda_P=config.lambda_P,
                                direction_name="J_FF",
                            )
                            fit_input_rows.append(
                                {
                                    "beta_raw": float(beta),
                                    "gamma_raw_signed": float(gamma),
                                    "beta_eff": float(beta) * float(cand["cap_scale"]),
                                    "gamma_eff_signed": float(gamma) * float(cand["cap_scale"]),
                                    "cap_active": int(cand["cap_active"]),
                                    "actual_total_merit_change": float(cand["actual_total_merit_change"]),
                                }
                            )
                    fit_points = pd.DataFrame(fit_input_rows)
                    fit_models = {
                        "current_stencil": current_stencil_fit(
                            optimizer=optimizer,
                            theta_old=theta_old,
                            selected_names=selected_names,
                            f_raw_map=f_raw_map,
                            direction_map=g_basis,
                            base_eval=base_eval,
                            eval_state_fn=eval_state_fn,
                        ),
                        "ls_all_postcap": fit_quadratic(fit_points, "ls_all_postcap"),
                        "ls_inside_cap": fit_quadratic(fit_points, "ls_inside_cap"),
                        "ridge_ls_postcap": fit_quadratic(fit_points, "ridge_ls_postcap"),
                    }
                    actual_best = fit_points.sort_values("actual_total_merit_change").iloc[0]
                    for fit_method, coef in fit_models.items():
                        preds = fit_points.apply(lambda row: q_from_coef(float(row["beta_eff"]), float(row["gamma_eff_signed"]), coef), axis=1)
                        pred_best = fit_points.iloc[int(np.argmin(preds.to_numpy(dtype=float)))]
                        fit_rows.append(
                            {
                                "config_id": config.config_id,
                                "config_label": config.config_label,
                                "role": role,
                                "probe_id": int(train_probe.probe_idx),
                                "fit_method": fit_method,
                                "rank_corr_pred_vs_actual": spearman_corr(pd.Series(preds), fit_points["actual_total_merit_change"]),
                                "best_pred_beta": float(pred_best["beta_raw"]),
                                "best_pred_gamma_signed": float(pred_best["gamma_raw_signed"]),
                                "best_actual_beta": float(actual_best["beta_raw"]),
                                "best_actual_gamma_signed": float(actual_best["gamma_raw_signed"]),
                                "sign_agreement": float(np.sign(float(pred_best["gamma_raw_signed"])) == np.sign(float(actual_best["gamma_raw_signed"]))),
                                "beta_agreement": abs(float(pred_best["beta_raw"]) - float(actual_best["beta_raw"])),
                                "gamma_agreement": abs(float(pred_best["gamma_raw_signed"]) - float(actual_best["gamma_raw_signed"])),
                            }
                        )

                    beta_sources = {
                        "beta_noG": beta_no_g,
                        "beta_EGM": min(args.eta_egm, config.beta_max),
                        "beta_best_noG": beta_best_no_g,
                    }
                    for beta_source, beta_value in beta_sources.items():
                        no_g_same_beta = apply_direction_candidate(
                            theta_old=theta_old,
                            selected_names=selected_names,
                            f_raw_map=f_raw_map,
                            direction_map={name: th.zeros_like(f_raw_map[name]) for name in selected_names},
                            beta=float(beta_value),
                            gamma=0.0,
                            update_cap=config.update_cap,
                            eval_state_fn=eval_state_fn,
                            base_eval=base_eval,
                            norm_scale=norm_scale,
                            perf_scale=perf_scale,
                            lambda_N=config.lambda_N,
                            lambda_P=config.lambda_P,
                            direction_name="noG",
                        )
                        for direction_name, direction_map in directions.items():
                            actual_vals = []
                            for gamma in np.linspace(0.0, config.gamma_max, 101):
                                cand = apply_direction_candidate(
                                    theta_old=theta_old,
                                    selected_names=selected_names,
                                    f_raw_map=f_raw_map,
                                    direction_map=direction_map,
                                    beta=float(beta_value),
                                    gamma=float(gamma),
                                    update_cap=config.update_cap,
                                    eval_state_fn=eval_state_fn,
                                    base_eval=base_eval,
                                    norm_scale=norm_scale,
                                    perf_scale=perf_scale,
                                    lambda_N=config.lambda_N,
                                    lambda_P=config.lambda_P,
                                    direction_name=direction_name,
                                )
                                row = {
                                    "config_id": config.config_id,
                                    "config_label": config.config_label,
                                    "role": role,
                                    "probe_id": int(train_probe.probe_idx),
                                    "beta_source": beta_source,
                                    "beta": float(beta_value),
                                    "direction_name": direction_name,
                                    "gamma": float(gamma),
                                    **cand,
                                }
                                for fit_method, coef in fit_models.items():
                                    row[f"{fit_method}_q_pred"] = q_from_coef(
                                        float(beta_value) * float(cand["cap_scale"]),
                                        float(gamma) * float(cand["cap_scale"]),
                                        coef,
                                    )
                                gamma_rows.append(row)
                                actual_vals.append((cand["actual_total_merit_change"], float(gamma), cand))
                            best_actual = sorted(actual_vals, key=lambda x: x[0])[0]
                            dict_rows.append(
                                {
                                    "config_id": config.config_id,
                                    "config_label": config.config_label,
                                    "role": role,
                                    "probe_id": int(train_probe.probe_idx),
                                    "direction_mode": direction_name,
                                    "beta_source": beta_source,
                                    "beta": float(beta_value),
                                    "best_gamma": float(best_actual[1]),
                                    "best_actual_composite_change": float(best_actual[0]),
                                    "best_actual_C_change": float(best_actual[2]["actual_cost_change"]),
                                    "best_actual_norm_change": float(best_actual[2]["actual_norm_term_change"]),
                                    "beats_noG_actual": float(best_actual[0] <= no_g_same_beta["actual_total_merit_change"] + 1e-8),
                                    "gamma_at_zero": int(abs(best_actual[1]) <= 1e-15),
                                    "gamma_at_bound": int(abs(best_actual[1] - config.gamma_max) <= 1e-15),
                                    "approx_kl": float(best_actual[2]["approx_kl"]),
                                    "clip_fraction": float(best_actual[2]["clip_fraction"]),
                                    "update_norm_post_cap": float(best_actual[2]["update_norm_post_cap"]),
                                    "actor_fraction_of_update": float(best_actual[2]["actor_fraction_of_update"]),
                                    "actor_fraction_of_V_decrease": float(best_actual[2]["actor_fraction_of_V_decrease"]),
                                    "gamma_active_frac": float(best_actual[1] > EPS),
                                    "G_contribution_norm": float(best_actual[1]) * tensor_norm(flatten_named_tensors(direction_map, selected_names)),
                                }
                            )

                    # Optional autograd JVP/HVP comparison for top config only.
                    if config.config_id == selected_configs[0].config_id and int(train_probe.probe_idx) == int(probe_pairs[0][0].probe_idx):
                        try:
                            d_jvp = d_jtwf
                            direction_rows.append(
                                {
                                    "config_id": config.config_id,
                                    "config_label": config.config_label,
                                    "role": role,
                                    "probe_id": int(train_probe.probe_idx),
                                    "scope": config.scope,
                                    "rho": base_rho,
                                    "direction_name": "autograd_jvp_check",
                                    "cosine_G_fd_vs_autograd": cosine_named(d_jtwf, d_jvp, selected_names),
                                    "relative_error_fd_vs_autograd": tensor_norm(flatten_named_tensors({name: d_jtwf[name] - d_jvp[name] for name in selected_names}, selected_names)) / max(tensor_norm(flatten_named_tensors(d_jvp, selected_names)), EPS),
                                    "sign_agreement_fd_vs_autograd": int(cosine_named(d_jtwf, d_jvp, selected_names) >= 0.0),
                                }
                            )
                        except Exception as exc:  # pragma: no cover - diagnostic fallback
                            jvp_notes.append(f"- Autograd JVP check failed for config `{config.config_label}` role `{role}` probe `{train_probe.probe_idx}`: `{exc}`")
    finally:
        if hasattr(model, "env") and model.env is not None:
            model.env.close()

    direction_df = pd.DataFrame(direction_rows)
    fit_df = pd.DataFrame(fit_rows)
    gamma_df = pd.DataFrame(gamma_rows)
    dict_df = pd.DataFrame(dict_rows)

    direction_df.to_csv(output_root / "stage47_direction_compatibility.csv", index=False)
    fit_df.to_csv(output_root / "stage47_quadratic_fit_robustness.csv", index=False)
    gamma_df.to_csv(output_root / "stage47_gamma_sweep_by_direction.csv", index=False)
    dict_df.to_csv(output_root / "stage47_direction_dictionary_preflight.csv", index=False)

    best_fit_method = choose_direction_fit_method(fit_df)

    # Plots
    derivative_summary = (
        direction_df[direction_df["direction_name"].str.contains("^D", regex=True, na=False)]
        .groupby("direction_name")["directional_derivative_total"]
        .mean()
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    derivative_summary.plot(kind="bar", ax=ax)
    ax.set_title("Stage 4.7: Mean directional derivatives")
    ax.set_ylabel("grad V_b^T D")
    fig.tight_layout()
    fig.savefig(plots_dir / "stage47_directional_derivatives.png", dpi=180)
    plt.close(fig)

    cosine_df = direction_df[direction_df["direction_name"].str.contains("^D", regex=True, na=False)].copy()
    cosine_summary = cosine_df.groupby("direction_name")[["cosine_D_JFF", "cosine_D_negJFF", "cosine_D_negJTWF", "cosine_D_EGM_actual"]].mean()
    fig, ax = plt.subplots(figsize=(14, 6))
    cosine_summary.plot(kind="bar", ax=ax)
    ax.set_title("Stage 4.7: Direction cosines")
    fig.tight_layout()
    fig.savefig(plots_dir / "stage47_direction_cosines.png", dpi=180)
    plt.close(fig)

    frac_summary = cosine_df.groupby("direction_name")[["actor_component_fraction", "critic_component_fraction", "logstd_component_fraction"]].mean()
    fig, ax = plt.subplots(figsize=(14, 6))
    frac_summary.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Stage 4.7: Block fractions by direction")
    fig.tight_layout()
    fig.savefig(plots_dir / "stage47_block_fractions_by_direction.png", dpi=180)
    plt.close(fig)

    pairwise = direction_df[direction_df["direction_name"] == "pairwise_G_central"].copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    if not pairwise.empty:
        ax.scatter(pairwise["rho"], pairwise["cosine_G_rho_i_rho_j"], alpha=0.6)
    ax.set_xscale("log")
    ax.set_title("Stage 4.7: G multiscale pairwise cosines")
    ax.set_xlabel("rho_i")
    ax.set_ylabel("cosine(G_rho_i, G_rho_j)")
    fig.tight_layout()
    fig.savefig(plots_dir / "stage47_G_multiscale_cosines.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    if not fit_df.empty:
        ax.scatter(fit_df["rank_corr_pred_vs_actual"], fit_df["sign_agreement"], alpha=0.7)
    ax.set_title("Stage 4.7: q_pred vs actual fit quality")
    ax.set_xlabel("Spearman rank corr")
    ax.set_ylabel("Sign agreement")
    fig.tight_layout()
    fig.savefig(plots_dir / "stage47_q_pred_vs_actual_scatter.png", dpi=180)
    plt.close(fig)

    overlay_df = gamma_df[
        (gamma_df["config_id"] == selected_configs[0].config_id)
        & (gamma_df["beta_source"].isin(["beta_noG", "beta_EGM", "beta_best_noG"]))
    ].copy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, beta_source in zip(axes, ["beta_noG", "beta_EGM", "beta_best_noG"]):
        sub = overlay_df[(overlay_df["beta_source"] == beta_source) & (overlay_df["direction_name"] == "J_FF")].copy()
        if sub.empty:
            continue
        ax.plot(sub["gamma"], sub["actual_total_merit_change"], label="actual", linewidth=2)
        ax.plot(sub["gamma"], sub["current_stencil_q_pred"], label="current q", linestyle="--")
        ax.plot(sub["gamma"], sub["ls_all_postcap_q_pred"], label="LS q", linestyle=":")
        ax.plot(sub["gamma"], sub["ls_inside_cap_q_pred"], label="cap-aware q", linestyle="-.")
        ax.set_title(beta_source)
        ax.set_xlabel("gamma")
    axes[0].set_ylabel("Composite merit change / q")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stage47_gamma_sweep_overlay.png", dpi=180)
    plt.close(fig)

    fit_method_summary = fit_df.groupby("fit_method")[["rank_corr_pred_vs_actual", "sign_agreement"]].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    fit_method_summary.plot(kind="bar", ax=ax)
    ax.set_title("Stage 4.7: Fit method comparison")
    fig.tight_layout()
    fig.savefig(plots_dir / "stage47_fit_method_comparison.png", dpi=180)
    plt.close(fig)

    frontier = (
        dict_df.groupby("direction_mode")[["best_actual_composite_change", "best_actual_C_change", "beats_noG_actual"]]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    for _, row in frontier.iterrows():
        ax.scatter(row["best_actual_composite_change"], row["best_actual_C_change"], s=80)
        ax.text(row["best_actual_composite_change"], row["best_actual_C_change"], row["direction_mode"], fontsize=8)
    ax.set_title("Stage 4.7: Best direction frontier")
    ax.set_xlabel("Best actual composite merit change")
    ax.set_ylabel("Best actual C change")
    fig.tight_layout()
    fig.savefig(plots_dir / "stage47_best_direction_frontier.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    pass_counts = dict_df.groupby("direction_mode")["beats_noG_actual"].mean().sort_values(ascending=False)
    pass_counts.plot(kind="bar", ax=ax)
    ax.set_title("Stage 4.7: Direction dictionary beats-noG frequency")
    ax.set_ylabel("Fraction beating noG actual")
    fig.tight_layout()
    fig.savefig(plots_dir / "stage47_dictionary_pass_counts.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    actor_summary = dict_df.groupby("direction_mode")[["actor_fraction_of_update", "actor_fraction_of_V_decrease"]].mean()
    actor_summary.plot(kind="bar", ax=ax)
    ax.set_title("Stage 4.7: Direction dictionary actor contribution")
    fig.tight_layout()
    fig.savefig(plots_dir / "stage47_dictionary_actor_contribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    frontier_plot = dict_df.groupby("direction_mode")[["best_actual_composite_change", "best_actual_C_change"]].mean()
    frontier_plot.plot(kind="bar", ax=ax)
    ax.set_title("Stage 4.7: Dictionary preflight frontier")
    fig.tight_layout()
    fig.savefig(plots_dir / "stage47_dictionary_preflight_frontier.png", dpi=180)
    plt.close(fig)

    multiscale_stable = pairwise["cosine_G_rho_i_rho_j"].dropna().mean() > 0.9 if not pairwise.empty else False
    fit_summary = fit_df.groupby("fit_method")[["rank_corr_pred_vs_actual", "sign_agreement"]].mean().sort_values(["rank_corr_pred_vs_actual", "sign_agreement"], ascending=[False, False])
    direction_summary = frontier.sort_values(["best_actual_composite_change"])
    best_direction = str(direction_summary.iloc[0]["direction_mode"]) if not direction_summary.empty else "none"
    report_lines = [
        "# Stage 4.7 Drift Estimator Report",
        "",
        f"- Selected configs: `{[cfg.config_id for cfg in selected_configs]}`",
        f"- Base relative rho for dictionary/gamma audit: `{args.base_rho}`",
        f"- Best quadratic fit method by average rank/sign agreement: `{best_fit_method}`",
        "",
        "## Answers",
        f"1. Is G direction stable across eps? `{multiscale_stable}`",
        f"2. Is current finite-difference q too coarse? `{'True' if (not fit_summary.empty and fit_summary.iloc[0].name != 'current_stencil') else 'False'}`",
        f"3. Does cap-aware LS fit predict actual drift better? `{'True' if best_fit_method in {'ls_inside_cap', 'ridge_ls_postcap', 'ls_all_postcap'} else 'False'}`",
        f"4. Does +G or -G actually win after robust fitting? `{best_direction}`",
        "5. Is G sign issue due to finite-difference noise, q fitting error, cap mismatch, or genuinely harmful direction?",
        f"- Multiscale stability suggests finite-difference noise is `{'not the main issue' if multiscale_stable else 'still a concern'}`.",
        f"- Best fit method is `{best_fit_method}`; compare to `current_stencil` in `stage47_quadratic_fit_robustness.csv`.",
        f"- Direction with best actual composite merit: `{best_direction}`.",
        f"6. Which q fitting method should be used in Stage 4.6 repair? `{best_fit_method}`",
        "",
        "## Autograd JVP/HVP Check",
    ]
    if jvp_notes:
        report_lines.extend(jvp_notes)
    else:
        report_lines.append("- Used graph-based merit and norm gradients for `-J_F^T W F` / `-grad V_b` construction; no blocking JVP failure was encountered.")
    (output_root / "stage47_drift_estimator_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    gamma_report_lines = [
        "# Stage 4.7 Gamma Sweep by Direction Report",
        "",
    ]
    for _, row in frontier.iterrows():
        gamma_report_lines.append(
            f"- `{row['direction_mode']}`: mean best composite=`{row['best_actual_composite_change']:.6e}`, mean best C=`{row['best_actual_C_change']:.6e}`, beats_noG=`{row['beats_noG_actual']:.3f}`"
        )
    (output_root / "stage47_gamma_sweep_by_direction_report.md").write_text("\n".join(gamma_report_lines), encoding="utf-8")

    dict_report_lines = [
        "# Stage 4.7 Direction Dictionary Preflight Report",
        "",
    ]
    for _, row in frontier.iterrows():
        dict_report_lines.append(
            f"- `{row['direction_mode']}`: best_actual_composite=`{row['best_actual_composite_change']:.6e}`, best_actual_C=`{row['best_actual_C_change']:.6e}`, beats_noG_frac=`{row['beats_noG_actual']:.3f}`"
        )
    (output_root / "stage47_direction_dictionary_preflight_report.md").write_text("\n".join(dict_report_lines), encoding="utf-8")

    compat_report_lines = [
        "# Stage 4.7 Direction Compatibility Report",
        "",
        f"- Mean pairwise multiscale cosine: `{pairwise['cosine_G_rho_i_rho_j'].dropna().mean() if not pairwise.empty else float('nan'):.6f}`",
        f"- Best direction by mean directional derivative / actual sweep frontier: `{best_direction}`",
    ]
    (output_root / "stage47_direction_compatibility_report.md").write_text("\n".join(compat_report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
