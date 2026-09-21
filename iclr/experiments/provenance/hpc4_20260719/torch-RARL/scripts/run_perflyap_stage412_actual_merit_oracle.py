from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from models.proposed_qp_new import apply_state_delta, flatten_named_tensors, named_difference, tensor_norm
from models.proposed_qp_perflyap import (
    ProposedQPPerfLyapOptimizer,
    block_norm,
)
from scripts.full_policy_optimizer_probe import (
    clone_state,
    collect_probe_batches,
    compute_loss_and_grads,
    egm_update,
    named_parameters,
    ppm_update,
    restore_state,
    sgd_update,
)
from scripts.run_perflyap_stage48_perf_audit import (
    build_eval_context,
    build_extended_eval_closure,
    grad_for_cost,
    grad_for_merit,
)
from scripts.run_perflyap_stage4_preflight import build_model, pair_probes, role_algo
from scripts.run_rawfg_M13_action_mean_audit import capture_env_state, make_raw_env, restore_env_state, unwrap_time_limit_and_base


@dataclass(frozen=True)
class OracleConfig:
    config_id: int
    scope: str
    lambda_N: float
    cost_mode: str
    beta_max: float
    gamma_max: float
    update_cap: float

    @property
    def label(self) -> str:
        return f"{self.scope}_n{self.lambda_N:g}_{self.cost_mode}_b{self.beta_max:g}_g{self.gamma_max:g}_cap{self.update_cap:g}"


MERIT_NAMES = [
    "actor_surrogate_cost",
    "unclipped_actor_surrogate_cost",
    "short_clean_return_cost",
    "short_rarl_return_cost",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 4.12 actual merit oracle sanity")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--baseline-summary", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eta-egm", type=float, default=1e-3)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--num-probes", type=int, default=4)
    parser.add_argument("--surrogate-beta-points", type=int, default=21)
    parser.add_argument("--surrogate-gamma-points", type=int, default=21)
    parser.add_argument("--return-beta-points", type=int, default=11)
    parser.add_argument("--return-gamma-points", type=int, default=11)
    parser.add_argument("--short-horizon", type=int, default=32)
    parser.add_argument("--return-episodes", type=int, default=3)
    parser.add_argument("--adv-strength", type=float, default=1.0)
    return parser.parse_args()


def build_configs() -> List[OracleConfig]:
    configs: List[OracleConfig] = []
    config_id = 0
    for scope in ["actor_mean_only", "actor_game", "actor_mean_heavy"]:
        for cost_mode in ["actor_surrogate_cost", "unclipped_actor_surrogate_cost"]:
            configs.append(
                OracleConfig(
                    config_id=config_id,
                    scope=scope,
                    lambda_N=0.0,
                    cost_mode=cost_mode,
                    beta_max=3e-2,
                    gamma_max=3e-5,
                    update_cap=0.005,
                )
            )
            config_id += 1
    return configs


def load_baseline_settings(path: pathlib.Path) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(path)
    out: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        out[str(row["method"])] = {
            "lr": float(row["protagonist_lr"]),
            "max_grad_norm": float(row["protagonist_max_grad_norm"]),
            "vf_coef": float(row["protagonist_vf_coef"]),
            "ppm_inner_steps": int(row["ppm_inner_steps"]) if int(row["ppm_inner_steps"]) > 0 else 0,
        }
    return out


def instantiate_helper(named_params, config: OracleConfig, args: argparse.Namespace):
    return ProposedQPPerfLyapOptimizer(
        [param for _, param in named_params],
        lr=1.0,
        perflyap_scope=config.scope,
        lambda_N=config.lambda_N,
        lambda_P=1.0,
        lambda_critic=0.0,
        logstd_weight=0.0,
        qp_fd_eps=1e-3,
        qp_beta_probe=1e-3,
        qp_gamma_probe=1e-6,
        qp_ridge=1e-8,
        qp_rho=1e-8,
        qp_beta_max=config.beta_max,
        qp_gamma_max=config.gamma_max,
        qp_max_update_norm=config.update_cap,
        qp_eps=1e-8,
        eta_egm_reference=args.eta_egm,
        g_sign_mode="auto_actual_preflight",
        role="protagonist",
        diagnostics_csv_path=None,
    )


def adam_update(theta_old: Dict[str, torch.Tensor], grads_old: Dict[str, torch.Tensor], lr: float, *, beta1: float, beta2: float, eps: float) -> Dict[str, torch.Tensor]:
    del beta1, beta2
    out: Dict[str, torch.Tensor] = {}
    for name, grad in grads_old.items():
        denom = grad.abs() + eps
        out[name] = theta_old[name] - lr * grad / denom
    return out


def evaluate_state(helper, eval_closure, theta_state, selected_names: Sequence[str]) -> Dict[str, object]:
    return helper._evaluate_state(
        eval_closure=eval_closure,
        theta_state=theta_state,
        selected_names=selected_names,
        backward=True,
        norm_scale=1.0,
        perf_scale=1.0,
    )


def apply_cap_to_update(update_map: Dict[str, torch.Tensor], selected_names: Sequence[str], cap: float, eps: float):
    update_vec = flatten_named_tensors(update_map, selected_names)
    norm_pre = tensor_norm(update_vec)
    scale = 1.0
    if np.isfinite(cap) and cap > 0.0 and norm_pre > cap:
        scale = cap / max(norm_pre, eps)
    capped = {name: tensor * scale for name, tensor in update_map.items()}
    norm_post = tensor_norm(flatten_named_tensors(capped, selected_names))
    return capped, float(norm_pre), float(norm_post), float(scale)


def protagonist_action(policy, obs: np.ndarray) -> np.ndarray:
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=policy.device)
    if obs_tensor.ndim == 1:
        obs_tensor = obs_tensor.unsqueeze(0)
    with torch.no_grad():
        action = policy._predict(obs_tensor, deterministic=True)
    return action.detach().cpu().numpy().squeeze(0)


def rollout_return_with_model(model, *, env_id: str, adv_strength: float, episodes: int, horizon: int, base_seed: int, control_adv: bool) -> Tuple[float, float]:
    raw_env = make_raw_env(env_id)
    returns: List[float] = []
    try:
        time_limit, base = unwrap_time_limit_and_base(raw_env)
        if control_adv:
            from utils.wrappers import AdversarialClassicControlWrapper

            raw_env.close()
            raw_env = AdversarialClassicControlWrapper(gym.make(env_id), adv_fraction=2.5, device=str(model.protagonist.policy.device))
            raw_env.operating_mode = "protagonist"
            raw_env._adv_policy = model.adversary.policy
            raw_env.adv_strength = float(adv_strength)
            time_limit, base = unwrap_time_limit_and_base(raw_env)

        for episode_id in range(episodes):
            obs, _ = raw_env.reset(seed=base_seed + episode_id)
            ep_ret = 0.0
            steps = 0
            done = False
            while not done and steps < horizon:
                action = protagonist_action(model.protagonist.policy, np.asarray(obs, dtype=np.float32))
                if isinstance(raw_env.action_space, gym.spaces.Box):
                    action = np.clip(action, raw_env.action_space.low, raw_env.action_space.high)
                obs, reward, terminated, truncated, _ = raw_env.step(action)
                ep_ret += float(reward)
                done = bool(terminated or truncated)
                steps += 1
            returns.append(ep_ret)
    finally:
        raw_env.close()
    mean_ret = float(np.mean(returns)) if returns else 0.0
    std_ret = float(np.std(returns)) if returns else 0.0
    return mean_ret, std_ret


def actual_merit_evaluator(
    merit_name: str,
    *,
    helper,
    eval_closure,
    selected_names: Sequence[str],
    model,
    env_id: str,
    episodes: int,
    horizon: int,
    base_seed: int,
) -> Callable[[Dict[str, torch.Tensor]], Dict[str, float]]:
    def eval_theta(theta_state: Dict[str, torch.Tensor]) -> Dict[str, float]:
        named_params = named_parameters(model.protagonist.policy)
        restore_state(named_params, theta_state)
        if merit_name == "actor_surrogate_cost":
            info = evaluate_state(helper, eval_closure, theta_state, selected_names)
            return {"merit": float(info["policy_component"]), "approx_kl": float(info["approx_kl"]), "clip_fraction": float(info["clip_fraction"])}
        if merit_name == "unclipped_actor_surrogate_cost":
            info = eval_closure(theta_override=theta_state, backward=True, grad_scope_names=list(selected_names))
            return {"merit": float(info["policy_loss_unclipped"]), "approx_kl": float(info["approx_kl"]), "clip_fraction": float(info["clip_fraction"])}
        if merit_name == "short_clean_return_cost":
            mean_ret, _ = rollout_return_with_model(model, env_id=env_id, adv_strength=0.0, episodes=episodes, horizon=horizon, base_seed=base_seed, control_adv=False)
            return {"merit": float(-mean_ret), "approx_kl": np.nan, "clip_fraction": np.nan}
        if merit_name == "short_rarl_return_cost":
            clean_ret, _ = rollout_return_with_model(model, env_id=env_id, adv_strength=0.0, episodes=episodes, horizon=horizon, base_seed=base_seed, control_adv=False)
            adv_ret, _ = rollout_return_with_model(model, env_id=env_id, adv_strength=1.0, episodes=episodes, horizon=horizon, base_seed=base_seed, control_adv=True)
            merit = -0.5 * clean_ret - 0.5 * adv_ret
            return {"merit": float(merit), "approx_kl": np.nan, "clip_fraction": np.nan}
        raise ValueError(merit_name)

    return eval_theta


def metric_row(
    *,
    helper,
    named_params,
    theta_old,
    theta_new,
    selected_names: Sequence[str],
    merit_name: str,
    merit_eval,
    train_state_before: Dict[str, float],
    config: OracleConfig,
    train_probe_id: int,
    val_probe_id: int,
    method: str,
    direction: str,
    beta: float | None,
    gamma: float | None,
    update_norm: float,
    gamma_active: int,
    fallback_to_noG: int,
) -> Dict[str, object]:
    before = merit_eval(theta_old)
    after = merit_eval(theta_new)
    restore_state(named_params, theta_old)
    all_names = [name for name, _ in named_params]
    diff = named_difference(theta_new, theta_old, all_names)
    actor_update_norm = block_norm(diff, all_names, "actor")
    logstd_update_norm = block_norm(diff, all_names, "logstd")
    critic_update_norm = block_norm(diff, all_names, "critic")
    return {
        "config_id": config.config_id,
        "config_label": config.label,
        "scope": config.scope,
        "lambda_N": config.lambda_N,
        "cost_mode": config.cost_mode,
        "merit_name": merit_name,
        "train_probe_id": train_probe_id,
        "val_probe_id": val_probe_id,
        "method": method,
        "direction": direction,
        "beta": float(beta) if beta is not None else np.nan,
        "gamma": float(gamma) if gamma is not None else np.nan,
        "update_norm": float(update_norm),
        "actual_merit_before": float(before["merit"]),
        "actual_merit_after": float(after["merit"]),
        "actual_merit_change": float(after["merit"] - before["merit"]),
        "actor_update_norm": actor_update_norm,
        "logstd_update_norm": logstd_update_norm,
        "critic_update_norm": critic_update_norm,
        "approx_kl": float(train_state_before["approx_kl"]) if np.isfinite(train_state_before["approx_kl"]) else float(after["approx_kl"]) if np.isfinite(after["approx_kl"]) else np.nan,
        "clip_fraction": float(train_state_before["clip_fraction"]) if np.isfinite(train_state_before["clip_fraction"]) else float(after["clip_fraction"]) if np.isfinite(after["clip_fraction"]) else np.nan,
        "gamma_active": int(gamma_active),
        "fallback_to_noG": int(fallback_to_noG),
    }


def direction_maps(helper, algo, train_probe, theta_old, selected_names, config: OracleConfig, args: argparse.Namespace):
    train_ctx = build_eval_context(algo, train_probe.rollout_data)
    train_eval = build_extended_eval_closure(train_ctx, config.cost_mode)
    base_train = evaluate_state(helper, train_eval, theta_old, selected_names)
    f_raw_map = {name: base_train["grads_selected"][name].detach().clone() for name in selected_names}
    g_plus_map, _ = helper._compute_g_raw(
        theta_old=theta_old,
        eval_closure=train_eval,
        selected_names=selected_names,
        f_raw_map=f_raw_map,
        norm_scale=1.0,
        perf_scale=1.0,
    )
    g_minus_map = {name: -g_plus_map[name] for name in selected_names}
    perf_grad_map = {name: -tensor for name, tensor in grad_for_cost(train_ctx, theta_old, selected_names, config.cost_mode, 1.0, helper.qp_eps).items()}
    merit_grad_map = {name: -tensor for name, tensor in grad_for_merit(helper, train_ctx, theta_old, selected_names, config.cost_mode, 1.0, 1.0).items()}
    named_params = named_parameters(algo.policy)
    old_eval = compute_loss_and_grads(algo, train_probe.rollout_data, max_grad_norm=1.0, vf_coef=1.0, ent_coef=args.ent_coef)
    theta_sgd = sgd_update(theta_old, old_eval["grads"], args.eta_egm)
    _, _, _, theta_egm = egm_update(algo, train_probe.rollout_data, theta_old, args.eta_egm, max_grad_norm=1.0, vf_coef=1.0, ent_coef=args.ent_coef)
    delta_egm = {name: theta_egm[name] - theta_old[name] for name in selected_names}
    delta_sgd = {name: theta_sgd[name] - theta_old[name] for name in selected_names}
    egm_actual_map = {
        name: (delta_egm[name] - delta_sgd[name]) / max(args.eta_egm * args.eta_egm, helper.qp_eps)
        for name in selected_names
    }
    return train_eval, base_train, f_raw_map, {
        "egm_plus_JF_F": g_plus_map,
        "egm_minus_JF_F": g_minus_map,
        "performance_grad": perf_grad_map,
        "merit_grad": merit_grad_map,
        "egm_actual_delta_direction": egm_actual_map,
    }


def no_g_grid(theta_old, f_raw_map, selected_names: Sequence[str], beta_max: float, beta_points: int, update_cap: float, eps: float):
    out = []
    for beta in np.linspace(0.0, beta_max, beta_points):
        update_map_pre = {name: -float(beta) * f_raw_map[name] for name in selected_names}
        update_map, norm_pre, norm_post, scale = apply_cap_to_update(update_map_pre, selected_names, update_cap, eps)
        beta_eff = float(beta) * scale
        theta_new = apply_state_delta(theta_old, update_map)
        out.append((float(beta), 0.0, beta_eff, 0.0, theta_new, norm_pre, norm_post))
    return out


def qp_grid(theta_old, f_raw_map, d_map, selected_names: Sequence[str], beta_max: float, gamma_max: float, beta_points: int, gamma_points: int, update_cap: float, eps: float):
    out = []
    for beta in np.linspace(0.0, beta_max, beta_points):
        for gamma in np.linspace(0.0, gamma_max, gamma_points):
            update_map_pre = {name: -float(beta) * f_raw_map[name] + float(gamma) * d_map[name] for name in selected_names}
            update_map, norm_pre, norm_post, scale = apply_cap_to_update(update_map_pre, selected_names, update_cap, eps)
            beta_eff = float(beta) * scale
            gamma_eff = float(gamma) * scale
            theta_new = apply_state_delta(theta_old, update_map)
            out.append((float(beta), float(gamma), beta_eff, gamma_eff, theta_new, norm_pre, norm_post))
    return out


def evaluate_oracle_candidates(
    *,
    helper,
    named_params,
    theta_old,
    selected_names: Sequence[str],
    merit_name: str,
    merit_eval,
    train_state_before: Dict[str, float],
    config: OracleConfig,
    train_probe_id: int,
    val_probe_id: int,
    f_raw_map,
    direction_dict,
    beta_points: int,
    gamma_points: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    no_g_candidates = no_g_grid(theta_old, f_raw_map, selected_names, config.beta_max, beta_points, config.update_cap, helper.qp_eps)
    no_g_rows = []
    for beta_raw, gamma_raw, beta_eff, gamma_eff, theta_new, norm_pre, norm_post in no_g_candidates:
        row = metric_row(
            helper=helper,
            named_params=named_params,
            theta_old=theta_old,
            theta_new=theta_new,
            selected_names=selected_names,
            merit_name=merit_name,
            merit_eval=merit_eval,
            train_state_before=train_state_before,
            config=config,
            train_probe_id=train_probe_id,
            val_probe_id=val_probe_id,
            method="noG_oracle",
            direction="noG",
            beta=beta_eff,
            gamma=gamma_eff,
            update_norm=norm_post,
            gamma_active=0,
            fallback_to_noG=0,
        )
        row["beta_raw"] = beta_raw
        row["gamma_raw"] = gamma_raw
        row["update_norm_pre_cap"] = norm_pre
        no_g_rows.append(row)
    best_no_g = min(no_g_rows, key=lambda r: r["actual_merit_change"])
    rows.extend(no_g_rows)

    qp_best_rows = []
    for direction_name, d_map in direction_dict.items():
        qp_rows = []
        for beta_raw, gamma_raw, beta_eff, gamma_eff, theta_new, norm_pre, norm_post in qp_grid(
            theta_old,
            f_raw_map,
            d_map,
            selected_names,
            config.beta_max,
            config.gamma_max,
            beta_points,
            gamma_points,
            config.update_cap,
            helper.qp_eps,
        ):
            row = metric_row(
                helper=helper,
                named_params=named_params,
                theta_old=theta_old,
                theta_new=theta_new,
                selected_names=selected_names,
                merit_name=merit_name,
                merit_eval=merit_eval,
                train_state_before=train_state_before,
                config=config,
                train_probe_id=train_probe_id,
                val_probe_id=val_probe_id,
                method="qp_oracle_grid",
                direction=direction_name,
                beta=beta_eff,
                gamma=gamma_eff,
                update_norm=norm_post,
                gamma_active=int(gamma_eff > helper.qp_eps),
                fallback_to_noG=0,
            )
            row["beta_raw"] = beta_raw
            row["gamma_raw"] = gamma_raw
            row["update_norm_pre_cap"] = norm_pre
            qp_rows.append(row)
        best_dir = min(qp_rows, key=lambda r: r["actual_merit_change"])
        qp_best_rows.append(best_dir)
        rows.extend(qp_rows)

    best_qp = min(qp_best_rows, key=lambda r: r["actual_merit_change"])
    rows.append(
        {
            **best_qp,
            "method": "qp_oracle_best",
            "fallback_to_noG": 0,
        }
    )
    safe_row = best_qp if best_qp["actual_merit_change"] <= best_no_g["actual_merit_change"] else best_no_g
    rows.append(
        {
            **safe_row,
            "method": "safe_qp_oracle",
            "direction": safe_row["direction"],
            "fallback_to_noG": int(best_qp["actual_merit_change"] > best_no_g["actual_merit_change"]),
        }
    )
    rows.append(
        {
            **best_no_g,
            "method": "noG_oracle_best",
            "direction": "noG",
            "fallback_to_noG": 0,
        }
    )
    rows.append(
        {
            **metric_row(
                helper=helper,
                named_params=named_params,
                theta_old=theta_old,
                theta_new=theta_old,
                selected_names=selected_names,
                merit_name=merit_name,
                merit_eval=merit_eval,
                train_state_before=train_state_before,
                config=config,
                train_probe_id=train_probe_id,
                val_probe_id=val_probe_id,
                method="zero",
                direction="zero",
                beta=0.0,
                gamma=0.0,
                update_norm=0.0,
                gamma_active=0,
                fallback_to_noG=0,
            ),
            "beta_raw": 0.0,
            "gamma_raw": 0.0,
            "update_norm_pre_cap": 0.0,
        }
    )
    return rows


def baseline_one_step_rows(
    *,
    model,
    algo,
    helper,
    named_params,
    theta_old,
    selected_names: Sequence[str],
    merit_name: str,
    merit_eval,
    train_state_before: Dict[str, float],
    config: OracleConfig,
    train_probe,
    val_probe,
    args: argparse.Namespace,
    baseline_settings,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    all_names = [name for name, _ in named_params]
    for method in ["sgd", "egm", "ppm", "adam"]:
        setting = baseline_settings[method]
        lr = float(setting["lr"])
        max_grad_norm = float(setting["max_grad_norm"])
        vf_coef = float(setting["vf_coef"])
        old_eval = compute_loss_and_grads(algo, train_probe.rollout_data, max_grad_norm=max_grad_norm, vf_coef=vf_coef, ent_coef=args.ent_coef)
        if method == "sgd":
            theta_new = sgd_update(theta_old, old_eval["grads"], lr)
        elif method == "egm":
            _, _, _, theta_new = egm_update(algo, train_probe.rollout_data, theta_old, lr, max_grad_norm=max_grad_norm, vf_coef=vf_coef, ent_coef=args.ent_coef)
        elif method == "ppm":
            inner_steps = max(int(setting["ppm_inner_steps"]), 1)
            _, theta_new, *_ = ppm_update(algo, train_probe.rollout_data, theta_old, lr, inner_steps=inner_steps, max_grad_norm=max_grad_norm, vf_coef=vf_coef, ent_coef=args.ent_coef)
        else:
            optimizer_group = algo.policy.optimizer.param_groups[0]
            beta1, beta2 = optimizer_group.get("betas", (0.9, 0.999))
            eps = float(optimizer_group.get("eps", 1e-8))
            theta_new = adam_update(theta_old, old_eval["grads"], lr, beta1=beta1, beta2=beta2, eps=eps)
        diff_all = named_difference(theta_new, theta_old, all_names)
        rows.append(
            metric_row(
                helper=helper,
                named_params=named_params,
                theta_old=theta_old,
                theta_new=theta_new,
                selected_names=selected_names,
                merit_name=merit_name,
                merit_eval=merit_eval,
                train_state_before=train_state_before,
                config=config,
                train_probe_id=int(train_probe.probe_idx),
                val_probe_id=int(val_probe.probe_idx),
                method=method,
                direction=method,
                beta=np.nan,
                gamma=np.nan,
                update_norm=tensor_norm(flatten_named_tensors(diff_all, all_names)),
                gamma_active=0,
                fallback_to_noG=0,
            )
        )
    return rows


def summarize(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["config_id", "config_label", "scope", "lambda_N", "cost_mode", "merit_name", "method", "direction"]
    for keys, group in detail_df.groupby(group_cols):
        config_id, config_label, scope, lambda_N, cost_mode, merit_name, method, direction = keys
        rows.append(
            {
                "config_id": int(config_id),
                "config_label": config_label,
                "scope": scope,
                "lambda_N": float(lambda_N),
                "cost_mode": cost_mode,
                "merit_name": merit_name,
                "method": method,
                "direction": direction,
                "rows": int(len(group)),
                "actual_merit_change_mean": float(group["actual_merit_change"].mean()),
                "actual_merit_change_min": float(group["actual_merit_change"].min()),
                "actual_C_like_change_mean": float(group["actual_merit_change"].mean()),
                "approx_kl_max": float(pd.to_numeric(group["approx_kl"], errors="coerce").max()),
                "clip_fraction_max": float(pd.to_numeric(group["clip_fraction"], errors="coerce").max()),
                "gamma_active_mean": float(pd.to_numeric(group["gamma_active"], errors="coerce").mean()),
                "fallback_to_noG_sum": int(pd.to_numeric(group["fallback_to_noG"], errors="coerce").fillna(0).sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_results(detail_df: pd.DataFrame, summary_df: pd.DataFrame, output_root: pathlib.Path) -> None:
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    focus = summary_df[summary_df["method"].isin(["noG_oracle_best", "qp_oracle_best", "safe_qp_oracle", "sgd", "egm", "ppm", "adam"])]
    fig, ax = plt.subplots(figsize=(10, 6))
    for merit_name, group in focus.groupby("merit_name"):
        sub = group[group["method"].isin(["noG_oracle_best", "qp_oracle_best", "safe_qp_oracle"])]
        ax.scatter(sub["actual_merit_change_mean"], sub["actual_C_like_change_mean"], label=merit_name, s=40, alpha=0.8)
    ax.set_xlabel("Actual merit change")
    ax.set_ylabel("Actual merit change")
    ax.set_title("Stage 4.12 actual merit frontier")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stage412_actual_merit_frontier.png", dpi=180)
    plt.close(fig)

    for merit_name in MERIT_NAMES:
        sub = focus[(focus["merit_name"] == merit_name) & (focus["method"].isin(["noG_oracle_best", "qp_oracle_best", "safe_qp_oracle"]))]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        ordered = sub.groupby("method")["actual_merit_change_mean"].mean().reindex(["noG_oracle_best", "qp_oracle_best", "safe_qp_oracle"])
        ax.bar(np.arange(len(ordered)), ordered.values)
        ax.set_xticks(np.arange(len(ordered)))
        ax.set_xticklabels(list(ordered.index), rotation=20, ha="right")
        ax.set_ylabel("Actual merit change")
        ax.set_title(f"Stage 4.12 QP vs noG actual merit: {merit_name}")
        fig.tight_layout()
        fig.savefig(plots_dir / "stage412_qp_vs_nog_actual_merit.png" if merit_name == MERIT_NAMES[0] else plots_dir / f"stage412_qp_vs_nog_actual_merit_{merit_name}.png", dpi=180)
        plt.close(fig)

    method_order = ["adam", "sgd", "egm", "ppm", "noG_oracle_best", "qp_oracle_best", "safe_qp_oracle"]
    agg = focus.groupby(["merit_name", "method"])["actual_merit_change_mean"].mean().reset_index()
    for merit_name, group in agg.groupby("merit_name"):
        ordered = group.set_index("method").reindex(method_order)
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(np.arange(len(method_order)), ordered["actual_merit_change_mean"])
        ax.set_xticks(np.arange(len(method_order)))
        ax.set_xticklabels(method_order, rotation=25, ha="right")
        ax.set_ylabel("Actual merit change")
        ax.set_title(f"Stage 4.12 one-step actual merit: {merit_name}")
        fig.tight_layout()
        fig.savefig(plots_dir / ("stage412_methods_one_step_actual_merit.png" if merit_name == MERIT_NAMES[0] else f"stage412_methods_one_step_actual_merit_{merit_name}.png"), dpi=180)
        plt.close(fig)

    gamma_df = focus[focus["method"].isin(["qp_oracle_best", "safe_qp_oracle"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    for merit_name, group in gamma_df.groupby("merit_name"):
        ax.bar(merit_name, float(group["gamma_active_mean"].mean()))
    ax.set_ylabel("Gamma active mean")
    ax.set_title("Stage 4.12 gamma usage")
    fig.tight_layout()
    fig.savefig(plots_dir / "stage412_gamma_usage.png", dpi=180)
    plt.close(fig)


def write_report(summary_df: pd.DataFrame, output_root: pathlib.Path) -> None:
    def merit_method_mean(merit_name: str, method: str) -> float:
        sub = summary_df[(summary_df["merit_name"] == merit_name) & (summary_df["method"] == method)]
        return float(sub["actual_merit_change_mean"].mean()) if not sub.empty else float("nan")

    lines = [
        "# Stage 4.12 Actual Merit Oracle Report",
        "",
        "- Scope of this oracle sanity: protagonist-side same-start audit.",
        "- Local merit selection uses actual post-update evaluation, not predicted q.",
        "- For rollout merits, seeds are shared across candidates and short horizon is fixed.",
        "",
    ]

    for merit_name in MERIT_NAMES:
        lines.append(f"## {merit_name}")
        lines.append("")
        cols = ["method", "direction", "actual_merit_change_mean", "approx_kl_max", "clip_fraction_max", "gamma_active_mean", "fallback_to_noG_sum"]
        sub = summary_df[summary_df["merit_name"] == merit_name]
        lines.append(sub[sub["method"].isin(["adam", "sgd", "egm", "ppm", "noG_oracle_best", "qp_oracle_best", "safe_qp_oracle"])][cols].sort_values("actual_merit_change_mean").to_csv(index=False))
        lines.append("")

    actor_qp = merit_method_mean("actor_surrogate_cost", "qp_oracle_best")
    actor_nog = merit_method_mean("actor_surrogate_cost", "noG_oracle_best")
    roll_qp = merit_method_mean("short_clean_return_cost", "qp_oracle_best")
    roll_nog = merit_method_mean("short_clean_return_cost", "noG_oracle_best")
    gamma_nonzero = float(summary_df[summary_df["method"] == "qp_oracle_best"]["gamma_active_mean"].mean())

    lines.extend(
        [
            "## Required Answers",
            "",
            f"1. Under actual actor surrogate merit, does QP oracle beat or equal noG? {'Yes' if actor_qp <= actor_nog else 'No'} (`QP={actor_qp:.6g}`, `noG={actor_nog:.6g}`).",
            f"2. Under actual rollout return merit, does QP oracle beat or equal noG? {'Yes' if roll_qp <= roll_nog else 'No'} (`QP={roll_qp:.6g}`, `noG={roll_nog:.6g}`).",
            f"3. If QP oracle still does not beat noG, is gamma always zero? {'No' if gamma_nonzero > 0 else 'Yes'} (gamma active mean `{gamma_nonzero:.3f}`).",
            "4. Which second direction actually helps actual merit? See per-merit `qp_oracle_best` direction rows in the summary table.",
            "5. Does noG oracle beat SGD/EGM/PPM one-step on the same merit? Check each merit table directly; this run reports all same-start one-step comparisons side by side.",
            "6. Does QP oracle beat SGD/EGM/PPM one-step on the same merit? Check each merit table directly; this run reports all same-start one-step comparisons side by side.",
            "7. If actual oracle QP beats noG but predicted QP does not, then the quadratic drift model is wrong. This run is purely actual-merit oracle, so compare with Stage 4.6/4.8 predicted results manually.",
            "8. If actual oracle QP does not beat noG, then no second direction adds value for this merit, and noG is locally optimal under the tested candidate family.",
        ]
    )
    (output_root / "stage412_actual_merit_oracle_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    baseline_settings = load_baseline_settings(pathlib.Path(args.baseline_summary))
    configs = build_configs()
    model = build_model(args)
    algo = role_algo(model, "protagonist")
    probes = collect_probe_batches(model, "protagonist", max(args.num_probes, 4))
    probe_pairs = pair_probes(probes)
    if not probe_pairs:
        raise RuntimeError("Need at least one train/validation probe pair for Stage 4.12.")
    train_probe, val_probe = probe_pairs[0]

    all_rows: List[Dict[str, object]] = []
    env_id = args.env

    for config in configs:
        named_params = named_parameters(algo.policy)
        helper = instantiate_helper(named_params, config, args)
        theta_old = clone_state(named_params)
        selected_names = helper._selected_names(named_params)
        train_eval, base_train, f_raw_map, direction_dict = direction_maps(helper, algo, train_probe, theta_old, selected_names, config, args)
        train_state_before = {
            "approx_kl": float(base_train["approx_kl"]),
            "clip_fraction": float(base_train["clip_fraction"]),
        }

        for merit_name in MERIT_NAMES:
            if merit_name.startswith("short_"):
                merit_eval = actual_merit_evaluator(
                    merit_name,
                    helper=helper,
                    eval_closure=train_eval,
                    selected_names=selected_names,
                    model=model,
                    env_id=env_id,
                    episodes=args.return_episodes,
                    horizon=args.short_horizon,
                    base_seed=args.seed + 1000 * config.config_id,
                )
                beta_points = args.return_beta_points
                gamma_points = args.return_gamma_points
            else:
                merit_eval = actual_merit_evaluator(
                    merit_name,
                    helper=helper,
                    eval_closure=train_eval,
                    selected_names=selected_names,
                    model=model,
                    env_id=env_id,
                    episodes=args.return_episodes,
                    horizon=args.short_horizon,
                    base_seed=args.seed + 1000 * config.config_id,
                )
                beta_points = args.surrogate_beta_points
                gamma_points = args.surrogate_gamma_points

            all_rows.extend(
                evaluate_oracle_candidates(
                    helper=helper,
                    named_params=named_params,
                    theta_old=theta_old,
                    selected_names=selected_names,
                    merit_name=merit_name,
                    merit_eval=merit_eval,
                    train_state_before=train_state_before,
                    config=config,
                    train_probe_id=int(train_probe.probe_idx),
                    val_probe_id=int(val_probe.probe_idx),
                    f_raw_map=f_raw_map,
                    direction_dict=direction_dict,
                    beta_points=beta_points,
                    gamma_points=gamma_points,
                )
            )
            all_rows.extend(
                baseline_one_step_rows(
                    model=model,
                    algo=algo,
                    helper=helper,
                    named_params=named_params,
                    theta_old=theta_old,
                    selected_names=selected_names,
                    merit_name=merit_name,
                    merit_eval=merit_eval,
                    train_state_before=train_state_before,
                    config=config,
                    train_probe=train_probe,
                    val_probe=val_probe,
                    args=args,
                    baseline_settings=baseline_settings,
                )
            )
            restore_state(named_params, theta_old)

    detail_df = pd.DataFrame(all_rows)
    detail_df.to_csv(output_root / "stage412_actual_merit_oracle_detail.csv", index=False)
    summary_df = summarize(detail_df)
    summary_df.to_csv(output_root / "stage412_actual_merit_oracle_summary.csv", index=False)
    plot_results(detail_df, summary_df, output_root)
    write_report(summary_df, output_root)


if __name__ == "__main__":
    main()
