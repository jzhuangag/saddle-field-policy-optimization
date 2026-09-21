from __future__ import annotations

import csv
import math
import os
import pathlib
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.optim import Optimizer

from models.proposed_qp_new import (
    NamedParams,
    NamedTensorMap,
    _block_mean_square,
    _is_finite_named_map,
    apply_state_delta,
    clone_named_state,
    cosine_similarity,
    flatten_named_tensors,
    named_difference,
    restore_named_state,
    tensor_norm,
)


def classify_perf_block(name: str) -> str:
    if "log_std" in name:
        return "logstd"
    if "value" in name:
        return "critic"
    return "actor"


def block_names(selected_names: Sequence[str], block: str) -> List[str]:
    return [name for name in selected_names if classify_perf_block(name) == block]


def block_vector(named_map: NamedTensorMap, selected_names: Sequence[str], block: str) -> torch.Tensor:
    names = block_names(selected_names, block)
    if not names:
        return torch.zeros(0)
    return flatten_named_tensors(named_map, names)


def block_norm(named_map: NamedTensorMap, selected_names: Sequence[str], block: str) -> float:
    return tensor_norm(block_vector(named_map, selected_names, block))


def _windows_safe_path(raw_path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(raw_path)
    if os.name != "nt":
        return path
    path_str = str(path)
    if path_str.startswith("\\\\?\\"):
        return path
    if len(path_str) < 240:
        return path
    resolved = str(path.resolve())
    if resolved.startswith("\\\\"):
        return pathlib.Path("\\\\?\\UNC\\" + resolved.lstrip("\\"))
    return pathlib.Path("\\\\?\\" + resolved)


class _PerformanceAlignedLyapunovQPBase(Optimizer):
    requires_eval_closure = True

    _VALID_SCOPES = {
        "actor_mean_only",
        "actor_game",
        "actor_mean_heavy",
        "critic_downweighted",
        "logstd_excluded",
        "full_policy_actor_weighted",
        "actor_mean_plus_adv_actor_mean",
    }
    _VALID_G_SIGN_MODES = {
        "plus",
        "minus",
        "auto_predicted",
        "auto_actual_preflight",
    }
    _VALID_COST_MODES = {
        "actor_surrogate_cost",
        "unclipped_actor_surrogate_cost",
        "mixed_clean_unclipped_actor_surrogate_cost",
        "mixed_rarl_unclipped_actor_surrogate_cost",
    }
    _VALID_SELECTOR_MODES = {
        "current_q_pred",
        "fixed_nog",
        "actual_surrogate_selector",
        "actual_mixed_selector",
        "ls_capaware",
        "safe_fixed_minusg",
    }
    _VALID_DIRECTION_MODES = {
        "egm_plus_JF_F",
        "egm_minus_JF_F",
        "performance_grad",
    }

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1.0,
        perflyap_scope: str = "actor_mean_only",
        lambda_N: float = 0.1,
        lambda_P: float = 1.0,
        lambda_critic: float = 0.0,
        logstd_weight: float = 0.0,
        qp_fd_eps: float = 1e-3,
        qp_beta_probe: float = 1e-3,
        qp_gamma_probe: float = 1e-6,
        qp_ridge: float = 1e-8,
        qp_rho: float = 1e-8,
        qp_beta_max: float = 1e-2,
        qp_gamma_max: float = 3e-5,
        qp_max_update_norm: float = 0.003,
        qp_eps: float = 1e-8,
        eta_egm_reference: float = 1e-3,
        g_sign_mode: str = "plus",
        qp_dense_fallback_points: int = 0,
        diagnostics_csv_path: Optional[str] = None,
        role: str = "policy",
        disable_g: bool = False,
        scale_ema_decay: float = 0.9,
        use_scale_normalization: bool = True,
        cost_mode: str = "actor_surrogate_cost",
        selector_mode: str = "current_q_pred",
        direction_mode: str = "egm_plus_JF_F",
        fixed_beta_raw: Optional[float] = None,
        fixed_gamma_raw: Optional[float] = None,
        selector_beta_grid: str = "",
        selector_gamma_grid: str = "",
        ls_fit_variant: str = "LS_all_grid",
        short_return_horizon: int = 16,
        short_return_episodes: int = 1,
        short_return_seed_offset: int = 0,
    ):
        super().__init__(params, defaults=dict(lr=lr))
        if perflyap_scope not in self._VALID_SCOPES:
            raise ValueError(f"Unsupported perflyap_scope={perflyap_scope!r}")
        if g_sign_mode not in self._VALID_G_SIGN_MODES:
            raise ValueError(f"Unsupported g_sign_mode={g_sign_mode!r}")
        if cost_mode not in self._VALID_COST_MODES:
            raise ValueError(f"Unsupported cost_mode={cost_mode!r}")
        if selector_mode not in self._VALID_SELECTOR_MODES:
            raise ValueError(f"Unsupported selector_mode={selector_mode!r}")
        if direction_mode not in self._VALID_DIRECTION_MODES:
            raise ValueError(f"Unsupported direction_mode={direction_mode!r}")
        self.optimizer_scope = "full_policy"
        self.perflyap_scope = perflyap_scope
        self.lambda_N = float(lambda_N)
        self.lambda_P = float(lambda_P)
        self.lambda_critic = float(lambda_critic)
        self.logstd_weight = float(logstd_weight)
        self.qp_fd_eps = max(float(qp_fd_eps), 1e-12)
        self.qp_beta_probe = max(float(qp_beta_probe), 1e-12)
        self.qp_gamma_probe = max(float(qp_gamma_probe), 1e-12)
        self.qp_ridge = max(float(qp_ridge), 0.0)
        self.qp_rho = max(float(qp_rho), 0.0)
        self.qp_beta_max = max(float(qp_beta_max), 0.0)
        self.qp_gamma_max = max(float(qp_gamma_max), 0.0)
        self.qp_max_update_norm = float(qp_max_update_norm)
        self.qp_eps = max(float(qp_eps), 1e-12)
        self.eta_egm_reference = max(float(eta_egm_reference), 1e-12)
        self.g_sign_mode = g_sign_mode
        self.qp_dense_fallback_points = max(int(qp_dense_fallback_points), 0)
        self.diagnostics_csv_path = diagnostics_csv_path
        self.role = role
        self.disable_g = bool(disable_g)
        self.scale_ema_decay = min(max(float(scale_ema_decay), 0.0), 1.0)
        self.use_scale_normalization = bool(use_scale_normalization)
        self.cost_mode = cost_mode
        self.selector_mode = selector_mode
        self.direction_mode = direction_mode
        self.fixed_beta_raw = None if fixed_beta_raw is None else float(fixed_beta_raw)
        self.fixed_gamma_raw = None if fixed_gamma_raw is None else float(fixed_gamma_raw)
        self.selector_beta_grid = self._parse_grid(selector_beta_grid, fallback=[0.0, 0.009, 0.015, self.qp_beta_max])
        self.selector_gamma_grid = self._parse_grid(selector_gamma_grid, fallback=[0.0, 1.5e-5, self.qp_gamma_max])
        self.ls_fit_variant = str(ls_fit_variant)
        self.short_return_horizon = max(int(short_return_horizon), 1)
        self.short_return_episodes = max(int(short_return_episodes), 1)
        self.short_return_seed_offset = int(short_return_seed_offset)
        self._norm_scale_ema: Optional[float] = None
        self._perf_scale_ema: Optional[float] = None
        self.last_step_metrics: Dict[str, float | int | str] = {}
        self._step_index = 0
        self._fieldnames = [
            "step_index",
            "active_role",
            "scope",
            "num_actor_params",
            "num_logstd_params",
            "num_critic_params",
            "lambda_N",
            "lambda_P",
            "lambda_critic",
            "logstd_weight",
            "fd_eps",
            "finite_difference_valid",
            "actor_F_norm",
            "logstd_F_norm",
            "critic_F_norm",
            "F_raw_norm",
            "G_raw_norm",
            "cosine_F_G",
            "l_beta",
            "l_gamma",
            "H_bb",
            "H_bg",
            "H_gg",
            "eig_min",
            "ridge_added",
            "q_pred",
            "q_pred_pre_cap",
            "q_pred_post_cap",
            "beta",
            "gamma",
            "beta_raw",
            "gamma_raw",
            "beta_eff",
            "gamma_eff",
            "beta_over_eta_egm",
            "gamma_over_eta_egm_squared",
            "cap_scale",
            "update_norm_pre_cap",
            "update_norm_post_cap",
            "cap_active",
            "V_before",
            "V_after",
            "actual_V_change",
            "C_before",
            "C_after",
            "actual_C_change",
            "norm_term_before",
            "norm_term_after",
            "perf_term_before",
            "perf_term_after",
            "actor_update_norm",
            "logstd_update_norm",
            "critic_update_norm",
            "actor_fraction_of_update",
            "logstd_fraction_of_update",
            "critic_fraction_of_update",
            "actor_fraction_of_V_decrease",
            "logstd_fraction_of_V_decrease",
            "critic_fraction_of_V_decrease",
            "approx_kl",
            "clip_fraction",
            "gamma_active_frac",
            "G_contribution_norm",
            "g_sign_mode",
            "selected_g_sign",
            "selector_mode",
            "direction_mode",
            "cost_mode",
            "plus_q_pred",
            "minus_q_pred",
            "plus_actual_change",
            "minus_actual_change",
            "no_g_reference_q_pred_post_cap",
            "no_g_reference_beta_raw",
            "dense_fallback_used",
            "fallback_reason",
            "fallback_to_noG",
            "q_or_ls_pred",
            "ls_fit_rank_corr",
            "beta_at_bound",
            "gamma_at_bound",
            "zero_update_flag",
            "norm_scale",
            "perf_scale",
            "policy_component_before",
            "policy_component_after",
            "policy_unclipped_component_before",
            "policy_unclipped_component_after",
            "critic_component_before",
            "critic_component_after",
            "logstd_component_before",
            "logstd_component_after",
            "mixed_merit_before",
            "mixed_merit_after",
            "mixed_merit_change",
            "short_clean_return_cost_before",
            "short_clean_return_cost_after",
            "short_clean_return_cost_change",
            "short_rarl_return_cost_before",
            "short_rarl_return_cost_after",
            "short_rarl_return_cost_change",
            "unclipped_actor_surrogate_change",
            "value_loss_change",
            "entropy_change",
        ]
        self._ensure_diagnostics_header()

    def _parse_grid(self, raw_value: str | Sequence[float], *, fallback: Sequence[float]) -> List[float]:
        if isinstance(raw_value, str):
            tokens = [token.strip() for token in raw_value.split(",") if token.strip()]
            if not tokens:
                values = [float(x) for x in fallback]
            else:
                values = [float(token) for token in tokens]
        else:
            values = [float(x) for x in raw_value] if raw_value else [float(x) for x in fallback]
        values = sorted({max(float(x), 0.0) for x in values if math.isfinite(float(x))})
        return values if values else [float(x) for x in fallback]

    def _ensure_diagnostics_header(self) -> None:
        if not self.diagnostics_csv_path:
            return
        diagnostics_path = _windows_safe_path(self.diagnostics_csv_path)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        if diagnostics_path.exists():
            return
        with diagnostics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writeheader()

    def _write_diagnostics_row(self, metrics: Dict[str, object]) -> None:
        if not self.diagnostics_csv_path:
            return
        diagnostics_path = _windows_safe_path(self.diagnostics_csv_path)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        with diagnostics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writerow({field: metrics.get(field) for field in self._fieldnames})

    def _scope_block_weights(self) -> Dict[str, float]:
        if self.perflyap_scope in {"actor_mean_only", "actor_game", "actor_mean_plus_adv_actor_mean"}:
            return {"actor": 1.0, "logstd": 0.0, "critic": 0.0}
        if self.perflyap_scope == "critic_downweighted":
            return {"actor": 1.0, "logstd": 0.1, "critic": 0.01}
        if self.perflyap_scope == "logstd_excluded":
            return {"actor": 1.0, "logstd": 0.0, "critic": 0.05}
        return {"actor": 1.0, "logstd": 0.1, "critic": 0.05}

    def _selected_names(self, named_params: NamedParams) -> List[str]:
        if self.perflyap_scope in {"actor_mean_only", "actor_game", "actor_mean_plus_adv_actor_mean"}:
            return [name for name, _ in named_params if classify_perf_block(name) == "actor"]
        if self.perflyap_scope == "logstd_excluded":
            return [name for name, _ in named_params if classify_perf_block(name) != "logstd"]
        return [name for name, _ in named_params]

    def _component_breakdown(self, eval_info: Dict[str, object], grads_selected: NamedTensorMap, selected_names: Sequence[str]) -> Dict[str, float]:
        block_weights = self._scope_block_weights()
        actor_norm_term = 0.5 * block_weights["actor"] * _block_mean_square(grads_selected, selected_names, "actor")
        logstd_norm_term = 0.5 * block_weights["logstd"] * _block_mean_square(grads_selected, selected_names, "logstd")
        critic_norm_term = 0.5 * block_weights["critic"] * _block_mean_square(grads_selected, selected_names, "critic")
        norm_term = actor_norm_term + logstd_norm_term + critic_norm_term
        if "unclipped" in self.cost_mode:
            policy_component = float(eval_info["policy_loss_unclipped"])
        else:
            policy_component = float(eval_info["policy_loss"])
        policy_unclipped_component = float(eval_info.get("policy_loss_unclipped", eval_info["policy_loss"]))
        critic_component = self.lambda_critic * float(eval_info["value_loss"])
        logstd_component = self.logstd_weight * float(eval_info["entropy_loss"])
        perf_term = policy_component + critic_component + logstd_component
        return {
            "actor_norm_term": float(actor_norm_term),
            "logstd_norm_term": float(logstd_norm_term),
            "critic_norm_term": float(critic_norm_term),
            "norm_term": float(norm_term),
            "policy_component": float(policy_component),
            "policy_unclipped_component": float(policy_unclipped_component),
            "critic_component": float(critic_component),
            "logstd_component": float(logstd_component),
            "perf_term": float(perf_term),
            "C": float(perf_term),
        }

    def _update_scales(self, norm_term: float, perf_term: float) -> Tuple[float, float]:
        abs_norm = max(abs(float(norm_term)), self.qp_eps)
        abs_perf = max(abs(float(perf_term)), self.qp_eps)
        if not self.use_scale_normalization:
            return 1.0, 1.0
        if self._norm_scale_ema is None:
            self._norm_scale_ema = abs_norm
        else:
            self._norm_scale_ema = self.scale_ema_decay * self._norm_scale_ema + (1.0 - self.scale_ema_decay) * abs_norm
        if self._perf_scale_ema is None:
            self._perf_scale_ema = abs_perf
        else:
            self._perf_scale_ema = self.scale_ema_decay * self._perf_scale_ema + (1.0 - self.scale_ema_decay) * abs_perf
        return max(self._norm_scale_ema, self.qp_eps), max(self._perf_scale_ema, self.qp_eps)

    def _merit_value(self, components: Dict[str, float], norm_scale: float, perf_scale: float) -> float:
        return self.lambda_N * components["norm_term"] / max(norm_scale, self.qp_eps) + self.lambda_P * components["perf_term"] / max(perf_scale, self.qp_eps)

    def _evaluate_state(
        self,
        *,
        eval_closure: Callable[..., Dict[str, object]],
        theta_state: NamedTensorMap,
        selected_names: Sequence[str],
        backward: bool,
        norm_scale: Optional[float] = None,
        perf_scale: Optional[float] = None,
        objective_mode: str = "total_loss",
    ) -> Dict[str, object]:
        eval_info = eval_closure(
            theta_override=theta_state,
            backward=backward,
            grad_scope_names=list(selected_names),
            objective_mode=objective_mode,
        )
        grads_selected = {name: eval_info["grads"][name].detach().clone() for name in selected_names}
        info = dict(eval_info)
        info["grads_selected"] = grads_selected
        components = self._component_breakdown(info, grads_selected, selected_names)
        info.update(components)
        if norm_scale is not None and perf_scale is not None:
            info["V_merit"] = self._merit_value(components, norm_scale, perf_scale)
        return info

    def _compute_g_raw(
        self,
        *,
        theta_old: NamedTensorMap,
        eval_closure: Callable[..., Dict[str, object]],
        selected_names: Sequence[str],
        f_raw_map: NamedTensorMap,
        norm_scale: float,
        perf_scale: float,
    ) -> Tuple[NamedTensorMap, bool]:
        if self.disable_g:
            return ({name: torch.zeros_like(f_raw_map[name]) for name in selected_names}, False)
        theta_plus = {name: tensor.clone() for name, tensor in theta_old.items()}
        for name in selected_names:
            theta_plus[name] = theta_old[name] + self.qp_fd_eps * f_raw_map[name]
        plus_eval = self._evaluate_state(
            eval_closure=eval_closure,
            theta_state=theta_plus,
            selected_names=selected_names,
            backward=True,
            norm_scale=norm_scale,
            perf_scale=perf_scale,
        )
        g_raw_map = {
            name: (plus_eval["grads_selected"][name] - f_raw_map[name]) / self.qp_fd_eps
            for name in selected_names
        }
        valid = _is_finite_named_map(g_raw_map, selected_names) and tensor_norm(flatten_named_tensors(g_raw_map, selected_names)) > self.qp_eps
        if not valid:
            return ({name: torch.zeros_like(f_raw_map[name]) for name in selected_names}, False)
        return g_raw_map, True

    def _resolve_direction_map(self, g_raw_map: NamedTensorMap, selected_names: Sequence[str]) -> NamedTensorMap:
        if self.direction_mode == "egm_minus_JF_F":
            return {name: -g_raw_map[name] for name in selected_names}
        return {name: g_raw_map[name] for name in selected_names}

    def _estimate_quadratic_coefficients(
        self,
        *,
        theta_old: NamedTensorMap,
        eval_closure: Callable[..., Dict[str, object]],
        selected_names: Sequence[str],
        f_raw_map: NamedTensorMap,
        g_raw_map: NamedTensorMap,
        v0: float,
        norm_scale: float,
        perf_scale: float,
    ) -> Dict[str, float | str]:
        p_map = {name: -f_raw_map[name] for name in selected_names}
        r_map = {name: g_raw_map[name] for name in selected_names}
        db = self.qp_beta_probe
        dg = self.qp_gamma_probe

        def merit_at(beta_scale: float, gamma_scale: float) -> float:
            theta_tmp = {name: tensor.clone() for name, tensor in theta_old.items()}
            for name in selected_names:
                theta_tmp[name] = theta_old[name] + beta_scale * p_map[name] + gamma_scale * r_map[name]
            eval_info = self._evaluate_state(
                eval_closure=eval_closure,
                theta_state=theta_tmp,
                selected_names=selected_names,
                backward=True,
                norm_scale=norm_scale,
                perf_scale=perf_scale,
            )
            return float(eval_info["V_merit"])

        vp_plus = merit_at(db, 0.0)
        vp_minus = merit_at(-db, 0.0)
        vr_plus = merit_at(0.0, dg)
        vr_minus = merit_at(0.0, -dg)
        vpp = merit_at(db, dg)
        vpm = merit_at(db, -dg)
        vmp = merit_at(-db, dg)
        vmm = merit_at(-db, -dg)
        l_beta = (vp_plus - vp_minus) / (2.0 * db)
        H_bb = (vp_plus - 2.0 * v0 + vp_minus) / (db * db)
        l_gamma = (vr_plus - vr_minus) / (2.0 * dg)
        H_gg = (vr_plus - 2.0 * v0 + vr_minus) / (dg * dg)
        H_bg = (vpp - vpm - vmp + vmm) / (4.0 * db * dg)
        hessian = torch.tensor([[H_bb, H_bg], [H_bg, H_gg]], dtype=torch.float64)
        eigvals = torch.linalg.eigvalsh(hessian)
        eig_min = float(eigvals.min().item())
        ridge_added = 0.0
        if (not math.isfinite(eig_min)) or eig_min < self.qp_rho:
            ridge_added = max(self.qp_rho - eig_min, self.qp_ridge, 0.0)
            hessian = hessian + ridge_added * torch.eye(2, dtype=torch.float64)
            eigvals = torch.linalg.eigvalsh(hessian)
            eig_min = float(eigvals.min().item())
        return {
            "raw_l_beta": float(l_beta),
            "raw_l_gamma": float(l_gamma),
            "raw_H_bb": float(H_bb),
            "raw_H_bg": float(H_bg),
            "raw_H_gg": float(H_gg),
            "l_beta": float(hessian.new_tensor(l_beta).item()),
            "l_gamma": float(hessian.new_tensor(l_gamma).item()),
            "H_bb": float(hessian[0, 0].item()),
            "H_bg": float(hessian[0, 1].item()),
            "H_gg": float(hessian[1, 1].item()),
            "eig_min": float(eig_min),
            "ridge_added": float(ridge_added),
        }

    def _coeffs_raw_no_g(self, coeffs: Dict[str, float | str]) -> Dict[str, float | str]:
        out = dict(coeffs)
        out["l_beta"] = float(coeffs["raw_l_beta"])
        out["l_gamma"] = 0.0
        out["H_bb"] = float(coeffs["raw_H_bb"])
        out["H_bg"] = 0.0
        out["H_gg"] = max(float(coeffs["raw_H_gg"]), self.qp_eps)
        return out

    @staticmethod
    def _q_value(beta: float, gamma: float, coeffs: Dict[str, float | str]) -> float:
        return (
            float(coeffs["l_beta"]) * beta
            + float(coeffs["l_gamma"]) * gamma
            + 0.5 * float(coeffs["H_bb"]) * beta * beta
            + float(coeffs["H_bg"]) * beta * gamma
            + 0.5 * float(coeffs["H_gg"]) * gamma * gamma
        )

    def _candidate_post_cap_metrics(
        self,
        *,
        beta_raw: float,
        gamma_raw: float,
        g_sign: str,
        coeffs: Dict[str, float | str],
        eta: float,
        f_raw_map: NamedTensorMap,
        g_raw_map: NamedTensorMap,
        selected_names: Sequence[str],
    ) -> Dict[str, float | int | str | NamedTensorMap]:
        signed_gamma_raw = gamma_raw if g_sign == "plus" else -gamma_raw
        beta_scaled = eta * beta_raw
        gamma_scaled_signed = eta * signed_gamma_raw
        update_map = {
            name: (-beta_scaled * f_raw_map[name]) + (gamma_scaled_signed * g_raw_map[name]) for name in selected_names
        }
        update_vec_pre = flatten_named_tensors(update_map, selected_names)
        update_norm_pre = tensor_norm(update_vec_pre)
        cap_scale = 1.0
        cap_active = 0
        if math.isfinite(self.qp_max_update_norm) and self.qp_max_update_norm > 0.0 and update_norm_pre > self.qp_max_update_norm:
            cap_scale = self.qp_max_update_norm / max(update_norm_pre, self.qp_eps)
            cap_active = 1
        beta_eff = beta_scaled * cap_scale
        gamma_eff_signed = gamma_scaled_signed * cap_scale
        gamma_eff = abs(gamma_eff_signed)
        update_map_capped = {name: tensor * cap_scale for name, tensor in update_map.items()}
        update_norm_post = tensor_norm(flatten_named_tensors(update_map_capped, selected_names))
        q_pred_pre_cap = self._q_value(beta_scaled, gamma_scaled_signed, coeffs)
        q_pred_post_cap = self._q_value(beta_eff, gamma_eff_signed, coeffs)
        return {
            "beta_raw": float(beta_raw),
            "gamma_raw": float(gamma_raw),
            "beta_eff": float(beta_eff),
            "gamma_eff": float(gamma_eff),
            "gamma_eff_signed": float(gamma_eff_signed),
            "selected_g_sign": g_sign,
            "cap_scale": float(cap_scale),
            "cap_active": int(cap_active),
            "update_norm_pre_cap": float(update_norm_pre),
            "update_norm_post_cap": float(update_norm_post),
            "q_pred_pre_cap": float(q_pred_pre_cap),
            "q_pred_post_cap": float(q_pred_post_cap),
            "update_map": update_map_capped,
        }

    def _solve_no_g(
        self,
        coeffs: Dict[str, float | str],
        *,
        eta: float,
        f_raw_map: NamedTensorMap,
        g_raw_map: NamedTensorMap,
        selected_names: Sequence[str],
        compare_with_raw_coeffs: bool,
    ) -> Dict[str, float | str]:
        coeffs_eval = self._coeffs_raw_no_g(coeffs) if compare_with_raw_coeffs else coeffs
        l_beta = float(coeffs_eval["l_beta"])
        H_bb = float(coeffs_eval["H_bb"])
        beta_star = 0.0
        candidates = [("zero", 0.0)]
        if H_bb > self.qp_eps and math.isfinite(H_bb):
            beta_star = -l_beta / H_bb
            candidates.append(("interior", min(max(beta_star, 0.0), self.qp_beta_max)))
        candidates.append(("beta_max", self.qp_beta_max))
        candidates.append(("egm_like", min(max(self.eta_egm_reference, 0.0), self.qp_beta_max)))
        best_case, best_beta = "zero", 0.0
        best_q = float("inf")
        for case, beta in candidates:
            metrics = self._candidate_post_cap_metrics(
                beta_raw=beta,
                gamma_raw=0.0,
                g_sign="plus",
                coeffs=coeffs_eval,
                eta=eta,
                f_raw_map=f_raw_map,
                g_raw_map=g_raw_map,
                selected_names=selected_names,
            )
            q_val = float(metrics["q_pred_post_cap"])
            if q_val < best_q and math.isfinite(q_val):
                best_q = q_val
                best_beta = beta
                best_case = case
                best_metrics = metrics
        if not math.isfinite(best_q):
            best_metrics = self._candidate_post_cap_metrics(
                beta_raw=0.0,
                gamma_raw=0.0,
                g_sign="plus",
                coeffs=coeffs_eval,
                eta=eta,
                f_raw_map=f_raw_map,
                g_raw_map=g_raw_map,
                selected_names=selected_names,
            )
        return {
            "beta": float(best_beta),
            "gamma": 0.0,
            "beta_at_bound": int(abs(best_beta - self.qp_beta_max) <= self.qp_eps and self.qp_beta_max > 0.0),
            "gamma_at_bound": 0,
            "selected_case": best_case,
            "q_pred": float(best_q),
            "q_pred_pre_cap": float(best_metrics["q_pred_pre_cap"]),
            "q_pred_post_cap": float(best_metrics["q_pred_post_cap"]),
            "beta_raw": float(best_metrics["beta_raw"]),
            "gamma_raw": 0.0,
            "beta_eff": float(best_metrics["beta_eff"]),
            "gamma_eff": 0.0,
            "cap_scale": float(best_metrics["cap_scale"]),
            "cap_active": int(best_metrics["cap_active"]),
            "update_norm_pre_cap": float(best_metrics["update_norm_pre_cap"]),
            "update_norm_post_cap": float(best_metrics["update_norm_post_cap"]),
            "selected_g_sign": "none",
            "update_map": best_metrics["update_map"],
            "fallback_reason": "",
            "dense_fallback_used": 0,
        }

    def _solve_two_direction_qp(
        self,
        coeffs: Dict[str, float | str],
        *,
        eta: float,
        f_raw_map: NamedTensorMap,
        g_raw_map: NamedTensorMap,
        selected_names: Sequence[str],
        g_sign: str,
        no_g_solution: Dict[str, float | str],
    ) -> Dict[str, float | str]:
        H_bb = float(coeffs["H_bb"])
        H_bg = float(coeffs["H_bg"])
        H_gg = float(coeffs["H_gg"])
        l_beta = float(coeffs["l_beta"])
        l_gamma = float(coeffs["l_gamma"])
        sign_scale = 1.0 if g_sign == "plus" else -1.0
        l_gamma_signed = sign_scale * l_gamma
        H_bg_signed = sign_scale * H_bg
        det = H_bb * H_gg - H_bg * H_bg
        candidates: List[Tuple[str, float, float]] = [
            ("zero", 0.0, 0.0),
            ("noG_selected", float(no_g_solution["beta_raw"]), 0.0),
            ("QP_forced_noG", float(no_g_solution["beta_raw"]), 0.0),
            ("EGM_like", min(max(self.eta_egm_reference, 0.0), self.qp_beta_max), min(max(self.eta_egm_reference * self.eta_egm_reference, 0.0), self.qp_gamma_max)),
            ("corner_00", 0.0, 0.0),
            ("corner_b0", self.qp_beta_max, 0.0),
            ("corner_0g", 0.0, self.qp_gamma_max),
            ("corner_bg", self.qp_beta_max, self.qp_gamma_max),
        ]
        if det > self.qp_eps:
            beta_star = (-l_beta * H_gg + H_bg_signed * l_gamma_signed) / det
            gamma_star = (-H_bb * l_gamma_signed + H_bg_signed * l_beta) / det
            if 0.0 <= beta_star <= self.qp_beta_max and 0.0 <= gamma_star <= self.qp_gamma_max and math.isfinite(beta_star) and math.isfinite(gamma_star):
                candidates.append(("interior", beta_star, gamma_star))
        if H_bb > self.qp_eps:
            candidates.append(("gamma0", min(max(-l_beta / H_bb, 0.0), self.qp_beta_max), 0.0))
            candidates.append(("gamma_max", min(max(-(l_beta + H_bg_signed * self.qp_gamma_max) / H_bb, 0.0), self.qp_beta_max), self.qp_gamma_max))
        if H_gg > self.qp_eps:
            candidates.append(("beta0", 0.0, min(max(-l_gamma_signed / H_gg, 0.0), self.qp_gamma_max)))
            candidates.append(("beta_max", self.qp_beta_max, min(max(-(l_gamma_signed + H_bg_signed * self.qp_beta_max) / H_gg, 0.0), self.qp_gamma_max)))
        best_case, best_beta, best_gamma = "corner_00", 0.0, 0.0
        best_q = float("inf")
        best_metrics: Optional[Dict[str, float | int | str | NamedTensorMap]] = None
        for case, beta, gamma in candidates:
            metrics = self._candidate_post_cap_metrics(
                beta_raw=beta,
                gamma_raw=gamma,
                g_sign=g_sign,
                coeffs=coeffs,
                eta=eta,
                f_raw_map=f_raw_map,
                g_raw_map=g_raw_map,
                selected_names=selected_names,
            )
            q_val = float(metrics["q_pred_post_cap"])
            if q_val < best_q and math.isfinite(q_val):
                best_q = q_val
                best_beta = beta
                best_gamma = gamma
                best_case = case
                best_metrics = metrics
        dense_fallback_used = 0
        fallback_reason = ""
        no_g_reference_metrics = self._candidate_post_cap_metrics(
            beta_raw=float(no_g_solution["beta_raw"]),
            gamma_raw=0.0,
            g_sign="plus",
            coeffs=coeffs,
            eta=eta,
            f_raw_map=f_raw_map,
            g_raw_map=g_raw_map,
            selected_names=selected_names,
        )
        no_g_q = float(no_g_reference_metrics["q_pred_post_cap"])
        if (best_metrics is None) or (best_q > no_g_q + 1e-10):
            if self.qp_dense_fallback_points >= 2:
                dense_fallback_used = 1
                beta_grid = torch.linspace(0.0, self.qp_beta_max, self.qp_dense_fallback_points, dtype=torch.float64).tolist()
                gamma_grid = torch.linspace(0.0, self.qp_gamma_max, self.qp_dense_fallback_points, dtype=torch.float64).tolist()
                for beta in beta_grid:
                    for gamma in gamma_grid:
                        metrics = self._candidate_post_cap_metrics(
                            beta_raw=float(beta),
                            gamma_raw=float(gamma),
                            g_sign=g_sign,
                            coeffs=coeffs,
                            eta=eta,
                            f_raw_map=f_raw_map,
                            g_raw_map=g_raw_map,
                            selected_names=selected_names,
                        )
                        q_val = float(metrics["q_pred_post_cap"])
                        if q_val < best_q and math.isfinite(q_val):
                            best_q = q_val
                            best_beta = float(beta)
                            best_gamma = float(gamma)
                            best_case = "dense_grid"
                            best_metrics = metrics
            if (best_metrics is None) or (best_q > no_g_q + 1e-10):
                best_case = "fallback_noG"
                best_beta = float(no_g_solution["beta_raw"])
                best_gamma = 0.0
                best_q = no_g_q
                best_metrics = no_g_reference_metrics
                fallback_reason = "qp_worse_than_nog"
        return {
            "beta": float(best_beta),
            "gamma": float(best_gamma),
            "beta_at_bound": int(abs(best_beta - self.qp_beta_max) <= self.qp_eps and self.qp_beta_max > 0.0),
            "gamma_at_bound": int(abs(best_gamma - self.qp_gamma_max) <= self.qp_eps and self.qp_gamma_max > 0.0),
            "selected_case": best_case,
            "q_pred": float(best_q),
            "q_pred_pre_cap": float(best_metrics["q_pred_pre_cap"]),
            "q_pred_post_cap": float(best_metrics["q_pred_post_cap"]),
            "beta_raw": float(best_metrics["beta_raw"]),
            "gamma_raw": float(best_metrics["gamma_raw"]),
            "beta_eff": float(best_metrics["beta_eff"]),
            "gamma_eff": float(best_metrics["gamma_eff"]),
            "cap_scale": float(best_metrics["cap_scale"]),
            "cap_active": int(best_metrics["cap_active"]),
            "update_norm_pre_cap": float(best_metrics["update_norm_pre_cap"]),
            "update_norm_post_cap": float(best_metrics["update_norm_post_cap"]),
            "selected_g_sign": str(best_metrics["selected_g_sign"]),
            "update_map": best_metrics["update_map"],
            "no_g_reference_q_pred_post_cap": float(no_g_reference_metrics["q_pred_post_cap"]),
            "fallback_reason": fallback_reason,
            "dense_fallback_used": dense_fallback_used,
        }

    def _evaluate_actual_candidate_merit(
        self,
        *,
        theta_old: NamedTensorMap,
        named_params: NamedParams,
        eval_closure: Callable[..., Dict[str, object]],
        selected_names: Sequence[str],
        update_map: NamedTensorMap,
        base_eval: Dict[str, object],
        norm_scale: float,
        perf_scale: float,
    ) -> float:
        theta_new = apply_state_delta(theta_old, update_map)
        eval_info = self._evaluate_state(
            eval_closure=eval_closure,
            theta_state=theta_new,
            selected_names=selected_names,
            backward=True,
            norm_scale=norm_scale,
            perf_scale=perf_scale,
        )
        restore_named_state(named_params, theta_old)
        return float(eval_info["V_merit"] - float(base_eval["V_merit"]))

    def _evaluate_actual_candidate_cost(
        self,
        *,
        theta_old: NamedTensorMap,
        named_params: NamedParams,
        eval_closure: Callable[..., Dict[str, object]],
        selected_names: Sequence[str],
        update_map: NamedTensorMap,
        base_cost: float,
    ) -> float:
        theta_new = apply_state_delta(theta_old, update_map)
        eval_info = eval_closure(theta_override=theta_new, backward=False, grad_scope_names=list(selected_names))
        restore_named_state(named_params, theta_old)
        cost_after = float(eval_info["policy_loss_unclipped"]) if "unclipped" in self.cost_mode else float(eval_info["policy_loss"])
        return cost_after - base_cost

    def _evaluate_candidate_selector_merit(
        self,
        *,
        theta_old: NamedTensorMap,
        named_params: NamedParams,
        eval_closure: Callable[..., Dict[str, object]],
        selected_names: Sequence[str],
        update_map: NamedTensorMap,
        base_eval: Dict[str, object],
        candidate_merit_evaluator: Optional[Callable[[NamedTensorMap, Sequence[str]], Dict[str, float]]] = None,
    ) -> Dict[str, float]:
        theta_new = apply_state_delta(theta_old, update_map)
        if candidate_merit_evaluator is not None:
            out = candidate_merit_evaluator(theta_new, selected_names)
            restore_named_state(named_params, theta_old)
            return {key: float(value) for key, value in out.items()}
        eval_info = eval_closure(theta_override=theta_new, backward=False, grad_scope_names=list(selected_names))
        restore_named_state(named_params, theta_old)
        if "unclipped" in self.cost_mode:
            merit_after = float(eval_info["policy_loss_unclipped"])
            actor_cost_after = float(eval_info["policy_loss_unclipped"])
        else:
            merit_after = float(eval_info["policy_loss"])
            actor_cost_after = float(eval_info["policy_loss"])
        return {
            "mixed_merit": merit_after,
            "actor_cost": actor_cost_after,
            "short_clean_return_cost": float("nan"),
            "short_rarl_return_cost": float("nan"),
            "value_loss": float(eval_info["value_loss"]),
            "entropy_loss": float(eval_info["entropy_loss"]),
        }

    def _enumerate_selector_candidates(
        self,
        *,
        eta: float,
        f_raw_map: NamedTensorMap,
        direction_map: NamedTensorMap,
        selected_names: Sequence[str],
    ) -> List[Tuple[float, float]]:
        if self.selector_mode == "actual_mixed_selector":
            fixed_beta = min(max(self.fixed_beta_raw if self.fixed_beta_raw is not None else self.qp_beta_max, 0.0), self.qp_beta_max)
            fixed_gamma = 0.0 if self.disable_g else min(max(self.fixed_gamma_raw or 0.0, 0.0), self.qp_gamma_max)
            return [(0.0, 0.0), (float(fixed_beta), 0.0), (float(fixed_beta), float(fixed_gamma))]
        beta_candidates = [min(max(beta, 0.0), self.qp_beta_max) for beta in self.selector_beta_grid]
        gamma_candidates = [min(max(gamma, 0.0), self.qp_gamma_max) for gamma in self.selector_gamma_grid]
        pairs: List[Tuple[float, float]] = []
        for beta in beta_candidates:
            for gamma in gamma_candidates:
                pairs.append((float(beta), float(gamma)))
        if self.fixed_beta_raw is not None:
            pairs.append((min(max(self.fixed_beta_raw, 0.0), self.qp_beta_max), 0.0 if self.disable_g else min(max(self.fixed_gamma_raw or 0.0, 0.0), self.qp_gamma_max)))
        pairs.append((0.0, 0.0))
        uniq: List[Tuple[float, float]] = []
        seen = set()
        for beta, gamma in pairs:
            key = (round(beta, 12), round(gamma, 12))
            if key in seen:
                continue
            seen.add(key)
            uniq.append((beta, gamma))
        return uniq

    @staticmethod
    def _spearman_rank_corr(actual_values: Sequence[float], pred_values: Sequence[float]) -> float:
        if len(actual_values) < 2:
            return float("nan")
        actual_series = np.asarray(actual_values, dtype=np.float64)
        pred_series = np.asarray(pred_values, dtype=np.float64)
        actual_rank = pd.Series(actual_series).rank(method="average")
        pred_rank = pd.Series(pred_series).rank(method="average")
        corr = actual_rank.corr(pred_rank, method="pearson")
        return float(corr) if corr is not None else float("nan")

    def _fit_ls_predictors(
        self,
        *,
        candidate_metrics: List[Dict[str, object]],
    ) -> Tuple[Optional[np.ndarray], float]:
        if len(candidate_metrics) < 5:
            return None, float("nan")
        beta_eff = np.asarray([float(row["beta_eff"]) for row in candidate_metrics], dtype=np.float64)
        gamma_eff_signed = np.asarray([float(row["gamma_eff_signed"]) for row in candidate_metrics], dtype=np.float64)
        target = np.asarray([float(row["actual_cost_change"]) for row in candidate_metrics], dtype=np.float64)
        features = np.column_stack(
            [
                beta_eff,
                gamma_eff_signed,
                0.5 * beta_eff * beta_eff,
                beta_eff * gamma_eff_signed,
                0.5 * gamma_eff_signed * gamma_eff_signed,
            ]
        )
        ridge = 1e-6 if self.ls_fit_variant == "ridge_LS" else 0.0
        weights = np.ones(len(target), dtype=np.float64)
        if self.ls_fit_variant == "weighted_LS_more_weight_near_zero":
            radius = np.sqrt(beta_eff * beta_eff + gamma_eff_signed * gamma_eff_signed)
            denom = max(float(np.quantile(radius, 0.8)), self.qp_eps)
            weights = 1.0 / (1.0 + (radius / denom) ** 2)
        lhs = features.T @ (features * weights[:, None]) + ridge * np.eye(features.shape[1])
        rhs = features.T @ (target * weights)
        try:
            coef = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            return None, float("nan")
        pred = features @ coef
        return coef, self._spearman_rank_corr(target, pred)

    @staticmethod
    def _ls_q_value(beta_eff: float, gamma_eff_signed: float, coef: np.ndarray) -> float:
        feats = np.asarray(
            [
                beta_eff,
                gamma_eff_signed,
                0.5 * beta_eff * beta_eff,
                beta_eff * gamma_eff_signed,
                0.5 * gamma_eff_signed * gamma_eff_signed,
            ],
            dtype=np.float64,
        )
        return float(feats @ coef)

    def _select_online_candidate(
        self,
        *,
        theta_old: NamedTensorMap,
        named_params: NamedParams,
        eval_closure: Callable[..., Dict[str, object]],
        selected_names: Sequence[str],
        base_eval: Dict[str, object],
        coeffs: Dict[str, float | str],
        eta: float,
        f_raw_map: NamedTensorMap,
        direction_map: NamedTensorMap,
        no_g_solution: Dict[str, float | str],
        candidate_merit_evaluator: Optional[Callable[[NamedTensorMap, Sequence[str]], Dict[str, float]]] = None,
    ) -> Dict[str, object]:
        def finalize_choice(
            chosen_metrics: Dict[str, object],
            *,
            q_or_ls_pred: float,
            ls_fit_rank_corr: float,
            fallback_reason: str,
            fallback_to_noG: int,
        ) -> Dict[str, object]:
            gamma_raw = float(chosen_metrics.get("gamma_raw", 0.0))
            gamma_eff = float(chosen_metrics.get("gamma_eff", 0.0))
            selected_g_sign = "minus" if gamma_eff > self.qp_eps and self.direction_mode == "egm_minus_JF_F" else ("plus" if gamma_eff > self.qp_eps else "none")
            return {
                **chosen_metrics,
                "beta": float(chosen_metrics.get("beta_raw", 0.0)),
                "gamma": gamma_raw,
                "beta_at_bound": int(abs(float(chosen_metrics.get("beta_raw", 0.0)) - self.qp_beta_max) <= self.qp_eps and self.qp_beta_max > 0.0),
                "gamma_at_bound": int(abs(gamma_raw - self.qp_gamma_max) <= self.qp_eps and self.qp_gamma_max > 0.0),
                "selected_case": str(chosen_metrics.get("selected_case", "selector")),
                "q_pred": float(chosen_metrics.get("q_pred_post_cap", q_or_ls_pred)),
                "selected_g_sign": selected_g_sign,
                "q_or_ls_pred": float(q_or_ls_pred),
                "ls_fit_rank_corr": float(ls_fit_rank_corr),
                "fallback_reason": fallback_reason,
                "fallback_to_noG": int(fallback_to_noG),
                "dense_fallback_used": 0,
            }

        base_cost = float(base_eval["policy_component"])
        base_mixed = float(base_eval["V_merit"])
        no_g_metrics = self._candidate_post_cap_metrics(
            beta_raw=float(no_g_solution["beta_raw"]),
            gamma_raw=0.0,
            g_sign="plus",
            coeffs=coeffs,
            eta=eta,
            f_raw_map=f_raw_map,
            g_raw_map=direction_map,
            selected_names=selected_names,
        )
        no_g_actual_cost = self._evaluate_actual_candidate_cost(
            theta_old=theta_old,
            named_params=named_params,
            eval_closure=eval_closure,
            selected_names=selected_names,
            update_map=no_g_metrics["update_map"],
            base_cost=base_cost,
        )
        no_g_selector_eval = self._evaluate_candidate_selector_merit(
            theta_old=theta_old,
            named_params=named_params,
            eval_closure=eval_closure,
            selected_names=selected_names,
            update_map=no_g_metrics["update_map"],
            base_eval=base_eval,
            candidate_merit_evaluator=candidate_merit_evaluator,
        )
        candidate_metrics: List[Dict[str, object]] = []
        for beta_raw, gamma_raw in self._enumerate_selector_candidates(
            eta=eta,
            f_raw_map=f_raw_map,
            direction_map=direction_map,
            selected_names=selected_names,
        ):
            metrics = self._candidate_post_cap_metrics(
                beta_raw=beta_raw,
                gamma_raw=0.0 if self.disable_g else gamma_raw,
                g_sign="plus",
                coeffs=coeffs,
                eta=eta,
                f_raw_map=f_raw_map,
                g_raw_map=direction_map,
                selected_names=selected_names,
            )
            metrics["actual_cost_change"] = self._evaluate_actual_candidate_cost(
                theta_old=theta_old,
                named_params=named_params,
                eval_closure=eval_closure,
                selected_names=selected_names,
                update_map=metrics["update_map"],
                base_cost=base_cost,
            )
            selector_eval = self._evaluate_candidate_selector_merit(
                theta_old=theta_old,
                named_params=named_params,
                eval_closure=eval_closure,
                selected_names=selected_names,
                update_map=metrics["update_map"],
                base_eval=base_eval,
                candidate_merit_evaluator=candidate_merit_evaluator,
            )
            metrics["actual_mixed_merit_after"] = float(selector_eval["mixed_merit"])
            metrics["actual_mixed_merit_change"] = float(selector_eval["mixed_merit"] - base_mixed)
            candidate_metrics.append(metrics)

        if self.selector_mode == "actual_mixed_selector":
            best_metrics = min(candidate_metrics, key=lambda row: float(row["actual_mixed_merit_change"]))
            no_g_actual_mixed = float(no_g_selector_eval["mixed_merit"] - base_mixed)
            fallback_to_nog = float(best_metrics["actual_mixed_merit_change"]) > no_g_actual_mixed + 1e-12
            chosen = no_g_metrics if fallback_to_nog else best_metrics
            return finalize_choice(
                chosen,
                q_or_ls_pred=float(best_metrics["actual_mixed_merit_change"]) if not fallback_to_nog else float(no_g_actual_mixed),
                ls_fit_rank_corr=float("nan"),
                fallback_reason="actual_mixed_selector_worse_than_nog" if fallback_to_nog else "",
                fallback_to_noG=int(fallback_to_nog),
            )

        if self.selector_mode == "actual_surrogate_selector":
            best_metrics = min(candidate_metrics, key=lambda row: float(row["actual_cost_change"]))
            fallback_to_nog = float(best_metrics["actual_cost_change"]) > no_g_actual_cost + 1e-12
            chosen = no_g_metrics if fallback_to_nog else best_metrics
            return finalize_choice(
                chosen,
                q_or_ls_pred=float(chosen["actual_cost_change"]) if chosen is best_metrics else float(no_g_actual_cost),
                ls_fit_rank_corr=float("nan"),
                fallback_reason="actual_selector_worse_than_nog" if fallback_to_nog else "",
                fallback_to_noG=int(fallback_to_nog),
            )

        if self.selector_mode == "safe_fixed_minusg":
            fixed_beta = self.fixed_beta_raw if self.fixed_beta_raw is not None else float(no_g_solution["beta_raw"])
            fixed_gamma = self.fixed_gamma_raw if self.fixed_gamma_raw is not None else 0.0
            fixed_metrics = self._candidate_post_cap_metrics(
                beta_raw=min(max(float(fixed_beta), 0.0), self.qp_beta_max),
                gamma_raw=min(max(float(fixed_gamma), 0.0), self.qp_gamma_max),
                g_sign="plus",
                coeffs=coeffs,
                eta=eta,
                f_raw_map=f_raw_map,
                g_raw_map=direction_map,
                selected_names=selected_names,
            )
            fixed_actual_cost = self._evaluate_actual_candidate_cost(
                theta_old=theta_old,
                named_params=named_params,
                eval_closure=eval_closure,
                selected_names=selected_names,
                update_map=fixed_metrics["update_map"],
                base_cost=base_cost,
            )
            if candidate_merit_evaluator is not None:
                fixed_selector_eval = self._evaluate_candidate_selector_merit(
                    theta_old=theta_old,
                    named_params=named_params,
                    eval_closure=eval_closure,
                    selected_names=selected_names,
                    update_map=fixed_metrics["update_map"],
                    base_eval=base_eval,
                    candidate_merit_evaluator=candidate_merit_evaluator,
                )
                no_g_actual_mixed = float(no_g_selector_eval["mixed_merit"] - base_mixed)
                fixed_actual_mixed = float(fixed_selector_eval["mixed_merit"] - base_mixed)
                fallback_to_nog = fixed_actual_mixed > no_g_actual_mixed + 1e-12
                compare_value = fixed_actual_mixed if not fallback_to_nog else no_g_actual_mixed
            else:
                fallback_to_nog = fixed_actual_cost > no_g_actual_cost + 1e-12
                compare_value = fixed_actual_cost if not fallback_to_nog else no_g_actual_cost
            chosen = no_g_metrics if fallback_to_nog else fixed_metrics
            return finalize_choice(
                chosen,
                q_or_ls_pred=float(compare_value),
                ls_fit_rank_corr=float("nan"),
                fallback_reason="fixed_minusg_worse_than_nog" if fallback_to_nog else "",
                fallback_to_noG=int(fallback_to_nog),
            )

        coef, rank_corr = self._fit_ls_predictors(candidate_metrics=candidate_metrics)
        if coef is None:
            return finalize_choice(
                no_g_metrics,
                q_or_ls_pred=float(no_g_actual_cost),
                ls_fit_rank_corr=float("nan"),
                fallback_reason="ls_fit_failed",
                fallback_to_noG=1,
            )
        for row in candidate_metrics:
            row["ls_pred"] = self._ls_q_value(float(row["beta_eff"]), float(row["gamma_eff_signed"]), coef)
        best_metrics = min(candidate_metrics, key=lambda row: float(row["ls_pred"]))
        fallback_to_nog = float(best_metrics["actual_cost_change"]) > no_g_actual_cost + 1e-12
        chosen = no_g_metrics if fallback_to_nog else best_metrics
        return finalize_choice(
            chosen,
            q_or_ls_pred=float(best_metrics["ls_pred"]) if not fallback_to_nog else float(no_g_actual_cost),
            ls_fit_rank_corr=float(rank_corr),
            fallback_reason="ls_selector_worse_than_nog" if fallback_to_nog else "",
            fallback_to_noG=int(fallback_to_nog),
        )

    def step(
        self,
        closure: Optional[Callable[[], torch.Tensor]] = None,
        *,
        eval_closure: Optional[Callable[..., Dict[str, object]]] = None,
        named_params: Optional[NamedParams] = None,
        candidate_merit_evaluator: Optional[Callable[[NamedTensorMap, Sequence[str]], Dict[str, float]]] = None,
    ):
        del closure
        if eval_closure is None or named_params is None:
            raise ValueError("Performance-aligned proposed optimizers require eval_closure and named_params.")

        selected_names = self._selected_names(named_params)
        theta_old = clone_named_state(named_params)
        eta = float(self.param_groups[0]["lr"])
        base_eval_unscaled = self._evaluate_state(
            eval_closure=eval_closure,
            theta_state=theta_old,
            selected_names=selected_names,
            backward=True,
        )
        norm_scale, perf_scale = self._update_scales(float(base_eval_unscaled["norm_term"]), float(base_eval_unscaled["perf_term"]))
        base_eval = self._evaluate_state(
            eval_closure=eval_closure,
            theta_state=theta_old,
            selected_names=selected_names,
            backward=True,
            norm_scale=norm_scale,
            perf_scale=perf_scale,
        )
        f_raw_map = {name: base_eval["grads_selected"][name].detach().clone() for name in selected_names}
        f_raw_vec = flatten_named_tensors(f_raw_map, selected_names)
        if self.direction_mode == "performance_grad":
            perf_grad_eval = self._evaluate_state(
                eval_closure=eval_closure,
                theta_state=theta_old,
                selected_names=selected_names,
                backward=True,
                norm_scale=norm_scale,
                perf_scale=perf_scale,
                objective_mode="performance_loss",
            )
            direction_map = {
                name: -perf_grad_eval["grads_selected"][name].detach().clone()
                for name in selected_names
            }
            finite_difference_valid = _is_finite_named_map(direction_map, selected_names) and tensor_norm(flatten_named_tensors(direction_map, selected_names)) > self.qp_eps
            g_raw_map = direction_map
        else:
            g_raw_map, finite_difference_valid = self._compute_g_raw(
                theta_old=theta_old,
                eval_closure=eval_closure,
                selected_names=selected_names,
                f_raw_map=f_raw_map,
                norm_scale=norm_scale,
                perf_scale=perf_scale,
            )
            direction_map = self._resolve_direction_map(g_raw_map, selected_names)
        g_raw_vec = flatten_named_tensors(direction_map, selected_names)
        coeffs = self._estimate_quadratic_coefficients(
            theta_old=theta_old,
            eval_closure=eval_closure,
            selected_names=selected_names,
            f_raw_map=f_raw_map,
            g_raw_map=direction_map,
            v0=float(base_eval["V_merit"]),
            norm_scale=norm_scale,
            perf_scale=perf_scale,
        )
        no_g_solution = self._solve_no_g(
            coeffs,
            eta=eta,
            f_raw_map=f_raw_map,
            g_raw_map=direction_map,
            selected_names=selected_names,
            compare_with_raw_coeffs=True,
        )
        no_g_reference_q_pred_post_cap = float("nan")
        no_g_reference_beta_raw = float(no_g_solution["beta_raw"])
        plus_solution = None
        minus_solution = None
        plus_actual_change = float("nan")
        minus_actual_change = float("nan")
        fallback_to_noG = 0
        q_or_ls_pred = float("nan")
        ls_fit_rank_corr = float("nan")
        if self.disable_g:
            if self.selector_mode == "fixed_nog":
                fixed_beta = float(self.fixed_beta_raw if self.fixed_beta_raw is not None else no_g_solution["beta_raw"])
                solution = self._candidate_post_cap_metrics(
                    beta_raw=min(max(fixed_beta, 0.0), self.qp_beta_max),
                    gamma_raw=0.0,
                    g_sign="plus",
                    coeffs=self._coeffs_raw_no_g(coeffs),
                    eta=eta,
                    f_raw_map=f_raw_map,
                    g_raw_map=direction_map,
                    selected_names=selected_names,
                )
                solution.update(
                    {
                        "beta": float(solution["beta_raw"]),
                        "gamma": 0.0,
                        "beta_at_bound": int(abs(float(solution["beta_raw"]) - self.qp_beta_max) <= self.qp_eps and self.qp_beta_max > 0.0),
                        "gamma_at_bound": 0,
                        "selected_case": "fixed_nog",
                        "q_pred": float(solution["q_pred_post_cap"]),
                        "selected_g_sign": "none",
                        "fallback_reason": "",
                        "dense_fallback_used": 0,
                        "fallback_to_noG": 0,
                    }
                )
            else:
                solution = no_g_solution
            selected_g_sign = "none"
        elif not finite_difference_valid:
            solution = no_g_solution
            selected_g_sign = "none"
        else:
            if self.selector_mode == "fixed_nog":
                fixed_beta = float(self.fixed_beta_raw if self.fixed_beta_raw is not None else no_g_solution["beta_raw"])
                solution = self._candidate_post_cap_metrics(
                    beta_raw=min(max(fixed_beta, 0.0), self.qp_beta_max),
                    gamma_raw=0.0,
                    g_sign="plus",
                    coeffs=self._coeffs_raw_no_g(coeffs),
                    eta=eta,
                    f_raw_map=f_raw_map,
                    g_raw_map=direction_map,
                    selected_names=selected_names,
                )
                solution.update(
                    {
                        "beta": float(solution["beta_raw"]),
                        "gamma": 0.0,
                        "beta_at_bound": int(abs(float(solution["beta_raw"]) - self.qp_beta_max) <= self.qp_eps and self.qp_beta_max > 0.0),
                        "gamma_at_bound": 0,
                        "selected_case": "fixed_nog",
                        "q_pred": float(solution["q_pred_post_cap"]),
                        "selected_g_sign": "none",
                        "fallback_reason": "",
                        "dense_fallback_used": 0,
                        "fallback_to_noG": 0,
                    }
                )
                selected_g_sign = "none"
            elif self.selector_mode in {"actual_surrogate_selector", "ls_capaware", "safe_fixed_minusg"}:
                solution = self._select_online_candidate(
                    theta_old=theta_old,
                    named_params=named_params,
                    eval_closure=eval_closure,
                    selected_names=selected_names,
                    base_eval=base_eval,
                    coeffs=coeffs,
                    eta=eta,
                    f_raw_map=f_raw_map,
                    direction_map=direction_map,
                    no_g_solution=no_g_solution,
                    candidate_merit_evaluator=candidate_merit_evaluator,
                )
                selected_g_sign = (
                    "minus"
                    if self.direction_mode == "egm_minus_JF_F" and float(solution["gamma_eff"]) > self.qp_eps
                    else ("plus" if float(solution["gamma_eff"]) > self.qp_eps else "none")
                )
                fallback_to_noG = int(solution.get("fallback_to_noG", 0))
                q_or_ls_pred = float(solution.get("q_or_ls_pred", float("nan")))
                ls_fit_rank_corr = float(solution.get("ls_fit_rank_corr", float("nan")))
            else:
                plus_solution = self._solve_two_direction_qp(
                    coeffs,
                    eta=eta,
                    f_raw_map=f_raw_map,
                    g_raw_map=direction_map,
                    selected_names=selected_names,
                    g_sign="plus",
                    no_g_solution=no_g_solution,
                )
                minus_solution = plus_solution
                no_g_reference_q_pred_post_cap = float(plus_solution.get("no_g_reference_q_pred_post_cap", float("nan")))
                solution = plus_solution
                selected_g_sign = "minus" if self.direction_mode == "egm_minus_JF_F" else "plus"
        beta = float(solution["beta"])
        gamma = 0.0 if self.disable_g else float(solution["gamma"])
        update_map = {name: tensor.clone() for name, tensor in solution["update_map"].items()}
        update_norm_pre = float(solution["update_norm_pre_cap"])
        update_norm_post = float(solution["update_norm_post_cap"])
        cap_active = int(solution["cap_active"])
        theta_new = apply_state_delta(theta_old, update_map)
        new_eval = self._evaluate_state(
            eval_closure=eval_closure,
            theta_state=theta_new,
            selected_names=selected_names,
            backward=True,
            norm_scale=norm_scale,
            perf_scale=perf_scale,
        )
        restore_named_state(named_params, theta_new)
        diff = named_difference(theta_new, theta_old, selected_names)
        actor_update_norm = block_norm(diff, selected_names, "actor")
        logstd_update_norm = block_norm(diff, selected_names, "logstd")
        critic_update_norm = block_norm(diff, selected_names, "critic")
        total_v_decrease = max(float(base_eval["V_merit"]) - float(new_eval["V_merit"]), self.qp_eps)
        actor_norm_decrease = self.lambda_N * (float(base_eval["actor_norm_term"]) - float(new_eval["actor_norm_term"])) / max(norm_scale, self.qp_eps)
        logstd_norm_decrease = self.lambda_N * (float(base_eval["logstd_norm_term"]) - float(new_eval["logstd_norm_term"])) / max(norm_scale, self.qp_eps)
        critic_norm_decrease = self.lambda_N * (float(base_eval["critic_norm_term"]) - float(new_eval["critic_norm_term"])) / max(norm_scale, self.qp_eps)
        actor_perf_decrease = self.lambda_P * (float(base_eval["policy_component"]) - float(new_eval["policy_component"])) / max(perf_scale, self.qp_eps)
        logstd_perf_decrease = self.lambda_P * (float(base_eval["logstd_component"]) - float(new_eval["logstd_component"])) / max(perf_scale, self.qp_eps)
        critic_perf_decrease = self.lambda_P * (float(base_eval["critic_component"]) - float(new_eval["critic_component"])) / max(perf_scale, self.qp_eps)
        actor_v_decrease = actor_norm_decrease + actor_perf_decrease
        logstd_v_decrease = logstd_norm_decrease + logstd_perf_decrease
        critic_v_decrease = critic_norm_decrease + critic_perf_decrease
        total_update_norm = max(update_norm_post, self.qp_eps)
        actor_param_names = block_names(selected_names, "actor")
        logstd_param_names = block_names(selected_names, "logstd")
        critic_param_names = block_names(selected_names, "critic")
        if candidate_merit_evaluator is not None:
            base_selector_eval = candidate_merit_evaluator(theta_old, selected_names)
            restore_named_state(named_params, theta_new)
            new_selector_eval = candidate_merit_evaluator(theta_new, selected_names)
            restore_named_state(named_params, theta_new)
        else:
            base_selector_eval = {
                "mixed_merit": float(base_eval["perf_term"]),
                "actor_cost": float(base_eval["policy_unclipped_component"] if "unclipped" in self.cost_mode else base_eval["policy_component"]),
                "short_clean_return_cost": float("nan"),
                "short_rarl_return_cost": float("nan"),
                "value_loss": float(new_eval["critic_component"] / max(self.lambda_critic, 1.0)) if self.lambda_critic != 0.0 else float(new_eval["critic_component"]),
                "entropy_loss": float(new_eval["logstd_component"] / max(self.logstd_weight, 1.0)) if self.logstd_weight != 0.0 else float(new_eval["logstd_component"]),
            }
            new_selector_eval = {
                "mixed_merit": float(new_eval["perf_term"]),
                "actor_cost": float(new_eval["policy_unclipped_component"] if "unclipped" in self.cost_mode else new_eval["policy_component"]),
                "short_clean_return_cost": float("nan"),
                "short_rarl_return_cost": float("nan"),
                "value_loss": float(new_eval["critic_component"] / max(self.lambda_critic, 1.0)) if self.lambda_critic != 0.0 else float(new_eval["critic_component"]),
                "entropy_loss": float(new_eval["logstd_component"] / max(self.logstd_weight, 1.0)) if self.logstd_weight != 0.0 else float(new_eval["logstd_component"]),
            }
        metrics: Dict[str, object] = {
            "step_index": self._step_index,
            "active_role": self.role,
            "scope": self.perflyap_scope,
            "num_actor_params": len(actor_param_names),
            "num_logstd_params": len(logstd_param_names),
            "num_critic_params": len(critic_param_names),
            "lambda_N": self.lambda_N,
            "lambda_P": self.lambda_P,
            "lambda_critic": self.lambda_critic,
            "logstd_weight": self.logstd_weight,
            "fd_eps": self.qp_fd_eps,
            "finite_difference_valid": int(finite_difference_valid),
            "actor_F_norm": block_norm(f_raw_map, selected_names, "actor"),
            "logstd_F_norm": block_norm(f_raw_map, selected_names, "logstd"),
            "critic_F_norm": block_norm(f_raw_map, selected_names, "critic"),
            "F_raw_norm": tensor_norm(f_raw_vec),
            "G_raw_norm": tensor_norm(g_raw_vec),
            "cosine_F_G": cosine_similarity(f_raw_vec, g_raw_vec),
            "l_beta": float(coeffs["l_beta"]),
            "l_gamma": float(coeffs["l_gamma"]),
            "H_bb": float(coeffs["H_bb"]),
            "H_bg": float(coeffs["H_bg"]),
            "H_gg": float(coeffs["H_gg"]),
            "eig_min": float(coeffs["eig_min"]),
            "ridge_added": float(coeffs["ridge_added"]),
            "q_pred": float(solution["q_pred_post_cap"]),
            "q_pred_pre_cap": float(solution["q_pred_pre_cap"]),
            "q_pred_post_cap": float(solution["q_pred_post_cap"]),
            "beta": beta,
            "gamma": gamma,
            "beta_raw": float(solution["beta_raw"]),
            "gamma_raw": float(solution["gamma_raw"]),
            "beta_eff": float(solution["beta_eff"]),
            "gamma_eff": float(solution["gamma_eff"]),
            "beta_over_eta_egm": float(solution["beta_eff"]) / self.eta_egm_reference,
            "gamma_over_eta_egm_squared": float(solution["gamma_eff"]) / max(self.eta_egm_reference * self.eta_egm_reference, self.qp_eps),
            "cap_scale": float(solution["cap_scale"]),
            "update_norm_pre_cap": update_norm_pre,
            "update_norm_post_cap": update_norm_post,
            "cap_active": cap_active,
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
            "gamma_active_frac": float(gamma > self.qp_eps),
            "G_contribution_norm": float(solution["gamma_eff"]) * tensor_norm(g_raw_vec),
            "g_sign_mode": self.g_sign_mode,
            "selected_g_sign": "none" if self.disable_g else selected_g_sign,
            "selector_mode": self.selector_mode,
            "direction_mode": self.direction_mode,
            "cost_mode": self.cost_mode,
            "plus_q_pred": float("nan") if plus_solution is None else float(plus_solution["q_pred_post_cap"]),
            "minus_q_pred": float("nan") if minus_solution is None else float(minus_solution["q_pred_post_cap"]),
            "plus_actual_change": plus_actual_change,
            "minus_actual_change": minus_actual_change,
            "no_g_reference_q_pred_post_cap": no_g_reference_q_pred_post_cap,
            "no_g_reference_beta_raw": no_g_reference_beta_raw,
            "dense_fallback_used": int(solution.get("dense_fallback_used", 0)),
            "fallback_reason": str(solution.get("fallback_reason", "")),
            "fallback_to_noG": int(solution.get("fallback_to_noG", fallback_to_noG)),
            "q_or_ls_pred": q_or_ls_pred,
            "ls_fit_rank_corr": ls_fit_rank_corr,
            "beta_at_bound": int(solution["beta_at_bound"]),
            "gamma_at_bound": int(solution["gamma_at_bound"]),
            "zero_update_flag": int(update_norm_post <= self.qp_eps),
            "norm_scale": float(norm_scale),
            "perf_scale": float(perf_scale),
            "policy_component_before": float(base_eval["policy_component"]),
            "policy_component_after": float(new_eval["policy_component"]),
            "policy_unclipped_component_before": float(base_eval["policy_unclipped_component"]),
            "policy_unclipped_component_after": float(new_eval["policy_unclipped_component"]),
            "critic_component_before": float(base_eval["critic_component"]),
            "critic_component_after": float(new_eval["critic_component"]),
            "logstd_component_before": float(base_eval["logstd_component"]),
            "logstd_component_after": float(new_eval["logstd_component"]),
            "mixed_merit_before": float(base_selector_eval["mixed_merit"]),
            "mixed_merit_after": float(new_selector_eval["mixed_merit"]),
            "mixed_merit_change": float(new_selector_eval["mixed_merit"] - base_selector_eval["mixed_merit"]),
            "short_clean_return_cost_before": float(base_selector_eval.get("short_clean_return_cost", float("nan"))),
            "short_clean_return_cost_after": float(new_selector_eval.get("short_clean_return_cost", float("nan"))),
            "short_clean_return_cost_change": float(new_selector_eval.get("short_clean_return_cost", float("nan")) - base_selector_eval.get("short_clean_return_cost", float("nan"))),
            "short_rarl_return_cost_before": float(base_selector_eval.get("short_rarl_return_cost", float("nan"))),
            "short_rarl_return_cost_after": float(new_selector_eval.get("short_rarl_return_cost", float("nan"))),
            "short_rarl_return_cost_change": float(new_selector_eval.get("short_rarl_return_cost", float("nan")) - base_selector_eval.get("short_rarl_return_cost", float("nan"))),
            "unclipped_actor_surrogate_change": float(new_eval["policy_unclipped_component"] - base_eval["policy_unclipped_component"]),
            "value_loss_change": float(new_selector_eval.get("value_loss", float("nan")) - base_selector_eval.get("value_loss", float("nan"))),
            "entropy_change": float(new_selector_eval.get("entropy_loss", float("nan")) - base_selector_eval.get("entropy_loss", float("nan"))),
        }
        self.last_step_metrics = metrics
        self._write_diagnostics_row(metrics)
        self._step_index += 1
        return new_eval["loss_tensor"]


class ProposedNoGPerfLyapOptimizer(_PerformanceAlignedLyapunovQPBase):
    def __init__(self, params: Iterable[torch.nn.Parameter], **kwargs):
        kwargs = dict(kwargs)
        kwargs["disable_g"] = True
        super().__init__(params, **kwargs)


class ProposedQPPerfLyapOptimizer(_PerformanceAlignedLyapunovQPBase):
    pass
