from __future__ import annotations

import csv
import math
import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.optim import Optimizer

from models.proposed_qp_new import (
    NamedParams,
    NamedTensorMap,
    _block_mean_square,
    _is_finite_named_map,
    apply_state_delta,
    block_norm,
    classify_parameter_block,
    clone_named_state,
    cosine_similarity,
    flatten_named_tensors,
    named_difference,
    restore_named_state,
    tensor_norm,
)


class _RawFGLyapunovQPBase(Optimizer):
    requires_eval_closure = True

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1.0,
        optimizer_scope: str = "full_policy",
        qp_fd_eps: float = 1e-3,
        qp_beta_probe: float = 1e-3,
        qp_gamma_probe: float = 1e-6,
        qp_ridge: float = 1e-8,
        qp_actor_weight: float = 1.0,
        qp_logstd_weight: float = 1.0,
        qp_critic_weight: float = 0.3,
        qp_beta_max: float = 1e-2,
        qp_gamma_max: float = 3e-5,
        qp_max_update_norm: float = float("inf"),
        qp_eps: float = 1e-8,
        diagnostics_csv_path: Optional[str] = None,
        role: str = "policy",
        disable_g: bool = False,
    ):
        super().__init__(params, defaults=dict(lr=lr))
        if optimizer_scope not in {"full_policy", "actor_game"}:
            raise ValueError(f"Unsupported optimizer_scope={optimizer_scope!r}")
        self.optimizer_scope = optimizer_scope
        self.qp_fd_eps = max(float(qp_fd_eps), 1e-12)
        self.qp_beta_probe = max(float(qp_beta_probe), 1e-12)
        self.qp_gamma_probe = max(float(qp_gamma_probe), 1e-12)
        self.qp_ridge = max(float(qp_ridge), 0.0)
        self.qp_actor_weight = float(qp_actor_weight)
        self.qp_logstd_weight = float(qp_logstd_weight)
        self.qp_critic_weight = float(qp_critic_weight)
        self.qp_beta_max = max(float(qp_beta_max), 0.0)
        self.qp_gamma_max = max(float(qp_gamma_max), 0.0)
        self.qp_max_update_norm = float(qp_max_update_norm)
        self.qp_eps = max(float(qp_eps), 1e-12)
        self.diagnostics_csv_path = diagnostics_csv_path
        self.role = role
        self.disable_g = bool(disable_g)
        self.last_step_metrics: Dict[str, float | int | str] = {}
        self._step_index = 0
        self._fieldnames = [
            "step_index",
            "active_role",
            "scope",
            "raw_fg_mode",
            "eta",
            "finite_difference_valid",
            "fd_eps",
            "F_raw_norm",
            "G_raw_norm",
            "cosine_F_G",
            "a",
            "b",
            "c",
            "h",
            "k",
            "H_det",
            "ridge_used",
            "q_condition_status",
            "beta_star_unclipped",
            "gamma_star_unclipped",
            "beta",
            "gamma",
            "beta_eff",
            "gamma_eff",
            "selected_case",
            "beta_at_bound",
            "gamma_at_bound",
            "gamma_active",
            "q_pred",
            "V_before",
            "V_after_actual",
            "actual_V_change",
            "loss_before",
            "loss_after_actual",
            "actual_loss_change",
            "approx_kl_after",
            "clip_fraction_after",
            "F_contribution_norm",
            "G_contribution_norm",
            "G_over_update_norm",
            "update_norm_pre_cap",
            "update_norm_post_cap",
            "cap_active",
            "actor_update_norm",
            "logstd_update_norm",
            "critic_update_norm",
            "zero_update_flag",
            "gamma_active_frac",
        ]
        self._ensure_diagnostics_header()

    def _ensure_diagnostics_header(self) -> None:
        if not self.diagnostics_csv_path:
            return
        os.makedirs(os.path.dirname(self.diagnostics_csv_path), exist_ok=True)
        if os.path.exists(self.diagnostics_csv_path):
            return
        with open(self.diagnostics_csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writeheader()

    def _write_diagnostics_row(self, metrics: Dict[str, object]) -> None:
        if not self.diagnostics_csv_path:
            return
        with open(self.diagnostics_csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writerow({field: metrics.get(field) for field in self._fieldnames})

    def _selected_names(self, named_params: NamedParams) -> List[str]:
        if self.optimizer_scope == "full_policy":
            return [name for name, _ in named_params]
        return [name for name, _ in named_params if classify_parameter_block(name) != "critic"]

    def _lyapunov_weights(self) -> Dict[str, float]:
        return {
            "actor": self.qp_actor_weight,
            "logstd": self.qp_logstd_weight,
            "critic": self.qp_critic_weight,
        }

    def _lyapunov_value(self, grads: NamedTensorMap, selected_names: Sequence[str]) -> float:
        weights = self._lyapunov_weights()
        total = 0.0
        for block, weight in weights.items():
            total += weight * _block_mean_square(grads, selected_names, block)
        return 0.5 * total

    def _evaluate_state(
        self,
        *,
        eval_closure: Callable[..., Dict[str, object]],
        theta_state: NamedTensorMap,
        selected_names: Sequence[str],
        backward: bool,
    ) -> Dict[str, object]:
        eval_info = eval_closure(theta_override=theta_state, backward=backward)
        grads = {name: eval_info["grads"][name].detach().clone() for name in selected_names}
        eval_info = dict(eval_info)
        eval_info["grads_selected"] = grads
        eval_info["V"] = self._lyapunov_value(grads, selected_names)
        return eval_info

    def _compute_g_raw(
        self,
        *,
        theta_old: NamedTensorMap,
        eval_closure: Callable[..., Dict[str, object]],
        selected_names: Sequence[str],
        f_raw_map: NamedTensorMap,
        base_eval: Dict[str, object],
    ) -> Tuple[NamedTensorMap, bool]:
        if self.disable_g:
            zero_map = {name: torch.zeros_like(base_eval["grads_selected"][name]) for name in selected_names}
            return zero_map, False
        theta_plus = {name: tensor.clone() for name, tensor in theta_old.items()}
        for name in selected_names:
            theta_plus[name] = theta_old[name] + self.qp_fd_eps * f_raw_map[name]
        plus_eval = self._evaluate_state(eval_closure=eval_closure, theta_state=theta_plus, selected_names=selected_names, backward=True)
        g_raw_map = {
            name: (plus_eval["grads_selected"][name] - base_eval["grads_selected"][name]) / self.qp_fd_eps
            for name in selected_names
        }
        valid = _is_finite_named_map(g_raw_map, selected_names) and tensor_norm(flatten_named_tensors(g_raw_map, selected_names)) > self.qp_eps
        if not valid:
            zero_map = {name: torch.zeros_like(base_eval["grads_selected"][name]) for name in selected_names}
            return zero_map, False
        return g_raw_map, True

    def _estimate_quadratic_coefficients(
        self,
        *,
        theta_old: NamedTensorMap,
        eval_closure: Callable[..., Dict[str, object]],
        selected_names: Sequence[str],
        f_raw_map: NamedTensorMap,
        g_raw_map: NamedTensorMap,
        v0: float,
    ) -> Dict[str, float | str]:
        p_map = {name: -f_raw_map[name] for name in selected_names}
        r_map = {name: g_raw_map[name] for name in selected_names}
        db = self.qp_beta_probe
        dg = self.qp_gamma_probe

        def V_at(beta_scale: float, gamma_scale: float) -> float:
            theta_tmp = {name: tensor.clone() for name, tensor in theta_old.items()}
            for name in selected_names:
                theta_tmp[name] = theta_old[name] + beta_scale * p_map[name] + gamma_scale * r_map[name]
            eval_info = self._evaluate_state(eval_closure=eval_closure, theta_state=theta_tmp, selected_names=selected_names, backward=True)
            return float(eval_info["V"])

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
        if (c <= self.qp_eps) or (k <= self.qp_eps) or (det <= self.qp_eps) or not math.isfinite(det):
            ridge_used = max(self.qp_ridge, 1e-12)
            c_r = c
            k_r = k
            det_r = det
            for _ in range(8):
                c_r = c + ridge_used
                k_r = k + ridge_used
                det_r = c_r * k_r - h * h
                if (c_r > self.qp_eps) and (k_r > self.qp_eps) and (det_r > self.qp_eps) and math.isfinite(det_r):
                    c = c_r
                    k = k_r
                    det = det_r
                    condition = "ridge_pd"
                    break
                ridge_used *= 10.0
            else:
                c = c_r
                k = k_r
                det = det_r
                condition = "indefinite_after_ridge"

        return {
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "h": float(h),
            "k": float(k),
            "H_det": float(det),
            "ridge_used": float(ridge_used),
            "q_condition_status": condition,
        }

    @staticmethod
    def _q_value(beta: float, gamma: float, a: float, b: float, c: float, h: float, k: float) -> float:
        return a * beta + b * gamma + 0.5 * c * beta * beta + h * beta * gamma + 0.5 * k * gamma * gamma

    def _solve_no_g(self, *, a: float, c: float) -> Dict[str, float | str]:
        beta_star = 0.0
        if c > self.qp_eps and math.isfinite(c):
            beta_star = -a / c
            candidates = [("interior" if 0.0 <= beta_star <= self.qp_beta_max else "beta_bound", min(max(beta_star, 0.0), self.qp_beta_max))]
        else:
            candidates = [("zero", 0.0), ("beta_max", self.qp_beta_max)]
        best_case = "zero"
        best_beta = 0.0
        best_q = float("inf")
        for case, beta in candidates:
            q_val = self._q_value(beta, 0.0, a, 0.0, c, 0.0, 1.0)
            if q_val < best_q:
                best_q = q_val
                best_beta = beta
                best_case = case
        return {
            "beta_star_unclipped": float(beta_star),
            "gamma_star_unclipped": 0.0,
            "beta": float(best_beta),
            "gamma": 0.0,
            "selected_case": best_case,
            "beta_at_bound": int(abs(best_beta - self.qp_beta_max) <= self.qp_eps and self.qp_beta_max > 0.0),
            "gamma_at_bound": 0,
            "gamma_active": 0,
            "q_pred": float(best_q),
        }

    def _solve_two_direction_qp(
        self,
        *,
        a: float,
        b: float,
        c: float,
        h: float,
        k: float,
        det: float,
        condition: str,
    ) -> Dict[str, float | str]:
        candidates: List[Tuple[str, float, float]] = [
            ("corner_00", 0.0, 0.0),
            ("corner_b0", self.qp_beta_max, 0.0),
            ("corner_0g", 0.0, self.qp_gamma_max),
            ("corner_bg", self.qp_beta_max, self.qp_gamma_max),
        ]
        beta_star = 0.0
        gamma_star = 0.0
        interior_valid = False
        if condition in {"pd", "ridge_pd"} and det > self.qp_eps:
            beta_star = (-a * k + h * b) / det
            gamma_star = (-c * b + h * a) / det
            if math.isfinite(beta_star) and math.isfinite(gamma_star) and 0.0 <= beta_star <= self.qp_beta_max and 0.0 <= gamma_star <= self.qp_gamma_max:
                candidates.append(("interior", beta_star, gamma_star))
                interior_valid = True
        if c > self.qp_eps:
            candidates.append(("gamma0", min(max(-a / c, 0.0), self.qp_beta_max), 0.0))
            candidates.append(("gamma_max", min(max(-(a + h * self.qp_gamma_max) / c, 0.0), self.qp_beta_max), self.qp_gamma_max))
        if k > self.qp_eps:
            candidates.append(("beta0", 0.0, min(max(-b / k, 0.0), self.qp_gamma_max)))
            candidates.append(("beta_max", self.qp_beta_max, min(max(-(b + h * self.qp_beta_max) / k, 0.0), self.qp_gamma_max)))

        best_case = "corner_00"
        best_beta = 0.0
        best_gamma = 0.0
        best_q = float("inf")
        for case, beta, gamma in candidates:
            q_val = self._q_value(beta, gamma, a, b, c, h, k)
            if q_val < best_q:
                best_q = q_val
                best_beta = beta
                best_gamma = gamma
                best_case = case
        return {
            "beta_star_unclipped": float(beta_star),
            "gamma_star_unclipped": float(gamma_star),
            "beta": float(best_beta),
            "gamma": float(best_gamma),
            "selected_case": best_case,
            "beta_at_bound": int(abs(best_beta - self.qp_beta_max) <= self.qp_eps and self.qp_beta_max > 0.0),
            "gamma_at_bound": int(abs(best_gamma - self.qp_gamma_max) <= self.qp_eps and self.qp_gamma_max > 0.0),
            "gamma_active": int(best_gamma > self.qp_eps),
            "q_pred": float(best_q),
            "interior_solution_valid": int(interior_valid),
        }

    def step(
        self,
        closure: Optional[Callable[[], torch.Tensor]] = None,
        *,
        eval_closure: Optional[Callable[..., Dict[str, object]]] = None,
        named_params: Optional[NamedParams] = None,
    ):
        del closure
        if eval_closure is None or named_params is None:
            raise ValueError("rawFG proposed optimizers require eval_closure and named_params")

        selected_names = self._selected_names(named_params)
        theta_old = clone_named_state(named_params)
        eta = float(self.param_groups[0]["lr"])
        base_eval = self._evaluate_state(eval_closure=eval_closure, theta_state=theta_old, selected_names=selected_names, backward=True)
        f_raw_map = {name: base_eval["grads_selected"][name].detach().clone() for name in selected_names}
        f_raw_vec = flatten_named_tensors(f_raw_map, selected_names)
        g_raw_map, finite_difference_valid = self._compute_g_raw(
            theta_old=theta_old,
            eval_closure=eval_closure,
            selected_names=selected_names,
            f_raw_map=f_raw_map,
            base_eval=base_eval,
        )
        g_raw_vec = flatten_named_tensors(g_raw_map, selected_names)

        coeffs = self._estimate_quadratic_coefficients(
            theta_old=theta_old,
            eval_closure=eval_closure,
            selected_names=selected_names,
            f_raw_map=f_raw_map,
            g_raw_map=g_raw_map,
            v0=float(base_eval["V"]),
        )
        if self.disable_g or (not finite_difference_valid):
            solution = self._solve_no_g(a=float(coeffs["a"]), c=float(coeffs["c"]))
        else:
            solution = self._solve_two_direction_qp(
                a=float(coeffs["a"]),
                b=float(coeffs["b"]),
                c=float(coeffs["c"]),
                h=float(coeffs["h"]),
                k=float(coeffs["k"]),
                det=float(coeffs["H_det"]),
                condition=str(coeffs["q_condition_status"]),
            )

        beta = float(solution["beta"])
        gamma = float(solution["gamma"])
        update_map: NamedTensorMap = {
            name: eta * ((-beta * f_raw_map[name]) + (gamma * g_raw_map[name])) for name in selected_names
        }
        update_vec_pre = flatten_named_tensors(update_map, selected_names)
        update_norm_pre = tensor_norm(update_vec_pre)
        update_norm_post = update_norm_pre
        cap_active = 0
        if math.isfinite(self.qp_max_update_norm) and self.qp_max_update_norm > 0.0 and update_norm_pre > self.qp_max_update_norm:
            scale = self.qp_max_update_norm / max(update_norm_pre, self.qp_eps)
            for name in selected_names:
                update_map[name] = update_map[name] * scale
            update_vec_post = flatten_named_tensors(update_map, selected_names)
            update_norm_post = tensor_norm(update_vec_post)
            cap_active = 1
        else:
            update_vec_post = update_vec_pre

        theta_new = apply_state_delta(theta_old, update_map)
        new_eval = self._evaluate_state(eval_closure=eval_closure, theta_state=theta_new, selected_names=selected_names, backward=True)
        restore_named_state(named_params, theta_new)
        diff = named_difference(theta_new, theta_old, selected_names)

        beta_eff = eta * beta
        gamma_eff = eta * gamma
        f_contribution_norm = abs(beta_eff) * tensor_norm(f_raw_vec)
        g_contribution_norm = abs(gamma_eff) * tensor_norm(g_raw_vec)
        metrics: Dict[str, object] = {
            "step_index": self._step_index,
            "active_role": self.role,
            "scope": self.optimizer_scope,
            "raw_fg_mode": 1,
            "eta": eta,
            "finite_difference_valid": int(finite_difference_valid),
            "fd_eps": self.qp_fd_eps,
            "F_raw_norm": tensor_norm(f_raw_vec),
            "G_raw_norm": tensor_norm(g_raw_vec),
            "cosine_F_G": cosine_similarity(f_raw_vec, g_raw_vec),
            "a": float(coeffs["a"]),
            "b": float(coeffs["b"]),
            "c": float(coeffs["c"]),
            "h": float(coeffs["h"]),
            "k": float(coeffs["k"]),
            "H_det": float(coeffs["H_det"]),
            "ridge_used": float(coeffs["ridge_used"]),
            "q_condition_status": str(coeffs["q_condition_status"]),
            "beta_star_unclipped": float(solution["beta_star_unclipped"]),
            "gamma_star_unclipped": float(solution["gamma_star_unclipped"]),
            "beta": beta,
            "gamma": gamma,
            "beta_eff": beta_eff,
            "gamma_eff": gamma_eff,
            "selected_case": str(solution["selected_case"]),
            "beta_at_bound": int(solution["beta_at_bound"]),
            "gamma_at_bound": int(solution["gamma_at_bound"]),
            "gamma_active": int(solution["gamma_active"]),
            "q_pred": float(solution["q_pred"]),
            "V_before": float(base_eval["V"]),
            "V_after_actual": float(new_eval["V"]),
            "actual_V_change": float(new_eval["V"] - base_eval["V"]),
            "loss_before": float(base_eval["total_loss"]),
            "loss_after_actual": float(new_eval["total_loss"]),
            "actual_loss_change": float(new_eval["total_loss"] - base_eval["total_loss"]),
            "approx_kl_after": float(new_eval["approx_kl"]),
            "clip_fraction_after": float(new_eval["clip_fraction"]),
            "F_contribution_norm": f_contribution_norm,
            "G_contribution_norm": g_contribution_norm,
            "G_over_update_norm": g_contribution_norm / max(update_norm_post, self.qp_eps),
            "update_norm_pre_cap": update_norm_pre,
            "update_norm_post_cap": update_norm_post,
            "cap_active": cap_active,
            "actor_update_norm": block_norm(diff, selected_names, "actor"),
            "logstd_update_norm": block_norm(diff, selected_names, "logstd"),
            "critic_update_norm": block_norm(diff, selected_names, "critic"),
            "zero_update_flag": int(update_norm_post <= self.qp_eps),
            "gamma_active_frac": float(gamma > self.qp_eps),
        }
        self.last_step_metrics = metrics
        self._write_diagnostics_row(metrics)
        self._step_index += 1
        return new_eval["loss_tensor"]


class ProposedNoGRawFGOptimizer(_RawFGLyapunovQPBase):
    def __init__(self, params: Iterable[torch.nn.Parameter], **kwargs):
        kwargs = dict(kwargs)
        kwargs["disable_g"] = True
        super().__init__(params, **kwargs)


class ProposedQPRawFGOptimizer(_RawFGLyapunovQPBase):
    pass
