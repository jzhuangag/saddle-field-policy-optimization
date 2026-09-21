from __future__ import annotations

import csv
import math
import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.optim import Optimizer

from models.optimizers import (
    NamedParams,
    NamedTensorMap,
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


def _state_is_finite(state: NamedTensorMap, names: Sequence[str]) -> bool:
    return all(torch.isfinite(state[name]).all() for name in names)


def _windows_safe_dir(path: str) -> str:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return path


class _ClosedLyapunovDriftBase(Optimizer):
    requires_eval_closure = True
    needs_return_merit = True

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1e-3,
        optimizer_scope: str = "full_policy",
        lambda_F: float = 1.0,
        lambda_R: float = 1.0,
        actor_weight: float = 1.0,
        logstd_weight: float = 1.0,
        critic_weight: float = 1.0,
        fd_eps: float = 1e-3,
        beta_probe: Optional[float] = None,
        gamma_probe: Optional[float] = None,
        ridge: float = 1e-8,
        beta_max: Optional[float] = None,
        gamma_max: Optional[float] = None,
        max_update_norm: float = 0.005,
        qp_eps: float = 1e-8,
        allow_fallback_to_egm: bool = False,
        fallback_tolerance: float = 0.0,
        cost_mode: str = "mixed_rarl_unclipped_actor_surrogate_cost",
        short_return_horizon: int = 16,
        short_return_episodes: int = 1,
        short_return_seed_offset: int = 0,
        diagnostics_csv_path: Optional[str] = None,
        role: str = "policy",
        **ignored_kwargs,
    ):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        if optimizer_scope not in {"full_policy", "actor_game", "actor_logstd_only"}:
            raise ValueError(f"Unsupported optimizer_scope={optimizer_scope!r}")
        self.optimizer_scope = optimizer_scope
        self.lambda_F = float(lambda_F)
        self.lambda_R = float(lambda_R)
        self.actor_weight = float(actor_weight)
        self.logstd_weight = float(logstd_weight)
        self.critic_weight = float(critic_weight)
        self.fd_eps = max(float(fd_eps), 1e-12)
        self.beta_probe = None if beta_probe is None else max(float(beta_probe), 1e-12)
        self.gamma_probe = None if gamma_probe is None else max(float(gamma_probe), 1e-12)
        self.ridge = max(float(ridge), 0.0)
        self.beta_max = None if beta_max is None else max(float(beta_max), 0.0)
        self.gamma_max = None if gamma_max is None else max(float(gamma_max), 0.0)
        self.max_update_norm = float(max_update_norm)
        self.qp_eps = max(float(qp_eps), 1e-12)
        self.allow_fallback_to_egm = bool(allow_fallback_to_egm)
        self.fallback_tolerance = float(fallback_tolerance)
        self.cost_mode = str(cost_mode)
        self.short_return_horizon = max(int(short_return_horizon), 1)
        self.short_return_episodes = max(int(short_return_episodes), 1)
        self.short_return_seed_offset = int(short_return_seed_offset)
        self.diagnostics_csv_path = diagnostics_csv_path
        self.role = role
        self.last_step_metrics: Dict[str, float | int | str] = {}
        self._step_index = 0
        self._fieldnames = [
            "step_index",
            "role",
            "scope",
            "variant",
            "field_term_before",
            "field_term_after",
            "return_term_before",
            "return_term_after",
            "V_before",
            "V_after",
            "actual_drift",
            "V_after_noG",
            "actual_drift_noG",
            "V_after_QP",
            "actual_drift_QP",
            "V_after_plusG_QP",
            "actual_drift_plusG_QP",
            "V_after_minusG_QP",
            "actual_drift_minusG_QP",
            "V_after_EGM",
            "actual_drift_EGM",
            "predicted_drift_noG",
            "predicted_drift_QP",
            "egm_V_after",
            "egm_actual_drift",
            "QP_better_than_noG",
            "QP_better_than_EGM",
            "noG_better_than_QP",
            "plusG_better_than_noG",
            "minusG_better_than_noG",
            "sign_select_better_than_noG",
            "nog_safe_qp_better_than_noG",
            "chosen_step",
            "qp_selected_active_set",
            "qp_candidate_count",
            "q_qp",
            "q_nog_in_qp_space",
            "predicted_inclusion_gap",
            "predicted_inclusion_pass",
            "fallback_to_egm",
            "fallback_reason",
            "beta",
            "gamma",
            "gamma_active",
            "beta_plus",
            "gamma_plus",
            "beta_minus",
            "gamma_minus",
            "gamma_active_plus",
            "gamma_active_minus",
            "beta_raw",
            "gamma_raw",
            "beta_probe",
            "gamma_probe",
            "beta_max",
            "gamma_max",
            "field_norm",
            "G_norm",
            "cos_F_G",
            "non_collinearity",
            "G_contribution_ratio",
            "G_contribution_norm",
            "G_contribution_ratio_plus",
            "G_contribution_ratio_minus",
            "V_plus_G",
            "V_minus_G",
            "G_plus_improves",
            "G_minus_improves",
            "G_sign_preference",
            "update_norm_pre_cap",
            "update_norm_post_cap",
            "cap_active",
            "actor_update_norm",
            "logstd_update_norm",
            "critic_update_norm",
            "egm_actor_update_norm",
            "egm_logstd_update_norm",
            "egm_critic_update_norm",
            "return_merit_available",
            "return_merit_source_before",
            "return_merit_source_after",
            "approx_kl_after",
            "clip_fraction_after",
            "applied_vs_nog_candidate_rel_diff",
            "applied_v_after_minus_nog_candidate",
        ]
        self._ensure_diagnostics_header()

    @property
    def variant_name(self) -> str:
        raise NotImplementedError

    def _ensure_diagnostics_header(self) -> None:
        if not self.diagnostics_csv_path:
            return
        diagnostics_path = _windows_safe_dir(self.diagnostics_csv_path)
        if os.path.exists(diagnostics_path):
            return
        with open(diagnostics_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writeheader()

    def _write_diagnostics_row(self, row: Dict[str, object]) -> None:
        if not self.diagnostics_csv_path:
            return
        diagnostics_path = _windows_safe_dir(self.diagnostics_csv_path)
        with open(diagnostics_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writerow({field: row.get(field) for field in self._fieldnames})

    def _selected_names(self, named_params: NamedParams) -> List[str]:
        if self.optimizer_scope == "full_policy":
            return [name for name, _ in named_params]
        return [name for name, _ in named_params if classify_parameter_block(name) != "critic"]

    def _field_term(self, grads: NamedTensorMap, names: Sequence[str]) -> float:
        total = 0.0
        for name in names:
            block = classify_parameter_block(name)
            if block == "actor":
                weight = self.actor_weight
            elif block == "logstd":
                weight = self.logstd_weight
            else:
                weight = self.critic_weight
            if weight == 0.0:
                continue
            flat = grads[name].reshape(-1)
            total += 0.5 * weight * float(torch.dot(flat, flat).item())
        return float(total)

    def _return_term(
        self,
        *,
        theta_state: NamedTensorMap,
        selected_names: Sequence[str],
        candidate_merit_evaluator: Optional[Callable[[NamedTensorMap, Sequence[str]], Dict[str, float]]],
        eval_info: Dict[str, object],
    ) -> Tuple[float, str, int]:
        if candidate_merit_evaluator is not None:
            merit = candidate_merit_evaluator(theta_state, selected_names)
            for key in ("short_rarl_return_cost", "short_clean_return_cost", "mixed_merit"):
                value = float(merit.get(key, float("nan")))
                if math.isfinite(value):
                    return value, key, 1
        value = float(eval_info.get("policy_loss_unclipped", eval_info.get("policy_loss", 0.0)))
        return value, "surrogate_fallback", 0

    def _evaluate_state(
        self,
        *,
        eval_closure: Callable[..., Dict[str, object]],
        theta_state: NamedTensorMap,
        selected_names: Sequence[str],
        candidate_merit_evaluator: Optional[Callable[[NamedTensorMap, Sequence[str]], Dict[str, float]]],
    ) -> Dict[str, object]:
        info = eval_closure(theta_override=theta_state, backward=True, grad_scope_names=list(selected_names))
        grads_selected = {name: info["grads"][name].detach().clone() for name in selected_names}
        field_term = self._field_term(grads_selected, selected_names)
        return_term, return_source, merit_available = self._return_term(
            theta_state=theta_state,
            selected_names=selected_names,
            candidate_merit_evaluator=candidate_merit_evaluator,
            eval_info=info,
        )
        payload = dict(info)
        payload["grads_selected"] = grads_selected
        payload["field_term"] = float(field_term)
        payload["return_term"] = float(return_term)
        payload["return_source"] = return_source
        payload["return_merit_available"] = int(merit_available)
        payload["V"] = float(self.lambda_F * field_term + self.lambda_R * return_term)
        return payload

    def _g_map(
        self,
        *,
        theta_old: NamedTensorMap,
        f_map: NamedTensorMap,
        selected_names: Sequence[str],
        eval_closure: Callable[..., Dict[str, object]],
    ) -> NamedTensorMap:
        theta_plus = {name: tensor.clone() for name, tensor in theta_old.items()}
        for name in selected_names:
            theta_plus[name] = theta_old[name] + self.fd_eps * f_map[name]
        plus_info = eval_closure(theta_override=theta_plus, backward=True, grad_scope_names=list(selected_names))
        plus_grads = {name: plus_info["grads"][name].detach().clone() for name in selected_names}
        return {name: (plus_grads[name] - f_map[name]) / self.fd_eps for name in selected_names}

    def _quadratic_1d(
        self,
        *,
        v0: float,
        v1: float,
        v2: float,
        delta: float,
    ) -> Tuple[float, float]:
        h = (v2 - 2.0 * v1 + v0) / max(delta * delta, self.qp_eps)
        l = (v1 - v0) / max(delta, self.qp_eps) - 0.5 * h * delta
        return float(l), float(h)

    def _solve_nog(
        self,
        *,
        theta_old: NamedTensorMap,
        p_map: NamedTensorMap,
        selected_names: Sequence[str],
        v0: float,
        eval_state: Callable[[NamedTensorMap], Dict[str, object]],
        eta: float,
    ) -> Dict[str, object]:
        delta = self.beta_probe if self.beta_probe is not None else max(eta, self.qp_eps)
        theta_1 = {name: tensor.clone() for name, tensor in theta_old.items()}
        theta_2 = {name: tensor.clone() for name, tensor in theta_old.items()}
        for name in selected_names:
            theta_1[name] = theta_old[name] + delta * p_map[name]
            theta_2[name] = theta_old[name] + (2.0 * delta) * p_map[name]
        v1 = float(eval_state(theta_1)["V"])
        v2 = float(eval_state(theta_2)["V"])
        l_beta, h_bb = self._quadratic_1d(v0=v0, v1=v1, v2=v2, delta=delta)
        denom = h_bb + self.ridge
        beta_raw = -l_beta / denom if abs(denom) > self.qp_eps else 0.0
        beta = max(beta_raw, 0.0)
        if self.beta_max is not None:
            beta = min(beta, self.beta_max)
        update_map = {name: beta * p_map[name] for name in selected_names}
        predicted_drift = float(l_beta * beta + 0.5 * h_bb * beta * beta)
        return {
            "l_beta": float(l_beta),
            "h_bb": float(h_bb),
            "beta_raw": float(beta_raw),
            "beta": float(beta),
            "predicted_drift": predicted_drift,
            "update_map": update_map,
        }

    def _solve_qp(
        self,
        *,
        theta_old: NamedTensorMap,
        p_map: NamedTensorMap,
        g_map: NamedTensorMap,
        selected_names: Sequence[str],
        v0: float,
        eval_state: Callable[[NamedTensorMap], Dict[str, object]],
        eta: float,
    ) -> Dict[str, object]:
        db = self.beta_probe if self.beta_probe is not None else max(eta, self.qp_eps)
        dg = self.gamma_probe if self.gamma_probe is not None else max(eta * eta, self.qp_eps)

        def point(beta_scale: float, gamma_scale: float) -> float:
            theta_tmp = {name: tensor.clone() for name, tensor in theta_old.items()}
            for name in selected_names:
                theta_tmp[name] = theta_old[name] + beta_scale * p_map[name] + gamma_scale * g_map[name]
            return float(eval_state(theta_tmp)["V"])

        v_b = point(db, 0.0)
        v_2b = point(2.0 * db, 0.0)
        v_g = point(0.0, dg)
        v_2g = point(0.0, 2.0 * dg)
        v_bg = point(db, dg)

        l_beta, h_bb = self._quadratic_1d(v0=v0, v1=v_b, v2=v_2b, delta=db)
        l_gamma, h_gg = self._quadratic_1d(v0=v0, v1=v_g, v2=v_2g, delta=dg)
        h_bg = (v_bg - v_b - v_g + v0) / max(db * dg, self.qp_eps)

        def q_value(beta_value: float, gamma_value: float) -> float:
            beta_v = float(beta_value)
            gamma_v = float(gamma_value)
            return float(
                l_beta * beta_v
                + l_gamma * gamma_v
                + 0.5 * h_bb * beta_v * beta_v
                + h_bg * beta_v * gamma_v
                + 0.5 * h_gg * gamma_v * gamma_v
            )

        def clamp_beta(beta_value: float) -> float:
            value = max(float(beta_value), 0.0)
            if self.beta_max is not None:
                value = min(value, float(self.beta_max))
            return float(value)

        def clamp_gamma(gamma_value: float) -> float:
            value = max(float(gamma_value), 0.0)
            if self.gamma_max is not None:
                value = min(value, float(self.gamma_max))
            return float(value)

        def beta_feasible(beta_value: float) -> bool:
            if not math.isfinite(beta_value):
                return False
            if beta_value < -1e-10:
                return False
            if self.beta_max is not None and beta_value > float(self.beta_max) + 1e-10:
                return False
            return True

        def gamma_feasible(gamma_value: float) -> bool:
            if not math.isfinite(gamma_value):
                return False
            if gamma_value < -1e-10:
                return False
            if self.gamma_max is not None and gamma_value > float(self.gamma_max) + 1e-10:
                return False
            return True

        candidates: List[Tuple[str, float, float, float]] = []

        def add_candidate(name: str, beta_value: float, gamma_value: float) -> None:
            beta_v = float(beta_value)
            gamma_v = float(gamma_value)
            if not beta_feasible(beta_v) or not gamma_feasible(gamma_v):
                return
            if not math.isfinite(beta_v) or not math.isfinite(gamma_v):
                return
            qv = q_value(beta_v, gamma_v)
            if not math.isfinite(qv):
                return
            candidates.append((name, beta_v, gamma_v, qv))

        # 1D noG restriction: this must be feasible inside the 2D QP at gamma=0.
        nog_sol = self._solve_nog(
            theta_old=theta_old,
            p_map=p_map,
            selected_names=selected_names,
            v0=v0,
            eval_state=eval_state,
            eta=eta,
        )
        beta_noG = float(nog_sol["beta"])

        add_candidate("corner_00", 0.0, 0.0)
        add_candidate("edge_gamma0_nog", beta_noG, 0.0)

        # Interior candidate from regularized system, ranked on the original q.
        hessian_reg = torch.tensor(
            [
                [h_bb + self.ridge, h_bg],
                [h_bg, h_gg + self.ridge],
            ],
            dtype=torch.float64,
        )
        linear = torch.tensor([-l_beta, -l_gamma], dtype=torch.float64)
        beta_raw = 0.0
        gamma_raw = 0.0
        try:
            solution = torch.linalg.solve(hessian_reg, linear)
            beta_raw = float(solution[0].item())
            gamma_raw = float(solution[1].item())
            add_candidate("interior", beta_raw, gamma_raw)
        except RuntimeError:
            beta_raw = 0.0
            gamma_raw = 0.0

        # Edge beta = 0.
        denom_gg = h_gg + self.ridge
        gamma_edge0_raw = -l_gamma / denom_gg if abs(denom_gg) > self.qp_eps else 0.0
        add_candidate("edge_beta0", 0.0, clamp_gamma(gamma_edge0_raw))

        # Edge gamma = 0, explicit independent 1D solve and the unclamped stationary point.
        denom_bb = h_bb + self.ridge
        beta_gamma0_raw = -l_beta / denom_bb if abs(denom_bb) > self.qp_eps else 0.0
        add_candidate("edge_gamma0_stationary", clamp_beta(beta_gamma0_raw), 0.0)

        if self.beta_max is not None:
            beta_bound = float(self.beta_max)
            add_candidate("corner_betaMax_0", beta_bound, 0.0)
            gamma_on_beta_max_raw = -(l_gamma + h_bg * beta_bound) / denom_gg if abs(denom_gg) > self.qp_eps else 0.0
            add_candidate("edge_betaMax", beta_bound, clamp_gamma(gamma_on_beta_max_raw))
        if self.gamma_max is not None:
            gamma_bound = float(self.gamma_max)
            add_candidate("corner_0_gammaMax", 0.0, gamma_bound)
            beta_on_gamma_max_raw = -(l_beta + h_bg * gamma_bound) / denom_bb if abs(denom_bb) > self.qp_eps else 0.0
            add_candidate("edge_gammaMax", clamp_beta(beta_on_gamma_max_raw), gamma_bound)
        if self.beta_max is not None and self.gamma_max is not None:
            add_candidate("corner_betaMax_gammaMax", float(self.beta_max), float(self.gamma_max))

        # Deduplicate numerically-equivalent candidates and rank on the original q.
        dedup: Dict[Tuple[int, int], Tuple[str, float, float, float]] = {}
        for candidate in candidates:
            key = (round(candidate[1] / 1e-12), round(candidate[2] / 1e-12))
            if key not in dedup or candidate[3] < dedup[key][3]:
                dedup[key] = candidate
        ranked_candidates = sorted(dedup.values(), key=lambda item: item[3])
        if not ranked_candidates:
            ranked_candidates = [("corner_00_fallback", 0.0, 0.0, q_value(0.0, 0.0))]

        selected_name, beta, gamma, q_selected = ranked_candidates[0]
        update_map = {name: beta * p_map[name] + gamma * g_map[name] for name in selected_names}
        predicted_drift = float(q_selected)
        q_noG_in_qp_space = float(q_value(beta_noG, 0.0))
        return {
            "l_beta": float(l_beta),
            "l_gamma": float(l_gamma),
            "h_bb": float(h_bb),
            "h_bg": float(h_bg),
            "h_gg": float(h_gg),
            "beta_raw": float(beta_raw),
            "gamma_raw": float(gamma_raw),
            "beta": float(beta),
            "gamma": float(gamma),
            "predicted_drift": predicted_drift,
            "update_map": update_map,
            "db": float(db),
            "dg": float(dg),
            "beta_noG": beta_noG,
            "q_qp": float(q_selected),
            "q_nog_in_qp_space": q_noG_in_qp_space,
            "predicted_inclusion_gap": float(q_selected - q_noG_in_qp_space),
            "predicted_inclusion_pass": int(q_selected <= q_noG_in_qp_space + 1e-6 * max(1.0, abs(q_selected), abs(q_noG_in_qp_space))),
            "selected_active_set": selected_name,
            "candidate_count": len(ranked_candidates),
            "candidate_q_values": ";".join(f"{name}:{qv:.12e}" for name, _, _, qv in ranked_candidates),
        }

    def _cap_update(self, update_map: NamedTensorMap, selected_names: Sequence[str]) -> Tuple[NamedTensorMap, float, float, int]:
        update_vec = flatten_named_tensors(update_map, selected_names)
        norm_pre = tensor_norm(update_vec)
        if math.isfinite(self.max_update_norm) and self.max_update_norm > 0.0 and norm_pre > self.max_update_norm:
            scale = self.max_update_norm / max(norm_pre, self.qp_eps)
            capped = {name: update_map[name] * scale for name in selected_names}
            return capped, float(norm_pre), tensor_norm(flatten_named_tensors(capped, selected_names)), 1
        return update_map, float(norm_pre), float(norm_pre), 0

    def _egm_state(
        self,
        *,
        theta_old: NamedTensorMap,
        f_old: NamedTensorMap,
        selected_names: Sequence[str],
        eval_closure: Callable[..., Dict[str, object]],
        eta: float,
    ) -> Tuple[NamedTensorMap, NamedTensorMap]:
        theta_half = {name: tensor.clone() for name, tensor in theta_old.items()}
        for name in selected_names:
            theta_half[name] = theta_old[name] - eta * f_old[name]
        half_info = eval_closure(theta_override=theta_half, backward=True, grad_scope_names=list(selected_names))
        f_half = {name: half_info["grads"][name].detach().clone() for name in selected_names}
        theta_egm = {name: tensor.clone() for name, tensor in theta_old.items()}
        for name in selected_names:
            theta_egm[name] = theta_old[name] - eta * f_half[name]
        return theta_egm, f_half

    def _step_impl(
        self,
        *,
        theta_old: NamedTensorMap,
        selected_names: Sequence[str],
        base_eval: Dict[str, object],
        eval_state: Callable[[NamedTensorMap], Dict[str, object]],
        eval_closure: Callable[..., Dict[str, object]],
        eta: float,
    ) -> Dict[str, object]:
        raise NotImplementedError

    def step(  # type: ignore[override]
        self,
        closure: Optional[Callable[[], torch.Tensor]] = None,
        *,
        eval_closure: Optional[Callable[..., Dict[str, object]]] = None,
        named_params: Optional[NamedParams] = None,
        candidate_merit_evaluator: Optional[Callable[[NamedTensorMap, Sequence[str]], Dict[str, float]]] = None,
    ):
        del closure
        if eval_closure is None or named_params is None:
            raise ValueError("Closed-form Lyapunov optimizer requires eval_closure and named_params.")

        selected_names = self._selected_names(named_params)
        theta_old = clone_named_state(named_params)
        eta = float(self.param_groups[0]["lr"])

        def eval_state(theta_state: NamedTensorMap) -> Dict[str, object]:
            return self._evaluate_state(
                eval_closure=eval_closure,
                theta_state=theta_state,
                selected_names=selected_names,
                candidate_merit_evaluator=candidate_merit_evaluator,
            )

        base_eval = eval_state(theta_old)
        step_out = self._step_impl(
            theta_old=theta_old,
            selected_names=selected_names,
            base_eval=base_eval,
            eval_state=eval_state,
            eval_closure=eval_closure,
            eta=eta,
        )

        chosen_state = step_out["theta_candidate"]
        chosen_eval = eval_state(chosen_state)
        chosen_diff = named_difference(chosen_state, theta_old, selected_names)
        nog_state = step_out.get("theta_nog_candidate")
        nog_eval = eval_state(nog_state) if nog_state is not None else chosen_eval
        qp_state = step_out.get("theta_qp_candidate", chosen_state)
        qp_eval = eval_state(qp_state) if qp_state is not None else chosen_eval
        plus_state = step_out.get("theta_plus_candidate")
        plus_eval = eval_state(plus_state) if plus_state is not None else qp_eval
        minus_state = step_out.get("theta_minus_candidate")
        minus_eval = eval_state(minus_state) if minus_state is not None else qp_eval

        theta_egm, _ = self._egm_state(
            theta_old=theta_old,
            f_old=base_eval["grads_selected"],
            selected_names=selected_names,
            eval_closure=eval_closure,
            eta=eta,
        )
        egm_eval = eval_state(theta_egm)
        egm_diff = named_difference(theta_egm, theta_old, selected_names)
        nog_diff = named_difference(nog_state, theta_old, selected_names) if nog_state is not None else chosen_diff

        fallback_to_egm = 0
        fallback_reason = ""
        final_state = chosen_state
        final_eval = chosen_eval
        final_diff = chosen_diff

        if self.allow_fallback_to_egm:
            chosen_bad = not math.isfinite(float(chosen_eval["V"]))
            worse_than_egm = float(chosen_eval["V"]) > float(egm_eval["V"]) + self.fallback_tolerance
            if chosen_bad or worse_than_egm:
                fallback_to_egm = 1
                fallback_reason = "nonfinite_proposed" if chosen_bad else "egm_better_actual_V"
                final_state = theta_egm
                final_eval = egm_eval
                final_diff = egm_diff

        restore_named_state(named_params, final_state)

        f_vec = flatten_named_tensors(base_eval["grads_selected"], selected_names)
        g_vec = flatten_named_tensors(step_out.get("g_map", {name: torch.zeros_like(base_eval["grads_selected"][name]) for name in selected_names}), selected_names)
        cos_fg = cosine_similarity(f_vec, g_vec)
        non_collinearity = float(math.sqrt(max(0.0, 1.0 - (0.0 if not math.isfinite(cos_fg) else cos_fg * cos_fg))))
        field_norm = tensor_norm(f_vec)
        g_norm = tensor_norm(g_vec)
        g_contribution_norm = abs(float(step_out.get("gamma", 0.0))) * g_norm
        f_contribution_norm = abs(float(step_out.get("beta", 0.0))) * field_norm
        g_contribution_ratio = float(g_contribution_norm / max(f_contribution_norm + g_contribution_norm, self.qp_eps))
        beta_plus = float(step_out.get("beta_plus", step_out.get("beta", 0.0)))
        gamma_plus = float(step_out.get("gamma_plus", step_out.get("gamma", 0.0)))
        beta_minus = float(step_out.get("beta_minus", 0.0))
        gamma_minus = float(step_out.get("gamma_minus", 0.0))
        plus_g_contrib_norm = abs(gamma_plus) * g_norm
        plus_f_contrib_norm = abs(beta_plus) * field_norm
        minus_g_contrib_norm = abs(gamma_minus) * g_norm
        minus_f_contrib_norm = abs(beta_minus) * field_norm
        g_contribution_ratio_plus = float(plus_g_contrib_norm / max(plus_g_contrib_norm + plus_f_contrib_norm, self.qp_eps))
        g_contribution_ratio_minus = float(minus_g_contrib_norm / max(minus_g_contrib_norm + minus_f_contrib_norm, self.qp_eps))

        v_plus_g = float("nan")
        v_minus_g = float("nan")
        g_plus_improves = 0
        g_minus_improves = 0
        g_sign_preference = "none"
        if g_norm > self.qp_eps and _state_is_finite(step_out.get("g_map", {}), selected_names):
            eps_g = 1e-3 / max(g_norm, self.qp_eps)
            theta_plus_g = {name: tensor.clone() for name, tensor in theta_old.items()}
            theta_minus_g = {name: tensor.clone() for name, tensor in theta_old.items()}
            for name in selected_names:
                theta_plus_g[name] = theta_old[name] + eps_g * step_out["g_map"][name]
                theta_minus_g[name] = theta_old[name] - eps_g * step_out["g_map"][name]
            v_plus_g = float(eval_state(theta_plus_g)["V"])
            v_minus_g = float(eval_state(theta_minus_g)["V"])
            g_plus_improves = int(math.isfinite(v_plus_g) and v_plus_g < float(base_eval["V"]))
            g_minus_improves = int(math.isfinite(v_minus_g) and v_minus_g < float(base_eval["V"]))
            if g_plus_improves and not g_minus_improves:
                g_sign_preference = "plus"
            elif g_minus_improves and not g_plus_improves:
                g_sign_preference = "minus"
            elif g_plus_improves and g_minus_improves:
                g_sign_preference = "both"
            else:
                g_sign_preference = "neither"

        row = {
            "step_index": self._step_index,
            "role": self.role,
            "scope": self.optimizer_scope,
            "variant": self.variant_name,
            "field_term_before": float(base_eval["field_term"]),
            "field_term_after": float(final_eval["field_term"]),
            "return_term_before": float(base_eval["return_term"]),
            "return_term_after": float(final_eval["return_term"]),
            "V_before": float(base_eval["V"]),
            "V_after": float(final_eval["V"]),
            "actual_drift": float(final_eval["V"] - base_eval["V"]),
            "V_after_noG": float(nog_eval["V"]),
            "actual_drift_noG": float(nog_eval["V"] - base_eval["V"]),
            "V_after_QP": float(qp_eval["V"]),
            "actual_drift_QP": float(qp_eval["V"] - base_eval["V"]),
            "V_after_plusG_QP": float(plus_eval["V"]),
            "actual_drift_plusG_QP": float(plus_eval["V"] - base_eval["V"]),
            "V_after_minusG_QP": float(minus_eval["V"]),
            "actual_drift_minusG_QP": float(minus_eval["V"] - base_eval["V"]),
            "V_after_EGM": float(egm_eval["V"]),
            "actual_drift_EGM": float(egm_eval["V"] - base_eval["V"]),
            "predicted_drift_noG": float(step_out.get("predicted_drift_noG", 0.0)),
            "predicted_drift_QP": float(step_out.get("predicted_drift_QP", 0.0)),
            "egm_V_after": float(egm_eval["V"]),
            "egm_actual_drift": float(egm_eval["V"] - base_eval["V"]),
            "QP_better_than_noG": int(float(qp_eval["V"]) < float(nog_eval["V"])),
            "QP_better_than_EGM": int(float(qp_eval["V"]) < float(egm_eval["V"])),
            "noG_better_than_QP": int(float(nog_eval["V"]) < float(qp_eval["V"])),
            "plusG_better_than_noG": int(float(plus_eval["V"]) < float(nog_eval["V"])),
            "minusG_better_than_noG": int(float(minus_eval["V"]) < float(nog_eval["V"])),
            "sign_select_better_than_noG": int(str(step_out.get("chosen_step", "")) in {"plusG", "minusG"} and float(chosen_eval["V"]) < float(nog_eval["V"])),
            "nog_safe_qp_better_than_noG": int(str(step_out.get("chosen_step", "")) in {"plusG", "minusG"} and float(chosen_eval["V"]) < float(nog_eval["V"])),
            "chosen_step": str(step_out.get("chosen_step", "qp")),
            "qp_selected_active_set": str(step_out.get("selected_active_set", "")),
            "qp_candidate_count": int(step_out.get("candidate_count", 0)),
            "q_qp": float(step_out.get("q_qp", step_out.get("predicted_drift_QP", 0.0))),
            "q_nog_in_qp_space": float(step_out.get("q_nog_in_qp_space", step_out.get("predicted_drift_noG", 0.0))),
            "predicted_inclusion_gap": float(step_out.get("predicted_inclusion_gap", 0.0)),
            "predicted_inclusion_pass": int(step_out.get("predicted_inclusion_pass", 1)),
            "fallback_to_egm": int(fallback_to_egm),
            "fallback_reason": fallback_reason,
            "beta": float(step_out.get("beta", 0.0)),
            "gamma": float(step_out.get("gamma", 0.0)),
            "gamma_active": int(abs(float(step_out.get("gamma", 0.0))) > self.qp_eps),
            "beta_plus": beta_plus,
            "gamma_plus": gamma_plus,
            "beta_minus": beta_minus,
            "gamma_minus": gamma_minus,
            "gamma_active_plus": int(abs(gamma_plus) > self.qp_eps),
            "gamma_active_minus": int(abs(gamma_minus) > self.qp_eps),
            "beta_raw": float(step_out.get("beta_raw", 0.0)),
            "gamma_raw": float(step_out.get("gamma_raw", 0.0)),
            "beta_probe": float(step_out.get("db", self.beta_probe if self.beta_probe is not None else eta)),
            "gamma_probe": float(step_out.get("dg", self.gamma_probe if self.gamma_probe is not None else eta * eta)),
            "beta_max": float(self.beta_max) if self.beta_max is not None else float("nan"),
            "gamma_max": float(self.gamma_max) if self.gamma_max is not None else float("nan"),
            "field_norm": field_norm,
            "G_norm": g_norm,
            "cos_F_G": cos_fg,
            "non_collinearity": non_collinearity,
            "G_contribution_ratio": g_contribution_ratio,
            "G_contribution_norm": g_contribution_norm,
            "G_contribution_ratio_plus": g_contribution_ratio_plus,
            "G_contribution_ratio_minus": g_contribution_ratio_minus,
            "V_plus_G": v_plus_g,
            "V_minus_G": v_minus_g,
            "G_plus_improves": g_plus_improves,
            "G_minus_improves": g_minus_improves,
            "G_sign_preference": g_sign_preference,
            "gamma_active_frac": float(abs(float(step_out.get("gamma", 0.0))) > self.qp_eps),
            "zero_update_flag": int(tensor_norm(flatten_named_tensors(final_diff, selected_names)) <= self.qp_eps),
            "update_norm_pre_cap": float(step_out.get("update_norm_pre_cap", 0.0)),
            "update_norm_post_cap": float(step_out.get("update_norm_post_cap", 0.0)),
            "cap_active": int(step_out.get("cap_active", 0)),
            "actor_update_norm": block_norm(final_diff, selected_names, "actor"),
            "logstd_update_norm": block_norm(final_diff, selected_names, "logstd"),
            "critic_update_norm": block_norm(final_diff, selected_names, "critic"),
            "egm_actor_update_norm": block_norm(egm_diff, selected_names, "actor"),
            "egm_logstd_update_norm": block_norm(egm_diff, selected_names, "logstd"),
            "egm_critic_update_norm": block_norm(egm_diff, selected_names, "critic"),
            "return_merit_available": int(base_eval["return_merit_available"]),
            "return_merit_source_before": str(base_eval["return_source"]),
            "return_merit_source_after": str(final_eval["return_source"]),
            "approx_kl_after": float(final_eval["approx_kl"]),
            "clip_fraction_after": float(final_eval["clip_fraction"]),
            "applied_vs_nog_candidate_rel_diff": float(
                tensor_norm(flatten_named_tensors(named_difference(final_state, nog_state, selected_names), selected_names))
                / max(1.0, tensor_norm(flatten_named_tensors(nog_diff, selected_names)))
            ) if nog_state is not None else 0.0,
            "applied_v_after_minus_nog_candidate": float(final_eval["V"] - nog_eval["V"]),
        }
        self.last_step_metrics = row
        self._write_diagnostics_row(row)
        self._step_index += 1
        return final_eval["loss_tensor"]


class ProposedNoGClosedLyapOptimizer(_ClosedLyapunovDriftBase):
    @property
    def variant_name(self) -> str:
        return "closed_nog"

    def _step_impl(
        self,
        *,
        theta_old: NamedTensorMap,
        selected_names: Sequence[str],
        base_eval: Dict[str, object],
        eval_state: Callable[[NamedTensorMap], Dict[str, object]],
        eval_closure: Callable[..., Dict[str, object]],
        eta: float,
    ) -> Dict[str, object]:
        p_map = {name: -base_eval["grads_selected"][name] for name in selected_names}
        sol = self._solve_nog(
            theta_old=theta_old,
            p_map=p_map,
            selected_names=selected_names,
            v0=float(base_eval["V"]),
            eval_state=eval_state,
            eta=eta,
        )
        update_map, norm_pre, norm_post, cap_active = self._cap_update(sol["update_map"], selected_names)
        theta_candidate = apply_state_delta(theta_old, update_map)
        return {
            "theta_candidate": theta_candidate,
            "theta_nog_candidate": theta_candidate,
            "theta_qp_candidate": theta_candidate,
            "beta": float(sol["beta"]),
            "gamma": 0.0,
            "beta_raw": float(sol["beta_raw"]),
            "gamma_raw": 0.0,
            "predicted_drift_noG": float(sol.get("predicted_drift", 0.0)),
            "predicted_drift_QP": float(sol.get("predicted_drift", 0.0)),
            "update_norm_pre_cap": float(norm_pre),
            "update_norm_post_cap": float(norm_post),
            "cap_active": int(cap_active),
            "db": float(self.beta_probe if self.beta_probe is not None else max(eta, self.qp_eps)),
            "g_map": {name: torch.zeros_like(base_eval["grads_selected"][name]) for name in selected_names},
        }


class ProposedQPClosedLyapOptimizer(_ClosedLyapunovDriftBase):
    @property
    def variant_name(self) -> str:
        return "closed_qp"

    def _step_impl(
        self,
        *,
        theta_old: NamedTensorMap,
        selected_names: Sequence[str],
        base_eval: Dict[str, object],
        eval_state: Callable[[NamedTensorMap], Dict[str, object]],
        eval_closure: Callable[..., Dict[str, object]],
        eta: float,
    ) -> Dict[str, object]:
        g_map = self._g_map(
            theta_old=theta_old,
            f_map=base_eval["grads_selected"],
            selected_names=selected_names,
            eval_closure=eval_closure,
        )
        if not _state_is_finite(g_map, selected_names) or tensor_norm(flatten_named_tensors(g_map, selected_names)) <= self.qp_eps:
            p_map = {name: -base_eval["grads_selected"][name] for name in selected_names}
            sol = self._solve_nog(
                theta_old=theta_old,
                p_map=p_map,
                selected_names=selected_names,
                v0=float(base_eval["V"]),
                eval_state=eval_state,
                eta=eta,
            )
            update_map, norm_pre, norm_post, cap_active = self._cap_update(sol["update_map"], selected_names)
            theta_candidate = apply_state_delta(theta_old, update_map)
            return {
                "theta_candidate": theta_candidate,
                "theta_nog_candidate": theta_candidate,
                "theta_qp_candidate": theta_candidate,
                "beta": float(sol["beta"]),
                "gamma": 0.0,
                "beta_raw": float(sol["beta_raw"]),
                "gamma_raw": 0.0,
                "predicted_drift_noG": float(sol.get("predicted_drift", 0.0)),
                "predicted_drift_QP": float(sol.get("predicted_drift", 0.0)),
                "update_norm_pre_cap": float(norm_pre),
                "update_norm_post_cap": float(norm_post),
                "cap_active": int(cap_active),
                "db": float(self.beta_probe if self.beta_probe is not None else max(eta, self.qp_eps)),
                "dg": float(self.gamma_probe if self.gamma_probe is not None else max(eta * eta, self.qp_eps)),
                "g_map": g_map,
            }

        p_map = {name: -base_eval["grads_selected"][name] for name in selected_names}
        nog_sol = self._solve_nog(
            theta_old=theta_old,
            p_map=p_map,
            selected_names=selected_names,
            v0=float(base_eval["V"]),
            eval_state=eval_state,
            eta=eta,
        )
        sol = self._solve_qp(
            theta_old=theta_old,
            p_map=p_map,
            g_map=g_map,
            selected_names=selected_names,
            v0=float(base_eval["V"]),
            eval_state=eval_state,
            eta=eta,
        )
        update_map, norm_pre, norm_post, cap_active = self._cap_update(sol["update_map"], selected_names)
        nog_update_map, _, _, _ = self._cap_update(nog_sol["update_map"], selected_names)
        return {
            "theta_candidate": apply_state_delta(theta_old, update_map),
            "theta_nog_candidate": apply_state_delta(theta_old, nog_update_map),
            "theta_qp_candidate": apply_state_delta(theta_old, update_map),
            "theta_plus_candidate": apply_state_delta(theta_old, update_map),
            "beta": float(sol["beta"]),
            "gamma": float(sol["gamma"]),
            "beta_raw": float(sol["beta_raw"]),
            "gamma_raw": float(sol["gamma_raw"]),
            "predicted_drift_noG": float(nog_sol.get("predicted_drift", 0.0)),
            "predicted_drift_QP": float(sol.get("predicted_drift", 0.0)),
            "beta_plus": float(sol["beta"]),
            "gamma_plus": float(sol["gamma"]),
            "update_norm_pre_cap": float(norm_pre),
            "update_norm_post_cap": float(norm_post),
            "cap_active": int(cap_active),
            "db": float(sol["db"]),
            "dg": float(sol["dg"]),
            "g_map": g_map,
            "chosen_step": "plusG",
        }


class ProposedQPClosedMinusGLyapOptimizer(_ClosedLyapunovDriftBase):
    @property
    def variant_name(self) -> str:
        return "closed_qp_minus_g"

    def _step_impl(
        self,
        *,
        theta_old: NamedTensorMap,
        selected_names: Sequence[str],
        base_eval: Dict[str, object],
        eval_state: Callable[[NamedTensorMap], Dict[str, object]],
        eval_closure: Callable[..., Dict[str, object]],
        eta: float,
    ) -> Dict[str, object]:
        g_map = self._g_map(
            theta_old=theta_old,
            f_map=base_eval["grads_selected"],
            selected_names=selected_names,
            eval_closure=eval_closure,
        )
        p_map = {name: -base_eval["grads_selected"][name] for name in selected_names}
        nog_sol = self._solve_nog(
            theta_old=theta_old,
            p_map=p_map,
            selected_names=selected_names,
            v0=float(base_eval["V"]),
            eval_state=eval_state,
            eta=eta,
        )
        if not _state_is_finite(g_map, selected_names) or tensor_norm(flatten_named_tensors(g_map, selected_names)) <= self.qp_eps:
            update_map, norm_pre, norm_post, cap_active = self._cap_update(nog_sol["update_map"], selected_names)
            theta_candidate = apply_state_delta(theta_old, update_map)
            return {
                "theta_candidate": theta_candidate,
                "theta_nog_candidate": theta_candidate,
                "theta_qp_candidate": theta_candidate,
                "beta": float(nog_sol["beta"]),
                "gamma": 0.0,
                "beta_raw": float(nog_sol["beta_raw"]),
                "gamma_raw": 0.0,
                "predicted_drift_noG": float(nog_sol.get("predicted_drift", 0.0)),
                "predicted_drift_QP": float(nog_sol.get("predicted_drift", 0.0)),
                "beta_minus": float(nog_sol["beta"]),
                "gamma_minus": 0.0,
                "update_norm_pre_cap": float(norm_pre),
                "update_norm_post_cap": float(norm_post),
                "cap_active": int(cap_active),
                "db": float(self.beta_probe if self.beta_probe is not None else max(eta, self.qp_eps)),
                "dg": float(self.gamma_probe if self.gamma_probe is not None else max(eta * eta, self.qp_eps)),
                "g_map": g_map,
                "chosen_step": "noG",
            }

        g_map_minus = {name: -g_map[name] for name in selected_names}
        minus_sol = self._solve_qp(
            theta_old=theta_old,
            p_map=p_map,
            g_map=g_map_minus,
            selected_names=selected_names,
            v0=float(base_eval["V"]),
            eval_state=eval_state,
            eta=eta,
        )
        minus_update_map, norm_pre, norm_post, cap_active = self._cap_update(minus_sol["update_map"], selected_names)
        nog_update_map, _, _, _ = self._cap_update(nog_sol["update_map"], selected_names)
        return {
            "theta_candidate": apply_state_delta(theta_old, minus_update_map),
            "theta_nog_candidate": apply_state_delta(theta_old, nog_update_map),
            "theta_qp_candidate": apply_state_delta(theta_old, minus_update_map),
            "theta_minus_candidate": apply_state_delta(theta_old, minus_update_map),
            "beta": float(minus_sol["beta"]),
            "gamma": float(minus_sol["gamma"]),
            "beta_raw": float(minus_sol["beta_raw"]),
            "gamma_raw": float(minus_sol["gamma_raw"]),
            "predicted_drift_noG": float(nog_sol.get("predicted_drift", 0.0)),
            "predicted_drift_QP": float(minus_sol.get("predicted_drift", 0.0)),
            "beta_minus": float(minus_sol["beta"]),
            "gamma_minus": float(minus_sol["gamma"]),
            "update_norm_pre_cap": float(norm_pre),
            "update_norm_post_cap": float(norm_post),
            "cap_active": int(cap_active),
            "db": float(minus_sol["db"]),
            "dg": float(minus_sol["dg"]),
            "g_map": g_map,
            "chosen_step": "minusG",
        }


class ProposedQPClosedSignSelectLyapOptimizer(_ClosedLyapunovDriftBase):
    @property
    def variant_name(self) -> str:
        return "closed_qp_sign_select"

    def _step_impl(
        self,
        *,
        theta_old: NamedTensorMap,
        selected_names: Sequence[str],
        base_eval: Dict[str, object],
        eval_state: Callable[[NamedTensorMap], Dict[str, object]],
        eval_closure: Callable[..., Dict[str, object]],
        eta: float,
    ) -> Dict[str, object]:
        g_map = self._g_map(
            theta_old=theta_old,
            f_map=base_eval["grads_selected"],
            selected_names=selected_names,
            eval_closure=eval_closure,
        )
        p_map = {name: -base_eval["grads_selected"][name] for name in selected_names}
        nog_sol = self._solve_nog(
            theta_old=theta_old,
            p_map=p_map,
            selected_names=selected_names,
            v0=float(base_eval["V"]),
            eval_state=eval_state,
            eta=eta,
        )
        nog_update_map, _, _, _ = self._cap_update(nog_sol["update_map"], selected_names)
        theta_nog = apply_state_delta(theta_old, nog_update_map)
        if not _state_is_finite(g_map, selected_names) or tensor_norm(flatten_named_tensors(g_map, selected_names)) <= self.qp_eps:
            return {
                "theta_candidate": theta_nog,
                "theta_nog_candidate": theta_nog,
                "theta_qp_candidate": theta_nog,
                "beta": float(nog_sol["beta"]),
                "gamma": 0.0,
                "beta_raw": float(nog_sol["beta_raw"]),
                "gamma_raw": 0.0,
                "predicted_drift_noG": float(nog_sol.get("predicted_drift", 0.0)),
                "predicted_drift_QP": float(nog_sol.get("predicted_drift", 0.0)),
                "update_norm_pre_cap": 0.0,
                "update_norm_post_cap": 0.0,
                "cap_active": 0,
                "db": float(self.beta_probe if self.beta_probe is not None else max(eta, self.qp_eps)),
                "dg": float(self.gamma_probe if self.gamma_probe is not None else max(eta * eta, self.qp_eps)),
                "g_map": g_map,
                "chosen_step": "noG",
            }

        plus_sol = self._solve_qp(
            theta_old=theta_old,
            p_map=p_map,
            g_map=g_map,
            selected_names=selected_names,
            v0=float(base_eval["V"]),
            eval_state=eval_state,
            eta=eta,
        )
        g_map_minus = {name: -g_map[name] for name in selected_names}
        minus_sol = self._solve_qp(
            theta_old=theta_old,
            p_map=p_map,
            g_map=g_map_minus,
            selected_names=selected_names,
            v0=float(base_eval["V"]),
            eval_state=eval_state,
            eta=eta,
        )
        plus_update_map, plus_norm_pre, plus_norm_post, plus_cap_active = self._cap_update(plus_sol["update_map"], selected_names)
        minus_update_map, minus_norm_pre, minus_norm_post, minus_cap_active = self._cap_update(minus_sol["update_map"], selected_names)
        theta_plus = apply_state_delta(theta_old, plus_update_map)
        theta_minus = apply_state_delta(theta_old, minus_update_map)
        plus_eval = eval_state(theta_plus)
        minus_eval = eval_state(theta_minus)
        use_plus = float(plus_eval["V"]) <= float(minus_eval["V"])
        chosen_theta = theta_plus if use_plus else theta_minus
        chosen_sol = plus_sol if use_plus else minus_sol
        chosen_norm_pre = plus_norm_pre if use_plus else minus_norm_pre
        chosen_norm_post = plus_norm_post if use_plus else minus_norm_post
        chosen_cap_active = plus_cap_active if use_plus else minus_cap_active
        return {
            "theta_candidate": chosen_theta,
            "theta_nog_candidate": theta_nog,
            "theta_qp_candidate": chosen_theta,
            "theta_plus_candidate": theta_plus,
            "theta_minus_candidate": theta_minus,
            "beta": float(chosen_sol["beta"]),
            "gamma": float(chosen_sol["gamma"]),
            "beta_raw": float(chosen_sol["beta_raw"]),
            "gamma_raw": float(chosen_sol["gamma_raw"]),
            "predicted_drift_noG": float(nog_sol.get("predicted_drift", 0.0)),
            "predicted_drift_QP": float(chosen_sol.get("predicted_drift", 0.0)),
            "beta_plus": float(plus_sol["beta"]),
            "gamma_plus": float(plus_sol["gamma"]),
            "beta_minus": float(minus_sol["beta"]),
            "gamma_minus": float(minus_sol["gamma"]),
            "update_norm_pre_cap": float(chosen_norm_pre),
            "update_norm_post_cap": float(chosen_norm_post),
            "cap_active": int(chosen_cap_active),
            "db": float(chosen_sol["db"]),
            "dg": float(chosen_sol["dg"]),
            "g_map": g_map,
            "chosen_step": "plusG" if use_plus else "minusG",
        }


class ProposedQPNogSafeSignSelectLyapOptimizer(_ClosedLyapunovDriftBase):
    @property
    def variant_name(self) -> str:
        return "closed_qp_nog_safe_sign_select"

    def _step_impl(
        self,
        *,
        theta_old: NamedTensorMap,
        selected_names: Sequence[str],
        base_eval: Dict[str, object],
        eval_state: Callable[[NamedTensorMap], Dict[str, object]],
        eval_closure: Callable[..., Dict[str, object]],
        eta: float,
    ) -> Dict[str, object]:
        g_map = self._g_map(
            theta_old=theta_old,
            f_map=base_eval["grads_selected"],
            selected_names=selected_names,
            eval_closure=eval_closure,
        )
        p_map = {name: -base_eval["grads_selected"][name] for name in selected_names}
        nog_sol = self._solve_nog(
            theta_old=theta_old,
            p_map=p_map,
            selected_names=selected_names,
            v0=float(base_eval["V"]),
            eval_state=eval_state,
            eta=eta,
        )
        nog_update_map, nog_norm_pre, nog_norm_post, nog_cap_active = self._cap_update(nog_sol["update_map"], selected_names)
        theta_nog = apply_state_delta(theta_old, nog_update_map)
        nog_eval = eval_state(theta_nog)
        if not _state_is_finite(g_map, selected_names) or tensor_norm(flatten_named_tensors(g_map, selected_names)) <= self.qp_eps:
            return {
                "theta_candidate": theta_nog,
                "theta_nog_candidate": theta_nog,
                "theta_qp_candidate": theta_nog,
                "beta": float(nog_sol["beta"]),
                "gamma": 0.0,
                "beta_raw": float(nog_sol["beta_raw"]),
                "gamma_raw": 0.0,
                "predicted_drift_noG": float(nog_sol.get("predicted_drift", 0.0)),
                "predicted_drift_QP": float(nog_sol.get("predicted_drift", 0.0)),
                "update_norm_pre_cap": float(nog_norm_pre),
                "update_norm_post_cap": float(nog_norm_post),
                "cap_active": int(nog_cap_active),
                "db": float(self.beta_probe if self.beta_probe is not None else max(eta, self.qp_eps)),
                "dg": float(self.gamma_probe if self.gamma_probe is not None else max(eta * eta, self.qp_eps)),
                "g_map": g_map,
                "chosen_step": "noG",
            }

        plus_sol = self._solve_qp(
            theta_old=theta_old,
            p_map=p_map,
            g_map=g_map,
            selected_names=selected_names,
            v0=float(base_eval["V"]),
            eval_state=eval_state,
            eta=eta,
        )
        g_map_minus = {name: -g_map[name] for name in selected_names}
        minus_sol = self._solve_qp(
            theta_old=theta_old,
            p_map=p_map,
            g_map=g_map_minus,
            selected_names=selected_names,
            v0=float(base_eval["V"]),
            eval_state=eval_state,
            eta=eta,
        )
        plus_update_map, plus_norm_pre, plus_norm_post, plus_cap_active = self._cap_update(plus_sol["update_map"], selected_names)
        minus_update_map, minus_norm_pre, minus_norm_post, minus_cap_active = self._cap_update(minus_sol["update_map"], selected_names)
        theta_plus = apply_state_delta(theta_old, plus_update_map)
        theta_minus = apply_state_delta(theta_old, minus_update_map)
        plus_eval = eval_state(theta_plus)
        minus_eval = eval_state(theta_minus)
        candidates = [
            ("noG", theta_nog, nog_eval, nog_sol, nog_norm_pre, nog_norm_post, nog_cap_active),
            ("plusG", theta_plus, plus_eval, plus_sol, plus_norm_pre, plus_norm_post, plus_cap_active),
            ("minusG", theta_minus, minus_eval, minus_sol, minus_norm_pre, minus_norm_post, minus_cap_active),
        ]
        chosen_step, chosen_theta, chosen_eval, chosen_sol, chosen_norm_pre, chosen_norm_post, chosen_cap_active = min(
            candidates,
            key=lambda item: float(item[2]["V"]),
        )
        return {
            "theta_candidate": chosen_theta,
            "theta_nog_candidate": theta_nog,
            "theta_qp_candidate": chosen_theta,
            "theta_plus_candidate": theta_plus,
            "theta_minus_candidate": theta_minus,
            "beta": float(chosen_sol["beta"]),
            "gamma": float(chosen_sol.get("gamma", 0.0)),
            "beta_raw": float(chosen_sol["beta_raw"]),
            "gamma_raw": float(chosen_sol.get("gamma_raw", 0.0)),
            "predicted_drift_noG": float(nog_sol.get("predicted_drift", 0.0)),
            "predicted_drift_QP": float(chosen_sol.get("predicted_drift", 0.0)),
            "beta_plus": float(plus_sol["beta"]),
            "gamma_plus": float(plus_sol["gamma"]),
            "beta_minus": float(minus_sol["beta"]),
            "gamma_minus": float(minus_sol["gamma"]),
            "update_norm_pre_cap": float(chosen_norm_pre),
            "update_norm_post_cap": float(chosen_norm_post),
            "cap_active": int(chosen_cap_active),
            "db": float(chosen_sol.get("db", self.beta_probe if self.beta_probe is not None else max(eta, self.qp_eps))),
            "dg": float(chosen_sol.get("dg", self.gamma_probe if self.gamma_probe is not None else max(eta * eta, self.qp_eps))),
            "g_map": g_map,
            "chosen_step": chosen_step,
        }
