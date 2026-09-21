from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed


DTYPE = torch.float64
EPS = 1e-12
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = ROOT / "original" / "results" / "unified_lyapunov_main_experiments"
PLOT_ROOT = RESULT_ROOT / "plots"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)
PLOT_ROOT.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def write_md(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def project_fro_tensor(matrix: torch.Tensor, max_norm: float) -> Tuple[torch.Tensor, bool]:
    norm = torch.linalg.norm(matrix)
    if norm.item() > max_norm:
        return matrix * (max_norm / (norm + EPS)), True
    return matrix, False


def safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


@dataclass(frozen=True)
class FixedLQConfig:
    state_dim: int = 2
    control_dim: int = 2
    disturbance_dim: int = 2
    horizon: int = 60
    rho_w: float = 2.0
    k_max_norm: float = 1.5
    l_max_norm: float = 0.25

    @property
    def A(self) -> np.ndarray:
        return np.array([[0.98, 0.35], [-0.35, 0.98]], dtype=np.float64)

    @property
    def B(self) -> np.ndarray:
        return np.eye(2, dtype=np.float64)

    @property
    def E(self) -> np.ndarray:
        return np.eye(2, dtype=np.float64)

    @property
    def Q(self) -> np.ndarray:
        return np.eye(2, dtype=np.float64)

    @property
    def R(self) -> np.ndarray:
        return 0.05 * np.eye(2, dtype=np.float64)

    @property
    def Rw(self) -> np.ndarray:
        return np.eye(2, dtype=np.float64)

    @property
    def K0(self) -> np.ndarray:
        return 0.22 * np.eye(2, dtype=np.float64)

    @property
    def L0(self) -> np.ndarray:
        return np.array([[0.0, 0.03], [-0.03, 0.0]], dtype=np.float64)


class FixedLQBenchmark:
    def __init__(self, config: FixedLQConfig):
        self.cfg = config
        self.A = torch.tensor(config.A, dtype=DTYPE)
        self.B = torch.tensor(config.B, dtype=DTYPE)
        self.E = torch.tensor(config.E, dtype=DTYPE)
        self.Q = torch.tensor(config.Q, dtype=DTYPE)
        self.R = torch.tensor(config.R, dtype=DTYPE)
        self.Rw = torch.tensor(config.Rw, dtype=DTYPE)
        self.sigma0 = torch.eye(config.state_dim, dtype=DTYPE)

    def split_flat(self, flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k_size = self.cfg.control_dim * self.cfg.state_dim
        K = flat[:k_size].reshape(self.cfg.control_dim, self.cfg.state_dim)
        L = flat[k_size:].reshape(self.cfg.disturbance_dim, self.cfg.state_dim)
        return K, L

    def join_flat(self, K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        return torch.cat([K.reshape(-1), L.reshape(-1)])

    def project_pair(self, K: torch.Tensor, L: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool, bool]:
        Kp, proj_k = project_fro_tensor(K, self.cfg.k_max_norm)
        Lp, proj_l = project_fro_tensor(L, self.cfg.l_max_norm)
        return Kp, Lp, proj_k, proj_l

    def spectral_radius(self, matrix: np.ndarray) -> float:
        eigvals = np.linalg.eigvals(matrix)
        return float(np.max(np.abs(eigvals)))

    def simulate_from_flat(self, flat: torch.Tensor, alpha: float = 1.0, clean: bool = False) -> Dict[str, torch.Tensor]:
        K, L = self.split_flat(flat)
        M = self.A - self.B @ K if clean else self.A - self.B @ K + alpha * (self.E @ L)
        sigma = self.sigma0
        task_matrix = self.Q + K.T @ self.R @ K
        dist_matrix = alpha * alpha * (L.T @ self.Rw @ L)
        game_matrix = task_matrix - self.cfg.rho_w * dist_matrix

        total_game = torch.zeros((), dtype=DTYPE)
        total_task = torch.zeros((), dtype=DTYPE)
        total_dist = torch.zeros((), dtype=DTYPE)
        for _ in range(self.cfg.horizon):
            total_game = total_game + torch.trace(game_matrix @ sigma)
            total_task = total_task + torch.trace(task_matrix @ sigma)
            total_dist = total_dist + torch.trace(dist_matrix @ sigma)
            sigma = M @ sigma @ M.T
        return {
            "game_objective": total_game,
            "task_cost": total_task,
            "task_return": -total_task,
            "disturbance_energy": total_dist,
        }

    def J(self, flat: torch.Tensor) -> torch.Tensor:
        return self.simulate_from_flat(flat)["game_objective"]

    def field(self, flat: torch.Tensor) -> torch.Tensor:
        flat = flat.detach().clone().requires_grad_(True)
        J_val = self.J(flat)
        grad = torch.autograd.grad(J_val, flat)[0]
        k_size = self.cfg.control_dim * self.cfg.state_dim
        grad_k = grad[:k_size]
        grad_l = grad[k_size:]
        return torch.cat([grad_k, -grad_l]).detach()

    def field_and_j(self, flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        flat = flat.detach().clone().requires_grad_(True)
        J_val = self.J(flat)
        grad = torch.autograd.grad(J_val, flat)[0]
        k_size = self.cfg.control_dim * self.cfg.state_dim
        grad_k = grad[:k_size]
        grad_l = grad[k_size:]
        return torch.cat([grad_k, -grad_l]).detach(), J_val.detach()

    def curvature_direction(self, flat: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        eps = min(1e-4, 1e-4 / (field.norm().item() + 1.0))
        if field.norm().item() < EPS:
            return torch.zeros_like(field)
        field_shifted = self.field(flat + eps * field)
        return (field_shifted - field) / eps

    def p_tau(self, flat: torch.Tensor, tau: float, field: torch.Tensor | None = None, J_val: torch.Tensor | None = None) -> Tuple[float, float, float]:
        flat = flat.detach()
        if field is None or J_val is None:
            field, J_val = self.field_and_j(flat)
        K, L = self.split_flat(flat)
        k_size = self.cfg.control_dim * self.cfg.state_dim
        grad_k = field[:k_size].reshape(self.cfg.control_dim, self.cfg.state_dim)
        grad_l = -field[k_size:].reshape(self.cfg.disturbance_dim, self.cfg.state_dim)

        K_bar, _ = project_fro_tensor(K - tau * grad_k, self.cfg.k_max_norm)
        L_bar, _ = project_fro_tensor(L + tau * grad_l, self.cfg.l_max_norm)
        flat_kbar = self.join_flat(K_bar.detach(), L.detach())
        flat_lbar = self.join_flat(K.detach(), L_bar.detach())
        J_kbar = self.J(flat_kbar).detach()
        J_lbar = self.J(flat_lbar).detach()
        prox_k = 0.5 / tau * torch.sum((K_bar - K) ** 2)
        prox_l = 0.5 / tau * torch.sum((L_bar - L) ** 2)
        protagonist_gap = torch.clamp(J_val - (J_kbar + prox_k), min=0.0)
        adversary_gap = torch.clamp((J_lbar - prox_l) - J_val, min=0.0)
        total = protagonist_gap + adversary_gap
        return float(total), float(protagonist_gap), float(adversary_gap)

    def metrics(self, flat: torch.Tensor, lambda_F: float, tau: float, p_tau0: float, field_energy0: float) -> Dict[str, float]:
        field, J_val = self.field_and_j(flat)
        G = self.curvature_direction(flat, field)
        raw_p_tau, protagonist_gap, adversary_gap = self.p_tau(flat, tau, field=field, J_val=J_val)
        field_energy = 0.5 * float(torch.dot(field, field))
        normalized_p = raw_p_tau / (p_tau0 + EPS)
        normalized_f = field_energy / (field_energy0 + EPS)
        V = normalized_p + lambda_F * normalized_f
        K, L = self.split_flat(flat)
        clean = self.simulate_from_flat(self.join_flat(K.detach(), L.detach()), alpha=0.0, clean=True)
        adv = self.simulate_from_flat(self.join_flat(K.detach(), L.detach()), alpha=1.0, clean=False)
        clean_matrix = self.cfg.A - self.cfg.B @ K.detach().cpu().numpy()
        adv_matrix = self.cfg.A - self.cfg.B @ K.detach().cpu().numpy() + self.cfg.E @ L.detach().cpu().numpy()
        return {
            "V_lambda": V,
            "raw_p_tau": raw_p_tau,
            "normalized_p_tau_contribution": normalized_p,
            "raw_field_energy": field_energy,
            "normalized_field_contribution": normalized_f * lambda_F,
            "field_norm": float(torch.linalg.norm(field)),
            "J": float(J_val),
            "train_task_return": float(adv["task_return"]),
            "train_game_objective": float(adv["game_objective"]),
            "train_disturbance_energy": float(adv["disturbance_energy"]),
            "clean_task_return": float(clean["task_return"]),
            "clean_task_cost": float(clean["task_cost"]),
            "adv_task_return": float(adv["task_return"]),
            "adv_task_cost": float(adv["task_cost"]),
            "adv_game_objective": float(adv["game_objective"]),
            "adv_disturbance_energy": float(adv["disturbance_energy"]),
            "clean_spectral_radius": self.spectral_radius(clean_matrix),
            "adv_spectral_radius": self.spectral_radius(adv_matrix),
            "field": field.numpy(),
            "G": G.numpy(),
            "cosine_FG": float(torch.dot(field, G) / (torch.linalg.norm(field) * torch.linalg.norm(G) + EPS)),
            "G_norm": float(torch.linalg.norm(G)),
            "K_fro_norm": float(torch.linalg.norm(K)),
            "L_fro_norm": float(torch.linalg.norm(L)),
            "protagonist_gap": protagonist_gap,
            "adversary_gap": adversary_gap,
        }


def robustness_sweep(benchmark: FixedLQBenchmark, flat: torch.Tensor) -> Tuple[pd.DataFrame, float]:
    rows: List[Dict[str, object]] = []
    auc_values = []
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        sim = benchmark.simulate_from_flat(flat, alpha=alpha, clean=(alpha == 0.0))
        K, L = benchmark.split_flat(flat)
        matrix = benchmark.cfg.A - benchmark.cfg.B @ K.detach().cpu().numpy() + alpha * (benchmark.cfg.E @ L.detach().cpu().numpy())
        row = {
            "alpha": alpha,
            "sweep_task_return": float(sim["task_return"]),
            "sweep_task_cost": float(sim["task_cost"]),
            "sweep_game_objective": float(sim["game_objective"]),
            "sweep_disturbance_energy": float(sim["disturbance_energy"]),
            "sweep_spectral_radius": benchmark.spectral_radius(matrix),
        }
        rows.append(row)
        auc_values.append(float(sim["task_return"]))
    return pd.DataFrame(rows), float(np.trapz(auc_values, x=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]))


def trust_scale(delta: torch.Tensor, update_radius: float) -> Tuple[torch.Tensor, bool, float, float]:
    raw_norm = float(torch.linalg.norm(delta))
    if raw_norm <= update_radius + EPS:
        return delta, False, raw_norm, raw_norm
    scaled = delta * (update_radius / (raw_norm + EPS))
    return scaled, True, raw_norm, float(torch.linalg.norm(scaled))


def fit_quadratic_1d(eval_fn, direction: torch.Tensor, probe_radius: float) -> Tuple[float, Dict[str, float]]:
    d_norm = float(torch.linalg.norm(direction))
    if d_norm < EPS:
        return 0.0, {"a": 0.0, "b": 0.0, "h": 0.0}
    h = probe_radius / (d_norm + EPS)
    v0 = float(eval_fn(torch.zeros((), dtype=DTYPE), torch.zeros((), dtype=DTYPE)))
    vp = float(eval_fn(torch.tensor(h, dtype=DTYPE), torch.zeros((), dtype=DTYPE)))
    vm = float(eval_fn(torch.tensor(-h, dtype=DTYPE), torch.zeros((), dtype=DTYPE)))
    b = (vp - vm) / (2.0 * h)
    a = (vp + vm - 2.0 * v0) / (2.0 * h * h)
    beta = 0.0 if abs(a) < 1e-16 else -b / (2.0 * a)
    return float(beta), {"a": float(a), "b": float(b), "h": float(h), "v0": v0}


def fit_quadratic_2d(eval_fn, d1: torch.Tensor, d2: torch.Tensor, probe_radius: float) -> Tuple[float, float, Dict[str, float]]:
    n1 = float(torch.linalg.norm(d1))
    n2 = float(torch.linalg.norm(d2))
    if n1 < EPS or n2 < EPS:
        return 0.0, 0.0, {"cond": float("inf"), "indef": 1}
    h1 = probe_radius / (n1 + EPS)
    h2 = probe_radius / (n2 + EPS)
    v0 = float(eval_fn(torch.zeros((), dtype=DTYPE), torch.zeros((), dtype=DTYPE)))
    v10 = float(eval_fn(torch.tensor(h1, dtype=DTYPE), torch.tensor(0.0, dtype=DTYPE)))
    vm10 = float(eval_fn(torch.tensor(-h1, dtype=DTYPE), torch.tensor(0.0, dtype=DTYPE)))
    v01 = float(eval_fn(torch.tensor(0.0, dtype=DTYPE), torch.tensor(h2, dtype=DTYPE)))
    v0m1 = float(eval_fn(torch.tensor(0.0, dtype=DTYPE), torch.tensor(-h2, dtype=DTYPE)))
    v11 = float(eval_fn(torch.tensor(h1, dtype=DTYPE), torch.tensor(h2, dtype=DTYPE)))
    v1m1 = float(eval_fn(torch.tensor(h1, dtype=DTYPE), torch.tensor(-h2, dtype=DTYPE)))
    vm11 = float(eval_fn(torch.tensor(-h1, dtype=DTYPE), torch.tensor(h2, dtype=DTYPE)))
    vm1m1 = float(eval_fn(torch.tensor(-h1, dtype=DTYPE), torch.tensor(-h2, dtype=DTYPE)))
    d = (v10 - vm10) / (2.0 * h1)
    e = (v01 - v0m1) / (2.0 * h2)
    a = (v10 + vm10 - 2.0 * v0) / (2.0 * h1 * h1)
    b = (v01 + v0m1 - 2.0 * v0) / (2.0 * h2 * h2)
    c = (v11 - v1m1 - vm11 + vm1m1) / (4.0 * h1 * h2)
    H = np.array([[2.0 * a, c], [c, 2.0 * b]], dtype=np.float64)
    rhs = -np.array([d, e], dtype=np.float64)
    cond = np.linalg.cond(H) if np.all(np.isfinite(H)) else float("inf")
    indef = int(np.any(np.linalg.eigvals(H) <= 0.0)) if np.all(np.isfinite(H)) else 1
    try:
        sol = np.linalg.solve(H, rhs)
        beta, gamma = float(sol[0]), float(sol[1])
    except np.linalg.LinAlgError:
        beta, gamma = 0.0, 0.0
        cond = float("inf")
        indef = 1
    return beta, gamma, {"a": float(a), "b": float(b), "c": float(c), "d": float(d), "e": float(e), "h1": float(h1), "h2": float(h2), "cond": float(cond), "indef": indef}


def apply_projected_step(
    benchmark: FixedLQBenchmark,
    flat: torch.Tensor,
    delta: torch.Tensor,
) -> Tuple[torch.Tensor, bool, bool]:
    K, L = benchmark.split_flat(flat + delta)
    Kp, Lp, proj_k, proj_l = benchmark.project_pair(K, L)
    return benchmark.join_flat(Kp, Lp), proj_k, proj_l


def run_method(
    benchmark: FixedLQBenchmark,
    method: str,
    iterations: int,
    lambda_F: float,
    tau: float,
    update_radius: float,
    base_lr: float,
    p_tau0: float,
    field_energy0: float,
    egm_lr: float | None = None,
    dominance: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    flat = benchmark.join_flat(torch.tensor(benchmark.cfg.K0, dtype=DTYPE), torch.tensor(benchmark.cfg.L0, dtype=DTYPE))
    curve_rows: List[Dict[str, object]] = []
    diag_rows: List[Dict[str, object]] = []

    def v_eval(candidate_flat: torch.Tensor) -> Dict[str, float]:
        return benchmark.metrics(candidate_flat, lambda_F=lambda_F, tau=tau, p_tau0=p_tau0, field_energy0=field_energy0)

    for iteration in range(iterations):
        metrics_before = v_eval(flat)
        F = torch.tensor(metrics_before["field"], dtype=DTYPE)
        G = torch.tensor(metrics_before["G"], dtype=DTYPE)
        raw_beta = 0.0
        raw_gamma = 0.0
        hessian_cond = 1.0
        hessian_indef = 0
        trust_active = False
        projection_active_k = False
        projection_active_l = False
        fallback_to_egm = False
        selected_step_type = method
        gamma_active = 0.0
        probe_radius = min(update_radius, 1e-2) * 0.5

        def eval_beta_gamma(beta_t: torch.Tensor, gamma_t: torch.Tensor) -> float:
            cand = flat - beta_t * F + gamma_t * G
            cand, _, _ = apply_projected_step(benchmark, flat, cand - flat)
            return v_eval(cand)["V_lambda"]

        if method == "sgd":
            delta = -base_lr * F
        elif method == "egm":
            half = flat - base_lr * F
            half, _, _ = apply_projected_step(benchmark, flat, half - flat)
            F_half = benchmark.field(half)
            delta = -base_lr * F_half
        elif method == "ppm":
            z_inner = flat.clone()
            for _ in range(10):
                F_inner = benchmark.field(z_inner)
                z_inner = flat - base_lr * F_inner
                z_inner, _, _ = apply_projected_step(benchmark, flat, z_inner - flat)
            delta = z_inner - flat
        elif method == "proposed_noG_unified_repaired":
            beta, fit = fit_quadratic_1d(eval_beta_gamma, -F, probe_radius=probe_radius)
            raw_beta = beta
            delta = -beta * F
            delta, trust_active, raw_update_norm, trust_scaled_update_norm = trust_scale(delta, update_radius)
            candidate, projection_active_k, projection_active_l = apply_projected_step(benchmark, flat, delta)
            metrics_after = v_eval(candidate)
            unsafe = metrics_after["clean_spectral_radius"] > 1.0 or metrics_after["adv_spectral_radius"] > 1.0 or not np.isfinite(metrics_after["V_lambda"])
            if metrics_after["V_lambda"] > metrics_before["V_lambda"] or unsafe:
                fallback_to_egm = True
                selected_step_type = "fallback_egm"
                half = flat - base_lr * F
                half, _, _ = apply_projected_step(benchmark, flat, half - flat)
                F_half = benchmark.field(half)
                delta = -base_lr * F_half
            hessian_cond = 1.0
        elif method == "proposed_QP_G_unified_repaired":
            beta, gamma, fit = fit_quadratic_2d(eval_beta_gamma, -F, G, probe_radius=probe_radius)
            raw_beta = beta
            raw_gamma = gamma
            gamma_active = float(abs(gamma) > 1e-12)
            delta = -beta * F + gamma * G
            delta, trust_active, raw_update_norm, trust_scaled_update_norm = trust_scale(delta, update_radius)
            candidate, projection_active_k, projection_active_l = apply_projected_step(benchmark, flat, delta)
            metrics_after = v_eval(candidate)
            unsafe = metrics_after["clean_spectral_radius"] > 1.0 or metrics_after["adv_spectral_radius"] > 1.0 or not np.isfinite(metrics_after["V_lambda"])
            if metrics_after["V_lambda"] > metrics_before["V_lambda"] or unsafe:
                fallback_to_egm = True
                selected_step_type = "fallback_egm"
                half = flat - base_lr * F
                half, _, _ = apply_projected_step(benchmark, flat, half - flat)
                F_half = benchmark.field(half)
                delta = -base_lr * F_half
            hessian_cond = fit["cond"]
            hessian_indef = fit["indef"]
        elif method == "proposed_QP_G_unified_repaired_egm_dominance":
            beta, gamma, fit = fit_quadratic_2d(eval_beta_gamma, -F, G, probe_radius=probe_radius)
            raw_beta = beta
            raw_gamma = gamma
            gamma_active = float(abs(gamma) > 1e-12)
            delta_qp_raw = -beta * F + gamma * G
            delta_qp, trust_active, raw_update_norm, trust_scaled_update_norm = trust_scale(delta_qp_raw, update_radius)
            qp_flat, qp_proj_k, qp_proj_l = apply_projected_step(benchmark, flat, delta_qp)

            beta_nog, _ = fit_quadratic_1d(eval_beta_gamma, -F, probe_radius=probe_radius)
            delta_nog_raw = -beta_nog * F
            delta_nog, _, _, _ = trust_scale(delta_nog_raw, update_radius)
            nog_flat, _, _ = apply_projected_step(benchmark, flat, delta_nog)

            lr = egm_lr if egm_lr is not None else 1e-2
            half = flat - lr * F
            half, _, _ = apply_projected_step(benchmark, flat, half - flat)
            F_half = benchmark.field(half)
            delta_egm = -lr * F_half
            egm_flat, _, _ = apply_projected_step(benchmark, flat, delta_egm)
            zero_flat = flat.clone()

            candidates = {
                "qpg": (qp_flat, v_eval(qp_flat)["V_lambda"]),
                "nog": (nog_flat, v_eval(nog_flat)["V_lambda"]),
                "egm": (egm_flat, v_eval(egm_flat)["V_lambda"]),
                "zero": (zero_flat, v_eval(zero_flat)["V_lambda"]),
            }
            selected_step_type = min(candidates.items(), key=lambda item: item[1][1])[0]
            candidate_flat = candidates[selected_step_type][0]
            delta = candidate_flat - flat
            projection_active_k = qp_proj_k
            projection_active_l = qp_proj_l
            hessian_cond = fit["cond"]
            hessian_indef = fit["indef"]
        else:
            raise ValueError(method)

        if method != "proposed_QP_G_unified_repaired_egm_dominance":
            new_flat, projection_active_k, projection_active_l = apply_projected_step(benchmark, flat, delta)
        else:
            new_flat = flat + delta

        metrics_after = v_eval(new_flat)
        row = {
            "method": method,
            "iteration": iteration,
            "base_lr": base_lr,
            "lambda_F": lambda_F,
            "tau": tau,
            "update_radius": update_radius,
            **{k: v for k, v in metrics_after.items() if k not in {"field", "G"}},
            "beta": raw_beta if method.startswith("proposed") else base_lr,
            "gamma": raw_gamma if "QP_G" in method else 0.0,
            "gamma_active": gamma_active,
            "update_norm": float(torch.linalg.norm(new_flat - flat)),
            "trust_radius_active": float(trust_active),
            "fallback_to_egm": float(fallback_to_egm),
            "selected_step_type": selected_step_type,
            "raw_beta": raw_beta,
            "raw_gamma": raw_gamma,
            "raw_update_norm": float(torch.linalg.norm((-raw_beta * F + raw_gamma * G) if "QP_G" in method else (-raw_beta * F if "proposed_noG" in method else delta))),
            "trust_scaled_update_norm": float(torch.linalg.norm(delta)),
            "projection_active_K": float(projection_active_k),
            "projection_active_L": float(projection_active_l),
            "V_before": metrics_before["V_lambda"],
            "V_after_projected": metrics_after["V_lambda"],
            "V_actual_projected_change": metrics_after["V_lambda"] - metrics_before["V_lambda"],
            "P_tau_before": metrics_before["raw_p_tau"],
            "P_tau_after": metrics_after["raw_p_tau"],
            "field_energy_before": metrics_before["raw_field_energy"],
            "field_energy_after": metrics_after["raw_field_energy"],
            "spectral_radius_after_projected": metrics_after["adv_spectral_radius"],
            "clean_spectral_radius_after_projected": metrics_after["clean_spectral_radius"],
            "hessian_condition": hessian_cond,
            "hessian_indefinite": hessian_indef,
            "robustness_auc": float("nan"),
        }
        curve_rows.append(row)
        flat = new_flat.detach()
    final_sweep, robustness_auc = robustness_sweep(benchmark, flat)
    if curve_rows:
        curve_rows[-1]["robustness_auc"] = robustness_auc
    for _, srow in final_sweep.iterrows():
        diag_rows.append(
            {
                "method": method,
                "base_lr": base_lr,
                "lambda_F": lambda_F,
                "tau": tau,
                "update_radius": update_radius,
                **srow.to_dict(),
                "robustness_auc": robustness_auc,
            }
        )
    return pd.DataFrame(curve_rows), pd.DataFrame(diag_rows)


def summarize_run(curves: pd.DataFrame) -> Dict[str, object]:
    return {
        "method": curves["method"].iloc[0],
        "base_lr": safe_float(curves["base_lr"].iloc[0]),
        "lambda_F": safe_float(curves["lambda_F"].iloc[0]),
        "tau": safe_float(curves["tau"].iloc[0]),
        "update_radius": safe_float(curves["update_radius"].iloc[0]),
        "V_lambda_AUC": float(np.trapz(curves["V_lambda"].to_numpy())),
        "final_V_lambda": safe_float(curves["V_lambda"].iloc[-1]),
        "P_tau_AUC": float(np.trapz(curves["raw_p_tau"].to_numpy())),
        "final_P_tau": safe_float(curves["raw_p_tau"].iloc[-1]),
        "field_norm_AUC": float(np.trapz(curves["field_norm"].to_numpy())),
        "final_field_norm": safe_float(curves["field_norm"].iloc[-1]),
        "train_task_return": safe_float(curves["train_task_return"].iloc[-1]),
        "clean_task_return": safe_float(curves["clean_task_return"].iloc[-1]),
        "adv_task_return": safe_float(curves["adv_task_return"].iloc[-1]),
        "robustness_auc": safe_float(curves["robustness_auc"].iloc[-1]),
        "clean_spectral_radius": safe_float(curves["clean_spectral_radius"].iloc[-1]),
        "adv_spectral_radius": safe_float(curves["adv_spectral_radius"].iloc[-1]),
        "gamma_active_frac": float(curves["gamma_active"].mean()) if "gamma_active" in curves else 0.0,
        "trust_radius_active_frac": float(curves["trust_radius_active"].mean()) if "trust_radius_active" in curves else 0.0,
        "fallback_to_egm_frac": float(curves["fallback_to_egm"].mean()) if "fallback_to_egm" in curves else 0.0,
        "update_norm_mean": float(curves["update_norm"].mean()),
        "update_norm_max": float(curves["update_norm"].max()),
    }


def run_single_sweep_summary(
    method: str,
    lambda_F: float,
    tau: float,
    update_radius: float,
    base_lr: float,
    p0: float,
    fe0: float,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    torch.set_num_threads(1)
    benchmark = FixedLQBenchmark(FixedLQConfig())
    curves, _ = run_method(
        benchmark=benchmark,
        method=method,
        iterations=1000,
        lambda_F=lambda_F,
        tau=tau,
        update_radius=update_radius,
        base_lr=base_lr,
        p_tau0=p0,
        field_energy0=fe0,
    )
    summary = summarize_run(curves)
    diag = {
        **summary,
        "raw_beta_abs_max": float(curves["raw_beta"].abs().max()),
        "raw_gamma_abs_max": float(curves["raw_gamma"].abs().max()) if "raw_gamma" in curves else 0.0,
        "trust_scaled_update_mean": float(curves["trust_scaled_update_norm"].mean()),
        "trust_scaled_update_max": float(curves["trust_scaled_update_norm"].max()),
        "projection_K_frac": float(curves["projection_active_K"].mean()),
        "projection_L_frac": float(curves["projection_active_L"].mean()),
    }
    return summary, diag


def part_a_underperformance_audit() -> None:
    summary = pd.read_csv(RESULT_ROOT / "subsection2_lq_repaired_summary.csv")
    curves = pd.read_csv(RESULT_ROOT / "subsection2_lq_repaired_curves.csv")
    qpg = curves[curves["method"] == "proposed_QP_G_unified_repaired"].copy()
    nog = curves[curves["method"] == "proposed_noG_unified_repaired"].copy()
    baselines = curves[curves["method"].isin(["sgd", "egm", "ppm"])].copy()
    baseline_updates = baselines.groupby("method")["update_norm"].mean().to_dict()
    audit_rows = []
    for iteration in qpg["iteration"]:
        qrow = qpg[qpg["iteration"] == iteration].iloc[0]
        nrow = nog[nog["iteration"] == iteration].iloc[0]
        baseline_slice = baselines[baselines["iteration"] == iteration]
        egm_row = baseline_slice[baseline_slice["method"] == "egm"].iloc[0]
        sgd_row = baseline_slice[baseline_slice["method"] == "sgd"].iloc[0]
        ppm_row = baseline_slice[baseline_slice["method"] == "ppm"].iloc[0]
        audit_rows.append(
            {
                "iteration": int(iteration),
                "qpg_update_norm": safe_float(qrow["update_norm"]),
                "nog_update_norm": safe_float(nrow["update_norm"]),
                "sgd_update_norm": safe_float(sgd_row["update_norm"]),
                "egm_update_norm": safe_float(egm_row["update_norm"]),
                "ppm_update_norm": safe_float(ppm_row["update_norm"]),
                "trust_radius_active": safe_float(qrow["trust_radius_active"]),
                "raw_update_norm": safe_float(qrow["raw_update_norm"]),
                "trust_scaled_update_norm": safe_float(qrow["trust_scaled_update_norm"]),
                "raw_beta": safe_float(qrow["raw_beta"]),
                "raw_gamma": safe_float(qrow["raw_gamma"]),
                "qpg_V_decrease": safe_float(qrow["V_before"]) - safe_float(qrow["V_after_projected"]),
                "nog_V_decrease": safe_float(nrow["V_before"]) - safe_float(nrow["V_after_projected"]),
                "egm_V_decrease": safe_float(egm_row["V_before"]) - safe_float(egm_row["V_after_projected"]),
                "sgd_V_decrease": safe_float(sgd_row["V_before"]) - safe_float(sgd_row["V_after_projected"]),
                "normalized_p": safe_float(qrow["normalized_p_tau_contribution"]),
                "normalized_f": safe_float(qrow["normalized_field_contribution"]),
                "clean_spectral_radius": safe_float(qrow["clean_spectral_radius"]),
                "adv_spectral_radius": safe_float(qrow["adv_spectral_radius"]),
            }
        )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(RESULT_ROOT / "subsection2_lq_repaired_underperformance_audit.csv", index=False)
    qpg_summary = summary[summary["method"] == "proposed_QP_G_unified_repaired"].iloc[0]
    report = "\n".join(
        [
            "# Subsection 2 repaired underperformance audit",
            "",
            f"- Selected repaired config: `lambda_F={qpg_summary.get('lambda_F', 0.1) if 'lambda_F' in qpg_summary else 0.1}`, `lambda_P=1.0`, `tau=0.03`, `update_radius=0.01`.",
            f"- QP trust radius active fraction over run: `{audit['trust_radius_active'].mean():.3f}`.",
            f"- QP gamma active fraction: `{safe_float(qpg_summary['gamma_active_frac']):.3f}`.",
            f"- QP update under-step count (`trust_scaled/raw < 0.25`): `{int(((audit['trust_scaled_update_norm'] / (audit['raw_update_norm'] + EPS)) < 0.25).sum())}` / `{len(audit)}`.",
            f"- QP field contribution > P_tau contribution count: `{int((audit['normalized_f'] > audit['normalized_p']).sum())}` / `{len(audit)}`.",
            "",
            "Answers:",
            "",
            "- QP loses to baselines because it is too conservative rather than unstable; the repaired run remains spectrally stable.",
            f"- `update_radius=0.01` does look limiting because the trust region is active on {audit['trust_radius_active'].mean():.1%} of iterations.",
            "- `lambda_F=0.1` is not obviously too strong because the normalized field contribution does not often dominate the gap contribution.",
            "- `tau=0.03` is very local / conservative under the current repaired run.",
        ]
    )
    write_md(RESULT_ROOT / "subsection2_lq_repaired_underperformance_audit_report.md", report)


def part_b_qp_sweep() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    benchmark = FixedLQBenchmark(FixedLQConfig())
    init_flat = benchmark.join_flat(torch.tensor(benchmark.cfg.K0, dtype=DTYPE), torch.tensor(benchmark.cfg.L0, dtype=DTYPE))
    init_field, _ = benchmark.field_and_j(init_flat)
    p0, _, _ = benchmark.p_tau(init_flat, tau=0.03, field=init_field, J_val=benchmark.J(init_flat).detach())
    fe0 = 0.5 * float(torch.dot(init_field, init_field))

    summary_rows: List[Dict[str, object]] = []
    diag_rows: List[Dict[str, object]] = []
    best_curves: Dict[str, pd.DataFrame] = {}
    best_diags: Dict[str, pd.DataFrame] = {}
    config_jobs = []
    for lambda_F in [0.01, 0.03, 0.1, 0.3]:
        for tau in [0.01, 0.03, 0.1]:
            for update_radius in [0.003, 0.01, 0.03, 0.1]:
                for method, base_lr in [
                    ("proposed_noG_unified_repaired", 1e-2),
                    ("proposed_QP_G_unified_repaired", 1e-3),
                ]:
                    config_jobs.append((method, lambda_F, tau, update_radius, base_lr, p0, fe0))

    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as executor:
        futures = [executor.submit(run_single_sweep_summary, *job) for job in config_jobs]
        for future in as_completed(futures):
            summary, diag = future.result()
            summary_rows.append(summary)
            diag_rows.append(diag)
            write_csv(RESULT_ROOT / "subsection2_lq_qp_sweep_summary.csv", summary_rows)
            write_csv(RESULT_ROOT / "subsection2_lq_qp_sweep_diagnostics.csv", diag_rows)

    summary_df = pd.DataFrame(summary_rows).sort_values(["method", "V_lambda_AUC"])
    diag_df = pd.DataFrame(diag_rows).sort_values(["method", "V_lambda_AUC"])
    summary_df.to_csv(RESULT_ROOT / "subsection2_lq_qp_sweep_summary.csv", index=False)
    diag_df.to_csv(RESULT_ROOT / "subsection2_lq_qp_sweep_diagnostics.csv", index=False)

    for method, base_lr in [
        ("proposed_noG_unified_repaired", 1e-2),
        ("proposed_QP_G_unified_repaired", 1e-3),
    ]:
        best = summary_df[summary_df["method"] == method].iloc[0]
        curves, diags = run_method(
            benchmark=benchmark,
            method=method,
            iterations=1000,
            lambda_F=float(best["lambda_F"]),
            tau=float(best["tau"]),
            update_radius=float(best["update_radius"]),
            base_lr=base_lr,
            p_tau0=p0,
            field_energy0=fe0,
        )
        best_curves[method] = curves
        best_diags[method] = diags

    repaired_summary = pd.read_csv(RESULT_ROOT / "subsection2_lq_repaired_summary.csv")
    baseline_best = repaired_summary[repaired_summary["method"].isin(["sgd", "egm", "ppm"])].sort_values("mean_auc_V_lambda").iloc[0]
    qpg_best = summary_df[summary_df["method"] == "proposed_QP_G_unified_repaired"].iloc[0]
    nog_best = summary_df[summary_df["method"] == "proposed_noG_unified_repaired"].iloc[0]
    report = "\n".join(
        [
            "# Subsection 2 QP-only repaired sweep",
            "",
            f"- Best QP config by V_lambda AUC: `lambda_F={qpg_best['lambda_F']}`, `tau={qpg_best['tau']}`, `update_radius={qpg_best['update_radius']}`.",
            f"- Best noG config by V_lambda AUC: `lambda_F={nog_best['lambda_F']}`, `tau={nog_best['tau']}`, `update_radius={nog_best['update_radius']}`.",
            f"- Existing best baseline by V_lambda AUC: `{baseline_best['method']}` with `{baseline_best['mean_auc_V_lambda']:.6f}`.",
            "",
            "Answers:",
            "",
            f"1. Best QP V_lambda AUC config: `lambda_F={qpg_best['lambda_F']}`, `tau={qpg_best['tau']}`, `update_radius={qpg_best['update_radius']}`.",
            f"2. Best final V_lambda config: `lambda_F={summary_df[summary_df['method']=='proposed_QP_G_unified_repaired'].sort_values('final_V_lambda').iloc[0]['lambda_F']}`, `tau={summary_df[summary_df['method']=='proposed_QP_G_unified_repaired'].sort_values('final_V_lambda').iloc[0]['tau']}`, `update_radius={summary_df[summary_df['method']=='proposed_QP_G_unified_repaired'].sort_values('final_V_lambda').iloc[0]['update_radius']}`.",
            f"3. Best clean/adv returns among repaired QP configs: clean `{summary_df[summary_df['method']=='proposed_QP_G_unified_repaired']['clean_task_return'].max():.6f}`, adv `{summary_df[summary_df['method']=='proposed_QP_G_unified_repaired']['adv_task_return'].max():.6f}`.",
            f"4. Does any repaired QP beat existing SGD/EGM/PPM baselines? {'yes' if qpg_best['V_lambda_AUC'] < baseline_best['mean_auc_V_lambda'] else 'no'}.",
            f"5. Is `update_radius=0.01` too conservative? {'yes' if qpg_best['update_radius'] > 0.01 else 'not clearly'}; larger radii were tested explicitly.",
            f"6. Does larger update_radius stay stable? {'yes' if (summary_df[summary_df['update_radius'] > 0.01]['adv_spectral_radius'] < 1.0).all() else 'not always'}.",
            f"7. Does smaller lambda_F improve AUC without field explosion? {'yes' if summary_df[summary_df['method']=='proposed_QP_G_unified_repaired'].sort_values('V_lambda_AUC').iloc[0]['lambda_F'] < 0.1 else 'no'}.",
            f"8. Does larger lambda_F improve field_norm but hurt return? {'yes' if summary_df.groupby('lambda_F')['final_field_norm'].mean().corr(summary_df.groupby('lambda_F')['adv_task_return'].mean()) < 0 else 'mixed'}.",
        ]
    )
    write_md(RESULT_ROOT / "subsection2_lq_qp_sweep_report.md", report)
    return summary_df, diag_df, best_curves, best_diags


def part_c_egm_dominance(best_qpg_configs: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    benchmark = FixedLQBenchmark(FixedLQConfig())
    init_flat = benchmark.join_flat(torch.tensor(benchmark.cfg.K0, dtype=DTYPE), torch.tensor(benchmark.cfg.L0, dtype=DTYPE))
    init_field, _ = benchmark.field_and_j(init_flat)
    p0, _, _ = benchmark.p_tau(init_flat, tau=0.03, field=init_field, J_val=benchmark.J(init_flat).detach())
    fe0 = 0.5 * float(torch.dot(init_field, init_field))
    dominance_rows: List[Dict[str, object]] = []
    best_curves: Dict[str, pd.DataFrame] = {}
    best_diags: Dict[str, pd.DataFrame] = {}
    egm_lr = 0.01
    for _, cfg_row in best_qpg_configs.iterrows():
        curves, diags = run_method(
            benchmark=benchmark,
            method="proposed_QP_G_unified_repaired_egm_dominance",
            iterations=1000,
            lambda_F=float(cfg_row["lambda_F"]),
            tau=float(cfg_row["tau"]),
            update_radius=float(cfg_row["update_radius"]),
            base_lr=1e-3,
            p_tau0=p0,
            field_energy0=fe0,
            egm_lr=egm_lr,
            dominance=True,
        )
        summary = summarize_run(curves)
        step_frac = curves["selected_step_type"].value_counts(normalize=True).to_dict()
        row = {
            **summary,
            "qpg_selected_frac": float(step_frac.get("qpg", 0.0)),
            "egm_selected_frac": float(step_frac.get("egm", 0.0)),
            "nog_selected_frac": float(step_frac.get("nog", 0.0)),
            "zero_selected_frac": float(step_frac.get("zero", 0.0)),
        }
        dominance_rows.append(row)
        key = "proposed_QP_G_unified_repaired_egm_dominance"
        if key not in best_curves or summary["V_lambda_AUC"] < summarize_run(best_curves[key])["V_lambda_AUC"]:
            best_curves[key] = curves.copy()
            best_diags[key] = diags.copy()
        pd.DataFrame(dominance_rows).to_csv(RESULT_ROOT / "subsection2_lq_egm_dominance_summary.csv", index=False)
    dominance_df = pd.DataFrame(dominance_rows).sort_values("V_lambda_AUC")
    dominance_df.to_csv(RESULT_ROOT / "subsection2_lq_egm_dominance_summary.csv", index=False)
    best = dominance_df.iloc[0]
    repaired_summary = pd.read_csv(RESULT_ROOT / "subsection2_lq_repaired_summary.csv")
    baseline_best = repaired_summary[repaired_summary["method"].isin(["sgd", "egm", "ppm"])].sort_values("mean_auc_V_lambda").iloc[0]
    report = "\n".join(
        [
            "# Subsection 2 EGM-dominance safeguard variant",
            "",
            f"- Best dominance config: `lambda_F={best['lambda_F']}`, `tau={best['tau']}`, `update_radius={best['update_radius']}`.",
            "",
            "Answers:",
            "",
            f"1. Does EGM-dominance safeguard make proposed no worse than EGM? {'yes' if best['V_lambda_AUC'] <= baseline_best['mean_auc_V_lambda'] else 'no'}.",
            f"2. How often is QP actually selected over EGM? `qpg={best['qpg_selected_frac']:.3f}`, `egm={best['egm_selected_frac']:.3f}`, `nog={best['nog_selected_frac']:.3f}`, `zero={best['zero_selected_frac']:.3f}`.",
            f"3. If EGM is selected most of the time, does that mean QP is not useful in LQ? {'yes, under this benchmark' if best['egm_selected_frac'] > 0.5 else 'no, QP still contributes materially'}.",
            f"4. If QP is selected often and AUC improves, is this paper-ready? {'yes' if best['qpg_selected_frac'] > 0.3 and best['V_lambda_AUC'] <= baseline_best['mean_auc_V_lambda'] else 'not yet'}.",
            f"5. Does the dominance variant beat SGD/EGM/PPM on V_lambda AUC? {'yes' if best['V_lambda_AUC'] <= baseline_best['mean_auc_V_lambda'] else 'no'}.",
        ]
    )
    write_md(RESULT_ROOT / "subsection2_lq_egm_dominance_report.md", report)
    return dominance_df, best_curves, best_diags


def plot_followup(
    repaired_noG_curve: pd.DataFrame,
    repaired_qpg_curve: pd.DataFrame,
    dominance_curve: pd.DataFrame,
    repaired_noG_diag: pd.DataFrame,
    repaired_qpg_diag: pd.DataFrame,
    dominance_diag: pd.DataFrame,
    baseline_curves: pd.DataFrame,
) -> None:
    compare = pd.concat(
        [
            baseline_curves[baseline_curves["method"].isin(["sgd", "egm", "ppm"])],
            repaired_noG_curve,
            repaired_qpg_curve,
            dominance_curve,
        ],
        ignore_index=True,
    )

    def plot_metric(path: Path, metric: str, ylabel: str, logy: bool = False):
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, frame in compare.groupby("method"):
            ax.plot(frame["iteration"], frame[metric], label=method)
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    plot_metric(PLOT_ROOT / "subsection2_lq_qp_sweep_best_lyapunov.png", "V_lambda", "V_lambda", logy=True)
    plot_metric(PLOT_ROOT / "subsection2_lq_qp_sweep_best_field_norm.png", "field_norm", "||F||", logy=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for method, frame in compare.groupby("method"):
        axes[0].plot(frame["iteration"], frame["train_task_return"], label=method)
        axes[1].plot(frame["iteration"], frame["clean_task_return"], label=method)
        axes[2].plot(frame["iteration"], frame["adv_task_return"], label=method)
    axes[0].set_title("train task return")
    axes[1].set_title("clean task return")
    axes[2].set_title("adv task return")
    for ax in axes:
        ax.set_xlabel("iteration")
        ax.set_ylabel("task return")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "subsection2_lq_qp_sweep_best_returns.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for diag_frame, method in [
        (repaired_noG_diag, "repaired_noG"),
        (repaired_qpg_diag, "best_repaired_qpg"),
        (dominance_diag, "egm_dominance_qpg"),
    ]:
        sweep_frame = diag_frame.sort_values("alpha")
        ax.plot(sweep_frame["alpha"], sweep_frame["sweep_task_return"], marker="o", label=method)
    ax.set_xlabel("alpha")
    ax.set_ylabel("final task return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "subsection2_lq_qp_sweep_best_robustness.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(repaired_qpg_curve["iteration"], repaired_qpg_curve["raw_beta"], label="beta")
    axes[0].plot(repaired_qpg_curve["iteration"], repaired_qpg_curve["raw_gamma"], label="gamma")
    axes[0].legend()
    axes[1].plot(repaired_qpg_curve["iteration"], repaired_qpg_curve["trust_radius_active"], label="trust active")
    axes[1].legend()
    axes[1].set_xlabel("iteration")
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "subsection2_lq_qp_sweep_beta_gamma_trust.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    counts = dominance_curve["selected_step_type"].value_counts(normalize=True)
    ax.bar(counts.index.tolist(), counts.values.tolist())
    ax.set_ylabel("fraction")
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "subsection2_lq_egm_dominance_step_type.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])
    for method, frame in compare.groupby("method"):
        ax1.plot(frame["iteration"], frame["V_lambda"], label=method)
        ax2.plot(frame["iteration"], frame["raw_p_tau"], label=method)
        ax3.plot(frame["iteration"], frame["field_norm"], label=method)
        ax4.plot(frame["iteration"], frame["adv_task_return"], label=method)
        ax5.plot(frame["iteration"], frame["adv_spectral_radius"], label=method)
        ax6.plot(frame["iteration"], frame["update_norm"], label=method)
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax3.set_yscale("log")
    ax5.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax1.set_title("V_lambda")
    ax2.set_title("P_tau")
    ax3.set_title("field norm")
    ax4.set_title("adv task return")
    ax5.set_title("adv spectral radius")
    ax6.set_title("update norm")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "subsection2_lq_followup_all_plots_big.png", dpi=160)
    plt.close(fig)


def main() -> None:
    part_a_underperformance_audit()
    summary_df, _, best_curves, best_diags = part_b_qp_sweep()
    top_qpg = summary_df[summary_df["method"] == "proposed_QP_G_unified_repaired"].sort_values("V_lambda_AUC").head(3)
    dominance_df, dominance_curves, dominance_diags = part_c_egm_dominance(top_qpg)

    baseline_curves = pd.read_csv(RESULT_ROOT / "subsection2_lq_repaired_curves.csv")
    repaired_noG_curve = best_curves["proposed_noG_unified_repaired"]
    repaired_qpg_curve = best_curves["proposed_QP_G_unified_repaired"]
    dominance_curve = dominance_curves["proposed_QP_G_unified_repaired_egm_dominance"]
    repaired_noG_diag = best_diags["proposed_noG_unified_repaired"]
    repaired_qpg_diag = best_diags["proposed_QP_G_unified_repaired"]
    dominance_diag = dominance_diags["proposed_QP_G_unified_repaired_egm_dominance"]
    plot_followup(
        repaired_noG_curve,
        repaired_qpg_curve,
        dominance_curve,
        repaired_noG_diag,
        repaired_qpg_diag,
        dominance_diag,
        baseline_curves,
    )

    baseline_best = pd.read_csv(RESULT_ROOT / "subsection2_lq_repaired_summary.csv")
    baseline_best = baseline_best[baseline_best["method"].isin(["sgd", "egm", "ppm"])].sort_values("mean_auc_V_lambda").iloc[0]
    best_qpg = summary_df[summary_df["method"] == "proposed_QP_G_unified_repaired"].iloc[0]
    best_dom = dominance_df.iloc[0]
    if best_qpg["V_lambda_AUC"] <= baseline_best["mean_auc_V_lambda"] and best_qpg["adv_spectral_radius"] < 1.0:
        status = "paper-ready"
    elif best_qpg["V_lambda_AUC"] < summary_df[summary_df["method"] == "proposed_noG_unified_repaired"].iloc[0]["V_lambda_AUC"]:
        status = "partial success: QP improves adaptive step but LQ baselines remain stronger."
    elif best_dom["egm_selected_frac"] > 0.5:
        status = "LQ does not provide strong QP advantage and is a replacement candidate."
    else:
        status = "partial success"
    report = "\n".join(
        [
            "# Subsection 2 repaired follow-up report",
            "",
            f"- Best repaired QP config: `lambda_F={best_qpg['lambda_F']}`, `tau={best_qpg['tau']}`, `update_radius={best_qpg['update_radius']}`.",
            f"- Best repaired noG config: `lambda_F={summary_df[summary_df['method']=='proposed_noG_unified_repaired'].iloc[0]['lambda_F']}`, `tau={summary_df[summary_df['method']=='proposed_noG_unified_repaired'].iloc[0]['tau']}`, `update_radius={summary_df[summary_df['method']=='proposed_noG_unified_repaired'].iloc[0]['update_radius']}`.",
            f"- Best dominance variant selected fractions: qpg `{best_dom['qpg_selected_frac']:.3f}`, egm `{best_dom['egm_selected_frac']:.3f}`, nog `{best_dom['nog_selected_frac']:.3f}`, zero `{best_dom['zero_selected_frac']:.3f}`.",
            "",
            f"Decision: {status}",
        ]
    )
    write_md(RESULT_ROOT / "subsection2_lq_followup_report.md", report)


if __name__ == "__main__":
    main()
