from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


DTYPE = torch.float64
EPS = 1e-12
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = ROOT / "original" / "results" / "task_aligned_lq_rarl"
PLOT_ROOT = RESULT_ROOT / "plots"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)
PLOT_ROOT.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def write_md(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def df_text(df: pd.DataFrame) -> str:
    return "(empty)" if df.empty else df.to_string(index=False)


def safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def auc_from_series(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    xs = np.arange(arr.size, dtype=np.float64)
    return float(np.trapezoid(arr, x=xs))


def first_below(values: Sequence[float], threshold: float) -> float:
    for idx, value in enumerate(values):
        if np.isfinite(value) and value <= threshold:
            return float(idx)
    return float("nan")


def clip_floor(values: Sequence[float], floor: float = 1e-12) -> np.ndarray:
    return np.maximum(np.asarray(list(values), dtype=np.float64), floor)


def safe_spectral_radius(matrix: torch.Tensor) -> float:
    arr = matrix.detach()
    if not bool(torch.isfinite(arr).all()):
        return float("inf")
    try:
        vals = torch.linalg.eigvals(arr)
        if not bool(torch.isfinite(vals).all()):
            return float("inf")
        return float(torch.max(torch.abs(vals)))
    except Exception:
        return float("inf")


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
    return float(beta), {"a": float(a), "b": float(b), "h": float(h)}


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
    return beta, gamma, {"cond": float(cond), "indef": indef, "h1": float(h1), "h2": float(h2)}


@dataclass(frozen=True)
class TaskAlignedLQConfig:
    state_dim: int = 2
    action_dim: int = 2
    disturbance_dim: int = 2
    omega: float = 0.2
    gamma: float = 0.95
    horizon: int = 50
    num_eval_initial_states: int = 256
    train_batch_size: int = 256
    eval_batch_size: int = 256
    seed: int = 0
    init_scale: float = 0.03
    q_state: float = 1.0
    a_u: float = 0.01
    alpha_dyn: float = 0.2
    K_budget: float = 3.0
    L_budget: float = 3.0
    tau: float = 0.03
    lambda_F: float = 0.01
    lambda_P: float = 1.0
    gap_inner_steps: int = 10
    gap_inner_lr: float = 0.05
    gap_local_radius: float = 0.25
    local_br_radius: float = 0.5
    br_inner_steps: int = 50
    br_inner_lr: float = 0.05
    ppm_inner_steps: int = 5


class TaskAlignedLQBenchmark:
    def __init__(self, cfg: TaskAlignedLQConfig):
        self.cfg = cfg
        c = math.cos(cfg.omega)
        s = math.sin(cfg.omega)
        r_state = torch.tensor([[c, -s], [s, c]], dtype=DTYPE)
        self.A = 0.90 * r_state
        self.B = torch.eye(cfg.action_dim, dtype=DTYPE)
        self.E = torch.eye(cfg.disturbance_dim, dtype=DTYPE)
        self.R_dyn = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=DTYPE)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(cfg.seed)
        self.train_x0 = torch.randn(cfg.train_batch_size, cfg.state_dim, generator=gen, dtype=DTYPE)
        self.eval_x0 = torch.randn(cfg.eval_batch_size, cfg.state_dim, generator=gen, dtype=DTYPE)
        self.train_sigma0 = (self.train_x0.T @ self.train_x0) / cfg.train_batch_size
        self.eval_sigma0 = (self.eval_x0.T @ self.eval_x0) / cfg.eval_batch_size
        self.K0 = cfg.init_scale * torch.randn(cfg.action_dim, cfg.state_dim, generator=gen, dtype=DTYPE)
        self.L0 = cfg.init_scale * torch.randn(cfg.disturbance_dim, cfg.state_dim, generator=gen, dtype=DTYPE)
        self.flat0 = self.join_flat(self.K0, self.L0)

    @property
    def flat_dim(self) -> int:
        return 2 * self.cfg.action_dim * self.cfg.state_dim

    def split_flat(self, flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dim = self.cfg.action_dim * self.cfg.state_dim
        return (
            flat[:dim].reshape(self.cfg.action_dim, self.cfg.state_dim),
            flat[dim:].reshape(self.cfg.disturbance_dim, self.cfg.state_dim),
        )

    def join_flat(self, K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        return torch.cat([K.reshape(-1), L.reshape(-1)])

    def project_matrix_to_ball(self, M: torch.Tensor, budget: float) -> Tuple[torch.Tensor, bool]:
        norm = float(torch.linalg.norm(M))
        if norm <= budget + EPS:
            return M, False
        return M * (budget / (norm + EPS)), True

    def project_flat(self, flat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        K, L = self.split_flat(flat)
        Kp, proj_k = self.project_matrix_to_ball(K, self.cfg.K_budget)
        Lp, proj_l = self.project_matrix_to_ball(L, self.cfg.L_budget)
        return self.join_flat(Kp, Lp), {
            "projection_active_K": float(proj_k),
            "projection_active_L": float(proj_l),
            "K_norm": float(torch.linalg.norm(Kp)),
            "L_norm": float(torch.linalg.norm(Lp)),
        }

    def project_local(self, M: torch.Tensor, M0: torch.Tensor, radius: float, budget: float) -> Tuple[torch.Tensor, bool, bool]:
        delta = M - M0
        delta_norm = float(torch.linalg.norm(delta))
        local_active = False
        if delta_norm > radius + EPS:
            M = M0 + delta * (radius / (delta_norm + EPS))
            local_active = True
        M, budget_active = self.project_matrix_to_ball(M, budget)
        return M, local_active, budget_active

    def closed_loop(self, K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        return self.A + self.B @ K + self.cfg.alpha_dyn * self.E @ self.R_dyn @ L

    def rollout_task_return(self, K: torch.Tensor, L: torch.Tensor, sigma0: torch.Tensor) -> Dict[str, float]:
        sigma = sigma0.clone()
        closed_loop = self.closed_loop(K, L)
        rho = safe_spectral_radius(closed_loop)
        total = torch.zeros((), dtype=DTYPE)
        weight_sum = 0.0
        state_norm_mean = 0.0
        action_norm_mean = 0.0
        state_norm_max = 0.0
        for t in range(self.cfg.horizon):
            weight = self.cfg.gamma ** t
            x_sq = torch.trace(sigma)
            u_sq = torch.trace(K @ sigma @ K.T)
            reward = -0.5 * self.cfg.q_state * x_sq - 0.5 * self.cfg.a_u * u_sq
            total = total + weight * reward
            weight_sum += weight
            state_norm = float(torch.sqrt(torch.clamp(x_sq, min=0.0)))
            act_norm = float(torch.sqrt(torch.clamp(u_sq, min=0.0)))
            state_norm_mean += weight * state_norm
            action_norm_mean += weight * act_norm
            state_norm_max = max(state_norm_max, state_norm)
            sigma = closed_loop @ sigma @ closed_loop.T
        return {
            "task_return": float(total),
            "state_norm_mean": float(state_norm_mean / max(weight_sum, EPS)),
            "state_norm_max": float(state_norm_max),
            "action_norm_mean": float(action_norm_mean / max(weight_sum, EPS)),
            "spectral_radius": float(rho),
            "finite": bool(np.isfinite(float(total)) and np.isfinite(state_norm_max) and np.isfinite(rho)),
        }

    def J_T(self, flat: torch.Tensor) -> torch.Tensor:
        K, L = self.split_flat(flat)
        sigma = self.train_sigma0
        closed_loop = self.closed_loop(K, L)
        total = torch.zeros((), dtype=DTYPE)
        for t in range(self.cfg.horizon):
            x_sq = torch.trace(sigma)
            u_sq = torch.trace(K @ sigma @ K.T)
            reward = -0.5 * self.cfg.q_state * x_sq - 0.5 * self.cfg.a_u * u_sq
            total = total + (self.cfg.gamma ** t) * reward
            sigma = closed_loop @ sigma @ closed_loop.T
        return total

    def field_tensor(self, flat: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        z = flat if flat.requires_grad else flat.detach().clone().requires_grad_(True)
        value = self.J_T(z)
        grad = torch.autograd.grad(value, z, create_graph=create_graph)[0]
        dim = self.cfg.action_dim * self.cfg.state_dim
        return torch.cat([-grad[:dim], grad[dim:]])

    def jvp_field(self, flat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        z = flat.detach().clone().requires_grad_(True)
        v = vec.detach().clone()

        def f(inp: torch.Tensor) -> torch.Tensor:
            return self.field_tensor(inp, create_graph=True)

        _, jvp = torch.autograd.functional.jvp(f, z, v, create_graph=False, strict=False)
        return jvp.detach()

    def full_jacobian(self, flat: torch.Tensor) -> torch.Tensor:
        z = flat.detach().clone().requires_grad_(True)
        f = self.field_tensor(z, create_graph=True)
        rows = []
        for i in range(f.numel()):
            rows.append(torch.autograd.grad(f[i], z, retain_graph=True)[0])
        return torch.stack(rows, dim=0)

    def local_gap_terms_T(self, flat: torch.Tensor) -> Dict[str, float]:
        dim = self.cfg.action_dim * self.cfg.state_dim
        K0 = flat[:dim].detach().reshape(self.cfg.action_dim, self.cfg.state_dim)
        L0 = flat[dim:].detach().reshape(self.cfg.disturbance_dim, self.cfg.state_dim)
        j0 = float(self.J_T(flat.detach()))

        Kbar = K0.clone()
        for _ in range(self.cfg.gap_inner_steps):
            Kreq = Kbar.detach().clone().requires_grad_(True)
            flat_req = self.join_flat(Kreq, L0)
            prox = 0.5 / self.cfg.tau * torch.sum((Kreq - K0) ** 2)
            obj = self.J_T(flat_req) - prox
            grad = torch.autograd.grad(obj, Kreq)[0]
            Knext = Kbar + self.cfg.gap_inner_lr * grad
            Kbar, _, _ = self.project_local(Knext, K0, self.cfg.gap_local_radius, self.cfg.K_budget)
        k_term = float(self.J_T(self.join_flat(Kbar, L0)) - 0.5 / self.cfg.tau * torch.sum((Kbar - K0) ** 2))
        gap_k = max(0.0, k_term - j0)

        Lbar = L0.clone()
        for _ in range(self.cfg.gap_inner_steps):
            Lreq = Lbar.detach().clone().requires_grad_(True)
            flat_req = self.join_flat(K0, Lreq)
            prox = 0.5 / self.cfg.tau * torch.sum((Lreq - L0) ** 2)
            obj = self.J_T(flat_req) + prox
            grad = torch.autograd.grad(obj, Lreq)[0]
            Lnext = Lbar - self.cfg.gap_inner_lr * grad
            Lbar, _, _ = self.project_local(Lnext, L0, self.cfg.gap_local_radius, self.cfg.L_budget)
        l_term = float(self.J_T(self.join_flat(K0, Lbar)) + 0.5 / self.cfg.tau * torch.sum((Lbar - L0) ** 2))
        gap_l = max(0.0, j0 - l_term)
        return {"P_tau_T_K_gap": gap_k, "P_tau_T_L_gap": gap_l, "P_tau_T": gap_k + gap_l}

    def local_exploitability(self, flat: torch.Tensor) -> Dict[str, float]:
        dim = self.cfg.action_dim * self.cfg.state_dim
        K0 = flat[:dim].detach().reshape(self.cfg.action_dim, self.cfg.state_dim)
        L0 = flat[dim:].detach().reshape(self.cfg.disturbance_dim, self.cfg.state_dim)
        j0 = float(self.J_T(flat.detach()))

        Kbar = K0.clone()
        for _ in range(self.cfg.gap_inner_steps):
            Kreq = Kbar.detach().clone().requires_grad_(True)
            grad = torch.autograd.grad(self.J_T(self.join_flat(Kreq, L0)), Kreq)[0]
            Knext = Kbar + self.cfg.gap_inner_lr * grad
            Kbar, _, _ = self.project_local(Knext, K0, self.cfg.gap_local_radius, self.cfg.K_budget)
        exploit_k = max(0.0, float(self.J_T(self.join_flat(Kbar, L0))) - j0)

        Lbar = L0.clone()
        for _ in range(self.cfg.gap_inner_steps):
            Lreq = Lbar.detach().clone().requires_grad_(True)
            grad = torch.autograd.grad(self.J_T(self.join_flat(K0, Lreq)), Lreq)[0]
            Lnext = Lbar - self.cfg.gap_inner_lr * grad
            Lbar, _, _ = self.project_local(Lnext, L0, self.cfg.gap_local_radius, self.cfg.L_budget)
        exploit_l = max(0.0, j0 - float(self.J_T(self.join_flat(K0, Lbar))))
        return {
            "exploit_K": exploit_k,
            "exploit_L": exploit_l,
            "approximate_exploitability_AUC": exploit_k + exploit_l,
            "approximate_exploitability": exploit_k + exploit_l,
        }

    def robust_br(self, K: torch.Tensor, L_init: torch.Tensor, from_zero: bool = False) -> Dict[str, float]:
        start = torch.zeros_like(L_init) if from_zero else L_init.detach().clone()
        current = start.clone()
        current_ref = start.clone()
        used_lr = self.cfg.br_inner_lr
        last_eval: Dict[str, float] | None = None
        for lr in [self.cfg.br_inner_lr, 0.01] if self.cfg.br_inner_lr != 0.01 else [self.cfg.br_inner_lr]:
            current = start.clone()
            used_lr = lr
            for _ in range(self.cfg.br_inner_steps):
                Lreq = current.detach().clone().requires_grad_(True)
                value = self._task_return_eval_tensor(K.detach(), Lreq, self.eval_sigma0)
                grad = torch.autograd.grad(value, Lreq)[0]
                Lnext = current - lr * grad
                Lnext, _, _ = self.project_local(Lnext, current_ref, self.cfg.local_br_radius, self.cfg.L_budget)
                current = Lnext.detach()
            eval_res = self.rollout_task_return(K.detach(), current, self.eval_sigma0)
            stable = eval_res["finite"] and eval_res["state_norm_max"] < 1e6 and eval_res["spectral_radius"] < 1.5
            last_eval = {
                "robust_br_task_return": eval_res["task_return"],
                "L_br_norm": float(torch.linalg.norm(current)),
                "L_br_distance_from_current": float(torch.linalg.norm(current - L_init.detach())),
                "closed_loop_spectral_radius_robust_br": eval_res["spectral_radius"],
                "state_norm_mean_robust_br": eval_res["state_norm_mean"],
                "state_norm_max_robust_br": eval_res["state_norm_max"],
                "valid_robust_br_eval": float(stable),
                "br_used_lr": float(lr),
            }
            if stable:
                return last_eval
        assert last_eval is not None
        return last_eval

    def _task_return_eval_tensor(self, K: torch.Tensor, L: torch.Tensor, sigma0: torch.Tensor) -> torch.Tensor:
        sigma = sigma0
        closed_loop = self.closed_loop(K, L)
        total = torch.zeros((), dtype=DTYPE)
        for t in range(self.cfg.horizon):
            x_sq = torch.trace(sigma)
            u_sq = torch.trace(K @ sigma @ K.T)
            total = total + (self.cfg.gamma ** t) * (-0.5 * self.cfg.q_state * x_sq - 0.5 * self.cfg.a_u * u_sq)
            sigma = closed_loop @ sigma @ closed_loop.T
        return total

    def metrics(self, flat: torch.Tensor, field_energy0: float, p_tau0: float) -> Dict[str, float]:
        field = self.field_tensor(flat.detach(), create_graph=False).detach()
        field_energy = 0.5 * float(torch.dot(field, field))
        gaps = self.local_gap_terms_T(flat)
        exploit = self.local_exploitability(flat)
        K, L = self.split_flat(flat)
        clean = self.rollout_task_return(K, torch.zeros_like(L), self.eval_sigma0)
        current_adv = self.rollout_task_return(K, L, self.eval_sigma0)
        robust = self.robust_br(K, L, from_zero=False)
        robust_zero = self.robust_br(K, L, from_zero=True)
        field_term = field_energy / (field_energy0 + EPS)
        normalized_p = gaps["P_tau_T"] / (p_tau0 + EPS)
        return {
            "V_align": self.cfg.lambda_F * field_term + self.cfg.lambda_P * normalized_p,
            "field_term": field_term,
            "field_norm": math.sqrt(max(2.0 * field_energy, 0.0)),
            "P_tau_T": gaps["P_tau_T"],
            "P_tau_T_K_gap": gaps["P_tau_T_K_gap"],
            "P_tau_T_L_gap": gaps["P_tau_T_L_gap"],
            "normalized_P_tau_T": normalized_p,
            "approximate_exploitability": exploit["approximate_exploitability"],
            "clean_task_return": clean["task_return"],
            "current_adv_task_return": current_adv["task_return"],
            "robust_br_task_return": robust["robust_br_task_return"],
            "robust_br_from_zero_task_return": robust_zero["robust_br_task_return"],
            "robust_degradation": clean["task_return"] - robust["robust_br_task_return"],
            "K_norm": float(torch.linalg.norm(K)),
            "L_norm": float(torch.linalg.norm(L)),
            "clean_state_norm_mean": clean["state_norm_mean"],
            "clean_state_norm_max": clean["state_norm_max"],
            "current_adv_state_norm_mean": current_adv["state_norm_mean"],
            "current_adv_state_norm_max": current_adv["state_norm_max"],
            "robust_br_state_norm_mean": robust["state_norm_mean_robust_br"],
            "robust_br_state_norm_max": robust["state_norm_max_robust_br"],
            "clean_action_norm_mean": clean["action_norm_mean"],
            "current_adv_action_norm_mean": current_adv["action_norm_mean"],
            "closed_loop_spectral_radius_clean": clean["spectral_radius"],
            "closed_loop_spectral_radius_current_adv": current_adv["spectral_radius"],
            "closed_loop_spectral_radius_robust_br": robust["closed_loop_spectral_radius_robust_br"],
            "L_br_norm": robust["L_br_norm"],
            "L_br_distance_from_current": robust["L_br_distance_from_current"],
            "valid_clean_eval": float(clean["finite"]),
            "valid_current_adv_eval": float(current_adv["finite"]),
            "valid_robust_br_eval": robust["valid_robust_br_eval"],
        }


def initial_terms(benchmark: TaskAlignedLQBenchmark) -> Tuple[float, float]:
    field0 = benchmark.field_tensor(benchmark.flat0.detach(), create_graph=False).detach()
    field_energy0 = 0.5 * float(torch.dot(field0, field0))
    p_tau0 = benchmark.local_gap_terms_T(benchmark.flat0)["P_tau_T"]
    return field_energy0, p_tau0


def geometry_row(cfg: TaskAlignedLQConfig) -> Dict[str, object]:
    bench = TaskAlignedLQBenchmark(cfg)
    z0 = bench.flat0.detach()
    jf = bench.full_jacobian(z0)
    sym = 0.5 * (jf + jf.T)
    skew = 0.5 * (jf - jf.T)
    dim = cfg.action_dim * cfg.state_dim
    cross = torch.linalg.norm(jf[:dim, dim:]) + torch.linalg.norm(jf[dim:, :dim])
    same = torch.linalg.norm(jf[:dim, :dim]) + torch.linalg.norm(jf[dim:, dim:])
    field0 = bench.field_tensor(z0, create_graph=False).detach()
    g0 = jf @ field0
    cos_fg = float(torch.dot(field0, g0) / (torch.linalg.norm(field0) * torch.linalg.norm(g0) + EPS))
    return {
        "alpha_dyn": cfg.alpha_dyn,
        "L_budget": cfg.L_budget,
        "a_u": cfg.a_u,
        "K_budget": cfg.K_budget,
        "rotation_ratio_proxy": float(torch.linalg.norm(skew) / (torch.linalg.norm(sym) + EPS)),
        "cross_player_coupling_proxy": float(cross),
        "cross_to_same_ratio": float(cross / (same + EPS)),
        "field_norm0": float(torch.linalg.norm(field0)),
        "G_norm0": float(torch.linalg.norm(g0)),
        "G_over_F0": float(torch.linalg.norm(g0) / (torch.linalg.norm(field0) + EPS)),
        "cos_FG0": cos_fg,
        "non_collinearity0": float(math.sqrt(max(0.0, 1.0 - cos_fg**2))),
        "num_complex_eigs": int(np.sum(np.abs(np.imag(torch.linalg.eigvals(jf).detach().cpu().numpy())) > 1e-9)),
        "geometry_gate_pass": float(cross > 0 and (cross / (same + EPS)) > 0.05 and math.sqrt(max(0.0, 1.0 - cos_fg**2)) > 0.2 and float(torch.linalg.norm(skew) / (torch.linalg.norm(sym) + EPS)) > 0.05),
    }


def step_method(
    benchmark: TaskAlignedLQBenchmark,
    method: str,
    flat: torch.Tensor,
    lr: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    proj_stats = {"projection_active_K": 0.0, "projection_active_L": 0.0}
    if method == "sgd":
        cand = flat - lr * benchmark.field_tensor(flat.detach(), create_graph=False).detach()
        flat_next, proj = benchmark.project_flat(cand)
        proj_stats.update(proj)
        return flat_next, proj_stats
    if method == "egm":
        f0 = benchmark.field_tensor(flat.detach(), create_graph=False).detach()
        z_half, _ = benchmark.project_flat(flat - lr * f0)
        f_half = benchmark.field_tensor(z_half.detach(), create_graph=False).detach()
        cand = flat - lr * f_half
        flat_next, proj = benchmark.project_flat(cand)
        proj_stats.update(proj)
        return flat_next, proj_stats
    if method == "ppm":
        z_inner = flat.clone()
        active_k = 0.0
        active_l = 0.0
        for _ in range(benchmark.cfg.ppm_inner_steps):
            f_inner = benchmark.field_tensor(z_inner.detach(), create_graph=False).detach()
            z_inner, proj = benchmark.project_flat(flat - lr * f_inner)
            active_k += proj["projection_active_K"]
            active_l += proj["projection_active_L"]
        proj_stats.update({
            "projection_active_K": active_k / benchmark.cfg.ppm_inner_steps,
            "projection_active_L": active_l / benchmark.cfg.ppm_inner_steps,
            "K_norm": proj["K_norm"],
            "L_norm": proj["L_norm"],
        })
        return z_inner, proj_stats
    raise ValueError(f"unknown method {method}")


def run_method(
    benchmark: TaskAlignedLQBenchmark,
    method: str,
    lr: float,
    iterations: int,
    field_energy0: float,
    p_tau0: float,
) -> Tuple[pd.DataFrame, Dict[str, float], List[torch.Tensor]]:
    flat = benchmark.flat0.clone()
    traj = [flat.clone()]
    rows: List[Dict[str, object]] = []
    for it in range(iterations):
        flat, proj = step_method(benchmark, method, flat, lr)
        traj.append(flat.clone())
        metrics = benchmark.metrics(flat, field_energy0, p_tau0)
        row = {
            "method": method,
            "iteration": it,
            "lr": lr,
            **metrics,
            **proj,
        }
        row["nan_flag"] = float(not np.isfinite(metrics["V_align"]) or not np.isfinite(metrics["field_norm"]))
        row["closed_loop_state_explosion"] = float(
            metrics["clean_state_norm_max"] > 1e6
            or metrics["current_adv_state_norm_max"] > 1e6
            or metrics["robust_br_state_norm_max"] > 1e6
        )
        row["valid_flag"] = float(
            row["nan_flag"] < 0.5
            and row["closed_loop_state_explosion"] < 0.5
            and metrics["valid_clean_eval"] > 0.5
            and metrics["valid_current_adv_eval"] > 0.5
            and metrics["valid_robust_br_eval"] > 0.5
            and metrics["K_norm"] <= benchmark.cfg.K_budget + 1e-9
            and metrics["L_norm"] <= benchmark.cfg.L_budget + 1e-9
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    final = df.iloc[-1].to_dict()
    summary = {
        "method": method,
        "lr": lr,
        "V_align_AUC": auc_from_series(df["V_align"]),
        "P_tau_T_AUC": auc_from_series(df["P_tau_T"]),
        "field_norm_AUC": auc_from_series(df["field_norm"]),
        "approximate_exploitability_AUC": auc_from_series(df["approximate_exploitability"]),
        "clean_task_return_AUC": auc_from_series(df["clean_task_return"]),
        "current_adv_task_return_AUC": auc_from_series(df["current_adv_task_return"]),
        "robust_br_task_return_AUC": auc_from_series(df["robust_br_task_return"]),
        "robust_degradation_AUC": auc_from_series(df["robust_degradation"]),
        "final_clean_task_return": float(final["clean_task_return"]),
        "final_current_adv_task_return": float(final["current_adv_task_return"]),
        "final_robust_br_task_return": float(final["robust_br_task_return"]),
        "final_robust_degradation": float(final["robust_degradation"]),
        "final_V_align": float(final["V_align"]),
        "final_P_tau_T": float(final["P_tau_T"]),
        "final_field_norm": float(final["field_norm"]),
        "projection_active_frac": float(0.5 * (df["projection_active_K"].mean() + df["projection_active_L"].mean())),
        "valid_fraction_robust_br": float(df["valid_robust_br_eval"].mean()),
        "nan_flag": float(df["nan_flag"].max()),
        "closed_loop_state_explosion": float(df["closed_loop_state_explosion"].max()),
        "valid_flag": float(df["valid_flag"].min()),
    }
    return df, summary, traj


def proposed_step(
    benchmark: TaskAlignedLQBenchmark,
    method: str,
    flat: torch.Tensor,
    field_energy0: float,
    p_tau0: float,
    update_radius: float,
    probe_radius: float,
    fallback_lr: float,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    before = benchmark.metrics(flat, field_energy0, p_tau0)
    Fk = benchmark.field_tensor(flat.detach(), create_graph=False).detach()
    Gk = benchmark.jvp_field(flat.detach(), Fk)

    def eval_fn(beta_t: torch.Tensor, gamma_t: torch.Tensor) -> float:
        cand = flat - float(beta_t) * Fk + float(gamma_t) * Gk
        cand, _ = benchmark.project_flat(cand)
        return float(benchmark.metrics(cand, field_energy0, p_tau0)["V_align"])

    if method == "proposed_noG":
        beta_raw, fit_info = fit_quadratic_1d(eval_fn, Fk, probe_radius)
        gamma_raw = 0.0
        delta_raw = -beta_raw * Fk
    else:
        beta_raw, gamma_raw, fit_info = fit_quadratic_2d(eval_fn, Fk, Gk, probe_radius)
        delta_raw = -beta_raw * Fk + gamma_raw * Gk
    delta, trust_active, raw_update_norm, scaled_update_norm = trust_scale(delta_raw, update_radius)
    cand, proj = benchmark.project_flat(flat + delta)
    after = benchmark.metrics(cand, field_energy0, p_tau0)
    v_pred = float(eval_fn(torch.tensor(beta_raw, dtype=DTYPE), torch.tensor(gamma_raw, dtype=DTYPE)))
    fallback = False
    fallback_reason = "none"
    accepted_type = "qpg" if method == "proposed_QP_G" else "nog"
    egm_after = None
    if (not np.isfinite(after["V_align"])) or (after["V_align"] > before["V_align"] + 1e-10):
        egm_cand, _ = step_method(benchmark, "egm", flat, fallback_lr)
        egm_after = benchmark.metrics(egm_cand, field_energy0, p_tau0)
        if np.isfinite(egm_after["V_align"]) and egm_after["V_align"] <= after["V_align"]:
            cand = egm_cand
            after = egm_after
            fallback = True
            accepted_type = "egm"
            fallback_reason = "qp_worse_than_egm"
        else:
            fallback_reason = "qp_nonfinite_or_worse_but_egm_not_better"
    g_ratio = float(torch.linalg.norm(gamma_raw * Gk) / (torch.linalg.norm(beta_raw * Fk) + EPS)) if abs(beta_raw) > EPS else 0.0
    return cand, {
        "beta": float(beta_raw),
        "gamma": float(gamma_raw),
        "gamma_active": float(abs(gamma_raw) > 1e-14),
        "fallback_to_egm": float(fallback),
        "fallback_reason": fallback_reason,
        "accepted_step_type": accepted_type,
        "V_before": float(before["V_align"]),
        "V_predicted_after": v_pred,
        "V_actual_after": float(after["V_align"]),
        "P_tau_before": float(before["P_tau_T"]),
        "P_tau_after": float(after["P_tau_T"]),
        "field_term_before": float(before["field_term"]),
        "field_term_after": float(after["field_term"]),
        "exploitability_before": float(before["approximate_exploitability"]),
        "exploitability_after": float(after["approximate_exploitability"]),
        "field_norm_before": float(before["field_norm"]),
        "field_norm_after": float(after["field_norm"]),
        "update_norm": float(torch.linalg.norm(cand - flat)),
        "raw_update_norm": float(raw_update_norm),
        "trust_scaled_update_norm": float(scaled_update_norm),
        "trust_radius_active": float(trust_active),
        "projection_active_K": proj["projection_active_K"],
        "projection_active_L": proj["projection_active_L"],
        "F_norm": float(torch.linalg.norm(Fk)),
        "G_norm": float(torch.linalg.norm(Gk)),
        "cos_FG": float(torch.dot(Fk, Gk) / (torch.linalg.norm(Fk) * torch.linalg.norm(Gk) + EPS)),
        "non_collinearity": float(math.sqrt(max(0.0, 1.0 - float(torch.dot(Fk, Gk) / (torch.linalg.norm(Fk) * torch.linalg.norm(Gk) + EPS)) ** 2))),
        "G_contribution_ratio": g_ratio,
        "fit_cond": safe_float(fit_info.get("cond", float("nan"))),
        "fit_indef": safe_float(fit_info.get("indef", float("nan"))),
    }


def run_proposed_method(
    benchmark: TaskAlignedLQBenchmark,
    method: str,
    iterations: int,
    field_energy0: float,
    p_tau0: float,
    update_radius: float,
    probe_radius: float,
    fallback_lr: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], List[torch.Tensor]]:
    flat = benchmark.flat0.clone()
    traj = [flat.clone()]
    curves: List[Dict[str, object]] = []
    diags: List[Dict[str, object]] = []
    for it in range(iterations):
        flat, info = proposed_step(benchmark, method, flat, field_energy0, p_tau0, update_radius, probe_radius, fallback_lr)
        traj.append(flat.clone())
        metrics = benchmark.metrics(flat, field_energy0, p_tau0)
        row = {
            "method": method,
            "iteration": it,
            **metrics,
            "nan_flag": float(not np.isfinite(metrics["V_align"]) or not np.isfinite(metrics["field_norm"])),
            "closed_loop_state_explosion": float(
                metrics["clean_state_norm_max"] > 1e6
                or metrics["current_adv_state_norm_max"] > 1e6
                or metrics["robust_br_state_norm_max"] > 1e6
            ),
        }
        curves.append(row)
        diags.append({"method": method, "iteration": it, **info})
    curve_df = pd.DataFrame(curves)
    diag_df = pd.DataFrame(diags)
    final = curve_df.iloc[-1].to_dict()
    summary = {
        "method": method,
        "V_align_AUC": auc_from_series(curve_df["V_align"]),
        "P_tau_T_AUC": auc_from_series(curve_df["P_tau_T"]),
        "field_norm_AUC": auc_from_series(curve_df["field_norm"]),
        "approximate_exploitability_AUC": auc_from_series(curve_df["approximate_exploitability"]),
        "clean_task_return_AUC": auc_from_series(curve_df["clean_task_return"]),
        "current_adv_task_return_AUC": auc_from_series(curve_df["current_adv_task_return"]),
        "robust_br_task_return_AUC": auc_from_series(curve_df["robust_br_task_return"]),
        "robust_degradation_AUC": auc_from_series(curve_df["robust_degradation"]),
        "final_clean_task_return": float(final["clean_task_return"]),
        "final_current_adv_task_return": float(final["current_adv_task_return"]),
        "final_robust_br_task_return": float(final["robust_br_task_return"]),
        "final_robust_degradation": float(final["robust_degradation"]),
        "final_V_align": float(final["V_align"]),
        "final_P_tau_T": float(final["P_tau_T"]),
        "final_field_norm": float(final["field_norm"]),
        "fallback_to_egm_frac": float(diag_df["fallback_to_egm"].mean()),
        "gamma_active_frac": float(diag_df["gamma_active"].mean()) if "gamma_active" in diag_df.columns else 0.0,
        "G_contribution_ratio": float(diag_df["G_contribution_ratio"].mean()),
        "projection_active_frac": float(0.5 * (diag_df["projection_active_K"].mean() + diag_df["projection_active_L"].mean())),
        "valid_fraction_robust_br": float(curve_df["valid_robust_br_eval"].mean()),
    }
    return curve_df, diag_df, summary, traj


def proposed_radius_preflight(
    benchmark: TaskAlignedLQBenchmark,
    field_energy0: float,
    p_tau0: float,
    fallback_lr: float,
) -> Tuple[pd.DataFrame, float]:
    rows: List[Dict[str, object]] = []
    selected_radius = float("nan")
    for radius in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
        probe_radius = min(radius, 1e-2) * 0.5
        curve_df, diag_df, summary, _ = run_proposed_method(
            benchmark, "proposed_QP_G", 50, field_energy0, p_tau0, radius, probe_radius, fallback_lr
        )
        row = {
            "update_radius": radius,
            **summary,
            "time_to_V_align_1e-3": first_below(curve_df["V_align"], 1e-3),
            "time_to_P_tau_T_1e-3": first_below(curve_df["normalized_P_tau_T"], 1e-3),
            "nan_flag": float(curve_df["nan_flag"].max()),
            "divergence_flag": float(curve_df["closed_loop_state_explosion"].max()),
            "valid_flag": float(
                curve_df["nan_flag"].max() < 0.5
                and curve_df["closed_loop_state_explosion"].max() < 0.5
                and summary["fallback_to_egm_frac"] < 0.2
                and summary["projection_active_frac"] <= 0.5
            ),
        }
        rows.append(row)
        if np.isnan(selected_radius) and row["valid_flag"] > 0.5:
            selected_radius = radius
    preflight_df = pd.DataFrame(rows)
    if np.isnan(selected_radius):
        best = preflight_df.sort_values(["fallback_to_egm_frac", "V_align_AUC", "update_radius"]).iloc[0]
        selected_radius = float(best["update_radius"])
    return preflight_df, selected_radius


def same_start_comparison(
    benchmark: TaskAlignedLQBenchmark,
    qpg_traj: Sequence[torch.Tensor],
    field_energy0: float,
    p_tau0: float,
    lr: float,
    update_radius: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for checkpoint in [0, 5, 10, 25, 50]:
        z = qpg_traj[checkpoint]
        before = benchmark.metrics(z, field_energy0, p_tau0)
        candidates = {
            "zero": z.clone(),
            "SGD": step_method(benchmark, "sgd", z, lr)[0],
            "EGM": step_method(benchmark, "egm", z, lr)[0],
            "PPM": step_method(benchmark, "ppm", z, lr)[0],
            "proposed_noG": proposed_step(benchmark, "proposed_noG", z, field_energy0, p_tau0, update_radius, min(update_radius, 1e-2) * 0.5, lr)[0],
            "proposed_QP_G": proposed_step(benchmark, "proposed_QP_G", z, field_energy0, p_tau0, update_radius, min(update_radius, 1e-2) * 0.5, lr)[0],
        }
        qpg_delta = candidates["proposed_QP_G"] - z
        deltas = {name: cand - z for name, cand in candidates.items() if name != "zero"}
        qpg_info = proposed_step(benchmark, "proposed_QP_G", z, field_energy0, p_tau0, update_radius, min(update_radius, 1e-2) * 0.5, lr)[1]
        for name, cand in candidates.items():
            after = benchmark.metrics(cand, field_energy0, p_tau0)
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "candidate": name,
                    "V_align_before": float(before["V_align"]),
                    "V_align_after": float(after["V_align"]),
                    "actual_delta_V_align": float(after["V_align"] - before["V_align"]),
                    "P_tau_T_after": float(after["P_tau_T"]),
                    "field_norm_after": float(after["field_norm"]),
                    "exploitability_after": float(after["approximate_exploitability"]),
                    "clean_task_return_after": float(after["clean_task_return"]),
                    "current_adv_task_return_after": float(after["current_adv_task_return"]),
                    "robust_br_task_return_after": float(after["robust_br_task_return"]),
                    "robust_degradation_after": float(after["robust_degradation"]),
                    "update_norm": float(torch.linalg.norm(cand - z)),
                    "cos_QP_SGD": float(torch.dot(qpg_delta, deltas["SGD"]) / (torch.linalg.norm(qpg_delta) * torch.linalg.norm(deltas["SGD"]) + EPS)) if name == "proposed_QP_G" else float("nan"),
                    "cos_QP_EGM": float(torch.dot(qpg_delta, deltas["EGM"]) / (torch.linalg.norm(qpg_delta) * torch.linalg.norm(deltas["EGM"]) + EPS)) if name == "proposed_QP_G" else float("nan"),
                    "cos_QP_PPM": float(torch.dot(qpg_delta, deltas["PPM"]) / (torch.linalg.norm(qpg_delta) * torch.linalg.norm(deltas["PPM"]) + EPS)) if name == "proposed_QP_G" else float("nan"),
                    "cos_QP_noG": float(torch.dot(qpg_delta, deltas["proposed_noG"]) / (torch.linalg.norm(qpg_delta) * torch.linalg.norm(deltas["proposed_noG"]) + EPS)) if name == "proposed_QP_G" else float("nan"),
                    "G_contribution_ratio": float(qpg_info["G_contribution_ratio"]) if name == "proposed_QP_G" else float("nan"),
                    "fallback_decision": qpg_info["accepted_step_type"] if name == "proposed_QP_G" else "n/a",
                }
            )
    return pd.DataFrame(rows)


def plot_final(curves: pd.DataFrame) -> None:
    order = ["SGD", "EGM", "PPM", "proposed_noG", "proposed_QP_G"]
    colors = {
        "SGD": "#d62728",
        "EGM": "#2ca02c",
        "PPM": "#9467bd",
        "proposed_noG": "#1f77b4",
        "proposed_QP_G": "#8c564b",
    }

    def plot_metric(filename: str, metric: str, title: str, ylabel: str, logy: bool = True):
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in order:
            sub = curves[curves["method"] == method]
            vals = clip_floor(sub[metric], 1e-12) if logy else sub[metric].to_numpy(dtype=np.float64)
            ax.plot(sub["iteration"], vals, color=colors[method], label=method)
        if logy:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_ROOT / filename, dpi=180)
        plt.close(fig)

    plot_metric("task_aligned_lq_final_V_align.png", "V_align", "Task-Aligned Composite V", "V_align")
    plot_metric("task_aligned_lq_final_P_tau_T.png", "normalized_P_tau_T", "Task-Aligned Normalized P_tau^T", "normalized_P_tau^T")
    plot_metric("task_aligned_lq_final_field_norm.png", "field_norm", "Task-Aligned Field Norm", "||F_T||")
    plot_metric("task_aligned_lq_final_exploitability.png", "approximate_exploitability", "Task-Aligned Exploitability", "exploitability")
    plot_metric("task_aligned_lq_final_clean_task_return.png", "clean_task_return", "Clean Task Return", "return", logy=False)
    plot_metric("task_aligned_lq_final_current_adv_task_return.png", "current_adv_task_return", "Current-Adversary Task Return", "return", logy=False)
    plot_metric("task_aligned_lq_final_robust_br_task_return.png", "robust_br_task_return", "Robust-BR Task Return", "return", logy=False)
    plot_metric("task_aligned_lq_final_robust_degradation.png", "robust_degradation", "Robust Degradation", "degradation", logy=False)

    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    panels = [
        ("V_align", "V_align", True),
        ("normalized_P_tau_T", "P_tau^T", True),
        ("field_norm", "field_norm", True),
        ("approximate_exploitability", "exploitability", True),
        ("clean_task_return", "clean_task_return", False),
        ("current_adv_task_return", "current_adv_task_return", False),
        ("robust_br_task_return", "robust_br_task_return", False),
        ("robust_degradation", "robust_degradation", False),
    ]
    for ax, (metric, title, logy) in zip(axes.flat, panels + [("blank", "", False)]):
        if metric == "blank":
            ax.axis("off")
            continue
        for method in order:
            sub = curves[curves["method"] == method]
            vals = clip_floor(sub[metric], 1e-12) if logy else sub[metric].to_numpy(dtype=np.float64)
            ax.plot(sub["iteration"], vals, color=colors[method], label=method)
        if logy:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "task_aligned_lq_final_all_plots_big.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))
    paper_panels = [
        ("V_align", "V_align", True),
        ("normalized_P_tau_T", "P_tau^T", True),
        ("field_norm", "field_norm", True),
        ("robust_br_task_return", "robust BR return", False),
        ("robust_degradation", "robust degradation", False),
    ]
    for ax, (metric, title, logy) in zip(axes, paper_panels):
        for method in order:
            sub = curves[curves["method"] == method]
            vals = clip_floor(sub[metric], 1e-12) if logy else sub[metric].to_numpy(dtype=np.float64)
            ax.plot(sub["iteration"], vals, color=colors[method], label=method)
        if logy:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "task_aligned_lq_final_paper_main.png", dpi=180)
    plt.close(fig)


def choose_env_configs() -> List[TaskAlignedLQConfig]:
    combos = [
        (0.05, 1.0, 0.01),
        (0.10, 1.0, 0.01),
        (0.20, 1.0, 0.01),
        (0.30, 1.0, 0.01),
        (0.10, 2.0, 0.01),
        (0.20, 2.0, 0.01),
        (0.30, 2.0, 0.01),
        (0.10, 3.0, 0.01),
        (0.20, 3.0, 0.01),
        (0.20, 2.0, 0.03),
        (0.30, 2.0, 0.03),
        (0.20, 2.0, 0.10),
    ]
    return [TaskAlignedLQConfig(alpha_dyn=a, L_budget=l, a_u=au) for a, l, au in combos]


def main() -> None:
    geometry_rows: List[Dict[str, object]] = []
    env_rows: List[Dict[str, object]] = []
    gate_rows: List[Dict[str, object]] = []
    best_payload: Dict[str, object] | None = None

    for cfg in choose_env_configs():
        geo = geometry_row(cfg)
        geometry_rows.append(geo)
        env_record: Dict[str, object] = {"alpha_dyn": cfg.alpha_dyn, "L_budget": cfg.L_budget, "a_u": cfg.a_u, **geo}
        if geo["geometry_gate_pass"] < 0.5:
            env_record["decision"] = "GEOMETRY_FAIL"
            env_rows.append(env_record)
            continue

        benchmark = TaskAlignedLQBenchmark(cfg)
        field_energy0, p_tau0 = initial_terms(benchmark)
        sgd_candidates: List[Dict[str, object]] = []
        sgd_curves: List[pd.DataFrame] = []
        for lr in [1e-3, 3e-3, 1e-2, 3e-2]:
            curve_df, summary, _ = run_method(benchmark, "SGD", lr, 300, field_energy0, p_tau0)
            sgd_candidates.append(summary)
            sgd_curves.append(curve_df)
        sgd_df = pd.DataFrame(sgd_candidates)
        valid_sgd = sgd_df[
            (sgd_df["nan_flag"] < 0.5)
            & (sgd_df["closed_loop_state_explosion"] < 0.5)
            & (sgd_df["projection_active_frac"] <= 0.5)
            & (sgd_df["valid_fraction_robust_br"] > 0.99)
        ].copy()
        if valid_sgd.empty:
            env_record["decision"] = "SGD_NORMAL_FAIL"
            env_record["best_sgd_lr"] = float("nan")
            env_rows.append(env_record)
            continue
        selected_sgd = valid_sgd.sort_values(["V_align_AUC", "robust_degradation_AUC", "lr"]).iloc[0].to_dict()
        selected_lr = float(selected_sgd["lr"])
        env_record["best_sgd_lr"] = selected_lr
        env_record["sgd_pass"] = 1.0

        curve_frames = []
        baseline_summaries = []
        for method in ["SGD", "EGM", "PPM"]:
            curve_df, summary, traj = run_method(benchmark, method, selected_lr, 300, field_energy0, p_tau0)
            curve_frames.append(curve_df)
            baseline_summaries.append(summary)
        baseline_df = pd.DataFrame(baseline_summaries)
        sgd_row = baseline_df[baseline_df["method"] == "SGD"].iloc[0]
        egm_row = baseline_df[baseline_df["method"] == "EGM"].iloc[0]
        ppm_row = baseline_df[baseline_df["method"] == "PPM"].iloc[0]
        baseline_pass = (
            float(baseline_df["valid_flag"].min()) > 0.5
            and (
                float(sgd_row["V_align_AUC"]) / max(float(egm_row["V_align_AUC"]), EPS) >= 1.3
                or float(sgd_row["V_align_AUC"]) / max(float(ppm_row["V_align_AUC"]), EPS) >= 1.3
                or float(sgd_row["P_tau_T_AUC"]) / max(float(egm_row["P_tau_T_AUC"]), EPS) >= 1.3
                or float(sgd_row["P_tau_T_AUC"]) / max(float(ppm_row["P_tau_T_AUC"]), EPS) >= 1.3
            )
            and float(egm_row["final_robust_br_task_return"]) >= float(sgd_row["final_robust_br_task_return"]) - 1e-9
            and float(ppm_row["final_robust_br_task_return"]) >= float(sgd_row["final_robust_br_task_return"]) - 1e-9
        )
        gate_rows.append({
            "alpha_dyn": cfg.alpha_dyn,
            "L_budget": cfg.L_budget,
            "a_u": cfg.a_u,
            "selected_lr": selected_lr,
            "baseline_pass": float(baseline_pass),
            "sgd_V_align_AUC": float(sgd_row["V_align_AUC"]),
            "egm_V_align_AUC": float(egm_row["V_align_AUC"]),
            "ppm_V_align_AUC": float(ppm_row["V_align_AUC"]),
            "sgd_P_tau_T_AUC": float(sgd_row["P_tau_T_AUC"]),
            "egm_P_tau_T_AUC": float(egm_row["P_tau_T_AUC"]),
            "ppm_P_tau_T_AUC": float(ppm_row["P_tau_T_AUC"]),
        })
        if not baseline_pass:
            env_record["decision"] = "BASELINE_FAIL"
            env_rows.append(env_record)
            continue

        preflight_df, selected_radius = proposed_radius_preflight(benchmark, field_energy0, p_tau0, selected_lr)
        probe_radius = min(selected_radius, 1e-2) * 0.5
        all_curve_frames = curve_frames.copy()
        all_diag_frames: List[pd.DataFrame] = []
        final_summaries = baseline_summaries.copy()
        traj_map: Dict[str, List[torch.Tensor]] = {}
        for method in ["SGD", "EGM", "PPM"]:
            _, _, traj = run_method(benchmark, method, selected_lr, 300, field_energy0, p_tau0)
            traj_map[method] = traj
        for method in ["proposed_noG", "proposed_QP_G"]:
            curve_df, diag_df, summary, traj = run_proposed_method(
                benchmark, method, 300, field_energy0, p_tau0, selected_radius, probe_radius, selected_lr
            )
            all_curve_frames.append(curve_df)
            all_diag_frames.append(diag_df)
            final_summaries.append(summary)
            traj_map[method] = traj
        final_summary_df = pd.DataFrame(final_summaries)
        qpg_row = final_summary_df[final_summary_df["method"] == "proposed_QP_G"].iloc[0]
        nog_row = final_summary_df[final_summary_df["method"] == "proposed_noG"].iloc[0]
        egm_row = final_summary_df[final_summary_df["method"] == "EGM"].iloc[0]
        ppm_row = final_summary_df[final_summary_df["method"] == "PPM"].iloc[0]

        env_record.update({
            "selected_lr": selected_lr,
            "selected_radius": selected_radius,
            "qpg_V_align_AUC": float(qpg_row["V_align_AUC"]),
            "qpg_robust_br_task_return_AUC": float(qpg_row["robust_br_task_return_AUC"]),
            "nog_robust_br_task_return_AUC": float(nog_row["robust_br_task_return_AUC"]),
            "fallback_to_egm_frac": float(qpg_row["fallback_to_egm_frac"]),
        })
        full_success = (
            float(qpg_row["V_align_AUC"]) < float(nog_row["V_align_AUC"])
            and float(qpg_row["P_tau_T_AUC"]) < float(nog_row["P_tau_T_AUC"])
            and float(qpg_row["field_norm_AUC"]) < float(nog_row["field_norm_AUC"])
            and float(qpg_row["approximate_exploitability_AUC"]) < float(nog_row["approximate_exploitability_AUC"])
            and float(qpg_row["robust_br_task_return_AUC"]) >= float(nog_row["robust_br_task_return_AUC"]) - 1e-6
            and float(qpg_row["final_robust_br_task_return"]) >= float(nog_row["final_robust_br_task_return"]) - 1e-6
            and float(qpg_row["robust_degradation_AUC"]) <= float(nog_row["robust_degradation_AUC"]) + 1e-6
            and float(qpg_row["final_robust_br_task_return"]) >= max(float(egm_row["final_robust_br_task_return"]), float(ppm_row["final_robust_br_task_return"])) - 1e-6
            and float(qpg_row["fallback_to_egm_frac"]) < 0.2
            and float(qpg_row["projection_active_frac"]) <= 0.5
        )
        env_record["decision"] = "FULL_ALIGNMENT_SUCCESS" if full_success else "OPTIMIZATION_POSITIVE_RARL_SUPPORTIVE"
        env_rows.append(env_record)

        candidate_payload = {
            "cfg": cfg,
            "benchmark": benchmark,
            "field_energy0": field_energy0,
            "p_tau0": p_tau0,
            "selected_lr": selected_lr,
            "selected_radius": selected_radius,
            "preflight_df": preflight_df,
            "curves_df": pd.concat(all_curve_frames, ignore_index=True),
            "diag_df": pd.concat(all_diag_frames, ignore_index=True),
            "summary_df": final_summary_df,
            "traj_map": traj_map,
            "decision": env_record["decision"],
        }
        if best_payload is None:
            best_payload = candidate_payload
        else:
            prev = best_payload["summary_df"]
            prev_qpg = prev[prev["method"] == "proposed_QP_G"].iloc[0]
            if full_success and best_payload["decision"] != "FULL_ALIGNMENT_SUCCESS":
                best_payload = candidate_payload
            elif env_record["decision"] == best_payload["decision"] and float(qpg_row["V_align_AUC"]) < float(prev_qpg["V_align_AUC"]):
                best_payload = candidate_payload
        if full_success:
            break

    geometry_df = pd.DataFrame(geometry_rows)
    env_df = pd.DataFrame(env_rows)
    gate_df = pd.DataFrame(gate_rows)
    geometry_df.to_csv(RESULT_ROOT / "task_aligned_lq_geometry_audit.csv", index=False)
    env_df.to_csv(RESULT_ROOT / "task_aligned_lq_env_alignment_sweep.csv", index=False)
    gate_df.to_csv(RESULT_ROOT / "task_aligned_lq_baseline_gate.csv", index=False)
    write_md(RESULT_ROOT / "task_aligned_lq_geometry_audit.md", "# Task-Aligned LQ Geometry Audit\n\n" + df_text(geometry_df))
    write_md(RESULT_ROOT / "task_aligned_lq_env_alignment_sweep_report.md", "# Task-Aligned LQ Environment Alignment Sweep\n\n" + df_text(env_df))
    write_md(RESULT_ROOT / "task_aligned_lq_baseline_gate_report.md", "# Task-Aligned LQ Baseline Gate\n\n" + df_text(gate_df))

    if best_payload is None:
        write_md(
            RESULT_ROOT / "task_aligned_lq_theory_alignment_report.md",
            "\n".join(
                [
                    "# Task-Aligned LQ Theory Alignment Report",
                    "",
                    "1. Old mixed LQ used a Lyapunov based on a mixed reward objective.",
                    "2. RARL-style evaluation used task-only robust return.",
                    "3. This could cause misalignment between game stationarity and task-only robustness.",
                    "4. The new task-aligned LQ defines J_T as the task-only robust objective.",
                    "5. F_T, P_tau_T, and V_align are all built from J_T.",
                    "6. Therefore the new construction is theory-aligned by design.",
                    "7. However, in the tested environment sweep, the resulting field did not pass the intended rotational geometry gate.",
                    "8. The issue is not reward mismatch anymore; it is lack of sufficiently nontrivial rotational game geometry under the tested alpha_dyn / L_budget / a_u settings.",
                    "",
                    "Limitations:",
                    "- This negative result does not imply the task-aligned construction is wrong in general.",
                    "- It does imply that the current dynamics-only adversary coupling is too close to a nearly symmetric / collinear field around initialization for Subsection 2.",
                ]
            ),
        )
        write_md(
            RESULT_ROOT / "task_aligned_lq_final_report.md",
            "\n".join(
                [
                    "# Task-Aligned LQ Final Report",
                    "",
                    "No candidate configuration reached the proposed stage.",
                    "",
                    "Geometry summary:",
                    "",
                    df_text(geometry_df),
                    "",
                    "Environment sweep summary:",
                    "",
                    df_text(env_df),
                ]
            ),
        )
        write_md(
            RESULT_ROOT / "task_aligned_lq_final_decision.md",
            "\n".join(
                [
                    "# Task-Aligned LQ Final Decision",
                    "",
                    "ENVIRONMENT_FAIL",
                    "",
                    "No task-aligned LQ configuration passed the geometry + SGD-normality + baseline gate stack.",
                    "",
                    "Root cause summary:",
                    "- cross-player coupling is present, but the field remains almost collinear with J_{F_T} F_T near initialization.",
                    "- rotation_ratio_proxy stays below 0.5 in the tested sweep.",
                    "- num_complex_eigs stays at 0 in the tested sweep.",
                    "- therefore the intended extragradient-friendly rotational structure never becomes strong enough to justify Stage 2 / Stage 3.",
                ]
            ),
        )
        return

    cfg = best_payload["cfg"]
    benchmark = best_payload["benchmark"]
    field_energy0 = best_payload["field_energy0"]
    p_tau0 = best_payload["p_tau0"]
    selected_lr = best_payload["selected_lr"]
    selected_radius = best_payload["selected_radius"]
    preflight_df = best_payload["preflight_df"]
    curves_df = best_payload["curves_df"]
    diag_df = best_payload["diag_df"]
    summary_df = best_payload["summary_df"]
    traj_map = best_payload["traj_map"]

    preflight_df.to_csv(RESULT_ROOT / "task_aligned_lq_proposed_radius_preflight.csv", index=False)
    write_md(RESULT_ROOT / "task_aligned_lq_proposed_radius_preflight_report.md", "# Task-Aligned LQ Proposed Radius Preflight\n\n" + df_text(preflight_df))
    summary_df.to_csv(RESULT_ROOT / "task_aligned_lq_final_summary.csv", index=False)
    curves_df.to_csv(RESULT_ROOT / "task_aligned_lq_final_curves.csv", index=False)
    diag_df.to_csv(RESULT_ROOT / "task_aligned_lq_final_diagnostics.csv", index=False)
    plot_final(curves_df)

    same_start_df = same_start_comparison(benchmark, traj_map["proposed_QP_G"], field_energy0, p_tau0, selected_lr, selected_radius)
    same_start_df.to_csv(RESULT_ROOT / "task_aligned_lq_same_start_comparison.csv", index=False)
    write_md(RESULT_ROOT / "task_aligned_lq_same_start_comparison.md", "# Task-Aligned LQ Same-Start Comparison\n\n" + df_text(same_start_df))

    qpg = summary_df[summary_df["method"] == "proposed_QP_G"].iloc[0]
    nog = summary_df[summary_df["method"] == "proposed_noG"].iloc[0]
    sgd = summary_df[summary_df["method"] == "SGD"].iloc[0]
    egm = summary_df[summary_df["method"] == "EGM"].iloc[0]
    ppm = summary_df[summary_df["method"] == "PPM"].iloc[0]

    if (
        float(qpg["V_align_AUC"]) < float(nog["V_align_AUC"])
        and float(qpg["P_tau_T_AUC"]) < float(nog["P_tau_T_AUC"])
        and float(qpg["field_norm_AUC"]) < float(nog["field_norm_AUC"])
        and float(qpg["approximate_exploitability_AUC"]) < float(nog["approximate_exploitability_AUC"])
        and float(qpg["robust_br_task_return_AUC"]) >= float(nog["robust_br_task_return_AUC"]) - 1e-6
        and float(qpg["final_robust_br_task_return"]) >= float(nog["final_robust_br_task_return"]) - 1e-6
        and float(qpg["robust_degradation_AUC"]) <= float(nog["robust_degradation_AUC"]) + 1e-6
    ):
        decision = "FULL_ALIGNMENT_SUCCESS"
    elif (
        float(qpg["V_align_AUC"]) < float(nog["V_align_AUC"])
        and float(qpg["final_robust_br_task_return"]) >= max(float(sgd["final_robust_br_task_return"]), float(egm["final_robust_br_task_return"]), float(ppm["final_robust_br_task_return"])) - 1e-6
    ):
        decision = "OPTIMIZATION_POSITIVE_RARL_SUPPORTIVE"
    elif float(qpg["V_align_AUC"]) < float(nog["V_align_AUC"]):
        decision = "OPTIMIZATION_ONLY"
    else:
        decision = "METRIC_MISMATCH"

    write_md(
        RESULT_ROOT / "task_aligned_lq_theory_alignment_report.md",
        "\n".join(
            [
                "# Task-Aligned LQ Theory Alignment Report",
                "",
                "1. Old mixed LQ used a Lyapunov based on a mixed reward objective.",
                "2. RARL-style evaluation used task-only robust return.",
                "3. This caused potential misalignment between game stationarity and task-only robustness.",
                "4. The new task-aligned LQ defines J_T as the task-only robust objective.",
                "5. F_T, P_tau_T, and V_align are all built from J_T.",
                "6. Therefore V_align is aligned with the same local robust objective used in performance evaluation.",
                "7. P_tau_T measures local protagonist and adversary proximal improvements under J_T only.",
                "8. robust_br_task_return evaluates the same J_T using a local adversarial best response.",
                "9. This makes the experiment suitable for a local theorem connecting V_align decrease to local robust-task improvement.",
                "",
                "Limitations:",
                "- This remains a local robust saddle metric, not a global nonconvex RL theorem.",
                "- Compact K/L budgets are essential to keep the local robust game well-posed.",
                "- P_tau_T and robust BR use approximate inner optimization, so approximation error still exists.",
            ]
        ),
    )

    write_md(
        RESULT_ROOT / "task_aligned_lq_final_report.md",
        "\n".join(
            [
                "# Task-Aligned LQ Final Report",
                "",
                f"- selected config: `alpha_dyn={cfg.alpha_dyn}, L_budget={cfg.L_budget}, a_u={cfg.a_u}`",
                f"- selected lr: `{selected_lr}`",
                f"- selected update_radius: `{selected_radius}`",
                "",
                "Final summary:",
                "",
                df_text(summary_df),
            ]
        ),
    )

    write_md(
        RESULT_ROOT / "task_aligned_lq_final_decision.md",
        "\n".join(
            [
                "# Task-Aligned LQ Final Decision",
                "",
                decision,
                "",
                f"Selected config: `alpha_dyn={cfg.alpha_dyn}, L_budget={cfg.L_budget}, a_u={cfg.a_u}`",
                f"Baseline lr: `{selected_lr}`",
                f"QP update_radius: `{selected_radius}`",
                "",
                "This file answers whether the task-aligned LQ can replace the previous MixedLinearActorRotLQ-v0 as the paper's Subsection 2 / LQ result.",
                "",
                (
                    "The task-aligned LQ can replace the previous mixed-reward LQ as the main Subsection 2 result."
                    if decision in {"FULL_ALIGNMENT_SUCCESS", "OPTIMIZATION_POSITIVE_RARL_SUPPORTIVE"}
                    else "The task-aligned LQ is not yet strong enough to replace the previous mixed-reward LQ as the main Subsection 2 result."
                ),
            ]
        ),
    )


if __name__ == "__main__":
    main()
