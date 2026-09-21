from __future__ import annotations

import math
from dataclasses import dataclass
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
RESULT_ROOT = ROOT / "original" / "results" / "aligned_rotational_lq_rarl"
PLOT_ROOT = RESULT_ROOT / "plots"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)
PLOT_ROOT.mkdir(parents=True, exist_ok=True)


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
class AlignedRotLQConfig:
    state_dim: int = 2
    action_dim: int = 2
    disturbance_dim: int = 2
    omega: float = 0.2
    gamma: float = 0.95
    horizon: int = 50
    train_batch_size: int = 256
    eval_batch_size: int = 256
    seed: int = 0
    init_scale: float = 0.03
    q_state: float = 1.0
    a_u: float = 0.01
    a_w: float = 0.01
    beta_rot: float = 1.0
    beta_sym: float = 0.1
    alpha_dyn: float = 0.1
    K_budget: float = 3.0
    L_budget: float = 2.0
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


class AlignedRotLQBenchmark:
    def __init__(self, cfg: AlignedRotLQConfig):
        self.cfg = cfg
        c = math.cos(cfg.omega)
        s = math.sin(cfg.omega)
        r_state = torch.tensor([[c, -s], [s, c]], dtype=DTYPE)
        self.A = 0.90 * r_state
        self.B = torch.eye(cfg.action_dim, dtype=DTYPE)
        self.E = torch.eye(cfg.disturbance_dim, dtype=DTYPE)
        self.R_dyn = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=DTYPE)
        self.H2 = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=DTYPE)
        self.S2 = torch.tensor([[1.0, 0.2], [0.2, -0.5]], dtype=DTYPE)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(cfg.seed)
        self.train_x0 = torch.randn(cfg.train_batch_size, cfg.state_dim, generator=gen, dtype=DTYPE)
        self.eval_x0 = torch.randn(cfg.eval_batch_size, cfg.state_dim, generator=gen, dtype=DTYPE)
        self.train_sigma0 = (self.train_x0.T @ self.train_x0) / cfg.train_batch_size
        self.eval_sigma0 = (self.eval_x0.T @ self.eval_x0) / cfg.eval_batch_size
        self.K0 = cfg.init_scale * torch.randn(cfg.action_dim, cfg.state_dim, generator=gen, dtype=DTYPE)
        self.L0 = cfg.init_scale * torch.randn(cfg.disturbance_dim, cfg.state_dim, generator=gen, dtype=DTYPE)
        self.flat0 = self.join_flat(self.K0, self.L0)

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

    def _reward_terms_tensor(self, sigma: torch.Tensor, K: torch.Tensor, L: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_sq = torch.trace(sigma)
        u_sq = torch.trace(K @ sigma @ K.T)
        w_sq = torch.trace(L @ sigma @ L.T)
        rot = torch.trace(sigma @ K.T @ self.H2 @ L)
        sym = torch.trace(sigma @ K.T @ self.S2 @ L)
        pure = -0.5 * self.cfg.q_state * x_sq - 0.5 * self.cfg.a_u * u_sq
        aligned = pure + 0.5 * self.cfg.a_w * w_sq + self.cfg.beta_rot * rot + self.cfg.beta_sym * sym
        return {
            "x_sq": x_sq,
            "u_sq": u_sq,
            "w_sq": w_sq,
            "rot": rot,
            "sym": sym,
            "pure": pure,
            "aligned": aligned,
        }

    def rollout_returns(self, K: torch.Tensor, L: torch.Tensor, sigma0: torch.Tensor) -> Dict[str, float]:
        sigma = sigma0.clone()
        closed_loop = self.closed_loop(K, L)
        rho = safe_spectral_radius(closed_loop)
        aligned_total = torch.zeros((), dtype=DTYPE)
        pure_total = torch.zeros((), dtype=DTYPE)
        weight_sum = 0.0
        state_norm_mean = 0.0
        action_norm_mean = 0.0
        state_norm_max = 0.0
        for t in range(self.cfg.horizon):
            weight = self.cfg.gamma ** t
            terms = self._reward_terms_tensor(sigma, K, L)
            aligned_total = aligned_total + weight * terms["aligned"]
            pure_total = pure_total + weight * terms["pure"]
            x_sq = terms["x_sq"]
            u_sq = terms["u_sq"]
            state_norm = float(torch.sqrt(torch.clamp(x_sq, min=0.0)))
            action_norm = float(torch.sqrt(torch.clamp(u_sq, min=0.0)))
            weight_sum += weight
            state_norm_mean += weight * state_norm
            action_norm_mean += weight * action_norm
            state_norm_max = max(state_norm_max, state_norm)
            sigma = closed_loop @ sigma @ closed_loop.T
        return {
            "aligned_return": float(aligned_total),
            "pure_return": float(pure_total),
            "state_norm_mean": float(state_norm_mean / max(weight_sum, EPS)),
            "state_norm_max": float(state_norm_max),
            "action_norm_mean": float(action_norm_mean / max(weight_sum, EPS)),
            "spectral_radius": float(rho),
            "finite": bool(np.isfinite(float(aligned_total)) and np.isfinite(float(pure_total)) and np.isfinite(state_norm_max) and np.isfinite(rho)),
        }

    def J_align(self, flat: torch.Tensor) -> torch.Tensor:
        K, L = self.split_flat(flat)
        sigma = self.train_sigma0
        closed_loop = self.closed_loop(K, L)
        total = torch.zeros((), dtype=DTYPE)
        for t in range(self.cfg.horizon):
            terms = self._reward_terms_tensor(sigma, K, L)
            total = total + (self.cfg.gamma ** t) * terms["aligned"]
            sigma = closed_loop @ sigma @ closed_loop.T
        return total

    def field_tensor(self, flat: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        z = flat if flat.requires_grad else flat.detach().clone().requires_grad_(True)
        jval = self.J_align(z)
        grad = torch.autograd.grad(jval, z, create_graph=create_graph)[0]
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

    def local_gap_terms(self, flat: torch.Tensor) -> Dict[str, float]:
        dim = self.cfg.action_dim * self.cfg.state_dim
        K0 = flat[:dim].detach().reshape(self.cfg.action_dim, self.cfg.state_dim)
        L0 = flat[dim:].detach().reshape(self.cfg.disturbance_dim, self.cfg.state_dim)
        j0 = float(self.J_align(flat.detach()))

        Kbar = K0.clone()
        for _ in range(self.cfg.gap_inner_steps):
            Kreq = Kbar.detach().clone().requires_grad_(True)
            prox = 0.5 / self.cfg.tau * torch.sum((Kreq - K0) ** 2)
            obj = self.J_align(self.join_flat(Kreq, L0)) - prox
            grad = torch.autograd.grad(obj, Kreq)[0]
            Knext = Kbar + self.cfg.gap_inner_lr * grad
            Kbar, _, _ = self.project_local(Knext, K0, self.cfg.gap_local_radius, self.cfg.K_budget)
        k_term = float(self.J_align(self.join_flat(Kbar, L0)) - 0.5 / self.cfg.tau * torch.sum((Kbar - K0) ** 2))
        gap_k = max(0.0, k_term - j0)

        Lbar = L0.clone()
        for _ in range(self.cfg.gap_inner_steps):
            Lreq = Lbar.detach().clone().requires_grad_(True)
            prox = 0.5 / self.cfg.tau * torch.sum((Lreq - L0) ** 2)
            obj = self.J_align(self.join_flat(K0, Lreq)) + prox
            grad = torch.autograd.grad(obj, Lreq)[0]
            Lnext = Lbar - self.cfg.gap_inner_lr * grad
            Lbar, _, _ = self.project_local(Lnext, L0, self.cfg.gap_local_radius, self.cfg.L_budget)
        l_term = float(self.J_align(self.join_flat(K0, Lbar)) + 0.5 / self.cfg.tau * torch.sum((Lbar - L0) ** 2))
        gap_l = max(0.0, j0 - l_term)
        return {
            "P_tau_align": gap_k + gap_l,
            "P_tau_K_gap": gap_k,
            "P_tau_L_gap": gap_l,
        }

    def local_exploitability(self, flat: torch.Tensor) -> Dict[str, float]:
        dim = self.cfg.action_dim * self.cfg.state_dim
        K0 = flat[:dim].detach().reshape(self.cfg.action_dim, self.cfg.state_dim)
        L0 = flat[dim:].detach().reshape(self.cfg.disturbance_dim, self.cfg.state_dim)
        j0 = float(self.J_align(flat.detach()))

        Kbar = K0.clone()
        for _ in range(self.cfg.gap_inner_steps):
            Kreq = Kbar.detach().clone().requires_grad_(True)
            grad = torch.autograd.grad(self.J_align(self.join_flat(Kreq, L0)), Kreq)[0]
            Knext = Kbar + self.cfg.gap_inner_lr * grad
            Kbar, _, _ = self.project_local(Knext, K0, self.cfg.gap_local_radius, self.cfg.K_budget)
        exploit_k = max(0.0, float(self.J_align(self.join_flat(Kbar, L0))) - j0)

        Lbar = L0.clone()
        for _ in range(self.cfg.gap_inner_steps):
            Lreq = Lbar.detach().clone().requires_grad_(True)
            grad = torch.autograd.grad(self.J_align(self.join_flat(K0, Lreq)), Lreq)[0]
            Lnext = Lbar - self.cfg.gap_inner_lr * grad
            Lbar, _, _ = self.project_local(Lnext, L0, self.cfg.gap_local_radius, self.cfg.L_budget)
        exploit_l = max(0.0, j0 - float(self.J_align(self.join_flat(K0, Lbar))))
        return {"approximate_exploitability": exploit_k + exploit_l}

    def robust_br(self, K: torch.Tensor, L_init: torch.Tensor) -> Dict[str, float]:
        start = L_init.detach().clone()
        current_ref = start.clone()
        last = None
        for lr in [self.cfg.br_inner_lr, 0.01] if self.cfg.br_inner_lr != 0.01 else [self.cfg.br_inner_lr]:
            current = start.clone()
            for _ in range(self.cfg.br_inner_steps):
                Lreq = current.detach().clone().requires_grad_(True)
                value = self._aligned_eval_tensor(K.detach(), Lreq, self.eval_sigma0)
                grad = torch.autograd.grad(value, Lreq)[0]
                Lnext = current - lr * grad
                Lnext, _, _ = self.project_local(Lnext, current_ref, self.cfg.local_br_radius, self.cfg.L_budget)
                current = Lnext.detach()
            aligned_eval = self.rollout_returns(K.detach(), current, self.eval_sigma0)
            stable = aligned_eval["finite"] and aligned_eval["state_norm_max"] < 1e6 and aligned_eval["spectral_radius"] < 2.0
            last = {
                "robust_br_aligned_return": aligned_eval["aligned_return"],
                "robust_br_pure_return": aligned_eval["pure_return"],
                "L_br_norm": float(torch.linalg.norm(current)),
                "L_br_distance_from_current": float(torch.linalg.norm(current - L_init.detach())),
                "closed_loop_spectral_radius_robust_br": aligned_eval["spectral_radius"],
                "robust_br_state_norm_mean": aligned_eval["state_norm_mean"],
                "robust_br_state_norm_max": aligned_eval["state_norm_max"],
                "valid_robust_br_eval": float(stable),
            }
            if stable:
                return last
        return last

    def _aligned_eval_tensor(self, K: torch.Tensor, L: torch.Tensor, sigma0: torch.Tensor) -> torch.Tensor:
        sigma = sigma0
        closed_loop = self.closed_loop(K, L)
        total = torch.zeros((), dtype=DTYPE)
        for t in range(self.cfg.horizon):
            total = total + (self.cfg.gamma ** t) * self._reward_terms_tensor(sigma, K, L)["aligned"]
            sigma = closed_loop @ sigma @ closed_loop.T
        return total

    def metrics(self, flat: torch.Tensor, field_energy0: float, p_tau0: float) -> Dict[str, float]:
        field = self.field_tensor(flat.detach(), create_graph=False).detach()
        field_energy = 0.5 * float(torch.dot(field, field))
        gaps = self.local_gap_terms(flat)
        exploit = self.local_exploitability(flat)
        K, L = self.split_flat(flat)
        clean = self.rollout_returns(K, torch.zeros_like(L), self.eval_sigma0)
        current = self.rollout_returns(K, L, self.eval_sigma0)
        robust = self.robust_br(K, L)
        field_term = field_energy / (field_energy0 + EPS)
        normalized_gap = gaps["P_tau_align"] / (p_tau0 + EPS)
        return {
            "V_align": self.cfg.lambda_F * field_term + self.cfg.lambda_P * normalized_gap,
            "field_term": field_term,
            "field_norm": math.sqrt(max(2.0 * field_energy, 0.0)),
            "P_tau_align": gaps["P_tau_align"],
            "P_tau_K_gap": gaps["P_tau_K_gap"],
            "P_tau_L_gap": gaps["P_tau_L_gap"],
            "normalized_P_tau_align": normalized_gap,
            "approximate_exploitability": exploit["approximate_exploitability"],
            "clean_aligned_return": clean["aligned_return"],
            "current_adv_aligned_return": current["aligned_return"],
            "robust_br_aligned_return": robust["robust_br_aligned_return"],
            "aligned_robust_degradation": clean["aligned_return"] - robust["robust_br_aligned_return"],
            "clean_pure_return": clean["pure_return"],
            "current_adv_pure_return": current["pure_return"],
            "robust_br_pure_return": robust["robust_br_pure_return"],
            "pure_robust_degradation": clean["pure_return"] - robust["robust_br_pure_return"],
            "K_norm": float(torch.linalg.norm(K)),
            "L_norm": float(torch.linalg.norm(L)),
            "clean_state_norm_mean": clean["state_norm_mean"],
            "clean_state_norm_max": clean["state_norm_max"],
            "current_adv_state_norm_mean": current["state_norm_mean"],
            "current_adv_state_norm_max": current["state_norm_max"],
            "robust_br_state_norm_mean": robust["robust_br_state_norm_mean"],
            "robust_br_state_norm_max": robust["robust_br_state_norm_max"],
            "clean_action_norm_mean": clean["action_norm_mean"],
            "current_adv_action_norm_mean": current["action_norm_mean"],
            "closed_loop_spectral_radius_clean": clean["spectral_radius"],
            "closed_loop_spectral_radius_current_adv": current["spectral_radius"],
            "closed_loop_spectral_radius_robust_br": robust["closed_loop_spectral_radius_robust_br"],
            "L_br_norm": robust["L_br_norm"],
            "L_br_distance_from_current": robust["L_br_distance_from_current"],
            "valid_clean_eval": float(clean["finite"]),
            "valid_current_adv_eval": float(current["finite"]),
            "valid_robust_br_eval": robust["valid_robust_br_eval"],
        }


def initial_terms(benchmark: AlignedRotLQBenchmark) -> Tuple[float, float]:
    field0 = benchmark.field_tensor(benchmark.flat0.detach(), create_graph=False).detach()
    field_energy0 = 0.5 * float(torch.dot(field0, field0))
    p_tau0 = benchmark.local_gap_terms(benchmark.flat0)["P_tau_align"]
    return field_energy0, p_tau0


def geometry_row(cfg: AlignedRotLQConfig) -> Dict[str, object]:
    bench = AlignedRotLQBenchmark(cfg)
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
    eigvals = torch.linalg.eigvals(jf).detach().cpu().numpy()
    num_complex = int(np.sum(np.abs(np.imag(eigvals)) > 1e-9))
    max_imag = float(np.max(np.abs(np.imag(eigvals)))) if eigvals.size else 0.0
    return {
        "beta_rot": cfg.beta_rot,
        "beta_sym": cfg.beta_sym,
        "alpha_dyn": cfg.alpha_dyn,
        "a_w": cfg.a_w,
        "L_budget": cfg.L_budget,
        "rotation_ratio_proxy": float(torch.linalg.norm(skew) / (torch.linalg.norm(sym) + EPS)),
        "cross_player_coupling_proxy": float(cross),
        "cross_to_same_ratio": float(cross / (same + EPS)),
        "field_norm0": float(torch.linalg.norm(field0)),
        "G_norm0": float(torch.linalg.norm(g0)),
        "G_over_F0": float(torch.linalg.norm(g0) / (torch.linalg.norm(field0) + EPS)),
        "cos_FG0": cos_fg,
        "non_collinearity0": float(math.sqrt(max(0.0, 1.0 - cos_fg**2))),
        "num_complex_eigs": num_complex,
        "max_imag_eig_abs": max_imag,
        "geometry_gate_pass": float(
            num_complex > 0
            and float(math.sqrt(max(0.0, 1.0 - cos_fg**2))) >= 0.2
            and float(torch.linalg.norm(skew) / (torch.linalg.norm(sym) + EPS)) >= 1.0
            and float(cross / (same + EPS)) > 0.05
            and float(torch.linalg.norm(g0) / (torch.linalg.norm(field0) + EPS)) > 0.05
        ),
    }


def step_method(benchmark: AlignedRotLQBenchmark, method: str, flat: torch.Tensor, lr: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    if method == "SGD":
        cand = flat - lr * benchmark.field_tensor(flat.detach(), create_graph=False).detach()
        return benchmark.project_flat(cand)
    if method == "EGM":
        f0 = benchmark.field_tensor(flat.detach(), create_graph=False).detach()
        z_half, _ = benchmark.project_flat(flat - lr * f0)
        f_half = benchmark.field_tensor(z_half.detach(), create_graph=False).detach()
        cand = flat - lr * f_half
        return benchmark.project_flat(cand)
    if method == "PPM":
        z_inner = flat.clone()
        active_k = 0.0
        active_l = 0.0
        proj = {"projection_active_K": 0.0, "projection_active_L": 0.0, "K_norm": 0.0, "L_norm": 0.0}
        for _ in range(benchmark.cfg.ppm_inner_steps):
            f_inner = benchmark.field_tensor(z_inner.detach(), create_graph=False).detach()
            z_inner, proj = benchmark.project_flat(flat - lr * f_inner)
            active_k += proj["projection_active_K"]
            active_l += proj["projection_active_L"]
        proj["projection_active_K"] = active_k / benchmark.cfg.ppm_inner_steps
        proj["projection_active_L"] = active_l / benchmark.cfg.ppm_inner_steps
        return z_inner, proj
    raise ValueError(f"unknown method {method}")


def run_method(
    benchmark: AlignedRotLQBenchmark,
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
        row = {"method": method, "iteration": it, "lr": lr, **metrics, **proj}
        row["nan_flag"] = float(not np.isfinite(metrics["V_align"]) or not np.isfinite(metrics["field_norm"]))
        row["state_explosion"] = float(
            metrics["clean_state_norm_max"] > 1e6
            or metrics["current_adv_state_norm_max"] > 1e6
            or metrics["robust_br_state_norm_max"] > 1e6
        )
        row["valid_flag"] = float(
            row["nan_flag"] < 0.5
            and row["state_explosion"] < 0.5
            and metrics["valid_clean_eval"] > 0.5
            and metrics["valid_current_adv_eval"] > 0.5
            and metrics["valid_robust_br_eval"] > 0.5
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    final = df.iloc[-1].to_dict()
    summary = {
        "method": method,
        "lr": lr,
        "V_align_AUC": auc_from_series(df["V_align"]),
        "P_tau_align_AUC": auc_from_series(df["P_tau_align"]),
        "field_norm_AUC": auc_from_series(df["field_norm"]),
        "exploitability_AUC": auc_from_series(df["approximate_exploitability"]),
        "robust_br_aligned_return_AUC": auc_from_series(df["robust_br_aligned_return"]),
        "aligned_robust_degradation_AUC": auc_from_series(df["aligned_robust_degradation"]),
        "final_robust_br_aligned_return": float(final["robust_br_aligned_return"]),
        "final_aligned_robust_degradation": float(final["aligned_robust_degradation"]),
        "final_clean_aligned_return": float(final["clean_aligned_return"]),
        "final_current_adv_aligned_return": float(final["current_adv_aligned_return"]),
        "final_clean_pure_return": float(final["clean_pure_return"]),
        "final_current_adv_pure_return": float(final["current_adv_pure_return"]),
        "final_robust_br_pure_return": float(final["robust_br_pure_return"]),
        "projection_active_frac": float(0.5 * (df["projection_active_K"].mean() + df["projection_active_L"].mean())),
        "valid_flag": float(df["valid_flag"].min()),
        "nan_flag": float(df["nan_flag"].max()),
        "state_explosion": float(df["state_explosion"].max()),
    }
    return df, summary, traj


def proposed_step(
    benchmark: AlignedRotLQBenchmark,
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
        cand, _ = benchmark.project_flat(flat - float(beta_t) * Fk + float(gamma_t) * Gk)
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
    accepted_type = "qpg" if method == "proposed_QP_G" else "nog"
    if (not np.isfinite(after["V_align"])) or (after["V_align"] > before["V_align"] + 1e-10):
        egm_cand, _ = step_method(benchmark, "EGM", flat, fallback_lr)
        egm_after = benchmark.metrics(egm_cand, field_energy0, p_tau0)
        if np.isfinite(egm_after["V_align"]) and egm_after["V_align"] <= after["V_align"]:
            cand = egm_cand
            after = egm_after
            fallback = True
            accepted_type = "egm"
    g_ratio = float(torch.linalg.norm(gamma_raw * Gk) / (torch.linalg.norm(beta_raw * Fk) + EPS)) if abs(beta_raw) > EPS else 0.0
    cos_fg = float(torch.dot(Fk, Gk) / (torch.linalg.norm(Fk) * torch.linalg.norm(Gk) + EPS))
    return cand, {
        "beta": float(beta_raw),
        "gamma": float(gamma_raw),
        "gamma_active": float(abs(gamma_raw) > 1e-14),
        "fallback_to_egm": float(fallback),
        "accepted_step_type": accepted_type,
        "V_before": float(before["V_align"]),
        "V_predicted_after": v_pred,
        "V_actual_after": float(after["V_align"]),
        "P_tau_before": float(before["P_tau_align"]),
        "P_tau_after": float(after["P_tau_align"]),
        "field_norm_before": float(before["field_norm"]),
        "field_norm_after": float(after["field_norm"]),
        "exploitability_before": float(before["approximate_exploitability"]),
        "exploitability_after": float(after["approximate_exploitability"]),
        "update_norm": float(torch.linalg.norm(cand - flat)),
        "raw_update_norm": float(raw_update_norm),
        "trust_scaled_update_norm": float(scaled_update_norm),
        "trust_radius_active": float(trust_active),
        "projection_active_K": proj["projection_active_K"],
        "projection_active_L": proj["projection_active_L"],
        "F_norm": float(torch.linalg.norm(Fk)),
        "G_norm": float(torch.linalg.norm(Gk)),
        "cos_FG": cos_fg,
        "non_collinearity": float(math.sqrt(max(0.0, 1.0 - cos_fg**2))),
        "G_contribution_ratio": g_ratio,
        "fit_cond": safe_float(fit_info.get("cond", float("nan"))),
        "fit_indef": safe_float(fit_info.get("indef", float("nan"))),
    }


def run_proposed_method(
    benchmark: AlignedRotLQBenchmark,
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
        curves.append({"method": method, "iteration": it, **metrics})
        diags.append({"method": method, "iteration": it, **info})
    curve_df = pd.DataFrame(curves)
    diag_df = pd.DataFrame(diags)
    final = curve_df.iloc[-1].to_dict()
    summary = {
        "method": method,
        "V_align_AUC": auc_from_series(curve_df["V_align"]),
        "P_tau_align_AUC": auc_from_series(curve_df["P_tau_align"]),
        "field_norm_AUC": auc_from_series(curve_df["field_norm"]),
        "exploitability_AUC": auc_from_series(curve_df["approximate_exploitability"]),
        "robust_br_aligned_return_AUC": auc_from_series(curve_df["robust_br_aligned_return"]),
        "aligned_robust_degradation_AUC": auc_from_series(curve_df["aligned_robust_degradation"]),
        "final_robust_br_aligned_return": float(final["robust_br_aligned_return"]),
        "final_aligned_robust_degradation": float(final["aligned_robust_degradation"]),
        "final_clean_aligned_return": float(final["clean_aligned_return"]),
        "final_current_adv_aligned_return": float(final["current_adv_aligned_return"]),
        "final_clean_pure_return": float(final["clean_pure_return"]),
        "final_current_adv_pure_return": float(final["current_adv_pure_return"]),
        "final_robust_br_pure_return": float(final["robust_br_pure_return"]),
        "fallback_to_egm_frac": float(diag_df["fallback_to_egm"].mean()),
        "gamma_active_frac": float(diag_df["gamma_active"].mean()),
        "G_contribution_ratio": float(diag_df["G_contribution_ratio"].mean()),
        "projection_active_frac": float(0.5 * (diag_df["projection_active_K"].mean() + diag_df["projection_active_L"].mean())),
    }
    return curve_df, diag_df, summary, traj


def proposed_radius_preflight(
    benchmark: AlignedRotLQBenchmark,
    field_energy0: float,
    p_tau0: float,
    fallback_lr: float,
) -> Tuple[pd.DataFrame, float]:
    rows: List[Dict[str, object]] = []
    selected = float("nan")
    for radius in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
        probe_radius = min(radius, 1e-2) * 0.5
        curve_df, _, summary, _ = run_proposed_method(benchmark, "proposed_QP_G", 50, field_energy0, p_tau0, radius, probe_radius, fallback_lr)
        row = {
            "update_radius": radius,
            **summary,
            "time_to_V_align_1e-3": first_below(curve_df["V_align"], 1e-3),
            "time_to_P_tau_align_1e-3": first_below(curve_df["normalized_P_tau_align"], 1e-3),
            "valid_flag": float(summary["fallback_to_egm_frac"] < 0.2 and summary["projection_active_frac"] <= 0.5),
        }
        rows.append(row)
        if np.isnan(selected) and row["valid_flag"] > 0.5:
            selected = radius
    df = pd.DataFrame(rows)
    if np.isnan(selected):
        selected = float(df.sort_values(["fallback_to_egm_frac", "V_align_AUC", "update_radius"]).iloc[0]["update_radius"])
    return df, selected


def same_start_comparison(
    benchmark: AlignedRotLQBenchmark,
    qpg_traj: Sequence[torch.Tensor],
    field_energy0: float,
    p_tau0: float,
    lr: float,
    update_radius: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    probe_radius = min(update_radius, 1e-2) * 0.5
    for checkpoint in [0, 5, 10, 25, 50]:
        z = qpg_traj[checkpoint]
        before = benchmark.metrics(z, field_energy0, p_tau0)
        qpg_cand, qpg_info = proposed_step(benchmark, "proposed_QP_G", z, field_energy0, p_tau0, update_radius, probe_radius, lr)
        candidates = {
            "zero": z.clone(),
            "SGD": step_method(benchmark, "SGD", z, lr)[0],
            "EGM": step_method(benchmark, "EGM", z, lr)[0],
            "PPM": step_method(benchmark, "PPM", z, lr)[0],
            "proposed_noG": proposed_step(benchmark, "proposed_noG", z, field_energy0, p_tau0, update_radius, probe_radius, lr)[0],
            "proposed_QP_G": qpg_cand,
        }
        qpg_delta = qpg_cand - z
        deltas = {name: cand - z for name, cand in candidates.items() if name != "zero"}
        for name, cand in candidates.items():
            after = benchmark.metrics(cand, field_energy0, p_tau0)
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "candidate": name,
                    "V_align_before": float(before["V_align"]),
                    "V_align_after": float(after["V_align"]),
                    "actual_delta_V_align": float(after["V_align"] - before["V_align"]),
                    "P_tau_align_after": float(after["P_tau_align"]),
                    "field_norm_after": float(after["field_norm"]),
                    "exploitability_after": float(after["approximate_exploitability"]),
                    "current_adv_aligned_return_after": float(after["current_adv_aligned_return"]),
                    "robust_br_aligned_return_after": float(after["robust_br_aligned_return"]),
                    "aligned_robust_degradation_after": float(after["aligned_robust_degradation"]),
                    "current_adv_pure_return_after": float(after["current_adv_pure_return"]),
                    "robust_br_pure_return_after": float(after["robust_br_pure_return"]),
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

    plot_metric("aligned_rot_lq_final_V_align.png", "V_align", "Aligned Composite V", "V_align")
    plot_metric("aligned_rot_lq_final_P_tau_align.png", "normalized_P_tau_align", "Aligned Normalized P_tau", "normalized_P_tau")
    plot_metric("aligned_rot_lq_final_field_norm.png", "field_norm", "Aligned Field Norm", "||F_align||")
    plot_metric("aligned_rot_lq_final_exploitability.png", "approximate_exploitability", "Aligned Exploitability", "exploitability")
    plot_metric("aligned_rot_lq_final_clean_aligned_return.png", "clean_aligned_return", "Clean Aligned Return", "return", logy=False)
    plot_metric("aligned_rot_lq_final_current_adv_aligned_return.png", "current_adv_aligned_return", "Current-Adversary Aligned Return", "return", logy=False)
    plot_metric("aligned_rot_lq_final_robust_br_aligned_return.png", "robust_br_aligned_return", "Robust-BR Aligned Return", "return", logy=False)
    plot_metric("aligned_rot_lq_final_aligned_robust_degradation.png", "aligned_robust_degradation", "Aligned Robust Degradation", "degradation", logy=False)
    plot_metric("aligned_rot_lq_final_clean_pure_return.png", "clean_pure_return", "Clean Pure Return", "return", logy=False)
    plot_metric("aligned_rot_lq_final_current_adv_pure_return.png", "current_adv_pure_return", "Current-Adversary Pure Return", "return", logy=False)
    plot_metric("aligned_rot_lq_final_robust_br_pure_return.png", "robust_br_pure_return", "Robust-BR Pure Return", "return", logy=False)
    plot_metric("aligned_rot_lq_final_pure_robust_degradation.png", "pure_robust_degradation", "Pure Robust Degradation", "degradation", logy=False)

    fig, axes = plt.subplots(4, 3, figsize=(16, 18))
    panels = [
        ("V_align", "V_align", True),
        ("normalized_P_tau_align", "P_tau_align", True),
        ("field_norm", "field_norm", True),
        ("approximate_exploitability", "exploitability", True),
        ("clean_aligned_return", "clean aligned", False),
        ("current_adv_aligned_return", "current adv aligned", False),
        ("robust_br_aligned_return", "robust BR aligned", False),
        ("aligned_robust_degradation", "aligned degradation", False),
        ("clean_pure_return", "clean pure", False),
        ("current_adv_pure_return", "current adv pure", False),
        ("robust_br_pure_return", "robust BR pure", False),
        ("pure_robust_degradation", "pure degradation", False),
    ]
    for ax, (metric, title, logy) in zip(axes.flat, panels):
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
    fig.savefig(PLOT_ROOT / "aligned_rot_lq_final_all_plots_big.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))
    panels = [
        ("V_align", "V_align", True),
        ("normalized_P_tau_align", "P_tau_align", True),
        ("field_norm", "field_norm", True),
        ("robust_br_aligned_return", "robust BR aligned", False),
        ("aligned_robust_degradation", "aligned degradation", False),
    ]
    for ax, (metric, title, logy) in zip(axes, panels):
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
    fig.savefig(PLOT_ROOT / "aligned_rot_lq_final_paper_main.png", dpi=180)
    plt.close(fig)


def choose_configs() -> List[AlignedRotLQConfig]:
    combos = [
        (0.1, 0.05, 0.01, 2.0),
        (0.3, 0.05, 0.01, 2.0),
        (0.5, 0.05, 0.01, 2.0),
        (0.7, 0.05, 0.01, 2.0),
        (1.0, 0.05, 0.01, 2.0),
        (0.1, 0.1, 0.01, 2.0),
        (0.3, 0.1, 0.01, 2.0),
        (0.5, 0.1, 0.01, 2.0),
        (0.7, 0.1, 0.01, 2.0),
        (1.0, 0.1, 0.01, 2.0),
        (0.3, 0.2, 0.01, 2.0),
        (0.5, 0.2, 0.01, 2.0),
        (0.7, 0.2, 0.01, 2.0),
        (0.7, 0.1, 0.003, 3.0),
        (0.7, 0.1, 0.03, 1.0),
    ]
    return [
        AlignedRotLQConfig(beta_rot=beta_rot, beta_sym=0.1 * beta_rot, alpha_dyn=alpha_dyn, a_w=a_w, L_budget=L_budget)
        for beta_rot, alpha_dyn, a_w, L_budget in combos
    ]


def main() -> None:
    geometry_rows: List[Dict[str, object]] = []
    sgd_rows: List[Dict[str, object]] = []
    baseline_rows: List[Dict[str, object]] = []
    config_rows: List[Dict[str, object]] = []
    best_payload: Dict[str, object] | None = None

    for cfg in choose_configs():
        geo = geometry_row(cfg)
        geometry_rows.append(geo)
        row = {"beta_rot": cfg.beta_rot, "beta_sym": cfg.beta_sym, "alpha_dyn": cfg.alpha_dyn, "a_w": cfg.a_w, "L_budget": cfg.L_budget, **geo}
        if geo["geometry_gate_pass"] < 0.5:
            row["decision"] = "GEOMETRY_FAIL"
            config_rows.append(row)
            continue

        benchmark = AlignedRotLQBenchmark(cfg)
        field_energy0, p_tau0 = initial_terms(benchmark)

        sgd_candidates: List[Dict[str, object]] = []
        for lr in [1e-3, 3e-3, 1e-2, 3e-2]:
            _, summary, _ = run_method(benchmark, "SGD", lr, 300, field_energy0, p_tau0)
            sgd_candidates.append(summary | {"beta_rot": cfg.beta_rot, "alpha_dyn": cfg.alpha_dyn, "a_w": cfg.a_w, "L_budget": cfg.L_budget})
        sgd_df = pd.DataFrame(sgd_candidates)
        sgd_rows.extend(sgd_candidates)
        valid_sgd = sgd_df[
            (sgd_df["valid_flag"] > 0.5)
            & (sgd_df["projection_active_frac"] <= 0.5)
            & (sgd_df["nan_flag"] < 0.5)
            & (sgd_df["state_explosion"] < 0.5)
            & np.isfinite(sgd_df["robust_br_aligned_return_AUC"])
        ].copy()
        if valid_sgd.empty:
            row["decision"] = "SGD_NORMAL_FAIL"
            config_rows.append(row)
            continue
        sgd_best = valid_sgd.sort_values(["V_align_AUC", "aligned_robust_degradation_AUC", "lr"]).iloc[0]
        selected_lr = float(sgd_best["lr"])

        baseline_summaries: List[Dict[str, object]] = []
        curve_frames: List[pd.DataFrame] = []
        traj_map: Dict[str, List[torch.Tensor]] = {}
        for method in ["SGD", "EGM", "PPM"]:
            curve_df, summary, traj = run_method(benchmark, method, selected_lr, 300, field_energy0, p_tau0)
            curve_frames.append(curve_df)
            baseline_summaries.append(summary)
            traj_map[method] = traj
        baseline_df = pd.DataFrame(baseline_summaries)
        sgd_row = baseline_df[baseline_df["method"] == "SGD"].iloc[0]
        egm_row = baseline_df[baseline_df["method"] == "EGM"].iloc[0]
        ppm_row = baseline_df[baseline_df["method"] == "PPM"].iloc[0]
        baseline_pass = (
            float(baseline_df["valid_flag"].min()) > 0.5
            and (
                float(sgd_row["V_align_AUC"]) / max(float(egm_row["V_align_AUC"]), EPS) >= 1.3
                or float(sgd_row["V_align_AUC"]) / max(float(ppm_row["V_align_AUC"]), EPS) >= 1.3
                or float(sgd_row["P_tau_align_AUC"]) / max(float(egm_row["P_tau_align_AUC"]), EPS) >= 1.3
                or float(sgd_row["P_tau_align_AUC"]) / max(float(ppm_row["P_tau_align_AUC"]), EPS) >= 1.3
            )
            and float(egm_row["final_robust_br_aligned_return"]) >= float(sgd_row["final_robust_br_aligned_return"]) - 1e-9
            and float(ppm_row["final_robust_br_aligned_return"]) >= float(sgd_row["final_robust_br_aligned_return"]) - 1e-9
        )
        baseline_rows.append({
            "beta_rot": cfg.beta_rot,
            "alpha_dyn": cfg.alpha_dyn,
            "a_w": cfg.a_w,
            "L_budget": cfg.L_budget,
            "selected_lr": selected_lr,
            "baseline_pass": float(baseline_pass),
            "sgd_V_align_AUC": float(sgd_row["V_align_AUC"]),
            "egm_V_align_AUC": float(egm_row["V_align_AUC"]),
            "ppm_V_align_AUC": float(ppm_row["V_align_AUC"]),
            "sgd_robust": float(sgd_row["final_robust_br_aligned_return"]),
            "egm_robust": float(egm_row["final_robust_br_aligned_return"]),
            "ppm_robust": float(ppm_row["final_robust_br_aligned_return"]),
        })
        if not baseline_pass:
            row["decision"] = "BASELINE_FAIL"
            config_rows.append(row)
            continue

        preflight_df, selected_radius = proposed_radius_preflight(benchmark, field_energy0, p_tau0, selected_lr)
        probe_radius = min(selected_radius, 1e-2) * 0.5
        final_summaries = baseline_summaries.copy()
        diag_frames: List[pd.DataFrame] = []
        all_curves = curve_frames.copy()
        for method in ["proposed_noG", "proposed_QP_G"]:
            curve_df, diag_df, summary, traj = run_proposed_method(benchmark, method, 300, field_energy0, p_tau0, selected_radius, probe_radius, selected_lr)
            final_summaries.append(summary)
            all_curves.append(curve_df)
            diag_frames.append(diag_df)
            traj_map[method] = traj
        summary_df = pd.DataFrame(final_summaries)
        qpg = summary_df[summary_df["method"] == "proposed_QP_G"].iloc[0]
        nog = summary_df[summary_df["method"] == "proposed_noG"].iloc[0]
        egm = summary_df[summary_df["method"] == "EGM"].iloc[0]
        ppm = summary_df[summary_df["method"] == "PPM"].iloc[0]
        sgd = summary_df[summary_df["method"] == "SGD"].iloc[0]
        full_alignment = (
            float(qpg["V_align_AUC"]) < float(nog["V_align_AUC"])
            and float(qpg["P_tau_align_AUC"]) < float(nog["P_tau_align_AUC"])
            and float(qpg["field_norm_AUC"]) < float(nog["field_norm_AUC"])
            and float(qpg["exploitability_AUC"]) < float(nog["exploitability_AUC"])
            and float(qpg["robust_br_aligned_return_AUC"]) >= float(nog["robust_br_aligned_return_AUC"]) - 1e-6
            and float(qpg["final_robust_br_aligned_return"]) >= float(nog["final_robust_br_aligned_return"]) - 1e-6
            and float(qpg["aligned_robust_degradation_AUC"]) <= float(nog["aligned_robust_degradation_AUC"]) + 1e-6
            and float(qpg["final_robust_br_aligned_return"]) >= max(float(sgd["final_robust_br_aligned_return"]), float(egm["final_robust_br_aligned_return"]), float(ppm["final_robust_br_aligned_return"])) - 1e-6
            and float(qpg["fallback_to_egm_frac"]) < 0.2
            and float(qpg["projection_active_frac"]) <= 0.5
        )
        decision = "FULL_ALIGNMENT_SUCCESS" if full_alignment else (
            "OPTIMIZATION_POSITIVE_RARL_SUPPORTIVE"
            if (
                float(qpg["V_align_AUC"]) < float(nog["V_align_AUC"])
                and float(qpg["final_robust_br_aligned_return"]) >= max(float(sgd["final_robust_br_aligned_return"]), float(egm["final_robust_br_aligned_return"]), float(ppm["final_robust_br_aligned_return"])) - 1e-6
            )
            else "OPTIMIZATION_ONLY"
        )
        row["decision"] = decision
        row["selected_lr"] = selected_lr
        row["selected_radius"] = selected_radius
        row["qpg_V_align_AUC"] = float(qpg["V_align_AUC"])
        row["qpg_robust_final"] = float(qpg["final_robust_br_aligned_return"])
        row["nog_robust_final"] = float(nog["final_robust_br_aligned_return"])
        row["qpg_fallback_frac"] = float(qpg["fallback_to_egm_frac"])
        config_rows.append(row)
        payload = {
            "cfg": cfg,
            "benchmark": benchmark,
            "field_energy0": field_energy0,
            "p_tau0": p_tau0,
            "selected_lr": selected_lr,
            "selected_radius": selected_radius,
            "preflight_df": preflight_df,
            "summary_df": summary_df,
            "curves_df": pd.concat(all_curves, ignore_index=True),
            "diag_df": pd.concat(diag_frames, ignore_index=True),
            "traj_map": traj_map,
            "decision": decision,
        }
        if best_payload is None:
            best_payload = payload
        else:
            prev_qpg = best_payload["summary_df"][best_payload["summary_df"]["method"] == "proposed_QP_G"].iloc[0]
            if decision == "FULL_ALIGNMENT_SUCCESS" and best_payload["decision"] != "FULL_ALIGNMENT_SUCCESS":
                best_payload = payload
            elif decision == best_payload["decision"] and float(qpg["V_align_AUC"]) < float(prev_qpg["V_align_AUC"]):
                best_payload = payload
        if full_alignment:
            break

    geometry_df = pd.DataFrame(geometry_rows)
    sgd_df = pd.DataFrame(sgd_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    config_df = pd.DataFrame(config_rows)
    geometry_df.to_csv(RESULT_ROOT / "aligned_rot_lq_geometry_audit.csv", index=False)
    sgd_df.to_csv(RESULT_ROOT / "aligned_rot_lq_sgd_gate.csv", index=False)
    baseline_df.to_csv(RESULT_ROOT / "aligned_rot_lq_baseline_gate.csv", index=False)
    config_df.to_csv(RESULT_ROOT / "aligned_rot_lq_config_sweep.csv", index=False)
    write_md(RESULT_ROOT / "aligned_rot_lq_geometry_audit.md", "# Aligned Rotational LQ Geometry Audit\n\n" + df_text(geometry_df))
    write_md(RESULT_ROOT / "aligned_rot_lq_sgd_gate_report.md", "# Aligned Rotational LQ SGD Gate\n\n" + df_text(sgd_df))
    write_md(RESULT_ROOT / "aligned_rot_lq_baseline_gate_report.md", "# Aligned Rotational LQ Baseline Gate\n\n" + df_text(baseline_df))
    write_md(RESULT_ROOT / "aligned_rot_lq_config_sweep_report.md", "# Aligned Rotational LQ Config Sweep\n\n" + df_text(config_df))

    if best_payload is None:
        write_md(
            RESULT_ROOT / "aligned_rot_lq_final_decision.md",
            "\n".join(
                [
                    "# Aligned Rotational LQ Final Decision",
                    "",
                    "GEOMETRY_FAIL" if geometry_df["geometry_gate_pass"].max() < 0.5 else "ENVIRONMENT_FAIL",
                    "",
                    "No tested configuration produced a clean full pipeline through geometry, baseline, and proposed stages.",
                    "",
                    "This benchmark cannot yet replace MixedLinearActorRotLQ-v0 as the paper LQ result.",
                ]
            ),
        )
        return

    cfg = best_payload["cfg"]
    summary_df = best_payload["summary_df"]
    curves_df = best_payload["curves_df"]
    diag_df = best_payload["diag_df"]
    preflight_df = best_payload["preflight_df"]
    field_energy0 = best_payload["field_energy0"]
    p_tau0 = best_payload["p_tau0"]
    benchmark = best_payload["benchmark"]
    selected_lr = best_payload["selected_lr"]
    selected_radius = best_payload["selected_radius"]
    traj_map = best_payload["traj_map"]

    preflight_df.to_csv(RESULT_ROOT / "aligned_rot_lq_proposed_radius_preflight.csv", index=False)
    write_md(RESULT_ROOT / "aligned_rot_lq_proposed_radius_preflight_report.md", "# Aligned Rotational LQ Proposed Radius Preflight\n\n" + df_text(preflight_df))
    summary_df.to_csv(RESULT_ROOT / "aligned_rot_lq_final_summary.csv", index=False)
    curves_df.to_csv(RESULT_ROOT / "aligned_rot_lq_final_curves.csv", index=False)
    diag_df.to_csv(RESULT_ROOT / "aligned_rot_lq_final_diagnostics.csv", index=False)
    plot_final(curves_df)
    same_df = same_start_comparison(benchmark, traj_map["proposed_QP_G"], field_energy0, p_tau0, selected_lr, selected_radius)
    same_df.to_csv(RESULT_ROOT / "aligned_rot_lq_same_start_comparison.csv", index=False)
    write_md(RESULT_ROOT / "aligned_rot_lq_same_start_comparison.md", "# Aligned Rotational LQ Same-Start Comparison\n\n" + df_text(same_df))

    qpg = summary_df[summary_df["method"] == "proposed_QP_G"].iloc[0]
    nog = summary_df[summary_df["method"] == "proposed_noG"].iloc[0]
    egm = summary_df[summary_df["method"] == "EGM"].iloc[0]
    ppm = summary_df[summary_df["method"] == "PPM"].iloc[0]
    sgd = summary_df[summary_df["method"] == "SGD"].iloc[0]
    if (
        float(qpg["V_align_AUC"]) < float(nog["V_align_AUC"])
        and float(qpg["P_tau_align_AUC"]) < float(nog["P_tau_align_AUC"])
        and float(qpg["field_norm_AUC"]) < float(nog["field_norm_AUC"])
        and float(qpg["exploitability_AUC"]) < float(nog["exploitability_AUC"])
        and float(qpg["robust_br_aligned_return_AUC"]) >= float(nog["robust_br_aligned_return_AUC"]) - 1e-6
        and float(qpg["final_robust_br_aligned_return"]) >= float(nog["final_robust_br_aligned_return"]) - 1e-6
        and float(qpg["aligned_robust_degradation_AUC"]) <= float(nog["aligned_robust_degradation_AUC"]) + 1e-6
    ):
        decision = "FULL_ALIGNMENT_SUCCESS"
    elif (
        float(qpg["V_align_AUC"]) < float(nog["V_align_AUC"])
        and float(qpg["final_robust_br_aligned_return"]) >= max(float(sgd["final_robust_br_aligned_return"]), float(egm["final_robust_br_aligned_return"]), float(ppm["final_robust_br_aligned_return"])) - 1e-6
    ):
        decision = "OPTIMIZATION_POSITIVE_RARL_SUPPORTIVE"
    else:
        decision = "OPTIMIZATION_ONLY"

    write_md(
        RESULT_ROOT / "aligned_rot_lq_final_report.md",
        "\n".join(
            [
                "# Aligned Rotational LQ Final Report",
                "",
                f"- selected config: `beta_rot={cfg.beta_rot}, beta_sym={cfg.beta_sym}, alpha_dyn={cfg.alpha_dyn}, a_w={cfg.a_w}, L_budget={cfg.L_budget}`",
                f"- selected lr: `{selected_lr}`",
                f"- selected update_radius: `{selected_radius}`",
                "",
                df_text(summary_df),
            ]
        ),
    )
    write_md(
        RESULT_ROOT / "aligned_rot_lq_final_decision.md",
        "\n".join(
            [
                "# Aligned Rotational LQ Final Decision",
                "",
                decision,
                "",
                (
                    "This benchmark can replace MixedLinearActorRotLQ-v0 as the paper LQ result."
                    if decision in {"FULL_ALIGNMENT_SUCCESS", "OPTIMIZATION_POSITIVE_RARL_SUPPORTIVE"}
                    else "This benchmark should not replace MixedLinearActorRotLQ-v0 as the paper LQ result."
                ),
            ]
        ),
    )


if __name__ == "__main__":
    main()
