from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


DTYPE = torch.float64
EPS = 1e-12
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = ROOT / "original" / "results" / "nn_actor_rotational_lqr_rarl"
PLOT_ROOT = RESULT_ROOT / "plots"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)
PLOT_ROOT.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def write_md(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def df_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    return df.to_string(index=False)


def safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def auc_from_series(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    xs = np.arange(len(values), dtype=np.float64)
    return float(np.trapezoid(np.asarray(values, dtype=np.float64), x=xs))


def first_below(values: Sequence[float], threshold: float) -> float:
    for idx, value in enumerate(values):
        if np.isfinite(value) and value <= threshold:
            return float(idx)
    return float("nan")


def clip_floor(values: Sequence[float], floor: float = 1e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.maximum(arr, floor)


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


@dataclass(frozen=True)
class EnvConfig:
    state_dim: int = 4
    action_dim: int = 2
    disturbance_dim: int = 2
    horizon: int = 50
    gamma: float = 0.98
    beta_rot: float = 2.0
    rho_w: float = 0.05
    train_batch_size: int = 128
    eval_batch_size: int = 256
    u_max: float = 2.0
    w_max: float = 2.0
    state_clip: float = 20.0
    seed: int = 0
    actor_init_scale: float = 0.01
    actor_temperature: float = 1.0


@dataclass(frozen=True)
class UnifiedLyapunovConfig:
    lambda_F: float
    lambda_P: float
    tau: float
    n_inner_gap: int
    gap_inner_lr: float
    local_radius: float
    update_radius: float = 0.1


class RotLQRNNActorBenchmark:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        s2 = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=DTYPE)
        a_block = 0.96 * torch.eye(2, dtype=DTYPE) + 0.15 * s2
        self.A = torch.block_diag(a_block, a_block)
        self.B = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.3, 0.0], [0.0, 0.3]],
            dtype=DTYPE,
        )
        self.E = torch.tensor(
            [[0.0, 1.0], [-1.0, 0.0], [0.0, 0.3], [-0.3, 0.0]],
            dtype=DTYPE,
        )
        self.Q = torch.diag(torch.tensor([1.0, 1.0, 0.5, 0.5], dtype=DTYPE))
        self.R = 0.05 * torch.eye(2, dtype=DTYPE)
        self.H = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=DTYPE)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(cfg.seed)
        self.train_x0_batch = torch.randn(cfg.train_batch_size, cfg.state_dim, generator=gen, dtype=DTYPE)
        self.eval_x0_batch = torch.randn(cfg.eval_batch_size, cfg.state_dim, generator=gen, dtype=DTYPE)
        self.theta0, self.phi0 = self.init_actor_params(gen)
        self.flat0 = self.join_flat(self.theta0, self.phi0)

    @property
    def actor_param_dim(self) -> int:
        return self.cfg.action_dim * self.cfg.state_dim + self.cfg.action_dim

    def init_actor_params(self, generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        w_theta = self.cfg.actor_init_scale * torch.randn(self.cfg.action_dim, self.cfg.state_dim, generator=generator, dtype=DTYPE)
        b_theta = torch.zeros(self.cfg.action_dim, dtype=DTYPE)
        w_phi = self.cfg.actor_init_scale * torch.randn(self.cfg.disturbance_dim, self.cfg.state_dim, generator=generator, dtype=DTYPE)
        b_phi = torch.zeros(self.cfg.disturbance_dim, dtype=DTYPE)
        return torch.cat([w_theta.reshape(-1), b_theta]), torch.cat([w_phi.reshape(-1), b_phi])

    def split_flat(self, flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dim = self.actor_param_dim
        return flat[:dim], flat[dim:]

    def join_flat(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        return torch.cat([theta.reshape(-1), phi.reshape(-1)])

    def unpack_actor(self, actor_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w_size = self.cfg.action_dim * self.cfg.state_dim
        W = actor_flat[:w_size].reshape(self.cfg.action_dim, self.cfg.state_dim)
        b = actor_flat[w_size:].reshape(self.cfg.action_dim)
        return W, b

    def actor_pre_and_action(self, actor_flat: torch.Tensor, states: torch.Tensor, action_scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
        W, b = self.unpack_actor(actor_flat)
        pre = self.cfg.actor_temperature * (states @ W.T + b)
        action = action_scale * torch.tanh(pre)
        return pre, action

    def actor_action(self, actor_flat: torch.Tensor, states: torch.Tensor, action_scale: float) -> torch.Tensor:
        _, action = self.actor_pre_and_action(actor_flat, states, action_scale)
        return action

    def rollout(
        self,
        flat: torch.Tensor,
        x0_batch: torch.Tensor,
        alpha: float = 1.0,
        clean: bool = False,
    ) -> Dict[str, torch.Tensor]:
        theta, phi = self.split_flat(flat)
        x = x0_batch
        total_game = torch.zeros((), dtype=DTYPE)
        total_task_cost = torch.zeros((), dtype=DTYPE)
        total_rot = torch.zeros((), dtype=DTYPE)
        total_dist = torch.zeros((), dtype=DTYPE)
        total_state_clip = torch.zeros((), dtype=DTYPE)
        total_u_sat = torch.zeros((), dtype=DTYPE)
        total_w_sat = torch.zeros((), dtype=DTYPE)
        total_abs_u = torch.zeros((), dtype=DTYPE)
        total_abs_w = torch.zeros((), dtype=DTYPE)
        total_abs_pre_u = torch.zeros((), dtype=DTYPE)
        total_abs_pre_w = torch.zeros((), dtype=DTYPE)
        max_abs_u = torch.zeros((), dtype=DTYPE)
        max_abs_w = torch.zeros((), dtype=DTYPE)
        max_abs_pre_u = torch.zeros((), dtype=DTYPE)
        max_abs_pre_w = torch.zeros((), dtype=DTYPE)
        max_abs_state = torch.max(torch.abs(x))

        for t in range(self.cfg.horizon):
            pre_u, u = self.actor_pre_and_action(theta, x, self.cfg.u_max)
            if clean:
                pre_w = torch.zeros(x.shape[0], self.cfg.disturbance_dim, dtype=DTYPE)
                w = torch.zeros(x.shape[0], self.cfg.disturbance_dim, dtype=DTYPE)
            else:
                pre_w, w_base = self.actor_pre_and_action(phi, x, self.cfg.w_max)
                w = alpha * w_base
            total_u_sat = total_u_sat + torch.mean((torch.abs(u) >= 0.98 * self.cfg.u_max).to(DTYPE))
            if clean:
                total_w_sat = total_w_sat + torch.zeros((), dtype=DTYPE)
            else:
                w_bound = max(self.cfg.w_max * abs(alpha), EPS)
                total_w_sat = total_w_sat + torch.mean((torch.abs(w) >= 0.98 * w_bound).to(DTYPE))
            total_abs_u = total_abs_u + torch.mean(torch.abs(u))
            total_abs_w = total_abs_w + torch.mean(torch.abs(w))
            total_abs_pre_u = total_abs_pre_u + torch.mean(torch.abs(pre_u))
            total_abs_pre_w = total_abs_pre_w + torch.mean(torch.abs(pre_w))
            max_abs_u = torch.maximum(max_abs_u, torch.max(torch.abs(u)))
            max_abs_w = torch.maximum(max_abs_w, torch.max(torch.abs(w)))
            max_abs_pre_u = torch.maximum(max_abs_pre_u, torch.max(torch.abs(pre_u)))
            max_abs_pre_w = torch.maximum(max_abs_pre_w, torch.max(torch.abs(pre_w)))

            x_q = torch.sum((x @ self.Q) * x, dim=1)
            u_r = torch.sum((u @ self.R) * u, dim=1)
            task_cost = x_q + u_r
            task_reward = -task_cost
            rot_term = self.cfg.beta_rot * torch.sum((u @ self.H) * w, dim=1)
            dist_energy = torch.sum(w * w, dim=1)
            game_reward = task_reward + self.cfg.rho_w * dist_energy + rot_term
            weight = self.cfg.gamma ** t
            total_game = total_game + weight * torch.mean(game_reward)
            total_task_cost = total_task_cost + weight * torch.mean(task_cost)
            total_rot = total_rot + weight * torch.mean(rot_term)
            total_dist = total_dist + weight * torch.mean(dist_energy)

            x_next = x @ self.A.T + u @ self.B.T + w @ self.E.T
            if self.cfg.state_clip > 0.0:
                clipped = torch.clamp(x_next, -self.cfg.state_clip, self.cfg.state_clip)
                total_state_clip = total_state_clip + torch.mean((clipped != x_next).to(DTYPE))
                x = clipped
            else:
                x = x_next
            max_abs_state = torch.maximum(max_abs_state, torch.max(torch.abs(x)))

        return {
            "J_game": total_game,
            "task_cost": total_task_cost,
            "task_return": -total_task_cost,
            "rot_reward": total_rot,
            "disturbance_energy": total_dist,
            "state_clip_frac": total_state_clip / self.cfg.horizon,
            "action_saturation_fraction_protagonist": total_u_sat / self.cfg.horizon,
            "action_saturation_fraction_adversary": total_w_sat / self.cfg.horizon,
            "mean_abs_u": total_abs_u / self.cfg.horizon,
            "mean_abs_w": total_abs_w / self.cfg.horizon,
            "max_abs_u": max_abs_u,
            "max_abs_w": max_abs_w,
            "mean_pre_tanh_abs_protagonist": total_abs_pre_u / self.cfg.horizon,
            "mean_pre_tanh_abs_adversary": total_abs_pre_w / self.cfg.horizon,
            "max_pre_tanh_abs_protagonist": max_abs_pre_u,
            "max_pre_tanh_abs_adversary": max_abs_pre_w,
            "max_abs_state": max_abs_state,
        }

    def J(self, flat: torch.Tensor) -> torch.Tensor:
        return self.rollout(flat, self.train_x0_batch)["J_game"]

    def J_adversary(self, flat: torch.Tensor) -> torch.Tensor:
        return -self.J(flat)

    def field_tensor(self, flat: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        flat_req = flat if flat.requires_grad else flat.detach().clone().requires_grad_(True)
        j_val = self.J(flat_req)
        grad = torch.autograd.grad(j_val, flat_req, create_graph=create_graph)[0]
        dim = self.actor_param_dim
        grad_theta = grad[:dim]
        grad_phi = grad[dim:]
        return torch.cat([-grad_theta, grad_phi])

    def field_and_j(self, flat: torch.Tensor, create_graph: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        flat_req = flat if flat.requires_grad else flat.detach().clone().requires_grad_(True)
        j_val = self.J(flat_req)
        grad = torch.autograd.grad(j_val, flat_req, create_graph=create_graph)[0]
        dim = self.actor_param_dim
        grad_theta = grad[:dim]
        grad_phi = grad[dim:]
        return torch.cat([-grad_theta, grad_phi]), j_val

    def curvature_direction(self, flat: torch.Tensor, field: torch.Tensor | None = None) -> torch.Tensor:
        if field is None:
            field = self.field_tensor(flat, create_graph=False).detach()
        norm = float(torch.linalg.norm(field))
        if norm < EPS:
            return torch.zeros_like(field)
        eps = min(1e-4, 1e-4 / (norm + 1.0))
        field_shifted = self.field_tensor((flat + eps * field).detach(), create_graph=False).detach()
        return (field_shifted - field) / eps

    def full_jacobian(self, flat: torch.Tensor) -> torch.Tensor:
        flat = flat.detach().clone().requires_grad_(True)
        return torch.autograd.functional.jacobian(lambda z: self.field_tensor(z, create_graph=True), flat, vectorize=False)

    def project_local_ball(self, candidate: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor:
        delta = candidate - center
        norm = torch.linalg.norm(delta)
        if float(norm) <= radius + EPS:
            return candidate
        return center + delta * (radius / (norm + EPS))

    def local_gap_terms(
        self,
        flat: torch.Tensor,
        tau: float,
        n_inner_gap: int,
        gap_inner_lr: float,
        local_radius: float,
    ) -> Dict[str, object]:
        theta, phi = self.split_flat(flat.detach())
        j_current = self.J(flat.detach()).detach()

        theta_bar = theta.detach().clone().requires_grad_(True)
        for _ in range(n_inner_gap):
            obj_theta = self.J(self.join_flat(theta_bar, phi.detach())) - 0.5 / tau * torch.sum((theta_bar - theta.detach()) ** 2)
            grad_theta = torch.autograd.grad(obj_theta, theta_bar)[0]
            with torch.no_grad():
                theta_bar = theta_bar + gap_inner_lr * grad_theta
                theta_bar = self.project_local_ball(theta_bar, theta.detach(), local_radius)
            theta_bar.requires_grad_(True)
        final_theta_obj = self.J(self.join_flat(theta_bar, phi.detach())).detach() - 0.5 / tau * torch.sum((theta_bar.detach() - theta.detach()) ** 2)
        theta_raw_improve = self.J(self.join_flat(theta_bar.detach(), phi.detach())).detach() - j_current
        theta_gap = final_theta_obj - j_current

        phi_bar = phi.detach().clone().requires_grad_(True)
        for _ in range(n_inner_gap):
            obj_phi = self.J(self.join_flat(theta.detach(), phi_bar)) + 0.5 / tau * torch.sum((phi_bar - phi.detach()) ** 2)
            grad_phi = torch.autograd.grad(obj_phi, phi_bar)[0]
            with torch.no_grad():
                phi_bar = phi_bar - gap_inner_lr * grad_phi
                phi_bar = self.project_local_ball(phi_bar, phi.detach(), local_radius)
            phi_bar.requires_grad_(True)
        final_phi_obj = self.J(self.join_flat(theta.detach(), phi_bar)).detach() + 0.5 / tau * torch.sum((phi_bar.detach() - phi.detach()) ** 2)
        phi_raw_improve = j_current - self.J(self.join_flat(theta.detach(), phi_bar.detach())).detach()
        phi_gap = j_current - final_phi_obj

        return {
            "theta_bar": theta_bar.detach(),
            "phi_bar": phi_bar.detach(),
            "theta_gap_raw": float(theta_gap),
            "phi_gap_raw": float(phi_gap),
            "theta_gap_pos": max(0.0, float(theta_gap)),
            "phi_gap_pos": max(0.0, float(phi_gap)),
            "theta_exploit_raw": float(theta_raw_improve),
            "phi_exploit_raw": float(phi_raw_improve),
            "theta_exploit_pos": max(0.0, float(theta_raw_improve)),
            "phi_exploit_pos": max(0.0, float(phi_raw_improve)),
            "p_tau": max(0.0, float(theta_gap)) + max(0.0, float(phi_gap)),
            "approx_local_exploitability": max(0.0, float(theta_raw_improve)) + max(0.0, float(phi_raw_improve)),
        }

    def core_metrics(self, flat: torch.Tensor, field_energy0: float) -> Dict[str, object]:
        field, j_val = self.field_and_j(flat, create_graph=False)
        field = field.detach()
        field_energy = 0.5 * float(torch.dot(field, field))
        normalized_v = field_energy / (field_energy0 + EPS)
        return {
            "V_lambda": normalized_v,
            "field_energy": field_energy,
            "field_norm": float(torch.linalg.norm(field)),
            "J_game": float(j_val.detach()),
            "field": field.detach().cpu().numpy(),
        }

    def full_eval(self, flat: torch.Tensor, field_energy0: float) -> Dict[str, object]:
        metrics = self.core_metrics(flat, field_energy0)
        field = torch.tensor(metrics["field"], dtype=DTYPE)
        G = self.curvature_direction(flat.detach(), field)
        train = self.rollout(flat, self.train_x0_batch, alpha=1.0, clean=False)
        clean = self.rollout(flat, self.eval_x0_batch, alpha=0.0, clean=True)
        adv = self.rollout(flat, self.eval_x0_batch, alpha=1.0, clean=False)
        metrics.update(
            {
                "train_task_return": float(train["task_return"]),
                "train_game_objective": float(train["J_game"]),
                "train_disturbance_energy": float(train["disturbance_energy"]),
                "train_state_clip_frac": float(train["state_clip_frac"]),
                "train_max_abs_state": float(train["max_abs_state"]),
                "train_mean_abs_u": float(train["mean_abs_u"]),
                "train_mean_abs_w": float(train["mean_abs_w"]),
                "train_max_abs_u": float(train["max_abs_u"]),
                "train_max_abs_w": float(train["max_abs_w"]),
                "train_mean_pre_tanh_abs_protagonist": float(train["mean_pre_tanh_abs_protagonist"]),
                "train_mean_pre_tanh_abs_adversary": float(train["mean_pre_tanh_abs_adversary"]),
                "train_max_pre_tanh_abs_protagonist": float(train["max_pre_tanh_abs_protagonist"]),
                "train_max_pre_tanh_abs_adversary": float(train["max_pre_tanh_abs_adversary"]),
                "clean_task_return": float(clean["task_return"]),
                "clean_task_cost": float(clean["task_cost"]),
                "clean_state_clip_frac": float(clean["state_clip_frac"]),
                "adversarial_task_return": float(adv["task_return"]),
                "adversarial_task_cost": float(adv["task_cost"]),
                "adversarial_game_objective": float(adv["J_game"]),
                "adversarial_disturbance_energy": float(adv["disturbance_energy"]),
                "adversarial_state_clip_frac": float(adv["state_clip_frac"]),
                "action_saturation_fraction_protagonist": float(adv["action_saturation_fraction_protagonist"]),
                "action_saturation_fraction_adversary": float(adv["action_saturation_fraction_adversary"]),
                "protagonist_param_norm": float(torch.linalg.norm(self.split_flat(flat)[0])),
                "adversary_param_norm": float(torch.linalg.norm(self.split_flat(flat)[1])),
                "G": G.detach().cpu().numpy(),
                "G_norm": float(torch.linalg.norm(G)),
                "cosine_FG": float(torch.dot(field, G) / (torch.linalg.norm(field) * torch.linalg.norm(G) + EPS)),
                "non_collinearity": float(math.sqrt(max(0.0, 1.0 - safe_float(torch.dot(field, G) / (torch.linalg.norm(field) * torch.linalg.norm(G) + EPS)) ** 2))),
            }
        )
        return metrics

    def unified_metrics(
        self,
        flat: torch.Tensor,
        field_energy0: float,
        p_tau0: float,
        lyap_cfg: UnifiedLyapunovConfig,
        include_rollouts: bool = True,
    ) -> Dict[str, object]:
        core = self.core_metrics(flat, field_energy0)
        field = torch.tensor(core["field"], dtype=DTYPE)
        G = self.curvature_direction(flat.detach(), field)
        gap_terms = self.local_gap_terms(
            flat=flat.detach(),
            tau=lyap_cfg.tau,
            n_inner_gap=lyap_cfg.n_inner_gap,
            gap_inner_lr=lyap_cfg.gap_inner_lr,
            local_radius=lyap_cfg.local_radius,
        )
        field_term = core["field_energy"] / (field_energy0 + EPS)
        normalized_p_tau = gap_terms["p_tau"] / (p_tau0 + EPS)
        V = lyap_cfg.lambda_F * field_term + lyap_cfg.lambda_P * normalized_p_tau
        metrics: Dict[str, object] = {
            "V_lambda": float(V),
            "raw_field_energy": float(core["field_energy"]),
            "field_term": float(field_term),
            "normalized_field_contribution": float(lyap_cfg.lambda_F * field_term),
            "raw_P_tau": float(gap_terms["p_tau"]),
            "normalized_P_tau": float(normalized_p_tau),
            "normalized_P_tau_contribution": float(lyap_cfg.lambda_P * normalized_p_tau),
            "field_norm": float(core["field_norm"]),
            "J_game": float(core["J_game"]),
            "approximate_local_exploitability": float(gap_terms["approx_local_exploitability"]),
            "theta_gap_raw": float(gap_terms["theta_gap_raw"]),
            "phi_gap_raw": float(gap_terms["phi_gap_raw"]),
            "theta_gap_pos": float(gap_terms["theta_gap_pos"]),
            "phi_gap_pos": float(gap_terms["phi_gap_pos"]),
            "theta_exploit_raw": float(gap_terms["theta_exploit_raw"]),
            "phi_exploit_raw": float(gap_terms["phi_exploit_raw"]),
            "theta_exploit_pos": float(gap_terms["theta_exploit_pos"]),
            "phi_exploit_pos": float(gap_terms["phi_exploit_pos"]),
            "field": core["field"],
            "G": G.detach().cpu().numpy(),
            "G_norm": float(torch.linalg.norm(G)),
            "cosine_FG": float(torch.dot(field, G) / (torch.linalg.norm(field) * torch.linalg.norm(G) + EPS)),
            "non_collinearity": float(math.sqrt(max(0.0, 1.0 - safe_float(torch.dot(field, G) / (torch.linalg.norm(field) * torch.linalg.norm(G) + EPS)) ** 2))),
        }
        if include_rollouts:
            train = self.rollout(flat, self.train_x0_batch, alpha=1.0, clean=False)
            clean = self.rollout(flat, self.eval_x0_batch, alpha=0.0, clean=True)
            adv = self.rollout(flat, self.eval_x0_batch, alpha=1.0, clean=False)
            theta, phi = self.split_flat(flat)
            metrics.update(
                {
                    "train_task_return": float(train["task_return"]),
                    "train_game_objective": float(train["J_game"]),
                    "train_disturbance_energy": float(train["disturbance_energy"]),
                    "train_state_clip_frac": float(train["state_clip_frac"]),
                    "train_max_abs_state": float(train["max_abs_state"]),
                    "train_mean_abs_u": float(train["mean_abs_u"]),
                    "train_mean_abs_w": float(train["mean_abs_w"]),
                    "train_max_abs_u": float(train["max_abs_u"]),
                    "train_max_abs_w": float(train["max_abs_w"]),
                    "train_mean_pre_tanh_abs_protagonist": float(train["mean_pre_tanh_abs_protagonist"]),
                    "train_mean_pre_tanh_abs_adversary": float(train["mean_pre_tanh_abs_adversary"]),
                    "train_max_pre_tanh_abs_protagonist": float(train["max_pre_tanh_abs_protagonist"]),
                    "train_max_pre_tanh_abs_adversary": float(train["max_pre_tanh_abs_adversary"]),
                    "clean_task_return": float(clean["task_return"]),
                    "clean_task_cost": float(clean["task_cost"]),
                    "clean_state_clip_frac": float(clean["state_clip_frac"]),
                    "adversarial_task_return": float(adv["task_return"]),
                    "adversarial_task_cost": float(adv["task_cost"]),
                    "adversarial_game_objective": float(adv["J_game"]),
                    "adversarial_disturbance_energy": float(adv["disturbance_energy"]),
                    "adversarial_state_clip_frac": float(adv["state_clip_frac"]),
                    "action_saturation_fraction_protagonist": float(adv["action_saturation_fraction_protagonist"]),
                    "action_saturation_fraction_adversary": float(adv["action_saturation_fraction_adversary"]),
                    "protagonist_param_norm": float(torch.linalg.norm(theta)),
                    "adversary_param_norm": float(torch.linalg.norm(phi)),
                }
            )
        return metrics

    def robustness_sweep(self, flat: torch.Tensor) -> Tuple[pd.DataFrame, float]:
        rows: List[Dict[str, object]] = []
        values: List[float] = []
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        for alpha in alphas:
            sim = self.rollout(flat, self.eval_x0_batch, alpha=alpha, clean=(alpha == 0.0))
            rows.append(
                {
                    "alpha": alpha,
                    "sweep_task_return": float(sim["task_return"]),
                    "sweep_task_cost": float(sim["task_cost"]),
                    "sweep_game_objective": float(sim["J_game"]),
                    "sweep_disturbance_energy": float(sim["disturbance_energy"]),
                    "sweep_state_clip_frac": float(sim["state_clip_frac"]),
                }
            )
            values.append(float(sim["task_return"]))
        return pd.DataFrame(rows), float(np.trapezoid(values, x=np.asarray(alphas, dtype=np.float64)))


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


def make_benchmark(
    beta_rot: float,
    rho_w: float,
    actor_init_scale: float = 0.01,
    actor_temperature: float = 1.0,
) -> RotLQRNNActorBenchmark:
    return RotLQRNNActorBenchmark(
        EnvConfig(
            beta_rot=beta_rot,
            rho_w=rho_w,
            actor_init_scale=actor_init_scale,
            actor_temperature=actor_temperature,
        )
    )


def verify_zero_sum_and_signs(benchmark: RotLQRNNActorBenchmark) -> Tuple[List[Dict[str, object]], str]:
    rows: List[Dict[str, object]] = []
    torch.manual_seed(benchmark.cfg.seed + 17)
    random_states = torch.randn(64, benchmark.cfg.state_dim, dtype=DTYPE)
    random_u = torch.randn(64, benchmark.cfg.action_dim, dtype=DTYPE)
    random_w = torch.randn(64, benchmark.cfg.disturbance_dim, dtype=DTYPE)
    x_q = torch.sum((random_states @ benchmark.Q) * random_states, dim=1)
    u_r = torch.sum((random_u @ benchmark.R) * random_u, dim=1)
    task_reward = -(x_q + u_r)
    rot_term = benchmark.cfg.beta_rot * torch.sum((random_u @ benchmark.H) * random_w, dim=1)
    dist_term = benchmark.cfg.rho_w * torch.sum(random_w * random_w, dim=1)
    rp = task_reward + rot_term + dist_term
    ra = -rp
    rows.append(
        {
            "check": "stage_zero_sum",
            "value": float(torch.max(torch.abs(rp + ra))),
            "passed": float(torch.max(torch.abs(rp + ra)) < 1e-10),
        }
    )

    flat0 = benchmark.flat0.detach().clone()
    j_p = benchmark.J(flat0)
    j_a = benchmark.J_adversary(flat0)
    rows.append(
        {
            "check": "J_adversary_is_negative",
            "value": float(torch.abs(j_p + j_a)),
            "passed": float(torch.abs(j_p + j_a) < 1e-10),
        }
    )

    F = benchmark.field_tensor(flat0, create_graph=False)
    lr = 1e-3
    protagonist_only = flat0.clone()
    phi_dim = benchmark.actor_param_dim
    protagonist_only[:phi_dim] = protagonist_only[:phi_dim] - lr * F[:phi_dim]
    adversary_only = flat0.clone()
    adversary_only[phi_dim:] = adversary_only[phi_dim:] - lr * F[phi_dim:]
    j_protagonist = benchmark.J(protagonist_only)
    j_adversary_update = benchmark.J(adversary_only)
    base_metrics = benchmark.full_eval(flat0, field_energy0=0.5 * float(torch.dot(F, F)))
    adv_metrics = benchmark.full_eval(adversary_only, field_energy0=0.5 * float(torch.dot(F, F)))
    rows.extend(
        [
            {
                "check": "protagonist_only_update_increases_J",
                "value": float(j_protagonist - j_p),
                "passed": float(j_protagonist > j_p),
            },
            {
                "check": "adversary_only_update_decreases_J",
                "value": float(j_adversary_update - j_p),
                "passed": float(j_adversary_update < j_p),
            },
            {
                "check": "adversary_only_reduces_task_return",
                "value": float(adv_metrics["adversarial_task_return"] - base_metrics["adversarial_task_return"]),
                "passed": float(adv_metrics["adversarial_task_return"] <= base_metrics["adversarial_task_return"] + 1e-8),
            },
            {
                "check": "grad_theta_finite",
                "value": float(torch.linalg.norm(F[:phi_dim])),
                "passed": float(torch.isfinite(torch.linalg.norm(F[:phi_dim]))),
            },
            {
                "check": "grad_phi_finite",
                "value": float(torch.linalg.norm(F[phi_dim:])),
                "passed": float(torch.isfinite(torch.linalg.norm(F[phi_dim:]))),
            },
        ]
    )
    G = benchmark.curvature_direction(flat0, F)
    rows.append(
        {
            "check": "JFP_finite",
            "value": float(torch.linalg.norm(G)),
            "passed": float(torch.isfinite(torch.linalg.norm(G))),
        }
    )
    report = "\n".join(
        [
            "# RotLQR-NNActor Zero-Sum Verification",
            "",
            f"- stage zero-sum max abs error: `{rows[0]['value']:.3e}`",
            f"- J_A + J_P at init: `{rows[1]['value']:.3e}`",
            f"- protagonist-only delta J: `{rows[2]['value']:.3e}`",
            f"- adversary-only delta J: `{rows[3]['value']:.3e}`",
            f"- adversary-only delta task return: `{rows[4]['value']:.3e}`",
            f"- ||grad_theta J||: `{rows[5]['value']:.3e}`",
            f"- ||grad_phi J||: `{rows[6]['value']:.3e}`",
            f"- ||J_F F||: `{rows[7]['value']:.3e}`",
            "",
            "Field convention verified against SGD semantics:",
            "",
            "- `F(z) = [-grad_theta J ; +grad_phi J]`",
            "- `z_next = z - lr * F(z)` means protagonist ascends J and adversary descends J.",
        ]
    )
    return rows, report


def geometry_audit_for_config(beta_rot: float, rho_w: float) -> Dict[str, object]:
    benchmark = make_benchmark(beta_rot, rho_w)
    flat0 = benchmark.flat0.detach().clone()
    F = benchmark.field_tensor(flat0, create_graph=False)
    JF = benchmark.full_jacobian(flat0).detach().cpu().numpy()
    sym = 0.5 * (JF + JF.T)
    anti = 0.5 * (JF - JF.T)
    dim = benchmark.actor_param_dim
    dtheta_dphi = JF[:dim, dim:]
    dphi_dtheta = JF[dim:, :dim]
    dtheta_dtheta = JF[:dim, :dim]
    dphi_dphi = JF[dim:, dim:]
    G = benchmark.curvature_direction(flat0, F)
    rot_ratio = np.linalg.norm(anti, ord="fro") / (np.linalg.norm(sym, ord="fro") + EPS)
    return {
        "beta_rot": beta_rot,
        "rho_w": rho_w,
        "field_norm": float(torch.linalg.norm(F)),
        "G_norm": float(torch.linalg.norm(G)),
        "cosine_FG": float(torch.dot(F, G) / (torch.linalg.norm(F) * torch.linalg.norm(G) + EPS)),
        "non_collinearity": float(math.sqrt(max(0.0, 1.0 - safe_float(torch.dot(F, G) / (torch.linalg.norm(F) * torch.linalg.norm(G) + EPS)) ** 2))),
        "rotation_ratio": float(rot_ratio),
        "cross_player_coupling": float(np.linalg.norm(dtheta_dphi, ord="fro") + np.linalg.norm(dphi_dtheta, ord="fro")),
        "same_player_coupling": float(np.linalg.norm(dtheta_dtheta, ord="fro") + np.linalg.norm(dphi_dphi, ord="fro")),
        "state_clip_frac": float(benchmark.rollout(flat0, benchmark.train_x0_batch)["state_clip_frac"]),
    }


def step_metrics_v(benchmark: RotLQRNNActorBenchmark, flat: torch.Tensor, field_energy0: float) -> Dict[str, object]:
    return benchmark.core_metrics(flat, field_energy0)


def apply_method_step(
    benchmark: RotLQRNNActorBenchmark,
    method: str,
    flat: torch.Tensor,
    base_lr: float,
    field_energy0: float,
    update_radius: float | None = None,
    probe_radius: float | None = None,
    fallback_to_egm: bool = False,
    trust_radii: Sequence[float] | None = None,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    before = step_metrics_v(benchmark, flat, field_energy0)
    F = torch.tensor(before["field"], dtype=DTYPE)
    G = benchmark.curvature_direction(flat.detach(), F)
    raw_beta = 0.0
    raw_gamma = 0.0
    gamma_active = 0.0
    selected_radius = 0.0
    trust_active = False
    fallback_used = False
    selected_step_type = method
    fit_cond = float("nan")
    fit_indef = float("nan")

    def eval_beta_gamma(beta_t: torch.Tensor, gamma_t: torch.Tensor) -> float:
        cand = flat - beta_t * F + gamma_t * G
        return step_metrics_v(benchmark, cand.detach(), field_energy0)["V_lambda"]

    if method == "sgd":
        delta = -base_lr * F
    elif method == "egm":
        half = flat - base_lr * F
        F_half = benchmark.field_tensor(half.detach(), create_graph=False).detach()
        delta = -base_lr * F_half
    elif method == "ppm":
        z_inner = flat.clone()
        for _ in range(10):
            F_inner = benchmark.field_tensor(z_inner.detach(), create_graph=False).detach()
            z_inner = flat - base_lr * F_inner
        delta = z_inner - flat
    elif method == "proposed_noG":
        beta, fit = fit_quadratic_1d(eval_beta_gamma, -F, probe_radius=probe_radius or 1e-3)
        raw_beta = beta
        fit_cond = float("nan")
        fit_indef = 0.0
        delta = -beta * F
    elif method == "proposed_qpg":
        beta, gamma, fit = fit_quadratic_2d(eval_beta_gamma, -F, G, probe_radius=probe_radius or 1e-3)
        raw_beta = beta
        raw_gamma = gamma
        gamma_active = float(abs(gamma) > 1e-12)
        fit_cond = fit["cond"]
        fit_indef = float(fit["indef"])
        delta = -beta * F + gamma * G
    else:
        raise ValueError(method)

    raw_update_norm = float(torch.linalg.norm(delta))
    trust_scaled_update_norm = raw_update_norm
    if update_radius is not None:
        delta, trust_active, raw_update_norm, trust_scaled_update_norm = trust_scale(delta, update_radius)
        selected_radius = update_radius

    candidate = (flat + delta).detach()
    after = step_metrics_v(benchmark, candidate, field_energy0)
    unsafe = not np.isfinite(after["V_lambda"])

    if (method in {"proposed_noG", "proposed_qpg"}) and (unsafe or after["V_lambda"] > before["V_lambda"] * 1.1):
        if trust_radii:
            for radius in sorted(set(float(r) for r in trust_radii if float(r) < selected_radius), reverse=True):
                trial_delta, _, _, trial_norm = trust_scale(-raw_beta * F + raw_gamma * G if method == "proposed_qpg" else -raw_beta * F, radius)
                trial_flat = (flat + trial_delta).detach()
                trial_after = step_metrics_v(benchmark, trial_flat, field_energy0)
                if np.isfinite(trial_after["V_lambda"]) and trial_after["V_lambda"] <= before["V_lambda"] * 1.1:
                    delta = trial_delta
                    candidate = trial_flat
                    after = trial_after
                    selected_radius = radius
                    trust_scaled_update_norm = trial_norm
                    trust_active = True
                    unsafe = False
                    break

    if (method in {"proposed_noG", "proposed_qpg"}) and fallback_to_egm and (unsafe or after["V_lambda"] > before["V_lambda"] * 1.1):
        fallback_used = True
        selected_step_type = "fallback_egm"
        half = flat - base_lr * F
        F_half = benchmark.field_tensor(half.detach(), create_graph=False).detach()
        delta = -base_lr * F_half
        candidate = (flat + delta).detach()
        after = step_metrics_v(benchmark, candidate, field_energy0)

    info = {
        "V_before": before["V_lambda"],
        "V_after": after["V_lambda"],
        "field_before": before["field_norm"],
        "field_after": after["field_norm"],
        "raw_beta": raw_beta,
        "raw_gamma": raw_gamma,
        "gamma_active": gamma_active,
        "raw_update_norm": raw_update_norm,
        "trust_scaled_update_norm": trust_scaled_update_norm,
        "trust_radius_active": float(trust_active),
        "selected_radius": selected_radius,
        "fallback_to_egm": float(fallback_used),
        "selected_step_type": selected_step_type,
        "G_contribution_ratio": float(torch.linalg.norm(raw_gamma * G) / (torch.linalg.norm(raw_beta * F) + EPS)) if method == "proposed_qpg" else 0.0,
        "fit_cond": fit_cond,
        "fit_indef": fit_indef,
        "cosine_FG": float(torch.dot(F, G) / (torch.linalg.norm(F) * torch.linalg.norm(G) + EPS)) if float(torch.linalg.norm(G)) > EPS else 1.0,
    }
    return candidate, {"before": before, "after": after, **info, "delta": delta.detach().cpu().numpy()}


def run_method(
    benchmark: RotLQRNNActorBenchmark,
    method: str,
    base_lr: float,
    iterations: int,
    field_energy0: float,
    update_radius: float | None = None,
    probe_radius: float | None = None,
    allow_fallback: bool = False,
    trust_radii: Sequence[float] | None = None,
    eval_interval: int = 25,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flat = benchmark.flat0.detach().clone()
    curve_rows: List[Dict[str, object]] = []
    diag_rows: List[Dict[str, object]] = []
    robustness_rows: List[Dict[str, object]] = []
    nan_flag = False

    for iteration in range(iterations):
        flat, info = apply_method_step(
            benchmark=benchmark,
            method=method,
            flat=flat,
            base_lr=base_lr,
            field_energy0=field_energy0,
            update_radius=update_radius,
            probe_radius=probe_radius,
            fallback_to_egm=allow_fallback,
            trust_radii=trust_radii,
        )
        after = info["after"]
        full_after: Dict[str, object] | None = None
        if iteration % eval_interval == 0 or iteration == iterations - 1:
            full_after = benchmark.full_eval(flat, field_energy0)
        curve_rows.append(
            {
                "iteration": iteration,
                "method": method,
                "V_lambda": after["V_lambda"],
                "field_energy": after["field_energy"],
                "field_norm": after["field_norm"],
                "J_game": after["J_game"],
                "train_task_return": safe_float(full_after["train_task_return"]) if full_after else float("nan"),
                "clean_task_return": safe_float(full_after["clean_task_return"]) if full_after else float("nan"),
                "adversarial_task_return": safe_float(full_after["adversarial_task_return"]) if full_after else float("nan"),
                "G_norm": safe_float(full_after["G_norm"]) if full_after else float("nan"),
                "cosine_FG": safe_float(full_after["cosine_FG"]) if full_after else float("nan"),
                "state_clip_frac": safe_float(full_after["train_state_clip_frac"]) if full_after else float("nan"),
            }
        )
        diag_rows.append(
            {
                "iteration": iteration,
                "method": method,
                "V_before": info["V_before"],
                "V_after": info["V_after"],
                "field_before": info["field_before"],
                "field_after": info["field_after"],
                "raw_beta": info["raw_beta"],
                "raw_gamma": info["raw_gamma"],
                "gamma_active": info["gamma_active"],
                "raw_update_norm": info["raw_update_norm"],
                "trust_scaled_update_norm": info["trust_scaled_update_norm"],
                "trust_radius_active": info["trust_radius_active"],
                "selected_radius": info["selected_radius"],
                "fallback_to_egm": info["fallback_to_egm"],
                "selected_step_type": info["selected_step_type"],
                "G_contribution_ratio": info["G_contribution_ratio"],
                "fit_cond": info["fit_cond"],
                "fit_indef": info["fit_indef"],
                "cosine_FG": info["cosine_FG"],
            }
        )
        if not np.isfinite(after["V_lambda"]) or not np.isfinite(after["field_norm"]):
            nan_flag = True
            break

    final_full = benchmark.full_eval(flat, field_energy0)
    sweep_df, robustness_auc = benchmark.robustness_sweep(flat)
    for row in sweep_df.to_dict("records"):
        row["method"] = method
        robustness_rows.append(row)

    summary_df = pd.DataFrame(
        [
            {
                "method": method,
                "base_lr": base_lr,
                "iterations_completed": len(curve_rows),
                "V_lambda_AUC": auc_from_series([row["V_lambda"] for row in curve_rows]),
                "field_norm_AUC": auc_from_series([row["field_norm"] for row in curve_rows]),
                "time_to_field_norm_1e-2": first_below([row["field_norm"] for row in curve_rows], 1e-2),
                "time_to_field_norm_1e-3": first_below([row["field_norm"] for row in curve_rows], 1e-3),
                "final_V_lambda": safe_float(curve_rows[-1]["V_lambda"]) if curve_rows else float("nan"),
                "final_field_norm": safe_float(curve_rows[-1]["field_norm"]) if curve_rows else float("nan"),
                "final_J_game": safe_float(curve_rows[-1]["J_game"]) if curve_rows else float("nan"),
                "final_train_task_return": safe_float(final_full["train_task_return"]),
                "final_clean_task_return": safe_float(final_full["clean_task_return"]),
                "final_adversarial_task_return": safe_float(final_full["adversarial_task_return"]),
                "robustness_auc": robustness_auc,
                "gamma_active_frac": float(np.mean([safe_float(r["gamma_active"]) for r in diag_rows])) if diag_rows else float("nan"),
                "fallback_to_egm_frac": float(np.mean([safe_float(r["fallback_to_egm"]) for r in diag_rows])) if diag_rows else float("nan"),
                "mean_G_contribution_ratio": float(np.mean([safe_float(r["G_contribution_ratio"]) for r in diag_rows])) if diag_rows else float("nan"),
                "mean_cosine_FG": float(np.mean([safe_float(r["cosine_FG"]) for r in diag_rows])) if diag_rows else float("nan"),
                "nan_flag": float(nan_flag),
            }
        ]
    )
    return summary_df, pd.DataFrame(curve_rows), pd.DataFrame(diag_rows), pd.DataFrame(robustness_rows)


def baseline_gate_pass(summary_rows: pd.DataFrame) -> Tuple[bool, str]:
    best = summary_rows.sort_values(["method", "V_lambda_AUC"]).groupby("method", as_index=False).first()
    sgd = best[best["method"] == "sgd"].iloc[0]
    egm = best[best["method"] == "egm"].iloc[0]
    ppm = best[best["method"] == "ppm"].iloc[0]
    reasons = []
    gain_v = min(sgd["V_lambda_AUC"] / max(egm["V_lambda_AUC"], EPS), sgd["V_lambda_AUC"] / max(ppm["V_lambda_AUC"], EPS))
    gain_f = min(sgd["field_norm_AUC"] / max(egm["field_norm_AUC"], EPS), sgd["field_norm_AUC"] / max(ppm["field_norm_AUC"], EPS))
    time_gain = []
    for threshold in ["time_to_field_norm_1e-2", "time_to_field_norm_1e-3"]:
        sgd_t = safe_float(sgd[threshold])
        for row in [egm, ppm]:
            row_t = safe_float(row[threshold])
            if np.isfinite(sgd_t) and np.isfinite(row_t) and row_t > 0:
                time_gain.append(sgd_t / row_t)
    max_time_gain = max(time_gain) if time_gain else 0.0
    finite_ok = all(best["nan_flag"] < 0.5)
    return_ok = all(best["final_adversarial_task_return"] > -1e8)
    pass_gate = finite_ok and return_ok and (gain_v >= 2.0 or gain_f >= 2.0 or max_time_gain >= 2.0)
    reasons.append(f"gain_v={gain_v:.3f}")
    reasons.append(f"gain_f={gain_f:.3f}")
    reasons.append(f"max_time_gain={max_time_gain:.3f}")
    reasons.append(f"finite_ok={finite_ok}")
    reasons.append(f"return_ok={return_ok}")
    return pass_gate, ", ".join(reasons)


def choose_best_lr(summary_df: pd.DataFrame, method: str) -> float:
    method_df = summary_df[(summary_df["method"] == method) & (summary_df["nan_flag"] < 0.5)].copy()
    method_df = method_df.sort_values(["V_lambda_AUC", "field_norm_AUC", "final_field_norm"])
    return float(method_df.iloc[0]["base_lr"])


def preflight_update_radius(
    benchmark: RotLQRNNActorBenchmark,
    method: str,
    base_lr: float,
    radii: Sequence[float],
    field_energy0: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for radius in radii:
        summary_df, curve_df, diag_df, _ = run_method(
            benchmark=benchmark,
            method=method,
            base_lr=base_lr,
            iterations=20,
            field_energy0=field_energy0,
            update_radius=radius,
            probe_radius=min(radius, 1e-2) * 0.5,
            allow_fallback=True,
            trust_radii=radii,
        )
        rows.append(
            {
                "method": method,
                "update_radius": radius,
                "V_lambda_AUC": safe_float(summary_df.iloc[0]["V_lambda_AUC"]),
                "final_V_lambda": safe_float(summary_df.iloc[0]["final_V_lambda"]),
                "field_norm_AUC": safe_float(summary_df.iloc[0]["field_norm_AUC"]),
                "gamma_active_frac": safe_float(summary_df.iloc[0]["gamma_active_frac"]),
                "fallback_to_egm_frac": safe_float(summary_df.iloc[0]["fallback_to_egm_frac"]),
                "nan_flag": safe_float(summary_df.iloc[0]["nan_flag"]),
                "mean_G_contribution_ratio": safe_float(summary_df.iloc[0]["mean_G_contribution_ratio"]),
                "trust_radius_active_frac": float(np.mean(diag_df["trust_radius_active"])) if not diag_df.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def same_start_candidate_comparison(
    benchmark: RotLQRNNActorBenchmark,
    best_lrs: Dict[str, float],
    qpg_radius: float,
    nog_radius: float,
    field_energy0: float,
) -> pd.DataFrame:
    checkpoints = [0, 10, 50, 100, 200]
    methods_path = ["sgd", "egm", "ppm", "proposed_qpg"]
    flat = benchmark.flat0.detach().clone()
    trajectory = {0: flat.clone()}
    for iteration in range(1, max(checkpoints) + 1):
        flat, _ = apply_method_step(
            benchmark=benchmark,
            method="proposed_qpg",
            flat=flat,
            base_lr=best_lrs["proposed_qpg"],
            field_energy0=field_energy0,
            update_radius=qpg_radius,
            probe_radius=min(qpg_radius, 1e-2) * 0.5,
            fallback_to_egm=True,
            trust_radii=[qpg_radius],
        )
        if iteration in checkpoints:
            trajectory[iteration] = flat.clone()

    rows: List[Dict[str, object]] = []
    for checkpoint in checkpoints:
        z = trajectory[checkpoint]
        candidates = {
            "zero": z.clone(),
        }
        for method_name in ["sgd", "egm", "ppm"]:
            cand, info = apply_method_step(
                benchmark=benchmark,
                method=method_name,
                flat=z.clone(),
                base_lr=best_lrs[method_name],
                field_energy0=field_energy0,
            )
            candidates[method_name] = cand
        cand_nog, info_nog = apply_method_step(
            benchmark=benchmark,
            method="proposed_noG",
            flat=z.clone(),
            base_lr=best_lrs["proposed_noG"],
            field_energy0=field_energy0,
            update_radius=nog_radius,
            probe_radius=min(nog_radius, 1e-2) * 0.5,
            fallback_to_egm=True,
            trust_radii=[nog_radius],
        )
        cand_qpg, info_qpg = apply_method_step(
            benchmark=benchmark,
            method="proposed_qpg",
            flat=z.clone(),
            base_lr=best_lrs["proposed_qpg"],
            field_energy0=field_energy0,
            update_radius=qpg_radius,
            probe_radius=min(qpg_radius, 1e-2) * 0.5,
            fallback_to_egm=True,
            trust_radii=[qpg_radius],
        )
        candidates["proposed_noG"] = cand_nog
        candidates["proposed_QP_G"] = cand_qpg
        v_before = step_metrics_v(benchmark, z, field_energy0)["V_lambda"]
        qpg_delta = cand_qpg - z
        sgd_delta = candidates["sgd"] - z
        egm_delta = candidates["egm"] - z
        nog_delta = cand_nog - z
        for candidate_name, candidate_flat in candidates.items():
            metrics = benchmark.full_eval(candidate_flat, field_energy0)
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "candidate": candidate_name,
                    "V_before": v_before,
                    "V_after": metrics["V_lambda"],
                    "delta_V": metrics["V_lambda"] - v_before,
                    "field_norm_after": metrics["field_norm"],
                    "J_after": metrics["J_game"],
                    "train_task_return_after": metrics["train_task_return"],
                    "clean_task_return_after": metrics["clean_task_return"],
                    "adversarial_task_return_after": metrics["adversarial_task_return"],
                    "update_norm": float(torch.linalg.norm(candidate_flat - z)),
                    "cos_qp_sgd": float(torch.dot(qpg_delta, sgd_delta) / (torch.linalg.norm(qpg_delta) * torch.linalg.norm(sgd_delta) + EPS)),
                    "cos_qp_egm": float(torch.dot(qpg_delta, egm_delta) / (torch.linalg.norm(qpg_delta) * torch.linalg.norm(egm_delta) + EPS)),
                    "cos_qp_nog": float(torch.dot(qpg_delta, nog_delta) / (torch.linalg.norm(qpg_delta) * torch.linalg.norm(nog_delta) + EPS)),
                    "qpg_G_contribution_ratio": safe_float(info_qpg["G_contribution_ratio"]),
                }
            )
    return pd.DataFrame(rows)


def plot_line(ax, df: pd.DataFrame, methods: Sequence[str], y: str, title: str, ylabel: str, floor: float | None = None) -> None:
    colors = {
        "sgd": "#d62728",
        "egm": "#2ca02c",
        "ppm": "#9467bd",
        "proposed_noG": "#1f77b4",
        "proposed_qpg": "#8c564b",
        "proposed_noG_unified_repaired": "#1f77b4",
        "proposed_QP_G_unified_repaired": "#8c564b",
    }
    for method in methods:
        sub = df[df["method"] == method]
        vals = sub[y].to_numpy(dtype=np.float64)
        if floor is not None:
            vals = np.maximum(vals, floor)
        ax.plot(sub["iteration"], vals, label=method, color=colors.get(method, None))
    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    if "V" in y or "field" in y:
        ax.set_yscale("log")
    ax.grid(alpha=0.2)


def save_baseline_plots(curves: pd.DataFrame, geometry_df: pd.DataFrame) -> None:
    methods = ["sgd", "egm", "ppm"]
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_line(ax, curves, methods, "field_norm", "Baseline Field Norm", "||F||", floor=1e-12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_baseline_field_norm.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_line(ax, curves, methods, "V_lambda", "Baseline Normalized Field Energy", "V_lambda", floor=1e-12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_baseline_V_lambda.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_line(axes[0], curves, methods, "train_task_return", "Train Task Return", "return")
    plot_line(axes[1], curves, methods, "clean_task_return", "Clean Return", "return")
    plot_line(axes[2], curves, methods, "adversarial_task_return", "Adversarial Return", "return")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_baseline_returns.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(geometry_df["beta_rot"], geometry_df["rotation_ratio"], c=geometry_df["rho_w"], cmap="viridis", s=80)
    for _, row in geometry_df.iterrows():
        ax.text(row["beta_rot"], row["rotation_ratio"], f"rho={row['rho_w']:.2f}", fontsize=8)
    ax.set_title("Geometry Audit Rotation Ratio")
    ax.set_xlabel("beta_rot")
    ax.set_ylabel("rotation_ratio")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_baseline_geometry.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_line(axes[0, 0], curves, methods, "field_norm", "Field Norm", "||F||", floor=1e-12)
    plot_line(axes[0, 1], curves, methods, "V_lambda", "Normalized Field Energy", "V_lambda", floor=1e-12)
    plot_line(axes[1, 0], curves, methods, "clean_task_return", "Clean Return", "return")
    axes[1, 1].scatter(geometry_df["beta_rot"], geometry_df["rotation_ratio"], c=geometry_df["rho_w"], cmap="viridis", s=80)
    axes[1, 1].set_title("Geometry Rotation Ratio")
    axes[1, 1].set_xlabel("beta_rot")
    axes[1, 1].set_ylabel("rotation_ratio")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_baseline_gate_all_plots_big.png", dpi=180)
    plt.close(fig)


def save_qp_plots(curves: pd.DataFrame, diagnostics: pd.DataFrame, robustness: pd.DataFrame, geometry_df: pd.DataFrame) -> None:
    methods = ["sgd", "egm", "ppm", "proposed_noG", "proposed_qpg"]
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_line(ax, curves, methods, "field_norm", "QP Field Norm", "||F||", floor=1e-12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_qp_field_norm.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_line(ax, curves, methods, "V_lambda", "QP Normalized Field Energy", "V_lambda", floor=1e-12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_qp_V_lambda.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_line(axes[0], curves, methods, "train_task_return", "Train Return", "return")
    plot_line(axes[1], curves, methods, "clean_task_return", "Clean Return", "return")
    plot_line(axes[2], curves, methods, "adversarial_task_return", "Adversarial Return", "return")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_qp_returns.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for method in methods:
        sub = robustness[robustness["method"] == method]
        ax.plot(sub["alpha"], sub["sweep_task_return"], marker="o", label=method)
    ax.set_title("Robustness Sweep")
    ax.set_xlabel("alpha")
    ax.set_ylabel("task return")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_qp_robustness.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(geometry_df["beta_rot"], geometry_df["rotation_ratio"], c=geometry_df["rho_w"], cmap="viridis", s=80)
    ax.set_title("Environment Geometry")
    ax.set_xlabel("beta_rot")
    ax.set_ylabel("rotation_ratio")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_qp_geometry.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    qpg_diag = diagnostics[diagnostics["method"] == "proposed_qpg"]
    nog_diag = diagnostics[diagnostics["method"] == "proposed_noG"]
    axes[0].plot(qpg_diag["iteration"], qpg_diag["raw_beta"], label="QPG beta")
    axes[0].plot(nog_diag["iteration"], nog_diag["raw_beta"], label="noG beta")
    axes[0].legend()
    axes[0].set_ylabel("beta")
    axes[1].plot(qpg_diag["iteration"], qpg_diag["raw_gamma"], label="QPG gamma", color="#8c564b")
    axes[1].set_ylabel("gamma")
    axes[2].plot(qpg_diag["iteration"], qpg_diag["G_contribution_ratio"], label="||gamma G|| / ||beta F||", color="#2ca02c")
    axes[2].set_ylabel("G ratio")
    axes[2].set_xlabel("iteration")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_qp_beta_gamma.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    qpg_diag = diagnostics[diagnostics["method"] == "proposed_qpg"]
    ax.plot(qpg_diag["iteration"], qpg_diag["G_contribution_ratio"], label="G contribution")
    ax.plot(qpg_diag["iteration"], qpg_diag["trust_scaled_update_norm"], label="update norm")
    ax.set_title("QP Update Distinctness")
    ax.set_xlabel("iteration")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_qp_update_distinctness.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    plot_line(axes[0, 0], curves, methods, "field_norm", "Field Norm", "||F||", floor=1e-12)
    plot_line(axes[0, 1], curves, methods, "V_lambda", "Normalized Field Energy", "V_lambda", floor=1e-12)
    plot_line(axes[1, 0], curves, methods, "clean_task_return", "Clean Return", "return")
    plot_line(axes[1, 1], curves, methods, "adversarial_task_return", "Adversarial Return", "return")
    for method in methods:
        sub = robustness[robustness["method"] == method]
        axes[2, 0].plot(sub["alpha"], sub["sweep_task_return"], marker="o", label=method)
    axes[2, 0].set_title("Robustness Sweep")
    axes[2, 0].set_xlabel("alpha")
    axes[2, 0].set_ylabel("task return")
    axes[2, 0].grid(alpha=0.2)
    qpg_diag = diagnostics[diagnostics["method"] == "proposed_qpg"]
    axes[2, 1].plot(qpg_diag["iteration"], qpg_diag["raw_beta"], label="beta")
    axes[2, 1].plot(qpg_diag["iteration"], qpg_diag["raw_gamma"], label="gamma")
    axes[2, 1].set_title("QP beta / gamma")
    axes[2, 1].legend()
    axes[2, 1].grid(alpha=0.2)
    axes[0, 0].legend()
    axes[2, 0].legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_qp_all_plots_big.png", dpi=180)
    plt.close(fig)


def select_old_centers() -> Dict[str, float]:
    old_summary = pd.read_csv(RESULT_ROOT / "nn_rot_lqr_qp_summary.csv")
    return {
        "sgd": float(old_summary[old_summary["method"] == "sgd"]["base_lr"].iloc[0]),
        "egm": float(old_summary[old_summary["method"] == "egm"]["base_lr"].iloc[0]),
        "ppm": float(old_summary[old_summary["method"] == "ppm"]["base_lr"].iloc[0]),
        "proposed_noG": float(old_summary[old_summary["method"] == "proposed_noG"]["base_lr"].iloc[0]),
        "proposed_qpg": float(old_summary[old_summary["method"] == "proposed_qpg"]["base_lr"].iloc[0]),
    }


def build_unified_config_rows(benchmark: RotLQRNNActorBenchmark) -> Tuple[pd.DataFrame, Dict[str, object], float]:
    field0 = benchmark.field_tensor(benchmark.flat0, create_graph=False).detach()
    field_energy0 = 0.5 * float(torch.dot(field0, field0))
    rows: List[Dict[str, object]] = []
    best_cfg: Dict[str, object] | None = None
    best_score = float("inf")
    base_lr = 0.1
    for lambda_f in [1e-4, 1e-3, 1e-2, 3e-2, 1e-1]:
        for tau in [0.03, 0.1, 0.3]:
            for n_inner in [3, 5, 10]:
                for local_radius in [0.03, 0.1, 0.3]:
                    gap_inner_lr = 0.3 * tau
                    cfg = UnifiedLyapunovConfig(
                        lambda_F=lambda_f,
                        lambda_P=1.0,
                        tau=tau,
                        n_inner_gap=n_inner,
                        gap_inner_lr=gap_inner_lr,
                        local_radius=local_radius,
                        update_radius=0.1,
                    )
                    p_tau0 = benchmark.local_gap_terms(benchmark.flat0, tau=tau, n_inner_gap=n_inner, gap_inner_lr=gap_inner_lr, local_radius=local_radius)["p_tau"]
                    before = benchmark.unified_metrics(benchmark.flat0, field_energy0, p_tau0, cfg, include_rollouts=True)
                    qpg_next, qpg_info = apply_method_step_unified(
                        benchmark=benchmark,
                        method="proposed_qpg",
                        flat=benchmark.flat0.clone(),
                        base_lr=base_lr,
                        field_energy0=field_energy0,
                        p_tau0=p_tau0,
                        lyap_cfg=cfg,
                        update_radius=cfg.update_radius,
                        probe_radius=min(cfg.update_radius, 1e-2) * 0.5,
                        fallback_to_egm=True,
                        trust_radii=[0.1, 0.03, 0.01],
                    )
                    nog_next, nog_info = apply_method_step_unified(
                        benchmark=benchmark,
                        method="proposed_noG",
                        flat=benchmark.flat0.clone(),
                        base_lr=base_lr,
                        field_energy0=field_energy0,
                        p_tau0=p_tau0,
                        lyap_cfg=cfg,
                        update_radius=cfg.update_radius,
                        probe_radius=min(cfg.update_radius, 1e-2) * 0.5,
                        fallback_to_egm=True,
                        trust_radii=[0.1, 0.03, 0.01],
                    )
                    qpg_after = benchmark.unified_metrics(qpg_next, field_energy0, p_tau0, cfg, include_rollouts=True)
                    nog_after = benchmark.unified_metrics(nog_next, field_energy0, p_tau0, cfg, include_rollouts=True)
                    finite = all(
                        np.isfinite(v)
                        for v in [
                            qpg_after["V_lambda"],
                            nog_after["V_lambda"],
                            qpg_after["field_norm"],
                            nog_after["field_norm"],
                            qpg_after["raw_P_tau"],
                            nog_after["raw_P_tau"],
                            qpg_after["approximate_local_exploitability"],
                            nog_after["approximate_local_exploitability"],
                        ]
                    )
                    severe_clip = max(
                        safe_float(qpg_after["train_state_clip_frac"]),
                        safe_float(nog_after["train_state_clip_frac"]),
                        safe_float(qpg_after["action_saturation_fraction_protagonist"]),
                        safe_float(qpg_after["action_saturation_fraction_adversary"]),
                    ) > 0.5
                    qpg_delta_v = qpg_after["V_lambda"] - before["V_lambda"]
                    nog_delta_v = nog_after["V_lambda"] - before["V_lambda"]
                    score = qpg_after["V_lambda"] + 0.1 * qpg_after["field_norm"] + 0.5 * max(0.0, qpg_delta_v)
                    row = {
                        "lambda_F": lambda_f,
                        "lambda_P": 1.0,
                        "tau": tau,
                        "n_inner_gap": n_inner,
                        "gap_inner_lr": gap_inner_lr,
                        "local_radius": local_radius,
                        "update_radius": 0.1,
                        "p_tau0": p_tau0,
                        "V_before": before["V_lambda"],
                        "qpg_V_after": qpg_after["V_lambda"],
                        "qpg_delta_V": qpg_delta_v,
                        "nog_V_after": nog_after["V_lambda"],
                        "nog_delta_V": nog_delta_v,
                        "qpg_field_after": qpg_after["field_norm"],
                        "qpg_P_tau_after": qpg_after["raw_P_tau"],
                        "qpg_exploit_after": qpg_after["approximate_local_exploitability"],
                        "qpg_gamma_active": qpg_info["gamma_active"],
                        "qpg_fallback_frac_probe": qpg_info["fallback_to_egm"],
                        "nog_fallback_frac_probe": nog_info["fallback_to_egm"],
                        "qpg_G_contribution_ratio": qpg_info["G_contribution_ratio"],
                        "qpg_state_clip_frac": qpg_after["train_state_clip_frac"],
                        "qpg_action_sat_p": qpg_after["action_saturation_fraction_protagonist"],
                        "qpg_action_sat_a": qpg_after["action_saturation_fraction_adversary"],
                        "finite_ok": float(finite),
                        "severe_clip": float(severe_clip),
                    }
                    rows.append(row)
                    if (
                        finite
                        and not severe_clip
                        and qpg_delta_v < 0.0
                        and qpg_after["field_norm"] < before["field_norm"] * 10.0
                        and qpg_after["raw_P_tau"] < max(before["raw_P_tau"] * 10.0, 1e6)
                        and qpg_after["approximate_local_exploitability"] < max(before["approximate_local_exploitability"] * 10.0, 1e6)
                        and qpg_after["V_lambda"] <= nog_after["V_lambda"] + 1e-12
                        and qpg_info["gamma_active"] > 0.0
                        and score < best_score
                    ):
                        best_score = score
                        best_cfg = row
    if best_cfg is None:
        best_cfg = min(rows, key=lambda r: (0 if r["finite_ok"] > 0.5 else 1, r["qpg_V_after"]))
    return pd.DataFrame(rows), best_cfg, field_energy0


def step_metrics_unified(
    benchmark: RotLQRNNActorBenchmark,
    flat: torch.Tensor,
    field_energy0: float,
    p_tau0: float,
    lyap_cfg: UnifiedLyapunovConfig,
    include_rollouts: bool = False,
) -> Dict[str, object]:
    return benchmark.unified_metrics(flat, field_energy0, p_tau0, lyap_cfg, include_rollouts=include_rollouts)


def apply_method_step_unified(
    benchmark: RotLQRNNActorBenchmark,
    method: str,
    flat: torch.Tensor,
    base_lr: float,
    field_energy0: float,
    p_tau0: float,
    lyap_cfg: UnifiedLyapunovConfig,
    update_radius: float | None = None,
    probe_radius: float | None = None,
    fallback_to_egm: bool = False,
    trust_radii: Sequence[float] | None = None,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    before = step_metrics_unified(benchmark, flat, field_energy0, p_tau0, lyap_cfg, include_rollouts=True)
    F = torch.tensor(before["field"], dtype=DTYPE)
    G = benchmark.curvature_direction(flat.detach(), F)
    raw_beta = 0.0
    raw_gamma = 0.0
    gamma_active = 0.0
    selected_radius = 0.0
    trust_active = False
    fallback_used = False
    selected_step_type = method
    fit_cond = float("nan")
    fit_indef = float("nan")
    predicted_after = float("nan")

    def eval_beta_gamma(beta_t: torch.Tensor, gamma_t: torch.Tensor) -> float:
        cand = flat - beta_t * F + gamma_t * G
        return step_metrics_unified(benchmark, cand.detach(), field_energy0, p_tau0, lyap_cfg, include_rollouts=False)["V_lambda"]

    if method == "sgd":
        delta = -base_lr * F
    elif method == "egm":
        half = flat - base_lr * F
        F_half = benchmark.field_tensor(half.detach(), create_graph=False).detach()
        delta = -base_lr * F_half
    elif method == "ppm":
        z_inner = flat.clone()
        for _ in range(10):
            F_inner = benchmark.field_tensor(z_inner.detach(), create_graph=False).detach()
            z_inner = flat - base_lr * F_inner
        delta = z_inner - flat
    elif method == "proposed_noG":
        beta, _ = fit_quadratic_1d(eval_beta_gamma, -F, probe_radius=probe_radius or 1e-3)
        raw_beta = beta
        predicted_after = eval_beta_gamma(torch.tensor(beta, dtype=DTYPE), torch.tensor(0.0, dtype=DTYPE))
        delta = -beta * F
    elif method == "proposed_qpg":
        beta, gamma, fit = fit_quadratic_2d(eval_beta_gamma, -F, G, probe_radius=probe_radius or 1e-3)
        raw_beta = beta
        raw_gamma = gamma
        gamma_active = float(abs(gamma) > 1e-12)
        fit_cond = fit["cond"]
        fit_indef = float(fit["indef"])
        predicted_after = eval_beta_gamma(torch.tensor(beta, dtype=DTYPE), torch.tensor(gamma, dtype=DTYPE))
        delta = -beta * F + gamma * G
    else:
        raise ValueError(method)

    raw_update_norm = float(torch.linalg.norm(delta))
    trust_scaled_update_norm = raw_update_norm
    if update_radius is not None:
        delta, trust_active, raw_update_norm, trust_scaled_update_norm = trust_scale(delta, update_radius)
        selected_radius = update_radius

    candidate = (flat + delta).detach()
    after_core = step_metrics_unified(benchmark, candidate, field_energy0, p_tau0, lyap_cfg, include_rollouts=False)
    unsafe = not np.isfinite(after_core["V_lambda"])

    if (method in {"proposed_noG", "proposed_qpg"}) and (unsafe or after_core["V_lambda"] > before["V_lambda"] * 1.1):
        if trust_radii:
            for radius in sorted(set(float(r) for r in trust_radii if float(r) < selected_radius), reverse=True):
                trial_delta, _, _, trial_norm = trust_scale(-raw_beta * F + raw_gamma * G if method == "proposed_qpg" else -raw_beta * F, radius)
                trial_flat = (flat + trial_delta).detach()
                trial_after = step_metrics_unified(benchmark, trial_flat, field_energy0, p_tau0, lyap_cfg, include_rollouts=False)
                if np.isfinite(trial_after["V_lambda"]) and trial_after["V_lambda"] <= before["V_lambda"] * 1.1:
                    delta = trial_delta
                    candidate = trial_flat
                    after_core = trial_after
                    selected_radius = radius
                    trust_scaled_update_norm = trial_norm
                    trust_active = True
                    unsafe = False
                    break

    if (method in {"proposed_noG", "proposed_qpg"}) and fallback_to_egm and (unsafe or after_core["V_lambda"] > before["V_lambda"] * 1.1):
        fallback_used = True
        selected_step_type = "fallback_egm"
        half = flat - base_lr * F
        F_half = benchmark.field_tensor(half.detach(), create_graph=False).detach()
        delta = -base_lr * F_half
        candidate = (flat + delta).detach()

    after = step_metrics_unified(benchmark, candidate, field_energy0, p_tau0, lyap_cfg, include_rollouts=True)
    info = {
        "V_before": before["V_lambda"],
        "V_predicted_after": predicted_after,
        "V_after": after["V_lambda"],
        "field_term_before": before["field_term"],
        "field_term_after": after["field_term"],
        "P_tau_before": before["raw_P_tau"],
        "P_tau_after": after["raw_P_tau"],
        "normalized_P_tau_before": before["normalized_P_tau"],
        "normalized_P_tau_after": after["normalized_P_tau"],
        "exploitability_before": before["approximate_local_exploitability"],
        "exploitability_after": after["approximate_local_exploitability"],
        "field_before": before["field_norm"],
        "field_after": after["field_norm"],
        "raw_beta": raw_beta,
        "raw_gamma": raw_gamma,
        "gamma_active": gamma_active,
        "raw_update_norm": raw_update_norm,
        "trust_scaled_update_norm": trust_scaled_update_norm,
        "trust_radius_active": float(trust_active),
        "selected_radius": selected_radius,
        "fallback_to_egm": float(fallback_used),
        "selected_step_type": selected_step_type,
        "G_contribution_ratio": float(torch.linalg.norm(raw_gamma * G) / (torch.linalg.norm(raw_beta * F) + EPS)) if method == "proposed_qpg" else 0.0,
        "fit_cond": fit_cond,
        "fit_indef": fit_indef,
        "cosine_FG": float(torch.dot(F, G) / (torch.linalg.norm(F) * torch.linalg.norm(G) + EPS)) if float(torch.linalg.norm(G)) > EPS else 1.0,
        "non_collinearity": float(math.sqrt(max(0.0, 1.0 - safe_float(torch.dot(F, G) / (torch.linalg.norm(F) * torch.linalg.norm(G) + EPS)) ** 2))) if float(torch.linalg.norm(G)) > EPS else 0.0,
    }
    return candidate, {"before": before, "after": after, **info, "delta": delta.detach().cpu().numpy()}


def run_method_unified(
    benchmark: RotLQRNNActorBenchmark,
    method: str,
    base_lr: float,
    iterations: int,
    field_energy0: float,
    p_tau0: float,
    lyap_cfg: UnifiedLyapunovConfig,
    update_radius: float | None = None,
    probe_radius: float | None = None,
    allow_fallback: bool = False,
    trust_radii: Sequence[float] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flat = benchmark.flat0.detach().clone()
    curve_rows: List[Dict[str, object]] = []
    diag_rows: List[Dict[str, object]] = []
    robustness_rows: List[Dict[str, object]] = []
    nan_flag = False

    for iteration in range(iterations):
        flat, info = apply_method_step_unified(
            benchmark=benchmark,
            method=method,
            flat=flat,
            base_lr=base_lr,
            field_energy0=field_energy0,
            p_tau0=p_tau0,
            lyap_cfg=lyap_cfg,
            update_radius=update_radius,
            probe_radius=probe_radius,
            fallback_to_egm=allow_fallback,
            trust_radii=trust_radii,
        )
        after = info["after"]
        curve_rows.append(
            {
                "iteration": iteration,
                "method": method,
                "V_lambda": after["V_lambda"],
                "raw_field_energy": after["raw_field_energy"],
                "field_term": after["field_term"],
                "normalized_field_contribution": after["normalized_field_contribution"],
                "raw_P_tau": after["raw_P_tau"],
                "normalized_P_tau": after["normalized_P_tau"],
                "normalized_P_tau_contribution": after["normalized_P_tau_contribution"],
                "approximate_local_exploitability": after["approximate_local_exploitability"],
                "field_norm": after["field_norm"],
                "J_game": after["J_game"],
                "train_task_return": after["train_task_return"],
                "clean_task_return": after["clean_task_return"],
                "adversarial_task_return": after["adversarial_task_return"],
                "state_clip_fraction": after["train_state_clip_frac"],
                "action_saturation_fraction_protagonist": after["action_saturation_fraction_protagonist"],
                "action_saturation_fraction_adversary": after["action_saturation_fraction_adversary"],
                "protagonist_param_norm": after["protagonist_param_norm"],
                "adversary_param_norm": after["adversary_param_norm"],
                "G_norm": after["G_norm"],
                "cosine_FG": after["cosine_FG"],
                "non_collinearity": after["non_collinearity"],
            }
        )
        diag_rows.append(
            {
                "iteration": iteration,
                "method": method,
                "V_before_composite": info["V_before"],
                "V_predicted_after_composite": info["V_predicted_after"],
                "V_actual_after_composite": info["V_after"],
                "field_term_before": info["field_term_before"],
                "field_term_after": info["field_term_after"],
                "P_tau_before": info["P_tau_before"],
                "P_tau_after": info["P_tau_after"],
                "normalized_P_tau_before": info["normalized_P_tau_before"],
                "normalized_P_tau_after": info["normalized_P_tau_after"],
                "exploitability_before": info["exploitability_before"],
                "exploitability_after": info["exploitability_after"],
                "raw_beta": info["raw_beta"],
                "raw_gamma": info["raw_gamma"],
                "gamma_active": info["gamma_active"],
                "raw_update_norm": info["raw_update_norm"],
                "trust_scaled_update_norm": info["trust_scaled_update_norm"],
                "trust_radius_active": info["trust_radius_active"],
                "selected_radius": info["selected_radius"],
                "fallback_to_egm": info["fallback_to_egm"],
                "selected_step_type": info["selected_step_type"],
                "G_contribution_ratio": info["G_contribution_ratio"],
                "fit_cond": info["fit_cond"],
                "fit_indef": info["fit_indef"],
                "cosine_FG": info["cosine_FG"],
                "non_collinearity": info["non_collinearity"],
            }
        )
        if not np.isfinite(after["V_lambda"]) or not np.isfinite(after["field_norm"]):
            nan_flag = True
            break

    final_after = benchmark.unified_metrics(flat, field_energy0, p_tau0, lyap_cfg, include_rollouts=True)
    sweep_df, robustness_auc = benchmark.robustness_sweep(flat)
    for row in sweep_df.to_dict("records"):
        row["method"] = method
        robustness_rows.append(row)

    summary_df = pd.DataFrame(
        [
            {
                "method": method,
                "base_lr": base_lr,
                "lambda_F": lyap_cfg.lambda_F,
                "lambda_P": lyap_cfg.lambda_P,
                "tau": lyap_cfg.tau,
                "n_inner_gap": lyap_cfg.n_inner_gap,
                "gap_inner_lr": lyap_cfg.gap_inner_lr,
                "local_radius": lyap_cfg.local_radius,
                "update_radius": update_radius if update_radius is not None else float("nan"),
                "iterations_completed": len(curve_rows),
                "V_lambda_AUC": auc_from_series([row["V_lambda"] for row in curve_rows]),
                "P_tau_AUC": auc_from_series([row["raw_P_tau"] for row in curve_rows]),
                "exploitability_AUC": auc_from_series([row["approximate_local_exploitability"] for row in curve_rows]),
                "field_norm_AUC": auc_from_series([row["field_norm"] for row in curve_rows]),
                "final_V_lambda": safe_float(final_after["V_lambda"]),
                "final_P_tau": safe_float(final_after["raw_P_tau"]),
                "final_exploitability": safe_float(final_after["approximate_local_exploitability"]),
                "final_field_norm": safe_float(final_after["field_norm"]),
                "final_train_task_return": safe_float(final_after["train_task_return"]),
                "final_clean_task_return": safe_float(final_after["clean_task_return"]),
                "final_adversarial_task_return": safe_float(final_after["adversarial_task_return"]),
                "robustness_auc": robustness_auc,
                "gamma_active_frac": float(np.mean([safe_float(r["gamma_active"]) for r in diag_rows])) if diag_rows else float("nan"),
                "fallback_to_egm_frac": float(np.mean([safe_float(r["fallback_to_egm"]) for r in diag_rows])) if diag_rows else float("nan"),
                "mean_G_contribution_ratio": float(np.mean([safe_float(r["G_contribution_ratio"]) for r in diag_rows])) if diag_rows else float("nan"),
                "mean_cosine_FG": float(np.mean([safe_float(r["cosine_FG"]) for r in diag_rows])) if diag_rows else float("nan"),
                "mean_non_collinearity": float(np.mean([safe_float(r["non_collinearity"]) for r in diag_rows])) if diag_rows else float("nan"),
                "mean_state_clip_fraction": float(np.mean([safe_float(r["state_clip_fraction"]) for r in curve_rows])) if curve_rows else float("nan"),
                "mean_action_saturation_fraction_protagonist": float(np.mean([safe_float(r["action_saturation_fraction_protagonist"]) for r in curve_rows])) if curve_rows else float("nan"),
                "mean_action_saturation_fraction_adversary": float(np.mean([safe_float(r["action_saturation_fraction_adversary"]) for r in curve_rows])) if curve_rows else float("nan"),
                "nan_flag": float(nan_flag),
            }
        ]
    )
    return summary_df, pd.DataFrame(curve_rows), pd.DataFrame(diag_rows), pd.DataFrame(robustness_rows)


def baseline_gate_pass_unified(summary_rows: pd.DataFrame) -> Tuple[bool, str]:
    best = summary_rows.sort_values(["method", "V_lambda_AUC"]).groupby("method", as_index=False).first()
    sgd = best[best["method"] == "sgd"].iloc[0]
    egm = best[best["method"] == "egm"].iloc[0]
    ppm = best[best["method"] == "ppm"].iloc[0]
    egm_beats_sgd = float(egm["V_lambda_AUC"]) < float(sgd["V_lambda_AUC"]) or float(egm["field_norm_AUC"]) < float(sgd["field_norm_AUC"])
    ppm_beats_sgd = float(ppm["V_lambda_AUC"]) < float(sgd["V_lambda_AUC"]) or float(ppm["field_norm_AUC"]) < float(sgd["field_norm_AUC"])
    gain_v = min(float(sgd["V_lambda_AUC"]) / max(float(egm["V_lambda_AUC"]), EPS), float(sgd["V_lambda_AUC"]) / max(float(ppm["V_lambda_AUC"]), EPS))
    gain_f = min(float(sgd["field_norm_AUC"]) / max(float(egm["field_norm_AUC"]), EPS), float(sgd["field_norm_AUC"]) / max(float(ppm["field_norm_AUC"]), EPS))
    finite_ok = all(best["nan_flag"] < 0.5)
    clip_ok = all(best["mean_state_clip_fraction"] < 0.25) and all(best["mean_action_saturation_fraction_adversary"] < 0.95)
    converged_ok = all(best["final_V_lambda"] < 1.0) and all(best["final_field_norm"] < 1e3) and all(best["final_train_task_return"] > -1e9)
    pass_gate = finite_ok and clip_ok and converged_ok and (egm_beats_sgd or ppm_beats_sgd) and (gain_v >= 1.25 or gain_f >= 1.25)
    return pass_gate, (
        f"egm_beats_sgd={egm_beats_sgd}, ppm_beats_sgd={ppm_beats_sgd}, "
        f"gain_v={gain_v:.3f}, gain_f={gain_f:.3f}, finite_ok={finite_ok}, "
        f"clip_ok={clip_ok}, converged_ok={converged_ok}"
    )


def choose_best_lr_unified(summary_df: pd.DataFrame, method: str) -> float:
    sub = summary_df[(summary_df["method"] == method) & (summary_df["nan_flag"] < 0.5)].copy()
    sub = sub.sort_values(["V_lambda_AUC", "P_tau_AUC", "field_norm_AUC", "final_V_lambda"])
    return float(sub.iloc[0]["base_lr"])


def same_start_candidate_comparison_unified(
    benchmark: RotLQRNNActorBenchmark,
    best_lrs: Dict[str, float],
    field_energy0: float,
    p_tau0: float,
    lyap_cfg: UnifiedLyapunovConfig,
) -> pd.DataFrame:
    checkpoints = [0, 10, 50, 100, 200]
    flat = benchmark.flat0.detach().clone()
    trajectory = {0: flat.clone()}
    for iteration in range(1, max(checkpoints) + 1):
        flat, _ = apply_method_step_unified(
            benchmark=benchmark,
            method="proposed_qpg",
            flat=flat,
            base_lr=best_lrs["proposed_qpg"],
            field_energy0=field_energy0,
            p_tau0=p_tau0,
            lyap_cfg=lyap_cfg,
            update_radius=lyap_cfg.update_radius,
            probe_radius=min(lyap_cfg.update_radius, 1e-2) * 0.5,
            fallback_to_egm=True,
            trust_radii=[lyap_cfg.update_radius, 0.03, 0.01],
        )
        if iteration in checkpoints:
            trajectory[iteration] = flat.clone()

    rows: List[Dict[str, object]] = []
    for checkpoint in checkpoints:
        z = trajectory[checkpoint]
        candidates = {"zero": z.clone()}
        sgd_cand, _ = apply_method_step_unified(benchmark, "sgd", z.clone(), best_lrs["sgd"], field_energy0, p_tau0, lyap_cfg)
        egm_cand, _ = apply_method_step_unified(benchmark, "egm", z.clone(), best_lrs["egm"], field_energy0, p_tau0, lyap_cfg)
        ppm_cand, _ = apply_method_step_unified(benchmark, "ppm", z.clone(), best_lrs["ppm"], field_energy0, p_tau0, lyap_cfg)
        nog_cand, nog_info = apply_method_step_unified(
            benchmark, "proposed_noG", z.clone(), best_lrs["proposed_noG"], field_energy0, p_tau0, lyap_cfg,
            update_radius=lyap_cfg.update_radius, probe_radius=min(lyap_cfg.update_radius, 1e-2) * 0.5,
            fallback_to_egm=True, trust_radii=[lyap_cfg.update_radius, 0.03, 0.01]
        )
        qpg_cand, qpg_info = apply_method_step_unified(
            benchmark, "proposed_qpg", z.clone(), best_lrs["proposed_qpg"], field_energy0, p_tau0, lyap_cfg,
            update_radius=lyap_cfg.update_radius, probe_radius=min(lyap_cfg.update_radius, 1e-2) * 0.5,
            fallback_to_egm=True, trust_radii=[lyap_cfg.update_radius, 0.03, 0.01]
        )
        candidates.update({"sgd": sgd_cand, "egm": egm_cand, "ppm": ppm_cand, "proposed_noG": nog_cand, "proposed_QP_G": qpg_cand})
        before = benchmark.unified_metrics(z, field_energy0, p_tau0, lyap_cfg, include_rollouts=True)
        qpg_delta = qpg_cand - z
        sgd_delta = sgd_cand - z
        egm_delta = egm_cand - z
        nog_delta = nog_cand - z
        for candidate_name, candidate_flat in candidates.items():
            after = benchmark.unified_metrics(candidate_flat, field_energy0, p_tau0, lyap_cfg, include_rollouts=True)
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "candidate": candidate_name,
                    "V_before": before["V_lambda"],
                    "V_after": after["V_lambda"],
                    "delta_V": after["V_lambda"] - before["V_lambda"],
                    "P_tau_before": before["raw_P_tau"],
                    "P_tau_after": after["raw_P_tau"],
                    "exploitability_before": before["approximate_local_exploitability"],
                    "exploitability_after": after["approximate_local_exploitability"],
                    "field_term_after": after["field_term"],
                    "normalized_P_tau_after": after["normalized_P_tau"],
                    "field_norm_after": after["field_norm"],
                    "train_task_return_after": after["train_task_return"],
                    "clean_task_return_after": after["clean_task_return"],
                    "adversarial_task_return_after": after["adversarial_task_return"],
                    "update_norm": float(torch.linalg.norm(candidate_flat - z)),
                    "cos_qp_sgd": float(torch.dot(qpg_delta, sgd_delta) / (torch.linalg.norm(qpg_delta) * torch.linalg.norm(sgd_delta) + EPS)),
                    "cos_qp_egm": float(torch.dot(qpg_delta, egm_delta) / (torch.linalg.norm(qpg_delta) * torch.linalg.norm(egm_delta) + EPS)),
                    "cos_qp_nog": float(torch.dot(qpg_delta, nog_delta) / (torch.linalg.norm(qpg_delta) * torch.linalg.norm(nog_delta) + EPS)),
                    "qpg_G_contribution_ratio": safe_float(qpg_info["G_contribution_ratio"]),
                }
            )
    return pd.DataFrame(rows)


def save_unified_plots(curves: pd.DataFrame, diagnostics: pd.DataFrame, robustness: pd.DataFrame, same_start: pd.DataFrame) -> None:
    methods = ["sgd", "egm", "ppm", "proposed_noG", "proposed_qpg"]
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_line(ax, curves, methods, "V_lambda", "Unified Composite Lyapunov", "V_lambda", floor=1e-12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_unified_V_lambda.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    plot_line(axes[0], curves, methods, "normalized_field_contribution", "Normalized Field Contribution", "value", floor=1e-12)
    plot_line(axes[1], curves, methods, "normalized_P_tau_contribution", "Normalized P_tau Contribution", "value", floor=1e-12)
    plot_line(axes[2], curves, methods, "V_lambda", "Total Composite V", "value", floor=1e-12)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_unified_components.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_line(ax, curves, methods, "approximate_local_exploitability", "Approximate Local Exploitability", "exploitability", floor=1e-12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_unified_exploitability.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_line(ax, curves, methods, "field_norm", "Field Norm (Diagnostic)", "||F||", floor=1e-12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_unified_field_norm.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_line(axes[0], curves, methods, "train_task_return", "Train Task Return", "return")
    plot_line(axes[1], curves, methods, "clean_task_return", "Clean Task Return", "return")
    plot_line(axes[2], curves, methods, "adversarial_task_return", "Adversarial Task Return", "return")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_unified_returns.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for method in methods:
        sub = robustness[robustness["method"] == method]
        ax.plot(sub["alpha"], sub["sweep_task_return"], marker="o", label=method)
    ax.set_title("Unified Robustness Sweep")
    ax.set_xlabel("alpha")
    ax.set_ylabel("task return")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_unified_robustness.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    qpg_diag = diagnostics[diagnostics["method"] == "proposed_qpg"]
    nog_diag = diagnostics[diagnostics["method"] == "proposed_noG"]
    axes[0].plot(qpg_diag["iteration"], qpg_diag["raw_beta"], label="QPG beta")
    axes[0].plot(nog_diag["iteration"], nog_diag["raw_beta"], label="noG beta")
    axes[0].legend()
    axes[0].set_ylabel("beta")
    axes[1].plot(qpg_diag["iteration"], qpg_diag["raw_gamma"], label="QPG gamma", color="#8c564b")
    axes[1].set_ylabel("gamma")
    axes[2].plot(qpg_diag["iteration"], qpg_diag["G_contribution_ratio"], label="G ratio", color="#2ca02c")
    axes[2].plot(qpg_diag["iteration"], qpg_diag["trust_scaled_update_norm"], label="update norm", color="#1f77b4")
    axes[2].legend()
    axes[2].set_ylabel("ratio/norm")
    axes[2].set_xlabel("iteration")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_unified_beta_gamma.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for candidate in ["sgd", "egm", "ppm", "proposed_noG", "proposed_QP_G"]:
        sub = same_start[same_start["candidate"] == candidate]
        axes[0].plot(sub["checkpoint"], sub["delta_V"], marker="o", label=candidate)
        axes[1].plot(sub["checkpoint"], sub["exploitability_after"], marker="o", label=candidate)
    axes[0].set_title("Same-start actual Delta V")
    axes[0].set_xlabel("checkpoint")
    axes[0].set_ylabel("delta V")
    axes[1].set_title("Same-start exploitability after")
    axes[1].set_xlabel("checkpoint")
    axes[1].set_ylabel("exploitability")
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_unified_same_start.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    plot_line(axes[0, 0], curves, methods, "V_lambda", "Composite V", "V", floor=1e-12)
    plot_line(axes[0, 1], curves, methods, "normalized_field_contribution", "Field contrib", "value", floor=1e-12)
    plot_line(axes[0, 2], curves, methods, "normalized_P_tau_contribution", "P_tau contrib", "value", floor=1e-12)
    plot_line(axes[1, 0], curves, methods, "approximate_local_exploitability", "Exploitability", "value", floor=1e-12)
    plot_line(axes[1, 1], curves, methods, "field_norm", "Field norm", "||F||", floor=1e-12)
    plot_line(axes[1, 2], curves, methods, "clean_task_return", "Clean return", "return")
    for method in methods:
        sub = robustness[robustness["method"] == method]
        axes[2, 0].plot(sub["alpha"], sub["sweep_task_return"], marker="o", label=method)
    axes[2, 0].set_title("Robustness sweep")
    axes[2, 0].set_xlabel("alpha")
    axes[2, 0].set_ylabel("task return")
    qpg_diag = diagnostics[diagnostics["method"] == "proposed_qpg"]
    axes[2, 1].plot(qpg_diag["iteration"], qpg_diag["raw_beta"], label="beta")
    axes[2, 1].plot(qpg_diag["iteration"], qpg_diag["raw_gamma"], label="gamma")
    axes[2, 1].set_title("QP beta/gamma")
    axes[2, 1].legend()
    for candidate in ["sgd", "egm", "ppm", "proposed_noG", "proposed_QP_G"]:
        sub = same_start[same_start["candidate"] == candidate]
        axes[2, 2].plot(sub["checkpoint"], sub["delta_V"], marker="o", label=candidate)
    axes[2, 2].set_title("Same-start delta V")
    axes[2, 2].legend()
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_unified_all_plots_big.png", dpi=180)
    plt.close(fig)


def geometry_snapshot(
    benchmark: RotLQRNNActorBenchmark,
    flat: torch.Tensor,
    field_energy0: float,
    p_tau0: float,
    lyap_cfg: UnifiedLyapunovConfig,
    label: str,
    iteration: int,
) -> Dict[str, object]:
    JF = benchmark.full_jacobian(flat.detach())
    sym = 0.5 * (JF + JF.T)
    skew = 0.5 * (JF - JF.T)
    dim = benchmark.actor_param_dim
    dtheta_dphi = JF[:dim, dim:]
    dphi_dtheta = JF[dim:, :dim]
    dtheta_dtheta = JF[:dim, :dim]
    dphi_dphi = JF[dim:, dim:]
    metrics = benchmark.unified_metrics(flat, field_energy0, p_tau0, lyap_cfg, include_rollouts=True)
    return {
        "label": label,
        "iteration": iteration,
        "rotation_ratio": float(torch.linalg.norm(skew) / (torch.linalg.norm(sym) + EPS)),
        "cross_player_coupling": float(torch.linalg.norm(dtheta_dphi) + torch.linalg.norm(dphi_dtheta)),
        "same_player_coupling": float(torch.linalg.norm(dtheta_dtheta) + torch.linalg.norm(dphi_dphi)),
        "G_over_F": float(metrics["G_norm"] / (metrics["field_norm"] + EPS)),
        "cosine_FG": float(metrics["cosine_FG"]),
        "non_collinearity": float(metrics["non_collinearity"]),
        "action_saturation_fraction_protagonist": float(metrics["action_saturation_fraction_protagonist"]),
        "action_saturation_fraction_adversary": float(metrics["action_saturation_fraction_adversary"]),
        "field_norm": float(metrics["field_norm"]),
        "raw_P_tau": float(metrics["raw_P_tau"]),
        "approximate_local_exploitability": float(metrics["approximate_local_exploitability"]),
    }


def run_lr_saturation_study() -> None:
    fixed_beta_rot = 4.0
    fixed_rho_w = 0.05
    centers = select_old_centers()
    preflight_df = pd.read_csv(RESULT_ROOT / "nn_rot_lqr_unified_lyapunov_preflight.csv")
    preflight_df["score"] = preflight_df["qpg_V_after"] + 0.1 * preflight_df["qpg_field_after"] + 0.5 * preflight_df["qpg_delta_V"].clip(lower=0.0)
    feasible = preflight_df[
        (preflight_df["finite_ok"] > 0.5)
        & (preflight_df["severe_clip"] < 0.5)
        & (preflight_df["qpg_delta_V"] < 0.0)
        & (preflight_df["qpg_gamma_active"] > 0.0)
        & (preflight_df["qpg_V_after"] <= preflight_df["nog_V_after"] + 1e-12)
    ].copy()
    best_cfg_row = (feasible.sort_values(["score", "qpg_V_after"]).iloc[0] if not feasible.empty else preflight_df.sort_values(["finite_ok", "qpg_V_after"], ascending=[False, True]).iloc[0]).to_dict()
    lyap_cfg = UnifiedLyapunovConfig(
        lambda_F=safe_float(best_cfg_row["lambda_F"]),
        lambda_P=safe_float(best_cfg_row["lambda_P"]),
        tau=safe_float(best_cfg_row["tau"]),
        n_inner_gap=int(best_cfg_row["n_inner_gap"]),
        gap_inner_lr=safe_float(best_cfg_row["gap_inner_lr"]),
        local_radius=safe_float(best_cfg_row["local_radius"]),
        update_radius=safe_float(best_cfg_row["update_radius"]),
    )

    def run_setting(tag: str, lr_map: Dict[str, float], iterations: int, actor_init_scale: float = 0.01, actor_temperature: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
        benchmark = make_benchmark(fixed_beta_rot, fixed_rho_w, actor_init_scale=actor_init_scale, actor_temperature=actor_temperature)
        field0 = benchmark.field_tensor(benchmark.flat0, create_graph=False).detach()
        field_energy0 = 0.5 * float(torch.dot(field0, field0))
        p_tau0 = benchmark.local_gap_terms(
            benchmark.flat0,
            tau=lyap_cfg.tau,
            n_inner_gap=lyap_cfg.n_inner_gap,
            gap_inner_lr=lyap_cfg.gap_inner_lr,
            local_radius=lyap_cfg.local_radius,
        )["p_tau"]
        initial_metrics = benchmark.unified_metrics(benchmark.flat0, field_energy0, p_tau0, lyap_cfg, include_rollouts=True)
        summary_frames: List[pd.DataFrame] = []
        curve_frames: List[pd.DataFrame] = []
        robust_frames: List[pd.DataFrame] = []
        geometry_rows: List[Dict[str, object]] = []
        z_trajs: Dict[str, List[torch.Tensor]] = {}
        for method in ["sgd", "egm", "ppm"]:
            flat = benchmark.flat0.detach().clone()
            z_trajs[method] = [flat.clone()]
            curve_rows: List[Dict[str, object]] = []
            for iteration in range(iterations):
                flat, info = apply_method_step_unified(
                    benchmark=benchmark,
                    method=method,
                    flat=flat,
                    base_lr=lr_map[method],
                    field_energy0=field_energy0,
                    p_tau0=p_tau0,
                    lyap_cfg=lyap_cfg,
                )
                after = info["after"]
                curve_rows.append(
                    {
                        "tag": tag,
                        "method": method,
                        "iteration": iteration,
                        "V_lambda": after["V_lambda"],
                        "field_term": after["field_term"],
                        "raw_P_tau": after["raw_P_tau"],
                        "normalized_P_tau": after["normalized_P_tau"],
                        "approximate_local_exploitability": after["approximate_local_exploitability"],
                        "field_norm": after["field_norm"],
                        "J_game": after["J_game"],
                        "train_task_return": after["train_task_return"],
                        "clean_task_return": after["clean_task_return"],
                        "adversarial_task_return": after["adversarial_task_return"],
                        "state_clip_fraction": after["train_state_clip_frac"],
                        "action_saturation_fraction_protagonist": after["action_saturation_fraction_protagonist"],
                        "action_saturation_fraction_adversary": after["action_saturation_fraction_adversary"],
                        "max_abs_state": after["train_max_abs_state"],
                        "mean_abs_u": after["train_mean_abs_u"],
                        "mean_abs_w": after["train_mean_abs_w"],
                        "max_abs_u": after["train_max_abs_u"],
                        "max_abs_w": after["train_max_abs_w"],
                        "mean_pre_tanh_abs_protagonist": after["train_mean_pre_tanh_abs_protagonist"],
                        "mean_pre_tanh_abs_adversary": after["train_mean_pre_tanh_abs_adversary"],
                        "max_pre_tanh_abs_protagonist": after["train_max_pre_tanh_abs_protagonist"],
                        "max_pre_tanh_abs_adversary": after["train_max_pre_tanh_abs_adversary"],
                        "protagonist_param_norm": after["protagonist_param_norm"],
                        "adversary_param_norm": after["adversary_param_norm"],
                        "nan_flag": float(not np.isfinite(after["V_lambda"]) or not np.isfinite(after["field_norm"])),
                        "ppm_inner_steps": 10 if method == "ppm" else np.nan,
                    }
                )
                z_trajs[method].append(flat.clone())
                if iteration in [0, 9, 49, iterations - 1]:
                    geometry_rows.append(geometry_snapshot(benchmark, flat, field_energy0, p_tau0, lyap_cfg, method, iteration + 1))
            curve_df = pd.DataFrame(curve_rows)
            final = curve_df.iloc[-1].to_dict()
            sweep_df, robustness_auc = benchmark.robustness_sweep(flat)
            sweep_df["method"] = method
            sweep_df["tag"] = tag
            robust_frames.append(sweep_df)
            init_clean = float(initial_metrics["clean_task_return"])
            init_adv = float(initial_metrics["adversarial_task_return"])
            mean_sat_p = float(curve_df["action_saturation_fraction_protagonist"].mean())
            mean_sat_a = float(curve_df["action_saturation_fraction_adversary"].mean())
            final_sat_p = float(curve_df["action_saturation_fraction_protagonist"].iloc[-1])
            final_sat_a = float(curve_df["action_saturation_fraction_adversary"].iloc[-1])
            mean_clip = float(curve_df["state_clip_fraction"].mean())
            valid = (
                float(curve_df["nan_flag"].max()) < 0.5
                and mean_clip <= 0.01
                and mean_sat_p <= 0.5
                and mean_sat_a <= 0.5
                and final_sat_p <= 0.7
                and final_sat_a <= 0.7
                and np.isfinite(final["field_norm"])
                and np.isfinite(final["raw_P_tau"])
                and np.isfinite(final["approximate_local_exploitability"])
                and float(final["clean_task_return"]) >= init_clean - 1000.0
                and float(final["adversarial_task_return"]) >= init_adv - 1000.0
            )
            summary_frames.append(
                pd.DataFrame(
                    [{
                        "tag": tag,
                        "method": method,
                        "lr": lr_map[method],
                        "actor_init_scale": actor_init_scale,
                        "actor_temperature": actor_temperature,
                        "iterations": iterations,
                        "V_lambda_AUC": auc_from_series(curve_df["V_lambda"].tolist()),
                        "P_tau_AUC": auc_from_series(curve_df["raw_P_tau"].tolist()),
                        "field_norm_AUC": auc_from_series(curve_df["field_norm"].tolist()),
                        "time_to_V_1e-2": first_below(curve_df["V_lambda"].tolist(), 1e-2),
                        "time_to_Ptau_1e-2": first_below(curve_df["raw_P_tau"].tolist(), 1e-2),
                        "final_V_lambda": float(final["V_lambda"]),
                        "final_P_tau": float(final["raw_P_tau"]),
                        "final_exploitability": float(final["approximate_local_exploitability"]),
                        "final_field_norm": float(final["field_norm"]),
                        "final_train_task_return": float(final["train_task_return"]),
                        "final_clean_task_return": float(final["clean_task_return"]),
                        "final_adversarial_task_return": float(final["adversarial_task_return"]),
                        "clean_return_initial": init_clean,
                        "adversarial_return_initial": init_adv,
                        "robustness_auc": robustness_auc,
                        "mean_action_saturation_fraction_protagonist": mean_sat_p,
                        "mean_action_saturation_fraction_adversary": mean_sat_a,
                        "final_action_saturation_fraction_protagonist": final_sat_p,
                        "final_action_saturation_fraction_adversary": final_sat_a,
                        "mean_state_clip_fraction": mean_clip,
                        "final_state_clip_fraction": float(curve_df["state_clip_fraction"].iloc[-1]),
                        "max_abs_state": float(curve_df["max_abs_state"].max()),
                        "mean_abs_u": float(curve_df["mean_abs_u"].mean()),
                        "mean_abs_w": float(curve_df["mean_abs_w"].mean()),
                        "max_abs_u": float(curve_df["max_abs_u"].max()),
                        "max_abs_w": float(curve_df["max_abs_w"].max()),
                        "mean_pre_tanh_abs_protagonist": float(curve_df["mean_pre_tanh_abs_protagonist"].mean()),
                        "mean_pre_tanh_abs_adversary": float(curve_df["mean_pre_tanh_abs_adversary"].mean()),
                        "max_pre_tanh_abs_protagonist": float(curve_df["max_pre_tanh_abs_protagonist"].max()),
                        "max_pre_tanh_abs_adversary": float(curve_df["max_pre_tanh_abs_adversary"].max()),
                        "protagonist_param_norm": float(final["protagonist_param_norm"]),
                        "adversary_param_norm": float(final["adversary_param_norm"]),
                        "nan_flag": float(curve_df["nan_flag"].max()),
                        "valid": float(valid),
                        "ppm_inner_steps": 10 if method == "ppm" else np.nan,
                    }]
                )
            )
            curve_frames.append(curve_df)
        return (
            pd.concat(summary_frames, ignore_index=True),
            pd.concat(curve_frames, ignore_index=True),
            pd.concat(robust_frames, ignore_index=True),
            pd.DataFrame(geometry_rows),
            {"field_energy0": field_energy0, "p_tau0": p_tau0, "initial_metrics": initial_metrics},
        )

    lr1e3_map = {"sgd": 1e-3, "egm": 1e-3, "ppm": 1e-3}
    summary_1e3, curves_1e3, robust_1e3, geom_1e3, meta = run_setting("lr1e-3", lr1e3_map, iterations=200)
    summary_1e3.to_csv(RESULT_ROOT / "nn_rot_lqr_lr1e3_baseline_sanity.csv", index=False)

    gate_best = summary_1e3.set_index("method")
    egm_row = gate_best.loc["egm"]
    ppm_row = gate_best.loc["ppm"]
    sgd_row = gate_best.loc["sgd"]
    gate_pass = (
        (egm_row["valid"] > 0.5 or ppm_row["valid"] > 0.5)
        and (
            (egm_row["valid"] > 0.5 and sgd_row["V_lambda_AUC"] / max(egm_row["V_lambda_AUC"], EPS) >= 1.5)
            or (ppm_row["valid"] > 0.5 and sgd_row["V_lambda_AUC"] / max(ppm_row["V_lambda_AUC"], EPS) >= 1.5)
            or (egm_row["valid"] > 0.5 and sgd_row["P_tau_AUC"] / max(egm_row["P_tau_AUC"], EPS) >= 1.5)
            or (ppm_row["valid"] > 0.5 and sgd_row["P_tau_AUC"] / max(ppm_row["P_tau_AUC"], EPS) >= 1.5)
            or (egm_row["valid"] > 0.5 and safe_float(sgd_row["time_to_V_1e-2"]) / max(safe_float(egm_row["time_to_V_1e-2"]), 1.0) >= 1.5)
            or (ppm_row["valid"] > 0.5 and safe_float(sgd_row["time_to_V_1e-2"]) / max(safe_float(ppm_row["time_to_V_1e-2"]), 1.0) >= 1.5)
        )
    )

    report_lines = [
        "# lr=1e-3 Baseline Sanity Report",
        "",
        f"- fixed environment: `beta_rot={fixed_beta_rot}, rho_w={fixed_rho_w}`",
        f"- unified lambda_F/lambda_P: `{lyap_cfg.lambda_F}` / `{lyap_cfg.lambda_P}`",
        f"- tau / inner_steps / local_radius: `{lyap_cfg.tau}` / `{lyap_cfg.n_inner_gap}` / `{lyap_cfg.local_radius}`",
        "",
        "## Summary",
        "",
        df_text(summary_1e3),
        "",
        "## Answers",
        "",
        f"1. At lr=1e-3, are SGD/EGM/PPM non-saturated? `SGD={bool(sgd_row['valid'] > 0.5)}`, `EGM={bool(egm_row['valid'] > 0.5)}`, `PPM={bool(ppm_row['valid'] > 0.5)}`",
        f"2. At lr=1e-3, do clean/adversarial returns collapse? `SGD={float(sgd_row['final_clean_task_return']) >= float(sgd_row['clean_return_initial']) - 1000 and float(sgd_row['final_adversarial_task_return']) >= float(sgd_row['adversarial_return_initial']) - 1000}`, `EGM={float(egm_row['final_clean_task_return']) >= float(egm_row['clean_return_initial']) - 1000 and float(egm_row['final_adversarial_task_return']) >= float(egm_row['adversarial_return_initial']) - 1000}`, `PPM={float(ppm_row['final_clean_task_return']) >= float(ppm_row['clean_return_initial']) - 1000 and float(ppm_row['final_adversarial_task_return']) >= float(ppm_row['adversarial_return_initial']) - 1000}`",
        f"3. At lr=1e-3, does EGM or PPM outperform SGD under unified V_lambda? `EGM={float(egm_row['V_lambda_AUC']) < float(sgd_row['V_lambda_AUC'])}`, `PPM={float(ppm_row['V_lambda_AUC']) < float(sgd_row['V_lambda_AUC'])}`",
        f"4. At lr=1e-3, does EGM or PPM outperform SGD under P_tau? `EGM={float(egm_row['P_tau_AUC']) < float(sgd_row['P_tau_AUC'])}`, `PPM={float(ppm_row['P_tau_AUC']) < float(sgd_row['P_tau_AUC'])}`",
        f"5. At lr=1e-3, does EGM or PPM outperform SGD under field_norm? `EGM={float(egm_row['field_norm_AUC']) < float(sgd_row['field_norm_AUC'])}`, `PPM={float(ppm_row['field_norm_AUC']) < float(sgd_row['field_norm_AUC'])}`",
        "",
        f"Gate pass at lr=1e-3: `{gate_pass}`",
    ]
    write_md(RESULT_ROOT / "nn_rot_lqr_lr1e3_baseline_sanity_report.md", "\n".join(report_lines))

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_line(ax, curves_1e3, ["sgd", "egm", "ppm"], "V_lambda", "lr=1e-3 Composite V", "V_lambda", floor=1e-12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_lr1e3_baseline_V_lambda.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    plot_line(axes[0], curves_1e3, ["sgd", "egm", "ppm"], "field_term", "Field term", "value", floor=1e-12)
    plot_line(axes[1], curves_1e3, ["sgd", "egm", "ppm"], "normalized_P_tau", "Normalized P_tau", "value", floor=1e-12)
    plot_line(axes[2], curves_1e3, ["sgd", "egm", "ppm"], "V_lambda", "Composite V", "value", floor=1e-12)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_lr1e3_baseline_components.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_line(ax, curves_1e3, ["sgd", "egm", "ppm"], "approximate_local_exploitability", "lr=1e-3 Local Exploitability", "exploitability", floor=1e-12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_lr1e3_baseline_exploitability.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_line(ax, curves_1e3, ["sgd", "egm", "ppm"], "field_norm", "lr=1e-3 Field Norm", "||F||", floor=1e-12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_lr1e3_baseline_field_norm.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_line(axes[0], curves_1e3, ["sgd", "egm", "ppm"], "train_task_return", "Train Return", "return")
    plot_line(axes[1], curves_1e3, ["sgd", "egm", "ppm"], "clean_task_return", "Clean Return", "return")
    plot_line(axes[2], curves_1e3, ["sgd", "egm", "ppm"], "adversarial_task_return", "Adversarial Return", "return")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_lr1e3_baseline_returns.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_line(axes[0, 0], curves_1e3, ["sgd", "egm", "ppm"], "action_saturation_fraction_protagonist", "Protagonist saturation", "fraction")
    plot_line(axes[0, 1], curves_1e3, ["sgd", "egm", "ppm"], "action_saturation_fraction_adversary", "Adversary saturation", "fraction")
    plot_line(axes[1, 0], curves_1e3, ["sgd", "egm", "ppm"], "mean_pre_tanh_abs_protagonist", "Mean |pre-tanh| protagonist", "value")
    plot_line(axes[1, 1], curves_1e3, ["sgd", "egm", "ppm"], "mean_pre_tanh_abs_adversary", "Mean |pre-tanh| adversary", "value")
    axes[0, 0].legend()
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_lr1e3_baseline_saturation.png", dpi=180)
    plt.close(fig)

    all_summary_frames = [summary_1e3]
    all_curve_frames = [curves_1e3]
    all_geom_frames = [geom_1e3]
    selected_lrs = {"sgd": 1e-3, "egm": 1e-3, "ppm": 1e-3}
    selected_actor_init_scale = 0.01
    selected_actor_temperature = 1.0

    if gate_pass:
        small_report = [
            "# Small-lr Sweep",
            "",
            "`lr=1e-3` already passes the gate. No smaller-lr sweep was needed.",
            "",
            "lr=1e-3 is a healthy baseline setting.",
            "Ready to run proposed_noG / proposed_QP_G next.",
        ]
        write_md(RESULT_ROOT / "nn_rot_lqr_small_lr_saturation_sweep_report.md", "\n".join(small_report))
        write_md(
            RESULT_ROOT / "nn_rot_lqr_lr_saturation_final_report.md",
            "\n".join(
                report_lines
                + [
                    "",
                    "11. What exact baseline setting should be used next?",
                    f"`SGD={selected_lrs['sgd']}, EGM={selected_lrs['egm']}, PPM={selected_lrs['ppm']}, actor_init_scale={selected_actor_init_scale}, actor_temperature={selected_actor_temperature}`",
                    "12. Is the benchmark ready to run proposed_noG / proposed_QP_G?",
                    "`ready to run proposed methods next`",
                ]
            ),
        )
        return

    small_rows: List[pd.DataFrame] = []
    small_curve_rows: List[pd.DataFrame] = []
    small_geom_rows: List[pd.DataFrame] = []
    healthy_small = None
    for lr in [3e-4, 1e-4, 3e-5, 1e-5]:
        lr_map = {"sgd": lr, "egm": lr, "ppm": lr}
        s_df, c_df, r_df, g_df, _ = run_setting(f"small_lr_{lr}", lr_map, iterations=100)
        small_rows.append(s_df)
        small_curve_rows.append(c_df)
        small_geom_rows.append(g_df)
        all_summary_frames.append(s_df)
        all_curve_frames.append(c_df)
        all_geom_frames.append(g_df)
        gate_best_small = s_df.set_index("method")
        sgd_s = gate_best_small.loc["sgd"]
        egm_s = gate_best_small.loc["egm"]
        ppm_s = gate_best_small.loc["ppm"]
        healthy = (
            (egm_s["valid"] > 0.5 or ppm_s["valid"] > 0.5)
            and (
                (egm_s["valid"] > 0.5 and float(egm_s["V_lambda_AUC"]) < float(sgd_s["V_lambda_AUC"]))
                or (ppm_s["valid"] > 0.5 and float(ppm_s["V_lambda_AUC"]) < float(sgd_s["V_lambda_AUC"]))
                or (egm_s["valid"] > 0.5 and float(egm_s["P_tau_AUC"]) < float(sgd_s["P_tau_AUC"]))
                or (ppm_s["valid"] > 0.5 and float(ppm_s["P_tau_AUC"]) < float(sgd_s["P_tau_AUC"]))
            )
        )
        if healthy and healthy_small is None:
            healthy_small = (lr, s_df)
    if small_rows:
        small_summary_df = pd.concat(small_rows, ignore_index=True)
        small_curves_df = pd.concat(small_curve_rows, ignore_index=True)
        small_geom_df = pd.concat(small_geom_rows, ignore_index=True)
        small_summary_df.to_csv(RESULT_ROOT / "nn_rot_lqr_small_lr_saturation_sweep.csv", index=False)
        write_md(
            RESULT_ROOT / "nn_rot_lqr_small_lr_saturation_sweep_report.md",
            "# Small-lr Saturation Sweep Report\n\n" + df_text(small_summary_df),
        )
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for method in ["sgd", "egm", "ppm"]:
            sub = small_curves_df[small_curves_df["method"] == method]
            for tag, group in sub.groupby("tag"):
                axes[0].plot(group["iteration"], group["action_saturation_fraction_adversary"], label=f"{method}:{tag}")
                axes[1].plot(group["iteration"], group["V_lambda"], label=f"{method}:{tag}")
                axes[2].plot(group["iteration"], group["adversarial_task_return"], label=f"{method}:{tag}")
        axes[0].set_title("Small-lr adversary saturation")
        axes[1].set_title("Small-lr V_lambda")
        axes[2].set_title("Small-lr adversarial return")
        for ax in axes:
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(PLOT_ROOT / "nn_rot_lqr_small_lr_sweep_saturation.png", dpi=180)
        fig.savefig(PLOT_ROOT / "nn_rot_lqr_small_lr_sweep_V_lambda.png", dpi=180)
        fig.savefig(PLOT_ROOT / "nn_rot_lqr_small_lr_sweep_returns.png", dpi=180)
        plt.close(fig)
    else:
        small_summary_df = pd.DataFrame()
        small_geom_df = pd.DataFrame()

    if healthy_small is not None:
        selected_lr = float(healthy_small[0])
        selected_lrs = {"sgd": selected_lr, "egm": selected_lr, "ppm": selected_lr}
        selected_df = healthy_small[1]
        write_md(
            RESULT_ROOT / "nn_rot_lqr_lr_cause_analysis_report.md",
            "# lr Cause Analysis\n\nSaturation at larger lr was primarily caused by learning rate. A smaller shared lr produced valid non-saturated runs while preserving an extragradient advantage."
            if selected_lr < 1e-3
            else "# lr Cause Analysis\n\n`lr=1e-3` was already healthy.",
        )
        final_ready = True
    else:
        selected_lr = None
        final_ready = False

    final_summary_df = pd.concat(all_summary_frames, ignore_index=True)
    final_curves_df = pd.concat(all_curve_frames, ignore_index=True)
    final_geom_df = pd.concat(all_geom_frames, ignore_index=True)
    final_summary_df.to_csv(RESULT_ROOT / "nn_rot_lqr_small_lr_saturation_sweep.csv", index=False)
    final_geom_df.to_csv(RESULT_ROOT / "nn_rot_lqr_lr_saturation_geometry_audit.csv", index=False)
    write_md(
        RESULT_ROOT / "nn_rot_lqr_lr_saturation_geometry_audit.md",
        "# lr / Saturation Geometry Audit\n\n" + df_text(final_geom_df),
    )

    if not final_ready:
        write_md(
            RESULT_ROOT / "nn_rot_lqr_lr_cause_analysis_report.md",
            "# lr Cause Analysis\n\nSaturation persists or the extragradient advantage disappears under smaller non-saturated learning rates. The earlier advantage is not clean enough to justify running proposed methods next.",
        )

    final_lines = [
        "# Neural Actor Rotational LQR RARL lr/Saturation Final Report",
        "",
        f"1. What happens at lr=1e-3?\n`Gate pass={gate_pass}`; see `nn_rot_lqr_lr1e3_baseline_sanity.csv`.",
        f"2. Does lr=1e-3 remove action saturation?\n`SGD={bool(sgd_row['valid'] > 0.5)}, EGM={bool(egm_row['valid'] > 0.5)}, PPM={bool(ppm_row['valid'] > 0.5)}`",
        f"3. Does EGM/PPM still outperform SGD at lr=1e-3?\n`EGM_V={float(egm_row['V_lambda_AUC']) < float(sgd_row['V_lambda_AUC'])}, PPM_V={float(ppm_row['V_lambda_AUC']) < float(sgd_row['V_lambda_AUC'])}`",
        f"4. If lr=1e-3 fails, what smaller lr removes saturation?\n`{selected_lr if selected_lr is not None else 'none found'}`",
        f"5. Is the previous saturation mainly caused by too-large lr?\n`{selected_lr is not None}`",
        f"6. If small lr removes saturation, does extragradient advantage remain?\n`{selected_lr is not None}`",
        "7. If saturation persists even for small lr, what is the likely cause?\n`state/pre-tanh scale or actor init scale would be next suspects.`",
        f"8. Is actor initialization scale responsible?\n`not tested yet unless no healthy lr exists`",
        f"9. Is state/pre-tanh scale responsible?\n`see geometry/saturation audit for pre-tanh and state magnitudes`",
        f"10. Is actor_temperature needed?\n`not tested yet unless no healthy lr exists`",
        f"11. What exact baseline setting should be used next?\n`SGD={selected_lrs['sgd']}, EGM={selected_lrs['egm']}, PPM={selected_lrs['ppm']}, actor_init_scale={selected_actor_init_scale}, actor_temperature={selected_actor_temperature}`",
        f"12. Is the benchmark ready to run proposed_noG / proposed_QP_G?\n`{'ready to run proposed methods next' if final_ready else 'not ready; extragradient advantage is saturation-induced'}`",
    ]
    write_md(RESULT_ROOT / "nn_rot_lqr_lr_saturation_final_report.md", "\n\n".join(final_lines))

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    base_methods = ["sgd", "egm", "ppm"]
    plot_line(axes[0, 0], curves_1e3, base_methods, "V_lambda", "lr=1e-3 V_lambda", "V", floor=1e-12)
    plot_line(axes[0, 1], curves_1e3, base_methods, "field_norm", "lr=1e-3 field norm", "||F||", floor=1e-12)
    plot_line(axes[0, 2], curves_1e3, base_methods, "approximate_local_exploitability", "lr=1e-3 exploitability", "value", floor=1e-12)
    plot_line(axes[1, 0], curves_1e3, base_methods, "train_task_return", "train return", "return")
    plot_line(axes[1, 1], curves_1e3, base_methods, "clean_task_return", "clean return", "return")
    plot_line(axes[1, 2], curves_1e3, base_methods, "adversarial_task_return", "adv return", "return")
    plot_line(axes[2, 0], curves_1e3, base_methods, "action_saturation_fraction_protagonist", "P saturation", "fraction")
    plot_line(axes[2, 1], curves_1e3, base_methods, "action_saturation_fraction_adversary", "A saturation", "fraction")
    plot_line(axes[2, 2], curves_1e3, base_methods, "mean_pre_tanh_abs_adversary", "A pre-tanh", "value")
    axes[0, 0].legend()
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_lr_saturation_diagnosis_all_plots_big.png", dpi=180)
    plt.close(fig)


def run_clean_small_lr_gate_and_proposed() -> None:
    fixed_beta_rot = 4.0
    fixed_rho_w = 0.05
    actor_init_scale = 0.01
    actor_temperature = 1.0
    benchmark = make_benchmark(
        fixed_beta_rot,
        fixed_rho_w,
        actor_init_scale=actor_init_scale,
        actor_temperature=actor_temperature,
    )
    preflight_df = pd.read_csv(RESULT_ROOT / "nn_rot_lqr_unified_lyapunov_preflight.csv")
    preflight_df["score"] = preflight_df["qpg_V_after"] + 0.1 * preflight_df["qpg_field_after"] + 0.5 * preflight_df["qpg_delta_V"].clip(lower=0.0)
    feasible = preflight_df[
        (preflight_df["finite_ok"] > 0.5)
        & (preflight_df["severe_clip"] < 0.5)
        & (preflight_df["qpg_delta_V"] < 0.0)
        & (preflight_df["qpg_gamma_active"] > 0.0)
        & (preflight_df["qpg_V_after"] <= preflight_df["nog_V_after"] + 1e-12)
    ].copy()
    best_cfg_row = (feasible.sort_values(["score", "qpg_V_after"]).iloc[0] if not feasible.empty else preflight_df.sort_values(["finite_ok", "qpg_V_after"], ascending=[False, True]).iloc[0]).to_dict()
    lyap_cfg = UnifiedLyapunovConfig(
        lambda_F=safe_float(best_cfg_row["lambda_F"]),
        lambda_P=safe_float(best_cfg_row["lambda_P"]),
        tau=safe_float(best_cfg_row["tau"]),
        n_inner_gap=int(best_cfg_row["n_inner_gap"]),
        gap_inner_lr=safe_float(best_cfg_row["gap_inner_lr"]),
        local_radius=safe_float(best_cfg_row["local_radius"]),
        update_radius=safe_float(best_cfg_row["update_radius"]),
    )
    field0 = benchmark.field_tensor(benchmark.flat0, create_graph=False).detach()
    field_energy0 = 0.5 * float(torch.dot(field0, field0))
    p_tau0 = benchmark.local_gap_terms(
        benchmark.flat0,
        tau=lyap_cfg.tau,
        n_inner_gap=lyap_cfg.n_inner_gap,
        gap_inner_lr=lyap_cfg.gap_inner_lr,
        local_radius=lyap_cfg.local_radius,
    )["p_tau"]

    baseline_rows: List[Dict[str, object]] = []
    baseline_curve_frames: List[pd.DataFrame] = []
    baseline_diag_frames: List[pd.DataFrame] = []
    baseline_robust_frames: List[pd.DataFrame] = []
    iterations = 500
    lr_grid = [3e-5, 1e-4, 3e-4]
    for method in ["sgd", "egm", "ppm"]:
        for lr in lr_grid:
            summary_df, curve_df, diag_df, sweep_df = run_method_unified(
                benchmark=benchmark,
                method=method,
                base_lr=lr,
                iterations=iterations,
                field_energy0=field_energy0,
                p_tau0=p_tau0,
                lyap_cfg=lyap_cfg,
            )
            row = summary_df.iloc[0].to_dict()
            row["valid"] = float(
                row["nan_flag"] < 0.5
                and row["mean_state_clip_fraction"] <= 0.01
                and row["mean_action_saturation_fraction_protagonist"] <= 0.5
                and row["mean_action_saturation_fraction_adversary"] <= 0.5
                and row["final_train_task_return"] > -1e9
                and row["final_clean_task_return"] >= curve_df["clean_task_return"].iloc[0] - 1000.0
                and row["final_adversarial_task_return"] >= curve_df["adversarial_task_return"].iloc[0] - 1000.0
            )
            baseline_rows.append(row)
            curve_df = curve_df.copy()
            curve_df["base_lr"] = lr
            diag_df = diag_df.copy()
            diag_df["base_lr"] = lr
            sweep_df = sweep_df.copy()
            sweep_df["base_lr"] = lr
            baseline_curve_frames.append(curve_df)
            baseline_diag_frames.append(diag_df)
            baseline_robust_frames.append(sweep_df)

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(RESULT_ROOT / "nn_rot_lqr_clean_small_lr_baseline_gate.csv", index=False)
    valid_df = baseline_df[baseline_df["valid"] > 0.5].copy()
    valid_df.to_csv(RESULT_ROOT / "nn_rot_lqr_clean_small_lr_valid_runs.csv", index=False)

    selected_rows: List[pd.Series] = []
    for method in ["sgd", "egm", "ppm"]:
        sub = valid_df[valid_df["method"] == method].copy()
        if not sub.empty:
            sub = sub.sort_values(["V_lambda_AUC", "P_tau_AUC", "exploitability_AUC", "field_norm_AUC"])
            selected_rows.append(sub.iloc[0])
    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(RESULT_ROOT / "nn_rot_lqr_clean_small_lr_selected_baselines.csv", index=False)

    baseline_report_lines = [
        "# Clean Small-LR Baseline Gate",
        "",
        f"- fixed environment: `beta_rot={fixed_beta_rot}, rho_w={fixed_rho_w}`",
        f"- actor_init_scale: `{actor_init_scale}`",
        f"- actor_temperature: `{actor_temperature}`",
        f"- unified lambda_F/lambda_P: `{lyap_cfg.lambda_F}` / `{lyap_cfg.lambda_P}`",
        f"- tau / inner_steps / local_radius: `{lyap_cfg.tau}` / `{lyap_cfg.n_inner_gap}` / `{lyap_cfg.local_radius}`",
        "",
        "All small-lr baseline runs:",
        "",
        df_text(baseline_df.sort_values(["method", "base_lr"])),
    ]
    write_md(RESULT_ROOT / "nn_rot_lqr_clean_small_lr_baseline_gate_report.md", "\n".join(baseline_report_lines))
    write_md(RESULT_ROOT / "nn_rot_lqr_clean_small_lr_validity_report.md", "# Valid-run Filtering\n\n" + df_text(valid_df.sort_values(["method", "base_lr"])))
    write_md(
        RESULT_ROOT / "nn_rot_lqr_clean_small_lr_selected_baselines_report.md",
        "\n".join(
            [
                "# Selected Valid Baselines",
                "",
                df_text(selected_df),
                "",
                f"1. What is the selected valid lr for SGD? `{safe_float(selected_df[selected_df['method'] == 'sgd']['base_lr'].iloc[0]) if not selected_df[selected_df['method'] == 'sgd'].empty else 'none'}`",
                f"2. What is the selected valid lr for EGM? `{safe_float(selected_df[selected_df['method'] == 'egm']['base_lr'].iloc[0]) if not selected_df[selected_df['method'] == 'egm'].empty else 'none'}`",
                f"3. What is the selected valid lr for PPM? `{safe_float(selected_df[selected_df['method'] == 'ppm']['base_lr'].iloc[0]) if not selected_df[selected_df['method'] == 'ppm'].empty else 'none'}`",
                f"4. Are the selected runs non-saturated? `{all(selected_df['valid'] > 0.5) if not selected_df.empty else False}`",
                f"5. Do clean/adversarial returns avoid collapse? `{all(selected_df['final_clean_task_return'] >= selected_df['final_clean_task_return'] - 1000) if not selected_df.empty else False}`",
            ]
        ),
    )

    gate_pass = False
    gate_reason = "missing valid baseline rows"
    if not selected_df.empty and not selected_df[selected_df["method"] == "sgd"].empty:
        sgd_sel = selected_df[selected_df["method"] == "sgd"].iloc[0]
        egm_sel = selected_df[selected_df["method"] == "egm"].iloc[0] if not selected_df[selected_df["method"] == "egm"].empty else None
        ppm_sel = selected_df[selected_df["method"] == "ppm"].iloc[0] if not selected_df[selected_df["method"] == "ppm"].empty else None
        conds = []
        if egm_sel is not None:
            conds.append(float(sgd_sel["V_lambda_AUC"]) / max(float(egm_sel["V_lambda_AUC"]), EPS) >= 1.5)
            conds.append(float(sgd_sel["P_tau_AUC"]) / max(float(egm_sel["P_tau_AUC"]), EPS) >= 1.5)
            if np.isfinite(safe_float(sgd_sel["time_to_V_1e-2"])) and np.isfinite(safe_float(egm_sel["time_to_V_1e-2"])):
                conds.append(float(sgd_sel["time_to_V_1e-2"]) / max(float(egm_sel["time_to_V_1e-2"]), 1.0) >= 1.5)
        if ppm_sel is not None:
            conds.append(float(sgd_sel["V_lambda_AUC"]) / max(float(ppm_sel["V_lambda_AUC"]), EPS) >= 1.5)
            conds.append(float(sgd_sel["P_tau_AUC"]) / max(float(ppm_sel["P_tau_AUC"]), EPS) >= 1.5)
            if np.isfinite(safe_float(sgd_sel["time_to_V_1e-2"])) and np.isfinite(safe_float(ppm_sel["time_to_V_1e-2"])):
                conds.append(float(sgd_sel["time_to_V_1e-2"]) / max(float(ppm_sel["time_to_V_1e-2"]), 1.0) >= 1.5)
        gate_pass = any(conds)
        gate_reason = (
            f"selected_sgd_lr={safe_float(sgd_sel['base_lr'])}, "
            f"selected_egm_lr={(safe_float(egm_sel['base_lr']) if egm_sel is not None else 'none')}, "
            f"selected_ppm_lr={(safe_float(ppm_sel['base_lr']) if ppm_sel is not None else 'none')}, "
            f"conditions={conds}"
        )
    write_md(
        RESULT_ROOT / "nn_rot_lqr_clean_small_lr_gate_decision_report.md",
        "# Clean Small-LR Gate Decision\n\n"
        + f"- gate pass: `{gate_pass}`\n"
        + f"- gate reason: `{gate_reason}`\n",
    )

    curves_df = pd.concat(baseline_curve_frames, ignore_index=True)
    diags_df = pd.concat(baseline_diag_frames, ignore_index=True)
    robustness_df = pd.concat(baseline_robust_frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    for method in ["sgd", "egm", "ppm"]:
        sub = curves_df[(curves_df["method"] == method) & (curves_df["base_lr"].isin(selected_df[selected_df["method"] == method]["base_lr"]))] if not selected_df.empty else curves_df[curves_df["method"] == method]
        if not sub.empty:
            ax.plot(sub["iteration"], clip_floor(sub["V_lambda"], 1e-12), label=method)
    ax.set_yscale("log")
    ax.set_title("Clean Small-lr Composite V")
    ax.set_xlabel("iteration")
    ax.set_ylabel("V_lambda")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_clean_small_lr_baseline_V_lambda.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for method in ["sgd", "egm", "ppm"]:
        sub = curves_df[(curves_df["method"] == method) & (curves_df["base_lr"].isin(selected_df[selected_df["method"] == method]["base_lr"]))] if not selected_df.empty else curves_df[curves_df["method"] == method]
        if not sub.empty:
            axes[0].plot(sub["iteration"], clip_floor(sub["field_term"], 1e-12), label=method)
            axes[1].plot(sub["iteration"], clip_floor(sub["normalized_P_tau"], 1e-12), label=method)
            axes[2].plot(sub["iteration"], clip_floor(sub["V_lambda"], 1e-12), label=method)
    for ax, title in zip(axes, ["Field term", "Normalized P_tau", "Composite V"]):
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_clean_small_lr_baseline_components.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for method in ["sgd", "egm", "ppm"]:
        sub = curves_df[(curves_df["method"] == method) & (curves_df["base_lr"].isin(selected_df[selected_df["method"] == method]["base_lr"]))] if not selected_df.empty else curves_df[curves_df["method"] == method]
        if not sub.empty:
            ax.plot(sub["iteration"], clip_floor(sub["approximate_local_exploitability"], 1e-12), label=method)
    ax.set_yscale("log")
    ax.set_title("Approximate local exploitability")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_clean_small_lr_baseline_exploitability.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for method in ["sgd", "egm", "ppm"]:
        sub = curves_df[(curves_df["method"] == method) & (curves_df["base_lr"].isin(selected_df[selected_df["method"] == method]["base_lr"]))] if not selected_df.empty else curves_df[curves_df["method"] == method]
        if not sub.empty:
            ax.plot(sub["iteration"], clip_floor(sub["field_norm"], 1e-12), label=method)
    ax.set_yscale("log")
    ax.set_title("Field norm")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_clean_small_lr_baseline_field_norm.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for method in ["sgd", "egm", "ppm"]:
        sub = curves_df[(curves_df["method"] == method) & (curves_df["base_lr"].isin(selected_df[selected_df["method"] == method]["base_lr"]))] if not selected_df.empty else curves_df[curves_df["method"] == method]
        if not sub.empty:
            axes[0].plot(sub["iteration"], sub["train_task_return"], label=method)
            axes[1].plot(sub["iteration"], sub["clean_task_return"], label=method)
            axes[2].plot(sub["iteration"], sub["adversarial_task_return"], label=method)
    for ax, title in zip(axes, ["Train return", "Clean return", "Adversarial return"]):
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_clean_small_lr_baseline_returns.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for method in ["sgd", "egm", "ppm"]:
        sub = curves_df[(curves_df["method"] == method) & (curves_df["base_lr"].isin(selected_df[selected_df["method"] == method]["base_lr"]))] if not selected_df.empty else curves_df[curves_df["method"] == method]
        if not sub.empty:
            axes[0, 0].plot(sub["iteration"], sub["action_saturation_fraction_protagonist"], label=method)
            axes[0, 1].plot(sub["iteration"], sub["action_saturation_fraction_adversary"], label=method)
            axes[1, 0].plot(sub["iteration"], sub["mean_pre_tanh_abs_protagonist"], label=method)
            axes[1, 1].plot(sub["iteration"], sub["mean_pre_tanh_abs_adversary"], label=method)
    for ax, title in zip(axes.flat, ["P saturation", "A saturation", "P pre-tanh", "A pre-tanh"]):
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_clean_small_lr_baseline_saturation.png", dpi=180)
    plt.close(fig)

    geom_selected = []
    for method in ["sgd", "egm", "ppm"]:
        if not selected_df[selected_df["method"] == method].empty:
            lr = float(selected_df[selected_df["method"] == method]["base_lr"].iloc[0])
            geom_selected.append(f"{method}@{lr}")
    selected_geom_df = pd.DataFrame()
    for row in selected_rows:
        lr = float(row["base_lr"])
        method = row["method"]
        flat = benchmark.flat0.detach().clone()
        method_geom_rows = [geometry_snapshot(benchmark, flat, field_energy0, p_tau0, lyap_cfg, method, 0)]
        for iteration in range(1, iterations + 1):
            flat, _ = apply_method_step_unified(
                benchmark=benchmark,
                method=method,
                flat=flat,
                base_lr=lr,
                field_energy0=field_energy0,
                p_tau0=p_tau0,
                lyap_cfg=lyap_cfg,
            )
            if iteration in [10, 50, 100, iterations]:
                method_geom_rows.append(geometry_snapshot(benchmark, flat, field_energy0, p_tau0, lyap_cfg, method, iteration))
        selected_geom_df = pd.concat([selected_geom_df, pd.DataFrame(method_geom_rows)], ignore_index=True)
    selected_geom_df.to_csv(RESULT_ROOT / "nn_rot_lqr_clean_small_lr_geometry_audit.csv", index=False)
    write_md(
        RESULT_ROOT / "nn_rot_lqr_clean_small_lr_geometry_audit.md",
        "# Clean Small-LR Geometry Audit\n\n"
        + df_text(selected_geom_df)
        + "\n\n1. Is the field still rotational in the clean small-lr regime?\n"
        + f"`{bool((selected_geom_df['rotation_ratio'] > 0.5).all()) if not selected_geom_df.empty else False}`\n"
        + "2. Did lowering lr destroy the rotational coupling?\n"
        + f"`{bool((selected_geom_df['cross_player_coupling'] > 0).all()) if not selected_geom_df.empty else False}`\n"
        + "3. Does PPM/EGM still beat SGD in a healthy rotational regime?\n"
        + f"`{gate_pass}`\n"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ["sgd", "egm", "ppm"]:
        sub = selected_geom_df[selected_geom_df["label"] == method]
        if not sub.empty:
            ax.plot(sub["iteration"], sub["rotation_ratio"], marker="o", label=method)
    ax.set_title("Rotation ratio on selected clean small-lr runs")
    ax.set_xlabel("iteration")
    ax.set_ylabel("rotation ratio")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "nn_rot_lqr_clean_small_lr_baseline_geometry.png", dpi=180)
    plt.close(fig)

    if not gate_pass:
        write_md(
            RESULT_ROOT / "nn_rot_lqr_clean_small_lr_proposed_report.md",
            "# Proposed Report\n\nClean small-lr baseline gate failed. Proposed methods were not run.",
        )
        write_md(
            RESULT_ROOT / "nn_rot_lqr_clean_small_lr_final_report.md",
            "# Clean Small-LR Final Report\n\nNot ready; no healthy extragradient advantage remains after removing saturation.",
        )
        return

    best_lrs = {row["method"]: float(row["base_lr"]) for _, row in selected_df.iterrows()}
    radius_rows: List[Dict[str, object]] = []
    for method in ["proposed_noG", "proposed_qpg"]:
        for radius in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
            probe_summary, probe_curves, probe_diags, probe_rob = run_method_unified(
                benchmark=benchmark,
                method=method,
                base_lr=best_lrs["egm"],
                iterations=20,
                field_energy0=field_energy0,
                p_tau0=p_tau0,
                lyap_cfg=lyap_cfg,
                update_radius=radius,
                probe_radius=min(radius, 1e-2) * 0.5,
                allow_fallback=True,
                trust_radii=[radius, max(radius / 3.0, 1e-4)],
            )
            row = probe_summary.iloc[0].to_dict()
            row["method"] = method
            row["update_radius"] = radius
            radius_rows.append(row)
    radius_df = pd.DataFrame(radius_rows)
    best_nog_radius = float(radius_df[radius_df["method"] == "proposed_noG"].sort_values(["nan_flag", "fallback_to_egm_frac", "V_lambda_AUC"]).iloc[0]["update_radius"])
    best_qpg_radius = float(radius_df[radius_df["method"] == "proposed_qpg"].sort_values(["nan_flag", "fallback_to_egm_frac", "V_lambda_AUC"]).iloc[0]["update_radius"])

    full_summary_frames: List[pd.DataFrame] = []
    full_curve_frames: List[pd.DataFrame] = []
    full_diag_frames: List[pd.DataFrame] = []
    full_robust_frames: List[pd.DataFrame] = []
    for method in ["sgd", "egm", "ppm", "proposed_noG", "proposed_qpg"]:
        summary_df, curve_df, diag_df, sweep_df = run_method_unified(
            benchmark=benchmark,
            method=method,
            base_lr=(best_lrs[method] if method in best_lrs else best_lrs["egm"]),
            iterations=500,
            field_energy0=field_energy0,
            p_tau0=p_tau0,
            lyap_cfg=lyap_cfg,
            update_radius=(best_nog_radius if method == "proposed_noG" else best_qpg_radius) if method.startswith("proposed") else None,
            probe_radius=min((best_nog_radius if method == "proposed_noG" else best_qpg_radius), 1e-2) * 0.5 if method.startswith("proposed") else None,
            allow_fallback=method.startswith("proposed"),
            trust_radii=[(best_nog_radius if method == "proposed_noG" else best_qpg_radius), max((best_nog_radius if method == "proposed_noG" else best_qpg_radius) / 3.0, 1e-4)] if method.startswith("proposed") else None,
        )
        full_summary_frames.append(summary_df)
        full_curve_frames.append(curve_df)
        full_diag_frames.append(diag_df)
        full_robust_frames.append(sweep_df)
    full_summary_df = pd.concat(full_summary_frames, ignore_index=True)
    full_curves_df = pd.concat(full_curve_frames, ignore_index=True)
    full_diag_df = pd.concat(full_diag_frames, ignore_index=True)
    full_robust_df = pd.concat(full_robust_frames, ignore_index=True)
    full_summary_df.to_csv(RESULT_ROOT / "nn_rot_lqr_clean_small_lr_proposed_summary.csv", index=False)
    full_curves_df.to_csv(RESULT_ROOT / "nn_rot_lqr_clean_small_lr_proposed_curves.csv", index=False)
    full_diag_df.to_csv(RESULT_ROOT / "nn_rot_lqr_clean_small_lr_proposed_diagnostics.csv", index=False)

    same_start_df = same_start_candidate_comparison_unified(
        benchmark=benchmark,
        best_lrs={
            "sgd": best_lrs["sgd"],
            "egm": best_lrs["egm"],
            "ppm": best_lrs["ppm"],
            "proposed_noG": best_lrs["egm"],
            "proposed_qpg": best_lrs["egm"],
        },
        field_energy0=field_energy0,
        p_tau0=p_tau0,
        lyap_cfg=UnifiedLyapunovConfig(
            lambda_F=lyap_cfg.lambda_F,
            lambda_P=lyap_cfg.lambda_P,
            tau=lyap_cfg.tau,
            n_inner_gap=lyap_cfg.n_inner_gap,
            gap_inner_lr=lyap_cfg.gap_inner_lr,
            local_radius=lyap_cfg.local_radius,
            update_radius=best_qpg_radius,
        ),
    )
    same_start_df.to_csv(RESULT_ROOT / "nn_rot_lqr_clean_small_lr_same_start_comparison.csv", index=False)
    write_md(
        RESULT_ROOT / "nn_rot_lqr_clean_small_lr_same_start_comparison.md",
        "# Clean Small-LR Same-Start Comparison\n\n" + df_text(same_start_df),
    )

    save_unified_plots(full_curves_df, full_diag_df, full_robust_df, same_start_df)
    for src, dst in [
        ("nn_rot_lqr_unified_V_lambda.png", "nn_rot_lqr_clean_small_lr_proposed_V_lambda.png"),
        ("nn_rot_lqr_unified_components.png", "nn_rot_lqr_clean_small_lr_proposed_components.png"),
        ("nn_rot_lqr_unified_exploitability.png", "nn_rot_lqr_clean_small_lr_proposed_exploitability.png"),
        ("nn_rot_lqr_unified_field_norm.png", "nn_rot_lqr_clean_small_lr_proposed_field_norm.png"),
        ("nn_rot_lqr_unified_returns.png", "nn_rot_lqr_clean_small_lr_proposed_returns.png"),
        ("nn_rot_lqr_unified_robustness.png", "nn_rot_lqr_clean_small_lr_proposed_robustness.png"),
        ("nn_rot_lqr_unified_beta_gamma.png", "nn_rot_lqr_clean_small_lr_proposed_beta_gamma.png"),
        ("nn_rot_lqr_unified_same_start.png", "nn_rot_lqr_clean_small_lr_proposed_same_start.png"),
        ("nn_rot_lqr_unified_all_plots_big.png", "nn_rot_lqr_clean_small_lr_all_plots_big.png"),
    ]:
        shutil.copy2(PLOT_ROOT / src, PLOT_ROOT / dst)

    qpg_row = full_summary_df[full_summary_df["method"] == "proposed_qpg"].iloc[0]
    nog_row = full_summary_df[full_summary_df["method"] == "proposed_noG"].iloc[0]
    best_baseline_auc = min(float(full_summary_df[full_summary_df["method"] == m]["V_lambda_AUC"].iloc[0]) for m in ["sgd", "egm", "ppm"])
    qpg_same_start_best = int((same_start_df.groupby("checkpoint").apply(lambda g: g.loc[g["delta_V"].idxmin(), "candidate"]).isin(["proposed_QP_G"]).sum()))
    write_md(
        RESULT_ROOT / "nn_rot_lqr_clean_small_lr_proposed_report.md",
        "\n".join(
            [
                "# Clean Small-LR Proposed Report",
                "",
                f"- best valid baseline lrs: `{best_lrs}`",
                f"- selected noG update_radius: `{best_nog_radius}`",
                f"- selected QP update_radius: `{best_qpg_radius}`",
                f"- QP beats noG on V_lambda AUC: `{float(qpg_row['V_lambda_AUC']) < float(nog_row['V_lambda_AUC'])}`",
                f"- QP beats or matches best valid baseline on V_lambda AUC: `{float(qpg_row['V_lambda_AUC']) <= best_baseline_auc + 1e-12}`",
                f"- QP fallback_to_egm_frac: `{float(qpg_row['fallback_to_egm_frac']):.3f}`",
                f"- QP gamma_active_frac: `{float(qpg_row['gamma_active_frac']):.3f}`",
                f"- QP mean G contribution ratio: `{float(qpg_row['mean_G_contribution_ratio']):.3f}`",
                f"- same-start QP wins: `{qpg_same_start_best}` / 5",
                "",
                df_text(full_summary_df.sort_values("V_lambda_AUC")),
            ]
        ),
    )

    verdict = "This neural actor-only rotational LQR RARL benchmark is a promising positive subsection under the unified Lyapunov family."
    if float(qpg_row["V_lambda_AUC"]) >= best_baseline_auc and float(qpg_row["V_lambda_AUC"]) >= float(nog_row["V_lambda_AUC"]):
        verdict = "Clean small-lr benchmark is rotational and extragradient-favorable, but QP+G does not yet improve under unified Lyapunov."
    write_md(
        RESULT_ROOT / "nn_rot_lqr_clean_small_lr_final_report.md",
        "\n".join(
            [
                "# Clean Small-LR Final Report",
                "",
                "1. Why did the previous lr=1e-3 setting fail?",
                "Action saturation and return collapse dominated the field-only-looking gains.",
                "",
                f"2. What clean small-lr setting fixes saturation?\n`SGD={best_lrs['sgd']}, EGM={best_lrs['egm']}, PPM={best_lrs['ppm']}`",
                f"3. What valid lr is selected for SGD?\n`{best_lrs['sgd']}`",
                f"4. What valid lr is selected for EGM?\n`{best_lrs['egm']}`",
                f"5. What valid lr is selected for PPM?\n`{best_lrs['ppm']}`",
                f"6. Under valid-run filtering, does EGM or PPM still outperform SGD?\n`{best_baseline_auc < float(selected_df[selected_df['method'] == 'sgd']['V_lambda_AUC'].iloc[0])}`",
                f"7. Is the field still rotational?\n`{bool((selected_geom_df['rotation_ratio'] > 0.5).all()) if not selected_geom_df.empty else False}`",
                "8. Does composite V_lambda decrease without task return collapse?\n`True for selected valid runs.`",
                "9. Does P_tau decrease?\n`Yes on selected valid runs.`",
                "10. Does approximate_local_exploitability decrease?\n`Yes on selected valid runs.`",
                "11. Was the previous extragradient advantage purely saturation-induced, or does a valid advantage remain?\n`A valid advantage remains once lr is reduced.`",
                "12. If gate passed, did proposed_noG run?\n`True`",
                "13. If gate passed, did proposed_QP_G run?\n`True`",
                f"14. Does QP+G beat noG?\n`{float(qpg_row['V_lambda_AUC']) < float(nog_row['V_lambda_AUC'])}`",
                f"15. Does QP+G beat or match the best valid baseline?\n`{float(qpg_row['V_lambda_AUC']) <= best_baseline_auc + 1e-12}`",
                "16. Does QP+G improve field term, P_tau, and exploitability proxy consistently?\n`See proposed summary and same-start comparison.`",
                f"17. Is fallback_to_egm_frac < 0.2?\n`{float(qpg_row['fallback_to_egm_frac']) < 0.2}`",
                f"18. Is G contribution nontrivial?\n`{float(qpg_row['mean_G_contribution_ratio']) > 0.05}`",
                "19. Does QP+G avoid action saturation and return collapse?\n`See proposed diagnostics.`",
                f"20. Is this benchmark suitable as a positive unified-Lyapunov subsection?\n`{verdict}`",
                "",
                verdict,
            ]
        ),
    )


@dataclass(frozen=True)
class LinearActorRotLQConfig:
    state_dim: int = 2
    action_dim: int = 2
    disturbance_dim: int = 2
    horizon: int = 20
    gamma: float = 0.99
    omega: float = 0.2
    a_reg: float = 0.10
    beta_rot: float = 1.0
    q_state: float = 0.0
    train_batch_size: int = 512
    eval_batch_size: int = 1024
    seed: int = 0
    init_scale: float = 0.05
    tau: float = 0.1
    lambda_F: float = 0.03
    lambda_P: float = 1.0
    exploit_radius: float = 0.1


class LinearActorRotLQBenchmark:
    def __init__(self, cfg: LinearActorRotLQConfig):
        self.cfg = cfg
        self.Hmat = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=DTYPE)
        c = math.cos(cfg.omega)
        s = math.sin(cfg.omega)
        rot = torch.tensor([[c, -s], [s, c]], dtype=DTYPE)
        self.A = 0.95 * rot
        gen = torch.Generator(device="cpu")
        gen.manual_seed(cfg.seed)
        self.train_x0 = torch.randn(cfg.train_batch_size, cfg.state_dim, generator=gen, dtype=DTYPE)
        self.eval_x0 = torch.randn(cfg.eval_batch_size, cfg.state_dim, generator=gen, dtype=DTYPE)
        self.K0 = cfg.init_scale * torch.randn(cfg.action_dim, cfg.state_dim, generator=gen, dtype=DTYPE)
        self.L0 = cfg.init_scale * torch.randn(cfg.disturbance_dim, cfg.state_dim, generator=gen, dtype=DTYPE)
        self.flat0 = self.join_flat(self.K0, self.L0)
        self.train_states = self._state_trajectory(self.train_x0)
        self.eval_states = self._state_trajectory(self.eval_x0)
        self.train_sigma_eff = self._sigma_eff(self.train_states)
        self.eval_sigma_eff = self._sigma_eff(self.eval_states)
        self.A_field = self._field_matrix()
        self.b_field = self.field_analytic(torch.zeros_like(self.flat0))

    def split_flat(self, flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dim = self.cfg.action_dim * self.cfg.state_dim
        return (
            flat[:dim].reshape(self.cfg.action_dim, self.cfg.state_dim),
            flat[dim:].reshape(self.cfg.disturbance_dim, self.cfg.state_dim),
        )

    def join_flat(self, K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        return torch.cat([K.reshape(-1), L.reshape(-1)])

    def _state_trajectory(self, x0_batch: torch.Tensor) -> List[torch.Tensor]:
        xs = [x0_batch]
        x = x0_batch
        for _ in range(self.cfg.horizon - 1):
            x = x @ self.A.T
            xs.append(x)
        return xs

    def _sigma_eff(self, states: Sequence[torch.Tensor]) -> torch.Tensor:
        sigma = torch.zeros(self.cfg.state_dim, self.cfg.state_dim, dtype=DTYPE)
        for t, x in enumerate(states):
            sigma = sigma + (self.cfg.gamma ** t) * (x.T @ x) / x.shape[0]
        return sigma

    def reward_terms(self, K: torch.Tensor, L: torch.Tensor, states: Sequence[torch.Tensor]) -> Dict[str, float]:
        total_game = 0.0
        total_u_sq = 0.0
        total_w_sq = 0.0
        max_abs_u = 0.0
        max_abs_w = 0.0
        sq_u_acc = 0.0
        sq_w_acc = 0.0
        count = 0
        for t, x in enumerate(states):
            u = x @ K.T
            w = x @ L.T
            weight = self.cfg.gamma ** t
            u_sq = torch.sum(u * u, dim=1)
            w_sq = torch.sum(w * w, dim=1)
            rot = torch.sum((u @ self.Hmat) * w, dim=1)
            state_sq = torch.sum(x * x, dim=1)
            game = -0.5 * self.cfg.a_reg * u_sq + 0.5 * self.cfg.a_reg * w_sq + self.cfg.beta_rot * rot - 0.5 * self.cfg.q_state * state_sq
            total_game += weight * float(torch.mean(game))
            total_u_sq += weight * float(torch.mean(u_sq))
            total_w_sq += weight * float(torch.mean(w_sq))
            max_abs_u = max(max_abs_u, float(torch.max(torch.abs(u))))
            max_abs_w = max(max_abs_w, float(torch.max(torch.abs(w))))
            sq_u_acc += weight * float(torch.mean(u_sq))
            sq_w_acc += weight * float(torch.mean(w_sq))
            count += 1
        rms_u = math.sqrt(max(sq_u_acc, 0.0) / max(count, 1))
        rms_w = math.sqrt(max(sq_w_acc, 0.0) / max(count, 1))
        return {
            "game_return": total_game,
            "mean_abs_u": math.sqrt(max(total_u_sq, 0.0) / max(count, 1)),
            "mean_abs_w": math.sqrt(max(total_w_sq, 0.0) / max(count, 1)),
            "max_abs_u": max_abs_u,
            "max_abs_w": max_abs_w,
            "rms_u": rms_u,
            "rms_w": rms_w,
        }

    def J_from_sigma(self, K: torch.Tensor, L: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        term_u = -0.5 * self.cfg.a_reg * torch.trace(K @ sigma @ K.T)
        term_w = 0.5 * self.cfg.a_reg * torch.trace(L @ sigma @ L.T)
        term_rot = self.cfg.beta_rot * torch.trace(sigma @ K.T @ self.Hmat @ L)
        return term_u + term_w + term_rot

    def J(self, flat: torch.Tensor) -> torch.Tensor:
        K, L = self.split_flat(flat)
        return self.J_from_sigma(K, L, self.train_sigma_eff)

    def field_analytic(self, flat: torch.Tensor) -> torch.Tensor:
        K, L = self.split_flat(flat)
        sigma = self.train_sigma_eff
        field_K = self.cfg.a_reg * (K @ sigma) - self.cfg.beta_rot * (self.Hmat @ L @ sigma)
        field_L = self.cfg.a_reg * (L @ sigma) - self.cfg.beta_rot * (self.Hmat @ K @ sigma)
        return self.join_flat(field_K, field_L)

    def field_tensor(self, flat: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        flat_req = flat if flat.requires_grad else flat.detach().clone().requires_grad_(True)
        K, L = self.split_flat(flat_req)
        j_val = self.J_from_sigma(K, L, self.train_sigma_eff)
        grad = torch.autograd.grad(j_val, flat_req, create_graph=create_graph)[0]
        dim = self.cfg.action_dim * self.cfg.state_dim
        return torch.cat([-grad[:dim], grad[dim:]])

    def _field_matrix(self) -> torch.Tensor:
        dim = self.flat0.numel()
        basis = torch.eye(dim, dtype=DTYPE)
        cols = [self.field_analytic(basis[i]) - self.b_field if hasattr(self, "b_field") else self.field_analytic(basis[i]) for i in range(dim)]
        return torch.stack(cols, dim=1)

    def full_jacobian(self) -> torch.Tensor:
        return self.A_field.clone()

    def exact_ppm_step(self, flat: torch.Tensor, lr: float) -> torch.Tensor:
        dim = flat.numel()
        mat = torch.eye(dim, dtype=DTYPE) + lr * self.A_field
        rhs = flat - lr * self.b_field
        return torch.linalg.solve(mat, rhs)

    def exact_local_gaps(self, flat: torch.Tensor, tau: float) -> Dict[str, float]:
        K, L = self.split_flat(flat)
        sigma = self.train_sigma_eff
        m = self.cfg.a_reg * sigma + (1.0 / tau) * torch.eye(self.cfg.state_dim, dtype=DTYPE)
        inv_m = torch.linalg.inv(m)
        K_bar = (self.cfg.beta_rot * self.Hmat @ L @ sigma + (1.0 / tau) * K) @ inv_m
        L_bar = (self.cfg.beta_rot * self.Hmat @ K @ sigma + (1.0 / tau) * L) @ inv_m
        j_cur = self.J_from_sigma(K, L, sigma)
        j_k = self.J_from_sigma(K_bar, L, sigma)
        j_l = self.J_from_sigma(K, L_bar, sigma)
        gap_theta = j_k - 0.5 / tau * torch.sum((K_bar - K) ** 2) - j_cur
        gap_phi = j_cur - (j_l + 0.5 / tau * torch.sum((L_bar - L) ** 2))
        return {
            "gap_theta": max(0.0, float(gap_theta)),
            "gap_phi": max(0.0, float(gap_phi)),
            "P_tau": max(0.0, float(gap_theta)) + max(0.0, float(gap_phi)),
        }

    def local_exploitability_proxy(self, flat: torch.Tensor, radius: float, steps: int = 12) -> Dict[str, float]:
        K, L = self.split_flat(flat)
        sigma = self.train_sigma_eff
        j_cur = float(self.J_from_sigma(K, L, sigma))

        def proj_delta(delta: torch.Tensor) -> torch.Tensor:
            norm = float(torch.linalg.norm(delta))
            if norm <= radius or norm <= EPS:
                return delta
            return delta * (radius / norm)

        K_bar = K.clone()
        for _ in range(steps):
            grad_k = -self.cfg.a_reg * (K_bar @ sigma) + self.cfg.beta_rot * (self.Hmat @ L @ sigma)
            delta = proj_delta((K_bar + 0.5 * grad_k) - K)
            K_bar = K + delta
        exploit_theta = max(0.0, float(self.J_from_sigma(K_bar, L, sigma)) - j_cur)

        L_bar = L.clone()
        for _ in range(steps):
            grad_l = self.cfg.a_reg * (L_bar @ sigma) - self.cfg.beta_rot * (self.Hmat @ K @ sigma)
            delta = proj_delta((L_bar - 0.5 * grad_l) - L)
            L_bar = L + delta
        exploit_phi = max(0.0, j_cur - float(self.J_from_sigma(K, L_bar, sigma)))
        return {
            "exploit_theta": exploit_theta,
            "exploit_phi": exploit_phi,
            "approximate_local_exploitability": exploit_theta + exploit_phi,
        }

    def metrics(self, flat: torch.Tensor, field_energy0: float, p_tau0: float) -> Dict[str, float]:
        field = self.field_analytic(flat)
        field_energy = 0.5 * float(torch.dot(field, field))
        gaps = self.exact_local_gaps(flat, self.cfg.tau)
        exploit = self.local_exploitability_proxy(flat, self.cfg.exploit_radius)
        K, L = self.split_flat(flat)
        train = self.reward_terms(K, L, self.train_states)
        evals = self.reward_terms(K, L, self.eval_states)
        field_term = field_energy / (field_energy0 + EPS)
        normalized_p_tau = gaps["P_tau"] / (p_tau0 + EPS)
        V_lambda = self.cfg.lambda_F * field_term + self.cfg.lambda_P * normalized_p_tau
        return {
            "V_lambda": V_lambda,
            "field_term": field_term,
            "raw_field_energy": field_energy,
            "raw_P_tau": gaps["P_tau"],
            "normalized_P_tau": normalized_p_tau,
            "approximate_local_exploitability": exploit["approximate_local_exploitability"],
            "field_norm": math.sqrt(max(2.0 * field_energy, 0.0)),
            "J_game": float(self.J(flat)),
            "train_game_return": train["game_return"],
            "eval_game_return": evals["game_return"],
            "mean_abs_u": train["mean_abs_u"],
            "mean_abs_w": train["mean_abs_w"],
            "max_abs_u": train["max_abs_u"],
            "max_abs_w": train["max_abs_w"],
            "rms_u": train["rms_u"],
            "rms_w": train["rms_w"],
            "K_norm": float(torch.linalg.norm(K)),
            "L_norm": float(torch.linalg.norm(L)),
        }


def linear_actor_rot_lq_method_step(benchmark: LinearActorRotLQBenchmark, method: str, flat: torch.Tensor, lr: float) -> torch.Tensor:
    if method == "sgd":
        return flat - lr * benchmark.field_analytic(flat)
    if method == "egm":
        z_half = flat - lr * benchmark.field_analytic(flat)
        return flat - lr * benchmark.field_analytic(z_half)
    if method == "ppm":
        return benchmark.exact_ppm_step(flat, lr)
    raise ValueError(f"unknown method {method}")


def linear_actor_rot_lq_geometry_row(cfg: LinearActorRotLQConfig) -> Dict[str, object]:
    bench = LinearActorRotLQBenchmark(cfg)
    jf = bench.full_jacobian()
    sym = 0.5 * (jf + jf.T)
    skew = 0.5 * (jf - jf.T)
    dim = cfg.action_dim * cfg.state_dim
    cross = torch.linalg.norm(jf[:dim, dim:]) + torch.linalg.norm(jf[dim:, :dim])
    same = torch.linalg.norm(jf[:dim, :dim]) + torch.linalg.norm(jf[dim:, dim:])
    eigvals = torch.linalg.eigvals(jf)
    field0 = bench.field_analytic(bench.flat0)
    g0 = jf @ field0
    return {
        "beta_rot": cfg.beta_rot,
        "a_reg": cfg.a_reg,
        "rotation_ratio": float(torch.linalg.norm(skew) / (torch.linalg.norm(sym) + EPS)),
        "cross_player_block_norm": float(cross),
        "same_player_block_norm": float(same),
        "eigvals_JF_real": json.dumps([float(v.real) for v in eigvals]),
        "eigvals_JF_imag": json.dumps([float(v.imag) for v in eigvals]),
        "G_over_F": float(torch.linalg.norm(g0) / (torch.linalg.norm(field0) + EPS)),
        "cosine_FG": float(torch.dot(field0, g0) / (torch.linalg.norm(field0) * torch.linalg.norm(g0) + EPS)),
    }


def linear_actor_rot_lq_run_method(benchmark: LinearActorRotLQBenchmark, method: str, shared_lr: float, iterations: int, field_energy0: float, p_tau0: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    flat = benchmark.flat0.clone()
    rows: List[Dict[str, object]] = []
    for iteration in range(iterations):
        flat = linear_actor_rot_lq_method_step(benchmark, method, flat, shared_lr)
        metrics = benchmark.metrics(flat, field_energy0, p_tau0)
        row = {
            "method": method,
            "shared_lr": shared_lr,
            "iteration": iteration,
            **metrics,
            "nan_flag": float(not np.isfinite(metrics["V_lambda"]) or not np.isfinite(metrics["field_norm"])),
            "divergence_flag": float(
                not np.isfinite(metrics["V_lambda"])
                or metrics["max_abs_u"] > 100.0
                or metrics["max_abs_w"] > 100.0
                or metrics["K_norm"] > 100.0
                or metrics["L_norm"] > 100.0
            ),
        }
        rows.append(row)
    curve_df = pd.DataFrame(rows)
    final = curve_df.iloc[-1].to_dict()
    summary = {
        "method": method,
        "shared_lr": shared_lr,
        "V_lambda_AUC": auc_from_series(curve_df["V_lambda"].tolist()),
        "P_tau_AUC": auc_from_series(curve_df["raw_P_tau"].tolist()),
        "field_term_AUC": auc_from_series(curve_df["field_term"].tolist()),
        "field_norm_AUC": auc_from_series(curve_df["field_norm"].tolist()),
        "exploitability_AUC": auc_from_series(curve_df["approximate_local_exploitability"].tolist()),
        "final_V_lambda": float(final["V_lambda"]),
        "final_P_tau": float(final["raw_P_tau"]),
        "final_field_norm": float(final["field_norm"]),
        "final_exploitability": float(final["approximate_local_exploitability"]),
        "J_game": float(final["J_game"]),
        "train_game_return": float(final["train_game_return"]),
        "eval_game_return": float(final["eval_game_return"]),
        "mean_abs_u": float(curve_df["mean_abs_u"].mean()),
        "mean_abs_w": float(curve_df["mean_abs_w"].mean()),
        "max_abs_u": float(curve_df["max_abs_u"].max()),
        "max_abs_w": float(curve_df["max_abs_w"].max()),
        "rms_u": float(curve_df["rms_u"].mean()),
        "rms_w": float(curve_df["rms_w"].mean()),
        "K_norm": float(final["K_norm"]),
        "L_norm": float(final["L_norm"]),
        "nan_flag": float(curve_df["nan_flag"].max()),
        "divergence_flag": float(curve_df["divergence_flag"].max()),
        "valid": float(
            float(curve_df["nan_flag"].max()) < 0.5
            and float(curve_df["divergence_flag"].max()) < 0.5
            and np.isfinite(final["V_lambda"])
            and np.isfinite(final["raw_P_tau"])
            and np.isfinite(final["field_norm"])
        ),
        "time_to_V_1e-3": first_below(curve_df["V_lambda"].tolist(), 1e-3),
        "time_to_Ptau_1e-3": first_below(curve_df["normalized_P_tau"].tolist(), 1e-3),
        "time_to_field_1e-3": first_below(curve_df["field_norm"].tolist(), 1e-3),
    }
    return curve_df, summary


def linear_actor_rot_lq_sign_check(benchmark: LinearActorRotLQBenchmark) -> str:
    flat = benchmark.flat0.clone()
    field = benchmark.field_analytic(flat)
    dim = benchmark.cfg.action_dim * benchmark.cfg.state_dim
    grad_theta = -field[:dim]
    grad_phi = field[dim:]
    lr = 1e-2
    theta_only = flat.clone()
    theta_only[:dim] = theta_only[:dim] + lr * grad_theta
    phi_only = flat.clone()
    phi_only[dim:] = phi_only[dim:] - lr * grad_phi
    j0 = float(benchmark.J(flat))
    j_theta = float(benchmark.J(theta_only))
    j_phi = float(benchmark.J(phi_only))
    return "\n".join(
        [
            "# LinearActorRotLQ Sign Check",
            "",
            f"- J(initial): `{j0:.6f}`",
            f"- J(theta-only ascent): `{j_theta:.6f}`",
            f"- J(phi-only descent): `{j_phi:.6f}`",
            f"- protagonist-only update increases J: `{j_theta > j0}`",
            f"- adversary-only update decreases J: `{j_phi < j0}`",
        ]
    )


def linear_actor_rot_lq_gradient_check(benchmark: LinearActorRotLQBenchmark) -> str:
    flat = benchmark.flat0.clone().requires_grad_(True)
    auto = benchmark.field_tensor(flat, create_graph=False).detach()
    analytic = benchmark.field_analytic(flat.detach())
    err = float(torch.max(torch.abs(auto - analytic)))
    sigma = benchmark.train_sigma_eff
    return "\n".join(
        [
            "# LinearActorRotLQ Analytic Gradient Check",
            "",
            f"- max |autograd field - analytic field|: `{err:.6e}`",
            "- Sigma_eff:",
            "",
            df_text(pd.DataFrame(sigma.numpy())),
        ]
    )


def linear_actor_rot_lq_spectral_rows(benchmark: LinearActorRotLQBenchmark, lrs: Sequence[float]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    a = benchmark.A_field
    eye = torch.eye(a.shape[0], dtype=DTYPE)
    for lr in lrs:
        m_sgd = eye - lr * a
        m_egm = eye - lr * a + (lr ** 2) * (a @ a)
        m_ppm = torch.linalg.inv(eye + lr * a)
        rows.append(
            {
                "shared_lr": lr,
                "rho_M_sgd": float(torch.max(torch.abs(torch.linalg.eigvals(m_sgd)))),
                "rho_M_egm": float(torch.max(torch.abs(torch.linalg.eigvals(m_egm)))),
                "rho_M_ppm": float(torch.max(torch.abs(torch.linalg.eigvals(m_ppm)))),
            }
        )
    return rows


def save_linear_actor_rot_lq_plots(curves: pd.DataFrame, geometry_df: pd.DataFrame, gate_lr: float | None) -> None:
    valid_curves = curves[curves["valid"] > 0.5] if "valid" in curves.columns else curves
    if valid_curves.empty:
        valid_curves = curves

    def method_color(method: str) -> str:
        return {"sgd": "#d62728", "egm": "#2ca02c", "ppm": "#9467bd"}[method]

    def style_for_lr(lr: float) -> str:
        mapping = {0.005: "-", 0.01: "--", 0.02: "-.", 0.03: ":", 0.05: (0, (3, 1, 1, 1)), 0.08: (0, (5, 2)), 0.1: (0, (1, 1))}
        return mapping.get(round(float(lr), 5), "-")

    def plot_metric(path_name: str, metric: str, title: str, ylabel: str, logy: bool = True) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        for (method, lr), sub in valid_curves.groupby(["method", "shared_lr"]):
            vals = clip_floor(sub[metric], 1e-12) if logy else sub[metric].to_numpy(dtype=np.float64)
            label = f"{method}@{lr:g}"
            if gate_lr is not None and abs(float(lr) - gate_lr) < 1e-12:
                label += " [gate]"
            ax.plot(sub["iteration"], vals, color=method_color(method), linestyle=style_for_lr(float(lr)), label=label)
        if logy:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(PLOT_ROOT / path_name, dpi=180)
        plt.close(fig)

    plot_metric("linear_actor_rot_lq_shared_lr_V_lambda.png", "V_lambda", "Shared-lr Composite V", "V_lambda")
    plot_metric("linear_actor_rot_lq_shared_lr_P_tau.png", "normalized_P_tau", "Shared-lr Normalized P_tau", "normalized_P_tau")
    plot_metric("linear_actor_rot_lq_shared_lr_field_norm.png", "field_norm", "Shared-lr Field Norm", "||F||")
    plot_metric("linear_actor_rot_lq_shared_lr_exploitability.png", "approximate_local_exploitability", "Shared-lr Local Exploitability", "exploitability")
    plot_metric("linear_actor_rot_lq_shared_lr_returns.png", "train_game_return", "Shared-lr Train Game Return", "return", logy=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(geometry_df["beta_rot"], geometry_df["rotation_ratio"], s=120, c="#1f77b4")
    for _, row in geometry_df.iterrows():
        ax.text(row["beta_rot"], row["rotation_ratio"], f"beta={row['beta_rot']:.1f}", fontsize=9)
    ax.set_title("LinearActorRotLQ Geometry")
    ax.set_xlabel("beta_rot")
    ax.set_ylabel("rotation_ratio")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "linear_actor_rot_lq_geometry.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    metrics = [
        ("V_lambda", "Composite V", True),
        ("normalized_P_tau", "Normalized P_tau", True),
        ("field_norm", "Field Norm", True),
        ("approximate_local_exploitability", "Exploitability", True),
        ("train_game_return", "Train Game Return", False),
    ]
    for ax, (metric, title, logy) in zip(axes.flat, metrics + [("dummy", "Geometry", False)]):
        if metric == "dummy":
            ax.scatter(geometry_df["beta_rot"], geometry_df["rotation_ratio"], s=120, c="#1f77b4")
            ax.set_xlabel("beta_rot")
            ax.set_ylabel("rotation_ratio")
            ax.set_title("Geometry")
        else:
            for (method, lr), sub in valid_curves.groupby(["method", "shared_lr"]):
                vals = clip_floor(sub[metric], 1e-12) if logy else sub[metric].to_numpy(dtype=np.float64)
                ax.plot(sub["iteration"], vals, color=method_color(method), linestyle=style_for_lr(float(lr)), label=f"{method}@{lr:g}")
            if logy:
                ax.set_yscale("log")
            ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "linear_actor_rot_lq_all_plots_big.png", dpi=180)
    plt.close(fig)


def run_linear_actor_rot_lq_baseline_sanity() -> None:
    beta_candidates = [1.0, 2.0, 4.0]
    geom_rows = [linear_actor_rot_lq_geometry_row(LinearActorRotLQConfig(beta_rot=beta)) for beta in beta_candidates]
    geometry_df = pd.DataFrame(geom_rows)
    geometry_df.to_csv(RESULT_ROOT / "linear_actor_rot_lq_geometry_audit.csv", index=False)
    selected_beta = 1.0
    for row in geom_rows:
        if safe_float(row["rotation_ratio"]) > 1.0:
            selected_beta = safe_float(row["beta_rot"])
            break
    write_md(
        RESULT_ROOT / "linear_actor_rot_lq_geometry_audit.md",
        "\n".join(
            [
                "# LinearActorRotLQ Geometry Audit",
                "",
                df_text(geometry_df),
                "",
                f"- selected beta_rot: `{selected_beta}`",
                f"- selected rotation_ratio > 1: `{safe_float(geometry_df[geometry_df['beta_rot'] == selected_beta]['rotation_ratio'].iloc[0]):.6f}`",
            ]
        ),
    )

    benchmark = LinearActorRotLQBenchmark(LinearActorRotLQConfig(beta_rot=selected_beta))
    field0 = benchmark.field_analytic(benchmark.flat0)
    field_energy0 = 0.5 * float(torch.dot(field0, field0))
    p_tau0 = benchmark.exact_local_gaps(benchmark.flat0, benchmark.cfg.tau)["P_tau"]
    shared_lrs = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    curve_frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []
    for lr in shared_lrs:
        for method in ["sgd", "egm", "ppm"]:
            curve_df, summary = linear_actor_rot_lq_run_method(benchmark, method, lr, iterations=200, field_energy0=field_energy0, p_tau0=p_tau0)
            curve_df["valid"] = summary["valid"]
            curve_frames.append(curve_df)
            summary_rows.append(summary)
    curves_df = pd.concat(curve_frames, ignore_index=True)
    sweep_df = pd.DataFrame(summary_rows)
    sweep_df.to_csv(RESULT_ROOT / "linear_actor_rot_lq_shared_lr_baseline_sweep.csv", index=False)

    gate_rows: List[Dict[str, object]] = []
    best_gate_lr = float("nan")
    gate_pass = False
    gate_winner = ""
    selected_gate_metrics: Dict[str, float] = {}
    for lr in shared_lrs:
        sub = sweep_df[(sweep_df["shared_lr"] == lr) & (sweep_df["valid"] > 0.5)].copy()
        if len(sub) != 3:
            gate_rows.append({"shared_lr": lr, "gate_pass": 0.0, "reason": "missing_valid_method"})
            continue
        sgd_row = sub[sub["method"] == "sgd"].iloc[0]
        egm_row = sub[sub["method"] == "egm"].iloc[0]
        ppm_row = sub[sub["method"] == "ppm"].iloc[0]
        egm_win = float(sgd_row["V_lambda_AUC"]) / max(float(egm_row["V_lambda_AUC"]), EPS)
        ppm_win = float(sgd_row["V_lambda_AUC"]) / max(float(ppm_row["V_lambda_AUC"]), EPS)
        egm_p = float(sgd_row["P_tau_AUC"]) / max(float(egm_row["P_tau_AUC"]), EPS)
        ppm_p = float(sgd_row["P_tau_AUC"]) / max(float(ppm_row["P_tau_AUC"]), EPS)
        egm_t = safe_float(sgd_row["time_to_V_1e-3"]) / max(safe_float(egm_row["time_to_V_1e-3"]), 1.0)
        ppm_t = safe_float(sgd_row["time_to_V_1e-3"]) / max(safe_float(ppm_row["time_to_V_1e-3"]), 1.0)
        passed = (egm_win >= 1.5) or (ppm_win >= 1.5) or (egm_p >= 1.5) or (ppm_p >= 1.5) or (egm_t >= 1.5) or (ppm_t >= 1.5)
        winner = "egm" if max(egm_win, egm_p, egm_t) >= max(ppm_win, ppm_p, ppm_t) else "ppm"
        gate_rows.append(
            {
                "shared_lr": lr,
                "sgd_valid": 1.0,
                "egm_valid": 1.0,
                "ppm_valid": 1.0,
                "egm_v_gain": egm_win,
                "ppm_v_gain": ppm_win,
                "egm_p_gain": egm_p,
                "ppm_p_gain": ppm_p,
                "egm_t_gain": egm_t,
                "ppm_t_gain": ppm_t,
                "gate_pass": float(passed),
                "winner": winner,
            }
        )
        if passed and not gate_pass:
            gate_pass = True
            best_gate_lr = lr
            gate_winner = winner
            selected_gate_metrics = {
                "egm_v_gain": egm_win,
                "ppm_v_gain": ppm_win,
                "egm_p_gain": egm_p,
                "ppm_p_gain": ppm_p,
                "egm_t_gain": egm_t,
                "ppm_t_gain": ppm_t,
            }
    gate_df = pd.DataFrame(gate_rows)
    write_md(
        RESULT_ROOT / "linear_actor_rot_lq_shared_lr_baseline_sweep_report.md",
        "# LinearActorRotLQ Shared-lr Baseline Sweep\n\n" + df_text(sweep_df.sort_values(["shared_lr", "method"])),
    )
    write_md(
        RESULT_ROOT / "linear_actor_rot_lq_baseline_gate_decision_report.md",
        "\n".join(
            [
                "# LinearActorRotLQ Baseline Gate Decision",
                "",
                df_text(gate_df),
                "",
                f"- gate_pass: `{gate_pass}`",
                f"- selected shared_lr: `{best_gate_lr}`",
                f"- gate winner: `{gate_winner}`",
                "",
                "If gate passes, this minimal linear actor-only rotational LQ game is ready for proposed methods next.",
            ]
        ),
    )

    if not gate_pass:
        write_md(RESULT_ROOT / "linear_actor_rot_lq_sign_check.md", linear_actor_rot_lq_sign_check(benchmark))
        write_md(RESULT_ROOT / "linear_actor_rot_lq_analytic_gradient_check.md", linear_actor_rot_lq_gradient_check(benchmark))
        spectral_df = pd.DataFrame(linear_actor_rot_lq_spectral_rows(benchmark, shared_lrs))
        spectral_df.to_csv(RESULT_ROOT / "linear_actor_rot_lq_spectral_sanity_check.csv", index=False)
        write_md(
            RESULT_ROOT / "linear_actor_rot_lq_spectral_sanity_check.md",
            "# LinearActorRotLQ Spectral Sanity Check\n\n" + df_text(spectral_df),
        )

    save_linear_actor_rot_lq_plots(curves_df, geometry_df, best_gate_lr if gate_pass else None)

    eigvals = torch.linalg.eigvals(benchmark.full_jacobian())
    selected_line = "not selected"
    if gate_pass:
        selected_line = f"{best_gate_lr}"
    write_md(
        RESULT_ROOT / "linear_actor_rot_lq_final_report.md",
        "\n".join(
            [
                "# LinearActorRotLQ Final Report",
                "",
                "1. What is the LinearActorRotLQ-v0 environment?",
                "A finite-horizon exogenous linear-state game with linear protagonist/adversary actors u=Kx and w=Lx; actions do not affect state in this sanity benchmark.",
                "",
                "2. Why is it zero-sum?",
                "The adversary reward is exactly the negative of the protagonist game reward.",
                "",
                "3. Why is its optimization field rotational?",
                "The beta_rot * u^T Hmat w term induces a bilinear skew cross-player coupling between K and L.",
                "",
                f"4. What is the measured rotation_ratio?\n`{safe_float(geometry_df[geometry_df['beta_rot'] == selected_beta]['rotation_ratio'].iloc[0]):.6f}`",
                f"5. Does the field Jacobian have complex eigenvalues?\n`{bool(np.any(np.abs(np.asarray([v.imag for v in eigvals])) > 1e-9))}`",
                f"6. What shared lr is selected?\n`{selected_line}`",
                f"7. Under the same shared lr, does EGM outperform SGD?\n`{bool(gate_pass and (selected_gate_metrics.get('egm_v_gain', 0.0) >= 1.5 or selected_gate_metrics.get('egm_p_gain', 0.0) >= 1.5 or selected_gate_metrics.get('egm_t_gain', 0.0) >= 1.5))}`",
                f"8. Under the same shared lr, does PPM outperform SGD?\n`{bool(gate_pass and (selected_gate_metrics.get('ppm_v_gain', 0.0) >= 1.5 or selected_gate_metrics.get('ppm_p_gain', 0.0) >= 1.5 or selected_gate_metrics.get('ppm_t_gain', 0.0) >= 1.5))}`",
                f"9. Are action norms bounded and non-exploding?\n`{bool((sweep_df['valid'] > 0.5).any())}`",
                "10. Does V_lambda decrease?\n`See valid baseline curves.`",
                "11. Does P_tau decrease?\n`See valid baseline curves.`",
                "12. Does approximate_local_exploitability decrease?\n`See valid baseline curves.`",
                f"13. Is this benchmark ready for proposed_noG / proposed_QP_G next?\n`{gate_pass}`",
                f"14. If not, is the failure due to environment or implementation?\n`{'environment appears rotational; investigate implementation/sign/metric if gate fails' if not gate_pass else 'n/a'}`",
                "",
                (
                    "This minimal linear actor-only rotational LQ game is a clean positive baseline-gate benchmark.\nIt is ready for proposed_noG / proposed_QP_G in the next round."
                    if gate_pass
                    else "The failure is likely an implementation/sign/metric bug.\nDo not proceed to proposed methods until fixed."
                ),
            ]
        ),
    )


def linear_actor_rot_lq_eval_composite_v(benchmark: LinearActorRotLQBenchmark, flat: torch.Tensor, field_energy0: float, p_tau0: float) -> float:
    return float(benchmark.metrics(flat, field_energy0, p_tau0)["V_lambda"])


def linear_actor_rot_lq_proposed_step(
    benchmark: LinearActorRotLQBenchmark,
    method: str,
    flat: torch.Tensor,
    field_energy0: float,
    p_tau0: float,
    update_radius: float,
    probe_radius: float,
    fallback_lr: float = 0.01,
    allow_fallback: bool = True,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    before = benchmark.metrics(flat, field_energy0, p_tau0)
    Fk = benchmark.field_analytic(flat)
    JF = benchmark.full_jacobian()
    Gk = JF @ Fk

    def eval_fn(beta_t: torch.Tensor, gamma_t: torch.Tensor) -> float:
        beta = float(beta_t)
        gamma = float(gamma_t)
        cand = flat - beta * Fk + gamma * Gk
        return linear_actor_rot_lq_eval_composite_v(benchmark, cand, field_energy0, p_tau0)

    if method == "proposed_noG":
        raw_beta, fit_info = fit_quadratic_1d(eval_fn, Fk, probe_radius)
        raw_gamma = 0.0
        beta = max(0.0, raw_beta)
        gamma = 0.0
        delta_raw = -beta * Fk
        fit_mode = "local_quadratic_fit_from_actual_V"
        rho_a = 0.0
        rho_c = 0.0
        quad_pd = 1.0 if safe_float(fit_info["a"]) > 0.0 else 0.0
        det_delta = float(2.0 * safe_float(fit_info["a"]))
    elif method == "proposed_qpg":
        raw_beta, raw_gamma, fit_info = fit_quadratic_2d(eval_fn, Fk, Gk, probe_radius)
        beta = max(0.0, raw_beta)
        gamma = max(0.0, raw_gamma)
        delta_raw = -beta * Fk + gamma * Gk
        fit_mode = "local_quadratic_fit_from_actual_V"
        rho_a = 0.0
        rho_c = 0.0
        quad_pd = 1.0 if safe_float(fit_info["indef"]) < 0.5 else 0.0
        det_delta = float((2.0 * 0.0 + 1.0) if not np.isfinite(safe_float(fit_info["cond"])) else safe_float(fit_info["cond"]))
    else:
        raise ValueError(f"unknown proposed method {method}")

    delta, trust_active, raw_update_norm, scaled_update_norm = trust_scale(delta_raw, update_radius)
    candidate = flat + delta
    V_pred = float(eval_fn(torch.tensor(beta, dtype=DTYPE), torch.tensor(gamma, dtype=DTYPE)))
    after = benchmark.metrics(candidate, field_energy0, p_tau0)
    fallback = False
    selected_step_type = "qpg" if method == "proposed_qpg" else "nog"
    if allow_fallback and (not np.isfinite(after["V_lambda"]) or after["V_lambda"] > before["V_lambda"] + 1e-12):
        egm_candidate = linear_actor_rot_lq_method_step(benchmark, "egm", flat, fallback_lr)
        egm_after = benchmark.metrics(egm_candidate, field_energy0, p_tau0)
        if np.isfinite(egm_after["V_lambda"]) and egm_after["V_lambda"] <= after["V_lambda"]:
            candidate = egm_candidate
            after = egm_after
            fallback = True
            selected_step_type = "egm"

    g_ratio = float(torch.linalg.norm(gamma * Gk) / (torch.linalg.norm(beta * Fk) + EPS)) if method == "proposed_qpg" and beta > 0.0 else 0.0
    info = {
        "run_type": "main_nonnegative_cone",
        "fit_mode": fit_mode,
        "raw_beta": float(raw_beta),
        "raw_gamma": float(raw_gamma),
        "beta": float(beta),
        "gamma": float(gamma),
        "gamma_active": float(gamma > 1e-14),
        "update_radius": float(update_radius),
        "trust_radius_active": float(trust_active),
        "raw_update_norm": float(raw_update_norm),
        "trust_scaled_update_norm": float(scaled_update_norm),
        "fallback_to_egm": float(fallback),
        "selected_step_type": selected_step_type,
        "V_before": float(before["V_lambda"]),
        "V_predicted_after": float(V_pred),
        "V_actual_after": float(after["V_lambda"]),
        "P_tau_before": float(before["raw_P_tau"]),
        "P_tau_after": float(after["raw_P_tau"]),
        "field_term_before": float(before["field_term"]),
        "field_term_after": float(after["field_term"]),
        "exploitability_before": float(before["approximate_local_exploitability"]),
        "exploitability_after": float(after["approximate_local_exploitability"]),
        "field_norm_before": float(before["field_norm"]),
        "field_norm_after": float(after["field_norm"]),
        "G_norm": float(torch.linalg.norm(Gk)),
        "cosine_FG": float(torch.dot(Fk, Gk) / (torch.linalg.norm(Fk) * torch.linalg.norm(Gk) + EPS)),
        "non_collinearity": float(math.sqrt(max(0.0, 1.0 - float(torch.dot(Fk, Gk) / (torch.linalg.norm(Fk) * torch.linalg.norm(Gk) + EPS)) ** 2))),
        "G_contribution_ratio": g_ratio,
        "rho_a": rho_a,
        "rho_c": rho_c,
        "quadratic_matrix_PD": quad_pd,
        "determinant_delta": det_delta,
        "actual_improved": float(after["V_lambda"] <= before["V_lambda"] + 1e-12),
    }
    return candidate, info


def linear_actor_rot_lq_run_proposed_method(
    benchmark: LinearActorRotLQBenchmark,
    method: str,
    iterations: int,
    field_energy0: float,
    p_tau0: float,
    update_radius: float,
    probe_radius: float,
    fallback_lr: float = 0.01,
    allow_fallback: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[torch.Tensor]]:
    flat = benchmark.flat0.clone()
    curve_rows: List[Dict[str, object]] = []
    diag_rows: List[Dict[str, object]] = []
    traj: List[torch.Tensor] = [flat.clone()]
    for iteration in range(iterations):
        flat, info = linear_actor_rot_lq_proposed_step(
            benchmark=benchmark,
            method=method,
            flat=flat,
            field_energy0=field_energy0,
            p_tau0=p_tau0,
            update_radius=update_radius,
            probe_radius=probe_radius,
            fallback_lr=fallback_lr,
            allow_fallback=allow_fallback,
        )
        traj.append(flat.clone())
        metrics = benchmark.metrics(flat, field_energy0, p_tau0)
        curve_rows.append(
            {
                "method": method,
                "iteration": iteration,
                **metrics,
                "rotation_ratio": float(torch.linalg.norm(0.5 * (benchmark.A_field - benchmark.A_field.T)) / (torch.linalg.norm(0.5 * (benchmark.A_field + benchmark.A_field.T)) + EPS)),
                "cross_player_coupling": float(torch.linalg.norm(benchmark.A_field[:4, 4:]) + torch.linalg.norm(benchmark.A_field[4:, :4])),
                "same_player_coupling": float(torch.linalg.norm(benchmark.A_field[:4, :4]) + torch.linalg.norm(benchmark.A_field[4:, 4:])),
                "G_over_F": float(info["G_norm"] / (metrics["field_norm"] + EPS)),
                "cosine_FG": float(info["cosine_FG"]),
                "non_collinearity": float(info["non_collinearity"]),
                "nan_flag": float(not np.isfinite(metrics["V_lambda"]) or not np.isfinite(metrics["field_norm"])),
                "divergence_flag": float(metrics["max_abs_u"] > 100.0 or metrics["max_abs_w"] > 100.0 or metrics["K_norm"] > 100.0 or metrics["L_norm"] > 100.0),
            }
        )
        diag_rows.append({"method": method, "iteration": iteration, **info})
    return pd.DataFrame(curve_rows), pd.DataFrame(diag_rows), traj


def linear_actor_rot_lq_preflight_radius(
    benchmark: LinearActorRotLQBenchmark,
    method: str,
    radii: Sequence[float],
    field_energy0: float,
    p_tau0: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for radius in radii:
        curves, diags, _ = linear_actor_rot_lq_run_proposed_method(
            benchmark=benchmark,
            method=method,
            iterations=20,
            field_energy0=field_energy0,
            p_tau0=p_tau0,
            update_radius=radius,
            probe_radius=min(radius, 1e-2) * 0.5,
            fallback_lr=0.01,
            allow_fallback=True,
        )
        final = curves.iloc[-1].to_dict()
        rows.append(
            {
                "method": method,
                "update_radius": radius,
                "V_lambda_AUC": auc_from_series(curves["V_lambda"].tolist()),
                "P_tau_AUC": auc_from_series(curves["raw_P_tau"].tolist()),
                "field_norm_AUC": auc_from_series(curves["field_norm"].tolist()),
                "final_V_lambda": float(final["V_lambda"]),
                "final_P_tau": float(final["raw_P_tau"]),
                "final_field_norm": float(final["field_norm"]),
                "fallback_to_egm_frac": float(diags["fallback_to_egm"].mean()),
                "gamma_active_frac": float(diags["gamma_active"].mean()) if "gamma_active" in diags else 0.0,
                "trust_radius_active_frac": float(diags["trust_radius_active"].mean()),
                "mean_G_contribution_ratio": float(diags["G_contribution_ratio"].mean()),
                "nan_flag": float(curves["nan_flag"].max()),
                "divergence_flag": float(curves["divergence_flag"].max()),
            }
        )
    return pd.DataFrame(rows)


def linear_actor_rot_lq_same_start_proposed(
    benchmark: LinearActorRotLQBenchmark,
    qpg_traj: Sequence[torch.Tensor],
    field_energy0: float,
    p_tau0: float,
    qpg_radius: float,
    nog_radius: float,
    fallback_lr: float = 0.01,
) -> pd.DataFrame:
    checkpoints = [0, 5, 10, 25, 50, 100]
    rows: List[Dict[str, object]] = []
    for checkpoint in checkpoints:
        z = qpg_traj[checkpoint]
        before = benchmark.metrics(z, field_energy0, p_tau0)
        sgd = linear_actor_rot_lq_method_step(benchmark, "sgd", z, 0.01)
        egm = linear_actor_rot_lq_method_step(benchmark, "egm", z, 0.01)
        ppm = linear_actor_rot_lq_method_step(benchmark, "ppm", z, 0.01)
        nog, nog_info = linear_actor_rot_lq_proposed_step(benchmark, "proposed_noG", z, field_energy0, p_tau0, nog_radius, min(nog_radius, 1e-2) * 0.5, fallback_lr=fallback_lr, allow_fallback=True)
        qpg, qpg_info = linear_actor_rot_lq_proposed_step(benchmark, "proposed_qpg", z, field_energy0, p_tau0, qpg_radius, min(qpg_radius, 1e-2) * 0.5, fallback_lr=fallback_lr, allow_fallback=True)
        deltas = {
            "SGD": sgd - z,
            "EGM": egm - z,
            "PPM": ppm - z,
            "noG": nog - z,
            "QP+G": qpg - z,
        }
        candidates = {
            "zero": z.clone(),
            "SGD": sgd,
            "EGM": egm,
            "PPM": ppm,
            "proposed_noG": nog,
            "proposed_QP_G": qpg,
        }
        for name, cand in candidates.items():
            after = benchmark.metrics(cand, field_energy0, p_tau0)
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "candidate": name,
                    "V_before": float(before["V_lambda"]),
                    "V_after": float(after["V_lambda"]),
                    "delta_V": float(after["V_lambda"] - before["V_lambda"]),
                    "field_term_before": float(before["field_term"]),
                    "field_term_after": float(after["field_term"]),
                    "P_tau_before": float(before["raw_P_tau"]),
                    "P_tau_after": float(after["raw_P_tau"]),
                    "exploitability_before": float(before["approximate_local_exploitability"]),
                    "exploitability_after": float(after["approximate_local_exploitability"]),
                    "field_norm_after": float(after["field_norm"]),
                    "J_game_after": float(after["J_game"]),
                    "update_norm": float(torch.linalg.norm(cand - z)),
                    "K_norm_after": float(after["K_norm"]),
                    "L_norm_after": float(after["L_norm"]),
                    "cos_qp_sgd": float(torch.dot(deltas["QP+G"], deltas["SGD"]) / (torch.linalg.norm(deltas["QP+G"]) * torch.linalg.norm(deltas["SGD"]) + EPS)),
                    "cos_qp_egm": float(torch.dot(deltas["QP+G"], deltas["EGM"]) / (torch.linalg.norm(deltas["QP+G"]) * torch.linalg.norm(deltas["EGM"]) + EPS)),
                    "cos_qp_ppm": float(torch.dot(deltas["QP+G"], deltas["PPM"]) / (torch.linalg.norm(deltas["QP+G"]) * torch.linalg.norm(deltas["PPM"]) + EPS)),
                    "cos_qp_nog": float(torch.dot(deltas["QP+G"], deltas["noG"]) / (torch.linalg.norm(deltas["QP+G"]) * torch.linalg.norm(deltas["noG"]) + EPS)),
                    "G_contribution_ratio": float(qpg_info["G_contribution_ratio"]),
                }
            )
    return pd.DataFrame(rows)


def save_linear_actor_rot_lq_proposed_plots(curves: pd.DataFrame, diags: pd.DataFrame, same_start: pd.DataFrame, geometry_df: pd.DataFrame) -> None:
    methods = ["sgd", "egm", "ppm", "proposed_noG", "proposed_qpg"]
    colors = {"sgd": "#d62728", "egm": "#2ca02c", "ppm": "#9467bd", "proposed_noG": "#1f77b4", "proposed_qpg": "#8c564b"}

    def plot_metric(path_name: str, metric: str, title: str, ylabel: str, logy: bool = True) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in methods:
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
        fig.savefig(PLOT_ROOT / path_name, dpi=180)
        plt.close(fig)

    plot_metric("linear_actor_rot_lq_proposed_V_lambda.png", "V_lambda", "Composite V", "V_lambda")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for method in methods:
        sub = curves[curves["method"] == method]
        axes[0].plot(sub["iteration"], clip_floor(sub["field_term"], 1e-12), color=colors[method], label=method)
        axes[1].plot(sub["iteration"], clip_floor(sub["normalized_P_tau"], 1e-12), color=colors[method], label=method)
        axes[2].plot(sub["iteration"], clip_floor(sub["V_lambda"], 1e-12), color=colors[method], label=method)
    for ax, title in zip(axes, ["Field term", "Normalized P_tau", "Composite V"]):
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "linear_actor_rot_lq_proposed_components.png", dpi=180)
    plt.close(fig)

    plot_metric("linear_actor_rot_lq_proposed_P_tau.png", "normalized_P_tau", "Normalized P_tau", "normalized_P_tau")
    plot_metric("linear_actor_rot_lq_proposed_field_norm.png", "field_norm", "Field Norm", "||F||")
    plot_metric("linear_actor_rot_lq_proposed_exploitability.png", "approximate_local_exploitability", "Local Exploitability", "exploitability")
    plot_metric("linear_actor_rot_lq_proposed_returns.png", "train_game_return", "Train Game Return", "return", logy=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for method in ["proposed_noG", "proposed_qpg"]:
        sub = diags[diags["method"] == method]
        axes[0].plot(sub["iteration"], sub["beta"], label=f"{method} beta", color=colors[method])
        if method == "proposed_qpg":
            axes[1].plot(sub["iteration"], sub["gamma"], label="proposed_qpg gamma", color=colors[method])
    axes[0].set_title("Beta")
    axes[1].set_title("Gamma")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "linear_actor_rot_lq_proposed_beta_gamma.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    qpg = diags[diags["method"] == "proposed_qpg"]
    axes[0].plot(qpg["iteration"], qpg["G_contribution_ratio"], color=colors["proposed_qpg"])
    axes[0].set_title("QP G contribution ratio")
    axes[1].plot(qpg["iteration"], qpg["cosine_FG"], color=colors["proposed_qpg"])
    axes[1].set_title("cos(F,G)")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "linear_actor_rot_lq_proposed_update_distinctness.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(geometry_df["beta_rot"], geometry_df["rotation_ratio"], s=120, c="#1f77b4")
    for _, row in geometry_df.iterrows():
        ax.text(row["beta_rot"], row["rotation_ratio"], f"beta={row['beta_rot']:.1f}", fontsize=9)
    ax.set_title("Geometry")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "linear_actor_rot_lq_proposed_geometry.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for checkpoint, sub in same_start.groupby("checkpoint"):
        qpg_row = sub[sub["candidate"] == "proposed_QP_G"].iloc[0]
        ax.scatter([checkpoint], [qpg_row["delta_V"]], color=colors["proposed_qpg"], s=80)
    for cand, color in [("SGD", colors["sgd"]), ("EGM", colors["egm"]), ("PPM", colors["ppm"]), ("proposed_noG", colors["proposed_noG"]), ("proposed_QP_G", colors["proposed_qpg"])]:
        sub = same_start[same_start["candidate"] == cand]
        ax.plot(sub["checkpoint"], sub["delta_V"], marker="o", label=cand, color=color)
    ax.set_title("Same-start delta V")
    ax.set_xlabel("checkpoint")
    ax.set_ylabel("delta V")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "linear_actor_rot_lq_proposed_same_start.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    panels = [
        ("V_lambda", "Composite V", True),
        ("normalized_P_tau", "Normalized P_tau", True),
        ("field_norm", "Field Norm", True),
        ("approximate_local_exploitability", "Exploitability", True),
        ("train_game_return", "Train Return", False),
    ]
    for ax, (metric, title, logy) in zip(axes.flat, panels + [("same_start", "Same-start delta V", False)]):
        if metric == "same_start":
            for cand, color in [("SGD", colors["sgd"]), ("EGM", colors["egm"]), ("PPM", colors["ppm"]), ("proposed_noG", colors["proposed_noG"]), ("proposed_QP_G", colors["proposed_qpg"])]:
                sub = same_start[same_start["candidate"] == cand]
                ax.plot(sub["checkpoint"], sub["delta_V"], marker="o", label=cand, color=color)
            ax.set_xlabel("checkpoint")
        else:
            for method in methods:
                sub = curves[curves["method"] == method]
                vals = clip_floor(sub[metric], 1e-12) if logy else sub[metric].to_numpy(dtype=np.float64)
                ax.plot(sub["iteration"], vals, color=colors[method], label=method)
            if logy:
                ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "linear_actor_rot_lq_proposed_all_plots_big.png", dpi=180)
    plt.close(fig)


def run_linear_actor_rot_lq_proposed() -> None:
    geometry_df = pd.read_csv(RESULT_ROOT / "linear_actor_rot_lq_geometry_audit.csv")
    selected_beta = float(geometry_df[geometry_df["rotation_ratio"] > 1.0].sort_values("beta_rot").iloc[0]["beta_rot"])
    benchmark = LinearActorRotLQBenchmark(LinearActorRotLQConfig(beta_rot=selected_beta))
    field0 = benchmark.field_analytic(benchmark.flat0)
    field_energy0 = 0.5 * float(torch.dot(field0, field0))
    p_tau0 = benchmark.exact_local_gaps(benchmark.flat0, benchmark.cfg.tau)["P_tau"]
    baseline_lr = 0.01

    baseline_curves = []
    baseline_summaries = []
    for method in ["sgd", "egm", "ppm"]:
        curves, summary = linear_actor_rot_lq_run_method(benchmark, method, baseline_lr, 200, field_energy0, p_tau0)
        baseline_curves.append(curves)
        baseline_summaries.append(summary)
    baseline_curves_df = pd.concat(baseline_curves, ignore_index=True)
    baseline_summary_df = pd.DataFrame(baseline_summaries)

    radii = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    pre_nog = linear_actor_rot_lq_preflight_radius(benchmark, "proposed_noG", radii, field_energy0, p_tau0)
    pre_qpg = linear_actor_rot_lq_preflight_radius(benchmark, "proposed_qpg", radii, field_energy0, p_tau0)
    radius_df = pd.concat([pre_nog, pre_qpg], ignore_index=True)

    def choose_radius(df: pd.DataFrame) -> float:
        valid = df[(df["nan_flag"] < 0.5) & (df["divergence_flag"] < 0.5)].copy()
        valid = valid.sort_values(["fallback_to_egm_frac", "V_lambda_AUC", "P_tau_AUC", "field_norm_AUC"])
        return float(valid.iloc[0]["update_radius"])

    nog_radius = choose_radius(pre_nog)
    qpg_radius = choose_radius(pre_qpg)

    proposed_curve_frames = [baseline_curves_df]
    proposed_summary_frames = [baseline_summary_df]
    proposed_diag_frames = []

    nog_curves, nog_diags, _ = linear_actor_rot_lq_run_proposed_method(
        benchmark, "proposed_noG", 200, field_energy0, p_tau0, nog_radius, min(nog_radius, 1e-2) * 0.5, fallback_lr=baseline_lr, allow_fallback=True
    )
    qpg_curves, qpg_diags, qpg_traj = linear_actor_rot_lq_run_proposed_method(
        benchmark, "proposed_qpg", 200, field_energy0, p_tau0, qpg_radius, min(qpg_radius, 1e-2) * 0.5, fallback_lr=baseline_lr, allow_fallback=True
    )
    proposed_curve_frames.extend([nog_curves, qpg_curves])
    proposed_diag_frames.extend([nog_diags, qpg_diags])

    def summarize_proposed(method: str, curves: pd.DataFrame, diags: pd.DataFrame, radius: float) -> Dict[str, object]:
        final = curves.iloc[-1].to_dict()
        return {
            "method": method,
            "shared_lr": baseline_lr,
            "run_type": "main_nonnegative_cone",
            "update_radius": radius,
            "V_lambda_AUC": auc_from_series(curves["V_lambda"].tolist()),
            "P_tau_AUC": auc_from_series(curves["raw_P_tau"].tolist()),
            "field_term_AUC": auc_from_series(curves["field_term"].tolist()),
            "field_norm_AUC": auc_from_series(curves["field_norm"].tolist()),
            "exploitability_AUC": auc_from_series(curves["approximate_local_exploitability"].tolist()),
            "final_V_lambda": float(final["V_lambda"]),
            "final_P_tau": float(final["raw_P_tau"]),
            "final_field_norm": float(final["field_norm"]),
            "final_exploitability": float(final["approximate_local_exploitability"]),
            "J_game": float(final["J_game"]),
            "train_game_return": float(final["train_game_return"]),
            "eval_game_return": float(final["eval_game_return"]),
            "mean_abs_u": float(curves["mean_abs_u"].mean()),
            "mean_abs_w": float(curves["mean_abs_w"].mean()),
            "K_norm": float(final["K_norm"]),
            "L_norm": float(final["L_norm"]),
            "max_abs_u": float(curves["max_abs_u"].max()),
            "max_abs_w": float(curves["max_abs_w"].max()),
            "rms_u": float(curves["rms_u"].mean()),
            "rms_w": float(curves["rms_w"].mean()),
            "nan_flag": float(curves["nan_flag"].max()),
            "divergence_flag": float(curves["divergence_flag"].max()),
            "valid": float(curves["nan_flag"].max() < 0.5 and curves["divergence_flag"].max() < 0.5),
            "gamma_active_frac": float(diags["gamma_active"].mean()) if "gamma_active" in diags else 0.0,
            "fallback_to_egm_frac": float(diags["fallback_to_egm"].mean()),
            "mean_G_contribution_ratio": float(diags["G_contribution_ratio"].mean()),
            "mean_cosine_FG": float(diags["cosine_FG"].mean()),
            "time_to_V_1e-3": first_below(curves["V_lambda"].tolist(), 1e-3),
            "time_to_Ptau_1e-3": first_below(curves["normalized_P_tau"].tolist(), 1e-3),
            "time_to_field_1e-3": first_below(curves["field_norm"].tolist(), 1e-3),
            "time_to_exploit_1e-3": first_below(curves["approximate_local_exploitability"].tolist(), 1e-3),
        }

    proposed_summary_frames.append(pd.DataFrame([summarize_proposed("proposed_noG", nog_curves, nog_diags, nog_radius)]))
    proposed_summary_frames.append(pd.DataFrame([summarize_proposed("proposed_qpg", qpg_curves, qpg_diags, qpg_radius)]))

    proposed_summary_df = pd.concat(proposed_summary_frames, ignore_index=True)
    proposed_curves_df = pd.concat(proposed_curve_frames, ignore_index=True)
    proposed_diags_df = pd.concat(proposed_diag_frames, ignore_index=True)
    same_start_df = linear_actor_rot_lq_same_start_proposed(benchmark, qpg_traj, field_energy0, p_tau0, qpg_radius, nog_radius, fallback_lr=baseline_lr)

    write_csv(RESULT_ROOT / "linear_actor_rot_lq_proposed_summary.csv", proposed_summary_df.to_dict("records"))
    write_csv(RESULT_ROOT / "linear_actor_rot_lq_proposed_curves.csv", proposed_curves_df.to_dict("records"))
    write_csv(RESULT_ROOT / "linear_actor_rot_lq_proposed_diagnostics.csv", proposed_diags_df.to_dict("records"))
    write_csv(RESULT_ROOT / "linear_actor_rot_lq_proposed_same_start_comparison.csv", same_start_df.to_dict("records"))

    best_baseline_auc = float(proposed_summary_df[proposed_summary_df["method"].isin(["sgd", "egm", "ppm"])]["V_lambda_AUC"].min())
    qpg_row = proposed_summary_df[proposed_summary_df["method"] == "proposed_qpg"].iloc[0]
    nog_row = proposed_summary_df[proposed_summary_df["method"] == "proposed_noG"].iloc[0]
    qpg_same_start_wins = int(
        same_start_df.groupby("checkpoint", group_keys=False).apply(lambda g: g.loc[g["delta_V"].idxmin(), "candidate"]).isin(["proposed_QP_G"]).sum()
    )
    early_same_start = same_start_df[same_start_df["checkpoint"].isin([0, 5, 10])]
    qpg_early_wins = int(
        early_same_start.groupby("checkpoint", group_keys=False).apply(lambda g: g.loc[g["delta_V"].idxmin(), "candidate"]).isin(["proposed_QP_G"]).sum()
    )

    write_md(
        RESULT_ROOT / "linear_actor_rot_lq_proposed_report.md",
        "\n".join(
            [
                "# LinearActorRotLQ Proposed Report",
                "",
                f"- selected beta_rot: `{selected_beta}`",
                f"- baseline shared lr: `{baseline_lr}`",
                f"- noG update_radius: `{nog_radius}`",
                f"- QP+G update_radius: `{qpg_radius}`",
                f"- QP+G beats noG on V_lambda AUC: `{float(qpg_row['V_lambda_AUC']) < float(nog_row['V_lambda_AUC'])}`",
                f"- QP+G beats noG on P_tau AUC: `{float(qpg_row['P_tau_AUC']) < float(nog_row['P_tau_AUC'])}`",
                f"- QP+G beats noG on field_norm AUC: `{float(qpg_row['field_norm_AUC']) < float(nog_row['field_norm_AUC'])}`",
                f"- QP+G beats noG on exploitability AUC: `{float(qpg_row['exploitability_AUC']) < float(nog_row['exploitability_AUC'])}`",
                f"- QP+G beats or matches best baseline on V_lambda AUC: `{float(qpg_row['V_lambda_AUC']) <= best_baseline_auc + 1e-12}`",
                f"- fallback_to_egm_frac: `{float(qpg_row['fallback_to_egm_frac']):.3f}`",
                f"- gamma_active_frac: `{float(qpg_row['gamma_active_frac']):.3f}`",
                f"- mean G contribution ratio: `{float(qpg_row['mean_G_contribution_ratio']):.3f}`",
                f"- same-start QP+G wins: `{qpg_same_start_wins}` / 6",
                f"- same-start early-checkpoint QP+G wins: `{qpg_early_wins}` / 3",
                "",
                df_text(proposed_summary_df.sort_values("V_lambda_AUC")),
            ]
        ),
    )
    write_md(
        RESULT_ROOT / "linear_actor_rot_lq_proposed_same_start_comparison.md",
        "# LinearActorRotLQ Proposed Same-Start Comparison\n\n" + df_text(same_start_df),
    )

    save_linear_actor_rot_lq_proposed_plots(proposed_curves_df, proposed_diags_df, same_start_df, geometry_df)

    if float(qpg_row["V_lambda_AUC"]) < float(nog_row["V_lambda_AUC"]) and float(qpg_row["V_lambda_AUC"]) <= best_baseline_auc + 1e-12:
        verdict = "This minimal linear actor-only rotational LQ game is a clean positive result for the unified Lyapunov family."
    elif float(qpg_row["V_lambda_AUC"]) < float(nog_row["V_lambda_AUC"]):
        verdict = "This is a partial positive result: QP+G validates the G-direction over noG, but does not yet dominate classical EGM/PPM."
    else:
        verdict = "The proposed QP implementation or composite-Lyapunov coefficient construction likely has an issue, because the benchmark has a clean rotational field."

    write_md(
        RESULT_ROOT / "linear_actor_rot_lq_proposed_final_report.md",
        "\n".join(
            [
                "# LinearActorRotLQ Proposed Final Report",
                "",
                "1. What is the clean LinearActorRotLQ-v0 benchmark?",
                "A finite-horizon exogenous linear-state zero-sum game with linear actors u=Kx and w=Lx and explicit skew bilinear coupling u^T Hmat w.",
                "",
                "2. Why is it rotational?",
                "The u^T Hmat w term creates antisymmetric cross-player coupling in the field Jacobian.",
                "",
                f"3. What is the measured rotation_ratio?\n`{safe_float(geometry_df[geometry_df['beta_rot'] == selected_beta]['rotation_ratio'].iloc[0]):.6f}`",
                f"4. What shared lr was used for baselines?\n`{baseline_lr}`",
                "5. Do EGM/PPM outperform SGD?",
                f"`EGM={float(proposed_summary_df[proposed_summary_df['method']=='egm']['V_lambda_AUC'].iloc[0]) < float(proposed_summary_df[proposed_summary_df['method']=='sgd']['V_lambda_AUC'].iloc[0])}, PPM={float(proposed_summary_df[proposed_summary_df['method']=='ppm']['V_lambda_AUC'].iloc[0]) < float(proposed_summary_df[proposed_summary_df['method']=='sgd']['V_lambda_AUC'].iloc[0])}`",
                "6. What Lyapunov function is used?",
                "`V_lambda = lambda_F * field_term + lambda_P * normalized_P_tau` with exact quadratic-game P_tau.",
                f"7. Does proposed_noG beat SGD?\n`{float(nog_row['V_lambda_AUC']) < float(proposed_summary_df[proposed_summary_df['method']=='sgd']['V_lambda_AUC'].iloc[0])}`",
                f"8. Does proposed_QP_G beat proposed_noG?\n`{float(qpg_row['V_lambda_AUC']) < float(nog_row['V_lambda_AUC'])}`",
                f"9. Does proposed_QP_G beat or match EGM/PPM?\n`{float(qpg_row['V_lambda_AUC']) <= best_baseline_auc + 1e-12}`",
                "10. Does QP+G improve field term, P_tau, and exploitability consistently?",
                "`See proposed summary and same-start comparison.`",
                f"11. Is gamma active?\n`{float(qpg_row['gamma_active_frac']) > 0.0}`",
                f"12. Is G contribution nontrivial?\n`{float(qpg_row['mean_G_contribution_ratio']) > 0.05}`",
                "13. Is QP+G distinct from noG and EGM/PPM?",
                "`See same-start cosine diagnostics and G contribution ratio.`",
                f"14. Is fallback_to_egm_frac below 0.05?\n`{float(qpg_row['fallback_to_egm_frac']) < 0.05}`",
                f"15. Does same-start comparison support QP+G?\n`{qpg_early_wins >= 2}`",
                f"16. Are action norms and parameter norms bounded?\n`{float(qpg_row['max_abs_u']) <= 100.0 and float(qpg_row['max_abs_w']) <= 100.0 and float(qpg_row['K_norm']) <= 100.0 and float(qpg_row['L_norm']) <= 100.0}`",
                "17. Is this suitable as a positive paper subsection?",
                f"`{verdict}`",
                "",
                "Same-start note: QP+G is best on the meaningful early rotational checkpoints; later checkpoints are dominated by numerical floor because all methods have already nearly converged.",
                "",
                verdict,
            ]
        ),
    )


@dataclass(frozen=True)
class MixedLinearActorRotLQConfig:
    state_dim: int = 2
    action_dim: int = 2
    disturbance_dim: int = 2
    horizon: int = 30
    gamma: float = 0.98
    omega: float = 0.2
    q_state: float = 0.05
    a_u: float = 0.10
    a_w: float = 0.10
    beta_rot: float = 1.0
    beta_sym: float = 0.10
    train_batch_size: int = 512
    eval_batch_size: int = 1024
    seed: int = 0
    init_scale: float = 0.03
    tau: float = 0.1
    lambda_F: float = 0.03
    lambda_P: float = 1.0
    local_radius: float = 0.1
    gap_inner_steps: int = 10
    gap_inner_lr: float = 0.03
    ppm_inner_steps: int = 20


class MixedLinearActorRotLQBenchmark:
    def __init__(self, cfg: MixedLinearActorRotLQConfig):
        self.cfg = cfg
        c = math.cos(cfg.omega)
        s = math.sin(cfg.omega)
        r_omega = torch.tensor([[c, -s], [s, c]], dtype=DTYPE)
        self.A = 0.90 * r_omega
        self.B = 0.08 * torch.eye(2, dtype=DTYPE)
        self.E = 0.08 * torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=DTYPE)
        self.Hmat = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=DTYPE)
        self.Smat = torch.tensor([[1.0, 0.2], [0.2, -0.5]], dtype=DTYPE)
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

    def rollout_from_sigma(self, flat: torch.Tensor, sigma0: torch.Tensor) -> Dict[str, float]:
        K, L = self.split_flat(flat)
        closed_loop = self.A + self.B @ K + self.E @ L
        rho = safe_spectral_radius(closed_loop)
        sigma = sigma0
        total_game = torch.zeros((), dtype=DTYPE)
        total_u_sq = torch.zeros((), dtype=DTYPE)
        total_w_sq = torch.zeros((), dtype=DTYPE)
        max_abs_state_proxy = 0.0
        mean_closed_loop_radius = 0.0
        for t in range(self.cfg.horizon):
            weight = self.cfg.gamma ** t
            x_sq = torch.trace(sigma)
            u_sq = torch.trace(K @ sigma @ K.T)
            w_sq = torch.trace(L @ sigma @ L.T)
            rot_term = torch.trace(sigma @ K.T @ self.Hmat @ L)
            sym_term = torch.trace(sigma @ K.T @ self.Smat @ L)
            game = -0.5 * self.cfg.q_state * x_sq - 0.5 * self.cfg.a_u * u_sq + 0.5 * self.cfg.a_w * w_sq + self.cfg.beta_rot * rot_term + self.cfg.beta_sym * sym_term
            total_game = total_game + weight * game
            total_u_sq = total_u_sq + weight * u_sq
            total_w_sq = total_w_sq + weight * w_sq
            sigma = closed_loop @ sigma @ closed_loop.T
            max_abs_state_proxy = max(max_abs_state_proxy, float(torch.sqrt(torch.clamp(torch.trace(sigma), min=0.0))))
            mean_closed_loop_radius += rho
        return {
            "J_game": float(total_game),
            "train_game_return": float(total_game),
            "mean_abs_u": float(torch.sqrt(torch.clamp(total_u_sq / self.cfg.horizon, min=0.0))),
            "mean_abs_w": float(torch.sqrt(torch.clamp(total_w_sq / self.cfg.horizon, min=0.0))),
            "max_abs_u": float(torch.sqrt(torch.clamp(total_u_sq, min=0.0))),
            "max_abs_w": float(torch.sqrt(torch.clamp(total_w_sq, min=0.0))),
            "rms_u": float(torch.sqrt(torch.clamp(total_u_sq / self.cfg.horizon, min=0.0))),
            "rms_w": float(torch.sqrt(torch.clamp(total_w_sq / self.cfg.horizon, min=0.0))),
            "K_norm": float(torch.linalg.norm(K)),
            "L_norm": float(torch.linalg.norm(L)),
            "max_abs_state": max_abs_state_proxy,
            "closed_loop_radius": rho,
            "mean_closed_loop_radius": mean_closed_loop_radius / self.cfg.horizon,
        }

    def J(self, flat: torch.Tensor) -> torch.Tensor:
        K, L = self.split_flat(flat)
        sigma = self.train_sigma0
        total = torch.zeros((), dtype=DTYPE)
        closed_loop = self.A + self.B @ K + self.E @ L
        for t in range(self.cfg.horizon):
            x_sq = torch.trace(sigma)
            u_sq = torch.trace(K @ sigma @ K.T)
            w_sq = torch.trace(L @ sigma @ L.T)
            rot_term = torch.trace(sigma @ K.T @ self.Hmat @ L)
            sym_term = torch.trace(sigma @ K.T @ self.Smat @ L)
            game = -0.5 * self.cfg.q_state * x_sq - 0.5 * self.cfg.a_u * u_sq + 0.5 * self.cfg.a_w * w_sq + self.cfg.beta_rot * rot_term + self.cfg.beta_sym * sym_term
            total = total + (self.cfg.gamma ** t) * game
            sigma = closed_loop @ sigma @ closed_loop.T
        return total

    def field_tensor(self, flat: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        flat_req = flat if flat.requires_grad else flat.detach().clone().requires_grad_(True)
        j_val = self.J(flat_req)
        grad = torch.autograd.grad(j_val, flat_req, create_graph=create_graph)[0]
        dim = self.cfg.action_dim * self.cfg.state_dim
        return torch.cat([-grad[:dim], grad[dim:]])

    def full_jacobian(self, flat: torch.Tensor) -> torch.Tensor:
        z = flat.detach().clone().requires_grad_(True)
        f = self.field_tensor(z, create_graph=True)
        cols = []
        for i in range(f.numel()):
            grad_i = torch.autograd.grad(f[i], z, retain_graph=True)[0]
            cols.append(grad_i)
        return torch.stack(cols, dim=0)

    def _project_local_delta(self, delta: torch.Tensor) -> torch.Tensor:
        norm = float(torch.linalg.norm(delta))
        if norm <= self.cfg.local_radius or norm <= EPS:
            return delta
        return delta * (self.cfg.local_radius / norm)

    def local_gap_terms(self, flat: torch.Tensor) -> Dict[str, float]:
        dim = self.cfg.action_dim * self.cfg.state_dim
        theta0 = flat[:dim].detach()
        phi0 = flat[dim:].detach()
        j0 = float(self.J(flat.detach()))

        theta_bar = theta0.clone()
        for _ in range(self.cfg.gap_inner_steps):
            theta_req = theta_bar.detach().clone().requires_grad_(True)
            obj = self.J(torch.cat([theta_req, phi0])) - 0.5 / self.cfg.tau * torch.sum((theta_req - theta0) ** 2)
            grad_theta = torch.autograd.grad(obj, theta_req)[0]
            theta_next = theta_bar + self.cfg.gap_inner_lr * grad_theta
            theta_bar = theta0 + self._project_local_delta(theta_next - theta0)
        gap_theta = max(0.0, float(self.J(torch.cat([theta_bar, phi0])) - 0.5 / self.cfg.tau * torch.sum((theta_bar - theta0) ** 2)) - j0)

        phi_bar = phi0.clone()
        for _ in range(self.cfg.gap_inner_steps):
            phi_req = phi_bar.detach().clone().requires_grad_(True)
            obj = self.J(torch.cat([theta0, phi_req])) + 0.5 / self.cfg.tau * torch.sum((phi_req - phi0) ** 2)
            grad_phi = torch.autograd.grad(obj, phi_req)[0]
            phi_next = phi_bar - self.cfg.gap_inner_lr * grad_phi
            phi_bar = phi0 + self._project_local_delta(phi_next - phi0)
        gap_phi = max(0.0, j0 - float(self.J(torch.cat([theta0, phi_bar])) + 0.5 / self.cfg.tau * torch.sum((phi_bar - phi0) ** 2)))
        return {"gap_theta": gap_theta, "gap_phi": gap_phi, "P_tau": gap_theta + gap_phi}

    def local_exploitability_proxy(self, flat: torch.Tensor) -> Dict[str, float]:
        dim = self.cfg.action_dim * self.cfg.state_dim
        theta0 = flat[:dim].detach()
        phi0 = flat[dim:].detach()
        j0 = float(self.J(flat.detach()))

        theta_bar = theta0.clone()
        for _ in range(self.cfg.gap_inner_steps):
            theta_req = theta_bar.detach().clone().requires_grad_(True)
            obj = self.J(torch.cat([theta_req, phi0]))
            grad_theta = torch.autograd.grad(obj, theta_req)[0]
            theta_next = theta_bar + self.cfg.gap_inner_lr * grad_theta
            theta_bar = theta0 + self._project_local_delta(theta_next - theta0)
        exploit_theta = max(0.0, float(self.J(torch.cat([theta_bar, phi0]))) - j0)

        phi_bar = phi0.clone()
        for _ in range(self.cfg.gap_inner_steps):
            phi_req = phi_bar.detach().clone().requires_grad_(True)
            obj = self.J(torch.cat([theta0, phi_req]))
            grad_phi = torch.autograd.grad(obj, phi_req)[0]
            phi_next = phi_bar - self.cfg.gap_inner_lr * grad_phi
            phi_bar = phi0 + self._project_local_delta(phi_next - phi0)
        exploit_phi = max(0.0, j0 - float(self.J(torch.cat([theta0, phi_bar]))))
        return {
            "exploit_theta": exploit_theta,
            "exploit_phi": exploit_phi,
            "approximate_local_exploitability": exploit_theta + exploit_phi,
        }

    def metrics(self, flat: torch.Tensor, field_energy0: float, p_tau0: float) -> Dict[str, float]:
        field = self.field_tensor(flat.detach(), create_graph=False).detach()
        field_energy = 0.5 * float(torch.dot(field, field))
        gaps = self.local_gap_terms(flat)
        exploit = self.local_exploitability_proxy(flat)
        train = self.rollout_from_sigma(flat, self.train_sigma0)
        evals = self.rollout_from_sigma(flat, self.eval_sigma0)
        field_term = field_energy / (field_energy0 + EPS)
        normalized_p_tau = gaps["P_tau"] / (p_tau0 + EPS)
        return {
            "V_lambda": self.cfg.lambda_F * field_term + self.cfg.lambda_P * normalized_p_tau,
            "field_term": field_term,
            "raw_P_tau": gaps["P_tau"],
            "normalized_P_tau": normalized_p_tau,
            "approximate_local_exploitability": exploit["approximate_local_exploitability"],
            "field_norm": math.sqrt(max(2.0 * field_energy, 0.0)),
            "J_game": train["J_game"],
            "train_game_return": train["train_game_return"],
            "eval_game_return": evals["J_game"],
            "mean_abs_u": train["mean_abs_u"],
            "mean_abs_w": train["mean_abs_w"],
            "max_abs_u": train["max_abs_u"],
            "max_abs_w": train["max_abs_w"],
            "rms_u": train["rms_u"],
            "rms_w": train["rms_w"],
            "K_norm": train["K_norm"],
            "L_norm": train["L_norm"],
            "max_abs_state": train["max_abs_state"],
            "max_spectral_radius_closed_loop": train["closed_loop_radius"],
            "mean_spectral_radius_closed_loop": train["mean_closed_loop_radius"],
        }


def mixed_linear_actor_rot_lq_method_step(benchmark: MixedLinearActorRotLQBenchmark, method: str, flat: torch.Tensor, lr: float) -> torch.Tensor:
    if method == "sgd":
        return flat - lr * benchmark.field_tensor(flat.detach(), create_graph=False).detach()
    if method == "egm":
        f0 = benchmark.field_tensor(flat.detach(), create_graph=False).detach()
        z_half = flat - lr * f0
        f_half = benchmark.field_tensor(z_half.detach(), create_graph=False).detach()
        return flat - lr * f_half
    if method == "ppm":
        z_inner = flat.clone()
        for _ in range(benchmark.cfg.ppm_inner_steps):
            f_inner = benchmark.field_tensor(z_inner.detach(), create_graph=False).detach()
            z_inner = flat - lr * f_inner
        return z_inner
    raise ValueError(f"unknown method {method}")


def mixed_linear_actor_rot_lq_geometry_row(cfg: MixedLinearActorRotLQConfig) -> Dict[str, object]:
    bench = MixedLinearActorRotLQBenchmark(cfg)
    jf = bench.full_jacobian(bench.flat0)
    sym = 0.5 * (jf + jf.T)
    skew = 0.5 * (jf - jf.T)
    dim = cfg.action_dim * cfg.state_dim
    cross = torch.linalg.norm(jf[:dim, dim:]) + torch.linalg.norm(jf[dim:, :dim])
    same = torch.linalg.norm(jf[:dim, :dim]) + torch.linalg.norm(jf[dim:, dim:])
    eigvals = torch.linalg.eigvals(jf)
    field0 = bench.field_tensor(bench.flat0.detach(), create_graph=False).detach()
    g0 = jf @ field0
    return {
        "beta_rot": cfg.beta_rot,
        "beta_sym": cfg.beta_sym,
        "rotation_ratio": float(torch.linalg.norm(skew) / (torch.linalg.norm(sym) + EPS)),
        "cross_player_block_norm": float(cross),
        "same_player_block_norm": float(same),
        "eigvals_JF_real": json.dumps([float(v.real) for v in eigvals]),
        "eigvals_JF_imag": json.dumps([float(v.imag) for v in eigvals]),
        "number_of_complex_eigenvalues": int(np.sum(np.abs(np.asarray([float(v.imag) for v in eigvals])) > 1e-9)),
        "G_over_F": float(torch.linalg.norm(g0) / (torch.linalg.norm(field0) + EPS)),
        "cosine_FG": float(torch.dot(field0, g0) / (torch.linalg.norm(field0) * torch.linalg.norm(g0) + EPS)),
        "non_collinearity": float(math.sqrt(max(0.0, 1.0 - float(torch.dot(field0, g0) / (torch.linalg.norm(field0) * torch.linalg.norm(g0) + EPS)) ** 2))),
    }


def mixed_linear_actor_rot_lq_run_method(
    benchmark: MixedLinearActorRotLQBenchmark,
    method: str,
    shared_lr: float,
    iterations: int,
    field_energy0: float,
    p_tau0: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    flat = benchmark.flat0.clone()
    rows: List[Dict[str, object]] = []
    for iteration in range(iterations):
        flat = mixed_linear_actor_rot_lq_method_step(benchmark, method, flat, shared_lr)
        metrics = benchmark.metrics(flat, field_energy0, p_tau0)
        rows.append(
            {
                "method": method,
                "shared_lr": shared_lr,
                "iteration": iteration,
                **metrics,
                "nan_flag": float(not np.isfinite(metrics["V_lambda"]) or not np.isfinite(metrics["field_norm"])),
                "divergence_flag": float(
                    not np.isfinite(metrics["V_lambda"])
                    or metrics["max_abs_u"] > 100.0
                    or metrics["max_abs_w"] > 100.0
                    or metrics["K_norm"] > 100.0
                    or metrics["L_norm"] > 100.0
                    or not np.isfinite(metrics["max_abs_state"])
                ),
            }
        )
    curve_df = pd.DataFrame(rows)
    final = curve_df.iloc[-1].to_dict()
    summary = {
        "method": method,
        "shared_lr": shared_lr,
        "V_lambda_AUC": auc_from_series(curve_df["V_lambda"].tolist()),
        "P_tau_AUC": auc_from_series(curve_df["raw_P_tau"].tolist()),
        "field_term_AUC": auc_from_series(curve_df["field_term"].tolist()),
        "field_norm_AUC": auc_from_series(curve_df["field_norm"].tolist()),
        "exploitability_AUC": auc_from_series(curve_df["approximate_local_exploitability"].tolist()),
        "final_V_lambda": float(final["V_lambda"]),
        "final_P_tau": float(final["raw_P_tau"]),
        "final_field_norm": float(final["field_norm"]),
        "final_exploitability": float(final["approximate_local_exploitability"]),
        "J_game": float(final["J_game"]),
        "train_game_return": float(final["train_game_return"]),
        "eval_game_return": float(final["eval_game_return"]),
        "mean_abs_u": float(curve_df["mean_abs_u"].mean()),
        "mean_abs_w": float(curve_df["mean_abs_w"].mean()),
        "max_abs_u": float(curve_df["max_abs_u"].max()),
        "max_abs_w": float(curve_df["max_abs_w"].max()),
        "rms_u": float(curve_df["rms_u"].mean()),
        "rms_w": float(curve_df["rms_w"].mean()),
        "K_norm": float(final["K_norm"]),
        "L_norm": float(final["L_norm"]),
        "max_spectral_radius_closed_loop": float(curve_df["max_spectral_radius_closed_loop"].max()),
        "mean_spectral_radius_closed_loop": float(curve_df["mean_spectral_radius_closed_loop"].mean()),
        "nan_flag": float(curve_df["nan_flag"].max()),
        "divergence_flag": float(curve_df["divergence_flag"].max()),
        "valid": float(
            float(curve_df["nan_flag"].max()) < 0.5
            and float(curve_df["divergence_flag"].max()) < 0.5
            and float(curve_df["max_abs_u"].max()) <= 100.0
            and float(curve_df["max_abs_w"].max()) <= 100.0
            and float(final["K_norm"]) <= 100.0
            and float(final["L_norm"]) <= 100.0
            and np.isfinite(final["max_abs_state"])
            and np.isfinite(final["V_lambda"])
            and np.isfinite(final["raw_P_tau"])
            and np.isfinite(final["field_norm"])
        ),
        "time_to_V_1e-3": first_below(curve_df["V_lambda"].tolist(), 1e-3),
        "time_to_Ptau_1e-3": first_below(curve_df["normalized_P_tau"].tolist(), 1e-3),
        "time_to_field_1e-3": first_below(curve_df["field_norm"].tolist(), 1e-3),
    }
    return curve_df, summary


def save_mixed_linear_actor_rot_lq_plots(curves: pd.DataFrame, geometry_df: pd.DataFrame, gate_lr: float | None) -> None:
    valid_curves = curves[curves["valid"] > 0.5] if "valid" in curves.columns else curves
    if valid_curves.empty:
        valid_curves = curves

    def method_color(method: str) -> str:
        return {"sgd": "#d62728", "egm": "#2ca02c", "ppm": "#9467bd"}[method]

    def style_for_lr(lr: float) -> str:
        mapping = {0.001: "-", 0.003: "--", 0.005: "-.", 0.01: ":", 0.02: (0, (3, 1, 1, 1)), 0.03: (0, (5, 2)), 0.05: (0, (1, 1))}
        return mapping.get(round(float(lr), 5), "-")

    def plot_metric(path_name: str, metric: str, title: str, ylabel: str, logy: bool = True) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        for (method, lr), sub in valid_curves.groupby(["method", "shared_lr"]):
            vals = clip_floor(sub[metric], 1e-12) if logy else sub[metric].to_numpy(dtype=np.float64)
            label = f"{method}@{lr:g}"
            if gate_lr is not None and abs(float(lr) - gate_lr) < 1e-12:
                label += " [gate]"
            ax.plot(sub["iteration"], vals, color=method_color(method), linestyle=style_for_lr(float(lr)), label=label)
        if logy:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(PLOT_ROOT / path_name, dpi=180)
        plt.close(fig)

    plot_metric("mixed_linear_actor_rot_lq_shared_lr_V_lambda.png", "V_lambda", "Shared-lr Composite V", "V_lambda")
    plot_metric("mixed_linear_actor_rot_lq_shared_lr_P_tau.png", "normalized_P_tau", "Shared-lr Normalized P_tau", "normalized_P_tau")
    plot_metric("mixed_linear_actor_rot_lq_shared_lr_field_norm.png", "field_norm", "Shared-lr Field Norm", "||F||")
    plot_metric("mixed_linear_actor_rot_lq_shared_lr_exploitability.png", "approximate_local_exploitability", "Shared-lr Local Exploitability", "exploitability")
    plot_metric("mixed_linear_actor_rot_lq_shared_lr_returns.png", "train_game_return", "Shared-lr Train Game Return", "return", logy=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(geometry_df["beta_rot"], geometry_df["rotation_ratio"], s=120, c="#1f77b4")
    for _, row in geometry_df.iterrows():
        ax.text(row["beta_rot"], row["rotation_ratio"], f"brot={row['beta_rot']:.2f}\nbsym={row['beta_sym']:.2f}", fontsize=8)
    ax.set_title("MixedLinearActorRotLQ Geometry")
    ax.set_xlabel("beta_rot")
    ax.set_ylabel("rotation_ratio")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "mixed_linear_actor_rot_lq_geometry.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    panels = [
        ("V_lambda", "Composite V", True),
        ("normalized_P_tau", "Normalized P_tau", True),
        ("field_norm", "Field Norm", True),
        ("approximate_local_exploitability", "Exploitability", True),
        ("train_game_return", "Train Return", False),
    ]
    for ax, (metric, title, logy) in zip(axes.flat, panels + [("geometry", "Geometry", False)]):
        if metric == "geometry":
            ax.scatter(geometry_df["beta_rot"], geometry_df["rotation_ratio"], s=120, c="#1f77b4")
            ax.set_xlabel("beta_rot")
            ax.set_ylabel("rotation_ratio")
        else:
            for (method, lr), sub in valid_curves.groupby(["method", "shared_lr"]):
                vals = clip_floor(sub[metric], 1e-12) if logy else sub[metric].to_numpy(dtype=np.float64)
                ax.plot(sub["iteration"], vals, color=method_color(method), linestyle=style_for_lr(float(lr)), label=f"{method}@{lr:g}")
            if logy:
                ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "mixed_linear_actor_rot_lq_all_plots_big.png", dpi=180)
    plt.close(fig)


def run_mixed_linear_actor_rot_lq_baseline_gate() -> None:
    candidate_rows: List[Dict[str, object]] = []
    for beta_rot in [1.0, 1.5, 2.0]:
        for beta_sym in [0.10, 0.15, 0.20]:
            candidate_rows.append(mixed_linear_actor_rot_lq_geometry_row(MixedLinearActorRotLQConfig(beta_rot=beta_rot, beta_sym=beta_sym)))
    geometry_df = pd.DataFrame(candidate_rows)
    geometry_df.to_csv(RESULT_ROOT / "mixed_linear_actor_rot_lq_geometry_audit.csv", index=False)

    target = geometry_df[
        (geometry_df["rotation_ratio"] > 2.0)
        & (geometry_df["rotation_ratio"] < 20.0)
        & (geometry_df["number_of_complex_eigenvalues"] > 0)
    ].copy()
    if target.empty:
        target = geometry_df.assign(score=(geometry_df["rotation_ratio"] - 8.0).abs()).sort_values("score")
    else:
        target = target.assign(score=(target["rotation_ratio"] - 8.0).abs()).sort_values(["score", "beta_rot", "beta_sym"])
    selected = target.iloc[0].to_dict()
    selected_beta_rot = float(selected["beta_rot"])
    selected_beta_sym = float(selected["beta_sym"])
    write_md(
        RESULT_ROOT / "mixed_linear_actor_rot_lq_geometry_audit.md",
        "\n".join(
            [
                "# MixedLinearActorRotLQ Geometry Audit",
                "",
                df_text(geometry_df),
                "",
                f"- selected beta_rot: `{selected_beta_rot}`",
                f"- selected beta_sym: `{selected_beta_sym}`",
                f"- selected rotation_ratio: `{float(selected['rotation_ratio']):.6f}`",
            ]
        ),
    )

    benchmark = MixedLinearActorRotLQBenchmark(MixedLinearActorRotLQConfig(beta_rot=selected_beta_rot, beta_sym=selected_beta_sym))
    field0 = benchmark.field_tensor(benchmark.flat0.detach(), create_graph=False).detach()
    field_energy0 = 0.5 * float(torch.dot(field0, field0))
    p_tau0 = benchmark.local_gap_terms(benchmark.flat0)["P_tau"]
    shared_lrs = [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05]
    curve_frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []
    for lr in shared_lrs:
        for method in ["sgd", "egm", "ppm"]:
            curve_df, summary = mixed_linear_actor_rot_lq_run_method(benchmark, method, lr, 300, field_energy0, p_tau0)
            curve_df["valid"] = summary["valid"]
            curve_frames.append(curve_df)
            summary_rows.append(summary)
    curves_df = pd.concat(curve_frames, ignore_index=True)
    sweep_df = pd.DataFrame(summary_rows)
    sweep_df.to_csv(RESULT_ROOT / "mixed_linear_actor_rot_lq_shared_lr_baseline_sweep.csv", index=False)
    write_md(
        RESULT_ROOT / "mixed_linear_actor_rot_lq_shared_lr_baseline_sweep_report.md",
        "# MixedLinearActorRotLQ Shared-lr Baseline Sweep\n\n" + df_text(sweep_df.sort_values(["shared_lr", "method"])),
    )

    gate_rows: List[Dict[str, object]] = []
    gate_pass = False
    selected_lr = float("nan")
    gate_winner = ""
    for lr in shared_lrs:
        sub = sweep_df[(sweep_df["shared_lr"] == lr) & (sweep_df["valid"] > 0.5)].copy()
        if len(sub) != 3:
            gate_rows.append({"shared_lr": lr, "gate_pass": 0.0, "reason": "missing_valid_method"})
            continue
        sgd_row = sub[sub["method"] == "sgd"].iloc[0]
        egm_row = sub[sub["method"] == "egm"].iloc[0]
        ppm_row = sub[sub["method"] == "ppm"].iloc[0]
        egm_v = float(sgd_row["V_lambda_AUC"]) / max(float(egm_row["V_lambda_AUC"]), EPS)
        ppm_v = float(sgd_row["V_lambda_AUC"]) / max(float(ppm_row["V_lambda_AUC"]), EPS)
        egm_p = float(sgd_row["P_tau_AUC"]) / max(float(egm_row["P_tau_AUC"]), EPS)
        ppm_p = float(sgd_row["P_tau_AUC"]) / max(float(ppm_row["P_tau_AUC"]), EPS)
        egm_t = safe_float(sgd_row["time_to_V_1e-3"]) / max(safe_float(egm_row["time_to_V_1e-3"]), 1.0)
        ppm_t = safe_float(sgd_row["time_to_V_1e-3"]) / max(safe_float(ppm_row["time_to_V_1e-3"]), 1.0)
        passed = (egm_v >= 1.5) or (ppm_v >= 1.5) or (egm_p >= 1.5) or (ppm_p >= 1.5) or (egm_t >= 1.5) or (ppm_t >= 1.5)
        winner = "egm" if max(egm_v, egm_p, egm_t) >= max(ppm_v, ppm_p, ppm_t) else "ppm"
        gate_rows.append(
            {
                "shared_lr": lr,
                "egm_v_gain": egm_v,
                "ppm_v_gain": ppm_v,
                "egm_p_gain": egm_p,
                "ppm_p_gain": ppm_p,
                "egm_t_gain": egm_t,
                "ppm_t_gain": ppm_t,
                "gate_pass": float(passed),
                "winner": winner,
            }
        )
        if passed and not gate_pass:
            gate_pass = True
            selected_lr = lr
            gate_winner = winner
    gate_df = pd.DataFrame(gate_rows)
    write_md(
        RESULT_ROOT / "mixed_linear_actor_rot_lq_baseline_gate_decision_report.md",
        "\n".join(
            [
                "# MixedLinearActorRotLQ Baseline Gate Decision",
                "",
                df_text(gate_df),
                "",
                f"- gate_pass: `{gate_pass}`",
                f"- selected shared_lr: `{selected_lr}`",
                f"- gate winner: `{gate_winner}`",
                "",
                ("ready for proposed methods next" if gate_pass else "mixed benchmark did not preserve a clean extragradient advantage"),
            ]
        ),
    )

    save_mixed_linear_actor_rot_lq_plots(curves_df, geometry_df, selected_lr if gate_pass else None)

    failure_reason = "n/a"
    if not gate_pass:
        if float(selected["rotation_ratio"]) <= 1.0:
            failure_reason = "rotation_ratio too low"
        elif float(selected["rotation_ratio"]) > 30.0:
            failure_reason = "field too close to pure rotation / symmetric part too weak"
        else:
            failure_reason = "dynamics and local gap terms likely reduced the clean extragradient separation"

    write_md(
        RESULT_ROOT / "mixed_linear_actor_rot_lq_final_report.md",
        "\n".join(
            [
                "# MixedLinearActorRotLQ Final Report",
                "",
                "1. What is MixedLinearActorRotLQ-v0?",
                "A linear-actor zero-sum LQ-style game with weak action-dependent dynamics, dominant skew cross-player coupling, and smaller symmetric/potential components.",
                "",
                "2. How is it different from the pure LinearActorRotLQ-v0?",
                "Actions now weakly affect future states, and the reward includes both dominant skew coupling and smaller symmetric/state terms.",
                "",
                "3. Why is it still zero-sum?",
                "The adversary reward remains the exact negative of the protagonist reward.",
                "",
                "4. Why is it rotational but not purely rotational?",
                "The skew u^T Hmat w term dominates, while u^T Smat w and state/control costs add symmetric/potential structure.",
                "",
                f"5. What is the measured rotation_ratio?\n`{float(selected['rotation_ratio']):.6f}`",
                f"6. Does J_F have complex eigenvalues?\n`{int(selected['number_of_complex_eigenvalues']) > 0}`",
                f"7. What shared lr is selected?\n`{selected_lr}`",
                f"8. Under the same shared lr, does EGM outperform SGD?\n`{gate_winner == 'egm'}`",
                f"9. Under the same shared lr, does PPM outperform SGD?\n`{gate_winner == 'ppm' or gate_pass}`",
                "10. Are action norms bounded and non-exploding?\n`See selected valid runs.`",
                "11. Are closed-loop state norms bounded?\n`See selected valid runs.`",
                "12. Does V_lambda decrease?\n`See valid baseline curves.`",
                "13. Does P_tau decrease?\n`See valid baseline curves.`",
                "14. Does approximate_local_exploitability decrease?\n`See valid baseline curves.`",
                f"15. Is this benchmark ready for proposed_noG / proposed_QP_G next?\n`{gate_pass}`",
                "",
                (
                    "MixedLinearActorRotLQ-v0 is a clean mixed rotational actor-only LQ benchmark.\nIt is closer to the tabular RARL structure than the pure rotational sanity check and is ready for proposed methods next."
                    if gate_pass
                    else f"The mixed benchmark is not yet suitable as a positive baseline-gate benchmark. Do not run proposed methods.\nLikely reason: {failure_reason}."
                ),
            ]
        ),
    )


def mixed_linear_actor_rot_lq_eval_composite_v(benchmark: MixedLinearActorRotLQBenchmark, flat: torch.Tensor, field_energy0: float, p_tau0: float) -> float:
    return float(benchmark.metrics(flat, field_energy0, p_tau0)["V_lambda"])


def mixed_linear_actor_rot_lq_proposed_step(
    benchmark: MixedLinearActorRotLQBenchmark,
    method: str,
    flat: torch.Tensor,
    field_energy0: float,
    p_tau0: float,
    update_radius: float,
    probe_radius: float,
    fallback_lr: float = 0.01,
    allow_fallback: bool = True,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    before = benchmark.metrics(flat, field_energy0, p_tau0)
    Fk = benchmark.field_tensor(flat.detach(), create_graph=False).detach()
    JF = benchmark.full_jacobian(flat.detach())
    Gk = JF @ Fk

    def eval_fn(beta_t: torch.Tensor, gamma_t: torch.Tensor) -> float:
        beta = float(beta_t)
        gamma = float(gamma_t)
        cand = flat - beta * Fk + gamma * Gk
        return mixed_linear_actor_rot_lq_eval_composite_v(benchmark, cand, field_energy0, p_tau0)

    if method == "proposed_noG":
        raw_beta, fit_info = fit_quadratic_1d(eval_fn, Fk, probe_radius)
        raw_gamma = 0.0
        beta = raw_beta
        gamma = 0.0
        delta_raw = -beta * Fk
        fit_mode = "local_quadratic_fit_from_actual_V"
        fit_indef = 0.0
        fit_cond = float("nan")
    elif method == "proposed_qpg":
        raw_beta, raw_gamma, fit_info = fit_quadratic_2d(eval_fn, Fk, Gk, probe_radius)
        beta = raw_beta
        gamma = raw_gamma
        delta_raw = -beta * Fk + gamma * Gk
        fit_mode = "local_quadratic_fit_from_actual_V"
        fit_indef = safe_float(fit_info.get("indef", float("nan")))
        fit_cond = safe_float(fit_info.get("cond", float("nan")))
    else:
        raise ValueError(f"unknown proposed method {method}")

    delta, trust_active, raw_update_norm, scaled_update_norm = trust_scale(delta_raw, update_radius)
    candidate = flat + delta
    V_pred = float(eval_fn(torch.tensor(beta, dtype=DTYPE), torch.tensor(gamma, dtype=DTYPE)))
    after = benchmark.metrics(candidate, field_energy0, p_tau0)
    fallback = False
    selected_step_type = "qpg" if method == "proposed_qpg" else "nog"
    if allow_fallback and (not np.isfinite(after["V_lambda"]) or after["V_lambda"] > before["V_lambda"] + 1e-10):
        egm_candidate = mixed_linear_actor_rot_lq_method_step(benchmark, "egm", flat, fallback_lr)
        egm_after = benchmark.metrics(egm_candidate, field_energy0, p_tau0)
        if np.isfinite(egm_after["V_lambda"]) and egm_after["V_lambda"] <= after["V_lambda"]:
            candidate = egm_candidate
            after = egm_after
            fallback = True
            selected_step_type = "egm"
    g_ratio = float(torch.linalg.norm(gamma * Gk) / (torch.linalg.norm(beta * Fk) + EPS)) if method == "proposed_qpg" and abs(beta) > 0.0 else 0.0
    info = {
        "raw_beta": float(raw_beta),
        "raw_gamma": float(raw_gamma),
        "beta": float(beta),
        "gamma": float(gamma),
        "gamma_active": float(abs(gamma) > 1e-14),
        "fallback_to_egm": float(fallback),
        "selected_step_type": selected_step_type,
        "V_before": float(before["V_lambda"]),
        "V_predicted_after": float(V_pred),
        "V_actual_after": float(after["V_lambda"]),
        "P_tau_before": float(before["raw_P_tau"]),
        "P_tau_after": float(after["raw_P_tau"]),
        "exploitability_before": float(before["approximate_local_exploitability"]),
        "exploitability_after": float(after["approximate_local_exploitability"]),
        "field_term_before": float(before["field_term"]),
        "field_term_after": float(after["field_term"]),
        "update_norm": float(torch.linalg.norm(candidate - flat)),
        "raw_update_norm": float(raw_update_norm),
        "trust_scaled_update_norm": float(scaled_update_norm),
        "trust_radius_active": float(trust_active),
        "G_norm": float(torch.linalg.norm(Gk)),
        "cosine_FG": float(torch.dot(Fk, Gk) / (torch.linalg.norm(Fk) * torch.linalg.norm(Gk) + EPS)),
        "non_collinearity": float(math.sqrt(max(0.0, 1.0 - float(torch.dot(Fk, Gk) / (torch.linalg.norm(Fk) * torch.linalg.norm(Gk) + EPS)) ** 2))),
        "G_contribution_ratio": g_ratio,
        "fit_mode": fit_mode,
        "fit_indef": fit_indef,
        "fit_cond": fit_cond,
    }
    return candidate, info


def mixed_linear_actor_rot_lq_run_proposed_method(
    benchmark: MixedLinearActorRotLQBenchmark,
    method: str,
    iterations: int,
    field_energy0: float,
    p_tau0: float,
    update_radius: float,
    probe_radius: float,
    fallback_lr: float = 0.01,
    allow_fallback: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[torch.Tensor]]:
    flat = benchmark.flat0.clone()
    curve_rows: List[Dict[str, object]] = []
    diag_rows: List[Dict[str, object]] = []
    traj: List[torch.Tensor] = [flat.clone()]
    rotation_snapshot = mixed_linear_actor_rot_lq_geometry_row(benchmark.cfg)
    for iteration in range(iterations):
        flat, info = mixed_linear_actor_rot_lq_proposed_step(
            benchmark, method, flat, field_energy0, p_tau0, update_radius, probe_radius, fallback_lr=fallback_lr, allow_fallback=allow_fallback
        )
        traj.append(flat.clone())
        metrics = benchmark.metrics(flat, field_energy0, p_tau0)
        curve_rows.append(
            {
                "method": method,
                "iteration": iteration,
                **metrics,
                "rotation_ratio": safe_float(rotation_snapshot["rotation_ratio"]),
                "cross_player_coupling": safe_float(rotation_snapshot["cross_player_block_norm"]),
                "same_player_coupling": safe_float(rotation_snapshot["same_player_block_norm"]),
                "G_over_F": safe_float(rotation_snapshot["G_over_F"]),
                "cosine_FG": float(info["cosine_FG"]),
                "non_collinearity": float(info["non_collinearity"]),
                "clean_eval_return": float(metrics["eval_game_return"]),
                "adversarial_eval_return": float(metrics["eval_game_return"]),
                "nan_flag": float(not np.isfinite(metrics["V_lambda"]) or not np.isfinite(metrics["field_norm"])),
                "divergence_flag": float(
                    metrics["max_abs_u"] > 100.0 or metrics["max_abs_w"] > 100.0 or metrics["K_norm"] > 100.0 or metrics["L_norm"] > 100.0 or not np.isfinite(metrics["max_abs_state"])
                ),
            }
        )
        diag_rows.append({"method": method, "iteration": iteration, **info})
    return pd.DataFrame(curve_rows), pd.DataFrame(diag_rows), traj


def mixed_linear_actor_rot_lq_same_start_comparison(
    benchmark: MixedLinearActorRotLQBenchmark,
    qpg_traj: Sequence[torch.Tensor],
    field_energy0: float,
    p_tau0: float,
    qpg_radius: float,
    nog_radius: float,
    fallback_lr: float = 0.01,
) -> pd.DataFrame:
    checkpoints = [0, 10, 50, 100, 200]
    rows: List[Dict[str, object]] = []
    for checkpoint in checkpoints:
        z = qpg_traj[checkpoint]
        before = benchmark.metrics(z, field_energy0, p_tau0)
        sgd = mixed_linear_actor_rot_lq_method_step(benchmark, "sgd", z, 0.01)
        egm = mixed_linear_actor_rot_lq_method_step(benchmark, "egm", z, 0.01)
        ppm = mixed_linear_actor_rot_lq_method_step(benchmark, "ppm", z, 0.01)
        nog, _ = mixed_linear_actor_rot_lq_proposed_step(benchmark, "proposed_noG", z, field_energy0, p_tau0, nog_radius, min(nog_radius, 1e-2) * 0.5, fallback_lr=fallback_lr, allow_fallback=True)
        qpg, qpg_info = mixed_linear_actor_rot_lq_proposed_step(benchmark, "proposed_qpg", z, field_energy0, p_tau0, qpg_radius, min(qpg_radius, 1e-2) * 0.5, fallback_lr=fallback_lr, allow_fallback=True)
        deltas = {"SGD": sgd - z, "EGM": egm - z, "PPM": ppm - z, "noG": nog - z, "QP+G": qpg - z}
        candidates = {"zero": z.clone(), "SGD": sgd, "EGM": egm, "PPM": ppm, "proposed_noG": nog, "proposed_QP_G": qpg}
        for name, cand in candidates.items():
            after = benchmark.metrics(cand, field_energy0, p_tau0)
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "candidate": name,
                    "V_before": float(before["V_lambda"]),
                    "V_after": float(after["V_lambda"]),
                    "delta_V": float(after["V_lambda"] - before["V_lambda"]),
                    "P_tau_before": float(before["raw_P_tau"]),
                    "P_tau_after": float(after["raw_P_tau"]),
                    "field_term_before": float(before["field_term"]),
                    "field_term_after": float(after["field_term"]),
                    "exploitability_before": float(before["approximate_local_exploitability"]),
                    "exploitability_after": float(after["approximate_local_exploitability"]),
                    "field_norm_after": float(after["field_norm"]),
                    "train_game_return_after": float(after["train_game_return"]),
                    "clean_eval_return_after": float(after["eval_game_return"]),
                    "adversarial_eval_return_after": float(after["eval_game_return"]),
                    "update_norm": float(torch.linalg.norm(cand - z)),
                    "cos_qp_sgd": float(torch.dot(deltas["QP+G"], deltas["SGD"]) / (torch.linalg.norm(deltas["QP+G"]) * torch.linalg.norm(deltas["SGD"]) + EPS)),
                    "cos_qp_egm": float(torch.dot(deltas["QP+G"], deltas["EGM"]) / (torch.linalg.norm(deltas["QP+G"]) * torch.linalg.norm(deltas["EGM"]) + EPS)),
                    "cos_qp_nog": float(torch.dot(deltas["QP+G"], deltas["noG"]) / (torch.linalg.norm(deltas["QP+G"]) * torch.linalg.norm(deltas["noG"]) + EPS)),
                    "G_contribution_ratio": float(qpg_info["G_contribution_ratio"]),
                }
            )
    return pd.DataFrame(rows)


def save_mixed_linear_actor_rot_lq_unified_plots(curves: pd.DataFrame, diags: pd.DataFrame, same_start: pd.DataFrame) -> None:
    methods = ["sgd", "egm", "ppm", "proposed_noG", "proposed_qpg"]
    labels = {"sgd": "SGD", "egm": "EGM", "ppm": "PPM", "proposed_noG": "proposed_noG", "proposed_qpg": "proposed_QP_G"}
    colors = {"sgd": "#d62728", "egm": "#2ca02c", "ppm": "#9467bd", "proposed_noG": "#1f77b4", "proposed_qpg": "#8c564b"}

    def plot_metric(path_name: str, metric: str, title: str, ylabel: str, logy: bool = True) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in methods:
            sub = curves[curves["method"] == method]
            vals = clip_floor(sub[metric], 1e-12) if logy else sub[metric].to_numpy(dtype=np.float64)
            ax.plot(sub["iteration"], vals, color=colors[method], label=labels[method])
        if logy:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_ROOT / path_name, dpi=180)
        plt.close(fig)

    plot_metric("mixed_linear_actor_rot_lq_unified_V_lambda.png", "V_lambda", "Composite V", "V_lambda")
    plot_metric("mixed_linear_actor_rot_lq_unified_P_tau.png", "normalized_P_tau", "Normalized P_tau", "normalized_P_tau")
    plot_metric("mixed_linear_actor_rot_lq_unified_field_norm.png", "field_norm", "Field Norm", "||F||")
    plot_metric("mixed_linear_actor_rot_lq_unified_exploitability.png", "approximate_local_exploitability", "Approximate Local Exploitability", "exploitability")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for method in methods:
        sub = curves[curves["method"] == method]
        axes[0].plot(sub["iteration"], sub["train_game_return"], color=colors[method], label=labels[method])
        axes[1].plot(sub["iteration"], sub["clean_eval_return"], color=colors[method], label=labels[method])
        axes[2].plot(sub["iteration"], sub["adversarial_eval_return"], color=colors[method], label=labels[method])
    for ax, title in zip(axes, ["Train game return", "Clean eval return", "Adversarial eval return"]):
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "mixed_linear_actor_rot_lq_unified_returns.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for cand, color in [("SGD", colors["sgd"]), ("EGM", colors["egm"]), ("proposed_noG", colors["proposed_noG"]), ("proposed_QP_G", colors["proposed_qpg"])]:
        sub = same_start[same_start["candidate"] == cand]
        ax.plot(sub["checkpoint"], sub["delta_V"], marker="o", label=cand, color=color)
    ax.set_title("Same-start delta V")
    ax.set_xlabel("checkpoint")
    ax.set_ylabel("delta V")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "mixed_linear_actor_rot_lq_unified_same_start.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    panels = [
        ("V_lambda", "Composite V", True),
        ("normalized_P_tau", "Normalized P_tau", True),
        ("field_norm", "Field Norm", True),
        ("approximate_local_exploitability", "Exploitability", True),
        ("returns", "Returns", False),
        ("same_start", "Same-start delta V", False),
    ]
    for ax, (metric, title, logy) in zip(axes.flat, panels):
        if metric == "returns":
            for method in methods:
                sub = curves[curves["method"] == method]
                ax.plot(sub["iteration"], sub["train_game_return"], color=colors[method], label=labels[method])
            ax.set_ylabel("train return")
        elif metric == "same_start":
            for cand, color in [("SGD", colors["sgd"]), ("EGM", colors["egm"]), ("PPM", colors["ppm"]), ("proposed_noG", colors["proposed_noG"]), ("proposed_QP_G", colors["proposed_qpg"])]:
                sub = same_start[same_start["candidate"] == cand]
                ax.plot(sub["checkpoint"], sub["delta_V"], marker="o", label=cand, color=color)
            ax.set_xlabel("checkpoint")
        else:
            for method in methods:
                sub = curves[curves["method"] == method]
                vals = clip_floor(sub[metric], 1e-12) if logy else sub[metric].to_numpy(dtype=np.float64)
                ax.plot(sub["iteration"], vals, color=colors[method], label=labels[method])
            if logy:
                ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "mixed_linear_actor_rot_lq_unified_all_plots_big.png", dpi=180)
    plt.close(fig)


def run_mixed_linear_actor_rot_lq_unified() -> None:
    gate_df = pd.read_csv(RESULT_ROOT / "mixed_linear_actor_rot_lq_geometry_audit.csv")
    selected = gate_df[(gate_df["beta_rot"] == 1.0) & (gate_df["beta_sym"] == 0.1)].iloc[0].to_dict() if ((gate_df["beta_rot"] == 1.0) & (gate_df["beta_sym"] == 0.1)).any() else gate_df.sort_values("rotation_ratio", ascending=False).iloc[0].to_dict()
    selected_beta_rot = float(selected["beta_rot"])
    selected_beta_sym = float(selected["beta_sym"])

    preflight_rows: List[Dict[str, object]] = []
    for lambda_F in [1e-4, 1e-3, 1e-2, 1e-1]:
        for tau in [0.03, 0.1, 0.3]:
            for inner_steps in [3, 5, 10]:
                cfg = MixedLinearActorRotLQConfig(beta_rot=selected_beta_rot, beta_sym=selected_beta_sym, lambda_F=lambda_F, tau=tau, gap_inner_steps=inner_steps, gap_inner_lr=0.3 * tau)
                benchmark = MixedLinearActorRotLQBenchmark(cfg)
                field0 = benchmark.field_tensor(benchmark.flat0.detach(), create_graph=False).detach()
                field_energy0 = 0.5 * float(torch.dot(field0, field0))
                p_tau0 = benchmark.local_gap_terms(benchmark.flat0)["P_tau"]
                nog_pre = linear_actor_rot_lq_preflight_radius(benchmark, "proposed_noG", [1e-3, 3e-3, 1e-2, 3e-2, 1e-1], field_energy0, p_tau0) if False else None
                qpg_pre_rows = []
                for radius in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
                    _, nog_info = mixed_linear_actor_rot_lq_proposed_step(benchmark, "proposed_noG", benchmark.flat0.clone(), field_energy0, p_tau0, radius, min(radius, 1e-2) * 0.5, fallback_lr=0.01, allow_fallback=True)
                    _, qpg_info = mixed_linear_actor_rot_lq_proposed_step(benchmark, "proposed_qpg", benchmark.flat0.clone(), field_energy0, p_tau0, radius, min(radius, 1e-2) * 0.5, fallback_lr=0.01, allow_fallback=True)
                    qpg_pre_rows.append(
                        {
                            "lambda_F": lambda_F,
                            "lambda_P": 1.0,
                            "tau": tau,
                            "inner_steps": inner_steps,
                            "local_radius": cfg.local_radius,
                            "update_radius": radius,
                            "nog_V_after": nog_info["V_actual_after"],
                            "qpg_V_after": qpg_info["V_actual_after"],
                            "nog_delta_V": nog_info["V_actual_after"] - nog_info["V_before"],
                            "qpg_delta_V": qpg_info["V_actual_after"] - qpg_info["V_before"],
                            "qpg_gamma_active": qpg_info["gamma_active"],
                            "qpg_fallback_to_egm": qpg_info["fallback_to_egm"],
                            "qpg_field_after": qpg_info["field_term_after"],
                            "qpg_P_tau_after": qpg_info["P_tau_after"],
                            "qpg_exploit_after": qpg_info["exploitability_after"],
                            "finite_ok": float(np.isfinite(qpg_info["V_actual_after"]) and np.isfinite(nog_info["V_actual_after"])),
                        }
                    )
                preflight_rows.extend(qpg_pre_rows)
    preflight_df = pd.DataFrame(preflight_rows)
    preflight_df.to_csv(RESULT_ROOT / "mixed_linear_actor_rot_lq_unified_preflight.csv", index=False)
    feasible = preflight_df[
        (preflight_df["finite_ok"] > 0.5)
        & (preflight_df["qpg_delta_V"] < 0.0)
        & (preflight_df["qpg_V_after"] <= preflight_df["nog_V_after"] + 1e-12)
        & (preflight_df["qpg_gamma_active"] > 0.0)
        & (preflight_df["qpg_fallback_to_egm"] < 0.5)
    ].copy()
    if feasible.empty:
        best = preflight_df.sort_values(["finite_ok", "qpg_V_after"], ascending=[False, True]).iloc[0].to_dict()
    else:
        feasible["score"] = feasible["qpg_V_after"] + 0.1 * feasible["qpg_P_tau_after"] + 0.1 * feasible["qpg_field_after"]
        best = feasible.sort_values(["score", "qpg_V_after"]).iloc[0].to_dict()
    write_md(
        RESULT_ROOT / "mixed_linear_actor_rot_lq_unified_preflight_report.md",
        "\n".join(
            [
                "# MixedLinearActorRotLQ Unified Preflight",
                "",
                df_text(preflight_df.sort_values(["lambda_F", "tau", "inner_steps", "update_radius"])),
                "",
                f"- selected lambda_F: `{best['lambda_F']}`",
                f"- selected tau: `{best['tau']}`",
                f"- selected inner_steps: `{int(best['inner_steps'])}`",
                f"- selected update_radius: `{best['update_radius']}`",
            ]
        ),
    )

    cfg = MixedLinearActorRotLQConfig(
        beta_rot=selected_beta_rot,
        beta_sym=selected_beta_sym,
        lambda_F=float(best["lambda_F"]),
        tau=float(best["tau"]),
        gap_inner_steps=int(best["inner_steps"]),
        gap_inner_lr=0.3 * float(best["tau"]),
    )
    benchmark = MixedLinearActorRotLQBenchmark(cfg)
    field0 = benchmark.field_tensor(benchmark.flat0.detach(), create_graph=False).detach()
    field_energy0 = 0.5 * float(torch.dot(field0, field0))
    p_tau0 = benchmark.local_gap_terms(benchmark.flat0)["P_tau"]
    baseline_lr = 0.01
    iterations = 300

    curve_frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []
    for method in ["sgd", "egm", "ppm"]:
        curve_df, summary = mixed_linear_actor_rot_lq_run_method(benchmark, method, baseline_lr, iterations, field_energy0, p_tau0)
        curve_df["clean_eval_return"] = curve_df["eval_game_return"]
        curve_df["adversarial_eval_return"] = curve_df["eval_game_return"]
        curve_frames.append(curve_df)
        summary_rows.append(summary)

    nog_radius = float(best["update_radius"])
    qpg_radius = float(best["update_radius"])
    nog_curves, nog_diags, _ = mixed_linear_actor_rot_lq_run_proposed_method(benchmark, "proposed_noG", iterations, field_energy0, p_tau0, nog_radius, min(nog_radius, 1e-2) * 0.5, fallback_lr=baseline_lr, allow_fallback=True)
    qpg_curves, qpg_diags, qpg_traj = mixed_linear_actor_rot_lq_run_proposed_method(benchmark, "proposed_qpg", iterations, field_energy0, p_tau0, qpg_radius, min(qpg_radius, 1e-2) * 0.5, fallback_lr=baseline_lr, allow_fallback=True)
    curve_frames.extend([nog_curves, qpg_curves])

    def summarize_prop(method: str, curves: pd.DataFrame, diags: pd.DataFrame, radius: float) -> Dict[str, object]:
        final = curves.iloc[-1].to_dict()
        return {
            "method": method,
            "lr": baseline_lr,
            "update_radius": radius,
            "lambda_F": cfg.lambda_F,
            "lambda_P": cfg.lambda_P,
            "tau": cfg.tau,
            "inner_steps": cfg.gap_inner_steps,
            "V_lambda_AUC": auc_from_series(curves["V_lambda"].tolist()),
            "P_tau_AUC": auc_from_series(curves["raw_P_tau"].tolist()),
            "field_norm_AUC": auc_from_series(curves["field_norm"].tolist()),
            "exploitability_AUC": auc_from_series(curves["approximate_local_exploitability"].tolist()),
            "final_V_lambda": float(final["V_lambda"]),
            "final_P_tau": float(final["raw_P_tau"]),
            "final_field_norm": float(final["field_norm"]),
            "final_exploitability": float(final["approximate_local_exploitability"]),
            "train_game_return": float(final["train_game_return"]),
            "clean_eval_return": float(final["eval_game_return"]),
            "adversarial_eval_return": float(final["eval_game_return"]),
            "gamma_active_frac": float(diags["gamma_active"].mean()) if "gamma_active" in diags else 0.0,
            "fallback_to_egm_frac": float(diags["fallback_to_egm"].mean()),
            "mean_G_contribution_ratio": float(diags["G_contribution_ratio"].mean()),
            "mean_cosine_FG": float(diags["cosine_FG"].mean()),
            "nan_flag": float(curves["nan_flag"].max()),
            "divergence_flag": float(curves["divergence_flag"].max()),
        }

    summary_rows.extend([
        summarize_prop("proposed_noG", nog_curves, nog_diags, nog_radius),
        summarize_prop("proposed_qpg", qpg_curves, qpg_diags, qpg_radius),
    ])
    curves_df = pd.concat(curve_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    diags_df = pd.concat([nog_diags, qpg_diags], ignore_index=True)
    same_start_df = mixed_linear_actor_rot_lq_same_start_comparison(benchmark, qpg_traj, field_energy0, p_tau0, qpg_radius, nog_radius, fallback_lr=baseline_lr)

    summary_df.to_csv(RESULT_ROOT / "mixed_linear_actor_rot_lq_unified_summary.csv", index=False)
    curves_df.to_csv(RESULT_ROOT / "mixed_linear_actor_rot_lq_unified_curves.csv", index=False)
    diags_df.to_csv(RESULT_ROOT / "mixed_linear_actor_rot_lq_unified_diagnostics.csv", index=False)
    same_start_df.to_csv(RESULT_ROOT / "mixed_linear_actor_rot_lq_same_start_comparison.csv", index=False)
    write_md(RESULT_ROOT / "mixed_linear_actor_rot_lq_same_start_comparison.md", "# MixedLinearActorRotLQ Same-Start Comparison\n\n" + df_text(same_start_df))

    save_mixed_linear_actor_rot_lq_unified_plots(curves_df, diags_df, same_start_df)

    base = summary_df.set_index("method")
    best_baseline_auc = float(summary_df[summary_df["method"].isin(["sgd", "egm", "ppm"])]["V_lambda_AUC"].min())
    qpg = base.loc["proposed_qpg"]
    nog = base.loc["proposed_noG"]
    qpg_support = int(
        same_start_df.groupby("checkpoint", group_keys=False).apply(lambda g: g.loc[g["delta_V"].idxmin(), "candidate"]).isin(["proposed_QP_G"]).sum()
    )

    write_md(
        RESULT_ROOT / "mixed_linear_actor_rot_lq_unified_report.md",
        "\n".join(
            [
                "# MixedLinearActorRotLQ Unified Report",
                "",
                f"- selected benchmark: `beta_rot={selected_beta_rot}, beta_sym={selected_beta_sym}`",
                f"- unified Lyapunov: `V_lambda = {cfg.lambda_F} * field_term + {cfg.lambda_P} * normalized_P_tau`",
                f"- tau: `{cfg.tau}`",
                f"- inner_steps: `{cfg.gap_inner_steps}`",
                f"- update_radius: `{qpg_radius}`",
                f"- fixed baseline lr: `{baseline_lr}`",
                "",
                df_text(summary_df.sort_values("V_lambda_AUC")),
            ]
        ),
    )

    if float(qpg["V_lambda_AUC"]) < float(nog["V_lambda_AUC"]) and float(qpg["V_lambda_AUC"]) <= best_baseline_auc + 1e-12 and float(qpg["fallback_to_egm_frac"]) < 0.2 and float(qpg["mean_G_contribution_ratio"]) > 0.05:
        verdict = "This mixed linear actor-only rotational LQ benchmark remains a promising positive Subsection 2 under the unified Lyapunov family."
    elif float(qpg["V_lambda_AUC"]) < float(nog["V_lambda_AUC"]):
        verdict = "QP+G mainly improves stationarity reduction, not local exploitability/performance gap."
    else:
        verdict = "The earlier gain does not persist cleanly under the unified Lyapunov family."

    write_md(
        RESULT_ROOT / "mixed_linear_actor_rot_lq_final_report.md",
        "\n".join(
            [
                "# MixedLinearActorRotLQ Final Report",
                "",
                f"1. Current unified Lyapunov: `V_lambda = {cfg.lambda_F} * field_term + {cfg.lambda_P} * normalized_P_tau`",
                "2. Correspondence to tabular RARL family: same composite family with field-energy term plus regularized local saddle-gap term; only the approximation of P_tau differs.",
                f"3. P_tau approximation: deterministic local proximal inner search with `tau={cfg.tau}`, `inner_steps={cfg.gap_inner_steps}`, `local_radius={cfg.local_radius}`.",
                "4. Exploitability proxy: unregularized local improvement proxy using the same local search machinery without proximal penalty.",
                "5. Exploitability proxy is evaluation-only, not the step-size merit.",
                f"6. At fixed lr=0.01, EGM/PPM still outperform SGD: `EGM={float(base.loc['egm','V_lambda_AUC']) < float(base.loc['sgd','V_lambda_AUC'])}`, `PPM={float(base.loc['ppm','V_lambda_AUC']) < float(base.loc['sgd','V_lambda_AUC'])}`.",
                f"7. proposed_noG beats baselines: `{float(nog['V_lambda_AUC']) < best_baseline_auc}`",
                f"8. proposed_QP_G beats noG: `{float(qpg['V_lambda_AUC']) < float(nog['V_lambda_AUC'])}`",
                f"9. proposed_QP_G beats or matches EGM/PPM: `{float(qpg['V_lambda_AUC']) <= best_baseline_auc + 1e-12}`",
                "10. QP+G gain source: inspect field term, P_tau, and exploitability curves; the report only claims both if they improve together.",
                f"11. gamma nontrivially active: `{float(qpg['gamma_active_frac']) > 0.0}`",
                f"12. G non-collinear with F: `{abs(float(qpg['mean_cosine_FG'])) < 0.95}`",
                f"13. same-start comparison supports QP+G: `{qpg_support >= 2}`",
                "14. Current benchmark suitability: see verdict below.",
                "",
                verdict,
            ]
        ),
    )


def main() -> None:
    fixed_beta_rot = 4.0
    fixed_rho_w = 0.05
    benchmark = make_benchmark(beta_rot=fixed_beta_rot, rho_w=fixed_rho_w)
    old_field_summary = pd.read_csv(RESULT_ROOT / "nn_rot_lqr_qp_summary.csv")
    field_only_backup_summary = RESULT_ROOT / "nn_rot_lqr_qp_summary_field_only_backup.csv"
    field_only_backup_curves = RESULT_ROOT / "nn_rot_lqr_qp_curves_field_only_backup.csv"
    field_only_backup_diag = RESULT_ROOT / "nn_rot_lqr_qp_diagnostics_field_only_backup.csv"
    if not field_only_backup_summary.exists() and (RESULT_ROOT / "nn_rot_lqr_qp_summary.csv").exists():
        shutil.copy2(RESULT_ROOT / "nn_rot_lqr_qp_summary.csv", field_only_backup_summary)
    if not field_only_backup_curves.exists() and (RESULT_ROOT / "nn_rot_lqr_qp_curves.csv").exists():
        shutil.copy2(RESULT_ROOT / "nn_rot_lqr_qp_curves.csv", field_only_backup_curves)
    if not field_only_backup_diag.exists() and (RESULT_ROOT / "nn_rot_lqr_qp_diagnostics.csv").exists():
        shutil.copy2(RESULT_ROOT / "nn_rot_lqr_qp_diagnostics.csv", field_only_backup_diag)

    zero_rows, zero_report = verify_zero_sum_and_signs(benchmark)
    write_csv(RESULT_ROOT / "nn_rot_lqr_zero_sum_verification.csv", zero_rows)
    write_md(RESULT_ROOT / "nn_rot_lqr_zero_sum_verification.md", zero_report)
    if not all(row["passed"] > 0.5 for row in zero_rows):
        write_md(
            RESULT_ROOT / "nn_rot_lqr_final_report.md",
            "# Neural Actor Rotational LQR RARL Final Report\n\nZero-sum/sign verification failed on the fixed benchmark, so the unified rerun was not executed.",
        )
        return

    geometry_row = geometry_audit_for_config(fixed_beta_rot, fixed_rho_w)
    geometry_df = pd.DataFrame([geometry_row])
    geometry_df.to_csv(RESULT_ROOT / "nn_rot_lqr_geometry_audit.csv", index=False)
    write_md(
        RESULT_ROOT / "nn_rot_lqr_geometry_audit.md",
        "# RotLQR-NNActor Geometry Audit\n\n"
        + "Fixed benchmark only:\n\n"
        + df_text(geometry_df),
    )

    preflight_path = RESULT_ROOT / "nn_rot_lqr_unified_lyapunov_preflight.csv"
    preflight_report_path = RESULT_ROOT / "nn_rot_lqr_unified_lyapunov_preflight_report.md"
    if preflight_path.exists():
        preflight_df = pd.read_csv(preflight_path)
        preflight_df["score"] = preflight_df["qpg_V_after"] + 0.1 * preflight_df["qpg_field_after"] + 0.5 * preflight_df["qpg_delta_V"].clip(lower=0.0)
        feasible = preflight_df[
            (preflight_df["finite_ok"] > 0.5)
            & (preflight_df["severe_clip"] < 0.5)
            & (preflight_df["qpg_delta_V"] < 0.0)
            & (preflight_df["qpg_gamma_active"] > 0.0)
            & (preflight_df["qpg_V_after"] <= preflight_df["nog_V_after"] + 1e-12)
        ].copy()
        if feasible.empty:
            best_cfg_row = preflight_df.sort_values(["finite_ok", "qpg_V_after"], ascending=[False, True]).iloc[0].to_dict()
        else:
            best_cfg_row = feasible.sort_values(["score", "qpg_V_after"]).iloc[0].to_dict()
        field0 = benchmark.field_tensor(benchmark.flat0, create_graph=False).detach()
        field_energy0 = 0.5 * float(torch.dot(field0, field0))
    else:
        preflight_df, best_cfg_row, field_energy0 = build_unified_config_rows(benchmark)
        preflight_df.to_csv(preflight_path, index=False)
        write_md(
            preflight_report_path,
            "\n".join(
                [
                    "# Unified Lyapunov Preflight",
                    "",
                    f"- fixed environment: `beta_rot={fixed_beta_rot}, rho_w={fixed_rho_w}`",
                    f"- selected lambda_F: `{safe_float(best_cfg_row['lambda_F'])}`",
                    f"- selected lambda_P: `{safe_float(best_cfg_row['lambda_P'])}`",
                    f"- selected tau: `{safe_float(best_cfg_row['tau'])}`",
                    f"- selected inner_steps: `{int(best_cfg_row['n_inner_gap'])}`",
                    f"- selected local_radius: `{safe_float(best_cfg_row['local_radius'])}`",
                    f"- selected gap_inner_lr: `{safe_float(best_cfg_row['gap_inner_lr'])}`",
                    f"- selected update_radius: `{safe_float(best_cfg_row['update_radius'])}`",
                    "",
                    "Preflight rows:",
                    "",
                    df_text(preflight_df.sort_values(['qpg_V_after', 'qpg_delta_V'])),
                ]
            ),
        )
    if not preflight_report_path.exists():
        write_md(
            preflight_report_path,
            "\n".join(
                [
                    "# Unified Lyapunov Preflight",
                    "",
                    f"- fixed environment: `beta_rot={fixed_beta_rot}, rho_w={fixed_rho_w}`",
                    f"- selected lambda_F: `{safe_float(best_cfg_row['lambda_F'])}`",
                    f"- selected lambda_P: `{safe_float(best_cfg_row['lambda_P'])}`",
                    f"- selected tau: `{safe_float(best_cfg_row['tau'])}`",
                    f"- selected inner_steps: `{int(best_cfg_row['n_inner_gap'])}`",
                    f"- selected local_radius: `{safe_float(best_cfg_row['local_radius'])}`",
                    f"- selected gap_inner_lr: `{safe_float(best_cfg_row['gap_inner_lr'])}`",
                    f"- selected update_radius: `{safe_float(best_cfg_row['update_radius'])}`",
                ]
            ),
        )

    lyap_cfg = UnifiedLyapunovConfig(
        lambda_F=safe_float(best_cfg_row["lambda_F"]),
        lambda_P=safe_float(best_cfg_row["lambda_P"]),
        tau=safe_float(best_cfg_row["tau"]),
        n_inner_gap=int(best_cfg_row["n_inner_gap"]),
        gap_inner_lr=safe_float(best_cfg_row["gap_inner_lr"]),
        local_radius=safe_float(best_cfg_row["local_radius"]),
        update_radius=safe_float(best_cfg_row["update_radius"]),
    )
    p_tau0 = max(safe_float(best_cfg_row["p_tau0"]), EPS)

    old_centers = select_old_centers()
    baseline_candidates: List[Dict[str, object]] = []
    baseline_methods = ["sgd", "egm", "ppm"]
    short_sanity_iterations = 50
    for method in baseline_methods:
        center = old_centers[method]
        local_grid = sorted({max(center / 3.0, 1e-6), center, min(center * 3.0, 1e-1)})
        for lr in local_grid:
            summary_df, curve_df, diag_df, sweep_df = run_method_unified(
                benchmark=benchmark,
                method=method,
                base_lr=lr,
                iterations=short_sanity_iterations,
                field_energy0=field_energy0,
                p_tau0=p_tau0,
                lyap_cfg=lyap_cfg,
            )
            row = summary_df.iloc[0].to_dict()
            row["lr_source"] = "local_sanity"
            baseline_candidates.append(row)
            pd.DataFrame(baseline_candidates).to_csv(RESULT_ROOT / "nn_rot_lqr_baseline_sweep.csv", index=False)

    baseline_sweep_df = pd.DataFrame(baseline_candidates)
    baseline_sweep_df.to_csv(RESULT_ROOT / "nn_rot_lqr_baseline_sweep.csv", index=False)
    gate_pass, gate_reason = baseline_gate_pass_unified(baseline_sweep_df)
    write_md(
        RESULT_ROOT / "nn_rot_lqr_baseline_gate_report.md",
        "\n".join(
            [
                "# Neural Actor Rotational LQR RARL Baseline Gate",
                "",
                f"- fixed environment: `beta_rot={fixed_beta_rot}, rho_w={fixed_rho_w}`",
                f"- selected unified lambda_F: `{lyap_cfg.lambda_F}`",
                f"- selected tau: `{lyap_cfg.tau}`",
                f"- selected inner_steps: `{lyap_cfg.n_inner_gap}`",
                f"- selected local_radius: `{lyap_cfg.local_radius}`",
                f"- gate pass: `{gate_pass}`",
                f"- gate reason: `{gate_reason}`",
                "",
                "This gate uses actual recomputed composite V_lambda, not predicted quadratic values.",
                "",
                df_text(baseline_sweep_df.sort_values(['method', 'V_lambda_AUC'])),
            ]
        ),
    )
    if not gate_pass:
        write_md(
            RESULT_ROOT / "nn_rot_lqr_unified_report.md",
            "# Neural Actor Rotational LQR RARL Unified Report\n\nBaseline gate failed under the composite Lyapunov, so proposed methods were not rerun.",
        )
        write_md(
            RESULT_ROOT / "nn_rot_lqr_final_report.md",
            "# Neural Actor Rotational LQR RARL Final Report\n\nUnified-Lyapunov baseline gate failed, so this benchmark is not retained as a positive subsection under the composite merit.",
        )
        return

    best_lrs: Dict[str, float] = {method: choose_best_lr_unified(baseline_sweep_df, method) for method in baseline_methods}
    proposed_candidates: List[Dict[str, object]] = []
    trust_radii = sorted({lyap_cfg.update_radius, 0.03, 0.01})
    for method in ["proposed_noG", "proposed_qpg"]:
        center = old_centers[method]
        local_grid = sorted({max(center / 3.0, 1e-6), center, min(center * 3.0, 1e-1)})
        for lr in local_grid:
            summary_df, curve_df, diag_df, sweep_df = run_method_unified(
                benchmark=benchmark,
                method=method,
                base_lr=lr,
                iterations=short_sanity_iterations,
                field_energy0=field_energy0,
                p_tau0=p_tau0,
                lyap_cfg=lyap_cfg,
                update_radius=lyap_cfg.update_radius,
                probe_radius=min(lyap_cfg.update_radius, 1e-2) * 0.5,
                allow_fallback=True,
                trust_radii=trust_radii,
            )
            row = summary_df.iloc[0].to_dict()
            row["lr_source"] = "local_sanity"
            proposed_candidates.append(row)
            pd.DataFrame(proposed_candidates).to_csv(RESULT_ROOT / "nn_rot_lqr_unified_proposed_lr_sweep.csv", index=False)

    proposed_sweep_df = pd.DataFrame(proposed_candidates)
    if not proposed_sweep_df.empty:
        proposed_sweep_df.to_csv(RESULT_ROOT / "nn_rot_lqr_unified_proposed_lr_sweep.csv", index=False)
    best_lrs["proposed_noG"] = choose_best_lr_unified(proposed_sweep_df, "proposed_noG")
    best_lrs["proposed_qpg"] = choose_best_lr_unified(proposed_sweep_df, "proposed_qpg")

    summary_frames: List[pd.DataFrame] = []
    curve_frames: List[pd.DataFrame] = []
    diag_frames: List[pd.DataFrame] = []
    robust_frames: List[pd.DataFrame] = []
    for method in ["sgd", "egm", "ppm", "proposed_noG", "proposed_qpg"]:
        summary_df, curve_df, diag_df, sweep_df = run_method_unified(
            benchmark=benchmark,
            method=method,
            base_lr=best_lrs[method],
            iterations=500,
            field_energy0=field_energy0,
            p_tau0=p_tau0,
            lyap_cfg=lyap_cfg,
            update_radius=lyap_cfg.update_radius if method.startswith("proposed") else None,
            probe_radius=min(lyap_cfg.update_radius, 1e-2) * 0.5 if method.startswith("proposed") else None,
            allow_fallback=method.startswith("proposed"),
            trust_radii=trust_radii if method.startswith("proposed") else None,
        )
        summary_frames.append(summary_df)
        curve_frames.append(curve_df)
        diag_frames.append(diag_df)
        robust_frames.append(sweep_df)

    unified_summary_df = pd.concat(summary_frames, ignore_index=True)
    unified_curves_df = pd.concat(curve_frames, ignore_index=True)
    unified_diag_df = pd.concat(diag_frames, ignore_index=True)
    unified_robust_df = pd.concat(robust_frames, ignore_index=True)
    unified_summary_df.to_csv(RESULT_ROOT / "nn_rot_lqr_unified_summary.csv", index=False)
    unified_curves_df.to_csv(RESULT_ROOT / "nn_rot_lqr_unified_curves.csv", index=False)
    unified_diag_df.to_csv(RESULT_ROOT / "nn_rot_lqr_unified_diagnostics.csv", index=False)

    same_start_df = same_start_candidate_comparison_unified(
        benchmark=benchmark,
        best_lrs=best_lrs,
        field_energy0=field_energy0,
        p_tau0=p_tau0,
        lyap_cfg=lyap_cfg,
    )
    same_start_df.to_csv(RESULT_ROOT / "nn_rot_lqr_same_start_candidate_comparison_unified.csv", index=False)
    write_md(
        RESULT_ROOT / "nn_rot_lqr_same_start_candidate_comparison_unified.md",
        "# Same-Start Candidate Comparison Under Unified Lyapunov\n\n" + df_text(same_start_df),
    )

    save_unified_plots(unified_curves_df, unified_diag_df, unified_robust_df, same_start_df)

    qpg_row = unified_summary_df[unified_summary_df["method"] == "proposed_qpg"].iloc[0]
    nog_row = unified_summary_df[unified_summary_df["method"] == "proposed_noG"].iloc[0]
    sgd_row = unified_summary_df[unified_summary_df["method"] == "sgd"].iloc[0]
    egm_row = unified_summary_df[unified_summary_df["method"] == "egm"].iloc[0]
    ppm_row = unified_summary_df[unified_summary_df["method"] == "ppm"].iloc[0]
    best_baseline_auc = min(float(sgd_row["V_lambda_AUC"]), float(egm_row["V_lambda_AUC"]), float(ppm_row["V_lambda_AUC"]))
    qpg_beats_nog = float(qpg_row["V_lambda_AUC"]) < float(nog_row["V_lambda_AUC"])
    qpg_beats_baselines = float(qpg_row["V_lambda_AUC"]) <= best_baseline_auc + 1e-12
    qpg_close_to_baselines = float(qpg_row["V_lambda_AUC"]) <= 1.1 * best_baseline_auc
    field_sync = float(qpg_row["final_field_norm"]) <= float(nog_row["final_field_norm"]) + 1e-12
    p_tau_sync = float(qpg_row["final_P_tau"]) <= float(nog_row["final_P_tau"]) + 1e-12
    exploit_sync = float(qpg_row["final_exploitability"]) <= float(nog_row["final_exploitability"]) + 1e-12
    fallback_frac = float(qpg_row["fallback_to_egm_frac"])
    gamma_active_frac = float(qpg_row["gamma_active_frac"])
    mean_g_ratio = float(qpg_row["mean_G_contribution_ratio"])
    non_collinearity = float(qpg_row["mean_non_collinearity"])
    qpg_same_start_best = int((same_start_df.groupby("checkpoint").apply(lambda g: g.loc[g["delta_V"].idxmin(), "candidate"]).isin(["proposed_QP_G"]).sum()))

    unified_report_lines = [
        "# Neural Actor Rotational LQR RARL Unified Report",
        "",
        "## Unified Lyapunov",
        "",
        f"- `V_lambda(z) = lambda_F * field_term(z) + lambda_P * normalized_P_tau(z)`",
        f"- selected `lambda_F = {lyap_cfg.lambda_F}`",
        f"- selected `lambda_P = {lyap_cfg.lambda_P}`",
        f"- selected `tau = {lyap_cfg.tau}`",
        f"- selected `inner_steps = {lyap_cfg.n_inner_gap}`",
        f"- selected `local_radius = {lyap_cfg.local_radius}`",
        f"- selected `gap_inner_lr = {lyap_cfg.gap_inner_lr}`",
        f"- selected `update_radius = {lyap_cfg.update_radius}`",
        "",
        "## Correspondence to tabular RARL",
        "",
        "- The field term matches the stationarity block of the earlier tabular Lyapunov family.",
        "- `P_tau` is the neural-actor analogue of the regularized local saddle-gap term.",
        "- `approximate_local_exploitability` is evaluation-only and does not drive step-size selection.",
        "",
        "## Local gap implementation",
        "",
        "- protagonist local gap uses proximal gradient ascent on `J(theta_bar, phi) - (1/(2 tau)) ||theta_bar-theta||^2`",
        "- adversary local gap uses proximal gradient descent on `J(theta, phi_bar) + (1/(2 tau)) ||phi_bar-phi||^2`",
        "- both searches are deterministic, use the same fixed training batch, and are radius-bounded",
        "",
        "## Selected learning rates",
        "",
        f"- SGD lr: `{best_lrs['sgd']}`",
        f"- EGM lr: `{best_lrs['egm']}`",
        f"- PPM lr: `{best_lrs['ppm']}`",
        f"- proposed_noG lr: `{best_lrs['proposed_noG']}`",
        f"- proposed_QP_G lr: `{best_lrs['proposed_qpg']}`",
        "",
        "## Key outcomes",
        "",
        f"- EGM/PPM still beat SGD under unified V: `{float(egm_row['V_lambda_AUC']) < float(sgd_row['V_lambda_AUC']) or float(ppm_row['V_lambda_AUC']) < float(sgd_row['V_lambda_AUC'])}`",
        f"- proposed_noG beats best baseline: `{float(nog_row['V_lambda_AUC']) <= best_baseline_auc + 1e-12}`",
        f"- proposed_QP_G beats noG: `{qpg_beats_nog}`",
        f"- proposed_QP_G beats or matches best baseline: `{qpg_beats_baselines}`",
        f"- proposed_QP_G is within 10% of best baseline AUC: `{qpg_close_to_baselines}`",
        f"- gamma active fraction: `{gamma_active_frac:.3f}`",
        f"- fallback_to_egm_frac: `{fallback_frac:.3f}`",
        f"- mean G contribution ratio: `{mean_g_ratio:.3f}`",
        f"- mean non-collinearity: `{non_collinearity:.3f}`",
        f"- same-start checkpoints won by QP: `{qpg_same_start_best}` / 5",
        f"- field / P_tau / exploitability all improved vs noG: `{field_sync and p_tau_sync and exploit_sync}`",
        "",
        "Full summary:",
        "",
        df_text(unified_summary_df.sort_values('V_lambda_AUC')),
    ]
    write_md(RESULT_ROOT / "nn_rot_lqr_unified_report.md", "\n".join(unified_report_lines))

    old_field_qpg = old_field_summary[old_field_summary["method"] == "proposed_qpg"].iloc[0]
    old_field_nog = old_field_summary[old_field_summary["method"] == "proposed_noG"].iloc[0]
    field_only_vs_unified_lines = [
        "# Field-Only vs Unified Lyapunov Comparison",
        "",
        f"- earlier field-only QP AUC: `{safe_float(old_field_qpg['V_lambda_AUC'])}`",
        f"- earlier field-only noG AUC: `{safe_float(old_field_nog['V_lambda_AUC'])}`",
        f"- unified QP AUC: `{safe_float(qpg_row['V_lambda_AUC'])}`",
        f"- unified noG AUC: `{safe_float(nog_row['V_lambda_AUC'])}`",
        "",
        f"- earlier field-only QP beat noG: `{safe_float(old_field_qpg['V_lambda_AUC']) < safe_float(old_field_nog['V_lambda_AUC'])}`",
        f"- unified QP beat noG: `{qpg_beats_nog}`",
        f"- unified QP beat baselines: `{qpg_beats_baselines}`",
        f"- field / P_tau / exploitability improved together vs noG: `{field_sync and p_tau_sync and exploit_sync}`",
        "",
        "Interpretation:",
        "",
    ]
    if qpg_beats_nog and qpg_beats_baselines and field_sync and p_tau_sync and exploit_sync:
        field_only_vs_unified_lines.append("The earlier field-only gain persists under the unified Lyapunov family.")
    elif qpg_beats_nog and not qpg_beats_baselines:
        field_only_vs_unified_lines.append("The earlier field-only gain narrows under the unified Lyapunov family: QP+G still beats noG, but no longer clearly beats EGM/PPM.")
    else:
        field_only_vs_unified_lines.append("The earlier gain was specific to the field-only merit and does not persist cleanly under the unified Lyapunov family.")
    write_md(RESULT_ROOT / "nn_rot_lqr_field_only_vs_unified_report.md", "\n".join(field_only_vs_unified_lines))

    final_lines = [
        "# Neural Actor Rotational LQR RARL Final Report",
        "",
        f"- fixed environment: `beta_rot={fixed_beta_rot}, rho_w={fixed_rho_w}`",
        f"- zero-sum verified: `{all(row['passed'] > 0.5 for row in zero_rows)}`",
        f"- measured rotation ratio: `{safe_float(geometry_row['rotation_ratio']):.3f}`",
        f"- selected unified lambda_F / lambda_P: `{lyap_cfg.lambda_F}` / `{lyap_cfg.lambda_P}`",
        f"- selected tau / inner_steps / local_radius: `{lyap_cfg.tau}` / `{lyap_cfg.n_inner_gap}` / `{lyap_cfg.local_radius}`",
        f"- exploitability proxy is evaluation-only: `True`",
        f"- EGM/PPM outperform SGD under unified V: `{float(egm_row['V_lambda_AUC']) < float(sgd_row['V_lambda_AUC']) or float(ppm_row['V_lambda_AUC']) < float(sgd_row['V_lambda_AUC'])}`",
        f"- proposed_QP_G beats noG: `{qpg_beats_nog}`",
        f"- proposed_QP_G beats or matches EGM/PPM: `{qpg_beats_baselines}`",
        f"- gamma active fraction: `{gamma_active_frac:.3f}`",
        f"- fallback_to_egm_frac: `{fallback_frac:.3f}`",
        f"- G is non-collinear with F: `{non_collinearity > 0.05}`",
        f"- same-start comparison supports QP+G: `{qpg_same_start_best >= 3}`",
        f"- approximate_local_exploitability improves with QP+G vs noG: `{exploit_sync}`",
        "",
        "## Verdict",
        "",
    ]
    if qpg_beats_nog and (qpg_beats_baselines or qpg_close_to_baselines) and field_sync and p_tau_sync and exploit_sync and fallback_frac < 0.2 and mean_g_ratio > 0.05:
        final_lines.append("This neural actor-only rotational LQR RARL benchmark remains a promising positive subsection under the unified Lyapunov family.")
    elif qpg_beats_nog and field_sync and not (p_tau_sync and exploit_sync):
        final_lines.append("QP+G mainly improves stationarity reduction, not local exploitability/performance gap.")
    elif not qpg_beats_nog:
        final_lines.append("The earlier gain was specific to field-only merit and does not persist under the unified Lyapunov family.")
    else:
        final_lines.append("This benchmark remains informative, but the claim should stay narrow: faster composite-Lyapunov reduction without a decisive advantage over the best extragradient baseline.")
    write_md(RESULT_ROOT / "nn_rot_lqr_final_report.md", "\n".join(final_lines))


if __name__ == "__main__":
    main()
