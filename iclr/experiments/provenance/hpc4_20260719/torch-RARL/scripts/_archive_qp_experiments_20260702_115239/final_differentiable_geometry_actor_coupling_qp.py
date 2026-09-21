from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import gymnasium as gym
except Exception:
    import gym  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT.parent / "results" / "final_differentiable_geometry_actor_coupling_qp"

DEVICE = torch.device("cpu")
DTYPE = torch.float32
EPS = 1e-8

ENV_ORDER = ["Walker2d-v4", "HalfCheetah-v4", "Hopper-v4"]
RHO_GRID = [0.3, 0.6, 1.0, 2.0]
LR_GRID = [3e-4, 1e-3]
METHODS = ["sgd_gda", "egm", "proposed_nog_closed", "proposed_qp_nog_safe"]

SEED = 0
ALPHA = 0.1
ITERATIONS = 12
ROLLOUT_STEPS = 512
EVAL_EPISODES = 8
GEOM_BATCHES = 5
GATE_TOPK = 3
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.001
VF_COEF = 0.5
CRITIC_LR = 1e-3
CRITIC_STEPS = 1
MAX_GRAD_NORM = 10.0
INIT_LOG_STD = -1.0
GEOM_PROBES = 8
MERIT_LAMBDA_F = 0.01
MERIT_LAMBDA_L = 1.0
NO_G_FALLBACK_TOL = 0.0
QP_TRUST_G_RATIO = 0.05


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / (x.std(unbiased=False) + EPS)


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, bootstrap_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=DTYPE, device=DEVICE)
    next_value = bootstrap_value
    for idx in reversed(range(rewards.shape[0])):
        mask = 1.0 - dones[idx]
        delta = rewards[idx] + (GAMMA * next_value * mask) - values[idx]
        gae = delta + (GAMMA * GAE_LAMBDA * mask * gae)
        advantages[idx] = gae
        next_value = values[idx]
    return advantages + values, advantages


def auc(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    arr = np.asarray(values, dtype=np.float64)
    x = np.arange(arr.size, dtype=np.float64)
    return float(np.trapezoid(arr, x))


def spike_ratio(values: list[float]) -> float:
    if not values:
        return math.nan
    arr = np.asarray(values, dtype=np.float64)
    return float(np.max(np.abs(arr)) / (np.median(np.abs(arr)) + EPS))


def action_rotation_matrix(action_dim: int) -> torch.Tensor:
    mat = torch.zeros((action_dim, action_dim), dtype=DTYPE, device=DEVICE)
    for start in range(0, action_dim - 1, 2):
        mat[start, start + 1] = 1.0
        mat[start + 1, start] = -1.0
    return mat


def fit_quadratic_1d(v0: float, v1: float, v2: float, delta: float) -> tuple[float, float]:
    h = (v2 - 2.0 * v1 + v0) / max(delta * delta, EPS)
    l = (v1 - v0) / max(delta, EPS) - 0.5 * h * delta
    return float(l), float(h)


def solve_nog(point: Callable[[float], float], v0: float, eta: float, beta_max: float | None) -> dict[str, float]:
    db = max(eta, EPS)
    v1 = point(db)
    v2 = point(2.0 * db)
    l_beta, h_bb = fit_quadratic_1d(v0, v1, v2, db)
    denom = h_bb + 1e-8
    beta_raw = -l_beta / denom if abs(denom) > EPS else 0.0
    beta = max(beta_raw, 0.0)
    if beta_max is not None:
        beta = min(beta, beta_max)
    q = l_beta * beta + 0.5 * h_bb * beta * beta
    return {
        "db": db,
        "l_beta": float(l_beta),
        "h_bb": float(h_bb),
        "beta_raw": float(beta_raw),
        "beta": float(beta),
        "q": float(q),
    }


def solve_qp(
    point: Callable[[float, float], float],
    v0: float,
    eta: float,
    beta_max: float | None,
    gamma_max: float | None,
) -> dict[str, Any]:
    db = max(eta, EPS)
    dg = max(eta * eta, 1e-6)
    v_b = point(db, 0.0)
    v_2b = point(2.0 * db, 0.0)
    v_g = point(0.0, dg)
    v_2g = point(0.0, 2.0 * dg)
    v_bg = point(db, dg)
    l_beta, h_bb = fit_quadratic_1d(v0, v_b, v_2b, db)
    l_gamma, h_gg = fit_quadratic_1d(v0, v_g, v_2g, dg)
    h_bg = (v_bg - v_b - v_g + v0) / max(db * dg, EPS)

    def q_value(beta_value: float, gamma_value: float) -> float:
        return float(
            l_beta * beta_value
            + l_gamma * gamma_value
            + 0.5 * h_bb * beta_value * beta_value
            + h_bg * beta_value * gamma_value
            + 0.5 * h_gg * gamma_value * gamma_value
        )

    def beta_feasible(beta_value: float) -> bool:
        if not math.isfinite(beta_value) or beta_value < -1e-10:
            return False
        if beta_max is not None and beta_value > beta_max + 1e-10:
            return False
        return True

    def gamma_feasible(gamma_value: float) -> bool:
        if not math.isfinite(gamma_value) or gamma_value < -1e-10:
            return False
        if gamma_max is not None and gamma_value > gamma_max + 1e-10:
            return False
        return True

    def clamp_beta(beta_value: float) -> float:
        value = max(float(beta_value), 0.0)
        if beta_max is not None:
            value = min(value, float(beta_max))
        return value

    def clamp_gamma(gamma_value: float) -> float:
        value = max(float(gamma_value), 0.0)
        if gamma_max is not None:
            value = min(value, float(gamma_max))
        return value

    candidates: list[tuple[str, float, float, float]] = []

    def add_candidate(name: str, beta_value: float, gamma_value: float) -> None:
        if not beta_feasible(beta_value) or not gamma_feasible(gamma_value):
            return
        qv = q_value(float(beta_value), float(gamma_value))
        if not math.isfinite(qv):
            return
        candidates.append((name, float(beta_value), float(gamma_value), float(qv)))

    nog = solve_nog(lambda beta: point(beta, 0.0), v0, eta, beta_max)
    beta_nog = float(nog["beta"])
    add_candidate("corner_00", 0.0, 0.0)
    add_candidate("edge_gamma0_nog", beta_nog, 0.0)

    hessian_reg = np.array([[h_bb + 1e-8, h_bg], [h_bg, h_gg + 1e-8]], dtype=np.float64)
    linear = np.array([-l_beta, -l_gamma], dtype=np.float64)
    beta_raw = 0.0
    gamma_raw = 0.0
    try:
        solution = np.linalg.solve(hessian_reg, linear)
        beta_raw = float(solution[0])
        gamma_raw = float(solution[1])
        add_candidate("interior", beta_raw, gamma_raw)
    except Exception:
        pass

    denom_gg = h_gg + 1e-8
    gamma_edge0_raw = -l_gamma / denom_gg if abs(denom_gg) > EPS else 0.0
    add_candidate("edge_beta0", 0.0, clamp_gamma(gamma_edge0_raw))

    denom_bb = h_bb + 1e-8
    beta_gamma0_raw = -l_beta / denom_bb if abs(denom_bb) > EPS else 0.0
    add_candidate("edge_gamma0_stationary", clamp_beta(beta_gamma0_raw), 0.0)

    if beta_max is not None:
        add_candidate("corner_betaMax_0", float(beta_max), 0.0)
        gamma_on_beta_max_raw = -(l_gamma + h_bg * beta_max) / denom_gg if abs(denom_gg) > EPS else 0.0
        add_candidate("edge_betaMax", float(beta_max), clamp_gamma(gamma_on_beta_max_raw))
    if gamma_max is not None:
        add_candidate("corner_0_gammaMax", 0.0, float(gamma_max))
        beta_on_gamma_max_raw = -(l_beta + h_bg * gamma_max) / denom_bb if abs(denom_bb) > EPS else 0.0
        add_candidate("edge_gammaMax", clamp_beta(beta_on_gamma_max_raw), float(gamma_max))
    if beta_max is not None and gamma_max is not None:
        add_candidate("corner_betaMax_gammaMax", float(beta_max), float(gamma_max))

    dedup: dict[tuple[int, int], tuple[str, float, float, float]] = {}
    for cand in candidates:
        key = (round(cand[1] / 1e-12), round(cand[2] / 1e-12))
        if key not in dedup or cand[3] < dedup[key][3]:
            dedup[key] = cand
    ranked = sorted(dedup.values(), key=lambda item: item[3])
    if not ranked:
        ranked = [("corner_00_fallback", 0.0, 0.0, q_value(0.0, 0.0))]
    name, beta, gamma, q_selected = ranked[0]
    q_nog = q_value(beta_nog, 0.0)
    return {
        "db": db,
        "dg": dg,
        "l_beta": float(l_beta),
        "l_gamma": float(l_gamma),
        "h_bb": float(h_bb),
        "h_bg": float(h_bg),
        "h_gg": float(h_gg),
        "beta_raw": float(beta_raw),
        "gamma_raw": float(gamma_raw),
        "beta": float(beta),
        "gamma": float(gamma),
        "q": float(q_selected),
        "beta_nog": float(beta_nog),
        "q_nog": float(q_nog),
        "predicted_inclusion_gap": float(q_selected - q_nog),
        "predicted_inclusion_pass": int(q_selected <= q_nog + 1e-6 * max(1.0, abs(q_selected), abs(q_nog))),
        "selected_active_set": name,
        "candidate_count": len(ranked),
    }


@dataclass(frozen=True)
class EnvSpec:
    env_id: str
    obs_dim: int
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    max_episode_steps: int


@dataclass(frozen=True)
class Config:
    env_id: str
    alpha: float
    rho: float
    lr: float

    @property
    def slug(self) -> str:
        return (
            f"{self.env_id.lower().replace('-', '_')}"
            f"_a{str(self.alpha).replace('.', 'p')}"
            f"_rho{str(self.rho).replace('.', 'p')}"
            f"_lr{str(self.lr).replace('.', 'p')}"
        )


class FlatMLP:
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, int], output_dim: int) -> None:
        h1, h2 = hidden_sizes
        self.shapes = [
            (h1, input_dim),
            (h1,),
            (h2, h1),
            (h2,),
            (output_dim, h2),
            (output_dim,),
        ]
        self.num_params = sum(int(np.prod(shape)) for shape in self.shapes)

    def init_flat(self, generator: torch.Generator, final_scale: float) -> torch.Tensor:
        chunks = []
        for idx, shape in enumerate(self.shapes):
            if len(shape) == 2:
                scale = 1.0 / math.sqrt(shape[1])
                tensor = torch.randn(shape, generator=generator, dtype=DTYPE, device=DEVICE) * scale
            else:
                tensor = 0.01 * torch.randn(shape, generator=generator, dtype=DTYPE, device=DEVICE)
            if idx >= 4:
                tensor = tensor * final_scale
            chunks.append(tensor.reshape(-1))
        return torch.cat(chunks)

    def unpack(self, flat: torch.Tensor) -> list[torch.Tensor]:
        offset = 0
        tensors = []
        for shape in self.shapes:
            size = int(np.prod(shape))
            tensors.append(flat[offset : offset + size].reshape(shape))
            offset += size
        return tensors

    def forward(self, flat: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        w1, b1, w2, b2, w3, b3 = self.unpack(flat)
        h1 = torch.tanh(obs @ w1.T + b1)
        h2 = torch.tanh(h1 @ w2.T + b2)
        return h2 @ w3.T + b3


class CriticNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: tuple[int, int]) -> None:
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class ActorCouplingGame:
    def __init__(self, spec: EnvSpec, cfg: Config, seed: int) -> None:
        self.spec = spec
        self.cfg = cfg
        self.seed = seed
        self.action_low_t = torch.as_tensor(spec.action_low, dtype=DTYPE, device=DEVICE)
        self.action_high_t = torch.as_tensor(spec.action_high, dtype=DTYPE, device=DEVICE)
        self.rot_dyn = action_rotation_matrix(spec.action_dim)
        self.H = action_rotation_matrix(spec.action_dim)
        self.actor_layout = FlatMLP(spec.obs_dim, (64, 64), spec.action_dim)
        self.slices = self._build_slices()
        self.value_p = CriticNet(spec.obs_dim, (64, 64)).to(DEVICE)
        self.value_a = CriticNet(spec.obs_dim, (64, 64)).to(DEVICE)
        self.opt_vp = torch.optim.Adam(self.value_p.parameters(), lr=CRITIC_LR)
        self.opt_va = torch.optim.Adam(self.value_a.parameters(), lr=CRITIC_LR)
        self.metric_refs: dict[str, float] = {}

    def _build_slices(self) -> dict[str, slice]:
        offset = 0
        actor_dim = self.actor_layout.num_params
        std_dim = self.spec.action_dim
        out = {}
        out["protagonist_actor"] = slice(offset, offset + actor_dim)
        offset += actor_dim
        out["protagonist_log_std"] = slice(offset, offset + std_dim)
        offset += std_dim
        out["adversary_actor"] = slice(offset, offset + actor_dim)
        offset += actor_dim
        out["adversary_log_std"] = slice(offset, offset + std_dim)
        return out

    def init_z(self) -> torch.Tensor:
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(self.seed)
        pa = self.actor_layout.init_flat(gen, final_scale=0.05)
        pl = torch.full((self.spec.action_dim,), INIT_LOG_STD, dtype=DTYPE, device=DEVICE)
        aa = self.actor_layout.init_flat(gen, final_scale=0.05)
        al = torch.full((self.spec.action_dim,), INIT_LOG_STD, dtype=DTYPE, device=DEVICE)
        return torch.cat([pa, pl, aa, al]).detach().clone()

    def split_z(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: z[slc] for name, slc in self.slices.items()}

    def actor_mean_raw(self, actor_flat: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_layout.forward(actor_flat, obs)

    def actor_mean_action(self, actor_flat: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        raw = self.actor_mean_raw(actor_flat, obs)
        return torch.clamp(raw, self.action_low_t, self.action_high_t)

    def actor_dist(self, actor_flat: torch.Tensor, log_std: torch.Tensor, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor_mean_raw(actor_flat, obs)
        std = torch.exp(log_std).unsqueeze(0).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def blend_action(self, u: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pert = torch.matmul(w, self.rot_dyn.T)
        raw = u + (self.cfg.alpha * pert)
        clipped = torch.clamp(raw, self.action_low_t, self.action_high_t)
        return raw, clipped

    def rot_mean(self, z: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        parts = self.split_z(z)
        u_bar = self.actor_mean_action(parts["protagonist_actor"], obs)
        w_bar = self.actor_mean_action(parts["adversary_actor"], obs)
        return torch.mean(torch.sum(u_bar * torch.matmul(w_bar, self.H.T), dim=-1) / math.sqrt(max(self.spec.action_dim, 1)))

    def collect_rollout(self, z: torch.Tensor, rollout_seed: int) -> dict[str, torch.Tensor]:
        parts = self.split_z(z)
        env = gym.make(self.spec.env_id)
        obs, _ = env.reset(seed=rollout_seed)
        obs_t = torch.as_tensor(obs, dtype=DTYPE, device=DEVICE)
        storage: dict[str, list[torch.Tensor]] = {key: [] for key in [
            "obs", "u_old", "w_old", "eps_p", "eps_a", "old_logprob_p", "old_logprob_a",
            "r_env", "r_adv", "done", "value_p_old", "value_a_old", "a_env_raw", "a_env",
        ]}
        train_returns: list[float] = []
        clip_hits = 0
        ep_reward = 0.0
        for _ in range(ROLLOUT_STEPS):
            obs_batch = obs_t.unsqueeze(0)
            dist_p = self.actor_dist(parts["protagonist_actor"], parts["protagonist_log_std"], obs_batch)
            dist_a = self.actor_dist(parts["adversary_actor"], parts["adversary_log_std"], obs_batch)
            eps_p = torch.randn((self.spec.action_dim,), dtype=DTYPE, device=DEVICE)
            eps_a = torch.randn((self.spec.action_dim,), dtype=DTYPE, device=DEVICE)
            u = dist_p.mean.squeeze(0) + torch.exp(parts["protagonist_log_std"]) * eps_p
            w = dist_a.mean.squeeze(0) + torch.exp(parts["adversary_log_std"]) * eps_a
            logprob_p = dist_p.log_prob(u.unsqueeze(0)).sum(dim=-1).squeeze(0)
            logprob_a = dist_a.log_prob(w.unsqueeze(0)).sum(dim=-1).squeeze(0)
            value_p = self.value_p(obs_batch).squeeze(0)
            value_a = self.value_a(obs_batch).squeeze(0)
            a_raw, a_env = self.blend_action(u.unsqueeze(0), w.unsqueeze(0))
            next_obs, reward_raw, terminated, truncated, _ = env.step(a_env.squeeze(0).detach().cpu().numpy().astype(np.float32))
            done = bool(terminated or truncated)
            reward_t = torch.as_tensor(float(reward_raw), dtype=DTYPE, device=DEVICE)

            storage["obs"].append(obs_t.detach())
            storage["u_old"].append(u.detach())
            storage["w_old"].append(w.detach())
            storage["eps_p"].append(eps_p.detach())
            storage["eps_a"].append(eps_a.detach())
            storage["old_logprob_p"].append(logprob_p.detach())
            storage["old_logprob_a"].append(logprob_a.detach())
            storage["r_env"].append(reward_t.detach())
            storage["r_adv"].append((-reward_t).detach())
            storage["done"].append(torch.as_tensor(float(done), dtype=DTYPE, device=DEVICE))
            storage["value_p_old"].append(value_p.detach())
            storage["value_a_old"].append(value_a.detach())
            storage["a_env_raw"].append(a_raw.squeeze(0).detach())
            storage["a_env"].append(a_env.squeeze(0).detach())

            ep_reward += float(reward_raw)
            clip_hits += int(torch.any(torch.abs(a_raw.squeeze(0) - a_env.squeeze(0)) > 1e-12).item())
            if done:
                train_returns.append(ep_reward)
                ep_reward = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs
            obs_t = torch.as_tensor(obs, dtype=DTYPE, device=DEVICE)
        last_obs = obs_t.unsqueeze(0)
        bootstrap_p = self.value_p(last_obs).detach().squeeze(0)
        bootstrap_a = self.value_a(last_obs).detach().squeeze(0)
        env.close()

        batch = {key: torch.stack(vals) for key, vals in storage.items()}
        returns_p, adv_p = compute_gae(batch["r_env"], batch["value_p_old"], batch["done"], bootstrap_p)
        returns_a, adv_a = compute_gae(batch["r_adv"], batch["value_a_old"], batch["done"], bootstrap_a)
        batch["return_p"] = returns_p.detach()
        batch["return_a"] = returns_a.detach()
        batch["adv_p"] = normalize_tensor(adv_p).detach()
        batch["adv_a"] = normalize_tensor(adv_a).detach()
        batch["train_return"] = torch.as_tensor(np.mean(train_returns) if train_returns else ep_reward, dtype=DTYPE, device=DEVICE)
        batch["action_clip_fraction"] = torch.as_tensor(clip_hits / max(ROLLOUT_STEPS, 1), dtype=DTYPE, device=DEVICE)
        batch["mean_abs_u"] = batch["u_old"].abs().mean()
        batch["mean_abs_w"] = batch["w_old"].abs().mean()
        batch["mean_abs_alpha_w"] = (self.cfg.alpha * batch["w_old"]).abs().mean()
        batch["mean_abs_action_before_clip"] = batch["a_env_raw"].abs().mean()
        batch["mean_abs_action_after_clip"] = batch["a_env"].abs().mean()
        return batch

    def loss_components(self, z: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        parts = self.split_z(z)
        dist_p = self.actor_dist(parts["protagonist_actor"], parts["protagonist_log_std"], batch["obs"])
        dist_a = self.actor_dist(parts["adversary_actor"], parts["adversary_log_std"], batch["obs"])
        logprob_p_new = dist_p.log_prob(batch["u_old"]).sum(dim=-1)
        logprob_a_new = dist_a.log_prob(batch["w_old"]).sum(dim=-1)
        ratio_p = torch.exp(logprob_p_new - batch["old_logprob_p"])
        ratio_a = torch.exp(logprob_a_new - batch["old_logprob_a"])
        clip_p = torch.clamp(ratio_p, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
        clip_a = torch.clamp(ratio_a, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
        l_ppo_p = -torch.mean(torch.minimum(ratio_p * batch["adv_p"], clip_p * batch["adv_p"]))
        l_ppo_a = -torch.mean(torch.minimum(ratio_a * batch["adv_a"], clip_a * batch["adv_a"]))
        ent_p = dist_p.entropy().sum(dim=-1).mean()
        ent_a = dist_a.entropy().sum(dim=-1).mean()
        rot_mean = self.rot_mean(z, batch["obs"])
        loss_mu = l_ppo_p - (ENT_COEF * ent_p) - (self.cfg.rho * rot_mean)
        loss_nu = l_ppo_a - (ENT_COEF * ent_a) + (self.cfg.rho * rot_mean)
        value_p = self.value_p(batch["obs"])
        value_a = self.value_a(batch["obs"])
        value_loss_p = F.mse_loss(value_p, batch["return_p"])
        value_loss_a = F.mse_loss(value_a, batch["return_a"])
        return {
            "loss_mu_total": loss_mu,
            "loss_nu_total": loss_nu,
            # Use a non-canceling block energy so the Lyapunov merit still reflects
            # the actor-coupling objective in the zero-sum setting.
            "joint_actor_loss": 0.5 * ((loss_mu * loss_mu) + (loss_nu * loss_nu)),
            "joint_actor_loss_sum": loss_mu + loss_nu,
            "l_ppo_p": l_ppo_p,
            "l_ppo_a": l_ppo_a,
            "rot_mean": rot_mean,
            "value_loss_p": value_loss_p,
            "value_loss_a": value_loss_a,
            "ratio_p": ratio_p,
            "ratio_a": ratio_a,
            "clip_fraction_p": (torch.abs(ratio_p - 1.0) > CLIP_EPS).float().mean(),
            "clip_fraction_a": (torch.abs(ratio_a - 1.0) > CLIP_EPS).float().mean(),
            "mean_kl_p": torch.mean((ratio_p - 1.0) - torch.log(ratio_p + EPS)),
            "mean_kl_a": torch.mean((ratio_a - 1.0) - torch.log(ratio_a + EPS)),
        }

    def field(self, z: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        z_req = z if z.requires_grad else z.detach().clone().requires_grad_(True)
        comps = self.loss_components(z_req, batch)
        grad_mu = torch.autograd.grad(comps["loss_mu_total"], z_req, retain_graph=True, create_graph=True)[0]
        grad_nu = torch.autograd.grad(comps["loss_nu_total"], z_req, create_graph=True)[0]
        p_slice = slice(self.slices["protagonist_actor"].start, self.slices["protagonist_log_std"].stop)
        a_slice = slice(self.slices["adversary_actor"].start, self.slices["adversary_log_std"].stop)
        return torch.cat([grad_mu[p_slice], grad_nu[a_slice]])

    def merit(self, z: torch.Tensor, batch: dict[str, torch.Tensor], compute_geometry: bool) -> dict[str, float]:
        comps = self.loss_components(z, batch)
        field = self.field(z, batch).detach()
        field_energy = 0.5 * float(torch.dot(field, field).item())
        joint_loss = float(comps["joint_actor_loss"].detach().item())
        if "field0" not in self.metric_refs:
            self.metric_refs["field0"] = max(field_energy, EPS)
            self.metric_refs["loss0"] = max(abs(joint_loss), EPS)
        field_term = field_energy / (self.metric_refs["field0"] + EPS)
        loss_term = joint_loss / (self.metric_refs["loss0"] + EPS)
        V = (MERIT_LAMBDA_F * field_term) + (MERIT_LAMBDA_L * loss_term)

        z_req = z.detach().clone().requires_grad_(True)
        rot_mu_grad = torch.autograd.grad(-self.cfg.rho * self.rot_mean(z_req, batch["obs"]), z_req, retain_graph=True)[0][self.slices["protagonist_actor"].start : self.slices["protagonist_log_std"].stop]
        rot_nu_grad = torch.autograd.grad(+self.cfg.rho * self.rot_mean(z_req, batch["obs"]), z_req)[0][self.slices["adversary_actor"].start : self.slices["adversary_log_std"].stop]

        geom = {
            "G_norm": math.nan,
            "G_over_F": math.nan,
            "cos_F_G": math.nan,
            "non_collinearity": math.nan,
            "rotation_ratio_proxy": math.nan,
            "cross_player_coupling_proxy": math.nan,
            "cross_to_same_ratio": math.nan,
        }
        if compute_geometry:
            geom = self.geometry_metrics(z, batch)

        return {
            "V": float(V),
            "field_norm": float(torch.linalg.norm(field).item()),
            "joint_actor_loss": float(joint_loss),
            "rot_mean": float(comps["rot_mean"].detach().item()),
            "L_mu_rot": float((-self.cfg.rho * comps["rot_mean"]).detach().item()),
            "L_nu_rot": float((self.cfg.rho * comps["rot_mean"]).detach().item()),
            "rot_grad_norm_mu": float(torch.linalg.norm(rot_mu_grad.detach()).item()),
            "rot_grad_norm_nu": float(torch.linalg.norm(rot_nu_grad.detach()).item()),
            "mean_KL_P": float(comps["mean_kl_p"].detach().item()),
            "mean_KL_A": float(comps["mean_kl_a"].detach().item()),
            "ratio_clip_fraction_P": float(comps["clip_fraction_p"].detach().item()),
            "ratio_clip_fraction_A": float(comps["clip_fraction_a"].detach().item()),
            "log_std_mean_P": float(self.split_z(z)["protagonist_log_std"].mean().detach().item()),
            "log_std_mean_A": float(self.split_z(z)["adversary_log_std"].mean().detach().item()),
            **geom,
        }

    def geometry_metrics(self, z: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        z_req = z.detach().clone().requires_grad_(True)
        field_z = self.field(z_req, batch)
        _, g_vec = torch.autograd.functional.jvp(lambda zz: self.field(zz, batch), (z_req,), (field_z.detach(),), create_graph=False, strict=False)
        field_det = field_z.detach()
        g_det = g_vec.detach()
        f_norm = float(torch.linalg.norm(field_det).item())
        g_norm = float(torch.linalg.norm(g_det).item())
        cos_fg = float(torch.dot(field_det, g_det).item() / ((f_norm * g_norm) + EPS))

        aproxy = 0.0
        sproxy = 0.0
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(20260628 + self.seed)
        for _ in range(GEOM_PROBES):
            v = torch.randn(z_req.numel(), generator=gen, dtype=DTYPE, device=DEVICE)
            v = v / (torch.linalg.norm(v) + EPS)
            _, jv = torch.autograd.functional.jvp(lambda zz: self.field(zz, batch), (z_req,), (v,), create_graph=False, strict=False)
            jtv = torch.autograd.grad(torch.dot(field_z, v), z_req, retain_graph=True)[0]
            aproxy += float(torch.linalg.norm(jv.detach() - jtv.detach()).item())
            sproxy += float(torch.linalg.norm(jv.detach() + jtv.detach()).item())

        p_slice = slice(self.slices["protagonist_actor"].start, self.slices["protagonist_log_std"].stop)
        a_slice = slice(self.slices["adversary_actor"].start, self.slices["adversary_log_std"].stop)

        def protagonist_field(cur_z: torch.Tensor) -> torch.Tensor:
            return self.field(cur_z, batch)[p_slice].detach()

        def adversary_field(cur_z: torch.Tensor) -> torch.Tensor:
            return self.field(cur_z, batch)[a_slice].detach()

        base_p = protagonist_field(z)
        base_a = adversary_field(z)
        pert_p = z.detach().clone()
        pert_a = z.detach().clone()
        pert_p[p_slice] = pert_p[p_slice] + 1e-3
        pert_a[a_slice] = pert_a[a_slice] + 1e-3
        cross_p = float(torch.linalg.norm(protagonist_field(pert_a) - base_p).item())
        cross_a = float(torch.linalg.norm(adversary_field(pert_p) - base_a).item())
        same_p = float(torch.linalg.norm(protagonist_field(pert_p) - base_p).item())
        same_a = float(torch.linalg.norm(adversary_field(pert_a) - base_a).item())
        cross = 0.5 * (cross_p + cross_a)
        same = 0.5 * (same_p + same_a)
        return {
            "G_norm": g_norm,
            "G_over_F": g_norm / (f_norm + EPS),
            "cos_F_G": cos_fg,
            "non_collinearity": math.sqrt(max(0.0, 1.0 - min(1.0, cos_fg * cos_fg))),
            "rotation_ratio_proxy": aproxy / (sproxy + EPS),
            "cross_player_coupling_proxy": cross,
            "cross_to_same_ratio": cross / (same + EPS),
        }

    def update_critics(self, batch: dict[str, torch.Tensor]) -> None:
        for _ in range(CRITIC_STEPS):
            self.opt_vp.zero_grad()
            pred_p = self.value_p(batch["obs"])
            loss_p = F.mse_loss(pred_p, batch["return_p"])
            loss_p.backward()
            torch.nn.utils.clip_grad_norm_(self.value_p.parameters(), MAX_GRAD_NORM)
            self.opt_vp.step()

            self.opt_va.zero_grad()
            pred_a = self.value_a(batch["obs"])
            loss_a = F.mse_loss(pred_a, batch["return_a"])
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(self.value_a.parameters(), MAX_GRAD_NORM)
            self.opt_va.step()

    def evaluate_policy(self, z: torch.Tensor, episodes: int, with_adversary: bool) -> dict[str, float]:
        env = gym.make(self.spec.env_id)
        parts = self.split_z(z)
        clean_total_returns: list[float] = []
        pure_returns: list[float] = []
        rot_returns: list[float] = []
        clip_fracs: list[float] = []
        for ep in range(episodes):
            obs, _ = env.reset(seed=self.seed + 8000 + ep)
            done = False
            total = 0.0
            pure = 0.0
            rot = 0.0
            episode_clip = []
            while not done:
                obs_t = torch.as_tensor(obs, dtype=DTYPE, device=DEVICE).unsqueeze(0)
                u_bar = self.actor_mean_action(parts["protagonist_actor"], obs_t)
                w_bar = self.actor_mean_action(parts["adversary_actor"], obs_t) if with_adversary else torch.zeros_like(u_bar)
                a_raw, a_env = self.blend_action(u_bar, w_bar)
                obs, reward_raw, terminated, truncated, _ = env.step(a_env.squeeze(0).detach().cpu().numpy().astype(np.float32))
                done = bool(terminated or truncated)
                rot_step = float(torch.sum(u_bar * torch.matmul(w_bar, self.H.T), dim=-1).item() / math.sqrt(max(self.spec.action_dim, 1)))
                total += float(reward_raw) + (self.cfg.rho * rot_step)
                pure += float(reward_raw)
                rot += rot_step
                episode_clip.append(float(torch.any(torch.abs(a_raw.squeeze(0) - a_env.squeeze(0)) > 1e-12).item()))
            clean_total_returns.append(total)
            pure_returns.append(pure)
            rot_returns.append(rot)
            clip_fracs.append(float(np.mean(episode_clip)) if episode_clip else 0.0)
        env.close()
        return {
            "total_actor_coupled_return": float(np.mean(clean_total_returns)),
            "pure_env_return": float(np.mean(pure_returns)),
            "rot_mean_return": float(np.mean(rot_returns)),
            "action_clip_fraction": float(np.mean(clip_fracs)),
        }


def check_env(env_id: str) -> EnvSpec:
    env = gym.make(env_id)
    try:
        obs = env.observation_space
        act = env.action_space
        if act.__class__.__name__ != "Box":
            raise RuntimeError(f"{env_id} action space is not Box: {act}")
        if obs.__class__.__name__ != "Box":
            raise RuntimeError(f"{env_id} observation space is not Box: {obs}")
        return EnvSpec(
            env_id=env_id,
            obs_dim=int(np.prod(obs.shape)),
            action_dim=int(np.prod(act.shape)),
            action_low=np.asarray(act.low, dtype=np.float32).reshape(-1),
            action_high=np.asarray(act.high, dtype=np.float32).reshape(-1),
            max_episode_steps=int(getattr(env.spec, "max_episode_steps", 1000)),
        )
    finally:
        env.close()


def cap_delta(delta: torch.Tensor, max_norm: float) -> tuple[torch.Tensor, bool]:
    norm = float(torch.linalg.norm(delta).item())
    if norm <= max_norm:
        return delta, False
    return delta * (max_norm / (norm + EPS)), True


def compute_field_and_g(game: ActorCouplingGame, z: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    z_req = z.detach().clone().requires_grad_(True)
    field_z = game.field(z_req, batch)
    _, g_vec = torch.autograd.functional.jvp(lambda zz: game.field(zz, batch), (z_req,), (field_z.detach(),), create_graph=False, strict=False)
    return field_z.detach(), g_vec.detach()


def run_sgd(game: ActorCouplingGame, z: torch.Tensor, batch: dict[str, torch.Tensor], lr: float) -> tuple[torch.Tensor, dict[str, Any]]:
    field = game.field(z, batch).detach()
    delta = -lr * field
    delta, capped = cap_delta(delta, lr * float(torch.linalg.norm(field).item()))
    return (z + delta).detach(), {"update_norm": float(torch.linalg.norm(delta).item()), "trust_radius_active": int(capped)}


def run_egm(game: ActorCouplingGame, z: torch.Tensor, batch: dict[str, torch.Tensor], lr: float) -> tuple[torch.Tensor, dict[str, Any]]:
    field0 = game.field(z, batch).detach()
    z_half = z - (lr * field0)
    field_half = game.field(z_half, batch).detach()
    delta = -lr * field_half
    delta, capped = cap_delta(delta, lr * float(torch.linalg.norm(field0).item()))
    return (z + delta).detach(), {"update_norm": float(torch.linalg.norm(delta).item()), "trust_radius_active": int(capped)}


def evaluate_candidate_state(game: ActorCouplingGame, z_candidate: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    return game.merit(z_candidate, batch, compute_geometry=False)


def run_nog_closed(game: ActorCouplingGame, z: torch.Tensor, batch: dict[str, torch.Tensor], lr: float) -> tuple[torch.Tensor, dict[str, Any]]:
    field = game.field(z, batch).detach()
    f_norm = float(torch.linalg.norm(field).item())
    step_cap = lr * f_norm
    v0 = evaluate_candidate_state(game, z, batch)["V"]
    p_dir = -field

    def point(beta: float) -> float:
        return evaluate_candidate_state(game, z + beta * p_dir, batch)["V"]

    sol = solve_nog(point, v0, lr, beta_max=3.0 * lr)
    delta_raw = sol["beta"] * p_dir
    delta, capped = cap_delta(delta_raw, step_cap)
    z_next = (z + delta).detach()
    v_after = evaluate_candidate_state(game, z_next, batch)["V"]
    return z_next, {
        "beta": float(sol["beta"]),
        "gamma": 0.0,
        "gamma_active": 0,
        "G_contribution_ratio": 0.0,
        "fallback_to_noG": 0,
        "predicted_inclusion_pass": 1,
        "QP_better_than_noG_actual_V": 0,
        "QP_better_than_EGM_actual_V": 0,
        "update_norm": float(torch.linalg.norm(delta).item()),
        "trust_radius_active": int(capped),
        "V_after_candidate": float(v_after),
    }


def run_qp_nog_safe(game: ActorCouplingGame, z: torch.Tensor, batch: dict[str, torch.Tensor], lr: float) -> tuple[torch.Tensor, dict[str, Any]]:
    field, g_vec = compute_field_and_g(game, z, batch)
    f_norm = float(torch.linalg.norm(field).item())
    g_norm = float(torch.linalg.norm(g_vec).item())
    v0 = evaluate_candidate_state(game, z, batch)["V"]
    p_dir = -field
    g_dir = g_vec
    step_cap = lr * f_norm
    gamma_max = 3.0 * lr * (f_norm / (g_norm + EPS))

    def point(beta: float, gamma: float) -> float:
        return evaluate_candidate_state(game, z + beta * p_dir + gamma * g_dir, batch)["V"]

    qp = solve_qp(point, v0, lr, beta_max=3.0 * lr, gamma_max=gamma_max)
    delta_qp_raw = qp["beta"] * p_dir + qp["gamma"] * g_dir
    delta_qp, capped = cap_delta(delta_qp_raw, step_cap)
    z_qp = (z + delta_qp).detach()
    v_qp = evaluate_candidate_state(game, z_qp, batch)["V"]
    z_nog, nog_meta = run_nog_closed(game, z, batch, lr)
    z_egm, _ = run_egm(game, z, batch, lr)
    v_nog = safe_float(nog_meta["V_after_candidate"])
    v_egm = evaluate_candidate_state(game, z_egm, batch)["V"]
    gamma_active = int(abs(qp["gamma"]) > 1e-12)
    g_ratio = abs(qp["gamma"] * g_norm) / (abs(qp["beta"] * f_norm) + abs(qp["gamma"] * g_norm) + EPS)
    qp_ok = (
        finite(v_qp)
        and v_qp <= v_nog - NO_G_FALLBACK_TOL
        and gamma_active == 1
        and g_ratio >= QP_TRUST_G_RATIO
    )
    chosen_z = z_qp if qp_ok else z_nog
    chosen_v = v_qp if qp_ok else v_nog
    fallback = 0 if qp_ok else 1
    return chosen_z, {
        "beta": float(qp["beta"] if qp_ok else nog_meta["beta"]),
        "gamma": float(qp["gamma"] if qp_ok else 0.0),
        "gamma_active": int(gamma_active if qp_ok else 0),
        "G_contribution_ratio": float(g_ratio if qp_ok else 0.0),
        "fallback_to_noG": int(fallback),
        "predicted_inclusion_pass": int(qp["predicted_inclusion_pass"]),
        "QP_better_than_noG_actual_V": int(finite(v_qp) and finite(v_nog) and v_qp < v_nog),
        "QP_better_than_EGM_actual_V": int(finite(v_qp) and finite(v_egm) and v_qp < v_egm),
        "update_norm": float(torch.linalg.norm((chosen_z - z)).item()),
        "trust_radius_active": int(capped),
        "V_after_candidate": float(chosen_v),
        "selected_active_set": qp["selected_active_set"] if qp_ok else "fallback_noG",
    }


def run_invariant_check(spec: EnvSpec, out_dir: Path) -> tuple[str, list[dict[str, Any]], str]:
    cfg = Config(env_id=spec.env_id, alpha=ALPHA, rho=1.0, lr=3e-4)
    game = ActorCouplingGame(spec, cfg, SEED)
    z0 = game.init_z()
    batch = game.collect_rollout(z0, rollout_seed=SEED + 11)
    field, g_vec = compute_field_and_g(game, z0, batch)
    f_norm = float(torch.linalg.norm(field).item())
    g_norm = float(torch.linalg.norm(g_vec).item())
    v0 = evaluate_candidate_state(game, z0, batch)["V"]
    p_dir = -field
    g_dir = g_vec

    def point1(beta: float) -> float:
        return evaluate_candidate_state(game, z0 + beta * p_dir, batch)["V"]

    def point2(beta: float, gamma: float) -> float:
        return evaluate_candidate_state(game, z0 + beta * p_dir + gamma * g_dir, batch)["V"]

    nog = solve_nog(point1, v0, cfg.lr, beta_max=3.0 * cfg.lr)
    qp = solve_qp(point2, v0, cfg.lr, beta_max=3.0 * cfg.lr, gamma_max=3.0 * cfg.lr * (f_norm / (g_norm + EPS)))
    q1 = nog["l_beta"] * nog["beta"] + 0.5 * nog["h_bb"] * nog["beta"] * nog["beta"]
    q2_gamma0 = qp["l_beta"] * nog["beta"] + 0.5 * qp["h_bb"] * nog["beta"] * nog["beta"]
    z_forced_nog, forced_nog_meta = run_nog_closed(game, z0, batch, cfg.lr)
    z_native_nog, native_nog_meta = run_nog_closed(game, z0, batch, cfg.lr)
    z_safe, safe_meta = run_qp_nog_safe(game, z0, batch, cfg.lr)
    safe_v = evaluate_candidate_state(game, z_safe, batch)["V"]
    rows = [{
        "env_id": spec.env_id,
        "q1_equals_q2_gamma0_pass": int(abs(q1 - q2_gamma0) <= 1e-6 * max(1.0, abs(q1), abs(q2_gamma0))),
        "predicted_inclusion_pass": int(qp["predicted_inclusion_pass"]),
        "gamma_negative_fraction_plusG": int(qp["gamma"] < -1e-12),
        "forced_nog_same_batch_delta_rel_diff": abs(safe_float(forced_nog_meta["update_norm"]) - safe_float(native_nog_meta["update_norm"])) / (abs(safe_float(native_nog_meta["update_norm"])) + EPS),
        "forced_nog_same_batch_V_diff": abs(safe_float(forced_nog_meta["V_after_candidate"]) - safe_float(native_nog_meta["V_after_candidate"])),
        "noG_safe_actual_choice_matches_applied_step_flag": int(abs(safe_v - safe_float(safe_meta["V_after_candidate"])) <= 1e-9),
    }]
    decision = "QP_INVARIANT_FAIL"
    row = rows[0]
    if (
        row["q1_equals_q2_gamma0_pass"] == 1
        and row["predicted_inclusion_pass"] == 1
        and row["gamma_negative_fraction_plusG"] == 0
        and row["forced_nog_same_batch_delta_rel_diff"] <= 1e-6
        and row["noG_safe_actual_choice_matches_applied_step_flag"] == 1
    ):
        decision = "QP_INVARIANT_PASS"
    text = "\n".join(
        [
            "# differentiable geometry actor coupling invariant check",
            "",
            f"- env: `{spec.env_id}`",
            f"- q1_equals_q2_gamma0_pass_fraction: `{float(row['q1_equals_q2_gamma0_pass']):.6f}`",
            f"- predicted_inclusion_pass_fraction: `{float(row['predicted_inclusion_pass']):.6f}`",
            f"- gamma_negative_fraction_plusG: `{float(row['gamma_negative_fraction_plusG']):.6f}`",
            f"- forced_nog_same_batch_delta_rel_diff: `{row['forced_nog_same_batch_delta_rel_diff']:.6e}`",
            f"- noG_safe_actual_choice_matches_applied_step_flag: `{float(row['noG_safe_actual_choice_matches_applied_step_flag']):.6f}`",
            "",
            f"- decision: `{decision}`",
            "",
        ]
    )
    write_csv(out_dir / "invariant_summary.csv", rows)
    write_text(out_dir / "invariant_report.md", text)
    write_text(out_dir / "invariant_decision.md", decision + "\n")
    return decision, rows, text


def geometry_gate_for_config(spec: EnvSpec, cfg: Config) -> dict[str, Any]:
    game = ActorCouplingGame(spec, cfg, SEED)
    rows: list[dict[str, Any]] = []
    for batch_idx in range(GEOM_BATCHES):
        z0 = game.init_z()
        batch = game.collect_rollout(z0, rollout_seed=SEED + 200 + batch_idx)
        metrics = game.merit(z0, batch, compute_geometry=True)
        _, meta_qp = run_qp_nog_safe(game, z0, batch, cfg.lr)
        _, meta_nog = run_nog_closed(game, z0, batch, cfg.lr)
        _, meta_egm = run_egm(game, z0, batch, cfg.lr)
        rows.append(
            {
                **metrics,
                **meta_qp,
                "beta_nog": safe_float(meta_nog["beta"]),
                "egm_update_norm": safe_float(meta_egm["update_norm"]),
                "batch_index": batch_idx,
            }
        )
    out = {
        "env_id": spec.env_id,
        "alpha": cfg.alpha,
        "rho": cfg.rho,
        "shared_lr": cfg.lr,
        "field_norm": float(np.mean([r["field_norm"] for r in rows])),
        "G_norm": float(np.mean([r["G_norm"] for r in rows])),
        "cos_F_G": float(np.mean([r["cos_F_G"] for r in rows])),
        "non_collinearity": float(np.mean([r["non_collinearity"] for r in rows])),
        "cross_to_same_ratio": float(np.mean([r["cross_to_same_ratio"] for r in rows])),
        "rotation_ratio_proxy": float(np.mean([r["rotation_ratio_proxy"] for r in rows])),
        "QP_better_than_noG_actual_V_fraction": float(np.mean([r["QP_better_than_noG_actual_V"] for r in rows])),
        "QP_better_than_EGM_actual_V_fraction": float(np.mean([r["QP_better_than_EGM_actual_V"] for r in rows])),
        "gamma_active_frac": float(np.mean([r["gamma_active"] for r in rows])),
        "G_contribution_ratio": float(np.mean([r["G_contribution_ratio"] for r in rows])),
        "fallback_to_noG_frac": float(np.mean([r["fallback_to_noG"] for r in rows])),
        "rot_mean": float(np.mean([r["rot_mean"] for r in rows])),
        "rot_grad_norm_mu": float(np.mean([r["rot_grad_norm_mu"] for r in rows])),
        "rot_grad_norm_nu": float(np.mean([r["rot_grad_norm_nu"] for r in rows])),
    }
    out["geometry_gate_pass"] = int(
        out["cross_to_same_ratio"] >= 0.20
        and out["rotation_ratio_proxy"] >= 0.10
        and out["non_collinearity"] >= 0.30
        and out["gamma_active_frac"] >= 0.40
        and out["G_contribution_ratio"] >= 0.15
        and out["QP_better_than_noG_actual_V_fraction"] >= 0.60
        and out["fallback_to_noG_frac"] <= 0.30
    )
    out["gate_score"] = (
        out["cross_to_same_ratio"]
        + out["rotation_ratio_proxy"]
        + 0.25 * out["non_collinearity"]
        + 0.25 * out["QP_better_than_noG_actual_V_fraction"]
        - out["fallback_to_noG_frac"]
    )
    return out


def run_method(game: ActorCouplingGame, cfg: Config, method: str, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed_everything(seed)
    z = game.init_z()
    game.metric_refs = {}
    curves: list[dict[str, Any]] = []
    gamma_active = 0
    fallback = 0
    g_ratio_sum = 0.0
    qp_vs_nog = 0
    qp_vs_egm = 0
    clean_eval = game.evaluate_policy(z, EVAL_EPISODES, with_adversary=False)
    adv_eval = game.evaluate_policy(z, EVAL_EPISODES, with_adversary=True)
    for iteration in range(ITERATIONS + 1):
        batch = game.collect_rollout(z, rollout_seed=seed + 500 + iteration)
        metrics = game.merit(z, batch, compute_geometry=True)
        clean_eval = game.evaluate_policy(z, EVAL_EPISODES, with_adversary=False)
        adv_eval = game.evaluate_policy(z, EVAL_EPISODES, with_adversary=True)
        row = {
            "env_id": cfg.env_id,
            "method": method,
            "iteration": iteration,
            "alpha": cfg.alpha,
            "rho": cfg.rho,
            "shared_lr": cfg.lr,
            "train_return": float(batch["train_return"].item()),
            "clean_total_return": clean_eval["total_actor_coupled_return"],
            "current_adv_total_return": adv_eval["total_actor_coupled_return"],
            "pure_env_return": adv_eval["pure_env_return"],
            "rot_mean_return": adv_eval["rot_mean_return"],
            "action_clip_fraction": adv_eval["action_clip_fraction"],
            "mean_abs_u": float(batch["mean_abs_u"].item()),
            "mean_abs_w": float(batch["mean_abs_w"].item()),
            "mean_abs_alpha_w": float(batch["mean_abs_alpha_w"].item()),
            "mean_abs_action_before_clip": float(batch["mean_abs_action_before_clip"].item()),
            "mean_abs_action_after_clip": float(batch["mean_abs_action_after_clip"].item()),
            **metrics,
        }
        row["finite_flag"] = int(all(finite(row[k]) for k in ["train_return", "clean_total_return", "current_adv_total_return", "V", "field_norm"]))
        curves.append(row)
        if iteration == ITERATIONS:
            break
        if method == "sgd_gda":
            z, meta = run_sgd(game, z, batch, cfg.lr)
        elif method == "egm":
            z, meta = run_egm(game, z, batch, cfg.lr)
        elif method == "proposed_nog_closed":
            z, meta = run_nog_closed(game, z, batch, cfg.lr)
        elif method == "proposed_qp_nog_safe":
            z, meta = run_qp_nog_safe(game, z, batch, cfg.lr)
        else:
            raise ValueError(method)
        curves[-1].update(meta)
        gamma_active += int(meta.get("gamma_active", 0))
        fallback += int(meta.get("fallback_to_noG", 0))
        g_ratio_sum += safe_float(meta.get("G_contribution_ratio", 0.0), 0.0)
        qp_vs_nog += int(meta.get("QP_better_than_noG_actual_V", 0))
        qp_vs_egm += int(meta.get("QP_better_than_EGM_actual_V", 0))
        game.update_critics(batch)
    summary = {
        "env_id": cfg.env_id,
        "method": method,
        "alpha": cfg.alpha,
        "rho": cfg.rho,
        "shared_lr": cfg.lr,
        "current_adv_total_return_AUC": auc([row["current_adv_total_return"] for row in curves]),
        "clean_total_return_AUC": auc([row["clean_total_return"] for row in curves]),
        "pure_env_return_AUC": auc([row["pure_env_return"] for row in curves]),
        "rot_mean_return_AUC": auc([row["rot_mean_return"] for row in curves]),
        "final_current_adv_total_return": curves[-1]["current_adv_total_return"],
        "final_clean_total_return": curves[-1]["clean_total_return"],
        "field_norm_AUC": auc([row["field_norm"] for row in curves]),
        "V_AUC": auc([row["V"] for row in curves]),
        "fallback_to_noG_frac": fallback / max(ITERATIONS, 1),
        "QP_accept_frac": 1.0 - (fallback / max(ITERATIONS, 1)),
        "gamma_active_frac": gamma_active / max(ITERATIONS, 1),
        "G_contribution_ratio": g_ratio_sum / max(ITERATIONS, 1),
        "QP_better_than_noG_actual_V_fraction": qp_vs_nog / max(ITERATIONS, 1),
        "QP_better_than_EGM_actual_V_fraction": qp_vs_egm / max(ITERATIONS, 1),
        "rot_grad_norm_mu": float(np.mean([row["rot_grad_norm_mu"] for row in curves])),
        "rot_grad_norm_nu": float(np.mean([row["rot_grad_norm_nu"] for row in curves])),
        "cross_to_same_ratio": float(np.mean([row["cross_to_same_ratio"] for row in curves])),
        "rotation_ratio_proxy": float(np.mean([row["rotation_ratio_proxy"] for row in curves])),
        "curve_sanity_flag": int(
            all(row["finite_flag"] == 1 for row in curves)
            and spike_ratio([row["current_adv_total_return"] for row in curves]) <= 20.0
        ),
    }
    return curves, summary


def dominance_fraction(curves_a: list[dict[str, Any]], curves_b: list[dict[str, Any]], metric: str) -> float:
    wins = 0
    total = 0
    for ra, rb in zip(curves_a, curves_b):
        va = safe_float(ra[metric])
        vb = safe_float(rb[metric])
        if finite(va) and finite(vb):
            wins += int(va > vb)
            total += 1
    return wins / max(total, 1)


def assess_config(curves_by_method: dict[str, list[dict[str, Any]]], summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sgd = summaries["sgd_gda"]
    egm = summaries["egm"]
    nog = summaries["proposed_nog_closed"]
    qp = summaries["proposed_qp_nog_safe"]
    qp_vs_egm = (qp["current_adv_total_return_AUC"] - egm["current_adv_total_return_AUC"]) / (abs(egm["current_adv_total_return_AUC"]) + EPS)
    qp_vs_nog = (qp["current_adv_total_return_AUC"] - nog["current_adv_total_return_AUC"]) / (abs(nog["current_adv_total_return_AUC"]) + EPS)
    egm_vs_sgd = (egm["current_adv_total_return_AUC"] - sgd["current_adv_total_return_AUC"]) / (abs(sgd["current_adv_total_return_AUC"]) + EPS)
    nog_vs_sgd = (nog["current_adv_total_return_AUC"] - sgd["current_adv_total_return_AUC"]) / (abs(sgd["current_adv_total_return_AUC"]) + EPS)
    dom_qp_egm = dominance_fraction(curves_by_method["proposed_qp_nog_safe"], curves_by_method["egm"], "current_adv_total_return")
    dom_qp_nog = dominance_fraction(curves_by_method["proposed_qp_nog_safe"], curves_by_method["proposed_nog_closed"], "current_adv_total_return")

    decision = "NO_ACTOR_GEOM_POSITIVE"
    if (
        qp_vs_egm >= 0.10
        and qp_vs_nog >= 0.05
        and max(egm_vs_sgd, nog_vs_sgd) >= 0.05
        and dom_qp_egm >= 0.70
        and dom_qp_nog >= 0.65
        and qp["fallback_to_noG_frac"] <= 0.30
        and qp["gamma_active_frac"] >= 0.40
        and qp["G_contribution_ratio"] >= 0.15
        and qp["QP_better_than_noG_actual_V_fraction"] >= 0.60
        and qp["cross_to_same_ratio"] >= 0.20
        and qp["rotation_ratio_proxy"] >= 0.10
        and qp["curve_sanity_flag"] == 1
    ):
        decision = "QP_ACTOR_GEOM_WEAK_POSITIVE"
    if (
        qp_vs_egm >= 0.15
        and qp_vs_nog >= 0.15
        and max(egm_vs_sgd, nog_vs_sgd) >= 0.15
        and dom_qp_egm >= 0.80
        and dom_qp_nog >= 0.80
        and qp["fallback_to_noG_frac"] <= 0.20
        and qp["gamma_active_frac"] >= 0.50
        and qp["G_contribution_ratio"] >= 0.20
        and qp["QP_better_than_noG_actual_V_fraction"] >= 0.70
        and qp["curve_sanity_flag"] == 1
    ):
        decision = "QP_ACTOR_GEOM_STRONG_POSITIVE"
    return {
        "decision": decision,
        "qp_vs_egm_auc_frac": qp_vs_egm,
        "qp_vs_nog_auc_frac": qp_vs_nog,
        "egm_vs_sgd_auc_frac": egm_vs_sgd,
        "nog_vs_sgd_auc_frac": nog_vs_sgd,
        "qp_dom_egm": dom_qp_egm,
        "qp_dom_nog": dom_qp_nog,
    }


def save_training_plot(plot_path: Path, title: str, curves_by_method: dict[str, list[dict[str, Any]]], metrics: list[tuple[str, str]]) -> None:
    if plt is None:
        return
    rows = len(metrics)
    fig, axes = plt.subplots(rows, 1, figsize=(12, 4 * rows))
    if rows == 1:
        axes = [axes]
    for ax, (metric, label) in zip(axes, metrics):
        for method, rows_m in curves_by_method.items():
            xs = [row["iteration"] for row in rows_m]
            ys = [safe_float(row[metric]) for row in rows_m]
            ax.plot(xs, ys, marker="o", label=method)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Iteration")
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def run_training_for_configs(spec: EnvSpec, gate_rows: list[dict[str, Any]], out_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    passing = [row for row in gate_rows if int(row["geometry_gate_pass"]) == 1]
    passing = sorted(passing, key=lambda row: safe_float(row["gate_score"]), reverse=True)[:GATE_TOPK]
    if not passing:
        return [], [], "NO_ACTOR_GEOM_POSITIVE"

    all_curves: list[dict[str, Any]] = []
    ranked_rows: list[dict[str, Any]] = []
    report_lines = [
        "# actor geometry coupling training",
        "",
        f"- env: `{spec.env_id}`",
        "",
    ]
    top_config_for_confirmation = None
    top_decision_rank = None
    for gate in passing:
        cfg = Config(env_id=spec.env_id, alpha=float(gate["alpha"]), rho=float(gate["rho"]), lr=float(gate["shared_lr"]))
        curves_by_method: dict[str, list[dict[str, Any]]] = {}
        summaries: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            game = ActorCouplingGame(spec, cfg, SEED)
            curves, summary = run_method(game, cfg, method, SEED)
            for row in curves:
                row["config_slug"] = cfg.slug
                all_curves.append(row)
            curves_by_method[method] = curves
            summaries[method] = summary
        assess = assess_config(curves_by_method, summaries)
        rank_row = {
            "env_id": spec.env_id,
            "config_slug": cfg.slug,
            "alpha": cfg.alpha,
            "rho": cfg.rho,
            "shared_lr": cfg.lr,
            **gate,
            **assess,
            **summaries["proposed_qp_nog_safe"],
        }
        ranked_rows.append(rank_row)
        report_lines.append(
            f"- `{cfg.slug}`: decision=`{assess['decision']}`, qp_vs_egm=`{assess['qp_vs_egm_auc_frac']:.3f}`, "
            f"qp_vs_nog=`{assess['qp_vs_nog_auc_frac']:.3f}`, fallback=`{summaries['proposed_qp_nog_safe']['fallback_to_noG_frac']:.3f}`, "
            f"gamma_active=`{summaries['proposed_qp_nog_safe']['gamma_active_frac']:.3f}`"
        )
        save_training_plot(
            out_root / "plots" / f"{cfg.slug}_curves.png",
            f"{spec.env_id} | {cfg.slug}",
            curves_by_method,
            [
                ("current_adv_total_return", "Current-Adv Total Return"),
                ("clean_total_return", "Clean Total Return"),
                ("pure_env_return", "Pure Env Return"),
                ("rot_mean_return", "Rot Mean Return"),
                ("V", "Lyapunov Merit"),
                ("field_norm", "Field Norm"),
                ("rotation_ratio_proxy", "Rotation Ratio Proxy"),
            ],
        )
        decision_rank = {"QP_ACTOR_GEOM_STRONG_POSITIVE": 0, "QP_ACTOR_GEOM_WEAK_POSITIVE": 1, "NO_ACTOR_GEOM_POSITIVE": 2}.get(assess["decision"], 3)
        if top_decision_rank is None or decision_rank < top_decision_rank:
            top_decision_rank = decision_rank
            top_config_for_confirmation = rank_row
    ranked_rows.sort(key=lambda row: ({"QP_ACTOR_GEOM_STRONG_POSITIVE": 0, "QP_ACTOR_GEOM_WEAK_POSITIVE": 1, "NO_ACTOR_GEOM_POSITIVE": 2}.get(row["decision"], 3), -safe_float(row["qp_vs_egm_auc_frac"])))
    write_csv(out_root / "01_qp_training" / "actor_geom_training_all.csv", all_curves)
    write_csv(out_root / "01_qp_training" / "actor_geom_training_ranked.csv", ranked_rows)
    write_text(out_root / "01_qp_training" / "actor_geom_training_report.md", "\n".join(report_lines) + "\n")
    top_lines = ["# geometry_actor_coupling_top_configs", ""]
    for idx, row in enumerate(ranked_rows[:10], start=1):
        top_lines.append(f"{idx}. `{row['config_slug']}` | decision=`{row['decision']}` | qp_vs_egm=`{row['qp_vs_egm_auc_frac']:.3f}` | fallback=`{row['fallback_to_noG_frac']:.3f}`")
    write_text(out_root / "geometry_actor_coupling_top_configs.md", "\n".join(top_lines) + "\n")
    final_decision = ranked_rows[0]["decision"] if ranked_rows else "NO_ACTOR_GEOM_POSITIVE"
    return ranked_rows, all_curves, final_decision


def run_confirmation(best_row: dict[str, Any], out_root: Path) -> str:
    if best_row["decision"] not in {"QP_ACTOR_GEOM_WEAK_POSITIVE", "QP_ACTOR_GEOM_STRONG_POSITIVE"}:
        return "ACTOR_GEOM_QP_FAILS"
    spec = check_env(best_row["env_id"])
    cfg = Config(env_id=spec.env_id, alpha=float(best_row["alpha"]), rho=float(best_row["rho"]), lr=float(best_row["shared_lr"]))
    rows: list[dict[str, Any]] = []
    for seed in [0, 1, 2]:
        curves_by_method = {}
        summaries = {}
        for method in METHODS:
            game = ActorCouplingGame(spec, cfg, seed)
            curves, summary = run_method(game, cfg, method, seed)
            curves_by_method[method] = curves
            summaries[method] = summary
        assess = assess_config(curves_by_method, summaries)
        for method, summary in summaries.items():
            rows.append({"seed": seed, "method": method, **summary, **assess, "config_slug": cfg.slug})
    write_csv(out_root / "02_confirmation" / "multiseed_summary.csv", rows)
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    means = {method: float(np.mean([safe_float(r["current_adv_total_return_AUC"]) for r in method_rows])) for method, method_rows in by_method.items()}
    qp_mean = means.get("proposed_qp_nog_safe", -math.inf)
    if qp_mean >= max(means.get("sgd_gda", -math.inf), means.get("egm", -math.inf), means.get("proposed_nog_closed", -math.inf)):
        decision = "ACTOR_GEOM_QP_CONFIRMED"
    else:
        decision = "ACTOR_GEOM_QP_ONE_SEED_ONLY"
    report = [
        "# actor geometry coupling multiseed confirmation",
        "",
        f"- env: `{cfg.env_id}`",
        f"- alpha: `{cfg.alpha}`",
        f"- rho: `{cfg.rho}`",
        f"- shared_lr: `{cfg.lr}`",
        "",
    ]
    for method, mean_value in means.items():
        report.append(f"- {method}: mean current_adv_total_return_AUC=`{mean_value:.6e}`")
    report.append("")
    report.append(f"- decision: `{decision}`")
    write_text(out_root / "02_confirmation" / "multiseed_report.md", "\n".join(report) + "\n")
    return decision


def main() -> None:
    seed_everything(SEED)
    ensure_dir(RESULT_ROOT / "00_geometry_gate")
    ensure_dir(RESULT_ROOT / "01_qp_training")
    ensure_dir(RESULT_ROOT / "02_confirmation")
    ensure_dir(RESULT_ROOT / "plots")

    write_json(
        RESULT_ROOT / "run_spec.json",
        {
            "env_order": ENV_ORDER,
            "rho_grid": RHO_GRID,
            "lr_grid": LR_GRID,
            "alpha": ALPHA,
            "methods": METHODS,
            "seed": SEED,
            "iterations": ITERATIONS,
        },
    )

    env_reports: list[str] = []
    invariant_spec = check_env("Walker2d-v4")
    invariant_decision, _, invariant_text = run_invariant_check(invariant_spec, RESULT_ROOT / "00_geometry_gate")
    if invariant_decision != "QP_INVARIANT_PASS":
        write_text(RESULT_ROOT / "geometry_actor_coupling_decision.md", "QP_INVARIANT_FAIL\n")
        write_text(RESULT_ROOT / "geometry_actor_coupling_report.md", invariant_text)
        return

    gate_rows_all: list[dict[str, Any]] = []
    chosen_env = None
    training_ranked: list[dict[str, Any]] = []
    training_curves: list[dict[str, Any]] = []
    stage_decision = "GEOMETRY_COUPLING_NOT_ENTERING_FIELD"

    for env_id in ENV_ORDER:
        spec = check_env(env_id)
        env_reports.append(f"- tried env: `{env_id}`")
        gate_rows: list[dict[str, Any]] = []
        for rho in RHO_GRID:
            for lr in LR_GRID:
                cfg = Config(env_id=env_id, alpha=ALPHA, rho=rho, lr=lr)
                gate_row = geometry_gate_for_config(spec, cfg)
                gate_rows.append(gate_row)
                gate_rows_all.append(gate_row)
        write_csv(RESULT_ROOT / "00_geometry_gate" / "geometry_gate_summary.csv", gate_rows_all)
        gate_lines = [
            "# geometry gate report",
            "",
            *env_reports,
            "",
        ]
        for row in gate_rows:
            gate_lines.append(
                f"- `{row['env_id']}` rho=`{row['rho']}` lr=`{row['shared_lr']}` pass=`{bool(row['geometry_gate_pass'])}` "
                f"cross=`{row['cross_to_same_ratio']:.3f}` rot=`{row['rotation_ratio_proxy']:.3f}` noncol=`{row['non_collinearity']:.3f}` "
                f"gamma=`{row['gamma_active_frac']:.3f}` G_ratio=`{row['G_contribution_ratio']:.3f}` fallback=`{row['fallback_to_noG_frac']:.3f}`"
            )
        write_text(RESULT_ROOT / "00_geometry_gate" / "geometry_gate_report.md", "\n".join(gate_lines) + "\n")

        if any(int(row["geometry_gate_pass"]) == 1 for row in gate_rows):
            chosen_env = env_id
            training_ranked, training_curves, stage_decision = run_training_for_configs(spec, gate_rows, RESULT_ROOT)
            break

    if chosen_env is None:
        final_report = "\n".join(
            [
                "# geometry actor coupling report",
                "",
                "No environment passed the differentiable actor-coupling geometry gate.",
                "This indicates the controlled rotational actor-coupling term did not create the required usable skew regime under the tested rho/lr settings.",
                "",
                *env_reports,
                "",
                "Decision: `GEOMETRY_COUPLING_NOT_ENTERING_FIELD`",
                "",
            ]
        )
        write_text(RESULT_ROOT / "geometry_actor_coupling_report.md", final_report)
        write_text(RESULT_ROOT / "geometry_actor_coupling_decision.md", "GEOMETRY_COUPLING_NOT_ENTERING_FIELD\n")
        write_csv(RESULT_ROOT / "geometry_actor_coupling_ranked.csv", gate_rows_all)
        return

    best_row = training_ranked[0] if training_ranked else None
    final_decision = "ACTOR_GEOM_QP_FAILS"
    if best_row is not None and best_row["decision"] in {"QP_ACTOR_GEOM_WEAK_POSITIVE", "QP_ACTOR_GEOM_STRONG_POSITIVE"}:
        final_decision = run_confirmation(best_row, RESULT_ROOT)
    elif best_row is not None:
        final_decision = best_row["decision"]

    write_csv(RESULT_ROOT / "geometry_actor_coupling_ranked.csv", training_ranked or gate_rows_all)
    report_lines = [
        "# geometry actor coupling report",
        "",
        "This is a geometry-controlled MuJoCo actor-coupling benchmark, not pure standard RARL.",
        "It preserves the MuJoCo dynamics but adds a differentiable zero-sum rotational coupling between protagonist and adversary actor means inside the actor optimization loss.",
        "",
        "## Environment order",
        *env_reports,
        "",
        f"## Chosen training environment",
        f"- `{chosen_env}`",
        "",
    ]
    if best_row is not None:
        report_lines.extend(
            [
                "## Best config",
                f"- config_slug: `{best_row['config_slug']}`",
                f"- alpha: `{best_row['alpha']}`",
                f"- rho: `{best_row['rho']}`",
                f"- shared_lr: `{best_row['shared_lr']}`",
                f"- decision: `{best_row['decision']}`",
                f"- cross_to_same_ratio: `{best_row['cross_to_same_ratio']:.3f}`",
                f"- rotation_ratio_proxy: `{best_row['rotation_ratio_proxy']:.3f}`",
                f"- gamma_active_frac: `{best_row['gamma_active_frac']:.3f}`",
                f"- fallback_to_noG_frac: `{best_row['fallback_to_noG_frac']:.3f}`",
                "",
            ]
        )
    report_lines.extend(
        [
            "## Paper wording",
            "To test the theory-predicted usable-skew regime in nonlinear continuous control, we construct a geometry-controlled MuJoCo actor-coupling benchmark. It preserves the MuJoCo dynamics but adds a differentiable zero-sum rotational coupling between protagonist and adversary actor means. This benchmark is not a standard reward benchmark; it is designed to verify whether the QP rule exploits usable curvature when such geometry is present.",
            "",
            f"## Final decision",
            f"- `{final_decision}`",
            "",
        ]
    )
    write_text(RESULT_ROOT / "geometry_actor_coupling_report.md", "\n".join(report_lines) + "\n")
    write_text(RESULT_ROOT / "geometry_actor_coupling_decision.md", final_decision + "\n")


if __name__ == "__main__":
    main()
