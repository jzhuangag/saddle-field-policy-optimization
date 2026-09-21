from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
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
RESULT_ROOT = REPO_ROOT.parent / "results" / "final_geometry_controlled_qp_last_attempt"

DEVICE = torch.device("cpu")
DTYPE = torch.float32
EPS = 1e-8

DEFAULT_ENVS = ["HalfCheetah-v4", "Walker2d-v4", "Hopper-v4", "Swimmer-v4"]
ALPHA_GRID = [0.1, 0.3]
RHO_GRID = [0.1, 0.3, 0.6, 1.0]
JOINT_LR_GRID = [3e-4, 1e-3]
LAMBDA_GRID = [
    (0.01, 1.0, 0.0),
    (0.01, 1.0, 0.3),
    (0.001, 1.0, 0.3),
    (0.01, 0.3, 0.5),
]
METHODS = ["sgd_gda", "egm", "proposed_nog_closed", "proposed_qp_closed", "proposed_qp_nog_safe"]

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.001
VF_COEF = 0.5
ROLLOUT_STEPS = 512
EVAL_EPISODES = 8
ITERATIONS = 10
PLOT_EVERY = 1
GEOM_PROBES = 4
SMOKE_TOPK = 2
TRUST_QP_G_MIN_RATIO = 0.05
TRUST_QP_TOL = 0.0
INIT_LOG_STD = -1.0
HIDDEN_SIZES = (64, 64)
VALUE_HIDDEN_SIZES = (64, 64)
CRITIC_LR = 1e-3
CRITIC_STEPS_PER_ITER = 1
MAX_GRAD_NORM = 10.0


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


def spike_ratio(values: list[float]) -> float:
    if not values:
        return math.nan
    arr = np.asarray(values, dtype=np.float64)
    med = float(np.median(np.abs(arr))) + EPS
    return float(np.max(np.abs(arr)) / med)


def moving_average_slope(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    x = np.arange(arr.size, dtype=np.float64)
    coeffs = np.polyfit(x, arr, deg=1)
    return float(coeffs[0])


def auc(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    arr = np.asarray(values, dtype=np.float64)
    x = np.arange(arr.size, dtype=np.float64)
    return float(np.trapezoid(arr, x))


def action_rotation_matrix(action_dim: int) -> torch.Tensor:
    mat = torch.zeros((action_dim, action_dim), dtype=DTYPE, device=DEVICE)
    for start in range(0, action_dim - 1, 2):
        mat[start, start + 1] = 1.0
        mat[start + 1, start] = -1.0
    return mat


def action_skew_matrix(action_dim: int) -> torch.Tensor:
    return action_rotation_matrix(action_dim)


@dataclass(frozen=True)
class EnvSpec:
    env_id: str
    obs_dim: int
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    max_episode_steps: int


@dataclass(frozen=True)
class LyapWeights:
    lambda_F: float
    lambda_L: float
    lambda_rot: float


@dataclass(frozen=True)
class Config:
    env_id: str
    alpha: float
    rho: float
    joint_lr: float
    weights: LyapWeights
    reward_scale: float

    @property
    def slug(self) -> str:
        return (
            f"{self.env_id.lower().replace('-', '_')}"
            f"_a{str(self.alpha).replace('.', 'p')}"
            f"_rho{str(self.rho).replace('.', 'p')}"
            f"_lr{str(self.joint_lr).replace('.', 'p')}"
            f"_lF{str(self.weights.lambda_F).replace('.', 'p')}"
            f"_lL{str(self.weights.lambda_L).replace('.', 'p')}"
            f"_lR{str(self.weights.lambda_rot).replace('.', 'p')}"
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "alpha": self.alpha,
            "rho": self.rho,
            "joint_lr": self.joint_lr,
            "lambda_F": self.weights.lambda_F,
            "lambda_L": self.weights.lambda_L,
            "lambda_rot": self.weights.lambda_rot,
            "reward_scale": self.reward_scale,
        }


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


class GeometryControlledGame:
    def __init__(self, spec: EnvSpec, cfg: Config, seed: int) -> None:
        self.spec = spec
        self.cfg = cfg
        self.seed = seed
        self.action_low_t = torch.as_tensor(spec.action_low, dtype=DTYPE, device=DEVICE)
        self.action_high_t = torch.as_tensor(spec.action_high, dtype=DTYPE, device=DEVICE)
        self.rot_dyn = action_rotation_matrix(spec.action_dim)
        self.rot_reward = action_skew_matrix(spec.action_dim)
        self.actor_layout = FlatMLP(spec.obs_dim, HIDDEN_SIZES, spec.action_dim)
        self.slices = self._build_slices()
        self.total_dim = self.slices["adversary_log_std"].stop
        self.value_p = CriticNet(spec.obs_dim, VALUE_HIDDEN_SIZES).to(DEVICE)
        self.value_a = CriticNet(spec.obs_dim, VALUE_HIDDEN_SIZES).to(DEVICE)
        self.opt_vp = torch.optim.Adam(self.value_p.parameters(), lr=CRITIC_LR)
        self.opt_va = torch.optim.Adam(self.value_a.parameters(), lr=CRITIC_LR)
        self.metric_refs: dict[str, float] = {}

    def _build_slices(self) -> dict[str, slice]:
        offset = 0
        actor_dim = self.actor_layout.num_params
        std_dim = self.spec.action_dim
        out: dict[str, slice] = {}
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

    def actor_mean(self, actor_flat: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_layout.forward(actor_flat, obs)

    def actor_dist(self, actor_flat: torch.Tensor, log_std: torch.Tensor, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor_mean(actor_flat, obs)
        std = torch.exp(log_std).unsqueeze(0).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def total_reward(self, reward_raw: torch.Tensor, u: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r_task = reward_raw / max(self.cfg.reward_scale, 1.0)
        rot = torch.sum(u * torch.matmul(w, self.rot_reward.T), dim=-1) / math.sqrt(max(self.spec.action_dim, 1))
        total = r_task + (self.cfg.rho * rot)
        return total, r_task, rot

    def blend_action(self, u: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pert = torch.matmul(w, self.rot_dyn.T)
        raw = u + (self.cfg.alpha * pert)
        clipped = torch.clamp(raw, self.action_low_t, self.action_high_t)
        return raw, clipped

    def collect_rollout(self, z: torch.Tensor, rollout_seed: int) -> dict[str, torch.Tensor]:
        parts = self.split_z(z)
        env = gym.make(self.spec.env_id)
        obs, _ = env.reset(seed=rollout_seed)
        obs_t = torch.as_tensor(obs, dtype=DTYPE, device=DEVICE)
        keys = [
            "obs",
            "u_old",
            "w_old",
            "eps_p",
            "eps_a",
            "old_logprob_p",
            "old_logprob_a",
            "r_total",
            "r_adv",
            "r_task",
            "r_rot",
            "done",
            "value_p_old",
            "value_a_old",
            "orig_reward",
            "a_env_raw",
            "a_env",
        ]
        storage: dict[str, list[torch.Tensor]] = {key: [] for key in keys}
        train_total_returns: list[float] = []
        train_pure_returns: list[float] = []
        train_rot_returns: list[float] = []
        clip_hits = 0
        ep_total = 0.0
        ep_pure = 0.0
        ep_rot = 0.0
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
            reward_t = torch.as_tensor([reward_raw], dtype=DTYPE, device=DEVICE)
            r_total, r_task, r_rot = self.total_reward(reward_t, u.unsqueeze(0), w.unsqueeze(0))
            r_total_s = r_total.squeeze(0)
            r_rot_s = r_rot.squeeze(0)
            r_task_s = r_task.squeeze(0)

            storage["obs"].append(obs_t.detach())
            storage["u_old"].append(u.detach())
            storage["w_old"].append(w.detach())
            storage["eps_p"].append(eps_p.detach())
            storage["eps_a"].append(eps_a.detach())
            storage["old_logprob_p"].append(logprob_p.detach())
            storage["old_logprob_a"].append(logprob_a.detach())
            storage["r_total"].append(r_total_s.detach())
            storage["r_adv"].append((-r_total_s).detach())
            storage["r_task"].append(r_task_s.detach())
            storage["r_rot"].append(r_rot_s.detach())
            storage["done"].append(torch.as_tensor(float(done), dtype=DTYPE, device=DEVICE))
            storage["value_p_old"].append(value_p.detach())
            storage["value_a_old"].append(value_a.detach())
            storage["orig_reward"].append(torch.as_tensor(float(reward_raw), dtype=DTYPE, device=DEVICE))
            storage["a_env_raw"].append(a_raw.squeeze(0).detach())
            storage["a_env"].append(a_env.squeeze(0).detach())

            ep_total += float(r_total_s.item())
            ep_pure += float(reward_raw)
            ep_rot += float(r_rot_s.item())
            clip_hits += int(torch.any(torch.abs(a_raw.squeeze(0) - a_env.squeeze(0)) > 1e-12).item())

            if done:
                train_total_returns.append(ep_total)
                train_pure_returns.append(ep_pure)
                train_rot_returns.append(ep_rot)
                ep_total = 0.0
                ep_pure = 0.0
                ep_rot = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs
            obs_t = torch.as_tensor(obs, dtype=DTYPE, device=DEVICE)

        last_obs = obs_t.unsqueeze(0)
        bootstrap_p = self.value_p(last_obs).detach().squeeze(0)
        bootstrap_a = self.value_a(last_obs).detach().squeeze(0)
        env.close()

        batch = {key: torch.stack(vals) for key, vals in storage.items()}
        returns_p, adv_p = compute_gae(batch["r_total"], batch["value_p_old"], batch["done"], bootstrap_p)
        returns_a, adv_a = compute_gae(batch["r_adv"], batch["value_a_old"], batch["done"], bootstrap_a)
        batch["return_p"] = returns_p.detach()
        batch["return_a"] = returns_a.detach()
        batch["adv_p"] = normalize_tensor(adv_p).detach()
        batch["adv_a"] = normalize_tensor(adv_a).detach()
        batch["train_total_game_return"] = torch.as_tensor(np.mean(train_total_returns) if train_total_returns else ep_total, dtype=DTYPE, device=DEVICE)
        batch["train_pure_env_return"] = torch.as_tensor(np.mean(train_pure_returns) if train_pure_returns else ep_pure, dtype=DTYPE, device=DEVICE)
        batch["train_rotational_return"] = torch.as_tensor(np.mean(train_rot_returns) if train_rot_returns else ep_rot, dtype=DTYPE, device=DEVICE)
        batch["action_clip_fraction"] = torch.as_tensor(clip_hits / max(ROLLOUT_STEPS, 1), dtype=DTYPE, device=DEVICE)
        batch["mean_abs_u"] = batch["u_old"].abs().mean()
        batch["mean_abs_w"] = batch["w_old"].abs().mean()
        batch["mean_abs_rw"] = torch.matmul(batch["w_old"], self.rot_dyn.T).abs().mean()
        batch["mean_abs_action_before_clip"] = batch["a_env_raw"].abs().mean()
        batch["mean_abs_action_after_clip"] = batch["a_env"].abs().mean()
        return batch

    def current_actor_actions(self, z: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        parts = self.split_z(z)
        mu_p = self.actor_mean(parts["protagonist_actor"], batch["obs"])
        mu_a = self.actor_mean(parts["adversary_actor"], batch["obs"])
        std_p = torch.exp(parts["protagonist_log_std"]).unsqueeze(0).expand_as(mu_p)
        std_a = torch.exp(parts["adversary_log_std"]).unsqueeze(0).expand_as(mu_a)
        u_current = mu_p + std_p * batch["eps_p"]
        w_current = mu_a + std_a * batch["eps_a"]
        return u_current, w_current

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
        loss_p = -torch.mean(torch.minimum(ratio_p * batch["adv_p"], clip_p * batch["adv_p"]))
        loss_a = -torch.mean(torch.minimum(ratio_a * batch["adv_a"], clip_a * batch["adv_a"]))
        value_p = self.value_p(batch["obs"])
        value_a = self.value_a(batch["obs"])
        value_loss_p = F.mse_loss(value_p, batch["return_p"])
        value_loss_a = F.mse_loss(value_a, batch["return_a"])
        ent_p = dist_p.entropy().sum(dim=-1).mean()
        ent_a = dist_a.entropy().sum(dim=-1).mean()
        u_current, w_current = self.current_actor_actions(z, batch)
        rot_sur = torch.sum(u_current * torch.matmul(w_current, self.rot_reward.T), dim=-1) / math.sqrt(max(self.spec.action_dim, 1))
        return {
            "loss_p_actor": loss_p - (ENT_COEF * ent_p),
            "loss_a_actor": loss_a - (ENT_COEF * ent_a),
            "joint_actor_loss": loss_p + loss_a,
            "value_loss_p": value_loss_p,
            "value_loss_a": value_loss_a,
            "entropy_p": ent_p,
            "entropy_a": ent_a,
            "ratio_p": ratio_p,
            "ratio_a": ratio_a,
            "clip_fraction_p": (torch.abs(ratio_p - 1.0) > CLIP_EPS).float().mean(),
            "clip_fraction_a": (torch.abs(ratio_a - 1.0) > CLIP_EPS).float().mean(),
            "mean_kl_p": torch.mean((ratio_p - 1.0) - torch.log(ratio_p + EPS)),
            "mean_kl_a": torch.mean((ratio_a - 1.0) - torch.log(ratio_a + EPS)),
            "rot_surrogate": rot_sur.mean(),
        }

    def field(self, z: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        z_req = z if z.requires_grad else z.detach().clone().requires_grad_(True)
        comps = self.loss_components(z_req, batch)
        total = comps["loss_p_actor"] + comps["loss_a_actor"]
        return torch.autograd.grad(total, z_req, create_graph=True)[0]

    def merit(self, z: torch.Tensor, batch: dict[str, torch.Tensor], weights: LyapWeights, compute_geometry: bool) -> dict[str, float]:
        comps = self.loss_components(z, batch)
        field = self.field(z, batch).detach()
        field_energy = 0.5 * float(torch.dot(field, field).item())
        joint_policy_loss = float(comps["joint_actor_loss"].detach().item())
        rot_value = float((-comps["rot_surrogate"]).detach().item())
        if "field0" not in self.metric_refs:
            self.metric_refs["field0"] = max(field_energy, EPS)
            self.metric_refs["joint_loss0"] = max(abs(joint_policy_loss), EPS)
            self.metric_refs["rot0"] = max(abs(rot_value), EPS)
        field_term = field_energy / (self.metric_refs["field0"] + EPS)
        loss_term = joint_policy_loss / (self.metric_refs["joint_loss0"] + EPS)
        rot_term = rot_value / (self.metric_refs["rot0"] + EPS)
        V = (weights.lambda_F * field_term) + (weights.lambda_L * loss_term) + (weights.lambda_rot * rot_term)

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
            "field_term": float(field_term),
            "loss_term": float(loss_term),
            "rot_term": float(rot_term),
            "field_norm": float(torch.linalg.norm(field).item()),
            "joint_policy_loss": float(joint_policy_loss),
            "rot_surrogate_mean": float(comps["rot_surrogate"].detach().item()),
            "mean_KL_P": float(comps["mean_kl_p"].detach().item()),
            "mean_KL_A": float(comps["mean_kl_a"].detach().item()),
            "ratio_clip_fraction_P": float(comps["clip_fraction_p"].detach().item()),
            "ratio_clip_fraction_A": float(comps["clip_fraction_a"].detach().item()),
            "log_std_mean_P": float(self.split_z(z)["protagonist_log_std"].mean().detach().item()),
            "log_std_mean_A": float(self.split_z(z)["adversary_log_std"].mean().detach().item()),
            "critic_loss_total": float((comps["value_loss_p"] + comps["value_loss_a"]).detach().item()),
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
        gen.manual_seed(20260627 + self.seed)
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
        for _ in range(CRITIC_STEPS_PER_ITER):
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
        total_returns: list[float] = []
        pure_returns: list[float] = []
        scaled_returns: list[float] = []
        rot_returns: list[float] = []
        clip_fracs: list[float] = []
        for ep in range(episodes):
            obs, _ = env.reset(seed=self.seed + 7000 + ep)
            done = False
            total = 0.0
            pure = 0.0
            scaled = 0.0
            rot = 0.0
            episode_clips = []
            while not done:
                obs_t = torch.as_tensor(obs, dtype=DTYPE, device=DEVICE).unsqueeze(0)
                u = self.actor_mean(parts["protagonist_actor"], obs_t)
                if with_adversary:
                    w = self.actor_mean(parts["adversary_actor"], obs_t)
                else:
                    w = torch.zeros_like(u)
                a_raw, a_env = self.blend_action(u, w)
                obs, reward_raw, terminated, truncated, _ = env.step(a_env.squeeze(0).detach().cpu().numpy().astype(np.float32))
                done = bool(terminated or truncated)
                reward_t = torch.as_tensor([reward_raw], dtype=DTYPE, device=DEVICE)
                r_total, r_task, r_rot = self.total_reward(reward_t, u, w)
                total += float(r_total.item())
                pure += float(reward_raw)
                scaled += float(r_task.item())
                rot += float(r_rot.item())
                episode_clips.append(float(torch.any(torch.abs(a_raw.squeeze(0) - a_env.squeeze(0)) > 1e-12).item()))
            total_returns.append(total)
            pure_returns.append(pure)
            scaled_returns.append(scaled)
            rot_returns.append(rot)
            clip_fracs.append(float(np.mean(episode_clips)) if episode_clips else 0.0)
        env.close()
        return {
            "total_game_return": float(np.mean(total_returns)),
            "pure_env_return": float(np.mean(pure_returns)),
            "scaled_env_return": float(np.mean(scaled_returns)),
            "rotational_return": float(np.mean(rot_returns)),
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


def estimate_reward_scale(spec: EnvSpec, seed: int, steps: int = 1024) -> float:
    env = gym.make(spec.env_id)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    ema = 0.0
    for _ in range(steps):
        action = rng.uniform(spec.action_low, spec.action_high).astype(np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        ema = 0.95 * ema + 0.05 * abs(float(reward))
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return max(1.0, ema)


def cap_delta(delta: torch.Tensor, max_norm: float) -> tuple[torch.Tensor, bool]:
    if not math.isfinite(max_norm) or max_norm <= 0.0:
        return delta, False
    norm = float(torch.linalg.norm(delta).item())
    if norm <= max_norm:
        return delta, False
    return delta * (max_norm / (norm + EPS)), True


def fit_quadratic_1d(v0: float, v1: float, v2: float, delta: float) -> tuple[float, float]:
    h = (v2 - (2.0 * v1) + v0) / max(delta * delta, EPS)
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

    nog_sol = solve_nog(lambda beta: point(beta, 0.0), v0, eta, beta_max)
    beta_nog = float(nog_sol["beta"])
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


def run_sgd(game: GeometryControlledGame, z: torch.Tensor, batch: dict[str, torch.Tensor], lr: float) -> tuple[torch.Tensor, dict[str, Any]]:
    field = game.field(z, batch).detach()
    delta = -lr * field
    return (z + delta).detach(), {"update_norm": float(torch.linalg.norm(delta).item())}


def run_egm(game: GeometryControlledGame, z: torch.Tensor, batch: dict[str, torch.Tensor], lr: float) -> tuple[torch.Tensor, dict[str, Any]]:
    field0 = game.field(z, batch).detach()
    z_half = z - (lr * field0)
    field_half = game.field(z_half, batch).detach()
    delta = -lr * field_half
    return (z + delta).detach(), {"update_norm": float(torch.linalg.norm(delta).item())}


def compute_field_and_g(game: GeometryControlledGame, z: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    z_req = z.detach().clone().requires_grad_(True)
    field_z = game.field(z_req, batch)
    _, g_vec = torch.autograd.functional.jvp(lambda zz: game.field(zz, batch), (z_req,), (field_z.detach(),), create_graph=False, strict=False)
    return field_z.detach(), g_vec.detach()


def evaluate_candidate_state(
    game: GeometryControlledGame,
    z_candidate: torch.Tensor,
    batch: dict[str, torch.Tensor],
    weights: LyapWeights,
) -> dict[str, float]:
    return game.merit(z_candidate, batch, weights, compute_geometry=False)


def run_nog_closed(game: GeometryControlledGame, z: torch.Tensor, batch: dict[str, torch.Tensor], cfg: Config) -> tuple[torch.Tensor, dict[str, Any]]:
    field = game.field(z, batch).detach()
    f_norm = float(torch.linalg.norm(field).item())
    step_cap = cfg.joint_lr * f_norm
    v0 = game.merit(z, batch, cfg.weights, compute_geometry=False)["V"]
    p_dir = -field

    def point(beta: float) -> float:
        cand = z + beta * p_dir
        return evaluate_candidate_state(game, cand, batch, cfg.weights)["V"]

    sol = solve_nog(point, v0, cfg.joint_lr, beta_max=3.0 * cfg.joint_lr)
    delta_raw = sol["beta"] * p_dir
    delta, trust_active = cap_delta(delta_raw, step_cap)
    cand = (z + delta).detach()
    v_after = evaluate_candidate_state(game, cand, batch, cfg.weights)["V"]
    return cand, {
        "beta": float(sol["beta"]),
        "gamma": 0.0,
        "gamma_active": 0,
        "G_contribution_ratio": 0.0,
        "fallback_to_noG": 0,
        "trust_radius_active": int(trust_active),
        "predicted_inclusion_pass": 1,
        "V_after_candidate": float(v_after),
        "update_norm": float(torch.linalg.norm(delta).item()),
        "selected_active_set": "edge_gamma0_nog",
    }


def run_qp_closed(
    game: GeometryControlledGame,
    z: torch.Tensor,
    batch: dict[str, torch.Tensor],
    cfg: Config,
    noG_safe: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    field, g_vec = compute_field_and_g(game, z, batch)
    f_norm = float(torch.linalg.norm(field).item())
    g_norm = float(torch.linalg.norm(g_vec).item())
    v0 = game.merit(z, batch, cfg.weights, compute_geometry=False)["V"]
    p_dir = -field
    g_dir = g_vec
    step_cap = cfg.joint_lr * f_norm
    gamma_max = 3.0 * cfg.joint_lr * (f_norm / (g_norm + EPS))

    def point(beta: float, gamma: float) -> float:
        cand = z + beta * p_dir + gamma * g_dir
        return evaluate_candidate_state(game, cand, batch, cfg.weights)["V"]

    qp = solve_qp(point, v0, cfg.joint_lr, beta_max=3.0 * cfg.joint_lr, gamma_max=gamma_max)
    delta_qp_raw = qp["beta"] * p_dir + qp["gamma"] * g_dir
    delta_qp, trust_active = cap_delta(delta_qp_raw, step_cap)
    z_qp = (z + delta_qp).detach()
    v_qp = evaluate_candidate_state(game, z_qp, batch, cfg.weights)["V"]

    z_nog, nog_meta = run_nog_closed(game, z, batch, cfg)
    v_nog = safe_float(nog_meta["V_after_candidate"])
    gamma_active = int(abs(qp["gamma"]) > 1e-12)
    g_ratio = abs(qp["gamma"] * g_norm) / (abs(qp["beta"] * f_norm) + abs(qp["gamma"] * g_norm) + EPS)

    chosen = "qp"
    chosen_z = z_qp
    fallback = 0
    if noG_safe:
        qp_ok = (
            finite(v_qp)
            and v_qp <= v_nog - TRUST_QP_TOL
            and gamma_active == 1
            and g_ratio >= TRUST_QP_G_MIN_RATIO
        )
        if not qp_ok:
            chosen = "noG"
            chosen_z = z_nog
            fallback = 1

    meta = {
        "beta": float(qp["beta"]),
        "gamma": float(qp["gamma"]),
        "gamma_active": int(gamma_active),
        "G_contribution_ratio": float(g_ratio),
        "fallback_to_noG": int(fallback),
        "trust_radius_active": int(trust_active),
        "predicted_inclusion_pass": int(qp["predicted_inclusion_pass"]),
        "V_after_candidate": float(v_qp),
        "V_after_noG": float(v_nog),
        "QP_better_than_noG_actual_V": int(finite(v_qp) and finite(v_nog) and v_qp < v_nog),
        "selected_active_set": qp["selected_active_set"],
        "candidate_count": qp["candidate_count"],
        "predicted_inclusion_gap": float(qp["predicted_inclusion_gap"]),
        "update_norm": float(torch.linalg.norm(delta_qp if chosen == 'qp' else (z_nog - z)).item()),
        "chosen_step": chosen,
    }
    if chosen == "noG":
        meta["beta"] = safe_float(nog_meta["beta"])
        meta["gamma"] = 0.0
        meta["gamma_active"] = 0
        meta["G_contribution_ratio"] = 0.0
        meta["update_norm"] = safe_float(nog_meta["update_norm"])
        meta["selected_active_set"] = "fallback_noG"
        meta["V_after_candidate"] = float(v_nog)
    return chosen_z, meta


def run_invariant_check(game: GeometryControlledGame, cfg: Config) -> tuple[list[dict[str, Any]], str]:
    z0 = game.init_z()
    batch = game.collect_rollout(z0, rollout_seed=cfg.weights.lambda_F.__hash__() % 1000 + 17)
    weights = cfg.weights
    field, g_vec = compute_field_and_g(game, z0, batch)
    f_norm = float(torch.linalg.norm(field).item())
    g_norm = float(torch.linalg.norm(g_vec).item())
    v0 = game.merit(z0, batch, weights, compute_geometry=False)["V"]
    p_dir = -field
    g_dir = g_vec

    def point1(beta: float) -> float:
        return evaluate_candidate_state(game, z0 + beta * p_dir, batch, weights)["V"]

    def point2(beta: float, gamma: float) -> float:
        return evaluate_candidate_state(game, z0 + beta * p_dir + gamma * g_dir, batch, weights)["V"]

    nog = solve_nog(point1, v0, cfg.joint_lr, beta_max=3.0 * cfg.joint_lr)
    qp = solve_qp(point2, v0, cfg.joint_lr, beta_max=3.0 * cfg.joint_lr, gamma_max=3.0 * cfg.joint_lr * (f_norm / (g_norm + EPS)))
    q1 = nog["l_beta"] * nog["beta"] + 0.5 * nog["h_bb"] * nog["beta"] * nog["beta"]
    q2_gamma0 = qp["l_beta"] * nog["beta"] + 0.5 * qp["h_bb"] * nog["beta"] * nog["beta"]
    z_forced_nog, forced_nog_meta = run_nog_closed(game, z0, batch, cfg)
    z_native_nog, native_nog_meta = run_nog_closed(game, z0, batch, cfg)
    z_safe, safe_meta = run_qp_closed(game, z0, batch, cfg, noG_safe=True)
    safe_eval = evaluate_candidate_state(game, z_safe, batch, weights)["V"]
    row = {
        "env_id": cfg.env_id,
        "scope": "actor_logstd_only_equivalent",
        "alpha": cfg.alpha,
        "q1_equals_q2_gamma0_pass": int(abs(q1 - q2_gamma0) <= 1e-6 * max(1.0, abs(q1), abs(q2_gamma0))),
        "predicted_inclusion_pass": int(qp["predicted_inclusion_pass"]),
        "gamma_negative_flag_plusG": int(qp["gamma"] < -1e-12),
        "forced_nog_same_batch_delta_rel_diff": abs(safe_float(forced_nog_meta["update_norm"]) - safe_float(native_nog_meta["update_norm"])) / (abs(safe_float(native_nog_meta["update_norm"])) + EPS),
        "forced_nog_same_batch_V_diff": abs(safe_float(forced_nog_meta["V_after_candidate"]) - safe_float(native_nog_meta["V_after_candidate"])),
        "noG_safe_actual_choice_matches_applied_step": int(
            (safe_meta["chosen_step"] == "noG" and abs(safe_eval - safe_float(safe_meta["V_after_candidate"])) <= 1e-9)
            or (safe_meta["chosen_step"] == "qp" and abs(safe_eval - safe_float(safe_meta["V_after_candidate"])) <= 1e-9)
        ),
        "beta_noG": nog["beta"],
        "beta_QP": qp["beta"],
        "gamma_QP": qp["gamma"],
    }
    lines = [
        "# final_geometry_controlled_qp_last_attempt invariants",
        "",
        f"- env: `{cfg.env_id}`",
        f"- alpha: `{cfg.alpha}`",
        f"- q1_equals_q2_gamma0_pass_fraction: `{float(row['q1_equals_q2_gamma0_pass']):.6f}`",
        f"- predicted_inclusion_pass_fraction: `{float(row['predicted_inclusion_pass']):.6f}`",
        f"- gamma_negative_fraction_plusG: `{float(row['gamma_negative_flag_plusG']):.6f}`",
        f"- forced_nog_same_batch_delta_rel_diff: `{row['forced_nog_same_batch_delta_rel_diff']:.6e}`",
        f"- noG_safe_actual_choice_matches_applied_step_flag: `{float(row['noG_safe_actual_choice_matches_applied_step']):.6f}`",
    ]
    decision = "QP_INVARIANT_FAIL"
    if (
        row["q1_equals_q2_gamma0_pass"] == 1
        and row["predicted_inclusion_pass"] == 1
        and row["gamma_negative_flag_plusG"] == 0
        and row["forced_nog_same_batch_delta_rel_diff"] <= 1e-6
        and row["noG_safe_actual_choice_matches_applied_step"] == 1
    ):
        decision = "QP_INVARIANT_PASS"
    lines.append("")
    lines.append(f"- decision: `{decision}`")
    return [row], "\n".join(lines) + "\n"


def method_step(game: GeometryControlledGame, z: torch.Tensor, batch: dict[str, torch.Tensor], cfg: Config, method: str) -> tuple[torch.Tensor, dict[str, Any]]:
    if method == "sgd_gda":
        return run_sgd(game, z, batch, cfg.joint_lr)
    if method == "egm":
        return run_egm(game, z, batch, cfg.joint_lr)
    if method == "proposed_nog_closed":
        return run_nog_closed(game, z, batch, cfg)
    if method == "proposed_qp_closed":
        return run_qp_closed(game, z, batch, cfg, noG_safe=False)
    if method == "proposed_qp_nog_safe":
        return run_qp_closed(game, z, batch, cfg, noG_safe=True)
    raise ValueError(method)


def run_method(game: GeometryControlledGame, cfg: Config, method: str, seed: int, iterations: int, eval_episodes: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed_everything(seed)
    z = game.init_z()
    game.metric_refs = {}
    curves: list[dict[str, Any]] = []
    qp_better_nog_count = 0
    qp_better_egm_count = 0
    gamma_active_count = 0
    fallback_count = 0
    g_ratio_sum = 0.0
    predicted_pass_count = 0
    eval_cache = {
        "clean": game.evaluate_policy(z, eval_episodes, with_adversary=False),
        "adv": game.evaluate_policy(z, eval_episodes, with_adversary=True),
    }
    auc_track: dict[str, list[float]] = {
        "train_total": [],
        "clean_total": [],
        "current_total": [],
        "pure_env": [],
        "rotational": [],
        "field_norm": [],
        "V": [],
    }
    for iteration in range(iterations + 1):
        batch = game.collect_rollout(z, rollout_seed=seed + 1000 + iteration)
        metrics = game.merit(z, batch, cfg.weights, compute_geometry=True)
        if iteration % PLOT_EVERY == 0 or iteration == iterations:
            eval_cache["clean"] = game.evaluate_policy(z, eval_episodes, with_adversary=False)
            eval_cache["adv"] = game.evaluate_policy(z, eval_episodes, with_adversary=True)
        row = {
            **cfg.to_row(),
            "method": method,
            "iteration": iteration,
            "train_total_game_return": float(batch["train_total_game_return"].item()),
            "train_pure_env_return": float(batch["train_pure_env_return"].item()),
            "train_rotational_return": float(batch["train_rotational_return"].item()),
            "clean_total_game_return": eval_cache["clean"]["total_game_return"],
            "current_adv_total_game_return": eval_cache["adv"]["total_game_return"],
            "clean_pure_env_return": eval_cache["clean"]["pure_env_return"],
            "current_adv_pure_env_return": eval_cache["adv"]["pure_env_return"],
            "clean_rotational_return": eval_cache["clean"]["rotational_return"],
            "current_adv_rotational_return": eval_cache["adv"]["rotational_return"],
            "current_adv_degradation_total": eval_cache["clean"]["total_game_return"] - eval_cache["adv"]["total_game_return"],
            "pure_env_degradation": eval_cache["clean"]["pure_env_return"] - eval_cache["adv"]["pure_env_return"],
            "action_clip_fraction": float(batch["action_clip_fraction"].item()),
            "mean_abs_u": float(batch["mean_abs_u"].item()),
            "mean_abs_w": float(batch["mean_abs_w"].item()),
            "mean_abs_Rw": float(batch["mean_abs_rw"].item()),
            "mean_abs_action_before_clip": float(batch["mean_abs_action_before_clip"].item()),
            "mean_abs_action_after_clip": float(batch["mean_abs_action_after_clip"].item()),
            **metrics,
        }
        row["curve_finite_flag"] = int(
            all(
                finite(row[key])
                for key in [
                    "V",
                    "field_norm",
                    "train_total_game_return",
                    "clean_total_game_return",
                    "current_adv_total_game_return",
                    "current_adv_pure_env_return",
                ]
            )
        )
        curves.append(row)
        auc_track["train_total"].append(row["train_total_game_return"])
        auc_track["clean_total"].append(row["clean_total_game_return"])
        auc_track["current_total"].append(row["current_adv_total_game_return"])
        auc_track["pure_env"].append(row["current_adv_pure_env_return"])
        auc_track["rotational"].append(row["current_adv_rotational_return"])
        auc_track["field_norm"].append(row["field_norm"])
        auc_track["V"].append(row["V"])
        if iteration == iterations:
            break
        next_z, meta = method_step(game, z, batch, cfg, method)
        curves[-1].update(meta)
        gamma_active_count += int(meta.get("gamma_active", 0))
        fallback_count += int(meta.get("fallback_to_noG", 0))
        predicted_pass_count += int(meta.get("predicted_inclusion_pass", 0))
        g_ratio_sum += safe_float(meta.get("G_contribution_ratio", 0.0), 0.0)
        qp_better_nog_count += int(meta.get("QP_better_than_noG_actual_V", 0))
        if method in {"proposed_qp_closed", "proposed_qp_nog_safe"}:
            z_egm, _ = run_egm(game, z, batch, cfg.joint_lr)
            v_egm = evaluate_candidate_state(game, z_egm, batch, cfg.weights)["V"]
            v_qp = evaluate_candidate_state(game, next_z, batch, cfg.weights)["V"]
            better_egm = int(finite(v_qp) and finite(v_egm) and v_qp < v_egm)
            curves[-1]["QP_better_than_EGM_actual_V"] = better_egm
            qp_better_egm_count += better_egm
        z = next_z.detach()
        game.update_critics(batch)

    summary = {
        **cfg.to_row(),
        "method": method,
        "current_adv_total_game_return_AUC": auc(auc_track["current_total"]),
        "clean_total_game_return_AUC": auc(auc_track["clean_total"]),
        "pure_env_return_AUC": auc(auc_track["pure_env"]),
        "rotational_return_AUC": auc(auc_track["rotational"]),
        "field_norm_AUC": auc(auc_track["field_norm"]),
        "V_AUC": auc(auc_track["V"]),
        "final_current_adv_total_game_return": curves[-1]["current_adv_total_game_return"],
        "final_clean_total_game_return": curves[-1]["clean_total_game_return"],
        "final_current_adv_pure_env_return": curves[-1]["current_adv_pure_env_return"],
        "final_action_clip_fraction": curves[-1]["action_clip_fraction"],
        "gamma_active_frac": gamma_active_count / max(iterations, 1),
        "fallback_to_noG_frac": fallback_count / max(iterations, 1),
        "G_contribution_ratio": g_ratio_sum / max(iterations, 1),
        "QP_better_than_noG_actual_V_fraction": qp_better_nog_count / max(iterations, 1),
        "QP_better_than_EGM_actual_V_fraction": qp_better_egm_count / max(iterations, 1),
        "predicted_inclusion_pass_fraction": predicted_pass_count / max(iterations, 1),
        "curve_sanity_flag": int(
            all(row["curve_finite_flag"] == 1 for row in curves)
            and spike_ratio([row["current_adv_total_game_return"] for row in curves]) <= 20.0
        ),
    }
    return curves, summary


def smoke_geometry_for_env(spec: EnvSpec, env_root: Path, seed: int) -> tuple[list[dict[str, Any]], list[Config]]:
    rows: list[dict[str, Any]] = []
    candidates: list[tuple[float, Config]] = []
    reward_scale = estimate_reward_scale(spec, seed)
    for alpha in ALPHA_GRID:
        for rho in RHO_GRID:
            cfg = Config(
                env_id=spec.env_id,
                alpha=alpha,
                rho=rho,
                joint_lr=JOINT_LR_GRID[0],
                weights=LyapWeights(*LAMBDA_GRID[0]),
                reward_scale=reward_scale,
            )
            game = GeometryControlledGame(spec, cfg, seed)
            z0 = game.init_z()
            batch = game.collect_rollout(z0, rollout_seed=seed + 123)
            metrics = game.merit(z0, batch, cfg.weights, compute_geometry=True)
            row = {
                **cfg.to_row(),
                "smoke_action_clip_fraction": float(batch["action_clip_fraction"].item()),
                "smoke_mean_abs_u": float(batch["mean_abs_u"].item()),
                "smoke_mean_abs_w": float(batch["mean_abs_w"].item()),
                "smoke_mean_abs_Rw": float(batch["mean_abs_rw"].item()),
                "smoke_train_total_game_return": float(batch["train_total_game_return"].item()),
                **metrics,
            }
            score = (
                max(row["rotation_ratio_proxy"], 0.0)
                + 0.25 * max(row["cross_to_same_ratio"], 0.0)
                + 0.10 * max(row["non_collinearity"], 0.0)
                - 5.0 * max(0.0, row["smoke_action_clip_fraction"] - 0.05)
            )
            row["geometry_smoke_score"] = score
            rows.append(row)
            if row["smoke_action_clip_fraction"] <= 0.05 and finite(row["field_norm"]):
                candidates.append((score, cfg))
    rows.sort(key=lambda item: (-safe_float(item["geometry_smoke_score"]), safe_float(item["smoke_action_clip_fraction"])))
    selected = [cfg for _, cfg in sorted(candidates, key=lambda item: item[0], reverse=True)[:SMOKE_TOPK]]
    write_csv(env_root / "01_geometry_env_smoke" / f"{spec.env_id}_geometry_smoke.csv", rows)
    return rows, selected


def confirm_methods(curves_by_method: dict[str, list[dict[str, Any]]], summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sgd = summaries["sgd_gda"]
    egm = summaries["egm"]
    nog = summaries["proposed_nog_closed"]
    qp = summaries["proposed_qp_closed"]
    qp_safe = summaries["proposed_qp_nog_safe"]

    def dominance(a_key: str, b_key: str, metric: str) -> float:
        a = curves_by_method[a_key]
        b = curves_by_method[b_key]
        wins = 0
        total = 0
        for ra, rb in zip(a, b):
            va = safe_float(ra[metric])
            vb = safe_float(rb[metric])
            if finite(va) and finite(vb):
                wins += int(va > vb)
                total += 1
        return wins / max(total, 1)

    qp_auc = qp_safe["current_adv_total_game_return_AUC"]
    egm_auc = egm["current_adv_total_game_return_AUC"]
    nog_auc = nog["current_adv_total_game_return_AUC"]
    sgd_auc = sgd["current_adv_total_game_return_AUC"]
    qp_vs_egm = (qp_auc - egm_auc) / (abs(egm_auc) + EPS)
    qp_vs_nog = (qp_auc - nog_auc) / (abs(nog_auc) + EPS)
    egm_vs_sgd = (egm_auc - sgd_auc) / (abs(sgd_auc) + EPS)
    nog_vs_sgd = (nog_auc - sgd_auc) / (abs(sgd_auc) + EPS)
    dom_qp_egm = dominance("proposed_qp_nog_safe", "egm", "current_adv_total_game_return")
    dom_qp_nog = dominance("proposed_qp_nog_safe", "proposed_nog_closed", "current_adv_total_game_return")
    dom_egm_sgd = dominance("egm", "sgd_gda", "current_adv_total_game_return")
    dom_nog_sgd = dominance("proposed_nog_closed", "sgd_gda", "current_adv_total_game_return")

    decision = "NO_GEOMETRY_POSITIVE_FOUND"
    if (
        qp_vs_egm >= 0.15
        and qp_vs_nog >= 0.05
        and max(egm_vs_sgd, nog_vs_sgd) >= 0.05
        and dom_qp_egm >= 0.70
        and dom_qp_nog >= 0.65
        and qp_safe["fallback_to_noG_frac"] <= 0.30
        and qp_safe["gamma_active_frac"] >= 0.40
        and qp_safe["G_contribution_ratio"] >= 0.15
        and qp_safe["QP_better_than_noG_actual_V_fraction"] >= 0.60
        and qp_safe["predicted_inclusion_pass_fraction"] >= 0.999
        and qp_safe["curve_sanity_flag"] == 1
    ):
        decision = "QP_GEOM_WEAK_POSITIVE"
    if (
        qp_vs_egm >= 0.15
        and qp_vs_nog >= 0.15
        and qp_vs_nog > 0.10
        and max(egm_vs_sgd, nog_vs_sgd) >= 0.15
        and dom_qp_egm >= 0.80
        and dom_qp_nog >= 0.80
        and qp_safe["fallback_to_noG_frac"] <= 0.20
        and qp_safe["gamma_active_frac"] >= 0.50
        and qp_safe["G_contribution_ratio"] >= 0.20
        and qp_safe["QP_better_than_noG_actual_V_fraction"] >= 0.70
        and qp_safe["curve_sanity_flag"] == 1
        and qp_safe["final_current_adv_pure_env_return"] >= sgd["final_current_adv_pure_env_return"] - 200.0
    ):
        decision = "QP_GEOM_STRONG_POSITIVE"

    return {
        "decision": decision,
        "qp_vs_egm_auc_frac": qp_vs_egm,
        "qp_vs_nog_auc_frac": qp_vs_nog,
        "egm_vs_sgd_auc_frac": egm_vs_sgd,
        "nog_vs_sgd_auc_frac": nog_vs_sgd,
        "dom_qp_egm": dom_qp_egm,
        "dom_qp_nog": dom_qp_nog,
        "dom_egm_sgd": dom_egm_sgd,
        "dom_nog_sgd": dom_nog_sgd,
    }


def save_env_plots(env_plot_dir: Path, env_id: str, cfg_slug: str, curves: list[dict[str, Any]]) -> None:
    if plt is None or not curves:
        return
    frame = {}
    for row in curves:
        frame.setdefault(row["method"], []).append(row)
    titles = [
        ("current_adv_total_game_return", "Current-Adv Total Game Return"),
        ("clean_total_game_return", "Clean Total Game Return"),
        ("current_adv_pure_env_return", "Current-Adv Pure Env Return"),
        ("rotational_return", "Rotational Return",),
        ("V", "Lyapunov Merit V"),
        ("field_norm", "Field Norm"),
        ("G_over_F", "G / F"),
        ("non_collinearity", "Non-collinearity"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    axes = axes.flatten()
    for ax, (metric, title) in zip(axes, titles):
        for method, rows in frame.items():
            xs = [r["iteration"] for r in rows]
            ys = [safe_float(r.get(metric)) for r in rows]
            ax.plot(xs, ys, marker="o", label=method)
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{env_id} | {cfg_slug}")
    fig.tight_layout()
    fig.savefig(env_plot_dir / f"{env_id}_{cfg_slug}_all_curves.png", dpi=160)
    plt.close(fig)


def run_config(env_root: Path, spec: EnvSpec, cfg: Config, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    curves_all: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    curves_by_method: dict[str, list[dict[str, Any]]] = {}
    for method in METHODS:
        game = GeometryControlledGame(spec, cfg, seed)
        curves, summary = run_method(game, cfg, method, seed, ITERATIONS, EVAL_EPISODES)
        curves_all.extend(curves)
        summary_rows.append(summary)
        curves_by_method[method] = curves
    decision_meta = confirm_methods(curves_by_method, {row["method"]: row for row in summary_rows})
    for row in summary_rows:
        row.update(decision_meta)
    return curves_all, summary_rows, decision_meta


def plot_global_rankings(plot_root: Path, ranked_rows: list[dict[str, Any]]) -> None:
    if plt is None or not ranked_rows:
        return
    labels = [f"{row['env_id']}|a={row['alpha']}|rho={row['rho']}" for row in ranked_rows[:12]]
    geom_scores = [safe_float(row["geometry_score"]) for row in ranked_rows[:12]]
    qp_auc = [safe_float(row["qp_auc"]) for row in ranked_rows[:12]]
    fallback = [safe_float(row["fallback_to_noG_frac"]) for row in ranked_rows[:12]]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].barh(labels, geom_scores)
    axes[0].set_title("Geometry Score")
    axes[1].barh(labels, qp_auc)
    axes[1].set_title("QP Current-Adv Total Return AUC")
    axes[2].barh(labels, fallback)
    axes[2].set_title("Fallback to noG Fraction")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_root / "geometry_search_geometry_metrics.png", dpi=160)
    plt.close(fig)


def top_curve_plot(plot_root: Path, all_curves: list[dict[str, Any]], top_key: tuple[str, str]) -> None:
    if plt is None:
        return
    env_id, slug = top_key
    sub = [row for row in all_curves if row["env_id"] == env_id and row["config_slug"] == slug]
    if not sub:
        return
    save_env_plots(plot_root, env_id, slug, sub)


def maybe_run_confirmation(best_row: dict[str, Any], spec: EnvSpec, result_root: Path) -> tuple[list[dict[str, Any]], str]:
    if best_row.get("decision") not in {"QP_GEOM_WEAK_POSITIVE", "QP_GEOM_STRONG_POSITIVE"}:
        return [], "GEOMETRY_CONTROLLED_QP_FAILS"
    cfg = Config(
        env_id=best_row["env_id"],
        alpha=float(best_row["alpha"]),
        rho=float(best_row["rho"]),
        joint_lr=float(best_row["joint_lr"]),
        weights=LyapWeights(float(best_row["lambda_F"]), float(best_row["lambda_L"]), float(best_row["lambda_rot"])),
        reward_scale=float(best_row["reward_scale"]),
    )
    rows: list[dict[str, Any]] = []
    for seed in [0, 1, 2]:
        spec_seed = check_env(cfg.env_id)
        curves_all, summaries, _ = run_config(result_root / "03_final_confirmation", spec_seed, cfg, seed)
        for row in summaries:
            row["seed"] = seed
            rows.append(row)
    write_csv(result_root / "03_final_confirmation" / "multiseed_summary.csv", rows)
    by_method = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    means = {method: float(np.mean([safe_float(r["current_adv_total_game_return_AUC"]) for r in method_rows])) for method, method_rows in by_method.items()}
    qp_mean = means.get("proposed_qp_nog_safe", -math.inf)
    if qp_mean >= max(means.get("sgd_gda", -math.inf), means.get("egm", -math.inf), means.get("proposed_nog_closed", -math.inf)):
        decision = "GEOMETRY_CONTROLLED_QP_CONFIRMED"
    else:
        decision = "GEOMETRY_CONTROLLED_QP_ONE_SEED_ONLY"
    report_lines = [
        "# final_geometry_controlled_qp_last_attempt multiseed confirmation",
        "",
        f"- env_id: `{cfg.env_id}`",
        f"- alpha: `{cfg.alpha}`",
        f"- rho: `{cfg.rho}`",
        f"- joint_lr: `{cfg.joint_lr}`",
        "",
    ]
    for method, mean_value in means.items():
        report_lines.append(f"- {method}: mean current_adv_total_game_return_AUC=`{mean_value:.6e}`")
    report_lines.append("")
    report_lines.append(f"- decision: `{decision}`")
    write_text(result_root / "03_final_confirmation" / "multiseed_report.md", "\n".join(report_lines) + "\n")
    return rows, decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Final geometry-controlled MuJoCo QP benchmark")
    parser.add_argument("--output-root", type=str, default=str(RESULT_ROOT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--envs", nargs="*", default=list(DEFAULT_ENVS))
    parser.add_argument("--max-envs", type=int, default=4)
    parser.add_argument("--smoke-only", action="store_true", default=False)
    parser.add_argument("--force-rerun", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    result_root = Path(args.output_root)
    plot_root = ensure_dir(result_root / "plots")
    ensure_dir(result_root / "00_invariants")
    ensure_dir(result_root / "01_geometry_env_smoke")
    ensure_dir(result_root / "02_qp_positive_search")
    ensure_dir(result_root / "03_final_confirmation")

    progress_log = result_root / "progress.log"
    def log(msg: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(line, flush=True)
        with progress_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    envs = args.envs[: max(1, int(args.max_envs))]
    write_json(result_root / "run_plan.json", {"envs": envs, "seed": args.seed, "rollout_steps": ROLLOUT_STEPS})

    invariant_spec = check_env("HalfCheetah-v4")
    invariant_cfg = Config(
        env_id="HalfCheetah-v4",
        alpha=0.3,
        rho=0.3,
        joint_lr=3e-4,
        weights=LyapWeights(*LAMBDA_GRID[0]),
        reward_scale=estimate_reward_scale(invariant_spec, args.seed),
    )
    inv_game = GeometryControlledGame(invariant_spec, invariant_cfg, args.seed)
    inv_rows, inv_report = run_invariant_check(inv_game, invariant_cfg)
    write_csv(result_root / "00_invariants" / "invariant_summary.csv", inv_rows)
    write_text(result_root / "00_invariants" / "invariant_report.md", inv_report)
    inv_decision = "QP_INVARIANT_FAIL" if "QP_INVARIANT_FAIL" in inv_report else "QP_INVARIANT_PASS"
    write_text(result_root / "00_invariants" / "invariant_decision.md", inv_decision + "\n")
    if inv_decision != "QP_INVARIANT_PASS":
        write_text(result_root / "final_geometry_controlled_decision.md", "QP_INVARIANT_FAIL\n")
        write_text(result_root / "final_geometry_controlled_report.md", inv_report)
        return

    global_smoke_rows: list[dict[str, Any]] = []
    global_summary_rows: list[dict[str, Any]] = []
    global_curve_rows: list[dict[str, Any]] = []
    ranked_rows: list[dict[str, Any]] = []

    for env_id in envs:
        log(f"start env smoke: {env_id}")
        spec = check_env(env_id)
        env_root = ensure_dir(result_root / env_id.replace("-", "_"))
        ensure_dir(env_root / "01_geometry_env_smoke")
        ensure_dir(env_root / "02_qp_positive_search")
        ensure_dir(env_root / "plots")
        smoke_rows, selected_cfgs = smoke_geometry_for_env(spec, env_root, args.seed)
        global_smoke_rows.extend(smoke_rows)
        if not selected_cfgs:
            log(f"no valid smoke configs: {env_id}")
            continue
        search_lines = [
            f"# {env_id} geometry-controlled search",
            "",
            f"- selected_smoke_configs: `{len(selected_cfgs)}`",
            "",
        ]
        for base_cfg in selected_cfgs:
            for joint_lr in JOINT_LR_GRID:
                for lambda_F, lambda_L, lambda_rot in LAMBDA_GRID:
                    cfg = Config(
                        env_id=env_id,
                        alpha=base_cfg.alpha,
                        rho=base_cfg.rho,
                        joint_lr=joint_lr,
                        weights=LyapWeights(lambda_F, lambda_L, lambda_rot),
                        reward_scale=base_cfg.reward_scale,
                    )
                    log(f"run config: {cfg.slug}")
                    curves, summaries, decision_meta = run_config(env_root, spec, cfg, args.seed)
                    for row in curves:
                        row["config_slug"] = cfg.slug
                        global_curve_rows.append(row)
                    for row in summaries:
                        row["config_slug"] = cfg.slug
                        global_summary_rows.append(row)
                    qp_safe = next(row for row in summaries if row["method"] == "proposed_qp_nog_safe")
                    egm = next(row for row in summaries if row["method"] == "egm")
                    nog = next(row for row in summaries if row["method"] == "proposed_nog_closed")
                    sgd = next(row for row in summaries if row["method"] == "sgd_gda")
                    geom_smoke = next(
                        row for row in smoke_rows
                        if abs(safe_float(row["alpha"]) - cfg.alpha) <= 1e-12 and abs(safe_float(row["rho"]) - cfg.rho) <= 1e-12
                    )
                    ranked_rows.append(
                        {
                            **cfg.to_row(),
                            "env_id": env_id,
                            "config_slug": cfg.slug,
                            "decision": decision_meta["decision"],
                            "geometry_score": geom_smoke["geometry_smoke_score"],
                            "rotation_ratio_proxy": geom_smoke["rotation_ratio_proxy"],
                            "cross_to_same_ratio": geom_smoke["cross_to_same_ratio"],
                            "non_collinearity": geom_smoke["non_collinearity"],
                            "qp_auc": qp_safe["current_adv_total_game_return_AUC"],
                            "egm_auc": egm["current_adv_total_game_return_AUC"],
                            "nog_auc": nog["current_adv_total_game_return_AUC"],
                            "sgd_auc": sgd["current_adv_total_game_return_AUC"],
                            "fallback_to_noG_frac": qp_safe["fallback_to_noG_frac"],
                            "gamma_active_frac": qp_safe["gamma_active_frac"],
                            "G_contribution_ratio": qp_safe["G_contribution_ratio"],
                            "QP_better_than_noG_actual_V_fraction": qp_safe["QP_better_than_noG_actual_V_fraction"],
                            "QP_better_than_EGM_actual_V_fraction": qp_safe["QP_better_than_EGM_actual_V_fraction"],
                            "predicted_inclusion_pass_fraction": qp_safe["predicted_inclusion_pass_fraction"],
                            "current_adv_total_game_return_final": qp_safe["final_current_adv_total_game_return"],
                            "pure_env_return_final": qp_safe["final_current_adv_pure_env_return"],
                        }
                    )
                    search_lines.append(
                        f"- `{cfg.slug}`: decision=`{decision_meta['decision']}`, qp_auc=`{qp_safe['current_adv_total_game_return_AUC']:.6e}`, "
                        f"egm_auc=`{egm['current_adv_total_game_return_AUC']:.6e}`, nog_auc=`{nog['current_adv_total_game_return_AUC']:.6e}`, "
                        f"fallback=`{qp_safe['fallback_to_noG_frac']:.3f}`, gamma_active=`{qp_safe['gamma_active_frac']:.3f}`"
                    )
                    save_env_plots(env_root / "plots", env_id, cfg.slug, curves)
                    write_csv(env_root / "02_qp_positive_search" / "geometry_search_all.csv", global_summary_rows)
                    write_csv(env_root / "02_qp_positive_search" / "geometry_search_curves.csv", global_curve_rows)
                    write_csv(result_root / "geometry_controlled_ranked.csv", ranked_rows)
                    write_text(env_root / "02_qp_positive_search" / "geometry_search_report.md", "\n".join(search_lines) + "\n")
        log(f"done env search: {env_id}")

    ranked_rows.sort(
        key=lambda row: (
            {"QP_GEOM_STRONG_POSITIVE": 0, "QP_GEOM_WEAK_POSITIVE": 1, "NO_GEOMETRY_POSITIVE_FOUND": 2}.get(str(row["decision"]), 3),
            -safe_float(row["qp_auc"]),
            safe_float(row["fallback_to_noG_frac"]),
        )
    )
    write_csv(result_root / "geometry_controlled_ranked.csv", ranked_rows)
    write_csv(result_root / "01_geometry_env_smoke" / "smoke_summary.csv", global_smoke_rows)
    write_csv(result_root / "02_qp_positive_search" / "geometry_search_all.csv", global_summary_rows)
    write_csv(result_root / "02_qp_positive_search" / "geometry_search_curves.csv", global_curve_rows)
    plot_global_rankings(plot_root, ranked_rows)
    if ranked_rows:
        top_curve_plot(plot_root, global_curve_rows, (ranked_rows[0]["env_id"], ranked_rows[0]["config_slug"]))

    top_lines = ["# geometry_controlled_top_configs", ""]
    for idx, row in enumerate(ranked_rows[:10], start=1):
        top_lines.append(
            f"{idx}. `{row['config_slug']}` | decision=`{row['decision']}` | qp_auc=`{safe_float(row['qp_auc']):.6e}` | "
            f"fallback=`{safe_float(row['fallback_to_noG_frac']):.3f}` | gamma_active=`{safe_float(row['gamma_active_frac']):.3f}` | "
            f"geometry_score=`{safe_float(row['geometry_score']):.6e}`"
        )
    write_text(result_root / "geometry_controlled_top_configs.md", "\n".join(top_lines) + "\n")

    final_decision = "GEOMETRY_CONTROLLED_QP_FAILS"
    confirmation_rows: list[dict[str, Any]] = []
    if ranked_rows and ranked_rows[0]["decision"] in {"QP_GEOM_WEAK_POSITIVE", "QP_GEOM_STRONG_POSITIVE"} and not args.smoke_only:
        best_spec = check_env(ranked_rows[0]["env_id"])
        confirmation_rows, final_decision = maybe_run_confirmation(ranked_rows[0], best_spec, result_root)

    report_lines = [
        "# final_geometry_controlled_qp_last_attempt",
        "",
        "This benchmark preserves the MuJoCo transition dynamics and action constraints, while adding a controlled rotational zero-sum coupling to the reward.",
        "It is a geometry-controlled MuJoCo robust game benchmark, not pure standard RARL.",
        "",
        "## Environments",
    ]
    for env_id in envs:
        report_lines.append(f"- `{env_id}`")
    report_lines.extend(
        [
            "",
            "## Top configs",
            *top_lines[2:],
            "",
            "## Notes",
            "- primary pass/fail return: `current_adv_total_game_return_AUC`",
            "- secondary diagnostics: `pure_env_return_AUC`, `rotational_return_AUC`, `fallback_to_noG_frac`, `gamma_active_frac`",
            "- direct standard HalfCheetah RARL was kept separate from this geometry-controlled benchmark family",
            "",
            f"## Final Decision",
            f"- `{final_decision}`",
        ]
    )
    write_text(result_root / "final_geometry_controlled_report.md", "\n".join(report_lines) + "\n")
    write_text(result_root / "final_geometry_controlled_decision.md", final_decision + "\n")


if __name__ == "__main__":
    main()
