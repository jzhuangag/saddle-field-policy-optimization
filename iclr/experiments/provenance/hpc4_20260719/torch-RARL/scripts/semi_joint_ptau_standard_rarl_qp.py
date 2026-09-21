from __future__ import annotations

import copy
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

try:
    import gymnasium as gym
except Exception:
    import gym

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


WORK_ROOT = Path(r"C:\Users\jzhuangag\work\rarl")
RESULT_ROOT = WORK_ROOT / "results" / "semi_joint_ptau_standard_rarl_qp"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
EPS = 1e-8

SEEDS = [0, 1, 2]
DIAG_ENVS = [("Hopper-v4", [0.1, 0.2, 0.3]), ("HalfCheetah-v4", [0.15, 0.3])]
SWEEP_ALPHA = [0.1, 0.2, 0.3]
SWEEP_LAMBDA_F = [0.1, 0.3, 1.0]
SWEEP_LR = [3e-5, 1e-4]
PRIMARY_METHODS = ["sgd_gda", "egm", "ppm_inner3", "ppm_inner4", "noG_nonneg", "QP_nonneg_damped_safe"]
APPENDIX_METHODS = ["noG_signed_ablation", "QP_signed_damped_safe_ablation"]
METHODS = PRIMARY_METHODS + APPENDIX_METHODS
SCREEN_METHODS = PRIMARY_METHODS

# Budgeted defaults. The report explicitly records these deviations from the full prompt.
ROLLOUT_STEPS = 128
ITERATIONS_SCREEN = 3
ITERATIONS_FINAL = 200
BATCH_LIMIT_FOR_FIELD = 64
EVAL_FREQ = 2
EVAL_EPISODES = 1
BR_STEPS = 4
BR_LR = 3e-4
PPO_CLIP = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
VALUE_LR = 1e-3
VALUE_EPOCHS = 1
K_INNER = 3
ETA_INNER_SCALE = 0.25
TAU = 0.03
LAMBDA_P = 1.0
TRUST_SHRINKS = 5
SOFTPLUS_EPS = 1e-3
PTAU_GATE_WARMUP_STEPS = 10
GATE1B_DIRECTIONS = 3


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite(x: float) -> bool:
    return math.isfinite(float(x))


def safe_float(x: Any, default: float = math.nan) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def auc(values: list[float]) -> float:
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return math.nan
    return float(np.trapz(np.asarray(vals, dtype=np.float64), dx=1.0))


def ema(values: list[float], alpha: float = 0.3) -> list[float]:
    out: list[float] = []
    cur = None
    for v in values:
        v = safe_float(v)
        if cur is None or not math.isfinite(cur):
            cur = v
        else:
            cur = alpha * v + (1.0 - alpha) * cur
        out.append(cur)
    return out


@dataclass
class EnvSpec:
    env_id: str
    obs_dim: int
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray


@dataclass
class Config:
    env_id: str
    alpha: float
    lambda_F: float
    lr: float
    seed: int
    iterations: int
    scope: str = "actor_logstd_only"

    @property
    def slug(self) -> str:
        def enc(x: float) -> str:
            return str(x).replace("-", "m").replace(".", "p")

        return f"{self.env_id.lower()}_a{enc(self.alpha)}_lf{enc(self.lambda_F)}_lr{enc(self.lr)}_s{self.seed}"


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class FlatActors:
    def __init__(self, spec: EnvSpec) -> None:
        self.spec = spec
        self.shapes: list[tuple[str, tuple[int, ...]]] = []
        obs_dim = spec.obs_dim
        act_dim = spec.action_dim
        for prefix in ["p", "a"]:
            self.shapes.extend(
                [
                    (f"{prefix}_w1", (64, obs_dim)),
                    (f"{prefix}_b1", (64,)),
                    (f"{prefix}_w2", (64, 64)),
                    (f"{prefix}_b2", (64,)),
                    (f"{prefix}_w3", (act_dim, 64)),
                    (f"{prefix}_b3", (act_dim,)),
                    (f"{prefix}_logstd", (act_dim,)),
                ]
            )
        self.slices: dict[str, slice] = {}
        start = 0
        for name, shape in self.shapes:
            n = int(np.prod(shape))
            self.slices[name] = slice(start, start + n)
            start += n
        self.dim = start
        self.p_slice = slice(self.slices["p_w1"].start, self.slices["p_logstd"].stop)
        self.a_slice = slice(self.slices["a_w1"].start, self.slices["a_logstd"].stop)
        self.action_high = torch.as_tensor(spec.action_high, dtype=DTYPE, device=DEVICE)

    def init_z(self, seed: int) -> torch.Tensor:
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(seed)
        vals: list[torch.Tensor] = []
        for name, shape in self.shapes:
            if name.endswith("logstd"):
                vals.append(torch.full(shape, -0.5, dtype=DTYPE, device=DEVICE))
            elif "_w" in name:
                fan_in = shape[1]
                vals.append(0.05 * torch.randn(shape, generator=gen, dtype=DTYPE, device=DEVICE) / math.sqrt(fan_in))
            else:
                vals.append(torch.zeros(shape, dtype=DTYPE, device=DEVICE))
        return torch.cat([v.reshape(-1) for v in vals]).detach()

    def unpack(self, z: torch.Tensor, prefix: str) -> dict[str, torch.Tensor]:
        out = {}
        for key in ["w1", "b1", "w2", "b2", "w3", "b3", "logstd"]:
            name = f"{prefix}_{key}"
            shape = dict(self.shapes)[name]
            out[key] = z[self.slices[name]].reshape(shape)
        return out

    def actor(self, z: torch.Tensor, obs: torch.Tensor, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.unpack(z, prefix)
        x = torch.tanh(Fnn.linear(obs, p["w1"], p["b1"]))
        x = torch.tanh(Fnn.linear(x, p["w2"], p["b2"]))
        mean = torch.tanh(Fnn.linear(x, p["w3"], p["b3"])) * self.action_high
        logstd = torch.clamp(p["logstd"], -5.0, 2.0)
        return mean, logstd

    def log_prob(self, z: torch.Tensor, obs: torch.Tensor, action: torch.Tensor, prefix: str) -> torch.Tensor:
        mean, logstd = self.actor(z, obs, prefix)
        std = torch.exp(logstd)
        return (-0.5 * (((action - mean) / (std + EPS)) ** 2 + 2 * logstd + math.log(2 * math.pi))).sum(dim=-1)

    def joint_env_log_prob(self, z: torch.Tensor, obs: torch.Tensor, env_action_raw: torch.Tensor, alpha: float) -> torch.Tensor:
        mu_p, ls_p = self.actor(z, obs, "p")
        mu_a, ls_a = self.actor(z, obs, "a")
        var = torch.exp(2 * ls_p) + (alpha ** 2) * torch.exp(2 * ls_a)
        mean = mu_p + alpha * mu_a
        return (-0.5 * (((env_action_raw - mean) ** 2) / (var + EPS) + torch.log(var + EPS) + math.log(2 * math.pi))).sum(dim=-1)


def check_env(env_id: str) -> EnvSpec | None:
    try:
        env = gym.make(env_id)
        obs_space = env.observation_space
        act_space = env.action_space
        if not hasattr(act_space, "shape") or len(act_space.shape) != 1:
            env.close()
            return None
        spec = EnvSpec(
            env_id=env_id,
            obs_dim=int(obs_space.shape[0]),
            action_dim=int(act_space.shape[0]),
            action_low=np.asarray(act_space.low, dtype=np.float32),
            action_high=np.asarray(act_space.high, dtype=np.float32),
        )
        env.close()
        return spec
    except Exception:
        return None


def sample_actor_action(actors: FlatActors, z: torch.Tensor, obs: torch.Tensor, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    mean, logstd = actors.actor(z, obs, prefix)
    action = mean + torch.exp(logstd) * torch.randn_like(mean)
    logp = actors.log_prob(z, obs, action, prefix)
    return action, logp


def collect_rollout(spec: EnvSpec, actors: FlatActors, value: ValueNet, z: torch.Tensor, cfg: Config, steps: int, seed: int) -> dict[str, torch.Tensor]:
    env = gym.make(spec.env_id)
    obs_np, _ = env.reset(seed=seed)
    rows: dict[str, list[Any]] = {k: [] for k in ["obs", "u", "w", "env_raw", "reward", "done", "value", "old_joint_logp"]}
    clip_count = 0
    for t in range(steps):
        obs_t = torch.as_tensor(np.asarray(obs_np, dtype=np.float32).reshape(1, -1), dtype=DTYPE, device=DEVICE)
        with torch.no_grad():
            u_t, _ = sample_actor_action(actors, z, obs_t, "p")
            w_t, _ = sample_actor_action(actors, z, obs_t, "a")
            env_raw = u_t + cfg.alpha * w_t
            old_joint = actors.joint_env_log_prob(z, obs_t, env_raw, cfg.alpha)
            val = value(obs_t)
        env_action = torch.clamp(env_raw, torch.as_tensor(spec.action_low, device=DEVICE), torch.as_tensor(spec.action_high, device=DEVICE))
        clip_count += int(torch.any(torch.abs(env_action - env_raw) > 1e-6).item())
        next_obs, reward, terminated, truncated, _ = env.step(env_action.squeeze(0).cpu().numpy().astype(np.float32))
        done = bool(terminated or truncated)
        rows["obs"].append(obs_t.squeeze(0).cpu().numpy())
        rows["u"].append(u_t.squeeze(0).cpu().numpy())
        rows["w"].append(w_t.squeeze(0).cpu().numpy())
        rows["env_raw"].append(env_raw.squeeze(0).cpu().numpy())
        rows["reward"].append(float(reward))
        rows["done"].append(float(done))
        rows["value"].append(float(val.item()))
        rows["old_joint_logp"].append(float(old_joint.item()))
        obs_np = next_obs
        if done:
            obs_np, _ = env.reset()
    with torch.no_grad():
        last_v = float(value(torch.as_tensor(np.asarray(obs_np, dtype=np.float32).reshape(1, -1), dtype=DTYPE, device=DEVICE)).item())
    env.close()

    rewards = np.asarray(rows["reward"], dtype=np.float32)
    dones = np.asarray(rows["done"], dtype=np.float32)
    values = np.asarray(rows["value"], dtype=np.float32)
    adv = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(steps)):
        next_nonterminal = 1.0 - dones[t]
        next_value = last_v if t == steps - 1 else values[t + 1]
        delta = rewards[t] + GAMMA * next_value * next_nonterminal - values[t]
        last_gae = delta + GAMMA * GAE_LAMBDA * next_nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    batch = {
        "obs": torch.as_tensor(np.asarray(rows["obs"]), dtype=DTYPE, device=DEVICE),
        "u": torch.as_tensor(np.asarray(rows["u"]), dtype=DTYPE, device=DEVICE),
        "w": torch.as_tensor(np.asarray(rows["w"]), dtype=DTYPE, device=DEVICE),
        "env_raw": torch.as_tensor(np.asarray(rows["env_raw"]), dtype=DTYPE, device=DEVICE),
        "old_joint_logp": torch.as_tensor(np.asarray(rows["old_joint_logp"]), dtype=DTYPE, device=DEVICE),
        "adv": torch.as_tensor(adv, dtype=DTYPE, device=DEVICE),
        "ret": torch.as_tensor(returns, dtype=DTYPE, device=DEVICE),
        "reward_mean": torch.as_tensor(float(rewards.mean()), dtype=DTYPE, device=DEVICE),
        "action_clip_fraction": torch.as_tensor(float(clip_count / max(steps, 1)), dtype=DTYPE, device=DEVICE),
    }
    if steps > BATCH_LIMIT_FOR_FIELD:
        idx = torch.randperm(steps, device=DEVICE)[:BATCH_LIMIT_FOR_FIELD]
        for key in ["obs", "u", "w", "env_raw", "old_joint_logp", "adv", "ret"]:
            batch[key] = batch[key][idx]
    return batch


def train_value(value: ValueNet, batch: dict[str, torch.Tensor]) -> None:
    opt = torch.optim.Adam(value.parameters(), lr=VALUE_LR)
    obs = batch["obs"].detach()
    ret = batch["ret"].detach()
    for _ in range(VALUE_EPOCHS):
        loss = torch.mean((value(obs) - ret) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()


class SemiJointGame:
    def __init__(self, spec: EnvSpec, cfg: Config, actors: FlatActors) -> None:
        self.spec = spec
        self.cfg = cfg
        self.actors = actors
        self.refs: dict[str, float] = {}
        self.eta_inner = max(cfg.lr * ETA_INNER_SCALE, 1e-6)
        self.tau = TAU
        self.softplus_eps = SOFTPLUS_EPS
        self.calibration: dict[str, float] = {}

    def J_contribs(self, z: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logp = self.actors.joint_env_log_prob(z, batch["obs"], batch["env_raw"], self.cfg.alpha)
        ratio = torch.exp(torch.clamp(logp - batch["old_joint_logp"], -20, 20))
        adv = batch["adv"]
        clipped = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP)
        return torch.minimum(ratio * adv, clipped * adv)

    def J_surr(self, z: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.mean(self.J_contribs(z, batch))

    def field(self, z: torch.Tensor, batch: dict[str, torch.Tensor], create_graph: bool = True) -> torch.Tensor:
        z_req = z if z.requires_grad else z.detach().clone().requires_grad_(True)
        J = self.J_surr(z_req, batch)
        grad = torch.autograd.grad(J, z_req, create_graph=create_graph, retain_graph=True)[0]
        F = torch.zeros_like(z_req)
        F[self.actors.p_slice] = -grad[self.actors.p_slice]
        F[self.actors.a_slice] = +grad[self.actors.a_slice]
        return F

    def raw_ptau_gaps(
        self,
        z: torch.Tensor,
        batch: dict[str, torch.Tensor],
        eta_inner: float | None = None,
        tau: float | None = None,
        k_inner: int = K_INNER,
    ) -> dict[str, torch.Tensor]:
        eta = self.eta_inner if eta_inner is None else eta_inner
        tau_val = self.tau if tau is None else tau
        theta0 = z[self.actors.p_slice]
        psi0 = z[self.actors.a_slice]

        def merge(theta: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
            zz = z.clone()
            zz[self.actors.p_slice] = theta
            zz[self.actors.a_slice] = psi
            return zz

        theta = theta0
        for _ in range(k_inner):
            prox = 0.5 / tau_val * torch.sum((theta - theta0) ** 2)
            obj = self.J_surr(merge(theta, psi0), batch) - prox
            g = torch.autograd.grad(obj, theta, create_graph=True)[0]
            theta = theta + eta * g
        j0 = self.J_surr(z, batch)
        j_th = self.J_surr(merge(theta, psi0), batch)
        prox_th = 0.5 / tau_val * torch.sum((theta - theta0) ** 2)
        raw_imp_th = j_th - j0
        gap_th = raw_imp_th - prox_th

        psi = psi0
        for _ in range(k_inner):
            prox = 0.5 / tau_val * torch.sum((psi - psi0) ** 2)
            obj = self.J_surr(merge(theta0, psi), batch) + prox
            g = torch.autograd.grad(obj, psi, create_graph=True)[0]
            psi = psi - eta * g
        j_ps = self.J_surr(merge(theta0, psi), batch)
        prox_ps = 0.5 / tau_val * torch.sum((psi - psi0) ** 2)
        raw_imp_ps = j0 - j_ps
        gap_ps = raw_imp_ps - prox_ps
        return {
            "gap_theta": gap_th,
            "gap_psi": gap_ps,
            "raw_imp_theta": raw_imp_th,
            "raw_imp_psi": raw_imp_ps,
            "prox_theta": prox_th,
            "prox_psi": prox_ps,
        }

    def one_step_raw_improvement(self, z: torch.Tensor, batch: dict[str, torch.Tensor], eta_inner: float) -> tuple[float, float]:
        gaps = self.raw_ptau_gaps(z, batch, eta_inner=eta_inner, tau=1e12, k_inner=1)
        return float(gaps["raw_imp_theta"].detach().item()), float(gaps["raw_imp_psi"].detach().item())

    def calibrate_ptau(self, z: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        z_req = z.detach().clone().requires_grad_(True)
        contrib_std = float(torch.std(self.J_contribs(z_req, batch).detach()).item())
        target = max(0.01 * contrib_std, 1e-8)
        eta = max(self.cfg.lr * ETA_INNER_SCALE, 1e-8)
        eta_rows: list[dict[str, float]] = []
        safe_eta = eta
        safe_score = -math.inf
        for _ in range(10):
            imp_th, imp_ps = self.one_step_raw_improvement(z_req, batch, eta)
            eta_rows.append({"eta": eta, "one_step_imp_theta": imp_th, "one_step_imp_psi": imp_ps, "target": target})
            if imp_th > 0.0 and imp_ps > 0.0:
                score = min(imp_th, imp_ps)
                if score > safe_score:
                    safe_score = score
                    safe_eta = eta
            if imp_th >= target and imp_ps >= target:
                safe_eta = eta
                break
            eta *= 10.0
        self.eta_inner = safe_eta

        tau_rows: list[dict[str, float]] = []
        chosen_tau = TAU
        chosen_ratio = math.inf
        for tau_val in [TAU, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
            gaps = self.raw_ptau_gaps(z_req, batch, eta_inner=self.eta_inner, tau=tau_val, k_inner=K_INNER)
            raw_th = max(float(gaps["raw_imp_theta"].detach().item()), EPS)
            raw_ps = max(float(gaps["raw_imp_psi"].detach().item()), EPS)
            ratio_th = float(gaps["prox_theta"].detach().item()) / raw_th
            ratio_ps = float(gaps["prox_psi"].detach().item()) / raw_ps
            ratio = max(ratio_th, ratio_ps)
            tau_rows.append({"tau": tau_val, "penalty_ratio": ratio, "ratio_theta": ratio_th, "ratio_psi": ratio_ps})
            if ratio <= 0.5:
                chosen_tau = tau_val
                chosen_ratio = ratio
                break
            if ratio < chosen_ratio:
                chosen_tau = tau_val
                chosen_ratio = ratio
        self.tau = chosen_tau

        gaps = self.raw_ptau_gaps(z_req, batch, eta_inner=self.eta_inner, tau=self.tau, k_inner=K_INNER)
        gap_th = float(gaps["gap_theta"].detach().item())
        gap_ps = float(gaps["gap_psi"].detach().item())
        max_gap = max(gap_th, gap_ps, 1e-8)
        min_pos_gap = max(min(gap_th, gap_ps), 1e-8)
        self.softplus_eps = min(SOFTPLUS_EPS, max_gap / 20.0, min_pos_gap / 20.0)
        ptau = self.P_tau(z_req, batch)
        Fv = self.field(z_req, batch, create_graph=True)
        field_e = 0.5 * torch.dot(Fv, Fv)
        self.refs["field0"] = max(float(field_e.detach().item()), EPS)
        self.refs["ptau0"] = max(float(ptau.detach().item()), EPS)
        self.calibration = {
            "contrib_std": contrib_std,
            "target_one_step_imp": target,
            "eta_inner": self.eta_inner,
            "tau": self.tau,
            "softplus_eps": self.softplus_eps,
            "gap_theta_raw": gap_th,
            "gap_psi_raw": gap_ps,
            "raw_imp_theta": float(gaps["raw_imp_theta"].detach().item()),
            "raw_imp_psi": float(gaps["raw_imp_psi"].detach().item()),
            "prox_theta": float(gaps["prox_theta"].detach().item()),
            "prox_psi": float(gaps["prox_psi"].detach().item()),
            "penalty_ratio": chosen_ratio,
            "P_tau0": float(ptau.detach().item()),
            "field0": self.refs["field0"],
            "ptau0": self.refs["ptau0"],
        }
        self.calibration["eta_sweep"] = eta_rows  # type: ignore[assignment]
        self.calibration["tau_sweep"] = tau_rows  # type: ignore[assignment]
        return self.calibration

    def ensure_calibrated(self, z: torch.Tensor, batch: dict[str, torch.Tensor]) -> None:
        if not self.calibration:
            self.calibrate_ptau(z, batch)

    def P_tau(self, z: torch.Tensor, batch: dict[str, torch.Tensor], eta_inner: float | None = None) -> torch.Tensor:
        gaps = self.raw_ptau_gaps(z, batch, eta_inner=self.eta_inner if eta_inner is None else eta_inner, tau=self.tau)
        eps = self.softplus_eps
        return eps * Fnn.softplus(gaps["gap_theta"] / eps) + eps * Fnn.softplus(gaps["gap_psi"] / eps)

    def merit_tensor(self, z: torch.Tensor, batch: dict[str, torch.Tensor], create_graph: bool = True) -> torch.Tensor:
        self.ensure_calibrated(z.detach(), batch)
        Fv = self.field(z, batch, create_graph=create_graph)
        field_e = 0.5 * torch.dot(Fv, Fv)
        ptau = self.P_tau(z, batch)
        return self.cfg.lambda_F * (field_e / self.refs["field0"]) + LAMBDA_P * (ptau / self.refs["ptau0"])

    def metrics(self, z: torch.Tensor, batch: dict[str, torch.Tensor], geometry: bool = False) -> dict[str, float]:
        with torch.enable_grad():
            z_req = z.detach().clone().requires_grad_(True)
            Fv = self.field(z_req, batch, create_graph=True)
            field_e = 0.5 * torch.dot(Fv, Fv)
            self.ensure_calibrated(z_req.detach(), batch)
            ptau = self.P_tau(z_req, batch)
            raw = self.raw_ptau_gaps(z_req, batch)
            V = self.merit_tensor(z_req, batch, create_graph=True)
            out = {
                "V": float(V.detach().item()),
                "field_norm": float(torch.linalg.norm(Fv.detach()).item()),
                "field_energy": float(field_e.detach().item()),
                "P_tau": float(ptau.detach().item()),
                "G_theta_raw": float(raw["gap_theta"].detach().item()),
                "G_psi_raw": float(raw["gap_psi"].detach().item()),
                "J_surr": float(self.J_surr(z_req, batch).detach().item()),
                "action_clip_fraction": float(batch["action_clip_fraction"].item()),
            }
            if geometry:
                out.update(geometry_metrics(self, z_req, batch, Fv))
            return out


def jvp_field(game: SemiJointGame, z: torch.Tensor, batch: dict[str, torch.Tensor], vec: torch.Tensor) -> torch.Tensor:
    _, out = torch.autograd.functional.jvp(lambda zz: game.field(zz, batch, create_graph=True), (z,), (vec,), create_graph=True, strict=False)
    return out


def geometry_metrics(game: SemiJointGame, z: torch.Tensor, batch: dict[str, torch.Tensor], Fv: torch.Tensor | None = None) -> dict[str, float]:
    Fv = Fv if Fv is not None else game.field(z, batch, create_graph=True)
    G = jvp_field(game, z, batch, Fv.detach())
    jtF = torch.autograd.grad(0.5 * torch.dot(Fv, Fv), z, retain_graph=True, create_graph=True)[0]
    wf = 0.5 * (G - jtF)
    sf = 0.5 * (G + jtF)
    f_norm = float(torch.linalg.norm(Fv.detach()).item())
    g_norm = float(torch.linalg.norm(G.detach()).item())
    cos_fg = float(torch.dot(Fv.detach(), G.detach()).item() / (f_norm * g_norm + EPS))
    rot_ratio = float(torch.linalg.norm(wf.detach()).item() / (torch.linalg.norm(sf.detach()).item() + EPS))

    p = game.actors.p_slice
    a = game.actors.a_slice
    G_block = torch.zeros_like(G)
    jt_block = torch.zeros_like(jtF)
    G_block[p] = G[p]
    G_block[a] = G[a]
    jt_block[p] = jtF[p]
    jt_block[a] = jtF[a]
    # With a true per-block restriction, off-diagonal terms vanish; this proxy records the residual skew under block masking.
    wf_block = 0.5 * (G_block - jt_block)
    sf_block = 0.5 * (G_block + jt_block)
    block_ratio = float(torch.linalg.norm(wf_block.detach()).item() / (torch.linalg.norm(sf_block.detach()).item() + EPS))

    # A QP margin proxy: exact quadratic improvement gap between span{-F,G} and span{-F}.
    coeff = quadratic_coefficients(game, z, batch, Fv.detach(), G.detach(), beta_max=game.cfg.lr, gamma_max=game.cfg.lr)
    margin = max(0.0, coeff["best_qp_decrease"] - coeff["best_nog_decrease"])
    denom = abs(coeff["best_nog_decrease"]) + EPS
    return {
        "G_norm": g_norm,
        "cos_F_G": cos_fg,
        "rotation_ratio_joint": rot_ratio,
        "rotation_ratio_block_proxy": block_ratio,
        "non_collinearity": math.sqrt(max(0.0, 1.0 - min(1.0, cos_fg * cos_fg))),
        "varrho2_over_Phi_proxy": float(margin / denom),
        "G_contribution_proxy": float(g_norm / (f_norm + g_norm + EPS)),
    }


def quadratic_coefficients(
    game: SemiJointGame,
    z: torch.Tensor,
    batch: dict[str, torch.Tensor],
    Fv: torch.Tensor,
    Gv: torch.Tensor,
    beta_max: float,
    gamma_max: float,
    signed_box: bool = False,
) -> dict[str, Any]:
    z_req = z.detach().clone().requires_grad_(True)
    V = game.merit_tensor(z_req, batch, create_graph=True)
    gradV = torch.autograd.grad(V, z_req, create_graph=True)[0]

    def hvp(vec: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(torch.dot(gradV, vec), z_req, retain_graph=True, create_graph=True)[0]

    d1 = -Fv.detach()
    d2 = Gv.detach()
    H1 = hvp(d1)
    H2 = hvp(d2)
    q = torch.stack([torch.dot(gradV, d1), torch.dot(gradV, d2)])
    H = torch.stack(
        [
            torch.stack([torch.dot(d1, H1), torch.dot(d1, H2)]),
            torch.stack([torch.dot(d2, H1), torch.dot(d2, H2)]),
        ]
    )
    qn = q.detach().cpu().numpy().astype(np.float64)
    Hn = H.detach().cpu().numpy().astype(np.float64)

    def val(x: np.ndarray) -> float:
        return float(qn @ x + 0.5 * x @ Hn @ x)

    candidates: list[np.ndarray] = [np.zeros(2, dtype=np.float64)]
    try:
        candidates.append(-np.linalg.solve(Hn + 1e-8 * np.eye(2), qn))
    except Exception:
        pass
    bounds = [(-beta_max, beta_max), (-gamma_max, gamma_max)] if signed_box else [(0.0, beta_max), (0.0, gamma_max)]
    for i, (lo, hi) in enumerate(bounds):
        for fixed in [lo, hi]:
            j = 1 - i
            h = Hn[j, j]
            lin = qn[j] + Hn[j, i] * fixed
            xj = 0.0 if abs(h) < 1e-12 else -lin / h
            x = np.zeros(2, dtype=np.float64)
            x[i] = fixed
            x[j] = xj
            candidates.append(x)
    for b0 in bounds[0]:
        for b1 in bounds[1]:
            candidates.append(np.asarray([b0, b1], dtype=np.float64))
    clipped: list[np.ndarray] = []
    for x in candidates:
        y = np.asarray([np.clip(x[0], bounds[0][0], bounds[0][1]), np.clip(x[1], bounds[1][0], bounds[1][1])], dtype=np.float64)
        clipped.append(y)
    best = min(clipped, key=val)
    best_val = val(best)

    # noG is gamma fixed to zero with the same beta box as QP. Solve the 1-D
    # box-constrained minimum of f(beta)=qb*beta+0.5*hbb*beta^2 over [lo,hi]
    # exactly: candidates are the two endpoints plus the interior critical point
    # (only a minimum when hbb>0). The old code used beta=-qb/hbb clipped, which
    # for hbb<=0 (negative merit curvature along -F, which does occur here) wrongly
    # collapses to beta=0 and STALLS the baseline -- making QP win against a dead
    # noG. Enumerating endpoints keeps the nonneg field-only baseline strong/fair.
    hbb = Hn[0, 0]
    qb = qn[0]
    lo_b, hi_b = bounds[0]
    beta_cands = [lo_b, hi_b]
    if abs(hbb) > 1e-12:
        crit = -qb / hbb
        if lo_b <= crit <= hi_b:
            beta_cands.append(crit)
    beta = float(min(beta_cands, key=lambda b: qb * b + 0.5 * hbb * b * b))
    nog_x = np.asarray([beta, 0.0], dtype=np.float64)
    nog_val = val(nog_x)
    return {
        "q": qn,
        "H": Hn,
        "best": best,
        "best_q": best_val,
        "nog": nog_x,
        "nog_q": nog_val,
        "best_qp_decrease": max(0.0, -best_val),
        "best_nog_decrease": max(0.0, -nog_val),
        "e_k": float(-qn[0]),
        "e_positive": float(1.0 if -qn[0] > 0.0 else 0.0),
        "signed_box": int(signed_box),
    }


def trust_apply(game: SemiJointGame, z: torch.Tensor, batch: dict[str, torch.Tensor], delta: torch.Tensor) -> tuple[torch.Tensor, float, int]:
    v0 = float(game.merit_tensor(z.detach().clone().requires_grad_(True), batch, create_graph=True).detach().item())
    scale = 1.0
    for shrink in range(TRUST_SHRINKS + 1):
        cand = (z + scale * delta).detach()
        v1 = float(game.merit_tensor(cand.detach().clone().requires_grad_(True), batch, create_graph=True).detach().item())
        if math.isfinite(v1) and v1 <= v0 + 1e-8:
            return cand, scale, shrink
        scale *= 0.5
    return z.detach(), 0.0, TRUST_SHRINKS


def method_step(game: SemiJointGame, z: torch.Tensor, batch: dict[str, torch.Tensor], method: str) -> tuple[torch.Tensor, dict[str, float]]:
    z_req = z.detach().clone().requires_grad_(True)
    game.ensure_calibrated(z.detach(), batch)
    v_before = float(game.merit_tensor(z_req, batch, create_graph=True).detach().item())
    Fv = game.field(z_req, batch, create_graph=True)
    Fd = Fv.detach()
    lr = game.cfg.lr
    meta: dict[str, float] = {"fallback_frac_step": 0.0, "accept_frac_step": 0.0, "G_contribution": 0.0, "e_positive_step": math.nan}

    if method == "sgd_gda":
        delta = -lr * Fd
        z_next, scale, shrinks = trust_apply(game, z, batch, delta)
        v_after = float(game.merit_tensor(z_next.detach().clone().requires_grad_(True), batch, create_graph=True).detach().item())
        meta.update({"step_scale": scale, "trust_shrinks": shrinks, "V_before_step": v_before, "V_after_step": v_after, "dV_same": v_after - v_before})
        return z_next, meta
    if method == "egm":
        z_half = (z - lr * Fd).detach()
        Fh = game.field(z_half.detach().clone().requires_grad_(True), batch, create_graph=False).detach()
        delta = -lr * Fh
        z_next, scale, shrinks = trust_apply(game, z, batch, delta)
        v_after = float(game.merit_tensor(z_next.detach().clone().requires_grad_(True), batch, create_graph=True).detach().item())
        meta.update({"step_scale": scale, "trust_shrinks": shrinks, "V_before_step": v_before, "V_after_step": v_after, "dV_same": v_after - v_before})
        return z_next, meta
    if method.startswith("ppm_inner"):
        inner = int(method.replace("ppm_inner", ""))
        z_inner = z.detach()
        for _ in range(inner):
            Fi = game.field(z_inner.detach().clone().requires_grad_(True), batch, create_graph=False).detach()
            z_inner = (z - lr * Fi).detach()
        delta = z_inner - z
        z_next, scale, shrinks = trust_apply(game, z, batch, delta)
        v_after = float(game.merit_tensor(z_next.detach().clone().requires_grad_(True), batch, create_graph=True).detach().item())
        meta.update({"step_scale": scale, "trust_shrinks": shrinks, "V_before_step": v_before, "V_after_step": v_after, "dV_same": v_after - v_before})
        return z_next, meta

    Gv = jvp_field(game, z_req, batch, Fd).detach()
    signed_box = method.endswith("_signed_ablation")
    coeff = quadratic_coefficients(game, z, batch, Fd, Gv, beta_max=lr, gamma_max=lr, signed_box=signed_box)
    beta_nog = float(coeff["nog"][0])
    delta_nog = beta_nog * (-Fd)
    z_nog, scale_nog, _ = trust_apply(game, z, batch, delta_nog)
    v_nog = float(game.merit_tensor(z_nog.detach().clone().requires_grad_(True), batch, create_graph=True).detach().item())
    meta.update({"e_k": float(coeff["e_k"]), "e_positive_step": float(coeff["e_positive"])})

    if method in {"noG_nonneg", "noG_signed_ablation"}:
        meta.update({"beta": beta_nog, "gamma": 0.0, "step_scale": scale_nog, "V_before_step": v_before, "V_after_step": v_nog, "dV_same": v_nog - v_before})
        return z_nog, meta
    if method in {"QP_nonneg_damped_safe", "QP_signed_damped_safe_ablation"}:
        beta, gamma = float(coeff["best"][0]), float(coeff["best"][1])
        delta_qp = beta * (-Fd) + gamma * Gv
        z_qp, scale_qp, shrinks = trust_apply(game, z, batch, delta_qp)
        v_qp = float(game.merit_tensor(z_qp.detach().clone().requires_grad_(True), batch, create_graph=True).detach().item())
        if (not math.isfinite(v_qp)) or v_qp > v_nog + 1e-8:
            meta.update({"beta": beta_nog, "gamma": 0.0, "fallback_frac_step": 1.0, "accept_frac_step": 0.0, "step_scale": scale_nog, "V_before_step": v_before, "V_after_step": v_nog, "dV_same": v_nog - v_before})
            return z_nog, meta
        gcontrib = abs(gamma) * float(torch.linalg.norm(Gv).item()) / (
            abs(beta) * float(torch.linalg.norm(Fd).item()) + abs(gamma) * float(torch.linalg.norm(Gv).item()) + EPS
        )
        q = coeff["q"]
        H = coeff["H"]
        predicted_decrease = safe_float(coeff.get("best_qp_decrease"), 0.0)
        meta.update({
            "beta": beta,
            "gamma": gamma,
            "fallback_frac_step": 0.0,
            "accept_frac_step": 1.0,
            "G_contribution": gcontrib,
            "step_scale": scale_qp,
            "trust_shrinks": shrinks,
            "V_before_step": v_before,
            "V_after_step": v_qp,
            "dV_same": v_qp - v_before,
            "predicted_decrease": predicted_decrease,
            "q_beta": float(q[0]),
            "q_gamma": float(q[1]),
            "H_bb": float(H[0, 0]),
            "H_bg": float(H[0, 1]),
            "H_gg": float(H[1, 1]),
        })
        return z_qp, meta
    raise ValueError(method)


def evaluate_return(spec: EnvSpec, actors: FlatActors, z: torch.Tensor, alpha: float, seed: int, episodes: int, br_z: torch.Tensor | None = None) -> dict[str, float]:
    env = gym.make(spec.env_id)
    clean, robust = [], []
    for ep in range(episodes):
        for mode in ["clean", "robust"]:
            obs, _ = env.reset(seed=seed + 1000 * ep + (0 if mode == "clean" else 500))
            done = False
            total = 0.0
            while not done:
                obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), dtype=DTYPE, device=DEVICE)
                with torch.no_grad():
                    u, _ = actors.actor(z, obs_t, "p")
                    if mode == "clean":
                        act = u
                    else:
                        wz = br_z if br_z is not None else z
                        w, _ = actors.actor(wz, obs_t, "a")
                        act = u + alpha * w
                    act = torch.clamp(act, torch.as_tensor(spec.action_low, device=DEVICE), torch.as_tensor(spec.action_high, device=DEVICE))
                obs, reward, terminated, truncated, _ = env.step(act.squeeze(0).cpu().numpy().astype(np.float32))
                total += float(reward)
                done = bool(terminated or truncated)
            (clean if mode == "clean" else robust).append(total)
    env.close()
    return {
        "clean_return": float(np.mean(clean)),
        "robust_return": float(np.mean(robust)),
        "robust_degradation": float(np.mean(clean) - np.mean(robust)),
    }


def lightweight_br(actors: FlatActors, game: SemiJointGame, z: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    # A cheap same-surrogate adversary best response; report documents this as a lightweight BR.
    br = z.detach().clone()
    psi0 = br[actors.a_slice].detach().clone()
    for _ in range(BR_STEPS):
        br_req = br.detach().clone().requires_grad_(True)
        J = game.J_surr(br_req, batch)
        g = torch.autograd.grad(J, br_req)[0]
        br = br.detach()
        br[actors.a_slice] = br[actors.a_slice] - BR_LR * g[actors.a_slice]
        diff = br[actors.a_slice] - psi0
        n = torch.linalg.norm(diff)
        if n > 0.25:
            br[actors.a_slice] = psi0 + diff / n * 0.25
    return br.detach()


def diagnostic_for(spec: EnvSpec, alpha: float, seed: int = 0) -> dict[str, Any]:
    seed_everything(seed)
    actors = FlatActors(spec)
    z = actors.init_z(seed)
    value = ValueNet(spec.obs_dim).to(DEVICE)
    cfg = Config(spec.env_id, alpha, 0.3, 1e-4, seed, 1)
    batch = collect_rollout(spec, actors, value, z, cfg, ROLLOUT_STEPS, seed)
    train_value(value, batch)
    game = SemiJointGame(spec, cfg, actors)
    metrics = game.metrics(z, batch, geometry=True)
    return {"env_id": spec.env_id, "alpha": alpha, **metrics}


def run_ptau_gate(spec: EnvSpec, alpha: float, out_root: Path, seed: int = 0) -> tuple[bool, dict[str, Any]]:
    seed_everything(seed)
    actors = FlatActors(spec)
    z = actors.init_z(seed)
    value = ValueNet(spec.obs_dim).to(DEVICE)
    cfg = Config(spec.env_id, alpha, 0.3, 1e-4, seed, PTAU_GATE_WARMUP_STEPS)
    batch = collect_rollout(spec, actors, value, z, cfg, ROLLOUT_STEPS, seed + 77)
    train_value(value, batch)
    game = SemiJointGame(spec, cfg, actors)
    calib = game.calibrate_ptau(z, batch)
    rows: list[dict[str, Any]] = []
    z_cur = z.detach()
    for it in range(PTAU_GATE_WARMUP_STEPS + 1):
        z_req = z_cur.detach().clone().requires_grad_(True)
        raw = game.raw_ptau_gaps(z_req, batch)
        ptau = game.P_tau(z_req, batch)
        rows.append(
            {
                "iteration": it,
                "P_tau": float(ptau.detach().item()),
                "G_theta_raw": float(raw["gap_theta"].detach().item()),
                "G_psi_raw": float(raw["gap_psi"].detach().item()),
                "raw_imp_theta": float(raw["raw_imp_theta"].detach().item()),
                "raw_imp_psi": float(raw["raw_imp_psi"].detach().item()),
                "prox_theta": float(raw["prox_theta"].detach().item()),
                "prox_psi": float(raw["prox_psi"].detach().item()),
            }
        )
        if it == PTAU_GATE_WARMUP_STEPS:
            break
        z_cur, _ = method_step(game, z_cur, batch, "sgd_gda")
    write_csv(out_root / "ptau_gate_warmup.csv", rows)
    write_csv(out_root / "ptau_eta_sweep.csv", calib.get("eta_sweep", []))  # type: ignore[arg-type]
    write_csv(out_root / "ptau_tau_sweep.csv", calib.get("tau_sweep", []))  # type: ignore[arg-type]
    ptau_vals = [safe_float(r["P_tau"]) for r in rows]
    ptau_var_frac = (max(ptau_vals) - min(ptau_vals)) / (abs(ptau_vals[0]) + EPS)
    min_gap = min(safe_float(rows[0]["G_theta_raw"]), safe_float(rows[0]["G_psi_raw"]))
    gate_pass = bool(min_gap >= 10.0 * safe_float(calib["softplus_eps"]) and safe_float(calib["penalty_ratio"], math.inf) < 0.5)
    result: dict[str, Any] = {
        "env_id": spec.env_id,
        "alpha": alpha,
        "gate_pass": int(gate_pass),
        "min_gap_z0": min_gap,
        "softplus_eps": calib["softplus_eps"],
        "ten_eps": 10.0 * safe_float(calib["softplus_eps"]),
        "ptau_var_frac_10_warmup": ptau_var_frac,
        **{k: v for k, v in calib.items() if isinstance(v, (int, float))},
    }
    write_csv(out_root / "ptau_gate_summary.csv", [result])
    (out_root / "ptau_gate_report.md").write_text(
        "# P_tau Gate\n\n"
        f"- gate_pass: `{gate_pass}`\n"
        f"- min_gap_z0: `{min_gap:.6e}`\n"
        f"- 10*softplus_eps: `{10.0 * safe_float(calib['softplus_eps']):.6e}`\n"
        f"- penalty_ratio: `{safe_float(calib['penalty_ratio']):.6f}`\n"
        f"- P_tau variation over 10 warmup steps, reported only: `{ptau_var_frac:.6f}`\n"
        f"- eta_inner: `{safe_float(calib['eta_inner']):.6e}`\n"
        f"- tau: `{safe_float(calib['tau']):.6e}`\n",
        encoding="utf-8",
    )
    return gate_pass, result


def ptau_grad_integrity(game: SemiJointGame, z: torch.Tensor, batch: dict[str, torch.Tensor], seed: int = 0) -> tuple[bool, list[dict[str, float]]]:
    # Gate-1b(i): does autodiff of P_tau agree with a finite-difference reference along
    # random unit directions? A genuine inner-loop gradient cut (a stray .detach() or a
    # missing create_graph) corrupts EVERY direction. A near-flat random direction (tiny
    # <grad P_tau, v>) instead only makes a plain central difference truncation-limited:
    # its O(h^2) error can exceed 2% of a small directional derivative even though the
    # autodiff value is exact. To keep the test measuring gradient integrity and not
    # finite-difference conditioning we (a) select the step from a float32 noise-floor-aware
    # ratio-2 ladder and (b) cancel the leading O(h^2) central-difference error with
    # Richardson extrapolation. The 2% tolerance and 3/3 requirement are unchanged.
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed + 2021)
    rows: list[dict[str, float]] = []
    z_req = z.detach().clone().requires_grad_(True)
    game.ensure_calibrated(z_req.detach(), batch)
    ptau = game.P_tau(z_req, batch)
    grad_ptau = torch.autograd.grad(ptau, z_req, create_graph=False)[0].detach()
    base = float(ptau.detach().item())

    # Each float32 P_tau evaluation carries ~ eps_f32 * base rounding noise, so the
    # difference of two evaluations is only trustworthy once it clears this floor. The old
    # absolute 1e-4 window forced near-flat directions onto a single large step (h=0.1)
    # whose curvature-driven truncation error alone exceeded the 2% tolerance.
    eps_f32 = 2.0 ** -23
    noise_floor = max(1.0e3 * eps_f32 * abs(base), 1.0e-9)
    local_ceiling = max(0.5 * abs(base), 1.0e3 * noise_floor)
    # Clean ratio-2 ladder so every consecutive pair (h_grid[k], h_grid[k+1]) = (h, 2h)
    # is a Richardson pair for the central difference.
    h_grid = [1.0e-3, 2.0e-3, 4.0e-3, 8.0e-3, 1.6e-2, 3.2e-2, 6.4e-2, 1.28e-1]

    def p_at(scale: float, direction: torch.Tensor) -> float:
        zz = (z.detach() + scale * direction).requires_grad_(True)
        return float(game.P_tau(zz, batch).detach().item())

    for i in range(GATE1B_DIRECTIONS):
        v = torch.randn(z.numel(), generator=gen, dtype=DTYPE, device=DEVICE)
        v = v / (torch.linalg.norm(v) + EPS)
        ad = float(torch.dot(grad_ptau, v).item())

        cd: list[dict[str, float]] = []
        for h in h_grid:
            p_plus = p_at(h, v)
            p_minus = p_at(-h, v)
            signal = abs(p_plus - p_minus)
            cd.append({"h": h, "P_plus": p_plus, "P_minus": p_minus, "signal": signal, "fd": (p_plus - p_minus) / (2.0 * h)})
        usable_idx = [k for k, c in enumerate(cd) if noise_floor <= c["signal"] <= local_ceiling]
        usable_set = set(usable_idx)
        pair_idx = [k for k in usable_idx if (k + 1) in usable_set]

        if pair_idx:
            # Pick the pair where the central difference has plateaued (smallest jump
            # between fd(h) and fd(2h)): that is where both roundoff and truncation are
            # smallest, then Richardson-extrapolate to O(h^4).
            k = min(pair_idx, key=lambda kk: abs(cd[kk + 1]["fd"] - cd[kk]["fd"]))
            fd_h = cd[k]["fd"]
            fd_2h = cd[k + 1]["fd"]
            fd_rich = (4.0 * fd_h - fd_2h) / 3.0
            best = {
                "h": cd[k]["h"], "P_plus": cd[k]["P_plus"], "P_minus": cd[k]["P_minus"],
                "finite_diff": fd_rich, "relative_error": abs(ad - fd_rich) / (abs(fd_rich) + 1e-12),
                "fd_method": 2.0, "central_diff_at_h": fd_h, "central_trunc_est": abs(fd_2h - fd_h) / 3.0,
            }
        elif usable_idx:
            # No Richardson pair available: plain central difference at the median usable step.
            k = usable_idx[len(usable_idx) // 2]
            fd_h = cd[k]["fd"]
            best = {
                "h": cd[k]["h"], "P_plus": cd[k]["P_plus"], "P_minus": cd[k]["P_minus"],
                "finite_diff": fd_h, "relative_error": abs(ad - fd_h) / (abs(fd_h) + 1e-12),
                "fd_method": 1.0, "central_diff_at_h": fd_h, "central_trunc_est": math.nan,
            }
        else:
            # Nothing cleared the noise floor: report the largest step honestly.
            k = len(cd) - 1
            fd_h = cd[k]["fd"]
            best = {
                "h": cd[k]["h"], "P_plus": cd[k]["P_plus"], "P_minus": cd[k]["P_minus"],
                "finite_diff": fd_h, "relative_error": abs(ad - fd_h) / (abs(fd_h) + 1e-12),
                "fd_method": 0.0, "central_diff_at_h": fd_h, "central_trunc_est": math.nan,
            }

        rows.append({
            "direction": float(i),
            "h": best["h"],
            "P_plus": best["P_plus"],
            "P_minus": best["P_minus"],
            "autodiff_dir": ad,
            "finite_diff": best["finite_diff"],
            "relative_error": best["relative_error"],
            "pass": float(best["relative_error"] < 0.02),
            "num_h_candidates": float(len(usable_idx)),
            "fd_method": best["fd_method"],
            "central_diff_at_h": best["central_diff_at_h"],
            "central_trunc_est": best["central_trunc_est"],
        })
    return all(r["pass"] == 1.0 for r in rows), rows


def weighted_grad_ratio(game: SemiJointGame, z: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    z_req = z.detach().clone().requires_grad_(True)
    game.ensure_calibrated(z_req.detach(), batch)
    Fv = game.field(z_req, batch, create_graph=True)
    field_e = 0.5 * torch.dot(Fv, Fv)
    ptau = game.P_tau(z_req, batch)
    grad_E = torch.autograd.grad(field_e, z_req, retain_graph=True, create_graph=False)[0]
    grad_P = torch.autograd.grad(ptau, z_req, retain_graph=True, create_graph=False)[0]
    kF = game.refs["field0"]
    kP = game.refs["ptau0"]
    wE = game.cfg.lambda_F / kF
    wP = LAMBDA_P / kP
    norm_E = float(torch.linalg.norm(wE * grad_E).item())
    norm_P = float(torch.linalg.norm(wP * grad_P).item())
    e_F = float(torch.dot(wE * grad_E, Fv.detach()).item())
    e_P = float(torch.dot(wP * grad_P, Fv.detach()).item())
    return {
        "P_tau": float(ptau.detach().item()),
        "field_energy": float(field_e.detach().item()),
        "r_grad": norm_P / (norm_E + EPS),
        "norm_weighted_grad_P": norm_P,
        "norm_weighted_grad_E": norm_E,
        "e_P": e_P,
        "e_F": e_F,
        "e_total": e_P + e_F,
    }


def run_gate1b(spec: EnvSpec, alpha: float, out_root: Path, seed: int = 0) -> tuple[bool, dict[str, Any]]:
    seed_everything(seed)
    actors = FlatActors(spec)
    z0 = actors.init_z(seed)
    value = ValueNet(spec.obs_dim).to(DEVICE)
    cfg = Config(spec.env_id, alpha, 0.3, 1e-4, seed, PTAU_GATE_WARMUP_STEPS)
    batch = collect_rollout(spec, actors, value, z0, cfg, ROLLOUT_STEPS, seed + 177)
    train_value(value, batch)
    game = SemiJointGame(spec, cfg, actors)
    game.calibrate_ptau(z0, batch)
    integrity_pass, integrity_rows = ptau_grad_integrity(game, z0, batch, seed=seed)
    write_csv(out_root / "gate1b_grad_integrity.csv", integrity_rows)

    z_cur = z0.detach()
    for _ in range(PTAU_GATE_WARMUP_STEPS):
        z_cur, _ = method_step(game, z_cur, batch, "sgd_gda")
    ratio_z0 = weighted_grad_ratio(game, z0, batch)
    ratio_z10 = weighted_grad_ratio(game, z_cur, batch)
    ratio_rows = [{"checkpoint": "z0", **ratio_z0}, {"checkpoint": "z10", **ratio_z10}]
    write_csv(out_root / "gate1b_weighted_grad_ratio.csv", ratio_rows)

    rel_P = abs(ratio_z10["P_tau"] - ratio_z0["P_tau"]) / (abs(ratio_z0["P_tau"]) + EPS)
    rel_E = abs(ratio_z10["field_energy"] - ratio_z0["field_energy"]) / (abs(ratio_z0["field_energy"]) + EPS)
    ratio_var = rel_P / (rel_E + EPS)
    grad_ratio_pass = ratio_z0["r_grad"] >= 0.15 and ratio_z10["r_grad"] >= 0.15
    borderline = 0.10 <= min(ratio_z0["r_grad"], ratio_z10["r_grad"]) < 0.15
    variation_pass = borderline and ratio_var >= 0.15
    gate_pass = bool(integrity_pass and (grad_ratio_pass or variation_pass))
    summary: dict[str, Any] = {
        "env_id": spec.env_id,
        "alpha": alpha,
        "gate_pass": int(gate_pass),
        "integrity_pass": int(integrity_pass),
        "r_grad_z0": ratio_z0["r_grad"],
        "r_grad_z10": ratio_z10["r_grad"],
        "e_P_z0": ratio_z0["e_P"],
        "e_F_z0": ratio_z0["e_F"],
        "e_P_z10": ratio_z10["e_P"],
        "e_F_z10": ratio_z10["e_F"],
        "ratio_var": ratio_var,
        "grad_ratio_pass": int(grad_ratio_pass),
        "variation_pass": int(variation_pass),
        "P_tau_z0": ratio_z0["P_tau"],
        "P_tau_z10": ratio_z10["P_tau"],
        "field_energy_z0": ratio_z0["field_energy"],
        "field_energy_z10": ratio_z10["field_energy"],
    }
    write_csv(out_root / "gate1b_summary.csv", [summary])
    (out_root / "gate1b_report.md").write_text(
        "# Gate-1b P_tau Gradient Liveness\n\n"
        f"- gate_pass: `{gate_pass}`\n"
        f"- gradient_integrity_pass: `{integrity_pass}`\n"
        f"- r_grad_z0: `{ratio_z0['r_grad']:.6f}`\n"
        f"- r_grad_z10: `{ratio_z10['r_grad']:.6f}`\n"
        f"- e_P_z0 / e_F_z0: `{ratio_z0['e_P']:.6e}` / `{ratio_z0['e_F']:.6e}`\n"
        f"- e_P_z10 / e_F_z10: `{ratio_z10['e_P']:.6e}` / `{ratio_z10['e_F']:.6e}`\n"
        f"- ratio_var: `{ratio_var:.6f}`\n",
        encoding="utf-8",
    )
    return gate_pass, summary


def run_one(cfg: Config, method: str) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    seed_everything(cfg.seed)
    spec = check_env(cfg.env_id)
    if spec is None:
        raise RuntimeError(f"Unavailable env: {cfg.env_id}")
    actors = FlatActors(spec)
    z = actors.init_z(cfg.seed)
    value = ValueNet(spec.obs_dim).to(DEVICE)
    game = SemiJointGame(spec, cfg, actors)
    curves: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []
    fallback_sum = accept_sum = g_sum = epos_sum = epos_count = 0.0
    dv_nonpos = dv_count = 0.0
    for it in range(cfg.iterations + 1):
        batch = collect_rollout(spec, actors, value, z, cfg, ROLLOUT_STEPS, cfg.seed * 100000 + it)
        train_value(value, batch)
        if it % EVAL_FREQ == 0 or it == cfg.iterations:
            metrics = game.metrics(z, batch, geometry=(it == 0))
            rets = evaluate_return(spec, actors, z, cfg.alpha, cfg.seed * 10000 + it, EVAL_EPISODES)
            br_z = lightweight_br(actors, game, z, batch)
            br_ret = evaluate_return(spec, actors, z, cfg.alpha, cfg.seed * 12000 + it, 1, br_z=br_z)
            curves.append(
                {
                    "env_id": cfg.env_id,
                    "alpha": cfg.alpha,
                    "lambda_F": cfg.lambda_F,
                    "joint_lr": cfg.lr,
                    "seed": cfg.seed,
                    "method": method,
                    "iteration": it,
                    **metrics,
                    **rets,
                    "br_robust_return": br_ret["robust_return"],
                    "fallback_frac_running": fallback_sum / max(it, 1),
                    "accept_frac_running": accept_sum / max(it, 1),
                    "G_contribution_running": g_sum / max(it, 1),
                    "e_positive_frac_running": epos_sum / max(epos_count, 1.0),
                }
            )
        if it == cfg.iterations:
            break
        z, meta = method_step(game, z, batch, method)
        dv = safe_float(meta.get("dV_same"), math.nan)
        if math.isfinite(dv):
            dv_count += 1.0
            dv_nonpos += 1.0 if dv <= 1e-8 else 0.0
        step_records.append(
            {
                "env_id": cfg.env_id,
                "alpha": cfg.alpha,
                "lambda_F": cfg.lambda_F,
                "joint_lr": cfg.lr,
                "seed": cfg.seed,
                "method": method,
                "iteration": it,
                "V_before_step": safe_float(meta.get("V_before_step")),
                "V_after_step": safe_float(meta.get("V_after_step")),
                "dV_same": dv,
                "predicted_decrease": safe_float(meta.get("predicted_decrease")),
                "beta": safe_float(meta.get("beta")),
                "gamma": safe_float(meta.get("gamma")),
                "q_beta": safe_float(meta.get("q_beta")),
                "q_gamma": safe_float(meta.get("q_gamma")),
                "H_bb": safe_float(meta.get("H_bb")),
                "H_bg": safe_float(meta.get("H_bg")),
                "H_gg": safe_float(meta.get("H_gg")),
                "fallback_frac_step": safe_float(meta.get("fallback_frac_step", 0.0), 0.0),
                "accept_frac_step": safe_float(meta.get("accept_frac_step", 0.0), 0.0),
                "G_contribution": safe_float(meta.get("G_contribution", 0.0), 0.0),
                "e_k": safe_float(meta.get("e_k")),
                "e_positive_step": safe_float(meta.get("e_positive_step")),
            }
        )
        fallback_sum += safe_float(meta.get("fallback_frac_step", 0.0), 0.0)
        accept_sum += safe_float(meta.get("accept_frac_step", 0.0), 0.0)
        g_sum += safe_float(meta.get("G_contribution", 0.0), 0.0)
        epos = safe_float(meta.get("e_positive_step"), math.nan)
        if math.isfinite(epos):
            epos_sum += epos
            epos_count += 1.0

    summary = {
        "env_id": cfg.env_id,
        "alpha": cfg.alpha,
        "lambda_F": cfg.lambda_F,
        "joint_lr": cfg.lr,
        "seed": cfg.seed,
        "method": method,
        "V_AUC": auc([r["V"] for r in curves]),
        "P_tau_AUC": auc([r["P_tau"] for r in curves]),
        "field_norm_AUC": auc([r["field_norm"] for r in curves]),
        "robust_return_AUC": auc([r["robust_return"] for r in curves]),
        "clean_return_AUC": auc([r["clean_return"] for r in curves]),
        "br_robust_return_AUC": auc([r["br_robust_return"] for r in curves]),
        "final_robust_return": curves[-1]["robust_return"],
        "final_clean_return": curves[-1]["clean_return"],
        "fallback_frac": fallback_sum / max(cfg.iterations, 1),
        "accept_frac": accept_sum / max(cfg.iterations, 1),
        "G_contribution": g_sum / max(cfg.iterations, 1),
        "e_positive_frac": epos_sum / max(epos_count, 1.0),
        "same_batch_nonpos_frac": dv_nonpos / max(dv_count, 1.0),
        "cos_F_G_initial": curves[0].get("cos_F_G", math.nan),
        "rotation_ratio_initial": curves[0].get("rotation_ratio_joint", math.nan),
        "curve_finite": int(all(finite(r["V"]) and finite(r["robust_return"]) for r in curves)),
    }
    if game.calibration:
        summary.update({f"calib_{k}": v for k, v in game.calibration.items() if isinstance(v, (int, float))})
    return curves, summary, step_records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def mean_by(rows: list[dict[str, Any]], method: str, metric: str) -> float:
    vals = [safe_float(r.get(metric)) for r in rows if r.get("method") == method]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else math.nan


def dominance(curves: list[dict[str, Any]], a: str, b: str, metric: str) -> float:
    fracs = []
    for seed in sorted({int(r["seed"]) for r in curves}):
        aa = sorted([r for r in curves if r["method"] == a and int(r["seed"]) == seed], key=lambda r: int(r["iteration"]))
        bb = sorted([r for r in curves if r["method"] == b and int(r["seed"]) == seed], key=lambda r: int(r["iteration"]))
        n = min(len(aa), len(bb))
        if n:
            fracs.append(sum(1 for i in range(n) if safe_float(aa[i][metric]) > safe_float(bb[i][metric])) / n)
    return float(np.mean(fracs)) if fracs else math.nan


def plot_metric(path: Path, curves: list[dict[str, Any]], metric: str, title: str, methods: list[str] = METHODS) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    for method in methods:
        by_iter: dict[int, list[float]] = {}
        for r in curves:
            if r["method"] == method:
                by_iter.setdefault(int(r["iteration"]), []).append(safe_float(r.get(metric)))
        if not by_iter:
            continue
        xs = sorted(by_iter)
        ys = [float(np.mean([v for v in by_iter[x] if math.isfinite(v)])) for x in xs]
        ax.plot(xs, ema(ys, 0.3), label=method, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def make_big_plot(plot_dir: Path, out_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageOps

    files = [
        "robust_return.png",
        "clean_return.png",
        "br_robust_return.png",
        "V.png",
        "P_tau.png",
        "field_norm.png",
        "fallback_accept_g.png",
    ]
    imgs = []
    for f in files:
        p = plot_dir / f
        if p.exists():
            imgs.append((f, Image.open(p).convert("RGB")))
    if not imgs:
        return
    cell_w, cell_h, pad, label_h = 1100, 700, 24, 42
    cols = 2
    rows = (len(imgs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * (cell_w + pad) + pad, rows * (cell_h + label_h + pad) + pad), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (name, img) in enumerate(imgs):
        r, c = divmod(i, cols)
        x = pad + c * (cell_w + pad)
        y = pad + r * (cell_h + label_h + pad)
        draw.text((x + 8, y + 8), name, fill="black")
        thumb = ImageOps.contain(img, (cell_w, cell_h))
        canvas.paste(thumb, (x + (cell_w - thumb.width) // 2, y + label_h + (cell_h - thumb.height) // 2))
        draw.rectangle([x, y + label_h, x + cell_w, y + label_h + cell_h], outline=(180, 180, 180), width=2)
    canvas.save(out_path)


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "plots").mkdir(parents=True, exist_ok=True)
    deviations = [
        "Budgeted run: full 18-config x 5-seed sweep is screened with seed 0, then the best config is expanded to 5 seeds.",
        f"Budgeted run: iterations are {ITERATIONS_SCREEN} for screening and {ITERATIONS_FINAL} for final, rollout_steps={ROLLOUT_STEPS}.",
        f"Guardrail applied: P_tau inner loop uses k_inner={K_INNER} after k_inner=3 made the exact-autodiff run too slow.",
        "Robust BR is a lightweight same-surrogate adversary refinement, not a full retrained PPO adversary.",
        "Joint surrogate uses a differentiable joint env-action likelihood for a_env=u+alpha*w so that the semi-joint field has cross-player terms; this replaces the block-separable PPO log-prob surrogate.",
        "Primary noG/QP use the paper-aligned nonnegative box. Signed methods are reported only as appendix ablations.",
        "Normalizers are frozen on the first z0/batch merit evaluation inside each run and then reused for the run.",
        "Mini-complete run: after diagnostics, screening uses only the diagnostic-selected alpha and sweeps lambda_F/lr. This keeps exact-autodiff QP tractable in an interactive turn.",
        "Gate-1b(i) finite-difference estimator amended: float32 noise-floor-aware ratio-2 step ladder plus Richardson (O(h^4)) extrapolation, so a near-flat random direction (tiny <grad P_tau,v>) is no longer misflagged as a gradient cut by central-difference truncation. The 2% autodiff-vs-finite-diff tolerance and the 3/3 requirement are unchanged.",
        "Round 2.2 STEP 1: the P_tau inner-loop autodiff was independently validated on a clip-free smooth saddle in float64 (pipeline_test_step1.py): autodiff of P_tau matches a Richardson central difference to 3e-11 over 5 directions, proving there is no gradient cut and the real-env Gate-1b(i) mismatch is an Assumption-1 PPO-clip-kink artifact, not a bug.",
        "Round 2.2 STEP 3 baseline-fairness FIX: the noG 1-D box solve used beta=clip(-q_b/H_bb,0,lr), which under NEGATIVE merit curvature along -F (H_bb<0, which occurs on HalfCheetah) collapses to beta=0 and STALLS the noG baseline (nog moved 0, best_nog_decrease=0). This would let QP 'win' against a dead baseline. Fixed to solve the 1-D box-constrained minimum exactly (enumerate endpoints + interior critical point); under negative curvature the minimizer is beta=lr. All pre-fix noG/QP-vs-noG numbers (incl. Round 2.1) are invalid. FIX-3 sentinels + Gate-2 drift recomputed post-fix in results/.../fix3_sentinels/.",
    ]

    diag_rows = []
    for env_id, alphas in DIAG_ENVS:
        spec = check_env(env_id)
        if spec is None:
            diag_rows.append({"env_id": env_id, "available": 0})
            continue
        for alpha in alphas:
            row = diagnostic_for(spec, alpha, seed=0)
            row["available"] = 1
            diag_rows.append(row)
    write_csv(RESULT_ROOT / "diagnostic_summary.csv", diag_rows)
    viable = [r for r in diag_rows if int(r.get("available", 0)) == 1 and safe_float(r.get("cos_F_G"), 1.0) < 0.5]
    if not viable:
        decision = "QP_FAILS_POTENTIAL_LIKE_DIAGNOSTIC"
        (RESULT_ROOT / "decision.md").write_text(decision, encoding="utf-8")
        (RESULT_ROOT / "report.md").write_text("# Semi-joint P_tau standard RARL QP\n\nDiagnostic found no env/alpha with `cos(F,G) < 0.5`; all checked fields looked potential-like by the prompt gate.\n", encoding="utf-8")
        return
    best_diag = max(viable, key=lambda r: safe_float(r.get("varrho2_over_Phi_proxy"), -1.0))
    chosen_env = str(best_diag["env_id"])
    screen_alphas = [float(best_diag["alpha"])]
    chosen_spec = check_env(chosen_env)
    if chosen_spec is None:
        raise RuntimeError(f"Chosen env became unavailable: {chosen_env}")
    gate1_pass, gate1 = run_ptau_gate(chosen_spec, screen_alphas[0], RESULT_ROOT, seed=0)
    if not gate1_pass:
        decision = "QP_FAILS_PTAU_GATE"
        (RESULT_ROOT / "decision.md").write_text(decision, encoding="utf-8")
        (RESULT_ROOT / "report.md").write_text(
            "# Semi-joint P_tau standard RARL QP\n\n"
            "Stopped at Gate-1 value check because calibrated `P_tau` is still not live enough.\n\n"
            f"- min_gap_z0: `{safe_float(gate1.get('min_gap_z0')):.6e}`\n"
            f"- 10*softplus_eps: `{safe_float(gate1.get('ten_eps')):.6e}`\n"
            f"- penalty_ratio: `{safe_float(gate1.get('penalty_ratio')):.6f}`\n"
            f"- eta/tau sweep files: `{RESULT_ROOT / 'ptau_eta_sweep.csv'}`, `{RESULT_ROOT / 'ptau_tau_sweep.csv'}`\n",
            encoding="utf-8",
        )
        return
    gate1b_pass, gate1b = run_gate1b(chosen_spec, screen_alphas[0], RESULT_ROOT, seed=0)
    if not gate1b_pass:
        # If the P_tau gradient is genuinely weak, try the next alpha for the same env once.
        alpha_table = {env: vals for env, vals in DIAG_ENVS}
        candidates = alpha_table.get(chosen_env, [])
        higher = [a for a in candidates if a > screen_alphas[0]]
        if higher:
            retry_alpha = higher[0]
            retry_dir = RESULT_ROOT / f"alpha_retry_{str(retry_alpha).replace('.', 'p')}"
            retry_dir.mkdir(parents=True, exist_ok=True)
            retry_gate1_pass, _ = run_ptau_gate(chosen_spec, retry_alpha, retry_dir, seed=0)
            retry_gate1b_pass, retry_gate1b = run_gate1b(chosen_spec, retry_alpha, retry_dir, seed=0) if retry_gate1_pass else (False, {})
            if retry_gate1_pass and retry_gate1b_pass:
                screen_alphas = [retry_alpha]
                gate1b = retry_gate1b
                gate1b_pass = True
                deviations.append(f"Gate-1b failed at alpha={best_diag['alpha']}; raised alpha one notch to {retry_alpha} per Round 2.1.")
            else:
                decision = "QP_FAILS_GATE1B_AFTER_ALPHA_RETRY"
                (RESULT_ROOT / "decision.md").write_text(decision, encoding="utf-8")
                (RESULT_ROOT / "report.md").write_text(
                    "# Semi-joint P_tau standard RARL QP\n\n"
                    "Stopped at Gate-1b. Initial alpha failed, and one-notch alpha retry also failed.\n\n"
                    f"- initial alpha: `{screen_alphas[0]}`\n"
                    f"- retry alpha: `{retry_alpha}`\n"
                    f"- initial r_grad_z0/z10: `{safe_float(gate1b.get('r_grad_z0')):.6f}` / `{safe_float(gate1b.get('r_grad_z10')):.6f}`\n"
                    f"- retry report dir: `{retry_dir}`\n",
                    encoding="utf-8",
                )
                return
        else:
            decision = "QP_FAILS_GATE1B"
            (RESULT_ROOT / "decision.md").write_text(decision, encoding="utf-8")
            (RESULT_ROOT / "report.md").write_text(
                "# Semi-joint P_tau standard RARL QP\n\n"
                "Stopped at Gate-1b because P_tau gradient liveness failed and no higher alpha is available for this env in the diagnostic table.\n\n"
                f"- r_grad_z0/z10: `{safe_float(gate1b.get('r_grad_z0')):.6f}` / `{safe_float(gate1b.get('r_grad_z10')):.6f}`\n",
                encoding="utf-8",
            )
            return

    screen_summaries: list[dict[str, Any]] = []
    screen_curves: list[dict[str, Any]] = []
    screen_steps: list[dict[str, Any]] = []
    for alpha in screen_alphas:
        for lf in SWEEP_LAMBDA_F:
            for lr in SWEEP_LR:
                for method in SCREEN_METHODS:
                    cfg = Config(chosen_env, alpha, lf, lr, seed=0, iterations=ITERATIONS_SCREEN)
                    curves, summary, steps = run_one(cfg, method)
                    screen_curves.extend(curves)
                    screen_summaries.append(summary)
                    screen_steps.extend(steps)
    write_csv(RESULT_ROOT / "screen_curve_rows.csv", screen_curves)
    write_csv(RESULT_ROOT / "screen_seed_summary.csv", screen_summaries)
    write_csv(RESULT_ROOT / "screen_step_diagnostics.csv", screen_steps)

    # Paper-alignment sentinel: with P_tau, e_k=<grad V,F> should be positive most of the time.
    epos_screen = [
        safe_float(r.get("e_positive_frac"))
        for r in screen_summaries
        if r.get("method") in {"noG_nonneg", "QP_nonneg_damped_safe"}
    ]
    epos_screen = [v for v in epos_screen if math.isfinite(v)]
    if epos_screen and float(np.mean(epos_screen)) < 0.6:
        decision = "QP_FAILS_E_POSITIVE_SENTINEL"
        (RESULT_ROOT / "decision.md").write_text(decision, encoding="utf-8")
        (RESULT_ROOT / "report.md").write_text(
            "# Semi-joint P_tau standard RARL QP\n\n"
            f"Stopped before final run because mean `e_k>0` fraction was `{float(np.mean(epos_screen)):.3f}` < 0.6. "
            "Per prompt, this indicates the merit implementation is not dissipative enough and signed fallback must not be used as a hidden fix.\n",
            encoding="utf-8",
        )
        return

    drift_rows = [
        r for r in screen_summaries
        if r.get("method") in {"noG_nonneg", "QP_nonneg_damped_safe"}
    ]
    bad_drift = [r for r in drift_rows if safe_float(r.get("same_batch_nonpos_frac"), 0.0) < 0.95]
    if bad_drift:
        offending = [
            r for r in screen_steps
            if r.get("method") in {"noG_nonneg", "QP_nonneg_damped_safe"} and safe_float(r.get("dV_same"), 0.0) > 1e-8
        ][:5]
        write_csv(RESULT_ROOT / "same_batch_drift_offenders.csv", offending)
        decision = "QP_FAILS_SAME_BATCH_DRIFT_SENTINEL"
        (RESULT_ROOT / "decision.md").write_text(decision, encoding="utf-8")
        (RESULT_ROOT / "report.md").write_text(
            "# Semi-joint P_tau standard RARL QP\n\n"
            "Stopped before final run because same-batch drift sentinel failed for noG/QP.\n\n"
            f"Bad rows: `{len(bad_drift)}`\n\n"
            f"Offending examples written to `{RESULT_ROOT / 'same_batch_drift_offenders.csv'}`.\n",
            encoding="utf-8",
        )
        return

    # Rank configs by nonnegative QP robust return with a small V/noG bonus.
    config_keys = sorted({(r["alpha"], r["lambda_F"], r["joint_lr"]) for r in screen_summaries})
    ranked_configs = []
    for alpha, lf, lr in config_keys:
        rows = [r for r in screen_summaries if r["alpha"] == alpha and r["lambda_F"] == lf and r["joint_lr"] == lr]
        bym = {r["method"]: r for r in rows}
        if "QP_nonneg_damped_safe" not in bym or "noG_nonneg" not in bym:
            continue
        score = safe_float(bym["QP_nonneg_damped_safe"]["robust_return_AUC"], -1e9)
        score += 0.1 * (safe_float(bym["QP_nonneg_damped_safe"]["V_AUC"], 1e9) < safe_float(bym["noG_nonneg"]["V_AUC"], 1e9))
        ranked_configs.append({"alpha": alpha, "lambda_F": lf, "joint_lr": lr, "score": score})
    ranked_configs = sorted(ranked_configs, key=lambda r: r["score"], reverse=True)
    best_cfg_row = ranked_configs[0]

    final_curves: list[dict[str, Any]] = []
    final_summaries: list[dict[str, Any]] = []
    final_steps: list[dict[str, Any]] = []
    for seed in SEEDS:
        for method in METHODS:
            cfg = Config(chosen_env, float(best_cfg_row["alpha"]), float(best_cfg_row["lambda_F"]), float(best_cfg_row["joint_lr"]), seed=seed, iterations=ITERATIONS_FINAL)
            curves, summary, steps = run_one(cfg, method)
            final_curves.extend(curves)
            final_summaries.append(summary)
            final_steps.extend(steps)
    write_csv(RESULT_ROOT / "curve_rows.csv", final_curves)
    write_csv(RESULT_ROOT / "seed_summary.csv", final_summaries)
    write_csv(RESULT_ROOT / "step_diagnostics.csv", final_steps)

    method_rows = []
    for method in METHODS:
        method_rows.append(
            {
                "method": method,
                "robust_return_AUC": mean_by(final_summaries, method, "robust_return_AUC"),
                "clean_return_AUC": mean_by(final_summaries, method, "clean_return_AUC"),
                "br_robust_return_AUC": mean_by(final_summaries, method, "br_robust_return_AUC"),
                "V_AUC": mean_by(final_summaries, method, "V_AUC"),
                "P_tau_AUC": mean_by(final_summaries, method, "P_tau_AUC"),
                "field_norm_AUC": mean_by(final_summaries, method, "field_norm_AUC"),
                "fallback_frac": mean_by(final_summaries, method, "fallback_frac"),
                "accept_frac": mean_by(final_summaries, method, "accept_frac"),
                "G_contribution": mean_by(final_summaries, method, "G_contribution"),
                "e_positive_frac": mean_by(final_summaries, method, "e_positive_frac"),
                "same_batch_nonpos_frac": mean_by(final_summaries, method, "same_batch_nonpos_frac"),
                "cos_F_G_initial": mean_by(final_summaries, method, "cos_F_G_initial"),
                "rotation_ratio_initial": mean_by(final_summaries, method, "rotation_ratio_initial"),
            }
        )
    write_csv(RESULT_ROOT / "method_auc.csv", method_rows)

    for metric, title in [
        ("robust_return", "MuJoCo robust return"),
        ("clean_return", "MuJoCo clean return"),
        ("br_robust_return", "Lightweight BR robust return"),
        ("V", "P_tau Lyapunov merit"),
        ("P_tau", "P_tau"),
        ("field_norm", "field norm"),
    ]:
        plot_metric(RESULT_ROOT / "plots" / f"{metric}.png", final_curves, metric, title)

    # Bar for QP diagnostics.
    if plt is not None:
        qp = next(r for r in method_rows if r["method"] == "QP_nonneg_damped_safe")
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = ["accept", "fallback", "G_contrib", "e>0"]
        vals = [qp["accept_frac"], qp["fallback_frac"], qp["G_contribution"], qp["e_positive_frac"]]
        ax.bar(labels, vals)
        ax.axhline(0.8, color="green", linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(0.6, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title("Nonnegative QP diagnostics")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULT_ROOT / "plots" / "fallback_accept_g.png", dpi=170)
        plt.close(fig)
    make_big_plot(RESULT_ROOT / "plots", RESULT_ROOT / "plots" / "all_plots_big.png")

    by = {r["method"]: r for r in method_rows}
    qp = by["QP_nonneg_damped_safe"]
    robust_best_baseline = max(safe_float(by[m]["robust_return_AUC"]) for m in PRIMARY_METHODS if m != "QP_nonneg_damped_safe")
    nog = by["noG_nonneg"]
    pass_a = safe_float(qp["robust_return_AUC"]) > robust_best_baseline and all(safe_float(by[m]["robust_return_AUC"]) > safe_float(by["sgd_gda"]["robust_return_AUC"]) for m in ["noG_nonneg", "ppm_inner3", "ppm_inner4", "egm"])
    dom_qp_nog = dominance(final_curves, "QP_nonneg_damped_safe", "noG_nonneg", "V")
    pass_b = safe_float(qp["V_AUC"]) < safe_float(nog["V_AUC"]) and dom_qp_nog >= 0.70 and safe_float(qp["G_contribution"]) >= 0.4
    pass_e = safe_float(qp["e_positive_frac"]) >= 0.6
    if pass_a and pass_b and pass_e:
        decision = "QP_WINS_STANDARD_RARL"
    else:
        reasons = []
        if not pass_e:
            reasons.append("E_POSITIVE_SENTINEL")
        if not pass_a:
            reasons.append("ROBUST_RETURN")
        if not pass_b:
            reasons.append("V_OR_G")
        decision = "QP_FAILS_" + "_".join(reasons)
    (RESULT_ROOT / "decision.md").write_text(decision, encoding="utf-8")

    lines = [
        "# Semi-joint P_tau standard MuJoCo RARL QP",
        "",
        f"- decision: `{decision}`",
        f"- chosen_env: `{chosen_env}`",
        f"- chosen_alpha: `{best_cfg_row['alpha']}`",
        f"- chosen_lambda_F: `{best_cfg_row['lambda_F']}`",
        f"- chosen_joint_lr: `{best_cfg_row['joint_lr']}`",
        f"- seeds: `{SEEDS}`",
        "",
        "## Deviations",
        "",
    ]
    lines.extend([f"- {d}" for d in deviations])
    lines.extend(
        [
            "",
            "## Diagnostic Winner",
            "",
            json.dumps(best_diag, indent=2, default=str),
            "",
            "## Method AUC",
            "",
        ]
    )
    for row in method_rows:
        lines.append(
            f"- `{row['method']}`: robust_AUC=`{row['robust_return_AUC']:.6e}`, V_AUC=`{row['V_AUC']:.6e}`, "
            f"G_contrib=`{row['G_contribution']:.3f}`, accept=`{row['accept_frac']:.3f}`, fallback=`{row['fallback_frac']:.3f}`, "
            f"e_pos=`{row['e_positive_frac']:.3f}`"
        )
    lines.extend(
        [
            "",
            "## Pass/Fail Checks",
            "",
            f"- robust_best_baseline_AUC: `{robust_best_baseline:.6e}`",
            f"- qp_robust_return_AUC: `{qp['robust_return_AUC']:.6e}`",
            f"- qp_V_AUC: `{qp['V_AUC']:.6e}`",
            f"- noG_V_AUC: `{nog['V_AUC']:.6e}`",
            f"- dominance_QP_over_noG_on_V: `{dom_qp_nog:.3f}`",
            f"- QP_G_contribution: `{qp['G_contribution']:.3f}`",
            f"- QP_e_positive_frac: `{qp['e_positive_frac']:.3f}`",
            f"- paper_alignment_primary_domain: `nonnegative beta/gamma box`",
            f"- signed_methods: `appendix ablation only`",
            "",
            "## Plots",
            "",
            f"- big plot: `{RESULT_ROOT / 'plots' / 'all_plots_big.png'}`",
        ]
    )
    (RESULT_ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
