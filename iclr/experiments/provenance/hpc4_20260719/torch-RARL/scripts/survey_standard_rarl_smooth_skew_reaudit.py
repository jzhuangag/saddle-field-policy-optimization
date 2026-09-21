from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / "actor_only_critic_rarl_subsection3.py"
PREV_PATH = SCRIPT_DIR / "survey_standard_rarl_skew_geometry.py"

base_spec = importlib.util.spec_from_file_location("actor_only_critic_rarl_subsection3", BASE_PATH)
base = importlib.util.module_from_spec(base_spec)
sys.modules[base_spec.name] = base
base_spec.loader.exec_module(base)

prev_spec = importlib.util.spec_from_file_location("survey_standard_rarl_skew_geometry", PREV_PATH)
prev = importlib.util.module_from_spec(prev_spec)
sys.modules[prev_spec.name] = prev
prev_spec.loader.exec_module(prev)


RESULT_ROOT = BASE_PATH.resolve().parents[2] / "results" / "survey_standard_rarl_smooth_skew_reaudit"
PLOT_ROOT = RESULT_ROOT / "plots"

DEVICE = base.DEVICE
DTYPE = base.DTYPE
EPS = base.EPS
SEED = 0
GAMMA = 0.99

ACTOR_HIDDEN = (64, 64)
CRITIC_HIDDEN = (128, 128)
CRITIC_LR = 1e-3
CRITIC_TRAIN_STEPS = 8000
BATCH_SIZE = 256
POLYAK_TAU = 0.005
CRITIC_AUDIT_INTERVAL = 1000
MC_HORIZON = 100
MC_SNAPSHOTS = 12
DIAG_BATCH_SIZE = 512
FD_PROBES = 16
OUTPUT_PROBES = 12
ACTOR_LR_GRID = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
SHORT_ACTOR_ITERS = 80
LAMBDA_F = 0.01
LAMBDA_P = 1.0
TAU = 0.03
GAP_INNER_STEPS = 2
LOCAL_GAP_RADIUS = 0.1
BR_INNER_STEPS = 20
BR_LR = 1e-4
LOCAL_BR_RADIUS = 0.25


@dataclass(frozen=True)
class CandidateSpec:
    env_name: str
    wrapper_type: str
    strength: float
    adv_dim: int
    warmup_steps: int


CANDIDATES = [
    CandidateSpec("Ant-v5", "mujoco_external_force_xz", 2.0, 2, 50000),
    CandidateSpec("HalfCheetah-v5", "action_disturbance_direct", 0.3, 6, 30000),
    CandidateSpec("HalfCheetah-v5", "action_disturbance_direct", 0.05, 6, 30000),
]

CRITIC_ACTIVATIONS = ["tanh", "softplus"]


def ensure_dirs() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    base.write_csv(path, rows)


def write_text(path: Path, text: str) -> None:
    base.write_text(path, text)


def finite(value: float) -> bool:
    return math.isfinite(float(value))


def corr_or_nan(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return math.nan
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if np.std(xx) < 1e-12 or np.std(yy) < 1e-12:
        return math.nan
    return float(np.corrcoef(xx, yy)[0, 1])


def rank_corr_or_nan(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return math.nan
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if np.std(xx) < 1e-12 or np.std(yy) < 1e-12:
        return math.nan
    xr = xx.argsort().argsort().astype(np.float64)
    yr = yy.argsort().argsort().astype(np.float64)
    return corr_or_nan(xr.tolist(), yr.tolist())


def mse_or_nan(x: list[float], y: list[float]) -> float:
    if len(x) == 0 or len(y) == 0:
        return math.nan
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    return float(np.mean((xx - yy) ** 2))


def rank_norm(values: dict[str, float], higher_better: bool = True) -> dict[str, float]:
    finite_items = [(k, v) for k, v in values.items() if finite(v)]
    if not finite_items:
        return {k: 0.0 for k in values}
    ordered = sorted(finite_items, key=lambda item: item[1], reverse=higher_better)
    n = max(len(ordered) - 1, 1)
    out = {k: 0.0 for k in values}
    for idx, (key, _) in enumerate(ordered):
        out[key] = 1.0 - (idx / n)
    return out


class SmoothCritic(nn.Module):
    def __init__(self, obs_dim: int, u_dim: int, w_dim: int, activation: str) -> None:
        super().__init__()
        self.activation = activation
        input_dim = obs_dim + u_dim + w_dim
        self.fc1 = nn.Linear(input_dim, CRITIC_HIDDEN[0])
        self.fc2 = nn.Linear(CRITIC_HIDDEN[0], CRITIC_HIDDEN[1])
        self.fc3 = nn.Linear(CRITIC_HIDDEN[1], 1)

    def act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "tanh":
            return torch.tanh(x)
        if self.activation == "softplus":
            return F.softplus(x)
        raise ValueError(self.activation)

    def forward(self, obs: torch.Tensor, u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, u, w], dim=-1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, u_dim: int, w_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.u = np.zeros((capacity, u_dim), dtype=np.float32)
        self.w = np.zeros((capacity, w_dim), dtype=np.float32)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs: np.ndarray, u: np.ndarray, w: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        idx = self.ptr
        self.obs[idx] = obs
        self.u[idx] = u
        self.w[idx] = w
        self.reward[idx] = reward
        self.next_obs[idx] = next_obs
        self.done[idx] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        idx = rng.integers(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx], dtype=DTYPE, device=DEVICE),
            "u": torch.as_tensor(self.u[idx], dtype=DTYPE, device=DEVICE),
            "w": torch.as_tensor(self.w[idx], dtype=DTYPE, device=DEVICE),
            "reward": torch.as_tensor(self.reward[idx], dtype=DTYPE, device=DEVICE),
            "next_obs": torch.as_tensor(self.next_obs[idx], dtype=DTYPE, device=DEVICE),
            "done": torch.as_tensor(self.done[idx], dtype=DTYPE, device=DEVICE),
        }

    def fixed_state_batch(self, batch_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        idx = rng.integers(0, self.size, size=min(batch_size, self.size))
        return {"obs": torch.as_tensor(self.obs[idx], dtype=DTYPE, device=DEVICE)}


class CandidateContext:
    def __init__(self, spec: CandidateSpec) -> None:
        self.spec = spec
        self.env = base.gym.make(spec.env_name)
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        self.obs_dim = int(np.prod(obs_space.shape))
        self.action_dim = int(np.prod(act_space.shape))
        body_id, body_name, body_names = prev.choose_main_body(self.env)
        self.info = prev.EnvInfo(
            env_name=spec.env_name,
            available=True,
            backend_type="mujoco" if hasattr(self.env.unwrapped, "model") else "gymnasium",
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            action_low=np.asarray(act_space.low, dtype=np.float32),
            action_high=np.asarray(act_space.high, dtype=np.float32),
            max_episode_steps=int(getattr(self.env.spec, "max_episode_steps", 1000)),
            has_mujoco=hasattr(self.env.unwrapped, "model"),
            has_xfrc_applied=bool(hasattr(getattr(self.env.unwrapped, "data", object()), "xfrc_applied")),
            body_names=body_names,
            notes="",
            main_body_id=body_id,
            main_body_name=body_name,
        )
        strength_name = ("alpha" if "action" in spec.wrapper_type else "force") + f"_{spec.strength}"
        self.cfg = prev.WrapperConfig(spec.wrapper_type, spec.strength, strength_name, spec.adv_dim, body_id, body_name)
        self.wrapper = prev.SurveyWrapper(self.info, self.cfg)
        self.theta_layout = base.FlatMLP(self.obs_dim, ACTOR_HIDDEN, self.action_dim)
        self.phi_layout = base.FlatMLP(self.obs_dim, ACTOR_HIDDEN, spec.adv_dim)
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(SEED)
        self.theta0 = self.theta_layout.init_flat(gen, final_scale=0.05).detach().clone()
        gen2 = torch.Generator(device=DEVICE)
        gen2.manual_seed(SEED + 1)
        self.phi0 = self.phi_layout.init_flat(gen2, final_scale=0.05).detach().clone()
        self.replay = ReplayBuffer(max(spec.warmup_steps + 1000, 60000), self.obs_dim, self.action_dim, spec.adv_dim)
        self.rng = np.random.default_rng(SEED)
        self.warmup_stats: dict[str, float] = {}
        self.snapshots: list[dict[str, np.ndarray]] = []
        self.diag_batch: dict[str, torch.Tensor] | None = None
        self.collect_replay()
        self.snapshots = self.collect_snapshots(MC_SNAPSHOTS)
        self.diag_batch = self.replay.fixed_state_batch(DIAG_BATCH_SIZE, np.random.default_rng(SEED + 777))
        self.env.close()

    def actor_protagonist(self, theta: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        raw = self.theta_layout.forward(theta, obs)
        return self.wrapper.scale_protagonist(raw)

    def actor_adversary(self, phi: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        raw = self.phi_layout.forward(phi, obs)
        return self.wrapper.scale_adversary(raw)

    def step_env(self, env, theta: torch.Tensor, phi: torch.Tensor, obs: np.ndarray, noise_u: float, noise_w: float) -> tuple[np.ndarray, float, bool, float, float, bool, np.ndarray, np.ndarray]:
        obs_t = torch.as_tensor(obs, dtype=DTYPE, device=DEVICE).unsqueeze(0)
        u = self.actor_protagonist(theta, obs_t).squeeze(0)
        w = self.actor_adversary(phi, obs_t).squeeze(0)
        u = torch.clamp(u + (noise_u * torch.randn_like(u)), self.wrapper.action_low, self.wrapper.action_high)
        w = torch.clamp(w + (noise_w * torch.randn_like(w)), -torch.ones_like(w), torch.ones_like(w))
        force_norm = 0.0
        clip_frac = 0.0
        step_error = False
        try:
            if self.cfg.wrapper_type.startswith("action_disturbance"):
                a_env, clip_frac = self.wrapper.action_step(u, w)
                next_obs, reward, terminated, truncated, _ = env.step(a_env.detach().cpu().numpy().astype(np.float32))
            else:
                force_norm, _ = self.wrapper.apply_force(env, w)
                next_obs, reward, terminated, truncated, _ = env.step(u.detach().cpu().numpy().astype(np.float32))
                self.wrapper.clear_force(env)
            done = terminated or truncated
            return np.asarray(next_obs, dtype=np.float32), float(reward), done, clip_frac, force_norm, step_error, u.detach().cpu().numpy().astype(np.float32), w.detach().cpu().numpy().astype(np.float32)
        except Exception:
            self.wrapper.clear_force(env)
            return np.zeros((self.obs_dim,), dtype=np.float32), 0.0, True, clip_frac, force_norm, True, np.zeros((self.action_dim,), dtype=np.float32), np.zeros((self.cfg.adv_dim,), dtype=np.float32)

    def collect_replay(self) -> None:
        env = base.gym.make(self.spec.env_name)
        obs, _ = env.reset(seed=SEED + 123)
        u_vals, w_vals, rw_vals, clips = [], [], [], []
        state_finite = True
        step_error = False
        for _ in range(self.spec.warmup_steps):
            next_obs, reward, done, clip_frac, force_norm, err, u_np, w_np = self.step_env(env, self.theta0, self.phi0, obs, 0.1, 0.2)
            self.replay.add(np.asarray(obs, dtype=np.float32), u_np, w_np, reward, next_obs, done)
            u_vals.append(u_np)
            w_vals.append(w_np)
            if self.cfg.wrapper_type == "action_disturbance_rotated":
                rw_vals.append((self.wrapper.r_dyn.cpu().numpy() @ w_np))
            elif self.cfg.wrapper_type == "action_disturbance_direct":
                rw_vals.append(w_np)
            else:
                rw_vals.append(w_np)
            clips.append(clip_frac)
            state_finite = state_finite and bool(np.isfinite(next_obs).all())
            step_error = step_error or err
            obs = next_obs
            if done:
                obs, _ = env.reset()
        env.close()
        self.warmup_stats = {
            "std_u": float(np.std(np.asarray(u_vals, dtype=np.float64))) if u_vals else 0.0,
            "std_w": float(np.std(np.asarray(w_vals, dtype=np.float64))) if w_vals else 0.0,
            "std_Rw": float(np.std(np.asarray(rw_vals, dtype=np.float64))) if rw_vals else 0.0,
            "action_clip_fraction": float(np.mean(clips)) if clips else 0.0,
            "state_finite_flag": int(state_finite),
            "step_error_flag": int(step_error),
        }

    def can_snapshot(self, env) -> bool:
        uw = env.unwrapped
        return hasattr(uw, "set_state") and hasattr(uw, "data") and hasattr(uw.data, "qpos") and hasattr(uw.data, "qvel")

    def collect_snapshots(self, count: int) -> list[dict[str, np.ndarray]]:
        env = base.gym.make(self.spec.env_name)
        if not self.can_snapshot(env):
            env.close()
            return []
        obs, _ = env.reset(seed=SEED + 999)
        snaps: list[dict[str, np.ndarray]] = []
        for _ in range(max(count * 6, count)):
            next_obs, _, done, _, _, err, _, _ = self.step_env(env, self.theta0, self.phi0, obs, 0.0, 0.0)
            if not err and len(snaps) < count:
                snaps.append({"obs": np.asarray(obs, dtype=np.float32).copy(), "qpos": env.unwrapped.data.qpos.copy(), "qvel": env.unwrapped.data.qvel.copy()})
            obs = next_obs
            if done:
                obs, _ = env.reset()
            if len(snaps) >= count:
                break
        env.close()
        return snaps


class SmoothAuditRun:
    def __init__(self, context: CandidateContext, activation: str) -> None:
        self.ctx = context
        self.activation = activation
        self.theta = context.theta0.detach().clone()
        self.phi = context.phi0.detach().clone()
        self.theta_target = self.theta.detach().clone()
        self.phi_target = self.phi.detach().clone()
        self.q = SmoothCritic(context.obs_dim, context.action_dim, context.cfg.adv_dim, activation).to(DEVICE)
        self.q_target = SmoothCritic(context.obs_dim, context.action_dim, context.cfg.adv_dim, activation).to(DEVICE)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=CRITIC_LR)
        self.rng = np.random.default_rng(SEED + 202)
        self.metric_refs: dict[str, float] = {}

    def actor_protagonist(self, theta: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return self.ctx.actor_protagonist(theta, obs)

    def actor_adversary(self, phi: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return self.ctx.actor_adversary(phi, obs)

    def polyak(self) -> None:
        self.theta_target = ((1.0 - POLYAK_TAU) * self.theta_target) + (POLYAK_TAU * self.theta)
        self.phi_target = ((1.0 - POLYAK_TAU) * self.phi_target) + (POLYAK_TAU * self.phi)
        with torch.no_grad():
            for t, o in zip(self.q_target.parameters(), self.q.parameters()):
                t.data.mul_(1.0 - POLYAK_TAU).add_(POLYAK_TAU * o.data)

    def critic_update(self) -> dict[str, float]:
        batch = self.ctx.replay.sample(BATCH_SIZE, self.rng)
        with torch.no_grad():
            u_next = self.actor_protagonist(self.theta_target, batch["next_obs"])
            w_next = self.actor_adversary(self.phi_target, batch["next_obs"])
            y = batch["reward"] + (GAMMA * (1.0 - batch["done"]) * self.q_target(batch["next_obs"], u_next, w_next))
        pred = self.q(batch["obs"], batch["u"], batch["w"])
        loss = torch.mean((pred - y) ** 2)
        self.q_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.q_opt.step()
        self.polyak()
        return {
            "critic_loss": float(loss.detach().item()),
            "Q_mean": float(pred.detach().mean().item()),
            "Q_std": float(pred.detach().std(unbiased=False).item()),
            "Q_abs_mean": float(pred.detach().abs().mean().item()),
            "target_mean": float(y.detach().mean().item()),
            "target_std": float(y.detach().std(unbiased=False).item()),
        }

    def mc_quality(self) -> dict[str, float]:
        if not self.ctx.snapshots:
            return {"corr_Q_MC": math.nan, "mse_Q_MC": math.nan, "rank_corr_Q_MC": math.nan}
        env = base.gym.make(self.ctx.spec.env_name)
        q_vals, mc_vals = [], []
        for snap in self.ctx.snapshots:
            try:
                env.reset(seed=SEED)
                env.unwrapped.set_state(snap["qpos"], snap["qvel"])
            except Exception:
                continue
            obs_t = torch.as_tensor(np.asarray(snap["obs"], dtype=np.float32), dtype=DTYPE, device=DEVICE).unsqueeze(0)
            u = self.actor_protagonist(self.theta, obs_t)
            w = self.actor_adversary(self.phi, obs_t)
            q_vals.append(float(self.q(obs_t, u, w).detach().item()))
            obs_np = np.asarray(snap["obs"], dtype=np.float32)
            disc = 1.0
            total = 0.0
            for _ in range(MC_HORIZON):
                next_obs, reward, done, _, _, err, _, _ = self.ctx.step_env(env, self.theta, self.phi, obs_np, 0.0, 0.0)
                if err:
                    break
                total += disc * reward
                disc *= GAMMA
                obs_np = next_obs
                if done:
                    break
            mc_vals.append(total)
        env.close()
        return {"corr_Q_MC": corr_or_nan(q_vals, mc_vals), "mse_Q_MC": mse_or_nan(q_vals, mc_vals), "rank_corr_Q_MC": rank_corr_or_nan(q_vals, mc_vals)}

    def actor_objective(self, z: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        theta = z[: self.ctx.theta_layout.num_params]
        phi = z[self.ctx.theta_layout.num_params :]
        u = self.actor_protagonist(theta, states)
        w = self.actor_adversary(phi, states)
        return self.q(states, u, w).mean()

    def actor_field(self, z: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        z_req = z if z.requires_grad else z.detach().clone().requires_grad_(True)
        j = self.actor_objective(z_req, states)
        grad = torch.autograd.grad(j, z_req, create_graph=True)[0]
        out = torch.zeros_like(z_req)
        tdim = self.ctx.theta_layout.num_params
        out[:tdim] = -grad[:tdim]
        out[tdim:] = +grad[tdim:]
        return out

    def local_gap(self, z: torch.Tensor, states: torch.Tensor, protagonist: bool, actor_lr: float, with_prox: bool) -> torch.Tensor:
        base_z = z.detach().clone()
        current = base_z.clone()
        tdim = self.ctx.theta_layout.num_params
        sl = slice(0, tdim) if protagonist else slice(tdim, current.numel())
        initial = base_z[sl].clone()
        j_start = self.actor_objective(base_z, states)
        inner_lr = 0.1 * actor_lr
        for _ in range(GAP_INNER_STEPS):
            cur = current.detach().clone().requires_grad_(True)
            j = self.actor_objective(cur, states)
            grad = torch.autograd.grad(j, cur)[0][sl]
            step = grad if protagonist else -grad
            if with_prox:
                step = step - ((cur[sl] - initial) / max(TAU, EPS))
            next_params = cur[sl] + (inner_lr * step)
            delta = next_params - initial
            norm = torch.linalg.norm(delta)
            if norm > LOCAL_GAP_RADIUS:
                next_params = initial + (delta * (LOCAL_GAP_RADIUS / (norm + EPS)))
            current = cur.detach().clone()
            current[sl] = next_params.detach()
        j_end = self.actor_objective(current, states)
        return torch.relu(j_end - j_start) if protagonist else torch.relu(j_start - j_end)

    def exploitability(self, z: torch.Tensor, states: torch.Tensor, actor_lr: float) -> tuple[float, float]:
        p = float(self.local_gap(z, states, True, actor_lr, False).detach().item())
        a = float(self.local_gap(z, states, False, actor_lr, False).detach().item())
        return p, a

    def diagnostic_metrics(self, z: torch.Tensor, states: torch.Tensor, actor_lr: float) -> dict[str, float]:
        field = self.actor_field(z, states).detach()
        energy = 0.5 * float(torch.dot(field, field).item())
        p_gap = float(self.local_gap(z, states, True, actor_lr, True).detach().item())
        a_gap = float(self.local_gap(z, states, False, actor_lr, True).detach().item())
        p_tau = p_gap + a_gap
        if "field0" not in self.metric_refs:
            self.metric_refs["field0"] = energy
            self.metric_refs["ptau0"] = max(p_tau, EPS)
        field_term = energy / (self.metric_refs["field0"] + EPS)
        p_tau_term = p_tau / (self.metric_refs["ptau0"] + EPS)
        ex_p, ex_a = self.exploitability(z, states, actor_lr)
        return {
            "V_std": (LAMBDA_F * field_term) + (LAMBDA_P * p_tau_term),
            "field_term": field_term,
            "P_tau_std": p_tau_term,
            "raw_P_tau_std": p_tau,
            "field_norm": float(torch.linalg.norm(field).item()),
            "approx_exploitability_std": ex_p + ex_a,
        }

    def fd_skew_metrics(self, z: torch.Tensor, states: torch.Tensor) -> dict[str, float]:
        field0 = self.actor_field(z, states).detach()
        d = z.numel()
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(20260617)
        skew_vals, sym_vals = [], []
        for _ in range(FD_PROBES):
            u = torch.randn(d, generator=gen, dtype=DTYPE, device=DEVICE)
            v = torch.randn(d, generator=gen, dtype=DTYPE, device=DEVICE)
            u = u / (torch.linalg.norm(u) + EPS)
            v = v / (torch.linalg.norm(v) + EPS)
            eps_u = 1e-4 / (float(torch.linalg.norm(u).item()) + 1e-12)
            eps_v = 1e-4 / (float(torch.linalg.norm(v).item()) + 1e-12)
            Ju = (self.actor_field(z + (eps_u * u), states).detach() - field0) / eps_u
            Jv = (self.actor_field(z + (eps_v * v), states).detach() - field0) / eps_v
            uv = abs(float(torch.dot(u, Jv).item() - torch.dot(v, Ju).item()))
            sv = abs(float(torch.dot(u, Jv).item() + torch.dot(v, Ju).item()))
            skew_vals.append(uv)
            sym_vals.append(sv)
        mean_skew = float(np.mean(skew_vals))
        mean_sym = float(np.mean(sym_vals))
        return {
            "fd_skew_ratio": mean_skew / (mean_sym + 1e-12),
            "mean_skew_bilinear": mean_skew,
            "mean_sym_bilinear": mean_sym,
            "median_skew_bilinear": float(np.median(skew_vals)),
            "median_sym_bilinear": float(np.median(sym_vals)),
        }

    def g_and_coupling_metrics(self, z: torch.Tensor, states: torch.Tensor) -> dict[str, float]:
        field = self.actor_field(z, states).detach()
        f_norm = float(torch.linalg.norm(field).item())
        eps_f = 1e-4 / (f_norm + 1e-12)
        g = (self.actor_field(z + (eps_f * field), states).detach() - field) / eps_f
        g_norm = float(torch.linalg.norm(g).item())
        cos_fg = float(torch.dot(field, g).item() / ((f_norm * g_norm) + EPS))
        tdim = self.ctx.theta_layout.num_params
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(20260618)
        v_theta = torch.randn(tdim, generator=gen, dtype=DTYPE, device=DEVICE)
        v_phi = torch.randn(z.numel() - tdim, generator=gen, dtype=DTYPE, device=DEVICE)
        v_theta = v_theta / (torch.linalg.norm(v_theta) + EPS)
        v_phi = v_phi / (torch.linalg.norm(v_phi) + EPS)
        eps_t = 1e-4
        eps_p = 1e-4
        base = field
        z_t = z.detach().clone()
        z_t[:tdim] = z_t[:tdim] + (eps_t * v_theta)
        z_p = z.detach().clone()
        z_p[tdim:] = z_p[tdim:] + (eps_p * v_phi)
        f_t = self.actor_field(z_t, states).detach()
        f_p = self.actor_field(z_p, states).detach()
        cross_phi_to_theta = float(torch.linalg.norm(f_p[:tdim] - base[:tdim]).item()) / eps_p
        cross_theta_to_phi = float(torch.linalg.norm(f_t[tdim:] - base[tdim:]).item()) / eps_t
        same_theta = float(torch.linalg.norm(f_t[:tdim] - base[:tdim]).item()) / eps_t
        same_phi = float(torch.linalg.norm(f_p[tdim:] - base[tdim:]).item()) / eps_p
        cross = 0.5 * (cross_phi_to_theta + cross_theta_to_phi)
        same = 0.5 * (same_theta + same_phi)
        return {
            "field_norm": f_norm,
            "G_norm": g_norm,
            "G_over_F": g_norm / (f_norm + EPS),
            "cos_F_G": cos_fg,
            "non_collinearity": math.sqrt(max(0.0, 1.0 - min(1.0, cos_fg * cos_fg))),
            "cross_phi_to_theta": cross_phi_to_theta,
            "cross_theta_to_phi": cross_theta_to_phi,
            "same_theta": same_theta,
            "same_phi": same_phi,
            "cross_player_coupling_proxy": cross,
            "same_player_proxy": same,
            "cross_to_same_ratio": cross / (same + 1e-12),
        }

    def output_geometry(self, states: torch.Tensor) -> dict[str, float]:
        states = states[: min(16, states.shape[0])]
        u0 = self.actor_protagonist(self.theta, states).detach()
        w0 = self.actor_adversary(self.phi, states).detach()
        y0 = torch.cat([u0.reshape(-1), w0.reshape(-1)])
        nu = u0.numel()
        nw = w0.numel()

        def f_out(y: torch.Tensor) -> torch.Tensor:
            u = y[:nu].reshape_as(u0).detach().clone().requires_grad_(True)
            w = y[nu:].reshape_as(w0).detach().clone().requires_grad_(True)
            q = self.q(states, u, w).mean()
            gu, gw = torch.autograd.grad(q, (u, w), create_graph=True)
            return torch.cat([-gu.reshape(-1), +gw.reshape(-1)])

        f0 = f_out(y0).detach()
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(20260619)
        skew_vals, sym_vals = [], []
        for _ in range(OUTPUT_PROBES):
            p = torch.randn(y0.shape, generator=gen, dtype=DTYPE, device=DEVICE)
            q = torch.randn(y0.shape, generator=gen, dtype=DTYPE, device=DEVICE)
            p = p / (torch.linalg.norm(p) + EPS)
            q = q / (torch.linalg.norm(q) + EPS)
            eps_p = 1e-4 / (float(torch.linalg.norm(p).item()) + 1e-12)
            eps_q = 1e-4 / (float(torch.linalg.norm(q).item()) + 1e-12)
            Jp = (f_out(y0 + (eps_p * p)).detach() - f0) / eps_p
            Jq = (f_out(y0 + (eps_q * q)).detach() - f0) / eps_q
            skew_vals.append(abs(float(torch.dot(p, Jq).item() - torch.dot(q, Jp).item())))
            sym_vals.append(abs(float(torch.dot(p, Jq).item() + torch.dot(q, Jp).item())))
        mean_skew = float(np.mean(skew_vals))
        mean_sym = float(np.mean(sym_vals))

        try:
            u = u0.detach().clone().requires_grad_(True)
            w = w0.detach().clone().requires_grad_(True)
            qval = self.q(states, u, w).mean()
            gu, gw = torch.autograd.grad(qval, (u, w), create_graph=True)
            u_flat = u.reshape(-1)
            w_flat = w.reshape(-1)
            gu_flat = gu.reshape(-1)
            gw_flat = gw.reshape(-1)
            du = u_flat.numel()
            dw = w_flat.numel()
            h_uu = torch.zeros((du, du), dtype=DTYPE)
            h_uw = torch.zeros((du, dw), dtype=DTYPE)
            h_wu = torch.zeros((dw, du), dtype=DTYPE)
            h_ww = torch.zeros((dw, dw), dtype=DTYPE)
            for i in range(du):
                grads = torch.autograd.grad(gu_flat[i], (u, w), retain_graph=True)
                h_uu[i] = grads[0].reshape(-1).detach().cpu()
                h_uw[i] = grads[1].reshape(-1).detach().cpu()
            for i in range(dw):
                grads = torch.autograd.grad(gw_flat[i], (u, w), retain_graph=True)
                h_wu[i] = grads[0].reshape(-1).detach().cpu()
                h_ww[i] = grads[1].reshape(-1).detach().cpu()
            j_out = torch.zeros((du + dw, du + dw), dtype=DTYPE)
            j_out[:du, :du] = -h_uu
            j_out[:du, du:] = -h_uw
            j_out[du:, :du] = +h_wu
            j_out[du:, du:] = +h_ww
            a = 0.5 * (j_out - j_out.T)
            s = 0.5 * (j_out + j_out.T)
            eigvals = np.linalg.eigvals(j_out.numpy().astype(np.float64))
            imag = np.abs(np.imag(eigvals))
            output_rotation_ratio = float(torch.linalg.norm(a).item() / (torch.linalg.norm(s).item() + EPS))
            output_num_complex_eigs = int(np.sum(imag > 1e-8))
            output_max_imag_eig = float(np.max(imag)) if imag.size else 0.0
            output_cross_to_diag_ratio = float(torch.linalg.norm(h_uw).item() / (torch.linalg.norm(h_uu).item() + torch.linalg.norm(h_ww).item() + EPS))
        except Exception:
            output_rotation_ratio = math.nan
            output_num_complex_eigs = 0
            output_max_imag_eig = math.nan
            output_cross_to_diag_ratio = math.nan

        return {
            "output_fd_skew_ratio": mean_skew / (mean_sym + 1e-12),
            "output_fd_mean_skew": mean_skew,
            "output_fd_mean_sym": mean_sym,
            "output_rotation_ratio": output_rotation_ratio,
            "output_num_complex_eigs": output_num_complex_eigs,
            "output_max_imag_eig": output_max_imag_eig,
            "output_cross_to_diag_ratio": output_cross_to_diag_ratio,
        }

    def evaluate_actor(self, theta: torch.Tensor, phi: torch.Tensor | None, episodes: int) -> dict[str, float]:
        env = base.gym.make(self.ctx.spec.env_name)
        returns = []
        clip_fracs = []
        for ep in range(episodes):
            obs, _ = env.reset(seed=SEED + 9100 + ep)
            disc = 1.0
            total = 0.0
            clips = []
            while True:
                obs_t = torch.as_tensor(obs, dtype=DTYPE, device=DEVICE).unsqueeze(0)
                u = self.actor_protagonist(theta, obs_t).squeeze(0)
                if phi is None:
                    a_env = torch.clamp(u, self.ctx.wrapper.action_low, self.ctx.wrapper.action_high)
                    clip_frac = float(((u - a_env).abs() > 1e-12).float().mean().item())
                    w = torch.zeros((self.ctx.cfg.adv_dim,), dtype=DTYPE, device=DEVICE)
                    force_clear = False
                else:
                    w = self.actor_adversary(phi, obs_t).squeeze(0)
                    if self.ctx.cfg.wrapper_type.startswith("action_disturbance"):
                        a_env, clip_frac = self.ctx.wrapper.action_step(u, w)
                        force_clear = False
                    else:
                        self.ctx.wrapper.apply_force(env, w)
                        a_env = torch.clamp(u, self.ctx.wrapper.action_low, self.ctx.wrapper.action_high)
                        clip_frac = 0.0
                        force_clear = True
                try:
                    next_obs, reward, terminated, truncated, _ = env.step(a_env.detach().cpu().numpy().astype(np.float32))
                finally:
                    if phi is not None and force_clear:
                        self.ctx.wrapper.clear_force(env)
                total += disc * float(reward)
                disc *= GAMMA
                clips.append(clip_frac)
                obs = np.asarray(next_obs, dtype=np.float32)
                if terminated or truncated:
                    break
            returns.append(total)
            clip_fracs.append(float(np.mean(clips)) if clips else 0.0)
        env.close()
        return {"return": float(np.mean(returns)), "clip": float(np.mean(clip_fracs))}

    def robust_br(self, theta: torch.Tensor, states: torch.Tensor) -> tuple[torch.Tensor, bool]:
        base_phi = self.phi.detach().clone()
        current = base_phi.clone()
        for _ in range(BR_INNER_STEPS):
            cur = current.detach().clone().requires_grad_(True)
            z = torch.cat([theta.detach(), cur])
            objective = self.actor_objective(z, states)
            grad = torch.autograd.grad(objective, cur)[0]
            next_phi = cur - (BR_LR * grad)
            delta = next_phi - base_phi
            norm = torch.linalg.norm(delta)
            if norm > LOCAL_BR_RADIUS:
                next_phi = base_phi + (delta * (LOCAL_BR_RADIUS / (norm + EPS)))
            current = next_phi.detach()
        return current, True


def short_update(run: SmoothAuditRun, z: torch.Tensor, states: torch.Tensor, method: str, actor_lr: float) -> tuple[torch.Tensor, dict[str, float]]:
    field = run.actor_field(z, states).detach()
    if method == "sgd":
        delta = -actor_lr * field
        return (z + delta).detach(), {"update_norm": float(torch.linalg.norm(delta).item())}
    if method == "egm":
        z_half = z - (actor_lr * field)
        field_half = run.actor_field(z_half, states).detach()
        delta = -actor_lr * field_half
        return (z + delta).detach(), {"update_norm": float(torch.linalg.norm(delta).item())}
    if method == "ppm":
        current = z.detach().clone()
        for _ in range(5):
            field_inner = run.actor_field(current, states).detach()
            current = z - (actor_lr * field_inner)
        delta = current - z
        return current.detach(), {"update_norm": float(torch.linalg.norm(delta).item())}
    raise ValueError(method)


def short_run(run_src: SmoothAuditRun, method: str, actor_lr: float) -> tuple[list[dict[str, Any]], dict[str, float]]:
    run = SmoothAuditRun(run_src.ctx, run_src.activation)
    run.theta = run_src.theta.detach().clone()
    run.phi = run_src.phi.detach().clone()
    run.theta_target = run_src.theta_target.detach().clone()
    run.phi_target = run_src.phi_target.detach().clone()
    run.q.load_state_dict(run_src.q.state_dict())
    run.q_target.load_state_dict(run_src.q_target.state_dict())
    run.metric_refs = {}
    states = run.ctx.diag_batch["obs"]
    curves: list[dict[str, Any]] = []
    clean_eval = run.evaluate_actor(run.theta, None, 3)
    adv_eval = run.evaluate_actor(run.theta, run.phi, 3)
    br_phi, br_valid = run.robust_br(run.theta, states[:128])
    br_eval = run.evaluate_actor(run.theta, br_phi, 3)
    for it in range(SHORT_ACTOR_ITERS + 1):
        z = torch.cat([run.theta, run.phi]).detach().clone()
        diag = run.diagnostic_metrics(z, states, actor_lr)
        if it % 10 == 0 or it == SHORT_ACTOR_ITERS:
            clean_eval = run.evaluate_actor(run.theta, None, 3)
            adv_eval = run.evaluate_actor(run.theta, run.phi, 3)
            br_phi, br_valid = run.robust_br(run.theta, states[:128])
            br_eval = run.evaluate_actor(run.theta, br_phi, 3)
        row = {
            "iteration": it,
            "method": method,
            "actor_lr": actor_lr,
            **diag,
            "clean_task_return": clean_eval["return"],
            "current_adv_task_return": adv_eval["return"],
            "robust_br_task_return": br_eval["return"],
            "robust_degradation": clean_eval["return"] - br_eval["return"],
            "valid_flag": int(all(finite(diag[k]) for k in ["V_std", "field_norm", "P_tau_std"]) and br_valid and clean_eval["clip"] <= 0.2 and adv_eval["clip"] <= 0.2),
        }
        curves.append(row)
        if it < SHORT_ACTOR_ITERS:
            next_z, meta = short_update(run, z, states, method, actor_lr)
            run.theta = next_z[: run.ctx.theta_layout.num_params].detach().clone()
            run.phi = next_z[run.ctx.theta_layout.num_params :].detach().clone()
            curves[-1].update(meta)
    vals_v = [r["V_std"] for r in curves]
    vals_p = [r["P_tau_std"] for r in curves]
    vals_f = [r["field_norm"] for r in curves]
    summary = {
        "valid_flag": int(all(r["valid_flag"] == 1 for r in curves)),
        "curve_normal_flag": int(curves[-1]["V_std"] <= curves[0]["V_std"] + 1e-8 and curves[-1]["P_tau_std"] <= curves[0]["P_tau_std"] + 1e-8 and base.spike_ratio(vals_v) <= 5.0 and base.spike_ratio(vals_p) <= 5.0),
        "V_AUC": float(sum(vals_v)),
        "P_tau_AUC": float(sum(vals_p)),
        "field_norm_AUC": float(sum(vals_f)),
        "clean_task_return_final": curves[-1]["clean_task_return"],
        "current_adv_task_return_final": curves[-1]["current_adv_task_return"],
        "robust_br_task_return_final": curves[-1]["robust_br_task_return"],
        "robust_degradation_final": curves[-1]["robust_degradation"],
    }
    return curves, summary


def label_row(row: dict[str, Any]) -> str:
    if int(row.get("wrapper_unstable", 0)) == 1:
        return "WRAPPER_UNSTABLE"
    if int(row.get("critic_unreliable", 0)) == 1:
        return "CRITIC_UNRELIABLE"
    fd = float(row.get("fd_skew_ratio", math.nan))
    ofd = float(row.get("output_fd_skew_ratio", math.nan))
    cross = float(row.get("cross_to_same_ratio", math.nan))
    adv = float(row.get("short_baseline_advantage", math.nan))
    if ((finite(fd) and fd >= 1e-2) or (finite(ofd) and ofd >= 1e-2)) and cross >= 0.1 and finite(adv) and adv >= 1.1:
        return "CONFIRMED_SKEW_CANDIDATE"
    if ((finite(fd) and fd >= 3e-3) or (finite(ofd) and ofd >= 3e-3)) and cross >= 0.05 and ((finite(adv) and adv >= 1.03) or not finite(adv)):
        return "MODERATE_SKEW_CANDIDATE"
    if cross < 0.05 and ((not finite(fd)) or fd < 1e-3) and ((not finite(ofd)) or ofd < 1e-3):
        return "POTENTIAL_LIKE"
    return "WEAK_GEOMETRY"


def plot_rank(rows: list[dict[str, Any]], metric: str, title: str, filename: str) -> None:
    if plt is None:
        return
    finite_rows = [r for r in rows if finite(float(r.get(metric, math.nan)))]
    if not finite_rows:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, f"No finite values for {metric}", ha="center", va="center")
        ax.set_axis_off()
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(PLOT_ROOT / filename, dpi=180)
        plt.close(fig)
        return
    finite_rows = sorted(finite_rows, key=lambda r: float(r[metric]), reverse=True)
    labels = [f"{r['env_name']}|{r['wrapper_type']}|{r['critic_activation']}" for r in finite_rows]
    vals = [float(r[metric]) for r in finite_rows]
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.5)))
    ax.barh(range(len(labels)), vals, color="#1f77b4")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / filename, dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    base.seed_everything(SEED)
    contexts = [CandidateContext(spec) for spec in CANDIDATES]
    critic_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for ctx in contexts:
        for activation in CRITIC_ACTIVATIONS:
            run = SmoothAuditRun(ctx, activation)
            critic_hist: list[dict[str, float]] = []
            for step in range(CRITIC_TRAIN_STEPS):
                stats = run.critic_update()
                if step % CRITIC_AUDIT_INTERVAL == 0 or step == CRITIC_TRAIN_STEPS - 1:
                    quality = run.mc_quality()
                    critic_hist.append({**stats, **quality, "critic_step": step})
            final_critic = critic_hist[-1]
            critic_row = {
                "env_name": ctx.spec.env_name,
                "wrapper_type": ctx.spec.wrapper_type,
                "strength": ctx.spec.strength,
                "critic_activation": activation,
                "critic_loss_final": final_critic["critic_loss"],
                "critic_loss_mean_last_1000": float(np.mean([r["critic_loss"] for r in critic_hist[-min(len(critic_hist), 2):]])),
                "Q_mean": final_critic["Q_mean"],
                "Q_std": final_critic["Q_std"],
                "Q_abs_mean": final_critic["Q_abs_mean"],
                "target_mean": final_critic["target_mean"],
                "target_std": final_critic["target_std"],
                "corr_Q_MC": final_critic["corr_Q_MC"],
                "mse_Q_MC": final_critic["mse_Q_MC"],
                "rank_corr_Q_MC": final_critic["rank_corr_Q_MC"],
            }
            critic_rows.append(critic_row)

            critic_unreliable = (
                (not finite(critic_row["critic_loss_final"]))
                or (not finite(critic_row["Q_abs_mean"]))
                or (critic_row["Q_abs_mean"] > 1e6)
                or (finite(critic_row["corr_Q_MC"]) and critic_row["corr_Q_MC"] < 0.0)
            )

            z = torch.cat([run.theta, run.phi]).detach().clone()
            states = ctx.diag_batch["obs"]
            diag = run.diagnostic_metrics(z, states, 1e-5)
            fd_geom = run.fd_skew_metrics(z, states)
            g_geom = run.g_and_coupling_metrics(z, states)
            out_geom = run.output_geometry(states)

            short_summary = {
                "SGD_V_AUC": math.nan,
                "EGM_V_AUC": math.nan,
                "PPM_V_AUC": math.nan,
                "SGD_P_tau_AUC": math.nan,
                "EGM_P_tau_AUC": math.nan,
                "PPM_P_tau_AUC": math.nan,
                "SGD_field_norm_AUC": math.nan,
                "EGM_field_norm_AUC": math.nan,
                "PPM_field_norm_AUC": math.nan,
                "short_baseline_advantage": math.nan,
                "clean_task_return_final": math.nan,
                "current_adv_task_return_final": math.nan,
                "robust_br_task_return_final": math.nan,
                "robust_degradation_final": math.nan,
            }

            if (not critic_unreliable) and finite(g_geom["cross_to_same_ratio"]) and g_geom["cross_to_same_ratio"] >= 0.05:
                best_lr = None
                best_auc = None
                best_sgd_summary = None
                cached: dict[tuple[str, float], dict[str, float]] = {}
                for lr in ACTOR_LR_GRID:
                    _, sgd_summary = short_run(run, "sgd", lr)
                    cached[("sgd", lr)] = sgd_summary
                    if sgd_summary["valid_flag"] == 1 and sgd_summary["curve_normal_flag"] == 1:
                        if best_auc is None or sgd_summary["V_AUC"] < best_auc:
                            best_auc = sgd_summary["V_AUC"]
                            best_lr = lr
                            best_sgd_summary = sgd_summary
                if best_lr is not None and best_sgd_summary is not None:
                    _, egm_summary = short_run(run, "egm", best_lr)
                    _, ppm_summary = short_run(run, "ppm", best_lr)
                    short_summary = {
                        "SGD_V_AUC": best_sgd_summary["V_AUC"],
                        "EGM_V_AUC": egm_summary["V_AUC"],
                        "PPM_V_AUC": ppm_summary["V_AUC"],
                        "SGD_P_tau_AUC": best_sgd_summary["P_tau_AUC"],
                        "EGM_P_tau_AUC": egm_summary["P_tau_AUC"],
                        "PPM_P_tau_AUC": ppm_summary["P_tau_AUC"],
                        "SGD_field_norm_AUC": best_sgd_summary["field_norm_AUC"],
                        "EGM_field_norm_AUC": egm_summary["field_norm_AUC"],
                        "PPM_field_norm_AUC": ppm_summary["field_norm_AUC"],
                        "short_baseline_advantage": best_sgd_summary["V_AUC"] / (min(egm_summary["V_AUC"], ppm_summary["V_AUC"]) + EPS),
                        "clean_task_return_final": max(best_sgd_summary["clean_task_return_final"], egm_summary["clean_task_return_final"], ppm_summary["clean_task_return_final"]),
                        "current_adv_task_return_final": max(best_sgd_summary["current_adv_task_return_final"], egm_summary["current_adv_task_return_final"], ppm_summary["current_adv_task_return_final"]),
                        "robust_br_task_return_final": max(best_sgd_summary["robust_br_task_return_final"], egm_summary["robust_br_task_return_final"], ppm_summary["robust_br_task_return_final"]),
                        "robust_degradation_final": min(best_sgd_summary["robust_degradation_final"], egm_summary["robust_degradation_final"], ppm_summary["robust_degradation_final"]),
                    }

            row = {
                "env_name": ctx.spec.env_name,
                "wrapper_type": ctx.spec.wrapper_type,
                "strength": ctx.spec.strength,
                "critic_activation": activation,
                **critic_row,
                **diag,
                **fd_geom,
                **g_geom,
                **out_geom,
                **short_summary,
                "wrapper_unstable": int(ctx.warmup_stats["state_finite_flag"] == 0 or ctx.warmup_stats["step_error_flag"] == 1),
                "critic_unreliable": int(critic_unreliable),
            }
            summary_rows.append(row)

    key_score = {}
    rank_fd = rank_norm({f"{r['env_name']}|{r['wrapper_type']}|{r['strength']}|{r['critic_activation']}": float(r["fd_skew_ratio"]) for r in summary_rows})
    rank_ofd = rank_norm({f"{r['env_name']}|{r['wrapper_type']}|{r['strength']}|{r['critic_activation']}": float(r["output_fd_skew_ratio"]) for r in summary_rows})
    rank_cross = rank_norm({f"{r['env_name']}|{r['wrapper_type']}|{r['strength']}|{r['critic_activation']}": float(r["cross_to_same_ratio"]) for r in summary_rows})
    rank_adv = rank_norm({f"{r['env_name']}|{r['wrapper_type']}|{r['strength']}|{r['critic_activation']}": float(r["short_baseline_advantage"]) for r in summary_rows})
    rank_non = rank_norm({f"{r['env_name']}|{r['wrapper_type']}|{r['strength']}|{r['critic_activation']}": float(r["non_collinearity"]) for r in summary_rows})
    critic_quality_raw = {}
    for r in summary_rows:
        key = f"{r['env_name']}|{r['wrapper_type']}|{r['strength']}|{r['critic_activation']}"
        cq = 0.0
        if finite(r["corr_Q_MC"]):
            cq += max(0.0, float(r["corr_Q_MC"]))
        if finite(r["critic_loss_final"]):
            cq += 1.0 / (1.0 + float(r["critic_loss_final"]))
        critic_quality_raw[key] = cq
    rank_cq = rank_norm(critic_quality_raw)
    for r in summary_rows:
        key = f"{r['env_name']}|{r['wrapper_type']}|{r['strength']}|{r['critic_activation']}"
        score = (
            0.25 * rank_fd.get(key, 0.0)
            + 0.25 * rank_ofd.get(key, 0.0)
            + 0.20 * rank_cross.get(key, 0.0)
            + 0.15 * rank_adv.get(key, 0.0)
            + 0.10 * rank_non.get(key, 0.0)
            + 0.05 * rank_cq.get(key, 0.0)
        )
        r["score"] = score
        r["label"] = label_row(r)

    summary_rows.sort(key=lambda r: float(r["score"]), reverse=True)
    write_csv(RESULT_ROOT / "smooth_reaudit_critic_quality.csv", critic_rows)
    output_rows = []
    for r in summary_rows:
        output_rows.append({
            "env_name": r["env_name"],
            "wrapper_type": r["wrapper_type"],
            "strength": r["strength"],
            "critic_activation": r["critic_activation"],
            "label": r["label"],
            "score": r["score"],
            "critic_loss_final": r["critic_loss_final"],
            "corr_Q_MC": r["corr_Q_MC"],
            "field_norm": r["field_norm"],
            "G_over_F": r["G_over_F"],
            "cos_F_G": r["cos_F_G"],
            "non_collinearity": r["non_collinearity"],
            "fd_skew_ratio": r["fd_skew_ratio"],
            "mean_skew_bilinear": r["mean_skew_bilinear"],
            "mean_sym_bilinear": r["mean_sym_bilinear"],
            "cross_to_same_ratio": r["cross_to_same_ratio"],
            "cross_player_coupling_proxy": r["cross_player_coupling_proxy"],
            "output_fd_skew_ratio": r["output_fd_skew_ratio"],
            "output_rotation_ratio": r["output_rotation_ratio"],
            "output_num_complex_eigs": r["output_num_complex_eigs"],
            "output_max_imag_eig": r["output_max_imag_eig"],
            "output_cross_to_diag_ratio": r["output_cross_to_diag_ratio"],
            "SGD_V_AUC": r["SGD_V_AUC"],
            "EGM_V_AUC": r["EGM_V_AUC"],
            "PPM_V_AUC": r["PPM_V_AUC"],
            "short_baseline_advantage": r["short_baseline_advantage"],
            "clean_task_return_final": r["clean_task_return_final"],
            "current_adv_task_return_final": r["current_adv_task_return_final"],
            "robust_br_task_return_final": r["robust_br_task_return_final"],
            "robust_degradation_final": r["robust_degradation_final"],
        })
    write_csv(RESULT_ROOT / "smooth_reaudit_summary.csv", output_rows)

    plot_rank(summary_rows, "fd_skew_ratio", "FD Skew Ratio", "smooth_reaudit_fd_skew_ratio_rank.png")
    plot_rank(summary_rows, "output_rotation_ratio", "Output Rotation Ratio", "smooth_reaudit_output_rotation_rank.png")
    plot_rank(summary_rows, "short_baseline_advantage", "Short Baseline Advantage", "smooth_reaudit_short_baseline_advantage_rank.png")
    plot_rank(summary_rows, "score", "Geometry Score", "smooth_reaudit_geometry_score_rank.png")

    lines = [
        "# smooth_reaudit_final_report",
        "",
        "1. Why the previous ReLU survey was insufficient for rotation evidence.",
        "   - The previous survey used a ReLU critic, so Hessian-based output-space rotation metrics collapsed toward zero almost everywhere.",
        "   - This re-audit replaced the critic with smooth activations and used finite-difference Jacobian antisymmetry for both parameter-space and output-space checks.",
        "",
        "2. Critic quality comparison between tanh and softplus.",
    ]
    for r in summary_rows:
        lines.append(
            f"   - {r['env_name']} / {r['wrapper_type']} / {r['strength']} / {r['critic_activation']}: critic_loss_final={float(r['critic_loss_final']):.6e}, corr_Q_MC={float(r['corr_Q_MC']) if finite(r['corr_Q_MC']) else math.nan}, rank_corr_Q_MC={float(r['rank_corr_Q_MC']) if finite(r['rank_corr_Q_MC']) else math.nan}"
        )
    lines.extend(["", "3. Finite-difference skew ratios for all candidates."])
    for r in summary_rows:
        lines.append(
            f"   - {r['env_name']} / {r['wrapper_type']} / {r['strength']} / {r['critic_activation']}: fd_skew_ratio={float(r['fd_skew_ratio']):.6e}, output_fd_skew_ratio={float(r['output_fd_skew_ratio']):.6e}, cross_to_same_ratio={float(r['cross_to_same_ratio']):.6e}"
        )
    lines.extend(["", "4. Output-space smooth geometry metrics."])
    for r in summary_rows:
        lines.append(
            f"   - {r['env_name']} / {r['wrapper_type']} / {r['strength']} / {r['critic_activation']}: output_rotation_ratio={float(r['output_rotation_ratio']) if finite(r['output_rotation_ratio']) else math.nan}, output_num_complex_eigs={int(r['output_num_complex_eigs'])}, output_max_imag_eig={float(r['output_max_imag_eig']) if finite(r['output_max_imag_eig']) else math.nan}"
        )
    lines.extend(["", "5. Short EGM/PPM sanity results."])
    for r in summary_rows:
        if finite(r["short_baseline_advantage"]):
            lines.append(
                f"   - {r['env_name']} / {r['wrapper_type']} / {r['strength']} / {r['critic_activation']}: short_baseline_advantage={float(r['short_baseline_advantage']):.3f}, SGD_V_AUC={float(r['SGD_V_AUC']):.3f}, EGM_V_AUC={float(r['EGM_V_AUC']):.3f}, PPM_V_AUC={float(r['PPM_V_AUC']):.3f}"
            )
    labels = {r["label"] for r in summary_rows}
    lines.extend(["", "6. Whether any candidate is confirmed as MODERATE or CONFIRMED skew."])
    if "CONFIRMED_SKEW_CANDIDATE" in labels or "MODERATE_SKEW_CANDIDATE" in labels:
        for r in summary_rows:
            if r["label"] in {"CONFIRMED_SKEW_CANDIDATE", "MODERATE_SKEW_CANDIDATE"}:
                lines.append(f"   - {r['env_name']} / {r['wrapper_type']} / {r['strength']} / {r['critic_activation']}: {r['label']}")
    else:
        lines.append("   - No standard RARL candidate was confirmed to have usable skew geometry after smooth-critic finite-difference re-audit.")
    lines.extend(["", "7. Recommendation for the next full Stage-1/Stage-2 experiment."])
    best = summary_rows[0]
    if best["label"] in {"CONFIRMED_SKEW_CANDIDATE", "MODERATE_SKEW_CANDIDATE"}:
        lines.append(f"   - Next full experiment: {best['env_name']} / {best['wrapper_type']} / strength={best['strength']} / critic={best['critic_activation']}.")
    else:
        lines.extend([
            "   - No standard RARL candidate was confirmed to have usable skew geometry after smooth-critic finite-difference re-audit.",
            "   - Keep Subsection 2 as the positive aligned LQ mechanism experiment.",
            "   - Use standard RARL results as geometry-boundary diagnostics.",
            "   - Do not run proposed on weak-geometry standard environments.",
        ])
    write_text(RESULT_ROOT / "smooth_reaudit_final_report.md", "\n".join(lines) + "\n")

    top_lines = ["# smooth_reaudit_top_candidates", ""]
    for rank, r in enumerate(summary_rows[:3], start=1):
        top_lines.extend([
            f"## Rank {rank}: {r['env_name']} / {r['wrapper_type']} / strength={r['strength']} / critic={r['critic_activation']}",
            f"- label: `{r['label']}`",
            f"- score: `{float(r['score']):.3f}`",
            f"- fd skew: `{float(r['fd_skew_ratio']):.6e}`",
            f"- output fd skew: `{float(r['output_fd_skew_ratio']):.6e}`",
            f"- cross_to_same_ratio: `{float(r['cross_to_same_ratio']):.6e}`",
            f"- short_baseline_advantage: `{float(r['short_baseline_advantage']) if finite(r['short_baseline_advantage']) else math.nan}`",
            f"- critic corr_Q_MC: `{float(r['corr_Q_MC']) if finite(r['corr_Q_MC']) else math.nan}`",
            "",
        ])
    if not any(r["label"] in {"CONFIRMED_SKEW_CANDIDATE", "MODERATE_SKEW_CANDIDATE"} for r in summary_rows):
        top_lines.extend([
            "No standard RARL candidate was confirmed to have usable skew geometry after smooth-critic finite-difference re-audit.",
            "",
            "Recommended stance:",
            "- keep Subsection 2 as the positive aligned LQ mechanism experiment;",
            "- use standard RARL results as geometry-boundary diagnostics;",
            "- do not run proposed on weak-geometry standard environments.",
            "",
        ])
    write_text(RESULT_ROOT / "smooth_reaudit_top_candidates.md", "\n".join(top_lines) + "\n")


if __name__ == "__main__":
    main()
