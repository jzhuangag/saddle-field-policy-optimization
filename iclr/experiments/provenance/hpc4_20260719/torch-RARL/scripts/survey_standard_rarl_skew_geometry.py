from __future__ import annotations

import csv
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

spec = importlib.util.spec_from_file_location("actor_only_critic_rarl_subsection3", BASE_PATH)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


RESULT_ROOT = BASE_PATH.resolve().parents[2] / "results" / "survey_standard_rarl_skew_geometry"
PLOT_ROOT = RESULT_ROOT / "plots"

DEVICE = base.DEVICE
DTYPE = base.DTYPE
EPS = base.EPS
SEED = 0
GAMMA = 0.99

ENV_CANDIDATES = [
    "Hopper-v5",
    "Walker2d-v5",
    "HalfCheetah-v5",
    "Ant-v5",
    "HumanoidStandup-v5",
    "InvertedPendulum-v5",
    "InvertedDoublePendulum-v5",
    "Pusher-v5",
    "Reacher-v5",
    "Swimmer-v5",
    "Pendulum-v1",
    "LunarLanderContinuous-v3",
    "LunarLanderContinuous-v2",
    "BipedalWalker-v3",
    "BipedalWalkerHardcore-v3",
]

ACTION_ALPHA_GRID = [0.02, 0.05, 0.1, 0.2, 0.3]
FORCE_SCALE_GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
WRAPPER_TYPES = [
    "action_disturbance_direct",
    "action_disturbance_rotated",
    "mujoco_external_force_xz",
    "mujoco_external_force_xyz",
]

ACTOR_HIDDEN = (64, 64)
CRITIC_HIDDEN = (128, 128)
WARMUP_SIMPLE = 10000
WARMUP_MUJOCO = 20000
CRITIC_TRAIN_STEPS = 3000
CRITIC_LR = 1e-3
BATCH_SIZE = 256
POLYAK_TAU = 0.005
CRITIC_AUDIT_INTERVAL = 1000
MC_HORIZON = 80
MC_SNAPSHOTS = 8
DIAG_BATCH_SIZE = 512
OUTPUT_DIAG_BATCH = 16
ROTATION_PROBES = 8
ACTOR_LR_GRID = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
SHORT_ACTOR_ITERS = 50
TOP_K_SANITY = 8
LAMBDA_F = 0.01
LAMBDA_P = 1.0
TAU = 0.03
GAP_INNER_STEPS = 2
LOCAL_GAP_RADIUS = 0.1
BR_INNER_STEPS = 10
BR_INNER_LR = 1e-4
LOCAL_BR_RADIUS = 0.25


def ensure_dirs() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    base.write_csv(path, rows)


def write_text(path: Path, text: str) -> None:
    base.write_text(path, text)


def finite(value: float) -> bool:
    return math.isfinite(float(value))


def spike_ratio(values: list[float]) -> float:
    return base.spike_ratio(values)


def mean_or_nan(values: list[float]) -> float:
    arr = [float(v) for v in values if finite(v)]
    return float(np.mean(arr)) if arr else math.nan


def corr_or_nan(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return math.nan
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if np.std(xx) < 1e-12 or np.std(yy) < 1e-12:
        return math.nan
    return float(np.corrcoef(xx, yy)[0, 1])


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


@dataclass(frozen=True)
class EnvInfo:
    env_name: str
    available: bool
    backend_type: str
    obs_dim: int | None
    action_dim: int | None
    action_low: np.ndarray | None
    action_high: np.ndarray | None
    max_episode_steps: int | None
    has_mujoco: bool
    has_xfrc_applied: bool
    body_names: list[str]
    notes: str
    main_body_id: int | None
    main_body_name: str | None


@dataclass(frozen=True)
class WrapperConfig:
    wrapper_type: str
    strength_value: float
    strength_name: str
    adv_dim: int
    body_id: int | None
    body_name: str | None


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
        idx = rng.integers(0, self.size, size=batch_size)
        return {"obs": torch.as_tensor(self.obs[idx], dtype=DTYPE, device=DEVICE)}


class CriticNet(nn.Module):
    def __init__(self, obs_dim: int, u_dim: int, w_dim: int) -> None:
        super().__init__()
        input_dim = obs_dim + u_dim + w_dim
        self.fc1 = nn.Linear(input_dim, CRITIC_HIDDEN[0])
        self.fc2 = nn.Linear(CRITIC_HIDDEN[0], CRITIC_HIDDEN[1])
        self.fc3 = nn.Linear(CRITIC_HIDDEN[1], 1)

    def forward(self, obs: torch.Tensor, u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, u, w], dim=-1)
        # A smooth critic is required because the experiment differentiates the
        # joint actor field a second time. ReLU would make the action Hessian,
        # and therefore the local game rotation, zero almost everywhere.
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


class GameBilinearCriticNet(nn.Module):
    """State-conditioned zero-sum critic with explicit cross-action structure."""

    def __init__(self, obs_dim: int, u_dim: int, w_dim: int, own_curvature_scale: float = 0.1) -> None:
        super().__init__()
        self.u_dim = u_dim
        self.w_dim = w_dim
        self.own_curvature_scale = float(own_curvature_scale)
        self.fc1 = nn.Linear(obs_dim, CRITIC_HIDDEN[0])
        self.fc2 = nn.Linear(CRITIC_HIDDEN[0], CRITIC_HIDDEN[1])
        output_dim = 1 + u_dim + w_dim + (u_dim * w_dim) + u_dim + w_dim
        self.head = nn.Linear(CRITIC_HIDDEN[1], output_dim)

    def forward(self, obs: torch.Tensor, u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.fc1(obs))
        h = F.silu(self.fc2(h))
        coeff = self.head(h)
        offset = 0
        value = coeff[:, offset]
        offset += 1
        linear_u = coeff[:, offset : offset + self.u_dim]
        offset += self.u_dim
        linear_w = coeff[:, offset : offset + self.w_dim]
        offset += self.w_dim
        bilinear = coeff[:, offset : offset + (self.u_dim * self.w_dim)].reshape(-1, self.u_dim, self.w_dim)
        offset += self.u_dim * self.w_dim
        quadratic_u = coeff[:, offset : offset + self.u_dim]
        offset += self.u_dim
        quadratic_w = coeff[:, offset : offset + self.w_dim]
        cross = torch.einsum("bi,bij,bj->b", u, bilinear, w)
        own = 0.5 * self.own_curvature_scale * (
            torch.sum(quadratic_u * u.square(), dim=-1)
            + torch.sum(quadratic_w * w.square(), dim=-1)
        )
        return value + torch.sum(linear_u * u, dim=-1) + torch.sum(linear_w * w, dim=-1) + cross + own


def choose_main_body(env) -> tuple[int | None, str | None, list[str]]:
    body_names: list[str] = []
    body_id = None
    body_name = None
    uw = env.unwrapped
    if hasattr(uw, "model") and hasattr(uw.model, "nbody"):
        for i in range(int(uw.model.nbody)):
            try:
                name = uw.model.body(i).name
            except Exception:
                try:
                    name = str(uw.model.body_id2name(i))
                except Exception:
                    name = ""
            body_names.append(name)
        preferred = ["torso", "pelvis", "body0", "cart", "mid", "back", "fingertip"]
        for pref in preferred:
            if pref in body_names:
                body_id = body_names.index(pref)
                body_name = pref
                break
        if body_id is None:
            for idx, name in enumerate(body_names):
                if idx == 0:
                    continue
                body_id = idx
                body_name = name or f"body_{idx}"
                break
    return body_id, body_name, body_names


def environment_availability() -> list[EnvInfo]:
    rows: list[dict[str, Any]] = []
    infos: list[EnvInfo] = []
    for env_name in ENV_CANDIDATES:
        try:
            env = base.gym.make(env_name)
            action_space = env.action_space
            obs_space = env.observation_space
            is_box = action_space.__class__.__name__ == "Box" and len(getattr(action_space, "shape", ())) == 1
            has_mujoco = hasattr(env.unwrapped, "data") and hasattr(env.unwrapped, "model")
            has_xfrc = has_mujoco and hasattr(env.unwrapped.data, "xfrc_applied")
            body_id, body_name, body_names = choose_main_body(env)
            info = EnvInfo(
                env_name=env_name,
                available=bool(is_box),
                backend_type=base.GYM_BACKEND,
                obs_dim=int(obs_space.shape[0]) if is_box else None,
                action_dim=int(action_space.shape[0]) if is_box else None,
                action_low=np.asarray(action_space.low, dtype=np.float32).copy() if is_box else None,
                action_high=np.asarray(action_space.high, dtype=np.float32).copy() if is_box else None,
                max_episode_steps=int(env.spec.max_episode_steps) if getattr(env, "spec", None) is not None else None,
                has_mujoco=bool(has_mujoco),
                has_xfrc_applied=bool(has_xfrc),
                body_names=body_names,
                notes="",
                main_body_id=body_id,
                main_body_name=body_name,
            )
            rows.append(
                {
                    "env_name": env_name,
                    "available": int(info.available),
                    "obs_dim": info.obs_dim if info.obs_dim is not None else "",
                    "action_dim": info.action_dim if info.action_dim is not None else "",
                    "action_low": info.action_low.tolist() if info.action_low is not None else "",
                    "action_high": info.action_high.tolist() if info.action_high is not None else "",
                    "max_episode_steps": info.max_episode_steps if info.max_episode_steps is not None else "",
                    "backend_type": info.backend_type,
                    "has_mujoco": int(info.has_mujoco),
                    "has_xfrc_applied": int(info.has_xfrc_applied),
                    "body_names": "|".join(info.body_names[:12]),
                    "notes": info.notes,
                }
            )
            env.close()
        except Exception as exc:
            info = EnvInfo(
                env_name=env_name,
                available=False,
                backend_type=base.GYM_BACKEND,
                obs_dim=None,
                action_dim=None,
                action_low=None,
                action_high=None,
                max_episode_steps=None,
                has_mujoco=False,
                has_xfrc_applied=False,
                body_names=[],
                notes=repr(exc),
                main_body_id=None,
                main_body_name=None,
            )
            rows.append(
                {
                    "env_name": env_name,
                    "available": 0,
                    "obs_dim": "",
                    "action_dim": "",
                    "action_low": "",
                    "action_high": "",
                    "max_episode_steps": "",
                    "backend_type": base.GYM_BACKEND,
                    "has_mujoco": 0,
                    "has_xfrc_applied": 0,
                    "body_names": "",
                    "notes": repr(exc),
                }
            )
        infos.append(info)
    write_csv(RESULT_ROOT / "survey_env_availability.csv", rows)
    lines = ["# survey_env_availability", ""]
    for info in infos:
        if info.available:
            lines.append(
                f"- {info.env_name}: available, obs_dim=`{info.obs_dim}`, action_dim=`{info.action_dim}`, max_episode_steps=`{info.max_episode_steps}`, has_mujoco=`{info.has_mujoco}`, has_xfrc_applied=`{info.has_xfrc_applied}`, main_body=`{info.main_body_name}`"
            )
        else:
            lines.append(f"- {info.env_name}: unavailable, notes=`{info.notes}`")
    write_text(RESULT_ROOT / "survey_env_availability.md", "\n".join(lines) + "\n")
    return infos


class SurveyWrapper:
    def __init__(self, info: EnvInfo, cfg: WrapperConfig) -> None:
        self.info = info
        self.cfg = cfg
        self.action_low = torch.as_tensor(info.action_low, dtype=DTYPE, device=DEVICE)
        self.action_high = torch.as_tensor(info.action_high, dtype=DTYPE, device=DEVICE)
        self.action_scale = 0.5 * (self.action_high - self.action_low)
        self.action_bias = 0.5 * (self.action_high + self.action_low)
        self.r_dyn = self.build_r_dyn(info.action_dim)

    def build_r_dyn(self, dim: int) -> torch.Tensor:
        mat = torch.zeros((dim, dim), dtype=DTYPE, device=DEVICE)
        for start in range(0, dim - 1, 2):
            mat[start, start + 1] = 1.0
            mat[start + 1, start] = -1.0
        return mat

    def scale_protagonist(self, raw: torch.Tensor) -> torch.Tensor:
        return self.action_bias + (self.action_scale * torch.tanh(raw))

    def scale_adversary(self, raw: torch.Tensor) -> torch.Tensor:
        if self.cfg.wrapper_type.startswith("action_disturbance"):
            return self.action_bias + (self.action_scale * torch.tanh(raw))
        return torch.tanh(raw)

    def perturb(self, w: torch.Tensor) -> torch.Tensor:
        if self.cfg.wrapper_type == "action_disturbance_direct":
            return w
        if self.cfg.wrapper_type == "action_disturbance_rotated":
            return self.r_dyn @ w
        raise ValueError(self.cfg.wrapper_type)

    def action_step(self, u: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, float]:
        a_env_raw = u + (self.cfg.strength_value * self.perturb(w))
        a_env = torch.clamp(a_env_raw, self.action_low, self.action_high)
        clip_fraction = float(((a_env_raw - a_env).abs() > 1e-12).float().mean().item())
        return a_env, clip_fraction

    def external_force_vector(self, w: torch.Tensor) -> torch.Tensor:
        if self.cfg.wrapper_type == "mujoco_external_force_xz_radial":
            radial = w / torch.clamp(torch.linalg.norm(w), min=1.0)
            return self.cfg.strength_value * torch.stack([radial[0], torch.zeros((), dtype=DTYPE, device=DEVICE), radial[1]])
        if self.cfg.wrapper_type == "mujoco_external_force_xz":
            return self.cfg.strength_value * torch.stack([w[0], torch.zeros((), dtype=DTYPE, device=DEVICE), w[1]])
        if self.cfg.wrapper_type == "mujoco_external_force_xyz":
            return self.cfg.strength_value * w
        raise ValueError(self.cfg.wrapper_type)

    def clear_force(self, env) -> None:
        if self.cfg.body_id is not None and hasattr(env.unwrapped.data, "xfrc_applied"):
            env.unwrapped.data.xfrc_applied[self.cfg.body_id, 0:3] = 0.0

    def apply_force(self, env, w: torch.Tensor) -> tuple[float, float]:
        force = self.external_force_vector(w)
        env.unwrapped.data.xfrc_applied[self.cfg.body_id, 0:3] = force.detach().cpu().numpy().astype(np.float64)
        return float(torch.linalg.norm(force).item()), float(torch.max(force.abs()).item())


class SurveyGame:
    def __init__(
        self,
        info: EnvInfo,
        cfg: WrapperConfig,
        seed: int = SEED,
        twin_critic: bool = False,
        actor_scope: str = "full",
        critic_arch: str = "mlp",
        own_curvature_scale: float = 0.1,
    ) -> None:
        base.seed_everything(seed)
        self.info = info
        self.cfg = cfg
        self.wrapper = SurveyWrapper(info, cfg)
        self.theta_layout = base.FlatMLP(info.obs_dim, (64, 64), info.action_dim)
        self.phi_layout = base.FlatMLP(info.obs_dim, (64, 64), cfg.adv_dim)
        self.theta = self.theta_layout.init_flat(torch.Generator(device=DEVICE).manual_seed(seed), final_scale=0.05).detach().clone()
        self.phi = self.phi_layout.init_flat(torch.Generator(device=DEVICE).manual_seed(seed + 1), final_scale=0.05).detach().clone()
        self.theta_target = self.theta.detach().clone()
        self.phi_target = self.phi.detach().clone()
        if critic_arch not in {"mlp", "game_bilinear"}:
            raise ValueError(f"Unsupported critic_arch={critic_arch!r}")
        self.critic_arch = critic_arch
        critic_factory = (
            (lambda: CriticNet(info.obs_dim, info.action_dim, cfg.adv_dim))
            if critic_arch == "mlp"
            else (lambda: GameBilinearCriticNet(info.obs_dim, info.action_dim, cfg.adv_dim, own_curvature_scale))
        )
        self.q_t = critic_factory().to(DEVICE)
        self.q_t_target = critic_factory().to(DEVICE)
        self.q_t_target.load_state_dict(self.q_t.state_dict())
        self.q_opt = torch.optim.Adam(self.q_t.parameters(), lr=CRITIC_LR)
        self.q_t2 = critic_factory().to(DEVICE) if twin_critic else None
        self.q_t2_target = critic_factory().to(DEVICE) if twin_critic else None
        self.q_opt2 = torch.optim.Adam(self.q_t2.parameters(), lr=CRITIC_LR) if twin_critic else None
        if twin_critic:
            self.q_t2_target.load_state_dict(self.q_t2.state_dict())
        self.critic_softmin_temperature = 1.0
        self.target_policy_noise = 0.0
        self.target_noise_clip = 0.0
        self.replay = ReplayBuffer(250000, info.obs_dim, info.action_dim, cfg.adv_dim)
        self.rng = np.random.default_rng(seed)
        self.theta_slice = slice(0, self.theta_layout.num_params)
        self.phi_slice = slice(self.theta_layout.num_params, self.theta_layout.num_params + self.phi_layout.num_params)
        if actor_scope not in {"full", "head"}:
            raise ValueError(f"Unsupported actor_scope={actor_scope!r}")
        self.actor_scope = actor_scope
        self.saddle_mask = torch.ones(self.theta_layout.num_params + self.phi_layout.num_params, dtype=DTYPE, device=DEVICE)
        if actor_scope == "head":
            self.saddle_mask.zero_()
            theta_head = int(np.prod(self.theta_layout.shapes[-2]) + np.prod(self.theta_layout.shapes[-1]))
            phi_head = int(np.prod(self.phi_layout.shapes[-2]) + np.prod(self.phi_layout.shapes[-1]))
            self.saddle_mask[self.theta_slice.stop - theta_head : self.theta_slice.stop] = 1.0
            self.saddle_mask[self.phi_slice.stop - phi_head : self.phi_slice.stop] = 1.0
        self.metric_refs: dict[str, float] = {}

    def current_z(self) -> torch.Tensor:
        return torch.cat([self.theta, self.phi]).detach().clone()

    def set_from_z(self, z: torch.Tensor) -> None:
        self.theta = z[self.theta_slice].detach().clone()
        self.phi = z[self.phi_slice].detach().clone()

    def actor_protagonist(self, theta: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        raw = self.theta_layout.forward(theta, obs)
        return self.wrapper.scale_protagonist(raw)

    def actor_adversary(self, phi: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        raw = self.phi_layout.forward(phi, obs)
        return self.wrapper.scale_adversary(raw)

    def actor_z(self, z: torch.Tensor, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor_protagonist(z[self.theta_slice], obs), self.actor_adversary(z[self.phi_slice], obs)

    def polyak(self) -> None:
        self.theta_target = ((1.0 - POLYAK_TAU) * self.theta_target) + (POLYAK_TAU * self.theta)
        self.phi_target = ((1.0 - POLYAK_TAU) * self.phi_target) + (POLYAK_TAU * self.phi)
        with torch.no_grad():
            for t, o in zip(self.q_t_target.parameters(), self.q_t.parameters()):
                t.data.mul_(1.0 - POLYAK_TAU).add_(POLYAK_TAU * o.data)
            if self.q_t2 is not None:
                for t, o in zip(self.q_t2_target.parameters(), self.q_t2.parameters()):
                    t.data.mul_(1.0 - POLYAK_TAU).add_(POLYAK_TAU * o.data)

    def step_env(self, env, u: torch.Tensor, w: torch.Tensor) -> tuple[np.ndarray, float, bool, float, float, bool]:
        force_norm = 0.0
        force_max = 0.0
        clip_frac = 0.0
        step_error = False
        try:
            if self.cfg.wrapper_type.startswith("action_disturbance"):
                a_env, clip_frac = self.wrapper.action_step(u, w)
                next_obs, reward, terminated, truncated, _ = env.step(a_env.detach().cpu().numpy().astype(np.float32))
            else:
                force_norm, force_max = self.wrapper.apply_force(env, w)
                next_obs, reward, terminated, truncated, _ = env.step(u.detach().cpu().numpy().astype(np.float32))
                self.wrapper.clear_force(env)
            done = terminated or truncated
            return np.asarray(next_obs, dtype=np.float32), float(reward), done, clip_frac, force_norm, step_error
        except Exception:
            self.wrapper.clear_force(env)
            step_error = True
            return np.zeros((self.info.obs_dim,), dtype=np.float32), 0.0, True, clip_frac, force_norm, step_error

    def evaluate_random_episode(self, env, adv_on: bool, noise_std: float) -> dict[str, float]:
        obs, _ = env.reset(seed=SEED + int(self.rng.integers(0, 1_000_000)))
        total = 0.0
        clip_fracs: list[float] = []
        force_norms: list[float] = []
        state_finite = True
        step_error = False
        steps = 0
        while True:
            obs_t = torch.as_tensor(obs, dtype=DTYPE, device=DEVICE).unsqueeze(0)
            u = self.actor_protagonist(self.theta, obs_t).squeeze(0)
            u = torch.clamp(u + (noise_std * torch.randn_like(u)), self.wrapper.action_low, self.wrapper.action_high)
            if adv_on:
                w = self.actor_adversary(self.phi, obs_t).squeeze(0)
                w = torch.clamp(w + (noise_std * torch.randn_like(w)), -torch.ones_like(w), torch.ones_like(w))
            else:
                w = torch.zeros((self.cfg.adv_dim,), dtype=DTYPE, device=DEVICE)
            next_obs, reward, done, clip_frac, force_norm, err = self.step_env(env, u, w)
            total += reward
            steps += 1
            clip_fracs.append(clip_frac)
            force_norms.append(force_norm)
            state_finite = state_finite and bool(np.isfinite(next_obs).all())
            step_error = step_error or err
            obs = next_obs
            if done or steps >= self.info.max_episode_steps:
                break
        return {
            "return": total,
            "episode_length": steps,
            "clip_frac": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "force_norm_mean": float(np.mean(force_norms)) if force_norms else 0.0,
            "force_norm_max": float(np.max(force_norms)) if force_norms else 0.0,
            "state_finite": int(state_finite),
            "step_error": int(step_error),
            "early_termination": int(steps < self.info.max_episode_steps),
        }

    def reward_target(self, reward_raw: float) -> float:
        return reward_raw

    def collect_replay(self, warmup_steps: int, noise_std: float) -> dict[str, float]:
        env = base.gym.make(self.info.env_name)
        obs, _ = env.reset(seed=SEED + 99)
        u_vals: list[np.ndarray] = []
        w_vals: list[np.ndarray] = []
        clip_fracs: list[float] = []
        force_norms: list[float] = []
        state_finite = True
        step_error = False
        for _ in range(warmup_steps):
            obs_t = torch.as_tensor(obs, dtype=DTYPE, device=DEVICE).unsqueeze(0)
            u = self.actor_protagonist(self.theta, obs_t).squeeze(0)
            w = self.actor_adversary(self.phi, obs_t).squeeze(0)
            u = torch.clamp(u + (noise_std * torch.randn_like(u)), self.wrapper.action_low, self.wrapper.action_high)
            w = torch.clamp(w + (noise_std * torch.randn_like(w)), -torch.ones_like(w), torch.ones_like(w))
            next_obs, reward, done, clip_frac, force_norm, err = self.step_env(env, u, w)
            self.replay.add(
                np.asarray(obs, dtype=np.float32),
                u.detach().cpu().numpy().astype(np.float32),
                w.detach().cpu().numpy().astype(np.float32),
                self.reward_target(reward),
                np.asarray(next_obs, dtype=np.float32),
                done,
            )
            u_vals.append(u.detach().cpu().numpy())
            w_vals.append(w.detach().cpu().numpy())
            clip_fracs.append(clip_frac)
            force_norms.append(force_norm)
            state_finite = state_finite and bool(np.isfinite(next_obs).all())
            step_error = step_error or err
            obs = next_obs
            if done:
                obs, _ = env.reset()
        env.close()
        rw_vals = []
        if self.cfg.wrapper_type == "action_disturbance_rotated":
            for w in w_vals:
                rw_vals.append((self.wrapper.r_dyn.cpu().numpy() @ w))
        elif self.cfg.wrapper_type == "action_disturbance_direct":
            rw_vals = list(w_vals)
        else:
            rw_vals = list(w_vals)
        return {
            "std_u": float(np.std(np.asarray(u_vals, dtype=np.float64))) if u_vals else 0.0,
            "std_w": float(np.std(np.asarray(w_vals, dtype=np.float64))) if w_vals else 0.0,
            "std_Rw": float(np.std(np.asarray(rw_vals, dtype=np.float64))) if rw_vals else 0.0,
            "action_clip_fraction": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "force_norm_mean": float(np.mean(force_norms)) if force_norms else 0.0,
            "force_norm_max": float(np.max(force_norms)) if force_norms else 0.0,
            "state_finite_flag": int(state_finite),
            "step_error_flag": int(step_error),
        }

    def critic_update(self) -> dict[str, float]:
        batch = self.replay.sample(BATCH_SIZE, self.rng)
        with torch.no_grad():
            u_next = self.actor_protagonist(self.theta_target, batch["next_obs"])
            w_next = self.actor_adversary(self.phi_target, batch["next_obs"])
            if self.target_policy_noise > 0.0:
                u_noise = torch.clamp(
                    self.target_policy_noise * torch.randn_like(u_next),
                    -self.target_noise_clip,
                    self.target_noise_clip,
                )
                w_noise = torch.clamp(
                    self.target_policy_noise * torch.randn_like(w_next),
                    -self.target_noise_clip,
                    self.target_noise_clip,
                )
                u_next = torch.clamp(u_next + u_noise, self.wrapper.action_low, self.wrapper.action_high)
                w_next = torch.clamp(w_next + w_noise, -1.0, 1.0)
            q_next = self.q_t_target(batch["next_obs"], u_next, w_next)
            if self.q_t2_target is not None:
                q_next = torch.minimum(q_next, self.q_t2_target(batch["next_obs"], u_next, w_next))
            y = batch["reward"] + (GAMMA * (1.0 - batch["done"]) * q_next)
        pred = self.q_t(batch["obs"], batch["u"], batch["w"])
        loss = torch.mean((pred - y) ** 2)
        self.q_opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = math.sqrt(sum(float(torch.sum(p.grad.detach() * p.grad.detach()).item()) for p in self.q_t.parameters() if p.grad is not None))
        self.q_opt.step()
        loss2_value = math.nan
        disagreement = math.nan
        if self.q_t2 is not None:
            batch2 = self.replay.sample(BATCH_SIZE, self.rng)
            with torch.no_grad():
                u_next2 = self.actor_protagonist(self.theta_target, batch2["next_obs"])
                w_next2 = self.actor_adversary(self.phi_target, batch2["next_obs"])
                if self.target_policy_noise > 0.0:
                    u_noise2 = torch.clamp(
                        self.target_policy_noise * torch.randn_like(u_next2),
                        -self.target_noise_clip,
                        self.target_noise_clip,
                    )
                    w_noise2 = torch.clamp(
                        self.target_policy_noise * torch.randn_like(w_next2),
                        -self.target_noise_clip,
                        self.target_noise_clip,
                    )
                    u_next2 = torch.clamp(u_next2 + u_noise2, self.wrapper.action_low, self.wrapper.action_high)
                    w_next2 = torch.clamp(w_next2 + w_noise2, -1.0, 1.0)
                q_next21 = self.q_t_target(batch2["next_obs"], u_next2, w_next2)
                q_next22 = self.q_t2_target(batch2["next_obs"], u_next2, w_next2)
                y2 = batch2["reward"] + GAMMA * (1.0 - batch2["done"]) * torch.minimum(q_next21, q_next22)
            pred2 = self.q_t2(batch2["obs"], batch2["u"], batch2["w"])
            loss2 = torch.mean((pred2 - y2) ** 2)
            self.q_opt2.zero_grad(set_to_none=True)
            loss2.backward()
            self.q_opt2.step()
            loss2_value = float(loss2.detach().item())
            with torch.no_grad():
                pred2_same = self.q_t2(batch["obs"], batch["u"], batch["w"])
                disagreement = float(
                    torch.mean(torch.abs(pred - pred2_same)).item()
                    / (torch.mean(0.5 * (torch.abs(pred) + torch.abs(pred2_same))).item() + EPS)
                )
        self.polyak()
        return {
            "critic_loss": float(loss.detach().item()),
            "critic2_loss": loss2_value,
            "critic_relative_disagreement": disagreement,
            "critic_grad_norm": float(grad_norm),
            "Q_mean": float(pred.detach().mean().item()),
            "Q_std": float(pred.detach().std(unbiased=False).item()),
            "Q_abs_mean": float(pred.detach().abs().mean().item()),
            "target_mean": float(y.detach().mean().item()),
            "target_std": float(y.detach().std(unbiased=False).item()),
        }

    def can_snapshot(self, env) -> bool:
        uw = env.unwrapped
        return hasattr(uw, "set_state") and hasattr(uw, "data") and hasattr(uw.data, "qpos") and hasattr(uw.data, "qvel")

    def collect_snapshots(self, count: int) -> list[dict[str, np.ndarray]]:
        env = base.gym.make(self.info.env_name)
        if not self.can_snapshot(env):
            env.close()
            return []
        obs, _ = env.reset(seed=SEED + 555)
        snaps: list[dict[str, np.ndarray]] = []
        for _ in range(max(count * 5, count)):
            obs_t = torch.as_tensor(obs, dtype=DTYPE, device=DEVICE).unsqueeze(0)
            u = self.actor_protagonist(self.theta, obs_t).squeeze(0)
            w = self.actor_adversary(self.phi, obs_t).squeeze(0)
            next_obs, _, done, _, _, err = self.step_env(env, u, w)
            if not err and len(snaps) < count:
                snaps.append(
                    {
                        # qpos/qvel are read after step_env, so pair them with the
                        # post-step observation rather than the stale input state.
                        "obs": np.asarray(next_obs, dtype=np.float32).copy(),
                        "qpos": env.unwrapped.data.qpos.copy(),
                        "qvel": env.unwrapped.data.qvel.copy(),
                    }
                )
            obs = next_obs
            if done:
                obs, _ = env.reset()
            if len(snaps) >= count:
                break
        env.close()
        return snaps

    def mc_quality(self, snapshots: list[dict[str, np.ndarray]]) -> dict[str, float]:
        if not snapshots:
            return {"corr_Q_MC": math.nan, "mse_Q_MC": math.nan}
        env = base.gym.make(self.info.env_name)
        q_vals: list[float] = []
        mc_vals: list[float] = []
        for snap in snapshots:
            env.reset(seed=SEED)
            env.unwrapped.set_state(snap["qpos"], snap["qvel"])
            obs_t = torch.as_tensor(np.asarray(snap["obs"], dtype=np.float32), dtype=DTYPE, device=DEVICE).unsqueeze(0)
            u = self.actor_protagonist(self.theta, obs_t)
            w = self.actor_adversary(self.phi, obs_t)
            q1 = self.q_t(obs_t, u, w)
            if self.q_t2 is None:
                q_value = q1
            else:
                q2 = self.q_t2(obs_t, u, w)
                temperature = max(float(self.critic_softmin_temperature), EPS)
                q_value = -temperature * torch.logsumexp(
                    torch.stack((-q1 / temperature, -q2 / temperature), dim=0), dim=0
                )
            q_vals.append(float(q_value.detach().item()))
            disc = 1.0
            total = 0.0
            for _ in range(MC_HORIZON):
                u_step = self.actor_protagonist(self.theta, obs_t).squeeze(0)
                w_step = self.actor_adversary(self.phi, obs_t).squeeze(0)
                next_obs, reward, done, _, _, err = self.step_env(env, u_step, w_step)
                if err:
                    break
                total += disc * reward
                disc *= GAMMA
                obs_t = torch.as_tensor(next_obs, dtype=DTYPE, device=DEVICE).unsqueeze(0)
                if done:
                    break
            mc_vals.append(total)
        env.close()
        return {"corr_Q_MC": corr_or_nan(q_vals, mc_vals), "mse_Q_MC": mse_or_nan(q_vals, mc_vals)}

    def actor_objective(self, z: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        # Frozen coordinates must be absent from the saddle Jacobian, not merely
        # assigned a zero update after differentiation.
        z_effective = z * self.saddle_mask + z.detach() * (1.0 - self.saddle_mask)
        theta = z_effective[self.theta_slice]
        phi = z_effective[self.phi_slice]
        u = self.actor_protagonist(theta, states)
        w = self.actor_adversary(phi, states)
        q1 = self.q_t(states, u, w)
        if self.q_t2 is None:
            return q1.mean()
        q2 = self.q_t2(states, u, w)
        temperature = max(float(self.critic_softmin_temperature), EPS)
        conservative_q = -temperature * torch.logsumexp(
            torch.stack((-q1 / temperature, -q2 / temperature), dim=0), dim=0
        )
        return conservative_q.mean()

    def actor_field(self, z: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        z_req = z if z.requires_grad else z.detach().clone().requires_grad_(True)
        j = self.actor_objective(z_req, states)
        grad = torch.autograd.grad(j, z_req, create_graph=True)[0]
        out = torch.zeros_like(z_req)
        out[self.theta_slice] = -grad[self.theta_slice]
        out[self.phi_slice] = +grad[self.phi_slice]
        return out * self.saddle_mask

    def local_gap(self, z: torch.Tensor, states: torch.Tensor, protagonist: bool, actor_lr: float, with_prox: bool) -> torch.Tensor:
        base_z = z.detach().clone()
        current = base_z.clone()
        sl = self.theta_slice if protagonist else self.phi_slice
        initial = base_z[sl].clone()
        j_start = self.actor_objective(base_z, states)
        inner_lr = 0.1 * actor_lr
        for _ in range(GAP_INNER_STEPS):
            cur = current.detach().clone().requires_grad_(True)
            j = self.actor_objective(cur, states)
            grad = torch.autograd.grad(j, cur)[0][sl]
            grad = grad * self.saddle_mask[sl]
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

    def diagnostic_metrics(self, z: torch.Tensor, diag_batch: dict[str, torch.Tensor], actor_lr: float, compute_geometry: bool) -> dict[str, float]:
        states = diag_batch["obs"]
        field = self.actor_field(z, states).detach()
        field_energy = 0.5 * float(torch.dot(field, field).item())
        p_gap = float(self.local_gap(z, states, True, actor_lr, True).detach().item())
        a_gap = float(self.local_gap(z, states, False, actor_lr, True).detach().item())
        p_tau = p_gap + a_gap
        if "field0" not in self.metric_refs:
            self.metric_refs["field0"] = field_energy
            self.metric_refs["ptau0"] = max(p_tau, EPS)
        field_term = field_energy / (self.metric_refs["field0"] + EPS)
        p_tau_term = p_tau / (self.metric_refs["ptau0"] + EPS)
        v_std = (LAMBDA_F * field_term) + (LAMBDA_P * p_tau_term)
        exploit_p, exploit_a = self.exploitability(z, states, actor_lr)
        out = {
            "V_std": v_std,
            "field_term": field_term,
            "P_tau_std": p_tau_term,
            "raw_P_tau_std": p_tau,
            "field_norm": float(torch.linalg.norm(field).item()),
            "approx_exploitability_std": exploit_p + exploit_a,
            "Q_std": float(self.actor_objective(z, states).detach().item()),
        }
        if compute_geometry:
            out.update(self.parameter_geometry(z, states, field))
        else:
            out.update(
                {
                    "G_norm": math.nan,
                    "G_over_F": math.nan,
                    "cos_F_G": math.nan,
                    "non_collinearity": math.nan,
                    "rotation_ratio_proxy": math.nan,
                    "A_proxy": math.nan,
                    "S_proxy": math.nan,
                    "jt_proxy_available": 0,
                    "cross_player_coupling_proxy": math.nan,
                    "same_player_proxy": math.nan,
                    "cross_to_same_ratio": math.nan,
                }
            )
        return out

    def parameter_geometry(self, z: torch.Tensor, states: torch.Tensor, field: torch.Tensor | None = None) -> dict[str, float]:
        z_req = z.detach().clone().requires_grad_(True)
        field = self.actor_field(z_req, states) if field is None else field
        _, g_vec = torch.autograd.functional.jvp(lambda zz: self.actor_field(zz, states), (z_req,), (field.detach(),), create_graph=False, strict=False)
        field_det = field.detach()
        g_det = g_vec.detach()
        f_norm = float(torch.linalg.norm(field_det).item())
        g_norm = float(torch.linalg.norm(g_det).item())
        cos_fg = float(torch.dot(field_det, g_det).item() / ((f_norm * g_norm) + EPS))
        # The theorem's directional rotation test is evaluated along F itself:
        # JF = SF + WF and J^T F = SF - WF.
        jtf = torch.autograd.grad(torch.dot(field, field_det), z_req, retain_graph=True)[0].detach()
        sf = 0.5 * (g_det + jtf)
        wf = 0.5 * (g_det - jtf)
        sf_norm = float(torch.linalg.norm(sf).item())
        wf_norm = float(torch.linalg.norm(wf).item())
        aproxy = 0.0
        sproxy = 0.0
        jt_available = 1
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(20260616)
        for _ in range(ROTATION_PROBES):
            v = torch.randn(z_req.numel(), generator=gen, dtype=DTYPE, device=DEVICE)
            v = v / (torch.linalg.norm(v) + EPS)
            _, jv = torch.autograd.functional.jvp(lambda zz: self.actor_field(zz, states), (z_req,), (v,), create_graph=False, strict=False)
            try:
                jtv = torch.autograd.grad(torch.dot(field, v), z_req, retain_graph=True)[0]
            except Exception:
                jt_available = 0
                break
            aproxy += float(torch.linalg.norm(jv.detach() - jtv.detach()).item())
            sproxy += float(torch.linalg.norm(jv.detach() + jtv.detach()).item())
        theta_slice = self.theta_slice
        phi_slice = self.phi_slice

        def p_field(cur_z: torch.Tensor) -> torch.Tensor:
            return self.actor_field(cur_z, states)[theta_slice].detach()

        def a_field(cur_z: torch.Tensor) -> torch.Tensor:
            return self.actor_field(cur_z, states)[phi_slice].detach()

        base_p = p_field(z)
        base_a = a_field(z)
        pert_theta = z.detach().clone()
        pert_phi = z.detach().clone()
        pert_theta[theta_slice] = pert_theta[theta_slice] + 1e-3 * self.saddle_mask[theta_slice]
        pert_phi[phi_slice] = pert_phi[phi_slice] + 1e-3 * self.saddle_mask[phi_slice]
        cross_p = float(torch.linalg.norm(p_field(pert_phi) - base_p).item()) / 1e-3
        cross_a = float(torch.linalg.norm(a_field(pert_theta) - base_a).item()) / 1e-3
        same_p = float(torch.linalg.norm(p_field(pert_theta) - base_p).item()) / 1e-3
        same_a = float(torch.linalg.norm(a_field(pert_phi) - base_a).item()) / 1e-3
        cross = 0.5 * (cross_p + cross_a)
        same = 0.5 * (same_p + same_a)
        return {
            "G_norm": g_norm,
            "G_over_F": g_norm / (f_norm + EPS),
            "cos_F_G": cos_fg,
            "non_collinearity": math.sqrt(max(0.0, 1.0 - min(1.0, cos_fg * cos_fg))),
            "rotation_ratio_proxy": aproxy / (sproxy + EPS) if jt_available else math.nan,
            "SF_norm": sf_norm,
            "WF_norm": wf_norm,
            "WF_over_SF": wf_norm / (sf_norm + EPS),
            "skew_dominant_along_F": int(wf_norm > sf_norm),
            "A_proxy": aproxy if jt_available else math.nan,
            "S_proxy": sproxy if jt_available else math.nan,
            "jt_proxy_available": jt_available,
            "cross_player_coupling_proxy": cross,
            "same_player_proxy": same,
            "cross_to_same_ratio": cross / (same + EPS),
        }

    def output_geometry(self, diag_batch: dict[str, torch.Tensor]) -> dict[str, float]:
        states = diag_batch["obs"][: min(OUTPUT_DIAG_BATCH, diag_batch["obs"].shape[0])]
        theta_u = self.actor_protagonist(self.theta, states).detach().clone().requires_grad_(True)
        phi_w = self.actor_adversary(self.phi, states).detach().clone().requires_grad_(True)

        def q_batch(u_var: torch.Tensor, w_var: torch.Tensor) -> torch.Tensor:
            return self.q_t(states, u_var, w_var).mean()

        q_val = q_batch(theta_u, phi_w)
        g_u, g_w = torch.autograd.grad(q_val, (theta_u, phi_w), create_graph=True)
        u_flat = theta_u.reshape(-1)
        w_flat = phi_w.reshape(-1)
        gu_flat = g_u.reshape(-1)
        gw_flat = g_w.reshape(-1)
        du = u_flat.numel()
        dw = w_flat.numel()
        h_uu = torch.zeros((du, du), dtype=DTYPE)
        h_uw = torch.zeros((du, dw), dtype=DTYPE)
        h_wu = torch.zeros((dw, du), dtype=DTYPE)
        h_ww = torch.zeros((dw, dw), dtype=DTYPE)
        for i in range(du):
            grads = torch.autograd.grad(gu_flat[i], (theta_u, phi_w), retain_graph=True)
            h_uu[i] = grads[0].reshape(-1).detach().cpu()
            h_uw[i] = grads[1].reshape(-1).detach().cpu()
        for i in range(dw):
            grads = torch.autograd.grad(gw_flat[i], (theta_u, phi_w), retain_graph=True)
            h_wu[i] = grads[0].reshape(-1).detach().cpu()
            h_ww[i] = grads[1].reshape(-1).detach().cpu()
        j_out = torch.zeros((du + dw, du + dw), dtype=DTYPE)
        j_out[:du, :du] = -h_uu
        j_out[:du, du:] = -h_uw
        j_out[du:, :du] = +h_wu
        j_out[du:, du:] = +h_ww
        output_a = 0.5 * (j_out - j_out.T)
        output_s = 0.5 * (j_out + j_out.T)
        eigvals = np.linalg.eigvals(j_out.numpy().astype(np.float64))
        imag = np.abs(np.imag(eigvals))
        return {
            "output_rotation_ratio": float(torch.linalg.norm(output_a).item() / (torch.linalg.norm(output_s).item() + EPS)),
            "output_cross_coupling_norm": float(torch.linalg.norm(h_uw).item()),
            "output_cross_to_diag_ratio": float(torch.linalg.norm(h_uw).item() / (torch.linalg.norm(h_uu).item() + torch.linalg.norm(h_ww).item() + EPS)),
            "output_num_complex_eigs": int(np.sum(imag > 1e-8)),
            "output_max_imag_eig": float(np.max(imag)) if imag.size else 0.0,
            "output_mean_abs_imag_eig": float(np.mean(imag)) if imag.size else 0.0,
        }

    def evaluate_actor(self, theta: torch.Tensor, phi: torch.Tensor | None, episodes: int) -> dict[str, float]:
        env = base.gym.make(self.info.env_name)
        clean_returns: list[float] = []
        adv_returns: list[float] = []
        clip_fracs: list[float] = []
        for ep in range(episodes):
            obs, _ = env.reset(seed=SEED + 9000 + ep)
            done = False
            total = 0.0
            clips: list[float] = []
            while not done:
                obs_t = torch.as_tensor(obs, dtype=DTYPE, device=DEVICE).unsqueeze(0)
                u = self.actor_protagonist(theta, obs_t).squeeze(0)
                if phi is None:
                    if self.cfg.wrapper_type.startswith("action_disturbance"):
                        a_env = torch.clamp(u, self.wrapper.action_low, self.wrapper.action_high)
                        clip_frac = float(((u - a_env).abs() > 1e-12).float().mean().item())
                    else:
                        a_env = torch.clamp(u, self.wrapper.action_low, self.wrapper.action_high)
                        clip_frac = float(((u - a_env).abs() > 1e-12).float().mean().item())
                    w = torch.zeros((self.cfg.adv_dim,), dtype=DTYPE, device=DEVICE)
                else:
                    w = self.actor_adversary(phi, obs_t).squeeze(0)
                    if self.cfg.wrapper_type.startswith("action_disturbance"):
                        a_env, clip_frac = self.wrapper.action_step(u, w)
                    else:
                        force_norm, force_max = self.wrapper.apply_force(env, w)
                        a_env = torch.clamp(u, self.wrapper.action_low, self.wrapper.action_high)
                        clip_frac = 0.0
                try:
                    next_obs, reward, terminated, truncated, _ = env.step(a_env.detach().cpu().numpy().astype(np.float32))
                finally:
                    if not self.cfg.wrapper_type.startswith("action_disturbance"):
                        self.wrapper.clear_force(env)
                # Standard RARL reports undiscounted episode task return.
                # Discounting remains in Bellman targets and the Q/MC audit.
                total += float(reward)
                clips.append(clip_frac)
                obs = np.asarray(next_obs, dtype=np.float32)
                done = terminated or truncated
            (adv_returns if phi is not None else clean_returns).append(total)
            clip_fracs.append(float(np.mean(clips)) if clips else 0.0)
        env.close()
        if phi is None:
            return {"return": float(np.mean(clean_returns)), "clip": float(np.mean(clip_fracs))}
        return {"return": float(np.mean(adv_returns)), "clip": float(np.mean(clip_fracs))}

    def robust_br(self, theta: torch.Tensor, diag_batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, bool]:
        base_phi = self.phi.detach().clone()
        current = base_phi.clone()
        states = diag_batch["obs"][: min(128, diag_batch["obs"].shape[0])]
        for _ in range(BR_INNER_STEPS):
            cur = current.detach().clone().requires_grad_(True)
            z = torch.cat([theta.detach(), cur])
            objective = self.actor_objective(z, states)
            grad = torch.autograd.grad(objective, cur)[0]
            next_phi = cur - (BR_INNER_LR * grad)
            delta = next_phi - base_phi
            norm = torch.linalg.norm(delta)
            if norm > LOCAL_BR_RADIUS:
                next_phi = base_phi + (delta * (LOCAL_BR_RADIUS / (norm + EPS)))
            current = next_phi.detach()
        return current, True


def preflight_for_env(info: EnvInfo) -> tuple[list[dict[str, Any]], list[WrapperConfig]]:
    rows: list[dict[str, Any]] = []
    selected_cfgs: list[WrapperConfig] = []
    if not info.available:
        return rows, selected_cfgs
    env = base.gym.make(info.env_name)
    candidate_wrappers: list[tuple[str, list[float], int]] = []
    candidate_wrappers.append(("action_disturbance_direct", ACTION_ALPHA_GRID, info.action_dim))
    if info.action_dim is not None and info.action_dim >= 2:
        candidate_wrappers.append(("action_disturbance_rotated", ACTION_ALPHA_GRID, info.action_dim))
    if info.has_xfrc_applied and info.main_body_id is not None:
        candidate_wrappers.append(("mujoco_external_force_xz", FORCE_SCALE_GRID, 2))
        candidate_wrappers.append(("mujoco_external_force_xyz", FORCE_SCALE_GRID, 3))
    for wrapper_type, strengths, adv_dim in candidate_wrappers:
        valid_strengths: list[dict[str, Any]] = []
        for strength in strengths:
            cfg = WrapperConfig(wrapper_type, strength, ("alpha" if "action" in wrapper_type else "force") + f"_{strength}", adv_dim, info.main_body_id, info.main_body_name)
            game = SurveyGame(info, cfg, SEED)
            clean_eps = []
            adv_eps = []
            for _ in range(5):
                clean_eps.append(game.evaluate_random_episode(env, adv_on=False, noise_std=0.2))
                adv_eps.append(game.evaluate_random_episode(env, adv_on=True, noise_std=0.2))
            mean_clean = float(np.mean([ep["return"] for ep in clean_eps]))
            mean_adv = float(np.mean([ep["return"] for ep in adv_eps]))
            degr = mean_clean - mean_adv
            degr_ratio = max(0.0, degr) / (abs(mean_clean) + EPS)
            early_term = float(np.mean([ep["early_termination"] for ep in adv_eps]))
            clip_frac = float(np.mean([ep["clip_frac"] for ep in adv_eps]))
            force_norm_mean = float(np.mean([ep["force_norm_mean"] for ep in adv_eps]))
            force_norm_max = float(np.max([ep["force_norm_max"] for ep in adv_eps]))
            state_ok = all(ep["state_finite"] == 1 for ep in adv_eps)
            step_ok = not any(ep["step_error"] == 1 for ep in adv_eps)
            row = {
                "env_name": info.env_name,
                "wrapper_type": wrapper_type,
                "strength_name": cfg.strength_name,
                "alpha_or_force_scale": strength,
                "mean_return_clean_random": mean_clean,
                "mean_return_adv_random": mean_adv,
                "return_degradation": degr,
                "return_degradation_ratio": degr_ratio,
                "mean_episode_length": float(np.mean([ep["episode_length"] for ep in adv_eps])),
                "early_termination_rate": early_term,
                "action_clip_fraction": clip_frac,
                "force_norm_mean": force_norm_mean,
                "force_norm_max": force_norm_max,
                "state_finite_flag": int(state_ok),
                "step_error_flag": int(not step_ok),
            }
            rows.append(row)
            valid = (
                degr_ratio >= 0.05
                and early_term <= 0.8
                and state_ok
                and step_ok
                and (clip_frac <= 0.2 if wrapper_type.startswith("action_disturbance") else True)
            )
            if valid:
                valid_strengths.append(row)
        if valid_strengths:
            smallest = min(valid_strengths, key=lambda r: float(r["alpha_or_force_scale"]))
            largest = max(valid_strengths, key=lambda r: float(r["alpha_or_force_scale"]))
            selected_cfgs.append(
                WrapperConfig(wrapper_type, float(smallest["alpha_or_force_scale"]), str(smallest["strength_name"]), adv_dim, info.main_body_id, info.main_body_name)
            )
            if float(largest["alpha_or_force_scale"]) != float(smallest["alpha_or_force_scale"]):
                selected_cfgs.append(
                    WrapperConfig(wrapper_type, float(largest["alpha_or_force_scale"]), str(largest["strength_name"]), adv_dim, info.main_body_id, info.main_body_name)
                )
    env.close()
    return rows, selected_cfgs


def survey_geometry_for_config(info: EnvInfo, cfg: WrapperConfig) -> tuple[dict[str, Any], list[dict[str, Any]], SurveyGame, dict[str, torch.Tensor]]:
    warmup_steps = WARMUP_MUJOCO if info.has_mujoco else WARMUP_SIMPLE
    best_game = None
    best_stats = None
    best_noise = None
    for noise_std in [0.1, 0.2]:
        game = SurveyGame(info, cfg, SEED)
        stats = game.collect_replay(warmup_steps, noise_std)
        key = (0 if stats["step_error_flag"] == 0 and stats["state_finite_flag"] == 1 else 1, 0 if stats["action_clip_fraction"] <= 0.2 else 1, -stats["std_w"])
        if best_stats is None or key < (
            0 if best_stats["step_error_flag"] == 0 and best_stats["state_finite_flag"] == 1 else 1,
            0 if best_stats["action_clip_fraction"] <= 0.2 else 1,
            -best_stats["std_w"],
        ):
            best_game = game
            best_stats = stats
            best_noise = noise_std
    assert best_game is not None and best_stats is not None and best_noise is not None
    snapshots = best_game.collect_snapshots(MC_SNAPSHOTS)
    critic_rows: list[dict[str, Any]] = []
    for step in range(CRITIC_TRAIN_STEPS):
        stats = best_game.critic_update()
        if step % CRITIC_AUDIT_INTERVAL == 0 or step == CRITIC_TRAIN_STEPS - 1:
            quality = best_game.mc_quality(snapshots)
            critic_rows.append({**stats, **quality, "critic_step": step})
    diag_batch = best_game.replay.fixed_state_batch(DIAG_BATCH_SIZE, np.random.default_rng(SEED + 7777))
    z = best_game.current_z()
    diag = best_game.diagnostic_metrics(z, diag_batch, actor_lr=1e-5, compute_geometry=True)
    out_geom = best_game.output_geometry(diag_batch)
    row = {
        "env_name": info.env_name,
        "wrapper_type": cfg.wrapper_type,
        "strength_name": cfg.strength_name,
        "alpha_or_force_scale": cfg.strength_value,
        "warmup_steps": warmup_steps,
        "warmup_noise_std": best_noise,
        **best_stats,
        **critic_rows[-1],
        **diag,
        **out_geom,
        "critic_healthy": int(
            finite(critic_rows[-1]["critic_loss"])
            and finite(critic_rows[-1]["Q_abs_mean"])
            and critic_rows[-1]["Q_abs_mean"] < 1e6
            and best_stats["state_finite_flag"] == 1
            and best_stats["step_error_flag"] == 0
        ),
    }
    return row, critic_rows, best_game, diag_batch


def short_sanity(game: SurveyGame, diag_batch: dict[str, torch.Tensor], base_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best_lr = None
    best_sgd_auc = None
    summaries: dict[tuple[str, float], dict[str, float]] = {}
    for lr in ACTOR_LR_GRID:
        curves, summary = short_run(game, diag_batch, "sgd", lr)
        summaries[("sgd", lr)] = summary
        rows.extend(curves)
        valid = summary["valid_flag"] == 1 and summary["curve_normal_flag"] == 1
        if valid and (best_sgd_auc is None or summary["V_std_AUC"] < best_sgd_auc):
            best_lr = lr
            best_sgd_auc = summary["V_std_AUC"]
    if best_lr is not None:
        for method in ["egm", "ppm"]:
            curves, summary = short_run(game, diag_batch, method, best_lr)
            summaries[(method, best_lr)] = summary
            rows.extend(curves)
    return rows


def short_run(game_src: SurveyGame, diag_batch: dict[str, torch.Tensor], method: str, actor_lr: float) -> tuple[list[dict[str, Any]], dict[str, float]]:
    game = SurveyGame(game_src.info, game_src.cfg, SEED)
    game.theta = game_src.theta.detach().clone()
    game.phi = game_src.phi.detach().clone()
    game.theta_target = game_src.theta_target.detach().clone()
    game.phi_target = game_src.phi_target.detach().clone()
    game.q_t.load_state_dict(game_src.q_t.state_dict())
    game.q_t_target.load_state_dict(game_src.q_t_target.state_dict())
    game.metric_refs = {}
    curves: list[dict[str, Any]] = []
    clean_eval = game.evaluate_actor(game.theta, None, 3)
    adv_eval = game.evaluate_actor(game.theta, game.phi, 3)
    br_phi, br_valid = game.robust_br(game.theta, diag_batch)
    br_eval = game.evaluate_actor(game.theta, br_phi, 3)
    for iteration in range(SHORT_ACTOR_ITERS + 1):
        z = game.current_z()
        diag = game.diagnostic_metrics(z, diag_batch, actor_lr, compute_geometry=False)
        if iteration % 10 == 0 or iteration == SHORT_ACTOR_ITERS:
            clean_eval = game.evaluate_actor(game.theta, None, 3)
            adv_eval = game.evaluate_actor(game.theta, game.phi, 3)
            br_phi, br_valid = game.robust_br(game.theta, diag_batch)
            br_eval = game.evaluate_actor(game.theta, br_phi, 3)
        row = {
            "env_name": game.info.env_name,
            "wrapper_type": game.cfg.wrapper_type,
            "strength_name": game.cfg.strength_name,
            "alpha_or_force_scale": game.cfg.strength_value,
            "method": method,
            "actor_lr": actor_lr,
            "iteration": iteration,
            **diag,
            "clean_task_return": clean_eval["return"],
            "current_adv_task_return": adv_eval["return"],
            "robust_br_task_return": br_eval["return"],
            "robust_degradation": clean_eval["return"] - br_eval["return"],
            "valid_flag": int(finite(diag["V_std"]) and finite(diag["field_norm"]) and br_valid),
        }
        curves.append(row)
        if iteration < SHORT_ACTOR_ITERS:
            next_z, meta = short_update(game, z, diag_batch["obs"], method, actor_lr)
            game.set_from_z(next_z)
            curves[-1].update(meta)
    vals_v = [row["V_std"] for row in curves]
    vals_p = [row["P_tau_std"] for row in curves]
    vals_f = [row["field_norm"] for row in curves]
    summary = {
        "method": method,
        "actor_lr": actor_lr,
        "valid_flag": int(all(row["valid_flag"] == 1 for row in curves)),
        "curve_normal_flag": int(curves[-1]["V_std"] <= curves[0]["V_std"] + 1e-8 and curves[-1]["P_tau_std"] <= curves[0]["P_tau_std"] + 1e-8 and spike_ratio(vals_v) <= 5.0 and spike_ratio(vals_p) <= 5.0),
        "V_std_AUC": float(sum(vals_v)),
        "P_tau_AUC": float(sum(vals_p)),
        "field_norm_AUC": float(sum(vals_f)),
        "robust_br_task_return_final": curves[-1]["robust_br_task_return"],
    }
    return curves, summary


def short_update(game: SurveyGame, z: torch.Tensor, states: torch.Tensor, method: str, actor_lr: float) -> tuple[torch.Tensor, dict[str, float]]:
    field = game.actor_field(z, states).detach()
    if method == "sgd":
        delta = -actor_lr * field
        return (z + delta).detach(), {"update_norm": float(torch.linalg.norm(delta).item())}
    if method == "egm":
        z_half = z - (actor_lr * field)
        field_half = game.actor_field(z_half, states).detach()
        delta = -actor_lr * field_half
        return (z + delta).detach(), {"update_norm": float(torch.linalg.norm(delta).item())}
    if method == "ppm":
        current = z.detach().clone()
        for _ in range(5):
            field_inner = game.actor_field(current, states).detach()
            current = z - (actor_lr * field_inner)
        delta = current - z
        return current.detach(), {"update_norm": float(torch.linalg.norm(delta).item())}
    raise ValueError(method)


def label_candidate(row: dict[str, Any], sanity_rows: list[dict[str, Any]] | None) -> str:
    if row.get("available") == 0:
        return "UNAVAILABLE"
    if row.get("critic_healthy", 1) == 0:
        return "CRITIC_UNRELIABLE"
    if row.get("step_error_flag", 0) == 1 or row.get("state_finite_flag", 1) == 0:
        return "WRAPPER_UNSTABLE"
    rotation = float(row.get("rotation_ratio_proxy", math.nan))
    out_rotation = float(row.get("output_rotation_ratio", math.nan))
    cross_ratio = float(row.get("cross_to_same_ratio", math.nan))
    complex_eigs = float(row.get("output_num_complex_eigs", 0.0))
    sanity_adv = math.nan
    if sanity_rows:
        sgd = min((r for r in sanity_rows if r["method"] == "sgd"), key=lambda r: float(r["actor_lr"]), default=None)
    if (finite(out_rotation) and out_rotation >= 1e-2) or (finite(rotation) and rotation >= 1e-3):
        if cross_ratio >= 0.1:
            return "STRONG_SKEW_CANDIDATE"
        return "MODERATE_SKEW_CANDIDATE"
    if cross_ratio >= 0.05:
        return "WEAK_GEOMETRY"
    if (not finite(rotation) or rotation < 1e-4) and cross_ratio < 0.05 and complex_eigs <= 0:
        return "POTENTIAL_LIKE"
    return "WEAK_GEOMETRY"


def plot_rankings(rows: list[dict[str, Any]]) -> None:
    if plt is None or not rows:
        return
    def top(metric: str, n: int = 12):
        return sorted([row for row in rows if finite(float(row.get(metric, math.nan)))], key=lambda r: float(r[metric]), reverse=True)[:n]
    plots = [
        ("rotation_ratio_proxy", "Parameter Rotation Ratio", "survey_rotation_ratio_rank.png"),
        ("cross_to_same_ratio", "Cross-to-Same Ratio", "survey_cross_coupling_rank.png"),
        ("output_rotation_ratio", "Output Rotation Ratio", "survey_output_rotation_rank.png"),
        ("geometry_score", "Geometry Score", "survey_geometry_score_rank.png"),
    ]
    for metric, title, filename in plots:
        top_rows = top(metric)
        if not top_rows:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.text(0.5, 0.5, f"No finite values for {metric}", ha="center", va="center")
            ax.set_axis_off()
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(PLOT_ROOT / filename, dpi=180)
            plt.close(fig)
            continue
        labels = [f"{row['env_name']}|{row['wrapper_type']}|{row['strength_name']}" for row in top_rows]
        vals = [float(row[metric]) for row in top_rows]
        fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.4)))
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
    infos = environment_availability()
    preflight_rows: list[dict[str, Any]] = []
    selected_configs: list[tuple[EnvInfo, WrapperConfig]] = []
    for info in infos:
        rows, cfgs = preflight_for_env(info)
        preflight_rows.extend(rows)
        selected_configs.extend((info, cfg) for cfg in cfgs)
    write_csv(RESULT_ROOT / "survey_preflight_summary.csv", preflight_rows)
    lines = ["# survey_preflight_report", ""]
    by_env = {}
    for row in preflight_rows:
        by_env.setdefault((row["env_name"], row["wrapper_type"]), []).append(row)
    for key, rows in by_env.items():
        valid = [r for r in rows if float(r["return_degradation_ratio"]) >= 0.05 and float(r["early_termination_rate"]) <= 0.8 and int(r["state_finite_flag"]) == 1 and int(r["step_error_flag"]) == 0 and (float(r["action_clip_fraction"]) <= 0.2 if "action_disturbance" in key[1] else True)]
        lines.append(f"- {key[0]} / {key[1]}: valid_strengths=`{[r['alpha_or_force_scale'] for r in valid]}`")
    write_text(RESULT_ROOT / "survey_preflight_report.md", "\n".join(lines) + "\n")

    geometry_rows: list[dict[str, Any]] = []
    critic_rows: list[dict[str, Any]] = []
    game_cache: dict[str, tuple[SurveyGame, dict[str, torch.Tensor], dict[str, Any]]] = {}
    for info, cfg in selected_configs:
        row, qrows, game, diag_batch = survey_geometry_for_config(info, cfg)
        key = f"{info.env_name}|{cfg.wrapper_type}|{cfg.strength_name}"
        game_cache[key] = (game, diag_batch, row)
        geometry_rows.append(row)
        for qrow in qrows:
            critic_rows.append({**qrow, "env_name": info.env_name, "wrapper_type": cfg.wrapper_type, "strength_name": cfg.strength_name, "alpha_or_force_scale": cfg.strength_value})
    write_csv(RESULT_ROOT / "survey_geometry_summary.csv", geometry_rows)
    write_csv(RESULT_ROOT / "survey_critic_quality.csv", critic_rows)

    prelim_scores = {}
    rank_rot = rank_norm({f"{r['env_name']}|{r['wrapper_type']}|{r['strength_name']}": float(r["rotation_ratio_proxy"]) for r in geometry_rows})
    rank_out = rank_norm({f"{r['env_name']}|{r['wrapper_type']}|{r['strength_name']}": float(r["output_rotation_ratio"]) for r in geometry_rows})
    rank_cross = rank_norm({f"{r['env_name']}|{r['wrapper_type']}|{r['strength_name']}": float(r["cross_to_same_ratio"]) for r in geometry_rows})
    rank_noncol = rank_norm({f"{r['env_name']}|{r['wrapper_type']}|{r['strength_name']}": float(r["non_collinearity"]) for r in geometry_rows})
    rank_complex = rank_norm({f"{r['env_name']}|{r['wrapper_type']}|{r['strength_name']}": float(r["output_num_complex_eigs"]) + float(r["output_max_imag_eig"]) for r in geometry_rows})
    for row in geometry_rows:
        key = f"{row['env_name']}|{row['wrapper_type']}|{row['strength_name']}"
        prelim_scores[key] = (
            0.25 * rank_rot.get(key, 0.0)
            + 0.25 * rank_out.get(key, 0.0)
            + 0.20 * rank_cross.get(key, 0.0)
            + 0.10 * rank_noncol.get(key, 0.0)
            + 0.10 * rank_complex.get(key, 0.0)
        )
    top_configs = sorted(geometry_rows, key=lambda r: prelim_scores[f"{r['env_name']}|{r['wrapper_type']}|{r['strength_name']}"], reverse=True)[:TOP_K_SANITY]
    sanity_rows_all: list[dict[str, Any]] = []
    short_summary_map: dict[str, dict[str, float]] = {}
    for row in top_configs:
        key = f"{row['env_name']}|{row['wrapper_type']}|{row['strength_name']}"
        game, diag_batch, _ = game_cache[key]
        sanity_rows = short_sanity(game, diag_batch, row)
        sanity_rows_all.extend(sanity_rows)
        sgd_rows = [r for r in sanity_rows if r["method"] == "sgd"]
        best_lr = None
        best_sgd_auc = None
        for lr in ACTOR_LR_GRID:
            sub = [r for r in sgd_rows if float(r["actor_lr"]) == float(lr)]
            if not sub:
                continue
            auc = float(sum(r["V_std"] for r in sub))
            valid = sub[-1]["V_std"] <= sub[0]["V_std"] + 1e-8
            if valid and (best_sgd_auc is None or auc < best_sgd_auc):
                best_sgd_auc = auc
                best_lr = lr
        if best_lr is not None:
            best_method_aucs = {}
            for method in ["sgd", "egm", "ppm"]:
                sub = [r for r in sanity_rows if r["method"] == method and float(r["actor_lr"]) == float(best_lr)]
                if sub:
                    best_method_aucs[method] = float(sum(r["V_std"] for r in sub))
            if {"sgd", "egm", "ppm"} <= best_method_aucs.keys():
                short_summary_map[key] = {
                    "short_SGD_V_AUC": best_method_aucs["sgd"],
                    "short_EGM_V_AUC": best_method_aucs["egm"],
                    "short_PPM_V_AUC": best_method_aucs["ppm"],
                    "short_baseline_advantage": best_method_aucs["sgd"] / (min(best_method_aucs["egm"], best_method_aucs["ppm"]) + EPS),
                }
    write_csv(RESULT_ROOT / "survey_short_baseline_sanity.csv", sanity_rows_all)

    rank_adv = rank_norm({k: v["short_baseline_advantage"] for k, v in short_summary_map.items() if finite(v["short_baseline_advantage"])})
    ranked_rows: list[dict[str, Any]] = []
    for row in geometry_rows:
        key = f"{row['env_name']}|{row['wrapper_type']}|{row['strength_name']}"
        score = prelim_scores.get(key, 0.0) + 0.10 * rank_adv.get(key, 0.0)
        summary = short_summary_map.get(
            key,
            {
                "short_SGD_V_AUC": math.nan,
                "short_EGM_V_AUC": math.nan,
                "short_PPM_V_AUC": math.nan,
                "short_baseline_advantage": math.nan,
            },
        )
        out = dict(row)
        out.update(summary)
        out["geometry_score"] = score
        out["recommended_next_step"] = label_candidate(out, None)
        ranked_rows.append(out)
    ranked_rows.sort(key=lambda r: float(r["geometry_score"]), reverse=True)
    for idx, row in enumerate(ranked_rows, start=1):
        row["rank"] = idx
    write_csv(RESULT_ROOT / "survey_ranked_candidates.csv", ranked_rows)
    plot_rankings(ranked_rows)

    final_lines = [
        "# survey_final_report",
        "",
        "1. Purpose of survey",
        "   Identify standard Gym/Gymnasium/MuJoCo continuous-control environments and adversary wrappers whose frozen-critic local actor field under the standard environment reward exhibits nontrivial skew/rotational geometry.",
        "",
        "2. Available/unavailable environments",
    ]
    for info in infos:
        final_lines.append(f"   - {info.env_name}: {'available' if info.available else 'unavailable'}")
    final_lines.extend(["", "3. Best candidates by geometry score"])
    for row in ranked_rows[:5]:
        final_lines.append(
            f"   - rank {row['rank']}: {row['env_name']} / {row['wrapper_type']} / {row['strength_name']} / score={float(row['geometry_score']):.3f} / label={row['recommended_next_step']}"
        )
    final_lines.extend(["", "4. Best candidates by output-space rotation"])
    for row in sorted(ranked_rows, key=lambda r: float(r["output_rotation_ratio"]), reverse=True)[:5]:
        final_lines.append(f"   - {row['env_name']} / {row['wrapper_type']} / {row['strength_name']} / output_rotation_ratio={float(row['output_rotation_ratio']):.6e}")
    final_lines.extend(["", "5. Best candidates by parameter-space rotation"])
    for row in sorted(ranked_rows, key=lambda r: float(r["rotation_ratio_proxy"]) if finite(float(r["rotation_ratio_proxy"])) else -1.0, reverse=True)[:5]:
        final_lines.append(f"   - {row['env_name']} / {row['wrapper_type']} / {row['strength_name']} / rotation_ratio_proxy={float(row['rotation_ratio_proxy']):.6e}")
    final_lines.extend(["", "6. Best candidates by short EGM/PPM advantage"])
    for row in sorted([r for r in ranked_rows if finite(float(r['short_baseline_advantage']))], key=lambda r: float(r["short_baseline_advantage"]), reverse=True)[:5]:
        final_lines.append(f"   - {row['env_name']} / {row['wrapper_type']} / {row['strength_name']} / short_baseline_advantage={float(row['short_baseline_advantage']):.3f}")
    final_lines.extend(["", "7. Potential-like environments"])
    for row in ranked_rows:
        if row["recommended_next_step"] == "POTENTIAL_LIKE":
            final_lines.append(f"   - {row['env_name']} / {row['wrapper_type']} / {row['strength_name']}")
    final_lines.extend(["", "8. Unstable wrappers"])
    for row in preflight_rows:
        if int(row["step_error_flag"]) == 1 or int(row["state_finite_flag"]) == 0:
            final_lines.append(f"   - {row['env_name']} / {row['wrapper_type']} / {row['strength_name']}")
    final_lines.extend(["", "9. Recommendation for the next full Stage-1/Stage-2 experiment"])
    if ranked_rows:
        best = ranked_rows[0]
        final_lines.append(
            f"   - Start with {best['env_name']} / {best['wrapper_type']} / {best['strength_name']} if its label is at least MODERATE_SKEW_CANDIDATE; otherwise no strong standard-environment skew candidate was found in this survey."
        )
    write_text(RESULT_ROOT / "survey_final_report.md", "\n".join(final_lines) + "\n")

    top_lines = ["# survey_top_candidates", ""]
    for row in ranked_rows[:3]:
        critic_loss_final = row.get("critic_loss_final", row.get("critic_loss", math.nan))
        top_lines.extend(
            [
                f"## Rank {row['rank']}: {row['env_name']} / {row['wrapper_type']} / {row['strength_name']}",
                f"- geometry_score: `{float(row['geometry_score']):.3f}`",
                f"- label: `{row['recommended_next_step']}`",
                f"- why selected: `rotation_ratio_proxy={float(row['rotation_ratio_proxy']):.6e}, output_rotation_ratio={float(row['output_rotation_ratio']):.6e}, cross_to_same_ratio={float(row['cross_to_same_ratio']):.6e}, short_baseline_advantage={float(row['short_baseline_advantage']) if finite(float(row['short_baseline_advantage'])) else math.nan}`",
                f"- risks: `critic_loss_final={float(critic_loss_final):.6e}, corr_Q_MC={float(row['corr_Q_MC']) if finite(float(row['corr_Q_MC'])) else math.nan}, early_termination_rate={float(row['early_termination_rate']) if finite(float(row['early_termination_rate'])) else math.nan}, action_clip_fraction={float(row['action_clip_fraction']) if finite(float(row['action_clip_fraction'])) else math.nan}`",
                f"- next settings: `env={row['env_name']}, wrapper={row['wrapper_type']}, strength={row['alpha_or_force_scale']}`",
                "",
            ]
        )
    write_text(RESULT_ROOT / "survey_top_candidates.md", "\n".join(top_lines) + "\n")


if __name__ == "__main__":
    main()
