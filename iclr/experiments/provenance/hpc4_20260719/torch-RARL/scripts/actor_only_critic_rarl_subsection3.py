from __future__ import annotations

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
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import gymnasium as gym
    GYM_BACKEND = "gymnasium"
except Exception:
    import gym
    GYM_BACKEND = "gym"


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT.parent / "results" / "actor_only_critic_rarl_subsection3"
PREFIX_BASE = "actor_only_critic_s3_"

DEVICE = torch.device("cpu")
DTYPE = torch.float32
EPS = 1e-8
SEED = 0

ENV_CANDIDATES = ["Reacher-v5", "LunarLanderContinuous-v3", "LunarLanderContinuous-v2"]

ACTOR_HIDDEN = (64, 64)
CRITIC_HIDDEN = (128, 128)
EXPLORATION_STD = 0.1
GAMMA = 0.99
POLYAK_TAU = 0.005
CRITIC_LR = 1e-3
CRITIC_UPDATES_PER_ITER = 1
TRAIN_BATCH_SIZE = 256
REPLAY_WARMUP_STEPS = 5000
ROLLOUT_STEPS_PER_ITER = 512
NUM_OUTER_ITERATIONS = 200
EVAL_EVERY = 10
P_TAU_EVAL_EVERY = 1
PPM_INNER_STEPS = 5

ACTOR_LR_GRID = [1e-5, 3e-5, 1e-4, 3e-4]
ALPHA_GRID = [0.01, 0.02, 0.05, 0.1]
BETA_ROT_GRID = [0.0, 0.01, 0.03, 0.1]
UPDATE_RADIUS_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]

LAMBDA_F = 0.01
LAMBDA_P = 1.0
GAP_INNER_STEPS = 3
LOCAL_GAP_RADIUS = 0.1
BR_INNER_STEPS = 20
BR_INNER_LR = 3e-4
LOCAL_BR_RADIUS = 0.25
ROTATION_RANDOM_VECS = 4

SGD_GATE_DECISIONS = ("SGD_NORMAL_PASS", "SGD_NORMAL_FAIL")
BASELINE_DECISIONS = (
    "BASELINE_READY_FOR_PROPOSED",
    "BASELINE_FAIL_CURVES_ABNORMAL",
    "BASELINE_FAIL_NO_EGM_PPM_ADVANTAGE",
    "BASELINE_FAIL_GEOMETRY_WEAK",
    "BASELINE_FAIL_RARL_PERFORMANCE",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def finite(value: float) -> bool:
    return math.isfinite(float(value))


def moving_average_slope(values: list[float], window: int = 5) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    if arr.size < window:
        xs = np.arange(arr.size, dtype=np.float64)
        return float(np.polyfit(xs, arr, 1)[0])
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smooth = np.convolve(arr, kernel, mode="valid")
    xs = np.arange(smooth.size, dtype=np.float64)
    return float(np.polyfit(xs, smooth, 1)[0]) if smooth.size > 1 else 0.0


def spike_ratio(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.max() / (np.median(arr) + EPS))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def ensure_dirs() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "plots").mkdir(parents=True, exist_ok=True)


def slugify_env(env_id: str) -> str:
    return env_id.lower().replace("-", "_")


class FlatMLP:
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...], output_dim: int) -> None:
        shapes: list[tuple[int, ...]] = []
        prev = input_dim
        for hidden in hidden_sizes:
            shapes.extend([(hidden, prev), (hidden,)])
            prev = hidden
        shapes.extend([(output_dim, prev), (output_dim,)])
        self.shapes = shapes
        self.num_params = sum(int(np.prod(shape)) for shape in self.shapes)

    def init_flat(self, generator: torch.Generator, final_scale: float) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        num_layers = len(self.shapes) // 2
        for idx, shape in enumerate(self.shapes):
            if len(shape) == 2:
                scale = 1.0 / math.sqrt(shape[1])
                tensor = torch.randn(shape, generator=generator, dtype=DTYPE, device=DEVICE) * scale
            else:
                tensor = 0.01 * torch.randn(shape, generator=generator, dtype=DTYPE, device=DEVICE)
            if idx >= (2 * (num_layers - 1)):
                tensor = tensor * final_scale
            chunks.append(tensor.reshape(-1))
        return torch.cat(chunks)

    def unpack(self, flat: torch.Tensor) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        offset = 0
        for shape in self.shapes:
            size = int(np.prod(shape))
            out.append(flat[offset : offset + size].reshape(shape))
            offset += size
        return out

    def forward(self, flat: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        tensors = self.unpack(flat)
        x = obs
        layers = len(tensors) // 2
        for layer in range(layers - 1):
            w = tensors[2 * layer]
            b = tensors[(2 * layer) + 1]
            x = torch.tanh(x @ w.T + b)
        w = tensors[-2]
        b = tensors[-1]
        return x @ w.T + b


class CriticNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        input_dim = obs_dim + (2 * action_dim)
        self.fc1 = nn.Linear(input_dim, CRITIC_HIDDEN[0])
        self.fc2 = nn.Linear(CRITIC_HIDDEN[0], CRITIC_HIDDEN[1])
        self.fc3 = nn.Linear(CRITIC_HIDDEN[1], 1)

    def forward(self, obs: torch.Tensor, u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, u, w], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.u = np.zeros((capacity, action_dim), dtype=np.float32)
        self.w = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward_game = np.zeros((capacity,), dtype=np.float32)
        self.reward_task_scaled = np.zeros((capacity,), dtype=np.float32)
        self.reward_task_raw = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        u: np.ndarray,
        w: np.ndarray,
        reward_game: float,
        reward_task_scaled: float,
        reward_task_raw: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.ptr
        self.obs[idx] = obs
        self.u[idx] = u
        self.w[idx] = w
        self.reward_game[idx] = reward_game
        self.reward_task_scaled[idx] = reward_task_scaled
        self.reward_task_raw[idx] = reward_task_raw
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
            "reward_game": torch.as_tensor(self.reward_game[idx], dtype=DTYPE, device=DEVICE),
            "reward_task_scaled": torch.as_tensor(self.reward_task_scaled[idx], dtype=DTYPE, device=DEVICE),
            "reward_task_raw": torch.as_tensor(self.reward_task_raw[idx], dtype=DTYPE, device=DEVICE),
            "next_obs": torch.as_tensor(self.next_obs[idx], dtype=DTYPE, device=DEVICE),
            "done": torch.as_tensor(self.done[idx], dtype=DTYPE, device=DEVICE),
        }

    def fixed_state_batch(self, batch_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        idx = rng.integers(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx], dtype=DTYPE, device=DEVICE),
        }


@dataclass(frozen=True)
class EnvChoice:
    env_id: str
    obs_dim: int
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray


@dataclass(frozen=True)
class WrapperConfig:
    reward_scale: float
    alpha_dyn: float
    a_u: float
    a_w: float
    beta_rot: float
    beta_sym: float
    use_rot_dyn: bool


def choose_environment() -> EnvChoice:
    rows: list[dict[str, Any]] = []
    selected: EnvChoice | None = None
    for env_id in ENV_CANDIDATES:
        try:
            env = gym.make(env_id)
            act = env.action_space
            obs = env.observation_space
            is_box = act.__class__.__name__ == "Box"
            action_shape = tuple(int(x) for x in getattr(act, "shape", ()))
            row = {
                "env_id": env_id,
                "available": int(is_box and len(action_shape) == 1),
                "backend": GYM_BACKEND,
                "action_space_type": act.__class__.__name__,
                "action_shape": str(action_shape),
                "obs_shape": str(tuple(int(x) for x in obs.shape)),
                "error": "",
            }
            if is_box and len(action_shape) == 1 and selected is None:
                selected = EnvChoice(
                    env_id=env_id,
                    obs_dim=int(obs.shape[0]),
                    action_dim=int(action_shape[0]),
                    action_low=np.asarray(act.low, dtype=np.float32).copy(),
                    action_high=np.asarray(act.high, dtype=np.float32).copy(),
                )
            env.close()
        except Exception as exc:
            row = {
                "env_id": env_id,
                "available": 0,
                "backend": GYM_BACKEND,
                "action_space_type": "",
                "action_shape": "",
                "obs_shape": "",
                "error": repr(exc),
            }
        rows.append(row)
    write_csv(RESULT_ROOT / "actor_only_critic_s3_env_availability.csv", rows)
    if selected is None:
        text = "# actor_only_critic_s3_env_availability_report\n\n- selected_env: `none`\n- status: `fail`\n"
        write_text(RESULT_ROOT / "actor_only_critic_s3_env_availability_report.md", text)
        raise RuntimeError("No supported continuous environment available.")
    text = "\n".join(
        [
            "# actor_only_critic_s3_env_availability_report",
            "",
            f"- backend: `{GYM_BACKEND}`",
            f"- selected_env: `{selected.env_id}`",
            f"- obs_dim: `{selected.obs_dim}`",
            f"- action_dim: `{selected.action_dim}`",
            f"- action_bounds_low: `{selected.action_low.tolist()}`",
            f"- action_bounds_high: `{selected.action_high.tolist()}`",
        ]
    ) + "\n"
    write_text(RESULT_ROOT / "actor_only_critic_s3_env_availability_report.md", text)
    return selected


class NaturalRARLEnv:
    def __init__(self, env_choice: EnvChoice, wrapper_cfg: WrapperConfig) -> None:
        self.env_choice = env_choice
        self.cfg = wrapper_cfg
        self.action_low = torch.as_tensor(env_choice.action_low, dtype=DTYPE, device=DEVICE)
        self.action_high = torch.as_tensor(env_choice.action_high, dtype=DTYPE, device=DEVICE)
        self.action_scale = 0.5 * (self.action_high - self.action_low)
        self.action_bias = 0.5 * (self.action_high + self.action_low)
        self.r_dyn = self.build_r_dyn(env_choice.action_dim)

    def build_r_dyn(self, action_dim: int) -> torch.Tensor:
        mat = torch.zeros((action_dim, action_dim), dtype=DTYPE, device=DEVICE)
        for start in range(0, action_dim - 1, 2):
            mat[start, start + 1] = 1.0
            mat[start + 1, start] = -1.0
        return mat

    def scale_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        return self.action_bias + (self.action_scale * torch.tanh(raw_action))

    def blend_action(self, u: torch.Tensor, w: torch.Tensor, mode_rot: bool) -> tuple[torch.Tensor, float]:
        perturb = (self.r_dyn @ w) if mode_rot else w
        a_env_raw = u + (self.cfg.alpha_dyn * perturb)
        a_env = torch.clamp(a_env_raw, self.action_low, self.action_high)
        clip_fraction = float(((a_env_raw - a_env).abs() > 1e-12).float().mean().item())
        return a_env, clip_fraction

    def reward_terms(self, reward_task_raw: float, u: torch.Tensor, w: torch.Tensor) -> tuple[float, float]:
        r_task_scaled = self.cfg.reward_scale * reward_task_raw
        reg = (-0.5 * self.cfg.a_u * float(torch.sum(u * u).item())) + (0.5 * self.cfg.a_w * float(torch.sum(w * w).item()))
        rot_shape = 0.0
        if self.cfg.beta_rot > 0.0 and self.env_choice.action_dim >= 2:
            h2 = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=DTYPE, device=DEVICE)
            s2 = torch.tensor([[1.0, 0.2], [0.2, -0.5]], dtype=DTYPE, device=DEVICE)
            u2 = u[:2]
            w2 = w[:2]
            rot_shape = (
                self.cfg.beta_rot * float(torch.dot(u2, h2 @ w2).item())
                + self.cfg.beta_sym * float(torch.dot(u2, s2 @ w2).item())
            )
        return r_task_scaled + reg + rot_shape, r_task_scaled


def random_policy_preflight(env_choice: EnvChoice) -> tuple[WrapperConfig, str]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED)
    env = gym.make(env_choice.env_id)
    obs, _ = env.reset(seed=SEED)
    reward_samples: list[float] = []
    u_norms: list[float] = []
    w_norms: list[float] = []
    rot_samples: list[float] = []
    mode_stats: dict[tuple[bool, float], list[float]] = {}
    r_dyn = NaturalRARLEnv(
        env_choice,
        WrapperConfig(reward_scale=1.0, alpha_dyn=0.01, a_u=0.001, a_w=0.001, beta_rot=0.0, beta_sym=0.0, use_rot_dyn=True),
    ).r_dyn.cpu().numpy()
    for _ in range(2048):
        u = rng.uniform(env_choice.action_low, env_choice.action_high).astype(np.float32)
        w = rng.uniform(env_choice.action_low, env_choice.action_high).astype(np.float32)
        reward_samples.append(0.0)
        u_norms.append(float(np.linalg.norm(u)))
        w_norms.append(float(np.linalg.norm(w)))
        if env_choice.action_dim >= 2:
            h2 = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
            rot_samples.append(float(u[:2].T @ (h2 @ w[:2])))
        for use_rot in [True, False]:
            for alpha in ALPHA_GRID:
                perturb = (r_dyn @ w) if use_rot else w
                a_raw = u + (alpha * perturb)
                a_clip = np.clip(a_raw, env_choice.action_low, env_choice.action_high)
                frac = float(np.mean(np.abs(a_raw - a_clip) > 1e-12))
                mode_stats.setdefault((use_rot, alpha), []).append(frac)
        a = np.clip(u, env_choice.action_low, env_choice.action_high)
        obs, reward_raw, terminated, truncated, _ = env.step(a)
        reward_samples[-1] = float(reward_raw)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    std_task = float(np.std(np.asarray(reward_samples, dtype=np.float64)))
    reward_scale = 1.0 / max(std_task, EPS)
    best_mode = True
    best_alpha = ALPHA_GRID[0]
    for use_rot in [True, False]:
        allowed = [alpha for alpha in ALPHA_GRID if float(np.mean(mode_stats[(use_rot, alpha)])) <= 0.05]
        if allowed:
            candidate_alpha = max(allowed)
            if use_rot and env_choice.action_dim >= 2:
                best_mode = True
                best_alpha = candidate_alpha
                break
            if not best_mode:
                best_alpha = max(best_alpha, candidate_alpha)
    for use_rot in [True, False]:
        for alpha in ALPHA_GRID:
            rows.append(
                {
                    "env_id": env_choice.env_id,
                    "reward_scale": reward_scale,
                    "std_r_task_raw": std_task,
                    "std_u_norm": float(np.std(u_norms)),
                    "std_w_norm": float(np.std(w_norms)),
                    "std_rot_term": float(np.std(rot_samples)) if rot_samples else 0.0,
                    "use_rot_dyn": int(use_rot),
                    "alpha_dyn": alpha,
                    "action_clip_fraction": float(np.mean(mode_stats[(use_rot, alpha)])),
                }
            )
    write_csv(RESULT_ROOT / f"{PREFIX_BASE}{slugify_env(env_choice.env_id)}_preflight.csv", rows)
    cfg = WrapperConfig(
        reward_scale=reward_scale,
        alpha_dyn=best_alpha,
        a_u=0.001,
        a_w=0.001,
        beta_rot=0.0,
        beta_sym=0.0,
        use_rot_dyn=best_mode and env_choice.action_dim >= 2,
    )
    text = "\n".join(
        [
            f"# {PREFIX_BASE}{slugify_env(env_choice.env_id)}_preflight_report",
            "",
            f"- env_id: `{env_choice.env_id}`",
            f"- reward_scale: `{cfg.reward_scale:.6e}`",
            f"- selected_alpha_dyn: `{cfg.alpha_dyn}`",
            f"- selected_use_rot_dyn: `{cfg.use_rot_dyn}`",
            f"- std_r_task_raw: `{std_task:.6e}`",
            f"- std_u_norm: `{float(np.std(u_norms)):.6e}`",
            f"- std_w_norm: `{float(np.std(w_norms)):.6e}`",
            f"- std_rot_term_first2: `{float(np.std(rot_samples)) if rot_samples else 0.0:.6e}`",
            "- beta_rot starts at `0.0` and is only increased later if baseline geometry fails.",
        ]
    ) + "\n"
    write_text(RESULT_ROOT / f"{PREFIX_BASE}{slugify_env(env_choice.env_id)}_preflight_report.md", text)
    return cfg, text


class ActorCriticRARL:
    def __init__(self, env_choice: EnvChoice, wrapper_cfg: WrapperConfig, actor_lr: float, seed: int) -> None:
        seed_everything(seed)
        self.env_choice = env_choice
        self.wrapper_cfg = wrapper_cfg
        self.actor_lr = actor_lr
        self.rarl_env = NaturalRARLEnv(env_choice, wrapper_cfg)
        self.actor_layout = FlatMLP(env_choice.obs_dim, ACTOR_HIDDEN, env_choice.action_dim)
        self.theta, self.phi = self.init_actor_params(seed)
        self.theta_target = self.theta.detach().clone()
        self.phi_target = self.phi.detach().clone()
        self.q_game = CriticNet(env_choice.obs_dim, env_choice.action_dim).to(DEVICE)
        self.q_game_target = CriticNet(env_choice.obs_dim, env_choice.action_dim).to(DEVICE)
        self.q_task = CriticNet(env_choice.obs_dim, env_choice.action_dim).to(DEVICE)
        self.q_task_target = CriticNet(env_choice.obs_dim, env_choice.action_dim).to(DEVICE)
        self.q_game_target.load_state_dict(self.q_game.state_dict())
        self.q_task_target.load_state_dict(self.q_task.state_dict())
        self.q_game_opt = torch.optim.Adam(self.q_game.parameters(), lr=CRITIC_LR)
        self.q_task_opt = torch.optim.Adam(self.q_task.parameters(), lr=CRITIC_LR)
        self.replay = ReplayBuffer(200000, env_choice.obs_dim, env_choice.action_dim)
        self.rng = np.random.default_rng(seed)
        self.metric_refs: dict[str, float] = {}
        self.field_slices = {
            "theta": slice(0, self.actor_layout.num_params),
            "phi": slice(self.actor_layout.num_params, 2 * self.actor_layout.num_params),
        }

    def init_actor_params(self, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(seed)
        theta = self.actor_layout.init_flat(gen, final_scale=0.05)
        phi = self.actor_layout.init_flat(gen, final_scale=0.05)
        return theta.detach().clone(), phi.detach().clone()

    def actor_raw(self, actor_flat: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_layout.forward(actor_flat, obs)

    def actor_action(self, actor_flat: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        raw = self.actor_raw(actor_flat, obs)
        return self.rarl_env.scale_action(raw)

    def actor_z(self, z: torch.Tensor, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta = z[self.field_slices["theta"]]
        phi = z[self.field_slices["phi"]]
        return self.actor_action(theta, obs), self.actor_action(phi, obs)

    def current_z(self) -> torch.Tensor:
        return torch.cat([self.theta, self.phi]).detach().clone()

    def set_from_z(self, z: torch.Tensor) -> None:
        self.theta = z[self.field_slices["theta"]].detach().clone()
        self.phi = z[self.field_slices["phi"]].detach().clone()

    def polyak_update(self) -> None:
        self.theta_target = ((1.0 - POLYAK_TAU) * self.theta_target) + (POLYAK_TAU * self.theta)
        self.phi_target = ((1.0 - POLYAK_TAU) * self.phi_target) + (POLYAK_TAU * self.phi)
        with torch.no_grad():
            for target, online in zip(self.q_game_target.parameters(), self.q_game.parameters()):
                target.data.mul_(1.0 - POLYAK_TAU).add_(POLYAK_TAU * online.data)
            for target, online in zip(self.q_task_target.parameters(), self.q_task.parameters()):
                target.data.mul_(1.0 - POLYAK_TAU).add_(POLYAK_TAU * online.data)

    def collect_random_warmup(self, steps: int) -> None:
        env = gym.make(self.env_choice.env_id)
        obs, _ = env.reset(seed=SEED + 11)
        for _ in range(steps):
            u = self.rng.uniform(self.env_choice.action_low, self.env_choice.action_high).astype(np.float32)
            w = self.rng.uniform(self.env_choice.action_low, self.env_choice.action_high).astype(np.float32)
            w_t = torch.as_tensor(w, dtype=DTYPE, device=DEVICE)
            u_t = torch.as_tensor(u, dtype=DTYPE, device=DEVICE)
            a_env, _ = self.rarl_env.blend_action(u_t, w_t, self.wrapper_cfg.use_rot_dyn)
            next_obs, reward_raw, terminated, truncated, _ = env.step(a_env.cpu().numpy().astype(np.float32))
            reward_game, reward_task_scaled = self.rarl_env.reward_terms(float(reward_raw), u_t, w_t)
            done = terminated or truncated
            self.replay.add(obs, u, w, reward_game, reward_task_scaled, float(reward_raw), next_obs, done)
            obs = next_obs
            if done:
                obs, _ = env.reset()
        env.close()

    def collect_rollout(self, steps: int, exploration_std: float) -> dict[str, float]:
        env = gym.make(self.env_choice.env_id)
        obs, _ = env.reset(seed=SEED + 100 + int(self.rng.integers(0, 1_000_000)))
        total_clip = 0.0
        total_task = 0.0
        total_game = 0.0
        for _ in range(steps):
            obs_t = torch.as_tensor(obs, dtype=DTYPE, device=DEVICE).unsqueeze(0)
            u = self.actor_action(self.theta, obs_t).squeeze(0)
            w = self.actor_action(self.phi, obs_t).squeeze(0)
            if exploration_std > 0.0:
                u = torch.clamp(u + (exploration_std * torch.randn_like(u)), self.rarl_env.action_low, self.rarl_env.action_high)
                w = torch.clamp(w + (exploration_std * torch.randn_like(w)), self.rarl_env.action_low, self.rarl_env.action_high)
            a_env, clip_frac = self.rarl_env.blend_action(u, w, self.wrapper_cfg.use_rot_dyn)
            next_obs, reward_raw, terminated, truncated, _ = env.step(a_env.cpu().numpy().astype(np.float32))
            reward_game, reward_task_scaled = self.rarl_env.reward_terms(float(reward_raw), u, w)
            done = terminated or truncated
            self.replay.add(
                np.asarray(obs, dtype=np.float32),
                u.detach().cpu().numpy().astype(np.float32),
                w.detach().cpu().numpy().astype(np.float32),
                float(reward_game),
                float(reward_task_scaled),
                float(reward_raw),
                np.asarray(next_obs, dtype=np.float32),
                done,
            )
            total_clip += clip_frac
            total_task += float(reward_raw)
            total_game += float(reward_game)
            obs = next_obs
            if done:
                obs, _ = env.reset()
        env.close()
        return {
            "train_action_clip_fraction": total_clip / max(steps, 1),
            "train_game_return_proxy": total_game / max(steps, 1),
            "train_task_return_proxy": total_task / max(steps, 1),
        }

    def critic_update(self) -> dict[str, float]:
        losses_game = []
        losses_task = []
        for _ in range(CRITIC_UPDATES_PER_ITER):
            batch = self.replay.sample(TRAIN_BATCH_SIZE, self.rng)
            with torch.no_grad():
                u_next = self.actor_action(self.theta_target, batch["next_obs"])
                w_next = self.actor_action(self.phi_target, batch["next_obs"])
                target_game = batch["reward_game"] + (GAMMA * (1.0 - batch["done"]) * self.q_game_target(batch["next_obs"], u_next, w_next))
                target_task = batch["reward_task_scaled"] + (GAMMA * (1.0 - batch["done"]) * self.q_task_target(batch["next_obs"], u_next, w_next))
            pred_game = self.q_game(batch["obs"], batch["u"], batch["w"])
            pred_task = self.q_task(batch["obs"], batch["u"], batch["w"])
            loss_game = torch.mean((pred_game - target_game) ** 2)
            loss_task = torch.mean((pred_task - target_task) ** 2)
            self.q_game_opt.zero_grad(set_to_none=True)
            loss_game.backward()
            self.q_game_opt.step()
            self.q_task_opt.zero_grad(set_to_none=True)
            loss_task.backward()
            self.q_task_opt.step()
            losses_game.append(float(loss_game.detach().item()))
            losses_task.append(float(loss_task.detach().item()))
        return {
            "critic_loss_game": float(np.mean(losses_game)),
            "critic_loss_task": float(np.mean(losses_task)),
        }

    def actor_objective(self, z: torch.Tensor, states: torch.Tensor, use_task_q: bool = False) -> torch.Tensor:
        u, w = self.actor_z(z, states)
        critic = self.q_task if use_task_q else self.q_game
        return critic(states, u, w).mean()

    def actor_field(self, z: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        z_req = z if z.requires_grad else z.detach().clone().requires_grad_(True)
        j_val = self.actor_objective(z_req, states, use_task_q=False)
        grad = torch.autograd.grad(j_val, z_req, create_graph=True)[0]
        out = torch.zeros_like(z_req)
        out[self.field_slices["theta"]] = -grad[self.field_slices["theta"]]
        out[self.field_slices["phi"]] = +grad[self.field_slices["phi"]]
        return out

    def local_gap(self, z: torch.Tensor, states: torch.Tensor, protagonist: bool, with_prox: bool, actor_lr: float) -> torch.Tensor:
        base = z.detach().clone()
        current = base.clone()
        actor_slice = self.field_slices["theta"] if protagonist else self.field_slices["phi"]
        initial = base[actor_slice].clone()
        j_start = self.actor_objective(base, states, use_task_q=False)
        inner_lr = 0.1 * actor_lr
        for _ in range(GAP_INNER_STEPS):
            cur = current.detach().clone().requires_grad_(True)
            j_val = self.actor_objective(cur, states, use_task_q=False)
            grad = torch.autograd.grad(j_val, cur)[0][actor_slice]
            if protagonist:
                step = grad
            else:
                step = -grad
            if with_prox:
                step = step - ((cur[actor_slice] - initial) / max(LOCAL_GAP_RADIUS, EPS))
            next_actor = cur[actor_slice] + (inner_lr * step)
            delta = next_actor - initial
            norm = torch.linalg.norm(delta)
            if norm > LOCAL_GAP_RADIUS:
                next_actor = initial + (delta * (LOCAL_GAP_RADIUS / (norm + EPS)))
            current = cur.detach().clone()
            current[actor_slice] = next_actor.detach()
        j_end = self.actor_objective(current, states, use_task_q=False)
        if protagonist:
            return torch.relu(j_end - j_start)
        return torch.relu(j_start - j_end)

    def exploitability_parts(self, z: torch.Tensor, states: torch.Tensor, actor_lr: float) -> tuple[float, float]:
        p = float(self.local_gap(z, states, True, False, actor_lr).detach().item())
        a = float(self.local_gap(z, states, False, False, actor_lr).detach().item())
        return p, a

    def geometry_metrics(self, z: torch.Tensor, states: torch.Tensor) -> dict[str, float]:
        z_req = z.detach().clone().requires_grad_(True)
        field = self.actor_field(z_req, states)
        _, g_vec = torch.autograd.functional.jvp(lambda zz: self.actor_field(zz, states), (z_req,), (field.detach(),), create_graph=False, strict=False)
        field_det = field.detach()
        g_det = g_vec.detach()
        f_norm = float(torch.linalg.norm(field_det).item())
        g_norm = float(torch.linalg.norm(g_det).item())
        cos_fg = float(torch.dot(field_det, g_det).item() / ((f_norm * g_norm) + EPS))
        aproxy = 0.0
        sproxy = 0.0
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(20260615)
        for _ in range(ROTATION_RANDOM_VECS):
            v = torch.randn(z_req.numel(), generator=gen, dtype=DTYPE, device=DEVICE)
            v = v / (torch.linalg.norm(v) + EPS)
            _, jv = torch.autograd.functional.jvp(lambda zz: self.actor_field(zz, states), (z_req,), (v,), create_graph=False, strict=False)
            jtv = torch.autograd.grad(torch.dot(field, v), z_req, retain_graph=True)[0]
            aproxy += float(torch.linalg.norm(jv.detach() - jtv.detach()).item())
            sproxy += float(torch.linalg.norm(jv.detach() + jtv.detach()).item())
        theta_slice = self.field_slices["theta"]
        phi_slice = self.field_slices["phi"]

        def p_field(cur_z: torch.Tensor) -> torch.Tensor:
            return self.actor_field(cur_z, states)[theta_slice].detach()

        def a_field(cur_z: torch.Tensor) -> torch.Tensor:
            return self.actor_field(cur_z, states)[phi_slice].detach()

        base_p = p_field(z)
        base_a = a_field(z)
        pert_theta = z.detach().clone()
        pert_phi = z.detach().clone()
        pert_theta[theta_slice] = pert_theta[theta_slice] + 1e-3
        pert_phi[phi_slice] = pert_phi[phi_slice] + 1e-3
        cross_p = float(torch.linalg.norm(p_field(pert_phi) - base_p).item())
        cross_a = float(torch.linalg.norm(a_field(pert_theta) - base_a).item())
        same_p = float(torch.linalg.norm(p_field(pert_theta) - base_p).item())
        same_a = float(torch.linalg.norm(a_field(pert_phi) - base_a).item())
        cross = 0.5 * (cross_p + cross_a)
        same = 0.5 * (same_p + same_a)
        return {
            "g_over_f": g_norm / (f_norm + EPS),
            "cos_fg": cos_fg,
            "non_collinearity": math.sqrt(max(0.0, 1.0 - min(1.0, cos_fg * cos_fg))),
            "rotation_ratio_proxy": aproxy / (sproxy + EPS),
            "cross_player_coupling_proxy": cross,
            "cross_to_same_ratio": cross / (same + EPS),
        }

    def diagnostic_metrics(self, z: torch.Tensor, diag_batch: dict[str, torch.Tensor], actor_lr: float, compute_geometry: bool) -> dict[str, float]:
        states = diag_batch["obs"]
        field = self.actor_field(z, states).detach()
        field_energy = 0.5 * float(torch.dot(field, field).item())
        p_gap = float(self.local_gap(z, states, True, True, actor_lr).detach().item())
        a_gap = float(self.local_gap(z, states, False, True, actor_lr).detach().item())
        p_tau = p_gap + a_gap
        if "field0" not in self.metric_refs:
            self.metric_refs["field0"] = field_energy
            self.metric_refs["ptau0"] = max(p_tau, EPS)
        field_term = field_energy / (self.metric_refs["field0"] + EPS)
        p_tau_term = p_tau / (self.metric_refs["ptau0"] + EPS)
        v_lambda = (LAMBDA_F * field_term) + (LAMBDA_P * p_tau_term)
        exploit_p, exploit_a = self.exploitability_parts(z, states, actor_lr)
        u = self.actor_action(z[self.field_slices["theta"]], states)
        w = self.actor_action(z[self.field_slices["phi"]], states)
        j_game = float(self.q_game(states, u, w).mean().detach().item())
        j_task = float(self.q_task(states, u, w).mean().detach().item())
        out = {
            "V_lambda": v_lambda,
            "field_term": field_term,
            "normalized_P_tau": p_tau_term,
            "raw_P_tau": p_tau,
            "field_norm": float(torch.linalg.norm(field).item()),
            "approximate_exploitability": exploit_p + exploit_a,
            "exploitability_protagonist": exploit_p,
            "exploitability_adversary": exploit_a,
            "P_tau_protagonist_gap": p_gap,
            "P_tau_adversary_gap": a_gap,
            "Q_game": j_game,
            "Q_task": j_task,
        }
        if compute_geometry:
            out.update(self.geometry_metrics(z, states))
        else:
            out.update(
                {
                    "g_over_f": math.nan,
                    "cos_fg": math.nan,
                    "non_collinearity": math.nan,
                    "rotation_ratio_proxy": math.nan,
                    "cross_player_coupling_proxy": math.nan,
                    "cross_to_same_ratio": math.nan,
                }
            )
        return out

    def evaluate_policy(
        self,
        theta: torch.Tensor,
        phi: torch.Tensor | None,
        episodes: int,
        mode: str,
    ) -> dict[str, float]:
        env = gym.make(self.env_choice.env_id)
        task_returns_raw: list[float] = []
        task_returns_scaled: list[float] = []
        game_returns: list[float] = []
        clip_fracs: list[float] = []
        for ep in range(episodes):
            obs, _ = env.reset(seed=SEED + 9000 + ep)
            done = False
            task_raw = 0.0
            task_scaled = 0.0
            game_total = 0.0
            ep_clips: list[float] = []
            while not done:
                obs_t = torch.as_tensor(obs, dtype=DTYPE, device=DEVICE).unsqueeze(0)
                u = self.actor_action(theta, obs_t).squeeze(0)
                if mode == "clean" or phi is None:
                    w = torch.zeros_like(u)
                    a_env = torch.clamp(u, self.rarl_env.action_low, self.rarl_env.action_high)
                    clip_frac = float(((u - a_env).abs() > 1e-12).float().mean().item())
                else:
                    w = self.actor_action(phi, obs_t).squeeze(0)
                    a_env, clip_frac = self.rarl_env.blend_action(u, w, self.wrapper_cfg.use_rot_dyn)
                next_obs, reward_raw, terminated, truncated, _ = env.step(a_env.detach().cpu().numpy().astype(np.float32))
                reward_game, reward_task_scaled = self.rarl_env.reward_terms(float(reward_raw), u, w)
                task_raw += float(reward_raw)
                task_scaled += float(reward_task_scaled)
                game_total += float(reward_game)
                ep_clips.append(clip_frac)
                obs = next_obs
                done = terminated or truncated
            task_returns_raw.append(task_raw)
            task_returns_scaled.append(task_scaled)
            game_returns.append(game_total)
            clip_fracs.append(float(np.mean(ep_clips)) if ep_clips else 0.0)
        env.close()
        return {
            "task_return_raw": float(np.mean(task_returns_raw)),
            "task_return_scaled": float(np.mean(task_returns_scaled)),
            "game_return": float(np.mean(game_returns)),
            "action_clip_fraction": float(np.mean(clip_fracs)),
        }

    def robust_br_adversary(self, theta: torch.Tensor, diag_batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, bool]:
        base = self.phi.detach().clone()
        current = base.clone()
        states = diag_batch["obs"]
        for _ in range(BR_INNER_STEPS):
            cur = current.detach().clone().requires_grad_(True)
            z = torch.cat([theta.detach(), cur])
            objective = self.actor_objective(z, states, use_task_q=True)
            grad = torch.autograd.grad(objective, cur)[0]
            next_phi = cur - (BR_INNER_LR * grad)
            delta = next_phi - base
            norm = torch.linalg.norm(delta)
            if norm > LOCAL_BR_RADIUS:
                next_phi = base + (delta * (LOCAL_BR_RADIUS / (norm + EPS)))
            current = next_phi.detach()
        return current, True


def run_actor_update(game: ActorCriticRARL, z: torch.Tensor, states: torch.Tensor, method: str, actor_lr: float) -> tuple[torch.Tensor, dict[str, float]]:
    field = game.actor_field(z, states).detach()
    if method == "sgd":
        delta = -actor_lr * field
        return (z + delta).detach(), {"update_norm": float(torch.linalg.norm(delta).item()), "fallback_to_egm_frac": 0.0, "gamma_active_frac": 0.0, "G_contribution_ratio": 0.0}
    if method == "egm":
        z_half = z - (actor_lr * field)
        field_half = game.actor_field(z_half, states).detach()
        delta = -actor_lr * field_half
        return (z + delta).detach(), {"update_norm": float(torch.linalg.norm(delta).item()), "fallback_to_egm_frac": 0.0, "gamma_active_frac": 0.0, "G_contribution_ratio": 0.0}
    if method == "ppm":
        current = z.detach().clone()
        for _ in range(PPM_INNER_STEPS):
            field_inner = game.actor_field(current, states).detach()
            current = z - (actor_lr * field_inner)
        delta = current - z
        return current.detach(), {"update_norm": float(torch.linalg.norm(delta).item()), "fallback_to_egm_frac": 0.0, "gamma_active_frac": 0.0, "G_contribution_ratio": 0.0}
    raise ValueError(method)


def run_training_method(
    env_choice: EnvChoice,
    wrapper_cfg: WrapperConfig,
    actor_lr: float,
    method: str,
    num_outer_iterations: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    game = ActorCriticRARL(env_choice, wrapper_cfg, actor_lr, SEED)
    game.collect_random_warmup(REPLAY_WARMUP_STEPS)
    diag_batch = game.replay.fixed_state_batch(TRAIN_BATCH_SIZE, np.random.default_rng(SEED + 777))
    curves: list[dict[str, Any]] = []
    health_ok = True
    clean_eval = game.evaluate_policy(game.theta, None, 5, "clean")
    adv_eval = game.evaluate_policy(game.theta, game.phi, 5, "current")
    br_phi, br_valid = game.robust_br_adversary(game.theta, diag_batch)
    br_eval = game.evaluate_policy(game.theta, br_phi, 5, "current")
    for iteration in range(num_outer_iterations + 1):
        z = game.current_z()
        diag = game.diagnostic_metrics(z, diag_batch, actor_lr, compute_geometry=(iteration % 10 == 0))
        critic_stats = game.critic_update()
        train_stats = {"train_action_clip_fraction": math.nan, "train_game_return_proxy": math.nan, "train_task_return_proxy": math.nan}
        if iteration < num_outer_iterations:
            train_stats = game.collect_rollout(ROLLOUT_STEPS_PER_ITER, EXPLORATION_STD)
        if iteration % EVAL_EVERY == 0 or iteration == num_outer_iterations:
            clean_eval = game.evaluate_policy(game.theta, None, 5, "clean")
            adv_eval = game.evaluate_policy(game.theta, game.phi, 5, "current")
            br_phi, br_valid = game.robust_br_adversary(game.theta, diag_batch)
            br_eval = game.evaluate_policy(game.theta, br_phi, 5, "current")
        row = {
            "env_id": env_choice.env_id,
            "method": method,
            "actor_lr": actor_lr,
            "iteration": iteration,
            **wrapper_cfg.__dict__,
            **diag,
            **critic_stats,
            **train_stats,
            "clean_task_return_raw": clean_eval["task_return_raw"],
            "clean_task_return_scaled": clean_eval["task_return_scaled"],
            "current_adv_task_return_raw": adv_eval["task_return_raw"],
            "current_adv_task_return_scaled": adv_eval["task_return_scaled"],
            "robust_br_task_return_raw": br_eval["task_return_raw"],
            "robust_br_task_return_scaled": br_eval["task_return_scaled"],
            "robust_degradation": clean_eval["task_return_raw"] - br_eval["task_return_raw"],
            "br_valid": int(br_valid),
            "action_clip_fraction_br": br_eval["action_clip_fraction"],
            "nan_flag": 0,
            "valid_flag": 1,
        }
        row["nan_flag"] = int(
            not all(
                finite(row[key])
                for key in [
                    "V_lambda",
                    "field_norm",
                    "raw_P_tau",
                    "critic_loss_game",
                    "critic_loss_task",
                    "Q_game",
                    "Q_task",
                    "clean_task_return_raw",
                    "current_adv_task_return_raw",
                    "robust_br_task_return_raw",
                ]
            )
        )
        valid = (
            row["nan_flag"] == 0
            and row["train_action_clip_fraction"] <= 0.05 if finite(row["train_action_clip_fraction"]) else True
            and row["clean_task_return_raw"] > -1e9
            and row["current_adv_task_return_raw"] > -1e9
            and row["br_valid"] == 1
            and row["action_clip_fraction_br"] <= 0.05
        )
        row["valid_flag"] = int(valid)
        health_ok = health_ok and bool(valid)
        curves.append(row)
        if iteration < num_outer_iterations:
            actor_batch = game.replay.fixed_state_batch(TRAIN_BATCH_SIZE, game.rng)
            next_z, meta = run_actor_update(game, game.current_z(), actor_batch["obs"], method, actor_lr)
            game.set_from_z(next_z)
            game.polyak_update()
            curves[-1].update(meta)
    summary = summarize_run(curves, actor_lr)
    summary["health_ok"] = int(health_ok)
    return curves, summary


def summarize_run(curves: list[dict[str, Any]], actor_lr: float) -> dict[str, Any]:
    first = curves[0]
    last = curves[-1]
    v_vals = [row["V_lambda"] for row in curves]
    p_vals = [row["normalized_P_tau"] for row in curves]
    f_vals = [row["field_norm"] for row in curves]
    return {
        "env_id": first["env_id"],
        "method": first["method"],
        "actor_lr": actor_lr,
        "valid_flag": int(all(int(row["valid_flag"]) == 1 for row in curves)),
        "V_lambda_start": first["V_lambda"],
        "V_lambda_final": last["V_lambda"],
        "V_lambda_AUC": float(sum(v_vals)),
        "V_lambda_moving_average_slope": moving_average_slope(v_vals),
        "V_lambda_spike_ratio": spike_ratio(v_vals),
        "P_tau_start": first["normalized_P_tau"],
        "P_tau_final": last["normalized_P_tau"],
        "P_tau_AUC": float(sum(p_vals)),
        "P_tau_moving_average_slope": moving_average_slope(p_vals),
        "P_tau_spike_ratio": spike_ratio(p_vals),
        "field_norm_start": first["field_norm"],
        "field_norm_final": last["field_norm"],
        "field_norm_AUC": float(sum(f_vals)),
        "field_norm_moving_average_slope": moving_average_slope(f_vals),
        "field_norm_spike_ratio": spike_ratio(f_vals),
        "clean_task_return_raw_final": last["clean_task_return_raw"],
        "current_adv_task_return_raw_final": last["current_adv_task_return_raw"],
        "robust_br_task_return_raw_final": last["robust_br_task_return_raw"],
        "robust_degradation_final": last["robust_degradation"],
        "rotation_ratio_proxy_final": last["rotation_ratio_proxy"] if finite(last["rotation_ratio_proxy"]) else math.nan,
        "cross_player_coupling_proxy_final": last["cross_player_coupling_proxy"] if finite(last["cross_player_coupling_proxy"]) else math.nan,
        "cross_to_same_ratio_final": last["cross_to_same_ratio"] if finite(last["cross_to_same_ratio"]) else math.nan,
    }


def curve_normal(summary: dict[str, Any]) -> bool:
    return (
        summary["V_lambda_final"] <= summary["V_lambda_start"] + 1e-8
        and summary["P_tau_final"] <= summary["P_tau_start"] + 1e-8
        and summary["V_lambda_spike_ratio"] <= 5.0
        and summary["P_tau_spike_ratio"] <= 5.0
        and summary["field_norm_spike_ratio"] <= 5.0
        and summary["field_norm_final"] <= (5.0 * summary["field_norm_start"] + 1e-8)
    )


def choose_sgd(summary_rows: list[dict[str, Any]]) -> tuple[str, float | None]:
    valid = [row for row in summary_rows if row["valid_flag"] == 1 and curve_normal(row)]
    if not valid:
        return "SGD_NORMAL_FAIL", None
    best = min(valid, key=lambda row: row["V_lambda_AUC"])
    return "SGD_NORMAL_PASS", float(best["actor_lr"])


def baseline_gate_decision(summary_rows: list[dict[str, Any]]) -> str:
    if any(row["valid_flag"] != 1 for row in summary_rows):
        return "BASELINE_FAIL_CURVES_ABNORMAL"
    if any(not curve_normal(row) for row in summary_rows):
        return "BASELINE_FAIL_CURVES_ABNORMAL"
    sgd = next(row for row in summary_rows if row["method"] == "sgd")
    egm = next(row for row in summary_rows if row["method"] == "egm")
    ppm = next(row for row in summary_rows if row["method"] == "ppm")
    winners = [egm, ppm]
    best = min(winners, key=lambda row: row["V_lambda_AUC"])
    if (
        max(sgd["V_lambda_AUC"] / (egm["V_lambda_AUC"] + EPS), sgd["V_lambda_AUC"] / (ppm["V_lambda_AUC"] + EPS)) < 1.3
        and max(sgd["P_tau_AUC"] / (egm["P_tau_AUC"] + EPS), sgd["P_tau_AUC"] / (ppm["P_tau_AUC"] + EPS)) < 1.3
    ):
        return "BASELINE_FAIL_NO_EGM_PPM_ADVANTAGE"
    if not (
        finite(best["cross_player_coupling_proxy_final"])
        and best["cross_player_coupling_proxy_final"] > 0.0
        and best["cross_to_same_ratio_final"] > 0.05
        and finite(best["non_collinearity_final"]) if "non_collinearity_final" in best else True
    ):
        return "BASELINE_FAIL_GEOMETRY_WEAK"
    if best["robust_br_task_return_raw_final"] < (sgd["robust_br_task_return_raw_final"] - 1e-6):
        return "BASELINE_FAIL_RARL_PERFORMANCE"
    return "BASELINE_READY_FOR_PROPOSED"


def make_plot(method_curves: dict[str, list[dict[str, Any]]], metric: str, title: str, path: Path) -> None:
    if plt is None:
        return
    colors = {"sgd": "#1f77b4", "egm": "#ff7f0e", "ppm": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, rows in method_curves.items():
        ax.plot([row["iteration"] for row in rows], [row[metric] for row in rows], label=method, color=colors.get(method, None), linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_performance_plot(method_curves: dict[str, list[dict[str, Any]]], path: Path) -> None:
    if plt is None:
        return
    colors = {"sgd": "#1f77b4", "egm": "#ff7f0e", "ppm": "#2ca02c"}
    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
    panels = [
        ("clean_task_return_raw", "Clean Task Return"),
        ("current_adv_task_return_raw", "Current Adversarial Task Return"),
        ("robust_br_task_return_raw", "Robust BR Task Return"),
        ("robust_degradation", "Robust Degradation"),
    ]
    for ax, (metric, title) in zip(axes, panels):
        for method, rows in method_curves.items():
            ax.plot([row["iteration"] for row in rows], [row[metric] for row in rows], label=method, color=colors.get(method, None), linewidth=1.8)
        ax.set_title(title)
    axes[0].legend()
    axes[-1].set_xlabel("Iteration")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_big_baseline_plot(method_curves: dict[str, list[dict[str, Any]]], path: Path) -> None:
    if plt is None:
        return
    colors = {"sgd": "#1f77b4", "egm": "#ff7f0e", "ppm": "#2ca02c"}
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    panels = [
        ("V_lambda", "V_lambda"),
        ("normalized_P_tau", "P_tau"),
        ("field_norm", "Field Norm"),
        ("approximate_exploitability", "Exploitability"),
        ("current_adv_task_return_raw", "Current Adv Task Return"),
        ("robust_br_task_return_raw", "Robust BR Task Return"),
    ]
    for ax, (metric, title) in zip(axes.flat, panels):
        for method, rows in method_curves.items():
            ax.plot([row["iteration"] for row in rows], [row[metric] for row in rows], label=method, color=colors.get(method, None), linewidth=1.6)
        ax.set_title(title)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_sgd_lr_plots(sgd_curves: list[dict[str, Any]], env_slug: str) -> None:
    if plt is None:
        return
    plot_dir = RESULT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in sgd_curves:
        label = f"sgd_lr_{row['actor_lr']}"
        groups.setdefault(label, []).append(row)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for metric, title, filename in [
        ("V_lambda", "SGD-only V_lambda", f"{PREFIX_BASE}{env_slug}_sgd_V_lambda.png"),
        ("normalized_P_tau", "SGD-only P_tau", f"{PREFIX_BASE}{env_slug}_sgd_P_tau.png"),
        ("field_norm", "SGD-only Field Norm", f"{PREFIX_BASE}{env_slug}_sgd_field_norm.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for idx, (label, rows) in enumerate(groups.items()):
            ax.plot([row["iteration"] for row in rows], [row[metric] for row in rows], label=label, color=colors[idx % len(colors)], linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
    panels = [
        ("clean_task_return_raw", "Clean Task Return"),
        ("current_adv_task_return_raw", "Current Adv Task Return"),
        ("robust_br_task_return_raw", "Robust BR Task Return"),
        ("robust_degradation", "Robust Degradation"),
    ]
    for ax, (metric, title) in zip(axes, panels):
        for idx, (label, rows) in enumerate(groups.items()):
            ax.plot([row["iteration"] for row in rows], [row[metric] for row in rows], label=label, color=colors[idx % len(colors)], linewidth=1.8)
        ax.set_title(title)
    axes[0].legend()
    axes[-1].set_xlabel("Iteration")
    fig.tight_layout()
    fig.savefig(plot_dir / f"{PREFIX_BASE}{env_slug}_sgd_rarl_performance.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    seed_everything(SEED)
    env_choice = choose_environment()
    env_slug = slugify_env(env_choice.env_id)
    prefix = f"{PREFIX_BASE}{env_slug}_"
    wrapper_cfg, _ = random_policy_preflight(env_choice)

    sgd_curves_all: list[dict[str, Any]] = []
    sgd_summaries: list[dict[str, Any]] = []
    for actor_lr in ACTOR_LR_GRID:
        curves, summary = run_training_method(env_choice, wrapper_cfg, actor_lr, "sgd", NUM_OUTER_ITERATIONS)
        for row in curves:
            row["curve_normal_flag"] = int(curve_normal(summarize_run(curves, actor_lr)))
        sgd_curves_all.extend(curves)
        sgd_summaries.append(summary)
        write_csv(RESULT_ROOT / f"{prefix}sgd_gate_curves.csv", sgd_curves_all)
        write_csv(RESULT_ROOT / f"{prefix}sgd_gate.csv", sgd_summaries)

    make_sgd_lr_plots(sgd_curves_all, env_slug)

    sgd_decision, selected_actor_lr = choose_sgd(sgd_summaries)
    sgd_lines = [
        f"# {prefix}sgd_gate_report",
        "",
        f"- env_id: `{env_choice.env_id}`",
        f"- selected_actor_lr: `{selected_actor_lr}`",
        f"- decision: `{sgd_decision}`",
        "",
    ]
    for row in sgd_summaries:
        sgd_lines.append(
            f"- lr `{row['actor_lr']}`: valid=`{bool(row['valid_flag'])}`, curve_normal=`{curve_normal(row)}`, "
            f"auc_V=`{row['V_lambda_AUC']:.6e}`, auc_P=`{row['P_tau_AUC']:.6e}`, auc_field=`{row['field_norm_AUC']:.6e}`, "
            f"robust_br_task_return_raw_final=`{row['robust_br_task_return_raw_final']:.6e}`"
        )
    write_text(RESULT_ROOT / f"{prefix}sgd_gate_report.md", "\n".join(sgd_lines) + "\n")

    final_decision = "SGD_NORMAL_FAIL"
    final_lines = [
        f"# {prefix}final_report",
        "",
        f"1. Environment used: `{env_choice.env_id}`.",
        f"2. RARL wrapper: protagonist action `u`, adversary action `w`, environment executes `a_env = u + alpha_dyn * R_dyn w` when `use_rot_dyn=True` else `a_env = u + alpha_dyn * w`.",
        f"3. Natural RARL reward only or weak rotational shaping? `beta_rot={wrapper_cfg.beta_rot}`, `beta_sym={wrapper_cfg.beta_sym}`.",
        f"4. Selected alpha_dyn=`{wrapper_cfg.alpha_dyn}`, reward_scale=`{wrapper_cfg.reward_scale:.6e}`, beta_rot=`{wrapper_cfg.beta_rot}`, beta_sym=`{wrapper_cfg.beta_sym}`.",
        f"5. Did SGD pass normality gate? `{sgd_decision == 'SGD_NORMAL_PASS'}`.",
    ]

    if sgd_decision == "SGD_NORMAL_FAIL" or selected_actor_lr is None:
        write_text(RESULT_ROOT / f"{prefix}final_report.md", "\n".join(final_lines + [f"6. Final decision: `{final_decision}`."]) + "\n")
        return

    baseline_curves: list[dict[str, Any]] = []
    baseline_summaries: list[dict[str, Any]] = []
    for method in ["sgd", "egm", "ppm"]:
        curves, summary = run_training_method(env_choice, wrapper_cfg, selected_actor_lr, method, NUM_OUTER_ITERATIONS)
        summary["curve_normal_flag"] = int(curve_normal(summary))
        summary["non_collinearity_final"] = curves[-1]["non_collinearity"] if finite(curves[-1]["non_collinearity"]) else math.nan
        baseline_curves.extend(curves)
        baseline_summaries.append(summary)
    write_csv(RESULT_ROOT / f"{prefix}baseline_gate_curves.csv", baseline_curves)
    write_csv(RESULT_ROOT / f"{prefix}baseline_gate.csv", baseline_summaries)
    geometry_rows = []
    for row in baseline_summaries:
        geometry_rows.append(
            {
                "method": row["method"],
                "actor_lr": row["actor_lr"],
                "rotation_ratio_proxy": row["rotation_ratio_proxy_final"],
                "cross_player_coupling_proxy": row["cross_player_coupling_proxy_final"],
                "cross_to_same_ratio": row["cross_to_same_ratio_final"],
                "g_over_f": next(curve["g_over_f"] for curve in reversed(baseline_curves) if curve["method"] == row["method"] and finite(curve["g_over_f"])),
                "cos_fg": next(curve["cos_fg"] for curve in reversed(baseline_curves) if curve["method"] == row["method"] and finite(curve["cos_fg"])),
                "non_collinearity": next(curve["non_collinearity"] for curve in reversed(baseline_curves) if curve["method"] == row["method"] and finite(curve["non_collinearity"])),
            }
        )
    write_csv(RESULT_ROOT / f"{prefix}geometry_audit.csv", geometry_rows)
    geometry_text = "\n".join(
        [f"# {prefix}geometry_audit", ""]
        + [
            f"- {row['method']}: rotation_ratio_proxy=`{row['rotation_ratio_proxy']:.6e}`, cross_player_coupling_proxy=`{row['cross_player_coupling_proxy']:.6e}`, "
            f"cross_to_same_ratio=`{row['cross_to_same_ratio']:.6e}`, ||G||/||F||=`{row['g_over_f']:.6e}`, cos(F,G)=`{row['cos_fg']:.6e}`, non_collinearity=`{row['non_collinearity']:.6e}`"
            for row in geometry_rows
        ]
    ) + "\n"
    write_text(RESULT_ROOT / f"{prefix}geometry_audit.md", geometry_text)

    baseline_decision = baseline_gate_decision(baseline_summaries)
    baseline_lines = [
        f"# {prefix}baseline_gate_report",
        "",
        f"- selected_actor_lr: `{selected_actor_lr}`",
        f"- decision: `{baseline_decision}`",
        "",
    ]
    for row in baseline_summaries:
        baseline_lines.append(
            f"- {row['method']}: valid=`{bool(row['valid_flag'])}`, curve_normal=`{bool(row['curve_normal_flag'])}`, auc_V=`{row['V_lambda_AUC']:.6e}`, "
            f"auc_P=`{row['P_tau_AUC']:.6e}`, auc_field=`{row['field_norm_AUC']:.6e}`, robust_br_task_return_raw_final=`{row['robust_br_task_return_raw_final']:.6e}`"
        )
    write_text(RESULT_ROOT / f"{prefix}baseline_gate_report.md", "\n".join(baseline_lines) + "\n")

    if plt is not None:
        method_curves = {method: [row for row in baseline_curves if row["method"] == method] for method in ["sgd", "egm", "ppm"]}
        make_plot(method_curves, "V_lambda", "Baseline V_lambda", RESULT_ROOT / "plots" / f"{prefix}baseline_V_lambda.png")
        make_plot(method_curves, "normalized_P_tau", "Baseline P_tau", RESULT_ROOT / "plots" / f"{prefix}baseline_P_tau.png")
        make_plot(method_curves, "field_norm", "Baseline Field Norm", RESULT_ROOT / "plots" / f"{prefix}baseline_field_norm.png")
        make_plot(method_curves, "approximate_exploitability", "Baseline Exploitability", RESULT_ROOT / "plots" / f"{prefix}baseline_exploitability.png")
        make_performance_plot(method_curves, RESULT_ROOT / "plots" / f"{prefix}baseline_rarl_performance.png")
        make_big_baseline_plot(method_curves, RESULT_ROOT / "plots" / f"{prefix}baseline_all_plots_big.png")

    final_decision = "BASELINE_READY_FOR_PROPOSED" if baseline_decision == "BASELINE_READY_FOR_PROPOSED" else "BASELINE_FAIL"
    final_lines.extend(
        [
            f"6. Did EGM/PPM pass baseline gate? `{baseline_decision == 'BASELINE_READY_FOR_PROPOSED'}`.",
            f"7. Is the actor field cross-coupled and nontrivially rotational? `{all(row['cross_player_coupling_proxy'] > 0.0 and row['cross_to_same_ratio'] > 0.05 for row in geometry_rows)}`.",
            "8. Proposed was not run in this round unless the baseline gate passed, and this script stops after one environment.",
            f"9. Final decision: `{final_decision if baseline_decision != 'BASELINE_READY_FOR_PROPOSED' else baseline_decision}``.",
        ]
    )
    write_text(RESULT_ROOT / f"{prefix}final_report.md", "\n".join(final_lines) + "\n")


if __name__ == "__main__":
    main()
