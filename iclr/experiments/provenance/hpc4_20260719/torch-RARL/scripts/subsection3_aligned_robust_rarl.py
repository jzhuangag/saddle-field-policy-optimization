from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / "actor_only_critic_rarl_subsection3.py"

spec = importlib.util.spec_from_file_location("actor_only_critic_rarl_subsection3", BASE_PATH)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


RESULT_ROOT = BASE_PATH.resolve().parents[2] / "results" / "subsection3_aligned_robust_rarl"
PLOT_ROOT = RESULT_ROOT / "plots"
PREFIX = "s3_swimmer_v5_"
ENV_ID = "Swimmer-v5"

SEED = 0
DEVICE = base.DEVICE
DTYPE = base.DTYPE
EPS = base.EPS
GAMMA = 0.99

REPLAY_WARMUP_STEPS_GRID = [30000, 50000]
CRITIC_PRETRAIN_STEPS = 5000
CRITIC_LR = 1e-3
BATCH_SIZE = 256
POLYAK_TAU = 0.005
CRITIC_AUDIT_INTERVAL = 1000
MC_HORIZON = 80
MC_SNAPSHOT_COUNT = 12

ACTOR_HIDDEN = (64, 64)
CRITIC_HIDDEN = (128, 128)
ACTOR_LR_GRID = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
NUM_ACTOR_ITERS = 150
PLOT_EVAL_INTERVAL = 10
PPM_INNER_STEPS = 5

LAMBDA_F = 0.01
LAMBDA_P = 1.0
GAP_INNER_STEPS = 3
TAU_GAP = 0.03
LOCAL_GAP_RADIUS = 0.1
BR_INNER_STEPS = 20
BR_INNER_LR_GRID = [1e-4, 3e-4]
LOCAL_BR_RADIUS = 0.25

ROTATION_RANDOM_VECS = 4
PRELIGHT_TRANSITIONS = 4096

ALPHA_GRID = [0.02, 0.05, 0.1, 0.2]
DISTURBANCE_TYPES = ["rotated", "direct"]
BETA_ROT_GRID = [0.0, 0.01, 0.03, 0.1, 0.3]
EXPLORATION_STD_GRID = [0.1, 0.2]

UPDATE_RADIUS_GRID = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
PROPOSED_PREFLIGHT_ITERS = 30


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    base.write_csv(path, rows)


def write_text(path: Path, text: str) -> None:
    base.write_text(path, text)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def finite(value: float) -> bool:
    return math.isfinite(float(value))


def spike_ratio(values: list[float]) -> float:
    return base.spike_ratio(values)


@dataclass(frozen=True)
class EnvChoice:
    env_id: str
    obs_dim: int
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    max_episode_steps: int


@dataclass(frozen=True)
class WrapperConfig:
    reward_scale: float
    alpha_dyn: float
    a_u: float
    a_w: float
    beta_rot: float
    beta_sym: float
    disturbance_type: str


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.u = np.zeros((capacity, action_dim), dtype=np.float32)
        self.w = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward_align = np.zeros((capacity,), dtype=np.float32)
        self.reward_pure_scaled = np.zeros((capacity,), dtype=np.float32)
        self.reward_pure_raw = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        u: np.ndarray,
        w: np.ndarray,
        reward_align: float,
        reward_pure_scaled: float,
        reward_pure_raw: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.ptr
        self.obs[idx] = obs
        self.u[idx] = u
        self.w[idx] = w
        self.reward_align[idx] = reward_align
        self.reward_pure_scaled[idx] = reward_pure_scaled
        self.reward_pure_raw[idx] = reward_pure_raw
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
            "reward_align": torch.as_tensor(self.reward_align[idx], dtype=DTYPE, device=DEVICE),
            "reward_pure_scaled": torch.as_tensor(self.reward_pure_scaled[idx], dtype=DTYPE, device=DEVICE),
            "reward_pure_raw": torch.as_tensor(self.reward_pure_raw[idx], dtype=DTYPE, device=DEVICE),
            "next_obs": torch.as_tensor(self.next_obs[idx], dtype=DTYPE, device=DEVICE),
            "done": torch.as_tensor(self.done[idx], dtype=DTYPE, device=DEVICE),
        }

    def fixed_state_batch(self, batch_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        idx = rng.integers(0, self.size, size=batch_size)
        return {"obs": torch.as_tensor(self.obs[idx], dtype=DTYPE, device=DEVICE)}


class AlignedRARLEnv:
    def __init__(self, env_choice: EnvChoice, cfg: WrapperConfig) -> None:
        self.env_choice = env_choice
        self.cfg = cfg
        self.action_low = torch.as_tensor(env_choice.action_low, dtype=DTYPE, device=DEVICE)
        self.action_high = torch.as_tensor(env_choice.action_high, dtype=DTYPE, device=DEVICE)
        self.action_scale = 0.5 * (self.action_high - self.action_low)
        self.action_bias = 0.5 * (self.action_high + self.action_low)
        self.r_dyn = self.build_r_dyn(env_choice.action_dim)
        self.h2 = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=DTYPE, device=DEVICE)
        self.s2 = torch.tensor([[1.0, 0.2], [0.2, -0.5]], dtype=DTYPE, device=DEVICE)

    def build_r_dyn(self, action_dim: int) -> torch.Tensor:
        mat = torch.zeros((action_dim, action_dim), dtype=DTYPE, device=DEVICE)
        for start in range(0, action_dim - 1, 2):
            mat[start, start + 1] = 1.0
            mat[start + 1, start] = -1.0
        return mat

    def scale_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        return self.action_bias + (self.action_scale * torch.tanh(raw_action))

    def perturb(self, w: torch.Tensor) -> torch.Tensor:
        if self.cfg.disturbance_type == "rotated":
            return self.r_dyn @ w
        if self.cfg.disturbance_type == "direct":
            return w
        raise ValueError(self.cfg.disturbance_type)

    def blend_action(self, u: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, float, torch.Tensor]:
        pert = self.perturb(w)
        a_env_raw = u + (self.cfg.alpha_dyn * pert)
        a_env = torch.clamp(a_env_raw, self.action_low, self.action_high)
        clip_fraction = float(((a_env_raw - a_env).abs() > 1e-12).float().mean().item())
        return a_env, clip_fraction, a_env_raw

    def reward_terms(self, reward_task_raw: float, u: torch.Tensor, w: torch.Tensor) -> tuple[float, float]:
        r_task_scaled = self.cfg.reward_scale * reward_task_raw
        reg = (-0.5 * self.cfg.a_u * float(torch.sum(u * u).item())) + (0.5 * self.cfg.a_w * float(torch.sum(w * w).item()))
        rot = 0.0
        if self.cfg.beta_rot != 0.0 and self.env_choice.action_dim >= 2:
            u2 = u[:2]
            w2 = w[:2]
            rot = (
                self.cfg.beta_rot * float(torch.dot(u2, self.h2 @ w2).item())
                + self.cfg.beta_sym * float(torch.dot(u2, self.s2 @ w2).item())
            )
        return r_task_scaled + reg + rot, r_task_scaled


class SwimmerAlignedGame:
    def __init__(self, env_choice: EnvChoice, cfg: WrapperConfig, actor_lr: float, seed: int) -> None:
        base.seed_everything(seed)
        self.env_choice = env_choice
        self.cfg = cfg
        self.actor_lr = actor_lr
        self.rng = np.random.default_rng(seed)
        self.rarl_env = AlignedRARLEnv(env_choice, cfg)
        self.actor_layout = base.FlatMLP(env_choice.obs_dim, ACTOR_HIDDEN, env_choice.action_dim)
        self.theta, self.phi = self.init_actor_params(seed)
        self.theta_target = self.theta.detach().clone()
        self.phi_target = self.phi.detach().clone()
        self.q_align = base.CriticNet(env_choice.obs_dim, env_choice.action_dim).to(DEVICE)
        self.q_align_target = base.CriticNet(env_choice.obs_dim, env_choice.action_dim).to(DEVICE)
        self.q_align_target.load_state_dict(self.q_align.state_dict())
        self.q_align_opt = torch.optim.Adam(self.q_align.parameters(), lr=CRITIC_LR)
        self.replay = ReplayBuffer(250000, env_choice.obs_dim, env_choice.action_dim)
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
            for target, online in zip(self.q_align_target.parameters(), self.q_align.parameters()):
                target.data.mul_(1.0 - POLYAK_TAU).add_(POLYAK_TAU * online.data)

    def actor_objective(self, z: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        u, w = self.actor_z(z, states)
        return self.q_align(states, u, w).mean()

    def actor_field(self, z: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        z_req = z if z.requires_grad else z.detach().clone().requires_grad_(True)
        j_val = self.actor_objective(z_req, states)
        grad = torch.autograd.grad(j_val, z_req, create_graph=True)[0]
        out = torch.zeros_like(z_req)
        out[self.field_slices["theta"]] = -grad[self.field_slices["theta"]]
        out[self.field_slices["phi"]] = +grad[self.field_slices["phi"]]
        return out

    def local_gap(self, z: torch.Tensor, states: torch.Tensor, protagonist: bool, with_prox: bool, actor_lr: float) -> torch.Tensor:
        base_z = z.detach().clone()
        current = base_z.clone()
        actor_slice = self.field_slices["theta"] if protagonist else self.field_slices["phi"]
        initial = base_z[actor_slice].clone()
        j_start = self.actor_objective(base_z, states)
        inner_lr = 0.1 * actor_lr
        for _ in range(GAP_INNER_STEPS):
            cur = current.detach().clone().requires_grad_(True)
            j_val = self.actor_objective(cur, states)
            grad = torch.autograd.grad(j_val, cur)[0][actor_slice]
            step = grad if protagonist else -grad
            if with_prox:
                step = step - ((cur[actor_slice] - initial) / max(TAU_GAP, EPS))
            next_actor = cur[actor_slice] + (inner_lr * step)
            delta = next_actor - initial
            norm = torch.linalg.norm(delta)
            if norm > LOCAL_GAP_RADIUS:
                next_actor = initial + (delta * (LOCAL_GAP_RADIUS / (norm + EPS)))
            current = cur.detach().clone()
            current[actor_slice] = next_actor.detach()
        j_end = self.actor_objective(current, states)
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
        if "field0_align" not in self.metric_refs:
            self.metric_refs["field0_align"] = field_energy
            self.metric_refs["ptau0_align"] = max(p_tau, EPS)
        field_term = field_energy / (self.metric_refs["field0_align"] + EPS)
        p_tau_term = p_tau / (self.metric_refs["ptau0_align"] + EPS)
        v_align = (LAMBDA_F * field_term) + (LAMBDA_P * p_tau_term)
        exploit_p, exploit_a = self.exploitability_parts(z, states, actor_lr)
        u, w = self.actor_z(z, states)
        j_align = float(self.q_align(states, u, w).mean().detach().item())
        out = {
            "V_align": v_align,
            "field_term_align": field_term,
            "normalized_P_tau_align": p_tau_term,
            "raw_P_tau_align": p_tau,
            "field_norm": float(torch.linalg.norm(field).item()),
            "approximate_exploitability_align": exploit_p + exploit_a,
            "exploitability_protagonist_align": exploit_p,
            "exploitability_adversary_align": exploit_a,
            "P_tau_align_protagonist_gap": p_gap,
            "P_tau_align_adversary_gap": a_gap,
            "Q_align": j_align,
        }
        if compute_geometry:
            geom = self.geometry_metrics(z, states)
            out.update(geom)
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

    def evaluate_policy(self, theta: torch.Tensor, phi: torch.Tensor | None, episodes: int, mode: str) -> dict[str, float]:
        env = base.gym.make(self.env_choice.env_id)
        aligned_returns: list[float] = []
        pure_returns_raw: list[float] = []
        pure_returns_scaled: list[float] = []
        clip_fracs: list[float] = []
        for ep in range(episodes):
            obs, _ = env.reset(seed=SEED + 9000 + ep)
            done = False
            disc = 1.0
            aligned_total = 0.0
            pure_raw = 0.0
            pure_scaled = 0.0
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
                    a_env, clip_frac, _ = self.rarl_env.blend_action(u, w)
                next_obs, reward_raw, terminated, truncated, _ = env.step(a_env.detach().cpu().numpy().astype(np.float32))
                reward_align, reward_pure_scaled = self.rarl_env.reward_terms(float(reward_raw), u, w)
                aligned_total += disc * float(reward_align)
                pure_raw += disc * float(reward_raw)
                pure_scaled += disc * float(reward_pure_scaled)
                disc *= GAMMA
                ep_clips.append(clip_frac)
                obs = next_obs
                done = terminated or truncated
            aligned_returns.append(aligned_total)
            pure_returns_raw.append(pure_raw)
            pure_returns_scaled.append(pure_scaled)
            clip_fracs.append(float(np.mean(ep_clips)) if ep_clips else 0.0)
        env.close()
        return {
            "aligned_return": float(np.mean(aligned_returns)),
            "pure_return_raw": float(np.mean(pure_returns_raw)),
            "pure_return_scaled": float(np.mean(pure_returns_scaled)),
            "action_clip_fraction": float(np.mean(clip_fracs)),
        }

    def robust_br_adversary(self, theta: torch.Tensor, diag_batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, bool]:
        base_phi = self.phi.detach().clone()
        best_phi = base_phi.clone()
        best_obj = math.inf
        states = diag_batch["obs"]
        for br_lr in BR_INNER_LR_GRID:
            current = base_phi.clone()
            for _ in range(BR_INNER_STEPS):
                cur = current.detach().clone().requires_grad_(True)
                z = torch.cat([theta.detach(), cur])
                objective = self.actor_objective(z, states)
                grad = torch.autograd.grad(objective, cur)[0]
                next_phi = cur - (br_lr * grad)
                delta = next_phi - base_phi
                norm = torch.linalg.norm(delta)
                if norm > LOCAL_BR_RADIUS:
                    next_phi = base_phi + (delta * (LOCAL_BR_RADIUS / (norm + EPS)))
                current = next_phi.detach()
            z_final = torch.cat([theta.detach(), current])
            obj = float(self.actor_objective(z_final, states).detach().item())
            if obj < best_obj:
                best_obj = obj
                best_phi = current.detach().clone()
        return best_phi, True


@dataclass
class FrozenBundle:
    env_choice: EnvChoice
    cfg: WrapperConfig
    theta0: torch.Tensor
    phi0: torch.Tensor
    q_align_state: dict[str, Any]
    q_align_target_state: dict[str, Any]
    diag_batch: dict[str, torch.Tensor]
    critic_quality: dict[str, float]


def ensure_dirs() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)


def choose_env() -> EnvChoice | None:
    rows: list[dict[str, Any]] = []
    try:
        env = base.gym.make(ENV_ID)
        action_space = env.action_space
        obs_space = env.observation_space
        choice = EnvChoice(
            env_id=ENV_ID,
            obs_dim=int(obs_space.shape[0]),
            action_dim=int(action_space.shape[0]),
            action_low=np.asarray(action_space.low, dtype=np.float32).copy(),
            action_high=np.asarray(action_space.high, dtype=np.float32).copy(),
            max_episode_steps=int(env.spec.max_episode_steps),
        )
        rows.append(
            {
                "env_id": ENV_ID,
                "available": 1,
                "obs_dim": choice.obs_dim,
                "action_dim": choice.action_dim,
                "action_low": choice.action_low.tolist(),
                "action_high": choice.action_high.tolist(),
                "max_episode_steps": choice.max_episode_steps,
                "error": "",
            }
        )
        write_csv(RESULT_ROOT / f"{PREFIX}env_availability.csv", rows)
        text = "\n".join(
            [
                f"# {PREFIX}env_availability_report",
                "",
                f"- env_id: `{choice.env_id}`",
                f"- obs_dim: `{choice.obs_dim}`",
                f"- action_dim: `{choice.action_dim}`",
                f"- action_low: `{choice.action_low.tolist()}`",
                f"- action_high: `{choice.action_high.tolist()}`",
                f"- max_episode_steps: `{choice.max_episode_steps}`",
                "- reward_components: `Gymnasium/MuJoCo Swimmer-v5 exposes scalar reward only; decomposition is not directly exported in the standard API.`",
            ]
        ) + "\n"
        write_text(RESULT_ROOT / f"{PREFIX}env_availability_report.md", text)
        env.close()
        return choice
    except Exception as exc:
        rows.append(
            {
                "env_id": ENV_ID,
                "available": 0,
                "obs_dim": "",
                "action_dim": "",
                "action_low": "",
                "action_high": "",
                "max_episode_steps": "",
                "error": repr(exc),
            }
        )
        write_csv(RESULT_ROOT / f"{PREFIX}env_availability.csv", rows)
        write_text(RESULT_ROOT / f"{PREFIX}env_unavailable.md", f"# {PREFIX}env_unavailable\n\n- env_id: `{ENV_ID}`\n- error: `{repr(exc)}`\n")
        return None


def preflight(choice: EnvChoice) -> tuple[WrapperConfig, list[WrapperConfig], dict[str, Any]]:
    env = base.gym.make(choice.env_id)
    random_rows: list[dict[str, Any]] = []
    reward_std_map: dict[tuple[str, float], float] = {}
    chosen_reward_std = None
    rotated_allowed: list[float] = []
    direct_allowed: list[float] = []
    rot_term_samples: list[float] = []
    sym_term_samples: list[float] = []
    u2_norms: list[float] = []
    w2_norms: list[float] = []
    rng = np.random.default_rng(SEED)
    for disturbance_type in DISTURBANCE_TYPES:
        for alpha in ALPHA_GRID:
            rarl_env = AlignedRARLEnv(choice, WrapperConfig(1.0, alpha, 0.001, 0.001, 0.0, 0.0, disturbance_type))
            obs, _ = env.reset(seed=SEED)
            reward_samples: list[float] = []
            clip_fracs: list[float] = []
            lengths: list[int] = []
            ep_len = 0
            for _ in range(PRELIGHT_TRANSITIONS):
                u = rng.uniform(choice.action_low, choice.action_high).astype(np.float32)
                w = rng.uniform(choice.action_low, choice.action_high).astype(np.float32)
                u_t = torch.as_tensor(u, dtype=DTYPE, device=DEVICE)
                w_t = torch.as_tensor(w, dtype=DTYPE, device=DEVICE)
                a_env, clip_frac, _ = rarl_env.blend_action(u_t, w_t)
                next_obs, reward_raw, terminated, truncated, _ = env.step(a_env.detach().cpu().numpy().astype(np.float32))
                reward_samples.append(float(reward_raw))
                clip_fracs.append(clip_frac)
                ep_len += 1
                if choice.action_dim >= 2:
                    rot_term_samples.append(float(torch.dot(u_t[:2], rarl_env.h2 @ w_t[:2]).item()))
                    sym_term_samples.append(float(torch.dot(u_t[:2], rarl_env.s2 @ w_t[:2]).item()))
                    u2_norms.append(float(torch.linalg.norm(u_t[:2]).item()))
                    w2_norms.append(float(torch.linalg.norm(w_t[:2]).item()))
                obs = next_obs
                if terminated or truncated:
                    lengths.append(ep_len)
                    ep_len = 0
                    obs, _ = env.reset()
            if ep_len > 0:
                lengths.append(ep_len)
            std_task = float(np.std(np.asarray(reward_samples, dtype=np.float64)))
            reward_std_map[(disturbance_type, alpha)] = std_task
            row = {
                "disturbance_type": disturbance_type,
                "alpha_dyn": alpha,
                "std_r_task_raw": std_task,
                "action_clip_fraction": float(np.mean(clip_fracs)),
                "episode_length_mean": float(np.mean(lengths)),
                "episode_length_std": float(np.std(lengths)),
            }
            random_rows.append(row)
            if disturbance_type == "rotated" and row["action_clip_fraction"] <= 0.05:
                rotated_allowed.append(alpha)
            if disturbance_type == "direct" and row["action_clip_fraction"] <= 0.05:
                direct_allowed.append(alpha)
    env.close()
    chosen_type = "rotated" if rotated_allowed else "direct"
    chosen_alpha = max(rotated_allowed) if rotated_allowed else (max(direct_allowed) if direct_allowed else ALPHA_GRID[0])
    chosen_reward_std = reward_std_map[(chosen_type, chosen_alpha)]
    reward_scale = 1.0 / max(chosen_reward_std, EPS)
    rot_std = float(np.std(np.asarray(rot_term_samples, dtype=np.float64))) if rot_term_samples else 0.0
    beta_candidates: list[tuple[float, float]] = []
    for beta in BETA_ROT_GRID:
        ratio = abs(beta) * rot_std / ((reward_scale * chosen_reward_std) + EPS)
        beta_candidates.append((beta, ratio))
    feasible = [item for item in beta_candidates if 0.1 <= item[1] <= 0.5]
    if feasible:
        beta_rot = min(feasible, key=lambda item: item[0])[0]
    else:
        beta_rot = min(beta_candidates, key=lambda item: abs(item[1] - 0.1))[0]
    beta_sym = 0.1 * beta_rot
    write_csv(RESULT_ROOT / f"{PREFIX}preflight.csv", random_rows)
    lines = [
        f"# {PREFIX}preflight_report",
        "",
        f"- chosen_disturbance_type: `{chosen_type}`",
        f"- chosen_alpha_dyn: `{chosen_alpha}`",
        f"- chosen_reward_scale: `{reward_scale:.6e}`",
        f"- chosen_beta_rot: `{beta_rot}`",
        f"- chosen_beta_sym: `{beta_sym}`",
        f"- rotated_alpha_allowed: `{rotated_allowed}`",
        f"- direct_alpha_allowed: `{direct_allowed}`",
        f"- std_u2_random: `{float(np.std(u2_norms)) if u2_norms else 0.0:.6e}`",
        f"- std_w2_random: `{float(np.std(w2_norms)) if w2_norms else 0.0:.6e}`",
        f"- std_rot_term_random: `{rot_std:.6e}`",
        f"- std_sym_term_random: `{float(np.std(np.asarray(sym_term_samples, dtype=np.float64))) if sym_term_samples else 0.0:.6e}`",
    ]
    write_text(RESULT_ROOT / f"{PREFIX}preflight_report.md", "\n".join(lines) + "\n")

    primary = WrapperConfig(reward_scale, chosen_alpha, 0.001, 0.001, beta_rot, beta_sym, chosen_type)
    candidates: list[WrapperConfig] = [primary]
    alt_type = "direct" if chosen_type == "rotated" else "rotated"
    alt_allowed = direct_allowed if alt_type == "direct" else rotated_allowed
    if alt_allowed:
        candidates.append(WrapperConfig(reward_scale, max(alt_allowed), 0.001, 0.001, beta_rot, beta_sym, alt_type))
    stronger_beta = None
    for beta in BETA_ROT_GRID:
        if beta > beta_rot:
            ratio = abs(beta) * rot_std / ((reward_scale * chosen_reward_std) + EPS)
            if ratio <= 0.5:
                stronger_beta = beta
                break
    if stronger_beta is not None:
        candidates.append(WrapperConfig(reward_scale, chosen_alpha, 0.001, 0.001, stronger_beta, 0.1 * stronger_beta, chosen_type))
    if len(rotated_allowed) >= 2 and chosen_type == "rotated":
        sorted_allowed = sorted(rotated_allowed)
        candidates.append(WrapperConfig(reward_scale, sorted_allowed[-2], 0.001, 0.001, beta_rot, beta_sym, chosen_type))
    seen = set()
    deduped: list[WrapperConfig] = []
    for cfg in candidates:
        key = (cfg.alpha_dyn, cfg.disturbance_type, cfg.beta_rot, cfg.beta_sym)
        if key not in seen:
            deduped.append(cfg)
            seen.add(key)
    return primary, deduped[:4], {"reward_scale": reward_scale, "rot_std": rot_std, "chosen_reward_std": chosen_reward_std}


def collect_warmup_and_snapshots(game: SwimmerAlignedGame, steps: int, exploration_std: float) -> tuple[list[dict[str, np.ndarray]], dict[str, float]]:
    env = base.gym.make(game.env_choice.env_id)
    obs, _ = env.reset(seed=SEED + 33)
    snapshots: list[dict[str, np.ndarray]] = []
    u_vals: list[np.ndarray] = []
    w_vals: list[np.ndarray] = []
    rw_vals: list[np.ndarray] = []
    clip_fracs: list[float] = []
    for _ in range(steps):
        obs_t = torch.as_tensor(obs, dtype=DTYPE, device=DEVICE).unsqueeze(0)
        u = game.actor_action(game.theta, obs_t).squeeze(0)
        w = game.actor_action(game.phi, obs_t).squeeze(0)
        if exploration_std > 0.0:
            u = torch.clamp(u + (exploration_std * torch.randn_like(u)), game.rarl_env.action_low, game.rarl_env.action_high)
            w = torch.clamp(w + (exploration_std * torch.randn_like(w)), game.rarl_env.action_low, game.rarl_env.action_high)
        if len(snapshots) < MC_SNAPSHOT_COUNT * 2:
            snapshots.append(
                {
                    "obs": np.asarray(obs, dtype=np.float32).copy(),
                    "qpos": env.unwrapped.data.qpos.copy(),
                    "qvel": env.unwrapped.data.qvel.copy(),
                }
            )
        a_env, clip_frac, a_env_raw = game.rarl_env.blend_action(u, w)
        next_obs, reward_raw, terminated, truncated, _ = env.step(a_env.detach().cpu().numpy().astype(np.float32))
        reward_align, reward_pure_scaled = game.rarl_env.reward_terms(float(reward_raw), u, w)
        done = terminated or truncated
        game.replay.add(
            np.asarray(obs, dtype=np.float32),
            u.detach().cpu().numpy().astype(np.float32),
            w.detach().cpu().numpy().astype(np.float32),
            float(reward_align),
            float(reward_pure_scaled),
            float(reward_raw),
            np.asarray(next_obs, dtype=np.float32),
            done,
        )
        u_vals.append(u.detach().cpu().numpy())
        w_vals.append(w.detach().cpu().numpy())
        rw_vals.append(game.rarl_env.perturb(w).detach().cpu().numpy())
        clip_fracs.append(clip_frac)
        obs = next_obs
        if done:
            obs, _ = env.reset()
    env.close()
    stats = {
        "std_u": float(np.std(np.asarray(u_vals, dtype=np.float64))),
        "std_w": float(np.std(np.asarray(w_vals, dtype=np.float64))),
        "std_Rw": float(np.std(np.asarray(rw_vals, dtype=np.float64))),
        "action_clip_fraction": float(np.mean(clip_fracs)),
    }
    return snapshots[:MC_SNAPSHOT_COUNT], stats


def critic_update(game: SwimmerAlignedGame) -> dict[str, float]:
    batch = game.replay.sample(BATCH_SIZE, game.rng)
    with torch.no_grad():
        u_next = game.actor_action(game.theta_target, batch["next_obs"])
        w_next = game.actor_action(game.phi_target, batch["next_obs"])
        y = batch["reward_align"] + (GAMMA * (1.0 - batch["done"]) * game.q_align_target(batch["next_obs"], u_next, w_next))
    pred = game.q_align(batch["obs"], batch["u"], batch["w"])
    loss = torch.mean((pred - y) ** 2)
    game.q_align_opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = math.sqrt(sum(float(torch.sum(p.grad.detach() * p.grad.detach()).item()) for p in game.q_align.parameters() if p.grad is not None))
    game.q_align_opt.step()
    game.polyak_update()
    return {
        "critic_loss": float(loss.detach().item()),
        "critic_grad_norm": float(grad_norm),
        "Q_align_mean": float(pred.detach().mean().item()),
        "Q_align_std": float(pred.detach().std(unbiased=False).item()),
        "Q_align_abs_mean": float(pred.detach().abs().mean().item()),
    }


def mc_quality(game: SwimmerAlignedGame, snapshots: list[dict[str, np.ndarray]]) -> dict[str, float]:
    env = base.gym.make(game.env_choice.env_id)
    q_vals: list[float] = []
    mc_vals: list[float] = []
    for snap in snapshots:
        env.reset(seed=SEED)
        env.unwrapped.set_state(snap["qpos"], snap["qvel"])
        obs_t = torch.as_tensor(np.asarray(snap["obs"], dtype=np.float32), dtype=DTYPE, device=DEVICE).unsqueeze(0)
        u = game.actor_action(game.theta, obs_t)
        w = game.actor_action(game.phi, obs_t)
        q_vals.append(float(game.q_align(obs_t, u, w).detach().item()))
        disc = 1.0
        total = 0.0
        for _ in range(MC_HORIZON):
            u_step = game.actor_action(game.theta, obs_t).squeeze(0)
            w_step = game.actor_action(game.phi, obs_t).squeeze(0)
            a_env, _, _ = game.rarl_env.blend_action(u_step, w_step)
            next_obs, reward_raw, terminated, truncated, _ = env.step(a_env.detach().cpu().numpy().astype(np.float32))
            reward_align, _ = game.rarl_env.reward_terms(float(reward_raw), u_step, w_step)
            total += disc * reward_align
            disc *= GAMMA
            obs_t = torch.as_tensor(next_obs, dtype=DTYPE, device=DEVICE).unsqueeze(0)
            if terminated or truncated:
                break
        mc_vals.append(total)
    env.close()
    corr = math.nan
    if len(q_vals) >= 2 and np.std(q_vals) > 1e-12 and np.std(mc_vals) > 1e-12:
        corr = float(np.corrcoef(np.asarray(q_vals), np.asarray(mc_vals))[0, 1])
    mse = float(np.mean((np.asarray(q_vals) - np.asarray(mc_vals)) ** 2)) if q_vals else math.nan
    return {"corr_Q_align_MC_align": corr, "mse_Q_align_MC_align": mse}


def pretrain_bundle(choice: EnvChoice, cfg: WrapperConfig) -> tuple[FrozenBundle, list[dict[str, Any]]]:
    best_noise = 0.1
    best_stats = None
    best_snapshots = None
    best_game = None
    probe_rows: list[dict[str, Any]] = []
    for noise_std in EXPLORATION_STD_GRID:
        game = SwimmerAlignedGame(choice, cfg, actor_lr=1e-5, seed=SEED)
        snapshots, stats = collect_warmup_and_snapshots(game, REPLAY_WARMUP_STEPS_GRID[0], noise_std)
        row = {"noise_std": noise_std, **stats}
        probe_rows.append(row)
        key = (0 if stats["action_clip_fraction"] <= 0.05 else 1, -stats["std_w"], -stats["std_Rw"])
        best_key = (0 if best_stats is not None and best_stats["action_clip_fraction"] <= 0.05 else 1, -(best_stats["std_w"] if best_stats else -1e9), -(best_stats["std_Rw"] if best_stats else -1e9))
        if best_game is None or key < best_key:
            best_noise = noise_std
            best_stats = stats
            best_snapshots = snapshots
            best_game = game
    assert best_game is not None and best_stats is not None and best_snapshots is not None
    if best_game.replay.size < REPLAY_WARMUP_STEPS_GRID[1]:
        # extend warmup to preferred larger coverage using chosen noise.
        extra_steps = REPLAY_WARMUP_STEPS_GRID[1] - best_game.replay.size
        extra_game = SwimmerAlignedGame(choice, cfg, actor_lr=1e-5, seed=SEED)
        best_snapshots, best_stats = collect_warmup_and_snapshots(extra_game, REPLAY_WARMUP_STEPS_GRID[1], best_noise)
        best_game = extra_game
    quality_rows: list[dict[str, Any]] = []
    for step in range(CRITIC_PRETRAIN_STEPS):
        stats = critic_update(best_game)
        if step % CRITIC_AUDIT_INTERVAL == 0 or step == CRITIC_PRETRAIN_STEPS - 1:
            stats = {**stats, **mc_quality(best_game, best_snapshots), "critic_step": step}
            quality_rows.append(stats)
    bundle = FrozenBundle(
        env_choice=choice,
        cfg=cfg,
        theta0=best_game.theta.detach().clone(),
        phi0=best_game.phi.detach().clone(),
        q_align_state=best_game.q_align.state_dict(),
        q_align_target_state=best_game.q_align_target.state_dict(),
        diag_batch={k: v.detach().clone() for k, v in best_game.replay.fixed_state_batch(BATCH_SIZE, np.random.default_rng(SEED + 777)).items()},
        critic_quality={**best_stats, **quality_rows[-1], "noise_std": best_noise},
    )
    return bundle, probe_rows + quality_rows


def clone_game(bundle: FrozenBundle, actor_lr: float) -> SwimmerAlignedGame:
    game = SwimmerAlignedGame(bundle.env_choice, bundle.cfg, actor_lr=actor_lr, seed=SEED)
    game.theta = bundle.theta0.detach().clone()
    game.phi = bundle.phi0.detach().clone()
    game.theta_target = bundle.theta0.detach().clone()
    game.phi_target = bundle.phi0.detach().clone()
    game.q_align.load_state_dict(bundle.q_align_state)
    game.q_align_target.load_state_dict(bundle.q_align_target_state)
    game.metric_refs = {}
    return game


def run_update(game: SwimmerAlignedGame, z: torch.Tensor, states: torch.Tensor, method: str, actor_lr: float, update_radius: float | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    field = game.actor_field(z, states).detach()
    f_norm = float(torch.linalg.norm(field).item())
    if method == "sgd":
        delta = -actor_lr * field
        return (z + delta).detach(), {"update_norm": float(torch.linalg.norm(delta).item()), "fallback_to_egm": 0.0, "gamma_active": 0.0, "G_contribution_ratio": 0.0}
    if method == "egm":
        z_half = z - (actor_lr * field)
        field_half = game.actor_field(z_half, states).detach()
        delta = -actor_lr * field_half
        return (z + delta).detach(), {"update_norm": float(torch.linalg.norm(delta).item()), "fallback_to_egm": 0.0, "gamma_active": 0.0, "G_contribution_ratio": 0.0}
    if method == "ppm":
        current = z.detach().clone()
        for _ in range(PPM_INNER_STEPS):
            field_inner = game.actor_field(current, states).detach()
            current = z - (actor_lr * field_inner)
        delta = current - z
        return current.detach(), {"update_norm": float(torch.linalg.norm(delta).item()), "fallback_to_egm": 0.0, "gamma_active": 0.0, "G_contribution_ratio": 0.0}

    def eval_v(cur_z: torch.Tensor) -> float:
        return float(game.diagnostic_metrics(cur_z, {"obs": states}, actor_lr, compute_geometry=False)["V_align"])

    if method == "proposed_noG":
        ts = [0.0, actor_lr, 2.0 * actor_lr]
        vals = np.asarray([eval_v(z - (t * field)) for t in ts], dtype=np.float64)
        a, b, c = np.polyfit(np.asarray(ts, dtype=np.float64), vals, 2)
        beta = float(-b / (2.0 * a)) if abs(a) > 1e-12 and a > 0 else actor_lr
        delta = -beta * field
        raw_norm = float(torch.linalg.norm(delta).item())
        trust_active = 0.0
        if update_radius is not None and raw_norm > update_radius:
            delta = delta * (update_radius / (raw_norm + EPS))
            trust_active = 1.0
        z_next = (z + delta).detach()
        v_before = eval_v(z)
        v_after = eval_v(z_next)
        fallback = 0.0
        if (not finite(v_after)) or v_after > (1.25 * v_before + 1e-12):
            z_next, egm_meta = run_update(game, z, states, "egm", actor_lr)
            delta = z_next - z
            v_after = eval_v(z_next)
            fallback = 1.0
        return z_next, {
            "update_norm": float(torch.linalg.norm(delta).item()),
            "beta": beta,
            "gamma": 0.0,
            "fallback_to_egm": fallback,
            "gamma_active": 0.0,
            "G_contribution_ratio": 0.0,
            "trust_radius_active": trust_active,
            "V_before": v_before,
            "V_after": v_after,
            "V_change": v_after - v_before,
            "prediction_error": math.nan,
            "cos_fg": math.nan,
            "g_over_f": math.nan,
            "non_collinearity": math.nan,
        }
    if method == "proposed_QP_G":
        z_req = z.detach().clone().requires_grad_(True)
        _, g_vec = torch.autograd.functional.jvp(lambda zz: game.actor_field(zz, states), (z_req,), (field,), create_graph=False, strict=False)
        g_vec = g_vec.detach()
        g_norm = float(torch.linalg.norm(g_vec).item())
        f_unit = -field / (torch.linalg.norm(field) + EPS)
        g_unit = g_vec / (torch.linalg.norm(g_vec) + EPS)
        scale = actor_lr
        sample_points = [
            (0.0, 0.0),
            (scale, 0.0),
            (-scale, 0.0),
            (0.0, scale),
            (0.0, -scale),
            (scale, scale),
        ]
        a_rows = []
        y_vals = []
        for x, y in sample_points:
            dz = (x * f_unit) + (y * g_unit)
            a_rows.append([1.0, x, y, x * x, x * y, y * y])
            y_vals.append(eval_v(z + dz))
        coeffs, *_ = np.linalg.lstsq(np.asarray(a_rows, dtype=np.float64), np.asarray(y_vals, dtype=np.float64), rcond=None)
        c0, c1, c2, c3, c4, c5 = coeffs.tolist()
        hess = np.asarray([[2.0 * c3, c4], [c4, 2.0 * c5]], dtype=np.float64)
        grad = np.asarray([c1, c2], dtype=np.float64)
        try:
            step_xy = -np.linalg.solve(hess + (1e-8 * np.eye(2)), grad)
        except np.linalg.LinAlgError:
            step_xy = np.zeros(2, dtype=np.float64)
        x, y = float(step_xy[0]), float(step_xy[1])
        delta = (x * f_unit) + (y * g_unit)
        raw_norm = float(torch.linalg.norm(delta).item())
        trust_active = 0.0
        if update_radius is not None and raw_norm > update_radius:
            delta = delta * (update_radius / (raw_norm + EPS))
            trust_active = 1.0
        z_next = (z + delta).detach()
        v_before = eval_v(z)
        v_after = eval_v(z_next)
        fallback = 0.0
        if (not finite(v_after)) or v_after > (1.25 * v_before + 1e-12):
            z_next, _ = run_update(game, z, states, "egm", actor_lr)
            delta = z_next - z
            v_after = eval_v(z_next)
            fallback = 1.0
        cos_fg = float(torch.dot(field, g_vec).item() / ((f_norm * g_norm) + EPS))
        return z_next, {
            "update_norm": float(torch.linalg.norm(delta).item()),
            "beta": x,
            "gamma": y,
            "fallback_to_egm": fallback,
            "gamma_active": float(abs(y) > 1e-12),
            "G_contribution_ratio": float(abs(y) / (abs(x) + EPS)),
            "trust_radius_active": trust_active,
            "V_before": v_before,
            "V_after": v_after,
            "V_change": v_after - v_before,
            "prediction_error": math.nan,
            "cos_fg": cos_fg,
            "g_over_f": g_norm / (f_norm + EPS),
            "non_collinearity": math.sqrt(max(0.0, 1.0 - min(1.0, cos_fg * cos_fg))),
        }
    raise ValueError(method)


def run_method(bundle: FrozenBundle, method: str, actor_lr: float, num_iters: int, update_radius: float | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    game = clone_game(bundle, actor_lr)
    diag_batch = {k: v.detach().clone() for k, v in bundle.diag_batch.items()}
    curves: list[dict[str, Any]] = []
    clean_eval = game.evaluate_policy(game.theta, None, 3, "clean")
    current_eval = game.evaluate_policy(game.theta, game.phi, 3, "current")
    br_phi, br_valid = game.robust_br_adversary(game.theta, diag_batch)
    robust_eval = game.evaluate_policy(game.theta, br_phi, 3, "current")
    for iteration in range(num_iters + 1):
        z = game.current_z()
        diag = game.diagnostic_metrics(z, diag_batch, actor_lr, compute_geometry=(iteration % PLOT_EVAL_INTERVAL == 0))
        if iteration % PLOT_EVAL_INTERVAL == 0 or iteration == num_iters:
            clean_eval = game.evaluate_policy(game.theta, None, 3, "clean")
            current_eval = game.evaluate_policy(game.theta, game.phi, 3, "current")
            br_phi, br_valid = game.robust_br_adversary(game.theta, diag_batch)
            robust_eval = game.evaluate_policy(game.theta, br_phi, 3, "current")
        row = {
            "method": method,
            "iteration": iteration,
            "actor_lr": actor_lr,
            **bundle.cfg.__dict__,
            **diag,
            "clean_aligned_return": clean_eval["aligned_return"],
            "current_adv_aligned_return": current_eval["aligned_return"],
            "robust_br_aligned_return": robust_eval["aligned_return"],
            "aligned_robust_degradation": clean_eval["aligned_return"] - robust_eval["aligned_return"],
            "clean_pure_return": clean_eval["pure_return_raw"],
            "current_adv_pure_return": current_eval["pure_return_raw"],
            "robust_br_pure_return": robust_eval["pure_return_raw"],
            "pure_robust_degradation": clean_eval["pure_return_raw"] - robust_eval["pure_return_raw"],
            "action_clip_fraction": current_eval["action_clip_fraction"],
            "robust_br_valid": int(br_valid),
            "protagonist_param_norm": float(torch.linalg.norm(z[game.field_slices["theta"]]).item()),
            "adversary_param_norm": float(torch.linalg.norm(z[game.field_slices["phi"]]).item()),
            "nan_flag": 0,
            "valid_flag": 1,
        }
        row["nan_flag"] = int(
            not all(
                finite(row[key])
                for key in [
                    "V_align",
                    "field_norm",
                    "raw_P_tau_align",
                    "approximate_exploitability_align",
                    "clean_aligned_return",
                    "current_adv_aligned_return",
                    "robust_br_aligned_return",
                ]
            )
        )
        row["valid_flag"] = int(
            row["nan_flag"] == 0
            and row["action_clip_fraction"] <= 0.05
            and row["robust_br_valid"] == 1
        )
        curves.append(row)
        if iteration < num_iters:
            next_z, meta = run_update(game, z, diag_batch["obs"], method, actor_lr, update_radius)
            game.set_from_z(next_z)
            curves[-1].update(meta)
    vals_v = [row["V_align"] for row in curves]
    vals_p = [row["normalized_P_tau_align"] for row in curves]
    vals_f = [row["field_norm"] for row in curves]
    vals_e = [row["approximate_exploitability_align"] for row in curves]
    first = curves[0]
    last = curves[-1]
    summary = {
        "method": method,
        "actor_lr": actor_lr,
        "valid_flag": int(all(int(r["valid_flag"]) == 1 for r in curves)),
        "V_align_start": first["V_align"],
        "V_align_final": last["V_align"],
        "V_align_AUC": float(sum(vals_v)),
        "V_align_spike_ratio": spike_ratio(vals_v),
        "P_tau_align_start": first["normalized_P_tau_align"],
        "P_tau_align_final": last["normalized_P_tau_align"],
        "P_tau_align_AUC": float(sum(vals_p)),
        "P_tau_align_spike_ratio": spike_ratio(vals_p),
        "field_norm_start": first["field_norm"],
        "field_norm_final": last["field_norm"],
        "field_norm_AUC": float(sum(vals_f)),
        "field_norm_spike_ratio": spike_ratio(vals_f),
        "exploitability_AUC": float(sum(vals_e)),
        "clean_aligned_return_final": last["clean_aligned_return"],
        "current_adv_aligned_return_final": last["current_adv_aligned_return"],
        "robust_br_aligned_return_final": last["robust_br_aligned_return"],
        "aligned_robust_degradation_final": last["aligned_robust_degradation"],
        "curve_normal_flag": int(
            last["V_align"] <= first["V_align"] + 1e-8
            and last["normalized_P_tau_align"] <= first["normalized_P_tau_align"] + 1e-8
            and spike_ratio(vals_v) <= 5.0
            and spike_ratio(vals_p) <= 5.0
            and spike_ratio(vals_f) <= 5.0
        ),
        "fallback_to_egm_frac": mean_or_zero([r.get("fallback_to_egm", 0.0) for r in curves]),
        "gamma_active_frac": mean_or_zero([r.get("gamma_active", 0.0) for r in curves]),
        "G_contribution_ratio": mean_or_zero([r.get("G_contribution_ratio", 0.0) for r in curves if finite(r.get("G_contribution_ratio", math.nan))]),
    }
    return curves, summary


def mean_or_zero(values: list[float]) -> float:
    arr = [float(v) for v in values if finite(v)]
    return float(np.mean(arr)) if arr else 0.0


def plot_curves(rows: list[dict[str, Any]], prefix_name: str, methods: list[str], final_stage: bool) -> None:
    if base.plt is None or not rows:
        return
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["method"], []).append(row)
    colors = {
        "sgd": "#1f77b4",
        "egm": "#ff7f0e",
        "ppm": "#2ca02c",
        "proposed_noG": "#d62728",
        "proposed_QP_G": "#9467bd",
    }
    name_map = {
        "V_align": f"{prefix_name}_V_align.png",
        "normalized_P_tau_align": f"{prefix_name}_P_tau_align.png",
        "field_norm": f"{prefix_name}_field_norm.png",
        "approximate_exploitability_align": f"{prefix_name}_exploitability.png",
    }
    for metric, name in name_map.items():
        fig, ax = base.plt.subplots(figsize=(7, 4))
        for method in methods:
            method_rows = groups.get(method, [])
            if not method_rows:
                continue
            ax.plot([r["iteration"] for r in method_rows], [r[metric] for r in method_rows], label=method, color=colors[method], linewidth=1.8)
        ax.set_title(metric)
        ax.set_xlabel("Iteration")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_ROOT / name, dpi=180)
        base.plt.close(fig)
    if final_stage:
        fig, axes = base.plt.subplots(4, 2, figsize=(14, 14), sharex=True)
        panels = [
            ("V_align", "V_align"),
            ("normalized_P_tau_align", "P_tau_align"),
            ("field_norm", "field_norm"),
            ("approximate_exploitability_align", "exploitability"),
            ("robust_br_aligned_return", "robust_br_aligned_return"),
            ("aligned_robust_degradation", "aligned_robust_degradation"),
            ("clean_pure_return", "pure_task_clean"),
            ("current_adv_pure_return", "pure_task_current"),
        ]
        for ax, (metric, title) in zip(axes.flat, panels):
            for method in methods:
                method_rows = groups.get(method, [])
                if not method_rows:
                    continue
                ax.plot([r["iteration"] for r in method_rows], [r[metric] for r in method_rows], label=method, color=colors[method], linewidth=1.6)
            ax.set_title(title)
        axes[0, 0].legend()
        fig.tight_layout()
        fig.savefig(PLOT_ROOT / f"{prefix_name}_all_plots_big.png", dpi=180)
        base.plt.close(fig)
        fig, axes = base.plt.subplots(1, 5, figsize=(20, 3.8))
        panels2 = [
            ("V_align", "V_align"),
            ("normalized_P_tau_align", "P_tau_align"),
            ("field_norm", "field_norm"),
            ("robust_br_aligned_return", "robust_BR_align"),
            ("aligned_robust_degradation", "align_degradation"),
        ]
        for ax, (metric, title) in zip(axes, panels2):
            for method in methods:
                method_rows = groups.get(method, [])
                if not method_rows:
                    continue
                ax.plot([r["iteration"] for r in method_rows], [r[metric] for r in method_rows], label=method, color=colors[method], linewidth=1.5)
            ax.set_title(title)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(PLOT_ROOT / f"{prefix_name}_paper_main.png", dpi=180)
        base.plt.close(fig)
    else:
        fig, axes = base.plt.subplots(5, 1, figsize=(8, 15), sharex=True)
        panels = [
            ("V_align", "V_align"),
            ("normalized_P_tau_align", "P_tau_align"),
            ("field_norm", "field_norm"),
            ("approximate_exploitability_align", "exploitability"),
            ("robust_br_aligned_return", "robust_br_aligned_return"),
        ]
        for ax, (metric, title) in zip(axes, panels):
            for method in methods:
                method_rows = groups.get(method, [])
                if not method_rows:
                    continue
                ax.plot([r["iteration"] for r in method_rows], [r[metric] for r in method_rows], label=method, color=colors[method], linewidth=1.7)
            ax.set_title(title)
        axes[0].legend()
        axes[-1].set_xlabel("Iteration")
        fig.tight_layout()
        suffix = "sgd_aligned_rarl_performance" if methods == ["sgd"] else "baseline_aligned_rarl_performance"
        fig.savefig(PLOT_ROOT / f"{prefix_name}_{suffix}.png", dpi=180)
        base.plt.close(fig)
        fig, axes = base.plt.subplots(3, 2, figsize=(14, 12))
        panels2 = [
            ("V_align", "V_align"),
            ("normalized_P_tau_align", "P_tau_align"),
            ("field_norm", "field_norm"),
            ("approximate_exploitability_align", "exploitability"),
            ("robust_br_aligned_return", "robust_BR"),
            ("aligned_robust_degradation", "align_degradation"),
        ]
        for ax, (metric, title) in zip(axes.flat, panels2):
            for method in methods:
                method_rows = groups.get(method, [])
                if not method_rows:
                    continue
                ax.plot([r["iteration"] for r in method_rows], [r[metric] for r in method_rows], label=method, color=colors[method], linewidth=1.6)
            ax.set_title(title)
        axes[0, 0].legend()
        fig.tight_layout()
        fig.savefig(PLOT_ROOT / f"{prefix_name}_all_plots_big.png", dpi=180)
        base.plt.close(fig)


def choose_geometry_cfg(choice: EnvChoice, candidates: list[WrapperConfig]) -> tuple[FrozenBundle | None, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    best_bundle = None
    best_score = None
    audit_path = RESULT_ROOT / f"{PREFIX}critic_quality_audit.csv"
    if audit_path.exists():
        audit_path.unlink()
    for cfg in candidates:
        bundle, quality_rows = pretrain_bundle(choice, cfg)
        row = {
            "alpha_dyn": cfg.alpha_dyn,
            "disturbance_type": cfg.disturbance_type,
            "reward_scale": cfg.reward_scale,
            "beta_rot": cfg.beta_rot,
            "beta_sym": cfg.beta_sym,
            "a_u": cfg.a_u,
            "a_w": cfg.a_w,
            "warmup_noise_std": bundle.critic_quality["noise_std"],
            "std_u": bundle.critic_quality["std_u"],
            "std_w": bundle.critic_quality["std_w"],
            "std_Rw": bundle.critic_quality["std_Rw"],
            "action_clip_fraction": bundle.critic_quality["action_clip_fraction"],
            "critic_loss": bundle.critic_quality["critic_loss"],
            "Q_align_mean": bundle.critic_quality["Q_align_mean"],
            "Q_align_std": bundle.critic_quality["Q_align_std"],
            "Q_align_abs_mean": bundle.critic_quality["Q_align_abs_mean"],
            "corr_Q_align_MC_align": bundle.critic_quality["corr_Q_align_MC_align"],
            "mse_Q_align_MC_align": bundle.critic_quality["mse_Q_align_MC_align"],
        }
        game = clone_game(bundle, actor_lr=1e-5)
        geom = game.diagnostic_metrics(game.current_z(), bundle.diag_batch, 1e-5, compute_geometry=True)
        row.update(
            {
                "field_norm": geom["field_norm"],
                "g_over_f": geom["g_over_f"],
                "cos_fg": geom["cos_fg"],
                "non_collinearity": geom["non_collinearity"],
                "rotation_ratio_proxy": geom["rotation_ratio_proxy"],
                "cross_player_coupling_proxy": geom["cross_player_coupling_proxy"],
                "cross_to_same_ratio": geom["cross_to_same_ratio"],
                "geometry_pass": int(
                    geom["cross_player_coupling_proxy"] > 0.0
                    and geom["cross_to_same_ratio"] > 0.05
                    and geom["non_collinearity"] > 0.2
                    and geom["rotation_ratio_proxy"] >= 1e-3
                ),
            }
        )
        rows.append(row)
        score = (
            (2.0 * row["geometry_pass"])
            + row["rotation_ratio_proxy"]
            + (0.1 * row["cross_to_same_ratio"])
            + (0.01 * row["cross_player_coupling_proxy"])
            + (0.001 * max(row["corr_Q_align_MC_align"], -10.0))
        )
        if row["action_clip_fraction"] <= 0.05 and finite(row["critic_loss"]) and (best_score is None or score > best_score):
            best_score = score
            best_bundle = bundle
        # append critic audit rows tagged by cfg
        audit_rows = []
        for qrow in quality_rows:
            qrow = dict(qrow)
            qrow.update({"alpha_dyn": cfg.alpha_dyn, "disturbance_type": cfg.disturbance_type, "beta_rot": cfg.beta_rot})
            audit_rows.append(qrow)
        if audit_path.exists():
            existing = list(csv.DictReader(audit_path.open(encoding="utf-8")))
            merged = [{k: v for k, v in row.items()} for row in existing] + audit_rows
            write_csv(audit_path, merged)
        else:
            write_csv(audit_path, audit_rows)
    write_csv(RESULT_ROOT / f"{PREFIX}geometry_audit.csv", rows)
    lines = [f"# {PREFIX}geometry_audit", ""]
    for row in rows:
        lines.append(
            f"- alpha_dyn=`{row['alpha_dyn']}`, disturbance_type=`{row['disturbance_type']}`, beta_rot=`{row['beta_rot']}`: clip=`{row['action_clip_fraction']:.6e}`, "
            f"corr_Q_align_MC=`{row['corr_Q_align_MC_align']:.3f}`, rotation=`{row['rotation_ratio_proxy']:.6e}`, cross=`{row['cross_player_coupling_proxy']:.6e}`, "
            f"cross_ratio=`{row['cross_to_same_ratio']:.6e}`, noncol=`{row['non_collinearity']:.6e}`, geometry_pass=`{bool(row['geometry_pass'])}`"
        )
    if not any(int(row["geometry_pass"]) == 1 for row in rows):
        lines.extend(["", "- No candidate reached `rotation_ratio_proxy >= 1e-3` together with the other geometry gate conditions."])
    write_text(RESULT_ROOT / f"{PREFIX}geometry_audit.md", "\n".join(lines) + "\n")
    # write consolidated critic audit report
    critic_rows = list(csv.DictReader((RESULT_ROOT / f"{PREFIX}critic_quality_audit.csv").open(encoding="utf-8")))
    report_lines = [f"# {PREFIX}critic_quality_audit", ""]
    for cfg_key in sorted({(row["alpha_dyn"], row["disturbance_type"], row["beta_rot"]) for row in critic_rows}):
        sub = [row for row in critic_rows if (row["alpha_dyn"], row["disturbance_type"], row["beta_rot"]) == cfg_key]
        last = sub[-1]
        report_lines.append(
            f"- alpha_dyn=`{cfg_key[0]}`, disturbance_type=`{cfg_key[1]}`, beta_rot=`{cfg_key[2]}`: "
            f"critic_loss_final=`{float(last['critic_loss']):.6e}`, Q_align_abs_mean_final=`{float(last['Q_align_abs_mean']):.6e}`, "
            f"corr_Q_align_MC=`{float(last['corr_Q_align_MC_align']):.3f}`"
        )
    write_text(RESULT_ROOT / f"{PREFIX}critic_quality_audit.md", "\n".join(report_lines) + "\n")
    lines_protocol = [
        f"# {PREFIX}frozen_critic_protocol",
        "",
        f"- replay_warmup_steps: `{REPLAY_WARMUP_STEPS_GRID[1]}`",
        f"- critic_pretrain_steps: `{CRITIC_PRETRAIN_STEPS}`",
        f"- critic_lr: `{CRITIC_LR}`",
        f"- batch_size: `{BATCH_SIZE}`",
        f"- gamma: `{GAMMA}`",
        f"- polyak_tau: `{POLYAK_TAU}`",
        "- during_actor_optimization: `critic frozen, target critic frozen, fixed diagnostic batch`",
    ]
    write_text(RESULT_ROOT / f"{PREFIX}frozen_critic_protocol.md", "\n".join(lines_protocol) + "\n")
    return best_bundle, rows


def sgd_gate(bundle: FrozenBundle) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, float | None]:
    all_curves: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for actor_lr in ACTOR_LR_GRID:
        curves, summary = run_method(bundle, "sgd", actor_lr, NUM_ACTOR_ITERS)
        all_curves.extend(curves)
        summaries.append(summary)
    write_csv(RESULT_ROOT / f"{PREFIX}sgd_gate_curves.csv", all_curves)
    write_csv(RESULT_ROOT / f"{PREFIX}sgd_gate.csv", summaries)
    valid_normals = [row for row in summaries if row["valid_flag"] == 1 and row["curve_normal_flag"] == 1]
    decision = "SGD_NORMAL_PASS" if valid_normals else "SGD_NORMAL_FAIL"
    best_lr = min(valid_normals, key=lambda row: row["V_align_AUC"])["actor_lr"] if valid_normals else None
    lines = [f"# {PREFIX}sgd_gate_report", "", f"- decision: `{decision}`", f"- best_actor_lr: `{best_lr}`", ""]
    for row in summaries:
        lines.append(
            f"- lr `{row['actor_lr']}`: valid=`{bool(row['valid_flag'])}`, curve_normal=`{bool(row['curve_normal_flag'])}`, "
            f"V_align_start=`{row['V_align_start']:.6e}`, V_align_final=`{row['V_align_final']:.6e}`, "
            f"P_tau_start=`{row['P_tau_align_start']:.6e}`, P_tau_final=`{row['P_tau_align_final']:.6e}`, "
            f"field_norm_start=`{row['field_norm_start']:.6e}`, field_norm_final=`{row['field_norm_final']:.6e}`, "
            f"robust_br_aligned_return_final=`{row['robust_br_aligned_return_final']:.6e}`"
        )
    write_text(RESULT_ROOT / f"{PREFIX}sgd_gate_report.md", "\n".join(lines) + "\n")
    plot_curves(all_curves, PREFIX + "sgd", ["sgd"], final_stage=False)
    return all_curves, summaries, decision, best_lr


def baseline_gate(bundle: FrozenBundle, actor_lr: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    all_curves: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for method in ["sgd", "egm", "ppm"]:
        curves, summary = run_method(bundle, method, actor_lr, NUM_ACTOR_ITERS)
        all_curves.extend(curves)
        summaries.append(summary)
    write_csv(RESULT_ROOT / f"{PREFIX}baseline_gate_curves.csv", all_curves)
    write_csv(RESULT_ROOT / f"{PREFIX}baseline_gate.csv", summaries)
    by_method = {row["method"]: row for row in summaries}
    sgd = by_method["sgd"]
    egm = by_method["egm"]
    ppm = by_method["ppm"]
    all_valid = all(row["valid_flag"] == 1 for row in summaries)
    all_normal = all(row["curve_normal_flag"] == 1 for row in summaries)
    egm_adv = (sgd["V_align_AUC"] / (egm["V_align_AUC"] + EPS) >= 1.3) or (sgd["P_tau_align_AUC"] / (egm["P_tau_align_AUC"] + EPS) >= 1.3)
    ppm_adv = (sgd["V_align_AUC"] / (ppm["V_align_AUC"] + EPS) >= 1.3) or (sgd["P_tau_align_AUC"] / (ppm["P_tau_align_AUC"] + EPS) >= 1.3)
    winner = None
    if egm_adv:
        winner = "egm"
    if ppm_adv and (winner is None or ppm["V_align_AUC"] < by_method[winner]["V_align_AUC"]):
        winner = "ppm"
    comparable_field = False
    robust_ok = False
    if winner is not None:
        win = by_method[winner]
        comparable_field = win["field_norm_AUC"] <= (1.1 * sgd["field_norm_AUC"] + EPS)
        robust_ok = win["robust_br_aligned_return_final"] >= (sgd["robust_br_aligned_return_final"] - 1e-6)
    geom_row = max(load_csv_rows(RESULT_ROOT / f"{PREFIX}geometry_audit.csv"), key=lambda row: float(row["rotation_ratio_proxy"]))
    geom_pass = (
        float(geom_row["cross_player_coupling_proxy"]) > 0.0
        and float(geom_row["cross_to_same_ratio"]) > 0.05
        and float(geom_row["non_collinearity"]) > 0.2
        and float(geom_row["rotation_ratio_proxy"]) >= 1e-3
    )
    decision = "BASELINE_READY_FOR_PROPOSED"
    if not geom_pass:
        decision = "BASELINE_FAIL_GEOMETRY_WEAK"
    elif not all_valid or not all_normal:
        decision = "BASELINE_FAIL_CURVES_ABNORMAL"
    elif winner is None or not comparable_field:
        decision = "BASELINE_FAIL_NO_EGM_PPM_ADVANTAGE"
    elif not robust_ok:
        decision = "BASELINE_FAIL_RARL_PERFORMANCE"
    lines = [f"# {PREFIX}baseline_gate_report", "", f"- decision: `{decision}`", f"- actor_lr: `{actor_lr}`", ""]
    for row in summaries:
        lines.append(
            f"- {row['method'].upper()}: valid=`{bool(row['valid_flag'])}`, curve_normal=`{bool(row['curve_normal_flag'])}`, "
            f"V_align_AUC=`{row['V_align_AUC']:.6e}`, P_tau_align_AUC=`{row['P_tau_align_AUC']:.6e}`, field_norm_AUC=`{row['field_norm_AUC']:.6e}`, "
            f"robust_br_aligned_return_final=`{row['robust_br_aligned_return_final']:.6e}`"
        )
    lines.extend(
        [
            "",
            f"- egm_beats_sgd: `{egm_adv}`",
            f"- ppm_beats_sgd: `{ppm_adv}`",
            f"- winner: `{winner}`",
            f"- winner_field_norm_comparable: `{comparable_field}`",
            f"- winner_robust_not_worse: `{robust_ok}`",
            f"- geometry_gate_pass: `{geom_pass}`",
        ]
    )
    write_text(RESULT_ROOT / f"{PREFIX}baseline_gate_report.md", "\n".join(lines) + "\n")
    plot_curves(all_curves, PREFIX + "baseline", ["sgd", "egm", "ppm"], final_stage=False)
    return all_curves, summaries, decision


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open(encoding="utf-8")))


def proposed_radius_preflight(bundle: FrozenBundle, actor_lr: float) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    selected: dict[str, float] = {}
    for method in ["proposed_noG", "proposed_QP_G"]:
        chosen = None
        for radius in UPDATE_RADIUS_GRID:
            curves, summary = run_method(bundle, method, actor_lr, PROPOSED_PREFLIGHT_ITERS, update_radius=radius)
            row = {
                "method": method,
                "update_radius": radius,
                "V_align_AUC": summary["V_align_AUC"],
                "P_tau_align_AUC": summary["P_tau_align_AUC"],
                "field_norm_AUC": summary["field_norm_AUC"],
                "exploitability_AUC": summary["exploitability_AUC"],
                "clean_aligned_return_final": summary["clean_aligned_return_final"],
                "current_adv_aligned_return_final": summary["current_adv_aligned_return_final"],
                "robust_br_aligned_return_final": summary["robust_br_aligned_return_final"],
                "fallback_to_egm_frac": summary["fallback_to_egm_frac"],
                "gamma_active_frac": summary["gamma_active_frac"],
                "G_contribution_ratio": summary["G_contribution_ratio"],
                "valid_flag": summary["valid_flag"],
            }
            rows.append(row)
            valid = (
                summary["valid_flag"] == 1
                and summary["V_align_final"] <= summary["V_align_start"] + 1e-8
                and summary["P_tau_align_final"] <= summary["P_tau_align_start"] + 1e-8
                and summary["fallback_to_egm_frac"] < 0.2
            )
            if chosen is None and valid:
                chosen = radius
        if chosen is None:
            best = min([row for row in rows if row["method"] == method and row["valid_flag"] == 1], key=lambda row: row["V_align_AUC"])
            chosen = best["update_radius"]
        selected[method] = float(chosen)
    write_csv(RESULT_ROOT / f"{PREFIX}proposed_radius_preflight.csv", rows)
    lines = [f"# {PREFIX}proposed_radius_preflight_report", ""]
    for method in ["proposed_noG", "proposed_QP_G"]:
        lines.append(f"- {method}: selected_update_radius=`{selected[method]}`")
    write_text(RESULT_ROOT / f"{PREFIX}proposed_radius_preflight_report.md", "\n".join(lines) + "\n")
    return selected, rows


def final_stage(bundle: FrozenBundle, actor_lr: float) -> str:
    radius_map, _ = proposed_radius_preflight(bundle, actor_lr)
    all_curves: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for method in ["sgd", "egm", "ppm", "proposed_noG", "proposed_QP_G"]:
        radius = radius_map.get(method)
        curves, summary = run_method(bundle, method, actor_lr, NUM_ACTOR_ITERS, update_radius=radius)
        all_curves.extend(curves)
        summaries.append(summary)
    write_csv(RESULT_ROOT / f"{PREFIX}final_curves.csv", all_curves)
    write_csv(RESULT_ROOT / f"{PREFIX}final_summary.csv", summaries)
    diag_rows = [row for row in all_curves if row["method"] in {"proposed_noG", "proposed_QP_G"}]
    write_csv(RESULT_ROOT / f"{PREFIX}final_diagnostics.csv", diag_rows)
    plot_curves(all_curves, PREFIX + "final", ["sgd", "egm", "ppm", "proposed_noG", "proposed_QP_G"], final_stage=True)
    by_method = {row["method"]: row for row in summaries}
    qpg = by_method["proposed_QP_G"]
    nog = by_method["proposed_noG"]
    baselines = [by_method["sgd"], by_method["egm"], by_method["ppm"]]
    qpg_beats_nog = (
        qpg["V_align_AUC"] < nog["V_align_AUC"]
        and qpg["P_tau_align_AUC"] < nog["P_tau_align_AUC"]
        and qpg["field_norm_AUC"] <= nog["field_norm_AUC"]
        and qpg["exploitability_AUC"] <= nog["exploitability_AUC"]
        and qpg["robust_br_aligned_return_final"] >= nog["robust_br_aligned_return_final"] - 1e-6
        and qpg["aligned_robust_degradation_final"] <= nog["aligned_robust_degradation_final"] + 1e-6
    )
    baseline_match = all(
        qpg["V_align_AUC"] <= base_row["V_align_AUC"] + 1e-6
        or qpg["robust_br_aligned_return_final"] >= base_row["robust_br_aligned_return_final"] - 1e-6
        for base_row in baselines
    )
    decision = "PROPOSED_FAIL"
    if qpg_beats_nog and baseline_match and qpg["fallback_to_egm_frac"] < 0.2 and qpg["gamma_active_frac"] > 0.05 and qpg["G_contribution_ratio"] > 0.01:
        robust_superior = all(
            qpg["robust_br_aligned_return_final"] >= base_row["robust_br_aligned_return_final"] - 1e-6
            and qpg["aligned_robust_degradation_final"] <= base_row["aligned_robust_degradation_final"] + 1e-6
            for base_row in baselines
        )
        decision = "PROPOSED_FULL_POSITIVE" if robust_superior else "PROPOSED_OPTIMIZATION_ONLY"
    lines = [f"# {PREFIX}final_report", "", f"- decision: `{decision}`", ""]
    for row in summaries:
        lines.append(
            f"- {row['method']}: V_align_AUC=`{row['V_align_AUC']:.6e}`, P_tau_align_AUC=`{row['P_tau_align_AUC']:.6e}`, "
            f"field_norm_AUC=`{row['field_norm_AUC']:.6e}`, exploitability_AUC=`{row['exploitability_AUC']:.6e}`, "
            f"robust_br_aligned_return_final=`{row['robust_br_aligned_return_final']:.6e}`, aligned_robust_degradation_final=`{row['aligned_robust_degradation_final']:.6e}`"
        )
    write_text(RESULT_ROOT / f"{PREFIX}final_report.md", "\n".join(lines) + "\n")
    return decision


def write_final_decision(decision: str, notes: list[str]) -> None:
    lines = [f"# {PREFIX}final_decision", "", f"- decision: `{decision}`", ""]
    lines.extend([f"- {note}" for note in notes])
    write_text(RESULT_ROOT / f"{PREFIX}final_decision.md", "\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()
    base.seed_everything(SEED)
    choice = choose_env()
    if choice is None:
        write_final_decision("ENV_UNAVAILABLE", ["Swimmer-v5 was unavailable in this environment."])
        return

    primary_cfg, geometry_candidates, preflight_meta = preflight(choice)
    best_bundle, geom_rows = choose_geometry_cfg(choice, geometry_candidates)
    if best_bundle is None:
        write_final_decision("GEOMETRY_FAIL", ["No candidate bundle passed clipping/critic sanity enough to proceed."])
        return

    best_geom = max(geom_rows, key=lambda row: (row["geometry_pass"], row["rotation_ratio_proxy"], row["cross_to_same_ratio"]))
    geom_pass = int(best_geom["geometry_pass"]) == 1
    if not geom_pass:
        write_final_decision(
            "GEOMETRY_FAIL",
            [
                f"Best geometry candidate still failed the gate: disturbance_type={best_geom['disturbance_type']}, alpha_dyn={best_geom['alpha_dyn']}, beta_rot={best_geom['beta_rot']}.",
                f"rotation_ratio_proxy={float(best_geom['rotation_ratio_proxy']):.6e}, cross_to_same_ratio={float(best_geom['cross_to_same_ratio']):.6e}.",
            ],
        )
        return

    _, sgd_summaries, sgd_decision, best_lr = sgd_gate(best_bundle)
    if sgd_decision != "SGD_NORMAL_PASS" or best_lr is None:
        write_final_decision("SGD_NORMAL_FAIL", ["Frozen-critic SGD gate failed on Swimmer-v5 aligned robust objective."])
        return

    _, baseline_summaries, baseline_decision = baseline_gate(best_bundle, best_lr)
    if baseline_decision != "BASELINE_READY_FOR_PROPOSED":
        write_final_decision("BASELINE_FAIL", [f"Baseline gate decision was {baseline_decision}."])
        return

    final_decision = final_stage(best_bundle, best_lr)
    write_final_decision(final_decision, [f"Swimmer-v5 completed proposed stage under aligned robust objective with actor_lr={best_lr}."])


if __name__ == "__main__":
    main()
