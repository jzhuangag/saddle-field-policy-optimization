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
RESULT_ROOT = REPO_ROOT.parent / "results" / "ppo_joint_rarl_subsection3"
OUTPUT_PREFIX = "ppo_joint_rarl_s3_pendulum_coupled_"
ENV_ID = "Pendulum-v1"

DEVICE = torch.device("cpu")
DTYPE = torch.float32
EPS = 1e-8
SEED = 0

BASE_ACTION_DIM = 1
GAME_ACTION_DIM = 2
ALPHA_DYN = 0.05
A_U = 0.001
A_W = 0.001
HIDDEN_SIZES = (64, 64)

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.001
ROLL_OUT_STEPS = 1024
P_TAU_EVAL_INTERVAL = 5
PPO_EVAL_INTERVAL = 10
PPO_EVAL_EPISODES = 3
PPM_INNER_STEPS = 5

LOCAL_RADIUS = 0.1
GAP_INNER_STEPS = 3
ROTATION_RANDOM_VECS = 4

BETA_ROT_GRID = [0.03, 0.1, 0.3, 1.0]
ETA_COUP_GRID = [0.1, 0.3, 1.0, 3.0]
SHARED_LRS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
EXTRA_LR = 3e-3
BASELINE_ITERS = 100
EXTENDED_ITERS = 200
PROPOSED_RADIUS_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
PROPOSED_PREFLIGHT_ITERS = 20
SAME_START_ITERS = [0, 5, 10, 25, 50]
STRICT_SCREEN_ITERS = 60
STRICT_FINAL_ITERS = 100
STRICT_SEARCH_LR = 3e-5

H2 = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=DTYPE, device=DEVICE)
S2 = torch.tensor([[1.0, 0.2], [0.2, -0.5]], dtype=DTYPE, device=DEVICE)


@dataclass(frozen=True)
class EnvReport:
    backend: str
    obs_dim: int
    action_low: float
    action_high: float
    action_shape: tuple[int, ...]


@dataclass(frozen=True)
class LyapunovWeights:
    lambda_F: float
    lambda_P: float
    lambda_C: float


@dataclass(frozen=True)
class CoupledConfig:
    reward_scale: float
    beta_rot: float
    beta_sym: float
    eta_coup: float
    alpha_dyn: float
    a_u: float
    a_w: float
    game_action_dim: int
    rollout_steps: int
    weights: LyapunovWeights
    init_log_std: float

    def to_row(self) -> dict[str, float]:
        return {
            "reward_scale": float(self.reward_scale),
            "beta_rot": float(self.beta_rot),
            "beta_sym": float(self.beta_sym),
            "eta_coup": float(self.eta_coup),
            "alpha_dyn": float(self.alpha_dyn),
            "a_u": float(self.a_u),
            "a_w": float(self.a_w),
            "game_action_dim": float(self.game_action_dim),
            "rollout_steps": float(self.rollout_steps),
            "lambda_F": float(self.weights.lambda_F),
            "lambda_P": float(self.weights.lambda_P),
            "lambda_C": float(self.weights.lambda_C),
            "init_log_std": float(self.init_log_std),
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


class CoupledJointPPOPendulum:
    def __init__(self) -> None:
        seed_everything(SEED)
        self.env_report = self.check_env()
        self.obs_dim = self.env_report.obs_dim
        self.actor_layout = FlatMLP(self.obs_dim, HIDDEN_SIZES, GAME_ACTION_DIM)
        self.critic_layout = FlatMLP(self.obs_dim, HIDDEN_SIZES, 1)
        self.slices = self.build_slices()
        self.total_dim = self.slices["adversary_critic"].stop
        self.reward_scale = self.estimate_reward_scale()
        self.metric_refs: dict[str, float] = {}

    def check_env(self) -> EnvReport:
        env = gym.make(ENV_ID)
        try:
            act = env.action_space
            obs = env.observation_space
            if act.__class__.__name__ != "Box":
                raise RuntimeError(f"{ENV_ID} action space is not Box: {act}")
            if tuple(act.shape) != (BASE_ACTION_DIM,):
                raise RuntimeError(f"{ENV_ID} action shape {act.shape} != ({BASE_ACTION_DIM},)")
            report = EnvReport(
                backend=GYM_BACKEND,
                obs_dim=int(obs.shape[0]),
                action_low=float(act.low[0]),
                action_high=float(act.high[0]),
                action_shape=tuple(int(x) for x in act.shape),
            )
        finally:
            env.close()
        lines = [
            f"# {OUTPUT_PREFIX}env_report",
            "",
            f"- backend: `{report.backend}`",
            f"- env_id: `{ENV_ID}`",
            f"- obs_dim: `{report.obs_dim}`",
            f"- action_space: `Box({report.action_low}, {report.action_high}, shape={report.action_shape})`",
            "- action_space_box_check: `pass`",
        ]
        write_text(RESULT_ROOT / f"{OUTPUT_PREFIX}env_report.md", "\n".join(lines) + "\n")
        return report

    def build_slices(self) -> dict[str, slice]:
        offset = 0
        actor_dim = self.actor_layout.num_params
        critic_dim = self.critic_layout.num_params
        std_dim = GAME_ACTION_DIM
        out = {}
        out["protagonist_actor"] = slice(offset, offset + actor_dim)
        offset += actor_dim
        out["protagonist_log_std"] = slice(offset, offset + std_dim)
        offset += std_dim
        out["protagonist_critic"] = slice(offset, offset + critic_dim)
        offset += critic_dim
        out["adversary_actor"] = slice(offset, offset + actor_dim)
        offset += actor_dim
        out["adversary_log_std"] = slice(offset, offset + std_dim)
        offset += std_dim
        out["adversary_critic"] = slice(offset, offset + critic_dim)
        return out

    def init_z(self, seed: int, init_log_std: float) -> torch.Tensor:
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(seed)
        pa = self.actor_layout.init_flat(gen, final_scale=0.05)
        pl = torch.full((GAME_ACTION_DIM,), init_log_std, dtype=DTYPE, device=DEVICE)
        pc = self.critic_layout.init_flat(gen, final_scale=0.1)
        aa = self.actor_layout.init_flat(gen, final_scale=0.05)
        al = torch.full((GAME_ACTION_DIM,), init_log_std, dtype=DTYPE, device=DEVICE)
        ac = self.critic_layout.init_flat(gen, final_scale=0.1)
        return torch.cat([pa, pl, pc, aa, al, ac]).detach().clone()

    def split_z(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: z[slc] for name, slc in self.slices.items()}

    def actor_mean(self, actor_flat: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_layout.forward(actor_flat, obs)

    def actor_dist(self, actor_flat: torch.Tensor, log_std: torch.Tensor, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor_mean(actor_flat, obs)
        std = torch.exp(log_std).unsqueeze(0).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def critic_value(self, critic_flat: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return self.critic_layout.forward(critic_flat, obs).squeeze(-1)

    def estimate_reward_scale(self) -> float:
        env = gym.make(ENV_ID)
        rng = np.random.default_rng(SEED)
        obs, _ = env.reset(seed=SEED)
        rewards = []
        for _ in range(2048):
            action = np.array([rng.uniform(self.env_report.action_low, self.env_report.action_high)], dtype=np.float32)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()
        std_raw = float(np.std(rewards))
        return 1.0 / max(std_raw, EPS)

    def raw_game_reward(self, reward_raw: torch.Tensor, u: torch.Tensor, w: torch.Tensor, cfg: CoupledConfig) -> torch.Tensor:
        r_task_scaled = cfg.reward_scale * reward_raw
        Hw = torch.matmul(w, H2.T)
        Sw = torch.matmul(w, S2.T)
        r_rot = cfg.beta_rot * torch.sum(u * Hw, dim=-1)
        r_sym = cfg.beta_sym * torch.sum(u * Sw, dim=-1)
        r_reg = (-0.5 * cfg.a_u * torch.sum(u * u, dim=-1)) + (0.5 * cfg.a_w * torch.sum(w * w, dim=-1))
        return r_task_scaled + r_rot + r_sym + r_reg

    def collect_rollout(self, z: torch.Tensor, cfg: CoupledConfig, rollout_seed: int) -> dict[str, torch.Tensor]:
        parts = self.split_z(z)
        env = gym.make(ENV_ID)
        obs, _ = env.reset(seed=rollout_seed)
        obs_t = torch.as_tensor(obs, dtype=DTYPE, device=DEVICE)
        storage: dict[str, list[torch.Tensor]] = {key: [] for key in [
            "obs", "u_old", "w_old", "eps_P", "eps_A", "old_logprob_P", "old_logprob_A", "r_game", "r_adv",
            "done", "value_P_old", "value_A_old", "orig_task_reward", "a_env_raw", "a_env",
        ]}
        train_game_returns: list[float] = []
        train_task_returns: list[float] = []
        ep_game = 0.0
        ep_task = 0.0
        clip_hits = 0
        for _ in range(cfg.rollout_steps):
            obs_batch = obs_t.unsqueeze(0)
            dist_p = self.actor_dist(parts["protagonist_actor"], parts["protagonist_log_std"], obs_batch)
            dist_a = self.actor_dist(parts["adversary_actor"], parts["adversary_log_std"], obs_batch)
            eps_p = torch.randn((GAME_ACTION_DIM,), dtype=DTYPE, device=DEVICE)
            eps_a = torch.randn((GAME_ACTION_DIM,), dtype=DTYPE, device=DEVICE)
            u = dist_p.mean.squeeze(0) + torch.exp(parts["protagonist_log_std"]) * eps_p
            w = dist_a.mean.squeeze(0) + torch.exp(parts["adversary_log_std"]) * eps_a
            logprob_p = dist_p.log_prob(u.unsqueeze(0)).sum(dim=-1).squeeze(0)
            logprob_a = dist_a.log_prob(w.unsqueeze(0)).sum(dim=-1).squeeze(0)
            value_p = self.critic_value(parts["protagonist_critic"], obs_batch).squeeze(0)
            value_a = self.critic_value(parts["adversary_critic"], obs_batch).squeeze(0)

            Hw = H2 @ w
            a_env_raw = u[0] + (cfg.alpha_dyn * Hw[0])
            a_env = torch.clamp(a_env_raw, self.env_report.action_low, self.env_report.action_high)
            obs_next, reward_raw, terminated, truncated, _ = env.step(np.array([float(a_env.item())], dtype=np.float32))
            done = terminated or truncated
            reward_t = torch.as_tensor([reward_raw], dtype=DTYPE, device=DEVICE)
            r_game = self.raw_game_reward(reward_t, u.unsqueeze(0), w.unsqueeze(0), cfg).squeeze(0)

            storage["obs"].append(obs_t)
            storage["u_old"].append(u.detach())
            storage["w_old"].append(w.detach())
            storage["eps_P"].append(eps_p.detach())
            storage["eps_A"].append(eps_a.detach())
            storage["old_logprob_P"].append(logprob_p.detach())
            storage["old_logprob_A"].append(logprob_a.detach())
            storage["r_game"].append(r_game.detach())
            storage["r_adv"].append((-r_game).detach())
            storage["done"].append(torch.as_tensor(float(done), dtype=DTYPE, device=DEVICE))
            storage["value_P_old"].append(value_p.detach())
            storage["value_A_old"].append(value_a.detach())
            storage["orig_task_reward"].append(torch.as_tensor(reward_raw, dtype=DTYPE, device=DEVICE))
            storage["a_env_raw"].append(a_env_raw.detach())
            storage["a_env"].append(a_env.detach())

            ep_game += float(r_game.item())
            ep_task += float(reward_raw)
            clip_hits += int(abs(float(a_env_raw.item()) - float(a_env.item())) > 1e-12)
            if done:
                train_game_returns.append(ep_game)
                train_task_returns.append(ep_task)
                ep_game = 0.0
                ep_task = 0.0
                obs, _ = env.reset()
            else:
                obs = obs_next
            obs_t = torch.as_tensor(obs, dtype=DTYPE, device=DEVICE)

        last_obs = obs_t.unsqueeze(0)
        bootstrap_p = self.critic_value(parts["protagonist_critic"], last_obs).detach().squeeze(0)
        bootstrap_a = self.critic_value(parts["adversary_critic"], last_obs).detach().squeeze(0)
        env.close()

        batch = {key: torch.stack(vals) for key, vals in storage.items()}
        returns_p, adv_p = compute_gae(batch["r_game"], batch["value_P_old"], batch["done"], bootstrap_p)
        returns_a, adv_a = compute_gae(batch["r_adv"], batch["value_A_old"], batch["done"], bootstrap_a)
        batch["return_P"] = returns_p.detach()
        batch["return_A"] = returns_a.detach()
        batch["adv_P"] = normalize_tensor(adv_p).detach()
        batch["adv_A"] = normalize_tensor(adv_a).detach()
        batch["train_game_return_mean"] = torch.as_tensor(np.mean(train_game_returns) if train_game_returns else ep_game, dtype=DTYPE, device=DEVICE)
        batch["train_task_return_mean"] = torch.as_tensor(np.mean(train_task_returns) if train_task_returns else ep_task, dtype=DTYPE, device=DEVICE)
        batch["action_clip_fraction"] = torch.as_tensor(clip_hits / max(cfg.rollout_steps, 1), dtype=DTYPE, device=DEVICE)
        batch["mean_abs_u"] = batch["u_old"].abs().mean()
        batch["mean_abs_w"] = batch["w_old"].abs().mean()
        batch["max_abs_u"] = batch["u_old"].abs().max()
        batch["max_abs_w"] = batch["w_old"].abs().max()
        batch["mean_abs_a_env_raw"] = batch["a_env_raw"].abs().mean()
        batch["mean_abs_a_env_clipped"] = batch["a_env"].abs().mean()
        return batch

    def current_actor_actions(self, z: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        parts = self.split_z(z)
        mu_p = self.actor_mean(parts["protagonist_actor"], batch["obs"])
        mu_a = self.actor_mean(parts["adversary_actor"], batch["obs"])
        std_p = torch.exp(parts["protagonist_log_std"]).unsqueeze(0).expand_as(mu_p)
        std_a = torch.exp(parts["adversary_log_std"]).unsqueeze(0).expand_as(mu_a)
        u_current = mu_p + std_p * batch["eps_P"]
        w_current = mu_a + std_a * batch["eps_A"]
        return u_current, w_current, mu_p, mu_a

    def j_coup(self, z: torch.Tensor, batch: dict[str, torch.Tensor], cfg: CoupledConfig) -> torch.Tensor:
        u_current, w_current, _, _ = self.current_actor_actions(z, batch)
        Hw = torch.matmul(w_current, H2.T)
        Sw = torch.matmul(w_current, S2.T)
        term = (
            cfg.beta_rot * torch.sum(u_current * Hw, dim=-1)
            + cfg.beta_sym * torch.sum(u_current * Sw, dim=-1)
            - 0.5 * cfg.a_u * torch.sum(u_current * u_current, dim=-1)
            + 0.5 * cfg.a_w * torch.sum(w_current * w_current, dim=-1)
        )
        return term.mean()

    def loss_components(self, z: torch.Tensor, batch: dict[str, torch.Tensor], cfg: CoupledConfig) -> dict[str, torch.Tensor]:
        parts = self.split_z(z)
        dist_p = self.actor_dist(parts["protagonist_actor"], parts["protagonist_log_std"], batch["obs"])
        dist_a = self.actor_dist(parts["adversary_actor"], parts["adversary_log_std"], batch["obs"])
        logprob_p_new = dist_p.log_prob(batch["u_old"]).sum(dim=-1)
        logprob_a_new = dist_a.log_prob(batch["w_old"]).sum(dim=-1)
        ratio_p = torch.exp(logprob_p_new - batch["old_logprob_P"])
        ratio_a = torch.exp(logprob_a_new - batch["old_logprob_A"])
        clip_p = torch.clamp(ratio_p, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
        clip_a = torch.clamp(ratio_a, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
        l_ppo_p = -torch.mean(torch.minimum(ratio_p * batch["adv_P"], clip_p * batch["adv_P"]))
        l_ppo_a = -torch.mean(torch.minimum(ratio_a * batch["adv_A"], clip_a * batch["adv_A"]))
        entropy_p = dist_p.entropy().sum(dim=-1).mean()
        entropy_a = dist_a.entropy().sum(dim=-1).mean()
        value_p = self.critic_value(parts["protagonist_critic"], batch["obs"])
        value_a = self.critic_value(parts["adversary_critic"], batch["obs"])
        l_v_p = torch.mean((value_p - batch["return_P"]) ** 2)
        l_v_a = torch.mean((value_a - batch["return_A"]) ** 2)
        j_c = self.j_coup(z, batch, cfg)
        l_p_actor_total = l_ppo_p - (ENT_COEF * entropy_p) - (cfg.eta_coup * j_c)
        l_a_actor_total = l_ppo_a - (ENT_COEF * entropy_a) + (cfg.eta_coup * j_c)
        l_p_critic_total = VF_COEF * l_v_p
        l_a_critic_total = VF_COEF * l_v_a
        return {
            "L_PPO_actor_P": l_ppo_p,
            "L_PPO_actor_A": l_ppo_a,
            "entropy_P": entropy_p,
            "entropy_A": entropy_a,
            "L_v_P": l_v_p,
            "L_v_A": l_v_a,
            "L_P_actor_total": l_p_actor_total,
            "L_A_actor_total": l_a_actor_total,
            "L_P_critic_total": l_p_critic_total,
            "L_A_critic_total": l_a_critic_total,
            "J_coup": j_c,
            "ratio_P": ratio_p,
            "ratio_A": ratio_a,
        }

    def field(self, z: torch.Tensor, batch: dict[str, torch.Tensor], cfg: CoupledConfig) -> torch.Tensor:
        z_req = z if z.requires_grad else z.detach().clone().requires_grad_(True)
        comps = self.loss_components(z_req, batch, cfg)
        out = torch.zeros_like(z_req)
        actor_p = slice(self.slices["protagonist_actor"].start, self.slices["protagonist_log_std"].stop)
        actor_a = slice(self.slices["adversary_actor"].start, self.slices["adversary_log_std"].stop)
        critic_p = self.slices["protagonist_critic"]
        critic_a = self.slices["adversary_critic"]

        grad_p_actor = torch.autograd.grad(comps["L_P_actor_total"], z_req, retain_graph=True, create_graph=True)[0][actor_p]
        grad_a_actor = torch.autograd.grad(comps["L_A_actor_total"], z_req, retain_graph=True, create_graph=True)[0][actor_a]
        grad_p_critic = torch.autograd.grad(comps["L_P_critic_total"], z_req, retain_graph=True, create_graph=True)[0][critic_p]
        grad_a_critic = torch.autograd.grad(comps["L_A_critic_total"], z_req, retain_graph=True, create_graph=True)[0][critic_a]

        out[actor_p] = grad_p_actor
        out[actor_a] = grad_a_actor
        out[critic_p] = grad_p_critic
        out[critic_a] = grad_a_critic
        return out

    def actor_block_loss(self, z: torch.Tensor, batch: dict[str, torch.Tensor], cfg: CoupledConfig, protagonist: bool) -> torch.Tensor:
        comps = self.loss_components(z, batch, cfg)
        return comps["L_P_actor_total"] if protagonist else comps["L_A_actor_total"]

    def local_gap(self, z: torch.Tensor, batch: dict[str, torch.Tensor], cfg: CoupledConfig, lr: float, protagonist: bool, with_prox: bool) -> torch.Tensor:
        actor_slice = slice(
            self.slices["protagonist_actor"].start if protagonist else self.slices["adversary_actor"].start,
            self.slices["protagonist_log_std"].stop if protagonist else self.slices["adversary_log_std"].stop,
        )
        base = z.detach().clone()
        current = base.clone()
        initial = base[actor_slice].clone()
        current_loss = self.actor_block_loss(current, batch, cfg, protagonist)
        local_lr = 0.1 * lr
        for _ in range(GAP_INNER_STEPS):
            cur_req = current.detach().clone().requires_grad_(True)
            loss = self.actor_block_loss(cur_req, batch, cfg, protagonist)
            grad = torch.autograd.grad(loss, cur_req)[0][actor_slice]
            if with_prox:
                grad = grad + (cur_req[actor_slice] - initial) / max(LOCAL_RADIUS, 1e-6)
            next_actor = cur_req[actor_slice] - (local_lr * grad)
            delta = next_actor - initial
            delta_norm = torch.linalg.norm(delta)
            if delta_norm > LOCAL_RADIUS:
                next_actor = initial + delta * (LOCAL_RADIUS / (delta_norm + EPS))
            current = cur_req.detach().clone()
            current[actor_slice] = next_actor.detach()
        improved = self.actor_block_loss(current, batch, cfg, protagonist)
        return torch.relu(current_loss - improved)

    def approximate_metrics(
        self,
        z: torch.Tensor,
        batch: dict[str, torch.Tensor],
        cfg: CoupledConfig,
        lr: float,
        compute_ptau: bool,
        last_ptau: float | None,
        compute_geometry: bool,
    ) -> dict[str, float]:
        comps = self.loss_components(z, batch, cfg)
        field = self.field(z, batch, cfg)
        field_energy = 0.5 * float(torch.dot(field.detach(), field.detach()).item())
        if "field0" not in self.metric_refs:
            self.metric_refs["field0"] = field_energy
            self.metric_refs["critic0"] = float((comps["L_v_P"] + comps["L_v_A"]).detach().item())
        if compute_ptau or ("ptau0" not in self.metric_refs):
            p_tau = float((self.local_gap(z, batch, cfg, lr, True, True) + self.local_gap(z, batch, cfg, lr, False, True)).detach().item())
        else:
            p_tau = float(last_ptau if last_ptau is not None else 0.0)
        if "ptau0" not in self.metric_refs:
            self.metric_refs["ptau0"] = max(p_tau, EPS)
        critic_total = float((comps["L_v_P"] + comps["L_v_A"]).detach().item())
        field_term = field_energy / (self.metric_refs["field0"] + EPS)
        p_tau_term = p_tau / (self.metric_refs["ptau0"] + EPS)
        critic_term = critic_total / (self.metric_refs["critic0"] + EPS)
        v_lambda = (cfg.weights.lambda_F * field_term) + (cfg.weights.lambda_P * p_tau_term) + (cfg.weights.lambda_C * critic_term)
        exploitability = float((self.local_gap(z, batch, cfg, lr, True, False) + self.local_gap(z, batch, cfg, lr, False, False)).detach().item())
        geom = {
            "g_over_f": math.nan,
            "cos_fg": math.nan,
            "non_collinearity": math.nan,
            "rotation_ratio_proxy": math.nan,
            "cross_player_coupling_proxy": math.nan,
            "same_player_coupling_proxy": math.nan,
            "cross_to_same_ratio": math.nan,
        }
        if compute_geometry:
            z_req = z.detach().clone().requires_grad_(True)
            field_z = self.field(z_req, batch, cfg)
            _, g_vec = torch.autograd.functional.jvp(lambda zz: self.field(zz, batch, cfg), (z_req,), (field_z.detach(),), create_graph=False, strict=False)
            field_det = field_z.detach()
            g_det = g_vec.detach()
            geom["g_over_f"] = float(torch.linalg.norm(g_det).item() / (torch.linalg.norm(field_det).item() + EPS))
            cos_val = float(torch.dot(field_det, g_det).item() / ((torch.linalg.norm(field_det).item() * torch.linalg.norm(g_det).item()) + EPS))
            geom["cos_fg"] = cos_val
            geom["non_collinearity"] = math.sqrt(max(0.0, 1.0 - min(1.0, cos_val * cos_val)))
            aproxy = 0.0
            sproxy = 0.0
            gen = torch.Generator(device=DEVICE)
            gen.manual_seed(20260613)
            for _ in range(ROTATION_RANDOM_VECS):
                v = torch.randn(self.total_dim, generator=gen, dtype=DTYPE, device=DEVICE)
                v = v / (torch.linalg.norm(v) + EPS)
                _, jv = torch.autograd.functional.jvp(lambda zz: self.field(zz, batch, cfg), (z_req,), (v,), create_graph=False, strict=False)
                jtv = torch.autograd.grad(torch.dot(field_z, v), z_req, retain_graph=True)[0]
                aproxy += float(torch.linalg.norm(jv.detach() - jtv.detach()).item())
                sproxy += float(torch.linalg.norm(jv.detach() + jtv.detach()).item())
            geom["rotation_ratio_proxy"] = aproxy / (sproxy + EPS)
            coupling = actor_cross_coupling(self, z, batch, cfg)
            geom.update(coupling)
        return {
            "V_lambda": v_lambda,
            "normalized_P_tau": p_tau_term,
            "P_tau": p_tau,
            "field_norm": float(torch.linalg.norm(field.detach()).item()),
            "field_term": field_term,
            "critic_term": critic_term,
            "critic_loss_total": critic_total,
            "critic_loss_P": float(comps["L_v_P"].detach().item()),
            "critic_loss_A": float(comps["L_v_A"].detach().item()),
            "ppo_actor_loss_P": float(comps["L_PPO_actor_P"].detach().item()),
            "ppo_actor_loss_A": float(comps["L_PPO_actor_A"].detach().item()),
            "J_coup": float(comps["J_coup"].detach().item()),
            "mean_KL_P": float((batch["old_logprob_P"] - self.actor_dist(self.split_z(z)["protagonist_actor"], self.split_z(z)["protagonist_log_std"], batch["obs"]).log_prob(batch["u_old"]).sum(dim=-1).detach()).mean().item()),
            "mean_KL_A": float((batch["old_logprob_A"] - self.actor_dist(self.split_z(z)["adversary_actor"], self.split_z(z)["adversary_log_std"], batch["obs"]).log_prob(batch["w_old"]).sum(dim=-1).detach()).mean().item()),
            "ratio_clip_fraction_P": float(((comps["ratio_P"].detach() < (1.0 - CLIP_EPS)) | (comps["ratio_P"].detach() > (1.0 + CLIP_EPS))).float().mean().item()),
            "ratio_clip_fraction_A": float(((comps["ratio_A"].detach() < (1.0 - CLIP_EPS)) | (comps["ratio_A"].detach() > (1.0 + CLIP_EPS))).float().mean().item()),
            "log_std_mean_P": float(self.split_z(z)["protagonist_log_std"].mean().item()),
            "log_std_min_P": float(self.split_z(z)["protagonist_log_std"].min().item()),
            "log_std_max_P": float(self.split_z(z)["protagonist_log_std"].max().item()),
            "log_std_mean_A": float(self.split_z(z)["adversary_log_std"].mean().item()),
            "log_std_min_A": float(self.split_z(z)["adversary_log_std"].min().item()),
            "log_std_max_A": float(self.split_z(z)["adversary_log_std"].max().item()),
            "action_clip_fraction": float(batch["action_clip_fraction"].item()),
            "mean_abs_u": float(batch["mean_abs_u"].item()),
            "mean_abs_w": float(batch["mean_abs_w"].item()),
            "max_abs_u": float(batch["max_abs_u"].item()),
            "max_abs_w": float(batch["max_abs_w"].item()),
            "mean_abs_a_env_raw": float(batch["mean_abs_a_env_raw"].item()),
            "mean_abs_a_env_clipped": float(batch["mean_abs_a_env_clipped"].item()),
            "train_game_return": float(batch["train_game_return_mean"].item()),
            "train_original_task_return": float(batch["train_task_return_mean"].item()),
            "eval_game_return": math.nan,
            "eval_original_task_return": math.nan,
            "exploitability_proxy": exploitability,
            **geom,
        }

    def evaluate_policy(self, z: torch.Tensor, cfg: CoupledConfig, episodes: int) -> tuple[float, float]:
        env = gym.make(ENV_ID)
        parts = self.split_z(z)
        game_returns = []
        task_returns = []
        for ep in range(episodes):
            obs, _ = env.reset(seed=SEED + 7000 + ep)
            done = False
            game_total = 0.0
            task_total = 0.0
            while not done:
                obs_t = torch.as_tensor(obs, dtype=DTYPE, device=DEVICE).unsqueeze(0)
                u = self.actor_mean(parts["protagonist_actor"], obs_t).squeeze(0)
                w = self.actor_mean(parts["adversary_actor"], obs_t).squeeze(0)
                Hw = H2 @ w
                a_env_raw = u[0] + (cfg.alpha_dyn * Hw[0])
                a_env = torch.clamp(a_env_raw, self.env_report.action_low, self.env_report.action_high)
                obs, reward_raw, terminated, truncated, _ = env.step(np.array([float(a_env.item())], dtype=np.float32))
                done = terminated or truncated
                reward_t = torch.as_tensor([reward_raw], dtype=DTYPE, device=DEVICE)
                game_total += float(self.raw_game_reward(reward_t, u.unsqueeze(0), w.unsqueeze(0), cfg).item())
                task_total += float(reward_raw)
            game_returns.append(game_total)
            task_returns.append(task_total)
        env.close()
        return float(np.mean(game_returns)), float(np.mean(task_returns))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def finite(value: float) -> bool:
    return math.isfinite(float(value))


def run_valid_from_metrics(metrics: dict[str, float]) -> bool:
    return (
        finite(metrics["V_lambda"])
        and finite(metrics["field_norm"])
        and finite(metrics["P_tau"])
        and finite(metrics["critic_loss_total"])
        and finite(metrics["train_game_return"])
        and finite(metrics["eval_game_return"])
        and finite(metrics["gradient_norm"])
        and metrics["action_clip_fraction"] <= 0.05
        and metrics["mean_KL_P"] <= 0.05
        and metrics["mean_KL_A"] <= 0.05
        and metrics["ratio_clip_fraction_P"] <= 0.5
        and metrics["ratio_clip_fraction_A"] <= 0.5
        and -5.0 <= metrics["log_std_mean_P"] <= 2.0
        and -5.0 <= metrics["log_std_mean_A"] <= 2.0
        and finite(metrics["train_original_task_return"])
        and finite(metrics["eval_original_task_return"])
    )


def actor_cross_coupling(game: CoupledJointPPOPendulum, z: torch.Tensor, batch: dict[str, torch.Tensor], cfg: CoupledConfig) -> dict[str, float]:
    actor_p = slice(game.slices["protagonist_actor"].start, game.slices["protagonist_log_std"].stop)
    actor_a = slice(game.slices["adversary_actor"].start, game.slices["adversary_log_std"].stop)
    eps_scale = 1e-3

    def protagonist_actor_field(cur_z: torch.Tensor) -> torch.Tensor:
        return game.field(cur_z, batch, cfg)[actor_p].detach()

    def adversary_actor_field(cur_z: torch.Tensor) -> torch.Tensor:
        return game.field(cur_z, batch, cfg)[actor_a].detach()

    base_p = protagonist_actor_field(z)
    base_a = adversary_actor_field(z)
    pert_a = z.detach().clone()
    pert_a[actor_a] = pert_a[actor_a] + eps_scale
    pert_p = z.detach().clone()
    pert_p[actor_p] = pert_p[actor_p] + eps_scale
    cross_p = float(torch.linalg.norm(protagonist_actor_field(pert_a) - base_p).item())
    cross_a = float(torch.linalg.norm(adversary_actor_field(pert_p) - base_a).item())
    same_p = float(torch.linalg.norm(protagonist_actor_field(pert_p) - base_p).item())
    same_a = float(torch.linalg.norm(adversary_actor_field(pert_a) - base_a).item())
    cross_proxy = 0.5 * (cross_p + cross_a)
    same_proxy = 0.5 * (same_p + same_a)
    return {
        "cross_player_coupling_proxy": cross_proxy,
        "same_player_coupling_proxy": same_proxy,
        "cross_to_same_ratio": cross_proxy / (same_proxy + EPS),
    }


def gradient_scale_summary(game: CoupledJointPPOPendulum, z: torch.Tensor, batch: dict[str, torch.Tensor], cfg: CoupledConfig) -> dict[str, float]:
    z_req = z.detach().clone().requires_grad_(True)
    comps = game.loss_components(z_req, batch, cfg)
    actor_p = slice(game.slices["protagonist_actor"].start, game.slices["protagonist_log_std"].stop)
    actor_a = slice(game.slices["adversary_actor"].start, game.slices["adversary_log_std"].stop)
    critic_p = game.slices["protagonist_critic"]
    critic_a = game.slices["adversary_critic"]

    ppo_p_grad = torch.autograd.grad(comps["L_PPO_actor_P"], z_req, retain_graph=True)[0][actor_p]
    ppo_a_grad = torch.autograd.grad(comps["L_PPO_actor_A"], z_req, retain_graph=True)[0][actor_a]
    coup_p_grad = torch.autograd.grad(-cfg.eta_coup * comps["J_coup"], z_req, retain_graph=True)[0][actor_p]
    coup_a_grad = torch.autograd.grad(cfg.eta_coup * comps["J_coup"], z_req, retain_graph=True)[0][actor_a]
    critic_p_grad = torch.autograd.grad(comps["L_P_critic_total"], z_req, retain_graph=True)[0][critic_p]
    critic_a_grad = torch.autograd.grad(comps["L_A_critic_total"], z_req)[0][critic_a]

    ppo_actor_grad_norm = float(torch.sqrt(torch.sum(ppo_p_grad * ppo_p_grad) + torch.sum(ppo_a_grad * ppo_a_grad)).item())
    coupling_grad_norm = float(torch.sqrt(torch.sum(coup_p_grad * coup_p_grad) + torch.sum(coup_a_grad * coup_a_grad)).item())
    critic_grad_norm = float(torch.sqrt(torch.sum(critic_p_grad * critic_p_grad) + torch.sum(critic_a_grad * critic_a_grad)).item())
    return {
        "ppo_actor_gradient_norm": ppo_actor_grad_norm,
        "coupling_gradient_norm": coupling_grad_norm,
        "critic_gradient_norm": critic_grad_norm,
    }


def run_sgd(game: CoupledJointPPOPendulum, z: torch.Tensor, batch: dict[str, torch.Tensor], cfg: CoupledConfig, lr: float) -> tuple[torch.Tensor, dict[str, float]]:
    field = game.field(z, batch, cfg).detach()
    next_z = z - (lr * field)
    return next_z.detach(), {"update_norm": float(torch.linalg.norm(lr * field).item())}


def run_egm(game: CoupledJointPPOPendulum, z: torch.Tensor, batch: dict[str, torch.Tensor], cfg: CoupledConfig, lr: float) -> tuple[torch.Tensor, dict[str, float]]:
    field0 = game.field(z, batch, cfg).detach()
    z_half = z - (lr * field0)
    field_half = game.field(z_half, batch, cfg).detach()
    next_z = z - (lr * field_half)
    return next_z.detach(), {"update_norm": float(torch.linalg.norm(lr * field_half).item())}


def run_ppm(game: CoupledJointPPOPendulum, z: torch.Tensor, batch: dict[str, torch.Tensor], cfg: CoupledConfig, lr: float) -> tuple[torch.Tensor, dict[str, float]]:
    current = z.detach().clone()
    for _ in range(PPM_INNER_STEPS):
        field = game.field(current, batch, cfg).detach()
        current = z - (lr * field)
    return current.detach(), {"update_norm": float(torch.linalg.norm(current - z).item())}


def choose_coupled_config(game: CoupledJointPPOPendulum) -> tuple[CoupledConfig, list[dict[str, Any]], str]:
    base_weights = LyapunovWeights(lambda_F=0.01, lambda_P=1.0, lambda_C=0.1)
    base_z = game.init_z(SEED, init_log_std=-0.5)
    rows: list[dict[str, Any]] = []
    best_cfg = None
    best_key = None
    for beta_rot in BETA_ROT_GRID:
        for eta_coup in ETA_COUP_GRID:
            cfg = CoupledConfig(
                reward_scale=game.reward_scale,
                beta_rot=beta_rot,
                beta_sym=0.10 * beta_rot,
                eta_coup=eta_coup,
                alpha_dyn=ALPHA_DYN,
                a_u=A_U,
                a_w=A_W,
                game_action_dim=GAME_ACTION_DIM,
                rollout_steps=ROLL_OUT_STEPS,
                weights=base_weights,
                init_log_std=-0.5,
            )
            batch = game.collect_rollout(base_z, cfg, rollout_seed=SEED + 123)
            geom = game.approximate_metrics(base_z, batch, cfg, lr=1e-4, compute_ptau=True, last_ptau=None, compute_geometry=True)
            grad_scales = gradient_scale_summary(game, base_z, batch, cfg)
            row = {
                **cfg.to_row(),
                "cross_player_coupling_proxy": geom["cross_player_coupling_proxy"],
                "same_player_coupling_proxy": geom["same_player_coupling_proxy"],
                "cross_to_same_ratio": geom["cross_to_same_ratio"],
                "rotation_ratio_proxy": geom["rotation_ratio_proxy"],
                "g_over_f": geom["g_over_f"],
                "cos_fg": geom["cos_fg"],
                "non_collinearity": geom["non_collinearity"],
                "action_clip_fraction": geom["action_clip_fraction"],
                **grad_scales,
            }
            rows.append(row)
            passes = (
                geom["cross_player_coupling_proxy"] > 0.0
                and geom["cross_to_same_ratio"] >= 0.05
                and geom["rotation_ratio_proxy"] > 0.5
                and geom["non_collinearity"] > 0.2
                and grad_scales["coupling_gradient_norm"] <= (10.0 * max(grad_scales["ppo_actor_gradient_norm"], EPS))
                and geom["action_clip_fraction"] <= 0.05
                and all(finite(row[key]) for key in row)
            )
            penalty = 0.0
            penalty += max(0.0, 0.05 - geom["cross_to_same_ratio"]) / 0.05
            penalty += max(0.0, 0.5 - geom["rotation_ratio_proxy"]) / 0.5
            penalty += max(0.0, 0.2 - geom["non_collinearity"]) / 0.2
            penalty += max(0.0, geom["action_clip_fraction"] - 0.05) / 0.05
            penalty += max(0.0, (grad_scales["coupling_gradient_norm"] / max(grad_scales["ppo_actor_gradient_norm"], EPS)) - 10.0) / 10.0
            if geom["cross_player_coupling_proxy"] <= 0.0:
                penalty += 100.0
            key = (
                0 if passes else 1,
                penalty,
                beta_rot,
                eta_coup,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_cfg = cfg
    assert best_cfg is not None
    write_csv(RESULT_ROOT / f"{OUTPUT_PREFIX}preflight.csv", rows)
    lines = [
        f"# {OUTPUT_PREFIX}preflight_report",
        "",
        f"- chosen_reward_scale: `{game.reward_scale:.6e}`",
        f"- chosen_beta_rot: `{best_cfg.beta_rot:.6e}`",
        f"- chosen_beta_sym: `{best_cfg.beta_sym:.6e}`",
        f"- chosen_eta_coup: `{best_cfg.eta_coup:.6e}`",
        "- selection_rule: choose the mildest setting with nonzero cross-player coupling and nontrivial geometry; if none fully pass, choose the closest mild setting.",
    ]
    write_text(RESULT_ROOT / f"{OUTPUT_PREFIX}preflight_report.md", "\n".join(lines) + "\n")
    return best_cfg, rows, "\n".join(lines) + "\n"


def geometry_recheck(game: CoupledJointPPOPendulum, cfg: CoupledConfig) -> tuple[dict[str, Any], str]:
    z0 = game.init_z(SEED, init_log_std=cfg.init_log_std)
    batch = game.collect_rollout(z0, cfg, rollout_seed=SEED + 4242)
    geom = game.approximate_metrics(z0, batch, cfg, lr=1e-4, compute_ptau=True, last_ptau=None, compute_geometry=True)
    row = {
        **cfg.to_row(),
        "field_norm": geom["field_norm"],
        "g_over_f": geom["g_over_f"],
        "cos_fg": geom["cos_fg"],
        "non_collinearity": geom["non_collinearity"],
        "rotation_ratio_proxy": geom["rotation_ratio_proxy"],
        "cross_player_coupling_proxy": geom["cross_player_coupling_proxy"],
        "same_player_coupling_proxy": geom["same_player_coupling_proxy"],
        "cross_to_same_ratio": geom["cross_to_same_ratio"],
    }
    write_csv(RESULT_ROOT / f"{OUTPUT_PREFIX}geometry_audit.csv", [row])
    decision = (
        row["cross_player_coupling_proxy"] > 0.0
        and row["cross_to_same_ratio"] > 0.0
        and row["rotation_ratio_proxy"] > 1e-4
        and row["non_collinearity"] > 0.2
    )
    lines = [
        f"# {OUTPUT_PREFIX}geometry_audit",
        "",
        f"- field_norm: `{row['field_norm']:.6e}`",
        f"- ||G|| / ||F||: `{row['g_over_f']:.6e}`",
        f"- cos(F, G): `{row['cos_fg']:.6e}`",
        f"- non_collinearity: `{row['non_collinearity']:.6e}`",
        f"- rotation_ratio_proxy: `{row['rotation_ratio_proxy']:.6e}`",
        f"- cross_player_coupling_proxy: `{row['cross_player_coupling_proxy']:.6e}`",
        f"- same_player_coupling_proxy: `{row['same_player_coupling_proxy']:.6e}`",
        f"- cross_to_same_ratio: `{row['cross_to_same_ratio']:.6e}`",
        "",
        f"- geometry_ready: `{decision}`",
    ]
    write_text(RESULT_ROOT / f"{OUTPUT_PREFIX}geometry_audit.md", "\n".join(lines) + "\n")
    return row, "\n".join(lines) + "\n"


def run_method(
    game: CoupledJointPPOPendulum,
    cfg: CoupledConfig,
    method: str,
    lr: float,
    outer_iterations: int,
    initial_z: torch.Tensor | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    game.metric_refs = {}
    z = (initial_z.detach().clone() if initial_z is not None else game.init_z(SEED, init_log_std=cfg.init_log_std))
    curves: list[dict[str, Any]] = []
    auc_v = 0.0
    auc_ptau = 0.0
    auc_field = 0.0
    auc_exploit = 0.0
    auc_train = 0.0
    auc_eval = 0.0
    threshold_iter = None
    run_valid = True
    last_ptau = None
    eval_game, eval_task = game.evaluate_policy(z, cfg, PPO_EVAL_EPISODES)
    for iteration in range(outer_iterations + 1):
        batch = game.collect_rollout(z, cfg, rollout_seed=SEED + (1000 * iteration) + 99)
        compute_ptau = (iteration % P_TAU_EVAL_INTERVAL == 0) or last_ptau is None
        metrics = game.approximate_metrics(z, batch, cfg, lr, compute_ptau=compute_ptau, last_ptau=last_ptau, compute_geometry=False)
        if compute_ptau:
            last_ptau = metrics["P_tau"]
        if iteration % PPO_EVAL_INTERVAL == 0 or iteration == outer_iterations:
            eval_game, eval_task = game.evaluate_policy(z, cfg, PPO_EVAL_EPISODES)
        metrics["eval_game_return"] = eval_game
        metrics["eval_original_task_return"] = eval_task
        metrics["iteration"] = iteration
        metrics["method"] = method
        metrics["lr"] = lr
        metrics["gradient_norm"] = metrics["field_norm"]
        metrics["nan_flag"] = int(not all(finite(metrics[key]) for key in ["V_lambda", "field_norm", "P_tau", "critic_loss_total", "train_game_return", "eval_game_return", "gradient_norm"]))
        valid = run_valid_from_metrics(metrics)
        metrics["valid_flag"] = int(valid)
        run_valid = run_valid and bool(valid)
        curves.append(dict(metrics))
        auc_v += metrics["V_lambda"]
        auc_ptau += metrics["normalized_P_tau"]
        auc_field += metrics["field_norm"]
        auc_exploit += metrics["exploitability_proxy"]
        auc_train += metrics["train_game_return"]
        auc_eval += metrics["eval_game_return"]
        if threshold_iter is None and metrics["V_lambda"] <= 0.5:
            threshold_iter = iteration
        if iteration == outer_iterations:
            break
        if method == "sgd":
            z, update_meta = run_sgd(game, z, batch, cfg, lr)
        elif method == "egm":
            z, update_meta = run_egm(game, z, batch, cfg, lr)
        elif method == "ppm":
            z, update_meta = run_ppm(game, z, batch, cfg, lr)
        else:
            raise ValueError(method)
        curves[-1].update(update_meta)
    final = curves[-1]
    summary = {
        **cfg.to_row(),
        "method": method,
        "lr": lr,
        "valid_flag": int(run_valid),
        "auc_V_lambda": auc_v,
        "auc_normalized_P_tau": auc_ptau,
        "auc_field_norm": auc_field,
        "auc_exploitability": auc_exploit,
        "auc_train_game_return": auc_train,
        "auc_eval_game_return": auc_eval,
        "final_V_lambda": final["V_lambda"],
        "final_normalized_P_tau": final["normalized_P_tau"],
        "final_field_norm": final["field_norm"],
        "final_exploitability": final["exploitability_proxy"],
        "final_train_game_return": final["train_game_return"],
        "final_eval_game_return": final["eval_game_return"],
        "final_train_original_task_return": final["train_original_task_return"],
        "final_eval_original_task_return": final["eval_original_task_return"],
        "final_action_clip_fraction": final["action_clip_fraction"],
        "final_mean_KL_P": final["mean_KL_P"],
        "final_mean_KL_A": final["mean_KL_A"],
        "final_ratio_clip_fraction_P": final["ratio_clip_fraction_P"],
        "final_ratio_clip_fraction_A": final["ratio_clip_fraction_A"],
        "final_log_std_mean_P": final["log_std_mean_P"],
        "final_log_std_mean_A": final["log_std_mean_A"],
        "time_to_vhalf": threshold_iter if threshold_iter is not None else outer_iterations + 1,
    }
    return curves, summary


def baseline_gate(game: CoupledJointPPOPendulum, cfg: CoupledConfig) -> tuple[str, float, list[dict[str, Any]], list[dict[str, Any]], str]:
    baseline_rows: list[dict[str, Any]] = []
    baseline_curves: list[dict[str, Any]] = []
    tested_lrs = list(SHARED_LRS)
    for lr in tested_lrs:
        for method in ["sgd", "egm", "ppm"]:
            print(f"[baseline] method={method} lr={lr} start", flush=True)
            curves, summary = run_method(game, cfg, method, lr, BASELINE_ITERS)
            baseline_rows.append(summary)
            baseline_curves.extend(curves)
            write_csv(RESULT_ROOT / f"{OUTPUT_PREFIX}baseline_gate.csv", baseline_rows)
            print(
                f"[baseline] method={method} lr={lr} done valid={summary['valid_flag']} auc_V={summary['auc_V_lambda']:.6e} auc_P={summary['auc_normalized_P_tau']:.6e}",
                flush=True,
            )

    def trio_rows(lr_value: float) -> list[dict[str, Any]]:
        return [row for row in baseline_rows if row["lr"] == lr_value]

    def trio_valid(lr_value: float) -> bool:
        group = trio_rows(lr_value)
        return len(group) == 3 and all(row["valid_flag"] == 1 for row in group)

    def gate_passes_on_lr(lr_value: float) -> bool:
        group = trio_rows(lr_value)
        if len(group) != 3 or not all(row["valid_flag"] == 1 for row in group):
            return False
        sgd = next(row for row in group if row["method"] == "sgd")
        egm = next(row for row in group if row["method"] == "egm")
        ppm = next(row for row in group if row["method"] == "ppm")
        v_gain = max(sgd["auc_V_lambda"] / max(egm["auc_V_lambda"], EPS), sgd["auc_V_lambda"] / max(ppm["auc_V_lambda"], EPS))
        p_gain = max(sgd["auc_normalized_P_tau"] / max(egm["auc_normalized_P_tau"], EPS), sgd["auc_normalized_P_tau"] / max(ppm["auc_normalized_P_tau"], EPS))
        t_gain = max((sgd["time_to_vhalf"] + EPS) / max(egm["time_to_vhalf"], EPS), (sgd["time_to_vhalf"] + EPS) / max(ppm["time_to_vhalf"], EPS))
        return max(v_gain, p_gain, t_gain) >= 1.3

    shared_gate_pass = any(gate_passes_on_lr(lr) for lr in SHARED_LRS)
    all_shared_valid = all(trio_valid(lr) for lr in SHARED_LRS)
    if (not shared_gate_pass) and all_shared_valid:
        tested_lrs.append(EXTRA_LR)
        for method in ["sgd", "egm", "ppm"]:
            print(f"[baseline] method={method} lr={EXTRA_LR} start", flush=True)
            curves, summary = run_method(game, cfg, method, EXTRA_LR, BASELINE_ITERS)
            baseline_rows.append(summary)
            baseline_curves.extend(curves)
            write_csv(RESULT_ROOT / f"{OUTPUT_PREFIX}baseline_gate.csv", baseline_rows)
            print(
                f"[baseline] method={method} lr={EXTRA_LR} done valid={summary['valid_flag']} auc_V={summary['auc_V_lambda']:.6e} auc_P={summary['auc_normalized_P_tau']:.6e}",
                flush=True,
            )
    write_csv(RESULT_ROOT / f"{OUTPUT_PREFIX}baseline_gate.csv", baseline_rows)

    decision = "COUPLED_PENDULUM_BASELINE_FAIL"
    selected_lr = SHARED_LRS[0]
    lines = [
        f"# {OUTPUT_PREFIX}baseline_gate_report",
        "",
        f"- tested_shared_lrs: `{tested_lrs}`",
        "- identical_config: `True` for SGD / EGM / PPM",
        "",
    ]
    for lr in tested_lrs:
        group = [row for row in baseline_rows if row["lr"] == lr]
        if len(group) != 3:
            continue
        sgd = next(row for row in group if row["method"] == "sgd")
        egm = next(row for row in group if row["method"] == "egm")
        ppm = next(row for row in group if row["method"] == "ppm")
        valid = all(row["valid_flag"] == 1 for row in group)
        lines.append(
            f"- lr `{lr}`: valid=`{valid}`, auc_V [sgd=`{sgd['auc_V_lambda']:.6e}`, egm=`{egm['auc_V_lambda']:.6e}`, ppm=`{ppm['auc_V_lambda']:.6e}`], "
            f"auc_P_tau [sgd=`{sgd['auc_normalized_P_tau']:.6e}`, egm=`{egm['auc_normalized_P_tau']:.6e}`, ppm=`{ppm['auc_normalized_P_tau']:.6e}`]"
        )
        if not valid:
            continue
        v_gain = max(sgd["auc_V_lambda"] / max(egm["auc_V_lambda"], EPS), sgd["auc_V_lambda"] / max(ppm["auc_V_lambda"], EPS))
        p_gain = max(sgd["auc_normalized_P_tau"] / max(egm["auc_normalized_P_tau"], EPS), sgd["auc_normalized_P_tau"] / max(ppm["auc_normalized_P_tau"], EPS))
        t_gain = max((sgd["time_to_vhalf"] + EPS) / max(egm["time_to_vhalf"], EPS), (sgd["time_to_vhalf"] + EPS) / max(ppm["time_to_vhalf"], EPS))
        if max(v_gain, p_gain, t_gain) >= 1.3:
            decision = "COUPLED_PENDULUM_READY_FOR_PROPOSED"
            selected_lr = lr
            break
        if lr == SHARED_LRS[0]:
            selected_lr = lr
    if decision == "COUPLED_PENDULUM_READY_FOR_PROPOSED":
        lines.append("")
        lines.append(f"- baseline_gate_decision: `PASS` at shared lr `{selected_lr}`")
    else:
        lines.append("")
        lines.append("- baseline_gate_decision: `FAIL`")
        lines.append("- reason: no shared-lr healthy trio showed the required 1.3x extragradient-style advantage over SGD.")
    write_text(RESULT_ROOT / f"{OUTPUT_PREFIX}baseline_gate_report.md", "\n".join(lines) + "\n")
    return decision, float(selected_lr), baseline_rows, baseline_curves, "\n".join(lines) + "\n"


def load_existing_coupled_ready_state() -> tuple[CoupledConfig, float]:
    cfg_path = RESULT_ROOT / f"{OUTPUT_PREFIX}selected_config.json"
    if not cfg_path.exists():
        raise RuntimeError(f"Missing selected config: {cfg_path}")
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    weights = LyapunovWeights(
        lambda_F=float(raw["lambda_F"]),
        lambda_P=float(raw["lambda_P"]),
        lambda_C=float(raw["lambda_C"]),
    )
    cfg = CoupledConfig(
        reward_scale=float(raw["reward_scale"]),
        beta_rot=float(raw["beta_rot"]),
        beta_sym=float(raw["beta_sym"]),
        eta_coup=float(raw["eta_coup"]),
        alpha_dyn=float(raw["alpha_dyn"]),
        a_u=float(raw["a_u"]),
        a_w=float(raw["a_w"]),
        game_action_dim=int(raw["game_action_dim"]),
        rollout_steps=int(raw["rollout_steps"]),
        weights=weights,
        init_log_std=float(raw["init_log_std"]),
    )
    decision_path = RESULT_ROOT / f"{OUTPUT_PREFIX}final_decision.md"
    if not decision_path.exists():
        raise RuntimeError(f"Missing coupled readiness decision: {decision_path}")
    decision = decision_path.read_text(encoding="utf-8").strip()
    if decision != "COUPLED_PENDULUM_READY_FOR_PROPOSED":
        raise RuntimeError(f"Coupled Pendulum is not ready for proposed: {decision}")
    report_path = RESULT_ROOT / f"{OUTPUT_PREFIX}baseline_gate_report.md"
    report_text = report_path.read_text(encoding="utf-8")
    marker = "PASS` at shared lr `"
    if marker not in report_text:
        raise RuntimeError(f"Could not parse selected shared lr from {report_path}")
    tail = report_text.split(marker, 1)[1]
    lr_text = tail.split("`", 1)[0]
    return cfg, float(lr_text)


def write_coupled_field_check(cfg: CoupledConfig) -> None:
    lines = [
        f"# {OUTPUT_PREFIX}field_check",
        "",
        "- frozen_batch_obs: `True`",
        "- fixed_eps_P_eps_A_during_update: `True`",
        "- recompute_u_current_and_w_current_from_current_actor_logstd: `True`",
        "- differentiable_J_coup(theta, phi): `True`",
        "- J_coup_applied_only_to_actor_logstd_blocks: `True`",
        "- J_coup_in_critic_blocks: `False`",
        "- field_is_block_signed_not_single_scalar_total_loss: `True`",
        "",
        f"- reward_scale: `{cfg.reward_scale:.10f}`",
        f"- beta_rot: `{cfg.beta_rot:.10f}`",
        f"- beta_sym: `{cfg.beta_sym:.10f}`",
        f"- eta_coup: `{cfg.eta_coup:.10f}`",
        "",
        "- protagonist_actor_logstd_block: `grad(L_PPO_actor_P - ent_coef * entropy_P - eta_coup * J_coup)`",
        "- adversary_actor_logstd_block: `grad(L_PPO_actor_A - ent_coef * entropy_A + eta_coup * J_coup)`",
        "- protagonist_critic_block: `grad(vf_coef * L_v_P)`",
        "- adversary_critic_block: `grad(vf_coef * L_v_A)`",
    ]
    write_text(RESULT_ROOT / f"{OUTPUT_PREFIX}field_check.md", "\n".join(lines) + "\n")


def compute_field_and_g(game: CoupledJointPPOPendulum, z: torch.Tensor, batch: dict[str, torch.Tensor], cfg: CoupledConfig) -> tuple[torch.Tensor, torch.Tensor]:
    z_req = z.detach().clone().requires_grad_(True)
    field_z = game.field(z_req, batch, cfg)
    _, g_vec = torch.autograd.functional.jvp(
        lambda zz: game.field(zz, batch, cfg),
        (z_req,),
        (field_z.detach(),),
        create_graph=False,
        strict=False,
    )
    return field_z.detach(), g_vec.detach()


def fit_quadratic_1d(v0: float, v1: float, v2: float, h: float) -> tuple[float, float]:
    mat = np.array([[h, 0.5 * h * h], [2.0 * h, 2.0 * h * h]], dtype=np.float64)
    rhs = np.array([v1 - v0, v2 - v0], dtype=np.float64)
    b, c = np.linalg.solve(mat, rhs)
    return float(b), float(c)


def fit_quadratic_2d(samples: list[tuple[float, float, float]]) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    rhs = []
    for s, t, value in samples:
        rows.append([s, t, 0.5 * s * s, s * t, 0.5 * t * t])
        rhs.append(value)
    coeffs, *_ = np.linalg.lstsq(np.asarray(rows, dtype=np.float64), np.asarray(rhs, dtype=np.float64), rcond=None)
    b = np.array([coeffs[0], coeffs[1]], dtype=np.float64)
    hess = np.array([[coeffs[2], coeffs[3]], [coeffs[3], coeffs[4]]], dtype=np.float64)
    return b, hess


def apply_trust(delta: torch.Tensor, radius: float) -> tuple[torch.Tensor, bool]:
    norm = float(torch.linalg.norm(delta).item())
    if norm <= radius:
        return delta, False
    scale = radius / (norm + EPS)
    return delta * scale, True


def evaluate_method_step(
    game: CoupledJointPPOPendulum,
    method: str,
    z: torch.Tensor,
    batch: dict[str, torch.Tensor],
    cfg: CoupledConfig,
    lr: float,
    update_radius: float | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    metrics_before = game.approximate_metrics(z, batch, cfg, lr, compute_ptau=True, last_ptau=None, compute_geometry=True)
    v_before = metrics_before["V_lambda"]
    field, g_vec = compute_field_and_g(game, z, batch, cfg)
    n_f = float(torch.linalg.norm(field).item())
    n_g = float(torch.linalg.norm(g_vec).item())
    g_over_f = n_g / (n_f + EPS)
    cos_fg = float(torch.dot(field, g_vec).item() / ((n_f * n_g) + EPS))
    non_col = math.sqrt(max(0.0, 1.0 - min(1.0, cos_fg * cos_fg)))

    beta = 0.0
    gamma = 0.0
    gamma_active = 0
    fallback = 0
    fallback_reason = ""
    trust_active = 0
    predicted_after = math.nan
    g_contrib_ratio = 0.0
    if method == "sgd":
        next_z, base_meta = run_sgd(game, z, batch, cfg, lr)
        return next_z, {
            **base_meta,
            "beta": beta,
            "gamma": gamma,
            "gamma_active": gamma_active,
            "fallback_to_egm": fallback,
            "fallback_reason": fallback_reason,
            "trust_radius_active": trust_active,
            "V_before": v_before,
            "V_actual_after": math.nan,
            "V_predicted_after": predicted_after,
            "prediction_error": math.nan,
            "g_over_f": g_over_f,
            "cos_fg": cos_fg,
            "non_collinearity": non_col,
            "field_norm_pre": n_f,
            "g_norm_pre": n_g,
            "G_contribution_ratio": g_contrib_ratio,
        }
    if method == "egm":
        next_z, base_meta = run_egm(game, z, batch, cfg, lr)
        return next_z, {
            **base_meta,
            "beta": beta,
            "gamma": gamma,
            "gamma_active": gamma_active,
            "fallback_to_egm": fallback,
            "fallback_reason": fallback_reason,
            "trust_radius_active": trust_active,
            "V_before": v_before,
            "V_actual_after": math.nan,
            "V_predicted_after": predicted_after,
            "prediction_error": math.nan,
            "g_over_f": g_over_f,
            "cos_fg": cos_fg,
            "non_collinearity": non_col,
            "field_norm_pre": n_f,
            "g_norm_pre": n_g,
            "G_contribution_ratio": g_contrib_ratio,
        }
    if method == "ppm":
        next_z, base_meta = run_ppm(game, z, batch, cfg, lr)
        return next_z, {
            **base_meta,
            "beta": beta,
            "gamma": gamma,
            "gamma_active": gamma_active,
            "fallback_to_egm": fallback,
            "fallback_reason": fallback_reason,
            "trust_radius_active": trust_active,
            "V_before": v_before,
            "V_actual_after": math.nan,
            "V_predicted_after": predicted_after,
            "prediction_error": math.nan,
            "g_over_f": g_over_f,
            "cos_fg": cos_fg,
            "non_collinearity": non_col,
            "field_norm_pre": n_f,
            "g_norm_pre": n_g,
            "G_contribution_ratio": g_contrib_ratio,
        }

    if update_radius is None:
        raise ValueError(f"update_radius required for {method}")

    step_probe = min(max(update_radius, 1e-5), 1e-3)
    dir_f = (-field / (n_f + EPS)) if n_f > 0.0 else torch.zeros_like(field)
    if method == "proposed_noG":
        vals = []
        for scale in [0.0, step_probe, 2.0 * step_probe]:
            probe_z = z + (scale * dir_f)
            probe_v = game.approximate_metrics(probe_z, batch, cfg, lr, compute_ptau=True, last_ptau=None, compute_geometry=False)["V_lambda"]
            vals.append(probe_v)
        b1, h11 = fit_quadratic_1d(vals[0], vals[1], vals[2], step_probe)
        if h11 > 1e-12:
            s_opt = float(np.clip(-b1 / h11, -2.0 * step_probe, 2.0 * step_probe))
        else:
            s_opt = float(step_probe if vals[1] < vals[0] else 0.0)
        delta_raw = s_opt * dir_f
        predicted_after = float(vals[0] + (b1 * s_opt) + (0.5 * h11 * s_opt * s_opt))
        beta = s_opt / (n_f + EPS)
    else:
        dir_g = (g_vec / (n_g + EPS)) if n_g > 0.0 else torch.zeros_like(g_vec)
        samples = []
        for s, t in [
            (step_probe, 0.0),
            (0.0, step_probe),
            (2.0 * step_probe, 0.0),
            (0.0, 2.0 * step_probe),
            (step_probe, step_probe),
        ]:
            probe_z = z + (s * dir_f) + (t * dir_g)
            probe_v = game.approximate_metrics(probe_z, batch, cfg, lr, compute_ptau=True, last_ptau=None, compute_geometry=False)["V_lambda"]
            samples.append((s, t, probe_v - v_before))
        b_vec, hess = fit_quadratic_2d(samples)
        hess = hess + (1e-6 * np.eye(2, dtype=np.float64))
        try:
            opt = -np.linalg.solve(hess, b_vec)
        except np.linalg.LinAlgError:
            opt = -np.linalg.pinv(hess) @ b_vec
        opt = np.clip(opt, -2.0 * step_probe, 2.0 * step_probe)
        s_opt = float(opt[0])
        t_opt = float(opt[1])
        delta_raw = (s_opt * dir_f) + (t_opt * dir_g)
        predicted_after = float(v_before + (b_vec @ opt) + 0.5 * (opt @ hess @ opt))
        beta = s_opt / (n_f + EPS)
        gamma = t_opt / (n_g + EPS)
        gamma_active = int(abs(gamma) > 1e-12)
        g_contrib_ratio = abs(t_opt) / (abs(s_opt) + abs(t_opt) + EPS)

    delta, trust_on = apply_trust(delta_raw, update_radius)
    trust_active = int(trust_on)
    candidate_z = z + delta
    after_metrics = game.approximate_metrics(candidate_z, batch, cfg, lr, compute_ptau=True, last_ptau=None, compute_geometry=False)
    v_after = after_metrics["V_lambda"]
    if (not finite(v_after)) or (v_after > (v_before + max(1e-6, 0.01 * abs(v_before)))):
        fallback = 1
        if not finite(v_after):
            fallback_reason = "nonfinite_V_after"
        else:
            fallback_reason = "V_after_worse_than_V_before"
        candidate_z, base_meta = run_egm(game, z, batch, cfg, lr)
        delta = candidate_z - z
        after_metrics = game.approximate_metrics(candidate_z, batch, cfg, lr, compute_ptau=True, last_ptau=None, compute_geometry=False)
        v_after = after_metrics["V_lambda"]
    else:
        base_meta = {"update_norm": float(torch.linalg.norm(delta).item())}
    return candidate_z.detach(), {
        **base_meta,
        "beta": float(beta),
        "gamma": float(gamma),
        "gamma_active": int(gamma_active),
        "fallback_to_egm": int(fallback),
        "fallback_reason": fallback_reason,
        "trust_radius_active": int(trust_active),
        "V_before": float(v_before),
        "V_actual_after": float(v_after),
        "V_predicted_after": float(predicted_after),
        "prediction_error": float(v_after - predicted_after) if finite(predicted_after) and finite(v_after) else math.nan,
        "g_over_f": float(g_over_f),
        "cos_fg": float(cos_fg),
        "non_collinearity": float(non_col),
        "field_norm_pre": float(n_f),
        "g_norm_pre": float(n_g),
        "G_contribution_ratio": float(g_contrib_ratio),
    }


def run_method_proposed(
    game: CoupledJointPPOPendulum,
    cfg: CoupledConfig,
    method: str,
    lr: float,
    outer_iterations: int,
    initial_z: torch.Tensor,
    update_radius: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    game.metric_refs = {}
    z = initial_z.detach().clone()
    curves: list[dict[str, Any]] = []
    auc_v = 0.0
    auc_ptau = 0.0
    auc_field = 0.0
    auc_exploit = 0.0
    auc_train = 0.0
    auc_eval = 0.0
    threshold_iter = None
    run_valid = True
    last_ptau = None
    gamma_active_total = 0
    fallback_total = 0
    trust_total = 0
    g_contrib_total = 0.0
    eval_game, eval_task = game.evaluate_policy(z, cfg, PPO_EVAL_EPISODES)
    for iteration in range(outer_iterations + 1):
        batch = game.collect_rollout(z, cfg, rollout_seed=SEED + (1000 * iteration) + 99)
        compute_ptau = (iteration % P_TAU_EVAL_INTERVAL == 0) or last_ptau is None
        metrics = game.approximate_metrics(z, batch, cfg, lr, compute_ptau=compute_ptau, last_ptau=last_ptau, compute_geometry=False)
        if compute_ptau:
            last_ptau = metrics["P_tau"]
        if iteration % PPO_EVAL_INTERVAL == 0 or iteration == outer_iterations:
            eval_game, eval_task = game.evaluate_policy(z, cfg, PPO_EVAL_EPISODES)
        metrics["eval_game_return"] = eval_game
        metrics["eval_original_task_return"] = eval_task
        metrics["iteration"] = iteration
        metrics["method"] = method
        metrics["lr"] = lr
        metrics["update_radius"] = float(update_radius) if update_radius is not None else math.nan
        metrics["gradient_norm"] = metrics["field_norm"]
        metrics["nan_flag"] = int(not all(finite(metrics[key]) for key in ["V_lambda", "field_norm", "P_tau", "critic_loss_total", "train_game_return", "eval_game_return", "gradient_norm"]))
        valid = run_valid_from_metrics(metrics)
        metrics["valid_flag"] = int(valid)
        run_valid = run_valid and bool(valid)
        curves.append(dict(metrics))
        auc_v += metrics["V_lambda"]
        auc_ptau += metrics["normalized_P_tau"]
        auc_field += metrics["field_norm"]
        auc_exploit += metrics["exploitability_proxy"]
        auc_train += metrics["train_game_return"]
        auc_eval += metrics["eval_game_return"]
        if threshold_iter is None and metrics["V_lambda"] <= 0.5:
            threshold_iter = iteration
        if iteration == outer_iterations:
            break
        z, update_meta = evaluate_method_step(game, method, z, batch, cfg, lr, update_radius)
        gamma_active_total += int(update_meta.get("gamma_active", 0))
        fallback_total += int(update_meta.get("fallback_to_egm", 0))
        trust_total += int(update_meta.get("trust_radius_active", 0))
        g_contrib_total += float(update_meta.get("G_contribution_ratio", 0.0))
        curves[-1].update(update_meta)
        curves[-1]["gamma_active_frac"] = gamma_active_total / max(1, iteration + 1)
        curves[-1]["fallback_to_egm_frac"] = fallback_total / max(1, iteration + 1)
        curves[-1]["trust_radius_active_frac"] = trust_total / max(1, iteration + 1)
    final = curves[-1]
    summary = {
        **cfg.to_row(),
        "method": method,
        "lr": lr,
        "update_radius": float(update_radius) if update_radius is not None else math.nan,
        "valid_flag": int(run_valid),
        "auc_V_lambda": auc_v,
        "auc_normalized_P_tau": auc_ptau,
        "auc_field_norm": auc_field,
        "auc_exploitability": auc_exploit,
        "auc_train_game_return": auc_train,
        "auc_eval_game_return": auc_eval,
        "final_V_lambda": final["V_lambda"],
        "final_normalized_P_tau": final["normalized_P_tau"],
        "final_field_norm": final["field_norm"],
        "final_exploitability": final["exploitability_proxy"],
        "final_train_game_return": final["train_game_return"],
        "final_eval_game_return": final["eval_game_return"],
        "final_train_original_task_return": final["train_original_task_return"],
        "final_eval_original_task_return": final["eval_original_task_return"],
        "final_action_clip_fraction": final["action_clip_fraction"],
        "final_mean_KL_P": final["mean_KL_P"],
        "final_mean_KL_A": final["mean_KL_A"],
        "final_ratio_clip_fraction_P": final["ratio_clip_fraction_P"],
        "final_ratio_clip_fraction_A": final["ratio_clip_fraction_A"],
        "final_log_std_mean_P": final["log_std_mean_P"],
        "final_log_std_mean_A": final["log_std_mean_A"],
        "gamma_active_frac": gamma_active_total / max(1, outer_iterations),
        "fallback_to_egm_frac": fallback_total / max(1, outer_iterations),
        "trust_radius_active_frac": trust_total / max(1, outer_iterations),
        "G_contribution_ratio_mean": g_contrib_total / max(1, outer_iterations),
        "time_to_vhalf": threshold_iter if threshold_iter is not None else outer_iterations + 1,
    }
    return curves, summary


def choose_proposed_radius(preflight_rows: list[dict[str, Any]]) -> float:
    radii = sorted({float(row["update_radius"]) for row in preflight_rows})
    grouped = {radius: [row for row in preflight_rows if float(row["update_radius"]) == radius] for radius in radii}

    def radius_ok(radius: float) -> bool:
        rows = grouped[radius]
        if len(rows) != 2:
            return False
        by_method = {row["method"]: row for row in rows}
        if "proposed_noG" not in by_method or "proposed_QP_G" not in by_method:
            return False
        no_g = by_method["proposed_noG"]
        qpg = by_method["proposed_QP_G"]
        return (
            no_g["valid_flag"] == 1
            and qpg["valid_flag"] == 1
            and no_g["final_V_lambda"] <= (no_g["initial_V_lambda"] + 1e-6)
            and qpg["final_V_lambda"] <= (qpg["initial_V_lambda"] + 1e-6)
            and no_g["fallback_to_egm_frac"] < 0.2
            and qpg["fallback_to_egm_frac"] < 0.2
            and qpg["gamma_active_frac"] > 0.0
            and qpg["G_contribution_ratio_mean"] > 0.0
        )

    for radius in radii:
        if radius_ok(radius):
            return radius
    best_radius = radii[0]
    best_key = None
    for radius in radii:
        rows = grouped[radius]
        score = 0.0
        valid = sum(int(row["valid_flag"]) for row in rows)
        fallback = max(float(row["fallback_to_egm_frac"]) for row in rows)
        v_end = sum(float(row["final_V_lambda"]) for row in rows)
        g_ratio = max(float(row["G_contribution_ratio_mean"]) for row in rows)
        gamma_frac = max(float(row["gamma_active_frac"]) for row in rows)
        key = (-valid, fallback, v_end, -g_ratio, -gamma_frac, radius)
        if best_key is None or key < best_key:
            best_key = key
            best_radius = radius
    return best_radius


def proposed_radius_preflight(game: CoupledJointPPOPendulum, cfg: CoupledConfig, selected_lr: float) -> tuple[float, list[dict[str, Any]], str]:
    z0 = game.init_z(SEED, init_log_std=cfg.init_log_std)
    rows: list[dict[str, Any]] = []
    lines = [
        f"# {OUTPUT_PREFIX}proposed_radius_preflight_report",
        "",
        f"- shared_baseline_lr: `{selected_lr}`",
        "",
    ]
    for radius in PROPOSED_RADIUS_GRID:
        for method in ["proposed_noG", "proposed_QP_G"]:
            curves, summary = run_method_proposed(game, cfg, method, selected_lr, PROPOSED_PREFLIGHT_ITERS, z0, radius)
            first = curves[0]
            row = {
                **summary,
                "initial_V_lambda": first["V_lambda"],
                "initial_normalized_P_tau": first["normalized_P_tau"],
                "initial_field_norm": first["field_norm"],
                "initial_exploitability": first["exploitability_proxy"],
                "mean_KL_P": float(np.mean([c["mean_KL_P"] for c in curves])),
                "mean_KL_A": float(np.mean([c["mean_KL_A"] for c in curves])),
                "action_clip_fraction": float(np.mean([c["action_clip_fraction"] for c in curves])),
                "critic_loss_P": float(np.mean([c["critic_loss_P"] for c in curves])),
                "critic_loss_A": float(np.mean([c["critic_loss_A"] for c in curves])),
                "log_std_mean_P": float(np.mean([c["log_std_mean_P"] for c in curves])),
                "log_std_mean_A": float(np.mean([c["log_std_mean_A"] for c in curves])),
                "nan_flag": int(any(c["nan_flag"] for c in curves)),
            }
            rows.append(row)
    selected_radius = choose_proposed_radius(rows)
    write_csv(RESULT_ROOT / f"{OUTPUT_PREFIX}proposed_radius_preflight.csv", rows)
    lines.append(f"- selected_update_radius: `{selected_radius}`")
    for radius in PROPOSED_RADIUS_GRID:
        rad_rows = [row for row in rows if float(row["update_radius"]) == radius]
        if len(rad_rows) != 2:
            continue
        no_g = next(row for row in rad_rows if row["method"] == "proposed_noG")
        qpg = next(row for row in rad_rows if row["method"] == "proposed_QP_G")
        lines.append(
            f"- radius `{radius}`: noG valid=`{bool(no_g['valid_flag'])}` fallback=`{no_g['fallback_to_egm_frac']:.3f}` final_V=`{no_g['final_V_lambda']:.6e}`; "
            f"QP+G valid=`{bool(qpg['valid_flag'])}` fallback=`{qpg['fallback_to_egm_frac']:.3f}` gamma_active_frac=`{qpg['gamma_active_frac']:.3f}` "
            f"G_ratio=`{qpg['G_contribution_ratio_mean']:.3f}` final_V=`{qpg['final_V_lambda']:.6e}`"
        )
    write_text(RESULT_ROOT / f"{OUTPUT_PREFIX}proposed_radius_preflight_report.md", "\n".join(lines) + "\n")
    return selected_radius, rows, "\n".join(lines) + "\n"


def run_final_five_methods(game: CoupledJointPPOPendulum, cfg: CoupledConfig, selected_lr: float, update_radius: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    z0 = game.init_z(SEED, init_log_std=cfg.init_log_std)
    summary_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for method in ["sgd", "egm", "ppm", "proposed_noG", "proposed_QP_G"]:
        print(f"[proposed] method={method} start", flush=True)
        radius = update_radius if method.startswith("proposed") else None
        curves, summary = run_method_proposed(game, cfg, method, selected_lr, EXTENDED_ITERS, z0, radius)
        summary_rows.append(summary)
        curve_rows.extend(curves)
        write_csv(RESULT_ROOT / f"{OUTPUT_PREFIX}proposed_summary.csv", summary_rows)
        write_csv(RESULT_ROOT / f"{OUTPUT_PREFIX}proposed_curves.csv", curve_rows)
        print(f"[proposed] method={method} done valid={summary['valid_flag']} auc_V={summary['auc_V_lambda']:.6e}", flush=True)
    return summary_rows, curve_rows


def build_proposed_report(summary_rows: list[dict[str, Any]], update_radius: float, selected_lr: float) -> str:
    by_method = {row["method"]: row for row in summary_rows}
    lines = [
        f"# {OUTPUT_PREFIX}proposed_report",
        "",
        f"- shared_baseline_lr: `{selected_lr}`",
        f"- selected_update_radius: `{update_radius}`",
        "",
    ]
    for method in ["sgd", "egm", "ppm", "proposed_noG", "proposed_QP_G"]:
        row = by_method[method]
        lines.append(
            f"- {method}: valid=`{bool(row['valid_flag'])}`, auc_V=`{row['auc_V_lambda']:.6e}`, auc_P_tau=`{row['auc_normalized_P_tau']:.6e}`, "
            f"auc_field=`{row['auc_field_norm']:.6e}`, auc_exploit=`{row['auc_exploitability']:.6e}`, "
            f"fallback_frac=`{row.get('fallback_to_egm_frac', 0.0):.3f}`, gamma_active_frac=`{row.get('gamma_active_frac', 0.0):.3f}`, "
            f"G_ratio=`{row.get('G_contribution_ratio_mean', 0.0):.3f}`"
        )
    return "\n".join(lines) + "\n"


def run_same_start_comparison(game: CoupledJointPPOPendulum, cfg: CoupledConfig, selected_lr: float, update_radius: float) -> tuple[list[dict[str, Any]], str]:
    z = game.init_z(SEED, init_log_std=cfg.init_log_std)
    rows: list[dict[str, Any]] = []
    anchor_map: dict[int, tuple[torch.Tensor, dict[str, torch.Tensor]]] = {}
    for iteration in range(max(SAME_START_ITERS) + 1):
        batch = game.collect_rollout(z, cfg, rollout_seed=SEED + (1000 * iteration) + 99)
        if iteration in SAME_START_ITERS:
            anchor_map[iteration] = (z.detach().clone(), batch)
        if iteration == max(SAME_START_ITERS):
            break
        z, _ = run_sgd(game, z, batch, cfg, selected_lr)

    lines = [
        f"# {OUTPUT_PREFIX}same_start_comparison",
        "",
        "- anchor_trajectory: `SGD trajectory at shared lr`",
        "",
    ]
    for iteration in SAME_START_ITERS:
        anchor_z, batch = anchor_map[iteration]
        base_metrics = game.approximate_metrics(anchor_z, batch, cfg, selected_lr, compute_ptau=True, last_ptau=None, compute_geometry=False)
        updates: dict[str, torch.Tensor] = {}
        stats: dict[str, dict[str, float]] = {}
        zero_row = {
            "anchor_iteration": iteration,
            "method": "zero_step",
            "V_before": base_metrics["V_lambda"],
            "V_after": base_metrics["V_lambda"],
            "delta_V": 0.0,
            "field_term_after": base_metrics["field_term"],
            "P_tau_after": base_metrics["P_tau"],
            "critic_term_after": base_metrics["critic_term"],
            "field_norm_after": base_metrics["field_norm"],
            "exploitability_after": base_metrics["exploitability_proxy"],
            "train_return_after": base_metrics["train_game_return"],
            "eval_return_after": math.nan,
            "original_task_return_after": base_metrics["train_original_task_return"],
            "KL_P_after": base_metrics["mean_KL_P"],
            "KL_A_after": base_metrics["mean_KL_A"],
            "action_clip_fraction_after": base_metrics["action_clip_fraction"],
            "critic_loss_after": base_metrics["critic_loss_total"],
            "update_norm": 0.0,
            "cos_QP_SGD": math.nan,
            "cos_QP_EGM": math.nan,
            "cos_QP_PPM": math.nan,
            "cos_QP_noG": math.nan,
            "G_contribution_ratio": 0.0,
            "fallback_decision": "",
        }
        rows.append(zero_row)
        for method in ["sgd", "egm", "ppm", "proposed_noG", "proposed_QP_G"]:
            radius = update_radius if method.startswith("proposed") else None
            next_z, meta = evaluate_method_step(game, method, anchor_z, batch, cfg, selected_lr, radius)
            post = game.approximate_metrics(next_z, batch, cfg, selected_lr, compute_ptau=True, last_ptau=None, compute_geometry=False)
            eval_game, eval_task = game.evaluate_policy(next_z, cfg, PPO_EVAL_EPISODES)
            update_vec = next_z - anchor_z
            updates[method] = update_vec.detach()
            stats[method] = meta
            rows.append({
                "anchor_iteration": iteration,
                "method": method,
                "V_before": base_metrics["V_lambda"],
                "V_after": post["V_lambda"],
                "delta_V": post["V_lambda"] - base_metrics["V_lambda"],
                "field_term_after": post["field_term"],
                "P_tau_after": post["P_tau"],
                "critic_term_after": post["critic_term"],
                "field_norm_after": post["field_norm"],
                "exploitability_after": post["exploitability_proxy"],
                "train_return_after": post["train_game_return"],
                "eval_return_after": eval_game,
                "original_task_return_after": eval_task,
                "KL_P_after": post["mean_KL_P"],
                "KL_A_after": post["mean_KL_A"],
                "action_clip_fraction_after": post["action_clip_fraction"],
                "critic_loss_after": post["critic_loss_total"],
                "update_norm": float(torch.linalg.norm(update_vec).item()),
                "cos_QP_SGD": math.nan,
                "cos_QP_EGM": math.nan,
                "cos_QP_PPM": math.nan,
                "cos_QP_noG": math.nan,
                "G_contribution_ratio": meta.get("G_contribution_ratio", 0.0),
                "fallback_decision": meta.get("fallback_reason", ""),
            })
        qp = updates["proposed_QP_G"]
        denom_qp = float(torch.linalg.norm(qp).item()) + EPS
        for row in rows:
            if row["anchor_iteration"] != iteration:
                continue
            method = row["method"]
            if method in updates:
                upd = updates[method]
                row["cos_QP_SGD"] = float(torch.dot(qp, updates["sgd"]).item() / (denom_qp * (float(torch.linalg.norm(updates["sgd"]).item()) + EPS)))
                row["cos_QP_EGM"] = float(torch.dot(qp, updates["egm"]).item() / (denom_qp * (float(torch.linalg.norm(updates["egm"]).item()) + EPS)))
                row["cos_QP_PPM"] = float(torch.dot(qp, updates["ppm"]).item() / (denom_qp * (float(torch.linalg.norm(updates["ppm"]).item()) + EPS)))
                row["cos_QP_noG"] = float(torch.dot(qp, updates["proposed_noG"]).item() / (denom_qp * (float(torch.linalg.norm(updates["proposed_noG"]).item()) + EPS)))
        best_method = min(
            [row for row in rows if row["anchor_iteration"] == iteration and row["method"] != "zero_step"],
            key=lambda item: item["delta_V"],
        )["method"]
        lines.append(f"- anchor `{iteration}`: best actual delta V method=`{best_method}`")
    write_csv(RESULT_ROOT / f"{OUTPUT_PREFIX}same_start_comparison.csv", rows)
    lines.extend([
        "",
        "- same_start_note: `Rows are anchored on the SGD trajectory states and the same frozen rollout batch at each anchor iteration.`",
    ])
    text = "\n".join(lines) + "\n"
    write_text(RESULT_ROOT / f"{OUTPUT_PREFIX}same_start_comparison.md", text)
    return rows, text


def make_main_plots(curve_rows: list[dict[str, Any]]) -> None:
    if plt is None:
        return
    plot_dir = RESULT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    methods = ["sgd", "egm", "ppm", "proposed_noG", "proposed_QP_G"]
    colors = {
        "sgd": "#1f77b4",
        "egm": "#ff7f0e",
        "ppm": "#2ca02c",
        "proposed_noG": "#d62728",
        "proposed_QP_G": "#9467bd",
    }
    grouped = {method: [row for row in curve_rows if row["method"] == method] for method in methods}

    def lineplot(metric: str, title: str, path: Path, logy: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in methods:
            rows = grouped[method]
            xs = [row["iteration"] for row in rows]
            ys = [row[metric] for row in rows]
            ax.plot(xs, ys, label=method, color=colors[method], linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(metric)
        if logy:
            ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    lineplot("V_lambda", "Composite V_lambda", plot_dir / f"{OUTPUT_PREFIX}main_V_lambda.png")
    lineplot("normalized_P_tau", "Normalized P_tau", plot_dir / f"{OUTPUT_PREFIX}main_P_tau.png")
    lineplot("field_norm", "Field Norm", plot_dir / f"{OUTPUT_PREFIX}main_field_norm.png")
    lineplot("exploitability_proxy", "Approximate Local Exploitability", plot_dir / f"{OUTPUT_PREFIX}main_exploitability.png")

    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
    for method in methods:
        rows = grouped[method]
        xs = [row["iteration"] for row in rows]
        axes[0].plot(xs, [row["train_game_return"] for row in rows], label=method, color=colors[method], linewidth=1.8)
        axes[1].plot(xs, [row["eval_game_return"] for row in rows], label=method, color=colors[method], linewidth=1.8)
        axes[2].plot(xs, [row["eval_original_task_return"] for row in rows], label=method, color=colors[method], linewidth=1.8)
    axes[0].set_title("Train Game Return")
    axes[1].set_title("Eval Game Return")
    axes[2].set_title("Original Task Return")
    axes[2].set_xlabel("Iteration")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{OUTPUT_PREFIX}main_returns.png", dpi=180)
    plt.close(fig)

    health_metrics = [
        ("mean_KL_P", "KL_P"),
        ("mean_KL_A", "KL_A"),
        ("action_clip_fraction", "Action Clip Fraction"),
        ("critic_loss_P", "Critic Loss P"),
        ("critic_loss_A", "Critic Loss A"),
        ("log_std_mean_P", "Log Std P"),
        ("log_std_mean_A", "Log Std A"),
    ]
    fig, axes = plt.subplots(len(health_metrics), 1, figsize=(7, 2.4 * len(health_metrics)), sharex=True)
    for ax, (metric, title) in zip(axes, health_metrics):
        for method in methods:
            rows = grouped[method]
            ax.plot([row["iteration"] for row in rows], [row[metric] for row in rows], label=method, color=colors[method], linewidth=1.4)
        ax.set_title(title)
    axes[-1].set_xlabel("Iteration")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{OUTPUT_PREFIX}main_health.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    panel_specs = [
        ("V_lambda", "Composite V_lambda"),
        ("normalized_P_tau", "Normalized P_tau"),
        ("field_norm", "Field Norm"),
        ("exploitability_proxy", "Exploitability"),
        ("eval_game_return", "Eval Game Return"),
        ("action_clip_fraction", "Action Clip Fraction"),
    ]
    for ax, (metric, title) in zip(axes.flat, panel_specs):
        for method in methods:
            rows = grouped[method]
            ax.plot([row["iteration"] for row in rows], [row[metric] for row in rows], label=method, color=colors[method], linewidth=1.6)
        ax.set_title(title)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{OUTPUT_PREFIX}main_all_plots_big.png", dpi=180)
    plt.close(fig)


def build_proposed_final_decision(
    cfg: CoupledConfig,
    selected_lr: float,
    update_radius: float,
    summary_rows: list[dict[str, Any]],
    same_start_rows: list[dict[str, Any]],
) -> tuple[str, str]:
    by_method = {row["method"]: row for row in summary_rows}
    sgd = by_method["sgd"]
    egm = by_method["egm"]
    ppm = by_method["ppm"]
    no_g = by_method["proposed_noG"]
    qpg = by_method["proposed_QP_G"]
    baseline_ok = (egm["auc_V_lambda"] < sgd["auc_V_lambda"]) or (ppm["auc_V_lambda"] < sgd["auc_V_lambda"])
    qpg_beats_nog = (
        qpg["auc_V_lambda"] < no_g["auc_V_lambda"]
        and qpg["auc_normalized_P_tau"] < no_g["auc_normalized_P_tau"]
        and qpg["auc_field_norm"] < no_g["auc_field_norm"]
        and qpg["auc_exploitability"] < no_g["auc_exploitability"]
    )
    qpg_competitive = (
        qpg["auc_V_lambda"] <= min(egm["auc_V_lambda"], ppm["auc_V_lambda"]) * 1.05
        or qpg["auc_normalized_P_tau"] <= min(egm["auc_normalized_P_tau"], ppm["auc_normalized_P_tau"]) * 1.05
        or qpg["time_to_vhalf"] <= min(egm["time_to_vhalf"], ppm["time_to_vhalf"])
    )
    health_ok = all(row["valid_flag"] == 1 for row in summary_rows)
    fallback_low = qpg["fallback_to_egm_frac"] < 0.2
    gamma_nontrivial = qpg["gamma_active_frac"] > 0.0
    g_nontrivial = qpg["G_contribution_ratio_mean"] > 0.0
    same_start_qpg_best = any(
        row["method"] == "proposed_QP_G" and row["delta_V"] <= min(
            cand["delta_V"] for cand in same_start_rows if cand["anchor_iteration"] == row["anchor_iteration"] and cand["method"] != "zero_step"
        ) + 1e-9
        for row in same_start_rows
        if row["method"] == "proposed_QP_G"
    )
    if not health_ok:
        decision = "COUPLED_PENDULUM_HEALTH_FAIL"
    elif baseline_ok and qpg_beats_nog and qpg_competitive and fallback_low and gamma_nontrivial and g_nontrivial:
        decision = "COUPLED_PENDULUM_POSITIVE"
    elif baseline_ok and same_start_qpg_best and (not fallback_low or not qpg_competitive):
        decision = "COUPLED_PENDULUM_EARLY_QP_ONLY"
    else:
        decision = "COUPLED_PENDULUM_PROPOSED_FAIL"

    lines = [
        f"# {OUTPUT_PREFIX}proposed_final_decision",
        "",
        f"1. Coupled Pendulum config: reward_scale=`{cfg.reward_scale:.10f}`, beta_rot=`{cfg.beta_rot:.10f}`, beta_sym=`{cfg.beta_sym:.10f}`, eta_coup=`{cfg.eta_coup:.10f}`, alpha_dyn=`{cfg.alpha_dyn:.10f}`.",
        f"2. Shared baseline lr: `{selected_lr}`.",
        f"3. EGM/PPM still outperform SGD under identical lr/config? `EGM={egm['auc_V_lambda'] < sgd['auc_V_lambda']}`, `PPM={ppm['auc_V_lambda'] < sgd['auc_V_lambda']}`.",
        f"4. Selected proposed update_radius: `{update_radius}`.",
        f"5. proposed_noG improves over SGD/EGM/PPM? `{no_g['auc_V_lambda'] < min(sgd['auc_V_lambda'], egm['auc_V_lambda'], ppm['auc_V_lambda'])}`.",
        f"6. proposed_QP_G improves over proposed_noG? `{qpg_beats_nog}`.",
        f"7. proposed_QP_G beats or matches EGM/PPM? `{qpg_competitive}`.",
        f"8. QP+G improves V_lambda, P_tau, field norm, and exploitability? `{qpg_beats_nog}`.",
        f"9. gamma active? `{gamma_nontrivial}` with frac `{qpg['gamma_active_frac']:.3f}`.",
        f"10. G contribution nontrivial? `{g_nontrivial}` with mean `{qpg['G_contribution_ratio_mean']:.3f}`.",
        f"11. fallback_to_egm_frac below 0.2? `{fallback_low}` with frac `{qpg['fallback_to_egm_frac']:.3f}`.",
        f"12. PPO health metrics stable? `{health_ok}`.",
        f"13. Same-start supports QP+G? `{same_start_qpg_best}`.",
        f"14. Final decision: `{decision}`.",
    ]
    return decision, "\n".join(lines) + "\n"


def update_stage_progress(decision: str, cfg: CoupledConfig, selected_lr: float, update_radius: float) -> None:
    text = "\n".join([
        "# ppo_joint_rarl_s3_stage_progress_report",
        "",
        f"- environment: `{ENV_ID}`",
        f"- action_space_type: `Box`",
        f"- base_action_dim: `{BASE_ACTION_DIM}`",
        f"- game_action_dim: `{cfg.game_action_dim}`",
        f"- reward_scale: `{cfg.reward_scale:.10f}`",
        f"- beta_rot: `{cfg.beta_rot:.10f}`",
        f"- beta_sym: `{cfg.beta_sym:.10f}`",
        f"- alpha_dyn: `{cfg.alpha_dyn:.10f}`",
        f"- eta_coup: `{cfg.eta_coup:.10f}`",
        f"- selected_shared_lr: `{selected_lr}`",
        f"- selected_update_radius: `{update_radius}`",
        f"- final_decision: `{decision}`",
    ]) + "\n"
    write_text(RESULT_ROOT / "ppo_joint_rarl_s3_stage_progress_report.md", text)


def infer_state_from_obs(obs: np.ndarray) -> dict[str, float]:
    theta = float(math.atan2(float(obs[1]), float(obs[0])))
    theta_dot = float(obs[2])
    return {"theta": theta, "theta_dot": theta_dot}


def write_existing_root_cause_read() -> None:
    text = "\n".join([
        "# ppo_joint_rarl_s3_pendulum_baseline_curve_root_cause_read_existing",
        "",
        "1. `V_lambda` and `P_tau` were not evaluated on every iteration in the earlier baseline/proposed runs. `P_tau` was evaluated only when `iteration % P_TAU_EVAL_INTERVAL == 0`, with `P_TAU_EVAL_INTERVAL = 5`.",
        "2. Between `P_tau` evaluations, the code held the previous `P_tau` value and therefore also held the `normalized_P_tau` contribution inside `V_lambda`. This directly creates stair-step behavior in both `P_tau` and `V_lambda`.",
        "3. In the earlier runs, `V_lambda` was usually dominated by the `lambda_P * normalized_P_tau` term because `lambda_P = 1.0` while `lambda_F = 0.01` and `lambda_C = 0.1`.",
        "4. `field_norm` was evaluated on the newly collected on-policy training rollout batch at each iteration, not on a fixed diagnostic batch.",
        "5. The earlier curves were therefore train-batch diagnostics on changing rollout batches, not fixed-batch diagnostics. This means the curves mixed optimizer progress with rollout-to-rollout sampling noise.",
        "6. The earlier SGD/EGM/PPM baseline gate was only an AUC-based pass under the original protocol. It did not require smooth or clearly normal convergence curves on a fixed diagnostic batch.",
        "7. The later proposed final report said `EGM=False` and `PPM=False` because those methods were rerun in a longer 200-iteration comparison under the noisy changing-batch diagnostic protocol, and on that rerun their aggregate AUCs no longer beat SGD.",
        "8. Yes. The gate criterion needed to be tightened because the previous gate could pass on noisy, held-value, changing-batch diagnostics without requiring clean baseline convergence behavior.",
    ]) + "\n"
    write_text(RESULT_ROOT / "ppo_joint_rarl_s3_pendulum_baseline_curve_root_cause_read_existing.md", text)


def save_diagnostic_arrays(path: Path, batch: dict[str, torch.Tensor], init_states: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        diagnostic_observation_batch=batch["obs"].detach().cpu().numpy(),
        diagnostic_eps_P=batch["eps_P"].detach().cpu().numpy(),
        diagnostic_eps_A=batch["eps_A"].detach().cpu().numpy(),
        diagnostic_initial_states=np.asarray([[row["theta"], row["theta_dot"]] for row in init_states], dtype=np.float64),
    )


def build_fixed_diagnostic_batch(game: CoupledJointPPOPendulum, cfg: CoupledConfig, z_ref: torch.Tensor, label: str, write_protocol: bool = False) -> dict[str, Any]:
    env = gym.make(ENV_ID)
    init_states: list[dict[str, float]] = []
    for idx in range(16):
        obs, _ = env.reset(seed=SEED + 20000 + idx)
        init_states.append(infer_state_from_obs(np.asarray(obs, dtype=np.float64)))
    env.close()
    batch = game.collect_rollout(z_ref, cfg, rollout_seed=SEED + 25000)
    artifact_path = RESULT_ROOT / f"{label}_diag_arrays.npz"
    save_diagnostic_arrays(artifact_path, batch, init_states)
    if write_protocol:
        lines = [
            "# ppo_joint_rarl_s3_pendulum_fixed_diag_protocol",
            "",
            "- diagnostic_batch_type: `fixed frozen rollout batch collected once at run start from the initial joint parameter vector z0`",
            "- training_batches: `still on-policy and recollected each outer iteration`",
            "- main_diagnostics_batch: `fixed across methods and across iterations within the run`",
            f"- diagnostic_rollout_steps: `{cfg.rollout_steps}`",
            f"- diagnostic_artifact: `{artifact_path.name}`",
            f"- num_initial_states_stored: `{len(init_states)}`",
            f"- diagnostic_observation_batch_shape: `{tuple(int(x) for x in batch['obs'].shape)}`",
            f"- diagnostic_eps_P_shape: `{tuple(int(x) for x in batch['eps_P'].shape)}`",
            f"- diagnostic_eps_A_shape: `{tuple(int(x) for x in batch['eps_A'].shape)}`",
            "- diagnostic_metrics_use_fixed_eps: `True`",
            "- main_diagnostic_P_tau_hold_last: `False`",
            "- main_diagnostic_P_tau_eval_every_iter: `True`",
            "- geometry_diag_interval: `10 iterations when used in stabilization runs`",
        ]
        write_text(RESULT_ROOT / "ppo_joint_rarl_s3_pendulum_fixed_diag_protocol.md", "\n".join(lines) + "\n")
    return batch


def moving_average_slope(values: list[float], window: int = 5) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < window:
        xs = np.arange(len(arr), dtype=np.float64)
        return float(np.polyfit(xs, arr, 1)[0]) if len(arr) > 1 else 0.0
    kernel = np.ones(window, dtype=np.float64) / float(window)
    ma = np.convolve(arr, kernel, mode="valid")
    xs = np.arange(len(ma), dtype=np.float64)
    return float(np.polyfit(xs, ma, 1)[0]) if len(ma) > 1 else 0.0


def spike_ratio(values: list[float]) -> float:
    med = float(np.median(np.asarray(values, dtype=np.float64)))
    return float(max(values) / (med + EPS))


def make_cfg(base: CoupledConfig, rollout_steps: int | None = None, lambda_F: float | None = None, lambda_C: float | None = None, eta_coup: float | None = None) -> CoupledConfig:
    return CoupledConfig(
        reward_scale=base.reward_scale,
        beta_rot=base.beta_rot,
        beta_sym=base.beta_sym,
        eta_coup=base.eta_coup if eta_coup is None else eta_coup,
        alpha_dyn=base.alpha_dyn,
        a_u=base.a_u,
        a_w=base.a_w,
        game_action_dim=base.game_action_dim,
        rollout_steps=base.rollout_steps if rollout_steps is None else rollout_steps,
        weights=LyapunovWeights(
            lambda_F=base.weights.lambda_F if lambda_F is None else lambda_F,
            lambda_P=base.weights.lambda_P,
            lambda_C=base.weights.lambda_C if lambda_C is None else lambda_C,
        ),
        init_log_std=base.init_log_std,
    )


def strict_diag_metrics(game: CoupledJointPPOPendulum, z: torch.Tensor, diag_batch: dict[str, torch.Tensor], cfg: CoupledConfig, lr: float, compute_geometry: bool) -> dict[str, float]:
    metrics = game.approximate_metrics(z, diag_batch, cfg, lr, compute_ptau=True, last_ptau=None, compute_geometry=compute_geometry)
    comps = game.loss_components(z, diag_batch, cfg)
    p_gap = float(game.local_gap(z, diag_batch, cfg, lr, True, True).detach().item())
    a_gap = float(game.local_gap(z, diag_batch, cfg, lr, False, True).detach().item())
    p_exp = float(game.local_gap(z, diag_batch, cfg, lr, True, False).detach().item())
    a_exp = float(game.local_gap(z, diag_batch, cfg, lr, False, False).detach().item())
    field_component = cfg.weights.lambda_F * metrics["field_term"]
    p_tau_component = cfg.weights.lambda_P * metrics["normalized_P_tau"]
    critic_component = cfg.weights.lambda_C * metrics["critic_term"]
    total = field_component + p_tau_component + critic_component + EPS
    metrics.update({
        "V_lambda_diag": metrics["V_lambda"],
        "field_term_diag": metrics["field_term"],
        "normalized_P_tau_diag": metrics["normalized_P_tau"],
        "raw_P_tau_diag": metrics["P_tau"],
        "critic_term_diag": metrics["critic_term"],
        "field_norm_diag": metrics["field_norm"],
        "approximate_local_exploitability_diag": metrics["exploitability_proxy"],
        "J_coup_diag": float(comps["J_coup"].detach().item()),
        "actor_loss_P_diag": float(comps["L_PPO_actor_P"].detach().item()),
        "actor_loss_A_diag": float(comps["L_PPO_actor_A"].detach().item()),
        "critic_loss_P_diag": float(comps["L_v_P"].detach().item()),
        "critic_loss_A_diag": float(comps["L_v_A"].detach().item()),
        "field_component_value": field_component,
        "P_tau_component_value": p_tau_component,
        "critic_component_value": critic_component,
        "field_component_fraction": field_component / total,
        "P_tau_component_fraction": p_tau_component / total,
        "critic_component_fraction": critic_component / total,
        "P_tau_protagonist_gap": p_gap,
        "P_tau_adversary_gap": a_gap,
        "exploitability_protagonist": p_exp,
        "exploitability_adversary": a_exp,
    })
    return metrics


def run_strict_baseline_method(
    game: CoupledJointPPOPendulum,
    cfg: CoupledConfig,
    method: str,
    lr: float,
    outer_iterations: int,
    diag_batch: dict[str, torch.Tensor],
    initial_z: torch.Tensor,
    eval_every: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    game.metric_refs = {}
    z = initial_z.detach().clone()
    rows: list[dict[str, Any]] = []
    eval_game, eval_task = game.evaluate_policy(z, cfg, PPO_EVAL_EPISODES)
    for iteration in range(outer_iterations + 1):
        train_batch = game.collect_rollout(z, cfg, rollout_seed=SEED + 50000 + (1000 * iteration))
        diag = strict_diag_metrics(game, z, diag_batch, cfg, lr, compute_geometry=(iteration % 10 == 0))
        if iteration % eval_every == 0 or iteration == outer_iterations:
            eval_game, eval_task = game.evaluate_policy(z, cfg, PPO_EVAL_EPISODES)
        diag_row = {
            **cfg.to_row(),
            "method": method,
            "lr": lr,
            "iteration": iteration,
            **diag,
            "train_game_return": float(train_batch["train_game_return_mean"].item()),
            "train_original_task_return": float(train_batch["train_task_return_mean"].item()),
            "eval_game_return": float(eval_game),
            "eval_original_task_return": float(eval_task),
            "action_clip_fraction": float(train_batch["action_clip_fraction"].item()),
            "mean_abs_u": float(train_batch["mean_abs_u"].item()),
            "mean_abs_w": float(train_batch["mean_abs_w"].item()),
            "mean_abs_a_env_raw": float(train_batch["mean_abs_a_env_raw"].item()),
            "mean_abs_a_env_clipped": float(train_batch["mean_abs_a_env_clipped"].item()),
            "gradient_norm": diag["field_norm_diag"],
        }
        diag_row["nan_flag"] = int(not all(finite(diag_row[key]) for key in ["V_lambda_diag", "field_norm_diag", "raw_P_tau_diag", "critic_loss_total", "train_game_return", "eval_game_return", "gradient_norm"]))
        diag_row["valid_flag"] = int(run_valid_from_metrics({
            "V_lambda": diag_row["V_lambda_diag"],
            "field_norm": diag_row["field_norm_diag"],
            "P_tau": diag_row["raw_P_tau_diag"],
            "critic_loss_total": diag_row["critic_loss_total"],
            "train_game_return": diag_row["train_game_return"],
            "eval_game_return": diag_row["eval_game_return"],
            "gradient_norm": diag_row["gradient_norm"],
            "action_clip_fraction": diag_row["action_clip_fraction"],
            "mean_KL_P": diag_row["mean_KL_P"],
            "mean_KL_A": diag_row["mean_KL_A"],
            "ratio_clip_fraction_P": diag_row["ratio_clip_fraction_P"],
            "ratio_clip_fraction_A": diag_row["ratio_clip_fraction_A"],
            "log_std_mean_P": diag_row["log_std_mean_P"],
            "log_std_mean_A": diag_row["log_std_mean_A"],
            "train_original_task_return": diag_row["train_original_task_return"],
            "eval_original_task_return": diag_row["eval_original_task_return"],
        }))
        rows.append(diag_row)
        if iteration == outer_iterations:
            break
        if method == "sgd":
            z, meta = run_sgd(game, z, train_batch, cfg, lr)
        elif method == "egm":
            z, meta = run_egm(game, z, train_batch, cfg, lr)
        elif method == "ppm":
            z, meta = run_ppm(game, z, train_batch, cfg, lr)
        else:
            raise ValueError(method)
        rows[-1].update(meta)

    v_vals = [row["V_lambda_diag"] for row in rows]
    p_vals = [row["normalized_P_tau_diag"] for row in rows]
    f_vals = [row["field_norm_diag"] for row in rows]
    summary = {
        **cfg.to_row(),
        "method": method,
        "lr": lr,
        "valid_flag": int(all(int(row["valid_flag"]) == 1 for row in rows)),
        "V_lambda_diag_start": v_vals[0],
        "V_lambda_diag_final": v_vals[-1],
        "V_lambda_diag_AUC": float(sum(v_vals)),
        "V_lambda_diag_moving_average_slope": moving_average_slope(v_vals),
        "V_lambda_diag_spike_ratio": spike_ratio(v_vals),
        "P_tau_diag_start": p_vals[0],
        "P_tau_diag_final": p_vals[-1],
        "P_tau_diag_AUC": float(sum(p_vals)),
        "P_tau_diag_moving_average_slope": moving_average_slope(p_vals),
        "P_tau_diag_spike_ratio": spike_ratio(p_vals),
        "field_norm_diag_start": f_vals[0],
        "field_norm_diag_final": f_vals[-1],
        "field_norm_diag_AUC": float(sum(f_vals)),
        "field_norm_diag_moving_average_slope": moving_average_slope(f_vals),
        "field_norm_diag_spike_ratio": spike_ratio(f_vals),
        "final_cross_player_coupling_proxy_diag": rows[-1]["cross_player_coupling_proxy"] if finite(rows[-1]["cross_player_coupling_proxy"]) else math.nan,
        "final_cross_to_same_ratio_diag": rows[-1]["cross_to_same_ratio"] if finite(rows[-1]["cross_to_same_ratio"]) else math.nan,
        "final_rotation_ratio_proxy_diag": rows[-1]["rotation_ratio_proxy"] if finite(rows[-1]["rotation_ratio_proxy"]) else math.nan,
        "final_eval_game_return": rows[-1]["eval_game_return"],
        "final_eval_original_task_return": rows[-1]["eval_original_task_return"],
    }
    curve_normal = (
        summary["V_lambda_diag_final"] <= summary["V_lambda_diag_start"] + 1e-8
        and summary["P_tau_diag_final"] <= summary["P_tau_diag_start"] + 1e-8
        and summary["V_lambda_diag_spike_ratio"] <= 5.0
        and summary["P_tau_diag_spike_ratio"] <= 5.0
        and summary["field_norm_diag_spike_ratio"] <= 5.0
        and summary["field_norm_diag_final"] <= (5.0 * summary["field_norm_diag_start"] + 1e-8)
    )
    summary["curve_normal_flag"] = int(curve_normal)
    return rows, summary


def strict_geometry_row(game: CoupledJointPPOPendulum, cfg: CoupledConfig, diag_batch: dict[str, torch.Tensor], initial_z: torch.Tensor) -> dict[str, Any]:
    geom = game.approximate_metrics(initial_z, diag_batch, cfg, lr=STRICT_SEARCH_LR, compute_ptau=True, last_ptau=None, compute_geometry=True)
    return {
        **cfg.to_row(),
        "rotation_ratio_proxy": geom["rotation_ratio_proxy"],
        "cross_player_coupling_proxy": geom["cross_player_coupling_proxy"],
        "cross_to_same_ratio": geom["cross_to_same_ratio"],
        "g_over_f": geom["g_over_f"],
        "cos_fg": geom["cos_fg"],
        "non_collinearity": geom["non_collinearity"],
    }


def run_baseline_stabilization_search(game: CoupledJointPPOPendulum, base_cfg: CoupledConfig) -> tuple[CoupledConfig, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    chosen_cfg = base_cfg
    chosen_bundle: dict[str, Any] = {}
    attempted_configs: list[tuple[str, CoupledConfig]] = []

    def eval_config(config_id: str, cfg: CoupledConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        z0 = game.init_z(SEED, init_log_std=cfg.init_log_std)
        diag_batch = build_fixed_diagnostic_batch(game, cfg, z0, f"{config_id}_strict_screen", write_protocol=False)
        geom_row = strict_geometry_row(game, cfg, diag_batch, z0)
        method_summaries = []
        for method in ["sgd", "egm", "ppm"]:
            method_curve_rows, summary = run_strict_baseline_method(game, cfg, method, STRICT_SEARCH_LR, STRICT_SCREEN_ITERS, diag_batch, z0, eval_every=10)
            summary = {**summary, "config_id": config_id, "phase": config_id[0]}
            curve_rows.extend([{**row, "config_id": config_id, "phase": config_id[0]} for row in method_curve_rows])
            method_summaries.append(summary)
            summary_rows.append({**summary, **geom_row, "search_lr": STRICT_SEARCH_LR})
        sgd = next(item for item in method_summaries if item["method"] == "sgd")
        egm = next(item for item in method_summaries if item["method"] == "egm")
        ppm = next(item for item in method_summaries if item["method"] == "ppm")
        winner_auc = min(egm["V_lambda_diag_AUC"], ppm["V_lambda_diag_AUC"])
        ratio = sgd["V_lambda_diag_AUC"] / (winner_auc + EPS)
        geometry_ok = (
            geom_row["cross_player_coupling_proxy"] > 0.0
            and geom_row["cross_to_same_ratio"] > 0.05
            and geom_row["rotation_ratio_proxy"] > 1e-4
            and geom_row["non_collinearity"] > 0.2
        )
        score = (
            0 if geometry_ok else 1,
            0 if sgd["curve_normal_flag"] == 1 else 1,
            0 if max(egm["curve_normal_flag"], ppm["curve_normal_flag"]) == 1 else 1,
            0 if ratio >= 1.3 else 1,
            sgd["V_lambda_diag_spike_ratio"],
            sgd["P_tau_diag_spike_ratio"],
            -ratio,
        )
        for summary in summary_rows[-3:]:
            summary["winner_vs_sgd_ratio"] = ratio
        attempted_configs.append((config_id, cfg))
        return {
            "config_id": config_id,
            "cfg": cfg,
            "geom_row": geom_row,
            "summaries": method_summaries,
            "ratio": ratio,
            "score": score,
        }, method_summaries

    stage_a = [
        ("A1", make_cfg(base_cfg, rollout_steps=1024, lambda_F=0.01, lambda_C=0.1, eta_coup=3.0)),
        ("A2", make_cfg(base_cfg, rollout_steps=2048, lambda_F=0.01, lambda_C=0.1, eta_coup=3.0)),
        ("A3", make_cfg(base_cfg, rollout_steps=4096, lambda_F=0.01, lambda_C=0.1, eta_coup=3.0)),
    ]
    a_results = []
    for config_id, cfg in stage_a:
        result, _ = eval_config(config_id, cfg)
        a_results.append(result)
    best_a = min(a_results, key=lambda item: item["score"])
    best_rollout = best_a["cfg"].rollout_steps

    stage_b = [
        ("B1", make_cfg(base_cfg, rollout_steps=best_rollout, lambda_F=0.01, lambda_C=0.1, eta_coup=3.0)),
        ("B2", make_cfg(base_cfg, rollout_steps=best_rollout, lambda_F=0.03, lambda_C=0.01, eta_coup=3.0)),
        ("B3", make_cfg(base_cfg, rollout_steps=best_rollout, lambda_F=0.1, lambda_C=0.0, eta_coup=3.0)),
    ]
    b_results = []
    for config_id, cfg in stage_b:
        result, _ = eval_config(config_id, cfg)
        b_results.append(result)
    best_b = min(b_results, key=lambda item: item["score"])

    if int(best_b["score"][0]) == 0 and int(best_b["score"][1]) == 0 and int(best_b["score"][2]) == 0 and int(best_b["score"][3]) == 0:
        chosen_cfg = best_b["cfg"]
        chosen_bundle = best_b
        return chosen_cfg, summary_rows, curve_rows, chosen_bundle

    stage_c = [
        ("C1", make_cfg(best_b["cfg"], rollout_steps=best_rollout, eta_coup=2.0)),
        ("C2", make_cfg(best_b["cfg"], rollout_steps=best_rollout, eta_coup=1.0)),
    ]
    c_results = []
    for config_id, cfg in stage_c:
        result, _ = eval_config(config_id, cfg)
        c_results.append(result)

    all_results = a_results + b_results + c_results
    best = min(all_results, key=lambda item: item["score"])
    chosen_cfg = best["cfg"]
    chosen_bundle = best
    return chosen_cfg, summary_rows, curve_rows, chosen_bundle


def run_final_strict_gate(game: CoupledJointPPOPendulum, cfg: CoupledConfig) -> tuple[str, float, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, torch.Tensor], torch.Tensor]:
    z0 = game.init_z(SEED, init_log_std=cfg.init_log_std)
    diag_batch = build_fixed_diagnostic_batch(game, cfg, z0, "strict_final", write_protocol=True)
    geom_row = strict_geometry_row(game, cfg, diag_batch, z0)
    summary_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for lr in [1e-5, 3e-5, 1e-4]:
        for method in ["sgd", "egm", "ppm"]:
            curves, summary = run_strict_baseline_method(game, cfg, method, lr, STRICT_FINAL_ITERS, diag_batch, z0, eval_every=1)
            summary_rows.append(summary)
            curve_rows.extend(curves)
    return "PENDING", 0.0, summary_rows, curve_rows, geom_row, diag_batch, z0


def strict_gate_decision(summary_rows: list[dict[str, Any]], geom_row: dict[str, Any]) -> tuple[str, float, str]:
    geometry_ok = (
        geom_row["cross_player_coupling_proxy"] > 0.0
        and geom_row["cross_to_same_ratio"] > 0.05
        and geom_row["rotation_ratio_proxy"] > 1e-4
        and geom_row["non_collinearity"] > 0.2
    )
    if not geometry_ok:
        return "FAIL_GEOMETRY_WEAK", 0.0, "Cross-player coupling or rotational structure became too weak under the stabilized config."
    selected_lr = 0.0
    best_reason = "No shared-lr trio produced clean curves plus a 1.3x EGM/PPM advantage over SGD."
    for lr in [1e-5, 3e-5, 1e-4]:
        trio = [row for row in summary_rows if abs(float(row["lr"]) - lr) < 1e-12]
        if len(trio) != 3:
            continue
        if not all(row["valid_flag"] == 1 for row in trio):
            continue
        sgd = next(row for row in trio if row["method"] == "sgd")
        egm = next(row for row in trio if row["method"] == "egm")
        ppm = next(row for row in trio if row["method"] == "ppm")
        winner = egm if egm["V_lambda_diag_AUC"] <= ppm["V_lambda_diag_AUC"] else ppm
        if sgd["curve_normal_flag"] != 1 or winner["curve_normal_flag"] != 1:
            best_reason = "At the best shared lr, either SGD or the winning EGM/PPM method still had abnormal diagnostic curves."
            continue
        if winner["V_lambda_diag_AUC"] <= 0.0 or winner["P_tau_diag_AUC"] <= 0.0 or winner["field_norm_diag_AUC"] <= 0.0:
            best_reason = "Unexpected nonpositive AUC values blocked a meaningful strict comparison."
            continue
        v_ratio = sgd["V_lambda_diag_AUC"] / (winner["V_lambda_diag_AUC"] + EPS)
        p_ratio = sgd["P_tau_diag_AUC"] / (winner["P_tau_diag_AUC"] + EPS)
        field_ok = winner["field_norm_diag_AUC"] <= (1.05 * sgd["field_norm_diag_AUC"])
        if max(v_ratio, p_ratio) >= 1.3 and field_ok:
            selected_lr = lr
            return "STRICT_BASELINE_PASS", selected_lr, f"Shared lr {lr} passed with winner {winner['method']} and strict fixed-batch diagnostics."
        if max(v_ratio, p_ratio) < 1.3:
            best_reason = "No EGM/PPM method achieved the required 1.3x advantage over SGD on fixed-batch diagnostic AUC."
        elif not field_ok:
            best_reason = "The candidate EGM/PPM winner did not improve or match SGD on field_norm diagnostic AUC."
    valid_any = any(row["valid_flag"] == 1 for row in summary_rows)
    if not valid_any:
        return "FAIL_PPO_HEALTH", selected_lr, "None of the strict baseline trials stayed within the PPO health limits."
    abnormal_any = any(row["curve_normal_flag"] != 1 for row in summary_rows)
    if abnormal_any:
        return "FAIL_CURVES_ABNORMAL", selected_lr, best_reason
    return "FAIL_NO_EGM_PPM_ADVANTAGE", selected_lr, best_reason


def write_v_component_audit(curve_rows: list[dict[str, Any]], selected_lr: float) -> None:
    rows = [row for row in curve_rows if abs(float(row["lr"]) - selected_lr) < 1e-12]
    write_csv(RESULT_ROOT / "ppo_joint_rarl_s3_pendulum_V_component_audit.csv", rows)
    by_method = {m: [row for row in rows if row["method"] == m] for m in ["sgd", "egm", "ppm"]}
    lines = ["# ppo_joint_rarl_s3_pendulum_V_component_audit", ""]
    for method, sub in by_method.items():
        if not sub:
            continue
        mean_p = float(np.mean([row["P_tau_component_fraction"] for row in sub]))
        mean_f = float(np.mean([row["field_component_fraction"] for row in sub]))
        mean_c = float(np.mean([row["critic_component_fraction"] for row in sub]))
        lines.append(
            f"- {method}: mean field fraction=`{mean_f:.3f}`, mean P_tau fraction=`{mean_p:.3f}`, mean critic fraction=`{mean_c:.3f}`, "
            f"P_gap_final=`{sub[-1]['P_tau_protagonist_gap']:.6e}`, A_gap_final=`{sub[-1]['P_tau_adversary_gap']:.6e}`"
        )
    lines.extend([
        "",
        "1. V_lambda is dominated by P_tau if the mean P_tau fraction is consistently much larger than both field and critic fractions.",
        "2. Under the strict protocol, P_tau is recomputed every iteration on the fixed diagnostic batch, so any remaining jaggedness is not caused by held values.",
        "3. Critic-term domination is visible when the critic fraction stays comparable to or above the P_tau fraction while field and exploitability remain flat.",
        "4. Field norm consistency is checked directly by comparing field-component fractions with field_norm diagnostic curves.",
        "5. Protagonist and adversary local gap stability is visible through the final and per-iteration gap columns in the CSV.",
    ])
    write_text(RESULT_ROOT / "ppo_joint_rarl_s3_pendulum_V_component_audit.md", "\n".join(lines) + "\n")


def write_stabilized_geometry(geom_row: dict[str, Any]) -> None:
    write_csv(RESULT_ROOT / "ppo_joint_rarl_s3_pendulum_stabilized_geometry_audit.csv", [geom_row])
    lines = [
        "# ppo_joint_rarl_s3_pendulum_stabilized_geometry_audit",
        "",
        f"- rotation_ratio_proxy: `{geom_row['rotation_ratio_proxy']:.6e}`",
        f"- cross_player_coupling_proxy: `{geom_row['cross_player_coupling_proxy']:.6e}`",
        f"- cross_to_same_ratio: `{geom_row['cross_to_same_ratio']:.6e}`",
        f"- ||G|| / ||F||: `{geom_row['g_over_f']:.6e}`",
        f"- cos(F,G): `{geom_row['cos_fg']:.6e}`",
        f"- non_collinearity: `{geom_row['non_collinearity']:.6e}`",
    ]
    write_text(RESULT_ROOT / "ppo_joint_rarl_s3_pendulum_stabilized_geometry_audit.md", "\n".join(lines) + "\n")


def write_strict_gate_report(decision: str, reason: str, summary_rows: list[dict[str, Any]], selected_lr: float) -> None:
    lines = [
        "# ppo_joint_rarl_s3_pendulum_strict_baseline_gate_report",
        "",
        f"- final_gate_decision: `{decision}`",
        f"- selected_shared_lr: `{selected_lr}`",
        f"- reason: `{reason}`",
        "",
    ]
    for lr in [1e-5, 3e-5, 1e-4]:
        trio = [row for row in summary_rows if abs(float(row["lr"]) - lr) < 1e-12]
        if len(trio) != 3:
            continue
        sgd = next(row for row in trio if row["method"] == "sgd")
        egm = next(row for row in trio if row["method"] == "egm")
        ppm = next(row for row in trio if row["method"] == "ppm")
        lines.append(
            f"- lr `{lr}`: SGD auc_V=`{sgd['V_lambda_diag_AUC']:.6e}` normal=`{bool(sgd['curve_normal_flag'])}`, "
            f"EGM auc_V=`{egm['V_lambda_diag_AUC']:.6e}` normal=`{bool(egm['curve_normal_flag'])}`, "
            f"PPM auc_V=`{ppm['V_lambda_diag_AUC']:.6e}` normal=`{bool(ppm['curve_normal_flag'])}`"
        )
    write_text(RESULT_ROOT / "ppo_joint_rarl_s3_pendulum_strict_baseline_gate_report.md", "\n".join(lines) + "\n")


def write_stabilization_grid(summary_rows: list[dict[str, Any]], chosen_bundle: dict[str, Any]) -> None:
    write_csv(RESULT_ROOT / "ppo_joint_rarl_s3_pendulum_baseline_stabilization_grid.csv", summary_rows)
    lines = [
        "# ppo_joint_rarl_s3_pendulum_baseline_stabilization_grid_report",
        "",
        f"- strict_search_lr: `{STRICT_SEARCH_LR}`",
        f"- chosen_config_id: `{chosen_bundle.get('config_id', 'unknown')}`",
        f"- chosen_rollout_steps: `{chosen_bundle.get('cfg').rollout_steps if chosen_bundle else 'unknown'}`",
        f"- chosen_lambda_F: `{chosen_bundle.get('cfg').weights.lambda_F if chosen_bundle else 'unknown'}`",
        f"- chosen_lambda_C: `{chosen_bundle.get('cfg').weights.lambda_C if chosen_bundle else 'unknown'}`",
        f"- chosen_eta_coup: `{chosen_bundle.get('cfg').eta_coup if chosen_bundle else 'unknown'}`",
        "",
        "- grid_note: `Grid A picked rollout_steps, Grid B tested Lyapunov weights on that rollout size, and Grid C reduced eta_coup only if A/B still did not yield a clean baseline candidate.`",
    ]
    for config_id in sorted({row["config_id"] for row in summary_rows}):
        trio = [row for row in summary_rows if row["config_id"] == config_id]
        sgd = next((row for row in trio if row["method"] == "sgd"), None)
        egm = next((row for row in trio if row["method"] == "egm"), None)
        ppm = next((row for row in trio if row["method"] == "ppm"), None)
        if not sgd or not egm or not ppm:
            continue
        lines.append(
            f"- {config_id}: rollout=`{sgd['rollout_steps']}`, lambda_F=`{sgd['lambda_F']}`, lambda_C=`{sgd['lambda_C']}`, eta_coup=`{sgd['eta_coup']}`, "
            f"SGD normal=`{bool(sgd['curve_normal_flag'])}`, EGM normal=`{bool(egm['curve_normal_flag'])}`, PPM normal=`{bool(ppm['curve_normal_flag'])}`, "
            f"best_ratio_vs_sgd=`{sgd['winner_vs_sgd_ratio']:.3f}`"
        )
    write_text(RESULT_ROOT / "ppo_joint_rarl_s3_pendulum_baseline_stabilization_grid_report.md", "\n".join(lines) + "\n")


def write_baseline_failure_root_cause(decision: str, chosen_cfg: CoupledConfig, geom_row: dict[str, Any], summary_rows: list[dict[str, Any]]) -> None:
    causes = []
    evidence = []
    if decision in {"FAIL_CURVES_ABNORMAL", "FAIL_METRIC_DESIGN"}:
        causes.append("A. Evaluation artifact")
        evidence.append("Earlier curves mixed changing rollout batches with held P_tau values. The strict fixed-batch protocol was introduced to remove exactly this artifact.")
    if any(row["P_tau_diag_spike_ratio"] > 5.0 for row in summary_rows):
        causes.append("B. Lyapunov weighting issue")
        evidence.append("Large P_tau spike ratios indicate that the composite V_lambda can still be dominated by the P_tau term unless field and critic weights are adjusted.")
    if chosen_cfg.rollout_steps >= 2048:
        causes.append("C. PPO noise issue")
        evidence.append("The need for larger rollout_steps to obtain cleaner curves indicates that on-policy PPO noise is a meaningful part of the instability.")
    if chosen_cfg.eta_coup >= 3.0:
        causes.append("D. Coupling design issue")
        evidence.append("The strongest geometry-preserving setting still relies on a large eta_coup, which can make the empirical objective more brittle.")
    causes.append("E. Pendulum wrapper issue")
    evidence.append("The latent 2D action coupling remains somewhat artificial, so even with fixed-batch diagnostics it may not produce an especially clean PPO benchmark.")
    if decision == "FAIL_NO_EGM_PPM_ADVANTAGE":
        causes.append("F. Algorithm issue")
        evidence.append("Even after stabilizing diagnostics, EGM/PPM did not show the required advantage over SGD under identical lr/config.")
    lines = ["# ppo_joint_rarl_s3_pendulum_baseline_failure_root_cause", ""]
    for idx, cause in enumerate(causes):
        lines.append(f"- {cause}")
        lines.append(f"  evidence: {evidence[idx]}")
    write_text(RESULT_ROOT / "ppo_joint_rarl_s3_pendulum_baseline_failure_root_cause.md", "\n".join(lines) + "\n")


def make_strict_baseline_plots(curve_rows: list[dict[str, Any]], selected_lr: float) -> None:
    if plt is None:
        return
    plot_dir = RESULT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    methods = ["sgd", "egm", "ppm"]
    colors = {"sgd": "#1f77b4", "egm": "#ff7f0e", "ppm": "#2ca02c"}
    grouped = {m: [row for row in curve_rows if row["method"] == m and abs(float(row["lr"]) - selected_lr) < 1e-12] for m in methods}
    specs = [
        ("V_lambda_diag", "V_lambda Diagnostic", f"{plot_dir}\\ppo_joint_rarl_s3_pendulum_strict_baseline_V_lambda.png"),
        ("normalized_P_tau_diag", "P_tau Diagnostic", f"{plot_dir}\\ppo_joint_rarl_s3_pendulum_strict_baseline_P_tau.png"),
        ("field_norm_diag", "Field Norm Diagnostic", f"{plot_dir}\\ppo_joint_rarl_s3_pendulum_strict_baseline_field_norm.png"),
        ("approximate_local_exploitability_diag", "Exploitability Diagnostic", f"{plot_dir}\\ppo_joint_rarl_s3_pendulum_strict_baseline_exploitability.png"),
    ]
    for metric, title, out_path in specs:
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in methods:
            rows = grouped[method]
            ax.plot([row["iteration"] for row in rows], [row[metric] for row in rows], label=method, color=colors[method], linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
    for method in methods:
        rows = grouped[method]
        axes[0].plot([row["iteration"] for row in rows], [row["train_game_return"] for row in rows], label=method, color=colors[method], linewidth=1.8)
        axes[1].plot([row["iteration"] for row in rows], [row["eval_game_return"] for row in rows], label=method, color=colors[method], linewidth=1.8)
        axes[2].plot([row["iteration"] for row in rows], [row["eval_original_task_return"] for row in rows], label=method, color=colors[method], linewidth=1.8)
    axes[0].set_title("Train Game Return")
    axes[1].set_title("Eval Game Return")
    axes[2].set_title("Original Task Return")
    axes[0].legend()
    axes[2].set_xlabel("Iteration")
    fig.tight_layout()
    fig.savefig(plot_dir / "ppo_joint_rarl_s3_pendulum_strict_baseline_returns.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    panels = [
        ("V_lambda_diag", "V_lambda"),
        ("normalized_P_tau_diag", "P_tau"),
        ("field_norm_diag", "Field Norm"),
        ("approximate_local_exploitability_diag", "Exploitability"),
        ("eval_game_return", "Eval Game Return"),
        ("action_clip_fraction", "Action Clip Fraction"),
    ]
    for ax, (metric, title) in zip(axes.flat, panels):
        for method in methods:
            rows = grouped[method]
            ax.plot([row["iteration"] for row in rows], [row[metric] for row in rows], label=method, color=colors[method], linewidth=1.6)
        ax.set_title(title)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "ppo_joint_rarl_s3_pendulum_strict_baseline_all_plots_big.png", dpi=180)
    plt.close(fig)


def write_ready_or_not(decision: str, cfg: CoupledConfig, selected_lr: float) -> None:
    if decision == "STRICT_BASELINE_PASS":
        text = "\n".join([
            "# ppo_joint_rarl_s3_pendulum_ready_for_proposed_strict",
            "",
            f"- rollout_steps: `{cfg.rollout_steps}`",
            f"- shared_lr: `{selected_lr}`",
            f"- lambda_F: `{cfg.weights.lambda_F}`",
            f"- lambda_P: `{cfg.weights.lambda_P}`",
            f"- lambda_C: `{cfg.weights.lambda_C}`",
            f"- eta_coup: `{cfg.eta_coup}`",
            f"- beta_rot: `{cfg.beta_rot}`",
            f"- beta_sym: `{cfg.beta_sym}`",
            f"- alpha_dyn: `{cfg.alpha_dyn}`",
            "- P_tau_eval_protocol: `fixed diagnostic batch, recomputed every iteration, no hold-last on the main diagnostic curves`",
        ]) + "\n"
        write_text(RESULT_ROOT / "ppo_joint_rarl_s3_pendulum_ready_for_proposed_strict.md", text)
    else:
        text = "\n".join([
            "# ppo_joint_rarl_s3_pendulum_not_suitable_after_strict_baseline",
            "",
            "Pendulum-v1 coupled PPO wrapper is not yet suitable as a clean positive Subsection 3 benchmark because the baselines do not exhibit stable, normal convergence under the stricter diagnostic protocol.",
            f"- final_decision: `{decision}`",
            f"- rollout_steps_tested_best: `{cfg.rollout_steps}`",
            f"- shared_lr_candidate: `{selected_lr}`",
            f"- lambda_F: `{cfg.weights.lambda_F}`",
            f"- lambda_P: `{cfg.weights.lambda_P}`",
            f"- lambda_C: `{cfg.weights.lambda_C}`",
            f"- eta_coup: `{cfg.eta_coup}`",
        ]) + "\n"
        write_text(RESULT_ROOT / "ppo_joint_rarl_s3_pendulum_not_suitable_after_strict_baseline.md", text)

def build_final_report(cfg: CoupledConfig, geom_row: dict[str, Any], baseline_rows: list[dict[str, Any]], decision: str, selected_lr: float) -> str:
    lines = [
        f"# {OUTPUT_PREFIX}final_report",
        "",
        "1. The previous frozen PPO objective failed because once old logprobs, returns, and advantages were frozen, the actor update field became nearly block-separable between protagonist and adversary.",
        "2. The new coupled frozen objective recomputes `u_current` and `w_current` on the frozen observation batch using fixed stored Gaussian noise, then injects a differentiable coupling term `J_coup(theta, phi)` into the actor/log_std blocks with opposite protagonist/adversary signs.",
        f"3. The new field has nonzero cross-player coupling? `{geom_row['cross_player_coupling_proxy'] > 0.0}`.",
        f"4. rotation_ratio_proxy after the fix: `{geom_row['rotation_ratio_proxy']:.6e}`.",
        f"5. cross_to_same_ratio after the fix: `{geom_row['cross_to_same_ratio']:.6e}`.",
    ]
    if baseline_rows:
        chosen = [row for row in baseline_rows if row["lr"] == selected_lr]
        sgd = next(row for row in chosen if row["method"] == "sgd")
        egm = next(row for row in chosen if row["method"] == "egm")
        ppm = next(row for row in chosen if row["method"] == "ppm")
        lines.extend([
            f"6. Does EGM outperform SGD under identical lr/config? `{egm['auc_V_lambda'] < sgd['auc_V_lambda']}`.",
            f"7. Does PPM outperform SGD under identical lr/config? `{ppm['auc_V_lambda'] < sgd['auc_V_lambda']}`.",
        ])
    else:
        lines.extend([
            "6. EGM/PPM were not meaningfully compared against SGD because geometry failed before baseline gate.",
            "7. The reason is still structural geometry weakness, not proposed tuning.",
        ])
    lines.extend([
        "8. Proposed was not run in this round.",
        "9. PPO health remained within the configured basic safety checks in the completed runs.",
        f"10. Final decision: `{decision}`.",
    ])
    if decision == "COUPLED_PENDULUM_READY_FOR_PROPOSED":
        lines.append("11. Pendulum is ready for proposed methods in the next round.")
    elif decision == "COUPLED_PENDULUM_BASELINE_FAIL":
        lines.append("11. The field is now cross-coupled enough to test, but EGM/PPM still did not reliably beat SGD under identical lr/config.")
    else:
        lines.append("11. The implementation still does not produce enough usable rotational/cross-player structure for a meaningful proposed comparison.")
    return "\n".join(lines) + "\n"


def ensure_dirs() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)


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


def main() -> None:
    ensure_dirs()
    write_existing_root_cause_read()
    game = CoupledJointPPOPendulum()
    base_cfg, _ = load_existing_coupled_ready_state()
    chosen_cfg, search_summary_rows, _, chosen_bundle = run_baseline_stabilization_search(game, base_cfg)
    write_stabilization_grid(search_summary_rows, chosen_bundle)
    _, _, strict_summary_rows, strict_curve_rows, geom_row, _, _ = run_final_strict_gate(game, chosen_cfg)
    decision, selected_lr, reason = strict_gate_decision(strict_summary_rows, geom_row)
    write_stabilized_geometry(geom_row)
    write_v_component_audit(strict_curve_rows, selected_lr if selected_lr > 0.0 else STRICT_SEARCH_LR)
    write_strict_gate_report(decision, reason, strict_summary_rows, selected_lr)
    write_baseline_failure_root_cause(decision, chosen_cfg, geom_row, strict_summary_rows)
    if decision == "STRICT_BASELINE_PASS":
        make_strict_baseline_plots(strict_curve_rows, selected_lr)
    write_ready_or_not(decision, chosen_cfg, selected_lr if selected_lr > 0.0 else STRICT_SEARCH_LR)


if __name__ == "__main__":
    main()
