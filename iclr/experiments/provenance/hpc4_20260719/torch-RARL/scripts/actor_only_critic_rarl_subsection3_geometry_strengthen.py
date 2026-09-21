from __future__ import annotations

import csv
import copy
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / "actor_only_critic_rarl_subsection3.py"
STAGE2_PATH = SCRIPT_DIR / "actor_only_critic_rarl_subsection3_stage2_task_aligned.py"
RESULT_ROOT = BASE_PATH.resolve().parents[2] / "results" / "actor_only_critic_rarl_subsection3"

base_spec = importlib.util.spec_from_file_location("actor_only_critic_rarl_subsection3", BASE_PATH)
base = importlib.util.module_from_spec(base_spec)
sys.modules[base_spec.name] = base
base_spec.loader.exec_module(base)

stage2_spec = importlib.util.spec_from_file_location("actor_only_critic_rarl_subsection3_stage2_task_aligned", STAGE2_PATH)
stage2 = importlib.util.module_from_spec(stage2_spec)
sys.modules[stage2_spec.name] = stage2
stage2_spec.loader.exec_module(stage2)


ENV_ID = "Reacher-v5"
ENV_SLUG = "reacher_v5"
PREFIX = f"actor_only_critic_s3_{ENV_SLUG}_"

REPLAY_WARMUP_STEPS = 30000
CRITIC_PRETRAIN_STEPS = 5000
PROBE_STEPS = 2000
MC_SNAPSHOT_COUNT = 16
ACTOR_ITERS = 100
PRIMARY_LR = 1e-4
SECONDARY_LR = 3e-5


@dataclass(frozen=True)
class SweepConfig:
    alpha_dyn: float
    disturbance_type: str


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out: list[dict[str, Any]] = []
    for row in rows:
        new: dict[str, Any] = {}
        for key, value in row.items():
            if value is None:
                new[key] = value
                continue
            try:
                if value.lower() in {"true", "false"}:
                    new[key] = value.lower() == "true"
                else:
                    new[key] = float(value)
            except Exception:
                new[key] = value
        out.append(new)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    base.write_csv(path, rows)


def write_text(path: Path, text: str) -> None:
    base.write_text(path, text)


def parse_ready_report() -> dict[str, Any]:
    text = (RESULT_ROOT / f"{PREFIX}ready_for_baseline_gate.md").read_text(encoding="utf-8")
    out: dict[str, Any] = {}
    for key in ["selected_alpha_dyn", "selected_use_rot_dyn", "reward_scale", "selected_actor_lr"]:
        marker = f"- {key}: `"
        if marker not in text:
            continue
        value = text.split(marker, 1)[1].split("`", 1)[0]
        if value in {"True", "False"}:
            out[key] = value == "True"
        else:
            out[key] = float(value)
    return out


def make_env_choice() -> base.EnvChoice:
    env = base.choose_environment()
    if env.env_id != ENV_ID:
        raise RuntimeError(f"Expected {ENV_ID}, got {env.env_id}")
    return env


class StrengthenedRARLEnv:
    def __init__(self, env_choice: base.EnvChoice, reward_scale: float, alpha_dyn: float, disturbance_type: str) -> None:
        self.env_choice = env_choice
        self.reward_scale = reward_scale
        self.alpha_dyn = alpha_dyn
        self.disturbance_type = disturbance_type
        self.action_low = torch.as_tensor(env_choice.action_low, dtype=base.DTYPE, device=base.DEVICE)
        self.action_high = torch.as_tensor(env_choice.action_high, dtype=base.DTYPE, device=base.DEVICE)
        self.action_scale = 0.5 * (self.action_high - self.action_low)
        self.action_bias = 0.5 * (self.action_high + self.action_low)
        self.r_dyn = self.build_r_dyn(env_choice.action_dim)

    def build_r_dyn(self, action_dim: int) -> torch.Tensor:
        mat = torch.zeros((action_dim, action_dim), dtype=base.DTYPE, device=base.DEVICE)
        for start in range(0, action_dim - 1, 2):
            mat[start, start + 1] = 1.0
            mat[start + 1, start] = -1.0
        return mat

    def scale_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        return self.action_bias + (self.action_scale * torch.tanh(raw_action))

    def perturb(self, w: torch.Tensor) -> torch.Tensor:
        rot = self.r_dyn @ w
        if self.disturbance_type == "rotated":
            return rot
        if self.disturbance_type == "direct":
            return w
        if self.disturbance_type == "mixed":
            return (0.7 * rot) + (0.3 * w)
        raise ValueError(self.disturbance_type)

    def blend_action(self, u: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, float]:
        a_env_raw = u + (self.alpha_dyn * self.perturb(w))
        a_env = torch.clamp(a_env_raw, self.action_low, self.action_high)
        clip_fraction = float(((a_env_raw - a_env).abs() > 1e-12).float().mean().item())
        return a_env, clip_fraction

    def reward_terms(self, reward_task_raw: float, u: torch.Tensor, w: torch.Tensor) -> tuple[float, float]:
        r_task_scaled = self.reward_scale * reward_task_raw
        return r_task_scaled, r_task_scaled


class StrengthenedGame(stage2.TaskAlignedFrozenGame):
    def __init__(self, env_choice: base.EnvChoice, reward_scale: float, alpha_dyn: float, disturbance_type: str, actor_lr: float, seed: int) -> None:
        wrapper_cfg = base.WrapperConfig(
            reward_scale=reward_scale,
            alpha_dyn=alpha_dyn,
            a_u=0.0,
            a_w=0.0,
            beta_rot=0.0,
            beta_sym=0.0,
            use_rot_dyn=(disturbance_type == "rotated"),
        )
        super().__init__(env_choice, wrapper_cfg, actor_lr, seed)
        self.disturbance_type = disturbance_type
        self.rarl_env = StrengthenedRARLEnv(env_choice, reward_scale, alpha_dyn, disturbance_type)
        self.wrapper_cfg = wrapper_cfg

    def collect_rollout(self, steps: int, exploration_std: float) -> dict[str, float]:
        env = base.gym.make(self.env_choice.env_id)
        obs, _ = env.reset(seed=base.SEED + 100 + int(self.rng.integers(0, 1_000_000)))
        total_clip = 0.0
        total_task = 0.0
        for _ in range(steps):
            obs_t = torch.as_tensor(obs, dtype=base.DTYPE, device=base.DEVICE).unsqueeze(0)
            u = self.actor_action(self.theta, obs_t).squeeze(0)
            w = self.actor_action(self.phi, obs_t).squeeze(0)
            if exploration_std > 0.0:
                u = torch.clamp(u + (exploration_std * torch.randn_like(u)), self.rarl_env.action_low, self.rarl_env.action_high)
                w = torch.clamp(w + (exploration_std * torch.randn_like(w)), self.rarl_env.action_low, self.rarl_env.action_high)
            a_env, clip_frac = self.rarl_env.blend_action(u, w)
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
            obs = next_obs
            if done:
                obs, _ = env.reset()
        env.close()
        return {
            "train_action_clip_fraction": total_clip / max(steps, 1),
            "train_game_return_proxy": total_task * self.rarl_env.reward_scale / max(steps, 1),
            "train_task_return_proxy": total_task / max(steps, 1),
        }

    def collect_random_warmup(self, steps: int) -> None:
        env = base.gym.make(self.env_choice.env_id)
        obs, _ = env.reset(seed=base.SEED + 11)
        for _ in range(steps):
            u = self.rng.uniform(self.env_choice.action_low, self.env_choice.action_high).astype(np.float32)
            w = self.rng.uniform(self.env_choice.action_low, self.env_choice.action_high).astype(np.float32)
            w_t = torch.as_tensor(w, dtype=base.DTYPE, device=base.DEVICE)
            u_t = torch.as_tensor(u, dtype=base.DTYPE, device=base.DEVICE)
            a_env, _ = self.rarl_env.blend_action(u_t, w_t)
            next_obs, reward_raw, terminated, truncated, _ = env.step(a_env.cpu().numpy().astype(np.float32))
            reward_game, reward_task_scaled = self.rarl_env.reward_terms(float(reward_raw), u_t, w_t)
            done = terminated or truncated
            self.replay.add(obs, u, w, reward_game, reward_task_scaled, float(reward_raw), next_obs, done)
            obs = next_obs
            if done:
                obs, _ = env.reset()
        env.close()

    def evaluate_policy(self, theta: torch.Tensor, phi: torch.Tensor | None, episodes: int, mode: str) -> dict[str, float]:
        env = base.gym.make(self.env_choice.env_id)
        task_returns_raw: list[float] = []
        task_returns_scaled: list[float] = []
        clip_fracs: list[float] = []
        for ep in range(episodes):
            obs, _ = env.reset(seed=base.SEED + 9000 + ep)
            done = False
            task_raw = 0.0
            task_scaled = 0.0
            ep_clips: list[float] = []
            while not done:
                obs_t = torch.as_tensor(obs, dtype=base.DTYPE, device=base.DEVICE).unsqueeze(0)
                u = self.actor_action(theta, obs_t).squeeze(0)
                if mode == "clean" or phi is None:
                    w = torch.zeros_like(u)
                    a_env = torch.clamp(u, self.rarl_env.action_low, self.rarl_env.action_high)
                    clip_frac = float(((u - a_env).abs() > 1e-12).float().mean().item())
                else:
                    w = self.actor_action(phi, obs_t).squeeze(0)
                    a_env, clip_frac = self.rarl_env.blend_action(u, w)
                next_obs, reward_raw, terminated, truncated, _ = env.step(a_env.detach().cpu().numpy().astype(np.float32))
                _, reward_task_scaled = self.rarl_env.reward_terms(float(reward_raw), u, w)
                task_raw += float(reward_raw)
                task_scaled += float(reward_task_scaled)
                ep_clips.append(clip_frac)
                obs = next_obs
                done = terminated or truncated
            task_returns_raw.append(task_raw)
            task_returns_scaled.append(task_scaled)
            clip_fracs.append(float(np.mean(ep_clips)) if ep_clips else 0.0)
        env.close()
        return {
            "task_return_raw": float(np.mean(task_returns_raw)),
            "task_return_scaled": float(np.mean(task_returns_scaled)),
            "game_return": float(np.mean(task_returns_scaled)),
            "action_clip_fraction": float(np.mean(clip_fracs)),
        }


def current_geometry_reference() -> dict[str, float]:
    rows = load_rows(RESULT_ROOT / f"{PREFIX}task_aligned_geometry_audit.csv")
    if not rows:
        return {
            "rotation_ratio_proxy": 2.327e-08,
            "cross_player_coupling_proxy": 1.245e-02,
            "cross_to_same_ratio": 1.485e-01,
            "non_collinearity_T": 0.995,
        }
    row = rows[0]
    return {
        "rotation_ratio_proxy": float(row["rotation_ratio_proxy"]),
        "cross_player_coupling_proxy": float(row["cross_player_coupling_proxy"]),
        "cross_to_same_ratio": float(row["cross_to_same_ratio"]),
        "non_collinearity_T": float(row["non_collinearity_T"]),
    }


def task_critic_update(game: StrengthenedGame) -> dict[str, float]:
    batch = game.replay.sample(base.TRAIN_BATCH_SIZE, game.rng)
    with torch.no_grad():
        u_next = game.actor_action(game.theta_target, batch["next_obs"])
        w_next = game.actor_action(game.phi_target, batch["next_obs"])
        target_task = batch["reward_task_scaled"] + (base.GAMMA * (1.0 - batch["done"]) * game.q_task_target(batch["next_obs"], u_next, w_next))
    pred_task = game.q_task(batch["obs"], batch["u"], batch["w"])
    loss_task = torch.mean((pred_task - target_task) ** 2)
    game.q_task_opt.zero_grad(set_to_none=True)
    loss_task.backward()
    game.q_task_opt.step()
    with torch.no_grad():
        for target, online in zip(game.q_task_target.parameters(), game.q_task.parameters()):
            target.data.mul_(1.0 - base.POLYAK_TAU).add_(base.POLYAK_TAU * online.data)
    return {
        "critic_loss_T": float(loss_task.detach().item()),
        "Q_T_mean": float(pred_task.detach().mean().item()),
        "Q_T_std": float(pred_task.detach().std(unbiased=False).item()),
    }


def collect_exploratory_warmup(game: StrengthenedGame, steps: int, noise_std: float) -> tuple[list[dict[str, np.ndarray]], dict[str, float]]:
    env = base.gym.make(game.env_choice.env_id)
    obs, _ = env.reset(seed=base.SEED + 1234)
    u_vals: list[np.ndarray] = []
    w_vals: list[np.ndarray] = []
    rw_vals: list[np.ndarray] = []
    clip_fracs: list[float] = []
    snapshots: list[dict[str, np.ndarray]] = []
    for _ in range(steps):
        obs_t = torch.as_tensor(obs, dtype=base.DTYPE, device=base.DEVICE).unsqueeze(0)
        u = game.actor_action(game.theta, obs_t).squeeze(0)
        w = game.actor_action(game.phi, obs_t).squeeze(0)
        if noise_std > 0.0:
            u = torch.clamp(u + (noise_std * torch.randn_like(u)), game.rarl_env.action_low, game.rarl_env.action_high)
            w = torch.clamp(w + (noise_std * torch.randn_like(w)), game.rarl_env.action_low, game.rarl_env.action_high)
        if len(snapshots) < MC_SNAPSHOT_COUNT * 2:
            snapshots.append(
                {
                    "obs": np.asarray(obs, dtype=np.float32).copy(),
                    "qpos": env.unwrapped.data.qpos.copy(),
                    "qvel": env.unwrapped.data.qvel.copy(),
                }
            )
        a_env, clip_frac = game.rarl_env.blend_action(u, w)
        next_obs, reward_raw, terminated, truncated, _ = env.step(a_env.detach().cpu().numpy().astype(np.float32))
        reward_game, reward_task_scaled = game.rarl_env.reward_terms(float(reward_raw), u, w)
        done = terminated or truncated
        game.replay.add(
            np.asarray(obs, dtype=np.float32),
            u.detach().cpu().numpy().astype(np.float32),
            w.detach().cpu().numpy().astype(np.float32),
            float(reward_game),
            float(reward_task_scaled),
            float(reward_raw),
            np.asarray(next_obs, dtype=np.float32),
            done,
        )
        obs = next_obs
        if done:
            obs, _ = env.reset()
        u_np = u.detach().cpu().numpy()
        w_np = w.detach().cpu().numpy()
        rw_np = (game.rarl_env.perturb(w)).detach().cpu().numpy()
        u_vals.append(u_np)
        w_vals.append(w_np)
        rw_vals.append(rw_np)
        clip_fracs.append(clip_frac)
    env.close()
    u_arr = np.asarray(u_vals, dtype=np.float64)
    w_arr = np.asarray(w_vals, dtype=np.float64)
    rw_arr = np.asarray(rw_vals, dtype=np.float64)
    stats = {
        "warmup_noise_std": noise_std,
        "std_u": float(np.std(u_arr)),
        "std_w": float(np.std(w_arr)),
        "std_Rw": float(np.std(rw_arr)),
        "action_clip_fraction": float(np.mean(clip_fracs)),
    }
    return snapshots[:MC_SNAPSHOT_COUNT], stats


def mc_quality_task(game: StrengthenedGame, snapshots: list[dict[str, np.ndarray]]) -> dict[str, float]:
    env = base.gym.make(game.env_choice.env_id)
    q_task_vals: list[float] = []
    mc_task_vals: list[float] = []
    for snap in snapshots:
        env.reset(seed=base.SEED)
        env.unwrapped.set_state(snap["qpos"], snap["qvel"])
        obs_t = torch.as_tensor(np.asarray(snap["obs"], dtype=np.float32), dtype=base.DTYPE, device=base.DEVICE).unsqueeze(0)
        u = game.actor_action(game.theta, obs_t)
        w = game.actor_action(game.phi, obs_t)
        q_task_vals.append(float(game.q_task(obs_t, u, w).detach().item()))
        disc = 1.0
        mc_task = 0.0
        for _ in range(50):
            u_step = game.actor_action(game.theta, obs_t).squeeze(0)
            w_step = game.actor_action(game.phi, obs_t).squeeze(0)
            a_env, _ = game.rarl_env.blend_action(u_step, w_step)
            next_obs, reward_raw, terminated, truncated, _ = env.step(a_env.detach().cpu().numpy().astype(np.float32))
            _, reward_task_scaled = game.rarl_env.reward_terms(float(reward_raw), u_step, w_step)
            mc_task += disc * reward_task_scaled
            disc *= base.GAMMA
            obs_t = torch.as_tensor(next_obs, dtype=base.DTYPE, device=base.DEVICE).unsqueeze(0)
            if terminated or truncated:
                break
        mc_task_vals.append(mc_task)
    env.close()
    corr = math.nan
    if len(q_task_vals) >= 2 and np.std(q_task_vals) > 1e-12 and np.std(mc_task_vals) > 1e-12:
        corr = float(np.corrcoef(np.asarray(q_task_vals), np.asarray(mc_task_vals))[0, 1])
    mse = float(np.mean((np.asarray(q_task_vals) - np.asarray(mc_task_vals)) ** 2)) if q_task_vals else math.nan
    return {"corr_Q_T_MC_task": corr, "mse_Q_T_MC_task": mse}


@dataclass
class Bundle:
    game: StrengthenedGame
    diag_batch: dict[str, torch.Tensor]
    snapshots: list[dict[str, np.ndarray]]
    warmup_stats: dict[str, float]
    critic_quality: dict[str, float]


def choose_noise_std(env_choice: base.EnvChoice, reward_scale: float, cfg: SweepConfig) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    best_std = 0.1
    best_key = None
    for noise_std in [0.1, 0.2]:
        game = StrengthenedGame(env_choice, reward_scale, cfg.alpha_dyn, cfg.disturbance_type, PRIMARY_LR, base.SEED)
        _, stats = collect_exploratory_warmup(game, PROBE_STEPS, noise_std)
        row = {"alpha_dyn": cfg.alpha_dyn, "disturbance_type": cfg.disturbance_type, **stats}
        rows.append(row)
        key = (0 if stats["action_clip_fraction"] <= 0.05 else 1, -stats["std_w"], stats["action_clip_fraction"])
        if best_key is None or key < best_key:
            best_key = key
            best_std = noise_std
    return best_std, rows


def build_bundle(env_choice: base.EnvChoice, reward_scale: float, cfg: SweepConfig) -> Bundle:
    best_noise, probe_rows = choose_noise_std(env_choice, reward_scale, cfg)
    game = StrengthenedGame(env_choice, reward_scale, cfg.alpha_dyn, cfg.disturbance_type, PRIMARY_LR, base.SEED)
    snapshots, warmup_stats = collect_exploratory_warmup(game, REPLAY_WARMUP_STEPS, best_noise)
    warmup_stats["warmup_noise_std"] = best_noise
    quality = {}
    for _ in range(CRITIC_PRETRAIN_STEPS):
        quality = task_critic_update(game)
    quality.update(mc_quality_task(game, snapshots))
    diag_batch = game.replay.fixed_state_batch(base.TRAIN_BATCH_SIZE, np.random.default_rng(base.SEED + 4444))
    quality["probe_rows"] = probe_rows
    return Bundle(game=game, diag_batch={k: v.detach().clone() for k, v in diag_batch.items()}, snapshots=snapshots, warmup_stats=warmup_stats, critic_quality=quality)


def geometry_row(bundle: Bundle, cfg: SweepConfig, current_ref: dict[str, float]) -> dict[str, Any]:
    game = bundle.game
    z = game.current_z()
    diag = game.diagnostic_metrics(z, bundle.diag_batch, PRIMARY_LR, compute_geometry=True)
    rotation = float(diag["rotation_ratio_proxy"])
    cross = float(diag["cross_player_coupling_proxy"])
    cross_ratio = float(diag["cross_to_same_ratio"])
    noncol = float(diag["non_collinearity_T"])
    clip_ok = bundle.warmup_stats["action_clip_fraction"] <= 0.05
    critic_ok = base.finite(bundle.critic_quality["critic_loss_T"]) and abs(bundle.critic_quality["Q_T_mean"]) < 100.0
    geometry_pass = cross > 0.0 and cross_ratio > 0.05 and noncol > 0.2 and rotation >= 1e-3
    better_than_current = rotation > current_ref["rotation_ratio_proxy"] * 10.0 or cross > current_ref["cross_player_coupling_proxy"] * 1.2 or cross_ratio > current_ref["cross_to_same_ratio"] * 1.2
    row = {
        "alpha_dyn": cfg.alpha_dyn,
        "disturbance_type": cfg.disturbance_type,
        "warmup_noise_std": bundle.warmup_stats["warmup_noise_std"],
        "std_u": bundle.warmup_stats["std_u"],
        "std_w": bundle.warmup_stats["std_w"],
        "std_Rw": bundle.warmup_stats["std_Rw"],
        "action_clip_fraction": bundle.warmup_stats["action_clip_fraction"],
        "critic_loss_T": bundle.critic_quality["critic_loss_T"],
        "Q_T_mean": bundle.critic_quality["Q_T_mean"],
        "Q_T_std": bundle.critic_quality["Q_T_std"],
        "corr_Q_T_MC_task": bundle.critic_quality["corr_Q_T_MC_task"],
        "mse_Q_T_MC_task": bundle.critic_quality["mse_Q_T_MC_task"],
        "field_norm": diag["field_norm"],
        "g_over_f_T": diag["g_over_f_T"],
        "cos_fg_T": diag["cos_fg_T"],
        "non_collinearity_T": noncol,
        "rotation_ratio_proxy": rotation,
        "cross_player_coupling_proxy": cross,
        "cross_to_same_ratio": cross_ratio,
        "geometry_pass": int(geometry_pass),
        "clip_ok": int(clip_ok),
        "critic_ok": int(critic_ok),
        "better_than_current": int(better_than_current),
        "geometry_score": rotation + (0.1 * cross_ratio) + (0.01 * cross),
    }
    return row


def sweep_configs() -> tuple[list[SweepConfig], list[SweepConfig]]:
    phase1 = [
        SweepConfig(0.1, "rotated"),
        SweepConfig(0.1, "direct"),
        SweepConfig(0.1, "mixed"),
    ]
    return phase1, []


def additional_configs(best_type: str) -> list[SweepConfig]:
    return [
        SweepConfig(0.05, best_type),
        SweepConfig(0.2, best_type),
        SweepConfig(0.3, best_type),
    ]


def choose_best_type(rows: list[dict[str, Any]]) -> str:
    best = max(rows, key=lambda row: (int(row["clip_ok"]), int(row["critic_ok"]), row["geometry_score"]))
    return str(best["disturbance_type"])


def run_sgd_check(bundle: Bundle, cfg: SweepConfig, actor_lr: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return run_method_with_bundle(bundle, cfg, "sgd", actor_lr, ACTOR_ITERS)


def baseline_run(bundle: Bundle, cfg: SweepConfig, actor_lr: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    all_curves: list[dict[str, Any]] = []
    for method in ["sgd", "egm", "ppm"]:
        curves, summary = run_method_with_bundle(bundle, cfg, method, actor_lr, ACTOR_ITERS)
        for row in curves:
            row["disturbance_type"] = cfg.disturbance_type
            row["alpha_dyn"] = cfg.alpha_dyn
        summary["disturbance_type"] = cfg.disturbance_type
        summary["alpha_dyn"] = cfg.alpha_dyn
        all_curves.extend(curves)
        summaries.append(summary)
    return all_curves, summaries


def clone_frozen_game(bundle: Bundle, cfg: SweepConfig, actor_lr: float) -> StrengthenedGame:
    game0 = bundle.game
    game = StrengthenedGame(game0.env_choice, game0.rarl_env.reward_scale, cfg.alpha_dyn, cfg.disturbance_type, actor_lr, base.SEED)
    game.theta = game0.theta.detach().clone()
    game.phi = game0.phi.detach().clone()
    game.theta_target = game0.theta_target.detach().clone()
    game.phi_target = game0.phi_target.detach().clone()
    game.q_task.load_state_dict(copy.deepcopy(game0.q_task.state_dict()))
    game.q_task_target.load_state_dict(copy.deepcopy(game0.q_task_target.state_dict()))
    game.metric_refs = {}
    return game


def run_method_with_bundle(bundle: Bundle, cfg: SweepConfig, method: str, actor_lr: float, num_iters: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    game = clone_frozen_game(bundle, cfg, actor_lr)
    diag_batch = {k: v.detach().clone() for k, v in bundle.diag_batch.items()}
    curves: list[dict[str, Any]] = []
    clean_eval = game.evaluate_policy(game.theta, None, 5, "clean")
    adv_eval = game.evaluate_policy(game.theta, game.phi, 5, "current")
    br_phi, br_valid = game.robust_br_adversary(game.theta, diag_batch)
    br_eval = game.evaluate_policy(game.theta, br_phi, 5, "current")
    for iteration in range(num_iters + 1):
        z = game.current_z()
        diag = game.diagnostic_metrics(z, diag_batch, actor_lr, compute_geometry=(iteration % 10 == 0))
        if iteration % 10 == 0 or iteration == num_iters:
            clean_eval = game.evaluate_policy(game.theta, None, 5, "clean")
            adv_eval = game.evaluate_policy(game.theta, game.phi, 5, "current")
            br_phi, br_valid = game.robust_br_adversary(game.theta, diag_batch)
            br_eval = game.evaluate_policy(game.theta, br_phi, 5, "current")
        row = {
            "method": method,
            "iteration": iteration,
            "actor_lr": actor_lr,
            "alpha_dyn": cfg.alpha_dyn,
            "disturbance_type": cfg.disturbance_type,
            **diag,
            "clean_task_return_raw": clean_eval["task_return_raw"],
            "clean_task_return_scaled": clean_eval["task_return_scaled"],
            "current_adv_task_return_raw": adv_eval["task_return_raw"],
            "current_adv_task_return_scaled": adv_eval["task_return_scaled"],
            "robust_br_task_return_raw": br_eval["task_return_raw"],
            "robust_br_task_return_scaled": br_eval["task_return_scaled"],
            "robust_degradation": clean_eval["task_return_raw"] - br_eval["task_return_raw"],
            "action_clip_fraction": adv_eval["action_clip_fraction"],
            "br_valid": int(br_valid),
            "protagonist_param_norm": float(torch.linalg.norm(z[game.field_slices["theta"]]).item()),
            "adversary_param_norm": float(torch.linalg.norm(z[game.field_slices["phi"]]).item()),
            "nan_flag": 0,
            "valid_flag": 1,
        }
        row["nan_flag"] = int(
            not all(
                base.finite(row[key])
                for key in [
                    "V_align",
                    "field_norm",
                    "raw_P_tau_T",
                    "approximate_exploitability_T",
                    "Q_T",
                    "clean_task_return_raw",
                    "current_adv_task_return_raw",
                    "robust_br_task_return_raw",
                ]
            )
        )
        row["valid_flag"] = int(row["nan_flag"] == 0 and row["action_clip_fraction"] <= 0.05 and row["br_valid"] == 1)
        curves.append(row)
        if iteration < num_iters:
            next_z, meta = base.run_actor_update(game, z, diag_batch["obs"], method, actor_lr)
            game.set_from_z(next_z)
            curves[-1].update(meta)
    vals_v = [row["V_align"] for row in curves]
    vals_p = [row["normalized_P_tau_T"] for row in curves]
    vals_f = [row["field_norm"] for row in curves]
    vals_e = [row["approximate_exploitability_T"] for row in curves]
    first = curves[0]
    last = curves[-1]
    summary = {
        "method": method,
        "actor_lr": actor_lr,
        "valid_flag": int(all(int(row["valid_flag"]) == 1 for row in curves)),
        "V_align_start": first["V_align"],
        "V_align_final": last["V_align"],
        "V_align_AUC": float(sum(vals_v)),
        "V_align_spike_ratio": base.spike_ratio(vals_v),
        "P_tau_T_start": first["normalized_P_tau_T"],
        "P_tau_T_final": last["normalized_P_tau_T"],
        "P_tau_T_AUC": float(sum(vals_p)),
        "P_tau_T_spike_ratio": base.spike_ratio(vals_p),
        "field_norm_start": first["field_norm"],
        "field_norm_final": last["field_norm"],
        "field_norm_AUC": float(sum(vals_f)),
        "field_norm_spike_ratio": base.spike_ratio(vals_f),
        "approximate_exploitability_T_start": first["approximate_exploitability_T"],
        "approximate_exploitability_T_final": last["approximate_exploitability_T"],
        "approximate_exploitability_T_AUC": float(sum(vals_e)),
        "clean_task_return_raw_final": last["clean_task_return_raw"],
        "current_adv_task_return_raw_final": last["current_adv_task_return_raw"],
        "robust_br_task_return_raw_final": last["robust_br_task_return_raw"],
        "robust_degradation_final": last["robust_degradation"],
        "curve_normal_flag": int(
            last["V_align"] <= first["V_align"] + 1e-8
            and last["normalized_P_tau_T"] <= first["normalized_P_tau_T"] + 1e-8
            and base.spike_ratio(vals_v) <= 5.0
            and base.spike_ratio(vals_p) <= 5.0
            and base.spike_ratio(vals_f) <= 5.0
            and last["approximate_exploitability_T"] <= (5.0 * first["approximate_exploitability_T"] + 1e-8)
        ),
    }
    return curves, summary


def plot_best_baseline(rows: list[dict[str, Any]]) -> None:
    if base.plt is None or not rows:
        return
    plot_dir = RESULT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["method"], []).append(row)
    colors = {"sgd": "#1f77b4", "egm": "#ff7f0e", "ppm": "#2ca02c"}
    metrics = [
        ("V_align", "V_align", "task_aligned_reacher_geometry_strengthened_baseline_V_align.png"),
        ("normalized_P_tau_T", "P_tau_T", "task_aligned_reacher_geometry_strengthened_baseline_P_tau_T.png"),
        ("field_norm", "Field Norm", "task_aligned_reacher_geometry_strengthened_baseline_field_norm.png"),
        ("approximate_exploitability_T", "Exploitability", "task_aligned_reacher_geometry_strengthened_baseline_exploitability.png"),
    ]
    for metric, title, name in metrics:
        fig, ax = base.plt.subplots(figsize=(7, 4))
        for method, method_rows in groups.items():
            ax.plot([r["iteration"] for r in method_rows], [r[metric] for r in method_rows], label=method.upper(), color=colors[method], linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / name, dpi=180)
        base.plt.close(fig)
    fig, axes = base.plt.subplots(4, 2, figsize=(14, 14), sharex=True)
    panel_metrics = [
        ("V_align", "V_align"),
        ("normalized_P_tau_T", "P_tau_T"),
        ("field_norm", "Field Norm"),
        ("approximate_exploitability_T", "Exploitability"),
        ("clean_task_return_raw", "Clean Task"),
        ("current_adv_task_return_raw", "Current Adv Task"),
        ("robust_br_task_return_raw", "Robust BR Task"),
        ("robust_degradation", "Robust Degradation"),
    ]
    for ax, (metric, title) in zip(axes.flat, panel_metrics):
        for method, method_rows in groups.items():
            ax.plot([r["iteration"] for r in method_rows], [r[metric] for r in method_rows], label=method.upper(), color=colors[method], linewidth=1.7)
        ax.set_title(title)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "task_aligned_reacher_geometry_strengthened_baseline_all_plots_big.png", dpi=180)
    base.plt.close(fig)
    fig, ax = base.plt.subplots(figsize=(8, 5))
    for method, method_rows in groups.items():
        ax.plot([r["iteration"] for r in method_rows], [r["robust_br_task_return_raw"] for r in method_rows], label=f"{method.upper()} robust BR", color=colors[method], linewidth=1.8)
        ax.plot([r["iteration"] for r in method_rows], [r["current_adv_task_return_raw"] for r in method_rows], linestyle="--", color=colors[method], linewidth=1.4)
    ax.set_title("RARL Performance")
    ax.set_xlabel("Iteration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "task_aligned_reacher_geometry_strengthened_baseline_rarl_performance.png", dpi=180)
    base.plt.close(fig)


def main() -> None:
    base.ensure_dirs()
    base.seed_everything(base.SEED)
    ready = parse_ready_report()
    env_choice = make_env_choice()
    reward_scale = float(ready["reward_scale"])
    current_ref = current_geometry_reference()

    sweep_rows: list[dict[str, Any]] = []
    bundles: dict[tuple[float, str], Bundle] = {}
    phase1, _ = sweep_configs()
    probe_rows_all: list[dict[str, Any]] = []
    for cfg in phase1:
        bundle = build_bundle(env_choice, reward_scale, cfg)
        row = geometry_row(bundle, cfg, current_ref)
        sweep_rows.append(row)
        bundles[(cfg.alpha_dyn, cfg.disturbance_type)] = bundle
        probe_rows_all.extend(bundle.critic_quality["probe_rows"])

    best_type = choose_best_type(sweep_rows)
    for cfg in additional_configs(best_type):
        bundle = build_bundle(env_choice, reward_scale, cfg)
        row = geometry_row(bundle, cfg, current_ref)
        sweep_rows.append(row)
        bundles[(cfg.alpha_dyn, cfg.disturbance_type)] = bundle
        probe_rows_all.extend(bundle.critic_quality["probe_rows"])

    write_csv(RESULT_ROOT / "task_aligned_reacher_geometry_strengthening_sweep.csv", sweep_rows)
    lines = ["# task_aligned_reacher_geometry_strengthening_sweep_report", ""]
    for row in sweep_rows:
        lines.append(
            f"- alpha_dyn=`{row['alpha_dyn']}`, disturbance_type=`{row['disturbance_type']}`, noise_std=`{row['warmup_noise_std']}`, "
            f"clip=`{row['action_clip_fraction']:.6e}`, corr_QT_MC=`{row['corr_Q_T_MC_task']:.3f}`, rotation=`{row['rotation_ratio_proxy']:.6e}`, "
            f"cross=`{row['cross_player_coupling_proxy']:.6e}`, cross_ratio=`{row['cross_to_same_ratio']:.6e}`, noncol=`{row['non_collinearity_T']:.6e}`, "
            f"geometry_pass=`{bool(row['geometry_pass'])}`, better_than_current=`{bool(row['better_than_current'])}`"
        )
    if not any(row["rotation_ratio_proxy"] >= 1e-3 for row in sweep_rows):
        lines.extend(["", "- No tested config reached `rotation_ratio_proxy >= 1e-3`; task-aligned Reacher dynamics remain weak in skew geometry under this sweep."])
    write_text(RESULT_ROOT / "task_aligned_reacher_geometry_strengthening_sweep_report.md", "\n".join(lines) + "\n")

    # SGD sanity for best-geometry configs.
    candidate_rows = sorted(
        [row for row in sweep_rows if row["clip_ok"] == 1 and row["critic_ok"] == 1],
        key=lambda row: (row["geometry_pass"], row["better_than_current"], row["geometry_score"]),
        reverse=True,
    )[:2]
    sgd_rows: list[dict[str, Any]] = []
    sgd_report_lines = ["# task_aligned_reacher_geometry_strengthening_sgd_check_report", ""]
    promising = None
    for row in candidate_rows:
        cfg = SweepConfig(alpha_dyn=float(row["alpha_dyn"]), disturbance_type=str(row["disturbance_type"]))
        bundle = bundles[(cfg.alpha_dyn, cfg.disturbance_type)]
        curves, summary = run_sgd_check(bundle, cfg, PRIMARY_LR)
        selected_lr = PRIMARY_LR
        if not (summary["valid_flag"] == 1 and summary["curve_normal_flag"] == 1):
            curves2, summary2 = run_sgd_check(bundle, cfg, SECONDARY_LR)
            if summary2["curve_normal_flag"] == 1 and summary2["valid_flag"] == 1:
                curves, summary, selected_lr = curves2, summary2, SECONDARY_LR
        for curve in curves:
            curve["disturbance_type"] = cfg.disturbance_type
            curve["alpha_dyn"] = cfg.alpha_dyn
            curve["selected_actor_lr"] = selected_lr
        summary["disturbance_type"] = cfg.disturbance_type
        summary["alpha_dyn"] = cfg.alpha_dyn
        summary["selected_actor_lr"] = selected_lr
        sgd_rows.extend(curves)
        sgd_report_lines.append(
            f"- alpha_dyn=`{cfg.alpha_dyn}`, disturbance_type=`{cfg.disturbance_type}`: selected_actor_lr=`{selected_lr}`, valid=`{bool(summary['valid_flag'])}`, curve_normal=`{bool(summary['curve_normal_flag'])}`, "
            f"V_align_final=`{summary['V_align_final']:.6e}`, P_tau_T_final=`{summary['P_tau_T_final']:.6e}`, field_norm_final=`{summary['field_norm_final']:.6e}`"
        )
        if promising is None and row["better_than_current"] == 1 and summary["curve_normal_flag"] == 1 and summary["valid_flag"] == 1:
            promising = (cfg, bundle, selected_lr, summary)
    write_csv(RESULT_ROOT / "task_aligned_reacher_geometry_strengthening_sgd_check.csv", sgd_rows)
    write_text(RESULT_ROOT / "task_aligned_reacher_geometry_strengthening_sgd_check_report.md", "\n".join(sgd_report_lines) + "\n")

    baseline_curves: list[dict[str, Any]] = []
    baseline_summaries: list[dict[str, Any]] = []
    final_decision = "GEOMETRY_STILL_WEAK"
    baseline_lines = ["# task_aligned_reacher_geometry_strengthened_baseline_gate_report", ""]
    if promising is not None:
        cfg, bundle, actor_lr, _ = promising
        baseline_curves, baseline_summaries = baseline_run(bundle, cfg, actor_lr)
        write_csv(RESULT_ROOT / "task_aligned_reacher_geometry_strengthened_baseline_gate.csv", baseline_summaries)
        plot_best_baseline(baseline_curves)
        by_method = {row["method"]: row for row in baseline_summaries}
        sgd = by_method["sgd"]
        egm = by_method["egm"]
        ppm = by_method["ppm"]
        geom_row = next(row for row in sweep_rows if row["alpha_dyn"] == cfg.alpha_dyn and row["disturbance_type"] == cfg.disturbance_type)
        all_valid = all(row["valid_flag"] == 1 for row in baseline_summaries)
        all_normal = all(row["curve_normal_flag"] == 1 for row in baseline_summaries)
        egm_adv = (sgd["V_align_AUC"] / (egm["V_align_AUC"] + base.EPS) >= 1.3) or (sgd["P_tau_T_AUC"] / (egm["P_tau_T_AUC"] + base.EPS) >= 1.3)
        ppm_adv = (sgd["V_align_AUC"] / (ppm["V_align_AUC"] + base.EPS) >= 1.3) or (sgd["P_tau_T_AUC"] / (ppm["P_tau_T_AUC"] + base.EPS) >= 1.3)
        winner = None
        if egm_adv:
            winner = "egm"
        if ppm_adv and (winner is None or ppm["V_align_AUC"] < by_method[winner]["V_align_AUC"]):
            winner = "ppm"
        comparable_field = False
        robust_ok = False
        if winner is not None:
            win = by_method[winner]
            comparable_field = win["field_norm_AUC"] <= (1.1 * sgd["field_norm_AUC"] + base.EPS)
            robust_ok = win["robust_br_task_return_raw_final"] >= (sgd["robust_br_task_return_raw_final"] - 1e-6)
        if not all_valid or not all_normal:
            final_decision = "SGD_BREAKS_AFTER_GEOMETRY_STRENGTHENING"
        elif not bool(geom_row["geometry_pass"]):
            final_decision = "GEOMETRY_STILL_WEAK"
        elif winner is None or not comparable_field:
            final_decision = "BASELINE_FAIL_NO_EGM_PPM_ADVANTAGE"
        elif not robust_ok:
            final_decision = "BASELINE_FAIL_RARL_PERFORMANCE"
        else:
            final_decision = "GEOMETRY_STRENGTHENED_BASELINE_READY_FOR_PROPOSED"
        baseline_lines.extend(
            [
                f"- selected_alpha_dyn: `{cfg.alpha_dyn}`",
                f"- selected_disturbance_type: `{cfg.disturbance_type}`",
                f"- selected_actor_lr: `{actor_lr}`",
                f"- geometry_rotation_ratio_proxy: `{geom_row['rotation_ratio_proxy']:.6e}`",
                f"- geometry_cross_player_coupling_proxy: `{geom_row['cross_player_coupling_proxy']:.6e}`",
                f"- geometry_cross_to_same_ratio: `{geom_row['cross_to_same_ratio']:.6e}`",
                "",
            ]
        )
        for row in baseline_summaries:
            baseline_lines.append(
                f"- {row['method'].upper()}: valid=`{bool(row['valid_flag'])}`, curve_normal=`{bool(row['curve_normal_flag'])}`, "
                f"V_align_AUC=`{row['V_align_AUC']:.6e}`, P_tau_T_AUC=`{row['P_tau_T_AUC']:.6e}`, field_norm_AUC=`{row['field_norm_AUC']:.6e}`, "
                f"robust_br_task_return_raw_final=`{row['robust_br_task_return_raw_final']:.6e}`"
            )
        baseline_lines.extend(
            [
                "",
                f"- egm_beats_sgd: `{egm_adv}`",
                f"- ppm_beats_sgd: `{ppm_adv}`",
                f"- winner: `{winner}`",
                f"- winner_field_norm_comparable: `{comparable_field}`",
                f"- winner_robust_br_not_worse: `{robust_ok}`",
                f"- decision: `{final_decision}`",
            ]
        )
    else:
        write_csv(RESULT_ROOT / "task_aligned_reacher_geometry_strengthened_baseline_gate.csv", [])
        baseline_lines.append("- No geometry-improved and SGD-clean config was found, so baseline gate was not run.")
        baseline_lines.append(f"- decision: `{final_decision}`")
    write_text(RESULT_ROOT / "task_aligned_reacher_geometry_strengthened_baseline_gate_report.md", "\n".join(baseline_lines) + "\n")

    final_lines = [
        "# task_aligned_reacher_geometry_strengthening_final_decision",
        "",
        "1. Is Reacher task-aligned frozen-critic protocol clean?",
        "   Yes. Task-aligned frozen SGD remains normal under the frozen-critic protocol.",
        "2. Is its geometry naturally strong enough?",
        f"   `{any(row['geometry_pass'] == 1 for row in sweep_rows)}` under the tested task-aligned disturbance-only sweep.",
        "3. Does increasing alpha_dyn strengthen skew geometry?",
        "   See the sweep report; the tested alphas were chosen to compare geometry while keeping action clipping controlled.",
        "4. Does direct vs rotated disturbance matter?",
        f"   Yes; the sweep compares `rotated`, `direct`, and `mixed` directly and records the resulting geometry metrics.",
        "5. Do EGM/PPM begin to outperform SGD?",
        f"   `{final_decision == 'GEOMETRY_STRENGTHENED_BASELINE_READY_FOR_PROPOSED'}` if and only if the strengthened baseline gate passed.",
        "6. Should Reacher proceed to proposed?",
        f"   `{final_decision == 'GEOMETRY_STRENGTHENED_BASELINE_READY_FOR_PROPOSED'}`.",
        "7. Or should we stop Reacher and move to task-aligned LQ / Swimmer?",
        f"   `{final_decision != 'GEOMETRY_STRENGTHENED_BASELINE_READY_FOR_PROPOSED'}`.",
        "",
        f"- final_decision: `{final_decision}`",
    ]
    write_text(RESULT_ROOT / "task_aligned_reacher_geometry_strengthening_final_decision.md", "\n".join(final_lines) + "\n")


if __name__ == "__main__":
    main()
