from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
SURVEY_PATH = SCRIPT_DIR / "survey_standard_rarl_skew_geometry.py"
spec = importlib.util.spec_from_file_location("vic_joint_survey", SURVEY_PATH)
survey = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = survey
spec.loader.exec_module(survey)

METHODS = ("gda", "egm", "ppm", "nog", "qpg")
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact joint-actor saddle-field RARL on native Gymnasium MuJoCo reward."
    )
    parser.add_argument("--env", default="HalfCheetah-v4")
    parser.add_argument("--mode", choices=("pretrain", "joint", "clean_optimizer"), default="joint")
    parser.add_argument("--method", choices=METHODS, default="qpg")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-scale", type=float, default=2.5)
    parser.add_argument("--force-body")
    parser.add_argument("--warmup-steps", type=int, default=20_000)
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--rollout-steps", type=int, default=250)
    parser.add_argument("--critic-updates-per-step", type=float, default=1.0)
    parser.add_argument("--twin-critic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--critic-softmin-temperature", type=float, default=1.0)
    parser.add_argument("--critic-arch", choices=("mlp", "game_bilinear"), default="mlp")
    parser.add_argument("--critic-own-curvature-scale", type=float, default=0.1)
    parser.add_argument("--load-pretrained-critic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--critic-disagreement-max", type=float, default=0.25)
    parser.add_argument("--critic-corr-min", type=float, default=0.60)
    parser.add_argument("--target-policy-noise", type=float, default=0.1)
    parser.add_argument("--target-noise-clip", type=float, default=0.2)
    parser.add_argument("--actor-update-interval", type=int, default=50)
    parser.add_argument("--actor-lr", type=float, default=3e-5)
    parser.add_argument("--actor-scope", choices=("full", "head"), default="full")
    parser.add_argument("--pretrain-actor-lr", type=float, default=1e-4)
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--exploration-std", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--geometry-probes", type=int, default=8)
    parser.add_argument("--ppm-inner", type=int, default=3)
    parser.add_argument("--qp-radius-mult", type=float, default=2.0)
    parser.add_argument("--qp-ridge", type=float, default=1e-6)
    parser.add_argument("--merit", choices=("composite", "field_energy"), default="composite")
    parser.add_argument("--lambda-f", type=float, default=0.01)
    parser.add_argument("--lambda-p", type=float, default=1.0)
    parser.add_argument("--gap-tau", type=float, default=0.03)
    parser.add_argument("--gap-inner-steps", type=int, default=2)
    parser.add_argument("--gap-inner-lr", type=float, default=3e-5)
    parser.add_argument("--gap-radius", type=float, default=0.1)
    parser.add_argument("--gap-softplus-eps", type=float, default=1e-3)
    parser.add_argument("--qp-backtrack-factor", type=float, default=0.5)
    parser.add_argument("--qp-backtrack-steps", type=int, default=6)
    parser.add_argument("--nog-safe-realized-merit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-ramp-steps", type=int, default=25_000)
    parser.add_argument("--force-start-fraction", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_env_info(env_name: str, force_body: str | None = None):
    env = survey.base.gym.make(env_name)
    body_id, body_name, body_names = survey.choose_main_body(env)
    if force_body is not None:
        if force_body not in body_names:
            env.close()
            raise ValueError(f"force body {force_body!r} not found; available bodies: {body_names}")
        body_id = body_names.index(force_body)
        body_name = force_body
    info = survey.EnvInfo(
        env_name=env_name,
        available=True,
        backend_type=survey.base.GYM_BACKEND,
        obs_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.shape[0]),
        action_low=np.asarray(env.action_space.low, dtype=np.float32).copy(),
        action_high=np.asarray(env.action_space.high, dtype=np.float32).copy(),
        max_episode_steps=int(env.spec.max_episode_steps),
        has_mujoco=hasattr(env.unwrapped, "data") and hasattr(env.unwrapped, "model"),
        has_xfrc_applied=hasattr(env.unwrapped.data, "xfrc_applied"),
        body_names=body_names,
        notes="",
        main_body_id=body_id,
        main_body_name=body_name,
    )
    env.close()
    if not info.has_xfrc_applied or body_id is None:
        raise RuntimeError(f"{env_name} does not expose MuJoCo xfrc_applied")
    return info


def field_energy(game, z: torch.Tensor, states: torch.Tensor) -> float:
    field = game.actor_field(z, states)
    return 0.5 * float(torch.dot(field, field).detach().item())


def _smooth_local_gap(game, z: torch.Tensor, states: torch.Tensor, protagonist: bool, args: argparse.Namespace) -> float:
    """Finite-inner-step proximal gap used by the paper's implemented merit."""
    base = z.detach().clone()
    current = base.clone()
    sl = game.theta_slice if protagonist else game.phi_slice
    initial = base[sl].clone()
    with torch.no_grad():
        j_start = game.actor_objective(base, states)
    tau = max(float(args.gap_tau), EPS)
    radius = max(float(args.gap_radius), EPS)
    for _ in range(args.gap_inner_steps):
        cur = current.detach().clone().requires_grad_(True)
        displacement = cur[sl] - initial
        prox = 0.5 * torch.dot(displacement, displacement) / tau
        objective = game.actor_objective(cur, states)
        envelope = objective - prox if protagonist else objective + prox
        grad = torch.autograd.grad(envelope, cur)[0][sl]
        proposal = cur[sl] + (args.gap_inner_lr * grad if protagonist else -args.gap_inner_lr * grad)
        delta = proposal - initial
        # A smooth coordinate-wise trust region keeps the finite inner map differentiable.
        bounded = initial + radius * torch.tanh(delta / radius)
        current = cur.detach().clone()
        current[sl] = bounded.detach()
    with torch.no_grad():
        displacement = current[sl] - initial
        prox = 0.5 * torch.dot(displacement, displacement) / tau
        j_end = game.actor_objective(current, states)
        raw_gap = (j_end - prox - j_start) if protagonist else (j_start - j_end - prox)
        eps = max(float(args.gap_softplus_eps), EPS)
        gap = torch.nn.functional.softplus(raw_gap / eps) * eps
    return float(gap.item())


def composite_merit(game, z: torch.Tensor, states: torch.Tensor, args: argparse.Namespace) -> dict[str, float]:
    energy = field_energy(game, z, states)
    if args.merit == "field_energy":
        return {"V": energy, "field_energy": energy, "P_tau": 0.0, "field_term": energy, "gap_term": 0.0}
    p_gap = _smooth_local_gap(game, z, states, True, args)
    a_gap = 0.0 if getattr(game, "single_agent", False) else _smooth_local_gap(game, z, states, False, args)
    p_tau = p_gap + a_gap
    if "composite_field0" not in game.metric_refs:
        game.metric_refs["composite_field0"] = max(energy, EPS)
        game.metric_refs["composite_ptau0"] = max(p_tau, EPS)
    field_term = energy / game.metric_refs["composite_field0"]
    gap_term = p_tau / game.metric_refs["composite_ptau0"]
    return {
        "V": (args.lambda_f * field_term) + (args.lambda_p * gap_term),
        "field_energy": energy,
        "P_tau": p_tau,
        "field_term": field_term,
        "gap_term": gap_term,
    }


def fit_qp_model(game, z, states, p_unit, g_unit, radius: float, ridge: float, args: argparse.Namespace):
    # Symmetric probes estimate the local quadratic model; the eventual step is
    # constrained to the nonnegative cone spanned by -F and +JF.
    probes = [
        (0.0, 0.0), (radius, 0.0), (-radius, 0.0),
        (0.0, radius), (0.0, -radius),
        (radius, radius), (radius, -radius), (-radius, radius),
    ]
    design, values = [], []
    for x, y in probes:
        design.append([1.0, x, y, 0.5 * x * x, x * y, 0.5 * y * y])
        values.append(composite_merit(game, z + x * p_unit + y * g_unit, states, args)["V"])
    coef, *_ = np.linalg.lstsq(np.asarray(design), np.asarray(values), rcond=None)
    _, lx, ly, hxx, hxy, hyy = coef.tolist()
    h = np.asarray([[hxx, hxy], [hxy, hyy]], dtype=np.float64)
    h = 0.5 * (h + h.T)
    min_eig = float(np.min(np.linalg.eigvalsh(h)))
    inflation = max(float(ridge), -min_eig + float(ridge))
    h = h + inflation * np.eye(2, dtype=np.float64)
    l = np.asarray([lx, ly], dtype=np.float64)
    return l, h, inflation


def solve_box_qp(l: np.ndarray, h: np.ndarray, bound: float, allow_g: bool):
    def objective(xy):
        return float(l @ xy + 0.5 * xy @ h @ xy)

    candidates = [np.zeros(2), np.asarray([bound, 0.0])]
    denom = max(float(h[0, 0]), EPS)
    candidates.append(np.asarray([np.clip(-l[0] / denom, 0.0, bound), 0.0]))
    if allow_g:
        candidates.extend([np.asarray([0.0, bound]), np.asarray([bound, bound])])
        denom_y = max(float(h[1, 1]), EPS)
        candidates.append(np.asarray([0.0, np.clip(-l[1] / denom_y, 0.0, bound)]))
        candidates.append(np.asarray([bound, np.clip(-(l[1] + h[1, 0] * bound) / denom_y, 0.0, bound)]))
        candidates.append(np.asarray([np.clip(-(l[0] + h[0, 1] * bound) / denom, 0.0, bound), bound]))
        try:
            interior = -np.linalg.solve(h, l)
            if np.all(np.isfinite(interior)) and np.all(interior >= 0.0) and np.all(interior <= bound):
                candidates.append(interior)
        except np.linalg.LinAlgError:
            pass
    best = min(candidates, key=objective)
    return best, objective(best)


def joint_update(game, states, method: str, actor_lr: float, ppm_inner: int, radius_mult: float, ridge: float, args: argparse.Namespace):
    z = game.current_z()
    z_req = z.detach().clone().requires_grad_(True)
    field = game.actor_field(z_req, states)
    f = field.detach()
    f_norm = float(torch.linalg.norm(f).item())
    if method == "gda":
        delta = -actor_lr * f
        return z + delta, {"beta": actor_lr, "gamma": 0.0, "gamma_active": 0, "fallback": 0}
    if method == "egm":
        half = z - actor_lr * f
        delta = -actor_lr * game.actor_field(half, states).detach()
        return z + delta, {"beta": actor_lr, "gamma": 0.0, "gamma_active": 0, "fallback": 0}
    if method == "ppm":
        current = z.detach().clone()
        for _ in range(ppm_inner):
            current = z - actor_lr * game.actor_field(current, states).detach()
        return current, {"beta": actor_lr, "gamma": 0.0, "gamma_active": 0, "fallback": 0}

    _, g = torch.autograd.functional.jvp(
        lambda zz: game.actor_field(zz, states), (z_req,), (f,), create_graph=False, strict=False
    )
    g = g.detach()
    g_norm = float(torch.linalg.norm(g).item())
    critic_disagreement = 0.0
    if game.q_t2 is not None:
        with torch.no_grad():
            u, w = game.actor_z(z, states)
            q1 = game.q_t(states, u, w)
            q2 = game.q_t2(states, u, w)
            critic_disagreement = float(
                torch.mean(torch.abs(q1 - q2)).item()
                / (torch.mean(0.5 * (torch.abs(q1) + torch.abs(q2))).item() + EPS)
            )
    corr_reliable = bool(getattr(game, "curvature_corr_reliable", False))
    curvature_reliable = critic_disagreement <= args.critic_disagreement_max and corr_reliable
    base_radius = max(actor_lr * f_norm, 1e-8)
    bound = radius_mult * base_radius
    p_unit = -f / (f_norm + EPS)
    g_unit = g / (g_norm + EPS)
    l, h, inflation = fit_qp_model(game, z, states, p_unit, g_unit, base_radius, ridge, args)
    reduced_gradient = float(h[0, 1] * l[0] - h[0, 0] * l[1])
    nog_xy, nog_predicted_change = solve_box_qp(l, h, bound, allow_g=False)
    qp_xy, qp_predicted_change = solve_box_qp(l, h, bound, allow_g=curvature_reliable)
    pred_tol = 1e-9 * max(1.0, abs(qp_predicted_change), abs(nog_predicted_change))
    if qp_predicted_change > nog_predicted_change + pred_tol:
        raise RuntimeError("2D QP failed to contain its noG boundary candidate")
    if method == "qpg":
        xy, predicted_change = qp_xy.copy(), qp_predicted_change
    else:
        xy, predicted_change = nog_xy.copy(), nog_predicted_change

    before_metrics = composite_merit(game, z, states, args)
    nog_delta = float(nog_xy[0]) * p_unit
    nog_metrics = composite_merit(game, z + nog_delta, states, args)
    delta = float(xy[0]) * p_unit + float(xy[1]) * g_unit
    after_metrics = composite_merit(game, z + delta, states, args)
    backtracks = 0
    while (
        (not math.isfinite(after_metrics["V"]) or after_metrics["V"] > before_metrics["V"])
        and backtracks < args.qp_backtrack_steps
    ):
        xy = xy * args.qp_backtrack_factor
        delta = float(xy[0]) * p_unit + float(xy[1]) * g_unit
        after_metrics = composite_merit(game, z + delta, states, args)
        backtracks += 1
    chose_nog = 0
    if args.nog_safe_realized_merit and nog_metrics["V"] < after_metrics["V"]:
        xy = nog_xy.copy()
        delta = nog_delta
        after_metrics = nog_metrics
        chose_nog = 1
    fallback = int(not math.isfinite(after_metrics["V"]))
    if fallback:
        xy = np.zeros(2, dtype=np.float64)
        delta = torch.zeros_like(z)
        after_metrics = before_metrics
    beta = float(xy[0]) / (f_norm + EPS)
    gamma = float(xy[1]) / (g_norm + EPS)
    return (z + delta).detach(), {
        "beta": beta,
        "gamma": gamma,
        "gamma_active": int(gamma > 1e-12),
        "critic_relative_disagreement": critic_disagreement,
        "curvature_corr_gate": int(corr_reliable),
        "curvature_reliability_gate": int(curvature_reliable),
        "fallback": fallback,
        "backtracks": backtracks,
        "chose_nog_boundary": chose_nog,
        "qp_pd_inflation": inflation,
        "qp_l_field": float(l[0]),
        "qp_l_G": float(l[1]),
        "qp_h_field_field": float(h[0, 0]),
        "qp_h_field_G": float(h[0, 1]),
        "qp_h_G_G": float(h[1, 1]),
        "qp_reduced_gradient": reduced_gradient,
        "predicted_composite_change": predicted_change,
        "predicted_nog_change": nog_predicted_change,
        "counterfactual_qpg_change": qp_predicted_change,
        "counterfactual_qpg_gain": nog_predicted_change - qp_predicted_change,
        "counterfactual_qpg_field_norm": float(qp_xy[0]),
        "counterfactual_qpg_G_norm": float(qp_xy[1]),
        "predicted_inclusion_gap": qp_predicted_change - nog_predicted_change,
        "predicted_inclusion_pass": int(qp_predicted_change <= nog_predicted_change + pred_tol),
        "V_before": before_metrics["V"],
        "V_after": after_metrics["V"],
        "V_after_nog": nog_metrics["V"],
        "realized_composite_change": after_metrics["V"] - before_metrics["V"],
        "realized_qp_le_nog": int(after_metrics["V"] <= nog_metrics["V"] + pred_tol),
        "field_energy_before": before_metrics["field_energy"],
        "field_energy_after": after_metrics["field_energy"],
        "P_tau_before": before_metrics["P_tau"],
        "P_tau_after": after_metrics["P_tau"],
        "G_contribution_norm": float(xy[1]),
        "G_over_update_norm": float(xy[1]) / (float(torch.linalg.norm(delta).item()) + EPS),
    }


def collect(game, env, obs, steps: int, noise_std: float, rng: np.random.Generator):
    episode_return = 0.0
    completed: list[float] = []
    for _ in range(steps):
        obs_t = torch.as_tensor(obs, dtype=survey.DTYPE, device=survey.DEVICE).unsqueeze(0)
        with torch.no_grad():
            u = game.actor_protagonist(game.theta, obs_t).squeeze(0)
            w = game.actor_adversary(game.phi, obs_t).squeeze(0)
            u = torch.clamp(u + noise_std * torch.randn_like(u), game.wrapper.action_low, game.wrapper.action_high)
            w = torch.clamp(w + noise_std * torch.randn_like(w), -1.0, 1.0)
        next_obs, reward, done, _, _, err = game.step_env(env, u, w)
        if err:
            raise RuntimeError("MuJoCo wrapper step failed")
        game.replay.add(obs, u.cpu().numpy(), w.cpu().numpy(), reward, next_obs, done)
        episode_return += reward
        obs = next_obs
        if done:
            completed.append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    return obs, completed


def set_force_scale(game, force_scale: float) -> None:
    cfg = dataclasses.replace(
        game.cfg,
        strength_value=float(force_scale),
        strength_name=f"force_{float(force_scale):g}",
    )
    game.cfg = cfg
    game.wrapper.cfg = cfg


def collect_clean(game, env, obs, steps: int, noise_std: float, rng: np.random.Generator):
    episode_return = 0.0
    completed: list[float] = []
    for _ in range(steps):
        obs_t = torch.as_tensor(obs, dtype=survey.DTYPE, device=survey.DEVICE).unsqueeze(0)
        with torch.no_grad():
            u = game.actor_protagonist(game.theta, obs_t).squeeze(0)
            u = torch.clamp(u + noise_std * torch.randn_like(u), game.wrapper.action_low, game.wrapper.action_high)
            w = torch.zeros((game.cfg.adv_dim,), dtype=survey.DTYPE, device=survey.DEVICE)
        next_obs, reward, done, _, _, err = game.step_env(env, u, w)
        if err:
            raise RuntimeError("MuJoCo clean pretraining step failed")
        game.replay.add(obs, u.cpu().numpy(), w.cpu().numpy(), reward, next_obs, done)
        episode_return += reward
        obs = next_obs
        if done:
            completed.append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    return obs, completed


def critic_update_clean(game) -> dict[str, float]:
    batch = game.replay.sample(survey.BATCH_SIZE, game.rng)
    with torch.no_grad():
        u_next = game.actor_protagonist(game.theta_target, batch["next_obs"])
        w_next = torch.zeros((u_next.shape[0], game.cfg.adv_dim), dtype=survey.DTYPE, device=survey.DEVICE)
        q_next = game.q_t_target(batch["next_obs"], u_next, w_next)
        if game.q_t2_target is not None:
            q_next = torch.minimum(q_next, game.q_t2_target(batch["next_obs"], u_next, w_next))
        target = batch["reward"] + survey.GAMMA * (1.0 - batch["done"]) * q_next
    pred = game.q_t(batch["obs"], batch["u"], batch["w"])
    loss = torch.mean((pred - target) ** 2)
    game.q_opt.zero_grad(set_to_none=True)
    loss.backward()
    game.q_opt.step()
    loss2_value = math.nan
    if game.q_t2 is not None:
        batch2 = game.replay.sample(survey.BATCH_SIZE, game.rng)
        with torch.no_grad():
            u_next2 = game.actor_protagonist(game.theta_target, batch2["next_obs"])
            w_next2 = torch.zeros((u_next2.shape[0], game.cfg.adv_dim), dtype=survey.DTYPE, device=survey.DEVICE)
            q_next21 = game.q_t_target(batch2["next_obs"], u_next2, w_next2)
            q_next22 = game.q_t2_target(batch2["next_obs"], u_next2, w_next2)
            target2 = batch2["reward"] + survey.GAMMA * (1.0 - batch2["done"]) * torch.minimum(q_next21, q_next22)
        pred2 = game.q_t2(batch2["obs"], batch2["u"], batch2["w"])
        loss2 = torch.mean((pred2 - target2) ** 2)
        game.q_opt2.zero_grad(set_to_none=True)
        loss2.backward()
        game.q_opt2.step()
        loss2_value = float(loss2.detach().item())
    game.polyak()
    return {"critic_loss": float(loss.detach().item()), "critic2_loss": loss2_value}


def load_pretrained(game, path: Path, load_critic: bool = True) -> None:
    payload = torch.load(path, map_location=survey.DEVICE, weights_only=False)
    game.theta = payload["theta"].to(survey.DEVICE).detach().clone()
    game.theta_target = payload["theta_target"].to(survey.DEVICE).detach().clone()
    if load_critic:
        game.q_t.load_state_dict(payload["critic"])
        game.q_t_target.load_state_dict(payload["critic_target"])
        if game.q_t2 is not None:
            game.q_t2.load_state_dict(payload.get("critic2", payload["critic"]))
            game.q_t2_target.load_state_dict(payload.get("critic2_target", payload["critic_target"]))


class SingleAgentGame:
    """A true protagonist-only view: phi is absent from the optimization vector."""

    single_agent = True

    def __init__(self, base_game) -> None:
        self.base = base_game
        self.theta_slice = slice(0, base_game.theta_layout.num_params)
        self.metric_refs: dict[str, float] = {}
        self.curvature_corr_reliable = False

    def __getattr__(self, name):
        return getattr(self.base, name)

    def current_z(self) -> torch.Tensor:
        return self.base.theta.detach().clone()

    def set_from_z(self, z: torch.Tensor) -> None:
        self.base.theta = z.detach().clone()

    def actor_z(self, z: torch.Tensor, states: torch.Tensor):
        u = self.base.actor_protagonist(z, states)
        w = torch.zeros((states.shape[0], self.base.cfg.adv_dim), dtype=survey.DTYPE, device=survey.DEVICE)
        return u, w

    def actor_objective(self, z: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        u, w = self.actor_z(z, states)
        q1 = self.base.q_t(states, u, w)
        if self.base.q_t2 is None:
            return q1.mean()
        q2 = self.base.q_t2(states, u, w)
        temperature = max(float(self.base.critic_softmin_temperature), EPS)
        return (-temperature * torch.logsumexp(
            torch.stack((-q1 / temperature, -q2 / temperature), dim=0), dim=0
        )).mean()

    def actor_field(self, z: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        z_req = z if z.requires_grad else z.detach().clone().requires_grad_(True)
        objective = self.actor_objective(z_req, states)
        return -torch.autograd.grad(objective, z_req, create_graph=True)[0]

    def polyak(self) -> None:
        self.base.polyak()


def clean_mc_quality(game: SingleAgentGame, snapshots: list[dict[str, np.ndarray]]) -> dict[str, float]:
    if not snapshots:
        return {"corr_Q_MC": math.nan, "mse_Q_MC": math.nan}
    env = survey.base.gym.make(game.info.env_name)
    q_vals: list[float] = []
    mc_vals: list[float] = []
    for snap in snapshots:
        env.reset(seed=survey.SEED)
        env.unwrapped.set_state(snap["qpos"], snap["qvel"])
        obs_t = torch.as_tensor(snap["obs"], dtype=survey.DTYPE, device=survey.DEVICE).unsqueeze(0)
        with torch.no_grad():
            u = game.actor_protagonist(game.theta, obs_t)
            w = torch.zeros((1, game.cfg.adv_dim), dtype=survey.DTYPE, device=survey.DEVICE)
            q1 = game.q_t(obs_t, u, w)
            q_value = q1 if game.q_t2 is None else torch.minimum(q1, game.q_t2(obs_t, u, w))
        q_vals.append(float(q_value.item()))
        total, discount = 0.0, 1.0
        for _ in range(survey.MC_HORIZON):
            with torch.no_grad():
                u_step = game.actor_protagonist(game.theta, obs_t).squeeze(0)
                w_step = torch.zeros((game.cfg.adv_dim,), dtype=survey.DTYPE, device=survey.DEVICE)
            next_obs, reward, done, _, _, err = game.step_env(env, u_step, w_step)
            if err:
                break
            total += discount * reward
            discount *= survey.GAMMA
            obs_t = torch.as_tensor(next_obs, dtype=survey.DTYPE, device=survey.DEVICE).unsqueeze(0)
            if done:
                break
        mc_vals.append(total)
    env.close()
    return {"corr_Q_MC": survey.corr_or_nan(q_vals, mc_vals), "mse_Q_MC": survey.mse_or_nan(q_vals, mc_vals)}


def hessian_symmetry_error(game: SingleAgentGame, states: torch.Tensor, seed: int) -> float:
    z = game.current_z().requires_grad_(True)
    generator = torch.Generator(device=survey.DEVICE).manual_seed(seed)
    v = torch.randn(z.shape, generator=generator, dtype=z.dtype, device=z.device)
    w = torch.randn(z.shape, generator=generator, dtype=z.dtype, device=z.device)
    v = v / (torch.linalg.norm(v) + EPS)
    w = w / (torch.linalg.norm(w) + EPS)
    _, jv = torch.autograd.functional.jvp(lambda zz: game.actor_field(zz, states), (z,), (v,), strict=False)
    _, jw = torch.autograd.functional.jvp(lambda zz: game.actor_field(zz, states), (z,), (w,), strict=False)
    numerator = torch.abs(torch.dot(v, jw) - torch.dot(w, jv))
    denominator = torch.abs(torch.dot(v, jw)) + torch.abs(torch.dot(w, jv)) + EPS
    return float((numerator / denominator).detach().item())


def run_clean_optimizer(base_game, args: argparse.Namespace) -> None:
    set_force_scale(base_game, 0.0)
    game = SingleAgentGame(base_game)
    env = survey.base.gym.make(args.env)
    rng = np.random.default_rng(args.seed + 20260725)
    obs, _ = env.reset(seed=args.seed + 1700)
    obs, initial_returns = collect_clean(game, env, obs, args.warmup_steps, args.exploration_std, rng)
    for _ in range(args.warmup_steps):
        critic_update_clean(game)

    rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    train_window = list(initial_returns)
    critic_budget = 0.0
    for step0 in range(0, args.total_steps + 1, args.rollout_steps):
        if step0 > 0:
            n = min(args.rollout_steps, args.total_steps - (step0 - args.rollout_steps))
            obs, returns = collect_clean(game, env, obs, n, args.exploration_std, rng)
            train_window.extend(returns)
            critic_budget += n * args.critic_updates_per_step
            for _ in range(int(critic_budget)):
                critic_update_clean(game)
            critic_budget -= int(critic_budget)
            for actor_idx in range(max(1, n // args.actor_update_interval)):
                states = game.replay.fixed_state_batch(survey.BATCH_SIZE, rng)["obs"]
                z_next, meta = joint_update(
                    game, states, args.method, args.actor_lr, args.ppm_inner,
                    args.qp_radius_mult, args.qp_ridge, args,
                )
                game.set_from_z(z_next)
                game.polyak()
                update_rows.append({"step": min(step0, args.total_steps), "actor_update": actor_idx, **meta})
        step = min(step0, args.total_steps)
        if step % args.eval_interval == 0 or step == args.total_steps:
            clean = game.evaluate_actor(game.theta, None, args.eval_episodes)
            snapshots = game.collect_snapshots(survey.MC_SNAPSHOTS)
            quality = clean_mc_quality(game, snapshots)
            corr = float(quality["corr_Q_MC"])
            game.curvature_corr_reliable = math.isfinite(corr) and corr >= args.critic_corr_min
            diag_states = game.replay.fixed_state_batch(min(32, survey.BATCH_SIZE), rng)["obs"]
            rows.append({
                "env": args.env, "method": args.method, "seed": args.seed, "step": step,
                "protagonist_env_steps": args.warmup_steps + step,
                "train_episode_return": float(np.mean(train_window[-10:])) if train_window else math.nan,
                "clean_return": clean["return"], "force_norm": 0.0,
                "skew_symmetry_error": hessian_symmetry_error(game, diag_states, args.seed + step),
                **quality,
            })
            save_checkpoint(args.output / f"checkpoint_{step:08d}.pt", base_game, args, step)
        if step >= args.total_steps:
            break
    env.close()
    write_csv(args.output / "convergence.csv", rows)
    write_csv(args.output / "updates.csv", update_rows)
    (args.output / "metadata.json").write_text(json.dumps({
        "algorithm": "true single-agent deterministic actor optimizer sanity check",
        "optimization_variables": ["protagonist_actor"],
        "absent_variables": ["adversary_actor"],
        "reward": "native Gymnasium task reward",
        "force": "identically zero in collection, critic objective, and evaluation",
        "field": "F=-grad_theta mean Q(s,pi_theta(s),w=0)",
        "curvature": "G=J_F F; conservative Hessian-gradient, not game rotation",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }, indent=2), encoding="utf-8")


def run_clean_pretraining(game, args: argparse.Namespace) -> None:
    env = survey.base.gym.make(args.env)
    rng = np.random.default_rng(args.seed + 20260721)
    obs, _ = env.reset(seed=args.seed + 700)
    obs, initial_returns = collect_clean(game, env, obs, args.warmup_steps, args.exploration_std, rng)
    for _ in range(args.warmup_steps):
        critic_update_clean(game)

    theta_param = torch.nn.Parameter(game.theta.detach().clone())
    actor_opt = torch.optim.Adam([theta_param], lr=args.pretrain_actor_lr)
    rows: list[dict[str, Any]] = []
    train_window = list(initial_returns)
    critic_budget = 0.0
    for step0 in range(0, args.total_steps + 1, args.rollout_steps):
        if step0 > 0:
            n = min(args.rollout_steps, args.total_steps - (step0 - args.rollout_steps))
            game.theta = theta_param.detach().clone()
            obs, returns = collect_clean(game, env, obs, n, args.exploration_std, rng)
            train_window.extend(returns)
            critic_budget += n * args.critic_updates_per_step
            for _ in range(int(critic_budget)):
                critic_update_clean(game)
            critic_budget -= int(critic_budget)
            for _ in range(max(1, n // args.actor_update_interval)):
                states = game.replay.fixed_state_batch(survey.BATCH_SIZE, rng)["obs"]
                u = game.actor_protagonist(theta_param, states)
                w = torch.zeros((states.shape[0], game.cfg.adv_dim), dtype=survey.DTYPE, device=survey.DEVICE)
                q1 = game.q_t(states, u, w)
                if game.q_t2 is None:
                    objective = q1.mean()
                else:
                    q2 = game.q_t2(states, u, w)
                    temperature = max(float(game.critic_softmin_temperature), EPS)
                    objective = (
                        -temperature
                        * torch.logsumexp(torch.stack((-q1 / temperature, -q2 / temperature), dim=0), dim=0)
                    ).mean()
                grad = torch.autograd.grad(objective, theta_param)[0]
                actor_opt.zero_grad(set_to_none=True)
                theta_param.grad = -grad
                actor_opt.step()
                game.theta = theta_param.detach().clone()
                game.polyak()
        step = min(step0, args.total_steps)
        if step % args.eval_interval == 0 or step == args.total_steps:
            game.theta = theta_param.detach().clone()
            clean = game.evaluate_actor(game.theta, None, args.eval_episodes)
            rows.append({
                "env": args.env,
                "seed": args.seed,
                "step": step,
                "total_clean_steps": args.warmup_steps + step,
                "train_episode_return": float(np.mean(train_window[-10:])) if train_window else math.nan,
                "clean_evaluation_return": clean["return"],
            })
            save_checkpoint(args.output / f"pretrain_checkpoint_{step:08d}.pt", game, args, step)
        if step >= args.total_steps:
            break
    env.close()
    write_csv(args.output / "pretrain_convergence.csv", rows)


def save_checkpoint(path: Path, game, args: argparse.Namespace, step: int) -> None:
    payload = {
            "theta": game.theta.cpu(), "phi": game.phi.cpu(),
            "theta_target": game.theta_target.cpu(), "phi_target": game.phi_target.cpu(),
            "critic": game.q_t.state_dict(), "critic_target": game.q_t_target.state_dict(),
            "step": step, "args": vars(args),
        }
    if game.q_t2 is not None:
        payload.update({"critic2": game.q_t2.state_dict(), "critic2_target": game.q_t2_target.state_dict()})
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    survey.SEED = args.seed
    survey.ROTATION_PROBES = args.geometry_probes
    survey.base.seed_everything(args.seed)
    info = build_env_info(args.env, args.force_body)
    cfg = survey.WrapperConfig("mujoco_external_force_xz_radial", args.force_scale, f"force_{args.force_scale}", 2, info.main_body_id, info.main_body_name)
    game = survey.SurveyGame(
        info,
        cfg,
        args.seed,
        twin_critic=args.twin_critic,
        actor_scope=args.actor_scope,
        critic_arch=args.critic_arch,
        own_curvature_scale=args.critic_own_curvature_scale,
    )
    game.critic_softmin_temperature = args.critic_softmin_temperature
    game.target_policy_noise = args.target_policy_noise
    game.target_noise_clip = args.target_noise_clip
    game.curvature_corr_reliable = False
    if args.pretrained_checkpoint is not None:
        load_pretrained(
            game,
            args.pretrained_checkpoint,
            load_critic=args.load_pretrained_critic and args.critic_arch == "mlp",
        )
    if args.mode == "pretrain":
        run_clean_pretraining(game, args)
        return
    if args.mode == "clean_optimizer":
        run_clean_optimizer(game, args)
        return
    target_force_scale = float(args.force_scale)
    start_force_scale = target_force_scale * float(np.clip(args.force_start_fraction, 0.0, 1.0))
    set_force_scale(game, start_force_scale)
    warm = game.collect_replay(args.warmup_steps, args.exploration_std)
    for _ in range(args.warmup_steps):
        game.critic_update()

    env = survey.base.gym.make(args.env)
    rng = np.random.default_rng(args.seed + 20260720)
    obs, _ = env.reset(seed=args.seed + 1000)
    rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    train_window: list[float] = []
    critic_budget = 0.0
    for step0 in range(0, args.total_steps + 1, args.rollout_steps):
        if step0 > 0:
            n = min(args.rollout_steps, args.total_steps - (step0 - args.rollout_steps))
            ramp_progress = 1.0 if args.force_ramp_steps <= 0 else min(1.0, float(step0) / args.force_ramp_steps)
            current_force_scale = start_force_scale + ramp_progress * (target_force_scale - start_force_scale)
            set_force_scale(game, current_force_scale)
            obs, returns = collect(game, env, obs, n, args.exploration_std, rng)
            train_window.extend(returns)
            critic_budget += n * args.critic_updates_per_step
            last_critic = {}
            for _ in range(int(critic_budget)):
                last_critic = game.critic_update()
            critic_budget -= int(critic_budget)
            actor_updates = max(1, n // args.actor_update_interval)
            for actor_idx in range(actor_updates):
                states = game.replay.fixed_state_batch(survey.BATCH_SIZE, rng)["obs"]
                z_next, meta = joint_update(
                    game, states, args.method, args.actor_lr, args.ppm_inner,
                    args.qp_radius_mult, args.qp_ridge, args,
                )
                game.set_from_z(z_next)
                game.polyak()
                update_rows.append({"step": min(step0, args.total_steps), "actor_update": actor_idx, **meta})
        step = min(step0, args.total_steps)
        if step % args.eval_interval == 0 or step == args.total_steps:
            diag_states = game.replay.fixed_state_batch(survey.BATCH_SIZE, rng)["obs"]
            geom = game.parameter_geometry(game.current_z(), diag_states)
            set_force_scale(game, target_force_scale)
            clean = game.evaluate_actor(game.theta, None, args.eval_episodes)
            robust = game.evaluate_actor(game.theta, game.phi, args.eval_episodes)
            snapshots = game.collect_snapshots(survey.MC_SNAPSHOTS)
            quality = game.mc_quality(snapshots)
            corr_value = float(quality.get("corr_Q_MC", math.nan))
            game.curvature_corr_reliable = math.isfinite(corr_value) and corr_value >= args.critic_corr_min
            rows.append({
                "env": args.env, "method": args.method, "seed": args.seed, "step": step,
                "protagonist_env_steps": args.warmup_steps + step,
                "train_episode_return": float(np.mean(train_window[-10:])) if train_window else math.nan,
                "training_force_scale": current_force_scale if step0 > 0 else start_force_scale,
                "clean_return": clean["return"], "own_adversary_robust_return": robust["return"],
                **geom, **quality, **warm,
            })
            save_checkpoint(args.output / f"checkpoint_{step:08d}.pt", game, args, step)
        if step >= args.total_steps:
            break
    env.close()
    write_csv(args.output / "convergence.csv", rows)
    write_csv(args.output / "updates.csv", update_rows)
    (args.output / "metadata.json").write_text(json.dumps({
        "algorithm": "centralized-critic exact joint deterministic actor game",
        "saddle_variables": ["protagonist_actor", "adversary_actor"],
        "actor_scope": args.actor_scope,
        "excluded_from_saddle": ["critic", "target_critic", "replay_buffer"],
        "critic": "independently minibatched twin smooth critics with a conservative soft-min actor objective" if args.twin_critic else "single smooth critic",
        "critic_arch": args.critic_arch,
        "critic_own_curvature_scale": args.critic_own_curvature_scale,
        "reward": "native Gymnasium task reward",
        "adversary": "state-dependent 2D x/z external force with Euclidean norm in [0, force_scale]",
        "joint_field": "F=[-grad_theta mean(Q), +grad_phi mean(Q)] on one fixed replay minibatch",
        "curvature": "G=J_F F via torch.func-style JVP",
        "merit": "normalized field energy plus smooth finite-inner-step proximal saddle gap" if args.merit == "composite" else "field energy only",
        "qp_inclusion": "the 2D box QP explicitly enumerates the noG boundary candidate; optional realized-merit safeguard selects noG on the same frozen batch",
        "force_curriculum": {"start_fraction": args.force_start_fraction, "ramp_steps": args.force_ramp_steps, "target": target_force_scale},
        "pretrained_checkpoint": str(args.pretrained_checkpoint) if args.pretrained_checkpoint else None,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
