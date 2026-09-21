from __future__ import annotations

import csv
import copy
import importlib.util
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / "actor_only_critic_rarl_subsection3.py"
RESULT_ROOT = BASE_PATH.resolve().parents[2] / "results" / "actor_only_critic_rarl_subsection3"

spec = importlib.util.spec_from_file_location("actor_only_critic_rarl_subsection3", BASE_PATH)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

ENV_ID = "Reacher-v5"
ENV_SLUG = "reacher_v5"
PREFIX = f"actor_only_critic_s3_{ENV_SLUG}_"

FROZEN_WARMUP_STEPS = 20000
FROZEN_CRITIC_TRAIN_STEPS = 3000
TASK_ALIGNED_LR = 1e-4
TASK_ALIGNED_ITERS = 100
MC_SNAPSHOT_COUNT = 16
MC_HORIZON = 50


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


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def parse_ready_report() -> dict[str, Any]:
    text = (RESULT_ROOT / f"{PREFIX}ready_for_baseline_gate.md").read_text(encoding="utf-8")
    out: dict[str, Any] = {}
    for key in ["selected_alpha_dyn", "selected_use_rot_dyn", "reward_scale", "a_u", "a_w", "selected_actor_lr"]:
        match = re.search(rf"- {re.escape(key)}: `([^`]+)`", text)
        if not match:
            continue
        val = match.group(1)
        if val in {"True", "False"}:
            out[key] = val == "True"
        else:
            out[key] = float(val)
    return out


def make_env_choice() -> base.EnvChoice:
    env = base.choose_environment()
    if env.env_id != ENV_ID:
        raise RuntimeError(f"Expected {ENV_ID}, got {env.env_id}")
    return env


def make_task_aligned_wrapper(env_choice: base.EnvChoice) -> base.WrapperConfig:
    ready = parse_ready_report()
    return base.WrapperConfig(
        reward_scale=float(ready["reward_scale"]),
        alpha_dyn=float(ready["selected_alpha_dyn"]),
        a_u=0.0,
        a_w=0.0,
        beta_rot=0.0,
        beta_sym=0.0,
        use_rot_dyn=bool(ready["selected_use_rot_dyn"]),
    )


def task_alignment_audit() -> bool:
    lines = [
        f"# {PREFIX}task_alignment_audit",
        "",
        "1. What reward was used to train the frozen critic?",
        "   The current frozen protocol still inherits the base `q_game` target `reward_game = reward_scale * r_task_raw - 0.5 a_u ||u||^2 + 0.5 a_w ||w||^2 + beta_rot term + beta_sym term`, while `q_task` is trained on `reward_task_scaled = reward_scale * r_task_raw` only.",
        "2. Does the critic target include only Reacher task reward?",
        "   `q_game`: no. `q_task`: yes.",
        "3. Does it include adversary energy term `+0.5 a_w ||w||^2`?",
        "   Yes for `q_game`; no for `q_task`.",
        "4. Does it include protagonist action penalty beyond the environment reward?",
        "   Yes for `q_game` through `-0.5 a_u ||u||^2`; no for `q_task`.",
        "5. Does it include `beta_rot u^T H w` or `beta_sym u^T S w`?",
        "   In the selected Reacher config both are numerically zero, so they are present in code but inactive in this run.",
        "6. Is `P_tau` computed from the same critic objective as robust task return?",
        "   No. Current `P_tau` uses `actor_objective(... use_task_q=False)` and therefore follows `q_game`, while robust BR uses `q_task`.",
        "7. Is `V_lambda` currently task-aligned or still mixed-game-aligned?",
        "   It is still mixed-game-aligned because both `field_term` and `P_tau` are built from `q_game`.",
        "8. Are robust BR returns evaluated using the same objective as `V_lambda/P_tau`?",
        "   No. Robust BR is task-only via `q_task`, while `V_lambda/P_tau` are game-critic based.",
        "",
        "Decision: `task_aligned_patch_needed = True`.",
    ]
    write_text(RESULT_ROOT / f"{PREFIX}task_alignment_audit.md", "\n".join(lines) + "\n")
    return False


class TaskAlignedFrozenGame(base.ActorCriticRARL):
    def actor_objective(self, z: torch.Tensor, states: torch.Tensor, use_task_q: bool = False) -> torch.Tensor:
        u, w = self.actor_z(z, states)
        return self.q_task(states, u, w).mean()

    def actor_field(self, z: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        z_req = z if z.requires_grad else z.detach().clone().requires_grad_(True)
        j_val = self.actor_objective(z_req, states, use_task_q=True)
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
        j_start = self.actor_objective(base_z, states, use_task_q=True)
        inner_lr = 0.1 * actor_lr
        for _ in range(base.GAP_INNER_STEPS):
            cur = current.detach().clone().requires_grad_(True)
            j_val = self.actor_objective(cur, states, use_task_q=True)
            grad = torch.autograd.grad(j_val, cur)[0][actor_slice]
            step = grad if protagonist else -grad
            if with_prox:
                step = step - ((cur[actor_slice] - initial) / max(base.LOCAL_GAP_RADIUS, base.EPS))
            next_actor = cur[actor_slice] + (inner_lr * step)
            delta = next_actor - initial
            norm = torch.linalg.norm(delta)
            if norm > base.LOCAL_GAP_RADIUS:
                next_actor = initial + (delta * (base.LOCAL_GAP_RADIUS / (norm + base.EPS)))
            current = cur.detach().clone()
            current[actor_slice] = next_actor.detach()
        j_end = self.actor_objective(current, states, use_task_q=True)
        if protagonist:
            return torch.relu(j_end - j_start)
        return torch.relu(j_start - j_end)

    def exploitability_parts(self, z: torch.Tensor, states: torch.Tensor, actor_lr: float) -> tuple[float, float]:
        p = float(self.local_gap(z, states, True, False, actor_lr).detach().item())
        a = float(self.local_gap(z, states, False, False, actor_lr).detach().item())
        return p, a

    def diagnostic_metrics(self, z: torch.Tensor, diag_batch: dict[str, torch.Tensor], actor_lr: float, compute_geometry: bool) -> dict[str, float]:
        states = diag_batch["obs"]
        field = self.actor_field(z, states).detach()
        field_energy = 0.5 * float(torch.dot(field, field).item())
        p_gap = float(self.local_gap(z, states, True, True, actor_lr).detach().item())
        a_gap = float(self.local_gap(z, states, False, True, actor_lr).detach().item())
        p_tau = p_gap + a_gap
        if "field0_T" not in self.metric_refs:
            self.metric_refs["field0_T"] = field_energy
            self.metric_refs["ptau0_T"] = max(p_tau, base.EPS)
        field_term = field_energy / (self.metric_refs["field0_T"] + base.EPS)
        p_tau_term = p_tau / (self.metric_refs["ptau0_T"] + base.EPS)
        v_align = (base.LAMBDA_F * field_term) + (base.LAMBDA_P * p_tau_term)
        exploit_p, exploit_a = self.exploitability_parts(z, states, actor_lr)
        u = self.actor_action(z[self.field_slices["theta"]], states)
        w = self.actor_action(z[self.field_slices["phi"]], states)
        j_task = float(self.q_task(states, u, w).mean().detach().item())
        out = {
            "V_align": v_align,
            "field_term_T": field_term,
            "normalized_P_tau_T": p_tau_term,
            "raw_P_tau_T": p_tau,
            "field_norm": float(torch.linalg.norm(field).item()),
            "approximate_exploitability_T": exploit_p + exploit_a,
            "exploitability_protagonist_T": exploit_p,
            "exploitability_adversary_T": exploit_a,
            "P_tau_T_protagonist_gap": p_gap,
            "P_tau_T_adversary_gap": a_gap,
            "Q_T": j_task,
        }
        if compute_geometry:
            geom = self.geometry_metrics(z, states)
            out.update(
                {
                    "g_over_f_T": geom["g_over_f"],
                    "cos_fg_T": geom["cos_fg"],
                    "non_collinearity_T": geom["non_collinearity"],
                    "rotation_ratio_proxy": geom["rotation_ratio_proxy"],
                    "cross_player_coupling_proxy": geom["cross_player_coupling_proxy"],
                    "cross_to_same_ratio": geom["cross_to_same_ratio"],
                }
            )
        else:
            out.update(
                {
                    "g_over_f_T": math.nan,
                    "cos_fg_T": math.nan,
                    "non_collinearity_T": math.nan,
                    "rotation_ratio_proxy": math.nan,
                    "cross_player_coupling_proxy": math.nan,
                    "cross_to_same_ratio": math.nan,
                }
            )
        return out

    def robust_br_adversary(self, theta: torch.Tensor, diag_batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, bool]:
        base_phi = self.phi.detach().clone()
        current = base_phi.clone()
        states = diag_batch["obs"]
        for _ in range(base.BR_INNER_STEPS):
            cur = current.detach().clone().requires_grad_(True)
            z = torch.cat([theta.detach(), cur])
            objective = self.actor_objective(z, states, use_task_q=True)
            grad = torch.autograd.grad(objective, cur)[0]
            next_phi = cur - (base.BR_INNER_LR * grad)
            delta = next_phi - base_phi
            norm = torch.linalg.norm(delta)
            if norm > base.LOCAL_BR_RADIUS:
                next_phi = base_phi + (delta * (base.LOCAL_BR_RADIUS / (norm + base.EPS)))
            current = next_phi.detach()
        return current, True


def collect_warmup_with_snapshots(game: TaskAlignedFrozenGame, steps: int) -> list[dict[str, np.ndarray]]:
    env = base.gym.make(game.env_choice.env_id)
    obs, _ = env.reset(seed=base.SEED + 1234)
    snapshots: list[dict[str, np.ndarray]] = []
    for _ in range(steps):
        u = game.rng.uniform(game.env_choice.action_low, game.env_choice.action_high).astype(np.float32)
        w = game.rng.uniform(game.env_choice.action_low, game.env_choice.action_high).astype(np.float32)
        if len(snapshots) < MC_SNAPSHOT_COUNT * 2:
            snapshots.append(
                {
                    "obs": np.asarray(obs, dtype=np.float32).copy(),
                    "qpos": env.unwrapped.data.qpos.copy(),
                    "qvel": env.unwrapped.data.qvel.copy(),
                }
            )
        u_t = torch.as_tensor(u, dtype=base.DTYPE, device=base.DEVICE)
        w_t = torch.as_tensor(w, dtype=base.DTYPE, device=base.DEVICE)
        a_env, _ = game.rarl_env.blend_action(u_t, w_t, game.wrapper_cfg.use_rot_dyn)
        next_obs, reward_raw, terminated, truncated, _ = env.step(a_env.cpu().numpy().astype(np.float32))
        reward_game, reward_task_scaled = game.rarl_env.reward_terms(float(reward_raw), u_t, w_t)
        done = terminated or truncated
        game.replay.add(np.asarray(obs, dtype=np.float32), u, w, reward_game, reward_task_scaled, float(reward_raw), np.asarray(next_obs, dtype=np.float32), done)
        obs = next_obs
        if done:
            obs, _ = env.reset()
    env.close()
    return snapshots[:MC_SNAPSHOT_COUNT]


def mc_quality_task(game: TaskAlignedFrozenGame, snapshots: list[dict[str, np.ndarray]]) -> dict[str, float]:
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
        while disc > (base.GAMMA ** MC_HORIZON):
            u_step = game.actor_action(game.theta, obs_t).squeeze(0)
            w_step = game.actor_action(game.phi, obs_t).squeeze(0)
            a_env, _ = game.rarl_env.blend_action(u_step, w_step, game.wrapper_cfg.use_rot_dyn)
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
    mse = math.nan
    if len(q_task_vals) >= 2 and np.std(q_task_vals) > 1e-12 and np.std(mc_task_vals) > 1e-12:
        corr = float(np.corrcoef(np.asarray(q_task_vals), np.asarray(mc_task_vals))[0, 1])
    if q_task_vals:
        mse = float(np.mean((np.asarray(q_task_vals) - np.asarray(mc_task_vals)) ** 2))
    return {"corr_Q_T_MC_task": corr, "mse_Q_T_MC_task": mse}


def task_critic_update(game: TaskAlignedFrozenGame) -> dict[str, float]:
    batch = game.replay.sample(base.TRAIN_BATCH_SIZE, game.rng)
    with torch.no_grad():
        u_next = game.actor_action(game.theta_target, batch["next_obs"])
        w_next = game.actor_action(game.phi_target, batch["next_obs"])
        target_task = batch["reward_task_scaled"] + (base.GAMMA * (1.0 - batch["done"]) * game.q_task_target(batch["next_obs"], u_next, w_next))
    pred_task = game.q_task(batch["obs"], batch["u"], batch["w"])
    loss_task = torch.mean((pred_task - target_task) ** 2)
    game.q_task_opt.zero_grad(set_to_none=True)
    loss_task.backward()
    grad_norm = math.sqrt(sum(float(torch.sum(p.grad.detach() * p.grad.detach()).item()) for p in game.q_task.parameters() if p.grad is not None))
    game.q_task_opt.step()
    with torch.no_grad():
        for target, online in zip(game.q_task_target.parameters(), game.q_task.parameters()):
            target.data.mul_(1.0 - base.POLYAK_TAU).add_(base.POLYAK_TAU * online.data)
    return {
        "critic_loss_T": float(loss_task.detach().item()),
        "critic_grad_norm_T": float(grad_norm),
        "Q_T_mean": float(pred_task.detach().mean().item()),
        "Q_T_std": float(pred_task.detach().std(unbiased=False).item()),
        "TD_target_T_mean": float(target_task.detach().mean().item()),
        "TD_target_T_std": float(target_task.detach().std(unbiased=False).item()),
    }


@dataclass
class FrozenBundle:
    env_choice: base.EnvChoice
    wrapper_cfg: base.WrapperConfig
    theta0: torch.Tensor
    phi0: torch.Tensor
    q_task_state: dict[str, Any]
    q_task_target_state: dict[str, Any]
    diag_batch: dict[str, torch.Tensor]
    critic_quality: dict[str, float]


def build_frozen_bundle(cfg: base.WrapperConfig) -> FrozenBundle:
    env_choice = make_env_choice()
    game = TaskAlignedFrozenGame(env_choice, cfg, actor_lr=TASK_ALIGNED_LR, seed=base.SEED)
    snapshots = collect_warmup_with_snapshots(game, FROZEN_WARMUP_STEPS)
    quality_rows: list[dict[str, Any]] = []
    for step in range(FROZEN_CRITIC_TRAIN_STEPS):
        stats = task_critic_update(game)
        if step % 200 == 0 or step == FROZEN_CRITIC_TRAIN_STEPS - 1:
            stats = {**stats, **mc_quality_task(game, snapshots), "critic_train_step": step}
            quality_rows.append(stats)
    write_csv(RESULT_ROOT / f"{PREFIX}task_aligned_critic_pretrain_quality.csv", quality_rows)
    diag_batch = game.replay.fixed_state_batch(base.TRAIN_BATCH_SIZE, np.random.default_rng(base.SEED + 7777))
    return FrozenBundle(
        env_choice=env_choice,
        wrapper_cfg=cfg,
        theta0=game.theta.detach().clone(),
        phi0=game.phi.detach().clone(),
        q_task_state=copy.deepcopy(game.q_task.state_dict()),
        q_task_target_state=copy.deepcopy(game.q_task_target.state_dict()),
        diag_batch={k: v.detach().clone() for k, v in diag_batch.items()},
        critic_quality=quality_rows[-1],
    )


def game_from_bundle(bundle: FrozenBundle) -> TaskAlignedFrozenGame:
    game = TaskAlignedFrozenGame(bundle.env_choice, bundle.wrapper_cfg, actor_lr=TASK_ALIGNED_LR, seed=base.SEED)
    game.theta = bundle.theta0.detach().clone()
    game.phi = bundle.phi0.detach().clone()
    game.theta_target = bundle.theta0.detach().clone()
    game.phi_target = bundle.phi0.detach().clone()
    game.q_task.load_state_dict(bundle.q_task_state)
    game.q_task_target.load_state_dict(bundle.q_task_target_state)
    game.metric_refs = {}
    return game


def run_method(bundle: FrozenBundle, method: str, actor_lr: float, num_iters: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    game = game_from_bundle(bundle)
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
            **bundle.wrapper_cfg.__dict__,
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


def write_stage1_check(bundle: FrozenBundle, curves: list[dict[str, Any]], summary: dict[str, Any]) -> bool:
    write_csv(RESULT_ROOT / f"{PREFIX}task_aligned_frozen_sgd_check.csv", curves)
    decision = "TASK_ALIGNMENT_PATCH_BREAKS_SGD"
    passed = bool(summary["valid_flag"] == 1 and summary["curve_normal_flag"] == 1)
    if passed:
        decision = "TASK_ALIGNED_FROZEN_SGD_PASS"
    lines = [
        f"# {PREFIX}task_aligned_frozen_sgd_check_report",
        "",
        f"- decision: `{decision}`",
        f"- actor_lr: `{TASK_ALIGNED_LR}`",
        f"- alpha_dyn: `{bundle.wrapper_cfg.alpha_dyn}`",
        f"- use_rot_dyn: `{bundle.wrapper_cfg.use_rot_dyn}`",
        f"- reward_scale: `{bundle.wrapper_cfg.reward_scale:.6e}`",
        f"- a_u: `{bundle.wrapper_cfg.a_u}`",
        f"- a_w: `{bundle.wrapper_cfg.a_w}`",
        f"- beta_rot: `{bundle.wrapper_cfg.beta_rot}`",
        f"- beta_sym: `{bundle.wrapper_cfg.beta_sym}`",
        f"- critic_pretrain_corr_Q_T_MC_task: `{bundle.critic_quality['corr_Q_T_MC_task']:.6e}`",
        f"- critic_pretrain_mse_Q_T_MC_task: `{bundle.critic_quality['mse_Q_T_MC_task']:.6e}`",
        f"- V_align_start: `{summary['V_align_start']:.6e}`",
        f"- V_align_final: `{summary['V_align_final']:.6e}`",
        f"- P_tau_T_start: `{summary['P_tau_T_start']:.6e}`",
        f"- P_tau_T_final: `{summary['P_tau_T_final']:.6e}`",
        f"- field_norm_start: `{summary['field_norm_start']:.6e}`",
        f"- field_norm_final: `{summary['field_norm_final']:.6e}`",
        f"- robust_br_task_return_raw_final: `{summary['robust_br_task_return_raw_final']:.6e}`",
    ]
    write_text(RESULT_ROOT / f"{PREFIX}task_aligned_frozen_sgd_check_report.md", "\n".join(lines) + "\n")
    return passed


def geometry_report(rows: list[dict[str, Any]]) -> tuple[bool, str]:
    geom_rows = [row for row in rows if base.finite(row["rotation_ratio_proxy"])]
    write_csv(RESULT_ROOT / f"{PREFIX}task_aligned_geometry_audit.csv", geom_rows)
    initial = min(geom_rows, key=lambda row: (row["iteration"], row["method"])) if geom_rows else None
    cross_ok = initial is not None and initial["cross_player_coupling_proxy"] > 0.0 and initial["cross_to_same_ratio"] > 0.05
    noncol_ok = initial is not None and initial["non_collinearity_T"] > 0.2
    rot_ok = initial is not None and initial["rotation_ratio_proxy"] > 1e-6
    lines = [f"# {PREFIX}task_aligned_geometry_audit", ""]
    if initial is not None:
        lines.extend(
            [
                f"- initial_method: `{initial['method']}`",
                f"- initial_iteration: `{initial['iteration']}`",
                f"- ||F_T||: `{initial['field_norm']:.6e}`",
                f"- ||G_T||/||F_T||: `{initial['g_over_f_T']:.6e}`",
                f"- cos(F_T,G_T): `{initial['cos_fg_T']:.6e}`",
                f"- non_collinearity: `{initial['non_collinearity_T']:.6e}`",
                f"- rotation_ratio_proxy: `{initial['rotation_ratio_proxy']:.6e}`",
                f"- cross_player_coupling_proxy: `{initial['cross_player_coupling_proxy']:.6e}`",
                f"- cross_to_same_ratio: `{initial['cross_to_same_ratio']:.6e}`",
            ]
        )
    lines.extend(
        [
            "",
            f"- geometry_gate_cross: `{cross_ok}`",
            f"- geometry_gate_noncollinearity: `{noncol_ok}`",
            f"- geometry_gate_rotation: `{rot_ok}`",
        ]
    )
    write_text(RESULT_ROOT / f"{PREFIX}task_aligned_geometry_audit.md", "\n".join(lines) + "\n")
    return bool(cross_ok and noncol_ok and rot_ok), "\n".join(lines) + "\n"


def plot_baselines(rows: list[dict[str, Any]]) -> None:
    if base.plt is None:
        return
    plot_dir = RESULT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["method"], []).append(row)
    colors = {"sgd": "#1f77b4", "egm": "#ff7f0e", "ppm": "#2ca02c"}
    single_plots = [
        ("V_align", "Task-Aligned Frozen Baseline V_align", f"{PREFIX}task_aligned_baseline_V_align.png"),
        ("normalized_P_tau_T", "Task-Aligned Frozen Baseline P_tau_T", f"{PREFIX}task_aligned_baseline_P_tau_T.png"),
        ("field_norm", "Task-Aligned Frozen Baseline Field Norm", f"{PREFIX}task_aligned_baseline_field_norm.png"),
        ("approximate_exploitability_T", "Task-Aligned Frozen Baseline Exploitability", f"{PREFIX}task_aligned_baseline_exploitability.png"),
        ("clean_task_return_raw", "Task-Aligned Frozen Baseline Clean Task Return", f"{PREFIX}task_aligned_baseline_clean_task_return.png"),
        ("current_adv_task_return_raw", "Task-Aligned Frozen Baseline Current Adv Task Return", f"{PREFIX}task_aligned_baseline_current_adv_task_return.png"),
        ("robust_br_task_return_raw", "Task-Aligned Frozen Baseline Robust BR Task Return", f"{PREFIX}task_aligned_baseline_robust_br_task_return.png"),
        ("robust_degradation", "Task-Aligned Frozen Baseline Robust Degradation", f"{PREFIX}task_aligned_baseline_robust_degradation.png"),
    ]
    for metric, title, name in single_plots:
        fig, ax = base.plt.subplots(figsize=(7, 4))
        for method, method_rows in groups.items():
            ax.plot([row["iteration"] for row in method_rows], [row[metric] for row in method_rows], label=method.upper(), linewidth=1.8, color=colors[method])
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
            ax.plot([row["iteration"] for row in method_rows], [row[metric] for row in method_rows], label=method.upper(), linewidth=1.7, color=colors[method])
        ax.set_title(title)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{PREFIX}task_aligned_baseline_all_plots_big.png", dpi=180)
    base.plt.close(fig)


def main() -> None:
    base.ensure_dirs()
    base.seed_everything(base.SEED)
    already_aligned = task_alignment_audit()
    if already_aligned:
        raise RuntimeError("Unexpected: current protocol detected as already aligned.")

    cfg = make_task_aligned_wrapper(make_env_choice())
    bundle = build_frozen_bundle(cfg)

    sgd_curves, sgd_summary = run_method(bundle, "sgd", TASK_ALIGNED_LR, TASK_ALIGNED_ITERS)
    if not write_stage1_check(bundle, sgd_curves, sgd_summary):
        final_lines = [
            f"# {PREFIX}task_aligned_stage2_final_report",
            "",
            "- decision: `TASK_ALIGNMENT_PATCH_BREAKS_SGD`",
            "- current protocol was not fully task-aligned, and after patching to `Q_T/F_T/P_tau_T/V_align`, frozen SGD no longer met the Stage 1 normality gate.",
        ]
        write_text(RESULT_ROOT / f"{PREFIX}task_aligned_stage2_final_report.md", "\n".join(final_lines) + "\n")
        return

    all_curves = list(sgd_curves)
    summaries = [sgd_summary]
    for method in ["egm", "ppm"]:
        curves, summary = run_method(bundle, method, TASK_ALIGNED_LR, TASK_ALIGNED_ITERS)
        all_curves.extend(curves)
        summaries.append(summary)

    write_csv(RESULT_ROOT / f"{PREFIX}task_aligned_baseline_gate_curves.csv", all_curves)
    write_csv(RESULT_ROOT / f"{PREFIX}task_aligned_baseline_gate.csv", summaries)
    geom_pass, _ = geometry_report(all_curves)
    plot_baselines(all_curves)

    by_method = {row["method"]: row for row in summaries}
    sgd = by_method["sgd"]
    egm = by_method["egm"]
    ppm = by_method["ppm"]
    all_valid = all(int(row["valid_flag"]) == 1 for row in summaries)
    all_normal = all(int(row["curve_normal_flag"]) == 1 for row in summaries)
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
    decision = "BASELINE_READY_FOR_PROPOSED"
    if not geom_pass:
        decision = "BASELINE_FAIL_GEOMETRY_WEAK"
    elif not all_valid:
        decision = "BASELINE_FAIL_RARL_PERFORMANCE"
    elif not all_normal:
        decision = "BASELINE_FAIL_CURVES_ABNORMAL"
    elif winner is None or not comparable_field:
        decision = "BASELINE_FAIL_NO_EGM_PPM_ADVANTAGE"
    elif not robust_ok:
        decision = "BASELINE_FAIL_RARL_PERFORMANCE"

    lines = [
        f"# {PREFIX}task_aligned_baseline_gate_report",
        "",
        f"- decision: `{decision}`",
        f"- actor_lr: `{TASK_ALIGNED_LR}`",
        f"- alpha_dyn: `{cfg.alpha_dyn}`",
        f"- use_rot_dyn: `{cfg.use_rot_dyn}`",
        f"- reward_scale: `{cfg.reward_scale:.6e}`",
        f"- critic_train_steps: `{FROZEN_CRITIC_TRAIN_STEPS}`",
        f"- critic_pretrain_corr_Q_T_MC_task: `{bundle.critic_quality['corr_Q_T_MC_task']:.6e}`",
        "",
    ]
    for row in summaries:
        lines.append(
            f"- {row['method'].upper()}: valid=`{bool(row['valid_flag'])}`, curve_normal=`{bool(row['curve_normal_flag'])}`, "
            f"V_align_AUC=`{row['V_align_AUC']:.6e}`, P_tau_T_AUC=`{row['P_tau_T_AUC']:.6e}`, field_norm_AUC=`{row['field_norm_AUC']:.6e}`, "
            f"robust_br_task_return_raw_final=`{row['robust_br_task_return_raw_final']:.6e}`"
        )
    lines.extend(
        [
            "",
            f"- egm_beats_sgd: `{egm_adv}`",
            f"- ppm_beats_sgd: `{ppm_adv}`",
            f"- winner: `{winner}`",
            f"- winner_field_norm_comparable: `{comparable_field}`",
            f"- winner_robust_br_not_worse_than_sgd: `{robust_ok}`",
            f"- geometry_gate_pass: `{geom_pass}`",
        ]
    )
    write_text(RESULT_ROOT / f"{PREFIX}task_aligned_baseline_gate_report.md", "\n".join(lines) + "\n")

    final_lines = [
        f"# {PREFIX}task_aligned_stage2_final_report",
        "",
        "1. Why online critic failed.",
        "   The online actor field was defined on a moving critic `Q_game`, so `F_k(z)=F(z;omega_k)` drifted while SGD was measured against critic-based residual metrics. That let `V_lambda/P_tau/field_norm` rise even when task returns improved.",
        "2. Why frozen critic fixed SGD normality.",
        "   Freezing the critic and fixing the diagnostic batch made the actor field stationary enough that SGD on the actor block produced clean decreasing curves.",
        "3. Whether the current protocol is truly task-aligned.",
        "   The original frozen protocol was not fully aligned because `V_lambda/P_tau` still used `q_game`. This Stage 2 patch replaces that with task-only `Q_T/F_T/P_tau_T/V_align`.",
        f"4. Whether task-aligned SGD remains normal. `{bool(sgd_summary['curve_normal_flag'])}`.",
        f"5. Whether EGM/PPM outperform SGD under identical lr/config. `EGM={egm_adv}`, `PPM={ppm_adv}`.",
        f"6. Whether geometry is nontrivially rotational/cross-coupled. `{geom_pass}`.",
        f"7. Whether RARL-style performance is aligned with V_align/P_tau_T. `winner_robust_ok={robust_ok}` with task-only robust BR evaluation.",
        f"8. Whether Reacher is ready for proposed. `{decision == 'BASELINE_READY_FOR_PROPOSED'}`.",
        "",
        f"- final_decision: `{decision}`",
    ]
    write_text(RESULT_ROOT / f"{PREFIX}task_aligned_stage2_final_report.md", "\n".join(final_lines) + "\n")


if __name__ == "__main__":
    main()
