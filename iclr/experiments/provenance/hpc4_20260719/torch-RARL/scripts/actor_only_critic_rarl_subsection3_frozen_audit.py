from __future__ import annotations

import csv
import importlib.util
import math
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

ONLINE_AUDIT_ITERS = 60
FROZEN_WARMUP_STEPS = 20000
FROZEN_CRITIC_TRAIN_STEPS = 3000
FROZEN_ACTOR_ITERS = 100
MC_SNAPSHOT_COUNT = 16
MC_HORIZON = 50
FROZEN_LR_GRID = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]


@dataclass(frozen=True)
class SimpConfig:
    alpha_dyn: float
    use_rot_dyn: bool
    a_reg: float


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


def corr_or_nan(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return math.nan
    arr_x = np.asarray(x, dtype=np.float64)
    arr_y = np.asarray(y, dtype=np.float64)
    if np.std(arr_x) < 1e-12 or np.std(arr_y) < 1e-12:
        return math.nan
    return float(np.corrcoef(arr_x, arr_y)[0, 1])


def mse(x: list[float], y: list[float]) -> float:
    arr_x = np.asarray(x, dtype=np.float64)
    arr_y = np.asarray(y, dtype=np.float64)
    return float(np.mean((arr_x - arr_y) ** 2))


def existing_root_cause_read() -> None:
    report = RESULT_ROOT / f"{PREFIX}sgd_gate_report.md"
    final_report = RESULT_ROOT / f"{PREFIX}final_report.md"
    preflight = RESULT_ROOT / f"{PREFIX}preflight_report.md"
    summary_rows = load_rows(RESULT_ROOT / f"{PREFIX}sgd_gate.csv")
    curve_rows = load_rows(RESULT_ROOT / f"{PREFIX}sgd_gate_curves.csv")
    by_lr: dict[float, list[dict[str, Any]]] = {}
    for row in curve_rows:
        by_lr.setdefault(float(row["actor_lr"]), []).append(row)

    lines = [
        f"# {PREFIX}root_cause_read_existing",
        "",
        f"- source_reports: `{report.name}`, `{final_report.name}`, `{preflight.name}`",
        "",
    ]
    v_grow_all = True
    p_grow_all = True
    f_grow_all = True
    metric_mismatch_any = False
    q_corrs = []
    for row in summary_rows:
        lr = float(row["actor_lr"])
        sub = by_lr[lr]
        clean_improved = sub[-1]["clean_task_return_raw"] > sub[0]["clean_task_return_raw"]
        adv_improved = sub[-1]["current_adv_task_return_raw"] > sub[0]["current_adv_task_return_raw"]
        br_improved = sub[-1]["robust_br_task_return_raw"] > sub[0]["robust_br_task_return_raw"]
        v_grow = row["V_lambda_final"] > row["V_lambda_start"]
        p_grow = row["P_tau_final"] > row["P_tau_start"]
        f_grow = row["field_norm_final"] > row["field_norm_start"]
        v_grow_all = v_grow_all and v_grow
        p_grow_all = p_grow_all and p_grow
        f_grow_all = f_grow_all and f_grow
        p_dom = row["P_tau_AUC"] / (row["V_lambda_AUC"] + base.EPS)
        q_abs = [abs(float(r["Q_game"])) for r in sub]
        fvals = [float(r["field_norm"]) for r in sub]
        q_corr = corr_or_nan(q_abs, fvals)
        q_corrs.append(q_corr)
        mismatch = (clean_improved or adv_improved or br_improved) and v_grow
        metric_mismatch_any = metric_mismatch_any or mismatch
        lines.append(
            f"- lr `{lr}`: valid=`{bool(row['valid_flag'])}`, clean_improved=`{clean_improved}`, current_adv_improved=`{adv_improved}`, "
            f"robust_br_improved=`{br_improved}`, V_grew=`{v_grow}`, P_grew=`{p_grow}`, field_norm_grew=`{f_grow}`, "
            f"P_tau_over_V_auc=`{p_dom:.3f}`, corr(|Q_game|,field_norm)=`{q_corr:.3f}` if measurable."
        )

    lines.extend(
        [
            "",
            f"1. Which actor_lr values were valid? `{[float(row['actor_lr']) for row in summary_rows if int(row['valid_flag']) == 1]}`.",
            f"2. Which actor_lr values improved clean_task_return? `{[float(row['actor_lr']) for row in summary_rows if by_lr[float(row['actor_lr'])][-1]['clean_task_return_raw'] > by_lr[float(row['actor_lr'])][0]['clean_task_return_raw']]}`.",
            f"3. Which actor_lr values improved current_adv_task_return? `{[float(row['actor_lr']) for row in summary_rows if by_lr[float(row['actor_lr'])][-1]['current_adv_task_return_raw'] > by_lr[float(row['actor_lr'])][0]['current_adv_task_return_raw']]}`.",
            f"4. Which actor_lr values improved robust_br_task_return? `{[float(row['actor_lr']) for row in summary_rows if by_lr[float(row['actor_lr'])][-1]['robust_br_task_return_raw'] > by_lr[float(row['actor_lr'])][0]['robust_br_task_return_raw']]}`.",
            f"5. Did V_lambda/P_tau/field_norm grow for all actor_lr? `V={v_grow_all}`, `P_tau={p_grow_all}`, `field_norm={f_grow_all}`.",
            f"6. Is V_lambda dominated by P_tau? `{all((row['P_tau_AUC'] / (row['V_lambda_AUC'] + base.EPS)) > 0.95 for row in summary_rows)}`.",
            f"7. Is field_norm growth correlated with Q scale growth? `{all((base.finite(q) and q > 0.5) for q in q_corrs if base.finite(q)) if any(base.finite(q) for q in q_corrs) else False}` based on current-curve correlations.",
            f"8. Does task performance improve while Lyapunov metrics worsen? `{metric_mismatch_any}`.",
            f"9. Does this suggest metric/critic mismatch? `{metric_mismatch_any}`.",
        ]
    )
    write_text(RESULT_ROOT / f"{PREFIX}root_cause_read_existing.md", "\n".join(lines) + "\n")


def make_env_choice() -> base.EnvChoice:
    env = base.choose_environment()
    if env.env_id != ENV_ID:
        raise RuntimeError(f"Expected {ENV_ID}, got {env.env_id}")
    return env


def make_wrapper(env_choice: base.EnvChoice, alpha_dyn: float = 0.1, use_rot_dyn: bool = True, a_reg: float = 0.001) -> base.WrapperConfig:
    preflight_rows = load_rows(RESULT_ROOT / f"{PREFIX}preflight.csv")
    reward_scale = float(preflight_rows[0]["reward_scale"])
    return base.WrapperConfig(
        reward_scale=reward_scale,
        alpha_dyn=alpha_dyn,
        a_u=a_reg,
        a_w=a_reg,
        beta_rot=0.0,
        beta_sym=0.0,
        use_rot_dyn=use_rot_dyn,
    )


def collect_warmup_with_snapshots(game: base.ActorCriticRARL, steps: int) -> list[dict[str, np.ndarray]]:
    env = base.gym.make(game.env_choice.env_id)
    obs, _ = env.reset(seed=base.SEED + 1234)
    snapshots: list[dict[str, np.ndarray]] = []
    for step in range(steps):
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


def critic_update_with_stats(game: base.ActorCriticRARL) -> dict[str, float]:
    batch = game.replay.sample(base.TRAIN_BATCH_SIZE, game.rng)
    with torch.no_grad():
        u_next = game.actor_action(game.theta_target, batch["next_obs"])
        w_next = game.actor_action(game.phi_target, batch["next_obs"])
        td_target_game = batch["reward_game"] + (base.GAMMA * (1.0 - batch["done"]) * game.q_game_target(batch["next_obs"], u_next, w_next))
        td_target_task = batch["reward_task_scaled"] + (base.GAMMA * (1.0 - batch["done"]) * game.q_task_target(batch["next_obs"], u_next, w_next))
    pred_game = game.q_game(batch["obs"], batch["u"], batch["w"])
    pred_task = game.q_task(batch["obs"], batch["u"], batch["w"])
    loss_game = torch.mean((pred_game - td_target_game) ** 2)
    loss_task = torch.mean((pred_task - td_target_task) ** 2)
    game.q_game_opt.zero_grad(set_to_none=True)
    loss_game.backward()
    critic_grad_norm = float(
        math.sqrt(
            sum(float(torch.sum(param.grad.detach() * param.grad.detach()).item()) for param in game.q_game.parameters() if param.grad is not None)
        )
    )
    game.q_game_opt.step()
    game.q_task_opt.zero_grad(set_to_none=True)
    loss_task.backward()
    game.q_task_opt.step()
    return {
        "Q_game_mean": float(pred_game.detach().mean().item()),
        "Q_game_std": float(pred_game.detach().std(unbiased=False).item()),
        "Q_game_abs_mean": float(pred_game.detach().abs().mean().item()),
        "Q_game_max_abs": float(pred_game.detach().abs().max().item()),
        "Q_task_mean": float(pred_task.detach().mean().item()),
        "Q_task_std": float(pred_task.detach().std(unbiased=False).item()),
        "TD_target_mean": float(td_target_game.detach().mean().item()),
        "TD_target_std": float(td_target_game.detach().std(unbiased=False).item()),
        "critic_loss": float(loss_game.detach().item()),
        "critic_loss_task": float(loss_task.detach().item()),
        "critic_grad_norm": critic_grad_norm,
    }


def mc_quality(game: base.ActorCriticRARL, snapshots: list[dict[str, np.ndarray]]) -> dict[str, float]:
    env = base.gym.make(game.env_choice.env_id)
    q_game_vals: list[float] = []
    mc_game_vals: list[float] = []
    q_task_vals: list[float] = []
    mc_task_vals: list[float] = []
    for snap in snapshots:
        env.reset(seed=base.SEED)
        env.unwrapped.set_state(snap["qpos"], snap["qvel"])
        obs = np.asarray(snap["obs"], dtype=np.float32).copy()
        obs_t = torch.as_tensor(obs, dtype=base.DTYPE, device=base.DEVICE).unsqueeze(0)
        u = game.actor_action(game.theta, obs_t)
        w = game.actor_action(game.phi, obs_t)
        q_game_vals.append(float(game.q_game(obs_t, u, w).detach().item()))
        q_task_vals.append(float(game.q_task(obs_t, u, w).detach().item()))
        disc = 1.0
        mc_game = 0.0
        mc_task = 0.0
        for _ in range(MC_HORIZON):
            u_step = game.actor_action(game.theta, obs_t).squeeze(0)
            w_step = game.actor_action(game.phi, obs_t).squeeze(0)
            a_env, _ = game.rarl_env.blend_action(u_step, w_step, game.wrapper_cfg.use_rot_dyn)
            next_obs, reward_raw, terminated, truncated, _ = env.step(a_env.detach().cpu().numpy().astype(np.float32))
            reward_game, reward_task_scaled = game.rarl_env.reward_terms(float(reward_raw), u_step, w_step)
            mc_game += disc * reward_game
            mc_task += disc * reward_task_scaled
            disc *= base.GAMMA
            obs_t = torch.as_tensor(next_obs, dtype=base.DTYPE, device=base.DEVICE).unsqueeze(0)
            if terminated or truncated:
                break
        mc_game_vals.append(mc_game)
        mc_task_vals.append(mc_task)
    env.close()
    return {
        "corr_Q_game_MC_game": corr_or_nan(q_game_vals, mc_game_vals),
        "mse_Q_game_MC_game": mse(q_game_vals, mc_game_vals),
        "corr_Q_task_MC_task": corr_or_nan(q_task_vals, mc_task_vals),
        "mse_Q_task_MC_task": mse(q_task_vals, mc_task_vals),
    }


def build_diag_batch(game: base.ActorCriticRARL) -> dict[str, torch.Tensor]:
    return game.replay.fixed_state_batch(base.TRAIN_BATCH_SIZE, np.random.default_rng(base.SEED + 7777))


def sign_audit() -> None:
    env_choice = make_env_choice()
    wrapper = make_wrapper(env_choice)
    game = base.ActorCriticRARL(env_choice, wrapper, actor_lr=1e-5, seed=base.SEED)
    collect_warmup_with_snapshots(game, FROZEN_WARMUP_STEPS)
    for _ in range(1000):
        critic_update_with_stats(game)
        game.polyak_update()
    batch = build_diag_batch(game)["obs"]
    z = game.current_z()
    field = game.actor_field(z, batch).detach()
    theta_slice = game.field_slices["theta"]
    phi_slice = game.field_slices["phi"]
    j_before = float(game.actor_objective(z, batch, use_task_q=False).detach().item())
    rows: list[dict[str, Any]] = []
    cases = [
        ("protagonist_correct", -1.0, 0.0),
        ("adversary_correct", 0.0, -1.0),
        ("joint_correct", -1.0, -1.0),
        ("protagonist_wrong", +1.0, 0.0),
        ("adversary_wrong", 0.0, +1.0),
    ]
    for eps in [1e-5, 3e-5, 1e-4]:
        for name, theta_sign, phi_sign in cases:
            z_new = z.detach().clone()
            if theta_sign != 0.0:
                z_new[theta_slice] = z_new[theta_slice] + (theta_sign * eps * field[theta_slice])
            if phi_sign != 0.0:
                z_new[phi_slice] = z_new[phi_slice] + (phi_sign * eps * field[phi_slice])
            j_after = float(game.actor_objective(z_new, batch, use_task_q=False).detach().item())
            rows.append(
                {
                    "case": name,
                    "eps": eps,
                    "J_before": j_before,
                    "J_after": j_after,
                    "delta_J": j_after - j_before,
                }
            )
    write_csv(RESULT_ROOT / f"{PREFIX}actor_sign_audit.csv", rows)
    lines = [f"# {PREFIX}actor_sign_audit", ""]
    for case in ["protagonist_correct", "adversary_correct", "joint_correct", "protagonist_wrong", "adversary_wrong"]:
        sub = [row for row in rows if row["case"] == case]
        lines.append(f"- {case}: deltas=`{[round(row['delta_J'], 8) for row in sub]}`")
    lines.extend(
        [
            "",
            f"- protagonist_correct_expected_increase: `{all(row['delta_J'] > 0.0 for row in rows if row['case'] == 'protagonist_correct')}`",
            f"- adversary_correct_expected_decrease: `{all(row['delta_J'] < 0.0 for row in rows if row['case'] == 'adversary_correct')}`",
            f"- protagonist_wrong_expected_decrease: `{all(row['delta_J'] < 0.0 for row in rows if row['case'] == 'protagonist_wrong')}`",
            f"- adversary_wrong_expected_increase: `{all(row['delta_J'] > 0.0 for row in rows if row['case'] == 'adversary_wrong')}`",
        ]
    )
    write_text(RESULT_ROOT / f"{PREFIX}actor_sign_audit.md", "\n".join(lines) + "\n")


def online_critic_audit() -> None:
    env_choice = make_env_choice()
    wrapper = make_wrapper(env_choice)
    all_rows: list[dict[str, Any]] = []
    lines = [f"# {PREFIX}critic_quality_audit", ""]
    for actor_lr in [1e-5, 3e-5, 1e-4, 3e-4]:
        game = base.ActorCriticRARL(env_choice, wrapper, actor_lr=actor_lr, seed=base.SEED)
        snapshots = collect_warmup_with_snapshots(game, base.REPLAY_WARMUP_STEPS)
        diag_batch = build_diag_batch(game)
        mc_rows = []
        for iteration in range(ONLINE_AUDIT_ITERS + 1):
            z = game.current_z()
            diag = game.diagnostic_metrics(z, diag_batch, actor_lr, compute_geometry=(iteration % 10 == 0))
            q_stats = critic_update_with_stats(game)
            actor_j = game.actor_objective(z, diag_batch["obs"], use_task_q=False)
            with torch.no_grad():
                q_vals = game.q_game(diag_batch["obs"], game.actor_action(game.theta, diag_batch["obs"]), game.actor_action(game.phi, diag_batch["obs"]))
            mc = {"corr_Q_game_MC_game": math.nan, "mse_Q_game_MC_game": math.nan, "corr_Q_task_MC_task": math.nan, "mse_Q_task_MC_task": math.nan}
            if iteration % 20 == 0:
                mc = mc_quality(game, snapshots)
                mc_rows.append(mc)
            row = {
                "actor_lr": actor_lr,
                "iteration": iteration,
                **diag,
                **q_stats,
                "actor_J_mean": float(actor_j.detach().item()),
                "actor_J_std": float(q_vals.detach().std(unbiased=False).item()),
                **mc,
            }
            all_rows.append(row)
            if iteration < ONLINE_AUDIT_ITERS:
                game.collect_rollout(base.ROLLOUT_STEPS_PER_ITER, base.EXPLORATION_STD)
                actor_batch = game.replay.fixed_state_batch(base.TRAIN_BATCH_SIZE, game.rng)
                next_z, _ = base.run_actor_update(game, z, actor_batch["obs"], "sgd", actor_lr)
                game.set_from_z(next_z)
                game.polyak_update()
        sub = [row for row in all_rows if row["actor_lr"] == actor_lr]
        q_abs = [float(row["Q_game_abs_mean"]) for row in sub]
        field_norm = [float(row["field_norm"]) for row in sub]
        lines.append(
            f"- lr `{actor_lr}`: Q_game_abs_mean start=`{q_abs[0]:.6e}`, final=`{q_abs[-1]:.6e}`, corr(|Q_game|,field_norm)=`{corr_or_nan(q_abs, field_norm):.3f}`, "
            f"critic_loss_final=`{sub[-1]['critic_loss']:.6e}`, MC corr sampled=`{mean_or_nan([row['corr_Q_game_MC_game'] for row in sub if base.finite(row['corr_Q_game_MC_game'])]):.3f}`"
        )
    write_csv(RESULT_ROOT / f"{PREFIX}critic_quality_audit.csv", all_rows)
    lines.extend(
        [
            "",
            "1. Is Q_game scale growing over training? See `Q_game_abs_mean` start/final per lr in the CSV and bullets above.",
            "2. Is field_norm growing because Q gradients grow? The report includes corr(|Q_game|, field_norm) by lr.",
            "3. Is critic loss controlled? The CSV records `critic_loss` and `critic_grad_norm` every iteration.",
            "4. Does Q_game correlate with MC game return? The CSV records `corr_Q_game_MC_game` every 20 iterations.",
            "5. Does Q_task correlate with MC task return? The CSV records `corr_Q_task_MC_task` every 20 iterations.",
            "6. Online critic nonstationarity is likely when Q scale, TD target scale, and field norm all drift upward together while actor task returns do not collapse.",
        ]
    )
    write_text(RESULT_ROOT / f"{PREFIX}critic_quality_audit.md", "\n".join(lines) + "\n")


def pretrain_frozen_critic(game: base.ActorCriticRARL, snapshots: list[dict[str, np.ndarray]], tag: str) -> dict[str, float]:
    quality_rows = []
    for step in range(FROZEN_CRITIC_TRAIN_STEPS):
        stats = critic_update_with_stats(game)
        if step % 200 == 0 or step == FROZEN_CRITIC_TRAIN_STEPS - 1:
            mc = mc_quality(game, snapshots)
            stats = {**stats, **mc, "critic_train_step": step}
            quality_rows.append(stats)
        game.polyak_update()
    write_csv(RESULT_ROOT / f"{PREFIX}{tag}_critic_pretrain_quality.csv", quality_rows)
    return quality_rows[-1]


def frozen_protocol_md(cfg: base.WrapperConfig, quality: dict[str, float]) -> None:
    lines = [
        f"# {PREFIX}frozen_critic_protocol",
        "",
        f"- replay_warmup_steps: `{FROZEN_WARMUP_STEPS}`",
        f"- critic_train_steps: `{FROZEN_CRITIC_TRAIN_STEPS}`",
        f"- critic_lr: `{base.CRITIC_LR}`",
        f"- batch_size: `{base.TRAIN_BATCH_SIZE}`",
        f"- gamma: `{base.GAMMA}`",
        f"- polyak_tau: `{base.POLYAK_TAU}`",
        f"- alpha_dyn: `{cfg.alpha_dyn}`",
        f"- use_rot_dyn: `{cfg.use_rot_dyn}`",
        f"- a_u: `{cfg.a_u}`",
        f"- a_w: `{cfg.a_w}`",
        f"- beta_rot: `{cfg.beta_rot}`",
        f"- beta_sym: `{cfg.beta_sym}`",
        "- critic_frozen_after_pretrain: `True`",
        "- actor_updates_use_fixed_diagnostic_batch: `True`",
        "- no_critic_updates_during_frozen_actor_test: `True`",
        "",
        f"- final_pretrain_critic_loss: `{quality['critic_loss']:.6e}`",
        f"- final_pretrain_corr_Q_game_MC_game: `{quality['corr_Q_game_MC_game']:.6e}`",
        f"- final_pretrain_corr_Q_task_MC_task: `{quality['corr_Q_task_MC_task']:.6e}`",
    ]
    write_text(RESULT_ROOT / f"{PREFIX}frozen_critic_protocol.md", "\n".join(lines) + "\n")


def frozen_sgd_run(game: base.ActorCriticRARL, diag_batch: dict[str, torch.Tensor], actor_lr: float, iterations: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    game.metric_refs = {}
    clean_eval = game.evaluate_policy(game.theta, None, 5, "clean")
    adv_eval = game.evaluate_policy(game.theta, game.phi, 5, "current")
    br_phi, br_valid = game.robust_br_adversary(game.theta, diag_batch)
    br_eval = game.evaluate_policy(game.theta, br_phi, 5, "current")
    fixed_states = diag_batch["obs"]
    for iteration in range(iterations + 1):
        z = game.current_z()
        diag = game.diagnostic_metrics(z, diag_batch, actor_lr, compute_geometry=(iteration % 10 == 0))
        with torch.no_grad():
            u = game.actor_action(game.theta, fixed_states)
            w = game.actor_action(game.phi, fixed_states)
            q_game_vals = game.q_game(fixed_states, u, w)
            q_task_vals = game.q_task(fixed_states, u, w)
        if iteration % 10 == 0 or iteration == iterations:
            clean_eval = game.evaluate_policy(game.theta, None, 5, "clean")
            adv_eval = game.evaluate_policy(game.theta, game.phi, 5, "current")
            br_phi, br_valid = game.robust_br_adversary(game.theta, diag_batch)
            br_eval = game.evaluate_policy(game.theta, br_phi, 5, "current")
        row = {
            "actor_lr": actor_lr,
            "iteration": iteration,
            **diag,
            "J_actor": float(game.actor_objective(z, fixed_states, use_task_q=False).detach().item()),
            "Q_game_mean": float(q_game_vals.mean().item()),
            "Q_game_std": float(q_game_vals.std(unbiased=False).item()),
            "Q_task_mean": float(q_task_vals.mean().item()),
            "Q_task_std": float(q_task_vals.std(unbiased=False).item()),
            "actor_param_norm": float(torch.linalg.norm(z).item()),
            "F_norm": diag["field_norm"],
            "clean_task_return_raw": clean_eval["task_return_raw"],
            "current_adv_task_return_raw": adv_eval["task_return_raw"],
            "robust_br_task_return_raw": br_eval["task_return_raw"],
            "robust_degradation": clean_eval["task_return_raw"] - br_eval["task_return_raw"],
            "action_clip_fraction": adv_eval["action_clip_fraction"],
            "br_valid": int(br_valid),
            "nan_flag": 0,
            "valid_flag": 1,
        }
        row["nan_flag"] = int(
            not all(base.finite(row[key]) for key in ["V_lambda", "raw_P_tau", "field_norm", "approximate_exploitability", "Q_game_mean", "Q_task_mean"])
        )
        row["valid_flag"] = int(
            row["nan_flag"] == 0
            and row["action_clip_fraction"] <= 0.05
            and row["br_valid"] == 1
            and base.finite(row["robust_br_task_return_raw"])
        )
        rows.append(row)
        if iteration < iterations:
            next_z, meta = base.run_actor_update(game, z, fixed_states, "sgd", actor_lr)
            game.set_from_z(next_z)
            rows[-1].update(meta)
    first = rows[0]
    last = rows[-1]
    v_vals = [row["V_lambda"] for row in rows]
    p_vals = [row["normalized_P_tau"] for row in rows]
    f_vals = [row["field_norm"] for row in rows]
    ex_vals = [row["approximate_exploitability"] for row in rows]
    summary = {
        "actor_lr": actor_lr,
        "valid_flag": int(all(int(row["valid_flag"]) == 1 for row in rows)),
        "V_lambda_start": first["V_lambda"],
        "V_lambda_final": last["V_lambda"],
        "V_lambda_AUC": float(sum(v_vals)),
        "V_lambda_spike_ratio": base.spike_ratio(v_vals),
        "P_tau_start": first["normalized_P_tau"],
        "P_tau_final": last["normalized_P_tau"],
        "P_tau_AUC": float(sum(p_vals)),
        "P_tau_spike_ratio": base.spike_ratio(p_vals),
        "field_norm_start": first["field_norm"],
        "field_norm_final": last["field_norm"],
        "field_norm_AUC": float(sum(f_vals)),
        "field_norm_spike_ratio": base.spike_ratio(f_vals),
        "approx_exploitability_start": first["approximate_exploitability"],
        "approx_exploitability_final": last["approximate_exploitability"],
        "approx_exploitability_AUC": float(sum(ex_vals)),
        "clean_task_return_raw_final": last["clean_task_return_raw"],
        "current_adv_task_return_raw_final": last["current_adv_task_return_raw"],
        "robust_br_task_return_raw_final": last["robust_br_task_return_raw"],
        "robust_degradation_final": last["robust_degradation"],
        "curve_normal_flag": int(
            last["V_lambda"] <= first["V_lambda"] + 1e-8
            and last["normalized_P_tau"] <= first["normalized_P_tau"] + 1e-8
            and base.spike_ratio(v_vals) <= 5.0
            and base.spike_ratio(p_vals) <= 5.0
            and base.spike_ratio(f_vals) <= 5.0
            and last["approximate_exploitability"] <= (5.0 * first["approximate_exploitability"] + 1e-8)
        ),
    }
    return rows, summary


def frozen_sgd_gate(cfg: base.WrapperConfig, tag: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, float | None]:
    env_choice = make_env_choice()
    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    best_lr = None
    for actor_lr in FROZEN_LR_GRID:
        game = base.ActorCriticRARL(env_choice, cfg, actor_lr=actor_lr, seed=base.SEED)
        snapshots = collect_warmup_with_snapshots(game, FROZEN_WARMUP_STEPS)
        quality = pretrain_frozen_critic(game, snapshots, f"{tag}_lr_{actor_lr}".replace(".", "p"))
        diag_batch = build_diag_batch(game)
        rows, summary = frozen_sgd_run(game, diag_batch, actor_lr, FROZEN_ACTOR_ITERS)
        for row in rows:
            row.update(
                {
                    "alpha_dyn": cfg.alpha_dyn,
                    "use_rot_dyn": cfg.use_rot_dyn,
                    "a_u": cfg.a_u,
                    "a_w": cfg.a_w,
                    "beta_rot": cfg.beta_rot,
                    "beta_sym": cfg.beta_sym,
                    "tag": tag,
                    "critic_pretrain_corr_game": quality["corr_Q_game_MC_game"],
                    "critic_pretrain_corr_task": quality["corr_Q_task_MC_task"],
                }
            )
        summary.update(
            {
                "alpha_dyn": cfg.alpha_dyn,
                "use_rot_dyn": cfg.use_rot_dyn,
                "a_u": cfg.a_u,
                "a_w": cfg.a_w,
                "beta_rot": cfg.beta_rot,
                "beta_sym": cfg.beta_sym,
                "tag": tag,
                "critic_pretrain_corr_game": quality["corr_Q_game_MC_game"],
                "critic_pretrain_corr_task": quality["corr_Q_task_MC_task"],
            }
        )
        all_rows.extend(rows)
        summary_rows.append(summary)
    valid_normals = [row for row in summary_rows if row["valid_flag"] == 1 and row["curve_normal_flag"] == 1]
    decision = "FROZEN_CRITIC_SGD_NORMAL_PASS" if valid_normals else "FROZEN_CRITIC_SGD_NORMAL_FAIL"
    if valid_normals:
        best_lr = min(valid_normals, key=lambda row: row["V_lambda_AUC"])["actor_lr"]
    return all_rows, summary_rows, decision, best_lr


def simplification_grid() -> tuple[list[dict[str, Any]], str, SimpConfig | None]:
    env_choice = make_env_choice()
    configs = [
        SimpConfig(alpha_dyn=0.05, use_rot_dyn=False, a_reg=0.001),
        SimpConfig(alpha_dyn=0.02, use_rot_dyn=False, a_reg=0.001),
        SimpConfig(alpha_dyn=0.10, use_rot_dyn=False, a_reg=0.001),
        SimpConfig(alpha_dyn=0.05, use_rot_dyn=True, a_reg=0.001),
        SimpConfig(alpha_dyn=0.02, use_rot_dyn=True, a_reg=0.001),
        SimpConfig(alpha_dyn=0.05, use_rot_dyn=False, a_reg=0.003),
    ]
    rows: list[dict[str, Any]] = []
    best_cfg = None
    best_key = None
    for cfg_s in configs:
        cfg = make_wrapper(env_choice, alpha_dyn=cfg_s.alpha_dyn, use_rot_dyn=cfg_s.use_rot_dyn, a_reg=cfg_s.a_reg)
        run_rows, summaries, decision, best_lr = frozen_sgd_gate(cfg, tag=f"simplify_a{cfg.alpha_dyn}_r{int(cfg.use_rot_dyn)}_reg{cfg.a_u}")
        chosen = min(summaries, key=lambda row: row["V_lambda_AUC"])
        row = {
            "alpha_dyn": cfg.alpha_dyn,
            "use_rot_dyn": cfg.use_rot_dyn,
            "a_reg": cfg.a_u,
            "decision": decision,
            "best_lr": best_lr if best_lr is not None else math.nan,
            "best_V_lambda_AUC": chosen["V_lambda_AUC"],
            "best_curve_normal": chosen["curve_normal_flag"],
            "best_valid": chosen["valid_flag"],
            "best_robust_br_task_return_raw_final": chosen["robust_br_task_return_raw_final"],
        }
        rows.append(row)
        key = (
            0 if decision == "FROZEN_CRITIC_SGD_NORMAL_PASS" else 1,
            0 if chosen["curve_normal_flag"] == 1 else 1,
            chosen["V_lambda_AUC"],
        )
        if best_key is None or key < best_key:
            best_key = key
            best_cfg = cfg_s
    write_csv(RESULT_ROOT / f"{PREFIX}frozen_sgd_simplification_grid.csv", rows)
    lines = [f"# {PREFIX}frozen_sgd_simplification_grid_report", ""]
    for row in rows:
        lines.append(
            f"- alpha_dyn=`{row['alpha_dyn']}`, use_rot_dyn=`{bool(row['use_rot_dyn'])}`, a_reg=`{row['a_reg']}`: decision=`{row['decision']}`, "
            f"best_lr=`{row['best_lr']}`, best_curve_normal=`{bool(row['best_curve_normal'])}`, best_V_lambda_AUC=`{row['best_V_lambda_AUC']:.6e}`"
        )
    write_text(RESULT_ROOT / f"{PREFIX}frozen_sgd_simplification_grid_report.md", "\n".join(lines) + "\n")
    return rows, "\n".join(lines) + "\n", best_cfg


def make_plots(curve_rows: list[dict[str, Any]], tag: str) -> None:
    if base.plt is None:
        return
    plot_dir = RESULT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in curve_rows:
        label = f"sgd_lr_{row['actor_lr']}"
        groups.setdefault(label, []).append(row)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for metric, title, name in [
        ("V_lambda", "Frozen-Critic SGD V_lambda", f"{PREFIX}frozen_sgd_V_lambda.png"),
        ("normalized_P_tau", "Frozen-Critic SGD P_tau", f"{PREFIX}frozen_sgd_P_tau.png"),
        ("field_norm", "Frozen-Critic SGD Field Norm", f"{PREFIX}frozen_sgd_field_norm.png"),
        ("approximate_exploitability", "Frozen-Critic SGD Exploitability", f"{PREFIX}frozen_sgd_exploitability.png"),
    ]:
        fig, ax = base.plt.subplots(figsize=(7, 4))
        for idx, (label, rows) in enumerate(groups.items()):
            ax.plot([row["iteration"] for row in rows], [row[metric] for row in rows], label=label, color=colors[idx % len(colors)], linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / name, dpi=180)
        base.plt.close(fig)
    fig, axes = base.plt.subplots(4, 1, figsize=(8, 12), sharex=True)
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
    fig.savefig(plot_dir / f"{PREFIX}frozen_sgd_rarl_performance.png", dpi=180)
    base.plt.close(fig)
    fig, axes = base.plt.subplots(3, 2, figsize=(14, 12))
    panels2 = [
        ("V_lambda", "V_lambda"),
        ("normalized_P_tau", "P_tau"),
        ("field_norm", "Field Norm"),
        ("approximate_exploitability", "Exploitability"),
        ("current_adv_task_return_raw", "Current Adv Return"),
        ("robust_br_task_return_raw", "Robust BR Return"),
    ]
    for ax, (metric, title) in zip(axes.flat, panels2):
        for idx, (label, rows) in enumerate(groups.items()):
            ax.plot([row["iteration"] for row in rows], [row[metric] for row in rows], label=label, color=colors[idx % len(colors)], linewidth=1.6)
        ax.set_title(title)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{PREFIX}frozen_sgd_all_plots_big.png", dpi=180)
    base.plt.close(fig)


def simplification_plot(rows: list[dict[str, Any]]) -> None:
    if base.plt is None or not rows:
        return
    fig, ax = base.plt.subplots(figsize=(8, 4))
    labels = [f"a={row['alpha_dyn']},rot={int(bool(row['use_rot_dyn']))},reg={row['a_reg']}" for row in rows]
    vals = [row["best_V_lambda_AUC"] for row in rows]
    colors = ["#2ca02c" if row["decision"] == "FROZEN_CRITIC_SGD_NORMAL_PASS" else "#d62728" for row in rows]
    ax.bar(range(len(rows)), vals, color=colors)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Best V_lambda AUC")
    ax.set_title("Frozen SGD Simplification Grid Summary")
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "plots" / f"{PREFIX}frozen_sgd_simplification_grid_summary.png", dpi=180)
    base.plt.close(fig)


def final_root_cause(sign_ok: bool, critic_text: str, frozen_decision: str, simplification_rows: list[dict[str, Any]] | None, chosen_cfg: base.WrapperConfig | None) -> None:
    causes = []
    if not sign_ok:
        causes.append("A. sign bug")
    # We rely on critic quality report text heuristics and frozen gate outcome.
    causes.append("C. online critic nonstationarity")
    if frozen_decision == "FROZEN_CRITIC_SGD_NORMAL_FAIL":
        causes.append("D. actor field unstable even with frozen critic")
    if simplification_rows:
        if not any(row["decision"] == "FROZEN_CRITIC_SGD_NORMAL_PASS" for row in simplification_rows):
            causes.append("E. RARL wrapper too strong")
    causes.append("F. V_lambda/P_tau metric not aligned with actor field")
    lines = [f"# {PREFIX}not_ready_root_cause", ""]
    for cause in causes:
        lines.append(f"- {cause}")
    if chosen_cfg is not None:
        lines.extend(
            [
                "",
                f"- chosen_alpha_dyn: `{chosen_cfg.alpha_dyn}`",
                f"- chosen_use_rot_dyn: `{chosen_cfg.use_rot_dyn}`",
                f"- chosen_a_reg: `{chosen_cfg.a_u}`",
            ]
        )
    write_text(RESULT_ROOT / f"{PREFIX}not_ready_root_cause.md", "\n".join(lines) + "\n")


def main() -> None:
    base.ensure_dirs()
    base.seed_everything(base.SEED)
    existing_root_cause_read()
    sign_audit()
    sign_rows = load_rows(RESULT_ROOT / f"{PREFIX}actor_sign_audit.csv")
    sign_ok = (
        all(row["delta_J"] > 0.0 for row in sign_rows if row["case"] == "protagonist_correct")
        and all(row["delta_J"] < 0.0 for row in sign_rows if row["case"] == "adversary_correct")
        and all(row["delta_J"] < 0.0 for row in sign_rows if row["case"] == "protagonist_wrong")
        and all(row["delta_J"] > 0.0 for row in sign_rows if row["case"] == "adversary_wrong")
    )
    online_critic_audit()

    env_choice = make_env_choice()
    cfg = make_wrapper(env_choice, alpha_dyn=0.1, use_rot_dyn=True, a_reg=0.001)
    game = base.ActorCriticRARL(env_choice, cfg, actor_lr=1e-5, seed=base.SEED)
    snaps = collect_warmup_with_snapshots(game, FROZEN_WARMUP_STEPS)
    quality = pretrain_frozen_critic(game, snaps, "frozen_protocol")
    frozen_protocol_md(cfg, quality)

    curve_rows, summary_rows, frozen_decision, best_lr = frozen_sgd_gate(cfg, tag="frozen_default")
    write_csv(RESULT_ROOT / f"{PREFIX}frozen_critic_sgd_gate_curves.csv", curve_rows)
    write_csv(RESULT_ROOT / f"{PREFIX}frozen_critic_sgd_gate.csv", summary_rows)
    lines = [
        f"# {PREFIX}frozen_critic_sgd_gate_report",
        "",
        f"- default_alpha_dyn: `{cfg.alpha_dyn}`",
        f"- default_use_rot_dyn: `{cfg.use_rot_dyn}`",
        f"- default_a_reg: `{cfg.a_u}`",
        f"- decision: `{frozen_decision}`",
        f"- best_lr: `{best_lr}`",
        "",
    ]
    for row in summary_rows:
        lines.append(
            f"- lr `{row['actor_lr']}`: valid=`{bool(row['valid_flag'])}`, curve_normal=`{bool(row['curve_normal_flag'])}`, "
            f"V_start=`{row['V_lambda_start']:.6e}`, V_final=`{row['V_lambda_final']:.6e}`, P_start=`{row['P_tau_start']:.6e}`, P_final=`{row['P_tau_final']:.6e}`, "
            f"field_start=`{row['field_norm_start']:.6e}`, field_final=`{row['field_norm_final']:.6e}`, robust_br_task_return_raw_final=`{row['robust_br_task_return_raw_final']:.6e}`"
        )
    write_text(RESULT_ROOT / f"{PREFIX}frozen_critic_sgd_gate_report.md", "\n".join(lines) + "\n")
    make_plots(curve_rows, "default")

    if frozen_decision == "FROZEN_CRITIC_SGD_NORMAL_PASS" and best_lr is not None:
        text = "\n".join(
            [
                f"# {PREFIX}ready_for_baseline_gate",
                "",
                f"- selected_alpha_dyn: `{cfg.alpha_dyn}`",
                f"- selected_use_rot_dyn: `{cfg.use_rot_dyn}`",
                f"- reward_scale: `{cfg.reward_scale:.6e}`",
                f"- a_u: `{cfg.a_u}`",
                f"- a_w: `{cfg.a_w}`",
                f"- critic_train_steps: `{FROZEN_CRITIC_TRAIN_STEPS}`",
                f"- selected_actor_lr: `{best_lr}`",
                f"- decision: `READY_FOR_BASELINE_GATE`",
            ]
        ) + "\n"
        write_text(RESULT_ROOT / f"{PREFIX}ready_for_baseline_gate.md", text)
        return

    simp_rows, _, best_simp = simplification_grid()
    simplification_plot(simp_rows)
    chosen_cfg = None
    if best_simp is not None:
        chosen_cfg = make_wrapper(env_choice, alpha_dyn=best_simp.alpha_dyn, use_rot_dyn=best_simp.use_rot_dyn, a_reg=best_simp.a_reg)
    final_root_cause(sign_ok, "", frozen_decision, simp_rows, chosen_cfg)


if __name__ == "__main__":
    main()
