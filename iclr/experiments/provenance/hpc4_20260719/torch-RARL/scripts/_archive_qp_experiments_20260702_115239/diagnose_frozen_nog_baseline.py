from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "final_actor_game_paper_ready_stabilization.py"
REPO_ROOT = SCRIPT_DIR.parents[1]
RESULT_ROOT = REPO_ROOT.parent / "results" / "diagnose_frozen_nog_baseline"


def load_base():
    spec = importlib.util.spec_from_file_location("paper_ready_base", BASE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_base()
torch = base.torch
gym = base.gym
plt = base.plt


ENV_ID = "HalfCheetah-v4"
RHO = 5.0
LAMBDA_U = 0.01
LAMBDA_W = 0.05
JOINT_LR = 3e-4
LAMBDA_F = 0.0003
LAMBDA_J = 1.0
SEED = 0
BATCH_SIZE = 2048
TRAIN_DATASET_SIZE = 16384
EVAL_DATASET_SIZE = 8192
NOG_STEPS = 50
TRAIN_ITERATIONS = 100
QP_ITERATIONS = 200
EVAL_FREQ = 10
SMALL_FIELD_COEFFS = [1e-4, 1e-3]


def ensure_dir(path: Path) -> Path:
    return base.ensure_dir(path)


def write_text(path: Path, text: str) -> None:
    base.write_text(path, text)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    base.write_csv(path, rows)


def safe_float(value: Any, default: float = math.nan) -> float:
    return base.safe_float(value, default)


def finite(value: Any) -> bool:
    return base.finite(value)


def auc(values: list[float]) -> float:
    return base.auc(values)


def config() -> Any:
    return base.Config(
        env_id=ENV_ID,
        rho=RHO,
        lambda_u=LAMBDA_U,
        lambda_w=LAMBDA_W,
        lr=JOINT_LR,
        lambda_F=LAMBDA_F,
        lambda_J=LAMBDA_J,
        batch_size=BATCH_SIZE,
    )


def collect_datasets(spec: Any) -> tuple[Any, Any]:
    return base.collect_state_dataset(spec, SEED, TRAIN_DATASET_SIZE, EVAL_DATASET_SIZE)


def make_game(spec: Any, train_states: Any, eval_states: Any, seed: int = SEED) -> Any:
    return base.MujocoStateActorGame(spec, config(), train_states, eval_states, seed)


def current_merit_components(game: Any, z: Any, obs: Any) -> dict[str, float]:
    terms = game.objective_terms(z, obs)
    field = game.field(z, obs).detach()
    field_energy = 0.5 * float(torch.dot(field, field).item())
    j_value = float(terms["J"].detach().item())
    if "field0" not in game.metric_refs:
        game.metric_refs["field0"] = max(field_energy, base.EPS)
        game.metric_refs["negJ0"] = max(abs(-j_value), base.EPS)
    field_term = field_energy / (game.metric_refs["field0"] + base.EPS)
    negJ_norm = (-j_value) / (game.metric_refs["negJ0"] + base.EPS)
    rot_term = (-game.cfg.rho * float(terms["rot_norm"].detach().item())) / (game.metric_refs["negJ0"] + base.EPS)
    u_term = (game.cfg.lambda_u * float(terms["u_energy"].detach().item())) / (game.metric_refs["negJ0"] + base.EPS)
    w_term = (-game.cfg.lambda_w * float(terms["w_energy"].detach().item())) / (game.metric_refs["negJ0"] + base.EPS)
    return {
        "field_energy": field_energy,
        "field_term": field_term,
        "actor_game_score": j_value,
        "negJ_term": negJ_norm,
        "rot_raw": float(terms["rot"].detach().item()),
        "rot_norm": float(terms["rot_norm"].detach().item()),
        "u_energy": float(terms["u_energy"].detach().item()),
        "w_energy": float(terms["w_energy"].detach().item()),
        "linear_field_component_value": game.cfg.lambda_F * field_term,
        "linear_actor_component_value": game.cfg.lambda_J * negJ_norm,
        "linear_rot_component_value": game.cfg.lambda_J * rot_term,
        "linear_u_component_value": game.cfg.lambda_J * u_term,
        "linear_w_component_value": game.cfg.lambda_J * w_term,
        "V": (game.cfg.lambda_F * field_term) + (game.cfg.lambda_J * negJ_norm),
    }


def variant_metrics(game: Any, z: Any, obs: Any, variant: str, score_field_coeff: float | None = None) -> dict[str, float]:
    cur = current_merit_components(game, z, obs)
    if variant == "current":
        out = dict(cur)
    elif variant == "field_only":
        out = dict(cur)
        out["V"] = cur["field_energy"]
    elif variant == "actualV_linesearch":
        out = dict(cur)
    elif variant == "score_oriented":
        coeff = 1e-4 if score_field_coeff is None else score_field_coeff
        out = dict(cur)
        out["V"] = (-cur["actor_game_score"]) + (coeff * cur["field_energy"])
    else:
        raise ValueError(f"Unknown variant {variant}")
    out["variant"] = variant
    if score_field_coeff is not None:
        out["score_field_coeff"] = score_field_coeff
    return out


def q1d_fit_from_metric_sequence(v0: float, v1: float, v2: float, db: float) -> tuple[float, float]:
    return base.fit_quadratic_1d(v0, v1, v2, db)


def q1d_diagnostics(game: Any, z: Any, obs: Any, eta: float) -> dict[str, float]:
    z_req = z.detach().clone().requires_grad_(True)
    f0 = game.field(z_req, obs).detach()
    db = max(float(eta), base.EPS)

    def metrics_at(beta: float) -> dict[str, float]:
        z_cand = base.apply_delta(z, -beta * f0)
        return current_merit_components(game, z_cand, obs)

    m0 = current_merit_components(game, z, obs)
    m1 = metrics_at(db)
    m2 = metrics_at(2.0 * db)
    l_beta, h_bb = q1d_fit_from_metric_sequence(m0["V"], m1["V"], m2["V"], db)
    beta_raw = (-l_beta / (h_bb + 1e-8)) if abs(h_bb + 1e-8) > base.EPS else 0.0
    beta = max(beta_raw, 0.0)
    beta_max = 10.0 * eta
    beta_clipped = min(beta, beta_max)
    q_at_beta = l_beta * beta_clipped + 0.5 * h_bb * beta_clipped * beta_clipped

    def component_linear(key: str) -> float:
        l_comp, _ = q1d_fit_from_metric_sequence(m0[key], m1[key], m2[key], db)
        return float(l_comp)

    return {
        "beta_unclipped": float(beta_raw),
        "beta_clipped": float(beta_clipped),
        "beta_max": float(beta_max),
        "q1D_linear_coeff": float(l_beta),
        "q1D_quadratic_coeff": float(h_bb),
        "q1D_at_0": 0.0,
        "q1D_at_beta": float(q_at_beta),
        "predicted_drift_noG": float(q_at_beta),
        "linear_from_field_energy": component_linear("linear_field_component_value"),
        "linear_from_actor_game_score": component_linear("linear_actor_component_value"),
        "linear_from_rot": component_linear("linear_rot_component_value"),
        "linear_from_u_energy": component_linear("linear_u_component_value"),
        "linear_from_w_energy": component_linear("linear_w_component_value"),
    }


def run_nog_variant_step(
    game: Any,
    z: Any,
    obs: Any,
    eta: float,
    variant: str,
    score_field_coeff: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    z_req = z.detach().clone().requires_grad_(True)
    f0 = game.field(z_req, obs).detach()
    before = variant_metrics(game, z, obs, variant, score_field_coeff)

    if variant in {"current", "field_only", "score_oriented"}:
        def point(beta: float) -> float:
            z_cand = base.apply_delta(z, -beta * f0)
            return variant_metrics(game, z_cand, obs, variant, score_field_coeff)["V"]

        nog = base.solve_nog(point, before["V"], eta, beta_max=10.0 * eta)
        beta = float(nog["beta"])
        meta = {
            "beta": beta,
            "beta_unclipped": float(nog["beta_raw"]),
            "beta_clipped": beta,
            "beta_active": int(beta > 1e-12),
            "selected_beta_source": "closed_form",
        }
    elif variant == "actualV_linesearch":
        beta_candidates = [eta, 0.5 * eta, 0.25 * eta, 0.125 * eta, 0.0625 * eta, 0.0]
        best_beta = 0.0
        best_v = before["V"]
        for beta in beta_candidates:
            z_cand = base.apply_delta(z, -beta * f0)
            v_cand = current_merit_components(game, z_cand, obs)["V"]
            if v_cand < best_v - 1e-12:
                best_v = v_cand
                best_beta = beta
        meta = {
            "beta": float(best_beta),
            "beta_unclipped": float(best_beta),
            "beta_clipped": float(best_beta),
            "beta_active": int(best_beta > 1e-12),
            "selected_beta_source": "actualV_linesearch",
        }
    else:
        raise ValueError(f"Unsupported variant {variant}")

    delta = -meta["beta"] * f0
    z_next = base.apply_delta(z, delta)
    after = variant_metrics(game, z_next, obs, variant, score_field_coeff)
    meta.update(
        {
            "update_norm": float(torch.linalg.norm(delta).item()),
            "actual_V_before": float(before["V"]),
            "actual_V_after": float(after["V"]),
            "actual_drift": float(after["V"] - before["V"]),
        }
    )
    return z_next, meta


def run_selected_qp(game: Any, z: Any, obs: Any, eta: float, selected_nog_variant: str, score_field_coeff: float | None = None) -> tuple[Any, dict[str, Any]]:
    z_nog, meta_nog = run_nog_variant_step(game, z, obs, eta, selected_nog_variant, score_field_coeff)
    z_full, meta_full, f_det, g_det, _ = base.qp_candidate_meta(game, z, obs, eta, normalize_g=False)
    v_nog = current_merit_components(game, z_nog, obs)["V"]
    tol = base.NO_G_SAFE_TOL * max(1.0, abs(v_nog))
    best_choice = {
        "kind": "noG",
        "eta": 0.0,
        "z": z_nog,
        "V": v_nog,
        "G_ratio": 0.0,
        "margin": 0.0,
        "beta": safe_float(meta_nog["beta"]),
        "gamma": 0.0,
        "gamma_active": 0,
        "fallback_reason": "gamma_inactive" if int(meta_full["gamma_active"]) == 0 else "selected_noG_by_actual_V",
        "chosen_step_type": "noG",
    }
    beta = safe_float(meta_full["beta"])
    gamma = safe_float(meta_full["gamma"])
    f_norm = float(torch.linalg.norm(f_det).item())
    fallback_reason = "gamma_inactive" if gamma <= 1e-12 else "no_eta_improves_actual_V"
    for eta_scale in base.ETA_LIST:
        delta = (-beta * f_det) + (eta_scale * gamma * g_det)
        z_eta = base.apply_delta(z, delta)
        v_eta = current_merit_components(game, z_eta, obs)["V"]
        g_term_norm = abs(eta_scale * gamma) * float(torch.linalg.norm(g_det).item())
        f_term_norm = abs(beta) * f_norm
        g_ratio = g_term_norm / (f_term_norm + g_term_norm + base.EPS)
        if v_eta <= (v_nog - tol) and v_eta < best_choice["V"] - 1e-12:
            best_choice = {
                "kind": "dampedG",
                "eta": float(eta_scale),
                "z": z_eta,
                "V": float(v_eta),
                "G_ratio": float(g_ratio),
                "margin": float(v_nog - v_eta),
                "beta": beta,
                "gamma": eta_scale * gamma,
                "gamma_active": int(gamma > 1e-12),
                "fallback_reason": "accept",
                "chosen_step_type": "dampedG",
            }
            fallback_reason = "accept"
    z_best = best_choice["z"]
    meta = {
        "update_norm": float(torch.linalg.norm(z_best - z).item()),
        "beta": float(best_choice["beta"]),
        "gamma": float(best_choice["gamma"]),
        "gamma_active": int(best_choice["gamma_active"]),
        "G_contribution_ratio": float(best_choice["G_ratio"]),
        "fallback_to_noG": int(best_choice["kind"] == "noG"),
        "predicted_inclusion_pass": int(meta_full["predicted_inclusion_pass"]),
        "selected_active_set": str(meta_full["selected_active_set"]),
        "V_before": float(current_merit_components(game, z, obs)["V"]),
        "V_after_candidate": float(best_choice["V"]),
        "cos_F_G": float(meta_full["cos_F_G"]),
        "non_collinearity": float(meta_full["non_collinearity"]),
        "chosen_eta": float(best_choice["eta"]),
        "fallback_reason": str(fallback_reason),
        "QP_accept_frac_given_gamma_active": float(best_choice["kind"] == "dampedG"),
        "QP_better_than_noG_actual_V": int(best_choice["V"] <= v_nog + 1e-12),
    }
    return z_best, meta


def actor_delta_on_eval(game: Any, z_before: Any, z_after: Any, obs: Any) -> float:
    return base.actor_mean_delta_norm_on_dataset(game, z_before, z_after, obs)


def plot_simple_lines(plot_path: Path, rows_by_label: dict[str, list[dict[str, Any]]], metric: str, title: str) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, rows in rows_by_label.items():
        xs = [safe_float(r["iteration"]) for r in rows]
        ys = [safe_float(r[metric]) for r in rows]
        ax.plot(xs, ys, label=label)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def plot_nog_steps(plot_path: Path, rows: list[dict[str, Any]]) -> None:
    if plt is None:
        return
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    xs = [r["step"] for r in rows]
    axes[0].plot(xs, [safe_float(r["beta_noG"]) for r in rows], label="beta_noG")
    axes[0].plot(xs, [safe_float(r["beta_unclipped"]) for r in rows], label="beta_unclipped")
    axes[0].plot(xs, [safe_float(r["beta_clipped"]) for r in rows], label="beta_clipped")
    axes[0].set_title("noG beta over steps")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(xs, [safe_float(r["predicted_drift_noG"]) for r in rows], label="predicted_drift")
    axes[1].plot(xs, [safe_float(r["actual_drift_noG"]) for r in rows], label="actual_drift")
    axes[1].plot(xs, [safe_float(r["q1D_linear_coeff"]) for r in rows], label="q1D_linear")
    axes[1].plot(xs, [safe_float(r["q1D_quadratic_coeff"]) for r in rows], label="q1D_quadratic")
    axes[1].set_title("noG drift diagnostics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def diagnose_nog(spec: Any, train_states: Any, eval_states: Any) -> tuple[str, list[dict[str, Any]], str]:
    game = make_game(spec, train_states, eval_states, SEED)
    z = game.init_z(seed_offset=SEED)
    game.metric_refs = {}
    eval_obs = game.eval_batch()
    rows: list[dict[str, Any]] = []
    nonnegative_linear = 0
    negative_linear_zero_beta = 0
    positive_beta_zero_delta = 0
    for step in range(NOG_STEPS):
        obs = game.sample_batch(BATCH_SIZE, 700000 + step)
        before = current_merit_components(game, z, eval_obs)
        fit = q1d_diagnostics(game, z, obs, JOINT_LR)
        z_next, meta = run_nog_variant_step(game, z, obs, JOINT_LR, "current")
        after = current_merit_components(game, z_next, eval_obs)
        param_delta_norm = float(torch.linalg.norm(z_next - z).item())
        actor_delta_norm = actor_delta_on_eval(game, z, z_next, eval_obs)
        row = {
            "step": step,
            "beta_noG": safe_float(meta["beta"]),
            "beta_unclipped": safe_float(fit["beta_unclipped"]),
            "beta_clipped": safe_float(fit["beta_clipped"]),
            "beta_active": int(safe_float(meta["beta"]) > 1e-12),
            "beta_max": safe_float(fit["beta_max"]),
            "q1D_linear_coeff": safe_float(fit["q1D_linear_coeff"]),
            "q1D_quadratic_coeff": safe_float(fit["q1D_quadratic_coeff"]),
            "q1D_at_0": 0.0,
            "q1D_at_beta": safe_float(fit["q1D_at_beta"]),
            "predicted_drift_noG": safe_float(fit["predicted_drift_noG"]),
            "actual_V_before": before["V"],
            "actual_V_after_noG": after["V"],
            "actual_drift_noG": after["V"] - before["V"],
            "field_energy_before": before["field_energy"],
            "field_energy_after": after["field_energy"],
            "actor_game_score_before": before["actor_game_score"],
            "actor_game_score_after": after["actor_game_score"],
            "rot_before": before["rot_norm"],
            "rot_after": after["rot_norm"],
            "u_energy_before": before["u_energy"],
            "u_energy_after": after["u_energy"],
            "w_energy_before": before["w_energy"],
            "w_energy_after": after["w_energy"],
            "parameter_delta_norm": param_delta_norm,
            "actor_mean_delta_norm_on_D_eval": actor_delta_norm,
            "linear_from_field_energy": fit["linear_from_field_energy"],
            "linear_from_actor_game_score": fit["linear_from_actor_game_score"],
            "linear_from_rot": fit["linear_from_rot"],
            "linear_from_u_energy": fit["linear_from_u_energy"],
            "linear_from_w_energy": fit["linear_from_w_energy"],
        }
        rows.append(row)
        nonnegative_linear += int(row["q1D_linear_coeff"] >= 0.0)
        negative_linear_zero_beta += int(row["q1D_linear_coeff"] < 0.0 and row["beta_clipped"] <= 1e-12)
        positive_beta_zero_delta += int(row["beta_clipped"] > 1e-12 and row["parameter_delta_norm"] <= 1e-12)
        z = z_next
    frac_nonneg = nonnegative_linear / max(len(rows), 1)
    frac_zero_from_quad = negative_linear_zero_beta / max(len(rows), 1)
    frac_apply_bug = positive_beta_zero_delta / max(len(rows), 1)
    if frac_nonneg >= 0.6:
        decision = "NOG_ZERO_BECAUSE_NOT_DESCENT_FOR_V"
    elif frac_zero_from_quad >= 0.6:
        decision = "NOG_ZERO_BECAUSE_QUADRATIC_OR_SCALE_BUG"
    elif frac_apply_bug >= 0.2:
        decision = "NOG_ACTIVE_BUT_APPLY_BUG"
    else:
        decision = "NOG_ZERO_BECAUSE_QUADRATIC_OR_SCALE_BUG"
    lines = [
        "# noG freeze diagnosis report",
        "",
        f"- env: `{ENV_ID}`",
        f"- config: `{config().slug}`",
        f"- frac_nonnegative_q1D_linear: `{frac_nonneg:.3f}`",
        f"- frac_negative_linear_but_zero_beta: `{frac_zero_from_quad:.3f}`",
        f"- frac_positive_beta_but_zero_delta: `{frac_apply_bug:.3f}`",
        f"- final decision: `{decision}`",
        "",
    ]
    return decision, rows, "\n".join(lines)


def run_method_variant(spec: Any, train_states: Any, eval_states: Any, method: str, iterations: int, variant_name: str | None = None, score_field_coeff: float | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    game = make_game(spec, train_states, eval_states, SEED)
    z = game.init_z(seed_offset=SEED)
    game.metric_refs = {}
    eval_obs = game.eval_batch()
    curves: list[dict[str, Any]] = []
    beta_active = 0
    nonzero_delta = 0
    actor_delta_nonzero = 0
    drifts: list[float] = []
    fallback = 0
    gamma_active_steps = 0
    accept_given_gamma = 0
    g_ratio_sum = 0.0
    for iteration in range(iterations):
        obs = game.sample_batch(BATCH_SIZE, 100000 + iteration)
        z_before = z
        if method == "sgd_gda":
            z, meta = base.run_sgd(game, z, obs, JOINT_LR)
        elif method == "egm":
            z, meta = base.run_egm(game, z, obs, JOINT_LR)
        elif method == "proposed_nog_closed_current":
            z, meta = run_nog_variant_step(game, z, obs, JOINT_LR, "current")
        elif method == "proposed_nog_field_only":
            z, meta = run_nog_variant_step(game, z, obs, JOINT_LR, "field_only")
        elif method == "proposed_nog_actualV_linesearch":
            z, meta = run_nog_variant_step(game, z, obs, JOINT_LR, "actualV_linesearch")
        elif method == "proposed_nog_score_oriented":
            z, meta = run_nog_variant_step(game, z, obs, JOINT_LR, "score_oriented", score_field_coeff)
        elif method == "proposed_qp_dampedG_nog_safe":
            selected_variant = variant_name if variant_name is not None else "current"
            z, meta = run_selected_qp(game, z, obs, JOINT_LR, selected_variant, score_field_coeff)
        else:
            raise ValueError(method)
        metrics = current_merit_components(game, z, eval_obs)
        param_delta_norm = float(torch.linalg.norm(z - z_before).item())
        actor_delta_norm = actor_delta_on_eval(game, z_before, z, eval_obs)
        beta_val = safe_float(meta.get("beta", 0.0), 0.0)
        beta_active += int(abs(beta_val) > 1e-12)
        nonzero_delta += int(param_delta_norm > 1e-12)
        actor_delta_nonzero += int(actor_delta_norm > 1e-12)
        drifts.append(safe_float(meta.get("actual_drift", meta.get("V_after_candidate", math.nan) - meta.get("V_before", math.nan))))
        fallback += int(meta.get("fallback_to_noG", 0))
        gamma_active_steps += int(meta.get("gamma_active", 0))
        accept_given_gamma += int(meta.get("gamma_active", 0) == 1 and meta.get("fallback_to_noG", 1) == 0)
        g_ratio_sum += safe_float(meta.get("G_contribution_ratio", 0.0), 0.0)
        if iteration % EVAL_FREQ == 0 or iteration == iterations - 1:
            curves.append(
                {
                    "iteration": iteration,
                    "method": method if score_field_coeff is None else f"{method}_sf{score_field_coeff}",
                    "actor_game_score": metrics["actor_game_score"],
                    "field_norm": math.sqrt(max(metrics["field_energy"] * 2.0, 0.0)),
                    "Lyapunov": metrics["V"],
                    "rot_norm": metrics["rot_norm"],
                    "u_energy": metrics["u_energy"],
                    "w_energy": metrics["w_energy"],
                    "beta": beta_val,
                    "parameter_delta_norm": param_delta_norm,
                    "actor_mean_delta_norm_on_D_eval": actor_delta_norm,
                    "fallback_to_noG": int(meta.get("fallback_to_noG", 0)),
                    "gamma_active": int(meta.get("gamma_active", 0)),
                    "G_contribution_ratio": safe_float(meta.get("G_contribution_ratio", 0.0), 0.0),
                    "finite_flag": int(all(finite(v) for v in [metrics["actor_game_score"], metrics["field_energy"], metrics["V"], param_delta_norm])),
                }
            )
    summary = {
        "method": method if score_field_coeff is None else f"{method}_sf{score_field_coeff}",
        "variant_name": variant_name or method,
        "score_field_coeff": score_field_coeff if score_field_coeff is not None else "",
        "actor_game_score_AUC": auc([safe_float(r["actor_game_score"]) for r in curves]),
        "field_norm_AUC": auc([safe_float(r["field_norm"]) for r in curves]),
        "Lyapunov_AUC": auc([safe_float(r["Lyapunov"]) for r in curves]),
        "actor_game_score_final": safe_float(curves[-1]["actor_game_score"]) if curves else math.nan,
        "beta_active_frac": beta_active / max(iterations, 1),
        "nonzero_parameter_delta_frac": nonzero_delta / max(iterations, 1),
        "actor_mean_delta_nonzero_frac": actor_delta_nonzero / max(iterations, 1),
        "actual_drift_noG_mean": float(np.nanmean(np.asarray(drifts, dtype=np.float64))) if drifts else math.nan,
        "fallback_to_noG_frac": fallback / max(iterations, 1),
        "gamma_active_frac": gamma_active_steps / max(iterations, 1),
        "QP_accept_frac_given_gamma_active": (accept_given_gamma / max(gamma_active_steps, 1)) if gamma_active_steps else 0.0,
        "effective_G_contribution_ratio": g_ratio_sum / max(iterations, 1),
        "curve_sanity_flag": int(all(int(r["finite_flag"]) == 1 for r in curves)),
    }
    return curves, summary


def choose_usable_nog(spec: Any, train_states: Any, eval_states: Any) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], str, float | None]:
    methods = [
        ("sgd_gda", None),
        ("egm", None),
        ("proposed_nog_closed_current", None),
        ("proposed_nog_field_only", None),
        ("proposed_nog_actualV_linesearch", None),
    ]
    curves_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for method, coeff in methods:
        curves, summary = run_method_variant(spec, train_states, eval_states, method, TRAIN_ITERATIONS)
        curves_rows.extend(curves)
        summary_rows.append(summary)
    candidate_rows = [r for r in summary_rows if r["method"].startswith("proposed_nog")]
    usable = [
        r for r in candidate_rows
        if r["beta_active_frac"] >= 0.50
        and r["nonzero_parameter_delta_frac"] >= 0.50
        and r["actor_mean_delta_nonzero_frac"] >= 0.50
        and r["curve_sanity_flag"] == 1
        and (r["actor_game_score_AUC"] >= max([x["actor_game_score_AUC"] for x in summary_rows if x["method"] == "sgd_gda"][0] * 1.05, -math.inf)
             or finite(r["actor_game_score_AUC"]))
    ]
    selected_variant = ""
    selected_coeff: float | None = None
    if not usable:
        for coeff in SMALL_FIELD_COEFFS:
            curves, summary = run_method_variant(spec, train_states, eval_states, "proposed_nog_score_oriented", TRAIN_ITERATIONS, score_field_coeff=coeff)
            curves_rows.extend(curves)
            summary_rows.append(summary)
        candidate_rows = [r for r in summary_rows if str(r["method"]).startswith("proposed_nog")]
        usable = [
            r for r in candidate_rows
            if r["beta_active_frac"] >= 0.50
            and r["nonzero_parameter_delta_frac"] >= 0.50
            and r["actor_mean_delta_nonzero_frac"] >= 0.50
            and r["curve_sanity_flag"] == 1
        ]
    if usable:
        usable.sort(key=lambda r: (-safe_float(r["actor_game_score_AUC"]), -safe_float(r["beta_active_frac"])))
        best = usable[0]
        if best["method"] == "proposed_nog_closed_current":
            selected_variant = "current"
        elif best["method"] == "proposed_nog_field_only":
            selected_variant = "field_only"
        elif best["method"] == "proposed_nog_actualV_linesearch":
            selected_variant = "actualV_linesearch"
        else:
            selected_variant = "score_oriented"
            selected_coeff = safe_float(best["score_field_coeff"])
        decision = "NOG_USABLE_BASELINE_FOUND"
    else:
        decision = "NOG_ZERO_BECAUSE_NOT_DESCENT_FOR_V"
    report_lines = ["# noG variant comparison report", ""]
    for row in sorted(summary_rows, key=lambda r: (-safe_float(r["actor_game_score_AUC"]), str(r["method"]))):
        report_lines.append(
            f"- `{row['method']}`: AUC=`{safe_float(row['actor_game_score_AUC']):.6e}`, beta_active=`{safe_float(row['beta_active_frac']):.3f}`, "
            f"nonzero_delta=`{safe_float(row['nonzero_parameter_delta_frac']):.3f}`, actor_delta=`{safe_float(row['actor_mean_delta_nonzero_frac']):.3f}`, "
            f"field_AUC=`{safe_float(row['field_norm_AUC']):.6e}`, decision_candidate=`{'usable' if row in usable else 'not_usable'}`"
        )
    if selected_variant:
        report_lines.extend(["", f"- selected_variant: `{selected_variant}`", f"- selected_score_field_coeff: `{selected_coeff}`"])
    report_lines.extend(["", f"- final decision: `{decision}`", ""])
    return decision, curves_rows, summary_rows, selected_variant, selected_coeff


def run_qp_repair(spec: Any, train_states: Any, eval_states: Any, selected_variant: str, selected_coeff: float | None) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], str]:
    methods = [
        ("sgd_gda", None),
        ("egm", None),
        ("selected_noG", None),
        ("proposed_qp_dampedG_nog_safe", None),
    ]
    curves_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for method, _ in methods:
        if method == "selected_noG":
            base_method = (
                "proposed_nog_closed_current" if selected_variant == "current"
                else "proposed_nog_field_only" if selected_variant == "field_only"
                else "proposed_nog_actualV_linesearch" if selected_variant == "actualV_linesearch"
                else "proposed_nog_score_oriented"
            )
            curves, summary = run_method_variant(spec, train_states, eval_states, base_method, QP_ITERATIONS, score_field_coeff=selected_coeff)
            summary["method"] = "selected_noG"
            for row in curves:
                row["method"] = "selected_noG"
            curves_rows.extend(curves)
            summary_rows.append(summary)
        else:
            variant_name = selected_variant if method == "proposed_qp_dampedG_nog_safe" else None
            curves, summary = run_method_variant(spec, train_states, eval_states, method, QP_ITERATIONS, variant_name=variant_name, score_field_coeff=selected_coeff)
            curves_rows.extend(curves)
            summary_rows.append(summary)
    lookup = {row["method"]: row for row in summary_rows}
    qp = lookup["proposed_qp_dampedG_nog_safe"]
    nog = lookup["selected_noG"]
    egm = lookup["egm"]
    sgd = lookup["sgd_gda"]
    qp_vs_nog = safe_float(qp["actor_game_score_AUC"]) / max(abs(safe_float(nog["actor_game_score_AUC"])), base.EPS)
    qp_vs_egm = safe_float(qp["actor_game_score_AUC"]) / max(abs(safe_float(egm["actor_game_score_AUC"])), base.EPS)
    base_beats_sgd = max(safe_float(nog["actor_game_score_AUC"]), safe_float(egm["actor_game_score_AUC"])) >= 1.05 * safe_float(sgd["actor_game_score_AUC"])

    def dominance(method_a: str, method_b: str) -> float:
        rows_a = [r for r in curves_rows if r["method"] == method_a]
        rows_b = [r for r in curves_rows if r["method"] == method_b]
        paired = min(len(rows_a), len(rows_b))
        if paired == 0:
            return math.nan
        wins = 0
        for idx in range(paired):
            wins += int(safe_float(rows_a[idx]["actor_game_score"]) >= safe_float(rows_b[idx]["actor_game_score"]))
        return wins / paired

    dom_nog = dominance("proposed_qp_dampedG_nog_safe", "selected_noG")
    dom_egm = dominance("proposed_qp_dampedG_nog_safe", "egm")
    if (
        qp_vs_nog >= 1.05
        and qp_vs_egm >= 1.05
        and base_beats_sgd
        and dom_nog >= 0.60
        and dom_egm >= 0.60
        and safe_float(qp["fallback_to_noG_frac"]) <= 0.30
        and safe_float(qp["QP_accept_frac_given_gamma_active"]) >= 0.70
    ):
        decision = "QP_POSITIVE_VS_REPAIRED_NOG"
    else:
        decision = "QP_FAILS_VS_REPAIRED_NOG"
    report_lines = [
        "# qp vs repaired noG report",
        "",
        f"- selected_noG_variant: `{selected_variant}`",
        f"- selected_score_field_coeff: `{selected_coeff}`",
        f"- qp_vs_nog_auc_ratio: `{qp_vs_nog:.3f}`",
        f"- qp_vs_egm_auc_ratio: `{qp_vs_egm:.3f}`",
        f"- qp_dom_nog: `{dom_nog:.3f}`",
        f"- qp_dom_egm: `{dom_egm:.3f}`",
        f"- qp_fallback_to_noG_frac: `{safe_float(qp['fallback_to_noG_frac']):.3f}`",
        f"- qp_accept_given_gamma_active: `{safe_float(qp['QP_accept_frac_given_gamma_active']):.3f}`",
        f"- final decision: `{decision}`",
        "",
    ]
    return decision, curves_rows, summary_rows, "\n".join(report_lines)


def main() -> None:
    ensure_dir(RESULT_ROOT)
    ensure_dir(RESULT_ROOT / "plots")
    spec = base.check_env(ENV_ID)
    if spec is None:
        write_text(RESULT_ROOT / "nog_freeze_diagnosis_decision.md", "ENV_UNAVAILABLE\n")
        write_text(RESULT_ROOT / "nog_freeze_diagnosis_report.md", f"# env unavailable\n\n- env: `{ENV_ID}`\n")
        return

    train_states, eval_states = collect_datasets(spec)
    write_text(
        RESULT_ROOT / "run_spec.json",
        json.dumps(
            {
                "env": ENV_ID,
                "rho": RHO,
                "lambda_u": LAMBDA_U,
                "lambda_w": LAMBDA_W,
                "joint_lr": JOINT_LR,
                "lambda_F": LAMBDA_F,
                "lambda_J": LAMBDA_J,
                "batch_size": BATCH_SIZE,
                "seed": SEED,
                "train_dataset_size": TRAIN_DATASET_SIZE,
                "eval_dataset_size": EVAL_DATASET_SIZE,
            },
            indent=2,
        ),
    )

    diag_decision, diag_rows, diag_report = diagnose_nog(spec, train_states, eval_states)
    write_csv(RESULT_ROOT / "nog_freeze_steps.csv", diag_rows)
    write_text(RESULT_ROOT / "nog_freeze_diagnosis_report.md", diag_report + "\n")
    plot_nog_steps(RESULT_ROOT / "plots" / "nog_beta_over_steps.png", diag_rows)

    variant_decision, variant_curves, variant_summary, selected_variant, selected_coeff = choose_usable_nog(spec, train_states, eval_states)
    write_csv(RESULT_ROOT / "nog_variant_comparison.csv", variant_summary)
    write_text(RESULT_ROOT / "nog_variant_comparison_report.md", (["# noG variant comparison", ""] + [str(r) for r in variant_summary]) and "\n".join([]))
    # overwrite with readable report
    _, _, _, _, _ = (variant_decision, variant_curves, variant_summary, selected_variant, selected_coeff)
    report_lines = ["# noG variant comparison report", ""]
    for row in sorted(variant_summary, key=lambda r: (-safe_float(r["actor_game_score_AUC"]), str(r["method"]))):
        report_lines.append(
            f"- `{row['method']}`: AUC=`{safe_float(row['actor_game_score_AUC']):.6e}`, beta_active=`{safe_float(row['beta_active_frac']):.3f}`, "
            f"nonzero_delta=`{safe_float(row['nonzero_parameter_delta_frac']):.3f}`, actor_delta=`{safe_float(row['actor_mean_delta_nonzero_frac']):.3f}`, "
            f"curve_sanity=`{int(row['curve_sanity_flag'])}`"
        )
    report_lines.extend(["", f"- variant decision: `{variant_decision}`", f"- selected variant: `{selected_variant or 'none'}`", f"- selected coeff: `{selected_coeff}`", ""])
    write_text(RESULT_ROOT / "nog_variant_comparison_report.md", "\n".join(report_lines) + "\n")
    plot_simple_lines(
        RESULT_ROOT / "plots" / "nog_variant_actor_game_score.png",
        {k: [r for r in variant_curves if r["method"] == k] for k in sorted({r["method"] for r in variant_curves})},
        "actor_game_score",
        "noG variant actor game score",
    )

    final_decision = diag_decision if variant_decision != "NOG_USABLE_BASELINE_FOUND" else variant_decision
    final_report = diag_report + "\n" + "\n".join(report_lines)

    if variant_decision == "NOG_USABLE_BASELINE_FOUND":
        qp_decision, qp_curves, qp_summary, qp_report = run_qp_repair(spec, train_states, eval_states, selected_variant, selected_coeff)
        write_csv(RESULT_ROOT / "qp_vs_repaired_nog_summary.csv", qp_summary)
        write_text(RESULT_ROOT / "qp_vs_repaired_nog_report.md", qp_report + "\n")
        write_text(RESULT_ROOT / "qp_vs_repaired_nog_decision.md", qp_decision + "\n")
        plot_simple_lines(
            RESULT_ROOT / "plots" / "qp_vs_repaired_nog_actor_game_score.png",
            {k: [r for r in qp_curves if r["method"] == k] for k in sorted({r["method"] for r in qp_curves})},
            "actor_game_score",
            "QP vs repaired noG actor game score",
        )
        final_decision = qp_decision if qp_decision.startswith("QP_") else variant_decision
        final_report += "\n\n" + qp_report

    write_text(RESULT_ROOT / "nog_freeze_diagnosis_decision.md", final_decision + "\n")
    write_text(RESULT_ROOT / "q_contains_nog_report.md", final_report + "\n")
    write_text(RESULT_ROOT / "q_contains_nog_decision.md", final_decision + "\n")


if __name__ == "__main__":
    main()
