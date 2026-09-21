from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


SCRIPT_DIR = Path(__file__).resolve().parent
SMOOTH_PATH = SCRIPT_DIR / "survey_standard_rarl_smooth_skew_reaudit.py"
BASE_PATH = SCRIPT_DIR / "actor_only_critic_rarl_subsection3.py"

smooth_spec = importlib.util.spec_from_file_location("survey_standard_rarl_smooth_skew_reaudit", SMOOTH_PATH)
smooth = importlib.util.module_from_spec(smooth_spec)
sys.modules[smooth_spec.name] = smooth
smooth_spec.loader.exec_module(smooth)

base_spec = importlib.util.spec_from_file_location("actor_only_critic_rarl_subsection3", BASE_PATH)
base = importlib.util.module_from_spec(base_spec)
sys.modules[base_spec.name] = base
base_spec.loader.exec_module(base)


RESULT_ROOT = BASE_PATH.resolve().parents[2] / "results" / "halfc_weak_skew_final_baseline_audit"
PLOT_ROOT = RESULT_ROOT / "plots"

DEVICE = base.DEVICE
DTYPE = base.DTYPE
EPS = base.EPS
SEED = 0

ENV_NAME = "HalfCheetah-v5"
WRAPPER_TYPE = "action_disturbance_direct"
ALPHA = 0.3
ADV_DIM = 6
WARMUP_STEPS = 50000
CRITIC_ACTIVATIONS = ["softplus", "tanh"]
CRITIC_TRAIN_STEPS = 10000
CRITIC_AUDIT_INTERVAL = 1000
ACTOR_LR_GRID = [3e-6, 1e-5, 3e-5, 1e-4]
SGD_SELECT_ITERS = 100
FINAL_ITERS = 300
PPM_INNER_STEPS = 5
EVAL_INTERVAL = 10
FD_PROBES = 32
LAMBDA_F = 0.01
LAMBDA_P = 1.0
GAP_INNER_STEPS = 2
TAU = 0.03
LOCAL_GAP_RADIUS = 0.1
G_UTILITY_CHECKPOINTS = [0, 10, 25, 50, 100, 200, 300]


def ensure_dirs() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    base.write_csv(path, rows)


def write_text(path: Path, text: str) -> None:
    base.write_text(path, text)


def finite(x: float) -> bool:
    return math.isfinite(float(x))


def clone_run(src: smooth.SmoothAuditRun) -> smooth.SmoothAuditRun:
    run = smooth.SmoothAuditRun(src.ctx, src.activation)
    run.theta = src.theta.detach().clone()
    run.phi = src.phi.detach().clone()
    run.theta_target = src.theta_target.detach().clone()
    run.phi_target = src.phi_target.detach().clone()
    run.q.load_state_dict(src.q.state_dict())
    run.q_target.load_state_dict(src.q_target.state_dict())
    run.metric_refs = {}
    return run


def diagnostic_with_std(run: smooth.SmoothAuditRun, z: torch.Tensor, states: torch.Tensor, actor_lr: float) -> dict[str, float]:
    return run.diagnostic_metrics(z, states, actor_lr)


def robust_br_best(run: smooth.SmoothAuditRun, theta: torch.Tensor, states: torch.Tensor) -> tuple[torch.Tensor, bool]:
    base_phi = run.phi.detach().clone()
    best_phi = base_phi.clone()
    best_obj = None
    valid = False
    for lr in [1e-4, 3e-4]:
        current = base_phi.clone()
        for _ in range(20):
            cur = current.detach().clone().requires_grad_(True)
            z = torch.cat([theta.detach(), cur])
            objective = run.actor_objective(z, states)
            grad = torch.autograd.grad(objective, cur)[0]
            next_phi = cur - (lr * grad)
            delta = next_phi - base_phi
            norm = torch.linalg.norm(delta)
            if norm > smooth.LOCAL_BR_RADIUS:
                next_phi = base_phi + (delta * (smooth.LOCAL_BR_RADIUS / (norm + EPS)))
            current = next_phi.detach()
        obj = float(run.actor_objective(torch.cat([theta.detach(), current]), states).detach().item())
        if best_obj is None or obj < best_obj:
            best_obj = obj
            best_phi = current.detach().clone()
            valid = True
    return best_phi, valid


def run_method(run_src: smooth.SmoothAuditRun, method: str, actor_lr: float, iterations: int, checkpoint_iters: set[int] | None = None) -> tuple[list[dict[str, Any]], dict[int, torch.Tensor]]:
    run = clone_run(run_src)
    states = run.ctx.diag_batch["obs"]
    curves: list[dict[str, Any]] = []
    saved: dict[int, torch.Tensor] = {}
    checkpoint_iters = checkpoint_iters or set()
    for iteration in range(iterations + 1):
        z = torch.cat([run.theta, run.phi]).detach().clone()
        if iteration in checkpoint_iters:
            saved[iteration] = z.detach().clone()
        if iteration % EVAL_INTERVAL == 0 or iteration == iterations:
            diag = diagnostic_with_std(run, z, states, actor_lr)
            clean_eval = run.evaluate_actor(run.theta, None, 3)
            adv_eval = run.evaluate_actor(run.theta, run.phi, 3)
            br_phi, br_valid = robust_br_best(run, run.theta, states[:128])
            br_eval = run.evaluate_actor(run.theta, br_phi, 3)
            curves.append(
                {
                    "iteration": iteration,
                    "method": method,
                    "actor_lr": actor_lr,
                    **diag,
                    "clean_task_return": clean_eval["return"],
                    "current_adv_task_return": adv_eval["return"],
                    "robust_br_task_return": br_eval["return"],
                    "robust_degradation": clean_eval["return"] - br_eval["return"],
                    "action_clip_fraction": max(clean_eval["clip"], adv_eval["clip"], br_eval["clip"]),
                    "actor_param_norm": float(torch.linalg.norm(run.theta).item()),
                    "adversary_param_norm": float(torch.linalg.norm(run.phi).item()),
                    "robust_br_valid": int(br_valid),
                }
            )
        if iteration < iterations:
            next_z, meta = smooth.short_update(run, z, states, method, actor_lr)
            run.theta = next_z[: run.ctx.theta_layout.num_params].detach().clone()
            run.phi = next_z[run.ctx.theta_layout.num_params :].detach().clone()
            if curves:
                curves[-1]["actor_update_norm"] = float(meta["update_norm"])
    return curves, saved


def select_best_lr(run_src: smooth.SmoothAuditRun) -> tuple[float | None, list[dict[str, Any]], dict[str, float] | None]:
    best_lr = None
    best_auc = None
    best_curves: list[dict[str, Any]] = []
    best_summary = None
    for lr in ACTOR_LR_GRID:
        curves, _ = run_method(run_src, "sgd", lr, SGD_SELECT_ITERS)
        vals_v = [r["V_std"] for r in curves]
        vals_p = [r["P_tau_std"] for r in curves]
        vals_f = [r["field_norm"] for r in curves]
        valid = (
            all(finite(v) for v in vals_v + vals_p + vals_f)
            and curves[-1]["robust_br_valid"] == 1
            and max(r["action_clip_fraction"] for r in curves) <= 0.2
        )
        summary = {
            "actor_lr": lr,
            "valid_flag": int(valid),
            "V_std_AUC": float(sum(vals_v)),
            "P_tau_std_AUC": float(sum(vals_p)),
            "field_norm_AUC": float(sum(vals_f)),
            "curve_normal_flag": int(curves[-1]["V_std"] <= curves[0]["V_std"] + 1e-8 and curves[-1]["P_tau_std"] <= curves[0]["P_tau_std"] + 1e-8),
        }
        if valid and summary["curve_normal_flag"] == 1:
            if best_auc is None or summary["V_std_AUC"] < best_auc:
                best_auc = summary["V_std_AUC"]
                best_lr = lr
                best_curves = curves
                best_summary = summary
    return best_lr, best_curves, best_summary


def fd_geometry(run: smooth.SmoothAuditRun) -> dict[str, float]:
    z = torch.cat([run.theta, run.phi]).detach().clone()
    states = run.ctx.diag_batch["obs"]
    field = run.actor_field(z, states).detach()
    field_norm = float(torch.linalg.norm(field).item())
    eps_f = 1e-4 / (field_norm + 1e-12)
    g = (run.actor_field(z + (eps_f * field), states).detach() - field) / eps_f
    g_norm = float(torch.linalg.norm(g).item())
    cos_fg = float(torch.dot(field, g).item() / ((field_norm * g_norm) + EPS))
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(20260620)
    d = z.numel()
    skew_vals, sym_vals = [], []
    for _ in range(FD_PROBES):
        u = torch.randn(d, generator=gen, dtype=DTYPE, device=DEVICE)
        v = torch.randn(d, generator=gen, dtype=DTYPE, device=DEVICE)
        u = u / (torch.linalg.norm(u) + EPS)
        v = v / (torch.linalg.norm(v) + EPS)
        eps_u = 1e-4 / (float(torch.linalg.norm(u).item()) + 1e-12)
        eps_v = 1e-4 / (float(torch.linalg.norm(v).item()) + 1e-12)
        Ju = (run.actor_field(z + (eps_u * u), states).detach() - field) / eps_u
        Jv = (run.actor_field(z + (eps_v * v), states).detach() - field) / eps_v
        skew_vals.append(abs(float(torch.dot(u, Jv).item() - torch.dot(v, Ju).item())))
        sym_vals.append(abs(float(torch.dot(u, Jv).item() + torch.dot(v, Ju).item())))
    mean_skew = float(np.mean(skew_vals))
    mean_sym = float(np.mean(sym_vals))
    tdim = run.ctx.theta_layout.num_params
    v_theta = torch.randn(tdim, generator=gen, dtype=DTYPE, device=DEVICE)
    v_phi = torch.randn(z.numel() - tdim, generator=gen, dtype=DTYPE, device=DEVICE)
    v_theta = v_theta / (torch.linalg.norm(v_theta) + EPS)
    v_phi = v_phi / (torch.linalg.norm(v_phi) + EPS)
    eps = 1e-4
    z_theta = z.detach().clone()
    z_phi = z.detach().clone()
    z_theta[:tdim] = z_theta[:tdim] + (eps * v_theta)
    z_phi[tdim:] = z_phi[tdim:] + (eps * v_phi)
    f_theta = run.actor_field(z_theta, states).detach()
    f_phi = run.actor_field(z_phi, states).detach()
    cross_phi_to_theta = float(torch.linalg.norm(f_phi[:tdim] - field[:tdim]).item()) / eps
    cross_theta_to_phi = float(torch.linalg.norm(f_theta[tdim:] - field[tdim:]).item()) / eps
    same_theta = float(torch.linalg.norm(f_theta[:tdim] - field[:tdim]).item()) / eps
    same_phi = float(torch.linalg.norm(f_phi[tdim:] - field[tdim:]).item()) / eps
    cross = 0.5 * (cross_phi_to_theta + cross_theta_to_phi)
    same = 0.5 * (same_theta + same_phi)
    out = run.output_geometry(states)
    return {
        "field_norm": field_norm,
        "G_norm": g_norm,
        "G_over_F": g_norm / (field_norm + EPS),
        "cos_F_G": cos_fg,
        "non_collinearity": math.sqrt(max(0.0, 1.0 - min(1.0, cos_fg * cos_fg))),
        "fd_skew_ratio": mean_skew / (mean_sym + 1e-12),
        "mean_skew_bilinear": mean_skew,
        "mean_sym_bilinear": mean_sym,
        "cross_phi_to_theta": cross_phi_to_theta,
        "cross_theta_to_phi": cross_theta_to_phi,
        "same_theta": same_theta,
        "same_phi": same_phi,
        "cross_player_coupling_proxy": cross,
        "same_player_proxy": same,
        "cross_to_same_ratio": cross / (same + 1e-12),
        "output_fd_skew_ratio": out["output_fd_skew_ratio"],
        "output_cross_to_diag_ratio": out["output_cross_to_diag_ratio"],
        "output_rotation_ratio": out["output_rotation_ratio"],
        "output_num_complex_eigs": out["output_num_complex_eigs"],
        "output_max_imag_eig": out["output_max_imag_eig"],
    }


def compute_v(run: smooth.SmoothAuditRun, z: torch.Tensor, actor_lr: float) -> float:
    return float(diagnostic_with_std(run, z, run.ctx.diag_batch["obs"], actor_lr)["V_std"])


def local_g_utility(run_src: smooth.SmoothAuditRun, z: torch.Tensor, actor_lr: float) -> dict[str, float]:
    run = clone_run(run_src)
    states = run.ctx.diag_batch["obs"]
    field = run.actor_field(z, states).detach()
    f_norm = float(torch.linalg.norm(field).item())
    eps_f = 1e-4 / (f_norm + 1e-12)
    g = (run.actor_field(z + (eps_f * field), states).detach() - field) / eps_f
    g_norm = float(torch.linalg.norm(g).item())
    v0 = compute_v(run, z, actor_lr)

    beta_grid = [0.0, 0.25 * actor_lr, 0.5 * actor_lr, 1.0 * actor_lr, 2.0 * actor_lr, 4.0 * actor_lr]
    noG_samples = []
    for beta in beta_grid:
        z_new = z - (beta * field)
        noG_samples.append((beta, compute_v(run, z_new, actor_lr)))
    x = np.asarray([b for b, _ in noG_samples], dtype=np.float64)
    y = np.asarray([v for _, v in noG_samples], dtype=np.float64)
    coef = np.polyfit(x, y, deg=2)
    dense = np.linspace(float(x.min()), float(x.max()), 101)
    pred = coef[0] * dense * dense + coef[1] * dense + coef[2]
    best_pred_idx = int(np.argmin(pred))
    beta_star = float(dense[best_pred_idx])
    best_pred_noG = float(v0 - pred[best_pred_idx])
    actual_best_noG = float(v0 - np.min(y))

    gamma_base = actor_lr * (f_norm / (g_norm + EPS))
    beta_vals = [0.0, 0.5 * actor_lr, 1.0 * actor_lr, 2.0 * actor_lr]
    gamma_vals = [-2.0 * gamma_base, -1.0 * gamma_base, -0.5 * gamma_base, 0.0, 0.5 * gamma_base, 1.0 * gamma_base, 2.0 * gamma_base]
    samples = []
    feats = []
    targets = []
    for beta in beta_vals:
        for gamma in gamma_vals:
            z_new = z - (beta * field) + (gamma * g)
            val = compute_v(run, z_new, actor_lr)
            samples.append((beta, gamma, val))
            feats.append([1.0, beta, gamma, beta * beta, beta * gamma, gamma * gamma])
            targets.append(val)
    X = np.asarray(feats, dtype=np.float64)
    Y = np.asarray(targets, dtype=np.float64)
    coef2, *_ = np.linalg.lstsq(X, Y, rcond=None)
    dense_b = np.linspace(min(beta_vals), max(beta_vals), 41)
    dense_g = np.linspace(min(gamma_vals), max(gamma_vals), 61)
    best_pred_val = None
    best_bg = (0.0, 0.0)
    for beta in dense_b:
        for gamma in dense_g:
            pred_v = float(coef2 @ np.asarray([1.0, beta, gamma, beta * beta, beta * gamma, gamma * gamma], dtype=np.float64))
            if best_pred_val is None or pred_v < best_pred_val:
                best_pred_val = pred_v
                best_bg = (float(beta), float(gamma))
    actual_best_2d = float(v0 - np.min(Y))
    best_pred_2d = float(v0 - best_pred_val)
    beta2, gamma2 = best_bg
    return {
        "V_before": v0,
        "best_predicted_decrease_noG": best_pred_noG,
        "best_actual_decrease_noG": actual_best_noG,
        "best_predicted_decrease_2D": best_pred_2d,
        "best_actual_decrease_2D": actual_best_2d,
        "G_utility_pred_ratio": best_pred_2d / (best_pred_noG + EPS),
        "G_utility_actual_ratio": actual_best_2d / (actual_best_noG + EPS),
        "gamma_star": gamma2,
        "beta_star_noG": beta_star,
        "beta_star_2D": beta2,
        "G_contribution_ratio_local": abs(gamma2) * g_norm / (abs(beta2) * f_norm + EPS),
        "twoD_step_valid": int(best_pred_2d >= 0.0 and actual_best_2d >= 0.0),
        "noG_step_valid": int(best_pred_noG >= 0.0 and actual_best_noG >= 0.0),
    }


def auc_from_curves(curves: list[dict[str, Any]], key: str) -> float:
    return float(sum(float(r[key]) for r in curves))


def plot_lines(rows: list[dict[str, Any]], ykey: str, title: str, filename: str) -> None:
    if plt is None or not rows:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = f"{row['critic_activation']}:{row['method']}"
        groups.setdefault(label, []).append(row)
    for label, sub in groups.items():
        sub = sorted(sub, key=lambda r: int(r["iteration"]))
        ax.plot([r["iteration"] for r in sub], [r[ykey] for r in sub], label=label)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / filename, dpi=180)
    plt.close(fig)


def plot_g_utility(rows: list[dict[str, Any]]) -> None:
    if plt is None or not rows:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["critic_activation"]), []).append(row)
    for label, sub in groups.items():
        sub = sorted(sub, key=lambda r: int(r["iteration"]))
        ax.plot([r["iteration"] for r in sub], [r["G_utility_actual_ratio"] for r in sub], marker="o", label=f"{label}:actual")
        ax.plot([r["iteration"] for r in sub], [r["G_utility_pred_ratio"] for r in sub], linestyle="--", label=f"{label}:pred")
    ax.axhline(1.05, color="red", linestyle=":", linewidth=1)
    ax.set_title("Local G Utility Ratios")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "halfc_local_G_utility.png", dpi=180)
    plt.close(fig)


def plot_big() -> None:
    if plt is None:
        return
    paths = [
        ("halfc_baseline_V_std.png", "V_std"),
        ("halfc_baseline_P_tau_std.png", "P_tau_std"),
        ("halfc_baseline_field_norm.png", "field_norm"),
        ("halfc_baseline_robust_task_return.png", "robust_br_task_return"),
        ("halfc_baseline_robust_degradation.png", "robust_degradation"),
        ("halfc_local_G_utility.png", "G utility"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    for ax, (name, title) in zip(axes.flatten(), paths):
        img = plt.imread(PLOT_ROOT / name)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "halfc_all_plots_big.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    base.seed_everything(SEED)
    spec = smooth.CandidateSpec(ENV_NAME, WRAPPER_TYPE, ALPHA, ADV_DIM, WARMUP_STEPS)
    ctx = smooth.CandidateContext(spec)

    summary_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    g_rows: list[dict[str, Any]] = []
    critic_fail_count = 0

    for activation in CRITIC_ACTIVATIONS:
        run = smooth.SmoothAuditRun(ctx, activation)
        critic_hist = []
        for step in range(CRITIC_TRAIN_STEPS):
            stats = run.critic_update()
            if step % CRITIC_AUDIT_INTERVAL == 0 or step == CRITIC_TRAIN_STEPS - 1:
                quality = run.mc_quality()
                critic_hist.append({**stats, **quality, "critic_step": step})
        final_critic = critic_hist[-1]
        critic_loss_mean_last = float(np.mean([r["critic_loss"] for r in critic_hist[-min(len(critic_hist), 2) :]]))
        critic_ok = (
            finite(final_critic["critic_loss"])
            and finite(final_critic["Q_abs_mean"])
            and final_critic["Q_abs_mean"] < 1e6
            and (not finite(final_critic["corr_Q_MC"]) or final_critic["corr_Q_MC"] >= 0.0)
        )
        if not critic_ok:
            critic_fail_count += 1
            summary_rows.append(
                {
                    "critic_activation": activation,
                    "selected_actor_lr": math.nan,
                    "critic_loss_final": final_critic["critic_loss"],
                    "critic_loss_mean_last_1000": critic_loss_mean_last,
                    "Q_mean": final_critic["Q_mean"],
                    "Q_std": final_critic["Q_std"],
                    "Q_abs_mean": final_critic["Q_abs_mean"],
                    "target_mean": final_critic["target_mean"],
                    "target_std": final_critic["target_std"],
                    "corr_Q_MC": final_critic["corr_Q_MC"],
                    "mse_Q_MC": final_critic["mse_Q_MC"],
                    "rank_corr_Q_MC": final_critic["rank_corr_Q_MC"],
                    "activation_valid": 0,
                }
            )
            continue

        geom = fd_geometry(run)
        best_lr, sgd_select_curves, sgd_select_summary = select_best_lr(run)
        if best_lr is None or sgd_select_summary is None:
            summary_rows.append(
                {
                    "critic_activation": activation,
                    "selected_actor_lr": math.nan,
                    "critic_loss_final": final_critic["critic_loss"],
                    "critic_loss_mean_last_1000": critic_loss_mean_last,
                    "Q_mean": final_critic["Q_mean"],
                    "Q_std": final_critic["Q_std"],
                    "Q_abs_mean": final_critic["Q_abs_mean"],
                    "target_mean": final_critic["target_mean"],
                    "target_std": final_critic["target_std"],
                    "corr_Q_MC": final_critic["corr_Q_MC"],
                    "mse_Q_MC": final_critic["mse_Q_MC"],
                    "rank_corr_Q_MC": final_critic["rank_corr_Q_MC"],
                    **geom,
                    "activation_valid": 0,
                }
            )
            continue

        method_curves = {}
        method_states = {}
        for method in ["sgd", "egm", "ppm"]:
            curves, saved = run_method(run, method, best_lr, FINAL_ITERS, set(G_UTILITY_CHECKPOINTS))
            for row in curves:
                row["critic_activation"] = activation
                curve_rows.append(row)
            method_curves[method] = curves
            method_states[method] = saved

        for it in G_UTILITY_CHECKPOINTS:
            if it not in method_states["sgd"]:
                continue
            z = method_states["sgd"][it]
            g_diag = local_g_utility(run, z, best_lr)
            g_rows.append({"critic_activation": activation, "iteration": it, **g_diag})

        sgd_curves = method_curves["sgd"]
        egm_curves = method_curves["egm"]
        ppm_curves = method_curves["ppm"]
        summary_rows.append(
            {
                "critic_activation": activation,
                "selected_actor_lr": best_lr,
                "critic_loss_final": final_critic["critic_loss"],
                "critic_loss_mean_last_1000": critic_loss_mean_last,
                "Q_mean": final_critic["Q_mean"],
                "Q_std": final_critic["Q_std"],
                "Q_abs_mean": final_critic["Q_abs_mean"],
                "target_mean": final_critic["target_mean"],
                "target_std": final_critic["target_std"],
                "corr_Q_MC": final_critic["corr_Q_MC"],
                "mse_Q_MC": final_critic["mse_Q_MC"],
                "rank_corr_Q_MC": final_critic["rank_corr_Q_MC"],
                **geom,
                "SGD_V_AUC": auc_from_curves(sgd_curves, "V_std"),
                "EGM_V_AUC": auc_from_curves(egm_curves, "V_std"),
                "PPM_V_AUC": auc_from_curves(ppm_curves, "V_std"),
                "SGD_P_tau_AUC": auc_from_curves(sgd_curves, "P_tau_std"),
                "EGM_P_tau_AUC": auc_from_curves(egm_curves, "P_tau_std"),
                "PPM_P_tau_AUC": auc_from_curves(ppm_curves, "P_tau_std"),
                "SGD_field_norm_AUC": auc_from_curves(sgd_curves, "field_norm"),
                "EGM_field_norm_AUC": auc_from_curves(egm_curves, "field_norm"),
                "PPM_field_norm_AUC": auc_from_curves(ppm_curves, "field_norm"),
                "SGD_robust_br_task_return_AUC": auc_from_curves(sgd_curves, "robust_br_task_return"),
                "EGM_robust_br_task_return_AUC": auc_from_curves(egm_curves, "robust_br_task_return"),
                "PPM_robust_br_task_return_AUC": auc_from_curves(ppm_curves, "robust_br_task_return"),
                "SGD_robust_degradation_AUC": auc_from_curves(sgd_curves, "robust_degradation"),
                "EGM_robust_degradation_AUC": auc_from_curves(egm_curves, "robust_degradation"),
                "PPM_robust_degradation_AUC": auc_from_curves(ppm_curves, "robust_degradation"),
                "baseline_advantage": auc_from_curves(sgd_curves, "V_std") / (min(auc_from_curves(egm_curves, "V_std"), auc_from_curves(ppm_curves, "V_std")) + EPS),
                "robust_br_valid_fraction": float(np.mean([r["robust_br_valid"] for r in sgd_curves + egm_curves + ppm_curves])),
                "activation_valid": 1,
            }
        )

    write_csv(RESULT_ROOT / "halfc_final_baseline_audit_curves.csv", curve_rows)
    write_csv(RESULT_ROOT / "halfc_local_G_utility.csv", g_rows)
    write_csv(RESULT_ROOT / "halfc_final_baseline_audit_summary.csv", summary_rows)

    plot_lines(curve_rows, "V_std", "HalfCheetah Baseline V_std", "halfc_baseline_V_std.png")
    plot_lines(curve_rows, "P_tau_std", "HalfCheetah Baseline P_tau_std", "halfc_baseline_P_tau_std.png")
    plot_lines(curve_rows, "field_norm", "HalfCheetah Baseline Field Norm", "halfc_baseline_field_norm.png")
    plot_lines(curve_rows, "robust_br_task_return", "HalfCheetah Robust BR Task Return", "halfc_baseline_robust_task_return.png")
    plot_lines(curve_rows, "robust_degradation", "HalfCheetah Robust Degradation", "halfc_baseline_robust_degradation.png")
    plot_g_utility(g_rows)
    plot_big()

    g_report_lines = ["# halfc_local_G_utility_report", ""]
    for row in g_rows:
        g_report_lines.append(
            f"- {row['critic_activation']} / iter {row['iteration']}: G_utility_actual_ratio={float(row['G_utility_actual_ratio']):.3f}, G_utility_pred_ratio={float(row['G_utility_pred_ratio']):.3f}, gamma_star={float(row['gamma_star']):.6e}, G_contribution_ratio_local={float(row['G_contribution_ratio_local']):.3f}"
        )
    write_text(RESULT_ROOT / "halfc_local_G_utility_report.md", "\n".join(g_report_lines) + "\n")

    report_lines = [
        "# halfc_final_baseline_audit_report",
        "",
        "1. Setup",
        f"- env: `{ENV_NAME}`",
        f"- wrapper: `{WRAPPER_TYPE}`",
        f"- alpha: `{ALPHA}`",
        "- objective: original environment reward only",
        "",
        "2. Critic quality and geometry",
    ]
    for row in summary_rows:
        report_lines.append(
            f"- {row['critic_activation']}: valid={int(row.get('activation_valid', 0))}, critic_loss_final={float(row['critic_loss_final']) if finite(row['critic_loss_final']) else math.nan}, corr_Q_MC={float(row['corr_Q_MC']) if finite(row['corr_Q_MC']) else math.nan}, fd_skew_ratio={float(row.get('fd_skew_ratio', math.nan)) if finite(row.get('fd_skew_ratio', math.nan)) else math.nan}, output_fd_skew_ratio={float(row.get('output_fd_skew_ratio', math.nan)) if finite(row.get('output_fd_skew_ratio', math.nan)) else math.nan}, cross_to_same_ratio={float(row.get('cross_to_same_ratio', math.nan)) if finite(row.get('cross_to_same_ratio', math.nan)) else math.nan}"
        )
    report_lines.extend(["", "3. Baseline-only results"])
    for row in summary_rows:
        if int(row.get("activation_valid", 0)) != 1:
            continue
        report_lines.append(
            f"- {row['critic_activation']}: lr={row['selected_actor_lr']}, baseline_advantage={float(row['baseline_advantage']):.3f}, SGD_V_AUC={float(row['SGD_V_AUC']):.3f}, EGM_V_AUC={float(row['EGM_V_AUC']):.3f}, PPM_V_AUC={float(row['PPM_V_AUC']):.3f}, robust_br_valid_fraction={float(row['robust_br_valid_fraction']):.3f}"
        )
    report_lines.extend(["", "4. Local G utility"])
    for activation in CRITIC_ACTIVATIONS:
        sub = [r for r in g_rows if r["critic_activation"] == activation]
        if sub:
            actuals = [float(r["G_utility_actual_ratio"]) for r in sub if finite(r["G_utility_actual_ratio"])]
            preds = [float(r["G_utility_pred_ratio"]) for r in sub if finite(r["G_utility_pred_ratio"])]
            report_lines.append(f"- {activation}: mean_actual_ratio={float(np.mean(actuals)) if actuals else math.nan:.3f}, mean_pred_ratio={float(np.mean(preds)) if preds else math.nan:.3f}")
    report_lines.extend([
        "",
        "5. Interpretation",
        "- measurable skew does not imply usable skew;",
        "- usable skew means it improves Lyapunov descent or baseline optimizer behavior;",
        "- standard RARL performance here uses original environment reward only.",
    ])
    write_text(RESULT_ROOT / "halfc_final_baseline_audit_report.md", "\n".join(report_lines) + "\n")

    if critic_fail_count == len(CRITIC_ACTIVATIONS):
        decision = "CRITIC_FAIL"
    else:
        best_adv = max([float(r.get("baseline_advantage", math.nan)) for r in summary_rows if int(r.get("activation_valid", 0)) == 1] or [math.nan])
        g_useful_counts = 0
        g_total = 0
        for row in g_rows:
            if finite(row["G_utility_actual_ratio"]):
                g_total += 1
                if float(row["G_utility_actual_ratio"]) > 1.1:
                    g_useful_counts += 1
        g_mostly_weak = g_total > 0 and sum(1 for r in g_rows if finite(r["G_utility_actual_ratio"]) and float(r["G_utility_actual_ratio"]) <= 1.05) >= math.ceil(0.6 * g_total)
        best_row = None
        best_row_adv = -math.inf
        for row in summary_rows:
            if int(row.get("activation_valid", 0)) == 1 and finite(row.get("baseline_advantage", math.nan)):
                if float(row["baseline_advantage"]) > best_row_adv:
                    best_row_adv = float(row["baseline_advantage"])
                    best_row = row
        robust_ok = False
        if best_row is not None:
            robust_ok = (
                float(best_row["EGM_robust_br_task_return_AUC"]) >= float(best_row["SGD_robust_br_task_return_AUC"]) - 1e-8
                or float(best_row["PPM_robust_br_task_return_AUC"]) >= float(best_row["SGD_robust_br_task_return_AUC"]) - 1e-8
            )
        if finite(best_adv) and best_adv >= 1.3 and robust_ok and g_useful_counts >= max(1, math.ceil(0.4 * g_total)):
            decision = "BASELINE_READY_FOR_PROPOSED"
        elif finite(best_adv) and 1.1 <= best_adv < 1.3 and robust_ok:
            decision = "BASELINE_WEAK_POSITIVE_BUT_NOT_STRONG"
        else:
            decision = "GEOMETRY_WEAK_CLOSE_STANDARD_RARL"

    decision_lines = [
        "# halfc_final_baseline_audit_decision",
        "",
        f"Decision: `{decision}`",
        "",
        "Summary:",
    ]
    if decision == "CRITIC_FAIL":
        decision_lines.append("- Both smooth critics failed quality checks.")
    elif decision == "GEOMETRY_WEAK_CLOSE_STANDARD_RARL":
        decision_lines.extend(
            [
                "- Measurable skew exists, but it does not translate into reliable optimizer-level EGM/PPM advantage.",
                "- The local two-direction span{-F,G} does not provide meaningful additional actual Lyapunov descent beyond span{-F} at most checkpoints.",
                "- This candidate should not proceed to proposed.",
                "- Subsection 3 standard-RARL should be closed as geometry-boundary diagnostics for this candidate.",
            ]
        )
    elif decision == "BASELINE_WEAK_POSITIVE_BUT_NOT_STRONG":
        decision_lines.extend(
            [
                "- There is a weak baseline advantage, but it is below the stronger threshold.",
                "- Robust task performance remains comparable or better.",
                "- This is not yet strong enough to justify proposed.",
            ]
        )
    else:
        decision_lines.extend(
            [
                "- Measurable skew translates into optimizer-level baseline advantage.",
                "- G provides meaningful local Lyapunov utility beyond -F frequently enough.",
                "- This candidate is ready for proposed, but proposed was not run in this script.",
            ]
        )
    write_text(RESULT_ROOT / "halfc_final_baseline_audit_decision.md", "\n".join(decision_lines) + "\n")


if __name__ == "__main__":
    main()
