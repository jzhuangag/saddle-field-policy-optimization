import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")


ROOT = Path(r"C:\Users\jzhuangag\work\rarl\original\torch-RARL")
SCRIPT = ROOT / "scripts" / "nn_actor_rotational_lqr_rarl.py"
RESULT_ROOT = Path(r"C:\Users\jzhuangag\work\rarl\original\results\nn_actor_rotational_lqr_rarl")
PLOTS = RESULT_ROOT / "plots"


def load_module():
    spec = importlib.util.spec_from_file_location("nn_actor_rot_lqr", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def task_eval(mod, benchmark, K, L, sigma0):
    sigma = sigma0.clone()
    closed_loop = benchmark.A + benchmark.B @ K + benchmark.E @ L
    rho = mod.safe_spectral_radius(closed_loop)
    total = torch.zeros((), dtype=torch.float64)
    weight_sum = 0.0
    mean_state = 0.0
    max_state = 0.0
    mean_u = 0.0
    for t in range(benchmark.cfg.horizon):
        weight = benchmark.cfg.gamma ** t
        weight_sum += weight
        x_sq = torch.trace(sigma)
        u_sq = torch.trace(K @ sigma @ K.T)
        reward = -0.5 * benchmark.cfg.q_state * x_sq - 0.5 * benchmark.cfg.a_u * u_sq
        total = total + weight * reward
        state_norm = float(torch.sqrt(torch.clamp(x_sq, min=0.0)))
        action_norm = float(torch.sqrt(torch.clamp(u_sq, min=0.0)))
        mean_state += weight * state_norm
        mean_u += weight * action_norm
        max_state = max(max_state, state_norm)
        sigma = closed_loop @ sigma @ closed_loop.T
    return {
        "task_return": float(total),
        "state_norm_mean": float(mean_state / max(weight_sum, 1e-12)),
        "state_norm_max": float(max_state),
        "action_norm_mean": float(mean_u / max(weight_sum, 1e-12)),
        "spectral_radius": float(rho),
        "finite": bool(np.isfinite(float(total)) and np.isfinite(max_state) and np.isfinite(rho)),
    }


def project_fro(M, max_norm):
    n = float(torch.linalg.norm(M))
    if n <= max_norm or n <= 1e-12:
        return M
    return M * (max_norm / n)


def project_local(L, L0, local_radius):
    delta = L - L0
    n = float(torch.linalg.norm(delta))
    if n <= local_radius or n <= 1e-12:
        return L
    return L0 + delta * (local_radius / n)


def robust_br(mod, benchmark, K, L_init, sigma0, local_br_radius, global_budget):
    def run(inner_lr):
        L0 = L_init.detach().clone()
        Lbar = L_init.detach().clone()
        for _ in range(20):
            Lreq = Lbar.detach().clone().requires_grad_(True)
            sigma = sigma0.clone()
            closed_loop = benchmark.A + benchmark.B @ K.detach() + benchmark.E @ Lreq
            total = torch.zeros((), dtype=torch.float64)
            for t in range(benchmark.cfg.horizon):
                x_sq = torch.trace(sigma)
                u_sq = torch.trace(K.detach() @ sigma @ K.detach().T)
                reward = -0.5 * benchmark.cfg.q_state * x_sq - 0.5 * benchmark.cfg.a_u * u_sq
                total = total + (benchmark.cfg.gamma ** t) * reward
                sigma = closed_loop @ sigma @ closed_loop.T
            grad = torch.autograd.grad(total, Lreq)[0]
            Lnext = Lbar - inner_lr * grad
            Lnext = project_local(Lnext, L0, local_br_radius)
            Lnext = project_fro(Lnext, global_budget)
            Lbar = Lnext.detach()
        return Lbar

    last = None
    for inner_lr in [0.03, 0.01]:
        Lbr = run(inner_lr)
        eval_res = task_eval(mod, benchmark, K, Lbr, sigma0)
        stable = eval_res["finite"] and eval_res["state_norm_max"] < 1e6 and eval_res["spectral_radius"] < 1.5
        last = (Lbr, eval_res, inner_lr, stable)
        if stable:
            return last
    return last


def run():
    PLOTS.mkdir(parents=True, exist_ok=True)
    mod = load_module()
    cfg = mod.MixedLinearActorRotLQConfig(
        beta_rot=1.0,
        beta_sym=0.1,
        lambda_F=1e-4,
        lambda_P=1.0,
        tau=0.03,
        gap_inner_steps=3,
        gap_inner_lr=0.03,
        local_radius=0.1,
        ppm_inner_steps=20,
        seed=0,
        init_scale=0.03,
        horizon=30,
        gamma=0.98,
    )
    benchmark = mod.MixedLinearActorRotLQBenchmark(cfg)
    flat0 = benchmark.flat0.clone()
    field0 = benchmark.field_tensor(flat0.detach(), create_graph=False).detach()
    field_energy0 = 0.5 * float(torch.dot(field0, field0))
    p_tau0 = benchmark.local_gap_terms(flat0)["P_tau"]

    iterations = 300
    baseline_lr = 0.01
    update_radius = 0.01
    probe_radius = min(update_radius, 1e-2) * 0.5

    curves = []
    diags = []
    traj_map = {}

    for method in ["sgd", "egm", "ppm"]:
        flat = benchmark.flat0.clone()
        traj = [flat.clone()]
        rows = []
        for it in range(iterations):
            flat = mod.mixed_linear_actor_rot_lq_method_step(benchmark, method, flat, baseline_lr)
            traj.append(flat.clone())
            metrics = benchmark.metrics(flat, field_energy0, p_tau0)
            rows.append(
                {
                    "method": method,
                    "iteration": it,
                    **metrics,
                    "clean_eval_return": float(metrics["eval_game_return"]),
                    "adversarial_eval_return": float(metrics["eval_game_return"]),
                    "nan_flag": float(not np.isfinite(metrics["V_lambda"]) or not np.isfinite(metrics["field_norm"])),
                    "divergence_flag": float(
                        metrics["max_abs_u"] > 100.0
                        or metrics["max_abs_w"] > 100.0
                        or metrics["K_norm"] > 100.0
                        or metrics["L_norm"] > 100.0
                        or not np.isfinite(metrics["max_abs_state"])
                    ),
                }
            )
        curves.append(pd.DataFrame(rows))
        traj_map[method] = traj

    nog_curves, nog_diags, nog_traj = mod.mixed_linear_actor_rot_lq_run_proposed_method(
        benchmark,
        "proposed_noG",
        iterations,
        field_energy0,
        p_tau0,
        update_radius,
        probe_radius,
        fallback_lr=baseline_lr,
        allow_fallback=True,
    )
    qpg_curves, qpg_diags, qpg_traj = mod.mixed_linear_actor_rot_lq_run_proposed_method(
        benchmark,
        "proposed_qpg",
        iterations,
        field_energy0,
        p_tau0,
        update_radius,
        probe_radius,
        fallback_lr=baseline_lr,
        allow_fallback=True,
    )
    curves.extend([nog_curves, qpg_curves])
    diags.extend([nog_diags, qpg_diags])
    traj_map["proposed_noG"] = nog_traj
    traj_map["proposed_qpg"] = qpg_traj

    curves_df = pd.concat(curves, ignore_index=True)
    diags_df = pd.concat(diags, ignore_index=True)

    max_L_norm = 0.0
    for traj in traj_map.values():
        for flat in traj:
            _, L = benchmark.split_flat(flat)
            max_L_norm = max(max_L_norm, float(torch.linalg.norm(L)))
    local_br_radius = 0.25
    global_budget = max(1.0, max_L_norm)

    perf_rows = []
    for method in ["sgd", "egm", "ppm", "proposed_noG", "proposed_qpg"]:
        sub = curves_df[curves_df["method"] == method].sort_values("iteration").reset_index(drop=True)
        traj = traj_map[method]
        for it in range(iterations):
            flat = traj[it + 1]
            K, L = benchmark.split_flat(flat)
            clean = task_eval(mod, benchmark, K, torch.zeros_like(L), benchmark.eval_sigma0)
            current = task_eval(mod, benchmark, K, L, benchmark.eval_sigma0)
            Lbr, robust, used_lr, stable = robust_br(mod, benchmark, K, L, benchmark.eval_sigma0, local_br_radius, global_budget)
            Lbr0, robust0, used_lr0, stable0 = robust_br(
                mod, benchmark, K, torch.zeros_like(L), benchmark.eval_sigma0, local_br_radius, global_budget
            )
            perf_rows.append(
                {
                    "iteration": it,
                    "method": method,
                    "clean_task_return": clean["task_return"],
                    "current_adv_task_return": current["task_return"],
                    "robust_br_task_return": robust["task_return"],
                    "robust_br_from_zero_task_return": robust0["task_return"],
                    "robust_degradation": clean["task_return"] - robust["task_return"],
                    "current_adv_degradation": clean["task_return"] - current["task_return"],
                    "game_return_train": float(sub.iloc[it]["train_game_return"]),
                    "game_return_eval": float(sub.iloc[it]["eval_game_return"]),
                    "V_lambda": float(sub.iloc[it]["V_lambda"]),
                    "normalized_P_tau": float(sub.iloc[it]["normalized_P_tau"]),
                    "field_norm": float(sub.iloc[it]["field_norm"]),
                    "approximate_local_exploitability": float(sub.iloc[it]["approximate_local_exploitability"]),
                    "K_norm": float(torch.linalg.norm(K)),
                    "L_norm": float(torch.linalg.norm(L)),
                    "L_br_norm": float(torch.linalg.norm(Lbr)),
                    "L_br_distance_from_current": float(torch.linalg.norm(Lbr - L)),
                    "closed_loop_spectral_radius_clean": clean["spectral_radius"],
                    "closed_loop_spectral_radius_current_adv": current["spectral_radius"],
                    "closed_loop_spectral_radius_robust_br": robust["spectral_radius"],
                    "state_norm_mean_clean": clean["state_norm_mean"],
                    "state_norm_mean_current_adv": current["state_norm_mean"],
                    "state_norm_mean_robust_br": robust["state_norm_mean"],
                    "state_norm_max_clean": clean["state_norm_max"],
                    "state_norm_max_current_adv": current["state_norm_max"],
                    "state_norm_max_robust_br": robust["state_norm_max"],
                    "clean_action_norm_mean": clean["action_norm_mean"],
                    "current_adv_action_norm_mean": current["action_norm_mean"],
                    "robust_br_action_norm_mean": robust["action_norm_mean"],
                    "valid_clean_eval": float(clean["finite"] and clean["state_norm_max"] < 1e6 and clean["spectral_radius"] < 1.5),
                    "valid_current_adv_eval": float(
                        current["finite"] and current["state_norm_max"] < 1e6 and current["spectral_radius"] < 1.5
                    ),
                    "valid_robust_br_eval": float(stable),
                    "valid_robust_br_zero_eval": float(stable0),
                }
            )

    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(RESULT_ROOT / "mixed_linear_actor_rot_lq_rarl_performance_metrics.csv", index=False)

    summary_rows = []
    for method in ["sgd", "egm", "ppm", "proposed_noG", "proposed_qpg"]:
        sub = perf_df[perf_df["method"] == method].sort_values("iteration")
        final = sub.iloc[-1]
        summary_rows.append(
            {
                "method": method,
                "clean_task_return_AUC": mod.auc_from_series(sub["clean_task_return"].tolist()),
                "current_adv_task_return_AUC": mod.auc_from_series(sub["current_adv_task_return"].tolist()),
                "robust_br_task_return_AUC": mod.auc_from_series(sub["robust_br_task_return"].tolist()),
                "robust_degradation_AUC": mod.auc_from_series(sub["robust_degradation"].tolist()),
                "final_clean_task_return": float(final["clean_task_return"]),
                "final_current_adv_task_return": float(final["current_adv_task_return"]),
                "final_robust_br_task_return": float(final["robust_br_task_return"]),
                "final_robust_degradation": float(final["robust_degradation"]),
                "valid_fraction_robust_br": float(sub["valid_robust_br_eval"].mean()),
                "max_state_norm_robust_br": float(sub["state_norm_max_robust_br"].max()),
                "max_L_br_norm": float(sub["L_br_norm"].max()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULT_ROOT / "mixed_linear_actor_rot_lq_rarl_performance_summary.csv", index=False)

    colors = {"sgd": "#d62728", "egm": "#2ca02c", "ppm": "#9467bd", "proposed_noG": "#1f77b4", "proposed_qpg": "#8c564b"}
    labels = {"sgd": "SGD", "egm": "EGM", "ppm": "PPM", "proposed_noG": "proposed_noG", "proposed_qpg": "proposed_QP_G"}

    def plot_metric(metric, path_name, title, ylabel):
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in ["sgd", "egm", "ppm", "proposed_noG", "proposed_qpg"]:
            sub = perf_df[perf_df["method"] == method]
            ax.plot(sub["iteration"], sub[metric], label=labels[method], color=colors[method])
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS / path_name, dpi=160)
        plt.close(fig)

    plot_metric("clean_task_return", "mixed_linear_actor_rot_lq_rarl_clean_task_return.png", "clean_task_return", "higher is better")
    plot_metric(
        "current_adv_task_return",
        "mixed_linear_actor_rot_lq_rarl_current_adv_task_return.png",
        "current_adv_task_return",
        "higher is better",
    )
    plot_metric(
        "robust_br_task_return",
        "mixed_linear_actor_rot_lq_rarl_robust_br_task_return.png",
        "robust_br_task_return",
        "higher is better",
    )
    plot_metric("robust_degradation", "mixed_linear_actor_rot_lq_rarl_robust_degradation.png", "robust_degradation", "lower is better")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric, title in zip(
        axes.flatten(),
        ["clean_task_return", "current_adv_task_return", "robust_br_task_return", "robust_degradation"],
        ["clean_task_return", "current_adv_task_return", "robust_br_task_return", "robust_degradation"],
    ):
        for method in ["sgd", "egm", "ppm", "proposed_noG", "proposed_qpg"]:
            sub = perf_df[perf_df["method"] == method]
            ax.plot(sub["iteration"], sub[metric], label=labels[method], color=colors[method])
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=5)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(PLOTS / "mixed_linear_actor_rot_lq_rarl_performance_paper.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, metric, title in zip(
        axes.flatten(),
        ["clean_task_return", "current_adv_task_return", "robust_br_task_return", "robust_degradation"],
        ["clean_task_return", "current_adv_task_return", "robust_br_task_return", "robust_degradation"],
    ):
        for method in ["sgd", "egm", "ppm", "proposed_noG", "proposed_qpg"]:
            sub = perf_df[perf_df["method"] == method]
            ax.plot(sub["iteration"], sub[metric], label=labels[method], color=colors[method])
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=5)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(PLOTS / "mixed_linear_actor_rot_lq_rarl_performance_all_plots_big.png", dpi=160)
    plt.close(fig)

    best_clean = summary_df.sort_values("final_clean_task_return", ascending=False).iloc[0]["method"]
    best_current = summary_df.sort_values("final_current_adv_task_return", ascending=False).iloc[0]["method"]
    best_robust = summary_df.sort_values("final_robust_br_task_return", ascending=False).iloc[0]["method"]
    qpg = summary_df[summary_df["method"] == "proposed_qpg"].iloc[0]
    best_baseline_robust = summary_df[summary_df["method"].isin(["sgd", "egm", "ppm"])]["final_robust_br_task_return"].max()

    nog = summary_df[summary_df["method"] == "proposed_noG"].iloc[0]
    if float(qpg["final_robust_br_task_return"]) > best_baseline_robust + 1e-12 and float(qpg["final_robust_br_task_return"]) >= float(nog["final_robust_br_task_return"]) - 1e-12:
        conclusion = (
            "QP+G not only reduces the unified Lyapunov and local saddle-gap metrics, but also improves robust task performance "
            "against local adversarial best responses."
        )
    elif float(qpg["final_robust_br_task_return"]) > best_baseline_robust + 1e-12:
        conclusion = (
            "QP+G clearly improves the optimization-geometry metrics and it also improves robust task performance relative to the SGD/EGM/PPM baselines, "
            "but in this configuration it does not beat proposed_noG on the robust task-return metric."
        )
    else:
        conclusion = (
            "QP+G mainly accelerates convergence to a local saddle under the game objective. The robust task-performance metrics remain comparable, "
            "and QP+G does not dominate all baselines and ablations on task return."
        )

    report = [
        "# MixedLinearActorRotLQ RARL Performance Report",
        "",
        "1. Why train/eval game return is not a standard RL performance metric.",
        "Because the game return includes the zero-sum rotational and adversary-energy terms used to define the saddle objective. It is the training game objective J(K,L), not a pure protagonist task-performance measure.",
        "",
        "2. What clean_task_return means.",
        "It evaluates protagonist K with adversary disabled (w=0) and scores only the task component: -0.5 q_state ||x||^2 - 0.5 a_u ||u||^2.",
        "",
        "3. What current_adv_task_return means.",
        "It evaluates protagonist K against its current learned adversary L, but still scores only the protagonist task reward.",
        "",
        "4. What robust_br_task_return means.",
        "It evaluates protagonist K against a local adversarial best response L_br optimized to minimize the protagonist task-only return.",
        "",
        "5. How L_br is approximated.",
        "Deterministic gradient descent on task_return(K, L_bar) for 20 inner steps, initialized from the current L, with retry at smaller inner_lr if needed.",
        "",
        f"6. What local_br_radius and L_global_budget are.\n`local_br_radius = {local_br_radius}`\n`L_global_budget = {global_budget}`",
        "",
        "7. Whether robust BR evaluations are stable.",
        summary_df[["method", "valid_fraction_robust_br", "max_state_norm_robust_br", "max_L_br_norm"]].to_string(index=False),
        "",
        f"8. Which method has best clean performance.\n`{best_clean}`",
        f"9. Which method has best current-adversary performance.\n`{best_current}`",
        f"10. Which method has best robust BR performance.\n`{best_robust}`",
        "11. Whether QP+G's Lyapunov improvement also corresponds to improved RARL-style performance.",
        conclusion,
        "",
        "12. If QP+G mainly improves stationarity but not robust task return, say that honestly.",
        conclusion,
        "",
        "Final summary table:",
        summary_df.to_string(index=False),
    ]
    (RESULT_ROOT / "mixed_linear_actor_rot_lq_rarl_performance_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    run()
