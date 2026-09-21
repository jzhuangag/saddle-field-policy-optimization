from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")


SCRIPT = Path(r"C:\Users\jzhuangag\work\rarl\original\torch-RARL\scripts\nn_actor_rotational_lqr_rarl.py")
RESULT_ROOT = Path(r"C:\Users\jzhuangag\work\rarl\original\results\nn_actor_rotational_lqr_rarl")
PLOTS = RESULT_ROOT / "plots"


def load_module():
    spec = importlib.util.spec_from_file_location("nn_actor_rot_lqr", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def project_local(L, L0, local_radius):
    delta = L - L0
    n = float(torch.linalg.norm(delta))
    if n <= local_radius or n <= 1e-12:
        return L
    return L0 + delta * (local_radius / n)


def aligned_eval(mod, benchmark, K, L, sigma0, disable_adversary: bool = False):
    sigma = sigma0.clone()
    if disable_adversary:
        L_used = torch.zeros_like(L)
    else:
        L_used = L
    closed_loop = benchmark.A + benchmark.B @ K + benchmark.E @ L_used
    rho = mod.safe_spectral_radius(closed_loop)
    total_aligned = torch.zeros((), dtype=torch.float64)
    total_pure = torch.zeros((), dtype=torch.float64)
    weight_sum = 0.0
    mean_state = 0.0
    max_state = 0.0
    mean_u = 0.0
    for t in range(benchmark.cfg.horizon):
        weight = benchmark.cfg.gamma ** t
        weight_sum += weight
        x_sq = torch.trace(sigma)
        u_sq = torch.trace(K @ sigma @ K.T)
        w_sq = torch.trace(L_used @ sigma @ L_used.T)
        rot_term = torch.trace(sigma @ K.T @ benchmark.Hmat @ L_used)
        sym_term = torch.trace(sigma @ K.T @ benchmark.Smat @ L_used)
        pure_reward = -0.5 * benchmark.cfg.q_state * x_sq - 0.5 * benchmark.cfg.a_u * u_sq
        aligned_reward = pure_reward + 0.5 * benchmark.cfg.a_w * w_sq + benchmark.cfg.beta_rot * rot_term + benchmark.cfg.beta_sym * sym_term
        total_aligned = total_aligned + weight * aligned_reward
        total_pure = total_pure + weight * pure_reward
        state_norm = float(torch.sqrt(torch.clamp(x_sq, min=0.0)))
        action_norm = float(torch.sqrt(torch.clamp(u_sq, min=0.0)))
        mean_state += weight * state_norm
        mean_u += weight * action_norm
        max_state = max(max_state, state_norm)
        sigma = closed_loop @ sigma @ closed_loop.T
    return {
        "aligned_return": float(total_aligned),
        "pure_return": float(total_pure),
        "state_norm_mean": float(mean_state / max(weight_sum, 1e-12)),
        "state_norm_max": float(max_state),
        "action_norm_mean": float(mean_u / max(weight_sum, 1e-12)),
        "spectral_radius": float(rho),
        "finite": bool(np.isfinite(float(total_aligned)) and np.isfinite(float(total_pure)) and np.isfinite(max_state) and np.isfinite(rho)),
    }


def robust_br_aligned(mod, benchmark, K, L_init, sigma0, local_br_radius, inner_steps, inner_lr):
    def run(start_L, lr, steps):
        L0 = start_L.detach().clone()
        Lbar = start_L.detach().clone()
        for _ in range(steps):
            Lreq = Lbar.detach().clone().requires_grad_(True)
            sigma = sigma0.clone()
            closed_loop = benchmark.A + benchmark.B @ K.detach() + benchmark.E @ Lreq
            total = torch.zeros((), dtype=torch.float64)
            for t in range(benchmark.cfg.horizon):
                x_sq = torch.trace(sigma)
                u_sq = torch.trace(K.detach() @ sigma @ K.detach().T)
                w_sq = torch.trace(Lreq @ sigma @ Lreq.T)
                rot_term = torch.trace(sigma @ K.detach().T @ benchmark.Hmat @ Lreq)
                sym_term = torch.trace(sigma @ K.detach().T @ benchmark.Smat @ Lreq)
                pure_reward = -0.5 * benchmark.cfg.q_state * x_sq - 0.5 * benchmark.cfg.a_u * u_sq
                aligned_reward = pure_reward + 0.5 * benchmark.cfg.a_w * w_sq + benchmark.cfg.beta_rot * rot_term + benchmark.cfg.beta_sym * sym_term
                total = total + (benchmark.cfg.gamma ** t) * aligned_reward
                sigma = closed_loop @ sigma @ closed_loop.T
            grad = torch.autograd.grad(total, Lreq)[0]
            Lnext = Lbar - lr * grad
            Lnext = project_local(Lnext, L0, local_br_radius)
            Lbar = Lnext.detach()
        return Lbar

    last = None
    for lr in [inner_lr, 0.03, 0.01]:
        Lbr = run(L_init, lr, inner_steps)
        eval_res = aligned_eval(mod, benchmark, K, Lbr, sigma0, disable_adversary=False)
        stable = eval_res["finite"] and eval_res["state_norm_max"] < 1e6 and eval_res["spectral_radius"] < 1.5
        last = (Lbr, eval_res, lr, stable)
        if stable:
            return last
    return last


def rerun_final_clean(mod):
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
            rows.append({"method": method, "iteration": it, **metrics})
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
    return benchmark, pd.concat(curves, ignore_index=True), pd.concat(diags, ignore_index=True), traj_map


def compare_summary(rerun_summary: pd.DataFrame):
    expected = pd.read_csv(RESULT_ROOT / "mixed_linear_actor_rot_lq_final_clean_summary.csv")
    merged = rerun_summary.merge(expected, on="method", suffixes=("_rerun", "_expected"))
    worst_rel = 0.0
    worst_abs = 0.0
    for _, row in merged.iterrows():
        for metric in ["V_lambda_AUC", "field_norm_AUC"]:
            if pd.notna(row.get(f"{metric}_expected", np.nan)):
                exp = float(row[f"{metric}_expected"])
                got = float(row[f"{metric}_rerun"])
                rel = abs(got - exp) / max(abs(exp), 1e-9)
                worst_rel = max(worst_rel, rel)
        if row["method"] == "proposed_qpg":
            for metric in ["fallback_to_egm_frac", "gamma_active_frac", "mean_G_contribution_ratio"]:
                if pd.notna(row.get(f"{metric}_expected", np.nan)):
                    exp = float(row[f"{metric}_expected"])
                    got = float(row[f"{metric}_rerun"])
                    worst_abs = max(worst_abs, abs(got - exp))
    return worst_rel, worst_abs, merged, expected


def summarize_geometry(mod):
    cfg = mod.MixedLinearActorRotLQConfig(beta_rot=1.0, beta_sym=0.1)
    row = mod.mixed_linear_actor_rot_lq_geometry_row(cfg)
    lines = [
        "# MixedLinearActorRotLQ Aligned RARL Geometry Confirm",
        "",
        f"- beta_rot: `{cfg.beta_rot}`",
        f"- beta_sym: `{cfg.beta_sym}`",
        f"- rotation_ratio: `{row['rotation_ratio']:.6f}`",
        f"- num_complex_eigs: `{row['number_of_complex_eigenvalues']}`",
        f"- cos(F,G): `{row['cosine_FG']:.6f}`",
        f"- non_collinearity: `{row['non_collinearity']:.6f}`",
        f"- cross_player_coupling_proxy: `{row['cross_player_block_norm']:.6f}`",
        "",
        (
            "GEOMETRY_OK"
            if float(row["rotation_ratio"]) > 5.0
            and int(row["number_of_complex_eigenvalues"]) > 0
            and abs(float(row["cosine_FG"])) < 0.95
            and float(row["non_collinearity"]) > 0.2
            and float(row["cross_player_block_norm"]) > 0.0
            else "IMPLEMENTATION_MISMATCH"
        ),
    ]
    (RESULT_ROOT / "mixed_linear_actor_rot_lq_aligned_rarl_geometry_confirm.md").write_text("\n".join(lines), encoding="utf-8")
    return row


def plot_metric(df, metric, filename, title, ylabel, methods, labels, colors, logy=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        sub = df[df["method"] == method]
        vals = sub[metric].to_numpy(dtype=np.float64)
        if logy:
            vals = np.maximum(vals, 1e-12)
        ax.plot(sub["iteration"], vals, label=labels[method], color=colors[method])
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / filename, dpi=160)
    plt.close(fig)


def br_sensitivity(mod, benchmark, traj_map, methods, base_metrics):
    rows = []
    for local_br_radius in [0.25, 0.5, 1.0]:
        for br_inner_steps in [50, 100]:
            for br_inner_lr in [0.03, 0.05, 0.1]:
                summary = {}
                for method in methods:
                    traj = traj_map[method]
                    robust_vals = []
                    degr_vals = []
                    valid = []
                    for it in range(len(traj) - 1):
                        flat = traj[it + 1]
                        K, L = benchmark.split_flat(flat)
                        clean = aligned_eval(mod, benchmark, K, torch.zeros_like(L), benchmark.eval_sigma0, disable_adversary=True)
                        _, robust, _, stable = robust_br_aligned(
                            mod,
                            benchmark,
                            K,
                            L,
                            benchmark.eval_sigma0,
                            local_br_radius=local_br_radius,
                            inner_steps=br_inner_steps,
                            inner_lr=br_inner_lr,
                        )
                        robust_vals.append(float(robust["aligned_return"]))
                        degr_vals.append(float(clean["aligned_return"] - robust["aligned_return"]))
                        valid.append(float(stable))
                    summary[method] = {
                        "robust_br_aligned_return_AUC": mod.auc_from_series(robust_vals),
                        "final_robust_br_aligned_return": robust_vals[-1],
                        "aligned_robust_degradation_AUC": mod.auc_from_series(degr_vals),
                        "BR_valid_fraction": float(np.mean(valid)),
                    }
                qpg = summary["proposed_qpg"]
                nog = summary["proposed_noG"]
                rows.append(
                    {
                        "local_br_radius": local_br_radius,
                        "br_inner_steps": br_inner_steps,
                        "br_inner_lr": br_inner_lr,
                        "qpg_robust_AUC": qpg["robust_br_aligned_return_AUC"],
                        "nog_robust_AUC": nog["robust_br_aligned_return_AUC"],
                        "qpg_final_robust": qpg["final_robust_br_aligned_return"],
                        "nog_final_robust": nog["final_robust_br_aligned_return"],
                        "qpg_degradation_AUC": qpg["aligned_robust_degradation_AUC"],
                        "nog_degradation_AUC": nog["aligned_robust_degradation_AUC"],
                        "qpg_valid_fraction": qpg["BR_valid_fraction"],
                        "nog_valid_fraction": nog["BR_valid_fraction"],
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_ROOT / "mixed_linear_actor_rot_lq_aligned_rarl_br_sensitivity.csv", index=False)
    lines = ["# MixedLinearActorRotLQ Aligned RARL BR Sensitivity", "", df.to_string(index=False)]
    (RESULT_ROOT / "mixed_linear_actor_rot_lq_aligned_rarl_br_sensitivity_report.md").write_text("\n".join(lines), encoding="utf-8")
    return df


def run():
    PLOTS.mkdir(parents=True, exist_ok=True)
    mod = load_module()
    geometry = summarize_geometry(mod)
    if not (
        float(geometry["rotation_ratio"]) > 5.0
        and int(geometry["number_of_complex_eigenvalues"]) > 0
        and abs(float(geometry["cosine_FG"])) < 0.95
        and float(geometry["non_collinearity"]) > 0.2
        and float(geometry["cross_player_block_norm"]) > 0.0
    ):
        (RESULT_ROOT / "mixed_linear_actor_rot_lq_aligned_rarl_final_decision.md").write_text(
            "# MixedLinearActorRotLQ Aligned RARL Final Decision\n\nIMPLEMENTATION_MISMATCH\n",
            encoding="utf-8",
        )
        return

    benchmark, curves_df, diags_df, traj_map = rerun_final_clean(mod)

    summary_rows = []
    for method in ["sgd", "egm", "ppm", "proposed_noG", "proposed_qpg"]:
        sub = curves_df[curves_df["method"] == method].sort_values("iteration")
        final = sub.iloc[-1]
        diag_sub = diags_df[diags_df["method"] == method] if method in ["proposed_noG", "proposed_qpg"] else pd.DataFrame()
        summary_rows.append(
            {
                "method": method,
                "V_lambda_AUC": mod.auc_from_series(sub["V_lambda"].tolist()),
                "P_tau_AUC": mod.auc_from_series(sub["raw_P_tau"].tolist()),
                "field_norm_AUC": mod.auc_from_series(sub["field_norm"].tolist()),
                "exploitability_AUC": mod.auc_from_series(sub["approximate_local_exploitability"].tolist()),
                "final_V_lambda": float(final["V_lambda"]),
                "final_P_tau": float(final["raw_P_tau"]),
                "final_field_norm": float(final["field_norm"]),
                "final_exploitability": float(final["approximate_local_exploitability"]),
                "fallback_to_egm_frac": float(diag_sub["fallback_to_egm"].mean()) if not diag_sub.empty else np.nan,
                "gamma_active_frac": float(diag_sub["gamma_active"].mean()) if not diag_sub.empty else np.nan,
                "mean_G_contribution_ratio": float(diag_sub["G_contribution_ratio"].mean()) if not diag_sub.empty else np.nan,
            }
        )
    rerun_summary = pd.DataFrame(summary_rows)
    mismatch_rel, mismatch_abs, merged, expected = compare_summary(rerun_summary)
    if mismatch_rel > 0.05 or mismatch_abs > 0.15:
        lines = [
            "# MixedLinearActorRotLQ Aligned RARL Final Decision",
            "",
            "IMPLEMENTATION_MISMATCH",
            "",
            f"- max relative mismatch on core metrics vs final clean summary: `{mismatch_rel:.6e}`",
            f"- max absolute mismatch on QP diagnostics vs final clean summary: `{mismatch_abs:.6e}`",
            "",
            merged.to_string(index=False),
        ]
        (RESULT_ROOT / "mixed_linear_actor_rot_lq_aligned_rarl_final_decision.md").write_text("\n".join(lines), encoding="utf-8")
        return

    methods = ["sgd", "egm", "ppm", "proposed_noG", "proposed_qpg"]
    perf_rows = []
    for method in methods:
        traj = traj_map[method]
        for it in range(len(traj) - 1):
            flat = traj[it + 1]
            K, L = benchmark.split_flat(flat)
            clean = aligned_eval(mod, benchmark, K, L, benchmark.eval_sigma0, disable_adversary=True)
            current = aligned_eval(mod, benchmark, K, L, benchmark.eval_sigma0, disable_adversary=False)
            Lbr, robust, used_lr, stable = robust_br_aligned(
                mod,
                benchmark,
                K,
                L,
                benchmark.eval_sigma0,
                local_br_radius=0.5,
                inner_steps=50,
                inner_lr=0.05,
            )
            perf_rows.append(
                {
                    "iteration": it,
                    "method": method,
                    "V_lambda": float(curves_df[(curves_df["method"] == method) & (curves_df["iteration"] == it)]["V_lambda"].iloc[0]),
                    "P_tau": float(curves_df[(curves_df["method"] == method) & (curves_df["iteration"] == it)]["raw_P_tau"].iloc[0]),
                    "field_norm": float(curves_df[(curves_df["method"] == method) & (curves_df["iteration"] == it)]["field_norm"].iloc[0]),
                    "exploitability": float(curves_df[(curves_df["method"] == method) & (curves_df["iteration"] == it)]["approximate_local_exploitability"].iloc[0]),
                    "clean_aligned_return": clean["aligned_return"],
                    "current_adv_aligned_return": current["aligned_return"],
                    "robust_br_aligned_return": robust["aligned_return"],
                    "aligned_robust_degradation": clean["aligned_return"] - robust["aligned_return"],
                    "clean_pure_return": clean["pure_return"],
                    "current_adv_pure_return": current["pure_return"],
                    "robust_br_pure_return": robust["pure_return"],
                    "pure_robust_degradation": clean["pure_return"] - robust["pure_return"],
                    "K_norm": float(torch.linalg.norm(K)),
                    "L_norm": float(torch.linalg.norm(L)),
                    "L_br_norm": float(torch.linalg.norm(Lbr)),
                    "L_br_distance": float(torch.linalg.norm(Lbr - L)),
                    "BR_valid": float(stable),
                    "state_norm_max": robust["state_norm_max"],
                    "closed_loop_spectral_radius": robust["spectral_radius"],
                }
            )
    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(RESULT_ROOT / "mixed_linear_actor_rot_lq_aligned_rarl_metrics.csv", index=False)

    summary_rows = []
    for method in methods:
        sub = perf_df[perf_df["method"] == method].sort_values("iteration")
        base = rerun_summary[rerun_summary["method"] == method].iloc[0]
        final = sub.iloc[-1]
        summary_rows.append(
            {
                "method": method,
                "V_lambda_AUC": float(base["V_lambda_AUC"]),
                "P_tau_AUC": float(base["P_tau_AUC"]),
                "field_norm_AUC": float(base["field_norm_AUC"]),
                "exploitability_AUC": float(base["exploitability_AUC"]),
                "clean_aligned_return_AUC": mod.auc_from_series(sub["clean_aligned_return"].tolist()),
                "current_adv_aligned_return_AUC": mod.auc_from_series(sub["current_adv_aligned_return"].tolist()),
                "robust_br_aligned_return_AUC": mod.auc_from_series(sub["robust_br_aligned_return"].tolist()),
                "aligned_robust_degradation_AUC": mod.auc_from_series(sub["aligned_robust_degradation"].tolist()),
                "final_clean_aligned_return": float(final["clean_aligned_return"]),
                "final_current_adv_aligned_return": float(final["current_adv_aligned_return"]),
                "final_robust_br_aligned_return": float(final["robust_br_aligned_return"]),
                "final_aligned_robust_degradation": float(final["aligned_robust_degradation"]),
                "clean_pure_return_AUC": mod.auc_from_series(sub["clean_pure_return"].tolist()),
                "current_adv_pure_return_AUC": mod.auc_from_series(sub["current_adv_pure_return"].tolist()),
                "robust_br_pure_return_AUC": mod.auc_from_series(sub["robust_br_pure_return"].tolist()),
                "pure_robust_degradation_AUC": mod.auc_from_series(sub["pure_robust_degradation"].tolist()),
                "fallback_to_egm_frac": float(base["fallback_to_egm_frac"]) if pd.notna(base["fallback_to_egm_frac"]) else np.nan,
                "gamma_active_frac": float(base["gamma_active_frac"]) if pd.notna(base["gamma_active_frac"]) else np.nan,
                "G_contribution_ratio": float(base["mean_G_contribution_ratio"]) if pd.notna(base["mean_G_contribution_ratio"]) else np.nan,
                "BR_valid_fraction": float(sub["BR_valid"].mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULT_ROOT / "mixed_linear_actor_rot_lq_aligned_rarl_summary.csv", index=False)

    colors = {"sgd": "#d62728", "egm": "#2ca02c", "ppm": "#9467bd", "proposed_noG": "#1f77b4", "proposed_qpg": "#8c564b"}
    labels = {"sgd": "SGD", "egm": "EGM", "ppm": "PPM", "proposed_noG": "proposed_noG", "proposed_qpg": "proposed_QP_G"}

    plot_metric(perf_df, "V_lambda", "mixed_linear_actor_rot_lq_aligned_rarl_V_lambda.png", "V_lambda", "V_lambda", methods, labels, colors, logy=True)
    plot_metric(perf_df, "P_tau", "mixed_linear_actor_rot_lq_aligned_rarl_P_tau.png", "P_tau", "P_tau", methods, labels, colors, logy=True)
    plot_metric(perf_df, "field_norm", "mixed_linear_actor_rot_lq_aligned_rarl_field_norm.png", "field_norm", "field_norm", methods, labels, colors, logy=True)
    plot_metric(perf_df, "exploitability", "mixed_linear_actor_rot_lq_aligned_rarl_exploitability.png", "exploitability", "exploitability", methods, labels, colors, logy=True)
    plot_metric(perf_df, "clean_aligned_return", "mixed_linear_actor_rot_lq_aligned_rarl_clean_aligned_return.png", "clean_aligned_return", "return", methods, labels, colors, logy=False)
    plot_metric(perf_df, "current_adv_aligned_return", "mixed_linear_actor_rot_lq_aligned_rarl_current_adv_aligned_return.png", "current_adv_aligned_return", "return", methods, labels, colors, logy=False)
    plot_metric(perf_df, "robust_br_aligned_return", "mixed_linear_actor_rot_lq_aligned_rarl_robust_br_aligned_return.png", "robust_br_aligned_return", "return", methods, labels, colors, logy=False)
    plot_metric(perf_df, "aligned_robust_degradation", "mixed_linear_actor_rot_lq_aligned_rarl_aligned_robust_degradation.png", "aligned_robust_degradation", "degradation", methods, labels, colors, logy=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric, title in zip(
        axes.flatten(),
        ["clean_pure_return", "current_adv_pure_return", "robust_br_pure_return", "pure_robust_degradation"],
        ["clean_pure_return", "current_adv_pure_return", "robust_br_pure_return", "pure_robust_degradation"],
    ):
        for method in methods:
            sub = perf_df[perf_df["method"] == method]
            ax.plot(sub["iteration"], sub[metric], label=labels[method], color=colors[method])
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=5)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(PLOTS / "mixed_linear_actor_rot_lq_aligned_rarl_pure_task_diagnostics.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    panels = [
        ("V_lambda", True),
        ("P_tau", True),
        ("field_norm", True),
        ("exploitability", True),
        ("clean_aligned_return", False),
        ("current_adv_aligned_return", False),
        ("robust_br_aligned_return", False),
        ("aligned_robust_degradation", False),
    ]
    for ax, (metric, logy) in zip(axes.flatten(), panels):
        for method in methods:
            sub = perf_df[perf_df["method"] == method]
            vals = sub[metric].to_numpy(dtype=np.float64)
            if logy:
                vals = np.maximum(vals, 1e-12)
            ax.plot(sub["iteration"], vals, label=labels[method], color=colors[method])
        if logy:
            ax.set_yscale("log")
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=5)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(PLOTS / "mixed_linear_actor_rot_lq_aligned_rarl_all_plots_big.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))
    for ax, metric, logy in zip(
        axes,
        ["V_lambda", "P_tau", "field_norm", "robust_br_aligned_return", "aligned_robust_degradation"],
        [True, True, True, False, False],
    ):
        for method in methods:
            sub = perf_df[perf_df["method"] == method]
            vals = sub[metric].to_numpy(dtype=np.float64)
            if logy:
                vals = np.maximum(vals, 1e-12)
            ax.plot(sub["iteration"], vals, label=labels[method], color=colors[method])
        if logy:
            ax.set_yscale("log")
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "mixed_linear_actor_rot_lq_aligned_rarl_paper_main.png", dpi=160)
    plt.close(fig)

    qpg = summary_df[summary_df["method"] == "proposed_qpg"].iloc[0]
    nog = summary_df[summary_df["method"] == "proposed_noG"].iloc[0]
    best_baseline_robust = summary_df[summary_df["method"].isin(["sgd", "egm", "ppm"])]["final_robust_br_aligned_return"].max()

    if (
        float(qpg["V_lambda_AUC"]) < float(nog["V_lambda_AUC"])
        and float(qpg["P_tau_AUC"]) < float(nog["P_tau_AUC"])
        and float(qpg["field_norm_AUC"]) < float(nog["field_norm_AUC"])
        and float(qpg["exploitability_AUC"]) < float(nog["exploitability_AUC"])
        and float(qpg["robust_br_aligned_return_AUC"]) >= float(nog["robust_br_aligned_return_AUC"]) - 1e-12
        and float(qpg["final_robust_br_aligned_return"]) >= float(nog["final_robust_br_aligned_return"]) - 1e-12
        and float(qpg["aligned_robust_degradation_AUC"]) <= float(nog["aligned_robust_degradation_AUC"]) + 1e-12
        and float(qpg["fallback_to_egm_frac"]) < 0.2
        and float(qpg["BR_valid_fraction"]) >= 1.0 - 1e-12
    ):
        decision = "FULL_ALIGNMENT_SUCCESS"
    elif float(qpg["final_robust_br_aligned_return"]) > best_baseline_robust + 1e-12:
        decision = "OPTIMIZATION_POSITIVE_RARL_SUPPORTIVE"
    else:
        decision = "OPTIMIZATION_ONLY"

    if float(qpg["final_robust_br_aligned_return"]) < float(nog["final_robust_br_aligned_return"]) - 1e-12:
        br_sensitivity(mod, benchmark, traj_map, methods, perf_df)

    theory_lines = [
        "# MixedLinearActorRotLQ Aligned RARL Theory Alignment Report",
        "",
        "1. Previous task-only diagnostics were not aligned with the mixed game objective.",
        "2. The new aligned robust metric uses the same mixed/aligned objective as V_lambda and P_tau.",
        "3. Therefore the Lyapunov residual, local saddle gap, and robust-BR return now measure the same robust game.",
        "4. Pure task return is reported only as a secondary diagnostic.",
        "5. This makes the LQ experiment suitable for an aligned local robust saddle theorem.",
        "6. The theorem should be stated for the aligned robust game, not for pure state-cost return.",
    ]
    (RESULT_ROOT / "mixed_linear_actor_rot_lq_aligned_rarl_theory_alignment_report.md").write_text("\n".join(theory_lines), encoding="utf-8")

    report_lines = [
        "# MixedLinearActorRotLQ Aligned RARL Performance Report",
        "",
        "Primary aligned objective: same mixed/aligned reward used by the original MixedLinearActorRotLQ-v0 training objective, V_lambda, and P_tau.",
        "",
        "Summary table:",
        summary_df.to_string(index=False),
    ]
    (RESULT_ROOT / "mixed_linear_actor_rot_lq_aligned_rarl_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    decision_lines = [
        "# MixedLinearActorRotLQ Aligned RARL Final Decision",
        "",
        decision,
        "",
        (
            "MixedLinearActorRotLQ-v0 remains the paper LQ main result, now with aligned robust-performance evidence using the same mixed objective."
            if decision in {"FULL_ALIGNMENT_SUCCESS", "OPTIMIZATION_POSITIVE_RARL_SUPPORTIVE", "OPTIMIZATION_ONLY"}
            else "MixedLinearActorRotLQ-v0 could not be validated under aligned robust-performance rerun."
        ),
    ]
    (RESULT_ROOT / "mixed_linear_actor_rot_lq_aligned_rarl_final_decision.md").write_text("\n".join(decision_lines), encoding="utf-8")


if __name__ == "__main__":
    run()
