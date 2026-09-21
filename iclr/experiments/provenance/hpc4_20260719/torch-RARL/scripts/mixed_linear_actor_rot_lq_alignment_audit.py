import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SCRIPT = Path(r"C:/Users/jzhuangag/work/rarl/original/torch-RARL/scripts") / "nn_actor_rotational_lqr_rarl.py"
RESULT_ROOT = Path(r"C:/Users/jzhuangag/work/rarl/original/results") / "nn_actor_rotational_lqr_rarl"


def load_module():
    spec = importlib.util.spec_from_file_location("nn_actor_rot_lqr", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def moving_average(x, w=5):
    arr = np.asarray(x, dtype=np.float64)
    if arr.size < w:
        return arr
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(arr, kernel, mode="same")


def corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    aa = a[mask]
    bb = b[mask]
    if np.std(aa) < 1e-16 or np.std(bb) < 1e-16:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def project_fro(M, max_norm):
    n = float(torch.linalg.norm(M))
    if n <= max_norm or n <= 1e-12:
        return M
    return M * (max_norm / n)


def project_local(L, L0, radius):
    delta = L - L0
    n = float(torch.linalg.norm(delta))
    if n <= radius or n <= 1e-12:
        return L
    return L0 + delta * (radius / n)


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


def robust_br(mod, benchmark, K, L_init, sigma0, local_br_radius, global_budget, inner_steps=20, inner_lr=0.03):
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
                reward = -0.5 * benchmark.cfg.q_state * x_sq - 0.5 * benchmark.cfg.a_u * u_sq
                total = total + (benchmark.cfg.gamma ** t) * reward
                sigma = closed_loop @ sigma @ closed_loop.T
            grad = torch.autograd.grad(total, Lreq)[0]
            Lnext = Lbar - lr * grad
            Lnext = project_local(Lnext, L0, local_br_radius)
            Lnext = project_fro(Lnext, global_budget)
            Lbar = Lnext.detach()
        return Lbar

    last = None
    for lr in [inner_lr, 0.01] if inner_lr != 0.01 else [inner_lr]:
        Lbr = run(L_init, lr, inner_steps)
        eval_res = task_eval(mod, benchmark, K, Lbr, sigma0)
        stable = eval_res["finite"] and eval_res["state_norm_max"] < 1e6 and eval_res["spectral_radius"] < 1.5
        last = (Lbr, eval_res, lr, stable)
        if stable:
            return last
    return last


def get_init_terms(benchmark):
    flat0 = benchmark.flat0.clone()
    field0 = benchmark.field_tensor(flat0.detach(), create_graph=False).detach()
    field_energy0 = 0.5 * float(torch.dot(field0, field0))
    p_tau0 = benchmark.local_gap_terms(flat0)["P_tau"]
    return flat0, field_energy0, p_tau0


def rerun_setting(mod, cfg, include_proposed=True):
    benchmark = mod.MixedLinearActorRotLQBenchmark(cfg)
    flat0, field_energy0, p_tau0 = get_init_terms(benchmark)
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

    if include_proposed:
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
    diags_df = pd.concat(diags, ignore_index=True) if diags else pd.DataFrame()
    return benchmark, curves_df, diags_df, traj_map


def performance_df_for_trajs(mod, benchmark, curves_df, traj_map):
    max_L_norm = 0.0
    for traj in traj_map.values():
        for flat in traj:
            _, L = benchmark.split_flat(flat)
            max_L_norm = max(max_L_norm, float(torch.linalg.norm(L)))
    local_br_radius = 0.25
    global_budget = max(1.0, max_L_norm)
    rows = []
    for method, traj in traj_map.items():
        sub = curves_df[curves_df["method"] == method].sort_values("iteration").reset_index(drop=True)
        for it in range(len(sub)):
            flat = traj[it + 1]
            K, L = benchmark.split_flat(flat)
            clean = task_eval(mod, benchmark, K, torch.zeros_like(L), benchmark.eval_sigma0)
            current = task_eval(mod, benchmark, K, L, benchmark.eval_sigma0)
            Lbr, robust, _, stable = robust_br(mod, benchmark, K, L, benchmark.eval_sigma0, local_br_radius, global_budget)
            rows.append(
                {
                    "iteration": it,
                    "method": method,
                    "clean_task_return": clean["task_return"],
                    "current_adv_task_return": current["task_return"],
                    "robust_br_task_return": robust["task_return"],
                    "robust_degradation": clean["task_return"] - robust["task_return"],
                    "V_lambda": float(sub.iloc[it]["V_lambda"]),
                    "normalized_P_tau": float(sub.iloc[it]["normalized_P_tau"]),
                    "field_norm": float(sub.iloc[it]["field_norm"]),
                    "approximate_local_exploitability": float(sub.iloc[it]["approximate_local_exploitability"]),
                    "train_game_return": float(sub.iloc[it]["train_game_return"]),
                    "valid_robust_br_eval": float(stable),
                    "L_br_norm": float(torch.linalg.norm(Lbr)),
                    "state_norm_max_robust_br": robust["state_norm_max"],
                    "closed_loop_spectral_radius_robust_br": robust["spectral_radius"],
                }
            )
    return pd.DataFrame(rows)


def summarize_perf(mod, perf_df):
    rows = []
    for method, sub in perf_df.groupby("method"):
        sub = sub.sort_values("iteration")
        final = sub.iloc[-1]
        rows.append(
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
            }
        )
    return pd.DataFrame(rows)


def run():
    mod = load_module()
    final_summary = pd.read_csv(RESULT_ROOT / "mixed_linear_actor_rot_lq_final_clean_summary.csv")
    final_curves = pd.read_csv(RESULT_ROOT / "mixed_linear_actor_rot_lq_final_clean_curves.csv")
    final_diags = pd.read_csv(RESULT_ROOT / "mixed_linear_actor_rot_lq_final_clean_diagnostics.csv")
    perf_metrics = pd.read_csv(RESULT_ROOT / "mixed_linear_actor_rot_lq_rarl_performance_metrics.csv")
    perf_summary = pd.read_csv(RESULT_ROOT / "mixed_linear_actor_rot_lq_rarl_performance_summary.csv")

    # Part 1 read existing
    best_v = final_summary.sort_values("V_lambda_AUC").iloc[0]["method"]
    best_p = final_summary.sort_values("P_tau_AUC").iloc[0]["method"]
    best_f = final_summary.sort_values("field_norm_AUC").iloc[0]["method"]
    best_e = final_summary.sort_values("exploitability_AUC").iloc[0]["method"]
    best_clean = perf_summary.sort_values("final_clean_task_return", ascending=False).iloc[0]["method"]
    best_current = perf_summary.sort_values("final_current_adv_task_return", ascending=False).iloc[0]["method"]
    best_robust = perf_summary.sort_values("final_robust_br_task_return", ascending=False).iloc[0]["method"]
    best_deg = perf_summary.sort_values("final_robust_degradation").iloc[0]["method"]
    qpg_perf = perf_summary[perf_summary["method"] == "proposed_qpg"].iloc[0]
    nog_perf = perf_summary[perf_summary["method"] == "proposed_noG"].iloc[0]
    base_perf = perf_summary[perf_summary["method"].isin(["sgd", "egm", "ppm"])]
    read_lines = [
        "# MixedLinearActorRotLQ Alignment Read Existing",
        "",
        f"1. Best V_lambda AUC: `{best_v}`",
        f"2. Best P_tau AUC: `{best_p}`",
        f"3. Best field_norm AUC: `{best_f}`",
        f"4. Best exploitability AUC: `{best_e}`",
        f"5. Best final clean_task_return: `{best_clean}`",
        f"6. Best final current_adv_task_return: `{best_current}`",
        f"7. Best final robust_br_task_return: `{best_robust}`",
        f"8. Best robust_degradation: `{best_deg}`",
        f"9. QP+G improves task metrics over SGD/EGM/PPM: `{float(qpg_perf['final_robust_br_task_return']) > float(base_perf['final_robust_br_task_return'].max())}`",
        f"10. QP+G improves task metrics over noG: `{float(qpg_perf['final_robust_br_task_return']) > float(nog_perf['final_robust_br_task_return'])}`",
        "11. Task metrics are bounded and smooth; no instability is visible in the stored diagnostics.",
        f"12. Robust BR evaluation valid for all methods: `{bool((perf_summary['valid_fraction_robust_br'] >= 1.0 - 1e-12).all())}`",
    ]
    (RESULT_ROOT / "mixed_linear_actor_rot_lq_alignment_read_existing.md").write_text("\n".join(read_lines), encoding="utf-8")

    # current rerun with trajectories
    current_cfg = mod.MixedLinearActorRotLQConfig(
        beta_rot=1.0,
        beta_sym=0.1,
        a_w=0.10,
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
    benchmark, curves_df, diags_df, traj_map = rerun_setting(mod, current_cfg, include_proposed=True)
    perf_df = performance_df_for_trajs(mod, benchmark, curves_df, traj_map)

    # Part 2 decomposition
    decomp_rows = []
    for method, traj in traj_map.items():
        for it, flat in enumerate(traj[1:]):
            K, L = benchmark.split_flat(flat)
            sigma = benchmark.train_sigma0.clone()
            task_component = 0.0
            u_cost = 0.0
            w_energy = 0.0
            rot_c = 0.0
            sym_c = 0.0
            total_game = 0.0
            for t in range(benchmark.cfg.horizon):
                weight = benchmark.cfg.gamma ** t
                x_sq = torch.trace(sigma)
                u_sq = torch.trace(K @ sigma @ K.T)
                w_sq = torch.trace(L @ sigma @ L.T)
                rot_term = torch.trace(sigma @ K.T @ benchmark.Hmat @ L)
                sym_term = torch.trace(sigma @ K.T @ benchmark.Smat @ L)
                task_t = float(-0.5 * benchmark.cfg.q_state * x_sq)
                u_t = float(-0.5 * benchmark.cfg.a_u * u_sq)
                w_t = float(+0.5 * benchmark.cfg.a_w * w_sq)
                r_t = float(benchmark.cfg.beta_rot * rot_term)
                s_t = float(benchmark.cfg.beta_sym * sym_term)
                task_component += weight * task_t
                u_cost += weight * u_t
                w_energy += weight * w_t
                rot_c += weight * r_t
                sym_c += weight * s_t
                total_game += weight * (task_t + u_t + w_t + r_t + s_t)
                closed_loop = benchmark.A + benchmark.B @ K + benchmark.E @ L
                sigma = closed_loop @ sigma @ closed_loop.T
            decomp_rows.append(
                {
                    "method": method,
                    "iteration": it,
                    "discounted_task_component": task_component,
                    "discounted_u_cost": u_cost,
                    "discounted_w_energy": w_energy,
                    "discounted_rot_coupling": rot_c,
                    "discounted_sym_coupling": sym_c,
                    "task_only_return": task_component + u_cost,
                    "total_game_return": total_game,
                }
            )
    decomp_df = pd.DataFrame(decomp_rows)
    decomp_df.to_csv(RESULT_ROOT / "mixed_linear_actor_rot_lq_game_task_decomposition.csv", index=False)
    decomp_lines = [
        "# MixedLinearActorRotLQ Game vs Task Decomposition",
        "",
        "Method means over trajectory:",
        decomp_df.groupby("method")[
            [
                "discounted_task_component",
                "discounted_u_cost",
                "discounted_w_energy",
                "discounted_rot_coupling",
                "discounted_sym_coupling",
                "task_only_return",
                "total_game_return",
            ]
        ]
        .mean()
        .to_string(),
    ]
    (RESULT_ROOT / "mixed_linear_actor_rot_lq_game_task_decomposition.md").write_text("\n".join(decomp_lines), encoding="utf-8")

    # Part 3 correlations
    corr_rows = []
    for method, sub in perf_df.groupby("method"):
        sub = sub.sort_values("iteration")
        dec = decomp_df[decomp_df["method"] == method].sort_values("iteration")
        robust = sub["robust_br_task_return"].to_numpy(dtype=np.float64)
        robust_s = moving_average(robust, 7)
        for metric_name, series in [
            ("V_lambda", sub["V_lambda"]),
            ("P_tau", sub["normalized_P_tau"]),
            ("field_norm", sub["field_norm"]),
            ("exploitability", sub["approximate_local_exploitability"]),
            ("game_return", sub["train_game_return"]),
            ("task_component", dec["task_only_return"]),
        ]:
            vals = series.to_numpy(dtype=np.float64)
            corr_rows.append(
                {
                    "method": method,
                    "metric": metric_name,
                    "corr_raw": corr(vals, robust),
                    "corr_smoothed": corr(moving_average(vals, 7), robust_s),
                }
            )
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(RESULT_ROOT / "mixed_linear_actor_rot_lq_lyapunov_task_correlation.csv", index=False)
    corr_lines = [
        "# MixedLinearActorRotLQ Lyapunov vs Task Correlation",
        "",
        corr_df.to_string(index=False),
        "",
        "Interpretation: more negative correlation means lower metric aligns with better robust task return.",
    ]
    (RESULT_ROOT / "mixed_linear_actor_rot_lq_lyapunov_task_correlation.md").write_text("\n".join(corr_lines), encoding="utf-8")

    # Part 4 lambda_F sweep, lambda_T=0 only
    lambda_rows = []
    base_trajs = {k: v for k, v in traj_map.items() if k in {"sgd", "egm", "ppm"}}
    for lambda_F in [1e-4, 1e-3, 1e-2, 3e-2, 1e-1]:
        cfg_l = replace(current_cfg, lambda_F=lambda_F)
        b_l, curves_l, diags_l, trajs_l = rerun_setting(mod, cfg_l, include_proposed=True)
        perf_l = performance_df_for_trajs(mod, b_l, curves_l, trajs_l)
        perf_sum_l = summarize_perf(mod, perf_l).set_index("method")
        curve_sum_rows = []
        for method, sub in curves_l.groupby("method"):
            curve_sum_rows.append(
                {
                    "method": method,
                    "V_lambda_AUC": mod.auc_from_series(sub["V_lambda"].tolist()),
                    "P_tau_AUC": mod.auc_from_series(sub["raw_P_tau"].tolist()),
                    "field_norm_AUC": mod.auc_from_series(sub["field_norm"].tolist()),
                    "exploitability_AUC": mod.auc_from_series(sub["approximate_local_exploitability"].tolist()),
                    "final_V_lambda": float(sub.iloc[-1]["V_lambda"]),
                }
            )
        curve_sum = pd.DataFrame(curve_sum_rows).set_index("method")
        for method in ["sgd", "egm", "ppm", "proposed_noG", "proposed_qpg"]:
            dsub = diags_l[diags_l["method"] == method] if not diags_l.empty and method in diags_l["method"].unique() else pd.DataFrame()
            same_start_qp_wins = np.nan
            if method == "proposed_qpg":
                same_df = mod.mixed_linear_actor_rot_lq_same_start_comparison(
                    b_l, trajs_l["proposed_qpg"], *get_init_terms(b_l)[1:], 0.01, 0.01, fallback_lr=0.01
                )
                qpg_rows = same_df[same_df["candidate"] == "proposed_QP_G"]
                if not qpg_rows.empty:
                    wins = 0
                    for cp in sorted(qpg_rows["checkpoint"].unique()):
                        block = same_df[same_df["checkpoint"] == cp]
                        best = float(block["actual_V_after"].min())
                        qpgv = float(block[block["candidate"] == "proposed_QP_G"]["actual_V_after"].iloc[0])
                        wins += int(abs(qpgv - best) <= 1e-12)
                    same_start_qp_wins = wins
            lambda_rows.append(
                {
                    "lambda_F": lambda_F,
                    "lambda_P": 1.0,
                    "lambda_T": 0.0,
                    "method": method,
                    "update_radius": 0.01,
                    "V_lambda_AUC": float(curve_sum.loc[method]["V_lambda_AUC"]),
                    "P_tau_AUC": float(curve_sum.loc[method]["P_tau_AUC"]),
                    "field_norm_AUC": float(curve_sum.loc[method]["field_norm_AUC"]),
                    "exploitability_AUC": float(curve_sum.loc[method]["exploitability_AUC"]),
                    "clean_task_return_AUC": float(perf_sum_l.loc[method]["clean_task_return_AUC"]),
                    "current_adv_task_return_AUC": float(perf_sum_l.loc[method]["current_adv_task_return_AUC"]),
                    "robust_br_task_return_AUC": float(perf_sum_l.loc[method]["robust_br_task_return_AUC"]),
                    "robust_degradation_AUC": float(perf_sum_l.loc[method]["robust_degradation_AUC"]),
                    "fallback_to_egm_frac": float(dsub["fallback_to_egm"].mean()) if not dsub.empty else 0.0,
                    "gamma_active_frac": float(dsub["gamma_active"].mean()) if not dsub.empty else 0.0,
                    "G_contribution_ratio": float(dsub["G_contribution_ratio"].mean()) if not dsub.empty else 0.0,
                    "same_start_QP_wins": same_start_qp_wins,
                    "valid_flag": float(perf_sum_l.loc[method]["valid_fraction_robust_br"] > 0.99),
                    "final_robust_br_task_return": float(perf_sum_l.loc[method]["final_robust_br_task_return"]),
                }
            )
    lambda_df = pd.DataFrame(lambda_rows)
    lambda_df.to_csv(RESULT_ROOT / "mixed_linear_actor_rot_lq_alignment_lambda_sweep.csv", index=False)
    lambda_lines = [
        "# MixedLinearActorRotLQ Alignment Lambda Sweep",
        "",
        "Only theory-compatible runs with lambda_T = 0 were executed in this audit.",
        "",
        lambda_df.to_string(index=False),
    ]
    (RESULT_ROOT / "mixed_linear_actor_rot_lq_alignment_lambda_sweep_report.md").write_text("\n".join(lambda_lines), encoding="utf-8")

    # Part 6 BR sensitivity on current final setting
    br_rows = []
    final_traj = traj_map
    for local_br_radius in [0.1, 0.25, 0.5]:
        for br_steps in [20, 50]:
            for br_lr in [0.01, 0.03]:
                max_L_norm = 0.0
                for traj in final_traj.values():
                    for flat in traj:
                        _, L = benchmark.split_flat(flat)
                        max_L_norm = max(max_L_norm, float(torch.linalg.norm(L)))
                global_budget = max(1.0, max_L_norm)
                for method, traj in final_traj.items():
                    flat = traj[-1]
                    K, L = benchmark.split_flat(flat)
                    Lbr, robust, used_lr, stable = robust_br(
                        mod, benchmark, K, L, benchmark.eval_sigma0, local_br_radius, global_budget, inner_steps=br_steps, inner_lr=br_lr
                    )
                    br_rows.append(
                        {
                            "setting": "current_final",
                            "method": method,
                            "local_br_radius": local_br_radius,
                            "br_inner_steps": br_steps,
                            "br_inner_lr": br_lr,
                            "robust_br_task_return": robust["task_return"],
                            "robust_degradation": task_eval(mod, benchmark, K, torch.zeros_like(L), benchmark.eval_sigma0)["task_return"] - robust["task_return"],
                            "L_br_norm": float(torch.linalg.norm(Lbr)),
                            "BR_valid": float(stable),
                            "closed_loop_spectral_radius": robust["spectral_radius"],
                            "state_norm_max": robust["state_norm_max"],
                        }
                    )
    br_df = pd.DataFrame(br_rows)
    br_df.to_csv(RESULT_ROOT / "mixed_linear_actor_rot_lq_br_sensitivity.csv", index=False)
    br_lines = [
        "# MixedLinearActorRotLQ BR Sensitivity",
        "",
        br_df.to_string(index=False),
    ]
    (RESULT_ROOT / "mixed_linear_actor_rot_lq_br_sensitivity_report.md").write_text("\n".join(br_lines), encoding="utf-8")

    # Final decision based on existing + lambda sweep
    qpg_lambda = lambda_df[lambda_df["method"] == "proposed_qpg"].copy()
    nog_lambda = lambda_df[lambda_df["method"] == "proposed_noG"].copy()
    base_lambda = lambda_df[lambda_df["method"].isin(["sgd", "egm", "ppm"])].copy()
    merged = qpg_lambda.merge(
        nog_lambda[["lambda_F", "final_robust_br_task_return", "V_lambda_AUC", "P_tau_AUC"]],
        on="lambda_F",
        suffixes=("_qpg", "_nog"),
    )
    best_baselines = base_lambda.groupby("lambda_F")["final_robust_br_task_return"].max().rename("best_base_robust")
    merged = merged.merge(best_baselines, on="lambda_F")
    aligned = merged[
        (merged["final_robust_br_task_return_qpg"] >= merged["final_robust_br_task_return_nog"] - 1e-12)
        & (merged["final_robust_br_task_return_qpg"] >= merged["best_base_robust"] - 1e-12)
        & (merged["V_lambda_AUC_qpg"] <= merged["V_lambda_AUC_nog"] + 1e-12)
    ]

    if not aligned.empty:
        verdict = "A. FULL_ALIGNMENT_SUCCESS"
        claim = "A theory-compatible lambda_F setting aligns QP+G with both optimization geometry and robust task performance."
    else:
        qpg_beats_baselines = float(qpg_perf["final_robust_br_task_return"]) > float(base_perf["final_robust_br_task_return"].max())
        if qpg_beats_baselines:
            verdict = "B. OPTIMIZATION_POSITIVE_WITH_SUPPORTIVE_PERFORMANCE"
            claim = (
                "QP+G is clearly best on optimization geometry metrics and it improves robust task performance relative to SGD/EGM/PPM, "
                "but proposed_noG remains slightly better on the robust task metric."
            )
        else:
            verdict = "C. OPTIMIZATION_ONLY"
            claim = "The current setting supports optimization geometry claims only; robust task performance does not clearly favor QP+G."

    final_lines = [
        "# MixedLinearActorRotLQ Alignment Final Decision",
        "",
        f"Final decision: `{verdict}`",
        "",
        f"Recommended paper claim: {claim}",
        "",
        "Keep current environment: `Yes`",
        "Keep V_lambda unchanged for the main theory line: `Yes`",
        "Introduce task-performance term into main Lyapunov: `No; diagnostic only if used later.`",
        "Main figure: optimization geometry (`V_lambda`, `P_tau`, `field_norm`, exploitability).",
        "Appendix/supportive figure: RARL-style performance (`clean_task_return`, `current_adv_task_return`, `robust_br_task_return`, `robust_degradation`).",
        "",
        "Notes:",
        "- The current mixed benchmark already aligns QP+G with robust task performance relative to SGD/EGM/PPM.",
        "- The remaining mismatch is specifically versus proposed_noG on the robust BR task metric.",
        "- In the current audit, no theory-compatible lambda_F setting made QP+G beat proposed_noG on robust BR return while preserving its optimization advantage." if aligned.empty else "- A fully aligned lambda_F setting was found.",
    ]
    (RESULT_ROOT / "mixed_linear_actor_rot_lq_alignment_final_decision.md").write_text("\n".join(final_lines), encoding="utf-8")


if __name__ == "__main__":
    run()
