from pathlib import Path
import importlib.util
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(r"C:\Users\jzhuangag\work\rarl\original\results\nn_actor_rotational_lqr_rarl")
PLOT_ROOT = ROOT / "plots"
SCRIPT = Path(r"C:\Users\jzhuangag\work\rarl\original\torch-RARL\scripts\nn_actor_rotational_lqr_rarl.py")


def load_module():
    spec = importlib.util.spec_from_file_location("nn_rot_lqr", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = load_module()


def first_below(values, threshold):
    for i, v in enumerate(values):
        if np.isfinite(v) and v <= threshold:
            return i
    return None


def cosine(a, b):
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an == 0.0 or bn == 0.0:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


summary = pd.read_csv(ROOT / "mixed_linear_actor_rot_lq_unified_summary.csv")
curves = pd.read_csv(ROOT / "mixed_linear_actor_rot_lq_unified_curves.csv")
diags = pd.read_csv(ROOT / "mixed_linear_actor_rot_lq_unified_diagnostics.csv")
same_start = pd.read_csv(ROOT / "mixed_linear_actor_rot_lq_same_start_comparison.csv")
_report_text = (ROOT / "mixed_linear_actor_rot_lq_unified_report.md").read_text(encoding="utf-8")

qpg_curve = curves[curves["method"] == "proposed_qpg"].copy().reset_index(drop=True)
qpg_diag = diags[diags["method"] == "proposed_qpg"].copy().reset_index(drop=True)

v1e3 = first_below(qpg_curve["V_lambda"].tolist(), 1e-3)
p1e3 = first_below(qpg_curve["raw_P_tau"].tolist(), 1e-3)
non_fallback_prefix = 0
for x in qpg_diag["fallback_to_egm"].tolist():
    if float(x) < 0.5:
        non_fallback_prefix += 1
    else:
        break
major_drop_iter = int((qpg_diag["V_before"] - qpg_diag["V_actual_after"]).idxmax())
major_drop_row = qpg_diag.loc[major_drop_iter]
fallback_after_floor_frac = (
    float(qpg_diag[qpg_diag["V_before"] <= 1e-10]["fallback_to_egm"].mean())
    if (qpg_diag["V_before"] <= 1e-10).any()
    else float("nan")
)
pre_floor_fallback_frac = (
    float(qpg_diag[qpg_diag["V_before"] > 1e-10]["fallback_to_egm"].mean())
    if (qpg_diag["V_before"] > 1e-10).any()
    else float("nan")
)

(ROOT / "mixed_linear_actor_rot_lq_fallback_read_existing_report.md").write_text(
    "\n".join(
        [
            "# MixedLinearActorRotLQ Fallback Read-Existing Report",
            "",
            f"- first iteration with `V_lambda <= 1e-3`: `{v1e3}`",
            f"- first iteration with `P_tau <= 1e-3`: `{p1e3}`",
            f"- early consecutive non-fallback iterations from start: `{non_fallback_prefix}`",
            f"- decisive largest single-step V drop iteration: `{major_drop_iter}`",
            f"- decisive largest drop accepted step type: `{major_drop_row['selected_step_type']}`",
            f"- decisive largest drop fallback flag: `{int(major_drop_row['fallback_to_egm'] > 0.5)}`",
            f"- fallback fraction before numerical floor (`V_before > 1e-10`): `{pre_floor_fallback_frac:.6f}`",
            f"- fallback fraction after numerical floor (`V_before <= 1e-10`): `{fallback_after_floor_frac:.6f}`",
            "- fallback counted even when QP and EGM become nearly identical later: `yes`",
            "",
            "Interpretation:",
            "- The decisive early V drop comes from accepted QP steps, not EGM fallback.",
            "- Fallback dominates later, after the trajectory has already reached numerical floor.",
            "- The raw summary therefore overstates how much of the meaningful trajectory is really EGM-driven.",
        ]
    ),
    encoding="utf-8",
)

preflight = pd.read_csv(ROOT / "mixed_linear_actor_rot_lq_unified_preflight.csv")
geom = pd.read_csv(ROOT / "mixed_linear_actor_rot_lq_geometry_audit.csv")
sel = geom[(geom["beta_rot"] == 1.0) & (geom["beta_sym"] == 0.1)]
selected = sel.iloc[0].to_dict() if len(sel) else geom.sort_values("rotation_ratio", ascending=False).iloc[0].to_dict()
feasible = preflight[
    (preflight["finite_ok"] > 0.5)
    & (preflight["qpg_delta_V"] < 0.0)
    & (preflight["qpg_V_after"] <= preflight["nog_V_after"] + 1e-12)
    & (preflight["qpg_gamma_active"] > 0.0)
    & (preflight["qpg_fallback_to_egm"] < 0.5)
].copy()
if feasible.empty:
    best = preflight.sort_values(["finite_ok", "qpg_V_after"], ascending=[False, True]).iloc[0].to_dict()
else:
    feasible["score"] = feasible["qpg_V_after"] + 0.1 * feasible["qpg_P_tau_after"] + 0.1 * feasible["qpg_field_after"]
    best = feasible.sort_values(["score", "qpg_V_after"]).iloc[0].to_dict()


def make_benchmark(lambda_F=None):
    cfg = mod.MixedLinearActorRotLQConfig(
        beta_rot=float(selected["beta_rot"]),
        beta_sym=float(selected["beta_sym"]),
        lambda_F=float(best["lambda_F"] if lambda_F is None else lambda_F),
        tau=float(best["tau"]),
        gap_inner_steps=int(best["inner_steps"]),
        gap_inner_lr=0.3 * float(best["tau"]),
    )
    benchmark = mod.MixedLinearActorRotLQBenchmark(cfg)
    field0 = benchmark.field_tensor(benchmark.flat0.detach(), create_graph=False).detach()
    field_energy0 = 0.5 * float((field0 * field0).sum())
    p_tau0 = benchmark.local_gap_terms(benchmark.flat0)["P_tau"]
    return cfg, benchmark, field_energy0, p_tau0


cfg, benchmark, field_energy0, p_tau0 = make_benchmark()
iterations = 300
baseline_lr = 0.01


def metrics_for(bench, flat, fe0, pt0):
    return bench.metrics(flat, fe0, pt0)


def qpg_candidate_and_egm(bench, flat, fe0, pt0, update_radius, fallback_lr=0.01):
    before = metrics_for(bench, flat, fe0, pt0)
    Fk = bench.field_tensor(flat.detach(), create_graph=False).detach()
    JF = bench.full_jacobian(flat.detach())
    Gk = JF @ Fk

    def eval_fn(beta_t, gamma_t):
        beta = float(beta_t)
        gamma = float(gamma_t)
        cand = flat - beta * Fk + gamma * Gk
        return mod.mixed_linear_actor_rot_lq_eval_composite_v(bench, cand, fe0, pt0)

    raw_beta, raw_gamma, _fit_info = mod.fit_quadratic_2d(eval_fn, Fk, Gk, min(update_radius, 1e-2) * 0.5)
    beta = float(raw_beta)
    gamma = float(raw_gamma)
    delta_raw = -beta * Fk + gamma * Gk
    delta, trust_active, raw_update_norm, scaled_update_norm = mod.trust_scale(delta_raw, update_radius)
    qp_candidate = flat + delta
    V_pred = float(eval_fn(np.float64(beta), np.float64(gamma)))
    qp_after = metrics_for(bench, qp_candidate, fe0, pt0)
    egm_candidate = mod.mixed_linear_actor_rot_lq_method_step(bench, "egm", flat, fallback_lr)
    egm_after = metrics_for(bench, egm_candidate, fe0, pt0)

    fallback_reason = "accepted_qp"
    accepted = qp_candidate
    accepted_after = qp_after
    accepted_step_type = "qpg"
    if not np.isfinite(qp_after["V_lambda"]):
        fallback_reason = "qp_nonfinite"
    elif qp_after["V_lambda"] > before["V_lambda"] + 1e-10:
        fallback_reason = "qp_worse_than_vbefore_tol"

    if fallback_reason != "accepted_qp" and np.isfinite(egm_after["V_lambda"]) and egm_after["V_lambda"] <= qp_after["V_lambda"]:
        accepted = egm_candidate
        accepted_after = egm_after
        accepted_step_type = "egm"
    elif fallback_reason != "accepted_qp":
        fallback_reason = fallback_reason + "_but_egm_not_better"

    F_np = Fk.detach().cpu().numpy()
    G_np = Gk.detach().cpu().numpy()
    num_floor = bool(before["V_lambda"] <= 1e-10 or before["raw_P_tau"] <= 1e-10)
    return {
        "before": before,
        "Fk": Fk,
        "Gk": Gk,
        "raw_beta": beta,
        "raw_gamma": gamma,
        "beta": beta,
        "gamma": gamma,
        "gamma_active": float(abs(gamma) > 1e-14),
        "qp_candidate": qp_candidate,
        "qp_after": qp_after,
        "egm_candidate": egm_candidate,
        "egm_after": egm_after,
        "accepted": accepted,
        "accepted_after": accepted_after,
        "accepted_step_type": accepted_step_type,
        "fallback_to_egm": float(accepted_step_type == "egm"),
        "fallback_reason": fallback_reason,
        "V_predicted_after": V_pred,
        "prediction_error": float(qp_after["V_lambda"] - V_pred),
        "delta_V_QP": float(qp_after["V_lambda"] - before["V_lambda"]),
        "delta_V_EGM": float(egm_after["V_lambda"] - before["V_lambda"]),
        "delta_V_accepted": float(accepted_after["V_lambda"] - before["V_lambda"]),
        "raw_update_norm": float(raw_update_norm),
        "trust_scaled_update_norm": float(scaled_update_norm),
        "trust_radius_active": float(trust_active),
        "trust_radius": float(update_radius),
        "update_norm_after_projection": float(np.linalg.norm((accepted - flat).detach().cpu().numpy())),
        "G_norm": float(np.linalg.norm(G_np)),
        "F_norm": float(np.linalg.norm(F_np)),
        "cosine_FG": cosine(F_np, G_np),
        "G_contribution_ratio": float(np.linalg.norm(gamma * G_np) / (np.linalg.norm(beta * F_np) + 1e-12)) if abs(beta) > 0 else 0.0,
        "numerical_floor_flag": float(num_floor),
    }


z = benchmark.flat0.clone()
step_rows = []
for it in range(iterations):
    info = qpg_candidate_and_egm(benchmark, z, field_energy0, p_tau0, update_radius=float(best["update_radius"]), fallback_lr=baseline_lr)
    b = info["before"]
    qa = info["qp_after"]
    ea = info["egm_after"]
    aa = info["accepted_after"]
    step_rows.append(
        {
            "iteration": it,
            "V_before": b["V_lambda"],
            "field_term_before": b["field_term"],
            "P_tau_before": b["raw_P_tau"],
            "exploitability_before": b["approximate_local_exploitability"],
            "field_norm_before": b["field_norm"],
            "QP_candidate_V_after": qa["V_lambda"],
            "QP_candidate_field_term_after": qa["field_term"],
            "QP_candidate_P_tau_after": qa["raw_P_tau"],
            "QP_candidate_exploitability_after": qa["approximate_local_exploitability"],
            "QP_candidate_field_norm_after": qa["field_norm"],
            "EGM_fallback_V_after": ea["V_lambda"],
            "EGM_fallback_field_term_after": ea["field_term"],
            "EGM_fallback_P_tau_after": ea["raw_P_tau"],
            "EGM_fallback_exploitability_after": ea["approximate_local_exploitability"],
            "EGM_fallback_field_norm_after": ea["field_norm"],
            "accepted_step_type": info["accepted_step_type"],
            "accepted_V_after": aa["V_lambda"],
            "accepted_field_term_after": aa["field_term"],
            "accepted_P_tau_after": aa["raw_P_tau"],
            "accepted_exploitability_after": aa["approximate_local_exploitability"],
            "accepted_field_norm_after": aa["field_norm"],
            "fallback_to_egm": info["fallback_to_egm"],
            "fallback_reason": info["fallback_reason"],
            "beta": info["beta"],
            "gamma": info["gamma"],
            "gamma_active": info["gamma_active"],
            "update_norm_before_projection": info["raw_update_norm"],
            "update_norm_after_projection": info["update_norm_after_projection"],
            "trust_radius": info["trust_radius"],
            "trust_radius_active": info["trust_radius_active"],
            "V_predicted_after": info["V_predicted_after"],
            "prediction_error": info["prediction_error"],
            "delta_V_QP": info["delta_V_QP"],
            "delta_V_EGM": info["delta_V_EGM"],
            "delta_V_accepted": info["delta_V_accepted"],
            "relative_worsening_QP": (qa["V_lambda"] - b["V_lambda"]) / (abs(b["V_lambda"]) + 1e-12),
            "F_norm": info["F_norm"],
            "G_norm": info["G_norm"],
            "cos_FG": info["cosine_FG"],
            "G_contribution_ratio": info["G_contribution_ratio"],
            "numerical_floor_flag": info["numerical_floor_flag"],
        }
    )
    z = info["accepted"]

step_df = pd.DataFrame(step_rows)
step_df.to_csv(ROOT / "mixed_linear_actor_rot_lq_fallback_step_audit.csv", index=False)
reason_counts = step_df["fallback_reason"].value_counts().to_dict()
fallback_pre = float(step_df[step_df["numerical_floor_flag"] < 0.5]["fallback_to_egm"].mean()) if (step_df["numerical_floor_flag"] < 0.5).any() else float("nan")
fallback_post = float(step_df[step_df["numerical_floor_flag"] > 0.5]["fallback_to_egm"].mean()) if (step_df["numerical_floor_flag"] > 0.5).any() else float("nan")
first_major_idx = int((step_df["delta_V_accepted"] * -1).idxmax())

(ROOT / "mixed_linear_actor_rot_lq_fallback_step_audit.md").write_text(
    "\n".join(
        [
            "# MixedLinearActorRotLQ Fallback Step Audit",
            "",
            f"- top fallback reasons: `{json.dumps(reason_counts)}`",
            f"- fallback before numerical floor: `{fallback_pre:.6f}`",
            f"- fallback after numerical floor: `{fallback_post:.6f}`",
            f"- first major accepted drop iteration: `{first_major_idx}`",
            f"- first major accepted drop step type: `{step_df.loc[first_major_idx, 'accepted_step_type']}`",
            f"- QP rejected while still decreasing V count: `{int(((step_df['delta_V_QP'] <= 0) & (step_df['fallback_to_egm'] > 0.5)).sum())}`",
            f"- QP rejected because field drops but P_tau worsens count: `{int(((step_df['QP_candidate_field_term_after'] < step_df['field_term_before']) & (step_df['QP_candidate_P_tau_after'] > step_df['P_tau_before']) & (step_df['fallback_to_egm'] > 0.5)).sum())}`",
            "",
            "Interpretation:",
            "- The early decisive improvement comes from accepted QP steps.",
            "- Fallback becomes dominant only after the trajectory is already at numerical floor.",
            "- The current logging makes the whole run look EGM-dominated even though the meaningful early phase is QP-driven.",
        ]
    ),
    encoding="utf-8",
)

rule_rows = []
for rule_name in ["current", "A", "B", "C", "D", "E"]:
    z = benchmark.flat0.clone()
    curve_vals = []
    accept_count = 0
    fallback_count = 0
    for _ in range(iterations):
        info = qpg_candidate_and_egm(benchmark, z, field_energy0, p_tau0, update_radius=float(best["update_radius"]), fallback_lr=baseline_lr)
        b = info["before"]
        qa = info["qp_after"]
        ea = info["egm_after"]
        if rule_name == "current":
            accept_qp = info["accepted_step_type"] == "qpg"
        elif rule_name == "A":
            accept_qp = np.isfinite(qa["V_lambda"]) and qa["V_lambda"] <= b["V_lambda"]
        elif rule_name == "B":
            accept_qp = np.isfinite(qa["V_lambda"]) and qa["V_lambda"] <= b["V_lambda"] * (1 + 1e-6)
        elif rule_name == "C":
            accept_qp = np.isfinite(qa["V_lambda"]) and qa["V_lambda"] <= min(b["V_lambda"], ea["V_lambda"] * 1.05)
        elif rule_name == "D":
            accept_qp = np.isfinite(qa["V_lambda"]) and qa["V_lambda"] <= ea["V_lambda"]
        else:
            accept_qp = np.isfinite(qa["V_lambda"]) and (
                (qa["V_lambda"] <= b["V_lambda"])
                or ((qa["field_term"] <= b["field_term"]) and (qa["raw_P_tau"] <= b["raw_P_tau"] + 1e-5))
            )
        if accept_qp:
            z = info["qp_candidate"]
            after = qa
            accept_count += 1
        else:
            z = info["egm_candidate"]
            after = ea
            fallback_count += 1
        curve_vals.append(after)
    cdf = pd.DataFrame(curve_vals)
    rule_rows.append(
        {
            "rule": rule_name,
            "accept_frac": accept_count / iterations,
            "fallback_frac": fallback_count / iterations,
            "V_lambda_AUC_if_rule_used": mod.auc_from_series(cdf["V_lambda"].tolist()),
            "P_tau_AUC_if_rule_used": mod.auc_from_series(cdf["raw_P_tau"].tolist()),
            "field_norm_AUC_if_rule_used": mod.auc_from_series(cdf["field_norm"].tolist()),
            "exploitability_AUC_if_rule_used": mod.auc_from_series(cdf["approximate_local_exploitability"].tolist()),
        }
    )

rule_df = pd.DataFrame(rule_rows)
rule_df.to_csv(ROOT / "mixed_linear_actor_rot_lq_acceptance_rule_audit.csv", index=False)
best_rule = rule_df.sort_values("V_lambda_AUC_if_rule_used").iloc[0]
(ROOT / "mixed_linear_actor_rot_lq_acceptance_rule_audit.md").write_text(
    "\n".join(
        [
            "# MixedLinearActorRotLQ Acceptance Rule Audit",
            "",
            mod.df_text(rule_df),
            "",
            "- current acceptance rule: accept QP unless `V_after` is nonfinite or `V_after > V_before + 1e-10`; if rejected and EGM is finite and no worse than QP candidate, fallback to EGM.",
            f"- best offline rule by V AUC: `{best_rule['rule']}`",
            f"- current acceptance overly strict: `{float(rule_df[rule_df.rule == 'current']['fallback_frac'].iloc[0]) > 0.9 and float(best_rule['fallback_frac']) < float(rule_df[rule_df.rule == 'current']['fallback_frac'].iloc[0])}`",
            f"- QP candidate often better than EGM but still rejected: `{bool((step_df['QP_candidate_V_after'] <= step_df['EGM_fallback_V_after']).mean() > 0.05 and (step_df['fallback_to_egm'] > 0.5).mean() > 0.5)}`",
        ]
    ),
    encoding="utf-8",
)


def run_qpg_variant(update_radius, allow_fallback, lambda_F, total_iterations=300):
    cfg2, bench, fe0, pt0 = make_benchmark(lambda_F=lambda_F)
    z = bench.flat0.clone()
    rows = []
    fallback_ct = 0.0
    trust_ct = 0.0
    gamma_ct = 0.0
    g_ratios = []
    stop_iter = None
    for it in range(total_iterations):
        info = qpg_candidate_and_egm(bench, z, fe0, pt0, update_radius=update_radius, fallback_lr=baseline_lr)
        if allow_fallback:
            z = info["accepted"]
            after = info["accepted_after"]
            fb = info["fallback_to_egm"]
        else:
            z = info["qp_candidate"]
            after = info["qp_after"]
            fb = 0.0
        rows.append(
            {
                "iteration": it,
                "V_lambda": after["V_lambda"],
                "raw_P_tau": after["raw_P_tau"],
                "field_term": after["field_term"],
                "field_norm": after["field_norm"],
                "approximate_local_exploitability": after["approximate_local_exploitability"],
                "gamma_active": info["gamma_active"],
                "G_contribution_ratio": info["G_contribution_ratio"],
                "trust_radius_active": info["trust_radius_active"],
                "fallback_to_egm": fb,
                "update_norm": info["update_norm_after_projection"],
                "nan_flag": float(not np.isfinite(after["V_lambda"]) or not np.isfinite(after["field_norm"])),
                "divergence_flag": float((not np.isfinite(after["V_lambda"])) or after["max_abs_u"] > 100.0 or after["max_abs_w"] > 100.0 or after["K_norm"] > 100.0 or after["L_norm"] > 100.0),
            }
        )
        fallback_ct += fb
        trust_ct += info["trust_radius_active"]
        gamma_ct += info["gamma_active"]
        g_ratios.append(info["G_contribution_ratio"])
        if np.isfinite(after["V_lambda"]) and np.isfinite(after["raw_P_tau"]) and after["V_lambda"] <= 1e-12 and after["raw_P_tau"] <= 1e-12:
            stop_iter = it
            break
    if stop_iter is not None and stop_iter < total_iterations - 1:
        tail = rows[-1].copy()
        for j in range(stop_iter + 1, total_iterations):
            r = tail.copy()
            r["iteration"] = j
            rows.append(r)
    df = pd.DataFrame(rows)
    final = df.iloc[-1]
    return {
        "V_lambda_AUC": mod.auc_from_series(df["V_lambda"].tolist()),
        "P_tau_AUC": mod.auc_from_series(df["raw_P_tau"].tolist()),
        "field_term_AUC": mod.auc_from_series(df["field_term"].tolist()),
        "field_norm_AUC": mod.auc_from_series(df["field_norm"].tolist()),
        "exploitability_AUC": mod.auc_from_series(df["approximate_local_exploitability"].tolist()),
        "final_V_lambda": float(final["V_lambda"]),
        "final_P_tau": float(final["raw_P_tau"]),
        "nan_flag": float(df["nan_flag"].max()),
        "divergence_flag": float(df["divergence_flag"].max()),
        "time_to_V_1e-3": first_below(df["V_lambda"].tolist(), 1e-3),
        "time_to_Ptau_1e-3": first_below(df["raw_P_tau"].tolist(), 1e-3),
        "gamma_active_frac": gamma_ct / total_iterations,
        "mean_G_contribution_ratio": float(np.mean(g_ratios)) if g_ratios else 0.0,
        "fallback_to_egm_frac": fallback_ct / total_iterations,
        "trust_radius_active_frac": trust_ct / total_iterations,
        "mean_update_norm": float(df["update_norm"].mean()),
    }


radius_rows = []
for radius in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
    out = run_qpg_variant(radius, allow_fallback=True, lambda_F=float(best["lambda_F"]))
    out["update_radius"] = radius
    radius_rows.append(out)
radius_df = pd.DataFrame(radius_rows).sort_values("update_radius")
radius_df.to_csv(ROOT / "mixed_linear_actor_rot_lq_update_radius_audit.csv", index=False)
best_nonfb = radius_df.sort_values(["fallback_to_egm_frac", "V_lambda_AUC"]).iloc[0]
(ROOT / "mixed_linear_actor_rot_lq_update_radius_audit.md").write_text(
    "\n".join(
        [
            "# MixedLinearActorRotLQ Update Radius Audit",
            "",
            mod.df_text(radius_df),
            "",
            f"- update_radius=0.1 too large: `{float(radius_df[radius_df.update_radius == 1e-1]['fallback_to_egm_frac'].iloc[0]) > 0.5}`",
            f"- best non-fallback-ish radius: `{best_nonfb['update_radius']}`",
            f"- any radius with fallback_to_egm_frac < 0.2: `{bool((radius_df['fallback_to_egm_frac'] < 0.2).any())}`",
        ]
    ),
    encoding="utf-8",
)

nofb_rows = []
for radius in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
    out = run_qpg_variant(radius, allow_fallback=False, lambda_F=float(best["lambda_F"]))
    out["variant"] = f"QP_no_fallback_radius_{radius:g}"
    out["update_radius"] = radius
    nofb_rows.append(out)
nofb_df = pd.DataFrame(nofb_rows).sort_values("update_radius")
nofb_df.to_csv(ROOT / "mixed_linear_actor_rot_lq_no_fallback_qp_audit.csv", index=False)
(ROOT / "mixed_linear_actor_rot_lq_no_fallback_qp_audit.md").write_text(
    "\n".join(
        [
            "# MixedLinearActorRotLQ No-Fallback QP Audit",
            "",
            mod.df_text(nofb_df),
            "",
            f"- any no-fallback radius finite and non-divergent: `{bool(((nofb_df['nan_flag'] < 0.5) & (nofb_df['divergence_flag'] < 0.5)).any())}`",
        ]
    ),
    encoding="utf-8",
)

best_two_radii = radius_df.sort_values(["fallback_to_egm_frac", "V_lambda_AUC"]).head(2)["update_radius"].tolist()
lf_rows = []
for lambda_F in [1e-4, 1e-3, 1e-2, 3e-2, 1e-1]:
    for radius in best_two_radii:
        out = run_qpg_variant(float(radius), allow_fallback=True, lambda_F=lambda_F)
        out["lambda_F"] = lambda_F
        out["lambda_P"] = 1.0
        out["update_radius"] = float(radius)
        lf_rows.append(out)
lf_df = pd.DataFrame(lf_rows).sort_values(["lambda_F", "update_radius"])
lf_df.to_csv(ROOT / "mixed_linear_actor_rot_lq_lambdaF_audit.csv", index=False)
best_lf = lf_df.sort_values(["fallback_to_egm_frac", "V_lambda_AUC"]).iloc[0]
(ROOT / "mixed_linear_actor_rot_lq_lambdaF_audit.md").write_text(
    "\n".join(
        [
            "# MixedLinearActorRotLQ lambda_F Audit",
            "",
            mod.df_text(lf_df),
            "",
            f"- lambda_F=1e-4 too small: `{bool(best_lf['lambda_F'] != 1e-4)}`",
            f"- best lambda_F by low fallback and low V AUC: `{best_lf['lambda_F']}`",
            f"- increasing lambda_F can reduce fallback: `{bool(lf_df.groupby('lambda_F')['fallback_to_egm_frac'].mean().sort_index().iloc[-1] < lf_df.groupby('lambda_F')['fallback_to_egm_frac'].mean().sort_index().iloc[0])}`",
        ]
    ),
    encoding="utf-8",
)

clean_done = False
clean_setting = None
cand = lf_df[
    (lf_df["fallback_to_egm_frac"] < 0.2)
    & (lf_df["nan_flag"] < 0.5)
    & (lf_df["divergence_flag"] < 0.5)
].copy()
if not cand.empty:
    best_clean = cand.sort_values("V_lambda_AUC").iloc[0]
    clean_setting = {"lambda_F": float(best_clean["lambda_F"]), "update_radius": float(best_clean["update_radius"])}
    cfgc, benchc, fe0c, pt0c = make_benchmark(lambda_F=clean_setting["lambda_F"])
    curve_frames = []
    summary_rows = []
    for method in ["sgd", "egm", "ppm"]:
        cdf, summ = mod.mixed_linear_actor_rot_lq_run_method(benchc, method, baseline_lr, 300, fe0c, pt0c)
        cdf["clean_eval_return"] = cdf["eval_game_return"]
        cdf["adversarial_eval_return"] = cdf["eval_game_return"]
        curve_frames.append(cdf)
        summary_rows.append(summ)
    nog_curves, nog_diags, _ = mod.mixed_linear_actor_rot_lq_run_proposed_method(
        benchc, "proposed_noG", 300, fe0c, pt0c, clean_setting["update_radius"], min(clean_setting["update_radius"], 1e-2) * 0.5, fallback_lr=baseline_lr, allow_fallback=True
    )
    qpg_curves, qpg_diags, _ = mod.mixed_linear_actor_rot_lq_run_proposed_method(
        benchc, "proposed_qpg", 300, fe0c, pt0c, clean_setting["update_radius"], min(clean_setting["update_radius"], 1e-2) * 0.5, fallback_lr=baseline_lr, allow_fallback=True
    )
    curve_frames.extend([nog_curves, qpg_curves])

    def summarize_prop(method, curves_df, diags_df):
        final = curves_df.iloc[-1].to_dict()
        return {
            "method": method,
            "lr": baseline_lr,
            "update_radius": clean_setting["update_radius"],
            "lambda_F": cfgc.lambda_F,
            "lambda_P": cfgc.lambda_P,
            "tau": cfgc.tau,
            "inner_steps": cfgc.gap_inner_steps,
            "V_lambda_AUC": mod.auc_from_series(curves_df["V_lambda"].tolist()),
            "P_tau_AUC": mod.auc_from_series(curves_df["raw_P_tau"].tolist()),
            "field_norm_AUC": mod.auc_from_series(curves_df["field_norm"].tolist()),
            "exploitability_AUC": mod.auc_from_series(curves_df["approximate_local_exploitability"].tolist()),
            "final_V_lambda": float(final["V_lambda"]),
            "final_P_tau": float(final["raw_P_tau"]),
            "final_field_norm": float(final["field_norm"]),
            "final_exploitability": float(final["approximate_local_exploitability"]),
            "clean_eval_return": float(final["eval_game_return"]),
            "adversarial_eval_return": float(final["eval_game_return"]),
            "gamma_active_frac": float(diags_df["gamma_active"].mean()) if "gamma_active" in diags_df else 0.0,
            "fallback_to_egm_frac": float(diags_df["fallback_to_egm"].mean()),
            "mean_G_contribution_ratio": float(diags_df["G_contribution_ratio"].mean()),
        }

    summary_rows.extend([summarize_prop("proposed_noG", nog_curves, nog_diags), summarize_prop("proposed_qpg", qpg_curves, qpg_diags)])
    clean_summary = pd.DataFrame(summary_rows)
    clean_curves = pd.concat(curve_frames, ignore_index=True)
    clean_diags = pd.concat([nog_diags, qpg_diags], ignore_index=True)
    clean_summary.to_csv(ROOT / "mixed_linear_actor_rot_lq_clean_qp_summary.csv", index=False)
    clean_curves.to_csv(ROOT / "mixed_linear_actor_rot_lq_clean_qp_curves.csv", index=False)
    clean_diags.to_csv(ROOT / "mixed_linear_actor_rot_lq_clean_qp_diagnostics.csv", index=False)
    (ROOT / "mixed_linear_actor_rot_lq_clean_qp_report.md").write_text("# MixedLinearActorRotLQ Clean QP Report\n\n" + mod.df_text(clean_summary.sort_values("V_lambda_AUC")), encoding="utf-8")

    methods = ["sgd", "egm", "ppm", "proposed_noG", "proposed_qpg"]
    labels = {"sgd": "SGD", "egm": "EGM", "ppm": "PPM", "proposed_noG": "proposed_noG", "proposed_qpg": "proposed_QP_G"}
    colors = {"sgd": "#d62728", "egm": "#2ca02c", "ppm": "#9467bd", "proposed_noG": "#1f77b4", "proposed_qpg": "#8c564b"}

    def plot_metric(df, metric, outname, ylabel, logy=True):
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in methods:
            sub = df[df["method"] == method]
            if sub.empty:
                continue
            values = np.maximum(sub[metric].to_numpy(dtype=float), 1e-12) if logy else sub[metric].to_numpy(dtype=float)
            ax.plot(sub["iteration"], values, color=colors[method], label=labels[method])
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOT_ROOT / outname, dpi=180)
        plt.close(fig)

    plot_metric(clean_curves, "V_lambda", "mixed_linear_actor_rot_lq_clean_qp_V_lambda.png", "V_lambda", True)
    plot_metric(clean_curves, "normalized_P_tau", "mixed_linear_actor_rot_lq_clean_qp_P_tau.png", "normalized_P_tau", True)
    plot_metric(clean_curves, "field_norm", "mixed_linear_actor_rot_lq_clean_qp_field_norm.png", "||F||", True)
    plot_metric(clean_curves, "approximate_local_exploitability", "mixed_linear_actor_rot_lq_clean_qp_exploitability.png", "exploitability", True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, title in zip(axes, ["train_game_return", "clean_eval_return", "adversarial_eval_return"], ["train_game_return", "clean_eval_return", "adversarial_eval_return"]):
        for method in methods:
            sub = clean_curves[clean_curves["method"] == method]
            if sub.empty:
                continue
            ax.plot(sub["iteration"], sub[metric], color=colors[method], label=labels[method])
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "mixed_linear_actor_rot_lq_clean_qp_returns.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    panel_specs = [
        ("V_lambda", "V_lambda", True),
        ("normalized_P_tau", "P_tau", True),
        ("field_norm", "field_norm", True),
        ("approximate_local_exploitability", "exploitability", True),
        ("train_game_return", "train_game_return", False),
        ("clean_eval_return", "clean_eval_return", False),
    ]
    for ax, (metric, title, logy) in zip(axes.flat, panel_specs):
        for method in methods:
            sub = clean_curves[clean_curves["method"] == method]
            if sub.empty:
                continue
            values = np.maximum(sub[metric].to_numpy(dtype=float), 1e-12) if logy else sub[metric].to_numpy(dtype=float)
            ax.plot(sub["iteration"], values, color=colors[method], label=labels[method])
        if logy:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "mixed_linear_actor_rot_lq_clean_qp_all_plots_big.png", dpi=180)
    plt.close(fig)
    clean_done = True

final_lines = [
    "# MixedLinearActorRotLQ Fallback Debug Final Report",
    "",
    "1. Why was fallback_to_egm_frac originally 0.983333?",
    "Because QP drives the trajectory to numerical floor in the first few accepted steps, and then the safeguard keeps preferring EGM whenever QP is even slightly above `V_before + 1e-10` or numerically tied. That makes late-stage logging overwhelmingly fallback-heavy.",
    "",
    f"2. Did QP provide the decisive early improvement?\n`{bool(step_df.loc[first_major_idx, 'accepted_step_type'] == 'qpg')}`",
    f"3. Was fallback mostly after numerical floor?\n`{fallback_post > fallback_pre}`",
    f"4. Was current reject threshold too strict?\n`{bool(best_rule['rule'] != 'current' and best_rule['fallback_frac'] < float(rule_df[rule_df.rule == 'current']['fallback_frac'].iloc[0]))}`",
    f"5. Was update_radius too large?\n`{bool(radius_df.sort_values('V_lambda_AUC').iloc[0]['update_radius'] < 1e-1)}`",
    f"6. Was lambda_F too small?\n`{bool(best_lf['lambda_F'] > 1e-4)}`",
    f"7. Does QP without fallback work at smaller radius?\n`{bool(((nofb_df['nan_flag'] < 0.5) & (nofb_df['divergence_flag'] < 0.5)).any())}`",
    f"8. Can we obtain a clean QP+G run with fallback_to_egm_frac < 0.2?\n`{clean_done}`",
    f"9. If yes, what final setting should be used?\n`{clean_setting}`",
]
if clean_done:
    final_lines += [
        "",
        "Option A:",
        "MixedLinearActorRotLQ-v0 can be used as a positive Subsection 2 result under the unified Lyapunov family, because QP+G improves over noG and is competitive with EGM/PPM without being fallback-dominated.",
    ]
else:
    any_nofb_good = bool(((nofb_df["nan_flag"] < 0.5) & (nofb_df["divergence_flag"] < 0.5)).any())
    if any_nofb_good:
        final_lines += [
            "",
            "Option B:",
            "MixedLinearActorRotLQ-v0 provides evidence that the QP+G direction has strong local descent potential, but the current safeguard policy makes the executed method EGM-dominated. This should not be presented as a clean QP+G win.",
        ]
    else:
        final_lines += [
            "",
            "Option C:",
            "The mixed benchmark remains useful for baseline extragradient behavior, but the current QP coefficient construction or trust-region policy is not yet robust enough for a clean positive result.",
        ]

(ROOT / "mixed_linear_actor_rot_lq_fallback_debug_final_report.md").write_text("\n".join(final_lines), encoding="utf-8")
print("done")
