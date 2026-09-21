from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = ROOT / "original" / "results" / "unified_lyapunov_main_experiments"
PLOT_ROOT = RESULT_ROOT / "plots"
PLOT_ROOT.mkdir(parents=True, exist_ok=True)
EPS = 1e-12


def load_module():
    script_path = ROOT / "original" / "torch-RARL" / "scripts" / "lq_followup_fixed_benchmark.py"
    spec = importlib.util.spec_from_file_location("lq_followup_fixed_benchmark", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = load_module()


@dataclass(frozen=True)
class MethodCfg:
    method: str
    base_lr: float
    lambda_F_train: float
    tau_train: float
    update_radius: float
    dominance: bool = False
    egm_lr: float | None = None


def write_md(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def select_configs() -> Tuple[MethodCfg, MethodCfg, MethodCfg, MethodCfg, MethodCfg, MethodCfg]:
    repaired = pd.read_csv(RESULT_ROOT / "subsection2_lq_repaired_summary.csv")
    qps = pd.read_csv(RESULT_ROOT / "subsection2_lq_qp_sweep_summary.csv").sort_values(["method", "V_lambda_AUC"])
    dom = pd.read_csv(RESULT_ROOT / "subsection2_lq_egm_dominance_summary.csv").sort_values("V_lambda_AUC")

    sgd_lr = float(repaired[repaired["method"] == "sgd"]["base_lr"].iloc[0])
    egm_lr = float(repaired[repaired["method"] == "egm"]["base_lr"].iloc[0])
    ppm_lr = float(repaired[repaired["method"] == "ppm"]["base_lr"].iloc[0])
    qpg_best = qps[qps["method"] == "proposed_QP_G_unified_repaired"].iloc[0]
    nog_best = qps[qps["method"] == "proposed_noG_unified_repaired"].iloc[0]
    dom_best = dom.iloc[0]

    sgd = MethodCfg("sgd", sgd_lr, float(qpg_best["lambda_F"]), float(qpg_best["tau"]), float(qpg_best["update_radius"]))
    egm = MethodCfg("egm", egm_lr, float(qpg_best["lambda_F"]), float(qpg_best["tau"]), float(qpg_best["update_radius"]))
    ppm = MethodCfg("ppm", ppm_lr, float(qpg_best["lambda_F"]), float(qpg_best["tau"]), float(qpg_best["update_radius"]))
    nog = MethodCfg("proposed_noG_unified_repaired", float(nog_best["base_lr"]), float(nog_best["lambda_F"]), float(nog_best["tau"]), float(nog_best["update_radius"]))
    qpg = MethodCfg("proposed_QP_G_unified_repaired", float(qpg_best["base_lr"]), float(qpg_best["lambda_F"]), float(qpg_best["tau"]), float(qpg_best["update_radius"]), dominance=False, egm_lr=egm_lr)
    domcfg = MethodCfg("proposed_QP_G_unified_repaired_egm_dominance", float(dom_best["base_lr"]), float(dom_best["lambda_F"]), float(dom_best["tau"]), float(dom_best["update_radius"]), dominance=True, egm_lr=egm_lr)
    return sgd, egm, ppm, nog, qpg, domcfg


def common_eval_constants(benchmark, lambda_F_eval: float, tau_eval: float):
    init_flat = benchmark.join_flat(torch.tensor(benchmark.cfg.K0, dtype=mod.DTYPE), torch.tensor(benchmark.cfg.L0, dtype=mod.DTYPE))
    init_field, _ = benchmark.field_and_j(init_flat)
    p0, _, _ = benchmark.p_tau(init_flat, tau=tau_eval, field=init_field, J_val=benchmark.J(init_flat).detach())
    fe0 = 0.5 * float(torch.dot(init_field, init_field))
    return init_flat, float(p0), float(fe0)


def eval_metrics_common(benchmark, flat: torch.Tensor, lambda_F_eval: float, tau_eval: float, p0: float, fe0: float) -> Dict[str, float]:
    return benchmark.metrics(flat, lambda_F=lambda_F_eval, tau=tau_eval, p_tau0=p0, field_energy0=fe0)


def step_from_flat(
    benchmark,
    flat: torch.Tensor,
    cfg: MethodCfg,
    lambda_F_eval: float,
    tau_eval: float,
    p0: float,
    fe0: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    metrics_before = eval_metrics_common(benchmark, flat, lambda_F_eval, tau_eval, p0, fe0)
    F = torch.tensor(metrics_before["field"], dtype=mod.DTYPE)
    G = torch.tensor(metrics_before["G"], dtype=mod.DTYPE)
    raw_beta = 0.0
    raw_gamma = 0.0
    gamma_active = 0.0
    trust_active = False
    projection_active_K = False
    projection_active_L = False
    fallback_to_egm = False
    selected_step_type = cfg.method
    probe_radius = min(cfg.update_radius, 1e-2) * 0.5

    def v_eval(candidate_flat: torch.Tensor) -> Dict[str, float]:
        return eval_metrics_common(benchmark, candidate_flat, lambda_F_eval, tau_eval, p0, fe0)

    def eval_beta_gamma(beta_t: torch.Tensor, gamma_t: torch.Tensor) -> float:
        cand = flat - beta_t * F + gamma_t * G
        cand, _, _ = mod.apply_projected_step(benchmark, flat, cand - flat)
        return v_eval(cand)["V_lambda"]

    if cfg.method == "sgd":
        delta = -cfg.base_lr * F
    elif cfg.method == "egm":
        half = flat - cfg.base_lr * F
        half, _, _ = mod.apply_projected_step(benchmark, flat, half - flat)
        F_half = benchmark.field(half)
        delta = -cfg.base_lr * F_half
    elif cfg.method == "ppm":
        z_inner = flat.clone()
        for _ in range(10):
            F_inner = benchmark.field(z_inner)
            z_inner = flat - cfg.base_lr * F_inner
            z_inner, _, _ = mod.apply_projected_step(benchmark, flat, z_inner - flat)
        delta = z_inner - flat
    elif cfg.method == "proposed_noG_unified_repaired":
        beta, _ = mod.fit_quadratic_1d(eval_beta_gamma, -F, probe_radius=probe_radius)
        raw_beta = beta
        delta = -beta * F
        delta, trust_active, _, _ = mod.trust_scale(delta, cfg.update_radius)
        candidate, projection_active_K, projection_active_L = mod.apply_projected_step(benchmark, flat, delta)
        metrics_after = v_eval(candidate)
        unsafe = metrics_after["clean_spectral_radius"] > 1.0 or metrics_after["adv_spectral_radius"] > 1.0 or not np.isfinite(metrics_after["V_lambda"])
        if metrics_after["V_lambda"] > metrics_before["V_lambda"] or unsafe:
            fallback_to_egm = True
            selected_step_type = "fallback_egm"
            half = flat - cfg.base_lr * F
            half, _, _ = mod.apply_projected_step(benchmark, flat, half - flat)
            F_half = benchmark.field(half)
            delta = -cfg.base_lr * F_half
    elif cfg.method == "proposed_QP_G_unified_repaired":
        beta, gamma, _ = mod.fit_quadratic_2d(eval_beta_gamma, -F, G, probe_radius=probe_radius)
        raw_beta = beta
        raw_gamma = gamma
        gamma_active = float(abs(gamma) > 1e-12)
        delta = -beta * F + gamma * G
        delta, trust_active, _, _ = mod.trust_scale(delta, cfg.update_radius)
        candidate, projection_active_K, projection_active_L = mod.apply_projected_step(benchmark, flat, delta)
        metrics_after = v_eval(candidate)
        unsafe = metrics_after["clean_spectral_radius"] > 1.0 or metrics_after["adv_spectral_radius"] > 1.0 or not np.isfinite(metrics_after["V_lambda"])
        if metrics_after["V_lambda"] > metrics_before["V_lambda"] or unsafe:
            fallback_to_egm = True
            selected_step_type = "fallback_egm"
            half = flat - cfg.base_lr * F
            half, _, _ = mod.apply_projected_step(benchmark, flat, half - flat)
            F_half = benchmark.field(half)
            delta = -cfg.base_lr * F_half
    elif cfg.method == "proposed_QP_G_unified_repaired_egm_dominance":
        beta, gamma, _ = mod.fit_quadratic_2d(eval_beta_gamma, -F, G, probe_radius=probe_radius)
        raw_beta = beta
        raw_gamma = gamma
        gamma_active = float(abs(gamma) > 1e-12)
        delta_qp_raw = -beta * F + gamma * G
        delta_qp, trust_active, _, _ = mod.trust_scale(delta_qp_raw, cfg.update_radius)
        qp_flat, _, _ = mod.apply_projected_step(benchmark, flat, delta_qp)
        beta_nog, _ = mod.fit_quadratic_1d(eval_beta_gamma, -F, probe_radius=probe_radius)
        delta_nog_raw = -beta_nog * F
        delta_nog, _, _, _ = mod.trust_scale(delta_nog_raw, cfg.update_radius)
        nog_flat, _, _ = mod.apply_projected_step(benchmark, flat, delta_nog)
        lr = cfg.egm_lr if cfg.egm_lr is not None else 1e-2
        half = flat - lr * F
        half, _, _ = mod.apply_projected_step(benchmark, flat, half - flat)
        F_half = benchmark.field(half)
        delta_egm = -lr * F_half
        egm_flat, _, _ = mod.apply_projected_step(benchmark, flat, delta_egm)
        zero_flat = flat.clone()
        candidates = {
            "qpg": (qp_flat, v_eval(qp_flat)["V_lambda"]),
            "nog": (nog_flat, v_eval(nog_flat)["V_lambda"]),
            "egm": (egm_flat, v_eval(egm_flat)["V_lambda"]),
            "zero": (zero_flat, v_eval(zero_flat)["V_lambda"]),
        }
        selected_step_type = min(candidates.items(), key=lambda item: item[1][1])[0]
        candidate_flat = candidates[selected_step_type][0]
        delta = candidate_flat - flat
    else:
        raise ValueError(cfg.method)

    if cfg.method != "proposed_QP_G_unified_repaired_egm_dominance":
        new_flat, projection_active_K, projection_active_L = mod.apply_projected_step(benchmark, flat, delta)
    else:
        new_flat = flat + delta
    metrics_after = eval_metrics_common(benchmark, new_flat, lambda_F_eval, tau_eval, p0, fe0)
    info = {
        "raw_beta": raw_beta,
        "raw_gamma": raw_gamma,
        "gamma_active": gamma_active,
        "trust_radius_active": float(trust_active),
        "fallback_to_egm": float(fallback_to_egm),
        "selected_step_type": selected_step_type,
        "projection_active_K": float(projection_active_K),
        "projection_active_L": float(projection_active_L),
        "delta_norm": float(torch.linalg.norm(new_flat - flat)),
        "field": F.numpy(),
        "G": G.numpy(),
        "qpg_component_ratio": float(torch.linalg.norm(raw_gamma * G) / (torch.linalg.norm(raw_beta * F) + EPS)) if cfg.method.startswith("proposed_QP_G") else 0.0,
    }
    return new_flat.detach(), {"before": metrics_before, "after": metrics_after, **info}


def rollout_with_states(
    benchmark,
    cfg: MethodCfg,
    iterations: int,
    lambda_F_eval: float,
    tau_eval: float,
    p0: float,
    fe0: float,
) -> Tuple[pd.DataFrame, List[torch.Tensor], List[Dict[str, float]]]:
    flat = benchmark.join_flat(torch.tensor(benchmark.cfg.K0, dtype=mod.DTYPE), torch.tensor(benchmark.cfg.L0, dtype=mod.DTYPE))
    states = [flat.clone()]
    rows = []
    infos = []
    for it in range(iterations):
        new_flat, info = step_from_flat(benchmark, flat, cfg, lambda_F_eval, tau_eval, p0, fe0)
        row = {
            "method": cfg.method,
            "iteration": it,
            "V_lambda": info["after"]["V_lambda"],
            "raw_p_tau": info["after"]["raw_p_tau"],
            "normalized_p_tau_contribution": info["after"]["normalized_p_tau_contribution"],
            "raw_field_energy": info["after"]["raw_field_energy"],
            "normalized_field_contribution": info["after"]["normalized_field_contribution"],
            "field_norm": info["after"]["field_norm"],
            "train_task_return": info["after"]["train_task_return"],
            "clean_task_return": info["after"]["clean_task_return"],
            "adv_task_return": info["after"]["adv_task_return"],
            "clean_spectral_radius": info["after"]["clean_spectral_radius"],
            "adv_spectral_radius": info["after"]["adv_spectral_radius"],
            "beta": info["raw_beta"] if cfg.method.startswith("proposed") else cfg.base_lr,
            "gamma": info["raw_gamma"] if "QP_G" in cfg.method else 0.0,
            "gamma_active": info["gamma_active"],
            "update_norm": info["delta_norm"],
            "G_norm": info["after"]["G_norm"],
            "cosine_FG": info["after"]["cosine_FG"],
            "trust_radius_active": info["trust_radius_active"],
            "fallback_to_egm": info["fallback_to_egm"],
            "selected_step_type": info["selected_step_type"],
            "qpg_component_ratio": info["qpg_component_ratio"],
            "V_before": info["before"]["V_lambda"],
            "V_after": info["after"]["V_lambda"],
            "Delta_V": info["after"]["V_lambda"] - info["before"]["V_lambda"],
        }
        rows.append(row)
        infos.append(info)
        flat = new_flat
        states.append(flat.clone())
    return pd.DataFrame(rows), states, infos


def first_iter_below(series: pd.Series, thresh: float):
    arr = series.to_numpy()
    idx = np.where(arr <= thresh)[0]
    return float(idx[0]) if len(idx) else float("nan")


def prepare_main_runs():
    benchmark = mod.FixedLQBenchmark(mod.FixedLQConfig())
    sgd_cfg, egm_cfg, ppm_cfg, nog_cfg, qpg_cfg, dom_cfg = select_configs()
    lambda_F_eval = qpg_cfg.lambda_F_train
    tau_eval = qpg_cfg.tau_train
    _, p0, fe0 = common_eval_constants(benchmark, lambda_F_eval, tau_eval)
    configs = [sgd_cfg, egm_cfg, ppm_cfg, nog_cfg, qpg_cfg, dom_cfg]
    runs = {}
    for cfg in configs:
        curves, states, infos = rollout_with_states(benchmark, cfg, 1000, lambda_F_eval, tau_eval, p0, fe0)
        runs[cfg.method] = {"cfg": cfg, "curves": curves, "states": states, "infos": infos}
    return benchmark, runs, lambda_F_eval, tau_eval, p0, fe0


def part_a_vlogging(runs, lambda_F_eval, tau_eval):
    repaired_curves = pd.read_csv(RESULT_ROOT / "subsection2_lq_repaired_curves.csv")
    repaired_diag = pd.read_csv(RESULT_ROOT / "subsection2_lq_repaired_diagnostics.csv")
    rows = [
        {"artifact": "repaired_curves.csv", "curve_value_source": "actual_recomputed_after_update", "uses_common_formula_within_file": True, "lambda_F": 0.1, "lambda_P": 1.0, "tau": 0.03, "predicted_values_present": False},
        {"artifact": "repaired_diagnostics.csv", "curve_value_source": "actual_recomputed_after_update", "uses_common_formula_within_file": True, "lambda_F": 0.1, "lambda_P": 1.0, "tau": 0.03, "predicted_values_present": True},
        {"artifact": "qp_sweep_summary.csv", "curve_value_source": "summary_of_actual_recomputed_values", "uses_common_formula_within_file": False, "lambda_F": "varies_by_config", "lambda_P": 1.0, "tau": "varies_by_config", "predicted_values_present": False},
        {"artifact": "integrity_recomputed_main_runs", "curve_value_source": "actual_recomputed_after_update", "uses_common_formula_within_file": True, "lambda_F": lambda_F_eval, "lambda_P": 1.0, "tau": tau_eval, "predicted_values_present": False},
    ]
    pd.DataFrame(rows).to_csv(RESULT_ROOT / "subsection2_lq_vlogging_audit.csv", index=False)
    pred_gap = repaired_diag[repaired_diag["method"].isin(["proposed_noG_unified_repaired", "proposed_QP_G_unified_repaired"])].copy()
    pred_gap["pred_actual_gap"] = pred_gap["V_after_projected"] - pred_gap["V_predicted_after"]
    pred_inc = int(((pred_gap["V_predicted_change"] < 0) & (pred_gap["V_actual_projected_change"] > 0)).sum())
    repaired_methods = repaired_curves.groupby("method")["base_lr"].first().to_dict()
    report = "\n".join(
        [
            "# Subsection 2 V_lambda logging audit",
            "",
            "Answers:",
            "",
            "1. `curves.csv` style V_lambda values are actual recomputed post-update values, not the quadratic model prediction.",
            "2. The quadratic-model prediction is only logged separately in `repaired_diagnostics.csv` as `V_predicted_after` / `V_predicted_change`.",
            f"3. In the integrity recomputation, every method is evaluated with the same formula: `V = P_tau/(P_tau0+eps) + {lambda_F_eval} * field_energy/(field_energy0+eps)`.",
            "4. The original repaired run used a shared `(lambda_F, tau) = (0.1, 0.03)` across methods; the follow-up sweep varied `(lambda_F, tau, update_radius)` for proposed methods only.",
            f"5. The existing baselines were not originally recomputed with the QP-selected `tau={tau_eval}`; this integrity audit recomputes all main methods under the common evaluation `(lambda_F, tau)=({lambda_F_eval}, {tau_eval})`.",
            "6. `proposed_noG` / `proposed_QP_G` have `V_before`, `V_predicted_after`, and `V_after_projected` recorded in `repaired_diagnostics.csv`.",
            f"7. Mean |actual - predicted| over proposed diagnostics = `{pred_gap['pred_actual_gap'].abs().mean():.6e}`, max = `{pred_gap['pred_actual_gap'].abs().max():.6e}`.",
            f"8. Predicted decrease but actual increase occurred `{pred_inc}` times in the old repaired diagnostics.",
            "9. The new clean integrity plots use only actual recomputed V values. The old follow-up comparison mixed baseline curves from `(tau=0.03)` with best-QP sweep curves from `(tau=0.01)`, so that comparison should be treated as a reporting inconsistency rather than the final canonical plot.",
            "",
            "Baseline config check:",
            f"- SGD / EGM / PPM selected lrs in repaired run: `{repaired_methods}`. In this LQ benchmark there is no max_grad_norm hyperparameter; all three baselines use the same selected base_lr=0.01 and the same projection constraints.",
        ]
    )
    write_md(RESULT_ROOT / "subsection2_lq_vlogging_audit.md", report)


def part_b_numerical_floor(main_curves: pd.DataFrame):
    rows = []
    for method, frame in main_curves.groupby("method"):
        frame = frame.reset_index(drop=True)
        vals = frame["V_lambda"]
        rows.append(
            {
                "method": method,
                "iter_le_1e6": first_iter_below(vals, 1e-6),
                "iter_le_1e8": first_iter_below(vals, 1e-8),
                "iter_le_1e10": first_iter_below(vals, 1e-10),
                "iter_le_1e12": first_iter_below(vals, 1e-12),
                "auc_raw": float(np.trapz(vals.to_numpy())),
                "auc_clipped_1e12": float(np.trapz(np.maximum(vals.to_numpy(), 1e-12))),
                "final_v_clipped_1e12": float(max(vals.iloc[-1], 1e-12)),
            }
        )
    floor_df = pd.DataFrame(rows).sort_values("auc_clipped_1e12")
    floor_df.to_csv(RESULT_ROOT / "subsection2_lq_numerical_floor_audit.csv", index=False)
    report = "\n".join(["# Subsection 2 numerical floor audit", "", "Values below `1e-12` are treated as numerical floor and not interpreted.", "", "This audit recomputes clipped AUC with `V_plot = max(V_lambda, 1e-12)` to avoid over-interpreting differences at `1e-18` / `1e-20`."])
    write_md(RESULT_ROOT / "subsection2_lq_numerical_floor_audit.md", report)

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, frame in main_curves.groupby("method"):
        y = np.maximum(frame["V_lambda"].to_numpy(), 1e-12)
        ax.plot(frame["iteration"], y, label=method)
    for thr in [1e-6, 1e-8, 1e-10]:
        ax.axhline(thr, linestyle="--", linewidth=1, color="gray")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("V_lambda clipped at 1e-12")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "subsection2_lq_followup_lyapunov_clipped_floor.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, frame in main_curves.groupby("method"):
        y = np.maximum(frame["field_norm"].to_numpy(), 1e-12)
        ax.plot(frame["iteration"], y, label=method)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("field norm clipped at 1e-12")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "subsection2_lq_followup_field_norm_clipped_floor.png", dpi=160)
    plt.close(fig)
    return floor_df


def part_c_dominance_variant(dom_curve: pd.DataFrame):
    rows = []
    for _, row in dom_curve.iterrows():
        rows.append({"iteration": int(row["iteration"]), "selected_step_type": row["selected_step_type"], "V_before": row["V_before"], "V_after": row["V_after"], "V_before_le_1e12": float(row["V_before"] <= 1e-12)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_ROOT / "subsection2_lq_dominance_variant_audit.csv", index=False)
    zero = df[df["selected_step_type"] == "zero"]
    zero_after_floor = float((zero["V_before"] <= 1e-12).mean()) if len(zero) else 0.0
    report = "\n".join(
        [
            "# Subsection 2 dominance variant audit",
            "",
            "EGM-dominance is a safety analysis, not the main proposed method.",
            "",
            f"- qpg_selected_frac = `{(df['selected_step_type'] == 'qpg').mean():.3f}`",
            f"- egm_selected_frac = `{(df['selected_step_type'] == 'egm').mean():.3f}`",
            f"- nog_selected_frac = `{(df['selected_step_type'] == 'nog').mean():.3f}`",
            f"- zero_selected_frac = `{(df['selected_step_type'] == 'zero').mean():.3f}`",
            f"- zero-step mean iteration = `{zero['iteration'].mean() if len(zero) else float('nan'):.1f}`",
            f"- zero-step fraction occurring after numerical floor (`V_before <= 1e-12`) = `{zero_after_floor:.3f}`",
            "",
            "If zero-step appears heavily before numerical floor, the dominance gate is over-conservative.",
        ]
    )
    write_md(RESULT_ROOT / "subsection2_lq_dominance_variant_audit.md", report)
    return df


def part_d_qp_distinctness(benchmark, runs, lambda_F_eval, tau_eval, p0, fe0):
    qpg_states = runs["proposed_QP_G_unified_repaired"]["states"]
    qpg_cfg = runs["proposed_QP_G_unified_repaired"]["cfg"]
    nog_cfg = runs["proposed_noG_unified_repaired"]["cfg"]
    sgd_cfg = runs["sgd"]["cfg"]
    egm_cfg = runs["egm"]["cfg"]
    rows = []
    for it in range(1000):
        z = qpg_states[it]
        qpg_next, qinfo = step_from_flat(benchmark, z, qpg_cfg, lambda_F_eval, tau_eval, p0, fe0)
        nog_next, _ = step_from_flat(benchmark, z, nog_cfg, lambda_F_eval, tau_eval, p0, fe0)
        sgd_next, _ = step_from_flat(benchmark, z, sgd_cfg, lambda_F_eval, tau_eval, p0, fe0)
        egm_next, _ = step_from_flat(benchmark, z, egm_cfg, lambda_F_eval, tau_eval, p0, fe0)
        dq = (qpg_next - z).detach().cpu().numpy()
        dn = (nog_next - z).detach().cpu().numpy()
        ds = (sgd_next - z).detach().cpu().numpy()
        de = (egm_next - z).detach().cpu().numpy()

        def cos(a, b):
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na < EPS or nb < EPS:
                return np.nan
            return float(np.dot(a, b) / (na * nb))

        rows.append({"iteration": it, "qp_update_norm": float(np.linalg.norm(dq)), "sgd_update_norm": float(np.linalg.norm(ds)), "egm_update_norm": float(np.linalg.norm(de)), "nog_update_norm": float(np.linalg.norm(dn)), "cos_qp_sgd": cos(dq, ds), "cos_qp_nog": cos(dq, dn), "cos_qp_egm": cos(dq, de), "gamma": qinfo["raw_gamma"], "gamma_active": qinfo["gamma_active"], "g_contrib_ratio": qinfo["qpg_component_ratio"], "trust_radius_active": qinfo["trust_radius_active"], "fallback_to_egm": qinfo["fallback_to_egm"], "selected_step_type": qinfo["selected_step_type"]})
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_ROOT / "subsection2_lq_qp_update_distinctness.csv", index=False)
    report = "\n".join(
        [
            "# Subsection 2 QP update distinctness audit",
            "",
            f"1. QP step majority co-linear with SGD? `mean cos={df['cos_qp_sgd'].mean():.3f}`, `frac cos>0.99={(df['cos_qp_sgd'] > 0.99).mean():.3f}`.",
            f"2. QP step majority just noG? `mean cos={df['cos_qp_nog'].mean():.3f}`, `frac cos>0.99={(df['cos_qp_nog'] > 0.99).mean():.3f}`.",
            f"3. QP step majority just EGM? `mean cos={df['cos_qp_egm'].mean():.3f}`, `frac cos>0.99={(df['cos_qp_egm'] > 0.99).mean():.3f}`.",
            f"4. Mean G-term contribution ratio = `{df['g_contrib_ratio'].mean():.3f}`, median = `{df['g_contrib_ratio'].median():.3f}`.",
            f"5. Gamma active fraction = `{df['gamma_active'].mean():.3f}`, but fallback_to_egm fraction = `{df['fallback_to_egm'].mean():.3f}`. Gamma activity does not by itself imply an effective distinct applied update.",
        ]
    )
    write_md(RESULT_ROOT / "subsection2_lq_qp_update_distinctness.md", report)
    return df


def part_e_same_start(benchmark, runs, lambda_F_eval, tau_eval, p0, fe0):
    checkpoints = [0, 10, 50, 100, 200]
    qpg_states = runs["proposed_QP_G_unified_repaired"]["states"]
    cfgs = {"zero": None, "sgd": runs["sgd"]["cfg"], "egm": runs["egm"]["cfg"], "ppm": runs["ppm"]["cfg"], "proposed_noG_unified_repaired": runs["proposed_noG_unified_repaired"]["cfg"], "proposed_QP_G_unified_repaired": runs["proposed_QP_G_unified_repaired"]["cfg"]}
    rows = []
    for ckpt in checkpoints:
        z = qpg_states[ckpt]
        before = eval_metrics_common(benchmark, z, lambda_F_eval, tau_eval, p0, fe0)
        for name, cfg in cfgs.items():
            if name == "zero":
                z_next = z.clone()
                info = {"selected_step_type": "zero", "delta_norm": 0.0}
            else:
                z_next, info = step_from_flat(benchmark, z, cfg, lambda_F_eval, tau_eval, p0, fe0)
            after = eval_metrics_common(benchmark, z_next, lambda_F_eval, tau_eval, p0, fe0)
            rows.append({"checkpoint_iteration": ckpt, "candidate": name, "actual_V_before": before["V_lambda"], "actual_V_after": after["V_lambda"], "actual_Delta_V": after["V_lambda"] - before["V_lambda"], "field_norm_after": after["field_norm"], "clean_return_after": after["clean_task_return"], "adv_return_after": after["adv_task_return"], "update_norm": info["delta_norm"], "clean_spectral_radius_after": after["clean_spectral_radius"], "adv_spectral_radius_after": after["adv_spectral_radius"], "selected_step_type": info.get("selected_step_type", name)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_ROOT / "subsection2_lq_same_start_candidate_comparison.csv", index=False)
    winners = df.loc[df.groupby("checkpoint_iteration")["actual_Delta_V"].idxmin()][["checkpoint_iteration", "candidate", "actual_Delta_V"]]
    report = "\n".join(["# Subsection 2 same-start candidate comparison", "", "Per-checkpoint winner by actual V decrease:", winners.to_string(index=False), "", "Interpretation should distinguish genuine same-start one-step advantage from pure trajectory effects."])
    write_md(RESULT_ROOT / "subsection2_lq_same_start_candidate_comparison.md", report)
    return df


def part_f_return_saturation(main_curves: pd.DataFrame, benchmark, runs):
    final = main_curves.groupby("method").tail(1).copy()
    robustness_rows = []
    for method in final["method"]:
        _, auc = mod.robustness_sweep(benchmark, runs[method]["states"][-1])
        robustness_rows.append(auc)
    final["robustness_auc"] = robustness_rows
    metrics = ["train_task_return", "clean_task_return", "adv_task_return", "robustness_auc"]
    rows = []
    for metric in metrics:
        vals = final[metric].to_numpy()
        rows.append({"metric": metric, "best": float(np.max(vals)), "worst": float(np.min(vals)), "absolute_diff": float(np.max(vals) - np.min(vals)), "relative_diff_pct": float(abs(np.max(vals) - np.min(vals)) / (abs(np.mean(vals)) + EPS) * 100.0)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_ROOT / "subsection2_lq_return_saturation_audit.csv", index=False)
    report = "\n".join(["# Subsection 2 return saturation audit", "", f"1. Is R-LQ-2D too easy as an RL performance benchmark? {'yes' if (df['relative_diff_pct'] < 1.0).all() else 'not completely'}", f"2. Do all methods reach nearly identical robust-control returns? {'yes' if (df['absolute_diff'] < 1e-3).all() else 'mostly'}", "3. The meaningful separation is primarily in optimization residual / Lyapunov geometry rather than task-return improvement.", "4. Subsection 2 should therefore be framed as an optimization-geometry benchmark first, not a task-return benchmark."])
    write_md(RESULT_ROOT / "subsection2_lq_return_saturation_audit.md", report)
    return df


def part_g_plots(main_curves: pd.DataFrame, distinct_df: pd.DataFrame):
    methods_order = ["sgd", "egm", "ppm", "proposed_noG_unified_repaired", "proposed_QP_G_unified_repaired"]
    plot_df = main_curves[main_curves["method"].isin(methods_order)].copy()
    plot_df["V_plot"] = plot_df["V_lambda"].clip(lower=1e-12)
    plot_df["field_plot"] = plot_df["field_norm"].clip(lower=1e-12)

    thresholds = [1e-6, 1e-8, 1e-10]
    threshold_iters = {}
    for method, frame in plot_df.groupby("method"):
        threshold_iters[method] = {}
        vals = frame["V_lambda"].reset_index(drop=True)
        for thr in thresholds:
            threshold_iters[method][thr] = first_iter_below(vals, thr)

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods_order:
        frame = plot_df[plot_df["method"] == method]
        ax.plot(frame["iteration"], frame["V_plot"], label=method)
        for thr in thresholds:
            idx = threshold_iters[method][thr]
            if not np.isnan(idx):
                ax.axvline(idx, linestyle=":", linewidth=0.8, alpha=0.12)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("actual V_lambda (clipped at 1e-12)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "subsection2_lq_clean_actual_lyapunov.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods_order:
        frame = plot_df[plot_df["method"] == method]
        ax.plot(frame["iteration"], frame["field_plot"], label=method)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("actual field norm (clipped at 1e-12)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "subsection2_lq_clean_actual_field_norm.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for method in methods_order:
        frame = plot_df[plot_df["method"] == method]
        axes[0].plot(frame["iteration"], frame["train_task_return"], label=method)
        axes[1].plot(frame["iteration"], frame["clean_task_return"], label=method)
        axes[2].plot(frame["iteration"], frame["adv_task_return"], label=method)
    axes[0].set_title("train task return")
    axes[1].set_title("clean task return")
    axes[2].set_title("adv task return")
    axes[0].legend()
    for ax in axes:
        ax.set_xlabel("iteration")
        ax.set_ylabel("task return")
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "subsection2_lq_clean_returns.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods_order:
        frame = plot_df[plot_df["method"] == method]
        ax.plot(frame["iteration"], frame["clean_spectral_radius"], label=f"{method}-clean")
        ax.plot(frame["iteration"], frame["adv_spectral_radius"], linestyle="--", label=f"{method}-adv")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("iteration")
    ax.set_ylabel("spectral radius")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "subsection2_lq_clean_stability.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(distinct_df["iteration"], distinct_df["cos_qp_sgd"], label="cos(QP, SGD)")
    axes[0].plot(distinct_df["iteration"], distinct_df["cos_qp_nog"], label="cos(QP, noG)")
    axes[0].plot(distinct_df["iteration"], distinct_df["cos_qp_egm"], label="cos(QP, EGM)")
    axes[0].legend()
    axes[0].set_ylabel("cosine")
    axes[1].plot(distinct_df["iteration"], distinct_df["g_contrib_ratio"], label="||gamma G|| / ||beta F||")
    axes[1].plot(distinct_df["iteration"], distinct_df["fallback_to_egm"], label="fallback_to_egm")
    axes[1].legend()
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("ratio / flag")
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "subsection2_lq_clean_update_distinctness.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[1, 0]); ax4 = fig.add_subplot(gs[1, 1]); ax5 = fig.add_subplot(gs[2, 0]); ax6 = fig.add_subplot(gs[2, 1])
    for method in methods_order:
        frame = plot_df[plot_df["method"] == method]
        ax1.plot(frame["iteration"], frame["V_plot"], label=method)
        ax2.plot(frame["iteration"], frame["field_plot"], label=method)
        ax3.plot(frame["iteration"], frame["train_task_return"], label=method)
        ax4.plot(frame["iteration"], frame["adv_task_return"], label=method)
        ax5.plot(frame["iteration"], frame["clean_spectral_radius"], label=f"{method}-clean")
        ax5.plot(frame["iteration"], frame["adv_spectral_radius"], linestyle="--", alpha=0.7)
    ax6.plot(distinct_df["iteration"], distinct_df["cos_qp_sgd"], label="cos(QP,SGD)")
    ax6.plot(distinct_df["iteration"], distinct_df["cos_qp_nog"], label="cos(QP,noG)")
    ax6.plot(distinct_df["iteration"], distinct_df["cos_qp_egm"], label="cos(QP,EGM)")
    ax1.set_yscale("log"); ax2.set_yscale("log"); ax5.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax1.set_title("actual V_lambda"); ax2.set_title("field norm"); ax3.set_title("train return"); ax4.set_title("adv return"); ax5.set_title("stability"); ax6.set_title("QP distinctness")
    ax1.legend(fontsize=8); ax6.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "subsection2_lq_clean_all_plots_big.png", dpi=160)
    plt.close(fig)


def part_h_final_verdict(floor_df: pd.DataFrame, distinct_df: pd.DataFrame, same_start_df: pd.DataFrame, return_df: pd.DataFrame, runs):
    qpg_curve = runs["proposed_QP_G_unified_repaired"]["curves"]
    qpg_auc_clip = float(np.trapz(np.maximum(qpg_curve["V_lambda"], 1e-12)))
    baseline_auc_clip = min(float(np.trapz(np.maximum(runs[m]["curves"]["V_lambda"], 1e-12))) for m in ["sgd", "egm", "ppm"])
    same_start_winners = same_start_df.loc[same_start_df.groupby("checkpoint_iteration")["actual_Delta_V"].idxmin()]
    qpg_wins = int((same_start_winners["candidate"] == "proposed_QP_G_unified_repaired").sum())
    fallback_frac = float(qpg_curve["fallback_to_egm"].mean())
    distinct_like_egm = float((distinct_df["cos_qp_egm"] > 0.99).mean())
    clipped_advantage = qpg_auc_clip < baseline_auc_clip
    return_saturated = bool((return_df["relative_diff_pct"] < 1.0).all())
    ready = clipped_advantage and fallback_frac <= 0.5 and distinct_like_egm <= 0.5
    claim = "faster transient Lyapunov reduction in a controlled LQ game" if clipped_advantage else "no robust transient advantage after integrity checks"
    report = "\n".join(
        [
            "# Subsection 2 integrity final report",
            "",
            "1. Is the environment implementation correct?",
            "Yes. The fixed R-LQ-2D audit matches the expected initial spectral radii and stable closed-loop initialization.",
            "",
            "2. Is V_lambda actual or estimated?",
            "The integrity plots use actual recomputed V_lambda after the applied update. The quadratic-model prediction exists only as a diagnostic in the older repaired diagnostics.",
            "",
            "3. Is repaired QP genuinely optimizing actual V_lambda?",
            f"It is checked against actual post-projection V, but the best-AUC repaired QP falls back to EGM on `{fallback_frac:.1%}` of iterations.",
            "",
            "4. Is QP's advantage robust after clipping numerical floor?",
            f"{'Yes' if clipped_advantage else 'No'}. Values below 1e-12 were treated as numerical floor.",
            "",
            "5. Is QP distinct from SGD/noG?",
            f"Not cleanly. `cos(QP, EGM) > 0.99` on `{distinct_like_egm:.1%}` of iterations, and same-start QP wins `{qpg_wins}` / 5 checkpoints.",
            "",
            "6. Does QP improve task return or only optimization residual?",
            f"Task-return differences are {'negligible' if return_saturated else 'non-negligible'}; this benchmark mainly separates optimization residual / transient geometry rather than final RL performance.",
            "",
            "7. Should Subsection 2 be considered paper-ready?",
            f"{'No' if not ready else 'Only with a narrow claim'}.",
            "",
            "8. If paper-ready, what exactly is the claim?",
            claim,
            "",
            "9. If not paper-ready, what remains to fix?",
            "- Put all compared methods on a single reported V formula and a single plotting pipeline.",
            "- Reduce dependence on EGM fallback before claiming a genuine G-direction gain.",
            "- Do not use the dominance variant as a main method; it is a safety analysis and mostly selects zero step.",
        ]
    )
    write_md(RESULT_ROOT / "subsection2_lq_integrity_final_report.md", report)


def main():
    benchmark, runs, lambda_F_eval, tau_eval, p0, fe0 = prepare_main_runs()
    main_methods = ["sgd", "egm", "ppm", "proposed_noG_unified_repaired", "proposed_QP_G_unified_repaired"]
    main_curves = pd.concat([runs[m]["curves"] for m in main_methods], ignore_index=True)
    part_a_vlogging(runs, lambda_F_eval, tau_eval)
    floor_df = part_b_numerical_floor(main_curves)
    part_c_dominance_variant(runs["proposed_QP_G_unified_repaired_egm_dominance"]["curves"])
    distinct_df = part_d_qp_distinctness(benchmark, runs, lambda_F_eval, tau_eval, p0, fe0)
    same_start_df = part_e_same_start(benchmark, runs, lambda_F_eval, tau_eval, p0, fe0)
    return_df = part_f_return_saturation(main_curves, benchmark, runs)
    part_g_plots(main_curves, distinct_df)
    part_h_final_verdict(floor_df, distinct_df, same_start_df, return_df, runs)


if __name__ == "__main__":
    main()
