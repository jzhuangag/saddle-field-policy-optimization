from __future__ import annotations

import argparse
import math
import pathlib
import subprocess
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.run_proposed_qp_new_v2_rawfg_audit import (
    build_control_manager,
    clone_state,
    collect_probe_batches,
    evaluate_state,
    finite_difference_g_raw,
    flatten_named_tensors,
    named_parameters,
    safe_div,
    tensor_norm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage M9 rawFG diagnostics")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument("--python-path", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--role", type=str, default="protagonist", choices=["protagonist", "adversary"])
    parser.add_argument("--eta-egm", type=float, default=1e-3)
    parser.add_argument("--fd-eps", type=float, default=1e-3)
    parser.add_argument("--beta-probe", type=float, default=1e-3)
    parser.add_argument("--gamma-probe", type=float, default=1e-6)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--actor-weight", type=float, default=1.0)
    parser.add_argument("--logstd-weight", type=float, default=1.0)
    parser.add_argument("--critic-weight", type=float, default=0.3)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--num-probes", type=int, default=4)
    return parser.parse_args()


def run_command(command: List[str], cwd: pathlib.Path, stdout_path: pathlib.Path, stderr_path: pathlib.Path) -> None:
    result = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def aggregate_effective_coefficients(m8_diag: pd.DataFrame, eta_egm: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for method, group in m8_diag.groupby("method"):
        eta_ext = float(group["eta"].dropna().iloc[0]) if "eta" in group.columns and not group["eta"].dropna().empty else np.nan
        beta_mean = float(group["beta"].mean()) if "beta" in group.columns else np.nan
        gamma_mean = float(group["gamma"].mean()) if "gamma" in group.columns else np.nan
        beta_eff_mean = float(group["beta_eff"].mean()) if "beta_eff" in group.columns else eta_ext * beta_mean
        gamma_eff_mean = float(group["gamma_eff"].mean()) if "gamma_eff" in group.columns else eta_ext * gamma_mean
        beta_eff_over_eta, _ = safe_div(beta_eff_mean, eta_egm, eps=1e-12)
        gamma_eff_over_eta2, _ = safe_div(gamma_eff_mean, eta_egm ** 2, eps=1e-12)
        rows.append(
            {
                "method": method,
                "eta_ext": eta_ext,
                "beta_mean": beta_mean,
                "gamma_mean": gamma_mean,
                "beta_eff_mean": beta_eff_mean,
                "gamma_eff_mean": gamma_eff_mean,
                "beta_eff_over_eta_egm": beta_eff_over_eta,
                "gamma_eff_over_eta_egm_squared": gamma_eff_over_eta2,
                "beta_at_bound_frac": float(group["beta_at_bound"].mean()) if "beta_at_bound" in group.columns else np.nan,
                "gamma_at_bound_frac": float(group["gamma_at_bound"].mean()) if "gamma_at_bound" in group.columns else np.nan,
                "update_norm_mean": float(group["update_norm_post_cap"].mean()) if "update_norm_post_cap" in group.columns else np.nan,
                "approx_kl_mean": float(group["approx_kl"].mean()) if "approx_kl" in group.columns else np.nan,
                "clip_fraction_mean": float(group["clip_fraction"].mean()) if "clip_fraction" in group.columns else np.nan,
                "actual_V_change_mean": float(group["actual_V_change"].mean()) if "actual_V_change" in group.columns else np.nan,
                "gamma_active_frac_mean": float(group["gamma_active_frac"].mean()) if "gamma_active_frac" in group.columns else np.nan,
                "G_contribution_norm_mean": float(group["G_contribution_norm"].mean()) if "G_contribution_norm" in group.columns else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)


def write_m9a_report(df: pd.DataFrame, path: pathlib.Path, eta_egm: float) -> None:
    qp01 = df[df["method"] == "proposed_qp_rawFG_eta0.1"].iloc[0]
    qp002 = df[df["method"] == "proposed_qp_rawFG_eta0.02"].iloc[0]
    report = [
        "# Stage M9A Effective Coefficient Audit",
        "",
        f"- Reference `eta_EGM = {eta_egm}`",
        f"- Reference `eta_EGM^2 = {eta_egm ** 2}`",
        "",
        f"1. Did eta=0.1 actually reach EGM-like beta_eff? `{bool(qp01['beta_eff_mean'] >= eta_egm)}`",
        f"2. Did eta=0.1 actually reach EGM-like gamma_eff? `{bool(qp01['gamma_eff_mean'] >= eta_egm ** 2)}`",
        f"3. Did eta=0.02 under-step relative to EGM? `{bool(qp002['beta_eff_mean'] < eta_egm and qp002['gamma_eff_mean'] < eta_egm ** 2)}`",
        f"4. Were beta/gamma frequently at bounds? `eta0.1 beta={qp01['beta_at_bound_frac']:.3f}, gamma={qp01['gamma_at_bound_frac']:.3f}; eta0.02 beta={qp002['beta_at_bound_frac']:.3f}, gamma={qp002['gamma_at_bound_frac']:.3f}`",
        f"5. Is QP underpowered because bounds were not eta-scaled? `{bool(qp01['beta_eff_mean'] < eta_egm or qp002['beta_eff_mean'] < eta_egm or qp002['gamma_eff_mean'] < eta_egm ** 2)}`",
        "",
        "## Proposed methods",
    ]
    for _, row in df.iterrows():
        report.extend(
            [
                "",
                f"- `{row['method']}`",
                f"  eta_ext={row['eta_ext']:.6g}, beta_mean={row['beta_mean']:.6e}, gamma_mean={row['gamma_mean']:.6e}",
                f"  beta_eff_mean={row['beta_eff_mean']:.6e}, gamma_eff_mean={row['gamma_eff_mean']:.6e}",
                f"  beta_eff / eta_EGM={row['beta_eff_over_eta_egm']:.6f}, gamma_eff / eta_EGM^2={row['gamma_eff_over_eta_egm_squared']:.6f}",
                f"  beta_at_bound_frac={row['beta_at_bound_frac']:.3f}, gamma_at_bound_frac={row['gamma_at_bound_frac']:.3f}",
                f"  update_norm_mean={row['update_norm_mean']:.6e}, approx_kl_mean={row['approx_kl_mean']:.6e}, clip_fraction_mean={row['clip_fraction_mean']:.6e}",
                f"  actual_V_change_mean={row['actual_V_change_mean']:.6e}",
            ]
        )
    path.write_text("\n".join(report), encoding="utf-8")


def run_rawfg_audit_case(
    args: argparse.Namespace,
    *,
    output_dir: pathlib.Path,
    external_eta: float,
    beta_max: float,
    gamma_max: float,
    update_cap: float | None = None,
) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        args.python_path,
        "scripts/run_proposed_qp_new_v2_rawfg_audit.py",
        "--output-dir",
        str(output_dir),
        "--env",
        args.env,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--role",
        args.role,
        "--num-probes",
        str(args.num_probes),
        "--eta-egm",
        str(args.eta_egm),
        "--external-eta",
        str(external_eta),
        "--fd-eps",
        str(args.fd_eps),
        "--beta-probe",
        str(args.beta_probe),
        "--gamma-probe",
        str(args.gamma_probe),
        "--ridge",
        str(args.ridge),
        "--actor-weight",
        str(args.actor_weight),
        "--logstd-weight",
        str(args.logstd_weight),
        "--critic-weight",
        str(args.critic_weight),
        "--max-grad-norm",
        str(args.max_grad_norm),
        "--vf-coef",
        str(args.vf_coef),
        "--single-beta-max",
        str(beta_max),
        "--single-gamma-max",
        str(gamma_max),
    ]
    if update_cap is not None:
        command.extend(["--update-cap", str(update_cap)])
    run_command(command, pathlib.Path(args.repo_dir), output_dir / "stdout.txt", output_dir / "stderr.txt")
    return output_dir / "raw_fg_qp_audit.csv"


def summarize_inclusion_case(df: pd.DataFrame, case_name: str, eta_ext: float) -> Dict[str, object]:
    qp = df[df["method"] == "proposed_qp_new_v2_rawFG"].copy()
    nog = df[df["method"] == "proposed_noG_new_v2_rawFG"].copy()
    egm = df[df["method"] == "egm"].copy()
    egm_exp = df[df["method"] == "egm_expansion"].copy()
    return {
        "case_name": case_name,
        "eta_ext": eta_ext,
        "beta_max": float(qp["beta_max"].dropna().iloc[0]),
        "gamma_max": float(qp["gamma_max"].dropna().iloc[0]),
        "beta_QP_mean": float(qp["beta_QP"].mean()),
        "gamma_QP_mean": float(qp["gamma_QP"].mean()),
        "beta_QP_over_eta_EGM_mean": float(qp["beta_QP_over_eta_EGM"].mean()),
        "gamma_QP_over_eta_EGM_squared_mean": float(qp["gamma_QP_over_eta_EGM_squared"].mean()),
        "beta_noG_mean": float(nog["beta_noG"].mean()),
        "beta_noG_over_eta_EGM_mean": float(nog["beta_noG_over_eta_EGM"].mean()),
        "cosine_QP_vs_EGM_mean": float(qp["cosine_delta_QP_vs_EGM"].mean()),
        "relerr_QP_vs_EGM_mean": float(qp["relative_error_delta_QP_vs_EGM"].mean()),
        "cosine_QP_vs_EGM_expansion_mean": float(qp["cosine_delta_QP_vs_EGM_expansion"].mean()),
        "relerr_QP_vs_EGM_expansion_mean": float(qp["relative_error_delta_QP_vs_EGM_expansion"].mean()),
        "V_change_QP_mean": float(qp["actual_V_change"].mean()),
        "V_change_noG_mean": float(nog["actual_V_change"].mean()),
        "V_change_EGM_mean": float(egm["actual_V_change"].mean()),
        "V_change_EGM_exp_mean": float(egm_exp["actual_V_change"].mean()),
        "PPO_loss_change_QP_mean": float(qp["PPO_loss_change"].mean()),
        "PPO_loss_change_noG_mean": float(nog["PPO_loss_change"].mean()),
        "PPO_loss_change_EGM_mean": float(egm["PPO_loss_change"].mean()),
        "update_norm_QP_mean": float(qp["update_norm_post_cap"].mean()) if "update_norm_post_cap" in qp.columns else float(qp["update_norm"].mean()),
        "update_norm_EGM_mean": float(egm["update_norm"].mean()),
        "approx_kl_QP_max": float(qp["approx_kl_after"].max()),
        "clip_fraction_QP_max": float(qp["clip_fraction_after"].max()),
        "gamma_active_frac": float(qp["gamma_active"].mean()),
        "G_contribution_norm_mean": float(qp["G_contribution_norm"].mean()),
        "beta_at_bound_frac": float(qp["beta_at_bound"].mean()),
        "gamma_at_bound_frac": float(qp["gamma_at_bound"].mean()),
    }


def write_m9b_report(df: pd.DataFrame, path: pathlib.Path) -> None:
    best = df.sort_values(["relerr_QP_vs_EGM_mean", "V_change_QP_mean"], ascending=[True, True]).iloc[0]
    report = [
        "# Stage M9B EGM Inclusion Audit",
        "",
        "This audit compares raw-F/G Lyapunov-QP feasible sets against EGM actual and EGM expansion on the same frozen model and minibatches.",
        "",
        f"- Best geometry match to EGM actual: `{best['case_name']}`",
        f"- Is QP closer to EGM actual or EGM expansion? `{'EGM actual' if best['relerr_QP_vs_EGM_mean'] <= best['relerr_QP_vs_EGM_expansion_mean'] else 'EGM expansion'}`",
        f"- Does QP decrease V more than noG in at least one case? `{bool((df['V_change_QP_mean'] < df['V_change_noG_mean']).any())}`",
        f"- Does any case keep `approx_kl_QP <= 0.1` and `clip_fraction_QP <= 0.8`? `{bool(((df['approx_kl_QP_max'] <= 0.1) & (df['clip_fraction_QP_max'] <= 0.8)).any())}`",
        "",
        "## Per-case summary",
    ]
    for _, row in df.iterrows():
        report.extend(
            [
                "",
                f"- `{row['case_name']}`",
                f"  eta_ext={row['eta_ext']:.6g}, beta_max={row['beta_max']:.6e}, gamma_max={row['gamma_max']:.6e}",
                f"  beta_QP_mean={row['beta_QP_mean']:.6e}, gamma_QP_mean={row['gamma_QP_mean']:.6e}",
                f"  beta_QP / eta_EGM={row['beta_QP_over_eta_EGM_mean']:.6f}, gamma_QP / eta_EGM^2={row['gamma_QP_over_eta_EGM_squared_mean']:.6f}",
                f"  cos(QP,EGM)={row['cosine_QP_vs_EGM_mean']:.6f}, relerr(QP,EGM)={row['relerr_QP_vs_EGM_mean']:.6f}",
                f"  cos(QP,EGMexp)={row['cosine_QP_vs_EGM_expansion_mean']:.6f}, relerr(QP,EGMexp)={row['relerr_QP_vs_EGM_expansion_mean']:.6f}",
                f"  V_change_QP={row['V_change_QP_mean']:.6e}, V_change_noG={row['V_change_noG_mean']:.6e}, V_change_EGM={row['V_change_EGM_mean']:.6e}",
                f"  PPO_loss_change_QP={row['PPO_loss_change_QP_mean']:.6e}, PPO_loss_change_EGM={row['PPO_loss_change_EGM_mean']:.6e}",
                f"  update_norm_QP={row['update_norm_QP_mean']:.6e}, update_norm_EGM={row['update_norm_EGM_mean']:.6e}",
                f"  approx_kl_QP_max={row['approx_kl_QP_max']:.6e}, clip_fraction_QP_max={row['clip_fraction_QP_max']:.6e}",
            ]
        )
    path.write_text("\n".join(report), encoding="utf-8")


def plot_m9b_geometry(df: pd.DataFrame, path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df["case_name"], df["cosine_QP_vs_EGM_mean"], marker="o", label="cos(QP, EGM)")
    axes[0].plot(df["case_name"], df["cosine_QP_vs_EGM_expansion_mean"], marker="s", label="cos(QP, EGM expansion)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_title("RawFG QP inclusion geometry")
    axes[0].set_ylabel("Cosine")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(df["case_name"], df["relerr_QP_vs_EGM_mean"], marker="o", label="relerr(QP, EGM)")
    axes[1].plot(df["case_name"], df["relerr_QP_vs_EGM_expansion_mean"], marker="s", label="relerr(QP, EGM expansion)")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_title("RawFG QP relative error")
    axes[1].set_ylabel("Relative error")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def compute_g_fd_rows(args: argparse.Namespace) -> pd.DataFrame:
    manager = build_control_manager(args)
    rarl_model = manager.setup_experiment()
    algo = rarl_model.protagonist if args.role == "protagonist" else rarl_model.adversary
    probes = collect_probe_batches(rarl_model, args.role, args.num_probes)
    named_params_list = named_parameters(algo.policy)
    selected_names = [name for name, _ in named_params_list]
    ent_coef = float(algo.ent_coef)
    rows: List[Dict[str, object]] = []

    for probe in probes:
        theta_old = clone_state(named_params_list)
        base_eval = evaluate_state(
            algo=algo,
            rollout_data=probe.rollout_data,
            named_params=named_params_list,
            theta_state=theta_old,
            max_grad_norm=args.max_grad_norm,
            vf_coef=args.vf_coef,
            ent_coef=ent_coef,
            actor_weight=args.actor_weight,
            logstd_weight=args.logstd_weight,
            critic_weight=args.critic_weight,
            selected_names=selected_names,
        )
        F_raw = {name: base_eval["grads"][name].clone() for name in selected_names}
        F_vec = flatten_named_tensors(F_raw, selected_names)
        F_norm = tensor_norm(F_vec)

        schemes = [("absolute", args.fd_eps, np.nan)]
        for rho in [1e-5, 1e-4, 1e-3]:
            schemes.append((f"relative_rho_{rho:.0e}", rho / max(F_norm, 1e-12), rho))

        ref_vec = None
        for scheme_name, fd_eps, rho in schemes:
            G_raw, valid, plus_eval = finite_difference_g_raw(
                algo=algo,
                rollout_data=probe.rollout_data,
                named_params=named_params_list,
                theta_old=theta_old,
                F_raw=F_raw,
                fd_eps=fd_eps,
                max_grad_norm=args.max_grad_norm,
                vf_coef=args.vf_coef,
                ent_coef=ent_coef,
                actor_weight=args.actor_weight,
                logstd_weight=args.logstd_weight,
                critic_weight=args.critic_weight,
                selected_names=selected_names,
            )
            G_vec = flatten_named_tensors(G_raw, selected_names)
            G_norm = tensor_norm(G_vec)
            if scheme_name == "absolute":
                ref_vec = G_vec.clone()
            cosine_vs_abs = np.nan
            if ref_vec is not None and valid:
                denom = float((th.norm(G_vec) * th.norm(ref_vec)).item())
                cosine_vs_abs = float(th.dot(G_vec, ref_vec).item() / denom) if denom > 1e-12 else 0.0
            g_over_f, denom_small = safe_div(G_norm, F_norm, eps=1e-12)
            rows.append(
                {
                    "probe_idx": probe.probe_idx,
                    "scheme": scheme_name,
                    "rho": rho,
                    "eps_fd": fd_eps,
                    "F_norm": F_norm,
                    "G_norm": G_norm,
                    "actual_probe_displacement_norm": fd_eps * F_norm,
                    "F_plus_norm": tensor_norm(flatten_named_tensors({name: plus_eval["grads"][name] for name in selected_names}, selected_names)),
                    "G_over_F_norm": g_over_f,
                    "G_over_F_norm_denominator_too_small": int(denom_small),
                    "finite_difference_valid": int(valid),
                    "cosine_G_vs_absolute": cosine_vs_abs,
                }
            )
    return pd.DataFrame(rows)


def write_m9c_report(df: pd.DataFrame, path: pathlib.Path) -> None:
    rel = df[df["scheme"].str.startswith("relative_")].copy()
    stable = bool((rel["finite_difference_valid"] == 1).all() and (rel["cosine_G_vs_absolute"].fillna(0.0) > 0.9).any())
    lines = [
        "# Stage M9C Finite-Difference G Audit",
        "",
        f"- Is raw `G = J_F F` numerically stable across tested finite-difference scales? `{stable}`",
        f"- Absolute `eps_fd` used in previous runs: `{df[df['scheme'] == 'absolute']['eps_fd'].iloc[0]:.6e}`",
        "",
        "## Relative-scale summary",
    ]
    for scheme, group in rel.groupby("scheme"):
        lines.extend(
            [
                "",
                f"- `{scheme}`",
                f"  mean eps_fd={group['eps_fd'].mean():.6e}",
                f"  mean displacement={group['actual_probe_displacement_norm'].mean():.6e}",
                f"  mean G/F={group['G_over_F_norm'].mean():.6e}",
                f"  mean cosine_vs_absolute={group['cosine_G_vs_absolute'].mean():.6f}",
                f"  finite_difference_valid_frac={group['finite_difference_valid'].mean():.3f}",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_cap_case(df: pd.DataFrame, cap_label: str) -> Dict[str, object]:
    qp = df[df["method"] == "proposed_qp_new_v2_rawFG"].copy()
    nog = df[df["method"] == "proposed_noG_new_v2_rawFG"].copy()
    core_numeric = ["beta_QP", "gamma_QP", "beta_eff", "gamma_eff", "update_norm_post_cap", "V_before", "V_after", "actual_V_change", "q_pred", "approx_kl_after", "clip_fraction_after", "gamma_active_frac", "G_contribution_norm"]
    core_nan = False
    for col in core_numeric:
        if col not in qp.columns:
            core_nan = True
            break
        vals = pd.to_numeric(qp[col], errors="coerce")
        if not vals.map(np.isfinite).all():
            core_nan = True
            break
    if not core_nan:
        for col in ["beta_noG", "beta_eff", "update_norm_post_cap", "V_before", "V_after", "actual_V_change", "q_pred", "approx_kl_after", "clip_fraction_after"]:
            vals = pd.to_numeric(nog[col], errors="coerce")
            if not vals.map(np.isfinite).all():
                core_nan = True
                break
    reasons: List[str] = []
    if core_nan:
        reasons.append("core_nan_or_inf")
    if float(qp["approx_kl_after"].max()) > 0.1:
        reasons.append("approx_kl_spike")
    if float(qp["clip_fraction_after"].max()) > 0.8:
        reasons.append("clip_fraction_saturates")
    if float(qp["actual_V_change"].mean()) >= float(nog["actual_V_change"].mean()):
        reasons.append("V_increase_mean_for_QP")
    if float(qp["gamma_active_frac"].mean()) <= 0.0:
        reasons.append("gamma_inactive_for_QP")
    if float(qp["G_contribution_norm"].mean()) <= 0.0:
        reasons.append("G_contribution_zero_for_QP")
    return {
        "cap_label": cap_label,
        "preflight_pass": len(reasons) == 0,
        "failure_reason": "pass" if not reasons else "|".join(reasons),
        "beta_mean": float(qp["beta_QP"].mean()),
        "gamma_mean": float(qp["gamma_QP"].mean()),
        "beta_eff_mean": float(qp["beta_eff"].mean()),
        "gamma_eff_mean": float(qp["gamma_eff"].mean()),
        "update_norm_pre_cap_mean": float(qp["update_norm_pre_cap"].mean()),
        "update_norm_post_cap_mean": float(qp["update_norm_post_cap"].mean()),
        "cap_active_frac": float(qp["cap_active"].mean()),
        "actual_V_change_QP_mean": float(qp["actual_V_change"].mean()),
        "actual_V_change_noG_mean": float(nog["actual_V_change"].mean()),
        "approx_kl_QP_max": float(qp["approx_kl_after"].max()),
        "clip_fraction_QP_max": float(qp["clip_fraction_after"].max()),
        "gamma_active_frac": float(qp["gamma_active_frac"].mean()),
        "G_contribution_norm_mean": float(qp["G_contribution_norm"].mean()),
    }


def write_m9d_report(df: pd.DataFrame, path: pathlib.Path) -> None:
    best = df.sort_values(["preflight_pass", "actual_V_change_QP_mean", "approx_kl_QP_max"], ascending=[False, True, True]).iloc[0]
    lines = [
        "# Stage M9D eta=1 Explicit-Cap Preflight",
        "",
        f"- Can eta=1 be made safe with an explicit update cap? `{bool(df['preflight_pass'].any())}`",
        f"- Recommended next online candidate from this diagnostic: `{best['cap_label']}`",
        "",
        "## Cap sweep summary",
    ]
    for _, row in df.iterrows():
        lines.extend(
            [
                "",
                f"- `{row['cap_label']}`",
                f"  preflight_pass={bool(row['preflight_pass'])}, failure_reason={row['failure_reason']}",
                f"  beta_mean={row['beta_mean']:.6e}, gamma_mean={row['gamma_mean']:.6e}",
                f"  beta_eff_mean={row['beta_eff_mean']:.6e}, gamma_eff_mean={row['gamma_eff_mean']:.6e}",
                f"  update_norm_pre_cap_mean={row['update_norm_pre_cap_mean']:.6e}, update_norm_post_cap_mean={row['update_norm_post_cap_mean']:.6e}, cap_active_frac={row['cap_active_frac']:.3f}",
                f"  actual_V_change_QP_mean={row['actual_V_change_QP_mean']:.6e}, actual_V_change_noG_mean={row['actual_V_change_noG_mean']:.6e}",
                f"  approx_kl_QP_max={row['approx_kl_QP_max']:.6e}, clip_fraction_QP_max={row['clip_fraction_QP_max']:.6e}",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_root / "preflight" / "stageM9"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # M9A
    m8_diag = pd.read_csv(output_root / "rawFG_M8_qp_diagnostics.csv")
    m8_diag = m8_diag[m8_diag["method"].str.contains("proposed")].copy()
    eff_df = aggregate_effective_coefficients(m8_diag, args.eta_egm)
    eff_df.to_csv(output_root / "stageM9_effective_coefficient_audit.csv", index=False)
    write_m9a_report(eff_df, output_root / "stageM9_effective_coefficient_audit_report.md", args.eta_egm)

    # M9B
    inclusion_cases = [
        ("case1_coeff_mode", 1.0, 1e-2, 3e-5),
        ("case2_eta01_scaled_bounds", 0.1, 1e-1, 3e-4),
        ("case3_eta002_scaled_bounds", 0.02, 5e-1, 1.5e-3),
    ]
    inclusion_rows: List[Dict[str, object]] = []
    for case_name, eta_ext, beta_max, gamma_max in inclusion_cases:
        csv_path = run_rawfg_audit_case(
            args,
            output_dir=tmp_dir / case_name,
            external_eta=eta_ext,
            beta_max=beta_max,
            gamma_max=gamma_max,
            update_cap=None,
        )
        inclusion_rows.append(summarize_inclusion_case(pd.read_csv(csv_path), case_name, eta_ext))
    inclusion_df = pd.DataFrame(inclusion_rows)
    inclusion_df.to_csv(output_root / "stageM9_egm_inclusion_audit.csv", index=False)
    write_m9b_report(inclusion_df, output_root / "stageM9_egm_inclusion_audit_report.md")
    plot_m9b_geometry(inclusion_df, plots_dir / "stageM9_egm_inclusion_geometry.png")

    # M9C
    g_fd_df = compute_g_fd_rows(args)
    g_fd_df.to_csv(output_root / "stageM9_G_finite_difference_audit.csv", index=False)
    write_m9c_report(g_fd_df, output_root / "stageM9_G_finite_difference_audit_report.md")

    # M9D
    cap_cases: List[Tuple[str, float | None]] = [
        ("cap_none", None),
        ("cap_0.003", 0.003),
        ("cap_0.005", 0.005),
        ("cap_0.01", 0.01),
    ]
    cap_rows: List[Dict[str, object]] = []
    for cap_label, cap_value in cap_cases:
        csv_path = run_rawfg_audit_case(
            args,
            output_dir=tmp_dir / f"eta1_{cap_label}",
            external_eta=1.0,
            beta_max=1e-2,
            gamma_max=3e-5,
            update_cap=cap_value,
        )
        cap_rows.append(summarize_cap_case(pd.read_csv(csv_path), cap_label))
    cap_df = pd.DataFrame(cap_rows)
    cap_df.to_csv(output_root / "stageM9_eta1_cap_preflight.csv", index=False)
    write_m9d_report(cap_df, output_root / "stageM9_eta1_cap_preflight_report.md")


if __name__ == "__main__":
    main()
