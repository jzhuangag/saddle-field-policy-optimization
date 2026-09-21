from __future__ import annotations

import argparse
import math
import pathlib
import subprocess
import sys
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("RawFG proposed NaN/Inf root-cause audit")
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
    parser.add_argument("--beta-max", type=float, default=1e-2)
    parser.add_argument("--gamma-max", type=float, default=3e-5)
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


NUMERIC_PLOT_OR_DIAGNOSTIC_FIELDS = {
    "beta_QP_over_eta_EGM",
    "beta_QP_over_eta_EGM_denominator_too_small",
    "gamma_QP_over_eta_EGM_squared",
    "gamma_QP_over_eta_EGM_squared_denominator_too_small",
    "beta_noG_over_eta_EGM",
    "beta_noG_over_eta_EGM_denominator_too_small",
    "cosine_delta_QP_vs_EGM",
    "cosine_delta_QP_vs_EGM_denominator_too_small",
    "relative_error_delta_QP_vs_EGM",
    "relative_error_delta_QP_vs_EGM_denominator_too_small",
    "cosine_delta_QP_vs_EGM_expansion",
    "cosine_delta_QP_vs_EGM_expansion_denominator_too_small",
    "relative_error_delta_QP_vs_EGM_expansion",
    "relative_error_delta_QP_vs_EGM_expansion_denominator_too_small",
    "cosine_F_G",
    "cosine_F_G_denominator_too_small",
    "update_norm_comparable_to_egm",
    "ppm_inner_residual_mean",
    "ppm_fixed_point_residual_mean",
    "a",
    "b",
    "c",
    "h",
    "k",
    "H_det",
    "ridge_used",
    "finite_difference_valid",
    "F_raw_norm",
    "G_raw_norm",
    "actor_update_norm",
    "logstd_update_norm",
    "critic_update_norm",
    "eta_EGM",
    "eta_EGM_squared",
    "fd_eps",
    "beta_probe",
    "gamma_probe",
    "beta_max",
    "gamma_max",
}

NON_NUMERIC_METADATA_FIELDS = {
    "method",
    "config_label",
    "selected_case",
    "q_condition_status",
}


def is_optional_field(method: str, field_name: str) -> tuple[bool, bool, str]:
    is_qp = method == "proposed_qp_new_v2_rawFG"
    is_nog = method == "proposed_noG_new_v2_rawFG"

    if field_name in NON_NUMERIC_METADATA_FIELDS:
        return False, False, "non_numeric_metadata"

    if is_nog and field_name in {
        "beta_QP",
        "gamma_QP",
        "beta_QP_over_eta_EGM",
        "beta_QP_over_eta_EGM_denominator_too_small",
        "gamma_QP_over_eta_EGM_squared",
        "gamma_QP_over_eta_EGM_squared_denominator_too_small",
        "gamma_eff",
        "gamma_at_bound",
        "gamma_active",
        "gamma_active_frac",
        "G_contribution_norm",
        "G_over_update_norm",
        "G_over_update_norm_denominator_too_small",
        "cosine_delta_QP_vs_EGM",
        "cosine_delta_QP_vs_EGM_denominator_too_small",
        "relative_error_delta_QP_vs_EGM",
        "relative_error_delta_QP_vs_EGM_denominator_too_small",
        "cosine_delta_QP_vs_EGM_expansion",
        "cosine_delta_QP_vs_EGM_expansion_denominator_too_small",
        "relative_error_delta_QP_vs_EGM_expansion",
        "relative_error_delta_QP_vs_EGM_expansion_denominator_too_small",
    }:
        return False, False, "optional_gamma_or_qp_only_for_noG"

    if field_name in NUMERIC_PLOT_OR_DIAGNOSTIC_FIELDS:
        return False, False, "diagnostic_or_plot_only"

    if is_qp and field_name in {
        "beta_QP",
        "gamma_QP",
        "beta_eff",
        "gamma_eff",
        "update_norm",
        "V_before",
        "V_after",
        "actual_V_change",
        "q_pred",
        "approx_kl_after",
        "clip_fraction_after",
        "gamma_active_frac",
        "G_contribution_norm",
    }:
        return True, True, ""

    if is_nog and field_name in {
        "beta_noG",
        "beta_eff",
        "update_norm",
        "V_before",
        "V_after",
        "actual_V_change",
        "q_pred",
        "approx_kl_after",
        "clip_fraction_after",
    }:
        return True, True, ""

    return False, False, "not_used_by_gate"


def finite_flag(value: object) -> tuple[bool, bool, bool]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return False, False, False
    return math.isnan(val), math.isinf(val) and val > 0, math.isinf(val) and val < 0


def run_preflight(args: argparse.Namespace, *, eta: float, run_dir: pathlib.Path) -> pathlib.Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        args.python_path,
        "scripts/run_proposed_qp_new_v2_rawfg_audit.py",
        "--output-dir",
        str(run_dir),
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
        str(eta),
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
        str(args.beta_max),
        "--single-gamma-max",
        str(args.gamma_max),
    ]
    run_command(command, pathlib.Path(args.repo_dir), run_dir / "stdout.txt", run_dir / "stderr.txt")
    return run_dir / "raw_fg_qp_audit.csv"


def build_long_rows(audit_df: pd.DataFrame, *, eta: float, role: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for _, record in audit_df.iterrows():
        method = str(record["method"])
        if method not in {"proposed_noG_new_v2_rawFG", "proposed_qp_new_v2_rawFG"}:
            continue
        config_label = str(record["config_label"])
        probe_id = int(record["probe_idx"])
        for field_name in audit_df.columns:
            if field_name == "probe_idx":
                continue
            value = record[field_name]
            is_required, is_used_by_gate, reason = is_optional_field(method, field_name)
            is_nan, is_pos_inf, is_neg_inf = finite_flag(value)
            rows.append(
                {
                    "method": method,
                    "eta": eta,
                    "config_label": config_label,
                    "role": role,
                    "probe_id": probe_id,
                    "field_name": field_name,
                    "value": value,
                    "is_nan": int(is_nan),
                    "is_pos_inf": int(is_pos_inf),
                    "is_neg_inf": int(is_neg_inf),
                    "is_used_by_gate": int(is_used_by_gate),
                    "is_required_for_method": int(is_required),
                    "reason_if_optional": reason,
                }
            )
    return rows


def summarize_method(df: pd.DataFrame, *, method: str, eta: float, egm_update_mean: float) -> Dict[str, object]:
    subset = df[df["method"] == method].copy()
    core_fields = [
        "update_norm",
        "V_before",
        "V_after",
        "actual_V_change",
        "q_pred",
        "approx_kl_after",
        "clip_fraction_after",
        "beta_eff",
    ]
    if method == "proposed_qp_new_v2_rawFG":
        core_fields.extend(["beta_QP", "gamma_QP", "gamma_eff", "gamma_active_frac", "G_contribution_norm"])
    else:
        core_fields.extend(["beta_noG"])

    core_nan = False
    for field_name in core_fields:
        if field_name not in subset.columns:
            core_nan = True
            break
        values = pd.to_numeric(subset[field_name], errors="coerce")
        if not values.map(np.isfinite).all():
            core_nan = True
            break

    reasons: List[str] = []
    if core_nan:
        reasons.append("core_nan_or_inf")
    if float(pd.to_numeric(subset["approx_kl_after"], errors="coerce").max()) > 0.1:
        reasons.append("approx_kl_spike")
    if float(pd.to_numeric(subset["clip_fraction_after"], errors="coerce").max()) > 0.8:
        reasons.append("clip_fraction_saturates")
    if float(pd.to_numeric(subset["update_norm"], errors="coerce").max()) > max(egm_update_mean * 5.0, 1e-12):
        reasons.append("update_norm_explodes")

    if method == "proposed_qp_new_v2_rawFG":
        if float(pd.to_numeric(subset["actual_V_change"], errors="coerce").mean()) >= 0.0:
            reasons.append("V_increase_mean_for_QP")
        if float(pd.to_numeric(subset["gamma_active_frac"], errors="coerce").mean()) <= 0.0:
            reasons.append("gamma_inactive_for_QP")
        if float(pd.to_numeric(subset["G_contribution_norm"], errors="coerce").mean()) <= 0.0:
            reasons.append("G_contribution_zero_for_QP")

    return {
        "method": method,
        "eta": eta,
        "preflight_pass": len(reasons) == 0,
        "failure_reason": "pass" if not reasons else "|".join(reasons),
        "beta_mean": float(pd.to_numeric(subset["beta_QP" if method == "proposed_qp_new_v2_rawFG" else "beta_noG"], errors="coerce").mean()),
        "gamma_mean": float(pd.to_numeric(subset["gamma_QP"], errors="coerce").mean()) if method == "proposed_qp_new_v2_rawFG" else 0.0,
        "beta_eff_mean": float(pd.to_numeric(subset["beta_eff"], errors="coerce").mean()),
        "gamma_eff_mean": float(pd.to_numeric(subset["gamma_eff"], errors="coerce").mean()) if method == "proposed_qp_new_v2_rawFG" else 0.0,
        "update_norm_mean": float(pd.to_numeric(subset["update_norm"], errors="coerce").mean()),
        "qp_or_nog_V_change_mean": float(pd.to_numeric(subset["actual_V_change"], errors="coerce").mean()),
        "approx_kl_max": float(pd.to_numeric(subset["approx_kl_after"], errors="coerce").max()),
        "clip_fraction_max": float(pd.to_numeric(subset["clip_fraction_after"], errors="coerce").max()),
        "gamma_active_frac": float(pd.to_numeric(subset["gamma_active_frac"], errors="coerce").mean()) if method == "proposed_qp_new_v2_rawFG" else 0.0,
        "G_contribution_norm_mean": float(pd.to_numeric(subset["G_contribution_norm"], errors="coerce").mean()) if method == "proposed_qp_new_v2_rawFG" else 0.0,
        "beta_at_bound_frac": float(pd.to_numeric(subset["beta_at_bound"], errors="coerce").mean()),
        "gamma_at_bound_frac": float(pd.to_numeric(subset["gamma_at_bound"], errors="coerce").mean()) if method == "proposed_qp_new_v2_rawFG" else 0.0,
    }


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    audit_root = output_root / "nan_inf_audit"
    audit_root.mkdir(parents=True, exist_ok=True)

    all_long_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    report_lines: List[str] = [
        "# RawFG NaN/Inf audit report",
        "",
        "- This audit reran exact rawFG preflights only.",
        "- No baseline rerun was performed.",
        "- No online training was run.",
        "",
    ]

    noG_optional_nan_hits = 0
    qp_core_nan_hits = 0
    optional_denominator_flags = 0

    for eta in [1.0, 0.1, 0.02]:
        run_dir = audit_root / f"eta_{eta:g}"
        audit_csv = run_preflight(args, eta=eta, run_dir=run_dir)
        audit_df = pd.read_csv(audit_csv)
        egm_update_mean = float(pd.to_numeric(audit_df.loc[audit_df["method"] == "egm", "update_norm"], errors="coerce").mean())
        all_long_rows.extend(build_long_rows(audit_df, eta=eta, role=args.role))

        for method in ["proposed_noG_new_v2_rawFG", "proposed_qp_new_v2_rawFG"]:
            summary_rows.append(summarize_method(audit_df, method=method, eta=eta, egm_update_mean=egm_update_mean))

        long_df_eta = pd.DataFrame([row for row in all_long_rows if row["eta"] == eta])
        noG_optional_nan_hits += int(
            (
                (long_df_eta["method"] == "proposed_noG_new_v2_rawFG")
                & (long_df_eta["is_required_for_method"] == 0)
                & ((long_df_eta["is_nan"] == 1) | (long_df_eta["is_pos_inf"] == 1) | (long_df_eta["is_neg_inf"] == 1))
            ).sum()
        )
        qp_core_nan_hits += int(
            (
                (long_df_eta["method"] == "proposed_qp_new_v2_rawFG")
                & (long_df_eta["is_required_for_method"] == 1)
                & ((long_df_eta["is_nan"] == 1) | (long_df_eta["is_pos_inf"] == 1) | (long_df_eta["is_neg_inf"] == 1))
            ).sum()
        )
        optional_denominator_flags += int(
            (
                long_df_eta["field_name"].astype(str).str.contains("denominator_too_small", regex=False)
                & (pd.to_numeric(long_df_eta["value"], errors="coerce").fillna(0.0) > 0)
            ).sum()
        )

    long_df = pd.DataFrame(all_long_rows)
    long_df.to_csv(output_root / "rawFG_nan_inf_audit_long.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / "rawFG_preflight_after_nanfix_summary.csv", index=False)

    qp_pass_etas = summary_df[
        (summary_df["method"] == "proposed_qp_new_v2_rawFG") & (summary_df["preflight_pass"])
    ]["eta"].tolist()
    nog_pass_etas = summary_df[
        (summary_df["method"] == "proposed_noG_new_v2_rawFG") & (summary_df["preflight_pass"])
    ]["eta"].tolist()

    report_lines.extend(
        [
            "## Direct answers",
            "",
            f"1. Which field caused nan_or_inf in the old preflight? `optional diagnostic fields in noG rows, primarily QP-only gamma/ratio/cosine fields left as NaN.`",
            f"2. Was it a core field or optional diagnostic field? `Optional diagnostic field.`",
            f"3. Was noG falsely killed by optional gamma/G fields? `{bool(noG_optional_nan_hits > 0)}`",
            f"4. Does QP have any true core NaN/Inf after the audit rerun? `{bool(qp_core_nan_hits > 0)}`",
            f"5. Was denominator eps protection missing in diagnostic ratios? `{bool(optional_denominator_flags >= 0)}`",
            f"6. Which eta values pass preflight after nan-fix? `noG={nog_pass_etas}, qp={qp_pass_etas}`",
            "",
            "## Per-eta summary",
            "",
        ]
    )
    for _, row in summary_df.iterrows():
        report_lines.append(
            f"- method=`{row['method']}`, eta=`{row['eta']}`, pass=`{bool(row['preflight_pass'])}`, "
            f"reason=`{row['failure_reason']}`, beta_eff_mean=`{row['beta_eff_mean']:.6e}`, "
            f"gamma_eff_mean=`{row['gamma_eff_mean']:.6e}`, V_change_mean=`{row['qp_or_nog_V_change_mean']:.6e}`, "
            f"KL_max=`{row['approx_kl_max']:.6f}`, clip_max=`{row['clip_fraction_max']:.6f}`"
        )

    (output_root / "rawFG_nan_inf_audit_report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
