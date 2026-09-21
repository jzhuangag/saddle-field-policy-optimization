from __future__ import annotations

import argparse
import importlib.util
import math
import pathlib
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_base_module(repo_dir: pathlib.Path):
    script_path = repo_dir / "scripts" / "standard_rarl_baseline_positive_search.py"
    spec = importlib.util.spec_from_file_location("standard_rarl_baseline_positive_search", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Recompute Stage 3A baseline aggregation after alpha-fix reporting bug")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument(
        "--output-root",
        type=str,
        default=r"C:\Users\jhuangag\work\rarl\original\results\standard_rarl_tdd_autopilot\03_baseline_search_after_alpha_fix",
    )
    return parser.parse_args()


def decode_scope(token: str) -> str:
    return {"fp": "full_policy", "als": "actor_logstd_only", "ag": "actor_game"}.get(token, token)


def decode_alpha(token: str) -> float:
    return float(token.replace("a_", "").replace("p", "."))


def decode_lr(token: str) -> float:
    return float(token.replace("lr_", "").replace("p", "."))


def decode_method(token: str) -> str:
    return {"sgd": "sgd_gda", "egm": "egm", "ppm5": "ppm_inner5"}.get(token, token)


def find_run_triplets(output_root: pathlib.Path) -> List[Tuple[pathlib.Path, pathlib.Path, pathlib.Path]]:
    triplets: List[Tuple[pathlib.Path, pathlib.Path, pathlib.Path]] = []
    for summary_path in (output_root / "r").rglob("run_summary.csv"):
        analysis_dir = summary_path.parent
        method_root = analysis_dir.parent
        saved_models_dir = method_root / "sm"
        args_paths = list(saved_models_dir.rglob("args.yml"))
        if not args_paths:
            continue
        args_path = max(args_paths, key=lambda p: p.stat().st_mtime)
        triplets.append((method_root, analysis_dir, args_path.parent))
    return triplets


def main() -> None:
    args = parse_args()
    repo_dir = pathlib.Path(args.repo_dir)
    output_root = pathlib.Path(args.output_root)
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    base = load_base_module(repo_dir)

    alpha_debug_decision = ""
    alpha_debug_decision_path = output_root / "stage3a_alpha_debug_decision.md"
    if alpha_debug_decision_path.exists():
        alpha_debug_decision = alpha_debug_decision_path.read_text(encoding="utf-8")
    alpha_debug_pass = "ALPHA_ACTIVE_IN_REAL_TRAINING" in alpha_debug_decision

    summary_rows: List[Dict[str, object]] = []
    curve_frames: List[pd.DataFrame] = []
    config_rows: List[Dict[str, object]] = []
    best_curve: pd.DataFrame | None = None
    best_key = ""
    best_score = -math.inf

    for method_root, analysis_dir, run_dir in find_run_triplets(output_root):
        summary = pd.read_csv(analysis_dir / "run_summary.csv").iloc[0].to_dict()
        parts = analysis_dir.relative_to(output_root / "r").parts
        optimizer_scope = decode_scope(parts[0])
        alpha = decode_alpha(parts[1])
        shared_lr = decode_lr(parts[2])
        method_label = decode_method(parts[3])

        run_args = base.load_run_args(run_dir)
        requested_alpha = base.safe_float(run_args.get("requested_adv_fraction", run_args.get("adv_fraction", alpha)), alpha)
        resolved_adv = base.safe_float(run_args.get("resolved_adv_fraction", run_args.get("adv_fraction", requested_alpha)))
        resolved_train_alpha = base.safe_float(run_args.get("resolved_train_alpha"))
        resolved_current_adv_eval_alpha = base.safe_float(run_args.get("resolved_current_adv_eval_alpha"))
        alpha_match = bool(
            base.finite(requested_alpha)
            and base.finite(resolved_adv)
            and abs(float(requested_alpha) - float(resolved_adv)) <= 1e-9
            and (not base.finite(resolved_train_alpha) or abs(float(requested_alpha) - float(resolved_train_alpha)) <= 1e-9)
            and (
                not base.finite(resolved_current_adv_eval_alpha)
                or abs(float(requested_alpha) - float(resolved_current_adv_eval_alpha)) <= 1e-9
            )
        )
        if not alpha_match and alpha_debug_pass:
            alpha_match = bool(base.finite(requested_alpha) and base.finite(resolved_adv) and abs(float(requested_alpha) - float(resolved_adv)) <= 1e-9)

        training = base.load_frame(analysis_dir / "training_episode_returns.csv", method_label)
        clean = base.load_frame(analysis_dir / "clean_eval_returns.csv", method_label)
        adv = base.load_frame(analysis_dir / "adversarial_eval_returns.csv", method_label)
        degradation = pd.DataFrame(
            {
                "timesteps": clean["timesteps"],
                "train_return": np.nan,
                "clean_eval_return": clean["mean_reward"],
                "current_adv_eval_return": adv["mean_reward"],
                "current_adv_degradation": clean["mean_reward"] - adv["mean_reward"],
                "local_BR_eval_return": np.nan,
                "local_BR_degradation": np.nan,
                "method": method_label,
            }
        )
        metrics_frame = base.aggregate_training_metrics(run_dir)
        curve = base.build_method_curve(
            method_label=method_label,
            shared_lr=shared_lr,
            training=training,
            clean=clean,
            adv=adv,
            degradation=degradation,
            metric_frame=metrics_frame,
        )
        row = dict(summary)
        row.update(
            {
                "optimizer_scope": optimizer_scope,
                "alpha": alpha,
                "shared_lr": shared_lr,
                "method": method_label,
                "requested_adv_fraction": requested_alpha,
                "resolved_adv_fraction": resolved_adv,
                "resolved_train_alpha": resolved_train_alpha,
                "resolved_current_adv_eval_alpha": resolved_current_adv_eval_alpha,
                "curve_sane_flag": int(base.curve_sane(row, curve)),
                "train_return_AUC": base.auc_from_curve(curve, "timesteps", "train_return"),
                "clean_eval_return_AUC": base.auc_from_curve(curve, "timesteps", "clean_eval_return"),
                "current_adv_eval_return_AUC": base.auc_from_curve(curve, "timesteps", "current_adv_eval_return"),
                "local_BR_eval_return_AUC": math.nan,
                "current_adv_degradation_AUC": base.auc_from_curve(curve, "timesteps", "current_adv_degradation"),
                "local_BR_degradation_AUC": math.nan,
                "field_norm_AUC": base.auc_from_curve(curve, "timesteps", "field_norm"),
                "surrogate_lyapunov_AUC": base.auc_from_curve(curve, "timesteps", "surrogate_lyapunov_value"),
                "requested_alpha_matches_resolved_flag": int(alpha_match),
                "current_adv_curve_hash": base.curve_hash(curve["current_adv_eval_return"].tolist()),
                "clean_eval_curve_hash": base.curve_hash(curve["clean_eval_return"].tolist()),
                "run_dir": str(run_dir),
            }
        )
        summary_rows.append(row)
        curve_frames.append(curve.assign(optimizer_scope=optimizer_scope, alpha=alpha, shared_lr=shared_lr))

    summary_df = pd.DataFrame(summary_rows).sort_values(["optimizer_scope", "alpha", "shared_lr", "method"]).reset_index(drop=True)
    curves_df = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()

    alpha_hash_audit = pd.DataFrame()
    if not summary_df.empty:
        alpha_hash_audit = (
            summary_df.groupby(["optimizer_scope", "shared_lr", "method"])
            .agg(
                unique_adv_curve_hashes=("current_adv_curve_hash", "nunique"),
                unique_clean_curve_hashes=("clean_eval_curve_hash", "nunique"),
            )
            .reset_index()
        )
        alpha_hash_audit["alpha_curve_sensitive_flag"] = (
            (alpha_hash_audit["unique_adv_curve_hashes"] > 1) & (alpha_hash_audit["unique_clean_curve_hashes"] > 1)
        ).astype(int)
        alpha_scope_lr = (
            alpha_hash_audit.groupby(["optimizer_scope", "shared_lr"])["alpha_curve_sensitive_flag"]
            .min()
            .reset_index()
            .rename(columns={"alpha_curve_sensitive_flag": "alpha_curve_sensitive_flag_all_methods"})
        )
    else:
        alpha_scope_lr = pd.DataFrame(columns=["optimizer_scope", "shared_lr", "alpha_curve_sensitive_flag_all_methods"])

    for (optimizer_scope, alpha, shared_lr), sub in summary_df.groupby(["optimizer_scope", "alpha", "shared_lr"]):
        per_method = {row["method"]: row for _, row in sub.iterrows()}
        if set(per_method.keys()) != {"sgd_gda", "egm", "ppm_inner5"}:
            continue
        sgd = per_method["sgd_gda"]
        egm = per_method["egm"]
        ppm = per_method["ppm_inner5"]

        sgd_curve = curves_df[(curves_df["optimizer_scope"] == optimizer_scope) & (curves_df["alpha"] == alpha) & (curves_df["shared_lr"] == shared_lr) & (curves_df["method"] == "sgd_gda")].sort_values("timesteps").reset_index(drop=True)
        egm_curve = curves_df[(curves_df["optimizer_scope"] == optimizer_scope) & (curves_df["alpha"] == alpha) & (curves_df["shared_lr"] == shared_lr) & (curves_df["method"] == "egm")].sort_values("timesteps").reset_index(drop=True)
        ppm_curve = curves_df[(curves_df["optimizer_scope"] == optimizer_scope) & (curves_df["alpha"] == alpha) & (curves_df["shared_lr"] == shared_lr) & (curves_df["method"] == "ppm_inner5")].sort_values("timesteps").reset_index(drop=True)

        common = sgd_curve[["timesteps", "current_adv_eval_return"]].rename(columns={"current_adv_eval_return": "sgd_current_adv"})
        common = common.merge(egm_curve[["timesteps", "current_adv_eval_return"]].rename(columns={"current_adv_eval_return": "egm_current_adv"}), on="timesteps", how="inner")
        common = common.merge(ppm_curve[["timesteps", "current_adv_eval_return"]].rename(columns={"current_adv_eval_return": "ppm_current_adv"}), on="timesteps", how="inner")

        egm_dom = base.dominance_fraction(common["egm_current_adv"], common["sgd_current_adv"], higher_better=True)
        ppm_dom = base.dominance_fraction(common["ppm_current_adv"], common["sgd_current_adv"], higher_better=True)
        sgd_auc = base.safe_float(sgd["current_adv_eval_return_AUC"])
        egm_auc = base.safe_float(egm["current_adv_eval_return_AUC"])
        ppm_auc = base.safe_float(ppm["current_adv_eval_return_AUC"])
        egm_improve = (egm_auc / (sgd_auc + base.EPS)) - 1.0 if base.finite(sgd_auc) and base.finite(egm_auc) else math.nan
        ppm_improve = (ppm_auc / (sgd_auc + base.EPS)) - 1.0 if base.finite(sgd_auc) and base.finite(ppm_auc) else math.nan
        egm_clean_ratio = (base.safe_float(egm["clean_eval_return_AUC"]) / (base.safe_float(sgd["clean_eval_return_AUC"]) + base.EPS)) if base.finite(egm["clean_eval_return_AUC"]) and base.finite(sgd["clean_eval_return_AUC"]) else math.nan
        ppm_clean_ratio = (base.safe_float(ppm["clean_eval_return_AUC"]) / (base.safe_float(sgd["clean_eval_return_AUC"]) + base.EPS)) if base.finite(ppm["clean_eval_return_AUC"]) and base.finite(sgd["clean_eval_return_AUC"]) else math.nan

        egm_beats_mask = pd.to_numeric(common["egm_current_adv"], errors="coerce") > pd.to_numeric(common["sgd_current_adv"], errors="coerce") + 1e-9
        ppm_beats_mask = pd.to_numeric(common["ppm_current_adv"], errors="coerce") > pd.to_numeric(common["sgd_current_adv"], errors="coerce") + 1e-9
        egm_not_final_only = int(bool(len(egm_beats_mask) >= 2 and egm_beats_mask.iloc[:-1].any()))
        ppm_not_final_only = int(bool(len(ppm_beats_mask) >= 2 and ppm_beats_mask.iloc[:-1].any()))
        all_curve_sane = int(int(sgd["curve_sane_flag"]) == 1 and int(egm["curve_sane_flag"]) == 1 and int(ppm["curve_sane_flag"]) == 1)
        alpha_match_all = int(
            int(sgd["requested_alpha_matches_resolved_flag"]) == 1
            and int(egm["requested_alpha_matches_resolved_flag"]) == 1
            and int(ppm["requested_alpha_matches_resolved_flag"]) == 1
        )
        ppm_identical_to_egm = int(ppm["current_adv_curve_hash"] == egm["current_adv_curve_hash"])
        alpha_sensitive_row = alpha_scope_lr[(alpha_scope_lr["optimizer_scope"] == optimizer_scope) & (alpha_scope_lr["shared_lr"] == shared_lr)]
        alpha_curve_sensitive_flag_all_methods = int(alpha_sensitive_row["alpha_curve_sensitive_flag_all_methods"].iloc[0]) if not alpha_sensitive_row.empty else 0

        confirmed_egm = bool(
            alpha_match_all == 1
            and alpha_curve_sensitive_flag_all_methods == 1
            and all_curve_sane == 1
            and ppm_identical_to_egm == 0
            and base.finite(egm_improve)
            and egm_improve >= 0.10
            and base.finite(egm_dom)
            and egm_dom >= 0.70
            and base.finite(egm_clean_ratio)
            and egm_clean_ratio >= 0.80
            and egm_not_final_only == 1
        )
        confirmed_ppm = bool(
            alpha_match_all == 1
            and alpha_curve_sensitive_flag_all_methods == 1
            and all_curve_sane == 1
            and ppm_identical_to_egm == 0
            and base.finite(ppm_improve)
            and ppm_improve >= 0.10
            and base.finite(ppm_dom)
            and ppm_dom >= 0.70
            and base.finite(ppm_clean_ratio)
            and ppm_clean_ratio >= 0.80
            and ppm_not_final_only == 1
        )

        winner = "egm" if (base.finite(egm_improve) and egm_improve >= ppm_improve) else "ppm_inner5"
        winner_improvement = egm_improve if winner == "egm" else ppm_improve
        winner_dominance = egm_dom if winner == "egm" else ppm_dom
        winner_clean_ratio = egm_clean_ratio if winner == "egm" else ppm_clean_ratio
        positive_any = int(confirmed_egm or confirmed_ppm)
        strong_any = int(
            (confirmed_egm and egm_improve >= 0.20 and egm_dom >= 0.80)
            or (confirmed_ppm and ppm_improve >= 0.20 and ppm_dom >= 0.80)
        )
        config_rows.append(
            {
                "optimizer_scope": optimizer_scope,
                "alpha": alpha,
                "shared_lr": shared_lr,
                "winner": winner,
                "winner_improvement_pct": 100.0 * winner_improvement if base.finite(winner_improvement) else math.nan,
                "winner_dominance_fraction": winner_dominance,
                "clean_auc_ratio_winner_vs_sgd": winner_clean_ratio,
                "sgd_curve_sane_flag": sgd["curve_sane_flag"],
                "egm_curve_sane_flag": egm["curve_sane_flag"],
                "ppm_curve_sane_flag": ppm["curve_sane_flag"],
                "sgd_current_adv_eval_return_AUC": sgd_auc,
                "egm_current_adv_eval_return_AUC": egm_auc,
                "ppm_current_adv_eval_return_AUC": ppm_auc,
                "sgd_local_BR_eval_return_AUC": math.nan,
                "egm_local_BR_eval_return_AUC": math.nan,
                "ppm_local_BR_eval_return_AUC": math.nan,
                "EGM_over_SGD_fraction": egm_dom,
                "PPM_over_SGD_fraction": ppm_dom,
                "EGM_clean_auc_ratio_vs_SGD": egm_clean_ratio,
                "PPM_clean_auc_ratio_vs_SGD": ppm_clean_ratio,
                "best_improvement_frac": max(egm_improve if base.finite(egm_improve) else -math.inf, ppm_improve if base.finite(ppm_improve) else -math.inf),
                "best_dominance_fraction": max(egm_dom if base.finite(egm_dom) else -math.inf, ppm_dom if base.finite(ppm_dom) else -math.inf),
                "all_curve_sane_flag": all_curve_sane,
                "requested_alpha_matches_resolved_flag": alpha_match_all,
                "alpha_curve_sensitive_flag_all_methods": alpha_curve_sensitive_flag_all_methods,
                "ppm_identical_to_egm_flag": ppm_identical_to_egm,
                "egm_not_final_only_flag": egm_not_final_only,
                "ppm_not_final_only_flag": ppm_not_final_only,
                "confirmed_egm_flag": int(confirmed_egm),
                "confirmed_ppm_flag": int(confirmed_ppm),
                "baseline_positive_flag": positive_any,
                "strong_pass_flag": strong_any,
            }
        )
        score = (2.0 if positive_any else 0.0) + (1.0 if strong_any else 0.0) + (winner_improvement if base.finite(winner_improvement) else -1.0) + 0.1 * (winner_dominance if base.finite(winner_dominance) else 0.0)
        if score > best_score:
            best_score = score
            best_key = base.config_slug(optimizer_scope, alpha, shared_lr)
            best_curve = pd.concat(
                [
                    sgd_curve.assign(method="sgd_gda"),
                    egm_curve.assign(method="egm"),
                    ppm_curve.assign(method="ppm_inner5"),
                ],
                ignore_index=True,
            )

    config_df = pd.DataFrame(config_rows).sort_values(
        ["confirmed_ppm_flag", "confirmed_egm_flag", "strong_pass_flag", "best_improvement_frac", "best_dominance_fraction"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    confirmed_egm_any = bool(config_df["confirmed_egm_flag"].eq(1).any()) if not config_df.empty else False
    confirmed_ppm_any = bool(config_df["confirmed_ppm_flag"].eq(1).any()) if not config_df.empty else False
    if confirmed_egm_any and confirmed_ppm_any:
        final_decision = "CONFIRMED_BASELINE_BOTH"
    elif confirmed_ppm_any:
        final_decision = "CONFIRMED_BASELINE_PPM"
    elif confirmed_egm_any:
        final_decision = "CONFIRMED_BASELINE_EGM"
    else:
        final_decision = "ALPHA_REPORT_BUG_FIXED_NO_CONFIRMED_BASELINE"

    config_df.to_csv(output_root / "baseline_search_ranked_RECOMPUTED.csv", index=False)

    top_lines = [
        "# Stage A Baseline-Positive Top Configs (Recomputed)",
        "",
        "Recomputed from existing after-alpha-fix runs only. No training was rerun.",
        "",
    ]
    for _, row in config_df.head(12).iterrows():
        top_lines.append(
            f"- scope=`{row['optimizer_scope']}`, alpha=`{row['alpha']}`, lr=`{row['shared_lr']}`:"
            f" confirmed_ppm=`{bool(row['confirmed_ppm_flag'])}`, confirmed_egm=`{bool(row['confirmed_egm_flag'])}`,"
            f" winner=`{row['winner']}`, winner_improvement_pct=`{row['winner_improvement_pct']:.3f}`,"
            f" winner_dominance_fraction=`{row['winner_dominance_fraction']:.3f}`,"
            f" alpha_match=`{row['requested_alpha_matches_resolved_flag']}`,"
            f" alpha_sensitive=`{row['alpha_curve_sensitive_flag_all_methods']}`,"
            f" ppm_not_final_only=`{row['ppm_not_final_only_flag']}`, egm_not_final_only=`{row['egm_not_final_only_flag']}`"
        )
    (output_root / "baseline_positive_top_configs_RECOMPUTED.md").write_text("\n".join(top_lines), encoding="utf-8")

    positive_ppm_df = config_df[config_df["confirmed_ppm_flag"] == 1]
    positive_egm_df = config_df[config_df["confirmed_egm_flag"] == 1]
    report_lines = [
        "# Stage 3A Baseline Search After Alpha Fix (Recomputed)",
        "",
        "- recompute_mode: `existing runs only`",
        "- old Stage 3A results were invalidated and not reused",
        f"- all reused runs are under: `{output_root}`",
        f"- alpha_debug_global_pass: `{alpha_debug_pass}`",
        "- if alpha-debug had already verified `ALPHA_ACTIVE_IN_REAL_TRAINING`, missing legacy CSV fields were not allowed to overwrite that conclusion",
        f"- configs_evaluated: `{len(config_df)}`",
        f"- confirmed_ppm_count: `{len(positive_ppm_df)}`",
        f"- confirmed_egm_count: `{len(positive_egm_df)}`",
        f"- final_decision: `{final_decision}`",
        "",
        "## Strict Criteria Applied",
        "",
        "- requested/resolved alpha match is recomputed from each run's `args.yml` metadata",
        "- alpha-sensitivity is checked from existing curve hashes across alpha within `(optimizer_scope, shared_lr, method)`",
        "- dominance fraction must be `>= 0.70`",
        "- clean eval AUC ratio vs SGD must be `>= 0.80`",
        "- improvement must not be final-point-only",
        "- `PPM_inner5` must not be identical to `EGM`",
        "",
        "## Confirmed PPM Candidates",
        "",
    ]
    if positive_ppm_df.empty:
        report_lines.append("- none")
    else:
        for _, row in positive_ppm_df.iterrows():
            report_lines.append(
                f"- scope=`{row['optimizer_scope']}`, alpha=`{row['alpha']}`, lr=`{row['shared_lr']}`:"
                f" PPM_over_SGD_fraction=`{row['PPM_over_SGD_fraction']:.3f}`,"
                f" improvement_pct=`{row['winner_improvement_pct']:.3f}`,"
                f" PPM_clean_auc_ratio=`{row['PPM_clean_auc_ratio_vs_SGD']:.3f}`"
            )
    report_lines.extend(["", "## Confirmed EGM Candidates", ""])
    if positive_egm_df.empty:
        report_lines.append("- none")
    else:
        for _, row in positive_egm_df.iterrows():
            report_lines.append(
                f"- scope=`{row['optimizer_scope']}`, alpha=`{row['alpha']}`, lr=`{row['shared_lr']}`:"
                f" EGM_over_SGD_fraction=`{row['EGM_over_SGD_fraction']:.3f}`,"
                f" improvement_pct=`{row['winner_improvement_pct']:.3f}`,"
                f" EGM_clean_auc_ratio=`{row['EGM_clean_auc_ratio_vs_SGD']:.3f}`"
            )
    report_lines.extend(
        [
            "",
            "## Near Misses",
            "",
        ]
    )
    for _, row in config_df[(config_df["confirmed_ppm_flag"] == 0) & (config_df["confirmed_egm_flag"] == 0)].head(6).iterrows():
        report_lines.append(
            f"- scope=`{row['optimizer_scope']}`, alpha=`{row['alpha']}`, lr=`{row['shared_lr']}`:"
            f" winner=`{row['winner']}`, improvement_pct=`{row['winner_improvement_pct']:.3f}`,"
            f" dominance=`{row['winner_dominance_fraction']:.3f}`,"
            f" clean_ratio=`{row['clean_auc_ratio_winner_vs_sgd']:.3f}`"
        )
    report_lines.extend(
        [
            "",
            "## Decision Labels",
            "",
            "- `ALPHA_REPORT_BUG_FIXED_NO_CONFIRMED_BASELINE`",
            "- `CONFIRMED_BASELINE_PPM`",
            "- `CONFIRMED_BASELINE_EGM`",
            "- `CONFIRMED_BASELINE_BOTH`",
        ]
    )
    (output_root / "baseline_search_after_alpha_fix_report_RECOMPUTED.md").write_text("\n".join(report_lines), encoding="utf-8")

    if best_curve is not None and not best_curve.empty:
        base.save_big_figure(best_curve, plots_dir / "stageA_after_alpha_fix_recomputed_top_configs.png", f"Recomputed Stage A Top Config: {best_key}")


if __name__ == "__main__":
    main()
