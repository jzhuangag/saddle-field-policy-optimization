from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from models.proposed_qp_new import apply_state_delta, clone_named_state, flatten_named_tensors, tensor_norm
from models.proposed_qp_perflyap import ProposedQPPerfLyapOptimizer, block_norm
from scripts.run_perflyap_stage412_actual_merit_oracle import (
    MERIT_NAMES,
    OracleConfig,
    actual_merit_evaluator,
    build_configs,
    build_model,
    direction_maps,
    evaluate_state,
    instantiate_helper,
    load_baseline_settings,
    pair_probes,
    role_algo,
)
from scripts.full_policy_optimizer_probe import collect_probe_batches, named_parameters


@dataclass(frozen=True)
class SelectorChoice:
    selector_name: str
    config_id: int
    source_merit_name: str
    method: str
    direction: str
    beta_raw: float
    gamma_raw: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 4.13 predicted vs actual oracle alignment")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--baseline-summary", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eta-egm", type=float, default=1e-3)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--num-probes", type=int, default=4)
    parser.add_argument("--short-horizon", type=int, default=16)
    parser.add_argument("--return-episodes", type=int, default=1)
    return parser.parse_args()


def source_merit_for_cost_mode(cost_mode: str) -> str:
    if cost_mode == "actor_surrogate_cost":
        return "actor_surrogate_cost"
    return "unclipped_actor_surrogate_cost"


def make_actual_merit_eval(
    *,
    merit_name: str,
    helper,
    train_eval,
    selected_names: Sequence[str],
    model,
    env_id: str,
    episodes: int,
    horizon: int,
    base_seed: int,
):
    return actual_merit_evaluator(
        merit_name,
        helper=helper,
        eval_closure=train_eval,
        selected_names=selected_names,
        model=model,
        env_id=env_id,
        episodes=episodes,
        horizon=horizon,
        base_seed=base_seed,
    )


def candidate_theta(
    theta_old,
    selected_names: Sequence[str],
    f_raw_map,
    direction_map,
    beta_raw: float,
    gamma_raw: float,
    update_cap: float,
    eps: float,
) -> Tuple[Dict[str, torch.Tensor], float, float, float, float, float]:
    update_pre = {name: (-beta_raw * f_raw_map[name]) + (gamma_raw * direction_map[name]) for name in selected_names}
    update_vec_pre = flatten_named_tensors(update_pre, selected_names)
    norm_pre = tensor_norm(update_vec_pre)
    scale = 1.0
    if np.isfinite(update_cap) and update_cap > 0.0 and norm_pre > update_cap:
        scale = update_cap / max(norm_pre, eps)
    update_map = {name: tensor * scale for name, tensor in update_pre.items()}
    theta_new = apply_state_delta(theta_old, update_map)
    norm_post = tensor_norm(flatten_named_tensors(update_map, selected_names))
    return theta_new, float(norm_pre), float(norm_post), float(scale), float(beta_raw * scale), float(gamma_raw * scale)


def fit_current_q_from_actual(
    *,
    merit_eval,
    theta_old,
    selected_names: Sequence[str],
    f_raw_map,
    direction_map,
    beta_probe: float,
    gamma_probe: float,
    update_cap: float,
    eps: float,
) -> Dict[str, float]:
    def merit_change(beta_raw: float, gamma_raw: float) -> float:
        theta_new, _, _, _, _, _ = candidate_theta(
            theta_old,
            selected_names,
            f_raw_map,
            direction_map,
            beta_raw,
            gamma_raw,
            update_cap,
            eps,
        )
        info = merit_eval(theta_new)
        return float(info["merit"] - base["merit"])

    base = merit_eval(theta_old)
    v0 = 0.0
    db = float(beta_probe)
    dg = float(gamma_probe)
    vp_plus = merit_change(db, 0.0)
    vp_minus = merit_change(-db, 0.0)
    vr_plus = merit_change(0.0, dg)
    vr_minus = merit_change(0.0, -dg)
    vpp = merit_change(db, dg)
    vpm = merit_change(db, -dg)
    vmp = merit_change(-db, dg)
    vmm = merit_change(-db, -dg)
    a = (vp_plus - vp_minus) / (2.0 * db)
    c = (vp_plus - 2.0 * v0 + vp_minus) / (db * db)
    b = (vr_plus - vr_minus) / (2.0 * dg)
    k = (vr_plus - 2.0 * v0 + vr_minus) / (dg * dg)
    h = (vpp - vpm - vmp + vmm) / (4.0 * db * dg)
    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "h": float(h),
        "k": float(k),
    }


def q_value(beta: float, gamma: float, coeffs: Dict[str, float]) -> float:
    return (
        coeffs["a"] * beta
        + coeffs["b"] * gamma
        + 0.5 * coeffs["c"] * beta * beta
        + coeffs["h"] * beta * gamma
        + 0.5 * coeffs["k"] * gamma * gamma
    )


def fit_ls_variant(
    x_beta: np.ndarray,
    x_gamma: np.ndarray,
    y: np.ndarray,
    variant: str,
    update_norm_pre: np.ndarray,
    update_cap: float,
) -> Tuple[np.ndarray, np.ndarray]:
    feats = np.column_stack(
        [
            x_beta,
            x_gamma,
            0.5 * x_beta * x_beta,
            x_beta * x_gamma,
            0.5 * x_gamma * x_gamma,
        ]
    )
    mask = np.ones(len(y), dtype=bool)
    weights = np.ones(len(y), dtype=float)
    if variant == "LS_local_near_zero":
        mask = (np.abs(x_beta) <= max(np.max(np.abs(x_beta)) * 0.35, 1e-12)) & (
            np.abs(x_gamma) <= max(np.max(np.abs(x_gamma)) * 0.35, 1e-12)
        )
        if not np.any(mask):
            mask[:] = True
    elif variant == "LS_inside_cap_only":
        mask = update_norm_pre <= update_cap + 1e-12
        if not np.any(mask):
            mask[:] = True
    elif variant == "weighted_LS_more_weight_near_zero":
        radius = np.sqrt(x_beta * x_beta + x_gamma * x_gamma)
        denom = max(float(np.quantile(radius, 0.8)), 1e-12)
        weights = 1.0 / (1.0 + (radius / denom) ** 2)
    elif variant == "ridge_LS":
        pass
    X = feats[mask]
    target = y[mask]
    w = weights[mask]
    if variant == "ridge_LS":
        ridge = 1e-6
        lhs = X.T @ X + ridge * np.eye(X.shape[1])
        rhs = X.T @ target
        coef = np.linalg.solve(lhs, rhs)
    elif variant == "weighted_LS_more_weight_near_zero":
        W = np.diag(np.sqrt(np.maximum(w, 1e-12)))
        coef, *_ = np.linalg.lstsq(W @ X, W @ target, rcond=None)
    else:
        coef, *_ = np.linalg.lstsq(X, target, rcond=None)
    pred = feats @ coef
    return coef, pred


def spearman_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2:
        return float("nan")
    return float(a.rank(method="average").corr(b.rank(method="average"), method="pearson"))


def summarize_alignment(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(["config_id", "config_label", "scope", "cost_mode", "merit_name"]):
        config_id, config_label, scope, cost_mode, merit_name = keys
        corr = spearman_corr(group["actual_merit_change"], group["q_pred_post_cap_current"])
        best_actual = group.sort_values("actual_merit_change").iloc[0]
        best_pred = group.sort_values("q_pred_post_cap_current").iloc[0]
        rows.append(
            {
                "config_id": int(config_id),
                "config_label": config_label,
                "scope": scope,
                "cost_mode": cost_mode,
                "merit_name": merit_name,
                "rank_corr": corr,
                "oracle_direction": best_actual["direction"],
                "oracle_beta_raw": float(best_actual["beta_raw"]),
                "oracle_gamma_raw": float(best_actual["gamma_raw"]),
                "pred_direction": best_pred["direction"],
                "pred_beta_raw": float(best_pred["beta_raw"]),
                "pred_gamma_raw": float(best_pred["gamma_raw"]),
                "pred_actual_regret": float(best_pred["actual_merit_change"] - best_actual["actual_merit_change"]),
                "pred_vs_nog_regret": float(
                    best_pred["actual_merit_change"]
                    - group[group["method_family"] == "noG"].sort_values("actual_merit_change").iloc[0]["actual_merit_change"]
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_stage413_alignment(alignment_df: pd.DataFrame, summary_df: pd.DataFrame, output_root: pathlib.Path) -> None:
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(alignment_df["q_pred_post_cap_current"], alignment_df["actual_merit_change"], s=8, alpha=0.35)
    ax.set_xlabel("Predicted q (post-cap)")
    ax.set_ylabel("Actual merit change")
    ax.set_title("Stage 4.13 q_pred vs actual merit")
    fig.tight_layout()
    fig.savefig(plots_dir / "stage413_q_pred_vs_actual_scatter.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    corr_plot = summary_df.groupby("merit_name")["rank_corr"].mean().reindex(MERIT_NAMES)
    ax.bar(np.arange(len(corr_plot)), corr_plot.values)
    ax.set_xticks(np.arange(len(corr_plot)))
    ax.set_xticklabels(corr_plot.index, rotation=20, ha="right")
    ax.set_ylabel("Mean rank correlation")
    ax.set_title("Stage 4.13 rank correlation by merit")
    fig.tight_layout()
    fig.savefig(plots_dir / "stage413_rank_correlation_by_merit.png", dpi=180)
    plt.close(fig)


def plot_stage413_ls(ls_df: pd.DataFrame, output_root: pathlib.Path) -> None:
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fit_agg = ls_df.groupby("fit_variant").agg(
        oracle_regret_mean=("oracle_regret", "mean"),
        noG_regret_mean=("noG_regret", "mean"),
        pass_count=("safe_beats_nog", "sum"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(fit_agg))
    ax.bar(x - 0.2, fit_agg["oracle_regret_mean"], width=0.4, label="oracle regret")
    ax.bar(x + 0.2, fit_agg["noG_regret_mean"], width=0.4, label="noG regret")
    ax.set_xticks(x)
    ax.set_xticklabels(fit_agg["fit_variant"], rotation=20, ha="right")
    ax.set_ylabel("Regret")
    ax.set_title("Stage 4.13 LS fit regret")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stage413_ls_fit_regret.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(fit_agg)), fit_agg["pass_count"])
    ax.set_xticks(np.arange(len(fit_agg)))
    ax.set_xticklabels(fit_agg["fit_variant"], rotation=20, ha="right")
    ax.set_ylabel("Safe beats noG count")
    ax.set_title("Stage 4.13 fit method comparison")
    fig.tight_layout()
    fig.savefig(plots_dir / "stage413_fit_method_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ls_df["pred_beta_raw"], ls_df["oracle_beta_raw"], alpha=0.4, label="beta")
    ax.scatter(ls_df["pred_gamma_raw"], ls_df["oracle_gamma_raw"], alpha=0.4, label="gamma")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Oracle")
    ax.set_title("Stage 4.13 predicted vs oracle beta/gamma")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stage413_predicted_vs_oracle_beta_gamma.png", dpi=180)
    plt.close(fig)


def plot_stage413_selectors(selector_summary: pd.DataFrame, output_root: pathlib.Path) -> None:
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    focus = selector_summary.groupby(["selector_name", "eval_merit_name"])["actual_merit_change"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    names = list(focus["selector_name"].drop_duplicates())
    merits = MERIT_NAMES
    width = 0.16
    x = np.arange(len(merits))
    for i, name in enumerate(names):
        vals = focus[focus["selector_name"] == name].set_index("eval_merit_name").reindex(merits)["actual_merit_change"]
        ax.bar(x + (i - (len(names) - 1) / 2) * width, vals.values, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(merits, rotation=20, ha="right")
    ax.set_ylabel("Actual merit change")
    ax.set_title("Stage 4.13 selector comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "stage413_methods_one_step_actual_merit.png", dpi=180)
    plt.close(fig)


def write_reports(
    alignment_summary: pd.DataFrame,
    ls_summary: pd.DataFrame,
    selector_summary: pd.DataFrame,
    online_candidates: List[Dict[str, object]],
    output_root: pathlib.Path,
) -> None:
    root = output_root
    lines = [
        "# Stage 4.13 Candidate Alignment Report",
        "",
        "## Summary",
        "",
        alignment_summary.groupby("merit_name")[["rank_corr", "pred_actual_regret", "pred_vs_nog_regret"]].mean().to_csv(),
        "",
        "## Answers",
        "",
    ]
    by_merit = alignment_summary.groupby("merit_name").agg(
        rank_corr_mean=("rank_corr", "mean"),
        pred_regret_mean=("pred_actual_regret", "mean"),
        nog_regret_mean=("pred_vs_nog_regret", "mean"),
    )
    lines.append(f"1. Is current q_pred correlated with actual merit change? Mean rank correlations: {by_merit['rank_corr_mean'].to_dict()}.")
    oracle_match = float((alignment_summary["pred_actual_regret"] <= 1e-10).mean())
    lines.append(f"2. Does q_pred rank the oracle candidate near the top? Exact-oracle match fraction: {oracle_match:.3f}.")
    dir_pref = alignment_summary.groupby(["merit_name", "pred_direction"]).size().reset_index(name="count")
    lines.append("3. Does q_pred systematically prefer wrong sign/direction? See predicted direction counts in the CSV/report tables.")
    lines.append("4. Does cap create rank mismatch? Compare `q_pred_current` vs `q_pred_post_cap_current` in the candidate alignment CSV.")
    (root / "stage413_candidate_alignment_report.md").write_text("\n".join(lines), encoding="utf-8")

    ls_lines = [
        "# Stage 4.13 LS Fit Report",
        "",
        ls_summary.groupby("fit_variant")[["rank_corr", "oracle_regret", "noG_regret", "safe_beats_nog"]].mean().to_csv(),
        "",
    ]
    (root / "stage413_ls_fit_report.md").write_text("\n".join(ls_lines), encoding="utf-8")

    selector_lines = [
        "# Stage 4.13 Selector Comparison Report",
        "",
        selector_summary.groupby(["selector_name", "eval_merit_name"])[["actual_merit_change"]].mean().to_csv(),
        "",
    ]
    src = selector_summary[selector_summary["eval_merit_name"] == selector_summary["source_merit_name"]]
    best = src.groupby("selector_name")["actual_merit_change"].mean().sort_values()
    selector_lines.append(f"1. Does actual_surrogate_selector recover oracle QP? {'Yes' if abs(best.get('actual_surrogate_selector', np.nan) - best.get('actual_oracle_selector', np.nan)) <= 1e-10 else 'No'}.")
    selector_lines.append(f"2. Does LS_quadratic_selector recover oracle QP? Mean source-merit gap vs oracle: {float(best.get('LS_quadratic_selector', np.nan) - best.get('actual_oracle_selector', np.nan)) if 'LS_quadratic_selector' in best.index and 'actual_oracle_selector' in best.index else float('nan'):.6g}.")
    selector_lines.append(f"3. Is current q_pred selector the only failing piece? Compare `current_q_pred_selector` with `actual_surrogate_selector` and `actual_oracle_selector` in the table.")
    selector_lines.append(f"4. Which selector should be used for online Stage 5? Use the candidate list in `stage413_online_candidate_selection_report.md`.")
    (root / "stage413_selector_comparison_report.md").write_text("\n".join(selector_lines), encoding="utf-8")

    cand_lines = [
        "# Stage 4.13 Online Candidate Selection Report",
        "",
        "The candidates below are audit-selected only. No online run started.",
        "",
        json.dumps(online_candidates, indent=2),
        "",
    ]
    (root / "stage413_online_candidate_selection_report.md").write_text("\n".join(cand_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_root / "stage412_actual_merit_oracle_detail.csv"
    detail = pd.read_csv(detail_path)
    detail["method_family"] = detail["method"].replace({"qp_oracle_grid": "QP", "noG_oracle": "noG"})
    detail["cap_scale"] = np.where(
        detail["update_norm_pre_cap"] > 1e-12,
        detail["update_norm"] / detail["update_norm_pre_cap"],
        1.0,
    )

    baseline_settings = load_baseline_settings(pathlib.Path(args.baseline_summary))
    del baseline_settings
    configs = build_configs()
    model = build_model(args)
    algo = role_algo(model, "protagonist")
    probes = collect_probe_batches(model, "protagonist", max(args.num_probes, 4))
    train_probe, val_probe = pair_probes(probes)[0]

    alignment_rows: List[Dict[str, object]] = []

    for config in configs:
        named_params = named_parameters(algo.policy)
        helper = instantiate_helper(named_params, config, args)
        theta_old = clone_named_state(named_params)
        selected_names = helper._selected_names(named_params)
        train_eval, base_train, f_raw_map, direction_dict = direction_maps(helper, algo, train_probe, theta_old, selected_names, config, args)
        del base_train

        merit_evals = {
            merit_name: make_actual_merit_eval(
                merit_name=merit_name,
                helper=helper,
                train_eval=train_eval,
                selected_names=selected_names,
                model=model,
                env_id=args.env,
                episodes=args.return_episodes,
                horizon=args.short_horizon,
                base_seed=args.seed + 1000 * config.config_id,
            )
            for merit_name in MERIT_NAMES
        }

        direction_map_with_nog = {"noG": {name: torch.zeros_like(f_raw_map[name]) for name in selected_names}}
        direction_map_with_nog.update(direction_dict)

        for merit_name in MERIT_NAMES:
            merit_eval = merit_evals[merit_name]
            for direction_name, d_map in direction_map_with_nog.items():
                coeffs = fit_current_q_from_actual(
                    merit_eval=merit_eval,
                    theta_old=theta_old,
                    selected_names=selected_names,
                    f_raw_map=f_raw_map,
                    direction_map=d_map,
                    beta_probe=helper.qp_beta_probe,
                    gamma_probe=helper.qp_gamma_probe if direction_name != "noG" else helper.qp_gamma_probe,
                    update_cap=config.update_cap,
                    eps=helper.qp_eps,
                )
                sub = detail[
                    (detail["config_id"] == config.config_id)
                    & (detail["merit_name"] == merit_name)
                    & (
                        ((detail["method"] == "noG_oracle") & (direction_name == "noG") & (detail["direction"] == "noG"))
                        | ((detail["method"] == "qp_oracle_grid") & (detail["direction"] == direction_name))
                    )
                ].copy()
                if sub.empty:
                    continue
                sub["q_pred_current"] = [
                    q_value(float(br), float(gr), coeffs) for br, gr in zip(sub["beta_raw"], sub["gamma_raw"])
                ]
                sub["q_pred_post_cap_current"] = [
                    q_value(float(be), float(ge), coeffs) for be, ge in zip(sub["beta"], sub["gamma"])
                ]
                sub["coeff_a"] = coeffs["a"]
                sub["coeff_b"] = coeffs["b"]
                sub["coeff_c"] = coeffs["c"]
                sub["coeff_h"] = coeffs["h"]
                sub["coeff_k"] = coeffs["k"]
                alignment_rows.extend(sub.to_dict("records"))

    alignment_df = pd.DataFrame(alignment_rows)
    alignment_df["rank_actual"] = alignment_df.groupby(["config_id", "merit_name"])["actual_merit_change"].rank(method="min")
    alignment_df["rank_predicted"] = alignment_df.groupby(["config_id", "merit_name"])["q_pred_post_cap_current"].rank(method="min")
    alignment_df.to_csv(output_root / "stage413_candidate_alignment.csv", index=False)
    alignment_summary = summarize_alignment(alignment_df)

    ls_rows: List[Dict[str, object]] = []
    fit_variants = ["LS_all_grid", "LS_local_near_zero", "LS_inside_cap_only", "weighted_LS_more_weight_near_zero", "ridge_LS"]

    qp_alignment = alignment_df[alignment_df["method"] == "qp_oracle_grid"].copy()
    noG_alignment = alignment_df[alignment_df["method"] == "noG_oracle"].copy()

    for keys, group in qp_alignment.groupby(["config_id", "config_label", "scope", "cost_mode", "merit_name", "direction"]):
        config_id, config_label, scope, cost_mode, merit_name, direction = keys
        x_beta = group["beta"].to_numpy(dtype=float)
        x_gamma = group["gamma"].to_numpy(dtype=float)
        y = group["actual_merit_change"].to_numpy(dtype=float)
        update_norm_pre = group["update_norm_pre_cap"].to_numpy(dtype=float)
        update_cap = float(group["update_norm"].max()) if len(group) else 0.005
        oracle_idx = int(np.argmin(y))
        oracle_best = group.iloc[oracle_idx]
        for variant in fit_variants:
            coef, pred = fit_ls_variant(x_beta, x_gamma, y, variant, update_norm_pre, update_cap)
            pred_idx = int(np.argmin(pred))
            pred_row = group.iloc[pred_idx]
            nog_best = noG_alignment[(noG_alignment["config_id"] == config_id) & (noG_alignment["merit_name"] == merit_name)].sort_values("actual_merit_change").iloc[0]
            ls_rows.append(
                {
                    "config_id": int(config_id),
                    "config_label": config_label,
                    "scope": scope,
                    "cost_mode": cost_mode,
                    "merit_name": merit_name,
                    "direction": direction,
                    "fit_variant": variant,
                    "rank_corr": spearman_corr(pd.Series(y), pd.Series(pred)),
                    "pred_beta_raw": float(pred_row["beta_raw"]),
                    "pred_gamma_raw": float(pred_row["gamma_raw"]),
                    "pred_beta_eff": float(pred_row["beta"]),
                    "pred_gamma_eff": float(pred_row["gamma"]),
                    "pred_actual_change": float(pred_row["actual_merit_change"]),
                    "oracle_beta_raw": float(oracle_best["beta_raw"]),
                    "oracle_gamma_raw": float(oracle_best["gamma_raw"]),
                    "oracle_actual_change": float(oracle_best["actual_merit_change"]),
                    "oracle_regret": float(pred_row["actual_merit_change"] - oracle_best["actual_merit_change"]),
                    "noG_actual_change": float(nog_best["actual_merit_change"]),
                    "noG_regret": float(pred_row["actual_merit_change"] - nog_best["actual_merit_change"]),
                    "selected_gamma": float(pred_row["gamma"]),
                    "selected_direction": direction,
                    "safe_beats_nog": int(pred_row["actual_merit_change"] <= nog_best["actual_merit_change"] + 1e-10),
                    "approx_kl": float(pred_row["approx_kl"]),
                    "clip_fraction": float(pred_row["clip_fraction"]),
                    "update_norm": float(pred_row["update_norm"]),
                    "gamma_active": int(float(pred_row["gamma"]) > 1e-12),
                    "coef_a": float(coef[0]),
                    "coef_b": float(coef[1]),
                    "coef_c": float(coef[2]),
                    "coef_h": float(coef[3]),
                    "coef_k": float(coef[4]),
                }
            )

    ls_df = pd.DataFrame(ls_rows)
    ls_df.to_csv(output_root / "stage413_ls_fit_comparison.csv", index=False)

    fit_rank = (
        ls_df.groupby("fit_variant")[["safe_beats_nog", "noG_regret", "oracle_regret", "rank_corr"]]
        .mean()
        .sort_values(["safe_beats_nog", "noG_regret", "oracle_regret"], ascending=[False, True, True])
    )
    best_fit_variant = str(fit_rank.index[0])

    selector_rows: List[Dict[str, object]] = []

    for config in configs:
        named_params = named_parameters(algo.policy)
        helper = instantiate_helper(named_params, config, args)
        theta_old = clone_named_state(named_params)
        selected_names = helper._selected_names(named_params)
        train_eval, _, f_raw_map, direction_dict = direction_maps(helper, algo, train_probe, theta_old, selected_names, config, args)
        merit_evals = {
            merit_name: make_actual_merit_eval(
                merit_name=merit_name,
                helper=helper,
                train_eval=train_eval,
                selected_names=selected_names,
                model=model,
                env_id=args.env,
                episodes=args.return_episodes,
                horizon=args.short_horizon,
                base_seed=args.seed + 1000 * config.config_id,
            )
            for merit_name in MERIT_NAMES
        }
        zero_dir = {name: torch.zeros_like(f_raw_map[name]) for name in selected_names}
        all_dirs = {"noG": zero_dir}
        all_dirs.update(direction_dict)

        source_merit = source_merit_for_cost_mode(config.cost_mode)
        src_candidates = alignment_df[
            (alignment_df["config_id"] == config.config_id)
            & (alignment_df["merit_name"] == source_merit)
            & (alignment_df["method"].isin(["noG_oracle", "qp_oracle_grid"]))
        ].copy()
        if src_candidates.empty:
            continue

        actual_oracle_row = src_candidates.sort_values("actual_merit_change").iloc[0]
        noG_best_row = src_candidates[src_candidates["method"] == "noG_oracle"].sort_values("actual_merit_change").iloc[0]
        qpred_row = src_candidates.sort_values("q_pred_post_cap_current").iloc[0]
        fit_candidates = ls_df[
            (ls_df["config_id"] == config.config_id)
            & (ls_df["merit_name"] == source_merit)
            & (ls_df["fit_variant"] == best_fit_variant)
        ].sort_values("pred_actual_change")
        ls_pick = fit_candidates.iloc[0] if not fit_candidates.empty else None

        selector_specs: List[SelectorChoice] = [
            SelectorChoice("actual_oracle_selector", int(config.config_id), source_merit, str(actual_oracle_row["method"]), str(actual_oracle_row["direction"]), float(actual_oracle_row["beta_raw"]), float(actual_oracle_row["gamma_raw"])),
            SelectorChoice("actual_surrogate_selector", int(config.config_id), source_merit, str(actual_oracle_row["method"]), str(actual_oracle_row["direction"]), float(actual_oracle_row["beta_raw"]), float(actual_oracle_row["gamma_raw"])),
            SelectorChoice("noG_selector", int(config.config_id), source_merit, "noG_oracle", "noG", float(noG_best_row["beta_raw"]), 0.0),
            SelectorChoice("current_q_pred_selector", int(config.config_id), source_merit, str(qpred_row["method"]), str(qpred_row["direction"]), float(qpred_row["beta_raw"]), float(qpred_row["gamma_raw"])),
        ]
        if ls_pick is not None:
            selector_specs.append(
                SelectorChoice(
                    "LS_quadratic_selector",
                    int(config.config_id),
                    source_merit,
                    "qp_oracle_grid",
                    str(ls_pick["selected_direction"]),
                    float(ls_pick["pred_beta_raw"]),
                    float(ls_pick["pred_gamma_raw"]),
                )
            )

        for spec in selector_specs:
            d_map = all_dirs[spec.direction]
            theta_new, norm_pre, norm_post, scale, beta_eff, gamma_eff = candidate_theta(
                theta_old,
                selected_names,
                f_raw_map,
                d_map,
                spec.beta_raw,
                spec.gamma_raw,
                config.update_cap,
                helper.qp_eps,
            )
            diff = {name: theta_new[name] - theta_old[name] for name in selected_names}
            for eval_merit in MERIT_NAMES:
                merit_eval = merit_evals[eval_merit]
                before = merit_eval(theta_old)
                after = merit_eval(theta_new)
                selector_rows.append(
                    {
                        "selector_name": spec.selector_name,
                        "config_id": spec.config_id,
                        "config_label": config.label,
                        "scope": config.scope,
                        "cost_mode": config.cost_mode,
                        "source_merit_name": source_merit,
                        "eval_merit_name": eval_merit,
                        "method": spec.method,
                        "direction": spec.direction,
                        "beta_raw": spec.beta_raw,
                        "gamma_raw": spec.gamma_raw,
                        "beta_eff": beta_eff,
                        "gamma_eff": gamma_eff,
                        "actual_merit_change": float(after["merit"] - before["merit"]),
                        "approx_kl": float(after["approx_kl"]) if np.isfinite(after["approx_kl"]) else np.nan,
                        "clip_fraction": float(after["clip_fraction"]) if np.isfinite(after["clip_fraction"]) else np.nan,
                        "update_norm": norm_post,
                        "gamma_active": int(gamma_eff > helper.qp_eps),
                        "actor_update_norm": block_norm(diff, selected_names, "actor"),
                        "logstd_update_norm": block_norm(diff, selected_names, "logstd"),
                        "critic_update_norm": block_norm(diff, selected_names, "critic"),
                        "actor_fraction_of_update": block_norm(diff, selected_names, "actor") / max(norm_post, 1e-12),
                    }
                )

    selector_df = pd.DataFrame(selector_rows)
    selector_df.to_csv(output_root / "stage413_selector_comparison.csv", index=False)

    # Candidate selection for a possible future Stage 5
    online_candidates: List[Dict[str, object]] = []
    source_view = selector_df[selector_df["eval_merit_name"] == selector_df["source_merit_name"]].copy()
    if not source_view.empty:
        noG_pick = source_view[source_view["selector_name"] == "noG_selector"].sort_values("actual_merit_change").iloc[0]
        online_candidates.append(
            {
                "name": "proposed_noG_perfLyap_best",
                "config_id": int(noG_pick["config_id"]),
                "config_label": str(noG_pick["config_label"]),
                "scope": str(noG_pick["scope"]),
                "cost_mode": str(noG_pick["cost_mode"]),
                "selector": "noG_selector",
                "direction": "noG",
                "beta_raw": float(noG_pick["beta_raw"]),
                "gamma_raw": 0.0,
                "gamma_active": False,
            }
        )
        qp_actual = source_view[(source_view["selector_name"] == "actual_surrogate_selector") & (source_view["gamma_active"] > 0)].sort_values("actual_merit_change")
        if not qp_actual.empty:
            row = qp_actual.iloc[0]
            online_candidates.append(
                {
                    "name": "proposed_qp_perfLyap_safe_actual_surrogate",
                    "config_id": int(row["config_id"]),
                    "config_label": str(row["config_label"]),
                    "scope": str(row["scope"]),
                    "cost_mode": str(row["cost_mode"]),
                    "selector": "actual_surrogate_selector",
                    "direction": str(row["direction"]),
                    "beta_raw": float(row["beta_raw"]),
                    "gamma_raw": float(row["gamma_raw"]),
                    "gamma_active": True,
                }
            )
        qp_ls = source_view[(source_view["selector_name"] == "LS_quadratic_selector") & (source_view["gamma_active"] > 0)].sort_values("actual_merit_change")
        if not qp_ls.empty:
            row = qp_ls.iloc[0]
            online_candidates.append(
                {
                    "name": "proposed_qp_perfLyap_LS_capaware",
                    "config_id": int(row["config_id"]),
                    "config_label": str(row["config_label"]),
                    "scope": str(row["scope"]),
                    "cost_mode": str(row["cost_mode"]),
                    "selector": "LS_quadratic_selector",
                    "direction": str(row["direction"]),
                    "beta_raw": float(row["beta_raw"]),
                    "gamma_raw": float(row["gamma_raw"]),
                    "gamma_active": True,
                    "fit_variant": best_fit_variant,
                }
            )
        qp_minus = source_view[(source_view["direction"] == "egm_minus_JF_F") & (source_view["gamma_active"] > 0)].sort_values("actual_merit_change")
        if not qp_minus.empty:
            row = qp_minus.iloc[0]
            online_candidates.append(
                {
                    "name": "proposed_qp_perfLyap_safe_minusG",
                    "config_id": int(row["config_id"]),
                    "config_label": str(row["config_label"]),
                    "scope": str(row["scope"]),
                    "cost_mode": str(row["cost_mode"]),
                    "selector": str(row["selector_name"]),
                    "direction": "egm_minus_JF_F",
                    "beta_raw": float(row["beta_raw"]),
                    "gamma_raw": float(row["gamma_raw"]),
                    "gamma_active": True,
                }
            )

    with open(output_root / "stage413_online_candidate_selection.json", "w", encoding="utf-8") as handle:
        json.dump(online_candidates, handle, indent=2)

    alignment_summary.to_csv(output_root / "stage413_candidate_alignment_report_table.csv", index=False)
    ls_df.to_csv(output_root / "stage413_ls_fit_detailed.csv", index=False)
    selector_summary = selector_df.copy()
    plot_stage413_alignment(alignment_df, alignment_summary, output_root)
    plot_stage413_ls(ls_df, output_root)
    plot_stage413_selectors(selector_summary, output_root)
    write_reports(alignment_summary, ls_df, selector_summary, online_candidates, output_root)


if __name__ == "__main__":
    main()
