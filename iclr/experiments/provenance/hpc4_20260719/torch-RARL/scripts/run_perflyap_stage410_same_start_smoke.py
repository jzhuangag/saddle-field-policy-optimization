from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from models.proposed_qp_new import apply_state_delta, flatten_named_tensors, named_difference, tensor_norm
from models.proposed_qp_perflyap import (
    ProposedQPPerfLyapOptimizer,
    block_norm,
    classify_perf_block,
)
from scripts.full_policy_optimizer_probe import (
    clone_state,
    collect_probe_batches,
    compute_loss_and_grads,
    egm_update,
    named_parameters,
    ppm_update,
    restore_state,
    sgd_update,
)
from scripts.run_perflyap_stage48_perf_audit import (
    build_eval_context,
    build_extended_eval_closure,
    solve_direction_mode,
)
from scripts.run_perflyap_stage49_extrapolation_audit import extrapolated_update
from scripts.run_perflyap_stage4_preflight import build_model, pair_probes, role_algo


@dataclass(frozen=True)
class SmokeConfig:
    config_id: int
    scope: str
    cost_mode: str
    beta_max: float = 3e-2
    gamma_max: float = 3e-5
    update_cap: float = 0.005
    fd_eps: float = 1e-3
    beta_probe: float = 1e-3
    gamma_probe: float = 1e-6
    ridge: float = 1e-8
    rho: float = 1e-8

    @property
    def label(self) -> str:
        return f"{self.scope}_{self.cost_mode}_b{self.beta_max:g}_g{self.gamma_max:g}_cap{self.update_cap:g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 4.10 same-start performance-only smoke audit")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--baseline-summary", type=str, required=True)
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-probes-per-role", type=int, default=4)
    parser.add_argument("--eta-egm", type=float, default=1e-3)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    return parser.parse_args()


def build_configs() -> List[SmokeConfig]:
    configs: List[SmokeConfig] = []
    config_id = 0
    for scope in ["actor_mean_only", "actor_game", "actor_mean_heavy"]:
        for cost_mode in ["actor_surrogate_cost", "unclipped_actor_surrogate_cost"]:
            configs.append(SmokeConfig(config_id=config_id, scope=scope, cost_mode=cost_mode))
            config_id += 1
    return configs


def load_baseline_settings(path: pathlib.Path) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(path)
    settings: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        settings[str(row["method"])] = {
            "lr": float(row["protagonist_lr"]),
            "max_grad_norm": float(row["protagonist_max_grad_norm"]),
            "vf_coef": float(row["protagonist_vf_coef"]),
            "ppm_inner_steps": int(row["ppm_inner_steps"]) if int(row["ppm_inner_steps"]) > 0 else 0,
        }
    return settings


def instantiate_helper(named_params, config: SmokeConfig, args: argparse.Namespace, role: str):
    return ProposedQPPerfLyapOptimizer(
        [param for _, param in named_params],
        lr=1.0,
        perflyap_scope=config.scope,
        lambda_N=0.0,
        lambda_P=1.0,
        lambda_critic=0.0,
        logstd_weight=0.0,
        qp_fd_eps=config.fd_eps,
        qp_beta_probe=config.beta_probe,
        qp_gamma_probe=config.gamma_probe,
        qp_ridge=config.ridge,
        qp_rho=config.rho,
        qp_beta_max=config.beta_max,
        qp_gamma_max=config.gamma_max,
        qp_max_update_norm=config.update_cap,
        qp_eps=1e-8,
        eta_egm_reference=args.eta_egm,
        g_sign_mode="auto_actual_preflight",
        role=role,
        diagnostics_csv_path=None,
    )


def adam_update(theta_old: Dict[str, torch.Tensor], grads_old: Dict[str, torch.Tensor], lr: float, *, beta1: float, beta2: float, eps: float) -> Dict[str, torch.Tensor]:
    del beta1, beta2  # first Adam step bias-corrects back to the raw gradient
    out: Dict[str, torch.Tensor] = {}
    for name, grad in grads_old.items():
        denom = grad.abs() + eps
        out[name] = theta_old[name] - lr * grad / denom
    return out


def evaluate_state(helper, eval_closure, theta_state, selected_names: Sequence[str]) -> Dict[str, object]:
    return helper._evaluate_state(
        eval_closure=eval_closure,
        theta_state=theta_state,
        selected_names=selected_names,
        backward=True,
        norm_scale=1.0,
        perf_scale=1.0,
    )


def update_row(
    *,
    helper,
    named_params,
    theta_old,
    theta_new,
    selected_names: Sequence[str],
    train_eval,
    val_eval,
    base_train,
    base_val,
    config: SmokeConfig,
    role: str,
    train_probe_id: int,
    val_probe_id: int,
    method: str,
    candidate_type: str,
    beta_eff: float | None,
    gamma_eff: float | None,
    selected_sign: str,
    source_solver_case: str,
    source_direction: str,
    update_norm_pre_cap: float | None,
    update_norm_post_cap: float,
    q_pred_post_cap: float | None,
) -> Dict[str, object]:
    train_after = evaluate_state(helper, train_eval, theta_new, selected_names)
    val_after = evaluate_state(helper, val_eval, theta_new, selected_names)
    restore_state(named_params, theta_old)

    all_names = [name for name, _ in named_params]
    diff_all = named_difference(theta_new, theta_old, all_names)
    total_update_norm = max(tensor_norm(flatten_named_tensors(diff_all, all_names)), helper.qp_eps)
    actor_update_norm = block_norm(diff_all, all_names, "actor")
    logstd_update_norm = block_norm(diff_all, all_names, "logstd")
    critic_update_norm = block_norm(diff_all, all_names, "critic")

    return {
        "config_id": config.config_id,
        "config_label": config.label,
        "scope": config.scope,
        "cost_mode": config.cost_mode,
        "role": role,
        "train_probe_id": train_probe_id,
        "val_probe_id": val_probe_id,
        "method": method,
        "candidate_type": candidate_type,
        "selected_sign": selected_sign,
        "source_solver_case": source_solver_case,
        "source_direction": source_direction,
        "beta_eff": float(beta_eff) if beta_eff is not None else np.nan,
        "gamma_eff": float(gamma_eff) if gamma_eff is not None else np.nan,
        "q_pred_post_cap": float(q_pred_post_cap) if q_pred_post_cap is not None else np.nan,
        "update_norm_pre_cap": float(update_norm_pre_cap) if update_norm_pre_cap is not None else np.nan,
        "update_norm_post_cap": float(update_norm_post_cap),
        "train_merit_before": float(base_train["V_merit"]),
        "train_merit_after": float(train_after["V_merit"]),
        "train_actual_merit_change": float(train_after["V_merit"] - base_train["V_merit"]),
        "train_C_before": float(base_train["C"]),
        "train_C_after": float(train_after["C"]),
        "train_actual_C_change": float(train_after["C"] - base_train["C"]),
        "train_policy_before": float(base_train["policy_component"]),
        "train_policy_after": float(train_after["policy_component"]),
        "train_approx_kl": float(train_after["approx_kl"]),
        "train_clip_fraction": float(train_after["clip_fraction"]),
        "val_merit_before": float(base_val["V_merit"]),
        "val_merit_after": float(val_after["V_merit"]),
        "val_actual_merit_change": float(val_after["V_merit"] - base_val["V_merit"]),
        "val_C_before": float(base_val["C"]),
        "val_C_after": float(val_after["C"]),
        "val_actual_C_change": float(val_after["C"] - base_val["C"]),
        "val_approx_kl": float(val_after["approx_kl"]),
        "val_clip_fraction": float(val_after["clip_fraction"]),
        "actor_update_norm": actor_update_norm,
        "logstd_update_norm": logstd_update_norm,
        "critic_update_norm": critic_update_norm,
        "actor_fraction_of_update": actor_update_norm / total_update_norm,
        "logstd_fraction_of_update": logstd_update_norm / total_update_norm,
        "critic_fraction_of_update": critic_update_norm / total_update_norm,
    }


def proposed_rows_for_probe(*, algo, role: str, train_probe, val_probe, config: SmokeConfig, args: argparse.Namespace) -> List[Dict[str, object]]:
    named_params = named_parameters(algo.policy)
    helper = instantiate_helper(named_params, config, args, role)
    train_eval = build_extended_eval_closure(build_eval_context(algo, train_probe.rollout_data), config.cost_mode)
    val_eval = build_extended_eval_closure(build_eval_context(algo, val_probe.rollout_data), config.cost_mode)
    theta_old = clone_state(named_params)
    selected_names = helper._selected_names(named_params)

    base_train = evaluate_state(helper, train_eval, theta_old, selected_names)
    base_val = evaluate_state(helper, val_eval, theta_old, selected_names)
    f_raw_map = {name: base_train["grads_selected"][name].detach().clone() for name in selected_names}
    g_plus_map, _ = helper._compute_g_raw(
        theta_old=theta_old,
        eval_closure=train_eval,
        selected_names=selected_names,
        f_raw_map=f_raw_map,
        norm_scale=1.0,
        perf_scale=1.0,
    )
    g_minus_map = {name: -g_plus_map[name] for name in selected_names}

    _, no_g_solution, plus_solution, _, _ = solve_direction_mode(
        helper,
        theta_old,
        train_eval,
        named_params,
        selected_names,
        base_train,
        1.0,
        1.0,
        f_raw_map,
        "egm_plus",
        g_plus_map,
        args.eta_egm,
    )
    _, _, minus_solution, _, _ = solve_direction_mode(
        helper,
        theta_old,
        train_eval,
        named_params,
        selected_names,
        base_train,
        1.0,
        1.0,
        f_raw_map,
        "egm_minus",
        g_minus_map,
        args.eta_egm,
    )

    no_g_theta = apply_state_delta(theta_old, no_g_solution["update_map"])
    rows = [
        update_row(
            helper=helper,
            named_params=named_params,
            theta_old=theta_old,
            theta_new=no_g_theta,
            selected_names=selected_names,
            train_eval=train_eval,
            val_eval=val_eval,
            base_train=base_train,
            base_val=base_val,
            config=config,
            role=role,
            train_probe_id=int(train_probe.probe_idx),
            val_probe_id=int(val_probe.probe_idx),
            method="proposed_noG_perfLyap_extrap_smoke",
            candidate_type="noG",
            beta_eff=float(no_g_solution["beta_eff"]),
            gamma_eff=0.0,
            selected_sign="none",
            source_solver_case=str(no_g_solution.get("selected_case", "")),
            source_direction="noG",
            update_norm_pre_cap=float(no_g_solution["update_norm_pre_cap"]),
            update_norm_post_cap=float(no_g_solution["update_norm_post_cap"]),
            q_pred_post_cap=float(no_g_solution["q_pred_post_cap"]),
        )
    ]

    extrap_specs = [
        ("plus", plus_solution, "minus_inside"),
        ("minus", minus_solution, "plus_inside"),
    ]
    cand_rows: List[Dict[str, object]] = []
    for sign_name, source_solution, shift_sign in extrap_specs:
        update_map, norm_pre, norm_post, beta_eff, gamma_eff, cap_active = extrapolated_update(
            helper=helper,
            named_params=named_params,
            theta_old=theta_old,
            selected_names=selected_names,
            train_eval=train_eval,
            base_eval=base_train,
            norm_scale=1.0,
            perf_scale=1.0,
            beta_raw=float(source_solution["beta_raw"]),
            gamma_raw=float(source_solution["gamma_raw"]),
            shift_sign=shift_sign,
        )
        theta_new = apply_state_delta(theta_old, update_map)
        row = update_row(
            helper=helper,
            named_params=named_params,
            theta_old=theta_old,
            theta_new=theta_new,
            selected_names=selected_names,
            train_eval=train_eval,
            val_eval=val_eval,
            base_train=base_train,
            base_val=base_val,
            config=config,
            role=role,
            train_probe_id=int(train_probe.probe_idx),
            val_probe_id=int(val_probe.probe_idx),
            method="proposed_qp_perfLyap_extrap_autoactual_smoke",
            candidate_type=f"extrap_{sign_name}",
            beta_eff=float(beta_eff),
            gamma_eff=float(gamma_eff),
            selected_sign=sign_name,
            source_solver_case="extrapolated_field",
            source_direction=f"linear_{sign_name}",
            update_norm_pre_cap=float(norm_pre),
            update_norm_post_cap=float(norm_post),
            q_pred_post_cap=float(source_solution["q_pred_post_cap"]),
        )
        row["cap_active"] = int(cap_active)
        cand_rows.append(row)

    best_row = min(cand_rows, key=lambda item: item["train_actual_merit_change"])
    for row in cand_rows:
        row["auto_selected"] = int(row["selected_sign"] == best_row["selected_sign"])
        rows.append(row)

    return rows


def baseline_rows_for_probe(*, algo, role: str, train_probe, val_probe, config: SmokeConfig, args: argparse.Namespace, baseline_settings: Dict[str, Dict[str, float]]) -> List[Dict[str, object]]:
    named_params = named_parameters(algo.policy)
    helper = instantiate_helper(named_params, config, args, role)
    train_eval = build_extended_eval_closure(build_eval_context(algo, train_probe.rollout_data), config.cost_mode)
    val_eval = build_extended_eval_closure(build_eval_context(algo, val_probe.rollout_data), config.cost_mode)
    theta_old = clone_state(named_params)
    selected_names = helper._selected_names(named_params)
    base_train = evaluate_state(helper, train_eval, theta_old, selected_names)
    base_val = evaluate_state(helper, val_eval, theta_old, selected_names)

    rows: List[Dict[str, object]] = []
    all_names = [name for name, _ in named_params]

    for method in ["sgd", "egm", "ppm", "adam"]:
        setting = baseline_settings[method]
        lr = float(setting["lr"])
        max_grad_norm = float(setting["max_grad_norm"])
        vf_coef = float(setting["vf_coef"])
        old_eval = compute_loss_and_grads(
            algo,
            train_probe.rollout_data,
            max_grad_norm=max_grad_norm,
            vf_coef=vf_coef,
            ent_coef=args.ent_coef,
        )
        if method == "sgd":
            theta_new = sgd_update(theta_old, old_eval["grads"], lr)
        elif method == "egm":
            _, _, _, theta_new = egm_update(
                algo,
                train_probe.rollout_data,
                theta_old,
                lr,
                max_grad_norm=max_grad_norm,
                vf_coef=vf_coef,
                ent_coef=args.ent_coef,
            )
        elif method == "ppm":
            inner_steps = max(int(setting["ppm_inner_steps"]), 1)
            _, theta_new, *_ = ppm_update(
                algo,
                train_probe.rollout_data,
                theta_old,
                lr,
                inner_steps=inner_steps,
                max_grad_norm=max_grad_norm,
                vf_coef=vf_coef,
                ent_coef=args.ent_coef,
            )
        else:
            optimizer_group = algo.policy.optimizer.param_groups[0]
            beta1, beta2 = optimizer_group.get("betas", (0.9, 0.999))
            eps = float(optimizer_group.get("eps", 1e-8))
            theta_new = adam_update(theta_old, old_eval["grads"], lr, beta1=beta1, beta2=beta2, eps=eps)

        rows.append(
            update_row(
                helper=helper,
                named_params=named_params,
                theta_old=theta_old,
                theta_new=theta_new,
                selected_names=selected_names,
                train_eval=train_eval,
                val_eval=val_eval,
                base_train=base_train,
                base_val=base_val,
                config=config,
                role=role,
                train_probe_id=int(train_probe.probe_idx),
                val_probe_id=int(val_probe.probe_idx),
                method=method,
                candidate_type=method,
                beta_eff=np.nan,
                gamma_eff=np.nan,
                selected_sign="none",
                source_solver_case=method,
                source_direction=method,
                update_norm_pre_cap=np.nan,
                update_norm_post_cap=tensor_norm(flatten_named_tensors(named_difference(theta_new, theta_old, all_names), all_names)),
                q_pred_post_cap=np.nan,
            )
        )
    restore_state(named_params, theta_old)
    return rows


def summarize(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    group_cols = ["config_id", "config_label", "scope", "cost_mode", "method"]
    for keys, group in detail_df.groupby(group_cols):
        config_id, config_label, scope, cost_mode, method = keys
        summary = {
            "config_id": int(config_id),
            "config_label": config_label,
            "scope": scope,
            "cost_mode": cost_mode,
            "method": method,
            "rows": int(len(group)),
            "train_actual_merit_change_mean": float(group["train_actual_merit_change"].mean()),
            "train_actual_C_change_mean": float(group["train_actual_C_change"].mean()),
            "val_actual_merit_change_mean": float(group["val_actual_merit_change"].mean()),
            "val_actual_C_change_mean": float(group["val_actual_C_change"].mean()),
            "train_approx_kl_mean": float(group["train_approx_kl"].mean()),
            "train_approx_kl_max": float(group["train_approx_kl"].max()),
            "train_clip_fraction_mean": float(group["train_clip_fraction"].mean()),
            "train_clip_fraction_max": float(group["train_clip_fraction"].max()),
            "update_norm_post_cap_mean": float(group["update_norm_post_cap"].mean()),
            "actor_fraction_of_update_mean": float(group["actor_fraction_of_update"].mean()),
        }
        if "auto_selected" in group.columns:
            summary["auto_selected_count"] = int(pd.to_numeric(group["auto_selected"], errors="coerce").fillna(0).sum())
        rows.append(summary)
    return pd.DataFrame(rows)


def plot_results(detail_df: pd.DataFrame, summary_df: pd.DataFrame, output_root: pathlib.Path) -> None:
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    order = [
        "adam",
        "sgd",
        "egm",
        "ppm",
        "proposed_noG_perfLyap_extrap_smoke",
        "proposed_qp_perfLyap_extrap_autoactual_smoke",
    ]

    best_configs = (
        summary_df[summary_df["method"] == "proposed_qp_perfLyap_extrap_autoactual_smoke"]
        .sort_values(["train_actual_merit_change_mean", "train_actual_C_change_mean"])
        .head(3)["config_id"]
        .tolist()
    )
    plot_df = detail_df[detail_df["config_id"].isin(best_configs)].copy()
    plot_df["method"] = pd.Categorical(plot_df["method"], categories=order, ordered=True)

    agg = (
        plot_df.groupby("method")[["train_actual_merit_change", "train_actual_C_change"]]
        .mean()
        .reindex(order)
        .dropna(how="all")
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(agg)), agg["train_actual_merit_change"])
    ax.set_xticks(np.arange(len(agg)))
    ax.set_xticklabels(list(agg.index), rotation=35, ha="right")
    ax.set_ylabel("Train merit change (lower is better)")
    ax.set_title("Same-start one-step performance-only merit smoke test")
    fig.tight_layout()
    fig.savefig(plots_dir / "stage410_same_start_merit.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(agg)), agg["train_actual_C_change"])
    ax.set_xticks(np.arange(len(agg)))
    ax.set_xticklabels(list(agg.index), rotation=35, ha="right")
    ax.set_ylabel("Train cost change (lower is better)")
    ax.set_title("Same-start one-step performance cost smoke test")
    fig.tight_layout()
    fig.savefig(plots_dir / "stage410_same_start_cost.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for method, group in agg.reset_index().groupby("method"):
        ax.scatter(group["train_actual_merit_change"], group["train_actual_C_change"], s=70)
        for _, row in group.iterrows():
            ax.annotate(method, (row["train_actual_merit_change"], row["train_actual_C_change"]))
    ax.set_xlabel("Train merit change")
    ax.set_ylabel("Train cost change")
    ax.set_title("Same-start one-step frontier")
    fig.tight_layout()
    fig.savefig(plots_dir / "stage410_same_start_frontier.png", dpi=180)
    plt.close(fig)


def write_report(summary_df: pd.DataFrame, output_root: pathlib.Path) -> None:
    report_lines = [
        "# Stage 4.10 Same-Start Smoke Report",
        "",
        "- Metric used for comparison: `performance-only merit = C_b` (no EMA scaling, no norm term).",
        "- Proposed comparison: `proposed_noG_perfLyap_extrap_smoke` vs `proposed_qp_perfLyap_extrap_autoactual_smoke`.",
        "- Proposed QP mode: EGM-like extrapolation with offline auto actual sign selection between plus/minus variants.",
        "",
    ]

    best_by_config = []
    for (config_label, cost_mode), group in summary_df.groupby(["config_label", "cost_mode"]):
        best = group.sort_values(["train_actual_merit_change_mean", "train_actual_C_change_mean"]).head(1)
        best_by_config.append(best)
        report_lines.append(f"## {config_label}")
        report_lines.append("")
        report_lines.append(group[["method", "train_actual_merit_change_mean", "train_actual_C_change_mean", "val_actual_merit_change_mean", "train_approx_kl_max", "train_clip_fraction_max"]].sort_values(["train_actual_merit_change_mean", "train_actual_C_change_mean"]).to_csv(index=False))
        report_lines.append("")

    best_df = pd.concat(best_by_config, ignore_index=True) if best_by_config else pd.DataFrame()
    if not best_df.empty:
        report_lines.append("## Best Method Per Config")
        report_lines.append("")
        report_lines.append(best_df[["config_label", "method", "train_actual_merit_change_mean", "train_actual_C_change_mean"]].to_csv(index=False))
        report_lines.append("")

    proposed = summary_df[summary_df["method"] == "proposed_qp_perfLyap_extrap_autoactual_smoke"]
    nog = summary_df[summary_df["method"] == "proposed_noG_perfLyap_extrap_smoke"]
    sgd = summary_df[summary_df["method"] == "sgd"]
    egm = summary_df[summary_df["method"] == "egm"]
    ppm = summary_df[summary_df["method"] == "ppm"]

    def count_beats(left: pd.DataFrame, right: pd.DataFrame, col: str) -> int:
        merged = left.merge(right, on=["config_id", "config_label", "scope", "cost_mode"], suffixes=("_l", "_r"))
        return int((merged[f"{col}_l"] < merged[f"{col}_r"]).sum())

    report_lines.extend(
        [
            "## Aggregate Answers",
            "",
            f"- Proposed extrap beats noG on train merit in `{count_beats(proposed, nog, 'train_actual_merit_change_mean')}` configs.",
            f"- Proposed extrap beats SGD on train merit in `{count_beats(proposed, sgd, 'train_actual_merit_change_mean')}` configs.",
            f"- Proposed extrap beats EGM on train merit in `{count_beats(proposed, egm, 'train_actual_merit_change_mean')}` configs.",
            f"- Proposed extrap beats PPM on train merit in `{count_beats(proposed, ppm, 'train_actual_merit_change_mean')}` configs.",
            "",
            "Interpretation:",
            "- If `noG` consistently beats proposed extrap here, then the current second direction is not helping even on the exact local metric it was designed for.",
            "- If EGM/PPM beat SGD here, that supports the idea that extra structure can help, but not that the current proposed `G` is the right way to use it.",
        ]
    )
    (output_root / "stage410_same_start_smoke_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    baseline_settings = load_baseline_settings(pathlib.Path(args.baseline_summary))
    configs = build_configs()
    model = build_model(args)

    all_rows: List[Dict[str, object]] = []
    for role in ("protagonist", "adversary"):
        algo = role_algo(model, role)
        probes = collect_probe_batches(model, role, max(args.num_probes_per_role, 4))
        for config in configs:
            for train_probe, val_probe in pair_probes(probes):
                all_rows.extend(
                    proposed_rows_for_probe(
                        algo=algo,
                        role=role,
                        train_probe=train_probe,
                        val_probe=val_probe,
                        config=config,
                        args=args,
                    )
                )
                all_rows.extend(
                    baseline_rows_for_probe(
                        algo=algo,
                        role=role,
                        train_probe=train_probe,
                        val_probe=val_probe,
                        config=config,
                        args=args,
                        baseline_settings=baseline_settings,
                    )
                )

    detail_df = pd.DataFrame(all_rows)
    detail_df.to_csv(output_root / "stage410_same_start_smoke_detail.csv", index=False)
    summary_df = summarize(detail_df)
    summary_df.to_csv(output_root / "stage410_same_start_smoke_summary.csv", index=False)
    plot_results(detail_df, summary_df, output_root)
    write_report(summary_df, output_root)


if __name__ == "__main__":
    main()
